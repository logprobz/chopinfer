#pragma once

#include <stdint.h>
#include "../common/ptx_control.cuh"

#ifndef CHOP_COMM_MAX_RANKS
#error "CHOP_COMM_MAX_RANKS is required"
#endif

namespace chop::comm {

// Most ranks of a collective and slots of every pointer array argument
// (kernel_core::contract::MAX_COMM_RANKS).
constexpr uint32_t kMaxRanks = CHOP_COMM_MAX_RANKS;

__device__ __forceinline__ uint32_t acquire_system(const unsigned int* address) {
  uint32_t value;
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void release_system(unsigned int* address, uint32_t value) {
  asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

struct alignas(16) Vector128 { uint32_t x, y, z, w; };

// Slot `rank` of a by-value pointer array of 2, 4 or kMaxRanks slots. An
// entry compiled for a fixed rank count declares that many slots; the rank
// generic entries declare kMaxRanks.
template <typename Pointers>
__device__ __forceinline__ auto peer(Pointers pointers, uint32_t rank) {
  static_assert(kMaxRanks == 8, "peer names eight slots");
  constexpr size_t slots = sizeof(Pointers) / sizeof(pointers.slot_0);
  static_assert(slots == 2 || slots == 4 || slots == kMaxRanks, "2, 4 or 8 slots");
  if constexpr (slots == 2) {
    return rank == 0 ? pointers.slot_0 : pointers.slot_1;
  } else if constexpr (slots == 4) {
    switch (rank) {
      case 0: return pointers.slot_0;
      case 1: return pointers.slot_1;
      case 2: return pointers.slot_2;
      default: return pointers.slot_3;
    }
  } else {
    switch (rank) {
      case 0: return pointers.slot_0;
      case 1: return pointers.slot_1;
      case 2: return pointers.slot_2;
      case 3: return pointers.slot_3;
      case 4: return pointers.slot_4;
      case 5: return pointers.slot_5;
      case 6: return pointers.slot_6;
      default: return pointers.slot_7;
    }
  }
}

template <typename T>
__device__ __forceinline__ float decode(T value) { return value; }
template <>
__device__ __forceinline__ float decode<uint16_t>(uint16_t value) {
  return __uint_as_float(static_cast<uint32_t>(value) << 16);
}
template <typename T>
__device__ __forceinline__ T encode(float value) { return value; }
template <>
__device__ __forceinline__ uint16_t encode<uint16_t>(float value) {
  const uint32_t bits = __float_as_uint(value);
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// One row of E::kWidth columns by one CTA (the local decode control).
template <typename E>
__device__ __forceinline__ void norm_row_bf16(const uint16_t* input, const E& epilogue) {
  __shared__ uint16_t values[E::kWidth];
  __shared__ float partial[32];
  float square_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < E::kWidth; col += blockDim.x)
    values[col] = epilogue.carry_element(decode(input[col]), col, square_sum);
  for (int offset = 16; offset > 0; offset >>= 1)
    square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = square_sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float warp_sum = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    if (threadIdx.x == 0) partial[0] = warp_sum;
  }
  __syncthreads();
  const float inv = epilogue.scale(partial[0]);
  for (uint32_t col = threadIdx.x; col < E::kWidth; col += blockDim.x)
    epilogue.finish_element(col, col, values[col], inv);
}

__device__ __forceinline__ uint32_t acquire_gpu(const unsigned int* address) {
  uint32_t value;
  asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void release_gpu(unsigned int* address, uint32_t value) {
  asm volatile("st.global.release.gpu.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

// Test-only lag, compiled into kernel-lab's comm_sm120_skew cubin: in every
// call one rank, chosen by the call's epoch, waits before a protocol step so
// its peers run as far ahead as the protocol allows. Serving cubins leave
// CHOP_COMM_TEST_SKEW_NS undefined and compile this away.
__device__ __forceinline__ void test_skew(uint32_t rank, uint32_t ranks, uint32_t epoch) {
#ifdef CHOP_COMM_TEST_SKEW_NS
  if (ranks == 0 || rank != epoch % ranks) return;
  uint64_t start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < CHOP_COMM_TEST_SKEW_NS);
#endif
}

#ifdef CHOP_COMM_TEST_SKEW_NS
// A slow consumer between two collectives on one rank's stream.
__device__ void test_spin() {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  test_skew(0, 1, 0);
}
#endif

// Isolated decode controls use the same arithmetic and flag operations as
// the fused path so graph timing separates local work from peer traffic.
template <typename E>
__device__ void local_norm(const uint16_t* input, const E& epilogue) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  norm_row_bf16(input, epilogue);
}

__device__ void local_flags(unsigned int* flags, unsigned int* epoch) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (threadIdx.x != 0) return;
  const uint32_t next = epoch[0] + 1;
  while (acquire_system(flags + 1) != next - 1) {}
  release_system(flags, next);
  while (acquire_system(flags) != next) {}
  release_system(flags + 1, next);
  epoch[0] = next;
}

// Each CTA owns one tile and one epoch cell. The peer inbox has one slot per
// source rank. Acknowledgments prevent graph replay from overwriting a slot
// before its consumer has finished reading it.
template <typename T, typename Inboxes, typename Flags>
__device__ void allreduce_oneshot(const T* input, T* output,
                                  Inboxes inboxes, Flags flags,
                                  unsigned int* epoch, uint64_t elements,
                                  uint32_t rank, uint32_t ranks) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint64_t tile = blockIdx.x;
  const uint64_t tiles = (elements + 4095) / 4096;
  if (tile >= tiles || rank >= ranks || ranks == 0 || ranks > kMaxRanks) return;
  const uint64_t first = tile * 4096;
  const uint64_t end = min(first + 4096, elements);
  const uint32_t next = epoch[tile] + 1;
  const uint64_t flag_index = static_cast<uint64_t>(rank) * tiles + tile;
  if (threadIdx.x == 0) {
    for (uint32_t dest = 0; dest < ranks; ++dest) {
      while (acquire_system(peer(flags, dest) + ranks * tiles + flag_index) != next - 1) {}
    }
  }
  __syncthreads();
  for (uint32_t dest = 0; dest < ranks; ++dest) {
    T* slot = peer(inboxes, dest) + static_cast<uint64_t>(rank) * elements;
    for (uint64_t i = first + threadIdx.x; i < end; i += blockDim.x)
      slot[i] = input[i];
    __syncthreads();
    if (threadIdx.x == 0)
      release_system(peer(flags, dest) + flag_index, next);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (uint32_t source = 0; source < ranks; ++source) {
      while (acquire_system(peer(flags, rank) + static_cast<uint64_t>(source) * tiles + tile) != next) {}
    }
  }
  __syncthreads();
  for (uint64_t i = first + threadIdx.x; i < end; i += blockDim.x) {
    float sum = 0.0f;
    for (uint32_t source = 0; source < ranks; ++source)
      sum += decode(peer(inboxes, rank)[static_cast<uint64_t>(source) * elements + i]);
    output[i] = encode<T>(sum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t source = 0; source < ranks; ++source)
      release_system(peer(flags, rank) + ranks * tiles + static_cast<uint64_t>(source) * tiles + tile, next);
    epoch[tile] = next;
  }
}

// One CTA owns one row of E::kWidth BF16 columns. Reduction and the
// epilogue (comm/epilogue/) stay in one launch; the rank-generic entry
// applies the epilogue's scalar forms.
template <typename E, typename Inboxes, typename Flags>
__device__ void allreduce_oneshot_norm(const uint16_t* input, Inboxes inboxes, Flags flags,
                                       unsigned int* epoch, uint64_t elements,
                                       uint32_t rank, uint32_t ranks, const E& epilogue) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint64_t tile = blockIdx.x;
  const uint64_t tiles = elements / E::kWidth;
  if (tile >= tiles || rank >= ranks || ranks == 0 || ranks > kMaxRanks) return;
  const uint32_t next = epoch[tile] + 1;
  const uint64_t flag_index = static_cast<uint64_t>(rank) * tiles + tile;
  if (threadIdx.x == 0) {
    for (uint32_t dest = 0; dest < ranks; ++dest)
      while (acquire_system(peer(flags, dest) + ranks * tiles + flag_index) != next - 1) {}
  }
  __syncthreads();
  const uint64_t first = tile * E::kWidth;
  for (uint32_t dest = 0; dest < ranks; ++dest) {
    uint16_t* slot = peer(inboxes, dest) + static_cast<uint64_t>(rank) * elements;
    for (uint64_t i = first + threadIdx.x; i < first + E::kWidth; i += blockDim.x)
      slot[i] = input[i];
    __syncthreads();
    if (threadIdx.x == 0) release_system(peer(flags, dest) + flag_index, next);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    for (uint32_t source = 0; source < ranks; ++source)
      while (acquire_system(peer(flags, rank) + static_cast<uint64_t>(source) * tiles + tile) != next) {}
  }
  __syncthreads();
  __shared__ uint16_t values[E::kWidth];
  __shared__ float partial[256];
  float square_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < E::kWidth; col += blockDim.x) {
    const uint64_t i = first + col;
    float value = 0.0f;
    for (uint32_t source = 0; source < ranks; ++source)
      value += decode(peer(inboxes, rank)[static_cast<uint64_t>(source) * elements + i]);
    values[col] = epilogue.carry_element(value, i, square_sum);
  }
  partial[threadIdx.x] = square_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t source = 0; source < ranks; ++source)
      release_system(peer(flags, rank) + ranks * tiles + static_cast<uint64_t>(source) * tiles + tile, next);
    epoch[tile] = next;
  }
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  const float inv = epilogue.scale(partial[0]);
  for (uint32_t col = threadIdx.x; col < E::kWidth; col += blockDim.x)
    epilogue.finish_element(first + col, col, values[col], inv);
}

// Each rank publishes its shard directly into every selected output. Flags
// make the all-gather graph replay safe without a host epoch update.
template <typename T, typename Outputs, typename Flags>
__device__ void allgather(const T* input, Outputs outputs, Flags flags,
                          unsigned int* epoch, uint64_t elements_per_rank,
                          uint32_t rank, uint32_t ranks, uint32_t target) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint64_t tile = blockIdx.x;
  const uint64_t tiles = (elements_per_rank + 4095) / 4096;
  if (tile >= tiles || rank >= ranks || ranks == 0 || ranks > kMaxRanks || target > ranks) return;
  const uint64_t first = tile * 4096;
  const uint64_t end = min(first + 4096, elements_per_rank);
  const uint32_t next = epoch[tile] + 1;
  const uint64_t flag = static_cast<uint64_t>(rank) * tiles + tile;
  if (threadIdx.x == 0) {
    // Acknowledge the previous call on every call: at the first one this
    // stores the zero the flags start with, and after the epoch wraps to
    // zero the senders need 0xffffffff.
    for (uint32_t source = 0; source < ranks; ++source)
      release_system(peer(flags, rank) + ranks * tiles + static_cast<uint64_t>(source) * tiles + tile, next - 1);
    for (uint32_t dest = 0; dest < ranks; ++dest)
      if (target == ranks || dest == target)
        while (acquire_system(peer(flags, dest) + ranks * tiles + flag) != next - 1) {}
  }
  __syncthreads();
  for (uint32_t dest = 0; dest < ranks; ++dest) {
    if (target != ranks && dest != target) continue;
    T* remote = peer(outputs, dest) + static_cast<uint64_t>(rank) * elements_per_rank;
    for (uint64_t i = first + threadIdx.x; i < end; i += blockDim.x)
      remote[i] = input[i];
    __syncthreads();
    if (threadIdx.x == 0) release_system(peer(flags, dest) + flag, next);
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (target == ranks || rank == target) {
      for (uint32_t source = 0; source < ranks; ++source)
        while (acquire_system(peer(flags, rank) + static_cast<uint64_t>(source) * tiles + tile) != next) {}
    }
    epoch[tile] = next;
  }
}

} // namespace chop::comm

#include "push.cuh"
