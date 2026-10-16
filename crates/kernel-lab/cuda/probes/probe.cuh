#pragma once
#include <stdint.h>
#include <cuda.h>
#include "../../../kernels-sm120/cuda/common/ptx_control.cuh"
#include "../../../kernels-sm120/cuda/common/ptx_tma.cuh"
__device__ __forceinline__ void probe_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
__device__ __forceinline__ void probe_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

namespace chop::probe {
__device__ void sm120(unsigned int* output) {
  probe_launch_dependents();
  probe_wait();
  if (threadIdx.x == 0) output[0] = 120;
}
__device__ void gate(const unsigned int* flag) {
  probe_launch_dependents();
  while (*reinterpret_cast<const volatile unsigned int*>(flag) == 0) {}
  probe_wait();
}
// Rank 0's gate: once the host sets its flag, it sets every other rank's gate
// flag with one peer store each, so all ranks start within about one P2P
// latency instead of one host copy per rank apart.
template <typename Peers>
__device__ void gate_release(const unsigned int* flag, Peers peers, uint32_t count) {
  probe_launch_dependents();
  if (threadIdx.x == 0) {
    while (*reinterpret_cast<const volatile unsigned int*>(flag) == 0) {}
    unsigned int* slots[8] = {peers.slot_0, peers.slot_1, peers.slot_2, peers.slot_3,
                              peers.slot_4, peers.slot_5, peers.slot_6, peers.slot_7};
    for (uint32_t i = 0; i < count && i < 8; ++i)
      *reinterpret_cast<volatile unsigned int*>(slots[i]) = 1;
  }
  probe_wait();
}
__device__ void flush(unsigned int* buffer, uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    buffer[i] = static_cast<unsigned int>(i);
}

__device__ void bandwidth_read(const unsigned int* input, unsigned int* checksum,
                               uint64_t words, uint32_t repeats) {
  probe_launch_dependents();
  probe_wait();
  unsigned int sum = 0;
  for (uint32_t repeat = 0; repeat < repeats; ++repeat)
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
      sum += input[i];
  __shared__ unsigned int partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(checksum, partial[0]);
}
__device__ void bandwidth_write(unsigned int* output, uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    output[i] = static_cast<unsigned int>(i);
}
__device__ void bandwidth_copy(const unsigned int* input, unsigned int* output,
                               uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    output[i] = input[i];
}

__device__ __forceinline__ uint4 streaming_load(const uint4* address) {
  uint4 value;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.b32 "
               "{%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void streaming_store(uint4* address, uint4 value) {
  asm volatile("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};"
               :: "l"(address), "r"(value.x), "r"(value.y),
                  "r"(value.z), "r"(value.w) : "memory");
}

__device__ __forceinline__ uint32_t vector_sum(uint4 value) {
  return value.x + value.y + value.z + value.w;
}

__device__ void bandwidth_read_vector(const uint4* input, unsigned int* checksum,
                                      uint64_t words, uint32_t repeats) {
  probe_launch_dependents();
  probe_wait();
  const uint64_t vectors = words / 4;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint32_t a = 0, b = 0, c = 0, d = 0;
  for (uint32_t repeat = 0; repeat < repeats; ++repeat) {
    for (uint64_t base = thread; base < vectors; base += 4 * stride) {
      if (base < vectors) a += vector_sum(streaming_load(input + base));
      if (base + stride < vectors) b += vector_sum(streaming_load(input + base + stride));
      if (base + 2 * stride < vectors) c += vector_sum(streaming_load(input + base + 2 * stride));
      if (base + 3 * stride < vectors) d += vector_sum(streaming_load(input + base + 3 * stride));
    }
  }
  __shared__ uint32_t partial[256];
  partial[threadIdx.x] = a + b + c + d;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(checksum, partial[0]);
}

__device__ void bandwidth_write_vector(uint4* output, uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  const uint64_t vectors = words / 4;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectors; index += stride) {
    const uint32_t first = static_cast<uint32_t>(4 * index);
    streaming_store(output + index, make_uint4(first, first + 1, first + 2, first + 3));
  }
}

__device__ void bandwidth_copy_vector(const uint4* input, uint4* output,
                                      uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  const uint64_t vectors = words / 4;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < vectors; index += stride)
    streaming_store(output + index, streaming_load(input + index));
}

// Pointer-addressed CTA TMA, double buffered to overlap the next 16 KiB load
// with consumption of the current tile. Every input word contributes to the
// checksum, so the load cannot disappear during compilation.
__device__ void bandwidth_read_tma(const unsigned int* input, unsigned int* checksum,
                                   uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  constexpr uint32_t tile_words = 4096;
  constexpr uint32_t tile_bytes = tile_words * sizeof(uint32_t);
  __shared__ alignas(128) uint32_t tiles[2][tile_words];
  __shared__ alignas(16) uint64_t barriers[2];
  __shared__ uint32_t partial[256];
  const uint64_t total_tiles = words / tile_words;
  const uint64_t count = total_tiles > blockIdx.x
      ? (total_tiles - 1 - blockIdx.x) / gridDim.x + 1 : 0;
  if (threadIdx.x == 0) {
    chop::ptx::mbarrier_init(&barriers[0], 1);
    chop::ptx::mbarrier_init(&barriers[1], 1);
    chop::ptx::fence_mbarrier_init();
    if (count != 0) {
      chop::ptx::mbarrier_arrive_expect_tx(&barriers[0], tile_bytes);
      chop::ptx::cp_async_bulk(tiles[0], input + blockIdx.x * tile_words,
                               tile_bytes, &barriers[0]);
    }
  }
  __syncthreads();
  uint32_t a = 0, b = 0, c = 0, d = 0;
  for (uint64_t iteration = 0; iteration < count; ++iteration) {
    const uint32_t slot = iteration & 1;
    if (threadIdx.x == 0) {
      chop::ptx::mbarrier_wait(&barriers[slot], (iteration / 2) & 1);
      if (iteration + 1 < count) {
        const uint32_t next = slot ^ 1;
        const uint64_t tile = blockIdx.x + (iteration + 1) * gridDim.x;
        chop::ptx::mbarrier_arrive_expect_tx(&barriers[next], tile_bytes);
        chop::ptx::cp_async_bulk(tiles[next], input + tile * tile_words,
                                 tile_bytes, &barriers[next]);
      }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < tile_words / 256; i += 4) {
      a += tiles[slot][threadIdx.x + (i + 0) * 256];
      b += tiles[slot][threadIdx.x + (i + 1) * 256];
      c += tiles[slot][threadIdx.x + (i + 2) * 256];
      d += tiles[slot][threadIdx.x + (i + 3) * 256];
    }
    __syncthreads();
  }
  partial[threadIdx.x] = a + b + c + d;
  __syncthreads();
  for (int offset = 128; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(checksum, partial[0]);
    chop::ptx::mbarrier_inval(&barriers[0]);
    chop::ptx::mbarrier_inval(&barriers[1]);
  }
}

__device__ void cluster_multicast(const void* map, unsigned int* output) {
  probe_launch_dependents();
  probe_wait();
  __shared__ alignas(128) float tile[32];
  __shared__ alignas(16) uint64_t bar;
  __shared__ float dsm_float;
  __shared__ uint32_t dsm_sum;
  if (threadIdx.x != 0) return;
  const uint32_t rank = chop::ptx::cluster_ctarank();
  const uint32_t size = chop::ptx::cluster_nctarank();
  dsm_float = 0.0f;
  dsm_sum = 0;
  chop::ptx::mbarrier_init(&bar, 1);
  chop::ptx::fence_mbarrier_init();
  chop::ptx::cluster_arrive_release();
  chop::ptx::cluster_wait_acquire();
  chop::ptx::mbarrier_arrive_expect_tx(&bar, 136);
  chop::ptx::cluster_arrive_release();
  chop::ptx::cluster_wait_acquire();
  if (rank == 0) {
    int32_t coord[2] = {0, 0};
    chop::ptx::tma_multicast_2d(tile, map, coord, &bar,
                                static_cast<uint16_t>((1u << size) - 1));
  }
  const uint32_t next = (rank + 1) % size;
  const uint32_t remote_bar = chop::ptx::cluster_map_shared(&bar, next);
  chop::ptx::cluster_st_async_f32(chop::ptx::cluster_map_shared(&dsm_float, next),
                                  static_cast<float>(rank + 1), remote_bar);
  chop::ptx::cluster_red_async_add_u32(chop::ptx::cluster_map_shared(&dsm_sum, next),
                                        rank + 1, remote_bar);
  const uint64_t deadline = clock64() + 2000000000ull;
  bool ready = false;
  while (clock64() < deadline && !(ready = chop::ptx::mbarrier_try_wait(&bar, 0))) {}
  output[rank * 4] = ready;
  output[rank * 4 + 1] = __float_as_uint(tile[0]);
  output[rank * 4 + 2] = __float_as_uint(dsm_float);
  output[rank * 4 + 3] = dsm_sum;
  chop::ptx::mbarrier_inval(&bar);
}

__device__ void cluster_dsm(unsigned int* output) {
  probe_launch_dependents();
  probe_wait();
  __shared__ alignas(16) uint64_t bar;
  __shared__ float value;
  __shared__ uint32_t sum;
  if (threadIdx.x != 0) return;
  const uint32_t rank = chop::ptx::cluster_ctarank();
  const uint32_t size = chop::ptx::cluster_nctarank();
  value = 0.0f;
  sum = 0;
  chop::ptx::mbarrier_init(&bar, 1);
  chop::ptx::fence_mbarrier_init();
  chop::ptx::mbarrier_arrive_expect_tx(&bar, 8);
  chop::ptx::cluster_arrive_release();
  chop::ptx::cluster_wait_acquire();
  const uint32_t next = (rank + 1) % size;
  const uint32_t remote_bar = chop::ptx::cluster_map_shared(&bar, next);
  chop::ptx::cluster_st_async_f32(chop::ptx::cluster_map_shared(&value, next),
                                  static_cast<float>(rank + 1), remote_bar);
  chop::ptx::cluster_red_async_add_u32(chop::ptx::cluster_map_shared(&sum, next),
                                        rank + 1, remote_bar);
  const uint64_t deadline = clock64() + 2000000000ull;
  bool ready = false;
  while (clock64() < deadline && !(ready = chop::ptx::mbarrier_try_wait(&bar, 0))) {}
  output[rank * 3] = ready;
  output[rank * 3 + 1] = __float_as_uint(value);
  output[rank * 3 + 2] = sum;
  chop::ptx::mbarrier_inval(&bar);
}

__device__ void cluster_control(unsigned int* output) {
  probe_launch_dependents();
  probe_wait();
  __shared__ alignas(16) uint32_t response[4];
  __shared__ alignas(16) uint64_t bar;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  chop::ptx::mbarrier_init(&bar, 1);
  chop::ptx::fence_mbarrier_init();
  chop::ptx::mbarrier_arrive_expect_tx(&bar, 16);
  chop::ptx::cluster_try_cancel(response, &bar);
  const uint64_t deadline = clock64() + 2000000000ull;
  bool ready = false;
  while (clock64() < deadline && !(ready = chop::ptx::mbarrier_try_wait(&bar, 0))) {}
  uint32_t first = 0;
  const bool canceled = ready && chop::ptx::cluster_query_cancel(response, first);
  output[0] = ready;
  output[1] = canceled;
  output[2] = first;
  chop::ptx::mbarrier_inval(&bar);
}

__device__ void gather4(const void* map, unsigned int* output) {
  probe_launch_dependents();
  probe_wait();
  __shared__ alignas(128) float tile[128];
  __shared__ alignas(16) uint64_t bar;
  if (threadIdx.x != 0) return;
  chop::ptx::mbarrier_init(&bar, 1);
  chop::ptx::fence_mbarrier_init();
  chop::ptx::mbarrier_arrive_expect_tx(&bar, 512);
  int32_t coord[5] = {0, 0, 1, 2, 3};
  chop::ptx::tma_gather4_2d(tile, map, coord, &bar);
  const uint64_t deadline = clock64() + 2000000000ull;
  bool ready = false;
  while (clock64() < deadline && !(ready = chop::ptx::mbarrier_try_wait(&bar, 0))) {}
  output[0] = ready;
  for (int row = 0; row < 4; ++row)
    output[row + 1] = __float_as_uint(tile[row * 32]);
  chop::ptx::mbarrier_inval(&bar);
}


__device__ void shared_limit(unsigned int* output, uint32_t bytes) {
  probe_launch_dependents();
  probe_wait();
  extern __shared__ unsigned char shared[];
  if (threadIdx.x == 0 && bytes != 0) {
    shared[bytes - 1] = 0x5a;
    output[0] = shared[bytes - 1];
  }
}

__device__ __forceinline__ uint64_t globaltimer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(value));
  return value;
}
__device__ void pdl_first(uint64_t* stamps, uint64_t cycles) {
  probe_launch_dependents();
  if (threadIdx.x == 0) {
    const uint64_t start = clock64();
    stamps[0] = globaltimer();
    while (clock64() - start < cycles) {}
    stamps[1] = globaltimer();
  }
}
__device__ void pdl_second(uint64_t* stamps) {
  probe_launch_dependents();
  if (threadIdx.x == 0) stamps[2] = globaltimer();
  probe_wait();
  if (threadIdx.x == 0) stamps[3] = globaltimer();
}


template <int Format>
__device__ __forceinline__ void mma_step(chop::ptx::D4& d,
                                          const chop::ptx::A4& a,
                                          const chop::ptx::B2& b,
                                          uint32_t scale_a, uint32_t scale_b) {
  if constexpr (Format == 0) chop::ptx::mma_bf16(d, a, b);
  else if constexpr (Format == 1)
    chop::ptx::mma_f8f6f4<chop::ptx::Element::e4m3, chop::ptx::Element::e4m3>(d, a, b);
  else if constexpr (Format == 2)
    chop::ptx::mma_f8f6f4<chop::ptx::Element::e2m1, chop::ptx::Element::e2m1>(d, a, b);
  else if constexpr (Format == 3)
    chop::ptx::mma_mxf8f6f4<chop::ptx::Element::e4m3, chop::ptx::Element::e4m3>
        (d, a, b, scale_a, scale_b);
  else if constexpr (Format == 4)
    chop::ptx::mma_mxf8f6f4<chop::ptx::Element::e2m1, chop::ptx::Element::e2m1>
        (d, a, b, scale_a, scale_b);
  else if constexpr (Format == 5)
    chop::ptx::mma_mxf4nvf4<2>(d, a, b, scale_a, scale_b);
  else if constexpr (Format == 6)
    chop::ptx::mma_mxf4nvf4<4>(d, a, b, scale_a, scale_b);
  else if constexpr (Format == 7)
    chop::ptx::mma_mxf4(d, a, b, scale_a, scale_b);
}

template <int Format>
__device__ void mma_throughput(float* output, const uint32_t* input,
                               uint32_t iterations) {
  probe_launch_dependents();
  probe_wait();
  const chop::ptx::A4 a{{input[0], input[1], input[2], input[3]}};
  const chop::ptx::B2 b{{input[4], input[5]}};
  const uint32_t scale_a = input[6], scale_b = input[7];
  chop::ptx::D4 d0{{0, 0, 0, 0}}, d1{{0, 0, 0, 0}};
  chop::ptx::D4 d2{{0, 0, 0, 0}}, d3{{0, 0, 0, 0}};
  chop::ptx::D4 d4{{0, 0, 0, 0}}, d5{{0, 0, 0, 0}};
  chop::ptx::D4 d6{{0, 0, 0, 0}}, d7{{0, 0, 0, 0}};
  for (uint32_t i = 0; i < iterations; ++i) {
    mma_step<Format>(d0, a, b, scale_a, scale_b);
    mma_step<Format>(d1, a, b, scale_a, scale_b);
    mma_step<Format>(d2, a, b, scale_a, scale_b);
    mma_step<Format>(d3, a, b, scale_a, scale_b);
    mma_step<Format>(d4, a, b, scale_a, scale_b);
    mma_step<Format>(d5, a, b, scale_a, scale_b);
    mma_step<Format>(d6, a, b, scale_a, scale_b);
    mma_step<Format>(d7, a, b, scale_a, scale_b);
  }
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  output[index] = d0.v[0] + d1.v[0] + d2.v[0] + d3.v[0]
                + d4.v[0] + d5.v[0] + d6.v[0] + d7.v[0];
}


template <typename Peers>
__device__ void p2p_pull(unsigned int* output, Peers peers, uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    output[i] = peers.slot_0[i];
}

template <typename Peers>
__device__ void p2p_push(const unsigned int* input, Peers peers, uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < words; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    peers.slot_0[i] = input[i];
}

// blockIdx.y selects the destination, so every destination is written at the
// same time. Comparing one destination with several isolates fan-out cost.
template <typename Peers>
__device__ void p2p_push_fanout(const unsigned int* input, Peers peers, uint64_t words) {
  probe_launch_dependents();
  probe_wait();
  unsigned int* destination = blockIdx.y == 0 ? peers.slot_0 : blockIdx.y == 1 ? peers.slot_1
      : blockIdx.y == 2 ? peers.slot_2 : blockIdx.y == 3 ? peers.slot_3
      : blockIdx.y == 4 ? peers.slot_4 : blockIdx.y == 5 ? peers.slot_5
      : blockIdx.y == 6 ? peers.slot_6 : peers.slot_7;
  const uint64_t vectors = words / 4;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < vectors; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    reinterpret_cast<uint4*>(destination)[i] = reinterpret_cast<const uint4*>(input)[i];
}

// Grid row 0 pushes input into peers.slot_0; row 1 copies peers.slot_1 (a
// buffer on another GPU) into `local`. Launching one or both rows compares a
// push, a pull, and a push to one peer concurrent with a pull from another.
template <typename Peers>
__device__ void p2p_push_pull(const unsigned int* input, Peers peers, unsigned int* local,
                              uint64_t words, uint32_t first_row) {
  probe_launch_dependents();
  probe_wait();
  const uint32_t row = blockIdx.y + first_row;
  const uint4* from = row == 0 ? reinterpret_cast<const uint4*>(input)
                               : reinterpret_cast<const uint4*>(peers.slot_1);
  uint4* to = row == 0 ? reinterpret_cast<uint4*>(peers.slot_0) : reinterpret_cast<uint4*>(local);
  const uint64_t vectors = words / 4;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < vectors; i += static_cast<uint64_t>(gridDim.x) * blockDim.x)
    to[i] = from[i];
}

__device__ __forceinline__ uint32_t p2p_acquire(const unsigned int* address) {
  uint32_t value;
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}
__device__ __forceinline__ void p2p_release(unsigned int* address, uint32_t value) {
  asm volatile("st.global.release.sys.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}
template <typename Peers>
__device__ void p2p_ping(unsigned int* local_flag, Peers peers,
                         uint64_t* stamps, uint32_t iterations) {
  probe_launch_dependents();
  probe_wait();
  if (threadIdx.x != 0) return;
  const uint64_t start = globaltimer();
  for (uint32_t epoch = 1; epoch <= iterations; ++epoch) {
    p2p_release(peers.slot_0, epoch);
    const uint64_t deadline = clock64() + 2000000000ull;
    while (p2p_acquire(local_flag) != epoch) {
      if (clock64() > deadline) { stamps[1] = 0; return; }
    }
  }
  stamps[0] = start;
  stamps[1] = globaltimer();
}
template <typename Peers>
__device__ void p2p_pong(unsigned int* local_flag, Peers peers,
                         uint32_t iterations) {
  probe_launch_dependents();
  probe_wait();
  if (threadIdx.x != 0) return;
  for (uint32_t epoch = 1; epoch <= iterations; ++epoch) {
    const uint64_t deadline = clock64() + 2000000000ull;
    while (p2p_acquire(local_flag) != epoch) {
      if (clock64() > deadline) return;
    }
    p2p_release(peers.slot_0, epoch);
  }
}

} // namespace chop::probe
