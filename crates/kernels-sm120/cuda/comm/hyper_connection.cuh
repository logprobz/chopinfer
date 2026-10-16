#pragma once
#include "../common/ptx_control.cuh"
#include "comm.cuh"
#include "../row/row.cuh"

// Hyper-connection all-reduces: the ranks' partials of each token, of the
// input type Input (comm/input/: BF16 or F32 elements), summed in rank
// order in F32 and handed, unrounded, to the epilogue functor E
// (comm/epilogue/, for example hc_inject_norm.cuh), which finishes each of
// the token's E::kStreams residual streams with one CTA of E::kThreads
// threads. The row width is E::kWidth. Inbox pointers have Input's element
// type; regions and strides count its elements, and the F32 sums and
// all-gathers stay F32 whatever the input.
namespace chop::comm::qwen38 {

// The geometry every hyper-connection entry checks for its epilogue E and
// input Input, before any division by it: nonempty rows and streams, one
// 256-thread CTA per (row, stream) group whose threads hold whole columns,
// and rows of whole 16-byte input vectors. The sending loops step over
// clusters of 256-thread CTAs and the epilogue over 256-column strides, so
// any other thread count would skip vectors or columns. The decode
// entries' LL layout adds its own checks (qwen38_decode::Rows).
template<class E, class Input>
struct Geometry {
  static_assert(E::kWidth > 0 && E::kStreams > 0, "rows need columns and streams");
  static_assert(E::kThreads == 256, "one 256-thread CTA per (row, stream) group");
  static_assert(E::kPer * E::kThreads == E::kWidth, "whole columns per epilogue thread");
  static_assert(Input::kPerVector * sizeof(typename Input::Element) == 16 &&
                E::kWidth % Input::kPerVector == 0, "rows of whole 16-byte input vectors");
  // The sum writers store each input vector's F32 sums as whole 16-byte
  // vectors (kPerVector / 4 of them): BF16 and F32 inputs only.
  static_assert((sizeof(typename Input::Element) == 2 || sizeof(typename Input::Element) == 4) &&
                Input::kPerVector % 4 == 0, "inputs of BF16 or F32 elements");
  static constexpr uint32_t kVectors = E::kWidth / Input::kPerVector;  // per row
};

// Every entry takes the rank count at run time: 2, 4 or 8 ranks, the
// contract's (comm::qwen38::RANKS), with one pointer slot per rank
// (comm.cuh's `peer`). Every entry traps on any other count before it
// touches a peer. Entries that sum over sources dispatch to a body compiled
// for the rank count, whose loops are constant: with a run-time count the
// reductions issued their peer loads one after another. Sums run in rank
// order, so every rank count gives the same bits on every rank.
constexpr __host__ __device__ bool valid_ranks(uint32_t rank, uint32_t ranks) {
  return (ranks == 2 || ranks == 4 || ranks == 8) && ranks <= kMaxRanks && rank < ranks;
}
// The contract's rank counts (comm::qwen38::RANKS) and no others.
static_assert(!valid_ranks(0, 0) && !valid_ranks(0, 1) && valid_ranks(1, 2) &&
              !valid_ranks(0, 3) && valid_ranks(3, 4) && !valid_ranks(0, 5) &&
              !valid_ranks(0, 6) && !valid_ranks(0, 7) && valid_ranks(7, 8) &&
              !valid_ranks(0, 9) && !valid_ranks(0, 16) && !valid_ranks(2, 2) &&
              !valid_ranks(8, 8), "hyper-connection entries take 2, 4 or 8 ranks");

// The one-shot entry (decode and prefill up to 32 rows): a single cluster
// serializes destinations across the entire GPU. Inboxes have two epoch
// banks of `ranks` slots and a fixed prepared-capacity stride; flags are
// [bank][source], 2 * ranks words. Every phase drains peer stores before
// changing destination; no peer fan-out occurs. The cluster barrier,
// system release and acquire are push.cuh's and comm.cuh's.
using push::cluster_sync;

template<class E, class Input, uint32_t Ranks, class Inboxes, class Flags>
__device__ void allreduce_hc_ranks(const typename Input::Element* input, const E& epilogue,
                                   Inboxes inboxes, Flags flags, uint32_t* epoch,
                                   const uint32_t* live_rows, uint32_t capacity,
                                   uint32_t rank) {
  static_assert(Geometry<E, Input>::kVectors > 0, "checked geometry");
  static_assert(sizeof(*peer(inboxes, 0)) == sizeof(typename Input::Element),
                "inboxes of the input's elements");
  constexpr uint32_t ranks = Ranks, Width = E::kWidth;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t rows = *live_rows;
  if (!valid_ranks(rank, ranks) || rows > capacity || capacity == 0 || capacity > 8192)
    asm volatile("trap;");
  const uint32_t next = *epoch + 1;
  const uint64_t stride = uint64_t(capacity) * Width;
  const uint64_t bank = uint64_t(next & 1) * ranks * stride;
  const uint64_t n = uint64_t(rows) * Width;
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const bool leader = tid == 0;
  const uint32_t flag_bank = (next & 1) * ranks;
  for (uint32_t step = 1; step < ranks; ++step) {
    const uint32_t dest = (rank + step) % ranks;
    const uint32_t source = (rank + ranks - step) % ranks;
    auto* target = reinterpret_cast<uint4*>(peer(inboxes, dest) + bank + rank * stride);
    const auto* src = reinterpret_cast<const uint4*>(input);
    for (uint64_t i = tid; i < n / Input::kPerVector; i += 2048) target[i] = src[i];
    __threadfence_system();
    cluster_sync();
    if (leader) {
      release_system(peer(flags, dest) + flag_bank + rank, next);
      while (acquire_system(peer(flags, rank) + flag_bank + source) != next) {}
    }
    cluster_sync();
  }
  // Each CTA owns one token at a time.
  __shared__ float sums[8];
  for (uint64_t token = blockIdx.x; token < rows; token += 8) {
    float y[E::kPer];
    #pragma unroll
    for (int j = 0; j < E::kPer; ++j) {
      const uint64_t i = token * Width + threadIdx.x + j * E::kThreads;
      y[j] = 0.f;
      #pragma unroll
      for (uint32_t source = 0; source < ranks; ++source)
        y[j] += Input::decode(source == rank ? input[i]
                                             : peer(inboxes, rank)[bank + source * stride + i]);
    }
    epilogue.finish_token(y, token, sums);
  }
  cluster_sync();
  if (leader) *epoch = next;
  cluster_sync();
}

template<class E, class Input, class Inboxes, class Flags>
__device__ void allreduce_hc(const typename Input::Element* input, const E& epilogue,
                            Inboxes inboxes, Flags flags, uint32_t* epoch,
                            const uint32_t* live_rows, uint32_t capacity,
                            uint32_t rank, uint32_t ranks) {
  if (!valid_ranks(rank, ranks)) asm volatile("trap;");
  if (ranks == 2)
    allreduce_hc_ranks<E, Input, 2>(input, epilogue, inboxes, flags, epoch, live_rows, capacity,
                                    rank);
  else if (ranks == 4)
    allreduce_hc_ranks<E, Input, 4>(input, epilogue, inboxes, flags, epoch, live_rows, capacity,
                                    rank);
  else
    allreduce_hc_ranks<E, Input, 8>(input, epilogue, inboxes, flags, epoch, live_rows, capacity,
                                    rank);
}

// Prefill: a phased two-shot in two kernels, after push.cuh's. Owner
// q holds rows [q * rows / ranks, (q + 1) * rows / ranks). One 8-CTA cluster
// sends to one destination at a time (rank + 1, + 2, ...) with a cluster
// barrier before each switch; the other CTAs reduce, inject and normalize.
//
// Scatter: every rank pushes each owner its rows of the BF16 partials; the
// owner sums the ranks in rank order in F32 (the one-shot entry's sum)
// into its rows of the F32 sum buffer. The last phase releases one flag per
// block of rows, so the sum overlaps it. Gather: the cluster pushes the
// owner's F32 sums into every peer's sum buffer a block of rows at a time,
// releasing one flag per block, while workers inject and normalize each
// (row, stream) group of a block as it arrives, with the one-shot entry's
// thread order. Each rank sends (ranks - 1) / ranks of the BF16 message and
// of the F32 sums, against ranks - 1 whole BF16 messages for the one-shot
// entry.
//
// Inbox regions sit at fixed offsets and strides from the state's capacity,
// never from the live rows: the one-shot banks for OneShotRows rows, the
// scatter slots [source][rows of the largest owner] in BF16, then the F32
// sums [capacity][Width] by absolute row. The one-shot entry runs with
// capacity OneShotRows on a shared state.
//
// The pair has its own flags and call generation. The flags are 64-bit
// slots after the one-shot entry's 2 * ranks words (`Layout`: the
// scatter's [source] and [block] flags and a u32 arrival counter, then the
// gather's [source][block] flags); a flag carries the generation of the
// call that published it and receivers compare for equality. The
// generation is a 64-bit counter at byte 8 of the epoch buffer (the
// one-shot entry keeps its own word 0), which the gather advances. Flags
// and generation start at zero and the first call uses generation 1, so no
// initial or stale value matches, and the counter does not wrap (the
// entries trap instead). A peer's next scatter rewrites only scatter slots
// and flags that this rank's finished scatter read, and its next gather
// waits for this rank's next scatter, which follows this rank's gather on
// the stream; every receiver consumes every live block of the gather
// before that scatter, including CTAs with no groups in the block. So no
// flag or region is rewritten before it is read, whatever the row counts of
// the two calls.
constexpr uint32_t OneShotRows = 32;
constexpr uint32_t Cluster = 8, Threads = 256;
constexpr uint32_t Blocks = 32;

// 64-bit flag slots of the scatter; the first follows the one-shot words.
// Slots ranks to 2 * ranks are unused. Regions follow the row width.
template<uint32_t Width>
struct Layout {
  uint32_t ranks;
  __device__ __forceinline__ uint32_t first_slot() const { return ranks; }
  __device__ __forceinline__ uint32_t scatter_flags() const { return 0; }       // [source]
  __device__ __forceinline__ uint32_t block_flags() const { return 2 * ranks; } // [block]
  // The u32 arrival counter after the last block slot, then 28 spare bytes.
  __device__ __forceinline__ uint32_t done() const {
    return 2 * (first_slot() + block_flags() + Blocks);
  }
  // Elements before the scatter slots: the one-shot banks.
  __device__ __forceinline__ uint64_t scatter_base() const {
    return 2ull * ranks * OneShotRows * Width;
  }
};

struct Span { uint32_t first, count; };

__device__ __forceinline__ Span owner_rows(uint32_t rows, uint32_t owner, uint32_t ranks) {
  const uint32_t first = uint32_t(uint64_t(owner) * rows / ranks);
  return {first, uint32_t(uint64_t(owner + 1) * rows / ranks) - first};
}

__device__ __forceinline__ uint64_t slot_rows(uint32_t capacity, uint32_t ranks) {
  return (capacity + ranks - 1) / ranks;
}

template<class T, uint32_t Width>
__device__ __forceinline__ float* sum_buffer(T* inbox, uint32_t capacity, Layout<Width> layout) {
  return reinterpret_cast<float*>(inbox + layout.scatter_base() +
                                  layout.ranks * slot_rows(capacity, layout.ranks) * Width);
}

// Blocks of at least 64 rows (320 KiB) keep the per-block flags cheap.
__device__ __forceinline__ uint32_t blocks_of(uint32_t count) {
  const uint32_t blocks = count / 64;
  return blocks == 0 ? 1 : blocks > Blocks ? Blocks : blocks;
}

__device__ __forceinline__ Span block_of(uint32_t count, uint32_t block) {
  const uint32_t blocks = blocks_of(count);
  const uint32_t first = uint32_t(uint64_t(block) * count / blocks);
  return {first, uint32_t(uint64_t(block + 1) * count / blocks) - first};
}

// push.cuh's copy (four 16-byte loads in flight per thread) over a byte
// count that is a multiple of 16.
__device__ __forceinline__ void copy(void* target, const void* source, uint64_t bytes,
                                     uint32_t worker, uint32_t workers) {
  static_assert(push::kThreads == Threads, "push copy stride");
  push::copy_vectors(reinterpret_cast<Vector128*>(target),
                     reinterpret_cast<const Vector128*>(source), bytes / 16, worker, workers);
}

// push.cuh's barrier before a destination switch.
using push::next_destination;

template<uint32_t Width>
__device__ __forceinline__ uint64_t* slots(uint32_t* flags, Layout<Width> layout) {
  return reinterpret_cast<uint64_t*>(flags) + layout.first_slot();
}

__device__ __forceinline__ void publish(uint64_t* p, uint64_t v) {
  asm volatile("st.global.release.sys.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// push::wait_system for the 64-bit flags: spin with relaxed loads, then
// acquire once.
__device__ __forceinline__ void wait_for(const uint64_t* address, uint64_t mark) {
  uint64_t value;
  do {
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
  } while (value != mark);
  asm volatile("ld.global.acquire.sys.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
}

// The pair's call generation: 64 bits at byte 8 of the epoch buffer.
__device__ __forceinline__ uint64_t* generation_of(uint32_t* epoch) {
  return reinterpret_cast<uint64_t*>(epoch) + 1;
}

// A geometry mismatch would leave peers waiting, and an exhausted
// generation would repeat old flags; fail the launch instead.
__device__ __forceinline__ void check_prefill(uint32_t rows, uint32_t capacity, uint32_t rank,
                                              uint32_t ranks, uint64_t generation) {
  if (!valid_ranks(rank, ranks) || rows > capacity || capacity == 0 || capacity > 8192 ||
      blockDim.x != Threads || gridDim.x <= Cluster || gridDim.x % Cluster ||
      generation == ~0ull)
    asm volatile("trap;");
}

// The scatter runs no epilogue; E gives the pair's row width.
template<class E, class Input, uint32_t Ranks, class Inboxes, class Flags>
__device__ void reduce_scatter_ranks(const typename Input::Element* input, Inboxes inboxes,
                                     Flags flags, uint32_t* epoch, const uint32_t* live_rows,
                                     uint32_t capacity, uint32_t rank) {
  static_assert(Geometry<E, Input>::kVectors > 0, "checked geometry");
  static_assert(sizeof(*peer(inboxes, 0)) == sizeof(typename Input::Element),
                "inboxes of the input's elements");
  using Element = typename Input::Element;
  constexpr uint32_t ranks = Ranks, Width = E::kWidth, kPerVector = Input::kPerVector;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t rows = *live_rows;
  const uint64_t generation = *generation_of(epoch);
  check_prefill(rows, capacity, rank, ranks, generation);
  const Layout<Width> layout{ranks};
  const uint64_t mark = generation + 1;
  const uint64_t slot = slot_rows(capacity, ranks) * Width;
  uint64_t* local = slots(peer(flags, rank), layout);
  if (blockIdx.x < Cluster) {
    for (uint32_t step = 1; step < ranks; ++step) {
      const uint32_t dest = (rank + step) % ranks;
      const Span span = owner_rows(rows, dest, ranks);
      Element* target = peer(inboxes, dest) + layout.scatter_base() + rank * slot;
      const Element* source = input + uint64_t(span.first) * Width;
      if (step + 1 < ranks) {
        copy(target, source, uint64_t(span.count) * Width * sizeof(Element), blockIdx.x, Cluster);
        next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0)
          publish(slots(peer(flags, dest), layout) + layout.scatter_flags() + rank, mark);
        continue;
      }
      for (uint32_t block = blockIdx.x; block < blocks_of(span.count); block += Cluster) {
        const Span part = block_of(span.count, block);
        copy(target + uint64_t(part.first) * Width, source + uint64_t(part.first) * Width,
             uint64_t(part.count) * Width * sizeof(Element), 0, 1);
        __syncthreads();
        if (threadIdx.x == 0)
          publish(slots(peer(flags, dest), layout) + layout.block_flags() + block, mark);
      }
    }
    return;
  }
  const uint32_t worker = blockIdx.x - Cluster;
  const uint32_t workers = gridDim.x - Cluster;
  const uint32_t last = (rank + 1) % ranks;   // source of the last phase
  if (threadIdx.x < ranks && threadIdx.x != rank && threadIdx.x != last)
    wait_for(local + layout.scatter_flags() + threadIdx.x, mark);
  __syncthreads();
  const Span span = owner_rows(rows, rank, ranks);
  const Element* inbox = peer(inboxes, rank) + layout.scatter_base();
  float* sums = sum_buffer(peer(inboxes, rank), capacity, layout) + uint64_t(span.first) * Width;
  const Element* own = input + uint64_t(span.first) * Width;
  // Test-only lag (kernel-lab's skew cubin): a late reader lets its peers
  // run ahead into the next calls.
  test_skew(rank, ranks, uint32_t(mark));
  for (uint32_t block = 0; block < blocks_of(span.count); ++block) {
    if (threadIdx.x == 0) wait_for(local + layout.block_flags() + block, mark);
    __syncthreads();
    const Span part = block_of(span.count, block);
    const uint64_t end = uint64_t(part.first + part.count) * Width / kPerVector;
    for (uint64_t v = uint64_t(part.first) * Width / kPerVector + uint64_t(worker) * Threads +
                      threadIdx.x;
         v < end; v += uint64_t(workers) * Threads) {
      uint4 parts[ranks];
      #pragma unroll
      for (uint32_t source = 0; source < ranks; ++source)
        parts[source] = reinterpret_cast<const uint4*>(
            source == rank ? own : inbox + source * slot)[v];
      float y[kPerVector] = {};
      #pragma unroll
      for (uint32_t source = 0; source < ranks; ++source) {
        const Element* values = reinterpret_cast<const Element*>(&parts[source]);
        #pragma unroll
        for (uint32_t k = 0; k < kPerVector; ++k) y[k] += Input::decode(values[k]);
      }
      // The vector's sums, four F32 per 16 bytes.
      #pragma unroll
      for (uint32_t q = 0; q < kPerVector / 4; ++q)
        reinterpret_cast<float4*>(sums)[kPerVector / 4 * v + q] =
            make_float4(y[4 * q], y[4 * q + 1], y[4 * q + 2], y[4 * q + 3]);
    }
  }
}

template<class E, class Input, class Inboxes, class Flags>
__device__ void reduce_scatter(const typename Input::Element* input, Inboxes inboxes,
                               Flags flags, uint32_t* epoch, const uint32_t* live_rows,
                               uint32_t capacity, uint32_t rank, uint32_t ranks) {
  if (!valid_ranks(rank, ranks)) asm volatile("trap;");
  if (ranks == 2)
    reduce_scatter_ranks<E, Input, 2>(input, inboxes, flags, epoch, live_rows, capacity, rank);
  else if (ranks == 4)
    reduce_scatter_ranks<E, Input, 4>(input, inboxes, flags, epoch, live_rows, capacity, rank);
  else
    reduce_scatter_ranks<E, Input, 8>(input, inboxes, flags, epoch, live_rows, capacity, rank);
}

// The gather reads only the F32 sums; Input places them in the inbox. The
// next gather cannot overwrite sums until every rank has entered the next
// scatter.
template<class E, class Input, class Inboxes, class Flags>
__device__ void allgather_hc_v2(const E& epilogue, Inboxes inboxes, Flags flags, uint32_t* epoch,
                                const uint32_t* live_rows, uint32_t capacity, uint32_t rank,
                                uint32_t ranks) {
  static_assert(Geometry<E, Input>::kVectors > 0, "checked geometry");
  static_assert(sizeof(*peer(inboxes, 0)) == sizeof(typename Input::Element),
                "inboxes of the input's elements");
  constexpr uint32_t Width = E::kWidth;
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t rows = *live_rows;
  const uint64_t generation = *generation_of(epoch);
  check_prefill(rows, capacity, rank, ranks, generation);
  const Layout<Width> layout{ranks};
  const uint64_t mark = generation + 1;
  uint64_t* local = slots(peer(flags, rank), layout);
  // The gather's [source][block] flags follow the counter and spare bytes.
  const uint32_t gather_blocks = 2 * ranks + Blocks + 4;
  unsigned int* done = peer(flags, rank) + layout.done();
  const float* sums = sum_buffer(peer(inboxes, rank), capacity, layout);
  const uint32_t workers = gridDim.x - Cluster;
  if (blockIdx.x < Cluster) {
    const Span own = owner_rows(rows, rank, ranks);
    const uint64_t first = uint64_t(own.first) * Width;
    for (uint32_t step = 1; step < ranks; ++step) {
      const uint32_t dest = (rank + step) % ranks;
      for (uint32_t block = 0; block < blocks_of(own.count); ++block) {
        const Span part = block_of(own.count, block);
        const uint64_t offset = first + uint64_t(part.first) * Width;
        copy(sum_buffer(peer(inboxes, dest), capacity, layout) + offset, sums + offset,
             uint64_t(part.count) * Width * 4, blockIdx.x, Cluster);
        next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0)
          publish(slots(peer(flags, dest), layout) + gather_blocks + rank * Blocks + block, mark);
      }
    }
  } else {
    __shared__ float partial[8];
    const uint32_t worker = blockIdx.x - Cluster;
    for (uint32_t step = 0; step < ranks; ++step) {
      const uint32_t source = (rank + ranks - step) % ranks;
      if (step == 1) test_skew(rank, ranks, uint32_t(mark));
      const Span span = owner_rows(rows, source, ranks);
      for (uint32_t block = 0; block < blocks_of(span.count); ++block) {
        if (step > 0) {
          if (threadIdx.x == 0)
            wait_for(local + gather_blocks + source * Blocks + block, mark);
          __syncthreads();
        }
        const Span part = block_of(span.count, block);
        const uint64_t first = uint64_t(span.first + part.first) * E::kStreams;
        const uint64_t end = first + uint64_t(part.count) * E::kStreams;
        for (uint64_t group = first + worker; group < end; group += workers) {
          const uint64_t token = group / E::kStreams;
          float y[E::kPer];
          #pragma unroll
          for (int j = 0; j < E::kPer; ++j)
            y[j] = sums[token * Width + threadIdx.x + j * E::kThreads];
          uint16_t values[E::kPer];
          epilogue.operand(group, values);
          epilogue.finish(y, values, epilogue.scale(group), group, partial);
          __syncthreads();
        }
      }
    }
  }
  // CTA 0 of the cluster (after its barriers, so the whole cluster read the
  // generation) and every worker arrive; the last clears the counter and
  // advances the generation, after every CTA of this call read it.
  __syncthreads();
  if (threadIdx.x != 0 || (blockIdx.x < Cluster && blockIdx.x != 0)) return;
  __threadfence();
  if (atomicAdd(done, 1u) == workers) {
    *done = 0;
    *generation_of(epoch) = mark;
  }
}
} // namespace chop::comm::qwen38
