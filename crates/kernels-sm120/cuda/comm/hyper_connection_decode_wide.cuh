#pragma once
#include "hyper_connection_decode.cuh"

// Wide decode hyper-connection all-reduce for up to kWideRows prepared rows
// (chop_comm_qwen38_allreduce_hc_wide_r<Ranks>_v1), bitwise as the decode
// and one-shot entries: every rank sums the partials in rank order in F32,
// never rounds the sum, and finishes each (row, stream) group with the
// epilogue E's finish. It extends the decode state: its regions follow the
// decode entry's in the inbox and its flag words follow the decode flags,
// and it shares the one-shot epoch word and arrival counter, so calls of
// either entry with any rows follow each other on one state.
//
// Two protocols, by prepared rows and the rank count's two-shot rows
// (CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R<Ranks>); both send to one peer at a
// time, and their last phase releases a flag per block of rows so workers
// start while it is in flight:
//   one-shot  push the live rows to each peer in turn; inbox and flags
//             double-buffered by epoch parity. Every call, zero live rows
//             included, releases a flag per peer and waits for every
//             peer's, as the decode one-shot does.
//   two-shot  owner q reduces columns [q W, (q + 1) W), W = width / Ranks:
//             push each owner its BF16 columns, reduce in F32 and push the
//             F32 sums to every peer (the first all-gather step goes to the
//             last reduce-scatter destination and reduces each block as it
//             lands). Its reduce-scatter flags are double-buffered by epoch
//             parity and acknowledge every call like the one-shot's: with
//             no live row nothing else is exchanged, and a peer must still
//             not run two calls ahead, or its next one-shot call could
//             rewrite a bank this rank reads.
// Blocks without a live row carry no flag on any rank, since ranks share
// the prepared and live rows. Workers (every CTA after the sending cluster,
// any number) handle (row, stream) groups worker, worker + workers, ...

#if !defined(CHOP_QWEN38_WIDE_BLOCK_ROWS) || !defined(CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R2) || \
    !defined(CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R4) || !defined(CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R8)
#error "wide hyper-connection decode comm defines are required"
#endif

namespace chop::comm::qwen38_decode {
constexpr uint32_t kWideRows = 256;
constexpr uint32_t kWideBlockRows = CHOP_QWEN38_WIDE_BLOCK_ROWS;
__host__ __device__ constexpr uint32_t wide_blocks(uint32_t rows) {
  return kSenders * ((rows + kSenders * kWideBlockRows - 1) / (kSenders * kWideBlockRows));
}
constexpr uint32_t kWideMaxBlocks = wide_blocks(kWideRows);
// Flag words after the decode one-shot's [parity][source]: one-shot
// [parity][source] and [parity][block], two-shot reduce-scatter
// [parity][source], all-gather [source], both last phases' [block], the
// two-shot's local ready word, then shape words and their flags
// [parity][source] (push::publish_shape).
constexpr uint32_t kWideOneShotFlags = 2 * kMaxRanks;
constexpr uint32_t kWideOneShotBlocks = kWideOneShotFlags + 2 * kMaxRanks;
constexpr uint32_t kWideScatterFlags = kWideOneShotBlocks + 2 * kWideMaxBlocks;
constexpr uint32_t kWideGatherFlags = kWideScatterFlags + 2 * kMaxRanks;
constexpr uint32_t kWideScatterBlocks = kWideGatherFlags + kMaxRanks;
constexpr uint32_t kWideGatherBlocks = kWideScatterBlocks + kWideMaxBlocks;
constexpr uint32_t kWideReady = kWideGatherBlocks + kWideMaxBlocks;
constexpr uint32_t kWideShape = kWideReady + 1;
constexpr uint32_t kWideShapeFlags = kWideShape + 2 * kMaxRanks;
constexpr uint32_t kWideFlagWords = kWideShapeFlags + 2 * kMaxRanks;

template<uint32_t Ranks>
__host__ __device__ constexpr uint32_t wide_two_shot_rows() {
  return Ranks == 2 ? CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R2
       : Ranks == 4 ? CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R4
                    : CHOP_QWEN38_WIDE_TWO_SHOT_ROWS_R8;
}

// Inbox bytes: the decode entry's regions, then the one-shot region
// [parity][source][kWideRows][width] input elements, then the two-shot's
// reduce-scatter slots [source][kWideRows][W] input elements and all-gather
// slots [source][kWideRows][W] F32.
template<class E, class I, uint32_t Ranks>
__host__ __device__ constexpr uint64_t wide_one_shot_base() {
  return 2ull * Ranks * kLlPlane<E, I> * 8 +
         2ull * Ranks * kOneShotStride<E, I> * sizeof(typename I::Element);
}
template<class E, class I, uint32_t Ranks>
__host__ __device__ constexpr uint64_t wide_scatter_base() {
  return wide_one_shot_base<E, I, Ranks>() +
         2ull * Ranks * kWideRows * E::kWidth * sizeof(typename I::Element);
}
template<class E, class I, uint32_t Ranks>
__host__ __device__ constexpr uint64_t wide_gather_base() {
  return wide_scatter_base<E, I, Ranks>() +
         uint64_t(kWideRows) * E::kWidth * sizeof(typename I::Element);
}
template<class E, class I, uint32_t Ranks>
__host__ __device__ constexpr uint64_t wide_inbox_bytes() {
  return wide_gather_base<E, I, Ranks>() + uint64_t(kWideRows) * E::kWidth * 4;
}

// qwen38_decode::wide_buffers computes the same sizes; its tests pin them.
static_assert(kWideFlagWords == 153, "wide_buffers in src/comm/qwen38_decode.rs");

struct RowSpan { uint32_t first, end; };

// Every rank must run the same prepared and live rows: blocks without a
// live row carry no flag, so a rank that disagreed would wait for a block
// its peer never flags. CTA 0 of the sending cluster stores its shape to
// every peer before its data and checks every peer's before any wait on
// them (push::publish_shape, push::check_shapes); a difference traps on
// every rank. The words are double-buffered by epoch parity: a peer may
// finish this call before this rank checks, but not the next one.
template<class E, class Input>
__device__ __forceinline__ uint32_t wide_shape(const Call<E, Input>& c, uint32_t rows) {
  return rows | (c.live << 16);
}
// Live rows of block `block` of `per` prepared rows.
__device__ __forceinline__ RowSpan wide_block(uint32_t block, uint32_t per, uint32_t rows,
                                              uint32_t live) {
  const uint32_t first = min(block * per, min(rows, live));
  return {first, min(first + per, min(rows, live))};
}

template<typename T, typename Inboxes>
__device__ __forceinline__ T* wide_region(Inboxes inboxes, uint32_t owner, uint64_t base) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(peer(inboxes, owner)) + base);
}

// The sending cluster (counted once, by CTA 0) and every worker arrive; the
// last clears the counter and advances the epoch after all of them read it.
// The cluster barrier keeps CTA 0 from arriving before the other sending
// CTAs have read the epoch: at two ranks no earlier barrier orders them.
template<class E, class Input>
__device__ __forceinline__ void wide_finish(const Call<E, Input>& c, uint32_t epoch) {
  __syncthreads();
  if (blockIdx.x < kSenders) cluster_sync();
  if (threadIdx.x != 0 || (blockIdx.x < kSenders && blockIdx.x != 0)) return;
  __threadfence();
  if (atomicAdd(c.epochs + kCounter, 1u) == gridDim.x - kSenders) {
    c.epochs[kCounter] = 0;
    c.epochs[kOneShotEpoch] = epoch;
  }
}

// Vectors in flight per thread in the two-shot's sends, reductions and
// resends. One: batches of four, loads before stores, measured slower on
// the sending CTAs (their remote stores set the pace).
constexpr uint32_t kWideUnroll = 1;
// The reduction loads one vector per rank for each it produces; at most 16
// in flight keep eight ranks within two CTAs' registers per SM.
template<uint32_t Ranks>
__host__ __device__ constexpr uint32_t wide_reduce_unroll() {
  return kWideUnroll * Ranks <= 16 ? kWideUnroll : (16 / Ranks > 0 ? 16 / Ranks : 1);
}

// load(v) for vectors first + start, + step, ... below end, stored with
// store(v, value), Unroll at a time.
template<uint32_t Unroll = kWideUnroll, class Load, class Store>
__device__ __forceinline__ void wide_batches(uint64_t first, uint64_t end, uint64_t start,
                                             uint64_t step, Load load, Store store) {
  uint64_t v = first + start;
  for (; v + (Unroll - 1) * step < end; v += Unroll * step) {
    decltype(load(v)) values[Unroll];
#pragma unroll
    for (uint32_t i = 0; i < Unroll; ++i) values[i] = load(v + i * step);
#pragma unroll
    for (uint32_t i = 0; i < Unroll; ++i) store(v + i * step, values[i]);
  }
  for (; v < end; v += step) store(v, load(v));
}

// The F32 sums of one input vector's elements, four per 16 bytes.
template<uint32_t N>
struct SumVectors { Vector128 v[N]; };

// Workers' loop over (row, stream) groups of the live rows: `wait(block)`
// runs on thread 0 when a group's block differs from the last one's, and
// `sum(row, y)` fills the group's rank-ordered F32 sums.
template<class E, class Input, class Wait, class Sum>
__device__ __forceinline__ void wide_groups(const Call<E, Input>& c, uint32_t per,
                                            unsigned int* local,
                                            Wait wait, Sum sum) {
  __shared__ float sums[8];
  const uint32_t workers = gridDim.x - kSenders;
  uint32_t waited = kWideMaxBlocks;
  for (uint64_t group = blockIdx.x - kSenders; group < uint64_t(c.live) * E::kStreams;
       group += workers) {
    const uint32_t row = static_cast<uint32_t>(group / E::kStreams);
    uint16_t residual[E::kPer];
    c.epilogue.operand(group, residual);
    const auto scale = c.epilogue.scale(group);
    const uint32_t block = row / per;
    if (block != waited && threadIdx.x == 0) wait(block);
    waited = block;
    __syncthreads();
    float y[E::kPer];
    sum(row, y);
    c.epilogue.finish(y, residual, scale, group, sums);
    // The next group's reduction reuses `sums`.
    __syncthreads();
  }
}

template<class E, class Input, uint32_t Ranks, class Inboxes, class Flags>
__device__ void wide_one_shot(const Call<E, Input>& c, uint32_t rows, Inboxes inboxes,
                              Flags flags) {
  using Element = typename Input::Element;
  constexpr uint32_t kWidth = E::kWidth, kVectors = qwen38_decode::kVectors<E, Input>, kPer = E::kPer;
  const uint32_t epoch = c.epochs[kOneShotEpoch] + 1;
  const uint32_t mark = tagged(epoch);
  const uint32_t parity = epoch & 1;
  const uint32_t blocks = wide_blocks(rows);
  const uint32_t per = (rows + blocks - 1) / blocks;
  constexpr uint64_t stride = uint64_t(kWideRows) * kVectors;  // vectors per source
  const uint64_t bank = uint64_t(parity) * Ranks * stride;
  unsigned int* local = peer(flags, c.rank);
  unsigned int* full = local + kWideOneShotFlags + parity * kMaxRanks;
  const uint32_t shape_words = kWideShape + parity * kMaxRanks;
  const uint32_t shape_flags = kWideShapeFlags + parity * kMaxRanks;
  if (blockIdx.x < kSenders) {
    if (blockIdx.x == 0)
      push::publish_shape(flags, c.rank, Ranks, shape_words, shape_flags, wide_shape(c, rows),
                          mark);
    const auto* src = reinterpret_cast<const Vector128*>(c.input);
    for (uint32_t step = 1; step < Ranks; ++step) {
      const uint32_t dest = (c.rank + step) % Ranks;
      test_skew(c.rank, Ranks, epoch);
      Vector128* target = wide_region<Vector128>(inboxes, dest, wide_one_shot_base<E, Input, Ranks>()) +
                          bank + c.rank * stride;
      unsigned int* dest_flags = peer(flags, dest);
      if (step + 1 < Ranks) {
        push::copy_vectors(target, src, uint64_t(c.live) * kVectors, blockIdx.x, kSenders);
        push::next_destination();
      } else {
        // The last phase flags each live block from the CTA that sent it.
        for (uint32_t block = blockIdx.x; block < blocks; block += kSenders) {
          const RowSpan live = wide_block(block, per, rows, c.live);
          if (live.first == live.end) continue;
          push::copy_vectors(target + uint64_t(live.first) * kVectors,
                             src + uint64_t(live.first) * kVectors,
                             uint64_t(live.end - live.first) * kVectors, 0, 1);
          __syncthreads();
          if (threadIdx.x == 0)
            release_system(dest_flags + kWideOneShotBlocks + parity * kWideMaxBlocks + block,
                           mark);
        }
      }
      // One flag per peer and call acknowledges it, whatever the live rows.
      // Workers wait for it before reading an earlier phase's rows; the last
      // phase's rows have their block flags.
      if (blockIdx.x == 0 && threadIdx.x == 0)
        release_system(dest_flags + kWideOneShotFlags + parity * kMaxRanks + c.rank, mark);
    }
    // This rank's next call may rewrite a peer's bank of this parity only
    // after that peer has published this call, which it does after
    // finishing the previous one.
    if (blockIdx.x == 0) {
      push::check_shapes(local, c.rank, Ranks, shape_words, shape_flags, wide_shape(c, rows),
                         mark);
      if (threadIdx.x < Ranks && threadIdx.x != c.rank) wait_system(full + threadIdx.x, mark);
      __syncthreads();
    }
  } else {
    const uint32_t last = (c.rank + 1) % Ranks;  // its last phase comes here
    if (threadIdx.x < Ranks && threadIdx.x != c.rank && threadIdx.x != last && c.live > 0)
      wait_system(full + threadIdx.x, mark);
    test_skew(c.rank, Ranks, epoch);
    const Element* received =
        reinterpret_cast<const Element*>(wide_region<Vector128>(inboxes, c.rank,
                                                                wide_one_shot_base<E, Input, Ranks>()) +
                                         bank);
    wide_groups(
        c, per, local,
        [&](uint32_t block) {
          wait_system(local + kWideOneShotBlocks + parity * kWideMaxBlocks + block, mark);
        },
        [&](uint32_t row, float (&y)[kPer]) {
#pragma unroll
          for (uint32_t j = 0; j < kPer; ++j) {
            const uint64_t i = uint64_t(row) * kWidth + threadIdx.x + j * kThreads;
            y[j] = 0.f;
#pragma unroll
            for (uint32_t source = 0; source < Ranks; ++source)
              y[j] += Input::decode(source == c.rank
                                        ? c.input[i]
                                        : received[source * stride * Input::kPerVector + i]);
          }
        });
  }
  wide_finish(c, epoch);
}

template<class E, class Input, uint32_t Ranks, class Inboxes, class Flags>
__device__ void wide_two_shot(const Call<E, Input>& c, uint32_t rows, Inboxes inboxes,
                              Flags flags) {
  constexpr uint32_t kWidth = E::kWidth, kVectors = qwen38_decode::kVectors<E, Input>, kPer = E::kPer;
  constexpr uint32_t kPerVector = Input::kPerVector;
  constexpr uint32_t N = kPerVector / 4;       // F32 sum vectors per input vector
  using Sums = SumVectors<N>;
  constexpr uint32_t W = kWidth / Ranks;       // columns per owner
  constexpr uint32_t WV = W / kPerVector;      // input vectors per owner and row
  static_assert(W % kPerVector == 0, "whole vectors per owner");
  constexpr uint64_t slot = uint64_t(kWideRows) * WV;  // input vectors per source
  const uint32_t epoch = c.epochs[kOneShotEpoch] + 1;
  const uint32_t mark = tagged(epoch);
  const uint32_t blocks = wide_blocks(rows);
  const uint32_t per = (rows + blocks - 1) / blocks;
  unsigned int* local = peer(flags, c.rank);
  const auto* input = reinterpret_cast<const Vector128*>(c.input);
  // Slot vector v of owner q: row v / WV, input vector (v / WV) * 320 + q WV + v % WV.
  auto column = [](uint64_t v, uint32_t owner) {
    return (v / WV) * kVectors + owner * WV + v % WV;
  };
  // F32 sums as N vectors per slot vector: v's kPerVector sums.
  Vector128* sums_of = wide_region<Vector128>(inboxes, c.rank, wide_gather_base<E, Input, Ranks>());
  const uint32_t parity = epoch & 1;
  unsigned int* acks = local + kWideScatterFlags + parity * kMaxRanks;
  const uint32_t shape_words = kWideShape + parity * kMaxRanks;
  const uint32_t shape_flags = kWideShapeFlags + parity * kMaxRanks;
  if (blockIdx.x < kSenders) {
    if (blockIdx.x == 0)
      push::publish_shape(flags, c.rank, Ranks, shape_words, shape_flags, wide_shape(c, rows),
                          mark);
    const uint64_t part = uint64_t(c.live) * WV;
    for (uint32_t step = 1; step < Ranks; ++step) {
      const uint32_t dest = (c.rank + step) % Ranks;
      test_skew(c.rank, Ranks, epoch);
      Vector128* target = wide_region<Vector128>(inboxes, dest, wide_scatter_base<E, Input, Ranks>()) +
                          c.rank * slot;
      auto load = [&](uint64_t v) { return input[column(v, dest)]; };
      auto store = [&](uint64_t v, Vector128 value) { target[v] = value; };
      if (step + 1 < Ranks) {
        wide_batches(0, part, blockIdx.x * kThreads + threadIdx.x, kSenders * kThreads, load,
                     store);
        push::next_destination();
      } else {
        for (uint32_t block = blockIdx.x; block < blocks; block += kSenders) {
          const RowSpan live = wide_block(block, per, rows, c.live);
          if (live.first == live.end) continue;
          wide_batches(uint64_t(live.first) * WV, uint64_t(live.end) * WV, threadIdx.x, kThreads,
                       load, store);
          __syncthreads();
          if (threadIdx.x == 0)
            release_system(peer(flags, dest) + kWideScatterBlocks + block, mark);
        }
      }
      // Every phase's flag, the last one's included, acknowledges the call.
      if (blockIdx.x == 0 && threadIdx.x == 0)
        release_system(peer(flags, dest) + kWideScatterFlags + parity * kMaxRanks + c.rank,
                       mark);
    }
    if (blockIdx.x == 0)
      push::check_shapes(local, c.rank, Ranks, shape_words, shape_flags, wide_shape(c, rows),
                         mark);
    if (c.live > 0) {
      const uint32_t last = (c.rank + 1) % Ranks;  // source of the last reduce-scatter phase
      if (threadIdx.x < Ranks && threadIdx.x != c.rank && threadIdx.x != last)
        wait_system(acks + threadIdx.x, mark);
      const Vector128* scatter = wide_region<const Vector128>(inboxes, c.rank,
                                                              wide_scatter_base<E, Input, Ranks>());
      for (uint32_t k = 0; k + 1 < Ranks; ++k) {
        const uint32_t dest = (c.rank + Ranks - 1 - k) % Ranks;
        const bool reducing = k == 0;
        const bool final_step = k + 2 == Ranks;
        Vector128* target =
            wide_region<Vector128>(inboxes, dest, wide_gather_base<E, Input, Ranks>()) +
            N * c.rank * slot;
        Vector128* own = sums_of + N * c.rank * slot;
        for (uint32_t block = blockIdx.x; block < blocks; block += kSenders) {
          const RowSpan live = wide_block(block, per, rows, c.live);
          if (live.first == live.end) continue;
          const uint64_t first = uint64_t(live.first) * WV, end = uint64_t(live.end) * WV;
          if (reducing) {
            if (threadIdx.x == 0) wait_system(local + kWideScatterBlocks + block, mark);
            __syncthreads();
            // The rank-ordered F32 sums of slot vector v's eight columns.
            auto reduce = [&](uint64_t v) {
              Vector128 parts[Ranks];
#pragma unroll
              for (uint32_t source = 0; source < Ranks; ++source)
                parts[source] = source == c.rank ? input[column(v, c.rank)]
                                                 : scatter[source * slot + v];
              float y[kPerVector] = {};
#pragma unroll
              for (uint32_t source = 0; source < Ranks; ++source)
                Input::accumulate(parts[source], y);
              Sums sums;
#pragma unroll
              for (uint32_t q = 0; q < N; ++q)
                sums.v[q] = {__float_as_uint(y[4 * q]), __float_as_uint(y[4 * q + 1]),
                             __float_as_uint(y[4 * q + 2]), __float_as_uint(y[4 * q + 3])};
              return sums;
            };
            wide_batches<wide_reduce_unroll<Ranks>()>(first, end, threadIdx.x, kThreads, reduce,
                         [&](uint64_t v, Sums sums) {
#pragma unroll
                           for (uint32_t q = 0; q < N; ++q) own[N * v + q] = sums.v[q];
#pragma unroll
                           for (uint32_t q = 0; q < N; ++q) target[N * v + q] = sums.v[q];
                         });
          } else {
            // Each thread resends the sums it wrote.
            wide_batches(
                first, end, threadIdx.x, kThreads,
                [&](uint64_t v) {
                  Sums sums;
#pragma unroll
                  for (uint32_t q = 0; q < N; ++q) sums.v[q] = own[N * v + q];
                  return sums;
                },
                [&](uint64_t v, Sums sums) {
#pragma unroll
                  for (uint32_t q = 0; q < N; ++q) target[N * v + q] = sums.v[q];
                });
          }
          if (final_step) {
            __syncthreads();
            if (threadIdx.x == 0)
              release_system(peer(flags, dest) + kWideGatherBlocks + block, mark);
          }
        }
        if (reducing) {
          // Workers read this rank's sums after the ready word; the cluster
          // barrier orders every CTA's stores before it.
          __syncthreads();
          cluster_sync();
          if (blockIdx.x == 0 && threadIdx.x == 0) {
            release_gpu(local + kWideReady, mark);
            if (!final_step) release_system(peer(flags, dest) + kWideGatherFlags + c.rank, mark);
          }
        } else if (!final_step) {
          push::next_destination();
          if (blockIdx.x == 0 && threadIdx.x == 0)
            release_system(peer(flags, dest) + kWideGatherFlags + c.rank, mark);
        }
      }
    }
    // This rank's next call may rewrite a peer's flags of this parity only
    // after that peer has published this call.
    if (blockIdx.x == 0) {
      if (threadIdx.x < Ranks && threadIdx.x != c.rank) wait_system(acks + threadIdx.x, mark);
      __syncthreads();
    }
  } else if (c.live > 0) {
    const uint32_t last = (c.rank + Ranks - 1) % Ranks;  // source of the last all-gather step
    if (threadIdx.x < Ranks) {
      if (threadIdx.x == c.rank) push::wait_gpu(local + kWideReady, mark);
      else if (threadIdx.x != last) wait_system(local + kWideGatherFlags + threadIdx.x, mark);
    }
    test_skew(c.rank, Ranks, epoch);
    const float* gathered = reinterpret_cast<const float*>(sums_of);
    wide_groups(
        c, per, local,
        [&](uint32_t block) { wait_system(local + kWideGatherBlocks + block, mark); },
        [&](uint32_t row, float (&y)[kPer]) {
#pragma unroll
          for (uint32_t j = 0; j < kPer; ++j) {
            const uint32_t col = threadIdx.x + j * kThreads;
            y[j] = gathered[(col / W) * slot * kPerVector + uint64_t(row) * W + col % W];
          }
        });
  }
  wide_finish(c, epoch);
}

// The wide entry for `Ranks` ranks, epilogue E and input Input: the decode
// entry's arguments, prepared rows 1 to kWideRows. The grid is the sending
// cluster plus any number of workers in whole clusters.
template<class E, class Input, uint32_t Ranks, class Inboxes, class Flags>
__device__ void allreduce_hc_wide(const typename Input::Element* input, const E& epilogue,
                                  Inboxes inboxes, Flags flags, uint32_t* epochs,
                                  const uint32_t* live_rows, uint32_t rows, uint32_t rank,
                                  uint32_t ranks) {
  static_assert(Rows<E, Input>::kVectors > 0, "checked geometry");
  static_assert(qwen38::valid_ranks(0, Ranks), "hyper-connection entries take 2, 4 or 8 ranks");
  static_assert(sizeof(Inboxes) == Ranks * sizeof(inboxes.slot_0) &&
                sizeof(Flags) == Ranks * sizeof(flags.slot_0), "one slot per rank");
  static_assert(sizeof(inboxes.slot_0[0]) == sizeof(typename Input::Element),
                "inboxes of the input's elements");
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  const uint32_t live = *live_rows;
  if (ranks != Ranks || rank >= Ranks || rows == 0 || rows > kWideRows || live > rows)
    asm volatile("trap;");
  if (gridDim.x <= kSenders || gridDim.x % kSenders != 0 || gridDim.y != 1 || gridDim.z != 1 ||
      blockDim.x != kThreads || ptx::cluster_nctarank() != kSenders)
    asm volatile("trap;");
  const Call<E, Input> c{input, epilogue, epochs, live, rank};
  if (rows >= wide_two_shot_rows<Ranks>()) wide_two_shot<E, Input, Ranks>(c, rows, inboxes, flags);
  else wide_one_shot<E, Input, Ranks>(c, rows, inboxes, flags);
}
} // namespace chop::comm::qwen38_decode
