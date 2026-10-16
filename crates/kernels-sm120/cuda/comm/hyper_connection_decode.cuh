#pragma once
#include "hyper_connection.cuh"

// Decode hyper-connection all-reduce for up to 32 rows: the rank-ordered
// F32 sum of the ranks' partials, finished per (row, stream) group by the
// epilogue functor E (for example comm/epilogue/hc_inject_norm.cuh),
// bitwise as chop_comm_qwen38_allreduce_hc_v1 (the same rank-order F32 sum
// and epilogue). Row width and stream count are E's (Rows below).
//
// Every path sends to one peer at a time (see README.md, "Measured PCIe
// behaviour"). The protocol follows the prepared rows, never the live count,
// so launch geometry and dispatch agree:
//   low latency  rows <= CHOP_QWEN38_LL_MAX_ROWS: one cluster of four CTAs
//                per row. Recursive doubling of 8-byte LL entries (two BF16
//                values and a NaN-pattern flag word), no fences or flags;
//                power-of-two ranks only. Each CTA exchanges a quarter of
//                the row (one part per stream) and normalizes one stream:
//                v2 copies each CTA's F32 quarter sums into the others over
//                DSMEM (low_latency), v3 reads the whole row from its local
//                inbox (low_latency_full_row).
//   one-shot     one 8-CTA cluster pushes the live rows to each peer in turn;
//                one CTA per (row, stream) waits for the peers' flags and
//                reduces, injects and normalizes its stream.
// Any live count up to the prepared rows is allowed, zero included: LL rows
// at or past it do nothing on every rank, and the one-shot sender waits for
// every peer's flags on every call.
// One entry per rank count (2, 4 or 8), each compiled for it with that many
// pointer slots; region sizes and strides follow it. Inbox: the LL region [parity][source][row][vector][4 x
// u64], then the one-shot region [parity][source][row][width] BF16. Flags:
// one-shot [parity][source]. Epochs (local): LL rows [0, 32), the one-shot
// epoch 32 and its arrival counter 33.

#if !defined(CHOP_QWEN38_LL_MAX_ROWS)
#error "CHOP_QWEN38_LL_MAX_ROWS is required"
#endif

namespace chop::comm::qwen38_decode {
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaxRows = 32;
constexpr uint32_t kSenders = 8;                   // one-shot sending cluster
constexpr uint32_t kOneShotEpoch = kMaxRows, kCounter = kMaxRows + 1;
static_assert(CHOP_QWEN38_LL_MAX_ROWS <= kMaxRows, "LL rows");

// Row geometry of the epilogue E's rows of E::kWidth columns and
// E::kStreams streams and of the input Input's 16-byte vectors, after the
// checks every hyper-connection entry makes (qwen38::Geometry). The LL
// cluster has one CTA per stream, each exchanging one part of the row, one
// vector per thread: it fits only up to eight streams (the most CTAs of a
// portable cluster) whose parts are whole vectors and at most a CTA's
// threads. A geometry that does not fit builds the decode entries without
// LL, and its cubin must then set CHOP_QWEN38_LL_MAX_ROWS to 0 so every
// prepared row count takes the one-shot. That one-shot also serves the row
// counts whose groups do not fill whole sending clusters (one row of four
// streams, for one): its group CTAs are padded to a multiple of kSenders,
// and the padding CTAs leave at entry before any load (kPadsGroups,
// padding_cta).
template<class E, class Input>
struct Rows : qwen38::Geometry<E, Input> {
  static constexpr uint32_t kWidth = E::kWidth, kStreams = E::kStreams;
  static constexpr uint32_t kPerVector = Input::kPerVector;  // elements per vector
  static constexpr uint32_t kVectors = qwen38::Geometry<E, Input>::kVectors;  // per row
  static constexpr uint32_t kParts = kStreams;              // LL cluster: one CTA per stream
  static constexpr uint32_t kPartVectors = kVectors / kParts;
  static constexpr uint32_t kPartWidth = kWidth / kParts;
  static constexpr uint32_t kPer = E::kPer;                 // columns per epilogue thread
  static constexpr uint64_t kLlPlane = uint64_t(kMaxRows) * kVectors * 4;  // u64 per source
  static constexpr uint64_t kOneShotStride = uint64_t(kMaxRows) * kWidth;  // elements per source
  static constexpr bool kLlFits = kParts <= 8 && kPartVectors * kParts == kVectors &&
                                  kPartVectors <= kThreads;
  static constexpr bool kPadsGroups = !kLlFits;
  // Group CTAs of the one-shot for `rows` prepared rows.
  __host__ __device__ static constexpr uint32_t group_ctas(uint32_t rows) {
    return kPadsGroups ? (rows * kStreams + kSenders - 1) / kSenders * kSenders : rows * kStreams;
  }
  // Whether CTA `block` of a one-shot grid for `rows` prepared rows is
  // padding: past the sending cluster and every (row, stream) group.
  __host__ __device__ static constexpr bool padding_cta(uint32_t block, uint32_t rows) {
    return kPadsGroups && block >= kSenders + rows * kStreams;
  }
  static_assert(kLlFits || CHOP_QWEN38_LL_MAX_ROWS == 0,
                "the LL layout needs at most eight streams of whole parts of at most 256 "
                "vectors; build this geometry with CHOP_QWEN38_LL_MAX_ROWS 0");
};
template<class E, class I> constexpr uint32_t kVectors = Rows<E, I>::kVectors;
template<class E, class I> constexpr uint32_t kParts = Rows<E, I>::kParts;
template<class E, class I> constexpr uint32_t kPartVectors = Rows<E, I>::kPartVectors;
template<class E, class I> constexpr uint32_t kPartWidth = Rows<E, I>::kPartWidth;
template<class E, class I> constexpr uint64_t kLlPlane = Rows<E, I>::kLlPlane;
template<class E, class I> constexpr uint64_t kOneShotStride = Rows<E, I>::kOneShotStride;

// The LL entries, flag tags, cluster barrier and waits are push.cuh's
// (comm.cuh includes it), so a fix to that protocol reaches this entry.
using push::cluster_sync;
using push::ll_flag;
using push::ll_receive;
using push::ll_send;
using push::tagged;
using push::wait_system;

template<class E, class Input>
struct Call {
  const typename Input::Element* input;
  const E& epilogue;
  uint32_t* epochs;
  uint32_t live;
  uint32_t rank;
  uint32_t rows;  // prepared rows
};

// The v2 LL path's row of F32 sums: each CTA writes its quarter into its
// own copy of the row and bulk-copies it (cp.async.bulk, one 2560-byte copy
// per peer) into every other CTA of the row's cluster, completing on the
// receiver's mbarrier. Two release/acquire cluster barriers around DSMEM
// loads cost about 3 us at one row: a release barrier first waits for this
// rank's LL stores to its peers. The mbarrier is initialized before the PDL
// wait (prepare_bulk_row); a relaxed cluster arrive there is waited for
// before the first copy. After its mbarrier completes, a CTA arrives again
// (relaxed) and waits for that arrival of every CTA only before it exits, so
// no copy still reads a CTA's memory when it leaves.
template<class E>
struct BulkRow {
  alignas(16) float sums[E::kWidth];
  alignas(8) uint64_t bar;
};

template<class E>
__device__ __forceinline__ BulkRow<E>& bulk_row() {
  __shared__ BulkRow<E> state;
  return state;
}

template<class E>
__device__ __forceinline__ void prepare_bulk_row() {
  if (threadIdx.x == 0) {
    ptx::mbarrier_init(&bulk_row<E>().bar, 1);
    ptx::fence_mbarrier_init();
  }
  ptx::cluster_arrive_relaxed();
}

__device__ __forceinline__ void bulk_copy_to_cluster(uint32_t remote, const void* local,
                                                     uint32_t bytes, uint32_t remote_bar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes "
               "[%0], [%1], %2, [%3];"
               :: "r"(remote), "r"(ptx::shared_address(local)), "r"(bytes), "r"(remote_bar)
               : "memory");
}

template<uint32_t Ranks, class E, class Input, class Inboxes>
__device__ void low_latency(const Call<E, Input>& c, Inboxes inboxes,
                            const uint16_t (&weight)[E::kPer]) {
  using R = Rows<E, Input>;
  constexpr uint32_t kStreams = R::kStreams, kVectors = R::kVectors, kParts = R::kParts;
  constexpr uint32_t kPartVectors = R::kPartVectors, kPartWidth = R::kPartWidth;
  constexpr uint32_t kPer = R::kPer, kPerVector = R::kPerVector;
  constexpr uint64_t kLlPlane = R::kLlPlane;
  static_assert(R::kLlFits, "LL layout");
  __shared__ Vector128 gathered[Ranks][kPartVectors];
  __shared__ float sums[8];
  const uint32_t row = blockIdx.x / kParts;
  const uint32_t part = blockIdx.x % kParts;
  const uint64_t group = uint64_t(row) * kStreams + part;
  const uint32_t epoch = c.epochs[row] + 1;
  const uint32_t flag = ll_flag(epoch);
  const uint64_t parity = (epoch & 1) * Ranks;
  const uint32_t vector = part * kPartVectors + threadIdx.x;
  const bool active = threadIdx.x < kPartVectors;
  auto slot = [&](uint32_t owner, uint32_t source) {
    return reinterpret_cast<uint64_t*>(peer(inboxes, owner)) + (parity + source) * kLlPlane +
           (uint64_t(row) * kVectors + vector) * 4;
  };
  // The residual stream and its inject scale do not depend on peers.
  uint16_t residual[kPer];
  c.epilogue.operand(group, residual);
  const auto scale = c.epilogue.scale(group);
  // The prepared rows' inputs exist whatever the live count. Every CTA of
  // a row's cluster sees the same live count, so whole clusters leave
  // together.
  Vector128 own{};
  if (active) own = reinterpret_cast<const Vector128*>(c.input)[uint64_t(row) * kVectors + vector];
  if (c.live > c.rows) asm volatile("trap;");
  if (row >= c.live) return;
  if (active) {
    gathered[c.rank][threadIdx.x] = own;
    uint32_t step = 0;
    for (uint32_t span = 1; span < Ranks; span <<= 1, ++step) {
      // Test-only lag (comm.cuh); serving cubins compile it away.
      test_skew(c.rank, Ranks, epoch);
      const uint32_t partner = c.rank ^ span;
      const uint32_t mine = c.rank & ~(span - 1);
      const uint32_t theirs = partner & ~(span - 1);
      for (uint32_t i = 0; i < span; ++i)
        ll_send(slot(partner, mine + i), gathered[mine + i][threadIdx.x], flag);
      for (uint32_t i = 0; i < span; ++i)
        gathered[theirs + i][threadIdx.x] = ll_receive(slot(c.rank, theirs + i), flag);
    }
    float y[kPerVector] = {};
    for (uint32_t source = 0; source < Ranks; ++source)
      Input::accumulate(gathered[source][threadIdx.x], y);
    float* mine = &bulk_row<E>().sums[part * kPartWidth + threadIdx.x * kPerVector];
#pragma unroll
    for (int k = 0; k < kPerVector; ++k) mine[k] = y[k];
    // The bulk copies read shared memory through the async proxy.
    ptx::fence_async_shared();
  }
  float y[kPer];
  BulkRow<E>& state = bulk_row<E>();
  __syncthreads();
  // Every CTA's mbarrier is initialized (prepare_bulk_row).
  ptx::cluster_wait_acquire();
  if (threadIdx.x == 0) {
    ptx::mbarrier_arrive_expect_tx(&state.bar, (kParts - 1) * kPartWidth * sizeof(float));
    const float* quarter = &state.sums[part * kPartWidth];
#pragma unroll
    for (uint32_t cta = 0; cta < kParts; ++cta)
      if (cta != part)
        bulk_copy_to_cluster(ptx::cluster_map_shared(quarter, cta), quarter,
                             kPartWidth * sizeof(float),
                             ptx::cluster_map_shared(&state.bar, cta));
  }
  // Every CTA read the epoch before it copied its quarter, so CTA 0 may
  // advance it once its row is complete.
  ptx::mbarrier_wait(&state.bar, 0);
  ptx::cluster_arrive_relaxed();
#pragma unroll
  for (uint32_t j = 0; j < kPer; ++j) y[j] = state.sums[threadIdx.x + j * kThreads];
  c.epilogue.finish(y, residual, scale, group, weight, sums);
  if (part == 0 && threadIdx.x == 0) c.epochs[row] = epoch;
  // No CTA leaves while a peer's copy may still read its quarter.
  ptx::cluster_wait_acquire();
}

// Read without clearing: every stream CTA consumes the same inbox entries.
// Keep the single-copy atomic loads and flag checks of push::ll_receive.
__device__ __forceinline__ Vector128 ll_read(const uint64_t* slot, uint32_t flag) {
  uint64_t a, b, c, d;
  do {
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(a), "=l"(b) : "l"(slot) : "memory");
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(c), "=l"(d) : "l"(slot + 2) : "memory");
  } while (static_cast<uint32_t>(a >> 32) != flag || static_cast<uint32_t>(b >> 32) != flag ||
           static_cast<uint32_t>(c >> 32) != flag || static_cast<uint32_t>(d >> 32) != flag);
  return {static_cast<uint32_t>(a), static_cast<uint32_t>(b),
          static_cast<uint32_t>(c), static_cast<uint32_t>(d)};
}

// v3 sends the same quarter as v2, but each CTA reads the whole row from
// its local inbox. No F32 quarter copies or receiver mbarriers remain.
template<uint32_t Ranks, class E, class Input, class Inboxes>
__device__ void low_latency_full_row(const Call<E, Input>& c, Inboxes inboxes,
                                     const uint16_t (&weight)[E::kPer]) {
  using R = Rows<E, Input>;
  // The local source stays in input; excluding it keeps eight ranks below
  // the static shared-memory limit as well as saving redundant stores.
  __shared__ Vector128 gathered[Ranks - 1][R::kVectors];
  __shared__ float reduced[R::kWidth];
  __shared__ float sums[8];
  const uint32_t row = blockIdx.x / R::kParts;
  const uint32_t part = blockIdx.x % R::kParts;
  const uint64_t group = uint64_t(row) * R::kStreams + part;
  const uint32_t epoch = c.epochs[row] + 1;
  const uint32_t flag = ll_flag(epoch);
  const uint64_t parity = (epoch & 1) * Ranks;
  // Rotate the row so the first kPartVectors threads own this CTA's
  // quarter. A vector keeps its thread at every hop, including forwarding.
  auto vector = [&](uint32_t v) { return (part * R::kPartVectors + v) % R::kVectors; };
  auto source_index = [&](uint32_t source) { return source < c.rank ? source : source - 1; };
  const auto* input = reinterpret_cast<const Vector128*>(c.input) + uint64_t(row) * R::kVectors;
  auto slot = [&](uint32_t owner, uint32_t source, uint32_t v) {
    return reinterpret_cast<uint64_t*>(peer(inboxes, owner)) + (parity + source) * R::kLlPlane +
           (uint64_t(row) * R::kVectors + vector(v)) * 4;
  };
  uint16_t residual[R::kPer];
  c.epilogue.operand(group, residual);
  const auto scale = c.epilogue.scale(group);
  Vector128 own{};
  if (threadIdx.x < R::kPartVectors) own = input[vector(threadIdx.x)];
  if (c.live > c.rows) asm volatile("trap;");
  if (row >= c.live) return;
  uint32_t step = 0;
  for (uint32_t span = 1; span < Ranks; span <<= 1, ++step) {
    test_skew(c.rank, Ranks, epoch);
    const uint32_t partner = c.rank ^ span;
    const uint32_t mine = c.rank & ~(span - 1);
    const uint32_t theirs = partner & ~(span - 1);
    if (threadIdx.x < R::kPartVectors)
      for (uint32_t i = 0; i < span; ++i) {
        const uint32_t source = mine + i;
        ll_send(slot(partner, source, threadIdx.x),
                source == c.rank ? own : gathered[source_index(source)][threadIdx.x], flag);
      }
    for (uint32_t v = threadIdx.x; v < R::kVectors; v += kThreads)
      for (uint32_t i = 0; i < span; ++i)
        gathered[source_index(theirs + i)][v] = ll_read(slot(c.rank, theirs + i, v), flag);
  }
  for (uint32_t v = threadIdx.x; v < R::kVectors; v += kThreads) {
    float y[R::kPerVector] = {};
    for (uint32_t source = 0; source < Ranks; ++source)
      Input::accumulate(source == c.rank ? input[vector(v)] : gathered[source_index(source)][v], y);
#pragma unroll
    for (uint32_t k = 0; k < R::kPerVector; ++k)
      reduced[vector(v) * R::kPerVector + k] = y[k];
  }
  // The sums consume every loaded value. After this CTA barrier all its
  // inbox reads are complete. Overlap the cluster rendezvous with finish;
  // it orders clearing, without a release fence on the outgoing LL stores.
  __syncthreads();
  ptx::cluster_arrive_relaxed();
  float y[R::kPer];
#pragma unroll
  for (uint32_t j = 0; j < R::kPer; ++j) y[j] = reduced[threadIdx.x + j * kThreads];
  // Share v2's exact finish path: a separate inject loop can let nvcc
  // contract its multiply/add and change the F32 values used by norm.
  c.epilogue.finish(y, residual, scale, group, weight, sums);
  ptx::cluster_wait_acquire();
  // All stream CTAs have read every source. Only the quarter owner clears
  // each entry, exactly once. A peer's next write to this parity is epoch
  // e+2: finishing e+1 requires this rank's e+1 input, which cannot be sent
  // until this call (and its clears) completes. Zero live rows leave the
  // per-row epoch alone. Clearing also prevents stale 12-bit epoch matches.
  if (threadIdx.x < R::kPartVectors) {
    for (uint32_t source = 0; source < Ranks; ++source) {
      if (source == c.rank) continue;
      uint64_t* entry = slot(c.rank, source, threadIdx.x);
      asm volatile("st.global.v2.b64 [%0], {%1,%1};" :: "l"(entry), "l"(0ull) : "memory");
      asm volatile("st.global.v2.b64 [%0], {%1,%1};" :: "l"(entry + 2), "l"(0ull) : "memory");
    }
  }
  if (part == 0 && threadIdx.x == 0) c.epochs[row] = epoch;
}

template<uint32_t Ranks, class E, class Input, class Inboxes, class Flags>
__device__ void one_shot(const Call<E, Input>& c, Inboxes inboxes, Flags flags,
                         const uint16_t (&weight)[E::kPer]) {
  using R = Rows<E, Input>;
  using Element = typename Input::Element;
  constexpr uint32_t kWidth = R::kWidth, kStreams = R::kStreams, kVectors = R::kVectors;
  constexpr uint32_t kPer = R::kPer;
  constexpr uint64_t kLlPlane = R::kLlPlane, kOneShotStride = R::kOneShotStride;
  const uint32_t epoch = c.epochs[kOneShotEpoch] + 1;
  const uint32_t mark = tagged(epoch);
  const uint32_t parity = epoch & 1;
  const uint64_t box = uint64_t(parity) * Ranks * kOneShotStride;
  auto inbox = [&](uint32_t owner) {
    return reinterpret_cast<Element*>(reinterpret_cast<char*>(peer(inboxes, owner)) +
                                      (2ull * Ranks * kLlPlane * 8)) + box;
  };
  unsigned int* local = peer(flags, c.rank);
  if (blockIdx.x < kSenders) {
    const uint64_t vectors = uint64_t(c.live) * kVectors;
    const auto* src = reinterpret_cast<const Vector128*>(c.input);
    // The payload stays in registers across destinations, so no phase
    // after the first waits for loads.
    constexpr uint32_t kSendPer = (kMaxRows * kVectors + kSenders * kThreads - 1) /
                                  (kSenders * kThreads);
    const uint64_t first = blockIdx.x * kThreads + threadIdx.x;
    Vector128 payload[kSendPer];
#pragma unroll
    for (uint32_t i = 0; i < kSendPer; ++i)
      if (first + i * kSenders * kThreads < vectors) payload[i] = src[first + i * kSenders * kThreads];
    for (uint32_t step = 1; step < Ranks; ++step) {
      const uint32_t dest = (c.rank + step) % Ranks;
      test_skew(c.rank, Ranks, epoch);
      auto* target = reinterpret_cast<Vector128*>(inbox(dest) + c.rank * kOneShotStride);
#pragma unroll
      for (uint32_t i = 0; i < kSendPer; ++i)
        if (first + i * kSenders * kThreads < vectors)
          target[first + i * kSenders * kThreads] = payload[i];
      // Every CTA of the cluster has issued its writes to this peer before
      // any writes to the next; the release below is cumulative through the
      // barrier.
      __syncthreads();
      cluster_sync();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        release_system(peer(flags, dest) + parity * Ranks + c.rank, mark);
      }
    }
    // Acknowledge every call, even with no live row: this rank's next call
    // may rewrite a peer's bank of this parity only after that peer has
    // published this call, which it does after finishing the previous one.
    if (blockIdx.x == 0) {
      if (threadIdx.x < Ranks && threadIdx.x != c.rank)
        wait_system(local + parity * Ranks + threadIdx.x, mark);
      __syncthreads();
    }
  } else {
    __shared__ float sums[8];
    const uint64_t group = blockIdx.x - kSenders;
    // Padding CTAs left at entry; the check stays as a guard.
    if constexpr (R::kPadsGroups) {
      if (group >= uint64_t(c.rows) * kStreams) return;
    }
    const uint32_t row = group / kStreams;
    // The residual stream and its inject scale load before the live check.
    uint16_t residual[kPer];
    c.epilogue.operand(group, residual);
    const auto scale = c.epilogue.scale(group);
    if (row >= c.live) return;
    test_skew(c.rank, Ranks, epoch);
    if (threadIdx.x < Ranks && threadIdx.x != c.rank) {
      wait_system(local + parity * Ranks + threadIdx.x, mark);
    }
    __syncthreads();
    const Element* received = inbox(c.rank);
    float y[kPer];
#pragma unroll
    for (uint32_t j = 0; j < kPer; ++j) {
      const uint64_t i = uint64_t(row) * kWidth + threadIdx.x + j * kThreads;
      y[j] = 0.f;
#pragma unroll
      for (uint32_t source = 0; source < Ranks; ++source)
        y[j] += Input::decode(source == c.rank ? c.input[i]
                                               : received[source * kOneShotStride + i]);
    }
    c.epilogue.finish(y, residual, scale, group, weight, sums);
  }
  // The sending cluster (counted once, by CTA 0) and every live group CTA
  // arrive; the last clears the counter and advances the epoch after all of
  // them read it.
  __syncthreads();
  if (threadIdx.x != 0 || (blockIdx.x < kSenders && blockIdx.x != 0)) return;
  __threadfence();
  if (atomicAdd(c.epochs + kCounter, 1u) == c.live * kStreams) {
    c.epochs[kCounter] = 0;
    c.epochs[kOneShotEpoch] = epoch;
  }
}

// The entry for `Ranks` ranks (chop_comm_qwen38_allreduce_hc_decode_r<Ranks>_v2,
// and _v3 with FullRow), epilogue E and input Input: its pointer arrays hold
// `Ranks` slots and every loop over ranks is constant, so the four-rank entry
// compiles as the TP4 entry did. `ranks` must equal `Ranks`; any other value
// traps before a peer is touched. It reads the stream's norm weights before
// the PDL wait and issues a row's local loads before it checks the live
// count, so no load waits for another.
template<class E, class Input, uint32_t Ranks, bool FullRow = false, class Inboxes, class Flags>
__device__ void allreduce_hc(const typename Input::Element* input, const E& epilogue,
                             Inboxes inboxes, Flags flags, uint32_t* epochs,
                             const uint32_t* live_rows, uint32_t rows, uint32_t rank,
                             uint32_t ranks) {
  using R = Rows<E, Input>;
  constexpr uint32_t kParts = R::kParts, kPer = R::kPer;
  // Wider rows and F32 transport extensions can exceed the static shared
  // limit with a whole-row gather. They retain the v2 quarter exchange.
  constexpr bool full_row = FullRow &&
      (Ranks - 1) * R::kVectors * sizeof(Vector128) + R::kWidth * sizeof(float) + 64 <= 48 * 1024;
  static_assert(qwen38::valid_ranks(0, Ranks), "hyper-connection entries take 2, 4 or 8 ranks");
  static_assert(sizeof(Inboxes) == Ranks * sizeof(inboxes.slot_0) &&
                sizeof(Flags) == Ranks * sizeof(flags.slot_0), "one slot per rank");
  static_assert(sizeof(inboxes.slot_0[0]) == sizeof(typename Input::Element),
                "inboxes of the input's elements");
  // Padding CTAs (kPadsGroups) hold no group. They leave from the by-value
  // rows and their block index alone, before tracing, parameter loads, the
  // dependency wait and the live and epoch reads, after the launch-dependents
  // trigger every CTA gives.
  if constexpr (R::kPadsGroups) {
    if (R::padding_cta(blockIdx.x, rows)) {
      ptx::griddep_launch_dependents();
      return;
    }
  }
  ptx::griddep_launch_dependents();
  const bool ll = rows <= CHOP_QWEN38_LL_MAX_ROWS;
  if constexpr (!full_row) {
    if (ll) prepare_bulk_row<E>();
  }
  // The norm weights of this CTA's stream are model constants.
  uint16_t norm_weight[kPer] = {};
  if (ll || blockIdx.x >= kSenders) {
    const uint32_t stream = (ll ? blockIdx.x : blockIdx.x - kSenders) % R::kStreams;
    epilogue.parameters(stream, norm_weight);
  }
  ptx::griddep_wait();
  const uint32_t live = *live_rows;
  // Ranks is a power of two, which recursive doubling needs. LL calls check
  // the live count after their loads (low_latency).
  if (ranks != Ranks || rank >= Ranks || rows == 0 || rows > kMaxRows ||
      (!ll && live > rows))
    asm volatile("trap;");
  // The grid and cluster must be the plan's for these rows: a mismatch
  // would skip rows or arrivals and never finish.
  if (gridDim.x != (ll ? rows * kParts : kSenders + R::group_ctas(rows)) || gridDim.y != 1 ||
      gridDim.z != 1 || blockDim.x != kThreads ||
      ptx::cluster_nctarank() != (ll ? kParts : kSenders))
    asm volatile("trap;");
  const Call<E, Input> c{input, epilogue, epochs, live, rank, rows};
  if constexpr (R::kLlFits) {
    if (ll) {
      if constexpr (full_row) low_latency_full_row<Ranks>(c, inboxes, norm_weight);
      else low_latency<Ranks>(c, inboxes, norm_weight);
    }
    else one_shot<Ranks>(c, inboxes, flags, norm_weight);
  } else {
    one_shot<Ranks>(c, inboxes, flags, norm_weight);
  }
}
} // namespace chop::comm::qwen38_decode
