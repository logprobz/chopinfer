#pragma once

// PCIe push all-reduce of BF16 rows whose fused epilogue is a functor the
// entry supplies (comm/epilogue/): a residual add and RMSNorm epilogue
// backs chop_comm_allreduce_oneshot_norm_bf16_v2 (up to 32 rows) and
// chop_comm_allreduce_wide_norm_bf16_v1 (the wide two-shot, below). The
// row width is the functor's (E::kWidth, see Rows); the transport owns
// protocols, flags, row geometry and the order in which a row's statistic
// is summed, the functor the arithmetic around them.
//
// Measured on four RTX PRO 6000 GPUs behind separate root ports: a GPU that
// writes (or reads) to two or more peers at the same time collapses to about
// 2 GB/s in total, while one destination per GPU runs at about 47 GB/s in any
// pairing and several senders into one receiver stay fast. Every path below
// therefore sends to one destination at a time: the sending cluster
// synchronizes before it switches destination.
//
// Protocols, chosen by rows (tokens) and ranks exactly as the Rust planner,
// from per-rank-count thresholds in tuning/comm.json:
//   low latency   rows <= the rank count's LL rows (power-of-two ranks only):
//                 recursive doubling of raw inputs in 8-byte LL entries (two
//                 BF16 values and a flag word), no fences or flags, one
//                 cluster per row.
//   one-shot      rows <= its one-shot rows: push the full input to each peer
//                 in turn, then reduce.
//   two-shot      above: push each owner its columns, reduce, push the sums
//                 back (power-of-two ranks of at least
//                 CHOP_COMM_TWO_SHOT_MIN_RANKS).
// The wide entries run a two-shot for up to CHOP_COMM_WIDE_MAX_ROWS rows at
// power-of-two ranks, whose last phases flag blocks of rows; graphs send it
// the rows above 32 up to the rank count's wide rows in tuning/comm.json.
// Every rank reduces all inputs in rank order in F32 and applies the
// epilogue in the same order, so outputs are bitwise identical.

#include <type_traits>
#include "../common/ptx_control.cuh"
#include "input/bf16.cuh"

#if !defined(CHOP_COMM_LL_MAX_ROWS) || !defined(CHOP_COMM_LL_CLUSTER) || \
    !defined(CHOP_COMM_PUSH_CLUSTER) || !defined(CHOP_COMM_LL_ROWS_BY_RANKS) || \
    !defined(CHOP_COMM_ONESHOT_ROWS_BY_RANKS) || !defined(CHOP_COMM_TWO_SHOT_MIN_RANKS) || \
    !defined(CHOP_COMM_ONE_SHOT_BASE) || !defined(CHOP_COMM_TWO_SHOT_BASE) || \
    !defined(CHOP_COMM_PHASED_BASE) || !defined(CHOP_COMM_TWO_SHOT_FLAGS) || \
    !defined(CHOP_COMM_WIDE_MAX_ROWS) || !defined(CHOP_COMM_WIDE_BLOCK_ROWS) || \
    !defined(CHOP_COMM_WIDE_GATHER_BASE) || !defined(CHOP_COMM_WIDE_SCATTER_BASE) || \
    !defined(CHOP_COMM_WIDE_FLAGS)
#error "comm push tuning defines are required"
#endif

namespace chop::comm::push {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kLlCluster = CHOP_COMM_LL_CLUSTER;
constexpr uint32_t kCluster = CHOP_COMM_PUSH_CLUSTER;
static_assert(kLlCluster == 2 || kLlCluster == 4, "one LL vector per thread");
static_assert(kCluster >= 1 && kCluster <= 8, "portable cluster size");

// Each protocol owns a byte region of the inbox, so a peer that already runs
// the next call with another protocol never overwrites data still being read.
// Strides use the largest row count of the protocol, not the current one.
using comm::kMaxRanks;
constexpr uint32_t kBulkMaxRows = 32;
constexpr uint64_t kOneShotBase = CHOP_COMM_ONE_SHOT_BASE;
constexpr uint64_t kTwoShotBase = CHOP_COMM_TWO_SHOT_BASE;
constexpr uint64_t kPhasedBase = CHOP_COMM_PHASED_BASE;

template <typename T, typename Inboxes>
__device__ __forceinline__ T* region(Inboxes inboxes, uint32_t owner, uint64_t base) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(peer(inboxes, owner)) + base);
}

enum class Protocol : uint32_t { LowLatency, OneShot, TwoShot };

// A threshold per rank count, packed one byte per count from ranks 1 in the
// low byte (the planner packs them), for 1 <= ranks <= kMaxRanks.
static_assert(kMaxRanks <= 8, "one byte per rank count in 64 bits");
__device__ __forceinline__ uint32_t by_ranks(uint64_t packed, uint32_t ranks) {
  return static_cast<uint32_t>(packed >> (8 * (ranks - 1))) & 0xffu;
}

// The planner checked the table: low latency only at power-of-two ranks and
// two-shot only at power-of-two ranks of at least kTwoShotMinRanks.
__device__ __forceinline__ Protocol protocol(uint32_t rows, uint32_t ranks) {
  if (rows <= by_ranks(CHOP_COMM_LL_ROWS_BY_RANKS, ranks)) return Protocol::LowLatency;
  if (rows <= by_ranks(CHOP_COMM_ONESHOT_ROWS_BY_RANKS, ranks)) return Protocol::OneShot;
  return Protocol::TwoShot;
}

// Peers write flag words of one protocol only: one-shot [0, 2 ranks),
// two-shot [kTwoShotFlags, + 2 ranks), phased from word 32 (below), wide
// from CHOP_COMM_WIDE_FLAGS (kWideScatterFlags and after). A peer
// may start the next call, with any protocol, while this rank still waits
// on a flag of the previous one, so no call may rewrite a word that another
// protocol's previous call still reads. Local ready and counter words are
// written only by this rank's own calls, which its stream orders.
constexpr uint32_t kTwoShotFlags = CHOP_COMM_TWO_SHOT_FLAGS;
// The fewest ranks the tuning sends through the two-shot; it sizes kPer.
constexpr uint32_t kTwoShotMinRanks = CHOP_COMM_TWO_SHOT_MIN_RANKS;
static_assert(kTwoShotMinRanks >= 2 && kTwoShotMinRanks <= kMaxRanks, "two-shot ranks");

// Row geometry for the epilogue's rows of E::kWidth BF16 columns. A width
// whose rows do not split into whole vectors per row thread and LL CTA, or
// whose rows overflow a protocol's inbox region, fails to compile.
template <typename E>
struct Rows {
  static constexpr uint32_t kVectors = E::kWidth / 8;           // 16-byte vectors per row
  static constexpr uint32_t kPer = kVectors / kThreads;          // vectors per row thread
  static constexpr uint32_t kLlVectors = kVectors / kLlCluster;  // vectors per LL CTA
  static constexpr uint32_t kLlWords = kVectors * 4;             // 64-bit LL words per row
  static_assert(E::kWidth % (8 * kThreads) == 0 && kLlVectors <= kThreads,
                "whole vectors per row thread and one LL vector per thread");
  static_assert(kVectors % kTwoShotMinRanks == 0, "two-shot column shares");
  static_assert(2ull * kMaxRanks * CHOP_COMM_LL_MAX_ROWS * kLlWords * 8 <= kOneShotBase,
                "LL region");
  static_assert(kOneShotBase + 2ull * kMaxRanks * kBulkMaxRows * kVectors * 16 <= kTwoShotBase,
                "one-shot region");
  static_assert(kTwoShotBase + 2ull * kBulkMaxRows * kVectors * 16 <= kPhasedBase,
                "two-shot region");
};
template <typename E> constexpr uint32_t kRowVectors = Rows<E>::kVectors;
template <typename E> constexpr uint32_t kRowPer = Rows<E>::kPer;
template <typename E> constexpr uint32_t kLlVectors = Rows<E>::kLlVectors;
template <typename E> constexpr uint32_t kLlRowWords = Rows<E>::kLlWords;
static_assert(kTwoShotFlags >= 82, "two-shot flags follow the phased words");

// Flags written by this path set bit 31, so values left by other protocols
// sharing the buffers never match. Every protocol advances epoch[0], so a
// word whose protocol has not run for 2^31 calls could match a later mark;
// a state that may stay idle in one protocol that long must be reset
// (flags and epochs zeroed on every rank) first.
__device__ __forceinline__ uint32_t tagged(uint32_t epoch) { return 0x80000000u | epoch; }

// Both BF16 halves of an LL flag are NaN patterns, so activations never look
// like a valid entry. Other bytes can: the LL region must start zeroed
// (comm::PushBuffers::zeroed_inbox_bytes), after which it holds only zeros
// and entries of calls in flight, because receivers clear what they read.
__device__ __forceinline__ uint32_t ll_flag(uint32_t epoch) {
  return 0xffc0ffc0u | (epoch & 0x3fu) | (((epoch >> 6) & 0x3fu) << 16);
}

// Each 64-bit element is single-copy atomic, so a matching flag word proves
// its two data values arrived.
__device__ __forceinline__ void ll_store(uint64_t* address, uint32_t first,
                                         uint32_t second, uint32_t flag) {
  const uint64_t x = (static_cast<uint64_t>(flag) << 32) | first;
  const uint64_t y = (static_cast<uint64_t>(flag) << 32) | second;
  asm volatile("st.volatile.global.v2.b64 [%0], {%1,%2};"
               :: "l"(address), "l"(x), "l"(y) : "memory");
}

__device__ __forceinline__ void ll_send(uint64_t* slot, Vector128 value, uint32_t flag) {
  ll_store(slot, value.x, value.y, flag);
  ll_store(slot + 2, value.z, value.w, flag);
}

// Wait for four entries, then clear them so the slot never matches a later
// flag. The next write to this slot comes two epochs later.
__device__ __forceinline__ Vector128 ll_receive(uint64_t* slot, uint32_t flag) {
  uint64_t a, b, c, d;
  do {
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(a), "=l"(b) : "l"(slot) : "memory");
    asm volatile("ld.volatile.global.v2.b64 {%0,%1}, [%2];"
                 : "=l"(c), "=l"(d) : "l"(slot + 2) : "memory");
  } while (static_cast<uint32_t>(a >> 32) != flag || static_cast<uint32_t>(b >> 32) != flag ||
           static_cast<uint32_t>(c >> 32) != flag || static_cast<uint32_t>(d >> 32) != flag);
  asm volatile("st.global.v2.b64 [%0], {%1,%1};" :: "l"(slot), "l"(0ull) : "memory");
  asm volatile("st.global.v2.b64 [%0], {%1,%1};" :: "l"(slot + 2), "l"(0ull) : "memory");
  return {static_cast<uint32_t>(a), static_cast<uint32_t>(b),
          static_cast<uint32_t>(c), static_cast<uint32_t>(d)};
}

__device__ __forceinline__ void accumulate(Vector128 value, float (&sums)[8]) {
  const uint32_t words[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    sums[2 * word] += decode(static_cast<uint16_t>(words[word]));
    sums[2 * word + 1] += decode(static_cast<uint16_t>(words[word] >> 16));
  }
}

__device__ __forceinline__ float cluster_load(const float* local, uint32_t cta) {
  float value;
  asm volatile("ld.shared::cluster.f32 %0, [%1];"
               : "=f"(value) : "r"(ptx::cluster_map_shared(local, cta)) : "memory");
  return value;
}

__device__ __forceinline__ void cluster_sync() {
  ptx::cluster_arrive_release();
  ptx::cluster_wait_acquire();
}

// Row sum over Ctas CTAs of one cluster. The order depends only on thread and
// CTA indices, never on the rank.
template <uint32_t Ctas>
__device__ __forceinline__ float row_sum(float value) {
  __shared__ float warp_sums[kThreads / 32];
  __shared__ float cta_sum;
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    float warp = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      warp += __shfl_xor_sync(0xffffffffu, warp, offset);
    if (threadIdx.x == 0) cta_sum = warp;
  }
  if constexpr (Ctas == 1) {
    __syncthreads();
    const float total = cta_sum;
    __syncthreads();
    return total;
  } else {
    cluster_sync();
    float total = 0.0f;
    for (uint32_t cta = 0; cta < Ctas; ++cta) total += cluster_load(&cta_sum, cta);
    cluster_sync();
    return total;
  }
}

// row_sum<Ctas> without cluster barriers, for the decode entries' LL path:
// a release barrier would first wait for this rank's LL stores to its peers
// to complete. Thread 0 of each CTA pushes its CTA sum into slot [cta] of
// every CTA with st.async, which completes on that CTA's mbarrier; each CTA
// waits for all slots and adds them in CTA order as row_sum does, so the
// bits are the same. prepare_async_sum() runs at kernel start, before the
// PDL wait; row_sum_async waits for its cluster arrive, by which time every
// CTA has long arrived. The receiver waits for every slot before it exits,
// so no CTA's shared memory is written after it leaves.
template <uint32_t Ctas>
struct AsyncSum {
#ifdef CHOP_COMM_TEST_FLAT_SUM
  float slots[Ctas * (kThreads / 32)];
#else
  float slots[Ctas];
#endif
  alignas(8) uint64_t bar;
};

template <uint32_t Ctas>
__device__ __forceinline__ AsyncSum<Ctas>& async_sum() {
  __shared__ AsyncSum<Ctas> state;
  return state;
}

template <uint32_t Ctas>
__device__ __forceinline__ void prepare_async_sum() {
  if (threadIdx.x == 0) {
    ptx::mbarrier_init(&async_sum<Ctas>().bar, 1);
    ptx::fence_mbarrier_init();
  }
  ptx::cluster_arrive_relaxed();
}

template <uint32_t Ctas>
__device__ __forceinline__ float row_sum_async(float value) {
  __shared__ float warp_sums[kThreads / 32];
  AsyncSum<Ctas>& state = async_sum<Ctas>();
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = value;
  __syncthreads();
  // Every CTA has initialized its mbarrier (prepare_async_sum).
  ptx::cluster_wait_acquire();
#ifdef CHOP_COMM_TEST_FLAT_SUM
  // Lab negative control: exchange warp sums before either CTA reduces
  // them, then flatten both CTA trees into one. Transport is unchanged.
  static_assert(Ctas * (kThreads / 32) <= 32, "one warp holds the wrong tree");
  if (threadIdx.x == 0) {
    ptx::mbarrier_arrive_expect_tx(&state.bar, Ctas * (kThreads / 32) * sizeof(float));
    const uint32_t me = ptx::cluster_ctarank();
    for (uint32_t cta = 0; cta < Ctas; ++cta)
      for (uint32_t warp = 0; warp < kThreads / 32; ++warp)
        ptx::cluster_st_async_f32(
            ptx::cluster_map_shared(&state.slots[me * (kThreads / 32) + warp], cta),
            warp_sums[warp], ptx::cluster_map_shared(&state.bar, cta));
  }
  ptx::mbarrier_wait(&state.bar, 0);
  const uint32_t lane = threadIdx.x & 31;
  float total = lane < Ctas * (kThreads / 32) ? state.slots[lane] : 0.0f;
  for (int offset = 16; offset > 0; offset >>= 1)
    total += __shfl_xor_sync(0xffffffffu, total, offset);
  return total;
#else
  if (threadIdx.x < 32) {
    float warp = threadIdx.x < kThreads / 32 ? warp_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      warp += __shfl_xor_sync(0xffffffffu, warp, offset);
    if (threadIdx.x == 0) {
      ptx::mbarrier_arrive_expect_tx(&state.bar, Ctas * sizeof(float));
      const uint32_t me = ptx::cluster_ctarank();
      for (uint32_t cta = 0; cta < Ctas; ++cta)
        ptx::cluster_st_async_f32(ptx::cluster_map_shared(&state.slots[me], cta), warp,
                                  ptx::cluster_map_shared(&state.bar, cta));
    }
  }
  ptx::mbarrier_wait(&state.bar, 0);
  float total = 0.0f;
  for (uint32_t cta = 0; cta < Ctas; ++cta) total += state.slots[cta];
  return total;
#endif
}

template <typename E>
struct Call {
  const uint16_t* input;
  const E& epilogue;
  unsigned int* epoch;
  uint32_t rows, rank, ranks;
  uint32_t live;  // rows below this carry input; a staged policy loads zeros for later rows
};

// Input vector `index` (row * row vectors + vector) through the input
// policy In (input/bf16.cuh: the vector as it is; input/routed_combine.cuh:
// a routed-row MoE combine). Eager lets a policy issue a dead row's loads
// before it knows the live count.
template <typename In, bool Eager = false, typename E>
__device__ __forceinline__ Vector128 input_vector(const Call<E>& c, uint64_t index) {
  return In::template load<Eager>(c, index);
}

// Recursive doubling: at step h each rank sends every input it holds to
// rank ^ 2^h. A step starts only after the previous partner's data arrived,
// by which time this rank's earlier writes have drained. Entries live at
// [parity][source][row][2048] in 64-bit units. Fast (the decode entries)
// loads a combined row's routed rows without waiting for the live count;
// AsyncNorm sums the row statistic over st.async (row_sum_async).
template <typename In, bool Fast, bool AsyncNorm, typename E, typename Inboxes>
__device__ void low_latency(const Call<E>& c, Inboxes inboxes,
                            const Vector128 (&parameter)[kRowPer<E>]) {
  __shared__ Vector128 gathered[8][kLlVectors<E>];
  const uint32_t row = blockIdx.x / kLlCluster;
  const uint32_t part = blockIdx.x % kLlCluster;
  const uint32_t vector = part * kLlVectors<E> + threadIdx.x;
  const bool active = threadIdx.x < kLlVectors<E>;
  const uint32_t epoch = c.epoch[row] + 1;
  const uint32_t flag = ll_flag(epoch);
  const uint64_t plane = static_cast<uint64_t>(CHOP_COMM_LL_MAX_ROWS) * kLlRowWords<E>;
  const uint64_t parity = (epoch & 1) * c.ranks;
  auto slot = [&](uint32_t owner, uint32_t source) {
    return reinterpret_cast<uint64_t*>(peer(inboxes, owner)) +
           (parity + source) * plane + static_cast<uint64_t>(row) * kLlRowWords<E> + vector * 4;
  };
  const uint64_t index = static_cast<uint64_t>(row) * kRowVectors<E> + vector;
  Vector128 operand{};
  if (active) {
    gathered[c.rank][threadIdx.x] = input_vector<In, Fast>(c, index);
    operand = c.epilogue.operand(index);
  }
  test_skew(c.rank, c.ranks, epoch);
  uint32_t step = 0;
  for (uint32_t span = 1; span < c.ranks; span <<= 1, ++step) {
    const uint32_t partner = c.rank ^ span;
    const uint32_t mine = c.rank & ~(span - 1);
    const uint32_t theirs = partner & ~(span - 1);
    if (active) {
      for (uint32_t i = 0; i < span; ++i)
        ll_send(slot(partner, mine + i), gathered[mine + i][threadIdx.x], flag);
      for (uint32_t i = 0; i < span; ++i)
        gathered[theirs + i][threadIdx.x] = ll_receive(slot(c.rank, theirs + i), flag);
    }
  }
  float square = 0.0f;
  Vector128 value{};
  if (active) {
    float sums[8] = {};
    for (uint32_t source = 0; source < c.ranks; ++source)
      accumulate(gathered[source][threadIdx.x], sums);
    value = E::carry(sums, operand, square);
    c.epilogue.keep(index, value);
  }
  float total;
  if constexpr (AsyncNorm) total = row_sum_async<kLlCluster>(square);
  else total = row_sum<kLlCluster>(square);
  const float inv = c.epilogue.scale(total);
  if (active) c.epilogue.finish(index, value, parameter[0], inv);
  // row_sum synchronized the cluster after every CTA read the epoch; with
  // row_sum_async, CTA 0 has every other CTA's sum, pushed after its read.
  if (part == 0 && threadIdx.x == 0) c.epoch[row] = epoch;
}

// Copies 16-byte vectors with four loads in flight per thread, so one
// cluster keeps a peer link busy.
__device__ __forceinline__ void copy_vectors(Vector128* to, const Vector128* from,
                                             uint64_t vectors, uint32_t worker,
                                             uint32_t workers) {
  constexpr uint32_t kUnroll = 4;
  const uint64_t step = static_cast<uint64_t>(workers) * kThreads;
  uint64_t v = worker * kThreads + threadIdx.x;
  for (; v + (kUnroll - 1) * step < vectors; v += kUnroll * step) {
    Vector128 values[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) values[i] = from[v + i * step];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) to[v + i * step] = values[i];
  }
  for (; v < vectors; v += step) to[v] = from[v];
}

// Every CTA of the cluster has issued its writes to this destination before
// any CTA writes to the next one. The release store of the following flag is
// cumulative through this barrier, so it also orders the data for receivers.
// A system fence in every CTA drains more completely but measured 2 to 3 us
// slower per call at 16 and 32 rows without avoiding any further collapse.
__device__ __forceinline__ void next_destination() {
  __syncthreads();
  cluster_sync();
}

// Spin with relaxed loads, then acquire once: an acquire load per iteration
// costs a fence each time.
__device__ __forceinline__ void wait_system(const unsigned int* address, uint32_t mark) {
  uint32_t value;
  do {
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  } while (value != mark);
  acquire_system(address);
}

__device__ __forceinline__ void wait_gpu(const unsigned int* address, uint32_t mark) {
  uint32_t value;
  do {
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  } while (value != mark);
  acquire_gpu(address);
}

// The sending cluster (counted once, by CTA 0) and every row CTA arrive on a
// local counter; the last arrival clears it for the next call and advances
// the shared epoch, after all of them read it. No other protocol writes the
// counter word and the next call starts only after this kernel completes.
__device__ __forceinline__ void finish(unsigned int* counter, unsigned int* epoch_cell,
                                       uint32_t epoch, uint32_t rows) {
  __syncthreads();
  if (threadIdx.x != 0 || (blockIdx.x < kCluster && blockIdx.x != 0)) return;
  __threadfence();
  if (atomicAdd(counter, 1u) == rows) {
    *counter = 0;
    epoch_cell[0] = epoch;
  }
}

// CTAs [0, kCluster) form the sending cluster; CTA kCluster + r normalizes
// row r and later CTAs exit. One-shot inbox: [parity][source][32 rows * 512]
// vectors. Flags: [parity * ranks + source], combined-rows ready word
// 2 * ranks (staged input), arrival counter 2 * ranks + 1. Fast (the decode
// entries) issues every source's loads of a row before summing them in rank
// order, instead of one source at a time, and from three ranks up to
// kPayloadMaxRows rows keeps the sending cluster's payload in registers.
// At TP4 that saved 0.6 to 1.5 us per call at 3 to 12 rows; at 14 to 16
// rows, where every thread stores four vectors at once, and at two ranks
// (one phase, nothing to reuse) it was slower.
constexpr uint32_t kPayloadMaxRows = 12;
// LlOrder sums a row's squares in low_latency's order: each of the kRowPer
// vector columns in its own CTA-shaped reduction, the column sums then added
// in column order, as the kLlCluster CTAs of a low-latency row are. Every
// other step already matches, so a row's outputs equal its low-latency bits
// at any row count (the verify entries, whose rows must equal an AR row's).
template <typename In, bool Fast, bool LlOrder, typename E, typename Inboxes, typename Flags>
__device__ void one_shot(const Call<E>& c, Inboxes inboxes, Flags flags,
                         const Vector128 (&parameter)[kRowPer<E>]) {
  constexpr uint32_t kRowVectors = push::kRowVectors<E>, kRowPer = push::kRowPer<E>;
  static_assert(!LlOrder || kRowPer == kLlCluster, "a row thread holds one vector per LL CTA");
  if (blockIdx.x >= kCluster + c.rows) return;
  const uint32_t epoch = c.epoch[0] + 1;
  const uint32_t mark = tagged(epoch);
  const uint32_t parity = epoch & 1;
  const uint64_t vectors = static_cast<uint64_t>(c.rows) * kRowVectors;
  unsigned int* local = peer(flags, c.rank);
  const uint64_t stride = static_cast<uint64_t>(kBulkMaxRows) * kRowVectors;
  const uint64_t box = static_cast<uint64_t>(parity) * c.ranks * stride;
  if (Fast && blockIdx.x < kCluster && c.ranks > 2 && c.rows <= kPayloadMaxRows) {
    // The payload stays in registers across destinations, so no phase after
    // the first waits for loads. A staged input is loaded once and also
    // kept in this rank's own inbox slot for the row CTAs (see below).
    constexpr uint32_t kSendPer = (kPayloadMaxRows * kRowVectors + kCluster * kThreads - 1) /
                                  (kCluster * kThreads);
    const uint64_t first = blockIdx.x * kThreads + threadIdx.x;
    Vector128 payload[kSendPer];
#pragma unroll
    for (uint32_t i = 0; i < kSendPer; ++i) {
      const uint64_t v = first + i * kCluster * kThreads;
      if (v < vectors) payload[i] = input_vector<In>(c, v);
    }
    if constexpr (In::kStaged) {
      Vector128* combined = region<Vector128>(inboxes, c.rank, kOneShotBase) + box +
                            static_cast<uint64_t>(c.rank) * stride;
#pragma unroll
      for (uint32_t i = 0; i < kSendPer; ++i) {
        const uint64_t v = first + i * kCluster * kThreads;
        if (v < vectors) combined[v] = payload[i];
      }
    }
    for (uint32_t step = 1; step < c.ranks; ++step) {
      const uint32_t dest = (c.rank + step) % c.ranks;
      Vector128* target = region<Vector128>(inboxes, dest, kOneShotBase) + box +
                          static_cast<uint64_t>(c.rank) * stride;
#pragma unroll
      for (uint32_t i = 0; i < kSendPer; ++i) {
        const uint64_t v = first + i * kCluster * kThreads;
        if (v < vectors) target[v] = payload[i];
      }
      if (In::kStaged && step == 1) __threadfence();
      next_destination();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        release_system(peer(flags, dest) + parity * c.ranks + c.rank, mark);
        if (In::kStaged && step == 1) release_gpu(local + 2 * c.ranks, mark);
      }
    }
  } else if (blockIdx.x < kCluster) {
    for (uint32_t step = 1; step < c.ranks; ++step) {
      const uint32_t dest = (c.rank + step) % c.ranks;
      Vector128* target = region<Vector128>(inboxes, dest, kOneShotBase) + box +
                          static_cast<uint64_t>(c.rank) * stride;
      // A staged input's first phase loads each vector once and keeps it
      // in this rank's own slot of its inbox, which no peer writes. It cannot
      // live in `output`: row CTAs normalize into `output` as soon as their
      // peers' rows arrive, possibly before later phases have resent it.
      Vector128* combined = region<Vector128>(inboxes, c.rank, kOneShotBase) + box +
                            static_cast<uint64_t>(c.rank) * stride;
      if (In::kStaged && step == 1) {
        for (uint64_t v = blockIdx.x * kThreads + threadIdx.x; v < vectors; v += kCluster * kThreads) {
          const Vector128 value = input_vector<In>(c, v);
          combined[v] = value;
          target[v] = value;
        }
        __threadfence();
      } else {
        copy_vectors(target, In::kStaged ? combined : reinterpret_cast<const Vector128*>(c.input),
                     vectors, blockIdx.x, kCluster);
      }
      next_destination();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        release_system(peer(flags, dest) + parity * c.ranks + c.rank, mark);
        if (In::kStaged && step == 1) release_gpu(local + 2 * c.ranks, mark);
      }
    }
  } else {
    const uint32_t row = blockIdx.x - kCluster;
    Vector128 operand[kRowPer];
#pragma unroll
    for (uint32_t i = 0; i < kRowPer; ++i)
      operand[i] = c.epilogue.operand(static_cast<uint64_t>(row) * kRowVectors + i * kThreads +
                                      threadIdx.x);
    // With a staged input and peers, this rank's rows come from the cluster.
    const bool staged = In::kStaged && c.ranks > 1;
    const Vector128* inbox = region<const Vector128>(inboxes, c.rank, kOneShotBase) + box;
    // Without staging this rank's own rows do not depend on peers.
    Vector128 own[kRowPer];
    if (!staged) {
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i)
        own[i] = input_vector<In>(c, static_cast<uint64_t>(row) * kRowVectors +
                                              i * kThreads + threadIdx.x);
    }
    test_skew(c.rank, c.ranks, epoch);
    if (threadIdx.x < c.ranks) {
      if (threadIdx.x != c.rank) wait_system(local + parity * c.ranks + threadIdx.x, mark);
      else if (staged) wait_gpu(local + 2 * c.ranks, mark);
    }
    __syncthreads();
    if (staged) {
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i)
        own[i] = inbox[static_cast<uint64_t>(c.rank) * stride +
                       static_cast<uint64_t>(row) * kRowVectors + i * kThreads + threadIdx.x];
    }
    Vector128 values[kRowPer];
    float square = 0.0f;
    // With LlOrder each vector column keeps its own square sum instead.
    float squares[LlOrder ? kRowPer : 1] = {};
    auto square_of = [&](uint32_t i) -> float& {
      if constexpr (LlOrder) return squares[i];
      else return square;
    };
    if constexpr (Fast) {
      Vector128 peers[kRowPer][kMaxRanks];
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i)
#pragma unroll
        for (uint32_t source = 0; source < kMaxRanks; ++source)
          if (source < c.ranks && source != c.rank)
            peers[i][source] = inbox[source * stride + static_cast<uint64_t>(row) * kRowVectors +
                                     i * kThreads + threadIdx.x];
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i) {
        float sums[8] = {};
#pragma unroll
        for (uint32_t source = 0; source < kMaxRanks; ++source)
          if (source < c.ranks) accumulate(source == c.rank ? own[i] : peers[i][source], sums);
        values[i] = E::carry(sums, operand[i], square_of(i));
      }
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i)
        c.epilogue.keep(static_cast<uint64_t>(row) * kRowVectors + i * kThreads + threadIdx.x,
                        values[i]);
    } else {
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i) {
        const uint64_t index = static_cast<uint64_t>(row) * kRowVectors + i * kThreads + threadIdx.x;
        float sums[8] = {};
        for (uint32_t source = 0; source < c.ranks; ++source)
          accumulate(source == c.rank ? own[i] : inbox[source * stride + index], sums);
        values[i] = E::carry(sums, operand[i], square_of(i));
        c.epilogue.keep(index, values[i]);
      }
    }
    const float inv = [&] {
      if constexpr (LlOrder) {
        // As row_sum<kLlCluster>: zero plus each CTA's sum in CTA order.
        float total = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kRowPer; ++i) total += row_sum<1>(squares[i]);
        return c.epilogue.scale(total);
      } else {
        return c.epilogue.scale(row_sum<1>(square));
      }
    }();
#pragma unroll
    for (uint32_t i = 0; i < kRowPer; ++i)
      c.epilogue.finish(static_cast<uint64_t>(row) * kRowVectors + i * kThreads + threadIdx.x,
                        values[i], parameter[i], inv);
  }
  finish(local + 2 * c.ranks + 1, c.epoch, epoch, c.rows);
}

// Owner q reduces columns [q * width, (q + 1) * width) of every row. Inbox:
// reduce-scatter [source][32 rows * width] then all-gather [source][32 rows *
// width] vectors. The stride uses the largest row count, not the current one:
// a peer's next scatter never overlaps this rank's previous gather, and its
// next gather waits for this rank's next scatter, whatever the row counts of
// the two calls. Flags: reduce-scatter kTwoShotFlags + [source], all-gather
// kTwoShotFlags + [ranks + source], local ready word 2 * ranks, arrival
// counter 2 * ranks + 1. LlOrder sums a row's squares in low_latency's
// order, as one_shot's does.
template <typename In, bool LlOrder, typename E, typename Inboxes, typename Flags>
__device__ void two_shot(const Call<E>& c, Inboxes inboxes, Flags flags,
                         const Vector128 (&parameter)[kRowPer<E>]) {
  constexpr uint32_t kRowVectors = push::kRowVectors<E>, kRowPer = push::kRowPer<E>;
  static_assert(!LlOrder || kRowPer == kLlCluster, "a row thread holds one vector per LL CTA");
  if (blockIdx.x >= kCluster + c.rows) return;
  const uint32_t epoch = c.epoch[0] + 1;
  const uint32_t mark = tagged(epoch);
  const uint32_t width = kRowVectors / c.ranks;
  const uint64_t part = static_cast<uint64_t>(c.rows) * width;
  const uint64_t stride = static_cast<uint64_t>(kBulkMaxRows) * width;
  unsigned int* local = peer(flags, c.rank);
  Vector128* gather = region<Vector128>(inboxes, c.rank, kTwoShotBase) + c.ranks * stride;
  if (blockIdx.x < kCluster) {
    const uint64_t first = blockIdx.x * kThreads + threadIdx.x;
    for (uint32_t step = 1; step < c.ranks; ++step) {
      const uint32_t dest = (c.rank + step) % c.ranks;
      Vector128* target = region<Vector128>(inboxes, dest, kTwoShotBase) + c.rank * stride;
      for (uint64_t v = first; v < part; v += kCluster * kThreads)
        target[v] = input_vector<In>(c, (v / width) * kRowVectors + dest * width + v % width);
      next_destination();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        release_system(peer(flags, dest) + kTwoShotFlags + c.rank, mark);
      }
    }
    // This rank's own columns and their epilogue operands do not depend on
    // peers, so their loads are issued before the wait for the reduce-scatter
    // flags.
    // An owner reduces a 1/ranks share of the columns and two-shot runs with
    // at least kTwoShotMinRanks ranks, so a thread owns at most kPer vectors.
    constexpr uint32_t kPer = (kBulkMaxRows * kRowVectors / kTwoShotMinRanks +
                               kCluster * kThreads - 1) / (kCluster * kThreads);
    Vector128 own[kPer], operand[kPer];
#pragma unroll
    for (uint32_t i = 0; i < kPer; ++i) {
      const uint64_t v = first + i * kCluster * kThreads;
      if (v < part) {
        const uint64_t index = (v / width) * kRowVectors + c.rank * width + v % width;
        own[i] = input_vector<In>(c, index);
        operand[i] = c.epilogue.operand(index);
      }
    }
    if (threadIdx.x < c.ranks && threadIdx.x != c.rank)
      wait_system(local + kTwoShotFlags + threadIdx.x, mark);
    __syncthreads();
    const Vector128* scatter = region<const Vector128>(inboxes, c.rank, kTwoShotBase);
    Vector128 reduced[kPer];
#pragma unroll
    for (uint32_t i = 0; i < kPer; ++i) {
      const uint64_t v = first + i * kCluster * kThreads;
      if (v >= part) continue;
      // Issue every peer load before summing in rank order.
      Vector128 peers[kMaxRanks];
#pragma unroll
      for (uint32_t source = 0; source < kMaxRanks; ++source)
        if (source < c.ranks && source != c.rank) peers[source] = scatter[source * stride + v];
      float sums[8] = {};
#pragma unroll
      for (uint32_t source = 0; source < kMaxRanks; ++source)
        if (source < c.ranks) accumulate(source == c.rank ? own[i] : peers[source], sums);
      float unused = 0.0f;
      reduced[i] = E::carry(sums, operand[i], unused);
      gather[c.rank * stride + v] = reduced[i];
    }
    // The cluster barrier orders every gather store before CTA 0's release,
    // and the all-gather sends from registers, so no per-thread fence.
    __syncthreads();
    cluster_sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) release_gpu(local + 2 * c.ranks, mark);
    // The last reduce-scatter destination comes first: no switch.
    for (uint32_t step = c.ranks - 1; step >= 1; --step) {
      const uint32_t dest = (c.rank + step) % c.ranks;
      Vector128* target = region<Vector128>(inboxes, dest, kTwoShotBase) +
                          (c.ranks + c.rank) * stride;
#pragma unroll
      for (uint32_t i = 0; i < kPer; ++i) {
        const uint64_t v = first + i * kCluster * kThreads;
        if (v < part) target[v] = reduced[i];
      }
      next_destination();
      if (blockIdx.x == 0 && threadIdx.x == 0)
        release_system(peer(flags, dest) + kTwoShotFlags + c.ranks + c.rank, mark);
    }
  } else {
    const uint32_t row = blockIdx.x - kCluster;
    if (threadIdx.x < c.ranks) {
      if (threadIdx.x == c.rank) wait_gpu(local + 2 * c.ranks, mark);
      else wait_system(local + kTwoShotFlags + c.ranks + threadIdx.x, mark);
    }
    __syncthreads();
    test_skew(c.rank, c.ranks, epoch);
    Vector128 values[kRowPer];
    float square = 0.0f;
    float squares[kRowPer];
#pragma unroll
    for (uint32_t i = 0; i < kRowPer; ++i) {
      const uint32_t vector = i * kThreads + threadIdx.x;
      const uint32_t owner = vector / width;
      values[i] = gather[owner * stride + row * width + vector % width];
      c.epilogue.keep(static_cast<uint64_t>(row) * kRowVectors + vector, values[i]);
      squares[i] = E::statistic(values[i]);
      square += squares[i];
    }
    const float inv = [&] {
      if constexpr (LlOrder) {
        // As row_sum<kLlCluster>: zero plus each CTA's sum in CTA order.
        float total = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kRowPer; ++i) total += row_sum<1>(squares[i]);
        return c.epilogue.scale(total);
      } else {
        return c.epilogue.scale(row_sum<1>(square));
      }
    }();
#pragma unroll
    for (uint32_t i = 0; i < kRowPer; ++i)
      c.epilogue.finish(static_cast<uint64_t>(row) * kRowVectors + i * kThreads + threadIdx.x,
                        values[i], parameter[i], inv);
  }
  finish(local + 2 * c.ranks + 1, c.epoch, epoch, c.rows);
}

// Wide two-shot for decode and verify graphs of up to kWideMaxRows rows
// (chop_comm_allreduce_wide_*). Owner q reduces columns [q * width, (q + 1)
// * width) of every row, as in two_shot, with slots sized for kWideMaxRows.
// The last reduce-scatter phase and the last all-gather phase release one
// flag per block of rows, so an owner reduces blocks and workers normalize
// rows while the rest of the phase is in flight. The first all-gather step
// goes to the last reduce-scatter destination (no switch) and fuses the
// owner's reduction: each block is summed as it lands, carried through the
// epilogue, kept (the epilogue's kept rows, its residual_output) and sent
// from registers. Later all-gather steps resend them from the kept rows,
// each thread the vectors it wrote. Workers (every CTA after the sending cluster, any number) wait
// for the peers' columns of a row and normalize it with one CTA per row, so
// outputs are bitwise those of two_shot; pre-norm rows also match one_shot
// and low_latency, whose norms sum squares in other orders.
//
// Inbox: reduce-scatter slots [source][kWideMaxRows * width] vectors from
// kWideScatterBase, all-gather slots the same from kWideGatherBase. The
// scatter slots overlap the phased region only: the phased stages read their
// inbox before any peer can finish the call, and this protocol reads its
// scatter slots before it sends any all-gather data, so a peer's next call
// never overwrites unread bytes of either. The gather slots, read last,
// overlap no other protocol's region. A peer's next gather waits for this
// rank's next scatter, whatever the row counts of the two calls.
constexpr uint32_t kWideMaxRows = CHOP_COMM_WIDE_MAX_ROWS;
constexpr uint32_t kWideBlockRows = CHOP_COMM_WIDE_BLOCK_ROWS;
constexpr uint64_t kWideGatherBase = CHOP_COMM_WIDE_GATHER_BASE;
constexpr uint64_t kWideScatterBase = CHOP_COMM_WIDE_SCATTER_BASE;
// Blocks come in multiples of the sending cluster, so every CTA owns one.
__host__ __device__ constexpr uint32_t wide_blocks(uint32_t rows) {
  return kCluster * ((rows + kCluster * kWideBlockRows - 1) / (kCluster * kWideBlockRows));
}
constexpr uint32_t kWideMaxBlocks = wide_blocks(kWideMaxRows);
// Flag words from CHOP_COMM_WIDE_FLAGS, above the two-shot's: reduce-scatter
// [source], all-gather [source], the last phases' [block] of each, the
// local ready word and arrival counter, then a shape word and its flag per
// source.
constexpr uint32_t kWideScatterFlags = CHOP_COMM_WIDE_FLAGS;
constexpr uint32_t kWideGatherFlags = kWideScatterFlags + kMaxRanks;
constexpr uint32_t kWideScatterBlocks = kWideGatherFlags + kMaxRanks;
constexpr uint32_t kWideGatherBlocks = kWideScatterBlocks + kWideMaxBlocks;
constexpr uint32_t kWideReady = kWideGatherBlocks + kWideMaxBlocks;
constexpr uint32_t kWideCounter = kWideReady + 1;
constexpr uint32_t kWideShape = kWideCounter + 1;
constexpr uint32_t kWideShapeFlags = kWideShape + kMaxRanks;
static_assert(kWideScatterFlags >= kTwoShotFlags + 2 * kMaxRanks, "wide flags follow the two-shot's");
static_assert(kWideMaxRows >= kBulkMaxRows && kWideBlockRows >= 1, "wide rows");
static_assert(kWideScatterBase >= kPhasedBase, "wide scatter slots overlap the phased region only");

// Every rank of a call must run the same rows and live rows: block flags
// follow them, so a rank that disagreed would wait for a block its peer
// never flags. Before its data, CTA 0 of the sending cluster stores its
// shape to every peer (thread t to peer t, a few bytes each) and releases a
// flag; after its reduce-scatter sends, and before any wait that depends
// on the shape, it checks every peer's and traps on a difference, so all
// ranks of a bad call fail instead of hanging. `words` and `shape_flags`
// index this rank's flag buffer by source.
template <typename Flags>
__device__ __forceinline__ void publish_shape(Flags flags, uint32_t rank, uint32_t ranks,
                                              uint32_t words, uint32_t shape_flags,
                                              uint32_t shape, uint32_t mark) {
  if (threadIdx.x < ranks && threadIdx.x != rank) {
    unsigned int* target = peer(flags, threadIdx.x);
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(target + words + rank), "r"(shape) : "memory");
    release_system(target + shape_flags + rank, mark);
  }
}

__device__ __forceinline__ void check_shapes(const unsigned int* local, uint32_t rank,
                                             uint32_t ranks, uint32_t words,
                                             uint32_t shape_flags, uint32_t shape,
                                             uint32_t mark) {
  if (threadIdx.x < ranks && threadIdx.x != rank) {
    wait_system(local + shape_flags + threadIdx.x, mark);
    uint32_t seen;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                 : "=r"(seen) : "l"(local + words + threadIdx.x) : "memory");
    if (seen != shape) __trap();
  }
}

// The last phases of the wide entries release one flag per block rather
// than one at their end (false would send them like the others), and the
// fused reduction keeps two vectors in flight per thread.
constexpr bool kWideBlocked = true;
constexpr uint32_t kWideReduceUnroll = 2;

// Rows [first, end) of block `block` of `per` rows.
struct RowSpan { uint32_t first, end; };
__device__ __forceinline__ RowSpan wide_block(uint32_t block, uint32_t per, uint32_t rows) {
  const uint32_t first = min(block * per, rows);
  return {first, min(first + per, rows)};
}

// Input vector of owner `owner`'s slot vector v.
// `width` is a power of two (512 vectors over power-of-two ranks; the entry
// traps on other rank counts), so shifts and masks replace the 64-bit
// division and remainder, which cost a software routine per vector.
template <typename E>
__device__ __forceinline__ uint64_t column_index(uint64_t v, uint32_t width, uint32_t owner) {
  const uint32_t shift = __ffs(width) - 1;
  return ((v >> shift) * kRowVectors<E>) + owner * width + (v & (width - 1));
}

// Sends the owner's columns of slot vectors [first, end), stepping by
// `step` threads from `start`, with several vectors in flight per thread.
template <typename In, typename E>
__device__ __forceinline__ void send_columns(Vector128* target, const Call<E>& c, uint32_t owner,
                                             uint32_t width, uint64_t first, uint64_t end,
                                             uint64_t start, uint64_t step) {
  constexpr uint32_t kUnroll = In::kStaged ? 2 : 4;
  uint64_t v = first + start;
  for (; v + (kUnroll - 1) * step < end; v += kUnroll * step) {
    Vector128 values[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i)
      values[i] = input_vector<In>(c, column_index<E>(v + i * step, width, owner));
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) target[v + i * step] = values[i];
  }
  for (; v < end; v += step) target[v] = input_vector<In>(c, column_index<E>(v, width, owner));
}

// Resends this rank's carried sums of slot vectors [first, end) from the
// epilogue's kept rows, each thread the vectors it wrote.
template <typename E>
__device__ __forceinline__ void resend_columns(Vector128* target, const Call<E>& c,
                                               uint32_t width, uint64_t first, uint64_t end) {
  constexpr uint32_t kUnroll = 4;
  uint64_t v = first + threadIdx.x;
  for (; v + (kUnroll - 1) * kThreads < end; v += kUnroll * kThreads) {
    Vector128 values[kUnroll];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i)
      values[i] = c.epilogue.kept(column_index<E>(v + i * kThreads, width, c.rank));
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) target[v + i * kThreads] = values[i];
  }
  for (; v < end; v += kThreads) target[v] = c.epilogue.kept(column_index<E>(v, width, c.rank));
}

// Reduces this rank's columns of slot vectors [first, end) in rank order,
// carries them through the epilogue, keeps the carried sums and sends them
// to `target` (null at one rank). Two vectors per thread keep their loads
// in flight together.
template <typename In, typename E>
__device__ __forceinline__ void reduce_columns(Vector128* target, const Vector128* scatter,
                                               uint64_t slot, const Call<E>& c, uint32_t width,
                                               uint64_t first, uint64_t end) {
  constexpr uint32_t kUnroll = kWideReduceUnroll;
  for (uint64_t base = first + threadIdx.x; base < end; base += kUnroll * kThreads) {
    Vector128 own[kUnroll], operands[kUnroll], peers[kUnroll][kMaxRanks];
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
      const uint64_t v = base + i * kThreads;
      if (v >= end) continue;
      const uint64_t index = column_index<E>(v, width, c.rank);
      own[i] = input_vector<In>(c, index);
      operands[i] = c.epilogue.operand(index);
#pragma unroll
      for (uint32_t source = 0; source < kMaxRanks; ++source)
        if (source < c.ranks && source != c.rank) peers[i][source] = scatter[source * slot + v];
    }
#pragma unroll
    for (uint32_t i = 0; i < kUnroll; ++i) {
      const uint64_t v = base + i * kThreads;
      if (v >= end) continue;
      float sums[8] = {};
#pragma unroll
      for (uint32_t source = 0; source < kMaxRanks; ++source)
        if (source < c.ranks) accumulate(source == c.rank ? own[i] : peers[i][source], sums);
      float unused = 0.0f;
      const Vector128 value = E::carry(sums, operands[i], unused);
      c.epilogue.keep(column_index<E>(v, width, c.rank), value);
      if (target) target[v] = value;
    }
  }
}

// CTAs [0, kCluster) form the sending cluster; every later CTA is a worker
// that normalizes rows worker, worker + workers, ...
template <typename In, typename E, typename Inboxes, typename Flags>
__device__ void wide(const Call<E>& c, Inboxes inboxes, Flags flags,
                     const Vector128 (&parameter)[kRowPer<E>]) {
  constexpr uint32_t kRowVectors = push::kRowVectors<E>, kRowPer = push::kRowPer<E>;
  static_assert(kWideGatherBase >= kTwoShotBase + 2ull * kBulkMaxRows * kRowVectors * 16 &&
                kWideGatherBase + uint64_t(kWideMaxRows) * kRowVectors * 16 <= kPhasedBase,
                "wide gather slots sit between the two-shot and phased regions");
  const uint32_t epoch = c.epoch[0] + 1;
  const uint32_t mark = tagged(epoch);
  const uint32_t width = kRowVectors / c.ranks;
  const uint64_t slot = static_cast<uint64_t>(kWideMaxRows) * width;
  const uint32_t blocks = wide_blocks(c.rows);
  const uint32_t per = (c.rows + blocks - 1) / blocks;
  unsigned int* local = peer(flags, c.rank);
  // The shape word: rows and live rows. No parity bank: a peer writes the
  // next call's only after this call's all-gather data, which follows the
  // check.
  const uint32_t shape = c.rows | (c.live << 16);
  if (blockIdx.x < kCluster) {
    if (blockIdx.x == 0)
      publish_shape(flags, c.rank, c.ranks, kWideShape, kWideShapeFlags, shape, mark);
    test_skew(c.rank, c.ranks, epoch);
    const uint64_t part = static_cast<uint64_t>(c.rows) * width;
    for (uint32_t step = 1; step < c.ranks; ++step) {
      const uint32_t dest = (c.rank + step) % c.ranks;
      Vector128* target = region<Vector128>(inboxes, dest, kWideScatterBase) + c.rank * slot;
      if (step + 1 < c.ranks || !kWideBlocked) {
        send_columns<In>(target, c, dest, width, 0, part, blockIdx.x * kThreads + threadIdx.x,
                              kCluster * kThreads);
        next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0)
          release_system(peer(flags, dest) + kWideScatterFlags + c.rank, mark);
        continue;
      }
      // The last phase flags each block from the CTA that sent it. Empty
      // blocks (rows that do not fill the last ones) carry no flag on any
      // rank: a flag nobody needs before the call ends could be rewritten
      // by a peer's next call before its reader loads it.
      for (uint32_t block = blockIdx.x; block < blocks; block += kCluster) {
        const RowSpan rows = wide_block(block, per, c.rows);
        if (rows.first == rows.end) continue;
        send_columns<In>(target, c, dest, width, uint64_t(rows.first) * width,
                              uint64_t(rows.end) * width, threadIdx.x, kThreads);
        __syncthreads();
        if (threadIdx.x == 0)
          release_system(peer(flags, dest) + kWideScatterBlocks + block, mark);
      }
    }
    if (blockIdx.x == 0)
      check_shapes(local, c.rank, c.ranks, kWideShape, kWideShapeFlags, shape, mark);
    // All-gather, starting with the last reduce-scatter destination: this
    // step reduces each block as its last part lands and sends the sums.
    const uint32_t last = (c.rank + 1) % c.ranks;  // source of the last reduce-scatter phase
    if (threadIdx.x < c.ranks && threadIdx.x != c.rank && (threadIdx.x != last || !kWideBlocked))
      wait_system(local + kWideScatterFlags + threadIdx.x, mark);
    const Vector128* scatter = region<const Vector128>(inboxes, c.rank, kWideScatterBase);
    // Steps ranks - 1 down to 1; one rank runs the reduction alone.
    const uint32_t steps = c.ranks > 1 ? c.ranks - 1 : 1;
    for (uint32_t k = 0; k < steps; ++k) {
      const uint32_t dest = (c.rank + c.ranks - 1 - k) % c.ranks;
      const bool reducing = k == 0;
      const bool final_step = k + 1 == steps;
      Vector128* target = c.ranks > 1
          ? region<Vector128>(inboxes, dest, kWideGatherBase) + c.rank * slot : nullptr;
      for (uint32_t block = blockIdx.x; block < blocks; block += kCluster) {
        const RowSpan rows = wide_block(block, per, c.rows);
        if (rows.first == rows.end) continue;
        const uint64_t first = uint64_t(rows.first) * width, end = uint64_t(rows.end) * width;
        if (reducing) {
          if (kWideBlocked && c.ranks > 1 && threadIdx.x == 0)
            wait_system(local + kWideScatterBlocks + block, mark);
          __syncthreads();
          reduce_columns<In>(target, scatter, slot, c, width, first, end);
        } else {
          resend_columns(target, c, width, first, end);
        }
        if (kWideBlocked && final_step && c.ranks > 1) {
          __syncthreads();
          if (threadIdx.x == 0)
            release_system(peer(flags, dest) + kWideGatherBlocks + block, mark);
        }
      }
      // Without block flags the last step ends like the others, with one
      // flag after the cluster barrier.
      const bool full_flag = !final_step || !kWideBlocked;
      if (reducing) {
        // Workers read this rank's kept sums after the ready
        // word; the cluster barrier orders every CTA's stores before it.
        __syncthreads();
        cluster_sync();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
          release_gpu(local + kWideReady, mark);
          if (full_flag && c.ranks > 1)
            release_system(peer(flags, dest) + kWideGatherFlags + c.rank, mark);
        }
      } else if (full_flag) {
        next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0)
          release_system(peer(flags, dest) + kWideGatherFlags + c.rank, mark);
      }
    }
  } else {
    const uint32_t worker = blockIdx.x - kCluster;
    const uint32_t workers = gridDim.x - kCluster;
    const uint32_t last = (c.rank + c.ranks - 1) % c.ranks;  // source of the last all-gather step
    if (threadIdx.x < c.ranks) {
      if (threadIdx.x == c.rank) wait_gpu(local + kWideReady, mark);
      else if (threadIdx.x != last || !kWideBlocked)
        wait_system(local + kWideGatherFlags + threadIdx.x, mark);
    }
    test_skew(c.rank, c.ranks, epoch);
    const Vector128* gathered = region<const Vector128>(inboxes, c.rank, kWideGatherBase);
    uint32_t waited = kWideMaxBlocks;
    for (uint32_t row = worker; row < c.rows; row += workers) {
      const uint32_t block = row / per;
      if (kWideBlocked && block != waited && c.ranks > 1 && threadIdx.x == 0)
        wait_system(local + kWideGatherBlocks + block, mark);
      waited = block;
      __syncthreads();
      Vector128 values[kRowPer];
      float square = 0.0f;
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i) {
        const uint32_t vector = i * kThreads + threadIdx.x;
        const uint32_t owner = vector >> (__ffs(width) - 1);
        const uint64_t index = static_cast<uint64_t>(row) * kRowVectors + vector;
        if (owner == c.rank) {
          values[i] = c.epilogue.kept(index);
        } else {
          values[i] = gathered[owner * slot + static_cast<uint64_t>(row) * width +
                               (vector & (width - 1))];
          c.epilogue.keep(index, values[i]);
        }
        square += E::statistic(values[i]);
      }
      const float inv = c.epilogue.scale(row_sum<1>(square));
#pragma unroll
      for (uint32_t i = 0; i < kRowPer; ++i)
        c.epilogue.finish(static_cast<uint64_t>(row) * kRowVectors + i * kThreads + threadIdx.x,
                          values[i], parameter[i], inv);
    }
  }
  finish(local + kWideCounter, c.epoch, epoch, gridDim.x - kCluster);
}

// Launch geometry per protocol; the planner computes the same values.
__device__ __forceinline__ uint32_t grid_for(Protocol protocol, uint32_t rows) {
  if (protocol == Protocol::LowLatency) return rows * kLlCluster;
  return kCluster * (1 + (rows + kCluster - 1) / kCluster);
}

} // namespace chop::comm::push

namespace chop::comm {

// In is the input policy (input/bf16.cuh by default; with
// input/routed_combine.cuh `input` holds the preweighted top-8 routed rows),
// and live_rows (nullable) bounds the rows that carry input. Fast selects the
// decode entries' variants (chop_comm_allreduce_decode_*), which take the
// same arguments, state and protocols and give the same bits. AsyncNorm
// (plain decode v2, decode combine v1 and verify combine v1) reduces the LL
// norm over st.async, retaining
// each CTA's reduction tree and the ordered sum of CTA results.
// LlOrder (the verify entries) sums one-shot and two-shot rows' statistics
// in the low-latency order (see one_shot), so every row's outputs are those
// of the same row alone. E is the epilogue functor (comm/epilogue/), whose
// width sets the row geometry.
template <typename E, typename In = input::Bf16, bool Fast = false,
          bool AsyncNorm = Fast && In::kStaged, bool LlOrder = false, typename Inboxes,
          typename Flags>
__device__ void allreduce_push(const uint16_t* input, Inboxes inboxes, Flags flags,
                               unsigned int* epoch, uint64_t elements, uint32_t rank,
                               uint32_t ranks, const E& epilogue, const uint32_t* live_rows = nullptr) {
  static_assert(sizeof(typename In::Element) == 2 && In::kPerVector == 8,
                "the push transport exchanges BF16 rows");
  ptx::griddep_launch_dependents();
  const uint32_t rows = static_cast<uint32_t>(elements / E::kWidth);
  if (elements % E::kWidth || rows == 0 || rank >= ranks || ranks == 0 || ranks > push::kMaxRanks) return;
  if (rows > push::kBulkMaxRows) __trap();
  const push::Protocol protocol = push::protocol(rows, ranks);
  // A geometry mismatch would leave peers waiting; fail the launch instead.
  if (blockDim.x != push::kThreads || gridDim.x != push::grid_for(protocol, rows)) __trap();
  // Guards a table the planner did not check: kPer and the column shares
  // hold only for these rank counts.
  if (protocol == push::Protocol::TwoShot &&
      (ranks < push::kTwoShotMinRanks || (ranks & (ranks - 1)) != 0))
    __trap();
  // The epilogue's row parameters (norm weights) are model constants, so
  // they are read before the PDL wait.
  const Vector128* parameters = epilogue.parameters();
  Vector128 parameter[push::kRowPer<E>] = {};
  if constexpr (AsyncNorm) {
    if (protocol == push::Protocol::LowLatency) push::prepare_async_sum<push::kLlCluster>();
  }
  if (protocol == push::Protocol::LowLatency) {
    if (threadIdx.x < push::kLlVectors<E>)
      parameter[0] = parameters[(blockIdx.x % push::kLlCluster) * push::kLlVectors<E> + threadIdx.x];
  } else if (blockIdx.x >= push::kCluster && blockIdx.x < push::kCluster + rows) {
#pragma unroll
    for (uint32_t i = 0; i < push::kRowPer<E>; ++i)
      parameter[i] = parameters[i * push::kThreads + threadIdx.x];
  }
  ptx::griddep_wait();
  const push::Call<E> call{input, epilogue, epoch, rows, rank, ranks,
                           live_rows ? min(*live_rows, rows) : rows};
  if (protocol == push::Protocol::LowLatency)
    push::low_latency<In, Fast, AsyncNorm>(call, inboxes, parameter);
  else if (protocol == push::Protocol::OneShot)
    push::one_shot<In, Fast, LlOrder>(call, inboxes, flags, parameter);
  else push::two_shot<In, LlOrder>(call, inboxes, flags, parameter);
}

// The wide two-shot (chop_comm_allreduce_wide_*): the v2 and combine
// arguments, for 1 to kWideMaxRows rows at power-of-two rank counts. The
// planner sends it the rows above the push path's 32; smaller row counts
// run it too. Any worker count may follow the sending cluster; ranks agree
// on the protocol whatever their grids. The epilogue must keep its carried
// rows (keep, kept): owners resend them and workers read their own.
template <typename E, typename In = input::Bf16, typename Inboxes, typename Flags>
__device__ void allreduce_wide(const uint16_t* input, Inboxes inboxes, Flags flags,
                               unsigned int* epoch, uint64_t elements, uint32_t rank,
                               uint32_t ranks, const E& epilogue, const uint32_t* live_rows = nullptr) {
  static_assert(sizeof(typename In::Element) == 2 && In::kPerVector == 8,
                "the push transport exchanges BF16 rows");
  ptx::griddep_launch_dependents();
  const uint64_t rows = elements / E::kWidth;
  // A bad call would leave peers waiting; fail the launch instead.
  if (elements % E::kWidth || rows == 0 || rows > push::kWideMaxRows || ranks == 0 ||
      ranks > push::kMaxRanks || (ranks & (ranks - 1)) != 0 || rank >= ranks)
    __trap();
  if (blockDim.x != push::kThreads || gridDim.x <= push::kCluster ||
      gridDim.x % push::kCluster != 0 || gridDim.y != 1 || gridDim.z != 1 ||
      ptx::cluster_nctarank() != push::kCluster)
    __trap();
  // The epilogue's row parameters are model constants, so workers read them
  // before the PDL wait.
  Vector128 parameter[push::kRowPer<E>] = {};
  if (blockIdx.x >= push::kCluster) {
#pragma unroll
    for (uint32_t i = 0; i < push::kRowPer<E>; ++i)
      parameter[i] = epilogue.parameters()[i * push::kThreads + threadIdx.x];
  }
  ptx::griddep_wait();
  const push::Call<E> call{input, epilogue, epoch, static_cast<uint32_t>(rows), rank, ranks,
                           live_rows ? min(*live_rows, static_cast<uint32_t>(rows))
                                     : static_cast<uint32_t>(rows)};
  push::wide<In>(call, inboxes, flags, parameter);
}

} // namespace chop::comm

// Phased two-shot for large all-reduces (prefill), in two kernels that keep
// the legacy reduce-scatter and gather arguments. Owner q holds the tiles
// [q * tiles / ranks, (q + 1) * tiles / ranks) of 4096 elements. The first
// CTA cluster sends to one destination at a time (rank + 1, + 2, ...), so
// every phase is a permutation of the ranks; the remaining CTAs reduce and
// normalize. Each rank sends 2 (ranks - 1) / ranks of the message, the
// all-reduce minimum.
//
// Scatter: each rank pushes the owner's tiles of its input into the owner's
// inbox slot, and the owner sums all ranks in rank order in F32 and writes
// the rounded sum into its own tiles of `output`; the fused scatter carries
// the F32 sum through the epilogue E (its one rounding, as the push
// protocols do). Gather: the owner pushes its tiles into every peer's
// output; the fused gather keeps them as the carried rows and its workers
// finish rows as their tiles arrive, summing each row's statistic in the
// low-latency order. A fused row is one tile, so E::kWidth must be kTile.
// The fused pair's rows are bitwise those of the push entries' verify
// variants and of the low-latency protocol.
// The flag words live above the push protocol's; both kernels use epoch[0],
// which the gather advances, so calls with any row count or protocol can
// follow each other on the same buffers.
namespace chop::comm::phased {

using push::kCluster;
using push::kThreads;
constexpr uint64_t kTile = 4096;
constexpr uint32_t kScatterFlags = 32;   // [source]
constexpr uint32_t kGatherFlags = 40;    // [source]
constexpr uint32_t kReady = 48;          // owner's pre-norm tiles written
constexpr uint32_t kDone = 49;           // gather arrival counter
constexpr uint32_t kBlocks = 32;         // flagged blocks of the last scatter phase
constexpr uint32_t kBlockFlags = 50;     // [block]

struct Span { uint64_t first, count; };

__device__ __forceinline__ Span owner_span(uint64_t elements, uint32_t owner, uint32_t ranks) {
  const uint64_t tiles = (elements + kTile - 1) / kTile;
  const uint64_t first = min(owner * tiles / ranks * kTile, elements);
  const uint64_t end = min((owner + 1) * tiles / ranks * kTile, elements);
  return {first, end - first};
}

// Elements per inbox slot: the largest owner span.
__device__ __forceinline__ uint64_t slot_elements(uint64_t elements, uint32_t ranks) {
  const uint64_t tiles = (elements + kTile - 1) / kTile;
  return (tiles + ranks - 1) / ranks * kTile;
}

// Copies count elements with 16-byte vectors; both pointers start on a tile.
template <typename T>
__device__ __forceinline__ void copy(T* target, const T* source, uint64_t count,
                                     uint32_t worker, uint32_t workers) {
  constexpr uint64_t kWidth = 16 / sizeof(T);
  const uint64_t vectors = count / kWidth;
  push::copy_vectors(reinterpret_cast<Vector128*>(target),
                     reinterpret_cast<const Vector128*>(source), vectors, worker, workers);
  const uint64_t step = static_cast<uint64_t>(workers) * kThreads;
  for (uint64_t i = vectors * kWidth + worker * kThreads + threadIdx.x; i < count; i += step)
    target[i] = source[i];
}

__device__ __forceinline__ bool valid_geometry(uint32_t rank, uint32_t ranks) {
  return rank < ranks && ranks > 0 && ranks <= push::kMaxRanks;
}

// A geometry mismatch would leave peers waiting; fail the launch instead.
__device__ __forceinline__ void check_grid() {
  if (blockDim.x != kThreads || gridDim.x <= kCluster || gridDim.x % kCluster) __trap();
}

// Blocks of at least 64 tiles (512 KiB of BF16) keep per-block flags cheap.
__device__ __forceinline__ uint32_t blocks_of(uint64_t count) {
  const uint64_t tiles = (count + kTile - 1) / kTile;
  const uint64_t blocks = tiles / 64;
  return blocks == 0 ? 1 : blocks > kBlocks ? kBlocks : static_cast<uint32_t>(blocks);
}

// Owner-relative element range of block b of a span; blocks are tile-aligned.
__device__ __forceinline__ Span block_of(uint64_t count, uint32_t block) {
  const uint64_t tiles = (count + kTile - 1) / kTile;
  const uint32_t blocks = blocks_of(count);
  const uint64_t first = min(block * tiles / blocks * kTile, count);
  const uint64_t end = min((block + 1) * tiles / blocks * kTile, count);
  return {first, end - first};
}

// Rank-ordered F32 sum of [first, first + count) of the owner's span; with
// an epilogue E, carried with the owner-relative residual instead of a
// plain rounding.
template <typename T, typename E = void>
__device__ __forceinline__ void sum_range(const T* input, const T* inbox, T* output,
                                          const T* residual, uint64_t slot, uint64_t first,
                                          uint64_t count, uint32_t rank, uint32_t ranks,
                                          uint32_t worker, uint32_t workers) {
  constexpr uint64_t kWidth = 16 / sizeof(T);
  const uint64_t step = static_cast<uint64_t>(workers) * kThreads;
  const uint64_t vectors = count / kWidth;
  for (uint64_t v = worker * kThreads + threadIdx.x; v < vectors; v += step) {
    const uint64_t at = first / kWidth + v;
    Vector128 parts[push::kMaxRanks];
#pragma unroll
    for (uint32_t source = 0; source < push::kMaxRanks; ++source)
      if (source < ranks)
        parts[source] = reinterpret_cast<const Vector128*>(
            source == rank ? input : inbox + source * slot)[at];
    float sums[kWidth] = {};
#pragma unroll
    for (uint32_t source = 0; source < push::kMaxRanks; ++source) {
      if (source >= ranks) break;
      const T* values = reinterpret_cast<const T*>(&parts[source]);
#pragma unroll
      for (uint64_t k = 0; k < kWidth; ++k) sums[k] += decode(values[k]);
    }
    Vector128 out;
    if constexpr (!std::is_void_v<E>) {
      static_assert(sizeof(T) == 2, "the round-once scatter is BF16");
      float unused = 0.0f;
      reinterpret_cast<Vector128*>(output)[at] = E::carry(
          sums, reinterpret_cast<const Vector128*>(residual)[at], unused);
      continue;
    }
    T* packed = reinterpret_cast<T*>(&out);
#pragma unroll
    for (uint64_t k = 0; k < kWidth; ++k) packed[k] = encode<T>(sums[k]);
    reinterpret_cast<Vector128*>(output)[at] = out;
  }
  for (uint64_t i = first + vectors * kWidth + worker * kThreads + threadIdx.x;
       i < first + count; i += step) {
    float sum = 0.0f;
    for (uint32_t source = 0; source < ranks; ++source)
      sum += decode(source == rank ? input[i] : inbox[source * slot + i]);
    if constexpr (!std::is_void_v<E>) output[i] = E::carry_scalar(sum, residual[i]);
    else output[i] = encode<T>(sum);
  }
}

// The last phase releases one flag per block, from the CTA that copied it,
// so workers sum each block while the rest of the phase is in flight.
template <typename T, typename E = void, typename Inboxes, typename Flags>
__device__ void reduce_scatter(const T* input, T* output, Inboxes inboxes, Flags flags,
                               const unsigned int* epoch, uint64_t elements,
                               uint32_t rank, uint32_t ranks, const T* residual = nullptr) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (!valid_geometry(rank, ranks) || elements == 0) return;
  check_grid();
  const uint32_t mark = push::tagged(epoch[0] + 1);
  const uint64_t slot = slot_elements(elements, ranks);
  unsigned int* local = peer(flags, rank);
  if (blockIdx.x < kCluster) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      local[kReady] = 0;
      local[kDone] = 0;
    }
    for (uint32_t step = 1; step < ranks; ++step) {
      const uint32_t dest = (rank + step) % ranks;
      const Span span = owner_span(elements, dest, ranks);
      T* target = push::region<T>(inboxes, dest, push::kPhasedBase) + rank * slot;
      if (step + 1 < ranks) {
        copy(target, input + span.first, span.count, blockIdx.x, kCluster);
        push::next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0)
          release_system(peer(flags, dest) + kScatterFlags + rank, mark);
        continue;
      }
      for (uint32_t block = blockIdx.x; block < blocks_of(span.count); block += kCluster) {
        const Span part = block_of(span.count, block);
        copy(target + part.first, input + span.first + part.first, part.count, 0, 1);
        __syncthreads();
        if (threadIdx.x == 0) release_system(peer(flags, dest) + kBlockFlags + block, mark);
      }
    }
    return;
  }
  const uint32_t worker = blockIdx.x - kCluster;
  const uint32_t workers = gridDim.x - kCluster;
  const uint32_t last = (rank + 1) % ranks;   // source of the last phase
  if (threadIdx.x < ranks && threadIdx.x != rank && threadIdx.x != last)
    push::wait_system(local + kScatterFlags + threadIdx.x, mark);
  __syncthreads();
  const Span span = owner_span(elements, rank, ranks);
  const T* inbox = push::region<const T>(inboxes, rank, push::kPhasedBase);
  for (uint32_t block = 0; block < blocks_of(span.count); ++block) {
    if (ranks > 1) {
      if (threadIdx.x == 0) push::wait_system(local + kBlockFlags + block, mark);
      __syncthreads();
    }
    const Span part = block_of(span.count, block);
    sum_range<T, E>(input + span.first, inbox, output + span.first,
                    std::is_void_v<E> ? nullptr : residual + span.first, slot, part.first,
                    part.count, rank, ranks, worker, workers);
  }
}

// The epilogue's finish of one row by one warp, summing its statistic in
// push::low_latency's order, so a row comes out as it would alone at any
// row count: vector v = lane + 32 i is thread v % 256 of LL CTA v / 256.
// Each vector's statistic sums from zero; each group of 32 vectors (an LL
// warp) sums by the same butterfly; the 8 warp sums of an LL CTA combine as
// row_sum's warp-0 butterfly does (lanes 8 to 31 add zeros); the CTA sums
// add to zero in CTA order. `pre` holds the carried row; it may alias
// `copy_to` (skipped).
template <typename E>
__device__ __forceinline__ void norm_row_warp(const uint16_t* pre, uint16_t* out,
                                              uint16_t* copy_to, const uint16_t* weights,
                                              float eps) {
  constexpr uint32_t kPer = push::kRowVectors<E> / 32;
  static_assert(E::kWidth == kTile, "a fused row is one tile");
  constexpr uint32_t kWarps = push::kThreads / 32;
  static_assert(kPer == push::kLlCluster * kWarps, "a row holds LL CTAs of 8 warps");
  const uint32_t lane = threadIdx.x & 31;
  Vector128 values[kPer];
  float warp_sums[kPer];
#pragma unroll
  for (uint32_t i = 0; i < kPer; ++i) {
    values[i] = reinterpret_cast<const Vector128*>(pre)[lane + 32 * i];
    float sum = E::statistic(values[i]);
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    warp_sums[i] = sum;
  }
  if (copy_to) {
#pragma unroll
    for (uint32_t i = 0; i < kPer; ++i)
      reinterpret_cast<Vector128*>(copy_to)[lane + 32 * i] = values[i];
  }
  float square = 0.0f;
#pragma unroll
  for (uint32_t cta = 0; cta < push::kLlCluster; ++cta) {
    const float* w = warp_sums + cta * kWarps;
    square += ((w[0] + w[4]) + (w[2] + w[6])) + ((w[1] + w[5]) + (w[3] + w[7]));
  }
  const float inv = E::inverse(square, eps);
#pragma unroll
  for (uint32_t i = 0; i < kPer; ++i)
    reinterpret_cast<Vector128*>(out)[lane + 32 * i] = E::normalize(
        values[i], reinterpret_cast<const Vector128*>(weights)[lane + 32 * i], inv);
}

__device__ __forceinline__ void wait_count(const unsigned int* address, uint32_t count) {
  uint32_t value;
  do {
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  } while (value != count);
  acquire_gpu(address);
}

// The plain gather has no epilogue (E void) and null norm weights. The
// fused gather follows the fused scatter, so the owner's tiles in `output`
// are the carried rows; it needs residual_output: the cluster sends the
// owner's tiles from it while workers finish `output` in place.
template <typename T, typename E = void, typename Outputs, typename Flags>
__device__ void allgather(T* output, Outputs outputs, Flags flags, unsigned int* epoch,
                          uint64_t elements, uint32_t rank, uint32_t ranks,
                          const uint16_t* weights, uint32_t eps_bits,
                          uint16_t* residual_output) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (!valid_geometry(rank, ranks) || elements == 0) return;
  check_grid();
  constexpr bool kFused = !std::is_void_v<E>;
  static_assert(!kFused || sizeof(T) == 2, "the fused gather is BF16");
  const bool fused = kFused && weights != nullptr;
  if (fused && (residual_output == nullptr || elements % kTile)) __trap();
  const uint32_t epoch_value = epoch[0] + 1;
  const uint32_t mark = push::tagged(epoch_value);
  unsigned int* local = peer(flags, rank);
  const uint32_t workers = gridDim.x - kCluster;
  const Span own = owner_span(elements, rank, ranks);
  if (blockIdx.x < kCluster) {
    const T* source = output;
    uint32_t step = 1;
    if constexpr (kFused) {
      if (fused) {
        // The first phase keeps the pre-norm tiles in residual_output and
        // sends them; later phases resend those.
        const uint32_t dest = (rank + 1) % ranks;
        auto* target = reinterpret_cast<Vector128*>(peer(outputs, dest));
        const auto* sums = reinterpret_cast<const Vector128*>(output);
        for (uint64_t v = own.first / 8 + blockIdx.x * kThreads + threadIdx.x;
             v < (own.first + own.count) / 8; v += kCluster * kThreads) {
          const Vector128 pre = sums[v];
          reinterpret_cast<Vector128*>(residual_output)[v] = pre;
          if (ranks > 1) target[v] = pre;
        }
        __threadfence();
        push::next_destination();
        if (blockIdx.x == 0 && threadIdx.x == 0) {
          __threadfence();
          atomicAdd(local + kReady, 1u);
          if (ranks > 1) release_system(peer(flags, dest) + kGatherFlags + rank, mark);
        }
        source = reinterpret_cast<const T*>(residual_output);
        step = 2;
      }
    }
    for (; step < ranks; ++step) {
      const uint32_t dest = (rank + step) % ranks;
      copy(reinterpret_cast<T*>(peer(outputs, dest)) + own.first, source + own.first,
           own.count, blockIdx.x, kCluster);
      push::next_destination();
      if (blockIdx.x == 0 && threadIdx.x == 0)
        release_system(peer(flags, dest) + kGatherFlags + rank, mark);
    }
  } else {
    const uint32_t worker = blockIdx.x - kCluster;
    if constexpr (kFused) {
      if (fused) {
        const float eps = __uint_as_float(eps_bits);
        auto* sums = reinterpret_cast<uint16_t*>(output);
        // The sending cluster wrote the owner's pre-norm tiles.
        if (threadIdx.x == 0) wait_count(local + kReady, 1);
        __syncthreads();
        const uint32_t warp = worker * (kThreads / 32) + threadIdx.x / 32;
        const uint32_t warps = workers * (kThreads / 32);
        for (uint32_t step = 0; step < ranks; ++step) {
          const uint32_t source = (rank + ranks - step) % ranks;
          if (step > 0) {
            if (threadIdx.x == 0) push::wait_system(local + kGatherFlags + source, mark);
            __syncthreads();
          }
          const Span span = owner_span(elements, source, ranks);
          for (uint64_t row = span.first / kTile + warp; row < (span.first + span.count) / kTile;
               row += warps) {
            // The owner's tiles already sit in residual_output; peers' arrive
            // in output and are copied there before normalization.
            const bool mine = step == 0;
            norm_row_warp<E>(mine ? residual_output + row * kTile : sums + row * kTile,
                             sums + row * kTile, mine ? nullptr : residual_output + row * kTile,
                             weights, eps);
          }
        }
      }
    }
    if (!fused) {
      if (threadIdx.x < ranks && threadIdx.x != rank)
        push::wait_system(local + kGatherFlags + threadIdx.x, mark);
    }
  }
  push::finish(local + kDone, epoch, epoch_value, workers);
}

} // namespace chop::comm::phased

namespace chop::comm {

template <typename T, typename Inboxes, typename Flags>
__device__ void reduce_scatter(const T* input, T* partial, Inboxes inboxes, Flags flags,
                               const unsigned int* epoch, uint64_t elements, uint32_t rank,
                               uint32_t ranks) {
  phased::reduce_scatter(input, partial, inboxes, flags, epoch, elements, rank, ranks);
}

// The fused pair with epilogue E: the reduce-scatter carries its sums with
// `residual` (one rounding), and the fused gather sends and finishes them.
template <typename E, typename Inboxes, typename Flags>
__device__ void reduce_scatter_presum(const uint16_t* input, uint16_t* partial,
                                      const uint16_t* residual, Inboxes inboxes, Flags flags,
                                      const unsigned int* epoch, uint64_t elements,
                                      uint32_t rank, uint32_t ranks) {
  if (residual == nullptr) __trap();
  phased::reduce_scatter<uint16_t, E>(input, partial, inboxes, flags, epoch, elements, rank,
                                      ranks, residual);
}

template <typename E, typename Outputs, typename Flags>
__device__ void allgather_presummed(uint16_t* output, Outputs outputs, Flags flags,
                                    unsigned int* epoch, uint64_t elements, uint32_t rank,
                                    uint32_t ranks, const uint16_t* weights, uint32_t eps_bits,
                                    uint16_t* residual_output) {
  if (weights == nullptr) __trap();
  phased::allgather<uint16_t, E>(output, outputs, flags, epoch, elements, rank, ranks, weights,
                                 eps_bits, residual_output);
}

template <typename T, typename Outputs, typename Flags>
__device__ void allgather_scattered(T* output, Outputs outputs, Flags flags,
                                    unsigned int* epoch, uint64_t elements, uint32_t rank,
                                    uint32_t ranks) {
  phased::allgather(output, outputs, flags, epoch, elements, rank, ranks, nullptr, 0, nullptr);
}

} // namespace chop::comm

// Batched row gather to one rank (logits). Every sender writes only to the
// target, which receives from several senders at once at full speed. Row
// indices come from a device list (or rows 0..count-1 when it is null); rank
// q's shard of row r lands at outputs[target] + (r * ranks + q) * shard.
// Each of kGatherCtas CTAs owns a contiguous slice of the rows' elements, its
// own epoch cell, one data flag per source and one acknowledgment word, so
// CTAs never synchronize with each other. At the start of a call the target
// acknowledges the previous one; senders wait for that before overwriting,
// on every call: the first call's acknowledgment, tagged(0), is not the zero
// the flags start with, and no epoch value (including the wrap through zero)
// skips the wait.
namespace chop::comm::rows {

using push::kThreads;
constexpr uint32_t kGatherCtas = 8;
constexpr uint32_t kDataFlags = 0;                        // [source * 8 + cta]
constexpr uint32_t kAckFlags = push::kMaxRanks * kGatherCtas;  // [cta]

template <typename T, typename Outputs, typename Flags>
__device__ void gather(const T* input, Outputs outputs, Flags flags, unsigned int* epoch,
                       const uint32_t* row_indices, uint32_t count, uint64_t shard,
                       uint32_t rank, uint32_t ranks, uint32_t target) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (rank >= ranks || ranks == 0 || ranks > push::kMaxRanks || target >= ranks ||
      count == 0 || shard == 0)
    return;
  if (blockDim.x != kThreads || gridDim.x != kGatherCtas) __trap();
  const uint32_t cta = blockIdx.x;
  const uint32_t epoch_value = epoch[cta] + 1;
  const uint32_t mark = push::tagged(epoch_value);
  unsigned int* local = peer(flags, rank);
  T* gathered = reinterpret_cast<T*>(peer(outputs, target));
  if (threadIdx.x == 0) {
    if (rank == target) {
      for (uint32_t source = 0; source < ranks; ++source)
        if (source != rank)
          release_system(peer(flags, source) + kAckFlags + cta, push::tagged(epoch_value - 1));
    } else {
      push::wait_system(local + kAckFlags + cta, push::tagged(epoch_value - 1));
    }
  }
  __syncthreads();
  // This CTA's slice of the flattened (row, element) space, in 16-byte
  // vectors when rows are a whole number of vectors.
  constexpr uint64_t kWidth = 16 / sizeof(T);
  const uint64_t width = shard % kWidth == 0 ? kWidth : 1;
  const uint64_t units = static_cast<uint64_t>(count) * shard / width;
  const uint64_t end = units * (cta + 1) / kGatherCtas;
  for (uint64_t unit = units * cta / kGatherCtas + threadIdx.x; unit < end; unit += kThreads) {
    const uint64_t index = unit * width;
    const uint64_t slot = index / shard;
    const uint64_t column = index % shard;
    const uint64_t row = row_indices ? row_indices[slot] : slot;
    const T* from = input + row * shard + column;
    T* to = gathered + (row * ranks + rank) * shard + column;
    if (width == kWidth) *reinterpret_cast<Vector128*>(to) = *reinterpret_cast<const Vector128*>(from);
    else *to = *from;
  }
  __syncthreads();
  if (rank != target) {
    if (threadIdx.x == 0)
      release_system(peer(flags, target) + kDataFlags + rank * kGatherCtas + cta, mark);
  } else if (threadIdx.x < ranks && threadIdx.x != rank) {
    push::wait_system(local + kDataFlags + threadIdx.x * kGatherCtas + cta, mark);
  }
  __syncthreads();
  if (threadIdx.x == 0) epoch[cta] = epoch_value;
}

// Row exchange for decode logits: rows are split into contiguous blocks of
// ceil(rows / ranks), block q owned by rank q, and every rank pushes its shard
// of each row to the row's owner, where rank q's shard of row r lands at
// outputs[owner] + (r * ranks + q) * shard as in the gather. In phase p a rank
// writes only to rank (rank + p) % ranks, and the cluster synchronizes before
// the next phase, so each GPU sends to one destination at a time. Flags hold
// one data word per (source, CTA) and one acknowledgment per (receiver, CTA);
// at the start of a call every rank acknowledges the previous call to all
// senders, and a sender waits for that before writing to a receiver on every
// call, as the gather does: the first acknowledgment, tagged(0), is not the
// zero the flags start with, and no epoch (including the wrap through zero)
// skips the wait.
constexpr uint32_t kExchangeCtas = 8;
constexpr uint32_t kExchangeData = 0;                                // [source * 8 + cta]
constexpr uint32_t kExchangeAck = push::kMaxRanks * kExchangeCtas;  // [receiver * 8 + cta]

__device__ __forceinline__ void owned_rows(uint32_t count, uint32_t ranks, uint32_t owner,
                                           uint32_t& first, uint32_t& last) {
  const uint32_t per = (count + ranks - 1) / ranks;
  first = min(owner * per, count);
  last = min(first + per, count);
}

template <typename T, typename Outputs, typename Flags>
__device__ void exchange(const T* input, Outputs outputs, Flags flags, unsigned int* epoch,
                         uint32_t count, uint64_t shard, uint32_t rank, uint32_t ranks) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  if (rank >= ranks || ranks == 0 || ranks > push::kMaxRanks || count == 0 || shard == 0)
    return;
  if (blockDim.x != kThreads || gridDim.x != kExchangeCtas) __trap();
  const uint32_t cta = blockIdx.x;
  const uint32_t epoch_value = epoch[cta] + 1;
  const uint32_t mark = push::tagged(epoch_value);
  unsigned int* local = peer(flags, rank);
  if (threadIdx.x == 0)
    for (uint32_t source = 0; source < ranks; ++source)
      if (source != rank)
        release_system(peer(flags, source) + kExchangeAck + rank * kExchangeCtas + cta,
                       push::tagged(epoch_value - 1));
  constexpr uint64_t kWidth = 16 / sizeof(T);
  const uint64_t width = shard % kWidth == 0 ? kWidth : 1;
  for (uint32_t phase = 0; phase < ranks; ++phase) {
    const uint32_t destination = (rank + phase) % ranks;
    uint32_t first, last;
    owned_rows(count, ranks, destination, first, last);
    if (first < last) {
      if (destination != rank) {
        if (threadIdx.x == 0)
          push::wait_system(local + kExchangeAck + destination * kExchangeCtas + cta,
                            push::tagged(epoch_value - 1));
        __syncthreads();
      }
      T* owner = reinterpret_cast<T*>(peer(outputs, destination));
      const uint64_t units = static_cast<uint64_t>(last - first) * shard / width;
      const uint64_t end = units * (cta + 1) / kExchangeCtas;
      for (uint64_t unit = units * cta / kExchangeCtas + threadIdx.x; unit < end;
           unit += kThreads) {
        const uint64_t index = unit * width;
        const uint64_t row = first + index / shard;
        const uint64_t column = index % shard;
        const T* from = input + row * shard + column;
        T* to = owner + (row * ranks + rank) * shard + column;
        if (width == kWidth)
          *reinterpret_cast<Vector128*>(to) = *reinterpret_cast<const Vector128*>(from);
        else *to = *from;
      }
      __syncthreads();
      if (destination != rank && threadIdx.x == 0)
        release_system(peer(flags, destination) + kExchangeData + rank * kExchangeCtas + cta,
                       mark);
    }
    push::next_destination();
  }
  uint32_t first, last;
  owned_rows(count, ranks, rank, first, last);
  if (first < last && threadIdx.x < ranks && threadIdx.x != rank)
    push::wait_system(local + kExchangeData + threadIdx.x * kExchangeCtas + cta, mark);
  __syncthreads();
  if (threadIdx.x == 0) epoch[cta] = epoch_value;
}

} // namespace chop::comm::rows

namespace chop::comm {

template <typename T, typename Outputs, typename Flags>
__device__ void gather_rows(const T* input, Outputs outputs, Flags flags, unsigned int* epoch,
                            const uint32_t* row_indices, uint32_t rows, uint64_t shard,
                            uint32_t rank, uint32_t ranks, uint32_t target) {
  rows::gather(input, outputs, flags, epoch, row_indices, rows, shard, rank, ranks, target);
}

template <typename T, typename Outputs, typename Flags>
__device__ void exchange_rows(const T* input, Outputs outputs, Flags flags, unsigned int* epoch,
                              uint32_t rows, uint64_t shard, uint32_t rank, uint32_t ranks) {
  rows::exchange(input, outputs, flags, epoch, rows, shard, rank, ranks);
}

} // namespace chop::comm
