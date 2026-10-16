#pragma once

#include "row.cuh"
#include "../common/rotary.cuh"

namespace chop::row {

template<int Rotary, bool Neox, bool Table>
__device__ __forceinline__ float rotated(const uint16_t* src, int dim,
                                          int64_t position, float theta,
                                          const float* cos_sin) {
  if (dim >= Rotary) return bf(src[dim]);
  constexpr int Pairs = Rotary / 2;
  int pair = Neox ? dim % Pairs : dim / 2;
  int mate = Neox ? (dim < Pairs ? dim + Pairs : dim - Pairs) : (dim ^ 1);
  bool second = Neox ? dim >= Pairs : (dim & 1);
  const float2 cs = Table ?
      make_float2(cos_sin[(uint64_t(position) * Pairs + pair) * 2],
                  cos_sin[(uint64_t(position) * Pairs + pair) * 2 + 1]) :
      rotary::angle<Rotary>(pair, position, theta);
  if constexpr (Rotary == 64 && Neox)
    return rotary::hf_rotate(bf(src[dim]), bf(src[mate]), cs, second);
  return rotary::rotate(bf(src[dim]), bf(src[mate]), cs, second);
}

template<bool Fp8>
__device__ __forceinline__ void cache_store(void* pool, uint64_t index,
                                            float value, float scale) {
  if constexpr (Fp8) {
    uint16_t codes = convert::pack_f32_e4m3({value / scale, 0.f});
    reinterpret_cast<uint8_t*>(pool)[index] = uint8_t(codes);
  } else {
    reinterpret_cast<uint16_t*>(pool)[index] = to_bf(value);
  }
}

// Cache layout is page or slot major, then token, KV head, dimension.
// Ring prefill emits only k_new/v_new; a later ring_write replaces the tail.
template<int QHeads, int KvHeads, int Qk, int V, int Rotary,
         bool Neox, bool Table, bool Ring, bool Prefill, bool Fp8, bool Pos64,
         int PageSize>
__device__ void rope_kv(const uint16_t* qkv, uint16_t* q,
                        uint16_t* k_new, uint16_t* v_new,
                        void* k_pool, void* v_pool,
                        const int32_t* block_tables, const int32_t* slot_ids,
                        const int32_t* token_batch, const void* positions,
                        const float* cos_sin, const uint32_t* live_tokens,
                        uint32_t max_pages, uint32_t ring_size,
                        uint64_t k_page_stride_bytes, uint64_t v_page_stride_bytes,
                        float theta, float v_multiplier,
                        float k_scale, float v_scale) {
  ptx::griddep_launch_dependents();
  ptx::griddep_wait();
  uint32_t token = blockIdx.x;
  const bool grouped = gridDim.y == QHeads / 4;
  uint32_t head = grouped ? blockIdx.y * 4 + threadIdx.x / 64 : blockIdx.y;
  uint32_t lane = grouped ? threadIdx.x % 64 : threadIdx.x;
  uint32_t step = grouped ? 64 : blockDim.x;
  if (token >= *live_tokens || head >= QHeads) return;
  int64_t position = Pos64 ? reinterpret_cast<const int64_t*>(positions)[token]
                           : reinterpret_cast<const int32_t*>(positions)[token];
  const int32_t batch = token_batch[token];
  constexpr int Stride = QHeads * Qk + KvHeads * (Qk + V);
  const uint64_t base = uint64_t(token) * Stride;
  const uint16_t* q_src = qkv + base + head * Qk;
  for (int d = lane; d < Qk; d += step)
    q[(uint64_t(token) * QHeads + head) * Qk + d] =
        to_bf(rotated<Rotary, Neox, Table>(q_src, d, position, theta, cos_sin));
  if (head >= KvHeads) return;
  const uint16_t* k_src = qkv + base + QHeads * Qk + head * Qk;
  const uint16_t* v_src = qkv + base + (QHeads + KvHeads) * Qk + head * V;
  uint64_t cache_token = 0;
  void* k_destination = k_pool;
  void* v_destination = v_pool;
  if constexpr (Ring) {
    cache_token = uint64_t(slot_ids[batch]) * ring_size + uint64_t(position) % ring_size;
  } else {
    const int32_t page = block_tables[uint64_t(batch) * max_pages + position / PageSize];
    cache_token = uint64_t(position) % PageSize;
    k_destination = reinterpret_cast<uint8_t*>(k_pool) + uint64_t(page) * k_page_stride_bytes;
    v_destination = reinterpret_cast<uint8_t*>(v_pool) + uint64_t(page) * v_page_stride_bytes;
  }
  for (int d = lane; d < Qk; d += step) {
    float value = rotated<Rotary, Neox, Table>(k_src, d, position, theta, cos_sin);
    if (k_new) k_new[(uint64_t(token) * KvHeads + head) * Qk + d] = to_bf(value);
    if constexpr (!Ring || !Prefill)
      cache_store<Fp8>(k_destination, (cache_token * KvHeads + head) * Qk + d,
                       value, k_scale);
  }
  for (int d = lane; d < V; d += step) {
    float value = bf(v_src[d]) * v_multiplier;
    if (v_new) v_new[(uint64_t(token) * KvHeads + head) * V + d] = to_bf(value);
    if constexpr (!Ring || !Prefill)
      cache_store<Fp8>(v_destination, (cache_token * KvHeads + head) * V + d,
                       value, v_scale);
  }
}

template<int KvHeads, int Qk, int V, bool Fp8, bool Pos64, bool SingleSequenceTail = false>
__device__ void ring_write(const uint16_t* k_new, const uint16_t* v_new,
                           void* k_ring, void* v_ring,
                           const int32_t* slot_ids, const int32_t* token_batch,
                           const void* positions, const int64_t* last_positions,
                           const uint32_t* live_tokens,
                           uint32_t ring_size, float k_scale, float v_scale) {
  begin();
  uint32_t token = blockIdx.x, head = blockIdx.y;
  if constexpr (SingleSequenceTail) {
    const uint32_t live = *live_tokens;
    token += live > ring_size ? live - ring_size : 0;
  }
  if (token >= *live_tokens || head >= KvHeads) return;
  int64_t position = Pos64 ? reinterpret_cast<const int64_t*>(positions)[token]
                           : reinterpret_cast<const int32_t*>(positions)[token];
  if (position + ring_size <= last_positions[token_batch[token]]) return;
  uint64_t cache_token = uint64_t(slot_ids[token_batch[token]]) * ring_size +
                         uint64_t(position) % ring_size;
  for (int d = threadIdx.x; d < Qk; d += blockDim.x)
    cache_store<Fp8>(k_ring, (cache_token * KvHeads + head) * Qk + d,
                     bf(k_new[(uint64_t(token) * KvHeads + head) * Qk + d]), k_scale);
  for (int d = threadIdx.x; d < V; d += blockDim.x)
    cache_store<Fp8>(v_ring, (cache_token * KvHeads + head) * V + d,
                     bf(v_new[(uint64_t(token) * KvHeads + head) * V + d]), v_scale);
}

} // namespace chop::row
