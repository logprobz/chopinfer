#pragma once
#include "../common/convert.cuh"
#include "../common/ptx.cuh"

namespace chop::row::replicated_embedding {
template<class Tables>
__device__ __forceinline__ const uint16_t* peer_table(Tables tables, uint32_t rank) {
  switch (rank) {
    case 0: return tables.slot_0;
    case 1: return tables.slot_1;
    case 2: return tables.slot_2;
    case 3: return tables.slot_3;
    case 4: return tables.slot_4;
    case 5: return tables.slot_5;
    case 6: return tables.slot_6;
    default: return tables.slot_7;
  }
}

// A caller can schedule any logical row with its own shared arena. This
// task needs no shared bytes and reads no communication flags or epochs.
template<int Width, int Ranks, int Streams, class Tables>
__device__ void task(Tables tables, const int32_t* ids, uint16_t* shard,
                    float* wide, uint16_t* output, uint16_t* expanded,
                    const uint32_t* live_rows, uint32_t shard_vocab,
                    uint32_t rank, uint32_t row, unsigned char* arena) {
  (void)arena;
  if (row >= *live_rows) return;
  const int64_t id = ids[row];
  const bool valid = id >= 0 && uint64_t(id) < uint64_t(shard_vocab) * Ranks;
  const uint32_t owner = valid ? uint32_t(id) / shard_vocab : 0;
  const uint32_t local = valid ? uint32_t(id) % shard_vocab : 0;
  for (int c = 4 * threadIdx.x; c < Width; c += 4 * 256) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint16_t bits = valid ? peer_table(tables, owner)[uint64_t(local) * Width + c + j] : 0;
      const uint16_t own = valid && owner == rank ? bits : 0;
      const uint64_t at = uint64_t(row) * Width + c + j;
      shard[at] = own;
      wide[at] = convert::bf16_to_f32(own);
      const float value = convert::bf16_to_f32(bits);
      float sum = 0.f;
      #pragma unroll
      for (int source = 0; source < Ranks; ++source)
        sum = __fadd_rn(sum, source == owner ? value : 0.f);
      const uint16_t reduced = convert::bf16_from_f32(sum);
      output[at] = reduced;
      #pragma unroll
      for (int stream = 0; stream < Streams; ++stream)
        expanded[(uint64_t(row) * Streams + stream) * Width + c + j] = reduced;
    }
  }
}
} // namespace chop::row::replicated_embedding
