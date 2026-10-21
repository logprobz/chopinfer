#pragma once
#include <stdint.h>

namespace chop::deepseek41 {
// One CTA per row and one thread per (layer, column): the same exact integer
// hash as `engram_hash`, whose single thread walks every layer and column of
// a row in turn. Every thread walks the row's whole n-gram: the ID checks do
// not depend on the layer, so a row is bad exactly when the serial walk would
// mark it, and then every ID of the row is zero and status bit 0 is set.
__device__ void engram_hash_rows(const uint32_t* tokens, const int32_t* batch,
                                 const int32_t* starts, const int64_t* history,
                                 const uint8_t* text, const uint64_t* constants,
                                 uint32_t* ids, uint32_t* status, uint32_t rows,
                                 uint32_t vocab, uint32_t ngram, uint32_t heads,
                                 uint32_t layers, uint32_t items, uint32_t stride) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  __shared__ uint32_t row_bad;
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  if (threadIdx.x == 0) row_bad = 0;
  __syncthreads();
  const uint32_t columns = (ngram - 1) * heads;
  const bool active = threadIdx.x < layers * columns;
  const uint32_t layer = active ? threadIdx.x / columns : 0;
  const uint32_t column = active ? threadIdx.x % columns : 0;
  const uint32_t target = column / heads + 1;
  const uint64_t compressed = constants[0], pad = constants[1];
  const uint64_t* map = constants + 2;
  const uint64_t* multipliers = map + vocab;
  const uint64_t* primes = multipliers + uint64_t(layers) * ngram;
  const uint64_t* offsets = primes + uint64_t(layers) * columns;
  const uint64_t at = uint64_t(layer) * columns + column;
  bool bad = status[row] != 0 || compressed == 0 || pad >= compressed;
  const int32_t item = batch[row];
  bad |= item < 0 || uint32_t(item) >= items;
  const int32_t first = bad ? -1 : starts[item];
  bad |= first < 0 || uint32_t(first) > row;
  uint64_t hashed = 0;
  if (!bad && active) {
    const uint64_t* mul = multipliers + uint64_t(layer) * ngram;
    uint64_t rolling = 0;
    bool blocked = false;
    for (uint32_t n = 0; n < ngram; ++n) {
      uint64_t value = pad;
      if (!blocked) {
        int64_t id;
        if (n <= row - uint32_t(first)) {
          const uint32_t back = row - n;
          const uint32_t token = tokens[back];
          if (batch[back] != item || token >= vocab || text[back] > 1) { bad = true; break; }
          if (text[back] && map[token] >= compressed) { bad = true; break; }
          id = text[back] ? int64_t(map[token]) : -1;
        } else {
          const uint32_t back = n - (row - uint32_t(first)) - 1;
          id = history[uint64_t(item) * (ngram - 1) + ngram - 2 - back];
        }
        blocked = id == -1;
        if (!blocked) {
          if (id < 0 || uint64_t(id) >= compressed) { bad = true; break; }
          value = uint64_t(id);
        }
      }
      rolling ^= value * mul[n];
      if (n == target) hashed = rolling;
    }
    bad |= primes[at] == 0;
  }
  if (bad) atomicOr(&row_bad, 1u);
  __syncthreads();
  if (row_bad) {
    if (threadIdx.x == 0) status[row] |= 1;
    if (active) ids[(uint64_t(layer) * stride + row) * columns + column] = 0;
  } else if (active) {
    ids[(uint64_t(layer) * stride + row) * columns + column] =
        uint32_t(hashed % primes[at] + offsets[at]);
  }
}
} // namespace chop::deepseek41
