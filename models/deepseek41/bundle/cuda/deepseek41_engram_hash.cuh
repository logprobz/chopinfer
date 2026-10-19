#pragma once
#include <stdint.h>

namespace chop::deepseek41 {
__device__ void engram_hash(const uint32_t* tokens, const int32_t* batch,
                            const int32_t* starts, const int64_t* history,
                            const uint8_t* text, const uint64_t* constants,
                            uint32_t* ids, uint32_t* status, uint32_t rows,
                            uint32_t vocab, uint32_t ngram, uint32_t heads,
                            uint32_t layers, uint32_t items, uint32_t stride) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const uint32_t columns = (ngram - 1) * heads;
  const uint64_t compressed = constants[0], pad = constants[1];
  const uint64_t* map = constants + 2;
  const uint64_t* multipliers = map + vocab;
  const uint64_t* primes = multipliers + uint64_t(layers) * ngram;
  const uint64_t* offsets = primes + uint64_t(layers) * columns;
  bool bad = status[row] != 0 || compressed == 0 || pad >= compressed;
  const int32_t item = batch[row];
  bad |= item < 0 || uint32_t(item) >= items;
  const int32_t first = bad ? -1 : starts[item];
  bad |= first < 0 || uint32_t(first) > row;
  if (!bad) {
    for (uint32_t layer = 0; layer < layers; ++layer) {
      const uint64_t* mul = multipliers + uint64_t(layer) * ngram;
      uint64_t rolling = 0;
      bool blocked = false;
      for (uint32_t n = 0; n < ngram; ++n) {
        uint64_t value = pad;
        if (!blocked) {
          int64_t id;
          if (n <= row - uint32_t(first)) {
            const uint32_t at = row - n;
            const uint32_t token = tokens[at];
            if (batch[at] != item || token >= vocab || text[at] > 1) { bad = true; break; }
            // Raw IDs are checked even when the position is non-text.
            if (text[at] && map[token] >= compressed) { bad = true; break; }
            id = text[at] ? int64_t(map[token]) : -1;
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
        if (n == 0) continue;
        for (uint32_t head = 0; head < heads; ++head) {
          const uint32_t column = (n - 1) * heads + head;
          const uint64_t at = uint64_t(layer) * columns + column;
          const uint64_t prime = primes[at];
          if (prime == 0) { bad = true; continue; }
          ids[(uint64_t(layer) * stride + row) * columns + column] = uint32_t(rolling % prime + offsets[at]);
        }
      }
      if (bad) break;
    }
  }
  if (bad) {
    status[row] |= 1;
    for (uint32_t layer = 0; layer < layers; ++layer)
      for (uint32_t column = 0; column < columns; ++column)
        ids[(uint64_t(layer) * stride + row) * columns + column] = 0;
  }
}
} // namespace chop::deepseek41
