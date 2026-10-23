// Host checks of the quad-lane gate queues (moe/chained_lanes.cuh): the
// word-wide weight and activation code checks against per-code rules, and
// the lane operand bytes against the row-tiled queues' shuffled bytes.
#include "moe/chained_lanes_checks.h"
#include <cstdio>
#include <cstdlib>
#include <random>
namespace checks = chop::moe::lanes::checks;

#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// streamed::check's rule per E2M1 code: UE8M0 255 is never valid, 253
// forbids magnitudes 6 and 7, 254 magnitudes 4 and above.
bool code_valid(uint32_t code, uint32_t scale) {
  const uint32_t magnitude = code & 7;
  return !(scale == 255 || (scale == 253 && magnitude >= 6) || (scale == 254 && magnitude >= 4));
}

bool group_naive(uint32_t codes, uint32_t scale) {
  for (int i = 0; i < 8; ++i)
    if (!code_valid((codes >> (4 * i)) & 15, scale)) return false;
  return true;
}

bool extreme_naive(uint32_t scales) {
  for (int i = 0; i < 4; ++i)
    if (((scales >> (8 * i)) & 255) > 252) return true;
  return false;
}

bool nan_naive(uint32_t codes) {
  for (int i = 0; i < 4; ++i)
    if (((codes >> (8 * i)) & 0x7f) == 0x7f) return true;
  return false;
}

// kernel_core::transform::QuadLaneLayout::offset within one block.
int lane_offset(int at) { return (at % 4) * 16 + (at / 16) * 4 + (at / 4) % 4; }

int main() {
  std::mt19937_64 random(0x6c616e6573);
  long words = 0, groups = 0, operands = 0;
  // Every byte value at every position of a scale word, beside edge and
  // random bytes elsewhere.
  const uint32_t edges[] = {0, 1, 127, 252, 253, 254, 255};
  for (int position = 0; position < 4; ++position)
    for (uint32_t value = 0; value < 256; ++value)
      for (uint32_t other : edges)
        for (int trial = 0; trial < 4; ++trial) {
          uint32_t scales = trial == 0 ? other * 0x01010101u : uint32_t(random());
          scales = (scales & ~(255u << (8 * position))) | (value << (8 * position));
          REQUIRE((checks::extreme_scales(scales) != 0) == extreme_naive(scales));
          uint32_t codes[4];
          for (auto& c : codes) c = uint32_t(random());
          if (trial == 3) codes[position] = 0x77777777u;
          bool naive = true;
          for (int g = 0; g < 4; ++g) naive = naive && group_naive(codes[g], (scales >> (8 * g)) & 255);
          REQUIRE(checks::weights_valid(codes, scales) == naive);
          ++words;
        }
  // Every code pair at every scale byte, the group rule.
  for (uint32_t scale = 0; scale < 256; ++scale)
    for (uint32_t low = 0; low < 16; ++low)
      for (uint32_t high = 0; high < 16; ++high)
        for (int position = 0; position < 7; ++position) {
          const uint32_t codes = (low | high << 4) << (4 * position);
          REQUIRE(checks::group_valid(codes, scale) == group_naive(codes, scale));
          ++groups;
        }
  // Every E4M3 code at every position.
  for (int position = 0; position < 4; ++position)
    for (uint32_t code = 0; code < 256; ++code)
      for (int trial = 0; trial < 64; ++trial) {
        uint32_t word = uint32_t(random()) & 0x7e7e7e7eu;
        word = (word & ~(255u << (8 * position))) | (code << (8 * position));
        REQUIRE((checks::e4m3_nan(word) != 0) == nan_naive(word));
        ++words;
      }
  for (int trial = 0; trial < 1 << 20; ++trial) {
    const uint32_t word = uint32_t(random());
    REQUIRE((checks::e4m3_nan(word) != 0) == nan_naive(word));
    REQUIRE((checks::extreme_scales(word) != 0) == extreme_naive(word));
    ++words;
  }
  for (uint32_t scale = 0; scale < 256; ++scale)
    REQUIRE(checks::extreme_activation_scale(scale) == (scale < 3 || scale > 246));
  // The operand bytes: the row-tiled queue converts byte t of the owner
  // lane's word 2 * half + w (w = 0 low, 1 high) of group `group`, where
  // owner `group` loaded block bytes 16 * group .. 16 * group + 15; the
  // quad-lane queue converts byte 2 * half + w of its own word `group`.
  for (int trial = 0; trial < 64; ++trial) {
    uint8_t block[64], lanes[64];
    for (auto& b : block) b = uint8_t(random());
    for (int at = 0; at < 64; ++at) lanes[lane_offset(at)] = block[at];
    for (int t = 0; t < 4; ++t)
      for (int group = 0; group < 4; ++group)
        for (int half = 0; half < 2; ++half)
          for (int w = 0; w < 2; ++w) {
            const int word = 2 * half + w;
            const uint8_t tiled = block[16 * group + 4 * word + t];
            const uint8_t lane = lanes[16 * t + 4 * group + word];
            REQUIRE(tiled == lane);
            ++operands;
          }
  }
  std::printf("LANES_CHECKS_COMPLETE words=%ld groups=%ld operands=%ld\n", words, groups, operands);
  return 0;
}
