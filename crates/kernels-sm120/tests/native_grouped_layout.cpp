// Host emulation of the native grouped tile's operand path: producer chunk
// stores, lane fragment loads and E2M1 expansion through the device
// header, into a model of the PTX m16n8k32 fragment layout. Each emulated
// group must equal its exact dot product in natural K order, and the warp
// loads must be bank-conflict free.
#include "moe/native_grouped_layout.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>
namespace lay = chop::moe::native_grouped::layout;

double e2m1(uint32_t byte_container) {
  // The instruction reads the code from bits 5:2 and needs the rest zero.
  assert((byte_container & 0xc3) == 0);
  const uint32_t code = (byte_container >> 2) & 15;
  const double magnitude[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (code & 8 ? -1 : 1) * magnitude[code & 7];
}
double e4m3(uint32_t code) {
  const int exponent = (code >> 3) & 15, mantissa = code & 7;
  const double value = exponent ? std::ldexp(1 + mantissa / 8.0, exponent - 7)
                                : std::ldexp(mantissa / 8.0, -6);
  return code & 0x80 ? -value : value;
}
double packed_e2m1(const uint8_t* row, int k) {
  const uint32_t code = (row[k / 2] >> (4 * (k & 1))) & 15;
  return e2m1(code << 2);
}

// Emulate whole stages: 16 weight rows by 8 routed rows over 128 K, with
// the producer's chunk stores, each lane's loads and the PTX layout:
// A register i holds row g + 8 (i & 1), K slots 4t + 16 (i >> 1) + byte;
// B register i holds column g, K slots 4t + 16 i + byte; D element i is
// A row g + 8 (i >> 1) by B column 2t + (i & 1). Returns the number of
// 32-K groups whose every element equals its exact natural-order dot
// product. `natural` loads activations in K order instead (a control).
int emulate(bool natural) {
  std::mt19937 rng(0x6e6174);
  int matched = 0;
  for (int trial = 0; trial < 200; ++trial) {
    std::vector<uint8_t> w_global(16 * 64), a_global(8 * 128);
    for (auto& b : w_global) b = uint8_t(rng());
    for (auto& b : a_global) {
      b = uint8_t(rng());
      if ((b & 0x7f) == 0x7f) b ^= 1;
    }
    std::vector<uint8_t> w_stage(16 * lay::WeightRowBytes), a_stage(8 * lay::ActRowBytes);
    for (int w = 0; w < 16; ++w)
      for (int q = 0; q < 4; ++q)
        std::memcpy(&w_stage[lay::weight_store(w, q)], &w_global[w * 64 + q * 16], 16);
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c)
        std::memcpy(&a_stage[lay::act_store(r, c)], &a_global[r * 128 + c * 16], 16);
    for (int q = 0; q < 4; ++q) {
      uint32_t a[32][4], b[32][2];
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane / 4, t = lane % 4;
        uint32_t lo, hi;
        std::memcpy(&lo, &w_stage[lay::weight_load(g, q, t)], 4);
        std::memcpy(&hi, &w_stage[lay::weight_load(g + 8, q, t)], 4);
        lay::expand_e2m1x8(lo, a[lane][0], a[lane][2]);
        lay::expand_e2m1x8(hi, a[lane][1], a[lane][3]);
        if (natural) {
          std::memcpy(&b[lane][0], &a_global[g * 128 + q * 32 + 4 * t], 4);
          std::memcpy(&b[lane][1], &a_global[g * 128 + q * 32 + 16 + 4 * t], 4);
        } else {
          std::memcpy(&b[lane][0], &a_stage[lay::act_load(g, q, t)], 4);
          std::memcpy(&b[lane][1], &a_stage[lay::act_load(g, q, t) + 4], 4);
        }
      }
      double d[16][8] = {};
      for (int row = 0; row < 16; ++row)
        for (int col = 0; col < 8; ++col)
          for (int slot = 0; slot < 32; ++slot) {
            const int t = (slot % 16) / 4, byte = slot % 4;
            const uint32_t av = a[4 * (row % 8) + t][(row >= 8) + 2 * (slot >= 16)];
            const uint32_t bv = b[4 * col + t][slot >= 16];
            d[row][col] += e2m1((av >> (8 * byte)) & 255) * e4m3((bv >> (8 * byte)) & 255);
          }
      bool equal = true;
      for (int lane = 0; lane < 32; ++lane)
        for (int i = 0; i < 4; ++i) {
          const int row = lay::d_weight_row(lane / 4, i), col = lay::d_route(lane % 4, i);
          double exact = 0;
          for (int k = 0; k < 32; ++k)
            exact += packed_e2m1(&w_global[row * 64 + q * 16], k) *
                     e4m3(a_global[col * 128 + q * 32 + k]);
          // Both sums are exact in F64: 32 terms of at most 6 significant bits.
          equal &= d[row][col] == exact;
        }
      matched += equal;
    }
  }
  return matched;
}

int main() {
  // The expansion places element j of a packed word at byte j of low/high.
  std::mt19937 words(0x65326d31);
  for (int sample = 0; sample < (1 << 20) + 2; ++sample) {
    const uint32_t word = sample == 0 ? 0u : sample == 1 ? ~0u : uint32_t(words());
    uint32_t low, high;
    lay::expand_e2m1x8(word, low, high);
    for (int j = 0; j < 8; ++j) {
      const uint32_t expected = (word >> (4 * j)) & 15;
      const uint32_t got = ((j < 4 ? low : high) >> (8 * (j & 3))) & 255;
      assert(got == expected << 2);
    }
  }
  // NaN detection on every byte value in every position.
  for (int code = 0; code < 256; ++code)
    for (int at = 0; at < 4; ++at) {
      const uint32_t word = uint32_t(code) << (8 * at) | (0x3e3e3e3eu & ~(0xffu << (8 * at)));
      assert((lay::e4m3_nan(word) != 0) == ((code & 0x7f) == 0x7f));
    }
  // Scale words of unaligned rows (18 bytes per row, as K = 576 has).
  {
    std::vector<uint8_t> scales(18 * 9);
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = uint8_t(i * 7 + 3);
    for (int row = 0; row < 8; ++row)
      for (int block = 0; block < 5; ++block) {
        const uint64_t at = uint64_t(row) * 18 + 4 * block, aligned = at & ~uint64_t(3);
        uint32_t first, second;
        std::memcpy(&first, &scales[aligned], 4);
        std::memcpy(&second, &scales[aligned + 4], 4);
        const uint32_t word = lay::scale_word(first, second, int((uint64_t(row) * 18) & 3));
        for (int q = 0; q < 4 && 4 * block + q < 18; ++q)
          assert(((word >> (8 * q)) & 255) == scales[at + q]);
      }
  }
  // Bank conflicts: a warp's fragment loads of eight rows.
  for (int base = 0; base < 128; base += 8)
    for (int q = 0; q < 4; ++q) {
      // Eight-byte activation loads run as two half-warp phases.
      for (int phase = 0; phase < 2; ++phase) {
        std::set<int> banks;
        for (int lane = 16 * phase; lane < 16 * phase + 16; ++lane) {
          const int offset = lay::act_load(base + lane / 4, q, lane % 4);
          banks.insert((offset / 4) % 32);
          banks.insert((offset / 4 + 1) % 32);
        }
        assert(banks.size() == 32);
      }
      std::set<int> banks;
      for (int lane = 0; lane < 32; ++lane)
        banks.insert((lay::weight_load(base + lane / 4, q, lane % 4) / 4) % 32);
      assert(banks.size() == 32);
    }

  assert(emulate(false) == 800);
  // Negative control: natural-order activation fragments must disagree.
  assert(emulate(true) == 0);
  std::printf("NATIVE_LAYOUT_COMPLETE groups=800 negative_control_rejected=1\n");
  return 0;
}
