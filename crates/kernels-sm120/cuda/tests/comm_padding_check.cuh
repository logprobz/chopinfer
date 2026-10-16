#pragma once

// Compile-time check for tests/comm_extension.rs: in the padded one-shot of
// a shape without LL (the stub epilogue over F32 rows 5120 wide, four
// streams), padding_cta marks exactly the CTAs past the sending cluster and
// every (row, stream) group, at every prepared row count.

#include "comm/hyper_connection_decode.cuh"
#include "comm/input/f32.cuh"
#include "tests/comm_epilogue_stub.cuh"

namespace stub {
template <class R>
constexpr bool padding_matches_the_plan() {
  for (uint32_t rows = 1; rows <= chop::comm::qwen38_decode::kMaxRows; ++rows) {
    const uint32_t grid = chop::comm::qwen38_decode::kSenders + R::group_ctas(rows);
    if (R::group_ctas(rows) % chop::comm::qwen38_decode::kSenders != 0) return false;
    for (uint32_t block = 0; block < grid; ++block) {
      const bool expected = block >= chop::comm::qwen38_decode::kSenders + rows * R::kStreams;
      if (R::padding_cta(block, rows) != expected) return false;
    }
  }
  return true;
}
using PaddedRows =
    chop::comm::qwen38_decode::Rows<Threads_<256>::Epilogue<5120, 4>, chop::comm::input::F32>;
static_assert(PaddedRows::kPadsGroups && padding_matches_the_plan<PaddedRows>(),
              "padding_cta marks exactly the padding CTAs");
}
