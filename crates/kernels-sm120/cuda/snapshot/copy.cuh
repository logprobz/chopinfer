#pragma once
// Snapshot copies (`kernel_core::snapshot_copy`): a snapshot's segments
// moved between bound views and pool pages, nothing computed. One thread
// per 16-byte vector of the snapshot's bytes (README.md).
#include <stdint.h>
#include "../common/ptx.cuh"

namespace chop::snapshot {

// One table segment: `length` bytes at `offset` of view `view`, at snapshot
// byte `at` (`SnapshotCopy::encode`).
struct Segment {
  uint64_t offset;
  uint64_t at;
  uint64_t length;
  uint64_t view;
};

constexpr uint64_t kVectorBytes = 16;

// The launch's eight view parameters, each an ordinary device pointer
// (null past the views the table binds): a by-value pointer array is for
// peer addresses, and the host refuses one that names a host-owned buffer
// such as a request's state.
struct Views {
  uint8_t* slot_0;
  uint8_t* slot_1;
  uint8_t* slot_2;
  uint8_t* slot_3;
  uint8_t* slot_4;
  uint8_t* slot_5;
  uint8_t* slot_6;
  uint8_t* slot_7;
};

// View `view` of the launch; null past its eight slots, which the host's
// table check never names.
__device__ __forceinline__ uint8_t* view_at(const Views& views, uint64_t view) {
  switch (view) {
    case 0: return views.slot_0;
    case 1: return views.slot_1;
    case 2: return views.slot_2;
    case 3: return views.slot_3;
    case 4: return views.slot_4;
    case 5: return views.slot_5;
    case 6: return views.slot_6;
    case 7: return views.slot_7;
    default: return nullptr;
  }
}

// Thread v copies the parts of snapshot bytes [16 v, 16 v + 16) that the
// table's segments hold, to the pages when `to_pages`, else back. Pages of
// whole vectors keep the run inside one page. A run that is one segment's
// whole vector with both ends 16-byte aligned moves as one access; any
// other run (a segment's unaligned start, a byte tail, a vector two
// segments share) moves byte by byte. Each snapshot byte has one thread and
// each view byte at most one segment, so no two threads write one byte.
__device__ void copy(const Views& views, uint8_t* pool, const uint64_t* table,
                     uint32_t segments, uint32_t pages, uint64_t page_bytes,
                     uint32_t to_pages) {
  ptx::griddep_launch_dependents();
  // The work that wrote the views or pages, and the table upload, precede.
  ptx::griddep_wait();
  if (segments == 0 || page_bytes == 0 || page_bytes % kVectorBytes != 0) return;
  const uint64_t start =
      (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVectorBytes;
  const uint64_t page = start / page_bytes;
  if (page >= pages) return;
  const Segment* listed = reinterpret_cast<const Segment*>(table + pages);
  // The first segment that ends past the vector's start.
  uint32_t low = 0;
  uint32_t high = segments;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    if (listed[middle].at + listed[middle].length <= start)
      low = middle + 1;
    else
      high = middle;
  }
  const uint64_t end = start + kVectorBytes;
  uint8_t* paged = pool + table[page] * page_bytes + start % page_bytes;
  for (uint32_t index = low; index < segments; ++index) {
    const Segment segment = listed[index];
    if (segment.at >= end) break;
    const uint64_t from = segment.at > start ? segment.at : start;
    const uint64_t stop = segment.at + segment.length;
    const uint64_t to = stop < end ? stop : end;
    uint8_t* held = view_at(views, segment.view) + segment.offset + (from - segment.at);
    uint8_t* in_page = paged + (from - start);
    const uint8_t* source = to_pages != 0 ? held : in_page;
    uint8_t* destination = to_pages != 0 ? in_page : held;
    const uint64_t aligned = reinterpret_cast<uintptr_t>(source) |
                             reinterpret_cast<uintptr_t>(destination);
    if (to - from == kVectorBytes && aligned % kVectorBytes == 0) {
      *reinterpret_cast<uint4*>(destination) = *reinterpret_cast<const uint4*>(source);
    } else {
      for (uint64_t byte = 0; byte < to - from; ++byte) destination[byte] = source[byte];
    }
  }
}

}  // namespace chop::snapshot
