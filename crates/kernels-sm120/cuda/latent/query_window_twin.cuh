#pragma once
// The decode query RoPE (query_parallel::rotate) and the fused window write
// (window_fused::run) of the same rows in one launch: grid rows x 2 with
// cluster (1, 2), 1024 threads. CTA y = 0 rotates the row's query heads with
// the served RoPE code; CTA y = 1 runs the served window norm, quantizer and
// store on its first 256 threads, and its other threads exit.
//
// Served, the RoPE commits its flags to the row status before the window
// write starts, and every window commit sees them. Here one cluster barrier
// orders them: every RoPE thread arrives with release semantics right after
// the RoPE's commit (its thread 0's atomic OR), the window threads arrive at
// their start, and the window's thread 0 waits on the barrier, with acquire
// semantics, before its first atomic OR (in norm_rotate). So every commit
// returns the served value and the status ends the same. Neither half reads
// what the other writes, so every rotated query, native query head, latent,
// code, scale and stored window cell is the served pair's. An idle row's
// RoPE arrives without committing. Every thread waits on the barrier once
// before it leaves, and exited threads count as arrived; no shared memory is
// read across the cluster. The RoPE releases dependents early, as served;
// the window CTA only after its stores, so the grid's dependents launch
// after the window write, as they do served.
#include "../common/ptx_control.cuh"
#include "../common/row_status.cuh"
#include "window_fused.cuh"

namespace chop::latent::query_window_twin {
// The window's first status update: wait for the RoPE's commit, then the
// served atomic OR. Later window commits follow it in the same thread.
struct WindowStatus {
  static __device__ __forceinline__ uint32_t update(uint32_t* status, uint64_t row, uint32_t flags) {
    ptx::cluster_wait_acquire();
    return chop::row::block_quantize::StatusOr::update(status, row, flags);
  }
};
} // namespace chop::latent::query_window_twin

namespace chop::row_status {
// The RoPE's commit, which every RoPE thread makes, then its arrival.
__device__ __forceinline__ uint32_t twin_rope_commit(uint32_t flags, uint32_t* status, uint64_t row) {
  const uint32_t failed = commit(flags, status, row);
  ptx::cluster_arrive_release();
  return failed;
}
} // namespace chop::row_status

// The served RoPE text with its one commit routed through twin_rope_commit.
#define commit twin_rope_commit
#include "query_parallel.cuh"
#undef commit

namespace chop::latent::query_window_twin {
template<bool Bf16>
__device__ void run(float* query, uint16_t* native_q, const float* query_cos_sin,
    uint32_t heads, const float* input, const uint16_t* weight, float* latent,
    const float* cos_sin, const uint32_t* live_rows, uint32_t eps_bits,
    uint32_t* status, uint64_t status_row_offset,
    uint8_t* quantized_codes, uint8_t* quantized_scales,
    const int32_t* positions, const int32_t* row_batch,
    const int32_t* first_positions, const int32_t* last_positions,
    uint8_t* codes, uint8_t* scales, uint32_t sequences, uint32_t slots,
    uint32_t circular, uint64_t code_stride, uint64_t scale_stride) {
  if (blockIdx.y == 0) {
    query_parallel::rotate(query, native_q, query_cos_sin, live_rows, heads, status, status_row_offset);
    if (blockIdx.x >= *live_rows) ptx::cluster_arrive_relaxed();
    ptx::cluster_wait_acquire();
    return;
  }
  if (threadIdx.x >= 256) return;
  // The window publishes nothing to the RoPE.
  ptx::cluster_arrive_relaxed();
  // window_fused::run with the window's first commit after the RoPE's.
  ptx::griddep_wait();
  if (blockIdx.x >= *live_rows) {
    ptx::griddep_launch_dependents();
    ptx::cluster_wait_acquire();
    return;
  }
  glue::norm_rotate<false, WindowStatus>(input,weight,latent,cos_sin,live_rows,__uint_as_float(eps_bits),status,status_row_offset);
  __syncthreads();
  chop::row::ceil_epilogue<512,true,0x38d1b717u>(latent,quantized_codes,quantized_scales,status,status_row_offset);
  __syncthreads();
  if constexpr (Bf16)
    glue::store_bf16<>(quantized_codes,quantized_scales,positions,row_batch,
        first_positions,last_positions,codes,scales,live_rows,sequences,slots,
        circular,code_stride,scale_stride,status,status_row_offset);
  else
    glue::store(quantized_codes,quantized_scales,positions,row_batch,
        first_positions,last_positions,codes,scales,live_rows,sequences,slots,
        circular,code_stride,scale_stride,status,status_row_offset);
  // Thread 0 waited in its first commit.
  if (threadIdx.x != 0) ptx::cluster_wait_acquire();
  ptx::griddep_launch_dependents();
}
} // namespace chop::latent::query_window_twin
