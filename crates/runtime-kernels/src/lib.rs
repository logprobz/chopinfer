#![forbid(unsafe_code)]

pub use runtime_kernel_contract::*;

pub const TARGET_ARCHITECTURE: &str = "sm_120f";

pub const SAMPLING_SIMPLE_PARTIALS_F32_SYMBOL: &str = "chop_sampling_simple_partials_f32_v1";
pub const SAMPLING_SIMPLE_TOP_PARTIALS_F32_SYMBOL: &str =
    "chop_sampling_simple_top_partials_f32_v1";
pub const SAMPLING_SIMPLE_FINISH_F32_SYMBOL: &str = "chop_sampling_simple_finish_f32_v1";
pub const SAMPLING_SIMPLE_PARTIALS_BF16_SYMBOL: &str = "chop_sampling_simple_partials_bf16_v1";
pub const SAMPLING_SIMPLE_TOP_PARTIALS_BF16_SYMBOL: &str =
    "chop_sampling_simple_top_partials_bf16_v1";
pub const SAMPLING_SIMPLE_FINISH_BF16_SYMBOL: &str = "chop_sampling_simple_finish_bf16_v1";
pub const SAMPLING_GREEDY_PARTIALS_1024_F32_SYMBOL: &str =
    "chop_sampling_greedy_partials_1024_f32_v1";
pub const SAMPLING_GREEDY_PARTIALS_1024_BF16_SYMBOL: &str =
    "chop_sampling_greedy_partials_1024_bf16_v1";
/// Filtered sampling stages in launch order: stats, histogram, collect,
/// resolve, gumbel and finish.
pub const SAMPLING_FILTERED_F32_SYMBOLS: [&str; 6] = [
    "chop_sampling_filtered_stats_f32_v1",
    "chop_sampling_filtered_histogram_f32_v1",
    "chop_sampling_filtered_collect_f32_v1",
    "chop_sampling_filtered_resolve_f32_v1",
    "chop_sampling_filtered_gumbel_f32_v1",
    "chop_sampling_filtered_finish_f32_v1",
];
pub const SAMPLING_FILTERED_BF16_SYMBOLS: [&str; 6] = [
    "chop_sampling_filtered_stats_bf16_v1",
    "chop_sampling_filtered_histogram_bf16_v1",
    "chop_sampling_filtered_collect_bf16_v1",
    "chop_sampling_filtered_resolve_bf16_v1",
    "chop_sampling_filtered_gumbel_bf16_v1",
    "chop_sampling_filtered_finish_bf16_v1",
];
/// Threads per block for each filtered stage.
pub const SAMPLING_FILTERED_THREADS: [u32; 6] = [256, 256, 256, 1024, 256, 32];

/// Histogram blocks per row, as `kernels_sm120::sampling::filtered_histogram_groups`.
pub const fn sampling_filtered_histogram_groups(tiles: u32, rows: u32) -> u32 {
    let spread = 384_u32.div_ceil(if rows == 0 { 1 } else { rows });
    let spread = if spread < tiles { spread } else { tiles };
    let fewest = tiles.div_ceil(4);
    if fewest > spread { fewest } else { spread }
}
#[cfg(target_os = "linux")]
pub static SAMPLING_CUBIN: &[u8] = include_bytes!(env!("RUNTIME_SAMPLING_CUBIN"));

#[cfg(not(target_os = "linux"))]
pub static SAMPLING_CUBIN: &[u8] = &[];
