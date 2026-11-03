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

/// Optional mapped-input sampler, loaded only by a mapped sampling submission.
#[cfg(target_os = "linux")]
pub static MAPPED_SAMPLING_CUBIN: &[u8] = include_bytes!(env!("RUNTIME_MAPPED_SAMPLING_CUBIN"));
#[cfg(not(target_os = "linux"))]
pub static MAPPED_SAMPLING_CUBIN: &[u8] = &[];

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{MAPPED_SAMPLING_CUBIN, SAMPLING_CUBIN};
    use kernel_build::{BUILD_REPORT_FILE, BuildReport, digest};
    use std::path::Path;

    #[test]
    fn both_host_sampling_reports_survive() {
        let mut directories = Vec::new();
        for (path, name, bytes) in [
            (
                env!("RUNTIME_SAMPLING_CUBIN"),
                "sampling_sm120",
                SAMPLING_CUBIN,
            ),
            (
                env!("RUNTIME_MAPPED_SAMPLING_CUBIN"),
                "sampling_mapped_sm120",
                MAPPED_SAMPLING_CUBIN,
            ),
        ] {
            let directory = Path::new(path).parent().unwrap();
            let report = BuildReport::read(&directory.join(BUILD_REPORT_FILE)).unwrap();
            assert_eq!(report.cubins.len(), 1);
            assert_eq!(report.cubins[0].name, name);
            report.verify_current().unwrap();
            assert_eq!(report.cubins[0].identity["cubin_sha256"], digest(bytes));
            directories.push(directory);
        }
        assert_ne!(directories[0], directories[1]);
    }
}
