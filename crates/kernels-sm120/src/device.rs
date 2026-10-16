//! The target device's limits: the RTX PRO 6000 Blackwell (sm_120, 188
//! SMs), for the occupancy arithmetic in `kernel_core::occupancy`.
//!
//! An entry keeps `ctas_per_sm` CTAs on an SM at once, the least of what its
//! registers, threads and shared memory allow. Clusters must also sit within
//! one GPC, so the driver keeps fewer clusters resident than the SMs divide
//! into: measured with `cuOccupancyMaxActiveClusters` on the target, clusters
//! of 8 CTAs are 22 at one CTA per SM (not 23) and 46 at two (not 47), and
//! clusters of 4 are 94 at two. [`TARGET`] takes 22 clusters of 8 per CTA
//! slot and scales it by cluster size, a lower bound on every one of those
//! measurements (22, 44 and 88).

use kernel_core::occupancy::Device;

/// The target's SM count and per-SM limits: 65,536 registers (allocated
/// per warp in units of 256), 102,400 bytes of shared memory (1,024
/// reserved per CTA), 1,536 threads and 32 CTA slots.
pub const TARGET: Device = Device {
    multiprocessors: 188,
    registers_per_sm: 65_536,
    register_unit: 256,
    shared_bytes_per_sm: 102_400,
    shared_reserved_per_cta: 1024,
    threads_per_sm: 1536,
    ctas_per_sm: 32,
    clusters_of_8_per_slot: 22,
};

/// The clusters of 8 CTAs the target keeps resident at two CTAs per SM, as
/// measured (kernel-lab, 2026-09-26): what its driver reports for the comm
/// entries.
pub const MEASURED_CLUSTERS_OF_8_AT_TWO_CTAS: u32 = 46;

#[cfg(test)]
mod tests {
    use kernel_core::occupancy::EntryResources;

    use super::*;

    fn entry(registers: u32, shared_static_bytes: u32) -> EntryResources {
        EntryResources {
            registers,
            shared_static_bytes,
        }
    }

    /// The target's measured counts: a decode all-reduce entry at 127
    /// registers, 21584 bytes of shared memory and 256 threads keeps 46
    /// clusters of 8 resident. Another entry (114 registers, 33864 bytes)
    /// keeps two CTAs per SM.
    #[test]
    fn the_model_bounds_the_measured_residency() {
        let per_rank = TARGET.ctas_per_sm(entry(127, 21_584), 256, 0);
        assert_eq!(per_rank, 2);
        assert_eq!(TARGET.resident_clusters(1, 8), 22);
        assert!(TARGET.resident_clusters(2, 8) <= MEASURED_CLUSTERS_OF_8_AT_TWO_CTAS);
        assert!(TARGET.resident_clusters(2, 4) <= 94);
        assert_eq!(TARGET.ctas_per_sm(entry(114, 33_864), 256, 0), 2);
        assert_eq!(TARGET.ctas_per_sm(entry(128, 33_864), 256, 0), 2);
        // Threads bound it too: 1536 per SM.
        assert_eq!(TARGET.ctas_per_sm(entry(16, 0), 1024, 0), 1);
    }
}
