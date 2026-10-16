//! How many CTAs and clusters of an entry a device keeps resident at once,
//! from the entry's audited resources and the device's per-SM limits. A
//! kernel set states its target's limits ([`Device`]); the arithmetic here
//! names no target.

/// One entry's resources as the compiler reported them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EntryResources {
    pub registers: u32,
    pub shared_static_bytes: u32,
}

/// A device's SM count and per-SM limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Device {
    pub multiprocessors: u32,
    pub registers_per_sm: u32,
    /// Registers are allocated per warp in units of this many.
    pub register_unit: u32,
    pub shared_bytes_per_sm: u32,
    /// Shared memory the driver reserves per CTA.
    pub shared_reserved_per_cta: u32,
    pub threads_per_sm: u32,
    pub ctas_per_sm: u32,
    /// Clusters of 8 CTAs kept resident per CTA an SM holds. Clusters sit
    /// within one GPC, so this is fewer than the SMs divide into; the
    /// device's owner states a lower bound on what its driver measures.
    pub clusters_of_8_per_slot: u32,
}

impl Device {
    /// CTAs of `threads` threads an SM holds at once with `resources` and
    /// `dynamic_shared` bytes of dynamic shared memory: the least its
    /// registers, threads, shared memory and CTA slots allow.
    #[must_use]
    pub fn ctas_per_sm(&self, resources: EntryResources, threads: u32, dynamic_shared: u32) -> u32 {
        let warps = threads.div_ceil(32).max(1);
        let per_warp =
            (resources.registers.max(1) * 32).div_ceil(self.register_unit) * self.register_unit;
        let by_registers = self.registers_per_sm / (per_warp * warps);
        // An entry that uses no shared memory, on a device that reserves
        // none per CTA, is not bounded by it.
        let shared = resources.shared_static_bytes + dynamic_shared + self.shared_reserved_per_cta;
        let by_shared = self
            .shared_bytes_per_sm
            .checked_div(shared)
            .unwrap_or(u32::MAX);
        let by_threads = self.threads_per_sm / (warps * 32);
        by_registers
            .min(by_shared)
            .min(by_threads)
            .min(self.ctas_per_sm)
    }

    /// The clusters of `cluster` CTAs the device keeps resident at
    /// `ctas_per_sm` CTAs per SM: [`Device::clusters_of_8_per_slot`] per CTA
    /// slot, scaled by cluster size.
    #[must_use]
    pub fn resident_clusters(&self, ctas_per_sm: u32, cluster: u32) -> u32 {
        let cluster = cluster.max(1);
        self.clusters_of_8_per_slot * ctas_per_sm * 8 / cluster
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEVICE: Device = Device {
        multiprocessors: 4,
        registers_per_sm: 8192,
        register_unit: 256,
        shared_bytes_per_sm: 16_384,
        shared_reserved_per_cta: 1024,
        threads_per_sm: 512,
        ctas_per_sm: 3,
        clusters_of_8_per_slot: 1,
    };

    fn entry(registers: u32, shared_static_bytes: u32) -> EntryResources {
        EntryResources {
            registers,
            shared_static_bytes,
        }
    }

    #[test]
    fn the_scarcest_resource_bounds_resident_ctas() {
        // Registers: 32 per thread over 4 warps take 4096 of 8192.
        assert_eq!(DEVICE.ctas_per_sm(entry(32, 0), 128, 0), 2);
        // A register count rounds up to the allocation unit per warp.
        assert_eq!(DEVICE.ctas_per_sm(entry(33, 0), 128, 0), 1);
        // Shared memory, with the reserved bytes per CTA.
        assert_eq!(DEVICE.ctas_per_sm(entry(8, 3072), 64, 1024), 3);
        assert_eq!(DEVICE.ctas_per_sm(entry(8, 3072), 64, 2048), 2);
        // Threads, and the CTA slots.
        assert_eq!(DEVICE.ctas_per_sm(entry(1, 0), 512, 0), 1);
        assert_eq!(DEVICE.ctas_per_sm(entry(1, 0), 32, 0), 3);
    }

    #[test]
    fn no_shared_memory_on_a_device_reserving_none_imposes_no_bound() {
        let device = Device {
            shared_reserved_per_cta: 0,
            ..DEVICE
        };
        assert_eq!(device.ctas_per_sm(entry(8, 0), 64, 0), 3);
        // Any shared memory still bounds it.
        assert_eq!(device.ctas_per_sm(entry(8, 8192), 64, 0), 2);
    }

    #[test]
    fn resident_clusters_scale_by_cluster_size() {
        assert_eq!(DEVICE.resident_clusters(2, 8), 2);
        assert_eq!(DEVICE.resident_clusters(2, 4), 4);
        assert_eq!(DEVICE.resident_clusters(1, 0), 8);
    }
}
