//! Device memory reports for host backends.

/// Device allocation counters needed by a probe receipt.
///
/// These are allocation bytes owned by this backend, not a claim about total
/// board use. The current count is sampled after instance creation and the
/// high-water count is sampled after the graph completes.
pub trait HostDeviceMemoryUsageV1 {
    fn current_device_memory_bytes(&self) -> u64;
    fn peak_device_memory_bytes(&self) -> u64;
    fn materialized_device_memory_bytes(&self) -> u64;
}

/// Who measured one per-rank memory report.
///
/// A device-free backend says so rather than reporting an empty board, because
/// a zero that came from measuring nothing reads exactly like a free device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostDeviceMemoryMeasurementV1 {
    /// The device driver answered in each rank's own context.
    DriverMeasured,
    /// A backend that holds no device and reports only what it tracked itself.
    DeviceFree,
}

impl HostDeviceMemoryMeasurementV1 {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DriverMeasured => "driver-measured",
            Self::DeviceFree => "device-free",
        }
    }
}

/// One rank's device memory as the backend holding its context observed it.
///
/// `free_bytes` and `total_bytes` are absent when this backend measured no
/// device. Allocation counts are this backend's own bookkeeping for the rank,
/// so they are always stated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostDeviceRankMemoryV1 {
    rank: u32,
    device_ordinal: i32,
    free_bytes: Option<u64>,
    total_bytes: Option<u64>,
    allocated_bytes: u64,
    peak_allocated_bytes: u64,
}

impl HostDeviceRankMemoryV1 {
    #[must_use]
    pub const fn new(
        rank: u32,
        device_ordinal: i32,
        free_bytes: Option<u64>,
        total_bytes: Option<u64>,
        allocated_bytes: u64,
        peak_allocated_bytes: u64,
    ) -> Self {
        Self {
            rank,
            device_ordinal,
            free_bytes,
            total_bytes,
            allocated_bytes,
            peak_allocated_bytes,
        }
    }

    #[must_use]
    pub const fn rank(self) -> u32 {
        self.rank
    }

    #[must_use]
    pub const fn device_ordinal(self) -> i32 {
        self.device_ordinal
    }

    #[must_use]
    pub const fn free_bytes(self) -> Option<u64> {
        self.free_bytes
    }

    #[must_use]
    pub const fn total_bytes(self) -> Option<u64> {
        self.total_bytes
    }

    #[must_use]
    pub const fn allocated_bytes(self) -> u64 {
        self.allocated_bytes
    }

    #[must_use]
    pub const fn peak_allocated_bytes(self) -> u64 {
        self.peak_allocated_bytes
    }
}

/// Every rank one backend currently holds a context for, and who measured them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostDeviceRankMemoryReportV1 {
    measurement: HostDeviceMemoryMeasurementV1,
    ranks: Vec<HostDeviceRankMemoryV1>,
}

impl HostDeviceRankMemoryReportV1 {
    #[must_use]
    pub const fn new(
        measurement: HostDeviceMemoryMeasurementV1,
        ranks: Vec<HostDeviceRankMemoryV1>,
    ) -> Self {
        Self { measurement, ranks }
    }

    #[must_use]
    pub const fn measurement(&self) -> HostDeviceMemoryMeasurementV1 {
        self.measurement
    }

    #[must_use]
    pub fn ranks(&self) -> &[HostDeviceRankMemoryV1] {
        &self.ranks
    }
}

/// A backend that can name each rank's own device memory.
///
/// A measurement this backend cannot take is a refusal, never a zero, so a
/// receipt never carries a memory observation nobody made.
///
/// This describes the device at the instant it is called and nothing else. A
/// caller that measures after a model is resident reads free memory with that
/// model already in it: that is not the memory that was available before the
/// load, and it is not a budget for a later one. A producer that needs either
/// has to measure before allocating, which no measurement taken afterwards can
/// reconstruct. State when the call was made alongside its numbers.
pub trait HostDeviceRankMemoryUsageV1 {
    fn rank_memory_v1(&self) -> Result<HostDeviceRankMemoryReportV1, String>;
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HostDeviceProbeServiceFaultV1 {
    ContextAcquire,
}

#[cfg(test)]
mod tests {
    use super::{
        HostDeviceMemoryMeasurementV1, HostDeviceMemoryUsageV1, HostDeviceProbeServiceFaultV1,
        HostDeviceRankMemoryUsageV1,
    };
    use crate::fake::FakeHostDeviceV1;
    use crate::services::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
    use runtime_bundle_abi::{BundleStatusV1, HostServicesBackendV1};

    #[test]
    fn a_device_free_backend_reports_tracked_ranks_and_no_board() {
        let config = HostInstanceConfigV1::new(
            HostTopologyV1::new(vec![0, 1]).expect("the test topology is valid"),
        );
        let backend = HostServicesV1Backend::new(FakeHostDeviceV1::new(), config);
        assert!(
            backend
                .with_device(HostDeviceRankMemoryUsageV1::rank_memory_v1)
                .is_err(),
            "a backend holding no context measures no rank"
        );
        let first = backend.select_context(0, 0).expect("context");
        let second = backend.select_context(1, 1).expect("context");
        let pool = backend.select_memory_pool(second).expect("pool");
        let stream = backend.create_stream(second, 0).expect("stream");
        let buffer = backend
            .allocate_device_buffer(pool, stream, 4096, 4096)
            .expect("buffer");

        let report = backend
            .with_device(HostDeviceRankMemoryUsageV1::rank_memory_v1)
            .expect("a device-free backend still reports what it tracked");
        assert_eq!(
            report.measurement(),
            HostDeviceMemoryMeasurementV1::DeviceFree
        );
        assert_eq!(report.ranks().len(), 2);
        assert_eq!(report.ranks()[0].rank(), 0);
        assert_eq!(report.ranks()[0].device_ordinal(), 0);
        assert_eq!(report.ranks()[0].allocated_bytes(), 0);
        assert_eq!(report.ranks()[1].rank(), 1);
        assert_eq!(report.ranks()[1].allocated_bytes(), 4096);
        assert_eq!(report.ranks()[1].peak_allocated_bytes(), 4096);
        for rank in report.ranks() {
            assert_eq!(rank.free_bytes(), None);
            assert_eq!(rank.total_bytes(), None);
        }

        backend
            .release_device_buffer(buffer)
            .expect("buffer release");
        let released = backend
            .with_device(HostDeviceRankMemoryUsageV1::rank_memory_v1)
            .expect("report");
        assert_eq!(released.ranks()[1].allocated_bytes(), 0);
        assert_eq!(released.ranks()[1].peak_allocated_bytes(), 4096);
        backend.release_stream(stream).expect("stream release");
        backend.release_memory_pool(pool).expect("pool release");
        backend.release_context(second).expect("context release");
        backend.release_context(first).expect("context release");
        assert_eq!(backend.live_resource_count(), 0);
    }

    #[test]
    fn instance_create_fault_runs_after_context_ownership_and_cleans_it() {
        let config = HostInstanceConfigV1::new(
            HostTopologyV1::new(vec![0]).expect("the test topology is valid"),
        );
        let backend = HostServicesV1Backend::new_with_probe_fault(
            FakeHostDeviceV1::new(),
            config,
            HostDeviceProbeServiceFaultV1::ContextAcquire,
        );

        assert_eq!(backend.select_context(0, 0), Err(BundleStatusV1::HostError));
        assert!(backend.probe_fault_observed());
        assert_eq!(backend.live_resource_count(), 0);
        assert!(
            backend
                .with_device(FakeHostDeviceV1::context_objects)
                .is_empty()
        );
    }

    #[test]
    fn sparse_probe_allocation_keeps_rank_scale_bytes_logical() {
        const RANK_BYTES: u64 = 87 * 1024 * 1024 * 1024;
        let config = HostInstanceConfigV1::new(
            HostTopologyV1::new(vec![0]).expect("the test topology is valid"),
        );
        let mut device = FakeHostDeviceV1::new();
        device.use_sparse_probe_allocations(true);
        let backend = HostServicesV1Backend::new(device, config);
        let context = backend.select_context(0, 0).expect("context");
        let pool = backend.select_memory_pool(context).expect("pool");
        let stream = backend.create_stream(context, 0).expect("stream");
        let buffer = backend
            .allocate_device_buffer(pool, stream, RANK_BYTES, 4096)
            .expect("logical rank allocation");
        assert_eq!(
            backend.with_device(HostDeviceMemoryUsageV1::current_device_memory_bytes),
            RANK_BYTES
        );
        assert_eq!(
            backend.with_device(HostDeviceMemoryUsageV1::materialized_device_memory_bytes),
            0
        );
        backend
            .release_device_buffer(buffer)
            .expect("buffer release");
        backend.release_stream(stream).expect("stream release");
        backend.release_memory_pool(pool).expect("pool release");
        backend.release_context(context).expect("context release");
        assert_eq!(backend.live_resource_count(), 0);
    }
}
