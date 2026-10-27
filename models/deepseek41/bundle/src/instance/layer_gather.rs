//! One layer's mapped-row slot. Its owner keeps the step's lifecycle state.
use crate::engram_rows::{Plane, PreparedGather};
use runtime_bundle_abi::{
    BundleAbiError, BundleCompiledImageHandleV1, BundleMappedGatherHandleV1, BundleStreamHandleV1,
    DeviceBufferSpanV1 as Span, HostServicesV1, MappedGatherPlaneV1, MappedGatherStatsV1,
};

/// Graph-backed targets attach their cache; eager prefills use `()`.
pub(crate) struct LayerGather<Graphs = ()> {
    pub layer: u32,
    pub gather: BundleMappedGatherHandleV1,
    pub indices: Span,
    pub destinations: Vec<Span>,
    pub graphs: Graphs,
}

pub(super) fn plane(
    image: BundleCompiledImageHandleV1,
    rank: u32,
    stream: BundleStreamHandleV1,
    source: &Plane,
    destination: Span,
) -> MappedGatherPlaneV1 {
    MappedGatherPlaneV1 {
        image,
        rank,
        view_id: source.view.clone(),
        row_bytes: source.row_bytes,
        index_offset: 0,
        stream,
        destination,
    }
}

impl<Graphs> LayerGather<Graphs> {
    pub(crate) fn create(
        services: &HostServicesV1,
        producer: BundleStreamHandleV1,
        layer: u32,
        indices: Span,
        planes: &[MappedGatherPlaneV1],
        graphs: impl FnOnce() -> Graphs,
    ) -> Result<Self, BundleAbiError> {
        let gather = services.create_mapped_gather(producer, indices, planes)?;
        let graphs = graphs();
        Ok(Self {
            layer,
            gather,
            indices,
            destinations: planes.iter().map(|plane| plane.destination).collect(),
            graphs,
        })
    }

    pub(crate) fn begin(
        &self,
        services: &HostServicesV1,
        rows: u64,
        route: PreparedGather,
    ) -> Result<(), BundleAbiError> {
        route.submitted();
        services.enqueue_mapped_gather(self.gather, rows)
    }

    /// Join one stream so callers retain their early/target ordering per rank.
    pub(crate) fn join(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
    ) -> Result<(), BundleAbiError> {
        services.wait_mapped_gather(self.gather, stream)
    }

    pub(crate) fn finish(
        &self,
        services: &HostServicesV1,
    ) -> Result<Option<MappedGatherStatsV1>, BundleAbiError> {
        services.complete_mapped_gather(self.gather)
    }

    pub(crate) fn release(&self, services: &HostServicesV1) -> Result<(), BundleAbiError> {
        services.release_mapped_gather(self.gather)
    }
}
