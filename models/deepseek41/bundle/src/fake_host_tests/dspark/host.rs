//! A logical zero-filled mapped table with bounded gather allocations.
//! All other services use the ordinary fake host, including span checks.
use super::*;
use runtime_bundle_abi::{BundleMappedGatherHandleV1, MappedGatherPlaneV1, MappedGatherStatsV1};

struct Gather {
    producer: BundleStreamHandleV1,
    indices: DeviceBufferSpanV1,
    planes: Vec<MappedGatherPlaneV1>,
    pending: Option<u64>,
    waited: std::collections::BTreeSet<u64>,
}
pub(in crate::fake_host_tests) struct MappedHost {
    host: Rc<Host>,
    gathers: std::cell::RefCell<std::collections::BTreeMap<u64, Gather>>,
}
impl MappedHost {
    pub(in crate::fake_host_tests) fn new(host: Rc<Host>) -> Self {
        Self {
            host,
            gathers: Default::default(),
        }
    }
}

macro_rules! forward {
    ($name:ident($($arg:ident: $ty:ty),*) -> $result:ty) => {
        fn $name(&self, $($arg: $ty),*) -> HostServiceResultV1<$result> {
            self.host.$name($($arg),*)
        }
    };
}
impl HostServicesBackendV1 for MappedHost {
    forward!(log(message: &[u8]) -> ());
    forward!(select_context(rank: u32, device_ordinal: i32) -> BundleContextHandleV1);
    forward!(release_context(context: BundleContextHandleV1) -> ());
    forward!(select_memory_pool(context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1);
    forward!(release_memory_pool(pool: BundleMemoryPoolHandleV1) -> ());
    forward!(allocate_device_buffer(pool: BundleMemoryPoolHandleV1, stream: BundleStreamHandleV1, byte_length: u64, alignment: u64) -> BundleDeviceBufferHandleV1);
    forward!(release_device_buffer(buffer: BundleDeviceBufferHandleV1) -> ());
    forward!(open_compiled_image(component_role: &[u8]) -> BundleCompiledImageHandleV1);
    forward!(upload_rank_arena(context: BundleContextHandleV1, pool: BundleMemoryPoolHandleV1, stream: BundleStreamHandleV1, image: BundleCompiledImageHandleV1, rank: u32) -> BundleRankArenaHandleV1);
    forward!(release_rank_arena(arena: BundleRankArenaHandleV1) -> ());
    forward!(resolve_compiled_image_view(image: BundleCompiledImageHandleV1, arena: BundleRankArenaHandleV1, rank: u32, view_id: &[u8]) -> CompiledImageSpanV1);
    forward!(release_compiled_image(image: BundleCompiledImageHandleV1) -> ());
    forward!(open_authenticated_component(component_role: &[u8]) -> BundleAuthenticatedComponentHandleV1);
    forward!(inspect_authenticated_component(component: BundleAuthenticatedComponentHandleV1) -> AuthenticatedComponentMetadataV1);
    forward!(read_authenticated_component(component: BundleAuthenticatedComponentHandleV1, role_destination: &mut [u8], bytes_destination: &mut [u8]) -> ());
    forward!(release_authenticated_component(component: BundleAuthenticatedComponentHandleV1) -> ());
    forward!(acquire_peer_access(owner_context: BundleContextHandleV1, peer_context: BundleContextHandleV1) -> BundlePeerAccessHandleV1);
    forward!(release_peer_access(peer_access: BundlePeerAccessHandleV1) -> ());
    forward!(create_stream(context: BundleContextHandleV1, stream_kind: u32) -> BundleStreamHandleV1);
    forward!(select_stage_stream(context: BundleContextHandleV1, rank: u32, stage_id: &[u8], stream_kind: u32) -> BundleStreamHandleV1);
    forward!(synchronize_stream(stream: BundleStreamHandleV1) -> ());
    forward!(release_stream(stream: BundleStreamHandleV1) -> ());
    forward!(create_event(context: BundleContextHandleV1) -> BundleEventHandleV1);
    forward!(record_event(event: BundleEventHandleV1, stream: BundleStreamHandleV1) -> ());
    forward!(wait_event(stream: BundleStreamHandleV1, event: BundleEventHandleV1) -> ());
    forward!(synchronize_event(event: BundleEventHandleV1) -> ());
    forward!(release_event(event: BundleEventHandleV1) -> ());
    forward!(copy_host_to_device(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, source: &[u8]) -> ());
    forward!(copy_device_to_host(stream: BundleStreamHandleV1, destination: &mut [u8], source: DeviceBufferSpanV1) -> ());
    forward!(copy_device_to_device(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, source: DeviceBufferSpanV1) -> ());
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        assert_eq!(
            self.host.stream_ranks.borrow()[&stream.slot()] + 1,
            destination_context.slot()
        );
        assert_ne!(destination_context, source_context);
        assert!(!self.host.capturing_on_device_of(stream.slot()));
        self.host.note(format!(
            "peer results {} {}",
            source_context.slot(),
            destination_context.slot()
        ));
        self.host.copy_device_to_device(stream, destination, source)
    }
    forward!(zero_device_buffer(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1) -> ());
    forward!(write_device_pointer_table(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, slots: &[Option<DeviceBufferSpanV1>]) -> ());
    forward!(load_cuda_module(context: BundleContextHandleV1, module_bytes: &[u8]) -> BundleCudaModuleHandleV1);
    forward!(resolve_cuda_symbol(module: BundleCudaModuleHandleV1, symbol_name: &[u8]) -> BundleCudaSymbolHandleV1);
    forward!(configure_cuda_symbol(symbol: BundleCudaSymbolHandleV1, dynamic_shared_memory_bytes: u32) -> ());
    forward!(launch_cuda_symbol(stream: BundleStreamHandleV1, symbol: BundleCudaSymbolHandleV1, config: LaunchConfigV1, arguments: &[KernelArgumentV1]) -> ());
    forward!(release_cuda_symbol(symbol: BundleCudaSymbolHandleV1) -> ());
    forward!(release_cuda_module(module: BundleCudaModuleHandleV1) -> ());
    forward!(allocate_physical_kv(request: &PhysicalKvRequestV1) -> BundlePhysicalKvHandleV1);
    forward!(create_kv_request(kv: BundlePhysicalKvHandleV1, quota_tokens: u64) -> BundleKvRequestHandleV1);
    forward!(prepare_kv(request: BundleKvRequestHandleV1, operation: &[u8]) -> BundleKvTransactionHandleV1);
    forward!(commit_kv(transaction: BundleKvTransactionHandleV1) -> ());
    forward!(cancel_kv(transaction: BundleKvTransactionHandleV1) -> ());
    forward!(resolve_kv_view(request: BundleKvRequestHandleV1, rank: u32, view_id: &[u8]) -> DeviceBufferSpanV1);
    forward!(release_kv_request(request: BundleKvRequestHandleV1) -> ());
    forward!(release_physical_kv(kv: BundlePhysicalKvHandleV1) -> ());
    forward!(acquire_serving_kv(layout_id: &[u8]) -> u64);
    forward!(resolve_serving_kv_pool(rank: u32) -> DeviceBufferSpanV1);
    forward!(query_speculation() -> BundleRawSpeculationV1);
    forward!(query_batch_limits() -> BundleRawBatchLimitsV1);
    forward!(query_cluster_residency(symbol: BundleCudaSymbolHandleV1, config: LaunchConfigV1) -> BundleRawClusterResidencyV1);
    forward!(allocate_state(request: &StateRequestV1) -> BundleStateHandleV1);
    forward!(resolve_state_view(state: BundleStateHandleV1, rank: u32) -> DeviceBufferSpanV1);
    forward!(release_state(state: BundleStateHandleV1) -> ());
    forward!(create_collective(request: &CollectiveRequestV1) -> BundleCollectiveHandleV1);
    forward!(execute_collective(collective: BundleCollectiveHandleV1, stream: BundleStreamHandleV1, operation: &[u8], arguments: &[KernelArgumentV1]) -> ());
    forward!(release_collective(collective: BundleCollectiveHandleV1) -> ());
    forward!(run_custom_operation(context: BundleContextHandleV1, stream: BundleStreamHandleV1, completion: CustomCompletionV1, operation_id: &[u8], operands: &[CustomOperandV1], operation: &CustomOperationV1<'_>) -> i32);
    forward!(begin_graph_capture(stream: BundleStreamHandleV1) -> ());
    forward!(end_graph_capture(stream: BundleStreamHandleV1) -> BundleGraphHandleV1);
    forward!(launch_graph(stream: BundleStreamHandleV1, graph: BundleGraphHandleV1) -> ());
    forward!(release_graph(graph: BundleGraphHandleV1) -> ());
    forward!(set_memory_pool_peer_access(pool: BundleMemoryPoolHandleV1, peer_context: BundleContextHandleV1, enabled: bool) -> ());
    forward!(upload_rank_arenas(uploads: &[RankArenaUploadV1]) -> Vec<BundleRankArenaHandleV1>);
    fn create_mapped_gather(
        &self,
        producer: BundleStreamHandleV1,
        indices: DeviceBufferSpanV1,
        planes: &[MappedGatherPlaneV1],
    ) -> HostServiceResultV1<BundleMappedGatherHandleV1> {
        self.host.check_span(indices);
        if planes.is_empty() || self.host.capturing_on_device_of(producer.slot()) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        for plane in planes {
            self.host.check_span(plane.destination);
            if plane.row_bytes == 0
                || !plane.destination.length.is_multiple_of(plane.row_bytes)
                || plane.index_offset + plane.destination.length / plane.row_bytes * 4
                    > indices.length
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
        let rank = *self
            .host
            .stream_ranks
            .borrow()
            .get(&producer.slot())
            .ok_or(BundleStatusV1::InvalidArgument)?;
        // An event proxy gives this fake gather the existing checked LIFO
        // resource lifetime. Production owns a dedicated gather handle.
        let event = self
            .host
            .create_event(BundleContextHandleV1::from_parts(1, rank + 1))?;
        let gather = BundleMappedGatherHandleV1::from_raw(event.raw);
        self.gathers.borrow_mut().insert(
            gather.raw,
            Gather {
                producer,
                indices,
                planes: planes.to_vec(),
                pending: None,
                waited: Default::default(),
            },
        );
        self.host
            .note(format!("async mapped create {}", gather.slot()));
        Ok(gather)
    }
    fn enqueue_mapped_gather(
        &self,
        handle: BundleMappedGatherHandleV1,
        rows: u64,
    ) -> HostServiceResultV1<()> {
        let mut gathers = self.gathers.borrow_mut();
        let gather = gathers
            .get_mut(&handle.raw)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if gather.pending.is_some() || self.host.capturing_on_device_of(gather.producer.slot()) {
            return Err(BundleStatusV1::InvalidState);
        }
        if rows == 0
            || gather
                .planes
                .iter()
                .any(|p| rows > p.destination.length / p.row_bytes)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        gather.pending = Some(rows);
        gather.waited.clear();
        self.host.note(format!(
            "async mapped ids {} {}",
            gather.indices.length, rows
        ));
        Ok(())
    }
    fn wait_mapped_gather(
        &self,
        handle: BundleMappedGatherHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        let (rows, planes) = {
            let mut gathers = self.gathers.borrow_mut();
            let gather = gathers
                .get_mut(&handle.raw)
                .ok_or(BundleStatusV1::StaleHandle)?;
            let rows = gather.pending.ok_or(BundleStatusV1::InvalidState)?;
            if self.host.capturing_on_device_of(stream.slot()) || !gather.waited.insert(stream.raw)
            {
                return Err(BundleStatusV1::InvalidState);
            }
            let planes = gather
                .planes
                .iter()
                .filter(|plane| plane.stream == stream)
                .cloned()
                .collect::<Vec<_>>();
            if planes.is_empty() {
                return Err(BundleStatusV1::InvalidArgument);
            }
            (rows, planes)
        };
        self.host
            .note(format!("async mapped wait {}", stream.slot()));
        for plane in planes {
            let destination = DeviceBufferSpanV1::new(
                plane.destination.buffer,
                plane.destination.offset,
                rows * plane.row_bytes,
            );
            self.gather_mapped_rows(
                stream,
                destination,
                MappedRowGatherV1 {
                    image: plane.image,
                    rank: plane.rank,
                    view_id: &plane.view_id,
                    row_bytes: plane.row_bytes,
                    rows: &vec![0; rows as usize],
                },
            )?;
        }
        Ok(())
    }
    fn complete_mapped_gather(
        &self,
        handle: BundleMappedGatherHandleV1,
    ) -> HostServiceResultV1<Option<MappedGatherStatsV1>> {
        let mut gathers = self.gathers.borrow_mut();
        let gather = gathers
            .get_mut(&handle.raw)
            .ok_or(BundleStatusV1::StaleHandle)?;
        if gather.pending.is_none() {
            return Err(BundleStatusV1::InvalidState);
        }
        if gather
            .planes
            .iter()
            .any(|plane| !gather.waited.contains(&plane.stream.raw))
        {
            return Ok(None);
        }
        gather.pending = None;
        self.host.note("async mapped complete".into());
        Ok(Some(MappedGatherStatsV1::default()))
    }
    fn release_mapped_gather(&self, handle: BundleMappedGatherHandleV1) -> HostServiceResultV1<()> {
        if !self.gathers.borrow().contains_key(&handle.raw) {
            return Err(BundleStatusV1::StaleHandle);
        }
        self.host
            .release_event(BundleEventHandleV1::from_raw(handle.raw))?;
        self.gathers.borrow_mut().remove(&handle.raw);
        self.host
            .note(format!("async mapped release {}", handle.slot()));
        Ok(())
    }
    fn gather_mapped_rows(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        gather: MappedRowGatherV1<'_>,
    ) -> HostServiceResultV1<()> {
        if self.host.capturing_on_device_of(stream.slot()) {
            return Err(BundleStatusV1::InvalidState);
        }
        let g = self.host.shape.geometry;
        let e = &g.shape.engram;
        let table = e.layers.iter().enumerate().find_map(|(index, layer)| {
            ["weight", "scale"].into_iter().find_map(|part| {
                (gather.view_id == format!("layers.{layer}.engram.embed.{part}").as_bytes())
                    .then_some((
                        e.rows[index],
                        if part == "weight" {
                            e.head_dim
                        } else {
                            e.head_dim / 32
                        },
                    ))
            })
        });
        let (rows, width) = table.ok_or(BundleStatusV1::NotFound)?;
        if gather.image.raw == 0
            || gather.rank >= self.host.shape.ranks()
            || gather.row_bytes != u64::from(width)
            || gather.rows.is_empty()
            || gather.rows.iter().any(|&row| row >= rows)
            || (gather.rows.len() as u64).checked_mul(gather.row_bytes) != Some(destination.length)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        self.host.note(format!(
            "mapped gather {} {}",
            gather.rank,
            String::from_utf8_lossy(gather.view_id)
        ));
        self.host
            .copy_host_to_device(stream, destination, &vec![0; destination.length as usize])
    }
}
