//! Record allocation ownership and the qualified runtime's implicit joins.
use super::*;

pub(super) struct OwnerHost {
    pub host: Rc<Host>,
    pub owners: RefCell<BTreeMap<u64, (u32, u64)>>,
    pub fail_sync: Cell<Option<u32>>,
    event: Cell<u32>,
}
impl OwnerHost {
    pub fn new(host: Rc<Host>) -> Self {
        Self {
            host,
            owners: RefCell::new(BTreeMap::new()),
            fail_sync: Cell::new(None),
            event: Cell::new(10_000_000),
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
impl HostServicesBackendV1 for OwnerHost {
    forward!(log(message: &[u8]) -> ());
    forward!(select_context(rank: u32, device_ordinal: i32) -> BundleContextHandleV1);
    forward!(release_context(context: BundleContextHandleV1) -> ());
    forward!(select_memory_pool(context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1);
    forward!(release_memory_pool(pool: BundleMemoryPoolHandleV1) -> ());
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
    forward!(create_event(context: BundleContextHandleV1) -> BundleEventHandleV1);
    forward!(record_event(event: BundleEventHandleV1, stream: BundleStreamHandleV1) -> ());
    forward!(wait_event(stream: BundleStreamHandleV1, event: BundleEventHandleV1) -> ());
    forward!(synchronize_event(event: BundleEventHandleV1) -> ());
    forward!(release_event(event: BundleEventHandleV1) -> ());
    forward!(copy_host_to_device(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, source: &[u8]) -> ());
    forward!(copy_device_to_host(stream: BundleStreamHandleV1, destination: &mut [u8], source: DeviceBufferSpanV1) -> ());
    forward!(copy_device_to_device(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, source: DeviceBufferSpanV1) -> ());
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
    fn allocate_device_buffer(
        &self,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        bytes: u64,
        alignment: u64,
    ) -> HostServiceResultV1<BundleDeviceBufferHandleV1> {
        let buffer = self
            .host
            .allocate_device_buffer(pool, stream, bytes, alignment)?;
        self.owners
            .borrow_mut()
            .insert(buffer.raw, (stream.slot(), bytes));
        self.host.note(format!(
            "owned-allocate {} {} {bytes}",
            buffer.raw,
            stream.slot()
        ));
        Ok(buffer)
    }
    fn release_device_buffer(&self, buffer: BundleDeviceBufferHandleV1) -> HostServiceResultV1<()> {
        self.host.note(format!("free {}", buffer.raw));
        self.host.release_device_buffer(buffer)?;
        self.owners.borrow_mut().remove(&buffer.raw);
        Ok(())
    }
    fn release_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        assert!(
            !self
                .owners
                .borrow()
                .values()
                .any(|&(owner, _)| owner == stream.slot())
        );
        self.host.release_stream(stream)
    }
    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        if self.fail_sync.get() == Some(stream.slot()) {
            self.fail_sync.set(None);
            return Err(BundleStatusV1::HostError);
        }
        self.host.synchronize_stream(stream)
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.host.copy_peer_to_peer(
            stream,
            destination_context,
            destination,
            source_context,
            source,
        )?;
        let event = self.event.get();
        self.event.set(event + 1);
        self.host.note(format!("record {} {event}", stream.slot()));
        let owners = self.owners.borrow();
        let source_owner = owners[&source.buffer.raw].0;
        let destination_owner = owners[&destination.buffer.raw].0;
        if source_owner != stream.slot() {
            self.host.note(format!("wait {source_owner} {event}"));
        }
        if destination_owner != stream.slot() && destination_owner != source_owner {
            self.host.note(format!("wait {destination_owner} {event}"));
        }
        Ok(())
    }
}
