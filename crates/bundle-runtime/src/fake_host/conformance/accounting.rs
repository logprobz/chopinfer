//! Observe the safe host-service boundary without changing model fake hosts.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use runtime_bundle_abi::*;

/// Resource type and full handle bits; peer grants also name their peer.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Identity(&'static str, u64, u64);

#[derive(Default)]
pub struct Accounting {
    calls: Cell<usize>,
    acquired: RefCell<Vec<Identity>>,
    released: RefCell<Vec<Identity>>,
    live: RefCell<BTreeMap<Identity, usize>>,
    violations: RefCell<Vec<Identity>>,
}

impl Accounting {
    pub fn calls(&self) -> usize {
        self.calls.get()
    }
    pub fn releases(&self) -> usize {
        self.released.borrow().len()
    }

    fn acquire(&self, identity: Identity) {
        self.acquired.borrow_mut().push(identity);
        *self.live.borrow_mut().entry(identity).or_default() += 1;
    }

    fn check_release(&self, identity: Identity) -> HostServiceResultV1<()> {
        if !self.live.borrow().contains_key(&identity) {
            self.violations.borrow_mut().push(identity);
            return Err(BundleStatusV1::InvalidState);
        }
        Ok(())
    }

    fn release(&self, identity: Identity) {
        let mut live = self.live.borrow_mut();
        let references = live.get_mut(&identity).expect("validated release");
        *references -= 1;
        if *references == 0 {
            live.remove(&identity);
        }
        self.released.borrow_mut().push(identity);
    }

    pub fn assert_released(&self) {
        assert!(
            self.violations.borrow().is_empty(),
            "duplicate or unknown releases: {:?}",
            self.violations.borrow()
        );
        assert!(
            self.live.borrow().is_empty(),
            "unreleased identities: {:?}",
            self.live.borrow()
        );
        assert_eq!(self.acquired.borrow().len(), self.released.borrow().len());
    }
}

/// Every method is forwarded, including optional services. Acquisitions are
/// recorded only after success, and failed releases keep their live lease.
/// Contexts, pools and peer edges can have multiple leases for one identity.
pub struct ObservedHost<B> {
    backend: Rc<B>,
    pub accounting: Accounting,
}

impl<B> ObservedHost<B> {
    pub fn new(backend: Rc<B>) -> Self {
        Self {
            backend,
            accounting: Accounting::default(),
        }
    }
    fn call(&self) {
        self.accounting.calls.set(self.accounting.calls.get() + 1);
    }
}

macro_rules! forward {
    ($name:ident($($argument:ident: $type:ty),*) -> $result:ty) => {
        fn $name(&self, $($argument: $type),*) -> HostServiceResultV1<$result> {
            self.call();
            self.backend.$name($($argument),*)
        }
    };
}
macro_rules! acquire {
    ($kind:literal, $name:ident($($argument:ident: $type:ty),*) -> $result:ty) => {
        fn $name(&self, $($argument: $type),*) -> HostServiceResultV1<$result> {
            self.call();
            let handle = self.backend.$name($($argument),*)?;
            self.accounting.acquire(Identity($kind, handle.raw, 0));
            Ok(handle)
        }
    };
}
macro_rules! release {
    ($kind:literal, $name:ident($argument:ident: $type:ty)) => {
        fn $name(&self, $argument: $type) -> HostServiceResultV1<()> {
            self.call();
            let identity = Identity($kind, $argument.raw, 0);
            self.accounting.check_release(identity)?;
            self.backend.$name($argument)?;
            self.accounting.release(identity);
            Ok(())
        }
    };
}

impl<B: HostServicesBackendV1> HostServicesBackendV1 for ObservedHost<B> {
    forward!(log(message: &[u8]) -> ());
    acquire!("context", select_context(rank: u32, device_ordinal: i32) -> BundleContextHandleV1);
    release!("context", release_context(context: BundleContextHandleV1));
    acquire!("pool", select_memory_pool(context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1);
    release!("pool", release_memory_pool(pool: BundleMemoryPoolHandleV1));
    acquire!("buffer", allocate_device_buffer(pool: BundleMemoryPoolHandleV1, stream: BundleStreamHandleV1, byte_length: u64, alignment: u64) -> BundleDeviceBufferHandleV1);
    release!("buffer", release_device_buffer(buffer: BundleDeviceBufferHandleV1));
    forward!(gather_mapped_rows(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, gather: MappedRowGatherV1<'_>) -> ());
    acquire!("image", open_compiled_image(component_role: &[u8]) -> BundleCompiledImageHandleV1);
    acquire!("arena", upload_rank_arena(context: BundleContextHandleV1, pool: BundleMemoryPoolHandleV1, stream: BundleStreamHandleV1, image: BundleCompiledImageHandleV1, rank: u32) -> BundleRankArenaHandleV1);
    release!("arena", release_rank_arena(arena: BundleRankArenaHandleV1));
    forward!(resolve_compiled_image_view(image: BundleCompiledImageHandleV1, arena: BundleRankArenaHandleV1, rank: u32, view_id: &[u8]) -> CompiledImageSpanV1);
    release!("image", release_compiled_image(image: BundleCompiledImageHandleV1));
    acquire!("component", open_authenticated_component(component_role: &[u8]) -> BundleAuthenticatedComponentHandleV1);
    forward!(inspect_authenticated_component(component: BundleAuthenticatedComponentHandleV1) -> AuthenticatedComponentMetadataV1);
    forward!(read_authenticated_component(component: BundleAuthenticatedComponentHandleV1, role_destination: &mut [u8], bytes_destination: &mut [u8]) -> ());
    release!("component", release_authenticated_component(component: BundleAuthenticatedComponentHandleV1));
    acquire!("peer", acquire_peer_access(owner_context: BundleContextHandleV1, peer_context: BundleContextHandleV1) -> BundlePeerAccessHandleV1);
    release!("peer", release_peer_access(peer_access: BundlePeerAccessHandleV1));
    acquire!("stream", create_stream(context: BundleContextHandleV1, stream_kind: u32) -> BundleStreamHandleV1);
    acquire!("stream", select_stage_stream(context: BundleContextHandleV1, rank: u32, stage_id: &[u8], stream_kind: u32) -> BundleStreamHandleV1);
    forward!(synchronize_stream(stream: BundleStreamHandleV1) -> ());
    release!("stream", release_stream(stream: BundleStreamHandleV1));
    acquire!("event", create_event(context: BundleContextHandleV1) -> BundleEventHandleV1);
    forward!(record_event(event: BundleEventHandleV1, stream: BundleStreamHandleV1) -> ());
    forward!(wait_event(stream: BundleStreamHandleV1, event: BundleEventHandleV1) -> ());
    forward!(synchronize_event(event: BundleEventHandleV1) -> ());
    release!("event", release_event(event: BundleEventHandleV1));
    forward!(copy_host_to_device(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, source: &[u8]) -> ());
    forward!(copy_device_to_host(stream: BundleStreamHandleV1, destination: &mut [u8], source: DeviceBufferSpanV1) -> ());
    forward!(copy_device_to_device(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, source: DeviceBufferSpanV1) -> ());
    forward!(copy_peer_to_peer(stream: BundleStreamHandleV1, destination_context: BundleContextHandleV1, destination: DeviceBufferSpanV1, source_context: BundleContextHandleV1, source: DeviceBufferSpanV1) -> ());
    forward!(zero_device_buffer(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1) -> ());
    forward!(write_device_pointer_table(stream: BundleStreamHandleV1, destination: DeviceBufferSpanV1, slots: &[Option<DeviceBufferSpanV1>]) -> ());
    acquire!("module", load_cuda_module(context: BundleContextHandleV1, module_bytes: &[u8]) -> BundleCudaModuleHandleV1);
    acquire!("symbol", resolve_cuda_symbol(module: BundleCudaModuleHandleV1, symbol_name: &[u8]) -> BundleCudaSymbolHandleV1);
    forward!(configure_cuda_symbol(symbol: BundleCudaSymbolHandleV1, dynamic_shared_memory_bytes: u32) -> ());
    forward!(launch_cuda_symbol(stream: BundleStreamHandleV1, symbol: BundleCudaSymbolHandleV1, config: LaunchConfigV1, arguments: &[KernelArgumentV1]) -> ());
    release!("symbol", release_cuda_symbol(symbol: BundleCudaSymbolHandleV1));
    release!("module", release_cuda_module(module: BundleCudaModuleHandleV1));
    acquire!("physical_kv", allocate_physical_kv(request: &PhysicalKvRequestV1) -> BundlePhysicalKvHandleV1);
    acquire!("kv_request", create_kv_request(kv: BundlePhysicalKvHandleV1, quota_tokens: u64) -> BundleKvRequestHandleV1);
    acquire!("kv_transaction", prepare_kv(request: BundleKvRequestHandleV1, operation: &[u8]) -> BundleKvTransactionHandleV1);
    release!("kv_transaction", commit_kv(transaction: BundleKvTransactionHandleV1));
    release!("kv_transaction", cancel_kv(transaction: BundleKvTransactionHandleV1));
    forward!(resolve_kv_view(request: BundleKvRequestHandleV1, rank: u32, view_id: &[u8]) -> DeviceBufferSpanV1);
    release!("kv_request", release_kv_request(request: BundleKvRequestHandleV1));
    release!("physical_kv", release_physical_kv(kv: BundlePhysicalKvHandleV1));
    forward!(acquire_serving_kv(layout_id: &[u8]) -> u64);
    forward!(resolve_serving_kv_pool(rank: u32) -> DeviceBufferSpanV1);
    forward!(query_speculation() -> BundleRawSpeculationV1);
    forward!(query_batch_limits() -> BundleRawBatchLimitsV1);
    forward!(query_cluster_residency(symbol: BundleCudaSymbolHandleV1, config: LaunchConfigV1) -> BundleRawClusterResidencyV1);
    acquire!("state", allocate_state(request: &StateRequestV1) -> BundleStateHandleV1);
    forward!(resolve_state_view(state: BundleStateHandleV1, rank: u32) -> DeviceBufferSpanV1);
    release!("state", release_state(state: BundleStateHandleV1));
    acquire!("collective", create_collective(request: &CollectiveRequestV1) -> BundleCollectiveHandleV1);
    forward!(execute_collective(collective: BundleCollectiveHandleV1, stream: BundleStreamHandleV1, operation: &[u8], arguments: &[KernelArgumentV1]) -> ());
    release!("collective", release_collective(collective: BundleCollectiveHandleV1));
    forward!(run_custom_operation(context: BundleContextHandleV1, stream: BundleStreamHandleV1, completion: CustomCompletionV1, operation_id: &[u8], operands: &[CustomOperandV1], operation: &CustomOperationV1<'_>) -> i32);
    forward!(begin_graph_capture(stream: BundleStreamHandleV1) -> ());
    acquire!("graph", end_graph_capture(stream: BundleStreamHandleV1) -> BundleGraphHandleV1);
    forward!(launch_graph(stream: BundleStreamHandleV1, graph: BundleGraphHandleV1) -> ());
    release!("graph", release_graph(graph: BundleGraphHandleV1));

    fn set_memory_pool_peer_access(
        &self,
        pool: BundleMemoryPoolHandleV1,
        peer_context: BundleContextHandleV1,
        enabled: bool,
    ) -> HostServiceResultV1<()> {
        self.call();
        let identity = Identity("pool_peer", pool.raw, peer_context.raw);
        if !enabled {
            self.accounting.check_release(identity)?;
        }
        self.backend
            .set_memory_pool_peer_access(pool, peer_context, enabled)?;
        if enabled {
            self.accounting.acquire(identity);
        } else {
            self.accounting.release(identity);
        }
        Ok(())
    }

    fn upload_rank_arenas(
        &self,
        uploads: &[RankArenaUploadV1],
    ) -> HostServiceResultV1<Vec<BundleRankArenaHandleV1>> {
        self.call();
        let handles = self.backend.upload_rank_arenas(uploads)?;
        for handle in &handles {
            self.accounting.acquire(Identity("arena", handle.raw, 0));
        }
        Ok(handles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_release_cannot_compensate_for_a_leak_of_another_identity() {
        // Each Held kind has a distinct namespace, even when handle bits match.
        for kind in [
            "image",
            "component",
            "context",
            "pool",
            "stream",
            "event",
            "arena",
            "buffer",
            "peer",
            "pool_peer",
            "module",
            "symbol",
            "state",
        ] {
            let accounting = Accounting::default();
            let first = Identity(kind, 11, 0);
            let second = Identity(kind, 12, 0);
            accounting.acquire(first);
            accounting.acquire(second);
            accounting.check_release(first).unwrap();
            accounting.release(first);
            assert_eq!(
                accounting.check_release(first),
                Err(BundleStatusV1::InvalidState)
            );
            assert_eq!(*accounting.live.borrow(), BTreeMap::from([(second, 1)]));
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || accounting.assert_released()
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn leases_generations_and_peer_edges_keep_their_full_identity() {
        let accounting = Accounting::default();
        let context = Identity("context", BundleContextHandleV1::from_parts(1, 7).raw, 0);
        accounting.acquire(context);
        accounting.acquire(context);
        for _ in 0..2 {
            accounting.check_release(context).unwrap();
            accounting.release(context);
        }
        accounting.assert_released();
        assert_eq!(
            accounting.check_release(context),
            Err(BundleStatusV1::InvalidState)
        );
        let pool = Identity("pool", context.1, 0);
        accounting.acquire(pool);
        assert_eq!(
            accounting.check_release(Identity(
                "pool",
                BundleMemoryPoolHandleV1::from_parts(2, 7).raw,
                0
            )),
            Err(BundleStatusV1::InvalidState)
        );
        let edge = Identity("pool_peer", pool.1, 13);
        accounting.acquire(edge);
        assert_eq!(
            accounting.check_release(Identity("pool_peer", pool.1, 14)),
            Err(BundleStatusV1::InvalidState)
        );
    }

    #[test]
    fn non_device_and_failed_services_are_counted() {
        let host = ObservedHost::new(Rc::new(UnsupportedHostServicesBackendV1));
        assert_eq!(host.accounting.calls(), 0);
        assert_eq!(
            host.log(b"unrecorded log"),
            Err(BundleStatusV1::Unsupported)
        );
        assert_eq!(host.select_context(0, 0), Err(BundleStatusV1::Unsupported));
        host.query_speculation().unwrap();
        assert_eq!(host.accounting.calls(), 3);
    }
}
