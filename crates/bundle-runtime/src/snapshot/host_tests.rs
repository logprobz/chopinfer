use std::cell::RefCell;
use std::rc::Rc;

use runtime_bundle_abi::*;

use super::fake::span;
use super::{Action, HostRankStreams, RankStreams, copy_prefix};

#[derive(Default)]
struct Host {
    calls: RefCell<Vec<(u64, &'static str)>>,
    stuck: Option<u64>,
    fail_copy: bool,
}

impl HostServicesBackendV1 for Host {
    crate::unused_host_services! {
    log(_message: &[u8]) -> ();
    select_context(_rank: u32, _device_ordinal: i32) -> BundleContextHandleV1;
    release_context(_context: BundleContextHandleV1) -> ();
    select_memory_pool(_context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1;
    set_memory_pool_peer_access(
                _pool: BundleMemoryPoolHandleV1,
                _peer_context: BundleContextHandleV1,
                _enabled: bool,
            ) -> ();
    release_memory_pool(_pool: BundleMemoryPoolHandleV1) -> ();
    allocate_device_buffer(
                _pool: BundleMemoryPoolHandleV1,
                _stream: BundleStreamHandleV1,
                _byte_length: u64,
                _alignment: u64,
            ) -> BundleDeviceBufferHandleV1;
    release_device_buffer(_buffer: BundleDeviceBufferHandleV1) -> ();
    open_compiled_image(_component_role: &[u8]) -> BundleCompiledImageHandleV1;
    upload_rank_arena(
                _context: BundleContextHandleV1,
                _pool: BundleMemoryPoolHandleV1,
                _stream: BundleStreamHandleV1,
                _image: BundleCompiledImageHandleV1,
                _rank: u32,
            ) -> BundleRankArenaHandleV1;
    release_rank_arena(_arena: BundleRankArenaHandleV1) -> ();
    resolve_compiled_image_view(
                _image: BundleCompiledImageHandleV1,
                _arena: BundleRankArenaHandleV1,
                _rank: u32,
                _view_id: &[u8],
            ) -> CompiledImageSpanV1;
    release_compiled_image(_image: BundleCompiledImageHandleV1) -> ();
    acquire_peer_access(
                _owner_context: BundleContextHandleV1,
                _peer_context: BundleContextHandleV1,
            ) -> BundlePeerAccessHandleV1;
    release_peer_access(_peer_access: BundlePeerAccessHandleV1) -> ();
    create_stream(
                _context: BundleContextHandleV1,
                _stream_kind: u32,
            ) -> BundleStreamHandleV1;
    select_stage_stream(
                _context: BundleContextHandleV1,
                _rank: u32,
                _stage_id: &[u8],
                _stream_kind: u32,
            ) -> BundleStreamHandleV1;
    release_stream(_stream: BundleStreamHandleV1) -> ();
    create_event(_context: BundleContextHandleV1) -> BundleEventHandleV1;
    record_event(
                _event: BundleEventHandleV1,
                _stream: BundleStreamHandleV1,
            ) -> ();
    wait_event(
                _stream: BundleStreamHandleV1,
                _event: BundleEventHandleV1,
            ) -> ();
    synchronize_event(_event: BundleEventHandleV1) -> ();
    release_event(_event: BundleEventHandleV1) -> ();
    copy_peer_to_peer(
                _stream: BundleStreamHandleV1,
                _destination_context: BundleContextHandleV1,
                _destination: DeviceBufferSpanV1,
                _source_context: BundleContextHandleV1,
                _source: DeviceBufferSpanV1,
            ) -> ();
    zero_device_buffer(
                _stream: BundleStreamHandleV1,
                _destination: DeviceBufferSpanV1,
            ) -> ();
    write_device_pointer_table(
                _stream: BundleStreamHandleV1,
                _destination: DeviceBufferSpanV1,
                _slots: &[Option<DeviceBufferSpanV1>],
            ) -> ();
    load_cuda_module(
                _context: BundleContextHandleV1,
                _module_bytes: &[u8],
            ) -> BundleCudaModuleHandleV1;
    resolve_cuda_symbol(
                _module: BundleCudaModuleHandleV1,
                _symbol_name: &[u8],
            ) -> BundleCudaSymbolHandleV1;
    configure_cuda_symbol(
                _symbol: BundleCudaSymbolHandleV1,
                _dynamic_shared_memory_bytes: u32,
            ) -> ();
    launch_cuda_symbol(
                _stream: BundleStreamHandleV1,
                _symbol: BundleCudaSymbolHandleV1,
                _config: LaunchConfigV1,
                _arguments: &[KernelArgumentV1],
            ) -> ();
    release_cuda_symbol(_symbol: BundleCudaSymbolHandleV1) -> ();
    release_cuda_module(_module: BundleCudaModuleHandleV1) -> ();
    allocate_physical_kv(_request: &PhysicalKvRequestV1) -> BundlePhysicalKvHandleV1;
    create_kv_request(
                _kv: BundlePhysicalKvHandleV1,
                _quota_tokens: u64,
            ) -> BundleKvRequestHandleV1;
    prepare_kv(
                _request: BundleKvRequestHandleV1,
                _operation: &[u8],
            ) -> BundleKvTransactionHandleV1;
    commit_kv(_transaction: BundleKvTransactionHandleV1) -> ();
    cancel_kv(_transaction: BundleKvTransactionHandleV1) -> ();
    resolve_kv_view(
                _request: BundleKvRequestHandleV1,
                _rank: u32,
                _view_id: &[u8],
            ) -> DeviceBufferSpanV1;
    release_kv_request(_request: BundleKvRequestHandleV1) -> ();
    release_physical_kv(_kv: BundlePhysicalKvHandleV1) -> ();
    allocate_state(_request: &StateRequestV1) -> BundleStateHandleV1;
    resolve_state_view(
                _state: BundleStateHandleV1,
                _rank: u32,
            ) -> DeviceBufferSpanV1;
    release_state(_state: BundleStateHandleV1) -> ();
    create_collective(_request: &CollectiveRequestV1) -> BundleCollectiveHandleV1;
    execute_collective(
                _collective: BundleCollectiveHandleV1,
                _stream: BundleStreamHandleV1,
                _operation: &[u8],
                _arguments: &[KernelArgumentV1],
            ) -> ();
    release_collective(_collective: BundleCollectiveHandleV1) -> ();
    run_custom_operation(
                _context: BundleContextHandleV1,
                _stream: BundleStreamHandleV1,
                _completion: CustomCompletionV1,
                _operation_id: &[u8],
                _operands: &[CustomOperandV1],
                _operation: &CustomOperationV1<'_>,
            ) -> i32;
    begin_graph_capture(_stream: BundleStreamHandleV1) -> ();
    end_graph_capture(_stream: BundleStreamHandleV1) -> BundleGraphHandleV1;
    launch_graph(_stream: BundleStreamHandleV1, _graph: BundleGraphHandleV1) -> ();
    release_graph(_graph: BundleGraphHandleV1) -> ();
        }

    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.calls.borrow_mut().push((stream.raw, "drain"));
        if self.stuck == Some(stream.raw) {
            Err(BundleStatusV1::HostError)
        } else {
            Ok(())
        }
    }

    fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        assert_eq!(destination.offset, 7);
        assert_eq!(destination.length, source.len() as u64);
        assert_eq!(source, [1, 2]);
        self.calls.borrow_mut().push((stream.raw, "upload"));
        Ok(())
    }

    fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        assert_eq!(source.offset, 7);
        destination.fill(9);
        self.calls.borrow_mut().push((stream.raw, "read"));
        Ok(())
    }

    fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        assert_eq!(source.offset, 7);
        assert_eq!(destination.offset, 11);
        self.calls.borrow_mut().push((stream.raw, "copy"));
        if self.fail_copy {
            Err(BundleStatusV1::HostError)
        } else {
            Ok(())
        }
    }
}

#[test]
fn host_streams_preserve_rank_order_and_drain_precedence() {
    let streams = [1, 2, 3].map(|id| BundleStreamHandleV1::from_parts(1, id));
    for failed in [None, Some(0), Some(2)] {
        for stuck in [None, Some(0), Some(2)] {
            let host = Rc::new(Host {
                stuck: stuck.map(|rank| streams[rank].raw),
                ..Host::default()
            });
            let table = HostServicesTableV1::new(Rc::clone(&host));
            let launch = |rank, stream: BundleStreamHandleV1, (): ()| {
                assert_eq!(stream, streams[rank]);
                host.calls.borrow_mut().push((stream.raw, "launch"));
                if failed == Some(rank) {
                    Err("launch")
                } else {
                    Ok(())
                }
            };
            let adapter = HostRankStreams {
                services: table.caller(),
                streams: &streams,
                launch: &launch,
                host_error: &|_| "host",
                refusal: &|_| "refused",
                unconfirmed: &|| "unconfirmed",
            };
            let result = copy_prefix(
                &adapter,
                vec![
                    vec![Action::Launch(())],
                    vec![Action::Launch(())],
                    vec![Action::Launch(())],
                ],
            );
            let attempted = failed.map_or(3, |rank| rank + 1);
            let expected = if stuck.is_some_and(|rank| rank < attempted) {
                Err("unconfirmed")
            } else if failed.is_some() {
                Err("launch")
            } else {
                Ok(vec![vec![]; 3])
            };
            assert_eq!(result, expected);
            let expected_calls: Vec<_> = streams[..attempted]
                .iter()
                .map(|stream| (stream.raw, "launch"))
                .chain(
                    streams[..attempted]
                        .iter()
                        .map(|stream| (stream.raw, "drain")),
                )
                .collect();
            assert_eq!(*host.calls.borrow(), expected_calls);
        }
    }
}

#[test]
fn host_streams_use_bounded_uploads_and_caller_error_policy() {
    let host = Rc::new(Host::default());
    let table = HostServicesTableV1::new(Rc::clone(&host));
    let streams = [BundleStreamHandleV1::from_parts(1, 1)];
    let adapter = HostRankStreams {
        services: table.caller(),
        streams: &streams,
        launch: &|_, _, (): ()| Ok(()),
        host_error: &|_| "host",
        refusal: &|_| "refused",
        unconfirmed: &|| "unconfirmed",
    };
    let mut from = span(1, 2);
    from.offset = 7;
    let mut to = span(2, 2);
    to.offset = 11;
    assert_eq!(copy_prefix(&adapter, vec![]), Err("refused"));
    assert_eq!(
        copy_prefix(
            &adapter,
            vec![vec![Action::Read {
                from: span(1, super::MAX_READ_BYTES + 1)
            }]]
        ),
        Err("refused")
    );
    assert!(host.calls.borrow().is_empty());
    assert_eq!(adapter.upload(0, span(1, 1), &[1, 2]), Err("host"));
    assert!(host.calls.borrow().is_empty());
    let result = copy_prefix(
        &adapter,
        vec![vec![
            Action::Upload {
                to: from,
                bytes: vec![1, 2],
            },
            Action::Copy { from, to },
            Action::Read { from },
        ]],
    );
    assert_eq!(result, Ok(vec![vec![9, 9]]));
    assert_eq!(
        *host.calls.borrow(),
        ["upload", "copy", "read", "drain"].map(|op| (streams[0].raw, op))
    );
}

#[test]
fn host_copy_failure_uses_conversion_and_still_drains() {
    let host = Rc::new(Host {
        fail_copy: true,
        ..Host::default()
    });
    let table = HostServicesTableV1::new(Rc::clone(&host));
    let streams = [BundleStreamHandleV1::from_parts(1, 1)];
    let adapter = HostRankStreams {
        services: table.caller(),
        streams: &streams,
        launch: &|_, _, (): ()| Ok(()),
        host_error: &|error| {
            assert!(matches!(
                error,
                BundleAbiError::Status(BundleStatusV1::HostError)
            ));
            "converted host failure"
        },
        refusal: &|_| "refused",
        unconfirmed: &|| "unconfirmed",
    };
    let mut from = span(1, 2);
    from.offset = 7;
    let mut to = span(2, 2);
    to.offset = 11;
    assert_eq!(
        copy_prefix(&adapter, vec![vec![Action::Copy { from, to }]]),
        Err("converted host failure")
    );
    assert_eq!(
        *host.calls.borrow(),
        [(streams[0].raw, "copy"), (streams[0].raw, "drain")]
    );
}
