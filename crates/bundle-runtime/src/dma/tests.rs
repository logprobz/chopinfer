use super::*;
use runtime_bundle_abi::BundleDeviceBufferHandleV1;

fn spans(first: u32, bytes: u64) -> Vec<Span> {
    (first..first + 4)
        .map(|slot| Span::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 256, bytes))
        .collect()
}

#[test]
fn uneven_rows_scatter_each_peer_once_and_gather_every_owner() {
    for rows in [512_u32, 513, 1023, 2048, 8191, 8192] {
        let width = 16;
        let elements = u64::from(rows) * u64::from(width);
        let inputs = spans(1, elements * 4);
        let outputs = spans(5, elements * 2);
        let inboxes = spans(9, u64::from(rows.div_ceil(4)) * u64::from(width) * 16);
        let call = Call::layout(&inboxes, &inputs, &outputs, rows, width).unwrap();
        assert_eq!(call.owners.iter().map(|o| o.rows).sum::<u32>(), rows);
        for destination in 0..4 {
            let owner = call.owners[destination];
            let scatter: Vec<_> = call
                .scatter
                .iter()
                .flatten()
                .filter(|t| t.destination == destination)
                .collect();
            let mut sources = scatter.iter().map(|t| t.source).collect::<Vec<_>>();
            sources.sort_unstable();
            assert_eq!(
                sources,
                (0..4).filter(|&r| r != destination).collect::<Vec<_>>()
            );
            for t in scatter {
                assert_eq!(t.input.offset, owner.input.offset);
                assert_eq!(t.input.length, owner.input.length);
                assert_eq!(
                    t.output.offset,
                    inboxes[destination].offset + t.source as u64 * owner.slot_elements * 4
                );
                assert_eq!(t.input.length, t.output.length);
            }
            let mut intervals = vec![(owner.output.offset, owner.output.length)];
            for t in call
                .gather
                .iter()
                .flatten()
                .filter(|t| t.destination == destination)
            {
                assert_eq!(t.input, call.owners[t.source].output);
                assert_eq!(t.output.buffer, outputs[destination].buffer);
                intervals.push((t.output.offset, t.output.length));
            }
            intervals.sort_unstable();
            let mut frontier = outputs[destination].offset;
            for (offset, length) in intervals {
                assert_eq!(offset, frontier);
                frontier += length;
            }
            assert_eq!(
                frontier,
                outputs[destination].offset + outputs[destination].length
            );
        }
        let moved = call
            .scatter
            .iter()
            .chain(&call.gather)
            .flatten()
            .map(|t| t.input.length)
            .sum::<u64>();
        assert_eq!(moved, elements * 18);
    }
}

#[test]
fn truncated_misaligned_and_overflowing_views_are_refused() {
    let input = spans(1, 512 * 16 * 4);
    let output = spans(5, 512 * 16 * 2);
    let inbox = spans(9, 512 * 16 * 4);
    for bytes in [0, 512 * 16 * 4 - 1] {
        let mut bad = inbox.clone();
        bad[0].length = bytes;
        assert!(Call::layout(&bad, &input, &output, 512, 16).is_err());
    }
    let mut bad = input.clone();
    bad[2].offset += 2;
    assert!(Call::layout(&inbox, &bad, &output, 512, 16).is_err());
    assert!(Call::layout(&inbox, &input, &output, 512, 15).is_err());
    assert!(Call::layout(&inbox, &input, &output, 0, 16).is_err());
    let mut overflow = input[0];
    overflow.offset = u64::MAX - 1;
    assert!(subspan(overflow, 4, 4).is_err());
    assert!(subspan(input[0], u64::MAX, 4).is_err());
}

use runtime_bundle_abi::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(Clone, Debug, PartialEq)]
enum Action {
    Record(u64, u64),
    Wait(u64, u64),
    Copy(u64, u64, u64, Span, Span),
    Launch(u64, u32),
    Drain(u64),
}

#[derive(Default)]
struct RecordingHost {
    calls: RefCell<Vec<Action>>,
    failure: Option<(usize, BundleStatusV1)>,
}

impl RecordingHost {
    fn record(&self, action: Action) -> HostServiceResultV1<()> {
        let mut calls = self.calls.borrow_mut();
        let index = calls.len();
        calls.push(action);
        match self.failure {
            Some((at, status)) if at == index => Err(status),
            _ => Ok(()),
        }
    }
}

impl HostServicesBackendV1 for RecordingHost {
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
    synchronize_event(_event: BundleEventHandleV1) -> ();
    release_event(_event: BundleEventHandleV1) -> ();
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
    copy_host_to_device(_stream: BundleStreamHandleV1, _destination: DeviceBufferSpanV1, _source: &[u8]) -> ();
    copy_device_to_host(_stream: BundleStreamHandleV1, _destination: &mut [u8], _source: DeviceBufferSpanV1) -> ();
    copy_device_to_device(_stream: BundleStreamHandleV1, _destination: DeviceBufferSpanV1, _source: DeviceBufferSpanV1) -> ();
        }
    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.record(Action::Drain(stream.raw))
    }
    fn record_event(
        &self,
        event: BundleEventHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        self.record(Action::Record(stream.raw, event.raw))
    }
    fn wait_event(
        &self,
        stream: BundleStreamHandleV1,
        event: BundleEventHandleV1,
    ) -> HostServiceResultV1<()> {
        self.record(Action::Wait(stream.raw, event.raw))
    }
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: Span,
        source_context: BundleContextHandleV1,
        source: Span,
    ) -> HostServiceResultV1<()> {
        self.record(Action::Copy(
            stream.raw,
            source_context.raw,
            destination_context.raw,
            source,
            destination,
        ))
    }
    fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        _config: LaunchConfigV1,
        _arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        self.record(Action::Launch(stream.raw, symbol.raw as u32))
    }
}

fn stream(id: u32) -> BundleStreamHandleV1 {
    BundleStreamHandleV1::from_parts(1, id)
}

fn ordered_ranks() -> Vec<OrderedRank> {
    (0..4)
        .map(|r| OrderedRank {
            owner: stream(21 + r),
            lane: stream(11 + r),
            context: BundleContextHandleV1::from_parts(1, r + 1),
            inbox: spans(9, 512 * 16 * 4)[r as usize],
            events: std::array::from_fn(|i| {
                BundleEventHandleV1::from_parts(1, 100 * r + i as u32 + 1)
            }),
            copied: std::array::from_fn(|i| {
                BundleEventHandleV1::from_parts(1, 100 * r + i as u32 + 20)
            }),
        })
        .collect()
}

fn resolved(id: u32) -> Resolved {
    use kernel_core::contract::{ArgumentList, LaunchPlan};
    crate::dispatch::resolve(
        &BTreeMap::from([("test".into(), BundleCudaSymbolHandleV1::from_parts(1, id))]),
        "test",
        LaunchPlan {
            grid: [1; 3],
            block: [32, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1; 3],
            pdl: false,
        },
        &ArgumentList(vec![]),
        &crate::dispatch::BufferBindings::default(),
    )
    .unwrap()
}

struct TestArithmetic {
    producers: bool,
    consumers: bool,
}
impl Arithmetic for TestArithmetic {
    fn rounds(&self) -> Result<Vec<Vec<Resolved>>, DispatchError> {
        Ok((0..4).map(|r| vec![resolved(200 + 4 * r)]).collect())
    }
    fn sums(&self) -> Result<Vec<Resolved>, DispatchError> {
        Ok((0..4).map(|r| resolved(300 + r)).collect())
    }
    fn producers(&self) -> Result<Option<Producers>, DispatchError> {
        Ok(self.producers.then(|| Producers {
            launches: (0..4)
                .map(|r| (0..4).map(|p| resolved(100 + 4 * r + p)).collect())
                .collect(),
            rounds: (0..4)
                .map(|r| {
                    (0..4)
                        .map(|owner| Some(resolved(200 + 4 * r + owner)))
                        .collect()
                })
                .collect(),
        }))
    }
    fn consumers(&self) -> Result<Option<Consumers>, DispatchError> {
        Ok(self.consumers.then(|| {
            (0..4)
                .map(|r| {
                    (0..4)
                        .map(|owner| vec![resolved(400 + 4 * r + owner)])
                        .collect()
                })
                .collect()
        }))
    }
}

// Reconstruct dependencies independently of the transport's phase counters.
fn ancestors(calls: &[Action]) -> (Vec<BTreeSet<usize>>, BTreeMap<u64, BTreeSet<usize>>) {
    let mut streams: BTreeMap<u64, BTreeSet<usize>> = BTreeMap::new();
    let mut events = BTreeMap::new();
    let mut before = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        let s = match call {
            Action::Record(s, _)
            | Action::Wait(s, _)
            | Action::Copy(s, ..)
            | Action::Launch(s, _)
            | Action::Drain(s) => *s,
        };
        let clock = streams.entry(s).or_default();
        if let Action::Wait(_, e) = call {
            clock.extend(events.get(e).expect("wait must follow a record"));
        }
        before.push(clock.clone());
        clock.insert(i);
        if let Action::Record(_, e) = call {
            events.insert(*e, clock.clone());
        }
    }
    (before, streams)
}

#[test]
fn ordered_execution_keeps_copy_bytes_and_dependencies_in_every_mode() {
    for payload in [Payload::F32, Payload::Bf16] {
        for push in [false, true] {
            for producers in [false, true] {
                for consumers in [false, true] {
                    let host = Rc::new(RecordingHost::default());
                    let table = HostServicesTableV1::new(Rc::clone(&host));
                    let ranks = ordered_ranks();
                    let streams = [1, 2, 3, 4].map(stream);
                    let inputs = spans(1, 512 * 16 * payload.bytes());
                    let outputs = spans(5, 512 * 16 * 2);
                    let inboxes = ranks.iter().map(OrderedRank::inbox).collect::<Vec<_>>();
                    let call =
                        Call::with_payload(&inboxes, &inputs, &outputs, 512, 16, payload).unwrap();
                    let arithmetic = TestArithmetic {
                        producers,
                        consumers,
                    };
                    call.execute_ordered(table.caller(), &ranks, &streams, push, &arithmetic)
                        .unwrap();
                    let first_len = host.calls.borrow().len();
                    // The next call must wait for each destination's intervening
                    // compute before overwriting that rank's reusable inbox.
                    for (rank, &stream) in streams.iter().enumerate() {
                        resolved(500 + rank as u32)
                            .submit(table.caller(), stream)
                            .unwrap();
                    }
                    let second = host.calls.borrow().len();
                    call.execute_ordered(table.caller(), &ranks, &streams, push, &arithmetic)
                        .unwrap();
                    let calls = host.calls.borrow();
                    let (before, ends) = ancestors(&calls);
                    assert_eq!(&calls[..first_len], &calls[second..]);
                    for start in [0, second] {
                        let index = |symbol| {
                            (start..start + first_len)
                                .find(|&i| matches!(calls[i], Action::Launch(_, s) if s == symbol))
                                .unwrap()
                        };
                        let mut pairs = BTreeSet::new();
                        for i in start..start + first_len {
                            let Action::Copy(issuer, src, dst, input, output) = calls[i] else {
                                continue;
                            };
                            let source = (src as u32 - 1) as usize;
                            let destination = (dst as u32 - 1) as usize;
                            assert_ne!(source, destination);
                            let scatter = input.buffer.raw as u32 <= 4;
                            assert!(pairs.insert((scatter, source, destination)));
                            let issuing_rank = if push { source } else { destination };
                            let lane = if scatter { producers } else { consumers };
                            assert_eq!(
                                issuer,
                                if lane {
                                    ranks[issuing_rank].lane.raw
                                } else {
                                    streams[issuing_rank].raw
                                }
                            );
                            if scatter {
                                if start == second {
                                    assert!(
                                        before[i].contains(&(first_len + destination)),
                                        "copy overwrites an inbox before its previous consumer"
                                    );
                                }
                                let bytes = 128 * 16 * payload.bytes();
                                assert_eq!(
                                    input,
                                    subspan(inputs[source], destination as u64 * bytes, bytes)
                                        .unwrap()
                                );
                                assert_eq!(
                                    output,
                                    subspan(inboxes[destination], source as u64 * bytes, bytes)
                                        .unwrap()
                                );
                                let round = index(
                                    200 + source as u32 * 4
                                        + if producers { destination as u32 } else { 0 },
                                );
                                assert!(
                                    before[i].contains(&round),
                                    "copy precedes rounded payload"
                                );
                                assert!(
                                    before[index(300 + destination as u32)].contains(&i),
                                    "sum precedes incoming copy"
                                );
                                if producers {
                                    let phase = (destination + 3 - source) % 4;
                                    assert!(
                                        before[round].contains(&index(
                                            100 + source as u32 * 4 + phase as u32
                                        ))
                                    );
                                }
                            } else {
                                let bytes = 128 * 16 * 2;
                                assert_eq!(
                                    input,
                                    subspan(outputs[source], source as u64 * bytes, bytes).unwrap()
                                );
                                assert_eq!(
                                    output,
                                    subspan(outputs[destination], source as u64 * bytes, bytes)
                                        .unwrap()
                                );
                                assert!(
                                    before[i].contains(&index(300 + source as u32)),
                                    "gather precedes owner sum"
                                );
                                if consumers {
                                    assert!(
                                        before[index(400 + destination as u32 * 4 + source as u32)]
                                            .contains(&i)
                                    );
                                }
                            }
                            for s in streams {
                                assert!(ends[&s.raw].contains(&i), "copy escapes final join");
                            }
                            if start == 0 {
                                for r in 0..4 {
                                    assert!(
                                        before[second + index(300 + r)].contains(&i),
                                        "next call reuses live peer data"
                                    );
                                }
                            }
                        }
                        assert_eq!(pairs.len(), 24);
                        for r in 0..4 {
                            if producers {
                                assert!(before[index(300 + r)].contains(&index(100 + 4 * r + 3)));
                            }
                            if consumers {
                                assert!(before[index(400 + 5 * r)].contains(&index(300 + r)));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ordered_execution_stops_at_each_failed_host_call() {
    for push in [false, true] {
        for intervals in [false, true] {
            let ranks = ordered_ranks();
            let streams = [1, 2, 3, 4].map(stream);
            let call = Call::with_payload(
                &spans(9, 512 * 16 * 4),
                &spans(1, 512 * 16 * 2),
                &spans(5, 512 * 16 * 2),
                512,
                16,
                Payload::Bf16,
            )
            .unwrap();
            let arithmetic = TestArithmetic {
                producers: intervals,
                consumers: intervals,
            };
            let host = Rc::new(RecordingHost::default());
            let table = HostServicesTableV1::new(Rc::clone(&host));
            call.execute_ordered(table.caller(), &ranks, &streams, push, &arithmetic)
                .unwrap();
            let expected = host.calls.borrow().clone();
            for at in 0..expected.len() {
                let host = Rc::new(RecordingHost {
                    failure: Some((at, BundleStatusV1::HostError)),
                    ..Default::default()
                });
                let table = HostServicesTableV1::new(Rc::clone(&host));
                assert!(matches!(
                    call.execute_ordered(table.caller(), &ranks, &streams, push, &arithmetic),
                    Err(DispatchError::Host(BundleAbiError::Status(
                        BundleStatusV1::HostError
                    )))
                ));
                assert_eq!(*host.calls.borrow(), expected[..=at]);
            }
        }
    }
}

#[test]
fn push_probe_checks_every_pair_and_only_first_refusal_selects_pull() {
    let ranks = ordered_ranks();
    let host = Rc::new(RecordingHost::default());
    let table = HostServicesTableV1::new(Rc::clone(&host));
    assert!(pushes(table.caller(), &ranks).unwrap());
    let expected = host.calls.borrow().clone();
    assert_eq!(expected.len(), 36);
    let mut pairs = BTreeSet::new();
    for triple in expected.chunks_exact(3) {
        let [
            Action::Record(owner, event),
            Action::Wait(issuer, waited),
            Action::Copy(s, src, dst, input, output),
        ] = triple
        else {
            panic!("probe must order the destination allocation before the copy")
        };
        assert_eq!(event, waited);
        assert_eq!(s, issuer);
        assert_eq!(*owner, ranks[(*dst as u32 - 1) as usize].owner.raw);
        assert_eq!(*s, ranks[(*src as u32 - 1) as usize].owner.raw);
        assert_eq!((input.length, output.length), (16, 16));
        assert_ne!(src, dst);
        assert!(pairs.insert((src, dst)));
    }
    assert_eq!(pairs.len(), 12);
    for status in [BundleStatusV1::InvalidArgument, BundleStatusV1::HostError] {
        for at in 0..expected.len() {
            let host = Rc::new(RecordingHost {
                failure: Some((at, status)),
                ..Default::default()
            });
            let table = HostServicesTableV1::new(Rc::clone(&host));
            let result = pushes(table.caller(), &ranks);
            if at == 2 && status == BundleStatusV1::InvalidArgument {
                assert!(!result.unwrap());
            } else {
                assert!(matches!(result, Err(BundleAbiError::Status(s)) if s == status));
            }
            assert_eq!(*host.calls.borrow(), expected[..=at]);
        }
    }
}

#[test]
fn ordered_drain_keeps_lane_order_and_stops_at_first_failure() {
    let ranks = ordered_ranks();
    let streams = [1, 2, 3, 4].map(stream);
    let sides = [31, 32, 33, 34].map(stream);
    let expected =
        [1, 2, 3, 4, 11, 12, 13, 14, 31, 32, 33, 34].map(|id| Action::Drain(stream(id).raw));
    for failure in
        std::iter::once(None).chain((0..12).map(|i| Some((i, BundleStatusV1::HostError))))
    {
        let host = Rc::new(RecordingHost {
            failure,
            ..Default::default()
        });
        let table = HostServicesTableV1::new(Rc::clone(&host));
        let result = drain(table.caller(), streams, &ranks, sides);
        assert_eq!(result.is_err(), failure.is_some());
        let count = failure.map_or(12, |(i, _)| i + 1);
        assert_eq!(*host.calls.borrow(), expected[..count]);
    }
}
