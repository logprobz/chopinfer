//! Synthetic state writes on the actual host calls, including copies during
//! commit and capture. Tags stand for row arithmetic; this checks addressing,
//! dispatch, retention and rollback, not GPU numerical correctness.
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode_with_cuts};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

thread_local! { pub(super) static REPLAY: RefCell<Option<Replay>> = const { RefCell::new(None) }; }

type Span = DeviceBufferSpanV1;
type Sink = RecordingSink<super::conformance::Limits>;
type ContextState = (u32, u64, u64, Vec<u64>, Option<(u64, u64)>, u64);

fn part(span: Span, offset: u64, length: u64) -> Span {
    assert!(offset + length <= span.length);
    Span::new(span.buffer, span.offset + offset, length)
}
fn contains(outer: Span, inner: Span) -> bool {
    outer.buffer == inner.buffer
        && inner.offset >= outer.offset
        && inner.offset + inner.length <= outer.offset + outer.length
}
pub(super) fn write(host: &FakeHost, destination: Span, bytes: Vec<u8>) {
    assert_eq!(destination.length as usize, bytes.len());
    host.memory.borrow_mut().write(
        destination,
        vec![(destination.length, Origin::Uploaded(bytes.into(), 0))],
    );
}
fn word(host: &FakeHost, span: Span, row: u64) -> u32 {
    let view = part(span, row * 4, 4);
    if host
        .memory
        .borrow()
        .read(view)
        .iter()
        .any(|(_, origin)| matches!(origin, Origin::Held(..)))
    {
        let uploads = host.uploads.borrow();
        let (&(_, start), bytes) = uploads
            .range(..=(view.buffer.raw, view.offset))
            .rev()
            .find(|((buffer, start), bytes)| {
                *buffer == view.buffer.raw
                    && *start <= view.offset
                    && *start + bytes.len() as u64 >= view.offset + 4
            })
            .unwrap_or_else(|| panic!("uninitialized word: {view:?}"));
        return u32::from_le_bytes(
            bytes[(view.offset - start) as usize..(view.offset - start + 4) as usize]
                .try_into()
                .unwrap(),
        );
    }
    u32::from_le_bytes(host.memory.borrow().bytes(view).try_into().unwrap())
}
fn tag(host: &FakeHost, span: Span, value: u32) {
    write(
        host,
        span,
        value.to_le_bytes().repeat(span.length as usize / 4),
    );
}

#[derive(Default)]
pub(super) struct Replay {
    target: Vec<Span>,
    drafter: Vec<Span>,
    // The arithmetic oracle supplies row values to feature packing, while
    // real gather indices, copy spans and kernel metadata select their rows.
    rows: [Vec<(u32, u32)>; 2],
    phase: BTreeMap<u32, usize>,
    projected: BTreeMap<u32, Vec<u32>>,
    dispatches: Vec<(String, u32)>,
    tail_writes: usize,
    feature_writes: usize,
    ring_writes: usize,
}
impl Replay {
    pub(super) fn launch(
        &mut self,
        host: &FakeHost,
        stream: BundleStreamHandleV1,
        name: &str,
        config: LaunchConfigV1,
        args: &[KernelArgumentV1],
    ) {
        let span = |index: usize| match args[index] {
            KernelArgumentV1::DeviceBuffer(span) => span,
            _ => panic!("{name}: argument {index} is not a buffer"),
        };
        let scalar = |index: usize| match args[index] {
            KernelArgumentV1::ScalarU32(value) => value,
            _ => panic!("{name}: argument {index} is not a word"),
        };
        let rank = host.device.stream_ranks.borrow()[&stream.slot()];
        if name.contains("_compressor_") && name.ends_with("_pool_v1") {
            let verify = name.contains("_verify_");
            let live = word(host, span(4), 0) as u64;
            let tail = span(if verify { 7 } else { 6 });
            for row in 0..live {
                let position = word(host, span(2), row);
                let item = word(host, span(3), row);
                if position == u32::MAX || item == u32::MAX || position % 2 == 1 {
                    continue;
                }
                let id = if verify {
                    let mut start = row;
                    while start > 0 && word(host, span(3), start - 1) == item {
                        start -= 1;
                    }
                    word(host, span(6), u64::from(item) * 6 + row - start)
                } else {
                    if (row + 1..live).any(|next| {
                        word(host, span(3), next) == item && word(host, span(2), next) != u32::MAX
                    }) {
                        continue;
                    }
                    word(host, span(5), u64::from(item))
                };
                tag(
                    host,
                    part(tail, u64::from(id) * 4096, 4096),
                    position ^ 0x74610000,
                );
                self.tail_writes += 1;
            }
        }
        // The feature gather has a BF16 hidden-width vector and three
        // target planes. Other row gathers retain their ordinary fake behavior.
        if name.contains("gather") && args.len() == 5 {
            let width = u64::from(FLASH.shape.hidden) * 2;
            let vectors = span(0).length / width;
            if vectors == 3 * u64::from(model_deepseek41_contract::serving::PREFILL_TOKENS) {
                let count = word(host, span(2), 0);
                let rows = &self.rows[*self.phase.get(&rank).unwrap_or(&0)];
                for vector in 0..u64::from(count) {
                    let index = word(host, span(1), vector);
                    let row = index % model_deepseek41_contract::serving::PREFILL_TOKENS;
                    let target = index / model_deepseek41_contract::serving::PREFILL_TOKENS;
                    let &(_, position) = rows.get(row as usize).expect("live feature row");
                    tag(
                        host,
                        part(span(3), vector * width, width),
                        position ^ (target << 24),
                    );
                    self.feature_writes += 1;
                }
            }
        }
        if name.contains("_dspark_quantize_w15360_bf16_") {
            let rows = config.grid_x;
            let values = (0..u64::from(rows))
                .map(|row| word(host, span(0), row * 7680))
                .collect();
            self.projected.insert(rank, values);
        }
        if name.contains("_n5120_k15360_") {
            // The projection's live-count pointer precedes its status pointer.
            let buffers: Vec<_> = args
                .iter()
                .filter_map(|arg| match arg {
                    KernelArgumentV1::DeviceBuffer(s) => Some(*s),
                    _ => None,
                })
                .collect();
            if let Some(values) = self.projected.get(&rank) {
                let rows = values.len() as u32;
                assert!(!buffers.is_empty());
                self.dispatches.push((name.to_owned(), rows));
            }
        }
        let fused = name == "chop_deepseek41_window_fused_native_v1";
        if name != "chop_deepseek41_latent_window_store_v1" && !fused {
            return;
        }
        let (positions, slots, last, codes, scales, live, cells, circular) = if fused {
            (
                span(10),
                span(11),
                span(13),
                span(14),
                span(15),
                span(4),
                scalar(17),
                scalar(18),
            )
        } else {
            (
                span(2),
                span(3),
                span(5),
                span(6),
                span(7),
                span(8),
                scalar(10),
                scalar(11),
            )
        };
        if circular == 0 {
            return;
        }
        let drafter = self.drafter.iter().any(|&s| contains(s, codes));
        let target = self.target.iter().any(|&s| contains(s, codes));
        if !drafter && !target {
            return;
        }
        let rows = u64::from(word(host, live, 0)).min(u64::from(config.grid_x));
        for row in 0..rows {
            let position = word(host, positions, row);
            let slot = word(host, slots, row);
            if position == u32::MAX || slot == u32::MAX {
                continue;
            }
            let end = word(host, last, u64::from(slot));
            if u64::from(position) + u64::from(cells) <= u64::from(end) {
                continue;
            }
            let value = if drafter {
                let values = self.projected.get(&rank).unwrap_or_else(|| {
                    panic!(
                        "missing main inputs: {:?}",
                        host.device
                            .journal
                            .borrow()
                            .iter()
                            .rev()
                            .filter(|l| l.starts_with("launch "))
                            .take(12)
                            .map(|l| l.split_whitespace().nth(2).unwrap().to_owned())
                            .collect::<Vec<_>>()
                    )
                });
                // Distinguish the main projection's two arithmetic routes.
                values[row as usize] ^ if values.len() > 64 { 0x40000000 } else { 0 }
            } else {
                let phase = usize::from(self.rows[1].contains(&(slot, position)));
                self.phase.insert(rank, phase);
                position
            };
            for (plane, width) in [(codes, 512), (scales, 16)] {
                tag(
                    host,
                    part(
                        plane,
                        (u64::from(slot) * u64::from(cells) + u64::from(position % cells)) * width,
                        width,
                    ),
                    value,
                );
            }
            self.ring_writes += 1;
        }
    }
}

#[derive(Clone, Copy)]
struct Item {
    request: u64,
    slot: u64,
    start: u64,
    end: u64,
    prompt: u64,
    phase: u8,
    anchor: bool,
}
impl Item {
    fn prefill(slot: u64, start: u64, end: u64, prompt: u64, anchor: bool) -> Self {
        Self {
            request: slot + 1,
            slot,
            start,
            end,
            prompt,
            phase: 0,
            anchor,
        }
    }
}
fn batch(items: &[Item]) -> Vec<u8> {
    let batch = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        items
            .iter()
            .map(|i| {
                let tokens = (i.start..i.end).map(|p| p as u32).collect();
                if i.phase == 2 {
                    BatchItemV1::verify(i.request, (i.end - i.start) as u32, false, tokens)
                } else {
                    BatchItemV1::new(
                        i.request,
                        i.phase == 1,
                        (i.end - i.start) as u32,
                        i.phase != 0 || i.end == i.prompt,
                        tokens,
                    )
                    .with_anchor(i.anchor)
                }
            })
            .collect(),
        items
            .iter()
            .map(|i| {
                BatchAppendV1::new(
                    i.start,
                    i.end - i.start,
                    i.end,
                    (0..i.end.div_ceil(64)).map(|p| i.slot * 24 + p).collect(),
                    None,
                )
            })
            .collect(),
    )
    .with_state_slots(items.iter().map(|i| i.slot).collect())
    .encode()
    .unwrap();
    encode_with_cuts(
        &batch,
        &items
            .iter()
            .filter(|i| i.phase == 0)
            .map(|i| Boundary {
                request_id: i.request,
                prompt_tokens: i.prompt,
                chunk_end: i.end,
                chunk_tokens: 8192,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

struct Run {
    host: Rc<Host>,
    instance: BundleInstanceV1,
    states: Vec<crate::buffers::StateSpans>,
    dspark: bool,
}
impl Run {
    fn new(dspark: bool, fused: bool) -> Self {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.kv_pages.set(2048);
        host.drafter.set(dspark);
        if dspark {
            host.speculation.set(Some((5, 32)));
        }
        let mut record = crate::config::tests::record(&host.shape, host.kv, 8, false);
        for (name, value) in [
            ("prefill_chunk_tokens", 8192.into()),
            ("prefill_upper_skip", true.into()),
            ("prefill_anchor_items", if fused { 8 } else { 0 }.into()),
            ("dspark_graph_handles_per_rank", 0.into()),
        ] {
            record = crate::config::tests::with_option(record, name, value);
        }
        let services =
            HostServicesTableV1::new(Rc::new(super::dspark::MappedHost::new(Rc::clone(&host))));
        let instance = api()
            .with_prefill_api(crate::api::BUNDLE_DEEPSEEK41_PREFILL_API_V1)
            .unwrap()
            .with_prefill_cuts_api(crate::instance::prefill::PREFILL_CUTS_API)
            .unwrap()
            .create(services, &record)
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        let (spares, states) = crate::instance::anchor_spans(instance.handle());
        for state in &states {
            for &(_, span) in &state.buffers {
                write(&host, span, vec![0xa5; span.length as usize]);
            }
        }
        for spare in &spares {
            write(&host, spare.zeros, vec![0; spare.zeros.length as usize]);
        }
        REPLAY.with(|replay| {
            replay.replace(Some(Replay {
                target: states
                    .iter()
                    .map(|s| s.named("window.ring").unwrap())
                    .chain(spares.iter().filter_map(|s| s.buffers[0]))
                    .collect(),
                drafter: states
                    .iter()
                    .filter_map(|s| s.named("dspark.ring"))
                    .collect(),
                ..Replay::default()
            }))
        });
        Self {
            host,
            instance,
            states,
            dspark,
        }
    }
    fn execute(&mut self, items: &[Item], cancel: bool) {
        REPLAY.with(|replay| {
            let mut replay = replay.borrow_mut();
            let replay = replay.as_mut().unwrap();
            replay.rows = std::array::from_fn(|phase| {
                items
                    .iter()
                    .filter(|i| usize::from(i.phase != 0) == phase)
                    .flat_map(|i| (i.start..i.end).map(move |p| (i.slot as u32, p as u32)))
                    .collect()
            });
            replay.phase.clear();
        });
        self.instance
            .execute_prefill_cuts(&batch(items), &mut Sink::default())
            .unwrap_or_else(|e| panic!("execute: {e:?}: {:?}", self.host.logs.borrow()));
        let result = if cancel {
            self.instance.cancel()
        } else if items.iter().any(|i| i.phase == 2) {
            self.instance.commit_accepted(
                &items
                    .iter()
                    .filter(|i| i.phase == 2)
                    .map(|i| BundleKeptRowsV1 {
                        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                        request_id: i.request,
                        kept_rows: (i.end - i.start) as u32,
                        reserved: 0,
                        reserved_tail: 0,
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            self.instance.commit()
        };
        result.unwrap_or_else(|e| panic!("commit: {e:?}: {:?}", self.host.logs.borrow()));
    }
    fn capture(&mut self, item: Item, end: u64) -> Vec<u8> {
        let pages: Vec<_> = (1024..1024 + self.instance.describe_prefix_state().unwrap()).collect();
        let before = self.host.device.journal.borrow().len();
        self.instance
            .capture_prefix_state(item.request, end, &pages)
            .unwrap_or_else(|e| panic!("capture: {e:?}: {:?}", self.host.logs.borrow()));
        let lines = &self.host.device.journal.borrow()[before..];
        let copies = super::prefill_anchor::copies(lines);
        // Snapshot copies are last, after any drafter catch-up copies.
        // Every captured destination is in the serving KV pool.
        let declaration =
            model_deepseek41_contract::state::request_state(&FLASH, KvStorage::Native, self.dspark);
        let bytes = bundle_runtime::snapshot::Table::new(
            &declaration,
            bundle_runtime::state::Extent {
                served: 8,
                bucket: 8,
                verify_rows: if self.dspark { 6 } else { 0 },
            },
            bundle_runtime::state::Holder::Slot(item.slot as u32),
        )
        .unwrap()
        .header_at();
        let mut result = Vec::new();
        let mut remaining = 4 * bytes;
        let mut first = None;
        for &(_, (buffer, offset, length)) in copies.iter().rev() {
            if remaining == 0 {
                break;
            }
            assert!(length <= remaining);
            let data = self.host.memory.borrow().bytes(Span::new(
                BundleDeviceBufferHandleV1 { raw: buffer },
                offset,
                length,
            ));
            result.push(data);
            first = Some((buffer, offset));
            remaining -= length;
        }
        assert_eq!(remaining, 0);
        result.reverse();
        let (buffer, offset) = first.unwrap();
        result.push(self.host.memory.borrow().bytes(Span::new(
            BundleDeviceBufferHandleV1 { raw: buffer },
            offset + bytes,
            declaration.header_bytes,
        )));
        result.into_iter().flatten().collect()
    }
    fn slot(&self, slot: u64) -> Vec<Vec<u8>> {
        let declaration =
            model_deepseek41_contract::state::request_state(&FLASH, KvStorage::Native, self.dspark);
        let extent = bundle_runtime::state::Extent {
            served: 8,
            bucket: 8,
            verify_rows: if self.dspark { 6 } else { 0 },
        };
        let mut result = Vec::new();
        for state in &self.states {
            for buffer in &declaration.buffers {
                let span = state.named(buffer.name).unwrap();
                let entries = buffer.entries(extent);
                for layer in 0..buffer.layers {
                    let mut at = layer * buffer.layer_stride(entries);
                    for &plane in &buffer.planes {
                        result.push(self.host.memory.borrow().bytes(part(
                            span,
                            at + slot * plane,
                            plane,
                        )));
                        at += entries * plane;
                    }
                }
            }
        }
        result
    }
    fn context(&self, request: u64) -> Option<ContextState> {
        crate::instance::dspark_context(self.instance.handle(), request).map(|c| {
            (
                c.slot,
                c.retained_start,
                c.end,
                c.projected().to_vec(),
                c.bulk(),
                c.lease(),
            )
        })
    }
    fn restore(&mut self, item: Item) {
        self.capture(item, item.end);
        self.instance.retire_request(item.request).unwrap();
        let pages: Vec<_> = (1024..1024 + self.instance.describe_prefix_state().unwrap()).collect();
        self.instance
            .restore_prefix_state_in_slot(
                item.request,
                &(0..item.end as u32).collect::<Vec<_>>(),
                &pages,
                item.slot,
            )
            .unwrap();
    }
}
impl Drop for Run {
    fn drop(&mut self) {
        REPLAY.with(|replay| replay.replace(None));
        if !std::thread::panicking() {
            self.instance.destroy().unwrap();
            assert_eq!(self.host.acquired.get(), self.host.released.get());
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Scenario {
    Cold,
    Odd,
    Near,
    Restore,
    Mixed,
    Eight,
    Cancel,
    TinyHead,
}
#[derive(PartialEq, Eq)]
struct Outcome {
    anchor: Vec<Vec<u8>>,
    end: Vec<Vec<u8>>,
    slots: Vec<Vec<Vec<u8>>>,
    committed_slots: Vec<Vec<Vec<u8>>>,
    before_capture: Vec<Option<ContextState>>,
    after_capture: Vec<Option<ContextState>>,
    dispatches: Vec<(String, u32)>,
}
fn run(dspark: bool, fused: bool, scenario: Scenario) -> Outcome {
    let mut run = Run::new(dspark, fused);
    let (start, end, count) = match scenario {
        Scenario::Cold | Scenario::Cancel => (0, 789, 1),
        Scenario::Odd | Scenario::Restore => (829, 900, 1),
        Scenario::Near => (768, 850, 1),
        Scenario::Mixed => (829, 900, 1),
        Scenario::Eight => (0, 789, 8),
        Scenario::TinyHead => (893, 900, 1),
    };
    if start > 0 {
        let setup = Item::prefill(0, 0, start, start, false);
        if start % 2 == 0 {
            run.execute(&[Item::prefill(0, 0, start - 1, start - 1, false)], false);
            run.execute(&[Item::prefill(0, start - 1, start, start, false)], false);
        } else {
            run.execute(&[setup], false);
        }
        if matches!(scenario, Scenario::Restore) {
            run.restore(setup);
        }
    }
    if matches!(scenario, Scenario::Mixed) {
        for slot in [1, 2] {
            run.execute(&[Item::prefill(slot, 0, 11, 11, false)], false);
        }
    }
    let mut items: Vec<_> = (0..count)
        .map(|slot| Item::prefill(slot, start, end + slot * 64, end + slot * 64, fused))
        .collect();
    if matches!(scenario, Scenario::Mixed) {
        items.push(Item {
            phase: 1,
            ..Item::prefill(1, 11, 12, 11, false)
        });
        items.push(Item {
            phase: 2,
            ..Item::prefill(2, 11, 14, 11, false)
        });
        if !dspark {
            assert_eq!(
                run.instance
                    .execute_prefill_cuts(&batch(&items), &mut Sink::default()),
                Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
            );
            items[2].phase = 1;
            items[2].end = 12;
        }
        items.rotate_left(1);
    }
    REPLAY.with(|replay| replay.borrow_mut().as_mut().unwrap().dispatches.clear());
    let cancelled = matches!(scenario, Scenario::Cancel);
    let anchor = end / 64 * 64;
    let head: Vec<_> = items
        .iter()
        .map(|&i| {
            if i.phase == 0 {
                Item {
                    end: i.end / 64 * 64,
                    anchor: false,
                    ..i
                }
            } else {
                i
            }
        })
        .collect();
    run.execute(if fused || cancelled { &items } else { &head }, cancelled);
    if cancelled {
        let pages: Vec<_> = (1024..1024 + run.instance.describe_prefix_state().unwrap()).collect();
        for boundary in [anchor, end] {
            assert!(
                run.instance
                    .capture_prefix_state(1, boundary, &pages)
                    .is_err()
            );
        }
        assert_eq!(
            crate::instance::ledger_state(run.instance.handle(), 0),
            (None, false)
        );
        return Outcome {
            anchor: Vec::new(),
            end: Vec::new(),
            slots: vec![run.slot(0)],
            committed_slots: Vec::new(),
            before_capture: vec![run.context(1)],
            after_capture: Vec::new(),
            dispatches: Vec::new(),
        };
    }
    let fused_slots = fused.then(|| items.iter().map(|i| run.slot(i.slot)).collect());
    let before_fused: Vec<_> = (0..count).map(|slot| run.context(slot + 1)).collect();
    let anchors: Vec<_> = items
        .iter()
        .filter(|i| i.phase == 0)
        .map(|&i| run.capture(i, i.end / 64 * 64))
        .collect();
    if !fused {
        let tails: Vec<_> = items
            .iter()
            .filter(|i| i.phase == 0)
            .map(|&i| Item {
                start: i.end / 64 * 64,
                anchor: false,
                ..i
            })
            .collect();
        run.execute(&tails, false);
    }
    let committed_slots =
        fused_slots.unwrap_or_else(|| items.iter().map(|i| run.slot(i.slot)).collect());
    let before_capture = if fused {
        before_fused
    } else {
        (0..count).map(|slot| run.context(slot + 1)).collect()
    };
    let ends: Vec<_> = items
        .iter()
        .filter(|i| i.phase == 0)
        .map(|&i| run.capture(i, i.end))
        .collect();
    for slot in 0..count {
        assert_eq!(
            crate::instance::ledger_state(run.instance.handle(), slot as u32),
            (Some((slot + 1, end + slot * 64)), false)
        );
    }
    let slots = items.iter().map(|i| run.slot(i.slot)).collect();
    let after_capture = (0..count).map(|slot| run.context(slot + 1)).collect();
    let mut dispatches = REPLAY.with(|replay| {
        let mut replay = replay.borrow_mut();
        let replay = replay.as_mut().unwrap();
        assert!(
            replay.tail_writes > 0,
            "compressor tails must contain generated bytes"
        );
        assert!(replay.ring_writes > 0);
        if dspark {
            assert!(replay.feature_writes > 0);
            assert!(!replay.dispatches.is_empty());
        }
        replay.dispatches.clone()
    });
    dispatches.sort();
    Outcome {
        anchor: anchors,
        end: ends,
        slots,
        committed_slots,
        before_capture,
        after_capture,
        dispatches,
    }
}

fn compare(scenario: Scenario) {
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for dspark in [false, true] {
        let fused = run(dspark, true, scenario);
        let split = run(dspark, false, scenario);
        assert!(
            fused.anchor == split.anchor,
            "{scenario:?} dspark={dspark}: anchor snapshot bytes"
        );
        assert!(
            fused.end == split.end,
            "{scenario:?} dspark={dspark}: end snapshot bytes"
        );
        assert!(
            fused.committed_slots == split.committed_slots,
            "{scenario:?} dspark={dspark}: committed slot bytes"
        );
        assert!(
            fused.slots == split.slots,
            "{scenario:?} dspark={dspark}: slot bytes"
        );
        assert_eq!(
            fused.before_capture, split.before_capture,
            "{scenario:?}: commit context"
        );
        assert_eq!(
            fused.after_capture, split.after_capture,
            "{scenario:?}: captured context"
        );
        assert_eq!(
            fused.dispatches, split.dispatches,
            "{scenario:?}: main projection journal"
        );
    }
}
#[test]
fn cold_start_matches_split() {
    compare(Scenario::Cold);
}
#[test]
fn odd_start_matches_split() {
    compare(Scenario::Odd);
}
#[test]
fn near_anchor_matches_split() {
    compare(Scenario::Near);
}
#[test]
fn restored_start_matches_split() {
    compare(Scenario::Restore);
}
#[test]
fn mixed_decode_and_verify_matches_split() {
    compare(Scenario::Mixed);
}
#[test]
fn eight_anchors_match_split() {
    compare(Scenario::Eight);
}
#[test]
fn cancel_matches_unanchored_rollback() {
    compare(Scenario::Cancel);
}
#[test]
fn tiny_head_preserves_staged_tail() {
    compare(Scenario::TinyHead);
}
