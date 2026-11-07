//! Prompt-end anchors kept inside an upper-skip item
//! (`instance::execute::anchors`): the spares receive the bytes a chunk
//! ending at the anchor leaves in its slot, and a capture there copies them.
use super::prefill_skip::create_profile;
use super::*;
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode_with_cuts};

type Sink = RecordingSink<super::conformance::Limits>;

/// A prompt-ending prefill of `count` tokens after `previous`, its item
/// flagged to keep its rounded prompt end when `anchor`.
fn anchored(request: u64, slot: u64, previous: u64, count: u64, anchor: bool) -> Vec<u8> {
    chunk(request, slot, previous, count, true, anchor)
}

fn chunk(request: u64, slot: u64, previous: u64, count: u64, last: bool, anchor: bool) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + count;
    let inputs = (previous..end).map(|token| token as u32).collect();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, count as u32, last, inputs).with_anchor(anchor)],
        vec![BatchAppendV1::new(
            previous,
            count,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// `batch` in a cuts envelope ending request `request`'s prompt of
/// `prompt` tokens at `end`.
fn cuts(batch: &[u8], request: u64, prompt: u64, end: u64) -> Vec<u8> {
    encode_with_cuts(
        batch,
        &[Boundary {
            request_id: request,
            prompt_tokens: prompt,
            chunk_end: end,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

fn create(host: &Rc<Host>) -> BundleInstanceV1 {
    create_profile(host, 8192, true)
}

/// A journal span: (buffer, offset, length).
type Span = (u64, u64, u64);

/// The spans a journal line names, in order.
fn spans(line: &str) -> Vec<Span> {
    let number = |text: &str, key: &str| -> Option<(u64, usize)> {
        let at = text.find(key)? + key.len();
        let digits = text[at..].chars().take_while(char::is_ascii_digit).count();
        Some((text[at..at + digits].parse().ok()?, at + digits))
    };
    let mut found = Vec::new();
    let mut rest = line;
    while let Some((buffer, after)) = number(rest, "raw: ") {
        let (offset, after_offset) = number(&rest[after..], "offset: ").unwrap();
        let (length, after_length) = number(&rest[after + after_offset..], "length: ").unwrap();
        found.push((buffer, offset, length));
        rest = &rest[after + after_offset + after_length..];
    }
    found
}

fn grid(line: &str) -> u64 {
    let at = line.find("grid_x: ").unwrap() + "grid_x: ".len();
    line[at..]
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn raw(span: DeviceBufferSpanV1) -> Span {
    (span.buffer.raw, span.offset, span.length)
}

fn inside(span: Span, buffer: DeviceBufferSpanV1) -> bool {
    span.0 == buffer.buffer.raw
        && span.1 >= buffer.offset
        && span.1 + span.2 <= buffer.offset + buffer.length
}

const STORE: &str = " chop_deepseek41_latent_window_store_v1 ";
const LAYERS: u64 = 40;
/// One ring entry of a layer: 136 E4M3 cells of 512 codes, then their
/// UE8M0 scales.
const CODES: u64 = 136 * 512;
const SCALES: u64 = 136 * 16;

/// The store launches of `lines` that write into `spare`, each after a
/// store into the slot rings `rings`, checked as an anchor's window write
/// of the packed rows `rows` whose anchor is position `last + 1`: the same
/// store, over the same rows' keys, into anchor spare 0 of each layer.
fn check_anchor_stores(
    lines: &[String],
    rings: DeviceBufferSpanV1,
    spare: DeviceBufferSpanV1,
    zeros: DeviceBufferSpanV1,
    rows: std::ops::Range<u64>,
    entries: u64,
) {
    let stores: Vec<_> = lines
        .iter()
        .filter(|line| line.starts_with("launch ") && line.contains(STORE))
        .collect();
    let mut layer = 0;
    // The step's positions start where layer 0's full-width store reads
    // them; an upper part's store reads them from its first row on.
    let mut base = None;
    let mut slot_positions = None;
    for (at, line) in stores.iter().enumerate() {
        let spans = spans(line);
        // codes, scales, positions, row sequences, last positions, ring
        // codes, ring scales, live rows, status; a staged store also binds
        // its first positions.
        if spans.len() != 9 {
            continue;
        }
        if inside(spans[5], rings) {
            base.get_or_insert(spans[2]);
            slot_positions = Some(spans[2]);
            continue;
        }
        if !inside(spans[5], spare) {
            continue;
        }
        let previous = spans_of_previous(&stores[..at], rings);
        assert!(previous.is_some(), "an anchor store follows a slot store");
        let n = rows.end - rows.start;
        assert_eq!(grid(line), n);
        // The same key rows the slot store read, from the anchor's ring:
        // its part's transient rows start at the part's first packed row.
        let previous = previous.unwrap();
        let base = base.unwrap();
        let part = (slot_positions.unwrap().1 - base.1) / 4;
        assert_eq!(spans[0].0, previous[0].0);
        assert_eq!(spans[0].1, previous[0].1 + (rows.start - part) * 512);
        assert_eq!(spans[0].2, n * 512);
        assert_eq!(spans[1].0, previous[1].0);
        assert_eq!(spans[1].1, previous[1].1 + (rows.start - part) * 16);
        assert_eq!(spans[1].2, n * 16);
        assert_eq!(spans[2], (base.0, base.1 + rows.start * 4, n * 4));
        assert_eq!(spans[3], (zeros.buffer.raw, zeros.offset, n * 4));
        assert_eq!(spans[4], (base.0, base.1 + (rows.end - 1) * 4, 4));
        assert_eq!(spare.length, entries * LAYERS * (CODES + SCALES));
        let base = spare.offset + layer * entries * (CODES + SCALES);
        assert_eq!(spans[5], (spare.buffer.raw, base, CODES));
        assert_eq!(spans[6], (spare.buffer.raw, base + entries * CODES, SCALES));
        layer += 1;
    }
    assert_eq!(layer, LAYERS);
}

fn spans_of_previous(stores: &[&String], rings: DeviceBufferSpanV1) -> Option<Vec<Span>> {
    let line = stores.last()?;
    let spans = spans(line);
    (spans.len() == 9 && inside(spans[5], rings)).then_some(spans)
}

/// Whether `line` is a device copy or one of the copies a snapshot copy
/// kernel launch makes (`FakeHost::run_snapshot_copy`).
fn is_copy(line: &str) -> bool {
    line.starts_with("copy ") || line.starts_with("kernel copy ")
}

/// The copies of `lines`, a snapshot copy kernel's among them: (source,
/// destination).
pub(super) fn copies(lines: &[String]) -> Vec<(Span, Span)> {
    lines
        .iter()
        .filter(|line| is_copy(line))
        .map(|line| {
            let spans = spans(line);
            assert_eq!(spans.len(), 2, "{line}");
            (spans[0], spans[1])
        })
        .collect()
}

/// Snapshot copies follow all drafter catch-up copies. Keep exactly the
/// declared payload across the four ranks, so catch-up staging is not
/// mistaken for a read of the snapshot's slot or spare.
fn snapshot_copies(lines: &[String], payload: u64) -> Vec<(Span, Span)> {
    let mut bytes = 0;
    let mut captured: Vec<_> = copies(lines)
        .into_iter()
        .rev()
        .take_while(|(source, _)| {
            if bytes == payload {
                return false;
            }
            bytes += source.2;
            true
        })
        .collect();
    assert_eq!(bytes, payload);
    captured.reverse();
    captured
}

/// Only Flash TP4 upper-skip with cuts declares an anchor allowance.
#[test]
fn instances_declare_their_spares_as_grid_anchor_items() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), KV_LAYOUTS[0]);
        for (chunk, skip, declared) in [(8192, true, 8_u32), (0, false, 0), (8192, false, 0)] {
            let mut instance = create_profile(&host, chunk, skip);
            let expected = if ranks == Ranks::TP4 {
                declared.min(4)
            } else {
                0
            };
            assert_eq!(instance.prefill_anchor_items(), Ok(Some(expected)));
            instance.destroy().unwrap();
        }
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// An unaligned prompt's single anchored step writes every layer's ring of
/// its rounded end into the spare, from the rows a cut there would end
/// with, and seeds only the tails: its ring is inside the item. A capture
/// at the anchor copies the spares alone; the prompt end, the slot.
#[test]
fn an_anchored_prompt_keeps_its_rounded_end_in_one_step() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host);
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let (rings, tails) = (
        states[0].named("window.ring").unwrap(),
        states[0].named("compressor.tail").unwrap(),
    );
    let (spare_rings, spare_tails) = (spares[0].buffers[0].unwrap(), spares[0].buffers[1].unwrap());
    let first = host.device.journal.borrow().len();
    let request = cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789);
    let mut sink = Sink::default();
    instance
        .execute_prefill_cuts(&request, &mut sink)
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 0),
        (Some((1, 789)), false)
    );
    let step: Vec<String> = host.device.journal.borrow()[first..].to_vec();
    let launch = step.iter().position(|l| l.starts_with("launch ")).unwrap();
    // Every tail layer's slot entry seeds the spare before any launch.
    let seeds: Vec<_> = copies(&step[..launch])
        .into_iter()
        .filter(|(_, destination)| {
            spares[0]
                .buffers
                .iter()
                .flatten()
                .any(|&b| inside(*destination, b))
        })
        .collect();
    assert_eq!(seeds.len(), 3);
    for (layer, &(source, destination)) in seeds.iter().enumerate() {
        let layer = layer as u64;
        assert_eq!(
            source,
            (raw(tails).0, tails.offset + layer * 4 * 32768, 32768)
        );
        assert_eq!(
            destination,
            (
                raw(spare_tails).0,
                spare_tails.offset + layer * 4 * 32768,
                32768
            )
        );
    }
    check_anchor_stores(&step, rings, spare_rings, spares[0].zeros, 632..768, 4);
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    for (tokens, from_spares) in [(768, true), (789, false)] {
        let before = host.device.journal.borrow().len();
        instance.capture_prefix_state(1, tokens, &pages).unwrap();
        let captured = copies(&host.device.journal.borrow()[before..]);
        let pieces = captured
            .iter()
            .filter(|(source, _)| {
                [rings, tails, spare_rings, spare_tails]
                    .iter()
                    .any(|&buffer| inside(*source, buffer))
            })
            .collect::<Vec<_>>();
        // Pieces continue across pages, so some take two copies.
        assert_eq!(
            pieces.iter().map(|(source, _)| source.2).sum::<u64>(),
            LAYERS * (CODES + SCALES) + 3 * 32768
        );
        assert!(pieces.iter().all(|(source, _)| {
            from_spares == (inside(*source, spare_rings) || inside(*source, spare_tails))
        }));
    }
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// An anchor less than a ring after its item's start keeps the older cells
/// of its ring from the slot: the spare is seeded with the slot's rings and
/// tails, and the item's rows up to the anchor are written over them. The
/// captured anchor restores into another slot and continues to the prompt
/// end.
#[test]
fn an_anchor_near_its_item_start_seeds_the_rings_and_restores() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host);
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let (rings, tails) = (
        states[0].named("window.ring").unwrap(),
        states[0].named("compressor.tail").unwrap(),
    );
    let spare_rings = spares[0].buffers[0].unwrap();
    let grid = cuts(&chunk(1, 0, 0, 8192, false, false), 1, 8257, 8192);
    instance
        .execute_prefill_cuts(&grid, &mut Sink::default())
        .unwrap();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let first = host.device.journal.borrow().len();
    let request = cuts(&anchored(1, 0, 8192, 65, true), 1, 8257, 8257);
    instance
        .execute_prefill_cuts(&request, &mut Sink::default())
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let step: Vec<String> = host.device.journal.borrow()[first..].to_vec();
    let launch = step.iter().position(|l| l.starts_with("launch ")).unwrap();
    let seeds: Vec<_> = copies(&step[..launch])
        .into_iter()
        .filter(|(_, destination)| {
            spares[0]
                .buffers
                .iter()
                .flatten()
                .any(|&b| inside(*destination, b))
        })
        .collect();
    assert_eq!(seeds.len(), 2 * LAYERS as usize + 3);
    assert_eq!(
        seeds
            .iter()
            .filter(|(source, _)| inside(*source, rings))
            .map(|(source, _)| source.2)
            .sum::<u64>(),
        LAYERS * (CODES + SCALES)
    );
    assert!(
        seeds
            .iter()
            .all(|(source, _)| inside(*source, rings) || inside(*source, tails))
    );
    check_anchor_stores(&step, rings, spare_rings, spares[0].zeros, 0..64, 4);
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    instance.capture_prefix_state(1, 8256, &pages).unwrap();
    instance.retire_request(1).unwrap();
    instance
        .restore_prefix_state_in_slot(2, &(0..8256).collect::<Vec<_>>(), &pages, 1)
        .unwrap();
    let mut sink = Sink::default();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(2, 1, 8256, 1, false), 2, 8257, 8257),
            &mut sink,
        )
        .unwrap();
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 1),
        (Some((2, 8257)), false)
    );
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Flags need an instance allowance and a granule strictly inside the item.
#[test]
fn anchor_flags_need_spares_an_interior_granule_and_an_upper_skip() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        for skip in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), KV_LAYOUTS[0]);
            let mut instance = create_profile(&host, 8192, skip);
            let supported = ranks == Ranks::TP4 && skip;
            for rows in [768, 50, 789] {
                let result = instance.execute_prefill_cuts(
                    &cuts(&anchored(1, 0, 0, rows, true), 1, rows, rows),
                    &mut Sink::default(),
                );
                if supported && rows == 789 {
                    result.unwrap();
                    instance
                        .commit()
                        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
                    instance.retire_request(1).unwrap();
                } else {
                    assert_eq!(
                        result,
                        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
                    );
                }
            }
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

/// An unanchored step neither reads nor writes the allocated anchor spares.
#[test]
fn unanchored_steps_leave_spares_untouched() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host);
    let (spares, _) = crate::instance::anchor_spans(instance.handle());
    let first = host.device.journal.borrow().len();
    for (start, rows, last) in [(0, 8192, false), (8192, 65, true)] {
        instance
            .execute_prefill_cuts(
                &cuts(
                    &chunk(1, 0, start, rows, last, false),
                    1,
                    8257,
                    start + rows,
                ),
                &mut Sink::default(),
            )
            .unwrap();
        instance
            .commit()
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    }
    let lines = host.device.journal.borrow();
    assert!(
        lines[first..]
            .iter()
            .filter(|line| line.starts_with("launch "))
            .count()
            > 1000
    );
    for line in &lines[first..] {
        for span in spans(line) {
            assert!(
                !spares.iter().any(|spare| spare
                    .buffers
                    .iter()
                    .flatten()
                    .any(|&b| inside(span, b)))
            );
        }
    }
    drop(lines);
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// At TP4 an anchored prompt captures its rounded end from every rank's
/// spares, restores it and continues to the prompt end.
#[test]
fn tp4_anchored_prompt_captures_restores_and_continues() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host);
    let mut sink = Sink::default();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 4171, true), 1, 4171, 4171),
            &mut sink,
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    instance.capture_prefix_state(1, 4160, &pages).unwrap();
    instance.retire_request(1).unwrap();
    instance
        .restore_prefix_state_in_slot(2, &(0..4160).collect::<Vec<_>>(), &pages, 0)
        .unwrap();
    let mut sink = Sink::default();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(2, 0, 4160, 11, false), 2, 4171, 4171),
            &mut sink,
        )
        .unwrap();
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// An item resumed at an odd position that also ends on one seeds its own
/// compressor tail entry before pooling. The anchor's spare copies the
/// slot's tails before that copy, as a chunk ending at the even anchor
/// leaves them unseeded.
#[test]
fn anchor_seeds_precede_the_tails_own_seed_copies() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host);
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let tails = states[0].named("compressor.tail").unwrap();
    let spare_tails = spares[0].buffers[1].unwrap();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 789, false), 1, 789, 789),
            &mut Sink::default(),
        )
        .unwrap();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    instance.capture_prefix_state(1, 789, &pages).unwrap();
    instance.retire_request(1).unwrap();
    instance
        .restore_prefix_state_in_slot(2, &(0..789).collect::<Vec<_>>(), &pages, 0)
        .unwrap();
    let first = host.device.journal.borrow().len();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(2, 0, 789, 300, true), 2, 1089, 1089),
            &mut Sink::default(),
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let step: Vec<String> = host.device.journal.borrow()[first..].to_vec();
    let launch = step.iter().position(|l| l.starts_with("launch ")).unwrap();
    let before: Vec<_> = copies(&step[..launch])
        .into_iter()
        .filter(|(source, _)| inside(*source, tails))
        .collect();
    let anchors: Vec<_> = before
        .iter()
        .take_while(|(_, destination)| inside(*destination, spare_tails))
        .collect();
    assert_eq!(anchors.len(), 3);
    assert!(anchors.iter().all(|(source, _)| inside(*source, tails)));
    let rest = &before[anchors.len()..];
    assert_eq!(rest.len(), 3);
    assert!(
        rest.iter()
            .all(|(source, destination)| inside(*source, tails) && inside(*destination, tails))
    );
    instance.capture_prefix_state(2, 1088, &pages).unwrap();
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

fn dspark_host(ranks: Ranks) -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, ranks).unwrap(), KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 32)));
    host
}

/// The bytes of one raw feature row, and the cells of a raw history.
fn feature_row() -> (u64, u64) {
    let row = FLASH.shape.dspark.targets.len() as u64 * u64::from(FLASH.shape.hidden) * 2;
    let cells = u64::from(FLASH.shape.attention.window)
        + u64::from(model_deepseek41_contract::serving::MAX_VERIFY_ROWS);
    (row, cells)
}

/// The slot features and their anchor spare of rank 0.
fn feature_spans(instance: &BundleInstanceV1) -> (DeviceBufferSpanV1, DeviceBufferSpanV1) {
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let index = model_deepseek41_contract::state::request_state(&FLASH, KvStorage::Native, true)
        .buffers
        .iter()
        .position(|buffer| buffer.name == "dspark.features")
        .unwrap();
    (
        states[0].named("dspark.features").unwrap(),
        spares[0].buffers[index].unwrap(),
    )
}

/// History rows from `position` in entry `entry` of `buffer`.
fn history(buffer: DeviceBufferSpanV1, entry: u64, position: u64, rows: u64) -> Span {
    let (row, cells) = feature_row();
    (
        raw(buffer).0,
        buffer.offset + (entry * cells + position % cells) * row,
        rows * row,
    )
}

/// The copies of `lines` that write either feature history, in order.
fn feature_writes(
    lines: &[String],
    features: DeviceBufferSpanV1,
    spare: DeviceBufferSpanV1,
) -> Vec<(Span, Span)> {
    copies(lines)
        .into_iter()
        .filter(|&(_, destination)| inside(destination, features) || inside(destination, spare))
        .collect()
}

/// A DSpark prompt keeps its rounded end inside one step: the commit
/// retains the head's raw rows (640 to 768), copies its whole history into
/// the anchor's spare, then retains the tail (768 to 789). Both histories
/// match the item cut at 768. A capture at the anchor reads only spares, it
/// restores into another slot and continues to the prompt end.
#[test]
fn dspark_anchored_prompt_keeps_its_raw_history_at_the_anchor() {
    let host = dspark_host(Ranks::TP4);
    let mut instance = create(&host);
    let (features, spare) = feature_spans(&instance);
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789),
            &mut Sink::default(),
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let before = host.device.journal.borrow().len();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let lines = host.device.journal.borrow()[before..].to_vec();
    let writes = feature_writes(&lines, features, spare);
    let (_, cells) = feature_row();
    assert_eq!(
        writes
            .iter()
            .map(|&(_, destination)| destination)
            .collect::<Vec<_>>(),
        [
            history(features, 0, 640, 30),
            history(features, 0, 670, 98),
            history(spare, 0, 0, cells),
            history(features, 0, 768, 21),
        ]
    );
    assert_eq!(writes[2].0, history(features, 0, 0, cells));
    assert!(
        writes
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != 2)
            .map(|(_, copy)| copy)
            .all(|&(source, _)| !inside(source, features) && !inside(source, spare))
    );
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    // The anchor's capture comes last, so its snapshot is the one restored.
    for (tokens, from_spares) in [(789, false), (768, true)] {
        let before = host.device.journal.borrow().len();
        instance.capture_prefix_state(1, tokens, &pages).unwrap();
        let payload = LAYERS * (CODES + SCALES) + 3 * 32768 + cells * feature_row().0;
        let captured = snapshot_copies(&host.device.journal.borrow()[before..], 4 * payload);
        let read = |buffers: &[DeviceBufferSpanV1]| {
            captured
                .iter()
                .filter(|(source, _)| buffers.iter().any(|&buffer| inside(*source, buffer)))
                .map(|(source, _)| source.2)
                .sum::<u64>()
        };
        let slot: Vec<_> = ["window.ring", "compressor.tail", "dspark.features"]
            .iter()
            .map(|name| states[0].named(name).unwrap())
            .collect();
        let kept: Vec<_> = spares[0].buffers.iter().flatten().copied().collect();
        assert_eq!(read(if from_spares { &kept } else { &slot }), payload);
        assert_eq!(read(if from_spares { &slot } else { &kept }), 0);
    }
    instance.retire_request(1).unwrap();
    instance
        .restore_prefix_state_in_slot(2, &(0..768).collect::<Vec<_>>(), &pages, 1)
        .unwrap();
    let mut sink = Sink::default();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(2, 1, 768, 21, false), 2, 789, 789),
            &mut sink,
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let context = crate::instance::dspark_context(instance.handle(), 2).unwrap();
    assert_eq!((context.slot, context.end), (1, 789));
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// An anchor within the drafter's history of its item's start keeps every
/// row of the item before it. Such an item retains all of its rows, so the
/// slot needs no further row.
#[test]
fn dspark_anchor_near_its_item_start_keeps_every_earlier_row() {
    let host = dspark_host(Ranks::TP4);
    let mut instance = create(&host);
    let (features, spare) = feature_spans(&instance);
    instance
        .execute_prefill_cuts(
            &cuts(&chunk(1, 0, 0, 8192, false, false), 1, 8257, 8192),
            &mut Sink::default(),
        )
        .unwrap();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 8192, 65, true), 1, 8257, 8257),
            &mut Sink::default(),
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let before = host.device.journal.borrow().len();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let lines = host.device.journal.borrow()[before..].to_vec();
    let (_, cells) = feature_row();
    assert_eq!(
        feature_writes(&lines, features, spare)
            .iter()
            .map(|&(_, destination)| destination)
            .collect::<Vec<_>>(),
        [
            history(features, 0, 8192, 64),
            history(spare, 0, 0, cells),
            history(features, 0, 8256, 1),
        ]
    );
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    instance.capture_prefix_state(1, 8256, &pages).unwrap();
    instance.retire_request(1).unwrap();
    instance
        .restore_prefix_state_in_slot(2, &(0..8256).collect::<Vec<_>>(), &pages, 1)
        .unwrap();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(2, 1, 8256, 1, false), 2, 8257, 8257),
            &mut Sink::default(),
        )
        .unwrap();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(
        crate::instance::dspark_context(instance.handle(), 2)
            .unwrap()
            .end,
        8257
    );
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Each drafter ring write of an anchored prompt advances the count a
/// successor compares before it serves a step beside prefill items
/// (`prefetch::Cache::ring_writes`), one count a projection launch; in
/// each commit, the two status planes a launch saves on rank 0 attest the
/// launches. A 789-token prompt's
/// commit projects its head of 768 rows from staging, one launch a stage. A
/// 140-token prompt's head of 128 rows defers to a bulk range that its
/// tail's commit catches up, one launch a stage. A prompt restored at 800
/// tokens catches up its restored window, 672 to 800, before its anchored
/// step, two launches a stage where the window ring wraps; its commit then
/// catches up the head's 32 deferred rows, 800 to 832, in one launch while
/// the tail keeps packed staging. Nothing else in either step writes.
#[test]
fn dspark_anchored_commits_count_every_ring_write() {
    let stages = u64::from(FLASH.dspark_stages());
    for (case, executed, committed) in [
        ("staged head", 0, stages),
        ("bulk head", 0, stages),
        ("restored rows", 2 * stages, 1),
    ] {
        let host = dspark_host(Ranks::TP4);
        let mut instance = create(&host);
        let handle = instance.handle();
        let writes = || crate::instance::dspark_ring_writes_for_test(handle, 0);
        let (request, slot, start, end) = match case {
            "staged head" => (1, 0, 0, 789),
            "bulk head" => (1, 0, 0, 140),
            _ => {
                instance
                    .execute_prefill_cuts(
                        &cuts(&anchored(1, 0, 0, 800, false), 1, 800, 800),
                        &mut Sink::default(),
                    )
                    .unwrap();
                instance.commit().unwrap();
                let pages: Vec<_> =
                    (130..130 + instance.describe_prefix_state().unwrap()).collect();
                instance.capture_prefix_state(1, 800, &pages).unwrap();
                instance.retire_request(1).unwrap();
                instance
                    .restore_prefix_state_in_slot(2, &(0..800).collect::<Vec<_>>(), &pages, 1)
                    .unwrap();
                (2, 1, 800, 840)
            }
        };
        let before = writes();
        instance
            .execute_prefill_cuts(
                &cuts(
                    &anchored(request, slot, start, end - start, true),
                    request,
                    end,
                    end,
                ),
                &mut Sink::default(),
            )
            .unwrap_or_else(|e| panic!("{case}: {e:?}: {:?}", host.logs.borrow()));
        assert_eq!(writes() - before, executed, "{case}: the anchored step");
        let status = crate::instance::dspark_successor_status_for_test(handle)[0].0;
        let begin = host.device.journal.borrow().len();
        instance
            .commit()
            .unwrap_or_else(|e| panic!("{case}: {e:?}: {:?}", host.logs.borrow()));
        assert_eq!(
            writes() - before - executed,
            committed,
            "{case}: its commit"
        );
        let stream = host.rank_streams.borrow()[&0];
        let saves = host.device.journal.borrow()[begin..]
            .iter()
            .filter(|line| {
                line.starts_with(&format!("copy {stream} "))
                    && spans(line).get(1).is_some_and(|&span| inside(span, status))
            })
            .count();
        assert_eq!(saves as u64, 2 * committed, "{case}: rank 0's saved status");
        instance.retire_request(request).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Two prompt-ending items of one step that each keep an anchor.
fn two_anchored() -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let batch = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::new(1, false, 789, true, (0..789).collect()).with_anchor(true),
            BatchItemV1::new(2, false, 200, true, (0..200).collect()).with_anchor(true),
        ],
        vec![
            BatchAppendV1::new(0, 789, 789, (0..13).collect(), None),
            BatchAppendV1::new(0, 200, 200, (16..20).collect(), None),
        ],
    )
    .with_state_slots(vec![0, 1])
    .encode()
    .unwrap();
    encode_with_cuts(
        &batch,
        &[
            Boundary {
                request_id: 1,
                prompt_tokens: 789,
                chunk_end: 789,
                chunk_tokens: 8192,
            },
            Boundary {
                request_id: 2,
                prompt_tokens: 200,
                chunk_end: 200,
                chunk_tokens: 8192,
            },
        ],
    )
    .unwrap()
}

/// Each anchored item of a step takes its own spare, in item order, and a
/// capture at either anchor reads that spare's entry.
#[test]
fn anchored_items_of_one_step_each_keep_their_own_spare() {
    let host = dspark_host(Ranks::TP4);
    let mut instance = create(&host);
    let (features, spare) = feature_spans(&instance);
    instance
        .execute_prefill_cuts(&two_anchored(), &mut Sink::default())
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let before = host.device.journal.borrow().len();
    instance
        .commit()
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    let lines = host.device.journal.borrow()[before..].to_vec();
    let (row, cells) = feature_row();
    // Both heads retain before their histories are copied into the spares.
    // Each tail then retains at its original offset in the packed target.
    let anchor_writes: Vec<_> = feature_writes(&lines, features, spare)
        .iter()
        .map(|&(_, destination)| destination)
        .take(8)
        .collect();
    assert_eq!(
        anchor_writes,
        [
            history(features, 0, 640, 30),
            history(features, 0, 670, 98),
            history(features, 1, 64, 70),
            history(features, 1, 134, 58),
            history(spare, 0, 0, cells),
            history(spare, 1, 0, cells),
            history(features, 0, 768, 21),
            history(features, 1, 192, 8),
        ]
    );
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    for (request, tokens, entry) in [(1, 768, 0), (2, 192, 1)] {
        let before = host.device.journal.borrow().len();
        instance
            .capture_prefix_state(request, tokens, &pages)
            .unwrap();
        let captured = copies(&host.device.journal.borrow()[before..]);
        let entry_span = DeviceBufferSpanV1::new(
            spare.buffer,
            spare.offset + entry * cells * row,
            cells * row,
        );
        assert_eq!(
            captured
                .iter()
                .filter(|(source, _)| inside(*source, spare))
                .map(|(source, _)| {
                    assert!(inside(*source, entry_span));
                    source.2
                })
                .sum::<u64>(),
            cells * row
        );
    }
    for request in [1, 2] {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A cancelled anchored step publishes neither its anchor nor its end; the
/// next step runs as before, and a later anchored step keeps its anchor.
#[test]
fn a_cancelled_anchored_step_keeps_no_anchor() {
    for dspark in [false, true] {
        let host = if dspark {
            dspark_host(Ranks::TP4)
        } else {
            FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0])
        };
        let mut instance = create(&host);
        let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
        instance
            .execute_prefill_cuts(
                &cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789),
                &mut Sink::default(),
            )
            .unwrap();
        instance.cancel().unwrap();
        for tokens in [768, 789] {
            assert_eq!(
                instance.capture_prefix_state(1, tokens, &pages),
                Err(runtime_bundle_abi::BundleAbiError::Status(
                    BundleStatusV1::InvalidState
                )),
                "{dspark} {tokens}"
            );
        }
        instance
            .execute_prefill_cuts(
                &cuts(&anchored(2, 0, 0, 789, true), 2, 789, 789),
                &mut Sink::default(),
            )
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        instance
            .commit()
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        for tokens in [768, 789] {
            instance.capture_prefix_state(2, tokens, &pages).unwrap();
        }
        // The next step ends the anchor: its spare may be rewritten.
        instance
            .execute_prefill_cuts(
                &cuts(&anchored(3, 1, 0, 70, true), 3, 70, 70),
                &mut Sink::default(),
            )
            .unwrap();
        instance
            .commit()
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        assert!(instance.capture_prefix_state(2, 768, &pages).is_err());
        for request in [2, 3] {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// The diagnostic state dump of a capture at an anchor reads the anchor's
/// spares: the snapshot payload at the anchor, without the drafter rings
/// a restore rebuilds, and the pages up to it.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn anchor_state_dumps_read_their_spares() {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for dspark in [false, true] {
        let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
            .join("deepseek41-tests")
            .join(format!(
                "anchor-state-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("enabled"), []).unwrap();
        let host = if dspark {
            dspark_host(Ranks::TP4)
        } else {
            FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0])
        };
        let mut instance = create(&host);
        crate::instance::set_dspark_state_capture_root_for_test(
            instance.handle(),
            Some(root.clone()),
        );
        instance
            .execute_prefill_cuts(
                &cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789),
                &mut Sink::default(),
            )
            .unwrap();
        instance
            .commit()
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
        for tokens in [768, 789] {
            instance
                .capture_prefix_state(1, tokens, &pages)
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        }
        let read = |name: &str| {
            let directory = root.join(name).join("request-1");
            let metadata: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("state.json")).unwrap())
                    .unwrap();
            let buffers: Vec<String> = metadata["pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|piece| piece["buffer"].as_str().unwrap().to_owned())
                .collect();
            let bytes: u64 = metadata["pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|piece| piece["bytes"].as_u64().unwrap())
                .sum();
            assert_eq!(
                std::fs::metadata(directory.join("rank-0.bin"))
                    .unwrap()
                    .len(),
                bytes
            );
            (metadata, buffers)
        };
        let (anchor, buffers) = read("snapshot-768");
        assert_eq!(anchor["end"], 768);
        assert_eq!(anchor["retained_start"], 640);
        assert!(anchor["projected"].as_array().unwrap().is_empty());
        assert!(!buffers.iter().any(|name| name == "dspark.ring"));
        assert_eq!(buffers.iter().any(|name| name == "dspark.features"), dspark);
        assert_eq!(
            buffers.iter().filter(|name| *name == "window.ring").count(),
            80
        );
        let (end, buffers) = read("snapshot-789");
        assert_eq!(end["end"], 789);
        assert_eq!(buffers.iter().any(|name| name == "dspark.ring"), dspark);
        assert!(root.join("chunk-789/request-1/state.json").is_file());
        assert!(!root.join("chunk-768").exists());
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        std::fs::remove_dir_all(root).unwrap();
    }
}

/// Default DMA reductions preserve each rank's anchor writes before capture and
/// restore, including the DSpark raw history and delayed gather joins.
#[test]
fn dma_anchors_capture_restore_and_continue_with_both_completion_modes() {
    use super::prefill_engram::{completion_backend, create_backend_with_options};
    for speculative in [false, true] {
        for delayed in [false, true] {
            let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
            if speculative {
                host.speculation.set(Some((5, 32)));
            }
            let mut instance =
                create_backend_with_options(&host, true, completion_backend(&host, delayed), &[])
                    .unwrap();
            let (spares, states) = crate::instance::anchor_spans(instance.handle());
            assert_eq!(spares.len(), 4);
            assert_eq!(states.len(), 4);
            let first = host.device.journal.borrow().len();
            instance
                .execute_prefill_cuts(
                    &cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789),
                    &mut Sink::default(),
                )
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            let step = host.device.journal.borrow()[first..].to_vec();
            assert!(
                step.iter()
                    .any(|line| line.contains("chop_comm_owned_sum_"))
            );
            assert_eq!(
                step.iter()
                    .filter(|line| line.starts_with("async mapped wait "))
                    .count(),
                8
            );
            assert_eq!(
                step.iter()
                    .filter(|line| line.starts_with("async mapped complete"))
                    .count(),
                2
            );
            let launch = step
                .iter()
                .position(|line| line.starts_with("launch "))
                .unwrap();
            let seeds = copies(&step[..launch]);
            // The DMA helper serves one AR sequence or sixteen DSpark
            // sequences; the per-step allowance caps the latter at eight.
            let entries = if speculative { 8 } else { 1 };
            assert_eq!(instance.prefill_anchor_items(), Ok(Some(entries as u32)));
            for (spare, state) in spares.iter().zip(&states) {
                let tails = state.named("compressor.tail").unwrap();
                let kept = spare.buffers[1].unwrap();
                assert_eq!(kept.length, entries * 3 * 32768);
                assert_eq!(tails.length % (3 * 32768), 0);
                let slots = tails.length / (3 * 32768);
                let tail_seeds: Vec<_> = seeds
                    .iter()
                    .filter(|(_, destination)| inside(*destination, kept))
                    .collect();
                assert_eq!(tail_seeds.len(), 3);
                for (layer, &&(source, destination)) in tail_seeds.iter().enumerate() {
                    assert_eq!(
                        source,
                        (
                            tails.buffer.raw,
                            tails.offset + layer as u64 * slots * 32768,
                            32768
                        )
                    );
                    assert_eq!(
                        destination,
                        (
                            kept.buffer.raw,
                            kept.offset + layer as u64 * entries * 32768,
                            32768
                        )
                    );
                }
                check_anchor_stores(
                    &step,
                    state.named("window.ring").unwrap(),
                    spare.buffers[0].unwrap(),
                    spare.zeros,
                    632..768,
                    entries,
                );
            }
            let commit_start = host.device.journal.borrow().len();
            instance
                .commit()
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            if speculative {
                let index = model_deepseek41_contract::state::request_state(
                    &FLASH,
                    KvStorage::Native,
                    true,
                )
                .buffers
                .iter()
                .position(|buffer| buffer.name == "dspark.features")
                .unwrap();
                let commit = host.device.journal.borrow()[commit_start..].to_vec();
                let (_, cells) = feature_row();
                for (spare, state) in spares.iter().zip(&states) {
                    let features = state.named("dspark.features").unwrap();
                    let kept = spare.buffers[index].unwrap();
                    let writes = feature_writes(&commit, features, kept);
                    assert_eq!(
                        writes
                            .iter()
                            .map(|&(_, destination)| destination)
                            .collect::<Vec<_>>(),
                        [
                            history(features, 0, 640, 30),
                            history(features, 0, 670, 98),
                            history(kept, 0, 0, cells),
                            history(features, 0, 768, 21),
                        ]
                    );
                    assert_eq!(writes[2].0, history(features, 0, 0, cells));
                    assert!(
                        writes
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| *index != 2)
                            .map(|(_, copy)| copy)
                            .all(|&(source, _)| !inside(source, features) && !inside(source, kept))
                    );
                }
            }
            let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
            let before = host.device.journal.borrow().len();
            instance.capture_prefix_state(1, 768, &pages).unwrap();
            let payload = LAYERS * (CODES + SCALES)
                + 3 * 32768
                + if speculative {
                    feature_row().0 * feature_row().1
                } else {
                    0
                };
            let captured = snapshot_copies(&host.device.journal.borrow()[before..], 4 * payload);
            for (spare, state) in spares.iter().zip(&states) {
                let buffers: Vec<_> = spare.buffers.iter().flatten().copied().collect();
                let payload = LAYERS * (CODES + SCALES)
                    + 3 * 32768
                    + if speculative {
                        feature_row().0 * feature_row().1
                    } else {
                        0
                    };
                assert_eq!(
                    captured
                        .iter()
                        .filter(|(source, _)| buffers.iter().any(|&b| inside(*source, b)))
                        .map(|(source, _)| source.2)
                        .sum::<u64>(),
                    payload
                );
                for name in ["window.ring", "compressor.tail", "dspark.features"] {
                    if let Some(buffer) = state.named(name) {
                        assert!(!captured.iter().any(|(source, _)| inside(*source, buffer)));
                    }
                }
            }
            instance.retire_request(1).unwrap();
            instance
                .restore_prefix_state_in_slot(2, &(0..768).collect::<Vec<_>>(), &pages, 0)
                .unwrap();
            let mut sink = Sink::default();
            instance
                .execute_prefill_cuts(
                    &cuts(&anchored(2, 0, 768, 21, false), 2, 789, 789),
                    &mut sink,
                )
                .unwrap();
            assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
            instance
                .commit()
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            if speculative {
                let context = crate::instance::dspark_context(instance.handle(), 2).unwrap();
                assert_eq!((context.slot, context.end), (0, 789));
            }
            instance.retire_request(2).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.acquired.get(), host.released.get());
        }
    }
}

/// Replay the journal's ring stores with position-tagged bytes. This checks
/// addressing and copy order without pretending to execute model arithmetic.
fn replay_ring_writes(host: &Host, memory: &mut Memory, lines: &[String]) {
    let span = |(buffer, offset, length): Span| {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1 { raw: buffer }, offset, length)
    };
    let uploaded = |view: Span, offset: u64, width: usize| {
        let uploads = host.uploads.borrow();
        let at = view.1 + offset;
        let (&(buffer, start), bytes) = uploads.range(..=(view.0, at)).next_back().unwrap();
        assert_eq!(buffer, view.0);
        let first = (at - start) as usize;
        bytes[first..first + width].to_vec()
    };
    for line in lines {
        let views = spans(line);
        if is_copy(line) {
            memory.copy(span(views[0]), span(views[1]));
        } else if line.starts_with("launch ") && line.contains(STORE) && views.len() == 9 {
            let positions = views[2];
            let slots = views[3];
            let last = views[4];
            // Spare slot IDs are zero-filled, while working slot IDs are uploaded.
            let spare = !host
                .uploads
                .borrow()
                .keys()
                .any(|&(buffer, _)| buffer == slots.0);
            for row in 0..grid(line) {
                let position =
                    i32::from_le_bytes(uploaded(positions, row * 4, 4).try_into().unwrap());
                let slot = if spare {
                    0
                } else {
                    i32::from_le_bytes(uploaded(slots, row * 4, 4).try_into().unwrap())
                };
                let end = if spare {
                    i64::from(i32::from_le_bytes(uploaded(last, 0, 4).try_into().unwrap()))
                } else {
                    i64::from(i32::from_le_bytes(
                        uploaded(last, slot as u64 * 4, 4).try_into().unwrap(),
                    ))
                };
                if i64::from(position) + 136 <= end || position < 0 || slot < 0 {
                    continue;
                }
                for (plane, bytes_per_row) in [(views[5], 512), (views[6], 16)] {
                    let cell = slot as u64 * 136 + position as u64 % 136;
                    let destination =
                        span((plane.0, plane.1 + cell * bytes_per_row, bytes_per_row));
                    let bytes: Rc<[u8]> = position
                        .to_le_bytes()
                        .repeat(bytes_per_row as usize / 4)
                        .into();
                    memory.write(
                        destination,
                        vec![(bytes_per_row, Origin::Uploaded(bytes, 0))],
                    );
                }
            }
        }
    }
}

/// A restored odd start only 67 rows before its anchor needs committed ring
/// cells. Compare journal-driven ring bytes and actual snapshot copies with
/// two ordinary steps, including another restore from the interior capture.
#[test]
fn odd_restored_continuation_matches_the_split_snapshot_and_slot() {
    let run = |fused: bool| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create(&host);
        let (spares, states) = crate::instance::anchor_spans(instance.handle());
        for state in &states {
            for name in ["window.ring", "compressor.tail"] {
                let span = state.named(name).unwrap();
                let bytes: Rc<[u8]> = vec![0xa5; span.length as usize].into();
                host.memory
                    .borrow_mut()
                    .write(span, vec![(span.length, Origin::Uploaded(bytes, 0))]);
            }
        }
        let execute =
            |instance: &mut BundleInstanceV1, request, slot, start, rows, terminal, anchor| {
                let mut memory = std::mem::take(&mut *host.memory.borrow_mut());
                let first = host.device.journal.borrow().len();
                instance
                    .execute_prefill_cuts(
                        &cuts(
                            &chunk(request, slot, start, rows, terminal, anchor),
                            request,
                            if terminal { start + rows } else { 900 },
                            start + rows,
                        ),
                        &mut Sink::default(),
                    )
                    .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
                replay_ring_writes(&host, &mut memory, &host.device.journal.borrow()[first..]);
                *host.memory.borrow_mut() = memory;
                instance
                    .commit()
                    .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            };
        let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
        let capture = |instance: &mut BundleInstanceV1, request, end| {
            let first = host.device.journal.borrow().len();
            instance.capture_prefix_state(request, end, &pages).unwrap();
            let captured = copies(&host.device.journal.borrow()[first..]);
            let buffers: Vec<_> = states
                .iter()
                .flat_map(|state| state.buffers.iter().map(|&(_, span)| span))
                .chain(
                    spares
                        .iter()
                        .flat_map(|spare| spare.buffers.iter().flatten().copied()),
                )
                .collect();
            captured
                .into_iter()
                .filter(|(source, _)| buffers.iter().any(|&b| inside(*source, b)))
                .flat_map(|(_, (buffer, offset, length))| {
                    host.memory.borrow().bytes(DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1 { raw: buffer },
                        offset,
                        length,
                    ))
                })
                .collect::<Vec<_>>()
        };
        execute(&mut instance, 1, 0, 0, 829, true, false);
        capture(&mut instance, 1, 829);
        instance.retire_request(1).unwrap();
        instance
            .restore_prefix_state_in_slot(2, &(0..829).collect::<Vec<_>>(), &pages, 1)
            .unwrap();
        if fused {
            execute(&mut instance, 2, 1, 829, 71, true, true);
        } else {
            execute(&mut instance, 2, 1, 829, 67, false, false);
        }
        let anchor = capture(&mut instance, 2, 896);
        if !fused {
            execute(&mut instance, 2, 1, 896, 4, true, false);
        }
        let end = capture(&mut instance, 2, 900);
        assert_eq!(
            crate::instance::ledger_state(instance.handle(), 1),
            (Some((2, 900)), false)
        );
        // Restore the interior snapshot and reproduce the same final payload.
        if fused {
            assert_eq!(capture(&mut instance, 2, 896), anchor);
            instance.retire_request(2).unwrap();
            instance
                .restore_prefix_state_in_slot(3, &(0..896).collect::<Vec<_>>(), &pages, 2)
                .unwrap();
            execute(&mut instance, 3, 2, 896, 4, true, false);
            assert_eq!(capture(&mut instance, 3, 900), end);
            instance.retire_request(3).unwrap();
        } else {
            instance.retire_request(2).unwrap();
        }
        assert!(!anchor.is_empty());
        assert_ne!(anchor, end);
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        (anchor, end)
    };
    assert_eq!(run(true), run(false));
}

/// The anchor tests with the `prefix_snapshot_copy_kernel` candidate on:
/// each snapshot of a slot or an anchor item's spares moves in one launch a
/// rank (`super::super::snapshot_kernel`).
mod snapshot_kernel {
    use super::super::snapshot_kernel::with_kernel;

    #[test]
    fn an_anchored_prompt_keeps_its_rounded_end_in_one_step() {
        with_kernel(super::an_anchored_prompt_keeps_its_rounded_end_in_one_step);
    }

    #[test]
    fn an_anchor_near_its_item_start_seeds_the_rings_and_restores() {
        with_kernel(super::an_anchor_near_its_item_start_seeds_the_rings_and_restores);
    }

    #[test]
    fn tp4_anchored_prompt_captures_restores_and_continues() {
        with_kernel(super::tp4_anchored_prompt_captures_restores_and_continues);
    }

    #[test]
    fn anchor_seeds_precede_the_tails_own_seed_copies() {
        with_kernel(super::anchor_seeds_precede_the_tails_own_seed_copies);
    }

    #[test]
    fn dspark_anchored_prompt_keeps_its_raw_history_at_the_anchor() {
        with_kernel(super::dspark_anchored_prompt_keeps_its_raw_history_at_the_anchor);
    }

    #[test]
    fn dspark_anchor_near_its_item_start_keeps_every_earlier_row() {
        with_kernel(super::dspark_anchor_near_its_item_start_keeps_every_earlier_row);
    }

    #[test]
    fn anchored_items_of_one_step_each_keep_their_own_spare() {
        with_kernel(super::anchored_items_of_one_step_each_keep_their_own_spare);
    }

    #[test]
    fn a_cancelled_anchored_step_keeps_no_anchor() {
        with_kernel(super::a_cancelled_anchored_step_keeps_no_anchor);
    }

    #[cfg(feature = "diagnostic-taps")]
    #[test]
    fn anchor_state_dumps_read_their_spares() {
        with_kernel(super::anchor_state_dumps_read_their_spares);
    }

    #[test]
    fn dma_anchors_capture_restore_and_continue_with_both_completion_modes() {
        with_kernel(super::dma_anchors_capture_restore_and_continue_with_both_completion_modes);
    }

    #[test]
    fn odd_restored_continuation_matches_the_split_snapshot_and_slot() {
        with_kernel(super::odd_restored_continuation_matches_the_split_snapshot_and_slot);
    }
}
