//! Prompt-end anchors kept inside an upper-skip item
//! (`instance::execute::anchors`): the spares receive the bytes a chunk
//! ending at the anchor leaves in its slot, and a capture there copies them.
use super::prefill_skip::{create_profile_with_options, try_create_profile_with_options};
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

fn create(host: &Rc<Host>, anchors: u32) -> BundleInstanceV1 {
    create_profile_with_options(
        host,
        8192,
        true,
        &[("prefill_anchor_items", anchors.into())],
    )
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
        let base = spare.offset + layer * (CODES + SCALES);
        assert_eq!(spans[5], (spare.buffer.raw, base, CODES));
        assert_eq!(spans[6], (spare.buffer.raw, base + CODES, SCALES));
        layer += 1;
    }
    assert_eq!(layer, LAYERS);
}

fn spans_of_previous(stores: &[&String], rings: DeviceBufferSpanV1) -> Option<Vec<Span>> {
    let line = stores.last()?;
    let spans = spans(line);
    (spans.len() == 9 && inside(spans[5], rings)).then_some(spans)
}

/// The copies of `lines`: (source, destination).
fn copies(lines: &[String]) -> Vec<(Span, Span)> {
    lines
        .iter()
        .filter(|line| line.starts_with("copy "))
        .map(|line| {
            let spans = spans(line);
            assert_eq!(spans.len(), 2, "{line}");
            (spans[0], spans[1])
        })
        .collect()
}

/// The instance declares its spares as its grid's anchor items: none by
/// default or without a grid, one each with the option.
#[test]
fn instances_declare_their_spares_as_grid_anchor_items() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    for (chunk, skip, anchors, declared) in [
        (8192, true, 0, 0),
        (8192, true, 1, 1),
        (8192, true, 8, 8),
        (0, false, 0, 0),
    ] {
        let mut instance = create_profile_with_options(
            &host,
            chunk,
            skip,
            &[("prefill_anchor_items", anchors.into())],
        );
        assert_eq!(instance.prefill_anchor_items(), Ok(Some(declared)));
        instance.destroy().unwrap();
    }
    assert_eq!(host.acquired.get(), host.released.get());
}

/// An unaligned prompt's single anchored step writes every layer's ring of
/// its rounded end into the spare, from the rows a cut there would end
/// with, and seeds only the tails: its ring is inside the item. A capture
/// at the anchor copies the spares alone; the prompt end, the slot.
#[test]
fn an_anchored_prompt_keeps_its_rounded_end_in_one_step() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host, 1);
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
    instance.commit().unwrap();
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 0),
        (Some((1, 789)), false)
    );
    let step: Vec<String> = host.device.journal.borrow()[first..].to_vec();
    let launch = step.iter().position(|l| l.starts_with("launch ")).unwrap();
    // Every tail layer's slot entry seeds the spare before any launch.
    let seeds = copies(&step[..launch]);
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
                spare_tails.offset + layer * 32768,
                32768
            )
        );
    }
    check_anchor_stores(&step, rings, spare_rings, spares[0].zeros, 632..768);
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
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host, 1);
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
    instance.commit().unwrap();
    let first = host.device.journal.borrow().len();
    let request = cuts(&anchored(1, 0, 8192, 65, true), 1, 8257, 8257);
    instance
        .execute_prefill_cuts(&request, &mut Sink::default())
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    let step: Vec<String> = host.device.journal.borrow()[first..].to_vec();
    let launch = step.iter().position(|l| l.starts_with("launch ")).unwrap();
    let seeds = copies(&step[..launch]);
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
    check_anchor_stores(&step, rings, spare_rings, spares[0].zeros, 0..64);
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
    instance.commit().unwrap();
    assert_eq!(
        crate::instance::ledger_state(instance.handle(), 1),
        (Some((2, 8257)), false)
    );
    instance.retire_request(2).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}

/// Without spares an anchor flag is ignored and a capture inside the item
/// is refused, so the host drops that anchor. With spares, an anchor item
/// must hold a granule multiple strictly inside it and end off the granule,
/// and DSpark or a full schedule refuses spares at create.
#[test]
fn anchor_flags_need_spares_an_interior_granule_and_an_ar_upper_skip() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    let mut plain = create(&host, 0);
    plain
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789),
            &mut Sink::default(),
        )
        .unwrap();
    plain.commit().unwrap();
    let pages: Vec<_> = (130..130 + plain.describe_prefix_state().unwrap()).collect();
    assert_eq!(
        plain.capture_prefix_state(1, 768, &pages),
        Err(runtime_bundle_abi::BundleAbiError::Status(
            BundleStatusV1::InvalidState
        ))
    );
    plain.capture_prefix_state(1, 789, &pages).unwrap();
    plain.retire_request(1).unwrap();
    plain.destroy().unwrap();
    let mut instance = create(&host, 1);
    for rows in [768, 50] {
        assert_eq!(
            instance.execute_prefill_cuts(
                &cuts(&anchored(1, 0, 0, rows, true), 1, rows, rows),
                &mut Sink::default()
            ),
            Err(runtime_bundle_abi::BundleAbiError::Status(
                BundleStatusV1::InvalidArgument
            )),
            "{rows}"
        );
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let dspark = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    dspark.drafter.set(true);
    dspark.speculation.set(Some((5, 32)));
    for (host, chunk, skip) in [(&dspark, 8192, true), (&host, 0, false)] {
        assert_eq!(
            try_create_profile_with_options(
                host,
                chunk,
                skip,
                &[("prefill_anchor_items", 1.into())]
            )
            .err(),
            Some(runtime_bundle_abi::BundleAbiError::Status(
                BundleStatusV1::Unsupported
            ))
        );
    }
}

/// Spares change nothing a step launches unless an item keeps an anchor:
/// the same prompt issues the same kernels, grids and copies either way.
#[test]
fn spares_leave_unanchored_steps_unchanged() {
    let launches = |anchors: u32| {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
        let mut instance = create(&host, anchors);
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
            instance.commit().unwrap();
        }
        let names: Vec<_> = host.device.journal.borrow()[first..]
            .iter()
            .filter(|line| line.starts_with("launch ") || line.starts_with("copy "))
            .map(|line| {
                let mut words = line.split_whitespace();
                let kind = words.next().unwrap().to_owned();
                if kind == "copy" {
                    return kind;
                }
                format!("{} {}", words.nth(1).unwrap(), grid(line))
            })
            .collect();
        instance.retire_request(1).unwrap();
        instance.destroy().unwrap();
        names
    };
    let plain = launches(0);
    assert!(plain.len() > 1000);
    assert_eq!(plain, launches(1));
}

/// At TP4 an anchored prompt captures its rounded end from every rank's
/// spares, restores it and continues to the prompt end.
#[test]
fn tp4_anchored_prompt_captures_restores_and_continues() {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host, 1);
    let mut sink = Sink::default();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 4171, true), 1, 4171, 4171),
            &mut sink,
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    assert_eq!(sink.views.iter().map(|v| v.raw().row_count).sum::<u32>(), 1);
    instance.commit().unwrap();
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
    instance.commit().unwrap();
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
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host, 1);
    let (spares, states) = crate::instance::anchor_spans(instance.handle());
    let tails = states[0].named("compressor.tail").unwrap();
    let spare_tails = spares[0].buffers[1].unwrap();
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 789, false), 1, 789, 789),
            &mut Sink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
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
    instance.commit().unwrap();
    let step: Vec<String> = host.device.journal.borrow()[first..].to_vec();
    let launch = step.iter().position(|l| l.starts_with("launch ")).unwrap();
    let before = copies(&step[..launch]);
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

/// The diagnostic state dump of a capture at an anchor reads the anchor's
/// spares: the snapshot payload at the anchor and the pages up to it.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn anchor_state_dumps_read_their_spares() {
    let root = std::path::PathBuf::from(std::env::var_os("CHOPINFER_DATA").unwrap())
        .join("deepseek41-tests")
        .join(format!("anchor-state-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("enabled"), []).unwrap();
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP1).unwrap(), KV_LAYOUTS[0]);
    let mut instance = create(&host, 1);
    crate::instance::set_dspark_state_capture_root_for_test(instance.handle(), Some(root.clone()));
    instance
        .execute_prefill_cuts(
            &cuts(&anchored(1, 0, 0, 789, true), 1, 789, 789),
            &mut Sink::default(),
        )
        .unwrap();
    instance.commit().unwrap();
    let pages: Vec<_> = (130..130 + instance.describe_prefix_state().unwrap()).collect();
    for tokens in [768, 789] {
        instance
            .capture_prefix_state(1, tokens, &pages)
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    }
    let read = |name: &str| {
        let directory = root.join(name).join("request-1");
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("state.json")).unwrap()).unwrap();
        let pieces = metadata["pieces"].as_array().unwrap();
        let bytes: u64 = pieces
            .iter()
            .map(|piece| piece["bytes"].as_u64().unwrap())
            .sum();
        assert_eq!(
            std::fs::metadata(directory.join("rank-0.bin"))
                .unwrap()
                .len(),
            bytes
        );
        let rings = pieces
            .iter()
            .filter(|piece| piece["buffer"] == "window.ring")
            .count();
        assert_eq!(rings, 80);
        metadata
    };
    let anchor = read("snapshot-768");
    assert_eq!(anchor["end"], 768);
    assert_eq!(anchor["retained_start"], 640);
    assert_eq!(read("snapshot-789")["end"], 789);
    assert!(root.join("chunk-789/request-1/state.json").is_file());
    assert!(!root.join("chunk-768").exists());
    instance.retire_request(1).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    std::fs::remove_dir_all(root).unwrap();
}
