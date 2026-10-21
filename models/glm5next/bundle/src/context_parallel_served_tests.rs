//! Full FP8 steps at context-parallel sizes 2 and 4 on the fake host, with
//! the projection stores bound from the precision component (as
//! `fake_host_tests::projection_stages_launch_through_the_stores` binds
//! them at size 1): an eager prefill whose indexer runs in pieces, a decode
//! step and a verify step of the target, and the MTP drafter's extend pass
//! and draft steps. The host sends striped page tables (one local page per
//! stripe of `size` pages, `runtime_kv::ContextParallelPagesV1`).
//!
//! The group's stages compose with the rest of the lowered program: every
//! launch outside the latent KV write, the index key write, the selection
//! and the latent attention (the projections with their row-split partial
//! all-reduces, the KDA, the MoE, the head at its bucket rows, the drafter's
//! rows) is the size-1 instance's launch of the same steps, symbol, grid
//! and scalars, on every rank in order. Each size's launch journal is
//! pinned, and create allocates exactly the group state the capacity plan
//! counts beyond the size-1 instance.

use std::rc::Rc;

use bundle_runtime::fake_host::check_journal;
use bundle_runtime::kv::KvFormat;
use model_glm5next_contract::contract::{DECODE_BATCH, Shape, has_mtp_layer};
use runtime_bundle_abi::{
    BundleKeptRowsV1, BundleStatusV1, HostServiceResultV1, HostServicesTableV1, LogitsRowSinkV1,
    OwnedDraftTokensV1, OwnedLogitsRowV1, OwnedLogitsViewV1, ResultSinkLimitsV1,
};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

use crate::fake_host_tests::{FakeHost, NoRows, api, issued};
use crate::sm120::engines::Engines;

/// A create record at context-parallel size `size` over `format`, naming
/// the precision component.
pub(crate) fn record(size: u32, format: KvFormat) -> Vec<u8> {
    let mut tail = Vec::new();
    if size > 1 {
        tail.extend(runtime_bundle_sdk::context_parallel::TAG);
        tail.extend_from_slice(&size.to_le_bytes());
    }
    tail.extend(crate::config::tests::options(2048));
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    crate::config::tests::record_naming(DECODE_BATCH, &tail, format, &roles)
}

/// The striped page table of a sequence of `end` tokens in slot `slot`: one
/// local page per stripe of `size` pages of 64 tokens.
fn pages(slot: u64, end: u64, size: u32) -> Vec<u64> {
    (0..end.div_ceil(64 * u64::from(size)))
        .map(|stripe| slot * 64 + stripe)
        .collect()
}

/// One step of one item in slot 0: `inputs` after `previous` tokens, a
/// prompt chunk (logits at its end when `last`), a decode row, a verify
/// item, or a verify item the MTP drafter drafts from its bonus token.
fn item_step(item: BatchItemV1, previous: u64, count: u64, size: u32) -> Vec<u8> {
    let end = previous + count;
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![item],
        vec![BatchAppendV1::new(
            previous,
            count,
            end,
            pages(0, end, size),
            None,
        )],
    )
    .with_state_slots(vec![0])
    .encode()
    .unwrap()
}

fn prompt(previous: u64, count: u64, last: bool, size: u32) -> Vec<u8> {
    let inputs = (previous..previous + count).map(|t| t as u32).collect();
    item_step(
        BatchItemV1::new(7, false, count as u32, last, inputs),
        previous,
        count,
        size,
    )
}

fn decode(previous: u64, size: u32) -> Vec<u8> {
    item_step(
        BatchItemV1::new(7, true, 1, true, vec![11]),
        previous,
        1,
        size,
    )
}

fn verify(previous: u64, inputs: &[u32], size: u32) -> Vec<u8> {
    item_step(
        BatchItemV1::verify(7, inputs.len() as u32, false, inputs.to_vec()),
        previous,
        inputs.len() as u64,
        size,
    )
}

fn drafted(previous: u64, rows: u32, bonus: u32, size: u32) -> Vec<u8> {
    item_step(
        BatchItemV1::verify(7, rows, true, vec![bonus]),
        previous,
        u64::from(rows),
        size,
    )
}

fn kept(kept_rows: u32) -> BundleKeptRowsV1 {
    BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id: 7,
        kept_rows,
        reserved_tail: 0,
    }
}

/// A sink of a step's drafts and logits views.
#[derive(Default)]
struct Drafts(Vec<Vec<u32>>);

impl LogitsRowSinkV1 for Drafts {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.0.push(drafts.tokens);
        Ok(())
    }

    fn emit_view(&mut self, _view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        Ok(())
    }
}

/// The bytes the fake device allocated.
fn allocated(host: &FakeHost) -> u64 {
    host.device
        .journal
        .borrow()
        .iter()
        .filter_map(|line| line.strip_prefix("allocate "))
        .map(|rest| rest.split(' ').next().unwrap().parse::<u64>().unwrap())
        .sum()
}

/// The steps one instance runs, with the launch lines they issued: the
/// target's prefill of 3000 rows (indexer pieces of 2048 and 952 rows), a
/// decode step and a verify step of four rows keeping two, or, on a
/// drafting instance, a prompt of ten rows and a drafted verify item of
/// four rows keeping two (the extend pass and the draft steps).
struct Run {
    launches: Vec<String>,
    allocated: u64,
}

fn run(size: u32, drafting: bool) -> Run {
    // The FP8 formats without and with the MTP drafter's layer.
    let format = Shape::TP4.kv_formats()[usize::from(drafting)];
    assert_eq!(has_mtp_layer(&format), drafting);
    let host = FakeHost::with_format((3, 32), format);
    // Size 1 keeps every prefill whole, as a group does, so that the
    // groups' extra allocations are their state alone.
    let mut instance = {
        api().create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &record(size, format),
        )
    }
    .unwrap_or_else(|error| panic!("size {size}: {error:?} {:?}", host.logs.borrow()));
    // The journals hold the prompt's eager launches.
    crate::instance::keep_last_chunks_eager(instance.handle());
    let allocated = allocated(&host);
    let start = host.device.journal.borrow().len();
    served_steps(&host, &mut instance, size, drafting);
    let launches = issued(&host, start)
        .into_iter()
        .filter(|line| line.starts_with("launch "))
        .collect();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get(), "size {size}");
    Run {
        launches,
        allocated,
    }
}

/// The steps of [`run`] at `size`: with the drafter, a prompt and a
/// drafted verify item; else a 3000-token prompt, a decode and a verify
/// step.
pub(crate) fn served_steps(
    host: &FakeHost,
    instance: &mut runtime_bundle_abi::BundleInstanceV1,
    size: u32,
    drafting: bool,
) {
    let fail = |error| panic!("size {size}: {error:?} {:?}", host.logs.borrow());
    if drafting {
        let mut sink = Drafts::default();
        instance
            .execute(&prompt(0, 10, true, size), &mut sink)
            .unwrap_or_else(fail);
        instance.commit().unwrap();
        let mut sink = Drafts::default();
        instance
            .execute(&drafted(10, 4, 21, size), &mut sink)
            .unwrap_or_else(fail);
        assert_eq!(sink.0.len(), 1, "size {size}: one item's drafts");
        instance.commit_accepted(&[kept(2)]).unwrap();
    } else {
        instance
            .execute(&prompt(0, 3000, true, size), &mut NoRows::default())
            .unwrap_or_else(fail);
        instance.commit().unwrap();
        instance
            .execute(&decode(3000, size), &mut NoRows::default())
            .unwrap_or_else(fail);
        instance.commit().unwrap();
        instance
            .execute(
                &verify(3001, &[21, 22, 23, 24], size),
                &mut NoRows::default(),
            )
            .unwrap_or_else(fail);
        instance.commit_accepted(&[kept(2)]).unwrap();
    }
}

/// `text` with every `marker` and the bracketed group after it (opened by
/// `open`, nested) replaced by `marker` alone.
fn collapse(text: &str, marker: &str, open: char, close: char) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(&format!("{marker}{open}")) {
        out.push_str(&rest[..start]);
        let mut depth = 0;
        let mut end = rest.len();
        for (offset, c) in rest[start..].char_indices() {
            if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
                if depth == 0 {
                    end = start + offset + 1;
                    break;
                }
            }
        }
        out.push_str(marker);
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}

/// A launch line's symbol, configuration and scalars, without its stream
/// and device spans.
fn shape_of(line: &str) -> (String, String) {
    let mut words = line.splitn(4, ' ');
    let (_, _, symbol) = (words.next(), words.next(), words.next().unwrap_or(""));
    let rest = words.next().unwrap_or("");
    // Spans name allocation handles, which the group state shifts.
    let rest = collapse(rest, "DeviceBufferSpanV1 ", '{', '}');
    let rest = collapse(
        &collapse(&rest, "DevicePointerArray", '(', ')'),
        "DeviceBuffer",
        '(',
        ')',
    );
    (symbol.to_owned(), rest)
}

/// Asserts that `group` and `single` (per stream, [`outside_the_group`])
/// hold the same launches, naming the first that differs.
pub(crate) fn assert_composes(
    size: u32,
    group: &[Vec<(String, String)>],
    single: &[Vec<(String, String)>],
) {
    assert_eq!(group.len(), single.len(), "size {size}: streams");
    for (stream, (group, single)) in group.iter().zip(single).enumerate() {
        if let Some(index) =
            (0..group.len().max(single.len())).find(|&i| group.get(i) != single.get(i))
        {
            panic!(
                "size {size} stream {stream} launch {index} of {} vs {}: {:?} vs {:?}",
                group.len(),
                single.len(),
                group.get(index),
                single.get(index)
            );
        }
    }
}

/// `launches` with each all-reduce (and the round after the pair) folded
/// into one item of its output kind and element count, so instances compose
/// by the reductions they run, not by their entries; the journal pins and
/// their launch-equivalence proof hold the entries themselves.
fn fold_reductions(launches: Vec<(String, String)>) -> Vec<(String, String)> {
    use model_glm5next_contract::sm120::engines as selected;
    let round = selected::ModelRow::Round.symbol();
    let elements = |rest: &str| {
        rest.split("ScalarU64(")
            .nth(1)
            .and_then(|tail| tail.split(')').next())
            .unwrap_or("")
            .to_owned()
    };
    let mut folded = Vec::new();
    let mut at = 0;
    while at < launches.len() {
        let (symbol, rest) = &launches[at];
        let symbol = symbol.as_str();
        let (calls, rounded) =
            if selected::PHASED_SYMBOLS.contains(&symbol) || symbol == selected::LL_SYMBOL {
                (1, true)
            } else if symbol == selected::SCATTER_SYMBOL {
                let rounded = launches.get(at + 2).is_some_and(|next| next.0 == round);
                (2 + usize::from(rounded), rounded)
            } else {
                folded.push(launches[at].clone());
                at += 1;
                continue;
            };
        let kind = if rounded { "rounded" } else { "f32" };
        folded.push((format!("all-reduce {kind}"), elements(rest)));
        at += calls;
    }
    folded
}

/// Each stream's launches outside the group's stages, in order, the streams
/// in order of their first launch, each all-reduce folded
/// ([`fold_reductions`]).
pub(crate) fn outside_the_group(launches: &[String]) -> Vec<Vec<(String, String)>> {
    let group = Engines::group_stage_symbols();
    let mut streams: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for line in launches {
        let stream = line.split(' ').nth(1).unwrap_or("").to_owned();
        let launch = shape_of(line);
        if group.contains(&launch.0) {
            continue;
        }
        match streams.iter_mut().find(|(slot, _)| *slot == stream) {
            Some((_, launches)) => launches.push(launch),
            None => streams.push((stream, vec![launch])),
        }
    }
    streams
        .into_iter()
        .map(|(_, launches)| fold_reductions(launches))
        .collect()
}

/// At sizes 2 and 4 with the projection stores bound, the target's
/// prefill (in indexer pieces), decode and verify steps run every stage;
/// outside the group's own stages each rank's launches are the size-1
/// instance's; create allocates exactly the planned group state beyond it;
/// and each size's launches are pinned. Re-pinned when commit launched one
/// state copy per buffer (12 launches) and its symbol joined create (the
/// handles), when the phased all-reduce joined create (the handles) and
/// reduced 2 to 32 decode and verify rows, and when the shared gate/up
/// overlap became the default (the shared gate/ups on the side streams;
/// `fake_host_tests::shared_overlap` proves the groups' launches the
/// one-stream program's).
/// Re-pinned when the strip-chain and prefill warp-tile entries took the
/// kernels that fence their ring reads before release (their symbols).
/// Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse and
/// its two symbols joined create (the handles).
/// Re-pinned when the decoded-tile projections took the balanced decoded
/// tiles (their symbols, grids and activation boxes).
/// Re-pinned when the projections' staged-tile cubin joined create (the
/// handles) and the BF16 prefill projections it holds took its entries.
/// Re-pinned when the stream expansion and the score compaction launched the
/// shared row copies (identity change, accuracy unaffected): their symbols and
/// shape arguments, and in create the row cubin's two symbols fewer and
/// `row_copy_sm120`'s two per rank (the handles).
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn served_groups_run_full_target_steps() {
    let single = run(1, false);
    let pinned = [
        (2, (16548, 0x7ffd74fedf71998f)),
        (4, (17076, 0x455c71639e015357)),
    ];
    let mut failures = Vec::new();
    for (size, expected) in pinned {
        let group = run(size, false);
        let planned = model_glm5next_contract::capacity::workspace(Shape::TP4.geometry, 4, size)
            .context_parallel;
        assert_eq!(
            group.allocated - single.allocated,
            4 * planned,
            "size {size}"
        );
        assert_composes(
            size,
            &outside_the_group(&group.launches),
            &outside_the_group(&single.launches),
        );
        // The group's own launches ran: every sparse layer's phased
        // gathers and exchange, and its barrier.
        for symbol in [
            model_glm5next_contract::sm120::context_parallel::PHASED_SYMBOL,
            model_glm5next_contract::sm120::context_parallel::ALLGATHER_SYMBOL,
        ] {
            assert!(
                group.launches.iter().any(|line| line.contains(symbol)),
                "size {size} {symbol}"
            );
        }
        let file = format!("glm5next-served-cp{size}-target.txt");
        if let Err(error) = check_journal(&group.launches, expected, &file) {
            failures.push(format!("size {size}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// [`served_groups_run_full_target_steps`] for the MTP drafter on a
/// drafting instance: its extend pass after the prompt and the drafted
/// verify item's extend pass and draft steps (each draft step localizing
/// the extend pass's shared key lists). Re-pinned when the drafter's head,
/// its draft rows' gather and argmax took the run's item bucket
/// (`program::draft_rows`: the head's entry, rows and tensor maps, the row
/// grids and the candidates' all-reduce elements), and when commit launched
/// one state copy per buffer and its symbol joined create (the handles),
/// when the phased all-reduce joined create (the handles) and reduced 2 to
/// 32 decode and verify rows, and when the shared gate/up overlap became the
/// default (as above).
/// Re-pinned when the strip-chain and prefill warp-tile entries took the
/// kernels that fence their ring reads before release (their symbols).
/// Re-pinned when prefill mixers from 144 rows ran the tiled mix-collapse and
/// its two symbols joined create (the handles).
/// Re-pinned when the decoded-tile projections took the balanced decoded
/// tiles (their symbols, grids and activation boxes).
/// Re-pinned when the projections' staged-tile cubin joined create (the
/// handles) and the BF16 prefill projections it holds took its entries.
/// Re-pinned when the stream expansion and the score compaction launched the
/// shared row copies, as for the target steps, and when the drafter's argmax
/// moved onto the shared sharded argmax (its partial, the pairs' two gathers
/// and its final per rank, and their bindings).
/// Re-pinned when GLM's own draft argmax kernels left its row cubin (two
/// fewer resolves per rank at create; the ids issued after them renumber).
#[test]
fn served_groups_run_the_drafters_passes() {
    let single = run(1, true);
    let pinned = [
        (2, (11712, 0x7b877079e9c893a9)),
        (4, (12000, 0x4fea46b842694b1b)),
    ];
    let mut failures = Vec::new();
    for (size, expected) in pinned {
        let group = run(size, true);
        let planned = model_glm5next_contract::capacity::workspace(Shape::TP4.geometry, 4, size)
            .context_parallel;
        assert_eq!(
            group.allocated - single.allocated,
            4 * planned,
            "size {size}"
        );
        assert_composes(
            size,
            &outside_the_group(&group.launches),
            &outside_the_group(&single.launches),
        );
        let file = format!("glm5next-served-cp{size}-drafter.txt");
        if let Err(error) = check_journal(&group.launches, expected, &file) {
            failures.push(format!("size {size}: {error}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// At sizes 2 and 4 an eager prefill whose exact shape ran before replays
/// captured programs with their group launches, as at size 1
/// (`fake_host_tests::repeated_prefill_shapes_replay_their_captured_programs`):
/// a prompt's 64-row chunks, then its chunks of 2,112 rows (indexer pieces
/// of 2,048 and 64 rows, each with its gathers, exchange and barrier), run
/// the first eagerly, capture every rank's program at the second, which
/// then replays it, and from the third on lower nothing and launch only
/// the replays. The capture holds exactly the launches the eager run
/// issued on every rank, so every group state sees the same calls in the
/// same order; every rank submits the same collectives. Destroy releases
/// every graph, newest first as the fake host enforces.
#[test]
fn served_groups_replay_repeated_prefill_shapes() {
    use crate::fake_host_tests::{count, launches_by_stream, matched};
    use crate::instance::lowered_programs;
    let format = Shape::TP4.kv_formats()[0];
    for size in [2, 4] {
        let host = FakeHost::with_format((3, 32), format);
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &record(size, format),
            )
            .unwrap_or_else(|error| panic!("size {size}: {error:?} {:?}", host.logs.borrow()));
        let mut previous = 0;
        let mut chunk = |instance: &mut runtime_bundle_abi::BundleInstanceV1, rows: u64| {
            let start = host.device.journal.borrow().len();
            let before = lowered_programs();
            instance
                .execute(&prompt(previous, rows, false, size), &mut NoRows::default())
                .unwrap_or_else(|error| panic!("size {size}: {error:?} {:?}", host.logs.borrow()));
            instance.commit().unwrap();
            assert!(matched(&host), "size {size}");
            previous += rows;
            (
                (
                    lowered_programs() - before,
                    count(&host, start, "captured "),
                    count(&host, start, "replay "),
                ),
                launches_by_stream(&host, start),
            )
        };
        for rows in [64, 2112] {
            let label = format!("size {size} {rows} rows");
            let (first, eager) = chunk(&mut instance, rows);
            assert_eq!(first, (4, 0, 0), "{label}: the first run is eager");
            assert!(eager.values().all(|launches| {
                launches.iter().any(|line| {
                    line.contains(model_glm5next_contract::sm120::context_parallel::PHASED_SYMBOL)
                })
            }));
            let (second, capture) = chunk(&mut instance, rows);
            assert_eq!(
                second,
                (4, 12, 12),
                "{label}: the second captures and replays"
            );
            assert_eq!(
                capture, eager,
                "{label}: the capture holds the eager launches"
            );
            for _ in 0..2 {
                let (later, launched) = chunk(&mut instance, rows);
                assert_eq!(later, (0, 0, 12), "{label}: later runs only replay");
                assert!(launched.is_empty(), "{label}");
            }
        }
        instance.destroy().unwrap();
        assert_eq!(
            count(&host, 0, "release graph "),
            count(&host, 0, "captured "),
            "size {size}: every captured piece is released by destroy"
        );
        assert_eq!(host.live(), Vec::<String>::new(), "size {size}");
    }
}
