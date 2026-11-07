//! `dspark_successor_prefill`: the successor the decode step before a
//! mixed step queued serves that step. Eight decoders draft at depth three
//! beside a cold prompt, a prefix hit restored from the pages, or a
//! prompt's last chunk; the five-head successor an allocated decode queues
//! is cut to the three heads the unallocated mixed step computes. Without
//! the option such a step discards it, its record noting that the option's
//! rule would have served it, and drafts inline. The reused step's drafts,
//! diagnostics, target journal and committed state are the inline step's.
use super::results::metadata;
use super::*;
use crate::dspark::execution::{Input, Noise};
use crate::dspark::prefetch::Feedback;
use crate::dspark::state::{Kind, Rows};
use model_deepseek41_contract::execution_options::{
    DIAGNOSTIC_STEP_LOGS, DSPARK_SUCCESSOR_PREFILL,
};
use runtime_bundle_abi::device_results::*;

const DECODERS: u64 = 8;
/// The mixed step's prefill request, in slot 9.
const PREFILL: u64 = 60;
const PREFILL_SLOT: u64 = 9;
/// The prompt a capture leaves in the pages, from slot 8.
const CAPTURED: u64 = 50;
const CAPTURED_ROWS: u64 = 141;
/// An idle committed request in slot 11, and a request restored for a later
/// step in slot 12.
const IDLE: u64 = 70;
const LATER: u64 = 71;

/// The mixed step's prefill item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Prefill {
    /// No prefill: the next step decodes only.
    None,
    /// A fresh 40-row prompt.
    Cold,
    /// The captured prompt restored for the request, then its last 20 rows.
    Restored,
    /// The last 40 rows of a prompt whose first 64 committed earlier.
    Chunk,
}

/// What happens between the decode step's commit and the mixed step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Between {
    Nothing,
    /// A decoder's prefix capture, which catches it up.
    Capture,
    /// Decoder 8 finishes and retires.
    Retire,
    /// Decoder 8 is preempted: captured, then retired.
    Preempt,
    /// The idle request is evicted.
    Evict,
    /// Another request is restored for a later step.
    RestoreOther,
    /// A committed ring write that changes no drafter state.
    RingWrite,
}

/// One mixed step and what it left.
struct Run {
    step: Vec<String>,
    drafts: Vec<(u64, Vec<u32>)>,
    rows: u32,
    /// The calibration records, without the instance ID.
    records: Vec<serde_json::Value>,
    /// The successor records of the mixed step.
    successor: Vec<serde_json::Value>,
    /// Each request's committed context after the mixed step's commit.
    contexts: Vec<Option<crate::dspark::state::Context>>,
    /// Each rank's draft IDs and confidence, and the reported heads.
    outputs: (Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)>, u32),
    streams: Vec<u32>,
}

/// The device result of an allocated decode step of every decoder at `k`
/// drafts from `start`, each accepting `accepted` drafts with bonus 7.
fn allocated_feedback(start: u64, k: u32, accepted: u32) -> Feedback {
    let items = DECODERS as usize;
    let mut packet =
        vec![0; crate::dspark::results::SAMPLE_OFFSET as usize + items * (k as usize + 1) * 256];
    for index in 0..items {
        let result = BundleAcceptanceResultV1 {
            cycle_epoch: 1,
            accepted_drafts: accepted,
            emitted_rows: accepted + 1,
            flags: BUNDLE_ACCEPTANCE_ALLOCATED_V1
                | if accepted < k {
                    BUNDLE_ACCEPTANCE_REJECTED_V1
                } else {
                    0
                },
            status: 0,
            failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
            verified_drafts: k,
        };
        packet[index * 32..index * 32 + 32].copy_from_slice(&result.to_le_bytes());
        let at = crate::dspark::results::SAMPLE_OFFSET as usize
            + (index * (k as usize + 1) + accepted as usize) * 256;
        packet[at..at + 4].copy_from_slice(&7_u32.to_le_bytes());
    }
    Feedback {
        allocated: true,
        metadata: metadata(DECODERS as u32, k, 1),
        packet,
        inputs: (0..DECODERS)
            .map(|slot| Input {
                rows: Rows {
                    request: slot + 1,
                    slot: slot as u32,
                    start,
                    kind: Kind::Decode { drafts: k },
                },
                tokens: vec![7],
                noise: Noise::GREEDY,
            })
            .collect(),
    }
}

fn pages(slot: u64, end: u64) -> Vec<u64> {
    (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect()
}

/// A step of the decoders `verifiers` verifying `k` drafts from `start`
/// beside `prefill`: (request, slot, committed tokens, rows).
fn mixed(prefill: Option<(u64, u64, u64, u64)>, verifiers: u64, start: u64, k: u32) -> Vec<u8> {
    let mut items = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    if let Some((request, slot, prefix, rows)) = prefill {
        let end = prefix + rows;
        items.push(BatchItemV1::new(
            request,
            false,
            rows as u32,
            true,
            (prefix..end).map(|at| at as u32).collect(),
        ));
        appends.push(BatchAppendV1::new(
            prefix,
            rows,
            end,
            pages(slot, end),
            None,
        ));
        slots.push(slot);
    }
    for slot in 0..verifiers {
        let rows = u64::from(k + 1);
        items.push(BatchItemV1::verify(slot + 1, k + 1, true, vec![7]));
        appends.push(BatchAppendV1::new(
            start,
            rows,
            start + rows,
            pages(slot, start + rows),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// The serving pages: eight a slot, then the snapshots'.
const KV_PAGES: u64 = 1024;
const SNAPSHOT_PAGE: u64 = 32 * 8;

/// The snapshot pages of a capture, clear of every slot's.
fn snapshot_pages(instance: &mut BundleInstanceV1) -> Vec<u64> {
    let count = instance.describe_prefix_state().unwrap();
    assert!(SNAPSHOT_PAGE + count <= KV_PAGES);
    (0..count).map(|page| SNAPSHOT_PAGE + page).collect()
}

/// An instance of 16 items and 64 verify rows with `options`, its pool
/// holding [`KV_PAGES`] pages.
fn create(options: &[(&str, serde_json::Value)]) -> (Rc<Host>, BundleInstanceV1) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.drafter.set(true);
    host.speculation.set(Some((5, 64)));
    host.kv_pages.set(KV_PAGES);
    let mut record = crate::config::tests::record(&shape, host.kv, 16, false);
    for (name, value) in options {
        record = crate::config::tests::with_option(record, name, value.clone());
    }
    let instance = api()
        .create(
            HostServicesTableV1::new(Rc::new(MappedHost::new(Rc::clone(&host)))),
            &record,
        )
        .unwrap_or_else(|e| panic!("drafter create: {e:?} {:?}", host.logs.borrow()));
    (host, instance)
}

fn successor_records(host: &Host, from: usize) -> Vec<serde_json::Value> {
    host.logs.borrow()[from..]
        .iter()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|record| record["phase"] == "dspark-prefetch")
        .collect()
}

/// The decoders prompted, a warm depth-three step that captures the
/// compact draft graph, then a decode step at depth three whose five-head
/// successor (an allocated step's) predicts two accepted drafts; then
/// `between`, and the mixed step beside `prefill`, committed with two
/// accepted drafts. `cancel` cancels the mixed step instead.
fn run(option: bool, prefill: Prefill, between: Between, cancel: bool) -> Run {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let (host, mut instance) = create(&[
        (DIAGNOSTIC_STEP_LOGS, true.into()),
        (DSPARK_SUCCESSOR_PREFILL, option.into()),
    ]);
    let prompt = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        (1..=DECODERS)
            .map(|request| BatchItemV1::new(request, false, 3, true, vec![7; 3]))
            .collect(),
        (0..DECODERS)
            .map(|slot| BatchAppendV1::new(0, 3, 3, pages(slot, 3), None))
            .collect(),
    )
    .with_state_slots((0..DECODERS).collect())
    .encode()
    .unwrap();
    execute(&mut instance, &host, &prompt);
    instance.commit().unwrap();
    let ids: Vec<u32> = (0..CAPTURED_ROWS as u32).collect();
    let snapshot = snapshot_pages(&mut instance);
    execute(
        &mut instance,
        &host,
        &super::super::prompt(CAPTURED, 8, 0, CAPTURED_ROWS, true),
    );
    instance.commit().unwrap();
    instance
        .capture_prefix_state(CAPTURED, CAPTURED_ROWS, &snapshot)
        .unwrap();
    instance.retire_request(CAPTURED).unwrap();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(IDLE, 11, 0, 3, true),
    );
    instance.commit().unwrap();
    if prefill == Prefill::Chunk {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(PREFILL, PREFILL_SLOT, 0, 64, false),
        );
        instance.commit().unwrap();
    }
    let decoders: Vec<_> = (0..DECODERS).map(|slot| (slot + 1, slot, 3)).collect();
    execute(&mut instance, &host, &verify_all(&decoders, 3));
    instance
        .commit_accepted(&(1..=DECODERS).map(|r| kept(r, 1)).collect::<Vec<_>>())
        .unwrap();
    let decoders: Vec<_> = (0..DECODERS).map(|slot| (slot + 1, slot, 4)).collect();
    execute(&mut instance, &host, &verify_all(&decoders, 3));
    crate::instance::dspark_prefetch_for_test(instance.handle(), allocated_feedback(4, 3, 2), true);
    instance
        .commit_accepted(&(1..=DECODERS).map(|r| kept(r, 3)).collect::<Vec<_>>())
        .unwrap();
    let mut verifiers = DECODERS;
    match between {
        Between::Nothing => {}
        Between::Capture => {
            let pages = snapshot_pages(&mut instance);
            instance.capture_prefix_state(1, 7, &pages).unwrap();
        }
        Between::Retire => {
            instance.retire_request(DECODERS).unwrap();
            verifiers -= 1;
        }
        Between::Preempt => {
            let pages = snapshot_pages(&mut instance);
            instance.capture_prefix_state(DECODERS, 7, &pages).unwrap();
            instance.retire_request(DECODERS).unwrap();
            verifiers -= 1;
        }
        Between::Evict => instance.retire_request(IDLE).unwrap(),
        Between::RestoreOther => instance
            .restore_prefix_state_in_slot(LATER, &ids, &snapshot, 12)
            .unwrap(),
        Between::RingWrite => {
            crate::instance::dspark_ring_writes_for_test(instance.handle(), 1);
        }
    }
    let item = match prefill {
        Prefill::None => None,
        Prefill::Cold => Some((PREFILL, PREFILL_SLOT, 0, 40)),
        Prefill::Restored => {
            instance
                .restore_prefix_state_in_slot(PREFILL, &ids, &snapshot, PREFILL_SLOT)
                .unwrap();
            Some((PREFILL, PREFILL_SLOT, CAPTURED_ROWS, 20))
        }
        Prefill::Chunk => Some((PREFILL, PREFILL_SLOT, 64, 40)),
    };
    let logs = host.logs.borrow().len();
    let begin = host.journal.borrow().len();
    let sink = execute(&mut instance, &host, &mixed(item, verifiers, 7, 3));
    let step = host.journal.borrow()[begin..].to_vec();
    let successor = successor_records(&host, logs);
    let outputs = crate::instance::dspark_draft_outputs_for_test(instance.handle());
    let mut requests: Vec<u64> = (1..=verifiers).collect();
    requests.extend([PREFILL, IDLE, LATER]);
    if cancel {
        instance.cancel().unwrap();
    } else {
        instance
            .commit_accepted(&(1..=verifiers).map(|r| kept(r, 3)).collect::<Vec<_>>())
            .unwrap();
    }
    let contexts = requests
        .iter()
        .map(|&request| crate::instance::dspark_context(instance.handle(), request))
        .collect();
    let records = records(&host)
        .into_iter()
        .map(|mut record| {
            record.as_object_mut().unwrap().remove("instance_id");
            record
        })
        .collect();
    let streams = (0..shape.ranks())
        .map(|rank| host.rank_streams.borrow()[&rank])
        .collect();
    let drafts = sink.draft_tokens();
    let rows = sink.views.iter().map(|view| view.raw().row_count).sum();
    let mut admitted: Vec<u64> = (1..=verifiers).collect();
    if prefill != Prefill::None {
        admitted.push(PREFILL);
    }
    if between != Between::Evict {
        admitted.push(IDLE);
    }
    if between == Between::RestoreOther {
        admitted.push(LATER);
    }
    for request in admitted {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    Run {
        step,
        drafts,
        rows,
        records,
        successor,
        contexts,
        outputs,
        streams,
    }
}

/// The step's lines through its rank-0 draft ID readback, and from it.
fn split(run: &Run, items: u64) -> (&[String], &[String]) {
    let ids = run.outputs.0[0].0;
    let readback = format!(
        "download {} {:?}",
        run.streams[0],
        DeviceBufferSpanV1::new(ids.buffer, ids.offset, items * 20)
    );
    let at = run
        .step
        .iter()
        .position(|line| *line == readback)
        .expect("the draft ID readback");
    run.step.split_at(at)
}

/// The zero lines that cut every decoder's draft IDs and confidence to
/// three heads on every rank.
fn head_cuts(run: &Run, items: u64) -> Vec<String> {
    let mut lines = Vec::new();
    for (&(ids, confidence), stream) in run.outputs.0.iter().zip(&run.streams) {
        for output in [ids, confidence] {
            for item in 0..items {
                let span =
                    DeviceBufferSpanV1::new(output.buffer, output.offset + (item * 5 + 3) * 4, 8);
                lines.push(format!("zero {stream} {span:?}"));
            }
        }
    }
    lines
}

fn count(lines: &[String], prefix: &str) -> usize {
    lines.iter().filter(|line| line.starts_with(prefix)).count()
}

/// A mixed step after a decode step reuses the decode step's five-head
/// successor beside a cold prompt, a restored prefix hit and a prompt's
/// last chunk: it cuts the drafts to three heads on every rank, runs no
/// decoder catch-up or draft, and from its draft readback issues the inline
/// step's journal, with the same drafts, rows, calibration records and
/// committed contexts. Without the option it discards the successor and
/// records that the option's rule would have served it.
#[test]
fn a_mixed_step_reuses_the_decode_steps_successor() {
    for prefill in [Prefill::Cold, Prefill::Restored, Prefill::Chunk] {
        let inline = run(false, prefill, Between::Nothing, false);
        let reused = run(true, prefill, Between::Nothing, false);
        assert_eq!(inline.successor.len(), 1, "{prefill:?}");
        assert_eq!(inline.successor[0]["event"], "discarded", "{prefill:?}");
        assert_eq!(inline.successor[0]["relaxed"], true, "{prefill:?}");
        assert_eq!(reused.successor.len(), 1, "{prefill:?}");
        assert_eq!(reused.successor[0]["event"], "reused", "{prefill:?}");
        assert_eq!(reused.successor[0]["prefill_items"], 1);
        assert_eq!(reused.successor[0]["heads"], 3);
        assert_eq!(reused.outputs.1, 3);
        assert_eq!(inline.outputs.1, 3);
        let (inline_draft, inline_target) = split(&inline, DECODERS);
        let (reused_draft, reused_target) = split(&reused, DECODERS);
        assert_eq!(inline_target, reused_target, "{prefill:?}");
        let cuts = head_cuts(&reused, DECODERS);
        assert_eq!(
            reused_draft
                .iter()
                .filter(|line| line.starts_with("zero "))
                .filter(|line| cuts.contains(line))
                .count(),
            cuts.len(),
            "{prefill:?}"
        );
        assert!(!inline_draft.iter().any(|line| cuts.contains(line)));
        // Per rank: the decoders' batched catch-up and their draft.
        assert_eq!(
            count(inline_draft, "replay ") - count(reused_draft, "replay "),
            2 * reused.streams.len(),
            "{prefill:?}"
        );
        assert_eq!(inline.drafts, reused.drafts);
        assert_eq!(inline.drafts.len(), DECODERS as usize);
        assert_eq!(inline.rows, reused.rows);
        assert_eq!(inline.records, reused.records, "{prefill:?}");
        assert!(
            reused
                .records
                .iter()
                .rev()
                .take(DECODERS as usize)
                .all(|record| record["computed_positions"] == 3)
        );
        assert_eq!(inline.contexts, reused.contexts, "{prefill:?}");
        assert!(reused.contexts[DECODERS as usize].is_some());
    }
}

/// Between the queue and the mixed step, a decoder's capture, a decoder's
/// retirement or preemption, an idle request's eviction, a restore for a
/// later step and a committed ring write each discard the successor, and
/// the step drafts inline as without the option.
#[test]
fn lifecycle_changes_before_the_mixed_step_discard_the_successor() {
    for between in [
        Between::Capture,
        Between::Retire,
        Between::Preempt,
        Between::Evict,
        Between::RestoreOther,
        Between::RingWrite,
    ] {
        let off = run(false, Prefill::Cold, between, false);
        let on = run(true, Prefill::Cold, between, false);
        assert_eq!(on.successor.len(), 1, "{between:?}");
        assert_eq!(on.successor[0]["event"], "discarded", "{between:?}");
        assert_eq!(off.successor[0]["relaxed"], false, "{between:?}");
        assert_eq!(off.step, on.step, "{between:?}");
        assert_eq!(off.records, on.records, "{between:?}");
        assert_eq!(off.contexts, on.contexts, "{between:?}");
    }
}

/// A cancelled mixed step that reused the successor retires every request
/// it touched, the reused decoders included, as the inline step's cancel
/// does; the idle request keeps its context.
#[test]
fn a_cancelled_mixed_step_retires_its_reused_decoders() {
    let inline = run(false, Prefill::Restored, Between::Nothing, true);
    let reused = run(true, Prefill::Restored, Between::Nothing, true);
    assert_eq!(reused.successor[0]["event"], "reused");
    assert_eq!(inline.contexts, reused.contexts);
    assert!(
        reused.contexts[..=DECODERS as usize]
            .iter()
            .all(Option::is_none)
    );
    assert!(reused.contexts[DECODERS as usize + 1].is_some());
    assert_eq!(inline.records, reused.records);
}

/// With the option, a decode step after an allocated one reuses its
/// five-head successor cut to the three heads its compact draft computes;
/// without it the step keeps all five, as before.
#[test]
fn a_decode_step_reuses_a_five_head_successor_cut_to_its_heads() {
    let five = run(false, Prefill::None, Between::Nothing, false);
    let three = run(true, Prefill::None, Between::Nothing, false);
    for (run, heads) in [(&five, 5), (&three, 3)] {
        assert_eq!(run.successor[0]["event"], "reused");
        assert_eq!(run.successor[0]["heads"], heads);
        assert_eq!(run.outputs.1, heads);
    }
    // A decode-only step drafts for the device target, which reads the
    // drafts at its boundary: the rank-0 hash follows the draft.
    let hash = |run: &Run| {
        let prefix = format!(
            "launch {} {} ",
            run.streams[0],
            crate::sm120::engram_hash_rows::SYMBOL
        );
        run.step
            .iter()
            .position(|line| line.starts_with(&prefix))
            .expect("the rank-0 hash")
    };
    let cuts = head_cuts(&three, DECODERS);
    assert!(
        cuts.iter()
            .all(|line| three.step[..hash(&three)].contains(line))
    );
    assert!(!cuts.iter().any(|line| five.step.contains(line)));
    assert_eq!(five.step[hash(&five)..], three.step[hash(&three)..]);
    assert_eq!(five.contexts, three.contexts);
}

/// A device copy that fails while the mixed step adopts the successor
/// poisons the drafter before any launch: the step fails and every later
/// step is refused.
#[test]
fn a_failed_adoption_poisons_before_any_launch() {
    let (host, mut instance) = create(&[(DSPARK_SUCCESSOR_PREFILL, true.into())]);
    for slot in 0..DECODERS {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    let decoders: Vec<_> = (0..DECODERS).map(|slot| (slot + 1, slot, 3)).collect();
    execute(&mut instance, &host, &verify_all(&decoders, 3));
    crate::instance::dspark_prefetch_for_test(instance.handle(), allocated_feedback(3, 3, 2), true);
    instance
        .commit_accepted(&(1..=DECODERS).map(|r| kept(r, 3)).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(
        crate::instance::dspark_successor_for_test(instance.handle()),
        Some(DECODERS as usize)
    );
    host.device.fail_copy.set(Some(1));
    let begin = host.journal.borrow().len();
    let mut sink = Sink::default();
    assert_eq!(
        instance
            .execute(
                &mixed(Some((PREFILL, PREFILL_SLOT, 0, 40)), DECODERS, 6, 3),
                &mut sink
            )
            .err(),
        Some(BundleAbiError::Status(BundleStatusV1::BundleError))
    );
    let step = host.journal.borrow()[begin..].to_vec();
    assert_eq!(
        count(&step, "copy "),
        1,
        "the ring copy, then the failed draft copy"
    );
    assert!(
        !step
            .iter()
            .any(|line| line.starts_with("launch ") || line.starts_with("replay "))
    );
    assert_eq!(
        crate::instance::dspark_successor_for_test(instance.handle()),
        None
    );
    let begin = host.journal.borrow().len();
    assert!(
        instance
            .execute(&verify_all(&decoders, 3), &mut Sink::default())
            .is_err()
    );
    assert!(
        host.journal.borrow()[begin..]
            .iter()
            .all(|line| !line.starts_with("launch ") && !line.starts_with("replay "))
    );
}

/// Every committed drafter ring writer advances the count a successor
/// records at its queue: a commit's projection of a prompt wider than the
/// retained history, a capture's catch-up, a step's batched catch-up and
/// a bulk range's per-stage passes. A successor's private projection and
/// draft, its adoption and the step that reuses it write nothing.
#[test]
fn committed_ring_writes_advance_the_count_and_a_successor_does_not() {
    let (host, mut instance) = create(&[
        (DIAGNOSTIC_STEP_LOGS, true.into()),
        (DSPARK_SUCCESSOR_PREFILL, true.into()),
    ]);
    let handle = instance.handle();
    let writes = || crate::instance::dspark_ring_writes_for_test(handle, 0);
    for slot in 0..DECODERS {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    assert_eq!(writes(), 0, "short prompts defer their rows");
    execute(
        &mut instance,
        &host,
        &super::super::prompt(CAPTURED, 8, 0, CAPTURED_ROWS, true),
    );
    assert_eq!(writes(), 0);
    instance.commit().unwrap();
    let committed = writes();
    assert!(committed > 0, "the commit projects every staged row");
    let pages = snapshot_pages(&mut instance);
    instance.capture_prefix_state(1, 3, &pages).unwrap();
    assert_eq!(writes(), committed + 1, "the capture catches decoder 1 up");
    instance.retire_request(CAPTURED).unwrap();
    execute(
        &mut instance,
        &host,
        &super::super::prompt(PREFILL, PREFILL_SLOT, 0, 64, true),
    );
    instance.commit().unwrap();
    assert_eq!(writes(), committed + 1, "a bulk commit defers its rows");
    let mut items: Vec<_> = (0..DECODERS).map(|slot| (slot + 1, slot, 3)).collect();
    items.push((PREFILL, PREFILL_SLOT, 64));
    execute(&mut instance, &host, &verify_all(&items, 3));
    let stages = u64::from(FLASH.dspark_stages());
    assert_eq!(
        writes(),
        committed + 2 + stages,
        "one batched catch-up and a pass a stage"
    );
    let mut kept_rows: Vec<_> = (1..=DECODERS).map(|r| kept(r, 1)).collect();
    kept_rows.push(kept(PREFILL, 1));
    instance.commit_accepted(&kept_rows).unwrap();
    instance.retire_request(PREFILL).unwrap();
    let decoders: Vec<_> = (0..DECODERS).map(|slot| (slot + 1, slot, 4)).collect();
    execute(&mut instance, &host, &verify_all(&decoders, 3));
    let queued = writes();
    assert_eq!(queued, committed + 3 + stages);
    crate::instance::dspark_prefetch_for_test(handle, allocated_feedback(4, 3, 2), true);
    instance
        .commit_accepted(&(1..=DECODERS).map(|r| kept(r, 3)).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(writes(), queued, "the successor's private work");
    let logs = host.logs.borrow().len();
    execute(
        &mut instance,
        &host,
        &mixed(Some((PREFILL + 1, PREFILL_SLOT, 0, 20)), DECODERS, 7, 3),
    );
    instance
        .commit_accepted(&(1..=DECODERS).map(|r| kept(r, 1)).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(writes(), queued, "the reusing step and its commit");
    assert_eq!(successor_records(&host, logs)[0]["event"], "reused");
    for request in (1..=DECODERS).chain([PREFILL + 1, IDLE]) {
        if crate::instance::dspark_context(handle, request).is_some() {
            instance.retire_request(request).unwrap();
        }
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
