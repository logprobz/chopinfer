//! The `dspark_early_prefill_engram` candidate: a DSpark step's prefill
//! hashes its rows and begins its prepared gathers before catch-up and
//! drafting instead of just before its launch. Each step's prefill id
//! uploads and gather enqueues are the ones the default begins, moved
//! ahead of the step's first launch; every other journal line (launches
//! with their arguments, uploads with their bytes, gathered rows, joins,
//! graph captures and replays, readbacks), every log line and every result
//! is unchanged. Steps the prepared gathers do not serve begin nothing
//! early, and a step refused after the early begin settles its gathers.
use super::*;
use crate::fake_host_tests::ar_engram::order;
use crate::fake_host_tests::prefill_engram::create_backend_with_options;
use crate::instance::execute::plan_timing::restore_without_snapshot;
use model_deepseek41_contract::execution_options::{
    DSPARK_EARLY_PREFILL_ENGRAM, DSPARK_GRAPH_HANDLES_PER_RANK, DSPARK_MIXED_PREPARED_ENGRAM,
    PREFILL_PLAN_CACHE_VERIFY,
};
use runtime_bundle_sdk::prefill::{Boundary, encode};

mod plan;

/// KV pages of a slot: slot `s` takes pages from `s * SLOT_PAGES`, room
/// for a prompt of about 100K tokens.
const SLOT_PAGES: u64 = 1600;

fn pages(slot: u64, end: u64) -> Vec<u64> {
    (0..end.div_ceil(64))
        .map(|page| slot * SLOT_PAGES + page)
        .collect()
}

/// The request verifying in slot `slot`.
fn verifier(slot: u64) -> u64 {
    100 + slot
}

/// A prefill item: `rows` rows of `request`'s prompt in `slot` after its
/// `prefix` committed tokens, ending the prompt.
#[derive(Clone, Copy, Debug)]
struct Prefill {
    request: u64,
    slot: u64,
    prefix: u64,
    rows: u64,
}

fn prefill(request: u64, slot: u64, prefix: u64, rows: u64) -> Prefill {
    Prefill {
        request,
        slot,
        prefix,
        rows,
    }
}

fn token(request: u64, position: u64) -> u32 {
    ((position * 7 + request) % 1000) as u32
}

/// What follows a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Then {
    /// Commit, accepting every verify row, and retire the prefills.
    Commit,
    /// Cancel, which invalidates every request the step touched: they
    /// retire and start again (the verifiers' prompts, the prefills'
    /// restores), and the same step runs again and commits.
    CancelRetry,
}

/// A step: its prefill items beside, when `verify`, every verifier's next
/// rows.
#[derive(Clone, Debug)]
struct Step {
    prefills: Vec<Prefill>,
    verify: bool,
    then: Then,
}

fn step(prefills: &[Prefill], verify: bool, then: Then) -> Step {
    Step {
        prefills: prefills.to_vec(),
        verify,
        then,
    }
}

/// The negotiated prefill envelope of `prefills` beside verify items of
/// `(slot, committed tokens, rows)`.
fn batch(prefills: &[Prefill], verifies: &[(u64, u64, u32)]) -> Vec<u8> {
    let (original, boundaries) = original(prefills, verifies);
    encode(&original, &boundaries).unwrap()
}

/// The original batch of `batch`, and its prefills' boundaries.
fn original(prefills: &[Prefill], verifies: &[(u64, u64, u32)]) -> (Vec<u8>, Vec<Boundary>) {
    let mut items = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    let mut boundaries = Vec::new();
    for p in prefills {
        let end = p.prefix + p.rows;
        items.push(BatchItemV1::new(
            p.request,
            false,
            p.rows as u32,
            true,
            (p.prefix..end).map(|at| token(p.request, at)).collect(),
        ));
        appends.push(BatchAppendV1::new(
            p.prefix,
            p.rows,
            end,
            pages(p.slot, end),
            None,
        ));
        slots.push(p.slot);
        boundaries.push(Boundary {
            request_id: p.request,
            prompt_tokens: end,
            chunk_end: end,
            chunk_tokens: 8192,
        });
    }
    for &(slot, start, rows) in verifies {
        let end = start + u64::from(rows);
        items.push(BatchItemV1::verify(verifier(slot), rows, true, vec![7]));
        appends.push(BatchAppendV1::new(
            start,
            u64::from(rows),
            end,
            pages(slot, end),
            None,
        ));
        slots.push(slot);
    }
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap();
    (original, boundaries)
}

/// What a recorded step runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Prompt,
    Mixed,
    Prefill,
}

/// One executed step: its kind, journal, drafts and logits views.
struct Recorded {
    kind: Kind,
    journal: Vec<String>,
    drafts: Vec<OwnedDraftTokensV1>,
    views: Vec<OwnedLogitsViewV1>,
}

/// What one schedule left.
struct Run {
    steps: Vec<Recorded>,
    logs: Vec<String>,
    /// The prefill gathers' id spans.
    indices: Vec<DeviceBufferSpanV1>,
}

/// A schedule's instance and traffic.
struct Schedule<'a> {
    /// The verify capacity and each verifier's rows a step.
    verify_rows: u32,
    widths: &'a [u32],
    /// Requests restored as `(request, slot, committed tokens)` before
    /// the steps, without their snapshot copies.
    restored: &'a [(u64, u32, u64)],
    steps: &'a [Step],
}

impl Schedule<'_> {
    /// Runs the schedule on an upper-skip instance with `options`: a
    /// three-token prompt per verifier (prefill-only steps, recorded
    /// too), the restores, then each step.
    fn run(&self, options: &[(&str, serde_json::Value)]) -> Run {
        let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
        host.speculation.set(Some((5, self.verify_rows)));
        host.kv_pages.set(24 * SLOT_PAGES);
        let mut instance = create_backend_with_options(
            &host,
            true,
            MappedHost::with_row_ids(Rc::clone(&host)),
            options,
        )
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        let indices = crate::instance::prefill_engram::indices(instance.handle())
            .into_iter()
            .map(|(_, span)| span)
            .collect();
        let mut steps = Vec::new();
        let mut record = |instance: &mut BundleInstanceV1, kind: Kind, request: &[u8]| {
            let begin = host.journal.borrow().len();
            let sink = run(instance, &host, request);
            steps.push(Recorded {
                kind,
                journal: host.journal.borrow()[begin..].to_vec(),
                drafts: sink.drafts,
                views: sink.views,
            });
        };
        let prompts =
            |instance: &mut BundleInstanceV1,
             record: &mut dyn FnMut(&mut BundleInstanceV1, Kind, &[u8])| {
                for slot in 0..self.widths.len() as u64 {
                    let prompt = batch(&[prefill(verifier(slot), slot, 0, 3)], &[]);
                    record(instance, Kind::Prompt, &prompt);
                    instance.commit().unwrap();
                }
                vec![3; self.widths.len()]
            };
        let restore = |instance: &BundleInstanceV1, prefills: &[Prefill]| {
            for &(request, slot, prefix) in self.restored {
                if prefills.iter().any(|p| p.request == request) {
                    let tokens: Vec<_> = (0..prefix).map(|at| token(request, at)).collect();
                    restore_without_snapshot(instance.handle(), request, slot, &tokens).unwrap();
                }
            }
        };
        let mut ends = prompts(&mut instance, &mut record);
        for step in self.steps {
            restore(&instance, &step.prefills);
            let verifies = |ends: &[u64]| -> Vec<_> {
                if !step.verify {
                    return Vec::new();
                }
                self.widths
                    .iter()
                    .zip(ends)
                    .enumerate()
                    .map(|(slot, (&width, &end))| (slot as u64, end, width))
                    .collect()
            };
            let kind = if step.verify {
                Kind::Mixed
            } else {
                Kind::Prefill
            };
            let mut current = verifies(&ends);
            record(&mut instance, kind, &batch(&step.prefills, &current));
            if step.then == Then::CancelRetry {
                instance.cancel().unwrap();
                for p in &step.prefills {
                    instance.retire_request(p.request).unwrap();
                }
                for slot in 0..self.widths.len() as u64 {
                    instance.retire_request(verifier(slot)).unwrap();
                }
                ends = prompts(&mut instance, &mut record);
                restore(&instance, &step.prefills);
                current = verifies(&ends);
                record(&mut instance, kind, &batch(&step.prefills, &current));
            }
            let kept: Vec<_> = current
                .iter()
                .map(|&(slot, _, width)| kept(verifier(slot), width))
                .collect();
            instance.commit_accepted(&kept).unwrap();
            for (end, &(_, _, width)) in ends.iter_mut().zip(&current) {
                *end += u64::from(width);
            }
            for p in &step.prefills {
                instance.retire_request(p.request).unwrap();
            }
        }
        for slot in 0..self.widths.len() as u64 {
            instance.retire_request(verifier(slot)).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        assert!(host.races.borrow().is_empty(), "{:#?}", host.races.borrow());
        let logs = host
            .logs
            .borrow()
            .iter()
            .map(|line| untimed(line))
            .collect();
        Run {
            steps,
            logs,
            indices,
        }
    }
}

/// `journal` split into the prefill gathers' begin, the uploads of their
/// ids and their enqueues in order, and every other line.
fn split(journal: &[String], indices: &[DeviceBufferSpanV1]) -> (Vec<String>, Vec<String>) {
    let ids = indices
        .iter()
        .map(|span| format!("buffer: {:?}, offset: {}, ", span.buffer, span.offset))
        .collect::<Vec<_>>();
    let enqueue = format!("async mapped ids {} ", indices[0].length);
    journal.iter().cloned().partition(|line| {
        (line.starts_with("upload ") && ids.iter().any(|span| line.contains(span)))
            || line.starts_with(&enqueue)
    })
}

/// The first line `a` and `b` differ at, with its neighbours.
#[track_caller]
fn assert_same_lines(a: &[String], b: &[String], context: &str) {
    if a == b {
        return;
    }
    let at = a
        .iter()
        .zip(b)
        .position(|(a, b)| a != b)
        .unwrap_or(a.len().min(b.len()));
    let near = |lines: &[String]| lines[at.saturating_sub(3)..(at + 3).min(lines.len())].to_vec();
    panic!(
        "{context}: {} and {} lines, first difference at {at}:\n{:#?}\n{:#?}",
        a.len(),
        b.len(),
        near(a),
        near(b)
    );
}

/// `line` without the host times a log line reports.
pub(in crate::fake_host_tests) fn untimed(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = ["\"micros\":", "_ms="]
        .iter()
        .filter_map(|marker| rest.find(marker).map(|at| at + marker.len()))
        .min()
    {
        out.push_str(&rest[..at]);
        let digits = rest[at..]
            .find(|c: char| !c.is_ascii_digit() && c != '.')
            .unwrap_or(rest.len() - at);
        out.push('N');
        rest = &rest[at + digits..];
    }
    out.push_str(rest);
    out
}

fn is_device_work(line: &str) -> bool {
    line.starts_with("launch ") || line.starts_with("replay ") || line.starts_with("capture ")
}

/// Each step of `on` issues `off`'s step with only the prefill gathers'
/// begin moved, and the same results; every log line matches. Returns how
/// many steps began early.
#[track_caller]
fn assert_moved_begins(off: &Run, on: &Run, context: &str) -> usize {
    assert_eq!(off.indices, on.indices);
    assert_eq!(off.steps.len(), on.steps.len(), "{context}");
    let mut early = 0;
    for (index, (a, b)) in off.steps.iter().zip(&on.steps).enumerate() {
        let context = format!("{context}, step {index}");
        let (begin_off, rest_off) = split(&a.journal, &off.indices);
        let (begin_on, rest_on) = split(&b.journal, &on.indices);
        assert_same_lines(&begin_off, &begin_on, &format!("{context}: the begins"));
        assert_same_lines(&rest_off, &rest_on, &context);
        assert_eq!(a.drafts, b.drafts, "{context}");
        assert_eq!(a.views, b.views, "{context}");
        if begin_on.is_empty() {
            continue;
        }
        // Two layers: their id uploads, then their enqueues.
        assert_eq!(begin_on.len(), 4, "{context}: {begin_on:?}");
        let last = b
            .journal
            .iter()
            .rposition(|line| line == &begin_on[3])
            .unwrap();
        assert!(
            !b.journal[..last].iter().any(|line| is_device_work(line)),
            "{context}: the gathers began after device work"
        );
        early += 1;
    }
    assert_same_lines(&off.logs, &on.logs, &format!("{context}: logs"));
    early
}

/// Whether each mixed step of `run` issued device work before its prefill
/// gathers began.
fn mixed_began_after_device_work(run: &Run) -> Vec<bool> {
    run.steps
        .iter()
        .filter(|step| step.kind == Kind::Mixed)
        .map(|step| {
            let (begin, _) = split(&step.journal, &run.indices);
            let at = step
                .journal
                .iter()
                .position(|line| line == &begin[0])
                .unwrap();
            step.journal[..at].iter().any(|line| is_device_work(line))
        })
        .collect()
}

fn with(
    options: &[(&'static str, serde_json::Value)],
    more: &[(&'static str, serde_json::Value)],
) -> Vec<(&'static str, serde_json::Value)> {
    options.iter().chain(more).cloned().collect()
}

/// Mixed C8 steps (eight four-row verify items) beside fresh prompts:
/// a 512-row prefill lowered and kept, the same shape again from its
/// template (and cancelled and run again), the 511 rows below upper-skip
/// DMA, a 2048-row miss, the 4097 rows past a 4096-row part, an 8000-row
/// prefill near the chunk, two prefill items in one step, and a prefill
/// step without verify items; graphed, eager (no graph handles), with
/// every reused plan lowered again, and beside the mixed verify-row
/// gathers. Every prompt and mixed step begins early, after nothing.
#[test]
fn early_begins_move_only_the_prefill_gather_begin() {
    use Then::*;
    let steps = [
        step(&[prefill(7, 8, 0, 512)], true, Commit),
        step(&[prefill(8, 9, 0, 512)], true, CancelRetry),
        step(&[prefill(9, 10, 0, 511)], true, Commit),
        step(&[prefill(10, 11, 0, 2048)], true, Commit),
        step(&[prefill(11, 12, 0, 4097)], true, Commit),
        step(&[prefill(12, 13, 0, 8000)], true, Commit),
        step(
            &[prefill(13, 14, 0, 300), prefill(14, 15, 0, 700)],
            true,
            Commit,
        ),
        step(&[prefill(15, 8, 0, 512)], false, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[],
        steps: &steps,
    };
    let on = [(DSPARK_EARLY_PREFILL_ENGRAM, true.into())];
    for (name, options) in [
        ("graphed", vec![]),
        ("eager", vec![(DSPARK_GRAPH_HANDLES_PER_RANK, 0.into())]),
        ("verified", vec![(PREFILL_PLAN_CACHE_VERIFY, true.into())]),
        ("mixed", vec![(DSPARK_MIXED_PREPARED_ENGRAM, true.into())]),
    ] {
        let off = schedule.run(&options);
        let early = schedule.run(&with(&options, &on));
        // Sixteen prompts (eight again after the cancel), seven mixed
        // steps, the retry and the prefill step.
        assert_eq!(off.steps.len(), 16 + 8 + 1);
        assert_eq!(assert_moved_begins(&off, &early, name), 16 + 8 + 1);
        assert_eq!(mixed_began_after_device_work(&off), [true; 8], "{name}");
        assert_eq!(mixed_began_after_device_work(&early), [false; 8], "{name}");
    }
}

/// A verify phase above the graphed 64 rows (eleven six-row items, 66
/// rows, eager) beside a 512-row prefill, as a new template and reused.
#[test]
fn eager_verify_early_begins_move_only_the_prefill_gather_begin() {
    use Then::*;
    let steps = [
        step(&[prefill(7, 11, 0, 512)], true, Commit),
        step(&[prefill(8, 12, 0, 512)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 192,
        widths: &[6; 11],
        restored: &[],
        steps: &steps,
    };
    let off = schedule.run(&[]);
    let on = schedule.run(&[(DSPARK_EARLY_PREFILL_ENGRAM, true.into())]);
    assert_eq!(assert_moved_begins(&off, &on, "eager verify"), 11 + 2);
    assert_eq!(mixed_began_after_device_work(&on), [false; 2]);
}

/// The live shape: a 1536-row prefill after a 93,000-token context beside
/// eight verify items, its template reused by the same rows after
/// another 128 tokens (cancelled and run again too), and a 1537-row miss.
/// With every reused plan lowered again.
#[test]
fn long_context_early_begins_move_only_the_prefill_gather_begin() {
    use Then::*;
    let steps = [
        step(&[prefill(20, 8, 93_000, 1536)], true, Commit),
        step(&[prefill(21, 9, 93_128, 1536)], true, CancelRetry),
        step(&[prefill(22, 10, 93_000, 1537)], true, Commit),
    ];
    let schedule = Schedule {
        verify_rows: 32,
        widths: &[4; 8],
        restored: &[(20, 8, 93_000), (21, 9, 93_128), (22, 10, 93_000)],
        steps: &steps,
    };
    let verify = [(PREFILL_PLAN_CACHE_VERIFY, true.into())];
    let off = schedule.run(&verify);
    let on = schedule.run(&with(
        &verify,
        &[(DSPARK_EARLY_PREFILL_ENGRAM, true.into())],
    ));
    // Sixteen prompts, three mixed steps and the retry.
    assert_eq!(assert_moved_begins(&off, &on, "long context"), 16 + 4);
    let kept = |run: &Run| {
        run.logs
            .iter()
            .filter(|line| line.contains("prefill plan templates: kept"))
            .count()
    };
    // The 1536 and 1537 rows keep templates; the shifted context and the
    // retry reuse the first.
    assert_eq!((kept(&off), kept(&on)), (2, 2));
}

/// Eight three-token prompts on an upper-skip instance with `options` on
/// `backend`, and a mixed C8 step of a 512-row prefill: its request.
fn prompted(
    options: &[(&str, serde_json::Value)],
    backend: impl FnOnce(Rc<Host>) -> MappedHost,
) -> (Rc<Host>, BundleInstanceV1, Vec<u8>) {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.speculation.set(Some((5, 32)));
    host.kv_pages.set(24 * SLOT_PAGES);
    let mut instance =
        create_backend_with_options(&host, true, backend(Rc::clone(&host)), options).unwrap();
    for slot in 0..8 {
        run(
            &mut instance,
            &host,
            &batch(&[prefill(verifier(slot), slot, 0, 3)], &[]),
        );
        instance.commit().unwrap();
    }
    let verifies: Vec<_> = (0..8).map(|slot| (slot, 3, 4)).collect();
    let mixed = batch(&[prefill(7, 8, 0, 512)], &verifies);
    (host, instance, mixed)
}

#[track_caller]
fn run(instance: &mut BundleInstanceV1, host: &Host, request: &[u8]) -> Sink {
    let mut sink = Sink::default();
    instance
        .execute_prefill(request, &mut sink)
        .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
    sink
}

fn spans(instance: &BundleInstanceV1) -> Vec<DeviceBufferSpanV1> {
    crate::instance::prefill_engram::indices(instance.handle())
        .into_iter()
        .map(|(_, span)| span)
        .collect()
}

/// A failed prefill join or completion refuses the mixed step as the
/// default's does, with the same journal but for the moved begin, and
/// publishes nothing. A failed enqueue, now before catch-up and
/// drafting, refuses the step before any device work.
#[test]
fn gather_failures_after_an_early_begin_refuse_as_the_default_does() {
    let on = [(DSPARK_EARLY_PREFILL_ENGRAM, true.into())];
    // Eight prompts join two layers on four ranks each and complete both.
    for (stage, after) in [("wait", 64 + 2), ("complete", 16), ("enqueue", 16)] {
        let run = |options: &[(&str, serde_json::Value)]| {
            let (host, mut instance, mixed) = prompted(options, |host| {
                MappedHost::with_row_ids(host).failing(stage, after)
            });
            let indices = spans(&instance);
            let begin = host.journal.borrow().len();
            let mut sink = Sink::default();
            assert!(instance.execute_prefill(&mixed, &mut sink).is_err());
            assert!(sink.views.is_empty());
            let step = host.journal.borrow()[begin..].to_vec();
            assert!(step.contains(&format!("async mapped failure {stage}")));
            let _ = instance.cancel();
            match instance.destroy() {
                Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
                Err(_) => assert!(instance.is_poisoned()),
            }
            (step, indices)
        };
        let (off, indices) = run(&[]);
        let (early, _) = run(&on);
        if stage == "enqueue" {
            assert!(!early.iter().any(|line| is_device_work(line)), "{early:#?}");
            assert!(off.iter().any(|line| is_device_work(line)));
            continue;
        }
        let (begin_off, rest_off) = split(&off, &indices);
        let (begin_on, rest_on) = split(&early, &indices);
        assert_eq!(begin_off.len(), 4, "{stage}");
        assert_same_lines(&begin_off, &begin_on, &format!("{stage}: the begins"));
        assert_same_lines(&rest_off, &rest_on, stage);
    }
}

/// A mixed step its target refuses after the early begin (here a reused
/// template's stale Engram plan, which the verify mode refuses) leaves
/// its gathers pending for the next execute's entry. That settle joins
/// them on every stream, fences every stream and completes both before
/// any later upload can reuse their rows.
#[test]
fn a_step_refused_after_an_early_begin_settles_its_gathers_at_the_next_entry() {
    let options = [
        (DSPARK_EARLY_PREFILL_ENGRAM, true.into()),
        (PREFILL_PLAN_CACHE_VERIFY, true.into()),
    ];
    let (host, mut instance, mixed) = prompted(&options, MappedHost::with_row_ids);
    run(&mut instance, &host, &mixed);
    let kept: Vec<_> = (0..8).map(|slot| kept(verifier(slot), 4)).collect();
    instance.commit_accepted(&kept).unwrap();
    instance.retire_request(7).unwrap();
    let verifies: Vec<_> = (0..8).map(|slot| (slot, 7, 4)).collect();
    let again = batch(&[prefill(8, 9, 0, 512)], &verifies);
    let begin = host.journal.borrow().len();
    let refused = crate::instance::execute::skip::with_stale_engram(|| {
        instance.execute_prefill(&again, &mut Sink::default())
    });
    assert!(refused.is_err());
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("differs from a fresh one"))
    );
    assert_eq!(
        crate::instance::prefill_engram::refusal_state(instance.handle()),
        (true, true)
    );
    {
        // The early begin, then the settle: a join on every stream, each
        // stream's fence after its join, and both completions.
        let journal = host.journal.borrow();
        let step = &journal[begin..];
        let kinds = order(step);
        assert_eq!(kinds.iter().filter(|&&kind| kind == "ids").count(), 2);
        let first = step
            .iter()
            .position(|line| line.starts_with("async mapped wait "))
            .expect("the refused step's settle");
        let settled = &step[first..];
        assert_eq!(
            order(settled),
            [vec!["wait"; 8], vec!["complete"; 2]].concat(),
            "{settled:#?}"
        );
        let waited: Vec<_> = settled
            .iter()
            .filter_map(|line| line.strip_prefix("async mapped wait "))
            .collect();
        for stream in &waited {
            let wait = settled
                .iter()
                .position(|line| line == &format!("async mapped wait {stream}"))
                .unwrap();
            assert!(
                settled[wait..]
                    .iter()
                    .any(|line| line == &format!("synchronize {stream}")),
                "stream {stream} is not fenced: {settled:#?}"
            );
        }
    }
    let settle = host.journal.borrow().len();
    crate::instance::prefill_engram::settle_refused(instance.handle()).unwrap();
    assert_eq!(host.journal.borrow().len(), settle);
    assert!(crate::instance::prefill_engram::refusal_state(instance.handle()).0);
    let _ = instance.cancel();
    match instance.destroy() {
        Ok(()) => assert_eq!(host.acquired.get(), host.released.get()),
        Err(_) => assert!(instance.is_poisoned()),
    }
}
