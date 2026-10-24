//! Kept upper-skip plans issue exactly what a fresh lowering issues.
use super::*;
use crate::instance::execute::plan_cache::{self, MAX_ROWS, Mode};
use bundle_runtime::fake_host::RecordingSink;
use runtime_bundle_sdk::prefill::{Boundary, encode};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

type Sink = RecordingSink<super::conformance::Limits>;

/// A whole prompt of `rows` tokens from `first`, in `slot` on `pages`,
/// scoring its last row or, when `scored`, every row (which takes the full
/// schedule: such an item has no chunk boundary).
struct Prompt {
    request: u64,
    slot: u64,
    rows: u32,
    first: u32,
    pages: Vec<u64>,
    scored: bool,
}

fn prompt(request: u64, slot: u64, rows: u32, first: u32, page: u64) -> Prompt {
    Prompt {
        request,
        slot,
        rows,
        first,
        pages: (0..u64::from(rows).div_ceil(64))
            .map(|p| page + p)
            .collect(),
        scored: false,
    }
}

fn scored(prompt: Prompt) -> Prompt {
    Prompt {
        scored: true,
        ..prompt
    }
}

/// One step of `prompts` then `decodes`: (request, slot, previous tokens,
/// token, first page).
fn batch(prompts: &[Prompt], decodes: &[(u64, u64, u64, u32, u64)]) -> Vec<u8> {
    let mut items = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    let mut boundaries = Vec::new();
    for p in prompts {
        let inputs = (p.first..p.first + p.rows).collect();
        items.push(
            BatchItemV1::new(p.request, false, p.rows, true, inputs).with_prompt_rows(p.scored),
        );
        let rows = u64::from(p.rows);
        appends.push(BatchAppendV1::new(0, rows, rows, p.pages.clone(), None));
        slots.push(p.slot);
        if p.scored {
            continue;
        }
        boundaries.push(Boundary {
            request_id: p.request,
            prompt_tokens: rows,
            chunk_end: rows,
            chunk_tokens: 8192,
        });
    }
    for &(request, slot, previous, token, page) in decodes {
        items.push(BatchItemV1::new(request, true, 1, true, vec![token]));
        let end = previous + 1;
        let pages = (0..end.div_ceil(64)).map(|p| page + p).collect();
        appends.push(BatchAppendV1::new(previous, 1, end, pages, None));
        slots.push(slot);
    }
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap();
    encode(&original, &boundaries).unwrap()
}

fn counts(instance: &BundleInstanceV1) -> (u64, u64) {
    plan_cache::counts(instance.handle())
}

enum Then {
    Commit,
    Cancel,
    /// Commit a DSpark verify step keeping one row of `request`.
    Accept(u64),
}

/// A short prompt beside a verify item of `request` in `slot` with 5 drafts
/// after `start` tokens, on `page`.
fn beside_verify(prompt: Prompt, request: u64, slot: u64, start: u64, page: u64) -> Vec<u8> {
    let mut items = vec![
        BatchItemV1::new(
            prompt.request,
            false,
            prompt.rows,
            true,
            (prompt.first..prompt.first + prompt.rows).collect(),
        ),
        BatchItemV1::verify(request, 6, true, vec![7]),
    ];
    let rows = u64::from(prompt.rows);
    let appends = vec![
        BatchAppendV1::new(0, rows, rows, prompt.pages.clone(), None),
        BatchAppendV1::new(start, 6, start + 6, vec![page], None),
    ];
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, std::mem::take(&mut items), appends)
        .with_state_slots(vec![prompt.slot, slot])
        .encode()
        .unwrap();
    encode(
        &original,
        &[Boundary {
            request_id: prompt.request,
            prompt_tokens: rows,
            chunk_end: rows,
            chunk_tokens: 8192,
        }],
    )
    .unwrap()
}

/// Steps over alternating prompts, slots and pages: repeated shapes, mixed
/// steps, a cancelled prefill and ineligible prefills. Returns the journal
/// and the instance's (hits, misses).
fn lifecycle(mode: Mode) -> (Vec<String>, (u64, u64)) {
    lifecycle_on(mode, false)
}

fn lifecycle_on(mode: Mode, dspark: bool) -> (Vec<String>, (u64, u64)) {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.drafter.set(dspark);
    if dspark {
        host.speculation.set(Some((5, 32)));
    }
    let mut instance = plan_cache::with_mode(mode, || {
        super::prefill_skip::create_profile(&host, 8192, true)
    });
    let mut steps: Vec<(Vec<u8>, Then, &[u64])> = vec![
        (batch(&[prompt(1, 0, 22, 100, 0)], &[]), Then::Commit, &[]),
        (batch(&[prompt(2, 1, 22, 300, 9)], &[]), Then::Commit, &[]),
        (
            batch(
                &[prompt(3, 2, 22, 500, 17)],
                &[(1, 0, 22, 7, 0), (2, 1, 22, 8, 9)],
            ),
            Then::Commit,
            &[],
        ),
        (batch(&[prompt(4, 3, 7, 700, 25)], &[]), Then::Commit, &[4]),
        // Every row scored: the full schedule, never a kept plan.
        (
            batch(&[scored(prompt(11, 3, 22, 800, 26))], &[]),
            Then::Commit,
            &[11],
        ),
        // The shortest and longest kept lengths, each reused once.
        (
            batch(&[prompt(12, 3, 1, 850, 27)], &[]),
            Then::Commit,
            &[12],
        ),
        (
            batch(&[prompt(13, 3, 1, 860, 28)], &[]),
            Then::Commit,
            &[13],
        ),
        (
            batch(&[prompt(14, 3, MAX_ROWS, 870, 29)], &[]),
            Then::Commit,
            &[14],
        ),
        (
            batch(&[prompt(15, 3, MAX_ROWS, 940, 30)], &[]),
            Then::Commit,
            &[15],
        ),
        // 22 new rows after a 64-row prefix (its first chunk on the full
        // schedule): never the kept 22-row plan.
        (
            encode(&super::prompt(16, 3, 0, 64, false), &[]).unwrap(),
            Then::Commit,
            &[],
        ),
        (
            encode(
                &super::prompt(16, 3, 64, 22, true),
                &[Boundary {
                    request_id: 16,
                    prompt_tokens: 86,
                    chunk_end: 86,
                    chunk_tokens: 8192,
                }],
            )
            .unwrap(),
            Then::Commit,
            &[16],
        ),
        // Cancelled: the request is invalidated and retires.
        (
            batch(&[prompt(5, 3, 22, 900, 28)], &[(3, 2, 22, 9, 17)]),
            Then::Cancel,
            &[5, 3],
        ),
        (
            batch(&[prompt(6, 3, 22, 1100, 30)], &[(1, 0, 23, 10, 0)]),
            Then::Commit,
            &[6],
        ),
        // Ineligible: two prompts, and a prompt longer than MAX_ROWS.
        (
            batch(
                &[prompt(7, 2, 22, 1300, 33), prompt(8, 3, 22, 1500, 35)],
                &[],
            ),
            Then::Commit,
            &[7, 8],
        ),
        (
            batch(
                &[prompt(9, 3, MAX_ROWS + 1, 1700, 37)],
                &[(2, 1, 23, 11, 9)],
            ),
            Then::Commit,
            &[9],
        ),
        (
            batch(&[prompt(10, 2, 7, 1900, 40)], &[]),
            Then::Commit,
            &[1, 2, 10],
        ),
    ];
    if dspark {
        // A prefill beside a verify item with drafts, reusing the 22-row plan.
        steps.push((batch(&[prompt(21, 0, 22, 2000, 0)], &[]), Then::Commit, &[]));
        steps.push((
            beside_verify(prompt(22, 1, 22, 2100, 9), 21, 0, 22, 0),
            Then::Accept(21),
            &[21, 22],
        ));
    }
    for (request, then, retire) in steps {
        let mut sink = Sink::default();
        instance
            .execute_prefill(&request, &mut sink)
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        match then {
            Then::Commit => instance.commit().unwrap(),
            Then::Cancel => instance.cancel().unwrap(),
            Then::Accept(request) => instance
                .commit_accepted(&[BundleKeptRowsV1 {
                    struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                    reserved: 0,
                    request_id: request,
                    kept_rows: 1,
                    reserved_tail: 0,
                }])
                .unwrap(),
        }
        for &request in retire {
            instance.retire_request(request).unwrap();
        }
    }
    let counts = counts(&instance);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    (journal, counts)
}

#[test]
fn kept_plans_issue_the_fresh_lowerings_journal() {
    let (fresh, counts) = lifecycle(Mode::Off);
    assert_eq!(counts, (0, 0));
    assert!(fresh.iter().filter(|l| l.starts_with("launch ")).count() > 50_000);
    // Four kept plans (1, 7, 22 and 64 rows); every other eligible prefill
    // reuses one. The scored, two-prompt, 65-row and prefixed prefills lower
    // fresh.
    for mode in [Mode::On, Mode::Verify] {
        let (kept, counts) = lifecycle(mode);
        assert_eq!(kept.len(), fresh.len(), "{mode:?}");
        if let Some(at) = (0..fresh.len()).find(|&at| kept[at] != fresh[at]) {
            panic!(
                "{mode:?} journal line {at}: {} against {}",
                kept[at], fresh[at]
            );
        }
        assert_eq!(counts, (7, 4), "{mode:?}");
    }
}

#[test]
fn a_dspark_target_keeps_plans_with_the_fresh_lowerings_journal() {
    let (fresh, counts) = lifecycle_on(Mode::Off, true);
    assert_eq!(counts, (0, 0));
    // The same plans as without DSpark, and the 22-row plan reused beside a
    // verify item with drafts.
    for mode in [Mode::On, Mode::Verify] {
        let (kept, counts) = lifecycle_on(mode, true);
        assert_eq!(kept.len(), fresh.len(), "{mode:?}");
        if let Some(at) = (0..fresh.len()).find(|&at| kept[at] != fresh[at]) {
            panic!(
                "{mode:?} journal line {at}: {} against {}",
                kept[at], fresh[at]
            );
        }
        assert_eq!(counts, (9, 4), "{mode:?}");
    }
}

/// A run's journal, its (hits, misses), and the rows it keeps and the plans
/// it dropped.
type Outcome = (Vec<String>, (u64, u64), (Vec<u32>, u64));

/// One prompt of each of `lengths` in turn, alone and retired after its
/// commit, on an instance keeping at most `plans` plans.
fn one_at_a_time(mode: Mode, plans: usize, lengths: &[u32]) -> Outcome {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    let mut instance = plan_cache::with_limit(mode, plans, || {
        super::prefill_skip::create_profile(&host, 8192, true)
    });
    for (index, &rows) in lengths.iter().enumerate() {
        let request = index as u64 + 1;
        let first = 100 + 70 * index as u32;
        let mut sink = Sink::default();
        instance
            .execute_prefill(
                &batch(&[prompt(request, 0, rows, first, 2 * index as u64)], &[]),
                &mut sink,
            )
            .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        instance.retire_request(request).unwrap();
    }
    let counts = counts(&instance);
    let kept = plan_cache::kept(instance.handle());
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    (journal, counts, kept)
}

#[test]
fn a_full_plan_cache_drops_its_least_recently_used_plan() {
    // With two plans kept: 3 and 5 are kept, 3 is reused, 9 drops 5 (the
    // least recently used), 5 is lowered again and drops 3, 9 is reused and
    // 3 is lowered again and drops 5.
    let lengths = [3, 5, 3, 9, 5, 9, 3];
    let (fresh, counts, kept) = one_at_a_time(Mode::Off, 2, &lengths);
    assert_eq!((counts, kept), ((0, 0), (vec![], 0)));
    for mode in [Mode::On, Mode::Verify] {
        let (journal, counts, kept) = one_at_a_time(mode, 2, &lengths);
        assert_eq!(journal.len(), fresh.len(), "{mode:?}");
        if let Some(at) = (0..fresh.len()).find(|&at| journal[at] != fresh[at]) {
            panic!(
                "{mode:?} journal line {at}: {} against {}",
                journal[at], fresh[at]
            );
        }
        assert_eq!(counts, (2, 5), "{mode:?}");
        assert_eq!(kept, (vec![3, 9], 3), "{mode:?}");
    }
}
