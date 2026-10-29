//! Upper-skip plan templates (`prefill_plan_templates`): a reused template
//! issues exactly what a fresh lowering issues, its key holds every input
//! but the positions, and a template whose index selections no longer
//! match falls back to a full lowering.
use super::*;
use crate::instance::execute::{plan_templates, plan_timing};
use bundle_runtime::fake_host::RecordingSink;
use bundle_runtime::template::Counts;
use model_deepseek41_contract::execution_options::{
    PREFILL_PLAN_CACHE_VERIFY, PREFILL_PLAN_TEMPLATES,
};
use runtime_bundle_sdk::prefill::{Boundary, encode};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

type Sink = RecordingSink<super::conformance::Limits>;

/// Pages a slot's request may use: slot `s` takes pages from `s * SLOT_PAGES`.
const SLOT_PAGES: u64 = 4096;

/// The prompt an unscored chunk belongs to, ending past every such chunk.
const UNSCORED_PROMPT: u64 = 81920;

/// A prompt chunk of `rows` rows after `prefix` tokens of `request` in
/// `slot`, scoring its last row when `scored`, with token IDs drawn from
/// `salt`.
#[derive(Clone, Copy)]
struct Chunk {
    request: u64,
    slot: u64,
    prefix: u64,
    rows: u64,
    scored: bool,
    salt: u64,
}

fn chunk(request: u64, slot: u64, prefix: u64, rows: u64) -> Chunk {
    Chunk {
        request,
        slot,
        prefix,
        rows,
        scored: true,
        salt: 0,
    }
}

fn token(salt: u64, position: u64) -> u32 {
    ((position * (2 * salt + 1) + salt) % 100_000) as u32
}

/// The original batch of `chunks` beside the one-row decode or six-row
/// verify items `decodes` (request, slot, committed tokens), and its
/// negotiated prefill envelope when its chunks end on the 8192-token grid:
/// a scored chunk ends its prompt, an unscored one a grid chunk of a longer
/// prompt.
fn batch(
    chunks: &[Chunk],
    decodes: &[(u64, u64, u64)],
    verify: bool,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let mut items = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    let mut boundaries = Vec::new();
    for c in chunks {
        let end = c.prefix + c.rows;
        let tokens = (c.prefix..end).map(|p| token(c.salt, p)).collect();
        items.push(BatchItemV1::new(
            c.request,
            false,
            c.rows as u32,
            c.scored,
            tokens,
        ));
        let pages = (0..end.div_ceil(64))
            .map(|page| c.slot * SLOT_PAGES + page)
            .collect();
        appends.push(BatchAppendV1::new(c.prefix, c.rows, end, pages, None));
        slots.push(c.slot);
        boundaries.push(Boundary {
            request_id: c.request,
            prompt_tokens: if c.scored { end } else { UNSCORED_PROMPT },
            chunk_end: end,
            chunk_tokens: 8192,
        });
    }
    for &(request, slot, previous) in decodes {
        let rows = if verify { 6 } else { 1 };
        items.push(if verify {
            BatchItemV1::verify(request, rows as u32, true, vec![7])
        } else {
            BatchItemV1::new(request, true, 1, true, vec![7])
        });
        let end = previous + rows;
        let pages = (0..end.div_ceil(64))
            .map(|page| slot * SLOT_PAGES + page)
            .collect();
        appends.push(BatchAppendV1::new(previous, rows, end, pages, None));
        slots.push(slot);
    }
    let original = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap();
    let envelope = encode(&original, &boundaries).ok();
    (original, envelope)
}

/// The negotiated prefill envelope of `batch`.
fn envelope(chunks: &[Chunk], decodes: &[(u64, u64, u64)], verify: bool) -> Vec<u8> {
    batch(chunks, decodes, verify)
        .1
        .expect("chunks on the 8192-token grid")
}

/// A TP4 host with room for three long-context slots.
fn host(dspark: bool) -> Rc<Host> {
    let host = FakeHost::new(Shape::new(&FLASH, Ranks::TP4).unwrap(), KV_LAYOUTS[0]);
    host.drafter.set(dspark);
    if dspark {
        host.speculation.set(Some((5, 32)));
    }
    host.kv_pages.set(3 * SLOT_PAGES);
    host
}

/// An upper-skip instance on the 8192-token grid keeping at most
/// `templates` templates (none at zero), lowering each reused one again
/// when `verify`.
fn create(host: &Rc<Host>, templates: usize, verify: bool) -> BundleInstanceV1 {
    super::prefill_skip::create_profile_with_options(
        host,
        8192,
        true,
        &[
            (PREFILL_PLAN_TEMPLATES, templates.into()),
            (PREFILL_PLAN_CACHE_VERIFY, verify.into()),
        ],
    )
}

fn counts(instance: &BundleInstanceV1) -> Counts {
    counts_of(instance).expect("an instance keeping templates")
}

fn counts_of(instance: &BundleInstanceV1) -> Option<Counts> {
    plan_templates::counts(instance.handle())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Outcome {
    /// The key's template, patched.
    Reused,
    /// No template of the key: a full lowering, kept.
    Kept,
    /// The key's template refused its patch: a full lowering, kept instead.
    Refused,
    /// Not a template's run: a full lowering, not kept.
    Lowered,
}

/// Plans `request` outside a step and says what the templates did.
fn outcome(instance: &BundleInstanceV1, request: &[u8]) -> Outcome {
    let before = counts(instance);
    plan_timing::plan(instance.handle(), request).unwrap_or_else(|e| panic!("{e}"));
    let after = counts(instance);
    let delta = (
        after.hits - before.hits,
        after.misses - before.misses,
        after.refused - before.refused,
        after.kept - before.kept,
        after.busy - before.busy,
    );
    match delta {
        (1, 0, 0, 0, 0) => Outcome::Reused,
        (0, 1, 0, 1, 0) => Outcome::Kept,
        (0, 0, 1, 1, 0) => Outcome::Refused,
        (0, 0, 0, 0, 0) => Outcome::Lowered,
        other => panic!("unexpected template counts {other:?}"),
    }
}

/// Every input of a long-context plan, changed one at a time: the step's
/// own values (tokens, request, slot, pages, positions) patch the template
/// of its key; every structural input (rows, scored rows, items, the
/// cached-attention range, the DMA, and with DSpark the prefix modulo the
/// ring that sets the retained feature rows) makes another key. Positions
/// whose index selections take other batches refuse the patch and keep a
/// second template of the key, which later steps of either context class
/// reuse. The verify mode lowers every reused template again and compares
/// the whole plan.
#[test]
fn templates_patch_their_step_values_and_key_their_structure() {
    use Outcome::*;
    for dspark in [false, true] {
        let host = host(dspark);
        let mut instance = create(&host, 64, true);
        let plain = |c: Chunk| batch(&[c], &[], false).0;
        let base = chunk(1, 0, 200_000, 512);
        // With DSpark, a prefix one past the base retains another feature
        // range; the cone of an AR plan does not follow the prefix.
        let shifted = if dspark { Kept } else { Reused };
        let cases: Vec<(&str, Vec<u8>, Outcome)> = vec![
            ("base", plain(base), Kept),
            ("again", plain(base), Reused),
            ("tokens", plain(Chunk { salt: 3, ..base }), Reused),
            (
                "request, slot and pages",
                plain(Chunk {
                    request: 2,
                    slot: 1,
                    ..base
                }),
                Reused,
            ),
            (
                "prefix by the ring",
                plain(Chunk {
                    prefix: 200_128,
                    ..base
                }),
                Reused,
            ),
            (
                "prefix by one",
                plain(Chunk {
                    prefix: 200_001,
                    ..base
                }),
                shifted,
            ),
            (
                "half the context",
                plain(Chunk {
                    prefix: 100_032,
                    ..base
                }),
                Refused,
            ),
            (
                "half the context by the ring",
                plain(Chunk {
                    prefix: 100_160,
                    ..base
                }),
                Reused,
            ),
            ("rows", plain(Chunk { rows: 511, ..base }), Kept),
            (
                "unscored",
                plain(Chunk {
                    scored: false,
                    ..base
                }),
                Kept,
            ),
            (
                "two items",
                batch(&[base, chunk(2, 1, 0, 64)], &[], false).0,
                Kept,
            ),
            ("cached attention", plain(chunk(1, 0, 8192, 8192)), Kept),
            (
                "cached attention at another offset",
                plain(chunk(1, 0, 16384, 8192)),
                Kept,
            ),
            (
                "cached attention again",
                plain(chunk(2, 0, 8192, 8192)),
                Reused,
            ),
            // The cached range starts at its first physical page.
            (
                "cached attention on other pages",
                plain(chunk(3, 1, 8192, 8192)),
                Kept,
            ),
            (
                "past the cached range",
                plain(chunk(1, 0, 40960, 8192)),
                Kept,
            ),
            (
                "past the cached range again",
                plain(chunk(1, 0, 49152, 8192)),
                Reused,
            ),
            ("base after the others", plain(base), Reused),
        ];
        for (name, request, expected) in &cases {
            assert_eq!(
                outcome(&instance, request),
                *expected,
                "dspark={dspark} {name}"
            );
        }
        // The DMA is part of the key: without it the base takes another
        // template, and with it the base reuses its own.
        let without =
            crate::instance::execute::skip::with_dma(false, || outcome(&instance, &plain(base)));
        assert_eq!(without, Kept, "dspark={dspark} without DMA");
        assert_eq!(outcome(&instance, &plain(base)), Reused, "dspark={dspark}");
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Plans that keep anchors are never kept, and an instance without the
/// candidate keeps no templates.
#[test]
fn anchored_plans_and_default_instances_keep_no_templates() {
    let host = host(true);
    let instance = super::prefill_skip::create_profile_with_options(
        &host,
        8192,
        true,
        &[
            (PREFILL_PLAN_TEMPLATES, 4.into()),
            (
                model_deepseek41_contract::execution_options::PREFILL_ANCHOR_ITEMS,
                2.into(),
            ),
        ],
    );
    let anchored = |request, slot| {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::new(request, false, 789, true, (0..789).collect()).with_anchor(true)],
            vec![BatchAppendV1::new(
                0,
                789,
                789,
                (0..13).map(|page| slot * SLOT_PAGES + page).collect(),
                None,
            )],
        )
        .with_state_slots(vec![slot])
        .encode()
        .unwrap()
    };
    for (request, slot) in [(1, 0), (2, 1)] {
        assert_eq!(
            outcome(&instance, &anchored(request, slot)),
            Outcome::Lowered
        );
    }
    let mut instance = instance;
    instance.destroy().unwrap();
    let mut default = super::prefill_skip::create_profile(&host, 8192, true);
    assert_eq!(plan_templates::counts(default.handle()), None);
    default.destroy().unwrap();
}

/// Without its index selections lowered again, a reused template issues
/// the base's selection widths at a longer context: the verify mode
/// refuses it. The patch is what keeps the kept plan exact.
#[test]
fn verify_refuses_a_template_whose_selections_were_not_patched() {
    let host = host(false);
    let mut instance = create(&host, 4, true);
    let request = |prefix| batch(&[chunk(1, 0, prefix, 512)], &[], false).0;
    assert_eq!(outcome(&instance, &request(200_000)), Outcome::Kept);
    let refused = crate::instance::execute::skip::without_patches(|| {
        plan_timing::plan(instance.handle(), &request(200_128))
    });
    assert!(refused.is_err(), "{refused:?}");
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|l| l.contains("differs from a fresh lowering"))
    );
    instance.destroy().unwrap();
}

/// What a step does after it executes.
#[derive(Clone, Copy)]
enum Then {
    Commit,
    /// Commit a verify step keeping one row of this verify item.
    Accept(u64),
}

/// A long-context lifecycle through the API, from restored prefixes: the
/// served 512-row prompt ends at 100K and 200K, alone and beside decode or
/// DSpark verify items, grid chunks past the cached range and, without
/// DSpark, a step that fails once planned and before its launches. Returns
/// the device journal and the template counts, if any.
fn lifecycle(dspark: bool, templates: usize, verify: bool) -> (Vec<String>, Option<Counts>) {
    let host = host(dspark);
    let mut instance = create(&host, templates, verify);
    let restore = |instance: &BundleInstanceV1, request: u64, slot: u64, tokens: u64| {
        let ids: Vec<_> = (0..tokens).map(|p| token(0, p)).collect();
        plan_timing::restore_without_snapshot(instance.handle(), request, slot as u32, &ids)
            .unwrap_or_else(|e| panic!("restore: {e}"));
    };
    let run = |instance: &mut BundleInstanceV1, request: &[u8], then: Then, retire: &[u64]| {
        let mut sink = Sink::default();
        instance
            .execute_prefill(request, &mut sink)
            .unwrap_or_else(|e| {
                let journal = host.device.journal.borrow();
                panic!(
                    "{e:?}: {:?} {:?}",
                    host.logs.borrow().last(),
                    &journal[journal.len().saturating_sub(6)..]
                )
            });
        match then {
            Then::Commit => instance.commit().unwrap(),
            Then::Accept(request_id) => instance
                .commit_accepted(&[BundleKeptRowsV1 {
                    struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
                    reserved: 0,
                    request_id,
                    kept_rows: 1,
                    reserved_tail: 0,
                }])
                .unwrap(),
        }
        for &request in retire {
            instance.retire_request(request).unwrap();
        }
    };
    let decode_then = |request| {
        if dspark {
            Then::Accept(request)
        } else {
            Then::Commit
        }
    };
    restore(&instance, 1, 0, 200_000);
    run(
        &mut instance,
        &envelope(&[chunk(1, 0, 200_000, 512)], &[], false),
        Then::Commit,
        &[],
    );
    restore(&instance, 2, 1, 200_128);
    run(
        &mut instance,
        &envelope(&[chunk(2, 1, 200_128, 512)], &[(1, 0, 200_512)], dspark),
        decode_then(1),
        &[1],
    );
    restore(&instance, 3, 0, 100_032);
    run(
        &mut instance,
        &envelope(&[chunk(3, 0, 100_032, 512)], &[], false),
        Then::Commit,
        &[2],
    );
    restore(&instance, 4, 1, 100_160);
    let next = envelope(&[chunk(4, 1, 100_160, 512)], &[(3, 0, 100_544)], dspark);
    if dspark {
        run(&mut instance, &next, decode_then(3), &[3, 4]);
    } else {
        // The step fails at its first upload, once it is planned and before
        // any launch: nothing is poisoned. The host fails its requests, and
        // a later step of the same structure reuses the template the failed
        // step patched. (A DSpark step uploads its drafter inputs first, and
        // a failed DSpark step is terminal, as `a_failed_launch_...` shows.)
        let planned = counts_of(&instance);
        host.fail_upload.set(Some(0));
        assert!(
            instance
                .execute_prefill(&next, &mut Sink::default())
                .is_err()
        );
        host.fail_upload.set(None);
        if let (Some(before), Some(after)) = (planned, counts_of(&instance)) {
            assert_eq!(
                after.hits,
                before.hits + 1,
                "the failed step reused its template"
            );
        }
        for request in [3, 4] {
            instance.retire_request(request).unwrap();
        }
    }
    restore(&instance, 6, 1, 100_288);
    run(
        &mut instance,
        &envelope(&[chunk(6, 1, 100_288, 512)], &[], false),
        Then::Commit,
        &[6],
    );
    // Grid chunks of a longer prompt, past the cached range and within one
    // class of index batches (40960 to 65535 positions), repeat their
    // structure.
    restore(&instance, 5, 2, 40960);
    for prefix in [40960, 49152, 57344] {
        let mut c = chunk(5, 2, prefix, 8192);
        c.scored = false;
        run(
            &mut instance,
            &envelope(&[c], &[], false),
            Then::Commit,
            &[],
        );
    }
    instance.retire_request(5).unwrap();
    let counts = counts_of(&instance);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    (journal, counts)
}

fn assert_same_journal(name: &str, actual: &[String], expected: &[String]) {
    if let Some(at) = (0..actual.len().min(expected.len())).find(|&at| actual[at] != expected[at]) {
        panic!(
            "{name} journal line {at}: {} against {}",
            actual[at], expected[at]
        );
    }
    assert_eq!(actual.len(), expected.len(), "{name} journal length");
}

#[test]
fn long_context_templates_issue_the_fresh_lowerings_journal() {
    for dspark in [false, true] {
        let (fresh, none) = lifecycle(dspark, 0, false);
        assert_eq!(none, None);
        assert!(fresh.iter().filter(|l| l.starts_with("launch ")).count() > 20_000);
        for verify in [false, true] {
            let (kept, counts) = lifecycle(dspark, 16, verify);
            let name = format!("dspark={dspark} verify={verify}");
            assert_same_journal(&name, &kept, &fresh);
            // The 200K end (kept), its 128-token shift (reused), the 100K end
            // (refused: other index batches, so kept beside it), its shifts
            // by 128 (failed, without DSpark) and 256, and the grid chunks
            // (kept, then reused twice).
            assert_eq!(
                counts,
                Some(Counts {
                    hits: 5,
                    misses: 2,
                    busy: 0,
                    refused: 1,
                    kept: 3,
                    evicted: 0,
                }),
                "{name}"
            );
        }
    }
}

/// A launch failure inside a reused template poisons the instance exactly
/// as it does in a full lowering.
#[test]
fn a_failed_launch_of_a_reused_template_poisons_the_instance() {
    for dspark in [false, true] {
        let mut journals = Vec::new();
        for templates in [0, 4] {
            let host = host(dspark);
            let mut instance = create(&host, templates, false);
            for (request, slot, prefix) in [(1, 0, 200_000), (2, 1, 200_128)] {
                let ids: Vec<_> = (0..prefix).map(|p| token(0, p)).collect();
                plan_timing::restore_without_snapshot(instance.handle(), request, slot, &ids)
                    .unwrap();
                let step = envelope(&[chunk(request, u64::from(slot), prefix, 512)], &[], false);
                if request == 2 {
                    // Partway through the second step's body.
                    host.device.fail_launch.set(Some(1500));
                    assert!(
                        instance
                            .execute_prefill(&step, &mut Sink::default())
                            .is_err()
                    );
                    continue;
                }
                instance
                    .execute_prefill(&step, &mut Sink::default())
                    .unwrap();
                instance.commit().unwrap();
            }
            if templates > 0 {
                // The failed step reused the first step's template.
                assert_eq!(counts(&instance).hits, 1);
            }
            assert_eq!(
                instance.execute_prefill(
                    &envelope(&[chunk(3, 2, 0, 512)], &[], false),
                    &mut Sink::default()
                ),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
            );
            assert_eq!(
                instance.destroy(),
                Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
            );
            journals.push(host.device.journal.borrow().clone());
        }
        assert_same_journal(&format!("dspark={dspark}"), &journals[1], &journals[0]);
    }
}

/// Host time of the planning scope at the served long-context shape, full
/// and from a template, in a release build with
/// `CHOPINFER_PLAN_TEMPLATE_BENCH=1`: `cargo test --release -p
/// bundle-deepseek41 --lib prefill_plan_templates::planning_time --
/// --nocapture`. Other runs plan each shape once and check that the
/// template serves the repeats.
#[test]
fn planning_time() {
    let timed = std::env::var("CHOPINFER_PLAN_TEMPLATE_BENCH").is_ok_and(|v| v == "1");
    let repeats = if timed { 41 } else { 1 };
    for dspark in [false, true] {
        for prefix in [100_032_u64, 200_000] {
            let request = batch(&[chunk(1, 0, prefix, 512)], &[], false).0;
            let mut times = Vec::new();
            for templates in [0, 16] {
                let host = host(dspark);
                let mut instance = create(&host, templates, false);
                plan_timing::plan(instance.handle(), &request).unwrap();
                times.push(plan_timing::time(instance.handle(), &request, repeats));
                if templates > 0 {
                    assert_eq!(counts(&instance).hits, 2 * repeats as u64);
                }
                instance.destroy().unwrap();
            }
            if !timed {
                continue;
            }
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            let launches = {
                let host = host(dspark);
                let mut instance = create(&host, 0, false);
                let launches = plan_timing::plan(instance.handle(), &request).unwrap().1;
                instance.destroy().unwrap();
                launches
            };
            println!(
                "dspark={dspark} prefix={prefix} launches={launches:?}: planning {:.3} ms in \
                 full (upper-skip lowering {:.3} ms), {:.3} ms from a template",
                ms(times[0].0),
                ms(times[0].1),
                ms(times[1].0),
            );
        }
    }
}
