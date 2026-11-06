//! Upper-skip plan templates (`plan_templates`): a reused template
//! issues exactly what a fresh lowering issues, its key holds every input
//! but the positions, and a template whose index selections no longer
//! match falls back to a full lowering.
use super::*;
use crate::instance::execute::{plan_templates, plan_timing};
use bundle_runtime::fake_host::RecordingSink;
use bundle_runtime::template::Counts;
use model_deepseek41_contract::execution_options::PREFILL_PLAN_CACHE_VERIFY;
use runtime_bundle_sdk::prefill::{Boundary, encode};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

mod failures;
mod groups;
mod metadata;

type Sink = RecordingSink<super::conformance::Limits>;

/// Pages a slot's request may use: slot `s` takes pages from `s * SLOT_PAGES`.
const SLOT_PAGES: u64 = 4096;

/// The prompt an unscored chunk belongs to, ending past every such chunk.
const UNSCORED_PROMPT: u64 = 81920;

/// A prompt chunk of `rows` rows after `prefix` tokens of `request` in
/// `slot`, its pages numbered from `pages`, scoring its last row when
/// `scored`, with token IDs drawn from `salt`.
#[derive(Clone, Copy)]
struct Chunk {
    request: u64,
    slot: u64,
    pages: u64,
    prefix: u64,
    rows: u64,
    scored: bool,
    salt: u64,
}

/// A chunk on the pages of its slot.
fn chunk(request: u64, slot: u64, prefix: u64, rows: u64) -> Chunk {
    Chunk {
        request,
        slot,
        pages: slot * SLOT_PAGES,
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
        let pages = (0..end.div_ceil(64)).map(|page| c.pages + page).collect();
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

/// An upper-skip instance on the 8192-token grid, lowering each reused
/// template again when `verify`.
fn create(host: &Rc<Host>, verify: bool) -> BundleInstanceV1 {
    super::prefill_skip::create_profile_with_options(
        host,
        8192,
        true,
        &[(PREFILL_PLAN_CACHE_VERIFY, verify.into())],
    )
}

fn counts(instance: &BundleInstanceV1) -> Counts {
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

/// Plans `request` outside a step and says what the templates did. A
/// reused template reuses its Engram plan too, made at its first step.
fn outcome(instance: &BundleInstanceV1, request: &[u8]) -> Outcome {
    let before = counts(instance);
    let planned = plan_timing::plan(instance.handle(), request).unwrap_or_else(|e| panic!("{e}"));
    let after = counts(instance);
    let delta = (
        after.hits - before.hits,
        after.misses - before.misses,
        after.refused - before.refused,
        after.kept - before.kept,
        after.busy - before.busy,
    );
    let outcome = match delta {
        (1, 0, 0, 0, 0) => Outcome::Reused,
        (0, 1, 0, 1, 0) => Outcome::Kept,
        (0, 0, 1, 1, 0) => Outcome::Refused,
        (0, 0, 0, 0, 0) => Outcome::Lowered,
        other => panic!("unexpected template counts {other:?}"),
    };
    if let Some(reused) = planned.engram {
        assert_eq!(reused, outcome == Outcome::Reused, "{outcome:?}");
    }
    outcome
}

/// Every input of a long-context plan, changed one at a time: the step's
/// own values (tokens, request, slot, pages, positions) patch the template
/// of its key, for one item and for packed items whose prefixes shift
/// apart; every structural input (rows, scored rows, the items and their
/// boundaries at the same total rows, the cached-attention range, the DMA,
/// and with DSpark the prefix modulo the ring that sets the retained
/// feature rows) makes another key. Positions whose index selections take
/// other batches refuse the patch and keep a second template of the key,
/// which later steps of either context class reuse. The verify mode lowers
/// every reused template again and compares the whole plan.
#[test]
fn templates_patch_their_step_values_and_key_their_structure() {
    use Outcome::*;
    for dspark in [false, true] {
        let host = host(dspark);
        let mut instance = create(&host, true);
        let plain = |c: Chunk| batch(&[c], &[], false).0;
        let packed = |a: Chunk, b: Chunk| batch(&[a, b], &[], false).0;
        let base = chunk(1, 0, 200_000, 512);
        let second = chunk(2, 1, 8192, 64);
        let by_ring = |c: Chunk| Chunk {
            prefix: c.prefix + 128,
            ..c
        };
        // The same total rows with the boundary between the items moved.
        let (moved, moved_second) = (
            Chunk { rows: 448, ..base },
            Chunk {
                rows: 128,
                ..second
            },
        );
        // With DSpark, a prefix one past the base retains another feature
        // range; the cone of an AR plan does not follow the prefix.
        let shifted = if dspark { Kept } else { Reused };
        let cases: Vec<(&str, Vec<u8>, Outcome)> = vec![
            ("base", plain(base), Kept),
            ("again", plain(base), Reused),
            ("tokens", plain(Chunk { salt: 3, ..base }), Reused),
            ("request", plain(Chunk { request: 2, ..base }), Reused),
            ("slot", plain(Chunk { slot: 1, ..base }), Reused),
            (
                "pages",
                plain(Chunk {
                    pages: 2 * SLOT_PAGES,
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
            ("two items", packed(base, second), Kept),
            (
                "two items, the first shifted",
                packed(by_ring(base), second),
                Reused,
            ),
            (
                "two items, the second shifted",
                packed(base, by_ring(second)),
                Reused,
            ),
            (
                "two items, other requests, slots and pages",
                packed(
                    Chunk {
                        request: 3,
                        slot: 2,
                        pages: 2 * SLOT_PAGES,
                        ..base
                    },
                    Chunk {
                        request: 4,
                        slot: 0,
                        pages: 0,
                        salt: 5,
                        ..second
                    },
                ),
                Reused,
            ),
            (
                "two items, the boundary moved",
                packed(moved, moved_second),
                Kept,
            ),
            (
                "two items, the boundary moved, both shifted",
                packed(by_ring(moved), by_ring(moved_second)),
                Reused,
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
        // A DSpark prefill takes prepared Engram gathers (an AR one hashes
        // its rows apart), so its reused templates reuse their Engram plans
        // (`outcome`).
        let engram = plan_timing::plan(instance.handle(), &plain(base))
            .unwrap()
            .engram;
        assert_eq!(engram, dspark.then_some(true), "dspark={dspark}");
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

/// Plans that keep anchors share a template across requests and slots,
/// and an instance keeps up to 64 templates.
#[test]
fn anchored_plans_share_templates_and_instances_keep_64() {
    let host = host(true);
    let instance = super::prefill_skip::create_profile(&host, 8192, true);
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
    for (request, slot, expected) in [(1, 0, Outcome::Kept), (2, 1, Outcome::Reused)] {
        assert_eq!(outcome(&instance, &anchored(request, slot)), expected);
    }
    let mut instance = instance;
    instance.destroy().unwrap();
    let mut default = super::prefill_skip::create_profile(&host, 8192, true);
    let request = batch(&[chunk(1, 0, 200_000, 512)], &[], false).0;
    assert_eq!(outcome(&default, &request), Outcome::Kept);
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|l| l.contains("; 1 of 64 templates,"))
    );
    default.destroy().unwrap();
}

/// Without its index selections lowered again, a reused template issues
/// the base's selection widths at a longer context: the verify mode
/// refuses it. The patch is what keeps the kept plan exact.
#[test]
fn verify_refuses_a_template_whose_selections_were_not_patched() {
    let host = host(false);
    let mut instance = create(&host, true);
    let request = |prefix| batch(&[chunk(1, 0, prefix, 512)], &[], false).0;
    assert_eq!(outcome(&instance, &request(200_000)), Outcome::Kept);
    let refused = crate::instance::execute::skip::without_patches(|| {
        plan_timing::plan(instance.handle(), &request(200_128))
    });
    assert!(refused.is_err());
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|l| l.contains("differs from a fresh lowering"))
    );
    instance.destroy().unwrap();
}

/// A reused Engram plan that differs from a fresh one, here a stale kept
/// plan whose first DMA join names another layer, passes the reuse audit,
/// which checks only the patched launches and the collectives. The verify
/// mode makes the plan again and refuses the step.
#[test]
fn verify_refuses_a_reused_engram_plan_that_differs_from_a_fresh_one() {
    for verify in [false, true] {
        let host = host(true);
        let mut instance = create(&host, verify);
        let request = |prefix| batch(&[chunk(1, 0, prefix, 512)], &[], false).0;
        assert_eq!(outcome(&instance, &request(200_000)), Outcome::Kept);
        let planned = crate::instance::execute::skip::with_stale_engram(|| {
            plan_timing::plan(instance.handle(), &request(200_128))
        });
        let refused = host
            .logs
            .borrow()
            .iter()
            .any(|l| l.contains("a reused prefill Engram plan differs from a fresh one"));
        if verify {
            assert!(planned.is_err() && refused, "verify={verify}");
        } else {
            let planned = planned.unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(planned.engram, Some(true));
            assert!(!refused);
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
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
/// the device journal and the template counts.
fn lifecycle(dspark: bool, verify: bool) -> (Vec<String>, Counts) {
    let host = host(dspark);
    let mut instance = create(&host, verify);
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
        let planned = counts(&instance);
        host.fail_upload.set(Some(0));
        assert!(
            instance
                .execute_prefill(&next, &mut Sink::default())
                .is_err()
        );
        host.fail_upload.set(None);
        assert_eq!(
            counts(&instance).hits,
            planned.hits + 1,
            "the failed step reused its template"
        );
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
    let counts = counts(&instance);
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let journal = host.device.journal.borrow().clone();
    (journal, counts)
}

/// The long-context lifecycle issues, with templates and with each reused
/// template lowered again, the journal pinned from fresh lowerings before
/// templates were kept by default. Both verification modes are captured by
/// the journal proof, with identical bodies sharing the same pin.
#[test]
fn long_context_templates_issue_the_fresh_lowerings_journal() {
    let mut mismatches = Vec::new();
    for dspark in [false, true] {
        let (name, expected) = if dspark {
            (
                "flash-tp4-dspark-long-context-templates",
                (195050, 0x69f33940baa393f1),
            )
        } else {
            (
                "flash-tp4-ar-long-context-templates",
                (166669, 0xbe1cbe2e5b85e219),
            )
        };
        for verify in [false, true] {
            let (journal, counts) = lifecycle(dspark, verify);
            assert!(journal.iter().filter(|l| l.starts_with("launch ")).count() > 20_000);
            // The 200K end (kept), its 128-token shift (reused), the 100K end
            // (refused: other index batches, so kept beside it), its shifts
            // by 128 (failed, without DSpark) and 256, and the grid chunks
            // (kept, then reused twice).
            assert_eq!(
                counts,
                Counts {
                    hits: 5,
                    misses: 2,
                    busy: 0,
                    refused: 1,
                    kept: 3,
                    evicted: 0,
                },
                "dspark={dspark} verify={verify}"
            );
            if let Err(error) = bundle_runtime::fake_host::check_journal(
                &journal,
                expected,
                &format!("{name}-{verify}.journal"),
            ) {
                mismatches.push(format!("{name} verify={verify}: {error}"));
            }
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// A launch failure inside a reused template poisons the instance exactly
/// as it does in a full lowering: the journal is the one pinned from a
/// fresh lowering before templates were kept by default, also covered by
/// the sealed journal proof.
#[test]
fn a_failed_launch_of_a_reused_template_poisons_the_instance() {
    let mut mismatches = Vec::new();
    for dspark in [false, true] {
        let host = host(dspark);
        let mut instance = create(&host, false);
        for (request, slot, prefix) in [(1, 0, 200_000), (2, 1, 200_128)] {
            let ids: Vec<_> = (0..prefix).map(|p| token(0, p)).collect();
            plan_timing::restore_without_snapshot(instance.handle(), request, slot, &ids).unwrap();
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
        // The failed step reused the first step's template.
        assert_eq!(counts(&instance).hits, 1);
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
        let (name, expected) = if dspark {
            (
                "flash-tp4-dspark-failed-template",
                (33405, 0xc3fb2109691815c9),
            )
        } else {
            ("flash-tp4-ar-failed-template", (31956, 0xa1bc5a570fd28ff9))
        };
        if let Err(error) = bundle_runtime::fake_host::check_journal(
            &host.device.journal.borrow(),
            expected,
            &format!("{name}.journal"),
        ) {
            mismatches.push(format!("{name}: {error}"));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// Host time of the planning scope at the served long-context shape from a
/// template, and of the upper-skip lowering in full, in a release build
/// with `CHOPINFER_PLAN_TEMPLATE_BENCH=1`: `cargo test --release -p
/// bundle-deepseek41 --lib prefill_plan_templates::planning_time --
/// --nocapture`. Other runs plan each shape once and check that the
/// template serves the repeats.
#[test]
fn planning_time() {
    let timed = std::env::var("CHOPINFER_PLAN_TEMPLATE_BENCH").is_ok_and(|v| v == "1");
    let repeats = if timed { 41 } else { 1 };
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
    for dspark in [false, true] {
        for prefix in [100_032_u64, 200_000] {
            let request = batch(&[chunk(1, 0, prefix, 512)], &[], false).0;
            let host = host(dspark);
            let mut instance = create(&host, false);
            let launches = plan_timing::plan(instance.handle(), &request)
                .unwrap()
                .launches;
            let (planned, lowered) = plan_timing::time(instance.handle(), &request, repeats);
            assert_eq!(counts(&instance).hits, 2 * repeats as u64);
            instance.destroy().unwrap();
            if timed {
                println!(
                    "dspark={dspark} prefix={prefix} launches={launches:?}: upper-skip lowering \
                     {:.3} ms in full, planning {:.3} ms from a template",
                    ms(lowered),
                    ms(planned),
                );
            }
        }
    }
}

/// Bytes of live allocations: glibc's in-use heap and mapped chunks.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn allocated() -> i64 {
    /// glibc's `struct mallinfo2`, ten `size_t` counters.
    #[repr(C)]
    struct Mallinfo2([usize; 10]);
    // SAFETY-REVIEWED: matches glibc's declaration: no arguments, and ten size_t
    // fields where index 4 is hblkhd and index 7 is uordblks.
    #[allow(unsafe_code)]
    unsafe extern "C" {
        fn mallinfo2() -> Mallinfo2;
    }
    // SAFETY-REVIEWED: mallinfo2 takes no arguments and returns
    // allocator-owned counters by value.
    #[allow(unsafe_code)]
    let Mallinfo2(counts) = unsafe { mallinfo2() };
    // hblkhd and uordblks.
    (counts[4] + counts[7]) as i64
}

/// Host memory a full cache of templates holds at the served long-context
/// shapes, in a release build with `CHOPINFER_PLAN_TEMPLATE_BENCH=1`:
/// `cargo test --release -p bundle-deepseek41 --lib
/// prefill_plan_templates::template_memory -- --nocapture`. A shape's first
/// key is one chunk after the context; the others move rows to a second,
/// short-context item, each boundary its own key. The templates after the
/// first hold the live bytes after planning every key less those after the
/// first, whose lowering also makes the instance's lasting state. Other runs
/// plan two keys of one shape and check that each kept its own template.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn template_memory() {
    let measured = std::env::var("CHOPINFER_PLAN_TEMPLATE_BENCH").is_ok_and(|v| v == "1");
    let keys: u64 = if measured {
        plan_templates::LIMIT as u64
    } else {
        2
    };
    let shapes: &[(bool, u64, u64)] = if measured {
        &[
            (false, 100_032, 512),
            (false, 200_000, 512),
            (true, 100_032, 512),
            (true, 200_000, 512),
            (false, 200_000, 8192),
            (true, 200_000, 8192),
        ]
    } else {
        &[(false, 200_000, 512)]
    };
    for &(dspark, prefix, rows) in shapes {
        let moved = rows / plan_templates::LIMIT as u64;
        let requests: Vec<_> = (0..keys)
            .map(|at| {
                let first = chunk(1, 0, prefix, rows - at * moved);
                let chunks = if at == 0 {
                    vec![first]
                } else {
                    vec![first, chunk(2, 1, 8192, at * moved)]
                };
                batch(&chunks, &[], false).0
            })
            .collect();
        let host = host(dspark);
        let mut instance = create(&host, false);
        let plan = |request: &[u8]| {
            plan_timing::plan(instance.handle(), request).unwrap_or_else(|e| panic!("{e}"));
        };
        plan(&requests[0]);
        let first = allocated();
        for request in &requests[1..] {
            plan(request);
        }
        let held = allocated() - first;
        let counts = counts(&instance);
        assert_eq!(
            (counts.misses, counts.kept, counts.evicted),
            (keys, keys, 0)
        );
        instance.destroy().unwrap();
        if measured {
            let mib = |bytes: f64| bytes / f64::from(1 << 20);
            let each = held as f64 / (keys - 1) as f64;
            println!(
                "dspark={dspark} prefix={prefix} rows={rows}: {} templates hold {:.1} MiB, \
                 {:.1} MiB each, so {keys} hold about {:.1} MiB",
                keys - 1,
                mib(held as f64),
                mib(each),
                mib(each * keys as f64),
            );
        }
    }
}
