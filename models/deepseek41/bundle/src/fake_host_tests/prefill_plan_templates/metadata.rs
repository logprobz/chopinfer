//! Metadata pins of fresh upper-skip lowerings (`plan_timing::describe`):
//! every body and head launch with its collective flag and stage, the
//! index-selection sites, the DMA plan, the row copies and the Engram plan.
//! The shapes are single items across the route boundaries of the prefill
//! envelope, packed, anchored and unscored items, AR with device and
//! host-mapped Engram tables, and DSpark, each with DMA on and off. The
//! pins were recorded before the lowering became incremental and hold it
//! to the plans it made before. Each shape also runs as a step, which
//! lowers its plan while it issues; that plan holds the same pin.
use super::*;
use crate::instance::execute::skip;
use model_deepseek41_contract::execution_options::PREFILL_PLAN_CACHE_VERIFY;

/// The committed tokens before each single item.
const PREFIX: u64 = 100_032;

/// Single-item row counts: each side of the class thresholds, the
/// remainder classes, the exact-row islands and the 4,096-row parts.
const ROWS: &[u64] = &[
    1, 32, 33, 48, 49, 64, 65, 112, 113, 127, 128, 192, 255, 256, 257, 319, 320, 448, 511, 512,
    513, 960, 1023, 1024, 1472, 1500, 2048, 2049, 2240, 2300, 2496, 2624, 2700, 3000, 4095, 4096,
    4097, 8191, 8192,
];

/// The original batch of a fresh anchored prompt of 789 tokens in slot 2.
fn anchored() -> Vec<u8> {
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(5, false, 789, true, (0..789).collect()).with_anchor(true)],
        vec![BatchAppendV1::new(
            0,
            789,
            789,
            (0..13).map(|page| 2 * SLOT_PAGES + page).collect(),
            None,
        )],
    )
    .with_state_slots(vec![2])
    .encode()
    .unwrap()
}

/// (name, original batch) of every pinned shape. Request 1 holds slot 0
/// at `PREFIX` tokens and request 2 slot 1 at 40,960.
fn shapes() -> Vec<(String, Vec<u8>)> {
    let mut shapes: Vec<_> = ROWS
        .iter()
        .map(|&rows| {
            let request = batch(&[chunk(1, 0, PREFIX, rows)], &[], false).0;
            (format!("r{rows}"), request)
        })
        .collect();
    for rows in [512, 4097] {
        let request = batch(&[chunk(3, 2, 0, rows)], &[], false).0;
        shapes.push((format!("fresh-r{rows}"), request));
    }
    let mut grid = chunk(2, 1, 40_960, 8192);
    grid.scored = false;
    shapes.push(("unscored-r8192".into(), batch(&[grid], &[], false).0));
    let packed = [chunk(1, 0, PREFIX, 1024), chunk(2, 1, 40_960, 512)];
    shapes.push(("packed-1024-512".into(), batch(&packed, &[], false).0));
    let packed = [chunk(2, 1, 40_960, 100), chunk(3, 2, 0, 2000)];
    shapes.push(("packed-100-2000".into(), batch(&packed, &[], false).0));
    shapes.push(("anchored-r789".into(), anchored()));
    shapes
}

/// The grid start before `PREFIX`.
const GRID: u64 = 98_304;

/// A pinned shape as a served step (`served`).
struct ServedShape {
    name: String,
    original: Vec<u8>,
    envelope: Vec<u8>,
    /// Whether the envelope carries cuts, as an anchored item's does.
    cuts: bool,
    /// Request 1's committed tokens.
    prefix: u64,
    /// Whether the step's plan is the pinned one's.
    pinned: bool,
}

/// Every pinned shape as a step a scheduler may send, and the cached
/// attention ranges of `templates_patch_their_step_values_...`. A single
/// item that crosses the 8192-token grid after `PREFIX` (8,191 and 8,192
/// rows) has no envelope there: it is served after `GRID` instead. Neither
/// those nor the cached ranges are pinned.
fn served() -> Vec<ServedShape> {
    let cached = [
        ("cached-r8192-after-8192", chunk(1, 0, 8192, 8192)),
        ("cached-r8192-after-16384", chunk(1, 0, 16_384, 8192)),
        (
            "cached-r8192-after-8192-other-pages",
            Chunk {
                pages: 2 * SLOT_PAGES,
                ..chunk(1, 0, 8192, 8192)
            },
        ),
    ];
    let cached = cached.into_iter().map(|(name, c)| {
        let (original, envelope) = batch(&[c], &[], false);
        ServedShape {
            name: name.into(),
            original,
            envelope: envelope.expect("a chunk on the grid"),
            cuts: false,
            prefix: c.prefix,
            pinned: false,
        }
    });
    shapes()
        .into_iter()
        .map(|(name, original)| {
            if name == "anchored-r789" {
                let boundary = Boundary {
                    request_id: 5,
                    prompt_tokens: 789,
                    chunk_end: 789,
                    chunk_tokens: 8192,
                };
                let envelope =
                    runtime_bundle_sdk::prefill::encode_with_cuts(&original, &[boundary]).unwrap();
                return ServedShape {
                    name,
                    original,
                    envelope,
                    cuts: true,
                    prefix: PREFIX,
                    pinned: true,
                };
            }
            let decoded = BatchRequestV1::decode(&original).unwrap();
            let chunks: Vec<_> = decoded
                .items()
                .iter()
                .zip(decoded.appends())
                .zip(decoded.state_slots().expect("state slots"))
                .map(|((item, append), &slot)| Chunk {
                    request: item.request_id(),
                    slot,
                    pages: slot * SLOT_PAGES,
                    prefix: append.previous_tokens(),
                    rows: u64::from(item.phase_tokens()),
                    scored: item.expects_logits(),
                    salt: 0,
                })
                .collect();
            if let (original, Some(envelope)) = batch(&chunks, &[], false) {
                return ServedShape {
                    name,
                    original,
                    envelope,
                    cuts: false,
                    prefix: PREFIX,
                    pinned: true,
                };
            }
            let chunks: Vec<_> = chunks
                .into_iter()
                .map(|c| Chunk { prefix: GRID, ..c })
                .collect();
            let (original, envelope) = batch(&chunks, &[], false);
            ServedShape {
                name: format!("{name}-after-{GRID}"),
                original,
                envelope: envelope.expect("a chunk from the grid start"),
                cuts: false,
                prefix: GRID,
                pinned: false,
            }
        })
        .chain(cached)
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Model {
    /// AR with its Engram tables in the device image.
    Ar,
    /// AR with host-mapped Engram tables: prepared gathers.
    ArMapped,
    DSpark,
}

/// Every shape's description on `model` against its pin, with DMA as
/// `dma`. Every mismatch is reported together: an unpinned shape as a
/// table entry, a differing one with its digest, the first with the file
/// of its lines.
fn check(model: Model, dma: bool, pins: &[(&str, (usize, u64))]) {
    let host = host(matches!(model, Model::DSpark));
    if matches!(model, Model::ArMapped) {
        host.drafter.set(true);
    }
    let mut instance = create(&host, false);
    for (request, slot, prefix) in [(1, 0, PREFIX), (2, 1, 40_960)] {
        let ids: Vec<_> = (0..prefix).map(|p| token(0, p)).collect();
        plan_timing::restore_without_snapshot(instance.handle(), request, slot, &ids).unwrap();
    }
    let mut mismatches = Vec::new();
    for (name, request) in shapes() {
        let lines = skip::with_dma(dma, || plan_timing::describe(instance.handle(), &request))
            .unwrap_or_else(|e| panic!("{model:?} dma={dma} {name}: {e}"));
        let (count, hash) = bundle_runtime::fake_host::journal_digest(&lines);
        match pins.iter().find(|(n, _)| *n == name) {
            Some(&(_, pin)) if pin == (count, hash) => {}
            // Only the first mismatch writes its lines (about 10 MB).
            Some(&(_, pin)) if mismatches.is_empty() => {
                let file = format!("prefill-plan-metadata-{model:?}-{dma}-{name}");
                let error = bundle_runtime::fake_host::check_journal(&lines, pin, &file);
                mismatches.push(format!("{name}: {}", error.unwrap_err()));
            }
            Some(_) => mismatches.push(format!("{name}: ({count}, {hash:#018x})")),
            None => mismatches.push(format!(
                "{name}: unpinned, (\"{name}\", ({count}, {hash:#018x})),"
            )),
        }
    }
    instance.destroy().unwrap();
    assert!(
        mismatches.is_empty(),
        "{model:?} dma={dma}:\n{}",
        mismatches.join("\n")
    );
}

/// A fresh instance on `model` with request 1 holding slot 0 at `prefix`
/// tokens and request 2 slot 1 at 40,960.
fn restored(model: Model, prefix: u64) -> BundleInstanceV1 {
    let host = host(matches!(model, Model::DSpark));
    if matches!(model, Model::ArMapped) {
        host.drafter.set(true);
    }
    let instance = super::super::prefill_skip::create_profile_with_options(
        &host,
        8192,
        true,
        &[(PREFILL_PLAN_CACHE_VERIFY, false.into())],
    );
    for (request, slot, prefix) in [(1, 0, prefix), (2, 1, 40_960)] {
        let ids: Vec<_> = (0..prefix).map(|p| token(0, p)).collect();
        plan_timing::restore_without_snapshot(instance.handle(), request, slot, &ids).unwrap();
    }
    instance
}

/// Every shape served as a step lowered while it issues on `model`, each
/// on a fresh instance, with DMA as `dma`: the streamed plan and its Engram
/// plan (`skip::describing_streamed`) hold the shape's pin, or, for a
/// shape no pin holds, describe as a fresh lowering does
/// (`plan_timing::describe`). Mismatches are reported together, as
/// `check` reports them.
fn check_streamed(model: Model, dma: bool, pins: &[(&str, (usize, u64))]) {
    let mut mismatches = Vec::new();
    for shape in served() {
        let name = &shape.name;
        let mut instance = restored(model, shape.prefix);
        let (executed, described) = skip::describing_streamed(|| {
            skip::with_dma(dma, || {
                if shape.cuts {
                    instance.execute_prefill_cuts(&shape.envelope, &mut Sink::default())
                } else {
                    instance.execute_prefill(&shape.envelope, &mut Sink::default())
                }
            })
        });
        executed.unwrap_or_else(|e| panic!("{model:?} dma={dma} {name}: {e:?}"));
        let _ = instance.cancel();
        let _ = instance.destroy();
        let [lines] = &described[..] else {
            panic!(
                "{model:?} dma={dma} {name}: {} streamed plans",
                described.len()
            );
        };
        let pin = if shape.pinned {
            pins.iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("{name} is pinned"))
                .1
        } else {
            let mut instance = restored(model, shape.prefix);
            let fresh = skip::with_dma(dma, || {
                plan_timing::describe(instance.handle(), &shape.original)
            })
            .unwrap_or_else(|e| panic!("{model:?} dma={dma} {name}: {e}"));
            instance.destroy().unwrap();
            bundle_runtime::fake_host::journal_digest(&fresh)
        };
        let (count, hash) = bundle_runtime::fake_host::journal_digest(lines);
        if pin == (count, hash) {
            continue;
        }
        if mismatches.is_empty() {
            let file = format!("prefill-plan-metadata-streamed-{model:?}-{dma}-{name}");
            let error = bundle_runtime::fake_host::check_journal(lines, pin, &file);
            mismatches.push(format!("{name}: {}", error.unwrap_err()));
        } else {
            mismatches.push(format!("{name}: ({count}, {hash:#018x})"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{model:?} dma={dma} streamed:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn streamed_ar_plans_hold_their_pins_with_dma() {
    check_streamed(Model::Ar, true, AR_DMA);
}

#[test]
fn streamed_ar_plans_hold_their_pins_without_dma() {
    check_streamed(Model::Ar, false, AR);
}

#[test]
fn streamed_mapped_ar_plans_hold_their_pins_with_dma() {
    check_streamed(Model::ArMapped, true, MAPPED_AR_DMA);
}

#[test]
fn streamed_mapped_ar_plans_hold_their_pins_without_dma() {
    check_streamed(Model::ArMapped, false, MAPPED_AR);
}

#[test]
fn streamed_dspark_plans_hold_their_pins_with_dma() {
    check_streamed(Model::DSpark, true, DSPARK_DMA);
}

#[test]
fn streamed_dspark_plans_hold_their_pins_without_dma() {
    check_streamed(Model::DSpark, false, DSPARK);
}

/// Two plans are the same only at the same sites and, where both are
/// templates that made one, with the same Engram plan: a template's sites
/// and Engram plan are what its later steps reuse.
#[test]
fn the_same_plan_has_the_same_sites_and_kept_engram_plan() {
    let host = host(true);
    let mut instance = create(&host, false);
    let ids: Vec<_> = (0..PREFIX).map(|p| token(0, p)).collect();
    plan_timing::restore_without_snapshot(instance.handle(), 1, 0, &ids).unwrap();
    let request = batch(&[chunk(1, 0, PREFIX, 512)], &[], false).0;
    let same = skip::with_dma(true, || plan_timing::same(instance.handle(), &request)).unwrap();
    assert_eq!(same, [true, false, true, false]);
    instance.destroy().unwrap();
}

#[test]
fn ar_plans_hold_their_pins_with_dma() {
    check(Model::Ar, true, AR_DMA);
}

#[test]
fn ar_plans_hold_their_pins_without_dma() {
    check(Model::Ar, false, AR);
}

#[test]
fn mapped_ar_plans_hold_their_pins_with_dma() {
    check(Model::ArMapped, true, MAPPED_AR_DMA);
}

#[test]
fn mapped_ar_plans_hold_their_pins_without_dma() {
    check(Model::ArMapped, false, MAPPED_AR);
}

#[test]
fn dspark_plans_hold_their_pins_with_dma() {
    check(Model::DSpark, true, DSPARK_DMA);
}

#[test]
fn dspark_plans_hold_their_pins_without_dma() {
    check(Model::DSpark, false, DSPARK);
}

// Recorded on 2026-11-06, before the lowering became incremental.

const AR_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9796, 0x05ab852e437f8ee8)),
    ("r32", (9484, 0x4f2b0d79601a78c0)),
    ("r33", (8660, 0xbac997f46be88271)),
    ("r48", (8660, 0xe7cee5ad171470ff)),
    ("r49", (8660, 0x1b84695095d07061)),
    ("r64", (8660, 0x657341332a53ab4b)),
    ("r65", (11656, 0x9b5d485f5312e9a2)),
    ("r112", (11320, 0x719c20dbd5907a1b)),
    ("r113", (11320, 0x6f156e24de680021)),
    ("r127", (11320, 0x5eb05c5aa2774523)),
    ("r128", (8660, 0x8ef7c4fb42b6d8e8)),
    ("r192", (9996, 0x202cb4fdc502da67)),
    ("r255", (13840, 0x8a6467acf6639284)),
    ("r256", (8692, 0x382b80bff08850ec)),
    ("r257", (11720, 0xbdb9403fd3062c7e)),
    ("r319", (11320, 0x45a972ba1d0219cf)),
    ("r320", (11212, 0xc0c3508d1d0337b7)),
    ("r448", (10028, 0xb2356ca9360bd56e)),
    ("r511", (13704, 0x084b70084ba17662)),
    ("r512", (11418, 0x2d996cb50c33c2ec)),
    ("r513", (14478, 0x1f163668746e8ce7)),
    ("r960", (12774, 0xa29bd40f5bd1b067)),
    ("r1023", (16062, 0xac116947844b37c1)),
    ("r1024", (11678, 0xdd371ab8b3a090f5)),
    ("r1472", (12162, 0x437c49c9a19c5a74)),
    ("r1500", (13626, 0xb60838f9abbf09a8)),
    ("r2048", (12014, 0x1f663e01014eb886)),
    ("r2049", (14018, 0x55f87e55d9ac0c1f)),
    ("r2240", (14970, 0x0b244630f91a7195)),
    ("r2300", (13546, 0x2fb2fa0a9cf7befe)),
    ("r2496", (14490, 0xf3dfef1022fec214)),
    ("r2624", (14810, 0x5ff9f7f91f81d9b4)),
    ("r2700", (13194, 0xf95f96d43441897e)),
    ("r3000", (13018, 0xd844907426f3076f)),
    ("r4095", (15186, 0xf0bef3730cd6e8b6)),
    ("r4096", (12514, 0xa30f22e565831fe4)),
    ("r4097", (15034, 0xbce8154ed4562239)),
    ("r8191", (15330, 0x549a0b7caaacd006)),
    ("r8192", (14322, 0xba6ce9bc4e1b1b8f)),
    ("fresh-r512", (11410, 0x867511d65b6cfd5d)),
    ("fresh-r4097", (14898, 0xe9b8d5c2bb03aaee)),
    ("unscored-r8192", (13986, 0x134e7acead7039fc)),
    ("packed-1024-512", (9984, 0x9eaeba490ef24e84)),
    ("packed-100-2000", (13472, 0xfbca1b99bdc3107c)),
    ("anchored-r789", (14910, 0xbd9049c7441ba66a)),
];

const AR: &[(&str, (usize, u64))] = &[
    ("r1", (9796, 0x05ab852e437f8ee8)),
    ("r32", (9484, 0x4f2b0d79601a78c0)),
    ("r33", (8660, 0xbac997f46be88271)),
    ("r48", (8660, 0xe7cee5ad171470ff)),
    ("r49", (8660, 0x1b84695095d07061)),
    ("r64", (8660, 0x657341332a53ab4b)),
    ("r65", (11656, 0x9b5d485f5312e9a2)),
    ("r112", (11320, 0x719c20dbd5907a1b)),
    ("r113", (11320, 0x6f156e24de680021)),
    ("r127", (11320, 0x5eb05c5aa2774523)),
    ("r128", (8660, 0x8ef7c4fb42b6d8e8)),
    ("r192", (9996, 0x202cb4fdc502da67)),
    ("r255", (13840, 0x8a6467acf6639284)),
    ("r256", (8692, 0x382b80bff08850ec)),
    ("r257", (11720, 0xbdb9403fd3062c7e)),
    ("r319", (11320, 0x45a972ba1d0219cf)),
    ("r320", (11212, 0xc0c3508d1d0337b7)),
    ("r448", (10028, 0xb2356ca9360bd56e)),
    ("r511", (13704, 0x084b70084ba17662)),
    ("r512", (8796, 0xbb1a64b10e6db372)),
    ("r513", (11856, 0x87e20dafcc451d20)),
    ("r960", (10152, 0x5b259239422d82eb)),
    ("r1023", (13440, 0xd8cb14f35b1d12d8)),
    ("r1024", (9056, 0x8d7cbba249a417e8)),
    ("r1472", (9540, 0xb7a32ad2624e159b)),
    ("r1500", (11004, 0xc8e93d14e750baff)),
    ("r2048", (9392, 0xfd320c264bd3be65)),
    ("r2049", (11396, 0xaf2f5a75e077a774)),
    ("r2240", (10236, 0xe28dc062582b4df3)),
    ("r2300", (10924, 0xe05c34818ce9f9cc)),
    ("r2496", (9900, 0x294ff30a1905e5d3)),
    ("r2624", (10268, 0xe6a0f257fda42302)),
    ("r2700", (10572, 0xf9cb3c143565f44e)),
    ("r3000", (10396, 0x6906159ff5e37cdf)),
    ("r4095", (12564, 0x01e179d96c32d730)),
    ("r4096", (9796, 0x1b2ba80a743a3a99)),
    ("r4097", (12412, 0x958534872b48a009)),
    ("r8191", (12708, 0x1b1ca5e2b0871ad7)),
    ("r8192", (9780, 0xc150c38bb433eac6)),
    ("fresh-r512", (8788, 0x3a38f87bf5243d5d)),
    ("fresh-r4097", (12276, 0x61256d3e0429f4ac)),
    ("unscored-r8192", (9444, 0xc5ab6d51b6afd28b)),
    ("packed-1024-512", (9984, 0x9eaeba490ef24e84)),
    ("packed-100-2000", (13472, 0xfbca1b99bdc3107c)),
    ("anchored-r789", (12288, 0xe1aa2af434cb9728)),
];

const MAPPED_AR_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x91960acde694b08b)),
    ("r32", (9485, 0xcb9fc9182ff26d87)),
    ("r33", (8661, 0x12b9c817f976d92a)),
    ("r48", (8661, 0xcf01de4d8f893300)),
    ("r49", (8661, 0x6e2e34dce852d4da)),
    ("r64", (8661, 0xe1cf48b18ff2169c)),
    ("r65", (11657, 0x2db48b0620abf1c1)),
    ("r112", (11321, 0x743c8eee1653a3a0)),
    ("r113", (11321, 0x8cded7fa8a839a7e)),
    ("r127", (11321, 0x5cf9c1f8a0a11558)),
    ("r128", (8661, 0xa1782e9a8cce8fcb)),
    ("r192", (9997, 0x30dbfa45bd3e3cc4)),
    ("r255", (13841, 0x5f9d0cf22209bb2b)),
    ("r256", (8693, 0xe3e84b04e42569f3)),
    ("r257", (11721, 0xeee82f0bcfee7321)),
    ("r319", (11321, 0x8c83f87838009ea4)),
    ("r320", (11213, 0xd7fb7d5da4961008)),
    ("r448", (10029, 0xbf3c30d937f1e4a1)),
    ("r511", (13705, 0x32806880a7897e85)),
    ("r512", (11419, 0xa35c6ab324965cf2)),
    ("r513", (14479, 0x28e79e61f579d315)),
    ("r960", (12775, 0x6eda8c85f01bb225)),
    ("r1023", (16063, 0xe2bdd4b13b7ecbcb)),
    ("r1024", (11679, 0x57522b044d91d87f)),
    ("r1472", (12163, 0x3b9c8ad0b677a8ea)),
    ("r1500", (13627, 0x765802e95f10c37a)),
    ("r2048", (12015, 0xaa8cdea564f3b7c8)),
    ("r2049", (14019, 0x5f4c61e4eda1026d)),
    ("r2240", (14971, 0xee9658fc0c195f0f)),
    ("r2300", (13547, 0x9fc848b674ab8bc8)),
    ("r2496", (14491, 0x4433151b206083da)),
    ("r2624", (14811, 0x43d17d67235931b2)),
    ("r2700", (13195, 0x7f351c802eb87918)),
    ("r3000", (13019, 0xb40378bcd7db5481)),
    ("r4095", (15187, 0xa7e3838246370d80)),
    ("r4096", (12515, 0x21e9e1b18ebefa8e)),
    ("r4097", (15035, 0xf41c65c9d01030cf)),
    ("r8191", (15331, 0x33ad0f130053a394)),
    ("r8192", (14323, 0x8736a4acdb1184c9)),
    ("fresh-r512", (11411, 0x5e299abf7b903cef)),
    ("fresh-r4097", (14899, 0x6cec7d77feb27828)),
    ("unscored-r8192", (13987, 0x26773f60f6f32cf6)),
    ("packed-1024-512", (9985, 0x896efbf80b710cc3)),
    ("packed-100-2000", (13473, 0xc25436961410e10f)),
    ("anchored-r789", (14911, 0x67706cdf59b342e8)),
];

const MAPPED_AR: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x91960acde694b08b)),
    ("r32", (9485, 0xcb9fc9182ff26d87)),
    ("r33", (8661, 0x12b9c817f976d92a)),
    ("r48", (8661, 0xcf01de4d8f893300)),
    ("r49", (8661, 0x6e2e34dce852d4da)),
    ("r64", (8661, 0xe1cf48b18ff2169c)),
    ("r65", (11657, 0x2db48b0620abf1c1)),
    ("r112", (11321, 0x743c8eee1653a3a0)),
    ("r113", (11321, 0x8cded7fa8a839a7e)),
    ("r127", (11321, 0x5cf9c1f8a0a11558)),
    ("r128", (8661, 0xa1782e9a8cce8fcb)),
    ("r192", (9997, 0x30dbfa45bd3e3cc4)),
    ("r255", (13841, 0x5f9d0cf22209bb2b)),
    ("r256", (8693, 0xe3e84b04e42569f3)),
    ("r257", (11721, 0xeee82f0bcfee7321)),
    ("r319", (11321, 0x8c83f87838009ea4)),
    ("r320", (11213, 0xd7fb7d5da4961008)),
    ("r448", (10029, 0xbf3c30d937f1e4a1)),
    ("r511", (13705, 0x32806880a7897e85)),
    ("r512", (8797, 0xe42e70225edffc0d)),
    ("r513", (11857, 0x77f2af1ac78fa587)),
    ("r960", (10153, 0x2e528c619f93e9e8)),
    ("r1023", (13441, 0x45046d15433b0e13)),
    ("r1024", (9057, 0xe98bbf588853ca93)),
    ("r1472", (9541, 0x0d73d01d37c5d76c)),
    ("r1500", (11005, 0xf9ec3f271a5ae614)),
    ("r2048", (9393, 0x31ce9f151502920e)),
    ("r2049", (11397, 0xd0b1791131fb8f17)),
    ("r2240", (10237, 0x18309d8526419e6c)),
    ("r2300", (10925, 0x059a7a25148b886b)),
    ("r2496", (9901, 0x57145e6496483134)),
    ("r2624", (10269, 0xbc64a6663ef11429)),
    ("r2700", (10573, 0x1d1901cf71511615)),
    ("r3000", (10397, 0x1685e50b29664a50)),
    ("r4095", (12565, 0x19b4e213074c517f)),
    ("r4096", (9797, 0xc963c76a2a419462)),
    ("r4097", (12413, 0xd4e93e5ed10c5372)),
    ("r8191", (12709, 0x19121160a51d2bcc)),
    ("r8192", (9781, 0xdce7c980edbe6975)),
    ("fresh-r512", (8789, 0x8838aaf51789b3a6)),
    ("fresh-r4097", (12277, 0x54c6b88b7e2a89e3)),
    ("unscored-r8192", (9445, 0xaf5fe47ec272e640)),
    ("packed-1024-512", (9985, 0x896efbf80b710cc3)),
    ("packed-100-2000", (13473, 0xc25436961410e10f)),
    ("anchored-r789", (12289, 0xb0a155e6f155586b)),
];

const DSPARK_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0xc9ebc4dfd99a5e1d)),
    ("r32", (9501, 0x85f71b239cf9d567)),
    ("r33", (8677, 0x1592da898e7564e2)),
    ("r48", (8677, 0x2905fb6396f4a67c)),
    ("r49", (8677, 0x30a51d4803ce793e)),
    ("r64", (8677, 0x2afa056afe485620)),
    ("r65", (11673, 0x84b7c6b0fdf34ad1)),
    ("r112", (11337, 0x76c1dfbf854fcd38)),
    ("r113", (11337, 0x98cc592ae5e29b3e)),
    ("r127", (11337, 0x58a47ea3be588022)),
    ("r128", (8677, 0xf8b332fdb3a82d7a)),
    ("r192", (10013, 0xef76ef416124b121)),
    ("r255", (13857, 0x83c2ee42a902a808)),
    ("r256", (8709, 0x7bb1b9392a709b5b)),
    ("r257", (11737, 0xcc1905e0f7eba6b0)),
    ("r319", (11393, 0x4e84a25a70dfdcd9)),
    ("r320", (11229, 0x16a8ffc38f191197)),
    ("r448", (10045, 0x86a7d44ef2ce235b)),
    ("r511", (13721, 0xc7f4ef81e3ba5125)),
    ("r512", (11435, 0xb192ace12f536211)),
    ("r513", (14495, 0xd2fd46fb0cbe0dcb)),
    ("r960", (12791, 0x89885ae86480b588)),
    ("r1023", (16079, 0xc07196026f8603da)),
    ("r1024", (11695, 0x30bde2f63cad2d75)),
    ("r1472", (12179, 0xfa5f7161b11c203d)),
    ("r1500", (13643, 0x671bac5c1e6b4a55)),
    ("r2048", (12031, 0xe4f87df1229b89ab)),
    ("r2049", (14035, 0x257b12223bca1b26)),
    ("r2240", (14987, 0x146a651060ccd26e)),
    ("r2300", (13563, 0xa4c69276d230ca94)),
    ("r2496", (14507, 0x14c13338bce0e11f)),
    ("r2624", (14827, 0x1d9623fc354308e1)),
    ("r2700", (14035, 0xdac20a6a37dd5fff)),
    ("r3000", (13843, 0x44cc7e8cc211c31c)),
    ("r4095", (15203, 0x05c834b531876bff)),
    ("r4096", (12531, 0x9ab1f41dde74ce3c)),
    ("r4097", (15051, 0x73e4dfb33b53907c)),
    ("r8191", (15347, 0x0fcab15a7fc93c13)),
    ("r8192", (14339, 0x3d38ab2a70cba21d)),
    ("fresh-r512", (11427, 0xb247ede5dcb1d1e4)),
    ("fresh-r4097", (14915, 0xbc68cdfbfc87e3a7)),
    ("unscored-r8192", (14003, 0x1f124d42e989fef6)),
    ("packed-1024-512", (10017, 0xf9f6613a4f9bae10)),
    ("packed-100-2000", (14177, 0x0dfc498cf857ba13)),
    ("anchored-r789", (14927, 0x855be5ec909d3144)),
];

const DSPARK: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0xc9ebc4dfd99a5e1d)),
    ("r32", (9501, 0x85f71b239cf9d567)),
    ("r33", (8677, 0x1592da898e7564e2)),
    ("r48", (8677, 0x2905fb6396f4a67c)),
    ("r49", (8677, 0x30a51d4803ce793e)),
    ("r64", (8677, 0x2afa056afe485620)),
    ("r65", (11673, 0x84b7c6b0fdf34ad1)),
    ("r112", (11337, 0x76c1dfbf854fcd38)),
    ("r113", (11337, 0x98cc592ae5e29b3e)),
    ("r127", (11337, 0x58a47ea3be588022)),
    ("r128", (8677, 0xf8b332fdb3a82d7a)),
    ("r192", (10013, 0xef76ef416124b121)),
    ("r255", (13857, 0x83c2ee42a902a808)),
    ("r256", (8709, 0x7bb1b9392a709b5b)),
    ("r257", (11737, 0xcc1905e0f7eba6b0)),
    ("r319", (11393, 0x4e84a25a70dfdcd9)),
    ("r320", (11229, 0x16a8ffc38f191197)),
    ("r448", (10045, 0x86a7d44ef2ce235b)),
    ("r511", (13721, 0xc7f4ef81e3ba5125)),
    ("r512", (8813, 0x7ba897b353fbc05e)),
    ("r513", (11873, 0x51bd61720297d853)),
    ("r960", (10169, 0x73c602a172c1690d)),
    ("r1023", (13457, 0x5dfe620f6846a65e)),
    ("r1024", (9073, 0x61ff98fc5765027d)),
    ("r1472", (9557, 0x0f989e84ebd68fbb)),
    ("r1500", (11021, 0x8652b474b875dd61)),
    ("r2048", (9409, 0x805d332a258c8b19)),
    ("r2049", (11413, 0xa375cba2a39a762a)),
    ("r2240", (10253, 0xd642e00869f0fae1)),
    ("r2300", (10941, 0x056fbc8a3cd5e74f)),
    ("r2496", (9917, 0xeb410897d1326b51)),
    ("r2624", (10285, 0x93a9586a5d123f32)),
    ("r2700", (11413, 0x5600805af7d471d6)),
    ("r3000", (11221, 0x800519bb534f0567)),
    ("r4095", (12581, 0xd996b53fc4eb348c)),
    ("r4096", (9813, 0x75e0672764eb7f0e)),
    ("r4097", (12429, 0xa5fd6b73c603e8e3)),
    ("r8191", (12725, 0x7502eac88e623863)),
    ("r8192", (9797, 0x4a06a34f97d07ecd)),
    ("fresh-r512", (8805, 0xa01405b510ca573d)),
    ("fresh-r4097", (12293, 0xc3522189304e8306)),
    ("unscored-r8192", (9461, 0x31ae89dd2f2000bc)),
    ("packed-1024-512", (10017, 0xf9f6613a4f9bae10)),
    ("packed-100-2000", (14177, 0x0dfc498cf857ba13)),
    ("anchored-r789", (12305, 0xeccb20bed6918bf7)),
];
