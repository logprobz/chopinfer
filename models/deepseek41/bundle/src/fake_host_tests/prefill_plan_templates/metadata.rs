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
    ("r1", (9796, 0x0e46c41e727e9692)),
    ("r32", (9484, 0x9fbcecf43d3060b4)),
    ("r33", (8660, 0x9ca8fe5148d1d953)),
    ("r48", (8660, 0xa496b13446bcd225)),
    ("r49", (8660, 0xa68d71da86bbd05b)),
    ("r64", (8660, 0xbcd9a99f8702e729)),
    ("r65", (11656, 0x73919915d7fe2de4)),
    ("r112", (11320, 0x0883a64f27c4ee99)),
    ("r113", (11320, 0x9fc873cd9083d1db)),
    ("r127", (11320, 0xc188f4a4a35800c5)),
    ("r128", (8660, 0x3e0678423db9108e)),
    ("r192", (9996, 0x4c798f5b847abdfd)),
    ("r255", (13840, 0xe1278c216bfbed68)),
    ("r256", (8692, 0xc707011a0795cd12)),
    ("r257", (11720, 0x8f335fd28ea086e6)),
    ("r319", (11320, 0x190c5a9ee83537ad)),
    ("r320", (11212, 0x4b7b7071d3441ed1)),
    ("r448", (10028, 0x366e062843307d68)),
    ("r511", (13704, 0x144fd31cc57f7ae0)),
    ("r512", (11418, 0x031d72dee8a05682)),
    ("r513", (14478, 0x168b8594e1a04d9d)),
    ("r960", (12774, 0xf3b17241535efb71)),
    ("r1023", (16062, 0x3fb223b2a038eaf7)),
    ("r1024", (11678, 0x8a2d7cb00698f2b9)),
    ("r1472", (12162, 0x2095b89724aa693e)),
    ("r1500", (13626, 0x3a7c838cb4f7eb2a)),
    ("r2048", (12014, 0x2db39c03b2f6c1fa)),
    ("r2049", (14018, 0xebaa9d394cfc4237)),
    ("r2240", (14970, 0x367d615146554745)),
    ("r2300", (13546, 0xa4dbc140d21c3c4e)),
    ("r2496", (14490, 0xccb1fb636009b83e)),
    ("r2624", (14810, 0x0788431a9db22a32)),
    ("r2700", (13194, 0x1da7b4f6d9064396)),
    ("r3000", (13018, 0x9f8c16415794b81b)),
    ("r4095", (15186, 0xf7d1e85d60002b28)),
    ("r4096", (12514, 0x795f3546249b85ac)),
    ("r4097", (15034, 0xe4f258ebede0e56b)),
    ("r8191", (15330, 0x543c2b506cddcf1e)),
    ("r8192", (14322, 0xb34f2cc63a4ef60d)),
    ("fresh-r512", (11410, 0xbed002175c0ef381)),
    ("fresh-r4097", (14898, 0xa0eec4026e94a688)),
    ("unscored-r8192", (13986, 0x56d8bd745ce1ff95)),
    ("packed-1024-512", (9984, 0xecf5dd525c76109a)),
    ("packed-100-2000", (13472, 0xff50af6085f47410)),
    ("anchored-r789", (14910, 0xb28a12fe20371904)),
];

const AR: &[(&str, (usize, u64))] = &[
    ("r1", (9796, 0x0e46c41e727e9692)),
    ("r32", (9484, 0x9fbcecf43d3060b4)),
    ("r33", (8660, 0x9ca8fe5148d1d953)),
    ("r48", (8660, 0xa496b13446bcd225)),
    ("r49", (8660, 0xa68d71da86bbd05b)),
    ("r64", (8660, 0xbcd9a99f8702e729)),
    ("r65", (11656, 0x73919915d7fe2de4)),
    ("r112", (11320, 0x0883a64f27c4ee99)),
    ("r113", (11320, 0x9fc873cd9083d1db)),
    ("r127", (11320, 0xc188f4a4a35800c5)),
    ("r128", (8660, 0x3e0678423db9108e)),
    ("r192", (9996, 0x4c798f5b847abdfd)),
    ("r255", (13840, 0xe1278c216bfbed68)),
    ("r256", (8692, 0xc707011a0795cd12)),
    ("r257", (11720, 0x8f335fd28ea086e6)),
    ("r319", (11320, 0x190c5a9ee83537ad)),
    ("r320", (11212, 0x4b7b7071d3441ed1)),
    ("r448", (10028, 0x366e062843307d68)),
    ("r511", (13704, 0x144fd31cc57f7ae0)),
    ("r512", (8796, 0x15014a07ae60c814)),
    ("r513", (11856, 0xb18d855094ce56c6)),
    ("r960", (10152, 0x85c612eca5d76519)),
    ("r1023", (13440, 0x863dafb9a995a62a)),
    ("r1024", (9056, 0xd23b85b237caceec)),
    ("r1472", (9540, 0x21102064595f1dbd)),
    ("r1500", (11004, 0x05c67b260bf9d74d)),
    ("r2048", (9392, 0x8d18485f804580e1)),
    ("r2049", (11396, 0x11605e62bb5ea24c)),
    ("r2240", (10236, 0x828f12db67ad3f43)),
    ("r2300", (10924, 0x05209430cea7ca7c)),
    ("r2496", (9900, 0x68cc7a1945ff2175)),
    ("r2624", (10268, 0x3095770d27f6736c)),
    ("r2700", (10572, 0x0000a3493a9f6316)),
    ("r3000", (10396, 0x0fe50546504c15bb)),
    ("r4095", (12564, 0x88f313a9afcebcc6)),
    ("r4096", (9796, 0x0dec4fc94fb88581)),
    ("r4097", (12412, 0x953ce198183ba337)),
    ("r8191", (12708, 0x223e3df50def624f)),
    ("r8192", (9780, 0xa19135e8e19fb598)),
    ("fresh-r512", (8788, 0xe440d1fdc8665e59)),
    ("fresh-r4097", (12276, 0xc0aaa578444033ea)),
    ("unscored-r8192", (9444, 0x478d87198f4d6d24)),
    ("packed-1024-512", (9984, 0xecf5dd525c76109a)),
    ("packed-100-2000", (13472, 0xff50af6085f47410)),
    ("anchored-r789", (12288, 0xb635def5ee8c3fae)),
];

const MAPPED_AR_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x05ba9d97edd7a4f5)),
    ("r32", (9485, 0x9d9200966991d13b)),
    ("r33", (8661, 0x0eb11988e2169fd4)),
    ("r48", (8661, 0x09ec9d9ce71f556e)),
    ("r49", (8661, 0x2540cf3888d7c66c)),
    ("r64", (8661, 0xb61ee00bb549c3f2)),
    ("r65", (11657, 0xcbee527b2d4cb8d3)),
    ("r112", (11321, 0x993c1aebc9dd11c6)),
    ("r113", (11321, 0x8a45c8c2686ffee0)),
    ("r127", (11321, 0x73964afdf695de92)),
    ("r128", (8661, 0x10e19aae2804e691)),
    ("r192", (9997, 0xcebd19dc0cbb61b2)),
    ("r255", (13841, 0x5e8d5375c9687a1f)),
    ("r256", (8693, 0xffcb6de59038e669)),
    ("r257", (11721, 0x563df0a071f66cf9)),
    ("r319", (11321, 0x042f8623d3175dca)),
    ("r320", (11213, 0x71660d10d109d8aa)),
    ("r448", (10029, 0xa015f2d278d4ce1b)),
    ("r511", (13705, 0x382d2fc9c6ed5753)),
    ("r512", (11419, 0xab1719fa538071e0)),
    ("r513", (14479, 0x12a17be14c5bf76b)),
    ("r960", (12775, 0x59c53889b0cd5447)),
    ("r1023", (16063, 0x229c30464bb40909)),
    ("r1024", (11679, 0x7570f884cc738a4b)),
    ("r1472", (12163, 0xf49f1e15fe94b394)),
    ("r1500", (13627, 0x5b1946119dce746c)),
    ("r2048", (12015, 0x3b7f4d66273a44f4)),
    ("r2049", (14019, 0xa64a99c846d48c45)),
    ("r2240", (14971, 0x90d07811e895877f)),
    ("r2300", (13547, 0xd194d5a497866e98)),
    ("r2496", (14491, 0x0a33d10e24b7f2d4)),
    ("r2624", (14811, 0xd4ee5ed7522976b8)),
    ("r2700", (13195, 0x2519ef01930392b0)),
    ("r3000", (13019, 0x9e546a643575be85)),
    ("r4095", (15187, 0xf8681d8f62c4ad8a)),
    ("r4096", (12515, 0x8e0c96efaefd5476)),
    ("r4097", (15035, 0xc2b4b85ed1425f41)),
    ("r8191", (15331, 0x2200d29e152c0d2c)),
    ("r8192", (14323, 0x41bff68fafc18a97)),
    ("fresh-r512", (11411, 0x766142f28e44535b)),
    ("fresh-r4097", (14899, 0x9ba04e5daa0dbc0a)),
    ("unscored-r8192", (13987, 0xca537a90f0d8c7fb)),
    ("packed-1024-512", (9985, 0x24624cc25782e389)),
    ("packed-100-2000", (13473, 0x94dd3ab70b53c6e3)),
    ("anchored-r789", (14911, 0x9a58a08f6d678bca)),
];

const MAPPED_AR: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x05ba9d97edd7a4f5)),
    ("r32", (9485, 0x9d9200966991d13b)),
    ("r33", (8661, 0x0eb11988e2169fd4)),
    ("r48", (8661, 0x09ec9d9ce71f556e)),
    ("r49", (8661, 0x2540cf3888d7c66c)),
    ("r64", (8661, 0xb61ee00bb549c3f2)),
    ("r65", (11657, 0xcbee527b2d4cb8d3)),
    ("r112", (11321, 0x993c1aebc9dd11c6)),
    ("r113", (11321, 0x8a45c8c2686ffee0)),
    ("r127", (11321, 0x73964afdf695de92)),
    ("r128", (8661, 0x10e19aae2804e691)),
    ("r192", (9997, 0xcebd19dc0cbb61b2)),
    ("r255", (13841, 0x5e8d5375c9687a1f)),
    ("r256", (8693, 0xffcb6de59038e669)),
    ("r257", (11721, 0x563df0a071f66cf9)),
    ("r319", (11321, 0x042f8623d3175dca)),
    ("r320", (11213, 0x71660d10d109d8aa)),
    ("r448", (10029, 0xa015f2d278d4ce1b)),
    ("r511", (13705, 0x382d2fc9c6ed5753)),
    ("r512", (8797, 0x3a793e17407c307f)),
    ("r513", (11857, 0x3e86ba80af1a4d5d)),
    ("r960", (10153, 0xad6d5196f30f694e)),
    ("r1023", (13441, 0x4fdf57b4deb149f5)),
    ("r1024", (9057, 0x55b3fbe1cff898c7)),
    ("r1472", (9541, 0x64ead1735b5753f6)),
    ("r1500", (11005, 0xa9b70896bda7cfca)),
    ("r2048", (9393, 0xa4959546a55cac1a)),
    ("r2049", (11397, 0xb26ee739d2b0169f)),
    ("r2240", (10237, 0xb417ef9685c4f4fc)),
    ("r2300", (10925, 0x7581d579ba4a0c5b)),
    ("r2496", (9901, 0x19ad67f5704216be)),
    ("r2624", (10269, 0x3f31c11376d89bc3)),
    ("r2700", (10573, 0xdbcdf3dfb3e6c96d)),
    ("r3000", (10397, 0xf8701131b2f775dc)),
    ("r4095", (12565, 0xdeae5f45a351a9d5)),
    ("r4096", (9797, 0x5491a2d2b674553a)),
    ("r4097", (12413, 0xaf798b5ff7730308)),
    ("r8191", (12709, 0x8122c316761cbd14)),
    ("r8192", (9781, 0x121fece37b169b57)),
    ("fresh-r512", (8789, 0x0499108a5f92dc92)),
    ("fresh-r4097", (12277, 0xd7a72f883e0ccf51)),
    ("unscored-r8192", (9445, 0x882abb48d7c4ab4b)),
    ("packed-1024-512", (9985, 0x24624cc25782e389)),
    ("packed-100-2000", (13473, 0x94dd3ab70b53c6e3)),
    ("anchored-r789", (12289, 0xb8a38b57b2f3c161)),
];

const DSPARK_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0x8fb83197121886f1)),
    ("r32", (9501, 0x8ba4116797c0cba7)),
    ("r33", (8677, 0xf114067368a36084)),
    ("r48", (8677, 0x08ae75bcaf8952c4)),
    ("r49", (8677, 0xac4c4592e4722e22)),
    ("r64", (8677, 0x48b6a4afc501500a)),
    ("r65", (11673, 0xa2a9feba30963b85)),
    ("r112", (11337, 0x42826ef5a596029e)),
    ("r113", (11337, 0xab4a9db79d3cb808)),
    ("r127", (11337, 0x9719597350beeefe)),
    ("r128", (8677, 0xc834c2d35e5aa136)),
    ("r192", (10013, 0x12c1594e9b2dc3db)),
    ("r255", (13857, 0x28c46c83d6f91abc)),
    ("r256", (8709, 0x21af26518b3f9a5f)),
    ("r257", (11737, 0x52c99f2309ad9188)),
    ("r319", (11393, 0x714c532468911731)),
    ("r320", (11229, 0x922be3f50f1e3fe5)),
    ("r448", (10045, 0x03b480bb4d1ac083)),
    ("r511", (13721, 0x24aa015951c04cdb)),
    ("r512", (11435, 0xea97694dfdc25bf1)),
    ("r513", (14495, 0xbe287e57bfb6578d)),
    ("r960", (12791, 0xfdceb82b35bd71bc)),
    ("r1023", (16079, 0xad6bd3d2c9dea64c)),
    ("r1024", (11695, 0xac3b1fb9b0ae71b5)),
    ("r1472", (12179, 0xa7eb13ab1a0595eb)),
    ("r1500", (13643, 0x91a95ec7c07746a7)),
    ("r2048", (12031, 0x56bc1d7795930773)),
    ("r2049", (14035, 0x3c2dfb7c6c282280)),
    ("r2240", (14987, 0xcacd8a97e7a43028)),
    ("r2300", (13563, 0xa6f74a5d295396b8)),
    ("r2496", (14507, 0x10599c140a7613bd)),
    ("r2624", (14827, 0x02cf3bd6446405f5)),
    ("r2700", (14035, 0x21d57fba668eddc7)),
    ("r3000", (13843, 0x865b5eda97d845ac)),
    ("r4095", (15203, 0xf73a509d7700161f)),
    ("r4096", (12531, 0xaa2ed172b8567684)),
    ("r4097", (15051, 0x8189d2a1f8ea5fe0)),
    ("r8191", (15347, 0xe52b1987a885bc95)),
    ("r8192", (14339, 0x081e624890cc9d29)),
    ("fresh-r512", (11427, 0x6473fe99e39f941c)),
    ("fresh-r4097", (14915, 0x95ee5b7577e6453b)),
    ("unscored-r8192", (14003, 0x914a808bb80a11fe)),
    ("packed-1024-512", (10017, 0xc9c85e92bf6aa89e)),
    ("packed-100-2000", (14177, 0xd7e34de18dac97de)),
    ("anchored-r789", (14927, 0x4e82960c39d6ca8a)),
];

const DSPARK: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0x8fb83197121886f1)),
    ("r32", (9501, 0x8ba4116797c0cba7)),
    ("r33", (8677, 0xf114067368a36084)),
    ("r48", (8677, 0x08ae75bcaf8952c4)),
    ("r49", (8677, 0xac4c4592e4722e22)),
    ("r64", (8677, 0x48b6a4afc501500a)),
    ("r65", (11673, 0xa2a9feba30963b85)),
    ("r112", (11337, 0x42826ef5a596029e)),
    ("r113", (11337, 0xab4a9db79d3cb808)),
    ("r127", (11337, 0x9719597350beeefe)),
    ("r128", (8677, 0xc834c2d35e5aa136)),
    ("r192", (10013, 0x12c1594e9b2dc3db)),
    ("r255", (13857, 0x28c46c83d6f91abc)),
    ("r256", (8709, 0x21af26518b3f9a5f)),
    ("r257", (11737, 0x52c99f2309ad9188)),
    ("r319", (11393, 0x714c532468911731)),
    ("r320", (11229, 0x922be3f50f1e3fe5)),
    ("r448", (10045, 0x03b480bb4d1ac083)),
    ("r511", (13721, 0x24aa015951c04cdb)),
    ("r512", (8813, 0x091ee5f896b14fbe)),
    ("r513", (11873, 0x118b186ac2897191)),
    ("r960", (10169, 0x767d5e367a2d2989)),
    ("r1023", (13457, 0x80c239a99e9ce948)),
    ("r1024", (9073, 0xd3edf126a261f63d)),
    ("r1472", (9557, 0x2c12d1def11444cd)),
    ("r1500", (11021, 0x8b42e3d2ee6fd0bb)),
    ("r2048", (9409, 0x83e7bd05e4c65b91)),
    ("r2049", (11413, 0x3f92d2aaf9dccea8)),
    ("r2240", (10253, 0xdc7e53628290093f)),
    ("r2300", (10941, 0x31fec59ab091fdb3)),
    ("r2496", (9917, 0x39b6af070222adc3)),
    ("r2624", (10285, 0x0363aeb77e278c3e)),
    ("r2700", (11413, 0x69714457483224de)),
    ("r3000", (11221, 0xa1c405acea9d2417)),
    ("r4095", (12581, 0x0d3dad91ec5f1fac)),
    ("r4096", (9813, 0x836c127fe14d5786)),
    ("r4097", (12429, 0xadcb5d43c02ad8d7)),
    ("r8191", (12725, 0x7bbc5dfdc7f2f745)),
    ("r8192", (9797, 0x0a6cc96932f19639)),
    ("fresh-r512", (8805, 0x8edc1a6937218795)),
    ("fresh-r4097", (12293, 0xc347a191eb24dcaa)),
    ("unscored-r8192", (9461, 0xea113a1e0ecad5a4)),
    ("packed-1024-512", (10017, 0xc9c85e92bf6aa89e)),
    ("packed-100-2000", (14177, 0xd7e34de18dac97de)),
    ("anchored-r789", (12305, 0xa0e246445c0a5bfd)),
];
