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
    ("r1", (9796, 0x15f9dfeaaeff9ea4)),
    ("r32", (9484, 0x2d9d1c06f3446c58)),
    ("r33", (8660, 0xbb8a8167c376b9f9)),
    ("r48", (8660, 0x468317b94a91d627)),
    ("r49", (8660, 0xd544b7d735d12fa9)),
    ("r64", (8660, 0x916977b9b477b5cb)),
    ("r65", (11656, 0xf5660809ad5007c2)),
    ("r112", (11320, 0x5c6e3c11a94cc42b)),
    ("r113", (11320, 0xe5ac06cb5575b1bd)),
    ("r127", (11320, 0xa2ec5ccb4a3e13df)),
    ("r128", (8660, 0x9c395e2bd43a5aa0)),
    ("r192", (9996, 0x65d7c08e74b2799f)),
    ("r255", (13840, 0xdd937f976c47da08)),
    ("r256", (8692, 0xea76523efadd7d6c)),
    ("r257", (11720, 0xc2ce3aaf4633c11a)),
    ("r319", (11320, 0x8c59d20ac37d962f)),
    ("r320", (11212, 0xe517e70dfda73f6f)),
    ("r448", (10028, 0xd075aa908ba0effa)),
    ("r511", (13704, 0xc61f4b587b0d6f6a)),
    ("r512", (11098, 0x2f8b1c5c69214eb2)),
    ("r513", (13902, 0x76bf32f4761c148b)),
    ("r960", (12454, 0xd36e0e84d6c67503)),
    ("r1023", (15486, 0x34040d212ce8f62f)),
    ("r1024", (11358, 0x3ab77440bcc3f829)),
    ("r1472", (12626, 0xa4f282647a2c087a)),
    ("r1500", (15426, 0x72d4d5d625df6b62)),
    ("r2048", (11958, 0x1cc049b6b304d68e)),
    ("r2049", (14970, 0x484f2dd5cd90b14f)),
    ("r2240", (15882, 0x4fc858a75d34411b)),
    ("r2300", (15106, 0x1a8167e4a5762248)),
    ("r2496", (15066, 0xdfa54a7b06536d21)),
    ("r2624", (15706, 0x5059abee4b249da7)),
    ("r2700", (13866, 0xdf9e64bca8434639)),
    ("r3000", (13690, 0x9b0dd203dd12a206)),
    ("r4095", (15058, 0xdcec48a61764f58d)),
    ("r4096", (12610, 0xafb490c5af35b093)),
    ("r4097", (15338, 0xf33e16f4a2d93157)),
    ("r8191", (15202, 0xc37b5f7f76fbb1c1)),
    ("r8192", (14738, 0xd7068a2b95507b86)),
    ("fresh-r512", (11090, 0x2cf1846922c56b73)),
    ("fresh-r4097", (15202, 0x6702eb7d717437ce)),
    ("unscored-r8192", (14402, 0x12641a876f5675ef)),
    ("packed-1024-512", (10088, 0xfef9b3273bba5097)),
    ("packed-100-2000", (14720, 0x24d75eb27128dae3)),
    ("anchored-r789", (14334, 0xc4518336d9723826)),
];

const AR: &[(&str, (usize, u64))] = &[
    ("r1", (9796, 0x15f9dfeaaeff9ea4)),
    ("r32", (9484, 0x2d9d1c06f3446c58)),
    ("r33", (8660, 0xbb8a8167c376b9f9)),
    ("r48", (8660, 0x468317b94a91d627)),
    ("r49", (8660, 0xd544b7d735d12fa9)),
    ("r64", (8660, 0x916977b9b477b5cb)),
    ("r65", (11656, 0xf5660809ad5007c2)),
    ("r112", (11320, 0x5c6e3c11a94cc42b)),
    ("r113", (11320, 0xe5ac06cb5575b1bd)),
    ("r127", (11320, 0xa2ec5ccb4a3e13df)),
    ("r128", (8660, 0x9c395e2bd43a5aa0)),
    ("r192", (9996, 0x65d7c08e74b2799f)),
    ("r255", (13840, 0xdd937f976c47da08)),
    ("r256", (8692, 0xea76523efadd7d6c)),
    ("r257", (11720, 0xc2ce3aaf4633c11a)),
    ("r319", (11320, 0x8c59d20ac37d962f)),
    ("r320", (11212, 0xe517e70dfda73f6f)),
    ("r448", (10028, 0xd075aa908ba0effa)),
    ("r511", (13704, 0xc61f4b587b0d6f6a)),
    ("r512", (8796, 0x3fc5f7d4b2bbd008)),
    ("r513", (11856, 0x6c0e64d21434d944)),
    ("r960", (10152, 0xea0a803417a7b25b)),
    ("r1023", (13440, 0xcbc3bd7d91fda724)),
    ("r1024", (9056, 0x69659b315d4e6998)),
    ("r1472", (10324, 0xc2eff399424cbe2d)),
    ("r1500", (13380, 0x9c8339aeb4c4db89)),
    ("r2048", (9656, 0x5866a0b736d9d6ed)),
    ("r2049", (12924, 0x49fc547ec9ffddcc)),
    ("r2240", (11468, 0xd57da2ac50544871)),
    ("r2300", (13060, 0x077350798c0532e6)),
    ("r2496", (10796, 0x8f3baff9658d9968)),
    ("r2624", (11484, 0xf73b7c8f32077e8b)),
    ("r2700", (11820, 0x73d3c687ddcc6893)),
    ("r3000", (11644, 0x635232818eb799f4)),
    ("r4095", (13012, 0x68549dbf8c8f5509)),
    ("r4096", (10212, 0x197b862be9dd6cd8)),
    ("r4097", (13292, 0x6df8599912100d5f)),
    ("r8191", (13156, 0x0a8e867163a5d8de)),
    ("r8192", (10196, 0x5f4ee95a66b89c75)),
    ("fresh-r512", (8788, 0x61155218039b957b)),
    ("fresh-r4097", (13156, 0x2738ab0fc6f181b4)),
    ("unscored-r8192", (9860, 0x7521fbfbdcc50f86)),
    ("packed-1024-512", (10088, 0xfef9b3273bba5097)),
    ("packed-100-2000", (14720, 0x24d75eb27128dae3)),
    ("anchored-r789", (12288, 0xf9759d50f0273e14)),
];

const MAPPED_AR_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x2c62d696f5e2a507)),
    ("r32", (9485, 0x70bfd9a95917780f)),
    ("r33", (8661, 0xdac3659ee91aae42)),
    ("r48", (8661, 0x5a7d7cc178655498)),
    ("r49", (8661, 0xa21407990bc90872)),
    ("r64", (8661, 0x82180d6de8b60d1c)),
    ("r65", (11657, 0x3e184c724e471661)),
    ("r112", (11321, 0x334ae592fc841530)),
    ("r113", (11321, 0xebe07377cd2134da)),
    ("r127", (11321, 0x0fe88340293b86f4)),
    ("r128", (8661, 0x576590a2b0035eb3)),
    ("r192", (9997, 0x2dec726557edd22c)),
    ("r255", (13841, 0x0286df8cd886843f)),
    ("r256", (8693, 0x0e8cc811c01e7273)),
    ("r257", (11721, 0x163e70cdb2530a5d)),
    ("r319", (11321, 0xdb1d0adc154e8404)),
    ("r320", (11213, 0x07b62e1449a1f250)),
    ("r448", (10029, 0x92c16f5b24fa450d)),
    ("r511", (13705, 0xe51260085e8e639d)),
    ("r512", (11099, 0x80a1858dbb4b715c)),
    ("r513", (13903, 0xdeb35db5bd01a1c5)),
    ("r960", (12455, 0x4a0b8e434a8df2dd)),
    ("r1023", (15487, 0x5bda34368b8cba61)),
    ("r1024", (11359, 0x899b4b5906c6d8c7)),
    ("r1472", (12627, 0xfb4711d0736b6ebc)),
    ("r1500", (15427, 0x47ba35bd7b604378)),
    ("r2048", (11959, 0x73f542333acaf9b0)),
    ("r2049", (14971, 0x0c8bfa2629422b95)),
    ("r2240", (15883, 0x0b603752c1766695)),
    ("r2300", (15107, 0x381b066f4002100a)),
    ("r2496", (15067, 0x6d2e672f5960d9df)),
    ("r2624", (15707, 0x4d7c14d9e282c965)),
    ("r2700", (13867, 0x240426a01fcc7ddb)),
    ("r3000", (13691, 0x4f4d50d8346c7d94)),
    ("r4095", (15059, 0xfbd3925d6a2e447f)),
    ("r4096", (12611, 0xff33fed7182abd39)),
    ("r4097", (15339, 0x5ca83b7db72077d9)),
    ("r8191", (15203, 0x94c3361f754a288f)),
    ("r8192", (14739, 0x20330c92e57f28d4)),
    ("fresh-r512", (11091, 0x079e6f04c4f1d699)),
    ("fresh-r4097", (15203, 0x2e9907804a675430)),
    ("unscored-r8192", (14403, 0x4703f291a673f719)),
    ("packed-1024-512", (10089, 0x9109ff2243aa5d0c)),
    ("packed-100-2000", (14721, 0xccdd4be6a0f3985c)),
    ("anchored-r789", (14335, 0x3960b16cc2668abc)),
];

const MAPPED_AR: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x2c62d696f5e2a507)),
    ("r32", (9485, 0x70bfd9a95917780f)),
    ("r33", (8661, 0xdac3659ee91aae42)),
    ("r48", (8661, 0x5a7d7cc178655498)),
    ("r49", (8661, 0xa21407990bc90872)),
    ("r64", (8661, 0x82180d6de8b60d1c)),
    ("r65", (11657, 0x3e184c724e471661)),
    ("r112", (11321, 0x334ae592fc841530)),
    ("r113", (11321, 0xebe07377cd2134da)),
    ("r127", (11321, 0x0fe88340293b86f4)),
    ("r128", (8661, 0x576590a2b0035eb3)),
    ("r192", (9997, 0x2dec726557edd22c)),
    ("r255", (13841, 0x0286df8cd886843f)),
    ("r256", (8693, 0x0e8cc811c01e7273)),
    ("r257", (11721, 0x163e70cdb2530a5d)),
    ("r319", (11321, 0xdb1d0adc154e8404)),
    ("r320", (11213, 0x07b62e1449a1f250)),
    ("r448", (10029, 0x92c16f5b24fa450d)),
    ("r511", (13705, 0xe51260085e8e639d)),
    ("r512", (8797, 0x64a7184c2668ded3)),
    ("r513", (11857, 0xb232e9c53f0f9feb)),
    ("r960", (10153, 0x44f3723890c95f98)),
    ("r1023", (13441, 0x4ae552409f96d73f)),
    ("r1024", (9057, 0x4ef125ebd37d2943)),
    ("r1472", (10325, 0x8f3f64984cf7683e)),
    ("r1500", (13381, 0x2bc8fb64e425ac26)),
    ("r2048", (9657, 0x62d0058e4cd85e26)),
    ("r2049", (12925, 0xd45014d3397f20cb)),
    ("r2240", (11469, 0x4994bc7a10a003ea)),
    ("r2300", (13061, 0x076fb97809229151)),
    ("r2496", (10797, 0xd47960e2ef2355d7)),
    ("r2624", (11485, 0x188fd50db0589d2c)),
    ("r2700", (11821, 0x751cc4c5541a52e8)),
    ("r3000", (11645, 0x19eed17202d4e407)),
    ("r4095", (13013, 0x8f1f2556660e5d2e)),
    ("r4096", (10213, 0x2736a0f5e6c6d67f)),
    ("r4097", (13293, 0x58477aa5c4cba7ac)),
    ("r8191", (13157, 0x2e1c03d099938b35)),
    ("r8192", (10197, 0x2fdbfdaef8721e3a)),
    ("fresh-r512", (8789, 0xb9b7aa48fa4dcf1c)),
    ("fresh-r4097", (13157, 0x7ee1a221da5aec7b)),
    ("unscored-r8192", (9861, 0xa9b16f058d0b2289)),
    ("packed-1024-512", (10089, 0x9109ff2243aa5d0c)),
    ("packed-100-2000", (14721, 0xccdd4be6a0f3985c)),
    ("anchored-r789", (12289, 0x5a6bc485bffd7b37)),
];

const DSPARK_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0xa964c7d9849715d1)),
    ("r32", (9501, 0xf37c69c129a32da5)),
    ("r33", (8677, 0x09d68f9b10bdad9c)),
    ("r48", (8677, 0xc6871e1dd1af94d2)),
    ("r49", (8677, 0xc57499f5d86758c6)),
    ("r64", (8677, 0x3e7265117160369c)),
    ("r65", (11673, 0xeba2da793aec3aa9)),
    ("r112", (11337, 0x59467ffaf28d22a4)),
    ("r113", (11337, 0xb8a3ac44a648f206)),
    ("r127", (11337, 0xcf5efa083a2da332)),
    ("r128", (8677, 0x692c35333250a6d6)),
    ("r192", (10013, 0xed2144f60c0a6009)),
    ("r255", (13857, 0xc7f01fa260b2895c)),
    ("r256", (8709, 0xa7d91c8b7ab41d3b)),
    ("r257", (11737, 0x1083715c2fe3d86c)),
    ("r319", (11393, 0x3d313bd28a760f71)),
    ("r320", (11229, 0xee9975427fbe908b)),
    ("r448", (10045, 0x7dd4cd7e10a2316b)),
    ("r511", (13721, 0xabaf6ed08529dac1)),
    ("r512", (11115, 0x838ce50b7f241b49)),
    ("r513", (13919, 0x619c0565ec4e0cb9)),
    ("r960", (12471, 0xe8a5e43c95ebf56e)),
    ("r1023", (15503, 0x05bd8bd6e7a365e4)),
    ("r1024", (11375, 0xd637cccdbe11cc7d)),
    ("r1472", (12643, 0x95401e9a76e788b2)),
    ("r1500", (15443, 0xbac34d4a187327d2)),
    ("r2048", (11975, 0xe2412de9bf122c64)),
    ("r2049", (14987, 0x8233463c6e60cd84)),
    ("r2240", (15899, 0x2c5b027f64056f85)),
    ("r2300", (15123, 0x8b3d0810bb86d585)),
    ("r2496", (15083, 0xe3c1af3161d1ad8c)),
    ("r2624", (15723, 0xe379ab1422a27b20)),
    ("r2700", (15139, 0x7e3148042f0f6fbd)),
    ("r3000", (14963, 0x645fc98acee661f0)),
    ("r4095", (15075, 0x1f2f399804117d26)),
    ("r4096", (12627, 0xb7bcdd5c1fee9aa1)),
    ("r4097", (15355, 0xac347e7c5f4e33b8)),
    ("r8191", (15219, 0x94c6b5f696fc57b4)),
    ("r8192", (14755, 0xcec72e3aa80728e2)),
    ("fresh-r512", (11107, 0x198a34a6f5f37c4c)),
    ("fresh-r4097", (15219, 0xa798e12a211ef4d3)),
    ("unscored-r8192", (14419, 0xb8bd3b060f336517)),
    ("packed-1024-512", (10121, 0xf0ae07379ebed461)),
    ("packed-100-2000", (15665, 0x8e9ba038b4311cd2)),
    ("anchored-r789", (14351, 0x10ecea3800540b2e)),
];

const DSPARK: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0xa964c7d9849715d1)),
    ("r32", (9501, 0xf37c69c129a32da5)),
    ("r33", (8677, 0x09d68f9b10bdad9c)),
    ("r48", (8677, 0xc6871e1dd1af94d2)),
    ("r49", (8677, 0xc57499f5d86758c6)),
    ("r64", (8677, 0x3e7265117160369c)),
    ("r65", (11673, 0xeba2da793aec3aa9)),
    ("r112", (11337, 0x59467ffaf28d22a4)),
    ("r113", (11337, 0xb8a3ac44a648f206)),
    ("r127", (11337, 0xcf5efa083a2da332)),
    ("r128", (8677, 0x692c35333250a6d6)),
    ("r192", (10013, 0xed2144f60c0a6009)),
    ("r255", (13857, 0xc7f01fa260b2895c)),
    ("r256", (8709, 0xa7d91c8b7ab41d3b)),
    ("r257", (11737, 0x1083715c2fe3d86c)),
    ("r319", (11393, 0x3d313bd28a760f71)),
    ("r320", (11229, 0xee9975427fbe908b)),
    ("r448", (10045, 0x7dd4cd7e10a2316b)),
    ("r511", (13721, 0xabaf6ed08529dac1)),
    ("r512", (8813, 0x74a49a07605eb0ee)),
    ("r513", (11873, 0xa9e7f9080d7e8309)),
    ("r960", (10169, 0x6b10ed3262da7213)),
    ("r1023", (13457, 0x248c240f6efc14d6)),
    ("r1024", (9073, 0xc717f6bcc5fd509d)),
    ("r1472", (10341, 0x02625a3f635ef0d0)),
    ("r1500", (13397, 0x6a728003e569ec3e)),
    ("r2048", (9673, 0x02fd93053fcc6ce0)),
    ("r2049", (12941, 0xf1f4a6620b5f02c0)),
    ("r2240", (11485, 0xf69750e990b5e49c)),
    ("r2300", (13077, 0x5d145c62c10f2590)),
    ("r2496", (10813, 0xe2a1554114ea5730)),
    ("r2624", (11501, 0xc24d662afdf03765)),
    ("r2700", (13093, 0x6f8ed268a0accf0a)),
    ("r3000", (12917, 0xc3ea204b70f63cf1)),
    ("r4095", (13029, 0x4bd8c7bfa78b1829)),
    ("r4096", (10229, 0x3a6710b4aefbe91d)),
    ("r4097", (13309, 0x9884d4b47dc3d407)),
    ("r8191", (13173, 0xee143dad7c82cf38)),
    ("r8192", (10213, 0xebb148c3aa7e12c0)),
    ("fresh-r512", (8805, 0x3421755fa4bfc385)),
    ("fresh-r4097", (13173, 0x898fd514b9c81dc6)),
    ("unscored-r8192", (9877, 0x4247534c71a4c423)),
    ("packed-1024-512", (10121, 0xf0ae07379ebed461)),
    ("packed-100-2000", (15665, 0x8e9ba038b4311cd2)),
    ("anchored-r789", (12305, 0x4e78cf86528d5753)),
];
