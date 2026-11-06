//! Metadata pins of fresh upper-skip lowerings (`plan_timing::describe`):
//! every body and head launch with its collective flag and stage, the
//! index-selection sites, the DMA plan, the row copies and the Engram plan.
//! The shapes are single items across the route boundaries of the prefill
//! envelope, packed, anchored and unscored items, AR with device and
//! host-mapped Engram tables, and DSpark, each with DMA on and off. The
//! pins were recorded before the lowering became incremental and hold it
//! to the plans it made before. Each shape also runs as a step lowered
//! while it issues (`prefill_streamed_lowering`), whose plan holds the
//! same pin.
use super::*;
use crate::instance::execute::skip;
use model_deepseek41_contract::execution_options::{
    PREFILL_PLAN_CACHE_VERIFY, PREFILL_STREAMED_LOWERING,
};

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
/// tokens and request 2 slot 1 at 40,960, lowering each plan no template
/// serves while it issues when `streamed`.
fn restored(model: Model, prefix: u64, streamed: bool) -> BundleInstanceV1 {
    let host = host(matches!(model, Model::DSpark));
    if matches!(model, Model::ArMapped) {
        host.drafter.set(true);
    }
    let instance = super::super::prefill_skip::create_profile_with_options(
        &host,
        8192,
        true,
        &[
            (PREFILL_PLAN_CACHE_VERIFY, false.into()),
            (PREFILL_STREAMED_LOWERING, streamed.into()),
        ],
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
        let mut instance = restored(model, shape.prefix, true);
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
            let mut instance = restored(model, shape.prefix, false);
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
    ("r33", (9280, 0xcf0d2a8b7b16ed7a)),
    ("r48", (9280, 0x062d00b811dc9ef0)),
    ("r49", (9280, 0x8d4c8d725cadcd08)),
    ("r64", (8660, 0x916977b9b477b5cb)),
    ("r65", (12264, 0x07a67492f5cbd39e)),
    ("r112", (11940, 0x6d880039219fd254)),
    ("r113", (11940, 0x3fa06c62895ec05c)),
    ("r127", (11940, 0x67d3991dbf60dff2)),
    ("r128", (9268, 0x5e2da09d5016a53e)),
    ("r192", (9996, 0x65d7c08e74b2799f)),
    ("r255", (14460, 0xb0c336bc78cc958b)),
    ("r256", (9284, 0x820669ac13d23d56)),
    ("r257", (12328, 0x7dab303c720a0adc)),
    ("r319", (11940, 0xda64483f9173fdee)),
    ("r320", (11212, 0xe517e70dfda73f6f)),
    ("r448", (10028, 0xd075aa908ba0effa)),
    ("r511", (14324, 0x362237c6c1696977)),
    ("r512", (11654, 0xd85566a7ecc8376d)),
    ("r513", (14510, 0x439bb29e887030b3)),
    ("r960", (12454, 0xd36e0e84d6c67503)),
    ("r1023", (16106, 0x02365631aaedace0)),
    ("r1024", (11842, 0xffa62eb048c36a28)),
    ("r1472", (12626, 0xa4f282647a2c087a)),
    ("r1500", (16034, 0x8e54bfa6b6770156)),
    ("r2048", (12298, 0x4e19353307bac4f3)),
    ("r2049", (15578, 0x68050ea97b19c95b)),
    ("r2240", (15882, 0x4fc858a75d34411b)),
    ("r2300", (15726, 0xf999cea67ca246fd)),
    ("r2496", (15066, 0xdfa54a7b06536d21)),
    ("r2624", (15706, 0x5059abee4b249da7)),
    ("r2700", (14474, 0xaec934e64a7cd16f)),
    ("r3000", (14310, 0xc60e2b7aa9d03751)),
    ("r4095", (15678, 0xe1880f3f85e48564)),
    ("r4096", (12870, 0xc1aae1933e0d150a)),
    ("r4097", (15946, 0x8ffe599188357e75)),
    ("r8191", (15822, 0xade0cc4b2252f6c0)),
    ("r8192", (14738, 0xd7068a2b95507b86)),
    ("fresh-r512", (11646, 0xa3a096abd6898bd6)),
    ("fresh-r4097", (15810, 0x681b505d405b1a12)),
    ("unscored-r8192", (14402, 0x12641a876f5675ef)),
    ("packed-1024-512", (10572, 0xcfed5784864bce6a)),
    ("packed-100-2000", (15588, 0xfbc9e7fbee3de0fa)),
    ("anchored-r789", (14942, 0x9fbb3e39d996eb28)),
];

const AR: &[(&str, (usize, u64))] = &[
    ("r1", (9796, 0x15f9dfeaaeff9ea4)),
    ("r32", (9484, 0x2d9d1c06f3446c58)),
    ("r33", (9280, 0xcf0d2a8b7b16ed7a)),
    ("r48", (9280, 0x062d00b811dc9ef0)),
    ("r49", (9280, 0x8d4c8d725cadcd08)),
    ("r64", (8660, 0x916977b9b477b5cb)),
    ("r65", (12264, 0x07a67492f5cbd39e)),
    ("r112", (11940, 0x6d880039219fd254)),
    ("r113", (11940, 0x3fa06c62895ec05c)),
    ("r127", (11940, 0x67d3991dbf60dff2)),
    ("r128", (9268, 0x5e2da09d5016a53e)),
    ("r192", (9996, 0x65d7c08e74b2799f)),
    ("r255", (14460, 0xb0c336bc78cc958b)),
    ("r256", (9284, 0x820669ac13d23d56)),
    ("r257", (12328, 0x7dab303c720a0adc)),
    ("r319", (11940, 0xda64483f9173fdee)),
    ("r320", (11212, 0xe517e70dfda73f6f)),
    ("r448", (10028, 0xd075aa908ba0effa)),
    ("r511", (14324, 0x362237c6c1696977)),
    ("r512", (9352, 0xdeb0a6316274f0b1)),
    ("r513", (12464, 0x09e4f8d9de1b3f48)),
    ("r960", (10152, 0xea0a803417a7b25b)),
    ("r1023", (14060, 0x7050390e91693c5b)),
    ("r1024", (9540, 0xb06fd7c12a21cc05)),
    ("r1472", (10324, 0xc2eff399424cbe2d)),
    ("r1500", (13988, 0xc8a6ef92cb79edc5)),
    ("r2048", (9996, 0x735db5b4de0e8d2a)),
    ("r2049", (13532, 0x51f5a55f3b58f488)),
    ("r2240", (11468, 0xd57da2ac50544871)),
    ("r2300", (13680, 0x12b6fef686e1b38d)),
    ("r2496", (10796, 0x8f3baff9658d9968)),
    ("r2624", (11484, 0xf73b7c8f32077e8b)),
    ("r2700", (12428, 0x86bb2fdd3e9d4d91)),
    ("r3000", (12264, 0xb154119d09e722fd)),
    ("r4095", (13632, 0x2f4865afda1f5936)),
    ("r4096", (10472, 0x4047cf7e5bdc21ab)),
    ("r4097", (13900, 0xfde0ecc65d05892d)),
    ("r8191", (13776, 0x79a8543aaa5639f1)),
    ("r8192", (10196, 0x5f4ee95a66b89c75)),
    ("fresh-r512", (9344, 0x4ff9d3cab8aae01c)),
    ("fresh-r4097", (13764, 0x60e79d096edfbbc8)),
    ("unscored-r8192", (9860, 0x7521fbfbdcc50f86)),
    ("packed-1024-512", (10572, 0xcfed5784864bce6a)),
    ("packed-100-2000", (15588, 0xfbc9e7fbee3de0fa)),
    ("anchored-r789", (12896, 0x993f3a247918eec2)),
];

const MAPPED_AR_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x2c62d696f5e2a507)),
    ("r32", (9485, 0x70bfd9a95917780f)),
    ("r33", (9281, 0x305731966d5d6301)),
    ("r48", (9281, 0xa6b36e719f660c87)),
    ("r49", (9281, 0xab59d38643ee8d6f)),
    ("r64", (8661, 0x82180d6de8b60d1c)),
    ("r65", (12265, 0x06bcc4789c890401)),
    ("r112", (11941, 0x62c2f0e122b597b7)),
    ("r113", (11941, 0xe3d92c299b8628ef)),
    ("r127", (11941, 0xbb7c3b4aa6aa2b95)),
    ("r128", (9269, 0xb1c428c8c8967899)),
    ("r192", (9997, 0x2dec726557edd22c)),
    ("r255", (14461, 0xd231497a944a31fc)),
    ("r256", (9285, 0x97db463f45af3ac1)),
    ("r257", (12329, 0xe93c49272961b303)),
    ("r319", (11941, 0x2fd7f35e9ed83f01)),
    ("r320", (11213, 0x07b62e1449a1f250)),
    ("r448", (10029, 0x92c16f5b24fa450d)),
    ("r511", (14325, 0xa7cb0ff183524b7c)),
    ("r512", (11655, 0xa57bed1bb5a3b01f)),
    ("r513", (14511, 0x0bcb3a8bbe49084d)),
    ("r960", (12455, 0x4a0b8e434a8df2dd)),
    ("r1023", (16107, 0x61e83be3bc88863e)),
    ("r1024", (11843, 0x81de6c8ea2104e52)),
    ("r1472", (12627, 0xfb4711d0736b6ebc)),
    ("r1500", (16035, 0x2eb36a3166fab5f4)),
    ("r2048", (12299, 0xf85fea2b29f47de5)),
    ("r2049", (15579, 0xa03085a0e37b274d)),
    ("r2240", (15883, 0x0b603752c1766695)),
    ("r2300", (15727, 0xa15c8acef9dbc54b)),
    ("r2496", (15067, 0x6d2e672f5960d9df)),
    ("r2624", (15707, 0x4d7c14d9e282c965)),
    ("r2700", (14475, 0xf34f29f2f46a1061)),
    ("r3000", (14311, 0x8ec0bee600da401b)),
    ("r4095", (15679, 0x3cb74a068be85f32)),
    ("r4096", (12871, 0xc6122691a8224c44)),
    ("r4097", (15947, 0x266176029c400b17)),
    ("r8191", (15823, 0x95b6801732aba0f6)),
    ("r8192", (14739, 0x20330c92e57f28d4)),
    ("fresh-r512", (11647, 0x1199fe3af7c2369c)),
    ("fresh-r4097", (15811, 0xef2b143174f966b4)),
    ("unscored-r8192", (14403, 0x4703f291a673f719)),
    ("packed-1024-512", (10573, 0x0bfb340e6c0a79c1)),
    ("packed-100-2000", (15589, 0xbbc3afa5801f63c9)),
    ("anchored-r789", (14943, 0x7786580c21bcf65e)),
];

const MAPPED_AR: &[(&str, (usize, u64))] = &[
    ("r1", (9797, 0x2c62d696f5e2a507)),
    ("r32", (9485, 0x70bfd9a95917780f)),
    ("r33", (9281, 0x305731966d5d6301)),
    ("r48", (9281, 0xa6b36e719f660c87)),
    ("r49", (9281, 0xab59d38643ee8d6f)),
    ("r64", (8661, 0x82180d6de8b60d1c)),
    ("r65", (12265, 0x06bcc4789c890401)),
    ("r112", (11941, 0x62c2f0e122b597b7)),
    ("r113", (11941, 0xe3d92c299b8628ef)),
    ("r127", (11941, 0xbb7c3b4aa6aa2b95)),
    ("r128", (9269, 0xb1c428c8c8967899)),
    ("r192", (9997, 0x2dec726557edd22c)),
    ("r255", (14461, 0xd231497a944a31fc)),
    ("r256", (9285, 0x97db463f45af3ac1)),
    ("r257", (12329, 0xe93c49272961b303)),
    ("r319", (11941, 0x2fd7f35e9ed83f01)),
    ("r320", (11213, 0x07b62e1449a1f250)),
    ("r448", (10029, 0x92c16f5b24fa450d)),
    ("r511", (14325, 0xa7cb0ff183524b7c)),
    ("r512", (9353, 0x66b9703438176176)),
    ("r513", (12465, 0xf8dcde5c35c13ceb)),
    ("r960", (10153, 0x44f3723890c95f98)),
    ("r1023", (14061, 0xda062d24abf69258)),
    ("r1024", (9541, 0xe2fdef3ab9823ebe)),
    ("r1472", (10325, 0x8f3f64984cf7683e)),
    ("r1500", (13989, 0xa49f671a2452f15e)),
    ("r2048", (9997, 0x16b2f6c3084a3d81)),
    ("r2049", (13533, 0xb997bf53ec748faf)),
    ("r2240", (11469, 0x4994bc7a10a003ea)),
    ("r2300", (13681, 0x5872d99166dcfe5a)),
    ("r2496", (10797, 0xd47960e2ef2355d7)),
    ("r2624", (11485, 0x188fd50db0589d2c)),
    ("r2700", (12429, 0xc6abc0cf3cf10a96)),
    ("r3000", (12265, 0x0c57e1ae08242e06)),
    ("r4095", (13633, 0x0ccaef6a769e73fd)),
    ("r4096", (10473, 0x934821078fdb193c)),
    ("r4097", (13901, 0x8eb3a0d9d6c62f4a)),
    ("r8191", (13777, 0xf2300e02ccae5af6)),
    ("r8192", (10197, 0x2fdbfdaef8721e3a)),
    ("fresh-r512", (9345, 0x85000e54980b32f3)),
    ("fresh-r4097", (13765, 0xdc03621aa00beca3)),
    ("unscored-r8192", (9861, 0xa9b16f058d0b2289)),
    ("packed-1024-512", (10573, 0x0bfb340e6c0a79c1)),
    ("packed-100-2000", (15589, 0xbbc3afa5801f63c9)),
    ("anchored-r789", (12897, 0x39d57c59309f5f91)),
];

const DSPARK_DMA: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0xa964c7d9849715d1)),
    ("r32", (9501, 0xf37c69c129a32da5)),
    ("r33", (9297, 0xecf2b1ce3598807a)),
    ("r48", (9297, 0xbc32702d7e43e314)),
    ("r49", (9297, 0x7dc509636573b7c8)),
    ("r64", (8677, 0x3e7265117160369c)),
    ("r65", (12281, 0xf0f46b0d36c8a1ea)),
    ("r112", (11957, 0x2a38bc487ba8f556)),
    ("r113", (11957, 0x181996e8831656ca)),
    ("r127", (11957, 0x614702d73732d212)),
    ("r128", (9285, 0x26c00aa04213aecf)),
    ("r192", (10013, 0xed2144f60c0a6009)),
    ("r255", (14477, 0x0816a9168cdf42ce)),
    ("r256", (9301, 0xcf8faab025898e16)),
    ("r257", (12345, 0xf36d43adfd3107c9)),
    ("r319", (12013, 0xca7852458d21e6c7)),
    ("r320", (11229, 0xee9975427fbe908b)),
    ("r448", (10045, 0x7dd4cd7e10a2316b)),
    ("r511", (14341, 0xd0b23ffb7223421b)),
    ("r512", (11671, 0x46c2dd1499ec6dc7)),
    ("r513", (14527, 0xf0f8c12992f18a78)),
    ("r960", (12471, 0xe8a5e43c95ebf56e)),
    ("r1023", (16123, 0x8b6c2d44afb42fe4)),
    ("r1024", (11859, 0xf77e9d34805069c5)),
    ("r1472", (12643, 0x95401e9a76e788b2)),
    ("r1500", (16051, 0x9f4ab3c1e70c812b)),
    ("r2048", (12315, 0x0d36d503f470c2f0)),
    ("r2049", (15595, 0xc362ec4d9268b9d5)),
    ("r2240", (15899, 0x2c5b027f64056f85)),
    ("r2300", (15743, 0xd8bd51b02ab2b6f3)),
    ("r2496", (15083, 0xe3c1af3161d1ad8c)),
    ("r2624", (15723, 0xe379ab1422a27b20)),
    ("r2700", (15747, 0x12e3870e09ca5b6e)),
    ("r3000", (15583, 0xe5bd4fedc78865ea)),
    ("r4095", (15695, 0x280f1beb9fa433e8)),
    ("r4096", (12887, 0x6eb50a05817de4c7)),
    ("r4097", (15963, 0xe052ac0432302a79)),
    ("r8191", (15839, 0xe796a448bf9d7ee6)),
    ("r8192", (14755, 0xcec72e3aa80728e2)),
    ("fresh-r512", (11663, 0x2e587423f298c58a)),
    ("fresh-r4097", (15827, 0xa4e4ef3302fc591a)),
    ("unscored-r8192", (14419, 0xb8bd3b060f336517)),
    ("packed-1024-512", (10605, 0x0c2696a1281c7983)),
    ("packed-100-2000", (16533, 0x1d926122e1eeb568)),
    ("anchored-r789", (14959, 0xf0a8edd28d5f5091)),
];

const DSPARK: &[(&str, (usize, u64))] = &[
    ("r1", (9813, 0xa964c7d9849715d1)),
    ("r32", (9501, 0xf37c69c129a32da5)),
    ("r33", (9297, 0xecf2b1ce3598807a)),
    ("r48", (9297, 0xbc32702d7e43e314)),
    ("r49", (9297, 0x7dc509636573b7c8)),
    ("r64", (8677, 0x3e7265117160369c)),
    ("r65", (12281, 0xf0f46b0d36c8a1ea)),
    ("r112", (11957, 0x2a38bc487ba8f556)),
    ("r113", (11957, 0x181996e8831656ca)),
    ("r127", (11957, 0x614702d73732d212)),
    ("r128", (9285, 0x26c00aa04213aecf)),
    ("r192", (10013, 0xed2144f60c0a6009)),
    ("r255", (14477, 0x0816a9168cdf42ce)),
    ("r256", (9301, 0xcf8faab025898e16)),
    ("r257", (12345, 0xf36d43adfd3107c9)),
    ("r319", (12013, 0xca7852458d21e6c7)),
    ("r320", (11229, 0xee9975427fbe908b)),
    ("r448", (10045, 0x7dd4cd7e10a2316b)),
    ("r511", (14341, 0xd0b23ffb7223421b)),
    ("r512", (9369, 0xd3a7ed97cffe38bc)),
    ("r513", (12481, 0x4cb2b4e38936cdf2)),
    ("r960", (10169, 0x6b10ed3262da7213)),
    ("r1023", (14077, 0x8a07b8b2ab1935ae)),
    ("r1024", (9557, 0x4c054e885c8be29d)),
    ("r1472", (10341, 0x02625a3f635ef0d0)),
    ("r1500", (14005, 0xf53fec8f9e44f6b3)),
    ("r2048", (10013, 0xec83d8848e0d1a04)),
    ("r2049", (13549, 0x5968cd049f766377)),
    ("r2240", (11485, 0xf69750e990b5e49c)),
    ("r2300", (13697, 0xb25bf56e56aa771e)),
    ("r2496", (10813, 0xe2a1554114ea5730)),
    ("r2624", (11501, 0xc24d662afdf03765)),
    ("r2700", (13701, 0xa242cc2cae616e2b)),
    ("r3000", (13537, 0xdeeac0527ddf0677)),
    ("r4095", (13649, 0x232f16e3603210ff)),
    ("r4096", (10489, 0x9e0326bcf60d9ba3)),
    ("r4097", (13917, 0x1f665604fb5cd360)),
    ("r8191", (13793, 0xdb775f027f2b451a)),
    ("r8192", (10213, 0xebb148c3aa7e12c0)),
    ("fresh-r512", (9361, 0x15681171a5bd126b)),
    ("fresh-r4097", (13781, 0xf1068465a9cbc32d)),
    ("unscored-r8192", (9877, 0x4247534c71a4c423)),
    ("packed-1024-512", (10605, 0x0c2696a1281c7983)),
    ("packed-100-2000", (16533, 0x1d926122e1eeb568)),
    ("anchored-r789", (12913, 0xc3781fe32d078204)),
];
