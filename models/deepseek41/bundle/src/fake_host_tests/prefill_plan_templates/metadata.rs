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
    ("r1472", (12946, 0x66b1ea91454a2ed6)),
    ("r1500", (16002, 0x96fda3eb9ba11696)),
    ("r2048", (12278, 0x12b0382689dcc58c)),
    ("r2049", (15546, 0xada9457eb28c65bf)),
    ("r2240", (16202, 0xcd99a87979a482c3)),
    ("r2300", (15682, 0xc1901800c9b31104)),
    ("r2496", (15386, 0xbbf5b35cebd88071)),
    ("r2624", (16026, 0x2a5a7e4659613f7b)),
    ("r2700", (14442, 0x7c48c5d092506bf5)),
    ("r3000", (14266, 0x9ec6562d4b06e00a)),
    ("r4095", (15634, 0x75022c7ec2969811)),
    ("r4096", (12930, 0x2c9aaf3366c4ecf7)),
    ("r4097", (15914, 0x093aeddcbda1b22b)),
    ("r8191", (15778, 0x1a5596a5e361098d)),
    ("r8192", (14738, 0xd7068a2b95507b86)),
    ("fresh-r512", (11410, 0x867511d65b6cfd5d)),
    ("fresh-r4097", (15778, 0x7d87db9d444b6656)),
    ("unscored-r8192", (14402, 0x12641a876f5675ef)),
    ("packed-1024-512", (10088, 0xcedaa9bf9e4d30a3)),
    ("packed-100-2000", (14720, 0x14ff2b09a03673c7)),
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
    ("r1472", (10324, 0xe350f4836a647965)),
    ("r1500", (13380, 0x8a6a84f4e03dc909)),
    ("r2048", (9656, 0x525e6e4bc11e0d17)),
    ("r2049", (12924, 0x3910711cdc960cd8)),
    ("r2240", (11468, 0x3bf4d8be876279bd)),
    ("r2300", (13060, 0xc9bd6fbf17cade2e)),
    ("r2496", (10796, 0x8ed59f45bec72a70)),
    ("r2624", (11484, 0xb27bd58be05e2dd3)),
    ("r2700", (11820, 0x6d9a02d54915897b)),
    ("r3000", (11644, 0xef5afb50c19cb384)),
    ("r4095", (13012, 0x2561a5c80da2f1a5)),
    ("r4096", (10212, 0xfc26e2e0c0364478)),
    ("r4097", (13292, 0x84b9ae8daef05f77)),
    ("r8191", (13156, 0xaa8bef7bcdafa336)),
    ("r8192", (10196, 0x5f4ee95a66b89c75)),
    ("fresh-r512", (8788, 0x3a38f87bf5243d5d)),
    ("fresh-r4097", (13156, 0xbd5fe9d7422ee47c)),
    ("unscored-r8192", (9860, 0x7521fbfbdcc50f86)),
    ("packed-1024-512", (10088, 0xcedaa9bf9e4d30a3)),
    ("packed-100-2000", (14720, 0x14ff2b09a03673c7)),
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
    ("r1472", (12947, 0x183b0be6e3207fd4)),
    ("r1500", (16003, 0x82249c9556770304)),
    ("r2048", (12279, 0x07943a5daa2cbc4e)),
    ("r2049", (15547, 0xfb2dbc26a7a83fdd)),
    ("r2240", (16203, 0xd13549900b4b4069)),
    ("r2300", (15683, 0xa54f6d021252d4e6)),
    ("r2496", (15387, 0x32b63d50735a059f)),
    ("r2624", (16027, 0xfbaa72227ed2ba75)),
    ("r2700", (14443, 0x474139f6d05f48d3)),
    ("r3000", (14267, 0xa60fb921c865b5a4)),
    ("r4095", (15635, 0x3d35067fbfc20da7)),
    ("r4096", (12931, 0xef9bb65cf45817e9)),
    ("r4097", (15915, 0x8ac07bfbe82e627d)),
    ("r8191", (15779, 0xa263cbbbb9f02b37)),
    ("r8192", (14739, 0x20330c92e57f28d4)),
    ("fresh-r512", (11411, 0x5e299abf7b903cef)),
    ("fresh-r4097", (15779, 0x739723df47e20628)),
    ("unscored-r8192", (14403, 0x4703f291a673f719)),
    ("packed-1024-512", (10089, 0x40c898ceb9d1cce8)),
    ("packed-100-2000", (14721, 0x820dd363c7d93350)),
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
    ("r1472", (10325, 0x6d9104d105e18d46)),
    ("r1500", (13381, 0xfd52126a6aba1da6)),
    ("r2048", (9657, 0x3f72311e1e86c448)),
    ("r2049", (12925, 0xbe2dc0604ae8f087)),
    ("r2240", (11469, 0x422ebda6f4b61516)),
    ("r2300", (13061, 0xdfc0ddb4d37361e9)),
    ("r2496", (10797, 0x468b2135e436facf)),
    ("r2624", (11485, 0x57f6fbf6cbe17ca4)),
    ("r2700", (11821, 0x0cba7c1ec20834e0)),
    ("r3000", (11645, 0xda59873c3c320417)),
    ("r4095", (13013, 0x37aeb9495b88760a)),
    ("r4096", (10213, 0x5afb7b9d8eb8779f)),
    ("r4097", (13293, 0x562aef2d3aba3034)),
    ("r8191", (13157, 0x66d37084bbe05f5d)),
    ("r8192", (10197, 0x2fdbfdaef8721e3a)),
    ("fresh-r512", (8789, 0x8838aaf51789b3a6)),
    ("fresh-r4097", (13157, 0x95dfeb45e3adc013)),
    ("unscored-r8192", (9861, 0xa9b16f058d0b2289)),
    ("packed-1024-512", (10089, 0x40c898ceb9d1cce8)),
    ("packed-100-2000", (14721, 0x820dd363c7d93350)),
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
    ("r1472", (12963, 0x994ba2d51f7b77a8)),
    ("r1500", (16019, 0x3ef5ca7b251d419a)),
    ("r2048", (12295, 0x08aeea6c45bdf2fc)),
    ("r2049", (15563, 0x5b3ec2b3645331e6)),
    ("r2240", (16219, 0xb7642ede9c76de4b)),
    ("r2300", (15699, 0x39cc6d50e64ab3d1)),
    ("r2496", (15403, 0x46cf7026754f346e)),
    ("r2624", (16043, 0x4e779da7b6da080e)),
    ("r2700", (15715, 0x1552968d774165dd)),
    ("r3000", (15539, 0xea91c1d50777f694)),
    ("r4095", (15651, 0x83c73b24084d0c28)),
    ("r4096", (12947, 0xc741951526ad3f0d)),
    ("r4097", (15931, 0x697722dc87659e7e)),
    ("r8191", (15795, 0x53ca1f008155520a)),
    ("r8192", (14755, 0xcec72e3aa80728e2)),
    ("fresh-r512", (11427, 0xb247ede5dcb1d1e4)),
    ("fresh-r4097", (15795, 0xd635d045560dc48d)),
    ("unscored-r8192", (14419, 0xb8bd3b060f336517)),
    ("packed-1024-512", (10121, 0xf43debe61f32f955)),
    ("packed-100-2000", (15665, 0xd490665532426f2c)),
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
    ("r1472", (10341, 0x777a2116ed1a99fa)),
    ("r1500", (13397, 0xeb2b73f1de75e14a)),
    ("r2048", (9673, 0x16d3627980f12fb8)),
    ("r2049", (12941, 0x6b7ef49465217742)),
    ("r2240", (11485, 0xb9b61f0b23be93d6)),
    ("r2300", (13077, 0x633872c6e4d7683c)),
    ("r2496", (10813, 0xe94afd25af869f0e)),
    ("r2624", (11501, 0xb2375e953fa2536b)),
    ("r2700", (13093, 0xd9b0ce045e7c9698)),
    ("r3000", (12917, 0x8ab5b3933242b96b)),
    ("r4095", (13029, 0xdcf716e658d70671)),
    ("r4096", (10229, 0xd8c7e45de4d6b019)),
    ("r4097", (13309, 0x3358d6c8303bbb05)),
    ("r8191", (13173, 0x27f27c4b1ec8d150)),
    ("r8192", (10213, 0xebb148c3aa7e12c0)),
    ("fresh-r512", (8805, 0xa01405b510ca573d)),
    ("fresh-r4097", (13173, 0x98b7fa4282593ccc)),
    ("unscored-r8192", (9877, 0x4247534c71a4c423)),
    ("packed-1024-512", (10121, 0xf43debe61f32f955)),
    ("packed-100-2000", (15665, 0xd490665532426f2c)),
    ("anchored-r789", (12305, 0xeccb20bed6918bf7)),
];
