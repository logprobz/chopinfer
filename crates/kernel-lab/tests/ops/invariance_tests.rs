//! The generated invariance tests of the `ops` adapters (see
//! `common/invariance.rs`): one GPU test per property over every declared
//! entry of the adapters' op kinds.

use crate::invariance::{
    ADAPTERS, Adapter, Element, inventory, run, uncovered, written, written_over,
};
use kernel_core::contract::DType;
use kernel_lab::gpu_test;
use kernel_lab::poison::{POISON, WriteMap};
use kernels_sm120::invariance::Property;

/// The adapters of this test binary, one per op kind.
fn adapters() -> Vec<Box<dyn Adapter>> {
    vec![
        Box::new(crate::gemm::invariance::Gemm::default()),
        Box::new(crate::attention::invariance::Attention),
        Box::new(crate::bundle_invariance::Bundle::default()),
        Box::new(crate::context_parallel_invariance::ContextParallel),
        Box::new(crate::linear::invariance::Linear),
        Box::new(crate::moe::invariance::Moe::default()),
        Box::new(crate::row::invariance::Row),
        Box::new(crate::sparse_attention::invariance::Sparse),
        Box::new(crate::prepared_invariance::Prepared(
            crate::sinkhorn_invariance::Sinkhorn,
        )),
        Box::new(crate::prepared_invariance::Prepared(
            crate::latent_invariance::Latent,
        )),
        Box::new(crate::pooled_invariance::Pooled::default()),
    ]
}

/// Every declared claim of every catalog entry has an adapter, and this
/// binary holds the adapters `ADAPTERS` places in it. A catalog with
/// profiles declares entries; one without contributes declarations or a
/// projection inventory.
#[test]
fn invariance_claims_have_adapters() {
    // The inventory holds every model's entries and extra declarations.
    let symbols = inventory()
        .into_iter()
        .map(|d| d.symbol)
        .collect::<std::collections::BTreeSet<_>>();
    let models = kernel_lab::catalogs().models();
    assert_eq!(models.len(), kernel_lab::MODELS.len());
    for model in models {
        let declared = (model.gemm)()
            .into_iter()
            .map(|v| v.symbol.clone())
            .chain((model.gemm_fused)().into_iter().map(|e| e.symbol))
            .chain((model.moe)().into_iter().map(|v| v.symbol))
            .chain((model.moe_routes)().into_iter().map(|e| e.symbol))
            .chain((model.gated)().into_iter().map(|e| e.symbol))
            .chain(
                (model.attention)()
                    .into_iter()
                    .flat_map(|v| v.declared_symbols().into_iter().map(|(s, _)| s)),
            )
            .chain((model.sparse)().into_iter().map(|e| e.symbol()))
            .chain((model.declared)().into_iter().map(|d| d.symbol))
            .collect::<Vec<_>>();
        if model.profiles.is_empty() {
            // A catalog without profiles compiles no per-profile cubin: it
            // contributes its projection inventory, which the prepared
            // projection jobs check, and any declarations it has.
            let (offers, _, copies) = (model.projections)().unwrap();
            assert!(
                !declared.is_empty() || !offers.is_empty() || !copies.is_empty(),
                "{}",
                model.name
            );
        } else {
            assert!(!declared.is_empty(), "{}", model.name);
        }
        for symbol in declared {
            assert!(symbols.contains(&symbol), "{}: {symbol}", model.name);
        }
    }
    let missing = uncovered();
    assert!(
        missing.is_empty(),
        "claims without an adapter:\n{}",
        missing.join("\n")
    );
    let mut placed = ADAPTERS
        .iter()
        .filter(|(_, _, binary)| *binary == "ops")
        .map(|(kind, ..)| *kind)
        .collect::<Vec<_>>();
    let mut held = adapters().iter().map(|a| a.kind()).collect::<Vec<_>>();
    placed.sort();
    held.sort();
    assert_eq!(placed, held);
}

/// The GPU handoff records this inventory before reserving a device, then
/// requires a successful receipt for every selected symbol and property.
#[test]
fn lab_shape_invariance_inventory() {
    let entries: Vec<_> = inventory()
        .into_iter()
        .filter(|e| e.symbol.contains("lab_shapes_") || e.symbol.contains("_block_topk"))
        .collect();
    assert!(!entries.is_empty());
    for entry in entries {
        for property in entry.invariance.claims() {
            println!(
                "invariance-claim {}",
                serde_json::json!({
                    "symbol": entry.symbol,
                    "property": format!("{property:?}"),
                })
            );
        }
    }
}

/// The shared write checks reject a region the launch left poisoned,
/// written non-finite, or written outside its map, which the bitwise
/// comparison alone accepts when both arms share them (no GPU).
#[test]
fn write_checks_reject_what_equal_bytes_accept() {
    let one = 0x3f80_u16.to_ne_bytes();
    // Four BF16 elements, the first three live.
    let map = WriteMap::of(DType::Bf16).prefix(3).unwrap();
    let mut good = vec![POISON; 8];
    good[..6].copy_from_slice(&one.repeat(3));
    assert!(written("out", &good, &map, Element::Bf16).is_ok());
    // An arm that wrote nothing equals another that wrote nothing.
    let unwritten = vec![POISON; 8];
    assert!(written("out", &unwritten, &map, Element::Bf16).is_err());
    let mut hole = good.clone();
    hole[2..4].fill(POISON);
    assert!(written("out", &hole, &map, Element::Bf16).is_err());
    let mut nan = good.clone();
    nan[..2].copy_from_slice(&0x7fc0_u16.to_ne_bytes());
    assert!(written("out", &nan, &map, Element::Bf16).is_err());
    let mut padding = good.clone();
    padding[6..8].copy_from_slice(&one);
    assert!(written("out", &padding, &map, Element::Bf16).is_err());
    // A map whose elements are not the buffer's.
    assert!(written("out", &good, &map, Element::F32).is_err());
    // Raw IDs: any value but the poison counts as written.
    let ids = WriteMap::of(DType::I32).prefix(1).unwrap();
    let mut id = vec![POISON; 8];
    id[..4].copy_from_slice(&7_u32.to_ne_bytes());
    assert!(written("ids", &id, &ids, Element::Raw(4)).is_ok());
    assert!(written("ids", &[POISON; 8], &ids, Element::Raw(4)).is_err());

    // In place: two F32 cache rows, the second poisoned for the launch.
    let mut before = 1.0_f32.to_ne_bytes().repeat(2);
    before.extend([POISON; 8]);
    let map = WriteMap::of(DType::F32).rows(2, [1]).unwrap();
    let mut after = before.clone();
    after[8..].copy_from_slice(&2.0_f32.to_ne_bytes().repeat(2));
    assert!(written_over("cache", &before, &after, &map, Element::F32).is_ok());
    assert!(written_over("cache", &before, &before, &map, Element::F32).is_err());
    let mut clobbered = after.clone();
    clobbered[..4].copy_from_slice(&3.0_f32.to_ne_bytes());
    assert!(written_over("cache", &before, &clobbered, &map, Element::F32).is_err());
    // A row that was never poisoned proves nothing.
    assert!(written_over("cache", &after, &after, &map, Element::F32).is_err());
}

fn harness(property: Property) {
    gpu_test(8 << 30, |session| {
        run(property, &mut adapters(), std::slice::from_ref(session))
    })
    .unwrap_or_else(|error| panic!("{property:?} invariance: {error}"));
}

#[test]
fn invariance_chunk() {
    harness(Property::Chunk);
}

#[test]
fn invariance_rows() {
    harness(Property::Rows);
}

#[test]
fn invariance_batch() {
    harness(Property::Batch);
}

#[test]
fn invariance_ties() {
    harness(Property::Ties);
}

#[test]
fn invariance_anchors() {
    harness(Property::Anchors);
}
