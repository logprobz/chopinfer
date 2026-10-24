//! The block-FP8 checkpoint's derived outputs, pinned by digest: every
//! rank's views, direct layout and weight plan, the fit and the capacity
//! table. Taking a quantization layout must leave them byte-identical; the
//! digests were taken before the layout parameter existed.
use crate::capacity::{DEVICE, Planner};
use crate::contract::Shape;
use crate::direct::{layout, plan, test_support};
use crate::geometry::FLASH;
use crate::views::rank_views;
use sha2::{Digest, Sha256};

fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn outputs() -> Vec<(String, String)> {
    let shape = Shape::TP4;
    let mut out = Vec::new();
    for rank in 0..shape.ranks() {
        let views = rank_views(shape, rank).unwrap();
        out.push((format!("rank {rank} views"), digest(&format!("{views:?}"))));
        let stored = layout(shape, rank).unwrap();
        out.push((
            format!("rank {rank} layout"),
            digest(&format!("{stored:?}")),
        ));
        let planned = plan(&test_support::request(rank, |_| {})).unwrap();
        out.push((format!("rank {rank} plan"), digest(&format!("{planned:?}"))));
    }
    let fit = crate::fit::check(shape, DEVICE.bytes, 32, true, 4);
    out.push(("fit".into(), digest(&format!("{fit:?}"))));
    let weights = crate::fit::weight_bytes(&FLASH, 4).unwrap();
    out.push(("weight bytes".into(), weights.to_string()));
    let table = Planner::new(&FLASH, 4).unwrap().table(DEVICE);
    out.push(("capacity table".into(), digest(&table)));
    out
}

#[test]
fn fp8_outputs_are_unchanged() {
    let observed = outputs();
    // Each rank's layout and plan re-pinned when the lowered BF16
    // projections' weight views took their zero rows to the row tile
    // (`crate::precision::stored_rows`); the views, fit and capacity, which
    // count the checkpoint's shares, are unchanged. Each rank's plan again
    // when the arena alignment fell from 4096 to 256 (the pool guarantee):
    // the plans differ only in `alignment` and the rank index's
    // `direct_identity`, which hashes the plan; views and layout keep their
    // 4096-byte offsets. Every rank's views, layout and plan again when the
    // routed experts split over the ranks (each rank a quarter of every
    // expert's intermediate instead of 72 whole experts); the fit, weight
    // bytes and capacity table are unchanged. The capacity table again when
    // its F32 MoE all-reduce term took the complete state formulas (5.46 GiB
    // of workspace, 2,112 fewer KV tokens at three drafts and 32 slots), and
    // when it counted the state create allocates before the KV pool (one
    // copy per slot without drafts, and the sixteen anchor copies).
    let expected: &[(&str, &str)] = &[
        (
            "rank 0 views",
            "47975d5f6593ec61bd1901c1a1a8e77398ae973c781cd6d9ef9da886dfea5dfb",
        ),
        (
            "rank 0 layout",
            "aa781bb7c03c33cfa11742ce71afc53729c350248c5298c11885440c6e2517d6",
        ),
        (
            "rank 0 plan",
            "9e75d75f001a33c000b6330cbbea6e7c4448c0178e19e903a1b2ffa090bd5e66",
        ),
        (
            "rank 1 views",
            "28dd3787f02ce218a30133ad383ecc0137f8693660ceeaf73b3a38e780960e7a",
        ),
        (
            "rank 1 layout",
            "bc45d54159263e9c9160afdcf5cf201b667e158779b9180cf1221be6ce4c872c",
        ),
        (
            "rank 1 plan",
            "cd40fe2692a270708423f247e52601d83a623aae1398c7cf3bf2dd6e08de263c",
        ),
        (
            "rank 2 views",
            "6d7b1f29f1a94c969532a719c6b3e55306c597b4642ff63e80ab0ad435770998",
        ),
        (
            "rank 2 layout",
            "a80776ff39667694a5598d98d80652abd2c17623eae4731bca56e097301aec05",
        ),
        (
            "rank 2 plan",
            "42e191a93ea67c583629495d033413970361fb1e9a8ee969f9270f85e5ff7d39",
        ),
        (
            "rank 3 views",
            "6f186b370f3beccceb3a97ac56a561668195b1906a4cf3c08dcf1fe54f3eb2f7",
        ),
        (
            "rank 3 layout",
            "1b851dd53cdfa864425ae766eac8aff5e50325ab7e9a8a78dcbf1d8a19c373cb",
        ),
        (
            "rank 3 plan",
            "5c5438b168fe74e2271955c0c2e5085ac02331d88377e2ea052c4a555b9e7075",
        ),
        (
            "fit",
            "06b4d1977a948b64f7e27326969c250cdd623e4069e6098abc9c3ea8a4b5d150",
        ),
        ("weight bytes", "83370801144"),
        // Re-pinned when the capacity plan counted the context-parallel
        // group buffers (the DCP 2 and 4 columns and the title), when the
        // owner gathers cut the received blocks to `size`, and when text
        // packages left out the vision weights and image packages gained
        // the encoder term and their own columns, and when the split routed
        // experts' activated rows took a quarter of the memory and the MoE
        // sublayers' F32 all-reduce took its own state.
        (
            "capacity table",
            "cdfa5b0f96409cbc87571aef6cafa3f01571273d4f7fd72dfe60a65ef291f024",
        ),
    ];
    assert_eq!(
        observed
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect::<Vec<_>>(),
        expected
    );
}
