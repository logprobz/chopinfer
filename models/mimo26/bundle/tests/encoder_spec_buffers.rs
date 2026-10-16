//! The buffer plan MiMo's two encoder declarations derive, region by
//! region: the 17 regions the bundle allocates on every rank. Both run
//! f32-pairs.v1 in `stream`, `a` and `b`, and stage their side inputs, so
//! they share those regions.

use bundle_mimo26::encoders::runtime;

/// Each region and its bytes.
const REGIONS: [(&str, u64); 17] = [
    ("arena", 32_768 * 4096 * 2),
    ("payload", 16 * 32 * 32 * 1536 * 4),
    ("source-rows", 8192 * 8),
    ("stream", 8192 * 1280 * 4),
    ("a", 8192 * 4608 * 4),
    ("b", 8192 * 9216 * 4),
    ("im2col", 8192 * 3072 * 4),
    ("joined", 15_004 * 20 * 4),
    ("count", 4),
    ("carry", 8192 * 1024 * 4),
    ("gathered", 8192 * 20 * 4),
    ("cos", 8192 * 32 * 4),
    ("sin", 8192 * 32 * 4),
    ("reorder", 8192 * 4),
    ("sequences", 8193 * 4),
    ("codebook-offsets", 21 * 4),
    ("staging", 64 << 20),
];

#[test]
fn plan_holds_the_encoders_regions() {
    let plan = runtime::plan_for(
        &model_mimo26_contract::geometry::V26_FLASH,
        model_mimo26_contract::encoders::Encoders::ALL,
    )
    .unwrap();
    let mut names: Vec<&str> = plan.regions.iter().map(|region| region.name).collect();
    names.sort();
    let mut expected: Vec<&str> = REGIONS.iter().map(|(name, _)| *name).collect();
    expected.sort();
    assert_eq!(names, expected);
    for (name, bytes) in REGIONS {
        assert_eq!(plan.region(name).unwrap().bytes, bytes, "{name}");
    }
    assert_eq!(
        plan.bytes(),
        REGIONS.iter().map(|(_, bytes)| bytes).sum::<u64>()
    );
    // Both encoders share the row regions and the staged side inputs; the
    // widest use sizes each.
    let users = |name: &str| {
        let mut encoders: Vec<&str> = plan
            .region(name)
            .unwrap()
            .uses
            .iter()
            .map(|u| u.encoder.as_str())
            .collect();
        encoders.dedup();
        encoders
    };
    for name in [
        "payload",
        "stream",
        "a",
        "b",
        "cos",
        "sin",
        "sequences",
        "count",
    ] {
        assert_eq!(users(name), ["mimo-vision", "mimo-audio"], "{name}");
    }
    assert_eq!(users("reorder"), ["mimo-vision"]);
    for name in ["im2col", "carry", "joined", "gathered", "codebook-offsets"] {
        assert_eq!(users(name), ["mimo-audio"], "{name}");
    }
}
