//! The buffer plan `encoder-spec` derives from Qwen3.8's vision declaration,
//! which the shared encoder runtime allocates on every rank, against the
//! buffers the bundle's own F32 tower allocated before it: every live
//! allocation by name and length, their total (the contract's
//! `BUFFER_BYTES`), and the aliasing of that tower.

use bundle_qwen38::contract::PREFILL_TOKENS;
use bundle_qwen38::encoder::{ARENA_ROWS, BUFFER_BYTES, VIT_ROWS};
use encoder_spec::Environment;
use encoder_spec::buffers::{self, Serving};

const VISION: &str = include_str!("../../contract/encoders/qwen38-vision.json");

#[test]
fn plan_matches_the_towers_buffers() {
    let vision = encoder_spec::parse(VISION, &Environment::new("qwen38")).unwrap();
    let serving = Serving {
        arena_rows: ARENA_ROWS,
        prefill_rows: u64::from(PREFILL_TOKENS),
        staging_bytes: 0,
    };
    let plan = buffers::plan(&[&vision], u64::from(VIT_ROWS), serving).unwrap();
    // The tower's allocations (P = 65,536 patches, widths in F32 columns):
    // `hidden` [P, 1152], `norm` [P, 1152], `wide` [P, 4352], cos and sin
    // [P, 36], the four position taps' indices and weights, two i32
    // sequence bounds, two u32 counts, the i64 prefill source rows and the
    // arena of 65,536 rows of 2560 BF16.
    let p = u64::from(VIT_ROWS);
    let allocations = [
        ("stream", p * 1152 * 4),
        ("a", p * 1152 * 4),
        ("b", p * 4352 * 4),
        ("cos", p * 36 * 4),
        ("sin", p * 36 * 4),
        ("tap-indices", p * 4 * 4),
        ("tap-weights", p * 4 * 4),
        ("sequences", 8),
        ("counts", 8),
        ("source-rows", 8192 * 8),
        ("arena", 65_536 * 2560 * 2),
    ];
    for (name, length) in allocations {
        assert_eq!(plan.region(name).unwrap().bytes, length, "{name}");
    }
    let mut planned: Vec<&str> = allocations.iter().map(|(name, _)| *name).collect();
    let mut names: Vec<&str> = plan.regions.iter().map(|region| region.name).collect();
    planned.sort();
    names.sort();
    assert_eq!(planned, names);
    assert_eq!(plan.bytes(), BUFFER_BYTES);
    assert_eq!(BUFFER_BYTES, 2_101_411_856);

    // The tower's aliasing: `hidden` the residual stream; `norm` each
    // LayerNorm output, the attention output and the merged BF16 rows;
    // `wide` the payload, qkv and its pair rows, fc1 and merger fc1.
    let uses = |name: &str| -> Vec<(Option<usize>, String)> {
        plan.region(name)
            .unwrap()
            .uses
            .iter()
            .map(|u| (u.stage, u.what.clone()))
            .collect()
    };
    let at = |stage: usize, what: &str| (Some(stage), what.to_owned());
    assert_eq!(
        uses("stream"),
        [at(0, "linear.pairs.v1"), at(2, "residual")]
    );
    assert_eq!(
        uses("a"),
        [
            at(2, "norm"),
            at(2, "attention"),
            at(3, "norm.layer-f32.v1"),
            at(6, "linear.pairs.v1")
        ]
    );
    assert_eq!(
        uses("b"),
        [
            (None, "payload".to_owned()),
            at(2, "qkv, then its pair rows"),
            at(2, "mlp"),
            at(5, "linear.pairs.v1")
        ]
    );
}
