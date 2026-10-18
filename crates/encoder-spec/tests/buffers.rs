//! Buffer plan refusals: a stage never writes the region it reads.

use encoder_spec::buffers::{self, Serving};
use encoder_spec::{Environment, parse};
use serde_json::Value;

const QWEN: &str = include_str!("../../../models/qwen38/contract/encoders/qwen38-vision.json");

const SERVING: Serving = Serving {
    arena_rows: 64,
    prefill_rows: 64,
    staging_bytes: 0,
};

/// Qwen3.8's declaration with a width-preserving linear and a second
/// transformer after its transformer: the linear would read and write the
/// residual stream, so the plan refuses the declaration.
#[test]
fn a_linear_between_transformers_is_refused() {
    let mut value: Value = serde_json::from_str(QWEN).unwrap();
    let stages = value["stages"].as_array_mut().unwrap();
    let transformer = stages[2].clone();
    let linear = serde_json::json!({
        "op": "linear.pairs.v1", "out": 1152, "bias": false,
        "weights": {"weight": "blocks.0.attn.proj.weight"}
    });
    stages.splice(3..3, [linear, transformer]);
    let declaration = parse(&value.to_string(), &Environment::new("qwen38")).unwrap();
    let error = buffers::plan(&[&declaration], 4096, SERVING).unwrap_err();
    assert!(error.0.contains("stages[3]"), "{error}");
    assert!(
        error
            .0
            .contains("would write region `stream`, which it reads"),
        "{error}"
    );

    // The committed declaration plans, each stage writing elsewhere than it
    // reads.
    let committed = parse(QWEN, &Environment::new("qwen38")).unwrap();
    assert!(buffers::plan(&[&committed], 4096, SERVING).is_ok());
}

const DSV41: &str = include_str!("fixtures/dsv41-vision.json");

/// The space-to-depth merge's cells, and every region after it, hold the
/// cells its recipe admits (a budget of 1024 tokens holds at most 1021),
/// not one per patch row; its gather indices have a region of their own.
/// Cells of another side than the recipe's are bounded by the rows alone.
#[test]
fn space_to_depth_rows_follow_the_recipe_cells() {
    let declaration = parse(DSV41, &Environment::new("dsv41")).unwrap();
    let plan = buffers::plan(&[&declaration], 9216, SERVING).unwrap();
    let merge = |plan: &buffers::BufferPlan| {
        plan.regions
            .iter()
            .flat_map(|region| &region.uses)
            .find(|at| at.stage == Some(3))
            .unwrap()
            .bytes
    };
    assert_eq!(merge(&plan), 1021 * 9216 * 4);
    assert_eq!(
        plan.owned("gather-indices", Some(3)).unwrap().bytes,
        1021 * 9 * 4
    );
    // The aligner's F32 GELU rows, then its BF16 output rows.
    for at in plan.regions.iter().flat_map(|region| &region.uses) {
        match at.stage {
            Some(4) => assert_eq!(at.bytes, 1021 * 5120 * 4, "{at:?}"),
            Some(5) => assert_eq!(at.bytes, 1021 * 5120 * 2, "{at:?}"),
            _ => {}
        }
    }

    let mut value: Value = serde_json::from_str(DSV41).unwrap();
    value["input"]["recipes"][0]["downsample"] = 2.into();
    value["input"]["capacity_rows"] = 4096.into();
    let other = parse(&value.to_string(), &Environment::new("dsv41")).unwrap();
    let plan = buffers::plan(&[&other], 4096, SERVING).unwrap();
    assert_eq!(merge(&plan), 4096 * 9216 * 4);
}
