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
