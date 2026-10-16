//! Qwen3.8's vision declaration against the code that implements it: the
//! encoder image views, the checkpoint config and the frontend record. The
//! declaration states the F32 vision tower main serves, pair attention
//! included (`attention.varlen.pairs.v1`).

use encoder_spec::components::Item;
use encoder_spec::weights::TensorInfo;
use encoder_spec::{Declaration, Environment, parse};
use model_qwen38_contract::encoder::views;
use serde_json::{Value, json};

const VISION: &str = include_str!("../encoders/qwen38-vision.json");
const DIGEST: &str = "4238b4a6316aaefba6a0f627c138e0da90a9f92b3ebb3b8ab16bfcc6b396f74b";

fn vision() -> Declaration {
    parse(VISION, &Environment::new("qwen38")).unwrap()
}

#[test]
fn declaration_keeps_its_digest() {
    let declaration = vision();
    assert_eq!(declaration.digest(), DIGEST);
    assert_eq!(
        parse(&declaration.canonical_json(), &Environment::new("qwen38")).unwrap(),
        declaration
    );
}

fn lookup(dtype_of: impl Fn(&str) -> &'static str) -> impl Fn(&str) -> Option<TensorInfo> {
    let layout = views::layout_flash();
    move |name| {
        layout
            .iter()
            .find(|view| view.name == name)
            .map(|view| TensorInfo {
                dtype: dtype_of(name).into(),
                shape: view.source_shape.clone(),
                file: None,
            })
    }
}

#[test]
fn declaration_binds_to_the_image_views() {
    let declaration = vision();
    declaration.bind(&lookup(|_| "BF16")).unwrap();
    assert!(declaration.tensors().into_iter().eq({
        let mut names: Vec<String> = views::layout_flash()
            .into_iter()
            .map(|view| view.name)
            .collect();
        names.sort();
        names
    }));
}

/// Confirm review N2: `linear.pairs.v1` reads BF16 weights; no transform
/// turns I8 or F32 into them.
#[test]
fn weights_bind_with_the_element_the_kernel_reads() {
    for dtype in ["I8", "F32"] {
        let error = vision()
            .bind(&lookup(|name| {
                if name.ends_with("patch_embed.proj.weight") {
                    dtype
                } else {
                    "BF16"
                }
            }))
            .unwrap_err();
        assert!(
            error.0.contains(&format!(
                "weights.weight block 0: gives {dtype}, the role reads bf16"
            )),
            "{error}"
        );
    }
}

#[test]
fn pins_hold_against_the_checkpoint_config() {
    let config: Value =
        serde_json::from_str(include_str!("../../compiler/fixtures/config.json")).unwrap();
    vision().check_config("config.json", &config).unwrap();
}

#[test]
fn declaration_agrees_with_the_frontend() {
    let section: Value =
        serde_json::from_str(include_str!("../../compiler/src/frontend-multimodal.json")).unwrap();
    let frontend = &section["multimodal"];
    let declaration = vision();
    let record = &frontend["encoders"][0];
    assert_eq!(record["name"], declaration.name.as_str());
    assert_eq!(record["payload_columns"], declaration.input.columns);
    assert_eq!(
        frontend["arena"]["hidden"],
        declaration.output.op.params().hidden
    );
    let recipe = serde_json::to_value(&declaration.input.recipes[0]).unwrap();
    let parameters = &frontend["modalities"][0]["recipe"]["parameters"];
    for key in ["patch", "temporal_patch", "merge", "maximum_pixels"] {
        assert_eq!(recipe[key], parameters[key], "{key}");
    }
    assert_eq!(record["payload_element"], json!(declaration.input.payload));
}

#[test]
fn rows_follow_the_grid() {
    let trace = vision()
        .trace(Item {
            rows: 32 * 32,
            grid: Some([1, 32, 32]),
        })
        .unwrap();
    assert_eq!(trace.last().unwrap().rows, 16 * 16);
}

#[test]
fn each_geometry_uses_the_actual_encoder_buffer_plan() {
    use model_qwen38_contract::{contract::PREFILL_TOKENS, encoder, geometry::GEOMETRIES};
    for geometry in GEOMETRIES {
        let declarations = encoder::declarations_for(geometry).unwrap();
        let rows = u64::from(encoder::MAX_IMAGE_PIXELS / geometry.vision_patch().pow(2));
        let serving = encoder_spec::buffers::Serving {
            arena_rows: encoder::ARENA_ROWS,
            prefill_rows: u64::from(PREFILL_TOKENS),
            staging_bytes: 0,
        };
        let plan = encoder_spec::buffers::plan(&[&declarations[0]], rows, serving).unwrap();
        assert_eq!(
            encoder::buffer_bytes(geometry),
            plan.bytes(),
            "{}",
            geometry.id
        );
        assert_eq!(
            plan.region("stream").unwrap().bytes,
            rows * u64::from(geometry.vision_hidden()) * 4
        );
        assert_eq!(
            plan.region("a").unwrap().bytes,
            rows * u64::from(geometry.vision_hidden())
                .max(u64::from(geometry.vision_out()) / geometry.vision_merge().pow(2) as u64)
                * 4
        );
        assert!(
            plan.region("b").unwrap().bytes
                >= rows * u64::from(geometry.vision_intermediate().next_multiple_of(128)) * 4
        );
    }
}
