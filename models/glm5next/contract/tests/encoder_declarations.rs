//! The vision encoder declaration (`encoders/glm5next-vision.json`): it
//! parses and keeps its digest, binds to every vision tensor of the
//! checkpoint, holds its pins against the checkpoint config, admits the
//! token-budget recipe's grids up to 8000 tokens, and plans its working
//! memory for one 8000-token image.
use encoder_spec::buffers::{self, Serving};
use encoder_spec::geometry::Media;
use encoder_spec::weights::TensorInfo;
use encoder_spec::{Declaration, Environment, parse};
use serde_json::Value;

const VISION: &str = include_str!("../encoders/glm5next-vision.json");
const DIGEST: &str = "ce4a85dec70ad3477baeef0742f02bc67de197da8d4682fb1ef0a0202277df28";
const TAG: &str = "image.zero-padded-token-budget.v1";

fn vision() -> Declaration {
    parse(VISION, &Environment::new("glm5next")).unwrap()
}

#[test]
fn declaration_keeps_its_digest() {
    assert_eq!(vision().digest(), DIGEST);
}

/// The checkpoint's vision tensors by name, from the tensor families.
fn vision_tensors() -> Vec<(String, TensorInfo)> {
    let families: Value =
        serde_json::from_str(include_str!("../fixtures/tensor-families.json")).unwrap();
    let mut tensors = Vec::new();
    for family in families.as_array().unwrap() {
        let name = family["name"].as_str().unwrap();
        let Some(name) = name.strip_prefix("model.visual.") else {
            continue;
        };
        let info = TensorInfo {
            dtype: family["dtype"].as_str().unwrap().into(),
            shape: family["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_u64().unwrap())
                .collect(),
            file: None,
        };
        if name.contains("{}") {
            for index in 0..family["count"].as_u64().unwrap() {
                tensors.push((name.replace("{}", &index.to_string()), info.clone()));
            }
        } else {
            tensors.push((name.to_owned(), info));
        }
    }
    tensors
}

#[test]
fn declaration_binds_every_vision_tensor() {
    let declaration = vision();
    let tensors = vision_tensors();
    declaration
        .bind(&|name: &str| {
            let name = name.strip_prefix("model.visual.").unwrap_or(name);
            tensors
                .iter()
                .find(|(known, _)| known == name)
                .map(|(_, info)| info.clone())
        })
        .unwrap();
    let mut names: Vec<String> = tensors
        .into_iter()
        .map(|(name, _)| format!("model.visual.{name}"))
        .collect();
    names.sort();
    let mapped: Vec<String> = declaration.tensors().into_iter().collect();
    assert_eq!(mapped, names);
    assert_eq!(mapped.len(), 24 * 14 + 11);
}

/// The 1176-wide patches are zero-padded to 1216 columns, a multiple of
/// the pair GEMM's 64-wide K steps, and the patch weight takes the same
/// zero columns; without them the weight does not bind.
#[test]
fn the_patch_gemm_reads_zero_padded_columns() {
    let declaration = vision();
    let inputs = declaration.inputs().unwrap();
    assert_eq!(inputs[0].width().unwrap(), 1176);
    assert_eq!(inputs[1].width().unwrap(), 1216);
    let mut value: Value = serde_json::from_str(VISION).unwrap();
    value["stages"][1]["weights"]["weight"]["transforms"] = serde_json::json!(["flatten"]);
    let unpadded = parse(&value.to_string(), &Environment::new("glm5next")).unwrap();
    let tensors = vision_tensors();
    let error = unpadded
        .bind(&|name: &str| {
            let name = name.strip_prefix("model.visual.").unwrap_or(name);
            tensors
                .iter()
                .find(|(known, _)| known == name)
                .map(|(_, info)| info.clone())
        })
        .unwrap_err();
    assert!(error.0.contains("the role reads [1024, 1216]"), "{error}");
}

#[test]
fn pins_hold_against_the_checkpoint_config() {
    let config: Value = serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
    vision().check_config("config.json", &config).unwrap();
}

#[test]
fn the_recipe_admits_grids_up_to_its_token_budget() {
    let declaration = vision();
    // 160 x 200 patches: 8000 merge blocks, the whole budget.
    let geometry = declaration
        .geometry(TAG, Media::Grid([1, 160, 200]))
        .unwrap();
    assert_eq!(geometry.input.rows, 32_000);
    assert_eq!(geometry.rows.last().unwrap().rows, 8000);
    assert_eq!(geometry.runs.len(), 1);
    for grid in [[1, 160, 202], [2, 8, 8], [1, 7, 8]] {
        assert!(
            declaration.geometry(TAG, Media::Grid(grid)).is_err(),
            "{grid:?}"
        );
    }
}

/// One 8000-token image's working memory (F32, no serving arena), in
/// bytes: `b` holds the transformer's gate and up rows (rows x 8192), `a`
/// the projector's (merged rows x 20480), `stream` the residual (rows x
/// 1024), and cos and sin rows x 32 each: rows x 57,600 plus 16 bytes of
/// sequence bounds and counts.
#[test]
fn one_image_plans_its_working_memory() {
    let declaration = vision();
    let serving = Serving {
        arena_rows: 0,
        prefill_rows: 0,
        staging_bytes: 0,
    };
    let plan = buffers::plan(&[&declaration], 32_000, serving).unwrap();
    assert_eq!(plan.bytes(), 32_000 * 57_600 + 16);
    let plan = buffers::plan(&[&declaration], 32_768, serving).unwrap();
    assert_eq!(plan.bytes(), 32_768 * 57_600 + 16);
}
