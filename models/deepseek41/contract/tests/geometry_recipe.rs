//! The same-architecture geometry recipe at the contract stage
//! (tests/model_geometry_recipe.py, docs/adding-a-model.md). With
//! `CHOPINFER_DEEPSEEK41_RECIPE_CONFIG` (a checkpoint `config.json`) and
//! `CHOPINFER_DEEPSEEK41_RECIPE_GEOMETRY` (the id the recipe added), the
//! config must select that geometry, divide over every supported rank count,
//! and changed copies must be refused by field before loading. Without them,
//! the synthetic fixture must be refused as an uncompiled geometry, unless
//! an extra geometry directory compiles it.
use model_deepseek41_contract::admission::{self, SUPPORTED_RANKS};
use model_deepseek41_contract::geometry::GEOMETRIES;
use serde_json::{Value, json};
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/deepseek41/geometries/synthetic-small")
}

fn read(path: &std::path::Path) -> Value {
    serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

fn set(config: &Value, path: &str, value: Value) -> Value {
    let mut changed = config.clone();
    let mut parent = &mut changed;
    let parts = path.split('.').collect::<Vec<_>>();
    for part in &parts[..parts.len() - 1] {
        parent = &mut parent[*part];
    }
    parent[parts[parts.len() - 1]] = value;
    changed
}

#[test]
fn every_compiled_geometry_divides_over_the_supported_rank_counts() {
    for geometry in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            admission::check_ranks(geometry, ranks)
                .unwrap_or_else(|e| panic!("{} TP{ranks}: {e}", geometry.id));
            println!("{} TP{ranks}: divides", geometry.id);
        }
    }
}

#[test]
fn the_recipe_geometry_is_admitted_and_its_changes_refused_by_field() {
    let added = std::env::var("CHOPINFER_DEEPSEEK41_RECIPE_GEOMETRY").ok();
    let config = match std::env::var_os("CHOPINFER_DEEPSEEK41_RECIPE_CONFIG") {
        Some(path) => read(std::path::Path::new(&path)),
        None => {
            let mut config = read(&fixture().join("config.json"));
            config["architectures"] = json!(["DeepseekV41ForCausalLM"]);
            config
        }
    };
    let compiled = GEOMETRIES.iter().find(|g| {
        serde_json::from_str::<Value>(g.document).unwrap()["text_config"]["hidden_size"]
            == config["text_config"]["hidden_size"]
    });
    let Some(expected) = added.as_deref().or(compiled.map(|g| g.id)) else {
        let error = admission::admit(&config).unwrap_err();
        assert!(error.starts_with("unsupported geometry: "), "{error}");
        println!("synthetic fixture refused as uncompiled: {error}");
        return;
    };
    let geometry = admission::admit(&config).unwrap_or_else(|e| panic!("{expected}: {e}"));
    assert_eq!(geometry.id, expected);
    for &ranks in SUPPORTED_RANKS {
        admission::check_ranks(geometry, ranks).unwrap();
        println!("{} TP{ranks}: admitted", geometry.id);
    }
    let refusals = [
        (
            "text_config.scoring_func",
            json!("softmax"),
            "unsupported text_config.scoring_func",
        ),
        (
            "text_config.compress_ratios",
            json!([0]),
            "unsupported text_config.compress_ratios",
        ),
        (
            "quantization_config.weight_block_size",
            json!([128, 128]),
            "unsupported quantization_config.weight_block_size",
        ),
        (
            "text_config.index_topk",
            json!(4096),
            "unsupported geometry: ",
        ),
    ];
    for (path, value, refusal) in refusals {
        let error = admission::admit(&set(&config, path, value)).unwrap_err();
        assert!(error.starts_with(refusal), "{path}: {error}");
    }
}
