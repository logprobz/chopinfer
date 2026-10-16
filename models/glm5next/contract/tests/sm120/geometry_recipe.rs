//! The same-architecture geometry recipe at the contract stage
//! (tests/model_geometry_recipe.py, docs/adding-a-model.md). Every compiled
//! geometry gets kernel profiles, a catalog, a TP4 weight plan, tuning
//! decisions and a capacity table from its data alone. With
//! `CHOPINFER_GLM5NEXT_RECIPE_CONFIG` (a checkpoint `config.json`) and
//! `CHOPINFER_GLM5NEXT_RECIPE_GEOMETRY` (the id the recipe added), the
//! config must select that geometry and changed copies must be refused by
//! field before loading. Without them, the synthetic fixture must be refused
//! as an uncompiled geometry, unless an extra geometry directory compiles it.
//! Every compiled geometry and the synthetic fixture must also give per-rank
//! shapes the SM120 latent attention and KDA engine planners accept.
use model_glm5next_contract::capacity::{DEVICE, Planner, Serving};
use model_glm5next_contract::geometry::{GEOMETRIES, Geometry};
use model_glm5next_contract::kernels::{KernelProfile, SERVED_RANKS};
use model_glm5next_contract::sm120::tuning;
use model_glm5next_contract::{admission, catalog, structure, weight_plan};
use serde_json::Value;
use std::path::PathBuf;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/glm5next/geometries/synthetic-small")
}

fn read(path: &std::path::Path) -> Value {
    serde_json::from_slice(
        &std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

/// `base` with every leaf of `overlay` written over it.
fn overlay(base: &mut Value, overlay_value: &Value) {
    if base.is_object() && overlay_value.is_object() {
        let base = base.as_object_mut().unwrap();
        for (key, value) in overlay_value.as_object().unwrap() {
            overlay(base.entry(key.clone()).or_insert(Value::Null), value);
        }
    } else {
        *base = overlay_value.clone();
    }
}

/// The pinned checkpoint config with `g`'s execution fields.
fn config_of(g: &Geometry) -> Value {
    let mut config: Value =
        serde_json::from_slice(include_bytes!("../../fixtures/config.json")).unwrap();
    overlay(&mut config, &serde_json::from_str(g.document).unwrap());
    config
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

fn number(config: &Value, path: &str) -> u32 {
    let value = path.split('.').fold(config, |value, key| &value[key]);
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or_else(|| panic!("{path}: {value}"))
}

/// Refusals of the latent attention and KDA engine entries at a config's
/// per-rank shapes, as `<engine> <entry>: <refusal>`.
fn engine_refusals(config: &Value, ranks: u32) -> Vec<String> {
    use kernel_core::per_channel_decay_delta_rule::{DecayGate, QkvConvs};
    use kernels_sm120::{Profile, ProfileNames, latent, linear::channel_decay};
    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "geometry-recipe",
        tag: "geometry_recipe",
    });
    let text = |path: &str| number(config, &format!("text_config.{path}"));
    let float = |path: &str| config["text_config"][path].as_f64().unwrap() as f32;
    let latent = latent::Geometry {
        heads: text("num_attention_heads") / ranks,
        latent: text("kv_lora_rank"),
        page_size: model_glm5next_contract::capacity::PAGE_TOKENS,
        max_keys: text("index_topk") + text("index_kpool") - 1,
        softmax_scale: (text("qk_head_dim") as f32).sqrt().recip(),
        nope: text("qk_nope_head_dim"),
        v_dim: text("v_head_dim"),
    };
    let lower_bound = config["text_config"]["linear_attn_config"]["gate_lower_bound"]
        .as_f64()
        .unwrap() as f32;
    let kda = channel_decay::Geometry {
        heads: text("linear_attn_config.num_heads") / ranks,
        head_dim: text("linear_attn_config.head_dim"),
        kernel: text("linear_attn_config.short_conv_kernel_size"),
        convs: QkvConvs::Separate,
        gate: DecayGate::BoundedSigmoid {
            lower_bound_bits: lower_bound.to_bits(),
        },
        norm_eps_bits: float("rms_norm_eps").to_bits(),
        ..channel_decay::SERVED
    };
    let mut refusals = Vec::new();
    for entry in latent::entries(PROFILE, latent) {
        if let Err(refusal) = entry.contract() {
            refusals.push(format!("latent {}: {}", entry.name, refusal.0));
        }
    }
    // Prefill runs the streamed prepare and scan.
    for entry in channel_decay::entries(PROFILE, kda)
        .into_iter()
        .chain(channel_decay::streamed_entries(PROFILE, kda))
    {
        if let Err(refusal) = entry.contract() {
            refusals.push(format!("kda {}: {}", entry.symbol, refusal.0));
        }
    }
    refusals
}

#[test]
fn every_geometry_and_the_fixture_fit_the_engine_domains() {
    let fixture_config = read(&fixture().join("config.json"));
    let configs = GEOMETRIES
        .iter()
        .map(|g| (g.id, config_of(g)))
        .chain([("synthetic-small fixture", fixture_config)]);
    for (name, config) in configs {
        for &ranks in SERVED_RANKS {
            let refusals = engine_refusals(&config, ranks);
            assert!(refusals.is_empty(), "{name} at TP{ranks}: {refusals:#?}");
        }
    }
}

fn plan_everything(g: &Geometry) {
    for &ranks in SERVED_RANKS {
        let profile = KernelProfile::new(g, ranks);
        assert_eq!((profile.geometry, profile.ranks), (g.id, ranks));
        let tensors = catalog::catalog(g);
        let ranks_plan = weight_plan::plan(g, &tensors, ranks).unwrap();
        assert_eq!(ranks_plan.len(), ranks as usize);
        let decisions = tuning::moe_decisions(g, ranks).unwrap();
        assert_eq!(decisions.len(), 4);
        assert_eq!(
            decisions.values().all(|d| d.status.is_none()),
            tuning::table(g).is_some()
        );
        let planner = Planner::new(g, ranks).unwrap();
        let capacity = planner
            .plan(
                Serving {
                    ranks,
                    context_parallel: 1,
                    slots: 1,
                    draft_tokens: 3,
                    kv: kernel_core::latent_attention::LatentKvFormat::Fp8E4M3Block128,
                    vision: true,
                },
                DEVICE,
            )
            .unwrap();
        assert!(capacity.kv_tokens > 0);
        println!(
            "{} TP{ranks}: {} tensors, {:.2} GiB per rank, MoE {:?}",
            g.id,
            tensors.len(),
            ranks_plan[0].bytes.total() as f64 / f64::from(1 << 30),
            decisions
                .iter()
                .map(|(op, d)| (op.as_str(), d.status.as_deref().unwrap_or("tuned")))
                .collect::<Vec<_>>()
        );
        println!("{}", planner.table(DEVICE));
    }
}

#[test]
fn every_geometry_is_admitted_and_planned_from_its_data() {
    for g in GEOMETRIES {
        let matched = admission::admit(&config_of(g)).unwrap();
        assert_eq!(matched.geometry.id, g.id);
        assert_eq!(matched.runtime.index_topk, g.limits.index_topk);
        plan_everything(g);
    }
}

#[test]
fn the_recipe_config_selects_its_geometry_or_the_fixture_is_refused() {
    let fixture_config = read(&fixture().join("config.json"));
    let geometry = read(&fixture().join("geometry.json"));
    structure::check(&fixture_config).unwrap();
    // The geometry file is the config's execution part.
    let mut execution = fixture_config.clone();
    overlay(&mut execution, &geometry);
    assert_eq!(execution, fixture_config);
    let (Some(path), Some(id)) = (
        std::env::var_os("CHOPINFER_GLM5NEXT_RECIPE_CONFIG"),
        std::env::var("CHOPINFER_GLM5NEXT_RECIPE_GEOMETRY").ok(),
    ) else {
        // Compiled only through an extra geometry directory.
        match admission::admit(&fixture_config) {
            Ok(matched) => {
                assert!(!matched.geometry.base, "{}", matched.geometry.id);
                println!("fixture compiled as {}", matched.geometry.id);
            }
            Err(error) => {
                assert_eq!(
                    error,
                    format!(
                        "unsupported geometry: flash: text_config.first_k_dense_replace expected 3, received 1; supported ids: {}",
                        GEOMETRIES
                            .iter()
                            .map(|g| g.id)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                );
                println!("uncompiled fixture refused: {error}");
            }
        }
        return;
    };
    let config = read(std::path::Path::new(&path));
    let matched = admission::admit(&config).unwrap();
    let g = matched.geometry;
    assert_eq!(g.id, id);
    assert!(
        g.base,
        "the recipe adds its geometry to models/glm5next/geometries"
    );
    assert!(tuning::table(g).is_none(), "a recipe geometry is untuned");
    assert_eq!(tuning::nearest(g)[0].id, "flash");
    admission::check_ranks(4).unwrap();
    plan_everything(g);
    let ids = GEOMETRIES
        .iter()
        .map(|g| g.id)
        .collect::<Vec<_>>()
        .join(", ");
    let context = config["text_config"]["max_position_embeddings"]
        .as_u64()
        .unwrap();
    // The recipe classes the geometry's rules admit (inherited from
    // supported.json unless the geometry entry narrows them).
    let rules: Value = serde_json::from_str(g.rules).unwrap();
    let classes = &rules["quantization_recipe_classes"];
    assert_eq!(*classes, serde_json::json!(["fp8", "modelopt"]));
    let cases = [
        (
            "text_config.max_position_embeddings",
            Value::from(context / 2),
            format!(
                "unsupported geometry: flash: text_config.first_k_dense_replace expected 3, received 1; {id}: text_config.max_position_embeddings expected {context}, received {}; supported ids: {ids}",
                context / 2
            ),
        ),
        (
            "text_config.scoring_func",
            "softmax".into(),
            "unsupported text_config.scoring_func: expected one of [\"sigmoid\"], received \"softmax\"".into(),
        ),
        (
            "model_type",
            "glm5".into(),
            "unsupported model_type: expected one of [\"glm5_next\"], received \"glm5\"".into(),
        ),
        (
            "quantization_config.quant_method",
            "unknown".into(),
            format!(
                "unsupported quantization_config.quant_method: expected one of {classes}, received \"unknown\""
            ),
        ),
        (
            "text_config.index_topk",
            Value::from(2048),
            format!(
                "unsupported runtime text_config.index_topk: expected a multiple of 4 in 4..={}, received 2048",
                g.limits.index_topk
            ),
        ),
    ];
    for (field, value, expected) in cases {
        let error = admission::admit(&set(&config, field, value)).unwrap_err();
        assert_eq!(error, expected);
        println!("refused: {error}");
    }
    assert_eq!(
        admission::check_ranks(2).unwrap_err(),
        "unsupported tensor parallel size 2: glm5next serves [4]"
    );
}
