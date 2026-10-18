//! The weight plan (`model_deepseek41_contract::weight_plan`) over the
//! synthetic checkpoint's headers (`model_deepseek41::synthetic`), which
//! name every tensor of the official export: every rank of every supported
//! rank count of each compiled geometry plans its text image, with the
//! views the bundle binds, aligned and disjoint records and a rank index
//! that names the plan's identity.
use model_deepseek41::synthetic;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, admit, check_ranks};
use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES, Geometry};
use model_deepseek41_contract::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE, index_role};
use model_deepseek41_contract::weight_plan::{VIEW_ALIGNMENT, plan};
use runtime_bundle_abi::weight_plan::{
    CheckpointFileV1, NamedBytesV1, WeightPlanImplementationV1, WeightPlanRequestV1,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// The config.json of `geometry`: its document plus the fields a geometry
/// leaves out, as the export writes them.
fn config(geometry: &Geometry) -> Value {
    let mut config: Value = serde_json::from_str(geometry.document).unwrap();
    config["architectures"] = json!(["DeepseekV41ForCausalLM"]);
    config["model_type"] = json!("deepseek_v41");
    config["transformers_version"] = json!("5.6.0");
    config["text_config"]["model_type"] = json!("deepseek_v41_text");
    config["text_config"]["initializer_range"] = json!(0.02);
    config["text_config"]["use_cache"] = json!(true);
    config["text_config"]["attention_dropout"] = json!(0.0);
    config["vision_config"]["model_type"] = json!("deepseek_v41_vision");
    config["quantization_config"] = json!({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4"
    });
    config
}

/// The direct-load request of `geometry`'s synthetic checkpoint for `rank`
/// of `ranks`.
fn request(geometry: &Geometry, ranks: u32, rank: u32) -> WeightPlanRequestV1 {
    let config = config(geometry);
    assert_eq!(admit(&config).unwrap().id, geometry.id);
    let tensors = synthetic::tensors(geometry).unwrap();
    let (header, index, payload) = synthetic::header(&tensors).unwrap();
    let data_offset = 8 + header.len() as u64;
    let mut identity = Sha256::new();
    identity.update(synthetic::FILE);
    identity.update(&header);
    WeightPlanRequestV1 {
        image_role: RANK_IMAGE_ROLE.into(),
        ranks,
        rank,
        checkpoint_identity: identity.finalize().into(),
        implementation: WeightPlanImplementationV1 {
            shared_transforms: [7; 32],
            bundle_module: [0; 32],
        },
        files: vec![CheckpointFileV1 {
            name: synthetic::FILE.into(),
            byte_length: data_offset + payload,
            data_offset,
            header,
        }],
        documents: vec![
            NamedBytesV1 {
                name: "config.json".into(),
                bytes: serde_json::to_vec(&config).unwrap(),
            },
            NamedBytesV1 {
                name: "model.safetensors.index.json".into(),
                bytes: index,
            },
        ],
        components: vec![],
        host_views: vec![],
    }
}

/// Every view the bundle binds at layer `layer` of `geometry`
/// (B-MODULES, the bundle's view list), and the global ones.
fn bound_views(geometry: &Geometry, layer: u32) -> Vec<String> {
    let mut views = vec!["embed.weight".to_owned(), "norm.weight".to_owned()];
    let l = format!("layers.{layer}.");
    for name in [
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "hc_ffn_fn",
        "hc_ffn_base",
        "hc_ffn_scale",
        "ffn.gate.bias",
        "ffn.gate.bias_vl",
        "ffn.experts.gate_up.weight",
        "ffn.experts.gate_up.scale",
        "ffn.experts.down.weight",
        "ffn.experts.down.scale",
        "attn.attn_sink",
    ] {
        views.push(format!("{l}{name}"));
    }
    for part in ["weight", "scale"] {
        for matrix in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
            views.push(format!("{l}attn.{matrix}.{part}"));
        }
        for matrix in ["w1", "w3", "w2"] {
            views.push(format!("{l}ffn.shared_experts.{matrix}.{part}"));
        }
    }
    let _ = geometry;
    views
}

fn check(geometry: &Geometry, ranks: u32) {
    let started = std::time::Instant::now();
    for rank in 0..ranks {
        let request = request(geometry, ranks, rank);
        let plan = plan(&request).unwrap().expect("the text image is planned");
        assert_eq!((plan.ranks, plan.rank), (ranks, rank));
        assert_eq!(plan.arena_bytes % VIEW_ALIGNMENT, 0);
        let mut end = 0;
        for record in &plan.records {
            assert!(record.arena_offset >= end, "records overlap");
            end = record.arena_offset + record.byte_length;
            assert!(record.sources.len() <= 64);
        }
        assert!(end <= plan.arena_bytes);
        let [component] = &plan.components[..] else {
            panic!("one rank index");
        };
        assert_eq!(component.name, index_role(rank));
        let index = bundle_runtime::index::parse(&component.bytes, rank, INDEX_SCHEMA).unwrap();
        let names: std::collections::BTreeSet<_> = index.views.keys().cloned().collect();
        for name in bound_views(geometry, 0) {
            assert!(
                names.contains(&name),
                "{} rank {rank}: no view {name}",
                geometry.id
            );
        }
        // The Engram tables are gathered from the mapped checkpoint.
        assert!(!names.iter().any(|name| name.contains("engram.embed")));
        let identity = hex(&Sha256::digest(
            plan.identity_document(&request.checkpoint_identity, &request.implementation),
        ));
        assert_eq!(index.direct_identity.as_deref(), Some(identity.as_str()));
        if rank == 0 {
            eprintln!(
                "{} tp{ranks}: {} views, {} records, {} arena bytes",
                geometry.id,
                names.len(),
                plan.records.len(),
                plan.arena_bytes
            );
        }
    }
    eprintln!(
        "{} tp{ranks}: planned every rank in {:?}",
        geometry.id,
        started.elapsed()
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_compiled_geometry_plans_every_rank() {
    for geometry in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(geometry, ranks).is_ok() && (geometry.id != FLASH.id || ranks == 4) {
                check(geometry, ranks);
            }
        }
    }
}

#[test]
fn other_image_roles_are_declined() {
    let mut request = request(&FLASH, 4, 0);
    request.image_role = "deepseek41.encoder-rank-images.v1".into();
    assert!(plan(&request).unwrap().is_none());
}
