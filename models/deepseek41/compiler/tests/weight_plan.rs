//! The weight plan (`model_deepseek41_contract::weight_plan`) over the
//! synthetic checkpoint's headers (`model_deepseek41::synthetic`), which
//! name every tensor of the official export: every rank of every supported
//! rank count of each compiled geometry plans its text image, with the
//! views the bundle binds, aligned and disjoint records and a rank index
//! that names the plan's identity. The Engram tables' code and scale planes
//! are the plan's mapped views, each its checkpoint tensor as the headers
//! state it.
use model_deepseek41::synthetic;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, admit, check_ranks};
use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES, Geometry};
use model_deepseek41_contract::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE, index_role};
use model_deepseek41_contract::weight_plan::{IMAGE_ALIGNMENT, VIEW_ALIGNMENT, device_bytes, plan};
use runtime_bundle_abi::weight_plan::{
    CheckpointFileV1, NamedBytesV1, ViewPlacementV1, WeightPlanImplementationV1,
    WeightPlanRequestV1, WeightPlanV1,
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
        census: Vec::new(),
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
    let tensors: std::collections::BTreeMap<_, _> = synthetic::tensors(geometry)
        .unwrap()
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect();
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
        // The exact F32 projection reads every BF16 dense weight (the
        // routers, the compressors' and indexers' projections, the head)
        // as row-major rows; none is packed in MMA fragment order.
        for (name, view) in &index.views {
            assert_ne!(
                view.layout_id, "bf16_mma_fragments_16x16_v1",
                "{} rank {rank}: {name}",
                geometry.id
            );
        }
        for name in ["layers.0.ffn.gate.weight", "head.weight"] {
            assert_eq!(
                index.views[name].layout_id, "bf16_row_major_v1",
                "{} rank {rank}: {name}",
                geometry.id
            );
        }
        // A row-major view holds its tensor's rows, or one rank's share of
        // them, and no padding to the fragment packing's row tiles (a
        // single-row confidence head, a 32-expert router).
        for (name, view) in &index.views {
            if view.layout_id != "bf16_row_major_v1" {
                continue;
            }
            let Some(tensor) = tensors.get(name.as_str()) else {
                continue;
            };
            assert!(
                [tensor.bytes(), tensor.bytes() / u64::from(ranks)].contains(&view.bytes),
                "{} rank {rank}: {name} is {} bytes for a {:?} tensor",
                geometry.id,
                view.bytes,
                tensor.shape
            );
        }
        check_mapped(geometry, &request, &plan, &index);
        let planned = identity(&request, &plan);
        assert_eq!(index.direct_identity.as_deref(), Some(planned.as_str()));
        // The mapped views are part of what the identity names.
        let unmapped = WeightPlanV1 {
            views: Vec::new(),
            ..plan.clone()
        };
        assert_ne!(identity(&request, &unmapped), planned);
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

/// The planned image identity of `plan` for `request`, as host and bundle
/// take it.
fn identity(request: &WeightPlanRequestV1, plan: &WeightPlanV1) -> String {
    hex(&Sha256::digest(plan.identity_document(
        &request.checkpoint_identity,
        &request.implementation,
    )))
}

/// The plan's mapped views are exactly the Engram tables' code and scale
/// planes, unprewarmed, each at its rank index offset and as long as its
/// checkpoint tensor; one plain copy of the whole tensor covers each and
/// nothing else does; they follow every device view, and the device holds
/// the arena less their cuts.
fn check_mapped(
    geometry: &Geometry,
    request: &WeightPlanRequestV1,
    plan: &WeightPlanV1,
    index: &bundle_runtime::index::RankIndex,
) {
    let header: Value = serde_json::from_slice(&request.files[0].header).unwrap();
    let planes: std::collections::BTreeSet<&str> = header
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .filter(|name| {
            name.ends_with(".engram.embed.weight") || name.ends_with(".engram.embed.scale")
        })
        .collect();
    assert_eq!(planes.len() as u32, 2 * geometry.engram_layers());
    for plane in &planes {
        let layer: u32 = plane.split('.').nth(1).unwrap().parse().unwrap();
        assert!(geometry.is_engram_layer(layer), "{plane}");
    }
    let mapped: std::collections::BTreeSet<&str> =
        plan.views.iter().map(|view| view.id.as_str()).collect();
    assert_eq!(mapped, planes, "the mapped views are the Engram planes");
    let mut cut = 0;
    for view in &plan.views {
        let at = format!(
            "{} tp{} rank {} {}",
            geometry.id, plan.ranks, plan.rank, view.id
        );
        assert_eq!(
            view.placement,
            ViewPlacementV1::Mapped { prewarm: false },
            "{at}"
        );
        assert_eq!(view.alignment, IMAGE_ALIGNMENT, "{at}");
        let entry = &header[&view.id];
        let [start, end] = [0, 1].map(|i| entry["data_offsets"][i].as_u64().unwrap());
        let shape: Vec<u64> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap())
            .collect();
        assert_eq!(view.byte_length, end - start, "{at}: the tensor's bytes");
        let listed = &index.views[&view.id];
        let (layout, dtype) = if view.id.ends_with(".weight") {
            ("mxfp8_e4m3_rows_v1", "F8_E4M3")
        } else {
            ("mxfp8_ue8m0_scale_rows_v1", "F8_E8M0")
        };
        assert_eq!(
            (listed.layout_id.as_str(), listed.dtype.as_str()),
            (layout, dtype),
            "{at}"
        );
        assert_eq!(
            (listed.offset, listed.bytes),
            (view.offset, view.byte_length),
            "{at}"
        );
        let span = view.offset..view.offset + view.byte_length;
        let covering: Vec<_> = plan
            .records
            .iter()
            .filter(|r| r.arena_offset < span.end && span.start < r.arena_offset + r.byte_length)
            .collect();
        let [record] = covering[..] else {
            panic!("{at}: {} records cover it", covering.len());
        };
        assert_eq!(
            (record.arena_offset, record.byte_length),
            (view.offset, view.byte_length),
            "{at}"
        );
        assert_eq!(
            record.transform,
            kernel_core::weight_transform::WeightTransform::Copy.id(),
            "{at}"
        );
        let [source] = &record.sources[..] else {
            panic!("{at}: one source");
        };
        assert_eq!(source.tensor, view.id, "{at}");
        assert!(
            matches!(
                (record.axis, &record.ranges[..]),
                (None, []) | (Some(0), [_])
            ) && record.ranges.iter().all(|rows| *rows == (0..shape[0])),
            "{at}: the whole tensor"
        );
        assert_eq!(record.stored_shape, shape, "{at}");
        for (name, other) in &index.views {
            if !mapped.contains(name.as_str()) {
                assert!(
                    other.offset + other.bytes <= view.offset,
                    "{at}: {name} after it"
                );
            }
        }
        cut += (span.end.next_multiple_of(plan.alignment)).min(plan.arena_bytes) - span.start;
    }
    assert_eq!(device_bytes(plan), plan.arena_bytes - cut);
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
    request.image_role = "deepseek41.other-images.v1".into();
    assert!(plan(&request).unwrap().is_none());
}

/// The encoder image of the flash checkpoint plans alike on every rank:
/// every view of the layout whole at its offset, the patch weight's 588
/// columns zero-padded to 640, the aligner's first weight through the
/// bundle's column permutation, and an index naming the plan's identity.
#[test]
fn the_encoder_image_plans_every_view_on_every_rank() {
    use model_deepseek41_contract::encoder::views;
    use model_deepseek41_contract::encoder_plan::SPACE_TO_DEPTH_COLUMNS;
    let layout = views::layout().unwrap();
    let mut first = None;
    for rank in 0..4 {
        let mut request = request(&FLASH, 4, rank);
        request.image_role = std::str::from_utf8(views::IMAGE_ROLE).unwrap().into();
        let plan = plan(&request)
            .unwrap()
            .expect("the encoder image is planned");
        assert_eq!(plan.arena_bytes, views::image_bytes().unwrap());
        assert_eq!(plan.records.len(), layout.len());
        for (record, view) in plan.records.iter().zip(&layout) {
            assert_eq!(record.arena_offset, view.offset);
            assert_eq!(record.byte_length, view.bytes());
            assert_eq!(record.stored_shape, view.shape);
            assert_eq!(record.sources[0].tensor, view.name);
            let expected = if view.name == "aligner.w1.weight" {
                SPACE_TO_DEPTH_COLUMNS
            } else {
                1
            };
            assert_eq!(record.transform, expected, "{}", view.name);
        }
        let [component] = &plan.components[..] else {
            panic!("one encoder index");
        };
        assert_eq!(component.name, views::index_role(rank));
        let index = views::parse(&component.bytes, rank).unwrap();
        let identity = hex(&Sha256::digest(
            plan.identity_document(&request.checkpoint_identity, &request.implementation),
        ));
        assert_eq!(index.direct_identity.as_deref(), Some(identity.as_str()));
        // Every rank stores the same records.
        let records = format!("{:?}", plan.records);
        assert_eq!(first.get_or_insert(records.clone()), &records);
    }
}
