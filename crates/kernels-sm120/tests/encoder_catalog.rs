//! Generated encoder catalogs on synthetic declarations: symbols derive from
//! (variant id, parameters), equal stages of two models share a symbol and
//! a body, and one declaration's entries never depend on another's. Each
//! model's own lock check is its bundle's `encoder_catalog_stability`.

use encoder_spec::components::rotary::RotaryComponent;
use encoder_spec::components::{Attention, Boundary, Component};
use encoder_spec::{Declaration, Environment, parse};
use kernel_core::contract::{GeluMode, MultimodalQuery};
use kernels_sm120::encoder::{
    self, Entry, Kernel, Registered, Shared, Slot, catalog, cubin_specs, model_catalog, slots,
};
use serde_json::{Value, json};

/// A small F32 tower of `model`: a patch GEMM, the learned position grid,
/// `blocks` pre-norm blocks with axial rotary, then a 2 x 2 merger.
fn tower(model: &str, hidden: u32) -> Value {
    json!({
        "schema": "chopinfer.encoder.v1",
        "model": model,
        "name": format!("{model}-vision"),
        "precision": "f32-pairs.v1",
        "checkpoint": {
            "prefix": "visual.",
            "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 2}}]
        },
        "input": {
            "recipes": [{"tag": "image.patch-grid.v1", "patch": 16, "temporal_patch": 2, "merge": 2,
                         "maximum_pixels": 1048576}],
            "patchify": "merge-blocks.v1", "payload": "f32", "columns": 1536, "capacity_rows": 4096
        },
        "sequences": {"kind": "item"},
        "image_layout": "view-order.v1",
        "stages": [
            {"op": "linear.pairs.v1", "out": 256, "bias": true,
             "weights": {"weight": {"tensor": "patch.weight", "transforms": ["flatten"]}, "bias": "patch.bias"}},
            {"op": "position.learned-grid.bilinear-align-corners.f32.v1", "table": [16, 16],
             "weights": {"table": "pos.weight"}},
            {"op": "transformer.pre-norm.v1", "blocks": 2, "width": 256,
             "norm": {"op": "norm.layer-f32.v1", "eps": 1e-6},
             "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 4, "kv_heads": 4, "head_dim": 64,
                           "qkv_bias": true, "proj_bias": true, "mask": "full"},
             "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 16, "theta": 10000, "style": "neox",
                        "table": "table.hf-f32-angle.v1"},
             "mlp": {"op": "mlp.gelu.v1", "hidden": hidden, "gelu": "tanh", "bias": true},
             "weights": {
                "norm1.weight": "b.{i}.n1.weight", "norm1.bias": "b.{i}.n1.bias",
                "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
                "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
                "norm2.weight": "b.{i}.n2.weight", "norm2.bias": "b.{i}.n2.bias",
                "up.weight": "b.{i}.up.weight", "up.bias": "b.{i}.up.bias",
                "down.weight": "b.{i}.down.weight", "down.bias": "b.{i}.down.bias"
             }},
            {"op": "norm.layer-f32.v1", "eps": 1e-6, "weights": {"weight": "m.n.weight", "bias": "m.n.bias"}},
            {"op": "merge.consecutive.v1", "rows": 4},
            {"op": "linear.pairs.v1", "out": 1024, "bias": true, "gelu": "erf",
             "weights": {"weight": "m.fc1.weight", "bias": "m.fc1.bias"}},
            {"op": "linear.pairs.v1", "out": 512, "bias": true, "output": "bf16",
             "weights": {"weight": "m.fc2.weight", "bias": "m.fc2.bias"}}
        ],
        "output": {"op": "layout.tokens.v1", "hidden": 512, "positions": "grid-3axis.v1", "media_mask": false},
        "parallel": ["replicated"]
    })
}

fn declaration(value: &Value) -> Declaration {
    let model = value["model"].as_str().unwrap();
    parse(&value.to_string(), &Environment::new(model)).unwrap()
}

fn symbols(entries: &[Entry]) -> Vec<&str> {
    entries.iter().map(|entry| entry.symbol.as_str()).collect()
}

#[test]
fn symbols_derive_from_variant_ids_and_parameters() {
    let entries = catalog(&declaration(&tower("example", 512)), &Shared).unwrap();
    // A readable stem, then 12 hex digits of the kernel's identity.
    for entry in &entries {
        let digest = entry.symbol.rsplit('_').nth(1).unwrap();
        assert_eq!(digest.len(), 12, "{}", entry.symbol);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }
    assert_eq!(
        entries.iter().map(Entry::stem).collect::<Vec<_>>(),
        [
            "chop_enc_linear_pairs_n256_k1536_bias_f32_v1",
            "chop_enc_position_learned_grid_bilinear_align_corners_f32_w256_table256_v1",
            "chop_enc_norm_layer_f32_w256_v1",
            "chop_enc_linear_pairs_n768_k256_bias_f32_v1",
            "chop_enc_rotary_axial_2d_q4_kv4_d64_r64_neox_v1",
            "chop_enc_attention_varlen_pairs_q4_kv4_d64_full_v1",
            "chop_enc_linear_pairs_n256_k256_bias_residual_f32_v1",
            "chop_enc_linear_pairs_n512_k256_bias_gelu_tanh_f32_v1",
            "chop_enc_linear_pairs_n256_k512_bias_residual_f32_v1",
            "chop_enc_linear_pairs_n1024_k1024_bias_gelu_erf_f32_v1",
            "chop_enc_linear_pairs_n512_k1024_bias_bf16_v1",
            "chop_enc_layout_tokens_w512_v1",
        ]
    );
    // Every entry plans for the declaration's capacity; the override runs in
    // the language model's prefill.
    for entry in &entries {
        let rows = if entry.symbol.starts_with("chop_enc_layout_") {
            encoder::OVERRIDE_ROWS
        } else {
            4096
        };
        assert_eq!(entry.capacity(), rows, "{}", entry.symbol);
        assert!(entry.plan(rows).is_ok());
        assert!(entry.plan(rows + 1).is_err());
        assert_eq!(entry.params, encoder::kernel::params(entry.query));
    }
    let MultimodalQuery::GemmPairs(fc1) = entries[7].query else {
        panic!("fc1 is a pairs GEMM");
    };
    assert_eq!(fc1.epilogues.gelu, Some(GeluMode::Tanh));
    // Rows and GEMMs in one cubin, attention in the other, each with only
    // its entries' headers.
    let [rows, attention] = cubin_specs(&entries).unwrap();
    assert_eq!(rows.entries.len(), 11);
    assert_eq!(
        rows.headers,
        [
            "#include \"encoder/linear_pairs.cuh\"",
            "#include \"encoder/rows/embedding_override.cuh\"",
            "#include \"encoder/rows/interpolated_position_add_f32.cuh\"",
            "#include \"encoder/rows/layernorm_f32.cuh\"",
            "#include \"encoder/rows/rope_qkv_pairs.cuh\"",
        ]
    );
    assert_eq!(
        symbols(&entries[5..6]),
        [attention.entries[0].symbol.as_str()]
    );
    assert_eq!(
        attention.headers,
        ["#include \"attention/varlen_pairs.cuh\""]
    );
}

/// Every kernel fills a named slot of its stage, an attention kernel one
/// per block, and the catalog is the slots' entries, each once.
#[test]
fn slots_name_each_kernels_stage_part_and_block() {
    let tower = declaration(&tower("alpha", 1024));
    let slots = slots(&tower, &Shared).unwrap();
    let named: Vec<(usize, &str, Option<u32>)> = slots
        .iter()
        .map(|(slot, _)| (slot.stage, slot.part, slot.block))
        .collect();
    assert_eq!(
        named,
        [
            (0, "linear", None),
            (1, "position", None),
            (2, "norm", None),
            (2, "qkv", None),
            (2, "rotary", None),
            (2, "attention", Some(0)),
            (2, "attention", Some(1)),
            (2, "proj", None),
            (2, "up", None),
            (2, "down", None),
            (3, "norm", None),
            (5, "linear", None),
            (6, "linear", None),
            (7, "layout", None),
        ]
    );
    let mut once: Vec<Entry> = Vec::new();
    for (_, entry) in slots {
        if !once.contains(&entry) {
            once.push(entry);
        }
    }
    assert_eq!(once, catalog(&tower, &Shared).unwrap());
    assert_eq!(Slot::default().part, "");
}

#[test]
fn equal_stages_of_two_models_share_symbols_and_bodies() {
    let first =
        cubin_specs(&catalog(&declaration(&tower("example", 512)), &Shared).unwrap()).unwrap();
    let second =
        cubin_specs(&catalog(&declaration(&tower("other", 1024)), &Shared).unwrap()).unwrap();
    let mut shared = 0;
    for (first, second) in first.iter().zip(&second) {
        for entry in &first.entries {
            if let Some(other) = second
                .entries
                .iter()
                .find(|other| other.symbol == entry.symbol)
            {
                assert_eq!(
                    (
                        &entry.params,
                        &entry.body,
                        entry.max_threads,
                        entry.min_blocks
                    ),
                    (
                        &other.params,
                        &other.body,
                        other.max_threads,
                        other.min_blocks
                    )
                );
                shared += 1;
            }
        }
    }
    // Only the MLP's two GEMMs differ.
    assert_eq!(shared, 10);
}

/// A model's registered stage: a plain row kernel after the merger.
struct Provided;

impl Registered for Provided {
    fn stage(
        &self,
        component: &dyn Component,
        input: &Boundary,
        capacity: u32,
    ) -> Result<Vec<Kernel>, String> {
        Ok(vec![Kernel::new(
            component.id(),
            None,
            MultimodalQuery::ResidualAddF32 {
                width: input.width()?,
                capacity,
            },
        )])
    }

    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        _: &Attention,
        _: u32,
    ) -> Result<Vec<Kernel>, String> {
        Err(format!("`{}` is not provided", rotary.id()))
    }
}

#[test]
fn a_declaration_never_changes_another_models_entries() {
    let example = declaration(&tower("example", 512));
    let units = |entries: &[Entry]| {
        cubin_specs(entries)
            .unwrap()
            .map(|spec| kernel_build::generate(&spec).unwrap().translation_unit)
    };
    let alone = units(&catalog(&example, &Shared).unwrap());
    // A model's kernels for its registered variants never reach a
    // declaration of shared variants.
    for registered in [&Shared as &dyn Registered, &Provided] {
        assert_eq!(units(&catalog(&example, registered).unwrap()), alone);
    }
    // A catalog over another model's declaration as well keeps every entry
    // of the first.
    let other = declaration(&tower("other", 1024));
    let both = model_catalog([&example, &other], &Shared).unwrap();
    let example_symbols = catalog(&example, &Shared).unwrap();
    assert!(example_symbols.iter().all(|entry| both.contains(entry)));
}

#[test]
fn variants_without_kernels_and_conflicting_symbols_are_refused() {
    let mut value = tower("example", 512);
    value["stages"][6].as_object_mut().unwrap().remove("output");
    value["stages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"op": "cast.bf16.v1"}));
    let error = catalog(&declaration(&value), &Shared).unwrap_err();
    assert!(
        error.0.contains("cast.bf16.v1` has no encoder kernel yet"),
        "{error}"
    );
    let entry = |capacity| Entry {
        symbol: "chop_enc_norm_layer_f32_w256_v1".into(),
        query: MultimodalQuery::LayerNormF32 {
            width: 256,
            eps_bits: 1e-6_f32.to_bits(),
            capacity,
        },
        params: Vec::new(),
    };
    assert!(encoder::union([entry(4096), entry(4096)]).is_ok());
    // One symbol never stands for two bodies.
    let mut wider = entry(4096);
    wider.query = MultimodalQuery::LayerNormF32 {
        width: 512,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: 4096,
    };
    assert!(encoder::union([entry(4096), wider]).is_err());
}

/// The q/k RMSNorm attention, the clamped SwiGLU and the projector stage
/// lower to their row entries (`tests/vision_encoder_compile.rs` checks the
/// order and builds them); the projector's GELU row pass is erf only.
#[test]
fn vision_variants_lower_and_tanh_projector_gelu_is_refused() {
    let lowered = |edit: &dyn Fn(&mut Value)| {
        let mut value = tower("example", 512);
        edit(&mut value);
        catalog(&declaration(&value), &Shared)
    };
    let projector = |gelu: &'static str| {
        move |d: &mut Value| {
            d["stages"][6] = json!({
                "op": "projector.norm-gelu-mlp.v1",
                "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
                "gelu": gelu,
                "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 1024, "bias": false, "limit": 10},
                "output": "bf16",
                "weights": {"norm.weight": "p.n.weight", "norm.bias": "p.n.bias",
                            "gate_up.weight": "p.gate_up.weight", "down.weight": "p.down.weight"}
            });
            d["output"]["hidden"] = 1024.into();
        }
    };
    let entries = lowered(&|d| {
        let attention = &mut d["stages"][2]["attention"];
        attention["op"] = "attention.varlen.qk-rms.pairs.v1".into();
        attention["qk_norm_eps"] = 1e-5.into();
        let weights = &mut d["stages"][2]["weights"];
        weights["q_norm.weight"] = "b.{i}.qn.weight".into();
        weights["k_norm.weight"] = "b.{i}.kn.weight".into();
        d["stages"][2]["mlp"] =
            json!({"op": "mlp.swiglu-clamped.v1", "hidden": 512, "bias": true, "limit": 10});
        let weights = d["stages"][2]["weights"].as_object_mut().unwrap();
        for role in ["up.weight", "up.bias"] {
            weights.remove(role);
        }
        weights.insert("gate_up.weight".into(), "b.{i}.gate_up.weight".into());
        weights.insert("gate_up.bias".into(), "b.{i}.gate_up.bias".into());
        projector("erf")(d);
    })
    .unwrap();
    let symbols = symbols(&entries);
    for stem in [
        "chop_enc_attention_varlen_qk_rms_pairs_qk_norm_q4_kv4_d64_",
        "chop_enc_mlp_swiglu_clamped_swiglu_h512_",
        "chop_enc_mlp_swiglu_clamped_swiglu_h1024_",
        "chop_enc_projector_norm_gelu_mlp_gelu_w1024_",
    ] {
        assert!(
            symbols.iter().any(|symbol| symbol.starts_with(stem)),
            "{stem} in {symbols:?}"
        );
    }
    let error = lowered(&projector("tanh")).unwrap_err();
    assert!(error.0.contains("the GELU row pass is erf only"), "{error}");
}

#[test]
fn symbols_are_injective_over_variant_ids_and_roles() {
    let rows = MultimodalQuery::ResidualAddF32 {
        width: 1024,
        capacity: 4096,
    };
    let symbol = |id: &str, role| encoder::symbol::derive(id, role, rows).unwrap();
    let symbols = [
        symbol("example:conv1d.foo-bar.v1", None),
        symbol("example:conv1d.foo.bar.v1", None),
        symbol("example:conv1d.foo.v1", Some("bar")),
        symbol("example:conv1d.foo-bar.v2", None),
        symbol("conv1d.foo-bar.v1", None),
    ];
    for (index, a) in symbols.iter().enumerate() {
        assert!(a.starts_with("chop_enc_"), "{a}");
        for b in &symbols[index + 1..] {
            assert_ne!(a, b);
        }
    }
    // Runtime values (capacity, eps) are launch parameters, not code.
    let other = MultimodalQuery::ResidualAddF32 {
        width: 1024,
        capacity: 8192,
    };
    assert_eq!(
        symbol("conv1d.foo-bar.v1", None),
        encoder::symbol::derive("conv1d.foo-bar.v1", None, other).unwrap()
    );
}

#[test]
fn stages_differing_only_in_runtime_values_share_one_entry() {
    // The merger's LayerNorm takes another epsilon than the blocks'.
    let mut value = tower("example", 512);
    value["stages"][3]["eps"] = json!(1e-5);
    let entries = catalog(&declaration(&value), &Shared).unwrap();
    let norms = entries
        .iter()
        .filter(|entry| matches!(entry.query, MultimodalQuery::LayerNormF32 { .. }))
        .count();
    assert_eq!(norms, 1);
    // One operation at two capacities: the entry grids for the larger.
    let at = |capacity: u32| {
        let mut value = tower("example", 512);
        value["input"]["capacity_rows"] = json!(capacity);
        value["input"]["recipes"][0]["maximum_pixels"] = json!(u64::from(capacity) * 256);
        catalog(&declaration(&value), &Shared).unwrap()
    };
    let both = encoder::union(at(4096).into_iter().chain(at(8192))).unwrap();
    assert_eq!(both.len(), at(4096).len());
    assert!(
        both.iter()
            .all(|entry| entry.capacity() == 8192 || entry.capacity() == encoder::OVERRIDE_ROWS)
    );
}

/// The `#include "..."` closure of a cubin's headers inside the CUDA tree.
fn closure(headers: &[String]) -> Vec<std::path::PathBuf> {
    let root = kernels_sm120::cuda_source_root();
    let mut pending: Vec<std::path::PathBuf> = headers
        .iter()
        .map(|line| root.join(line.split('"').nth(1).unwrap()))
        .collect();
    let mut seen = Vec::new();
    while let Some(path) = pending.pop() {
        let path = path.canonicalize().unwrap();
        if seen.contains(&path) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix("#include \"") {
                let include = rest.split('"').next().unwrap();
                pending.push(path.parent().unwrap().join(include));
            }
        }
        seen.push(path);
    }
    seen
}

#[test]
fn row_cubins_include_only_the_templates_their_entries_call() {
    // A row template's header is in a cubin's source closure only if one of
    // its entries calls it, so editing another variant's kernel never moves
    // this cubin's lock.
    let entries = catalog(&declaration(&tower("example", 512)), &Shared).unwrap();
    let [rows, _] = cubin_specs(&entries).unwrap();
    let bodies = rows
        .entries
        .iter()
        .map(|entry| entry.body.as_str())
        .collect::<String>();
    // Encoder sources only: the shared engines (`gemm/`, `row/`, `common/`)
    // are frozen once a lock references them.
    let root = kernels_sm120::cuda_source_root().canonicalize().unwrap();
    for path in closure(&rows.headers) {
        let relative = path.strip_prefix(&root).unwrap();
        if !relative.starts_with("encoder") && !relative.starts_with("multimodal") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for definition in text.split("__device__ void ").skip(1) {
            let name = definition.split(['(', '<']).next().unwrap();
            assert!(
                bodies.contains(&format!("chop::multimodal::{name}")),
                "{} defines {name}, which no entry calls",
                path.display()
            );
        }
    }
}

/// A grouped-query pairs tower of RMS norms, window and key-0 sink
/// attention with a full-attention override, a reordered block and a SwiGLU
/// MLP, then an audio-like pairs stack of convolutions and a causal 1D
/// rotary transformer with a window override and a carry: each lowers to
/// the F32 row passes of `f32-pairs.v1`.
#[test]
fn pair_towers_lower_rms_reorders_swiglu_convolutions_and_carries() {
    let mut vision = tower("example", 512);
    let transformer = &mut vision["stages"][2];
    transformer["norm"] = json!({"op": "norm.rms-f32.v1", "eps": 1e-6});
    transformer["attention"] = json!({
        "op": "attention.varlen.pairs.v1", "q_heads": 8, "kv_heads": 2, "head_dim": 64,
        "qkv_bias": true, "proj_bias": true, "mask": {"window": [64, 64]}, "sink": "key0"});
    transformer["overrides"] =
        json!([{"blocks": "even", "attention": {"mask": "full", "sink": "none"}}]);
    transformer["reorder"] =
        json!([{"blocks": "odd", "reorder": {"op": "reorder.merge-group-columns.v1"}}]);
    transformer["mlp"] = json!({"op": "mlp.swiglu.v1", "hidden": 768, "bias": true});
    let weights = transformer["weights"].as_object_mut().unwrap();
    for role in ["norm1.bias", "norm2.bias", "up.weight", "up.bias"] {
        weights.remove(role);
    }
    weights.insert(
        "sinks".into(),
        json!({"tensor": "b.{i}.sinks", "blocks": "odd"}),
    );
    weights.insert("gate_up.weight".into(), json!("b.{i}.gate_up.weight"));
    weights.insert("gate_up.bias".into(), json!("b.{i}.gate_up.bias"));
    let entries = catalog(&declaration(&vision), &Shared).unwrap();
    let stems = entries.iter().map(Entry::stem).collect::<Vec<_>>();
    for stem in [
        "chop_enc_norm_rms_f32_w256_v1",
        "chop_enc_attention_varlen_pairs_q8_kv2_d64_full_v1",
        "chop_enc_attention_varlen_pairs_q8_kv2_d64_left64_right64_sink_v1",
        "chop_enc_rotary_axial_2d_q8_kv2_d64_r64_neox_v1",
        "chop_enc_reorder_merge_group_columns_w256_v1",
        "chop_enc_linear_pairs_n1536_k256_bias_f32_v1",
        "chop_enc_mlp_swiglu_swiglu_h768_v1",
        "chop_enc_linear_pairs_n256_k768_bias_residual_f32_v1",
    ] {
        assert!(stems.contains(&stem.to_owned()), "{stem} not in {stems:?}");
    }
    let parts = encoder::slots(&declaration(&vision), &Shared)
        .unwrap()
        .into_iter()
        .filter(|(slot, _)| slot.stage == 2)
        .map(|(slot, _)| slot.part)
        .collect::<Vec<_>>();
    assert_eq!(
        parts,
        [
            "norm",
            "qkv",
            "rotary",
            "attention",
            "attention",
            "proj",
            "reorder",
            "up",
            "swiglu",
            "down"
        ]
    );

    let audio = json!({
        "schema": "chopinfer.encoder.v1",
        "model": "example",
        "name": "example-audio",
        "precision": "f32-pairs.v1",
        "checkpoint": {
            "prefix": "audio.",
            "config": [{"file": "config.json", "pointer": "/audio_config", "values": {"layers": 4}}]
        },
        "input": {
            "recipes": [{"tag": "audio.log-mel.v1", "mel_bins": 128, "maximum_frames": 60001}],
            "payload": "f32", "columns": 128, "capacity_rows": 8192,
            "chunking": "whole-sequences.v1"
        },
        "sequences": {"kind": "segments", "input_rows": 6000},
        "image_layout": "view-order.v1",
        "stages": [
            {"op": "conv1d.im2col.v1", "kernel": 3, "stride": 1, "pad": [1, 1], "out": 256,
             "bias": true, "gelu": "erf",
             "weights": {"weight": {"tensor": "c1.weight", "transforms": ["taps_first"]},
                         "bias": "c1.bias"}},
            {"op": "conv1d.im2col.v1", "kernel": 3, "stride": 2, "pad": [1, 1], "out": 256,
             "bias": true, "gelu": "erf",
             "weights": {"weight": {"tensor": "c2.weight", "transforms": ["taps_first"]},
                         "bias": "c2.bias"}},
            {"op": "transformer.pre-norm.v1", "blocks": 4, "width": 256,
             "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
             "attention": {"op": "attention.varlen.pairs.v1", "q_heads": 4, "kv_heads": 4,
                           "head_dim": 64, "qkv_bias": true, "proj_bias": true, "mask": "causal"},
             "overrides": [{"blocks": "even", "attention": {"mask": {"causal_window": 128}}}],
             "rotary": {"op": "rotary.1d.v1", "pairs": 32, "theta": 10000, "style": "neox",
                        "table": "table.hf-f32-angle.v1", "positions": "within-sequence"},
             "carry": [{"save_after": 1, "add_after": 3}],
             "mlp": {"op": "mlp.gelu.v1", "hidden": 512, "gelu": "erf", "bias": true},
             "weights": {
                "norm1.weight": "b.{i}.n1.weight", "norm1.bias": "b.{i}.n1.bias",
                "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
                "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
                "norm2.weight": "b.{i}.n2.weight", "norm2.bias": "b.{i}.n2.bias",
                "up.weight": "b.{i}.up.weight", "up.bias": "b.{i}.up.bias",
                "down.weight": "b.{i}.down.weight", "down.bias": "b.{i}.down.bias"
             }},
            {"op": "norm.layer-f32.v1", "eps": 1e-5, "weights": {"weight": "n.weight", "bias": "n.bias"}},
            {"op": "group.repeat-last.v1", "rows": 2},
            {"op": "linear.pairs.v1", "out": 512, "bias": false, "output": "bf16",
             "weights": {"weight": "out.weight"}}
        ],
        "output": {"op": "layout.tokens.v1", "hidden": 512, "positions": "sequential", "media_mask": false},
        "parallel": ["replicated"]
    });
    let entries = catalog(&declaration(&audio), &Shared).unwrap();
    let stems = entries.iter().map(Entry::stem).collect::<Vec<_>>();
    for stem in [
        "chop_enc_conv1d_im2col_unfold_c128_k3_s1_p1_1_v1",
        "chop_enc_linear_pairs_n256_k384_bias_gelu_erf_f32_v1",
        "chop_enc_conv1d_im2col_unfold_c256_k3_s2_p1_1_v1",
        "chop_enc_rotary_1d_q4_kv4_d64_r64_neox_v1",
        "chop_enc_attention_varlen_pairs_q4_kv4_d64_causal_v1",
        "chop_enc_attention_varlen_pairs_q4_kv4_d64_causal_left128_v1",
        "chop_enc_transformer_pre_norm_carry_w256_v1",
    ] {
        assert!(stems.contains(&stem.to_owned()), "{stem} not in {stems:?}");
    }
    // A GELU row pass rounds BF16 rows; under f32-pairs.v1 it is the GEMM's.
    let mut rows = audio.clone();
    rows["stages"][1].as_object_mut().unwrap().remove("gelu");
    rows["stages"][1]["gelu_rows"] = json!("erf");
    let error = catalog(&declaration(&rows), &Shared).unwrap_err();
    assert!(error.0.contains("not a row pass"), "{error}");
}

#[test]
fn retired_bf16_queries_cannot_build_gpu_entries() {
    let entries = catalog(&declaration(&tower("example", 512)), &Shared).unwrap();
    let pair = entries
        .iter()
        .find_map(|entry| match entry.query {
            MultimodalQuery::GemmPairs(query)
                if query.output == kernel_core::contract::DType::Bf16 =>
            {
                Some(query)
            }
            _ => None,
        })
        .unwrap();
    for query in [
        MultimodalQuery::Gemm(pair),
        MultimodalQuery::GemmBf16(pair),
        MultimodalQuery::LayerNorm {
            width: 256,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: 256,
        },
        MultimodalQuery::AudioRopeBf16 { capacity: 256 },
        MultimodalQuery::ContiguousAttention {
            q_heads: 16,
            kv_heads: 16,
            head_dim: 64,
            causal: false,
            window_left: None,
            window_right: None,
            key0_sink: false,
            capacity: 256,
        },
    ] {
        let kernel = Kernel::new("retired.example.v1", None, query);
        assert!(kernel.entry().unwrap_err().contains("retired"));
        assert!(
            encoder::kernel::body(query)
                .unwrap_err()
                .0
                .contains("retired")
        );
        assert!(
            encoder::kernel::plan("retired", query, 1)
                .unwrap_err()
                .0
                .contains("retired")
        );
    }
}
