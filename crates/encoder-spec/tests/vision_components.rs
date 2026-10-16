//! The per-head q/k RMSNorm attention, the clamped SwiGLU and the
//! norm-GELU-MLP projector over a fixture tower: 24 blocks of width 1024
//! (16 heads of 64, SwiGLU 4096), a 2 x 2 merge of block-major rows into a
//! linear, and a projector of 10240 to 4096. Their roles, refusals, buffer
//! plan and cost.

use encoder_spec::buffers::{self, Serving};
use encoder_spec::components::{Context, StageOp, Variant};
use encoder_spec::geometry::{Media, grid_positions};
use encoder_spec::{Environment, Patchify, parse};
use serde_json::{Value, json};

/// One 8000-token image: 32,000 patch rows.
const ROWS: u64 = 32_000;

fn fixture() -> Value {
    json!({
      "schema": "chopinfer.encoder.v1",
      "model": "fixture",
      "name": "fixture-vision",
      "precision": "f32-pairs.v1",
      "checkpoint": {
        "prefix": "visual.",
        "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 24}}]
      },
      "input": {
        "recipes": [
          {"tag": "image.patch-grid.v1", "patch": 14, "temporal_patch": 2, "merge": 2,
           "maximum_pixels": 6_272_000}
        ],
        "patchify": "merge-blocks.v1",
        "payload": "f32",
        "columns": 1176,
        "capacity_rows": 32768
      },
      "sequences": {"kind": "item"},
      "image_layout": "view-order.v1",
      "stages": [
        {
          "op": "linear.pairs.v1", "out": 1024, "bias": true,
          "weights": {
            "weight": {"tensor": "patch_embed.proj.weight", "transforms": ["flatten"]},
            "bias": "patch_embed.proj.bias"
          }
        },
        {
          "op": "transformer.pre-norm.v1", "blocks": 24, "width": 1024,
          "norm": {"op": "norm.rms-f32.v1", "eps": 1e-5},
          "attention": {
            "op": "attention.varlen.qk-rms.pairs.v1", "q_heads": 16, "kv_heads": 16,
            "head_dim": 64, "qkv_bias": true, "proj_bias": true, "mask": "full",
            "qk_norm_eps": 1e-5
          },
          "rotary": {
            "op": "rotary.axial-2d.v1", "pairs_per_axis": 16, "theta": 10000,
            "style": "neox", "table": "table.hf-f32-angle.v1"
          },
          "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 4096, "bias": true, "limit": 10},
          "weights": {
            "norm1.weight": "blocks.{i}.norm1.weight",
            "norm2.weight": "blocks.{i}.norm2.weight",
            "qkv.weight": "blocks.{i}.attn.qkv.weight",
            "qkv.bias": "blocks.{i}.attn.qkv.bias",
            "q_norm.weight": "blocks.{i}.attn.q_norm.weight",
            "k_norm.weight": "blocks.{i}.attn.k_norm.weight",
            "proj.weight": "blocks.{i}.attn.proj.weight",
            "proj.bias": "blocks.{i}.attn.proj.bias",
            "gate_up.weight": {"concat_rows": [
              "blocks.{i}.mlp.gate_proj.weight", "blocks.{i}.mlp.up_proj.weight"]},
            "gate_up.bias": {"concat_rows": [
              "blocks.{i}.mlp.gate_proj.bias", "blocks.{i}.mlp.up_proj.bias"]},
            "down.weight": "blocks.{i}.mlp.down_proj.weight",
            "down.bias": "blocks.{i}.mlp.down_proj.bias"
          }
        },
        {
          "op": "norm.rms-f32.v1", "eps": 1e-5,
          "weights": {"weight": "post_layernorm.weight"}
        },
        {"op": "merge.consecutive.v1", "rows": 4},
        {
          "op": "linear.pairs.v1", "out": 4096, "bias": true,
          "weights": {
            "weight": {"tensor": "downsample.weight",
                       "transforms": ["flatten", {"space_to_depth_cols": 2}]},
            "bias": "downsample.bias"
          }
        },
        {
          "op": "linear.pairs.v1", "out": 4096, "bias": false,
          "weights": {"weight": "merger.proj.weight"}
        },
        {
          "op": "projector.norm-gelu-mlp.v1",
          "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
          "gelu": "erf",
          "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 10240, "bias": false, "limit": 10},
          "output": "bf16",
          "weights": {
            "norm.weight": "merger.post_projection_norm.weight",
            "norm.bias": "merger.post_projection_norm.bias",
            "gate_up.weight": {"concat_rows": [
              "merger.gate_proj.weight", "merger.up_proj.weight"]},
            "down.weight": "merger.down_proj.weight"
          }
        }
      ],
      "output": {"op": "layout.tokens.v1", "hidden": 4096, "positions": "grid-3axis.v1",
                 "media_mask": false},
      "parallel": ["replicated", "record-parallel", "sequence-split"]
    })
}

fn environment() -> Environment {
    Environment::new("fixture")
}

#[track_caller]
fn refuses(edit: impl FnOnce(&mut Value), needle: &str) {
    let mut value = fixture();
    edit(&mut value);
    let error = parse(&value.to_string(), &environment()).expect_err("accepted the edit");
    assert!(error.0.contains(needle), "expected `{needle}` in: {error}");
}

#[test]
fn the_fixture_validates_with_the_new_roles() {
    let declaration = parse(&fixture().to_string(), &environment()).unwrap();
    let context = Context {
        model: &declaration.model,
        precision: declaration.precision,
        patchify: declaration.input.patchify,
        sequences: &declaration.sequences,
        capacity_rows: u64::from(declaration.input.capacity_rows),
    };
    let inputs = declaration.inputs().unwrap();
    let roles = |index: usize| -> Vec<(String, Vec<u64>)> {
        let component = declaration.stages[index].op.component().unwrap();
        component
            .roles(&context, &inputs[index])
            .unwrap()
            .into_iter()
            .map(|role| (role.name, role.shape))
            .collect()
    };
    let transformer = roles(1);
    for (name, shape) in [
        ("q_norm.weight", vec![64]),
        ("k_norm.weight", vec![64]),
        ("gate_up.weight", vec![8192, 1024]),
        ("gate_up.bias", vec![8192]),
        ("down.weight", vec![1024, 4096]),
    ] {
        assert!(
            transformer.contains(&(name.to_owned(), shape.clone())),
            "{name} {shape:?} in {transformer:?}"
        );
    }
    assert_eq!(
        roles(6),
        [
            ("norm.weight".to_owned(), vec![4096]),
            ("norm.bias".to_owned(), vec![4096]),
            ("gate_up.weight".to_owned(), vec![20480, 4096]),
            ("down.weight".to_owned(), vec![4096, 10240]),
        ]
    );
    // The merge reads whole 2 x 2 blocks of block-major rows, so the
    // downsample linear takes 4 x 1024 position-major columns.
    assert_eq!(inputs[4].width().unwrap(), 4096);
    assert_eq!(declaration.output.op.id(), "layout.tokens.v1");
    let StageOp::Projector(_) = &declaration.stages[6].op else {
        panic!("stage 6 is not a projector");
    };
    // The digest is taken over the typed parameters and survives a reparse.
    let again = parse(&declaration.canonical_json(), &environment()).unwrap();
    assert_eq!(again.digest(), declaration.digest());
}

#[test]
fn the_new_variants_refuse_what_they_cannot_run() {
    refuses(
        |d| {
            d["stages"][1]["attention"]
                .as_object_mut()
                .unwrap()
                .remove("qk_norm_eps");
        },
        "needs `qk_norm_eps`",
    );
    refuses(
        |d| d["stages"][1]["attention"]["op"] = "attention.varlen.pairs.v1".into(),
        "`qk_norm_eps` belongs to attention.varlen.qk-rms.pairs.v1",
    );
    refuses(
        |d| d["stages"][1]["attention"]["qk_norm_eps"] = 0.into(),
        "qk_norm_eps 0.0 is 0.0 in F32",
    );
    refuses(
        |d| d["stages"][1]["mlp"]["limit"] = (-1).into(),
        "limit -1.0 is -1.0 in F32",
    );
    refuses(
        |d| d["stages"][1]["mlp"]["limit"] = 1e39.into(),
        "is inf in F32",
    );
    refuses(
        |d| d["stages"][1]["mlp"]["round_linear"] = true.into(),
        "unknown field `round_linear`",
    );
    refuses(
        |d| d["stages"][6]["output"] = "f32".into(),
        "`output: f32` is the default",
    );
    refuses(
        |d| d["stages"][6]["norm"]["op"] = "norm.layer.v1".into(),
        "not defined under precision policy",
    );
    refuses(
        |d| d["stages"][6]["mlp"]["hidden"] = 0.into(),
        "hidden is zero",
    );
    refuses(
        |d| d["stages"][6]["act"] = "erf".into(),
        "unknown field `act`",
    );
    // A missing norm role, and a weight the projector does not read.
    refuses(
        |d| {
            d["stages"][6]["weights"]
                .as_object_mut()
                .unwrap()
                .remove("norm.bias");
        },
        "norm.bias",
    );
    refuses(
        |d| d["stages"][6]["weights"]["down.bias"] = "merger.down_proj.bias".into(),
        "down.bias",
    );
    // Rounded rows read by a later stage.
    refuses(
        |d| {
            let stages = d["stages"].as_array_mut().unwrap();
            stages.push(json!({"op": "norm.rms-f32.v1", "eps": 1e-5,
                               "weights": {"weight": "extra.weight"}}));
        },
        "reads rows already rounded to BF16",
    );
    refuses(
        |d| d["stages"][3]["op"] = "projector.norm-gelu-mlp.v2".into(),
        "unknown variant `projector.norm-gelu-mlp.v2`",
    );
}

/// HF's vision position ids (`transformers.vision_utils
/// .get_vision_position_ids`): an `ij` meshgrid reshaped to `(h / m, m,
/// w / m, m)`, its middle axes swapped, flattened.
fn hf_block_major(h: u64, w: u64, m: u64) -> Vec<[u64; 2]> {
    let mut out = vec![[0; 2]; (h * w) as usize];
    let (bh, bw) = (h / m, w / m);
    // Source index of the reshaped (a, b, c, d) element is
    // ((a * m + b) * bw + c) * m + d; after the transpose the flattened
    // order is (a, c, b, d).
    let mut k = 0;
    for a in 0..bh {
        for c in 0..bw {
            for b in 0..m {
                for d in 0..m {
                    let source = ((a * m + b) * bw + c) * m + d;
                    out[k] = [source / w, source % w];
                    k += 1;
                }
            }
        }
    }
    out
}

#[test]
fn merge_block_positions_follow_the_reference_order() {
    for (h, w) in [(2, 2), (4, 6), (8, 2), (10, 14)] {
        assert_eq!(
            grid_positions(Patchify::MergeBlocks, 2, h, w),
            hf_block_major(h, w, 2),
            "{h} x {w}"
        );
    }
    // The geometry's rotary positions repeat the grid's per temporal grid.
    let declaration = parse(&fixture().to_string(), &environment()).unwrap();
    let geometry = declaration
        .geometry("image.patch-grid.v1", Media::Grid([1, 4, 6]))
        .unwrap();
    let step = geometry.runs[0]
        .steps
        .iter()
        .find(|step| step.stage == 1)
        .unwrap();
    let Some(encoder_spec::geometry::Positions::Grid(positions)) = &step.positions else {
        panic!("the transformer has no grid positions");
    };
    assert_eq!(positions, &hf_block_major(4, 6, 2));
}

/// The working memory of runs of one 8000-token image (32,000 patch rows,
/// 8,000 merged rows), F32 throughout:
/// - `b`: rows x 8192, the transformer's gate and up rows, larger than the
///   payload (rows x 1176), q, k and v (rows x 3072) and the merged rows;
/// - `a`: the larger of rows x 4096 (the SwiGLU product) and merged x 20480
///   (the projector's gate and up rows);
/// - `stream`: rows x 1024, the residual stream;
/// - rotary cos and sin: rows x 32 each; `cu_seqlens` and live counts.
#[test]
fn one_image_plans_its_working_memory() {
    let declaration = parse(&fixture().to_string(), &environment()).unwrap();
    let serving = Serving {
        arena_rows: 0,
        prefill_rows: 0,
        staging_bytes: 0,
    };
    let plan = buffers::plan(&[&declaration], ROWS, serving).unwrap();
    let merged = ROWS / 4;
    let region = |name: &str| plan.region(name).unwrap().bytes;
    assert_eq!(region("b"), ROWS * 8192 * 4);
    assert!(ROWS * 8192 > (ROWS * 1176).max(ROWS * 3072).max(merged * 10240));
    assert_eq!(region("a"), (ROWS * 4096).max(merged * 20480) * 4);
    assert_eq!(region("stream"), ROWS * 1024 * 4);
    let rotary: u64 = ["cos", "sin"].iter().map(|name| region(name)).sum();
    assert_eq!(rotary, 2 * ROWS * 32 * 4);
    let named = region("a") + region("b") + region("stream") + rotary;
    assert_eq!(named, 1_843_200_000);
    let small = plan.bytes() - named;
    assert!(small < 64, "{small} bytes of sequences and counts");
    let names: Vec<&str> = plan.regions.iter().map(|region| region.name).collect();
    assert!(!names.contains(&"hidden"), "{names:?}");
}

#[test]
fn costs_count_the_norms_and_the_projector() {
    let declaration = parse(&fixture().to_string(), &environment()).unwrap();
    let (h, w) = (8, 12);
    let rows = h * w;
    let merged = rows / 4;
    let geometry = declaration
        .geometry("image.patch-grid.v1", Media::Grid([1, h, w]))
        .unwrap();
    let costs = declaration.cost(&geometry.runs[0]).unwrap();
    let gemms = 2 * rows * 1024 * (3072 + 1024 + 2 * 4096 + 4096);
    let attention = 4 * 16 * 64 * rows * rows;
    let norms = 2 * 8 * rows * 1024 + 8 * rows * 32 * 64;
    assert_eq!(costs[1].flops, 24 * (gemms + attention + norms));
    let projector = 8 * merged * 4096 + 2 * merged * 4096 * 20480 + 2 * merged * 10240 * 4096;
    assert_eq!(costs[6].flops, projector);
}

/// The fixture with its 1176-wide patches zero-padded to 1216 columns
/// before the patch GEMM, whose weight takes zero columns to match.
fn padded() -> Value {
    let mut value = fixture();
    let stages = value["stages"].as_array_mut().unwrap();
    stages.insert(0, json!({"op": "pad.zero-columns.v1", "columns": 1216}));
    stages[1]["weights"]["weight"]["transforms"] = json!(["flatten", {"pad_cols": 1216}]);
    value
}

#[test]
fn zero_column_padding_widens_the_patch_rows() {
    let declaration = parse(&padded().to_string(), &environment()).unwrap();
    let inputs = declaration.inputs().unwrap();
    assert_eq!(inputs[0].width().unwrap(), 1176);
    assert_eq!(inputs[1].width().unwrap(), 1216);
    // The padded rows sit in `a`, which the projector already sizes
    // larger, so the working memory is unchanged.
    let serving = Serving {
        arena_rows: 0,
        prefill_rows: 0,
        staging_bytes: 0,
    };
    let plan = buffers::plan(&[&declaration], ROWS, serving).unwrap();
    let unpadded = parse(&fixture().to_string(), &environment()).unwrap();
    let before = buffers::plan(&[&unpadded], ROWS, serving).unwrap();
    assert_eq!(plan.bytes(), before.bytes());
    let pad = plan
        .regions
        .iter()
        .find(|region| {
            region
                .uses
                .iter()
                .any(|at| at.what == "pad.zero-columns.v1")
        })
        .unwrap();
    assert_eq!(pad.name, "a");

    let refuses = |edit: &dyn Fn(&mut Value), needle: &str| {
        let mut value = padded();
        edit(&mut value);
        let error = parse(&value.to_string(), &environment()).expect_err("accepted the edit");
        assert!(error.0.contains(needle), "expected `{needle}` in: {error}");
    };
    refuses(
        &|d| d["stages"][0]["columns"] = 1176.into(),
        "not wider than its input's 1176",
    );
    refuses(
        &|d| d["stages"][0]["rows"] = 4.into(),
        "unknown field `rows`",
    );
}
