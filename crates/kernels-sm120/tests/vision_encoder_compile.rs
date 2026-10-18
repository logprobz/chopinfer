//! Build and audit the encoder cubins of a vision tower with per-head q/k
//! RMSNorm, clamped SwiGLU MLPs and a norm-GELU-MLP projector (24 blocks of
//! width 1024 with 16 heads of 64, a 2 x 2 merge into 4096, a projector
//! through 10240) without opening a GPU: sm_120f, exact symbols, no spills
//! or stack, PDL in every row entry, and the new row entries' arithmetic in
//! their SASS. Its 14 x 14 x 2 x 3 patches (1176 columns) are zero-padded
//! to 1216 columns, since the pair GEMM takes K in multiples of 64.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use encoder_spec::{Declaration, Environment, parse};
use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::MultimodalQuery;
use kernels_sm120::encoder::{self, Entry, Shared};
use serde_json::json;

fn tower() -> Declaration {
    let text = json!({
      "schema": "chopinfer.encoder.v1",
      "model": "fixture",
      "name": "fixture-vision",
      "precision": "f32-pairs.v1",
      "checkpoint": {
        "prefix": "visual.",
        "config": [{"file": "config.json", "pointer": "/vision_config", "values": {"depth": 24}}]
      },
      "input": {
        "recipes": [{"tag": "image.patch-grid.v1", "patch": 14, "temporal_patch": 2, "merge": 2,
                     "maximum_pixels": 6_272_000}],
        "patchify": "merge-blocks.v1", "payload": "f32", "columns": 1176, "capacity_rows": 32768
      },
      "sequences": {"kind": "item"},
      "image_layout": "view-order.v1",
      "stages": [
        {"op": "pad.zero-columns.v1", "columns": 1216},
        {"op": "linear.pairs.v1", "out": 1024, "bias": true,
         "weights": {"weight": {"tensor": "patch.weight",
                                "transforms": ["flatten", {"pad_cols": 1216}]},
                     "bias": "patch.bias"}},
        {"op": "transformer.pre-norm.v1", "blocks": 24, "width": 1024,
         "norm": {"op": "norm.rms-f32.v1", "eps": 1e-5},
         "attention": {"op": "attention.varlen.qk-rms.pairs.v1", "q_heads": 16, "kv_heads": 16,
                       "head_dim": 64, "qkv_bias": true, "proj_bias": true, "mask": "full",
                       "qk_norm_eps": 1e-5},
         "rotary": {"op": "rotary.axial-2d.v1", "pairs_per_axis": 16, "theta": 10000,
                    "style": "neox", "table": "table.hf-f32-angle.v1"},
         "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 4096, "bias": true, "limit": 10},
         "weights": {
           "norm1.weight": "b.{i}.n1.weight", "norm2.weight": "b.{i}.n2.weight",
           "qkv.weight": "b.{i}.qkv.weight", "qkv.bias": "b.{i}.qkv.bias",
           "q_norm.weight": "b.{i}.qn.weight", "k_norm.weight": "b.{i}.kn.weight",
           "proj.weight": "b.{i}.proj.weight", "proj.bias": "b.{i}.proj.bias",
           "gate_up.weight": "b.{i}.gate_up.weight", "gate_up.bias": "b.{i}.gate_up.bias",
           "down.weight": "b.{i}.down.weight", "down.bias": "b.{i}.down.bias"
         }},
        {"op": "norm.rms-f32.v1", "eps": 1e-5, "weights": {"weight": "post.weight"}},
        {"op": "merge.consecutive.v1", "rows": 4},
        {"op": "linear.pairs.v1", "out": 4096, "bias": true,
         "weights": {"weight": {"tensor": "down.weight",
                                "transforms": ["flatten", {"space_to_depth_cols": 2}]},
                     "bias": "down.bias"}},
        {"op": "linear.pairs.v1", "out": 4096, "bias": false,
         "weights": {"weight": "m.proj.weight"}},
        {"op": "projector.norm-gelu-mlp.v1",
         "norm": {"op": "norm.layer-f32.v1", "eps": 1e-5},
         "gelu": "erf",
         "mlp": {"op": "mlp.swiglu-clamped.v1", "hidden": 10240, "bias": false, "limit": 10},
         "output": "bf16",
         "weights": {"norm.weight": "m.n.weight", "norm.bias": "m.n.bias",
                     "gate_up.weight": "m.gate_up.weight", "down.weight": "m.down.weight"}}
      ],
      "output": {"op": "layout.tokens.v1", "hidden": 4096, "positions": "grid-3axis.v1",
                 "media_mask": false},
      "parallel": ["replicated"]
    });
    parse(&text.to_string(), &Environment::new("fixture")).unwrap()
}

fn sass_of(cubin: &Path, symbol: &str) -> String {
    let output = Command::new("/usr/local/cuda/bin/cuobjdump")
        .args(["-sass", "-fun", symbol])
        .arg(cubin)
        .output()
        .expect("inspect SASS");
    assert!(output.status.success(), "cuobjdump failed for {symbol}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// The tower lowers to the new row entries between its GEMMs, in order.
#[test]
fn the_tower_lowers_to_its_row_entries() {
    let slots = encoder::slots(&tower(), &Shared).unwrap();
    let parts: Vec<(usize, &str)> = slots
        .iter()
        .filter(|(slot, _)| slot.stage <= 2 && slot.block.is_none() || slot.stage == 7)
        .map(|(slot, _)| (slot.stage, slot.part))
        .collect();
    assert_eq!(
        parts,
        [
            (0, "pad"),
            (1, "linear"),
            (2, "norm"),
            (2, "qkv"),
            (2, "qk_norm"),
            (2, "rotary"),
            (2, "proj"),
            (2, "up"),
            (2, "swiglu"),
            (2, "down"),
            (7, "norm"),
            (7, "gelu"),
            (7, "up"),
            (7, "swiglu"),
            (7, "down"),
        ]
    );
    let query = |stage: usize, part: &str| {
        slots
            .iter()
            .find(|(slot, _)| slot.stage == stage && slot.part == part)
            .map(|(_, entry)| entry.query)
            .unwrap()
    };
    assert_eq!(
        query(2, "qk_norm"),
        MultimodalQuery::QkRmsNormF32 {
            q_heads: 16,
            kv_heads: 16,
            head_dim: 64,
            eps_bits: 1e-5_f32.to_bits(),
            capacity: 32768,
        }
    );
    let limit_bits = 10.0_f32.to_bits();
    assert_eq!(
        query(2, "swiglu"),
        MultimodalQuery::SwigluClampedF32 {
            hidden: 4096,
            limit_bits,
            capacity: 32768,
        }
    );
    assert_eq!(
        query(7, "swiglu"),
        MultimodalQuery::SwigluClampedF32 {
            hidden: 10240,
            limit_bits,
            capacity: 32768,
        }
    );
    assert_eq!(
        query(7, "gelu"),
        MultimodalQuery::GeluRowsF32 {
            width: 4096,
            capacity: 32768,
        }
    );
    // The payload padded to 1216 columns, then the patch GEMM's K.
    assert_eq!(
        query(0, "pad"),
        MultimodalQuery::PadColumnsF32 {
            input: 1176,
            output: 1216,
            capacity: 32768,
        }
    );
    let MultimodalQuery::GemmPairs(patch) = query(1, "linear") else {
        panic!("the patch embedding is not a pair GEMM");
    };
    assert_eq!((patch.n, patch.k), (1024, 1216));
    // The projector's GEMMs: gate and up without bias, then the BF16 rows
    // of the language model, with no residual.
    let MultimodalQuery::GemmPairs(up) = query(7, "up") else {
        panic!("the projector's up projection is not a pair GEMM");
    };
    assert_eq!((up.n, up.k, up.epilogues.bias), (20480, 4096, false));
    let MultimodalQuery::GemmPairs(down) = query(7, "down") else {
        panic!("the projector's down projection is not a pair GEMM");
    };
    assert_eq!((down.n, down.k), (4096, 10240));
    assert!(!down.epilogues.residual_add && !down.epilogues.bias);
    assert_eq!(down.output, kernel_core::contract::DType::Bf16);
    // The tower's down projection adds into the residual stream.
    let MultimodalQuery::GemmPairs(down) = query(2, "down") else {
        panic!("the tower's down projection is not a pair GEMM");
    };
    assert!(down.epilogues.residual_add && down.epilogues.bias);
}

#[test]
fn the_tower_cubins_compile_and_audit() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join(
        format!("vision-encoder-cubin-{}-{stamp}", std::process::id()),
    );
    fs::create_dir_all(&root).unwrap();
    let entries = encoder::catalog(&tower(), &Shared).unwrap();
    let specs = encoder::cubin_specs(&entries).unwrap();
    let cubins = build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("vision encoder cubins build and audit");
    for cubin in &cubins {
        for report in &cubin.entries {
            eprintln!(
                "{}: {} registers, {} static shared bytes",
                report.symbol, report.registers, report.shared_static_bytes
            );
            assert_eq!(
                (report.local_bytes, report.stack_bytes),
                (0, 0),
                "{} spills",
                report.symbol
            );
        }
    }
    let rows = cubins
        .iter()
        .find(|cubin| {
            cubin
                .path
                .ends_with(format!("{}.cubin", encoder::ROWS_GEMM_CUBIN))
        })
        .expect("the rows and GEMM cubin");
    let sass_of = |symbol: &str| {
        let sass = sass_of(&rows.path, symbol);
        assert!(
            sass.contains(symbol),
            "{symbol} is not in {}",
            rows.path.display()
        );
        sass
    };
    let new: Vec<&Entry> = entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.query,
                MultimodalQuery::QkRmsNormF32 { .. }
                    | MultimodalQuery::SwigluClampedF32 { .. }
                    | MultimodalQuery::GeluRowsF32 { .. }
                    | MultimodalQuery::PadColumnsF32 { .. }
            )
        })
        .collect();
    assert_eq!(new.len(), 5, "pad, q/k norm, two clamped SwiGLUs, GELU");
    for entry in new {
        let sass = sass_of(&entry.symbol);
        fs::write(root.join(format!("{}.sass", entry.symbol)), &sass).unwrap();
        for opcode in ["ACQBULK", "PREEXIT"] {
            assert!(sass.contains(opcode), "{} lacks {opcode}", entry.symbol);
        }
        for opcode in ["HMMA", "ATOM", "RED."] {
            assert!(!sass.contains(opcode), "{} has {opcode}", entry.symbol);
        }
        let expected: &[&str] = match entry.query {
            // The butterfly over the lanes and the reciprocal square root.
            MultimodalQuery::QkRmsNormF32 { .. } => &["SHFL.BFLY", "MUFU.RSQ", "FFMA"],
            // The exponential of SiLU; the clamps are compares and selects,
            // never FMNMX, which would turn a NaN into the limit.
            MultimodalQuery::SwigluClampedF32 { .. } => &["MUFU.EX2", "FSETP", "FSEL"],
            MultimodalQuery::GeluRowsF32 { .. } => &["FMUL"],
            // A copy and zero stores: no arithmetic on the values.
            _ => &["STG.E"],
        };
        for opcode in expected {
            assert!(sass.contains(opcode), "{} lacks {opcode}", entry.symbol);
        }
        if matches!(entry.query, MultimodalQuery::PadColumnsF32 { .. }) {
            for opcode in ["FADD", "FMUL", "FFMA"] {
                assert!(!sass.contains(opcode), "{} has {opcode}", entry.symbol);
            }
        }
        if matches!(entry.query, MultimodalQuery::SwigluClampedF32 { .. }) {
            assert!(!sass.contains("FMNMX"), "{} has FMNMX", entry.symbol);
        }
    }
    eprintln!("SASS in {}", root.display());
}

/// The BF16 cast's entry alone (`cast.bf16.v1` over 5120-wide rows): PDL,
/// one conversion an element and no other arithmetic, no trap, atomic,
/// spill or stack.
#[test]
fn the_bf16_cast_compiles_to_conversions_alone() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join(format!("encoder-cast-cubin-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let entry = encoder::Kernel::new(
        "cast.bf16.v1",
        None,
        MultimodalQuery::CastBf16F32 {
            width: 5120,
            capacity: 1024,
        },
    )
    .entry()
    .unwrap();
    let [rows, _] = encoder::cubin_specs(std::slice::from_ref(&entry)).unwrap();
    let cubins = build(
        &toolchain,
        &[rows],
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("the cast cubin builds and audits");
    let [cubin] = cubins.as_slice() else {
        panic!("one cubin");
    };
    for report in &cubin.entries {
        assert_eq!((report.local_bytes, report.stack_bytes), (0, 0));
    }
    let sass = sass_of(&cubin.path, &entry.symbol);
    assert!(sass.contains(&entry.symbol));
    for opcode in ["ACQBULK", "PREEXIT", "F2FP.BF16.F32"] {
        assert!(sass.contains(opcode), "{} lacks {opcode}", entry.symbol);
    }
    for opcode in ["BPT.TRAP", "ATOM", "RED.", "FADD", "FMUL", "FFMA", "HMMA"] {
        assert!(!sass.contains(opcode), "{} has {opcode}", entry.symbol);
    }
}
