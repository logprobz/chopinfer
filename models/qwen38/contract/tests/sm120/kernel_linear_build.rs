//! CPU-only checks of the linear-attention catalog, planner and ptxas audit.
//! No cubin is loaded on a GPU.

use std::collections::BTreeSet;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::{NumericalContract, RecurrentPhase, RecurrentQuery};
use kernels_sm120::linear::{Bucket, Stage, cubin_specs};
use model_qwen38_contract::sm120::{
    catalogs,
    linear::{catalog, qwen38_catalog},
};

#[test]
fn qwen38_catalog_plans_capacity_grids() {
    // TP4's entries, then TP2's.
    assert_eq!(
        catalog().len(),
        model_qwen38_contract::sm120::PROFILES.len() * 8
    );
    for profile in model_qwen38_contract::sm120::PROFILES {
        let shape = model_qwen38_contract::sm120::geometry::shape(profile).unwrap();
        let key_heads = shape.linear_key_heads();
        let value_heads = shape.linear_value_heads();
        let value_tiles =
            shape.geometry.linear_value_head() / kernels_sm120::linear::RECURRENT_ROWS;
        let conv_blocks = shape.linear_qkv().div_ceil(256);
        let workspace_per_chunk = u64::from(value_heads) * kernels_sm120::linear::WORK_FLOATS * 4
            + u64::from(
                key_heads
                    * 2
                    * kernels_sm120::linear::CHUNK
                    * (shape.geometry.linear_head() + 8)
                    * 2,
            )
            + u64::from(
                value_heads * kernels_sm120::linear::CHUNK * shape.geometry.linear_value_head() * 2,
            );
        let entries = qwen38_catalog(profile);
        assert_eq!(entries.len(), 8);
        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for entry in &entries {
            assert_eq!(entry.profile, profile);
            let spec = entry.contract().unwrap();
            assert!(keys.insert(format!("{:?}", spec.key)), "{}", entry.symbol);
            ids.insert(entry.query.numerical_contract_id().unwrap().to_string());
            let full = entry.capacity();
            let plan = entry
                .plan(full, full.sequences.min(full.rows), full.sequences)
                .unwrap();
            assert!(plan.pdl);
            assert!(plan.dynamic_shared_bytes <= 101_376);
            // Grids depend on the bucket only, never on live counts.
            assert_eq!(entry.plan(full, 1, 1).unwrap(), plan, "{}", entry.symbol);
            assert!(entry.plan(full, full.rows + 1, full.sequences).is_err());
            assert!(entry.plan(full, 1, full.sequences + 1).is_err());
            let oversized = Bucket {
                rows: full.rows * 2,
                sequences: full.sequences * 2,
            };
            assert!(entry.plan(oversized, 1, 1).is_err());
        }
        for expected in [
            "recurrent.causal-conv1d-depthwise-silu.prefill.v1",
            "recurrent.causal-conv1d-depthwise-silu.decode.v1",
            "recurrent.causal-conv1d-depthwise-silu.verify.v1",
            "recurrent.gated-delta-rule.prefill.v1",
            "recurrent.gated-delta-rule.decode.v1",
            "recurrent.gated-delta-rule.verify.v1",
            "row.rmsnorm-gated-sigmoid.v1",
        ] {
            assert!(ids.contains(expected), "missing {expected}");
        }
        let find = |symbol: &str| {
            entries
                .iter()
                .find(|e| e.symbol == symbol.replacen("qwen38", profile.tag(), 1))
                .unwrap()
        };
        let bucket = |rows, sequences| Bucket { rows, sequences };
        let decode = find("chop_linear_qwen38_delta_decode_v1");
        assert_eq!(
            decode.plan(bucket(32, 32), 1, 1).unwrap().grid,
            [value_tiles, value_heads, 32]
        );
        assert_eq!(
            decode.plan(bucket(1, 1), 1, 1).unwrap().grid,
            [value_tiles, value_heads, 1]
        );
        assert!(decode.plan(bucket(32, 32), 2, 1).is_err());
        assert!(decode.plan(bucket(8, 4), 1, 1).is_err());
        let verify = find("chop_linear_qwen38_delta_verify_v1");
        assert!(matches!(
            verify.query,
            RecurrentQuery::GatedDeltaRule {
                phase: RecurrentPhase::Verify { max_rows: 8 },
                ..
            }
        ));
        // 256 sequences of up to 8 rows (CONC-QWEN); the 32-sequence buckets
        // plan as before.
        assert_eq!(verify.capacity(), bucket(2048, 256));
        assert_eq!(
            verify.plan(bucket(256, 32), 64, 8).unwrap().grid,
            [value_tiles, value_heads, 32]
        );
        assert!(verify.plan(bucket(256, 32), 65, 8).is_err());
        for sequences in [33, 64, 100, 128, 255, 256] {
            assert_eq!(
                verify
                    .plan(bucket(sequences, sequences), sequences, sequences)
                    .unwrap()
                    .grid,
                [value_tiles, value_heads, sequences]
            );
            assert_eq!(
                decode
                    .plan(bucket(sequences, sequences), sequences, sequences)
                    .unwrap()
                    .grid,
                [value_tiles, value_heads, sequences]
            );
            let conv = find("chop_linear_qwen38_conv_verify_v1");
            assert_eq!(
                conv.plan(bucket(sequences, sequences), sequences, sequences)
                    .unwrap()
                    .grid,
                [conv_blocks, sequences, 1]
            );
        }
        assert_eq!(decode.capacity(), bucket(256, 256));
        assert!(decode.plan(bucket(512, 512), 1, 1).is_err());
        let prepare = find("chop_linear_qwen38_delta_prefill_prepare_v1");
        let scan = find("chop_linear_qwen38_delta_prefill_scan_v1");
        assert_eq!(prepare.stage, Stage::Prepare);
        assert_eq!(scan.stage, Stage::Scan);
        assert_eq!(prepare.capacity(), bucket(8192, 256));
        // The 32-sequence bucket a 32-item instance plans, as before.
        let full = bucket(8192, 32);
        let (prepare_plan, scan_plan) = (
            prepare.plan(full, 8192, 1).unwrap(),
            scan.plan(full, 8192, 1).unwrap(),
        );
        assert_eq!(prepare_plan.grid, [key_heads, 160, 1]);
        assert_eq!(scan_plan.grid, [value_tiles, value_heads, 32]);
        assert_eq!(prepare_plan.workspace_bytes, scan_plan.workspace_bytes);
        assert_eq!(prepare_plan.workspace_bytes, 160 * workspace_per_chunk);
        assert_eq!(
            prepare_plan.dynamic_shared_bytes,
            70_176 + value_heads / key_heads * 1280
        );
        assert_eq!(scan_plan.dynamic_shared_bytes, 100_136);
        let small = prepare.plan(bucket(100, 2), 100, 2).unwrap();
        assert_eq!(small.grid, [key_heads, 4, 1]);
        assert_eq!(small.workspace_bytes, 4 * workspace_per_chunk);
        let conv = find("chop_linear_qwen38_conv_prefill_v1");
        assert_eq!(
            conv.plan(full, 8192, 32).unwrap().grid,
            [conv_blocks, 160, 1]
        );
        // 256 prefill items: one partial chunk each on top of the row chunks.
        let wide = prepare.plan(bucket(8192, 256), 8192, 256).unwrap();
        assert_eq!(wide.grid, [key_heads, 384, 1]);
        assert_eq!(wide.workspace_bytes, 384 * workspace_per_chunk);
        assert_eq!(
            scan.plan(bucket(8192, 256), 8192, 256).unwrap().grid,
            [value_tiles, value_heads, 256]
        );
        let norm = find("chop_linear_qwen38_gated_norm_v1");
        assert_eq!(
            norm.plan(norm.capacity(), 8192, 8192).unwrap().grid,
            [8192, 1, 1]
        );
    }
}

#[test]
fn qwen38_linear_entries_compile_without_spills() {
    assert!(
        cubin_specs(
            catalogs(),
            Some(
                kernels_sm120::Model::new(
                    kernels_sm120::profile_names!("other-model", "other", 4, 2),
                    model_qwen38_contract::sm120::config::CONFIG
                )
                .at(4)
            )
        )
        .is_empty()
    );
    let specs = cubin_specs(catalogs(), None);
    assert_eq!(specs.len(), model_qwen38_contract::sm120::PROFILES.len());
    assert_eq!(specs[0].name, "linear_qwen38_flash_sm120");
    for profile in model_qwen38_contract::sm120::PROFILES {
        let name = model_qwen38_contract::sm120::linear::cubin_name(profile);
        let spec = specs.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(spec.entries.len(), 8);
    }
    let toolchain = Toolchain::detect().expect("CUDA 13.2 compiler");
    assert!(!toolchain.skipped, "ptxas coverage is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!("linear-build-{}-{stamp}", std::process::id()));
    fs::create_dir(&scratch).unwrap();
    let built = build(
        &toolchain,
        &specs,
        &scratch.join("out"),
        &scratch.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    for entry in built.iter().flat_map(|cubin| &cubin.entries) {
        assert_eq!(entry.local_bytes, 0, "{} spilled", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{} has stack", entry.symbol);
        eprintln!("{} registers {}", entry.symbol, entry.registers);
    }
    fs::remove_dir_all(scratch).unwrap();
}

/// TP4's derived GEMM shapes equal the literals the TP4 catalogs, their
/// locks and tuning keys were written with (REVIEW-A4 finding 1).
#[test]
fn derived_tp4_shapes_equal_the_former_literals() {
    assert_eq!(
        (
            model_qwen38_contract::sm120::linear::QWEN38_KEY_HEADS,
            model_qwen38_contract::sm120::linear::QWEN38_CONV_CHANNELS
        ),
        (4, 2560)
    );
    let tp4 = model_qwen38_contract::sm120::config::RankShape { ranks: 4 };
    assert_eq!(
        (
            tp4.linear_qkv(),
            tp4.linear_value(),
            tp4.attn_row(),
            tp4.q_proj_rows()
        ),
        (2560, 1536, 3584, 3072)
    );
    assert_eq!(model_qwen38_contract::sm120::config::EXPERTS, 512);
}
