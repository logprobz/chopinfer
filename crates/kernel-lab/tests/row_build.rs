//! CPU-only validation of every row entry of the lab's row engine (every
//! model's, `kernel_lab::catalogs`), cubin audit, and launch metadata.

use kernel_build::{BuildOptions, EntryParam, Toolchain};
use kernel_core::contract::{Argument, RotarySource, RowQuery, RuntimeCounts};
use kernels_sm120::row::{capacity_bucket, cubin_specs, smallest_bucket};

/// Every row of the lab's row engine cubin: the kernel base's and every
/// model's.
fn catalog() -> Vec<kernels_sm120::row::RowInstantiation> {
    kernels_sm120::row::catalog(kernel_lab::catalogs())
}
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn row_catalog_compiles_and_passes_cubin_audits() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(
        !toolchain.skipped,
        "SM120 compile coverage is required on Linux"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!("row-build-{}-{stamp}", std::process::id()));
    fs::create_dir(&scratch).expect("create row build scratch");
    let specs = cubin_specs(kernel_lab::catalogs(), None);
    let entries = catalog();
    assert!(
        kernels_sm120::row::own_catalog()
            .iter()
            .all(|entry| entry.profile != model_qwen38_contract::sm120::config::PROFILE)
    );
    for profile in kernel_lab::catalogs().profiles() {
        let filtered = cubin_specs(kernel_lab::catalogs(), Some(profile));
        assert_eq!(
            filtered
                .iter()
                .map(|spec| spec.entries.len())
                .sum::<usize>(),
            entries
                .iter()
                .filter(|entry| entry.profile == profile)
                .count()
        );
    }
    assert_eq!(
        specs.iter().map(|spec| spec.entries.len()).sum::<usize>(),
        entries.len()
    );
    for entry in &entries {
        entry.contract().expect("valid row instantiation");
    }
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &scratch.join("out"),
        &scratch.join("row-kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit all row entries");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), entries.len());
    for report in &built[0].entries {
        eprintln!(
            "{} registers={} local={} stack={}",
            report.symbol, report.registers, report.local_bytes, report.stack_bytes
        );
        assert_eq!(report.local_bytes, 0, "{} spilled", report.symbol);
        let row = entries
            .iter()
            .find(|row| row.symbol == report.symbol)
            .unwrap();
        let expected_stack = if matches!(
            row.query,
            RowQuery::Rope {
                rotary_dim: 64,
                source: RotarySource::Computed,
                ..
            }
        ) {
            40
        } else {
            0
        };
        assert_eq!(
            report.stack_bytes, expected_stack,
            "{} stack frame changed",
            report.symbol
        );
        assert!(
            report.registers > 0,
            "{} has no register report",
            report.symbol
        );
    }
    let output = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .expect("cuobjdump -sass");
    assert!(output.status.success(), "cuobjdump -sass failed");
    let sass = String::from_utf8(output.stdout).expect("SASS is UTF-8");
    assert!(sass.contains("PREEXIT"), "PDL launch instruction absent");
    assert!(sass.contains("ACQBULK"), "PDL wait instruction absent");
    assert!(sass.contains("LDG.E.128"), "128-bit row loads absent");
    assert!(sass.contains("STG.E.128"), "128-bit row stores absent");
    fs::remove_dir_all(scratch).expect("remove row build scratch");
}

#[test]
fn planner_uses_capacity_and_checks_abi_kinds() {
    let rows = catalog();
    assert!(capacity_bucket(1) && capacity_bucket(32) && capacity_bucket(8192));
    assert!(!capacity_bucket(0) && !capacity_bucket(33) && !capacity_bucket(16384));
    for (rows, bucket) in [
        (0, 1),
        (1, 1),
        (3, 4),
        (32, 32),
        (33, 64),
        (50, 64),
        (954, 1024),
    ] {
        assert_eq!(smallest_bucket(rows), Some(bucket), "{rows} rows");
    }
    assert_eq!(smallest_bucket(8192), Some(8192));
    assert_eq!(smallest_bucket(8193), None);
    let norm = rows
        .iter()
        .find(|row| row.symbol == "chop_row_rms_mimo_hf4096_v1")
        .expect("MiMo RMSNorm entry");
    let counts = RuntimeCounts {
        m: 31,
        batch: 31,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    let plan = norm.plan(32, counts).expect("capacity 32 plan");
    assert_eq!(plan.grid, [32, 1, 1]);
    assert_eq!(plan.block, [256, 1, 1]);
    assert!(plan.pdl);
    assert!(norm.plan(16, counts).is_err());
    assert!(norm.plan(33, counts).is_err());
    let pointers = vec![Argument::DeviceBuffer(1); 6];
    let mut values = pointers;
    values.push(Argument::U32(1e-6_f32.to_bits()));
    assert!(norm.pack_arguments(values).is_ok());
    assert!(norm.pack_arguments(vec![Argument::U32(0); 7]).is_err());
    let page16 = rows
        .iter()
        .find(|row| row.symbol == "chop_row_rope_mimo_ga_bf16_table_p16_pos64_v1")
        .expect("page-16 RoPE entry");
    let page64 = rows
        .iter()
        .find(|row| row.symbol == "chop_row_rope_mimo_ga_bf16_table_p64_pos64_v1")
        .expect("page-64 RoPE entry");
    assert_ne!(
        page16.contract().unwrap().key,
        page64.contract().unwrap().key
    );
    let stride_names = page16
        .params
        .iter()
        .filter_map(|param| match param {
            EntryParam::U64 { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(stride_names, ["k_page_stride_bytes", "v_page_stride_bytes"]);
    let args = page16
        .params
        .iter()
        .map(|param| match param {
            EntryParam::DevicePointer { .. } => Argument::DeviceBuffer(1),
            EntryParam::U32 { .. } => Argument::U32(1),
            EntryParam::U64 { .. } => Argument::U64(4096),
            other => panic!("unexpected RoPE ABI kind {other:?}"),
        })
        .collect::<Vec<_>>();
    assert!(page16.pack_arguments(args).is_ok());
    assert_eq!(page16.plan(32, counts).unwrap().grid[1], 16);
    assert_eq!(page16.plan(512, counts).unwrap().grid[1], 4);
    let gather = rows
        .iter()
        .find(|row| row.symbol == "chop_row_gather_mimo_v1")
        .expect("MiMo gather entry");
    assert_eq!(gather.plan(32, counts).unwrap().grid[0], 32 * 8 + 256 * 31);
    let nvfp4 = rows
        .iter()
        .find(|row| row.symbol == "chop_row_quant_nvfp4_4096_v1")
        .expect("NVFP4 quantizer entry");
    assert_eq!(nvfp4.plan(32, counts).unwrap().workspace_bytes, 32 * 4 + 4);
    let permute = rows
        .iter()
        .find(|row| row.symbol == "chop_row_permute_count_mimo_v1")
        .expect("MiMo permutation entry");
    assert_eq!(permute.plan(32, counts).unwrap().workspace_bytes, 256 * 4);
    assert_eq!(permute.plan(32, counts).unwrap().block[0], 256);
    assert_eq!(permute.plan(512, counts).unwrap().block[0], 1024);
}
