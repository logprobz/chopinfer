//! Build and audit the lagged Sinkhorn hyper-connection cubins without
//! opening a GPU: sm_120f, exact symbols, no spills or stack, shared memory
//! within the limit, PDL in every entry, F32 FMA arithmetic with no
//! tensor-core instructions or atomics, and the mix-collapse's butterfly,
//! barriers, exponentials and L2 prefetch.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::sinkhorn_hc::lagged::{self, Op, lab_catalog, lab_cubin_specs, twin_probe};

fn sass_of(cubin: &Path, symbol: &str) -> String {
    let output = Command::new("/usr/local/cuda/bin/cuobjdump")
        .args(["-sass", "-fun", symbol])
        .arg(cubin)
        .output()
        .expect("inspect SASS");
    assert!(output.status.success(), "cuobjdump failed for {symbol}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn lagged_sinkhorn_hc_entries_compile_and_audit() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join(
        format!("sinkhorn-hc-lagged-cubin-{}-{stamp}", std::process::id()),
    );
    fs::create_dir_all(&root).unwrap();
    let cubins = build(
        &toolchain,
        &lab_cubin_specs(),
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("lagged Sinkhorn hyper-connection cubins build and audit");
    let mut counts = cubins
        .iter()
        .map(|c| (c.name.as_str(), c.entries.len()))
        .collect::<Vec<_>>();
    counts.sort();
    assert_eq!(
        counts,
        [
            ("sinkhorn_hc_hcl4x5120_head_f32_reverse_streams_sm120", 1),
            ("sinkhorn_hc_hcl4x5120_head_f32_round_bf16_sm120", 1),
            ("sinkhorn_hc_hcl4x5120_head_f32_sm120", 1),
            ("sinkhorn_hc_hcl4x5120_probe_sm120", 1),
            ("sinkhorn_hc_hcl4x5120_sm120", 3),
            ("sinkhorn_hc_hclt4x5120_probe_sm120", 1),
        ]
    );
    let mut catalog = lab_catalog();
    catalog.extend(lagged::HEAD_F32_MUTANTS.map(|(suffix, _)| lagged::head_f32_mutant(suffix)));
    let twin = twin_probe();
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
            let mix = report.symbol == twin.symbol
                || catalog.iter().any(|entry| {
                    entry.symbol == report.symbol && matches!(entry.op, Op::MixCollapse { .. })
                });
            assert!(
                mix || catalog.iter().any(|entry| entry.symbol == report.symbol),
                "{} is no catalog entry",
                report.symbol
            );
            // The row's 40 KiB of streams, plus the reductions and cells,
            // as the per-sublayer audit allows.
            let limit = if mix { 41_984 + 1024 } else { 1024 };
            assert!(
                report.shared_static_bytes <= limit,
                "{} static shared {}",
                report.symbol,
                report.shared_static_bytes
            );
            let sass = sass_of(&cubin.path, &report.symbol);
            for opcode in ["ACQBULK", "PREEXIT", "FFMA"] {
                assert!(sass.contains(opcode), "{} lacks {opcode}", report.symbol);
            }
            let unrounded = report.symbol.contains("weighted_head_f32")
                && !report.symbol.contains("_round_bf16_");
            assert_eq!(
                sass.contains("BF16.F32"),
                !unrounded,
                "{} output conversion",
                report.symbol
            );
            for opcode in ["HMMA", "QMMA", "OMMA", "DMMA", "ATOM", "RED."] {
                assert!(!sass.contains(opcode), "{} has {opcode}", report.symbol);
            }
            if mix {
                for opcode in ["SHFL.BFLY", "BAR.SYNC", "MUFU.EX2", "MUFU.RSQ", "CCTL"] {
                    assert!(sass.contains(opcode), "{} lacks {opcode}", report.symbol);
                }
            }
        }
    }
    fs::remove_dir_all(root).unwrap();
}
