//! Build and audit the Sinkhorn hyper-connection cubins without opening a
//! GPU: sm_120f, exact symbols, no spills or stack, shared memory within the
//! limit, PDL in every entry, F32 FMA arithmetic with no tensor-core
//! instructions, and the mix-collapse's butterfly, barriers, exponentials
//! and L2 prefetch.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::DType;
use kernel_core::sinkhorn_hyper_connection::SinkhornMix;
use kernels_sm120::sinkhorn_hc::{
    Entry, Geometry, Op, SERVED, cubin_spec, entries, lab_catalog, lab_cubin_specs,
};
use kernels_sm120::{Profile, ProfileNames};

const F32_FN_PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "sinkhorn-hc-f32-fn-probe",
    tag: "f32fn_probe",
});

/// The mix-collapse with F32 fn at the 4 x 5120 width the oracle covers:
/// the fn dtype and width are free in the template, so their compilation is
/// checked here although no served shape uses them.
fn f32_fn_entries() -> Vec<Entry> {
    let geometry = Geometry {
        mix: SinkhornMix {
            width: 5120,
            eps_bits: 1e-20_f32.to_bits(),
            fn_dtype: DType::F32,
            ..SERVED.mix
        },
        capacity: 64,
    };
    entries(F32_FN_PROFILE, geometry, true)
        .into_iter()
        .filter(|entry| matches!(entry.op, Op::MixCollapse { .. }))
        .collect()
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

#[test]
fn sinkhorn_hc_entries_compile_and_audit() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join(format!("sinkhorn-hc-cubin-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let wide = f32_fn_entries();
    let mut specs = lab_cubin_specs();
    specs.push(cubin_spec(
        format!("sinkhorn_hc_{}_sm120", F32_FN_PROFILE.tag()),
        &wide,
    ));
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
    .expect("Sinkhorn hyper-connection cubins build and audit");
    let mut counts = cubins
        .iter()
        .map(|c| (c.name.as_str(), c.entries.len()))
        .collect::<Vec<_>>();
    counts.sort();
    assert_eq!(
        counts,
        [
            ("sinkhorn_hc_f32fn_probe_sm120", 2),
            ("sinkhorn_hc_hc4x4096_probe_sm120", 1),
            ("sinkhorn_hc_hc4x4096_sm120", 3),
        ]
    );
    let all = lab_catalog().into_iter().chain(wide).collect::<Vec<_>>();
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
            let entry = all
                .iter()
                .find(|entry| entry.symbol == report.symbol)
                .expect("a catalog entry");
            let mix = matches!(entry.op, Op::MixCollapse { .. });
            // The row's streams, plus the reductions and the comb cells.
            let limit = if mix { 41_984 + 1024 } else { 1024 };
            assert!(
                report.shared_static_bytes <= limit,
                "{} static shared {}",
                report.symbol,
                report.shared_static_bytes
            );
            let sass = sass_of(&cubin.path, &report.symbol);
            for opcode in ["ACQBULK", "PREEXIT", "BF16.F32"] {
                assert!(sass.contains(opcode), "{} lacks {opcode}", report.symbol);
            }
            for opcode in ["HMMA", "QMMA", "OMMA", "DMMA", "ATOM", "RED."] {
                assert!(!sass.contains(opcode), "{} has {opcode}", report.symbol);
            }
            if mix {
                for opcode in [
                    "FFMA",
                    "SHFL.BFLY",
                    "BAR.SYNC",
                    "MUFU.EX2",
                    "MUFU.RSQ",
                    "CCTL",
                ] {
                    assert!(sass.contains(opcode), "{} lacks {opcode}", report.symbol);
                }
            }
            if entry.op == Op::Expand {
                assert!(sass.contains("FFMA"), "{} lacks FFMA", report.symbol);
            }
        }
    }
    fs::remove_dir_all(root).unwrap();
}
