//! Build and audit the per-channel-decay delta rule cubin without opening a
//! GPU: sm_120f, exact symbols, no spills or stack, shared memory within the
//! limit, PDL in every entry, and F32 FMA arithmetic with no tensor-core
//! (TF32 or BF16) instructions.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::per_channel_decay_delta_rule::DecayGate;
use kernels_sm120::linear::Stage;
use kernels_sm120::linear::channel_decay::{
    Entry, Query, SERVED, SHARED_LIMIT, catalog, cubin_spec, entries,
};
use kernels_sm120::{Profile, ProfileNames};

const SOFTPLUS_PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "channel-decay-softplus-probe",
    tag: "softplus_probe",
});

/// The softplus gate's decode, verify and prefill entries: the form is free
/// in the templates, so its compilation is checked here although no served
/// shape uses it.
fn softplus_entries() -> Vec<Entry> {
    entries(
        SOFTPLUS_PROFILE,
        kernels_sm120::linear::channel_decay::Geometry {
            gate: DecayGate::Softplus,
            ..SERVED
        },
    )
    .into_iter()
    .filter(|entry| matches!(entry.query, Query::Rule(_)) && entry.symbol.contains("delta"))
    .collect()
}

fn sass_of(cubin: &PathBuf, symbol: &str) -> String {
    let output = Command::new("/usr/local/cuda/bin/cuobjdump")
        .args(["-sass", "-fun", symbol])
        .arg(cubin)
        .output()
        .expect("inspect SASS");
    assert!(output.status.success(), "cuobjdump failed for {symbol}");
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn channel_decay_entries_compile_and_audit() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into())).join(
        format!("channel-decay-cubin-{}-{stamp}", std::process::id()),
    );
    fs::create_dir_all(&root).unwrap();
    let served = catalog();
    let softplus = softplus_entries();
    let specs = [
        cubin_spec(served[0].profile, &served),
        cubin_spec(SOFTPLUS_PROFILE, &softplus),
    ];
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
    .expect("per-channel-decay delta rule cubins build and audit");
    assert_eq!(cubins.len(), 2);
    assert_eq!(cubins[0].entries.len(), 8);
    assert_eq!(cubins[1].entries.len(), 4);
    for (cubin, list) in cubins.iter().zip([&served, &softplus]) {
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
            // SM120 reports its 1 KiB of reserved system shared memory.
            assert!(
                report.shared_static_bytes <= 1024,
                "{} static shared {}",
                report.symbol,
                report.shared_static_bytes
            );
        }
        for entry in list.iter() {
            assert!(entry.dynamic_shared_bytes() <= SHARED_LIMIT);
            let sass = sass_of(&cubin.path, &entry.symbol);
            for opcode in ["ACQBULK", "PREEXIT"] {
                assert!(sass.contains(opcode), "{} lacks {opcode}", entry.symbol);
            }
            for opcode in ["HMMA", "QMMA", "OMMA", "DMMA"] {
                assert!(!sass.contains(opcode), "{} has {opcode}", entry.symbol);
            }
            let rule = matches!(entry.query, Query::Rule(_));
            if rule {
                assert!(sass.contains("FFMA"), "{} lacks FFMA", entry.symbol);
            }
            let decays = entry.symbol.contains("delta") && entry.stage != Stage::Scan;
            if decays {
                assert!(sass.contains("MUFU.EX2"), "{} lacks MUFU.EX2", entry.symbol);
            }
            if entry.stage != Stage::Single {
                assert!(sass.contains("BAR.SYNC"), "{} lacks BAR.SYNC", entry.symbol);
            }
            if entry.symbol.contains("delta_decode") || entry.symbol.contains("delta_verify") {
                assert!(
                    sass.contains("SHFL.BFLY"),
                    "{} lacks SHFL.BFLY",
                    entry.symbol
                );
            }
        }
    }
    fs::remove_dir_all(root).unwrap();
}
