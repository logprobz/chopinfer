//! Build and audit the quantized SwiGLU cubin without opening a GPU:
//! sm_120f, exact symbols, no spills, stack or shared memory, PDL in every
//! entry, the exponential and the warp's amax shuffles, and no tensor-core
//! instructions or atomics.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernels_sm120::row::silu_mul_quantize::{LAB_CUBIN, lab_cubin_specs, lab_entries};

#[test]
fn silu_mul_quantize_entries_compile_and_audit() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join(format!("silu-mul-quantize-{}-{stamp}", std::process::id()));
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
    .expect("quantized SwiGLU cubin builds and audits");
    assert_eq!(cubins.len(), 1);
    assert_eq!(cubins[0].name, LAB_CUBIN);
    let symbols = lab_entries()
        .into_iter()
        .map(|e| e.symbol)
        .collect::<Vec<_>>();
    assert_eq!(cubins[0].entries.len(), symbols.len());
    for report in &cubins[0].entries {
        assert!(symbols.contains(&report.symbol), "{}", report.symbol);
        assert_eq!(
            (
                report.local_bytes,
                report.stack_bytes,
                report.shared_static_bytes
            ),
            (0, 0, 0),
            "{}",
            report.symbol
        );
        let output = Command::new("/usr/local/cuda/bin/cuobjdump")
            .args(["-sass", "-fun", &report.symbol])
            .arg(&cubins[0].path)
            .output()
            .expect("inspect SASS");
        assert!(output.status.success());
        let sass = String::from_utf8_lossy(&output.stdout);
        for opcode in ["ACQBULK", "PREEXIT", "MUFU.EX2", "SHFL.BFLY", "F2FP"] {
            assert!(sass.contains(opcode), "{} lacks {opcode}", report.symbol);
        }
        for opcode in ["HMMA", "QMMA", "OMMA", "ATOM", "RED."] {
            assert!(!sass.contains(opcode), "{} has {opcode}", report.symbol);
        }
    }
    fs::remove_dir_all(root).unwrap();
}
