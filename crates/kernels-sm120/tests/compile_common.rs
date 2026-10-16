//! Compile-only CUDA coverage. This test never loads a cubin or opens a GPU.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn run(program: &Path, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("could not run {program:?}: {error}"));
    assert!(
        output.status.success(),
        "{program:?} {args:?} failed ({}):\n{}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("tool output is UTF-8")
}

fn compile(nvcc: &Path, source: &Path, cubin: &Path) {
    let source = source.to_str().expect("source path is UTF-8");
    let cubin = cubin.to_str().expect("cubin path is UTF-8");
    run(
        nvcc,
        &[
            "-arch=sm_120f",
            "-cubin",
            "-std=c++20",
            "-O3",
            source,
            "-o",
            cubin,
        ],
    );
}

#[test]
fn common_headers_compile_to_sm120_sass_without_local_memory() {
    let cuda_home = std::env::var_os("CHOPINFER_CUDA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    if !cuda_home.join("bin/nvcc").is_file() {
        eprintln!("skipping SM120 compile test: configured nvcc is absent");
        return;
    }
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA 13.2 toolchain");
    let nvcc = toolchain.home.join("bin/nvcc");
    let cuobjdump = toolchain.home.join("bin/cuobjdump");
    assert!(cuobjdump.is_file(), "cuobjdump is absent");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda/tests");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock is before epoch")
        .as_nanos();
    let scratch = std::env::temp_dir().join(format!(
        "chopinfer-sm120-compile-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&scratch).expect("create CUDA compile scratch");
    let cases: &[(&str, &[&str])] = &[
        (
            "common_mma",
            &[
                "HMMA.16816",
                "QMMA.16832",
                "QMMA.SF",
                "OMMA.SF.16864.F32.E2M1.E2M1.E8",
                "OMMA.SF.16864.F32.E2M1.E2M1.E8.4X",
                "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
                "HMMA.SP",
            ],
        ),
        (
            "common_matrix",
            &[
                "LDSM.16.M88",
                "LDSM.16.MT88",
                "LDSM.U4x16P64TO8.M816",
                "LDSM.U6x16P32TO8.M816",
                "LDSM.U4x16P64TO8.MT1616",
                "LDSM.U6x16P32TO8.MT1616",
                "STSM.16.M88",
                "STSM.16.MT88",
                "STSM.8.MT168",
            ],
        ),
        (
            "common_tma",
            &[
                "UTMALDG.1D",
                "UTMALDG.2D",
                "UTMALDG.3D",
                "UTMALDG.4D",
                "UTMALDG.5D",
                "UTMASTG.1D",
                "UTMASTG.2D",
                "UTMASTG.3D",
                "UTMASTG.4D",
                "UTMASTG.5D",
                "UTMAREDG",
                "UTMAPF",
                "UTMACCTL.PF",
                "UTMALDG.2D.GATHER4",
                "SYNCS.",
            ],
        ),
        (
            "common_control",
            &[
                "ELECT",
                "USETMAXREG",
                "PREEXIT",
                "ACQBULK",
                "UGETNEXTWORKID",
                "REDAS",
                "REDG.E.ADD.F32",
                "REDG.E.ADD.BF16x2",
            ],
        ),
        (
            "common_convert",
            &[
                "F2FP",
                "E4M3",
                "E5M2",
                "E2M1",
                "E2M3",
                "E3M2",
                "F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RP",
                "F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RZ",
                "F2FP.SATFINITE.E8.BF16.UNPACK_B_MERGE_C.RZ",
                "F2FP.BF16.E8.UNPACK_B",
            ],
        ),
        ("common_pipeline", &["SYNCS.", "USETMAXREG"]),
    ];
    for (name, opcodes) in cases {
        let source = root.join(format!("{name}.cu"));
        let cubin = scratch.join(format!("{name}.cubin"));
        compile(&nvcc, &source, &cubin);
        let file = cubin.to_str().expect("cubin path is UTF-8");
        let sass = run(&cuobjdump, &["-sass", file]);
        for opcode in *opcodes {
            assert!(sass.contains(opcode), "{name} did not emit {opcode}");
        }
        if *name == "common_mma" {
            for a in ["E4M3", "E5M2", "E3M2", "E2M3", "E2M1"] {
                for b in ["E4M3", "E5M2", "E3M2", "E2M3", "E2M1"] {
                    for prefix in ["QMMA.16832.F32", "QMMA.SF.16832.F32"] {
                        let opcode = format!("{prefix}.{a}.{b}");
                        assert!(sass.contains(&opcode), "missing {opcode}");
                    }
                }
            }
        }
        let resources = run(&cuobjdump, &["-res-usage", file]);
        let locals: Vec<_> = resources
            .split_whitespace()
            .filter_map(|word| word.strip_prefix("LOCAL:"))
            .collect();
        assert!(!locals.is_empty(), "{name} has no resource report");
        assert!(
            locals.iter().all(|bytes| *bytes == "0"),
            "{name} uses local memory:\n{resources}"
        );
    }
    fs::remove_dir_all(scratch).expect("remove CUDA compile scratch");
}
