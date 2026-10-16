//! Compile and audit every grouped expert entry of the kernel base's own
//! catalog without loading a cubin (a model audits its own entries).

use kernel_build::{BuildOptions, Toolchain};
use kernel_core::contract::RuntimeCounts;
use kernels_sm120::moe;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn catalog_entries_build_and_emit_mma_tma() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(
        !toolchain.skipped,
        "MoE compile audit requires Linux and CUDA"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "chopinfer-moe-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    let lock = out_dir.join("kernels.lock");
    let specs = moe::cubin_specs(&kernels_sm120::Catalogs::base(), None);
    let catalog = moe::catalog(&kernels_sm120::Catalogs::base());
    let expected = catalog
        .iter()
        .map(|v| 1 + usize::from(v.reduction_symbol().is_some()))
        .sum::<usize>();
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out_dir,
        &lock,
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit MoE cubins");
    assert_eq!(built.len(), specs.len());
    assert_eq!(
        built.iter().map(|c| c.entries.len()).sum::<usize>(),
        expected
    );
    if let Some(report_path) = std::env::var_os("CHOPINFER_MOE_REPORT_PATH") {
        let mut report = String::from(
            "symbol\tregisters\tstatic_shared_bytes\tdynamic_shared_bytes\tlocal_bytes\n",
        );
        for cubin in &built {
            for entry in &cubin.entries {
                let variant = catalog
                    .iter()
                    .find(|v| {
                        entry.symbol == v.symbol
                            || entry.symbol == v.symbol.replace("_v1", "_reduce_v1")
                    })
                    .expect("entry belongs to catalog");
                let dynamic = if entry.symbol.ends_with("_reduce_v1") {
                    0
                } else {
                    variant
                        .plan(RuntimeCounts {
                            m: variant.tokens_capacity,
                            batch: 0,
                            context: 0,
                            routed_rows: variant.query.m_capacity,
                            split_k: variant.knobs.split_k_capacity,
                        })
                        .unwrap()
                        .dynamic_shared_bytes
                };
                report.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\n",
                    entry.symbol,
                    entry.registers,
                    entry.shared_static_bytes,
                    dynamic,
                    entry.local_bytes
                ));
            }
        }
        fs::write(report_path, report).expect("write MoE resource report");
    }
    for cubin in &built {
        for entry in &cubin.entries {
            assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
            let variant = catalog
                .iter()
                .find(|v| {
                    entry.symbol == v.symbol
                        || entry.symbol == v.symbol.replace("_v1", "_reduce_v1")
                })
                .expect("entry belongs to catalog");
            let dynamic = if entry.symbol.ends_with("_reduce_v1") {
                0
            } else {
                variant
                    .plan(RuntimeCounts {
                        m: variant.tokens_capacity,
                        batch: 0,
                        context: 0,
                        routed_rows: variant.query.m_capacity,
                        split_k: variant.knobs.split_k_capacity,
                    })
                    .unwrap()
                    .dynamic_shared_bytes
            };
            assert!(
                entry.shared_static_bytes + dynamic <= 101_376,
                "{} exceeds SM120 shared memory",
                entry.symbol
            );
        }
        let output = Command::new(toolchain.home.join("bin/cuobjdump"))
            .arg("-sass")
            .arg(&cubin.path)
            .output()
            .expect("cuobjdump SASS");
        assert!(
            output.status.success(),
            "cuobjdump failed for {}",
            cubin.name
        );
        let sass = String::from_utf8(output.stdout).expect("SASS is UTF-8");
        let mma = if cubin.name.contains("_a8") && cubin.name.contains("nvfp4") {
            "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"
        } else if cubin.name.contains("_a8") && cubin.name.contains("fp8block") {
            "QMMA.16832"
        } else if cubin.name.contains("_a8")
            && (cubin.name.contains("mxfp4") || cubin.name.contains("mxfp8"))
        {
            "QMMA.SF.16832"
        } else {
            "HMMA.16816"
        };
        let expected = vec![mma, "UTMALDG.2D", "UTMAPF"];
        for opcode in expected {
            assert!(sass.contains(opcode), "{} lacks {opcode}", cubin.name);
        }
        if cubin.name.contains("_a8") && cubin.name.contains("nvfp4") {
            assert!(
                sass.contains("LDSM"),
                "{} lacks native FP4 ldmatrix",
                cubin.name
            );
        }
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}
