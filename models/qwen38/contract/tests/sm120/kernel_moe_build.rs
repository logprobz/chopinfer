//! Compile and audit Qwen3.8's grouped expert and route cubins without
//! loading one, with its catalog entries registered.

use kernel_build::{BuildOptions, Toolchain};
use kernel_core::contract::RuntimeCounts;
use kernels_sm120::moe;
use model_qwen38_contract::sm120;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn qwen_expert_and_route_cubins_build_and_emit_their_instructions() {
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
        "chopinfer-qwen-moe-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    let specs = model_qwen38_contract::sm120::PROFILES
        .into_iter()
        .flat_map(|profile| {
            moe::cubin_specs(model_qwen38_contract::sm120::catalogs(), Some(profile))
        })
        .collect::<Vec<_>>();
    let catalog = sm120::moe::catalog();
    let routes = sm120::moe::route_catalog()
        .into_iter()
        .chain(sm120::moe::parallel_route_catalog())
        .collect::<Vec<_>>();
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out_dir,
        &out_dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit Qwen3.8 MoE cubins");
    assert_eq!(built.len(), specs.len());
    assert_eq!(
        built.iter().map(|c| c.entries.len()).sum::<usize>(),
        routes.len() + catalog.len()
    );
    for cubin in &built {
        let route = cubin
            .entries
            .iter()
            .all(|entry| routes.iter().any(|v| v.symbol == entry.symbol));
        for entry in &cubin.entries {
            assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
            let dynamic = if route {
                0
            } else {
                let variant = catalog
                    .iter()
                    .find(|v| entry.symbol == v.symbol)
                    .expect("entry belongs to catalog");
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
        let expected = if route {
            vec!["REDUX", "MATCH", "SHFL", "PREEXIT", "ACQBULK"]
        } else {
            // FP8 fragments convert in packed pairs before BF16 MMA; decode
            // streams them through per-warp cp.async rings and prefill
            // through the producer rings with ldmatrix activations.
            let mut expected = vec!["HMMA.16816", "F2FP.BF16.E4M3", "LDGSTS.E.BYPASS.128"];
            if catalog.iter().any(|v| {
                v.phase == moe::Phase::Prefill
                    && cubin.entries.iter().any(|entry| entry.symbol == v.symbol)
            }) {
                expected.push("LDSM.16.M88.4");
            }
            expected
        };
        for opcode in expected {
            assert!(sass.contains(opcode), "{} lacks {opcode}", cubin.name);
        }
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}
