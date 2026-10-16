//! Compile and audit MiMo V2.6 Flash's grouped expert, route, fused router
//! and fused down-combine cubins without loading one.

use kernel_build::{BuildOptions, Toolchain};
use kernel_core::contract::RuntimeCounts;
use kernels_sm120::moe;
use model_mimo26_contract::sm120;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::BTreeMap, fs};

#[test]
fn mimo_expert_and_route_cubins_build_and_emit_their_instructions() {
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
        "chopinfer-mimo-moe-build-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out_dir).expect("create test output directory");
    let lock = out_dir.join("kernels.lock");
    let specs = model_mimo26_contract::sm120::PROFILES
        .into_iter()
        .flat_map(|profile| moe::cubin_specs(sm120::catalogs(), Some(profile)))
        .collect::<Vec<_>>();
    let catalog = sm120::moe::catalog();
    let mut route_entries = BTreeMap::new();
    let mut route_cubins = BTreeMap::new();
    for owner in sm120::geometry::MODELS {
        let routes = sm120::moe::route_cubins_for(*owner);
        assert_eq!(
            routes
                .iter()
                .map(|spec| spec.entries.len())
                .collect::<Vec<_>>(),
            [1, 6, 5]
        );
        for (kind, spec) in routes.into_iter().enumerate() {
            route_cubins.insert(spec.name, kind);
            for entry in spec.entries {
                let shared = if kind == 1 {
                    let tokens: u32 = entry
                        .symbol
                        .rsplit_once("_t")
                        .unwrap()
                        .1
                        .split('_')
                        .next()
                        .unwrap()
                        .parse()
                        .unwrap();
                    let symbol = format!(
                        "chop_gemm_{}_router_m{tokens}_v1",
                        owner.geometry.kernel_tag
                    );
                    let variant = sm120::gemm::catalog()
                        .into_iter()
                        .find(|v| v.symbol == symbol)
                        .unwrap();
                    kernels_sm120::gemm::plan(
                        &variant.key,
                        RuntimeCounts {
                            m: tokens,
                            batch: tokens,
                            context: 0,
                            routed_rows: 0,
                            split_k: kernels_sm120::gemm::choose_split_k(&variant.key).unwrap(),
                        },
                    )
                    .unwrap()
                    .dynamic_shared_bytes
                } else {
                    0
                };
                route_entries.insert(entry.symbol, (kind, shared));
            }
        }
    }
    let expected = route_entries.len()
        + catalog
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
                if route_entries
                    .get(&entry.symbol)
                    .is_some_and(|(kind, _)| *kind != 1)
                {
                    report.push_str(&format!(
                        "{}\t{}\t{}\t0\t{}\n",
                        entry.symbol, entry.registers, entry.shared_static_bytes, entry.local_bytes
                    ));
                    continue;
                }
                if let Some((1, dynamic)) = route_entries.get(&entry.symbol) {
                    let dynamic = *dynamic;
                    report.push_str(&format!(
                        "{}\t{}\t{}\t{}\t{}\n",
                        entry.symbol,
                        entry.registers,
                        entry.shared_static_bytes,
                        dynamic,
                        entry.local_bytes
                    ));
                    continue;
                }
                let variant = catalog
                    .iter()
                    .find(|v| {
                        entry.symbol == v.symbol
                            || v.reduction_symbol().as_deref() == Some(entry.symbol.as_str())
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
            if route_entries
                .get(&entry.symbol)
                .is_some_and(|(kind, _)| *kind != 1)
            {
                assert!(entry.shared_static_bytes <= 101_376);
                continue;
            }
            if let Some((1, dynamic)) = route_entries.get(&entry.symbol) {
                let dynamic = *dynamic;
                assert!(entry.shared_static_bytes + dynamic <= 101_376);
                continue;
            }
            let variant = catalog
                .iter()
                .find(|v| {
                    entry.symbol == v.symbol
                        || v.reduction_symbol().as_deref() == Some(entry.symbol.as_str())
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
        if route_cubins.get(&cubin.name) == Some(&0) {
            assert!(sass.contains("REDUX"), "route entry lacks warp reduction");
            assert!(
                sass.contains("SHFL"),
                "route entry lacks warp top-8 selection"
            );
            continue;
        }
        if route_cubins.get(&cubin.name) == Some(&2) {
            assert!(sass.contains("HMMA.16816"), "fused down lacks BF16 MMA");
            assert!(
                sass.contains("LDG.E.NA.128"),
                "fused down lacks fragment loads"
            );
            continue;
        }
        if route_cubins.get(&cubin.name) == Some(&1) {
            assert!(sass.contains("HMMA.16816"), "fused router lacks BF16 MMA");
            assert!(sass.contains("REDUX"), "fused router lacks top-8 reduction");
            continue;
        }
        let variants = catalog
            .iter()
            .filter(|v| cubin.entries.iter().any(|entry| entry.symbol == v.symbol))
            .collect::<Vec<_>>();
        use kernel_core::contract::WeightFormat;
        use kernels_sm120::moe::InputPath;
        let first = variants.first().expect("MoE cubin has catalog variants");
        let native_fp4 =
            first.path == InputPath::A8Pregather && first.query.weight == WeightFormat::Nvfp4;
        let mma = match (first.path, first.query.weight) {
            (InputPath::A8Pregather, WeightFormat::Nvfp4) => "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X",
            (InputPath::A8Pregather, WeightFormat::Fp8Block128x128(_)) => "QMMA.16832",
            (InputPath::A8Pregather, WeightFormat::Mxfp4 | WeightFormat::Mxfp8) => "QMMA.SF.16832",
            _ => "HMMA.16816",
        };
        assert!(
            variants
                .iter()
                .all(|v| v.path == first.path && v.query.weight == first.query.weight)
        );
        let expected = if variants.iter().any(|v| v.fragment_weights()) {
            // The catalog's schedules use per-warp cp.async rings and
            // persistent producer rings according to the declared variant.
            let ring = variants.iter().any(|v| v.fragment_decode());
            let producer = variants.iter().any(|v| v.fragment_prefill());
            match (ring, producer) {
                (true, true) => vec![mma, "LDG.E.128", "LDGSTS.E.BYPASS.128", "LDSM.16.M88.4"],
                (false, true) => vec![mma, "LDGSTS.E.BYPASS.128", "LDSM.16.M88.4", "LDS.128"],
                (true, false) => vec![mma, "LDGSTS.E.BYPASS.128", "LDS.128"],
                (false, false) => vec![mma, "UBLKCP.S.G", "LDS.128"],
            }
        } else {
            vec![mma, "UTMALDG.2D", "UTMAPF"]
        };
        for opcode in expected {
            assert!(sass.contains(opcode), "{} lacks {opcode}", cubin.name);
        }
        if native_fp4 {
            assert!(
                sass.contains("LDSM"),
                "{} lacks native FP4 ldmatrix",
                cubin.name
            );
        }
    }
    fs::remove_dir_all(out_dir).expect("remove hermetic test output");
}
