//! CPU compilation and exact payload admission of quantized chained experts.

use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{
    ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, OutputQuantization, RoutedCapacity,
    SwigluColumns, WeightFormat,
};
use kernel_core::gated_mlp::{
    GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint, SwigluClamp,
};
use kernel_core::moe_chained::{ChainedK, ChainedMlpProjectionQuery};
use kernels_sm120::moe::chained::{self, quantized, row_bias};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn projection(gate_up: bool, k: u32, row_tile: u32) -> quantized::Projection {
    let query = ChainedMlpProjectionQuery {
        projection: GatedMlpProjectionQuery {
            gemm: GemmQuery {
                m_capacity: 32 * 3,
                n: if gate_up { 128 } else { 64 },
                k,
                weight: WeightFormat::Mxfp4,
                activation: ActivationFormat::Mxfp8,
                output: if gate_up { DType::Bf16 } else { DType::F32 },
                epilogues: Epilogues {
                    swiglu: gate_up.then_some(SwigluColumns::ConcatenatedHalves),
                    output_quantization: gate_up.then_some(OutputQuantization::Mxfp8),
                    ..Epilogues::default()
                },
                grouped: Some(GroupedGemm {
                    experts: 128,
                    top_k: 3,
                    capacity: RoutedCapacity::TotalRows(96),
                }),
            },
            projection: if gate_up {
                GatedMlpProjection::GateUp
            } else {
                GatedMlpProjection::Down
            },
            route_weight: Some(RouteWeightPoint::BeforeDown),
            swiglu_clamp: gate_up.then_some(SwigluClamp::symmetric(10.0)),
        },
        k: ChainedK::new(k).unwrap(),
    };
    quantized::Projection::new(query, row_tile, 32).unwrap()
}

fn output_dir(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("chopinfer-{label}-{}-{stamp}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    dir
}

#[test]
fn mxfp4_admission_matches_exact_bf16_over_every_code_and_scale() {
    for scale in 0..=255_u8 {
        for code in 0..16_u8 {
            let pair = code | code << 4;
            let value = SmallFloat::E2M1.decode_f64(code) * decode_ue8m0_f64(scale);
            let operand = f64::from(bf16_to_f32(f32_to_bf16(value as f32)));
            let valid = scale != 255 && operand.is_finite() && operand == value;
            assert_eq!(
                quantized::admit_mxfp4_payload(&[pair; 16], &[scale]).is_ok(),
                valid,
                "code {code}, scale {scale}"
            );
        }
    }
    for pair in [0x06, 0x60, 0x0e, 0xe0] {
        assert!(quantized::admit_mxfp4_payload(&[pair; 16], &[253]).is_err());
    }
    assert!(quantized::admit_mxfp4_payload(&[0x88; 16], &[254]).is_ok());
    assert!(quantized::admit_mxfp4_payload(&[0; 15], &[127]).is_err());
}

#[test]
fn quantized_arguments_and_admission_keep_their_domains() {
    for gate_up in [false, true] {
        let p = projection(gate_up, 576, 16);
        assert_eq!(p.query.k.loop_k, 640);
        assert_eq!(p.plan(32).unwrap().block, [128, 1, 1]);
        assert!(p.plan(0).is_err());
        assert!(p.plan(33).is_err());
        assert!(p.arguments([1; 10]).is_ok());
        for index in 0..10 {
            let mut args = [1; 10];
            args[index] = 0;
            assert_eq!(p.arguments(args).is_ok(), !gate_up && index == 8);
        }
    }
    for scale in 0..=255_u8 {
        for code in 0..=255_u8 {
            assert_eq!(
                quantized::admit_activations(&[code; 32], &[scale]),
                chained::formats::admit_mxfp8_payload(&[code; 32], &[scale])
            );
        }
    }
}

#[test]
fn quantized_chains_compile_without_spills_tf32_or_float_atomics() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the chained compile gate"
    );
    let out = output_dir("quantized-chains");
    let projections = [16, 32, 64]
        .into_iter()
        .flat_map(|rows| [projection(true, 128, rows), projection(false, 576, rows)])
        .collect::<Vec<_>>();
    let spec = quantized::cubin_spec("chop_moe_chain_quantized_compile", &projections);
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile the quantized chained engine");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), projections.len());
    for entry in &built[0].entries {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        assert!(entry.shared_static_bytes <= 48 * 1024);
    }
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let sass = String::from_utf8(sass.stdout).unwrap();
    assert!(sass.contains("HMMA.16816.F32.BF16"));
    for forbidden in ["TF32", "ATOM", "RED.E.", "REDG"] {
        assert!(!sass.contains(forbidden), "forbidden {forbidden}");
    }
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn existing_split_cubins_still_match_their_locked_bytes() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the preservation gate"
    );
    let out = output_dir("preserved-chains");
    let specs = chained::split::cubin_specs().unwrap();
    let lock = LockDir::new(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kernel-lab/kernels.lock.d"),
    )
    .family("moe_chained_split", specs.iter().map(|s| &s.name));
    let built = kernel_build::build_in_families(
        &toolchain,
        &specs,
        &out,
        &lock,
        BuildOptions {
            lock_update: false,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
        None,
    )
    .expect("existing chained cubins and resources must match their locks");
    assert_eq!(built.len(), specs.len());
    std::fs::remove_dir_all(out).unwrap();
}

#[test]
fn row_bias_routes_compile_for_both_expert_geometries_without_spills() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(
        !toolchain.skipped,
        "CUDA is required for the route compile gate"
    );
    let out = output_dir("row-bias-routes");
    let routers = [(384, 6), (128, 3)]
        .into_iter()
        .map(|(experts, top_k)| row_bias::Router {
            experts,
            top_k,
            capacity: 64,
        })
        .collect::<Vec<_>>();
    let routes = routers
        .iter()
        .flat_map(|r| {
            [32, 64].map(|capacity| row_bias::Route {
                local: chained::LocalRoute {
                    experts: r.experts,
                    local_experts: r.experts,
                    top_k: r.top_k,
                    capacity,
                    pad: if capacity == 32 { 16 } else { 64 },
                },
            })
        })
        .collect::<Vec<_>>();
    let spec = row_bias::cubin_spec("chop_moe_row_bias_compile", &routers, &routes).unwrap();
    let expected = spec.entries.len();
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile the row-bias route engine");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), expected);
    assert!(built[0].entries.iter().all(|e| e.local_bytes == 0));
    let sass = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(sass.status.success());
    let sass = String::from_utf8(sass.stdout).unwrap();
    for forbidden in ["TF32", "HMMA", "ATOM", "RED.E.", "REDG"] {
        assert!(!sass.contains(forbidden), "forbidden {forbidden}");
    }
    std::fs::remove_dir_all(out).unwrap();
}
