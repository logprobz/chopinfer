//! Exhaust the MX A16 operand domain. Admission checks every code and scale;
//! every admitted pair reaches each production converter, with both signs
//! in the two lanes. Invalid operands are refused, never rounded into a gate.
use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, Toolchain};
use kernel_core::{
    codec::{SmallFloat, bf16_to_f32, f64_to_bf16},
    contract::{DType, WeightFormat},
    precision::{
        DecodeOrder, NibbleOrder,
        packed::{MatrixView, ScaleBytes},
    },
};
use kernel_lab::{
    LabResult, Session,
    poison::{self, Replay, WriteMap},
};
use runtime_cuda::{KernelArgument, LaunchConfig};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, sync::OnceLock};

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn probe() -> LabResult<PathBuf> {
    static BUILT: OnceLock<LabResult<PathBuf>> = OnceLock::new();
    BUILT
        .get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let out = kernel_lab::receipt_root()
                .join(format!("mx-operands-build-{}", std::process::id()));
            std::fs::create_dir_all(&out).map_err(err)?;
            let entries = [true, false]
                .into_iter()
                .map(|fp4| {
                    EntrySpec::new(
                        format!("mx_operand_{}_v1", if fp4 { 4 } else { 8 }),
                        vec![
                            EntryParam::DevicePointer {
                                name: "input".into(),
                                cxx_type: "const uint32_t".into(),
                            },
                            EntryParam::DevicePointer {
                                name: "output".into(),
                                cxx_type: "uint32_t".into(),
                            },
                            EntryParam::U32 {
                                name: "count".into(),
                            },
                            EntryParam::U32 {
                                name: "path".into(),
                            },
                        ],
                        format!("chop::probe::mx_operands<{fp4}>(input, output, count, path);"),
                        32,
                        1,
                    )
                })
                .collect();
            let spec = CubinSpec {
                name: "mx_operands".into(),
                sources_root: root.join(".."),
                include_dirs: vec!["kernels-sm120/cuda".into()],
                headers: vec!["#include \"kernel-lab/cuda/probes/mx_operands.cuh\"".into()],
                defines: vec![],
                entries,
            };
            let built = kernel_build::build(
                &Toolchain::detect().map_err(err)?,
                &[spec],
                &out,
                &out.join("scratch.lock"),
                BuildOptions {
                    lock_update: true,
                    ..Default::default()
                },
            )
            .map_err(err)?;
            Ok(built[0].path.clone())
        })
        .clone()
}

fn cases(fp4: bool) -> (Vec<u32>, Vec<u8>, usize) {
    let format = if fp4 {
        SmallFloat::E2M1
    } else {
        SmallFloat::E4M3
    };
    let mut input = Vec::new();
    let mut expected = Vec::new();
    let mut rejected = 0;
    for code in 0..=if fp4 { 15_u8 } else { 255 } {
        for scale in 0..=255_u8 {
            let factor = if scale == 255 {
                f64::NAN
            } else {
                2_f64.powi(i32::from(scale) - 127)
            };
            let exact = format.decode_f64(code) * factor;
            let bits = f64_to_bf16(exact);
            let admitted =
                exact.is_finite() && f64::from(bf16_to_f32(bits)).to_bits() == exact.to_bits();
            let codes = vec![if fp4 { code | code << 4 } else { code }; if fp4 { 16 } else { 32 }];
            let scales = [scale];
            let view = MatrixView {
                tensor: "probe.weight",
                format: if fp4 {
                    WeightFormat::Mxfp4
                } else {
                    WeightFormat::Mxfp8
                },
                rows: 1,
                k: 32,
                row_origin: 0,
                k_origin: 0,
                nibble_order: NibbleOrder::EvenLow,
                codes: &codes,
                local: Some(ScaleBytes {
                    dtype: DType::Ue8M0,
                    bytes: &scales,
                }),
                global: None,
            };
            assert_eq!(
                view.validate_bf16_operands(DecodeOrder::ExactScaledOperand)
                    .is_ok(),
                admitted,
                "FP{} code {code}, scale {scale}",
                if fp4 { 4 } else { 8 }
            );
            if !admitted {
                rejected += 1;
                continue;
            }
            let opposite = code ^ if fp4 { 8 } else { 128 };
            for position in 0..if fp4 { 16 } else { 1 } {
                input.push(
                    u32::from(code)
                        | (u32::from(opposite) << if fp4 { 4 } else { 8 })
                        | (u32::from(scale) << 16)
                        | (position << 24),
                );
                let pair = u32::from(bits) | (u32::from(bits ^ 0x8000) << 16);
                for _ in 0..4 {
                    expected.extend_from_slice(&pair.to_le_bytes());
                }
            }
        }
    }
    assert_eq!(
        input.len() / if fp4 { 16 } else { 1 } + rejected,
        if fp4 { 4096 } else { 65536 }
    );
    (input, expected, rejected)
}

fn run(session: &Session, fp4: bool, input: &[u32], path: u32) -> LabResult<Vec<u8>> {
    let module = session
        .context
        .load_module(&std::fs::read(probe()?).map_err(err)?)
        .map_err(err)?;
    let function = module
        .function(&format!("mx_operand_{}_v1", if fp4 { 4 } else { 8 }))
        .map_err(err)?;
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let source = session.allocate(bytes.len())?;
    poison::write(session, &source, 0, &bytes)?;
    let output = session.allocate(input.len() * 16 + 16)?;
    let config = LaunchConfig {
        grid: ((input.len() as u32).div_ceil(32), 1, 1),
        block: (32, 1, 1),
        shared_memory_bytes: 0,
        cluster: LaunchConfig::NO_CLUSTER,
        programmatic_stream_serialization: false,
    };
    let launch = || {
        function
            .launch(
                &session.stream,
                config,
                &[
                    KernelArgument::Buffer(&source),
                    KernelArgument::Buffer(&output),
                    KernelArgument::U32(input.len() as u32),
                    KernelArgument::U32(path),
                ],
            )
            .map_err(err)
    };
    // The probe writes all four pairs per input. The final 16 bytes are a
    // guard tail, from mx_operands.cuh's index and output strides.
    let mut replay = Replay::new(session).output(
        "operands",
        &output,
        WriteMap::of(DType::Bf16).byte_prefix(input.len() * 16)?,
    );
    replay.eager(launch)?;
    replay.relaunch(launch)?;
    session.stream.begin_capture().map_err(err)?;
    launch()?;
    let graph = session
        .stream
        .end_capture()
        .map_err(err)?
        .instantiate()
        .map_err(err)?;
    replay.graph(&graph)?;
    Ok(replay.expected("operands").unwrap()[..input.len() * 16].to_vec())
}

fn disjoint_cases() -> (Vec<u32>, Vec<u8>) {
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for code in [4_u8, 5, 6, 7, 12, 13, 14, 15] {
        let bits = f64_to_bf16(SmallFloat::E2M1.decode_f64(code));
        let pair = u32::from(bits) | (u32::from(bits ^ 0x8000) << 16);
        for position in 0..16 {
            let slot = 2 * (position / 8) + position % 2;
            for other in (0..4).filter(|other| *other != slot) {
                for scale in 253..=254 {
                    inputs.push(
                        u32::from(code)
                            | (u32::from(code ^ 8) << 4)
                            | (127 << 16)
                            | (position << 24)
                            | (other << 28)
                            | ((scale - 253) << 30)
                            | (1 << 31),
                    );
                    for _ in 0..4 {
                        expected.extend_from_slice(&pair.to_le_bytes());
                    }
                }
            }
        }
    }
    assert_eq!(inputs.len(), 768);
    (inputs, expected)
}

#[test]
fn mx_operand_probe_compiles() {
    probe().expect("compile production conversion paths");
}

#[test]
fn mx_operand_admission_exhausts_codes_and_scales() {
    for fp4 in [true, false] {
        cases(fp4);
    }
    disjoint_cases();
}

#[test]
fn mx_disjoint_extreme_scales_do_not_trap() {
    kernel_lab::gpu_test(16 << 20, |session| {
        let (input, expected) = disjoint_cases();
        let actual = run(session, true, &input, 0)?;
        if actual != expected {
            return Err("a scale in another slot changed a finite MXFP4 operand".into());
        }
        Ok(())
    })
    .expect("independent fragment scale slots");
}

#[test]
fn mx_operands_match_f64_bitwise() {
    kernel_lab::gpu_test(16 << 20, |session| {
        let mut counts = BTreeMap::new();
        for fp4 in [true, false] {
            let (input, expected, rejected) = cases(fp4);
            let actual = run(session, fp4, &input, 0)?;
            if let Some(at) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                return Err(format!(
                    "FP{} input {:#x}, output byte {at}: got {}, expected {}",
                    if fp4 { 4 } else { 8 },
                    input[at / 16],
                    actual[at],
                    expected[at]
                ));
            }
            counts.insert(
                format!("fp{}", if fp4 { 4 } else { 8 }),
                json!({"admitted": input.len() / if fp4 { 16 } else { 1 }, "refused": rejected,
                    "expansion_cases": input.len()}),
            );
        }
        let mut receipt = session.metadata.clone();
        receipt.insert("exhaustive_operand_cases".into(), json!(counts));
        receipt.insert("f64_bitwise".into(), json!(true));
        receipt.insert("poisoned_relaunch_and_graph".into(), json!(true));
        kernel_lab::write_receipt("mx_operand_exactness", session.ordinal, &receipt)?;
        Ok(())
    })
    .expect("exact MX A16 operands");
}
