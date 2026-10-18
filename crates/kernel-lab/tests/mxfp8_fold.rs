use kernel_core::backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{DType, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_lab::kit::{
    self, LabModules, OpResult, Outcome, Report, Required, cases::mxfp8_fold as cases,
};
use kernels_sm120::gemm::mxfp8_fold::{self, LAB_CUBIN};

fn check(report: &Report, gpu: bool) {
    assert_eq!(report.ops.len(), 14);
    assert_eq!(report.prepared().len(), 14);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    for (op, entry) in report.ops.iter().zip(mxfp8_fold::lab_entries()) {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required fold refused")
        };
        for name in if gpu {
            vec!["declaration", "need", "conformance", kit::PDL]
        } else {
            vec!["declaration", "need"]
        } {
            let check = checks.iter().find(|c| c.name == name).unwrap();
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{} {name}: {:?}",
                op.name,
                check.outcome
            );
        }
        if gpu {
            for property in cases::need(entry.query.m_capacity).exact.claims() {
                let name = kit::properties::name(property);
                let check = checks.iter().find(|c| c.name == name).unwrap();
                assert!(
                    matches!(check.outcome, Outcome::Passed(_)),
                    "{} {name}: {:?}",
                    op.name,
                    check.outcome
                );
            }
        }
    }
}

#[test]
fn mxfp8_fold_prepares_every_case() {
    assert_eq!(kernel_lab::cubin_entries(LAB_CUBIN).unwrap().len(), 18);
    let report = kit::conform(&cases::sm120(), &LabModules, None, &cases::required()).unwrap();
    check(&report, false);
}

#[test]
fn mxfp8_fold_has_native_mma_pdl_and_no_tf32() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let dir = kernel_lab::receipt_root().join(format!("mxfp8-fold-audit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mxfp8_fold.cubin");
    std::fs::write(&path, kernel_lab::cubin(LAB_CUBIN).unwrap()).unwrap();
    let result = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(result.status.success());
    let sass = String::from_utf8(result.stdout).unwrap();
    assert!(!sass.to_uppercase().contains("TF32"));
    assert!(!sass.contains("BPT.TRAP"), "device data must never trap");
    for section in sass.split("Function : ").skip(1) {
        let (symbol, body) = section.split_once('\n').unwrap();
        let steps = if symbol.ends_with("_fold256_control_v1") {
            8
        } else {
            4
        };
        assert!(
            body.matches("QMMA.SF.16832.F32.E4M3.E4M3.E8").count() >= steps,
            "{symbol}: missing {steps} native scale-block steps"
        );
        assert!(body.contains("PREEXIT"), "{symbol}: no PDL release");
        assert!(body.contains("ACQBULK"), "{symbol}: no PDL wait");
    }
    std::fs::write(dir.join("sass.txt"), sass).unwrap();
}

struct SingleChainControl {
    inner: Prepared<Mxfp8FoldGemmBuffers<View>>,
    info: OpInfo,
    symbol: String,
}
impl PreparedOp<Mxfp8FoldGemmBuffers<View>> for SingleChainControl {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(b, workspace, counts)?;
        for launch in &mut launches {
            launch.symbol = self.symbol.clone();
        }
        Ok(launches)
    }
}

fn require_output_failure(report: &Report, exact: bool) {
    assert_eq!(report.prepared().len(), 1, "{:?}", report.failures());
    let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
        unreachable!()
    };
    let result = &checks
        .iter()
        .find(|c| c.name == "conformance")
        .unwrap()
        .outcome;
    assert!(
        matches!(result,Outcome::Failed(reason) if if exact { reason.contains("output byte") && reason.contains("the reference") } else { reason.contains("numeric gate failed") }),
        "control failed for the wrong reason: {result:?}"
    );
}

fn mutant<'a>(entry: &mxfp8_fold::Entry, symbol: String, rows: u32) -> Required<'a> {
    let q = entry.query;
    Required::new(
        symbol.clone(),
        Need::rows(rows),
        move |backend, need| {
            let inner = backend.mxfp8_fold_gemm(&q, need)?;
            let mut info = inner.info().clone();
            info.symbols = vec![(symbol.clone(), 0)];
            info.implementation.push_str(".negative-control");
            Ok(Box::new(SingleChainControl {
                inner,
                info,
                symbol: symbol.clone(),
            }) as Prepared<Mxfp8FoldGemmBuffers<View>>)
        },
        move |op| cases::case(q, rows, op),
    )
    .required()
}

fn invalid_activations_set_status_and_zero_outputs(
    session: &kernel_lab::Session,
) -> kernel_lab::LabResult<()> {
    use kernel_harness::launch::{Bindings, launch};
    const OFFSET: usize = 3;
    for entry in mxfp8_fold::lab_entries()
        .into_iter()
        .filter(|e| e.query.n == 7 || e.query.groups > 1)
    {
        let q = entry.query;
        let rows = q.m_capacity.min(3);
        let bad_row = rows - 1;
        let module = session.module(LAB_CUBIN)?;
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let backend = kernels_sm120::backend::Sm120::new().with_mxfp8_fold(entry.clone());
        let op = backend
            .mxfp8_fold_gemm(&q, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let sizes = entry.bytes(rows);
        for (code, scale, invalid, previous) in [
            (56, 127, false, 0),
            (56, 127, false, 0x80),
            (127, 127, true, 0x80),
            (255, 127, true, 0),
            (56, 255, true, 0),
            (126, 254, true, 0),
            (1, 0, true, 0),
        ] {
            for live in [0, rows.saturating_sub(1), rows] {
                let mut initial = vec![0x4000_u32; OFFSET + rows as usize + 2];
                initial[OFFSET..OFFSET + rows as usize].fill(0);
                initial[OFFSET + bad_row as usize] = previous;
                let mut images = [
                    vec![56; sizes.activations as usize],
                    vec![127; sizes.activation_scales as usize],
                    vec![56; sizes.weights as usize],
                    vec![127; sizes.weight_scales as usize],
                    vec![0xa5; sizes.output as usize],
                    live.to_ne_bytes().to_vec(),
                    initial.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                ];
                // The last group's invalid activation must zero all groups
                // of this row, leaving other rows and status offsets intact.
                let at = (bad_row * q.activation_row_stride
                    + (q.groups - 1) * q.activation_group_stride
                    + q.k / 2) as usize;
                images[0][at] = code;
                images[1][at / 32] = scale;
                let mut buffers = Vec::new();
                for image in &images {
                    let buffer = session.allocate(image.len())?;
                    buffer
                        .copy_from_bytes(image, &session.stream)
                        .map_err(|e| e.to_string())?;
                    assert!(
                        buffer
                            .device_address_at(0, image.len())
                            .map_err(|e| e.to_string())?
                            > u64::from(u32::MAX)
                    );
                    buffers.push(buffer);
                }
                let view = |i: usize| View {
                    handle: i as u64,
                    bytes: images[i].len() as u64,
                };
                let views = Mxfp8FoldGemmBuffers {
                    activations: view(0),
                    activation_scales: view(1),
                    weights: view(2),
                    weight_scales: view(3),
                    output: view(4),
                    live_rows: view(5),
                    status: view(6),
                    status_row_offset: OFFSET as u64,
                };
                let bindings = Bindings {
                    buffers: buffers
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (i as u64, b))
                        .collect(),
                };
                for call in op
                    .launches(&views, View::EMPTY, Counts::rows(rows))
                    .map_err(|e| e.to_string())?
                {
                    launch(
                        &session.context,
                        &function,
                        &session.stream,
                        call.plan,
                        &call.arguments,
                        &bindings,
                    )?;
                }
                let mut actual = vec![0; images[4].len()];
                buffers[4]
                    .copy_to_bytes(&mut actual, &session.stream)
                    .map_err(|e| e.to_string())?;
                let mut expected = images[4].clone();
                let width = (q.groups * q.n) as usize;
                let element = if q.output == DType::F32 { 4 } else { 2 };
                for row in 0..live as usize {
                    let value = if row == bad_row as usize && (invalid || previous != 0) {
                        0.0
                    } else {
                        q.k as f32
                    };
                    let bits = if q.output == DType::F32 {
                        value.to_ne_bytes().to_vec()
                    } else {
                        kernel_core::codec::f32_to_bf16(value)
                            .to_ne_bytes()
                            .to_vec()
                    };
                    expected[row * width * element..(row + 1) * width * element]
                        .copy_from_slice(&bits.repeat(width));
                }
                assert_eq!(
                    actual, expected,
                    "code={code} scale={scale} live={live} query={q:?}"
                );
                if invalid && bad_row < live {
                    initial[OFFSET + bad_row as usize] |= kernel_core::mxfp8_gemm::NON_FINITE_INPUT;
                }
                let mut actual = vec![0; images[6].len()];
                buffers[6]
                    .copy_to_bytes(&mut actual, &session.stream)
                    .map_err(|e| e.to_string())?;
                assert_eq!(
                    actual,
                    initial
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn mxfp8_fold_gpu() {
    kernel_lab::gpu_test(1 << 30, |session| {
        invalid_activations_set_status_and_zero_outputs(session)?;
        let backend = cases::sm120();
        let report = kit::conform(&backend, &LabModules, Some(session), &cases::required())?;
        check(&report, true);
        let witness = mxfp8_fold::lab_entries()
            .into_iter()
            .find(|e| e.query.k == 65536 && e.query.output == DType::F32)
            .unwrap();
        let single = mutant(&witness, witness.single_chain_control_spec().symbol, 1);
        let single_report = kit::conform(&backend, &LabModules, Some(session), &[single])?;
        require_output_failure(&single_report, true);
        let wide = mutant(&witness, witness.wide_fold_control_spec().symbol, 1);
        let wide_report = kit::conform(&backend, &LabModules, Some(session), &[wide])?;
        require_output_failure(&wide_report, true);
        let scales = mxfp8_fold::lab_entries()
            .into_iter()
            .find(|e| e.query.n == 65 && e.query.k == 160 && e.query.output == DType::F32)
            .unwrap();
        let wrong_scales = mutant(&scales, scales.scale_row_control_spec().symbol, 3);
        let scale_report = kit::conform(&backend, &LabModules, Some(session), &[wrong_scales])?;
        require_output_failure(&scale_report, true);
        let good = mxfp8_fold::lab_entries()
            .into_iter()
            .find(|e| e.query.groups == 2 && e.query.output == DType::F32)
            .unwrap()
            .query;
        let wrong = mxfp8_fold::group_offset_control();
        let q = wrong.query;
        let wrong_backend = kernels_sm120::backend::Sm120::new().with_mxfp8_fold(wrong);
        let offsets = Required::new(
            "group-offset negative control",
            Need::rows(3),
            move |backend, need| backend.mxfp8_fold_gemm(&q, need),
            move |op| cases::case(good, 3, op),
        )
        .required();
        let offset_report = kit::conform(&wrong_backend, &LabModules, Some(session), &[offsets])?;
        require_output_failure(&offset_report, false);
        session.receipt(
            "mxfp8-fold",
            std::collections::BTreeMap::from([
                ("report".into(), report.json()),
                ("single_chain_control".into(), single_report.json()),
                ("fold256_control".into(), wide_report.json()),
                ("scale_row_control".into(), scale_report.json()),
                ("group_offset_control".into(), offset_report.json()),
                ("power_cap_watts".into(), serde_json::json!(250)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}
