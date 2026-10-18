use kernel_lab::kit::cases::block_quantize;
use kernel_lab::kit::{self, LabModules, OpResult, Outcome, Report};
use kernels_sm120::row::block_quantize::{LAB_CUBIN, lab_entries};

fn check(report: &Report, gpu: bool) {
    assert_eq!(report.ops.len(), 48);
    assert_eq!(report.prepared().len(), 48);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required quantizer refused")
        };
        for name in [
            "declaration",
            "need",
            "conformance",
            kit::PDL,
            "property rows",
            "property chunk",
            "property batch",
        ] {
            if !gpu && !["declaration", "need"].contains(&name) {
                continue;
            }
            let check = checks
                .iter()
                .find(|c| c.name == name)
                .unwrap_or_else(|| panic!("{}: missing {name}", op.name));
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {name}: {:?}",
                op.name,
                check.outcome
            );
        }
    }
}

#[test]
fn block_quantize_prepares_every_case() {
    assert_eq!(
        kernel_lab::cubin_entries(LAB_CUBIN).unwrap().len(),
        lab_entries().len()
    );
    let report = kit::conform(
        &block_quantize::sm120(),
        &LabModules,
        None,
        &block_quantize::required(),
    )
    .unwrap();
    check(&report, false);
}

#[test]
fn block_quantize_has_pdl_and_no_tf32() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let dir =
        kernel_lab::receipt_root().join(format!("block-quantize-audit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("block_quantize.cubin");
    std::fs::write(&path, kernel_lab::cubin(LAB_CUBIN).unwrap()).unwrap();
    let result = std::process::Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&path)
        .output()
        .unwrap();
    assert!(result.status.success());
    let sass = String::from_utf8(result.stdout).unwrap();
    assert!(!sass.to_uppercase().contains("TF32"));
    assert!(
        !sass.to_uppercase().contains("TRAP"),
        "quantizer data must never trap"
    );
    assert!(sass.contains("PREEXIT"));
    assert!(sass.contains("ACQBULK"));
    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir(dir).unwrap();
}

#[test]
fn block_quantize_bitwise_gpu() {
    kernel_lab::gpu_test(256 << 20, |session| {
        let report = kit::conform(
            &block_quantize::sm120(),
            &LabModules,
            Some(session),
            &block_quantize::required(),
        )?;
        check(&report, true);
        session.receipt(
            "block-quantize",
            std::collections::BTreeMap::from([
                ("report".into(), report.json()),
                ("power_cap_watts".into(), serde_json::json!(250)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
}

fn status_cases() -> Vec<kit::Required<'static>> {
    let mut cases = block_quantize::invalid_required();
    cases.push(block_quantize::sequence::required());
    cases
}

#[test]
fn status_cases_prepare_every_format_and_shared_sequence() {
    let report =
        kit::conform(&block_quantize::sm120(), &LabModules, None, &status_cases()).unwrap();
    assert_eq!(report.prepared().len(), 49);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
}

#[test]
fn invalid_rows_and_shared_status_pass_poison_replay_on_a_gpu() {
    kernel_lab::gpu_test(256 << 20, |session| {
        let report = kit::conform(
            &block_quantize::sm120(),
            &LabModules,
            Some(session),
            &status_cases(),
        )?;
        assert_eq!(report.prepared().len(), 49);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        assert!(
            report.not_exercised().is_empty(),
            "{:?}",
            report.not_exercised()
        );
        session.receipt(
            "block-quantize-status",
            std::collections::BTreeMap::from([("report".into(), report.json())]),
        )?;
        Ok(())
    })
    .unwrap();
}

mod mutants {
    use super::*;
    use kernel_core::backend::{
        Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
    };
    use kernel_core::contract::Refusal;
    use kernel_core::precision::quantizer::{
        BlockQuantizeBuffers, BlockQuantizeQuery, QuantizeOutput,
    };
    use kernel_lab::families::block_quantize_mutants as family;
    struct BackendControl {
        fault: &'static str,
    }
    struct Op {
        inner: Prepared<BlockQuantizeBuffers<View>>,
        info: OpInfo,
        fault: &'static str,
    }
    impl Backend for BackendControl {
        fn name(&self) -> &str {
            "block-quantizer-status-control"
        }
        fn block_quantize(
            &self,
            q: &BlockQuantizeQuery,
            need: Need,
        ) -> Result<Prepared<BlockQuantizeBuffers<View>>, Refused> {
            let inner = block_quantize::sm120().block_quantize(q, need)?;
            let mut info = inner.info().clone();
            for (symbol, _) in &mut info.symbols {
                *symbol = family::symbol(symbol, self.fault);
            }
            Ok(Box::new(Op {
                inner,
                info,
                fault: self.fault,
            }))
        }
    }
    impl PreparedOp<BlockQuantizeBuffers<View>> for Op {
        fn info(&self) -> &OpInfo {
            &self.info
        }
        fn launches(
            &self,
            b: &BlockQuantizeBuffers<View>,
            workspace: View,
            counts: Counts,
        ) -> Result<Vec<Launch>, Refusal> {
            let mut launches = self.inner.launches(b, workspace, counts)?;
            for launch in &mut launches {
                launch.symbol = family::symbol(&launch.symbol, self.fault);
            }
            Ok(launches)
        }
    }
    fn required(fault: &str) -> Vec<kit::Required<'static>> {
        let entry = family::entries()
            .into_iter()
            .find(|e| (e.query.output == QuantizeOutput::Bf16) == (fault == "drop_output"))
            .unwrap();
        let query = entry.query;
        let mut cases = vec![
            kit::Required::new(
                format!("block status {query:?}"),
                Need::rows(16),
                move |backend, need| backend.block_quantize(&query, need),
                move |op| block_quantize::case_with_status(query, 16, op, true),
            )
            .required(),
        ];
        if fault == "missing_or" {
            cases.push(block_quantize::sequence::required());
        }
        cases
    }
    #[test]
    fn faulty_device_status_is_rejected_by_unchanged_cases() {
        kernel_lab::gpu_test(256<<20,|session| {
            for fault in family::FAULTS {
                let cases=required(fault);
                let positive=kit::conform(&block_quantize::sm120(),&LabModules,Some(session),&cases)?;
                assert_eq!(positive.prepared().len(),cases.len());
                assert!(positive.failures().is_empty(),"{:?}",positive.failures());
                assert!(positive.not_exercised().is_empty());
                let report=kit::conform(&BackendControl {fault},&LabModules,Some(session),&cases)?;
                assert_eq!(report.prepared().len(),cases.len());
                for op in &report.ops {
                    let OpResult::Prepared {checks,..}=&op.result else {panic!("mutant did not prepare")};
                    assert!(checks.iter().any(|c|c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(e) if (e.contains("status")||e.contains("output")||e.contains("scales")) && !e.contains("CUDA") && !e.contains("unbound"))),"{fault}: {checks:?}");
                }
                session.receipt(&format!("block-status-{}",fault.replace('_',"-")),std::collections::BTreeMap::from([("report".into(),report.json())]))?;
            }
            Ok(())
        }).unwrap();
    }
}
