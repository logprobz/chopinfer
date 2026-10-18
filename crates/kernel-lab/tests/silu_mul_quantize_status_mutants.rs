//! Require real device defects to fail the unchanged status oracle.
mod status_mutants {
    use kernel_core::backend::{
        Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
    };
    use kernel_core::contract::Refusal;
    use kernel_core::gated_mlp::quantized::{SiluMulQuantizeBuffers, SiluMulQuantizeQuery};
    use kernel_lab::families::silu_mul_quantize_mutants as family;
    use kernel_lab::kit::cases::silu_mul_quantize;
    use kernel_lab::kit::{self, LabModules, OpResult, Outcome};
    struct BackendControl {
        fault: &'static str,
    }
    struct Op {
        inner: Prepared<SiluMulQuantizeBuffers<View>>,
        info: OpInfo,
        fault: &'static str,
    }
    impl Backend for BackendControl {
        fn name(&self) -> &str {
            "silu-status-control"
        }
        fn silu_mul_quantize(
            &self,
            q: &SiluMulQuantizeQuery,
            need: Need,
        ) -> Result<Prepared<SiluMulQuantizeBuffers<View>>, Refused> {
            let inner = silu_mul_quantize::sm120().silu_mul_quantize(q, need)?;
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
    impl PreparedOp<SiluMulQuantizeBuffers<View>> for Op {
        fn info(&self) -> &OpInfo {
            &self.info
        }
        fn launches(
            &self,
            b: &SiluMulQuantizeBuffers<View>,
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
    fn required() -> Vec<kit::Required<'static>> {
        let entry = family::entry();
        let query = entry.query;
        vec![
            kit::Required::new(
                format!("SwiGLU status {query:?}"),
                Need::rows(16),
                move |backend, need| backend.silu_mul_quantize(&query, need),
                move |op| silu_mul_quantize::case_with_status(query, 16, op, true),
            )
            .required(),
            silu_mul_quantize::sequence::required(),
        ]
    }

    #[test]
    fn faulty_device_status_is_rejected_by_unchanged_cases() {
        kernel_lab::gpu_test(256<<20,|session| {
            for fault in family::FAULTS {
                let cases=required();
                let positive=kit::conform(&silu_mul_quantize::sm120(),&LabModules,Some(session),&cases)?;
                assert_eq!(positive.prepared().len(),cases.len());
                assert!(positive.failures().is_empty(),"{:?}",positive.failures());
                assert!(positive.not_exercised().is_empty());
                let report=kit::conform(&BackendControl {fault},&LabModules,Some(session),&cases)?;
                assert_eq!(report.prepared().len(),cases.len());
                for op in &report.ops {
                    let OpResult::Prepared {checks,..}=&op.result else {panic!("mutant did not prepare")};
                    assert!(checks.iter().any(|c|c.name=="conformance" && matches!(&c.outcome,Outcome::Failed(e) if (e.contains("status")||e.contains("codes") || e.contains("output")||e.contains("scales")) && !e.contains("CUDA") && !e.contains("unbound"))),"{fault}: {checks:?}");
                }
                session.receipt(&format!("silu-status-{}",fault.replace('_',"-")),std::collections::BTreeMap::from([("report".into(),report.json())]))?;
            }
            Ok(())
        }).unwrap();
    }
}
