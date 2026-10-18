//! Bitwise retained-window staging through poisoned graph and PDL replay.
use kernel_core::backend::{Backend, Need};
use kernel_harness::conformance::Probes;
use kernel_harness::kit::cases::latent_window_stage::{self as cases, Scenario};
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::latent_window_stage as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::window_stage::Mutation;
use runtime_cuda::Module;
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf};
fn backend() -> Sm120 {
    Sm120::new().with_latent_window_stage(family::entry())
}
struct FamilyModules {
    directory: PathBuf,
}
impl Modules for FamilyModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for spec in family::cubins() {
            for entry in spec.entries {
                lookup.insert(entry.symbol, spec.name.clone());
            }
        }
        let modules = RefCell::new(BTreeMap::<String, Module>::new());
        Ok(Box::new(move |symbol| {
            let name = lookup
                .get(symbol)
                .ok_or_else(|| format!("missing latent glue symbol {symbol}"))?;
            let mut modules = modules.borrow_mut();
            if !modules.contains_key(name) {
                let bytes = std::fs::read(self.directory.join(format!("{name}.cubin")))
                    .map_err(|e| e.to_string())?;
                modules.insert(name.clone(), session.load_cubin(&bytes)?);
            }
            modules[name].function(symbol).map_err(|e| e.to_string())
        }))
    }
    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let bytes = std::fs::read(
            self.directory
                .join(format!("{}.cubin", kernels_sm120::probes::CUBIN)),
        )
        .map_err(|e| e.to_string())?;
        let module = session.load_cubin(&bytes)?;
        Ok(Some(Probes {
            live: LiveProducer::with_function(
                session,
                module
                    .function(kernels_sm120::probes::LIVE)
                    .map_err(|e| e.to_string())?,
            )?,
            copy: Some(CopyProducer::with_function(
                module
                    .function(kernels_sm120::probes::COPY)
                    .map_err(|e| e.to_string())?,
            )),
        }))
    }
}

fn required() -> Vec<kit::Required<'static>> {
    Scenario::ALL.into_iter().map(cases::required).collect()
}
fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 7, "{:?}", report.ops);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("stage refused")
            };
            for c in checks {
                assert!(
                    matches!(c.outcome, Outcome::Passed(_)),
                    "{} {}: {:?}",
                    op.name,
                    c.name,
                    c.outcome
                );
            }
        }
    }
}
#[test]
fn stage_cases_prepare_and_preserve_tail_padding() {
    let modules = FamilyModules {
        directory: PathBuf::new(),
    };
    check(
        &kit::conform(&backend(), &modules, None, &required()).unwrap(),
        false,
    );
    for mutation in Mutation::ALL {
        let report = kit::conform(&backend(), &modules, None, &[cases::mutant(mutation)]).unwrap();
        assert_eq!(report.prepared().len(), 1);
        assert!(report.failures().is_empty());
    }
    for scenario in Scenario::ALL {
        let case = cases::case(
            scenario,
            backend()
                .latent_window_stage(&cases::QUERY, Need::rows(scenario.capacity()))
                .unwrap(),
        )
        .unwrap();
        let images = case.images();
        for live in kernel_harness::poison::live_counts(scenario.capacity())
            .into_iter()
            .filter(|&n| n > 0)
        {
            for plane in ["staged.codes", "staged.scales"] {
                assert!(!case.state_ranges(plane, &images, live).unwrap().is_empty());
            }
        }
    }
}
#[test]
fn window_stage_passes_poison_graph_pdl_and_device_controls_on_gpu() {
    let result = kernel_harness::gpu_test(256 << 20, |session| {
        let directory = PathBuf::from(
            std::env::var_os("CHOPINFER_WINDOW_STAGE_CUBINS")
                .ok_or("prebuilt stage cubins required")?,
        );
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = FamilyModules { directory };
        let report = kit::conform(&backend(), &modules, Some(session), &required())?;
        session.receipt(
            "window-stage",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, true);
        for mutation in Mutation::ALL {
            let report = kit::conform(
                &backend(),
                &modules,
                Some(session),
                &[cases::mutant(mutation)],
            )?;
            session.receipt(
                &format!("window-stage-control-{mutation:?}"),
                BTreeMap::from([("report".into(), report.json())]),
            )?;
            let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
                panic!("mutant refused")
            };
            assert!(
                checks.iter().any(|c| c.name == "conformance"
                    && matches!(&c.outcome,Outcome::Failed(error) if error.contains("staged."))),
                "mutant escaped: {checks:?}"
            );
            for c in checks {
                if let Outcome::Failed(error) = &c.outcome {
                    assert!(
                        !error.contains("CUDA") && !error.contains("trap"),
                        "device fault: {error}"
                    );
                }
            }
        }
        // Two prefill steps: append the first chunk's already encoded rows
        // to the observed first stage, then compare the next gathered prefix.
        let functions = modules.load(session)?;
        let mut stages = Vec::new();
        for scenario in [Scenario::SplitFirst, Scenario::SplitSecond] {
            let case = cases::case(
                scenario,
                backend()
                    .latent_window_stage(&cases::QUERY, Need::rows(2))
                    .unwrap(),
            )?;
            let images = case.images();
            let output = case.launch(session, &*functions, &images, 2)?;
            case.judge_outputs(&images, 2, &output)?;
            stages.push(output);
        }
        let mut first = stages.remove(0);
        let next = stages.remove(0);
        for (plane, width) in [("staged.codes", 512), ("staged.scales", 16)] {
            let bytes = first.get_mut(plane).unwrap();
            for p in 140..172 {
                for col in 0..width {
                    bytes[(p - 13) as usize * width + col] = cases::payload(p, col, width == 16);
                }
            }
            assert_eq!(&next[plane][..127 * width], &bytes[32 * width..159 * width]);
        }
        session.receipt(
            "window-stage-split-prefill",
            BTreeMap::from([
                ("prefix_rows".into(), serde_json::json!(127)),
                ("chunk_rows".into(), serde_json::json!(32)),
            ]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required staging test skipped");
    }
}
