//! Poison, graph, PDL and numerical controls for native latent glue.
use kernel_core::latent_glue::{
    LatentQueryRope, LatentWindowKeys, LatentWindowWrite, WindowAddressing,
};
use kernel_harness::conformance::Probes;
use kernel_harness::kit::cases::latent_glue::{self, KeyMutation};
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome, Required};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::latent_glue as family;
use kernels_sm120::backend::Sm120;
use kernels_sm120::latent::glue::Mutation;
use runtime_cuda::Module;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn backend() -> Sm120 {
    let b = family::entries()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_latent_glue);
    family::quantizers()
        .into_iter()
        .fold(b, Sm120::with_block_quantize)
}
fn ring_keys() -> LatentWindowKeys {
    LatentWindowKeys {
        window: 128,
        addressing: WindowAddressing::Circular { slots: 136 },
    }
}
fn required() -> Vec<Required<'static>> {
    let mut cases = [1, 3, 64]
        .into_iter()
        .map(|heads| {
            latent_glue::query_rope(
                LatentQueryRope {
                    heads,
                    latent: 512,
                    rotary: 64,
                },
                17,
            )
        })
        .collect::<Vec<_>>();
    for addressing in [
        WindowAddressing::Circular { slots: 136 },
        WindowAddressing::Staged { slots: 160 },
        WindowAddressing::Staged { slots: 32 },
    ] {
        cases.push(latent_glue::window_keys(
            LatentWindowKeys {
                window: 128,
                addressing,
            },
            17,
        ));
    }
    for (addressing, capacity, first) in [
        (WindowAddressing::Circular { slots: 136 }, 17, 130),
        (WindowAddressing::Circular { slots: 136 }, 320, 0),
        (WindowAddressing::Staged { slots: 160 }, 32, 128),
    ] {
        cases.push(latent_glue::window_write(
            LatentWindowWrite {
                epsilon_bits: 1e-20_f32.to_bits(),
                latent: 512,
                rotary: 64,
                addressing,
            },
            capacity,
            first,
        ));
    }
    let write = LatentWindowWrite {
        latent: 512,
        rotary: 64,
        epsilon_bits: 1e-20_f32.to_bits(),
        addressing: WindowAddressing::Circular { slots: 136 },
    };
    cases.push(latent_glue::window_f32_regression(write));
    cases.push(latent_glue::window_invalid_data(write));
    cases.push(latent_glue::window_invalid_data(LatentWindowWrite {
        addressing: WindowAddressing::Staged { slots: 160 },
        ..write
    }));
    cases.push(latent_glue::key_invalid_data(ring_keys()));
    cases.push(latent_glue::key_invalid_data(LatentWindowKeys {
        addressing: WindowAddressing::Staged { slots: 160 },
        ..ring_keys()
    }));
    cases
}
struct FamilyModules {
    directory: PathBuf,
}
impl Modules for FamilyModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for spec in family::cubins()
            .into_iter()
            .chain(kernel_lab_families::latent_glue_mutants::cubins())
        {
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
fn check(report: &kit::Report, expected: usize, gpu: bool) {
    assert_eq!(report.prepared().len(), expected, "{:?}", report.ops);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("required glue op refused")
            };
            for check in checks {
                assert!(
                    matches!(check.outcome, Outcome::Passed(_)),
                    "{}: {:?}",
                    check.name,
                    check.outcome
                );
            }
        }
    }
}
#[test]
fn all_latent_glue_cases_prepare_and_declare() {
    let modules = FamilyModules {
        directory: PathBuf::new(),
    };
    check(
        &kit::conform(&backend(), &modules, None, &required()).unwrap(),
        14,
        false,
    );
    for mutation in Mutation::ALL {
        check(
            &kit::conform(
                &backend(),
                &modules,
                None,
                &[latent_glue::device_mutant(mutation)],
            )
            .unwrap(),
            1,
            false,
        );
    }
    for mutation in [KeyMutation::Outside, KeyMutation::Stale] {
        let case = latent_glue::window_key_control(ring_keys(), 17, mutation);
        check(
            &kit::conform(&backend(), &modules, None, &[case]).unwrap(),
            1,
            false,
        );
    }
}
#[test]
fn latent_glue_passes_poison_graph_pdl_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test(256 << 20, |session| {
        let directory = std::env::var_os("CHOPINFER_LATENT_GLUE_CUBINS")
            .map(PathBuf::from)
            .ok_or("prebuilt latent glue cubins required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = FamilyModules { directory };
        let report = kit::conform(&backend(), &modules, Some(session), &required())?;
        session.receipt(
            "latent-glue",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, 14, true);
        for (mutation, reason) in [
            (KeyMutation::Outside, "outside slot domain"),
            (KeyMutation::Stale, "visible positions or order differ"),
        ] {
            let case = latent_glue::window_key_control(ring_keys(), 17, mutation);
            let report = kit::conform(&backend(), &modules, Some(session), &[case])?;
            session.receipt(
                &format!("latent-glue-key-control-{mutation:?}"),
                BTreeMap::from([("report".into(), report.json())]),
            )?;
            assert_eq!(report.prepared().len(), 1);
            let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
                panic!("key control not prepared")
            };
            assert!(
                checks.iter().any(|check| check.name == "conformance"
                    && matches!(&check.outcome, Outcome::Failed(error) if error.contains(reason))),
                "key mutant escaped: {checks:?}"
            );
            for check in checks {
                if let Outcome::Failed(error) = &check.outcome {
                    assert!(
                        error.contains("window-key oracle:"),
                        "unexpected negative-control failure: {error}"
                    );
                }
            }
        }
        for mutation in Mutation::ALL {
            let report = kit::conform(
                &backend(),
                &modules,
                Some(session),
                &[latent_glue::device_mutant(mutation)],
            )?;
            session.receipt(
                &format!("latent-glue-device-control-{mutation:?}"),
                BTreeMap::from([("report".into(), report.json())]),
            )?;
            assert_eq!(report.prepared().len(), 1);
            let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
                panic!("mutant refused")
            };
            let numerical_failure = |error: &str| match mutation {
                Mutation::WindowEdge => {
                    error.contains("window-key oracle: visible positions or order differ")
                }
                Mutation::WrongTail => {
                    error.contains("query")
                        && (error.contains("outside the write map")
                            || error.contains("the reference"))
                }
                Mutation::ReversedPair | Mutation::DoubleRound => {
                    (error.contains("query byte") || error.contains("native_q byte"))
                        && error.contains("the reference")
                }
                Mutation::InputRound => {
                    error.contains("latent byte") && error.contains("the reference")
                }
            };
            assert!(
                checks.iter().any(|c| c.name == "conformance"
                    && matches!(&c.outcome,Outcome::Failed(error) if numerical_failure(error))),
                "mutant escaped: {checks:?}"
            );
            for c in checks {
                if let Outcome::Failed(error) = &c.outcome {
                    assert!(
                        !error.contains("CUDA") && !error.contains("trap"),
                        "device fault is not negative-control evidence: {error}"
                    );
                }
            }
        }
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required latent glue test skipped");
    }
}

/// The first window-write shape from the faulting suite, isolated for the
/// operator's memcheck/racecheck gate. This includes stage omission replay.
#[test]
fn isolated_window_write_passes_poison_graph_and_pdl_on_gpu() {
    let result = kernel_harness::gpu_test(256 << 20, |session| {
        let directory = std::env::var_os("CHOPINFER_LATENT_GLUE_CUBINS")
            .map(PathBuf::from)
            .ok_or("prebuilt latent glue cubins required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let query = LatentWindowWrite {
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let report = kit::conform(
            &backend(),
            &FamilyModules { directory },
            Some(session),
            &[latent_glue::window_write(query, 17, 130)],
        )?;
        session.receipt(
            "isolated-window-write",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, 1, true);
        let OpResult::Prepared { checks, .. } = &report.ops[0].result else {
            unreachable!()
        };
        let Outcome::Passed(evidence) = &checks
            .iter()
            .find(|c| c.name == "conformance")
            .unwrap()
            .outcome
        else {
            unreachable!()
        };
        for output in ["latent", "quantized_codes"] {
            let guards = evidence["omission_guards"]
                .as_object()
                .expect("guard declarations");
            let (producer, _) = guards
                .iter()
                .find(|(_, value)| value.as_str() == Some(output))
                .expect("producer guard");
            assert!(
                evidence["guarded_omissions"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|entry| entry["stage"].as_str() == Some(producer)),
                "missing guarded omission of {output}"
            );
        }
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required isolated write skipped");
    }
}
