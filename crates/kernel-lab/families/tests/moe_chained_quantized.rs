//! Scoped preparation and GPU conformance for the quantized MoE family.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

use kernel_core::router::RowBiasRouterQuery;
use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome, Required};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::moe_chained_gate_up as multi_chain;
use kernel_lab_families::moe_chained_output_quantize as output_quantize;
use kernel_lab_families::moe_chained_quantized as family;
use kernels_sm120::backend::Sm120;
use runtime_cuda::Module;

fn required() -> Vec<Required<'static>> {
    let mut required = family::routers()
        .into_iter()
        .map(|r| {
            kit::cases::moe_chained_quantized::router(
                RowBiasRouterQuery {
                    experts: r.experts,
                    top_k: r.top_k,
                    scaling_factor: 1.5,
                },
                r.capacity,
            )
        })
        .collect::<Vec<_>>();
    required.extend(
        family::projections()
            .into_iter()
            .filter(|p| !p.gate_up())
            .map(|p| kit::cases::moe_chained_quantized::down(p, 32)),
    );
    required
}

fn isolated_down() -> Vec<Required<'static>> {
    let projection = family::projections()
        .into_iter()
        .find(|p| {
            let grouped = p.query.projection.gemm.grouped.unwrap();
            !p.gate_up() && p.row_tile == 16 && grouped.experts == 384 && grouped.top_k == 6
        })
        .expect("the failing routed-down geometry must remain in the family");
    vec![kit::cases::moe_chained_quantized::down(projection, 32)]
}

fn gate_up_projections() -> Vec<kernels_sm120::moe::chained::quantized::Projection> {
    family::projections()
        .into_iter()
        .filter(|p| p.gate_up())
        .chain(multi_chain::projections())
        .collect()
}

fn gate_up_required() -> Vec<Required<'static>> {
    gate_up_projections()
        .into_iter()
        .map(|p| kit::cases::moe_chained_quantized::gate_up(p, 32))
        .collect()
}

fn backend() -> Sm120 {
    let backend = family::routers()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_row_bias_router);
    let backend = family::routes()
        .into_iter()
        .fold(backend, Sm120::with_row_bias_route);
    let backend = output_quantize::entries()
        .into_iter()
        .fold(backend, Sm120::with_block_quantize);
    family::projections()
        .into_iter()
        .chain(multi_chain::projections())
        .fold(backend, Sm120::with_quantized_chained_projection)
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
        for spec in output_quantize::cubins() {
            for entry in spec.entries {
                lookup.insert(entry.symbol, format!("output-quantize/{}", spec.name));
            }
        }
        for spec in multi_chain::cubins() {
            for entry in spec.entries {
                lookup.insert(entry.symbol, format!("gate-up/{}", spec.name));
            }
        }
        let loaded = RefCell::new(BTreeMap::<String, Module>::new());
        Ok(Box::new(move |symbol| {
            let name = lookup
                .get(symbol)
                .ok_or_else(|| format!("family has no symbol {symbol}"))?;
            let mut loaded = loaded.borrow_mut();
            if !loaded.contains_key(name) {
                let bytes = std::fs::read(self.directory.join(format!("{name}.cubin")))
                    .map_err(|e| e.to_string())?;
                loaded.insert(name.clone(), session.load_cubin(&bytes)?);
            }
            loaded[name].function(symbol).map_err(|e| e.to_string())
        }))
    }

    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let bytes = std::fs::read(self.directory.join(format!("{}.cubin", family::PROBES)))
            .map_err(|e| e.to_string())?;
        let module = session.load_cubin(&bytes)?;
        Ok(Some(Probes {
            live: LiveProducer::with_function(
                session,
                module
                    .function(family::LIVE_PRODUCER)
                    .map_err(|e| e.to_string())?,
            )?,
            copy: Some(CopyProducer::with_function(
                module
                    .function(family::COPY_PRODUCER)
                    .map_err(|e| e.to_string())?,
            )),
        }))
    }
}

fn check(report: &kit::Report, gpu: bool, expected: usize) {
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    assert_eq!(report.prepared().len(), expected);
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            panic!("required op refused");
        };
        for check in checks {
            if gpu {
                // The generic property runner has no tie fixture. This
                // case checks tied keys against its emitted F32 scores
                // in every conformance phase instead.
                if check.name == "property ties" {
                    assert!(matches!(check.outcome, Outcome::NotExercised(_)));
                    continue;
                }
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
fn every_chained_case_prepares_and_declares() {
    let modules = FamilyModules {
        directory: PathBuf::new(),
    };
    let report = kit::conform(&backend(), &modules, None, &required()).unwrap();
    check(&report, false, 6);
    let isolated = kit::conform(&backend(), &modules, None, &isolated_down()).unwrap();
    check(&isolated, false, 1);
    let gate_up = kit::conform(&backend(), &modules, None, &gate_up_required()).unwrap();
    check(&gate_up, false, 5);
    let controls = gate_up_projections()
        .into_iter()
        .map(|p| kit::cases::moe_chained_quantized::gate_up_unit_weight_control(p, 32))
        .collect::<Vec<_>>();
    check(
        &kit::conform(&backend(), &modules, None, &controls).unwrap(),
        false,
        5,
    );
}

fn run_gpu(cases: &[Required<'_>], receipt: &str, negative_control: bool) {
    assert!(!cases.is_empty());
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        let directory = std::env::var_os("CHOPINFER_MOE_CHAIN_CUBINS")
            .map(PathBuf::from)
            .ok_or("CHOPINFER_MOE_CHAIN_CUBINS is required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        kernel_build::report::BuildReport::read(
            &directory.join("output-quantize/kernel-build-report.json"),
        )
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
        kernel_build::report::BuildReport::read(
            &directory.join("gate-up/kernel-build-report.json"),
        )
        .and_then(|r| r.verify_current())
        .map_err(|e| e.to_string())?;
        let report = kit::conform(
            &backend(),
            &FamilyModules { directory },
            Some(session),
            cases,
        )?;
        session.receipt(
            receipt,
            BTreeMap::from([
                ("report".into(), report.json()),
                (
                    "selection_gate".into(),
                    serde_json::json!(
                        "exact ids and weights over emitted F32 scores, including ties"
                    ),
                ),
            ]),
        )?;
        if negative_control {
            assert_eq!(report.prepared().len(), cases.len());
            for op in &report.ops {
                let OpResult::Prepared { checks, .. } = &op.result else {
                    panic!("control not prepared")
                };
                assert!(
                    checks.iter().any(|c| c.name == "conformance"
                        && matches!(
                    &c.outcome, Outcome::Failed(error) if error.contains("gate/up BF16 oracle:"))),
                    "unit weights must fail the numerical oracle: {checks:?}"
                );
                for check in checks {
                    if let Outcome::Failed(error) = &check.outcome {
                        assert!(
                            error.contains("gate/up BF16 oracle:"),
                            "unexpected control failure: {error}"
                        );
                    }
                }
            }
        } else {
            check(&report, true, cases.len());
        }
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required GPU case skipped");
    }
}

#[test]
fn chained_cases_pass_poison_graph_and_pdl_on_gpu() {
    run_gpu(&required(), "quantized-chained-moe", false);
}

#[test]
fn routed_down_e384_top6_tile16_passes_poison_graph_and_pdl_on_gpu() {
    run_gpu(&isolated_down(), "quantized-chained-down-isolated", false);
}

#[test]
fn gate_up_cases_pass_poison_graph_and_pdl_on_gpu() {
    run_gpu(&gate_up_required(), "quantized-chained-gate-up", false);
    let controls = gate_up_projections()
        .into_iter()
        .map(|p| kit::cases::moe_chained_quantized::gate_up_unit_weight_control(p, 32))
        .collect::<Vec<_>>();
    run_gpu(
        &controls,
        "quantized-chained-gate-up-unit-weight-control",
        true,
    );
}
