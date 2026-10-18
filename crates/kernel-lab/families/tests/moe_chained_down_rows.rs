//! Scoped preparation and GPU conformance for the quantized MoE family.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome, Required};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::moe_chained_down_rows as family;
use kernel_lab_families::moe_chained_quantized as probes;
use kernels_sm120::backend::Sm120;
use runtime_cuda::Module;

fn required() -> Vec<Required<'static>> {
    family::projections()
        .into_iter()
        .map(|p| kit::cases::moe_chained_quantized::down(p, 32))
        .collect()
}
fn backend() -> Sm120 {
    let backend = family::routers()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_row_bias_router);
    let backend = family::routes()
        .into_iter()
        .fold(backend, Sm120::with_row_bias_route);
    family::projections()
        .into_iter()
        .fold(backend, Sm120::with_quantized_chained_projection)
}

struct FamilyModules {
    directory: PathBuf,
}

impl Modules for FamilyModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for spec in family::cubins().into_iter().chain(probes::cubins()) {
            for entry in spec.entries {
                lookup.insert(entry.symbol, spec.name.clone());
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
        let bytes = std::fs::read(self.directory.join(format!("{}.cubin", probes::PROBES)))
            .map_err(|e| e.to_string())?;
        let module = session.load_cubin(&bytes)?;
        Ok(Some(Probes {
            live: LiveProducer::with_function(
                session,
                module
                    .function(probes::LIVE_PRODUCER)
                    .map_err(|e| e.to_string())?,
            )?,
            copy: Some(CopyProducer::with_function(
                module
                    .function(probes::COPY_PRODUCER)
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
}

fn run_gpu(cases: &[Required<'_>], receipt: &str) {
    assert!(!cases.is_empty());
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        let directory = std::env::var_os("CHOPINFER_MOE_DOWN_ROWS_CUBINS")
            .map(PathBuf::from)
            .ok_or("CHOPINFER_MOE_DOWN_ROWS_CUBINS is required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
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
        check(&report, true, cases.len());
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required GPU case skipped");
    }
}

#[test]
fn down_rows_pass_poison_graph_and_pdl_on_gpu() {
    run_gpu(&required(), "quantized-chained-down-rows");
}
#[test]
fn down_rows_first_case_on_gpu() {
    run_gpu(&required()[..1], "quantized-chained-down-rows-first");
}
