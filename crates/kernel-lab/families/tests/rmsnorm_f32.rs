//! F64 conformance, poison, graph and PDL for F32-output RMSNorm.
use kernel_core::backend::Need;
use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome, Required, cases};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::rmsnorm_f32 as family;
use kernels_sm120::backend::Sm120;
use runtime_cuda::Module;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn backend() -> Sm120 {
    family::entries()
        .into_iter()
        .fold(Sm120::new(), |b, (q, symbol)| b.with_rmsnorm(q, symbol))
}
fn required() -> Vec<Required<'static>> {
    family::entries()
        .into_iter()
        .map(|(q, _)| cases::rmsnorm(format!("plain F32 norm {}", q.width), q, Need::rows(32)))
        .collect()
}
struct FamilyModules {
    directory: PathBuf,
}
impl Modules for FamilyModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let lookup = family::cubins()
            .into_iter()
            .flat_map(|spec| {
                spec.entries
                    .into_iter()
                    .map(move |entry| (entry.symbol, spec.name.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let modules = RefCell::new(BTreeMap::<String, Module>::new());
        Ok(Box::new(move |symbol| {
            let name = lookup
                .get(symbol)
                .ok_or_else(|| format!("missing norm symbol {symbol}"))?;
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
fn check(report: &kit::Report, gpu: bool) {
    assert_eq!(report.prepared().len(), 3, "{:?}", report.ops);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("required norm refused")
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
fn all_f32_norm_cases_prepare_and_declare() {
    check(
        &kit::conform(
            &backend(),
            &FamilyModules {
                directory: PathBuf::new(),
            },
            None,
            &required(),
        )
        .unwrap(),
        false,
    );
}
#[test]
fn f32_norm_passes_f64_poison_graph_and_pdl_on_gpu() {
    let result = kernel_harness::gpu_test(64 << 20, |session| {
        let directory = std::env::var_os("CHOPINFER_RMSNORM_F32_CUBINS")
            .map(PathBuf::from)
            .ok_or("prebuilt F32 norm cubins required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let report = kit::conform(
            &backend(),
            &FamilyModules { directory },
            Some(session),
            &required(),
        )?;
        session.receipt(
            "rmsnorm-f32",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        check(&report, true);
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required F32 norm test skipped");
    }
}
