//! Model registration of the native prefill prefix gather.
#[path = "../../../bundle/src/sm120/latent_window_stage.rs"]
#[allow(dead_code)]
mod registration;
use kernel_harness::conformance::Probes;
use kernel_harness::kit::cases::latent_window_stage::{self as cases, Scenario};
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::GEOMETRIES};
use runtime_cuda::Module;
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf};
struct FamilyModules {
    directory: PathBuf,
}
impl Modules for FamilyModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for (_, spec) in registration::cubins(GEOMETRIES)?
            .into_iter()
            .chain(registration::key_cubins(GEOMETRIES)?)
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

#[test]
fn native_window_stage_prepares_for_all_geometries_and_ranks() {
    let modules = FamilyModules {
        directory: PathBuf::new(),
    };
    let cases = Scenario::ALL
        .into_iter()
        .map(cases::required)
        .collect::<Vec<_>>();
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = registration::register(Sm120::new(), g, ranks).unwrap();
            let report = kit::conform(&backend, &modules, None, &cases).unwrap();
            assert_eq!(report.prepared().len(), 7);
            let bounded = kit::conform(
                &backend,
                &modules,
                None,
                &[kit::cases::latent_staged_window_keys::required()],
            )
            .unwrap();
            assert_eq!(bounded.prepared().len(), 1);
            assert!(bounded.failures().is_empty(), "{:?}", bounded.failures());
            assert!(report.failures().is_empty(), "{:?}", report.failures());
        }
    }
}
#[test]
fn native_window_stage_passes_poison_graph_and_pdl_on_gpu() {
    let result = kernel_harness::gpu_test(256 << 20, |session| {
        let directory = PathBuf::from(
            std::env::var_os("CHOPINFER_DEEPSEEK41_WINDOW_STAGE_CUBINS")
                .ok_or("prebuilt native stage cubins required")?,
        );
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = FamilyModules { directory };
        let backend = registration::register(Sm120::new(), &GEOMETRIES[0], 1)?;
        let cases = Scenario::ALL
            .into_iter()
            .map(cases::required)
            .collect::<Vec<_>>();
        let report = kit::conform(&backend, &modules, Some(session), &cases)?;
        session.receipt(
            "native-window-stage",
            BTreeMap::from([("report".into(), report.json())]),
        )?;
        assert_eq!(report.prepared().len(), 7);
        assert!(report.failures().is_empty(), "{:?}", report.failures());
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("native stage refused")
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
        let bounded = kit::conform(
            &backend,
            &modules,
            Some(session),
            &[kit::cases::latent_staged_window_keys::required()],
        )?;
        session.receipt(
            "native-staged-window-keys",
            BTreeMap::from([("report".into(), bounded.json())]),
        )?;
        let OpResult::Prepared { checks, .. } = &bounded.ops[0].result else {
            panic!("native bounded lists refused")
        };
        assert!(
            checks
                .iter()
                .all(|c| matches!(c.outcome, Outcome::Passed(_))),
            "{checks:?}"
        );
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required native staging skipped");
    }
}
