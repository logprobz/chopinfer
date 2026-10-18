//! Native glue with the model's pinned window and compressed RoPE tables.
#[path = "../../../bundle/src/sm120/latent_glue.rs"]
#[allow(dead_code)]
mod registration;
use kernel_core::backend::Need;
use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome, Required};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::own_ops::rope::{self, RopeTable};
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS,
    geometry::{GEOMETRIES, Geometry},
    latent_glue as contracts,
};
use runtime_cuda::Module;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn required(g: &Geometry, ranks: u32) -> Vec<Required<'static>> {
    let q = contracts::query_rope(g, ranks).unwrap();
    let w = contracts::window_write(g, contracts::ring(g));
    let mut cases = Vec::new();
    for table in [RopeTable::Window, RopeTable::Compressed] {
        let cos_sin = rope::reference(table, &[0, 1, 135, 136, 70_001, 900_001, 127, 128]).unwrap();
        cases.push(
            Required::new(
                format!("native query {table:?}"),
                Need::rows(8),
                move |b, need| b.latent_query_rope(&q, need),
                move |op| kit::cases::latent_glue::query_case_with_table(q, 8, op, cos_sin.clone()),
            )
            .required(),
        );
        let cos_sin =
            rope::reference(table, &(0..8).map(|row| 130 + row / 2).collect::<Vec<_>>()).unwrap();
        cases.push(
            Required::new(
                format!("native window {table:?}"),
                Need::rows(8),
                move |b, need| b.latent_window_write(&w, need),
                move |op| {
                    kit::cases::latent_glue::write_case_with_table(w, 8, 130, op, cos_sin.clone())
                },
            )
            .required(),
        );
    }
    cases.push(kit::cases::latent_glue::window_keys(
        contracts::window_keys(g, contracts::ring(g)),
        8,
    ));
    cases.push(
        kit::cases::rmsnorm(
            "native query norm F32".into(),
            contracts::query_norm(g),
            Need::rows(8),
        )
        .required(),
    );
    cases
}
struct ModelModules {
    directory: PathBuf,
}
impl Modules for ModelModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for (_, spec) in registration::cubins(GEOMETRIES)? {
            for entry in spec.entries {
                lookup.insert(entry.symbol, spec.name.clone());
            }
        }
        let modules = RefCell::new(BTreeMap::<String, Module>::new());
        Ok(Box::new(move |symbol| {
            let name = lookup
                .get(symbol)
                .ok_or_else(|| format!("missing native glue symbol {symbol}"))?;
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
    assert_eq!(report.prepared().len(), 6, "{:?}", report.ops);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("native glue refused")
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
fn native_glue_cases_prepare_for_every_geometry_and_rank() {
    let modules = ModelModules {
        directory: PathBuf::new(),
    };
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = registration::register(Sm120::new(), g, ranks).unwrap();
            check(
                &kit::conform(&backend, &modules, None, &required(g, ranks)).unwrap(),
                false,
            );
        }
    }
}
#[test]
fn native_glue_passes_pinned_tables_and_poison_on_gpu() {
    let result = kernel_harness::gpu_test(512 << 20, |session| {
        let directory = std::env::var_os("CHOPINFER_DEEPSEEK41_GLUE_CUBINS")
            .map(PathBuf::from)
            .ok_or("prebuilt native glue cubins required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = ModelModules { directory };
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                let backend = registration::register(Sm120::new(), g, ranks)?;
                let report = kit::conform(&backend, &modules, Some(session), &required(g, ranks))?;
                session.receipt(
                    &format!("native-glue-{}-tp{ranks}", g.id),
                    BTreeMap::from([("report".into(), report.json())]),
                )?;
                check(&report, true);
            }
        }
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required native glue test skipped");
    }
}
