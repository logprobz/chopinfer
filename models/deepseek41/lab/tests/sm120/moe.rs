//! The model's native routed catalog, using the shared conformance cases.
//! Cubins and PDL probes are prebuilt and checksummed before a GPU job.

#[path = "../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod registration;

use kernel_harness::conformance::Probes;
use kernel_harness::kit::modules::Functions;
use kernel_harness::kit::{self, Modules, OpResult, Outcome, Required};
use kernel_harness::poison::{CopyProducer, LiveProducer};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::moe::chained::quantized::Projection;
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS,
    geometry::{GEOMETRIES, Geometry},
    moe,
};
use runtime_cuda::Module;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn routers(g: &Geometry) -> Vec<Required<'static>> {
    moe::expert_sets(g)
        .into_iter()
        .map(|set| kit::cases::moe_chained_quantized::router(moe::router(g, set), 2))
        .collect()
}

fn projection_backend(g: &Geometry, ranks: u32, entry: Projection) -> Sm120 {
    // Pin this compute entry so capacity selection cannot replace the
    // one-token prefill fixture with the smaller decode registration.
    let backend = registration::routers(g)
        .into_iter()
        .fold(Sm120::new(), Sm120::with_row_bias_router);
    let backend = registration::routes(g)
        .into_iter()
        .fold(backend, Sm120::with_row_bias_route);
    let backend = registration::quantizers(g, ranks)
        .unwrap()
        .into_iter()
        .fold(backend, Sm120::with_block_quantize);
    backend.with_quantized_chained_projection(entry)
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
                .ok_or_else(|| format!("missing model MoE symbol {symbol}"))?;
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
        // The shared chained lab owns these diagnostic probes. The job
        // snapshots their cubin beside the model cubins; no lab crate links.
        let bytes = std::fs::read(self.directory.join("chop_moe_chain_quantized_probes.cubin"))
            .map_err(|e| e.to_string())?;
        let module = session.load_cubin(&bytes)?;
        Ok(Some(Probes {
            live: LiveProducer::with_function(
                session,
                module
                    .function("chop_moe_chain_probe_live_v1")
                    .map_err(|e| e.to_string())?,
            )?,
            copy: Some(CopyProducer::with_function(
                module
                    .function("chop_moe_chain_probe_copy_v1")
                    .map_err(|e| e.to_string())?,
            )),
        }))
    }
}

fn check(report: &kit::Report, expected: usize, gpu: bool) {
    assert!(expected > 0);
    assert_eq!(report.prepared().len(), expected);
    assert!(report.failures().is_empty(), "{:?}", report.failures());
    if gpu {
        for op in &report.ops {
            let OpResult::Prepared { checks, .. } = &op.result else {
                panic!("required model MoE op refused")
            };
            for check in checks {
                if check.name == "property ties" {
                    assert!(matches!(check.outcome, Outcome::NotExercised(_)));
                } else {
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
}

#[test]
fn model_moe_router_cases_prepare() {
    let modules = ModelModules {
        directory: PathBuf::new(),
    };
    let mut count = 0;
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = registration::register(Sm120::new(), g, ranks).unwrap();
            let cases = moe::expert_sets(g)
                .into_iter()
                .map(|set| kit::cases::moe_chained_quantized::router(moe::router(g, set), 2))
                .collect::<Vec<_>>();
            let report = kit::conform(&backend, &modules, None, &cases).unwrap();
            check(&report, cases.len(), false);
            count += cases.len();
        }
    }
    assert_eq!(
        count,
        GEOMETRIES
            .iter()
            .map(|g| moe::expert_sets(g).len() * SUPPORTED_RANKS.len())
            .sum::<usize>()
    );
    assert!(count > 0);
}

#[test]
fn model_moe_native_cases_select_the_declared_tile() {
    use kernel_core::backend::{Backend, Need};
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            for entry in registration::projections(g, ranks).unwrap() {
                let backend = projection_backend(g, ranks, entry.clone());
                let usage = OperationUse {
                    phase: if entry.row_tile == 16 {
                        ExecutionPhase::Decode
                    } else {
                        ExecutionPhase::Prefill
                    },
                    capacity: 1,
                    graph: false,
                };
                let prepared = backend
                    .chained_mlp_projection(&entry.query, usage, Need::rows(1))
                    .unwrap();
                assert_eq!(prepared.route_tile(), entry.row_tile);
                assert_eq!(prepared.info().symbols[0].0, entry.symbol);
            }
        }
    }
}

#[test]
fn model_moe_native_cases_pass_on_gpu() {
    let result = kernel_harness::gpu_test(8 << 30, |session| {
        let directory = std::env::var_os("CHOPINFER_DEEPSEEK41_MOE_CUBINS")
            .map(PathBuf::from)
            .ok_or("prebuilt model MoE cubins are required")?;
        kernel_build::report::BuildReport::read(&directory.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(|e| e.to_string())?;
        let modules = ModelModules { directory };
        for g in GEOMETRIES {
            for &ranks in SUPPORTED_RANKS {
                // Native widths use one token to bound the f64 oracle.
                // Shared family cases also cover partial and full batches.
                for entry in registration::projections(g, ranks)? {
                    let backend = projection_backend(g, ranks, entry.clone());
                    let symbol = entry.symbol.clone();
                    let case = if entry.gate_up() {
                        kit::cases::moe_chained_quantized::gate_up(entry, 1)
                    } else {
                        kit::cases::moe_chained_quantized::down(entry, 1)
                    };
                    let report = kit::conform(&backend, &modules, Some(session), &[case])?;
                    session.receipt(
                        &format!("model-moe-{symbol}"),
                        BTreeMap::from([("report".into(), report.json())]),
                    )?;
                    check(&report, 1, true);
                }
                let cases = routers(g);
                let backend = registration::register(Sm120::new(), g, ranks)?;
                let report = kit::conform(&backend, &modules, Some(session), &cases)?;
                session.receipt(
                    &format!("model-moe-{}-tp{ranks}", g.id),
                    BTreeMap::from([("report".into(), report.json())]),
                )?;
                check(&report, cases.len(), true);
            }
        }
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(result.is_some(), "required model MoE test skipped");
    }
}
