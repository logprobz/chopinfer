//! The model's native routed catalog, using the shared conformance cases.
//! Cubins and PDL probes are prebuilt and checksummed before a GPU job.

#[path = "../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
pub(super) mod registration;
#[path = "../../sm120/routed_moe.rs"]
mod staged;

use super::row_tiled_bank;
use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::contract::MoeQuery;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_chained::ChainedMlpProjectionQuery;
use kernel_core::moe_experts::PreparedExperts;
use kernel_core::precision::OperationUse;
use kernel_core::router::{RowBiasRouteBuffers, RowBiasRouterQuery};
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

/// The backend of one projection entry, as the bundle serves it: an entry
/// it serves only as a row-tiled build (`row_tiled_bank`) prepares through
/// the served wrapper (`moe_row_tiles_op`), which launches that build and
/// declares the row-tiled bank the kit's case then stores.
fn served_backend(g: &'static Geometry, ranks: u32, entry: Projection) -> Box<dyn Backend> {
    let backend = projection_backend(g, ranks, entry.clone());
    if !row_tiled_bank::symbols().contains_key(&entry.symbol) {
        return Box::new(backend);
    }
    let experts = entry.query.projection.gemm.grouped.unwrap().experts;
    Box::new(Served {
        backend,
        g,
        ranks,
        set: moe::expert_sets(g)
            .into_iter()
            .find(|s| s.count == experts)
            .unwrap(),
        kind: if entry.gate_up() {
            GatedMlpProjection::GateUp
        } else {
            GatedMlpProjection::Down
        },
    })
}

struct Served {
    backend: Sm120,
    g: &'static Geometry,
    ranks: u32,
    set: moe::Experts,
    kind: GatedMlpProjection,
}
impl Backend for Served {
    fn name(&self) -> &str {
        self.backend.name()
    }
    fn chained_mlp_projection(
        &self,
        op: &ChainedMlpProjectionQuery,
        usage: OperationUse,
        need: Need,
    ) -> Result<Box<dyn PreparedExperts>, Refused> {
        let op = self.backend.chained_mlp_projection(op, usage, need)?;
        bundle_deepseek41::sm120::moe_row_tiles_op::prepare(
            self.g, self.ranks, self.set, self.kind, usage, op,
        )
    }
    fn moe_row_bias_route(
        &self,
        router: &RowBiasRouterQuery,
        routes: &MoeQuery,
        rank: u32,
        route_tile: u32,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<RowBiasRouteBuffers<View>>, Refused> {
        self.backend
            .moe_row_bias_route(router, routes, rank, route_tile, usage, need)
    }
}

struct ModelModules {
    directory: PathBuf,
}
impl Modules for ModelModules {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let mut lookup = BTreeMap::new();
        for spec in staged::cubins(GEOMETRIES)? {
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
    use kernel_core::precision::ExecutionPhase;
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

/// The cubins `build_moe` stages are the bundle's routed MoE family: the
/// modules and entries its lock pins, so the build verifies against it.
#[test]
fn model_moe_staged_cubins_are_the_locked_family() {
    let lock = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../bundle/kernels.lock.d")
        .join(format!("{}.lock", registration::FAMILY));
    let lock: serde_json::Value = serde_json::from_slice(&std::fs::read(lock).unwrap()).unwrap();
    // The lock holds the family's modules by name.
    let locked: BTreeMap<String, Vec<String>> = lock
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let entries = m["entries"].as_array().unwrap();
            (
                m["name"].as_str().unwrap().to_owned(),
                entries
                    .iter()
                    .map(|e| e["symbol"].as_str().unwrap().to_owned())
                    .collect(),
            )
        })
        .collect();
    let staged: BTreeMap<String, Vec<String>> = staged::cubins(GEOMETRIES)
        .unwrap()
        .into_iter()
        .map(|s| (s.name, s.entries.into_iter().map(|e| e.symbol).collect()))
        .collect();
    assert_eq!(staged, locked);
}

/// Every op the model MoE GPU test conforms launches only entries of the
/// staged cubins: the route ops and each projection, the row-tiled builds
/// the bundle serves included.
#[test]
fn model_moe_native_cases_launch_staged_entries() {
    use kernel_core::precision::ExecutionPhase;
    let staged: std::collections::BTreeSet<String> = staged::cubins(GEOMETRIES)
        .unwrap()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect();
    let mut tiled = 0;
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            for entry in registration::projections(g, ranks).unwrap() {
                let backend = served_backend(g, ranks, entry.clone());
                let usage = OperationUse {
                    phase: if entry.row_tile == 16 {
                        ExecutionPhase::Decode
                    } else {
                        ExecutionPhase::Prefill
                    },
                    capacity: 1,
                    graph: false,
                };
                let experts = backend
                    .chained_mlp_projection(&entry.query, usage, Need::rows(1))
                    .unwrap();
                let grouped = entry.query.projection.gemm.grouped.unwrap();
                let route = backend
                    .moe_row_bias_route(
                        &RowBiasRouterQuery {
                            experts: grouped.experts,
                            top_k: grouped.top_k,
                            scaling_factor: 1.5,
                        },
                        &MoeQuery::LocalRoutes {
                            experts: grouped.experts,
                            local_experts: grouped.experts,
                            top_k: grouped.top_k,
                            capacity: 1,
                        },
                        0,
                        experts.route_tile(),
                        usage,
                        Need::rows(1),
                    )
                    .unwrap();
                for (symbol, _) in route.info().symbols.iter().chain(&experts.info().symbols) {
                    assert!(staged.contains(symbol), "{symbol}");
                }
                if let Some(build) = row_tiled_bank::symbols().get(&entry.symbol) {
                    assert_eq!(&experts.info().symbols[0].0, build);
                    tiled += 1;
                }
            }
        }
    }
    // The column 8192-token gate/up of the Flash TP4 backbone bank.
    assert_eq!(tiled, 1);
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
                    let backend = served_backend(g, ranks, entry.clone());
                    let symbol = entry.symbol.clone();
                    let case = if entry.gate_up() {
                        kit::cases::moe_chained_quantized::gate_up(entry, 1)
                    } else {
                        kit::cases::moe_chained_quantized::down(entry, 1)
                    };
                    let report = kit::conform(backend.as_ref(), &modules, Some(session), &[case])?;
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
