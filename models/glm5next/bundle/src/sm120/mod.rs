//! The bundle's SM120 glue that names `kernels_sm120` directly: the engine
//! entries it embeds (`engines`), the backend an instance prepares its ops
//! and projections on (`crate::ops`, `crate::projections`), GLM's own
//! kernels (`own_ops`) and the target an instance asks the rest of
//! (`crate::target::Target` on `own_ops::Sm120Own`).

#[cfg(test)]
pub mod device;
pub(crate) mod encoders;
pub mod engines;
pub mod hidden_copy;
#[cfg(test)]
mod launch_tests;
#[cfg(test)]
mod moe_tests;
pub mod own_ops;
mod routes;
#[cfg(test)]
pub(crate) mod tables;
mod target;

use std::sync::Arc;

use kernel_core::backend::Backend;
use kernel_core::contract::{LaunchPlan, StaticQuery};
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use kernel_core::precision::{OperationSpec, OperationUse};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as SinkhornQuery;
use kernels_sm120::backend::{DenseForm, NamedOffers, ProjectionOffer, Sm120};
use kernels_sm120::latent;
use kernels_sm120::linear::Stage as KdaStage;
use kernels_sm120::linear::channel_decay::{self, PrefillSchedule};
use kernels_sm120::row::rounding::PartialRounding;
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::prepared;
use model_glm5next_contract::sm120::engines::{
    BF16_LOW_LATENCY_ROWS, ModelRow, ROUNDED_LOW_LATENCY_ROWS,
};

use crate::ops::Served;
use crate::ops::reduce::{Admission, CommBackends};
use engines::Engines;

/// The bundle's chained MoE locks (the dense MLP's and shared expert's, the
/// split routed experts'), whose records carry each entry's registers and
/// static shared memory as ptxas reported them.
const MOE_LOCKS: [&str; 2] = [
    include_str!("../../kernels.lock.d/moe_chained.lock"),
    include_str!("../../kernels.lock.d/moe_chained_split.lock"),
];

/// The SM120 backend an instance of `shape` prepares its ops on, built once
/// at create from the engines it embeds, and the queries those entries
/// serve. Each family registers the entries its ops run.
pub fn backend(engines: &Engines, shape: Shape) -> (Box<dyn Backend>, Served) {
    let mut served = Vec::new();
    let kda = kda_entries(engines);
    served.extend(kda.iter().map(|entry| match entry.query {
        channel_decay::Query::Rule(query) => StaticQuery::PerChannelDecayDeltaRule(query),
        channel_decay::Query::Norm(query) => StaticQuery::Recurrent(query),
    }));
    let mut backend = Sm120::new().with_channel_decay(kda);
    // The latent entries, with the slice-widened forms the FP8 prefill
    // attention runs and the column merges the split attention runs.
    for entry in &engines.latent {
        served.push(match entry.query {
            latent::Query::Latent(query) => StaticQuery::LatentAttention(query),
            latent::Query::HeadGemm(query) => StaticQuery::HeadBatchedGemm(query),
        });
        backend = backend.with_latent(entry.clone());
    }
    for slices in &engines.latent_slices {
        backend = backend.with_latent_slices(slices.clone());
    }
    for merge in &engines.latent_merge {
        backend = backend.with_latent_merge_columns(merge.clone());
    }
    // The pooled indexer entries, with the one-split forms a selection
    // runs at one split and the tree and radix merges it runs over two or
    // more.
    served.extend(engines.indexer.iter().map(|entry| entry.query()));
    backend = backend
        .with_pooled_indexer(engines.indexer.clone())
        .with_indexer_one_split(engines.indexer_rows.clone())
        .with_indexer_merge_tree(engines.indexer_merge.clone())
        .with_indexer_merge_radix(engines.indexer_radix.clone());
    // The Sinkhorn mixer's entries: the split and tiled mix-collapse a
    // prefill mixer runs by rows, and the split fused with the input norm
    // the decode and verify graphs run.
    let sinkhorn = |query| StaticQuery::SinkhornHyperConnection(query);
    served.extend(engines.sinkhorn.iter().map(|entry| sinkhorn(entry.query)));
    served.push(sinkhorn(engines.sinkhorn_split.query));
    served.extend(
        engines
            .sinkhorn_tiled
            .iter()
            .map(|entry| sinkhorn(entry.query)),
    );
    if let SinkhornQuery::MixCollapse { mix, capacity } = engines.sinkhorn_mix_norm.query {
        served.push(sinkhorn(SinkhornQuery::MixCollapseNorm {
            mix,
            rmsnorm_eps_bits: engines.sinkhorn_mix_norm.norm.epsilon_bits,
            capacity,
        }));
    }
    backend = backend
        .with_sinkhorn(engines.sinkhorn.clone())
        .with_sinkhorn_split(engines.sinkhorn_split.clone())
        .with_sinkhorn_tiled(engines.sinkhorn_tiled.clone())
        .with_sinkhorn_mix_norm(engines.sinkhorn_mix_norm.clone());
    // The context-parallel partial merges and key localizations, with what
    // their launches keep (`context_parallel::Entry::invariance`), and the
    // phased group gather the group's gathers and exchange run (its barrier
    // stays on the shared all-gather).
    let context_parallel = model_glm5next_contract::sm120::context_parallel::entries(shape);
    served.extend(
        context_parallel
            .iter()
            .map(|entry| StaticQuery::ContextParallel(entry.query())),
    );
    let claims = context_parallel
        .iter()
        .map(|entry| (entry.symbol(), entry.invariance()))
        .collect::<Vec<_>>();
    backend = backend
        .with_context_parallel(context_parallel)
        .with_context_parallel_claims(claims)
        .with_phased_group_gather();
    served.dedup();
    let comm = Admission::ALL
        .iter()
        .map(|&admission| {
            let backend: Arc<dyn Backend> = Arc::new(comm_backend(engines, admission));
            (admission, backend)
        })
        .collect();
    let own = own_ops::Sm120Own::new(shape);
    let served = Served(served, CommBackends(comm), Arc::new(own), Arc::new(own));
    let backend = with_rows(with_moe(backend, engines), engines)
        .with_projections(Box::new(NamedOffers(Offers)))
        .with_route_tags(routes::entries(engines));
    (Box::new(backend), served)
}

/// The KDA entries the lowering runs: the prefill delta rule's prepare and
/// scan under the streamed schedule (bitwise the staged pair's), every other
/// stage's one entry. Registering both prefill schedules would tie them.
fn kda_entries(engines: &Engines) -> Vec<channel_decay::Entry> {
    engines
        .kda
        .iter()
        .filter(|entry| {
            (entry.schedule == PrefillSchedule::Streamed)
                == matches!(entry.stage, KdaStage::Prepare | KdaStage::Scan)
        })
        .cloned()
        .collect()
}

/// The backend the all-reduces prepare on when create admits `admission`:
/// the BF16-partials all-reduce's and the MoE sublayers' rounded
/// all-reduce's low-latency entries at their admitted grids and their
/// phased entries where admitted, GLM's round and widen rows for the
/// composed forms (with what they declare, `own_ops::declared_kernels`),
/// and the split comm epilogue.
fn comm_backend(engines: &Engines, admission: Admission) -> Sm120 {
    comm_backend_over(engines, admission, rounding(engines.shape))
}

/// GLM's round and widen rows of `shape`'s hidden rows, with what they
/// declare together.
fn rounding(shape: Shape) -> PartialRounding {
    PartialRounding {
        width: shape.profile.hidden,
        round: ModelRow::Round.entry(shape),
        widen: ModelRow::Widen.entry(shape),
        invariance: own_ops::declared(&ModelRow::Round.symbol())
            .meet(own_ops::declared(&ModelRow::Widen.symbol())),
    }
}

/// [`comm_backend`] over the rounding rows `rounding`.
fn comm_backend_over(engines: &Engines, admission: Admission, rounding: PartialRounding) -> Sm120 {
    let shape = engines.shape;
    let mut backend = Sm120::new()
        .with_partial_rounding(rounding)
        .with_sinkhorn_epilogue(engines.comm_epilogue.clone());
    if admission.low_latency {
        let plan = low_latency_plan(shape).expect("the low-latency all-reduce plans its rows");
        let rounded = rounded_low_latency_plan(shape)
            .expect("the low-latency rounded all-reduce plans its rows");
        backend = backend
            .with_low_latency_bf16_all_reduce(plan.grid[0])
            .with_low_latency_rounded_all_reduce(rounded.grid[0]);
    }
    if admission.phased {
        backend = backend
            .with_phased_bf16_all_reduce()
            .with_phased_f32_all_reduce();
    }
    backend
}

/// `backend` with the chained MoE's entries: every chained projection of
/// every checkpoint layout, the decode and prefill routes over every expert,
/// the F32 routed sum and the sum that adds a row (the split cubins'), and
/// the MoE locks' resources, against which the backend admits tile-pair and
/// decoded-tile forms.
fn with_moe(backend: Sm120, engines: &Engines) -> Sm120 {
    use kernels_sm120::moe::chained::split::PREFIX;
    let mut backend = engines
        .moe
        .iter()
        .cloned()
        .fold(backend, Sm120::with_chained_projection);
    for route in engines.routes {
        backend = backend.with_local_route(route, PREFIX);
    }
    backend = backend
        .with_routed_sum(engines.sum, PREFIX)
        .with_routed_sum_add(engines.sum, PREFIX);
    for lock in MOE_LOCKS {
        let resources =
            kernels_sm120::backend::locked_resources(lock).expect("the bundle's MoE locks parse");
        backend = backend.with_entry_resources(resources);
    }
    backend
}

/// GLM's dense projection offers: each use's shared tiled GEMM offer and
/// layout copy, under the identity the precision component authenticates
/// for it (`prepared::bind`). A use bound to its part of a stacked entry
/// has no offer: it runs in its stacked launch, a GLM own op.
struct Offers;

impl ImplementationProvider for Offers {
    type Handle = ProjectionOffer;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<ProjectionOffer>, String> {
        let bound = prepared::bind(operation, usage)?;
        if bound.handle.stacked.is_some() {
            return Err(format!(
                "operation {} runs at {usage:?} in its stacked entry",
                operation.id
            ));
        }
        Ok(Binding {
            identity: bound.identity,
            handle: ProjectionOffer {
                offer: bound.handle.generated.offer,
                copy: bound.handle.copy,
                dense: DenseForm::Tensor,
                live_output: false,
            },
        })
    }
}

/// The row entries the row ops run: the RMSNorm of each width the program
/// normalizes, the embedding gather of a rank's vocabulary shard, the
/// shared row copies (`engines::ROW_COPY_CUBIN`), the shared gather by
/// index (`engines::ROW_GATHER_CUBIN`) and the shared argmax entries the
/// drafter's argmax runs.
fn with_rows(mut backend: Sm120, engines: &Engines) -> Sm120 {
    for (index, (requirement, symbol)) in engines.norms.iter().enumerate() {
        // Widths that coincide normalize through one entry.
        if !engines.norms[..index].contains(&(*requirement, symbol.clone())) {
            backend = backend.with_rmsnorm(*requirement, symbol.clone());
        }
    }
    for entry in kernels_sm120::row::copy::entries() {
        backend = backend.with_row_copy(entry);
    }
    backend = backend.with_row_gather(model_glm5next_contract::sm120::engines::ROW_GATHER_SYMBOL);
    model_glm5next_contract::sm120::engines::argmax_entries()
        .into_iter()
        .fold(backend, Sm120::with_row_argmax)
        .with_sharded_embedding(engines.embedding.clone())
}

/// The low-latency all-reduce's plan at its largest grid
/// (`BF16_LOW_LATENCY_ROWS` rows), the one create admits against the
/// devices' residency.
pub(crate) fn low_latency_plan(shape: Shape) -> Result<LaunchPlan, String> {
    let elements = u64::from(BF16_LOW_LATENCY_ROWS) * u64::from(shape.profile.hidden);
    kernels_sm120::comm::ll_bf16::plan(shape.ranks(), elements).map_err(|r| r.to_string())
}

/// The MoE sublayers' low-latency round entry's plan at its largest grid
/// (`ROUNDED_LOW_LATENCY_ROWS` rows), the one create admits against the
/// devices' residency.
pub(crate) fn rounded_low_latency_plan(shape: Shape) -> Result<LaunchPlan, String> {
    let elements = u64::from(ROUNDED_LOW_LATENCY_ROWS) * u64::from(shape.profile.hidden);
    kernels_sm120::comm::ll_f32::plan(shape.ranks(), elements).map_err(|r| r.to_string())
}

#[cfg(test)]
mod comm_tests {
    use kernel_core::backend::Refused;
    use kernel_core::properties::Invariance;

    use super::*;
    use crate::ops::{Asks, reduce::ReduceOps};

    /// The all-reduces and comm epilogues of every admission outcome ask
    /// what every phase needs, which the composed form and the epilogue's
    /// widen row keep through GLM's rounding rows' declarations; rounding
    /// rows that declare nothing refuse create.
    #[test]
    fn the_composed_all_reduce_keeps_the_asks_through_the_rounding_rows() {
        let shape = Shape::TP4;
        let engines = Engines::new(shape).unwrap();
        let comm = |rounding: &dyn Fn() -> PartialRounding| {
            CommBackends(
                Admission::ALL
                    .iter()
                    .map(|&admission| {
                        let backend: Arc<dyn Backend> =
                            Arc::new(comm_backend_over(&engines, admission, rounding()));
                        (admission, backend)
                    })
                    .collect(),
            )
        };
        let asks = Asks::default();
        ReduceOps::prepare(&comm(&|| rounding(shape)), shape, &asks).unwrap();
        let asked = asks.asked();
        let composed = asked
            .iter()
            .filter(|asked| asked.what.starts_with("reduce.all_reduce"))
            .filter(|asked| asked.what.contains("phased: false"))
            .collect::<Vec<_>>();
        assert_eq!(composed.len(), 2 * shape.ranks() as usize);
        let epilogues = asked
            .iter()
            .filter(|asked| asked.what.starts_with("reduce.epilogue"));
        assert_eq!(epilogues.clone().count(), 4 * shape.ranks() as usize);
        for asked in composed.into_iter().chain(epilogues) {
            assert_ne!(asked.need.exact, Invariance::NONE, "{}", asked.what);
            assert!(asked.granted.provides(asked.need.exact), "{}", asked.what);
        }
        let undeclared = || PartialRounding {
            invariance: Invariance::NONE,
            ..rounding(shape)
        };
        let refused = ReduceOps::prepare(&comm(&undeclared), shape, &Asks::default());
        assert!(
            matches!(refused, Err(Refused::Inexact { .. })),
            "{:?}",
            refused.err()
        );
    }

    /// The comm epilogue asks its input norm in the 256-lane tree every
    /// hidden-width norm sums in, which the bundle's tree entry states; an
    /// epilogue entry stating the column order refuses create.
    #[test]
    fn the_comm_epilogue_asks_the_tree_norm_its_entry_sums() {
        use kernel_core::rmsnorm::RmsNormReduction;
        let shape = Shape::TP4;
        let comm = |engines: &Engines| {
            CommBackends(
                Admission::ALL
                    .iter()
                    .map(|&admission| {
                        let backend: Arc<dyn Backend> =
                            Arc::new(comm_backend_over(engines, admission, rounding(shape)));
                        (admission, backend)
                    })
                    .collect(),
            )
        };
        let engines = Engines::new(shape).unwrap();
        assert_eq!(
            engines.comm_epilogue.norm,
            RmsNormReduction::WarpTree { threads: 256 }
        );
        ReduceOps::prepare(&comm(&engines), shape, &Asks::default()).unwrap();
        let mut ordered = Engines::new(shape).unwrap();
        ordered.comm_epilogue.norm = RmsNormReduction::Ordered;
        assert!(ReduceOps::prepare(&comm(&ordered), shape, &Asks::default()).is_err());
    }
}
