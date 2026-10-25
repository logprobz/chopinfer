//! The ops an instance prepares once through `kernel_core::backend`, and
//! their launches over a step's buffers by role.
//!
//! GLM states each op as a kernel-core contract and the exactness it needs;
//! a backend (`crate::sm120::backend`) prepares it when the instance is
//! created. A step then binds its spans to the op's roles and calls
//! `PreparedOp::launches`: no catalog search or entry choice happens per
//! step. One file per family holds its ops as they move here.

pub(crate) mod cp;
pub(crate) mod draft;
pub(crate) mod exact;
mod indexer;
mod kda;
mod latent;
mod logits;
pub(crate) mod moe;
pub mod own;
mod projection;
pub(crate) mod reduce;
mod rows;
mod sinkhorn;

pub(crate) use logits::PeerViews;
pub(crate) use rows::{CopyViews, EmbeddingViews, NormViews};

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use bundle_runtime::dispatch::BufferBindings;
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::contract::{Refusal, StaticQuery};
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::properties::Invariance;
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::quantization::Layout;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::target::Target;

pub use reduce::StateRegions;

/// What an instance prepares its ops for: its shape, the indexer's runtime
/// top-k and its context-parallel group size (1 without groups).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub shape: Shape,
    /// The checkpoint layout, which gives each MoE projection its weights'
    /// format.
    pub layout: &'static Layout,
    pub index_topk: u32,
    pub context_parallel: u32,
}

/// The queries the backend's registered entries serve
/// (`crate::sm120::backend`). A family states each op in model math and
/// takes the served query that matches it, whose capacities are the
/// compiled entries'. Then the backends the all-reduces prepare on, one per
/// outcome of create's admission of their spinning entries, GLM's own
/// kernels on the same target, and the target itself, which builds the
/// launches no op prepares.
#[derive(Clone)]
pub struct Served(
    pub Vec<StaticQuery>,
    pub reduce::CommBackends,
    pub Arc<dyn OwnOps>,
    pub Arc<dyn Target>,
);

impl Served {
    /// The served queries `pick` keeps, in registration order.
    pub(crate) fn find<T>(&self, pick: impl FnMut(&StaticQuery) -> Option<T>) -> Vec<T> {
        self.0.iter().filter_map(pick).collect()
    }
}

/// Every op an instance prepares at create.
pub struct Ops {
    pub(crate) kda: kda::KdaOps,
    pub(crate) latent: latent::LatentOps,
    pub(crate) indexer: indexer::IndexerOps,
    pub(crate) sinkhorn: sinkhorn::SinkhornOps,
    /// Absent at context-parallel group size 1.
    pub(crate) cp: Option<cp::CpOps>,
    pub(crate) moe: moe::MoeOps,
    pub(crate) reduce: reduce::ReduceOps,
    pub(crate) rows: rows::RowOps,
    pub(crate) logits: logits::LogitsOps,
    pub(crate) draft: draft::DraftOps,
    pub(crate) own: own::OwnRowOps,
    /// What each op asked at create and was granted, in prepare order.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the invariance journal reads the asks")
    )]
    pub(crate) asked: Vec<Asked>,
    /// The target the ops were prepared on.
    pub(crate) target: Arc<dyn Target>,
    /// What the ops were prepared for.
    pub(crate) plan: Plan,
}

impl Ops {
    /// Prepares every op of an instance of `plan` on `backend`, over the
    /// queries its entries serve.
    pub fn prepare(backend: &dyn Backend, served: &Served, plan: Plan) -> Result<Ops, Refused> {
        let asks = Asks::default();
        Ok(Ops {
            kda: kda::KdaOps::prepare(backend, served, plan.shape, &asks)?,
            latent: latent::LatentOps::prepare(backend, served, plan, &asks)?,
            indexer: indexer::IndexerOps::prepare(backend, served, plan, &asks)?,
            sinkhorn: sinkhorn::SinkhornOps::prepare(backend, served, plan.shape, &asks)?,
            cp: cp::CpOps::prepare(backend, served, plan, &asks)?,
            moe: moe::MoeOps::prepare(backend, plan.shape, plan.layout, &asks)?,
            reduce: reduce::ReduceOps::prepare(&served.1, plan.shape, &asks)?,
            rows: rows::RowOps::prepare(backend, plan.shape, &asks)?,
            logits: logits::LogitsOps::prepare(backend, plan.shape, &asks)?,
            draft: draft::DraftOps::prepare(backend, plan.shape, &asks)?,
            own: own::OwnRowOps::prepare(
                &served.2,
                plan.shape,
                crate::lower::eh_uses()?,
                served.3.as_ref(),
                &asks,
            )?,
            asked: asks.asked(),
            target: served.3.clone(),
            plan,
        })
    }

    /// The ops of `plan` on `target`'s backend, prepared once per process
    /// for each plan and shared read-only by every instance after: the ops
    /// are a pure function of the plan on one target, and the Sinkhorn ops'
    /// plans of every row count take most of a prepare's time.
    pub fn shared(target: &dyn Target, plan: Plan) -> Result<Arc<Ops>, Refused> {
        type Cache = Vec<(Plan, Arc<Ops>)>;
        static PREPARED: Mutex<Cache> = Mutex::new(Vec::new());
        // A panic while preparing leaves no entry, so the list stays usable.
        let mut prepared = PREPARED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, ops)) = prepared.iter().find(|(at, _)| *at == plan) {
            return Ok(ops.clone());
        }
        let (backend, served) = target.backend(plan.layout).map_err(Refused::Rule)?;
        let ops = Arc::new(Ops::prepare(backend.as_ref(), &served, plan)?);
        prepared.push((plan, ops.clone()));
        Ok(ops)
    }

    /// The comm state regions every rank allocates for the instance's
    /// all-reduces (`buffers::CommState::allocate`).
    pub fn comm_regions(&self) -> StateRegions {
        self.reduce.regions()
    }

    /// The comm state regions every rank allocates for each of the
    /// drafter argmax's pair gathers, on an instance that runs the drafter.
    pub fn draft_regions(&self) -> [u64; 2] {
        self.draft.regions()
    }

    /// The ops of `shape` with the compiled top-k in groups of
    /// `context_parallel` ranks on the SM120 backend of the block-FP8
    /// export, for tests that lower stages without an instance.
    #[cfg(test)]
    pub(crate) fn for_tests(shape: Shape, context_parallel: u32) -> Ops {
        let plan = Plan {
            shape,
            layout: &model_glm5next_contract::quantization::FP8_BLOCK128,
            index_topk: shape.profile.index_topk_max,
            context_parallel,
        };
        Ops::for_plan(plan)
    }

    /// The ops of `plan` on the SM120 backend, for tests.
    #[cfg(test)]
    pub(crate) fn for_plan(plan: Plan) -> Ops {
        let target = crate::sm120::own_ops::Sm120Own::new(plan.shape);
        let (backend, served) = target.backend(plan.layout).unwrap();
        Ops::prepare(backend.as_ref(), &served, plan).unwrap()
    }
}

/// One op's ask at create: what GLM needs of it, and what the backend
/// granted (`OpInfo::guarantees`, which provide the need, or the backend
/// refused the op and create with it).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Asked {
    /// The op: its family, role and phase, with its rank where it has one.
    pub what: String,
    pub need: Need,
    pub granted: Invariance,
    /// The backend's versioned implementation (`OpInfo::implementation`).
    pub implementation: String,
    /// The symbols the op may launch.
    pub symbols: Vec<String>,
}

#[cfg(test)]
thread_local! {
    /// Asks on this thread drop their exactness, for the test that the
    /// asks choose what the rows alone choose.
    pub(crate) static WITHOUT_EXACT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
thread_local! {
    /// While some, each launch the ops lowered on this thread: its symbol,
    /// and the implementation and guarantees of the op that launched it.
    pub(crate) static LOWERED: RefCell<Option<Vec<(String, String, Invariance)>>> =
        const { RefCell::new(None) };
}

/// A prepared op whose launches the thread's [`LOWERED`] record lists, for
/// launches the lowering takes from outside `Bind` (the projections').
#[cfg(test)]
pub(crate) struct Recorded<B>(pub Prepared<B>);

#[cfg(test)]
impl<B> PreparedOp<B> for Recorded<B> {
    fn route(&self, counts: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        self.0.route(counts, launch)
    }

    fn info(&self) -> &OpInfo {
        self.0.info()
    }

    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let launches = self.0.launches(buffers, workspace, counts)?;
        record(self.info(), &launches);
        Ok(launches)
    }
}

/// Lists `launches` of the op of `info` in the thread's [`LOWERED`] record.
#[cfg(test)]
fn record(info: &OpInfo, launches: &[Launch]) {
    LOWERED.with(|lowered| {
        if let Some(lowered) = lowered.borrow_mut().as_mut() {
            lowered.extend(launches.iter().map(|launch| {
                (
                    launch.symbol.clone(),
                    info.implementation.clone(),
                    info.guarantees,
                )
            }));
        }
    });
}

/// The asks of one create, recorded as each op prepares.
#[derive(Default)]
pub(crate) struct Asks(RefCell<Vec<Asked>>);

/// A prepared op whose backend choices can be read.
pub(crate) trait Info {
    fn op_info(&self) -> &OpInfo;
}

impl<B> Info for Prepared<B> {
    fn op_info(&self) -> &OpInfo {
        self.info()
    }
}

impl Info for Box<dyn PreparedExperts> {
    fn op_info(&self) -> &OpInfo {
        self.info()
    }
}

impl Asks {
    /// The asks recorded, in the order the ops prepared.
    pub(crate) fn asked(self) -> Vec<Asked> {
        self.0.into_inner()
    }

    /// The op `prepare` makes for `need`, its ask recorded as `what`.
    pub(crate) fn ask<T: Info>(
        &self,
        what: impl Into<String>,
        need: Need,
        prepare: impl FnOnce(Need) -> Result<T, Refused>,
    ) -> Result<T, Refused> {
        let need = Self::asked_need(need);
        let prepared = prepare(need)?;
        self.record(what, need, prepared.op_info());
        Ok(prepared)
    }

    /// `need` as an op is asked for it: with its rows alone on a thread
    /// that drops the asks' exactness ([`WITHOUT_EXACT`]).
    pub(crate) fn asked_need(need: Need) -> Need {
        #[cfg(test)]
        let need = match WITHOUT_EXACT.with(std::cell::Cell::get) {
            true => Need::rows(need.capacity),
            false => need,
        };
        need
    }

    /// Records the ask `what` of an op another prepared for `need`
    /// ([`Asks::asked_need`]), whose info is `info`.
    pub(crate) fn record(&self, what: impl Into<String>, need: Need, info: &OpInfo) {
        self.0.borrow_mut().push(Asked {
            what: what.into(),
            need,
            granted: info.guarantees,
            implementation: info.implementation.clone(),
            symbols: info.symbols.iter().map(|(s, _)| s.clone()).collect(),
        });
    }
}

/// The one launch of an op that launches once.
pub(crate) fn one<T>(launches: Vec<T>) -> Result<T, String> {
    let count = launches.len();
    let [launch] = <[T; 1]>::try_from(launches)
        .map_err(|_| format!("an op of one launch launched {count}"))?;
    Ok(launch)
}

/// A launch with whether it waits for its peer ranks, as the op that made
/// it states (`Launch::collective`).
pub struct Lowered {
    pub launch: PreparedLaunch,
    pub collective: bool,
}

impl Lowered {
    /// A launch that waits for no peer and that no op states facts of:
    /// one of GLM's own row kernels its builders still make (the stream
    /// expansion, the router's score compaction), a dense GEMM or layout
    /// copy of the projection API (`bundle_runtime::projection`), or
    /// the encoder rows' injection.
    pub fn local(launch: PreparedLaunch) -> Self {
        Self {
            launch,
            collective: false,
        }
    }
}

/// The launches of `lowered`, without their facts.
pub fn launches(lowered: Vec<Lowered>) -> Vec<PreparedLaunch> {
    lowered.into_iter().map(|lowered| lowered.launch).collect()
}

/// Spans bound for one stage's launches, as role views.
#[derive(Default)]
pub(crate) struct Bind(BufferBindings);

impl Bind {
    pub(crate) fn view(&mut self, span: DeviceBufferSpanV1) -> View {
        View {
            handle: self.0.bind(span),
            bytes: span.length,
        }
    }

    /// `op`'s launches over `buffers` as the pipeline's launches, every one
    /// with these bindings and the facts the op states of it.
    pub(crate) fn launch<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lowered_op(op.as_ref(), buffers, counts)
    }

    /// An expert projection's launches with the facts the op states of
    /// each, as [`Bind::launch`]'s.
    pub(crate) fn lowered_experts(
        self,
        op: &dyn PreparedExperts,
        buffers: &GatedMlpBuffers<View>,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lowered_op::<GatedMlpBuffers<View>>(op, buffers, counts)
    }

    /// [`Bind::launch`] of an op that takes backend workspace, bound to
    /// `workspace`.
    pub(crate) fn launch_within<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        let launches = op.launches(buffers, workspace, counts)?;
        let routes: Vec<_> = launches
            .iter()
            .map(|launch| kernel_core::route::selected(op.as_ref(), counts, launch))
            .collect();
        let mut lowered = self.lowered(op.info(), launches);
        for (item, route) in lowered.iter_mut().zip(routes) {
            item.launch.bindings = item.launch.bindings.with_route(route);
        }
        Ok(lowered)
    }

    fn lowered_op<B>(
        self,
        op: &dyn PreparedOp<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        if op.info().workspace != 0 {
            return Err(Refusal("GLM 5.3 Flash binds no backend workspace"));
        }
        let launches = op.launches(buffers, View::EMPTY, counts)?;
        let routes: Vec<_> = launches
            .iter()
            .map(|launch| kernel_core::route::selected(op, counts, launch))
            .collect();
        let mut lowered = self.lowered(op.info(), launches);
        for (item, route) in lowered.iter_mut().zip(routes) {
            item.launch.bindings = item.launch.bindings.with_route(route);
        }
        Ok(lowered)
    }

    /// `launches` of the op of `info` with these bindings and the facts
    /// the op states of each.
    fn lowered(&self, info: &OpInfo, launches: Vec<Launch>) -> Vec<Lowered> {
        #[cfg(test)]
        record(info, &launches);
        #[cfg(not(test))]
        let _ = info;
        launches
            .into_iter()
            .map(|launch| Lowered {
                collective: launch.collective,
                launch: self.prepared(launch),
            })
            .collect()
    }

    fn prepared(&self, launch: Launch) -> PreparedLaunch {
        PreparedLaunch {
            symbol: launch.symbol,
            rows: launch.rows,
            plan: launch.plan,
            arguments: launch.arguments,
            bindings: self.0.clone(),
        }
    }
}
