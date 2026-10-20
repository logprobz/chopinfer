//! The ops an instance prepares once through `kernel_core::backend` and
//! the model's `OwnOps`, and their launches over a step's buffers by role.
//!
//! DeepSeek states each op as a kernel-core contract (or one of its own)
//! and the exactness it needs; the target's kernels (`crate::Kernels`)
//! prepare it when the instance is created. A step then binds its spans to
//! the op's roles and calls `PreparedOp::launches`: no catalog search or
//! entry choice happens per step. One file per family holds its ops.

pub(crate) mod compressor;
pub(crate) mod dense;
pub(crate) mod engram;
pub(crate) mod engram_hash;
pub(crate) mod exact;
pub(crate) mod exact_f32;
pub(crate) mod fold_stream;
mod hc;
pub(crate) mod head;
pub(crate) mod indexer;
pub(crate) mod latent;
pub(crate) mod logits;
pub(crate) mod moe;
pub(crate) mod norm_prefill;
pub(crate) mod norm_quantize;
mod reduce;
mod row_buckets;
mod rows;
pub(crate) mod shared;

pub(crate) use compressor::{PlaneViews, PoolingViews, WriteViews};
pub(crate) use dense::GemmViews;
pub(crate) use hc::{ExpandViews, HeadViews, MixViews};
#[cfg(test)]
pub(crate) use reduce::with_bf16_gather;
pub(crate) use reduce::{PeerViews, PrefillDma, ReduceViews, WidenViews};
#[cfg(test)]
pub(crate) use reduce::{with_dma, with_dma_setting};
pub(crate) use rows::{EmbeddingViews, NormViews, StreamViews, verify_rows};

use std::cell::RefCell;
use std::collections::BTreeMap;

use bundle_runtime::dispatch::BufferBindings;
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::Kernels;
use crate::shape::Shape;

/// The model's name in refusals.
pub(crate) const NAME: &str = "DeepSeek V4.1";

/// Every op an instance prepares at create. A family the kernels refuse is
/// absent and named in `missing`: create refuses such an instance, except
/// the fake-host tests that run the lowered stages alone.
pub struct Ops {
    pub(crate) prefill_dma: PrefillDma,
    pub(crate) expert_set: model_deepseek41_contract::moe::Experts,
    pub(crate) rows: Option<rows::RowOps>,
    pub(crate) norm_quantize: Option<norm_quantize::NormQuantizeOps>,
    pub(crate) norm_prefill: Option<norm_prefill::NormPrefillOps>,
    pub(crate) reduce: Option<reduce::ReduceOps>,
    pub(crate) hc: Option<hc::HcOps>,
    pub(crate) moe: Option<moe::MoeOps>,
    pub(crate) compressor: Option<compressor::CompressorOps>,
    pub(crate) dense: Option<dense::DenseOps>,
    pub(crate) shared: Option<shared::SharedOps>,
    pub(crate) indexer: Option<indexer::IndexerOps>,
    pub(crate) select: Option<indexer::SelectOps>,
    pub(crate) glue: Option<latent::GlueOps>,
    pub(crate) logits: Option<logits::LogitsOps>,
    pub(crate) engram: Option<engram::EngramOps>,
    pub(crate) head: Option<head::HeadOps>,
    pub(crate) exact_f32: Option<exact_f32::ExactF32Ops>,
    /// The attention of each compress ratio the kernels serve.
    pub(crate) attention: Vec<latent::AttentionOps>,
    /// Each family the kernels refused, with the refusal.
    pub(crate) missing: Vec<String>,
    /// What each op asked at create and was granted, in prepare order.
    pub(crate) asked: Vec<Asked>,
    /// The kernels' names: the shared contracts' backend and the model's
    /// own ops'.
    pub(crate) backends: [String; 2],
}

impl Ops {
    /// The largest prepared shared projection or activation workspace,
    /// including paired gate/up and every decode bucket the backend offers.
    pub(crate) fn shared_workspace(&self) -> u64 {
        self.asked
            .iter()
            .filter(|asked| {
                asked.what.starts_with("dense.moe.shared-")
                    || asked.what.starts_with("dense.shared-pair")
                    || asked.what.starts_with("shared.")
            })
            .map(|asked| asked.workspace)
            .max()
            .unwrap_or(0)
    }

    /// Prepares every op of an instance of `shape` on `kernels` over KV
    /// pages in `storage`, recording each family they refuse.
    pub(crate) fn prepare(kernels: &Kernels, shape: &Shape, storage: KvStorage) -> Ops {
        let mut ops = Self::prepare_set(
            kernels,
            shape,
            storage,
            model_deepseek41_contract::moe::Experts {
                count: shape.geometry.shape.moe.experts,
                top_k: shape.geometry.shape.moe.top_k,
            },
        );
        if reduce::requested(shape)
            && shape.ranks() == 4
            && shape.dims().hidden.is_multiple_of(1024)
            && shape.geometry.layers() < 256
        {
            let asks = Asks::default();
            if let Some(reduce) = &mut ops.reduce {
                let backend = Kernels::prefill_reduction_backend(shape);
                if let Err(error) = reduce.prepare_prefill(backend.as_ref(), shape, &asks) {
                    ops.missing.push(format!("prefill BF16 gather: {error}"));
                }
            }
            ops.asked.extend(asks.asked());
        }
        let dma = reduce::dma_requested();
        if dma.enabled()
            && shape.ranks() == 4
            && shape.geometry == &model_deepseek41_contract::geometry::FLASH
        {
            let asks = Asks::default();
            asks.record(
                "reduce.prefill_dma_owner",
                Need::rows(8192),
                &Kernels::prefill_dma_sum_info(),
            );
            ops.asked.extend(asks.asked());
            ops.prefill_dma = dma;
        }
        ops
    }

    /// Prepare the same backbone operations for one declared expert bank.
    /// The target and drafter retain separate operation sets and router widths.
    pub(crate) fn prepare_set(
        kernels: &Kernels,
        shape: &Shape,
        storage: KvStorage,
        expert_set: model_deepseek41_contract::moe::Experts,
    ) -> Ops {
        let asks = Asks::default();
        let mut missing = Vec::new();
        let names = [kernels.backend.name(), kernels.own.name()].join(" and ");
        let rows = family(
            &mut missing,
            &names,
            "row",
            rows::RowOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let norm_quantize = family(
            &mut missing,
            &names,
            "fused norm",
            norm_quantize::NormQuantizeOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let norm_prefill = if shape.geometry == &model_deepseek41_contract::geometry::FLASH {
            family(
                &mut missing,
                &names,
                "prefill fused norm",
                norm_prefill::NormPrefillOps::prepare(kernels.backend.as_ref(), &asks),
            )
        } else {
            None
        };
        let reduce = family(
            &mut missing,
            &names,
            "all-reduce",
            reduce::ReduceOps::prepare(
                kernels.backend.as_ref(),
                kernels.own.as_ref(),
                shape,
                &asks,
            ),
        );
        let hc = family(
            &mut missing,
            &names,
            "hyper-connection",
            hc::HcOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let moe = family(
            &mut missing,
            &names,
            "routed MoE",
            moe::MoeOps::prepare(kernels.backend.as_ref(), shape, &asks, expert_set),
        );
        let compressor = family(
            &mut missing,
            &names,
            "compressor",
            compressor::CompressorOps::prepare(
                kernels.backend.as_ref(),
                kernels.own.as_ref(),
                shape,
                storage,
                &asks,
            ),
        );
        let dense = family(
            &mut missing,
            &names,
            "dense projection",
            dense::DenseOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let shared = family(
            &mut missing,
            &names,
            "shared expert",
            shared::SharedOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let indexer = family(
            &mut missing,
            &names,
            "indexer",
            indexer::IndexerOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let select = family(
            &mut missing,
            &names,
            "index selection",
            indexer::SelectOps::prepare(kernels.backend.as_ref(), shape, storage, &asks),
        );
        let glue = family(
            &mut missing,
            &names,
            "latent glue",
            latent::GlueOps::prepare(
                kernels.backend.as_ref(),
                kernels.own.as_ref(),
                shape,
                &asks,
                storage,
            ),
        );
        let logits = family(
            &mut missing,
            &names,
            "logits",
            logits::LogitsOps::prepare(
                kernels.backend.as_ref(),
                kernels.gather_state,
                shape,
                &asks,
            ),
        );
        let engram = family(
            &mut missing,
            &names,
            "Engram",
            engram::EngramOps::prepare(
                kernels.backend.as_ref(),
                kernels.own.as_ref(),
                shape,
                &asks,
            ),
        );
        let head = family(
            &mut missing,
            &names,
            "LM head",
            head::HeadOps::prepare(kernels.backend.as_ref(), shape, &asks),
        );
        let exact_f32 = family(
            &mut missing,
            &names,
            "exact F32 projection",
            exact_f32::ExactF32Ops::prepare(
                kernels.backend.as_ref(),
                kernels.own.as_ref(),
                shape,
                &asks,
                expert_set.count,
            ),
        );
        let mut ratios: Vec<u32> = shape.geometry.shape.compress_ratios.to_vec();
        ratios.sort_unstable();
        ratios.dedup();
        let attention = ratios
            .into_iter()
            .filter_map(|ratio| {
                family(
                    &mut missing,
                    &names,
                    &format!("ratio-{ratio} latent attention"),
                    latent::AttentionOps::prepare(kernels, shape, ratio, storage, &asks),
                )
            })
            .collect();
        Ops {
            prefill_dma: PrefillDma::Off,
            expert_set,
            rows,
            norm_quantize,
            norm_prefill,
            reduce,
            hc,
            moe,
            compressor,
            dense,
            shared,
            indexer,
            select,
            glue,
            logits,
            engram,
            head,
            exact_f32,
            attention,
            missing,
            asked: asks.asked(),
            backends: [
                kernels.backend.name().to_owned(),
                kernels.own.name().to_owned(),
            ],
        }
    }
}

impl Ops {
    /// The shared workspace every rank holds: backend scratch and the
    /// attention partials bound to the same span, used by successive ops.
    pub(crate) fn workspace(&self) -> u64 {
        self.asked
            .iter()
            .map(|asked| asked.workspace)
            .chain(self.attention.iter().map(latent::AttentionOps::workspace))
            .max()
            .unwrap_or(0)
    }

    /// Refuses a rank index that stores a weight view an exact F32
    /// projection or the head reads in another layout than its op reads:
    /// such a view binds and runs, but on scrambled weights.
    pub(crate) fn check_weight_layouts(
        &self,
        g: &Geometry,
        index: &bundle_runtime::index::RankIndex,
    ) -> Result<(), String> {
        let mut expected = match &self.exact_f32 {
            Some(ops) => ops.weight_layouts(g)?,
            None => Vec::new(),
        };
        if let Some(head) = &self.head {
            expected.push(("head.weight".to_owned(), head.weight_layout()?));
        }
        for (name, layout) in expected {
            if let Some(view) = index.views.get(&name)
                && view.layout_id != layout
            {
                return Err(format!(
                    "{name} is stored as {}, but its op reads {layout}",
                    view.layout_id
                ));
            }
        }
        Ok(())
    }

    /// The dynamic shared memory each entry an op launches takes: the most
    /// any op states for it. Load configures every entry above the default
    /// (`acquire::load_module`); an entry no op launches takes none.
    pub(crate) fn shared_bytes(&self) -> BTreeMap<String, u32> {
        let mut bytes = BTreeMap::new();
        for (symbol, shared) in self.asked.iter().flat_map(|asked| &asked.symbols) {
            let most = bytes.entry(symbol.clone()).or_insert(0);
            *most = (*most).max(*shared);
        }
        bytes
    }
}

/// A family's ops, or `None` with the refusal recorded in `missing`.
fn family<T>(
    missing: &mut Vec<String>,
    kernels: &str,
    name: &str,
    prepared: Result<T, Refused>,
) -> Option<T> {
    match prepared {
        Ok(ops) => Some(ops),
        Err(refusal) => {
            missing.push(format!("{kernels} refuse {NAME}'s {name} ops: {refusal}"));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use model_deepseek41_contract::own_ops::OwnOps;

    struct Nothing;

    impl Backend for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    impl OwnOps for Nothing {
        fn name(&self) -> &str {
            "nothing own"
        }
    }

    #[test]
    fn kernels_without_the_ops_prepare_nothing_and_name_each_family() {
        let shape = Shape::new(
            &model_deepseek41_contract::geometry::FLASH,
            bundle_runtime::geometry::Ranks::TP4,
        )
        .unwrap();
        let kernels = Kernels {
            backend: Box::new(Nothing),
            own: Box::new(Nothing),
            gather_state: [0; 2],
        };
        let ops = Ops::prepare(&kernels, &shape, KvStorage::Native);
        assert!(ops.rows.is_none() && ops.reduce.is_none() && ops.hc.is_none());
        assert!(ops.moe.is_none() && ops.compressor.is_none() && ops.dense.is_none());
        assert!(ops.asked.is_empty());
        assert_eq!(ops.backends, ["nothing", "nothing own"]);
        assert!(
            ops.missing[0].starts_with("nothing and nothing own refuse DeepSeek V4.1's row ops"),
            "{:?}",
            ops.missing
        );
        assert!(
            ops.missing[1]
                .starts_with("nothing and nothing own refuse DeepSeek V4.1's fused norm ops")
        );
        assert!(ops.norm_quantize.is_none());
        assert!(ops.norm_prefill.is_none());
        assert!(
            ops.missing[2].starts_with(
                "nothing and nothing own refuse DeepSeek V4.1's prefill fused norm ops"
            )
        );
        assert!(
            ops.missing[3]
                .starts_with("nothing and nothing own refuse DeepSeek V4.1's all-reduce ops")
        );
    }
}

/// One op's ask at create: what DeepSeek needs of it, and what the backend
/// granted (`OpInfo::guarantees`, which provide the need, or the backend
/// refused the op and create with it).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Asked {
    /// The op: its family and role, with its rank where it has one.
    pub what: String,
    pub need: Need,
    pub granted: Invariance,
    /// The backend's versioned implementation (`OpInfo::implementation`).
    pub implementation: String,
    /// The entries its launches name, each with the dynamic shared memory
    /// the host configures it for at load (`OpInfo::symbols`).
    pub symbols: Vec<(String, u32)>,
    /// The backend workspace its launches take (`OpInfo::workspace`).
    pub workspace: u64,
}

/// The asks of one create, recorded as each op prepares.
#[derive(Default)]
pub(crate) struct Asks(RefCell<Vec<Asked>>);

impl Asks {
    /// The asks recorded, in the order the ops prepared.
    pub(crate) fn asked(self) -> Vec<Asked> {
        self.0.into_inner()
    }

    /// The op `prepare` makes for `need`, its ask recorded as `what`.
    pub(crate) fn ask<B>(
        &self,
        what: impl Into<String>,
        need: Need,
        prepare: impl FnOnce(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Prepared<B>, Refused> {
        let prepared = prepare(need)?;
        self.record(what, need, prepared.info());
        Ok(prepared)
    }

    /// Records the ask `what` of an op prepared for `need`, which `info`
    /// describes.
    pub(crate) fn record(&self, what: impl Into<String>, need: Need, info: &OpInfo) {
        self.0.borrow_mut().push(Asked {
            what: what.into(),
            need,
            granted: info.guarantees,
            implementation: info.implementation.clone(),
            symbols: info.symbols.clone(),
            workspace: info.workspace,
        });
    }
}

/// A launch with whether it waits for its peer ranks, as the op that made
/// it states (`Launch::collective`).
pub struct Lowered {
    pub launch: PreparedLaunch,
    pub collective: bool,
}

/// Spans bound for one stage's launches, as role views.
#[derive(Default)]
pub(crate) struct Bind(BufferBindings);

#[cfg(test)]
mod independent_tests;

impl Bind {
    /// Bounded row-local launches with no workspace or persistent grid.
    pub(crate) fn launch_independent<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        if op.info().workspace != 0
            || !op
                .info()
                .guarantees
                .provides(Invariance::any_rows(counts.rows))
        {
            return Err(Refusal("overlap needs independent rows without workspace"));
        }
        let launches = op.launches(buffers, View::EMPTY, counts)?;
        if launches.is_empty()
            || launches.iter().any(|launch| {
                launch.persistent
                    || launch.collective
                    || launch.plan.workspace_bytes != 0
                    || launch.plan.cluster != [1, 1, 1]
            })
        {
            return Err(Refusal("overlap needs bounded local producers"));
        }
        Ok(launches
            .into_iter()
            .map(|launch| self.lowered(launch))
            .collect())
    }

    pub(crate) fn initialize<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
    ) -> Result<Vec<Lowered>, Refusal> {
        Ok(op
            .initialize(buffers)?
            .into_iter()
            .map(|launch| self.lowered(launch))
            .collect())
    }
    pub(crate) fn view(&mut self, span: DeviceBufferSpanV1) -> View {
        View {
            handle: self.0.bind(span),
            bytes: span.length,
        }
    }

    /// `op`'s launches over `buffers` as the pipeline's launches, every one
    /// with these bindings and the facts the op states of it, for an op
    /// that takes no backend workspace.
    pub(crate) fn launch<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lowered_op(op.as_ref(), buffers, None, counts)
    }

    /// `op`'s launches over `buffers`, its backend workspace, if it takes
    /// one, bound to `workspace`: the rank's region every op's workspace
    /// fits (`Ops::workspace`).
    pub(crate) fn launch_within<B>(
        self,
        op: &Prepared<B>,
        buffers: &B,
        workspace: DeviceBufferSpanV1,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lowered_op(op.as_ref(), buffers, Some(workspace), counts)
    }

    /// An expert projection's launches, its backend workspace bound to
    /// `workspace`, the rank's region every op's workspace fits
    /// (`Ops::workspace`).
    pub(crate) fn launch_experts(
        self,
        op: &dyn kernel_core::moe_experts::PreparedExperts,
        buffers: &kernel_core::moe_experts::GatedMlpBuffers<View>,
        workspace: DeviceBufferSpanV1,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        self.lowered_op(op, buffers, Some(workspace), counts)
    }

    fn lowered_op<B>(
        mut self,
        op: &dyn PreparedOp<B>,
        buffers: &B,
        workspace: Option<DeviceBufferSpanV1>,
        counts: Counts,
    ) -> Result<Vec<Lowered>, Refusal> {
        let bytes = op.info().workspace;
        let workspace = match (bytes, workspace) {
            (0, _) => View::EMPTY,
            (_, Some(span)) if span.length >= bytes => {
                self.view(DeviceBufferSpanV1::new(span.buffer, span.offset, bytes))
            }
            _ => return Err(Refusal("DeepSeek V4.1 binds no workspace this large")),
        };
        let launches = op.launches(buffers, workspace, counts)?;
        Ok(launches
            .into_iter()
            .map(|launch| self.lowered(launch))
            .collect())
    }

    fn lowered(&self, launch: Launch) -> Lowered {
        Lowered {
            collective: launch.collective,
            launch: PreparedLaunch {
                symbol: launch.symbol,
                rows: launch.rows,
                plan: launch.plan,
                arguments: launch.arguments,
                bindings: self.0.clone(),
            },
        }
    }
}
