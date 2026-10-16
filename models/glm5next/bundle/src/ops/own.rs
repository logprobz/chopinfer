//! GLM's own row ops (`model_glm5next_contract::own_ops`), prepared once at
//! create on the target's own kernels: the embedding's widening before its
//! all-reduce, the drafter's row inputs, input halves, residual add and
//! head rows, an eager prefill's gather of the rows that
//! need logits, and the KDA layers' stacked input projections of the
//! decode and verify graphs. Each is asked at create for the exactness GLM
//! relies on of it (`OwnRowOps::prepare`).

use std::sync::Arc;

use kernel_core::allocation::Traversal;
use kernel_core::backend::{Counts, Need, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::precision::OperationUse;
use model_glm5next_contract::contract::{PREFILL_TOKENS, Shape};
use model_glm5next_contract::own_ops::OwnOps;
use model_glm5next_contract::own_ops::draft::{DraftGather, DraftGatherBuffers};
use model_glm5next_contract::own_ops::rows::{
    EhInput, EhInputBuffers, GatherRows, GatherRowsBuffers, ResidualAdd, ResidualAddBuffers,
    WidenRows, WidenRowsBuffers,
};
use model_glm5next_contract::own_ops::stacked::StackedProjectionBuffers;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use super::{Asks, Bind, Lowered, exact};
use crate::target::Target;

/// A KDA layer's stacked input projections at one decode or verify use:
/// the prepared op, and each part's projection and first column in the rows
/// of the buffer its consumer reads, in part order.
pub struct Stacked {
    pub usage: OperationUse,
    pub parts: Vec<(&'static str, u32)>,
    pub op: Prepared<StackedProjectionBuffers<View>>,
}

/// The KDA layers' stacked input projections of an instance: the
/// projections one stacked launch runs, in part order, and the prepared ops
/// at each decode and verify use a stacked entry runs.
#[derive(Default)]
pub struct StackedOps {
    pub projections: &'static [&'static str],
    pub ops: Vec<Stacked>,
}

/// The own row ops of an instance, prepared for a prefill's rows, the most
/// any phase runs.
pub(crate) struct OwnRowOps {
    /// The own kernels, which create and commit prepare their state copies
    /// and admissions on.
    kernels: Arc<dyn OwnOps>,
    widen_rows: Prepared<WidenRowsBuffers<View>>,
    residual_add: Prepared<ResidualAddBuffers<View>>,
    gather_rows: Prepared<GatherRowsBuffers<View>>,
    draft_gather: Prepared<DraftGatherBuffers<View>>,
    /// The drafter's input halves at each use of its input projection.
    eh_input: Vec<(OperationUse, Prepared<EhInputBuffers<View>>)>,
    /// The stacked KDA input projections at each use that runs them.
    stacked: StackedOps,
}

/// `spans` bound as `op`'s role views, and its launches over them.
pub(crate) fn launch<S>(
    op: &Prepared<S::Mapped<View>>,
    spans: S,
    counts: Counts,
) -> Result<Vec<Lowered>, Refusal>
where
    S: Traversal<Span>,
{
    let mut bind = Bind::default();
    let buffers = spans.map(|span| bind.view(span));
    bind.launch(op, &buffers, counts)
}

impl OwnRowOps {
    /// Prepares `shape`'s own row ops on `own`, each asked through `asks`
    /// (`exact`): the drafter's input halves at each of `eh_uses`, with the
    /// stacked projections `target` prepares. The embedding's widening runs
    /// in every phase and asks what every phase needs, the gather of the
    /// rows that need logits a prefill's chunk split, and each verify
    /// bucket's stacked projections one-row bits up to the bucket's rows
    /// (decode's ask nothing, the one-row reference they are). The
    /// drafter's own rows (its residual add, row-input gather and input
    /// halves) only choose drafts, which the target's verify accepts or
    /// rejects, so they ask nothing.
    pub(crate) fn prepare(
        kernels: &Arc<dyn OwnOps>,
        shape: Shape,
        eh_uses: Vec<OperationUse>,
        target: &dyn Target,
        asks: &Asks,
    ) -> Result<OwnRowOps, Refused> {
        let own = kernels.as_ref();
        let rows = Need::rows(PREFILL_TOKENS);
        let eh_input = eh_uses
            .into_iter()
            .map(|usage| {
                let what = format!("own.eh_input {:?} {}", usage.phase, usage.capacity);
                let op = asks.ask(what, Need::rows(usage.capacity), |need| {
                    own.eh_input(&EhInput::of(shape), usage, need)
                })?;
                Ok((usage, op))
            })
            .collect::<Result<_, Refused>>()?;
        let every_phase = exact::need(PREFILL_TOKENS, exact::every_phase(exact::verify_rows()));
        let widen_rows = asks.ask("own.widen", every_phase, |need| {
            own.widen_rows(&WidenRows::of(shape), need)
        })?;
        let residual_add = asks.ask("own.residual_add", rows, |need| {
            own.residual_add(&ResidualAdd::of(shape), need)
        })?;
        let gather_rows = asks.ask(
            "own.gather_rows",
            exact::need(PREFILL_TOKENS, exact::PREFILL),
            |need| own.gather_rows(&GatherRows::of(shape), need),
        )?;
        let draft_gather = asks.ask("own.draft_gather", rows, |need| {
            own.draft_gather(&DraftGather::of(shape), need)
        })?;
        let stacked_need = |usage: OperationUse| {
            Asks::asked_need(exact::need(
                usage.capacity,
                exact::of_phase(usage.phase, usage.capacity),
            ))
        };
        let stacked = target.stacked(own, &stacked_need)?;
        for op in &stacked.ops {
            let what = format!("own.stacked {:?} {}", op.usage.phase, op.usage.capacity);
            asks.record(what, stacked_need(op.usage), op.op.info());
        }
        Ok(OwnRowOps {
            kernels: kernels.clone(),
            widen_rows,
            residual_add,
            gather_rows,
            draft_gather,
            eh_input,
            stacked,
        })
    }

    /// The own kernels the ops were prepared on.
    pub(crate) fn kernels(&self) -> &dyn OwnOps {
        self.kernels.as_ref()
    }

    /// Widens `rows` BF16 rows to F32, exactly.
    pub(crate) fn widen_rows(
        &self,
        rows: u32,
        spans: WidenRowsBuffers<Span>,
    ) -> Result<Vec<Lowered>, Refusal> {
        launch(&self.widen_rows, spans, Counts::rows(rows))
    }

    /// Adds `y` into `x` over `rows` rows.
    pub(crate) fn residual_add(
        &self,
        rows: u32,
        spans: ResidualAddBuffers<Span>,
    ) -> Result<Vec<Lowered>, Refusal> {
        launch(&self.residual_add, spans, Counts::rows(rows))
    }

    /// Gathers at most `rows` rows by index; with `grid` nonzero, over the
    /// row bucket holding `grid` rows (a prefill's gather of its scored
    /// rows).
    pub(crate) fn gather_rows(
        &self,
        rows: u32,
        grid: u32,
        spans: GatherRowsBuffers<Span>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let counts = Counts {
            padded_rows: grid,
            ..Counts::rows(rows)
        };
        launch(&self.gather_rows, spans, counts)
    }

    /// The drafter rows' hidden inputs and tokens of `rows` rows.
    pub(crate) fn draft_gather(
        &self,
        rows: u32,
        spans: DraftGatherBuffers<Span>,
    ) -> Result<Vec<Lowered>, Refusal> {
        launch(&self.draft_gather, spans, Counts::rows(rows))
    }

    /// The drafter's input halves of `rows` rows at its input projection's
    /// use `usage`.
    pub(crate) fn eh_input(
        &self,
        usage: OperationUse,
        rows: u32,
        spans: EhInputBuffers<Span>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .eh_input
            .iter()
            .find(|(of, _)| *of == usage)
            .ok_or(Refusal("the drafter's input halves serve no such use"))?;
        launch(op, spans, Counts::rows(rows))
    }

    /// The stacked KDA input projections at `usage`, if a stacked entry runs
    /// them.
    pub(crate) fn stacked(&self, usage: OperationUse) -> Option<&Stacked> {
        self.stacked
            .ops
            .iter()
            .find(|stacked| stacked.usage == usage)
    }

    /// The projections a stacked launch runs, in part order; none where no
    /// stacked entry runs.
    pub(crate) fn stacked_projections(&self) -> &'static [&'static str] {
        self.stacked.projections
    }

    /// Whether the stacked projections launch `symbol`: a use the precision
    /// component bound to it runs only in their launch.
    pub(crate) fn is_stacked(&self, symbol: &str) -> bool {
        self.stacked.ops.iter().any(|stacked| {
            stacked
                .op
                .info()
                .symbols
                .iter()
                .any(|(launched, _)| launched == symbol)
        })
    }
}

impl Stacked {
    /// The stacked projections of `rows` rows over `spans`, with the
    /// projections' scratch partial as the op's workspace.
    pub(crate) fn launch(
        &self,
        rows: u32,
        spans: StackedProjectionBuffers<Span>,
        workspace: Span,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = spans.map(|span| bind.view(span));
        let workspace = bind.view(workspace);
        bind.launch_within(&self.op, &buffers, workspace, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::properties::Invariance;
    use model_glm5next_contract::own_ops::stacked::StackedProjection;

    use super::*;
    use crate::sm120::own_ops::Sm120Own;

    /// GLM's own kernels as kernels that declare nothing: each op prepares
    /// for its rows alone and is refused where its need asks exactness, as
    /// a backend refuses what its declarations do not keep.
    struct Undeclared(Sm120Own);

    fn undeclared<B>(
        need: Need,
        prepare: impl FnOnce(Need) -> Result<Prepared<B>, Refused>,
    ) -> Result<Prepared<B>, Refused> {
        if need.exact != Invariance::NONE {
            return Err(Refused::Inexact {
                needed: need.exact,
                guaranteed: Invariance::NONE,
            });
        }
        prepare(need)
    }

    impl OwnOps for Undeclared {
        fn name(&self) -> &str {
            "undeclared"
        }

        fn widen_rows(
            &self,
            op: &WidenRows,
            need: Need,
        ) -> Result<Prepared<WidenRowsBuffers<View>>, Refused> {
            undeclared(need, |need| self.0.widen_rows(op, need))
        }

        fn residual_add(
            &self,
            op: &ResidualAdd,
            need: Need,
        ) -> Result<Prepared<ResidualAddBuffers<View>>, Refused> {
            undeclared(need, |need| self.0.residual_add(op, need))
        }

        fn gather_rows(
            &self,
            op: &GatherRows,
            need: Need,
        ) -> Result<Prepared<GatherRowsBuffers<View>>, Refused> {
            undeclared(need, |need| self.0.gather_rows(op, need))
        }

        fn eh_input(
            &self,
            op: &EhInput,
            usage: OperationUse,
            need: Need,
        ) -> Result<Prepared<EhInputBuffers<View>>, Refused> {
            undeclared(need, |need| self.0.eh_input(op, usage, need))
        }

        fn draft_gather(
            &self,
            op: &DraftGather,
            need: Need,
        ) -> Result<Prepared<DraftGatherBuffers<View>>, Refused> {
            undeclared(need, |need| self.0.draft_gather(op, need))
        }

        fn stacked_projection(
            &self,
            op: &StackedProjection,
            usage: OperationUse,
            need: Need,
        ) -> Result<Prepared<StackedProjectionBuffers<View>>, Refused> {
            undeclared(need, |need| self.0.stacked_projection(op, usage, need))
        }
    }

    /// The own row ops ask at create what GLM relies on of them, and
    /// create refuses own kernels that do not keep it.
    #[test]
    fn own_row_ops_ask_at_create_and_refuse_what_their_kernels_do_not_keep() {
        let shape = Shape::TP4;
        let target = Sm120Own::new(shape);
        let prepare = |kernels: Arc<dyn OwnOps>, asks: &Asks| {
            OwnRowOps::prepare(
                &kernels,
                shape,
                crate::lower::eh_uses().unwrap(),
                &target,
                asks,
            )
        };
        let asks = Asks::default();
        prepare(Arc::new(target), &asks).unwrap();
        let asked = asks.asked();
        let need = |what: &str| {
            let asked = asked.iter().find(|asked| asked.what == what);
            asked.unwrap_or_else(|| panic!("no ask {what}")).need.exact
        };
        let verify = exact::verify_rows();
        assert_eq!(need("own.widen"), exact::every_phase(verify));
        assert_eq!(need("own.gather_rows"), exact::PREFILL);
        assert_eq!(need("own.residual_add"), Invariance::NONE);
        assert_eq!(need("own.stacked Verify 8"), exact::verify(8));
        assert_eq!(need("own.stacked Decode 8"), Invariance::NONE);
        assert!(
            asked
                .iter()
                .all(|asked| asked.granted.provides(asked.need.exact))
        );
        let refused = prepare(Arc::new(Undeclared(target)), &Asks::default());
        assert!(
            matches!(refused, Err(Refused::Inexact { .. })),
            "{:?}",
            refused.err()
        );
    }
}
