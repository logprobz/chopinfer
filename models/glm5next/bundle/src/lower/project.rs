//! Lowering of the projection API's stages (`bundle_runtime::projection`):
//! each lowered projection's GEMM (with its layout copy) over the rank's
//! store, from the buffer its producer writes into the one its consumer
//! reads; the indexer's pooling gate through the target (`Target::pool_gate`); the
//! drafter's input halves copied side by side for its input projection;
//! and a KDA layer's projections of its normed input as one stacked launch
//! in the decode and verify graphs.

use bundle_runtime::projection::{self, Input};
use kernel_core::backend::Refused;
use kernel_core::precision::{ExecutionPhase, OperationUse};
use model_glm5next_contract::own_ops::rows::EhInputBuffers;
use model_glm5next_contract::own_ops::stacked::StackedProjectionBuffers;
use model_glm5next_contract::precision;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view};
use crate::buffers::Buffer;
use crate::ops::{Lowered, one, own};
use crate::program::{Phase, draft_rows};
use crate::target::PoolGateSpans;

/// Projection `id`'s declared use that serves `rows` rows of `phase`: its
/// prefill use, or its smallest decode or verify bucket holding them.
pub(crate) fn usage(id: &str, phase: Phase, rows: u32) -> Result<OperationUse, String> {
    let wanted = match phase {
        Phase::Prefill { .. } => ExecutionPhase::Prefill,
        Phase::Decode(_) => ExecutionPhase::Decode,
        Phase::Verify { .. } => ExecutionPhase::Verify,
    };
    precision::uses(id)?
        .into_iter()
        .map(|(usage, _)| usage)
        .filter(|usage| usage.phase == wanted && usage.capacity >= rows)
        .min_by_key(|usage| usage.capacity)
        .ok_or_else(|| format!("{id} declares no use of {rows} rows in {phase:?}"))
}

/// What lowered projection `id` reads, writes and adds: its activation,
/// its output (through its layout copy, where it has one) and its
/// residual.
fn io(id: &str) -> Option<(Buffer, Buffer, Option<Buffer>)> {
    use Buffer::*;
    Some(match id {
        "linear.q" | "linear.k" | "linear.v" => (Normed, KdaQkv, None),
        "linear.forget-a" => (Normed, KdaForgetDown, None),
        "linear.forget-b" => (KdaForgetDown, KdaGate, None),
        "linear.beta" => (Normed, KdaBeta, None),
        "linear.gate-a" => (Normed, KdaGateDown, None),
        "linear.gate-b" => (KdaGateDown, KdaOutputGate, None),
        "linear.output" => (KdaNorm, Partial, None),
        "latent.q-a" | "mtp.latent.q-a" => (Normed, QueryDown, None),
        "latent.q-b" | "mtp.latent.q-b" => (QueryNormed, QueryUp, None),
        "latent.kv-a" | "mtp.latent.kv-a" => (Normed, LatentRaw, None),
        "latent.indexer.q" | "mtp.latent.indexer.q" => (QueryNormed, IndexQuery, None),
        "latent.indexer.k" | "mtp.latent.indexer.k" => (Normed, IndexKey, None),
        "latent.indexer.weights" | "mtp.latent.indexer.weights" => (Normed, IndexWeights, None),
        "latent.output" | "mtp.latent.output" => (ValueOut, Partial, None),
        "dense.down" => (DenseIntermediate, Partial, None),
        // The shared expert's down projection adds the routed sum.
        "moe.shared.down" | "mtp.moe.shared.down" => (DenseIntermediate, Partial, Some(MlpOut)),
        "mtp.eh" => (DraftEhInput, DraftHidden, None),
        // The router's padded scores, compacted after the GEMM.
        "moe.router" | "mtp.moe.router" => (Normed, RouterScores, None),
        _ => return None,
    })
}

impl Lowering<'_> {
    /// The launches of projection `id`'s use for `rows` rows of `phase`, its
    /// instance at `layer` (none for the head), reading `activation` and
    /// `live_rows`, adding `residual`, into `output`, over the rank's
    /// projection scratch, or its side scratch when the launch runs on the
    /// side stream (`side`, `crate::shared_overlap`).
    #[allow(clippy::too_many_arguments)]
    fn bound_projection(
        &self,
        phase: Phase,
        id: &str,
        layer: Option<u32>,
        rows: u32,
        activation: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
        residual: Option<DeviceBufferSpanV1>,
        output: DeviceBufferSpanV1,
        side: bool,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(projections) = self.projections else {
            return Ok(None);
        };
        let operation = projections
            .instances
            .get(&(id.to_owned(), layer.map(u64::from)))
            .ok_or_else(|| format!("the precision component has no {id} at {layer:?}"))?;
        let usage = usage(id, phase, rows)?;
        let handle = projections
            .store
            .handle(operation, usage)
            .map_err(|r| format!("{operation}: {r}"))?;
        // A use bound to its part of a stacked entry runs only in that
        // entry's launch (`stacked_kda_inputs`), never as its GEMM and copy.
        if let Some(stage) = handle
            .identity()
            .stages
            .iter()
            .find(|stage| self.ops.own.is_stacked(stage))
        {
            return Err(format!(
                "{operation} is bound to the stacked stage {stage}, not its GEMM"
            ));
        }
        projection::launch(
            handle,
            projection::Phase {
                usage,
                rows,
                padded_rows: rows,
            },
            Input {
                activation,
                live_rows,
                routing: None,
                residual,
                row_offset: 0,
                shard_rows: 0,
                side: &[],
                scalars: &[],
            },
            output,
            &if side {
                projections.side_scratch
            } else {
                projections.scratch
            }
            .scratch(),
        )
        .map(|launches| Some(launches.into_iter().map(Lowered::local).collect()))
        .map_err(|r| format!("{operation}: {r}"))
    }

    /// A layer's projection `id` over the phase's rows.
    pub(super) fn project(
        &self,
        phase: Phase,
        layer: u32,
        id: &str,
    ) -> Result<Option<Vec<Lowered>>, String> {
        match id {
            "latent.indexer.pool-gate" => return self.pool_gate(phase, layer, "latent.indexer.k"),
            "mtp.latent.indexer.pool-gate" => {
                return self.pool_gate(phase, layer, "mtp.latent.indexer.k");
            }
            _ => {}
        }
        let Some((input, mut output, mut residual)) = io(id) else {
            return Ok(None);
        };
        // In the decode and verify graphs the shared expert's down
        // projection runs beside the routed experts, over its own scratch:
        // it adds zeros into the MLP output, which the routed sum then adds
        // into the partial (`crate::program`'s MoE sublayer).
        let side = residual == Some(Buffer::MlpOut) && !matches!(phase, Phase::Prefill { .. });
        if side {
            (output, residual) = (Buffer::MlpOut, Some(Buffer::ZeroRows));
            // The zero rows cover the widest decode or verify step only.
            let rows = u64::from(phase.rows()) * u64::from(self.shape.profile.hidden) * 4;
            if self.scratch(Buffer::ZeroRows).length < rows {
                return Err(format!(
                    "{id}: {phase:?} adds more rows than the zero rows hold"
                ));
            }
        }
        let Some(mut launches) = self.bound_projection(
            phase,
            id,
            Some(layer),
            phase.rows(),
            self.scratch(input),
            self.inputs.live_tokens,
            residual.map(|buffer| self.scratch(buffer)),
            self.scratch(output),
            side,
        )?
        else {
            return Ok(None);
        };
        // The router's GEMM writes its weight's 320 padded rows' scores; the
        // route reads every expert's 288.
        if output == Buffer::RouterScores {
            launches.push(self.compact_scores(phase)?);
        }
        Ok(Some(launches))
    }

    /// The stacked projections that run projection `id` of a KDA layer in
    /// `phase`: a decode or verify phase's bucket of at most 32 rows on an
    /// instance whose projections lower (`own::StackedOps`).
    pub(super) fn stacked_ops(&self, phase: Phase, id: &str) -> Option<&own::Stacked> {
        let projections = self.ops.own.stacked_projections();
        if self.projections.is_none()
            || matches!(phase, Phase::Prefill { .. })
            || !projections.contains(&id)
        {
            return None;
        }
        let used = usage(projections.first()?, phase, phase.rows()).ok()?;
        self.ops.own.stacked(used)
    }

    /// Layer `layer`'s KDA projections of its normed input in one launch of
    /// their stacked projections: each part over its projection's weights
    /// at the phase's use, into the buffer its consumer reads from the
    /// column its layout copy would place its rows at.
    pub(super) fn stacked_kda_inputs(&self, phase: Phase, layer: u32) -> Result<Lowered, String> {
        let first = self.ops.own.stacked_projections().first().copied();
        let (Some(projections), Some(stacked)) = (
            self.projections,
            first.and_then(|first| self.stacked_ops(phase, first)),
        ) else {
            return Err(format!("no stacked KDA input projections in {phase:?}"));
        };
        let rows = phase.rows();
        let used = stacked.usage;
        let symbols = &stacked.op.info().symbols;
        let mut weights = Vec::with_capacity(stacked.parts.len());
        let mut outputs = Vec::with_capacity(stacked.parts.len());
        for &(id, column) in &stacked.parts {
            if usage(id, phase, rows)? != used {
                return Err(format!("{id} runs another use than its stacked part"));
            }
            let operation = projections
                .instances
                .get(&(id.to_owned(), Some(u64::from(layer))))
                .ok_or_else(|| format!("the precision component has no {id} at {layer}"))?;
            let handle = projections
                .store
                .handle(operation, used)
                .map_err(|r| format!("{operation}: {r}"))?;
            // The component bound this use to the stacked stage the op
            // launches.
            if !handle
                .identity()
                .stages
                .iter()
                .eq(symbols.iter().map(|(symbol, _)| symbol))
            {
                return Err(format!(
                    "{operation} at {used:?} is bound to {:?}, not its stacked stage",
                    handle.identity().stages
                ));
            }
            weights.push(handle.weights().codes);
            let Some((Buffer::Normed, output, None)) = io(id) else {
                return Err(format!("{id} does not read the normed input"));
            };
            let span = self.scratch(output);
            let offset = u64::from(column) * 2;
            if span.length < offset {
                return Err(format!("{id}'s output starts past its buffer"));
            }
            outputs.push(DeviceBufferSpanV1::new(
                span.buffer,
                span.offset + offset,
                span.length - offset,
            ));
        }
        let scratch = projections.scratch.scratch();
        let spans = StackedProjectionBuffers {
            x: self.scratch(Buffer::Normed),
            weights,
            outputs,
            tickets: scratch.semaphores,
        };
        one(stacked
            .launch(rows, spans, scratch.partial)
            .map_err(|refusal| format!("the stacked KDA input projections: {refusal}"))?)
    }

    /// The indexer's pooling gate logits `[rows, indexer_dim]` of layer
    /// `layer` from the sublayer input: a BF16 `[indexer_dim, hidden]` side
    /// tensor of the gated pooling, not a declared projection, so the target
    /// launches it (`Target::pool_gate`) over the entry of the same shape
    /// the layer's key projection `key` uses at the phase's use, with that
    /// use's scratch.
    fn pool_gate(
        &self,
        phase: Phase,
        layer: u32,
        key: &str,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(projections) = self.projections else {
            return Ok(None);
        };
        let rows = phase.rows();
        let weight = self.buffers.view(&layer_view(
            layer,
            "self_attn.indexer.index_kpool_compress_gate",
        ))?;
        let scratch = projections.scratch.scratch();
        let spans = PoolGateSpans {
            weight,
            activation: self.scratch(Buffer::Normed),
            output: self.scratch(Buffer::IndexGate),
            partial: scratch.partial,
            semaphores: scratch.semaphores,
        };
        let launch = self
            .ops
            .target
            .pool_gate(key, usage(key, phase, rows)?, rows, spans)?;
        Ok(Some(vec![Lowered::local(launch)]))
    }

    /// The vocabulary-split head's F32 logits of the rank's shard: a
    /// prefill's gathered logits rows (none without them), a decode or
    /// verify phase's rows, or a drafter run's draft rows (`draft_rows`).
    pub(super) fn head(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let (rows, activation, live_rows) = match (self.draft, phase) {
            (Some(_), _) => (
                draft_rows(phase),
                self.scratch(Buffer::HeadRows),
                self.scratch(Buffer::DraftCount),
            ),
            (None, Phase::Prefill { .. }) if self.head_rows == 0 => {
                return Ok(self.projections.map(|_| Vec::new()));
            }
            (None, Phase::Prefill { .. }) => (
                self.head_rows,
                self.scratch(Buffer::HeadRows),
                self.inputs.logit_rows,
            ),
            (None, _) => (
                phase.rows(),
                self.scratch(Buffer::Normed),
                self.inputs.live_tokens,
            ),
        };
        self.bound_projection(
            phase,
            "head",
            None,
            rows,
            activation,
            live_rows,
            None,
            self.scratch(Buffer::Logits),
            false,
        )
    }

    /// The drafter's input projection's input: the normalized embedding
    /// (`Normed`) and target hidden row (`Collapsed`) copied side by side
    /// into `[rows, 2 * hidden]`, at the projection's use for the phase.
    pub(super) fn eh_input(&self, phase: Phase) -> Result<Vec<Lowered>, String> {
        let rows = phase.rows();
        let spans = EhInputBuffers {
            embed: self.scratch(Buffer::Normed),
            hidden: self.scratch(Buffer::Collapsed),
            output: self.scratch(Buffer::DraftEhInput),
            live_rows: self.inputs.live_tokens,
        };
        self.ops
            .own
            .eh_input(usage("mtp.eh", phase, rows)?, rows, spans)
            .map_err(|refusal| format!("the drafter's input halves: {refusal}"))
    }
}

/// The uses of the drafter's input projection, at which its input halves
/// run.
pub(crate) fn eh_uses() -> Result<Vec<OperationUse>, Refused> {
    Ok(precision::uses("mtp.eh")
        .map_err(Refused::Rule)?
        .into_iter()
        .map(|(usage, _)| usage)
        .collect())
}
