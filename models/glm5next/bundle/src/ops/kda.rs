//! The KDA ops: each recurrent phase's short convolution and
//! per-channel-decay delta rule (`PerChannelDecayDeltaRuleQuery`), and the
//! gated norm of the rule's output (`RecurrentQuery::GatedRmsNorm`).
//!
//! Each op is the model's query at the capacity its served entry compiles
//! (`Served`): a verify step of up to `MAX_VERIFY_ROWS` rows per item runs
//! the verify op of the widest served rows per item. Launches carry the
//! phase's rows and its items (`Lowering::kda_counts`): a decode bucket's
//! one row per item, a verify step's rows over its items. Prefill ops plan
//! every launch for every served item (`plan_at_capacity`), whatever items
//! an eager prefill carries.

use kernel_core::backend::{Backend, Prepared, Refused, View};
use kernel_core::contract::{GateActivation, RecurrentPhase, RecurrentQuery, StaticQuery};
use kernel_core::per_channel_decay_delta_rule::{
    PerChannelDecayDeltaRuleQuery as Kda, PerChannelDecayOp,
};
use kernel_core::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers};
use model_glm5next_contract::contract::{DECODE_BATCH, PREFILL_TOKENS, Shape};

use super::{Asks, Served, exact};
use crate::program::Phase;

/// The KDA ops of an instance: by phase (prefill, decode, verify), the
/// convolution and the delta rule, and one gated norm for every phase.
pub(crate) struct KdaOps {
    conv: [Prepared<ConvBuffers<View>>; 3],
    delta: [Prepared<DeltaRuleBuffers<View>>; 3],
    pub(crate) norm: Prepared<GatedNormBuffers<View>>,
}

/// The index of `phase`'s ops.
fn index(phase: Phase) -> usize {
    match phase {
        Phase::Prefill { .. } => 0,
        Phase::Decode(_) => 1,
        Phase::Verify { .. } => 2,
    }
}

/// `phase`'s name in asks.
fn name(phase: Phase) -> &'static str {
    ["prefill", "decode", "verify"][index(phase)]
}

/// Whether an op compiled for `served` serves a step of `step`: the same
/// phase, and for verify at least the step's rows per item.
fn serves(served: RecurrentPhase, step: RecurrentPhase) -> bool {
    match (served, step) {
        (RecurrentPhase::Verify { max_rows }, RecurrentPhase::Verify { max_rows: rows }) => {
            rows <= max_rows
        }
        _ => served == step,
    }
}

/// The one served query `model` states at its served phase and capacity
/// that serves `phase`.
fn served(
    served: &Served,
    what: &str,
    phase: RecurrentPhase,
    model: impl Fn(RecurrentPhase, u32) -> Kda,
) -> Result<Kda, Refused> {
    let found = served.find(|query| match *query {
        StaticQuery::PerChannelDecayDeltaRule(
            query @ (Kda::ShortConv {
                phase: at,
                capacity,
                ..
            }
            | Kda::DeltaRule {
                phase: at,
                capacity,
                ..
            }),
        ) if query == model(at, capacity) && serves(at, phase) => Some(query),
        _ => None,
    });
    match found[..] {
        [query] => Ok(query),
        _ => Err(Refused::Rule(format!(
            "GLM 5.3 Flash's KDA {what} at {phase:?} has {} served queries",
            found.len()
        ))),
    }
}

impl KdaOps {
    pub(super) fn prepare(
        backend: &dyn Backend,
        served_queries: &Served,
        shape: Shape,
        asks: &Asks,
    ) -> Result<KdaOps, Refused> {
        let widest = crate::buffers::widest_step_rows();
        // An eager prefill carries every served item, a decode or verify
        // step at most every item.
        let phases = [
            (
                Phase::Prefill {
                    rows: PREFILL_TOKENS,
                },
                PREFILL_TOKENS,
            ),
            (Phase::Decode(DECODE_BATCH), DECODE_BATCH),
            (
                Phase::Verify {
                    rows: widest,
                    items: DECODE_BATCH,
                },
                widest,
            ),
        ];
        let op = |query, phase: Phase| PerChannelDecayOp {
            query,
            plan_at_capacity: matches!(phase, Phase::Prefill { .. }),
        };
        let conv = phases.map(|(phase, rows)| {
            let query = served(
                served_queries,
                "convolution",
                phase.recurrent(),
                |at, rows| shape.kda_conv(at, rows),
            )?;
            let need = exact::need(rows, exact::with_state(exact::of_phase(phase, rows)));
            asks.ask(format!("kda.conv {}", name(phase)), need, |need| {
                backend.short_conv(&op(query, phase), DECODE_BATCH, need)
            })
        });
        let delta = phases.map(|(phase, rows)| {
            let query = served(
                served_queries,
                "delta rule",
                phase.recurrent(),
                |at, rows| shape.kda_delta_rule(at, rows),
            )?;
            // The rule chunks at absolute multiples of the granule, and the
            // scheduler's chunks start there, so every cut is covered.
            let need = exact::need(rows, exact::with_state(exact::of_phase(phase, rows)));
            asks.ask(format!("kda.delta {}", name(phase)), need, |need| {
                backend.channel_decay_delta_rule(&op(query, phase), DECODE_BATCH, need)
            })
        });
        let p = shape.profile;
        let norms = served_queries.find(|query| match *query {
            StaticQuery::Recurrent(
                query @ RecurrentQuery::GatedRmsNorm {
                    width,
                    head_dim,
                    eps_bits,
                    gate: GateActivation::Sigmoid,
                    ..
                },
            ) if width == p.linear_heads * p.linear_head_dim
                && head_dim == p.linear_head_dim
                && eps_bits == shape.geometry.eps().to_bits() =>
            {
                Some(query)
            }
            _ => None,
        });
        let [norm] = norms[..] else {
            return Err(Refused::Rule(format!(
                "GLM 5.3 Flash's KDA gated norm has {} served queries",
                norms.len()
            )));
        };
        let every_phase = exact::need(PREFILL_TOKENS, exact::every_phase(exact::verify_rows()));
        let [conv_prefill, conv_decode, conv_verify] = conv;
        let [delta_prefill, delta_decode, delta_verify] = delta;
        Ok(KdaOps {
            conv: [conv_prefill?, conv_decode?, conv_verify?],
            delta: [delta_prefill?, delta_decode?, delta_verify?],
            norm: asks.ask("kda.norm", every_phase, |need| {
                backend.gated_rmsnorm(&norm, need)
            })?,
        })
    }

    /// The convolution of `phase`.
    pub(crate) fn conv(&self, phase: Phase) -> &Prepared<ConvBuffers<View>> {
        &self.conv[index(phase)]
    }

    /// The delta rule of `phase`.
    pub(crate) fn delta(&self, phase: Phase) -> &Prepared<DeltaRuleBuffers<View>> {
        &self.delta[index(phase)]
    }
}
