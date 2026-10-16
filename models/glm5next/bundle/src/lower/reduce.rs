//! Lowering of the all-reduces through the instance's prepared ops
//! (`crate::ops::reduce`): each attention and dense MLP sublayer's F32 rank
//! partials reduced as BF16 partials (every rank's partial rounded once to
//! BF16, the BF16 values summed in rank order from F32 zero and the sum
//! rounded once to BF16) into the rows the Sinkhorn expansion reads, the
//! comm epilogue that composes that all-reduce with the expansion and the
//! next mixer and input norm, and each MoE sublayer's F32 partials summed
//! in F32 and rounded once (the rounded all-reduce), which no comm epilogue
//! fuses. The backend chooses each row count's entry or composition at
//! prepare; a step takes the ops of create's admission outcome and binds
//! its spans. The embedding's BF16 shard rows are widened into the partial
//! first.

use model_glm5next_contract::own_ops::rows::WidenRowsBuffers;

use super::Lowering;
use crate::buffers::Buffer;
use crate::ops::Lowered;
use crate::ops::reduce::{Admission, EpilogueViews, PeerViews, ReduceViews};
use crate::program::{Phase, Stage, Sublayer, moe_sublayer};

impl Lowering<'_> {
    /// The admission outcome a step's all-reduces take: the one create
    /// left.
    fn admission(&self) -> Admission {
        let comm = &self.buffers.comm;
        Admission {
            low_latency: comm.low_latency,
            phased: comm.phased,
        }
    }

    /// The views of a BF16-partials all-reduce (or with `rounded`, of the
    /// MoE sublayers' rounded all-reduce) under `admission` whose state
    /// regions are `state`: the partial, the rounded rows and the live count
    /// of this lowering's rows, and every rank's rounded rows and comm state
    /// regions.
    fn reduce_views(
        &self,
        state: &[u64],
        admission: Admission,
        rounded: bool,
    ) -> Result<ReduceViews, String> {
        let peers = self
            .peers
            .iter()
            .map(|peer| {
                Ok(PeerViews {
                    output: peer.reduced,
                    state: match rounded {
                        true => peer.rounded_regions(admission, state.len())?,
                        false => peer.all_reduce_regions(admission, state.len())?,
                    },
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(ReduceViews {
            input: self.scratch(Buffer::Partial),
            output: self.scratch(Buffer::Reduced),
            live_rows: Some(self.inputs.live_tokens),
            peers,
        })
    }

    /// The all-reduce of the sublayer's partials, rounded once into the
    /// rows the expansion reads.
    pub(super) fn reduce(&self, phase: Phase) -> Result<Vec<Lowered>, String> {
        let admission = self.admission();
        let ops = &self.ops.reduce;
        let state = ops
            .reduce_state(admission, self.rank)
            .map_err(|refusal| refusal.to_string())?;
        let views = self.reduce_views(state, admission, false)?;
        ops.reduce(admission, self.rank, phase.rows(), &views)
            .map_err(|refusal| format!("all-reduce: {refusal}"))
    }

    /// The all-reduce at the end of `(layer, sublayer)`: a MoE sublayer's
    /// F32 partials summed in F32 and rounded once, any other's as BF16
    /// partials ([`Self::reduce`]), into the rows the expansion reads.
    pub(super) fn sublayer_reduce(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Vec<Lowered>, String> {
        if !moe_sublayer(self.shape, layer, sublayer) {
            return self.reduce(phase);
        }
        let admission = self.admission();
        let ops = &self.ops.reduce;
        let state = ops
            .rounded_state(admission, self.rank)
            .map_err(|refusal| refusal.to_string())?;
        let views = self.reduce_views(state, admission, true)?;
        ops.rounded(admission, self.rank, phase.rows(), &views)
            .map_err(|refusal| format!("MoE all-reduce: {refusal}"))
    }

    /// The embedding's shard rows widened into the partial (an own op),
    /// all-reduced and rounded: exact, as one rank holds each row.
    pub(super) fn embedding_reduce(&self, phase: Phase) -> Result<Vec<Lowered>, String> {
        let spans = WidenRowsBuffers {
            input: self.scratch(Buffer::Collapsed),
            output: self.scratch(Buffer::Partial),
            live_rows: self.inputs.live_tokens,
        };
        let widen = self.own_launch("the embedding's widening", || {
            self.ops.own.widen_rows(phase.rows(), spans)
        })?;
        let mut launches = vec![widen];
        launches.extend(self.reduce(phase)?);
        Ok(launches)
    }

    /// The comm epilogue at the end of `ended`: the all-reduce of its
    /// partials, then the expansion from the streams it read into the
    /// other buffer and `next`'s mix-collapse and input norm, as the op
    /// launches them.
    fn reduce_expand_mix_norm(
        &self,
        phase: Phase,
        (layer, sublayer): (u32, Sublayer),
        (next_layer, next_sublayer): (u32, Sublayer),
    ) -> Result<Vec<Lowered>, String> {
        let admission = self.admission();
        let ops = &self.ops.reduce;
        let state = ops
            .epilogue_state(admission, self.rank)
            .map_err(|refusal| refusal.to_string())?;
        let (input, output) = super::sinkhorn::streams(layer, sublayer);
        let prefix = super::sinkhorn::mixer_prefix(next_sublayer);
        let view = |name: &str| self.buffers.view(&super::layer_view(next_layer, name));
        let views = EpilogueViews {
            reduce: self.reduce_views(state, admission, false)?,
            streams: self.scratch(input),
            reduced: self.scratch(Buffer::Reduced),
            expanded: self.scratch(output),
            mixer: [
                view(&format!("{prefix}_fn"))?,
                view(&format!("{prefix}_base"))?,
                view(&format!("{prefix}_scale"))?,
            ],
            post: self.scratch(Buffer::Post),
            comb: self.scratch(Buffer::Comb),
            collapsed: self.scratch(Buffer::Collapsed),
            norm_weight: view(super::rows::input_norm_weight(next_sublayer))?,
            normed: self.scratch(Buffer::Normed),
            partials: self.scratch(Buffer::MixPartials),
            tickets: self.scratch(Buffer::MixTickets),
            live_rows: self.inputs.live_tokens,
        };
        ops.epilogue(admission, self.rank, phase.rows(), &views)
            .map_err(|refusal| format!("comm epilogue: {refusal}"))
    }

    /// The sublayer after `(layer, sublayer)` in the target's program,
    /// whose mixer and input norm a comm epilogue at the end of
    /// `(layer, sublayer)` runs: none after the last layer's MLP (the head
    /// collapse follows), in the drafter's layer or after a MoE sublayer
    /// (whose rounded all-reduce of F32 partials no epilogue fuses).
    fn next_sublayer(&self, layer: u32, sublayer: Sublayer) -> Option<(u32, Sublayer)> {
        let layers = self.shape.geometry.layers();
        match sublayer {
            _ if layer >= layers || moe_sublayer(self.shape, layer, sublayer) => None,
            Sublayer::Attention => Some((layer, Sublayer::Mlp)),
            Sublayer::Mlp => (layer + 1 < layers).then_some((layer + 1, Sublayer::Attention)),
        }
    }

    /// With the comm epilogue, each sublayer boundary's stages: the reduce
    /// lowers to the fused launches and the expand, the next mixer and the
    /// next input norm to none. Other stages lower as without it (`None`).
    pub(super) fn comm_epilogue_stage(
        &self,
        phase: Phase,
        stage: Stage,
    ) -> Result<Option<Vec<Lowered>>, String> {
        // The sublayer before `(layer, sublayer)`, whose epilogue ran its
        // mixer and input norm.
        let fused_into = |layer: u32, sublayer: Sublayer| match sublayer {
            Sublayer::Mlp => self.next_sublayer(layer, Sublayer::Attention).is_some(),
            Sublayer::Attention => {
                layer > 0 && self.next_sublayer(layer - 1, Sublayer::Mlp).is_some()
            }
        };
        Ok(match stage {
            Stage::Reduce(layer, sublayer) => match self.next_sublayer(layer, sublayer) {
                Some(next) => Some(self.reduce_expand_mix_norm(phase, (layer, sublayer), next)?),
                None => None,
            },
            Stage::HcExpand(layer, sublayer) if self.next_sublayer(layer, sublayer).is_some() => {
                Some(Vec::new())
            }
            Stage::HcMix(layer, sublayer) | Stage::InputNorm(layer, sublayer)
                if fused_into(layer, sublayer) =>
            {
                Some(Vec::new())
            }
            _ => None,
        })
    }
}
