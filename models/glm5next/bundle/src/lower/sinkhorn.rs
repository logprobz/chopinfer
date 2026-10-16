//! Lowering of the Sinkhorn mixer through the instance's prepared ops
//! (`crate::ops::sinkhorn`).

use super::{Lowering, layer_view};
use crate::buffers::Buffer;
use crate::ops::{Bind, Lowered, one};
use crate::program::{Phase, Sublayer};
use kernel_core::backend::{Counts, View};
use kernel_core::sinkhorn_hyper_connection::{
    MixCollapseBuffers, MixCollapseNormBuffers, MixScratch, MixerWeights, StreamExpandBuffers,
    StreamHeadBuffers,
};

/// The sublayer's ordinal: the residual streams alternate between two
/// buffers, sublayer `s` reading `Streams{s % 2}` and expanding into the
/// other.
pub(crate) fn streams(layer: u32, sublayer: Sublayer) -> (Buffer, Buffer) {
    let ordinal = 2 * layer + u32::from(sublayer == Sublayer::Mlp);
    if ordinal.is_multiple_of(2) {
        (Buffer::Streams0, Buffer::Streams1)
    } else {
        (Buffer::Streams1, Buffer::Streams0)
    }
}

/// The prefix of a sublayer's mixer weights.
pub(super) fn mixer_prefix(sublayer: Sublayer) -> &'static str {
    match sublayer {
        Sublayer::Attention => "hc_attn",
        Sublayer::Mlp => "hc_ffn",
    }
}

impl Lowering<'_> {
    /// Layer `layer`'s `sublayer` mixer weights.
    fn mixer(
        &self,
        bind: &mut Bind,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<MixerWeights<View>, String> {
        let prefix = mixer_prefix(sublayer);
        let mut weight = |name: &str| -> Result<View, String> {
            Ok(bind.view(
                self.buffers
                    .view(&layer_view(layer, &format!("{prefix}_{name}")))?,
            ))
        };
        Ok(MixerWeights {
            fn_: weight("fn")?,
            base: weight("base")?,
            scale: weight("scale")?,
        })
    }

    /// The split mix-collapse's chain sums and tickets.
    fn mix_scratch(&self, bind: &mut Bind) -> MixScratch<View> {
        MixScratch {
            partials: Some(bind.view(self.scratch(Buffer::MixPartials))),
            tickets: Some(bind.view(self.scratch(Buffer::MixTickets))),
        }
    }

    /// The sublayer's mixer over the streams it reads: the mix-collapse op
    /// runs the split or a tiled form by the launch's rows, all matching
    /// the one-CTA entry bit for bit.
    pub(crate) fn hc_mix(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Lowered, String> {
        let mut bind = Bind::default();
        let buffers = MixCollapseBuffers {
            x: bind.view(self.scratch(streams(layer, sublayer).0)),
            mixer: self.mixer(&mut bind, layer, sublayer)?,
            collapsed: bind.view(self.scratch(Buffer::Collapsed)),
            post: bind.view(self.scratch(Buffer::Post)),
            comb: bind.view(self.scratch(Buffer::Comb)),
            scratch: self.mix_scratch(&mut bind),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        one(bind
            .launch(&self.ops.sinkhorn.mix, &buffers, Counts::rows(phase.rows()))
            .map_err(|refusal| format!("Sinkhorn mix-collapse: {refusal}"))?)
    }

    /// The sublayer's mixer over the streams it reads and its input norm of
    /// the collapsed row, in one launch: the split mix-collapse's and the
    /// row engine's norm's outputs bit for bit.
    pub(crate) fn hc_mix_norm(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Lowered, String> {
        let weight = super::rows::input_norm_weight(sublayer);
        let mut bind = Bind::default();
        let buffers = MixCollapseNormBuffers {
            x: bind.view(self.scratch(streams(layer, sublayer).0)),
            mixer: self.mixer(&mut bind, layer, sublayer)?,
            norm_weight: bind.view(self.buffers.view(&layer_view(layer, weight))?),
            collapsed: bind.view(self.scratch(Buffer::Collapsed)),
            post: bind.view(self.scratch(Buffer::Post)),
            comb: bind.view(self.scratch(Buffer::Comb)),
            normed: bind.view(self.scratch(Buffer::Normed)),
            scratch: self.mix_scratch(&mut bind),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        one(bind
            .launch(
                &self.ops.sinkhorn.mix_norm,
                &buffers,
                Counts::rows(phase.rows()),
            )
            .map_err(|refusal| format!("Sinkhorn mix-collapse with norm: {refusal}"))?)
    }

    /// The sublayer's all-reduced output expanded into the other streams.
    pub(crate) fn hc_expand(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Lowered, String> {
        let (input, output) = streams(layer, sublayer);
        let mut bind = Bind::default();
        let buffers = StreamExpandBuffers {
            x: bind.view(self.scratch(input)),
            y: bind.view(self.scratch(Buffer::Reduced)),
            post: bind.view(self.scratch(Buffer::Post)),
            comb: bind.view(self.scratch(Buffer::Comb)),
            output: bind.view(self.scratch(output)),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        one(bind
            .launch(
                &self.ops.sinkhorn.expand,
                &buffers,
                Counts::rows(phase.rows()),
            )
            .map_err(|refusal| format!("Sinkhorn expand: {refusal}"))?)
    }

    /// The mean of the streams the last sublayer expanded into.
    pub(crate) fn hc_head(&self, phase: Phase) -> Result<Lowered, String> {
        let last = self.shape.layers() - 1;
        let mut bind = Bind::default();
        let buffers = StreamHeadBuffers {
            x: bind.view(self.scratch(streams(last, Sublayer::Mlp).1)),
            output: bind.view(self.scratch(Buffer::Collapsed)),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        one(bind
            .launch(
                &self.ops.sinkhorn.head,
                &buffers,
                Counts::rows(phase.rows()),
            )
            .map_err(|refusal| format!("Sinkhorn head: {refusal}"))?)
    }

    /// Whether `phase` runs main layer `layer`'s mixer and input norm as one
    /// launch ([`Self::hc_mix_norm`]): in the decode and verify graphs.
    /// Prefill keeps the split mix-collapse and the row engine's norm.
    pub(crate) fn fuses_input_norm(&self, phase: Phase, layer: u32) -> bool {
        !matches!(phase, Phase::Prefill { .. }) && layer < self.shape.geometry.layers()
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::latent_attention::LatentKvFormat;

    use crate::journal_tests::{head_rows, inputs};
    use crate::lower::Lowering;
    use crate::program::{Phase, Stage, stages};
    use crate::row_ops_tests::Rig;

    /// Every prefill mixer at the served schedule's edges, on every rank:
    /// the split below 144 rows over 16 CTAs per row, the 4-row tile from
    /// 144 to 1504 rows and the 8-row tile from 1505 rows, one CTA per tile.
    #[test]
    fn prefill_mixers_run_the_scheduled_entry_at_the_schedule_edges() {
        let rig = Rig::new(LatentKvFormat::Fp8E4M3Block128, false);
        for (rows, symbol, grid) in [
            (143, "split", [16, 143, 1]),
            (144, "tiled_r4", [36, 1, 1]),
            (1504, "tiled_r4", [376, 1, 1]),
            (1505, "tiled_r8", [189, 1, 1]),
        ] {
            let phase = Phase::Prefill { rows };
            let symbol = format!("chop_sinkhorn_hc_glm5next_mix_collapse_{symbol}_v1");
            for rank in 0..rig.shape.ranks() {
                let inputs = inputs(rank, phase);
                let lowering = Lowering {
                    inputs: &inputs,
                    head_rows: head_rows(phase),
                    ..rig.lowering(rank, phase, None)
                };
                let mut mixers = 0;
                for stage in stages(rig.shape, phase) {
                    if !matches!(stage, Stage::HcMix(..)) {
                        continue;
                    }
                    let launches = lowering.lower(phase, stage).unwrap().unwrap();
                    let [launch] = launches.as_slice() else {
                        panic!("{phase:?} {stage:?}: {} launches", launches.len());
                    };
                    let what = || format!("rank {rank} {phase:?} {stage:?}");
                    assert_eq!(launch.symbol, symbol, "{}", what());
                    assert_eq!(launch.rows, rows, "{}", what());
                    assert_eq!(launch.plan.grid, grid, "{}", what());
                    mixers += 1;
                }
                assert_eq!(mixers, 2 * rig.shape.layers(), "{phase:?}");
            }
        }
    }
}
