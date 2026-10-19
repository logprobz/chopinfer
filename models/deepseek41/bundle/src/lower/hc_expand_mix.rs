use super::{Lowering, layer_view, mixer, pre, rows, streams, sublayer_index};
use crate::{
    ops::Lowered,
    program::{Phase, Sublayer},
};
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixBuffers, SinkhornRoundExpandMixBuffers, SinkhornRoundNormBuffers,
};
use model_deepseek41_contract::geometry::Geometry;

/// Only adjacent sublayers whose intervening stages preserve the streams.
/// Engram modifies the next attention's streams, and the final expand feeds
/// the head. Neither boundary can reuse the earlier projection partials.
fn successor(g: &Geometry, layer: u32, sublayer: Sublayer) -> Option<(u32, Sublayer)> {
    if layer >= g.layers() {
        return None;
    }
    match sublayer {
        Sublayer::Attention => Some((layer, Sublayer::Mlp)),
        Sublayer::Mlp if layer + 1 < g.layers() && !g.is_engram_layer(layer + 1) => {
            Some((layer + 1, Sublayer::Attention))
        }
        _ => None,
    }
}
impl Lowering<'_> {
    fn has_expand_mix(&self, phase: Phase) -> bool {
        self.ops.hc.as_ref().is_some_and(|ops| {
            ops.has_expand_mix()
                && (matches!(phase, Phase::Decode(_)) || ops.has_finish_norm(phase))
        })
    }
    pub(super) fn mix_was_fused(&self, phase: Phase, layer: u32, sublayer: Sublayer) -> bool {
        if !self.has_expand_mix(phase) {
            return false;
        }
        let previous = match sublayer {
            Sublayer::Mlp => Some((layer, Sublayer::Attention)),
            Sublayer::Attention => layer.checked_sub(1).map(|l| (l, Sublayer::Mlp)),
        };
        previous
            .is_some_and(|(l, s)| successor(self.shape.geometry, l, s) == Some((layer, sublayer)))
    }
    pub(super) fn finish_norm_was_fused(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<bool, String> {
        if !self
            .ops
            .hc
            .as_ref()
            .is_some_and(|hc| hc.has_finish_norm(phase))
            || !self.mix_was_fused(phase, layer, sublayer)
        {
            return Ok(false);
        }
        let previous = match sublayer {
            Sublayer::Mlp => (layer, Sublayer::Attention),
            Sublayer::Attention => (layer - 1, Sublayer::Mlp),
        };
        // A non-deferred all-reduce still needs the standalone norm after
        // expand/mix. Require the same admission as the preceding stage.
        Ok(self
            .round_expand_with_next_mix(phase, previous.0, previous.1)?
            .is_some())
    }

    pub(super) fn expand_with_next_mix(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Option<Vec<Lowered>>, String> {
        // The preceding reduction stage already wrote the round and both
        // HC results when its producer exposed the deferred conversion.
        if self
            .round_expand_with_next_mix(phase, layer, sublayer)?
            .is_some()
        {
            return Ok(Some(Vec::new()));
        }
        if !self.has_expand_mix(phase) {
            return Ok(None);
        }
        let Some((next_layer, next_sublayer)) = successor(self.shape.geometry, layer, sublayer)
        else {
            return Ok(None);
        };
        let s = &self.buffers.scratch;
        let index = sublayer_index(layer, sublayer);
        let row = |span| rows(span, phase.rows());
        let weight = |name: &str| {
            self.buffers.view(&layer_view(
                next_layer,
                &format!("{}_{name}", mixer(next_sublayer)),
            ))
        };
        let buffers = SinkhornExpandMixBuffers {
            x: row(streams(s, index)),
            y: row(s.reduced),
            post_in: row(s.post),
            comb_in: row(s.comb),
            expanded: row(streams(s, index + 1)),
            fn_: weight("fn")?,
            base: weight("base")?,
            scale: weight("scale")?,
            pre_in: row(pre(s, index)),
            collapsed: row(s.collapsed),
            pre: row(pre(s, index + 1)),
            post: row(s.post),
            comb: row(s.comb),
            live_rows: self.inputs.live_tokens,
        };
        self.ops
            .hc
            .as_ref()
            .unwrap()
            .expand_mix(phase.rows(), buffers, self.buffers.workspace)
            .map(Some)
            .map_err(|e| format!("expand and next lagged mixer: {e}"))
    }

    pub(super) fn round_expand_with_next_mix(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if !self.has_expand_mix(phase) {
            return Ok(None);
        }
        let hc = self.ops.hc.as_ref().unwrap();
        if !hc.has_round_expand_mix() {
            return Ok(None);
        }
        let Some((next_layer, next_sublayer)) = successor(self.shape.geometry, layer, sublayer)
        else {
            return Ok(None);
        };
        let Some(reduce) = self.ops.reduce.as_ref() else {
            return Ok(None);
        };
        let views = self.reduce_views(phase, |s| s.reduced)?;
        let Some((mut launches, sum)) = reduce
            .defer_round(self.rank, phase.rows(), &views)
            .map_err(|e| format!("deferred all-reduce round: {e}"))?
        else {
            return Ok(None);
        };
        let s = &self.buffers.scratch;
        let index = sublayer_index(layer, sublayer);
        let row = |span| rows(span, phase.rows());
        let weight = |name: &str| {
            self.buffers.view(&layer_view(
                next_layer,
                &format!("{}_{name}", mixer(next_sublayer)),
            ))
        };
        let buffers = SinkhornRoundExpandMixBuffers {
            x: row(streams(s, index)),
            sum,
            rounded: row(s.reduced),
            post_in: row(s.post),
            comb_in: row(s.comb),
            expanded: row(streams(s, index + 1)),
            fn_: weight("fn")?,
            base: weight("base")?,
            scale: weight("scale")?,
            pre_in: row(pre(s, index)),
            collapsed: row(s.collapsed),
            pre: row(pre(s, index + 1)),
            post: row(s.post),
            comb: row(s.comb),
            live_rows: self.inputs.live_tokens,
        };
        let next = if hc.has_finish_norm(phase) {
            let norm = self.input_norm_views(phase, next_layer, next_sublayer)?;
            hc.round_expand_mix_norm(
                phase,
                SinkhornRoundNormBuffers { hc: buffers, norm },
                self.buffers.workspace,
            )
        } else {
            hc.round_expand_mix(phase.rows(), buffers, self.buffers.workspace)
        };
        launches.extend(next.map_err(|e| format!("round, expand and next lagged mixer: {e}"))?);
        Ok(Some(launches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{Stage, stages};
    use model_deepseek41_contract::geometry::FLASH;
    #[test]
    fn fused_boundaries_follow_the_program_and_stop_at_engram_and_head() {
        let program = stages(&FLASH, Phase::Decode(1));
        let mut count = 0;
        for (at, stage) in program.iter().enumerate() {
            let Stage::HcExpand(l, s) = *stage else {
                continue;
            };
            if let Some((layer, sublayer)) = successor(&FLASH, l, s) {
                let next = program[at + 1..]
                    .iter()
                    .find(|s| !matches!(s, Stage::DsparkFeature(_)))
                    .unwrap();
                assert_eq!(*next, Stage::HcMix(layer, sublayer));
                count += 1;
            }
        }
        assert_eq!(count, 77);
        for &engram in FLASH.shape.engram.layers {
            assert_eq!(successor(&FLASH, engram - 1, Sublayer::Mlp), None);
        }
        assert_eq!(successor(&FLASH, FLASH.layers() - 1, Sublayer::Mlp), None);
        assert_eq!(successor(&FLASH, FLASH.layers(), Sublayer::Attention), None);
    }
}
