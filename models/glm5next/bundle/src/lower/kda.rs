//! Lowering of the KDA convolution, delta rule and gated norm through the
//! instance's prepared ops (`crate::ops::kda`).

use kernel_core::backend::{Counts, View};
use kernel_core::recurrent::{ConvBuffers, DeltaRuleBuffers, GatedNormBuffers, RecurrentSlots};
use model_glm5next_contract::geometry::LayerKind;

use super::{Lowering, view_name};
use crate::buffers::Buffer;
use crate::ops::{Bind, Lowered, one};
use crate::program::Phase;

impl Lowering<'_> {
    /// The counts of a KDA launch of `phase`: its rows over its items. A
    /// decode bucket carries one row per item, a verify step its items; an
    /// eager prefill carries at most the items its per-item arrays hold
    /// (the instance's). The prefill ops plan for every served item.
    fn kda_counts(&self, phase: Phase) -> Counts {
        let items = match phase {
            Phase::Prefill { .. } => (self.inputs.slot_ids.length / 4) as u32,
            Phase::Decode(rows) => rows,
            Phase::Verify { items, .. } => items,
        };
        Counts::rows(phase.rows()).with_items(items, 0)
    }

    /// The per-sequence inputs a KDA launch of `phase` reads: each item's
    /// slot and the live item count, a prefill's row offsets, start
    /// positions, initial flags and anchors, a verify step's row offsets and
    /// checkpoints.
    fn kda_slots(&self, bind: &mut Bind, phase: Phase) -> RecurrentSlots<View> {
        let mut scratch = |buffer| bind.view(self.scratch(buffer));
        let (row_offsets, initial, anchor_ids, checkpoint_ids) = match phase {
            Phase::Prefill { .. } => (
                Some(scratch(Buffer::CuSeqlens)),
                Some(scratch(Buffer::Initial)),
                Some(scratch(Buffer::AnchorIds)),
                None,
            ),
            Phase::Decode(_) => (None, None, None, None),
            Phase::Verify { .. } => (
                Some(scratch(Buffer::CuRows)),
                None,
                None,
                Some(scratch(Buffer::CheckpointIds)),
            ),
        };
        let prefill = matches!(phase, Phase::Prefill { .. });
        RecurrentSlots {
            row_offsets,
            start_positions: prefill.then(|| bind.view(self.inputs.prefix_len)),
            slot_ids: bind.view(self.inputs.slot_ids),
            initial,
            anchor_ids,
            checkpoint_ids,
            live_sequences: bind.view(self.inputs.live_batch),
        }
    }

    /// The KDA convolution of `layer`: its q, k and v projections
    /// (`KdaQkv`) through the layer's short convolution into `KdaConv`.
    pub(crate) fn kda_conv(&self, phase: Phase, layer: u32) -> Result<Lowered, String> {
        let width = self.shape.profile.linear_heads * self.shape.profile.linear_head_dim;
        let ordinal = self.linear_ordinal(layer)?;
        let mut bind = Bind::default();
        // Binding point: the weight plan widens the BF16 convolution
        // weights to the F32 the op reads.
        let mut weights = Vec::new();
        for name in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
            weights.push(bind.view(self.buffers.view(&view_name(layer, name))?));
        }
        let buffers = ConvBuffers {
            input: bind.view(self.scratch(Buffer::KdaQkv)),
            input_stride: 3 * width,
            weights,
            state: bind.view(self.buffers.state.layer("kda.conv", ordinal)?),
            output: bind.view(self.scratch(Buffer::KdaConv)),
            slots: self.kda_slots(&mut bind, phase),
        };
        one(bind
            .launch(self.ops.kda.conv(phase), &buffers, self.kda_counts(phase))
            .map_err(|refusal| format!("KDA convolution: {refusal}"))?)
    }

    /// The KDA delta rule of `layer`: one launch at decode and verify, the
    /// chunk prepare and the scan over one workspace at prefill.
    pub(crate) fn kda_delta(&self, phase: Phase, layer: u32) -> Result<Vec<Lowered>, String> {
        let p = self.shape.profile;
        let width = p.linear_heads * p.linear_head_dim;
        let ordinal = self.linear_ordinal(layer)?;
        let mut bind = Bind::default();
        let buffers = DeltaRuleBuffers {
            qkv: bind.view(self.scratch(Buffer::KdaConv)),
            qkv_stride: 3 * width,
            decay: bind.view(self.scratch(Buffer::KdaGate)),
            decay_stride: width,
            beta: bind.view(self.scratch(Buffer::KdaBeta)),
            beta_stride: p.linear_heads,
            a_log: bind.view(self.buffers.view(&view_name(layer, "A_log"))?),
            dt_bias: bind.view(self.buffers.view(&view_name(layer, "dt_bias"))?),
            state: bind.view(self.buffers.state.layer("kda.state", ordinal)?),
            output: bind.view(self.scratch(Buffer::KdaOut)),
            workspace: matches!(phase, Phase::Prefill { .. })
                .then(|| bind.view(self.scratch(Buffer::Workspace))),
            slots: self.kda_slots(&mut bind, phase),
        };
        bind.launch(self.ops.kda.delta(phase), &buffers, self.kda_counts(phase))
            .map_err(|refusal| format!("KDA delta rule: {refusal}"))
    }

    /// The KDA gated norm of `layer`'s rule output (`KdaOut`) by its output
    /// gate, each row on its own.
    pub(crate) fn kda_norm(&self, phase: Phase, layer: u32) -> Result<Lowered, String> {
        let width = self.shape.profile.linear_heads * self.shape.profile.linear_head_dim;
        let mut bind = Bind::default();
        let buffers = GatedNormBuffers {
            input: bind.view(self.scratch(Buffer::KdaOut)),
            gate: bind.view(self.scratch(Buffer::KdaOutputGate)),
            gate_stride: width,
            weight: bind.view(self.buffers.view(&view_name(layer, "o_norm.weight"))?),
            output: bind.view(self.scratch(Buffer::KdaNorm)),
            live_rows: bind.view(self.inputs.live_tokens),
        };
        one(bind
            .launch(&self.ops.kda.norm, &buffers, Counts::rows(phase.rows()))
            .map_err(|refusal| format!("KDA gated norm: {refusal}"))?)
    }

    /// A KDA layer's index among the KDA layers.
    fn linear_ordinal(&self, layer: u32) -> Result<u64, String> {
        let g = self.shape.geometry;
        if layer >= g.layers() || g.layer_kind(layer) != LayerKind::Linear {
            return Err(format!("layer {layer} is not a KDA layer"));
        }
        Ok((0..layer)
            .filter(|&i| g.layer_kind(i) == LayerKind::Linear)
            .count() as u64)
    }
}
