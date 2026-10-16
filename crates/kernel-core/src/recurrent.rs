//! Buffer roles of the recurrent family: the causal convolution and the
//! gated delta rule (`contract::RecurrentQuery`), the per-channel-decay
//! delta rule and its short convolution
//! (`per_channel_decay_delta_rule::PerChannelDecayDeltaRuleQuery`), and the
//! gated RMSNorm of their output.
//!
//! # State
//!
//! A recurrent layer keeps one state per sequence in a pool of entries (the
//! query's `state_elements`, F32 for a delta rule and BF16 for a
//! convolution), addressed by entry id, never by batch order. The host owns
//! the pool and the entry ids; an op only reads and writes the entries the
//! ids of its launch name ([`RecurrentSlots`]):
//!
//! - Prefill reads entry `slot_ids[s]` when `initial[s]` is nonzero (a zero
//!   state otherwise) and writes the state after the sequence's last row
//!   back to it. When `anchor_ids[s]` is nonnegative and the call ends
//!   inside one of the op's prefill chunks that starts after the call's
//!   first row, the op also writes the state at the start of that chunk to
//!   entry `anchor_ids[s]`: the state a call ending there would write,
//!   which exact prefix capture takes inside a step. Ops that do guarantee
//!   `Anchors::Spare`; the guarantee's chunk granule names the chunks.
//! - Decode advances entry `slot_ids[s]` in place, one row per sequence.
//! - Verify reads the committed state from entry `slot_ids[s]`, leaves it
//!   unchanged, and writes the state after row `r` of sequence `s` to
//!   checkpoint entry `checkpoint_ids[s * max_rows + r]`, so a commit of
//!   `kept` rows keeps checkpoint `kept - 1`. Verify rows equal to decode
//!   rows (each row's output and checkpoint those one-row decode steps give)
//!   is the guarantee `Rows::OneRow`.
//!
//! Two sequences of a launch never share an entry; a sequence's anchor or
//! checkpoint entry may alias its own source entry.
//!
//! Row counts and offsets are host-known per step, and live counts are
//! device scalars, so one launch plan serves every graph replay of its
//! counts.

crate::scratch_record! {
    /// The per-sequence operands of a recurrent launch: one entry per
    /// sequence of the launch (`backend::Counts::items`, at least its live
    /// sequences) in each i32 list, and the device count of live sequences.
    /// A phase reads the roles it names and ignores the rest:
    ///
    /// - prefill: `row_offsets`, `slot_ids`, `initial`, `anchor_ids`, and
    ///   `start_positions` where the op's chunks start at absolute
    ///   positions;
    /// - decode: `slot_ids`;
    /// - verify: `row_offsets`, `slot_ids` and `checkpoint_ids`
    ///   (`items * max_rows` entries).
    ///
    /// `row_offsets` holds `items + 1` absolute row offsets into the
    /// row operands, so a launch may cover any row range of its buffers;
    /// `start_positions[s]` is the absolute position of sequence `s`'s
    /// first row.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RecurrentSlots {
        pub row_offsets: option,
        pub start_positions: option,
        pub slot_ids: value,
        pub initial: option,
        pub anchor_ids: option,
        pub checkpoint_ids: option,
        pub live_sequences: value,
    }
}

crate::scratch_record! {
    /// A depthwise causal convolution's buffers: BF16 `input` rows of
    /// `input_stride` elements whose first columns are the channels, the
    /// convolution `weights` (one `[channels, kernel]` weight, or the q, k
    /// and v weights in order where the query binds them separately), the
    /// layer's state pool, the BF16 `output` rows `[rows, channels]` and
    /// the slots.
    #[derive(Clone, Debug, Eq, PartialEq)]
    pub struct ConvBuffers {
        pub input: value,
        pub input_stride: scalar(u32),
        pub weights: values,
        pub state: value,
        pub output: value,
        pub slots: record(RecurrentSlots),
    }
}

crate::scratch_record! {
    /// A delta rule's buffers: the BF16 `qkv` rows of `qkv_stride` elements
    /// (q, then k, then v), the BF16 decay input rows (the gated delta
    /// rule's `a`, one per value head; the per-channel rule's gate
    /// projection, one per key channel) and beta logit rows, each with its
    /// row stride, the F32 `a_log` and `dt_bias`, the layer's F32 state
    /// pool, the BF16 `output` rows `[rows, value_heads, value_dim]`, the
    /// slots, and a prefill's workspace of the bytes the op sizes
    /// (`OpInfo::bytes`, role `workspace`), which no launch preserves.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DeltaRuleBuffers {
        pub qkv: value,
        pub qkv_stride: scalar(u32),
        pub decay: value,
        pub decay_stride: scalar(u32),
        pub beta: value,
        pub beta_stride: scalar(u32),
        pub a_log: value,
        pub dt_bias: value,
        pub state: value,
        pub output: value,
        pub workspace: option,
        pub slots: record(RecurrentSlots),
    }
}

crate::scratch_record! {
    /// A gated RMSNorm's buffers: BF16 `input` and `output` rows `[rows,
    /// width]`, BF16 `gate` rows of `gate_stride` elements, the BF16
    /// per-head `weight` `[head_dim]` and the device count of live rows.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct GatedNormBuffers {
        pub input: value,
        pub gate: value,
        pub gate_stride: scalar(u32),
        pub weight: value,
        pub output: value,
        pub live_rows: value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation::Traversal;

    #[test]
    fn roles_visit_by_path_and_skip_scalars() {
        let slots = RecurrentSlots {
            row_offsets: Some(1),
            start_positions: None,
            slot_ids: 2,
            initial: Some(3),
            anchor_ids: Some(4),
            checkpoint_ids: None,
            live_sequences: 5,
        };
        let conv = ConvBuffers {
            input: 10,
            input_stride: 2560,
            weights: vec![11, 12, 13],
            state: 14,
            output: 15,
            slots,
        };
        let mut paths = Vec::new();
        conv.visit_named(&mut |path, &handle| paths.push(format!("{path}={handle}")));
        assert_eq!(
            paths,
            [
                "input=10",
                "weights[0]=11",
                "weights[1]=12",
                "weights[2]=13",
                "state=14",
                "output=15",
                "slots.row_offsets=1",
                "slots.slot_ids=2",
                "slots.initial=3",
                "slots.anchor_ids=4",
                "slots.live_sequences=5",
            ]
        );
        let mapped = conv.map(|handle| handle * 2);
        assert_eq!(mapped.input_stride, 2560);
        assert_eq!(mapped.weights, [22, 24, 26]);
    }
}
