//! Tail rotation and causal window addressing around native latent attention.
//!
//! Projection and normalization outputs stay F32 through rotation. Query
//! storage is rounded once, after rotation, to native attention's BF16 Q operand.
//! BF16 Q and BF16 probabilities are vendor attention-kernel boundaries
//! adopted by the latent-attention contract. The F32-TQ reference keeps both
//! F32; model qualification measures that gap. Window KV instead rounds at
//! the shared trained quantizer's input, after its F32 rotation.

use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};
use crate::latent_attention::{LatentKvFormat, LatentWindowBuffers};

/// Adjacent pairs in the final `rotary` channels of each F32 head. Each
/// product and the following sum or difference rounds separately to F32.
/// The unrotated prefix stays unchanged. The op rotates the F32 input in
/// place and writes the entire head to a separate BF16 Q output, rounding
/// exactly once after rotation. The input and BF16 output must not overlap.
/// Nonfinite inputs or tables set `NONFINITE_INPUT`; nonfinite arithmetic or
/// BF16 conversion sets `UNREPRESENTABLE_OUTPUT`. Any previous or new row
/// flag preserves the entire F32 query and writes positive-zero native Q.
/// Status uses the shared quantizer flags and sticky OR protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LatentQueryRope {
    pub heads: u32,
    pub latent: u32,
    pub rotary: u32,
}

impl Validate for LatentQueryRope {
    fn validate(&self) -> Result<(), Refusal> {
        if self.heads == 0
            || self.latent == 0
            || self.rotary == 0
            || self.rotary > self.latent
            || !self.latent.is_multiple_of(2)
            || !self.rotary.is_multiple_of(2)
        {
            return Err(Refusal(
                "latent query RoPE requires heads and an even tail within each head",
            ));
        }
        self.heads
            .checked_mul(self.latent)
            .ok_or(Refusal("latent query RoPE row width overflows"))?;
        Ok(())
    }
}

impl NumericalContract for LatentQueryRope {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "latent.query-tail-rope.f32-to-bf16.v1.h{}.d{}.r{}",
            self.heads, self.latent, self.rotary
        )))
    }
}

/// Mapping absolute token positions to physical window slots. A circular
/// allocation may be larger than the visible window, for example 136 slots
/// for a 128-token window and eight provisional verify rows. A staged
/// allocation starts at an explicit absolute position for each sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum WindowAddressing {
    Circular { slots: u32 },
    Staged { slots: u32 },
}

impl WindowAddressing {
    fn tag(self) -> &'static str {
        match self {
            Self::Circular { .. } => "circular",
            Self::Staged { .. } => "staged",
        }
    }

    pub fn slots(self) -> u32 {
        match self {
            Self::Circular { slots } | Self::Staged { slots } => slots,
        }
    }

    pub fn slot(self, position: i32, first: i32) -> Result<u32, Refusal> {
        self.validate()?;
        if position < 0 {
            return Err(Refusal("a stored window position must be nonnegative"));
        }
        match self {
            Self::Circular { slots } => Ok(position as u32 % slots),
            Self::Staged { slots } => {
                let offset = i64::from(position) - i64::from(first);
                if first < 0 || offset < 0 || offset >= i64::from(slots) {
                    return Err(Refusal("window position lies outside staged storage"));
                }
                Ok(offset as u32)
            }
        }
    }
}

impl Validate for WindowAddressing {
    fn validate(&self) -> Result<(), Refusal> {
        if self.slots() == 0 || self.slots() > i32::MAX as u32 {
            return Err(Refusal("window slots must fit positive I32"));
        }
        Ok(())
    }
}

/// F32 projection rows and BF16 weights, F32 plain RMSNorm, F32 tail rotation,
/// shared ceil E4M3/UE8M0 quantization per 32, then a write to separate code and scale
/// planes. Every quantizer input rounds to BF16 before amax or code choice.
/// Its result is bitwise `latent_attention_oracle::encode_latent_row` of
/// that BF16 input in `Fp8E4M3Ue8M0Block32` format.
///
/// A circular write keeps only positions in the final `slots` positions
/// ending at each sequence's supplied `last_position`. This makes prefill
/// ring seeding race-free even when the chunk exceeds the ring. Staging
/// writes every valid supplied position. Invalid sequence ids, negative
/// positions, positions after the supplied last position, and positions
/// outside staging write neither plane. Valid destinations are unique.
/// Nonfinite source values set `NONFINITE_INPUT`; nonfinite norm or rotation
/// results set `UNREPRESENTABLE_OUTPUT`. The shared quantizer adds its flags.
/// Any prior or new flag preserves both destination planes, including the
/// previous circular cell. It is not a valid new key. The caller clears the
/// selected status words once per ordered step and rejects the step before
/// accepting outputs or advancing retained bounds if any word is nonzero.
/// The caller stages attention inputs or reserves enough circular headroom
/// so a write cannot overwrite any key needed by another query in the step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LatentWindowWrite {
    /// Native code/scale planes or their exact BF16 products in one plane.
    /// BF16 has no scale plane; its scale view and stride are ignored.
    pub kv: LatentKvFormat,
    pub latent: u32,
    pub rotary: u32,
    pub addressing: WindowAddressing,
    /// Positive finite F32 epsilon. The norm computes two adjacent squares
    /// per lane with F32 FMA, reduces 256 lanes in warp-down trees (16..1),
    /// then the eight warp sums in the same tree padded with zeros. It uses
    /// separate F32 mean and epsilon addition, sqrt then reciprocal, and
    /// applies the weight after scaling, without a BF16 intermediate.
    pub epsilon_bits: u32,
}

impl Validate for LatentWindowWrite {
    fn validate(&self) -> Result<(), Refusal> {
        window_storage_width(self.kv, self.latent)?;
        LatentQueryRope {
            heads: 1,
            latent: self.latent,
            rotary: self.rotary,
        }
        .validate()?;
        if !self.latent.is_multiple_of(128) {
            return Err(Refusal(
                "native window rows require whole 128-channel latent blocks",
            ));
        }
        let eps = f32::from_bits(self.epsilon_bits);
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Refusal(
                "window RMSNorm epsilon must be positive finite F32",
            ));
        }
        self.addressing.validate()
    }
}

impl NumericalContract for LatentWindowWrite {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "latent.window-write.input-f32.norm-f32.post-rope-bf16.ceil32{}.v1.d{}.r{}.{}.s{}.eps{:08x}",
            if self.kv == LatentKvFormat::Bf16 {
                ".bf16"
            } else {
                ""
            },
            self.latent,
            self.rotary,
            self.addressing.tag(),
            self.addressing.slots(),
            self.epsilon_bits
        )))
    }
}

/// Bytes in each stored row. Both modes use the same trained ceil quantizer.
/// BF16 stores exact code-times-scale products, including the zero sign. A
/// nonzero product must be normal, finite and exactly BF16. Invalid products
/// set `UNREPRESENTABLE_OUTPUT` and preserve the previous destination cell.
pub fn window_storage_width(kv: LatentKvFormat, latent: u32) -> Result<u32, Refusal> {
    match kv {
        LatentKvFormat::Fp8E4M3Ue8M0Block32 => Ok(latent),
        LatentKvFormat::Bf16 => latent
            .checked_mul(2)
            .ok_or(Refusal("BF16 window width overflows")),
        _ => Err(Refusal(
            "window storage requires native ceil32 or exact BF16 products",
        )),
    }
}

/// Each query at p sees absolute positions max(0, p+1-window)..=p, in
/// increasing order, followed by -1 holes to the fixed `window` width.
/// Every output entry is either -1 or a valid slot in [0, slots). Indices
/// name circular slots or staged rows, never compressed positions. Visibility
/// is checked in absolute positions before modulo: stale or future positions
/// never become live keys by wrapping, and an out-of-range physical index
/// must never be reduced modulo the ring capacity.
/// A negative query position or an absent staged position is a -1 hole.
/// An invalid staged sequence id writes no row. The live count excludes padding.
/// This is ordinary causal visibility; bidirectional draft blocks use the
/// separate `latent_visibility::WindowBlockVisibility` contract. Staged
/// storage here must be fully populated. Gathered prefixes with holes use
/// `latent_staged_window_keys::LatentStagedWindowKeys` instead.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LatentWindowKeys {
    pub window: u32,
    pub addressing: WindowAddressing,
}

impl Validate for LatentWindowKeys {
    fn validate(&self) -> Result<(), Refusal> {
        self.addressing.validate()?;
        if self.window == 0 || self.window > i32::MAX as u32 {
            return Err(Refusal("visible window width must fit positive I32"));
        }
        if matches!(self.addressing, WindowAddressing::Circular { slots } if slots < self.window) {
            return Err(Refusal("circular storage must cover the visible window"));
        }
        Ok(())
    }
}

impl NumericalContract for LatentWindowKeys {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "latent.window-keys.causal-oldest-first.v1.w{}.{}.s{}",
            self.window,
            self.addressing.tag(),
            self.addressing.slots()
        )))
    }
}

crate::scratch_record! {
    /// F32 in-place `query` and separate BF16 `native_q`, both
    /// `[rows, heads, latent]`; F32 `[rows, rotary/2, cos/sin]` from the
    /// model's pinned table generator (shared across heads); device U32
    /// `live_rows`. Rows outside that count are untouched and unread.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentQueryRopeBuffers {
        pub query: value,
        pub native_q: value,
        pub cos_sin: value,
        pub live_rows: value,
        /// Sticky U32 status per input row; the offset is in words.
        pub status: value,
        pub status_row_offset: scalar(u64),
    }
}

crate::scratch_record! {
    /// I32 positions per query and output `[rows, window]` I32 slots.
    /// Staged storage also binds each row's I32 sequence and each sequence's
    /// I32 first absolute position. Circular lists bind neither optional
    /// view. `sequences` bounds staged sequence indices.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentWindowKeyBuffers {
        pub positions: value,
        pub row_batch: option,
        pub first_positions: option,
        pub sequences: scalar(u32),
        pub key_lists: value,
        pub live_rows: value,
    }
}

crate::scratch_record! {
    /// F32 `input` projection rows and BF16 `weight` of width latent.
    /// F32 `latent` scratch receives the normalized, rotated rows.
    /// Quantized scratch is
    /// dense `[rows, latent]` codes and `[rows, latent/32]` UE8M0 scales.
    /// `window` is the attention reader's exact pair of independently
    /// strided planes; its key-list fields are not read by this writer.
    /// `positions` and `row_batch` are I32 per row. `last_positions` gives
    /// each sequence's last position in the write batch. Staging also binds
    /// its I32 first absolute positions; circular addressing binds None.
    /// The storage descriptor supplies the physical extent: a 136-cell
    /// circular writer uses `Staged { slots: 136 }` on that reader view.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct LatentWindowWriteBuffers {
        pub input: value,
        pub weight: value,
        pub latent: value,
        pub cos_sin: value,
        pub positions: value,
        pub row_batch: value,
        pub first_positions: option,
        pub last_positions: value,
        pub window: record(LatentWindowBuffers),
        pub quantized_codes: value,
        pub quantized_scales: value,
        pub live_rows: value,
        /// Sticky U32 status per input row; the offset is in words.
        pub status: value,
        pub status_row_offset: scalar(u64),
    }
}
