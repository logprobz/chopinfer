//! Tail rotation and causal window addressing around native latent attention.
//!
//! Projection and normalization outputs stay F32 through rotation. Query
//! storage is rounded once, after rotation, to native attention's BF16 Q operand.
//! BF16 Q and BF16 probabilities are vendor attention-kernel boundaries
//! adopted by the latent-attention contract. The F32-TQ reference keeps both
//! F32; model qualification measures that gap. Window KV instead rounds at
//! the shared trained quantizer's input, after its F32 rotation.

use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};

/// Adjacent pairs in the final `rotary` channels of each F32 head. Each
/// product and the following sum or difference rounds separately to F32.
/// The unrotated prefix stays unchanged. The op rotates the F32 input in
/// place and writes the entire head to a separate BF16 Q output, rounding
/// exactly once after rotation. The input and BF16 output must not overlap.
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

/// F32 kv_norm output, then in-place F32 tail rotation, shared ceil
/// E4M3/UE8M0 quantization per 32, then a write to separate code and scale
/// planes. Every quantizer input rounds to BF16 before amax or code choice.
/// Its result is bitwise `latent_attention_oracle::encode_latent_row` of
/// that BF16 input in `Fp8E4M3Ue8M0Block32` format.
///
/// A circular write keeps only positions in the final `slots` positions
/// ending at each sequence's supplied `last_position`. This makes prefill
/// ring seeding race-free even when the chunk exceeds the ring. Staging
/// writes every supplied position. Positions are unique within a sequence.
/// The caller stages attention inputs or reserves enough circular headroom
/// so a write cannot overwrite any key needed by another query in the step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LatentWindowWrite {
    pub latent: u32,
    pub rotary: u32,
    pub addressing: WindowAddressing,
}

impl Validate for LatentWindowWrite {
    fn validate(&self) -> Result<(), Refusal> {
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
        self.addressing.validate()
    }
}

impl NumericalContract for LatentWindowWrite {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(format!(
            "latent.window-write.post-rope-bf16.ceil32.v1.d{}.r{}.{}.s{}",
            self.latent,
            self.rotary,
            self.addressing.tag(),
            self.addressing.slots()
        )))
    }
}

/// Each query at p sees absolute positions max(0, p+1-window)..=p, in
/// increasing order, followed by -1 holes to the fixed `window` width.
/// Indices name circular slots or staged rows, never compressed positions.
/// A query position of -1 writes only holes without reading sequence data.
/// This is ordinary causal visibility; bidirectional draft blocks use the
/// separate `latent_visibility::WindowBlockVisibility` contract.
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
