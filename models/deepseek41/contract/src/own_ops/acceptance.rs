//! Integer acceptance over existing sampler records; no resampling or confidence policy.
//! The caller retains full sampler payload validation, including requested top
//! logprobs and constraint masks, before publishing any successful decision.
use kernel_core::backend::Need;
use kernel_core::contract::Refusal;
use kernel_core::sampling::{OUTPUT_BYTES, STATUS_EMPTY, STATUS_SAMPLED};

pub const CONTRACT: &str = "deepseek41.dspark-acceptance.u32.v1";
pub const ITEM_BYTES: usize = 40;
pub const RESULT_BYTES: usize = 32;
pub const BLOCK: usize = 5;
pub const NONE: u32 = u32::MAX;
pub const TERMINAL: u32 = 1;
pub const REJECTED: u32 = 2;
pub const OK: u32 = 0;
pub const BAD_DESCRIPTOR: u32 = 1;
pub const BAD_SAMPLE: u32 = 2;
pub const EMPTY: u32 = 3;
pub const BAD_PROPOSAL: u32 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Acceptance {
    pub vocab: u32,
    pub sample_rows: u32,
    pub draft_items: u32,
    pub stop_ids: u32,
}
impl Acceptance {
    pub fn validate(self, need: Need) -> Result<(), Refusal> {
        if self.vocab == 0
            || self.sample_rows == 0
            || self.sample_rows > 192
            || self.draft_items > 32
            || need.capacity == 0
            || need.capacity > 32
        {
            return Err(Refusal("DSpark acceptance capacity is invalid"));
        }
        Ok(())
    }
}

/// One item: epoch, remaining tokens, first sample, draft item, K, stop range,
/// and a zero reserved word. K=0 uses NONE for draft_item.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Item {
    pub epoch: u64,
    pub remaining: u64,
    pub first_sample: u32,
    pub draft_item: u32,
    pub k: u32,
    pub stop_offset: u32,
    pub stop_count: u32,
    pub reserved: u32,
}
impl Item {
    pub fn encode(self) -> [u8; ITEM_BYTES] {
        let mut bytes = [0; ITEM_BYTES];
        bytes[..8].copy_from_slice(&self.epoch.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.remaining.to_le_bytes());
        for (i, word) in [
            self.first_sample,
            self.draft_item,
            self.k,
            self.stop_offset,
            self.stop_count,
            self.reserved,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[16 + i * 4..20 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// A request failure (EMPTY) discards its tentative prefix. Every other
/// failure refuses the complete view, even if it occurs in a rejected suffix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Decision {
    pub epoch: u64,
    pub accepted: u32,
    pub kept: u32,
    pub flags: u32,
    pub failure: u32,
    pub failure_row: u32,
    pub reserved: u32,
}
impl Decision {
    pub fn failed(epoch: u64, failure: u32, row: u32) -> Self {
        Self {
            epoch,
            accepted: 0,
            kept: 0,
            flags: 0,
            failure,
            failure_row: row,
            reserved: 0,
        }
    }
    pub fn encode(self) -> [u8; RESULT_BYTES] {
        let mut bytes = [0; RESULT_BYTES];
        bytes[..8].copy_from_slice(&self.epoch.to_le_bytes());
        for (i, word) in [
            self.accepted,
            self.kept,
            self.flags,
            self.failure,
            self.failure_row,
            self.reserved,
        ]
        .into_iter()
        .enumerate()
        {
            bytes[8 + i * 4..12 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, Refusal> {
        if bytes.len() != RESULT_BYTES {
            return Err(Refusal("DSpark acceptance result size differs"));
        }
        Ok(Self {
            epoch: u64_at(bytes, 0),
            accepted: u32_at(bytes, 8),
            kept: u32_at(bytes, 12),
            flags: u32_at(bytes, 16),
            failure: u32_at(bytes, 20),
            failure_row: u32_at(bytes, 24),
            reserved: u32_at(bytes, 28),
        })
    }
}

kernel_core::scratch_record! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AcceptanceBuffers {
        pub items: value,
        pub proposals: value,
        pub samples: value,
        pub stops: value,
        pub results: value,
    }
}

pub struct Input<'a> {
    pub items: &'a [Item],
    pub proposals: &'a [[u32; BLOCK]],
    pub samples: &'a [u8],
    pub stops: &'a [u32],
}

pub fn oracle(op: Acceptance, input: Input<'_>) -> Result<Vec<Decision>, Refusal> {
    op.validate(Need::rows(
        u32::try_from(input.items.len()).map_err(|_| Refusal("too many items"))?,
    ))?;
    if input.proposals.len() < op.draft_items as usize
        || input.samples.len() < op.sample_rows as usize * OUTPUT_BYTES
        || input.stops.len() < op.stop_ids as usize
    {
        return Err(Refusal("DSpark acceptance oracle operands are short"));
    }
    Ok(input
        .items
        .iter()
        .enumerate()
        .map(|(index, &item)| {
            let fail = |failure, row| Decision::failed(item.epoch, failure, row);
            let preceding = &input.items[..index];
            let first = preceding
                .last()
                .map_or(0, |p| u64::from(p.first_sample) + u64::from(p.k) + 1);
            let draft = preceding.iter().filter(|p| p.k > 0).count();
            if item.reserved != 0
                || item.remaining == 0
                || item.k > BLOCK as u32
                || u64::from(item.first_sample) != first
                || u64::from(item.first_sample) + u64::from(item.k) + 1 > u64::from(op.sample_rows)
                || u64::from(item.stop_offset) + u64::from(item.stop_count) > u64::from(op.stop_ids)
                || if item.k == 0 {
                    item.draft_item != NONE
                } else {
                    item.draft_item as usize != draft || item.draft_item >= op.draft_items
                }
            {
                return fail(BAD_DESCRIPTOR, NONE);
            }
            let stops = &input.stops[item.stop_offset as usize..][..item.stop_count as usize];
            if stops.iter().any(|&id| id >= op.vocab) || stops.windows(2).any(|w| w[0] >= w[1]) {
                return fail(BAD_DESCRIPTOR, NONE);
            }
            let proposed = (item.k > 0).then(|| &input.proposals[item.draft_item as usize]);
            if proposed.is_some_and(|ids| ids.iter().any(|&id| id >= op.vocab)) {
                return fail(BAD_PROPOSAL, NONE);
            }
            let sample = |r: u32| {
                &input.samples[(item.first_sample + r) as usize * OUTPUT_BYTES..][..OUTPUT_BYTES]
            };
            for r in 0..=item.k {
                let bytes = sample(r);
                match u32_at(bytes, 4) {
                    STATUS_EMPTY => {}
                    STATUS_SAMPLED
                        if u32_at(bytes, 0) < op.vocab
                            && f64::from_bits(u64_at(bytes, 8)).is_finite() => {}
                    _ => return fail(BAD_SAMPLE, item.first_sample + r),
                }
            }
            let mut result = fail(OK, NONE);
            for r in 0..=item.k {
                let bytes = sample(r);
                if u32_at(bytes, 4) == STATUS_EMPTY {
                    return fail(EMPTY, item.first_sample + r);
                }
                let token = u32_at(bytes, 0);
                let matched = r < item.k && proposed.expect("draft position")[r as usize] == token;
                result.accepted += u32::from(matched);
                result.kept += 1;
                result.flags = (u32::from(stops.binary_search(&token).is_ok()) * TERMINAL)
                    | (u32::from(r < item.k && !matched) * REJECTED);
                if result.flags & TERMINAL != 0
                    || u64::from(result.kept) == item.remaining
                    || !matched
                {
                    break;
                }
            }
            result
        })
        .collect())
}

pub fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("checked U32 span"))
}
pub fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("checked U64 span"))
}

pub mod allocated;

#[cfg(test)]
mod tests;
