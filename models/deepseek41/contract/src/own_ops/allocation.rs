//! Verify allocation (`chop.verify-allocation.v1`): after drafting, each
//! decode-phase item verifies a prefix of its reserved drafts, chosen by the
//! SDK's rule ([`runtime_bundle_sdk::allocate_verify_drafts_v1`]) from the
//! drafter's confidence, so the target computes exactly the step's width.
//! The op computes each item's verified drafts and the maps between the
//! reserved layout (one row plus `reserved` drafts per item, back to back)
//! and the computed layout (one row plus its verified drafts per item):
//! `rows[w]`, the reserved row computed row `w` verifies, and `expand[r]`,
//! the computed row reserved row `r` takes, its item's last verified row
//! past its verified drafts. Integers and the rule's rounded f32 sigmoid
//! only, so every rank and the oracle agree bit for bit.
use kernel_core::backend::Need;
use kernel_core::contract::Refusal;
use runtime_bundle_sdk::{
    SamplingKindV1, VerifyAllocationV1, VerifyCalibrationV1, allocate_verify_drafts_v1,
};

pub const CONTRACT: &str = "deepseek41.verify-allocation.f32-rn.v1";
/// Draft positions a block reserves at most.
pub const BLOCK: usize = 5;
/// Per item: reserved drafts, kind (0 greedy, 1 sampled), its confidence
/// row (`NONE` without drafts) and its first reserved row.
pub const ITEM_WORDS: usize = 4;
/// Width, floor, then `(scale, offset)` f32 bits per position for greedy
/// items and then for sampled ones.
pub const RULE_WORDS: usize = 2 + 4 * BLOCK;
pub const NONE: u32 = u32::MAX;
/// Status words: the allocation was made, or its inputs were inconsistent
/// and nothing past the status was written.
pub const OK: u32 = 0;
pub const BAD_INPUT: u32 = 1;

/// Capacities: items, reserved rows and computed rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VerifyAllocate {
    pub items: u32,
    pub reserved_rows: u32,
    pub width: u32,
}

impl VerifyAllocate {
    pub fn validate(self, need: Need) -> Result<(), Refusal> {
        if self.items == 0
            || self.items > 32
            || self.reserved_rows < self.items
            || self.reserved_rows > self.items * (1 + BLOCK as u32)
            || self.width < self.items
            || self.width > self.reserved_rows
            || need.capacity == 0
            || need.capacity > self.items
        {
            return Err(Refusal("verify allocation capacity is invalid"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// U32 items `[items, ITEM_WORDS]`, F32 confidence `[draft rows,
    /// BLOCK]`, U32 rule `[RULE_WORDS]` and the device U32 live item count;
    /// outputs U32 status `[1]`, verified drafts and computed first rows
    /// `[items]`, `rows` `[width]` and `expand` `[reserved_rows]`. No output
    /// aliases an input.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct VerifyAllocateBuffers {
        pub items: value,
        pub scores: value,
        pub rule: value,
        pub live_items: value,
        pub status: value,
        pub verified: value,
        pub starts: value,
        pub rows: value,
        pub expand: value,
    }
}

/// One decode-phase item as the op reads it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Item {
    pub reserved: u32,
    pub kind: SamplingKindV1,
    pub score_row: u32,
    pub first_row: u32,
}

impl Item {
    pub fn encode(self) -> [u32; ITEM_WORDS] {
        [
            self.reserved,
            u32::from(self.kind == SamplingKindV1::Sampled),
            self.score_row,
            self.first_row,
        ]
    }
}

/// The rule record of an SDK allocation; positions past its calibration
/// are zero (no reservation reaches them).
pub fn encode_rule(allocation: &VerifyAllocationV1) -> [u32; RULE_WORDS] {
    let mut words = [0; RULE_WORDS];
    words[0] = allocation.width();
    words[1] = allocation.floor();
    let calibration = allocation.calibration();
    for (k, kind) in [SamplingKindV1::Greedy, SamplingKindV1::Sampled]
        .into_iter()
        .enumerate()
    {
        for position in 0..calibration.positions().min(BLOCK) {
            let (scale, offset) = calibration.pair(kind, position);
            words[2 + k * 2 * BLOCK + 2 * position] = scale.to_bits();
            words[3 + k * 2 * BLOCK + 2 * position] = offset.to_bits();
        }
    }
    words
}

/// What the op writes after an OK status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Allocation {
    pub verified: Vec<u32>,
    pub starts: Vec<u32>,
    pub rows: Vec<u32>,
    pub expand: Vec<u32>,
}

fn check(rule: &[u32; RULE_WORDS], items: &[Item], scores: &[[f32; BLOCK]]) -> Result<(), Refusal> {
    let mut next = 0u32;
    let mut drafting = 0u32;
    for item in items {
        let consistent = item.reserved as usize <= BLOCK
            && item.first_row == next
            && if item.reserved == 0 {
                item.score_row == NONE
            } else {
                item.score_row == drafting && (item.score_row as usize) < scores.len()
            };
        if !consistent {
            return Err(Refusal("verify allocation items are inconsistent"));
        }
        drafting += u32::from(item.reserved > 0);
        next += 1 + item.reserved;
    }
    if items.is_empty() || rule[0] < items.len() as u32 || rule[0] > next {
        return Err(Refusal("verify allocation width is outside its items"));
    }
    Ok(())
}

fn layout(items: &[Item], verified: Vec<u32>) -> Allocation {
    let mut starts = Vec::with_capacity(items.len());
    let (mut rows, mut expand) = (Vec::new(), Vec::new());
    for (item, &k) in items.iter().zip(&verified) {
        let start = rows.len() as u32;
        starts.push(start);
        rows.extend((0..=k).map(|offset| item.first_row + offset));
        expand.extend((0..=item.reserved).map(|offset| start + offset.min(k)));
    }
    Allocation {
        verified,
        starts,
        rows,
        expand,
    }
}

/// The oracle: the SDK rule's verified drafts and their layout.
pub fn allocate_reference(
    rule: &[u32; RULE_WORDS],
    items: &[Item],
    scores: &[[f32; BLOCK]],
) -> Result<Allocation, Refusal> {
    check(rule, items, scores)?;
    let pairs = |kind: usize| -> Vec<(f32, f32)> {
        (0..BLOCK)
            .map(|position| {
                let base = 2 + kind * 2 * BLOCK + 2 * position;
                (f32::from_bits(rule[base]), f32::from_bits(rule[base + 1]))
            })
            .collect()
    };
    let calibration = VerifyCalibrationV1::new(&pairs(0), &pairs(1))
        .map_err(|_| Refusal("verify allocation calibration is not finite"))?;
    let kinds: Vec<_> = items.iter().map(|item| item.kind).collect();
    let allocation = VerifyAllocationV1::new(rule[0], rule[1], calibration, kinds.clone());
    let reserved: Vec<u32> = items.iter().map(|item| item.reserved).collect();
    let none = [0.0; BLOCK];
    let rows: Vec<&[f32]> = items
        .iter()
        .map(|item| {
            if item.reserved == 0 {
                &none[..]
            } else {
                &scores[item.score_row as usize][..]
            }
        })
        .collect();
    let verified = allocate_verify_drafts_v1(&allocation, &reserved, &rows, &kinds);
    Ok(layout(items, verified))
}

/// The kernel's schedule on the CPU: each candidate's survival is its own
/// product through its position and its rank is the count of candidates
/// before it in (survival descending, item, position) order; a candidate
/// is verified when its rank is below the rows left after the floor.
pub fn allocate_ranked(
    rule: &[u32; RULE_WORDS],
    items: &[Item],
    scores: &[[f32; BLOCK]],
) -> Result<Allocation, Refusal> {
    check(rule, items, scores)?;
    let (width, floor) = (rule[0], rule[1]);
    let fits = |f: u32| {
        items
            .iter()
            .map(|item| 1 + u64::from(f.min(item.reserved)))
            .sum::<u64>()
            <= u64::from(width)
    };
    let floor = (0..=floor).rev().find(|&f| fits(f)).unwrap_or(0);
    let used: u64 = items
        .iter()
        .map(|item| 1 + u64::from(floor.min(item.reserved)))
        .sum();
    let left = u64::from(width).saturating_sub(used);
    let survival = |i: usize, j: usize| {
        let item = items[i];
        let kind = usize::from(item.kind == SamplingKindV1::Sampled);
        let mut survival = 1.0_f32;
        for position in 0..=j {
            let base = 2 + kind * 2 * BLOCK + 2 * position;
            let (scale, offset) = (f32::from_bits(rule[base]), f32::from_bits(rule[base + 1]));
            let score = scores[item.score_row as usize][position];
            survival *= runtime_bundle_sdk::sigmoid_v1(scale * score + offset);
        }
        survival
    };
    let candidates: Vec<(usize, usize, f32)> = items
        .iter()
        .enumerate()
        .flat_map(|(i, item)| (floor..item.reserved).map(move |j| (i, j as usize)))
        .map(|(i, j)| (i, j, survival(i, j)))
        .collect();
    let mut verified: Vec<u32> = items.iter().map(|item| floor.min(item.reserved)).collect();
    for &(i, j, s) in &candidates {
        let rank = candidates
            .iter()
            .filter(|&&(i2, j2, s2)| s2 > s || (s2 == s && (i2, j2) < (i, j)))
            .count() as u64;
        if rank < left {
            verified[i] += 1;
        }
    }
    Ok(layout(items, verified))
}

#[cfg(test)]
mod tests;
