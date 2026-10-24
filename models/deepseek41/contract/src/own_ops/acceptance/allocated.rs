//! Acceptance of an allocated verify step (`own_ops::allocation`), in the
//! host result encoding. Each descriptor keeps its reservation: `k` is the
//! item's reserved drafts c_i, which place its 1 + c_i sample rows, and the
//! allocation's verified drafts K'_i (at most c_i) bound the walk. Every
//! reserved row is validated, as rows past 1 + K'_i repeat the last verified
//! row's distribution. A result the host may publish (OK or EMPTY) sets
//! `ALLOCATED` and carries K'_i in its last word; a failed allocation fails
//! every item's descriptor.
use super::{Acceptance, BLOCK, Decision, Input, NONE, REJECTED, TERMINAL, u32_at, u64_at};
use kernel_core::backend::Need;
use kernel_core::contract::Refusal;
use kernel_core::sampling::{OUTPUT_BYTES, STATUS_EMPTY, STATUS_SAMPLED};

pub const CONTRACT: &str = "deepseek41.dspark-allocated-acceptance.u32.v1";
/// `runtime_bundle_abi::device_results::BUNDLE_ACCEPTANCE_ALLOCATED_V1`.
pub const ALLOCATED: u32 = 4;
/// The host encoding's failures (`BundleAcceptanceResultV1::status`).
pub const OK: u32 = 0;
pub const BAD_DESCRIPTOR: u32 = 1;
pub const EMPTY: u32 = 2;
pub const BAD_SAMPLE: u32 = 3;
pub const BAD_PROPOSAL: u32 = 4;

kernel_core::scratch_record! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AllocatedAcceptanceBuffers {
        pub items: value,
        pub verified: value,
        pub status: value,
        pub proposals: value,
        pub samples: value,
        pub stops: value,
        pub results: value,
    }
}

/// The decisions of `input`'s items, whose allocation wrote `status`
/// (`allocation::OK` or a failure) and each item's `verified` drafts.
pub fn oracle(
    op: Acceptance,
    input: Input<'_>,
    verified: &[u32],
    status: u32,
) -> Result<Vec<Decision>, Refusal> {
    op.validate(Need::rows(
        u32::try_from(input.items.len()).map_err(|_| Refusal("too many items"))?,
    ))?;
    if input.proposals.len() < op.draft_items as usize
        || input.samples.len() < op.sample_rows as usize * OUTPUT_BYTES
        || input.stops.len() < op.stop_ids as usize
        || verified.len() < input.items.len()
    {
        return Err(Refusal(
            "DSpark allocated acceptance oracle operands are short",
        ));
    }
    Ok(input
        .items
        .iter()
        .zip(verified)
        .enumerate()
        .map(|(index, (&item, &verified))| {
            let fail = |failure, row| Decision::failed(item.epoch, failure, row);
            let preceding = &input.items[..index];
            let first = preceding
                .last()
                .map_or(0, |p| u64::from(p.first_sample) + u64::from(p.k) + 1);
            let draft = preceding.iter().filter(|p| p.k > 0).count();
            if status != super::super::allocation::OK
                || verified > item.k
                || item.reserved != 0
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
            let mut result = Decision {
                flags: ALLOCATED,
                reserved: verified,
                ..fail(OK, NONE)
            };
            for r in 0..=verified {
                let bytes = sample(r);
                if u32_at(bytes, 4) == STATUS_EMPTY {
                    return Decision {
                        flags: ALLOCATED,
                        reserved: verified,
                        ..fail(EMPTY, r)
                    };
                }
                let token = u32_at(bytes, 0);
                let matched =
                    r < verified && proposed.expect("draft position")[r as usize] == token;
                result.accepted += u32::from(matched);
                result.kept += 1;
                result.flags = ALLOCATED
                    | (u32::from(stops.binary_search(&token).is_ok()) * TERMINAL)
                    | (u32::from(r < verified && !matched) * REJECTED);
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

#[cfg(test)]
mod tests;
