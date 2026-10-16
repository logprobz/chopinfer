//! MTP layer 0 mapped to the qualified target projection and SWA stages.

use kernel_core::contract::Refusal;
use model_mimo26_contract::own_ops::retain::{History, RetainRows};
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::program::Stage;
use crate::program_lower::{Lowered, RankStage};
use crate::selection::MimoPhase as _;
use crate::topology::Tp;

/// Exact audited cubins used by the bundle, for the kernel-lab layer gate.
pub fn qualification_cubins() -> impl Iterator<Item = (&'static [u8], &'static [&'static str])> {
    crate::CUBINS
        .iter()
        .map(|cubin| (cubin.bytes, cubin.symbols))
}

/// The caller supplies MTP aliases for target weight names, a separate ring,
/// and a zero residual for the initial EH reduction.
pub const BLOCK: [Stage; 11] = [
    Stage::AttentionReduceNorm(0),
    Stage::Qkv(1),
    Stage::RopeForSlidingWindow(1),
    Stage::AttentionPartial(1),
    Stage::AttentionMerge(1),
    Stage::SlidingWindowWrite(1),
    Stage::AttentionOutput(1),
    Stage::AttentionReduceNorm(1),
    Stage::DenseGateUp(0),
    Stage::DenseDown(0),
    Stage::MlpReduceNorm(0),
];

/// Whether `rank`'s EH K shard reads the hidden state (the second half of
/// cat[embedding, hidden]) rather than the embedding, and which
/// `mtp_eh_k`-wide shard of that 4096-wide source it packs.
pub fn eh_shard(tp: crate::topology::Tp, rank: u32) -> (bool, u32) {
    let (hidden, offset) = tp.mtp_eh_source(rank as usize);
    (hidden, (offset / tp.mtp_eh_k()) as u32)
}

/// Normalize the rank's half of cat[embedding, hidden], then slice its K
/// shard (`Tp::mtp_eh_slice`) into `packed`.
pub fn pair(
    rank: &RankStage<'_>,
    hidden: DeviceBufferSpanV1,
    norm_weight: DeviceBufferSpanV1,
    packed: DeviceBufferSpanV1,
) -> Result<Vec<Lowered>, Refusal> {
    let capacity = rank.phase.capacity()?;
    let columns = rank.tp.mtp_eh_k();
    if rank.rank >= rank.tp.ranks() || packed.length < u64::from(capacity) * columns * 2 {
        return Err(Refusal("MTP pair shard geometry"));
    }
    let (reads_hidden, _) = eh_shard(rank.tp, rank.rank);
    let mut result = Vec::new();
    if !reads_hidden {
        result.extend(rank.lower_with_facts(Stage::Embedding)?);
    }
    result.push(rank.ops.hidden_norm(
        rank.phase,
        rank.bucket,
        rank.switches,
        rank.live_tokens,
        if reads_hidden {
            hidden
        } else {
            rank.activation
        },
        norm_weight,
        rank.normalized,
        rank.inputs.live_tokens,
    )?);
    result.push(rank.ops.mtp_eh_slice(
        rank.rank,
        rank.phase,
        rank.bucket,
        rank.normalized,
        packed,
        rank.inputs.live_tokens,
    )?);
    Ok(result)
}

/// The write of every decode and verify row's final normalized hidden
/// state into its slot's history: whole rows, in `DRAFT_ITEMS` slots of the
/// window plus the retained slack rows.
pub fn retain(tp: Tp) -> RetainRows {
    let hidden = tp.geometry.hidden();
    RetainRows {
        width: hidden,
        input_stride: hidden,
        history: History {
            stride: hidden,
            capacity: crate::retain::capacity(tp.geometry.window() as u64) as u32,
            slots: crate::contract::DRAFT_ITEMS,
        },
    }
}

pub fn eh(rank: &RankStage<'_>, packed: DeviceBufferSpanV1) -> Result<Lowered, Refusal> {
    let mut calls = rank.bound_projection(
        "mtp.eh.0.bf16",
        rank.phase,
        rank.live_tokens,
        packed,
        None,
        rank.scratch.hidden_partial,
    )?;
    if calls.len() != 1 {
        return Err(Refusal("MTP EH projection stage count differs"));
    }
    Ok(calls.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first half of the ranks read the embedding, the second the hidden
    /// state, each at consecutive shards; TP4 is ranks 0,1 then 2,3.
    #[test]
    fn eh_shards_split_the_embedding_then_the_hidden_state() {
        let shards = |tp: Tp| {
            (0..tp.ranks())
                .map(|rank| eh_shard(tp, rank))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            shards(Tp::TP4),
            [(false, 0), (false, 1), (true, 0), (true, 1)]
        );
        assert_eq!(shards(Tp::TP2), [(false, 0), (true, 0)]);
    }
}
