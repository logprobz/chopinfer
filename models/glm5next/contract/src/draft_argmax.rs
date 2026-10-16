//! The MTP drafter's argmax over the head's vocabulary shards, on the
//! shared sharded argmax (`RowQuery::ShardedArgmaxPartial` and
//! `RowQuery::ShardedArgmaxFinal`): each rank's partial over its F32
//! logits shard writes one (score, id) pair per draft, two all-gathers put
//! every rank's scores and ids on every rank in rank order, and each rank's
//! final writes each draft's token to its drafting item's latest draft and
//! to its verify row's input token.
//!
//! The order is the contract's: larger logits first, -0 equal to +0, a NaN
//! of any sign or payload below every other value (-inf included), equal
//! logits to the lower global id; so a row of only NaN drafts global id 0,
//! always in the vocabulary. Every rank computes the same tokens.

use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::collective::GatherBuffers;
use kernel_core::contract::{ArgmaxLayout, CommQuery, DType, Refusal, RowQuery};
use kernel_core::oracle;
use kernel_core::properties::Invariance;
use kernel_core::rows::{ArgmaxFinalBuffers, ArgmaxPairs, ArgmaxPartialBuffers, ArgmaxScatter};

use crate::contract::{DRAFT_ITEMS, Shape};

/// The drafts a launch serves at most, and the pairs each rank gathers:
/// one per drafter item.
pub const CAPACITY: u32 = DRAFT_ITEMS;

/// Rank `rank`'s argmax over its shard of `shape`'s head: the F32 logits of
/// `vocab_rows` global ids from `rank * vocab_rows`.
pub fn partial_query(shape: Shape) -> RowQuery {
    RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: shape.profile.vocab_rows,
        ranks: shape.ranks(),
        logits: DType::F32,
    }
}

/// The all-gather of one word of each rank's [`CAPACITY`] pairs: the F32
/// scores, and the I32 ids moved as their bits.
pub fn gather_query(shape: Shape) -> CommQuery {
    CommQuery::AllGather {
        ranks: shape.ranks(),
        elements_per_rank: u64::from(CAPACITY),
        dtype: DType::F32,
    }
}

/// The final over every rank's gathered pairs, rank-major at [`CAPACITY`].
pub fn final_query(shape: Shape) -> RowQuery {
    RowQuery::ShardedArgmaxFinal {
        ranks: shape.ranks(),
        pairs: ArgmaxLayout::RankMajor { stride: CAPACITY },
    }
}

kernel_core::scratch_record! {
    /// One rank's buffers of the draft argmax: its F32 `logits` shard rows
    /// `[drafts, vocab_rows]` and the device `u32` count of their live
    /// columns (the whole shard), the all-gathers of the pairs' F32
    /// `scores` and I32 `ids` (each gather's input is the rank's
    /// [`CAPACITY`] pairs, each rank's output its `[ranks, CAPACITY]`
    /// gathered pairs), each draft's I32 drafting item in `items` and the
    /// `u32` `drafts` they index, each draft's I32 verify row in
    /// `verify_rows` and the `u32` verify input tokens they index when
    /// present, and the device `u32` count of live drafts. The live drafts'
    /// items are distinct, as are their verify rows; an index outside its
    /// buffer writes nothing.
    #[derive(Clone, Debug)]
    pub struct DraftArgmaxBuffers {
        pub logits: value,
        pub live_width: value,
        pub scores: record(GatherBuffers),
        pub ids: record(GatherBuffers),
        pub items: value,
        pub drafts: value,
        pub verify_rows: value,
        pub verify_tokens: option,
        pub live_drafts: value,
    }
}

/// One rank's draft argmax, prepared once: its partial, the two gathers
/// and its final, launched in that order. Launches read `Counts::rows`
/// (the drafts the launch grids, at most [`CAPACITY`]); the gathers move
/// their whole blocks at any count.
pub struct DraftArgmax {
    rank: u32,
    partial: Prepared<ArgmaxPartialBuffers<View>>,
    scores: Prepared<GatherBuffers<View>>,
    ids: Prepared<GatherBuffers<View>>,
    last: Prepared<ArgmaxFinalBuffers<View>>,
    info: OpInfo,
}

impl DraftArgmax {
    /// Rank `rank`'s draft argmax of `shape` on `backend`.
    pub fn prepare(backend: &dyn Backend, shape: Shape, rank: u32) -> Result<Self, Refused> {
        Self::prepare_for(backend, shape, rank, Need::rows(CAPACITY))
    }

    /// [`Self::prepare`], refused unless what its parts guarantee together
    /// provides `need`'s exactness, of at most [`CAPACITY`] drafts.
    pub fn prepare_for(
        backend: &dyn Backend,
        shape: Shape,
        rank: u32,
        need: Need,
    ) -> Result<Self, Refused> {
        if need.capacity == 0 || need.capacity > CAPACITY {
            return Err(Refused::Rule(format!(
                "a draft argmax serves 1 to {CAPACITY} drafts"
            )));
        }
        let rows = Need::rows(need.capacity);
        let partial = backend.sharded_argmax_partial(&partial_query(shape), rank, rows)?;
        let gather = gather_query(shape);
        let scores = backend.all_gather(&gather, rank, None, Need::rows(1))?;
        let ids = backend.all_gather(&gather, rank, None, Need::rows(1))?;
        let last = backend.sharded_argmax_final(&final_query(shape), rows)?;
        let parts = [partial.info(), scores.info(), ids.info(), last.info()];
        let guarantees = parts
            .iter()
            .map(|info| info.guarantees)
            .reduce(Invariance::meet)
            .unwrap_or(Invariance::NONE);
        let mut info = OpInfo::new(
            "glm5next.draft-argmax",
            guarantees,
            parts.iter().flat_map(|info| info.symbols.clone()).collect(),
        );
        info.state = [scores.info(), ids.info()]
            .iter()
            .flat_map(|info| info.state.clone())
            .collect();
        need.check(&info)?;
        Ok(Self {
            rank,
            partial,
            scores,
            ids,
            last,
            info,
        })
    }
}

impl PreparedOp<DraftArgmaxBuffers<View>> for DraftArgmax {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &DraftArgmaxBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let gathered = |gather: &GatherBuffers<View>| {
            gather
                .ranks
                .get(self.rank as usize)
                .map(|views| views.output)
                .ok_or(Refusal("the draft argmax lacks its rank's gathered pairs"))
        };
        let partial = ArgmaxPartialBuffers {
            logits: buffers.logits,
            live_width: buffers.live_width,
            pairs: ArgmaxPairs {
                scores: buffers.scores.input,
                ids: buffers.ids.input,
                peer: false,
            },
            live_rows: buffers.live_drafts,
        };
        let mut scatter = vec![ArgmaxScatter {
            rows: buffers.items,
            output: buffers.drafts,
        }];
        if let Some(verify_tokens) = buffers.verify_tokens {
            scatter.push(ArgmaxScatter {
                rows: buffers.verify_rows,
                output: verify_tokens,
            });
        }
        let last = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: gathered(&buffers.scores)?,
                ids: gathered(&buffers.ids)?,
                peer: false,
            },
            output: None,
            scatter,
            live_rows: buffers.live_drafts,
        };
        let block = Counts::rows(1);
        let mut launches = self.partial.launches(&partial, View::EMPTY, counts)?;
        launches.extend(self.scores.launches(&buffers.scores, View::EMPTY, block)?);
        launches.extend(self.ids.launches(&buffers.ids, View::EMPTY, block)?);
        launches.extend(self.last.launches(&last, View::EMPTY, counts)?);
        Ok(launches)
    }
}

/// The token of one draft from every rank's shard of its logits, in rank
/// order: the contract's argmax over the whole vocabulary.
pub fn token(shards: &[&[f32]]) -> Result<u32, Refusal> {
    let id = oracle::sharded_argmax(shards)?;
    u32::try_from(id).map_err(|_| Refusal("a draft id exceeds u32"))
}

/// What every rank's launch of `rows` rows writes, from each rank's
/// `[rows, shard]` logits (`logits[rank]`): each live draft's [`token`] at
/// `drafts[items[d]]` and `verify_tokens[verify_rows[d]]`, an index
/// outside its buffer skipped. Two live drafts with one destination are
/// refused (the launch would leave it unspecified).
#[allow(clippy::too_many_arguments)]
pub fn draft_argmax(
    logits: &[&[f32]],
    live_drafts: u32,
    rows: u32,
    items: &[i32],
    drafts: &mut [u32],
    verify_rows: &[i32],
    verify_tokens: Option<&mut [u32]>,
) -> Result<(), Refusal> {
    let live = live_drafts.min(rows) as usize;
    let shard = logits
        .first()
        .map(|rank| rank.len() / rows.max(1) as usize)
        .ok_or(Refusal("a draft argmax needs ranks"))?;
    if shard == 0 || logits.iter().any(|rank| rank.len() < live * shard) {
        return Err(Refusal("a draft argmax's logits are too few"));
    }
    let tokens = (0..live)
        .map(|draft| {
            let row = logits
                .iter()
                .map(|rank| &rank[draft * shard..(draft + 1) * shard])
                .collect::<Vec<_>>();
            token(&row)
        })
        .collect::<Result<Vec<_>, _>>()?;
    scatter(&tokens, items, drafts)?;
    if let Some(verify_tokens) = verify_tokens {
        scatter(&tokens, verify_rows, verify_tokens)?;
    }
    Ok(())
}

/// `output[rows[d]] = tokens[d]` for every index inside `output`.
fn scatter(tokens: &[u32], rows: &[i32], output: &mut [u32]) -> Result<(), Refusal> {
    let mut written = vec![false; output.len()];
    for (draft, &token) in tokens.iter().enumerate() {
        let at = *rows
            .get(draft)
            .ok_or(Refusal("a draft argmax's draft rows are too few"))?;
        let Some(at) = usize::try_from(at).ok().filter(|&at| at < output.len()) else {
            continue;
        };
        if std::mem::replace(&mut written[at], true) {
            return Err(Refusal("two drafts write one entry"));
        }
        output[at] = token;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The queries are valid math, and the final's rank-major stride holds
    /// every draft a launch serves.
    #[test]
    fn queries_validate_and_hold_the_drafts() {
        use kernel_core::contract::Validate;
        let shape = Shape::TP4;
        partial_query(shape).validate().unwrap();
        gather_query(shape).validate().unwrap();
        final_query(shape).validate().unwrap();
        let RowQuery::ShardedArgmaxPartial {
            vocab_per_rank,
            ranks,
            ..
        } = partial_query(shape)
        else {
            unreachable!("a partial query");
        };
        assert_eq!(vocab_per_rank * ranks, shape.geometry.vocab());
    }

    /// The oracle drafts NaN lowest and ties to the lower global id across
    /// ranks, writes both outputs and skips indexes outside them.
    #[test]
    fn the_oracle_drafts_the_contract_order_into_both_outputs() {
        let nan = f32::from_bits(0xffc0_0001);
        // Two ranks of two-column shards, three drafts: a cross-rank tie, a
        // row of only NaN, and -0 against +0 under a NaN.
        let ranks: [&[f32]; 2] = [
            &[1.0, 3.0, nan, f32::NAN, -0.0, nan],
            &[3.0, 2.0, f32::NAN, nan, 0.0, -1.0],
        ];
        let mut drafts = vec![7; 4];
        let mut verify = vec![9; 5];
        draft_argmax(
            &ranks,
            3,
            3,
            &[2, 0, -1],
            &mut drafts,
            &[4, 1, 3],
            Some(&mut verify),
        )
        .unwrap();
        assert_eq!(drafts, [0, 7, 1, 7]);
        assert_eq!(verify, [9, 0, 9, 0, 1]);
        assert!(draft_argmax(&ranks, 2, 3, &[1, 1, 0], &mut drafts, &[0, 1, 2], None).is_err());
    }
}
