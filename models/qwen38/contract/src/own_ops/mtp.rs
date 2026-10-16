//! The MTP drafter's own rows (`cuda/qwen38_speculation.cuh` in the bundle):
//! its input rows and tokens, its two input norms, the token row added to
//! every stream, and the draft argmax over the vocabulary shards with the
//! drafts scattered to their slots and verify rows.

use std::collections::BTreeSet;

use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Refusal, Validate};
use kernel_core::oracle::{NormOffset, RmsNorm};

use super::{Bounded, copy, entry, sum_bf16};
use crate::contract::Shape;

/// The drafter's input rows: row `i` copies the target's `wide`-value X row
/// `sources[i]` of `last` when that is at least zero, else entry
/// `-1 - sources[i]` of the retained rows; when `token_sources[i]` is at
/// least zero the row's token becomes `drafts[token_sources[i]]`. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpGather {
    pub wide: u32,
}

impl MtpGather {
    pub fn of(shape: Shape) -> MtpGather {
        MtpGather { wide: shape.wide() }
    }
}

impl Validate for MtpGather {
    fn validate(&self) -> Result<(), Refusal> {
        if self.wide == 0 {
            return Err(Refusal("an MTP gather needs row values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An MTP gather's buffers: the target's last X rows, the retained X
    /// rows, each row's source and token source, the drafts, the drafter's
    /// X rows, its token ids and the device `u32` count of live rows. The
    /// X rows are copied in 16-byte vectors and start at 16-byte alignment;
    /// the tables and the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MtpGatherBuffers {
        pub last: value,
        pub retained: value,
        pub sources: value,
        pub token_sources: value,
        pub drafts: value,
        pub x: value,
        pub token_ids: value,
        pub live_rows: value,
    }
}

/// What the gather kernel writes over its first `rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn mtp_gather(
    op: &MtpGather,
    last: &[u8],
    retained: &[u8],
    sources: &[i32],
    token_sources: &[i32],
    drafts: &[i32],
    x: &mut [u8],
    token_ids: &mut [i32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let bytes = u64::from(op.wide) * 2;
    for row in 0..rows.min(live_rows) {
        let source = entry(sources, row as usize, "sources")?;
        let to = u64::from(row) * bytes;
        if source >= 0 {
            copy(x, to, last, source as u64 * bytes, bytes)?;
        } else {
            copy(
                x,
                to,
                retained,
                (-1 - i64::from(source)) as u64 * bytes,
                bytes,
            )?;
        }
        let token = entry(token_sources, row as usize, "token_sources")?;
        if token >= 0 {
            let draft = entry(
                drafts,
                token as usize,
                "a token source is outside the drafts",
            )?;
            *token_ids
                .get_mut(row as usize)
                .ok_or(Refusal("token_ids is too small"))? = draft;
        }
    }
    Ok(())
}

/// The drafter's two input norms, Gemma RMSNorm (`x / rms * (1 + w)`): each
/// X row's `hidden * streams` values over all streams together with the
/// `hidden * streams`-entry weight, and each token embedding's `hidden`
/// values with the `hidden`-entry weight, both with epsilon `eps`. All
/// math.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpNorms {
    pub hidden: u32,
    pub streams: u32,
    pub eps: f32,
}

impl MtpNorms {
    pub fn of(shape: Shape) -> MtpNorms {
        MtpNorms {
            hidden: shape.hidden(),
            streams: shape.streams(),
            eps: 1e-6,
        }
    }
}

impl Validate for MtpNorms {
    fn validate(&self) -> Result<(), Refusal> {
        if self.hidden == 0 || self.streams == 0 || self.eps <= 0.0 || !self.eps.is_finite() {
            return Err(Refusal("MTP norms need rows and a positive epsilon"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// MTP norms' buffers: the X rows and their weight, the token
    /// embeddings and their weight, both normalized outputs and the device
    /// `u32` count of live rows. Every view starts at 2-byte alignment, the
    /// count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MtpNormsBuffers {
        pub x: value,
        pub hidden_weight: value,
        pub embed: value,
        pub embed_weight: value,
        pub x_normed: value,
        pub embed_normed: value,
        pub live_rows: value,
    }
}

/// The F64 norms of `rows` rows, each element bound by one BF16 rounding
/// plus F32 accumulation slack: `|reference| * (2^-8 + 1e-5) + 1e-6`. The
/// kernels' block reduction order and `rsqrtf` are SM120's, so the
/// reference is not bitwise.
pub fn mtp_norms(
    op: &MtpNorms,
    x: &[u16],
    hidden_weight: &[u16],
    embed: &[u16],
    embed_weight: &[u16],
    rows: u32,
) -> Result<[Bounded; 2], Refusal> {
    op.validate()?;
    let norm = |values: &[u16], weight: &[u16], width: u32| -> Result<Bounded, Refusal> {
        let count = rows as usize * width as usize;
        let values = values
            .get(..count)
            .ok_or(Refusal("an MTP norm input is too small"))?;
        if weight.len() != width as usize {
            return Err(Refusal("an MTP norm weight has the wrong width"));
        }
        let widen = |bits: &[u16]| {
            bits.iter()
                .map(|&v| f64::from(bf16_to_f32(v)))
                .collect::<Vec<_>>()
        };
        let reference = RmsNorm::f64(f64::from(op.eps), NormOffset::OnePlusWeight).apply(
            &widen(values),
            &widen(weight),
            width as usize,
        )?;
        let bounds = reference
            .values
            .iter()
            .map(|value| value.abs() * (2_f64.powi(-8) + 1e-5) + 1e-6)
            .collect();
        Ok(Bounded {
            values: reference.values,
            bounds,
        })
    };
    Ok([
        norm(x, hidden_weight, op.hidden * op.streams)?,
        norm(embed, embed_weight, op.hidden)?,
    ])
}

/// `x[r][s] = hidden[r][s] + token[r]` over `streams` streams of `hidden`
/// values: summed in F32 and rounded once to BF16 (ties to even). All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpCombine {
    pub hidden: u32,
    pub streams: u32,
}

impl MtpCombine {
    pub fn of(shape: Shape) -> MtpCombine {
        MtpCombine {
            hidden: shape.hidden(),
            streams: shape.streams(),
        }
    }
}

impl Validate for MtpCombine {
    fn validate(&self) -> Result<(), Refusal> {
        if self.hidden == 0 || self.streams == 0 {
            return Err(Refusal("an MTP combine needs streams of hidden values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An MTP combine's buffers: the hidden rows, the token rows, the
    /// output rows and the device `u32` count of live rows. Every view
    /// starts at 2-byte alignment, the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct MtpCombineBuffers {
        pub hidden: value,
        pub token: value,
        pub x: value,
        pub live_rows: value,
    }
}

/// What the combine kernel computes over its first `rows` rows.
pub fn mtp_combine(
    op: &MtpCombine,
    hidden: &[u16],
    token: &[u16],
    x: &mut [u16],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let (width, streams) = (op.hidden as usize, op.streams as usize);
    let live = rows.min(live_rows) as usize;
    if hidden.len() < live * streams * width
        || token.len() < live * width
        || x.len() < live * streams * width
    {
        return Err(Refusal("an MTP combine buffer is too small"));
    }
    for index in 0..live * streams * width {
        let (row, column) = (index / (streams * width), index % width);
        x[index] = sum_bf16(&[hidden[index], token[row * width + column]]);
    }
    Ok(())
}

/// Each head row's best logit of rank `rank`'s `shard_vocab`-entry shard:
/// four words, the F32 score's bits, the global id `rank * shard_vocab +
/// local`, then two zeros. The best is the highest score and, among equal
/// scores, the lowest id (+0 and -0 equal, the lower id's bits kept); NaN
/// never wins; a row without a winner (every logit NaN or negative
/// infinity) gives negative infinity at local id 0. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftArgmaxPartial {
    pub shard_vocab: u32,
    pub rank: u32,
}

impl DraftArgmaxPartial {
    pub fn of(shape: Shape, rank: u32) -> DraftArgmaxPartial {
        DraftArgmaxPartial {
            shard_vocab: shape.vocab_per_rank,
            rank,
        }
    }
}

impl Validate for DraftArgmaxPartial {
    fn validate(&self) -> Result<(), Refusal> {
        if self.shard_vocab == 0
            || u64::from(self.rank + 1) * u64::from(self.shard_vocab) > u64::from(u32::MAX)
        {
            return Err(Refusal("a draft argmax needs a shard of 32-bit ids"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A partial draft argmax's buffers: the shard's BF16 logits, the
    /// `[rows][4]` `u32` bests and the device `u32` count of live rows. The
    /// logits start at 2-byte alignment, the bests and the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DraftArgmaxPartialBuffers {
        pub logits: value,
        pub best: value,
        pub live_rows: value,
    }
}

/// What the partial argmax kernel writes over its first `rows` rows.
pub fn draft_argmax_partial(
    op: &DraftArgmaxPartial,
    logits: &[u16],
    best: &mut [u32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let vocab = op.shard_vocab as usize;
    for row in 0..rows.min(live_rows) as usize {
        let values = logits
            .get(row * vocab..(row + 1) * vocab)
            .ok_or(Refusal("draft argmax logits are too small"))?;
        let mut winner = (f32::NEG_INFINITY, None);
        for (local, &bits) in values.iter().enumerate() {
            let score = bf16_to_f32(bits);
            if score > winner.0 {
                winner = (score, Some(local as u32));
            }
        }
        let out = best
            .get_mut(row * 4..row * 4 + 4)
            .ok_or(Refusal("draft argmax bests are too small"))?;
        out.copy_from_slice(&[
            winner.0.to_bits(),
            op.rank * op.shard_vocab + winner.1.unwrap_or(0),
            0,
            0,
        ]);
    }
    Ok(())
}

/// Each head row's draft from every rank's best (`[rows][ranks][4]` words,
/// as [`DraftArgmaxPartial`] writes them): the highest score, the lowest id
/// on a tie, a number over NaN, ranks scanned in order from rank 0. The
/// draft becomes `drafts[slots[row]]` and, where `targets[row]` is at least
/// zero, `token_ids[targets[row]]`. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftArgmaxFinal {
    pub ranks: u32,
}

impl DraftArgmaxFinal {
    pub fn of(shape: Shape) -> DraftArgmaxFinal {
        DraftArgmaxFinal {
            ranks: shape.ranks(),
        }
    }
}

impl Validate for DraftArgmaxFinal {
    fn validate(&self) -> Result<(), Refusal> {
        if self.ranks == 0 {
            return Err(Refusal("a draft argmax needs ranks"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A final draft argmax's buffers: every rank's bests, each row's draft
    /// slot and verify row, the drafts, the verify token ids and the device
    /// `u32` count of live rows. Every view starts at 4-byte alignment.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DraftArgmaxFinalBuffers {
        pub bests: value,
        pub slots: value,
        pub targets: value,
        pub drafts: value,
        pub token_ids: value,
        pub live_rows: value,
    }
}

/// The draft of one row's `ranks` candidates.
pub fn draft_of(candidates: &[u32]) -> Result<u32, Refusal> {
    let candidate = |rank: usize| -> Result<(f32, u32), Refusal> {
        let words = candidates
            .get(4 * rank..4 * rank + 2)
            .ok_or(Refusal("a draft row misses a rank's best"))?;
        Ok((f32::from_bits(words[0]), words[1]))
    };
    let (mut score, mut id) = candidate(0)?;
    for rank in 1..candidates.len() / 4 {
        let (other, other_id) = candidate(rank)?;
        if other > score || (other == score && other_id < id) || (score.is_nan() && !other.is_nan())
        {
            (score, id) = (other, other_id);
        }
    }
    Ok(id)
}

/// What the final argmax kernel writes over its first `rows` rows. Two rows
/// drafting into one slot, or filling one verify row, are refused: the
/// kernel's result would depend on block order.
#[allow(clippy::too_many_arguments)]
pub fn draft_argmax_final(
    op: &DraftArgmaxFinal,
    bests: &[u32],
    slots: &[i32],
    targets: &[i32],
    drafts: &mut [i32],
    token_ids: &mut [i32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let words = op.ranks as usize * 4;
    let (mut drafted, mut filled) = (BTreeSet::new(), BTreeSet::new());
    for row in 0..rows.min(live_rows) as usize {
        let candidates = bests
            .get(row * words..(row + 1) * words)
            .ok_or(Refusal("draft argmax bests are too small"))?;
        let id = draft_of(candidates)? as i32;
        let slot = entry(slots, row, "slots")?;
        if !drafted.insert(slot) {
            return Err(Refusal("two draft rows share a slot"));
        }
        *usize::try_from(slot)
            .ok()
            .and_then(|slot| drafts.get_mut(slot))
            .ok_or(Refusal("a draft slot is outside the drafts"))? = id;
        let target = entry(targets, row, "targets")?;
        if target >= 0 {
            if !filled.insert(target) {
                return Err(Refusal("two draft rows fill one verify row"));
            }
            *token_ids
                .get_mut(target as usize)
                .ok_or(Refusal("a verify row is outside the token ids"))? = id;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::codec::bf16_to_f32;

    #[test]
    fn gathers_copy_last_or_retained_rows_and_draft_tokens() {
        let op = MtpGather { wide: 2 };
        let last = (0..12).collect::<Vec<u8>>();
        let retained = (100..112).collect::<Vec<u8>>();
        let mut x = vec![0xaa; 16];
        let mut tokens = [1, 2, 3, 4];
        mtp_gather(
            &op,
            &last,
            &retained,
            &[2, -3, 0],
            &[-1, 1, 0],
            &[70, 71],
            &mut x,
            &mut tokens,
            3,
            4,
        )
        .unwrap();
        assert_eq!(x[..4], [8, 9, 10, 11]);
        assert_eq!(x[4..8], [108, 109, 110, 111]);
        assert_eq!(x[8..12], [0, 1, 2, 3]);
        assert_eq!(x[12..], [0xaa; 4]);
        assert_eq!(tokens, [1, 71, 70, 4]);
    }

    #[test]
    fn combines_round_each_streams_sum_once() {
        let op = MtpCombine {
            hidden: 2,
            streams: 2,
        };
        let bf = kernel_core::codec::f32_to_bf16;
        let hidden = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(bf);
        let token = [0.5, 0.25, 9.0, 9.0].map(bf);
        let mut x = [0xaaaa; 8];
        mtp_combine(&op, &hidden, &token, &mut x, 1, 2).unwrap();
        assert_eq!(x[..4], [1.5, 2.25, 3.5, 4.25].map(bf));
        assert_eq!(x[4..], [0xaaaa; 4]);
    }

    #[test]
    fn norms_are_gemma_rmsnorm_of_both_inputs() {
        let op = MtpNorms {
            hidden: 2,
            streams: 2,
            eps: 1e-6,
        };
        let bf = kernel_core::codec::f32_to_bf16;
        let x = [1.0, 2.0, 3.0, 4.0].map(bf);
        let embed = [3.0, 4.0].map(bf);
        let [hidden, token] = mtp_norms(
            &op,
            &x,
            &[0.0, 0.5, 0.0, 0.0].map(bf),
            &embed,
            &[0.0, 0.0].map(bf),
            1,
        )
        .unwrap();
        let rms = (30.0_f64 / 4.0 + 1e-6).sqrt();
        assert!((hidden.values[1] - 2.0 / rms * 1.5).abs() < 1e-12);
        let rms = (25.0_f64 / 2.0 + 1e-6).sqrt();
        assert!((token.values[0] - 3.0 / rms).abs() < 1e-12);
        assert!(
            token
                .check_bf16(&[bf((3.0 / rms) as f32), bf((4.0 / rms) as f32)])
                .is_ok()
        );
    }

    /// The highest logit with the lowest id, NaN never winning, and
    /// agreement with kernel-core's sharded argmax on NaN-free shards.
    #[test]
    fn drafts_take_the_best_logit_with_the_lowest_id() {
        let bf = kernel_core::codec::f32_to_bf16;
        let shards = [
            [1.0, 9.0, 9.0, f32::NAN].map(bf),
            [9.0, 2.0, 9.0, 0.0].map(bf),
            [f32::NAN, f32::NEG_INFINITY, f32::NAN, f32::NAN].map(bf),
        ];
        let mut bests = vec![0; 3 * 4];
        for (rank, shard) in shards.iter().enumerate() {
            let op = DraftArgmaxPartial {
                shard_vocab: 4,
                rank: rank as u32,
            };
            let mut best = [7; 4];
            draft_argmax_partial(&op, shard, &mut best, 1, 1).unwrap();
            bests[rank * 4..rank * 4 + 4].copy_from_slice(&best);
        }
        assert_eq!(bests[..4], [9.0_f32.to_bits(), 1, 0, 0]);
        assert_eq!(bests[4..8], [9.0_f32.to_bits(), 4, 0, 0]);
        assert_eq!(bests[8..], [f32::NEG_INFINITY.to_bits(), 8, 0, 0]);
        assert_eq!(draft_of(&bests).unwrap(), 1);
        let widen = |shard: &[u16]| shard.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<_>>();
        let (first, second) = (widen(&shards[0][..3]), widen(&shards[1]));
        let free = [&first[..], &second[..]];
        assert_eq!(
            kernel_core::oracle::sharded_argmax(&free).unwrap(),
            draft_of(&bests[..8]).unwrap() as usize
        );
        // A number over NaN, whatever its rank.
        let nan_first = [f32::NAN.to_bits(), 3, 0, 0, 1.0_f32.to_bits(), 9, 0, 0];
        assert_eq!(draft_of(&nan_first).unwrap(), 9);
        // Scatter to slots and verify rows.
        let op = DraftArgmaxFinal { ranks: 2 };
        let rows = [bests[..8].to_vec(), nan_first.to_vec()].concat();
        let (mut drafts, mut tokens) = ([-7; 3], [-9; 4]);
        draft_argmax_final(
            &op,
            &rows,
            &[2, 0],
            &[3, -1],
            &mut drafts,
            &mut tokens,
            2,
            2,
        )
        .unwrap();
        assert_eq!(drafts, [9, -7, 1]);
        assert_eq!(tokens, [-9, -9, -9, 1]);
        assert!(
            draft_argmax_final(
                &op,
                &rows,
                &[1, 1],
                &[-1, -1],
                &mut drafts,
                &mut tokens,
                2,
                2
            )
            .is_err()
        );
    }
}
