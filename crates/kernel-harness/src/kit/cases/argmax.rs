//! Cases of the sharded argmax (`RowQuery::ShardedArgmaxPartial`,
//! `RowQuery::ShardedArgmaxFinal`), with
//! SM120's shared entries (`row::argmax`): one rank's partial over F32 and
//! BF16 logits, the final over given pairs of either layout into a dense
//! and scattered outputs, and on every rank of a group the chain a model
//! runs, partial, an all-gather of the pairs and the final over the
//! rank-major gathered pairs.
//!
//! Rows hold [`LIVE`] live logits of a [`VOCAB`]-column shard, the columns
//! past the live width +inf, which fails any row that reads them. Each row
//! is one of [`KINDS`] kinds, which a rank's rows rotate through: random
//! values of the kit's generator with an F32-only near-tie, ties of the
//! largest value, -0 and +0 winners in both orders, infinities, NaN of
//! both signs, quiet and signaling with payloads, mixed among values, NaN
//! with -inf, -inf with one NaN, and only NaN. Of the chain's rows, the
//! last is only NaN on every rank (global id 0) and the one before ties on
//! every rank. Every result is exact bits: pairs, gathered pairs and ids.
//!
//! A partial and a final over row-major pairs are row ops, so the kit
//! checks the shared entries' claims on them: row, chunk and batch for the
//! partials, row and chunk for the final, whose scattered outputs are
//! in-place state compared at each row's destination. Scattered rows have
//! distinct destinations, the contract's precondition, which the oracle
//! checks. A final over rank-major pairs reads a row's pairs at a stride,
//! so that case leaves the claims unchecked.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::collective::{CollectiveCase, RequiredCollective, Slot};
use crate::kit::{OpCase, Required, Spec, bf16_values};
use crate::poison::WriteMap;
use kernel_core::backend::{
    Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::collective::{GatherBuffers, RankViews};
use kernel_core::contract::{ArgmaxLayout, CommQuery, DType, Refusal, RowQuery};
use kernel_core::oracle;
use kernel_core::properties::Property;
use kernel_core::rows::{ArgmaxFinalBuffers, ArgmaxPairs, ArgmaxPartialBuffers, ArgmaxScatter};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::argmax;

/// The rows every case serves.
pub const CAPACITY: u32 = 16;

/// Columns of one rank's shard: not a multiple of a partial's CTA.
pub const VOCAB: u32 = 1000;

/// Live columns of each row.
pub const LIVE: u32 = 993;

/// The rank counts the cases serve.
pub const RANKS: [u32; 2] = [2, 4];

/// The row kinds a rank's rows rotate through.
pub const KINDS: usize = 10;

/// Elements of each scattered output: a prime, so the case's row maps are
/// distinct.
const SCATTERED: usize = 37;

/// BF16 NaNs: positive and negative, quiet and signaling, with payloads.
const BF16_NANS: [u16; 4] = [0x7fc0, 0xffc0, 0x7f81, 0xffa5];

fn nan(at: usize) -> f32 {
    bf16_to_f32(BF16_NANS[at % BF16_NANS.len()])
}

/// BF16 bits of `value`: NaNs keep their sign and payload, which every NaN
/// here holds in its high half.
fn to_bf16(value: f32) -> u16 {
    if value.is_nan() {
        (value.to_bits() >> 16) as u16
    } else {
        f32_to_bf16(value)
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The live logits of a row of `kind` from `seed`.
fn kind_row(kind: usize, seed: u64) -> Vec<f32> {
    let live = LIVE as usize;
    let mut row = bf16_values(seed, live)
        .into_iter()
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    match kind {
        // Above the generator's values, distinct in F32 and equal in BF16:
        // the F32 row takes the later column, the BF16 row the earlier.
        0 => {
            row[100] = 2.5;
            row[200] = f32::from_bits(2.5_f32.to_bits() + 1);
        }
        1 => [64, 640, 900].into_iter().for_each(|col| row[col] = 4.0),
        2 | 3 => {
            row.iter_mut().for_each(|v| *v = -v.abs());
            let (first, second) = if kind == 2 { (-0.0, 0.0) } else { (0.0, -0.0) };
            row[7] = first;
            row[70] = second;
        }
        4 => {
            row[0] = f32::NEG_INFINITY;
            row[500] = f32::INFINITY;
            row[300] = f32::INFINITY;
        }
        5 => {
            (0..live).step_by(3).for_each(|col| row[col] = nan(col));
            row[401] = 4.0;
        }
        6 => {
            row.iter_mut()
                .enumerate()
                .for_each(|(col, v)| *v = nan(col));
            row[600] = f32::NEG_INFINITY;
            row[800] = f32::NEG_INFINITY;
        }
        7 => {
            row.fill(f32::NEG_INFINITY);
            row[0] = nan(1);
        }
        8 => row
            .iter_mut()
            .enumerate()
            .for_each(|(col, v)| *v = nan(col)),
        _ => {}
    }
    row
}

/// Rank `rank`'s row `row` of a group of `ranks`: [`VOCAB`] F32 values, the
/// live ones of the row's kind and the rest +inf. The chain's last row is
/// only NaN and the one before ties at column 10 on every rank.
fn shard_row(row: usize, rank: u32, ranks: u32) -> Vec<f32> {
    let seed = 0x6172_676d_6178 ^ (u64::from(ranks) << 40) ^ (u64::from(rank) << 20) ^ row as u64;
    let last = CAPACITY as usize - 1;
    let kind = match row {
        _ if row == last => 8,
        _ if row == last - 1 => 9,
        _ => (row + 3 * rank as usize) % KINDS,
    };
    let mut values = kind_row(kind, seed);
    if row == last - 1 {
        values[10] = 8.0;
    }
    values.resize(VOCAB as usize, f32::INFINITY);
    values
}

/// Rank `rank`'s logits image of every row in `logits`' dtype.
fn shard_image(logits: DType, rank: u32, ranks: u32) -> Vec<u8> {
    let values = (0..CAPACITY as usize)
        .flat_map(|row| shard_row(row, rank, ranks))
        .collect::<Vec<_>>();
    match logits {
        DType::Bf16 => values
            .into_iter()
            .flat_map(|v| to_bf16(v).to_le_bytes())
            .collect(),
        _ => f32_bytes(&values),
    }
}

/// The F32 logits of the image at `path`, of `logits`' dtype.
fn logit_values(images: &Images, path: &str, logits: DType) -> LabResult<Vec<f32>> {
    match logits {
        DType::Bf16 => Ok(images.u16s(path)?.into_iter().map(bf16_to_f32).collect()),
        _ => images.f32s(path),
    }
}

/// Each row's partial pair over the live columns of `values` (rows of
/// [`VOCAB`]) on rank `rank`, global ids.
fn partials(values: &[f32], rank: u32) -> LabResult<Vec<(f32, i32)>> {
    values
        .chunks_exact(VOCAB as usize)
        .map(|row| {
            let [(score, local)] = oracle::sharded_argmax_partials(&[&row[..LIVE as usize]])
                .map_err(|e| e.0.to_string())?[..]
            else {
                unreachable!("one shard, one pair");
            };
            Ok((score, (rank * VOCAB) as i32 + local as i32))
        })
        .collect()
}

fn scores(pairs: &[(f32, i32)]) -> Vec<u8> {
    pairs
        .iter()
        .flat_map(|(score, _)| score.to_le_bytes())
        .collect()
}

fn ids(pairs: &[(f32, i32)]) -> Vec<u8> {
    pairs.iter().flat_map(|(_, id)| id.to_le_bytes()).collect()
}

/// The partial query of `ranks` ranks over `logits`.
fn partial_query(logits: DType, ranks: u32) -> RowQuery {
    RowQuery::ShardedArgmaxPartial {
        vocab_per_rank: VOCAB,
        ranks,
        logits,
    }
}

/// The final query of `ranks` ranks over `pairs`.
fn final_query(ranks: u32, pairs: ArgmaxLayout) -> RowQuery {
    RowQuery::ShardedArgmaxFinal { ranks, pairs }
}

/// The case of rank `rank`'s partial of `ranks` ranks over `logits` at
/// [`CAPACITY`] rows: the pairs of the oracle as exact bits.
pub fn partial_case<'a>(
    logits: DType,
    rank: u32,
    ranks: u32,
    prepared: Prepared<ArgmaxPartialBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let roles = ArgmaxPartialBuffers {
        logits: Spec::input(
            logits,
            (Rows::Live, VOCAB as usize),
            shard_image(logits, rank, ranks),
        ),
        live_width: Spec::input(DType::I32, (Rows::Fixed(1), 1), LIVE.to_le_bytes().to_vec()),
        pairs: ArgmaxPairs {
            scores: Spec::output(DType::F32, (Rows::Live, 1)),
            ids: Spec::output(DType::I32, (Rows::Live, 1)),
            peer: false,
        },
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let pairs = partials(&logit_values(images, "logits", logits)?, rank)?;
        Ok(vec![
            ("pairs.scores".into(), Expected::Bits(scores(&pairs))),
            ("pairs.ids".into(), Expected::Bits(ids(&pairs))),
        ])
    };
    Ok(OpCase::new(prepared, CAPACITY, roles)?.oracle(oracle))
}

/// Rank `rank`'s pair of row `row` in a final's case of `ranks` ranks,
/// by the row's kind: distinct scores; a tie of the largest on two ranks;
/// equal scores whose ids fall with the rank; -0 and +0; NaN but one -inf;
/// only NaN, ids falling with the rank; two +inf; NaN at the lowest id.
fn given_pair(row: usize, rank: u32, ranks: u32) -> (f32, i32) {
    let (r, n) = (rank as usize, ranks as usize);
    let rising = (r * VOCAB as usize + (row * 7 + r * 3) % VOCAB as usize) as i32;
    let falling = ((n - 1 - r) * VOCAB as usize + row) as i32;
    let value = bf16_to_f32(bf16_values(0x7061_6972 ^ (row * 8 + r) as u64, 1)[0]);
    match row % 8 {
        0 => (value, rising),
        1 => (if r == 1 || r == n - 1 { 4.0 } else { value }, rising),
        2 => (1.0, falling),
        3 => (
            match r {
                0 => 0.0,
                1 => -0.0,
                _ => -1.0,
            },
            falling,
        ),
        4 => (
            if r == n / 2 {
                f32::NEG_INFINITY
            } else {
                nan(row + r)
            },
            rising,
        ),
        5 => (nan(row + r), falling),
        6 => (if r >= n - 2 { f32::INFINITY } else { value }, rising),
        _ => (if r == 0 { nan(row) } else { value }, rising),
    }
}

/// Where a final's case reads and writes: its pairs' layout, and whether
/// it writes a dense output and the two scattered outputs.
#[derive(Clone, Copy, Debug)]
pub struct Form {
    pub pairs: ArgmaxLayout,
    pub dense: bool,
    pub scattered: bool,
}

/// Row `row`'s index into scattered output `which`: a map of distinct
/// elements, with one negative and one past the output on each.
fn scatter_index(which: usize, row: usize) -> i32 {
    match (which, row) {
        (0, 4) => -2,
        (0, 9) => SCATTERED as i32 + 5,
        (1, 2) => -7,
        (1, 12) => i32::MAX,
        (0, _) => ((row * 5 + 3) % SCATTERED) as i32,
        _ => ((row * 11 + 1) % SCATTERED) as i32,
    }
}

/// The scattered output `which` of a final: its row indexes as an input,
/// and [`SCATTERED`] elements of in-place state holding an image, of which
/// the live rows' indexes inside it are written and the rest kept. As a
/// state, the property checks compare each row's destination.
fn scattered<'a>(which: usize) -> ArgmaxScatter<Spec<'a>> {
    let path = format!("scatter[{which}].rows");
    let indexes = (0..CAPACITY as usize)
        .map(|row| scatter_index(which, row))
        .collect::<Vec<_>>();
    let written = move |images: &Images, live: u32| -> LabResult<WriteMap> {
        let rows = images.i32s(&path)?;
        WriteMap::of(DType::I32).ranges(
            rows[..live as usize]
                .iter()
                .filter_map(|&at| usize::try_from(at).ok().filter(|&at| at < SCATTERED))
                .map(|at| at..at + 1),
        )
    };
    ArgmaxScatter {
        rows: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&indexes)),
        output: Spec::state(
            DType::I32,
            (Rows::Fixed(SCATTERED), 1),
            i32_bytes(
                &(0..SCATTERED as i32)
                    .map(|at| 7_000 + at)
                    .collect::<Vec<_>>(),
            ),
            Write::Map(Box::new(written)),
        ),
    }
}

/// The bytes of scattered output `which` after a launch of `live` rows
/// with the ids of every row: its image, each live row's id at its index
/// inside it. Live rows that share an index are outside the contract
/// (`rows::ArgmaxScatter`), so such a case is refused.
fn scattered_bits(images: &Images, which: usize, ids: &[i32], live: u32) -> LabResult<Vec<u8>> {
    let rows = images.i32s(&format!("scatter[{which}].rows"))?;
    let mut output = images.i32s(&format!("scatter[{which}].output"))?;
    let mut written = vec![false; output.len()];
    for (&at, &id) in rows.iter().zip(ids).take(live as usize) {
        if let Ok(at) = usize::try_from(at)
            && at < output.len()
        {
            if std::mem::replace(&mut written[at], true) {
                return Err(format!(
                    "two live rows scatter to element {at} of scatter[{which}]"
                ));
            }
            output[at] = id;
        }
    }
    Ok(i32_bytes(&output))
}

/// Each row's (score, id) pairs, read from `scores` and `ids` laid out as
/// `pairs`, over `ranks` ranks.
fn rows_of(scores: &[f32], ids: &[i32], ranks: u32, pairs: ArgmaxLayout) -> Vec<Vec<(f32, usize)>> {
    (0..CAPACITY as usize)
        .map(|row| {
            (0..ranks as usize)
                .map(|rank| {
                    let at = match pairs {
                        ArgmaxLayout::RowMajor => row * ranks as usize + rank,
                        ArgmaxLayout::RankMajor { stride } => rank * stride as usize + row,
                    };
                    (scores[at], ids[at] as usize)
                })
                .collect()
        })
        .collect()
}

fn finals(rows: &[Vec<(f32, usize)>]) -> LabResult<Vec<i32>> {
    rows.iter()
        .map(|pairs| {
            oracle::sharded_argmax_final(pairs)
                .map(|id| id as i32)
                .map_err(|e| e.0.to_string())
        })
        .collect()
}

/// The case of a final of `ranks` ranks in `form` at [`CAPACITY`] rows,
/// over given pairs ([`given_pair`]): the oracle's ids as exact bits in
/// each output.
pub fn final_case<'a>(
    ranks: u32,
    form: Form,
    prepared: Prepared<ArgmaxFinalBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (capacity, n) = (CAPACITY as usize, ranks as usize);
    let mut given = vec![(0.0, 0); capacity * n];
    for row in 0..capacity {
        for rank in 0..n {
            let at = match form.pairs {
                ArgmaxLayout::RowMajor => row * n + rank,
                ArgmaxLayout::RankMajor { stride } => rank * stride as usize + row,
            };
            given[at] = given_pair(row, rank as u32, ranks);
        }
    }
    let dims = match form.pairs {
        ArgmaxLayout::RowMajor => (Rows::Live, n),
        ArgmaxLayout::RankMajor { .. } => (Rows::Fixed(n * capacity), 1),
    };
    let roles = ArgmaxFinalBuffers {
        pairs: ArgmaxPairs {
            scores: Spec::input(DType::F32, dims, scores(&given)),
            ids: Spec::input(DType::I32, dims, ids(&given)),
            peer: false,
        },
        output: form
            .dense
            .then(|| Spec::output(DType::I32, (Rows::Live, 1))),
        scatter: if form.scattered {
            vec![scattered(0), scattered(1)]
        } else {
            Vec::new()
        },
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let rows = rows_of(
            &images.f32s("pairs.scores")?,
            &images.i32s("pairs.ids")?,
            ranks,
            form.pairs,
        );
        let ids = finals(&rows)?;
        let mut expected = Vec::new();
        if form.dense {
            expected.push(("output".into(), Expected::Bits(i32_bytes(&ids))));
        }
        if form.scattered {
            for which in 0..2 {
                expected.push((
                    format!("scatter[{which}].output"),
                    Expected::Bits(scattered_bits(images, which, &ids, live)?),
                ));
            }
        }
        Ok(expected)
    };
    let case = OpCase::new(prepared, CAPACITY, roles)?.oracle(oracle);
    Ok(match form.pairs {
        ArgmaxLayout::RowMajor => case,
        ArgmaxLayout::RankMajor { .. } => {
            case.without_row_properties("rank-major pairs lie at a stride, not in rows of the case")
        }
    })
}

/// SM120 with the shared argmax entries.
pub fn sm120() -> Sm120 {
    argmax::entries()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_row_argmax)
}

/// The one-rank ops, each required: the partial over F32 and BF16 logits
/// on the first and last rank of each rank count, and the final of each
/// rank count over row-major pairs into a dense output, over row-major
/// pairs into a dense output and the scattered outputs, and over
/// rank-major pairs into the scattered outputs alone.
pub fn required<'a>() -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mut ops = Vec::new();
    for ranks in RANKS {
        for logits in [DType::F32, DType::Bf16] {
            for rank in [0, ranks - 1] {
                let query = partial_query(logits, ranks);
                ops.push(Required::new(
                    format!("sharded argmax partial {logits:?} rank {rank} of {ranks}"),
                    need,
                    move |backend, need| backend.sharded_argmax_partial(&query, rank, need),
                    move |prepared| partial_case(logits, rank, ranks, prepared),
                ));
            }
        }
        let rank_major = ArgmaxLayout::RankMajor { stride: CAPACITY };
        for form in [
            Form {
                pairs: ArgmaxLayout::RowMajor,
                dense: true,
                scattered: false,
            },
            Form {
                pairs: ArgmaxLayout::RowMajor,
                dense: true,
                scattered: true,
            },
            Form {
                pairs: rank_major,
                dense: false,
                scattered: true,
            },
        ] {
            let query = final_query(ranks, form.pairs);
            ops.push(Required::new(
                format!("sharded argmax final of {ranks} {form:?}"),
                need,
                move |backend, need| backend.sharded_argmax_final(&query, need),
                move |prepared| final_case(ranks, form, prepared),
            ));
        }
    }
    ops.into_iter().map(Required::required).collect()
}

kernel_core::scratch_record! {
    /// One rank's buffers of the argmax chain: its logits shard, its live
    /// width, the all-gathers of its pair scores and ids (each gather's
    /// input is the partial's pairs, each rank's output that rank's
    /// rank-major gathered pairs), the final's dense output and the live
    /// rows.
    #[derive(Clone, Debug)]
    pub struct ChainBuffers {
        pub logits: value,
        pub live_width: value,
        pub scores: record(GatherBuffers),
        pub ids: record(GatherBuffers),
        pub output: value,
        pub live_rows: value,
    }
}

/// One rank's argmax chain, as a model launches it: the partial into its
/// pairs, an all-gather of the scores and one of the ids onto every rank,
/// and the final over its own rank-major gathered pairs.
pub struct Chain {
    rank: u32,
    partial: Prepared<ArgmaxPartialBuffers<View>>,
    scores: Prepared<GatherBuffers<View>>,
    ids: Prepared<GatherBuffers<View>>,
    last: Prepared<ArgmaxFinalBuffers<View>>,
    info: OpInfo,
}

impl Chain {
    /// Rank `rank`'s chain of `ranks` ranks over `logits` at `need`, the
    /// gathers' blocks the pairs of `need.capacity` rows.
    pub fn prepare(
        backend: &dyn Backend,
        logits: DType,
        ranks: u32,
        rank: u32,
        need: Need,
    ) -> Result<Self, Refused> {
        let partial = backend.sharded_argmax_partial(&partial_query(logits, ranks), rank, need)?;
        let gather = CommQuery::AllGather {
            ranks,
            elements_per_rank: u64::from(need.capacity),
            dtype: DType::F32,
        };
        let scores = backend.all_gather(&gather, rank, None, Need::rows(1))?;
        let ids = backend.all_gather(&gather, rank, None, Need::rows(1))?;
        let pairs = ArgmaxLayout::RankMajor {
            stride: need.capacity,
        };
        let last = backend.sharded_argmax_final(&final_query(ranks, pairs), need)?;
        let parts = [partial.info(), scores.info(), ids.info(), last.info()];
        let guarantees = parts
            .iter()
            .map(|info| info.guarantees)
            .reduce(kernel_core::properties::Invariance::meet)
            .expect("four parts");
        let mut info = OpInfo::new(
            "kit.argmax-chain",
            guarantees,
            parts.iter().flat_map(|info| info.symbols.clone()).collect(),
        );
        info.state = [scores.info(), ids.info()]
            .iter()
            .flat_map(|info| info.state.clone())
            .collect();
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

impl PreparedOp<ChainBuffers<View>> for Chain {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &ChainBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let gathered = |gather: &GatherBuffers<View>| {
            gather
                .ranks
                .get(self.rank as usize)
                .map(|views| views.output)
                .ok_or(Refusal("the chain lacks its rank's gathered pairs"))
        };
        let partial = ArgmaxPartialBuffers {
            logits: buffers.logits,
            live_width: buffers.live_width,
            pairs: ArgmaxPairs {
                scores: buffers.scores.input,
                ids: buffers.ids.input,
                peer: false,
            },
            live_rows: buffers.live_rows,
        };
        let last = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: gathered(&buffers.scores)?,
                ids: gathered(&buffers.ids)?,
                peer: false,
            },
            output: Some(buffers.output),
            scatter: Vec::new(),
            live_rows: buffers.live_rows,
        };
        // Every gather moves its whole block at one row.
        let mut launches = self.partial.launches(&partial, View::EMPTY, counts)?;
        launches.extend(
            self.scores
                .launches(&buffers.scores, View::EMPTY, Counts::rows(1))?,
        );
        launches.extend(
            self.ids
                .launches(&buffers.ids, View::EMPTY, Counts::rows(1))?,
        );
        launches.extend(self.last.launches(&last, View::EMPTY, counts)?);
        Ok(launches)
    }
}

/// A rank's gathered pairs: [`CAPACITY`] per rank, of which each rank's
/// live rows are written (the rest copy the partial's unwritten fill).
fn gathered_output<'a>(ranks: u32, dtype: DType) -> Spec<'a> {
    let written = move |_: &Images, live: u32| -> LabResult<WriteMap> {
        WriteMap::of(dtype).ranges((0..ranks as usize).map(|rank| {
            let start = rank * CAPACITY as usize;
            start..start + live as usize
        }))
    };
    Spec::output_with(
        dtype,
        (Rows::Fixed(ranks as usize * CAPACITY as usize), 1),
        Write::Map(Box::new(written)),
    )
}

/// Rank `me`'s record of a gather of the pairs `dtype` in a chain of
/// `ranks` ranks, its state regions `state` bytes: its partial's pairs as
/// input, its own gathered pairs, and every other rank's by that rank's
/// path under `gather`.
fn gather_record<'a>(
    ranks: u32,
    me: u32,
    gather: &str,
    dtype: DType,
    state: &[u64],
) -> GatherBuffers<Slot<'a>> {
    GatherBuffers {
        input: Slot::Own(Spec::output(dtype, (Rows::Live, 1))),
        ranks: (0..ranks)
            .map(|rank| RankViews {
                output: if rank == me {
                    Slot::Own(gathered_output(ranks, dtype))
                } else {
                    Slot::peer(rank, format!("{gather}.ranks[{rank}].output"))
                },
                state: (0..state.len())
                    .map(|at| {
                        if rank == me {
                            Slot::State(state[at])
                        } else {
                            Slot::peer(rank, format!("{gather}.ranks[{rank}].state[{at}]"))
                        }
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// The chain of `ranks` ranks over `logits` at [`CAPACITY`] rows: every
/// rank's pairs, gathered pairs and ids as the oracle's bits, the ids the
/// same on every rank.
pub fn chain_case<'a>(
    logits: DType,
    ranks: u32,
    ops: Vec<Prepared<ChainBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = ops
        .first()
        .map(|op| op.info().state.clone())
        .unwrap_or_default();
    let half = state.len() / 2;
    let records = (0..ranks)
        .map(|me| ChainBuffers {
            logits: Slot::Own(Spec::input(
                logits,
                (Rows::Live, VOCAB as usize),
                shard_image(logits, me, ranks),
            )),
            live_width: Slot::Own(Spec::input(
                DType::I32,
                (Rows::Fixed(1), 1),
                LIVE.to_le_bytes().to_vec(),
            )),
            scores: gather_record(ranks, me, "scores", DType::F32, &state[..half]),
            ids: gather_record(ranks, me, "ids", DType::I32, &state[half..]),
            output: Slot::Own(Spec::output(DType::I32, (Rows::Live, 1))),
            live_rows: Slot::Own(Spec::live()),
        })
        .collect();
    let oracle = move |images: &[Images], _: u32| -> LabResult<Vec<Vec<(String, Expected)>>> {
        let pairs = images
            .iter()
            .zip(0..)
            .map(|(images, rank)| partials(&logit_values(images, "logits", logits)?, rank))
            .collect::<LabResult<Vec<_>>>()?;
        let rows = (0..CAPACITY as usize)
            .map(|row| {
                pairs
                    .iter()
                    .map(|rank| (rank[row].0, rank[row].1 as usize))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let ids_of = finals(&rows)?;
        let gathered_scores = pairs
            .iter()
            .flat_map(|rank| scores(rank))
            .collect::<Vec<_>>();
        let gathered_ids = pairs.iter().flat_map(|rank| ids(rank)).collect::<Vec<_>>();
        Ok((0..images.len())
            .map(|rank| {
                vec![
                    ("scores.input".into(), Expected::Bits(scores(&pairs[rank]))),
                    ("ids.input".into(), Expected::Bits(ids(&pairs[rank]))),
                    (
                        format!("scores.ranks[{rank}].output"),
                        Expected::Bits(gathered_scores.clone()),
                    ),
                    (
                        format!("ids.ranks[{rank}].output"),
                        Expected::Bits(gathered_ids.clone()),
                    ),
                    ("output".into(), Expected::Bits(i32_bytes(&ids_of))),
                ]
            })
            .collect())
    };
    Ok(CollectiveCase::new(ops, CAPACITY, records)?
        .oracle(oracle)
        .same_on_every_rank(&["output"])
        .without_properties(
            &[Property::Chunk, Property::Rows, Property::Batch],
            "the gathered pairs are blocks of every row, not rows of the case",
        ))
}

/// The chain over F32 and over BF16 logits at any rank count, each
/// required.
pub fn required_chains<'a>() -> Vec<RequiredCollective<'a>> {
    [DType::F32, DType::Bf16]
        .into_iter()
        .map(|logits| {
            RequiredCollective::new(
                format!("sharded argmax chain {logits:?}"),
                Need::rows(CAPACITY),
                move |backend, ranks, rank, need| {
                    Chain::prepare(backend, logits, ranks, rank, need)
                        .map(|chain| Box::new(chain) as Prepared<ChainBuffers<View>>)
                },
                move |ranks, ops| chain_case(logits, ranks, ops),
            )
            .required()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind's winner is the one its comment names, in F32 and in
    /// BF16, so a case's rows cover what the module states.
    #[test]
    fn row_kinds_win_where_they_say() {
        let won = |kind: usize, bf16: bool| {
            let row = kind_row(kind, 7)
                .into_iter()
                .map(|v| if bf16 { bf16_to_f32(to_bf16(v)) } else { v })
                .collect::<Vec<_>>();
            let [(score, id)] = oracle::sharded_argmax_partials(&[&row]).unwrap()[..] else {
                unreachable!()
            };
            (score.to_bits(), id)
        };
        assert_eq!(won(0, false), (2.5_f32.to_bits() + 1, 200));
        assert_eq!(won(0, true), (2.5_f32.to_bits(), 100));
        assert_eq!(won(1, false), (4.0_f32.to_bits(), 64));
        assert_eq!(won(2, true), ((-0.0_f32).to_bits(), 7));
        assert_eq!(won(3, true), (0.0_f32.to_bits(), 7));
        assert_eq!(won(4, false), (f32::INFINITY.to_bits(), 300));
        assert_eq!(won(5, true), (4.0_f32.to_bits(), 401));
        assert_eq!(won(6, false), (f32::NEG_INFINITY.to_bits(), 600));
        assert_eq!(won(7, true), (f32::NEG_INFINITY.to_bits(), 1));
        assert_eq!(won(8, true), (u32::from(BF16_NANS[0]) << 16, 0));
        for &bits in &BF16_NANS {
            assert_eq!(to_bf16(bf16_to_f32(bits)), bits);
        }
    }

    /// Live rows that share a destination are outside the contract: the
    /// oracle refuses them rather than pick a winner.
    #[test]
    fn scattered_rows_must_have_distinct_destinations() {
        let images = Images::default()
            .with("scatter[0].rows", i32_bytes(&[3, -1, 3, 9]))
            .with("scatter[0].output", i32_bytes(&[0; SCATTERED]));
        let ids = [10, 11, 12, 13];
        let two = scattered_bits(&images, 0, &ids, 2).unwrap();
        assert_eq!(&two[12..16], &10_i32.to_le_bytes());
        assert!(scattered_bits(&images, 0, &ids, 3).is_err());
    }

    /// The chain's last row is only NaN on every rank and the one before
    /// ties on every rank: both yield rank 0's ids.
    #[test]
    fn chain_rows_cover_all_nan_and_cross_rank_ties() {
        for ranks in RANKS {
            let row = |row: usize| {
                (0..ranks)
                    .map(|rank| shard_row(row, rank, ranks))
                    .collect::<Vec<_>>()
            };
            let last = row(CAPACITY as usize - 1);
            let shards = last.iter().map(|s| &s[..LIVE as usize]).collect::<Vec<_>>();
            assert!(shards.iter().all(|s| s.iter().all(|v| v.is_nan())));
            assert_eq!(oracle::sharded_argmax(&shards), Ok(0));
            let tied = row(CAPACITY as usize - 2);
            let shards = tied.iter().map(|s| &s[..LIVE as usize]).collect::<Vec<_>>();
            assert_eq!(oracle::sharded_argmax(&shards), Ok(10));
        }
    }
}
