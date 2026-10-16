//! Collective cases of the kit's multi-rank runner
//! ([`crate::kit::collective`]): the F32 all-reduce, the BF16-partials
//! all-reduce and the rounded all-reduce (`CommQuery::AllReduceRounded`),
//! each through its entries and through its composed form, the fused BF16
//! all-reduce with a residual and the HF RMSNorm at each use that chooses
//! its entries (decode, decode in a graph, prefill, exact verify; plain and
//! routed rows), the all-gather onto every rank and onto one, the row
//! gather and row exchange of sharded rows, and the context-parallel
//! group gather onto every member and onto one; with SM120 registered as
//! the lab's cubins allow and the required set at a rank count.
//!
//! Every rank's row inputs differ (one seed per rank) while the residual
//! and the norm weight are the same on every rank, as a model's are.
//! Values are the kit's BF16 generator's, so the F32 sums the oracles take
//! of a few of them are exact.
//!
//! Oracles: the all-reduce within the F64 sum's bound and the same bits on
//! every rank; the BF16-partials all-reduce as exact bits
//! (`oracle::all_reduce_bf16_partials`) over F32 inputs off the BF16 grid,
//! the same on every rank, and the rounded all-reduce as exact bits
//! (`oracle::all_reduce_rounded`) over the same inputs; the fused all-reduce
//! as the contract's epilogue gate
//! (`contract::RmsNormEpilogue`): the residual output bit exact (the F32
//! sum of the ranks' rows in rank order plus the residual, rounded to
//! BF16), every rank's output equal, and each normalized output within two
//! BF16 steps of HF's ordered reference over that residual (the weights
//! exceed one); routed rows first sum their eight routes in route order in
//! F32 and round to BF16, as the combine does; the gathers and row moves
//! as exact bits.

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::collective::{CollectiveCase, RequiredCollective, Slot};
use crate::kit::{Spec, bf16_bytes, bf16_values};
use crate::poison::WriteMap;
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::collective::{
    AllReduceBuffers, FusedAllReduceBuffers, GatherBuffers, RankViews, ReduceInput,
    RowGatherBuffers, ShardedRows, owned_rows,
};
use kernel_core::context_parallel::{GroupTransfer, GroupTransferOp};
use kernel_core::contract::{CommQuery, DType, RmsNormEpilogue, RmsNormKind};
use kernel_core::oracle;
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::{Invariance, Property};
use kernel_core::rmsnorm::FusedResidual;
use kernels_sm120::backend::Sm120;

/// The row width of the all-reduces: the fused entries' hidden width.
const WIDTH: usize = 4096;

/// The rows of each all-reduce use.
const DECODE_ROWS: u32 = 32;
const GRAPH_ROWS: u32 = 256;
const PREFILL_ROWS: u32 = 64;
const REDUCE_ROWS: u32 = 64;

/// The routed rows each combine form sums.
const ROUTES: u32 = 8;

/// The BF16-partials all-reduce's row width and rows: past the phased
/// turns entry's first rows, so every form of SM120's table launches.
const PARTIALS_WIDTH: usize = 2048;
const PARTIALS_ROWS: u32 = kernels_sm120::backend::BF16_TURNS_ROWS + 8;

/// Both sides of every bound of SM120's table of the BF16-partials
/// all-reduce: the low-latency entry at one row, the phased one-shot to
/// four, the two-shot without workers to 32, with them to 1023, then the
/// turns entry.
pub const PARTIALS_COUNTS: [u32; 8] = [1, 2, 4, 5, 32, 33, 1023, 1024];

/// Graphs of planned rows of each form of that table replayed with fewer
/// device live rows, zero included, in turn on one state: a served 32-row
/// bucket at one live row, zero-live calls between live ones, and each
/// form at zero, some and all of its rows.
const PARTIALS_LIVE_BELOW: [(u32, &[u32]); 5] = [
    (32, &[1, 0, 32, 0, 5]),
    (1, &[0, 1]),
    (4, &[0, 2, 4]),
    (33, &[0, 7]),
    (kernels_sm120::backend::BF16_TURNS_ROWS, &[0, 1, 1024]),
];

/// The RMSNorm epsilon of the fused all-reduces.
const EPS: f32 = 1e-6;

/// Elements of one rank's shard of a sharded row, and of a gathered block:
/// not multiples of a tile.
const SHARD: usize = 1000;
const BLOCK: usize = 5000;

/// Rows of the row moves and steps of the gathers.
const MOVE_ROWS: u32 = 16;
const GATHER_STEPS: u32 = 4;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `count` F32 values of the kit's BF16 generator from `seed`.
fn f32_values(seed: u64, count: usize) -> Vec<f32> {
    bf16_values(seed, count)
        .into_iter()
        .map(bf16_to_f32)
        .collect()
}

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// Every rank's views for rank `me`'s record: rank `me`'s own output at
/// `output` and its state regions of `state` bytes, every other rank's by
/// that rank's paths in its own record.
fn rank_views<'a>(
    ranks: u32,
    me: u32,
    own: impl Fn() -> Slot<'a>,
    output: impl Fn(u32) -> String,
    state: &[u64],
) -> Vec<RankViews<Slot<'a>>> {
    (0..ranks)
        .map(|rank| RankViews {
            output: if rank == me {
                own()
            } else {
                Slot::peer(rank, output(rank))
            },
            state: state
                .iter()
                .enumerate()
                .map(|(at, &bytes)| {
                    if rank == me {
                        Slot::State(bytes)
                    } else {
                        Slot::peer(rank, format!("ranks[{rank}].state[{at}]"))
                    }
                })
                .collect(),
        })
        .collect()
}

/// The state regions every rank's op lists.
fn states<B>(ops: &[Prepared<B>]) -> Vec<u64> {
    ops.first()
        .map(|op| op.info().state.clone())
        .unwrap_or_default()
}

/// The F32 all-reduce of `ranks` ranks over [`REDUCE_ROWS`] rows.
pub fn all_reduce_case<'a>(
    ranks: u32,
    ops: Vec<Prepared<AllReduceBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = states(&ops);
    let rows = REDUCE_ROWS as usize;
    let records = (0..ranks)
        .map(|me| AllReduceBuffers {
            input: Slot::Own(Spec::input(
                DType::F32,
                (Rows::Live, WIDTH),
                f32_bytes(&f32_values(100 + u64::from(me), rows * WIDTH)),
            )),
            output: Slot::Own(Spec::output(DType::F32, (Rows::Live, WIDTH))),
            live_rows: None,
            ranks: rank_views(
                ranks,
                me,
                || Slot::peer(me, "output"),
                |_| "output".into(),
                &state,
            ),
        })
        .collect();
    Ok(CollectiveCase::new(ops, REDUCE_ROWS, records)?
        .oracle(move |images, _| {
            let inputs = images
                .iter()
                .map(|images| {
                    Ok(images
                        .f32s("input")?
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect())
                })
                .collect::<LabResult<Vec<Vec<f64>>>>()?;
            let sum = oracle::all_reduce(&inputs, None, None).map_err(|e| e.0.to_string())?;
            Ok((0..inputs.len())
                .map(|_| vec![("output".to_owned(), Expected::Values(sum.clone()))])
                .collect())
        })
        .same_on_every_rank(&["output"]))
}

/// `count` F32 values from `seed` off the BF16 grid: the kit's BF16 values
/// with low mantissa bits set, every seventh exactly halfway between two
/// BF16 values (a tie, which rounds to even).
fn off_grid(seed: u64, count: usize) -> Vec<f32> {
    let low = bf16_values(seed ^ 0x5a5a_5a5a, count);
    bf16_values(seed, count)
        .into_iter()
        .zip(low)
        .enumerate()
        .map(|(at, (value, low))| {
            let low = if at % 7 == 0 {
                0x8000
            } else {
                u32::from(low) | 1
            };
            f32::from_bits(bf16_to_f32(value).to_bits() | low)
        })
        .collect()
}

/// Rank `rank`'s special value of pattern `pattern` among `ranks` ranks,
/// or `None` where the rank keeps its finite value. Patterns past the list
/// keep every rank's.
fn special(pattern: usize, rank: u32, ranks: u32) -> Option<f32> {
    let nan = |sign: u32, payload: u32| f32::from_bits(sign << 31 | 0x7fc0_0000 | payload);
    let last = ranks - 1;
    match (pattern, rank) {
        // A quiet NaN with a payload, a negative one, a signaling one.
        (0, 0) => Some(nan(0, 0x1234)),
        (1, r) if r == last => Some(nan(1, 0x55)),
        (2, 1) => Some(f32::from_bits(0x7f80_0001)),
        // Infinities, alone and of opposite signs.
        (3, 0) => Some(f32::INFINITY),
        (4, 1) => Some(f32::NEG_INFINITY),
        (5, 0) => Some(f32::INFINITY),
        (5, 1) => Some(f32::NEG_INFINITY),
        // Finite partials that round to opposite infinities.
        (6, 0) => Some(f32::MAX),
        (6, 1) => Some(-f32::MAX),
        // An F32 sum past the F32 range, and a finite F32 sum past the
        // BF16 range.
        (7, 0 | 1) => Some(3.0e38),
        (8, 0) => Some(bf16_to_f32(0x7f7f)),
        (8, 1) => Some(1.0e36),
        // A subnormal partial alone, and negative zeros on every rank.
        (9, 0) => Some(1.0e-40),
        (9, _) => Some(0.0),
        (10, _) => Some(-0.0),
        _ => None,
    }
}

/// Patterns of [`special`]: every element's pattern is its index modulo
/// this, so every row holds each.
const SPECIAL_PERIOD: usize = 16;

/// The inputs of a BF16-partials case: rank `me`'s `count` values among
/// `ranks` ranks.
pub type PartialsInputs = fn(u32, u32, usize) -> Vec<f32>;

/// Finite values off the BF16 grid.
pub fn finite_partials(me: u32, _ranks: u32, count: usize) -> Vec<f32> {
    off_grid(700 + u64::from(me), count)
}

/// [`finite_partials`] with [`special`] values in every row: NaNs of
/// either sign and payload, infinities, partials and sums past the BF16
/// and F32 ranges, a subnormal and negative zeros.
pub fn special_partials(me: u32, ranks: u32, count: usize) -> Vec<f32> {
    finite_partials(me, ranks, count)
        .into_iter()
        .enumerate()
        .map(|(at, value)| special(at % SPECIAL_PERIOD, me, ranks).unwrap_or(value))
        .collect()
}

/// What a partials case's ranks sum: BF16 partials of their F32 rows
/// (`CommQuery::AllReduce` of dtype BF16), or the F32 rows themselves with
/// the sum rounded once (`CommQuery::AllReduceRounded`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Partials {
    Bf16,
    Rounded,
}

impl Partials {
    /// The query of `ranks` ranks over rows of [`PARTIALS_WIDTH`].
    fn query(self, ranks: u32) -> CommQuery {
        let elements = PARTIALS_WIDTH as u64;
        match self {
            Partials::Bf16 => CommQuery::AllReduce {
                ranks,
                elements,
                dtype: DType::Bf16,
                fused_residual: false,
                fused_rmsnorm: None,
            },
            Partials::Rounded => CommQuery::AllReduceRounded { ranks, elements },
        }
    }

    /// The contract's oracle of the sums.
    fn sums(self, inputs: &[Vec<f32>]) -> LabResult<Vec<u16>> {
        match self {
            Partials::Bf16 => oracle::all_reduce_bf16_partials(inputs),
            Partials::Rounded => oracle::all_reduce_rounded(inputs),
        }
        .map_err(|e| e.0.to_string())
    }
}

/// The BF16-partials all-reduce of `ranks` ranks over [`PARTIALS_ROWS`]
/// rows of [`PARTIALS_WIDTH`] of `inputs`, stepping through
/// [`PARTIALS_COUNTS`] and replaying [`PARTIALS_LIVE_BELOW`]; with
/// `properties` false, its claims' property checks are reported as not
/// exercised (a run that checks them on another case).
pub fn bf16_partials_case<'a>(
    inputs: PartialsInputs,
    properties: bool,
    ranks: u32,
    ops: Vec<Prepared<AllReduceBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    partials_case(Partials::Bf16, inputs, properties, ranks, ops)
}

/// [`bf16_partials_case`] of the all-reduce `partials` names.
pub fn partials_case<'a>(
    partials: Partials,
    inputs: PartialsInputs,
    properties: bool,
    ranks: u32,
    ops: Vec<Prepared<AllReduceBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = states(&ops);
    let (rows, width) = (PARTIALS_ROWS as usize, PARTIALS_WIDTH);
    let records = (0..ranks)
        .map(|me| AllReduceBuffers {
            input: Slot::Own(Spec::input(
                DType::F32,
                (Rows::Live, width),
                f32_bytes(&inputs(me, ranks, rows * width)),
            )),
            output: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, width))),
            live_rows: Some(Slot::Own(Spec::live())),
            ranks: rank_views(
                ranks,
                me,
                || Slot::peer(me, "output"),
                |_| "output".into(),
                &state,
            ),
        })
        .collect();
    let case = CollectiveCase::new(ops, PARTIALS_ROWS, records)?
        .oracle(move |images, live| {
            let elements = live as usize * width;
            let inputs = images
                .iter()
                .map(|images| Ok(images.f32s("input")?[..elements].to_vec()))
                .collect::<LabResult<Vec<_>>>()?;
            // No live row has a sum; the live-below check replays zero
            // live rows and compares none.
            let sums = match elements {
                0 => Vec::new(),
                _ => partials.sums(&inputs)?,
            };
            let mut bits = bf16_bytes(&sums);
            bits.resize(rows * width * 2, 0);
            Ok((0..inputs.len())
                .map(|_| vec![("output".to_owned(), Expected::Bits(bits.clone()))])
                .collect())
        })
        .same_on_every_rank(&["output"])
        .counts(&PARTIALS_COUNTS)
        .live_below(&PARTIALS_LIVE_BELOW);
    Ok(match properties {
        true => case,
        false => case.without_properties(
            &[Property::Rows, Property::Chunk, Property::Batch],
            "checked on the finite case of the full collectives run",
        ),
    })
}

/// A fused all-reduce's input: plain rows, or each row's routed rows.
fn routes(input: ReduceInput) -> usize {
    match input {
        ReduceInput::Rows => 1,
        ReduceInput::Routed { routes } => routes as usize,
    }
}

/// The fused all-reduce of `ranks` ranks of `input` at `rows` rows.
pub fn fused_all_reduce_case<'a>(
    ranks: u32,
    input: ReduceInput,
    rows: u32,
    ops: Vec<Prepared<FusedAllReduceBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = states(&ops);
    let (n, routes) = (rows as usize, routes(input));
    let routed = routes > 1;
    let residual = bf16_bytes(&bf16_values(200, n * WIDTH));
    let weight = bf16_bytes(&bf16_values(201, WIDTH));
    let records = (0..ranks)
        .map(|me| FusedAllReduceBuffers {
            input: Slot::Own(Spec::input(
                DType::Bf16,
                (Rows::Live, routes * WIDTH),
                bf16_bytes(&bf16_values(210 + u64::from(me), n * routes * WIDTH)),
            )),
            residual: FusedResidual {
                input: Slot::Own(Spec::input(
                    DType::Bf16,
                    (Rows::Live, WIDTH),
                    residual.clone(),
                )),
                output: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, WIDTH))),
            },
            weight: Slot::Own(Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), WIDTH),
                weight.clone(),
            )),
            output: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, WIDTH))),
            live_rows: routed.then(|| Slot::Own(Spec::live())),
            ranks: rank_views(
                ranks,
                me,
                || Slot::peer(me, "output"),
                |_| "output".into(),
                &state,
            ),
        })
        .collect();
    Ok(CollectiveCase::new(ops, rows, records)?
        .oracle(move |images, live| {
            let (residual, normed) = fused_oracle(images, live, n, routes)?;
            let ranks = images.len();
            Ok((0..ranks)
                .map(|_| {
                    let normed = normed.clone();
                    let judge = move |bytes: &[u8]| -> LabResult<()> {
                        for (at, &expected) in normed.iter().enumerate() {
                            let actual = u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]]);
                            if (i32::from(actual) - i32::from(expected)).abs() > 2 {
                                return Err(format!(
                                    "output element {at} is {actual:#06x}, HF's reference \
                                     {expected:#06x}: more than two BF16 steps"
                                ));
                            }
                        }
                        Ok(())
                    };
                    vec![
                        (
                            "residual.output".to_owned(),
                            Expected::Bits(residual.clone()),
                        ),
                        ("output".to_owned(), Expected::Judge(Box::new(judge))),
                    ]
                })
                .collect())
        })
        .same_on_every_rank(&["output", "residual.output"]))
}

/// The fused all-reduce's expected residual output (the whole buffer,
/// zeros past `live` rows) and normalized BF16 words of the `live` rows:
/// each rank's row (its routes summed in route order in F32 and rounded to
/// BF16), summed over ranks in rank order in F32 with the residual, rounded
/// to BF16; HF's ordered RMSNorm of that.
fn fused_oracle(
    images: &[Images],
    live: u32,
    rows: usize,
    routes: usize,
) -> LabResult<(Vec<u8>, Vec<u16>)> {
    let live = live as usize;
    let residual = images[0].u16s("residual.input")?;
    let weight = widen(&images[0].u16s("weight")?);
    let inputs = images
        .iter()
        .map(|images| images.u16s("input"))
        .collect::<LabResult<Vec<_>>>()?;
    let mut sums = vec![0_u16; rows * WIDTH];
    for row in 0..live {
        for e in 0..WIDTH {
            let rank_row = |input: &[u16]| {
                let routed = (0..routes).fold(0.0_f32, |sum, route| {
                    sum + bf16_to_f32(input[(row * routes + route) * WIDTH + e])
                });
                if routes > 1 {
                    bf16_to_f32(f32_to_bf16(routed))
                } else {
                    routed
                }
            };
            let sum = inputs
                .iter()
                .fold(0.0_f32, |sum, input| sum + rank_row(input));
            sums[row * WIDTH + e] = f32_to_bf16(sum + bf16_to_f32(residual[row * WIDTH + e]));
        }
    }
    let normed = oracle::rmsnorm(
        &widen(&sums[..live * WIDTH]),
        WIDTH,
        &weight,
        f64::from(EPS),
        RmsNormKind::HfBf16Intermediate,
        None,
    )
    .map_err(|e| e.0.to_string())?
    .values
    .into_iter()
    .map(|v| f32_to_bf16(v as f32))
    .collect();
    Ok((bf16_bytes(&sums), normed))
}

/// The all-gather of `ranks` ranks' blocks of [`BLOCK`] F32 elements onto
/// `target`, or onto every rank.
pub fn all_gather_case<'a>(
    ranks: u32,
    target: Option<u32>,
    ops: Vec<Prepared<GatherBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = states(&ops);
    let whole = ranks as usize * BLOCK;
    let receives = move |rank: u32| target.is_none_or(|target| target == rank);
    let records = (0..ranks)
        .map(|me| GatherBuffers {
            input: Slot::Own(Spec::input(
                DType::F32,
                (Rows::Fixed(1), BLOCK),
                f32_bytes(&f32_values(300 + u64::from(me), BLOCK)),
            )),
            ranks: rank_views(
                ranks,
                me,
                || received(receives(me), whole),
                |rank| format!("ranks[{rank}].output"),
                &state,
            ),
        })
        .collect();
    Ok(
        CollectiveCase::new(ops, GATHER_STEPS, records)?.oracle(move |images, _| {
            let blocks = images
                .iter()
                .map(|images| images.bytes("input").map(<[u8]>::to_vec))
                .collect::<LabResult<Vec<_>>>()?
                .concat();
            Ok((0..images.len() as u32)
                .map(|rank| {
                    let path = format!("ranks[{rank}].output");
                    if receives(rank) {
                        vec![(path, Expected::Bits(blocks.clone()))]
                    } else {
                        Vec::new()
                    }
                })
                .collect())
        }),
    )
}

/// A receiving rank's whole F32 output of `elements`, written at every
/// step, or a view a non-receiving rank must leave untouched.
fn received<'a>(receives: bool, elements: usize) -> Slot<'a> {
    Slot::Own(if receives {
        Spec::output_with(DType::F32, (Rows::Fixed(1), elements), Write::All)
    } else {
        Spec::untouched(elements * 4)
    })
}

/// The row gather of `ranks` ranks' shards of [`MOVE_ROWS`] rows onto
/// rank `target`.
pub fn gather_rows_case<'a>(
    rows: ShardedRows,
    target: u32,
    ops: Vec<Prepared<RowGatherBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = states(&ops);
    let ranks = rows.ranks;
    let (n, element) = (MOVE_ROWS as usize, element_bytes(rows.dtype)?);
    let whole = ranks as usize * SHARD;
    let records = (0..ranks)
        .map(|me| RowGatherBuffers {
            input: Slot::Own(Spec::input(
                rows.dtype,
                (Rows::Live, SHARD),
                shards(rows.dtype, 400 + u64::from(me), n * SHARD),
            )),
            indices: None,
            ranks: rank_views(
                ranks,
                me,
                || {
                    Slot::Own(if me == target {
                        Spec::output(rows.dtype, (Rows::Live, whole))
                    } else {
                        Spec::untouched(n * whole * element)
                    })
                },
                |rank| format!("ranks[{rank}].output"),
                &state,
            ),
        })
        .collect();
    Ok(
        CollectiveCase::new(ops, MOVE_ROWS, records)?.oracle(move |images, live| {
            let assembled = assembled(images, live, n, SHARD * element, 0..live)?;
            Ok((0..ranks)
                .map(|rank| {
                    if rank == target {
                        vec![(
                            format!("ranks[{rank}].output"),
                            Expected::Bits(assembled.clone()),
                        )]
                    } else {
                        Vec::new()
                    }
                })
                .collect())
        }),
    )
}

/// The row exchange of `ranks` ranks' shards of [`MOVE_ROWS`] rows onto
/// the ranks that own them (`collective::owned_rows`).
pub fn exchange_rows_case<'a>(
    rows: ShardedRows,
    ops: Vec<Prepared<GatherBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let state = states(&ops);
    let ranks = rows.ranks;
    let (n, element) = (MOVE_ROWS as usize, element_bytes(rows.dtype)?);
    let whole = ranks as usize * SHARD;
    let dtype = rows.dtype;
    let records = (0..ranks)
        .map(|me| GatherBuffers {
            input: Slot::Own(Spec::input(
                dtype,
                (Rows::Live, SHARD),
                shards(dtype, 500 + u64::from(me), n * SHARD),
            )),
            ranks: rank_views(
                ranks,
                me,
                || {
                    Slot::Own(Spec::output_with(
                        dtype,
                        (Rows::Live, whole),
                        Write::Map(Box::new(move |_, live| {
                            let owned = owned_rows(live, ranks, me);
                            WriteMap::of(dtype)
                                .rows(whole, owned.start as usize..owned.end as usize)
                        })),
                    ))
                },
                |rank| format!("ranks[{rank}].output"),
                &state,
            ),
        })
        .collect();
    Ok(CollectiveCase::new(ops, MOVE_ROWS, records)?
        .oracle(move |images, live| {
            (0..ranks)
                .map(|rank| {
                    let owned = owned_rows(live, ranks, rank);
                    let bytes = assembled(images, live, n, SHARD * element, owned)?;
                    Ok(vec![(
                        format!("ranks[{rank}].output"),
                        Expected::Bits(bytes),
                    )])
                })
                .collect()
        })
        .without_properties(
            &[Property::Rows, Property::Chunk],
            "a row's owner follows the step's rows (collective::owned_rows), so a row alone or \
             in another chunk lands on another rank",
        ))
}

/// The whole rows `0..rows` a row move assembles from every rank's
/// shards, `shard` bytes each, with only rows `kept` (of the first `live`)
/// filled and zeros elsewhere.
fn assembled(
    images: &[Images],
    live: u32,
    rows: usize,
    shard: usize,
    kept: std::ops::Range<u32>,
) -> LabResult<Vec<u8>> {
    let ranks = images.len();
    let inputs = images
        .iter()
        .map(|images| images.bytes("input"))
        .collect::<LabResult<Vec<_>>>()?;
    let mut whole = vec![0; rows * ranks * shard];
    for row in kept.start as usize..(kept.end.min(live)) as usize {
        for (rank, input) in inputs.iter().enumerate() {
            whole[(row * ranks + rank) * shard..][..shard]
                .copy_from_slice(&input[row * shard..][..shard]);
        }
    }
    Ok(whole)
}

fn element_bytes(dtype: DType) -> LabResult<usize> {
    match dtype {
        DType::F32 => Ok(4),
        DType::Bf16 => Ok(2),
        other => Err(format!("row moves of {other:?} are not covered")),
    }
}

/// `count` elements of `dtype` from `seed`.
fn shards(dtype: DType, seed: u64, count: usize) -> Vec<u8> {
    match dtype {
        DType::F32 => f32_bytes(&f32_values(seed, count)),
        _ => bf16_bytes(&bf16_values(seed, count)),
    }
}

/// The group transfer `op` of its members' blocks: a gather onto every
/// member or onto its owner.
pub fn group_transfer_case<'a>(
    op: GroupTransferOp,
    ops: Vec<Prepared<GatherBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let owner = match op.transfer {
        GroupTransfer::Gather => None,
        GroupTransfer::OwnerGather { owner } => Some(owner),
        _ => return Err("the kit covers group gathers onto every member or an owner".into()),
    };
    let state = states(&ops);
    let words = op.words as usize;
    let whole = op.size as usize * words;
    let receives = move |rank: u32| owner.is_none_or(|owner| owner == rank);
    let records = (0..op.size)
        .map(|me| GatherBuffers {
            input: Slot::Own(Spec::input(
                DType::F32,
                (Rows::Fixed(1), words),
                f32_bytes(&f32_values(600 + u64::from(me), words)),
            )),
            ranks: rank_views(
                op.size,
                me,
                || received(receives(me), whole),
                |rank| format!("ranks[{rank}].output"),
                &state,
            ),
        })
        .collect();
    Ok(
        CollectiveCase::new(ops, GATHER_STEPS, records)?.oracle(move |images, _| {
            let blocks = images
                .iter()
                .map(|images| images.bytes("input").map(<[u8]>::to_vec))
                .collect::<LabResult<Vec<_>>>()?
                .concat();
            Ok((0..images.len() as u32)
                .map(|rank| {
                    if receives(rank) {
                        vec![(
                            format!("ranks[{rank}].output"),
                            Expected::Bits(blocks.clone()),
                        )]
                    } else {
                        Vec::new()
                    }
                })
                .collect())
        }),
    )
}

/// The lab's BF16-partials and rounded registrations on `backend`: the
/// low-latency entries for SM120's tables' rows of [`PARTIALS_WIDTH`] and
/// the shared rounding rows at that width, and, when `phased`, the phased
/// BF16 and F32 entries.
fn partials(backend: Sm120, phased: bool) -> LabResult<Sm120> {
    let rows = u64::from(kernels_sm120::backend::BF16_LOW_LATENCY_ROWS);
    let ctas = kernels_sm120::comm::ll_bf16::ctas(rows * PARTIALS_WIDTH as u64)
        .map_err(|e| e.0.to_string())?;
    let rounded_rows = u64::from(kernels_sm120::backend::ROUNDED_LOW_LATENCY_ROWS);
    let rounded_ctas = kernels_sm120::comm::ll_f32::ctas(rounded_rows * PARTIALS_WIDTH as u64)
        .map_err(|e| e.0.to_string())?;
    let backend = backend
        .with_low_latency_bf16_all_reduce(ctas)
        .with_low_latency_rounded_all_reduce(rounded_ctas)
        .with_partial_rounding(kernels_sm120::row::rounding::shared(PARTIALS_WIDTH as u32));
    Ok(match phased {
        true => backend
            .with_phased_bf16_all_reduce()
            .with_phased_f32_all_reduce(),
        false => backend,
    })
}

/// SM120 as a model registers its BF16-partials and rounded all-reduces
/// without the phased entries (their grids not admitted): the low-latency
/// entries at their rows and the composed forms at every other row count.
pub fn sm120_composed() -> LabResult<Sm120> {
    partials(Sm120::new(), false)
}

/// SM120 with the collectives `cubins` hold: the kernel base's, the
/// BF16-partials all-reduce's entries (`partials`), and the exact
/// all-reduce entries (`comm::VERIFY_SYMBOL` and its combine form) under
/// the comm catalog's declaration, when a cubin of `cubins` holds them.
pub fn sm120(cubins: crate::Cubins) -> LabResult<Sm120> {
    let backend = partials(Sm120::new(), true)?;
    let (plain, combine) = (
        kernels_sm120::comm::VERIFY_SYMBOL,
        kernels_sm120::comm::VERIFY_COMBINE_SYMBOL,
    );
    if cubins.holding(plain).is_none() || cubins.holding(combine).is_none() {
        return Ok(backend);
    }
    let invariance = kernels_sm120::comm::declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == plain)
        .map(|entry| entry.invariance)
        .ok_or("the comm catalog declares no exact all-reduce")?;
    Ok(backend.with_exact_all_reduce(plain, combine, invariance))
}

/// The fused all-reduce query of `ranks` ranks.
fn fused_query(ranks: u32) -> CommQuery {
    CommQuery::AllReduce {
        ranks,
        elements: WIDTH as u64,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: RmsNormKind::HfBf16Intermediate,
            width: WIDTH as u32,
            eps: EPS,
        }),
    }
}

/// A required fused all-reduce of `input` at `usage`, needing `exact`.
fn fused<'a>(
    name: &str,
    input: ReduceInput,
    usage: OperationUse,
    exact: Invariance,
) -> RequiredCollective<'a> {
    let rows = usage.capacity;
    RequiredCollective::new(
        format!("fused all-reduce {name} rows {rows}"),
        Need::rows(rows).with_exact(exact),
        move |backend, ranks, rank, need| {
            backend.fused_all_reduce(&fused_query(ranks), input, usage, rank, need)
        },
        move |ranks, ops| fused_all_reduce_case(ranks, input, rows, ops),
    )
}

/// The BF16-partials all-reduce, required.
fn bf16_partials<'a>(
    form: &str,
    values: &str,
    inputs: PartialsInputs,
    properties: bool,
) -> RequiredCollective<'a> {
    partials_required_op(Partials::Bf16, form, values, inputs, properties)
}

/// The all-reduce `partials` names, required.
fn partials_required_op<'a>(
    partials: Partials,
    form: &str,
    values: &str,
    inputs: PartialsInputs,
    properties: bool,
) -> RequiredCollective<'a> {
    let name = match partials {
        Partials::Bf16 => "BF16 partials",
        Partials::Rounded => "rounded F32",
    };
    RequiredCollective::new(
        format!("all-reduce {name} {form} {values} width {PARTIALS_WIDTH} rows {PARTIALS_ROWS}"),
        Need::rows(PARTIALS_ROWS),
        move |backend, ranks, rank, need| backend.all_reduce(&partials.query(ranks), rank, need),
        move |ranks, ops| partials_case(partials, inputs, properties, ranks, ops),
    )
    .required()
}

/// The BF16-partials all-reduce through SM120's entries, on finite and on
/// special values: the part of [`required`] a run of this op alone takes,
/// with the finite case's property checks when `properties` holds. The
/// special values add no interaction between rows, so their case leaves
/// the property checks to the finite one.
pub fn partials_required<'a>(properties: bool) -> Vec<RequiredCollective<'a>> {
    vec![
        bf16_partials("entries", "finite", finite_partials, properties),
        bf16_partials("entries", "special values", special_partials, false),
        partials_required_op(
            Partials::Rounded,
            "entries",
            "finite",
            finite_partials,
            properties,
        ),
        partials_required_op(
            Partials::Rounded,
            "entries",
            "special values",
            special_partials,
            false,
        ),
    ]
}

/// The collectives a backend registered as [`sm120_composed`] serves: the
/// BF16-partials and rounded all-reduces through their composed forms.
pub fn composed_required<'a>() -> Vec<RequiredCollective<'a>> {
    vec![
        bf16_partials("composed", "finite", finite_partials, true),
        bf16_partials("composed", "special values", special_partials, false),
        partials_required_op(
            Partials::Rounded,
            "composed",
            "finite",
            finite_partials,
            true,
        ),
        partials_required_op(
            Partials::Rounded,
            "composed",
            "special values",
            special_partials,
            false,
        ),
    ]
}

/// The collectives' required set at any rank count: every op the runner
/// covers, each required (SM120 serves them at 2 and 4 ranks).
pub fn required<'a>() -> Vec<RequiredCollective<'a>> {
    let usage = |phase, capacity, graph| OperationUse {
        phase,
        capacity,
        graph,
    };
    let routed = ReduceInput::Routed { routes: ROUTES };
    let mut ops = vec![
        RequiredCollective::new(
            format!("all-reduce F32 width {WIDTH} rows {REDUCE_ROWS}"),
            Need::rows(REDUCE_ROWS),
            |backend, ranks, rank, need| {
                let query = CommQuery::AllReduce {
                    ranks,
                    elements: WIDTH as u64,
                    dtype: DType::F32,
                    fused_residual: false,
                    fused_rmsnorm: None,
                };
                backend.all_reduce(&query, rank, need)
            },
            all_reduce_case,
        ),
        fused(
            "decode",
            ReduceInput::Rows,
            usage(ExecutionPhase::Decode, DECODE_ROWS, false),
            Invariance::NONE,
        ),
        fused(
            "decode routed",
            routed,
            usage(ExecutionPhase::Decode, DECODE_ROWS, false),
            Invariance::NONE,
        ),
        fused(
            "decode graph",
            ReduceInput::Rows,
            usage(ExecutionPhase::Decode, GRAPH_ROWS, true),
            Invariance::NONE,
        ),
        fused(
            "prefill",
            ReduceInput::Rows,
            usage(ExecutionPhase::Prefill, PREFILL_ROWS, false),
            Invariance::NONE,
        ),
        fused(
            "exact verify",
            ReduceInput::Rows,
            usage(ExecutionPhase::Verify, DECODE_ROWS, true),
            Invariance::one_row(DECODE_ROWS),
        ),
        fused(
            "exact verify routed",
            routed,
            usage(ExecutionPhase::Verify, DECODE_ROWS, true),
            Invariance::one_row(DECODE_ROWS),
        ),
    ];
    for target in [None, Some(0)] {
        ops.push(RequiredCollective::new(
            format!(
                "all-gather F32 {BLOCK} per rank onto {}",
                target.map_or_else(|| "every rank".into(), |rank| format!("rank {rank}"))
            ),
            Need::rows(GATHER_STEPS),
            move |backend, ranks, rank, need| {
                let query = CommQuery::AllGather {
                    ranks,
                    elements_per_rank: BLOCK as u64,
                    dtype: DType::F32,
                };
                backend.all_gather(&query, rank, target, need)
            },
            move |ranks, ops| all_gather_case(ranks, target, ops),
        ));
    }
    for dtype in [DType::Bf16, DType::F32] {
        let sharded = move |ranks| ShardedRows {
            ranks,
            shard: SHARD as u64,
            dtype,
        };
        ops.push(RequiredCollective::new(
            format!("row gather {dtype:?} shard {SHARD} onto rank 0"),
            Need::rows(MOVE_ROWS),
            move |backend, ranks, rank, need| backend.gather_rows(&sharded(ranks), rank, 0, need),
            move |ranks, ops| gather_rows_case(sharded(ranks), 0, ops),
        ));
        ops.push(RequiredCollective::new(
            format!("row exchange {dtype:?} shard {SHARD}"),
            Need::rows(MOVE_ROWS),
            move |backend, ranks, rank, need| backend.exchange_rows(&sharded(ranks), rank, need),
            move |ranks, ops| exchange_rows_case(sharded(ranks), ops),
        ));
    }
    for transfer in [
        GroupTransfer::Gather,
        GroupTransfer::OwnerGather { owner: 1 },
    ] {
        let op = move |size| GroupTransferOp {
            transfer,
            size,
            words: BLOCK as u64,
        };
        ops.push(RequiredCollective::new(
            format!("group transfer {transfer:?} {BLOCK} words"),
            Need::rows(GATHER_STEPS),
            move |backend, ranks, rank, need| backend.group_transfer(&op(ranks), rank, need),
            move |ranks, ops| group_transfer_case(op(ranks), ops),
        ));
    }
    let mut ops = ops
        .into_iter()
        .map(RequiredCollective::required)
        .collect::<Vec<_>>();
    ops.extend(partials_required(true));
    ops
}
