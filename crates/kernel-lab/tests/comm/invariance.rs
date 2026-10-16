//! The comm invariance adapter: every `OpKind::Comm` entry that claims a
//! property (`kernels_sm120::comm::declared_entries` and the Qwen contract
//! declarations). Units are rows. Each
//! rank's input row derives from the rank and the row's seed only, so a
//! row's data never depends on its position, and a unit holds the row's
//! outputs on every rank that writes it, in rank order. Every call runs on
//! poisoned outputs and must write exactly its rows; no call carries state
//! into the next. After every call, each output of
//! each rank is checked against its write map (`check_output`): the rows
//! the rank receives written, finite for the all-reduce and norm results,
//! every other row untouched. Inboxes, flags and epochs are never poisoned.
//!
//! - MiMo's fused hidden all-reduce (residual add and RMSNorm); a unit is
//!   each rank's normalized and pre-norm rows. The round-once prefill pair
//!   runs a prompt's chunks as the bundle's prefill op does (prepared by
//!   the bundle's backend, `sm120::backend`: the exact entries up to 32
//!   rows, the pair above) and other cases as the pair at any row count.
//!   The verify, decode one-shot and combine entries run as themselves; a
//!   verify row alone runs as the bundle's exact verify op launches one row
//!   (the entries of a one-row decode step).
//! - Qwen3.8's hyper-connection all-reduce through `launches`: the
//!   one-shot entry up to `ONE_SHOT_ROWS` rows, the phased pair above; a
//!   unit is each rank's residual and normed rows.
//! - Qwen3.8's lower-latency decode hyper-connection all-reduce of a rank
//!   count (`hc_decode::SYMBOLS_V2`), as a decode or K3 verify step
//!   calls it: a call of R rows takes the graph bucket of R prepared rows
//!   (the next power of two, so three rows run the one-shot of four with
//!   three live) on a fresh decode state, and a row alone is a call of one
//!   prepared row (low latency); a unit is each rank's residual and normed
//!   rows. An entry of R ranks runs on the run's first R ranks.
//! - The BF16-partials all-reduce's entries (`comm::ll_bf16`,
//!   `comm::phased_bf16`, `comm::phased_turns_bf16`) and the rounded
//!   all-reduce's (`comm::ll_f32`'s round entry and `comm::phased_f32`'s
//!   rounding entries), each launched as
//!   itself at any row count on one state (the phased entries at SM120's
//!   senders and workers, live rows from `live_rows`); the inputs are F32
//!   values off the BF16 grid, and a unit is each rank's BF16 output row.
//!
//! Entries run on ranks of distinct devices. An entry a run cannot place
//! is reported not qualified on this host, never as tested: the eight-rank
//! decode all-reduce runs only on a host of eight devices.
//! - The row gathers (to rank 0) and exchanges: a unit is the gathered row
//!   on the rank that receives it.
//! - The shared all-gather (`context_parallel::group::ALLGATHER_SYMBOL`) of
//!   a group of every rank: a call of R rows gathers each rank's block of R
//!   rows of `SHARD` words into every rank's output, rank-major; a unit is
//!   the row's words of every source on every rank. Its flags and epochs
//!   are zeroed before every call (their layout follows the call's tiles),
//!   with every rank idle in between.
//! - The phased group gather (`context_parallel::phased::SYMBOL`, the lab's
//!   `comm_group_phased_lab_sm120`): the shared all-gather's call, units and
//!   state, through the phased transport.
//! - The F32 reduce-scatter and scattered all-gather pair (the comm
//!   engine's two-shot all-reduce of F32 rows of `SHARD` values): a call of
//!   R rows sums every rank's rows into every rank's output; a unit is the
//!   row's sum on every rank. Flags and epochs are zeroed before every call.

use super::*;
use crate::invariance::{Adapter, Element, Units, written};
use bundle_mimo26::{sm120, sm120::launch::comm::push_symbols, switches::Switches, topology::Tp};
use kernel_core::backend::{Counts, Need, View};
use kernel_core::collective::{FusedAllReduceBuffers, RankViews, ReduceInput};
use kernel_core::contract::{ArgumentList, LaunchPlan, SigmoidMeanHyperConnectionQuery};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm::FusedResidual;
use kernels_sm120::comm;
use kernels_sm120::context_parallel::group::{self, GatherState, GroupRank};
use kernels_sm120::invariance::{Declared, OpKind, Rows};
use model_qwen38_contract::sm120::comm::{self as hc, decode as hc_decode};
use runtime_cuda::Function;

/// Rows every call's buffers hold.
const CAPACITY: usize = 128;
/// Prompt rows of a chunk case past the 32-row thresholds (MiMo's exact
/// entries and Qwen's `ONE_SHOT_ROWS`), so splits put chunks on both sides.
const PROMPT: u32 = 96;
const HIDDEN: usize = 4096;
/// Preweighted routed rows a combine entry sums per row.
const ROUTED: usize = 8;
const EPS: f32 = 1e-6;
/// Elements of one rank's shard of a gathered row.
const SHARD: usize = 1024;
const QWEN_WIDTH: usize = 2560;
const QWEN_WIDE: usize = 4 * QWEN_WIDTH;
const V2_SYMBOL: &str = "chop_comm_allreduce_oneshot_norm_bf16_v2";
/// Rows of the low-latency BF16 entry's state: its 128 CTAs.
const PARTIALS_LL_ROWS: usize = 64;
/// Rows of the low-latency round entry's state: its 128 CTAs.
const ROUND_LL_ROWS: usize = 32;
/// SM120's senders of every phased BF16 call and the workers of its
/// two-shot and turns calls past 32 rows.
const PARTIALS_SENDERS: u32 = 16;
const PARTIALS_WORKERS: u32 = 32;
const COMBINE_SYMBOL: &str = "chop_comm_allreduce_combine_norm_bf16_v1";

pub struct Comm;

/// The entries one MiMo hidden all-reduce runs.
#[derive(Clone, Copy)]
enum Mimo {
    /// The bundle's prefill op.
    Prefill,
    /// The bundle's exact verify op; `true` for its routed input.
    Exact(bool),
    /// The round-once pair at any row count.
    Pair,
    /// One push entry; `true` for the combine entries' routed input.
    Push(&'static str, bool),
}

#[derive(Clone, Copy)]
enum Driver {
    Mimo(Mimo),
    Qwen,
    /// A Qwen3.8 lower-latency decode entry and its rank count.
    QwenDecode(&'static str, u32),
    /// A row gather or exchange entry and its element bytes.
    Gather(&'static str, usize),
    Exchange(&'static str, usize),
    /// The shared F32 all-gather.
    AllGather,
    /// A BF16-partials all-reduce entry.
    Partials(&'static str),
    /// The phased group gather.
    PhasedGather,
    /// The F32 reduce-scatter then scattered all-gather.
    ScatterPair,
}

/// The lab cubin of the phased group gather.
const PHASED_CUBIN: &str = "comm_group_phased_lab_sm120";
const SCATTER_F32: &str = "chop_comm_reduce_scatter_f32_v1";
const GATHER_SCATTERED_F32: &str = "chop_comm_allgather_scattered_f32_v1";

struct Case {
    driver: Driver,
    prompt: u32,
    batch: usize,
}

fn case(entry: &Declared) -> LabResult<Case> {
    let (driver, prompt, batch) = driver_of(&entry.symbol)?;
    // A one-row claim bounds the rows of a call.
    let batch = match entry.invariance.rows {
        Rows::OneRow { max } => batch.min(max as usize),
        Rows::None => batch,
    };
    Ok(Case {
        driver,
        prompt,
        batch,
    })
}

/// The driver of a per-rank symbol, its prompt rows and its batch rows.
fn driver_of(symbol: &str) -> LabResult<(Driver, u32, usize)> {
    let push = |symbol, combine| Driver::Mimo(Mimo::Push(symbol, combine));
    Ok(match symbol {
        comm::PREFILL_ROUND_ONCE_SYMBOL | comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL => {
            (Driver::Mimo(Mimo::Pair), PROMPT, 40)
        }
        // A chunk of the verify entries holds at most their 32 rows.
        comm::VERIFY_SYMBOL => (push(comm::VERIFY_SYMBOL, false), 32, 32),
        comm::VERIFY_COMBINE_SYMBOL => (push(comm::VERIFY_COMBINE_SYMBOL, true), 32, 32),
        V2_SYMBOL => (push(V2_SYMBOL, false), 0, 4),
        COMBINE_SYMBOL => (push(COMBINE_SYMBOL, true), 0, 4),
        comm::DECODE_COMBINE_SYMBOL => (push(comm::DECODE_COMBINE_SYMBOL, true), 0, 4),
        comm::DECODE_SYMBOL_V2 => (push(comm::DECODE_SYMBOL_V2, false), 0, 2),
        hc::SYMBOL => (Driver::Qwen, PROMPT, 16),
        hc::SCATTER_SYMBOL | hc::GATHER_SYMBOL => (Driver::Qwen, PROMPT, 40),
        symbol
            if [hc_decode::SYMBOLS_V2, hc_decode::SYMBOLS_V3]
                .iter()
                .flatten()
                .any(|&s| s == symbol) =>
        {
            let ranks = hc::RANKS
                .into_iter()
                .find(|&ranks| {
                    [hc_decode::symbol_v2(ranks), hc_decode::symbol_v3(ranks)]
                        .into_iter()
                        .any(|s| s.is_ok_and(|s| s == symbol))
                })
                .ok_or_else(|| format!("{symbol}: no rank count"))?;
            let symbol = if symbol.ends_with("_v3") {
                hc_decode::symbol_v3(ranks)
            } else {
                hc_decode::symbol_v2(ranks)
            }
            .map_err(|e| e.to_string())?;
            (Driver::QwenDecode(symbol, ranks), 0, 4)
        }
        "chop_comm_gather_rows_f32_v1" => {
            (Driver::Gather("chop_comm_gather_rows_f32_v1", 4), PROMPT, 4)
        }
        "chop_comm_gather_rows_bf16_v1" => (
            Driver::Gather("chop_comm_gather_rows_bf16_v1", 2),
            PROMPT,
            4,
        ),
        "chop_comm_exchange_rows_f32_v1" => (
            Driver::Exchange("chop_comm_exchange_rows_f32_v1", 4),
            PROMPT,
            4,
        ),
        "chop_comm_exchange_rows_bf16_v1" => (
            Driver::Exchange("chop_comm_exchange_rows_bf16_v1", 2),
            PROMPT,
            4,
        ),
        group::ALLGATHER_SYMBOL => (Driver::AllGather, PROMPT, 4),
        // The low-latency entry's state and the one-shot hold at most 64
        // rows.
        comm::ll_bf16::SYMBOL => (Driver::Partials(comm::ll_bf16::SYMBOL), 32, 32),
        comm::phased_bf16::ONE_SHOT_SYMBOL => {
            (Driver::Partials(comm::phased_bf16::ONE_SHOT_SYMBOL), 32, 32)
        }
        comm::phased_bf16::SYMBOL => (Driver::Partials(comm::phased_bf16::SYMBOL), PROMPT, 40),
        comm::phased_turns_bf16::SYMBOL => (
            Driver::Partials(comm::phased_turns_bf16::SYMBOL),
            PROMPT,
            40,
        ),
        // The low-latency round entry's state holds at most 32 rows.
        comm::ll_f32::ROUND_SYMBOL => (Driver::Partials(comm::ll_f32::ROUND_SYMBOL), 32, 32),
        comm::phased_f32::ONE_SHOT_ROUND_SYMBOL => (
            Driver::Partials(comm::phased_f32::ONE_SHOT_ROUND_SYMBOL),
            32,
            32,
        ),
        comm::phased_f32::ROUND_SYMBOL => {
            (Driver::Partials(comm::phased_f32::ROUND_SYMBOL), PROMPT, 40)
        }
        kernels_sm120::context_parallel::phased::SYMBOL => (Driver::PhasedGather, PROMPT, 4),
        SCATTER_F32 | GATHER_SCATTERED_F32 => (Driver::ScatterPair, PROMPT, 4),
        symbol => return Err(format!("{symbol}: no comm invariance driver")),
    })
}

/// Whether `symbol` is a rounded all-reduce entry (`comm::ll_f32`'s round
/// entry and `comm::phased_f32`'s rounding entries), which sums F32
/// partials where the other partials entries sum BF16 ones.
fn rounded(symbol: &str) -> bool {
    [
        comm::ll_f32::ROUND_SYMBOL,
        comm::phased_f32::ONE_SHOT_ROUND_SYMBOL,
        comm::phased_f32::ROUND_SYMBOL,
    ]
    .contains(&symbol)
}

/// The ranks a driver's call takes: an entry of a rank count's, or the
/// run's.
fn ranks_of(driver: Driver, sessions: &[Session]) -> usize {
    match driver {
        Driver::QwenDecode(_, ranks) => ranks as usize,
        _ => sessions.len(),
    }
}

#[path = "../common/comm_setup.rs"]
mod comm_setup;

#[test]
fn decode_invariance_launch_setup_is_valid_without_cuda() {
    let shape = hc_decode::TRANSPORT.shape.epilogue;
    let width = u64::from(shape.width);
    let wide = width * u64::from(shape.streams);
    for ranks in [2, 4, 8] {
        let state = hc_decode::buffers(ranks).unwrap();
        let mut buffers = BTreeMap::from([
            (100, CAPACITY as u64 * width * 2),
            (101, CAPACITY as u64 * wide * 2),
            (102, CAPACITY as u64 * u64::from(shape.streams) * 2),
            (103, wide * 2),
            (104, CAPACITY as u64 * wide * 2),
            (105, CAPACITY as u64 * wide * 2),
            (108, state.epoch_bytes),
            (109, 4),
        ]);
        for peer in 0..u64::from(ranks) {
            buffers.insert(1 + peer, state.inbox_bytes);
            buffers.insert(11 + peer, state.flags_bytes);
        }
        for symbol in [hc_decode::symbol_v2(ranks), hc_decode::symbol_v3(ranks)] {
            let (driver, _, _) = driver_of(symbol.unwrap()).unwrap();
            for rows in 1..=hc_decode::MAX_ROWS as usize {
                for rank in 0..ranks as usize {
                    let calls = stages(driver, ranks as usize, rank, rows).unwrap();
                    assert_eq!(calls.len(), 1);
                    for (symbol, plan, args) in calls {
                        comm_setup::check(
                            hc_decode::TRANSPORT.cubin,
                            &symbol,
                            plan,
                            &args,
                            &buffers,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}

/// The all-gather driver's launches bind the comm cubin's entry over the
/// rig's buffers at every row count, gathering exactly the call's words.
#[test]
fn all_gather_invariance_launch_setup_is_valid_without_cuda() {
    for ranks in [2, 4] {
        let state = group::gather_buffers(ranks as u32, (CAPACITY * SHARD) as u64).unwrap();
        let mut buffers = BTreeMap::from([
            (100, (CAPACITY * SHARD * 4) as u64),
            (102, state.epoch_bytes),
        ]);
        for peer in 0..ranks as u64 {
            buffers.insert(1 + peer, (CAPACITY * SHARD * ranks * 4) as u64);
            buffers.insert(11 + peer, state.flag_bytes);
        }
        for rows in [1, 2, 5, 64, CAPACITY] {
            for rank in 0..ranks {
                let calls = stages(Driver::AllGather, ranks, rank, rows).unwrap();
                assert_eq!(calls.len(), 1);
                let (symbol, plan, args) = &calls[0];
                assert_eq!(
                    args.0[4],
                    kernel_core::contract::Argument::U64((rows * SHARD) as u64)
                );
                // Fewer tiles than the state holds.
                assert!(u64::from(plan.grid[0]) * 2 * ranks as u64 * 4 <= state.flag_bytes);
                comm_setup::check("comm_sm120", symbol, *plan, args, &buffers).unwrap();
            }
        }
    }
}

/// Each BF16-partials and rounded entry's launches at every row count its
/// case takes bind this rig's buffers, in the lab cubin that holds it.
#[test]
fn partials_invariance_launch_setup_is_valid_without_cuda() {
    for ranks in [2_u64, 4] {
        let turns =
            comm::phased_turns_bf16::buffers(ranks as u32, (CAPACITY * HIDDEN) as u64).unwrap();
        let ll =
            comm::ll_bf16::state_bytes(ranks as u32, (PARTIALS_LL_ROWS * HIDDEN) as u64).unwrap();
        let f32_state =
            comm::phased_f32::buffers(ranks as u32, (CAPACITY * HIDDEN) as u64).unwrap();
        let round_ll =
            comm::ll_f32::state_bytes(ranks as u32, (ROUND_LL_ROWS * HIDDEN) as u64).unwrap();
        for (symbol, cubin, inbox, flags, epoch) in [
            (
                comm::ll_bf16::SYMBOL,
                comm::ll_bf16::LAB_CUBIN,
                ll.inbox,
                4,
                ll.epochs,
            ),
            (
                comm::ll_f32::ROUND_SYMBOL,
                comm::ll_f32::LAB_CUBIN,
                round_ll.inbox,
                4,
                round_ll.epochs,
            ),
            (
                comm::phased_bf16::ONE_SHOT_SYMBOL,
                comm::phased_bf16::LAB_CUBIN,
                turns.inbox_bytes,
                turns.flag_bytes,
                turns.epoch_bytes,
            ),
            (
                comm::phased_bf16::SYMBOL,
                comm::phased_bf16::LAB_CUBIN,
                turns.inbox_bytes,
                turns.flag_bytes,
                turns.epoch_bytes,
            ),
            (
                comm::phased_turns_bf16::SYMBOL,
                comm::phased_turns_bf16::LAB_CUBIN,
                turns.inbox_bytes,
                turns.flag_bytes,
                turns.epoch_bytes,
            ),
            (
                comm::phased_f32::ONE_SHOT_ROUND_SYMBOL,
                comm::phased_f32::LAB_CUBIN,
                f32_state.inbox_bytes,
                f32_state.flag_bytes,
                f32_state.epoch_bytes,
            ),
            (
                comm::phased_f32::ROUND_SYMBOL,
                comm::phased_f32::LAB_CUBIN,
                f32_state.inbox_bytes,
                f32_state.flag_bytes,
                f32_state.epoch_bytes,
            ),
        ] {
            let mut buffers = BTreeMap::from([
                (100, (CAPACITY * HIDDEN * 4) as u64),
                (101, (CAPACITY * HIDDEN * 2) as u64),
                (102, epoch),
                (105, 4),
            ]);
            for peer in 0..ranks {
                buffers.insert(1 + peer, inbox);
                buffers.insert(11 + peer, flags);
                buffers.insert(21 + peer, (CAPACITY * HIDDEN * 2) as u64);
            }
            let (driver, prompt, batch) = driver_of(symbol).unwrap();
            let most = prompt.max(batch as u32) as usize;
            for rows in 1..=most {
                for rank in 0..ranks as usize {
                    let calls = stages(driver, ranks as usize, rank, rows).unwrap();
                    assert_eq!(calls.len(), 1);
                    let (launched, plan, args) = &calls[0];
                    assert_eq!(launched, symbol);
                    comm_setup::check(cubin, symbol, *plan, args, &buffers).unwrap();
                }
            }
        }
    }
}

/// The phased gather's and the F32 pair's launches bind their cubins'
/// entries over the rig's buffers at every row count, the pair as the comm
/// engine's two-shot all-reduce of the call's elements.
#[test]
fn phased_and_pair_invariance_launch_setup_is_valid_without_cuda() {
    for ranks in [2, 4] {
        let gather = group::gather_buffers(ranks as u32, (CAPACITY * SHARD) as u64).unwrap();
        let pair =
            comm::phased_buffers(ranks as u32, (CAPACITY * SHARD) as u64, DType::F32).unwrap();
        for (driver, cubin) in [
            (Driver::PhasedGather, PHASED_CUBIN),
            (Driver::ScatterPair, "comm_sm120"),
        ] {
            let mut buffers = BTreeMap::from([
                (100, (CAPACITY * SHARD * 4) as u64),
                (101, (CAPACITY * SHARD * 4 * ranks) as u64),
                (102, gather.epoch_bytes.max(pair.epoch_bytes)),
            ]);
            for peer in 0..ranks as u64 {
                buffers.insert(1 + peer, (CAPACITY * SHARD * ranks * 4) as u64);
                buffers.insert(11 + peer, gather.flag_bytes.max(pair.flag_bytes));
                buffers.insert(21 + peer, (CAPACITY * SHARD * 4) as u64);
            }
            for rows in [1, 2, 5, 64, CAPACITY] {
                for rank in 0..ranks {
                    let calls = stages(driver, ranks, rank, rows).unwrap();
                    let expected = match driver {
                        Driver::ScatterPair => vec![SCATTER_F32, GATHER_SCATTERED_F32],
                        _ => vec![kernels_sm120::context_parallel::phased::SYMBOL],
                    };
                    assert_eq!(
                        calls.iter().map(|c| c.0.as_str()).collect::<Vec<_>>(),
                        expected
                    );
                    for (symbol, plan, args) in &calls {
                        comm_setup::check(cubin, symbol, *plan, args, &buffers).unwrap();
                    }
                }
            }
        }
    }
}

/// Every declared comm entry that claims a property has a driver, so the
/// GPU harness never meets one it cannot launch.
#[test]
fn every_claiming_comm_entry_has_a_driver() {
    let missing = kernels_sm120::invariance::declarations(kernel_lab::catalogs())
        .into_iter()
        .filter(|entry| entry.op == OpKind::Comm && !entry.invariance.claims().is_empty())
        .filter_map(|entry| case(&entry).err())
        .collect::<Vec<_>>();
    assert!(missing.is_empty(), "{missing:?}");
}

/// The bundle's prefill op launches the exact entries up to 32 rows and the
/// round-once pair above; its exact verify op launches a row alone on the
/// one-row decode entries.
#[test]
fn bundle_op_drivers_launch_the_bundle_routes() {
    let prefill = Driver::Mimo(Mimo::Prefill);
    let exact = |combine| Driver::Mimo(Mimo::Exact(combine));
    let symbols = |driver, ranks, rows| {
        stages(driver, ranks, 0, rows)
            .unwrap()
            .into_iter()
            .map(|stage| stage.0)
            .collect::<Vec<_>>()
    };
    for ranks in [2, 4] {
        let pair = [
            comm::PREFILL_ROUND_ONCE_SYMBOL,
            comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
        ];
        assert_eq!(symbols(prefill, ranks, 33), pair);
        assert_eq!(symbols(prefill, ranks, 96), pair);
        assert_eq!(symbols(prefill, ranks, 32), [comm::VERIFY_SYMBOL]);
        for combine in [false, true] {
            let (plain, combined) = push_symbols(ranks as u32, 1);
            assert_eq!(
                symbols(exact(combine), ranks, 1),
                [if combine { combined } else { plain }]
            );
        }
    }
}

#[test]
fn prefill_invariance_drivers_launch_the_declared_family() {
    for ranks in [2, 4] {
        let symbols = [hc::SCATTER_SYMBOL, hc::GATHER_SYMBOL];
        for symbol in symbols {
            let (driver, _, _) = driver_of(symbol).unwrap();
            let pair = stages(driver, ranks, 0, 40).unwrap();
            assert_eq!(
                pair.iter()
                    .map(|stage| stage.0.as_str())
                    .collect::<Vec<_>>(),
                symbols
            );
            assert_eq!(pair[0].1.grid[0], 48);
            let short = stages(driver, ranks, 0, 1).unwrap();
            assert_eq!(short[0].0, hc::SYMBOL);
        }
    }
}

/// Each call of `calls` (row seeds) on one set of buffers, its units in
/// order.
fn run_calls<'a>(
    sessions: &[Session],
    entry: &Declared,
    driver: Driver,
    calls: impl IntoIterator<Item = &'a [u64]>,
) -> LabResult<Units> {
    let named = |error: String| format!("{}: {error}", entry.symbol);
    // An entry of a rank count runs on the run's first ranks of that count.
    let sessions = sessions
        .get(..ranks_of(driver, sessions))
        .ok_or_else(|| named("takes more ranks than the run has".into()))?;
    let rig = Rig::new(sessions, driver).map_err(named)?;
    let mut units = Vec::new();
    for seeds in calls {
        units.extend(rig.call(sessions, driver, seeds).map_err(named)?);
    }
    Ok(units)
}

impl Adapter for Comm {
    fn kind(&self) -> OpKind {
        OpKind::Comm
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        case(entry).map_or(PROMPT, |case| case.prompt)
    }

    /// The low-latency round entry's state holds [`ROUND_LL_ROWS`] rows,
    /// its 128 CTAs; every other entry takes the default counts.
    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        let most = match entry.symbol == comm::ll_f32::ROUND_SYMBOL {
            true => max.min(ROUND_LL_ROWS as u32),
            false => max,
        };
        crate::invariance::row_counts(most)
    }

    /// Prompt row r takes seed r; every chunk is its own call.
    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        _rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let driver = match case(entry)?.driver {
            Driver::Mimo(Mimo::Pair) => Driver::Mimo(Mimo::Prefill),
            driver => driver,
        };
        let mut start = 0;
        let chunks = ends
            .iter()
            .map(|&end| {
                let seeds = (u64::from(start)..u64::from(end)).collect::<Vec<_>>();
                start = end;
                seeds
            })
            .collect::<Vec<_>>();
        run_calls(sessions, entry, driver, chunks.iter().map(Vec::as_slice))
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        run_calls(sessions, entry, case(entry)?.driver, [seeds])
    }

    /// Each row in a call of one row. The verify entries' one-row rows are
    /// those the bundle's exact verify op launches for one row: the entries
    /// of a one-row decode step.
    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let mut driver = case(entry)?.driver;
        if let Driver::Mimo(Mimo::Push(symbol, combine)) = driver
            && [comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL].contains(&symbol)
        {
            driver = Driver::Mimo(Mimo::Exact(combine));
        }
        run_calls(
            sessions,
            entry,
            driver,
            seeds.iter().map(std::slice::from_ref),
        )
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        case(entry).map_or(4, |case| case.batch)
    }

    /// An entry needs as many ranks on distinct devices as it takes.
    fn unqualified(&self, sessions: &[Session], entry: &Declared) -> Option<String> {
        let case = case(entry).ok()?;
        let ranks = ranks_of(case.driver, sessions);
        (ranks > sessions.len()).then(|| {
            format!(
                "takes {ranks} ranks on distinct devices; this host's run has {}",
                sessions.len()
            )
        })
    }
}

/// A seeded 64-bit word for `salt` of `seed` (splitmix64).
fn mix(seed: u64, salt: u64) -> u64 {
    let mut z = seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Salts of the inputs every rank shares; a rank's own rows take
/// `rank + 1`.
const SHARED: u64 = 1 << 40;
const LOGITS: u64 = 2 << 40;

/// `width` BF16 values of every seed in turn, as bytes.
fn bf16_rows(seeds: &[u64], salt: u64, width: usize, scale: f32) -> Vec<u8> {
    seeds
        .iter()
        .flat_map(|&seed| pattern(mix(seed, salt), width, scale))
        .flat_map(u16::to_ne_bytes)
        .collect()
}

/// `width` raw elements of `bytes` each per seed (NaN payloads included),
/// none all poison bytes.
fn raw_rows(seeds: &[u64], salt: u64, width: usize, bytes: usize) -> Vec<u8> {
    seeds
        .iter()
        .flat_map(|&seed| {
            (0..width as u64).flat_map(move |i| {
                let mut element = mix(mix(seed, salt), i).to_ne_bytes()[..bytes].to_vec();
                if element.iter().all(|&byte| byte == POISON) {
                    element[0] = 0x7f;
                }
                element
            })
        })
        .collect()
}

fn hidden_query(ranks: u32, rows: u32) -> CommQuery {
    CommQuery::AllReduce {
        ranks,
        elements: u64::from(rows) * HIDDEN as u64,
        dtype: DType::Bf16,
        fused_residual: true,
        fused_rmsnorm: Some(RmsNormEpilogue {
            kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
            width: HIDDEN as u32,
            eps: EPS,
        }),
    }
}

fn qwen_query(ranks: u32) -> SigmoidMeanHyperConnectionQuery {
    SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks,
        streams: 4,
        width: QWEN_WIDTH as u32,
        eps_bits: EPS.to_bits(),
        capacity: CAPACITY as u32,
    }
}

/// Unit r of `rows`: row r of every part in order.
fn interleave(parts: &[Vec<u8>], rows: usize) -> Units {
    (0..rows)
        .map(|row| {
            parts
                .iter()
                .flat_map(|part| {
                    let width = part.len() / rows;
                    part[row * width..(row + 1) * width].iter().copied()
                })
                .collect()
        })
        .collect()
}

type Stage = (String, LaunchPlan, ArgumentList);

/// Rank `rank`'s launches of `rows` rows of the bundle's hidden all-reduce
/// op as its backend (`sm120::backend`) prepares it for `usage` and `need`,
/// of routed rows when `routed`, over this rig's buffers.
fn bundle_reduce(
    ranks: u32,
    rank: u32,
    rows: u32,
    usage: OperationUse,
    need: Need,
    routed: bool,
) -> LabResult<Vec<Stage>> {
    let err = |e: kernel_core::contract::Refusal| e.to_string();
    let backend = sm120::backend(Tp::TP4, Switches::configured(), true).map_err(err)?;
    let input = if routed {
        ReduceInput::Routed {
            routes: ROUTED as u32,
        }
    } else {
        ReduceInput::Rows
    };
    let op = backend
        .fused_all_reduce(&hidden_query(ranks, 1), input, usage, rank, need)
        .map_err(|e| format!("{e:?}"))?;
    let state = comm::fused_state_buffers(ranks, CAPACITY as u32).map_err(err)?;
    let view = |handle: u64, bytes: u64| View { handle, bytes };
    let hidden = (CAPACITY * HIDDEN * 2) as u64;
    let buffers = FusedAllReduceBuffers {
        input: view(100, ROUTED as u64 * hidden),
        residual: FusedResidual {
            input: view(103, hidden),
            output: view(108, hidden),
        },
        weight: view(104, HIDDEN as u64 * 2),
        output: view(101, hidden),
        live_rows: routed.then(|| view(105, 4)),
        ranks: (0..u64::from(ranks))
            .map(|peer| RankViews {
                output: view(21 + peer, hidden),
                // Launches take only the rank's own epoch.
                state: vec![
                    view(1 + peer, state.inbox_bytes),
                    view(11 + peer, state.flag_bytes),
                    view(102, state.epoch_bytes),
                ],
            })
            .collect(),
    };
    Ok(op
        .launches(&buffers, View::EMPTY, Counts::rows(rows))
        .map_err(err)?
        .into_iter()
        .map(|launch| (launch.symbol, launch.plan, launch.arguments))
        .collect())
}
/// A cubin and the entries a rig resolves from it.
type Entries = (&'static str, Vec<&'static str>);

/// Every rank's buffers by binding id and its resolved entries. Ids: 100
/// and up the rank's own buffers; peers' buffers bind at `base + peer`.
struct Rig {
    buffers: Vec<BTreeMap<u64, DeviceBuffer>>,
    functions: Vec<BTreeMap<&'static str, Function>>,
}

impl Rig {
    fn new(sessions: &[Session], driver: Driver) -> LabResult<Self> {
        let ranks = sessions.len() as u32;
        let err = |e: kernel_core::contract::Refusal| e.to_string();
        let rows = |width: usize, bytes: usize| CAPACITY * width * bytes;
        let mut weights = None;
        let (sizes, cubins): (Vec<(u64, usize)>, Vec<Entries>) = match driver {
            Driver::Mimo(_) => {
                let state = comm::fused_state_buffers(ranks, CAPACITY as u32).map_err(err)?;
                weights = Some((
                    104,
                    bf16_rows(&[9], SHARED, HIDDEN, 1.0)
                        .chunks_exact(2)
                        .flat_map(|pair| {
                            let bits = u16::from_ne_bytes([pair[0], pair[1]]);
                            let value = 1.0 + 0.5 * kernel_core::codec::bf16_to_f32(bits);
                            kernel_core::codec::f32_to_bf16(value).to_ne_bytes()
                        })
                        .collect::<Vec<_>>(),
                ));
                (
                    vec![
                        (100, rows(ROUTED * HIDDEN, 2)),
                        (101, rows(HIDDEN, 2)),
                        (102, state.epoch_bytes as usize),
                        (103, rows(HIDDEN, 2)),
                        (104, HIDDEN * 2),
                        (105, 4),
                        (106, state.inbox_bytes as usize),
                        (107, state.flag_bytes as usize),
                        (108, rows(HIDDEN, 2)),
                    ],
                    vec![
                        (comm::DECODE_CUBIN_V2, vec![comm::DECODE_SYMBOL_V2]),
                        (
                            "comm_sm120",
                            vec![
                                V2_SYMBOL,
                                COMBINE_SYMBOL,
                                comm::DECODE_COMBINE_SYMBOL,
                                comm::PREFILL_ROUND_ONCE_SYMBOL,
                                comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
                            ],
                        ),
                        (
                            model_mimo26_contract::sm120::comm::VERIFY_CUBIN,
                            vec![comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL],
                        ),
                    ],
                )
            }
            Driver::Qwen => {
                let state = hc::buffers(qwen_query(ranks)).map_err(err)?;
                weights = Some((103, bf16_rows(&[13], SHARED, QWEN_WIDE, 1.0)));
                (
                    vec![
                        (100, rows(QWEN_WIDTH, 2)),
                        (101, rows(QWEN_WIDE, 2)),
                        (102, rows(4, 2)),
                        (103, QWEN_WIDE * 2),
                        (104, rows(QWEN_WIDE, 2)),
                        (105, rows(QWEN_WIDE, 2)),
                        (106, state.inbox_bytes as usize),
                        (107, state.flags_bytes as usize),
                        (108, state.epoch_bytes as usize),
                        (109, 4),
                    ],
                    vec![(
                        "comm_qwen38",
                        vec![hc::SYMBOL, hc::SCATTER_SYMBOL, hc::GATHER_SYMBOL],
                    )],
                )
            }
            Driver::QwenDecode(symbol, entry_ranks) => {
                if entry_ranks != ranks {
                    return Err(format!("takes {entry_ranks} ranks, given {ranks}"));
                }
                // Flags, epochs and the LL region start zeroed, as below.
                let state = hc_decode::buffers(ranks).map_err(err)?;
                weights = Some((103, bf16_rows(&[13], SHARED, QWEN_WIDE, 1.0)));
                (
                    vec![
                        (100, rows(QWEN_WIDTH, 2)),
                        (101, rows(QWEN_WIDE, 2)),
                        (102, rows(4, 2)),
                        (103, QWEN_WIDE * 2),
                        (104, rows(QWEN_WIDE, 2)),
                        (105, rows(QWEN_WIDE, 2)),
                        (106, state.inbox_bytes as usize),
                        (107, state.flags_bytes as usize),
                        (108, state.epoch_bytes as usize),
                        (109, 4),
                    ],
                    vec![("comm_qwen38_decode", vec![symbol])],
                )
            }
            Driver::Gather(symbol, bytes) | Driver::Exchange(symbol, bytes) => (
                vec![
                    (100, rows(SHARD, bytes)),
                    (101, rows(SHARD * ranks as usize, bytes)),
                    (
                        102,
                        comm::GATHER_ROWS_EPOCH_BYTES.max(comm::EXCHANGE_ROWS_EPOCH_BYTES) as usize,
                    ),
                    (
                        107,
                        comm::GATHER_ROWS_FLAG_BYTES.max(comm::EXCHANGE_ROWS_FLAG_BYTES) as usize,
                    ),
                ],
                vec![("comm_sm120", vec![symbol])],
            ),
            Driver::Partials(symbol) => {
                let (inbox, flags, epoch, cubin) = if symbol == comm::ll_bf16::SYMBOL {
                    let state =
                        comm::ll_bf16::state_bytes(ranks, (PARTIALS_LL_ROWS * HIDDEN) as u64)
                            .map_err(err)?;
                    (state.inbox, 4, state.epochs, comm::ll_bf16::LAB_CUBIN)
                } else if symbol == comm::ll_f32::ROUND_SYMBOL {
                    let state = comm::ll_f32::state_bytes(ranks, (ROUND_LL_ROWS * HIDDEN) as u64)
                        .map_err(err)?;
                    (state.inbox, 4, state.epochs, comm::ll_f32::LAB_CUBIN)
                } else if rounded(symbol) {
                    let state = comm::phased_f32::buffers(ranks, (CAPACITY * HIDDEN) as u64)
                        .map_err(err)?;
                    (
                        state.inbox_bytes,
                        state.flag_bytes,
                        state.epoch_bytes,
                        comm::phased_f32::LAB_CUBIN,
                    )
                } else {
                    // The turns entry's flags hold the phased entries'.
                    let state = comm::phased_turns_bf16::buffers(ranks, (CAPACITY * HIDDEN) as u64)
                        .map_err(err)?;
                    let cubin = match symbol == comm::phased_turns_bf16::SYMBOL {
                        true => comm::phased_turns_bf16::LAB_CUBIN,
                        false => comm::phased_bf16::LAB_CUBIN,
                    };
                    (
                        state.inbox_bytes,
                        state.flag_bytes,
                        state.epoch_bytes,
                        cubin,
                    )
                };
                (
                    vec![
                        (100, rows(HIDDEN, 4)),
                        (101, rows(HIDDEN, 2)),
                        (102, epoch as usize),
                        (105, 4),
                        (106, inbox as usize),
                        (107, flags as usize),
                    ],
                    vec![(cubin, vec![symbol])],
                )
            }
            Driver::AllGather | Driver::PhasedGather => {
                let state = group::gather_buffers(ranks, (CAPACITY * SHARD) as u64).map_err(err)?;
                let cubin = match driver {
                    Driver::AllGather => ("comm_sm120", vec![group::ALLGATHER_SYMBOL]),
                    _ => (
                        PHASED_CUBIN,
                        vec![kernels_sm120::context_parallel::phased::SYMBOL],
                    ),
                };
                (
                    vec![
                        (100, rows(SHARD, 4)),
                        (101, rows(SHARD * ranks as usize, 4)),
                        (102, state.epoch_bytes as usize),
                        (107, state.flag_bytes as usize),
                    ],
                    vec![cubin],
                )
            }
            Driver::ScatterPair => {
                let state = comm::phased_buffers(ranks, (CAPACITY * SHARD) as u64, DType::F32)
                    .map_err(err)?;
                (
                    vec![
                        (100, rows(SHARD, 4)),
                        (101, rows(SHARD, 4)),
                        (102, state.epoch_bytes as usize),
                        (106, state.inbox_bytes as usize),
                        (107, state.flag_bytes as usize),
                    ],
                    vec![("comm_sm120", vec![SCATTER_F32, GATHER_SCATTERED_F32])],
                )
            }
        };
        let mut rig = Rig {
            buffers: Vec::new(),
            functions: Vec::new(),
        };
        for session in sessions {
            // Flags, epochs and the low-latency inbox region start zeroed.
            let mut buffers = BTreeMap::new();
            for &(id, bytes) in &sizes {
                let buffer = session.allocate(bytes.max(4))?;
                buffer
                    .zero_at_async(0, buffer.byte_len(), &session.stream)
                    .map_err(|e| e.to_string())?;
                buffers.insert(id, buffer);
            }
            if let Some((id, bytes)) = &weights {
                buffers[id]
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            // Resolve every entry now: module and function loads must not
            // run while a peer waits.
            let mut functions = BTreeMap::new();
            for (cubin, symbols) in &cubins {
                let module = session.module(cubin)?;
                for &symbol in symbols {
                    let function = module
                        .function(symbol)
                        .map_err(|e| format!("{symbol}: {e}"))?;
                    functions.insert(symbol, function);
                }
            }
            rig.buffers.push(buffers);
            rig.functions.push(functions);
        }
        Ok(rig)
    }

    /// One call of `seeds.len()` rows on every rank, each unit row r's
    /// outputs.
    fn call(&self, sessions: &[Session], driver: Driver, seeds: &[u64]) -> LabResult<Units> {
        let rows = seeds.len();
        if rows == 0 || rows > CAPACITY {
            return Err(format!("a call of {rows} rows (capacity {CAPACITY})"));
        }
        let outputs = outputs(driver);
        let peers: &[(u64, u64)] = match driver {
            Driver::Mimo(_) | Driver::Partials(_) => &[(1, 106), (11, 107), (21, 101)],
            Driver::Qwen | Driver::QwenDecode(..) => &[(1, 106), (11, 107)],
            Driver::Gather(..)
            | Driver::Exchange(..)
            | Driver::AllGather
            | Driver::PhasedGather => &[(1, 101), (11, 107)],
            Driver::ScatterPair => &[(1, 106), (11, 107), (21, 101)],
        };
        // The all-gathers' flag layouts follow the call's tiles, and the
        // pair's phases its elements, so each call starts on zeroed flags
        // and epochs.
        let fresh: &[u64] = match driver {
            Driver::AllGather | Driver::PhasedGather | Driver::ScatterPair => &[102, 107],
            _ => &[],
        };
        poison::synchronize(sessions)?;
        for (rank, session) in sessions.iter().enumerate() {
            for (id, bytes) in inputs(driver, rank, seeds) {
                self.buffers[rank][&id]
                    .copy_from_bytes(&bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            for id in fresh {
                let buffer = &self.buffers[rank][id];
                buffer
                    .zero_at_async(0, buffer.byte_len(), &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            let targets = outputs
                .iter()
                .map(|id| &self.buffers[rank][id])
                .collect::<Vec<_>>();
            poison::poison(session, &targets)?;
        }
        if !fresh.is_empty() {
            // No rank signals a peer before every peer's state is zeroed.
            poison::synchronize(sessions)?;
        }
        let bound = |rank: usize| {
            let mut bound = self.buffers[rank]
                .iter()
                .map(|(&id, buffer)| (id, buffer))
                .collect::<BTreeMap<_, _>>();
            for &(base, id) in peers {
                for (peer, buffers) in self.buffers.iter().enumerate() {
                    bound.insert(base + peer as u64, &buffers[&id]);
                }
            }
            bound
        };
        for (rank, session) in sessions.iter().enumerate() {
            let bound = bound(rank);
            for (symbol, plan, args) in stages(driver, sessions.len(), rank, rows)? {
                let function = self.functions[rank]
                    .get(symbol.as_str())
                    .ok_or_else(|| format!("{symbol} is not loaded"))?;
                launch::launch(
                    &session.context,
                    function,
                    &session.stream,
                    plan,
                    &args,
                    &Bindings {
                        buffers: bound.clone(),
                    },
                )?;
                kernel_lab::launched(&symbol);
            }
        }
        poison::synchronize(sessions)?;
        self.read(sessions, driver, rows, outputs)
    }

    /// Each row's outputs after a call of `rows` rows on poisoned outputs.
    /// Every output of every rank is checked first (`check_output`).
    fn read(
        &self,
        sessions: &[Session],
        driver: Driver,
        rows: usize,
        outputs: &[u64],
    ) -> LabResult<Units> {
        let ranks = sessions.len();
        // Each rank's outputs in the order of `outputs`.
        let mut held = Vec::new();
        for (rank, session) in sessions.iter().enumerate() {
            let mut own = Vec::new();
            for &id in outputs {
                let bytes = poison::read(session, &self.buffers[rank][&id])?;
                check_output(
                    &format!("{rows}-row call rank {rank} output {id}"),
                    driver,
                    ranks,
                    rank,
                    rows,
                    id,
                    &bytes,
                )?;
                own.push(bytes);
            }
            held.push(own);
        }
        let (width, element) = output_row(driver, ranks);
        let full = width * element.bytes();
        match driver {
            Driver::Mimo(_)
            | Driver::Qwen
            | Driver::QwenDecode(..)
            | Driver::Partials(_)
            | Driver::ScatterPair => {
                let parts = held
                    .into_iter()
                    .flatten()
                    .map(|mut bytes| {
                        bytes.truncate(rows * full);
                        bytes
                    })
                    .collect::<Vec<_>>();
                Ok(interleave(&parts, rows))
            }
            Driver::Gather(..) => {
                // Only the target, rank 0, receives rows.
                let gathered = &held[0][0][..rows * full];
                Ok(gathered.chunks(full).map(<[u8]>::to_vec).collect())
            }
            Driver::AllGather | Driver::PhasedGather => {
                // Every rank receives every source's block of rows, source
                // q's row r at `(q * rows + r) * SHARD`.
                let row = SHARD * 4;
                Ok((0..rows)
                    .map(|r| {
                        held.iter()
                            .flat_map(|own| {
                                (0..ranks).flat_map(move |q| {
                                    own[0][(q * rows + r) * row..][..row].iter().copied()
                                })
                            })
                            .collect()
                    })
                    .collect())
            }
            Driver::Exchange(..) => {
                // Each owner receives its block of rows.
                let mut units = vec![Vec::new(); rows];
                for (owner, own) in held.iter().enumerate() {
                    let owned = comm::exchange_rows_owned(rows as u32, ranks as u32, owner as u32);
                    let owned = owned.start as usize..owned.end as usize;
                    for (unit, row) in units[owned.clone()].iter_mut().zip(owned) {
                        *unit = own[0][row * full..(row + 1) * full].to_vec();
                    }
                }
                if units.iter().any(Vec::is_empty) {
                    return Err(format!("a {rows}-row exchange left a row without an owner"));
                }
                Ok(units)
            }
        }
    }
}

/// The buffers a call of `driver` writes on every rank, by binding id;
/// each is poisoned before every call. Inboxes, flags and epochs are never
/// poisoned.
fn outputs(driver: Driver) -> &'static [u64] {
    match driver {
        // The normed `output` and the pre-norm `residual_output`.
        Driver::Mimo(_) => &[101, 108],
        // `residual_out` and `normed`.
        Driver::Qwen | Driver::QwenDecode(..) => &[104, 105],
        // The rank's `outputs` slot, which peers write into (the pair's
        // sum, which every rank's gather writes).
        Driver::Gather(..)
        | Driver::Exchange(..)
        | Driver::AllGather
        | Driver::PhasedGather
        | Driver::ScatterPair => &[101],
        // The rounded sums, which the phased entries' peers write too.
        Driver::Partials(_) => &[101],
    }
}

/// One row of a driver's outputs on `ranks` ranks: its elements and their
/// encoding.
///
/// - MiMo: `HIDDEN` BF16 values in both outputs, the fused all-reduce's
///   `elements` of `rows * HIDDEN` (`comm::OneShotCall`; the v2 ABI's
///   `residual_output` holds the rounded pre-norm sum of the same rows).
/// - Qwen3.8: the four streams of `QWEN_WIDTH` BF16 values of the
///   hyper-connection inject and grouped norm, in both outputs
///   (`hc::EPILOGUE`).
/// - Gathers and exchanges: every rank's shard of `SHARD` elements, rank
///   q's at `(r * ranks + q) * shard` (`comm::GatherRowsCall`,
///   `row_exchange::ExchangeRowsCall`). They move bits, NaN payloads
///   included, so their elements are raw: 4 bytes for the F32 entries, 2
///   (one BF16 value) for the BF16 entries.
/// - The all-gather: `ranks * SHARD` raw 4-byte words per row of the call,
///   laid out source-major (`group::gather_arguments`); a call writes the
///   same number of words from the output's start.
/// - The BF16-partials entries: `HIDDEN` BF16 sums.
fn output_row(driver: Driver, ranks: usize) -> (usize, Element) {
    match driver {
        Driver::Partials(_) => (HIDDEN, Element::Bf16),
        Driver::Mimo(_) => (HIDDEN, Element::Bf16),
        Driver::Qwen | Driver::QwenDecode(..) => (QWEN_WIDE, Element::Bf16),
        Driver::Gather(_, bytes) | Driver::Exchange(_, bytes) => {
            (ranks * SHARD, Element::Raw(bytes))
        }
        Driver::AllGather | Driver::PhasedGather => (ranks * SHARD, Element::Raw(4)),
        Driver::ScatterPair => (SHARD, Element::F32),
    }
}

/// The rows of output `id` that rank `rank` of `ranks` holds after a call of
/// `rows` rows, as a write map, and the map's elements. Rows past the call's
/// live rows, and rows a rank does not receive, must stay untouched.
///
/// - MiMo and Qwen3.8: every rank writes rows `0..rows` of both outputs.
///   The Qwen entries take the live count from `live_rows` (109, set to the
///   call's rows), so the decode entry's prepared rows past it stay
///   untouched (`hc_decode::Call`).
/// - Gathers: only the target, rank 0, receives rows `0..rows`; every other
///   rank's output stays untouched.
/// - Exchanges: each rank receives the block it owns,
///   `comm::exchange_rows_owned`; trailing ranks may own none.
/// - The all-gather: every rank receives `ranks * rows * SHARD` words.
fn output_map(
    driver: Driver,
    ranks: usize,
    rank: usize,
    rows: usize,
    id: u64,
) -> LabResult<(WriteMap, Element)> {
    if !outputs(driver).contains(&id) {
        return Err(format!("binding {id} is not an output of the call"));
    }
    if rank >= ranks {
        return Err(format!("rank {rank} of {ranks}"));
    }
    let (width, element) = output_row(driver, ranks);
    let live = match driver {
        Driver::Mimo(_)
        | Driver::Qwen
        | Driver::QwenDecode(..)
        | Driver::AllGather
        | Driver::Partials(_)
        | Driver::PhasedGather
        | Driver::ScatterPair => 0..rows,
        Driver::Gather(..) if rank == 0 => 0..rows,
        Driver::Gather(..) => 0..0,
        Driver::Exchange(..) => {
            let owned = comm::exchange_rows_owned(rows as u32, ranks as u32, rank as u32);
            owned.start as usize..owned.end as usize
        }
    };
    let map = element.write_map()?.rows(width, live)?;
    Ok((map, element))
}

/// Checks the bytes of output `id` on `rank` after a call of `rows` rows on
/// poisoned outputs against `output_map`: each mapped element written (and
/// finite for the arithmetic outputs), every other byte still the poison. A
/// rank that receives no row must leave the whole output poisoned.
fn check_output(
    label: &str,
    driver: Driver,
    ranks: usize,
    rank: usize,
    rows: usize,
    id: u64,
    bytes: &[u8],
) -> LabResult<()> {
    let (map, element) = output_map(driver, ranks, rank, rows, id)?;
    if map.written_bytes() == 0 {
        poison::check(label, bytes, &map, Fill::Pattern(&[POISON]))
    } else {
        written(label, bytes, &map, element)
    }
}

/// Rank `rank`'s input bytes of a call over `seeds`, by binding id.
fn inputs(driver: Driver, rank: usize, seeds: &[u64]) -> Vec<(u64, Vec<u8>)> {
    let own = rank as u64 + 1;
    let live = (seeds.len() as u32).to_ne_bytes().to_vec();
    match driver {
        Driver::Mimo(route) => {
            let input = if matches!(route, Mimo::Push(_, true) | Mimo::Exact(true)) {
                bf16_rows(seeds, own, ROUTED * HIDDEN, 0.02)
            } else {
                bf16_rows(seeds, own, HIDDEN, 0.05)
            };
            vec![
                (100, input),
                (103, bf16_rows(seeds, SHARED, HIDDEN, 1.0)),
                (105, live),
            ]
        }
        Driver::Qwen | Driver::QwenDecode(..) => vec![
            (100, bf16_rows(seeds, own, QWEN_WIDTH, 0.5)),
            (101, bf16_rows(seeds, SHARED, QWEN_WIDE, 1.0)),
            (102, bf16_rows(seeds, LOGITS, 4, 1.0)),
            (109, live),
        ],
        Driver::Gather(_, bytes) | Driver::Exchange(_, bytes) => {
            vec![(100, raw_rows(seeds, own, SHARD, bytes))]
        }
        Driver::AllGather | Driver::PhasedGather => vec![(100, raw_rows(seeds, own, SHARD, 4))],
        Driver::Partials(_) => vec![(100, f32_rows(seeds, own, HIDDEN)), (105, live)],
        Driver::ScatterPair => vec![(100, f32_grid_rows(seeds, own, SHARD))],
    }
}

/// `width` F32 values in (-1, 1) of every seed in turn, as bytes: most of
/// them off the BF16 grid, so a partial rounded twice or not at all shows.
fn f32_rows(seeds: &[u64], salt: u64, width: usize) -> Vec<u8> {
    seeds
        .iter()
        .flat_map(|&seed| {
            (0..width as u64).map(move |i| {
                let bits = mix(mix(seed, salt), i) >> 40;
                bits as f32 / (1 << 23) as f32 - 1.0
            })
        })
        .flat_map(f32::to_ne_bytes)
        .collect()
}

/// `width` F32 values per seed in `[-2, 2)` on a 2^-12 grid, so rank sums
/// stay finite and their order shows in the low bits only through
/// rounding of values off the grid's sums.
fn f32_grid_rows(seeds: &[u64], salt: u64, width: usize) -> Vec<u8> {
    seeds
        .iter()
        .flat_map(|&seed| {
            (0..width as u64).flat_map(move |i| {
                let bits = mix(mix(seed, salt), i);
                let value = (bits >> 40) as f32 / (1u64 << 24) as f32 * 4.0 - 2.0;
                value.to_ne_bytes()
            })
        })
        .collect()
}

/// Rank `rank`'s launches of a call of `rows` rows.
fn stages(driver: Driver, ranks: usize, rank: usize, rows: usize) -> LabResult<Vec<Stage>> {
    let err = |e: kernel_core::contract::Refusal| e.to_string();
    let peers = |base: u64| (base..base + ranks as u64).collect::<Vec<_>>();
    let (rank, ranks, rows) = (rank as u32, ranks as u32, rows as u32);
    let base = comm::OneShotCall {
        input: 100,
        output: 101,
        peer_inboxes: peers(1),
        peer_flags: peers(11),
        epoch: 102,
        elements: u64::from(rows) * HIDDEN as u64,
        rank,
        ranks,
    };
    Ok(match driver {
        // The prefill op's row counts each take their own entries, so an op
        // of this rig's capacity launches what the bundle's of 8192 rows does
        // (which, like this one, needs only its rows).
        Driver::Mimo(Mimo::Prefill) => bundle_reduce(
            ranks,
            rank,
            rows,
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 8192,
                graph: false,
            },
            Need::rows(CAPACITY as u32),
            false,
        )?,
        Driver::Mimo(Mimo::Exact(combine)) => {
            let capacity = comm::verify_entry_rows(ranks).map_err(err)?;
            bundle_reduce(
                ranks,
                rank,
                rows,
                OperationUse {
                    phase: ExecutionPhase::Verify,
                    capacity,
                    graph: true,
                },
                Need::rows(capacity).with_exact(Invariance::one_row(capacity)),
                combine,
            )?
        }
        Driver::Mimo(Mimo::Pair) => {
            // The pair's grid at 33 rows serves fewer; the kernels take any
            // row count.
            let planned = rows.max(33);
            let plan = comm::plan(&hidden_query(ranks, planned), planned).map_err(err)?;
            let gather =
                comm::round_once_gather_arguments(&base, peers(21), 104, EPS, 108).map_err(err)?;
            vec![
                (
                    comm::PREFILL_ROUND_ONCE_SYMBOL.into(),
                    plan,
                    comm::round_once_scatter_arguments(base, 103).map_err(err)?,
                ),
                (comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL.into(), plan, gather),
            ]
        }
        Driver::Mimo(Mimo::Push(symbol, combine)) => {
            let call = comm::FusedWithResidualOutputCall {
                fused: comm::FusedOneShotCall {
                    base,
                    residual: Some(103),
                    weights: 104,
                    epsilon: EPS,
                },
                residual_output: 108,
            };
            let args = if combine {
                comm::fused_combine_arguments(call, Some(105))
            } else {
                comm::fused_arguments_v2(call)
            };
            vec![(
                symbol.into(),
                comm::decode_plan(ranks, rows).map_err(err)?,
                args.map_err(err)?,
            )]
        }
        Driver::Qwen => hc::launches(
            qwen_query(ranks),
            rows,
            hc::Call {
                input: 100,
                residual: 101,
                logits: 102,
                weight: 103,
                residual_out: 104,
                normed: 105,
                inboxes: peers(1),
                flags: peers(11),
                epoch: 108,
                live_rows: 109,
                capacity: CAPACITY as u32,
                rank,
            },
        )
        .map_err(err)?
        .into_iter()
        .map(|launch| (launch.symbol.into(), launch.plan, launch.arguments))
        .collect(),
        Driver::QwenDecode(symbol, _) => {
            // The graph bucket of the rows, all of them live.
            let prepared = rows.next_power_of_two();
            vec![(
                symbol.into(),
                hc_decode::plan(prepared).map_err(err)?,
                hc_decode::arguments(hc_decode::Call {
                    input: 100,
                    residual: 101,
                    logits: 102,
                    weight: 103,
                    residual_out: 104,
                    normed: 105,
                    inboxes: peers(1),
                    flags: peers(11),
                    epochs: 108,
                    live_rows: 109,
                    rows: prepared,
                    rank,
                })
                .map_err(err)?,
            )]
        }
        Driver::Gather(symbol, _) => vec![(
            symbol.into(),
            comm::gather_rows_plan(),
            comm::gather_rows_arguments(comm::GatherRowsCall {
                input: 100,
                outputs: peers(1),
                flags: peers(11),
                epoch: 102,
                row_indices: None,
                rows,
                shard: SHARD as u64,
                rank,
                ranks,
                target: 0,
            })
            .map_err(err)?,
        )],
        Driver::AllGather | Driver::PhasedGather => {
            let words = u64::from(rows) * SHARD as u64;
            let states = (0..u64::from(ranks))
                .map(|peer| GatherState {
                    output: 1 + peer,
                    flags: 11 + peer,
                    epoch: 102,
                })
                .collect::<Vec<_>>();
            let group_rank = GroupRank::of(rank, ranks, ranks).map_err(err)?;
            if let Driver::PhasedGather = driver {
                use kernels_sm120::context_parallel::phased;
                vec![(
                    phased::SYMBOL.into(),
                    phased::plan(),
                    phased::gather_arguments(group_rank, &states, 100, words).map_err(err)?,
                )]
            } else {
                vec![(
                    group::ALLGATHER_SYMBOL.into(),
                    group::gather_plan(words).map_err(err)?,
                    group::gather_arguments(group_rank, &states, 100, words).map_err(err)?,
                )]
            }
        }
        Driver::ScatterPair => {
            // As the comm engine's F32 all-reduce launches its two-shot
            // pair (`backend::all_reduce`): the scatter over the rank's
            // inboxes, then the gather into every rank's output.
            let elements = u64::from(rows) * SHARD as u64;
            let query = CommQuery::AllReduce {
                ranks,
                elements,
                dtype: DType::F32,
                fused_residual: false,
                fused_rmsnorm: None,
            };
            let plan = comm::plan_variant(comm::Variant::TwoShot, &query, rows).map_err(err)?;
            let scatter = comm::arguments(comm::OneShotCall {
                input: 100,
                output: 101,
                peer_inboxes: peers(1),
                peer_flags: peers(11),
                epoch: 102,
                elements,
                rank,
                ranks,
            })
            .map_err(err)?;
            let gather = ArgumentList(vec![
                kernel_core::contract::Argument::DeviceBuffer(101),
                kernel_core::contract::Argument::DevicePointerArray(peers(21)),
                kernel_core::contract::Argument::DevicePointerArray(peers(11)),
                kernel_core::contract::Argument::DeviceBuffer(102),
                kernel_core::contract::Argument::U64(elements),
                kernel_core::contract::Argument::U32(rank),
                kernel_core::contract::Argument::U32(ranks),
            ]);
            vec![
                (SCATTER_F32.into(), plan, scatter),
                (GATHER_SCATTERED_F32.into(), plan, gather),
            ]
        }
        Driver::Partials(symbol) if symbol == comm::ll_bf16::SYMBOL => {
            let elements = u64::from(rows) * HIDDEN as u64;
            vec![(
                symbol.into(),
                comm::ll_bf16::plan(ranks, elements).map_err(err)?,
                comm::ll_bf16::arguments(&comm::ll_bf16::Call {
                    input: 100,
                    output: 101,
                    peer_inboxes: peers(1),
                    epochs: 102,
                    elements,
                    capacity_elements: (PARTIALS_LL_ROWS * HIDDEN) as u64,
                    rank,
                    ranks,
                })
                .map_err(err)?,
            )]
        }
        Driver::Partials(symbol) if symbol == comm::ll_f32::ROUND_SYMBOL => {
            let elements = u64::from(rows) * HIDDEN as u64;
            vec![(
                symbol.into(),
                comm::ll_f32::plan(ranks, elements).map_err(err)?,
                comm::ll_f32::arguments(&comm::ll_f32::Call {
                    input: 100,
                    output: 101,
                    peer_inboxes: peers(1),
                    epochs: 102,
                    elements,
                    capacity_elements: (ROUND_LL_ROWS * HIDDEN) as u64,
                    rank,
                    ranks,
                })
                .map_err(err)?,
            )]
        }
        Driver::Partials(symbol) if rounded(symbol) => {
            let workers = match symbol == comm::phased_f32::ONE_SHOT_ROUND_SYMBOL {
                true => 0,
                false => PARTIALS_WORKERS,
            };
            let call = comm::phased_f32::Call {
                input: 100,
                output: 101,
                inboxes: peers(1),
                outputs: peers(21),
                flags: peers(11),
                epoch: 102,
                elements: u64::from(rows) * HIDDEN as u64,
                row_elements: HIDDEN as u32,
                live_rows: Some(105),
                rank,
                ranks,
                senders: PARTIALS_SENDERS,
            };
            vec![(
                symbol.into(),
                comm::phased_f32::plan_with(PARTIALS_SENDERS, workers).map_err(err)?,
                comm::phased_f32::arguments(symbol, call).map_err(err)?,
            )]
        }
        Driver::Partials(symbol) => {
            let workers = match symbol == comm::phased_bf16::ONE_SHOT_SYMBOL {
                true => 0,
                false => PARTIALS_WORKERS,
            };
            let call = comm::phased_bf16::Call {
                input: 100,
                output: 101,
                inboxes: peers(1),
                outputs: peers(21),
                flags: peers(11),
                epoch: 102,
                elements: u64::from(rows) * HIDDEN as u64,
                row_elements: HIDDEN as u32,
                live_rows: Some(105),
                rank,
                ranks,
                senders: PARTIALS_SENDERS,
            };
            vec![(
                symbol.into(),
                comm::phased_bf16::plan_with(PARTIALS_SENDERS, workers).map_err(err)?,
                comm::phased_bf16::arguments(symbol, call).map_err(err)?,
            )]
        }
        Driver::Exchange(symbol, _) => vec![(
            symbol.into(),
            comm::exchange_rows_plan(),
            comm::exchange_rows_arguments(comm::ExchangeRowsCall {
                input: 100,
                outputs: peers(1),
                flags: peers(11),
                epoch: 102,
                rows,
                shard: SHARD as u64,
                rank,
                ranks,
            })
            .map_err(err)?,
        )],
    })
}

/// The write checks of every driver's outputs reject an output a call left
/// unwritten, a hole, a NaN in an arithmetic output, a write past the live
/// rows or into a row another rank receives, and accept a correct image;
/// the BF16 exchange moves 2-byte elements.
#[test]
fn comm_write_checks_reject_unwritten_outputs() {
    let exchange = "chop_comm_exchange_rows_bf16_v1";
    let Ok((Driver::Exchange(symbol, 2), ..)) = driver_of(exchange) else {
        panic!("the BF16 exchange moves 2-byte elements");
    };
    assert_eq!(symbol, exchange);
    let (ranks, rows) = (4, 3);
    let drivers = [
        Driver::Mimo(Mimo::Pair),
        Driver::Mimo(Mimo::Push(COMBINE_SYMBOL, true)),
        Driver::Qwen,
        Driver::QwenDecode(hc_decode::SYMBOLS_V2[1], 4),
        Driver::Gather("chop_comm_gather_rows_f32_v1", 4),
        Driver::Gather("chop_comm_gather_rows_bf16_v1", 2),
        Driver::Exchange("chop_comm_exchange_rows_f32_v1", 4),
        Driver::Exchange(exchange, 2),
        Driver::AllGather,
        Driver::Partials(comm::ll_bf16::SYMBOL),
        Driver::Partials(comm::ll_f32::ROUND_SYMBOL),
        Driver::Partials(comm::phased_turns_bf16::SYMBOL),
        Driver::PhasedGather,
        Driver::ScatterPair,
    ];
    for driver in drivers {
        let (width, element) = output_row(driver, ranks);
        let row = width * element.bytes();
        for rank in 0..ranks {
            for &id in outputs(driver) {
                let label = format!("rank {rank} output {id}");
                let check =
                    |bytes: &[u8]| check_output(&label, driver, ranks, rank, rows, id, bytes);
                let (map, _) = output_map(driver, ranks, rank, rows, id).unwrap();
                let live = map.ranges().to_vec();
                // Three live rows on four ranks: the exchange's last rank
                // and every gather rank but the target receive none.
                let receives = match driver {
                    Driver::Gather(..) => rank == 0,
                    Driver::Exchange(..) => rank < 3,
                    _ => true,
                };
                assert_eq!(map.written_bytes() > 0, receives, "{label}");
                let poisoned = vec![POISON; CAPACITY * row];
                let mut image = poisoned.clone();
                let value = match element {
                    // A raw move keeps NaN payloads.
                    Element::Raw(n) => [0xc0, 0x7f, 0x01, 0x02][..n].to_vec(),
                    _ => 0x3fc0_u16.to_ne_bytes().to_vec(),
                };
                for range in &live {
                    image[range.clone()].copy_from_slice(&value.repeat(range.len() / value.len()));
                }
                assert!(check(&image).is_ok(), "{label}: {:?}", check(&image));
                // A write into a row this rank must not hold: the first row
                // past the ones it receives, or row 0 when it receives a
                // later block or none.
                let mut stray = image.clone();
                let past = live.last().map_or(0, |range| range.end);
                let at = if live.first().is_some_and(|range| range.start > 0) {
                    0
                } else {
                    past
                };
                stray[at] = 0x00;
                assert!(check(&stray).is_err(), "{label}: a stray write at {at}");
                if !receives {
                    continue;
                }
                // Nothing written.
                assert!(check(&poisoned).is_err(), "{label}: unwritten");
                // A hole: the last element of the last live row.
                let mut hole = image.clone();
                hole[past - element.bytes()..past].fill(POISON);
                assert!(check(&hole).is_err(), "{label}: a hole");
                // A NaN where an arithmetic output must be finite.
                if element == Element::Bf16 {
                    let mut nan = image.clone();
                    nan[live[0].start..][..2].copy_from_slice(&0x7fc0_u16.to_ne_bytes());
                    assert!(check(&nan).is_err(), "{label}: NaN");
                }
            }
        }
    }
    // The decode entry's prepared row past three live rows stays untouched.
    let decode = Driver::QwenDecode(hc_decode::SYMBOLS_V2[1], 4);
    let (map, _) = output_map(decode, 4, 0, 3, 105).unwrap();
    assert_eq!(map.ranges(), std::slice::from_ref(&(0..3 * QWEN_WIDE * 2)));
    // An exchange's owner holds its block only.
    let (map, _) = output_map(Driver::Exchange(exchange, 2), 4, 1, 5, 101).unwrap();
    assert_eq!(
        map.ranges(),
        std::slice::from_ref(&(2 * 4 * SHARD * 2..4 * 4 * SHARD * 2))
    );
    // A binding that is not an output has no map.
    assert!(output_map(Driver::Qwen, 4, 0, 3, 106).is_err());
}
