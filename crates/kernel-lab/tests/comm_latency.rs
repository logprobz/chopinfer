//! Decode all-reduce latency at 1 to 32 rows: the
//! lower-latency entries against the current ones bit for bit, and per-call
//! timing of both.
//!
//! - `mimo_decode_bitwise_g1`: `comm::DECODE_SYMBOL_V2` against
//!   `chop_comm_allreduce_oneshot_norm_bf16_v2` and
//!   `comm::DECODE_COMBINE_SYMBOL` against
//!   `chop_comm_allreduce_combine_norm_bf16_v1`, at every row count from 1
//!   to 32 (combine at full, partial, single and zero live rows), eagerly on
//!   one state per entry and in replayed PDL sequences of mixed row counts.
//!   The decode combine entry is also compared with itself on two states.
//! - `mimo_verify_rows_bitwise_g1`: every row of an exact verify all-reduce
//!   (`comm::VERIFY_SYMBOL` and `comm::VERIFY_COMBINE_SYMBOL` above the
//!   low-latency rows) against the same row reduced alone at one row by the
//!   entry an AR decode step takes.
//! - `mimo_decode_skew_g1`: the decode entries, then the bundle's selection
//!   of decode and current entries per call, on one state under the skew
//!   build's lag and across the epoch wrap, each call with its own inputs.
//!   The decode entries also run mixed with current, verify and phased
//!   calls.
//! - `qwen_decode_bitwise_g1`: `hc_decode::symbol_v2` against the standalone
//!   one-shot entry at every row count its plan takes and the same live
//!   counts, then v3 against v2 with poisoned per-bucket graph replays.
//! - `decode_latency_g4` (`CHOPINFER_KERNEL_BENCH=1`): per-call medians of
//!   both entries of each pair, 64 PDL calls per graph, and the MiMo entries
//!   at one rank as a control.
//!
//! Dense inputs mix cancelling pairs with values spread over 32 binades, so
//! another summation order of the ranks (or of a row's top-8 routed values)
//! changes the BF16 residual outputs of every row (`order_observable` checks
//! this on the host). The decode fixtures also cancel large pairs, require
//! order differences after BF16 rounding, and tag each call's payload per
//! source rank. All tests run at the selected rank count, 2, 4 or 8
//! (`CHOPINFER_KERNEL_GPUS`; each rank needs a distinct device).
//! Benchmarks write JSON under `CHOPINFER_COMM_LAT_OUT` (required), never
//! into the worktree.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{ArgumentList, CommQuery, DType, LaunchPlan, RmsNormEpilogue};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi};
use kernels_sm120::comm;
use model_mimo26_contract::sm120::comm::VERIFY_CUBIN;
use model_qwen38_contract::sm120::comm::{
    self as qwen38, Call as OneShotCall, SYMBOL as ONE_SHOT_SYMBOL,
    TRANSPORT as ONE_SHOT_TRANSPORT, buffers as one_shot_buffers, decode as hc_decode,
    plan as one_shot_plan, standalone_arguments, standalone_inbox_bytes,
};
use runtime_cuda::{DeviceBuffer, GraphExec};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[path = "common/comm_inputs.rs"]
mod comm_inputs;
#[path = "common/comm_setup.rs"]
mod comm_setup;

const MAX_ROWS: usize = 32;
const HIDDEN: usize = 4096;
const TOP_K: usize = 8;
const HC_WIDTH: usize = 2560;
const HC_STREAMS: usize = 4;
const HC_WIDE: usize = HC_WIDTH * HC_STREAMS;
const CALLS_PER_GRAPH: usize = 64;

fn mix(index: u64, seed: u64) -> u64 {
    let mut x = (index ^ seed.rotate_left(17)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^ (x >> 32)
}

fn pattern(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    (0..len as u64)
        .map(|index| f32_to_bf16(((mix(index, seed) % 2001) as f32 - 1000.0) / 1000.0 * scale))
        .collect()
}

/// Values whose exponents spread over 32 binades, drawn per element and
/// rank, so the rank-ordered F32 sums round in between and another order
/// changes their bits (about 8 % of the sums at four ranks, 30 % at eight).
fn partials(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    (0..len as u64)
        .map(|index| {
            let x = mix(index, seed);
            let value = ((x % 2001) as f32 - 1000.0) / 1000.0 * scale;
            let binade = ((x >> 20) % 32) as i32 - 16;
            f32_to_bf16(value * f32::powi(2.0, binade))
        })
        .collect()
}

/// Values of `n` summands (ranks, or a row's top-8 slots) at element `index`.
/// With `cancel` (and three or more summands), a pair of +-2^20 to 2^22 at
/// two positions drawn per element and values of `scale` (at most 0.05)
/// elsewhere: F32 rounds the values summed while one half of the pair is in
/// the sum to its grid of 1/8 or coarser, so the sum keeps other values in
/// another order. Otherwise values spread over 32 binades, as `partials`.
fn summands(index: u64, seed: u64, n: usize, scale: f32, cancel: bool) -> Vec<u16> {
    let mut values: Vec<f32> = (0..n as u64)
        .map(|position| {
            let x = mix(index, seed.wrapping_add(position.wrapping_mul(0x1_0001)));
            let value = ((x % 2001) as f32 - 1000.0) / 1000.0 * scale;
            if cancel {
                value
            } else {
                value * f32::powi(2.0, ((x >> 20) % 32) as i32 - 16)
            }
        })
        .collect();
    if cancel && n >= 3 {
        let x = mix(index, seed ^ 0x00ca_9ce1);
        let first = (x % n as u64) as usize;
        let second = (first + 1 + ((x >> 8) % (n as u64 - 1)) as usize) % n;
        let big = f32::powi(2.0, 20 + ((x >> 16) % 3) as i32);
        let big = if (x >> 24) & 1 == 0 { big } else { -big };
        values[first] = big;
        values[second] = -big;
    }
    values.into_iter().map(f32_to_bf16).collect()
}

/// MiMo plain inputs of `ranks` ranks, rows of `HIDDEN`: half the elements
/// cancel across the ranks, the rest spread over 32 binades.
fn mimo_inputs(ranks: usize, seed: u64) -> Vec<Vec<u16>> {
    let mut inputs = vec![Vec::with_capacity(MAX_ROWS * HIDDEN); ranks];
    for index in 0..(MAX_ROWS * HIDDEN) as u64 {
        let cancel = mix(index, seed ^ 0x0063_616e) & 1 == 0;
        for (rank, value) in summands(index, seed, ranks, 0.05, cancel)
            .into_iter()
            .enumerate()
        {
            inputs[rank].push(value);
        }
    }
    inputs
}

/// MiMo routed inputs of `ranks` ranks, `[row][slot][HIDDEN]`: a third of
/// the elements cancel across the ranks after the top-8 sum (slot 0 holds
/// the rank's summand, the other slots values of 0.001), a third cancel
/// across each rank's eight slots, the rest spread over 32 binades.
fn mimo_routed(ranks: usize, seed: u64) -> Vec<Vec<u16>> {
    let mut routed = vec![vec![0_u16; MAX_ROWS * TOP_K * HIDDEN]; ranks];
    for row in 0..MAX_ROWS {
        for column in 0..HIDDEN {
            let index = (row * HIDDEN + column) as u64;
            let class = mix(index, seed ^ 0x0063_6c73) % 3;
            let across = summands(index, seed, ranks, 0.02, true);
            for (rank, values) in routed.iter_mut().enumerate() {
                let own = seed + 1000 * (rank as u64 + 1);
                let slots = match class {
                    0 => (0..TOP_K as u64)
                        .map(|slot| match slot {
                            0 => across[rank],
                            _ => {
                                let x = mix(index, own + slot) % 2001;
                                f32_to_bf16((x as f32 - 1000.0) / 1000.0 * 0.001)
                            }
                        })
                        .collect(),
                    1 => summands(index, own, TOP_K, 0.02, true),
                    _ => summands(index, own, TOP_K, 0.02, false),
                };
                for (slot, value) in slots.into_iter().enumerate() {
                    values[(row * TOP_K + slot) * HIDDEN + column] = value;
                }
            }
        }
    }
    routed
}

/// A rank's summand at element `index` as the kernels load it: its input,
/// or (`combine`) the BF16 of the F32 sum of the row's routed values in
/// slot order `slots`.
fn summand(input: &[u16], index: usize, combine: bool, slots: &[usize]) -> f32 {
    if !combine {
        return bf16_to_f32(input[index]);
    }
    let (row, column) = (index / HIDDEN, index % HIDDEN);
    let sum = slots.iter().fold(0.0_f32, |s, &slot| {
        s + bf16_to_f32(input[(row * TOP_K + slot) * HIDDEN + column])
    });
    bf16_to_f32(f32_to_bf16(sum))
}

/// Ways a reduction could sum the ranks instead of in rank order.
#[derive(Clone, Copy, Debug)]
enum RankOrder {
    Forward,
    Reverse,
    /// Starting at rank k, wrapping around.
    Rotate(usize),
    /// Pairwise: halves summed separately, then added.
    Tree,
}

fn rank_sum(values: &[f32], order: RankOrder) -> f32 {
    let n = values.len();
    match order {
        RankOrder::Forward => values.iter().fold(0.0, |s, v| s + v),
        RankOrder::Reverse => values.iter().rev().fold(0.0, |s, v| s + v),
        RankOrder::Rotate(k) => (0..n).fold(0.0, |s, i| s + values[(k + i) % n]),
        RankOrder::Tree if n == 1 => values[0],
        RankOrder::Tree => {
            rank_sum(&values[..n / 2], RankOrder::Tree)
                + rank_sum(&values[n / 2..], RankOrder::Tree)
        }
    }
}

/// The summation order is observable in the MiMo residual outputs, which
/// the kernels round to BF16 (and normalize from): in every row, each other
/// order of the ranks (reversed, rotated, pairwise; three or more ranks, as
/// two sum alike in either order) and, with `combine`, the reversed order
/// of the top-8 slots, changes at least 1/64 of the row's outputs.
fn order_observable(inputs: &[Vec<u16>], combine: bool, residual: &[u16]) -> LabResult<()> {
    let ranks = inputs.len();
    let forward_slots: Vec<usize> = (0..TOP_K).collect();
    let reverse_slots: Vec<usize> = (0..TOP_K).rev().collect();
    let mut orders = Vec::new();
    if ranks >= 3 {
        orders.push(RankOrder::Reverse);
        orders.extend((1..ranks).map(RankOrder::Rotate));
        orders.push(RankOrder::Tree);
    }
    let output = |sum: f32, index: usize| f32_to_bf16(sum + bf16_to_f32(residual[index]));
    for row in 0..MAX_ROWS {
        let mut changed = vec![0_usize; orders.len() + 1];
        for column in 0..HIDDEN {
            let index = row * HIDDEN + column;
            let values: Vec<f32> = inputs
                .iter()
                .map(|input| summand(input, index, combine, &forward_slots))
                .collect();
            let expected = output(rank_sum(&values, RankOrder::Forward), index);
            for (k, &order) in orders.iter().enumerate() {
                changed[k] += usize::from(output(rank_sum(&values, order), index) != expected);
            }
            if combine {
                let reversed: Vec<f32> = inputs
                    .iter()
                    .map(|input| summand(input, index, true, &reverse_slots))
                    .collect();
                let other = output(rank_sum(&reversed, RankOrder::Forward), index);
                changed[orders.len()] += usize::from(other != expected);
            }
        }
        let names = orders
            .iter()
            .map(|order| format!("ranks {order:?}"))
            .chain(combine.then(|| "top-8 reversed".to_owned()));
        for (name, &count) in names.zip(&changed) {
            if count * 64 < HIDDEN {
                return Err(format!(
                    "row {row}: {name} changes only {count} of {HIDDEN} outputs; the bitwise check would be weak"
                ));
            }
        }
    }
    Ok(())
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn upload_bytes(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn check_received_clears(bytes: &[u8]) -> LabResult<()> {
    if bytes.is_empty() {
        return Err("no LL inbox bytes checked".into());
    }
    if let Some(at) = bytes.iter().position(|&value| value != 0) {
        return Err(format!(
            "LL inbox byte {at} was not cleared: {:#04x}",
            bytes[at]
        ));
    }
    Ok(())
}

/// The comm cubins, each loaded once for collective calls, the named entries
/// resolved before any launch, and the cubin of each entry.
struct Loaded {
    modules: BTreeMap<String, CollectiveModules>,
    cubins: BTreeMap<String, String>,
}

fn load(sessions: &[Session], symbols: &[(&str, String)]) -> LabResult<Loaded> {
    let mut modules: BTreeMap<String, CollectiveModules> = BTreeMap::new();
    let mut cubins = BTreeMap::new();
    for (cubin, symbol) in symbols {
        if !modules.contains_key(*cubin) {
            modules.insert(cubin.to_string(), CollectiveModules::load(sessions, cubin)?);
        }
        modules[*cubin].function(symbol)?;
        cubins.insert(symbol.clone(), cubin.to_string());
    }
    Ok(Loaded { modules, cubins })
}

impl Loaded {
    /// One call of `symbol` on every rank (or `only_rank`'s), rank `r`
    /// taking `call(r)`.
    fn launch<'a>(
        &self,
        sessions: &[Session],
        symbol: &str,
        plan: LaunchPlan,
        only_rank: Option<usize>,
        call: impl Fn(usize) -> LabResult<RankCall<'a>>,
    ) -> LabResult<()> {
        let modules = &self.modules[self.cubins[symbol].as_str()];
        launch::launch_collective(sessions, modules, symbol, plan, only_rank, call)
    }
}

fn synchronize(sessions: &[Session]) -> LabResult<()> {
    for session in sessions {
        session.stream.synchronize().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Poisons buffers `ids` of every rank once every rank is idle.
fn poison_ids(
    sessions: &[Session],
    buffers: &[BTreeMap<u64, DeviceBuffer>],
    ids: &[u64],
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for (session, owned) in sessions.iter().zip(buffers) {
        let targets = ids.iter().map(|id| &owned[id]).collect::<Vec<_>>();
        poison::poison(session, &targets)?;
    }
    Ok(())
}

/// The bytes of `buffer` after calls on poisoned outputs that wrote its
/// first `rows` rows of `width` BF16 values: live rows written, dead rows
/// still poison.
fn written_rows(
    session: &Session,
    buffer: &DeviceBuffer,
    rows: usize,
    width: usize,
    label: &str,
) -> LabResult<Vec<u8>> {
    written_rows_with_fill(session, buffer, rows, width, label, POISON)
}

fn written_rows_with_fill(
    session: &Session,
    buffer: &DeviceBuffer,
    rows: usize,
    width: usize,
    label: &str,
    fill: u8,
) -> LabResult<Vec<u8>> {
    let bytes = poison::read(session, buffer)?;
    poison::check(
        label,
        &bytes,
        &WriteMap::of(DType::Bf16).prefix(rows * width)?,
        Fill::Pattern(&[fill]),
    )?;
    Ok(bytes)
}

fn out_dir() -> LabResult<PathBuf> {
    let dir = std::env::var_os("CHOPINFER_COMM_LAT_OUT")
        .map(PathBuf::from)
        .ok_or("set CHOPINFER_COMM_LAT_OUT to a directory outside the worktree")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn write_json(name: &str, value: &Value) -> LabResult<()> {
    let path = out_dir()?.join(name);
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("{}: {e}", path.display()))
}

fn bench_enabled() -> bool {
    std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1")
}

/// Row counts of a benchmark: `CHOPINFER_COMM_LAT_ROWS` (comma separated)
/// or `default`.
fn bench_rows(default: &[u32]) -> Vec<u32> {
    std::env::var("CHOPINFER_COMM_LAT_ROWS")
        .ok()
        .map(|list| {
            list.split(',')
                .map(|part| part.trim().parse().expect("integer row count"))
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

// --- MiMo -----------------------------------------------------------------

/// A current entry and the decode entry that replaces it.
#[derive(Clone, Copy)]
struct DecodePair {
    reference: &'static str,
    candidate: &'static str,
    combine: bool,
}

impl DecodePair {
    fn skew_reference(self, chosen: &'static str) -> &'static str {
        if chosen == comm::PREFILL_ROUND_ONCE_SYMBOL {
            comm::VERIFY_SYMBOL
        } else if chosen.contains("_verify_") {
            chosen
        } else {
            self.reference
        }
    }
}

const MIMO_PAIRS: [DecodePair; 2] = [
    DecodePair {
        reference: "chop_comm_allreduce_oneshot_norm_bf16_v2",
        candidate: comm::DECODE_SYMBOL_V2,
        combine: false,
    },
    DecodePair {
        reference: "chop_comm_allreduce_combine_norm_bf16_v1",
        candidate: comm::DECODE_COMBINE_SYMBOL,
        combine: true,
    },
];

/// The plain or combine decode entry.
fn decode_symbol(combine: bool) -> &'static str {
    if combine {
        comm::DECODE_COMBINE_SYMBOL
    } else {
        comm::DECODE_SYMBOL_V2
    }
}

/// The cubin of a comm entry: plain decode v2 has its own.
fn entry_cubin(symbol: &str) -> &'static str {
    if symbol == comm::DECODE_SYMBOL_V2 {
        comm::DECODE_CUBIN_V2
    } else {
        "comm_sm120"
    }
}

/// Determinism check of the decode combine entry, compared on separate
/// states and through eager execution and graph replay.
const COMBINE_SELF_PAIR: DecodePair = DecodePair {
    reference: comm::DECODE_COMBINE_SYMBOL,
    candidate: comm::DECODE_COMBINE_SYMBOL,
    combine: true,
};

const INPUT: u64 = 100;
const ROUTED: u64 = 101;
const RESIDUAL: u64 = 102;
const WEIGHTS: u64 = 103;
const LIVE: u64 = 104;
/// Per-state buffers: output, residual output, epoch, inbox and flags of
/// state `slot` at `110 + 10 slot + k`; peer views from 1000.
fn state_id(slot: u64, k: u64) -> u64 {
    110 + 10 * slot + k
}
fn peer_inbox(slot: u64, peer: usize) -> u64 {
    1000 + 100 * slot + peer as u64
}
fn peer_flags(slot: u64, peer: usize) -> u64 {
    1000 + 100 * slot + 50 + peer as u64
}

struct DecodeLab<'a> {
    sessions: &'a [Session],
    buffers: Vec<BTreeMap<u64, DeviceBuffer>>,
    loaded: Loaded,
    sizes: comm::PushBuffers,
    states: u64,
    capacity: usize,
    host_inputs: Vec<Vec<u16>>,
    host_residual: Vec<u16>,
}

/// The input, residual and live-count buffers a MiMo call reads.
#[derive(Clone, Copy)]
struct CallInputs {
    input: u64,
    residual: u64,
    live: u64,
}

impl CallInputs {
    /// The lab's shared inputs: `INPUT` or `ROUTED`, `RESIDUAL`, `LIVE`.
    fn shared(combine: bool) -> Self {
        Self {
            input: if combine { ROUTED } else { INPUT },
            residual: RESIDUAL,
            live: LIVE,
        }
    }
}

impl<'a> DecodeLab<'a> {
    /// `states` comm states per rank; `modules` names the cubins whose
    /// entries `symbols` come from (entries renamed as those cubins do).
    fn new(sessions: &'a [Session], states: u64, symbols: &[(&str, String)]) -> LabResult<Self> {
        Self::with_capacity(sessions, states, symbols, MAX_ROWS)
    }

    fn with_capacity(
        sessions: &'a [Session],
        states: u64,
        symbols: &[(&str, String)],
        capacity: usize,
    ) -> LabResult<Self> {
        let ranks = sessions.len();
        let sizes = comm::fused_state_buffers(ranks as u32, capacity.max(MAX_ROWS + 1) as u32)
            .map_err(|e| e.to_string())?;
        let host_inputs = mimo_inputs(ranks, 40)
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .cycle()
                    .take(capacity * HIDDEN)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let host_routed = mimo_routed(ranks, 60)
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .cycle()
                    .take(capacity * TOP_K * HIDDEN)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let host_residual = pattern(41, capacity * HIDDEN, 1.0);
        order_observable(&host_inputs, false, &host_residual)?;
        order_observable(&host_routed, true, &host_residual)?;
        let host_weights: Vec<u16> = pattern(7, HIDDEN, 1.0)
            .into_iter()
            .map(|w| f32_to_bf16(0.75 + bf16_to_f32(w).abs() * 0.5))
            .collect();
        let mut buffers = Vec::new();
        for (rank, session) in sessions.iter().enumerate() {
            let mut b = BTreeMap::new();
            b.insert(INPUT, session.upload_bf16(&host_inputs[rank])?);
            b.insert(ROUTED, session.upload_bf16(&host_routed[rank])?);
            b.insert(RESIDUAL, session.upload_bf16(&host_residual)?);
            b.insert(WEIGHTS, session.upload_bf16(&host_weights)?);
            b.insert(
                LIVE,
                upload_bytes(session, &(capacity as u32).to_ne_bytes())?,
            );
            for slot in 0..states {
                for k in [0, 1] {
                    b.insert(state_id(slot, k), session.allocate(capacity * HIDDEN * 2)?);
                }
                b.insert(
                    state_id(slot, 2),
                    upload_bytes(session, &vec![0; sizes.epoch_bytes as usize])?,
                );
                // Only the low-latency prefix must start zeroed.
                let mut inbox = vec![0xa5; sizes.inbox_bytes as usize];
                inbox[..sizes.zeroed_inbox_bytes as usize].fill(0);
                b.insert(state_id(slot, 3), upload_bytes(session, &inbox)?);
                b.insert(
                    state_id(slot, 4),
                    upload_bytes(session, &vec![0; sizes.flag_bytes as usize])?,
                );
            }
            buffers.push(b);
        }
        let loaded = load(sessions, symbols)?;
        synchronize(sessions)?;
        let lab = Self {
            sessions,
            buffers,
            loaded,
            sizes,
            states,
            capacity,
            host_inputs,
            host_residual,
        };
        for slot in 0..states {
            lab.clear_outputs(slot)?;
        }
        Ok(lab)
    }

    fn install_norm_order_row(&self) -> LabResult<()> {
        // This row exposes a flattened warp tree in the final BF16 output,
        // as the CPU negative control below checks. Only rank 0 contributes;
        // residuals are zero and weights are one, for both input families.
        let sensitive = partials(998, HIDDEN, 0.05).repeat(MAX_ROWS);
        let zeros = vec![0_u16; MAX_ROWS * HIDDEN];
        let weights = vec![f32_to_bf16(1.0); HIDDEN];
        for (rank, session) in self.sessions.iter().enumerate() {
            let row = if rank == 0 { &sensitive } else { &zeros };
            let mut routed = vec![0_u16; MAX_ROWS * TOP_K * HIDDEN];
            for r in 0..MAX_ROWS {
                routed[r * TOP_K * HIDDEN..r * TOP_K * HIDDEN + HIDDEN]
                    .copy_from_slice(&row[r * HIDDEN..(r + 1) * HIDDEN]);
            }
            for (id, values) in [
                (INPUT, row.as_slice()),
                (ROUTED, &routed),
                (RESIDUAL, &zeros),
                (WEIGHTS, &weights),
            ] {
                let bytes: Vec<_> = values
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect();
                self.buffers[rank][&id]
                    .copy_from_bytes(&bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
        }
        synchronize(self.sessions)
    }

    fn ranks(&self) -> usize {
        self.sessions.len()
    }

    fn bindings(&self, rank: usize) -> Bindings<'_> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> =
            self.buffers[rank].iter().map(|(&k, v)| (k, v)).collect();
        for slot in 0..self.states {
            for (peer, owned) in self.buffers.iter().enumerate() {
                buffers.insert(peer_inbox(slot, peer), &owned[&state_id(slot, 3)]);
                buffers.insert(peer_flags(slot, peer), &owned[&state_id(slot, 4)]);
            }
        }
        Bindings { buffers }
    }

    fn arguments(
        &self,
        rank: usize,
        slot: u64,
        rows: u32,
        combine: bool,
        out: (u64, u64),
        inputs: CallInputs,
    ) -> LabResult<ArgumentList> {
        let ranks = self.ranks();
        let call = comm::FusedWithResidualOutputCall {
            fused: comm::FusedOneShotCall {
                base: comm::OneShotCall {
                    input: inputs.input,
                    output: out.0,
                    peer_inboxes: (0..ranks).map(|p| peer_inbox(slot, p)).collect(),
                    peer_flags: (0..ranks).map(|p| peer_flags(slot, p)).collect(),
                    epoch: state_id(slot, 2),
                    elements: u64::from(rows) * HIDDEN as u64,
                    rank: rank as u32,
                    ranks: ranks as u32,
                },
                residual: Some(inputs.residual),
                weights: WEIGHTS,
                epsilon: 1e-6,
            },
            residual_output: out.1,
        };
        if combine {
            comm::fused_combine_arguments(call, Some(inputs.live))
        } else {
            comm::fused_arguments_v2(call)
        }
        .map_err(|e| e.to_string())
    }

    /// The plan of `symbol`: the decode entries take `decode_plan`, the
    /// current ones the v2 plan.
    fn plan(&self, symbol: &str, rows: u32) -> LabResult<LaunchPlan> {
        let ranks = self.ranks() as u32;
        if symbol.contains("_decode_") {
            comm::decode_plan(ranks, rows)
        } else {
            let mut query = CommQuery::AllReduce {
                ranks,
                elements: u64::from(rows) * HIDDEN as u64,
                dtype: DType::Bf16,
                fused_residual: true,
                fused_rmsnorm: Some(RmsNormEpilogue {
                    kind: kernel_core::contract::RmsNormKind::HfBf16Intermediate,
                    width: HIDDEN as u32,
                    eps: 1e-6,
                }),
            };
            if symbol.contains("_wide_") {
                let CommQuery::AllReduce { elements, .. } = &mut query else {
                    unreachable!()
                };
                *elements = u64::from(rows.max(33)) * HIDDEN as u64;
                comm::graph_plan(&query, rows.max(33))
            } else {
                comm::plan(&query, rows)
            }
        }
        .map_err(|e| e.to_string())
    }

    /// One call of `symbol` on every rank (or `only_rank`'s).
    fn launch(
        &self,
        only_rank: Option<usize>,
        symbol: &str,
        slot: u64,
        rows: u32,
        combine: bool,
    ) -> LabResult<()> {
        self.launch_into(
            only_rank,
            symbol,
            slot,
            rows,
            combine,
            (state_id(slot, 0), state_id(slot, 1)),
            CallInputs::shared(combine),
        )
    }

    /// `launch` into the output and residual output buffers `out`, from
    /// `inputs`.
    #[allow(clippy::too_many_arguments)]
    fn launch_into(
        &self,
        only_rank: Option<usize>,
        symbol: &str,
        slot: u64,
        rows: u32,
        combine: bool,
        out: (u64, u64),
        inputs: CallInputs,
    ) -> LabResult<()> {
        if symbol.contains("_reduce_scatter_residual_bf16_") {
            if combine {
                return Err("phased fixture takes plain inputs".into());
            }
            return self.launch_phased_into(only_rank, symbol, slot, rows, out, inputs);
        }
        let plan = self.plan(symbol.trim_start_matches("chop_comm_skew_"), rows)?;
        self.loaded
            .launch(self.sessions, symbol, plan, only_rank, |rank| {
                Ok(RankCall {
                    args: self.arguments(rank, slot, rows, combine, out, inputs)?,
                    bindings: self.bindings(rank),
                })
            })
    }

    fn launch_phased_into(
        &self,
        only_rank: Option<usize>,
        scatter: &str,
        slot: u64,
        rows: u32,
        out: (u64, u64),
        inputs: CallInputs,
    ) -> LabResult<()> {
        // The phased stages accept short calls with the 33-row plan.
        let plan = self.plan(scatter, MAX_ROWS as u32 + 1)?;
        let gather = scatter.replace("reduce_scatter_residual", "allgather_presummed_norm");
        for (symbol, scatter_stage) in [(scatter, true), (gather.as_str(), false)] {
            self.loaded
                .launch(self.sessions, symbol, plan, only_rank, |rank| {
                    let base = comm::OneShotCall {
                        input: inputs.input,
                        output: out.0,
                        peer_inboxes: (0..self.ranks()).map(|p| peer_inbox(slot, p)).collect(),
                        peer_flags: (0..self.ranks()).map(|p| peer_flags(slot, p)).collect(),
                        epoch: state_id(slot, 2),
                        elements: u64::from(rows) * HIDDEN as u64,
                        rank: rank as u32,
                        ranks: self.ranks() as u32,
                    };
                    let mut bindings = self.bindings(rank);
                    let peers = (0..self.ranks())
                        .map(|peer| {
                            let id = 40_000 + peer as u64;
                            bindings.buffers.insert(id, &self.buffers[peer][&out.0]);
                            id
                        })
                        .collect();
                    let args = if scatter_stage {
                        comm::round_once_scatter_arguments(base, inputs.residual)
                    } else {
                        comm::round_once_gather_arguments(&base, peers, WEIGHTS, 1e-6, out.1)
                    }
                    .map_err(|e| e.to_string())?;
                    Ok(RankCall { args, bindings })
                })?;
        }
        Ok(())
    }

    /// Output buffers `first`, `first + 1`, ... on every rank, poisoned.
    fn add_outputs(&mut self, first: u64, count: u64) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            for id in first..first + count {
                let buffer = session.allocate(self.capacity * HIDDEN * 2)?;
                self.buffers[rank].insert(id, buffer);
            }
        }
        poison_ids(
            self.sessions,
            &self.buffers,
            &(first..first + count).collect::<Vec<_>>(),
        )
    }

    /// Every epoch cell of state `slot` set to `value`.
    fn seed_epochs(&self, slot: u64, value: u32) -> LabResult<()> {
        let cells = self.sizes.epoch_bytes as usize / 4;
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&state_id(slot, 2)]
                .copy_from_bytes(&u32_bytes(&vec![value; cells]), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    fn set_live(&self, live: u32) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&LIVE]
                .copy_from_bytes(&live.to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    /// Every LL slot starts zero and every received word must return to zero.
    /// Other protocols never write this prefix. Read it only after all ranks
    /// finish, including after a sequence with different row counts.
    fn check_ll_cleared(&self, slot: u64) -> LabResult<()> {
        synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            let mut bytes = vec![0; self.sizes.zeroed_inbox_bytes as usize];
            self.buffers[rank][&state_id(slot, 3)]
                .copy_to_bytes(&mut bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            check_received_clears(&bytes).map_err(|e| format!("rank {rank} state {slot}: {e}"))?;
        }
        Ok(())
    }

    /// Poisons the output and residual output of state `slot`.
    fn clear_outputs(&self, slot: u64) -> LabResult<()> {
        poison_ids(
            self.sessions,
            &self.buffers,
            &[state_id(slot, 0), state_id(slot, 1)],
        )
    }

    /// Output and residual output of state `slot` on `rank`.
    fn outputs(&self, rank: usize, slot: u64) -> LabResult<[Vec<u16>; 2]> {
        let session = &self.sessions[rank];
        Ok([
            session.download_bf16(
                &self.buffers[rank][&state_id(slot, 0)],
                self.capacity * HIDDEN,
            )?,
            session.download_bf16(
                &self.buffers[rank][&state_id(slot, 1)],
                self.capacity * HIDDEN,
            )?,
        ])
    }

    /// Both states' outputs hold their first `rows` rows and poison past
    /// them, state 1's outputs equal state 0's on every rank, and rank 0's
    /// equal every other rank's. The MiMo entries write every row below the
    /// call's rows of both outputs, the combine entries' rows at or past
    /// live with a zero sum (comm README "rows at or past `*live_rows`
    /// contribute zero"; push.cuh `input_vector`).
    fn compare(&self, what: &str, rows: usize) -> LabResult<()> {
        self.check_ll_cleared(0)?;
        self.check_ll_cleared(1)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for slot in [0, 1] {
                for k in [0, 1] {
                    written_rows(
                        session,
                        &self.buffers[rank][&state_id(slot, k)],
                        rows,
                        HIDDEN,
                        &format!("{what}: rank {rank} state {slot} output {k}"),
                    )?;
                }
            }
        }
        let first = self.outputs(0, 0)?;
        for rank in 0..self.ranks() {
            let expected = self.outputs(rank, 0)?;
            let actual = self.outputs(rank, 1)?;
            for (name, (e, a)) in ["output", "residual output"]
                .iter()
                .zip(expected.iter().zip(&actual))
            {
                if let Some(index) = e.iter().zip(a).position(|(x, y)| x != y) {
                    return Err(format!(
                        "{what}: rank {rank} {name} differs from the current entry at {index} ({:#06x} against {:#06x})",
                        a[index], e[index]
                    ));
                }
            }
            if expected != first {
                return Err(format!("{what}: rank {rank} differs from rank 0"));
            }
        }
        Ok(())
    }

    /// The pre-norm rows of a plain call equal the host's rank-ordered F32
    /// sum plus the residual, rounded once (the reference itself is checked).
    fn check_plain_reference(&self, rows: usize, host_inputs: &[Vec<u16>]) -> LabResult<()> {
        let [_, residual] = self.outputs(0, 0)?;
        for i in 0..rows * HIDDEN {
            let sum = host_inputs
                .iter()
                .fold(0.0_f32, |s, x| s + bf16_to_f32(x[i]));
            let expected = f32_to_bf16(sum + bf16_to_f32(self.host_residual[i]));
            if residual[i] != expected {
                return Err(format!(
                    "reference rows {rows}: pre-norm differs from the host at {i}"
                ));
            }
        }
        Ok(())
    }
}

fn mimo_bitwise(sessions: &[Session]) -> LabResult<Value> {
    let ranks = sessions.len();
    let pairs: Vec<_> = MIMO_PAIRS.into_iter().chain([COMBINE_SELF_PAIR]).collect();
    let symbols: Vec<(&str, String)> = pairs
        .iter()
        .flat_map(|p| [p.reference, p.candidate])
        .map(|symbol| (entry_cubin(symbol), symbol.to_owned()))
        .collect();
    let lab = DecodeLab::new(sessions, 2, &symbols)?;
    let mut cases = 0;
    // Every row count on one state per entry: the protocols switch as the
    // rows grow, and the calls advance the shared epochs.
    for rows in 1..=MAX_ROWS as u32 {
        for pair in &pairs {
            let lives = if pair.combine {
                let mut lives = vec![rows, rows - 1, 1, 0];
                lives.dedup();
                lives
            } else {
                vec![rows]
            };
            for live in lives {
                lab.set_live(live)?;
                for slot in [0, 1] {
                    lab.clear_outputs(slot)?;
                }
                lab.launch(None, pair.reference, 0, rows, pair.combine)?;
                synchronize(sessions)?;
                lab.launch(None, pair.candidate, 1, rows, pair.combine)?;
                synchronize(sessions)?;
                if !pair.combine && (rows == 1 || rows == 17 || rows == 32) {
                    lab.check_plain_reference(rows as usize, &lab.host_inputs)?;
                }
                lab.compare(
                    &format!("{} rows {rows} live {live}", pair.candidate),
                    rows as usize,
                )?;
                cases += 1;
            }
        }
    }
    // Replayed sequences of mixed rows, PDL on distinct devices: the last
    // call of each row count leaves its rows in the outputs, so both states
    // must end with the same buffers.
    let sequence: [u32; 9] = [32, 1, 17, 3, 16, 2, 9, 31, 1];
    // Every call writes its rows, so the sequence writes its widest call's.
    let widest = sequence.into_iter().max().unwrap_or(0) as usize;
    for pair in &pairs {
        lab.set_live(if pair.combine { 29 } else { MAX_ROWS as u32 })?;
        for slot in [0, 1] {
            lab.clear_outputs(slot)?;
        }
        for &rows in &sequence {
            lab.launch(None, pair.reference, 0, rows, pair.combine)?;
        }
        synchronize(sessions)?;
        let replay = RankReplay::new(sessions, |rank, _| {
            sequence
                .iter()
                .try_for_each(|&rows| lab.launch(Some(rank), pair.candidate, 1, rows, pair.combine))
        })?;
        for round in 0..3 {
            lab.clear_outputs(1)?;
            replay.launch(sessions)?;
            synchronize(sessions)?;
            lab.compare(
                &format!("{} sequence replay {round}", pair.candidate),
                widest,
            )?;
            cases += 1;
        }
    }
    lab.install_norm_order_row()?;
    for rows in [1, 2] {
        lab.set_live(rows)?;
        for pair in &pairs {
            lab.clear_outputs(0)?;
            lab.clear_outputs(1)?;
            lab.launch(None, pair.reference, 0, rows, pair.combine)?;
            synchronize(sessions)?;
            lab.launch(None, pair.candidate, 1, rows, pair.combine)?;
            synchronize(sessions)?;
            lab.compare(&format!("{} norm-order row", pair.candidate), rows as usize)?;
            cases += 1;
        }
    }
    Ok(json!({
        "ranks": ranks,
        "rows": format!("1..={MAX_ROWS}"),
        "pairs": pairs.iter().map(|p| json!([p.reference, p.candidate])).collect::<Vec<_>>(),
        "cases": cases,
        "state_bytes": {"inbox": lab.sizes.inbox_bytes, "flags": lab.sizes.flag_bytes},
    }))
}

/// Buffers of a row reduced alone: its input, routed rows and residual
/// copied to row 0, and live counts of one and zero rows.
fn row_input(r: usize) -> u64 {
    300 + r as u64
}
fn row_routed(r: usize) -> u64 {
    340 + r as u64
}
fn row_residual(r: usize) -> u64 {
    380 + r as u64
}
const LIVE_ONE: u64 = 420;
const LIVE_ZERO: u64 = 421;
/// Output and residual output of the one-row references.
const ALONE_OUT: u64 = 430;

/// The entries an exact verify step of `rows` rows takes (the bundle's
/// `launch::comm::exact_symbols`): at low-latency rows those of a decode
/// step of that many rows, above the verify entries.
fn exact_symbol(ranks: u32, rows: u32, combine: bool) -> LabResult<&'static str> {
    let low_latency = comm::push_protocol(ranks, rows).map_err(|e| e.to_string())?
        == comm::PushProtocol::LowLatency;
    Ok(if !low_latency {
        if combine {
            comm::VERIFY_COMBINE_SYMBOL
        } else {
            comm::VERIFY_SYMBOL
        }
    } else {
        decode_symbol(combine)
    })
}

/// Exact verify rows: row r of an exact all-reduce of R rows (R up to the
/// verify entries' one-shot rows, full and partial live rows with the
/// combine) equals the same row reduced alone at one row by the entry an AR
/// decode step takes, in the normalized and the residual output, on every
/// rank. Rows past the live count reduce as a zero routed input alone.
fn mimo_verify_rows(sessions: &[Session]) -> LabResult<Value> {
    let ranks = sessions.len();
    let symbols: Vec<(&str, String)> = MIMO_PAIRS
        .iter()
        .flat_map(|p| [p.reference, p.candidate])
        .map(|symbol| (entry_cubin(symbol), symbol.to_owned()))
        .chain(
            [comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL]
                .map(|symbol| (VERIFY_CUBIN, symbol.to_owned())),
        )
        .collect();
    let mut lab = DecodeLab::new(sessions, 2, &symbols)?;
    let routed = mimo_routed(ranks, 60);
    for (rank, session) in sessions.iter().enumerate() {
        for r in 0..MAX_ROWS {
            let buffers = &mut lab.buffers[rank];
            buffers.insert(
                row_input(r),
                session.upload_bf16(&lab.host_inputs[rank][r * HIDDEN..(r + 1) * HIDDEN])?,
            );
            buffers.insert(
                row_routed(r),
                session.upload_bf16(&routed[rank][r * TOP_K * HIDDEN..(r + 1) * TOP_K * HIDDEN])?,
            );
            buffers.insert(
                row_residual(r),
                session.upload_bf16(&lab.host_residual[r * HIDDEN..(r + 1) * HIDDEN])?,
            );
        }
        lab.buffers[rank].insert(LIVE_ONE, upload_bytes(session, &1_u32.to_ne_bytes())?);
        lab.buffers[rank].insert(LIVE_ZERO, upload_bytes(session, &0_u32.to_ne_bytes())?);
    }
    lab.add_outputs(ALONE_OUT, 2)?;
    let most = comm::verify_entry_rows(ranks as u32).map_err(|e| e.to_string())?;
    let mut cases = 0;
    for rows in 1..=most.min(MAX_ROWS as u32) {
        for combine in [false, true] {
            let candidate = exact_symbol(ranks as u32, rows, combine)?;
            let alone = exact_symbol(ranks as u32, 1, combine)?;
            let lives = if combine && rows > 1 {
                vec![rows, rows - 1]
            } else {
                vec![rows]
            };
            for live in lives {
                lab.set_live(live)?;
                lab.clear_outputs(1)?;
                lab.launch(None, candidate, 1, rows, combine)?;
                synchronize(sessions)?;
                for r in 0..rows as usize {
                    poison_ids(sessions, &lab.buffers, &[ALONE_OUT, ALONE_OUT + 1])?;
                    let inputs = CallInputs {
                        input: if combine { row_routed(r) } else { row_input(r) },
                        residual: row_residual(r),
                        live: if r < live as usize {
                            LIVE_ONE
                        } else {
                            LIVE_ZERO
                        },
                    };
                    lab.launch_into(
                        None,
                        alone,
                        0,
                        1,
                        combine,
                        (ALONE_OUT, ALONE_OUT + 1),
                        inputs,
                    )?;
                    synchronize(sessions)?;
                    for (rank, session) in sessions.iter().enumerate() {
                        let [output, residual] = lab.outputs(rank, 1)?;
                        for (name, step_rows, id) in [
                            ("output", &output, ALONE_OUT),
                            ("residual output", &residual, ALONE_OUT + 1),
                        ] {
                            let alone_row =
                                session.download_bf16(&lab.buffers[rank][&id], HIDDEN)?;
                            let row = &step_rows[r * HIDDEN..(r + 1) * HIDDEN];
                            if let Some(at) = row.iter().zip(&alone_row).position(|(a, b)| a != b) {
                                return Err(format!(
                                    "{candidate} rows {rows} live {live} row {r} rank {rank} {name}: element {at} is {:#06x}, alone ({alone}) {:#06x}",
                                    row[at], alone_row[at]
                                ));
                            }
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    Ok(json!({
        "ranks": ranks,
        "rows": format!("1..={}", most.min(MAX_ROWS as u32)),
        "entries": [comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL],
        "rows_checked": cases,
    }))
}

/// (rows, combine) of the MiMo synchronization sequence: every protocol of
/// the decode entries (LL, one-shot, two-shot), growing and shrinking,
/// plain and combined.
const MIMO_SEQUENCE: [(u32, bool); 14] = [
    (1, false),
    (2, true),
    (3, false),
    (16, true),
    (17, false),
    (32, true),
    (4, false),
    (2, false),
    (8, true),
    (1, true),
    (12, false),
    (31, true),
    (5, true),
    (16, false),
];

/// (rows, combine) of the sequence the bundle's selection runs
/// (`launch::comm::push_symbols`): the plain and combine
/// decode entries alternating at one and two rows and both at 12 rows, the
/// current entries at 13, and a last plain call at one row before the
/// replay's first decode combine call.
const SELECTED_SEQUENCE: [(u32, bool); 20] = [
    (1, true),
    (1, false),
    (1, true),
    (1, false),
    (2, true),
    (2, false),
    (2, true),
    (12, false),
    (12, true),
    (13, false),
    (13, true),
    (12, false),
    (13, true),
    (3, false),
    (3, true),
    (32, true),
    (32, false),
    (2, false),
    (2, true),
    (1, false),
];

const TRANSITION_TWO_SHOT: [&str; 2] = [
    "chop_comm_transition_allreduce_decode_norm_bf16_v2",
    "chop_comm_transition_allreduce_decode_combine_norm_bf16_v1",
];

fn transition_calls() -> Vec<(u32, bool, &'static str)> {
    let mut calls = Vec::new();
    for family in 0..6 {
        for previous_combine in [false, true] {
            if family == 4 && previous_combine {
                continue;
            }
            for combine in [false, true] {
                for rows in [1, 2] {
                    let previous = match (family, previous_combine) {
                        (0, false) => (2, comm::DECODE_SYMBOL_V2),
                        (0, true) => (2, comm::DECODE_COMBINE_SYMBOL),
                        (1, false) => (8, comm::VERIFY_SYMBOL),
                        (1, true) => (8, comm::VERIFY_COMBINE_SYMBOL),
                        (2, false) => (8, "chop_comm_allreduce_oneshot_norm_bf16_v2"),
                        (2, true) => (8, "chop_comm_allreduce_combine_norm_bf16_v1"),
                        (3, _) => (32, TRANSITION_TWO_SHOT[usize::from(previous_combine)]),
                        (4, _) => (16, comm::PREFILL_ROUND_ONCE_SYMBOL),
                        (5, false) => (64, comm::WIDE_SYMBOL),
                        (5, true) => (64, comm::WIDE_COMBINE_SYMBOL),
                        _ => unreachable!(),
                    };
                    calls.push((previous.0, previous_combine, previous.1));
                    calls.push((rows, combine, decode_symbol(combine)));
                }
            }
        }
    }
    calls
}

#[test]
fn transitions_cover_every_predecessor_and_selected_entry() {
    let calls = transition_calls();
    assert_eq!(calls.len(), 88);
    let mut edges = std::collections::BTreeSet::new();
    for pair in calls.chunks_exact(2) {
        let (rows, combine, symbol) = pair[1];
        assert_eq!(symbol, decode_symbol(combine));
        assert!([1, 2].contains(&rows));
        edges.insert((pair[0].2, combine, rows));
    }
    assert_eq!(edges.len(), 44);
    for symbol in calls.chunks_exact(2).map(|pair| pair[0].2) {
        for combine in [false, true] {
            for rows in [1, 2] {
                assert!(edges.contains(&(symbol, combine, rows)));
            }
        }
    }
}

#[test]
fn decode_v2_after_each_family_g1() {
    gpu_test_multi(2 << 30, |sessions| {
        let calls: Vec<_> = transition_calls()
            .into_iter()
            .map(|(rows, combine, _)| (rows, combine))
            .collect();
        let data = decode_skew(sessions, &calls, Selection::Transitions)?;
        receipt(sessions, "decode-v2-transitions", data)
    })
    .expect("v2 LL after every protocol, with poisoned outputs and distinct inputs");
}

/// Which entry each call of a skew sequence takes.
#[derive(Clone, Copy)]
enum Selection {
    /// Always the decode entry.
    Decode,
    /// Alternate the decode entries with the current, verify and phased
    /// families.
    Mixed,
    /// Every input family and LL row count immediately follows each protocol.
    Transitions,
    /// The decode or the current entry, as the bundle selects.
    Bundle,
}

impl Selection {
    fn symbol(self, ranks: u32, rows: u32, combine: bool, call: usize) -> LabResult<&'static str> {
        let pair = MIMO_PAIRS[usize::from(combine)];
        if matches!(self, Self::Transitions) {
            let (expected_rows, expected_combine, symbol) = transition_calls()[call];
            assert_eq!((rows, combine), (expected_rows, expected_combine));
            return Ok(symbol);
        }
        if matches!(self, Self::Mixed) {
            return Ok(match call % 6 {
                2 => pair.reference,
                3 if combine => comm::VERIFY_COMBINE_SYMBOL,
                3 => comm::VERIFY_SYMBOL,
                4 => comm::PREFILL_ROUND_ONCE_SYMBOL,
                _ => pair.candidate,
            });
        }
        let decode = match self {
            Selection::Mixed | Selection::Transitions => unreachable!(),
            Selection::Decode => true,
            Selection::Bundle => {
                comm::push_protocol(ranks, rows).map_err(|e| e.to_string())?;
                rows <= comm::DECODE_MAX_ROWS
            }
        };
        Ok(if decode {
            pair.candidate
        } else {
            pair.reference
        })
    }
}

/// The decode entries under cross-rank skew, as comm_sync.rs checks the
/// current ones: `calls` queued on every rank without host barriers, eagerly
/// and as replayed graphs, on the skew build (the rank chosen by the epoch
/// lags 20 us at a protocol step) and one state whose epochs start just
/// below the u32 wrap. The bundle gives each row bucket its own state, which
/// its plain and combined calls share; here every call shares one, whatever
/// its rows and entry. Each call reads its own inputs (per rank, and live
/// counts that change), so a stale peer row from another call shows, and
/// writes its own outputs, which must equal the current entry's (run call by
/// call on a separate state) on every rank.
fn decode_skew(
    sessions: &[Session],
    calls: &[(u32, bool)],
    selection: Selection,
) -> LabResult<Value> {
    let skew = |symbol: &str| symbol.replacen("chop_comm_", "chop_comm_skew_", 1);
    let ranks = sessions.len();
    let mut symbols: Vec<(&str, String)> = Vec::new();
    for p in MIMO_PAIRS {
        symbols.push(("comm_sm120", p.reference.to_owned()));
        symbols.push(("comm_sm120_skew", skew(p.reference)));
        symbols.push((
            if p.candidate == comm::DECODE_SYMBOL_V2 {
                "comm_decode_v2_skew_sm120"
            } else {
                "comm_sm120_skew"
            },
            skew(p.candidate),
        ));
    }
    for symbol in [comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL] {
        symbols.push((VERIFY_CUBIN, symbol.to_owned()));
        symbols.push(("comm_verify_skew_sm120", skew(symbol)));
    }
    for symbol in [
        comm::PREFILL_ROUND_ONCE_SYMBOL,
        comm::PREFILL_ROUND_ONCE_GATHER_SYMBOL,
    ] {
        symbols.push(("comm_sm120_skew", skew(symbol)));
    }
    let chosen = calls
        .iter()
        .enumerate()
        .map(|(call, &(rows, combine))| selection.symbol(ranks as u32, rows, combine, call))
        .collect::<LabResult<Vec<_>>>()?;
    for symbol in [comm::WIDE_SYMBOL, comm::WIDE_COMBINE_SYMBOL] {
        symbols.push(("comm_sm120", symbol.to_owned()));
        symbols.push(("comm_sm120_skew", skew(symbol)));
    }
    for symbol in TRANSITION_TWO_SHOT {
        symbols.push(("comm_sm120_transition", symbol.to_owned()));
    }
    let capacity = calls
        .iter()
        .map(|call| call.0 as usize)
        .max()
        .unwrap_or(MAX_ROWS)
        .max(MAX_ROWS);
    let mut lab = DecodeLab::with_capacity(sessions, 2, &symbols, capacity)?;
    let count = calls.len() as u64;
    // Call k: reference outputs 5000 + 4k and + 1, candidate + 2 and + 3;
    // input, residual and live count 20000 + 4k, + 1 and + 2.
    lab.add_outputs(5000, 4 * count)?;
    let inputs = |k: u64| CallInputs {
        input: 20000 + 4 * k,
        residual: 20001 + 4 * k,
        live: 20002 + 4 * k,
    };
    let mut lives = Vec::new();
    for (k, &(rows, combine)) in calls.iter().enumerate() {
        let seed = 3000 + 10 * k as u64;
        let values = if combine {
            mimo_routed(ranks, seed)
        } else {
            mimo_inputs(ranks, seed)
        };
        let values: Vec<Vec<u16>> = values
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .cycle()
                    .take(capacity * HIDDEN * if combine { TOP_K } else { 1 })
                    .collect()
            })
            .collect();
        let residual = pattern(seed + 1, capacity * HIDDEN, 1.0);
        let live = if matches!(selection, Selection::Transitions) {
            rows
        } else {
            [rows, rows.saturating_sub(1), 1, 0][k % 4]
        };
        lives.push(live);
        let ids = inputs(k as u64);
        for (rank, session) in sessions.iter().enumerate() {
            let b = &mut lab.buffers[rank];
            b.insert(ids.input, session.upload_bf16(&values[rank])?);
            b.insert(ids.residual, session.upload_bf16(&residual)?);
            b.insert(ids.live, upload_bytes(session, &live.to_ne_bytes())?);
        }
    }
    synchronize(sessions)?;
    let pair = |combine: bool| MIMO_PAIRS[usize::from(combine)];
    for (k, &(rows, combine)) in calls.iter().enumerate() {
        let k = k as u64;
        let out = (5000 + 4 * k, 5001 + 4 * k);
        lab.launch_into(
            None,
            if chosen[k as usize].contains("_wide_") {
                chosen[k as usize]
            } else if chosen[k as usize].contains("_transition_") {
                if combine {
                    comm::WIDE_COMBINE_SYMBOL
                } else {
                    comm::WIDE_SYMBOL
                }
            } else {
                pair(combine).skew_reference(chosen[k as usize])
            },
            0,
            rows,
            combine,
            out,
            inputs(k),
        )?;
        synchronize(sessions)?;
    }
    let queue = |only_rank: Option<usize>| -> LabResult<()> {
        for (k, &(rows, combine)) in calls.iter().enumerate() {
            let k = k as u64;
            let out = (5002 + 4 * k, 5003 + 4 * k);
            let symbol = if chosen[k as usize].contains("_transition_") {
                chosen[k as usize].to_owned()
            } else {
                skew(chosen[k as usize])
            };
            lab.launch_into(only_rank, &symbol, 1, rows, combine, out, inputs(k))?;
        }
        Ok(())
    };
    // Both entries write every row below the call's rows of both outputs
    // (dead rows at or past live with a zero sum) and nothing past them
    // (comm README "rows at or past `*live_rows` contribute zero").
    let check = |what: &str| -> LabResult<()> {
        lab.check_ll_cleared(0)?;
        lab.check_ll_cleared(1)?;
        for (rank, session) in sessions.iter().enumerate() {
            for k in 0..count {
                let rows = calls[k as usize].0 as usize;
                for half in 0..2 {
                    let written = |id: u64, entry: &str| {
                        written_rows(
                            session,
                            &lab.buffers[rank][&id],
                            rows,
                            HIDDEN,
                            &format!("{what}: rank {rank} call {k} {entry} output {half}"),
                        )
                    };
                    let expected = written(5000 + 4 * k + half, "current")?;
                    if written(5002 + 4 * k + half, "candidate")? != expected {
                        return Err(format!(
                            "{what}: rank {rank} call {k} {:?} ({}) differs from the current entry",
                            calls[k as usize], chosen[k as usize]
                        ));
                    }
                }
            }
        }
        Ok(())
    };
    let clear = |lab: &DecodeLab| -> LabResult<()> {
        let ids = (0..count)
            .flat_map(|k| [5002 + 4 * k, 5003 + 4 * k])
            .collect::<Vec<_>>();
        poison_ids(sessions, &lab.buffers, &ids)
    };
    lab.seed_epochs(1, u32::MAX - 5)?;
    queue(None)?;
    synchronize(sessions)?;
    check("eager across the epoch wrap")?;
    let replay = RankReplay::new(sessions, |rank, _| queue(Some(rank)))?;
    for round in 0..3 {
        clear(&lab)?;
        replay.launch(sessions)?;
        synchronize(sessions)?;
        check(&format!("graph replay {round}"))?;
    }
    Ok(json!({
        "ranks": ranks,
        "calls": calls,
        "symbols": chosen,
        "live": lives,
        "lag_ns": 20_000,
        "epoch_seed": u32::MAX - 5,
        "replays": 3,
    }))
}

const LL_REUSE_EPOCHS: [u32; 6] = [4095, 8191, 8192, 12288, 12289, 16385];

fn ll_flag_reuse(sessions: &[Session]) -> LabResult<Value> {
    let pairs = [MIMO_PAIRS[0], COMBINE_SELF_PAIR];
    let symbols: Vec<_> = pairs
        .iter()
        .flat_map(|pair| {
            [
                ("comm_sm120", pair.reference.to_owned()),
                (
                    if pair.candidate == comm::DECODE_SYMBOL_V2 {
                        "comm_decode_v2_skew_sm120"
                    } else {
                        "comm_sm120_skew"
                    },
                    variant_symbol("skew", pair.candidate),
                ),
            ]
        })
        .collect();
    let mut cases = 0;
    for rows in [1, 2] {
        for pair in pairs {
            let lab = DecodeLab::new(sessions, 2, &symbols)?;
            let symbol = variant_symbol("skew", pair.candidate);
            lab.set_live(rows)?;
            let replay = RankReplay::new(sessions, |rank, _| {
                lab.launch(Some(rank), &symbol, 1, rows, pair.combine)
            })?;
            // Each adjacent pair is 4096 epochs apart, using the same flag
            // and parity. Pairs straddle the flag wrap without going back.
            for (call, epoch) in LL_REUSE_EPOCHS.into_iter().enumerate() {
                for (rank, session) in sessions.iter().enumerate() {
                    let seed = 50_000 + 100 * call as u64 + rank as u64;
                    let input = partials(
                        seed,
                        MAX_ROWS * HIDDEN * if pair.combine { TOP_K } else { 1 },
                        0.05,
                    );
                    let residual = pattern(60_000 + call as u64, MAX_ROWS * HIDDEN, 1.0);
                    for (id, values) in [
                        (if pair.combine { ROUTED } else { INPUT }, input),
                        (RESIDUAL, residual),
                    ] {
                        let bytes: Vec<_> = values
                            .iter()
                            .flat_map(|value| value.to_ne_bytes())
                            .collect();
                        lab.buffers[rank][&id]
                            .copy_from_bytes(&bytes, &session.stream)
                            .map_err(|e| e.to_string())?;
                    }
                }
                synchronize(sessions)?;
                lab.seed_epochs(1, epoch - 1)?;
                lab.clear_outputs(0)?;
                lab.clear_outputs(1)?;
                lab.launch(None, pair.reference, 0, rows, pair.combine)?;
                synchronize(sessions)?;
                if call % 2 == 0 {
                    lab.launch(None, &symbol, 1, rows, pair.combine)?;
                } else {
                    replay.launch(sessions)?;
                }
                synchronize(sessions)?;
                lab.compare(&format!("same LL flag at epoch {epoch}"), rows as usize)?;
                cases += 1;
            }
        }
    }
    Ok(json!({"ranks": sessions.len(), "epochs": LL_REUSE_EPOCHS,
        "cases": cases, "lag_ns": 20_000, "cleared_ll_inboxes": true}))
}

#[test]
fn ll_clear_check_rejects_stale_words() {
    let mut inbox = [0; 32];
    check_received_clears(&inbox).unwrap();
    for at in [0, 4, 16, 20] {
        inbox[at] = 0xff;
        assert!(check_received_clears(&inbox).is_err());
        inbox[at] = 0;
    }
    assert!(check_received_clears(&[]).is_err());
    for pair in LL_REUSE_EPOCHS.chunks_exact(2) {
        assert_eq!(pair[1] - pair[0], 4096);
        assert_eq!(pair[0] & 0xfff, pair[1] & 0xfff);
        assert_eq!(pair[0] & 1, pair[1] & 1);
    }
}

#[test]
fn decode_ll_flag_reuse_g1() {
    gpu_test_multi(512 << 20, |sessions| {
        let data = ll_flag_reuse(sessions)?;
        receipt(sessions, "decode-ll-flag-reuse", data)
    })
    .expect("LL receive clearing and same-flag reuse under skew");
}

#[test]
fn decode_flat_tree_negative_g1() {
    gpu_test_multi(512 << 20, |sessions| {
        let pairs = [MIMO_PAIRS[0]];
        let symbols: Vec<_> = pairs
            .iter()
            .flat_map(|pair| {
                [
                    ("comm_sm120", pair.reference.to_owned()),
                    (
                        "comm_decode_flat_sm120",
                        variant_symbol("flat", pair.candidate),
                    ),
                ]
            })
            .collect();
        let lab = DecodeLab::new(sessions, 2, &symbols)?;
        lab.install_norm_order_row()?;
        let mut rejected = 0;
        for rows in [1, 2] {
            lab.set_live(rows)?;
            for pair in pairs {
                lab.clear_outputs(0)?;
                lab.clear_outputs(1)?;
                lab.launch(None, pair.reference, 0, rows, pair.combine)?;
                synchronize(sessions)?;
                let wrong = variant_symbol("flat", pair.candidate);
                lab.launch(None, &wrong, 1, rows, pair.combine)?;
                synchronize(sessions)?;
                let expected = lab.outputs(0, 0)?;
                let actual = lab.outputs(0, 1)?;
                if expected[1] != actual[1] {
                    return Err("wrong-tree control changed the pre-norm row".into());
                }
                for rank in 0..sessions.len() {
                    if lab.outputs(rank, 0)? != expected || lab.outputs(rank, 1)? != actual {
                        return Err("wrong-tree control disagrees across ranks".into());
                    }
                }
                // The same bitwise comparison must reject only the normalized
                // output, after all write maps and cleared inboxes pass.
                let mismatch = expected[0].iter().zip(&actual[0])
                    .position(|(a, b)| a != b)
                    .ok_or_else(|| format!("wrong tree went undetected: {wrong}, {rows} rows"))?;
                let expected_error = format!(
                    "{wrong}: rank 0 output differs from the current entry at {mismatch} ({:#06x} against {:#06x})",
                    actual[0][mismatch], expected[0][mismatch]
                );
                match lab.compare(&wrong, rows as usize) {
                    Err(error) if error == expected_error => rejected += 1,
                    Err(error) => return Err(error),
                    Ok(()) => {
                        return Err(format!("wrong tree went undetected: {wrong}, {rows} rows"));
                    }
                }
            }
        }
        receipt(
            sessions,
            "decode-flat-tree-negative",
            json!({"ranks": sessions.len(), "rejected": rejected, "seed": 998}),
        )
    })
    .expect("GPU bitwise comparison must reject the flattened tree");
}

// --- Qwen3.8 ----------------------------------------------------------------

const Q_INPUT: u64 = 200;
const Q_RESIDUAL: u64 = 201;
const Q_LOGITS: u64 = 202;
const Q_WEIGHT: u64 = 203;
const Q_LIVE: u64 = 204;
/// The standalone one-shot reference's inbox, flags and epoch, and the
/// state slot of their peer views.
const Q_ONE_SHOT: [u64; 3] = [206, 207, 208];
const ONE_SHOT_SLOT: u64 = 9;

/// The standalone one-shot's query at the decode buckets' capacity.
fn one_shot_query(ranks: usize) -> kernel_core::contract::SigmoidMeanHyperConnectionQuery {
    kernel_core::contract::SigmoidMeanHyperConnectionQuery::AllReduceInjectNorm {
        ranks: ranks as u32,
        streams: HC_STREAMS as u32,
        width: HC_WIDTH as u32,
        eps_bits: hc_decode::TRANSPORT.eps_bits,
        capacity: MAX_ROWS as u32,
    }
}
// Two BF16 summands almost cancel this BF16 residual after injection at
// logit -320 (z = -80). The separate F32 multiply/add and F64-tail product give
// distinct BF16 residual outputs. Keep every source marker intact.
fn inject_threshold_witness(inputs: &mut [Vec<u16>], residual: &mut [u16], call: u64) {
    let ranks = inputs.len();
    let scale = f32::powi(2.0, (call % 8) as i32);
    for row in 0..MAX_ROWS {
        for quarter in 0..4 {
            let column = quarter * (HC_WIDTH / 4) + ranks;
            for (rank, input) in inputs.iter_mut().enumerate() {
                let value = match rank {
                    0 => bf16_to_f32(0x7888),
                    1 => bf16_to_f32(0x72e2),
                    _ => 0.0,
                };
                input[row * HC_WIDTH + column] = f32_to_bf16(value * scale);
            }
            for stream in 0..HC_STREAMS {
                residual[(row * HC_STREAMS + stream) * HC_WIDTH + column] =
                    f32_to_bf16(bf16_to_f32(0xbf4c) * scale);
            }
        }
    }
}

fn q_state(slot: u64, k: u64) -> u64 {
    210 + 10 * slot + k
}

// The first generation reproduces an inject-contraction witness found
// during qualification, before running the broader shape/replay matrix.
const INJECT_ROUNDING_CALL: u64 = 136;

fn output_differences(expected: &[u16], actual: &[u16]) -> Value {
    assert_eq!(expected.len(), actual.len());
    let mut count = 0;
    let mut examples = Vec::new();
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if want != got {
            count += 1;
            if examples.len() < 8 {
                examples.push(json!({"index": index, "expected": format!("{want:#06x}"),
                    "actual": format!("{got:#06x}")}));
            }
        }
    }
    json!({"elements": expected.len(), "differing_elements": count, "first_differences": examples})
}

fn decode_lab_call(
    ranks: usize,
    rank: usize,
    slot: u64,
    rows: u32,
    io: [u64; 3],
) -> LabResult<(LaunchPlan, ArgumentList)> {
    let plan = hc_decode::plan(rows).map_err(|e| e.to_string())?;
    let args = hc_decode::arguments(hc_decode::Call {
        input: io[0],
        residual: Q_RESIDUAL,
        logits: Q_LOGITS,
        weight: Q_WEIGHT,
        residual_out: io[1],
        normed: io[2],
        inboxes: (0..ranks).map(|p| peer_inbox(slot, p)).collect(),
        flags: (0..ranks).map(|p| peer_flags(slot, p)).collect(),
        epochs: q_state(slot, 4),
        live_rows: Q_LIVE,
        rows,
        rank: rank as u32,
    })
    .map_err(|e| e.to_string())?;
    Ok((plan, args))
}

/// The standalone one-shot reference's call on its own state, at the decode
/// buckets' capacity and the live count `Q_LIVE`, without PDL.
fn one_shot_lab_call(
    ranks: usize,
    rank: usize,
    io: [u64; 3],
) -> LabResult<(LaunchPlan, ArgumentList)> {
    let mut plan =
        one_shot_plan(one_shot_query(ranks), MAX_ROWS as u32).map_err(|e| e.to_string())?;
    plan.pdl = false;
    let args = standalone_arguments(OneShotCall {
        input: io[0],
        residual: Q_RESIDUAL,
        logits: Q_LOGITS,
        weight: Q_WEIGHT,
        residual_out: io[1],
        normed: io[2],
        inboxes: (0..ranks).map(|p| peer_inbox(ONE_SHOT_SLOT, p)).collect(),
        flags: (0..ranks).map(|p| peer_flags(ONE_SHOT_SLOT, p)).collect(),
        epoch: Q_ONE_SHOT[2],
        live_rows: Q_LIVE,
        capacity: MAX_ROWS as u32,
        rank: rank as u32,
    })
    .map_err(|e| e.to_string())?;
    Ok((plan, args))
}

struct QwenLab<'a> {
    sessions: &'a [Session],
    buffers: Vec<BTreeMap<u64, DeviceBuffer>>,
    loaded: Loaded,
    states: u64,
}

impl<'a> QwenLab<'a> {
    fn new(sessions: &'a [Session], states: u64, symbols: &[(&str, String)]) -> LabResult<Self> {
        let ranks = sessions.len() as u32;
        let sizes = hc_decode::buffers(ranks).map_err(|e| e.to_string())?;
        let one_shot =
            one_shot_buffers(one_shot_query(ranks as usize)).map_err(|e| e.to_string())?;
        let host_partials = comm_inputs::inputs(ranks as usize, MAX_ROWS, HC_WIDTH, 0);
        comm_inputs::require_order_sensitive(&host_partials, HC_WIDTH)?;
        let mut buffers = Vec::new();
        for (rank, session) in sessions.iter().enumerate() {
            let mut b = BTreeMap::new();
            b.insert(Q_INPUT, session.upload_bf16(&host_partials[rank])?);
            let mut residual = pattern(1, MAX_ROWS * HC_WIDE, 1.0);
            comm_inputs::clear_marker_residuals(&mut residual, ranks as usize);
            b.insert(Q_RESIDUAL, session.upload_bf16(&residual)?);
            b.insert(
                Q_LOGITS,
                session.upload_bf16(&pattern(2, MAX_ROWS * HC_STREAMS, 4.0))?,
            );
            b.insert(Q_WEIGHT, session.upload_bf16(&pattern(3, HC_WIDE, 0.5))?);
            b.insert(
                Q_LIVE,
                upload_bytes(session, &(MAX_ROWS as u32).to_ne_bytes())?,
            );
            b.insert(
                Q_ONE_SHOT[0],
                session.allocate(standalone_inbox_bytes(ranks, MAX_ROWS as u32) as usize)?,
            );
            b.insert(
                Q_ONE_SHOT[1],
                upload_bytes(session, &vec![0; one_shot.flags_bytes as usize])?,
            );
            b.insert(
                Q_ONE_SHOT[2],
                upload_bytes(session, &vec![0; one_shot.epoch_bytes as usize])?,
            );
            for slot in 0..states {
                for k in [0, 1] {
                    b.insert(q_state(slot, k), session.allocate(MAX_ROWS * HC_WIDE * 2)?);
                }
                let mut inbox = vec![0xff; sizes.inbox_bytes as usize];
                inbox[..sizes.zeroed_inbox_bytes as usize].fill(0);
                b.insert(q_state(slot, 2), upload_bytes(session, &inbox)?);
                b.insert(
                    q_state(slot, 3),
                    upload_bytes(session, &vec![0; sizes.flags_bytes as usize])?,
                );
                b.insert(
                    q_state(slot, 4),
                    upload_bytes(session, &vec![0; sizes.epoch_bytes as usize])?,
                );
            }
            buffers.push(b);
        }
        let loaded = load(sessions, symbols)?;
        synchronize(sessions)?;
        let lab = Self {
            sessions,
            buffers,
            loaded,
            states,
        };
        for slot in 0..states {
            lab.clear_outputs(slot)?;
        }
        Ok(lab)
    }

    fn bindings(&self, rank: usize) -> Bindings<'_> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> =
            self.buffers[rank].iter().map(|(&k, v)| (k, v)).collect();
        for slot in 0..self.states {
            for (peer, owned) in self.buffers.iter().enumerate() {
                buffers.insert(peer_inbox(slot, peer), &owned[&q_state(slot, 2)]);
                buffers.insert(peer_flags(slot, peer), &owned[&q_state(slot, 3)]);
            }
        }
        for (peer, owned) in self.buffers.iter().enumerate() {
            buffers.insert(peer_inbox(ONE_SHOT_SLOT, peer), &owned[&Q_ONE_SHOT[0]]);
            buffers.insert(peer_flags(ONE_SHOT_SLOT, peer), &owned[&Q_ONE_SHOT[1]]);
        }
        Bindings { buffers }
    }

    /// One call of `reference` writing state 0's outputs from input `io[0]`:
    /// the standalone one-shot (`ONE_SHOT_SYMBOL`, at the decode buckets'
    /// capacity and the live count) or a decode entry for `rows`.
    fn launch_reference(&self, reference: &str, rows: u32, io: [u64; 3]) -> LabResult<()> {
        if reference != ONE_SHOT_SYMBOL {
            return self.launch_with_io(None, reference, 0, rows, io);
        }
        let ranks = self.sessions.len();
        let (plan, _) = one_shot_lab_call(ranks, 0, io)?;
        self.loaded
            .launch(self.sessions, reference, plan, None, |rank| {
                let (_, args) = one_shot_lab_call(ranks, rank, io)?;
                Ok(RankCall {
                    args,
                    bindings: self.bindings(rank),
                })
            })
    }

    /// One call of `symbol` on every rank (or `only_rank`'s).
    fn launch(
        &self,
        only_rank: Option<usize>,
        symbol: &str,
        slot: u64,
        rows: u32,
    ) -> LabResult<()> {
        self.launch_with_io(
            only_rank,
            symbol,
            slot,
            rows,
            [Q_INPUT, q_state(slot, 0), q_state(slot, 1)],
        )
    }

    /// `io` holds input, residual output and normed output handles.
    fn launch_with_io(
        &self,
        only_rank: Option<usize>,
        symbol: &str,
        slot: u64,
        rows: u32,
        io: [u64; 3],
    ) -> LabResult<()> {
        let ranks = self.sessions.len();
        let (plan, _) = decode_lab_call(ranks, 0, slot, rows, io)?;
        self.loaded
            .launch(self.sessions, symbol, plan, only_rank, |rank| {
                let (_, args) = decode_lab_call(ranks, rank, slot, rows, io)?;
                Ok(RankCall {
                    args,
                    bindings: self.bindings(rank),
                })
            })
    }

    /// Update fixed graph input addresses only after all ranks are idle.
    fn set_input(&self, id: u64, call: u64, reverse: bool) -> LabResult<()> {
        let inputs = comm_inputs::inputs(self.sessions.len(), MAX_ROWS, HC_WIDTH, call);
        self.copy_inputs(id, inputs, reverse)
    }

    fn copy_inputs(&self, id: u64, mut inputs: Vec<Vec<u16>>, reverse: bool) -> LabResult<()> {
        synchronize(self.sessions)?;
        comm_inputs::require_order_sensitive(&inputs, HC_WIDTH)?;
        if reverse {
            inputs.reverse();
        }
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&id]
                .copy_from_bytes(&comm_inputs::bytes(&inputs[rank]), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    fn set_inject_inputs(&self, call: u64) -> LabResult<()> {
        let mut inputs = comm_inputs::inputs(self.sessions.len(), MAX_ROWS, HC_WIDTH, call);
        for input in &mut inputs {
            for value in input {
                *value = f32_to_bf16(bf16_to_f32(*value) * f32::powi(2.0, 80));
            }
        }
        let mut residual = pattern(1, MAX_ROWS * HC_WIDE, 1.0);
        comm_inputs::clear_marker_residuals(&mut residual, self.sessions.len());
        inject_threshold_witness(&mut inputs, &mut residual, call);
        self.copy_inputs(Q_INPUT, inputs, false)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&Q_RESIDUAL]
                .copy_from_bytes(&comm_inputs::bytes(&residual), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    fn set_inject_logits(&self, shift: usize) -> LabResult<()> {
        // BF16 neighbors around the -320 logit threshold, gradual
        // underflow of the F32 sigmoid, and complete product underflow.
        let logits = [
            -318.0, -320.0, -322.0, -352.0, -400.0, -576.0, -640.0, -1600.0,
        ];
        let values: Vec<_> = (0..MAX_ROWS * HC_STREAMS)
            .map(|i| f32_to_bf16(logits[(i + shift) % logits.len()]))
            .collect();
        synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&Q_LOGITS]
                .copy_from_bytes(&comm_inputs::bytes(&values), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    fn set_nonfinite_logits(&self, shift: usize) -> LabResult<()> {
        let logits = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let values: Vec<_> = (0..MAX_ROWS * HC_STREAMS)
            .map(|i| f32_to_bf16(logits[(i + shift) % logits.len()]))
            .collect();
        synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&Q_LOGITS]
                .copy_from_bytes(&comm_inputs::bytes(&values), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    fn fill_outputs(&self, slot: u64, fill: u8) -> LabResult<()> {
        synchronize(self.sessions)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            for k in [0, 1] {
                poison::fill_all(session, &self.buffers[rank][&q_state(slot, k)], &[fill])?;
            }
        }
        Ok(())
    }

    fn set_live(&self, live: u32) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            self.buffers[rank][&Q_LIVE]
                .copy_from_bytes(&live.to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
        }
        synchronize(self.sessions)
    }

    /// Poisons the residual output and normed output of state `slot`.
    fn clear_outputs(&self, slot: u64) -> LabResult<()> {
        poison_ids(
            self.sessions,
            &self.buffers,
            &[q_state(slot, 0), q_state(slot, 1)],
        )
    }

    fn outputs(&self, rank: usize, ids: [u64; 2]) -> LabResult<[Vec<u16>; 2]> {
        let session = &self.sessions[rank];
        Ok([
            session.download_bf16(&self.buffers[rank][&ids[0]], MAX_ROWS * HC_WIDE)?,
            session.download_bf16(&self.buffers[rank][&ids[1]], MAX_ROWS * HC_WIDE)?,
        ])
    }

    /// Both states' outputs hold their first `live` rows and poison past
    /// them, and state 1's equal state 0's on every rank and rank 0's. The
    /// entries write only rows below the live count: hyper_connection_decode.cuh
    /// returns from every CTA whose row is at or past live.
    fn compare(&self, what: &str, live: usize) -> LabResult<()> {
        self.compare_with(what, live, [q_state(1, 0), q_state(1, 1)])
    }

    fn compare_with(&self, what: &str, live: usize, ids: [u64; 2]) -> LabResult<()> {
        self.compare_with_fill(what, live, ids, POISON)
    }

    fn compare_with_fill(&self, what: &str, live: usize, ids: [u64; 2], fill: u8) -> LabResult<()> {
        let mut write_errors = Vec::new();
        for (rank, session) in self.sessions.iter().enumerate() {
            for (slot, outputs) in [[q_state(0, 0), q_state(0, 1)], ids]
                .into_iter()
                .enumerate()
            {
                for (k, id) in outputs.into_iter().enumerate() {
                    if let Err(error) = written_rows_with_fill(
                        session,
                        &self.buffers[rank][&id],
                        live,
                        HC_WIDE,
                        &format!("{what}: rank {rank} state {slot} output {k}"),
                        fill,
                    ) {
                        write_errors.push(error);
                    }
                }
            }
        }
        let first = self.outputs(0, [q_state(0, 0), q_state(0, 1)])?;
        let mut reports = Vec::new();
        let mut different = false;
        for rank in 0..self.sessions.len() {
            let expected = self.outputs(rank, [q_state(0, 0), q_state(0, 1)])?;
            let actual = self.outputs(rank, ids)?;
            for (k, name) in ["residual_out", "normed"].into_iter().enumerate() {
                let difference = output_differences(&expected[k], &actual[k]);
                let reference_difference = output_differences(&first[k], &expected[k]);
                different |= expected[k] != actual[k] || expected[k] != first[k];
                reports.push(json!({"rank": rank, "output": name,
                    "candidate_vs_reference": difference,
                    "reference_vs_rank0": reference_difference}));
            }
        }
        let report = json!(reports);
        if !write_errors.is_empty() {
            return Err(format!(
                "{what}: output write checks failed: {}; bit differences: {report}",
                write_errors.join("; ")
            ));
        }
        if different {
            return Err(format!(
                "{what}: outputs differ from the current entry: {report}"
            ));
        }
        Ok(())
    }
}

/// Row counts the Qwen decode plan takes: the LL rows and even rows above.
fn decode_rows() -> Vec<u32> {
    (1..=MAX_ROWS as u32)
        .filter(|&rows| hc_decode::plan(rows).is_ok())
        .collect()
}

/// `candidate` against `reference`, the standalone one-shot
/// (`ONE_SHOT_SYMBOL`) or another decode entry.
fn qwen_bitwise(sessions: &[Session], reference: &str, candidate: &str) -> LabResult<Value> {
    let ranks = sessions.len() as u32;
    let reference_cubin = if reference == ONE_SHOT_SYMBOL {
        ONE_SHOT_TRANSPORT.cubin
    } else {
        hc_decode::TRANSPORT.cubin
    };
    let symbols = [
        (reference_cubin, reference.to_owned()),
        (hc_decode::TRANSPORT.cubin, candidate.to_owned()),
    ];
    let mut lab = QwenLab::new(sessions, 2, &symbols)?;
    let mut cases = 0;
    let mut call = INJECT_ROUNDING_CALL - 1;
    let mut wrong_order_rejected = false;
    let mut stale_input_rejected = false;
    for rows in decode_rows() {
        let replay = RankReplay::new(sessions, |rank, _| {
            lab.launch(Some(rank), candidate, 1, rows)
        })?;
        let mut lives = vec![rows, rows - 1, 1, 0];
        lives.dedup();
        for live in lives {
            lab.set_live(live)?;
            for graph in [false, true] {
                call += 1;
                lab.set_input(Q_INPUT, call, false)?;
                for slot in [0, 1] {
                    lab.clear_outputs(slot)?;
                }
                // Candidate first, on its own state. A graph never follows
                // an eager candidate with the same input generation.
                if graph {
                    replay.launch(sessions)?;
                } else {
                    lab.launch(None, candidate, 1, rows)?;
                }
                synchronize(sessions)?;
                lab.launch_reference(reference, rows, [Q_INPUT, q_state(0, 0), q_state(0, 1)])?;
                synchronize(sessions)?;
                lab.compare(
                    &format!("{candidate} rows {rows} live {live} graph {graph} call {call}"),
                    live as usize,
                )?;
                cases += 1;
                if cases == 1 {
                    // A real oracle launch with deliberately reordered
                    // rank inputs must fail the full output comparison.
                    if ranks >= 4 {
                        lab.set_input(Q_INPUT, call, true)?;
                        lab.clear_outputs(0)?;
                        lab.launch_reference(
                            reference,
                            rows,
                            [Q_INPUT, q_state(0, 0), q_state(0, 1)],
                        )?;
                        synchronize(sessions)?;
                        match lab.compare("wrong rank order", live as usize) {
                            Err(e) if e.contains("outputs differ from the current entry:") => {}
                            Err(e) => return Err(e),
                            Ok(()) => {
                                return Err("reversed rank inputs escaped bitwise detection".into());
                            }
                        }
                        wrong_order_rejected = true;
                    }
                    lab.set_input(Q_INPUT, call - 1, false)?;
                    lab.clear_outputs(0)?;
                    lab.launch_reference(reference, rows, [Q_INPUT, q_state(0, 0), q_state(0, 1)])?;
                    synchronize(sessions)?;
                    match lab.compare("previous call input", live as usize) {
                        Err(e) if e.contains("outputs differ from the current entry:") => {}
                        Err(e) => return Err(e),
                        Ok(()) => {
                            return Err("previous-call inputs escaped bitwise detection".into());
                        }
                    }
                    stale_input_rejected = true;
                }
            }
        }
    }
    // Every call gets distinct input and output addresses. Keep one state
    // across mixed protocols and change every input again before replay.
    let sequence: [u32; 7] = [32, 1, 16, 2, 4, 30, 8];
    let ios: Vec<_> = (0..sequence.len() as u64)
        .map(|k| [3000 + 3 * k, 3001 + 3 * k, 3002 + 3 * k])
        .collect();
    for (rank, session) in sessions.iter().enumerate() {
        for io in &ios {
            for (id, words) in [
                (io[0], MAX_ROWS * HC_WIDTH),
                (io[1], MAX_ROWS * HC_WIDE),
                (io[2], MAX_ROWS * HC_WIDE),
            ] {
                lab.buffers[rank].insert(id, session.allocate(words * 2)?);
            }
        }
    }
    lab.set_live(1)?;
    let replay = RankReplay::new(sessions, |rank, _| {
        sequence
            .iter()
            .zip(&ios)
            .try_for_each(|(&rows, &io)| lab.launch_with_io(Some(rank), candidate, 1, rows, io))
    })?;
    let outputs: Vec<_> = ios.iter().flat_map(|io| [io[1], io[2]]).collect();
    for round in 0..3 {
        for io in &ios {
            call += 1;
            lab.set_input(io[0], call, false)?;
        }
        poison_ids(sessions, &lab.buffers, &outputs)?;
        replay.launch(sessions)?;
        synchronize(sessions)?;
        for (&rows, io) in sequence.iter().zip(&ios) {
            lab.clear_outputs(0)?;
            lab.launch_reference(reference, rows, [io[0], q_state(0, 0), q_state(0, 1)])?;
            synchronize(sessions)?;
            lab.compare_with(
                &format!("{candidate} sequence replay {round} rows {rows}"),
                1,
                [io[1], io[2]],
            )?;
            cases += 1;
        }
    }
    // Exercise both sides of the inject threshold with large finite y.
    // Tail products stay observable even where an F32 sigmoid is zero.
    for rows in [1, 2, 4] {
        lab.set_live(rows)?;
        let tail_replay = RankReplay::new(sessions, |rank, _| {
            lab.launch(Some(rank), candidate, 1, rows)
        })?;
        for shift in 0..8 {
            lab.set_inject_logits(shift)?;
            for graph in [false, true] {
                call += 1;
                lab.set_inject_inputs(call)?;
                for slot in [0, 1] {
                    lab.clear_outputs(slot)?;
                }
                if graph {
                    tail_replay.launch(sessions)?;
                } else {
                    lab.launch(None, candidate, 1, rows)?;
                }
                synchronize(sessions)?;
                lab.launch_reference(reference, rows, [Q_INPUT, q_state(0, 0), q_state(0, 1)])?;
                synchronize(sessions)?;
                lab.compare(
                    &format!("inject tail rows {rows} shift {shift} graph {graph} call {call}"),
                    rows as usize,
                )?;
                cases += 1;
            }
        }
    }
    // A finite fill cannot collide with a valid NaN output. Keep bitwise
    // equality of NaN payloads and the same live/dead write checks.
    for rows in [1, 2, 4] {
        lab.set_live(rows)?;
        let replay = RankReplay::new(sessions, |rank, _| {
            lab.launch(Some(rank), candidate, 1, rows)
        })?;
        for shift in 0..3 {
            lab.set_nonfinite_logits(shift)?;
            for graph in [false, true] {
                call += 1;
                lab.set_input(Q_INPUT, call, false)?;
                for slot in [0, 1] {
                    lab.fill_outputs(slot, 0x7f)?;
                }
                if graph {
                    replay.launch(sessions)?;
                } else {
                    lab.launch(None, candidate, 1, rows)?;
                }
                synchronize(sessions)?;
                lab.launch_reference(reference, rows, [Q_INPUT, q_state(0, 0), q_state(0, 1)])?;
                synchronize(sessions)?;
                lab.compare_with_fill(
                    &format!(
                        "nonfinite logits rows {rows} shift {shift} graph {graph} call {call}"
                    ),
                    rows as usize,
                    [q_state(1, 0), q_state(1, 1)],
                    0x7f,
                )?;
                cases += 1;
            }
        }
    }
    Ok(json!({
        "ranks": ranks,
        "rows": decode_rows(),
        "pair": [reference, candidate],
        "cases": cases,
        "first_input_generation": INJECT_ROUNDING_CALL,
        "input_generations": call - INJECT_ROUNDING_CALL + 1,
        "wrong_order_rejected": (ranks >= 4).then_some(wrong_order_rejected),
        "stale_input_rejected": stale_input_rejected,
    }))
}

// --- Tests ------------------------------------------------------------------

fn receipt(sessions: &[Session], name: &str, mut data: Value) -> LabResult<()> {
    data["devices"] = json!(
        sessions
            .iter()
            .map(|s| s.metadata.clone())
            .collect::<Vec<_>>()
    );
    if std::env::var_os("CHOPINFER_COMM_LAT_OUT").is_some() {
        write_json(&format!("{name}-r{}.json", sessions.len()), &data)?;
    }
    eprintln!("{name}: {data}");
    Ok(())
}

/// CPU: the test inputs make the summation order visible: in the MiMo BF16
/// residual outputs of every row at two, four and eight ranks (the lab's
/// inputs and a skew call's), and in BF16-rounded collective sums at four and eight.
#[test]
fn inputs_expose_the_summation_order() {
    let residual = pattern(41, MAX_ROWS * HIDDEN, 1.0);
    for ranks in [2, 4, 8] {
        for seed in [40, 3000] {
            order_observable(&mimo_inputs(ranks, seed), false, &residual)
                .unwrap_or_else(|e| panic!("plain, {ranks} ranks, seed {seed}: {e}"));
        }
        for seed in [60, 3010] {
            order_observable(&mimo_routed(ranks, seed), true, &residual)
                .unwrap_or_else(|e| panic!("combine, {ranks} ranks, seed {seed}: {e}"));
        }
    }
    for ranks in [4, 8] {
        let inputs = comm_inputs::inputs(ranks, MAX_ROWS, HC_WIDTH, 0);
        comm_inputs::require_order_sensitive(&inputs, HC_WIDTH).expect("order-sensitive inputs");
    }
}

#[test]
fn decode_lab_launch_setup_is_valid_without_cuda() {
    assert_eq!(MAX_ROWS as u32, hc_decode::MAX_ROWS);
    assert_eq!(HC_WIDTH as u32, hc_decode::TRANSPORT.shape.epilogue.width);
    assert_eq!(
        HC_STREAMS as u32,
        hc_decode::TRANSPORT.shape.epilogue.streams
    );
    for ranks in [2, 4, 8] {
        let state = hc_decode::buffers(ranks as u32).unwrap();
        let entries = [
            hc_decode::symbol_v2(ranks as u32),
            hc_decode::symbol_v3(ranks as u32),
        ]
        .map(Result::unwrap);
        let cubin = hc_decode::TRANSPORT.cubin;
        for rows in decode_rows() {
            // All finite/nonfinite, replay and timing cases share this
            // builder. Audit every supported bucket, not only timed rows.
            for (slot, symbol) in entries.iter().enumerate() {
                let slot = slot as u64;
                for io in [
                    [Q_INPUT, q_state(slot, 0), q_state(slot, 1)],
                    [3000, 3001, 3002],
                ] {
                    let input_bytes = (MAX_ROWS * HC_WIDTH * 2) as u64;
                    let output_bytes = (MAX_ROWS * HC_WIDE * 2) as u64;
                    let mut buffers = BTreeMap::from([
                        (io[0], input_bytes),
                        (io[1], output_bytes),
                        (io[2], output_bytes),
                        (Q_RESIDUAL, output_bytes),
                        (Q_LOGITS, (MAX_ROWS * HC_STREAMS * 2) as u64),
                        (Q_WEIGHT, (HC_WIDE * 2) as u64),
                        (Q_LIVE, 4),
                        (q_state(slot, 4), state.epoch_bytes),
                    ]);
                    for peer in 0..ranks {
                        buffers.insert(peer_inbox(slot, peer), state.inbox_bytes);
                        buffers.insert(peer_flags(slot, peer), state.flags_bytes);
                    }
                    for rank in 0..ranks {
                        let (plan, args) = decode_lab_call(ranks, rank, slot, rows, io).unwrap();
                        comm_setup::check(cubin, symbol, plan, &args, &buffers).unwrap();
                    }
                }
            }
        }
        // The standalone one-shot reference on its own state.
        let one_shot = one_shot_buffers(one_shot_query(ranks)).unwrap();
        for io in [[Q_INPUT, q_state(0, 0), q_state(0, 1)], [3000, 3001, 3002]] {
            let output_bytes = (MAX_ROWS * HC_WIDE * 2) as u64;
            let mut buffers = BTreeMap::from([
                (io[0], (MAX_ROWS * HC_WIDTH * 2) as u64),
                (io[1], output_bytes),
                (io[2], output_bytes),
                (Q_RESIDUAL, output_bytes),
                (Q_LOGITS, (MAX_ROWS * HC_STREAMS * 2) as u64),
                (Q_WEIGHT, (HC_WIDE * 2) as u64),
                (Q_LIVE, 4),
                (Q_ONE_SHOT[2], one_shot.epoch_bytes),
            ]);
            for peer in 0..ranks {
                buffers.insert(
                    peer_inbox(ONE_SHOT_SLOT, peer),
                    standalone_inbox_bytes(ranks as u32, MAX_ROWS as u32),
                );
                buffers.insert(peer_flags(ONE_SHOT_SLOT, peer), one_shot.flags_bytes);
            }
            for rank in 0..ranks {
                let (plan, args) = one_shot_lab_call(ranks, rank, io).unwrap();
                comm_setup::check(
                    ONE_SHOT_TRANSPORT.cubin,
                    ONE_SHOT_SYMBOL,
                    plan,
                    &args,
                    &buffers,
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn mismatch_reports_count_every_element_even_when_examples_are_bounded() {
    let expected = vec![0x1234; 32];
    let mut actual = expected.clone();
    for value in &mut actual[10..] {
        *value = 0x5678;
    }
    let report = output_differences(&expected, &actual);
    assert_eq!(report["differing_elements"], 22);
    assert_eq!(report["first_differences"].as_array().unwrap().len(), 8);
    assert_eq!(report["first_differences"][0]["index"], 10);
    assert_eq!(
        output_differences(&expected, &expected)["differing_elements"],
        0
    );
}

#[test]
fn inject_contraction_changes_norm_without_changing_bf16_residual() {
    fn warp_sum(mut values: [f32; 32]) -> f32 {
        for offset in [16, 8, 4, 2, 1] {
            let before = values;
            for lane in 0..32 - offset {
                values[lane] += before[lane + offset];
            }
        }
        values[0]
    }
    // Model the emitted FFMA square accumulation and the two shuffle
    // trees of sum_cta. Only the inject differs between the two models.
    fn norm(values: &[f32], weight: &[u16]) -> Vec<u16> {
        let squares: Vec<_> = (0..256)
            .map(|thread| {
                (0..HC_WIDTH / 256).fold(0.0_f32, |sum, j| {
                    let v = values[thread + j * 256];
                    v.mul_add(v, sum)
                })
            })
            .collect();
        let mut warp_sums = [0.0; 32];
        for (warp, values) in squares.chunks_exact(32).enumerate() {
            warp_sums[warp] = warp_sum(values.try_into().unwrap());
        }
        let mean = warp_sum(warp_sums).mul_add(1.0 / HC_WIDTH as f32, 1e-6);
        let inv = (1.0_f64 / f64::from(mean).sqrt()) as f32;
        values
            .iter()
            .zip(weight)
            .map(|(&v, &w)| f32_to_bf16((v * inv) * (1.0 + bf16_to_f32(w))))
            .collect()
    }
    let inputs = comm_inputs::inputs(2, 1, HC_WIDTH, INJECT_ROUNDING_CALL);
    let y: Vec<_> = (0..HC_WIDTH)
        .map(|c| bf16_to_f32(inputs[0][c]) + bf16_to_f32(inputs[1][c]))
        .collect();
    let mut residual = pattern(1, HC_WIDE, 1.0);
    comm_inputs::clear_marker_residuals(&mut residual, 2);
    let logits = pattern(2, HC_STREAMS, 4.0);
    let weights = pattern(3, HC_WIDE, 0.5);
    let mut expected = [Vec::new(), Vec::new()];
    let mut fused = [Vec::new(), Vec::new()];
    for (stream, &logit) in logits.iter().enumerate() {
        let z = bf16_to_f32(logit) / HC_STREAMS as f32;
        let scale = 2.0 * (1.0 / (1.0 + (-z).exp()));
        let begin = stream * HC_WIDTH;
        let x = &residual[begin..begin + HC_WIDTH];
        let weight = &weights[begin..begin + HC_WIDTH];
        let ordinary: Vec<_> = y
            .iter()
            .zip(x)
            .map(|(&y, &x)| bf16_to_f32(x) + scale * y)
            .collect();
        let contracted: Vec<_> = y
            .iter()
            .zip(x)
            .map(|(&y, &x)| scale.mul_add(y, bf16_to_f32(x)))
            .collect();
        if stream == 0 {
            assert_eq!(ordinary[2466].to_bits(), 0xba1c5000);
            assert_eq!(contracted[2466].to_bits(), 0xba1c4e00);
        }
        expected[0].extend(ordinary.iter().copied().map(f32_to_bf16));
        fused[0].extend(contracted.iter().copied().map(f32_to_bf16));
        expected[1].extend(norm(&ordinary, weight));
        fused[1].extend(norm(&contracted, weight));
    }
    assert_eq!(
        output_differences(&expected[0], &fused[0])["differing_elements"],
        0
    );
    let report = output_differences(&expected[1], &fused[1]);
    assert_eq!(report["differing_elements"], 1);
    assert_eq!(
        report["first_differences"][0],
        json!({"index": 2466, "expected": "0xb9f9", "actual": "0xb9f8"})
    );
}

#[test]
fn inject_threshold_rejects_a_strict_comparison() {
    let z = bf16_to_f32(f32_to_bf16(-320.0)) / HC_STREAMS as f32;
    assert_eq!(z, -80.0);
    let scale = 2.0 * (1.0 / (1.0 + (-z).exp()));
    for ranks in [2, 4, 8] {
        for call in 0..8 {
            let mut inputs = comm_inputs::inputs(ranks, MAX_ROWS, HC_WIDTH, call);
            let mut residual = vec![0; MAX_ROWS * HC_WIDE];
            inject_threshold_witness(&mut inputs, &mut residual, call);
            comm_inputs::require_order_sensitive(&inputs, HC_WIDTH).unwrap();
            for row in 0..MAX_ROWS {
                for quarter in 0..4 {
                    let column = quarter * (HC_WIDTH / 4) + ranks;
                    let y = inputs.iter().fold(0.0_f32, |sum, input| {
                        sum + bf16_to_f32(input[row * HC_WIDTH + column])
                    });
                    let x = bf16_to_f32(residual[row * HC_WIDE + column]);
                    let tail = (f64::from(y) * f64::from(z).exp()) as f32;
                    // The deliberately wrong strict comparison chooses
                    // the rounded tail product at the exact threshold.
                    let wrong = f32_to_bf16(if z > -80.0 {
                        x + scale * y
                    } else {
                        2.0_f32.mul_add(tail, x)
                    });
                    // Permit nearby expf/reciprocal rounding: the witness
                    // stays visible across two scale ULPs in either direction.
                    for delta in -2..=2 {
                        let near =
                            f32::from_bits(scale.to_bits().checked_add_signed(delta).unwrap());
                        let expected = f32_to_bf16(x + near * y);
                        assert_ne!(expected, wrong, "ranks {ranks}, call {call}, delta {delta}");
                    }
                }
            }
        }
    }
}

#[test]
fn inject_tail_inputs_expose_float_sigmoid_underflow() {
    let y = f32::powi(2.0, 80);
    let z = -144.0_f32;
    let expected = f32_to_bf16(2.0 * (f64::from(y) * f64::from(z).exp()) as f32);
    let wrong = f32_to_bf16(2.0 * y * z.exp());
    assert_ne!(expected, 0);
    assert_eq!(wrong, 0);
    assert_ne!(expected, wrong);
}

/// CPU: the old MiMo inputs (32-binade values only) hide the rank order in
/// the BF16 residual outputs of some rows, which `order_observable` reports.
#[test]
fn spread_inputs_hide_the_rank_order() {
    let residual = pattern(41, MAX_ROWS * HIDDEN, 1.0);
    let inputs: Vec<_> = (0..4)
        .map(|rank| partials(rank + 40, MAX_ROWS * HIDDEN, 0.05))
        .collect();
    assert!(order_observable(&inputs, false, &residual).is_err());
}

#[test]
fn skew_compares_decode_entries_with_the_current_entries() {
    for (candidate, verify) in [
        (comm::DECODE_SYMBOL_V2, comm::VERIFY_SYMBOL),
        (comm::DECODE_COMBINE_SYMBOL, comm::VERIFY_COMBINE_SYMBOL),
    ] {
        let pair = DecodePair {
            reference: "baseline",
            candidate,
            combine: false,
        };
        assert_eq!(pair.skew_reference(candidate), pair.reference);
        assert_eq!(pair.skew_reference(pair.reference), pair.reference);
        assert_eq!(pair.skew_reference(verify), verify);
        assert_eq!(
            pair.skew_reference(comm::PREFILL_ROUND_ONCE_SYMBOL),
            comm::VERIFY_SYMBOL
        );
    }
}

/// CPU negative control: flattening the row's warp sums into one tree
/// loses contributions that the original per-256-thread trees preserve.
#[test]
fn async_sum_keeps_the_cta_tree() {
    let xor_sum = |mut lanes: [f32; 32]| {
        for offset in [16, 8, 4, 2, 1] {
            let before = lanes;
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value += before[lane ^ offset];
            }
        }
        lanes[0]
    };
    for groups in [2, 4] {
        // These warp sums can come from BF16 row values 4096, 1 and 1,
        // each in one vector of its warp, with every other value zero.
        let mut warps = [0.0; 32];
        warps[0] = 16_777_216.0;
        warps[8] = 1.0;
        warps[9] = 1.0;
        let grouped = (0..groups).fold(0.0_f32, |total, group| {
            let mut lanes = [0.0; 32];
            lanes[..8].copy_from_slice(&warps[group * 8..(group + 1) * 8]);
            total + xor_sum(lanes)
        });
        assert_eq!(grouped.to_bits(), 16_777_218.0_f32.to_bits());
        let wrong = xor_sum(warps);
        assert_eq!(wrong.to_bits(), 16_777_216.0_f32.to_bits());
        assert_ne!(
            grouped.to_bits(),
            wrong.to_bits(),
            "wrong tree went undetected"
        );
    }
    let row = partials(998, HIDDEN, 0.05);
    let mut warps = [0.0_f32; 32];
    for (warp, values) in row.chunks_exact(32 * 8).enumerate() {
        let mut lanes = [0.0_f32; 32];
        for (lane, vector) in values.chunks_exact(8).enumerate() {
            for pair in vector.chunks_exact(2) {
                let (a, b) = (bf16_to_f32(pair[0]), bf16_to_f32(pair[1]));
                lanes[lane] += a * a + b * b;
            }
        }
        warps[warp] = xor_sum(lanes);
    }
    let grouped = (0..2).fold(0.0_f32, |total, group| {
        let mut lanes = [0.0; 32];
        lanes[..8].copy_from_slice(&warps[group * 8..(group + 1) * 8]);
        total + xor_sum(lanes)
    });
    let normalize = |sum: f32| {
        let inv = 1.0 / (sum / HIDDEN as f32 + 1e-6).sqrt();
        row.iter()
            .map(|&value| f32_to_bf16(bf16_to_f32(value) * inv))
            .collect::<Vec<_>>()
    };
    let expected = normalize(grouped);
    let wrong = normalize(xor_sum(warps));
    assert!(
        expected.iter().zip(wrong).any(|(a, b)| *a != b),
        "wrong tree is hidden by BF16 rounding"
    );
}

#[test]
fn mimo_decode_bitwise_g1() {
    gpu_test_multi(256 << 20, |sessions| {
        let data = mimo_bitwise(sessions)?;
        receipt(sessions, "mimo-decode-bitwise", data)
    })
    .expect("MiMo decode all-reduce bitwise against the current entries");
}

#[test]
fn mimo_verify_rows_bitwise_g1() {
    gpu_test_multi(512 << 20, |sessions| {
        let data = mimo_verify_rows(sessions)?;
        receipt(sessions, "mimo-verify-rows-bitwise", data)
    })
    .expect("MiMo exact verify rows all-reduce bitwise against one row alone");
}

#[test]
fn mimo_decode_skew_g1() {
    gpu_test_multi(1 << 30, |sessions| {
        let data = json!({
            "decode": decode_skew(sessions, &MIMO_SEQUENCE, Selection::Decode)?,
            "selected": decode_skew(sessions, &SELECTED_SEQUENCE, Selection::Bundle)?,
            "mixed": decode_skew(sessions, &MIMO_SEQUENCE, Selection::Mixed)?,
        });
        receipt(sessions, "mimo-decode-skew", data)
    })
    .expect("MiMo decode all-reduce under rank skew");
}

#[test]
fn qwen_decode_bitwise_g1() {
    gpu_test_multi(256 << 20, |sessions| {
        if !qwen38::RANKS.contains(&(sessions.len() as u32)) {
            return Err(format!(
                "Qwen decode comm takes 2, 4 or 8 ranks, not {}",
                sessions.len()
            ));
        }
        let ranks = sessions.len() as u32;
        let entries = [hc_decode::symbol_v2(ranks), hc_decode::symbol_v3(ranks)]
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        let data = json!({"pairs": [
            qwen_bitwise(sessions, ONE_SHOT_SYMBOL, entries[0])?,
            qwen_bitwise(sessions, entries[0], entries[1])?,
        ]});
        receipt(sessions, "qwen-decode-bitwise", data)
    })
    .expect("Qwen decode all-reduce bitwise against the current entries");
}

/// Per-call median of a graph of `CALLS_PER_GRAPH` calls on every rank.
fn time_calls(sessions: &[Session], graphs: &[GraphExec]) -> LabResult<Value> {
    let timing = kernel_lab::time_multi_graph(sessions, graphs, CALLS_PER_GRAPH, 3, 9)?;
    let per_call = |ms: f32| f64::from(ms) * 1000.0 / CALLS_PER_GRAPH as f64;
    Ok(json!({
        "median_us": per_call(timing.critical_path.median_ms),
        "min_us": per_call(timing.critical_path.min_ms),
        "per_rank_median_us": timing.per_rank.iter().map(|t| per_call(t.median_ms)).collect::<Vec<_>>(),
        "gpu_rounds": timing.gpu_rounds,
    }))
}

/// A test build's name of an entry, `chop_comm_{tag}_*` (build.rs: `skew`
/// lags one rank, `flat` sums the LL norm in one flat tree).
fn variant_symbol(tag: &str, symbol: &str) -> String {
    symbol.replacen("chop_comm_", &format!("chop_comm_{tag}_"), 1)
}

/// Per-call medians at every row count: the current entry, the decode entry,
/// the verify entry where it serves, and the current entry again last, so a
/// drift between the first and last runs shows.
fn mimo_latency(sessions: &[Session]) -> LabResult<Vec<Value>> {
    let ranks = sessions.len() as u32;
    let mut symbols: Vec<(&str, String)> = Vec::new();
    for p in MIMO_PAIRS {
        symbols.push(("comm_sm120", p.reference.to_owned()));
        symbols.push((entry_cubin(p.candidate), p.candidate.to_owned()));
    }
    for symbol in [comm::VERIFY_SYMBOL, comm::VERIFY_COMBINE_SYMBOL] {
        symbols.push((VERIFY_CUBIN, symbol.to_owned()));
    }
    let verify_rows = comm::verify_entry_rows(ranks).map_err(|e| e.to_string())?;
    let lab = DecodeLab::new(sessions, 3, &symbols)?;
    let mut results = Vec::new();
    for rows in bench_rows(&(1..=MAX_ROWS as u32).collect::<Vec<_>>()) {
        lab.set_live(rows)?;
        let protocol = comm::push_protocol(ranks, rows).map_err(|e| e.to_string())?;
        for pair in MIMO_PAIRS {
            // (state, symbol, label)
            let mut entries = vec![
                (0, pair.reference.to_owned(), "current"),
                (1, pair.candidate.to_owned(), "new"),
            ];
            // The exact verify rows entries, beside the decode entries.
            if rows <= verify_rows {
                let verify = if pair.combine {
                    comm::VERIFY_COMBINE_SYMBOL
                } else {
                    comm::VERIFY_SYMBOL
                };
                entries.push((2, verify.to_owned(), "verify"));
                entries.push((1, pair.candidate.to_owned(), "new-again"));
            }
            entries.push((0, pair.reference.to_owned(), "current-again"));
            for (slot, symbol, label) in entries {
                let graphs = sessions
                    .iter()
                    .enumerate()
                    .map(|(rank, session)| {
                        session.capture(|_| {
                            (0..CALLS_PER_GRAPH).try_for_each(|_| {
                                lab.launch(Some(rank), &symbol, slot, rows, pair.combine)
                            })
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let mut timing = time_calls(sessions, &graphs)?;
                timing["model"] = json!("mimo");
                timing["symbol"] = json!(symbol);
                timing["arm"] = json!(label);
                timing["combine"] = json!(pair.combine);
                timing["rows"] = json!(rows);
                timing["ranks"] = json!(ranks);
                timing["protocol"] = json!(format!("{protocol:?}"));
                eprintln!("{timing}");
                results.push(timing);
            }
        }
    }
    Ok(results)
}

/// Multi-rank A/B windows: alternate A/B and B/A, with independent cache
/// eviction, rank gates and clock records from time_multi_graph in each arm.
/// Each window has three samples; pair its slowest-rank median with the
/// other arm's median. Single-rank time_ab cannot gate a collective.
fn time_calls_ab(sessions: &[Session], graphs: &[Vec<GraphExec>; 2]) -> LabResult<Value> {
    let mut samples = [Vec::new(), Vec::new()];
    let mut windows = Vec::new();
    let mut ratios = Vec::new();
    for round in 0..9 {
        let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        let mut pair = [0.0; 2];
        for arm in order {
            let timing =
                kernel_lab::time_multi_graph(sessions, &graphs[arm], CALLS_PER_GRAPH, 1, 3)?;
            let us = f64::from(timing.critical_path.median_ms) * 1000.0 / CALLS_PER_GRAPH as f64;
            pair[arm] = us;
            samples[arm].push(us);
            windows.push(json!({"round": round, "arm": arm, "median_us": us,
                "gpu_rounds": timing.gpu_rounds}));
        }
        ratios.push(pair[1] / pair[0]);
    }
    for arm in &mut samples {
        arm.sort_by(f64::total_cmp);
    }
    ratios.sort_by(f64::total_cmp);
    Ok(json!({"median_us": [samples[0][4], samples[1][4]],
        "paired_ratio_b_over_a": ratios[4], "windows": windows,
        "samples_per_window": 3}))
}

fn qwen_latency(sessions: &[Session]) -> LabResult<Vec<Value>> {
    let ranks = sessions.len() as u32;
    let entries = [
        hc_decode::symbol_v2(ranks).map_err(|e| e.to_string())?,
        hc_decode::symbol_v3(ranks).map_err(|e| e.to_string())?,
    ];
    let symbols: Vec<_> = entries
        .iter()
        .map(|s| ("comm_qwen38_decode", s.to_string()))
        .collect();
    let lab = QwenLab::new(sessions, 2, &symbols)?;
    let mut results = Vec::new();
    let rows_list: Vec<u32> = bench_rows(&decode_rows())
        .into_iter()
        .filter(|&rows| hc_decode::plan(rows).is_ok())
        .collect();
    for rows in rows_list {
        lab.set_live(rows)?;
        let capture = |slot: usize| -> LabResult<Vec<GraphExec>> {
            sessions
                .iter()
                .enumerate()
                .map(|(rank, session)| {
                    session.capture(|_| {
                        (0..CALLS_PER_GRAPH).try_for_each(|_| {
                            lab.launch(Some(rank), entries[slot], slot as u64, rows)
                        })
                    })
                })
                .collect()
        };
        let graphs = [capture(0)?, capture(1)?];
        let mut timing = time_calls_ab(sessions, &graphs)?;
        timing["model"] = json!("qwen");
        timing["pair"] = json!(entries);
        timing["rows"] = json!(rows);
        timing["ranks"] = json!(ranks);
        eprintln!("{timing}");
        results.push(timing);
    }
    Ok(results)
}

#[test]
fn decode_latency_g4() {
    if !bench_enabled() {
        kernel_lab::skip("decode all-reduce latency: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs")
            .unwrap();
        return;
    }
    gpu_test_multi(512 << 20, |sessions| {
        let ranks = sessions.len();
        let mut results = mimo_latency(sessions)?;
        if std::env::var("CHOPINFER_COMM_LAT_CONTROL").as_deref() != Ok("0") {
            results.extend(mimo_latency(&sessions[..1])?);
        }
        if qwen38::RANKS.contains(&(ranks as u32)) {
            results.extend(qwen_latency(sessions)?);
        }
        receipt(
            sessions,
            "decode-latency",
            json!({"calls_per_graph": CALLS_PER_GRAPH, "results": results}),
        )
    })
    .expect("decode all-reduce latency");
}
