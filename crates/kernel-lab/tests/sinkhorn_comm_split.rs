//! The split Sinkhorn mHC comm epilogue (`kernels_sm120::comm::sinkhorn_split`)
//! at the served 4 x 4096 mixer, after the unchanged F32 all-reduce, against
//! the unfused launch sequence it replaces, bit for bit, on the GPU. Run the
//! rows transport with `CHOPINFER_KERNEL_GPU=<ordinal> cargo
//! test -p kernel-lab --release --test sinkhorn_comm_split --
//! --test-threads=1`, and the one-shot transport with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3` (TP4); without GPUs only the CPU contract
//! test runs. The structure follows `tests/sinkhorn_comm_epilogue.rs`, the
//! one-CTA epilogue's qualification.
//!
//! The unfused sequence is the served one: the F32 one-shot all-reduce with
//! 4096-element tiles (`comm_sm120`), the round of each sum to BF16 (on the
//! host: one cvt.rn per element), the Sinkhorn expand and the one-CTA
//! mix-collapse (`sinkhorn_hc_hc4x4096_sm120`) and the HF-ordered input
//! RMSNorm (the reference entry of `sinkhorn_epilogue_hc4x4096_sm120`).
//!
//! - Rows transport, one GPU, at decode buckets (1, 2, 4, 16, 32), verify
//!   rows (6 x 32) and prefill chunks (2048, 8192): from the same F32 sums,
//!   every output (y, X', post, comb, collapsed, normed) equals the unfused
//!   launches' bit for bit, the tickets are left zero, and the outputs meet
//!   the kernel-core oracles (expand and norm exactly, post and collapsed
//!   within the 128-K accumulation gate).
//! - After the one-shot, four GPUs, at every row count it serves (1, 2, 7,
//!   16): the split path (the F32 one-shot into its own sums, then the
//!   split epilogue, PDL between them) and the unfused path alternate on one
//!   one-shot state; every rank's sums equal the host's rank-order sums, and
//!   every split output equals the unfused sequence's and rank 0's, eagerly
//!   and through each rank's captured graph (the streams' producer, the
//!   one-shot and the epilogue), at every live count.
//! - Both cases replayed on poisoned buffers (eager, relaunch, graph,
//!   PDL) at every live count `poison::live_counts` lists, after producers
//!   whose outputs are poisoned and checked too: the streams' producer (an
//!   expand); the one-shot's partials' producer (the lab's PDL prefetch
//!   probe), which holds each rank's partials at NaN until after a tail
//!   that follows its trigger of its dependents, then copies the call's own
//!   partials in; and the one-shot, which produces the sums (over one rank
//!   on persistent state for the rows transport). The one-shot reads its
//!   partials only after its dependency wait, so the sums, and the epilogue
//!   after them, see the call's partials only if every wait holds. Rows
//!   past the live count keep their poison, post and comb (read, then
//!   overwritten) keep their image there, the chain sums of the live rows
//!   are written and the tickets stay zero. The eager sums are compared
//!   with the host's, and each rows-transport live count's eager outputs
//!   with the unfused sequence, independently of the replays'
//!   self-comparison. Each mutation must change the outputs it corrupts
//!   against known good bits: any producer left out and, for the rows
//!   transport, the post and comb stores undone (against the unfused
//!   sequence). Launch, synchronization and readback errors propagate
//!   rather than count as caught.
//! - The epilogue's dependency wait, on one GPU: its sums come from a
//!   watch producer (`probe_pdl_change_watch_sm120`, E23's
//!   `probe_pdl_watch_sm120` for a dependent that releases no flag), which
//!   holds them at NaN after triggering its dependents until the epilogue
//!   has written the first word of its round, which it does only from the
//!   sums, or a hold passes, and records which came first. The served
//!   epilogue behind it (50 us hold) must never get ahead and must
//!   reproduce the eager bits. The late-wait mutant
//!   (`sinkhorn_epilogue_split_late_wait_hc4x4096_sm120`, its wait after
//!   its reads of the sums and streams; 200 ms hold) is caught when it got
//!   ahead and its round differs from the eager bits with the producers'
//!   outputs unchanged, fails the test if its round differs otherwise or
//!   the producers' outputs changed, and is reported inconclusive (receipt
//!   and a stderr line) when it never got ahead and nothing changed: PDL
//!   then did not overlap the two launches. The epilogue's wait is on its
//!   own rank's sums, so the four-rank case does not repeat it.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, CommQuery, DType, LaunchPlan};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::comm::sinkhorn_epilogue::{self, EpilogueBuffers};
use kernels_sm120::comm::sinkhorn_split::{self, SplitBuffers};
use kernels_sm120::comm::{self, OneShotCall};
use kernels_sm120::sinkhorn_hc::{self, Op, SERVED};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const RANKS: usize = 4;
const EPS: f32 = 1e-5;
const SLICES: usize = 16;
/// Decode buckets, verify rows (6 x 32) and prefill chunks.
const ROWS_BUCKETS: [usize; 8] = [1, 2, 4, 16, 32, 6 * 32, 2048, 8192];
/// Every row count the F32 one-shot serves at TP4 that a bucket takes.
const ONE_SHOT_BUCKETS: [usize; 4] = [1, 2, 7, 16];

const COMM_CUBIN: &str = "comm_sm120";
const SPLIT_ROWS_CUBIN: &str = "sinkhorn_epilogue_split_hc4x4096_sm120";
const NORM_CUBIN: &str = "sinkhorn_epilogue_hc4x4096_sm120";
const SINKHORN_CUBIN: &str = "sinkhorn_hc_hc4x4096_sm120";
/// The split epilogue with its dependency wait after its reads of the sums
/// and streams, a mutant.
const LATE_WAIT_CUBIN: &str = "sinkhorn_epilogue_split_late_wait_hc4x4096_sm120";
const PROBE_CUBIN: &str = "probe_pdl_prefetch_sm120";
/// The partials' producer's tail between its trigger and its writes.
const PRODUCER_TAIL_NS: u64 = 50_000;
const WATCH_CUBIN: &str = "probe_pdl_change_watch_sm120";
/// The sums' watch producer's hold behind the served epilogue.
const SERVED_HOLD_NS: u64 = 50_000;
/// Its hold behind the late-wait mutant.
const MUTANT_HOLD_NS: u64 = 200_000_000;

// Buffer handles of one rank.
const X0: u64 = 1;
const Y0: u64 = 2;
const POST0: u64 = 3;
const COMB0: u64 = 4;
const X: u64 = 5;
const SUMMED: u64 = 6;
const Y: u64 = 7;
const POST_U: u64 = 8;
const COMB_U: u64 = 9;
const X1_U: u64 = 10;
const COLLAPSED_U: u64 = 11;
const NORMED_U: u64 = 12;
const POST_F: u64 = 13;
const COMB_F: u64 = 14;
const REDUCED_F: u64 = 15;
const X1_F: u64 = 16;
const COLLAPSED_F: u64 = 17;
const NORMED_F: u64 = 18;
const FN: u64 = 19;
const BASE: u64 = 20;
const SCALE: u64 = 21;
const WEIGHT: u64 = 22;
const LIVE: u64 = 23;
const PARTIAL: u64 = 24;
const SUMMED_F: u64 = 25;
const EPOCH: u64 = 26;
const CHAIN_SUMS: u64 = 27;
const TICKETS: u64 = 28;
/// The call's partials, which the partials' producer copies into `PARTIAL`.
const FRESH: u64 = 29;
/// Every rank's one-shot inbox and flags: `INBOX + rank`, `FLAGS + rank`.
const INBOX: u64 = 100;
const FLAGS: u64 = 200;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn split_entry() -> sinkhorn_split::Entry {
    sinkhorn_split::catalog()
}

/// The late-wait mutant's symbol.
fn late_wait_symbol() -> String {
    split_entry().symbol.replacen(
        "chop_sinkhorn_epilogue_",
        sinkhorn_split::LATE_WAIT_PREFIX,
        1,
    )
}

fn sinkhorn_entry(op: Op) -> sinkhorn_hc::Entry {
    sinkhorn_hc::catalog()
        .into_iter()
        .find(|entry| entry.op == op)
        .expect("a Sinkhorn catalog entry")
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// BF16 bits of random sign and mantissa over the binades
    /// `low..=high`, with zeros, negative zeros and a subnormal among them.
    fn bf16(&mut self, len: usize, low: i32, high: i32) -> Vec<u16> {
        (0..len)
            .map(|i| {
                let state = self.next();
                match i % 509 {
                    0 => 0x0000,
                    1 => 0x8000,
                    2 => 0x0001,
                    _ => {
                        let span = (high - low + 1) as u64;
                        let exponent = (low + ((state >> 8) % span) as i32 + 127) as u16;
                        let sign = ((state >> 40) & 1) as u16;
                        (sign << 15) | (exponent << 7) | ((state >> 24) & 0x7f) as u16
                    }
                }
            })
            .collect()
    }

    /// Non-negative BF16 bits in [0, scale).
    fn unit(&mut self, len: usize, scale: f32) -> Vec<u16> {
        (0..len)
            .map(|_| f32_to_bf16(((self.next() >> 40) % 4096) as f32 / 4096.0 * scale))
            .collect()
    }

    /// F32 rank partials of order one with full mantissas, zeros and
    /// negative zeros.
    fn partial(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| match i % 701 {
                0 => 0.0,
                1 => -0.0,
                _ => {
                    let state = self.next();
                    let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
                    let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
                    sign * unit * 2f32.powi(((state >> 8) % 5) as i32 - 3)
                }
            })
            .collect()
    }
}

/// One bucket's operands, replicated over ranks but for the partials.
struct Inputs {
    rows: usize,
    x0: Vec<u16>,
    y0: Vec<u16>,
    post0: Vec<u16>,
    comb0: Vec<u16>,
    /// The streams the sublayer read, the expand of (x0, y0, post0, comb0).
    streams: Vec<u16>,
    /// The sublayer's post and comb.
    post: Vec<u16>,
    comb: Vec<u16>,
    partials: Vec<Vec<f32>>,
    weights: Vec<u16>,
    base: Vec<f32>,
    scale: [f32; 3],
    weight: Vec<u16>,
}

impl Inputs {
    fn new(rows: usize, seed: u64) -> Self {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let x0 = rng.bf16(rows * FLAT, -4, 2);
        let y0 = rng.bf16(rows * H, -6, 1);
        let post0 = rng.unit(rows * N, 2.0);
        let comb0 = rng.unit(rows * CELLS, 1.0);
        let expand = Query::Expand {
            streams: N as u32,
            width: H as u32,
            capacity: rows as u32,
        };
        let streams = oracle::expand(expand, &x0, &y0, &post0, &comb0).unwrap();
        let base = (0..MIXES)
            .map(|_| ((rng.next() >> 40) % 2048) as f32 / 1024.0 - 1.0)
            .collect();
        Self {
            rows,
            post: rng.unit(rows * N, 2.0),
            comb: rng.unit(rows * CELLS, 1.0),
            partials: (0..RANKS).map(|_| rng.partial(rows * H)).collect(),
            // Logits of order one at K = 16,384.
            weights: rng.bf16(MIXES * FLAT, -10, -6),
            base,
            scale: [0.5, 0.75, 2.0],
            weight: rng.unit(H, 2.0),
            x0,
            y0,
            post0,
            comb0,
            streams,
        }
    }

    /// New partials for another call over the same operands.
    fn repartition(&mut self, seed: u64) {
        let mut rng = Rng(seed | 1);
        self.partials = (0..RANKS).map(|_| rng.partial(self.rows * H)).collect();
    }

    /// The rank-order F32 sum of every rank's partial, from zero, as the
    /// one-shot and the phased pair sum it.
    fn summed(&self) -> Vec<f32> {
        (0..self.rows * H)
            .map(|i| {
                self.partials
                    .iter()
                    .fold(0.0_f32, |sum, partial| sum + partial[i])
            })
            .collect()
    }
}

fn halves(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn words(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn exact<T: PartialEq + std::fmt::Debug>(
    label: &str,
    actual: &[T],
    expected: &[T],
) -> LabResult<()> {
    kernel_core::oracle::check_exact(actual, expected).map_err(|at| {
        format!(
            "{label}: element {at} is {:?}, expected {:?}",
            actual.get(at),
            expected.get(at)
        )
    })
}

fn gate(label: &str, actual: &[f64], reference: &OracleOutput) -> LabResult<()> {
    check_numeric(actual, reference, DType::Bf16).map_err(|f| {
        format!(
            "{label}: element {} is {}, reference {}, error {} over bound {}",
            f.index, f.actual, f.reference, f.error, f.bound
        )
    })
}

/// The input norm the unfused path runs, exactly: each square rounded
/// alone, summed from zero in column order, `1 / sqrtf(mean + eps)`, the
/// normalized value rounded to BF16 before the weight multiply.
fn hf_rmsnorm(collapsed: &[u16], weight: &[u16]) -> Vec<u16> {
    collapsed
        .chunks_exact(H)
        .flat_map(|row| {
            let total = row.iter().fold(0.0_f32, |sum, &x| {
                let v = bf16_to_f32(x);
                sum + v * v
            });
            let inv = 1.0 / (total / H as f32 + EPS).sqrt();
            row.iter().zip(weight).map(move |(&x, &w)| {
                let n = bf16_to_f32(f32_to_bf16(bf16_to_f32(x) * inv));
                f32_to_bf16(n * bf16_to_f32(w))
            })
        })
        .collect()
}

/// Runs `run` over row chunks on every core, in row order.
fn by_rows<T: Send>(rows: usize, run: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(per)
            .map(|start| {
                let run = &run;
                scope.spawn(move || run(start..(start + per).min(rows)))
            })
            .collect::<Vec<_>>();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// The kernel-core oracles of one bucket's outputs (`x1`, `collapsed`,
/// `post`, `normed`) from its streams, sums and operands: the expand
/// exactly, the normed row exactly from the collapsed row, and post and
/// collapsed within the f64 mix-collapse gate.
fn oracle_checks(
    inputs: &Inputs,
    y: &[u16],
    x1: &[u16],
    collapsed: &[u16],
    post: &[u16],
    normed: &[u16],
) -> LabResult<()> {
    let rows = inputs.rows;
    let expand = Query::Expand {
        streams: N as u32,
        width: H as u32,
        capacity: rows as u32,
    };
    let expected =
        oracle::expand(expand, &inputs.streams, y, &inputs.post, &inputs.comb).map_err(err)?;
    exact("expand", x1, &expected)?;
    exact("input norm", normed, &hf_rmsnorm(collapsed, &inputs.weight))?;
    let x = decode(x1);
    let weights = decode(&inputs.weights);
    let base = inputs
        .base
        .iter()
        .map(|&v| f64::from(v))
        .collect::<Vec<_>>();
    let scale = inputs.scale.map(f64::from);
    let mix = MixWeights {
        projection: &weights,
        base: &base,
        scale: &scale,
    };
    let full = by_rows(rows, |r| {
        let query = Query::MixCollapse {
            mix: SERVED.mix,
            capacity: r.len() as u32,
        };
        oracle::mix_collapse(query, &x[r.start * FLAT..r.end * FLAT], mix).unwrap()
    });
    let concat = |part: fn(&oracle::MixCollapseOutput) -> &OracleOutput| {
        let mut out = OracleOutput {
            values: Vec::new(),
            sum_abs_products: Vec::new(),
            reduction_lengths: Vec::new(),
        };
        for output in &full {
            let output = part(output);
            out.values.extend(&output.values);
            out.sum_abs_products.extend(&output.sum_abs_products);
            out.reduction_lengths.extend(&output.reduction_lengths);
        }
        out
    };
    gate(
        "collapsed from the streams (128-K accumulation)",
        &decode(collapsed),
        &concat(|o| &o.collapsed),
    )?;
    gate("post from the streams", &decode(post), &concat(|o| &o.post))
}

/// One rank's buffers and modules.
struct Rank {
    buffers: BTreeMap<u64, DeviceBuffer>,
    rows: usize,
    _modules: Vec<Module>,
    expand: Function,
    mix: Function,
    norm: Function,
    split_rows: Function,
    /// The late-wait mutant of `split_rows`.
    split_late: Function,
    /// The F32 one-shot over this rank alone, which produces `SUMMED` in
    /// the rows transport's replays.
    reduce: Option<Function>,
    /// The producer of `PARTIAL` for the one-shot.
    partials: Option<PartialProducer>,
    /// With one rank, the watch producer of `SUMMED` from `FRESH`.
    watch: Option<WatchProducer>,
}

/// The watch producer of the single-rank sums (`probe_pdl_change_watch`):
/// after its own dependency wait it fills `SUMMED` with NaN and reads the
/// first word of the epilogue's round (`REDUCED_F`, poisoned), triggers its
/// dependents, and holds until that word changes or its hold passes, then
/// copies the sums from `FRESH`. The record says whether the epilogue wrote
/// its round, and so read the sums, before the copy.
struct WatchProducer {
    _module: Module,
    function: Function,
    blocks: u32,
    record: DeviceBuffer,
}

impl WatchProducer {
    fn new(session: &Session, rows: usize) -> LabResult<Self> {
        let module = session.module(WATCH_CUBIN)?;
        let function = module
            .function("chop_probe_pdl_change_watch_producer_v1")
            .map_err(err)?;
        Ok(Self {
            _module: module,
            function,
            blocks: rows.min(64) as u32,
            record: session.allocate(8)?,
        })
    }

    /// One launch, PDL after the stream's previous launch: `output` from
    /// `fresh`, watching the first word of `watch`.
    fn launch(
        &self,
        session: &Session,
        fresh: &DeviceBuffer,
        output: &DeviceBuffer,
        watch: &DeviceBuffer,
        hold_ns: u64,
    ) -> LabResult<()> {
        poison::fill_all(session, &self.record, &[0xff])?;
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (self.blocks, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(fresh),
                    KernelArgument::Buffer(output),
                    KernelArgument::U64((output.byte_len() / 4) as u64),
                    KernelArgument::Buffer(watch),
                    KernelArgument::U64(hold_ns),
                    KernelArgument::Buffer(&self.record),
                ],
            )
            .map_err(err)
    }

    /// Whether the dependent got ahead, and the nanoseconds held, after the
    /// stream synchronized. A record the producer did not write is an
    /// error.
    fn record(&self, session: &Session) -> LabResult<(bool, u32)> {
        let bytes = poison::read(session, &self.record)?;
        let word = |i: usize| u32::from_ne_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        match word(0) {
            0 => Ok((false, word(1))),
            1 => Ok((true, word(1))),
            other => Err(format!("the watch producer recorded {other:#x}")),
        }
    }
}

/// The device producer of a rank's one-shot partials, the lab's PDL
/// prefetch probe with one operand: after its own dependency wait it fills
/// `PARTIAL` with NaN, triggers its dependents, and only after its tail
/// copies the call's partials from `FRESH`. A one-shot that reads its
/// partials before its dependency wait sums NaN, and so does an epilogue
/// that reads the sums before its own.
struct PartialProducer {
    _module: Module,
    function: Function,
    blocks: u32,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    live: DeviceBuffer,
    descriptor: DeviceBuffer,
}

impl PartialProducer {
    fn new(
        session: &Session,
        fresh: &DeviceBuffer,
        partial: &DeviceBuffer,
        blocks: u32,
    ) -> LabResult<Self> {
        let module = session.module(PROBE_CUBIN)?;
        let function = module
            .function("chop_probe_pdl_prefetch_producer_v1")
            .map_err(err)?;
        let bytes = partial.byte_len();
        let mut descriptor = Vec::new();
        for buffer in [fresh, partial] {
            descriptor.extend_from_slice(
                &buffer
                    .device_address_at(0, bytes)
                    .map_err(err)?
                    .to_ne_bytes(),
            );
        }
        descriptor.extend_from_slice(&((bytes / 4) as u32).to_ne_bytes());
        // Bit 1: NaN poison; bit 0 clear: `FRESH` keeps the call's partials.
        descriptor.extend_from_slice(&2_u32.to_ne_bytes());
        let zeros = |bytes: usize| -> LabResult<DeviceBuffer> {
            let buffer = session.allocate(bytes)?;
            poison::fill_all(session, &buffer, &[0])?;
            Ok(buffer)
        };
        let staged = session.allocate(descriptor.len())?;
        poison::write(session, &staged, 0, &descriptor)?;
        Ok(Self {
            _module: module,
            function,
            blocks,
            eviction: zeros(16)?,
            checksum: session.allocate(blocks as usize * 128 * 4)?,
            live: zeros(16)?,
            descriptor: staged,
        })
    }

    /// One launch, PDL after the stream's previous launch.
    fn launch(&self, session: &Session) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (self.blocks, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(&self.descriptor),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(&self.live),
                    KernelArgument::U32(0),
                    KernelArgument::U64(PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

impl Rank {
    /// A rank's buffers, with one-shot state for `comm_ranks` ranks (none
    /// when 0) and then the partials' producer; with one rank, also the
    /// one-shot over this rank alone, whose partial is the four ranks' sums.
    fn new(session: &Session, inputs: &Inputs, rank: usize, comm_ranks: usize) -> LabResult<Self> {
        let rows = inputs.rows;
        let sinkhorn = session.module(SINKHORN_CUBIN)?;
        let reference = session.module(NORM_CUBIN)?;
        let split = session.module(SPLIT_ROWS_CUBIN)?;
        let late = session.module(LATE_WAIT_CUBIN)?;
        let function = |module: &Module, symbol: &str| module.function(symbol).map_err(err);
        let (_, norm_symbol) = sinkhorn_epilogue::reference_norm();
        let expand = function(&sinkhorn, &sinkhorn_entry(Op::Expand).symbol)?;
        let mix = function(
            &sinkhorn,
            &sinkhorn_entry(Op::MixCollapse { probe: false }).symbol,
        )?;
        let norm = function(&reference, &norm_symbol)?;
        let split_rows = function(&split, &split_entry().symbol)?;
        let split_late = function(&late, &late_wait_symbol())?;
        let zeros = |bytes: usize| -> LabResult<DeviceBuffer> {
            let buffer = session.allocate(bytes)?;
            poison::fill_all(session, &buffer, &[0])?;
            Ok(buffer)
        };
        let mut buffers = BTreeMap::new();
        buffers.insert(X0, session.upload_bf16(&inputs.x0)?);
        buffers.insert(Y0, session.upload_bf16(&inputs.y0)?);
        buffers.insert(POST0, session.upload_bf16(&inputs.post0)?);
        buffers.insert(COMB0, session.upload_bf16(&inputs.comb0)?);
        buffers.insert(X, session.upload_bf16(&inputs.streams)?);
        buffers.insert(FN, session.upload_bf16(&inputs.weights)?);
        buffers.insert(BASE, session.upload_f32(&inputs.base)?);
        buffers.insert(SCALE, session.upload_f32(&inputs.scale)?);
        buffers.insert(WEIGHT, session.upload_bf16(&inputs.weight)?);
        buffers.insert(PARTIAL, session.upload_f32(&inputs.partials[rank])?);
        buffers.insert(SUMMED, session.upload_f32(&inputs.summed())?);
        buffers.insert(TICKETS, zeros(rows * 4)?);
        for (handle, bytes) in [
            (Y, rows * H * 2),
            (POST_U, rows * N * 2),
            (COMB_U, rows * CELLS * 2),
            (X1_U, rows * FLAT * 2),
            (COLLAPSED_U, rows * H * 2),
            (NORMED_U, rows * H * 2),
            (POST_F, rows * N * 2),
            (COMB_F, rows * CELLS * 2),
            (REDUCED_F, rows * H * 2),
            (X1_F, rows * FLAT * 2),
            (COLLAPSED_F, rows * H * 2),
            (NORMED_F, rows * H * 2),
            (SUMMED_F, rows * H * 4),
            (
                CHAIN_SUMS,
                sinkhorn_hc::split::partials_bytes(SERVED.mix, rows as u32) as usize,
            ),
            (LIVE, 4),
        ] {
            buffers.insert(handle, session.allocate(bytes)?);
        }
        if comm_ranks > 0 {
            // The one-shot's state of `rows` rows (one tile per row): an
            // inbox slot of the whole reduction per source, a data flag and
            // an acknowledgment per rank and tile, an epoch per tile. It
            // persists across calls, as the protocol requires.
            let tiles = rows;
            buffers.insert(INBOX, session.allocate(comm_ranks * rows * H * 4)?);
            buffers.insert(FLAGS, zeros(2 * comm_ranks * tiles * 4)?);
            buffers.insert(EPOCH, zeros(tiles * 4)?);
        }
        let partials = if comm_ranks > 0 {
            let fresh = match comm_ranks {
                1 => inputs.summed(),
                _ => inputs.partials[rank].clone(),
            };
            buffers.insert(FRESH, session.upload_f32(&fresh)?);
            poison::write(session, &buffers[&PARTIAL], 0, &words(&fresh))?;
            Some(PartialProducer::new(
                session,
                &buffers[&FRESH],
                &buffers[&PARTIAL],
                rows.min(64) as u32,
            )?)
        } else {
            None
        };
        let watch = match comm_ranks {
            1 => Some(WatchProducer::new(session, rows)?),
            _ => None,
        };
        let mut modules = vec![sinkhorn, reference, split, late];
        let reduce = match comm_ranks {
            1 => {
                let comm = session.module(COMM_CUBIN)?;
                let reduce = function(&comm, "chop_comm_allreduce_oneshot_f32_v1")?;
                modules.push(comm);
                Some(reduce)
            }
            _ => None,
        };
        let rank = Self {
            buffers,
            rows,
            _modules: modules,
            expand,
            mix,
            norm,
            split_rows,
            split_late,
            reduce,
            partials,
            watch,
        };
        rank.reset(session, inputs, rows)?;
        Ok(rank)
    }

    fn buffer(&self, handle: u64) -> &DeviceBuffer {
        &self.buffers[&handle]
    }

    /// Restores the sublayer's post and comb on both paths and sets the
    /// live count.
    fn reset(&self, session: &Session, inputs: &Inputs, live: usize) -> LabResult<()> {
        self.reset_unfused(session, inputs)?;
        for (handle, image) in [(POST_F, &inputs.post), (COMB_F, &inputs.comb)] {
            poison::write(session, self.buffer(handle), 0, &halves(image))?;
        }
        poison::write(session, self.buffer(LIVE), 0, &(live as u32).to_ne_bytes())
    }

    /// Restores the unfused path's post and comb alone.
    fn reset_unfused(&self, session: &Session, inputs: &Inputs) -> LabResult<()> {
        for (handle, image) in [(POST_U, &inputs.post), (COMB_U, &inputs.comb)] {
            poison::write(session, self.buffer(handle), 0, &halves(image))?;
        }
        Ok(())
    }

    /// The F32 one-shot over this rank alone: `PARTIAL` into `SUMMED`
    /// (`0 + p`, every row of the bucket), on this rank's persistent state.
    fn reduce_one(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let function = self.reduce.as_ref().ok_or("no single-rank one-shot")?;
        let plan = LaunchPlan {
            grid: [self.rows as u32, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl,
        };
        let args = comm::arguments(OneShotCall {
            input: PARTIAL,
            output: SUMMED,
            peer_inboxes: vec![INBOX],
            peer_flags: vec![FLAGS],
            epoch: EPOCH,
            elements: (self.rows * H) as u64,
            rank: 0,
            ranks: 1,
        })
        .map_err(err)?;
        self.launch(session, function, plan, &args)
    }

    /// The split outputs of the live rows against the unfused sequence over
    /// the same streams and the sums now in `SUMMED`, rounded on the host:
    /// every output bit for bit, the round included.
    fn against_unfused(
        &self,
        session: &Session,
        inputs: &Inputs,
        live: usize,
        label: &str,
    ) -> LabResult<()> {
        session.stream.synchronize().map_err(err)?;
        let sums = session.download_f32(self.buffer(SUMMED), self.rows * H)?;
        let y = sums.into_iter().map(f32_to_bf16).collect::<Vec<_>>();
        poison::write(session, self.buffer(Y), 0, &halves(&y))?;
        self.reset_unfused(session, inputs)?;
        self.unfused(session)?;
        session.stream.synchronize().map_err(err)?;
        let outputs = self.outputs(session, live)?;
        exact(&format!("{label}: round"), &outputs[0][0], &y[..live * H])?;
        same_outputs(label, &outputs)?;
        self.tickets_zero(session, label)
    }

    fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self.buffers.iter().map(|(&k, v)| (k, v)).collect(),
        }
    }

    fn launch(
        &self,
        session: &Session,
        function: &Function,
        plan: LaunchPlan,
        args: &ArgumentList,
    ) -> LabResult<()> {
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            args,
            &self.bindings(),
        )
    }

    fn sinkhorn(&self, session: &Session, op: Op, handles: &[u64], pdl: bool) -> LabResult<()> {
        let entry = sinkhorn_entry(op);
        let mut plan = entry.plan(self.rows as u32, 0).map_err(err)?;
        plan.pdl = pdl;
        let args = entry
            .pack_arguments(handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect())
            .map_err(err)?;
        let function = match op {
            Op::Expand => &self.expand,
            _ => &self.mix,
        };
        self.launch(session, function, plan, &args)
    }

    /// The producer of the streams the epilogue reads: the expand of
    /// (x0, y0, post0, comb0) into `X`.
    fn producer(&self, session: &Session) -> LabResult<()> {
        self.sinkhorn(session, Op::Expand, &[X0, Y0, POST0, COMB0, X, LIVE], false)
    }

    /// The producer of the one-shot's partials: `PARTIAL` from `FRESH`.
    fn partials(&self, session: &Session) -> LabResult<()> {
        self.partials
            .as_ref()
            .ok_or("no partials producer")?
            .launch(session)
    }

    /// The unfused launches after the round: expand, mix-collapse, norm.
    fn unfused(&self, session: &Session) -> LabResult<()> {
        self.sinkhorn(
            session,
            Op::Expand,
            &[X, Y, POST_U, COMB_U, X1_U, LIVE],
            false,
        )?;
        self.sinkhorn(
            session,
            Op::MixCollapse { probe: false },
            &[X1_U, FN, BASE, SCALE, COLLAPSED_U, POST_U, COMB_U, LIVE],
            false,
        )?;
        let (norm, _) = sinkhorn_epilogue::reference_norm();
        let plan = LaunchPlan {
            grid: [self.rows as u32, 1, 1],
            ..norm.plan(0).map_err(err)?
        };
        let args = norm
            .arguments(kernels_sm120::row::rmsnorm::RmsNormBuffers {
                x: COLLAPSED_U,
                weight: WEIGHT,
                residual: None,
                y: NORMED_U,
                live_rows: LIVE,
            })
            .map_err(err)?;
        self.launch(session, &self.norm, plan, &args)
    }

    /// The split path's epilogue buffers.
    fn epilogue() -> SplitBuffers {
        SplitBuffers {
            epilogue: EpilogueBuffers {
                streams: X,
                post: POST_F,
                comb: COMB_F,
                reduced: REDUCED_F,
                expanded: X1_F,
                fn_: FN,
                base: BASE,
                scale: SCALE,
                collapsed: COLLAPSED_F,
                weight: WEIGHT,
                normed: NORMED_F,
                live_rows: LIVE,
                eps_bits: EPS.to_bits(),
            },
            partials: CHAIN_SUMS,
            tickets: TICKETS,
        }
    }

    /// The split epilogue over `SUMMED`.
    fn split_rows(&self, session: &Session, pdl: bool) -> LabResult<()> {
        self.split_rows_over(session, SUMMED, pdl)
    }

    /// The split epilogue over the F32 sums in `summed`.
    fn split_rows_over(&self, session: &Session, summed: u64, pdl: bool) -> LabResult<()> {
        self.split_with(session, &self.split_rows, summed, pdl)
    }

    /// The late-wait mutant in place of the split epilogue.
    fn split_late_over(&self, session: &Session, summed: u64, pdl: bool) -> LabResult<()> {
        self.split_with(session, &self.split_late, summed, pdl)
    }

    fn split_with(
        &self,
        session: &Session,
        function: &Function,
        summed: u64,
        pdl: bool,
    ) -> LabResult<()> {
        let entry = split_entry();
        let mut plan = entry.plan(self.rows as u32, RANKS as u32).map_err(err)?;
        plan.pdl = pdl;
        let args = sinkhorn_split::rows_arguments(summed, Self::epilogue()).map_err(err)?;
        self.launch(session, function, plan, &args)
    }

    fn tickets_zero(&self, session: &Session, label: &str) -> LabResult<()> {
        let tickets = poison::read(session, self.buffer(TICKETS))?;
        exact(
            &format!("{label}: tickets"),
            &tickets,
            &vec![0; self.rows * 4],
        )
    }

    fn bf16(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<u16>> {
        session.download_bf16(self.buffer(handle), self.rows * per_row)
    }

    /// The split and unfused outputs of the live rows, in the order
    /// y, X', post, comb, collapsed, normed.
    fn outputs(&self, session: &Session, live: usize) -> LabResult<[[Vec<u16>; 6]; 2]> {
        let take = |handle, per_row| -> LabResult<Vec<u16>> {
            let mut values = self.bf16(session, handle, per_row)?;
            values.truncate(live * per_row);
            Ok(values)
        };
        Ok([
            [
                take(Y, H)?,
                take(X1_U, FLAT)?,
                take(POST_U, N)?,
                take(COMB_U, CELLS)?,
                take(COLLAPSED_U, H)?,
                take(NORMED_U, H)?,
            ],
            [
                take(REDUCED_F, H)?,
                take(X1_F, FLAT)?,
                take(POST_F, N)?,
                take(COMB_F, CELLS)?,
                take(COLLAPSED_F, H)?,
                take(NORMED_F, H)?,
            ],
        ])
    }
}

const OUTPUTS: [&str; 6] = ["y", "X'", "post", "comb", "collapsed", "normed"];

/// The split outputs equal the unfused ones bit for bit.
fn same_outputs(label: &str, [unfused, fused]: &[[Vec<u16>; 6]; 2]) -> LabResult<()> {
    for ((name, unfused), fused) in OUTPUTS.iter().zip(unfused).zip(fused) {
        exact(&format!("{label}: split {name}"), fused, unfused)?;
    }
    Ok(())
}

/// The rows transport at one bucket: from the same sums, the fused outputs
/// equal the unfused launches' and the oracles.
fn rows_bucket(session: &Session, rows: usize) -> LabResult<Value> {
    let inputs = Inputs::new(rows, 0x5eed + rows as u64);
    let rank = Rank::new(session, &inputs, 0, 0)?;
    let y = inputs
        .summed()
        .into_iter()
        .map(f32_to_bf16)
        .collect::<Vec<_>>();
    poison::write(session, rank.buffer(Y), 0, &halves(&y))?;
    rank.unfused(session)?;
    rank.split_rows(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let outputs = rank.outputs(session, rows)?;
    exact("round", &outputs[0][0], &y)?;
    same_outputs(&format!("{rows} rows"), &outputs)?;
    rank.tickets_zero(session, &format!("{rows} rows"))?;
    let [_, x1, post, _, collapsed, normed] = &outputs[1];
    oracle_checks(&inputs, &y, x1, collapsed, post, normed)?;
    Ok(json!({"rows": rows, "bitwise": true}))
}

/// The poisoned replays of one transport's split launch on one rank: the
/// produced streams and the outputs past `live` keep their poison, post and
/// comb their image, the chain sums of the live rows are written and the
/// tickets stay zero.
#[allow(clippy::single_range_in_vec_init)]
fn replay<'a>(
    session: &'a Session,
    rank: &'a Rank,
    inputs: &Inputs,
    live: usize,
) -> LabResult<Replay<'a>> {
    let prefix = |per_row: usize| WriteMap::of(DType::Bf16).prefix(live * per_row);
    let chain_row = sinkhorn_hc::split::partials_bytes(SERVED.mix, 1) as usize;
    Ok(Replay::new(session)
        // The producer's streams: poisoned before every producer, so a
        // consumer that reads them before the producer wrote them, or a
        // producer that wrote nothing, fails.
        .output("streams", rank.buffer(X), prefix(FLAT)?)
        .output("reduced", rank.buffer(REDUCED_F), prefix(H)?)
        .output("expanded", rank.buffer(X1_F), prefix(FLAT)?)
        .output("collapsed", rank.buffer(COLLAPSED_F), prefix(H)?)
        .output("normed", rank.buffer(NORMED_F), prefix(H)?)
        .output(
            "chain sums",
            rank.buffer(CHAIN_SUMS),
            WriteMap::of(DType::F32).byte_prefix(live * chain_row)?,
        )
        .state(
            "post",
            rank.buffer(POST_F),
            halves(&inputs.post),
            WriteMap::of(DType::Bf16).none()?,
            vec![0..live * N * 2],
        )
        .state(
            "comb",
            rank.buffer(COMB_F),
            halves(&inputs.comb),
            WriteMap::of(DType::Bf16).none()?,
            vec![0..live * CELLS * 2],
        )
        .state(
            "tickets",
            rank.buffer(TICKETS),
            vec![0; rank.rows * 4],
            WriteMap::of(DType::I32).none()?,
            Vec::new(),
        ))
}

fn rows_replay_bucket(session: &Session, rows: usize) -> LabResult<Value> {
    let inputs = Inputs::new(rows, 0xfeed + rows as u64);
    // The single-rank one-shot reads the four ranks' sums as its partial,
    // which the partials' producer writes.
    let rank = Rank::new(session, &inputs, 0, 1)?;
    // The streams' producer, the partials' producer (PDL), then the sums'
    // producer (which publishes them at the end of its protocol) and the
    // epilogue, PDL on both when `pdl`.
    let chain = |pdl: bool| -> LabResult<()> {
        rank.producer(session)?;
        rank.partials(session)?;
        rank.reduce_one(session, pdl)?;
        rank.split_rows(session, pdl)
    };
    let graph = session.capture(|_| chain(true))?;
    let replay_of = |live: usize| -> LabResult<Replay<'_>> {
        Ok(replay(session, &rank, &inputs, live)?
            .output(
                "partials",
                rank.buffer(PARTIAL),
                WriteMap::of(DType::F32).prefix(rows * H)?,
            )
            .output(
                "sums",
                rank.buffer(SUMMED),
                WriteMap::of(DType::F32).prefix(rows * H)?,
            ))
    };
    let summed = bits(&inputs.summed());
    let mut evidence = Vec::new();
    for live in poison::live_counts(rows as u32) {
        let live = live as usize;
        let label = format!("{rows} rows, live {live}");
        rank.reset(session, &inputs, live)?;
        let mut replay = replay_of(live)?;
        replay.eager(|| chain(false))?;
        // The eager bits against the host's sums (0 + the four ranks' sum)
        // and the unfused sequence, independently of the replays'
        // self-comparison.
        let sums = session.download_f32(rank.buffer(SUMMED), rows * H)?;
        exact(&format!("{label}: sums"), &bits(&sums), &summed)?;
        rank.against_unfused(session, &inputs, live, &label)?;
        replay.relaunch(|| chain(false))?;
        replay.graph(&graph)?;
        replay.pdl(
            || rank.producer(session),
            || {
                rank.partials(session)?;
                rank.reduce_one(session, true)?;
                rank.split_rows(session, true)
            },
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        // The mutations at one partial live count (all rows of one row).
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            mutations = rows_mutations(session, &rank, &inputs, live, &replay, &label)?;
        }
        evidence.push(json!({"live": live, "replay": receipt, "mutations_failed": mutations}));
    }
    Ok(json!({"rows": rows, "live_counts": evidence}))
}

/// The checks catch a missing producer, an early read and missing state
/// stores. With the streams', the partials' or the sums' producer left out,
/// the output the mutation corrupts differs from the eager bits. The
/// epilogue's dependency wait is checked behind the sums' watch producer
/// (`watch_mutation`). Synchronization and readback errors propagate rather
/// than count as a caught mutation. With
/// the epilogue's post and comb stores undone after a correct launch, the
/// comparison with the unfused sequence fails (the replay's own check lets
/// read-modify-write state keep its image): exactly the split post and comb
/// differ from the unfused ones. Returns the mutations caught, which must
/// be all.
fn rows_mutations(
    session: &Session,
    rank: &Rank,
    inputs: &Inputs,
    live: usize,
    eager: &Replay<'_>,
    label: &str,
) -> LabResult<Vec<Value>> {
    let mut caught = Vec::new();
    for (name, streams, partials, sums, corrupted) in [
        ("no streams producer", false, true, true, "streams"),
        ("no partials producer", true, false, true, "partials"),
        ("no sums producer", true, true, false, "sums"),
    ] {
        let label = format!("{label}, {name}");
        session.stream.synchronize().map_err(err)?;
        rank.reset(session, inputs, live)?;
        eager.prepare(Phase::Pdl)?;
        if streams {
            rank.producer(session)?;
        }
        if partials {
            rank.partials(session)?;
        }
        if sums {
            rank.reduce_one(session, true)?;
        }
        rank.split_rows(session, true)?;
        session.stream.synchronize().map_err(err)?;
        let differing = differing(session, rank, eager, SUMMED, &label)?;
        if !differing.contains(&corrupted) {
            return Err(format!(
                "{label}: the {corrupted} equal the eager bits (differing: {differing:?})"
            ));
        }
        caught.push(json!({"mutation": name, "output": corrupted}));
    }
    caught.push(watch_mutation(session, rank, inputs, live, eager, label)?);
    // A correct launch, then the post and comb stores undone.
    rank.reset(session, inputs, live)?;
    rank.producer(session)?;
    rank.partials(session)?;
    rank.reduce_one(session, false)?;
    rank.split_rows(session, false)?;
    session.stream.synchronize().map_err(err)?;
    rank.against_unfused(session, inputs, live, label)?;
    for (handle, image) in [(POST_F, &inputs.post), (COMB_F, &inputs.comb)] {
        poison::write(session, rank.buffer(handle), 0, &halves(image))?;
    }
    session.stream.synchronize().map_err(err)?;
    // The unfused outputs the comparison above checked, against the split
    // outputs now: exactly the undone post and comb differ. Readback errors
    // propagate rather than count as caught.
    let [unfused, split] = rank.outputs(session, live)?;
    let differing = OUTPUTS
        .iter()
        .zip(unfused.iter().zip(&split))
        .filter(|(_, (unfused, split))| unfused != split)
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    if differing != ["post", "comb"] {
        return Err(format!(
            "{label}: with the post and comb stores undone, {differing:?} differ from the unfused outputs, not post and comb"
        ));
    }
    caught.push(json!({"mutation": "post and comb stores undone", "outputs": differing}));
    Ok(caught)
}

/// The epilogue's dependency wait behind the sums' watch producer (streams'
/// producer, then the watch producer of `SUMMED`, then the epilogue, PDL to
/// the last two; no partials or one-shot). First the served epilogue with
/// the served hold: it must not get ahead, and its streams, round and sums
/// must equal the eager bits. Then the late-wait mutant with the mutant's
/// hold: if it got ahead, its round must differ from the eager bits and the
/// streams and sums must not, and it is caught; if it did not and nothing
/// differs, PDL did not overlap the launches and it is inconclusive
/// (reported, not caught); anything else fails. Synchronization and
/// readback errors propagate.
fn watch_mutation(
    session: &Session,
    rank: &Rank,
    inputs: &Inputs,
    live: usize,
    eager: &Replay<'_>,
    label: &str,
) -> LabResult<Value> {
    let watch = rank.watch.as_ref().ok_or("no watch producer")?;
    let run = |late_wait: bool, hold_ns: u64| -> LabResult<(bool, u32, Vec<&'static str>)> {
        session.stream.synchronize().map_err(err)?;
        rank.reset(session, inputs, live)?;
        eager.prepare(Phase::Pdl)?;
        rank.producer(session)?;
        watch.launch(
            session,
            rank.buffer(FRESH),
            rank.buffer(SUMMED),
            rank.buffer(REDUCED_F),
            hold_ns,
        )?;
        if late_wait {
            rank.split_late_over(session, SUMMED, true)?;
        } else {
            rank.split_rows(session, true)?;
        }
        session.stream.synchronize().map_err(err)?;
        let (ahead, held) = watch.record(session)?;
        // This chain has no partials' producer: its partials keep their
        // poison.
        let differing = differing(session, rank, eager, SUMMED, label)?
            .into_iter()
            .filter(|output| *output != "partials")
            .collect();
        Ok((ahead, held, differing))
    };
    let (ahead, held, differing) = run(false, SERVED_HOLD_NS)?;
    if ahead || !differing.is_empty() {
        return Err(format!(
            "{label}: the served epilogue behind the watch producer got ahead {ahead} \
             (held {held} ns), differing {differing:?}"
        ));
    }
    let name = "dependency wait after the sum reads";
    let (ahead, held, differing) = run(true, MUTANT_HOLD_NS)?;
    let outcome = match (ahead, differing.as_slice()) {
        (true, ["reduced"]) => "caught",
        (false, []) => {
            eprintln!(
                "{label}, {name}: inconclusive: the mutant never got ahead of the watch \
                 producer (held {held} ns) and nothing differs"
            );
            "inconclusive"
        }
        _ => {
            return Err(format!(
                "{label}, {name}: got ahead {ahead} (held {held} ns), differing {differing:?}, \
                 not only the round"
            ));
        }
    };
    Ok(json!({
        "mutation": name,
        "outcome": outcome,
        "output": "reduced",
        "ahead": ahead,
        "held_ns": held,
        "served_held_ns": SERVED_HOLD_NS,
    }))
}

/// The outputs a mutation may corrupt, by replay output name.
const MUTATED: [(&str, u64); 4] = [
    ("streams", X),
    ("partials", PARTIAL),
    ("reduced", REDUCED_F),
    ("sums", SUMMED_F),
];

/// Whether a mutation was caught on one rank: the outputs among `MUTATED`
/// (with `sums` the rank's sums buffer) that differ from `replay`'s eager
/// bits, read back after the mutated launches. Readback errors propagate.
fn differing(
    session: &Session,
    rank: &Rank,
    replay: &Replay<'_>,
    sums: u64,
    label: &str,
) -> LabResult<Vec<&'static str>> {
    let mut differing = Vec::new();
    for (output, handle) in MUTATED {
        let handle = if output == "sums" { sums } else { handle };
        let actual = poison::read(session, rank.buffer(handle))?;
        let eager = replay
            .expected(output)
            .ok_or_else(|| format!("{label}: no eager {output}"))?;
        if actual != eager {
            differing.push(output);
        }
    }
    Ok(differing)
}

/// Every rank's state, loaded for collective calls.
struct Ranks<'a> {
    sessions: &'a [Session],
    ranks: Vec<Rank>,
    comm: CollectiveModules,
    rows: usize,
}

impl Ranks<'_> {
    fn call(&self, rank: usize, output: u64) -> OneShotCall {
        OneShotCall {
            input: PARTIAL,
            output,
            peer_inboxes: (0..RANKS as u64).map(|peer| INBOX + peer).collect(),
            peer_flags: (0..RANKS as u64).map(|peer| FLAGS + peer).collect(),
            epoch: EPOCH,
            elements: (self.rows * H) as u64,
            rank: rank as u32,
            ranks: RANKS as u32,
        }
    }

    /// Rank `rank`'s own buffers and every peer's inbox and flags.
    fn bindings(&self, rank: usize) -> Bindings<'_> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        for (&handle, buffer) in &self.ranks[rank].buffers {
            if handle != INBOX && handle != FLAGS {
                buffers.insert(handle, buffer);
            }
        }
        for (peer, state) in self.ranks.iter().enumerate() {
            buffers.insert(INBOX + peer as u64, state.buffer(INBOX));
            buffers.insert(FLAGS + peer as u64, state.buffer(FLAGS));
        }
        Bindings { buffers }
    }

    /// The F32 one-shot into `output` on every rank (or `only`).
    fn reduce(&self, output: u64, only: Option<usize>, pdl: bool) -> LabResult<()> {
        let query = CommQuery::AllReduce {
            ranks: RANKS as u32,
            elements: (self.rows * H) as u64,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let mut plan = comm::plan(&query, 1).map_err(err)?;
        plan.pdl = pdl;
        if comm::choose(&query).map_err(err)? != comm::Variant::OneShot {
            return Err(format!("{} rows do not reduce in one shot", self.rows));
        }
        launch::launch_collective(
            self.sessions,
            &self.comm,
            "chop_comm_allreduce_oneshot_f32_v1",
            plan,
            only,
            |rank| {
                Ok(RankCall {
                    args: comm::arguments(self.call(rank, output)).map_err(err)?,
                    bindings: self.bindings(rank),
                })
            },
        )
    }

    /// The split path on every rank (or `only`): the one-shot into
    /// `SUMMED_F`, then the split epilogue over it, PDL between them and
    /// after the previous launch when `pdl`.
    fn split_path(&self, only: Option<usize>, pdl: bool) -> LabResult<()> {
        self.reduce(SUMMED_F, only, pdl)?;
        for (rank, session) in self.sessions.iter().enumerate() {
            if only.is_none_or(|only| only == rank) {
                self.ranks[rank].split_rows_over(session, SUMMED_F, pdl)?;
            }
        }
        Ok(())
    }

    /// The call's partials: in `FRESH` for the partials' producer, and in
    /// `PARTIAL` for the unfused one-shot.
    fn upload_partials(&self, inputs: &Inputs) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            for handle in [FRESH, PARTIAL] {
                poison::write(
                    session,
                    self.ranks[rank].buffer(handle),
                    0,
                    &words(&inputs.partials[rank]),
                )?;
            }
        }
        Ok(())
    }

    /// Poisons every rank's `PARTIAL`, so the split path sums only what its
    /// producer wrote.
    fn poison_partials(&self) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            poison::fill_all(session, self.ranks[rank].buffer(PARTIAL), &[poison::POISON])?;
        }
        Ok(())
    }

    /// The producers the split path reads on every rank: the streams' and
    /// the partials' (each when asked), the partials' last, so the
    /// one-shot's dependency wait is on it.
    fn producers(&self, streams: bool, partials: bool) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            if streams {
                self.ranks[rank].producer(session)?;
            }
            if partials {
                self.ranks[rank].partials(session)?;
            }
        }
        Ok(())
    }

    /// Every rank's split path sums equal the host's rank-order sums.
    fn split_sums(&self, inputs: &Inputs, label: &str) -> LabResult<()> {
        let summed = bits(&inputs.summed());
        for (rank, session) in self.sessions.iter().enumerate() {
            let sums = session.download_f32(self.ranks[rank].buffer(SUMMED_F), self.rows * H)?;
            exact(
                &format!("{label}: rank {rank} split path sums"),
                &bits(&sums),
                &summed,
            )?;
        }
        Ok(())
    }

    /// One unfused call on every rank: the one-shot into `SUMMED`, every
    /// rank's sums against the host's, the round on the host, then the row
    /// launches.
    fn unfused(&self, inputs: &Inputs) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        self.reduce(SUMMED, None, false)?;
        poison::synchronize(self.sessions)?;
        let summed = inputs.summed();
        let y = summed.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        for (rank, session) in self.sessions.iter().enumerate() {
            let sums = session.download_f32(self.ranks[rank].buffer(SUMMED), self.rows * H)?;
            exact(
                &format!("rank {rank} one-shot sums"),
                &bits(&sums),
                &bits(&summed),
            )?;
            poison::write(session, self.ranks[rank].buffer(Y), 0, &halves(&y))?;
            self.ranks[rank].unfused(session)?;
        }
        poison::synchronize(self.sessions)
    }

    /// Every rank's split outputs equal its unfused ones and rank 0's, its
    /// split path's sums the host's, and its tickets are zero.
    fn compare(&self, inputs: &Inputs, live: usize, label: &str) -> LabResult<()> {
        self.split_sums(inputs, label)?;
        let mut first = None;
        for (rank, session) in self.sessions.iter().enumerate() {
            let outputs = self.ranks[rank].outputs(session, live)?;
            same_outputs(&format!("{label}: rank {rank}"), &outputs)?;
            self.ranks[rank].tickets_zero(session, &format!("{label}: rank {rank}"))?;
            match &first {
                None => first = Some(outputs),
                Some(first) => same_outputs(
                    &format!("{label}: rank {rank} against rank 0"),
                    &[first[1].clone(), outputs[1].clone()],
                )?,
            }
        }
        Ok(())
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// One row count of the one-shot on four ranks: the split and unfused
/// paths alternate on one one-shot state, eagerly and through captured
/// graphs, at every live count, the split path after the device producers
/// of its streams and partials; then the poisoned replays and the
/// mutations.
fn one_shot_bucket(sessions: &[Session], rows: usize) -> LabResult<Value> {
    let mut inputs = Inputs::new(rows, 0x1d1e + rows as u64);
    let ranks = Ranks {
        sessions,
        ranks: sessions
            .iter()
            .enumerate()
            .map(|(rank, session)| Rank::new(session, &inputs, rank, RANKS))
            .collect::<LabResult<_>>()?,
        comm: CollectiveModules::load(sessions, COMM_CUBIN)?,
        rows,
    };
    // Each rank's graph holds the streams' producer, the partials'
    // producer, the one-shot and the epilogue, PDL to the last three.
    let graphs = kernel_lab::topology::RankReplay::new(sessions, |rank, session| {
        ranks.ranks[rank].producer(session)?;
        ranks.ranks[rank].partials(session)?;
        ranks.split_path(Some(rank), true)
    })?;
    let mut calls = Vec::new();
    for (call, live) in poison::live_counts(rows as u32).into_iter().enumerate() {
        let live = live as usize;
        for graph in [false, true] {
            inputs.repartition(0xca11 + 2 * call as u64 + u64::from(graph));
            poison::synchronize(sessions)?;
            ranks.upload_partials(&inputs)?;
            for (rank, session) in sessions.iter().enumerate() {
                ranks.ranks[rank].reset(session, &inputs, live)?;
            }
            ranks.unfused(&inputs)?;
            ranks.poison_partials()?;
            if graph {
                graphs.launch(sessions)?;
            } else {
                ranks.producers(true, true)?;
                ranks.split_path(None, false)?;
            }
            poison::synchronize(sessions)?;
            ranks.compare(
                &inputs,
                live,
                &format!("{rows} rows, live {live}, graph {graph}"),
            )?;
            calls.push(json!({"live": live, "graph": graph}));
        }
    }
    // The poisoned replays: every rank prepared before any launches.
    let mut evidence = Vec::new();
    for live in poison::live_counts(rows as u32) {
        let live = live as usize;
        poison::synchronize(sessions)?;
        for (rank, session) in sessions.iter().enumerate() {
            ranks.ranks[rank].reset(session, &inputs, live)?;
        }
        let mut replays = sessions
            .iter()
            .zip(&ranks.ranks)
            .map(|(session, rank)| {
                Ok(replay(session, rank, &inputs, live)?
                    .output(
                        "partials",
                        rank.buffer(PARTIAL),
                        WriteMap::of(DType::F32).prefix(rows * H)?,
                    )
                    .output(
                        "sums",
                        rank.buffer(SUMMED_F),
                        WriteMap::of(DType::F32).prefix(rows * H)?,
                    ))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let phases: [(Phase, bool); 4] = [
            (Phase::Eager, false),
            (Phase::Relaunch, false),
            (Phase::Graph, false),
            (Phase::Pdl, true),
        ];
        for (phase, pdl) in phases {
            poison::synchronize(sessions)?;
            for replay in &replays {
                replay.prepare(phase)?;
            }
            match phase {
                Phase::Graph => graphs.launch(sessions)?,
                _ => {
                    ranks.producers(true, true)?;
                    ranks.split_path(None, pdl)?;
                }
            }
            poison::synchronize(sessions)?;
            for (rank, replay) in replays.iter_mut().enumerate() {
                replay
                    .finish(phase, pdl)
                    .map_err(|e| format!("{rows} rows, live {live}, rank {rank}: {e}"))?;
            }
            if phase == Phase::Eager {
                // The bits every replay must reproduce, against the host's.
                ranks.split_sums(&inputs, &format!("{rows} rows, live {live}, eager"))?;
            }
        }
        let mut receipt = BTreeMap::new();
        replays[0].record(&mut receipt);
        // The mutations at one partial live count (all rows of one row).
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            mutations = one_shot_mutations(&ranks, &inputs, &replays, live)?;
        }
        evidence.push(json!({"live": live, "replay": receipt, "mutations_failed": mutations}));
    }
    Ok(json!({"rows": rows, "calls": calls, "live_counts": evidence}))
}

/// The split path's checks catch a missing producer: with the streams' or
/// the partials' producer left out, the output the mutation corrupts
/// differs from the eager bits on some rank. Synchronization and readback
/// errors propagate rather than count as a caught mutation. Returns the
/// mutations caught, which must be all.
fn one_shot_mutations(
    ranks: &Ranks<'_>,
    inputs: &Inputs,
    replays: &[Replay<'_>],
    live: usize,
) -> LabResult<Vec<Value>> {
    let rows = ranks.rows;
    let mut caught = Vec::new();
    for (name, streams, partials, corrupted) in [
        ("no streams producer", false, true, "streams"),
        ("no partials producer", true, false, "partials"),
    ] {
        let label = format!("{rows} rows, live {live}, {name}");
        poison::synchronize(ranks.sessions)?;
        for (rank, session) in ranks.sessions.iter().enumerate() {
            ranks.ranks[rank].reset(session, inputs, live)?;
        }
        for replay in replays {
            replay.prepare(Phase::Pdl)?;
        }
        ranks.producers(streams, partials)?;
        ranks.split_path(None, true)?;
        poison::synchronize(ranks.sessions)?;
        // Each rank's outputs that differ from its eager bits.
        let mut all = Vec::new();
        for (rank, (session, replay)) in ranks.sessions.iter().zip(replays).enumerate() {
            for output in differing(session, &ranks.ranks[rank], replay, SUMMED_F, &label)? {
                all.push((rank, output));
            }
        }
        let ranks_corrupted = all
            .iter()
            .filter(|(_, output)| *output == corrupted)
            .count();
        if ranks_corrupted == 0 {
            return Err(format!(
                "{label}: no rank's {corrupted} differ from the eager bits (differing: {all:?})"
            ));
        }
        caught.push(
            json!({"mutation": name, "output": corrupted, "ranks_differing": ranks_corrupted}),
        );
    }
    Ok(caught)
}

#[test]
fn sinkhorn_comm_split_cpu_contract() {
    for (cubin, symbol) in [
        (SPLIT_ROWS_CUBIN, split_entry().symbol),
        (LATE_WAIT_CUBIN, late_wait_symbol()),
        (NORM_CUBIN, sinkhorn_epilogue::reference_norm().1),
        (SINKHORN_CUBIN, sinkhorn_entry(Op::Expand).symbol),
        (
            PROBE_CUBIN,
            "chop_probe_pdl_prefetch_producer_v1".to_string(),
        ),
        (
            WATCH_CUBIN,
            "chop_probe_pdl_change_watch_producer_v1".to_string(),
        ),
    ] {
        assert!(
            kernel_lab::cubin_entries(cubin)
                .unwrap()
                .contains(&symbol.as_str()),
            "{cubin} lacks {symbol}"
        );
        kernel_lab::cubin(cubin).unwrap();
    }
    assert!(!kernel_lab::comm_cubin(SPLIT_ROWS_CUBIN).unwrap());
    assert!(!kernel_lab::comm_cubin(LATE_WAIT_CUBIN).unwrap());
    for rows in ONE_SHOT_BUCKETS {
        let query = CommQuery::AllReduce {
            ranks: RANKS as u32,
            elements: (rows * H) as u64,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert_eq!(comm::choose(&query).unwrap(), comm::Variant::OneShot);
    }
    for rows in ROWS_BUCKETS {
        let plan = split_entry().plan(rows as u32, RANKS as u32).unwrap();
        assert_eq!(plan.grid, [SLICES as u32, rows as u32, 1]);
    }
    // The host's references: the rank-order sum from zero, the exact norm.
    let mut inputs = Inputs::new(2, 1);
    inputs.partials = vec![
        vec![1.0; 2 * H],
        vec![2.0; 2 * H],
        vec![-0.5; 2 * H],
        vec![0.25; 2 * H],
    ];
    assert!(inputs.summed().iter().all(|&v| v == 2.75));
    let ones = vec![f32_to_bf16(1.0); H];
    assert!(
        hf_rmsnorm(&ones, &ones)
            .iter()
            .all(|&v| v == f32_to_bf16(1.0))
    );
}

#[test]
fn the_split_rows_epilogue_matches_the_unfused_launches_bitwise() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let buckets = ROWS_BUCKETS
            .iter()
            .map(|&rows| rows_bucket(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_split_rows"));
        receipt.insert("buckets".into(), json!(buckets));
        session.receipt("sinkhorn_comm_split_rows", receipt)?;
        Ok(())
    })
    .expect("the split rows epilogue against the unfused launches");
}

#[test]
fn the_split_rows_epilogue_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let buckets = ROWS_BUCKETS
            .iter()
            .map(|&rows| rows_replay_bucket(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_split_rows_poison"));
        receipt.insert("buckets".into(), json!(buckets));
        session.receipt("sinkhorn_comm_split_rows_poison", receipt)?;
        Ok(())
    })
    .expect("the split rows epilogue's poisoned replays");
}

#[test]
fn the_split_epilogue_after_the_one_shot_matches_the_unfused_launches_on_four_ranks() {
    kernel_lab::gpu_test_multi(2 << 30, |sessions| {
        if sessions.len() != RANKS {
            return kernel_lab::skip("the one-shot runs on four GPUs (TP4)");
        }
        let buckets = ONE_SHOT_BUCKETS
            .iter()
            .map(|&rows| one_shot_bucket(sessions, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_split_one_shot"));
        receipt.insert("buckets".into(), json!(buckets));
        sessions[0].receipt("sinkhorn_comm_split_one_shot", receipt)?;
        Ok(())
    })
    .expect("the split epilogue after the one-shot on four ranks");
}
