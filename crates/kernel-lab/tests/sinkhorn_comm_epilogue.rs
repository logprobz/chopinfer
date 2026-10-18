//! The Sinkhorn mHC comm epilogue (`kernels_sm120::comm::sinkhorn_epilogue`)
//! at the served 4 x 4096 mixer, against the unfused launch sequence it
//! replaces, bit for bit, on the GPU. Run the rows transport with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test
//! sinkhorn_comm_epilogue -- --test-threads=1`, and the one-shot transport
//! with `CHOPINFER_KERNEL_GPUS=0,1,2,3` (TP4); without GPUs only the CPU
//! contract test runs.
//!
//! The unfused sequence is the served one: the F32 one-shot all-reduce
//! (`comm_sm120`), the round of each sum to BF16, the Sinkhorn expand and
//! the next mix-collapse (`sinkhorn_hc_hc4x4096_sm120`) and the HF-ordered
//! input RMSNorm (the reference entry of `sinkhorn_epilogue_hc4x4096_sm120`).
//! The round runs on the host: it is one cvt.rn per element, which the
//! model-owned round row reproduces bit for bit (its own lab test).
//!
//! - Rows transport, one GPU, at decode buckets (1, 16, 32), verify rows
//!   (6 x 32) and a prefill chunk (8192): from the same F32 sums, every
//!   output (y, X', post, comb, collapsed, normed) equals the unfused
//!   launches' bit for bit; X' equals the kernel-core expand exactly, the
//!   normed row the exact F32 HF RMSNorm of the collapsed row, and post and
//!   collapsed the f64 mix-collapse oracle within its gate (the 128-K
//!   accumulation bound of the 16,384-wide projection).
//! - One-shot transport, four GPUs, at every row count it serves (1, 2, 7,
//!   16): the fused and unfused entries alternate on one comm state; every
//!   rank's sums equal the host's rank-order sums and every output equals
//!   the unfused sequence's and rank 0's, eagerly and through each rank's
//!   captured graph, at every live count.
//! - Both transports replayed on poisoned buffers (eager, relaunch, graph,
//!   PDL) at every live count `poison::live_counts` lists, after producers
//!   whose outputs are poisoned and checked too: the streams' producer (an
//!   expand) for both; for the rows transport the sums' producer, an F32
//!   one-shot over one rank on persistent state; for the one-shot transport
//!   the partials' producer (`probe_pdl_watch_sm120`), which holds each
//!   rank's partials at NaN after its trigger of its dependents, then copies
//!   the call's own partials in, and records whether the one-shot published
//!   (so read its partials) before that write: never, for the served
//!   one-shot, in any replay. The one-shot's
//!   eager and graph calls run after that producer too, on partials the
//!   unfused call left poisoned. Rows past the live count keep their
//!   poison, and post and comb (read, then overwritten) keep their image
//!   there. Each rows-transport live count's eager outputs are also
//!   compared with the unfused sequence, and the checks must fail with any
//!   producer left out, with the post and comb stores undone (rows), and
//!   with the one-shot's dependency wait moved after its partial reads
//!   (`comm_sinkhorn_late_wait_hc4x4096_sm120`, under PDL). For that mutant
//!   the producer holds its write until the mutant has published or
//!   `LATE_WAIT_HOLD_NS` passed: every rank whose mutant got ahead must sum
//!   differently from the eager bits, and if none got ahead (no PDL
//!   overlap) the mutation is reported inconclusive, neither caught nor
//!   missed.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, ArgumentList, CommQuery, DType, LaunchPlan};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_lab::launch::{self, Bindings, CollectiveModules, RankCall};
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::comm::sinkhorn_epilogue::{self, EpilogueBuffers, Transport};
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
/// Decode buckets, verify rows (6 x 32) and a prefill chunk.
const ROWS_BUCKETS: [usize; 5] = [1, 16, 32, 6 * 32, 8192];
/// Every row count the F32 one-shot serves at TP4 that a bucket takes.
const ONE_SHOT_BUCKETS: [usize; 4] = [1, 2, 7, 16];

const COMM_CUBIN: &str = "comm_sm120";
const FUSED_COMM_CUBIN: &str = "comm_sinkhorn_hc4x4096_sm120";
const ROWS_CUBIN: &str = "sinkhorn_epilogue_hc4x4096_sm120";
const SINKHORN_CUBIN: &str = "sinkhorn_hc_hc4x4096_sm120";
/// The one-shot with its dependency wait after its partial reads, a mutant.
const LATE_WAIT_CUBIN: &str = "comm_sinkhorn_late_wait_hc4x4096_sm120";
const PROBE_CUBIN: &str = "probe_pdl_watch_sm120";
const PROBE_SYMBOL: &str = "chop_probe_pdl_watch_producer_v1";
/// The partials' producer's hold between its trigger and its write.
const PRODUCER_TAIL_NS: u64 = 50_000;
/// Its longest hold for the late-wait mutant, which publishes as soon as
/// it runs: only a consumer PDL did not overlap with it waits this long.
const LATE_WAIT_HOLD_NS: u64 = 200_000_000;

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
/// The call's partials, which the partials' producer copies into `PARTIAL`.
const FRESH: u64 = 27;
/// Every rank's one-shot inbox and flags: `INBOX + rank`, `FLAGS + rank`.
const INBOX: u64 = 100;
const FLAGS: u64 = 200;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn served_entry(transport: Transport) -> sinkhorn_epilogue::Entry {
    sinkhorn_epilogue::catalog()
        .into_iter()
        .find(|entry| entry.transport == transport)
        .expect("a catalog entry")
}

/// The late-wait mutant's symbol.
fn late_wait_symbol() -> String {
    served_entry(Transport::OneShot).symbol.replacen(
        "chop_comm_",
        sinkhorn_epilogue::LATE_WAIT_PREFIX,
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
    fused_rows: Function,
    /// The F32 one-shot over this rank alone, which produces `SUMMED` in
    /// the rows transport's replays.
    reduce: Option<Function>,
    /// The producer of `PARTIAL` for the one-shot transport.
    partials: Option<PartialProducer>,
}

/// The device producer of a rank's one-shot partials, the lab's PDL watch
/// probe: after its own dependency wait it fills `PARTIAL` with NaN, reads
/// the one-shot's epoch of tile 0, triggers its dependents, and holds its
/// write until the rank's one-shot has released its data flag of tile 0 to
/// itself (which it does only after reading its partials) or the hold
/// passes; then it copies the call's partials from `FRESH`. Its record says
/// whether the one-shot got ahead of it.
struct PartialProducer {
    _module: Module,
    function: Function,
    fresh: u64,
    output: u64,
    words: u64,
    epoch: u64,
    flag: u64,
    record: DeviceBuffer,
}

impl PartialProducer {
    /// The producer of `partial` from `fresh`, watching the one-shot data
    /// flag of tile 0 from `rank` (of `tiles`) in this rank's `flags`.
    fn new(
        session: &Session,
        [fresh, partial, epoch, flags]: [&DeviceBuffer; 4],
        rank: usize,
        tiles: usize,
    ) -> LabResult<Self> {
        let module = session.module(PROBE_CUBIN)?;
        let function = module.function(PROBE_SYMBOL).map_err(err)?;
        let bytes = partial.byte_len();
        let address = |buffer: &DeviceBuffer, offset: usize, len: usize| {
            buffer.device_address_at(offset, len).map_err(err)
        };
        let record = session.allocate(8)?;
        poison::fill_all(session, &record, &[0])?;
        Ok(Self {
            _module: module,
            function,
            fresh: address(fresh, 0, bytes)?,
            output: address(partial, 0, bytes)?,
            words: (bytes / 4) as u64,
            epoch: address(epoch, 0, 4)?,
            flag: address(flags, rank * tiles * 4, 4)?,
            record,
        })
    }

    /// One launch holding at most `hold_ns`, PDL after the stream's
    /// previous launch.
    fn launch(&self, session: &Session, hold_ns: u64) -> LabResult<()> {
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (1, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::U64(self.fresh),
                    KernelArgument::U64(self.output),
                    KernelArgument::U64(self.words),
                    KernelArgument::U64(self.epoch),
                    KernelArgument::U64(self.flag),
                    KernelArgument::U64(hold_ns),
                    KernelArgument::Buffer(&self.record),
                ],
            )
            .map_err(err)
    }

    /// Whether the last launch's one-shot got ahead of its write, and the
    /// nanoseconds it held.
    fn ahead(&self, session: &Session) -> LabResult<(bool, u32)> {
        let bytes = poison::read(session, &self.record)?;
        let word = |i: usize| u32::from_ne_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        Ok((word(0) == 1, word(1)))
    }
}

impl Rank {
    /// A rank's buffers, with one-shot state for `comm_ranks` ranks (none
    /// when 0); with one rank, also the one-shot over this rank alone, and
    /// with more, the partials' producer.
    fn new(session: &Session, inputs: &Inputs, rank: usize, comm_ranks: usize) -> LabResult<Self> {
        let rows = inputs.rows;
        let sinkhorn = session.module(SINKHORN_CUBIN)?;
        let fused = session.module(ROWS_CUBIN)?;
        let function = |module: &Module, symbol: &str| module.function(symbol).map_err(err);
        let (_, norm_symbol) = sinkhorn_epilogue::reference_norm();
        let expand = function(&sinkhorn, &sinkhorn_entry(Op::Expand).symbol)?;
        let mix = function(
            &sinkhorn,
            &sinkhorn_entry(Op::MixCollapse { probe: false }).symbol,
        )?;
        let norm = function(&fused, &norm_symbol)?;
        let fused_rows = function(&fused, &served_entry(Transport::Rows).symbol)?;
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
            let zeros = |bytes: usize| -> LabResult<DeviceBuffer> {
                let buffer = session.allocate(bytes)?;
                poison::fill_all(session, &buffer, &[0])?;
                Ok(buffer)
            };
            buffers.insert(FLAGS, zeros(2 * comm_ranks * tiles * 4)?);
            buffers.insert(EPOCH, zeros(tiles * 4)?);
        }
        let partials = if comm_ranks > 1 {
            buffers.insert(FRESH, session.upload_f32(&inputs.partials[rank])?);
            Some(PartialProducer::new(
                session,
                [
                    &buffers[&FRESH],
                    &buffers[&PARTIAL],
                    &buffers[&EPOCH],
                    &buffers[&FLAGS],
                ],
                rank,
                rows,
            )?)
        } else {
            None
        };
        let mut modules = vec![sinkhorn, fused];
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
            fused_rows,
            reduce,
            partials,
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

    /// The fused outputs of the live rows against the unfused sequence over
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
        same_outputs(label, &outputs)
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

    /// The producer of the one-shot's partials: `PARTIAL` from `FRESH`,
    /// holding at most `hold_ns`.
    fn partials(&self, session: &Session, hold_ns: u64) -> LabResult<()> {
        self.partials
            .as_ref()
            .ok_or("no partials producer")?
            .launch(session, hold_ns)
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
                status: None,
                status_row_offset: 0,
                x: COLLAPSED_U,
                weight: WEIGHT,
                residual: None,
                y: NORMED_U,
                live_rows: LIVE,
            })
            .map_err(err)?;
        self.launch(session, &self.norm, plan, &args)
    }

    /// The fused path's epilogue buffers.
    fn epilogue() -> EpilogueBuffers {
        EpilogueBuffers {
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
        }
    }

    /// The rows transport over `SUMMED`.
    fn fused_rows(&self, session: &Session, pdl: bool) -> LabResult<()> {
        let entry = served_entry(Transport::Rows);
        let mut plan = entry.plan(self.rows as u32, RANKS as u32).map_err(err)?;
        plan.pdl = pdl;
        let args = sinkhorn_epilogue::rows_arguments(SUMMED, Self::epilogue()).map_err(err)?;
        self.launch(session, &self.fused_rows, plan, &args)
    }

    fn bf16(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<u16>> {
        session.download_bf16(self.buffer(handle), self.rows * per_row)
    }

    /// The fused and unfused outputs of the live rows, in the order
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

/// The fused outputs equal the unfused ones bit for bit.
fn same_outputs(label: &str, [unfused, fused]: &[[Vec<u16>; 6]; 2]) -> LabResult<()> {
    for ((name, unfused), fused) in OUTPUTS.iter().zip(unfused).zip(fused) {
        exact(&format!("{label}: fused {name}"), fused, unfused)?;
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
    rank.fused_rows(session, false)?;
    session.stream.synchronize().map_err(err)?;
    let outputs = rank.outputs(session, rows)?;
    exact("round", &outputs[0][0], &y)?;
    same_outputs(&format!("{rows} rows"), &outputs)?;
    let [_, x1, post, _, collapsed, normed] = &outputs[1];
    oracle_checks(&inputs, &y, x1, collapsed, post, normed)?;
    Ok(json!({"rows": rows, "bitwise": true}))
}

/// The poisoned replays of one transport's fused launch on one rank: the
/// produced streams and the outputs past `live` keep their poison, post and
/// comb their image.
#[allow(clippy::single_range_in_vec_init)]
fn replay<'a>(
    session: &'a Session,
    rank: &'a Rank,
    inputs: &Inputs,
    live: usize,
) -> LabResult<Replay<'a>> {
    let prefix = |per_row: usize| WriteMap::of(DType::Bf16).prefix(live * per_row);
    Ok(Replay::new(session)
        // The producer's streams: poisoned before every producer, so a
        // consumer that reads them before the producer wrote them, or a
        // producer that wrote nothing, fails.
        .output("streams", rank.buffer(X), prefix(FLAT)?)
        .output("reduced", rank.buffer(REDUCED_F), prefix(H)?)
        .output("expanded", rank.buffer(X1_F), prefix(FLAT)?)
        .output("collapsed", rank.buffer(COLLAPSED_F), prefix(H)?)
        .output("normed", rank.buffer(NORMED_F), prefix(H)?)
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
        ))
}

fn rows_replay_bucket(session: &Session, rows: usize) -> LabResult<Value> {
    let inputs = Inputs::new(rows, 0xfeed + rows as u64);
    let rank = Rank::new(session, &inputs, 0, 1)?;
    // The single-rank one-shot reads the four ranks' sums as its partial.
    poison::write(session, rank.buffer(PARTIAL), 0, &words(&inputs.summed()))?;
    // The streams' producer, then the sums' producer (which publishes them
    // at the end of its protocol), then the epilogue, PDL on both
    // dependents when `pdl`.
    let chain = |pdl: bool| -> LabResult<()> {
        rank.producer(session)?;
        rank.reduce_one(session, pdl)?;
        rank.fused_rows(session, pdl)
    };
    let graph = session.capture(|_| chain(true))?;
    let replay_of = |live: usize| -> LabResult<Replay<'_>> {
        Ok(replay(session, &rank, &inputs, live)?.output(
            "sums",
            rank.buffer(SUMMED),
            WriteMap::of(DType::F32).prefix(rows * H)?,
        ))
    };
    let mut evidence = Vec::new();
    for live in poison::live_counts(rows as u32) {
        let live = live as usize;
        let label = format!("{rows} rows, live {live}");
        rank.reset(session, &inputs, live)?;
        let mut replay = replay_of(live)?;
        replay.eager(|| chain(false))?;
        // The eager bits against the unfused sequence, independently of the
        // replays' self-comparison.
        rank.against_unfused(session, &inputs, live, &label)?;
        replay.relaunch(|| chain(false))?;
        replay.graph(&graph)?;
        replay.pdl(
            || rank.producer(session),
            || {
                rank.reduce_one(session, true)?;
                rank.fused_rows(session, true)
            },
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        // The mutations at one partial live count (all rows of one row).
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            mutations = rows_mutations(session, &rank, &inputs, live, &replay_of, &label)?;
        }
        evidence.push(json!({"live": live, "replay": receipt, "mutations_failed": mutations}));
    }
    Ok(json!({"rows": rows, "live_counts": evidence}))
}

/// The checks catch a missing producer and missing state stores: with the
/// streams' producer or the sums' producer left out, the replay fails; with
/// the epilogue's post and comb stores undone after a correct launch, the
/// comparison with the unfused sequence fails (the replay's own check lets
/// read-modify-write state keep its image). Returns the mutations that
/// failed, which must be all of them.
fn rows_mutations<'a>(
    session: &'a Session,
    rank: &'a Rank,
    inputs: &Inputs,
    live: usize,
    replay_of: &dyn Fn(usize) -> LabResult<Replay<'a>>,
    label: &str,
) -> LabResult<Vec<&'static str>> {
    let mut failed = Vec::new();
    for (name, streams, sums) in [
        ("no streams producer", false, true),
        ("no sums producer", true, false),
    ] {
        rank.reset(session, inputs, live)?;
        let mut replay = replay_of(live)?;
        let outcome = replay.eager(|| {
            if streams {
                rank.producer(session)?;
            }
            if sums {
                rank.reduce_one(session, false)?;
            }
            rank.fused_rows(session, false)
        });
        session.stream.synchronize().map_err(err)?;
        if outcome.is_ok() {
            return Err(format!("{label}: the replay passed with {name}"));
        }
        failed.push(name);
    }
    // A correct launch, then the post and comb stores undone.
    rank.reset(session, inputs, live)?;
    rank.producer(session)?;
    rank.reduce_one(session, false)?;
    rank.fused_rows(session, false)?;
    session.stream.synchronize().map_err(err)?;
    rank.against_unfused(session, inputs, live, label)?;
    for (handle, image) in [(POST_F, &inputs.post), (COMB_F, &inputs.comb)] {
        poison::write(session, rank.buffer(handle), 0, &halves(image))?;
    }
    if rank.against_unfused(session, inputs, live, label).is_ok() {
        return Err(format!("{label}: undone post and comb stores passed"));
    }
    failed.push("post and comb stores undone");
    Ok(failed)
}

/// Every rank's state, loaded for collective calls.
struct Ranks<'a> {
    sessions: &'a [Session],
    ranks: Vec<Rank>,
    comm: CollectiveModules,
    fused: CollectiveModules,
    late_wait: CollectiveModules,
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

    /// The unfused F32 one-shot into `SUMMED` on every rank (or `only`).
    fn unfused_reduce(&self, only: Option<usize>) -> LabResult<()> {
        let query = CommQuery::AllReduce {
            ranks: RANKS as u32,
            elements: (self.rows * H) as u64,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let plan = comm::plan(&query, 1).map_err(err)?;
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
                    args: comm::arguments(self.call(rank, SUMMED)).map_err(err)?,
                    bindings: self.bindings(rank),
                })
            },
        )
    }

    /// The fused one-shot on every rank (or `only`), its sums into
    /// `SUMMED_F`.
    fn fused_reduce(&self, only: Option<usize>, pdl: bool) -> LabResult<()> {
        let symbol = served_entry(Transport::OneShot).symbol;
        self.one_shot(&self.fused, &symbol, only, pdl)
    }

    /// The late-wait mutant in place of the fused one-shot on every rank.
    fn late_wait_reduce(&self, pdl: bool) -> LabResult<()> {
        self.one_shot(&self.late_wait, &late_wait_symbol(), None, pdl)
    }

    fn one_shot(
        &self,
        modules: &CollectiveModules,
        symbol: &str,
        only: Option<usize>,
        pdl: bool,
    ) -> LabResult<()> {
        let entry = served_entry(Transport::OneShot);
        let mut plan = entry.plan(self.rows as u32, RANKS as u32).map_err(err)?;
        plan.pdl = pdl;
        launch::launch_collective(self.sessions, modules, symbol, plan, only, |rank| {
            Ok(RankCall {
                args: sinkhorn_epilogue::one_shot_arguments(
                    self.call(rank, SUMMED_F),
                    Rank::epilogue(),
                )
                .map_err(err)?,
                bindings: self.bindings(rank),
            })
        })
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

    /// Poisons every rank's `PARTIAL`, so the fused call sums only what its
    /// producer wrote.
    fn poison_partials(&self) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            poison::fill_all(session, self.ranks[rank].buffer(PARTIAL), &[poison::POISON])?;
        }
        Ok(())
    }

    /// The producers the fused one-shot reads on every rank: the streams'
    /// and the partials' (each when asked, the latter holding at most
    /// `hold_ns`), the partials' last, so the one-shot's dependency wait is
    /// on it.
    fn producers(&self, streams: bool, partials: bool, hold_ns: u64) -> LabResult<()> {
        for (rank, session) in self.sessions.iter().enumerate() {
            if streams {
                self.ranks[rank].producer(session)?;
            }
            if partials {
                self.ranks[rank].partials(session, hold_ns)?;
            }
        }
        Ok(())
    }

    /// Each rank's partials' producer's record of its last launch: whether
    /// the one-shot got ahead of its write, and how long it held.
    fn ahead(&self) -> LabResult<Vec<(bool, u32)>> {
        self.sessions
            .iter()
            .zip(&self.ranks)
            .map(|(session, rank)| {
                rank.partials
                    .as_ref()
                    .ok_or("no partials producer")?
                    .ahead(session)
            })
            .collect()
    }

    /// Every rank's fused sums equal the host's rank-order sums.
    fn fused_sums(&self, inputs: &Inputs, label: &str) -> LabResult<()> {
        let summed = bits(&inputs.summed());
        for (rank, session) in self.sessions.iter().enumerate() {
            let sums = session.download_f32(self.ranks[rank].buffer(SUMMED_F), self.rows * H)?;
            exact(
                &format!("{label}: rank {rank} fused sums"),
                &bits(&sums),
                &summed,
            )?;
        }
        Ok(())
    }

    /// One unfused call on every rank: the one-shot, every rank's sums
    /// against the host's, the round on the host, then the row launches.
    fn unfused(&self, inputs: &Inputs) -> LabResult<()> {
        poison::synchronize(self.sessions)?;
        self.unfused_reduce(None)?;
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

    /// Every rank's fused outputs equal its unfused ones and rank 0's, and
    /// its fused sums the host's.
    fn compare(&self, inputs: &Inputs, live: usize, label: &str) -> LabResult<()> {
        self.fused_sums(inputs, label)?;
        let mut first = None;
        for (rank, session) in self.sessions.iter().enumerate() {
            let outputs = self.ranks[rank].outputs(session, live)?;
            same_outputs(&format!("{label}: rank {rank}"), &outputs)?;
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

/// One row count of the one-shot transport on four ranks: the fused and
/// unfused one-shots alternate on one state, eagerly and through captured
/// graphs, at every live count, the fused one after the device producers of
/// its streams and partials; then the poisoned replays and the mutations.
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
        fused: CollectiveModules::load(sessions, FUSED_COMM_CUBIN)?,
        late_wait: CollectiveModules::load(sessions, LATE_WAIT_CUBIN)?,
        rows,
    };
    // Each rank's graph holds the streams' producer, the partials' producer
    // and the epilogue, PDL to the last two.
    let graphs = kernel_lab::topology::RankReplay::new(sessions, |rank, session| {
        ranks.ranks[rank].producer(session)?;
        ranks.ranks[rank].partials(session, PRODUCER_TAIL_NS)?;
        ranks.fused_reduce(Some(rank), true)
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
                ranks.producers(true, true, PRODUCER_TAIL_NS)?;
                ranks.fused_reduce(None, false)?;
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
                    ranks.producers(true, true, PRODUCER_TAIL_NS)?;
                    ranks.fused_reduce(None, pdl)?;
                }
            }
            poison::synchronize(sessions)?;
            for (rank, replay) in replays.iter_mut().enumerate() {
                replay
                    .finish(phase, pdl)
                    .map_err(|e| format!("{rows} rows, live {live}, rank {rank}: {e}"))?;
            }
            // The served one-shot never reads its partials before their
            // producer wrote them.
            if let Some(rank) = ranks.ahead()?.iter().position(|&(ahead, _)| ahead) {
                return Err(format!(
                    "{rows} rows, live {live}, {}: rank {rank}'s one-shot published before its \
                     partials' producer wrote them",
                    phase.name()
                ));
            }
            if phase == Phase::Eager {
                // The bits every replay must reproduce, against the host's.
                ranks.fused_sums(&inputs, &format!("{rows} rows, live {live}, eager"))?;
            }
        }
        let mut receipt = BTreeMap::new();
        replays[0].record(&mut receipt);
        // The mutations at one partial live count (all rows of one row).
        let mut mutations = Vec::new();
        if live == (rows / 2).max(1) {
            mutations = one_shot_mutations(&ranks, &inputs, &replays, live)?;
        }
        evidence.push(json!({"live": live, "replay": receipt, "mutations": mutations}));
    }
    Ok(json!({"rows": rows, "calls": calls, "live_counts": evidence}))
}

/// The one-shot's checks catch a missing producer and an early partial
/// read: with the streams' or the partials' producer left out, the output
/// left unproduced differs from the eager bits on some rank. With the
/// one-shot's dependency wait moved after its partial reads (the late-wait
/// mutant, PDL after the partials' producer as served), the producer holds
/// its write until the mutant has published or `LATE_WAIT_HOLD_NS` passed,
/// and records which: every rank whose mutant got ahead must sum
/// differently from the eager bits (which equal the host's), with the
/// producers' outputs equal to the eager bits on every rank. If no rank got
/// ahead (PDL did not overlap them) and every sum equals the eager bits,
/// the mutation is inconclusive, neither caught nor missed. Readback errors
/// propagate. Returns each mutation's outcome.
fn one_shot_mutations(
    ranks: &Ranks<'_>,
    inputs: &Inputs,
    replays: &[Replay<'_>],
    live: usize,
) -> LabResult<Vec<Value>> {
    let rows = ranks.rows;
    let mut outcomes = Vec::new();
    for (name, streams, partials, late_wait, corrupted) in [
        ("no streams producer", false, true, false, "streams"),
        ("no partials producer", true, false, false, "partials"),
        (
            "dependency wait after the partial reads",
            true,
            true,
            true,
            "sums",
        ),
    ] {
        let label = format!("{rows} rows, live {live}, {name}");
        poison::synchronize(ranks.sessions)?;
        for (rank, session) in ranks.sessions.iter().enumerate() {
            ranks.ranks[rank].reset(session, inputs, live)?;
        }
        for replay in replays {
            replay.prepare(Phase::Pdl)?;
        }
        let hold = if late_wait {
            LATE_WAIT_HOLD_NS
        } else {
            PRODUCER_TAIL_NS
        };
        ranks.producers(streams, partials, hold)?;
        if late_wait {
            ranks.late_wait_reduce(true)?;
        } else {
            ranks.fused_reduce(None, true)?;
        }
        poison::synchronize(ranks.sessions)?;
        // Each rank's outputs that differ from its eager bits.
        let mut differing = Vec::new();
        for (rank, (session, replay)) in ranks.sessions.iter().zip(replays).enumerate() {
            for (output, handle) in [("streams", X), ("partials", PARTIAL), ("sums", SUMMED_F)] {
                let actual = poison::read(session, ranks.ranks[rank].buffer(handle))?;
                let eager = replay
                    .expected(output)
                    .ok_or_else(|| format!("{label}: no eager {output}"))?;
                if actual != eager {
                    differing.push((rank, output));
                }
            }
        }
        let ranks_corrupted = differing
            .iter()
            .filter(|(_, output)| *output == corrupted)
            .count();
        if !late_wait {
            if ranks_corrupted == 0 {
                return Err(format!(
                    "{label}: no rank's {corrupted} differ from the eager bits (differing: {differing:?})"
                ));
            }
            outcomes.push(
                json!({"mutation": name, "outcome": "caught", "output": corrupted,
                                 "ranks_differing": ranks_corrupted}),
            );
            continue;
        }
        if differing.iter().any(|(_, output)| *output != corrupted) {
            return Err(format!(
                "{label}: the producers did not complete as in the eager run (differing: {differing:?})"
            ));
        }
        let ahead = ranks.ahead()?;
        let ranks_ahead = (0..ahead.len())
            .filter(|&rank| ahead[rank].0)
            .collect::<Vec<_>>();
        if let Some(rank) = ranks_ahead
            .iter()
            .find(|&&rank| !differing.contains(&(rank, corrupted)))
        {
            return Err(format!(
                "{label}: rank {rank}'s mutant read its partials before their producer wrote \
                 them, yet its sums equal the eager bits (ranks ahead: {ranks_ahead:?})"
            ));
        }
        let outcome = if ranks_corrupted > 0 {
            "caught"
        } else {
            eprintln!(
                "{label}: inconclusive: no rank's mutant got ahead of its partials' producer \
                 within {LATE_WAIT_HOLD_NS} ns (no PDL overlap)"
            );
            "inconclusive"
        };
        outcomes.push(json!({
            "mutation": name,
            "outcome": outcome,
            "output": corrupted,
            "ranks_differing": ranks_corrupted,
            "ranks_ahead": ranks_ahead,
            "held_ns": ahead.iter().map(|&(_, held)| held).collect::<Vec<_>>(),
        }));
    }
    Ok(outcomes)
}

#[test]
fn sinkhorn_comm_epilogue_cpu_contract() {
    for (cubin, symbol) in [
        (FUSED_COMM_CUBIN, served_entry(Transport::OneShot).symbol),
        (ROWS_CUBIN, served_entry(Transport::Rows).symbol),
        (ROWS_CUBIN, sinkhorn_epilogue::reference_norm().1),
        (SINKHORN_CUBIN, sinkhorn_entry(Op::Expand).symbol),
        (LATE_WAIT_CUBIN, late_wait_symbol()),
        (PROBE_CUBIN, PROBE_SYMBOL.to_string()),
    ] {
        assert!(
            kernel_lab::cubin_entries(cubin)
                .unwrap()
                .contains(&symbol.as_str()),
            "{cubin} lacks {symbol}"
        );
        kernel_lab::cubin(cubin).unwrap();
    }
    assert!(kernel_lab::comm_cubin(FUSED_COMM_CUBIN).unwrap());
    assert!(kernel_lab::comm_cubin(LATE_WAIT_CUBIN).unwrap());
    assert!(!kernel_lab::comm_cubin(ROWS_CUBIN).unwrap());
    // The one-shot transport serves the buckets the F32 one-shot takes at
    // TP4, and the rows transport the rest.
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
    for transport in [Transport::OneShot, Transport::Rows] {
        for rows in ROWS_BUCKETS {
            let plan = served_entry(transport)
                .plan(rows as u32, RANKS as u32)
                .unwrap();
            assert_eq!(plan.grid, [rows as u32, 1, 1]);
        }
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
fn the_rows_epilogue_matches_the_unfused_launches_bitwise() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let buckets = ROWS_BUCKETS
            .iter()
            .map(|&rows| rows_bucket(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_epilogue_rows"));
        receipt.insert("buckets".into(), json!(buckets));
        session.receipt("sinkhorn_comm_epilogue_rows", receipt)?;
        Ok(())
    })
    .expect("the rows epilogue against the unfused launches");
}

#[test]
fn the_rows_epilogue_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let buckets = ROWS_BUCKETS
            .iter()
            .map(|&rows| rows_replay_bucket(session, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_epilogue_rows_poison"));
        receipt.insert("buckets".into(), json!(buckets));
        session.receipt("sinkhorn_comm_epilogue_rows_poison", receipt)?;
        Ok(())
    })
    .expect("the rows epilogue's poisoned replays");
}

#[test]
fn the_one_shot_epilogue_matches_the_unfused_launches_on_four_ranks() {
    kernel_lab::gpu_test_multi(2 << 30, |sessions| {
        if sessions.len() != RANKS {
            return kernel_lab::skip("the one-shot epilogue runs on four GPUs (TP4)");
        }
        let buckets = ONE_SHOT_BUCKETS
            .iter()
            .map(|&rows| one_shot_bucket(sessions, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_comm_epilogue_one_shot"));
        receipt.insert("buckets".into(), json!(buckets));
        sessions[0].receipt("sinkhorn_comm_epilogue_one_shot", receipt)?;
        Ok(())
    })
    .expect("the one-shot epilogue on four ranks");
}
