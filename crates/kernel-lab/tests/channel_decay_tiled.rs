//! The per-channel-decay delta rule's tiled prefill stages
//! (`kernels_sm120::linear::channel_decay::tiled_catalog`, cubin
//! `channel_decay_tiled_h16x128_sm120`) against the staged stages of
//! `channel_decay_h16x128_sm120` bit for bit: the prepare's whole workspace,
//! the scan's output and the whole state pool (slots and anchors), at 64,
//! 2,048 and 8,192 rows, mixed phases and lengths, every live sequence
//! count and the served bucket of 32 sequences; poisoned eager, relaunch,
//! graph and PDL launches behind the convolution over the same batches and
//! live counts, workspace included; and the lab-only mutants
//! (`TILED_MUTANTS`), each of which must change an observed output. The
//! staged entries' own oracle, invariance and replay tests are
//! `tests/channel_decay.rs`.
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --locked --offline --test channel_decay_tiled`; the latency
//! table also needs `CHOPINFER_KERNEL_BENCH=1`. Without a GPU every GPU test
//! skips (and fails under `CHOPINFER_KERNEL_REQUIRE=1`).

use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::linear::channel_decay::{
    self, Entry, PrefillSchedule, SERVED, SERVED_PROFILE, TILED_MUTANTS,
};
use kernels_sm120::linear::{Bucket, Stage};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

const H: usize = SERVED.heads as usize;
const D: usize = SERVED.head_dim as usize;
const WIDTH: usize = H * D;
const CHANNELS: usize = 3 * WIDTH;
const HISTORY: usize = SERVED.kernel as usize - 1;
const CONV_ENTRY: usize = HISTORY * CHANNELS;
const STATE: usize = H * D * D;
/// Row strides of the inputs, each padded past its payload.
const INPUT_STRIDE: usize = CHANNELS + 64;
const F_STRIDE: usize = WIDTH + 64;
const B_STRIDE: usize = H + 8;
/// Sequences of a served prefill bucket: a package plans every prefill
/// chunk for its 32 items, so most scan CTAs find no live sequence.
const SERVED_SEQUENCES: usize = 32;

// ------------------------------------------------------------ helpers ----

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    fn bf16(&mut self, len: usize, mean: f64, scale: f64) -> Vec<u16> {
        (0..len)
            .map(|_| f32_to_bf16((mean + self.normal() * scale) as f32))
            .collect()
    }
    fn f32(&mut self, len: usize, scale: f64) -> Vec<f32> {
        (0..len).map(|_| (self.normal() * scale) as f32).collect()
    }
}

fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn cu(lengths: &[usize]) -> Vec<i32> {
    let mut cu = vec![0_i32];
    for &len in lengths {
        cu.push(cu.last().unwrap() + len as i32);
    }
    cu
}
/// The local row of a call's prefix anchor (`channel_decay_common.cuh`).
fn anchor_row(position: usize, rows: usize) -> Option<usize> {
    let tail = (position + rows) % 64;
    (tail != 0 && rows > tail).then(|| rows - tail)
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}
fn zeros(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    upload(session, &vec![0; bytes])
}

/// The first differing byte of two images, or `None`.
fn first_difference(name: &str, actual: &[u8], expected: &[u8]) -> Option<String> {
    if actual.len() != expected.len() {
        return Some(format!(
            "{name}: {} versus {} bytes",
            actual.len(),
            expected.len()
        ));
    }
    actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .map(|at| {
            format!(
                "{name}: byte {at} is {:#04x}, the staged stages wrote {:#04x}",
                actual[at], expected[at]
            )
        })
}

/// The staged catalog entry `name` (`conv_prefill`, `delta_prefill_prepare`,
/// `delta_prefill_scan`).
fn staged(name: &str) -> Entry {
    let symbol = format!("chop_channel_decay_{}_{name}_v1", SERVED_PROFILE.tag());
    channel_decay::catalog()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing entry {symbol}"))
}

/// The tiled entry of `stage`, under `mutant`'s symbol when given.
fn tiled(stage: Stage, mutant: Option<&str>) -> Entry {
    let mut entry = channel_decay::tiled_catalog()
        .into_iter()
        .find(|entry| entry.stage == stage)
        .expect("a tiled entry per stage");
    if let Some(mutant) = mutant {
        entry.symbol = channel_decay::tiled_mutant_symbol(&entry.symbol, mutant);
    }
    entry
}

fn staged_module(session: &Session) -> LabResult<Module> {
    session.module(&channel_decay::cubin_specs().remove(0).name)
}
fn tiled_module_name(mutant: Option<&str>) -> String {
    match mutant {
        None => format!("channel_decay_tiled_{}_sm120", SERVED_PROFILE.tag()),
        Some(mutant) => format!(
            "channel_decay_tiled_{mutant}_{}_sm120",
            SERVED_PROFILE.tag()
        ),
    }
}

/// Named device buffers and scalars, bound to entry parameters by name.
#[derive(Default)]
struct Binds<'a> {
    buffers: BTreeMap<&'static str, &'a DeviceBuffer>,
    scalars: BTreeMap<&'static str, u32>,
}

impl<'a> Binds<'a> {
    fn buffer(mut self, name: &'static str, buffer: &'a DeviceBuffer) -> Self {
        self.buffers.insert(name, buffer);
        self
    }
    fn scalar(mut self, name: &'static str, value: u32) -> Self {
        self.scalars.insert(name, value);
        self
    }
}

/// An entry resolved in a loaded cubin, planned for `bucket`.
struct Kernel {
    entry: Entry,
    function: Function,
    bucket: Bucket,
}

impl Kernel {
    fn load(module: &Module, entry: Entry, bucket: Bucket) -> LabResult<Self> {
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let shared = entry.dynamic_shared_bytes();
        if shared > 0 {
            function
                .set_max_dynamic_shared_memory(shared)
                .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            entry,
            function,
            bucket,
        })
    }

    fn launch(
        &self,
        session: &Session,
        stream: &Stream,
        binds: &Binds<'_>,
        live_rows: usize,
        live_sequences: usize,
        pdl: bool,
    ) -> LabResult<()> {
        let names = binds.buffers.keys().copied().collect::<Vec<_>>();
        let values = self
            .entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => names
                    .iter()
                    .position(|n| n == name)
                    .map(|i| Argument::DeviceBuffer(i as u64 + 1))
                    .ok_or_else(|| format!("missing buffer {name}")),
                EntryParam::U32 { name } => binds
                    .scalars
                    .get(name.as_str())
                    .copied()
                    .map(Argument::U32)
                    .ok_or_else(|| format!("missing scalar {name}")),
                other => Err(format!("unexpected parameter {other:?}")),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let args = self
            .entry
            .pack_arguments(values)
            .map_err(|e| e.to_string())?;
        let mut plan = self
            .entry
            .plan(self.bucket, live_rows as u32, live_sequences as u32)
            .map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let bindings = Bindings {
            buffers: binds
                .buffers
                .values()
                .enumerate()
                .map(|(i, &b)| (i as u64 + 1, b))
                .collect(),
        };
        launch::launch(
            &session.context,
            &self.function,
            stream,
            plan,
            &args,
            &bindings,
        )?;
        kernel_lab::launched(&self.entry.symbol);
        Ok(())
    }
}

// --------------------------------------------------------------- cases ----

/// A prefill batch: every sequence's rows, absolute start position and
/// initial flag, planned at `bucket_sequences` (at least the sequences).
/// Slots are `2 s + 1` and anchors `2 S + s` for S sequences, so unused
/// entries sit between and after the live ones.
struct Batch {
    name: String,
    lengths: Vec<usize>,
    positions: Vec<usize>,
    initial: Vec<bool>,
    bucket_sequences: usize,
    seed: u64,
    /// Every gate input at the bound (`f` 30, `dt_bias` 0, `a` 10).
    at_bound: bool,
}

impl Batch {
    fn single(name: &str, rows: usize, position: usize, initial: bool, seed: u64) -> Self {
        Self {
            name: name.into(),
            lengths: vec![rows],
            positions: vec![position],
            initial: vec![initial],
            bucket_sequences: SERVED_SEQUENCES,
            seed,
            at_bound: false,
        }
    }
    fn sequences(&self) -> usize {
        self.lengths.len()
    }
    fn rows(&self) -> usize {
        self.lengths.iter().sum()
    }
    fn live_rows(&self, live: usize) -> usize {
        self.lengths[..live].iter().sum()
    }
    fn bucket(&self) -> Bucket {
        Bucket {
            rows: self.rows() as u32,
            sequences: self.bucket_sequences as u32,
        }
    }
    fn slots(&self) -> Vec<i32> {
        (0..self.sequences()).map(|s| (2 * s + 1) as i32).collect()
    }
    fn anchors(&self) -> Vec<i32> {
        (0..self.sequences())
            .map(|s| (2 * self.sequences() + s) as i32)
            .collect()
    }
    fn entries(&self) -> usize {
        3 * self.sequences() + 1
    }
    /// Pool entries the first `live` sequences only write (their anchors,
    /// when the call ends off a multiple of 64 with one inside), and the
    /// slots they read and rewrite.
    fn writes(&self, live: usize) -> (Vec<i32>, Vec<i32>) {
        let anchors = self.anchors();
        (
            (0..live)
                .filter(|&s| anchor_row(self.positions[s], self.lengths[s]).is_some())
                .map(|s| anchors[s])
                .collect(),
            self.slots()[..live].to_vec(),
        )
    }
    /// Workspace bytes the prefill of the first `live` sequences writes.
    fn live_workspace(&self, live: usize) -> usize {
        let chunks = (0..live)
            .map(|s| (self.positions[s] % 64 + self.lengths[s]).div_ceil(64))
            .sum::<usize>();
        chunks * H * channel_decay::WORK_FLOATS as usize * 4
    }
}

fn entry_ranges(ids: &[i32], bytes: usize) -> Vec<std::ops::Range<usize>> {
    ids.iter()
        .map(|&id| id as usize * bytes..(id as usize + 1) * bytes)
        .collect()
}

/// Host operands of a batch: the convolution's input, weights and pool, the
/// delta rule's gate inputs and pool.
struct Operands {
    input: Vec<u16>,
    weights: [Vec<f32>; 3],
    conv_pool: Vec<u16>,
    f: Vec<u16>,
    b: Vec<u16>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    delta_pool: Vec<f32>,
}

impl Operands {
    fn new(batch: &Batch) -> Self {
        let (rows, entries) = (batch.rows(), batch.entries());
        let mut rng = Rng(batch.seed);
        let weight = |rng: &mut Rng| {
            rng.bf16(WIDTH * 4, 0.0, 0.4)
                .into_iter()
                .map(kernel_core::codec::bf16_to_f32)
                .collect::<Vec<_>>()
        };
        let mut operands = Self {
            input: rng.bf16(rows * INPUT_STRIDE, 0.0, 1.0),
            weights: [weight(&mut rng), weight(&mut rng), weight(&mut rng)],
            conv_pool: rng.bf16(entries * CONV_ENTRY, 0.0, 1.0),
            // Mostly slow decays with some near the bound.
            f: rng.bf16(rows * F_STRIDE, -2.0, 1.5),
            b: rng.bf16(rows * B_STRIDE, 0.0, 2.0),
            a_log: (0..H)
                .map(|_| (1.0 + 15.0 * rng.uniform()).ln() as f32)
                .collect(),
            dt_bias: rng.f32(WIDTH, 1.0),
            delta_pool: rng.f32(entries * STATE, 0.5),
        };
        if batch.at_bound {
            // a = 10 and x = 30: sigmoid(300) is 1, so every g is the bound
            // and G reaches -320 per chunk.
            operands.f.iter_mut().for_each(|f| *f = f32_to_bf16(30.0));
            operands.dt_bias.iter_mut().for_each(|b| *b = 0.0);
            operands.a_log.iter_mut().for_each(|a| *a = 10_f32.ln());
        }
        operands
    }
}

/// Device buffers of a batch, named by the entry parameters that read them.
struct Device {
    input: DeviceBuffer,
    weight_q: DeviceBuffer,
    weight_k: DeviceBuffer,
    weight_v: DeviceBuffer,
    conv_state: DeviceBuffer,
    f: DeviceBuffer,
    b: DeviceBuffer,
    a_log: DeviceBuffer,
    dt_bias: DeviceBuffer,
    delta_state: DeviceBuffer,
    /// Convolution output `[rows, CHANNELS]`, the delta rule's `qkv`.
    conv_out: DeviceBuffer,
    /// Delta-rule output `[rows, WIDTH]`.
    core: DeviceBuffer,
    cu: DeviceBuffer,
    start_positions: DeviceBuffer,
    slot_ids: DeviceBuffer,
    initial: DeviceBuffer,
    anchor_ids: DeviceBuffer,
    live_sequences: DeviceBuffer,
    workspace: DeviceBuffer,
}

impl Device {
    fn new(session: &Session, batch: &Batch, operands: &Operands) -> LabResult<Self> {
        let rows = batch.rows();
        let workspace = staged("delta_prefill_prepare").workspace_bytes(batch.bucket()) as usize;
        let positions = batch
            .positions
            .iter()
            .map(|&p| p as i32)
            .collect::<Vec<_>>();
        let initial = batch
            .initial
            .iter()
            .map(|&x| i32::from(x))
            .collect::<Vec<_>>();
        Ok(Self {
            input: upload(session, &bytes_u16(&operands.input))?,
            weight_q: upload(session, &bytes_f32(&operands.weights[0]))?,
            weight_k: upload(session, &bytes_f32(&operands.weights[1]))?,
            weight_v: upload(session, &bytes_f32(&operands.weights[2]))?,
            conv_state: upload(session, &bytes_u16(&operands.conv_pool))?,
            f: upload(session, &bytes_u16(&operands.f))?,
            b: upload(session, &bytes_u16(&operands.b))?,
            a_log: upload(session, &bytes_f32(&operands.a_log))?,
            dt_bias: upload(session, &bytes_f32(&operands.dt_bias))?,
            delta_state: upload(session, &bytes_f32(&operands.delta_pool))?,
            conv_out: zeros(session, rows * CHANNELS * 2)?,
            core: zeros(session, rows * WIDTH * 2)?,
            cu: upload(session, &bytes_i32(&cu(&batch.lengths)))?,
            start_positions: upload(session, &bytes_i32(&positions))?,
            slot_ids: upload(session, &bytes_i32(&batch.slots()))?,
            initial: upload(session, &bytes_i32(&initial))?,
            anchor_ids: upload(session, &bytes_i32(&batch.anchors()))?,
            live_sequences: upload(session, &(batch.sequences() as u32).to_ne_bytes())?,
            workspace: zeros(session, workspace)?,
        })
    }

    fn metadata<'a>(&'a self, binds: Binds<'a>) -> Binds<'a> {
        binds
            .buffer("cu_seqlens", &self.cu)
            .buffer("start_positions", &self.start_positions)
            .buffer("slot_ids", &self.slot_ids)
            .buffer("initial", &self.initial)
            .buffer("anchor_ids", &self.anchor_ids)
            .buffer("live_sequences", &self.live_sequences)
            .buffer("workspace", &self.workspace)
    }

    fn conv(&self) -> Binds<'_> {
        self.metadata(
            Binds::default()
                .buffer("input", &self.input)
                .buffer("weight_q", &self.weight_q)
                .buffer("weight_k", &self.weight_k)
                .buffer("weight_v", &self.weight_v)
                .buffer("state", &self.conv_state)
                .buffer("output", &self.conv_out)
                .scalar("input_stride", INPUT_STRIDE as u32),
        )
    }

    fn delta(&self) -> Binds<'_> {
        self.metadata(
            Binds::default()
                .buffer("qkv", &self.conv_out)
                .buffer("f", &self.f)
                .buffer("b", &self.b)
                .buffer("a_log", &self.a_log)
                .buffer("dt_bias", &self.dt_bias)
                .buffer("state", &self.delta_state)
                .buffer("output", &self.core)
                .scalar("qkv_stride", CHANNELS as u32)
                .scalar("f_stride", F_STRIDE as u32)
                .scalar("b_stride", B_STRIDE as u32),
        )
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(
            session,
            &self.live_sequences,
            0,
            &(live as u32).to_ne_bytes(),
        )
    }
}

/// The convolution and both schedules' delta-rule stages of a batch.
struct Stages {
    conv: Kernel,
    prepare: [Kernel; 2],
    scan: [Kernel; 2],
}

const STAGED: usize = 0;
const TILED: usize = 1;

impl Stages {
    fn load(session: &Session, batch: &Batch, mutant: Option<&str>) -> LabResult<Self> {
        let staged_module = staged_module(session)?;
        let tiled_module = session.module(&tiled_module_name(mutant))?;
        let bucket = batch.bucket();
        Ok(Self {
            conv: Kernel::load(&staged_module, staged("conv_prefill"), bucket)?,
            prepare: [
                Kernel::load(&staged_module, staged("delta_prefill_prepare"), bucket)?,
                Kernel::load(&tiled_module, tiled(Stage::Prepare, mutant), bucket)?,
            ],
            scan: [
                Kernel::load(&staged_module, staged("delta_prefill_scan"), bucket)?,
                Kernel::load(&tiled_module, tiled(Stage::Scan, mutant), bucket)?,
            ],
        })
    }

    fn conv(&self, s: &Session, d: &Device, rows: usize, live: usize, pdl: bool) -> LabResult<()> {
        self.conv.launch(s, &s.stream, &d.conv(), rows, live, pdl)
    }

    /// The prepare of `prepare` then the scan of `scan` (`STAGED` or
    /// `TILED`) on `stream`.
    #[allow(clippy::too_many_arguments)]
    fn delta(
        &self,
        s: &Session,
        stream: &Stream,
        d: &Device,
        prepare: usize,
        scan: usize,
        rows: usize,
        live: usize,
        pdl: bool,
    ) -> LabResult<()> {
        self.prepare[prepare].launch(s, stream, &d.delta(), rows, live, pdl)?;
        self.scan[scan].launch(s, stream, &d.delta(), rows, live, pdl)
    }
}

/// The images a run leaves: the whole workspace, delta output and delta
/// pool.
struct Images {
    workspace: Vec<u8>,
    core: Vec<u8>,
    pool: Vec<u8>,
}

impl Images {
    fn read(session: &Session, device: &Device) -> LabResult<Self> {
        Ok(Self {
            workspace: poison::read(session, &device.workspace)?,
            core: poison::read(session, &device.core)?,
            pool: poison::read(session, &device.delta_state)?,
        })
    }

    /// Every difference from `expected`; with `workspace` false the
    /// workspace is not compared (a scan alone does not write it).
    fn differences(&self, expected: &Images, workspace: bool) -> Vec<String> {
        let mut found = Vec::new();
        if workspace {
            found.extend(first_difference(
                "workspace",
                &self.workspace,
                &expected.workspace,
            ));
        }
        found.extend(first_difference("output", &self.core, &expected.core));
        found.extend(first_difference("state pool", &self.pool, &expected.pool));
        found
    }
}

/// Restores the delta pool, poisons the workspace and the delta output.
fn reset_delta(session: &Session, device: &Device, operands: &Operands) -> LabResult<()> {
    poison::write(
        session,
        &device.delta_state,
        0,
        &bytes_f32(&operands.delta_pool),
    )?;
    poison::fill_all(session, &device.workspace, &[POISON])?;
    poison::fill_all(session, &device.core, &[POISON])
}

/// The staged chain's images at `live` sequences after the convolution
/// (whose output stays in `conv_out`), with a poisoned workspace and
/// output.
fn staged_images(
    session: &Session,
    stages: &Stages,
    device: &Device,
    operands: &Operands,
    live_rows: usize,
    live: usize,
) -> LabResult<Images> {
    reset_delta(session, device, operands)?;
    stages.delta(
        session,
        &session.stream,
        device,
        STAGED,
        STAGED,
        live_rows,
        live,
        false,
    )?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Images::read(session, device)
}

/// Runs the convolution once at `live` sequences from a restored pool, so
/// `conv_out` holds the delta rule's input.
fn run_conv(
    session: &Session,
    stages: &Stages,
    device: &Device,
    operands: &Operands,
    live_rows: usize,
    live: usize,
) -> LabResult<()> {
    poison::write(
        session,
        &device.conv_state,
        0,
        &bytes_u16(&operands.conv_pool),
    )?;
    poison::fill_all(session, &device.conv_out, &[POISON])?;
    stages.conv(session, device, live_rows, live, false)?;
    session.stream.synchronize().map_err(|e| e.to_string())
}

/// The batches of the bitwise comparison.
fn batches() -> Vec<Batch> {
    let mut list = vec![
        Batch {
            bucket_sequences: 1,
            ..Batch::single("rows64_exact_bucket", 64, 0, true, 0x6401)
        },
        Batch::single("rows64", 64, 0, true, 0x6402),
        Batch::single("rows2048", 2048, 0, true, 0x2048),
        // Starts off a multiple of 64 and ends off one: a short first chunk
        // and an anchor.
        Batch::single("rows2048_phase37", 2048, 4096 + 37, true, 0x2049),
        Batch::single("rows8192", 8192, 0, false, 0x8192),
        Batch::single("rows8192_phase5", 8192, 5, true, 0x8193),
        Batch {
            at_bound: true,
            ..Batch::single("rows256_gate_bound", 256, 0, true, 0xb0b0)
        },
    ];
    // Ends off a multiple of 64 with one inside (anchors written), exactly
    // on one, and short calls at phases 0, 5, 36 and 27.
    let mixed = |name: &str, bucket_sequences: usize| Batch {
        name: name.into(),
        lengths: vec![65, 64, 1, 130, 37, 100],
        positions: vec![0, 0, 64, 5, 100, 27],
        initial: vec![true, false, true, true, false, true],
        bucket_sequences,
        seed: 0x6006,
        at_bound: false,
    };
    list.push(mixed("mixed", 6));
    list.push(mixed("mixed_served_bucket", SERVED_SEQUENCES));
    // A full served bucket of 32 sequences at random lengths and phases.
    let mut rng = Rng(0x3232);
    let lengths = (0..SERVED_SEQUENCES)
        .map(|_| 1 + (rng.next() % 200) as usize)
        .collect::<Vec<_>>();
    let positions = (0..SERVED_SEQUENCES)
        .map(|_| (rng.next() % 1000) as usize)
        .collect();
    let initial = (0..SERVED_SEQUENCES).map(|_| rng.next() % 3 != 0).collect();
    list.push(Batch {
        name: "many_sequences".into(),
        lengths,
        positions,
        initial,
        bucket_sequences: SERVED_SEQUENCES,
        seed: 0x3233,
        at_bound: false,
    });
    list
}

/// Every live sequence count of `batch`: `poison::live_counts`' order
/// (capacity, 0, 1, the middle, capacity - 1), so no launch can rely on the
/// previous count, then every other count ascending.
fn every_live_count(batch: &Batch) -> Vec<usize> {
    let capacity = batch.sequences();
    let mut counts = poison::live_counts(capacity as u32)
        .into_iter()
        .map(|live| live as usize)
        .collect::<Vec<_>>();
    for live in 0..=capacity {
        if !counts.contains(&live) {
            counts.push(live);
        }
    }
    counts
}

/// The GPU matrix: every batch at every live count, for the bitwise
/// comparison and the poisoned replays alike.
fn matrix() -> Vec<(Batch, Vec<usize>)> {
    batches()
        .into_iter()
        .map(|batch| {
            let counts = every_live_count(&batch);
            (batch, counts)
        })
        .collect()
}

/// (batch, live count) pairs of [`matrix`]; the lab job's summary requires
/// this many bitwise and replay receipts.
const MATRIX_PAIRS: usize = 61;

/// Every pairing of the schedules against the staged pair at one live count:
/// the tiled prepare's whole workspace, the tiled scan over the staged
/// workspace, and the tiled pair, each from poisoned buffers.
fn compare_live(
    session: &Session,
    stages: &Stages,
    device: &Device,
    operands: &Operands,
    live_rows: usize,
    live: usize,
) -> LabResult<Vec<String>> {
    let expected = staged_images(session, stages, device, operands, live_rows, live)?;
    let mut failures = Vec::new();
    for (label, prepare, scan) in [
        ("tiled prepare, staged scan", TILED, STAGED),
        ("staged prepare, tiled scan", STAGED, TILED),
        ("tiled prepare, tiled scan", TILED, TILED),
    ] {
        reset_delta(session, device, operands)?;
        stages.delta(
            session,
            &session.stream,
            device,
            prepare,
            scan,
            live_rows,
            live,
            false,
        )?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let images = Images::read(session, device)?;
        failures.extend(
            images
                .differences(&expected, true)
                .into_iter()
                .map(|d| format!("{label}: {d}")),
        );
    }
    Ok(failures)
}

#[test]
fn channel_decay_tiled_prefill_matches_staged_bit_for_bit_gpu() {
    gpu_test(8 << 30, |session| {
        let mut failures = Vec::new();
        let mut pairs = 0;
        for (batch, counts) in matrix() {
            let operands = Operands::new(&batch);
            let device = Device::new(session, &batch, &operands)?;
            let stages = Stages::load(session, &batch, None)?;
            for live in counts {
                pairs += 1;
                let live_rows = batch.live_rows(live);
                device.set_live(session, live)?;
                run_conv(session, &stages, &device, &operands, live_rows, live)?;
                let found = compare_live(session, &stages, &device, &operands, live_rows, live)?;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!(batch.name));
                receipt.insert("bucket_rows".into(), json!(batch.rows()));
                receipt.insert("bucket_sequences".into(), json!(batch.bucket_sequences));
                receipt.insert("live".into(), json!(live));
                receipt.insert("live_rows".into(), json!(live_rows));
                receipt.insert("correct".into(), json!(found.is_empty()));
                receipt.insert("differences".into(), json!(found));
                session.receipt("channel_decay_tiled_bitwise", receipt)?;
                failures.extend(
                    found
                        .into_iter()
                        .map(|d| format!("{} live {live}: {d}", batch.name)),
                );
            }
        }
        if pairs != MATRIX_PAIRS {
            failures.push(format!("{pairs} of {MATRIX_PAIRS} matrix pairs ran"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("the tiled prefill stages reproduce the staged stages bit for bit");
}

// ------------------------------------------------------ poisoned replays ----

/// The tiled chain (staged convolution, tiled prepare and scan) of `batch`
/// under poisoned eager, relaunch, two graph replays and a PDL pair whose
/// producer is the convolution, at each of `counts`; the eager workspace,
/// output and state pool equal the staged chain's. Only live output rows,
/// the live sequences' state entries and the live chunks' workspace may
/// change: the prepare writes every F32 of a live chunk (zeros above the
/// diagonal of P), and the workspace past the live chunks stays poisoned.
/// Returns the live counts run.
fn replay_batch(session: &Session, batch: &Batch, counts: &[usize]) -> LabResult<usize> {
    let operands = Operands::new(batch);
    let device = Device::new(session, batch, &operands)?;
    let stages = Stages::load(session, batch, None)?;
    let (rows, sequences) = (batch.rows(), batch.sequences());
    let chain = |stream: &Stream, live_rows: usize, live: usize, pdl: bool| -> LabResult<()> {
        stages
            .conv
            .launch(session, stream, &device.conv(), live_rows, live, pdl)?;
        stages.delta(session, stream, &device, TILED, TILED, live_rows, live, pdl)
    };
    let graph = session.capture(|stream| chain(stream, rows, sequences, true))?;
    for &live in counts {
        let live_rows = batch.live_rows(live);
        device.set_live(session, live)?;
        run_conv(session, &stages, &device, &operands, live_rows, live)?;
        let expected = staged_images(session, &stages, &device, &operands, live_rows, live)?;
        let (written, updated) = batch.writes(live);
        let mut checks = Replay::new(session)
            .output(
                "conv_output",
                &device.conv_out,
                WriteMap::of(DType::Bf16).prefix(live_rows * CHANNELS)?,
            )
            .output(
                "delta_output",
                &device.core,
                WriteMap::of(DType::Bf16).prefix(live_rows * WIDTH)?,
            )
            .state(
                "conv_state",
                &device.conv_state,
                bytes_u16(&operands.conv_pool),
                WriteMap::of(DType::Bf16).byte_ranges(entry_ranges(&written, CONV_ENTRY * 2))?,
                entry_ranges(&updated, CONV_ENTRY * 2),
            )
            .state(
                "delta_state",
                &device.delta_state,
                bytes_f32(&operands.delta_pool),
                WriteMap::of(DType::F32).byte_ranges(entry_ranges(&written, STATE * 4))?,
                entry_ranges(&updated, STATE * 4),
            )
            // Ranges derived from the lengths and positions, not the planner.
            .output(
                "workspace",
                &device.workspace,
                WriteMap::of(DType::F32).byte_prefix(batch.live_workspace(live))?,
            );
        let label = |error: String| format!("{} live {live} of {sequences}: {error}", batch.name);
        checks
            .eager(|| chain(&session.stream, live_rows, live, false))
            .map_err(label)?;
        for (name, image) in [
            ("workspace", &expected.workspace),
            ("delta_output", &expected.core),
            ("delta_state", &expected.pool),
        ] {
            let eager = checks.expected(name).ok_or("missing eager image")?;
            if let Some(difference) = first_difference(name, eager, image) {
                return Err(label(difference));
            }
        }
        checks
            .relaunch(|| chain(&session.stream, live_rows, live, false))
            .map_err(label)?;
        checks.graph(&graph).map_err(label)?;
        checks.graph(&graph).map_err(label)?;
        checks
            .pdl(
                || stages.conv(session, &device, live_rows, live, false),
                || {
                    stages.delta(
                        session,
                        &session.stream,
                        &device,
                        TILED,
                        TILED,
                        live_rows,
                        live,
                        true,
                    )
                },
                true,
            )
            .map_err(label)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("family".into(), json!("channel_decay_tiled"));
        receipt.insert("case".into(), json!(batch.name));
        receipt.insert("bucket_sequences".into(), json!(batch.bucket_sequences));
        receipt.insert("bucket_rows".into(), json!(rows));
        receipt.insert("live".into(), json!(live));
        receipt.insert("live_rows".into(), json!(live_rows));
        receipt.insert("equals_staged".into(), json!(true));
        receipt.insert("correct".into(), json!(true));
        checks.record(&mut receipt);
        session.receipt("poisoned_replay", receipt)?;
    }
    Ok(counts.len())
}

#[test]
fn channel_decay_tiled_poisoned_replays_match_eager_gpu() {
    gpu_test(8 << 30, |session| {
        let mut pairs = 0;
        for (batch, counts) in matrix() {
            pairs += replay_batch(session, &batch, &counts)?;
        }
        if pairs == MATRIX_PAIRS {
            Ok(())
        } else {
            Err(format!("{pairs} of {MATRIX_PAIRS} matrix pairs ran"))
        }
    })
    .expect("poisoned eager, relaunch, graph and PDL launches of the tiled stages agree");
}

// ----------------------------------------------------------- mutants ----

/// The lab's change-watch PDL producer (`probe_pdl_change_watch_sm120`,
/// family `sinkhorn_epilogue_split`) of the delta rule's `qkv`: after its
/// own dependency wait it fills `qkv` with NaN and reads the first word of
/// the workspace (poisoned; the prepare writes it only from `qkv`),
/// triggers its dependents, and holds until that word changes or
/// `hold_ns` passes; only then does it copy the convolution's output from
/// `fresh`. A stage that reads `qkv` before its dependency wait reads NaN.
struct WatchProducer {
    _module: Module,
    function: Function,
    fresh: DeviceBuffer,
    record: DeviceBuffer,
}

const WATCH_CUBIN: &str = "probe_pdl_change_watch_sm120";
const WATCH_SYMBOL: &str = "chop_probe_pdl_change_watch_producer_v1";
/// Holds of the watch producer: the served stages wait for it, so it holds
/// fully; a mutant's early reads happen within microseconds of the trigger.
const SERVED_HOLD_NS: u64 = 2_000_000;
const MUTANT_HOLD_NS: u64 = 20_000_000;

impl WatchProducer {
    /// The producer of `device.conv_out` from its current bytes.
    fn new(session: &Session, device: &Device) -> LabResult<Self> {
        let module = session.module(WATCH_CUBIN)?;
        let function = module.function(WATCH_SYMBOL).map_err(|e| e.to_string())?;
        let fresh = upload(session, &poison::read(session, &device.conv_out)?)?;
        Ok(Self {
            _module: module,
            function,
            fresh,
            record: session.allocate(8)?,
        })
    }

    /// One launch on the stream, ahead of the PDL dependents.
    fn launch(&self, session: &Session, device: &Device, hold_ns: u64) -> LabResult<()> {
        poison::fill_all(session, &self.record, &[POISON])?;
        poison::fill_all(session, &device.conv_out, &[POISON])?;
        self.function
            .launch(
                &session.stream,
                launch::simple_config(
                    (64, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[
                    runtime_cuda::KernelArgument::Buffer(&self.fresh),
                    runtime_cuda::KernelArgument::Buffer(&device.conv_out),
                    runtime_cuda::KernelArgument::U64((device.conv_out.byte_len() / 4) as u64),
                    runtime_cuda::KernelArgument::Buffer(&device.workspace),
                    runtime_cuda::KernelArgument::U64(hold_ns),
                    runtime_cuda::KernelArgument::Buffer(&self.record),
                ],
            )
            .map_err(|e| e.to_string())
    }

    /// After the stream synchronized: whether the dependent wrote the
    /// watched word before the copy, and the nanoseconds held. A record
    /// the producer did not write, or a copy it did not finish, is an
    /// error.
    fn record(&self, session: &Session, device: &Device) -> LabResult<(bool, u32)> {
        let bytes = poison::read(session, &self.record)?;
        let word = |i: usize| u32::from_ne_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let ahead = match word(0) {
            0 => false,
            1 => true,
            other => return Err(format!("the watch producer recorded {other:#x}")),
        };
        let fresh = poison::read(session, &self.fresh)?;
        if let Some(difference) = first_difference(
            "watch producer output",
            &poison::read(session, &device.conv_out)?,
            &fresh,
        ) {
            return Err(difference);
        }
        Ok((ahead, word(1)))
    }

    /// The tiled pair of `stages` as the producer's PDL dependents from
    /// poisoned buffers: the images, whether the prepare got ahead, and the
    /// hold.
    fn run(
        &self,
        session: &Session,
        stages: &Stages,
        device: &Device,
        operands: &Operands,
        rows: usize,
        hold_ns: u64,
    ) -> LabResult<(Images, bool, u32)> {
        reset_delta(session, device, operands)?;
        self.launch(session, device, hold_ns)?;
        stages.delta(
            session,
            &session.stream,
            device,
            TILED,
            TILED,
            rows,
            1,
            true,
        )?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let (ahead, held) = self.record(session, device)?;
        Ok((Images::read(session, device)?, ahead, held))
    }
}

/// Runs `mutant`'s tiled stages on a 2,048-row call that starts and ends
/// off multiples of 64 from poisoned buffers. The late-wait mutant runs as
/// the PDL dependent of the watch producer, after the served tiled pair
/// passes behind it with a full hold and the staged bits; the others run
/// eagerly. The mutant is caught when an output, the workspace or the pool
/// differs from the staged stages' bits; a launch error, a missing producer
/// record or an unfinished producer copy fails the test.
fn mutant_caught(session: &Session, mutant: &str) -> LabResult<Vec<String>> {
    let batch = Batch::single("mutant", 2048, 4096 + 37, true, 0x3a7a);
    let operands = Operands::new(&batch);
    let device = Device::new(session, &batch, &operands)?;
    let stages = Stages::load(session, &batch, Some(mutant))?;
    let rows = batch.rows();
    run_conv(session, &stages, &device, &operands, rows, 1)?;
    let expected = staged_images(session, &stages, &device, &operands, rows, 1)?;
    if mutant == "late_wait" {
        let watch = WatchProducer::new(session, &device)?;
        let served = Stages::load(session, &batch, None)?;
        let (images, ahead, held) =
            watch.run(session, &served, &device, &operands, rows, SERVED_HOLD_NS)?;
        let differing = images.differences(&expected, true);
        if ahead || u64::from(held) < SERVED_HOLD_NS || !differing.is_empty() {
            return Err(format!(
                "the served tiled pair behind the watch producer: ahead {ahead}, held {held} ns, \
                 differing {differing:?}"
            ));
        }
        let (images, ahead, held) =
            watch.run(session, &stages, &device, &operands, rows, MUTANT_HOLD_NS)?;
        return Ok(images
            .differences(&expected, true)
            .into_iter()
            .map(|d| format!("behind the watch producer (ahead {ahead}, held {held} ns): {d}"))
            .collect());
    }
    reset_delta(session, &device, &operands)?;
    stages.delta(
        session,
        &session.stream,
        &device,
        TILED,
        TILED,
        rows,
        1,
        false,
    )?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(Images::read(session, &device)?.differences(&expected, true))
}

#[test]
fn channel_decay_tiled_checks_reject_every_mutant_gpu() {
    gpu_test(8 << 30, |session| {
        let mut missed = Vec::new();
        for (mutant, _) in TILED_MUTANTS {
            let found = mutant_caught(session, mutant)?;
            let mut receipt = BTreeMap::new();
            receipt.insert("mutant".into(), json!(mutant));
            receipt.insert("caught".into(), json!(!found.is_empty()));
            receipt.insert("observed".into(), json!(found));
            receipt.insert("correct".into(), json!(!found.is_empty()));
            session.receipt("channel_decay_tiled_mutant", receipt)?;
            if found.is_empty() {
                missed.push(mutant);
            }
        }
        if missed.is_empty() {
            Ok(())
        } else {
            Err(format!("mutants not caught: {missed:?}"))
        }
    })
    .expect("every tiled mutant changes an observed output");
}

// ----------------------------------------------------------- latency ----

/// Median and minimum milliseconds of the staged and tiled prepare, scan
/// and pair for one sequence of 64, 2,048 and 8,192 rows in the served
/// bucket of 32 sequences, after a bitwise check of the tiled pair. Opt-in:
/// `CHOPINFER_KERNEL_BENCH=1`.
#[test]
fn channel_decay_tiled_prefill_latency_gpu() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("channel_decay_tiled latency: set CHOPINFER_KERNEL_BENCH=1")
            .expect("skip allowed");
        return;
    }
    gpu_test(8 << 30, |session| {
        let mut table = Vec::new();
        for rows in [64, 2048, 8192] {
            let batch = Batch::single(&format!("rows{rows}"), rows, 0, true, 0x7100 + rows as u64);
            let operands = Operands::new(&batch);
            let device = Device::new(session, &batch, &operands)?;
            let stages = Stages::load(session, &batch, None)?;
            run_conv(session, &stages, &device, &operands, rows, 1)?;
            let found = compare_live(session, &stages, &device, &operands, rows, 1)?;
            if !found.is_empty() {
                return Err(format!("rows {rows}: {}", found.join("; ")));
            }
            let mut timings = BTreeMap::new();
            for (name, schedule) in [("staged", STAGED), ("tiled", TILED)] {
                let prepare = kernel_lab::time_named(
                    session,
                    &format!("channel_decay_{name}_prepare_{rows}"),
                    3,
                    15,
                    |stream| {
                        stages.prepare[schedule].launch(
                            session,
                            stream,
                            &device.delta(),
                            rows,
                            1,
                            true,
                        )
                    },
                )?;
                let scan = kernel_lab::time_named(
                    session,
                    &format!("channel_decay_{name}_scan_{rows}"),
                    3,
                    15,
                    |stream| {
                        stages.scan[schedule].launch(
                            session,
                            stream,
                            &device.delta(),
                            rows,
                            1,
                            true,
                        )
                    },
                )?;
                let pair = kernel_lab::time_named(
                    session,
                    &format!("channel_decay_{name}_pair_{rows}"),
                    3,
                    15,
                    |stream| {
                        stages.delta(session, stream, &device, schedule, schedule, rows, 1, true)
                    },
                )?;
                for (stage, timing) in [("prepare", prepare), ("scan", scan), ("pair", pair)] {
                    timings.insert(
                        format!("{name}_{stage}"),
                        json!({"median_ms": timing.median_ms, "min_ms": timing.min_ms}),
                    );
                    table.push(format!(
                        "rows {rows:>5} {name:>6} {stage:>7}: median {:.4} ms, min {:.4} ms",
                        timing.median_ms, timing.min_ms
                    ));
                }
            }
            let mut receipt = BTreeMap::new();
            receipt.insert("rows".into(), json!(rows));
            receipt.insert("bucket_sequences".into(), json!(SERVED_SEQUENCES));
            receipt.insert("tiled_bitwise".into(), json!(true));
            receipt.insert("timings".into(), json!(timings));
            session.receipt("channel_decay_tiled_latency", receipt)?;
        }
        eprintln!("{}", table.join("\n"));
        Ok(())
    })
    .expect("the staged and tiled prefill stages are timed");
}

// --------------------------------------------------------------- CPU ----

/// The GPU matrix cannot silently shrink: every batch (64, 2,048 and 8,192
/// rows, the gate bound, the mixed batches and the full 32-sequence bucket)
/// runs every live count from 0 to its sequences, starting with
/// `poison::live_counts`' order.
#[test]
fn channel_decay_tiled_gpu_matrix_covers_every_live_count() {
    let matrix = matrix();
    let names = matrix
        .iter()
        .map(|(batch, _)| batch.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "rows64_exact_bucket",
            "rows64",
            "rows2048",
            "rows2048_phase37",
            "rows8192",
            "rows8192_phase5",
            "rows256_gate_bound",
            "mixed",
            "mixed_served_bucket",
            "many_sequences",
        ]
    );
    for (batch, counts) in &matrix {
        let capacity = batch.sequences();
        let leading = poison::live_counts(capacity as u32);
        assert!(
            counts.iter().zip(&leading).all(|(&a, &b)| a == b as usize),
            "{}",
            batch.name
        );
        let mut sorted = counts.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..=capacity).collect::<Vec<_>>(), "{}", batch.name);
    }
    let (full, counts) = matrix.last().expect("batches");
    assert_eq!(
        (full.sequences(), counts.len()),
        (SERVED_SEQUENCES, SERVED_SEQUENCES + 1)
    );
    assert_eq!(
        matrix.iter().map(|(_, counts)| counts.len()).sum::<usize>(),
        MATRIX_PAIRS
    );
}

#[test]
fn channel_decay_tiled_lab_cubins_are_built() {
    let tiled = channel_decay::tiled_catalog();
    assert!(
        tiled
            .iter()
            .all(|entry| entry.schedule == PrefillSchedule::Tiled)
    );
    let mut names = vec![(tiled_module_name(None), None)];
    names.extend(
        TILED_MUTANTS
            .iter()
            .map(|(mutant, _)| (tiled_module_name(Some(mutant)), Some(*mutant))),
    );
    for (name, mutant) in names {
        let mut built = kernel_lab::cubin_entries(&name)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .to_vec();
        built.sort_unstable();
        let mut wanted = tiled
            .iter()
            .map(|entry| match mutant {
                None => entry.symbol.clone(),
                Some(mutant) => channel_decay::tiled_mutant_symbol(&entry.symbol, mutant),
            })
            .collect::<Vec<_>>();
        wanted.sort_unstable();
        assert_eq!(built, wanted, "{name}");
    }
}
