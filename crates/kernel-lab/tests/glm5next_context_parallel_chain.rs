//! One sparse latent layer of the glm5next bundle at context-parallel sizes 2
//! (two pairs) and 4 on four GPUs, against the same layer at size 1: the
//! bundle's own lowering (`bundle_glm5next::lower::Lowering`, which
//! `lower/context_parallel.rs` extends) runs from the cubins the bundle
//! embeds, under its own symbols, plans and bound arguments.
//!
//! The lowering binds fake buffer handles, as the bundle's fake-host tests
//! do; the lab backs each handle on its rank's GPU with one allocation over
//! the window of it that some launch, operand or check touches, and passes
//! every bound span (peer outputs of the group all-gathers included) as its
//! device address. Per step the layer's stages run in program order
//! (`bundle_glm5next::launch::records`): the latent KV write (each row's
//! position localized to its owner), the index key write (owned pools and
//! the replicated tail), the index selection (per-rank selection, the pair
//! all-gather and the exact top-k merge) and the latent attention (key
//! localization, the query all-gather, per-source partial attention, the
//! exchange of the partial blocks to their owners, the partial merge and
//! the group barrier); the all-gathers and the exchange run the phased
//! group gather.
//!
//! The steps ([`steps`]), over twenty items, each in the phase the instance
//! picks: three prefills (the first of 4300 rows, whose indexer stages run
//! in the bundle's three pieces with one cut inside an item; items out of
//! item order in the second), decode steps at every bucket (1 to 32) and
//! verify steps at every token bucket (2 to 32), whose tails go to
//! checkpoints. Each step's inputs are packed as the instance uploads them:
//! the phase's arena by its `InputLayout`, its table rows in the phase's
//! item order (-1 past an item's pages), each piece in its own region with
//! its table rows from its first item's, and a verify step's tail
//! checkpoint table. Every rank shares one logical page table per item, as
//! the host sends it (`runtime_kv::ContextParallelPagesV1`: each entry is a
//! stripe of `size` pages, one local page on every group rank), with tables
//! and page ids drawn independently of the size-1 run's.
//!
//! The MTP drafter's layer runs the same way ([`drafter_steps`], over the
//! formats that page it, its stages from `drafter_records`): extend passes
//! as a prefill and a verify, and draft steps that compact the last extend
//! pass's rows (`DraftShareRows`) into `SharedKeyLists`, the lists a draft
//! step's attention localizes; a draft step's selection is compared there.
//!
//! Per step, against the size-1 layer of the same rank and operands:
//!
//! - selection: every live row of the key list is equal bit for bit;
//! - KV and pooled keys: each page a group rank owns
//!   (`ContextParallelPagesV1::rank_pages`) equals the size-1 page of the
//!   same tokens over the layer's whole page region, every local page it
//!   does not own stays poisoned, and the indexer tail equals the size-1
//!   tail; the size-1 run writes every appended latent row and completed
//!   pool (no element left poisoned);
//! - attention: both outputs pass the f64 oracle's gate
//!   (`latent_attention_oracle::latent_attention`, half a BF16 ulp plus
//!   `2^-22 K S`) on sampled rows over the size-1 pool and lists; the
//!   differing elements and the largest BF16 ulp distance between them are
//!   recorded;
//! - tail carry: where an item resumes across an internal piece cut with
//!   live tail slots ([`live_tail_cuts`]), the group's layer run with that
//!   slot's tail zeroed at the cut must not match size 1;
//! - poison: KV pages and the outputs start poisoned (the indexer tail
//!   starts zeroed, as create zeroes the host's tail state), rows past the
//!   live count stay untouched, and each step runs five times from the same
//!   state (eager, relaunched, twice from one captured graph per rank, and
//!   eager behind a PDL producer on every rank that holds the step's operand
//!   rows, positions, batch indices and live count poisoned until after its
//!   tail, [`Producer`]), each run equal to the size-1 results and the
//!   attention output equal to the first run's bit for bit.
//!
//! The GPU checks are three tests, one per layer and format group
//! (`target_fp8_*`, `target_bf16_*`, `drafter_*`), each at sizes 2 and 4.
//! Run one with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3 CHOPINFER_KERNEL_TIMEOUT_SECS=900 cargo test -p kernel-lab --release --test glm5next_context_parallel_chain <name> -- --nocapture --test-threads=1`;
//! without four GPUs they skip. The f64 oracle gates every item's first and
//! last rows and random rows of each step, over a bounded thread pool, and
//! later runs of a step are checked bit for bit against its first. The CPU test checks the lowered sequences, the
//! embedded symbols and the host tables.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use bundle_glm5next::buffers::{
    Buffer, CommState, DrafterBuffers, RankBuffers, StatePools, allocate_scratch, checkpoint_table,
    kda_checkpoint_stride,
};
use bundle_glm5next::context_parallel::{ContextParallel, RankState};
use bundle_glm5next::launch::{drafter_records, records};
use bundle_glm5next::program::{DraftPass, Phase, Stage};
use bundle_glm5next::step::{PIECE_ROWS, items as ledger_items, pieces, side_inputs};
use bundle_glm5next::{
    lower::{Lowering, Piece},
    ops::{Ops, Plan},
    sm120::{self, engines::Engines},
};
use bundle_runtime::dispatch;
use bundle_runtime::geometry::Items;
use bundle_runtime::graphs::verify_bucket_within;
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::kv::KvFormat;
use bundle_runtime::ledger::{Ledger, Placement, Rules, Step as LedgerStep};
use bundle_runtime::pipeline::PreparedLaunch;
use bundle_runtime::state::Extent;
use bundle_runtime::upload::{DeviceInputs, InputLayout, PhaseArena};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat};
use kernel_core::latent_attention_oracle::{
    LatentAttentionInput, LatentAttentionResult, LatentPaging, latent_attention,
};
use kernel_core::oracle::{OracleOutput, check_numeric, numeric_bound};
use kernel_lab::poison::POISON;
use kernel_lab::topology::RankReplay;
use kernel_lab::{LabResult, Session, gpu_test_multi, launch};
use kernels_sm120::context_parallel::Kind as CpKind;
use kernels_sm120::context_parallel::group::ALLGATHER_SYMBOL;
use kernels_sm120::context_parallel::phased;
use kernels_sm120::latent;
use kernels_sm120::pooled_indexer::Kind as IndexerKind;
use model_glm5next_contract::contract::{
    DECODE_BATCH, Glm as Model, MAX_VERIFY_ROWS, PREFILL_GRANULE, Shape, has_mtp_layer,
    latent_format,
};
use model_glm5next_contract::direct::layout;
use model_glm5next_contract::sm120::engines::pooled_indexer_geometry;
use model_glm5next_contract::{
    catalog::DType as StoredType,
    context_parallel::{Dims, rank_bytes},
};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1, KernelArgumentV1};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, LaunchConfig, Module};
use runtime_kv::ContextParallelPagesV1;
use serde_json::{Value, json};

const SHAPE: Shape = Shape::TP4;
const RANKS: usize = 4;
const PAGE: usize = 64;
const ITEMS: usize = 20;
/// Each item's state slot (`slot_ids`), out of item order.
fn slot(item: usize) -> i32 {
    ((item * 7 + 2) % DECODE_BATCH as usize) as i32
}
/// The poison of key list outputs: no position, and not -1.
const LIST_FILL: [u8; 4] = [0xa5, 0x5a, 0xc3, 0x3c];
/// The layer's indexer weight views the key write binds.
const INDEXER_VIEWS: [&str; 3] = [
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.index_kpool_compress_ape",
];

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

/// A deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// Uniform in [-scale, scale), as BF16.
    fn bf16(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| {
                let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                f32_to_bf16(unit * scale)
            })
            .collect()
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            values.swap(i, self.below(i + 1));
        }
    }
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------- steps ----

/// One step: its phase and each item's appended rows `(item, first, count)`,
/// in the phase's item order. A row's batch index is its item's place in
/// that order, and the phase's table rows follow it, as the host packs them.
struct StepSpec {
    phase: Phase,
    items: Vec<(usize, usize, usize)>,
    /// The MTP drafter's pass, or `None` for the target's layer.
    pass: Option<DraftPass>,
    /// A draft step's `DraftShareRows`: each item's last row of the last
    /// extend pass, whose key list it reuses.
    share: Vec<i32>,
}

impl StepSpec {
    fn live(&self) -> usize {
        self.items.iter().map(|&(_, _, count)| count).sum()
    }

    /// Each live row's item and position.
    fn rows(&self) -> Vec<(usize, usize)> {
        self.items
            .iter()
            .flat_map(|&(item, first, count)| (first..first + count).map(move |p| (item, p)))
            .collect()
    }
}

/// How a step's rows run.
#[derive(Clone, Copy)]
enum Kind {
    Prefill,
    Decode,
    Verify,
}

/// Steps over items' running lengths, each in the phase the instance
/// would pick (`instance/execute.rs` `phase_of`): an eager prefill at its
/// live rows, a decode graph at the items' bucket, a verify graph at the
/// rows' token bucket and the items' bucket, widened to hold the rows.
#[derive(Default)]
struct Script {
    lengths: [usize; ITEMS],
    steps: Vec<StepSpec>,
}

impl Script {
    fn push(&mut self, kind: Kind, appended: &[(usize, usize)]) {
        let live = appended.iter().map(|&(_, count)| count).sum::<usize>() as u32;
        let bucket = items()
            .bucket(appended.len() as u32)
            .expect("a served item count");
        let phase = match kind {
            Kind::Prefill => Phase::Prefill { rows: live },
            Kind::Decode => {
                assert!(appended.iter().all(|&(_, count)| count == 1));
                Phase::Decode(bucket)
            }
            Kind::Verify => {
                let rows = verify_bucket_within::<Model>(live, items()).expect("verify rows");
                let held = rows.div_ceil(MAX_VERIFY_ROWS).next_power_of_two();
                Phase::Verify {
                    rows,
                    items: bucket.max(held),
                }
            }
        };
        let items = appended
            .iter()
            .map(|&(item, count)| {
                let first = self.lengths[item];
                self.lengths[item] += count;
                (item, first, count)
            })
            .collect();
        self.steps.push(StepSpec {
            phase,
            items,
            pass: None,
            share: Vec::new(),
        });
    }

    /// The drafter's extend pass over appended committed rows, in the phase
    /// of the target step it follows.
    fn extend(&mut self, kind: Kind, appended: &[(usize, usize)]) {
        self.push(kind, appended);
        self.steps.last_mut().expect("pushed").pass = Some(DraftPass::Extend);
    }

    /// A draft step of one row per item, `ahead` rows past its committed
    /// length, reusing each item's last row of the last extend pass. It
    /// commits nothing.
    fn draft_step(&mut self, appended: &[usize], ahead: usize) {
        let extend = self
            .steps
            .iter()
            .rev()
            .find(|step| step.pass == Some(DraftPass::Extend))
            .expect("an extend pass first");
        let mut last = BTreeMap::new();
        let mut row = 0;
        for &(item, _, count) in &extend.items {
            row += count;
            last.insert(item, row as i32 - 1);
        }
        let share = appended.iter().map(|item| last[item]).collect();
        let bucket = items()
            .bucket(appended.len() as u32)
            .expect("a served item count");
        self.steps.push(StepSpec {
            phase: Phase::Decode(bucket),
            items: appended
                .iter()
                .map(|&item| (item, self.lengths[item] + ahead, 1))
                .collect(),
            pass: Some(DraftPass::Step),
            share,
        });
    }
}

/// The MTP drafter's passes on its own layer: an extend pass as a prefill
/// of three items (1300, 600 and 90 rows), a draft step of each, an extend
/// pass as a verify of three rows each (over the draft step's position),
/// then two chained draft steps (the second of two items out of order, one
/// row further), each compacting the extend pass's last rows into
/// `SharedKeyLists`.
fn drafter_steps() -> Vec<StepSpec> {
    let mut script = Script::default();
    script.extend(Kind::Prefill, &[(0, 1300), (1, 600), (2, 90)]);
    script.draft_step(&[0, 1, 2], 0);
    script.extend(Kind::Verify, &[(0, 3), (1, 3), (2, 3)]);
    script.draft_step(&[0, 1, 2], 0);
    script.draft_step(&[2, 0], 1);
    script.steps
}

/// Which layer a run chains: the target's first sparse layer, or the MTP
/// drafter's layer over the formats that page it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scenario {
    Target,
    Drafter,
}

impl Scenario {
    fn steps(self) -> Vec<StepSpec> {
        match self {
            Self::Target => steps(),
            Self::Drafter => drafter_steps(),
        }
    }

    fn layer(self) -> u32 {
        match self {
            Self::Target => sparse_layer(),
            Self::Drafter => SHAPE.geometry.layers(),
        }
    }

    fn format(self, kv: LatentKvFormat) -> LabResult<KvFormat> {
        let mtp = self == Self::Drafter;
        SHAPE
            .kv_formats()
            .into_iter()
            .find(|format| latent_format(format) == kv && has_mtp_layer(format) == mtp)
            .ok_or(format!("no {kv:?} format for {self:?}"))
    }
}

/// The steps, in order:
///
/// - a prefill of 4300 rows the bundle runs its indexer stages over in
///   three pieces (`bundle_glm5next::step::pieces`: item 0's first 2048
///   rows, its last 952 rows with item 1's first 1088, then item 1's last
///   212);
/// - a prefill of two items out of item order;
/// - a continuation of item 2 by 2500 rows from position 130, which the
///   bundle cuts at its row 2048: position 2178 resumes with two live tail
///   slots carried across the cut ([`live_tail_cuts`]);
/// - a prefill of items 3 to 19, of 30 to 174 rows each (one piece; at size
///   4 no item reaches group rank 3's pages);
/// - decode steps at every bucket: 4 (three items), 2 (two items out of
///   order), 1, 8, 16 and 32 (seven, thirteen and twenty items); item 0
///   completes pool 750 across the second prefill and the first two
///   decodes;
/// - verify steps at every token bucket: 8 (items of 4 and 3 rows; item 1
///   completes pool 325), 2, 4, 16 (four items) and 32 (eight items), each
///   over items no later step touches, since a verify step's tail goes to
///   checkpoints a commit would copy.
fn steps() -> Vec<StepSpec> {
    let mut script = Script::default();
    script.push(Kind::Prefill, &[(0, 3000), (1, 1300)]);
    script.push(Kind::Prefill, &[(2, 130), (0, 2)]);
    script.push(Kind::Prefill, &[(2, 2500)]);
    let short = (3..ITEMS)
        .map(|item| (item, 30 + 9 * (item - 3)))
        .collect::<Vec<_>>();
    script.push(Kind::Prefill, &short);
    let decode = |items: std::ops::Range<usize>| items.map(|item| (item, 1)).collect::<Vec<_>>();
    script.push(Kind::Decode, &[(0, 1), (1, 1), (2, 1)]);
    script.push(Kind::Decode, &[(2, 1), (0, 1)]);
    script.push(Kind::Decode, &[(5, 1)]);
    script.push(Kind::Decode, &decode(3..10));
    script.push(Kind::Decode, &decode(3..16));
    script.push(Kind::Decode, &decode(0..ITEMS));
    script.push(Kind::Verify, &[(1, 4), (0, 3)]);
    script.push(Kind::Verify, &[(2, 2)]);
    script.push(Kind::Verify, &[(3, 4)]);
    script.push(
        Kind::Verify,
        &(4..8).map(|item| (item, 4)).collect::<Vec<_>>(),
    );
    script.push(
        Kind::Verify,
        &(8..16).map(|item| (item, 4)).collect::<Vec<_>>(),
    );
    script.steps
}

/// The internal piece cuts of `step`'s eager prefill that an item resumes
/// across with live tail slots, its position not a multiple of the pool:
/// `(piece, item, position)` of each piece that starts so.
fn live_tail_cuts(step: &StepSpec, phase: &PhaseInputs) -> LabResult<Vec<(usize, u64, i32)>> {
    if !matches!(step.phase, Phase::Prefill { .. }) {
        return Ok(Vec::new());
    }
    let pool = SHAPE.profile.index_pool as i32;
    let pieces = pieces(phase, PREFILL_GRANULE)?;
    Ok(pieces
        .windows(2)
        .enumerate()
        .filter_map(|(index, pair)| {
            let (before, after) = (&pair[0].1, &pair[1].1);
            let item = after.request_ids[0];
            let resumes = before.request_ids.last() == Some(&item);
            (resumes && after.prefix_len[0] % pool != 0).then_some((
                index + 1,
                item,
                after.prefix_len[0],
            ))
        })
        .collect())
}

/// Each item's length after the last step.
fn final_lengths(steps: &[StepSpec]) -> [usize; ITEMS] {
    let mut ends = [0; ITEMS];
    for step in steps {
        for &(item, first, count) in &step.items {
            ends[item] = ends[item].max(first + count);
        }
    }
    ends
}

/// The host operands of one step, shared by every run: the replicated rows
/// (latent, indexer key, gate, query and head weights) and each tensor
/// rank's absorbed queries of its own heads.
struct StepData {
    batch: Vec<i32>,
    latent: Vec<u16>,
    key: Vec<u16>,
    gate: Vec<u16>,
    index_q: Vec<u16>,
    index_w: Vec<u16>,
    queries: Vec<Vec<u16>>,
}

impl StepData {
    fn new(step: &StepSpec, rng: &mut Rng) -> Self {
        let p = SHAPE.profile;
        let rows = step.rows();
        let batch = step
            .items
            .iter()
            .enumerate()
            .flat_map(|(index, &(_, _, count))| std::iter::repeat_n(index as i32, count))
            .collect();
        let n = rows.len();
        let heads = p.latent_heads as usize;
        let latent = p.kv_rank as usize;
        Self {
            batch,
            latent: rng.bf16(n * latent, 2.0),
            key: rng.bf16(n * p.indexer_dim as usize, 2.0),
            gate: rng.bf16(n * p.indexer_dim as usize, 1.0),
            index_q: rng.bf16(n * (p.indexer_heads * p.indexer_dim) as usize, 1.0),
            index_w: rng.bf16(n * p.indexer_heads as usize, 1.0),
            queries: (0..RANKS)
                .map(|_| rng.bf16(n * heads * latent, 0.5))
                .collect(),
        }
    }
}

// ------------------------------------------------------- fake handles ----

/// The fake handles the bundle's spans name: each slot's rank and the
/// window `[start, end)` of it that something touches.
#[derive(Default)]
struct Handles {
    slots: RefCell<Vec<Slot>>,
}

/// A handle's rank and touched window.
type Slot = (usize, Option<(u64, u64)>);

impl Handles {
    fn allocate(&self, rank: usize, bytes: u64) -> DeviceBufferSpanV1 {
        let mut slots = self.slots.borrow_mut();
        slots.push((rank, None));
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, slots.len() as u32),
            0,
            bytes,
        )
    }

    fn touch(&self, span: DeviceBufferSpanV1) {
        let mut slots = self.slots.borrow_mut();
        let slot = &mut slots[span.buffer.slot() as usize - 1];
        let end = span.offset + span.length;
        slot.1 = Some(match slot.1 {
            None => (span.offset, end),
            Some((start, last)) => (start.min(span.offset), last.max(end)),
        });
    }

    /// Bytes each rank's windows take.
    fn bytes_per_rank(&self) -> [u64; RANKS] {
        let mut bytes = [0; RANKS];
        for &(rank, window) in self.slots.borrow().iter() {
            if let Some((start, end)) = window {
                bytes[rank] += end - start;
            }
        }
        bytes
    }
}

/// Every device span one prepared launch binds.
fn bound_spans(launch: &PreparedLaunch) -> LabResult<Vec<DeviceBufferSpanV1>> {
    let arguments = dispatch::arguments(&launch.arguments, &launch.bindings).map_err(err)?;
    let mut spans = Vec::new();
    for argument in arguments {
        match argument {
            KernelArgumentV1::DeviceBuffer(span) => spans.push(span),
            KernelArgumentV1::DevicePointerArray(array) => {
                spans.extend(array.into_iter().flatten())
            }
            _ => {}
        }
    }
    Ok(spans)
}

/// One launch argument with its spans resolved to device addresses.
enum Arg {
    U32(u32),
    I32(i32),
    U64(u64),
    Pointers(Vec<u64>),
}

/// One resolved launch.
struct Call {
    symbol: String,
    config: LaunchConfig,
    args: Vec<Arg>,
}

impl Call {
    fn resolve(
        launch: &PreparedLaunch,
        address: &impl Fn(DeviceBufferSpanV1) -> LabResult<u64>,
    ) -> LabResult<Self> {
        let arguments = dispatch::arguments(&launch.arguments, &launch.bindings).map_err(err)?;
        let args = arguments
            .into_iter()
            .map(|argument| {
                Ok(match argument {
                    KernelArgumentV1::DeviceBuffer(span) => Arg::U64(address(span)?),
                    KernelArgumentV1::NullDevicePointer => Arg::U64(0),
                    KernelArgumentV1::ScalarU32(value) => Arg::U32(value),
                    KernelArgumentV1::ScalarI32(value) => Arg::I32(value),
                    KernelArgumentV1::ScalarU64(value) => Arg::U64(value),
                    KernelArgumentV1::DevicePointerArray(array) => Arg::Pointers(
                        array
                            .into_iter()
                            .map(|span| span.map_or(Ok(0), address))
                            .collect::<LabResult<_>>()?,
                    ),
                    other => return Err(format!("{}: unsupported {other:?}", launch.symbol)),
                })
            })
            .collect::<LabResult<_>>()?;
        Ok(Self {
            symbol: launch.symbol.clone(),
            config: launch::config(launch.plan)?,
            args,
        })
    }

    fn arguments(&self) -> Vec<KernelArgument<'_>> {
        self.args
            .iter()
            .map(|arg| match arg {
                Arg::U32(value) => KernelArgument::U32(*value),
                Arg::I32(value) => KernelArgument::I32(*value),
                Arg::U64(value) => KernelArgument::U64(*value),
                Arg::Pointers(addresses) => KernelArgument::DevicePointerArray(addresses),
            })
            .collect()
    }
}

// ------------------------------------------------------------ instance ----

/// One rank's bundle buffers, input arenas and the layer's indexer views
/// (span, stored dtype and index in [`INDEXER_VIEWS`]).
struct RankSetup {
    buffers: RankBuffers,
    arenas: Arenas,
    views: Vec<(DeviceBufferSpanV1, StoredType, usize)>,
}

/// One rank's input arenas as create allocates them: one per phase kind
/// (prefill, decode, verify), each with its block table, and the indexer
/// pieces' arena of [`MAX_PIECES`] regions.
struct Arenas {
    prefill: PhaseArena,
    decode: PhaseArena,
    verify: PhaseArena,
    pieces: DeviceBufferSpanV1,
}

/// Indexer pieces the pieces' arena holds.
const MAX_PIECES: u64 = 4;

fn items() -> Items {
    Items::new(DECODE_BATCH).expect("the served items")
}

/// One indexer piece's input layout and its region's stride
/// (`instance/execute.rs`).
fn piece_layout() -> (InputLayout, u64) {
    let layout = InputLayout::within(PIECE_ROWS, 1, items());
    let stride = layout.end.next_multiple_of(256);
    (layout, stride)
}

/// One rank's inputs of one step: the phase's spans, its indexer pieces
/// and the bytes the host uploads for it.
struct StepInputs {
    inputs: DeviceInputs,
    pieces: Vec<Piece>,
    uploads: Vec<(DeviceBufferSpanV1, Vec<u8>)>,
}

/// The layer at one context-parallel size on every rank: their buffers,
/// the group state and each item's logical page table.
struct Instance {
    size: usize,
    format: KvFormat,
    layer: u32,
    ranks: Vec<RankSetup>,
    cp: Option<ContextParallel>,
    peers: Vec<CommState>,
    tables: Vec<Vec<i32>>,
}

impl Instance {
    fn new(
        handles: &Handles,
        size: usize,
        format: KvFormat,
        layer: u32,
        lengths: &[usize; ITEMS],
        rng: &mut Rng,
    ) -> LabResult<Self> {
        let drafter = has_mtp_layer(&format);
        let pages = ContextParallelPagesV1::new(size, PAGE, RANKS).map_err(err)?;
        let stripes = lengths
            .iter()
            .map(|&end| end.div_ceil(pages.stripe_tokens()))
            .collect::<Vec<_>>();
        // Spare pages that no table names, which must stay poisoned.
        let pool_pages = stripes.iter().sum::<usize>() + 3;
        let mut ids = (0..pool_pages as i32).collect::<Vec<_>>();
        rng.shuffle(&mut ids);
        let mut ids = ids.into_iter();
        let tables = stripes
            .iter()
            .map(|&n| ids.by_ref().take(n).collect())
            .collect();
        let items = u64::from(DECODE_BATCH);
        let max_pages = u64::from(SHAPE.model_limits().max_pages());
        // The comm state regions an instance's all-reduces and its drafter
        // argmax's pair gathers keep.
        let (regions, draft) = {
            let engines = Engines::new(SHAPE).map_err(err)?;
            let (backend, served) = sm120::backend(&engines, SHAPE);
            let plan = Plan {
                shape: SHAPE,
                layout: engines.layout,
                index_topk: SHAPE.profile.index_topk_max,
                context_parallel: 1,
            };
            let ops = Ops::prepare(backend.as_ref(), &served, plan)
                .map_err(|refused| refused.to_string())?;
            (ops.comm_regions(), drafter.then(|| ops.draft_regions()))
        };
        let mut ranks = Vec::new();
        let mut peers = Vec::new();
        for rank in 0..RANKS {
            let mut allocate = |bytes| Ok::<_, ()>(handles.allocate(rank, bytes));
            let scratch = allocate_scratch(SHAPE, drafter, &mut allocate)
                .map_err(err)?
                .map_err(err)?;
            let reduced = scratch[&Buffer::Reduced];
            let comm = CommState::allocate(SHAPE, draft, reduced, regions, &mut allocate)
                .map_err(err)?
                .map_err(err)?;
            let extent = Extent {
                served: DECODE_BATCH,
                bucket: DECODE_BATCH,
                verify_rows: MAX_VERIFY_ROWS,
            };
            let tail = handles.allocate(rank, items * SHAPE.indexer_tail_slot_bytes(drafter));
            let retained = StatePools::retained_bytes(SHAPE, &format, extent)
                .map(|bytes| handles.allocate(rank, bytes));
            let state = StatePools::new(
                SHAPE,
                &format,
                extent,
                model_glm5next_contract::contract::head_rows(),
                retained,
                tail,
                &mut allocate,
            )
            .map_err(err)?
            .map_err(err)?;
            // The drafter's staging, on an instance that runs it.
            let drafter_buffers = drafter
                .then(|| DrafterBuffers::allocate(SHAPE, &mut allocate))
                .transpose()
                .map_err(err)?;
            let kv_pool = handles.allocate(rank, pool_pages as u64 * format.page_bytes);
            let stored = layout(SHAPE, rank as u32)?;
            let image_bytes = stored
                .iter()
                .map(|entry| entry.offset + entry.bytes())
                .max()
                .unwrap_or(0);
            let image = handles.allocate(rank, image_bytes);
            let mut views = BTreeMap::new();
            let mut indexer = Vec::new();
            for entry in &stored {
                let span = DeviceBufferSpanV1::new(image.buffer, entry.offset, entry.bytes());
                if let Some(index) = INDEXER_VIEWS.iter().position(|name| {
                    entry.view.name == format!("model.language_model.layers.{layer}.{name}")
                }) {
                    indexer.push((span, entry.dtype, index));
                }
                views.insert(entry.view.name.clone(), span);
            }
            if indexer.len() != INDEXER_VIEWS.len() {
                return Err(format!("layer {layer} lacks an indexer view"));
            }
            let arena = |bytes: u64| PhaseArena {
                arena: handles.allocate(rank, bytes),
                block_tables: handles.allocate(rank, 4 * items * max_pages),
            };
            let arenas = Arenas {
                prefill: arena(InputLayout::prefill_largest_within::<Model>(self::items()).end),
                decode: arena(InputLayout::decode_within(self::items()).end),
                verify: arena(InputLayout::verify_within::<Model>(self::items()).end),
                pieces: handles.allocate(rank, MAX_PIECES * piece_layout().1),
            };
            peers.push(comm.clone());
            ranks.push(RankSetup {
                buffers: RankBuffers {
                    scratch,
                    comm,
                    state,
                    views,
                    kv_pool,
                    drafter: drafter_buffers,
                },
                arenas,
                views: indexer,
            });
        }
        let cp = if size > 1 {
            let bytes = rank_bytes(Dims::of(SHAPE), size as u32)?;
            Some(ContextParallel {
                size: size as u32,
                ranks: (0..RANKS)
                    .map(|rank| {
                        RankState::allocate(&bytes, |bytes| {
                            Ok::<_, ()>(handles.allocate(rank, bytes))
                        })
                        .map_err(err)
                    })
                    .collect::<LabResult<_>>()?,
            })
        } else {
            None
        };
        let instance = Self {
            size,
            format,
            layer,
            ranks,
            cp,
            peers,
            tables,
        };
        instance.touch_operands(handles)?;
        Ok(instance)
    }

    fn kv(&self) -> LatentKvFormat {
        latent_format(&self.format)
    }

    fn ordinal(&self) -> u64 {
        SHAPE.sparse_ordinal(self.layer).expect("a sparse layer")
    }

    /// The layer's offset in a page and its bytes there.
    fn page_region(&self) -> (usize, usize) {
        let bytes = SHAPE.page_layer_bytes(self.kv());
        ((self.ordinal() * bytes) as usize, bytes as usize)
    }

    /// Rank `rank`'s tail entry `entry` (a slot's own) of the layer.
    fn tail_entry(&self, rank: usize, entry: u64) -> LabResult<DeviceBufferSpanV1> {
        let state = &self.ranks[rank].buffers.state;
        let index = state
            .state
            .buffers
            .iter()
            .position(|buffer| buffer.name == "indexer.tail")
            .ok_or("no indexer tail")?;
        Ok(state.entry(index, self.ordinal(), entry))
    }

    fn tail(&self, rank: usize) -> LabResult<DeviceBufferSpanV1> {
        self.ranks[rank]
            .buffers
            .state
            .layer("indexer.tail", self.ordinal())
    }

    /// The spans the test writes or reads besides the launches'.
    fn touch_operands(&self, handles: &Handles) -> LabResult<()> {
        for rank in 0..RANKS {
            let setup = &self.ranks[rank];
            let arenas = &setup.arenas;
            for span in [
                arenas.prefill.arena,
                arenas.prefill.block_tables,
                arenas.decode.arena,
                arenas.decode.block_tables,
                arenas.verify.arena,
                arenas.verify.block_tables,
                arenas.pieces,
                setup.buffers.kv_pool,
                self.tail(rank)?,
            ] {
                handles.touch(span);
            }
            for buffer in [
                Buffer::Latent,
                Buffer::IndexKey,
                Buffer::IndexGate,
                Buffer::IndexQuery,
                Buffer::IndexWeights,
                Buffer::QueryAbsorbed,
                Buffer::KeyLists,
                Buffer::AttentionOut,
                Buffer::TailCheckpointIds,
            ] {
                handles.touch(setup.buffers.get(buffer));
            }
            if setup.buffers.drafter.is_some() {
                for buffer in [Buffer::SharedKeyLists, Buffer::DraftShareRows] {
                    handles.touch(setup.buffers.get(buffer));
                }
            }
            for &(span, ..) in &setup.views {
                handles.touch(span);
            }
            if let Some(cp) = &self.cp {
                for span in cp.ranks[rank].zeroed() {
                    handles.touch(span);
                }
            }
        }
        Ok(())
    }

    /// Every rank's launches of the layer's stages in `phase`, in program
    /// order, registering every span they bind.
    fn lower(
        &self,
        engines: &Engines,
        step: &StepSpec,
        inputs: &[StepInputs],
        handles: &Handles,
    ) -> LabResult<Vec<Vec<PreparedLaunch>>> {
        let (layer, phase) = (self.layer, step.phase);
        let records = match step.pass {
            None => records(SHAPE, self.kv(), phase)?,
            Some(pass) => drafter_records(SHAPE, self.kv(), phase, pass)?,
        };
        let stages = records
            .into_iter()
            .map(|record| record.stage)
            .filter(|stage| {
                matches!(stage,
                    Stage::LatentKvWrite(l) | Stage::IndexKeyWrite(l) | Stage::IndexSelect(l)
                    | Stage::LatentAttention(l) if *l == layer)
                    || *stage == Stage::DraftShareSelection
            })
            .collect::<Vec<_>>();
        // A draft step reuses the extend pass's selection, compacted.
        let expected = match step.pass {
            Some(DraftPass::Step) => vec![
                Stage::LatentKvWrite(layer),
                Stage::DraftShareSelection,
                Stage::LatentAttention(layer),
            ],
            _ => vec![
                Stage::LatentKvWrite(layer),
                Stage::IndexKeyWrite(layer),
                Stage::IndexSelect(layer),
                Stage::LatentAttention(layer),
            ],
        };
        if stages != expected {
            return Err(format!("layer {layer}'s stages are {stages:?}"));
        }
        // The ops an instance of the group prepares on its backend.
        let (backend, served) = sm120::backend(engines, SHAPE);
        let plan = Plan {
            shape: SHAPE,
            layout: engines.layout,
            index_topk: SHAPE.profile.index_topk_max,
            context_parallel: self.cp.as_ref().map_or(1, |cp| cp.size),
        };
        let ops =
            Ops::prepare(backend.as_ref(), &served, plan).map_err(|refused| refused.to_string())?;
        (0..RANKS)
            .map(|rank| {
                let setup = &self.ranks[rank];
                let lowering = Lowering {
                    shape: SHAPE,
                    format: &self.format,
                    buffers: &setup.buffers,
                    inputs: &inputs[rank].inputs,
                    ops: &ops,
                    rank: rank as u32,
                    index_topk: SHAPE.profile.index_topk_max,
                    pieces: &inputs[rank].pieces,
                    row_offset: 0,
                    peers: &self.peers,
                    head_rows: phase.rows(),
                    context_parallel: self.cp.as_ref(),
                    draft: step.pass,
                    verify_tokens: None,
                    projections: None,
                };
                let mut launches = Vec::new();
                for stage in stages.iter().copied() {
                    launches.extend(
                        lowering
                            .lower(phase, stage)?
                            .ok_or(format!("{stage:?} does not lower"))?,
                    );
                }
                for launch in &launches {
                    for span in bound_spans(launch)? {
                        handles.touch(span);
                    }
                }
                Ok(launches)
            })
            .collect()
    }

    /// The phase inputs of `step` as `PackedStep::pack_striped` builds
    /// them: in the step's item order, each item's logical table holding the
    /// stripes of its new length.
    fn phase_inputs(&self, step: &StepSpec) -> PhaseInputs {
        let stripe = PAGE * self.size;
        let prefill = matches!(step.phase, Phase::Prefill { .. });
        let verify = matches!(step.phase, Phase::Verify { .. });
        let mut phase = PhaseInputs {
            decode: !prefill,
            verify,
            ..PhaseInputs::default()
        };
        for (index, &(item, first, count)) in step.items.iter().enumerate() {
            let end = first + count;
            phase.item_order.push(index);
            phase.request_ids.push(item as u64);
            phase.emits_row.push(true);
            phase.prompt_rows.push(prefill);
            phase.anchors.push(false);
            phase.verify_rows.push(verify);
            phase.slot_ids.push(slot(item));
            phase.q_start.push(phase.token_ids.len() as i32);
            phase.q_len.push(count as i32);
            phase.prefix_len.push(first as i32);
            phase.seq_lens.push(end as i32);
            phase.last_positions.push(end as i64 - 1);
            phase
                .block_tables
                .push(self.tables[item][..end.div_ceil(stripe)].to_vec());
            for position in first..end {
                phase.token_ids.push(0);
                phase.positions.push(position as i32);
                phase.token_batch.push(index as i32);
            }
        }
        phase
    }

    /// The device table of `phase`: a row of `max_pages` entries per item,
    /// -1 past its pages.
    fn table(phase: &PhaseInputs) -> Vec<i32> {
        let max_pages = SHAPE.model_limits().max_pages() as usize;
        let mut rows = vec![-1; DECODE_BATCH as usize * max_pages];
        for (item, table) in phase.block_tables.iter().enumerate() {
            rows[item * max_pages..item * max_pages + table.len()].copy_from_slice(table);
        }
        rows
    }

    /// Every step's inputs ([`Self::step_inputs`]) over one ledger, as the
    /// instance keeps it: host placement under the declared state's rules,
    /// each step begun from its phase's items (`bundle_glm5next::step::items`)
    /// and committed with every row kept.
    fn steps_inputs(&self, steps: &[StepSpec]) -> LabResult<Vec<Vec<StepInputs>>> {
        let state = &self.ranks[0].buffers.state.state;
        let mut ledger = Ledger::new(DECODE_BATCH, Placement::Host, Rules::of(state), 1 << 20);
        steps
            .iter()
            .map(|step| {
                // The drafter's passes run inside a target step and open no
                // ledger step of their own.
                if step.pass.is_some() {
                    return self.step_inputs(step, None);
                }
                let phase = self.phase_inputs(step);
                let opened = ledger
                    .begin(&ledger_items(&phase), Vec::new())
                    .map_err(err)?
                    .clone();
                let inputs = self.step_inputs(step, Some(&opened))?;
                let kept = opened
                    .items
                    .iter()
                    .filter(|placed| placed.verify)
                    .map(|placed| (placed.request, placed.rows))
                    .collect::<Vec<_>>();
                ledger.commit(&kept).map_err(err)?;
                Ok(inputs)
            })
            .collect()
    }

    /// Every rank's inputs of `step`, as the instance uploads them
    /// (`instance/execute.rs`): the phase's arena packed by its layout and
    /// its table, an eager prefill's indexer pieces when it has more than
    /// one (each in its own region, its table rows from its first item's),
    /// a target verify phase's tail checkpoint table from
    /// `bundle_glm5next::step::side_inputs` over the open ledger step
    /// `opened`, checked against each item's slot's checkpoints, and a draft
    /// step's `DraftShareRows`.
    fn step_inputs(
        &self,
        step: &StepSpec,
        opened: Option<&LedgerStep>,
    ) -> LabResult<Vec<StepInputs>> {
        let phase = self.phase_inputs(step);
        let items = items();
        let layout = match step.phase {
            Phase::Prefill { .. } => InputLayout::prefill_within::<Model>(&phase, 0, items),
            Phase::Decode(_) => InputLayout::decode_within(items),
            Phase::Verify { .. } => InputLayout::verify_within::<Model>(items),
        };
        let packed = layout
            .pack(&phase, &[])
            .ok_or("the step does not fit its input layout")?;
        let table = i32_bytes(&Self::table(&phase));
        let pieces = match step.phase {
            Phase::Prefill { .. } => pieces(&phase, PREFILL_GRANULE)?,
            _ => Vec::new(),
        };
        if pieces.len() as u64 > MAX_PIECES {
            return Err(format!("{} indexer pieces", pieces.len()));
        }
        let row_bytes = 4 * u64::from(SHAPE.model_limits().max_pages());
        self.ranks
            .iter()
            .map(|setup| {
                let arena = match step.phase {
                    Phase::Prefill { .. } => setup.arenas.prefill,
                    Phase::Decode(_) => setup.arenas.decode,
                    Phase::Verify { .. } => setup.arenas.verify,
                };
                let mut uploads = vec![
                    (arena.arena, packed.clone()),
                    (arena.block_tables, table.clone()),
                ];
                let (piece_layout, stride) = piece_layout();
                let mut rank_pieces = Vec::new();
                if pieces.len() > 1 {
                    for (index, (first, piece)) in pieces.iter().enumerate() {
                        let item = phase
                            .item_order
                            .iter()
                            .position(|&order| order == piece.item_order[0])
                            .unwrap_or(0) as u64;
                        let region = DeviceBufferSpanV1::new(
                            setup.arenas.pieces.buffer,
                            setup.arenas.pieces.offset + index as u64 * stride,
                            piece_layout.end,
                        );
                        let tables = arena.block_tables;
                        let block_tables = DeviceBufferSpanV1::new(
                            tables.buffer,
                            tables.offset + item * row_bytes,
                            tables.length - item * row_bytes,
                        );
                        uploads.push((
                            region,
                            piece_layout
                                .pack(piece, &[])
                                .ok_or("a piece does not fit its layout")?,
                        ));
                        rank_pieces.push(Piece {
                            first: *first,
                            rows: piece.live_tokens(),
                            inputs: DeviceInputs::within(
                                PhaseArena {
                                    arena: region,
                                    block_tables,
                                },
                                &piece_layout,
                            ),
                        });
                    }
                }
                if step.pass == Some(DraftPass::Step) {
                    uploads.push((
                        setup.buffers.get(Buffer::DraftShareRows),
                        i32_bytes(&step.share),
                    ));
                }
                if let (Phase::Verify { .. }, None, Some(opened)) = (step.phase, step.pass, opened)
                {
                    let state = &setup.buffers.state;
                    let tails = state
                        .state
                        .buffers
                        .iter()
                        .find(|buffer| buffer.name == "indexer.tail")
                        .ok_or("no indexer tail")?;
                    let ids = step
                        .items
                        .iter()
                        .map(|&(item, _, count)| {
                            (0..count as u32)
                                .map(|row| {
                                    tails
                                        .checkpoint(state.extent, slot(item) as u32, row)
                                        .map(|entry| entry as i32)
                                        .ok_or(format!("item {item} row {row}: no checkpoint"))
                                })
                                .collect::<LabResult<Vec<_>>>()
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let stride = pooled_indexer_geometry(SHAPE).verify_rows;
                    let rows = step
                        .items
                        .iter()
                        .map(|&(_, _, count)| count as u32)
                        .collect::<Vec<_>>();
                    let side = side_inputs(
                        &state.state,
                        state.extent,
                        opened,
                        0,
                        &rows,
                        kda_checkpoint_stride(SHAPE)?,
                        stride,
                    )?;
                    // The serving table must name each row's checkpoint of
                    // its item's own (shuffled) slot.
                    let placed = opened.items.iter().map(|p| p.slot).collect::<Vec<_>>();
                    let slots = step
                        .items
                        .iter()
                        .map(|&(item, ..)| slot(item) as u32)
                        .collect::<Vec<_>>();
                    if placed != slots || side.tail_checkpoint_ids != checkpoint_table(stride, &ids)
                    {
                        return Err(format!(
                            "the tail checkpoint table {:?} is not each slot's {ids:?}",
                            side.tail_checkpoint_ids
                        ));
                    }
                    uploads.push((
                        setup.buffers.get(Buffer::TailCheckpointIds),
                        i32_bytes(&side.tail_checkpoint_ids),
                    ));
                }
                Ok(StepInputs {
                    inputs: DeviceInputs::within(arena, &layout),
                    pieces: rank_pieces,
                    uploads,
                })
            })
            .collect()
    }

    /// The pages group rank `rank` holds of item `item` at `tokens` tokens,
    /// as `(local page, physical page)`.
    fn rank_pages(
        &self,
        rank: usize,
        item: usize,
        tokens: usize,
    ) -> LabResult<Vec<(usize, usize)>> {
        let pages = ContextParallelPagesV1::new(self.size, PAGE, RANKS).map_err(err)?;
        Ok(pages
            .rank_pages(
                pages.group_rank(rank),
                self.tables[item].iter().map(|&page| page as usize),
                tokens,
            )
            .map_err(err)?
            .into_iter()
            .map(|page| (page.local_page, page.first_position / PAGE))
            .collect())
    }
}

/// The index of the first index key write among `calls`: each indexer
/// piece's key write follows the previous one's.
fn key_write_start(calls: &[Call], engines: &Engines) -> LabResult<usize> {
    let symbols = engines
        .indexer
        .iter()
        .filter(|entry| matches!(entry.kind, IndexerKind::KeyWrite(_)))
        .map(|entry| entry.symbol())
        .collect::<BTreeSet<_>>();
    calls
        .iter()
        .position(|call| symbols.contains(&call.symbol))
        .ok_or("no index key write".into())
}

/// The first sparse layer.
fn sparse_layer() -> u32 {
    (0..SHAPE.layers())
        .find(|&layer| SHAPE.sparse_ordinal(layer).is_some())
        .expect("a sparse layer")
}

// ----------------------------------------------------------- cpu check ----

/// The target steps include every decode bucket and every verify token
/// bucket, which [`the_layer_lowers`] lowers.
#[test]
fn the_target_steps_cover_every_decode_and_verify_bucket() {
    let target = steps();
    let buckets = |verify: bool| {
        target
            .iter()
            .filter_map(|step| match step.phase {
                Phase::Decode(rows) if !verify => Some(rows),
                Phase::Verify { rows, .. } if verify => Some(rows),
                _ => None,
            })
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(buckets(false), BTreeSet::from([1, 2, 4, 8, 16, 32]));
    assert_eq!(buckets(true), BTreeSet::from([2, 4, 8, 16, 32]));
}

/// Every rank's lowered layer at sizes 1, 2 and 4, for the target's
/// steps (prefill in three and two indexer pieces, every decode and verify
/// bucket) and the drafter's passes: the same symbols in the same order on
/// every rank of a group (the all-gathers pair up), `size + 3` group
/// all-gathers per indexer piece (the pairs, the queries, one per owner and
/// the barrier; a draft step selects nothing, so `size + 2`) and none at
/// size 1, every symbol embedded, every argument resolvable, a draft step
/// localizing `SharedKeyLists`, the oracle sampling every item's edge rows, the host tables covering each item's pages
/// once per group, and the backed windows fitting a GPU.
/// One test per scenario and KV format.
fn the_layer_lowers(scenario: Scenario, kv: LatentKvFormat) {
    let engines = Engines::new(SHAPE).unwrap();
    let steps = scenario.steps();
    let lengths = final_lengths(&steps);
    // The oracle gates every item's first and last rows of every step.
    for step in &steps {
        let rows = sampled(step, &mut Rng(0x5a));
        let mut row = 0;
        for &(item, _, count) in &step.items {
            assert!(
                rows.contains(&row) && rows.contains(&(row + count - 1)),
                "{scenario:?} {:?}: item {item}'s edge rows are not sampled",
                step.phase
            );
            row += count;
        }
    }
    for size in [1, 2, 4] {
        let label = format!("{scenario:?} {kv:?} size {size}");
        let handles = Handles::default();
        let instance = Instance::new(
            &handles,
            size,
            scenario.format(kv).unwrap(),
            scenario.layer(),
            &lengths,
            &mut Rng(0x5eed + size as u64),
        )
        .unwrap();
        // The continuation resumes item 2 at position 2178 across
        // its internal cut, with live tail slots; no other cut does.
        let cuts = steps
            .iter()
            .map(|step| live_tail_cuts(step, &instance.phase_inputs(step)).unwrap())
            .collect::<Vec<_>>()
            .concat();
        match scenario {
            Scenario::Target => assert_eq!(cuts, [(1, 2, 2178)], "{label}"),
            Scenario::Drafter => assert!(cuts.is_empty(), "{label}"),
        }
        let localize = model_glm5next_contract::sm120::context_parallel::entries(SHAPE)
            .into_iter()
            .find(|entry| {
                entry.kind == CpKind::LocalizeKeys { ranks: size as u32 }
                    && entry.geometry.list_width == list_width() as u32
            });
        let steps_inputs = instance.steps_inputs(&steps).unwrap();
        for (index, (step, inputs)) in steps.iter().zip(steps_inputs).enumerate() {
            let launches = instance.lower(&engines, step, &inputs, &handles).unwrap();
            // The first prefill's indexer runs in the bundle's three
            // pieces (a cut inside item 1, at a granule), the
            // continuation in two.
            let firsts = inputs[0].pieces.iter().map(|p| p.first).collect::<Vec<_>>();
            let pieces = match (scenario, index) {
                (Scenario::Target, 0) => vec![0, 2048, 4088],
                (Scenario::Target, 2) => vec![0, 2048],
                _ => vec![],
            };
            assert_eq!(firsts, pieces, "{label}");
            let symbols = launches
                .iter()
                .map(|rank| rank.iter().map(|l| l.symbol.clone()).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            assert!(symbols.iter().all(|s| *s == symbols[0]), "{label}");
            let count = |symbol: &str| symbols[0].iter().filter(|s| *s == symbol).count();
            // Per piece: the phased gathers of the pairs (not in a
            // draft step) and the queries and the phased exchange of
            // the partial blocks, then the barrier.
            let phased_per_piece = match step.pass {
                Some(DraftPass::Step) => 2,
                _ => 3,
            };
            let pieces = if size == 1 {
                0
            } else {
                inputs[0].pieces.len().max(1)
            };
            assert_eq!(
                (count(phased::SYMBOL), count(ALLGATHER_SYMBOL)),
                (pieces * phased_per_piece, pieces),
                "{label} {:?}",
                step.phase
            );
            for symbol in &symbols[0] {
                assert!(
                    bundle_glm5next::cubin_of(symbol).is_some(),
                    "{symbol} is not embedded"
                );
            }
            for (rank, launches) in launches.iter().enumerate() {
                for launch in launches {
                    Call::resolve(launch, &|span| Ok(span.offset)).unwrap();
                }
                // The attention localizes the lists the size-1
                // attention reads.
                if let Some(localize) = &localize {
                    let read = match step.pass {
                        Some(DraftPass::Step) => Buffer::SharedKeyLists,
                        _ => Buffer::KeyLists,
                    };
                    let lists = localize
                        .params()
                        .iter()
                        .position(|param| {
                            matches!(param, kernel_build::EntryParam::DevicePointer {
                                name, ..
                            } if name == "lists")
                        })
                        .unwrap();
                    let bound = launches
                        .iter()
                        .filter(|launch| launch.symbol == localize.symbol())
                        .map(|launch| {
                            dispatch::arguments(&launch.arguments, &launch.bindings).unwrap()[lists]
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    let own = instance.ranks[rank].buffers.get(read);
                    assert!(!bound.is_empty(), "{label}");
                    assert!(
                        bound.iter().all(|argument| matches!(argument,
                            KernelArgumentV1::DeviceBuffer(span)
                                if span.buffer == own.buffer)),
                        "{label} {:?} rank {rank}",
                        step.pass
                    );
                }
            }
        }
        for (item, &tokens) in lengths.iter().enumerate() {
            if tokens == 0 {
                continue;
            }
            for group in 0..RANKS / size {
                let mut physical = BTreeSet::new();
                for rank in group * size..(group + 1) * size {
                    let pages = instance.rank_pages(rank, item, tokens).unwrap();
                    let locals = pages.iter().map(|p| p.0).collect::<BTreeSet<_>>();
                    assert_eq!(locals.len(), pages.len());
                    for (_, page) in pages {
                        assert_eq!(page % size, rank % size);
                        assert!(physical.insert(page));
                    }
                }
                assert_eq!(physical.len(), tokens.div_ceil(PAGE));
            }
        }
        for bytes in handles.bytes_per_rank() {
            assert!(bytes < 6 << 30, "{label}: {bytes} bytes");
        }
    }
}

#[test]
fn the_layer_lowers_to_the_group_stage_sequence_target_fp8() {
    the_layer_lowers(Scenario::Target, LatentKvFormat::Fp8E4M3Block128);
}

#[test]
fn the_layer_lowers_to_the_group_stage_sequence_target_bf16() {
    the_layer_lowers(Scenario::Target, LatentKvFormat::Bf16);
}

#[test]
fn the_layer_lowers_to_the_group_stage_sequence_drafter_fp8() {
    the_layer_lowers(Scenario::Drafter, LatentKvFormat::Fp8E4M3Block128);
}

#[test]
fn the_layer_lowers_to_the_group_stage_sequence_drafter_bf16() {
    the_layer_lowers(Scenario::Drafter, LatentKvFormat::Bf16);
}

// ----------------------------------------------------------- gpu memory ----

/// Each touched handle's window, backed on its rank's GPU.
struct Memory {
    windows: BTreeMap<u32, Window>,
}

/// One handle's backing on its rank: the window of the handle from `start`,
/// `len` bytes, then, for a shadowed handle, a second `len` bytes the window
/// saves to and loads from on the device.
struct Window {
    rank: usize,
    buffer: DeviceBuffer,
    start: u64,
    len: usize,
}

impl Memory {
    /// Allocates every window, poisoned (shadowed ones twice over), fills
    /// each of `filled` in both halves with its pattern, then zeroes
    /// `zeroed`.
    fn new(
        sessions: &[Session],
        handles: &Handles,
        zeroed: &[DeviceBufferSpanV1],
        shadowed: &[DeviceBufferSpanV1],
        filled: &[(DeviceBufferSpanV1, &[u8])],
    ) -> LabResult<Self> {
        let shadows = shadowed
            .iter()
            .map(|span| span.buffer.slot())
            .collect::<BTreeSet<_>>();
        let mut windows = BTreeMap::new();
        for (index, &(rank, window)) in handles.slots.borrow().iter().enumerate() {
            let Some((start, end)) = window else {
                continue;
            };
            let slot = index as u32 + 1;
            let len = (end - start).max(16) as usize;
            let copies = if shadows.contains(&slot) { 2 } else { 1 };
            let session = &sessions[rank];
            let buffer = session.allocate(copies * len)?;
            kernel_lab::poison::poison(session, &[&buffer])?;
            windows.insert(
                slot,
                Window {
                    rank,
                    buffer,
                    start,
                    len,
                },
            );
        }
        let memory = Self { windows };
        for &(span, pattern) in filled {
            let (rank, window, offset) = memory.locate(span)?;
            let end = offset + span.length as usize;
            kernel_lab::poison::fill(
                &sessions[rank],
                &window.buffer,
                &[offset..end, window.len + offset..window.len + end],
                pattern,
            )?;
        }
        for &span in zeroed {
            let (rank, window, offset) = memory.locate(span)?;
            window
                .buffer
                .zero_at_async(offset, span.length as usize, &sessions[rank].stream)
                .map_err(err)?;
        }
        kernel_lab::poison::synchronize(sessions)?;
        Ok(memory)
    }

    /// Copies `span` into its window's shadow (`save`) or back (`load`), on
    /// its rank's stream.
    fn shadow(&self, sessions: &[Session], span: DeviceBufferSpanV1, save: bool) -> LabResult<()> {
        let (rank, window, offset) = self.locate(span)?;
        if window.buffer.byte_len() < 2 * window.len {
            return Err(format!("{span:?} has no shadow"));
        }
        let (from, to) = if save {
            (offset, window.len + offset)
        } else {
            (window.len + offset, offset)
        };
        window
            .buffer
            .copy_within_async(from, to, span.length as usize, &sessions[rank].stream)
            .map_err(err)
    }

    fn locate(&self, span: DeviceBufferSpanV1) -> LabResult<(usize, &Window, usize)> {
        let window = self
            .windows
            .get(&span.buffer.slot())
            .ok_or(format!("no window backs {span:?}"))?;
        if span.offset < window.start
            || span.offset + span.length > window.start + window.len as u64
        {
            return Err(format!("{span:?} lies outside its window"));
        }
        Ok((window.rank, window, (span.offset - window.start) as usize))
    }

    fn address(&self, span: DeviceBufferSpanV1) -> LabResult<u64> {
        let (_, window, offset) = self.locate(span)?;
        window
            .buffer
            .device_address_at(offset, span.length.max(1) as usize)
            .map_err(err)
    }

    fn write(
        &self,
        sessions: &[Session],
        span: DeviceBufferSpanV1,
        at: usize,
        bytes: &[u8],
    ) -> LabResult<()> {
        if (at + bytes.len()) as u64 > span.length {
            return Err(format!(
                "a write of {} bytes overruns {span:?}",
                bytes.len()
            ));
        }
        let (rank, window, offset) = self.locate(span)?;
        window
            .buffer
            .copy_from_bytes_at(offset + at, bytes, &sessions[rank].stream)
            .map_err(err)
    }

    fn read(
        &self,
        sessions: &[Session],
        span: DeviceBufferSpanV1,
        at: usize,
        len: usize,
    ) -> LabResult<Vec<u8>> {
        if (at + len) as u64 > span.length {
            return Err(format!("a read of {len} bytes overruns {span:?}"));
        }
        let (rank, window, offset) = self.locate(span)?;
        let session = &sessions[rank];
        let mut bytes = vec![0; len];
        window
            .buffer
            .copy_to_bytes_at_async(offset + at, &mut bytes, &session.stream)
            .map_err(err)?;
        session.stream.synchronize().map_err(err)?;
        Ok(bytes)
    }
}

/// Every embedded function the steps launch, on every rank.
struct Functions {
    _modules: Vec<BTreeMap<&'static str, Module>>,
    functions: Vec<BTreeMap<String, Function>>,
}

impl Functions {
    fn load(
        sessions: &[Session],
        engines: &Engines,
        symbols: &BTreeSet<String>,
    ) -> LabResult<Self> {
        let shared = engines.shared_bytes();
        let mut modules = Vec::new();
        let mut functions = Vec::new();
        for session in sessions {
            let mut rank_modules = BTreeMap::new();
            let mut rank_functions = BTreeMap::new();
            for symbol in symbols {
                let cubin = bundle_glm5next::cubin_of(symbol)
                    .ok_or(format!("no embedded cubin holds {symbol}"))?;
                if !rank_modules.contains_key(cubin) {
                    let bytes = bundle_glm5next::embedded_cubin(cubin)
                        .ok_or(format!("no embedded cubin {cubin}"))?;
                    rank_modules.insert(cubin, session.load_cubin(bytes)?);
                }
                let function = rank_modules[cubin].function(symbol).map_err(err)?;
                let bytes = shared.get(symbol).copied().unwrap_or(0);
                if bytes > 48 * 1024 {
                    function.set_max_dynamic_shared_memory(bytes).map_err(err)?;
                }
                rank_functions.insert(symbol.clone(), function);
            }
            modules.push(rank_modules);
            functions.push(rank_functions);
        }
        Ok(Self {
            _modules: modules,
            functions,
        })
    }

    /// Rank `rank`'s calls in order on its stream.
    fn submit(&self, sessions: &[Session], rank: usize, calls: &[Call]) -> LabResult<()> {
        for call in calls {
            self.functions[rank][&call.symbol]
                .launch(&sessions[rank].stream, call.config, &call.arguments())
                .map_err(|e| format!("rank {rank} {}: {e}", call.symbol))?;
        }
        Ok(())
    }
}

/// The lab's PDL producer probe on one rank (`probe_pdl_prefetch_sm120`, as
/// `context_parallel_comm.rs` runs it): it overwrites each staged input with
/// poison and the live row count with zero, triggers its dependents, and
/// only after its tail copies the inputs back and sets the live count, so a
/// chain kernel that reads either before its dependency wait sees poison.
struct Producer {
    _module: Module,
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
}

impl Producer {
    fn new(session: &Session) -> LabResult<Self> {
        let module = session.module("probe_pdl_prefetch_sm120")?;
        let function = module
            .function("chop_probe_pdl_prefetch_producer_v1")
            .map_err(err)?;
        let eviction = session.allocate(16)?;
        eviction
            .copy_from_bytes(&[0; 16], &session.stream)
            .map_err(err)?;
        Ok(Self {
            _module: module,
            function,
            eviction,
            checksum: session.allocate(128 * 4)?,
        })
    }

    /// Launches the producer over `operands` (input span, its bytes, NaN
    /// rather than zero poison), which it saves on the device, poisons and
    /// restores, ahead of `live`'s readers; returns the save buffers, to
    /// keep until the stream drains.
    fn launch(
        &self,
        session: &Session,
        memory: &Memory,
        operands: &[(DeviceBufferSpanV1, usize, bool)],
        live: DeviceBufferSpanV1,
        rows: u32,
    ) -> LabResult<Vec<DeviceBuffer>> {
        let mut staged = Vec::new();
        let mut descriptor = Vec::new();
        for &(span, bytes, floating) in operands {
            // The producer saves each input here before poisoning it.
            let saved = session.allocate(bytes.max(16))?;
            let input = DeviceBufferSpanV1::new(span.buffer, span.offset, bytes as u64);
            descriptor.extend_from_slice(
                &saved
                    .device_address_at(0, bytes)
                    .map_err(err)?
                    .to_ne_bytes(),
            );
            descriptor.extend_from_slice(&memory.address(input)?.to_ne_bytes());
            descriptor.extend_from_slice(&((bytes / 4) as u32).to_ne_bytes());
            // Bit 0: save the input first; bit 1: floating poison.
            descriptor.extend_from_slice(&(1 | u32::from(floating) << 1).to_ne_bytes());
            staged.push(saved);
        }
        let descriptors = session.allocate(descriptor.len())?;
        descriptors
            .copy_from_bytes(&descriptor, &session.stream)
            .map_err(err)?;
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
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(&descriptors),
                    KernelArgument::U32(operands.len() as u32),
                    KernelArgument::U64(memory.address(live)?),
                    KernelArgument::U32(rows),
                    KernelArgument::U64(50_000),
                ],
            )
            .map_err(err)?;
        staged.push(descriptors);
        Ok(staged)
    }
}

// -------------------------------------------------------------- a run ----

/// One instance's backed memory and resolved calls per step.
struct Run<'a> {
    instance: &'a Instance,
    memory: Memory,
    calls: Vec<Vec<Vec<Call>>>,
}

/// What one run of a step left: per rank its live key lists, its attention
/// output over the step's bucket, its layer pages ([`Run::read_layer`]) and
/// its tail.
struct Results {
    lists: Vec<Vec<i32>>,
    out: Vec<Vec<u16>>,
    pools: Vec<Vec<u8>>,
    tails: Vec<Vec<u8>>,
}

impl<'a> Run<'a> {
    fn new(
        sessions: &[Session],
        instance: &'a Instance,
        handles: &Handles,
        launches: &[Vec<Vec<PreparedLaunch>>],
    ) -> LabResult<Self> {
        let zeroed = instance
            .cp
            .iter()
            .flat_map(|cp| cp.ranks.iter().flat_map(RankState::zeroed))
            .chain(
                (0..RANKS)
                    .map(|rank| instance.tail(rank))
                    .collect::<LabResult<Vec<_>>>()?,
            )
            .collect::<Vec<_>>();
        // The pools and tails save and restore on the device; the outputs
        // poison from a shadow of their fill.
        let mut shadowed = Vec::new();
        let mut filled = Vec::new();
        for (rank, setup) in instance.ranks.iter().enumerate() {
            let buffers = &setup.buffers;
            shadowed.extend([
                buffers.kv_pool,
                instance.tail(rank)?,
                buffers.get(Buffer::AttentionOut),
                buffers.get(Buffer::KeyLists),
            ]);
            filled.push((buffers.get(Buffer::KeyLists), &LIST_FILL[..]));
            if buffers.drafter.is_some() {
                shadowed.push(buffers.get(Buffer::SharedKeyLists));
                filled.push((buffers.get(Buffer::SharedKeyLists), &LIST_FILL[..]));
            }
        }
        let memory = Memory::new(sessions, handles, &zeroed, &shadowed, &filled)?;
        let calls = launches
            .iter()
            .map(|step| {
                step.iter()
                    .map(|rank| {
                        rank.iter()
                            .map(|launch| Call::resolve(launch, &|span| memory.address(span)))
                            .collect::<LabResult<Vec<_>>>()
                    })
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        for setup in &instance.ranks {
            // The indexer weights, the same on every rank of every run: the
            // norm weight about 1, its bias and the position embedding
            // about 0.
            for &(span, dtype, index) in &setup.views {
                let mut weights = Rng(0x1d3 + index as u64);
                let elements = span.length as usize / dtype.bytes() as usize;
                let (center, scale) = if index == 0 { (1.0, 0.25) } else { (0.0, 0.5) };
                let values = weights
                    .bf16(elements, scale)
                    .into_iter()
                    .map(|bits| bf16_to_f32(bits) + center)
                    .collect::<Vec<_>>();
                let bytes = match dtype {
                    StoredType::F32 => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                    StoredType::Bf16 => {
                        u16_bytes(&values.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>())
                    }
                    other => return Err(format!("an indexer view of {other:?}")),
                };
                memory.write(sessions, span, 0, &bytes)?;
            }
        }
        Ok(Self {
            instance,
            memory,
            calls,
        })
    }

    /// Uploads a step's inputs and writes its operands on every rank.
    fn stage(&self, sessions: &[Session], data: &StepData, inputs: &[StepInputs]) -> LabResult<()> {
        for (rank, setup) in self.instance.ranks.iter().enumerate() {
            for (span, bytes) in &inputs[rank].uploads {
                self.memory.write(sessions, *span, 0, bytes)?;
            }
            let write = |span, bytes: Vec<u8>| self.memory.write(sessions, span, 0, &bytes);
            let buffers = &setup.buffers;
            write(buffers.get(Buffer::Latent), u16_bytes(&data.latent))?;
            write(buffers.get(Buffer::IndexKey), u16_bytes(&data.key))?;
            write(buffers.get(Buffer::IndexGate), u16_bytes(&data.gate))?;
            write(buffers.get(Buffer::IndexQuery), u16_bytes(&data.index_q))?;
            write(buffers.get(Buffer::IndexWeights), u16_bytes(&data.index_w))?;
            write(
                buffers.get(Buffer::QueryAbsorbed),
                u16_bytes(&data.queries[rank]),
            )?;
        }
        Ok(())
    }

    /// Rank `rank`'s inputs of `step` a PDL producer holds poisoned: the
    /// operand rows (BF16, NaN poison), the phase's positions and batch
    /// indices and a draft step's `DraftShareRows` (zero poison), each with
    /// its bytes.
    fn producer_operands(
        &self,
        rank: usize,
        step: &StepSpec,
        data: &StepData,
        inputs: &[StepInputs],
    ) -> Vec<(DeviceBufferSpanV1, usize, bool)> {
        let buffers = &self.instance.ranks[rank].buffers;
        let phase = &inputs[rank].inputs;
        let live = step.live();
        let mut operands = vec![
            (buffers.get(Buffer::Latent), 2 * data.latent.len(), true),
            (buffers.get(Buffer::IndexKey), 2 * data.key.len(), true),
            (buffers.get(Buffer::IndexGate), 2 * data.gate.len(), true),
            (
                buffers.get(Buffer::IndexQuery),
                2 * data.index_q.len(),
                true,
            ),
            (
                buffers.get(Buffer::IndexWeights),
                2 * data.index_w.len(),
                true,
            ),
            (
                buffers.get(Buffer::QueryAbsorbed),
                2 * data.queries[rank].len(),
                true,
            ),
            (phase.positions, 4 * live, false),
            (phase.token_batch, 4 * live, false),
        ];
        if step.pass == Some(DraftPass::Step) {
            operands.push((
                buffers.get(Buffer::DraftShareRows),
                4 * step.share.len(),
                false,
            ));
        }
        operands
    }

    /// Poisons the outputs over the step's bucket on every rank.
    fn poison_outputs(&self, sessions: &[Session], step: &StepSpec) -> LabResult<()> {
        let bucket = step.phase.rows() as usize;
        let rows = |span: DeviceBufferSpanV1, bytes: usize| {
            DeviceBufferSpanV1::new(span.buffer, span.offset, (bucket * bytes) as u64)
        };
        for setup in &self.instance.ranks {
            let lists = setup.buffers.get(lists_buffer(step));
            let out = setup.buffers.get(Buffer::AttentionOut);
            // From the shadows the fills left: LIST_FILL and POISON.
            self.memory
                .shadow(sessions, rows(lists, 4 * list_width()), false)?;
            self.memory
                .shadow(sessions, rows(out, out_row_bytes()), false)?;
        }
        kernel_lab::poison::synchronize(sessions)
    }

    /// Rank `rank`'s region of the layer in every local page, page after
    /// page: all the layer's kernels write of the pool.
    fn read_layer(&self, sessions: &[Session], rank: usize) -> LabResult<Vec<u8>> {
        let pool = self.instance.ranks[rank].buffers.kv_pool;
        let (offset, bytes) = self.instance.page_region();
        let page_bytes = self.instance.format.page_bytes as usize;
        let mut out = Vec::new();
        for page in 0..pool.length as usize / page_bytes {
            out.extend(
                self.memory
                    .read(sessions, pool, page * page_bytes + offset, bytes)?,
            );
        }
        Ok(out)
    }

    /// Saves (or, `restore`, loads back) every rank's pool and tail in
    /// their device shadows.
    fn state(&self, sessions: &[Session], restore: bool) -> LabResult<()> {
        for (rank, setup) in self.instance.ranks.iter().enumerate() {
            self.memory
                .shadow(sessions, setup.buffers.kv_pool, !restore)?;
            self.memory
                .shadow(sessions, self.instance.tail(rank)?, !restore)?;
        }
        kernel_lab::poison::synchronize(sessions)
    }

    fn submit(&self, sessions: &[Session], functions: &Functions, step: usize) -> LabResult<()> {
        for rank in 0..RANKS {
            functions.submit(sessions, rank, &self.calls[step][rank])?;
        }
        Ok(())
    }

    fn results(&self, sessions: &[Session], step: &StepSpec) -> LabResult<Results> {
        let bucket = step.phase.rows() as usize;
        let live = step.live();
        let mut results = Results {
            lists: Vec::new(),
            out: Vec::new(),
            pools: Vec::new(),
            tails: Vec::new(),
        };
        for (rank, setup) in self.instance.ranks.iter().enumerate() {
            let buffers = &setup.buffers;
            let lists = self.memory.read(
                sessions,
                buffers.get(lists_buffer(step)),
                0,
                bucket * list_width() * 4,
            )?;
            let live_bytes = live * list_width() * 4;
            if lists[live_bytes..]
                .chunks_exact(4)
                .any(|word| word != LIST_FILL)
            {
                return Err(format!(
                    "rank {rank}: a key list row past the live count changed"
                ));
            }
            results.lists.push(i32s(&lists[..live_bytes]));
            let out = self.memory.read(
                sessions,
                buffers.get(Buffer::AttentionOut),
                0,
                bucket * out_row_bytes(),
            )?;
            if out[live * out_row_bytes()..].iter().any(|&b| b != POISON) {
                return Err(format!(
                    "rank {rank}: an attention row past the live count changed"
                ));
            }
            let out = u16s(&out[..live * out_row_bytes()]);
            if let Some(i) = out.iter().position(|&x| x == 0xffff) {
                return Err(format!("rank {rank}: attention element {i} still poisoned"));
            }
            results.out.push(out);
            results.pools.push(self.read_layer(sessions, rank)?);
            let tail = self.instance.tail(rank)?;
            results
                .tails
                .push(self.memory.read(sessions, tail, 0, tail.length as usize)?);
        }
        Ok(results)
    }
}

/// The global key lists `step` leaves: a draft step's shared lists, else
/// the selection's.
fn lists_buffer(step: &StepSpec) -> Buffer {
    match step.pass {
        Some(DraftPass::Step) => Buffer::SharedKeyLists,
        _ => Buffer::KeyLists,
    }
}

fn list_width() -> usize {
    SHAPE.latent_keys() as usize
}

fn out_row_bytes() -> usize {
    let p = SHAPE.profile;
    (p.latent_heads * p.kv_rank) as usize * 2
}

// --------------------------------------------------------------- checks ----

/// The size-1 run writes each appended latent row and each pool the step
/// completes: no element of them is left poisoned. `pool` is the layer's
/// pages ([`Run::read_layer`]).
fn check_writes(reference: &Instance, pool: &[u8], step: &StepSpec) -> LabResult<()> {
    let kv = reference.kv();
    let (_, region) = reference.page_region();
    let latent = SHAPE.profile.kv_rank as usize;
    let element = match kv {
        LatentKvFormat::Bf16 => 2,
        LatentKvFormat::Fp8E4M3Block128 => 1,
        LatentKvFormat::Fp8E4M3Ue8M0Block32 | LatentKvFormat::Fp4E2M1E4M3Block16 => {
            unreachable!("native latent formats need their own conformance cases")
        }
    };
    let latent_page = SHAPE.latent_page_bytes(kv) as usize;
    let group = SHAPE.profile.index_pool as usize;
    let key_bytes = SHAPE.profile.indexer_dim as usize * 2;
    for (item, position) in step.rows() {
        let page = reference.tables[item][position / PAGE] as usize;
        let base = page * region;
        let row = base + position % PAGE * latent * element;
        if pool[row..row + latent * element]
            .chunks_exact(element)
            .any(|e| e.iter().all(|&b| b == POISON))
        {
            return Err(format!(
                "item {item} position {position}: latent row unwritten"
            ));
        }
        // A draft step writes no indexer keys.
        if position % group == group - 1 && step.pass != Some(DraftPass::Step) {
            let key = base + latent_page + position % PAGE / group * key_bytes;
            if pool[key..key + key_bytes]
                .chunks_exact(2)
                .any(|e| e == [POISON, POISON])
            {
                return Err(format!(
                    "item {item} pool {}: pooled key unwritten",
                    position / group
                ));
            }
        }
    }
    Ok(())
}

/// Each page a group rank owns equals the size-1 page of the same tokens
/// over the layer's page region, every other local page stays poisoned,
/// and the tails are equal.
fn check_state(
    label: &str,
    reference: &Instance,
    expected: &Results,
    instance: &Instance,
    actual: &Results,
    lengths: &[usize; ITEMS],
) -> LabResult<()> {
    let (_, bytes) = reference.page_region();
    let region = |pool: &[u8], page: usize| pool[page * bytes..(page + 1) * bytes].to_vec();
    for rank in 0..RANKS {
        let pool = &actual.pools[rank];
        let mut owned = BTreeSet::new();
        for (item, &tokens) in lengths.iter().enumerate() {
            for (local, physical) in instance.rank_pages(rank, item, tokens)? {
                owned.insert(local);
                let reference_page = reference.tables[item][physical] as usize;
                if region(pool, local) != region(&expected.pools[0], reference_page) {
                    return Err(format!(
                        "{label} rank {rank} item {item} page {physical} (local {local}) \
                         differs from the size-1 page {reference_page}"
                    ));
                }
            }
        }
        for local in 0..pool.len() / bytes {
            if !owned.contains(&local) && region(pool, local).iter().any(|&b| b != POISON) {
                return Err(format!(
                    "{label} rank {rank}: unowned local page {local} written"
                ));
            }
        }
        if actual.tails[rank] != expected.tails[rank] {
            return Err(format!("{label} rank {rank}: the indexer tail differs"));
        }
    }
    Ok(())
}

/// A draft step's shared lists are, per item, the key list of that item's
/// last row of the preceding extend pass (`extend`, whose lists `saved`
/// are), found here from the extend pass's items, not from the step's
/// `DraftShareRows`.
fn check_shared(
    step: &StepSpec,
    extend: &StepSpec,
    saved: &[Vec<i32>],
    shared: &[Vec<i32>],
) -> LabResult<()> {
    let width = list_width();
    let mut last = BTreeMap::new();
    let mut end = 0;
    for &(item, _, count) in &extend.items {
        end += count;
        last.insert(item, end - 1);
    }
    for (row, &(item, ..)) in step.items.iter().enumerate() {
        let source = *last
            .get(&item)
            .ok_or(format!("item {item} has no extend row"))?;
        for rank in 0..RANKS {
            if shared[rank][row * width..(row + 1) * width]
                != saved[rank][source * width..(source + 1) * width]
            {
                return Err(format!(
                    "rank {rank}: draft row {row} (item {item}) is not the extend pass's \
                     row {source}"
                ));
            }
        }
    }
    Ok(())
}

/// Ordered distance of two BF16 values in ulps.
fn ulps(a: u16, b: u16) -> u32 {
    let key = |x: u16| {
        if x & 0x8000 != 0 {
            -i32::from(x & 0x7fff)
        } else {
            i32::from(x)
        }
    };
    key(a).abs_diff(key(b))
}

/// Random rows the oracle samples per step and rank beyond the items' edge
/// rows.
const RANDOM_ROWS: usize = 8;

/// Most threads the oracle runs at once: the oracle bounds the test's host
/// time, and the host is shared.
const ORACLE_THREADS: usize = 16;

/// The rows the oracle samples: every item's first and last rows, then
/// [`RANDOM_ROWS`] random rows (all rows of a step with no more).
fn sampled(step: &StepSpec, rng: &mut Rng) -> Vec<usize> {
    let live = step.live();
    let mut rows = BTreeSet::new();
    let mut row = 0;
    for &(_, _, count) in &step.items {
        rows.extend([row, row + count - 1]);
        row += count;
    }
    let wanted = (rows.len() + RANDOM_ROWS).min(live);
    while rows.len() < wanted {
        rows.insert(rng.below(live));
    }
    rows.into_iter().collect()
}

/// The normalized attention entry's query of `kv` in `phase`.
fn attention_query(
    engines: &Engines,
    kv: LatentKvFormat,
    phase: Phase,
) -> LabResult<LatentAttentionQuery> {
    engines
        .latent
        .iter()
        .find_map(|entry| match entry.query {
            latent::Query::Latent(
                query @ LatentAttentionQuery::Attention {
                    kv: k,
                    phase: p,
                    output: LatentAttentionOutput::Normalized,
                    ..
                },
            ) if k == kv && p == phase.attention() => Some(query),
            _ => None,
        })
        .ok_or(format!("no normalized attention of {kv:?}"))
}

/// The f64 reference of every rank's sampled rows over the size-1 pool,
/// tables and key lists, each rank's rows in order: (rank, row) tasks over
/// at most [`ORACLE_THREADS`] threads, since the oracle dominates the
/// test's host time.
fn reference_outputs(
    engines: &Engines,
    reference: &Instance,
    results: &Results,
    (step, data): (&StepSpec, &StepData),
    rows: &[usize],
) -> LabResult<Vec<OracleOutput>> {
    let p = SHAPE.profile;
    let width = (p.latent_heads * p.kv_rank) as usize;
    let lists_width = list_width();
    let query = attention_query(engines, reference.kv(), step.phase)?;
    let tables = Instance::table(&reference.phase_inputs(step));
    let (_, region) = reference.page_region();
    let one = |rank: usize, row: usize| -> LabResult<OracleOutput> {
        let q = data.queries[rank][row * width..(row + 1) * width]
            .iter()
            .map(|&bits| f64::from(bf16_to_f32(bits)))
            .collect::<Vec<_>>();
        let result = latent_attention(
            query,
            LatentAttentionInput {
                q: &q,
                row_batch: &[data.batch[row]],
                key_lists: &results.lists[rank][row * lists_width..(row + 1) * lists_width],
                budget: (p.index_topk_max + p.index_pool - 1) as usize,
                paging: LatentPaging {
                    block_tables: &tables,
                    max_pages: SHAPE.model_limits().max_pages() as usize,
                    page_stride: region,
                },
                pool: &results.pools[rank],
            },
        )
        .map_err(err)?;
        match result {
            LatentAttentionResult::Normalized(output) => Ok(output),
            LatentAttentionResult::Partial(_) => Err("a partial reference".into()),
        }
    };
    let tasks = (0..RANKS)
        .flat_map(|rank| rows.iter().map(move |&row| (rank, row)))
        .collect::<Vec<_>>();
    // At most `ORACLE_THREADS` threads take the next task from a shared
    // counter, so rows of many keys do not queue behind one thread.
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(ORACLE_THREADS)
        .min(tasks.len().max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::thread::scope(|scope| {
        let (one, next, tasks) = (&one, &next, &tasks);
        (0..threads)
            .map(|_| {
                scope.spawn(move || {
                    let mut done = Vec::new();
                    loop {
                        let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(rank, row)) = tasks.get(index) else {
                            return done;
                        };
                        done.push((index, one(rank, row)));
                    }
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().expect("an oracle thread"))
            .collect::<BTreeMap<_, _>>()
    });
    let outputs = done.into_values().collect::<LabResult<Vec<_>>>()?;
    Ok(outputs
        .chunks(rows.len())
        .map(|rank| OracleOutput {
            values: rank.iter().flat_map(|o| o.values.iter().copied()).collect(),
            sum_abs_products: rank
                .iter()
                .flat_map(|o| o.sum_abs_products.iter().copied())
                .collect(),
            reduction_lengths: rank
                .iter()
                .flat_map(|o| o.reduction_lengths.iter().copied())
                .collect(),
        })
        .collect())
}

/// The gate of `out`'s sampled rows: its largest error over the bound, or
/// the failure.
fn gate(out: &[u16], rows: &[usize], reference: &OracleOutput) -> Result<f64, String> {
    let width = out_row_bytes() / 2;
    let actual = rows
        .iter()
        .flat_map(|&r| out[r * width..(r + 1) * width].iter())
        .map(|&bits| f64::from(bf16_to_f32(bits)))
        .collect::<Vec<_>>();
    check_numeric(&actual, reference, DType::Bf16).map_err(err)?;
    Ok(actual
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let error = (a - reference.values[i]).abs();
            if error == 0.0 {
                0.0
            } else {
                error / numeric_bound(reference, i, DType::Bf16)
            }
        })
        .fold(0.0, f64::max))
}

// ------------------------------------------------------------- the test ----

/// One format at one group size: the size-1 layer and the group's, step
/// by step.
fn run_size(
    sessions: &[Session],
    engines: &Engines,
    scenario: Scenario,
    kv: LatentKvFormat,
    size: usize,
    report: &mut Vec<Value>,
) -> LabResult<()> {
    let steps = scenario.steps();
    let lengths = final_lengths(&steps);
    let layer = scenario.layer();
    let format = scenario.format(kv)?;
    // Each instance's own handles, so each run backs only its own windows.
    let handles = [Handles::default(), Handles::default()];
    let reference = Instance::new(&handles[0], 1, format, layer, &lengths, &mut Rng(0x11))?;
    let group = Instance::new(
        &handles[1],
        size,
        format,
        layer,
        &lengths,
        &mut Rng(0x22 + size as u64),
    )?;
    let reference_inputs = reference.steps_inputs(&steps)?;
    let group_inputs = group.steps_inputs(&steps)?;
    let lower = |instance: &Instance, inputs: &[Vec<StepInputs>], handles: &Handles| {
        steps
            .iter()
            .zip(inputs)
            .map(|(step, inputs)| instance.lower(engines, step, inputs, handles))
            .collect::<LabResult<Vec<_>>>()
    };
    let reference_launches = lower(&reference, &reference_inputs, &handles[0])?;
    let group_launches = lower(&group, &group_inputs, &handles[1])?;
    let symbols = reference_launches
        .iter()
        .chain(&group_launches)
        .flatten()
        .flatten()
        .map(|launch| launch.symbol.clone())
        .collect::<BTreeSet<_>>();
    let functions = Functions::load(sessions, engines, &symbols)?;
    let reference = Run::new(sessions, &reference, &handles[0], &reference_launches)?;
    let group = Run::new(sessions, &group, &handles[1], &group_launches)?;
    let producers = sessions
        .iter()
        .map(Producer::new)
        .collect::<LabResult<Vec<_>>>()?;
    let mut rng = Rng(0xd11_0000 + size as u64);
    let label = format!("{scenario:?} {kv:?} size {size}");
    // The last extend pass's items and size-1 key lists, per rank.
    let mut extended: Option<(&StepSpec, Vec<Vec<i32>>)> = None;
    for (index, step) in steps.iter().enumerate() {
        let label = format!("{label} step {index} {:?}", step.phase);
        let data = StepData::new(step, &mut rng);
        reference.stage(sessions, &data, &reference_inputs[index])?;
        group.stage(sessions, &data, &group_inputs[index])?;
        // The size-1 layer, once.
        reference.poison_outputs(sessions, step)?;
        kernel_lab::poison::synchronize(sessions)?;
        reference.submit(sessions, &functions, index)?;
        kernel_lab::poison::synchronize(sessions)?;
        let expected = reference.results(sessions, step)?;
        check_writes(reference.instance, &expected.pools[0], step)
            .map_err(|e| format!("{label} size 1: {e}"))?;
        match step.pass {
            Some(DraftPass::Extend) => extended = Some((step, expected.lists.clone())),
            Some(DraftPass::Step) => {
                let (extend, lists) = extended.as_ref().ok_or("a draft step before any extend")?;
                check_shared(step, extend, lists, &expected.lists)
                    .map_err(|e| format!("{label} size 1: {e}"))?;
            }
            None => {}
        }
        let rows = sampled(step, &mut rng);
        let references =
            reference_outputs(engines, reference.instance, &expected, (step, &data), &rows)?;
        // The group's layer: eager, relaunched, then twice from one graph,
        // each from the same state on poisoned outputs.
        group.state(sessions, false)?;
        // The tail an item carries across an internal piece cut matters: the
        // group's layer with that tail zeroed at the cut must not match.
        for (piece, item, position) in live_tail_cuts(step, &group.instance.phase_inputs(step))? {
            let label = format!("{label} tail zeroed at piece {piece}");
            let cut = key_write_start(&group.calls[index][0], engines)? + piece;
            kernel_lab::poison::synchronize(sessions)?;
            group.state(sessions, true)?;
            group.poison_outputs(sessions, step)?;
            kernel_lab::poison::synchronize(sessions)?;
            for rank in 0..RANKS {
                functions.submit(sessions, rank, &group.calls[index][rank][..cut])?;
            }
            kernel_lab::poison::synchronize(sessions)?;
            for rank in 0..RANKS {
                let entry = group
                    .instance
                    .tail_entry(rank, slot(item as usize) as u64)?;
                group
                    .memory
                    .write(sessions, entry, 0, &vec![0; entry.length as usize])?;
            }
            for rank in 0..RANKS {
                functions.submit(sessions, rank, &group.calls[index][rank][cut..])?;
            }
            kernel_lab::poison::synchronize(sessions)?;
            let detected = match group.results(sessions, step) {
                Err(_) => true,
                Ok(actual) => {
                    (0..RANKS).any(|rank| actual.lists[rank] != expected.lists[rank])
                        || check_state(
                            &label,
                            reference.instance,
                            &expected,
                            group.instance,
                            &actual,
                            &lengths,
                        )
                        .is_err()
                }
            };
            if !detected {
                return Err(format!(
                    "{label}: item {item}'s tail carried to position {position} was zeroed \
                     and the layer still matched size 1"
                ));
            }
            println!("{label}: item {item} at position {position}, detected");
        }
        let mut first: Option<Vec<Vec<u16>>> = None;
        let mut graphs = None;
        for (call, mode) in ["eager", "relaunch", "graph", "graph replay", "PDL"]
            .iter()
            .enumerate()
        {
            let label = format!("{label} {mode}");
            kernel_lab::poison::synchronize(sessions)?;
            group.state(sessions, true)?;
            group.poison_outputs(sessions, step)?;
            kernel_lab::poison::synchronize(sessions)?;
            if call < 2 {
                group.submit(sessions, &functions, index)?;
            } else if call == 4 {
                // Behind each rank's producer, which holds the step's
                // inputs poisoned until after its tail.
                let mut staged = Vec::new();
                for (rank, producer) in producers.iter().enumerate() {
                    let operands = group.producer_operands(rank, step, &data, &group_inputs[index]);
                    staged.push(producer.launch(
                        &sessions[rank],
                        &group.memory,
                        &operands,
                        group_inputs[index][rank].inputs.live_tokens,
                        step.live() as u32,
                    )?);
                    functions.submit(sessions, rank, &group.calls[index][rank])?;
                }
                kernel_lab::poison::synchronize(sessions)?;
                drop(staged);
            } else {
                if graphs.is_none() {
                    graphs = Some(RankReplay::new(sessions, |rank, _| {
                        functions.submit(sessions, rank, &group.calls[index][rank])
                    })?);
                }
                graphs.as_ref().expect("captured").launch(sessions)?;
            }
            kernel_lab::poison::synchronize(sessions)?;
            let actual = group
                .results(sessions, step)
                .map_err(|e| format!("{label}: {e}"))?;
            for rank in 0..RANKS {
                if actual.lists[rank] != expected.lists[rank] {
                    let i = actual.lists[rank]
                        .iter()
                        .zip(&expected.lists[rank])
                        .position(|(a, e)| a != e)
                        .unwrap_or(0);
                    return Err(format!(
                        "{label} rank {rank}: key list entry {i} (row {}) differs from size 1",
                        i / list_width()
                    ));
                }
            }
            check_state(
                &label,
                reference.instance,
                &expected,
                group.instance,
                &actual,
                &lengths,
            )?;
            match &first {
                Some(first) => {
                    if *first != actual.out {
                        return Err(format!("{label}: the attention output differs from eager"));
                    }
                }
                None => {
                    for (rank, oracle) in references.iter().enumerate() {
                        let group_gate = gate(&actual.out[rank], &rows, oracle)
                            .map_err(|e| format!("{label} rank {rank} size {size}: {e}"))?;
                        let reference_gate = gate(&expected.out[rank], &rows, oracle)
                            .map_err(|e| format!("{label} rank {rank} size 1: {e}"))?;
                        let differing = actual.out[rank]
                            .iter()
                            .zip(&expected.out[rank])
                            .filter(|(a, e)| a != e)
                            .count();
                        let max_ulps = actual.out[rank]
                            .iter()
                            .zip(&expected.out[rank])
                            .map(|(&a, &e)| ulps(a, e))
                            .max()
                            .unwrap_or(0);
                        report.push(json!({
                            "scenario": format!("{scenario:?}"),
                            "kv": format!("{kv:?}"),
                            "size": size,
                            "step": index,
                            "phase": format!("{:?}", step.phase),
                            "rank": rank,
                            "sampled_rows": rows.len(),
                            "group_max_error_over_bound": group_gate,
                            "size1_max_error_over_bound": reference_gate,
                            "differing_elements": differing,
                            "elements": actual.out[rank].len(),
                            "max_bf16_ulps": max_ulps,
                        }));
                        println!(
                            "{label} rank {rank}: gates {group_gate:.3} (size {size}) \
                             {reference_gate:.3} (size 1), {differing} of {} elements differ, \
                             at most {max_ulps} BF16 ulps",
                            actual.out[rank].len()
                        );
                    }
                    first = Some(actual.out);
                }
            }
        }
    }
    Ok(())
}

/// `scenario`'s layer at context-parallel sizes 2 (two pairs) and 4 against
/// size 1 for each of `formats`, with a receipt named after `kind`.
fn run_cases(scenario: Scenario, formats: &[LatentKvFormat], kind: &str) {
    gpu_test_multi(8 << 30, |sessions| {
        if sessions.len() != RANKS {
            return Err(format!(
                "the TP4 layer runs on four GPUs, not {}",
                sessions.len()
            ));
        }
        let engines = Engines::new(SHAPE)?;
        let mut report = Vec::new();
        let mut failures = Vec::new();
        for &kv in formats {
            for size in [2, 4] {
                let started = std::time::Instant::now();
                if let Err(error) = run_size(sessions, &engines, scenario, kv, size, &mut report) {
                    failures.push(error);
                }
                println!(
                    "{scenario:?} {kv:?} size {size}: {:.1} s",
                    started.elapsed().as_secs_f64()
                );
            }
        }
        let receipt = sessions[0].receipt(
            &format!("glm5next_context_parallel_chain_{kind}"),
            BTreeMap::from([
                ("cases".to_owned(), json!(report)),
                ("failures".to_owned(), json!(failures)),
            ]),
        )?;
        println!("receipt: {}", receipt.display());
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("group layers match the size-1 layer");
}

/// The target's layer with FP8 KV at sizes 2 and 4 against size 1, over
/// prefill (in pieces), decode and verify steps: selection and KV bit for
/// bit, attention within the oracle's gate, through eager, relaunched,
/// graph-replayed and PDL-producer runs on poisoned outputs.
#[test]
fn target_fp8_layers_match_the_size_one_layer() {
    run_cases(
        Scenario::Target,
        &[LatentKvFormat::Fp8E4M3Block128],
        "target_fp8",
    );
}

/// [`target_fp8_layers_match_the_size_one_layer`] with BF16 KV.
#[test]
fn target_bf16_layers_match_the_size_one_layer() {
    run_cases(Scenario::Target, &[LatentKvFormat::Bf16], "target_bf16");
}

/// The MTP drafter's extend and draft-step passes on its layer, FP8 and
/// BF16 KV, at sizes 2 and 4 against size 1.
#[test]
fn drafter_layers_match_the_size_one_layer() {
    run_cases(
        Scenario::Drafter,
        &[LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16],
        "drafter",
    );
}
