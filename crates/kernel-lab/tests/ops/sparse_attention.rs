//! Q-5b: the QSA indexer and block-sparse attention against the
//! Q-4 f64 oracles. Each case writes every sequence's history into the indexer
//! caches with chunked key writes, then runs one step: key write, indexer
//! query, block scores, top-k selection and sparse attention. Block keys and
//! queries are checked against the oracle, scores and attention against the
//! oracle evaluated on the kernels' own inputs, and the selection bit for bit
//! against the oracle's top-k of the kernel scores. A focused test checks
//! block-key pooling under cancellation. Run in release with
//! `CHOPINFER_KERNEL_GPU=<ordinal>`; `CHOPINFER_QSA_CASE=<name,...>` selects
//! cases and `CHOPINFER_QSA_BENCH=1` also times every stage (benchmark windows
//! only). `CHOPINFER_QWEN38_RANKS=2` runs TP2 attention (12 query heads)
//! from its own cubin; the indexer entries are
//! TP4's at every topology. Those lean runs (`kernel_lab::qwen38_lean`) take
//! only `LEAN_CASES`, small contexts that launch each attention entry and
//! each dtype's split merge, and mark their receipts `"lean": true`.
//! `qsa_wide_decode_matches_oracles` runs the same step for 33 to 256 decode
//! rows in the bundle's graph buckets of 64, 128 and 256 rows, whose rows past
//! the live count must stay untouched.

use std::collections::BTreeMap;
use std::fs;

use crate::pool_offsets::{self, SentinelPool};
use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{AttentionPhase, DType, SparseAttentionQuery};
use kernel_core::gated_attention_oracle::mrope_axis;
use kernel_core::kv_fp8;
use kernel_core::oracle::accumulation_bound;
use kernel_core::sparse_attention_oracle::{self as qsa, IndexerKeyCache, SparseKv};
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session, check_output, gpu_test, launch, time};
use kernels_sm120::Profile;
use kernels_sm120::attention::sparse::{Entry, Geometry, Kind, catalog};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module, Stream};
use serde_json::{Value, json};

#[path = "sparse_invariance.rs"]
pub(crate) mod invariance;
#[path = "qsa_live.rs"]
mod live;
#[path = "qsa_grid.rs"]
mod qsa_grid;

const INDEX_HEADS: usize = 4;
const INDEX_DIM: usize = 128;
const WIDTH: usize = (INDEX_HEADS + 1) * INDEX_DIM;
const DIM: usize = 256;
// Fixed-profile probes use the production contract. Shared fixtures below
// take these dimensions from their topology instead.
const PAGE: usize = model_qwen38_contract::sm120::attention::SPARSE.page_size as usize;
const BLOCK: usize = model_qwen38_contract::sm120::attention::SPARSE.block as usize;
const TOP: usize = model_qwen38_contract::sm120::attention::SPARSE.top_blocks as usize;
const RING: usize = model_qwen38_contract::sm120::attention::SPARSE.ring as usize;
/// FP8 K and V scales; they differ, so a swapped pair cannot pass.
const FP8_SCALES: [f32; 2] = [0.5, 0.25];
/// Block-key page stride: one page's 4 KiB of block keys plus a 1 KiB gap
/// that must stay untouched.
const BLOCK_PAGE_STRIDE: u64 = 5120;
/// Rows of one history key write launch.
const HISTORY_ROWS: usize = 8191;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    Random,
    /// Most raw keys repeat one vector and the queries' rotary halves are
    /// zero, so most blocks score exactly the same.
    Ties,
    /// Zero indexer queries: every block scores +0.
    ZeroQuery,
}

#[derive(Clone, Debug)]
struct Case {
    name: String,
    phase: AttentionPhase,
    fp8: bool,
    /// FP8 K and V rounded from general BF16 values, also
    /// checked against the attention of the unrounded values.
    rounded: bool,
    /// Per sequence: cached tokens and rows this step.
    sequences: Vec<(usize, usize)>,
    pattern: Pattern,
}

/// The lean run's cases: each dtype's decode (buckets of 8 rows, 64 splits
/// and the merge), verify (256 rows, 2 splits, then the rollback decode) and
/// prefill (2048 rows, no split), over contexts of at most 8,194 tokens that
/// span many pages and more than `TOP` blocks, so the selection matters.
/// `decode_mixed_b8` spans contexts 1 to 8,192 around the dense limit,
/// `verify_c8194_b8` the rollback's ring edge, `prefill_mixed` two
/// sequences with dense and sparse rows, and the FP8 prefill the rounded
/// values checked against the BF16 attention.
const LEAN_CASES: [&str; 6] = [
    "decode_mixed_b8",
    "decode_fp8_c8192_b8",
    "verify_c8194_b8",
    "verify_fp8_c8192_b8",
    "prefill_mixed",
    "prefill_fp8_rounded_c8192",
];

/// The cases of a run: all of them, or `LEAN_CASES` when `lean`.
fn run_cases(lean: bool) -> Vec<Case> {
    cases()
        .into_iter()
        .filter(|case| !lean || LEAN_CASES.contains(&case.name.as_str()))
        .collect()
}

/// Live rows of the wide decode cases (Qwen above 32). A count that is not a
/// power of two runs in the next power-of-two bucket, as the bundle's decode
/// graphs do: 33, 100 and 255 rows leave 31, 28 and 1 rows past the live
/// count, and the buckets split each row's attention 8, 4 and 2 ways.
const WIDE_ROWS: [usize; 6] = [33, 64, 100, 128, 255, 256];

/// Contexts the wide decode rows cycle through: mostly short rows, the two
/// sides of the dense limit (2,051 and 2,052 tokens) and sparse rows of
/// 4,097 and 8,192 tokens. The middle row sees 65,536 tokens instead.
const WIDE_CONTEXTS: [usize; 16] = [
    1, 17, 2051, 5, 130, 2052, 33, 9, 8192, 64, 2, 257, 65, 3, 4097, 4,
];

/// The wide cases of lean runs: one partial and one full bucket.
const WIDE_LEAN_CASES: [&str; 2] = ["decode_wide_b100", "decode_fp8_wide_b256"];

/// The wide decode cases of both KV dtypes at every `WIDE_ROWS` count, or
/// `WIDE_LEAN_CASES` when `lean`.
fn wide_cases(lean: bool) -> Vec<Case> {
    let mut cases = Vec::new();
    for fp8 in [false, true] {
        for rows in WIDE_ROWS {
            let name = format!("decode{}_wide_b{rows}", if fp8 { "_fp8" } else { "" });
            if lean && !WIDE_LEAN_CASES.contains(&name.as_str()) {
                continue;
            }
            let context = |row: usize| {
                if row == rows / 2 {
                    65_536
                } else {
                    WIDE_CONTEXTS[row % WIDE_CONTEXTS.len()]
                }
            };
            cases.push(Case {
                name,
                phase: AttentionPhase::Decode,
                fp8,
                rounded: false,
                sequences: (0..rows).map(|row| (context(row) - 1, 1)).collect(),
                pattern: Pattern::Random,
            });
        }
    }
    cases
}

/// The prepared row bucket of a step of `rows` rows: for decode the bundle's
/// graph bucket, past 32 rows the next power of two up to 256.
fn prepared_rows(phase: AttentionPhase, rows: usize) -> u32 {
    match phase {
        AttentionPhase::Decode => [1, 8, 32, 64, 128, 256]
            .into_iter()
            .find(|&b| b as usize >= rows)
            .unwrap(),
        AttentionPhase::Verify => 256,
        AttentionPhase::Prefill => 2048,
    }
}

fn cases() -> Vec<Case> {
    let case = |name: &str, phase, fp8, sequences: Vec<(usize, usize)>, pattern| Case {
        name: name.into(),
        phase,
        fp8,
        rounded: name.contains("rounded"),
        sequences,
        pattern,
    };
    // Decode rows see `context` tokens, the last one their own.
    let decode = |name: &str, fp8, contexts: &[usize], pattern| {
        case(
            name,
            AttentionPhase::Decode,
            fp8,
            contexts.iter().map(|&c| (c - 1, 1)).collect(),
            pattern,
        )
    };
    // Verify items end at `context` tokens with 8, 7, ... rows.
    let verify = |name: &str, fp8, context: usize, items: usize| {
        case(
            name,
            AttentionPhase::Verify,
            fp8,
            (0..items)
                .map(|i| {
                    let rows = 8 - i % 8;
                    (context - rows, rows)
                })
                .collect(),
            Pattern::Random,
        )
    };
    let prefill = |name: &str, fp8, sequences: Vec<(usize, usize)>, pattern| {
        case(name, AttentionPhase::Prefill, fp8, sequences, pattern)
    };
    use Pattern::*;
    vec![
        decode("decode_c1_b1", false, &[1], Random),
        decode("decode_c1_b8", false, &[1; 8], Random),
        decode("decode_c1_b32", false, &[1; 32], Random),
        decode("decode_c2051_b1", false, &[2051], Random),
        decode("decode_c2052_b1", false, &[2052], Random),
        decode("decode_c2052_b8", false, &[2052; 8], Random),
        decode("decode_c8192_b8", false, &[8192; 8], Random),
        decode("decode_c8192_b32", false, &[8192; 32], Random),
        decode("decode_c65536_b1", false, &[65536], Random),
        decode("decode_c65536_b32", false, &[65536; 32], Random),
        decode("decode_c262144_b1", false, &[262_144], Random),
        decode("decode_c262144_b8", false, &[262_144; 8], Random),
        decode(
            "decode_mixed_b8",
            false,
            &[1, 5, 2051, 2052, 3001, 8192, 4097, 130],
            Random,
        ),
        decode("decode_ties_c8192_b1", false, &[8192], Ties),
        decode("decode_zero_query_c8192_b1", false, &[8192], ZeroQuery),
        // 16,384 tied blocks overflow the gathered bin.
        decode("decode_zero_query_c65536_b1", false, &[65536], ZeroQuery),
        decode("decode_fp8_c8192_b8", true, &[8192; 8], Random),
        decode("decode_fp8_c262144_b1", true, &[262_144], Random),
        verify("verify_c2052_b8", false, 2052, 8),
        verify("verify_c8192_b8", false, 8192, 8),
        // The first item's rollback keeps one of eight rows with its open
        // block starting two positions before the item, ten before the last
        // rejected row: a smaller ring would lose a kept key.
        verify("verify_c8194_b8", false, 8194, 8),
        verify("verify_c65536_b32", false, 65536, 32),
        verify("verify_fp8_c8192_b8", true, 8192, 8),
        decode("decode_fp8_rounded_c8192_b8", true, &[8192; 8], Random),
        verify("verify_fp8_rounded_c8192_b8", true, 8192, 8),
        prefill("prefill_c2051", false, vec![(3, 2048)], Random),
        prefill("prefill_c8192", false, vec![(6144, 2048)], Random),
        prefill("prefill_c65536", false, vec![(63_488, 2048)], Random),
        prefill("prefill_c262144", false, vec![(260_096, 2048)], Random),
        prefill("prefill_mixed", false, vec![(3001, 999), (0, 1049)], Random),
        prefill("prefill_ties_c8192", false, vec![(6144, 2048)], Ties),
        prefill("prefill_fp8_c8192", true, vec![(6144, 2048)], Random),
        prefill(
            "prefill_fp8_rounded_c8192",
            true,
            vec![(6144, 2048)],
            Random,
        ),
    ]
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [-scale, scale).
    fn uniform(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
    fn bf16(&mut self, scale: f32) -> u16 {
        f32_to_bf16(self.uniform(scale))
    }
}

fn entry(name: &str) -> Entry {
    catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.name == format!("qwen38_{name}"))
        .unwrap_or_else(|| panic!("no QSA entry {name}"))
}

/// The Qwen3.8 topology whose QSA attention entries the step runs
/// (`kernel_lab::qwen38_profile`).
#[derive(Clone, Copy, Debug)]
struct Topology {
    geometry: Geometry,
    indexer_profile: Profile,
    profile: Profile,
    ranks: u32,
    /// Query heads per rank over the one KV head.
    heads: usize,
    /// A lean run (`kernel_lab::qwen38_lean`): `LEAN_CASES` only.
    lean: bool,
}

fn topology() -> LabResult<Topology> {
    let profile = kernel_lab::qwen38_profile()?;
    let shape = kernel_lab::qwen38_shape()?;
    let geometry = model_qwen38_contract::sm120::attention::sparse_geometry(profile)
        .ok_or("no QSA geometry for the Qwen3.8 profile")?;
    if geometry.q_heads != shape.q_heads() {
        return Err(format!(
            "QSA geometry has {} query heads, the shape {}",
            geometry.q_heads,
            shape.q_heads()
        ));
    }
    Ok(Topology {
        geometry,
        indexer_profile: model_qwen38_contract::sm120::config::PROFILE,
        profile,
        ranks: shape.ranks,
        heads: shape.q_heads() as usize,
        lean: kernel_lab::qwen38_lean()?,
    })
}

fn lab_indexer_topology(profile: Profile, heads: u32) -> Topology {
    let geometry = kernel_lab::lab_shapes::sparse_geometry(profile, heads);
    let control = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|e| {
            e.kind == Kind::Attention
                && e.geometry.q_heads == geometry.q_heads
                && e.geometry.kv_heads == geometry.kv_heads
                && e.geometry.head_dim == geometry.head_dim
                && e.geometry.top_blocks == geometry.top_blocks
        })
        .expect("production attention control");
    assert_eq!(
        (
            control.geometry.block,
            control.geometry.top_blocks,
            control.geometry.page_size,
            control.geometry.softmax_scale.to_bits(),
        ),
        (
            geometry.block,
            geometry.top_blocks,
            geometry.page_size,
            geometry.softmax_scale.to_bits(),
        ),
        "borrowed attention geometry",
    );
    Topology {
        geometry,
        indexer_profile: profile,
        profile: control.profile,
        ranks: control.profile.ranks().unwrap_or(4),
        heads: geometry.q_heads as usize,
        lean: false,
    }
}

fn lab_attention_topology(profile: Profile) -> Topology {
    if kernel_lab::lab_shapes::NEW_SPARSE_SHAPES
        .iter()
        .any(|s| s.profile == profile)
    {
        let geometry = kernel_lab::lab_shapes::sparse_geometry(profile, 4);
        let indexer_profile = catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|e| {
                let g = e.geometry;
                e.kind == Kind::Query
                    && g.index_heads == geometry.index_heads
                    && g.rotary_dim == geometry.rotary_dim
                    && g.mrope == geometry.mrope
                    && g.top_blocks == geometry.top_blocks
                    && g.block == geometry.block
                    && g.ring == geometry.ring
                    && g.page_size == geometry.page_size
                    && g.theta == geometry.theta
                    && g.eps == geometry.eps
            })
            .expect("matching lab indexer")
            .profile;
        return Topology {
            geometry,
            indexer_profile,
            profile,
            ranks: if geometry.q_heads > 8 { 2 } else { 4 },
            heads: geometry.q_heads as usize,
            lean: false,
        };
    }
    if let Some(index) = kernel_lab::lab_shapes::WIDE_SPARSE_PROFILES
        .iter()
        .position(|&p| p == profile)
    {
        let geometry = kernel_lab::lab_shapes::sparse_geometry(profile, 8);
        return Topology {
            geometry,
            indexer_profile: kernel_lab::lab_shapes::WIDE_SPARSE_PROFILES[0],
            profile,
            ranks: if index == 0 { 4 } else { 2 },
            heads: geometry.q_heads as usize,
            lean: false,
        };
    }
    let geometry = kernel_lab::lab_shapes::sparse_geometry(profile, 4);
    let indexer_profile = topology()
        .expect("production indexer control")
        .indexer_profile;
    let control = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|e| e.profile == indexer_profile && e.kind == Kind::Query)
        .expect("production indexer entry")
        .geometry;
    assert_eq!(geometry.index_heads, control.index_heads);
    assert_eq!(geometry.index_dim, control.index_dim);
    assert_eq!(geometry.rotary_dim, control.rotary_dim);
    assert_eq!(geometry.theta.to_bits(), control.theta.to_bits());
    assert_eq!(geometry.mrope, control.mrope);
    assert_eq!(geometry.eps.to_bits(), control.eps.to_bits());
    assert_eq!(geometry.block, control.block);
    assert_eq!(geometry.ring, control.ring);
    assert_eq!(geometry.top_blocks, control.top_blocks);
    assert_eq!(geometry.max_context, control.max_context);
    assert_eq!(geometry.page_size, control.page_size);
    Topology {
        geometry,
        indexer_profile,
        profile,
        ranks: indexer_profile
            .ranks()
            .expect("production indexer rank count"),
        heads: geometry.q_heads as usize,
        lean: false,
    }
}

fn lab_indexer_topologies() -> Vec<Topology> {
    let filter = std::env::var("CHOPINFER_QSA_SHAPE").ok();
    let shapes: Vec<_> = kernel_lab::lab_shapes::indexer_shapes()
        .filter(|(profile, _, _)| filter.as_ref().is_none_or(|f| f == profile.tag()))
        .map(|(profile, heads, _)| lab_indexer_topology(profile, heads))
        .collect();
    assert!(
        !shapes.is_empty(),
        "CHOPINFER_QSA_SHAPE selects no lab indexer"
    );
    shapes
}

impl Topology {
    fn indexer(&self, name: &str) -> Entry {
        catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|entry| {
                entry.profile == self.indexer_profile
                    && entry.name == format!("{}_{name}", self.indexer_profile.tag())
            })
            .unwrap_or_else(|| panic!("missing indexer entry {name}"))
    }

    /// The topology's attention entry `attn_{dtype}_{phase}`.
    fn attention(&self, dtype: &str, phase: &str) -> LabResult<Entry> {
        let name = format!("{}_attn_{dtype}_{phase}", self.profile.tag());
        let entry = catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|entry| entry.name == name && entry.profile == self.profile)
            .ok_or_else(|| format!("no QSA entry {name}"))?;
        if entry.geometry.q_heads as usize != self.heads {
            return Err(format!(
                "{name} has {} query heads, not {}",
                entry.geometry.q_heads, self.heads
            ));
        }
        Ok(entry)
    }

    /// Marks receipts of other topologies than TP4, whose receipts stay as
    /// they were, and of lean runs.
    fn mark(&self, receipt: &mut BTreeMap<String, Value>) {
        if kernel_lab::lab_shapes::PROFILES.contains(&self.indexer_profile)
            || kernel_lab::lab_shapes::PROFILES.contains(&self.profile)
        {
            receipt.insert("indexer_profile".into(), json!(self.indexer_profile.tag()));
            receipt.insert("index_heads".into(), json!(self.geometry.index_heads));
            receipt.insert("attention_profile".into(), json!(self.profile.tag()));
            receipt.insert("top_blocks".into(), json!(self.geometry.top_blocks));
            receipt.insert("rotary_dim".into(), json!(self.geometry.rotary_dim));
        }
        if self.ranks != 4 {
            receipt.insert("qwen38_ranks".into(), json!(self.ranks));
        }
        if self.lean {
            receipt.insert("lean".into(), json!(true));
        }
    }
}

/// TP4's QSA cubin, which holds the indexer and TP4's attention, and TP2's
/// cubin holds its attention entries.
struct Modules {
    topology: Topology,
    indexer: Module,
    attention: Option<Module>,
}

impl Modules {
    fn load(session: &Session) -> LabResult<Self> {
        Self::for_topology(session, topology()?)
    }

    fn for_topology(session: &Session, topology: Topology) -> LabResult<Self> {
        let attention = if topology.profile == topology.indexer_profile {
            None
        } else {
            Some(session.module(&format!(
                "sparse_attention_{}_sm120",
                topology.profile.tag()
            ))?)
        };
        Ok(Self {
            topology,
            indexer: session.module(&format!(
                "sparse_attention_{}_sm120",
                topology.indexer_profile.tag()
            ))?,
            attention,
        })
    }

    fn attention(&self) -> &Module {
        self.attention.as_ref().unwrap_or(&self.indexer)
    }
}

fn module(session: &Session) -> LabResult<Module> {
    let bytes: &[u8] = include_bytes!(concat!(
        env!("OUT_DIR"),
        "/sparse_attention_qwen38_sm120.cubin"
    ));
    session
        .context
        .load_module(bytes)
        .map_err(|e| e.to_string())
}

fn function(module: &Module, symbol: &str, dynamic_shared_bytes: u32) -> LabResult<Function> {
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    if dynamic_shared_bytes > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    Ok(function)
}

/// The launch config of a plan; `CHOPINFER_QSA_NO_PDL=1` drops PDL so a
/// profiler times each kernel without its overlap with the previous one.
fn config(mut plan: kernel_core::contract::LaunchPlan) -> LabResult<runtime_cuda::LaunchConfig> {
    if !pdl_enabled() {
        plan.pdl = false;
    }
    launch::config(plan)
}

/// Whether the step's launches keep their PDL attribute.
fn pdl_enabled() -> bool {
    std::env::var("CHOPINFER_QSA_NO_PDL").as_deref() != Ok("1")
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

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<Vec<u8>> {
    let mut out = vec![0; bytes];
    buffer
        .copy_to_bytes_at_async(0, &mut out, &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(out)
}

/// A cache plane `offset` bytes into `buffer`.
fn plane(buffer: &DeviceBuffer, offset: u64) -> KernelArgument<'_> {
    if offset == 0 {
        KernelArgument::Buffer(buffer)
    } else {
        KernelArgument::Range {
            buffer,
            offset: offset as usize,
            byte_len: buffer.byte_len() - offset as usize,
        }
    }
}

fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}
fn as_i32(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn decode_bf16(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(bf16_to_f32(v))).collect()
}

/// One sequence's host data: raw indexer keys, RoPE positions and KV codes
/// for every token, and its physical pages.
struct Sequence {
    head_dim: usize,
    cached: usize,
    rows: usize,
    keys: Vec<u16>,
    rope: Vec<[i32; 3]>,
    k: Vec<u8>,
    v: Vec<u8>,
    /// The BF16 values the rounded FP8 codes came from; empty otherwise.
    k_bf16: Vec<u16>,
    v_bf16: Vec<u16>,
    pages: Vec<i32>,
    slot: i32,
}

impl Sequence {
    fn total(&self) -> usize {
        self.cached + self.rows
    }
    /// Element `dim` of `token` in the K (`plane` 0) or V (1) codes.
    fn kv_value(&self, plane: usize, token: usize, dim: usize, fp8: bool) -> f64 {
        let codes = if plane == 0 { &self.k } else { &self.v };
        if fp8 {
            SmallFloat::E4M3.decode_f64(codes[token * self.head_dim + dim])
                * f64::from(FP8_SCALES[plane])
        } else {
            let at = 2 * (token * self.head_dim + dim);
            f64::from(bf16_to_f32(u16::from_ne_bytes([codes[at], codes[at + 1]])))
        }
    }
    /// Replaces the K (`plane` 0) or V (1) codes from `token` on with one
    /// token's `fresh` codes. Rounded FP8 sources take the BF16 values the
    /// fresh codes store exactly, so the unrounded reference reads the
    /// token the cache holds.
    fn rewrite_kv(&mut self, plane: usize, token: usize, fresh: &[u8]) {
        let (codes, source) = if plane == 0 {
            (&mut self.k, &mut self.k_bf16)
        } else {
            (&mut self.v, &mut self.v_bf16)
        };
        codes.truncate(token * fresh.len());
        codes.extend_from_slice(fresh);
        if !source.is_empty() {
            source.truncate(token * self.head_dim);
            source.extend(
                fresh
                    .iter()
                    .map(|&code| f32_to_bf16(kv_fp8::decode(code, FP8_SCALES[plane]) as f32)),
            );
        }
    }
}

/// The step's packed rows.
#[derive(Default)]
struct Step {
    rows: usize,
    row_batch: Vec<i32>,
    positions: Vec<i32>,
    rope: Vec<[i32; 3]>,
    projection: Vec<u16>,
    q: Vec<u16>,
}

struct Fixture {
    case: Case,
    topology: Topology,
    sequences: Vec<Sequence>,
    step: Step,
    /// Trailing positions whose ring rows the checks compare: all 16, or
    /// the open block after a verify rollback, whose rejected rows may still
    /// hold older ring slots.
    ring_checked: usize,
    max_pages: u32,
    total_pages: usize,
    /// The largest context every launch plan covers.
    context: u32,
    prepared: u32,
    /// Rows the query, score, selection and output buffers hold: a decode
    /// step's prepared bucket, whose rows past the live count must keep
    /// their poison, else the live rows.
    padded_rows: usize,
    score_stride: u32,
    cos_sin: Vec<f32>,
    q_norm: Vec<u16>,
    k_norm: Vec<u16>,
    // Device state.
    d_cos_sin: DeviceBuffer,
    d_q_norm: DeviceBuffer,
    d_k_norm: DeviceBuffer,
    d_tables: DeviceBuffer,
    d_slots: DeviceBuffer,
    d_ring: DeviceBuffer,
    d_ring_positions: DeviceBuffer,
    /// -1 for every sequence: no key write keeps an anchor ring, so its
    /// anchor buffer is never written.
    d_no_anchor: DeviceBuffer,
    d_anchor_ring: DeviceBuffer,
    d_block_keys: DeviceBuffer,
    d_k: DeviceBuffer,
    d_v: DeviceBuffer,
    /// Byte offsets of the K, V and block-key planes in `d_k`, `d_v` and
    /// `d_block_keys`, and their page strides: separate buffers by default,
    /// one shared pool in the high-id case.
    plane_offsets: [u64; 3],
    k_stride: u64,
    v_stride: u64,
    block_stride: u64,
    d_projection: DeviceBuffer,
    d_rope: DeviceBuffer,
    d_row_batch: DeviceBuffer,
    d_positions: DeviceBuffer,
    d_live: DeviceBuffer,
    d_q_index: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_selected: DeviceBuffer,
    d_q: DeviceBuffer,
    d_out: DeviceBuffer,
    d_workspace: DeviceBuffer,
}

struct Kernels {
    query: (Entry, Function),
    key_write: (Entry, Function),
    scores: (Entry, Function),
    topk: (Entry, Function),
    attention: (Entry, Function),
    merge: Function,
}

impl Kernels {
    fn load(modules: &Modules, case: &Case) -> LabResult<Self> {
        let module = &modules.indexer;
        let load = |name: &str| -> LabResult<(Entry, Function)> {
            let entry = modules.topology.indexer(name);
            let function = function(module, &entry.symbol(), entry.dynamic_shared_bytes())?;
            Ok((entry, function))
        };
        let phase = match case.phase {
            AttentionPhase::Decode => "decode",
            AttentionPhase::Verify => "verify",
            AttentionPhase::Prefill => "prefill",
        };
        let entry = modules
            .topology
            .attention(if case.fp8 { "fp8" } else { "bf16" }, phase)?;
        let attention_module = modules.attention();
        let attention = (
            entry.clone(),
            function(
                attention_module,
                &entry.symbol(),
                entry.dynamic_shared_bytes(),
            )?,
        );
        let merge = function(attention_module, &entry.merge_symbol().unwrap(), 0)?;
        Ok(Self {
            query: load("index_query")?,
            key_write: load("index_key_write")?,
            scores: load(&format!("block_scores_{phase}"))?,
            topk: load("block_topk")?,
            attention,
            merge,
        })
    }
}

fn cos_sin_table(positions: usize, geometry: &Geometry) -> Vec<f32> {
    (0..positions)
        .flat_map(|position| {
            kernel_core::gated_attention_oracle::mrope_cos_sin(
                [position; 3],
                geometry.rotary_dim as usize,
                f64::from(geometry.theta),
                geometry.mrope,
            )
            .into_iter()
            .flat_map(|(cos, sin)| [cos as f32, sin as f32])
        })
        .collect()
}

/// The sizes a fixture's buffers hold: step rows (a decode step's whole
/// bucket), sequences, physical pages and the largest context, which fixes
/// the table and score strides, the cos/sin table and every launch plan.
#[derive(Clone, Copy, Debug)]
struct Capacity {
    rows: usize,
    sequences: usize,
    pages: usize,
    context: usize,
}

impl Capacity {
    /// Room for exactly `case`'s step, with three decoy pages.
    fn of(case: &Case, geometry: Geometry) -> Self {
        let page_size = geometry.page_size as usize;
        let rows = case.sequences.iter().map(|&(_, rows)| rows).sum::<usize>();
        Self {
            rows: if case.phase == AttentionPhase::Decode {
                rows.max(prepared_rows(case.phase, rows) as usize)
            } else {
                rows
            },
            sequences: case.sequences.len(),
            pages: case
                .sequences
                .iter()
                .map(|&(cached, rows)| (cached + rows).div_ceil(page_size))
                .sum::<usize>()
                + 3,
            context: case
                .sequences
                .iter()
                .map(|&(cached, rows)| cached + rows)
                .max()
                .unwrap_or(1),
        }
    }
}

fn prepare(session: &Session, case: &Case, topology: Topology) -> LabResult<Fixture> {
    let started = std::time::Instant::now();
    let mut fixture = Fixture::allocate(
        session,
        case,
        topology,
        Capacity::of(case, topology.geometry),
    )?;
    fixture.load(session, case)?;
    eprintln!(
        "{}: prepared {} rows in {:.1}s",
        case.name,
        fixture.step.rows,
        started.elapsed().as_secs_f64()
    );
    Ok(fixture)
}

impl Fixture {
    /// Allocates the buffers of `capacity` for `case`'s phase and KV dtype.
    /// The caches and outputs start poisoned; `load` uploads each step.
    fn allocate(
        session: &Session,
        case: &Case,
        topology: Topology,
        capacity: Capacity,
    ) -> LabResult<Self> {
        let page_size = topology.geometry.page_size as usize;
        let top_blocks = topology.geometry.top_blocks as usize;
        let ring_size = topology.geometry.ring as usize;
        let geometry = topology.geometry;
        assert_eq!(
            geometry.kv_heads, 1,
            "QSA fixture requires one KV head per rank"
        );
        let projection_width = (geometry.index_heads + 1) as usize * geometry.index_dim as usize;
        let head_dim = geometry.head_dim as usize;
        let index_heads = geometry.index_heads as usize;
        let index_dim = geometry.index_dim as usize;
        let heads = topology.heads;
        let Capacity {
            rows,
            sequences,
            pages,
            context,
        } = capacity;
        let prepared = prepared_rows(case.phase, rows);
        let max_pages = context.div_ceil(page_size);
        let score_stride = geometry.score_stride(context as u32);
        let cos_sin = cos_sin_table(context + 4, &geometry);
        // Page strides exceed one page block, as in a pool shared by layers,
        // and the gaps hold decoys; K and V strides differ.
        let page_bytes = page_size * head_dim * if case.fp8 { 1 } else { 2 };
        let (k_stride, v_stride) = (page_bytes * 3 / 2, page_bytes * 2);
        let block_stride = BLOCK_PAGE_STRIDE as usize;
        let rows_capacity = HISTORY_ROWS.max(rows);
        let buffer = |bytes: usize| session.allocate(bytes.max(16));
        Ok(Self {
            d_cos_sin: upload(session, &bytes_f32(&cos_sin))?,
            d_q_norm: buffer(index_dim * 2)?,
            d_k_norm: buffer(index_dim * 2)?,
            d_tables: buffer(sequences * max_pages * 4)?,
            d_slots: buffer(sequences * 4)?,
            // Poisoned: NaN keys and -1 positions show unwritten rows.
            d_ring: upload(session, &vec![0xff; sequences * ring_size * index_dim * 2])?,
            d_ring_positions: upload(session, &vec![0xff; sequences * ring_size * 3 * 4])?,
            d_no_anchor: upload(session, &bytes_i32(&vec![-1; sequences]))?,
            d_anchor_ring: buffer(16)?,
            d_block_keys: upload(session, &vec![0xff; pages * block_stride])?,
            d_k: buffer(pages * k_stride)?,
            d_v: buffer(pages * v_stride)?,
            plane_offsets: [0; 3],
            k_stride: k_stride as u64,
            v_stride: v_stride as u64,
            block_stride: BLOCK_PAGE_STRIDE,
            d_projection: session.allocate(rows_capacity * projection_width * 2)?,
            d_rope: session.allocate(rows_capacity * 12)?,
            d_row_batch: session.allocate(rows_capacity * 4)?,
            d_positions: session.allocate(rows_capacity * 4)?,
            d_live: session.allocate(4)?,
            d_q_index: upload(session, &vec![0xff; rows * index_heads * index_dim * 2])?,
            d_scores: upload(session, &vec![0xff; rows * score_stride as usize * 4])?,
            d_selected: upload(session, &vec![0xff; rows * top_blocks * 4])?,
            d_q: buffer(rows * heads * head_dim * 2)?,
            d_out: upload(session, &vec![0xff; rows * heads * head_dim * 2])?,
            // Room for 64 splits of every decode or verify row.
            d_workspace: session.allocate(if case.phase == AttentionPhase::Prefill {
                16
            } else {
                prepared as usize * heads * 64 * (head_dim + 2) * 4
            })?,
            case: case.clone(),
            topology,
            sequences: Vec::new(),
            step: Step::default(),
            ring_checked: ring_size,
            max_pages: max_pages as u32,
            total_pages: pages,
            context: context as u32,
            prepared,
            padded_rows: rows,
            score_stride,
            cos_sin,
            q_norm: Vec::new(),
            k_norm: Vec::new(),
        })
    }

    /// Uploads one step of `case` into the buffers, from a generator seeded
    /// by the case name: each sequence's slot, its table row over shuffled
    /// physical pages and its KV codes (the other pages and every page's gap
    /// hold decoys), the step rows' attention queries and the norm weights.
    /// The packed rows go up after any history (`upload_step`).
    fn load(&mut self, session: &Session, case: &Case) -> LabResult<()> {
        let page_size = self.topology.geometry.page_size as usize;
        let block_size = self.topology.geometry.block as usize;
        let ring_size = self.topology.geometry.ring as usize;
        let head_dim = self.topology.geometry.head_dim as usize;
        let index_heads = self.topology.geometry.index_heads as usize;
        let index_dim = self.topology.geometry.index_dim as usize;
        let heads = self.topology.heads;
        let batch = case.sequences.len();
        let needed = Capacity::of(case, self.topology.geometry);
        if (case.phase, case.fp8) != (self.case.phase, self.case.fp8)
            || needed.rows > self.padded_rows
            || needed.sequences * ring_size * index_dim * 2 > self.d_ring.byte_len()
            || needed.pages > self.total_pages
            || needed.context > self.context as usize
        {
            return Err(format!(
                "{}: the step does not fit the fixture's buffers",
                case.name
            ));
        }
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (case.name.len() as u64) << 32);
        for byte in case.name.bytes() {
            rng.0 = rng.0.rotate_left(5) ^ u64::from(byte);
            rng.next();
        }
        let shared_key: Vec<u16> = (0..index_dim).map(|_| rng.bf16(2.0)).collect();
        // Per plane, the FP8 codes whose scaled values lie in [-2, 2].
        let fp8_codes = FP8_SCALES.map(|scale| {
            (0..=255_u8)
                .filter(|&code| {
                    let value = kv_fp8::decode(code, scale);
                    value.is_finite() && value.abs() <= 2.0
                })
                .collect::<Vec<u8>>()
        });
        let pages_per: Vec<usize> = case
            .sequences
            .iter()
            .map(|&(cached, rows)| (cached + rows).div_ceil(page_size))
            .collect();
        // Physical pages are a shuffled permutation, with decoys never mapped.
        let mut physical: Vec<i32> = (0..self.total_pages as i32).collect();
        for i in (1..physical.len()).rev() {
            let j = (rng.next() % (i as u64 + 1)) as usize;
            physical.swap(i, j);
        }
        let mut next_page = 0;
        let mut sequences = Vec::with_capacity(batch);
        for (b, &(cached, rows)) in case.sequences.iter().enumerate() {
            let total = cached + rows;
            let mut keys = Vec::with_capacity(total * index_dim);
            for t in 0..total {
                if case.pattern == Pattern::Ties && (t / block_size) % 97 != 5 {
                    keys.extend_from_slice(&shared_key);
                } else {
                    keys.extend((0..index_dim).map(|_| rng.bf16(2.0)));
                }
            }
            // Text rows share one position on every axis; alternate sequences
            // offset the H and W axes to exercise the M-RoPE sections.
            let (dh, dw) = ((b % 3) as i32, (2 * (b % 2)) as i32);
            let rope = (0..total as i32).map(|p| [p, p + dh, p + dw]).collect();
            let kv = |rng: &mut Rng, plane: usize| -> (Vec<u8>, Vec<u16>) {
                if case.rounded {
                    let values: Vec<u16> = (0..total * head_dim).map(|_| rng.bf16(2.0)).collect();
                    let codes = values
                        .iter()
                        .map(|&v| kv_fp8::encode(bf16_to_f32(v), FP8_SCALES[plane]))
                        .collect();
                    (codes, values)
                } else if case.fp8 {
                    let codes = &fp8_codes[plane];
                    let codes = (0..total * head_dim)
                        .map(|_| codes[(rng.next() % codes.len() as u64) as usize])
                        .collect();
                    (codes, Vec::new())
                } else {
                    let codes = (0..total * head_dim)
                        .flat_map(|_| rng.bf16(2.0).to_ne_bytes())
                        .collect();
                    (codes, Vec::new())
                }
            };
            let (k, k_bf16) = kv(&mut rng, 0);
            let (v, v_bf16) = kv(&mut rng, 1);
            let pages = physical[next_page..next_page + pages_per[b]].to_vec();
            next_page += pages_per[b];
            sequences.push(Sequence {
                head_dim: self.topology.geometry.head_dim as usize,
                cached,
                rows,
                keys,
                rope,
                k,
                v,
                k_bf16,
                v_bf16,
                pages,
                slot: (batch - 1 - b) as i32,
            });
        }
        let max_pages = self.max_pages as usize;
        let mut tables = vec![0_i32; batch * max_pages];
        for (b, sequence) in sequences.iter().enumerate() {
            tables[b * max_pages..][..sequence.pages.len()].copy_from_slice(&sequence.pages);
        }

        // The step rows, sequence by sequence.
        let mut step = Step::default();
        for (b, sequence) in sequences.iter().enumerate() {
            for t in sequence.cached..sequence.total() {
                step.row_batch.push(b as i32);
                step.positions.push(t as i32);
                step.rope.push(sequence.rope[t]);
                for _ in 0..index_heads {
                    for d in 0..index_dim {
                        let value = match case.pattern {
                            Pattern::ZeroQuery => 0,
                            Pattern::Ties if d < self.topology.geometry.rotary_dim as usize => 0,
                            _ => rng.bf16(2.0),
                        };
                        step.projection.push(value);
                    }
                }
                step.projection
                    .extend_from_slice(&sequence.keys[t * index_dim..(t + 1) * index_dim]);
                step.q.extend((0..heads * head_dim).map(|_| rng.bf16(2.0)));
            }
        }
        step.rows = step.row_batch.len();
        let q_norm: Vec<u16> = (0..index_dim).map(|_| rng.bf16(0.5)).collect();
        let k_norm: Vec<u16> = (0..index_dim).map(|_| rng.bf16(0.5)).collect();

        // Paged K and V at the fixture's page strides.
        let element = if case.fp8 { 1 } else { 2 };
        let (k_stride, v_stride) = (self.k_stride as usize, self.v_stride as usize);
        let mut k_pages: Vec<u8> = (0..self.total_pages * k_stride)
            .map(|_| rng.next() as u8)
            .collect();
        let mut v_pages: Vec<u8> = (0..self.total_pages * v_stride)
            .map(|_| rng.next() as u8)
            .collect();
        for sequence in &sequences {
            for t in 0..sequence.total() {
                let page = sequence.pages[t / page_size] as usize;
                let within = (t % page_size) * head_dim * element;
                let from = t * head_dim * element;
                k_pages[page * k_stride + within..][..head_dim * element]
                    .copy_from_slice(&sequence.k[from..from + head_dim * element]);
                v_pages[page * v_stride + within..][..head_dim * element]
                    .copy_from_slice(&sequence.v[from..from + head_dim * element]);
            }
        }
        let slots: Vec<i32> = sequences.iter().map(|s| s.slot).collect();
        for (buffer, bytes) in [
            (&self.d_q_norm, bytes_u16(&q_norm)),
            (&self.d_k_norm, bytes_u16(&k_norm)),
            (&self.d_tables, bytes_i32(&tables)),
            (&self.d_slots, bytes_i32(&slots)),
            (&self.d_k, k_pages),
            (&self.d_v, v_pages),
            (&self.d_q, bytes_u16(&step.q)),
        ] {
            poison::write(session, buffer, 0, &bytes)?;
        }
        self.case = case.clone();
        self.sequences = sequences;
        self.step = step;
        self.q_norm = q_norm;
        self.k_norm = k_norm;
        Ok(())
    }

    fn element(&self) -> usize {
        if self.case.fp8 { 1 } else { 2 }
    }

    /// Uploads packed rows: projection rows hold `q` then the raw key.
    fn upload_rows(
        &self,
        session: &Session,
        row_batch: &[i32],
        positions: &[i32],
        rope: &[[i32; 3]],
        projection: &[u16],
    ) -> LabResult<()> {
        let stream = &session.stream;
        let staged = [
            bytes_i32(row_batch),
            bytes_i32(positions),
            bytes_i32(&rope.iter().flatten().copied().collect::<Vec<_>>()),
            bytes_u16(projection),
            (row_batch.len() as u32).to_ne_bytes().to_vec(),
        ];
        let targets = [
            &self.d_row_batch,
            &self.d_positions,
            &self.d_rope,
            &self.d_projection,
            &self.d_live,
        ];
        for (buffer, bytes) in targets.into_iter().zip(&staged) {
            buffer
                .copy_from_bytes_at(0, bytes, stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Writes every sequence's history into the ring and block-key caches,
    /// packing each launch with up to `HISTORY_ROWS / batch` rows of every
    /// sequence that still has history.
    fn write_history(&self, session: &Session, kernels: &Kernels) -> LabResult<()> {
        let index_heads = self.topology.geometry.index_heads as usize;
        let index_dim = self.topology.geometry.index_dim as usize;
        let batch = self.sequences.len();
        let share = (HISTORY_ROWS / batch).max(1);
        let mut done = vec![0; batch];
        loop {
            let (mut row_batch, mut positions, mut rope, mut projection) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (b, sequence) in self.sequences.iter().enumerate() {
                let end = (done[b] + share).min(sequence.cached);
                for t in done[b]..end {
                    row_batch.push(b as i32);
                    positions.push(t as i32);
                    rope.push(sequence.rope[t]);
                    projection.extend(std::iter::repeat_n(0, index_heads * index_dim));
                    projection
                        .extend_from_slice(&sequence.keys[t * index_dim..(t + 1) * index_dim]);
                }
                done[b] = end;
            }
            if row_batch.is_empty() {
                return Ok(());
            }
            self.upload_rows(session, &row_batch, &positions, &rope, &projection)?;
            self.launch_key_write(&session.stream, kernels, 8192)?;
        }
    }

    fn launch_key_write(&self, stream: &Stream, kernels: &Kernels, prepared: u32) -> LabResult<()> {
        let (entry, function) = &kernels.key_write;
        let plan = entry.plan(prepared, 1).map_err(|e| e.0.to_string())?;
        function
            .launch(
                stream,
                config(plan)?,
                &[
                    KernelArgument::Buffer(&self.d_projection),
                    KernelArgument::Buffer(&self.d_k_norm),
                    KernelArgument::Buffer(&self.d_rope),
                    KernelArgument::Buffer(&self.d_cos_sin),
                    KernelArgument::Buffer(&self.d_row_batch),
                    KernelArgument::Buffer(&self.d_positions),
                    KernelArgument::Buffer(&self.d_live),
                    KernelArgument::Buffer(&self.d_slots),
                    KernelArgument::Buffer(&self.d_ring),
                    KernelArgument::Buffer(&self.d_ring_positions),
                    KernelArgument::Buffer(&self.d_no_anchor),
                    KernelArgument::Buffer(&self.d_no_anchor),
                    KernelArgument::Buffer(&self.d_anchor_ring),
                    KernelArgument::Buffer(&self.d_anchor_ring),
                    KernelArgument::Buffer(&self.d_tables),
                    plane(&self.d_block_keys, self.plane_offsets[2]),
                    KernelArgument::U32(self.max_pages),
                    KernelArgument::U64(self.block_stride),
                    KernelArgument::U32(self.topology.geometry.eps.to_bits()),
                ],
            )
            .map_err(|e| e.to_string())?;
        kernel_lab::launched(&entry.symbol());
        Ok(())
    }

    fn max_context(&self) -> u32 {
        self.context
    }

    fn launch_query(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        let (entry, function) = &kernels.query;
        let plan = entry
            .plan(self.prepared, self.max_context())
            .map_err(|e| e.0.to_string())?;
        function
            .launch(
                stream,
                config(plan)?,
                &[
                    KernelArgument::Buffer(&self.d_projection),
                    KernelArgument::Buffer(&self.d_q_norm),
                    KernelArgument::Buffer(&self.d_rope),
                    KernelArgument::Buffer(&self.d_cos_sin),
                    KernelArgument::Buffer(&self.d_live),
                    KernelArgument::Buffer(&self.d_q_index),
                    KernelArgument::U32(self.topology.geometry.eps.to_bits()),
                ],
            )
            .map_err(|e| e.to_string())?;
        kernel_lab::launched(&entry.symbol());
        Ok(())
    }

    fn launch_scores(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        let (entry, function) = &kernels.scores;
        let plan = entry
            .plan(self.prepared, self.max_context())
            .map_err(|e| e.0.to_string())?;
        function
            .launch(
                stream,
                config(plan)?,
                &[
                    KernelArgument::Buffer(&self.d_q_index),
                    plane(&self.d_block_keys, self.plane_offsets[2]),
                    KernelArgument::Buffer(&self.d_tables),
                    KernelArgument::Buffer(&self.d_row_batch),
                    KernelArgument::Buffer(&self.d_positions),
                    KernelArgument::Buffer(&self.d_live),
                    KernelArgument::Buffer(&self.d_scores),
                    KernelArgument::U32(self.max_pages),
                    KernelArgument::U64(self.block_stride),
                    KernelArgument::U32(self.score_stride),
                ],
            )
            .map_err(|e| e.to_string())?;
        kernel_lab::launched(&entry.symbol());
        Ok(())
    }

    fn launch_topk(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        let (entry, function) = &kernels.topk;
        let plan = entry
            .plan(self.prepared, self.max_context())
            .map_err(|e| e.0.to_string())?;
        function
            .launch(
                stream,
                config(plan)?,
                &[
                    KernelArgument::Buffer(&self.d_scores),
                    KernelArgument::Buffer(&self.d_positions),
                    KernelArgument::Buffer(&self.d_live),
                    KernelArgument::Buffer(&self.d_selected),
                    KernelArgument::U32(self.score_stride),
                ],
            )
            .map_err(|e| e.to_string())?;
        kernel_lab::launched(&entry.symbol());
        Ok(())
    }

    fn launch_attention(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        self.launch_attention_partial(stream, kernels)?;
        self.launch_merge(stream, kernels)
    }

    /// The attention kernel: normalized output with one split, else the
    /// split partials `launch_merge` folds.
    fn launch_attention_partial(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        let (entry, function) = &kernels.attention;
        let geometry = self.topology.geometry;
        let mut plan = entry
            .plan(self.prepared, self.max_context())
            .map_err(|e| e.0.to_string())?;
        let splits = self.splits(entry);
        plan.grid[2] = splits;
        function
            .launch(
                stream,
                config(plan)?,
                &[
                    KernelArgument::Buffer(&self.d_q),
                    plane(&self.d_k, self.plane_offsets[0]),
                    plane(&self.d_v, self.plane_offsets[1]),
                    KernelArgument::Buffer(&self.d_tables),
                    KernelArgument::Buffer(&self.d_row_batch),
                    KernelArgument::Buffer(&self.d_positions),
                    KernelArgument::Buffer(&self.d_live),
                    KernelArgument::Buffer(&self.d_selected),
                    KernelArgument::Buffer(&self.d_workspace),
                    KernelArgument::Buffer(&self.d_out),
                    KernelArgument::U32(self.max_pages),
                    KernelArgument::U64(self.k_stride),
                    KernelArgument::U64(self.v_stride),
                    KernelArgument::U32(splits),
                    KernelArgument::U32(geometry.softmax_scale.to_bits()),
                    KernelArgument::U32(FP8_SCALES[0].to_bits()),
                    KernelArgument::U32(FP8_SCALES[1].to_bits()),
                ],
            )
            .map_err(|e| e.to_string())?;
        kernel_lab::launched(&entry.symbol());
        Ok(())
    }

    /// The merge of the split partials, when there is more than one split.
    fn launch_merge(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        let entry = &kernels.attention.0;
        let splits = self.splits(entry);
        if splits > 1 {
            let merge = kernel_core::contract::LaunchPlan {
                grid: [self.prepared, entry.geometry.q_heads, 1],
                block: [256, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            };
            kernels
                .merge
                .launch(
                    stream,
                    config(merge)?,
                    &[
                        KernelArgument::Buffer(&self.d_workspace),
                        KernelArgument::Buffer(&self.d_live),
                        KernelArgument::Buffer(&self.d_out),
                        KernelArgument::U32(splits),
                    ],
                )
                .map_err(|e| e.to_string())?;
            kernel_lab::launched(&entry.merge_symbol().ok_or("an attention entry")?);
        }
        Ok(())
    }

    /// The planner's attention splits, or `CHOPINFER_QSA_SPLITS` (1 to 64)
    /// when tuning decode and verify.
    fn splits(&self, entry: &Entry) -> u32 {
        std::env::var("CHOPINFER_QSA_SPLITS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&splits| {
                (1..=64).contains(&splits) && self.case.phase != AttentionPhase::Prefill
            })
            .unwrap_or_else(|| entry.splits(self.prepared))
    }

    /// The step's kernels in order.
    fn launch_step(&self, stream: &Stream, kernels: &Kernels) -> LabResult<()> {
        self.launch_key_write(stream, kernels, self.prepared)?;
        self.launch_query(stream, kernels)?;
        self.launch_scores(stream, kernels)?;
        self.launch_topk(stream, kernels)?;
        self.launch_attention(stream, kernels)
    }

    fn upload_step(&self, session: &Session) -> LabResult<()> {
        self.upload_rows(
            session,
            &self.step.row_batch,
            &self.step.positions,
            &self.step.rope,
            &self.step.projection,
        )
    }

    /// The query, scores, selections, attention and ring rows the step
    /// writes, for comparing two runs of one step bit for bit.
    fn outputs(&self, session: &Session) -> LabResult<Vec<u8>> {
        let top_blocks = self.topology.geometry.top_blocks as usize;
        let ring_size = self.topology.geometry.ring as usize;
        let head_dim = self.topology.geometry.head_dim as usize;
        let index_heads = self.topology.geometry.index_heads as usize;
        let index_dim = self.topology.geometry.index_dim as usize;
        let rows = self.padded_rows;
        let mut all = download(session, &self.d_q_index, rows * index_heads * index_dim * 2)?;
        all.extend(download(
            session,
            &self.d_scores,
            rows * self.score_stride as usize * 4,
        )?);
        all.extend(download(session, &self.d_selected, rows * top_blocks * 4)?);
        all.extend(download(
            session,
            &self.d_out,
            rows * self.topology.heads * head_dim * 2,
        )?);
        all.extend(download(
            session,
            &self.d_ring,
            self.sequences.len() * ring_size * index_dim * 2,
        )?);
        Ok(all)
    }

    /// Poisoned checks of the step (SPARSE.md, sparse_indexer.cuh and
    /// sparse.cuh). Outputs: the query and attention rows of every live row;
    /// the scores of the complete blocks of rows with more than `top_blocks` of
    /// them; the first `min(complete, top_blocks)` selections of every live row;
    /// rows past the live count stay untouched. The attention workspace is
    /// scratch that only the live rows' split partials may change (sparse.cuh
    /// `partial_index`: `heads * splits * (head_dim + 2)` F32 words per row). The
    /// ring keys, their RoPE positions and the block keys are
    /// in-place state restored before every launch: block-key rows the step
    /// completes and ring rows it only writes start poisoned, while ring rows
    /// a sequence's leader reads before rewriting them are restored.
    fn replay<'a>(&'a self, session: &'a Session, kernels: &Kernels) -> LabResult<Replay<'a>> {
        let page_size = self.topology.geometry.page_size as usize;
        let block_size = self.topology.geometry.block as usize;
        let top_blocks = self.topology.geometry.top_blocks as usize;
        let ring_size = self.topology.geometry.ring as usize;
        let head_dim = self.topology.geometry.head_dim as usize;
        let index_heads = self.topology.geometry.index_heads as usize;
        let index_dim = self.topology.geometry.index_dim as usize;
        let rows = self.step.rows;
        let partials = self.workspace_partials(kernels);
        let stride = self.score_stride as usize;
        let complete = |row: usize| (self.step.positions[row] as usize + 1) / block_size;
        let q_index = WriteMap::of(DType::Bf16).prefix(rows * index_heads * index_dim)?;
        let scores = WriteMap::of(DType::F32).byte_ranges(
            (0..rows)
                .filter(|&row| complete(row) > top_blocks)
                .map(|row| row * stride * 4..(row * stride + complete(row)) * 4),
        )?;
        let selected = WriteMap::of(DType::I32).byte_ranges((0..rows).map(|row| {
            row * top_blocks * 4..(row * top_blocks + complete(row).min(top_blocks)) * 4
        }))?;
        let out = WriteMap::of(DType::Bf16).prefix(rows * self.topology.heads * head_dim)?;
        let (mut ring_written, mut ring_read, mut blocks) = (Vec::new(), Vec::new(), Vec::new());
        for sequence in self.sequences.iter().filter(|sequence| sequence.rows > 0) {
            let (first, end) = (sequence.cached, sequence.total());
            let ring_row =
                |position: usize| sequence.slot as usize * ring_size + position % ring_size;
            ring_written.extend((end.saturating_sub(ring_size).max(first)..end).map(ring_row));
            // The leader pools a block begun before the launch from the ring
            // when the launch completes it.
            let start = first - first % block_size;
            if start < first && start + block_size - 1 < end {
                ring_read.extend((start..first).map(ring_row));
            }
            for position in (first..end).filter(|position| position % block_size == block_size - 1)
            {
                let block = position / block_size;
                let page = sequence.pages[block * block_size / page_size] as usize;
                let at = page * BLOCK_PAGE_STRIDE as usize
                    + (block % (page_size / block_size)) * index_dim * 2;
                blocks.push(at..at + index_dim * 2);
            }
        }
        let rows_of = |bytes: usize, list: &mut dyn Iterator<Item = usize>| {
            list.map(|row| row * bytes..(row + 1) * bytes)
                .collect::<Vec<_>>()
        };
        let write_only = ring_written
            .iter()
            .copied()
            .filter(|row| !ring_read.contains(row))
            .collect::<Vec<_>>();
        let updated = ring_written
            .iter()
            .copied()
            .filter(|row| ring_read.contains(row))
            .collect::<Vec<_>>();
        let (key_row, rope_row) = (index_dim * 2, 12);
        Ok(Replay::new(session)
            .output("query", &self.d_q_index, q_index)
            .output("scores", &self.d_scores, scores)
            .output("selection", &self.d_selected, selected)
            .output("attention", &self.d_out, out)
            .scratch_within(
                "attention workspace",
                &self.d_workspace,
                std::iter::once(0..partials).collect(),
            )
            .state(
                "ring keys",
                &self.d_ring,
                poison::read(session, &self.d_ring)?,
                WriteMap::of(DType::Bf16)
                    .byte_ranges(rows_of(key_row, &mut write_only.iter().copied()))?,
                rows_of(key_row, &mut updated.iter().copied()),
            )
            .state(
                "ring positions",
                &self.d_ring_positions,
                poison::read(session, &self.d_ring_positions)?,
                WriteMap::of(DType::I32)
                    .byte_ranges(rows_of(rope_row, &mut write_only.iter().copied()))?,
                rows_of(rope_row, &mut updated.iter().copied()),
            )
            .state(
                "block keys",
                &self.d_block_keys,
                poison::read(session, &self.d_block_keys)?,
                WriteMap::of(DType::Bf16).byte_ranges(blocks)?,
                Vec::new(),
            ))
    }

    /// The workspace bytes of the live rows' split partials (sparse.cuh
    /// `partial_index`), packed by row; none with one split, whose attention
    /// writes the output directly.
    fn workspace_partials(&self, kernels: &Kernels) -> usize {
        let head_dim = self.topology.geometry.head_dim as usize;
        let splits = self.splits(&kernels.attention.0) as usize;
        if splits == 1 {
            return 0;
        }
        self.step.rows * self.topology.heads * splits * (head_dim + 2) * 4
    }

    /// Rows the oracle checks: every row, or an even sample plus the
    /// sequences' first and last rows for long prefill chunks.
    fn checked_rows(&self) -> Vec<usize> {
        let rows = self.step.rows;
        if rows <= 64 {
            return (0..rows).collect();
        }
        let mut picked: Vec<usize> = (0..48).map(|i| i * rows / 48).collect();
        let mut first = 0;
        for sequence in &self.sequences {
            picked.extend([first, first + 1, first + sequence.rows - 1]);
            // The rows around the dense limit.
            for position in [2050, 2051, 2052] {
                if (sequence.cached..sequence.total()).contains(&position) {
                    picked.push(first + position - sequence.cached);
                }
            }
            first += sequence.rows;
        }
        picked.retain(|&row| row < rows);
        picked.sort_unstable();
        picked.dedup();
        picked
    }
}

fn tables_for(fixture: &Fixture, rope: &[[i32; 3]]) -> Vec<Vec<(f64, f64)>> {
    let rotary_pairs = fixture.topology.geometry.rotary_dim as usize / 2;
    let mrope = fixture.topology.geometry.mrope;
    rope.iter()
        .map(|position| {
            (0..rotary_pairs)
                .map(|pair| {
                    let p = position[mrope_axis(pair, mrope)] as usize;
                    (
                        f64::from(fixture.cos_sin[(p * rotary_pairs + pair) * 2]),
                        f64::from(fixture.cos_sin[(p * rotary_pairs + pair) * 2 + 1]),
                    )
                })
                .collect()
        })
        .collect()
}

fn fail(case: &str, stage: &str, receipt: &BTreeMap<String, Value>, error: String) -> String {
    format!(
        "{case} {stage}: {error}; worst={}",
        receipt.get("worst_element").cloned().unwrap_or(Value::Null)
    )
}

/// Oracle checks of one eager step.
fn check(session: &Session, fixture: &Fixture) -> LabResult<BTreeMap<String, Value>> {
    let page_size = fixture.topology.geometry.page_size as usize;
    let block_size = fixture.topology.geometry.block as usize;
    let top_blocks = fixture.topology.geometry.top_blocks as usize;
    let ring_size = fixture.topology.geometry.ring as usize;
    let head_dim = fixture.topology.geometry.head_dim as usize;
    let index_heads = fixture.topology.geometry.index_heads as usize;
    let index_dim = fixture.topology.geometry.index_dim as usize;
    let geometry = fixture.topology.geometry;
    let name = &fixture.case.name;
    let mut summary = BTreeMap::new();
    let block_page = geometry.block_page_bytes() as usize;
    let block_stride = BLOCK_PAGE_STRIDE as usize;
    let block_bytes = download(
        session,
        &fixture.d_block_keys,
        fixture.total_pages * block_stride,
    )?;
    if let Some(page) = (0..fixture.total_pages).find(|page| {
        block_bytes[page * block_stride + block_page..(page + 1) * block_stride]
            .iter()
            .any(|&byte| byte != 0xff)
    }) {
        return Err(format!(
            "{name}: a key write touched the gap after block-key page {page}"
        ));
    }
    // Every slot: the live sequences of a live-count step need not hold the
    // first ones.
    let ring_bytes = as_u16(&poison::read(session, &fixture.d_ring)?);
    let ring_positions = as_i32(&poison::read(session, &fixture.d_ring_positions)?);

    // Key write: every block of every sequence and the final ring rows.
    let mut gpu_blocks = Vec::with_capacity(fixture.sequences.len());
    let mut worst_key = BTreeMap::new();
    for sequence in &fixture.sequences {
        let total = sequence.total();
        let blocks = total / block_size;
        let mut keys = Vec::with_capacity(blocks * index_dim);
        for block in 0..blocks {
            let page = sequence.pages[block * block_size / page_size] as usize;
            let at = page * block_stride + (block % (page_size / block_size)) * index_dim * 2;
            keys.extend(as_u16(&block_bytes[at..at + index_dim * 2]));
        }
        if blocks > 0 {
            let query = SparseAttentionQuery::IndexerKeyWrite {
                head_dim: geometry.index_dim,
                rotary_dim: geometry.rotary_dim,
                theta_bits: geometry.theta.to_bits(),
                mrope: geometry.mrope,
                eps_bits: geometry.eps.to_bits(),
                block: block_size as u32,
                ring: ring_size as u32,
                capacity: total as u32,
            };
            let mut ring = vec![0.0; ring_size * index_dim];
            let mut ring_rope = vec![[0; 3]; ring_size];
            let mut block_keys = vec![0.0; blocks * index_dim];
            let rope: Vec<[usize; 3]> = sequence
                .rope
                .iter()
                .map(|p| p.map(|x| x as usize))
                .collect();
            let writes = qsa::indexer_key_write(
                query,
                &decode_bf16(&sequence.keys),
                &(0..total).collect::<Vec<_>>(),
                &rope,
                &decode_bf16(&fixture.k_norm),
                IndexerKeyCache {
                    ring_keys: &mut ring,
                    ring_positions: &mut ring_rope,
                    block_keys: &mut block_keys,
                },
            )
            .map_err(|e| format!("{name}: key write oracle: {}", e.0))?;
            let actual: Vec<f32> = keys.iter().map(|&k| bf16_to_f32(k)).collect();
            check_output(&actual, &writes.keys, DType::Bf16, &mut worst_key)
                .map_err(|e| fail(name, "block keys", &worst_key, e))?;
        }
        // The ring holds the last raw keys and their positions.
        let slot = sequence.slot as usize;
        for t in total.saturating_sub(fixture.ring_checked)..total {
            let row = slot * ring_size + t % ring_size;
            if ring_bytes[row * index_dim..(row + 1) * index_dim]
                != sequence.keys[t * index_dim..(t + 1) * index_dim]
                || ring_positions[row * 3..row * 3 + 3] != sequence.rope[t]
            {
                return Err(format!("{name}: ring row of position {t} differs"));
            }
        }
        gpu_blocks.push(keys);
    }
    summary.insert(
        "block_keys_worst".into(),
        json!(worst_key.get("worst_element")),
    );

    // Query.
    let rows = fixture.step.rows;
    let q_index = as_u16(&download(
        session,
        &fixture.d_q_index,
        rows * index_heads * index_dim * 2,
    )?);
    let query = SparseAttentionQuery::IndexerQueryGemmaNormMrope {
        heads: geometry.index_heads,
        head_dim: geometry.index_dim,
        rotary_dim: geometry.rotary_dim,
        theta_bits: geometry.theta.to_bits(),
        mrope: geometry.mrope,
        eps_bits: geometry.eps.to_bits(),
        capacity: rows as u32,
    };
    let projected_q: Vec<f64> = fixture
        .step
        .projection
        .chunks_exact(
            (fixture.topology.geometry.index_heads + 1) as usize
                * fixture.topology.geometry.index_dim as usize,
        )
        .flat_map(|row| decode_bf16(&row[..index_heads * index_dim]))
        .collect();
    let rope: Vec<[usize; 3]> = fixture
        .step
        .rope
        .iter()
        .map(|p| p.map(|x| x as usize))
        .collect();
    let tables = tables_for(fixture, &fixture.step.rope);
    let reference = qsa::indexer_query(
        query,
        &projected_q,
        &decode_bf16(&fixture.q_norm),
        &rope,
        Some(&tables),
    )
    .map_err(|e| format!("{name}: query oracle: {}", e.0))?;
    let mut receipt = BTreeMap::new();
    let actual: Vec<f32> = q_index.iter().map(|&x| bf16_to_f32(x)).collect();
    check_output(&actual, &reference, DType::Bf16, &mut receipt)
        .map_err(|e| fail(name, "query", &receipt, e))?;
    summary.insert("query_worst".into(), json!(receipt.get("worst_element")));

    // Scores, selection and attention per checked row.
    let scores = as_f32(&download(
        session,
        &fixture.d_scores,
        rows * fixture.score_stride as usize * 4,
    )?);
    let selected = as_i32(&download(
        session,
        &fixture.d_selected,
        rows * top_blocks * 4,
    )?);
    let heads = fixture.topology.heads;
    let out = as_u16(&download(
        session,
        &fixture.d_out,
        rows * heads * head_dim * 2,
    )?);
    let score_query = SparseAttentionQuery::BlockScores {
        heads: geometry.index_heads,
        head_dim: geometry.index_dim,
        max_blocks: geometry.max_blocks(),
        capacity: 1,
    };
    let topk_query = SparseAttentionQuery::BlockTopK {
        block: block_size as u32,
        top_blocks: top_blocks as u32,
        max_blocks: geometry.max_blocks(),
        capacity: 1,
    };
    let checked = fixture.checked_rows();
    let mut sparse_rows = 0;
    let mut tie_rows = 0;
    let mut score_worst = BTreeMap::new();
    let mut attention_worst = BTreeMap::new();
    let results: Vec<LabResult<RowCheck>> = std::thread::scope(|scope| {
        let workers: Vec<_> = checked
            .chunks(checked.len().div_ceil(16).max(1))
            .map(|chunk| {
                let host = HostView {
                    geometry: fixture.topology.geometry,
                    case: &fixture.case,
                    sequences: &fixture.sequences,
                    step: &fixture.step,
                    score_stride: fixture.score_stride as usize,
                    heads,
                };
                let gpu_blocks = &gpu_blocks;
                let q_index = &q_index;
                let scores = &scores;
                let selected = &selected;
                let out = &out;
                scope.spawn(move || -> LabResult<_> {
                    let mut results = Vec::new();
                    for &row in chunk {
                        results.push(check_row(
                            &host,
                            row,
                            gpu_blocks,
                            q_index,
                            scores,
                            selected,
                            out,
                            score_query,
                            topk_query,
                        ));
                    }
                    Ok(results)
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| match worker.join().unwrap() {
                Ok(results) => results,
                Err(error) => vec![Err(error)],
            })
            .collect()
    });
    let mut fp8_sigmas = 0.0_f64;
    for result in results {
        let (sparse, tie, score_receipt, attention_receipt) = result?;
        sparse_rows += usize::from(sparse);
        tie_rows += usize::from(tie);
        if let Some(sigmas) = attention_receipt
            .get("fp8_kv_vs_bf16")
            .and_then(|check| check["max_error_sigmas"].as_f64())
        {
            fp8_sigmas = fp8_sigmas.max(sigmas);
        }
        worst(&mut score_worst, score_receipt);
        worst(&mut attention_worst, attention_receipt);
    }
    // Every other row's selection, from the kernel scores.
    let others: Vec<usize> = (0..rows).filter(|row| !checked.contains(row)).collect();
    let host = HostView {
        geometry: fixture.topology.geometry,
        case: &fixture.case,
        sequences: &fixture.sequences,
        step: &fixture.step,
        score_stride: fixture.score_stride as usize,
        heads,
    };
    let selections: Vec<LabResult<(bool, bool)>> = std::thread::scope(|scope| {
        let workers: Vec<_> = others
            .chunks(others.len().div_ceil(16).max(1))
            .map(|chunk| {
                let (host, scores, selected) = (&host, &scores, &selected);
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|&row| {
                            let visible = host.step.positions[row] as usize + 1;
                            let complete = visible / block_size;
                            let row_scores: Vec<f64> = if complete > top_blocks {
                                scores[row * host.score_stride..][..complete]
                                    .iter()
                                    .map(|&s| f64::from(s))
                                    .collect()
                            } else {
                                vec![0.0; complete]
                            };
                            check_selection(host, row, &row_scores, selected, topk_query)
                                .map(|(_, tie)| (complete > top_blocks, tie))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect()
    });
    for result in selections {
        let (sparse, tie) = result?;
        sparse_rows += usize::from(sparse);
        tie_rows += usize::from(tie);
    }
    if fixture.case.rounded {
        summary.insert("fp8_kv_vs_bf16_max_error_sigmas".into(), json!(fp8_sigmas));
    }
    // A decode bucket's rows past the live count keep their poison.
    let padded = fixture.padded_rows;
    if padded > rows {
        let score_row = fixture.score_stride as usize * 4;
        for (stage, buffer, row_bytes) in [
            ("query", &fixture.d_q_index, index_heads * index_dim * 2),
            ("scores", &fixture.d_scores, score_row),
            ("selection", &fixture.d_selected, top_blocks * 4),
            ("attention", &fixture.d_out, heads * head_dim * 2),
        ] {
            let bytes = download(session, buffer, padded * row_bytes)?;
            if let Some(at) = bytes[rows * row_bytes..].iter().position(|&b| b != 0xff) {
                return Err(format!(
                    "{name}: {stage} wrote row {} past the {rows} live rows of a {} row bucket",
                    rows + at / row_bytes,
                    fixture.prepared
                ));
            }
        }
        summary.insert("prepared_rows".into(), json!(fixture.prepared));
        summary.insert("untouched_padding_rows".into(), json!(padded - rows));
    }
    summary.insert("checked_rows".into(), json!(checked.len()));
    summary.insert("selection_rows".into(), json!(rows));
    summary.insert("sparse_rows".into(), json!(sparse_rows));
    summary.insert("threshold_tie_rows".into(), json!(tie_rows));
    summary.insert(
        "scores_worst".into(),
        json!(score_worst.get("worst_element")),
    );
    summary.insert(
        "attention_worst".into(),
        json!(attention_worst.get("worst_element")),
    );
    Ok(summary)
}

/// Keeps the receipt whose worst element has the larger error.
fn worst(into: &mut BTreeMap<String, Value>, candidate: BTreeMap<String, Value>) {
    let error = |r: &BTreeMap<String, Value>| {
        r.get("worst_element")
            .and_then(|w| w["error"].as_f64())
            .unwrap_or(-1.0)
    };
    if error(&candidate) > error(into) {
        *into = candidate;
    }
}

/// The row's key set from the oracle's top-k of `row_scores`, which must
/// equal the kernel's selection bit for bit, and whether the selection
/// threshold was tied (a dropped block scores the same as the lowest kept).
fn check_selection(
    fixture: &HostView<'_>,
    row: usize,
    row_scores: &[f64],
    selected: &[i32],
    topk_query: SparseAttentionQuery,
) -> LabResult<(Vec<usize>, bool)> {
    let block_size = fixture.geometry.block as usize;
    let top_blocks = fixture.geometry.top_blocks as usize;
    let name = &fixture.case.name;
    let visible = fixture.step.positions[row] as usize + 1;
    let complete = visible / block_size;
    let expected = qsa::block_topk(topk_query, row_scores, visible)
        .map_err(|e| format!("{name}: top-k oracle: {}", e.0))?;
    let count = complete.min(top_blocks);
    let mut tokens: Vec<usize> = selected[row * top_blocks..row * top_blocks + count]
        .iter()
        .flat_map(|&block| {
            let block = block as usize;
            block * block_size..(block + 1) * block_size
        })
        .collect();
    tokens.extend(complete * block_size..visible);
    if tokens != expected {
        let index = tokens
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(tokens.len().min(expected.len()));
        return Err(format!(
            "{name}: row {row} selection differs at key {index}: gpu {:?} oracle {:?}",
            tokens.get(index),
            expected.get(index)
        ));
    }
    let tie = complete > top_blocks && {
        let kept: std::collections::HashSet<usize> = tokens
            .iter()
            .map(|t| t / block_size)
            .filter(|&b| b < complete)
            .collect();
        let threshold = kept
            .iter()
            .map(|&b| row_scores[b])
            .fold(f64::INFINITY, f64::min);
        (0..complete).any(|b| !kept.contains(&b) && row_scores[b] == threshold)
    };
    Ok((expected, tie))
}

/// Whether a checked row is sparse and tied at its threshold, and the
/// receipts of its score and attention checks.
type RowCheck = (bool, bool, BTreeMap<String, Value>, BTreeMap<String, Value>);

/// The host data one row check reads; shared by the checking threads.
struct HostView<'a> {
    geometry: Geometry,
    case: &'a Case,
    sequences: &'a [Sequence],
    step: &'a Step,
    score_stride: usize,
    /// Query heads of the attention rows.
    heads: usize,
}

#[allow(clippy::too_many_arguments)]
fn check_row(
    fixture: &HostView<'_>,
    row: usize,
    gpu_blocks: &[Vec<u16>],
    q_index: &[u16],
    scores: &[f32],
    selected: &[i32],
    out: &[u16],
    score_query: SparseAttentionQuery,
    topk_query: SparseAttentionQuery,
) -> LabResult<RowCheck> {
    let block_size = fixture.geometry.block as usize;
    let top_blocks = fixture.geometry.top_blocks as usize;
    let head_dim = fixture.geometry.head_dim as usize;
    let index_heads = fixture.geometry.index_heads as usize;
    let index_dim = fixture.geometry.index_dim as usize;
    let name = &fixture.case.name;
    let heads = fixture.heads;
    let b = fixture.step.row_batch[row] as usize;
    let sequence = &fixture.sequences[b];
    let visible = fixture.step.positions[row] as usize + 1;
    let complete = visible / block_size;
    let sparse = complete > top_blocks;
    let mut score_receipt = BTreeMap::new();
    let row_scores: Vec<f64> = if sparse {
        let q = decode_bf16(
            &q_index[row * index_heads * index_dim..(row + 1) * index_heads * index_dim],
        );
        let keys = decode_bf16(&gpu_blocks[b][..complete * index_dim]);
        let reference = qsa::block_scores(score_query, &q, &keys)
            .map_err(|e| format!("{name}: score oracle: {}", e.0))?;
        let actual = &scores[row * fixture.score_stride..][..complete];
        check_output(actual, &reference, DType::F32, &mut score_receipt)
            .map_err(|e| fail(name, &format!("scores row {row}"), &score_receipt, e))?;
        actual.iter().map(|&s| f64::from(s)).collect()
    } else {
        vec![0.0; complete]
    };
    let (expected, tie) = check_selection(fixture, row, &row_scores, selected, topk_query)?;
    let tokens = &expected;

    // Attention over the selected key set, from a compact 4-token-page cache.
    let fp8 = fixture.case.fp8;
    let mut compact_of = vec![0_usize; visible.div_ceil(block_size)];
    let mut blocks: Vec<usize> = tokens.iter().map(|t| t / block_size).collect();
    blocks.dedup();
    let mut keys = Vec::with_capacity(blocks.len() * block_size * head_dim);
    let mut values = Vec::with_capacity(keys.capacity());
    for (index, &block) in blocks.iter().enumerate() {
        compact_of[block] = index;
        for token in block * block_size..(block + 1) * block_size {
            for d in 0..head_dim {
                let (k, v) = if token < sequence.total() {
                    (
                        sequence.kv_value(0, token, d, fp8),
                        sequence.kv_value(1, token, d, fp8),
                    )
                } else {
                    (0.0, 0.0)
                };
                keys.push(k);
                values.push(v);
            }
        }
    }
    let query = SparseAttentionQuery::SparseAttention {
        q_heads: heads as u32,
        kv_heads: fixture.geometry.kv_heads,
        head_dim: fixture.geometry.head_dim,
        block: block_size as u32,
        top_blocks: top_blocks as u32,
        softmax_scale_bits: fixture.geometry.softmax_scale.to_bits(),
        page_size: block_size as u32,
        kv_dtype: if fp8 { DType::Fp8E4M3 } else { DType::Bf16 },
        phase: fixture.case.phase,
        capacity: 1,
    };
    let q = decode_bf16(&fixture.step.q[row * heads * head_dim..(row + 1) * heads * head_dim]);
    let reference = qsa::sparse_attention(
        query,
        &q,
        std::slice::from_ref(&expected),
        SparseKv {
            keys: &keys,
            values: &values,
            block_table: &compact_of,
        },
    )
    .map_err(|e| format!("{name}: attention oracle: {}", e.0))?;
    let actual: Vec<f32> = out[row * heads * head_dim..(row + 1) * heads * head_dim]
        .iter()
        .map(|&x| bf16_to_f32(x))
        .collect();
    let mut attention_receipt = BTreeMap::new();
    check_output(&actual, &reference, DType::Bf16, &mut attention_receipt)
        .map_err(|e| fail(name, &format!("attention row {row}"), &attention_receipt, e))?;
    if !sparse && head_dim == 128 {
        let dense = kernel_core::oracle::attention(
            &kernel_core::contract::AttentionQuery {
                q_heads: heads as u32,
                kv_heads: fixture.geometry.kv_heads,
                qk_dim: head_dim as u32,
                v_dim: head_dim as u32,
                page_size: block_size as u32,
                kv_dtype: if fp8 { DType::Fp8E4M3 } else { DType::Bf16 },
                layout: kernel_core::contract::AttentionLayout::Paged,
                causal: true,
                window_left: None,
                sink: false,
                softcap: None,
                softmax_scale: fixture.geometry.softmax_scale,
                phase: fixture.case.phase,
                max_context_capacity: visible as u32,
                max_batch_capacity: 1,
            },
            &kernel_core::oracle::PagedAttention {
                q: &q,
                keys: &keys,
                values: &values,
                page_tables: std::slice::from_ref(&compact_of),
                query_lengths: &[1],
                query_starts: &[visible - 1],
                key_lengths: &[visible],
                sink_logits: None,
            },
        )
        .map_err(|e| format!("{name}: dense attention oracle: {}", e.0))?;
        if reference
            .values
            .iter()
            .zip(&dense.values)
            .any(|(a, b)| (a - b).abs() > 1e-14)
        {
            return Err(format!("{name}: sparse and dense attention oracles differ"));
        }
        check_output(&actual, &dense, DType::Bf16, &mut attention_receipt)
            .map_err(|e| fail(name, "dense attention", &attention_receipt, e))?;
        attention_receipt.insert("dense_oracle_checked".into(), json!(true));
    }
    if fixture.case.rounded {
        let original = |values: &[u16], token: usize| -> Vec<f64> {
            decode_bf16(&values[token * head_dim..(token + 1) * head_dim])
        };
        let keys: Vec<f64> = tokens
            .iter()
            .flat_map(|&t| original(&sequence.k_bf16, t))
            .collect();
        let values: Vec<f64> = tokens
            .iter()
            .flat_map(|&t| original(&sequence.v_bf16, t))
            .collect();
        let (mut exact, mut sigma) = (Vec::new(), Vec::new());
        for head in 0..heads {
            let (output, deviation) =
                kernel_core::kv_fp8::attention_sigma(kernel_core::kv_fp8::Fp8Attention {
                    query: &q[head * head_dim..(head + 1) * head_dim],
                    keys: &keys,
                    values: &values,
                    qk_dim: head_dim,
                    v_dim: head_dim,
                    softmax_scale: f64::from(fixture.geometry.softmax_scale),
                    k_scale: f64::from(FP8_SCALES[0]),
                    v_scale: f64::from(FP8_SCALES[1]),
                    sink: None,
                });
            exact.extend(output);
            sigma.extend(deviation);
        }
        kernel_lab::check_fp8_kv_output(
            &actual,
            &exact,
            &sigma,
            &reference,
            &mut attention_receipt,
        )
        .map_err(|e| fail(name, &format!("FP8 KV row {row}"), &attention_receipt, e))?;
    }
    Ok((sparse, tie, score_receipt, attention_receipt))
}

fn selected(case: &Case) -> bool {
    std::env::var("CHOPINFER_QSA_CASE")
        .ok()
        .is_none_or(|filter| filter.split(',').any(|name| name == case.name))
}

fn run_case(session: &Session, modules: &Modules, case: &Case, bench: bool) -> LabResult<()> {
    let ring_size = modules.topology.geometry.ring as usize;
    let index_dim = modules.topology.geometry.index_dim as usize;
    let fixture = prepare(session, case, modules.topology)?;
    let kernels = Kernels::load(modules, case)?;
    // The workspace holds every split the planner or `CHOPINFER_QSA_SPLITS`
    // may pick.
    let workspace = kernels.attention.0.workspace_bytes(fixture.prepared);
    if workspace > fixture.d_workspace.byte_len() as u64 {
        return Err(format!(
            "{}: {workspace} workspace bytes exceed the fixture's {}",
            case.name,
            fixture.d_workspace.byte_len()
        ));
    }
    let started = std::time::Instant::now();
    fixture.write_history(session, &kernels)?;
    fixture.upload_step(session)?;
    let ring_before = download(
        session,
        &fixture.d_ring,
        fixture.sequences.len() * ring_size * index_dim * 2,
    )?;
    let ring_positions_before = download(
        session,
        &fixture.d_ring_positions,
        fixture.sequences.len() * ring_size * 12,
    )?;
    let labelled = |error: String| format!("{}: {error}", case.name);
    let mut checks = fixture.replay(session, &kernels)?;
    checks
        .eager(|| fixture.launch_step(&session.stream, &kernels))
        .map_err(labelled)?;
    eprintln!(
        "{}: history and step in {:.1}s",
        case.name,
        started.elapsed().as_secs_f64()
    );
    let checked = std::time::Instant::now();
    let mut receipt = check(session, &fixture)?;
    eprintln!(
        "{}: oracle checks in {:.1}s",
        case.name,
        checked.elapsed().as_secs_f64()
    );

    // Graph replay, a second eager run and a run behind a PDL producer must
    // match bit for bit, each on poisoned outputs and from the state the
    // step began with.
    let restore = || -> LabResult<()> {
        fixture
            .d_ring
            .copy_from_bytes(&ring_before, &session.stream)
            .map_err(|e| e.to_string())?;
        fixture
            .d_ring_positions
            .copy_from_bytes(&ring_positions_before, &session.stream)
            .map_err(|e| e.to_string())
    };
    let graph = session.capture(|stream| fixture.launch_step(stream, &kernels))?;
    checks.graph(&graph).map_err(labelled)?;
    checks
        .relaunch(|| fixture.launch_step(&session.stream, &kernels))
        .map_err(labelled)?;
    let probe = session.module("probe_sm120")?;
    let producer = probe
        .function("chop_probe_pdl_first_v1")
        .map_err(|e| e.to_string())?;
    let stamps = session.allocate(32)?;
    checks
        .pdl(
            || {
                producer
                    .launch(
                        &session.stream,
                        launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?,
                        &[
                            KernelArgument::Buffer(&stamps),
                            KernelArgument::U64(100_000),
                        ],
                    )
                    .map_err(|e| e.to_string())
            },
            || fixture.launch_step(&session.stream, &kernels),
            pdl_enabled(),
        )
        .map_err(labelled)?;
    checks.record(&mut receipt);
    receipt.insert("case".into(), json!(case.name));
    receipt.insert("rows".into(), json!(fixture.step.rows));
    receipt.insert("sequences".into(), json!(case.sequences));
    receipt.insert(
        "kv_dtype".into(),
        json!(if case.fp8 { "fp8" } else { "bf16" }),
    );
    receipt.insert("splits".into(), json!(fixture.splits(&kernels.attention.0)));
    receipt.insert("correct".into(), json!(true));
    fixture.topology.mark(&mut receipt);
    session.receipt("sparse_attention", receipt.clone())?;
    eprintln!(
        "{}: correct, graph, determinism and PDL bitwise ({} sparse rows, {} threshold ties)",
        case.name, receipt["sparse_rows"], receipt["threshold_tie_rows"]
    );
    if bench {
        restore()?;
        benchmark(session, &fixture, &kernels)?;
    }
    if case.phase == AttentionPhase::Verify {
        restore()?;
        fixture.launch_step(&session.stream, &kernels)?;
        rollback(session, modules, fixture)?;
    }
    Ok(())
}

/// After a verify step, speculation keeps some of each item's rows and
/// decodes a new token at the next position. The new rows overwrite the
/// rejected ones' cache slots; the step must match the oracle of the kept
/// sequence, so the ring still holds every key an open block needs.
fn rollback(session: &Session, modules: &Modules, mut fixture: Fixture) -> LabResult<()> {
    let page_size = fixture.topology.geometry.page_size as usize;
    let block_size = fixture.topology.geometry.block as usize;
    let head_dim = fixture.topology.geometry.head_dim as usize;
    let index_heads = fixture.topology.geometry.index_heads as usize;
    let index_dim = fixture.topology.geometry.index_dim as usize;
    let mut rng = Rng(0x0b5e_55ed_0000_0001 ^ fixture.step.rows as u64);
    let element = fixture.element();
    let page_bytes = page_size * head_dim * element;
    let mut step = Step {
        rows: 0,
        row_batch: Vec::new(),
        positions: Vec::new(),
        rope: Vec::new(),
        projection: Vec::new(),
        q: Vec::new(),
    };
    for (b, sequence) in fixture.sequences.iter_mut().enumerate() {
        // Items keep 1 to rows - 1 rows (a one-row item redecodes its
        // position), so the new token lands inside the verified span and a
        // kept open block can sit up to ten positions behind a rejected row.
        let p = sequence.cached
            + (1 + b % sequence.rows.saturating_sub(1).max(1)).min(sequence.rows - 1);
        sequence.keys.truncate(p * index_dim);
        sequence.keys.extend((0..index_dim).map(|_| rng.bf16(2.0)));
        sequence.rope.truncate(p + 1);
        for (plane, stride) in [(0, page_bytes * 3 / 2), (1, page_bytes * 2)] {
            let fresh: Vec<u8> = (0..head_dim * element)
                .map(|_| rng.next() as u8 & 0x3f)
                .collect();
            sequence.rewrite_kv(plane, p, &fresh);
            let buffer = if plane == 0 {
                &fixture.d_k
            } else {
                &fixture.d_v
            };
            let at = sequence.pages[p / page_size] as usize * stride
                + (p % page_size) * head_dim * element;
            buffer
                .copy_from_bytes_at(at, &fresh, &session.stream)
                .map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
        }
        sequence.cached = p;
        sequence.rows = 1;
        step.row_batch.push(b as i32);
        step.positions.push(p as i32);
        step.rope.push(sequence.rope[p]);
        step.projection
            .extend((0..index_heads * index_dim).map(|_| rng.bf16(2.0)));
        step.projection
            .extend_from_slice(&sequence.keys[p * index_dim..(p + 1) * index_dim]);
        step.q
            .extend((0..fixture.topology.heads * head_dim).map(|_| rng.bf16(2.0)));
    }
    step.rows = step.row_batch.len();
    fixture
        .d_q
        .copy_from_bytes(&bytes_u16(&step.q), &session.stream)
        .map_err(|e| e.to_string())?;
    fixture.step = step;
    // The verify step's outputs fill the rows past the new ones.
    fixture.padded_rows = fixture.step.rows;
    fixture.case.phase = AttentionPhase::Decode;
    fixture.case.name = format!("{}_rollback", fixture.case.name);
    fixture.prepared = [1, 8, 32]
        .into_iter()
        .find(|&bucket| bucket as usize >= fixture.step.rows)
        .ok_or("rollback rows exceed the decode buckets")?;
    fixture.ring_checked = block_size;
    // The plans cover the kept sequences' contexts.
    fixture.context = fixture
        .sequences
        .iter()
        .map(Sequence::total)
        .max()
        .ok_or("a rollback without sequences")? as u32;
    let kernels = Kernels::load(modules, &fixture.case)?;
    fixture.upload_step(session)?;
    let mut checks = fixture.replay(session, &kernels)?;
    checks
        .eager(|| fixture.launch_step(&session.stream, &kernels))
        .map_err(|e| format!("{}: {e}", fixture.case.name))?;
    let mut receipt = check(session, &fixture)?;
    receipt.insert("case".into(), json!(fixture.case.name));
    receipt.insert("rows".into(), json!(fixture.step.rows));
    receipt.insert("correct".into(), json!(true));
    fixture.topology.mark(&mut receipt);
    session.receipt("sparse_attention", receipt)?;
    eprintln!("{}: correct", fixture.case.name);
    Ok(())
}

fn find_peak(ordinal: i32) -> (f64, f64, String) {
    let directory = kernel_lab::receipt_root().join("probes");
    let latest = fs::read_dir(directory)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let value: Value = serde_json::from_slice(&fs::read(entry.path()).ok()?).ok()?;
            (value["ordinal"].as_i64() == Some(i64::from(ordinal))
                && value.pointer("/dram_read/gb_per_second").is_some())
            .then_some((entry, value))
        })
        .max_by_key(|(entry, _)| entry.file_name());
    if let Some((entry, value)) = latest
        && let Some(dram) = value
            .pointer("/dram_read/gb_per_second")
            .and_then(Value::as_f64)
        && let Some(l2) = value
            .pointer("/l2_read/gb_per_second")
            .and_then(Value::as_f64)
    {
        return (dram, l2, entry.path().display().to_string());
    }
    // Without a probe receipt, assume nominal SM120 DRAM and L2 read peaks (GB/s).
    (1330.0, 3570.0, "fallback: nominal DRAM and L2 peaks".into())
}

/// Logical bytes of each stage, for the roofline.
struct Traffic {
    key_write: f64,
    query: f64,
    scores: f64,
    topk: f64,
    attention: f64,
    /// A dense attention step's KV reads: every visible key of every row
    /// (decode and verify) or of every sequence once (prefill).
    dense: f64,
}

fn traffic(fixture: &Fixture, score_rows: usize) -> Traffic {
    let block_size = fixture.topology.geometry.block as usize;
    let top_blocks = fixture.topology.geometry.top_blocks as usize;
    let head_dim = fixture.topology.geometry.head_dim as usize;
    let index_heads = fixture.topology.geometry.index_heads as usize;
    let index_dim = fixture.topology.geometry.index_dim as usize;
    let rows = fixture.step.rows;
    let element = fixture.element() as f64;
    let mut scores = 0.0;
    let mut topk = 0.0;
    let mut attention = 0.0;
    let mut dense = 0.0;
    for sequence in &fixture.sequences {
        let mut group_max = 0;
        for i in 0..sequence.rows {
            let visible = sequence.cached + i + 1;
            let complete = visible / block_size;
            if complete > top_blocks {
                group_max = group_max.max(complete);
                scores += complete as f64 * 4.0;
                topk += complete as f64 * 4.0 + top_blocks as f64 * 4.0;
            }
            if (i + 1) % score_rows == 0 || i + 1 == sequence.rows {
                scores += group_max as f64 * index_dim as f64 * 2.0;
                group_max = 0;
            }
            let keys = complete.min(top_blocks) * block_size + visible - complete * block_size;
            attention += keys as f64 * head_dim as f64 * 2.0 * element;
            if fixture.case.phase != AttentionPhase::Prefill {
                dense += visible as f64 * head_dim as f64 * 2.0 * element;
            }
        }
        if fixture.case.phase == AttentionPhase::Prefill {
            dense += sequence.total() as f64 * head_dim as f64 * 2.0 * element;
        }
    }
    let q_bytes = rows as f64 * fixture.topology.heads as f64 * head_dim as f64 * 2.0;
    attention += 2.0 * q_bytes;
    Traffic {
        key_write: rows as f64 * index_dim as f64 * 2.0 * 1.25,
        query: rows as f64 * index_heads as f64 * index_dim as f64 * 4.0,
        scores,
        topk,
        attention,
        dense: dense + 2.0 * q_bytes,
    }
}

fn benchmark(session: &Session, fixture: &Fixture, kernels: &Kernels) -> LabResult<()> {
    let (dram, l2, peak_source) = find_peak(session.ordinal);
    let probe = session.module("probe_sm120")?;
    let tiny = probe
        .function("chop_probe_sm120_v1")
        .map_err(|e| e.to_string())?;
    let tiny_output = session.allocate(4)?;
    let launch_floor = kernel_lab::time_named(session, "launch_floor", 2, 5, |stream| {
        tiny.launch(
            stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?,
            &[KernelArgument::Buffer(&tiny_output)],
        )
        .map_err(|e| e.to_string())
    })?
    .median_ms
        * 1000.0;
    let score_rows = match fixture.case.phase {
        AttentionPhase::Decode => 1,
        AttentionPhase::Verify => 8,
        AttentionPhase::Prefill => 16,
    };
    let bytes = traffic(fixture, score_rows);
    let mut receipt = BTreeMap::new();
    receipt.insert("case".into(), json!(fixture.case.name));
    receipt.insert("peak_source".into(), json!(peak_source));
    receipt.insert("dram_gb_per_second".into(), json!(dram));
    receipt.insert("l2_gb_per_second".into(), json!(l2));
    receipt.insert("launch_floor_us".into(), json!(launch_floor));
    receipt.insert("splits".into(), json!(fixture.splits(&kernels.attention.0)));
    receipt.insert(
        "kv_dtype".into(),
        json!(if fixture.case.fp8 { "fp8" } else { "bf16" }),
    );
    let mut stage =
        |name: &str, logical: f64, work: &dyn Fn(&Stream) -> LabResult<()>| -> LabResult<f64> {
            let measured = kernel_lab::time_named(
                session,
                &format!("{}::{name}", fixture.case.name),
                3,
                9,
                work,
            )?;
            let us = f64::from(measured.median_ms) * 1000.0;
            receipt.insert(
                name.into(),
                json!({"median_us": us, "min_us": f64::from(measured.min_ms) * 1000.0,
                "logical_bytes": logical,
                "dram_roofline_percent": logical / (us * 1e-6) / (dram * 1e9) * 100.0}),
            );
            Ok(us)
        };
    let key_write = stage("key_write", bytes.key_write, &|stream| {
        fixture.launch_key_write(stream, kernels, fixture.prepared)
    })?;
    let query = stage("query", bytes.query, &|stream| {
        fixture.launch_query(stream, kernels)
    })?;
    let scores = stage("scores", bytes.scores, &|stream| {
        fixture.launch_scores(stream, kernels)
    })?;
    let topk = stage("topk", bytes.topk, &|stream| {
        fixture.launch_topk(stream, kernels)
    })?;
    let attention = stage("attention", bytes.attention, &|stream| {
        fixture.launch_attention(stream, kernels)
    })?;
    let total = bytes.key_write + bytes.query + bytes.scores + bytes.topk + bytes.attention;
    let step = stage("step", total, &|stream| {
        fixture.launch_step(stream, kernels)
    })?;
    // The same step replayed as one CUDA graph, as a bundle launches it.
    let graph = session.capture(|stream| fixture.launch_step(stream, kernels))?;
    let graph_step = stage("step_graph", total, &|stream| {
        graph.launch(stream).map_err(|e| e.to_string())
    })?;
    let dense_floor_us = bytes.dense / (dram * 1e9) * 1e6;
    receipt.insert("dense_kv_bytes".into(), json!(bytes.dense));
    receipt.insert("dense_dram_floor_us".into(), json!(dense_floor_us));
    receipt.insert(
        "dense_floor_over_sparse_step".into(),
        json!(dense_floor_us / graph_step),
    );
    receipt.insert(
        "indexer_share_of_stages".into(),
        json!(
            (key_write + query + scores + topk) / (key_write + query + scores + topk + attention)
        ),
    );
    fixture.topology.mark(&mut receipt);
    let path = session.receipt("sparse_attention_bench", receipt)?;
    eprintln!(
        "{} bench: key_write {key_write:.1} us, query {query:.1}, scores {scores:.1}, topk {topk:.1}, attention {attention:.1}, step {step:.1}, graph {graph_step:.1} (dense DRAM floor {dense_floor_us:.1} us, launch floor {launch_floor:.1}) -> {}",
        fixture.case.name,
        path.display()
    );
    Ok(())
}

#[test]
fn qsa_indexer_and_sparse_attention_match_oracles() {
    let bench = std::env::var("CHOPINFER_QSA_BENCH").as_deref() == Ok("1");
    let result = gpu_test(8 << 30, |session| {
        let modules = Modules::load(session)?;
        for case in run_cases(modules.topology.lean)
            .iter()
            .filter(|case| selected(case))
        {
            run_case(session, &modules, case, bench)?;
        }
        Ok(())
    });
    result.expect("QSA indexer and sparse attention");
}

/// Each shape runs the full context/batch/verify/chunk matrix. The filters
/// split device qualification into bounded jobs without changing the cases.
#[test]
fn qsa_lab_indexer_shapes_match_oracles() {
    gpu_test(8 << 30, |session| {
        let cases: Vec<_> = cases().into_iter().filter(selected).collect();
        if cases.is_empty() {
            return Err("CHOPINFER_QSA_CASE selects no lab indexer cases".into());
        }
        for topology in lab_indexer_topologies() {
            let modules = Modules::for_topology(session, topology)?;
            for case in &cases {
                eprintln!("{}: {}", topology.indexer_profile.tag(), case.name);
                run_case(session, &modules, case, false)?;
            }
        }
        Ok(())
    })
    .expect("lab indexer shapes against their own scores and oracles");
}

/// Direct operands isolate late heads, ReLU before head reduction, forced
/// F32 ties, the zero threshold, and the selector's overflow-bin fallback.
#[test]
fn qsa_lab_indexer_shapes_adversarial_selection() {
    gpu_test(64 << 20, |session| {
        for topology in lab_indexer_topologies() {
            let modules = Modules::for_topology(session, topology)?;
            let module = &modules.indexer;
            let heads = topology.geometry.index_heads as usize;
            // Keep the fixed block-cache probe layout explicit.
            assert_eq!(topology.geometry.index_dim as usize, INDEX_DIM);
            assert_eq!(topology.geometry.block as usize, BLOCK);
            let top = topology.geometry.top_blocks as usize;
            assert_eq!(topology.geometry.page_size as usize, PAGE);
            for head in [0, heads - 1] {
                for k in 20..=60 {
                    let (base, keys) = boundary_pair(top, 2_f32.powi(-k), 1.5 * 2_f32.powi(-k));
                    let mut q = vec![0; heads * INDEX_DIM];
                    q[head * INDEX_DIM..(head + 1) * INDEX_DIM].copy_from_slice(&base[..INDEX_DIM]);
                    let label = format!("head {head} forced tie 2^-{k}");
                    let results = check_direct_selection(
                        session,
                        module,
                        topology,
                        &label,
                        &q,
                        &keys,
                        (top + 1) * BLOCK - 1,
                    )?;
                    if k >= 25 && results.iter().any(|r| r.1[top - 1] != r.1[top]) {
                        return Err(format!("{label}: expected equal F32 boundary scores"));
                    }
                }
            }
            let mut q = vec![0; heads * INDEX_DIM];
            q[64] = f32_to_bf16(1.0);
            for head in 1..heads {
                q[head * INDEX_DIM + 64] = f32_to_bf16(-8.0);
            }
            let keys: Vec<_> = (0..4096 * INDEX_DIM)
                .map(|i| f32_to_bf16(if i % INDEX_DIM == 64 { 1.0 } else { 0.0 }))
                .collect();
            let results = check_direct_selection(
                session,
                module,
                topology,
                "ReLU before head sum, all equal",
                &q,
                &keys,
                4096 * BLOCK - 1,
            )?;
            if results
                .iter()
                .any(|r| !r.0 || r.1.iter().any(|&s| s <= 0.0))
            {
                return Err("all-equal positive scores must keep the lowest blocks".into());
            }
            let q: Vec<_> = (0..heads * INDEX_DIM)
                .map(|i| f32_to_bf16(if i % INDEX_DIM < 8 { 1.0 } else { 0.0 }))
                .collect();
            let mut rng = Rng(0x2e40_7e5a);
            let spacing = 8192 / top;
            let zeros = 4096 - 4096 / spacing;
            let keys: Vec<_> = (0..4096 * INDEX_DIM)
                .map(|i| {
                    let x = rng.uniform(1.0);
                    f32_to_bf16(match (i % INDEX_DIM < 8, i / INDEX_DIM % spacing == 0) {
                        (true, true) => x.abs(),
                        (true, false) => -x.abs() - 0.01,
                        _ => x,
                    })
                })
                .collect();
            let results = check_direct_selection(
                session,
                module,
                topology,
                &format!("{zeros} exact zeros"),
                &q,
                &keys,
                4096 * BLOCK - 1,
            )?;
            if results
                .iter()
                .any(|r| !r.0 || r.1.iter().filter(|s| s.to_bits() == 0).count() != zeros)
            {
                return Err(
                    "zero threshold must have the expected exact +0 scores and exact selection"
                        .into(),
                );
            }
            check_topk_scores(session, module, topology)?;
        }
        Ok(())
    })
    .expect("lab indexer exact F32 selection on adversarial operands");
}

/// A fixture's pages moved to high ids by `relocate`.
struct Relocated {
    pool: SentinelPool,
    /// Each low page and the high id that holds it.
    moved: Vec<(u64, u64)>,
    groups: Value,
}

/// Moves every page the fixture's sequences use into one sentinel pool of
/// Qwen serving pages, each twelve layers of K, V and block-key planes with
/// this case's layer first, at the ids `pool_offsets::high_ids` picks, and
/// points the fixture's planes and block tables there.
fn relocate(session: &Session, fixture: &mut Fixture) -> LabResult<Relocated> {
    let page_size = fixture.topology.geometry.page_size as usize;
    let block_size = fixture.topology.geometry.block as usize;
    let head_dim = fixture.topology.geometry.head_dim as usize;
    let index_dim = fixture.topology.geometry.index_dim as usize;
    let kv_block = (page_size * head_dim * fixture.element()) as u64;
    let key_block = (page_size / block_size * index_dim * 2) as u64;
    let stride = 12 * (2 * kv_block + key_block);
    let mut pages: Vec<u64> = fixture
        .sequences
        .iter()
        .flat_map(|sequence| sequence.pages.iter().map(|&page| page as u64))
        .collect();
    pages.sort_unstable();
    pages.dedup();
    // Block keys are BF16 whatever the KV dtype.
    let ids = pool_offsets::high_ids(stride, 2, pages.len().div_ceil(2));
    let moved: Vec<(u64, u64)> = pages
        .iter()
        .copied()
        .zip(ids.alternating(pages.len()))
        .collect();
    let entries = moved.iter().map(|&(_, to)| to).max().unwrap_or(0) + 1;
    let mut pool = SentinelPool::new(session, entries * stride)?;
    for &(from, to) in &moved {
        pool.copy_from(
            session,
            to * stride,
            &fixture.d_k,
            from * fixture.k_stride,
            kv_block,
        )?;
        pool.copy_from(
            session,
            to * stride + kv_block,
            &fixture.d_v,
            from * fixture.v_stride,
            kv_block,
        )?;
        // The key writes fill this page's block keys.
        pool.mark(to * stride + 2 * kv_block, key_block);
    }
    let map: BTreeMap<u64, u64> = moved.iter().copied().collect();
    let max_pages = fixture.max_pages as usize;
    let mut tables = vec![0_i32; fixture.sequences.len() * max_pages];
    for (b, sequence) in fixture.sequences.iter().enumerate() {
        for (p, &page) in sequence.pages.iter().enumerate() {
            tables[b * max_pages + p] = pool_offsets::i32_ids(&[map[&(page as u64)]])[0];
        }
    }
    fixture.d_tables = upload(session, &bytes_i32(&tables))?;
    fixture.d_k = pool.buffer.clone();
    fixture.d_v = pool.buffer.clone();
    fixture.d_block_keys = pool.buffer.clone();
    fixture.plane_offsets = [0, kv_block, 2 * kv_block];
    (fixture.k_stride, fixture.v_stride, fixture.block_stride) = (stride, stride, stride);
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(Relocated {
        pool,
        moved,
        groups: ids.describe(),
    })
}

/// The QSA key write, block scores, selection and sparse attention with
/// every page past the 32-bit lines (`pool_offsets`), K, V and block keys
/// sharing one pool of Qwen serving pages: every block key, the ring, the
/// indexer queries, scores, selection and output equal the low-id run's bit
/// for bit, and no other pool byte changes. The low run keeps the oracle
/// checks.
#[test]
fn qsa_high_ids_match_low_ids() {
    gpu_test(pool_offsets::need_bytes(9 << 30), |session| {
        let modules = Modules::load(session)?;
        for name in ["decode_mixed_b8", "verify_fp8_c8192_b8", "prefill_mixed"] {
            let case = cases()
                .into_iter()
                .find(|case| case.name == name)
                .ok_or_else(|| format!("missing QSA case {name}"))?;
            let kernels = Kernels::load(&modules, &case)?;
            let step = |fixture: &Fixture| -> LabResult<Vec<u8>> {
                fixture.write_history(session, &kernels)?;
                fixture.upload_step(session)?;
                fixture.launch_step(&session.stream, &kernels)?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                fixture.outputs(session)
            };
            let low = prepare(session, &case, modules.topology)?;
            let low_outputs = step(&low)?;
            check(session, &low)?;
            let mut high = prepare(session, &case, modules.topology)?;
            let Relocated {
                pool,
                moved,
                groups,
            } = relocate(session, &mut high)?;
            let high_outputs = step(&high)?;
            pool_offsets::same(
                &case.name,
                "indexer queries, scores, selection, output and ring",
                &low_outputs,
                &high_outputs,
            )?;
            let key_block = (PAGE / BLOCK * INDEX_DIM * 2) as u64;
            for &(from, to) in &moved {
                pool_offsets::same(
                    &case.name,
                    &format!("block keys of page {to}"),
                    &pool_offsets::read(
                        session,
                        &low.d_block_keys,
                        from * BLOCK_PAGE_STRIDE,
                        key_block,
                    )?,
                    &pool.read(
                        session,
                        to * high.block_stride + high.plane_offsets[2],
                        key_block,
                    )?,
                )?;
            }
            let scanned = pool.check_untouched(session, &case.name)?;
            let mut receipt = json!({
                "pages": groups,
                "pages_moved": moved.len(),
                "pool_bytes": pool.bytes(),
                "kv_dtype": if case.fp8 { "fp8" } else { "bf16" },
                "rows": high.step.rows,
                "outputs_bitwise": true,
                "block_keys_bitwise": true,
                "sentinel_bytes_checked": scanned,
            });
            if modules.topology.lean {
                receipt["lean"] = json!(true);
            }
            pool_offsets::receipt(session, "qsa", &case.name, receipt)?;
        }
        Ok(())
    })
    .expect("QSA at high page ids");
}

/// The wide Qwen decode cases: 33 to 256 rows of mixed contexts in
/// the bundle's graph buckets of 64, 128 and 256 rows, each through the full
/// step, its oracle checks, the untouched rows past the live count and the
/// graph, determinism and PDL replays. Without a GPU it still checks that
/// every entry plans each bucket at the catalog's own capacity and that the
/// attention splits, so the merge runs.
#[test]
fn qsa_wide_decode_matches_oracles() {
    let topology = topology().expect("Qwen3.8 topology");
    let cases = wide_cases(topology.lean);
    for case in &cases {
        let rows = case.sequences.len();
        let prepared = prepared_rows(case.phase, rows);
        assert!(
            prepared.is_power_of_two() && (rows..2 * rows).contains(&(prepared as usize)),
            "{}: {rows} rows in a {prepared} row bucket",
            case.name
        );
        let context = case.sequences.iter().map(|&(c, r)| c + r).max().unwrap() as u32;
        let attention = topology
            .attention(if case.fp8 { "fp8" } else { "bf16" }, "decode")
            .expect("attention entry");
        let indexer = [
            "index_query",
            "index_key_write",
            "block_scores_decode",
            "block_topk",
        ];
        for entry in indexer.map(entry).iter().chain([&attention]) {
            if let Err(refusal) = entry.plan(prepared, context) {
                panic!(
                    "{}: {} refuses {prepared} rows: {}",
                    case.name, entry.name, refusal.0
                );
            }
        }
        assert!(
            attention.splits(prepared) > 1 && attention.merge_plan(prepared).unwrap().is_some(),
            "{}: the attention of {prepared} rows does not split",
            case.name
        );
    }
    let bench = std::env::var("CHOPINFER_QSA_BENCH").as_deref() == Ok("1");
    let result = gpu_test(8 << 30, |session| {
        let modules = Modules::load(session)?;
        for case in cases.iter().filter(|case| selected(case)) {
            run_case(session, &modules, case, bench)?;
        }
        Ok(())
    });
    result.expect("QSA wide decode");
}

/// Launches QSA entry `name` for `prepared` rows that see at most `context`
/// tokens, and waits for it.
fn launch_named(
    session: &Session,
    module: &Module,
    name: &str,
    prepared: u32,
    context: u32,
    arguments: &[KernelArgument<'_>],
) -> LabResult<()> {
    let entry = entry(name);
    launch_entry(session, module, &entry, prepared, context, arguments)
}

fn launch_entry(
    session: &Session,
    module: &Module,
    entry: &Entry,
    prepared: u32,
    context: u32,
    arguments: &[KernelArgument<'_>],
) -> LabResult<()> {
    let function = function(module, &entry.symbol(), entry.dynamic_shared_bytes())?;
    let plan = entry.plan(prepared, context).map_err(|e| e.0.to_string())?;
    function
        .launch(&session.stream, config(plan)?, arguments)
        .map_err(|e| e.to_string())?;
    kernel_lab::launched(&entry.symbol());
    session.stream.synchronize().map_err(|e| e.to_string())
}

/// The review's pooling cancellation: a block's raw keys have 2^25, 1,
/// -2^25 and 1 at coordinate 64 and 0.5 at 65. An F32 sum in position order
/// loses the first 1 and pools (0.25, 0.5); the oracle's F64 sum pools (0.5,
/// 0.5), which the zero-weight norm maps to (8, 8). One launch pools the
/// block from the projection rows; splits 1 + 3, 2 + 2 and 3 + 1 complete it
/// from the ring. The stored block key must match the key-write oracle.
#[test]
fn qsa_block_key_pooling_keeps_cancelled_terms() {
    gpu_test(64 << 20, |session| {
        let module = module(session)?;
        let geometry = model_qwen38_contract::sm120::attention::SPARSE;
        let mut raw = vec![0_u16; BLOCK * INDEX_DIM];
        for (t, x) in [2_f32.powi(25), 1.0, -(2_f32.powi(25)), 1.0]
            .into_iter()
            .enumerate()
        {
            raw[t * INDEX_DIM + 64] = f32_to_bf16(x);
            raw[t * INDEX_DIM + 65] = f32_to_bf16(0.5);
        }
        let rope: Vec<[i32; 3]> = (0..BLOCK as i32).map(|p| [p, p, p]).collect();
        let mut ring = vec![0.0; RING * INDEX_DIM];
        let mut ring_rope = vec![[0; 3]; RING];
        let mut block_keys = vec![0.0; INDEX_DIM];
        let writes = qsa::indexer_key_write(
            SparseAttentionQuery::IndexerKeyWrite {
                head_dim: 128,
                rotary_dim: 64,
                theta_bits: geometry.theta.to_bits(),
                mrope: geometry.mrope,
                eps_bits: geometry.eps.to_bits(),
                block: BLOCK as u32,
                ring: RING as u32,
                capacity: BLOCK as u32,
            },
            &decode_bf16(&raw),
            &(0..BLOCK).collect::<Vec<_>>(),
            &rope
                .iter()
                .map(|p| p.map(|x| x as usize))
                .collect::<Vec<_>>(),
            &[0.0; INDEX_DIM],
            IndexerKeyCache {
                ring_keys: &mut ring,
                ring_positions: &mut ring_rope,
                block_keys: &mut block_keys,
            },
        )
        .map_err(|e| e.0.to_string())?;
        let d_weight = upload(session, &bytes_u16(&[0; INDEX_DIM]))?;
        let d_cos_sin = upload(session, &bytes_f32(&cos_sin_table(BLOCK, &geometry)))?;
        let d_slots = upload(session, &bytes_i32(&[0]))?;
        let d_table = upload(session, &bytes_i32(&[0]))?;
        let d_no_anchor = upload(session, &bytes_i32(&[-1]))?;
        for split in [&[4][..], &[1, 3], &[2, 2], &[3, 1]] {
            let d_ring = upload(session, &vec![0xff; RING * INDEX_DIM * 2])?;
            let d_ring_positions = upload(session, &[0xff; RING * 3 * 4])?;
            let d_block_keys = upload(session, &vec![0xff; BLOCK_PAGE_STRIDE as usize])?;
            let mut first = 0;
            for &rows in split {
                let mut projection = Vec::with_capacity(rows * WIDTH);
                for t in first..first + rows {
                    projection.extend(std::iter::repeat_n(0, INDEX_HEADS * INDEX_DIM));
                    projection.extend_from_slice(&raw[t * INDEX_DIM..(t + 1) * INDEX_DIM]);
                }
                let d_projection = upload(session, &bytes_u16(&projection))?;
                let d_rope = upload(
                    session,
                    &bytes_i32(
                        &rope[first..first + rows]
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                    ),
                )?;
                let d_row_batch = upload(session, &bytes_i32(&vec![0; rows]))?;
                let d_positions = upload(
                    session,
                    &bytes_i32(&(first as i32..(first + rows) as i32).collect::<Vec<_>>()),
                )?;
                let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
                launch_named(
                    session,
                    &module,
                    "index_key_write",
                    BLOCK as u32,
                    BLOCK as u32,
                    &[
                        KernelArgument::Buffer(&d_projection),
                        KernelArgument::Buffer(&d_weight),
                        KernelArgument::Buffer(&d_rope),
                        KernelArgument::Buffer(&d_cos_sin),
                        KernelArgument::Buffer(&d_row_batch),
                        KernelArgument::Buffer(&d_positions),
                        KernelArgument::Buffer(&d_live),
                        KernelArgument::Buffer(&d_slots),
                        KernelArgument::Buffer(&d_ring),
                        KernelArgument::Buffer(&d_ring_positions),
                        KernelArgument::Buffer(&d_no_anchor),
                        KernelArgument::Buffer(&d_no_anchor),
                        KernelArgument::Buffer(&d_no_anchor),
                        KernelArgument::Buffer(&d_no_anchor),
                        KernelArgument::Buffer(&d_table),
                        KernelArgument::Buffer(&d_block_keys),
                        KernelArgument::U32(1),
                        KernelArgument::U64(BLOCK_PAGE_STRIDE),
                        KernelArgument::U32(geometry.eps.to_bits()),
                    ],
                )?;
                first += rows;
            }
            let key = as_u16(&download(session, &d_block_keys, INDEX_DIM * 2)?);
            let actual: Vec<f32> = key.iter().map(|&k| bf16_to_f32(k)).collect();
            let mut receipt = BTreeMap::new();
            check_output(&actual, &writes.keys, DType::Bf16, &mut receipt).map_err(|e| {
                fail(
                    "pooling cancellation",
                    &format!("split {split:?}"),
                    &receipt,
                    e,
                )
            })?;
            if actual[64] != 8.0 || actual[65] != 8.0 {
                return Err(format!(
                    "split {split:?}: pooled coordinates 64 and 65 are {} and {}, not 8 and 8",
                    actual[64], actual[65]
                ));
            }
            eprintln!("pooling cancellation, split {split:?}: block key (8, 8)");
        }
        Ok(())
    })
    .expect("QSA block-key pooling under cancellation");
}

/// Blocks written directly to a block-key cache whose physical pages are
/// 0, 1, ... in order, and the one-sequence table that maps them.
fn upload_block_keys(
    session: &Session,
    keys: &[u16],
) -> LabResult<(DeviceBuffer, DeviceBuffer, u32)> {
    let blocks = keys.len() / INDEX_DIM;
    let pages = blocks.div_ceil(PAGE / BLOCK);
    let mut bytes = vec![0xff_u8; pages * BLOCK_PAGE_STRIDE as usize];
    for block in 0..blocks {
        let at = block / (PAGE / BLOCK) * BLOCK_PAGE_STRIDE as usize
            + block % (PAGE / BLOCK) * INDEX_DIM * 2;
        bytes[at..at + INDEX_DIM * 2].copy_from_slice(&bytes_u16(
            &keys[block * INDEX_DIM..(block + 1) * INDEX_DIM],
        ));
    }
    let table: Vec<i32> = (0..pages as i32).collect();
    Ok((
        upload(session, &bytes)?,
        upload(session, &bytes_i32(&table))?,
        pages as u32,
    ))
}

/// The top-k key set of `scores` (one per complete block), ties to the lower
/// block, as tokens.
fn topk_tokens(scores: &[f64], visible: usize) -> LabResult<Vec<usize>> {
    topk_tokens_at(scores, visible, TOP)
}

fn topk_tokens_at(scores: &[f64], visible: usize, top: usize) -> LabResult<Vec<usize>> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    qsa::block_topk(
        SparseAttentionQuery::BlockTopK {
            block: BLOCK as u32,
            top_blocks: top as u32,
            max_blocks: geometry.max_blocks(),
            capacity: 1,
        },
        scores,
        visible,
    )
    .map_err(|e| e.0.to_string())
}

/// One scoring entry's phase, key set as tokens and F32 scores.
type EntrySelection = (&'static str, Vec<usize>, Vec<f32>);

/// Runs every scoring entry and the top-k on one row at `position` over a
/// query and a block-key cache already uploaded (`upload_block_keys`).
/// Returns, per entry, its phase, key set as tokens and F32 scores; each set
/// must equal the top-k of its own F32 scores.
fn direct_selection(
    session: &Session,
    module: &Module,
    topology: Topology,
    label: &str,
    q: &[u16],
    (d_keys, d_table, max_pages): (&DeviceBuffer, &DeviceBuffer, u32),
    position: usize,
) -> LabResult<Vec<EntrySelection>> {
    let geometry = topology.geometry;
    let top = geometry.top_blocks as usize;
    let visible = position + 1;
    let blocks = visible / BLOCK;
    let score_stride = geometry.score_stride(visible as u32);
    let d_q = upload(session, &bytes_u16(q))?;
    let d_row_batch = upload(session, &bytes_i32(&[0]))?;
    let d_positions = upload(session, &bytes_i32(&[position as i32]))?;
    let d_live = upload(session, &1_u32.to_ne_bytes())?;
    let mut results = Vec::new();
    for phase in ["decode", "verify", "prefill"] {
        let d_scores = upload(session, &vec![0xff; score_stride as usize * 4])?;
        let d_selected = upload(session, &vec![0xff; top * 4])?;
        launch_entry(
            session,
            module,
            &topology.indexer(&format!("block_scores_{phase}")),
            1,
            visible as u32,
            &[
                KernelArgument::Buffer(&d_q),
                KernelArgument::Buffer(d_keys),
                KernelArgument::Buffer(d_table),
                KernelArgument::Buffer(&d_row_batch),
                KernelArgument::Buffer(&d_positions),
                KernelArgument::Buffer(&d_live),
                KernelArgument::Buffer(&d_scores),
                KernelArgument::U32(max_pages),
                KernelArgument::U64(BLOCK_PAGE_STRIDE),
                KernelArgument::U32(score_stride),
            ],
        )?;
        launch_entry(
            session,
            module,
            &topology.indexer("block_topk"),
            1,
            visible as u32,
            &[
                KernelArgument::Buffer(&d_scores),
                KernelArgument::Buffer(&d_positions),
                KernelArgument::Buffer(&d_live),
                KernelArgument::Buffer(&d_selected),
                KernelArgument::U32(score_stride),
            ],
        )?;
        let scores = as_f32(&download(session, &d_scores, blocks * 4)?);
        let selected = as_i32(&download(session, &d_selected, top.min(blocks) * 4)?);
        let mut tokens: Vec<usize> = selected
            .iter()
            .flat_map(|&b| b as usize * BLOCK..(b as usize + 1) * BLOCK)
            .collect();
        tokens.extend(blocks * BLOCK..visible);
        let f32_tokens = topk_tokens_at(
            &scores.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
            visible,
            top,
        )?;
        if tokens != f32_tokens {
            return Err(format!(
                "{label}, {phase} scores: top-k differs from the top-k of its F32 scores"
            ));
        }
        results.push((phase, tokens, scores));
    }
    Ok(results)
}

/// Runs every scoring entry and the top-k on one row at `position` over a
/// query and block keys written directly. The numerical contract accepts selection
/// from the F32 scores: each entry's key set must equal the top-k of its own
/// scores. Returns, per entry, whether that set also equals the oracle's
/// top-k of its F64 scores, and the F32 scores.
fn check_direct_selection(
    session: &Session,
    module: &Module,
    topology: Topology,
    label: &str,
    q: &[u16],
    keys: &[u16],
    position: usize,
) -> LabResult<Vec<(bool, Vec<f32>)>> {
    let geometry = topology.geometry;
    let top = geometry.top_blocks as usize;
    let visible = position + 1;
    let (d_keys, d_table, max_pages) = upload_block_keys(session, keys)?;
    let exact = qsa::block_scores(
        SparseAttentionQuery::BlockScores {
            heads: geometry.index_heads,
            head_dim: geometry.index_dim,
            max_blocks: geometry.max_blocks(),
            capacity: 1,
        },
        &decode_bf16(q),
        &decode_bf16(keys),
    )
    .map_err(|e| e.0.to_string())?;
    let exact_tokens = topk_tokens_at(&exact.values, visible, top)?;
    let mut results = Vec::new();
    for (phase, tokens, scores) in direct_selection(
        session,
        module,
        topology,
        label,
        q,
        (&d_keys, &d_table, max_pages),
        position,
    )? {
        let mut score_receipt = BTreeMap::new();
        check_output(&scores, &exact, DType::F32, &mut score_receipt)
            .map_err(|e| fail(label, phase, &score_receipt, e))?;
        let matches_exact = tokens == exact_tokens;
        session.receipt(
            "sparse_attention",
            BTreeMap::from([
                ("case".into(), json!(format!("{label}, {phase} scores"))),
                ("selection".into(), json!("f32")),
                ("top_blocks".into(), json!(top)),
                (
                    "indexer_profile".into(),
                    json!(topology.indexer_profile.tag()),
                ),
                ("matches_f64_oracle".into(), json!(matches_exact)),
                ("correct".into(), json!(true)),
            ]),
        )?;
        results.push((matches_exact, scores));
    }
    Ok(results)
}

/// One row with top + 1 blocks. Head 0 has ones at coordinates 64 and 65;
/// the last two blocks straddle the selection boundary with low/high scores.
fn boundary_pair(top: usize, low: f32, high: f32) -> (Vec<u16>, Vec<u16>) {
    let mut q = vec![0_u16; INDEX_HEADS * INDEX_DIM];
    q[64] = f32_to_bf16(1.0);
    q[65] = f32_to_bf16(1.0);
    let mut keys = vec![0_u16; (top + 1) * INDEX_DIM];
    for block in 0..=top {
        let (x, y) = match block {
            b if b == top - 1 => (1.0, low),
            b if b == top => (1.0, high),
            _ => (2.0, 0.0),
        };
        keys[block * INDEX_DIM + 64] = f32_to_bf16(x);
        keys[block * INDEX_DIM + 65] = f32_to_bf16(y);
    }
    (q, keys)
}

/// The review's F32 near tie at the selection boundary: blocks 511 and 512 of
/// `boundary_pair` with 2^-26 and 2^-25 have exact scores 1.3e-9 apart but
/// equal F32 scores, so the F32 selection keeps block 511 where the exact
/// one keeps 512 (The numerical contract accepts this). Reversed, both keep 511.
/// Then forced ties of 2^-k for k from 20 to 60. Every scoring entry's
/// selection must be the top-k of its F32 scores; whether it matches the
/// exact oracle is recorded.
#[test]
fn qsa_f32_selection_at_near_ties() {
    gpu_test(64 << 20, |session| {
        let module = module(session)?;
        let mut cases = vec![
            ("near tie".to_owned(), 2_f32.powi(-26), 2_f32.powi(-25)),
            (
                "near tie, reversed".to_owned(),
                2_f32.powi(-25),
                2_f32.powi(-26),
            ),
        ];
        for k in [20, 30, 34, 40, 50, 60] {
            cases.push((
                format!("forced tie 2^-{k}"),
                2_f32.powi(-k),
                1.5 * 2_f32.powi(-k),
            ));
        }
        for (label, low, high) in cases {
            let (q, keys) = boundary_pair(TOP, low, high);
            let results =
                check_direct_selection(session, &module, topology()?, &label, &q, &keys, 2051)?;
            eprintln!(
                "{label}: F32 top-k (scores {} and {}); matches the F64 oracle per entry: {:?}",
                results[0].1[511],
                results[0].1[512],
                results.iter().map(|r| r.0).collect::<Vec<_>>()
            );
        }
        Ok(())
    })
    .expect("QSA F32 selection at near ties");
}

/// A zero threshold: 4096 blocks of which only every sixteenth scores above
/// zero, so T is +0 and 3840 blocks tie exactly at zero in F32 and in F64;
/// the selection keeps the positive blocks, then the lowest zeros, and
/// equals the F64 oracle's.
#[test]
fn qsa_f32_selection_at_a_zero_threshold() {
    gpu_test(64 << 20, |session| {
        let module = module(session)?;
        let mut rng = Rng(0x2e40_7e5a);
        let q: Vec<u16> = (0..INDEX_HEADS * INDEX_DIM)
            .map(|i| f32_to_bf16(if i % INDEX_DIM < 8 { 1.0 } else { 0.0 }))
            .collect();
        let keys: Vec<u16> = (0..4096 * INDEX_DIM)
            .map(|i| {
                let x = rng.uniform(1.0);
                f32_to_bf16(match (i % INDEX_DIM < 8, i / INDEX_DIM % 16 == 0) {
                    (true, true) => x.abs(),
                    (true, false) => -x.abs() - 0.01,
                    _ => x,
                })
            })
            .collect();
        let results = check_direct_selection(
            session,
            &module,
            topology()?,
            "zero threshold",
            &q,
            &keys,
            4096 * BLOCK - 1,
        )?;
        if results.iter().any(|r| !r.0) {
            return Err("zero threshold: F32 selection differs from the F64 oracle".into());
        }
        eprintln!("zero threshold: F32 selection equals the F64 oracle's");
        Ok(())
    })
    .expect("QSA F32 selection at a zero threshold");
}

/// One exported full-attention layer of a `qwen38-oracle qsa-capture
/// --export` directory: a bundle dump's projection rows and the engine
/// operands the oracle emulates from them, with the kernel emulation's and
/// the F64 key sets of the exported rows. The model crate's export test
/// reads its export back under these same rules.
struct ExportedLayer {
    layer: u64,
    /// Every row's BF16 projection, [tokens, 5 * 128].
    qk: Vec<u16>,
    norms: Vec<u16>,
    /// The emulated block keys, [tokens / 4, 128].
    keys: Vec<u16>,
    rows: Vec<usize>,
    /// The emulated queries of `rows`, [rows, 4 * 128].
    q: Vec<u16>,
    exact: Vec<Vec<usize>>,
    kernel: Vec<Vec<usize>>,
}

impl ExportedLayer {
    fn read(dir: &std::path::Path, layer: u64) -> LabResult<Self> {
        let bits = |name: &str| -> LabResult<Vec<u16>> {
            let path = dir.join(format!("layer{layer:02}.{name}"));
            let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes.len() % 2 != 0 {
                return Err(format!(
                    "{}: odd byte count {}",
                    path.display(),
                    bytes.len()
                ));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect())
        };
        let manifest: Value = serde_json::from_slice(
            &fs::read(dir.join(format!("layer{layer:02}.json"))).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if manifest["table"] != "hf" {
            return Err(format!(
                "layer {layer}: operands use table {}, the kernels read the HF table",
                manifest["table"]
            ));
        }
        let field = |name: &str| manifest[name].clone();
        let exported = Self {
            layer,
            qk: bits("qk.bf16")?,
            norms: bits("norm.bf16")?,
            keys: bits("keys.bf16")?,
            q: bits("q.bf16")?,
            rows: serde_json::from_value(field("rows")).map_err(|e| e.to_string())?,
            exact: serde_json::from_value(field("exact")).map_err(|e| e.to_string())?,
            kernel: serde_json::from_value(field("kernel")).map_err(|e| e.to_string())?,
        };
        let tokens = exported.qk.len() / WIDTH;
        let n = exported.rows.len();
        let sparse = |&row: &usize| row < tokens && (row + 1) / BLOCK > TOP;
        // A set is TOP ascending distinct blocks of its row's complete blocks.
        let canonical = |set: &Vec<usize>, row: usize| {
            set.len() == TOP
                && set.windows(2).all(|w| w[0] < w[1])
                && set.last().is_some_and(|&b| b < (row + 1) / BLOCK)
        };
        if n == 0
            || exported.qk.len() != tokens * WIDTH
            || exported.norms.len() != 2 * INDEX_DIM
            || exported.keys.len() != tokens / BLOCK * INDEX_DIM
            || exported.q.len() != n * INDEX_HEADS * INDEX_DIM
            || exported.exact.len() != n
            || exported.kernel.len() != n
            || !exported.rows.iter().all(sparse)
        {
            return Err(format!(
                "layer {layer}: the export holds no sparse rows or its shapes disagree"
            ));
        }
        for (index, &row) in exported.rows.iter().enumerate() {
            if !canonical(&exported.exact[index], row) || !canonical(&exported.kernel[index], row) {
                return Err(format!(
                    "layer {layer} row {row}: a key set is not {TOP} ascending distinct blocks \
                     of the row"
                ));
            }
        }
        Ok(exported)
    }
}

/// Runs `index_key_write` over every exported row in launches of
/// `HISTORY_ROWS` and `index_query` on the exported rows, with the HF table,
/// and returns the device's block keys and queries.
fn device_operands(
    session: &Session,
    module: &Module,
    exported: &ExportedLayer,
) -> LabResult<(Vec<u16>, Vec<u16>)> {
    let geometry = model_qwen38_contract::sm120::attention::SPARSE;
    let eps = geometry.eps.to_bits();
    let tokens = exported.qk.len() / WIDTH;
    let context = tokens as u32;
    let d_cos_sin = upload(session, &bytes_f32(&cos_sin_table(tokens, &geometry)))?;
    let d_q_norm = upload(session, &bytes_u16(&exported.norms[..INDEX_DIM]))?;
    let d_k_norm = upload(session, &bytes_u16(&exported.norms[INDEX_DIM..]))?;
    let blocks = tokens / BLOCK;
    let pages = blocks.div_ceil(PAGE / BLOCK).max(1);
    let d_block_keys = upload(session, &vec![0xff; pages * BLOCK_PAGE_STRIDE as usize])?;
    let d_table = upload(session, &bytes_i32(&(0..pages as i32).collect::<Vec<_>>()))?;
    let d_slots = upload(session, &bytes_i32(&[0]))?;
    let d_ring = upload(session, &vec![0xff; RING * INDEX_DIM * 2])?;
    let d_ring_positions = upload(session, &[0xff; RING * 3 * 4])?;
    let d_no_anchor = upload(session, &bytes_i32(&[-1]))?;
    let mut first = 0;
    while first < tokens {
        let rows = HISTORY_ROWS.min(tokens - first);
        let d_projection = upload(
            session,
            &bytes_u16(&exported.qk[first * WIDTH..(first + rows) * WIDTH]),
        )?;
        let positions: Vec<i32> = (first as i32..(first + rows) as i32).collect();
        let rope: Vec<i32> = positions.iter().flat_map(|&p| [p, p, p]).collect();
        let d_rope = upload(session, &bytes_i32(&rope))?;
        let d_row_batch = upload(session, &bytes_i32(&vec![0; rows]))?;
        let d_positions = upload(session, &bytes_i32(&positions))?;
        let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
        launch_named(
            session,
            module,
            "index_key_write",
            rows as u32,
            context,
            &[
                KernelArgument::Buffer(&d_projection),
                KernelArgument::Buffer(&d_k_norm),
                KernelArgument::Buffer(&d_rope),
                KernelArgument::Buffer(&d_cos_sin),
                KernelArgument::Buffer(&d_row_batch),
                KernelArgument::Buffer(&d_positions),
                KernelArgument::Buffer(&d_live),
                KernelArgument::Buffer(&d_slots),
                KernelArgument::Buffer(&d_ring),
                KernelArgument::Buffer(&d_ring_positions),
                KernelArgument::Buffer(&d_no_anchor),
                KernelArgument::Buffer(&d_no_anchor),
                KernelArgument::Buffer(&d_no_anchor),
                KernelArgument::Buffer(&d_no_anchor),
                KernelArgument::Buffer(&d_table),
                KernelArgument::Buffer(&d_block_keys),
                KernelArgument::U32(pages as u32),
                KernelArgument::U64(BLOCK_PAGE_STRIDE),
                KernelArgument::U32(eps),
            ],
        )?;
        first += rows;
    }
    let paged = as_u16(&download(
        session,
        &d_block_keys,
        pages * BLOCK_PAGE_STRIDE as usize,
    )?);
    let stride = BLOCK_PAGE_STRIDE as usize / 2;
    let keys = (0..blocks)
        .flat_map(|b| {
            let at = b / (PAGE / BLOCK) * stride + b % (PAGE / BLOCK) * INDEX_DIM;
            paged[at..at + INDEX_DIM].to_vec()
        })
        .collect();
    let mut q = Vec::with_capacity(exported.q.len());
    for chunk in exported.rows.chunks(HISTORY_ROWS) {
        let projection: Vec<u16> = chunk
            .iter()
            .flat_map(|&r| exported.qk[r * WIDTH..(r + 1) * WIDTH].to_vec())
            .collect();
        let rope: Vec<i32> = chunk.iter().flat_map(|&r| [r as i32; 3]).collect();
        let d_projection = upload(session, &bytes_u16(&projection))?;
        let d_rope = upload(session, &bytes_i32(&rope))?;
        let d_live = upload(session, &(chunk.len() as u32).to_ne_bytes())?;
        let d_q = upload(
            session,
            &vec![0xff; chunk.len() * INDEX_HEADS * INDEX_DIM * 2],
        )?;
        launch_named(
            session,
            module,
            "index_query",
            chunk.len() as u32,
            context,
            &[
                KernelArgument::Buffer(&d_projection),
                KernelArgument::Buffer(&d_q_norm),
                KernelArgument::Buffer(&d_rope),
                KernelArgument::Buffer(&d_cos_sin),
                KernelArgument::Buffer(&d_live),
                KernelArgument::Buffer(&d_q),
                KernelArgument::U32(eps),
            ],
        )?;
        q.extend(as_u16(&download(
            session,
            &d_q,
            chunk.len() * INDEX_HEADS * INDEX_DIM * 2,
        )?));
    }
    Ok((keys, q))
}

/// The F32 selection on a model's operands. `CHOPINFER_QSA_OPERANDS` names a
/// `qwen38-oracle qsa-capture --export` directory: per full-attention layer
/// of a bundle dump, every row's BF16 projection as the engine computed it,
/// the engine's queries and block keys as the oracle emulates them from
/// those rows (policy `engine`), and for the exported rows the F64 key sets
/// of those operands and the emulated kernel's sets. Per layer:
///
/// - `index_key_write` and `index_query` run on the projection rows; their
///   BF16 outputs are compared bit for bit with the emulated operands.
/// - Every scoring entry and the top-k run on each exported row of the
///   device's own operands. Each set must be the top-k of its own F32
///   scores, and the scores must meet the F32 bound against F64 scores of
///   the same operands (`check_output`). A set may differ from the F64 top-k
///   only by in-bound near ties: every F64-kept block the device drops must
///   score within the two blocks' bounds of every block it keeps instead.
///
/// The result separates what is required from what is measured: the bound,
/// the near-tie rule and a nonzero row count must hold (the test fails
/// otherwise); the operand mismatches, the rows whose device set differs
/// from the F64 set or from the emulated kernel, and the rows whose F64 set
/// of the device's operands differs from the exported one
/// (`export_consistent`) are counted. The emulation counts as validated only
/// where all of them are zero. With
/// `CHOPINFER_QSA_OPERANDS_OUT` the summary is written to that new file.
#[test]
fn qsa_f32_selection_on_oracle_operands() {
    let Some(dir) = std::env::var_os("CHOPINFER_QSA_OPERANDS").map(std::path::PathBuf::from) else {
        kernel_lab::skip("the QSA oracle-operand check: CHOPINFER_QSA_OPERANDS is unset")
            .expect("QSA F32 selection on oracle operands");
        return;
    };
    let out = std::env::var_os("CHOPINFER_QSA_OPERANDS_OUT").map(std::path::PathBuf::from);
    gpu_test(2 << 30, |session| {
        let module = module(session)?;
        let geometry = model_qwen38_contract::sm120::attention::SPARSE;
        let mut layers: Vec<u64> = fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().into_string().ok()?;
                name.strip_prefix("layer")?
                    .strip_suffix(".json")?
                    .parse()
                    .ok()
            })
            .collect();
        layers.sort_unstable();
        if layers.is_empty() {
            return Err(format!("{}: no layer manifests", dir.display()));
        }
        let mut summary = Vec::new();
        let mut checked = 0;
        for layer in layers {
            let exported = ExportedLayer::read(&dir, layer)?;
            let (keys, q) = device_operands(session, &module, &exported)?;
            let mismatches = |a: &[u16], b: &[u16]| a.iter().zip(b).filter(|(x, y)| x != y).count();
            let key_mismatches = mismatches(&keys, &exported.keys);
            let query_mismatches = mismatches(&q, &exported.q);
            let device_keys = upload_block_keys(session, &keys)?;
            let blocks_of = |tokens: &[usize]| -> Vec<usize> {
                tokens[..TOP * BLOCK]
                    .iter()
                    .step_by(BLOCK)
                    .map(|t| t / BLOCK)
                    .collect()
            };
            let mut per_entry = BTreeMap::<&str, [usize; 4]>::new();
            let mut examples = Vec::new();
            for (index, &row) in exported.rows.iter().enumerate() {
                let complete = (row + 1) / BLOCK;
                let width = INDEX_HEADS * INDEX_DIM;
                let q_row = &q[index * width..(index + 1) * width];
                let reference = qsa::block_scores(
                    SparseAttentionQuery::BlockScores {
                        heads: INDEX_HEADS as u32,
                        head_dim: INDEX_DIM as u32,
                        max_blocks: geometry.max_blocks(),
                        capacity: 1,
                    },
                    &decode_bf16(q_row),
                    &decode_bf16(&keys[..complete * INDEX_DIM]),
                )
                .map_err(|e| e.0.to_string())?;
                let exact = blocks_of(&topk_tokens(&reference.values, row + 1)?);
                let tolerance = |b: usize| {
                    accumulation_bound(
                        reference.reduction_lengths[b],
                        reference.sum_abs_products[b],
                    )
                };
                let label = format!("layer {layer} row {row}");
                let results = direct_selection(
                    session,
                    &module,
                    topology()?,
                    &label,
                    q_row,
                    (&device_keys.0, &device_keys.1, device_keys.2),
                    row,
                )?;
                for (phase, tokens, scores) in results {
                    let mut receipt = BTreeMap::new();
                    check_output(&scores, &reference, DType::F32, &mut receipt)
                        .map_err(|e| fail(&label, &format!("{phase} scores"), &receipt, e))?;
                    let device = blocks_of(&tokens);
                    let dropped: Vec<usize> = exact
                        .iter()
                        .filter(|b| !device.contains(b))
                        .copied()
                        .collect();
                    let added: Vec<usize> = device
                        .iter()
                        .filter(|b| !exact.contains(b))
                        .copied()
                        .collect();
                    for &d in &dropped {
                        for &a in &added {
                            let gap = reference.values[d] - reference.values[a];
                            if gap > tolerance(d) + tolerance(a) {
                                return Err(format!(
                                    "{label}, {phase}: keeps block {a} over {d}, {gap:e} apart, \
                                     outside the scores' bounds"
                                ));
                            }
                        }
                    }
                    let counts = per_entry.entry(phase).or_default();
                    counts[0] += 1;
                    let off_f64 = device != exact;
                    let off_emulation = device != exported.kernel[index];
                    counts[1] += usize::from(off_f64);
                    counts[2] += usize::from(off_emulation);
                    counts[3] += usize::from(exact != exported.exact[index]);
                    if (off_f64 || off_emulation) && examples.len() < 16 {
                        examples.push(json!({
                            "row": row,
                            "entry": phase,
                            "differs_from_f64": off_f64,
                            "differs_from_emulation": off_emulation,
                            "emulation_differs_from_f64":
                                exported.kernel[index] != exported.exact[index],
                        }));
                    }
                }
                checked += 1;
            }
            let entries: BTreeMap<String, Value> = per_entry
                .iter()
                .map(|(phase, [rows, f64_rows, emulated, exported_f64])| {
                    (
                        phase.to_string(),
                        json!({"rows": rows, "differ_from_f64": f64_rows,
                               "differ_from_emulation": emulated,
                               "f64_differs_from_exported_f64": exported_f64}),
                    )
                })
                .collect();
            let emulation_off = exported
                .kernel
                .iter()
                .zip(&exported.exact)
                .filter(|(k, s)| k != s)
                .count();
            let set_disagreements = per_entry.values().map(|c| c[2]).sum::<usize>();
            // The F64 sets of the device's operands must match the exported
            // F64 sets before the emulated sets are compared with them.
            let export_disagreements = per_entry.values().map(|c| c[3]).sum::<usize>();
            let validated = key_mismatches == 0
                && query_mismatches == 0
                && set_disagreements == 0
                && export_disagreements == 0;
            eprintln!(
                "layer {layer}: {} rows, key/query mismatches {key_mismatches}/{query_mismatches}, \
                 emulation differs from F64 on {emulation_off}; {entries:?}",
                exported.rows.len()
            );
            let record = json!({
                "layer": exported.layer,
                "rows": exported.rows.len(),
                "block_key_elements": keys.len(),
                "block_key_mismatches": key_mismatches,
                "query_elements": q.len(),
                "query_mismatches": query_mismatches,
                "emulation_differs_from_f64": emulation_off,
                "export_consistent": export_disagreements == 0,
                "entries": entries,
                "emulation": if validated { "validated" } else { "differs from the device" },
                "examples": examples,
            });
            session.receipt(
                "sparse_attention",
                BTreeMap::from([
                    (
                        "case".into(),
                        json!(format!("oracle operands, layer {layer}")),
                    ),
                    ("selection".into(), json!("f32")),
                    ("operands".into(), json!(dir.display().to_string())),
                    ("result".into(), record.clone()),
                    ("correct".into(), json!(true)),
                ]),
            )?;
            summary.push(record);
        }
        if checked == 0 {
            return Err(format!("{}: no row was checked", dir.display()));
        }
        if let Some(out) = &out {
            use std::io::Write;
            let value = json!({"operands": dir.display().to_string(), "layers": summary});
            fs::File::create_new(out)
                .and_then(|mut f| f.write_all(&serde_json::to_vec_pretty(&value).expect("json")))
                .map_err(|e| format!("{}: {e}", out.display()))?;
        }
        Ok(())
    })
    .expect("QSA F32 selection on oracle operands");
}

/// The export reader refuses layers without sparse rows, with shapes that
/// disagree or with a table the kernels do not read, so the operand check
/// cannot pass without checking a row.
#[test]
fn qsa_operand_exports_refuse_empty_or_malformed_layers() {
    let dir = std::env::temp_dir().join(format!("qsa-operands-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let tokens = (TOP + 1) * BLOCK;
    let write = |name: &str, elements: usize| {
        fs::write(
            dir.join(format!("layer03.{name}")),
            vec![0_u8; 2 * elements],
        )
        .unwrap();
    };
    write("qk.bf16", tokens * WIDTH);
    write("norm.bf16", 2 * INDEX_DIM);
    write("keys.bf16", tokens / BLOCK * INDEX_DIM);
    let canonical: Vec<usize> = (0..TOP).collect();
    let manifest_with = |rows: &[usize], table: &str, set: &[usize]| {
        let value = json!({"layer": 3, "table": table, "rows": rows,
            "exact": vec![set; rows.len()], "kernel": vec![&canonical; rows.len()],
            "gap": vec![0.0; rows.len()]});
        fs::write(dir.join("layer03.json"), value.to_string()).unwrap();
        write("q.bf16", rows.len() * INDEX_HEADS * INDEX_DIM);
    };
    let manifest = |rows: &[usize], table: &str| manifest_with(rows, table, &canonical);
    let row = tokens - 1;
    manifest(&[row], "hf");
    assert!(ExportedLayer::read(&dir, 3).is_ok());
    manifest(&[], "hf");
    assert!(ExportedLayer::read(&dir, 3).is_err(), "no rows");
    manifest(&[TOP * BLOCK], "hf");
    assert!(ExportedLayer::read(&dir, 3).is_err(), "a dense row");
    manifest(&[row], "f64-angle");
    assert!(ExportedLayer::read(&dir, 3).is_err(), "another table");
    let mut descending = canonical.clone();
    descending.swap(0, 1);
    let mut repeated = canonical.clone();
    repeated[1] = 0;
    let mut outside = canonical.clone();
    outside[TOP - 1] = TOP + 1;
    for (set, case) in [
        (&descending, "a set out of order"),
        (&repeated, "a repeated block"),
        (&outside, "a block past the row"),
    ] {
        manifest_with(&[row], "hf", set);
        assert!(ExportedLayer::read(&dir, 3).is_err(), "{case}");
    }
    manifest(&[row], "hf");
    fs::write(
        dir.join("layer03.qk.bf16"),
        vec![0_u8; 2 * tokens * WIDTH + 1],
    )
    .unwrap();
    assert!(ExportedLayer::read(&dir, 3).is_err(), "an odd byte count");
    write("qk.bf16", tokens * WIDTH);
    write("keys.bf16", 1);
    assert!(ExportedLayer::read(&dir, 3).is_err(), "short keys");
    fs::remove_dir_all(&dir).unwrap();
}

/// The selector alone on F32 scores written directly: distinct scores, a few
/// levels with heavy ties (T's bin larger than the gather), one bin holding
/// every block, all equal, and all zero. The selection must equal the
/// oracle's top-k of those scores.
#[test]
fn qsa_topk_selects_exact_f32_scores() {
    gpu_test(64 << 20, |session| {
        check_topk_scores(session, &module(session)?, topology()?)
    })
    .expect("QSA top-k on exact F32 scores");
}

fn check_topk_scores(session: &Session, module: &Module, topology: Topology) -> LabResult<()> {
    let geometry = topology.geometry;
    let top = geometry.top_blocks as usize;
    let stride = geometry.max_blocks() as usize;
    let mut rng = Rng(0x5e1e_c7ed);
    let rows: Vec<(usize, Vec<f32>)> = vec![
        (
            65_536,
            (0..65_536)
                .map(|_| f32::from_bits((rng.next() >> 40) as u32 % 0x3f80_0000))
                .collect(),
        ),
        (
            65_536,
            (0..65_536)
                .map(|_| (rng.next() % 64) as f32 / 16.0)
                .collect(),
        ),
        (
            20_000,
            (0..20_000)
                .map(|_| 1.0 + (rng.next() % 4096) as f32 / 16_384.0)
                .collect(),
        ),
        (3_000, vec![0.5; 3_000]),
        (513, vec![0.0; 513]),
    ];
    let mut scores = vec![0_f32; rows.len() * stride];
    for (row, (_, values)) in rows.iter().enumerate() {
        scores[row * stride..][..values.len()].copy_from_slice(values);
    }
    let positions: Vec<i32> = rows
        .iter()
        .map(|(blocks, _)| (blocks * BLOCK - 1) as i32)
        .collect();
    let d_scores = upload(session, &bytes_f32(&scores))?;
    let d_positions = upload(session, &bytes_i32(&positions))?;
    let d_live = upload(session, &(rows.len() as u32).to_ne_bytes())?;
    let d_selected = upload(session, &vec![0xff; rows.len() * top * 4])?;
    launch_entry(
        session,
        module,
        &topology.indexer("block_topk"),
        rows.len() as u32,
        geometry.max_context,
        &[
            KernelArgument::Buffer(&d_scores),
            KernelArgument::Buffer(&d_positions),
            KernelArgument::Buffer(&d_live),
            KernelArgument::Buffer(&d_selected),
            KernelArgument::U32(stride as u32),
        ],
    )?;
    let selected = as_i32(&download(session, &d_selected, rows.len() * top * 4)?);
    for (row, (blocks, values)) in rows.iter().enumerate() {
        let expected = topk_tokens_at(
            &values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            blocks * BLOCK,
            top,
        )?;
        let tokens: Vec<usize> = selected[row * top..(row + 1) * top]
            .iter()
            .flat_map(|&b| b as usize * BLOCK..(b as usize + 1) * BLOCK)
            .collect();
        if tokens != expected {
            return Err(format!(
                "top-k row {row} ({blocks} blocks) differs from the oracle"
            ));
        }
    }
    eprintln!("top-k selector: {} rows exact", rows.len());
    session.receipt(
        "sparse_attention",
        BTreeMap::from([
            (
                "case".into(),
                json!("direct top-k distinct, heavy ties, all equal, zero"),
            ),
            (
                "indexer_profile".into(),
                json!(topology.indexer_profile.tag()),
            ),
            ("rows".into(), json!(rows.len())),
            ("selection".into(), json!("f32")),
            ("top_blocks".into(), json!(top)),
            ("correct".into(), json!(true)),
        ]),
    )?;
    Ok(())
}

#[test]
fn qsa_catalog_covers_every_phase() {
    let entries: Vec<_> = catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|e| !kernel_lab::lab_shapes::PROFILES.contains(&e.profile))
        .collect();
    for phase in ["prefill", "decode", "verify"] {
        for dtype in ["bf16", "fp8"] {
            let attention = entry(&format!("attn_{dtype}_{phase}"));
            assert_eq!(attention.kind, Kind::Attention);
            assert!(attention.merge_symbol().is_some());
        }
        assert!(matches!(
            entry(&format!("block_scores_{phase}")).kind,
            Kind::Scores { .. }
        ));
    }
    // The attention entries the selected topology's step runs.
    let topology = topology().expect("Qwen3.8 topology");
    for phase in ["prefill", "decode", "verify"] {
        for dtype in ["bf16", "fp8"] {
            let attention = topology.attention(dtype, phase).expect("attention entry");
            assert_eq!(attention.kind, Kind::Attention);
            assert!(attention.merge_symbol().is_some());
        }
    }
    // TP4's twelve entries, then TP2's six; the
    // indexer entries are rank-invariant.
    let count = |profile| entries.iter().filter(|e| e.profile == profile).count();
    assert_eq!(count(model_qwen38_contract::sm120::config::PROFILE), 12);
    for profile in model_qwen38_contract::sm120::PROFILES[1..].iter().copied() {
        assert_eq!(count(profile), 6);
    }
    assert_eq!(entries.len(), 18);
}

/// The lean cases (all selected topologies' cases; TP2 runs only
/// these) launch each dtype's decode, verify and prefill attention, the
/// merge after more than one split at least once per dtype, and at least
/// one row that selects among more than `TOP` blocks.
#[test]
fn qsa_lean_cases_launch_every_entry_and_split() {
    let topology = topology().expect("Qwen3.8 topology");
    let lean = run_cases(true);
    assert_eq!(lean.len(), LEAN_CASES.len(), "a lean case is missing");
    for fp8 in [false, true] {
        let dtype = if fp8 { "fp8" } else { "bf16" };
        let mut split = false;
        for (phase, name) in [
            (AttentionPhase::Decode, "decode"),
            (AttentionPhase::Verify, "verify"),
            (AttentionPhase::Prefill, "prefill"),
        ] {
            let entry = topology.attention(dtype, name).expect("attention entry");
            let of_phase: Vec<&Case> = lean
                .iter()
                .filter(|case| case.fp8 == fp8 && case.phase == phase)
                .collect();
            assert!(!of_phase.is_empty(), "no lean {dtype} {name} case");
            for case in of_phase {
                let rows = case.sequences.iter().map(|&(_, rows)| rows).sum();
                split |= entry.splits(prepared_rows(phase, rows)) > 1;
                let context = case.sequences.iter().map(|&(c, r)| c + r).max().unwrap();
                assert!(
                    context / BLOCK > TOP,
                    "{}: no row selects among more than {TOP} blocks",
                    case.name
                );
            }
        }
        assert!(split, "no lean {dtype} case splits its attention");
    }
}

#[rustfmt::skip] #[path = "qsa_rope_positions.rs"] mod qsa_rope_positions;

/// A key write that keeps an anchor ring writes, into its anchor slot, the
/// raw keys and RoPE positions of the `RING` positions before the anchor end,
/// where a ring whose sequence ended there would hold them, and leaves the
/// live ring and block keys as a write without it does. Two launches of one
/// sequence: rows 0..70 anchored at 40, then rows 70..100 anchored at 90.
#[test]
fn qsa_key_write_keeps_an_anchor_ring() {
    gpu_test(64 << 20, |session| {
        let module = module(session)?;
        let geometry = model_qwen38_contract::sm120::attention::SPARSE;
        let total = 100;
        let mut state = 0x5eed_a7c4_u32;
        let raw = (0..total * INDEX_DIM)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f32_to_bf16(((state >> 8) as f32 / (1 << 24) as f32) * 2.0 - 1.0)
            })
            .collect::<Vec<_>>();
        let rope: Vec<[i32; 3]> = (0..total as i32).map(|p| [p, p + 1, p + 2]).collect();
        let d_weight = upload(session, &bytes_u16(&[0; INDEX_DIM]))?;
        let d_cos_sin = upload(session, &bytes_f32(&cos_sin_table(total, &geometry)))?;
        let d_slots = upload(session, &bytes_i32(&[0]))?;
        let pages = total.div_ceil(PAGE);
        let d_table = upload(session, &bytes_i32(&(0..pages as i32).collect::<Vec<_>>()))?;
        // Anchor slot 1 of two.
        let d_anchor_slots = upload(session, &bytes_i32(&[1]))?;
        let mut outputs = Vec::new();
        for anchored in [false, true] {
            let d_ring = upload(session, &vec![0xff; RING * INDEX_DIM * 2])?;
            let d_ring_positions = upload(session, &[0xff; RING * 3 * 4])?;
            let d_anchor_ring = upload(session, &vec![0xff; 2 * RING * INDEX_DIM * 2])?;
            let d_anchor_positions = upload(session, &[0xff; 2 * RING * 3 * 4])?;
            let d_block_keys = upload(session, &vec![0xff; pages * BLOCK_PAGE_STRIDE as usize])?;
            let mut kept = Vec::new();
            for (first, rows, end) in [(0_usize, 70_usize, 40_i32), (70, 30, 90)] {
                let mut projection = Vec::with_capacity(rows * WIDTH);
                for t in first..first + rows {
                    projection.extend(std::iter::repeat_n(0, INDEX_HEADS * INDEX_DIM));
                    projection.extend_from_slice(&raw[t * INDEX_DIM..(t + 1) * INDEX_DIM]);
                }
                let d_projection = upload(session, &bytes_u16(&projection))?;
                let d_rope = upload(
                    session,
                    &bytes_i32(
                        &rope[first..first + rows]
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                    ),
                )?;
                let d_row_batch = upload(session, &bytes_i32(&vec![0; rows]))?;
                let d_positions = upload(
                    session,
                    &bytes_i32(&(first as i32..(first + rows) as i32).collect::<Vec<_>>()),
                )?;
                let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
                let d_anchor_ends = upload(session, &bytes_i32(&[end]))?;
                let d_none = upload(session, &bytes_i32(&[-1]))?;
                launch_named(
                    session,
                    &module,
                    "index_key_write",
                    128,
                    total as u32,
                    &[
                        KernelArgument::Buffer(&d_projection),
                        KernelArgument::Buffer(&d_weight),
                        KernelArgument::Buffer(&d_rope),
                        KernelArgument::Buffer(&d_cos_sin),
                        KernelArgument::Buffer(&d_row_batch),
                        KernelArgument::Buffer(&d_positions),
                        KernelArgument::Buffer(&d_live),
                        KernelArgument::Buffer(&d_slots),
                        KernelArgument::Buffer(&d_ring),
                        KernelArgument::Buffer(&d_ring_positions),
                        KernelArgument::Buffer(if anchored { &d_anchor_slots } else { &d_none }),
                        KernelArgument::Buffer(&d_anchor_ends),
                        KernelArgument::Buffer(&d_anchor_ring),
                        KernelArgument::Buffer(&d_anchor_positions),
                        KernelArgument::Buffer(&d_table),
                        KernelArgument::Buffer(&d_block_keys),
                        KernelArgument::U32(pages as u32),
                        KernelArgument::U64(BLOCK_PAGE_STRIDE),
                        KernelArgument::U32(geometry.eps.to_bits()),
                    ],
                )?;
                if anchored {
                    let keys = as_u16(&download(
                        session,
                        &d_anchor_ring,
                        2 * RING * INDEX_DIM * 2,
                    )?);
                    let positions =
                        as_i32(&download(session, &d_anchor_positions, 2 * RING * 3 * 4)?);
                    kept.push((end, keys, positions));
                }
            }
            let mut state = download(session, &d_ring, RING * INDEX_DIM * 2)?;
            state.extend(download(session, &d_ring_positions, RING * 3 * 4)?);
            state.extend(download(
                session,
                &d_block_keys,
                pages * BLOCK_PAGE_STRIDE as usize,
            )?);
            outputs.push(state);
            for (end, keys, positions) in kept {
                let end = end as usize;
                for p in end - RING..end {
                    let row = RING + p % RING;
                    if keys[row * INDEX_DIM..(row + 1) * INDEX_DIM]
                        != raw[p * INDEX_DIM..(p + 1) * INDEX_DIM]
                        || positions[row * 3..row * 3 + 3] != rope[p]
                    {
                        return Err(format!("anchor end {end}: anchor ring row of {p} differs"));
                    }
                }
                if keys[..RING * INDEX_DIM].iter().any(|&k| k != 0xffff) {
                    return Err(format!("anchor end {end}: anchor slot 0 was written"));
                }
            }
        }
        if outputs[0] != outputs[1] {
            return Err("the live ring or block keys differ with an anchor".into());
        }
        eprintln!("anchor ring: rows 24..40 and 74..90 kept, live state unchanged");
        Ok(())
    })
    .expect("QSA key write keeps an anchor ring");
}

#[test]
fn small_sparse_fixture_matches_oracles() {
    gpu_test(256 * 1024 * 1024, |session| {
        let modules = Modules::load(session)?;
        for phase in [
            AttentionPhase::Prefill,
            AttentionPhase::Decode,
            AttentionPhase::Verify,
        ] {
            let case = Case {
                name: format!("small_sparse_{phase:?}"),
                phase,
                fp8: false,
                rounded: false,
                sequences: vec![(2052, 1), (7, 1)],
                pattern: Pattern::Random,
            };
            run_case(session, &modules, &case, false)?;
        }
        Ok(())
    })
    .expect("small sparse fixtures");
}

#[test]
fn lab_shapes_sparse_head_128_matches_oracles() {
    gpu_test(256 * 1024 * 1024, |session| {
        for profile in kernel_lab::lab_shapes::sparse_attention_profiles() {
            let modules = Modules::for_topology(session, lab_attention_topology(profile))?;
            for phase in [
                AttentionPhase::Prefill,
                AttentionPhase::Decode,
                AttentionPhase::Verify,
            ] {
                for fp8 in [false, true] {
                    let rows = if phase == AttentionPhase::Decode {
                        1
                    } else {
                        3
                    };
                    let case = Case {
                        name: format!("{}_{phase:?}_fp8_{fp8}", profile.tag()),
                        phase,
                        fp8,
                        rounded: fp8,
                        sequences: vec![(2052, rows), (63, rows)],
                        pattern: Pattern::Random,
                    };
                    run_case(session, &modules, &case, false)?;
                }
            }
        }
        Ok(())
    })
    .expect("head-128 sparse attention fixtures");
}

#[test]
fn lab_shapes_qualified_sparse_matches_oracles() {
    gpu_test(256 * 1024 * 1024, |session| {
        for s in kernel_lab::lab_shapes::NEW_SPARSE_SHAPES
            .iter()
            .filter(|s| s.attention)
        {
            let modules = Modules::for_topology(session, lab_attention_topology(s.profile))?;
            for phase in [
                AttentionPhase::Prefill,
                AttentionPhase::Decode,
                AttentionPhase::Verify,
            ] {
                for fp8 in [false, true] {
                    for context in [s.top_blocks * 4 + 3, s.top_blocks * 4 + 4] {
                        let rows = if phase == AttentionPhase::Decode {
                            1
                        } else {
                            4
                        };
                        let case = Case {
                            name: format!("{}_{phase:?}_c{context}_fp8_{fp8}", s.profile.tag()),
                            phase,
                            fp8,
                            rounded: fp8,
                            sequences: vec![(context as usize - rows, rows), (63, rows)],
                            pattern: Pattern::Random,
                        };
                        run_case(session, &modules, &case, false)?;
                    }
                }
            }
        }
        Ok(())
    })
    .expect("qualified sparse instances at the selection boundary");
}

#[test]
fn lab_sparse_profiles_use_matching_controls() {
    let topologies = kernel_lab::lab_shapes::indexer_shapes()
        .map(|(profile, heads, _)| lab_indexer_topology(profile, heads))
        .chain(kernel_lab::lab_shapes::sparse_attention_profiles().map(lab_attention_topology));
    for topology in topologies {
        for phase in ["prefill", "decode", "verify"] {
            for dtype in ["bf16", "fp8"] {
                let entry = topology.attention(dtype, phase).unwrap();
                let actual = entry.geometry;
                let expected = topology.geometry;
                assert_eq!(actual.q_heads, expected.q_heads);
                assert_eq!(actual.kv_heads, expected.kv_heads);
                assert_eq!(actual.head_dim, expected.head_dim);
                assert_eq!(actual.block, expected.block);
                assert_eq!(actual.top_blocks, expected.top_blocks);
                assert_eq!(actual.page_size, expected.page_size);
                assert_eq!(
                    actual.softmax_scale.to_bits(),
                    expected.softmax_scale.to_bits()
                );
            }
        }
    }
}

#[test]
fn rollback_rewrites_keep_rounded_kv_sources() {
    let head_dim = 4;
    let values: Vec<u16> = [
        0.3, -1.7, 0.02, 1.9, -0.6, 0.45, -0.011, 1.1, 0.8, -0.2, 1.3, -1.9,
    ]
    .map(f32_to_bf16)
    .to_vec();
    let rounded = |plane: usize| -> Vec<u8> {
        values
            .iter()
            .map(|&v| kv_fp8::encode(bf16_to_f32(v), FP8_SCALES[plane]))
            .collect()
    };
    let mut sequence = Sequence {
        head_dim,
        cached: 1,
        rows: 2,
        keys: Vec::new(),
        rope: Vec::new(),
        k: rounded(0),
        v: rounded(1),
        k_bf16: values.clone(),
        v_bf16: values.clone(),
        pages: Vec::new(),
        slot: 0,
    };
    for plane in [0, 1] {
        sequence.rewrite_kv(plane, 1, &[0x00, 0x05, 0x2a, 0x3f]);
    }
    // The FP8 check's unrounded reference must store the codes the cache holds.
    for (plane, codes, source) in [
        (0, &sequence.k, &sequence.k_bf16),
        (1, &sequence.v, &sequence.v_bf16),
    ] {
        let stored: Vec<u8> = source
            .iter()
            .map(|&v| kv_fp8::encode(bf16_to_f32(v), FP8_SCALES[plane]))
            .collect();
        assert_eq!(&stored, codes, "plane {plane}");
    }

    // BF16 caches keep no unrounded sources.
    let mut sequence = Sequence {
        k: bytes_u16(&values),
        v: bytes_u16(&values),
        k_bf16: Vec::new(),
        v_bf16: Vec::new(),
        ..sequence
    };
    for plane in [0, 1] {
        sequence.rewrite_kv(plane, 1, &[0x3f; 8]);
    }
    assert_eq!(sequence.k.len(), 16);
    assert!(sequence.k_bf16.is_empty() && sequence.v_bf16.is_empty());
}

#[test]
fn lab_shapes_wide_sparse_matches_oracles() {
    gpu_test(8 << 30, |session| {
        let cases: Vec<_> = wide_shape_cases().into_iter().filter(selected).collect();
        assert!(
            !cases.is_empty(),
            "CHOPINFER_QSA_CASE selects no wide cases"
        );
        for profile in kernel_lab::lab_shapes::WIDE_SPARSE_PROFILES {
            let modules = Modules::for_topology(session, lab_attention_topology(profile))?;
            for case in &cases {
                run_case(session, &modules, case, false)?;
            }
        }
        Ok(())
    })
    .expect("wide sparse instances at 256 kept blocks");
}

fn wide_shape_cases() -> Vec<Case> {
    let mut matrix = cases();
    for phase in [
        AttentionPhase::Prefill,
        AttentionPhase::Decode,
        AttentionPhase::Verify,
    ] {
        for fp8 in [false, true] {
            for context in [1027, 1028] {
                let rows = if phase == AttentionPhase::Decode {
                    1
                } else {
                    3
                };
                matrix.push(Case {
                    name: format!("wide_{phase:?}_c{context}_fp8_{fp8}"),
                    phase,
                    fp8,
                    rounded: fp8,
                    sequences: vec![(context - rows, rows)],
                    pattern: Pattern::Random,
                });
            }
        }
    }
    matrix
}

#[test]
fn wide_matrix_crosses_the_kept_block_boundary() {
    let matrix = wide_shape_cases();
    assert_eq!(matrix.len(), 45);
    for phase in [
        AttentionPhase::Prefill,
        AttentionPhase::Decode,
        AttentionPhase::Verify,
    ] {
        for fp8 in [false, true] {
            for context in [1027, 1028] {
                assert!(matrix.iter().any(|c| {
                    c.phase == phase
                        && c.fp8 == fp8
                        && c.sequences
                            .iter()
                            .any(|&(prefix, rows)| prefix + rows == context)
                }));
            }
        }
    }
}
