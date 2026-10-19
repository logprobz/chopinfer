//! Latent attention: the latent KV write, attention over gathered key lists
//! of a shared latent (K == V) with its split merge, and the head-batched
//! BF16 GEMMs that absorb the up projection (`cuda/latent/*.cuh`,
//! `cuda/latent/README.md`). Decode and verify rows cut their keys into
//! [`MAX_SPLITS`] fixed splits at every bucket, so a row's bits do not
//! depend on the rows that share its step. Contracts and oracles are kernel-core's
//! `latent_attention` and `latent_attention_oracle`.
//!
//! A model lists its entries with [`entries`] for its geometry and builds
//! them with [`cubin_spec`].
//!
//! Declared exactness ([`Entry::invariance`]): every entry is row-local.
//! The KV write stores each row's latent into its own cache slot (a warp
//! per row; an FP8 row's block scales from that row's own blocks), so it
//! claims one-row bits up to its capacity, any chunk split and row
//! independence; the cache is paged, so anchors do not apply. A prefill
//! attention row runs in one CTA over one split whose units follow the
//! launch's key budget alone, and keeps its bits at every row count and
//! chunk split (`Invariance::any_rows`). A decode or verify row cuts its
//! keys into [`MAX_SPLITS`] fixed splits at every bucket (the bucket only
//! groups splits into CTAs), each split from a fresh softmax state, and
//! its merge folds the splits in ascending order per row and head: one-row
//! bits up to the capacity and row independence, no chunk claim (they serve
//! no prefill). Decode and verify entries compile the same kernel (the
//! traits leave out the phase and capacity), so a verify row's one-row bits
//! are the decode entry's. The head-batched GEMMs add each output's 128-K
//! chunks in order, each an MMA chain from zero, with a row's M slot fixed
//! by its index: one-row bits up to the capacity, any chunk split and row
//! independence. Every claim is between launches of equal launch scalars
//! (the key budget, the softmax scale).

pub mod glue;
pub mod merge_columns;
pub mod query_parallel;
pub mod slices;
pub mod sources;
pub mod sources_prefill;
pub mod sources_prefill_cache;

use std::path::PathBuf;

use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    AttentionPhase, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::latent_attention::{
    HeadBatchedGemmQuery, HeadWeightLayout, LatentAttentionOutput, LatentAttentionQuery,
    LatentKvFormat,
};
use kernel_core::properties::Invariance;

const SMS: u32 = 188;
/// The splits of every decode and verify row; `max_splits` in
/// attention.cuh. The bucket only groups them into CTAs.
pub const MAX_SPLITS: u32 = 64;
/// Threads of a split merge CTA; `merge_threads` in attention.cuh.
const MERGE_THREADS: u32 = 128;
/// SM120's dynamic shared memory limit per CTA.
const SHARED_LIMIT: u32 = 101_376;
/// Heads one attention CTA takes; `Attention::head_tile`.
const HEAD_TILE: u32 = 16;
/// Rows one KV write CTA takes; `KvWrite::rows_per_cta`.
const WRITE_ROWS: u32 = 4;
/// The head-batched GEMM's tile; `HeadGemm::tile_m`, `tile_n` and `chunk`.
const GEMM_TILE_M: u32 = 64;
const GEMM_TILE_N: u32 = 64;
const GEMM_CHUNK: u32 = 128;
/// The widest latent row an SM120 latent entry takes.
const MAX_LATENT: u32 = 1024;
/// The most heads of an SM120 latent attention.
const MAX_HEADS: u32 = 128;

/// The latent attention geometry of one model profile at one rank count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Local query heads.
    pub heads: u32,
    /// Latent width, the shared key and value row.
    pub latent: u32,
    pub page_size: u32,
    /// Key list entries per row: the selected tokens plus the incomplete
    /// pool tail.
    pub max_keys: u32,
    pub softmax_scale: f32,
    /// Each head's query width before absorption (`W_UK_h` [nope, latent]).
    pub nope: u32,
    /// Each head's value width after projection (`W_UV_h` [v_dim, latent]).
    pub v_dim: u32,
}

/// An entry's operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Query {
    Latent(LatentAttentionQuery),
    HeadGemm(HeadBatchedGemmQuery),
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_{kernel}`; the symbol is `chop_latent_{name}_v1`.
    pub name: String,
    pub query: Query,
}

fn pointer(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}
fn u32_param(name: &str) -> EntryParam {
    EntryParam::U32 { name: name.into() }
}
fn u64_param(name: &str) -> EntryParam {
    EntryParam::U64 { name: name.into() }
}

/// The parts of an attention query the engine compiles.
struct AttentionShape {
    heads: u32,
    latent: u32,
    max_keys: u32,
    page_size: u32,
    fp8: bool,
    partial: bool,
    phase: AttentionPhase,
}

/// Dynamic shared bytes of an attention CTA with `keys`-key units;
/// `Attention::smem`.
fn attention_smem(latent: u32, fp8: bool, keys: u32) -> u32 {
    let blocks = latent / 128;
    let tile = keys * latent * 2;
    let stage = if fp8 {
        keys * latent + keys * blocks * 4
    } else {
        tile
    };
    2 * stage + if fp8 { tile } else { 0 } + 16 * latent * 2 + blocks * keys * 16 * 4 + 2 * keys * 4
}

/// Keys per attention unit: 32, or 16 when 32 does not fit shared memory.
fn unit_keys(latent: u32, fp8: bool) -> Option<u32> {
    [32, 16]
        .into_iter()
        .find(|&keys| attention_smem(latent, fp8, keys) <= SHARED_LIMIT)
}

/// Checks `query` against SM120's limits, which its contract leaves open:
/// latent widths up to 1024, pages of a power of two of tokens, attention
/// heads in 16s up to 128 whose tiles fit shared memory, and GEMM N in 64s
/// and K in 128s. This original engine refuses extensions and native
/// formats by name; their contracts do not imply a kernel implementation.
pub fn check_limits(query: Query) -> Result<(), Refusal> {
    if let Query::Latent(q) = query {
        if let LatentAttentionQuery::Attention {
            sink,
            tail_rope_dims,
            second_fp4_keys,
            ..
        } = q
        {
            if sink {
                return Err(Refusal(
                    "latent attention with an attention sink is declared but not supported",
                ));
            }
            if tail_rope_dims != 0 {
                return Err(Refusal(
                    "latent attention with tail RoPE and its inverse is declared but not supported",
                ));
            }
            if second_fp4_keys {
                return Err(Refusal(
                    "latent attention with a second FP4 key source is declared but not supported",
                ));
            }
        }
        let (LatentAttentionQuery::KvWrite { kv, .. } | LatentAttentionQuery::Attention { kv, .. }) =
            q;
        if !matches!(kv, LatentKvFormat::Bf16 | LatentKvFormat::Fp8E4M3Block128) {
            return Err(Refusal(
                "native latent storage requires a source-aware engine",
            ));
        }
    }
    match query {
        Query::Latent(
            LatentAttentionQuery::KvWrite {
                latent, page_size, ..
            }
            | LatentAttentionQuery::Attention {
                latent, page_size, ..
            },
        ) => {
            if latent > MAX_LATENT {
                return Err(Refusal("latent width exceeds 1024 on SM120"));
            }
            if !page_size.is_power_of_two() {
                return Err(Refusal(
                    "latent pages hold a power of two of tokens on SM120",
                ));
            }
        }
        Query::HeadGemm(query) => {
            if !query.n.is_multiple_of(GEMM_TILE_N) || !query.k.is_multiple_of(GEMM_CHUNK) {
                return Err(Refusal(
                    "head-batched GEMM needs N in 64s and K in 128s on SM120",
                ));
            }
        }
    }
    if let Query::Latent(LatentAttentionQuery::Attention {
        heads, latent, kv, ..
    }) = query
    {
        if heads > MAX_HEADS {
            return Err(Refusal("latent attention takes at most 128 heads on SM120"));
        }
        if !heads.is_multiple_of(HEAD_TILE) {
            return Err(Refusal("latent attention needs heads in 16s on SM120"));
        }
        if unit_keys(latent, kv == LatentKvFormat::Fp8E4M3Block128).is_none() {
            return Err(Refusal(
                "latent attention width exceeds SM120 shared memory",
            ));
        }
    }
    Ok(())
}

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_latent_{}_v1", self.name)
    }

    /// The split merge symbol of a decode or verify attention entry.
    pub fn merge_symbol(&self) -> Option<String> {
        self.attention()
            .filter(|shape| shape.phase != AttentionPhase::Prefill)
            .map(|_| format!("chop_latent_{}_merge_v1", self.name))
    }

    /// The entry's declared exactness (see the module docs), which its split
    /// merge shares: one-row bits up to its capacity and row independence,
    /// with any chunk split for the KV write, prefill attention and the
    /// head GEMMs. Nothing for an entry its kernel refuses.
    pub fn invariance(&self) -> Invariance {
        if self.contract().is_err() {
            return Invariance::NONE;
        }
        match self.attention() {
            Some(shape) if shape.phase != AttentionPhase::Prefill => {
                Invariance::one_row(self.capacity())
            }
            _ => Invariance::any_rows(self.capacity()),
        }
    }

    /// Each symbol the entry launches with its declaration: its own and its
    /// split merge's.
    pub fn declared_symbols(&self) -> Vec<(String, Invariance)> {
        let invariance = self.invariance();
        std::iter::once(self.symbol())
            .chain(self.merge_symbol())
            .map(|symbol| (symbol, invariance))
            .collect()
    }

    pub fn capacity(&self) -> u32 {
        match self.query {
            Query::Latent(
                LatentAttentionQuery::KvWrite { capacity, .. }
                | LatentAttentionQuery::Attention { capacity, .. },
            ) => capacity,
            Query::HeadGemm(query) => query.m_capacity,
        }
    }

    fn attention(&self) -> Option<AttentionShape> {
        match self.query {
            Query::Latent(LatentAttentionQuery::Attention {
                heads,
                latent,
                max_keys,
                page_size,
                kv,
                phase,
                output,
                ..
            }) => Some(AttentionShape {
                heads,
                latent,
                max_keys,
                page_size,
                fp8: kv == LatentKvFormat::Fp8E4M3Block128,
                partial: output == LatentAttentionOutput::Partial,
                phase,
            }),
            _ => None,
        }
    }

    /// Keys per attention unit (`Attention::keys`): 32, or 16 when 32 does
    /// not fit shared memory.
    pub fn unit_keys(&self) -> Option<u32> {
        let shape = self.attention()?;
        unit_keys(shape.latent, shape.fp8)
    }

    /// Checks the query against its contract and the engine's limits
    /// ([`check_limits`]).
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let (static_query, family) = match self.query {
            Query::Latent(query) => {
                query.validate()?;
                query.numerical_contract_id()?;
                (
                    StaticQuery::LatentAttention(query),
                    if self.attention().is_some() {
                        OpFamily::Attention
                    } else {
                        OpFamily::Row
                    },
                )
            }
            Query::HeadGemm(query) => {
                query.validate()?;
                query.numerical_contract_id()?;
                (StaticQuery::HeadBatchedGemm(query), OpFamily::Gemm)
            }
        };
        check_limits(self.query)?;
        let spec = InstantiationSpec {
            family,
            key: CompileKey {
                query: static_query,
                knobs: self.knobs(),
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol(),
        };
        spec.validate()?;
        Ok(spec)
    }

    fn knobs(&self) -> CompileKnobs {
        let (tile_m, tile_n, tile_k, warps, split) = match self.query {
            Query::Latent(LatentAttentionQuery::KvWrite { latent, .. }) => {
                (WRITE_ROWS as u16, latent as u16, 128, WRITE_ROWS as u8, 1)
            }
            Query::Latent(LatentAttentionQuery::Attention { latent, .. }) => (
                HEAD_TILE as u16,
                self.unit_keys().unwrap_or(16) as u16,
                128,
                (latent / 128).max(1) as u8,
                MAX_SPLITS as u8,
            ),
            Query::HeadGemm(_) => (
                GEMM_TILE_M as u16,
                GEMM_TILE_N as u16,
                GEMM_CHUNK as u16,
                4,
                1,
            ),
        };
        CompileKnobs {
            tile_m,
            tile_n,
            tile_k,
            stages: if matches!(
                self.query,
                Query::Latent(LatentAttentionQuery::KvWrite { .. })
            ) {
                1
            } else {
                2
            },
            swizzle: 0,
            warps,
            split_k_capacity: split,
        }
    }

    /// Threads per CTA of the entry's main kernel.
    pub fn threads(&self) -> u32 {
        match self.query {
            Query::Latent(LatentAttentionQuery::KvWrite { .. }) => WRITE_ROWS * 32,
            Query::Latent(LatentAttentionQuery::Attention { latent, .. }) => latent / 128 * 32,
            Query::HeadGemm(_) => 128,
        }
    }

    /// Dynamic shared bytes of the main kernel. Launchers opt in above 48
    /// KiB.
    pub fn dynamic_shared_bytes(&self) -> u32 {
        match (self.query, self.attention(), self.unit_keys()) {
            (_, Some(shape), Some(keys)) => attention_smem(shape.latent, shape.fp8, keys),
            (Query::HeadGemm(_), ..) => 2 * (GEMM_TILE_M + GEMM_TILE_N) * GEMM_CHUNK * 2,
            _ => 0,
        }
    }

    /// Attention splits per row: [`MAX_SPLITS`] for decode and verify at
    /// every bucket, one for prefill. Neither depends on the rows of a step,
    /// so neither does a row's sum.
    pub fn splits(&self) -> u32 {
        match self.attention() {
            Some(shape) if shape.phase != AttentionPhase::Prefill => MAX_SPLITS,
            _ => 1,
        }
    }

    /// CTAs per row and head tile for a prepared row bucket, each taking
    /// `splits / groups` consecutive splits: enough CTAs to fill the SMs
    /// twice, a power of two up to the splits. Only the grid follows the
    /// bucket.
    pub fn groups(&self, prepared_rows: u32) -> u32 {
        let splits = self.splits();
        let Some(shape) = self.attention().filter(|_| splits > 1) else {
            return 1;
        };
        let ctas = prepared_rows.max(1) * (shape.heads / HEAD_TILE).max(1);
        (2 * SMS).div_ceil(ctas).next_power_of_two().min(splits)
    }

    /// Workspace bytes for a prepared row bucket: every split's F32 partial,
    /// or none with one split.
    pub fn workspace_bytes(&self, prepared_rows: u32) -> u64 {
        let splits = self.splits();
        match self.attention() {
            Some(shape) if splits > 1 => {
                u64::from(prepared_rows)
                    * u64::from(splits)
                    * u64::from(shape.heads)
                    * u64::from(shape.latent + 2)
                    * 4
            }
            _ => 0,
        }
    }

    /// The main kernel's launch for `prepared_rows` rows (a graph bucket, at
    /// most the capacity). CTAs past the live row count return after the PDL
    /// wait, so one plan serves every live count and graph replay.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.capacity() {
            return Err(Refusal("latent rows exceed the prepared capacity"));
        }
        let grid = match self.query {
            Query::Latent(LatentAttentionQuery::KvWrite { .. }) => {
                [prepared_rows.div_ceil(WRITE_ROWS), 1, 1]
            }
            Query::Latent(LatentAttentionQuery::Attention { heads, .. }) => {
                [prepared_rows, heads / HEAD_TILE, self.groups(prepared_rows)]
            }
            Query::HeadGemm(query) => [
                prepared_rows.div_ceil(GEMM_TILE_M),
                query.n / GEMM_TILE_N,
                query.heads,
            ],
        };
        let plan = LaunchPlan {
            grid,
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            workspace_bytes: self.workspace_bytes(prepared_rows),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The split merge launch of a decode or verify attention entry.
    pub fn merge_plan(&self, prepared_rows: u32) -> Result<Option<LaunchPlan>, Refusal> {
        let Some(shape) = self.attention() else {
            return Ok(None);
        };
        if self.merge_symbol().is_none() || self.splits() == 1 {
            return Ok(None);
        }
        let plan = LaunchPlan {
            grid: [prepared_rows, shape.heads, 1],
            block: [MERGE_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: self.workspace_bytes(prepared_rows),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(Some(plan))
    }

    /// The outputs of an attention entry: BF16 `out`, or the F32 partial.
    fn outputs(partial: bool) -> Vec<EntryParam> {
        if partial {
            vec![
                pointer("out_max", "float"),
                pointer("out_den", "float"),
                pointer("out_num", "float"),
            ]
        } else {
            vec![pointer("out", "uint16_t")]
        }
    }

    /// The main kernel's launch ABI. A decode or verify attention kernel
    /// writes only split partials; its merge writes the outputs.
    pub fn params(&self) -> Vec<EntryParam> {
        match self.query {
            Query::Latent(LatentAttentionQuery::KvWrite { .. }) => vec![
                pointer("latent", "const uint16_t"),
                pointer("row_batch", "const int32_t"),
                pointer("positions", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("block_tables", "const int32_t"),
                pointer("pool", "uint8_t"),
                u32_param("max_pages"),
                u64_param("page_stride_bytes"),
            ],
            Query::Latent(LatentAttentionQuery::Attention { output, .. }) => {
                let mut params = vec![
                    pointer("q", "const uint16_t"),
                    pointer("pool", "const uint8_t"),
                    pointer("block_tables", "const int32_t"),
                    pointer("row_batch", "const int32_t"),
                    pointer("key_lists", "const int32_t"),
                    pointer("live_rows", "const uint32_t"),
                    pointer("workspace", "float"),
                ];
                if self.splits() == 1 {
                    params.extend(Self::outputs(output == LatentAttentionOutput::Partial));
                }
                params.extend([
                    u32_param("max_pages"),
                    u64_param("page_stride_bytes"),
                    u32_param("budget"),
                    u32_param("splits"),
                    u32_param("scale_bits"),
                ]);
                params
            }
            Query::HeadGemm(query) => vec![
                pointer("x", "const uint16_t"),
                pointer("w", "const uint16_t"),
                pointer("live_rows", "const uint32_t"),
                pointer(
                    "y",
                    if query.output == DType::F32 {
                        "float"
                    } else {
                        "uint16_t"
                    },
                ),
                u64_param("head_stride"),
            ],
        }
    }

    /// The merge ABI of a splitting attention entry.
    pub fn merge_params(&self) -> Vec<EntryParam> {
        let partial = self.attention().is_some_and(|shape| shape.partial);
        let mut params = vec![
            pointer("workspace", "const float"),
            pointer("live_rows", "const uint32_t"),
        ];
        params.extend(Self::outputs(partial));
        params.push(u32_param("splits"));
        params
    }

    fn attention_traits(&self) -> Option<String> {
        let shape = self.attention()?;
        Some(format!(
            "chop::latent::Attention<{},{},{},{},{},{},{}>",
            shape.heads,
            shape.latent,
            shape.page_size,
            shape.max_keys,
            shape.fp8,
            shape.partial,
            self.unit_keys()?,
        ))
    }

    fn output_arguments(partial: bool) -> &'static str {
        if partial {
            "nullptr,out_max,out_den,out_num"
        } else {
            "out,nullptr,nullptr,nullptr"
        }
    }

    fn body(&self) -> String {
        match self.query {
            Query::Latent(LatentAttentionQuery::KvWrite {
                latent,
                page_size,
                kv,
                ..
            }) => format!(
                "chop::latent::kv_write<chop::latent::KvWrite<{latent},{page_size},{}>>(latent,\
                 row_batch,positions,live_rows,block_tables,pool,max_pages,page_stride_bytes);",
                kv == LatentKvFormat::Fp8E4M3Block128
            ),
            Query::Latent(LatentAttentionQuery::Attention { .. }) if self.splits() > 1 => format!(
                "chop::latent::attention_fixed_splits<{}>(q,pool,block_tables,row_batch,\
                 key_lists,live_rows,workspace,max_pages,page_stride_bytes,budget,splits,\
                 scale_bits);",
                self.attention_traits().expect("an attention entry"),
            ),
            Query::Latent(LatentAttentionQuery::Attention { output, .. }) => format!(
                "chop::latent::attention<{}>(q,pool,block_tables,row_batch,key_lists,live_rows,\
                 workspace,{},max_pages,page_stride_bytes,budget,splits,scale_bits);",
                self.attention_traits().expect("an attention entry"),
                Self::output_arguments(output == LatentAttentionOutput::Partial)
            ),
            Query::HeadGemm(query) => format!(
                "chop::latent::head_gemm<chop::latent::HeadGemm<{},{},{},{},{}>>(x,w,live_rows,y,\
                 head_stride);",
                query.heads,
                query.n,
                query.k,
                query.layout == HeadWeightLayout::KByN,
                query.output == DType::F32
            ),
        }
    }

    /// The entry's kernels: the main one, and the split merge of a decode or
    /// verify attention entry.
    pub fn entry_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.contract()?;
        let mut specs = vec![EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            self.threads(),
            1,
        )];
        if let (Some(merge), Some(shape)) = (self.merge_symbol(), self.attention()) {
            let traits = self.attention_traits().expect("an attention entry");
            specs.push(EntrySpec::new(
                merge,
                self.merge_params(),
                format!(
                    "chop::latent::merge<{traits}>(workspace,live_rows,{},splits);",
                    Self::output_arguments(shape.partial)
                ),
                MERGE_THREADS,
                1,
            ));
        }
        Ok(specs)
    }
}

/// The latent attention entries of `profile` with geometry `g`, named
/// `{tag}_{kernel}`:
///
/// - `kv_write_{bf16,fp8}`, 8192 rows;
/// - `attn_{bf16,fp8}_{prefill,decode,verify}` and their `_partial`
///   context-parallel forms: prefill 8192 rows in one split, decode 256 rows
///   and verify 256 rows (32 items of up to 8 rows), each with a split merge;
/// - `absorb_key` (`W_UK^T`, KByN) and `absorb_value` (`W_UV`, NByK), BF16
///   output, 8192 rows.
pub fn entries(profile: Profile, g: Geometry) -> Vec<Entry> {
    let entry = |name: String, query| Entry {
        profile,
        name: format!("{}_{name}", profile.tag()),
        query,
    };
    let mut entries = Vec::new();
    for (dtype, kv) in [
        ("bf16", LatentKvFormat::Bf16),
        ("fp8", LatentKvFormat::Fp8E4M3Block128),
    ] {
        entries.push(entry(
            format!("kv_write_{dtype}"),
            Query::Latent(LatentAttentionQuery::KvWrite {
                latent: g.latent,
                page_size: g.page_size,
                kv,
                capacity: 8192,
            }),
        ));
        for (phase_name, phase, capacity) in [
            ("prefill", AttentionPhase::Prefill, 8192),
            ("decode", AttentionPhase::Decode, 256),
            ("verify", AttentionPhase::Verify, 256),
        ] {
            for (suffix, output) in [
                ("", LatentAttentionOutput::Normalized),
                ("_partial", LatentAttentionOutput::Partial),
            ] {
                entries.push(entry(
                    format!("attn_{dtype}_{phase_name}{suffix}"),
                    Query::Latent(LatentAttentionQuery::Attention {
                        heads: g.heads,
                        latent: g.latent,
                        max_keys: g.max_keys,
                        softmax_scale_bits: g.softmax_scale.to_bits(),
                        page_size: g.page_size,
                        kv,
                        phase,
                        output,
                        sink: false,
                        tail_rope_dims: 0,
                        second_fp4_keys: false,
                        capacity,
                    }),
                ));
            }
        }
    }
    for (name, n, k, layout) in [
        ("absorb_key", g.latent, g.nope, HeadWeightLayout::KByN),
        ("absorb_value", g.v_dim, g.latent, HeadWeightLayout::NByK),
    ] {
        entries.push(entry(
            name.to_owned(),
            Query::HeadGemm(HeadBatchedGemmQuery {
                heads: g.heads,
                n,
                k,
                layout,
                output: DType::Bf16,
                m_capacity: 8192,
            }),
        ));
    }
    entries
}

/// One cubin holding `entries` (one profile's), named
/// `latent_attention_{tag}_sm120`.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for entry in entries {
        if entry.profile != profile {
            return Err(Refusal("a latent entry belongs to another profile"));
        }
        specs.extend(entry.entry_specs()?);
    }
    Ok(CubinSpec {
        name: format!("latent_attention_{}_sm120", profile.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec![
            "#include \"latent/attention.cuh\"".into(),
            "#include \"latent/fixed_splits.cuh\"".into(),
            "#include \"latent/kv_write.cuh\"".into(),
            "#include \"latent/head_gemm.cuh\"".into(),
        ],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "latent-test",
        tag: "latent_test",
    });

    fn geometry() -> Geometry {
        Geometry {
            heads: 16,
            latent: 512,
            page_size: 64,
            max_keys: 2051,
            softmax_scale: 1.0 / 16.0,
            nope: 256,
            v_dim: 256,
        }
    }

    fn find(entries: &[Entry], suffix: &str) -> Entry {
        entries
            .iter()
            .find(|e| e.name.ends_with(suffix))
            .unwrap()
            .clone()
    }

    #[test]
    fn served_entries_plan_within_sm120_limits() {
        let entries = entries(PROFILE, geometry());
        assert_eq!(entries.len(), 2 * (1 + 3 * 2) + 2);
        for entry in &entries {
            let spec = entry.contract().unwrap();
            assert_eq!(spec.symbol, entry.symbol());
            for rows in [1, 2, 4, 6, 8, 16, 24, 32, entry.capacity()] {
                let rows = rows.min(entry.capacity());
                let plan = entry.plan(rows).unwrap();
                assert!(plan.dynamic_shared_bytes <= SHARED_LIMIT);
                assert_eq!(plan.block[0], entry.threads());
                if let Some(merge) = entry.merge_plan(rows).unwrap() {
                    assert_eq!(merge.grid, [rows, 16, 1]);
                    assert_eq!(merge.workspace_bytes, plan.workspace_bytes);
                }
            }
        }
        let decode = find(&entries, "attn_fp8_decode");
        assert_eq!(decode.unit_keys(), Some(32));
        assert_eq!(decode.threads(), 128);
        // 64 splits at every bucket; 2 x 188 CTAs: one split per CTA for
        // one row (512 capped at 64), four per CTA at 32 rows, all 64 in
        // one CTA from 376 rows.
        assert_eq!(decode.splits(), 64);
        assert_eq!(decode.groups(1), 64);
        assert_eq!(decode.groups(8), 64);
        assert_eq!(decode.groups(32), 16);
        assert_eq!(decode.groups(256), 2);
        assert_eq!(decode.plan(32).unwrap().grid, [32, 1, 16]);
        assert_eq!(
            decode.plan(32).unwrap().workspace_bytes,
            32 * 64 * 16 * 514 * 4
        );
        assert_eq!(
            decode.merge_plan(256).unwrap().unwrap().workspace_bytes,
            256 * 64 * 16 * 514 * 4
        );
        let prefill = find(&entries, "attn_bf16_prefill_partial");
        assert_eq!(prefill.splits(), 1);
        assert_eq!(prefill.groups(1), 1);
        assert_eq!(prefill.merge_symbol(), None);
        assert_eq!(prefill.plan(8192).unwrap().workspace_bytes, 0);
        let key = find(&entries, "absorb_key");
        assert_eq!(key.plan(1).unwrap().grid, [1, 8, 16]);
        assert_eq!(key.plan(8192).unwrap().grid, [128, 8, 16]);
        assert_eq!(
            find(&entries, "absorb_value").plan(5).unwrap().grid,
            [1, 4, 16]
        );
        assert_eq!(
            find(&entries, "kv_write_fp8").plan(9).unwrap().grid,
            [3, 1, 1]
        );
        assert!(decode.plan(257).is_err());
    }

    #[test]
    fn shared_memory_fits_at_the_served_width() {
        // BF16: two 32 KiB stages, 16 KiB of queries, 8 KiB of exchanged
        // chains and the valid flags.
        assert_eq!(
            attention_smem(512, false, 32),
            65_536 + 16_384 + 8_192 + 256
        );
        // FP8: two stages of codes and scales, then the widened tile.
        assert_eq!(
            attention_smem(512, true, 32),
            2 * (16_384 + 512) + 32_768 + 16_384 + 8_192 + 256
        );
        // Width 768 falls back to 16-key units; 1024 fits neither.
        let wide = |latent| {
            find(
                &entries(
                    PROFILE,
                    Geometry {
                        latent,
                        ..geometry()
                    },
                ),
                "attn_bf16_decode",
            )
        };
        assert_eq!(wide(768).unit_keys(), Some(16));
        assert_eq!(wide(768).plan(1).unwrap().dynamic_shared_bytes, 80_000);
        assert_eq!(
            wide(1024).contract().unwrap_err(),
            Refusal("latent attention width exceeds SM120 shared memory")
        );
    }

    #[test]
    fn declared_extensions_and_engine_limits_refuse_by_name() {
        let base = entries(PROFILE, geometry());
        let attention = find(&base, "attn_fp8_decode");
        let Query::Latent(query) = attention.query else {
            unreachable!()
        };
        let LatentAttentionQuery::Attention {
            heads,
            latent,
            max_keys,
            softmax_scale_bits,
            page_size,
            kv,
            phase,
            output,
            capacity,
            ..
        } = query
        else {
            unreachable!()
        };
        let with = |heads, sink, tail_rope_dims, second_fp4_keys| Entry {
            query: Query::Latent(LatentAttentionQuery::Attention {
                heads,
                latent,
                max_keys,
                softmax_scale_bits,
                page_size,
                kv,
                phase,
                output,
                sink,
                tail_rope_dims,
                second_fp4_keys,
                capacity,
            }),
            ..attention.clone()
        };
        for (entry, refusal) in [
            (
                with(heads, true, 0, false),
                "latent attention with an attention sink is declared but not supported",
            ),
            (
                with(heads, false, 64, false),
                "latent attention with tail RoPE and its inverse is declared but not supported",
            ),
            (
                with(heads, false, 0, true),
                "latent attention second source requires a native FP8 or BF16 window",
            ),
            (
                with(8, false, 0, false),
                "latent attention needs heads in 16s on SM120",
            ),
        ] {
            assert_eq!(entry.contract().unwrap_err(), Refusal(refusal));
            assert_eq!(entry.plan(1).unwrap_err(), Refusal(refusal));
            assert!(entry.entry_specs().is_err());
        }
        for kv in [
            LatentKvFormat::Fp8E4M3Ue8M0Block32,
            LatentKvFormat::Fp4E2M1E4M3Block16,
        ] {
            let mut native = attention.clone();
            if let Query::Latent(LatentAttentionQuery::Attention { kv: format, .. }) =
                &mut native.query
            {
                *format = kv;
            }
            assert_eq!(
                native.contract().unwrap_err().0,
                "native latent storage requires a source-aware engine"
            );
            assert!(native.entry_specs().is_err());
            assert_eq!(
                check_limits(Query::Latent(LatentAttentionQuery::KvWrite {
                    latent,
                    page_size,
                    kv,
                    capacity,
                }))
                .unwrap_err()
                .0,
                "native latent storage requires a source-aware engine"
            );
        }
        let key = find(&base, "absorb_key");
        let Query::HeadGemm(gemm) = key.query else {
            unreachable!()
        };
        // The contract allows any N and K; SM120's tiles and MMA do not.
        for (n, k) in [(520, 256), (512, 100)] {
            let narrow = Entry {
                query: Query::HeadGemm(HeadBatchedGemmQuery { n, k, ..gemm }),
                ..key.clone()
            };
            assert_eq!(
                narrow.contract().unwrap_err(),
                Refusal("head-batched GEMM needs N in 64s and K in 128s on SM120")
            );
        }
    }

    #[test]
    fn limits_the_contract_leaves_open_still_refuse() {
        let base = entries(PROFILE, geometry());
        let Query::Latent(attention) = find(&base, "attn_bf16_decode").query else {
            unreachable!()
        };
        let LatentAttentionQuery::Attention {
            max_keys,
            softmax_scale_bits,
            kv,
            phase,
            output,
            capacity,
            ..
        } = attention
        else {
            unreachable!()
        };
        let with = |heads, latent, page_size| {
            Query::Latent(LatentAttentionQuery::Attention {
                heads,
                latent,
                max_keys,
                softmax_scale_bits,
                page_size,
                kv,
                phase,
                output,
                sink: false,
                tail_rope_dims: 0,
                second_fp4_keys: false,
                capacity,
            })
        };
        let write = |latent, page_size| {
            Query::Latent(LatentAttentionQuery::KvWrite {
                latent,
                page_size,
                kv,
                capacity: 8192,
            })
        };
        for (query, refusal) in [
            (
                with(144, 512, 64),
                "latent attention takes at most 128 heads on SM120",
            ),
            (with(16, 2048, 64), "latent width exceeds 1024 on SM120"),
            (
                with(16, 512, 48),
                "latent pages hold a power of two of tokens on SM120",
            ),
            (write(2048, 64), "latent width exceeds 1024 on SM120"),
            (
                write(512, 48),
                "latent pages hold a power of two of tokens on SM120",
            ),
        ] {
            let Query::Latent(contract) = query else {
                unreachable!()
            };
            assert_eq!(contract.validate(), Ok(()), "{refusal}");
            let entry = Entry {
                query,
                ..find(&base, "attn_bf16_decode")
            };
            assert_eq!(check_limits(query), Err(Refusal(refusal)));
            assert_eq!(entry.contract().unwrap_err(), Refusal(refusal));
            assert_eq!(entry.plan(1).unwrap_err(), Refusal(refusal));
        }
    }

    #[test]
    fn entry_abis_follow_the_output_mode() {
        let entries = entries(PROFILE, geometry());
        let names = |params: Vec<EntryParam>| {
            params
                .into_iter()
                .map(|p| match p {
                    EntryParam::DevicePointer { name, .. }
                    | EntryParam::U32 { name }
                    | EntryParam::U64 { name } => name,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>()
        };
        let prefill = find(&entries, "attn_bf16_prefill");
        let normalized = find(&entries, "attn_bf16_verify");
        let partial = find(&entries, "attn_bf16_verify_partial");
        assert!(names(prefill.params()).contains(&"out".to_owned()));
        assert!(
            !names(find(&entries, "attn_bf16_prefill_partial").params())
                .contains(&"out".to_owned())
        );
        // Decode and verify kernels write only split partials.
        for entry in [&normalized, &partial] {
            let params = names(entry.params());
            assert!(params.contains(&"workspace".to_owned()));
            assert!(!params.iter().any(|name| name.starts_with("out")));
        }
        assert!(names(normalized.merge_params()).contains(&"out".to_owned()));
        assert!(names(partial.merge_params()).contains(&"out_num".to_owned()));
        let specs = partial.entry_specs().unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(
            specs[1].symbol,
            "chop_latent_latent_test_attn_bf16_verify_partial_merge_v1"
        );
        assert!(specs[0].body.contains(
            "chop::latent::attention_fixed_splits<chop::latent::Attention<16,512,64,2051,false,\
             true,32>>"
        ));
        assert!(
            prefill.entry_specs().unwrap()[0]
                .body
                .contains("chop::latent::attention<chop::latent::Attention<16,512,64,2051,false,")
        );
        let spec = cubin_spec(PROFILE, &entries).unwrap();
        assert_eq!(spec.name, "latent_attention_latent_test_sm120");
        // Twelve attention entries, eight merges, two writes, two GEMMs.
        assert_eq!(spec.entries.len(), 24);
    }
}

pub mod window_stage;

pub mod staged_window_keys;

pub mod window_fused;

pub mod sources_decode;
