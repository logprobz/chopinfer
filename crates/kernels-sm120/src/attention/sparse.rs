//! QSA block-sparse attention with a learned indexer: the
//! indexer query, the raw-key ring and block-key cache write, block scoring,
//! top-k block selection and the paged sparse attention with its split merge
//! (`cuda/attention/sparse*.cuh`, `cuda/attention/SPARSE.md`).

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::arithmetic::DeclaredCubin;
use crate::invariance::{Anchors, Batch, Chunk, Invariance, Rows, Ties};
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    AttentionPhase, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, SparseAttentionQuery, StaticQuery, Target, Validate,
};

const SMS: u32 = 188;
/// Blocks one fixed-grid scoring CTA covers; `Scores::chunk_blocks` in sparse_indexer.cuh.
fn score_chunk_blocks(rows: u32) -> u32 {
    if rows == 1 { 64 } else { 1024 }
}
/// Blocks the top-k bitmasks cover; `TopK::max_blocks`.
const TOPK_MAX_BLOCKS: u32 = 65_536;
/// The widest split of one attention row.
const MAX_SPLITS: u32 = 64;
/// Threads of a split merge CTA; `merge_warps` in sparse.cuh.
const MERGE_THREADS: u32 = 256;

/// Static QSA geometry of one model profile, shared by its entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub index_heads: u32,
    pub index_dim: u32,
    pub rotary_dim: u32,
    pub theta: f32,
    pub mrope: [u32; 3],
    pub eps: f32,
    pub block: u32,
    /// Raw-key ring rows per sequence slot: a block plus a verify step.
    pub ring: u32,
    pub top_blocks: u32,
    pub max_context: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub page_size: u32,
    pub softmax_scale: f32,
}

impl Geometry {
    pub fn max_blocks(&self) -> u32 {
        self.max_context / self.block
    }

    /// Block-key cache bytes of one page: `page_size / block` BF16 rows.
    pub fn block_page_bytes(&self) -> u64 {
        u64::from(self.page_size / self.block) * u64::from(self.index_dim) * 2
    }

    /// Scores row stride, in floats, for rows that see at most `context`
    /// tokens.
    pub fn score_stride(&self, context: u32) -> u32 {
        (context / self.block).max(1).next_multiple_of(64)
    }

    fn indexer_traits(&self) -> String {
        format!(
            "chop::attention::sparse::Indexer<{},{},{},{},{},{},{},{},{}>",
            self.index_heads,
            self.index_dim,
            self.rotary_dim,
            self.mrope[1],
            self.mrope[2],
            self.block,
            self.ring,
            self.page_size,
            self.top_blocks,
        )
    }

    fn attention_traits(&self, fp8: bool, keys: u32) -> String {
        format!(
            "chop::attention::sparse::Attention<{},{},{},{},{},{},{fp8},{keys}>",
            self.q_heads, self.kv_heads, self.head_dim, self.page_size, self.block, self.top_blocks,
        )
    }

    fn validate(&self) -> Result<(), Refusal> {
        let group = self.q_heads.checked_div(self.kv_heads).unwrap_or(0);
        if ![4, 8, 16].contains(&self.index_heads)
            || self.index_dim != 128
            || ![32, 64].contains(&self.rotary_dim)
            || self.mrope.iter().sum::<u32>() != self.rotary_dim / 2
            || self.block < 2
            || self.ring < self.block
            || !self.ring.is_multiple_of(self.block)
            || !self.page_size.is_multiple_of(self.block)
            || ![16, 32, 64].contains(&self.page_size)
            || ![128, 256].contains(&self.head_dim)
            || self.kv_heads == 0
            || !self.q_heads.is_multiple_of(self.kv_heads)
            || group > 16
            || !self.max_context.is_multiple_of(self.block)
            || self.max_blocks() > TOPK_MAX_BLOCKS
        {
            return Err(Refusal("QSA geometry is outside the SM120 engine"));
        }
        Ok(())
    }
}

/// Refuses QSA queries the SM120 engine does not instantiate: indexer
/// heads of other than 128 dimensions or rotary widths other than 32 and
/// 64, other than 4, 8 or 16 heads sharing a score, attention heads of
/// other than 128 or 256 dimensions or more than 16 per KV head, and pages
/// other than 16, 32 or 64 tokens. `Geometry` holds the rest of the
/// engine's limits.
pub fn check_query(query: &SparseAttentionQuery) -> Result<(), Refusal> {
    let supported = match *query {
        SparseAttentionQuery::IndexerQueryGemmaNormMrope {
            heads,
            head_dim,
            rotary_dim,
            ..
        } => [4, 8, 16].contains(&heads) && head_dim == 128 && [32, 64].contains(&rotary_dim),
        SparseAttentionQuery::IndexerKeyWrite {
            head_dim,
            rotary_dim,
            ..
        } => head_dim == 128 && [32, 64].contains(&rotary_dim),
        SparseAttentionQuery::BlockScores {
            heads, head_dim, ..
        } => [4, 8, 16].contains(&heads) && head_dim == 128,
        SparseAttentionQuery::BlockTopK { max_blocks, .. } => max_blocks <= TOPK_MAX_BLOCKS,
        SparseAttentionQuery::SparseAttention {
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            ..
        } => {
            [128, 256].contains(&head_dim)
                && q_heads
                    .checked_div(kv_heads)
                    .is_some_and(|group| group <= 16)
                && [16, 32, 64].contains(&page_size)
        }
    };
    if !supported {
        return Err(Refusal("QSA query is outside the SM120 engine"));
    }
    Ok(())
}

/// Which kernel an entry instantiates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Indexer query norm and M-RoPE.
    Query,
    /// Raw-key ring and block-key cache write.
    KeyWrite,
    /// Block scores; a CTA takes `rows` consecutive rows, or with `by_run`
    /// the rows of one sequence (a verify item) in groups of `rows`.
    Scores { rows: u32, by_run: bool },
    /// Top-k block selection.
    TopK,
    /// Sparse attention; its entry also has a split merge.
    Attention,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub name: String,
    pub geometry: Geometry,
    pub kind: Kind,
    pub query: SparseAttentionQuery,
    /// Prefill entries keep a row's bits over prompt splits at whole
    /// blocks: a block's key is written once its rows are all present, and
    /// rows see only completed blocks plus their own raw keys. The query
    /// norm and the top-k are row-local; the key write also leaves the
    /// raw-key ring at anchor rows. The entries of a model's exact verify
    /// steps (`verify_rows`) also keep one-row bits over their rows.
    pub invariance: Invariance,
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

/// The packed step rows every entry takes after its data pointers.
fn step_rows() -> [EntryParam; 3] {
    [
        pointer("row_batch", "const int32_t"),
        pointer("positions", "const int32_t"),
        pointer("live_rows", "const uint32_t"),
    ]
}

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_qsa_{}_v1", self.name)
    }

    /// An attention entry's declared arithmetic: its query, geometry and
    /// splits from each prepared row count on, through its capacity.
    fn split_arithmetic(&self) -> String {
        let mut splits: Vec<(u32, u32)> = Vec::new();
        for rows in 1..=self.capacity() {
            let count = self.splits(rows);
            if splits.last().is_none_or(|&(_, last)| last != count) {
                splits.push((rows, count));
            }
        }
        format!(
            "sparse-attention.v1; query={:?}; geometry={:?}; splits_from_rows={splits:?}",
            self.query, self.geometry
        )
    }

    /// The split merge symbol of an attention entry.
    pub fn merge_symbol(&self) -> Option<String> {
        (self.kind == Kind::Attention).then(|| format!("chop_qsa_{}_merge_v1", self.name))
    }

    pub fn capacity(&self) -> u32 {
        match self.query {
            SparseAttentionQuery::IndexerQueryGemmaNormMrope { capacity, .. }
            | SparseAttentionQuery::IndexerKeyWrite { capacity, .. }
            | SparseAttentionQuery::BlockScores { capacity, .. }
            | SparseAttentionQuery::BlockTopK { capacity, .. }
            | SparseAttentionQuery::SparseAttention { capacity, .. } => capacity,
        }
    }

    fn prefill(&self) -> bool {
        matches!(
            self.query,
            SparseAttentionQuery::SparseAttention {
                phase: AttentionPhase::Prefill,
                ..
            }
        )
    }

    /// Keys per attention unit (`Attention::keys`): prefill takes 16-key
    /// units so two CTAs share an SM; decode and verify take 32.
    pub fn unit_keys(&self) -> u32 {
        if self.prefill() { 16 } else { 32 }
    }

    fn fp8(&self) -> bool {
        matches!(
            self.query,
            SparseAttentionQuery::SparseAttention {
                kv_dtype: DType::Fp8E4M3,
                ..
            }
        )
    }

    /// Checks the query against the entry's kind and geometry.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.query.validate()?;
        check_query(&self.query)?;
        self.geometry.validate()?;
        let g = self.geometry;
        let consistent = match (self.kind, self.query) {
            (
                Kind::Query,
                SparseAttentionQuery::IndexerQueryGemmaNormMrope {
                    heads,
                    head_dim,
                    rotary_dim,
                    theta_bits,
                    mrope,
                    eps_bits,
                    ..
                },
            ) => {
                heads == g.index_heads
                    && head_dim == g.index_dim
                    && rotary_dim == g.rotary_dim
                    && theta_bits == g.theta.to_bits()
                    && mrope == g.mrope
                    && eps_bits == g.eps.to_bits()
            }
            (
                Kind::KeyWrite,
                SparseAttentionQuery::IndexerKeyWrite {
                    head_dim,
                    rotary_dim,
                    theta_bits,
                    mrope,
                    eps_bits,
                    block,
                    ring,
                    ..
                },
            ) => {
                head_dim == g.index_dim
                    && rotary_dim == g.rotary_dim
                    && theta_bits == g.theta.to_bits()
                    && mrope == g.mrope
                    && eps_bits == g.eps.to_bits()
                    && block == g.block
                    && ring == g.ring
            }
            (
                Kind::Scores { rows, by_run },
                SparseAttentionQuery::BlockScores {
                    heads,
                    head_dim,
                    max_blocks,
                    capacity,
                },
            ) => {
                [1, 4, 8, 16, 32, 64].contains(&rows)
                    && rows <= 256 / g.index_heads
                    && (!by_run || capacity <= 256)
                    && heads == g.index_heads
                    && head_dim == g.index_dim
                    && max_blocks == g.max_blocks()
            }
            (
                Kind::TopK,
                SparseAttentionQuery::BlockTopK {
                    block,
                    top_blocks,
                    max_blocks,
                    ..
                },
            ) => block == g.block && top_blocks == g.top_blocks && max_blocks == g.max_blocks(),
            (
                Kind::Attention,
                SparseAttentionQuery::SparseAttention {
                    q_heads,
                    kv_heads,
                    head_dim,
                    block,
                    top_blocks,
                    softmax_scale_bits,
                    page_size,
                    ..
                },
            ) => {
                q_heads == g.q_heads
                    && kv_heads == g.kv_heads
                    && head_dim == g.head_dim
                    && block == g.block
                    && top_blocks == g.top_blocks
                    && softmax_scale_bits == g.softmax_scale.to_bits()
                    && page_size == g.page_size
            }
            _ => false,
        };
        if !consistent {
            return Err(Refusal("QSA entry query differs from its kind or geometry"));
        }
        let spec = InstantiationSpec {
            family: if self.kind == Kind::Attention {
                OpFamily::Attention
            } else {
                OpFamily::Row
            },
            key: CompileKey {
                query: StaticQuery::SparseAttention(self.query),
                knobs: self.knobs(),
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol(),
        };
        spec.validate()?;
        self.query.numerical_contract_id()?;
        Ok(spec)
    }

    fn knobs(&self) -> CompileKnobs {
        let (tile_m, tile_n, warps) = match self.kind {
            Kind::Query => (
                1,
                self.geometry.index_dim as u16,
                self.geometry.index_heads as u8,
            ),
            Kind::KeyWrite => (4, 128, 4),
            Kind::Scores { rows, .. } => (
                (rows * self.geometry.index_heads).next_multiple_of(16) as u16,
                64,
                8,
            ),
            Kind::TopK => (1, 512, 16),
            Kind::Attention => (16, self.unit_keys() as u16, 4),
        };
        CompileKnobs {
            tile_m,
            tile_n,
            tile_k: 16,
            stages: if matches!(self.kind, Kind::Scores { .. }) {
                3
            } else {
                2
            },
            swizzle: 0,
            warps,
            split_k_capacity: if self.kind == Kind::Attention {
                MAX_SPLITS as u8
            } else {
                1
            },
        }
    }

    /// Threads per CTA of the entry's main kernel.
    pub fn threads(&self) -> u32 {
        match self.kind {
            Kind::Query => 32 * self.geometry.index_heads,
            Kind::KeyWrite | Kind::Attention => 128,
            Kind::Scores { .. } => 256,
            Kind::TopK => 512,
        }
    }

    /// Dynamic shared bytes of the main kernel: `Scores::smem`, `TopK::smem`
    /// and `Attention::smem`. Launchers opt in above 48 KiB.
    pub fn dynamic_shared_bytes(&self) -> u32 {
        let g = self.geometry;
        match self.kind {
            Kind::Scores { rows, .. } => {
                let q_bytes = (rows * g.index_heads).div_ceil(16) * 16 * g.index_dim * 2;
                let stages = if q_bytes > 32_768 || rows == 1 { 2 } else { 3 };
                stages * 64 * g.index_dim * 2 + q_bytes
            }
            // The digit histogram, the kept and tied bitmasks and the
            // gathered bin.
            Kind::TopK => (2048 + 2 * TOPK_MAX_BLOCKS / 32 + 2 * 8192) * 4,
            Kind::Attention => {
                let unit = self.unit_keys() * g.head_dim * 2;
                let padded = (g.q_heads / g.kv_heads).div_ceil(8) * 8;
                4 * unit + padded * g.head_dim * 2 + g.top_blocks * 4
            }
            _ => 0,
        }
    }

    /// Attention splits per row for a prepared row bucket: enough CTAs to
    /// fill the SMs twice, one split (direct output) past that. Prefill
    /// always takes one split, so a prompt row's sum and output formula do
    /// not depend on the rows of its step.
    pub fn splits(&self, prepared_rows: u32) -> u32 {
        if self.kind != Kind::Attention || self.prefill() {
            return 1;
        }
        let rows = prepared_rows.max(1) * self.geometry.kv_heads;
        let target = (2 * SMS).div_ceil(rows);
        if target <= 1 {
            1
        } else {
            target.next_power_of_two().min(MAX_SPLITS)
        }
    }

    /// Attention workspace bytes for a prepared row bucket: the F32 split
    /// partials, or none with direct output.
    pub fn workspace_bytes(&self, prepared_rows: u32) -> u64 {
        let splits = self.splits(prepared_rows);
        if self.kind != Kind::Attention || splits == 1 {
            return 0;
        }
        let g = self.geometry;
        u64::from(prepared_rows)
            * u64::from(g.q_heads)
            * u64::from(splits)
            * u64::from(g.head_dim + 2)
            * 4
    }

    /// The main kernel's launch for `prepared_rows` rows (a graph bucket,
    /// at most the capacity) that see at most `context` tokens. CTAs past
    /// the live row count return after the PDL wait, so one plan serves
    /// every live count and graph replay.
    pub fn plan(&self, prepared_rows: u32, context: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        self.plan_checked(prepared_rows, context)
    }

    /// [`Entry::plan`] of an entry whose contract was checked.
    pub(crate) fn plan_checked(
        &self,
        prepared_rows: u32,
        context: u32,
    ) -> Result<LaunchPlan, Refusal> {
        let g = self.geometry;
        if prepared_rows == 0
            || prepared_rows > self.capacity()
            || context == 0
            || context > g.max_context
        {
            return Err(Refusal("QSA rows or context exceed the prepared capacity"));
        }
        let grid = match self.kind {
            Kind::Query | Kind::TopK => [prepared_rows, 1, 1],
            Kind::KeyWrite => [prepared_rows.div_ceil(4), 1, 1],
            Kind::Scores { rows, by_run } => {
                let row_tiles = if by_run {
                    prepared_rows
                } else {
                    prepared_rows.div_ceil(rows)
                };
                let chunks = (context / g.block)
                    .max(1)
                    .div_ceil(score_chunk_blocks(rows));
                if rows == 1 && !by_run {
                    [(row_tiles * chunks).min(2 * SMS), 1, 1]
                } else {
                    [row_tiles, chunks, 1]
                }
            }
            Kind::Attention => [prepared_rows, g.kv_heads, self.splits(prepared_rows)],
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

    /// The split merge launch of an attention entry, when its bucket splits.
    pub fn merge_plan(&self, prepared_rows: u32) -> Result<Option<LaunchPlan>, Refusal> {
        if self.kind != Kind::Attention || self.splits(prepared_rows) == 1 {
            return Ok(None);
        }
        let plan = LaunchPlan {
            grid: [prepared_rows, self.geometry.q_heads, 1],
            block: [MERGE_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: self.workspace_bytes(prepared_rows),
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(Some(plan))
    }

    /// The main kernel's launch ABI.
    pub fn params(&self) -> Vec<EntryParam> {
        let mut params = match self.kind {
            Kind::Query => vec![
                pointer("projection", "const uint16_t"),
                pointer("weight", "const uint16_t"),
                pointer("rope_positions", "const int32_t"),
                pointer("cos_sin", "const float"),
                pointer("live_rows", "const uint32_t"),
                pointer("q", "uint16_t"),
                u32_param("eps_bits"),
            ],
            Kind::KeyWrite => {
                let mut params = vec![
                    pointer("projection", "const uint16_t"),
                    pointer("weight", "const uint16_t"),
                    pointer("rope_positions", "const int32_t"),
                    pointer("cos_sin", "const float"),
                ];
                params.extend(step_rows());
                params.extend([
                    pointer("slot_ids", "const int32_t"),
                    pointer("ring_keys", "uint16_t"),
                    pointer("ring_positions", "int32_t"),
                    pointer("anchor_slots", "const int32_t"),
                    pointer("anchor_ends", "const int32_t"),
                    pointer("anchor_keys", "uint16_t"),
                    pointer("anchor_positions", "int32_t"),
                    pointer("block_tables", "const int32_t"),
                    pointer("block_keys", "uint8_t"),
                    u32_param("max_pages"),
                    u64_param("block_page_stride_bytes"),
                    u32_param("eps_bits"),
                ]);
                params
            }
            Kind::Scores { .. } => {
                let mut params = vec![
                    pointer("q", "const uint16_t"),
                    pointer("block_keys", "const uint8_t"),
                    pointer("block_tables", "const int32_t"),
                ];
                params.extend(step_rows());
                params.extend([
                    pointer("scores", "float"),
                    u32_param("max_pages"),
                    u64_param("block_page_stride_bytes"),
                    u32_param("score_stride"),
                ]);
                params
            }
            Kind::TopK => vec![
                pointer("scores", "const float"),
                pointer("positions", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("selected", "int32_t"),
                u32_param("score_stride"),
            ],
            Kind::Attention => {
                let mut params = vec![
                    pointer("q", "const uint16_t"),
                    pointer("k_cache", "const void"),
                    pointer("v_cache", "const void"),
                    pointer("block_tables", "const int32_t"),
                ];
                params.extend(step_rows());
                params.extend([
                    pointer("selected", "const int32_t"),
                    pointer("workspace", "float"),
                    pointer("out", "uint16_t"),
                    u32_param("max_pages"),
                    u64_param("k_page_stride_bytes"),
                    u64_param("v_page_stride_bytes"),
                    u32_param("splits"),
                    u32_param("scale_bits"),
                    u32_param("k_scale_bits"),
                    u32_param("v_scale_bits"),
                ]);
                params
            }
        };
        params.shrink_to_fit();
        params
    }

    /// The merge ABI of an attention entry.
    pub fn merge_params() -> Vec<EntryParam> {
        vec![
            pointer("workspace", "const float"),
            pointer("live_rows", "const uint32_t"),
            pointer("out", "uint16_t"),
            u32_param("splits"),
        ]
    }

    fn body(&self) -> String {
        let indexer = self.geometry.indexer_traits();
        match self.kind {
            Kind::Query => format!(
                "chop::attention::sparse::indexer_query<{indexer}>(projection,weight,\
                 rope_positions,cos_sin,live_rows,q,eps_bits);"
            ),
            Kind::KeyWrite => format!(
                "chop::attention::sparse::indexer_key_write<{indexer}>(projection,weight,\
                 rope_positions,cos_sin,row_batch,positions,live_rows,slot_ids,ring_keys,\
                 ring_positions,anchor_slots,anchor_ends,anchor_keys,anchor_positions,\
                 block_tables,block_keys,max_pages,block_page_stride_bytes,eps_bits);"
            ),
            Kind::Scores { rows, by_run } => format!(
                "chop::attention::sparse::block_scores<{indexer},{rows},{by_run},{}>(q,block_keys,\
                 block_tables,row_batch,positions,live_rows,scores,max_pages,\
                 block_page_stride_bytes,score_stride);",
                rows == 1 && !by_run
            ),
            Kind::TopK => format!(
                "chop::attention::sparse::block_topk<{indexer}>(scores,positions,live_rows,\
                 selected,score_stride);"
            ),
            Kind::Attention => format!(
                "chop::attention::sparse::attention<{}>(q,k_cache,v_cache,block_tables,\
                 row_batch,positions,live_rows,selected,workspace,out,max_pages,\
                 k_page_stride_bytes,v_page_stride_bytes,splits,scale_bits,k_scale_bits,\
                 v_scale_bits);",
                self.geometry.attention_traits(self.fp8(), self.unit_keys())
            ),
        }
    }

    /// The entry's kernels: the main one, and the merge of an attention entry.
    /// Each cubin entry of this entry, with its declaration. Prefill takes
    /// one split, so its merge never runs and claims nothing; decode and
    /// verify merges declare as their attention entries.
    pub fn declared_symbols(&self) -> Vec<(String, Invariance)> {
        let prefill = matches!(
            self.query,
            SparseAttentionQuery::SparseAttention {
                phase: AttentionPhase::Prefill,
                ..
            }
        );
        let merge = self.merge_symbol().map(|symbol| {
            let invariance = if prefill {
                Invariance::NONE
            } else {
                self.invariance
            };
            (symbol, invariance)
        });
        std::iter::once((self.symbol(), self.invariance))
            .chain(merge)
            .collect()
    }

    pub fn entry_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.contract()?;
        let mut specs = vec![EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            self.threads(),
            1,
        )];
        if let Some(merge) = self.merge_symbol() {
            specs.push(EntrySpec::new(
                merge,
                Self::merge_params(),
                format!(
                    "chop::attention::sparse::merge<{}>(workspace,live_rows,out,splits);",
                    self.geometry.attention_traits(self.fp8(), self.unit_keys())
                ),
                MERGE_THREADS,
                1,
            ));
        }
        Ok(specs)
    }
}

/// A prefill entry whose rows keep their bits over splits at whole blocks.
fn blocks(g: Geometry) -> Invariance {
    Invariance {
        chunk: Chunk::Granule(g.block),
        batch: Batch::Independent,
        ..Invariance::NONE
    }
}

/// The key write's claims: block keys pooled in position order keep a
/// row's bits over splits at whole blocks, and the spare anchor ring is
/// copied from the launch's rows alone (`sparse_indexer.cuh`, positions
/// `[end - ring, end)`), so it is complete only for an anchor at least a
/// ring after the launch's first row.
fn key_write(g: Geometry) -> Invariance {
    blocks(g).with_anchors(Anchors::SpareAfter { rows: g.ring })
}

/// `invariance` with one-row bits for the rows of a model's exact verify
/// steps of up to `rows` rows (one request's drafter steps): the rows of
/// one verify item, launched together, give each row the bits of a one-row
/// decode step at its position after the rows before it wrote their keys
/// (the AR sequence). The step runs the index query, the key write, the
/// decode block scores, the top-k and the decode attention with its merge,
/// at a bucket of the same attention split count as AR's.
fn verify_rows(invariance: Invariance, rows: u32) -> Invariance {
    Invariance {
        rows: Rows::OneRow { max: rows },
        ..invariance
    }
}

/// Every model's QSA entries in `catalogs`, in order.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.sparse)())
        .collect()
}

/// The replicated indexer's entries of geometry `g`, named
/// `{tag}_{kernel}` after `profile`, whose rows of exact verify steps of up
/// to `exact_rows` rows keep one-row bits. They do not depend on the rank
/// count. Prefill rows score and select in chunks of at most 2048 rows (512
/// MiB of F32 scores at 262,144 tokens).
pub fn indexer_entries(profile: Profile, g: Geometry, exact_rows: u32) -> Vec<Entry> {
    let theta_bits = g.theta.to_bits();
    let eps_bits = g.eps.to_bits();
    let entry = |name: &str, kind, query| Entry {
        profile,
        name: format!("{}_{name}", profile.tag()),
        geometry: g,
        kind,
        query,
        // A verify item's rows score each other's block keys, so the decode
        // scores claim one-row bits, not row independence.
        invariance: match (name, kind) {
            (_, Kind::Query) => verify_rows(Invariance::PREFILL, exact_rows),
            (_, Kind::KeyWrite) => verify_rows(key_write(g), exact_rows),
            (_, Kind::TopK) => {
                verify_rows(Invariance::PREFILL.with_ties(Ties::LowerIndex), exact_rows)
            }
            ("block_scores_prefill", _) => blocks(g),
            ("block_scores_decode", _) => verify_rows(Invariance::NONE, exact_rows),
            _ => Invariance::NONE,
        },
    };
    let mut entries = vec![
        entry(
            "index_query",
            Kind::Query,
            SparseAttentionQuery::IndexerQueryGemmaNormMrope {
                heads: g.index_heads,
                head_dim: g.index_dim,
                rotary_dim: g.rotary_dim,
                theta_bits,
                mrope: g.mrope,
                eps_bits,
                capacity: 8192,
            },
        ),
        entry(
            "index_key_write",
            Kind::KeyWrite,
            SparseAttentionQuery::IndexerKeyWrite {
                head_dim: g.index_dim,
                rotary_dim: g.rotary_dim,
                theta_bits,
                mrope: g.mrope,
                eps_bits,
                block: g.block,
                ring: g.ring,
                capacity: 8192,
            },
        ),
        entry(
            "block_topk",
            Kind::TopK,
            SparseAttentionQuery::BlockTopK {
                block: g.block,
                top_blocks: g.top_blocks,
                max_blocks: g.max_blocks(),
                capacity: 2048,
            },
        ),
    ];
    // Decode scores one row per CTA column up to the package maximum of 256
    // items (CONC-QWEN); verify's by-run scan holds 256 rows in one CTA.
    for (name, rows, by_run, capacity) in [
        ("block_scores_decode", 1, false, 256),
        ("block_scores_verify", 8, true, 256),
        (
            "block_scores_prefill",
            64.min(256 / g.index_heads.max(1)),
            false,
            2048,
        ),
    ] {
        entries.push(entry(
            name,
            Kind::Scores { rows, by_run },
            SparseAttentionQuery::BlockScores {
                heads: g.index_heads,
                head_dim: g.index_dim,
                max_blocks: g.max_blocks(),
                capacity,
            },
        ));
    }
    entries
}

/// The sparse attention entries of `profile` with geometry `g`, named
/// `{tag}_attn_{dtype}_{phase}`, whose decode rows of exact verify steps of
/// up to `exact_rows` rows keep one-row bits: prefill takes 8192 rows,
/// decode 256 (one CTA row per decode row; 32 and fewer plan as before) and
/// verify 32 items of 8 rows.
pub fn attention_entries(profile: Profile, g: Geometry, exact_rows: u32) -> Vec<Entry> {
    let mut entries = Vec::new();
    for (dtype, kv_dtype) in [("bf16", DType::Bf16), ("fp8", DType::Fp8E4M3)] {
        for (phase_name, phase, capacity) in [
            ("prefill", AttentionPhase::Prefill, 8192),
            ("decode", AttentionPhase::Decode, 256),
            ("verify", AttentionPhase::Verify, 256),
        ] {
            entries.push(Entry {
                profile,
                name: format!("{}_attn_{dtype}_{phase_name}", profile.tag()),
                geometry: g,
                kind: Kind::Attention,
                // Exact verify rows run the decode entry, whose rows attend
                // to each other's keys: one-row bits, not row independence.
                // Its merge runs there too (64 splits from one to eleven
                // rows) and declares the same.
                invariance: match phase {
                    AttentionPhase::Prefill => blocks(g),
                    AttentionPhase::Decode => verify_rows(Invariance::NONE, exact_rows),
                    AttentionPhase::Verify => Invariance::NONE,
                },
                query: SparseAttentionQuery::SparseAttention {
                    q_heads: g.q_heads,
                    kv_heads: g.kv_heads,
                    head_dim: g.head_dim,
                    block: g.block,
                    top_blocks: g.top_blocks,
                    softmax_scale_bits: g.softmax_scale.to_bits(),
                    page_size: g.page_size,
                    kv_dtype,
                    phase,
                    capacity,
                },
            });
        }
    }
    entries
}

/// One cubin per profile of `catalogs` that has QSA entries; `None` builds
/// them all.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    declared_cubin_specs(catalogs, profile)
        .into_iter()
        .map(DeclaredCubin::into_spec)
        .collect()
}

/// [`cubin_specs`] with the declared arithmetic of each attention entry and
/// its merge: the splits it takes at launch for each prepared row count
/// (`Entry::splits`). The indexer entries compile theirs in.
pub fn declared_cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<DeclaredCubin> {
    let mut declared = BTreeMap::new();
    let all = catalog(catalogs);
    let specs = catalogs
        .profiles()
        .into_iter()
        .filter(|&owner| profile.is_none_or(|selected| selected == owner))
        .filter_map(|owner| {
            let entries = all
                .iter()
                .filter(|entry| entry.profile == owner)
                .map(|entry| {
                    if let Some(merge) = entry.merge_symbol() {
                        let arithmetic = entry.split_arithmetic();
                        declared.insert(merge, format!("merge; {arithmetic}"));
                        declared.insert(entry.symbol(), arithmetic);
                    }
                    entry.entry_specs().expect("catalog QSA entries are valid")
                })
                .collect::<Vec<_>>();
            (!entries.is_empty()).then(|| CubinSpec {
                name: format!("sparse_attention_{}_sm120", owner.tag()),
                sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
                include_dirs: vec![PathBuf::from(".")],
                headers: vec!["#include \"attention/sparse.cuh\"".into()],
                defines: vec![],
                entries: entries.into_iter().flatten().collect(),
            })
        })
        .collect();
    DeclaredCubin::all(specs, declared).expect("QSA attention entries declare their splits")
}
