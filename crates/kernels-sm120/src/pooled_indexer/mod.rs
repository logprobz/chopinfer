//! Pooled top-k indexer engine: the key LayerNorm and gated group pooling
//! write with its per-sequence tail state, streaming score and top-k
//! selection (global or per rank), the exact merge of per-rank selections
//! and the selection compaction (`cuda/pooled_indexer/*.cuh`,
//! `cuda/pooled_indexer/README.md`). Planners never allocate; live row
//! counts stay on the device and grids are sized for the prepared rows.
//!
//! Declared exactness ([`Entry::invariance`], [`Entry::declared_symbols`]):
//!
//! - The key write normalizes each row in its warp with a fixed tree and
//!   pools a group from the four tokens' BF16 normalized keys and gates in
//!   offset order, whether earlier tokens come from the launch's rows or
//!   from the tail state (which holds exactly those BF16 values). A prefill
//!   key write therefore leaves the same pooled keys and live tail slots
//!   under any chunk split, and a sequence's rows ignore the other
//!   sequences' (`Chunk::Any`, `Batch::Independent`). It keeps per-slot
//!   tail state but no spare: the state at an anchor on a multiple of the
//!   group is empty (`Anchors::Aligned`). A verify key write's rows, pooled
//!   keys and per-row checkpoints equal successive one-row decode key
//!   writes of the same template (`Rows::OneRow` up to its capacity; the
//!   decode entry is its one-row entry). A decode key write claims
//!   nothing. Tail bytes past the live slots follow the sequence's history
//!   and are never claimed.
//! - A selection's split kernel writes per-split lists whose bits follow
//!   the split count, which the prepared rows set: it declares nothing and
//!   is composed with its split merge. The merge, and every form written
//!   in its place (`merge_tree`, `merge_radix`, `rows`), give each row the
//!   bits of its one-row launch: a pool's score is one 128-K MMA chain from
//!   zero then a head-ordered FMA chain whatever split or tile holds it,
//!   and every merge and trim selects exactly over distinct keys (score
//!   bits, then the inverted pool), equal scores keeping the lower pool.
//!   They claim one-row bits up to the capacity, any chunk split, row
//!   independence and `Ties::LowerIndex`; so do the per-rank merges
//!   (`TopKMerge`), which merge each row's rank lists alone by the same
//!   exact order, whatever the launch's rows. The compaction gathers whole
//!   list rows and claims one-row bits, any chunk split and row
//!   independence.

pub mod block_merge;
pub mod head_sum;
pub mod merge_radix;
pub mod merge_tree;
pub mod position;
pub mod position_bf16;
pub mod position_blocks;
pub mod position_candidates_runtime;
pub mod position_query;
pub mod position_query_parallel;
pub mod position_simt_tile;
pub mod rows;

use std::path::PathBuf;

use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan, MAX_COMM_RANKS,
    NumericalContract, OpFamily, RecurrentPhase, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::gated_pooling::GatedGroupPooling;
use kernel_core::pooled_indexer::{
    IndexerGranularity, IndexerKeyFormat, IndexerSelectionMode, PooledIndexerQuery,
};
use kernel_core::properties::{Anchors, Invariance, Ties};

const SMS: u32 = 188;

/// Refuses a per-rank selection over more ranks than SM120's collectives
/// hold.
pub fn check_ranks(query: &PooledIndexerQuery) -> Result<(), Refusal> {
    match *query {
        PooledIndexerQuery::Select {
            mode: IndexerSelectionMode::PerRank { ranks },
            ..
        }
        | PooledIndexerQuery::HeadSumSelect {
            mode: IndexerSelectionMode::PerRank { ranks },
            ..
        } if ranks as usize > MAX_COMM_RANKS => {
            Err(Refusal("per-rank indexer selection needs 2 to 8 ranks"))
        }
        _ => Ok(()),
    }
}
/// The widest split of one row's pools.
pub const MAX_SPLITS: u32 = 64;
/// Pools one scoring tile covers; `Select::tile`.
pub const TILE_POOLS: u32 = 64;
const KEY_WRITE_WARPS: u32 = 4;
const SELECT_THREADS: u32 = 256;
const COMPACT_THREADS: u32 = 128;

/// Static indexer geometry of one profile, shared by its entries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub heads: u32,
    pub head_dim: u32,
    /// Tokens per pool.
    pub group: u32,
    pub page_size: u32,
    /// The compiled top-k maximum in tokens; the runtime value is at most
    /// this.
    pub max_top_k: u32,
    pub always_tail: bool,
    pub position_dtype: DType,
    pub key_norm_eps: f32,
    /// Rows of one verify item.
    pub verify_rows: u32,
}

impl Geometry {
    pub fn pooling(&self) -> GatedGroupPooling {
        GatedGroupPooling {
            group: self.group,
            width: self.head_dim,
            position_dtype: self.position_dtype,
        }
    }

    /// Pools a row keeps at most.
    pub fn max_kept(&self) -> u32 {
        self.max_top_k / self.group
    }

    /// Entries of a key list row.
    pub fn list_width(&self) -> u32 {
        self.max_top_k + if self.always_tail { self.group - 1 } else { 0 }
    }

    /// Refuses shapes the CUDA templates do not instantiate.
    fn validate(&self) -> Result<(), Refusal> {
        if self.heads == 0
            || !self.heads.is_multiple_of(16)
            || self.heads > 64
            || self.head_dim != 128
            || !matches!(self.group, 2 | 4 | 8)
            || !self.max_kept().is_power_of_two()
            || self.max_kept() < TILE_POOLS
            || !self.page_size.is_multiple_of(self.group)
            || self.verify_rows == 0
            || self.verify_rows > 16
        {
            return Err(Refusal(
                "pooled indexer geometry is outside the SM120 engine",
            ));
        }
        Ok(())
    }

    fn select_traits(&self, per_rank: bool) -> String {
        format!(
            "chop::pooled_indexer::Select<{},{},{},{},{},{},{per_rank}>",
            self.heads, self.head_dim, self.group, self.page_size, self.max_top_k, self.always_tail,
        )
    }

    fn key_write_traits(&self, max_rows: u32) -> String {
        format!(
            "chop::pooled_indexer::KeyWrite<{},{},{},{},{max_rows}>",
            self.head_dim,
            self.group,
            self.page_size,
            self.position_dtype == DType::F32,
        )
    }
}

/// Which kernel an entry instantiates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Key norm, tail state and pooled key write.
    KeyWrite(RecurrentPhase),
    /// Scores and split top-k; its entry also has a split merge. `ranks`
    /// selects the per-rank output.
    Select { ranks: Option<u32> },
    /// Exact merge of `ranks` per-rank selections into the key list.
    TopKMerge { ranks: u32 },
    /// Key list row gather.
    Compact,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_{kernel}`.
    pub name: String,
    pub geometry: Geometry,
    pub kind: Kind,
    /// Rows (key write, selection and merge) or output rows (compaction) a
    /// launch may carry.
    pub capacity: u32,
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

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_pooled_indexer_{}_v1", self.name)
    }

    /// The split merge symbol of a selection entry.
    pub fn merge_symbol(&self) -> Option<String> {
        matches!(self.kind, Kind::Select { .. })
            .then(|| format!("chop_pooled_indexer_{}_merge_v1", self.name))
    }

    /// What the entry's launches guarantee for its output (the module
    /// docs): a selection's is its split merge's, which writes the output.
    /// Nothing for an entry its kernel refuses.
    pub fn invariance(&self) -> Invariance {
        if self.contract().is_err() {
            return Invariance::NONE;
        }
        let selected = Invariance::any_rows(self.capacity).with_ties(Ties::LowerIndex);
        match self.kind {
            Kind::KeyWrite(RecurrentPhase::Prefill) => {
                Invariance::PREFILL.with_anchors(Anchors::Aligned {
                    granule: self.geometry.group,
                })
            }
            Kind::KeyWrite(RecurrentPhase::Decode) => Invariance::NONE.with_anchors(Anchors::None),
            Kind::KeyWrite(RecurrentPhase::Verify { .. }) => {
                Invariance::one_row(self.capacity).with_anchors(Anchors::None)
            }
            Kind::Select { .. } | Kind::TopKMerge { .. } => selected,
            Kind::Compact => Invariance::any_rows(self.capacity),
        }
    }

    /// Each symbol the entry launches with its declaration: a selection's
    /// split kernel declares nothing ([`Entry::composed`]), its split merge
    /// the selection's claims.
    pub fn declared_symbols(&self) -> Vec<(String, Invariance)> {
        match self.merge_symbol() {
            Some(merge) => vec![
                (self.symbol(), Invariance::NONE),
                (merge, self.invariance()),
            ],
            None => vec![(self.symbol(), self.invariance())],
        }
    }

    /// A selection's split kernel and its split merge, which reads the
    /// kernel's per-split lists as the next launch: the kernel keeps the
    /// merge's claims there (`crate::invariance::Composed`).
    pub fn composed(&self) -> Option<crate::invariance::Composed> {
        self.merge_symbol()
            .map(|follower| crate::invariance::Composed {
                symbol: self.symbol(),
                follower,
            })
    }

    /// The entry's kernel-core query.
    pub fn query(&self) -> StaticQuery {
        let g = self.geometry;
        match self.kind {
            Kind::KeyWrite(phase) => StaticQuery::PooledIndexer(PooledIndexerQuery::KeyWrite {
                pooling: g.pooling(),
                key_norm_eps_bits: g.key_norm_eps.to_bits(),
                page_size: g.page_size,
                key_format: IndexerKeyFormat::Bf16,
                phase,
                capacity: self.capacity,
            }),
            Kind::Select { ranks } => StaticQuery::PooledIndexer(PooledIndexerQuery::Select {
                heads: g.heads,
                head_dim: g.head_dim,
                group: g.group,
                max_top_k: g.max_top_k,
                always_tail: g.always_tail,
                granularity: IndexerGranularity::Pool,
                key_format: IndexerKeyFormat::Bf16,
                mode: ranks.map_or(IndexerSelectionMode::Global, |ranks| {
                    IndexerSelectionMode::PerRank { ranks }
                }),
                page_size: g.page_size,
                capacity: self.capacity,
            }),
            Kind::TopKMerge { ranks } => {
                StaticQuery::ContextParallel(ContextParallelQuery::TopKMerge {
                    ranks,
                    group: g.group,
                    max_top_k: g.max_top_k,
                    always_tail: g.always_tail,
                    capacity: self.capacity,
                })
            }
            Kind::Compact => StaticQuery::PooledIndexer(PooledIndexerQuery::CompactSelection {
                width: g.list_width(),
                capacity: self.capacity,
            }),
        }
    }

    /// Checks the query and the geometry against the engine.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.geometry.validate()?;
        let query = self.query();
        match query {
            StaticQuery::PooledIndexer(q) => {
                q.validate()?;
                check_ranks(&q)?;
                q.numerical_contract_id()?;
            }
            StaticQuery::ContextParallel(q) => {
                q.validate()?;
                crate::context_parallel::check_query(&q)?;
                q.numerical_contract_id()?;
            }
            _ => unreachable!("the engine's queries"),
        }
        if let Kind::KeyWrite(RecurrentPhase::Verify { max_rows }) = self.kind
            && max_rows != self.geometry.verify_rows
        {
            return Err(Refusal(
                "a verify key write takes the geometry's verify rows",
            ));
        }
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query,
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
        let (tile_m, tile_n, warps, stages) = match self.kind {
            Kind::KeyWrite(_) => (KEY_WRITE_WARPS as u16, 128, KEY_WRITE_WARPS as u8, 1),
            Kind::Select { .. } => (self.geometry.heads as u16, TILE_POOLS as u16, 8, 2),
            Kind::TopKMerge { .. } => (1, self.geometry.max_kept() as u16, 8, 1),
            Kind::Compact => (1, 128, 4, 1),
        };
        CompileKnobs {
            tile_m,
            tile_n,
            tile_k: 16,
            stages,
            swizzle: 0,
            warps,
            split_k_capacity: if matches!(self.kind, Kind::Select { .. }) {
                MAX_SPLITS as u8
            } else {
                1
            },
        }
    }

    /// Threads per CTA of the entry's main kernel.
    pub fn threads(&self) -> u32 {
        match self.kind {
            Kind::KeyWrite(_) => 32 * KEY_WRITE_WARPS,
            Kind::Select { .. } | Kind::TopKMerge { .. } => SELECT_THREADS,
            Kind::Compact => COMPACT_THREADS,
        }
    }

    /// Dynamic shared bytes of the main kernel (`Select::smem`); launchers
    /// opt in above 48 KiB.
    pub fn dynamic_shared_bytes(&self) -> u32 {
        let g = self.geometry;
        match self.kind {
            Kind::Select { .. } => {
                let row = g.head_dim * 2;
                2 * TILE_POOLS * row
                    + g.heads * row
                    + TILE_POOLS * (g.heads + 1) * 4
                    + g.heads * 4
                    + 2 * TILE_POOLS * 4
                    + topk_smem(g.max_kept())
            }
            _ => 0,
        }
    }

    /// Selection splits per row for a prepared row bucket: enough CTAs to
    /// fill the SMs twice, rounded up to a power of two and capped.
    pub fn splits(&self, prepared_rows: u32) -> u32 {
        if !matches!(self.kind, Kind::Select { .. }) {
            return 1;
        }
        select_splits(prepared_rows)
    }

    /// Workspace bytes for a prepared row bucket: each split's kept keys,
    /// u64 `[rows][splits][max_kept]`. It does not depend on the context.
    pub fn workspace_bytes(&self, prepared_rows: u32) -> u64 {
        if !matches!(self.kind, Kind::Select { .. }) {
            return 0;
        }
        u64::from(prepared_rows)
            * u64::from(self.splits(prepared_rows))
            * u64::from(self.geometry.max_kept())
            * 8
    }

    /// Refuses a launch's context-parallel group rank and size: a key write
    /// takes any group of 1 to 8 ranks, a `Global` selection only 0 of 1, and
    /// a `PerRank` selection a rank of its rank count or 0 of 1 (one rank
    /// owning every pool: the global selection as pairs).
    pub fn check_context_parallel(&self, rank: u32, size: u32) -> Result<(), Refusal> {
        let valid = match self.kind {
            Kind::KeyWrite(_) => (1..=8).contains(&size) && rank < size,
            Kind::Select { ranks: None } => (rank, size) == (0, 1),
            Kind::Select { ranks: Some(ranks) } => {
                (size == ranks && rank < size) || (rank, size) == (0, 1)
            }
            Kind::TopKMerge { .. } | Kind::Compact => (rank, size) == (0, 1),
        };
        if !valid {
            return Err(Refusal(
                "pooled indexer context-parallel rank or size does not fit the entry",
            ));
        }
        Ok(())
    }

    /// Refuses a runtime top-k the entry cannot select.
    pub fn check_top_k(&self, top_k: u32) -> Result<(), Refusal> {
        let g = self.geometry;
        if top_k == 0 || top_k > g.max_top_k || !top_k.is_multiple_of(g.group) {
            return Err(Refusal(
                "runtime indexer top-k is outside the compiled maximum",
            ));
        }
        Ok(())
    }

    /// The main kernel's launch for `prepared_rows` rows (a graph bucket, at
    /// most the capacity). CTAs past the live row count return after the PDL
    /// wait, so one plan serves every live count and graph replay.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.capacity {
            return Err(Refusal("pooled indexer rows exceed the prepared capacity"));
        }
        let grid = match self.kind {
            Kind::KeyWrite(_) => [prepared_rows.div_ceil(KEY_WRITE_WARPS), 1, 1],
            Kind::Select { .. } => [prepared_rows, self.splits(prepared_rows), 1],
            Kind::TopKMerge { .. } | Kind::Compact => [prepared_rows, 1, 1],
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

    /// The split merge launch of a selection entry.
    pub fn merge_plan(&self, prepared_rows: u32) -> Result<Option<LaunchPlan>, Refusal> {
        if !matches!(self.kind, Kind::Select { .. }) {
            return Ok(None);
        }
        let main = self.plan(prepared_rows)?;
        let plan = LaunchPlan {
            grid: [prepared_rows, 1, 1],
            block: [SELECT_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            ..main
        };
        plan.validate()?;
        Ok(Some(plan))
    }

    /// The main kernel's launch ABI.
    pub fn params(&self) -> Vec<EntryParam> {
        let g = self.geometry;
        match self.kind {
            Kind::KeyWrite(phase) => {
                let mut params = vec![
                    pointer("key", "const uint16_t"),
                    u32_param("key_stride"),
                    pointer("gate", "const uint16_t"),
                    u32_param("gate_stride"),
                    pointer("norm_weight", "const uint16_t"),
                    pointer("norm_bias", "const uint16_t"),
                    pointer(
                        "position",
                        if g.position_dtype == DType::F32 {
                            "const float"
                        } else {
                            "const uint16_t"
                        },
                    ),
                    pointer("row_batch", "const int32_t"),
                    pointer("positions", "const int32_t"),
                    pointer("live_rows", "const uint32_t"),
                    pointer("slot_ids", "const int32_t"),
                ];
                if matches!(phase, RecurrentPhase::Verify { .. }) {
                    params.push(pointer("checkpoint_ids", "const int32_t"));
                }
                params.extend([
                    pointer("tail_state", "uint16_t"),
                    pointer("page_tables", "const int32_t"),
                    pointer("pooled_keys", "uint8_t"),
                    u32_param("max_pages"),
                    u64_param("page_stride_bytes"),
                    u32_param("eps_bits"),
                    u32_param("cp_rank"),
                    u32_param("cp_size"),
                ]);
                params
            }
            Kind::Select { .. } => vec![
                pointer("q", "const uint16_t"),
                pointer("head_weights", "const uint16_t"),
                pointer("pooled_keys", "const uint8_t"),
                pointer("page_tables", "const int32_t"),
                pointer("row_batch", "const int32_t"),
                pointer("positions", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("workspace", "uint64_t"),
                u32_param("max_pages"),
                u64_param("page_stride_bytes"),
                u32_param("top_k"),
                u32_param("splits"),
                u32_param("cp_rank"),
                u32_param("cp_size"),
            ],
            Kind::TopKMerge { .. } => vec![
                pointer("pairs", "const uint32_t"),
                pointer("positions", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("lists", "int32_t"),
                u32_param("top_k"),
                u32_param("rank_rows"),
            ],
            Kind::Compact => vec![
                pointer("lists", "const int32_t"),
                pointer("source", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("out", "int32_t"),
            ],
        }
    }

    /// The split merge ABI of a selection entry: `Global` writes i32 key
    /// lists, `PerRank` u32 (score bits, pool) pairs.
    pub fn merge_params(&self) -> Vec<EntryParam> {
        let output = match self.kind {
            Kind::Select { ranks: Some(_) } => "uint32_t",
            _ => "int32_t",
        };
        vec![
            pointer("workspace", "const uint64_t"),
            pointer("positions", "const int32_t"),
            pointer("live_rows", "const uint32_t"),
            pointer("output", output),
            u32_param("top_k"),
            u32_param("splits"),
        ]
    }

    fn body(&self) -> String {
        let g = self.geometry;
        match self.kind {
            Kind::KeyWrite(phase) => {
                let (max_rows, checkpoints) = match phase {
                    RecurrentPhase::Verify { max_rows } => (max_rows, "checkpoint_ids"),
                    _ => (0, "nullptr"),
                };
                format!(
                    "const chop::pooled_indexer::KeyWriteArgs args{{key,key_stride,gate,\
                     gate_stride,norm_weight,norm_bias,position,row_batch,positions,0u,slot_ids,\
                     {checkpoints},tail_state,page_tables,pooled_keys,max_pages,\
                     page_stride_bytes,__uint_as_float(eps_bits),cp_rank,cp_size}};\
                     chop::pooled_indexer::key_write<{}>(args,live_rows);",
                    g.key_write_traits(max_rows)
                )
            }
            Kind::Select { ranks } => format!(
                "chop::pooled_indexer::select_split<{}>(q,head_weights,pooled_keys,page_tables,\
                 row_batch,positions,live_rows,workspace,max_pages,page_stride_bytes,top_k,\
                 splits,cp_rank,cp_size);",
                g.select_traits(ranks.is_some())
            ),
            Kind::TopKMerge { ranks } => format!(
                "chop::pooled_indexer::topk_merge<{},{ranks}>(pairs,positions,live_rows,lists,\
                 top_k,rank_rows);",
                g.select_traits(false)
            ),
            Kind::Compact => format!(
                "chop::pooled_indexer::compact_selection<{}>(lists,source,live_rows,out);",
                g.list_width()
            ),
        }
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
        if let (Some(merge), Kind::Select { ranks }) = (self.merge_symbol(), self.kind) {
            specs.push(EntrySpec::new(
                merge,
                self.merge_params(),
                format!(
                    "chop::pooled_indexer::select_merge<{}>(workspace,positions,live_rows,\
                     output,top_k,splits);",
                    self.geometry.select_traits(ranks.is_some())
                ),
                SELECT_THREADS,
                1,
            ));
        }
        Ok(specs)
    }
}

/// Where a context-parallel group rank keeps a pooled key: row `row` of
/// its local page `table[stripe]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolAddress {
    pub stripe: u64,
    pub row: u32,
}

/// The address of pool `pool` on group rank `rank` of `size` (README.md
/// "Pages"), `None` when another rank owns its page: physical page j of a
/// sequence lives on group rank `j % size` at stripe `j / size`. The key
/// write's store rule.
pub fn pool_address(g: &Geometry, pool: u64, rank: u32, size: u32) -> Option<PoolAddress> {
    let token = pool * u64::from(g.group);
    let physical = token / u64::from(g.page_size);
    (physical % u64::from(size) == u64::from(rank)).then(|| PoolAddress {
        stripe: physical / u64::from(size),
        row: (token % u64::from(g.page_size) / u64::from(g.group)) as u32,
    })
}

/// Pools group rank `rank` of `size` owns among a sequence's first `pools`
/// (`rank_pools` in select.cuh).
pub fn rank_pools(g: &Geometry, pools: u64, rank: u32, size: u32) -> u64 {
    let page_pools = u64::from(g.page_size / g.group);
    let stripe = page_pools * u64::from(size);
    let before = u64::from(rank) * page_pools;
    let tail = pools % stripe;
    pools / stripe * page_pools + tail.saturating_sub(before).min(page_pools)
}

/// The sequence pool of group rank `rank`'s local pool `local`, at row
/// `local % (page_size / group)` of stripe `local / (page_size / group)`
/// (`rank_pool` in select.cuh).
pub fn rank_pool(g: &Geometry, local: u64, rank: u32, size: u32) -> u64 {
    let page_pools = u64::from(g.page_size / g.group);
    (local / page_pools * u64::from(size) + u64::from(rank)) * page_pools + local % page_pools
}

/// Selection splits per row for a prepared row bucket: enough CTAs to fill
/// the SMs twice, rounded up to a power of two and capped.
fn select_splits(prepared_rows: u32) -> u32 {
    (2 * SMS)
        .div_ceil(prepared_rows.max(1))
        .next_power_of_two()
        .min(MAX_SPLITS)
}

/// Shared bytes of the streaming top-k buffer (`TopK::smem`).
fn topk_smem(max_kept: u32) -> u32 {
    2 * max_kept * 8 + 16
}

/// The served entries of geometry `g` named `{tag}_{kernel}` after
/// `profile`: key writes for prefill chunks up to 2048 rows, decode batches
/// up to 32 and verify batches of 32 items; one selection for every phase up
/// to 2048 rows, its per-rank forms and their exact merges at 2 and 4
/// ranks; and the compaction of decode-batch key lists.
pub fn entries(profile: Profile, g: Geometry) -> Vec<Entry> {
    let entry = |name: &str, kind, capacity| Entry {
        profile,
        name: format!("{}_{name}", profile.tag()),
        geometry: g,
        kind,
        capacity,
    };
    let mut entries = vec![
        entry(
            "key_write_prefill",
            Kind::KeyWrite(RecurrentPhase::Prefill),
            2048,
        ),
        entry(
            "key_write_decode",
            Kind::KeyWrite(RecurrentPhase::Decode),
            32,
        ),
        entry(
            "key_write_verify",
            Kind::KeyWrite(RecurrentPhase::Verify {
                max_rows: g.verify_rows,
            }),
            32 * g.verify_rows,
        ),
        entry("select", Kind::Select { ranks: None }, 2048),
    ];
    for ranks in [2, 4] {
        entries.push(entry(
            &format!("select_per_rank{ranks}"),
            Kind::Select { ranks: Some(ranks) },
            2048,
        ));
        entries.push(entry(
            &format!("topk_merge{ranks}"),
            Kind::TopKMerge { ranks },
            2048,
        ));
    }
    entries.push(entry("compact", Kind::Compact, 32));
    entries
}

/// Every model's pooled indexer entries in `catalogs`, in order.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.pooled_indexer)())
        .collect()
}

/// The cubin of one profile's entries.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for entry in entries.iter().filter(|entry| entry.profile == profile) {
        specs.extend(entry.entry_specs()?);
    }
    Ok(CubinSpec {
        name: format!("pooled_indexer_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec![
            "#include \"pooled_indexer/key_write.cuh\"".into(),
            "#include \"pooled_indexer/select.cuh\"".into(),
        ],
        defines: vec![],
        entries: specs,
    })
}

/// One cubin per profile of `catalogs` that has pooled indexer entries;
/// `None` builds them all.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    let all = catalog(catalogs);
    catalogs
        .profiles()
        .into_iter()
        .filter(|&owner| profile.is_none_or(|selected| selected == owner))
        .filter(|&owner| all.iter().any(|entry| entry.profile == owner))
        .map(|owner| cubin_spec(owner, &all).expect("catalog pooled indexer entries are valid"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "pooled-indexer-test",
        tag: "pooled_indexer_test",
    });

    fn geometry() -> Geometry {
        Geometry {
            heads: 32,
            head_dim: 128,
            group: 4,
            page_size: 64,
            max_top_k: 2048,
            always_tail: true,
            position_dtype: DType::Bf16,
            key_norm_eps: 1e-6,
            verify_rows: 6,
        }
    }

    #[test]
    fn served_entries_validate_and_plan() {
        let entries = entries(PROFILE, geometry());
        assert_eq!(entries.len(), 9);
        for entry in &entries {
            entry.contract().unwrap();
            let plan = entry.plan(entry.capacity).unwrap();
            assert_eq!(plan.block[0], entry.threads());
            assert!(entry.plan(entry.capacity + 1).is_err());
            assert!(entry.plan(0).is_err());
        }
        let select = &entries[3];
        assert_eq!(select.dynamic_shared_bytes(), 58_256);
        assert_eq!(select.plan(1).unwrap().grid, [1, 64, 1]);
        assert_eq!(select.plan(32).unwrap().grid, [32, 16, 1]);
        assert_eq!(select.plan(2048).unwrap().grid, [2048, 1, 1]);
        assert_eq!(select.merge_plan(6).unwrap().unwrap().grid, [6, 1, 1]);
        assert_eq!(entries[0].plan(2048).unwrap().grid, [512, 1, 1]);
        assert_eq!(entries[2].capacity, 192);
        select.check_top_k(2048).unwrap();
        select.check_top_k(512).unwrap();
        for top_k in [0, 2050, 2052] {
            assert!(select.check_top_k(top_k).is_err());
        }
    }

    /// The top-k merges grant the one-row, chunk, batch and tie asks a
    /// context-parallel step makes of them: a 64-row prefill chunk and a
    /// 4-row verify step, at 2 and 4 ranks.
    #[test]
    fn merges_grant_exact_asks_of_context_parallel_steps() {
        use kernel_core::backend::{Backend, Need};
        let entries = entries(PROFILE, geometry());
        let backend = crate::backend::Sm120::new().with_pooled_indexer(entries.clone());
        for ranks in [2, 4] {
            let merge = entries
                .iter()
                .find(|e| e.kind == Kind::TopKMerge { ranks })
                .unwrap();
            assert_eq!(
                merge.invariance(),
                Invariance::any_rows(2048).with_ties(Ties::LowerIndex)
            );
            let StaticQuery::ContextParallel(op) = merge.query() else {
                panic!("a merge is a context-parallel query");
            };
            for (rows, exact) in [
                (64, Invariance::any_rows(64).with_ties(Ties::LowerIndex)),
                (4, Invariance::one_row(4)),
            ] {
                let prepared = backend
                    .indexer_merge(&op, 2048, Need::rows(rows).with_exact(exact))
                    .unwrap_or_else(|e| panic!("{ranks} ranks at {rows} rows: {e}"));
                assert!(prepared.info().guarantees.provides(exact));
            }
        }
    }

    #[test]
    fn context_parallel_launch_scalars_fit_their_entries() {
        let entries = entries(PROFILE, geometry());
        let find = |name: &str| entries.iter().find(|e| e.name.ends_with(name)).unwrap();
        find("_key_write_prefill")
            .check_context_parallel(3, 4)
            .unwrap();
        assert!(
            find("_key_write_prefill")
                .check_context_parallel(4, 4)
                .is_err()
        );
        find("_select").check_context_parallel(0, 1).unwrap();
        assert!(find("_select").check_context_parallel(0, 2).is_err());
        find("_select_per_rank4")
            .check_context_parallel(3, 4)
            .unwrap();
        assert!(
            find("_select_per_rank4")
                .check_context_parallel(1, 2)
                .is_err()
        );
        find("_select_per_rank4")
            .check_context_parallel(0, 1)
            .unwrap();
        find("_select_per_rank2")
            .check_context_parallel(1, 2)
            .unwrap();
    }

    /// The selection keeps no `[rows, pools]` buffer: its workspace is each
    /// split's kept keys, whatever the context.
    #[test]
    fn workspace_is_bounded_independent_of_context() {
        let entries = entries(PROFILE, geometry());
        let select = &entries[3];
        for rows in [1, 6, 32, 192, 2048] {
            let splits = select.splits(rows);
            assert_eq!(
                select.workspace_bytes(rows),
                u64::from(rows) * u64::from(splits) * 512 * 8
            );
            assert!(splits <= MAX_SPLITS && splits.is_power_of_two());
        }
        assert_eq!(select.workspace_bytes(1), 64 * 4096);
        assert_eq!(select.workspace_bytes(2048), 2048 * 4096);
        assert!((1..=2048).all(|rows| select.workspace_bytes(rows) <= 8 << 20));
        assert_eq!(entries[0].workspace_bytes(2048), 0);
    }

    #[test]
    fn unsupported_shapes_refuse() {
        for g in [
            Geometry {
                heads: 24,
                ..geometry()
            },
            Geometry {
                head_dim: 64,
                ..geometry()
            },
            Geometry {
                group: 16,
                ..geometry()
            },
            Geometry {
                max_top_k: 2044,
                ..geometry()
            },
            Geometry {
                verify_rows: 17,
                ..geometry()
            },
        ] {
            assert!(
                entries(PROFILE, g)
                    .iter()
                    .all(|entry| entry.contract().is_err()),
                "{g:?}"
            );
        }
    }
}

pub mod position_live;

pub mod position_valid;
