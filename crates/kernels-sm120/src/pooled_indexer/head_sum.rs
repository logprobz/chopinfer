//! Head-sum selection entries (`cuda/pooled_indexer/head_sum.cuh`,
//! `cuda/pooled_indexer/README.md` "Head-sum selection"): the pooled
//! indexer's streaming selection, per-rank pairs and exact merge, scored
//! with the block-sparse indexer's unweighted head sum
//! (`PooledIndexerQuery::HeadSumSelect`). The entries take the keys and
//! queries that indexer writes: BF16 block keys `[page][page / group][128]`
//! at their own page stride, through the KV page table. They build into
//! their own cubin, so the weighted selection's cubins do not change.

use std::path::PathBuf;

use super::{
    MAX_SPLITS, SELECT_THREADS, TILE_POOLS, pointer, select_splits, topk_smem, u32_param, u64_param,
};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan, NumericalContract, OpFamily, Refusal,
    StaticQuery, Target, Validate,
};
use kernel_core::pooled_indexer::{IndexerSelectionMode, PooledIndexerQuery};

/// Query rows of the scoring tile: the heads, then zero rows.
const Q_ROWS: u32 = 16;

/// Static geometry of one profile's head-sum entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub heads: u32,
    pub head_dim: u32,
    /// Tokens per pool (block).
    pub group: u32,
    pub page_size: u32,
    /// The compiled top-k maximum in tokens.
    pub max_top_k: u32,
    pub always_tail: bool,
}

impl Geometry {
    /// Pools a row keeps at most.
    pub fn max_kept(&self) -> u32 {
        self.max_top_k / self.group
    }

    /// Entries of a key list row.
    pub fn list_width(&self) -> u32 {
        self.max_top_k + if self.always_tail { self.group - 1 } else { 0 }
    }

    /// Refuses shapes the CUDA template does not instantiate: four heads of
    /// 128 (the contract's eight and sixteen are not built).
    fn validate(&self) -> Result<(), Refusal> {
        if self.heads != 4
            || self.head_dim != 128
            || !matches!(self.group, 2 | 4 | 8)
            || !self.max_kept().is_power_of_two()
            || self.max_kept() < TILE_POOLS
            || !self.page_size.is_multiple_of(self.group)
        {
            return Err(Refusal(
                "head-sum indexer geometry is outside the SM120 engine",
            ));
        }
        Ok(())
    }

    fn traits(&self, per_rank: bool) -> String {
        format!(
            "chop::pooled_indexer::HeadSumSelect<{},{},{},{},{},{},{per_rank}>",
            self.heads, self.head_dim, self.group, self.page_size, self.max_top_k, self.always_tail,
        )
    }
}

/// Which kernel an entry instantiates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// Scores and split top-k; its entry also has a split merge. `ranks`
    /// selects the per-rank output.
    Select { ranks: Option<u32> },
    /// Exact merge of `ranks` per-rank selections into the key list.
    TopKMerge { ranks: u32 },
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_head_sum_{kernel}`.
    pub name: String,
    pub geometry: Geometry,
    pub kind: Kind,
    /// Rows a launch may carry.
    pub capacity: u32,
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

    /// The entry's kernel-core query.
    pub fn query(&self) -> StaticQuery {
        let g = self.geometry;
        match self.kind {
            Kind::Select { ranks } => {
                StaticQuery::PooledIndexer(PooledIndexerQuery::HeadSumSelect {
                    heads: g.heads,
                    head_dim: g.head_dim,
                    group: g.group,
                    max_top_k: g.max_top_k,
                    always_tail: g.always_tail,
                    mode: ranks.map_or(IndexerSelectionMode::Global, |ranks| {
                        IndexerSelectionMode::PerRank { ranks }
                    }),
                    page_size: g.page_size,
                    capacity: self.capacity,
                })
            }
            Kind::TopKMerge { ranks } => {
                StaticQuery::ContextParallel(ContextParallelQuery::TopKMerge {
                    ranks,
                    group: g.group,
                    max_top_k: g.max_top_k,
                    always_tail: g.always_tail,
                    capacity: self.capacity,
                })
            }
        }
    }

    /// Checks the query and the geometry against the engine.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.geometry.validate()?;
        let query = self.query();
        match query {
            StaticQuery::PooledIndexer(q) => {
                q.validate()?;
                super::check_ranks(&q)?;
                q.numerical_contract_id()?;
            }
            StaticQuery::ContextParallel(q) => {
                q.validate()?;
                crate::context_parallel::check_query(&q)?;
                q.numerical_contract_id()?;
            }
            _ => unreachable!("the engine's queries"),
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
        let (tile_m, tile_n, stages, split_k_capacity) = match self.kind {
            Kind::Select { .. } => (Q_ROWS as u16, TILE_POOLS as u16, 2, MAX_SPLITS as u8),
            Kind::TopKMerge { .. } => (1, self.geometry.max_kept() as u16, 1, 1),
        };
        CompileKnobs {
            tile_m,
            tile_n,
            tile_k: 16,
            stages,
            swizzle: 0,
            warps: (SELECT_THREADS / 32) as u8,
            split_k_capacity,
        }
    }

    /// Dynamic shared bytes of the split kernel (`HeadSumSelect::smem`).
    pub fn dynamic_shared_bytes(&self) -> u32 {
        let g = self.geometry;
        match self.kind {
            Kind::Select { .. } => {
                let row = g.head_dim * 2;
                2 * TILE_POOLS * row
                    + Q_ROWS * row
                    + TILE_POOLS * 4
                    + 2 * TILE_POOLS * 4
                    + topk_smem(g.max_kept())
            }
            Kind::TopKMerge { .. } => 0,
        }
    }

    /// Selection splits per row for a prepared row bucket, as the weighted
    /// selection splits.
    pub fn splits(&self, prepared_rows: u32) -> u32 {
        match self.kind {
            Kind::Select { .. } => select_splits(prepared_rows),
            Kind::TopKMerge { .. } => 1,
        }
    }

    /// Workspace bytes: each split's kept keys, u64 `[rows][splits][max_kept]`.
    pub fn workspace_bytes(&self, prepared_rows: u32) -> u64 {
        match self.kind {
            Kind::Select { .. } => {
                u64::from(prepared_rows)
                    * u64::from(self.splits(prepared_rows))
                    * u64::from(self.geometry.max_kept())
                    * 8
            }
            Kind::TopKMerge { .. } => 0,
        }
    }

    /// Refuses a launch's context-parallel group rank and size: a `Global`
    /// selection takes only 0 of 1, a `PerRank` selection a rank of its rank
    /// count or 0 of 1, the merge 0 of 1.
    pub fn check_context_parallel(&self, rank: u32, size: u32) -> Result<(), Refusal> {
        let valid = match self.kind {
            Kind::Select { ranks: Some(ranks) } => {
                (size == ranks && rank < size) || (rank, size) == (0, 1)
            }
            Kind::Select { ranks: None } | Kind::TopKMerge { .. } => (rank, size) == (0, 1),
        };
        if !valid {
            return Err(Refusal(
                "head-sum indexer context-parallel rank or size does not fit the entry",
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

    /// The main kernel's launch for `prepared_rows` rows; CTAs past the live
    /// row count return after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.capacity {
            return Err(Refusal(
                "head-sum indexer rows exceed the prepared capacity",
            ));
        }
        let plan = LaunchPlan {
            grid: [prepared_rows, self.splits(prepared_rows), 1],
            block: [SELECT_THREADS, 1, 1],
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
            dynamic_shared_bytes: 0,
            ..main
        };
        plan.validate()?;
        Ok(Some(plan))
    }

    /// The main kernel's launch ABI: the weighted selection's without head
    /// weights, or the merge's.
    pub fn params(&self) -> Vec<EntryParam> {
        match self.kind {
            Kind::Select { .. } => vec![
                pointer("q", "const uint16_t"),
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
        }
    }

    /// The split merge ABI: `Global` writes i32 key lists, `PerRank` u32
    /// (score bits, pool) pairs.
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
            Kind::Select { ranks } => format!(
                "chop::pooled_indexer::head_sum_split<{}>(q,pooled_keys,page_tables,row_batch,\
                 positions,live_rows,workspace,max_pages,page_stride_bytes,top_k,splits,cp_rank,\
                 cp_size);",
                g.traits(ranks.is_some())
            ),
            Kind::TopKMerge { ranks } => format!(
                "chop::pooled_indexer::topk_merge<{},{ranks}>(pairs,positions,live_rows,lists,\
                 top_k,rank_rows);",
                g.traits(false)
            ),
        }
    }

    pub fn entry_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.contract()?;
        let mut specs = vec![EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            SELECT_THREADS,
            1,
        )];
        if let (Some(merge), Kind::Select { ranks }) = (self.merge_symbol(), self.kind) {
            specs.push(EntrySpec::new(
                merge,
                self.merge_params(),
                format!(
                    "chop::pooled_indexer::select_merge<{}>(workspace,positions,live_rows,\
                     output,top_k,splits);",
                    self.geometry.traits(ranks.is_some())
                ),
                SELECT_THREADS,
                1,
            ));
        }
        Ok(specs)
    }
}

/// The head-sum entries of geometry `g` named `{tag}_head_sum_{kernel}`
/// after `profile`: the selection for up to 2048 rows, its per-rank forms
/// and their exact merges at 2 and 4 ranks.
pub fn entries(profile: Profile, g: Geometry) -> Vec<Entry> {
    let entry = |name: &str, kind| Entry {
        profile,
        name: format!("{}_head_sum_{name}", profile.tag()),
        geometry: g,
        kind,
        capacity: 2048,
    };
    let mut entries = vec![entry("select", Kind::Select { ranks: None })];
    for ranks in [2, 4] {
        entries.push(entry(
            &format!("select_per_rank{ranks}"),
            Kind::Select { ranks: Some(ranks) },
        ));
        entries.push(entry(
            &format!("topk_merge{ranks}"),
            Kind::TopKMerge { ranks },
        ));
    }
    entries
}

/// The cubin of one profile's head-sum entries.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for entry in entries.iter().filter(|entry| entry.profile == profile) {
        specs.extend(entry.entry_specs()?);
    }
    Ok(CubinSpec {
        name: format!("pooled_indexer_head_sum_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"pooled_indexer/head_sum.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "head-sum-test",
        tag: "head_sum_test",
    });

    fn geometry() -> Geometry {
        Geometry {
            heads: 4,
            head_dim: 128,
            group: 4,
            page_size: 64,
            max_top_k: 2048,
            always_tail: true,
        }
    }

    #[test]
    fn entries_validate_and_plan() {
        let entries = entries(PROFILE, geometry());
        assert_eq!(entries.len(), 5);
        for entry in &entries {
            entry.contract().unwrap();
            assert!(entry.plan(entry.capacity).is_ok());
            assert!(entry.plan(entry.capacity + 1).is_err());
            assert!(entry.plan(0).is_err());
        }
        let select = &entries[0];
        assert_eq!(select.name, "head_sum_test_head_sum_select");
        // 32 KiB of key stages, 4 KiB of query tile, the scores, flags and
        // the top-k of 512 pools: below the 48 KiB opt-in.
        assert_eq!(select.dynamic_shared_bytes(), 45_840);
        assert_eq!(select.plan(1).unwrap().grid, [1, 64, 1]);
        assert_eq!(select.plan(2048).unwrap().grid, [2048, 1, 1]);
        assert_eq!(select.merge_plan(6).unwrap().unwrap().grid, [6, 1, 1]);
        assert_eq!(select.workspace_bytes(1), 64 * 512 * 8);
        assert_eq!(entries[2].merge_plan(6).unwrap(), None);
        select.check_top_k(2048).unwrap();
        assert!(select.check_top_k(2050).is_err());
        let spec = cubin_spec(PROFILE, &entries).unwrap();
        assert_eq!(spec.name, "pooled_indexer_head_sum_head_sum_test_sm120");
        assert_eq!(spec.entries.len(), 8);
    }

    #[test]
    fn context_parallel_ranks_fit_their_entries() {
        let entries = entries(PROFILE, geometry());
        let find = |name: &str| entries.iter().find(|e| e.name.ends_with(name)).unwrap();
        find("_select").check_context_parallel(0, 1).unwrap();
        assert!(find("_select").check_context_parallel(1, 2).is_err());
        find("_select_per_rank2")
            .check_context_parallel(1, 2)
            .unwrap();
        find("_select_per_rank2")
            .check_context_parallel(0, 1)
            .unwrap();
        assert!(
            find("_select_per_rank2")
                .check_context_parallel(1, 4)
                .is_err()
        );
        assert!(find("_topk_merge4").check_context_parallel(1, 4).is_err());
    }

    #[test]
    fn other_shapes_refuse() {
        for g in [
            Geometry {
                heads: 8,
                ..geometry()
            },
            Geometry {
                heads: 32,
                ..geometry()
            },
            Geometry {
                head_dim: 64,
                ..geometry()
            },
            Geometry {
                max_top_k: 2044,
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
