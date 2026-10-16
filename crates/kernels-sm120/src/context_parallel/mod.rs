//! Context-parallel entries: the cross-rank merge of latent attention
//! partials and the key list localization (`cuda/context_parallel/merge.cuh`,
//! `cuda/context_parallel/README.md`), and the group-scoped launch of the
//! shared comm transport ([`group`]). Planners never allocate; live row
//! counts stay on the device and grids are sized for the prepared rows.

pub mod group;
pub mod phased;

use std::path::PathBuf;

use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan, MAX_COMM_RANKS, NumericalContract,
    OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::properties::Invariance;

const MERGE_THREADS: u32 = 128;
const LOCALIZE_THREADS: u32 = 256;

/// The group sizes the served entries instantiate.
pub const GROUP_SIZES: [u32; 2] = [2, 4];

/// Refuses a context-parallel query past SM120's group limit: the comm
/// entries that move a group's partials, queries and selections hold
/// `MAX_COMM_RANKS` members. The attention partials' head width granule is
/// their templates' (`paged_partials`, `selected_partials`).
pub fn check_query(query: &ContextParallelQuery) -> Result<(), Refusal> {
    let ranks = match *query {
        ContextParallelQuery::PartialMerge { ranks, .. }
        | ContextParallelQuery::TopKMerge { ranks, .. }
        | ContextParallelQuery::LocalizeKeys { ranks, .. }
        | ContextParallelQuery::OwnedQkNormRopeKvWrite { ranks, .. }
        | ContextParallelQuery::OwnedIndexerKeyWrite { ranks, .. }
        | ContextParallelQuery::TopKMergeBlocks { ranks, .. } => ranks,
        ContextParallelQuery::PagedAttentionPartials { .. }
        | ContextParallelQuery::SelectedAttentionPartials { .. } => return Ok(()),
    };
    if ranks as usize > MAX_COMM_RANKS {
        return Err(Refusal(
            "an SM120 context-parallel group holds at most 8 ranks",
        ));
    }
    Ok(())
}

/// Static shape of one profile's context-parallel entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    /// Heads one rank merges: its own attention heads.
    pub heads: u32,
    /// Latent width of a partial numerator and of the merged output.
    pub width: u32,
    /// Tokens of one physical KV page.
    pub page_size: u32,
    /// Entries of a key list row.
    pub list_width: u32,
    /// Rows a launch may carry.
    pub capacity: u32,
}

/// Which kernel an entry instantiates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    /// The rank-ordered merge of `ranks` gathered partials.
    PartialMerge { ranks: u32 },
    /// One group rank's local key lists in a group of `ranks`.
    LocalizeKeys { ranks: u32 },
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_{kernel}`.
    pub name: String,
    pub geometry: Geometry,
    pub kind: Kind,
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

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_context_parallel_{}_v1", self.name)
    }

    /// What every launch of the entry keeps, for a model to declare
    /// (`Sm120::with_context_parallel_claims`): one-row bits up to its
    /// capacity, any chunk split and row independence. Each CTA computes
    /// one row's values from that row's inputs alone, in a fixed order,
    /// and rows at or past the live count return: the merge's CTA folds
    /// its (row, head) pair over the ranks in rank order, the
    /// localization maps each position of its row by integer arithmetic.
    /// No split, grid or other row changes a row's bits.
    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.geometry.capacity)
    }

    /// The entry's kernel-core query.
    pub fn query(&self) -> ContextParallelQuery {
        let g = self.geometry;
        match self.kind {
            Kind::PartialMerge { ranks } => ContextParallelQuery::PartialMerge {
                ranks,
                heads: g.heads,
                width: g.width,
                capacity: g.capacity,
            },
            Kind::LocalizeKeys { ranks } => ContextParallelQuery::LocalizeKeys {
                ranks,
                page: g.page_size,
                width: g.list_width,
                capacity: g.capacity,
            },
        }
    }

    /// Checks the query against the engine.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let query = self.query();
        query.validate()?;
        check_query(&query)?;
        query.numerical_contract_id()?;
        let (tile_m, tile_n, warps) = match self.kind {
            Kind::PartialMerge { .. } => (1, self.geometry.width, MERGE_THREADS / 32),
            Kind::LocalizeKeys { .. } => (1, LOCALIZE_THREADS, LOCALIZE_THREADS / 32),
        };
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::ContextParallel(query),
                knobs: CompileKnobs {
                    tile_m,
                    tile_n: u16::try_from(tile_n)
                        .map_err(|_| Refusal("context-parallel tile exceeds u16"))?,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: warps as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol(),
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn threads(&self) -> u32 {
        match self.kind {
            Kind::PartialMerge { .. } => MERGE_THREADS,
            Kind::LocalizeKeys { .. } => LOCALIZE_THREADS,
        }
    }

    /// The launch for `prepared_rows` rows (a graph bucket, at most the
    /// capacity). CTAs past the live row count return after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.geometry.capacity {
            return Err(Refusal(
                "context-parallel rows exceed the prepared capacity",
            ));
        }
        let grid = match self.kind {
            Kind::PartialMerge { .. } => [prepared_rows, self.geometry.heads, 1],
            Kind::LocalizeKeys { .. } => [
                prepared_rows,
                self.geometry.list_width.div_ceil(LOCALIZE_THREADS),
                1,
            ],
        };
        let plan = LaunchPlan {
            grid,
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Refuses a merge's launch scalars. The CTA of (row, head) reads pair
    /// `row * input_heads + head_offset + head` of each of the ranks'
    /// segments of `rank_rows * input_heads` pairs (merge.cuh): the rank's
    /// heads must fit one row of `input_heads` pairs, every live row's pairs
    /// must lie in the segment, and the segments must hold the prepared rows.
    pub fn check_merge(&self, prepared_rows: u32, layout: MergeLayout) -> Result<(), Refusal> {
        let Kind::PartialMerge { .. } = self.kind else {
            return Err(Refusal("merge scalars belong to a partial merge entry"));
        };
        let MergeLayout {
            rank_rows,
            input_heads,
            head_offset,
        } = layout;
        let heads = u64::from(self.geometry.heads);
        let (input_heads, head_offset) = (u64::from(input_heads), u64::from(head_offset));
        if prepared_rows == 0
            || rank_rows < prepared_rows
            || input_heads == 0
            || head_offset % input_heads + heads > input_heads
            || (u64::from(prepared_rows) - 1) * input_heads + head_offset + heads
                > u64::from(rank_rows) * input_heads
        {
            return Err(Refusal(
                "partial merge rows or heads do not fit the gathered segments",
            ));
        }
        Ok(())
    }

    /// Floats of one rank's gathered partial segment for `rank_rows` rows of
    /// `input_heads` heads: `m`, `l`, then `o`.
    pub fn segment_floats(&self, rank_rows: u32, input_heads: u32) -> u64 {
        u64::from(rank_rows) * u64::from(input_heads) * (u64::from(self.geometry.width) + 2)
    }

    /// Refuses a localization's group rank.
    pub fn check_rank(&self, cp_rank: u32) -> Result<(), Refusal> {
        match self.kind {
            Kind::LocalizeKeys { ranks } if cp_rank < ranks => Ok(()),
            _ => Err(Refusal("key localization group rank is outside its group")),
        }
    }

    /// The launch ABI.
    pub fn params(&self) -> Vec<EntryParam> {
        match self.kind {
            Kind::PartialMerge { .. } => vec![
                pointer("partials", "const float"),
                pointer("live_rows", "const uint32_t"),
                pointer("out", "uint16_t"),
                u32_param("rank_rows"),
                u32_param("input_heads"),
                u32_param("head_offset"),
            ],
            Kind::LocalizeKeys { .. } => vec![
                pointer("lists", "const int32_t"),
                pointer("live_rows", "const uint32_t"),
                pointer("out", "int32_t"),
                u32_param("cp_rank"),
            ],
        }
    }

    fn body(&self) -> String {
        let g = self.geometry;
        match self.kind {
            Kind::PartialMerge { ranks } => format!(
                "chop::context_parallel::partial_merge<chop::context_parallel::Merge<{ranks},{},{}>>(\
                 partials,live_rows,out,rank_rows,input_heads,head_offset);",
                g.heads, g.width
            ),
            Kind::LocalizeKeys { ranks } => format!(
                "chop::context_parallel::localize_keys<chop::context_parallel::Localize<{ranks},{},{}>>(\
                 lists,live_rows,out,cp_rank);",
                g.page_size, g.list_width
            ),
        }
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            self.threads(),
            1,
        ))
    }
}

/// Where one rank's partial merge reads its heads in the group's partial
/// segments: the merge's `rank_rows`, `input_heads` and `head_offset`
/// scalars.
///
/// A group rank runs the heads-`H` partial attention entry once per source
/// block s of the gathered queries `[N][rows][H][width]` (the query
/// all-gather's rank-major output): block s, at element
/// [`query_block_offset`], is exactly the row-major `[rows, H, width]`
/// query the entry reads. Its partial outputs for block s go to
/// [`partial_block_offsets`] of the rank's segment, block-major: `m
/// [N][rows][H]`, then `l` (same), then `o [N][rows][H][width]`. Block s
/// holds group rank s's heads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MergeLayout {
    pub rank_rows: u32,
    pub input_heads: u32,
    pub head_offset: u32,
}

impl MergeLayout {
    /// Every rank's whole segment all-gathered: the owner's block of each
    /// segment is `rows` rows of `heads` pairs at pair `owner * rows *
    /// heads`.
    pub fn gathered(rows: u32, heads: u32, ranks: u32, owner: u32) -> Self {
        Self {
            rank_rows: ranks * rows,
            input_heads: heads,
            head_offset: owner * rows * heads,
        }
    }

    /// Only the owner's block of each segment, exchanged to the owner
    /// (the row exchange: `rows` = ranks, one block per destination).
    pub fn exchanged(rows: u32, heads: u32) -> Self {
        Self {
            rank_rows: rows,
            input_heads: heads,
            head_offset: 0,
        }
    }

    /// The pair the CTA of (row, head) reads in each segment (merge.cuh).
    pub fn pair(self, row: u32, head: u32) -> u64 {
        u64::from(row) * u64::from(self.input_heads) + u64::from(self.head_offset) + u64::from(head)
    }

    /// Pairs of one segment; `l` starts here and `o` at twice it.
    pub fn pairs(self) -> u64 {
        u64::from(self.rank_rows) * u64::from(self.input_heads)
    }
}

/// The element of source block `source` in gathered queries `[ranks][rows]
/// [heads][width]`, where the partial attention launch for that block reads.
pub fn query_block_offset(rows: u32, heads: u32, width: u32, source: u32) -> u64 {
    u64::from(source) * u64::from(rows) * u64::from(heads) * u64::from(width)
}

/// The floats in a rank's segment where its partial attention launch for
/// source block `block` writes `m`, `l` and `o` (block-major, as
/// [`MergeLayout`] reads them).
pub fn partial_block_offsets(
    rows: u32,
    heads: u32,
    width: u32,
    ranks: u32,
    block: u32,
) -> [u64; 3] {
    let pairs = u64::from(rows) * u64::from(heads);
    let all = pairs * u64::from(ranks);
    let first = pairs * u64::from(block);
    [first, all + first, 2 * all + first * u64::from(width)]
}

/// The served entries of geometry `g` named `{tag}_{kernel}` after
/// `profile`: at each of [`GROUP_SIZES`], the partial merge, the key list
/// localization and the localization of a step's appended positions (a list
/// of one per row), which routes each KV row write to its owner.
pub fn entries(profile: Profile, g: Geometry) -> Vec<Entry> {
    let positions = Geometry { list_width: 1, ..g };
    GROUP_SIZES
        .into_iter()
        .flat_map(|ranks| {
            [
                (
                    format!("partial_merge{ranks}"),
                    Kind::PartialMerge { ranks },
                    g,
                ),
                (
                    format!("localize_keys{ranks}"),
                    Kind::LocalizeKeys { ranks },
                    g,
                ),
                (
                    format!("localize_positions{ranks}"),
                    Kind::LocalizeKeys { ranks },
                    positions,
                ),
            ]
        })
        .map(|(name, kind, geometry)| Entry {
            profile,
            name: format!("{}_{name}", profile.tag()),
            geometry,
            kind,
        })
        .collect()
}

/// The cubin of one profile's entries.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let specs = entries
        .iter()
        .filter(|entry| entry.profile == profile)
        .map(Entry::entry_spec)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubinSpec {
        name: format!("context_parallel_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"context_parallel/merge.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

/// One model's entries for a context-parallel group of `ranks` on one
/// profile, as its catalog lists them (`ModelCatalog::context_parallel`):
/// the owner-routed writes, the per-rank selection with its split merge,
/// the block-id merge, the selected partials, and the partial merge and
/// position localization ([`Entry`]), with the cubins that hold them.
#[derive(Clone, Debug)]
pub struct GroupEntries {
    pub profile: Profile,
    /// The group size every entry serves.
    pub ranks: u32,
    pub owned: Vec<crate::owned_writes::Entry>,
    pub select: crate::pooled_indexer::head_sum::Entry,
    pub block_merge: crate::pooled_indexer::block_merge::Entry,
    pub partials: Vec<crate::selected_partials::Entry>,
    /// The partial merge and the position localization.
    pub group: Vec<Entry>,
    /// The cubins the model builds for the entries.
    pub cubins: Vec<CubinSpec>,
}

impl GroupEntries {
    /// Every entry symbol, with the selection's split merge, in the order
    /// the fields list them.
    pub fn symbols(&self) -> Vec<String> {
        self.owned
            .iter()
            .map(crate::owned_writes::Entry::symbol)
            .chain([self.select.symbol()])
            .chain(self.select.merge_symbol())
            .chain([self.block_merge.symbol()])
            .chain(
                self.partials
                    .iter()
                    .map(crate::selected_partials::Entry::symbol),
            )
            .chain(self.group.iter().map(Entry::symbol))
            .collect()
    }

    /// Refuses cubins that do not hold exactly the entries' symbols.
    pub fn check(&self) -> Result<(), Refusal> {
        let mut built = self
            .cubins
            .iter()
            .flat_map(|cubin| cubin.entries.iter().map(|entry| entry.symbol.clone()))
            .collect::<Vec<_>>();
        let mut symbols = self.symbols();
        built.sort();
        symbols.sort();
        if built != symbols {
            return Err(Refusal(
                "a context-parallel group's cubins do not hold its entries",
            ));
        }
        Ok(())
    }
}

/// Every model's context-parallel cubins of `profile` (every profile's when
/// `None`), in catalog order.
pub fn cubin_specs(catalogs: &crate::Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.context_parallel)())
        .filter(|group| profile.is_none_or(|selected| selected == group.profile))
        .flat_map(|group| group.cubins)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "context-parallel-test",
        tag: "context_parallel_test",
    });

    fn geometry() -> Geometry {
        Geometry {
            heads: 16,
            width: 512,
            page_size: 64,
            list_width: 2051,
            capacity: 8192,
        }
    }

    #[test]
    fn served_entries_validate_and_plan() {
        let entries = entries(PROFILE, geometry());
        assert_eq!(entries.len(), 6);
        for entry in &entries {
            entry.contract().unwrap();
            assert!(entry.plan(8193).is_err());
            assert!(entry.plan(0).is_err());
        }
        assert_eq!(entries[0].name, "context_parallel_test_partial_merge2");
        assert_eq!(entries[0].plan(32).unwrap().grid, [32, 16, 1]);
        assert_eq!(entries[1].plan(32).unwrap().grid, [32, 9, 1]);
        assert_eq!(entries[2].name, "context_parallel_test_localize_positions2");
        assert_eq!(entries[2].plan(32).unwrap().grid, [32, 1, 1]);
        assert_eq!(entries[4].plan(8192).unwrap().block, [256, 1, 1]);
        let spec = cubin_spec(PROFILE, &entries).unwrap();
        assert_eq!(spec.name, "context_parallel_context_parallel_test_sm120");
        assert_eq!(spec.entries.len(), 6);
    }

    #[test]
    fn launch_scalars_fit_their_entries() {
        let entries = entries(PROFILE, geometry());
        let (merge, localize) = (&entries[3], &entries[4]);
        // A group of 4 gathers 64 heads; rank 3 merges heads 48..64.
        let layout = |rank_rows, input_heads, head_offset| MergeLayout {
            rank_rows,
            input_heads,
            head_offset,
        };
        // A row-major [rows][64] segment: rank 3's heads are 48..64.
        merge.check_merge(32, layout(32, 64, 48)).unwrap();
        assert!(merge.check_merge(32, layout(32, 64, 49)).is_err());
        assert!(merge.check_merge(32, layout(31, 64, 0)).is_err());
        // Block-major segments: owner 3's block of 32 rows at group size 4.
        let gathered = MergeLayout::gathered(32, 16, 4, 3);
        assert_eq!(gathered, layout(128, 16, 3 * 32 * 16));
        merge.check_merge(32, gathered).unwrap();
        assert!(merge.check_merge(33, gathered).is_err());
        assert!(
            merge
                .check_merge(32, layout(128, 16, 3 * 32 * 16 + 1))
                .is_err()
        );
        assert!(
            merge
                .check_merge(32, MergeLayout::gathered(32, 16, 4, 4))
                .is_err()
        );
        merge
            .check_merge(32, MergeLayout::exchanged(32, 16))
            .unwrap();
        assert_eq!(gathered.pair(1, 2), 16 + 1536 + 2);
        assert_eq!(query_block_offset(32, 16, 512, 3), 3 * 32 * 16 * 512);
        assert_eq!(
            partial_block_offsets(32, 16, 512, 4, 3),
            [1536, 2048 + 1536, 4096 + 1536 * 512]
        );
        assert!(merge.check_rank(0).is_err());
        assert_eq!(merge.segment_floats(32, 64), 32 * 64 * 514);
        localize.check_rank(3).unwrap();
        assert!(localize.check_rank(4).is_err());
        assert!(
            localize
                .check_merge(1, MergeLayout::exchanged(1, 16))
                .is_err()
        );
    }

    #[test]
    fn unsupported_shapes_refuse() {
        for g in [
            Geometry {
                heads: 0,
                ..geometry()
            },
            Geometry {
                width: 0,
                ..geometry()
            },
            Geometry {
                capacity: 0,
                ..geometry()
            },
        ] {
            assert!(
                entries(PROFILE, g)
                    .iter()
                    .filter(|entry| matches!(entry.kind, Kind::PartialMerge { .. }))
                    .all(|entry| entry.contract().is_err())
            );
        }
        let no_page = Geometry {
            page_size: 0,
            ..geometry()
        };
        assert!(
            entries(PROFILE, no_page)
                .iter()
                .filter(|entry| matches!(entry.kind, Kind::LocalizeKeys { .. }))
                .all(|entry| entry.contract().is_err())
        );
    }
}
