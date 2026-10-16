//! The exact merge of per-rank head-sum selections written as block ids
//! (`cuda/pooled_indexer/block_merge.cuh`,
//! `ContextParallelQuery::TopKMergeBlocks`): the `[rows, max_top_k / group]`
//! ascending list the block-sparse attention reads, in its own cubin.

use std::path::PathBuf;

use super::SELECT_THREADS;
use super::head_sum::Geometry;
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan, NumericalContract, OpFamily, Refusal,
    StaticQuery, Target, Validate,
};

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    /// `{tag}_topk_merge_blocks{ranks}`.
    pub name: String,
    pub geometry: Geometry,
    pub ranks: u32,
    /// Rows a launch may carry.
    pub capacity: u32,
}

fn pointer(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

impl Entry {
    pub fn symbol(&self) -> String {
        format!("chop_pooled_indexer_{}_v1", self.name)
    }

    pub fn query(&self) -> ContextParallelQuery {
        let g = self.geometry;
        ContextParallelQuery::TopKMergeBlocks {
            ranks: self.ranks,
            group: g.group,
            max_top_k: g.max_top_k,
            capacity: self.capacity,
        }
    }

    /// Block ids a row holds: `max_top_k / group`.
    pub fn width(&self) -> u32 {
        self.geometry.max_kept()
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let g = self.geometry;
        // The head-sum traits the merge instantiates.
        if g.heads != 4
            || g.head_dim != 128
            || !matches!(g.group, 2 | 4 | 8)
            || !g.max_kept().is_power_of_two()
            || g.max_kept() < 64
            || !g.page_size.is_multiple_of(g.group)
        {
            return Err(Refusal(
                "block-id merge geometry is outside the SM120 engine",
            ));
        }
        let query = self.query();
        query.validate()?;
        crate::context_parallel::check_query(&query)?;
        query.numerical_contract_id()?;
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::ContextParallel(query),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: g.max_kept() as u16,
                    tile_k: 16,
                    stages: 1,
                    swizzle: 0,
                    warps: (SELECT_THREADS / 32) as u8,
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

    /// One CTA per prepared row; rows past the live count return after the
    /// PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.capacity {
            return Err(Refusal("block-id merge rows exceed the prepared capacity"));
        }
        let plan = LaunchPlan {
            grid: [prepared_rows, 1, 1],
            block: [SELECT_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn params(&self) -> Vec<EntryParam> {
        vec![
            pointer("pairs", "const uint32_t"),
            pointer("live_rows", "const uint32_t"),
            pointer("blocks", "int32_t"),
            EntryParam::U32 {
                name: "top_k".into(),
            },
            EntryParam::U32 {
                name: "rank_rows".into(),
            },
        ]
    }

    fn body(&self) -> String {
        let g = self.geometry;
        format!(
            "chop::pooled_indexer::topk_merge_blocks<chop::pooled_indexer::HeadSumSelect<{},{},{},\
             {},{},{},false>,{}>(pairs,live_rows,blocks,top_k,rank_rows);",
            g.heads, g.head_dim, g.group, g.page_size, g.max_top_k, g.always_tail, self.ranks
        )
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            SELECT_THREADS,
            1,
        ))
    }
}

/// The block-id merges of geometry `g` at each group size in `sizes`, named
/// `{tag}_topk_merge_blocks{ranks}`, for up to 2048 rows.
pub fn entries(profile: Profile, g: Geometry, sizes: &[u32]) -> Vec<Entry> {
    sizes
        .iter()
        .map(|&ranks| Entry {
            profile,
            name: format!("{}_topk_merge_blocks{ranks}", profile.tag()),
            geometry: g,
            ranks,
            capacity: 2048,
        })
        .collect()
}

/// The cubin of one profile's block-id merges, `block_merge_{tag}_sm120`.
pub fn cubin_spec(profile: Profile, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let specs = entries
        .iter()
        .filter(|entry| entry.profile == profile)
        .map(Entry::entry_spec)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubinSpec {
        name: format!("block_merge_{}_sm120", profile.tag()),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec!["#include \"pooled_indexer/block_merge.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "block-merge-test",
        tag: "block_merge_test",
    });

    const GEOMETRY: Geometry = Geometry {
        heads: 4,
        head_dim: 128,
        group: 4,
        page_size: 64,
        max_top_k: 2048,
        always_tail: true,
    };

    #[test]
    fn entries_validate_and_plan() {
        let entries = entries(PROFILE, GEOMETRY, &[2, 4]);
        assert_eq!(entries.len(), 2);
        for entry in &entries {
            entry.contract().unwrap();
            assert_eq!(entry.plan(7).unwrap().grid, [7, 1, 1]);
            assert!(entry.plan(2049).is_err());
            assert_eq!(entry.width(), 512);
        }
        assert!(
            entries[0]
                .body()
                .contains("HeadSumSelect<4,128,4,64,2048,true,false>,2>")
        );
        assert_eq!(
            cubin_spec(PROFILE, &entries).unwrap().name,
            "block_merge_block_merge_test_sm120"
        );
        assert!(
            super::entries(PROFILE, GEOMETRY, &[1])[0]
                .contract()
                .is_err()
        );
    }
}
