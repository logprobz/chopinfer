//! The tree merge of a selection's split lists
//! (`cuda/pooled_indexer/merge_tree.cuh`, README "Tree merge"): the key
//! list or per-rank pairs of a selection entry's `select_merge`, bit for
//! bit, from the same workspace and launch ABI, with each row's lists folded
//! in registers by a bitonic merge tree instead of the streaming top-k. In
//! its own cubin, `pooled_indexer_merge_tree_{tag}_sm120`.

use std::path::PathBuf;

use super::{Entry as Select, Kind, MAX_SPLITS};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan, OpFamily, Refusal, Target, Validate,
};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "pooled_indexer/merge_tree.cuh";
/// Warps per CTA: each folds `splits / WARPS` lists before the warps fold
/// pairwise.
pub const WARPS: u32 = 16;
/// Lab-only mutant: the row's last split list left out.
pub const SKIP_LAST_DEFINE: &str = "CHOP_MERGE_TREE_TEST_SKIP_LAST";
/// Lab-only mutant: keys compared by their score bits alone, so equal
/// scores no longer rank the lower pool first.
pub const SCORE_ONLY_DEFINE: &str = "CHOP_MERGE_TREE_TEST_SCORE_ONLY";
/// Lab-only mutant: the dependency wait after the reads of the live count,
/// the positions and the lists.
pub const LATE_WAIT_DEFINE: &str = "CHOP_MERGE_TREE_TEST_LATE_WAIT";
/// The lab's mutants: (symbol suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_skip_last", SKIP_LAST_DEFINE),
    ("_score_only", SCORE_ONLY_DEFINE),
    ("_late_wait", LATE_WAIT_DEFINE),
];

/// The tree merge of one selection entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The selection whose workspace the merge reads and whose merge it
    /// replaces.
    pub select: Select,
    /// Appended to the symbol (lab mutants); empty when served.
    pub suffix: String,
}

impl Entry {
    /// The tree merge of `select`, which must be a selection entry.
    pub fn of(select: &Select) -> Result<Entry, Refusal> {
        if !matches!(select.kind, Kind::Select { .. }) {
            return Err(Refusal("a tree merge folds a selection's split lists"));
        }
        Ok(Entry {
            select: select.clone(),
            suffix: String::new(),
        })
    }

    /// The form's declared exactness: its selection's split merge's, whose
    /// outputs it writes bit for bit; nothing for a lab mutant or a form
    /// its kernel refuses.
    pub fn invariance(&self) -> kernel_core::properties::Invariance {
        if !self.suffix.is_empty() || self.contract().is_err() {
            return kernel_core::properties::Invariance::NONE;
        }
        self.select.invariance()
    }

    pub fn profile(&self) -> Profile {
        self.select.profile
    }

    pub fn symbol(&self) -> String {
        format!(
            "chop_pooled_indexer_{}_merge_tree{}_v1",
            self.select.name, self.suffix
        )
    }

    fn per_rank(&self) -> bool {
        matches!(self.select.kind, Kind::Select { ranks: Some(_) })
    }

    /// The selection's contract with the merge's knobs.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let select = self.select.contract()?;
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: select.key.query,
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: self.select.geometry.max_kept() as u16,
                    tile_k: 16,
                    stages: 1,
                    swizzle: 0,
                    warps: WARPS as u8,
                    split_k_capacity: MAX_SPLITS as u8,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol(),
        };
        spec.validate()?;
        Ok(spec)
    }

    /// One CTA per prepared row, as the selection's merge plan (with its
    /// workspace); rows past the live count return after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let merge = self
            .select
            .merge_plan(prepared_rows)?
            .expect("a selection has a merge");
        let plan = LaunchPlan {
            block: [32 * WARPS, 1, 1],
            ..merge
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The selection merge's ABI.
    pub fn params(&self) -> Vec<EntryParam> {
        self.select.merge_params()
    }

    fn body(&self) -> String {
        format!(
            "chop::pooled_indexer::select_merge_tree<{},{WARPS}>(workspace,positions,live_rows,\
             output,top_k,splits);",
            self.select.geometry.select_traits(self.per_rank())
        )
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            32 * WARPS,
            1,
        ))
    }
}

/// The tree merges of every selection among `entries`, in order.
pub fn entries(entries: &[Select]) -> Vec<Entry> {
    entries
        .iter()
        .filter_map(|select| Entry::of(select).ok())
        .collect()
}

/// The cubin name of `profile`'s tree merges, with a mutant's suffix.
pub fn cubin_name(profile: Profile, suffix: &str) -> String {
    format!("pooled_indexer_merge_tree{suffix}_{}_sm120", profile.tag())
}

/// The cubin of `profile`'s merges among `entries`, compiled with each of
/// `defines` set to 1.
pub fn cubin_spec(
    profile: Profile,
    entries: &[Entry],
    suffix: &str,
    defines: &[&str],
) -> Result<CubinSpec, Refusal> {
    let specs = entries
        .iter()
        .filter(|entry| entry.profile() == profile)
        .map(Entry::entry_spec)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CubinSpec {
        name: cubin_name(profile, suffix),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: defines
            .iter()
            .map(|define| ((*define).to_owned(), "1".to_owned()))
            .collect(),
        entries: specs,
    })
}

/// The lab's cubins of `profile`'s merges among `entries`: the served form,
/// then each mutant in a cubin of its own with its suffix on every symbol.
pub fn lab_cubin_specs(profile: Profile, entries: &[Entry]) -> Result<Vec<CubinSpec>, Refusal> {
    let mut specs = vec![cubin_spec(profile, entries, "", &[])?];
    for (suffix, define) in MUTANTS {
        let mutants = entries
            .iter()
            .map(|entry| Entry {
                suffix: suffix.into(),
                ..entry.clone()
            })
            .collect::<Vec<_>>();
        specs.push(cubin_spec(profile, &mutants, suffix, &[define])?);
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProfileNames;
    use crate::pooled_indexer::{Geometry, entries as select_entries};
    use kernel_core::contract::DType;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "merge-tree-test",
        tag: "merge_tree_test",
    });

    fn selections() -> Vec<Select> {
        select_entries(
            PROFILE,
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
            },
        )
    }

    #[test]
    fn merges_take_each_selection_merges_plan_and_abi() {
        let selections = selections();
        let merges = entries(&selections);
        // The global selection and its per-rank forms at 2 and 4 ranks.
        assert_eq!(merges.len(), 3);
        for merge in &merges {
            merge.contract().unwrap();
            for rows in [1, 4, 32, 192, 2048] {
                let plan = merge.plan(rows).unwrap();
                let reference = merge.select.merge_plan(rows).unwrap().unwrap();
                assert_eq!(plan.grid, reference.grid);
                assert_eq!(plan.workspace_bytes, reference.workspace_bytes);
                assert_eq!(plan.block, [512, 1, 1]);
                assert!(plan.pdl);
            }
            assert!(merge.plan(2049).is_err());
            assert_eq!(merge.params(), merge.select.merge_params());
        }
        assert_eq!(
            merges[0].symbol(),
            "chop_pooled_indexer_merge_tree_test_select_merge_tree_v1"
        );
        assert!(
            merges[0]
                .body()
                .contains("Select<32,128,4,64,2048,true,false>,16>")
        );
        assert!(merges[1].body().contains("2048,true,true>,16>"));
        assert!(
            selections
                .iter()
                .filter(|entry| !matches!(entry.kind, Kind::Select { .. }))
                .all(|entry| Entry::of(entry).is_err())
        );
    }

    #[test]
    fn lab_cubins_hold_the_served_merges_and_each_mutant() {
        let merges = entries(&selections());
        let specs = lab_cubin_specs(PROFILE, &merges).unwrap();
        assert_eq!(specs.len(), 1 + MUTANTS.len());
        assert_eq!(
            specs[0].name,
            "pooled_indexer_merge_tree_merge_tree_test_sm120"
        );
        assert!(specs[0].defines.is_empty());
        for (spec, (suffix, define)) in specs[1..].iter().zip(MUTANTS) {
            assert_eq!(spec.name, cubin_name(PROFILE, suffix));
            assert_eq!(spec.defines, [(define.to_owned(), "1".to_owned())]);
            assert!(
                spec.entries
                    .iter()
                    .all(|entry| entry.symbol.ends_with(&format!("{suffix}_v1")))
            );
        }
    }
}
