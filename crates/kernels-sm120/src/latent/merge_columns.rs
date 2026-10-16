//! Column-parallel split merge (`cuda/latent/merge_columns.cuh`, README
//! "Column merge"): a decode or verify attention entry's split merge
//! outputs bit for bit, from the same workspace and launch ABI, with each
//! thread folding one column of one row and head after loading every
//! split's value of it. Its grid adds the latent width in 128-column CTAs.
//! In its own cubin, `latent_merge_columns_{tag}_sm120`.

use std::path::PathBuf;

use super::{Entry as Attention, MAX_SPLITS};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{InstantiationSpec, LaunchPlan, Refusal, Target, Validate};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "latent/merge_columns.cuh";
/// Columns and threads of one CTA; `merge_columns` in merge_columns.cuh.
pub const COLUMNS: u32 = 128;
/// Lab-only mutant: the last split left out of every fold.
pub const SKIP_LAST_DEFINE: &str = "CHOP_LATENT_MERGE_COLUMNS_TEST_SKIP_LAST";
/// Lab-only mutant: the numerators folded in descending split order.
pub const DESCENDING_DEFINE: &str = "CHOP_LATENT_MERGE_COLUMNS_TEST_DESCENDING";
/// Lab-only mutant: the dependency wait after the reads of the live count
/// and the partials.
pub const LATE_WAIT_DEFINE: &str = "CHOP_LATENT_MERGE_COLUMNS_TEST_LATE_WAIT";
/// The lab's mutants: (symbol suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_skip_last", SKIP_LAST_DEFINE),
    ("_descending", DESCENDING_DEFINE),
    ("_late_wait", LATE_WAIT_DEFINE),
];

/// The column merge of one splitting (decode or verify) attention entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The attention entry whose split merge this one replaces.
    pub attention: Attention,
    /// Appended to the symbol (lab mutants); empty when served.
    pub suffix: String,
}

impl Entry {
    /// The form's declared exactness: its attention entry's split merge's,
    /// since it writes that merge's outputs bit for bit at every row count
    /// from the same workspace; nothing for a lab mutant or a form its
    /// kernel refuses.
    pub fn invariance(&self) -> kernel_core::properties::Invariance {
        if !self.suffix.is_empty() || self.contract().is_err() {
            return kernel_core::properties::Invariance::NONE;
        }
        self.attention.invariance()
    }

    /// The column merge of `attention`, which must split its rows.
    pub fn of(attention: &Attention) -> Result<Entry, Refusal> {
        if attention.merge_symbol().is_none()
            || attention.splits() == 1
            || attention.splits() > MAX_SPLITS
        {
            return Err(Refusal(
                "a column merge folds a splitting attention entry's partials",
            ));
        }
        let shape = attention.attention().expect("a merging entry attends");
        if !shape.latent.is_multiple_of(COLUMNS) {
            return Err(Refusal("column merge CTAs cover the latent row"));
        }
        Ok(Entry {
            attention: attention.clone(),
            suffix: String::new(),
        })
    }

    pub fn profile(&self) -> Profile {
        self.attention.profile
    }

    pub fn symbol(&self) -> String {
        format!(
            "chop_latent_{}_merge_columns{}_v1",
            self.attention.name, self.suffix
        )
    }

    /// The attention entry's contract under this symbol.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let base = self.attention.contract()?;
        let mut key = base.key;
        key.target = Target::Sm120f;
        let spec = InstantiationSpec {
            family: base.family,
            key,
            symbol: self.symbol(),
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Grid [rows, heads, latent / 128] of 128 threads over the split
    /// merge's workspace; rows past the live count return after the PDL
    /// wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if prepared_rows == 0 || prepared_rows > self.attention.capacity() {
            return Err(Refusal("latent rows exceed the prepared capacity"));
        }
        let merge = self
            .attention
            .merge_plan(prepared_rows)?
            .expect("a splitting entry has a merge");
        let shape = self.attention.attention().expect("a merging entry attends");
        let plan = LaunchPlan {
            grid: [prepared_rows, shape.heads, shape.latent / COLUMNS],
            block: [COLUMNS, 1, 1],
            ..merge
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The split merge's ABI.
    pub fn params(&self) -> Vec<EntryParam> {
        self.attention.merge_params()
    }

    fn body(&self) -> String {
        let shape = self.attention.attention().expect("a merging entry attends");
        format!(
            "chop::latent::merge_columns_of<{}>(workspace,live_rows,{},splits);",
            self.attention
                .attention_traits()
                .expect("a merging entry attends"),
            Attention::output_arguments(shape.partial)
        )
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            COLUMNS,
            1,
        ))
    }
}

/// The column merges of the splitting attention entries among `entries`,
/// in order.
pub fn entries(entries: &[Attention]) -> Vec<Entry> {
    entries
        .iter()
        .filter_map(|attention| Entry::of(attention).ok())
        .collect()
}

/// The cubin name of `profile`'s column merges, with a mutant's suffix.
pub fn cubin_name(profile: Profile, suffix: &str) -> String {
    format!("latent_merge_columns{suffix}_{}_sm120", profile.tag())
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
    use crate::latent::{Geometry, entries as latent_entries};

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "latent-merge-columns-test",
        tag: "latent_merge_columns_test",
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

    #[test]
    fn merges_take_the_splitting_entries_workspace_and_abi() {
        let attention = latent_entries(PROFILE, geometry());
        let merges = entries(&attention);
        // Decode and verify over BF16 and FP8 pages, normalized and partial.
        assert_eq!(merges.len(), 8);
        assert!(merges.iter().any(|merge| {
            merge.symbol()
                == "chop_latent_latent_merge_columns_test_attn_fp8_verify_merge_columns_v1"
        }));
        for merge in &merges {
            merge.contract().unwrap();
            for rows in [1, 4, 32, 256] {
                let plan = merge.plan(rows).unwrap();
                let reference = merge.attention.merge_plan(rows).unwrap().unwrap();
                assert_eq!(plan.grid, [rows, 16, 4]);
                assert_eq!(plan.block, [128, 1, 1]);
                assert_eq!(plan.workspace_bytes, reference.workspace_bytes);
                assert_eq!(plan.dynamic_shared_bytes, 0);
                assert!(plan.pdl);
            }
            assert!(merge.plan(257).is_err());
            assert_eq!(merge.params(), merge.attention.merge_params());
        }
        assert!(
            merges[0]
                .body()
                .contains("merge_columns_of<chop::latent::Attention<16,512,64,2051,")
        );
        for other in attention
            .iter()
            .filter(|entry| entry.merge_symbol().is_none())
        {
            assert!(Entry::of(other).is_err(), "{}", other.name);
        }
    }

    #[test]
    fn lab_cubins_hold_the_served_merges_and_each_mutant() {
        let merges = entries(&latent_entries(PROFILE, geometry()));
        let specs = lab_cubin_specs(PROFILE, &merges).unwrap();
        assert_eq!(specs.len(), 1 + MUTANTS.len());
        assert_eq!(
            specs[0].name,
            "latent_merge_columns_latent_merge_columns_test_sm120"
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
