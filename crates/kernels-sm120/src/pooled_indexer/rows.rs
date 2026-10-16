//! One-split selection (`cuda/pooled_indexer/select_rows.cuh`, README
//! "One-split selection"): a selection entry's key lists (`Global`) or
//! per-rank pairs at one split per row, bit for bit, from one kernel that
//! scores every pool of its rows and writes their output itself. No split
//! workspace or merge launch exists, the page lookups run a tile ahead of
//! the copies, and each warp sums its own pools' head chains, so one
//! barrier per tile remains.
//!
//! Two forms, by rows per CTA: one row in a 256-thread CTA that shares an
//! SM with a second, or two rows of one sequence sharing each key tile in a
//! 512-thread CTA with three key stages. In its own cubin,
//! `pooled_indexer_rows_{tag}_sm120`.

use std::path::PathBuf;

use super::{Entry as Select, Kind, SELECT_THREADS, TILE_POOLS};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{InstantiationSpec, LaunchPlan, Refusal, Target, Validate};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "pooled_indexer/select_rows.cuh";
/// SM120's shared memory per SM, and the 1 KiB the driver reserves per
/// resident CTA.
const SM_SHARED: u32 = 102_400;
const CTA_RESERVED: u32 = 1_024;
/// Lab-only mutant: each tile's copies read the page table entries of the
/// tile before it.
pub const STALE_PAGE_DEFINE: &str = "CHOP_POOLED_ROWS_TEST_STALE_PAGE";
/// Lab-only mutant: the dependency wait after the query and head weight
/// loads.
pub const LATE_WAIT_DEFINE: &str = "CHOP_POOLED_ROWS_TEST_LATE_WAIT";
/// Lab-only mutant: a `Global` row's final cut keeps one key more than the
/// best `top_k / group`.
pub const KEEP_EXTRA_DEFINE: &str = "CHOP_POOLED_ROWS_TEST_KEEP_EXTRA";
/// The lab's mutants: (symbol suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_stale_page", STALE_PAGE_DEFINE),
    ("_late_wait", LATE_WAIT_DEFINE),
    ("_keep_extra", KEEP_EXTRA_DEFINE),
];
/// Lab-only: thread 0's cycles per phase in place of each `Global` row's
/// first 20 list entries (`CHOP_ROWS_CLOCK` in select_rows.cuh).
pub const CLOCKS_DEFINE: &str = "CHOP_POOLED_ROWS_TEST_CLOCKS";
/// The suffix of the lab's phase-clock cubin and symbols.
pub const CLOCKS_SUFFIX: &str = "_clocks";
/// Rows per CTA of the two forms.
pub const FORMS: [u32; 2] = [1, 2];

/// The one-split form of a selection entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The selection whose outputs at one split this one writes.
    pub select: Select,
    /// Rows per CTA, 1 or 2.
    pub rows: u32,
    /// Appended to the symbol (lab mutants); empty when served.
    pub suffix: String,
}

impl Entry {
    /// The `rows`-row form of `select`, a selection entry of at most 32
    /// heads.
    pub fn of(select: &Select, rows: u32) -> Result<Entry, Refusal> {
        let entry = Entry {
            select: select.clone(),
            rows,
            suffix: String::new(),
        };
        entry.check()?;
        Ok(entry)
    }

    fn check(&self) -> Result<(), Refusal> {
        if !matches!(self.select.kind, Kind::Select { .. }) {
            return Err(Refusal("one-split selection takes a selection entry"));
        }
        if !FORMS.contains(&self.rows) {
            return Err(Refusal("one-split selection takes one or two rows per CTA"));
        }
        // The selection's own geometry refusal first: the checks below
        // divide by its group.
        self.select.geometry.validate()?;
        if self.select.geometry.heads > 32 {
            return Err(Refusal(
                "one-split selection holds at most 32 heads' query fragments",
            ));
        }
        // A trim spreads the row's buffer of twice its kept pools evenly
        // over the row's threads (`RowSelect::capacity`).
        if 2 * self.select.geometry.max_kept() < SELECT_THREADS {
            return Err(Refusal(
                "one-split selection keeps at least 128 pools per row",
            ));
        }
        Ok(())
    }

    /// The form's declared exactness: its selection's (the split kernel and
    /// merge's outputs at one split, bit for bit); nothing for a lab mutant
    /// or a form its kernel refuses.
    pub fn invariance(&self) -> kernel_core::properties::Invariance {
        if !self.suffix.is_empty() || self.contract().is_err() {
            return kernel_core::properties::Invariance::NONE;
        }
        self.select.invariance()
    }

    pub fn profile(&self) -> Profile {
        self.select.profile
    }

    fn per_rank(&self) -> bool {
        matches!(self.select.kind, Kind::Select { ranks: Some(_) })
    }

    pub fn symbol(&self) -> String {
        format!(
            "chop_pooled_indexer_{}_rows{}{}_v1",
            self.select.name, self.rows, self.suffix
        )
    }

    /// Visible pools a one-row CTA writes through its bitmap, a bit per
    /// pool in the key stages (`RowSelect::bitmap_pools`); a row with more
    /// sorts its list instead, as every two-row CTA does (0).
    pub fn bitmap_pools(&self) -> u32 {
        if self.rows == 1 {
            self.stages() * TILE_POOLS * self.select.geometry.head_dim * 2 * 8
        } else {
            0
        }
    }

    /// Key stages: two beside a second resident CTA, three for two rows.
    pub fn stages(&self) -> u32 {
        if self.rows == 1 { 2 } else { 3 }
    }

    /// CTAs of the entry that share an SM: the launch bounds' minimum.
    pub fn ctas_per_sm(&self) -> u32 {
        if self.rows == 1 { 2 } else { 1 }
    }

    /// Threads per CTA: the selection's 256 per row.
    pub fn threads(&self) -> u32 {
        SELECT_THREADS * self.rows
    }

    /// Dynamic shared bytes (`RowSelect::smem`): the key stages, each row's
    /// top-k buffer, each warp's ReLU dots (two rows only; one row keeps
    /// them in its key rows), each row's head weights, two tiles' offer
    /// counts per warp, and each row's radix select.
    pub fn dynamic_shared_bytes(&self) -> u32 {
        let g = self.select.geometry;
        let stage = TILE_POOLS * g.head_dim * 2;
        let topk = 2 * g.max_kept() * 8 + 16;
        let warps = self.threads() / 32;
        let relu = if self.rows == 1 {
            0
        } else {
            warps * 8 * (g.heads + 4) * 4
        };
        let radix = 256 * 4 + 8 * 4 + 16;
        self.stages() * stage + self.rows * (topk + g.heads * 4 + radix) + relu + 2 * warps * 4
    }

    /// The selection's contract under this symbol, its knobs marking the
    /// rows, warps and stages of the one split.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.check()?;
        let base = self.select.contract()?;
        let mut key = base.key;
        key.knobs.tile_m = (self.select.geometry.heads * self.rows) as u16;
        key.knobs.warps = (self.threads() / 32) as u8;
        key.knobs.stages = self.stages() as u8;
        key.knobs.split_k_capacity = 1;
        key.target = Target::Sm120f;
        let spec = InstantiationSpec {
            family: base.family,
            key,
            symbol: self.symbol(),
        };
        spec.validate()?;
        if (self.dynamic_shared_bytes() + CTA_RESERVED) * self.ctas_per_sm() > SM_SHARED {
            return Err(Refusal(
                "one-split selection does not fit its CTAs' shared memory per SM",
            ));
        }
        Ok(spec)
    }

    /// One CTA per `rows` rows of a bucket the selection plans; rows past
    /// the live count return after the PDL wait.
    pub fn plan(&self, prepared_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let select = self.select.plan(prepared_rows)?;
        let plan = LaunchPlan {
            grid: [prepared_rows.div_ceil(self.rows), 1, 1],
            block: [self.threads(), 1, 1],
            dynamic_shared_bytes: self.dynamic_shared_bytes(),
            workspace_bytes: 0,
            ..select
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The selection's ABI with its merge's output in place of the split
    /// workspace, and no split count.
    pub fn params(&self) -> Vec<EntryParam> {
        let output = if self.per_rank() {
            "uint32_t"
        } else {
            "int32_t"
        };
        self.select
            .params()
            .into_iter()
            .filter_map(|param| match &param {
                EntryParam::DevicePointer { name, .. } if name == "workspace" => {
                    Some(EntryParam::DevicePointer {
                        name: "output".into(),
                        cxx_type: output.into(),
                    })
                }
                EntryParam::U32 { name } if name == "splits" => None,
                _ => Some(param),
            })
            .collect()
    }

    fn body(&self) -> String {
        format!(
            "chop::pooled_indexer::select_rows<chop::pooled_indexer::RowSelect<{},{},{}>>(q,\
             head_weights,pooled_keys,page_tables,row_batch,positions,live_rows,output,max_pages,\
             page_stride_bytes,top_k,cp_rank,cp_size);",
            self.select.geometry.select_traits(self.per_rank()),
            self.rows,
            self.stages()
        )
    }

    pub fn entry_spec(&self) -> Result<EntrySpec, Refusal> {
        self.contract()?;
        Ok(EntrySpec::new(
            self.symbol(),
            self.params(),
            self.body(),
            self.threads(),
            self.ctas_per_sm(),
        ))
    }
}

/// The `rows`-row forms of the selection entries among `entries`, in
/// order.
pub fn entries(entries: &[Select], rows: u32) -> Vec<Entry> {
    entries
        .iter()
        .filter_map(|select| Entry::of(select, rows).ok())
        .collect()
}

/// The cubin name of `profile`'s one-split entries, with a mutant's suffix.
pub fn cubin_name(profile: Profile, suffix: &str) -> String {
    format!("pooled_indexer_rows{suffix}_{}_sm120", profile.tag())
}

/// The cubin of `profile`'s entries among `entries`, compiled with each of
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

/// The lab's cubins of `profile`'s entries among `entries`: the served
/// forms, then each mutant of the `Global` entries and their phase clocks,
/// each in a cubin of its own with its suffix on every symbol.
pub fn lab_cubin_specs(profile: Profile, entries: &[Entry]) -> Result<Vec<CubinSpec>, Refusal> {
    let mut specs = vec![cubin_spec(profile, entries, "", &[])?];
    for (suffix, define) in MUTANTS.into_iter().chain([(CLOCKS_SUFFIX, CLOCKS_DEFINE)]) {
        let mutants = entries
            .iter()
            .filter(|entry| !entry.per_rank())
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
        name: "pooled-rows-test",
        tag: "pooled_rows_test",
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
    fn forms_take_the_selection_entries_plans_and_abi() {
        let selects = select_entries(PROFILE, geometry());
        for rows in FORMS {
            let forms = entries(&selects, rows);
            // The global selection and both per-rank selections.
            assert_eq!(
                forms.iter().map(Entry::symbol).collect::<Vec<_>>(),
                [
                    format!("chop_pooled_indexer_pooled_rows_test_select_rows{rows}_v1"),
                    format!("chop_pooled_indexer_pooled_rows_test_select_per_rank2_rows{rows}_v1"),
                    format!("chop_pooled_indexer_pooled_rows_test_select_per_rank4_rows{rows}_v1"),
                ]
            );
            for entry in &forms {
                entry.contract().unwrap();
                for prepared in [1, 7, 376, 2048] {
                    let plan = entry.plan(prepared).unwrap();
                    assert_eq!(plan.grid, [prepared.div_ceil(rows), 1, 1]);
                    assert_eq!(plan.block, [256 * rows, 1, 1]);
                    assert_eq!(plan.workspace_bytes, 0);
                    assert!(plan.pdl);
                }
                assert!(entry.plan(2049).is_err());
                let names = entry
                    .params()
                    .iter()
                    .map(|param| match param {
                        EntryParam::DevicePointer { name, .. }
                        | EntryParam::U32 { name }
                        | EntryParam::U64 { name } => name.clone(),
                        other => panic!("{other:?}"),
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    names,
                    [
                        "q",
                        "head_weights",
                        "pooled_keys",
                        "page_tables",
                        "row_batch",
                        "positions",
                        "live_rows",
                        "output",
                        "max_pages",
                        "page_stride_bytes",
                        "top_k",
                        "cp_rank",
                        "cp_size",
                    ]
                );
                assert_eq!(entry.entry_spec().unwrap().min_blocks, 3 - rows);
            }
            assert!(
                forms[0].body().contains(
                    "RowSelect<chop::pooled_indexer::Select<32,128,4,64,2048,true,false>,"
                )
            );
        }
        // One row: two CTAs' shared memory per SM; two rows: three stages.
        assert_eq!(entries(&selects, 1)[0].dynamic_shared_bytes(), 42_240);
        assert_eq!(entries(&selects, 2)[0].dynamic_shared_bytes(), 86_528);
        // Key writes, merges and the compaction have no one-split form;
        // neither do 64 heads or three rows.
        for other in selects
            .iter()
            .filter(|entry| !matches!(entry.kind, Kind::Select { .. }))
        {
            assert!(Entry::of(other, 1).is_err(), "{}", other.name);
        }
        assert!(Entry::of(&selects[3], 3).is_err());
        let wide = select_entries(
            PROFILE,
            Geometry {
                heads: 64,
                ..geometry()
            },
        );
        assert!(Entry::of(&wide[3], 1).is_err());
        // A trim spreads twice the kept pools over 256 threads: 128 kept
        // pools at least; the selection itself admits 64.
        let kept = |max_top_k| {
            select_entries(
                PROFILE,
                Geometry {
                    max_top_k,
                    ..geometry()
                },
            )
        };
        assert!(Entry::of(&kept(256)[3], 1).is_err());
        assert!(Entry::of(&kept(256)[3], 2).is_err());
        Entry::of(&kept(512)[3], 1).unwrap();
        Entry::of(&kept(512)[3], 2).unwrap();
        // A selection of malformed geometry (pools of 0) is refused, not a
        // division by zero.
        let mut malformed = selects[3].clone();
        malformed.geometry.group = 0;
        for rows in FORMS {
            assert!(Entry::of(&malformed, rows).is_err());
        }
    }

    #[test]
    fn the_one_row_bitmap_covers_a_million_tokens_at_group_four() {
        // A bit per pool in the two key stages: 1M tokens of pools of 4. A
        // row past it, or past 512K tokens at pools of 2, sorts its list.
        for group in [2, 4] {
            let selects = select_entries(
                PROFILE,
                Geometry {
                    group,
                    ..geometry()
                },
            );
            assert_eq!(entries(&selects, 1)[0].bitmap_pools(), 262_144);
        }
    }

    #[test]
    fn lab_cubins_hold_the_forms_and_each_global_mutant() {
        let selects = select_entries(PROFILE, geometry());
        let forms = [entries(&selects, 1), entries(&selects, 2)].concat();
        let specs = lab_cubin_specs(PROFILE, &forms).unwrap();
        assert_eq!(specs.len(), 2 + MUTANTS.len());
        assert_eq!(specs[0].name, "pooled_indexer_rows_pooled_rows_test_sm120");
        assert_eq!(specs[0].entries.len(), 6);
        assert!(specs[0].defines.is_empty());
        let extra = MUTANTS.into_iter().chain([(CLOCKS_SUFFIX, CLOCKS_DEFINE)]);
        for (spec, (suffix, define)) in specs[1..].iter().zip(extra) {
            assert_eq!(spec.name, cubin_name(PROFILE, suffix));
            assert_eq!(spec.defines, [(define.to_owned(), "1".to_owned())]);
            assert_eq!(spec.entries.len(), 2);
            assert!(
                spec.entries
                    .iter()
                    .all(|entry| entry.symbol.ends_with(&format!("{suffix}_v1")))
            );
        }
    }
}
