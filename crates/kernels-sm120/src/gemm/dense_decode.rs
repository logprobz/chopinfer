//! Dense decode folds with separate schedule and arithmetic identities.
use super::mxfp8_fold;
use kernel_build::{CubinSpec, EntryParam, EntrySpec, tuning};
use kernel_core::contract::{DType, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Finish {
    Ticket,
    Merge,
}

/// `dense_decode::Knobs` in the CUDA template. Finish is 0 (ticket) or 1
/// (merge) in a tuning row, whose values are counts or flags.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Knobs {
    pub tile_n: u32,
    pub row_groups: u32,
    pub stages: u32,
    pub weight_prefetch: bool,
    pub finish: Finish,
}

/// Arithmetic is an entry identity, never a tuning knob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Arithmetic {
    pub activation: Activation,
    pub scale: Scale,
    pub chain: Chain,
    pub reduction: Reduction,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    E4m3CeilUe8m0,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scale {
    Ue8m0PerK32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Chain {
    FourK32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduction {
    AscendingF32,
}

impl Knobs {
    pub const SPLIT: Self = Self {
        tile_n: 16,
        row_groups: 1,
        stages: 2,
        weight_prefetch: false,
        finish: Finish::Merge,
    };

    pub fn validate(self) -> Result<(), Refusal> {
        let valid = match self.finish {
            Finish::Merge => self == Self::SPLIT,
            Finish::Ticket => {
                [32, 64].contains(&self.tile_n)
                    && [1, 2].contains(&self.row_groups)
                    && self.stages == 3
                    && (!self.weight_prefetch || self.row_groups == 1)
            }
        };
        if !valid {
            return Err(Refusal("unsupported dense decode schedule knobs"));
        }
        Ok(())
    }

    pub fn arithmetic(self) -> Arithmetic {
        Arithmetic {
            activation: Activation::E4m3CeilUe8m0,
            scale: Scale::Ue8m0PerK32,
            chain: Chain::FourK32,
            reduction: Reduction::AscendingF32,
        }
    }

    pub fn row_knobs(self) -> tuning::Knobs {
        use tuning::Knob::{Count, Flag};
        [
            ("tile_n".into(), Count(self.tile_n)),
            ("row_groups".into(), Count(self.row_groups)),
            ("stages".into(), Count(self.stages)),
            ("weight_prefetch".into(), Flag(self.weight_prefetch)),
            (
                "finish".into(),
                Count(u32::from(self.finish == Finish::Merge)),
            ),
        ]
        .into()
    }

    pub fn from_row(row: &tuning::Row) -> Result<Self, String> {
        if row.schedule != "dense_decode" {
            return Err(format!("{}: expected dense_decode schedule", row.key));
        }
        let count = |name: &str| {
            row.knobs
                .get(name)
                .and_then(|k| k.count())
                .ok_or_else(|| format!("{}: knob {name} needs a count", row.key))
        };
        let parsed = Self {
            tile_n: count("tile_n")?,
            row_groups: count("row_groups")?,
            stages: count("stages")?,
            weight_prefetch: row
                .knobs
                .get("weight_prefetch")
                .and_then(|k| k.flag())
                .ok_or_else(|| format!("{}: weight_prefetch needs a flag", row.key))?,
            finish: match count("finish")? {
                0 => Finish::Ticket,
                1 => Finish::Merge,
                _ => return Err(format!("{}: finish must be 0 or 1", row.key)),
            },
        };
        if row.knobs.len() != parsed.row_knobs().len() {
            return Err(format!("{}: unknown dense decode knobs", row.key));
        }
        parsed.validate().map_err(|e| format!("{}: {e}", row.key))?;
        Ok(parsed)
    }

    pub fn tag(self) -> String {
        format!(
            "n{}r{}s{}w{}f{}",
            self.tile_n,
            self.row_groups,
            self.stages,
            u8::from(self.weight_prefetch),
            u8::from(self.finish == Finish::Merge)
        )
    }

    fn cxx(self) -> String {
        format!(
            "chop::gemm::dense_decode::Knobs{{.tile_n={},.row_groups={},.stages={},.weight_prefetch={},.finish=chop::gemm::dense_decode::Finish::{}}}",
            self.tile_n,
            self.row_groups,
            self.stages,
            self.weight_prefetch,
            match self.finish {
                Finish::Ticket => "ticket",
                Finish::Merge => "merge",
            }
        )
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: Mxfp8FoldGemmQuery,
    pub symbol: String,
    pub paired: bool,
    pub knobs: Knobs,
}

pub fn supported_reuse_rows(rows: u32) -> bool {
    rows == 6 || (8..=32).contains(&rows)
}

impl Entry {
    pub fn new(
        query: Mxfp8FoldGemmQuery,
        stem: &str,
        paired: bool,
        knobs: Knobs,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            query,
            symbol: format!("{stem}_e4m3_ue8m0_k32x4_f32_{}_v1", knobs.tag()),
            paired,
            knobs,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn legacy(&self) -> Result<mxfp8_fold::Entry, Refusal> {
        mxfp8_fold::Entry::new(self.query, &self.symbol)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.knobs.validate()?;
        self.legacy()?;
        let maximum = if self.knobs.finish == Finish::Merge {
            32
        } else {
            64
        };
        if self.query.m_capacity > maximum {
            return Err(Refusal("dense decode capacity exceeds its schedule"));
        }
        if self.knobs.row_groups == 2 && self.query.output != DType::F32 {
            return Err(Refusal("dense decode row reuse requires F32 output"));
        }
        self.workspace(self.query.m_capacity)?;
        Ok(())
    }

    pub fn arithmetic(&self) -> Arithmetic {
        self.knobs.arithmetic()
    }

    pub fn tiles(&self, rows: u32) -> u64 {
        u64::from(self.query.n.div_ceil(self.knobs.tile_n))
            * u64::from(rows.max(1).div_ceil(16 * self.knobs.row_groups))
            * u64::from(self.query.groups)
            * if self.paired { 2 } else { 1 }
    }

    pub fn header(&self, rows: u32) -> Result<u64, Refusal> {
        if self.knobs.finish == Finish::Merge {
            return Ok(256);
        }
        self.tiles(rows)
            .checked_mul(4)
            .map(|n| 256 + n.div_ceil(256) * 256)
            .filter(|&n| n <= u64::from(u32::MAX))
            .ok_or(Refusal("dense decode header exceeds device index bounds"))
    }

    /// The ticket partition is a runtime scalar and never changes fold order.
    pub fn folds_per_cta(&self, rows: u32) -> u32 {
        if self.knobs.finish == Finish::Merge {
            return 2;
        }
        let wanted = (u64::from(super::SM_COUNT) * 2)
            .div_ceil(self.tiles(rows))
            .max(1);
        let available = (u64::from(self.query.k.div_ceil(128)) / wanted).clamp(2, 16) as u32;
        1 << available.ilog2()
    }

    pub fn workspace(&self, rows: u32) -> Result<u64, Refusal> {
        let partials = [
            u64::from(rows),
            u64::from(self.query.groups),
            u64::from(self.query.n),
            u64::from(self.query.k.div_ceil(128)),
            if self.paired { 8 } else { 4 },
        ]
        .into_iter()
        .try_fold(1_u64, |n, v| n.checked_mul(v))
        .ok_or(Refusal("dense decode partials overflow"))?;
        self.header(rows)?
            .checked_add(partials)
            .filter(|&n| n <= kernel_core::precision::MAX_TENSOR_BYTES)
            .ok_or(Refusal(
                "dense decode workspace exceeds the supported bound",
            ))
    }

    pub fn plans(&self, rows: u32, folds: u32) -> Result<Vec<LaunchPlan>, Refusal> {
        self.validate()?;
        if rows > self.query.m_capacity
            || (self.knobs.finish == Finish::Merge && folds != 2)
            || (self.knobs.row_groups == 2
                && (!supported_reuse_rows(rows) || ![2, 4, 8].contains(&folds)))
            || ![2, 4, 8, 16].contains(&folds)
        {
            return Err(Refusal("unsupported dense decode rows or fold partition"));
        }
        let q = self.query;
        let banks = if self.paired { 2 } else { 1 };
        let z = q
            .groups
            .checked_mul(q.k.div_ceil(128).div_ceil(folds))
            .and_then(|v| v.checked_mul(banks))
            .ok_or(Refusal("dense decode grid overflows"))?;
        let plan = |grid, threads| LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        let mut plans = vec![
            plan([rows.max(1), 1, 1], 128),
            plan(
                [
                    q.n.div_ceil(self.knobs.tile_n),
                    rows.max(1).div_ceil(16 * self.knobs.row_groups),
                    z,
                ],
                self.knobs.tile_n * 4,
            ),
        ];
        if self.knobs.finish == Finish::Merge {
            let outputs = u64::from(rows) * u64::from(q.groups) * u64::from(q.n) * u64::from(banks);
            let grid = u32::try_from(outputs.max(1).div_ceil(128))
                .map_err(|_| Refusal("dense decode merge grid overflows"))?;
            plans.push(plan([grid, 1, 1], 128));
        }
        for plan in &plans {
            crate::backend::check_plan(plan)?;
        }
        Ok(plans)
    }

    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.specs_with_weights(false)
    }

    /// Admission and merge keep their ABI; only the reader selects packed codes.
    pub fn packed_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        super::mxfp8_fold_packed::check(self.query.groups, self.query.n, self.query.k)?;
        self.specs_with_weights(true)
    }

    fn specs_with_weights(&self, packed: bool) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let q = self.query;
        let ns = "chop::gemm::dense_decode";
        let args = format!(
            "{},{},{},{},{},{},{},{},{},{ns}::E4m3CeilUe8m0,{ns}::Ue8m0PerK32,{ns}::FourK32,{ns}::AscendingF32",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            self.knobs.cxx(),
            q.output == DType::F32,
            self.paired,
            packed
        );
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let rows = || EntryParam::U32 {
            name: "rows".into(),
        };
        let offset = || EntryParam::U64 {
            name: "status_row_offset".into(),
        };
        let output = if q.output == DType::F32 {
            "float"
        } else {
            "uint16_t"
        };
        let split = self.knobs.finish == Finish::Merge;
        let work = if split { "partials" } else { "workspace" };
        let admission = vec![
            ptr("activations", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr(work, if split { "uint32_t" } else { "uint8_t" }),
            ptr("live_rows", "const uint32_t"),
            ptr("status", "uint32_t"),
            offset(),
            rows(),
        ];
        let admit = if split { "admit" } else { "initialize" };
        let mut result = vec![EntrySpec::new(
            format!("{}_admit_v1", self.symbol),
            admission,
            format!(
                "{ns}::{admit}<{args}>(activations,activation_scales,{work},live_rows,status,status_row_offset,rows);"
            ),
            128,
            1,
        )];
        let mut compute = vec![
            ptr("activations", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("weights", "const uint8_t"),
            ptr("weight_scales", "const uint8_t"),
            ptr("weights2", "const uint8_t"),
            ptr("weight_scales2", "const uint8_t"),
        ];
        let (name, values) = if split {
            compute.extend([
                ptr("partials", "float"),
                ptr("live_rows", "const uint32_t"),
                ptr("status", "const uint32_t"),
                offset(),
                rows(),
            ]);
            (
                "partials",
                "activations,activation_scales,weights,weight_scales,weights2,weight_scales2,partials,live_rows,status,status_row_offset,rows",
            )
        } else {
            compute.extend([
                ptr("output", output),
                ptr("output2", output),
                ptr("workspace", "uint8_t"),
                ptr("live_rows", "const uint32_t"),
                rows(),
                EntryParam::U32 {
                    name: "folds_per_cta".into(),
                },
            ]);
            (
                "run",
                "activations,activation_scales,weights,weight_scales,weights2,weight_scales2,output,output2,workspace,live_rows,rows,folds_per_cta",
            )
        };
        let symbol = if split {
            format!("{}_partials_v1", self.symbol)
        } else {
            self.symbol.clone()
        };
        result.push(EntrySpec::new(
            if packed {
                super::mxfp8_fold_packed::symbol(&symbol)
            } else {
                symbol
            },
            compute,
            format!("{ns}::{name}<{args}>({values});"),
            self.knobs.tile_n * 4,
            if self.knobs.finish == Finish::Ticket
                && self.knobs.row_groups == 1
                && self.knobs.tile_n == 32
            {
                4
            } else {
                1
            },
        ));
        if split {
            result.push(EntrySpec::new(self.symbol.clone(), vec![ptr("partials", "const float"), ptr("output", output), ptr("output2", output), ptr("live_rows", "const uint32_t"), ptr("status", "const uint32_t"), offset(), rows()], format!("{ns}::merge<{args}>(partials,output,output2,live_rows,status,status_row_offset,rows);"), 128, 1));
        }
        Ok(result)
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry], packed: bool) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("dense decode cubin requires entries"));
    }
    let specs = entries
        .iter()
        .map(|entry| {
            if packed {
                entry.packed_specs()
            } else {
                entry.specs()
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut symbols = std::collections::BTreeSet::new();
    if specs.iter().any(|spec| !symbols.insert(&spec.symbol)) {
        return Err(Refusal("duplicate dense decode symbol"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/dense_decode.cuh\"".into()],
        defines: vec![],
        entries: specs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{mxfp8_fold_decode, mxfp8_fold_reuse, mxfp8_fold_ticket};
    use kernel_core::precision::execution::BlockScaledExecution;

    fn query(n: u32, k: u32, groups: u32, output: DType) -> Mxfp8FoldGemmQuery {
        Mxfp8FoldGemmQuery {
            m_capacity: 32,
            n,
            k,
            groups,
            output,
            activation_row_stride: groups * (k + 32) + 32,
            activation_group_stride: k + 32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        }
    }

    fn schedules() -> Vec<Knobs> {
        let mut result = vec![Knobs::SPLIT];
        for tile_n in [32, 64] {
            for (row_groups, weight_prefetch) in [(1, false), (1, true), (2, false)] {
                result.push(Knobs {
                    tile_n,
                    row_groups,
                    stages: 3,
                    weight_prefetch,
                    finish: Finish::Ticket,
                });
            }
        }
        result
    }

    #[test]
    fn rows_require_exact_knobs_and_preserve_policy_identity() {
        let mut tags = std::collections::BTreeSet::new();
        for knobs in schedules() {
            let row = tuning::Row {
                key: tuning::Key {
                    op: "dense.decode".into(),
                    weights: "mxfp8".into(),
                    shape: [("n".into(), 512)].into(),
                    bucket: 32,
                    rows: None,
                },
                schedule: "dense_decode".into(),
                knobs: knobs.row_knobs(),
            };
            assert_eq!(Knobs::from_row(&row).unwrap(), knobs);
            assert_eq!(knobs.arithmetic(), Knobs::SPLIT.arithmetic());
            assert!(tags.insert(knobs.tag()));
            for name in row.knobs.keys() {
                let mut invalid = row.clone();
                invalid.knobs.remove(name);
                assert!(Knobs::from_row(&invalid).is_err());
                let mut invalid = row.clone();
                invalid.knobs.insert(
                    name.clone(),
                    match row.knobs[name] {
                        tuning::Knob::Flag(_) => tuning::Knob::Count(1),
                        tuning::Knob::Count(_) => tuning::Knob::Flag(true),
                    },
                );
                assert!(Knobs::from_row(&invalid).is_err());
            }
            for extra in ["activation", "scale", "chain", "reduction", "typo"] {
                let mut invalid = row.clone();
                invalid.knobs.insert(extra.into(), tuning::Knob::Count(1));
                assert!(Knobs::from_row(&invalid).is_err());
            }
        }
        for knobs in [
            Knobs {
                stages: 3,
                ..Knobs::SPLIT
            },
            Knobs {
                weight_prefetch: true,
                ..Knobs::SPLIT
            },
            Knobs {
                finish: Finish::Ticket,
                ..Knobs::SPLIT
            },
            Knobs {
                tile_n: 32,
                row_groups: 2,
                stages: 3,
                weight_prefetch: true,
                finish: Finish::Ticket,
            },
        ] {
            assert!(knobs.validate().is_err());
        }
    }

    fn same_specs(old: &[EntrySpec], new: &[EntrySpec]) {
        assert_eq!(old.len(), new.len());
        for (old, new) in old.iter().zip(new) {
            assert_ne!(old.symbol, new.symbol);
            assert_eq!(old.params, new.params);
            assert_eq!(old.max_threads, new.max_threads);
            assert_eq!(old.min_blocks, new.min_blocks);
            assert_eq!(old.allowed_local_bytes, new.allowed_local_bytes);
            assert!(new.body.contains("Knobs{.tile_n="));
            for policy in ["E4m3CeilUe8m0", "Ue8m0PerK32", "FourK32", "AscendingF32"] {
                assert!(new.body.contains(policy));
            }
        }
    }

    #[test]
    fn template_preserves_reference_abis_workspaces_and_every_row_grid() {
        for (n, k, groups) in [
            (40, 576, 2),
            (72, 1280, 1),
            (2304, 576, 2),
            (1152, 2048, 1),
            (5120, 2048, 1),
            (8192, 1280, 1),
        ] {
            for output in [DType::F32, DType::Bf16] {
                for paired in [false, true] {
                    let q = query(n, k, groups, output);
                    let split = mxfp8_fold_decode::Entry::new(q, "old_split", paired).unwrap();
                    let entry = Entry::new(q, "new_split", paired, Knobs::SPLIT).unwrap();
                    same_specs(&split.specs(), &entry.specs().unwrap());
                    let mut old = split.specs();
                    old[1] = split.packed_spec().unwrap();
                    same_specs(&old, &entry.packed_specs().unwrap());
                    for rows in 0..=32 {
                        assert_eq!(
                            split.workspace(rows).unwrap(),
                            entry.workspace(rows).unwrap()
                        );
                        assert_eq!(
                            split.plans(rows).unwrap().as_slice(),
                            entry.plans(rows, 2).unwrap()
                        );
                    }
                    for prefetch in [false, true] {
                        let mut old =
                            mxfp8_fold_ticket::Entry::new(q, "old_ticket", paired).unwrap();
                        if prefetch {
                            old = old.with_weight_prefetch();
                        }
                        let knobs = Knobs {
                            tile_n: old.tile_n(),
                            row_groups: 1,
                            stages: 3,
                            weight_prefetch: prefetch,
                            finish: Finish::Ticket,
                        };
                        let entry = Entry::new(q, "new_ticket", paired, knobs).unwrap();
                        same_specs(&old.specs(), &entry.specs().unwrap());
                        let mut packed = old.specs();
                        packed[1] = old.packed_spec().unwrap();
                        same_specs(&packed, &entry.packed_specs().unwrap());
                        for rows in 0..=32 {
                            assert_eq!(old.folds_per_cta(rows), entry.folds_per_cta(rows));
                            assert_eq!(
                                old.workspace(rows).unwrap(),
                                entry.workspace(rows).unwrap()
                            );
                            assert_eq!(
                                old.plans(rows).unwrap().as_slice(),
                                entry.plans(rows, entry.folds_per_cta(rows)).unwrap()
                            );
                        }
                        if output == DType::F32 && !prefetch {
                            for tile_n in [32, 64] {
                                let old = mxfp8_fold_reuse::Entry {
                                    ticket: old.clone(),
                                    tile_n,
                                };
                                let entry = Entry::new(
                                    q,
                                    "new_reuse",
                                    paired,
                                    Knobs {
                                        tile_n,
                                        row_groups: 2,
                                        ..knobs
                                    },
                                )
                                .unwrap();
                                same_specs(&old.specs(), &entry.specs().unwrap());
                                for rows in (1..=32).filter(|&r| supported_reuse_rows(r)) {
                                    assert_eq!(
                                        old.workspace(rows).unwrap(),
                                        entry.workspace(rows).unwrap()
                                    );
                                    for folds in [2, 4, 8] {
                                        assert_eq!(
                                            old.plans(rows, folds).unwrap().as_slice(),
                                            entry.plans(rows, folds).unwrap()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn template_refuses_invalid_rows_partitions_and_duplicates() {
        let q = query(512, 576, 1, DType::F32);
        let entry = Entry::new(q, "split", false, Knobs::SPLIT).unwrap();
        assert!(entry.plans(33, 2).is_err());
        for folds in [0, 1, 3, 4, 8, 16, u32::MAX] {
            assert!(entry.plans(1, folds).is_err());
        }
        assert!(cubin_spec("empty", &[], true).is_err());
        assert!(cubin_spec("duplicate", &[entry.clone(), entry.clone()], true).is_err());
        let reuse = Entry::new(
            q,
            "reuse",
            false,
            Knobs {
                tile_n: 32,
                row_groups: 2,
                stages: 3,
                weight_prefetch: false,
                finish: Finish::Ticket,
            },
        )
        .unwrap();
        for rows in [0, 1, 2, 3, 4, 5, 7, 33] {
            assert!(reuse.plans(rows, 4).is_err());
        }
        assert!(
            Entry::new(
                Mxfp8FoldGemmQuery {
                    output: DType::Bf16,
                    ..q
                },
                "reuse",
                false,
                reuse.knobs
            )
            .is_err()
        );
        assert_eq!(
            cubin_spec("split", &[entry], true).unwrap().entries.len(),
            3
        );
    }
}
