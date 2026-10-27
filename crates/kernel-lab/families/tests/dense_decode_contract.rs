//! Frozen decode contracts against the shared template.
mod tests {
    use kernel_build::{EntrySpec, tuning};
    use kernel_core::precision::execution::BlockScaledExecution;
    use kernel_core::{contract::DType, mxfp8_gemm::Mxfp8FoldGemmQuery};
    use kernel_lab_families::dense_decode_reference::{
        mxfp8_fold_decode, mxfp8_fold_reuse, mxfp8_fold_ticket,
    };
    use kernels_sm120::gemm::dense_decode::*;

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
            let mut row = tuning::Row {
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
            let folds = (knobs.row_groups == 2).then_some(4);
            if let Some(folds) = folds {
                row.knobs.insert("folds".into(), tuning::Knob::Count(folds));
            }
            assert_eq!(Knobs::from_row(&row).unwrap(), (knobs, folds));
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

    #[test]
    fn runtime_partition_and_output_type_are_not_implicit() {
        let knobs = Knobs {
            row_groups: 2,
            ..Knobs::ticket(512)
        };
        let mut row = tuning::Row {
            key: tuning::Key {
                op: "reuse.fold.decode".into(),
                weights: "mxfp8".into(),
                shape: [("n".into(), 512)].into(),
                bucket: 32,
                rows: None,
            },
            schedule: "dense_decode".into(),
            knobs: knobs.row_knobs(),
        };
        assert!(Knobs::from_row(&row).is_err());
        for folds in [0, 1, 3, 16] {
            row.knobs.insert("folds".into(), tuning::Knob::Count(folds));
            assert!(Knobs::from_row(&row).is_err());
        }
        for folds in [2, 4, 8] {
            row.knobs.insert("folds".into(), tuning::Knob::Count(folds));
            assert_eq!(Knobs::from_row(&row).unwrap(), (knobs, Some(folds)));
        }
        row.knobs
            .insert("row_groups".into(), tuning::Knob::Count(1));
        assert!(Knobs::from_row(&row).is_err());
        let q = query(512, 576, 1, DType::F32);
        let a = Entry::new(q, "same_shape", false, Knobs::SPLIT).unwrap();
        let b = Entry::new(
            Mxfp8FoldGemmQuery {
                output: DType::Bf16,
                ..q
            },
            "same_shape",
            false,
            Knobs::SPLIT,
        )
        .unwrap();
        assert_ne!(a.symbol, b.symbol);
        cubin_spec("both_outputs", &[a, b], true).unwrap();
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
