use super::*;
#[test]
fn cpu_matrix() {
    let mut embedded = std::collections::BTreeSet::new();
    let specs = family::cubins();
    let symbols = specs
        .iter()
        .flat_map(|s| s.entries.iter().map(|e| e.symbol.clone()))
        .collect::<std::collections::BTreeSet<_>>();
    let mut selections = Vec::new();
    for case in cases() {
        let mut a = Arena {
            session: None,
            next: 0,
            buffers: BTreeMap::new(),
        };
        let live = a.bytes(&case.rows.to_le_bytes()).unwrap();
        let (input, scales, prefix) = source(&mut a, case, live).unwrap();
        let weights = [
            weights(&mut a, case.set.count as usize, 1152, 5120).unwrap(),
            weights(&mut a, case.set.count as usize, 5120, 576).unwrap(),
        ];
        let inputs = Inputs {
            input,
            scales,
            live,
            weights,
            logits: a
                .filled(u64::from(case.rows) * u64::from(case.set.count) * 4, 0)
                .unwrap(),
        };
        let baseline = arm(&mut a, case, inputs).unwrap();
        for (kind, part) in [
            (GatedMlpProjection::GateUp, "gateup"),
            (GatedMlpProjection::Down, "down"),
        ] {
            let ops = component(&baseline, part);
            let grouped =
                registration::prefill_selected(&FLASH, 4, case.set.count, kind, case.rows);
            assert_eq!(
                ops.len(),
                if grouped { 4 } else { 1 },
                "{} {part}",
                case.label()
            );
            assert_eq!(ops.iter().any(|o| o.symbol.ends_with("_pack_v1")), grouped);
            selections.push(serde_json::json!({"case":case.label(),"part":part,"grouped":grouped,"symbols":ops.iter().map(|o|&o.symbol).collect::<Vec<_>>()}));
        }
        for op in prefix.iter().chain(component(&baseline, "full").iter()) {
            let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
                .expect("embedded serving symbol");
            assert!(!bytes.is_empty());
            embedded.insert(name);
        }
        for v in family::VARIANTS {
            let mut new = arm(&mut a, case, inputs).unwrap();
            candidate(&mut a, case, &mut new, v).unwrap();
            for op in new.launches.iter().chain(&new.clock_launches) {
                op.plan.validate_geometry().unwrap();
                assert!(op.plan.dynamic_shared_bytes <= 101376);
                if op.symbol.starts_with("chop_exact_prefill") {
                    assert!(symbols.contains(&op.symbol));
                }
            }
        }
    }
    println!(
        "exact_prefill_dry_run {}",
        serde_json::json!({"family":family::FAMILY,"cases":cases().len(),"variants":family::VARIANTS.iter().map(|v|v.name).collect::<Vec<_>>(),"required_ordinals":1,"modules":[family::CUBIN],"symbols":symbols,"embedded_modules":embedded,"baseline_selection":selections})
    );
}
#[test]
fn queue_partition_and_resources() {
    for max in [64_u32, 128] {
        for count in 0..=8192 {
            let full = count / max;
            let tail = count % max;
            let tails = if tail == 0 {
                0
            } else if tail <= 8 {
                8
            } else if tail <= 32 {
                32
            } else if tail <= 64 {
                64
            } else {
                128
            };
            assert!(full * max + tails >= count);
            assert!(full * max + tails < count + max);
        }
    }
    for p in family::projections() {
        for v in family::VARIANTS {
            for r in family::tile_rows(v) {
                assert!(family::shared_bytes(&p, v, r) <= 101376);
            }
            assert!(
                family::launches(
                    &p,
                    v,
                    0,
                    family::Buffers {
                        h: [1; 10],
                        counts: 1,
                        active: 1,
                        scratch: 1,
                        decoded: 1,
                        stamps: 1
                    },
                    false,
                    false
                )
                .is_err()
            );
            assert!(
                family::launches(
                    &p,
                    v,
                    8193,
                    family::Buffers {
                        h: [1; 10],
                        counts: 1,
                        active: 1,
                        scratch: 1,
                        decoded: 1,
                        stamps: 1
                    },
                    false,
                    false
                )
                .is_err()
            );
        }
    }
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_EXACT_PREFILL_CUBINS"]
fn gpu_matrix() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 16 << 30, run).unwrap();
}
