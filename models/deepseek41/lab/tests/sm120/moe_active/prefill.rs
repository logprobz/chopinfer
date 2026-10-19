//! Full prefill MoE comparison. Serving stays on the locked old entries.
use super::*;
use kernels_sm120::moe::chained::prefill as grouped;
mod input;
mod timing;

const PARTS: [&str; 9] = [
    "input_quant",
    "router_projection",
    "routing",
    "gateup",
    "intermediate_quant",
    "down",
    "shared",
    "combine",
    "full",
];

fn cases(set: moe::Experts) -> Vec<Case> {
    [512, 2048, 4096]
        .into_iter()
        .flat_map(|rows| {
            [Routing::All, Routing::Seeded, Routing::Concentrated].map(|routing| Case {
                rows,
                phase: ExecutionPhase::Prefill,
                set,
                routing,
            })
        })
        .collect()
}

fn build(
    a: &mut Arena<'_>,
    case: Case,
    live: View,
    source: &input::Source,
    routed: [QuantizedWeights<View>; 2],
    shared: [QuantizedWeights<View>; 3],
) -> LabResult<Vec<Arm>> {
    [Mode::Old, Mode::PrefillGrouped]
        .into_iter()
        .map(|mode| {
            let inputs = Inputs {
                input: a.filled(u64::from(case.rows) * 5120, 0)?,
                scales: a.filled(u64::from(case.rows) * 160, 0)?,
                live,
                logits: a.filled(u64::from(case.rows) * u64::from(case.set.count) * 4, 0)?,
                weights: routed,
            };
            let mut arm = arm(a, case, mode, inputs, shared)?;
            input::prepend(a, case, source, inputs, &mut arm)?;
            Ok(arm)
        })
        .collect()
}

fn component(arm: &Arm, part: &str) -> Vec<Launch> {
    if part == "shared" {
        return arm.side.clone();
    }
    if part == "combine" {
        return arm.sum.clone();
    }
    if part == "full" {
        return arm
            .launches
            .iter()
            .chain(&arm.side)
            .chain(&arm.sum)
            .cloned()
            .collect();
    }
    arm.launches
        .iter()
        .filter(|op| match part {
            "input_quant" => op.symbol.contains("moe_ceil_w5120_"),
            "router_projection" => {
                op.symbol.contains("dense_f32_") || op.symbol.contains("dspark_router_")
            }
            "routing" => op.symbol.contains("sqrt_softplus_row_bias_route_"),
            "gateup" => op.symbol.contains("before_gateup_"),
            "intermediate_quant" => {
                op.symbol.contains("moe_ceil_") && !op.symbol.contains("moe_ceil_w5120_")
            }
            "down" => op.symbol.contains("before_down_"),
            _ => unreachable!(),
        })
        .cloned()
        .collect()
}
fn mutant(arm: &Arm) -> Vec<Launch> {
    component(arm, "full")
        .into_iter()
        .map(|mut op| {
            if op.symbol.contains("before_down_") && !op.symbol.contains("_pack_v1") {
                op.symbol = grouped::negative_symbol(&op.symbol);
            }
            op
        })
        .collect()
}

fn compare(
    a: &mut Arena<'_>,
    case: Case,
    routed: [QuantizedWeights<View>; 2],
    shared: [QuantizedWeights<View>; 3],
    calibration: &serde_json::Value,
) -> LabResult<()> {
    let session = a.session.unwrap();
    let live = a.bytes(&case.rows.to_le_bytes())?;
    let source = input::source(a, case)?;
    let arms = build(a, case, live, &source, routed, shared)?;
    let bad = mutant(&arms[1]);
    let mut functions = BTreeMap::new();
    for op in arms
        .iter()
        .flat_map(|a| component(a, "full"))
        .chain(bad.iter().cloned())
    {
        if !functions.contains_key(&op.symbol) {
            let (_, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
                .ok_or_else(|| format!("missing {}", op.symbol))?;
            functions.insert(
                op.symbol.clone(),
                session
                    .load_cubin(bytes)?
                    .function(&op.symbol)
                    .map_err(|e| e.to_string())?,
            );
        }
    }
    let bindings = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    let issue = |ops: &[Launch], s: &Stream| -> LabResult<()> {
        for op in ops {
            launch(
                &session.context,
                &functions[&op.symbol],
                s,
                op.plan,
                &op.arguments,
                &bindings,
            )?;
        }
        Ok(())
    };
    let graphs = arms
        .iter()
        .map(|arm| session.capture(|s| issue(&component(arm, "full"), s)))
        .collect::<LabResult<Vec<_>>>()?;
    let bad_graph = session.capture(|s| issue(&bad, s))?;
    let mut stages = BTreeMap::new();
    for part in PARTS {
        let graphs = arms
            .iter()
            .map(|arm| {
                let ops = component(arm, part);
                if ops.is_empty() {
                    return Err(format!("missing component {part}"));
                }
                session.capture(|s| issue(&ops, s))
            })
            .collect::<LabResult<Vec<_>>>()?;
        stages.insert(part, graphs);
    }
    // The old path is measured first, with all intermediate inputs produced
    // by its complete graph. Candidate timing and selection come later.
    graphs[0]
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    let mut baseline = BTreeMap::new();
    for part in PARTS {
        baseline.insert(
            part,
            timing::pair(session, |_, s| {
                stages[part][0].launch(s).map_err(|e| e.to_string())
            })?,
        );
    }
    println!(
        "PREFILL_BASELINE {}",
        serde_json::json!({"case":case.label(),"calibration":calibration,"timing":baseline})
    );
    let poison = |arm: &Arm| -> LabResult<()> {
        for view in &arm.poison {
            a.buffers[&view.handle]
                .copy_from_bytes(&vec![255; view.bytes as usize], &session.stream)
                .map_err(|e| e.to_string())?;
        }
        a.buffers[&arm.status.handle]
            .copy_from_bytes(&vec![0; arm.status.bytes as usize], &session.stream)
            .map_err(|e| e.to_string())
    };
    let equal = |rows: u32| -> LabResult<()> {
        for (index, (&old, &new)) in arms[0].outputs.iter().zip(&arms[1].outputs).enumerate() {
            let left = a.read(old)?;
            if left != a.read(new)? {
                return Err(format!(
                    "{} live {rows} output {index}: bitwise mismatch",
                    case.label()
                ));
            }
            let tail = rows as usize * left.len() / case.rows as usize;
            let written = &left[..tail];
            let valid = match index {
                0 => written.chunks_exact(4).all(|v| {
                    (0..case.set.count as i32).contains(&i32::from_le_bytes(v.try_into().unwrap()))
                }),
                1 | 5 | 6 | 7 | 10 | 11 | 14 => written
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()),
                2 => written
                    .chunks_exact(2)
                    .all(|v| u16::from_le_bytes(v.try_into().unwrap()) & 0x7f80 != 0x7f80),
                3 | 8 | 12 => written.iter().all(|v| v & 0x7f != 0x7f),
                4 | 9 | 13 => !written.contains(&255),
                _ => unreachable!(),
            };
            if !valid || left[tail..].iter().any(|&v| v != 255) {
                return Err(format!(
                    "output {index}: invalid live value or changed inactive tail"
                ));
            }
        }
        for arm in &arms {
            if a.read(arm.status)?.iter().any(|&v| v != 0) {
                return Err("nonzero quantizer status".into());
            }
        }
        Ok(())
    };
    let live_replays = [case.rows, case.rows / 2 + 1, 0, case.rows];
    for rows in live_replays {
        a.buffers[&live.handle]
            .copy_from_bytes(&rows.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        for (i, arm) in arms.iter().enumerate() {
            poison(arm)?;
            graphs[i]
                .launch(&session.stream)
                .map_err(|e| e.to_string())?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())?;
        equal(rows)?;
    }
    poison(&arms[1])?;
    bad_graph
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    if a.read(arms[0].outputs[11])? == a.read(arms[1].outputs[11])? {
        return Err("omitted-chain negative control escaped final-output comparison".into());
    }
    poison(&arms[1])?;
    graphs[1]
        .launch(&session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    equal(case.rows)?;
    let counts = a
        .read(arms[0].route_counts)?
        .chunks_exact(4)
        .take(case.set.count as usize)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(counts.iter().sum::<u32>(), case.rows * case.set.top_k);
    let active = counts.iter().filter(|&&n| n != 0).count();
    if case.routing == Routing::All {
        assert_eq!(active, case.set.count as usize);
    }
    if case.routing == Routing::Concentrated {
        assert_eq!(active, case.set.top_k as usize);
    }
    let mut timing = BTreeMap::new();
    for part in PARTS {
        timing.insert(
            part,
            timing::pair(session, |i, s| {
                stages[part][i].launch(s).map_err(|e| e.to_string())
            })?,
        );
    }
    let record = serde_json::json!({"case":case.label(),"rows":case.rows,"experts":case.set.count,"top_k":case.set.top_k,
        "routing":format!("{:?}",case.routing),"counts":counts,"active_experts":active,"ranks":4,
        "bitwise":true,"negative_detected":true,"compared_outputs":15,"live_replays":live_replays,
        "selection":"old prefill","candidate":"grouped exact 128-K","router_weights":"synthetic BF16 identity prefix and dense common suffix",
        "cache":"warm graph replay","calibration":calibration,"timing":timing});
    println!("PREFILL_RESULT {record}");
    session.receipt(
        "moe-prefill-bitwise",
        serde_json::from_value(record).map_err(|e| e.to_string())?,
    )?;
    Ok(())
}

fn suite(bank: usize) {
    kernel_harness::gpu_test(16 << 30, |session| {
        let calibration = timing::calibration(session)?;
        println!("PREFILL_CALIBRATION {calibration}");
        let set = moe::expert_sets(&FLASH)[bank];
        let mut a = Arena {
            session: Some(session),
            next: 0,
            buffers: BTreeMap::new(),
        };
        let routed = [
            weights(&mut a, set.count as usize, 1152, 5120)?,
            weights(&mut a, set.count as usize, 5120, 576)?,
        ];
        let shared = shared::weights(&mut a)?;
        let weights_end = a.next;
        let mut errors = Vec::new();
        for case in cases(set) {
            if let Err(error) = compare(&mut a, case, routed, shared, &calibration) {
                errors.push(format!("{}: {error}", case.label()));
            }
            a.buffers.retain(|&h, _| h <= weights_end);
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("\n"))
        }
    })
    .unwrap();
}
#[test]
fn prefill_backbone_gpu() {
    suite(0);
}
#[test]
fn prefill_dspark_gpu() {
    suite(1);
}

#[test]
fn prefill_cpu_dry_run() {
    for projection in registration::prefill_projections(&FLASH).unwrap() {
        let entries = grouped::entries(&projection);
        assert!(grouped::workspace_bytes(&projection) < 65536);
        for stage in grouped::STAGES {
            let name = grouped::symbol(&projection, stage);
            let entry = entries.iter().find(|e| e.symbol == name).unwrap();
            for rows in [1, 512, 2048, 4096, 8192] {
                let plan = grouped::plan(&projection, stage, rows).unwrap();
                assert_eq!(plan.block[0], entry.max_threads);
            }
            assert!(grouped::plan(&projection, stage, 8193).is_err());
        }
    }
    let mut visited = 0;
    for set in moe::expert_sets(&FLASH) {
        let mut a = Arena {
            session: None,
            next: 0,
            buffers: BTreeMap::new(),
        };
        let routed = [
            weights(&mut a, set.count as usize, 1152, 5120).unwrap(),
            weights(&mut a, set.count as usize, 5120, 576).unwrap(),
        ];
        let shared = shared::weights(&mut a).unwrap();
        for case in cases(set) {
            let live = a.filled(4, 0).unwrap();
            let source = input::source(&mut a, case).unwrap();
            let arms = build(&mut a, case, live, &source, routed, shared).unwrap();
            for arm in &arms {
                assert_eq!(arm.outputs.len(), 15);
                for part in PARTS {
                    assert!(!component(arm, part).is_empty(), "{part}");
                }
            }
            assert!(
                !arms[0]
                    .launches
                    .iter()
                    .any(|op| op.symbol.contains("_prefill_") && op.symbol.contains("before_"))
            );
            assert_eq!(
                arms[1]
                    .launches
                    .iter()
                    .filter(|op| op.symbol.contains("_prefill_") && op.symbol.contains("before_"))
                    .count(),
                8
            );
            for op in arms
                .iter()
                .flat_map(|a| component(a, "full"))
                .chain(mutant(&arms[1]))
            {
                assert!(
                    bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol).is_some(),
                    "{}",
                    op.symbol
                );
                op.plan.validate_geometry().unwrap();
            }
            visited += 1;
        }
    }
    assert_eq!(visited, 18);
}
