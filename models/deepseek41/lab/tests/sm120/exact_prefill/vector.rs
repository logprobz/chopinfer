//! Compare vector stores against current serving and their scalar controls.
use super::*;
#[path = "../../../sm120/exact_prefill_vector.rs"]
#[allow(dead_code)]
mod vector_family;
const ENV: &str = "CHOPINFER_DEEPSEEK41_VECTOR_PREFILL_CUBINS";

fn variants() -> Vec<family::Variant> {
    vector_family::VARIANTS
        .iter()
        .map(|name| {
            family::VARIANTS
                .into_iter()
                .find(|v| v.name == *name)
                .unwrap()
        })
        .collect()
}

fn serving_arm(a: &mut Arena<'_>, case: Case, inputs: Inputs) -> LabResult<Arm> {
    let mut result = arm(a, case, inputs)?;
    let usage = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: case.rows,
        graph: false,
    };
    if bundle_deepseek41::sm120::moe_pipeline::selected(
        &FLASH,
        4,
        case.set,
        GatedMlpProjection::GateUp,
        usage,
    ) {
        let down = component(&result, "down");
        candidate(
            a,
            case,
            &mut result,
            family::VARIANTS
                .into_iter()
                .find(|v| v.name == "decoded128")
                .unwrap(),
        )?;
        vectorize(&mut result);
        let mut ops = Vec::new();
        let mut inserted = false;
        for op in result.launches {
            if op.symbol.contains("before_down_") {
                if !inserted {
                    ops.extend(down.clone());
                    inserted = true;
                }
            } else {
                ops.push(op);
            }
        }
        result.launches = ops;
    }
    Ok(result)
}

fn vectorize(arm: &mut Arm) {
    for op in arm.launches.iter_mut().chain(&mut arm.clock_launches) {
        if op.symbol.starts_with("chop_exact_prefill_") {
            op.symbol = vector_family::symbol(&op.symbol);
        }
    }
}

#[test]
fn vector_cpu_matrix() {
    let symbols = vector_family::cubins()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect::<std::collections::BTreeSet<_>>();
    let mut embedded = std::collections::BTreeSet::new();
    let mut selections = Vec::new();
    for case in cases() {
        let mut a = Arena {
            session: None,
            next: 0,
            buffers: BTreeMap::new(),
        };
        let live = a.bytes(&case.rows.to_le_bytes()).unwrap();
        let (input, scales, prefix) = source(&mut a, case, live).unwrap();
        let weights = banks(&mut a, case.set).unwrap();
        let inputs = Inputs {
            input,
            scales,
            live,
            weights,
            logits: a
                .filled(u64::from(case.rows) * u64::from(case.set.count) * 4, 0)
                .unwrap(),
        };
        let serving = serving_arm(&mut a, case, inputs).unwrap();
        let backend = registration::register(Sm120::new(), &FLASH, 4).unwrap();
        let usage = OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: case.rows,
            graph: false,
        };
        for (kind, part) in [
            (GatedMlpProjection::GateUp, "gateup"),
            (GatedMlpProjection::Down, "down"),
        ] {
            let query = moe::projection(&FLASH, 4, case.set, kind, case.rows).unwrap();
            let base = backend
                .chained_mlp_projection(&query, usage, Need::rows(case.rows))
                .unwrap();
            let prepared = bundle_deepseek41::sm120::moe_pipeline::prepare(
                &FLASH, 4, case.set, kind, usage, base,
            )
            .unwrap();
            let expected = prepared
                .info()
                .symbols
                .iter()
                .filter(|(s, _)| s.contains(&format!("before_{part}_")))
                .map(|(s, _)| s.clone())
                .collect::<Vec<_>>();
            let actual = component(&serving, part)
                .iter()
                .map(|o| o.symbol.clone())
                .collect::<Vec<_>>();
            // Serving binds the separately qualified grouped down at this
            // shape; this arm keeps its column down as the gate reference.
            if bundle_deepseek41::sm120::moe_grouped_prefill::selected(
                &FLASH, 4, case.set, kind, usage,
            ) {
                assert_eq!(kind, GatedMlpProjection::Down);
                assert_eq!(expected.len(), 2);
                assert!(expected[0].ends_with("_group_m128n128s2c1_pack_v1"));
                assert!(expected[1].ends_with("_group_m128n128s2c1_v1"));
                assert!(actual.iter().all(|s| !s.contains("_group_")));
                selections.push(serde_json::json!({
                    "case": case.label(), "part": part, "symbols": actual, "serving": expected
                }));
                continue;
            }
            assert_eq!(actual, expected);
            selections.push(serde_json::json!({
                "case": case.label(), "part": part, "symbols": actual
            }));
        }
        for op in prefix.iter().chain(component(&serving, "full").iter()) {
            let (name, bytes) =
                bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol).unwrap();
            assert!(!bytes.is_empty());
            embedded.insert(name);
        }
        for variant in variants() {
            let mut scalar = arm(&mut a, case, inputs).unwrap();
            candidate(&mut a, case, &mut scalar, variant).unwrap();
            for op in component(&scalar, "full") {
                let (name, bytes) =
                    bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol).unwrap();
                assert!(!bytes.is_empty());
                embedded.insert(name);
            }
            vectorize(&mut scalar);
            for op in scalar.launches.iter().chain(&scalar.clock_launches) {
                op.plan.validate_geometry().unwrap();
                if op.symbol.starts_with("chop_exact_prefill_") {
                    assert!(symbols.contains(&op.symbol));
                }
            }
        }
    }
    println!(
        "vector_prefill_dry_run {}",
        serde_json::json!({
            "family": vector_family::FAMILY, "cases": 16, "variants": vector_family::VARIANTS,
            "required_ordinals": 1, "modules": [vector_family::CUBIN],
            "symbols": symbols, "embedded_modules": embedded, "baseline_selection": selections
        })
    );
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_VECTOR_PREFILL_CUBINS"]
fn vector_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 20 << 30, run).unwrap();
}

fn run(session: &Session) -> LabResult<()> {
    let directory = std::path::PathBuf::from(std::env::var(ENV).map_err(|e| e.to_string())?);
    let bytes = std::fs::read(directory.join(format!("{}.cubin", vector_family::CUBIN)))
        .map_err(|e| e.to_string())?;
    let vector_module = session.load_cubin(&bytes)?;
    let mut embedded = BTreeMap::new();
    let mut functions = BTreeMap::<String, Function>::new();
    for set in moe::expert_sets(&FLASH) {
        let mut a = Arena {
            session: Some(session),
            next: 0,
            buffers: BTreeMap::new(),
        };
        let weights = banks(&mut a, set)?;
        let weights_keep = a.next;
        for case in cases().into_iter().filter(|c| c.set == set) {
            let live = a.bytes(&case.rows.to_le_bytes())?;
            let (input, scales, prefix) = source(&mut a, case, live)?;
            let inputs = Inputs {
                input,
                scales,
                live,
                weights,
                logits: a.bytes(&case.logits())?,
            };
            let serving = serving_arm(&mut a, case, inputs)?;
            let serving_keep = a.next;
            for variant in variants() {
                let mut scalar = arm(&mut a, case, inputs)?;
                candidate(&mut a, case, &mut scalar, variant)?;
                let mut vector = arm(&mut a, case, inputs)?;
                candidate(&mut a, case, &mut vector, variant)?;
                vectorize(&mut vector);
                let arms = [&serving, &scalar, &vector];
                let mut ops = prefix.clone();
                ops.extend(arms.iter().flat_map(|arm| component(arm, "full")));
                ops.extend(vector.clock_launches.clone());
                let mut negative = component(&vector, "full");
                if variant.name == "pipe64" {
                    for op in &mut negative {
                        if op.symbol.contains("before_down_") && op.symbol.contains("_r") {
                            op.symbol = op.symbol.replace("_vector_v1", "_omit_vector_v1");
                        }
                    }
                    ops.extend(negative.clone());
                }
                for op in &ops {
                    if functions.contains_key(&op.symbol) {
                        continue;
                    }
                    let function = if op.symbol.ends_with("_vector_v1") {
                        vector_module
                            .function(&op.symbol)
                            .map_err(|e| e.to_string())?
                    } else {
                        let (name, bytes) =
                            bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
                                .ok_or_else(|| format!("missing {}", op.symbol))?;
                        if !embedded.contains_key(name) {
                            embedded.insert(name, session.load_cubin(bytes)?);
                        }
                        embedded[name]
                            .function(&op.symbol)
                            .map_err(|e| e.to_string())?
                    };
                    if op.plan.dynamic_shared_bytes > 0 {
                        function
                            .set_max_dynamic_shared_memory(op.plan.dynamic_shared_bytes)
                            .map_err(|e| e.to_string())?;
                    }
                    functions.insert(op.symbol.clone(), function);
                }
                let bindings = Bindings {
                    buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
                };
                let issue = |ops: &[Launch], stream: &Stream| -> LabResult<()> {
                    for op in ops {
                        launch(
                            &session.context,
                            &functions[&op.symbol],
                            stream,
                            op.plan,
                            &op.arguments,
                            &bindings,
                        )?;
                    }
                    Ok(())
                };
                issue(&prefix, &session.stream)?;
                let graphs = arms
                    .iter()
                    .map(|arm| session.capture(|s| issue(&component(arm, "full"), s)))
                    .collect::<LabResult<Vec<_>>>()?;
                for count in [case.rows, case.rows - 7, 0, case.rows] {
                    a.buffers[&live.handle]
                        .copy_from_bytes(&count.to_le_bytes(), &session.stream)
                        .map_err(|e| e.to_string())?;
                    for i in 0..3 {
                        compare::poison(&a, arms[i])?;
                        graphs[i]
                            .launch(&session.stream)
                            .map_err(|e| e.to_string())?;
                    }
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    compare::equal(&a, case, &serving, &scalar, count)?;
                    compare::equal(&a, case, &serving, &vector, count)?;
                }
                let mut times = serde_json::Map::new();
                let mut scalar_times = serde_json::Map::new();
                for part in ["gateup", "down", "projections", "full"] {
                    let stages = arms
                        .iter()
                        .map(|arm| session.capture(|s| issue(&component(arm, part), s)))
                        .collect::<LabResult<Vec<_>>>()?;
                    times.insert(
                        part.into(),
                        timing::pair(session, |i, s| {
                            stages[if i == 0 { 0 } else { 2 }]
                                .launch(s)
                                .map_err(|e| e.to_string())
                        })?,
                    );
                    scalar_times.insert(
                        part.into(),
                        timing::pair(session, |i, s| {
                            stages[i + 1].launch(s).map_err(|e| e.to_string())
                        })?,
                    );
                }
                issue(&vector.clock_launches, &session.stream)?;
                issue(&vector.sum, &session.stream)?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                compare::equal(&a, case, &serving, &vector, case.rows)?;
                let clocks = compare::clock_table(&a, &vector)?;
                let tiles = compare::utilization(&a, &serving, variant)?;
                if variant.name == "pipe64" {
                    let before = a.read(vector.outputs[6])?;
                    issue(&negative, &session.stream)?;
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    if before == a.read(vector.outputs[6])? {
                        return Err("vector omitted-chain control escaped".into());
                    }
                }
                println!(
                    "vector_prefill_result {}",
                    serde_json::json!({
                        "case": case.label(), "experts": set.count, "top_k": set.top_k,
                        "input_rows": case.rows, "routed_rows": case.rows * set.top_k,
                        "routing": if case.skewed { "skewed" } else { "balanced" },
                        "variant": variant.name, "bitwise": true, "scalar_bitwise": true,
                        "clock_bitwise": true, "boundary_replays": 4,
                        "negative_control_caught": variant.name == "pipe64",
                        "serving_enabled": false, "timing": times, "scalar_timing": scalar_times,
                        "clock": clocks, "tile_utilization": tiles,
                        "baseline": {
                            "gateup": component(&serving, "gateup").iter().map(|o| &o.symbol).collect::<Vec<_>>(),
                            "down": component(&serving, "down").iter().map(|o| &o.symbol).collect::<Vec<_>>()
                        }
                    })
                );
                drop(graphs);
                drop(bindings);
                a.buffers.retain(|&h, _| h <= serving_keep);
            }
            a.buffers.retain(|&h, _| h <= weights_keep);
        }
    }
    Ok(())
}
