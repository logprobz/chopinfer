//! Compare current TMA tails with separate full-tile and fold dispatch.
use super::*;
use kernels_sm120::gemm::mxfp8_prefill_split as split;
const ROWS: [u32; 8] = [1, 17, 127, 129, 1000, 4095, 8150, 8191];

fn split_module(e: &Entry) -> String {
    if e.query.m_capacity == 257 {
        kernel_lab_families::mxfp8_prefill_split::CUBIN.into()
    } else {
        models()
            .into_iter()
            .find(|(model, _, _)| model.symbol == e.symbol)
            .expect("declared model shape")
            .2
            .replacen("mxfp8_prefill_", "mxfp8_prefill_split_", 1)
    }
}
fn symbols() -> BTreeMap<String, Vec<String>> {
    let mut all = required_symbols();
    for (e, _, _) in entries() {
        let names = all.entry(split_module(&e)).or_default();
        names.extend([
            split::validation_symbol(&e),
            split::symbol(&e),
            split::tail_symbol(&e),
        ]);
        if e.query.k == 65536 {
            names.push(split::control_spec(&e).symbol);
        }
    }
    for names in all.values_mut() {
        names.sort();
        names.dedup();
    }
    all
}

#[test]
fn plans_symbols_and_launch_attributes_cover_partial_tiles() {
    let mut all_resources = BTreeMap::new();
    if std::env::var_os("CHOPINFER_PREFILL_TMA_CUBINS").is_some() {
        let locks = root().parent().unwrap().join("locks");
        for file in std::fs::read_dir(locks).unwrap() {
            let records: serde_json::Value =
                serde_json::from_slice(&std::fs::read(file.unwrap().path()).unwrap()).unwrap();
            for cubin in records.as_array().unwrap() {
                for r in cubin["resources"].as_array().unwrap() {
                    all_resources.insert(
                        (
                            cubin["name"].as_str().unwrap().to_owned(),
                            r["symbol"].as_str().unwrap().to_owned(),
                        ),
                        (
                            r["max_threads"].as_u64().unwrap() as u32,
                            r["shared_static_bytes"].as_u64().unwrap() as u32,
                        ),
                    );
                }
            }
        }
        for (module, names) in symbols() {
            let out = std::process::Command::new("readelf")
                .args(["-Ws", "--wide"])
                .arg(root().join(format!("{module}.cubin")))
                .output()
                .unwrap();
            assert!(out.status.success());
            let text = String::from_utf8(out.stdout).unwrap();
            for symbol in names {
                assert!(
                    text.lines().any(|line| line.contains("FUNC")
                        && line.contains("GLOBAL")
                        && !line.contains(" UND ")
                        && line.split_whitespace().last() == Some(symbol.as_str())),
                    "{module}: {symbol}"
                );
                assert!(all_resources.contains_key(&(module.clone(), symbol.clone())));
                println!("TAIL_SYMBOL {module} {symbol} PASS");
            }
        }
    }
    for (e, old_module, current_module) in entries() {
        let old = Sm120::new()
            .with_mxfp8_fold(e.clone())
            .mxfp8_fold_gemm(&e.query, Need::rows(e.query.m_capacity))
            .unwrap();
        let current = Sm120::new()
            .with_mxfp8_prefill(e.clone())
            .mxfp8_prefill_gemm(&e.query, Need::rows(e.query.m_capacity))
            .unwrap();
        let candidate = Sm120::new()
            .with_mxfp8_prefill_split(e.clone())
            .mxfp8_prefill_gemm(&e.query, Need::rows(e.query.m_capacity))
            .unwrap();
        assert_eq!(old.info().bytes, candidate.info().bytes);
        assert_eq!(old.info().guarantees, candidate.info().guarantees);
        let split_module = split_module(&e);
        for rows in ROWS
            .into_iter()
            .chain([128, 256])
            .filter(|&r| r <= e.query.m_capacity)
        {
            let b = views(&e, rows);
            let ls = candidate
                .launches(&b, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let splitting = rows >= 128 && rows % 128 != 0;
            assert_eq!(ls.len(), if splitting { 3 } else { 2 });
            if splitting {
                assert_eq!(
                    ls[1].plan.grid,
                    [e.query.n.div_ceil(128), rows / 128, e.query.groups]
                );
                assert_eq!(ls[2].plan.grid, [e.query.n.div_ceil(32), 8, e.query.groups]);
                assert_eq!(
                    ls[2].arguments,
                    old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap()[0].arguments
                );
            }
            for (op, module) in [
                (&old, &old_module),
                (&current, &current_module),
                (
                    &candidate,
                    if splitting {
                        &split_module
                    } else {
                        &current_module
                    },
                ),
            ] {
                for l in op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap() {
                    assert_eq!(l.plan.workspace_bytes, 0);
                    if !all_resources.is_empty() {
                        let (threads, static_shared) =
                            all_resources[&(module.clone(), l.symbol.clone())];
                        let shared = op
                            .info()
                            .symbols
                            .iter()
                            .find(|(s, _)| s == &l.symbol)
                            .unwrap()
                            .1;
                        let mut applied = None;
                        launch_checks::configure_shared(shared, |bytes| {
                            applied = Some(bytes);
                            Ok(())
                        })
                        .unwrap();
                        launch_checks::audit(l.plan, threads, static_shared, applied)
                            .unwrap_or_else(|e| {
                                panic!("{}: {e}", launch_checks::context(module, &l, shared))
                            });
                        println!(
                            "TAIL_CONFIG {} PASS",
                            launch_checks::context(module, &l, shared)
                        );
                        if e.query.k == 65536 && l.symbol == split::tail_symbol(&e) {
                            let mut control = l.clone();
                            control.symbol = split::control_spec(&e).symbol;
                            let (threads, static_shared) =
                                all_resources[&(module.clone(), control.symbol.clone())];
                            launch_checks::audit(control.plan, threads, static_shared, None)
                                .unwrap();
                            println!(
                                "TAIL_CONFIG {} PASS",
                                launch_checks::context(module, &control, 0)
                            );
                        }
                    }
                }
            }
            let mut bad = b;
            bad.output.bytes -= 1;
            assert!(
                candidate
                    .launches(&bad, View::EMPTY, Counts::rows(rows))
                    .is_err()
            );
        }
        assert!(
            Sm120::new()
                .with_mxfp8_prefill(e.clone())
                .with_mxfp8_prefill_split(e.clone())
                .mxfp8_prefill_gemm(&e.query, Need::rows(1))
                .is_err()
        );
    }
    if let Some(path) = std::env::var_os("CHOPINFER_PREFILL_TAIL_EXPORT") {
        let manifest = serde_json::json!({"symbols":symbols(),"rows":ROWS,"model_shapes":models().len(),
            "bitwise":models().len()*13+kernel_lab_families::mxfp8_prefill::entries().len()*5+1,
            "timings":models().len()*8,"negative":1});
        std::fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }
}

#[test]
fn partial_tile_strategies_match_bits_and_time_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("tail comparison needs idle GPUs").unwrap();
        return;
    }
    let probe = std::fs::read(root().join("probe_sm120.cubin"))
        .unwrap()
        .leak();
    let cubins = Cubins::new(
        Box::leak(Box::new([("probe_sm120", &*probe)])),
        &[(
            "probe_sm120",
            &["chop_probe_gate_v1", "chop_probe_flush_v1"],
        )],
    );
    let result = kernel_harness::gpu_test_with(cubins, 4 << 30, |session| {
        let mut modules = BTreeMap::new();
        for name in symbols().keys().filter(|n| *n != "probe_sm120") {
            modules.insert(
                name.clone(),
                session.load_cubin(
                    &std::fs::read(root().join(format!("{name}.cubin")))
                        .map_err(|e| e.to_string())?,
                )?,
            );
        }
        let (mut bits, mut times) = (0, 0);
        for (e, old, current) in models() {
            for rows in ROWS {
                let candidate = split_module(&e);
                if rows < 128 {
                    case_strategies(
                        session,
                        &modules,
                        &e,
                        &current,
                        &old,
                        rows,
                        rows,
                        0,
                        true,
                        false,
                        [1, 0],
                    )?;
                    bits += 1;
                } else {
                    case_strategies(
                        session,
                        &modules,
                        &e,
                        &old,
                        &candidate,
                        rows,
                        rows,
                        0,
                        false,
                        false,
                        [0, 2],
                    )?;
                    case_strategies(
                        session,
                        &modules,
                        &e,
                        &current,
                        &candidate,
                        rows,
                        rows,
                        0,
                        true,
                        false,
                        [1, 2],
                    )?;
                    bits += 2;
                }
                times += 1;
            }
        }
        for e in kernel_lab_families::mxfp8_prefill::entries() {
            for (rows, live, pattern) in [
                (129, 128, 0),
                (257, 0, 2),
                (257, 17, 0),
                (257, 129, 2),
                (257, 256, 0),
            ] {
                case_strategies(
                    session,
                    &modules,
                    &e,
                    kernel_lab_families::mxfp8_prefill::CUBIN,
                    &split_module(&e),
                    rows,
                    live,
                    pattern,
                    false,
                    false,
                    [0, 2],
                )?;
                bits += 1;
            }
        }
        let witness = kernel_lab_families::mxfp8_prefill::entries()
            .into_iter()
            .find(|e| e.query.k == 65536 && e.query.output == DType::F32)
            .unwrap();
        case_strategies(
            session,
            &modules,
            &witness,
            kernel_lab_families::mxfp8_prefill::CUBIN,
            &split_module(&witness),
            129,
            129,
            0,
            false,
            true,
            [0, 2],
        )?;
        bits += 1;
        assert_eq!(
            bits,
            models().len() * 13 + kernel_lab_families::mxfp8_prefill::entries().len() * 5 + 1
        );
        assert_eq!(times, models().len() * 8);
        println!("TAIL_COMPLETE bitwise={bits} timings={times} negative=1");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "required tail comparison skipped");
}

#[test]
fn measured_policy_keeps_known_slow_tails() {
    let shapes = models();
    for (rows, expected_splits) in [
        (128, 0),
        (129, 20),
        (1000, 17),
        (4095, 15),
        (8150, 16),
        (8191, 15),
        (8192, 0),
    ] {
        let mut count = 0;
        for (e, _, _) in &shapes {
            let policy = registration::prefill_tail_policy(e.query);
            let op = Sm120::new()
                .with_mxfp8_prefill_split_policy(e.clone(), policy)
                .mxfp8_prefill_gemm(&e.query, Need::rows(8192))
                .unwrap();
            let calls = op
                .launches(&views(e, rows), View::EMPTY, Counts::rows(rows))
                .unwrap();
            count += usize::from(calls.len() == 3);
            assert_eq!(calls.len() == 3, policy.splits(rows));
            assert_eq!(
                calls[1].symbol,
                if calls.len() == 3 {
                    split::symbol(e)
                } else {
                    tiled::symbol(e)
                }
            );
        }
        assert_eq!(count, expected_splits, "rows={rows}");
    }
    let e = shapes[0].0.clone();
    for invalid in [
        split::TailPolicy {
            max_rows: 127,
            max_tail_rows: 127,
        },
        split::TailPolicy {
            max_rows: 8192,
            max_tail_rows: 0,
        },
        split::TailPolicy {
            max_rows: 8192,
            max_tail_rows: 128,
        },
    ] {
        assert!(
            Sm120::new()
                .with_mxfp8_prefill_split_policy(e.clone(), invalid)
                .mxfp8_prefill_gemm(&e.query, Need::rows(8192))
                .is_err()
        );
    }
    let one = split::TailPolicy {
        max_tail_rows: 16,
        ..split::TailPolicy::ALL
    };
    assert!(one.splits(144));
    assert!(!one.splits(145));
    let seven = split::TailPolicy {
        max_tail_rows: 112,
        ..split::TailPolicy::ALL
    };
    assert!(seven.splits(240));
    assert!(!seven.splits(241));
    for entries in [[split::TailPolicy::ALL, one], [one, split::TailPolicy::ALL]] {
        assert!(
            Sm120::new()
                .with_mxfp8_prefill_split_policy(e.clone(), entries[0])
                .with_mxfp8_prefill_split_policy(e.clone(), entries[1])
                .mxfp8_prefill_gemm(&e.query, Need::rows(8192))
                .is_err()
        );
    }
}
