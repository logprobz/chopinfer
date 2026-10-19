//! Exact comparison of every declared prefill dense shape and generic tails.
#[path = "prefill_mxfp8_launch.rs"]
mod launch_checks;
#[path = "../../../bundle/src/sm120/mxfp8_fold.rs"]
#[allow(dead_code)]
mod registration;
use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::contract::{Argument, DType};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{Cubins, LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::gemm::{mxfp8_fold::Entry, mxfp8_prefill as tiled};
use model_deepseek41_contract::geometry::GEOMETRIES;
use runtime_cuda::{Function, Module, Stream};
use std::{collections::BTreeMap, path::PathBuf};

fn models() -> Vec<(Entry, String, String)> {
    GEOMETRIES
        .iter()
        .flat_map(|g| {
            registration::prefill_catalog(g)
                .unwrap()
                .into_iter()
                .map(|e| {
                    (
                        e,
                        registration::cubin_name(g),
                        registration::prefill_cubin_name(g),
                    )
                })
        })
        .collect()
}
fn root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_PREFILL_TMA_CUBINS").expect("prebuilt prefill cubins required"),
    )
}
fn views(e: &Entry, rows: u32) -> Mxfp8FoldGemmBuffers<View> {
    let b = e.bytes(rows);
    let v = |handle, bytes| View { handle, bytes };
    Mxfp8FoldGemmBuffers {
        activations: v(0, b.activations),
        activation_scales: v(1, b.activation_scales),
        weights: v(2, b.weights),
        weight_scales: v(3, b.weight_scales),
        output: v(4, b.output),
        live_rows: v(5, 4),
        status: v(6, (u64::from(rows) + 5) * 4),
        status_row_offset: 3,
    }
}
fn entries() -> Vec<(Entry, String, String)> {
    models()
        .into_iter()
        .chain(
            kernel_lab_families::mxfp8_prefill::entries()
                .into_iter()
                .map(|e| {
                    (
                        e,
                        kernel_lab_families::mxfp8_prefill::CUBIN.into(),
                        kernel_lab_families::mxfp8_prefill::CUBIN.into(),
                    )
                }),
        )
        .collect()
}
fn required_symbols() -> BTreeMap<String, Vec<String>> {
    let mut all = BTreeMap::<String, Vec<String>>::new();
    for (e, old, new) in entries() {
        all.entry(old).or_default().push(e.symbol.clone());
        all.entry(new.clone())
            .or_default()
            .extend([tiled::validation_symbol(&e), tiled::symbol(&e)]);
        if e.query.k == 65536 {
            all.entry(new).or_default().push(tiled::control_symbol(&e));
        }
    }
    all.insert(
        "probe_sm120".into(),
        vec!["chop_probe_gate_v1".into(), "chop_probe_flush_v1".into()],
    );
    for symbols in all.values_mut() {
        symbols.sort();
        symbols.dedup();
    }
    all
}

#[test]
fn prefill_tma_plans_cover_declared_shapes_and_preserve_refusals() {
    let model = models();
    assert!(!model.is_empty());
    for (e, _, _) in entries() {
        let old = Sm120::new()
            .with_mxfp8_fold(e.clone())
            .mxfp8_fold_gemm(&e.query, Need::rows(e.query.m_capacity))
            .unwrap();
        let new = Sm120::new()
            .with_mxfp8_prefill(e.clone())
            .mxfp8_prefill_gemm(&e.query, Need::rows(e.query.m_capacity))
            .unwrap();
        assert_eq!(old.info().bytes, new.info().bytes);
        assert_eq!(old.info().weight_layouts, new.info().weight_layouts);
        assert_eq!(old.info().guarantees, new.info().guarantees);
        for rows in [1, 129, e.query.m_capacity] {
            let b = views(&e, rows);
            let ls = new.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            assert_eq!(ls.len(), 2);
            assert_eq!(ls[0].plan.grid, [rows, 1, 1]);
            assert_eq!(
                ls[1].plan.grid,
                [e.query.n.div_ceil(128), rows.div_ceil(128), e.query.groups]
            );
            assert_eq!(ls[1].plan.dynamic_shared_bytes, tiled::SHARED);
            for (i, handle, dims) in [
                (
                    0,
                    0,
                    vec![u64::from(e.query.activation_row_stride), u64::from(rows)],
                ),
                (
                    1,
                    2,
                    vec![
                        u64::from(e.query.k),
                        u64::from(e.query.groups) * u64::from(e.query.n),
                    ],
                ),
            ] {
                let Argument::TensorMap(map) = &ls[1].arguments.0[i] else {
                    panic!("missing TMA map")
                };
                assert_eq!(map.base_buffer, handle);
                assert_eq!(map.dims, dims);
                assert_eq!(map.box_dims, [128, 128]);
            }
            for bad in [
                Mxfp8FoldGemmBuffers {
                    output: View {
                        bytes: b.output.bytes - 1,
                        ..b.output
                    },
                    ..b
                },
                Mxfp8FoldGemmBuffers {
                    status_row_offset: u64::MAX,
                    ..b
                },
                Mxfp8FoldGemmBuffers {
                    status: b.weights,
                    ..b
                },
            ] {
                assert!(new.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            }
        }
        let mut competing = e.clone();
        competing.symbol.push_str("_other_v1");
        assert!(
            Sm120::new()
                .with_mxfp8_prefill(e.clone())
                .with_mxfp8_prefill(competing)
                .mxfp8_prefill_gemm(&e.query, Need::rows(1))
                .is_err()
        );
    }
    let mut tp4_calls = BTreeMap::<String, u32>::new();
    for r in registration::registrations(&model_deepseek41_contract::geometry::FLASH, 4, 0).unwrap()
    {
        if r.usage.phase != kernel_core::precision::ExecutionPhase::Prefill {
            continue;
        }
        let calls = match r.projection.as_str() {
            "layer.wq_a"
            | "layer.wq_b"
            | "layer.wkv"
            | "layer.wo_a"
            | "layer.wo_b"
            | "layer.shared.gate-up"
            | "layer.shared.down" => 40,
            "csa.indexer.wq_b" => 8,
            "engram.wkv" => 2,
            _ => 0,
        };
        *tp4_calls.entry(r.entry.symbol).or_default() += calls;
    }
    assert_eq!(tp4_calls.values().sum::<u32>(), 330);
    let model_shapes = model
        .iter()
        .map(|(e, _, _)| {
            serde_json::json!({
                "symbol": e.symbol, "n": e.query.n, "k": e.query.k,
                "groups": e.query.groups, "rows": [1024, 2048, 8192],
            })
        })
        .collect::<Vec<_>>();
    let edge_shapes = kernel_lab_families::mxfp8_prefill::entries()
        .iter()
        .map(|e| {
            serde_json::json!({
                "symbol": e.symbol, "n": e.query.n, "k": e.query.k,
                "groups": e.query.groups, "f32": e.query.output == DType::F32,
            })
        })
        .collect::<Vec<_>>();
    let manifest = serde_json::json!({
        "model_cases": model_shapes.len() * 4,
        "edge_cases": edge_shapes.len() * 5,
        "timings": model_shapes.len() * 3,
        "model_shapes": model_shapes,
        "edge_shapes": edge_shapes,
        "tp4_calls": tp4_calls,
        "negative_controls": 1,
        "symbols": required_symbols(),
        "floor_tflops": 400,
    });
    if let Some(path) = std::env::var_os("CHOPINFER_PREFILL_TMA_EXPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    }
    println!(
        "TMA_PLAN shapes={} generic={}",
        model.len(),
        kernel_lab_families::mxfp8_prefill::entries().len()
    );
}

#[test]
fn prefill_tma_snapshot_has_every_module_and_symbol() {
    if std::env::var_os("CHOPINFER_PREFILL_TMA_CUBINS").is_none() {
        kernel_harness::skip("prefill symbol audit: set CHOPINFER_PREFILL_TMA_CUBINS").unwrap();
        return;
    }
    for (module, symbols) in required_symbols() {
        let path = root().join(format!("{module}.cubin"));
        let out = std::process::Command::new("readelf")
            .arg("-Ws")
            .arg(&path)
            .output()
            .unwrap();
        assert!(out.status.success());
        let text = String::from_utf8(out.stdout).unwrap();
        for symbol in symbols {
            assert!(
                text.lines()
                    .any(|l| l.split_whitespace().last() == Some(&symbol)),
                "{module}: {symbol}"
            );
            println!("TMA_SYMBOL {module} {symbol} PASS");
        }
    }
}
#[test]
fn prefill_tma_launch_configs_match_function_limits() {
    if std::env::var_os("CHOPINFER_PREFILL_TMA_CUBINS").is_none() {
        kernel_harness::skip("prefill launch audit: set CHOPINFER_PREFILL_TMA_CUBINS").unwrap();
        return;
    }
    let mut resources = BTreeMap::new();
    for file in ["old-model.lock", "new-model.lock", "lab.lock", "probe.lock"] {
        let cubins: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root().parent().unwrap().join("locks").join(file)).unwrap(),
        )
        .unwrap();
        for cubin in cubins.as_array().unwrap() {
            for entry in cubin["resources"].as_array().unwrap() {
                resources.insert(
                    (
                        cubin["name"].as_str().unwrap().to_owned(),
                        entry["symbol"].as_str().unwrap().to_owned(),
                    ),
                    (
                        entry["max_threads"].as_u64().unwrap() as u32,
                        entry["shared_static_bytes"].as_u64().unwrap() as u32,
                    ),
                );
            }
        }
    }
    let mut checked = BTreeMap::<String, Vec<String>>::new();
    let mut check = |module: &str, launch: Launch, shared| {
        let mut applied = None;
        launch_checks::configure_shared(shared, |bytes| {
            applied = Some(bytes);
            Ok(())
        })
        .unwrap();
        let (max_threads, static_shared) = resources[&(module.to_owned(), launch.symbol.clone())];
        let at = launch_checks::context(module, &launch, shared);
        launch_checks::audit(launch.plan, max_threads, static_shared, applied)
            .unwrap_or_else(|error| panic!("{at}: {error}"));
        println!(
            "TMA_CONFIG {at} max_threads={max_threads} static_shared={static_shared} applied={applied:?} PASS"
        );
        checked
            .entry(module.into())
            .or_default()
            .push(launch.symbol);
    };
    for (entry, old_module, new_module) in entries() {
        let old = Sm120::new()
            .with_mxfp8_fold(entry.clone())
            .mxfp8_fold_gemm(&entry.query, Need::rows(entry.query.m_capacity))
            .unwrap();
        let new = Sm120::new()
            .with_mxfp8_prefill(entry.clone())
            .mxfp8_prefill_gemm(&entry.query, Need::rows(entry.query.m_capacity))
            .unwrap();
        let rows = if entry.query.m_capacity == 8192 {
            vec![1024, 2048, 8192]
        } else {
            vec![1, 17, 128, 257]
        };
        for rows in rows {
            for (op, module) in [(&old, &old_module), (&new, &new_module)] {
                for launch in op
                    .launches(&views(&entry, rows), View::EMPTY, Counts::rows(rows))
                    .unwrap()
                {
                    let shared = op
                        .info()
                        .symbols
                        .iter()
                        .find(|(s, _)| s == &launch.symbol)
                        .unwrap()
                        .1;
                    if entry.query.k == 65536 && launch.symbol == tiled::symbol(&entry) {
                        let mut control = launch.clone();
                        control.symbol = tiled::control_symbol(&entry);
                        check(module, control, shared);
                    }
                    check(module, launch, shared);
                }
            }
        }
    }
    // The write eviction and gate used by time_ab have no function attributes.
    for (symbol, grid, block) in [
        ("chop_probe_gate_v1", [1, 1, 1], [32, 1, 1]),
        ("chop_probe_flush_v1", [1024, 1, 1], [256, 1, 1]),
    ] {
        check(
            "probe_sm120",
            Launch {
                symbol: symbol.into(),
                rows: 1,
                plan: kernel_core::contract::LaunchPlan {
                    grid,
                    block,
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1, 1, 1],
                    pdl: false,
                },
                arguments: kernel_core::contract::ArgumentList(vec![]),
                collective: false,
                persistent: false,
            },
            0,
        );
    }
    for symbols in checked.values_mut() {
        symbols.sort();
        symbols.dedup();
    }
    assert_eq!(checked, required_symbols());
    println!(
        "TMA_CONFIG_COMPLETE modules={} symbols={}",
        checked.len(),
        checked.values().map(Vec::len).sum::<usize>()
    );
}

fn random(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}
fn equal(a: &[u8], b: &[u8], label: &str) {
    assert_eq!(a.len(), b.len());
    if let Some(i) = a.iter().zip(b).position(|(x, y)| x != y) {
        panic!("{label}: byte {i}, old {} new {}", a[i], b[i]);
    }
}

fn configured_function(
    module: &Module,
    name: &str,
    launch: &Launch,
    shared: u32,
) -> LabResult<Function> {
    let at = launch_checks::context(name, launch, shared);
    let function = module
        .function(&launch.symbol)
        .map_err(|error| format!("{at}: resolve: {error}"))?;
    launch_checks::configure_shared(shared, |bytes| {
        function
            .set_max_dynamic_shared_memory(bytes)
            .map_err(|error| error.to_string())
    })
    .map_err(|error| format!("{at}: cuFuncSetAttribute: {error}"))?;
    Ok(function)
}

#[allow(clippy::too_many_arguments)]
fn case(
    session: &Session,
    modules: &BTreeMap<String, Module>,
    e: &Entry,
    old_module: &str,
    new_module: &str,
    rows: u32,
    live: u32,
    pattern: u32,
    timing: bool,
    negative: bool,
) -> LabResult<()> {
    let q = e.query;
    let sizes = e.bytes(rows);
    let mut seed = 0x972d34b1u32 ^ rows ^ q.n ^ q.k ^ pattern;
    let witness = q.k == 65536 && q.n == 1;
    let codes = |size: u64, seed: &mut u32| {
        (0..size)
            .map(|_| {
                let r = random(seed);
                (r % 127) as u8 | ((r >> 24) as u8 & 128)
            })
            .collect::<Vec<_>>()
    };
    let mut activation = codes(sizes.activations, &mut seed);
    let mut a_scales = (0..sizes.activation_scales)
        .map(|_| 120 + (random(&mut seed) % 11) as u8)
        .collect::<Vec<_>>();
    let mut weight = codes(sizes.weights, &mut seed);
    let mut w_scales = (0..sizes.weight_scales)
        .map(|_| 121 + (random(&mut seed) % 13) as u8)
        .collect::<Vec<_>>();
    if witness {
        activation.fill(0);
        a_scales.fill(127);
        weight.fill(0x38);
        w_scales.fill(127);
        for row in 0..rows as usize {
            activation[row * q.k as usize] = 1;
            a_scales[row * (q.k / 32) as usize] = 161;
            for k in (128..q.k as usize).step_by(32) {
                activation[row * q.k as usize + k] = 0x38;
            }
        }
    }
    let mut status = vec![0x4000u32; rows as usize + 5];
    status[3..3 + rows as usize].fill(0);
    if pattern == 2 {
        for row in 0..rows {
            let at = (row * q.activation_row_stride
                + (q.groups - 1) * q.activation_group_stride
                + q.k / 2) as usize;
            match row % 7 {
                0 => {
                    activation[at] = 127;
                }
                1 => {
                    a_scales[at / 32] = 255;
                }
                2 => {
                    activation[at] = 126;
                    a_scales[at / 32] = 254;
                }
                3 => {
                    activation[at] = 1;
                    a_scales[at / 32] = 0;
                }
                4 => {
                    status[3 + row as usize] = 0x80;
                }
                5 => {
                    activation[at] = 128;
                }
                _ => {}
            }
        }
    }
    let initial_status = status
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let images = [
        activation,
        a_scales,
        weight,
        w_scales,
        vec![0xa5; sizes.output as usize],
        live.to_le_bytes().to_vec(),
        initial_status.clone(),
    ];
    let mut buffers = Vec::new();
    for image in &images {
        let b = session.allocate(image.len())?;
        b.copy_from_bytes(image, &session.stream)
            .map_err(|e| e.to_string())?;
        buffers.push(b);
    }
    drop(images);
    let b = views(e, rows);
    let old = Sm120::new()
        .with_mxfp8_fold(e.clone())
        .mxfp8_fold_gemm(&q, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let new = Sm120::new()
        .with_mxfp8_prefill(e.clone())
        .mxfp8_prefill_gemm(&q, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let ops = [old, new];
    let plans = ops
        .iter()
        .map(|op| {
            op.launches(&b, View::EMPTY, Counts::rows(rows))
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut functions = Vec::new();
    for (arm, module) in [old_module, new_module].into_iter().enumerate() {
        functions.push(
            plans[arm]
                .iter()
                .map(|l| {
                    let shared = ops[arm]
                        .info()
                        .symbols
                        .iter()
                        .find(|(symbol, _)| symbol == &l.symbol)
                        .ok_or_else(|| format!("undeclared function {}", l.symbol))?
                        .1;
                    configured_function(&modules[module], module, l, shared)
                })
                .collect::<LabResult<Vec<_>>>()?,
        );
    }
    let bindings = Bindings {
        buffers: buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u64, b))
            .collect(),
    };
    let arm_context = |arm: usize| {
        plans[arm]
            .iter()
            .map(|l| {
                launch_checks::context(
                    [old_module, new_module][arm],
                    l,
                    l.plan.dynamic_shared_bytes,
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let fire = |arm: usize, stream: &Stream| {
        for (l, f) in plans[arm].iter().zip(&functions[arm]) {
            launch(&session.context, f, stream, l.plan, &l.arguments, &bindings).map_err(
                |error| {
                    format!(
                        "{}: {error}",
                        launch_checks::context(
                            [old_module, new_module][arm],
                            l,
                            l.plan.dynamic_shared_bytes
                        )
                    )
                },
            )?;
        }
        Ok(())
    };
    let restore = || {
        buffers[4]
            .copy_from_bytes(&vec![0xa5; sizes.output as usize], &session.stream)
            .map_err(|e| e.to_string())?;
        buffers[6]
            .copy_from_bytes(&initial_status, &session.stream)
            .map_err(|e| e.to_string())
    };
    restore()?;
    fire(0, &session.stream)?;
    let expected = kernel_harness::poison::read(session, &buffers[4])?;
    let expected_status = kernel_harness::poison::read(session, &buffers[6])?;
    let written = live as usize
        * q.groups as usize
        * q.n as usize
        * if q.output == DType::F32 { 4 } else { 2 };
    assert!(
        expected[written..].iter().all(|&b| b == 0xa5),
        "base touched dead rows"
    );
    restore()?;
    fire(1, &session.stream)?;
    equal(
        &expected,
        &kernel_harness::poison::read(session, &buffers[4])?,
        "eager output",
    );
    equal(
        &expected_status,
        &kernel_harness::poison::read(session, &buffers[6])?,
        "eager status",
    );
    let graphs = [
        session
            .capture(|s| fire(0, s))
            .map_err(|error| format!("capture {}: {error}", arm_context(0)))?,
        session
            .capture(|s| fire(1, s))
            .map_err(|error| format!("capture {}: {error}", arm_context(1)))?,
    ];
    restore()?;
    graphs[1]
        .launch(&session.stream)
        .map_err(|error| format!("replay {}: {error}", arm_context(1)))?;
    equal(
        &expected,
        &kernel_harness::poison::read(session, &buffers[4])?,
        "poisoned graph output",
    );
    equal(
        &expected_status,
        &kernel_harness::poison::read(session, &buffers[6])?,
        "poisoned graph status",
    );
    if negative {
        restore()?;
        let mut ls = plans[1].clone();
        ls[1].symbol = tiled::control_symbol(e);
        for l in ls {
            let f = configured_function(
                &modules[new_module],
                new_module,
                &l,
                l.plan.dynamic_shared_bytes,
            )?;
            launch(
                &session.context,
                &f,
                &session.stream,
                l.plan,
                &l.arguments,
                &bindings,
            )
            .map_err(|error| {
                format!(
                    "{}: {error}",
                    launch_checks::context(new_module, &l, l.plan.dynamic_shared_bytes)
                )
            })?;
        }
        assert!(
            expected != kernel_harness::poison::read(session, &buffers[4])?,
            "persistent-chain control survived"
        );
        println!("TMA_NEGATIVE {} PASS", e.symbol);
    }
    println!(
        "TMA_BITWISE {} rows={rows} live={live} pattern={pattern} PASS",
        e.symbol
    );
    if timing {
        restore()?;
        let pair = kernel_harness::time_ab(
            session,
            1,
            5,
            [kernel_harness::eviction::EvictionMode::Write; 2],
            |arm, stream| graphs[arm].launch(stream)
                .map_err(|error| format!("timed replay {}: {error}", arm_context(arm))),
        ).map_err(|error| format!("timing old=[{}] new=[{}]; probe gate/flush dynamic_shared=0 cluster=[1,1,1]: {error}", arm_context(0), arm_context(1)))?;
        let flops = 2.0 * f64::from(rows) * f64::from(q.n) * f64::from(q.k) * f64::from(q.groups);
        println!(
            "TMA_TIMING {} rows={rows} old_ms={} new_ms={} ratio={} tflops={} floor_ms={}",
            e.symbol,
            pair.arms[0].median_ms,
            pair.arms[1].median_ms,
            pair.paired_ratio,
            flops / (f64::from(pair.arms[1].median_ms) * 1e9),
            flops / 400e9
        );
    }
    Ok(())
}

#[test]
fn prefill_tma_matches_every_declared_shape_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("prefill comparison: select idle GPUs").unwrap();
        return;
    }
    let probe = std::fs::read(root().join("probe_sm120.cubin"))
        .unwrap()
        .leak();
    let bytes = Box::leak(Box::new([("probe_sm120", &*probe)]));
    let cubins = Cubins::new(
        bytes,
        &[(
            "probe_sm120",
            &["chop_probe_gate_v1", "chop_probe_flush_v1"],
        )],
    );
    let result = kernel_harness::gpu_test_with(cubins, 4 << 30, |session| {
        let mut modules = BTreeMap::new();
        for name in required_symbols().keys().filter(|n| *n != "probe_sm120") {
            modules.insert(
                name.clone(),
                session.load_cubin(
                    &std::fs::read(root().join(format!("{name}.cubin")))
                        .map_err(|e| e.to_string())?,
                )?,
            );
        }
        let (mut count, mut timings, mut negative) = (0, 0, 0);
        for (e, old, new) in models() {
            for rows in [1024, 2048, 8192] {
                case(
                    session, &modules, &e, &old, &new, rows, rows, 0, true, false,
                )
                .map_err(|error| format!("shape={:?} rows={rows} live={rows}: {error}", e.query))?;
                count += 1;
                timings += 1;
            }
            case(
                session, &modules, &e, &old, &new, 1024, 1023, 1, false, false,
            )
            .map_err(|error| format!("shape={:?} rows=1024 live=1023: {error}", e.query))?;
            count += 1;
        }
        for e in kernel_lab_families::mxfp8_prefill::entries() {
            for (rows, live, pattern) in [
                (1, 1, 0),
                (17, 16, 0),
                (128, 128, 0),
                (257, 129, 2),
                (257, 0, 2),
            ] {
                let control = e.query.k == 65536 && e.query.output == DType::F32 && rows == 128;
                case(
                    session,
                    &modules,
                    &e,
                    kernel_lab_families::mxfp8_prefill::CUBIN,
                    kernel_lab_families::mxfp8_prefill::CUBIN,
                    rows,
                    live,
                    pattern,
                    false,
                    control,
                )
                .map_err(|error| format!("shape={:?} rows={rows} live={live}: {error}", e.query))?;
                count += 1;
                negative += usize::from(control);
            }
        }
        assert_eq!(
            count,
            models().len() * 4 + kernel_lab_families::mxfp8_prefill::entries().len() * 5
        );
        assert_eq!(negative, 1);
        assert_eq!(timings, models().len() * 3);
        println!("TMA_COMPLETE bitwise={count} negative={negative} timings={timings}");
        Ok(())
    })
    .unwrap();
    assert!(result.is_some(), "required prefill comparison skipped");
}
