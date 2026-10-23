//! The cluster split-K folds (`cluster_family`) against the served fold of
//! the same projection and row count: the bundle's selector and cubins, whose
//! lock records equal main's (a cubin digest per served module is printed
//! with every case). Inputs are the captured layer-0 operands of the shape
//! (the 32-row verify capture or its first rows; synthetic where the
//! captures hold no such shape), with main's captured outputs checked at 32
//! rows.
//!
//! `CHOPINFER_FOLD_CLUSTER_MODE`:
//! - `check`: every row count up to the entry's row tile, bit for bit
//!   (outputs and status) at full, short with rejected rows, empty and full
//!   live counts, eagerly and as a graph; the omitted-first-fold control must
//!   differ and the delayed admission must still clear rejected rows. Then
//!   every captured input of the shape the row tile holds (decode-8 layers 0
//!   and 39 at 1 and 8 rows, verify-32 layers 0 and 39 at 32 rows), each
//!   against main's captured output as well.
//! - `memcheck`: `check` at the entry's row tile only.
//! - `racecheck`: full and short live counts, eagerly, at the row tile.
//! - `time`: `check` at the row tile, then served/candidate graph replays,
//!   warm and cold, and the candidate's resident clusters.
//!
//! `CHOPINFER_FOLD_CLUSTER_VARIANT` keeps one label (`cluster_family::label`)
//! and `CHOPINFER_FOLD_CLUSTER_ROWS` the entries of one row tile.
use super::*;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Check,
    Memcheck,
    Racecheck,
    Time,
}

fn mode() -> LabResult<Mode> {
    Ok(
        match std::env::var("CHOPINFER_FOLD_CLUSTER_MODE")
            .map_err(|e| e.to_string())?
            .as_str()
        {
            "check" => Mode::Check,
            "memcheck" => Mode::Memcheck,
            "racecheck" => Mode::Racecheck,
            "time" => Mode::Time,
            _ => {
                return Err(
                    "CHOPINFER_FOLD_CLUSTER_MODE: check, memcheck, racecheck or time".into(),
                );
            }
        },
    )
}

type Entry = kernels_sm120::gemm::mxfp8_fold_cluster::Entry;

pub fn entries() -> Vec<Entry> {
    let kept = std::env::var("CHOPINFER_FOLD_CLUSTER_VARIANT").ok();
    let rows = std::env::var("CHOPINFER_FOLD_CLUSTER_ROWS")
        .ok()
        .map(|v| v.parse::<u32>().expect("CHOPINFER_FOLD_CLUSTER_ROWS"));
    cluster_family::entries()
        .into_iter()
        .filter(|e| kept.as_ref().is_none_or(|k| *k == cluster_family::label(e)))
        .filter(|e| rows.is_none_or(|r| r == e.tile_rows))
        .collect()
}

/// The served projection base of a candidate's shape.
fn base_of(e: &Entry) -> mxfp8_fold::Entry {
    let q = e.base.query;
    family::bases()
        .into_iter()
        .find(|b| (b.query.n, b.query.k, b.query.groups) == (q.n, q.k, q.groups))
        .expect("served base of a cluster shape")
}

fn row_counts(e: &Entry, mode: Mode) -> Vec<u32> {
    if mode == Mode::Check {
        (1..=e.tile_rows).collect()
    } else {
        vec![e.tile_rows]
    }
}

/// The projection a shape serves in an attention sublayer.
fn projection(base: &mxfp8_fold::Entry) -> &'static str {
    match (base.query.n, base.query.k, base.query.groups) {
        (1280, 5120, 1) => "attention.q-a",
        (8192, 1280, 1) => "attention.q-b",
        (512, 5120, 1) => "attention.kv",
        (1024, 4096, 2) => "attention.o-a",
        (5120, 2048, 1) => "attention.o-b",
        _ => panic!("no attention projection of this shape"),
    }
}

/// The served calls of a shape at `rows`, as DenseOps lowers a backbone
/// layer's decode or verify fold: the bundle's row-local fold where its
/// table names one, otherwise the selected fold (`reference`).
fn served_calls(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Calls> {
    if let Some(e) = sm120::fold_local_rows::served(projection(base), rows) {
        let work = a.filled(e.workspace(), 0)?;
        return Ok(Calls {
            init: vec![],
            run: e.launches(b, work, rows).map_err(|e| e.to_string())?,
            state: None,
        });
    }
    reference(a, base, b, rows)
}

/// Functions of `calls`: a bundle cubin's where the bundle holds the
/// symbol, otherwise the cluster family's.
fn load(s: &Session, calls: &[Launch]) -> LabResult<Functions> {
    struct Cache(std::cell::RefCell<BTreeMap<String, Module>>);
    let cache = s.state(|| Ok(Cache(std::cell::RefCell::new(BTreeMap::new()))))?;
    let mut loaded = cache.0.borrow_mut();
    let mut map = BTreeMap::new();
    for call in calls {
        if map.contains_key(&call.symbol) {
            continue;
        }
        let (name, bytes) = match fold_cubin_for_symbol(&call.symbol) {
            Some((name, bytes)) => (name.to_owned(), bytes.to_vec()),
            None if call.symbol.starts_with("chop_fold_cluster_") => {
                let name = cluster_family::CUBIN.to_owned();
                let bytes = std::fs::read(root().join("cubins").join(format!("{name}.cubin")))
                    .map_err(|e| format!("{name}: {e}"))?;
                (name, bytes)
            }
            None => return Err(format!("no cubin holds {}", call.symbol)),
        };
        if !loaded.contains_key(&name) {
            loaded.insert(name.clone(), s.load_cubin(&bytes)?);
        }
        let f = loaded[&name]
            .function(&call.symbol)
            .map_err(|e| e.to_string())?;
        f.set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
        map.insert(call.symbol.clone(), f);
    }
    Ok(Functions {
        map,
        _modules: Vec::new(),
    })
}

fn candidate_calls(a: &mut Arena<'_>, e: &Entry, b: &Buffers<View>, rows: u32) -> LabResult<Calls> {
    let work = a.filled(e.workspace(), 0)?;
    Ok(Calls {
        init: vec![],
        run: e.launches(b, work, rows).map_err(|e| e.to_string())?,
        state: None,
    })
}

fn renamed(calls: &[Launch], index: usize, suffix: &str) -> Vec<Launch> {
    let mut v = calls.to_vec();
    v[index].symbol.push_str(suffix);
    v
}

fn eager(calls: &[Launch]) -> Vec<Launch> {
    let mut v = calls.to_vec();
    for c in &mut v {
        c.plan.pdl = false;
    }
    v
}

/// The digest of every bundle cubin `calls` load.
fn served_modules(calls: &Calls) -> Value {
    let mut map = serde_json::Map::new();
    for c in calls.init.iter().chain(&calls.run) {
        if let Some((name, bytes)) = fold_cubin_for_symbol(&c.symbol) {
            map.insert(name.into(), json!(kernel_build::digest(bytes)));
        }
    }
    Value::Object(map)
}

fn case(s: &Session, e: &Entry, input: &data::Input, mode: Mode) -> LabResult<Value> {
    let base = base_of(e);
    let rows = input.rows;
    let mut a = Arena::new(Some(s));
    let b = [
        common(&mut a, &base, rows, input)?,
        common(&mut a, &base, rows, input)?,
    ];
    let calls = [
        served_calls(&mut a, &base, &b[0], rows)?,
        candidate_calls(&mut a, e, &b[1], rows)?,
    ];
    let omit = renamed(&calls[1].run, 1, "_omit_v1");
    let delayed = renamed(&calls[1].run, 0, "_delayed_v1");
    let all = calls
        .iter()
        .flat_map(|c| c.init.iter().chain(&c.run))
        .chain(&omit)
        .chain(&delayed)
        .cloned()
        .collect::<Vec<_>>();
    let f = load(s, &all)?;
    issue(s, &a, &f, &calls[0].init, &s.stream)?;
    let mut record = json!({"label":cluster_family::label(e),"n":input.n,"k":input.k,"groups":input.groups,
        "rows":rows,"source":input.id,"captured":input.expected.is_some(),
        "served":calls[0].run.iter().map(|c| &c.symbol).collect::<Vec<_>>(),
        "served_modules":served_modules(&calls[0]),
        "candidate":calls[1].run.iter().map(|c| json!({"symbol":c.symbol,"grid":c.plan.grid,
            "block":c.plan.block,"cluster":c.plan.cluster})).collect::<Vec<_>>()});
    let configurations: Vec<(u32, bool, bool)> = if mode == Mode::Racecheck {
        vec![(rows, false, false), (rows.saturating_sub(1), true, false)]
    } else {
        [
            (rows, false),
            (rows.saturating_sub(1), true),
            (0, false),
            (rows, false),
        ]
        .into_iter()
        .flat_map(|(live, bad)| [(live, bad, false), (live, bad, true)])
        .collect()
    };
    let graphs = if mode == Mode::Racecheck {
        None
    } else {
        Some(
            calls
                .iter()
                .map(|c| s.capture(|stream| issue(s, &a, &f, &c.run, stream)))
                .collect::<LabResult<Vec<_>>>()?,
        )
    };
    let mut comparisons = 0;
    for &(live, bad, graph) in &configurations {
        for arm in 0..2 {
            reset(&a, &b[arm], input, rows, live, bad)?;
            match (&graphs, graph) {
                (Some(g), true) => g[arm].launch(&s.stream).map_err(|e| e.to_string())?,
                _ => issue(s, &a, &f, &eager(&calls[arm].run), &s.stream)?,
            }
        }
        compare(&a, &b, input, rows, live, bad, None).map_err(|err| {
            format!(
                "{} rows {rows} live {live} bad {bad} graph {graph}: {err}",
                cluster_family::label(e)
            )
        })?;
        comparisons += 1;
    }
    record["comparisons"] = json!(comparisons);
    record["bitwise"] = json!(true);
    if mode == Mode::Racecheck {
        return Ok(record);
    }
    // Omitting the first fold must change the result; admission delayed past
    // the fold's start must still clear the rejected rows.
    reset(&a, &b[1], input, rows, rows, false)?;
    issue(s, &a, &f, &eager(&omit), &s.stream)?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    reset(&a, &b[0], input, rows, rows, false)?;
    issue(s, &a, &f, &eager(&calls[0].run), &s.stream)?;
    if a.read(b[0].output)? == a.read(b[1].output)? {
        return Err(format!(
            "{}: omit-first control was ineffective",
            cluster_family::label(e)
        ));
    }
    for (arm, buffer) in b.iter().enumerate() {
        reset(&a, buffer, input, rows, rows, true)?;
        issue(
            s,
            &a,
            &f,
            if arm == 0 { &calls[0].run } else { &delayed },
            &s.stream,
        )?;
    }
    compare(&a, &b, input, rows, rows, true, None)
        .map_err(|err| format!("{} delayed admission: {err}", cluster_family::label(e)))?;
    record["negative"] = json!(true);
    if mode == Mode::Time
        && let Some(graphs) = &graphs
    {
        for buffer in &b {
            reset(&a, buffer, input, rows, rows, false)?;
        }
        for (name, cold) in [("warm", false), ("cold", true)] {
            record[name] = timing::pair(s, cold, |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
        }
        let run = &calls[1].run[1];
        let config = runtime_cuda::LaunchConfig {
            grid: run.plan.grid.into(),
            block: run.plan.block.into(),
            shared_memory_bytes: run.plan.dynamic_shared_bytes,
            cluster: run.plan.cluster.into(),
            programmatic_stream_serialization: true,
        };
        let clusters = f.map[&run.symbol]
            .max_active_clusters(config, false)
            .map_err(|e| e.to_string())?;
        let launched =
            run.plan.grid.iter().product::<u32>() / run.plan.cluster.iter().product::<u32>();
        record["clusters"] =
            json!({"launched":launched,"resident":clusters,"one_wave":clusters >= launched});
    }
    Ok(record)
}

pub fn gpu(s: &Session) -> LabResult<()> {
    let mode = mode()?;
    let mut cases = 0;
    for e in entries() {
        let base = base_of(&e);
        let mut inputs = Vec::new();
        for rows in row_counts(&e, mode) {
            inputs.push(profile::input(&base, rows)?);
        }
        if mode == Mode::Check {
            inputs.extend(
                data::captured(&base, true)?
                    .into_iter()
                    .filter(|i| i.rows <= e.tile_rows),
            );
        }
        for input in &inputs {
            let record = case(s, &e, input, mode)?;
            println!("FOLD_CLUSTER_CASE {record}");
            cases += 1;
        }
    }
    if cases == 0 {
        return Err("no cluster fold case ran".into());
    }
    println!(
        "FOLD_CLUSTER_DONE {}",
        json!({"mode":std::env::var("CHOPINFER_FOLD_CLUSTER_MODE").unwrap_or_default(),"cases":cases})
    );
    Ok(())
}

/// Every case's served and candidate calls without opening CUDA: the plan
/// the GPU run must reproduce, with the expected case count.
#[test]
fn cluster_catalog() {
    let mut rows = Vec::new();
    for e in cluster_family::entries() {
        e.validate().unwrap();
        let base = base_of(&e);
        for r in 1..=e.tile_rows {
            let mut a = Arena::new(None);
            let input = data::empty(&base, r);
            let b = common(&mut a, &base, r, &input).unwrap();
            let served = served_calls(&mut a, &base, &b, r).unwrap();
            let candidate = candidate_calls(&mut a, &e, &b, r).unwrap();
            for c in served.init.iter().chain(&served.run).chain(&candidate.run) {
                c.plan.validate_geometry().unwrap();
            }
            assert!(
                served
                    .run
                    .iter()
                    .all(|c| fold_cubin_for_symbol(&c.symbol).is_some())
            );
            assert_eq!(candidate.run.len(), 2);
            assert_eq!(candidate.run[1].plan.cluster, [1, 1, e.splits]);
            rows.push(json!({"label":cluster_family::label(&e),"n":base.query.n,"k":base.query.k,
                "groups":base.query.groups,"tile_rows":e.tile_rows,"rows":r,
                "served":served.run.iter().map(|c| &c.symbol).collect::<Vec<_>>(),
                "served_modules":served_modules(&served),
                "candidate":candidate.run.iter().map(|c| json!({"symbol":c.symbol,"grid":c.plan.grid,
                    "cluster":c.plan.cluster})).collect::<Vec<_>>()}));
        }
    }
    let check: u32 = cluster_family::entries().iter().map(|e| e.tile_rows).sum();
    assert_eq!(rows.len() as u32, check);
    println!(
        "FOLD_CLUSTER_CATALOG {}",
        json!({"family":cluster_family::FAMILY,"cubin":cluster_family::CUBIN,"entries":cluster_family::entries().len(),
            "check_cases":check,"tile_cases":cluster_family::entries().len(),"cases":rows})
    );
}

#[test]
#[ignore = "requires CHOPINFER_FOLD_CORE_ROOT, CHOPINFER_FOLD_CLUSTER_MODE and assigned idle GPUs"]
fn gpu_cluster() {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    static TABLE: OnceLock<Vec<(&'static str, &'static [u8])>> = OnceLock::new();
    let bytes =
        BYTES.get_or_init(|| std::fs::read(root().join("cubins/probe_sm120.cubin")).unwrap());
    let cubins = kernel_harness::Cubins::new(
        TABLE.get_or_init(|| vec![("probe_sm120", bytes)]),
        &[(
            "probe_sm120",
            &["chop_probe_read_v1", "chop_probe_flush_v1"],
        )],
    );
    assert!(
        kernel_harness::gpu_test_with(cubins, 6 << 30, gpu)
            .unwrap()
            .is_some()
    );
}
