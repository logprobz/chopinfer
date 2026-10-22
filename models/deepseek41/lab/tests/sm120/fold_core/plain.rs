//! The served dense folds with swizzled shared rows (the bundle's cubins)
//! against main's served cubins (staged as `main_<name>`, digests checked
//! against main's lock at staging) and against the same entries of this
//! tree built with the unswizzled placement (`plain_family`,
//! `CHOP_FOLD_PLAIN_ROWS`), on identical inputs. Cases are the Flash TP4
//! projections decode lowers singly (`profile::bases`) and the shared
//! gate/up pair, at every decode row bucket; the 32-row WQ_b bucket has two
//! cases, the row-local fold of AR decode and the fold of verify. A case
//! whose calls all load an unswizzled module is recorded without GPU work:
//! every arm would run the same cubin.
//!
//! `CHOPINFER_FOLD_PLAIN_MODE`:
//! - `check`: every changed case bit for bit, plain and served against
//!   main, at full, short with rejected rows, empty and full live counts,
//!   eagerly and as a graph.
//! - `memcheck`: `check` at 16 and 32 rows.
//! - `racecheck`: full and short live counts, eagerly, at 16 and 32 rows.
//! - `time`: `check`, then main/served and plain/served graph timing, warm
//!   and cold, and each swizzled deep ring of the shape (`deep_family`) bit
//!   for bit against the served output and timed served/deep.
//! - `eager`: each arm and deep ring launched once without programmatic
//!   serialization, for a launch-filtered profiler, in the printed order.
//!
//! `CHOPINFER_FOLD_PLAIN_ROWS` (a comma list) keeps those row counts.
use super::*;

pub const MAX_ROWS: u32 = 64;
const TIMED_ROWS: [u32; 2] = [16, 32];

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Check,
    Memcheck,
    Racecheck,
    Time,
    Eager,
}

fn mode() -> LabResult<Mode> {
    Ok(
        match std::env::var("CHOPINFER_FOLD_PLAIN_MODE")
            .map_err(|e| e.to_string())?
            .as_str()
        {
            "check" => Mode::Check,
            "memcheck" => Mode::Memcheck,
            "racecheck" => Mode::Racecheck,
            "time" => Mode::Time,
            "eager" => Mode::Eager,
            _ => {
                return Err(
                    "CHOPINFER_FOLD_PLAIN_MODE: check, memcheck, racecheck, time or eager".into(),
                );
            }
        },
    )
}

fn rows_of(mode: Mode) -> LabResult<Vec<u32>> {
    let all: Vec<u32> = if mode == Mode::Check {
        (1..=MAX_ROWS).collect()
    } else {
        TIMED_ROWS.to_vec()
    };
    match std::env::var("CHOPINFER_FOLD_PLAIN_ROWS") {
        Ok(list) => {
            let kept = list
                .split(',')
                .map(|v| v.trim().parse::<u32>().map_err(|e| e.to_string()))
                .collect::<LabResult<Vec<_>>>()?;
            if kept.is_empty() || kept.iter().any(|r| !all.contains(r)) {
                return Err("CHOPINFER_FOLD_PLAIN_ROWS outside the mode's rows".into());
            }
            Ok(kept)
        }
        Err(_) => Ok(all),
    }
}

/// One served projection: a single fold or the shared gate/up pair, and
/// for WQ_b at 32 rows the AR decode route (`local`) or the verify route.
#[derive(Clone)]
pub struct Shape {
    pub base: mxfp8_fold::Entry,
    pub paired: bool,
}

pub fn shapes() -> Vec<Shape> {
    let mut all: Vec<_> = profile::bases()
        .into_iter()
        .map(|base| Shape {
            base,
            paired: false,
        })
        .collect();
    let gate = family::bases()
        .into_iter()
        .find(|b| (b.query.n, b.query.k, b.query.groups) == (576, 5120, 1))
        .expect("shared gate/up base");
    all.push(Shape {
        base: gate,
        paired: true,
    });
    all
}

/// The routes of `shape` at `rows`: "verify" (the general decode fold) and,
/// for the AR decode bucket the bundle serves row-locally, "decode".
pub fn routes(shape: &Shape, rows: u32) -> Vec<&'static str> {
    let q = shape.base.query;
    if !shape.paired && rows == 32 && (q.n, q.k, q.groups) == (8192, 1280, 1) {
        vec!["decode", "verify"]
    } else {
        vec!["verify"]
    }
}

fn label(shape: &Shape) -> String {
    let q = shape.base.query;
    format!(
        "n{}_k{}_g{}{}",
        q.n,
        q.k,
        q.groups,
        if shape.paired { "_pair" } else { "" }
    )
}

/// One arm's buffers: the first bank and, for a pair, the second bank that
/// shares its activations, live rows and status.
#[derive(Clone, Copy)]
struct Banks {
    first: Buffers<View>,
    second: Option<Buffers<View>>,
}

struct Inputs {
    first: data::Input,
    second: Option<data::Input>,
}

fn inputs(shape: &Shape, rows: u32) -> LabResult<Inputs> {
    let first = if rows <= 32 {
        profile::input(&shape.base, rows)?
    } else {
        data::synthetic(&shape.base, rows)
    };
    let second = shape.paired.then(|| {
        let mut second = data::synthetic(&shape.base, rows);
        (second.weights, second.weight_scales) = data::second_bank(&shape.base);
        second.codes = first.codes.clone();
        second.scales = first.scales.clone();
        second.id = "synthetic-second".into();
        second
    });
    Ok(Inputs { first, second })
}

/// Arm buffers that read `shared`'s immutable weights when given.
fn banks(
    a: &mut Arena<'_>,
    shape: &Shape,
    rows: u32,
    input: &Inputs,
    shared: Option<&Banks>,
) -> LabResult<Banks> {
    let first = common_with(a, &shape.base, rows, &input.first, shared.map(|s| &s.first))?;
    let second = match &input.second {
        None => None,
        Some(second) => {
            let size = shape.base.bytes(rows);
            let (weights, weight_scales) = match shared.and_then(|s| s.second) {
                Some(s) => (s.weights, s.weight_scales),
                None if a.session.is_some() => {
                    (a.bytes(&second.weights)?, a.bytes(&second.weight_scales)?)
                }
                None => (a.filled(size.weights, 0)?, a.filled(size.weight_scales, 0)?),
            };
            Some(Buffers {
                weights,
                weight_scales,
                output: a.filled(size.output + 64, 255)?,
                ..first
            })
        }
    };
    Ok(Banks { first, second })
}

/// `common`, reusing another arm's read-only weights.
fn common_with(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    rows: u32,
    input: &data::Input,
    shared: Option<&Buffers<View>>,
) -> LabResult<Buffers<View>> {
    let Some(shared) = shared else {
        return common(a, base, rows, input);
    };
    let size = base.bytes(rows);
    Ok(Buffers {
        activations: if a.session.is_some() {
            a.bytes(&input.codes)?
        } else {
            a.filled(size.activations, 0)?
        },
        activation_scales: if a.session.is_some() {
            a.bytes(&input.scales)?
        } else {
            a.filled(size.activation_scales, 0)?
        },
        weights: shared.weights,
        weight_scales: shared.weight_scales,
        output: a.filled(size.output + 64, 255)?,
        live_rows: a.bytes(&rows.to_le_bytes())?,
        status: a.filled(size.status + 20, 0)?,
        status_row_offset: 3,
    })
}

/// The served calls of `shape` at `rows` on `route`, as DenseOps lowers
/// them: the stream fold of a qualified bucket, otherwise the decode fold,
/// and the row-local fold for WQ_b on AR decode at 32 rows.
fn served(a: &mut Arena<'_>, shape: &Shape, b: &Banks, rows: u32, route: &str) -> LabResult<Calls> {
    if !shape.paired {
        return if route == "decode" {
            profile::served(a, &shape.base, &b.first, rows)
        } else {
            reference(a, &shape.base, &b.first, rows)
        };
    }
    let second = b.second.ok_or("pair without a second bank")?;
    let backend = backend();
    let q = sm120::mxfp8_fold::routed_query(shape.base.query, rows);
    if let Ok(op) = backend.mxfp8_fold_gemm_stream(&q, true, Need::rows(rows)) {
        let state = a.filled(
            op.info()
                .bytes_of("state")
                .ok_or("missing paired stream state")?,
            0,
        )?;
        let packed = [
            a.filled(b.first.weights.bytes, 0)?,
            a.filled(second.weights.bytes, 0)?,
        ];
        let views = Mxfp8FoldStreamBuffers {
            first: b.first,
            second: Some(second),
            packed_weights: packed,
            state,
        };
        let work = a.filled(op.info().workspace.max(4), 0)?;
        return Ok(Calls {
            init: op.initialize(&views).map_err(|e| e.to_string())?,
            run: op
                .launches(&views, work, Counts::rows(rows))
                .map_err(|e| e.to_string())?,
            state: None,
        });
    }
    let op = backend
        .mxfp8_fold_gemm_pair(&q, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let work = a.filled(op.info().workspace.max(4), 0)?;
    Ok(Calls {
        init: vec![],
        run: op
            .launches(&[b.first, second], work, Counts::rows(rows))
            .map_err(|e| e.to_string())?,
        state: None,
    })
}

/// The plain family's cubin names.
fn plain_names() -> &'static std::collections::BTreeSet<String> {
    static NAMES: OnceLock<std::collections::BTreeSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| plain_family::cubins().into_iter().map(|s| s.name).collect())
}

/// The arms in comparison order: main's served cubins (staged as
/// `main_<name>`, the reference), the plain rebuild and the swizzled served
/// cubins of this tree.
pub const ARMS: [&str; 3] = ["main", "plain", "served"];
const MAIN: usize = 0;
const SERVED: usize = 2;

/// The served module of `symbol` and whether this change swizzled it (the
/// plain family rebuilds it).
pub fn modules(symbol: &str) -> LabResult<(&'static str, bool)> {
    let (name, _) =
        fold_cubin_for_symbol(symbol).ok_or_else(|| format!("missing served {symbol}"))?;
    let plain = format!("{}{name}", plain_family::PREFIX);
    Ok((name, plain_names().contains(&plain)))
}

/// The module arm `arm` loads for `symbol`: a swizzled cubin's main or plain
/// copy, otherwise the served cubin itself.
pub fn arm_module(symbol: &str, arm: usize) -> LabResult<String> {
    if symbol.starts_with("chop_fold_deep_") {
        return Ok(deep_family::CUBIN.to_owned());
    }
    let (served, swizzled) = modules(symbol)?;
    Ok(match (arm, swizzled) {
        (MAIN, true) => format!("main_{served}"),
        (1, true) => format!("{}{served}", plain_family::PREFIX),
        _ => served.to_owned(),
    })
}

/// Arm `arm`'s functions of `calls`.
fn load(s: &Session, calls: &[Launch], arm: usize) -> LabResult<Functions> {
    struct Cache(std::cell::RefCell<BTreeMap<String, Module>>);
    let cache = s.state(|| Ok(Cache(std::cell::RefCell::new(BTreeMap::new()))))?;
    let mut loaded = cache.0.borrow_mut();
    let mut map = BTreeMap::new();
    for call in calls {
        if map.contains_key(&call.symbol) {
            continue;
        }
        let name = arm_module(&call.symbol, arm)?;
        if !loaded.contains_key(&name) {
            let bytes = if fold_cubin_for_symbol(&call.symbol).is_some_and(|(n, _)| n == name) {
                fold_cubin_for_symbol(&call.symbol).unwrap().1.to_vec()
            } else {
                std::fs::read(root().join("cubins").join(format!("{name}.cubin")))
                    .map_err(|e| format!("{name}: {e}"))?
            };
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

fn reset_banks(
    a: &Arena<'_>,
    b: &Banks,
    input: &Inputs,
    rows: u32,
    live: u32,
    bad: bool,
) -> LabResult<()> {
    reset(a, &b.first, &input.first, rows, live, bad)?;
    if let Some(second) = b.second {
        a.fill(second.output, 255)?;
    }
    Ok(())
}

/// `other` against `reference`, both banks: outputs and status bit for
/// bit, guards, cleared rejected rows and captured outputs.
fn compare_banks(
    a: &Arena<'_>,
    reference: &Banks,
    other: &Banks,
    input: &Inputs,
    rows: u32,
    live: u32,
    bad: bool,
) -> LabResult<()> {
    compare(
        a,
        &[reference.first, other.first],
        &input.first,
        rows,
        live,
        bad,
        None,
    )?;
    if let (Some(x), Some(y), Some(second)) = (reference.second, other.second, &input.second) {
        compare(a, &[x, y], second, rows, live, bad, None)
            .map_err(|e| format!("second bank: {e}"))?;
    }
    Ok(())
}

fn without_pdl(calls: &[Launch]) -> Vec<Launch> {
    let mut plain = calls.to_vec();
    for op in &mut plain {
        op.plan.pdl = false;
    }
    plain
}

/// Whether some call of `calls` runs a swizzled module.
fn changed(calls: &[Launch]) -> LabResult<bool> {
    for call in calls {
        if modules(&call.symbol)?.1 {
            return Ok(true);
        }
    }
    Ok(false)
}

fn symbols(calls: &Calls) -> Vec<String> {
    calls
        .init
        .iter()
        .chain(&calls.run)
        .map(|c| c.symbol.clone())
        .collect()
}

fn case(s: &Session, shape: &Shape, rows: u32, route: &str, mode: Mode) -> LabResult<Value> {
    let input = inputs(shape, rows)?;
    let mut a = Arena::new(Some(s));
    let first = banks(&mut a, shape, rows, &input, None)?;
    let b = [
        first,
        banks(&mut a, shape, rows, &input, Some(&first))?,
        banks(&mut a, shape, rows, &input, Some(&first))?,
    ];
    let calls = [
        served(&mut a, shape, &b[0], rows, route)?,
        served(&mut a, shape, &b[1], rows, route)?,
        served(&mut a, shape, &b[2], rows, route)?,
    ];
    let names = symbols(&calls[SERVED]);
    if calls.iter().any(|c| symbols(c) != names) {
        return Err("arms selected different calls".into());
    }
    let all: Vec<Launch> = calls[SERVED]
        .init
        .iter()
        .chain(&calls[SERVED].run)
        .cloned()
        .collect();
    let mut record = json!({"shape":label(shape),"n":shape.base.query.n,"k":shape.base.query.k,
        "groups":shape.base.query.groups,"paired":shape.paired,"rows":rows,"route":route,
        "source":input.first.id,"captured":input.first.expected.is_some(),"symbols":names,
        "modules":all.iter().map(|c| (0..3).map(|arm| arm_module(&c.symbol, arm)).collect::<LabResult<Vec<_>>>()).collect::<LabResult<Vec<_>>>()?});
    if !changed(&all)? {
        record["same_modules"] = json!(true);
        return Ok(record);
    }
    record["same_modules"] = json!(false);
    let f = [load(s, &all, 0)?, load(s, &all, 1)?, load(s, &all, 2)?];
    // The launch order a launch-filtered profiler sees, by arm.
    let mut order = Vec::new();
    for arm in 0..3 {
        issue(s, &a, &f[arm], &calls[arm].init, &s.stream)?;
        let name = format!("{}-init", ARMS[arm]);
        order.extend(calls[arm].init.iter().map(|c| json!([name, c.symbol])));
    }
    if mode == Mode::Eager {
        let deep = deep_calls_of(&mut a, shape, rows, &b[SERVED], route)?;
        let df = load(
            s,
            &deep
                .iter()
                .flat_map(|(_, c)| c.run.clone())
                .collect::<Vec<_>>(),
            SERVED,
        )?;
        for arm in 0..3 {
            reset_banks(&a, &b[arm], &input, rows, rows, false)?;
            issue(s, &a, &f[arm], &without_pdl(&calls[arm].run), &s.stream)?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            order.extend(calls[arm].run.iter().map(|c| json!([ARMS[arm], c.symbol])));
        }
        for (name, c) in &deep {
            issue(s, &a, &df, &without_pdl(&c.run), &s.stream)?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            for call in &c.run {
                order.push(json!([name, call.symbol]));
            }
        }
        record["eager"] = json!(order);
        return Ok(record);
    }
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
            (0..3)
                .map(|arm| s.capture(|stream| issue(s, &a, &f[arm], &calls[arm].run, stream)))
                .collect::<LabResult<Vec<_>>>()?,
        )
    };
    let mut comparisons = 0;
    for &(live, bad, graph) in &configurations {
        for (arm, banks) in b.iter().enumerate() {
            reset_banks(&a, banks, &input, rows, live, bad)?;
            match (&graphs, graph) {
                (Some(g), true) => g[arm].launch(&s.stream).map_err(|e| e.to_string())?,
                _ => issue(s, &a, &f[arm], &without_pdl(&calls[arm].run), &s.stream)?,
            }
        }
        for arm in 1..3 {
            compare_banks(&a, &b[MAIN], &b[arm], &input, rows, live, bad)
                .map_err(|e| format!("{} rows {rows} {route} {}: {e}", label(shape), ARMS[arm]))?;
        }
        comparisons += 1;
    }
    record["comparisons"] = json!(comparisons);
    record["bitwise"] = json!(true);
    if mode == Mode::Time
        && let Some(graphs) = &graphs
    {
        for banks in &b {
            reset_banks(&a, banks, &input, rows, rows, false)?;
        }
        // Main/served and plain/served graph replays, warm and cold.
        for (key, reference) in [("main_served", MAIN), ("plain_served", 1)] {
            record[key] = json!({});
            for (name, cold) in [("warm", false), ("cold", true)] {
                record[key][name] = timing::pair(s, cold, |arm, stream| {
                    graphs[if arm == 0 { reference } else { SERVED }]
                        .launch(stream)
                        .map_err(|e| e.to_string())
                })?;
            }
        }
        // Each swizzled deep ring on the served arm's inputs, bit for bit
        // against the served output, timed served/deep.
        let deep = deep_calls_of(&mut a, shape, rows, &b[SERVED], route)?;
        record["deep"] = json!({});
        if !deep.is_empty() {
            reset_banks(&a, &b[SERVED], &input, rows, rows, false)?;
            graphs[SERVED]
                .launch(&s.stream)
                .map_err(|e| e.to_string())?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let served_out = a.read(b[SERVED].first.output)?;
            for (name, c) in &deep {
                let df = load(s, &c.run, SERVED)?;
                let deep_graph = s.capture(|stream| issue(s, &a, &df, &c.run, stream))?;
                reset_banks(&a, &b[SERVED], &input, rows, rows, false)?;
                deep_graph.launch(&s.stream).map_err(|e| e.to_string())?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                if a.read(b[SERVED].first.output)? != served_out {
                    return Err(format!(
                        "{name} differs from the served fold: {} rows {rows}",
                        label(shape)
                    ));
                }
                let mut entry = json!({"symbols":c.run.iter().map(|c| &c.symbol).collect::<Vec<_>>(),"bitwise":true});
                for (time, cold) in [("warm", false), ("cold", true)] {
                    entry[time] = timing::pair(s, cold, |arm, stream| {
                        if arm == 0 {
                            &graphs[SERVED]
                        } else {
                            &deep_graph
                        }
                        .launch(stream)
                        .map_err(|e| e.to_string())
                    })?;
                }
                record["deep"][name] = entry;
            }
        }
    }
    Ok(record)
}

/// The swizzled deep rings of a single shape at 16 or 32 rows, on `b`.
fn deep_calls_of(
    a: &mut Arena<'_>,
    shape: &Shape,
    rows: u32,
    b: &Banks,
    route: &str,
) -> LabResult<Vec<(String, Calls)>> {
    if shape.paired || !TIMED_ROWS.contains(&rows) || route != routes(shape, rows)[0] {
        return Ok(vec![]);
    }
    profile::deep(&shape.base)
        .into_iter()
        .filter(|e| e.swizzle)
        .map(|e| {
            let work = a.filled(e.workspace(), 0)?;
            Ok((
                deep_family::label(&e),
                Calls {
                    init: vec![],
                    run: e
                        .launches(&b.first, work, rows)
                        .map_err(|e| e.to_string())?,
                    state: None,
                },
            ))
        })
        .collect()
}

pub fn gpu(s: &Session) -> LabResult<()> {
    let mode = mode()?;
    let kept = rows_of(mode)?;
    let mut cases = 0;
    let mut changed = 0;
    for shape in shapes() {
        for &rows in &kept {
            for route in routes(&shape, rows) {
                let record = case(s, &shape, rows, route, mode)?;
                if record["same_modules"] == json!(false) {
                    changed += 1;
                }
                println!("FOLD_PLAIN_CASE {record}");
                cases += 1;
            }
        }
    }
    println!(
        "FOLD_PLAIN_DONE {}",
        json!({"mode":std::env::var("CHOPINFER_FOLD_PLAIN_MODE").unwrap_or_default(),"rows":kept,"cases":cases,"changed":changed})
    );
    Ok(())
}

/// Every case's calls, modules and plans without opening CUDA: the plan the
/// GPU run must reproduce.
#[test]
fn plain_catalog() {
    let mut rows = Vec::new();
    for shape in shapes() {
        for r in 1..=MAX_ROWS {
            for route in routes(&shape, r) {
                let mut a = Arena::new(None);
                let input = Inputs {
                    first: data::empty(&shape.base, r),
                    second: shape.paired.then(|| data::empty(&shape.base, r)),
                };
                let plain = banks(&mut a, &shape, r, &input, None).unwrap();
                let b = banks(&mut a, &shape, r, &input, Some(&plain)).unwrap();
                let calls = served(&mut a, &shape, &b, r, route).unwrap();
                let all: Vec<_> = calls.init.iter().chain(&calls.run).collect();
                for c in &all {
                    c.plan.validate_geometry().unwrap();
                }
                let changed = changed(
                    &calls
                        .init
                        .iter()
                        .chain(&calls.run)
                        .cloned()
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let deep: Vec<_> = deep_calls_of(&mut a, &shape, r, &b, route)
                    .unwrap()
                    .into_iter()
                    .map(|(name, c)| json!({"label":name,"symbols":c.run.iter().map(|c| &c.symbol).collect::<Vec<_>>()}))
                    .collect();
                rows.push(json!({"shape":label(&shape),"n":shape.base.query.n,"k":shape.base.query.k,
                    "groups":shape.base.query.groups,"paired":shape.paired,"rows":r,"route":route,
                    "changed":changed,
                    "calls":all.iter().map(|c| json!({"init":calls.init.iter().any(|i| std::ptr::eq(i, *c)),
                        "symbol":c.symbol,"modules":(0..3).map(|arm| arm_module(&c.symbol, arm).unwrap()).collect::<Vec<_>>(),
                        "grid":c.plan.grid,"block":c.plan.block,"shared":c.plan.dynamic_shared_bytes})).collect::<Vec<_>>(),
                    "deep":deep}));
            }
        }
    }
    let names: Vec<_> = plain_family::cubins().into_iter().map(|s| s.name).collect();
    println!(
        "FOLD_PLAIN_CATALOG {}",
        json!({"family":plain_family::FAMILY,"arms":ARMS,"cubins":names,"max_rows":MAX_ROWS,"timed_rows":TIMED_ROWS,"cases":rows})
    );
}

#[test]
#[ignore = "requires CHOPINFER_FOLD_CORE_ROOT, CHOPINFER_FOLD_PLAIN_MODE and assigned idle GPUs"]
fn gpu_plain() {
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
