//! Compare isolated fold cores with the complete served selector.
use bundle_deepseek41::sm120::{self, fold_cubin_for_symbol};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    mxfp8_gemm::{Mxfp8FoldGemmBuffers as Buffers, Mxfp8FoldStreamBuffers},
};
use kernel_harness::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use kernels_sm120::{
    backend::Sm120,
    gemm::{mxfp8_fold, mxfp8_fold_tiled as decode, mxfp8_prefill_matrix as prefill},
};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, sync::OnceLock};
mod arena;
mod data;
#[path = "../../../sm120/fold_core.rs"]
mod family;
mod timing;
use arena::Arena;

fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_FOLD_CORE_ROOT")
        .expect("staged fold root")
        .into()
}
fn is_decode() -> bool {
    std::env::var("CHOPINFER_FOLD_CORE_PHASE").unwrap_or_else(|_| "decode".into()) == "decode"
}
fn backend() -> &'static Sm120 {
    static BACKEND: OnceLock<Sm120> = OnceLock::new();
    BACKEND.get_or_init(|| {
        let b = sm120::mxfp8_fold::register(Sm120::new(), &FLASH, 4).unwrap();
        let b = sm120::mxfp8_fold_stream::register(b, &FLASH, 4).unwrap();
        let b = sm120::mxfp8_fold_direct::register(b, &FLASH, 4).unwrap();
        sm120::projection_reuse::register(b, &FLASH, 4).unwrap()
    })
}
#[derive(Clone)]
enum Candidate {
    Decode(decode::Entry),
    Prefill(prefill::Entry),
}
impl Candidate {
    fn base(&self) -> mxfp8_fold::Entry {
        match self {
            Self::Decode(e) => e.packed.ticket.legacy().unwrap(),
            Self::Prefill(e) => e.base.clone(),
        }
    }
    fn label(&self) -> String {
        match self {
            Self::Decode(e) => e.schedule.name().into(),
            Self::Prefill(e) => format!("m{}s{}", e.tile_rows, e.stages),
        }
    }
}
fn candidates() -> Vec<Candidate> {
    let all: Vec<_> = if is_decode() {
        family::decode_entries()
            .into_iter()
            .map(Candidate::Decode)
            .collect()
    } else {
        family::prefill_entries()
            .into_iter()
            .map(Candidate::Prefill)
            .collect()
    };
    let selected = std::env::var("CHOPINFER_FOLD_CORE_VARIANT").ok();
    let result: Vec<_> = all
        .into_iter()
        .filter(|c| selected.as_ref().is_none_or(|name| *name == c.label()))
        .collect();
    assert!(!result.is_empty(), "unknown core variant");
    result
}
fn row_counts(mem: bool) -> Vec<u32> {
    if is_decode() {
        if mem {
            vec![8, 32]
        } else {
            (1..=8).chain([16, 32]).collect()
        }
    } else if mem {
        vec![65, 129]
    } else {
        vec![64, 192, 448, 960, 1472, 2496, 4096, 8192]
    }
}
fn common(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    rows: u32,
    input: &data::Input,
) -> LabResult<Buffers<View>> {
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
        weights: if a.session.is_some() {
            a.bytes(&input.weights)?
        } else {
            a.filled(size.weights, 0)?
        },
        weight_scales: if a.session.is_some() {
            a.bytes(&input.weight_scales)?
        } else {
            a.filled(size.weight_scales, 0)?
        },
        output: a.filled(size.output + 64, 255)?,
        live_rows: a.bytes(&rows.to_le_bytes())?,
        status: a.filled(size.status + 20, 0)?,
        status_row_offset: 3,
    })
}
struct Calls {
    init: Vec<Launch>,
    run: Vec<Launch>,
    state: Option<View>,
}
fn reference(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Calls> {
    let backend = backend();
    if is_decode() {
        let q = sm120::mxfp8_fold::routed_query(base.query, rows);
        if let Ok(op) = backend.mxfp8_fold_gemm_stream(&q, false, Need::rows(rows)) {
            let state = a.filled(
                op.info()
                    .bytes_of("state")
                    .ok_or("missing baseline state")?,
                0,
            )?;
            let packed = a.filled(b.weights.bytes, 0)?;
            let views = Mxfp8FoldStreamBuffers {
                first: *b,
                second: None,
                packed_weights: [packed, View::EMPTY],
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
            .mxfp8_fold_gemm(&q, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let work = a.filled(op.info().workspace.max(4), 0)?;
        return Ok(Calls {
            init: vec![],
            run: op
                .launches(b, work, Counts::rows(rows))
                .map_err(|e| e.to_string())?,
            state: None,
        });
    }
    let mut q = base.query;
    q.m_capacity = 8192;
    let op = backend
        .mxfp8_prefill_gemm(&q, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let work = a.filled(op.info().workspace.max(4), 0)?;
    Ok(Calls {
        init: vec![],
        run: op
            .launches(b, work, Counts::rows(rows))
            .map_err(|e| e.to_string())?,
        state: None,
    })
}
fn candidate(a: &mut Arena<'_>, c: &Candidate, b: &Buffers<View>, rows: u32) -> LabResult<Calls> {
    match c {
        Candidate::Decode(e) => {
            let state = a.filled(e.state_bytes(32).map_err(|e| e.to_string())?, 0)?;
            let packed = a.filled(e.packed.packed_bytes(), 0)?;
            let views = Mxfp8FoldStreamBuffers {
                first: *b,
                second: None,
                packed_weights: [packed, View::EMPTY],
                state,
            };
            Ok(Calls {
                init: vec![e.initialize(&views).map_err(|e| e.to_string())?],
                run: vec![e.launch(&views, rows, None).map_err(|e| e.to_string())?],
                state: Some(state),
            })
        }
        Candidate::Prefill(e) => Ok(Calls {
            init: vec![],
            run: e
                .prepare(rows)
                .map_err(|e| e.to_string())?
                .launches(b, View::EMPTY, Counts::rows(rows))
                .map_err(|e| e.to_string())?,
            state: None,
        }),
    }
}
struct Functions {
    map: BTreeMap<String, Function>,
    _modules: Vec<Module>,
}
fn functions(s: &Session, calls: &[Launch]) -> LabResult<Functions> {
    struct Cache(std::cell::RefCell<BTreeMap<String, Module>>);
    let cache = s.state(|| Ok(Cache(std::cell::RefCell::new(BTreeMap::new()))))?;
    let mut modules = cache.0.borrow_mut();
    let mut map = BTreeMap::new();
    for call in calls {
        if map.contains_key(&call.symbol) {
            continue;
        }
        let external = if call.symbol.starts_with("chop_fold_core_") {
            Some(family::DECODE)
        } else if call.symbol.contains("_matrix_") {
            Some(family::PREFILL)
        } else {
            None
        };
        let (name, bytes) = if let Some(name) = external {
            (
                name,
                std::fs::read(root().join("cubins").join(format!("{name}.cubin")))
                    .map_err(|e| e.to_string())?,
            )
        } else {
            let (name, bytes) = fold_cubin_for_symbol(&call.symbol)
                .ok_or_else(|| format!("missing served {}", call.symbol))?;
            (name, bytes.to_vec())
        };
        if !modules.contains_key(name) {
            modules.insert(name.to_owned(), s.load_cubin(&bytes)?);
        }
        let f = modules[name]
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
fn issue(
    s: &Session,
    a: &Arena<'_>,
    f: &Functions,
    calls: &[Launch],
    stream: &Stream,
) -> LabResult<()> {
    let bindings = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    for call in calls {
        launch(
            &s.context,
            &f.map[&call.symbol],
            stream,
            call.plan,
            &call.arguments,
            &bindings,
        )?;
    }
    Ok(())
}
fn status(rows: u32, live: u32, bad: bool) -> Vec<u8> {
    (0..rows + 5)
        .flat_map(|i| {
            let v = if i < 3 || i >= rows + 3 {
                0xaabbccdd_u32
            } else if bad && i == live + 2 && live > 1 {
                0x80
            } else {
                0
            };
            v.to_le_bytes()
        })
        .collect()
}
fn reset(
    a: &Arena<'_>,
    b: &Buffers<View>,
    input: &data::Input,
    rows: u32,
    live: u32,
    bad: bool,
) -> LabResult<()> {
    a.fill(b.output, 255)?;
    a.write(b.live_rows, &live.to_le_bytes())?;
    a.write(b.status, &status(rows, live, bad))?;
    let mut codes = input.codes.clone();
    if bad && live > 0 {
        codes[0] = 127;
    }
    a.write(b.activations, &codes)
}
fn compare(
    a: &Arena<'_>,
    b: &[Buffers<View>; 2],
    input: &data::Input,
    rows: u32,
    live: u32,
    bad: bool,
    state: Option<View>,
) -> LabResult<()> {
    let old = a.read(b[0].output)?;
    let new = a.read(b[1].output)?;
    if old != new || a.read(b[0].status)? != a.read(b[1].status)? {
        return Err(format!(
            "output/status mismatch rows={rows} live={live} bad={bad} byte={:?}",
            old.iter().zip(&new).position(|(x, y)| x != y)
        ));
    }
    let width = input.n as usize * input.groups as usize * 4;
    if new[live as usize * width..].iter().any(|&v| v != 255) {
        return Err("inactive output or guard overwritten".into());
    }
    for r in 0..live as usize {
        let row = &new[r * width..(r + 1) * width];
        if bad && (r == 0 || r + 1 == live as usize) {
            if row.iter().any(|&v| v != 0) {
                return Err("rejected row not cleared".into());
            }
        } else if let Some(expected) = &input.expected {
            if row != &expected[r * width..(r + 1) * width] {
                return Err("served baseline differs from captured expected output".into());
            }
        } else if row
            .chunks_exact(4)
            .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
        {
            return Err("nonfinite live output".into());
        }
    }
    if let Some(v) = state
        && a.read(v)?.iter().any(|&v| v != 0)
    {
        return Err("candidate state did not reset".into());
    }
    Ok(())
}
fn check_case(
    s: &Session,
    c: &Candidate,
    rows: u32,
    input: &data::Input,
    timed: bool,
    mem: bool,
) -> LabResult<()> {
    let base = c.base();
    let mut a = Arena::new(Some(s));
    let b = [
        common(&mut a, &base, rows, input)?,
        common(&mut a, &base, rows, input)?,
    ];
    let calls = [
        reference(&mut a, &base, &b[0], rows)?,
        candidate(&mut a, c, &b[1], rows)?,
    ];
    let mut omit = calls[1].run.clone();
    let last = omit.last_mut().unwrap();
    match c {
        Candidate::Decode(_) => last.symbol.push_str("_omit_v1"),
        Candidate::Prefill(_) => last.symbol.push_str("_omit_first_v1"),
    }
    let mut delayed = calls[1].run.clone();
    if is_decode() {
        delayed.last_mut().unwrap().symbol.push_str("_delayed_v1");
    }
    let all = calls
        .iter()
        .flat_map(|v| v.init.iter().chain(v.run.iter()))
        .chain(omit.iter())
        .chain(delayed.iter())
        .cloned()
        .collect::<Vec<_>>();
    let f = functions(s, &all)?;
    for op in &calls {
        issue(s, &a, &f, &op.init, &s.stream)?;
    }
    let graphs = calls
        .iter()
        .map(|op| s.capture(|stream| issue(s, &a, &f, &op.run, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    let mut comparisons = 0;
    for (live, bad) in [
        (rows, false),
        (rows.saturating_sub(1), true),
        (0, false),
        (rows, false),
    ] {
        for graph in [false, true] {
            for arm in 0..2 {
                reset(&a, &b[arm], input, rows, live, bad)?;
                if graph {
                    graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                } else {
                    let mut plain = calls[arm].run.clone();
                    for op in &mut plain {
                        op.plan.pdl = false;
                    }
                    issue(s, &a, &f, &plain, &s.stream)?;
                }
            }
            compare(&a, &b, input, rows, live, bad, calls[1].state)?;
            comparisons += 1;
        }
    }
    // Both controls run outside timing. The delayed admission must still clear
    // whole rows, and omitting the first fold must change the result.
    reset(&a, &b[1], input, rows, rows, false)?;
    issue(s, &a, &f, &omit, &s.stream)?;
    if a.read(b[0].output)? == a.read(b[1].output)? {
        return Err("omit-first control was ineffective".into());
    }
    if is_decode() {
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
        compare(&a, &b, input, rows, rows, true, calls[1].state)?;
    }
    let mut record = json!({"phase":if is_decode(){"decode"}else{"prefill"},"n":input.n,"k":input.k,"groups":input.groups,"rows":rows,"variant":c.label(),"source":input.id,"comparisons":comparisons,"bitwise":true,"negative":true,"delayed_admission":is_decode(),"memcheck":mem,"reference":calls[0].run.iter().map(|v|&v.symbol).collect::<Vec<_>>()});
    if timed {
        for buffer in &b {
            reset(&a, buffer, input, rows, rows, false)?;
        }
        for (name, cold) in [("warm", false), ("cold", true)] {
            record[name] = timing::pair(s, cold, |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
        }
    }
    println!("FOLD_CORE_CASE {record}");
    Ok(())
}
fn gpu(s: &Session) -> LabResult<()> {
    let mem = std::env::var("CHOPINFER_FOLD_CORE_MEMCHECK").as_deref() == Ok("1");
    let mut count = 0;
    for c in candidates() {
        let base = c.base();
        for rows in row_counts(mem) {
            let input = data::synthetic(&base, rows);
            check_case(s, &c, rows, &input, !mem, mem)?;
            count += 1;
        }
        if !mem {
            for input in data::captured(&base, is_decode())? {
                let rows = input.rows;
                check_case(s, &c, rows, &input, true, false)?;
                count += 1;
            }
        }
    }
    println!(
        "FOLD_CORE_DONE {}",
        json!({"phase":if is_decode(){"decode"}else{"prefill"},"memcheck":mem,"cases":count})
    );
    Ok(())
}
#[test]
#[ignore = "requires CHOPINFER_FOLD_CORE_ROOT and assigned idle GPUs"]
fn gpu_pair() {
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
#[test]
fn cpu_catalog() {
    assert_eq!(family::bases().len(), 9);
    let mut symbols = std::collections::BTreeSet::new();
    let mut references: BTreeMap<String, Value> = BTreeMap::new();
    let mut plans = Vec::new();
    for spec in family::cubins() {
        for e in spec.entries {
            assert!(symbols.insert(e.symbol));
        }
    }
    for c in candidates() {
        for rows in row_counts(false).into_iter().chain(row_counts(true)) {
            let base = c.base();
            let mut a = Arena::new(None);
            let input = data::empty(&base, rows);
            let b = common(&mut a, &base, rows, &input).unwrap();
            let old = reference(&mut a, &base, &b, rows).unwrap();
            let new = candidate(&mut a, &c, &b, rows).unwrap();
            for launch in old
                .init
                .iter()
                .chain(&old.run)
                .chain(&new.init)
                .chain(&new.run)
            {
                launch.plan.validate_geometry().unwrap();
                if let Some((name, bytes)) = fold_cubin_for_symbol(&launch.symbol) {
                    let r = references.entry(name.into()).or_insert_with(
                        || json!({"sha256":kernel_build::digest(bytes),"symbols":[]}),
                    );
                    let list = r["symbols"].as_array_mut().unwrap();
                    if !list.contains(&json!(launch.symbol)) {
                        list.push(json!(launch.symbol));
                    }
                } else {
                    assert!(symbols.contains(&launch.symbol), "{}", launch.symbol);
                }
                plans.push(json!({"symbol":launch.symbol,"grid":launch.plan.grid,"block":launch.plan.block,"shared":launch.plan.dynamic_shared_bytes}));
            }
        }
    }
    println!(
        "FOLD_CORE_CATALOG {}",
        json!({"family":family::FAMILY,"phase":if is_decode(){"decode"}else{"prefill"},"rows":row_counts(false),"memcheck_rows":row_counts(true),"variants":candidates().iter().map(|c|{let q=c.base().query;json!({"n":q.n,"k":q.k,"groups":q.groups,"variant":c.label()})}).collect::<Vec<_>>(),"candidate_symbols":symbols,"references":references,"plans":plans})
    );
}
