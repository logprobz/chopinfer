//! The served decode folds at the C16 and C32 buckets, alone, for timing
//! and Nsight Compute: no candidate, no comparison. Each case is one base
//! shape at 16 or 32 rows on its captured operands where the captures hold
//! them (the 32-row verify capture, or its first 16 rows), synthetic
//! otherwise. The Flash TP4 32-row query projection is the row-local fold
//! the bundle selects for that bucket (`DenseOps::local_query`).
//!
//! Each case also runs every deep ring fold (`deep_family`) of its shape on
//! the same buffers, which must reproduce the served output bit for bit.
//!
//! `CHOPINFER_FOLD_CORE_PROFILE=time` replays each case's graph after warm
//! and cold caches, A/A for the served fold and served/deep for each deep
//! fold; `=eager` launches each case's calls once without programmatic
//! serialization and no graph, for a launch-filtered profiler.
//! `CHOPINFER_FOLD_CORE_PROFILE_ROWS=16` or `=32` keeps one bucket.
use super::*;

pub const ROWS: [u32; 2] = [16, 32];

/// Shapes the C16/C32 attention and shared-expert decode sublayers launch
/// singly. The shared gate and up run as one pair and are not here.
pub fn bases() -> Vec<mxfp8_fold::Entry> {
    family::bases()
        .into_iter()
        .filter(|b| b.query.n != 576)
        .collect()
}

pub fn served(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Calls> {
    let q = base.query;
    if rows == 32 && (q.n, q.k, q.groups) == (8192, 1280, 1) {
        let local = local_family::entries()
            .into_iter()
            .find(|e| e.tile_n == 32)
            .ok_or("missing served local fold")?;
        let work = a.filled(local.workspace(), 0)?;
        return Ok(Calls {
            init: vec![],
            run: local.launches(b, work, rows).map_err(|e| e.to_string())?,
            state: None,
        });
    }
    reference(a, base, b, rows)
}

pub fn input(base: &mxfp8_fold::Entry, rows: u32) -> LabResult<data::Input> {
    let captured = data::captured(base, true)?;
    let Some(mut input) = captured
        .into_iter()
        .find(|i| i.rows == 32 && i.id.contains("-0-"))
    else {
        return Ok(data::synthetic(base, rows));
    };
    let q = base.query;
    let keep = |bytes: &mut Vec<u8>, width: usize| bytes.truncate(rows as usize * width);
    keep(&mut input.codes, (q.k * q.groups) as usize);
    keep(&mut input.scales, (q.k * q.groups / 32) as usize);
    if let Some(expected) = &mut input.expected {
        keep(expected, (q.n * q.groups * 4) as usize);
    }
    if rows < 32 {
        input.id = input.id.replace("-prefix", &format!("-first{rows}"));
        // A prefix of the 32-row capture was not served at 16 rows.
        input.expected = None;
    }
    input.rows = rows;
    Ok(input)
}

/// Deep ring folds of the base shape.
pub fn deep(base: &mxfp8_fold::Entry) -> Vec<kernels_sm120::gemm::mxfp8_fold_local_rows::Entry> {
    let q = base.query;
    deep_family::entries()
        .into_iter()
        .filter(|e| {
            let d = e.base.query;
            (d.n, d.k, d.groups) == (q.n, q.k, q.groups)
        })
        .collect()
}

fn deep_calls(
    a: &mut Arena<'_>,
    e: &kernels_sm120::gemm::mxfp8_fold_local_rows::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Calls> {
    let work = a.filled(e.workspace(), 0)?;
    Ok(Calls {
        init: vec![],
        run: e.launches(b, work, rows).map_err(|e| e.to_string())?,
        state: None,
    })
}

fn eager_issue(s: &Session, a: &Arena<'_>, f: &Functions, calls: &[Launch]) -> LabResult<()> {
    let mut plain = calls.to_vec();
    for op in &mut plain {
        op.plan.pdl = false;
    }
    issue(s, a, f, &plain, &s.stream)?;
    s.stream.synchronize().map_err(|e| e.to_string())
}

fn case(s: &Session, base: &mxfp8_fold::Entry, rows: u32, eager: bool) -> LabResult<()> {
    let input = input(base, rows)?;
    let mut a = Arena::new(Some(s));
    let b = common(&mut a, base, rows, &input)?;
    let calls = served(&mut a, base, &b, rows)?;
    let deep = deep(base)
        .into_iter()
        .map(|e| Ok((deep_family::label(&e), deep_calls(&mut a, &e, &b, rows)?)))
        .collect::<LabResult<Vec<_>>>()?;
    let all = calls
        .init
        .iter()
        .chain(&calls.run)
        .chain(deep.iter().flat_map(|(_, c)| &c.run))
        .cloned()
        .collect::<Vec<_>>();
    let f = functions(s, &all)?;
    issue(s, &a, &f, &calls.init, &s.stream)?;
    reset(&a, &b, &input, rows, rows, false)?;
    let mut record = json!({"n":input.n,"k":input.k,"groups":input.groups,"rows":rows,"source":input.id,
        "served":calls.run.iter().map(|c|json!({"symbol":c.symbol,"grid":c.plan.grid,"block":c.plan.block})).collect::<Vec<_>>(),
        "deep":{}});
    if eager {
        eager_issue(s, &a, &f, &calls.run)?;
        for (label, c) in &deep {
            eager_issue(s, &a, &f, &c.run)?;
            record["deep"][label] =
                json!({"symbols":c.run.iter().map(|c|&c.symbol).collect::<Vec<_>>()});
        }
    } else {
        let graph = s.capture(|stream| issue(s, &a, &f, &calls.run, stream))?;
        graph.launch(&s.stream).map_err(|e| e.to_string())?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let served_out = a.read(b.output)?;
        if let Some(expected) = &input.expected {
            let width = (input.n * input.groups * 4) as usize;
            if served_out[..rows as usize * width] != expected[..] {
                return Err(format!(
                    "served fold differs from its capture: {}",
                    input.id
                ));
            }
            record["matches_capture"] = json!(true);
        }
        for (name, cold) in [("warm", false), ("cold", true)] {
            record[name] = timing::pair(s, cold, |_, stream| {
                graph.launch(stream).map_err(|e| e.to_string())
            })?;
        }
        for (label, c) in &deep {
            let deep_graph = s.capture(|stream| issue(s, &a, &f, &c.run, stream))?;
            reset(&a, &b, &input, rows, rows, false)?;
            deep_graph.launch(&s.stream).map_err(|e| e.to_string())?;
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let out = a.read(b.output)?;
            if out != served_out {
                return Err(format!(
                    "{label} differs from the served fold: {} byte {:?}",
                    input.id,
                    out.iter().zip(&served_out).position(|(x, y)| x != y)
                ));
            }
            let mut entry = json!({"symbols":c.run.iter().map(|c|&c.symbol).collect::<Vec<_>>(),"bitwise":true});
            for (name, cold) in [("warm", false), ("cold", true)] {
                entry[name] = timing::pair(s, cold, |arm, stream| {
                    let g = if arm == 0 { &graph } else { &deep_graph };
                    g.launch(stream).map_err(|e| e.to_string())
                })?;
            }
            record["deep"][label] = entry;
        }
    }
    println!("FOLD_PROFILE_CASE {record}");
    Ok(())
}

pub fn gpu(s: &Session) -> LabResult<()> {
    let mode = std::env::var("CHOPINFER_FOLD_CORE_PROFILE").map_err(|e| e.to_string())?;
    let eager = match mode.as_str() {
        "eager" => true,
        "time" => false,
        _ => return Err("CHOPINFER_FOLD_CORE_PROFILE must be time or eager".into()),
    };
    let kept = match std::env::var("CHOPINFER_FOLD_CORE_PROFILE_ROWS") {
        Ok(v) => vec![v.parse::<u32>().map_err(|e| e.to_string())?],
        Err(_) => ROWS.to_vec(),
    };
    if kept.iter().any(|r| !ROWS.contains(r)) {
        return Err("CHOPINFER_FOLD_CORE_PROFILE_ROWS must be 16 or 32".into());
    }
    let mut cases = 0;
    for base in bases() {
        for &rows in &kept {
            case(s, &base, rows, eager)?;
            cases += 1;
        }
    }
    println!(
        "FOLD_PROFILE_DONE {}",
        json!({"mode":mode,"rows":kept,"cases":cases})
    );
    Ok(())
}

/// Every served call of the profile resolves to a bundle or lab cubin and a
/// valid plan, without opening CUDA.
#[test]
fn served_profile_catalog() {
    let mut rows = Vec::new();
    for base in bases() {
        for r in ROWS {
            let mut a = Arena::new(None);
            let input = data::empty(&base, r);
            let b = common(&mut a, &base, r, &input).unwrap();
            let calls = served(&mut a, &base, &b, r).unwrap();
            let deep = deep(&base)
                .iter()
                .map(|e| deep_calls(&mut a, e, &b, r))
                .collect::<LabResult<Vec<_>>>()
                .unwrap();
            let init = calls.init.iter().map(|c| (true, c));
            let run = calls.run.iter().chain(deep.iter().flat_map(|c| &c.run));
            for (init, c) in init.chain(run.map(|c| (false, c))) {
                c.plan.validate_geometry().unwrap();
                assert!(
                    c.symbol.starts_with("chop_fold_local_")
                        || c.symbol.starts_with("chop_fold_deep_")
                        || fold_cubin_for_symbol(&c.symbol).is_some(),
                    "{}",
                    c.symbol
                );
                rows.push(json!({"n":base.query.n,"k":base.query.k,"groups":base.query.groups,"rows":r,"init":init,"symbol":c.symbol,"shared":c.plan.dynamic_shared_bytes,"grid":c.plan.grid,"block":c.plan.block}));
            }
        }
    }
    println!("FOLD_PROFILE_CATALOG {}", json!(rows));
}
