//! Compare existing fold schedules independently of the serving selector.
use super::*;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use std::{collections::BTreeSet, io::Read, os::unix::fs::OpenOptionsExt};

const CONFIG: &str = "CHOPINFER_FOLD_SCHEDULE_PAIRS";

#[derive(Clone, Debug, PartialEq, Eq)]
struct Schedule {
    name: String,
    knobs: BTreeMap<String, u32>,
}

struct Case {
    op: String,
    base: mxfp8_fold::Entry,
    paired: bool,
    rows: Vec<u32>,
    schedules: [Schedule; 2],
}

fn count(value: &Value) -> LabResult<u32> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| "expected a u32 count".into())
}

fn schedule(value: &Value) -> LabResult<Schedule> {
    let name = value["schedule"].as_str().ok_or("missing schedule")?;
    let knobs = value["knobs"]
        .as_object()
        .ok_or("missing schedule knobs")?
        .iter()
        .map(|(key, value)| Ok((key.clone(), count(value)?)))
        .collect::<LabResult<BTreeMap<_, _>>>()?;
    match name {
        "split" | "ticket" | "prefetch" if knobs.is_empty() => {}
        "reuse"
            if knobs.len() == 2
                && knobs.get("folds").is_some_and(|n| [2, 4, 8].contains(n))
                && knobs.get("tile_n").is_some_and(|n| [32, 64].contains(n)) => {}
        _ => return Err("unknown schedule or knobs".into()),
    }
    Ok(Schedule {
        name: name.into(),
        knobs,
    })
}

fn parse(value: &Value) -> LabResult<Vec<Case>> {
    if value["schema"] != "deepseek-fold-schedule-pairs-v1" {
        return Err("wrong schedule pair schema".into());
    }
    let mut cases = Vec::new();
    let mut identities = BTreeSet::new();
    for item in value["cases"].as_array().ok_or("missing cases")? {
        let op = item["op"].as_str().ok_or("missing operation")?;
        if !["dense_fold.fold.decode", "reuse.fold.decode"].contains(&op) {
            return Err("unsupported schedule pair operation".into());
        }
        let shape = &item["shape"];
        let n = count(&shape["n"])?;
        let k = count(&shape["k"])?;
        let groups = count(&shape["groups"])?;
        let paired = match count(&shape["paired"])? {
            0 => false,
            1 => true,
            _ => return Err("paired must be zero or one".into()),
        };
        let base = sm120::mxfp8_fold::decode_catalog(&FLASH)?
            .into_iter()
            .find(|e| (e.query.n, e.query.k, e.query.groups, e.paired) == (n, k, groups, paired))
            .ok_or("shape has no existing fold entry")?
            .legacy()
            .map_err(|e| e.to_string())?;
        let rows = item["rows"]
            .as_array()
            .ok_or("missing rows")?
            .iter()
            .map(count)
            .collect::<LabResult<Vec<_>>>()?;
        if rows.is_empty()
            || rows.iter().any(|r| !(1..=64).contains(r))
            || rows.windows(2).any(|r| r[0] >= r[1])
        {
            return Err("rows must be nonempty, ascending and within 1..64".into());
        }
        let schedules = [schedule(&item["a"])?, schedule(&item["b"])?];
        for &rows in &rows {
            if !identities.insert((op.to_owned(), n, k, groups, paired, rows)) {
                return Err("duplicate schedule pair cell".into());
            }
        }
        cases.push(Case {
            op: op.into(),
            base,
            paired,
            rows,
            schedules,
        });
    }
    if cases.is_empty() || count(&value["expected_cells"])? as usize != identities.len() {
        return Err("schedule pair cell count differs".into());
    }
    Ok(cases)
}

fn configured() -> LabResult<Vec<Case>> {
    let path = std::env::var_os(CONFIG).ok_or("missing schedule pair configuration")?;
    let mut bytes = Vec::new();
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0x20000)
        .open(path)
        .map_err(|e| e.to_string())?
        .take((1 << 20) + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 1 << 20 {
        return Err("schedule pair configuration too large".into());
    }
    parse(&serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)
}

fn matches(q: Mxfp8FoldGemmQuery, other: Mxfp8FoldGemmQuery) -> bool {
    (q.n, q.k, q.groups) == (other.n, other.k, other.groups)
}

fn backend(case: &Case, rows: u32, schedule: &Schedule) -> LabResult<Sm120> {
    let q = case.base.query;
    let backend = Sm120::new().with_packed_fold_weights();
    Ok(match schedule.name.as_str() {
        "split" => {
            let mut entry = sm120::mxfp8_fold::decode_catalog(&FLASH)?
                .into_iter()
                .find(|e| matches(q, e.query) && e.paired == case.paired)
                .ok_or("missing existing split entry")?;
            entry.query = sm120::mxfp8_fold::routed_query(entry.query, rows);
            backend.with_mxfp8_fold_decode(entry)
        }
        "ticket" | "prefetch" => {
            let entries = if schedule.name == "prefetch" {
                sm120::mxfp8_fold::prefetch_catalog(&FLASH)?
            } else {
                sm120::mxfp8_fold::ticket_catalog(&FLASH)?
            };
            let mut entry = entries
                .into_iter()
                .find(|e| matches(q, e.query) && e.paired == case.paired)
                .ok_or("missing existing ticket entry")?;
            entry.query = sm120::mxfp8_fold::routed_query(entry.query, rows);
            backend.with_mxfp8_fold_ticket(entry)
        }
        "reuse" => {
            let mut entry = sm120::projection_reuse::entries()?
                .into_iter()
                .find(|e| {
                    matches(q, e.ticket.query)
                        && e.ticket.paired == case.paired
                        && e.tile_n == schedule.knobs["tile_n"]
                })
                .ok_or("missing existing reuse entry")?;
            entry.ticket.query = sm120::mxfp8_fold::routed_query(entry.ticket.query, rows);
            backend.with_mxfp8_fold_reuse(entry, schedule.knobs["folds"])
        }
        _ => return Err("unknown fold schedule".into()),
    })
}

struct Prepared {
    calls: Vec<Launch>,
    workspace: View,
}

fn prepare(
    arena: &mut Arena<'_>,
    case: &Case,
    rows: u32,
    schedule: &Schedule,
    buffers: &[Buffers<View>],
) -> LabResult<Prepared> {
    let backend = backend(case, rows, schedule)?;
    let query = sm120::mxfp8_fold::routed_query(case.base.query, rows);
    let (calls, workspace) = if case.paired {
        let op = backend
            .mxfp8_fold_gemm_pair(&query, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let workspace = arena.filled(op.info().workspace.max(4), 0xa5)?;
        let calls = op
            .launches(&[buffers[0], buffers[1]], workspace, Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        (calls, workspace)
    } else {
        let op = backend
            .mxfp8_fold_gemm(&query, Need::rows(rows))
            .map_err(|e| e.to_string())?;
        let workspace = arena.filled(op.info().workspace.max(4), 0xa5)?;
        let calls = op
            .launches(&buffers[0], workspace, Counts::rows(rows))
            .map_err(|e| e.to_string())?;
        (calls, workspace)
    };
    if calls.is_empty()
        || calls
            .iter()
            .any(|c| fold_cubin_for_symbol(&c.symbol).is_none())
    {
        return Err("schedule requires a kernel outside the existing bundle".into());
    }
    Ok(Prepared { calls, workspace })
}

fn check_case(session: &Session, case: &Case, rows: u32) -> LabResult<()> {
    let mut arena = Arena::new(Some(session));
    let mut inputs = vec![data::synthetic_rank(
        &case.base,
        rows,
        session.ordinal as u32 % 4,
    )];
    if case.paired {
        let mut second = data::synthetic_rank(&case.base, rows, session.ordinal as u32 % 4);
        (second.weights, second.weight_scales) = data::second_bank(&case.base);
        inputs.push(second);
    }
    let mut buffers = Vec::new();
    for _ in 0..2 {
        let mut banks = Vec::new();
        for input in &inputs {
            let plain = common(&mut arena, &case.base, rows, input)?;
            banks.push(packed_views(&mut arena, &case.base, &plain)?);
        }
        if case.paired {
            banks[1].activations = banks[0].activations;
            banks[1].activation_scales = banks[0].activation_scales;
            banks[1].live_rows = banks[0].live_rows;
            banks[1].status = banks[0].status;
        }
        buffers.push(banks);
    }
    let prepared = (0..2)
        .map(|arm| prepare(&mut arena, case, rows, &case.schedules[arm], &buffers[arm]))
        .collect::<LabResult<Vec<_>>>()?;
    let calls = prepared
        .iter()
        .flat_map(|p| p.calls.clone())
        .collect::<Vec<_>>();
    let functions = functions(session, &calls)?;
    let graphs = prepared
        .iter()
        .map(|p| session.capture(|stream| issue(session, &arena, &functions, &p.calls, stream)))
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
                arena.fill(prepared[arm].workspace, 0xa5)?;
                for (bank, input) in inputs.iter().enumerate() {
                    reset(&arena, &buffers[arm][bank], input, rows, live, bad)?;
                }
                if graph {
                    graphs[arm]
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                } else {
                    let mut eager = prepared[arm].calls.clone();
                    for call in &mut eager {
                        call.plan.pdl = false;
                    }
                    issue(session, &arena, &functions, &eager, &session.stream)?;
                }
            }
            for (bank, input) in inputs.iter().enumerate() {
                compare(
                    &arena,
                    &[buffers[0][bank], buffers[1][bank]],
                    input,
                    rows,
                    live,
                    bad,
                    None,
                )?;
            }
            comparisons += 1;
        }
    }
    // A null candidate must fail the same byte check used above.
    for (bank, input) in inputs.iter().enumerate() {
        reset(&arena, &buffers[1][bank], input, rows, rows, false)?;
        if compare(
            &arena,
            &[buffers[0][bank], buffers[1][bank]],
            input,
            rows,
            rows,
            false,
            None,
        )
        .is_ok()
        {
            return Err("null candidate was accepted".into());
        }
    }
    let expected = buffers[0]
        .iter()
        .map(|b| Ok((arena.read(b.output)?, arena.read(b.status)?)))
        .collect::<LabResult<Vec<_>>>()?;
    for banks in &buffers {
        for (bank, input) in inputs.iter().enumerate() {
            reset(&arena, &banks[bank], input, rows, rows, false)?;
        }
    }
    // Batch device work into one graph launch so host issue gaps do not
    // dominate these short kernels. The duplicate A arm uses the same graph.
    const REPLAYS: usize = 32;
    let timed_graphs = prepared
        .iter()
        .map(|p| {
            session.capture(|stream| {
                for _ in 0..REPLAYS {
                    issue(session, &arena, &functions, &p.calls, stream)?;
                }
                Ok(())
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut timing = timing::arms(session, false, 3, |arm, stream| {
        timed_graphs[usize::from(arm == 2)]
            .launch(stream)
            .map_err(|e| e.to_string())
    })?;
    timing["replays_per_sample"] = json!(REPLAYS);
    timing["graph_launches_per_sample"] = json!(1);
    for (bank, input) in inputs.iter().enumerate() {
        compare(
            &arena,
            &[buffers[0][bank], buffers[1][bank]],
            input,
            rows,
            rows,
            false,
            None,
        )?;
        if arena.read(buffers[0][bank].output)? != expected[bank].0
            || arena.read(buffers[0][bank].status)? != expected[bank].1
        {
            return Err("batched timing changed the checked output/status bytes".into());
        }
    }
    println!(
        "SCHEDULE_PAIR_CASE {}",
        json!({
            "op": case.op, "n": case.base.query.n, "k": case.base.query.k,
            "groups": case.base.query.groups, "paired": case.paired, "rows": rows,
            "device": session.ordinal, "a": {"schedule":case.schedules[0].name,"knobs":case.schedules[0].knobs},
            "b": {"schedule":case.schedules[1].name,"knobs":case.schedules[1].knobs},
            "symbols": prepared.iter().map(|p|p.calls.iter().map(|c|&c.symbol).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "comparisons": comparisons, "bitwise": true, "null_rejected": true,
            "timing": timing, "arms": ["a", "a", "b"], "cache": "warm"
        })
    );
    Ok(())
}

#[test]
#[ignore = "requires CHOPINFER_FOLD_SCHEDULE_PAIRS"]
fn dry_run() {
    let cases = configured().unwrap();
    let mut count = 0;
    for case in cases {
        for &rows in &case.rows {
            let mut arena = Arena::new(None);
            let input = data::empty(&case.base, rows);
            let first = common(&mut arena, &case.base, rows, &input).unwrap();
            let mut banks = vec![first];
            if case.paired {
                let mut second = common(&mut arena, &case.base, rows, &input).unwrap();
                second.activations = first.activations;
                second.activation_scales = first.activation_scales;
                second.live_rows = first.live_rows;
                second.status = first.status;
                banks.push(second);
            }
            for schedule in &case.schedules {
                prepare(&mut arena, &case, rows, schedule, &banks).unwrap();
            }
            count += 1;
        }
    }
    println!("SCHEDULE_PAIR_DRY_RUN {{\"cells\":{count}}}");
}

#[test]
#[ignore = "requires CHOPINFER_FOLD_SCHEDULE_PAIRS and an assigned idle GPU"]
fn gpu_pair() {
    let cases = configured().unwrap();
    let expected = cases.iter().map(|c| c.rows.len()).sum::<usize>();
    assert_eq!(
        kernel_harness::selected_ordinals().unwrap().unwrap().len(),
        1
    );
    assert!(
        kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 2 << 30, |session| {
            let mut count = 0;
            let mut rejected = 0;
            for case in &cases {
                for &rows in &case.rows {
                    if let Err(error) = check_case(session, case, rows) {
                        if !error.starts_with("output/status mismatch") {
                            return Err(error);
                        }
                        rejected += 1;
                        println!(
                            "SCHEDULE_PAIR_CASE {}",
                            json!({
                                "op":case.op, "n":case.base.query.n, "k":case.base.query.k,
                                "groups":case.base.query.groups, "paired":case.paired, "rows":rows,
                                "device":session.ordinal, "bitwise":false, "error":error
                            })
                        );
                    }
                    count += 1;
                }
            }
            assert_eq!(count, expected);
            println!("SCHEDULE_PAIR_DONE {{\"cells\":{count},\"rejected\":{rejected}}}");
            if rejected != 0 {
                return Err(format!("{rejected} schedule cells are not byte-equal"));
            }
            Ok(())
        })
        .unwrap()
        .is_some()
    );
}

#[test]
fn reject_unknown_schedule_knobs_and_empty_matrix() {
    assert!(schedule(&json!({"schedule":"reuse","knobs":{"folds":3,"tile_n":32}})).is_err());
    assert!(schedule(&json!({"schedule":"split","knobs":{"folds":4}})).is_err());
    assert!(
        parse(&json!({"schema":"deepseek-fold-schedule-pairs-v1","expected_cells":0,"cases":[]}))
            .is_err()
    );
}

#[test]
fn explicit_prefetch_does_not_follow_the_split_selector() {
    let spec = json!({
        "schema":"deepseek-fold-schedule-pairs-v1", "expected_cells":1,
        "cases":[{"op":"dense_fold.fold.decode", "shape":{"n":5120,"k":576,"groups":1,"paired":0},
            "rows":[17], "a":{"schedule":"split","knobs":{}},
            "b":{"schedule":"prefetch","knobs":{}}}]
    });
    let cases = parse(&spec).unwrap();
    let case = &cases[0];
    let mut arena = Arena::new(None);
    let input = data::empty(&case.base, 17);
    let buffer = common(&mut arena, &case.base, 17, &input).unwrap();
    let arms = case
        .schedules
        .iter()
        .map(|s| prepare(&mut arena, case, 17, s, &[buffer]).unwrap().calls)
        .collect::<Vec<_>>();
    assert_ne!(
        arms[0].last().unwrap().symbol,
        arms[1].last().unwrap().symbol
    );
    assert!(arms[1].last().unwrap().symbol.contains("prefetch"));
    let mut wrong_count = spec.clone();
    wrong_count["expected_cells"] = json!(2);
    assert!(parse(&wrong_count).is_err());
    let mut duplicate = spec;
    duplicate["cases"][0]["rows"] = json!([17, 17]);
    assert!(parse(&duplicate).is_err());
}
