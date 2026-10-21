//! The narrow pair (../../../router_narrow.md): the bound router-in-norm
//! step (the certifying norm writing its widened rows, the 64-expert router
//! reading them on its SIMT path) against the same norm without the widened
//! write and the 64-expert router whose SIMT path widens the norm's BF16
//! rows itself. Every compared output must be byte for byte the bound
//! step's, and the candidate's widened rows must stay poisoned. Worker 0
//! takes the clean pattern, worker 1 the outliers, whose uncertified rows
//! and experts reach the SIMT path; with CHOPINFER_DEEPSEEK41_ROUTER_NARROW_
//! MEMCHECK=1 the cases stop at 2049 live rows and nothing is timed.
use super::*;
use bundle_deepseek41::sm120::router_norm_entries as bound;

pub const WORKER: &str = "CHOPINFER_DEEPSEEK41_ROUTER_NARROW_WORKER";
pub const NARROW_MEMCHECK: &str = "CHOPINFER_DEEPSEEK41_ROUTER_NARROW_MEMCHECK";
pub const NARROW_LIVES: [u32; 4] = [1, 65, 2049, 8192];
/// The outputs both arms write, compared byte for byte.
pub const NARROW_COMPARED: [&str; 6] = ["normed", "codes", "scales", "status", "flags", "logits"];

pub fn narrow_cases(worker: u32, memcheck: bool) -> Vec<(Pattern, u32)> {
    let pattern = if worker == 0 {
        Pattern::Clean
    } else {
        Pattern::Outliers
    };
    NARROW_LIVES
        .into_iter()
        .filter(|&l| !memcheck || l < CAPACITY)
        .map(|l| (pattern, l))
        .collect()
}

/// The bound certifying norm over `o`, writing its widened rows or not.
fn norm(i: &Inputs, o: &Outputs, rows: u32, widened: bool) -> LabResult<Launch> {
    let entry = bound::norm();
    let mut arguments = entry
        .reference
        .arguments(RmsNormQuantizeBuffers {
            input: i.input.handle,
            weight: i.norm_weight.handle,
            normed: o.normed.handle,
            widened: widened.then_some(o.widened.handle),
            codes: o.codes.handle,
            scales: o.scales.handle,
            live_rows: i.live.handle,
            status: o.status.handle,
            status_row_offset: 0,
        })
        .map_err(|e| e.to_string())?;
    arguments.0.push(ptr(o.rows));
    Ok(make(
        entry.reference.symbol.clone(),
        rows,
        entry.plan(rows).map_err(|e| e.to_string())?,
        arguments.0,
    ))
}

/// The bound step (`narrow` false) or the candidate.
pub fn step(i: &Inputs, o: &Outputs, rows: u32, narrow: bool) -> LabResult<Vec<Launch>> {
    let e = family::router();
    let (symbol, activation) = if narrow {
        (family::normed_router_symbol(), Argument::NullDevicePointer)
    } else {
        (family::served_router_symbol(), ptr(o.widened))
    };
    Ok(vec![
        norm(i, o, rows, !narrow)?,
        make(
            symbol,
            rows,
            certified::wide_plan(&e, family::TILE_ROWS, family::SERVED_FEATURES, rows)
                .map_err(|e| e.to_string())?,
            vec![
                activation,
                ptr(o.normed),
                ptr(o.rows),
                ptr(i.weight),
                ptr(o.logits),
                ptr(i.live),
                Argument::U32(rows),
                ptr(i.weight_domain),
            ],
        ),
    ])
}

fn compared(o: &Outputs) -> [View; 6] {
    [o.normed, o.codes, o.scales, o.status, o.rows, o.logits]
}

fn case(
    runner: &mut Runner<'_>,
    session: &Session,
    pattern: Pattern,
    live: u32,
    timed: bool,
) -> LabResult<(serde_json::Value, bool)> {
    let mut a = Arena::device(session);
    let i = inputs(&mut a, pattern, live)?;
    let s = outputs(&mut a, false)?;
    let c = outputs(&mut a, false)?;
    let bound_ops = step(&i, &s, CAPACITY, false)?;
    let narrow_ops = step(&i, &c, CAPACITY, true)?;
    reset(&mut a, &s, Prior::Clear)?;
    reset(&mut a, &c, Prior::Clear)?;
    runner.run(&mut a, &bound_ops)?;
    runner.run(&mut a, &narrow_ops)?;
    let mut differing = serde_json::Map::new();
    let mut passed = true;
    for ((name, x), y) in NARROW_COMPARED.iter().zip(compared(&s)).zip(compared(&c)) {
        let (x, y) = (a.read(x)?, a.read(y)?);
        let n = x.iter().zip(&y).filter(|(p, q)| p != q).count();
        passed &= n == 0;
        differing.insert((*name).into(), n.into());
    }
    let untouched = a.read(c.widened)?.iter().all(|&b| b == 0xa5);
    let flags = u32s(&a.read(s.rows)?);
    let uncertified = flags[..live as usize].iter().filter(|&&f| f == 0).count();
    // The bound step's widened rows hold its BF16 rows widened, so the
    // candidate reads the same values.
    let normed = a.read(s.normed)?;
    let widened = a.read(s.widened)?;
    let live_bytes = live as usize * K as usize;
    let consistent = normed[..live_bytes * 2]
        .chunks_exact(2)
        .zip(widened[..live_bytes * 4].chunks_exact(4))
        .all(|(h, w)| (u32::from(u16::from_le_bytes([h[0], h[1]])) << 16).to_le_bytes() == w);
    passed &= untouched && consistent;
    let timing = if timed && live == CAPACITY {
        runner.ensure(&bound_ops)?;
        runner.ensure(&narrow_ops)?;
        let graphs = [&bound_ops, &narrow_ops]
            .iter()
            .map(|ops| session.capture(|st| runner.issue(&a, ops, st)))
            .collect::<LabResult<Vec<_>>>()?;
        timing::pair(session, |arm, st| {
            graphs[arm].launch(st).map_err(|e| e.to_string())
        })?
    } else {
        serde_json::Value::Null
    };
    Ok((
        serde_json::json!({
            "pattern": format!("{pattern:?}").to_lowercase(),
            "live": live,
            "differing_bytes": differing,
            "candidate_widened_untouched": untouched,
            "bound_widened_consistent": consistent,
            "uncertified_rows": uncertified,
            "timing": timing,
            "passed": passed,
        }),
        passed,
    ))
}

pub fn run_narrow(session: &Session) -> LabResult<()> {
    let worker: u32 = std::env::var(WORKER)
        .map_err(|e| format!("{WORKER}: {e}"))?
        .parse()
        .map_err(|_| "worker must be 0 or 1")?;
    if worker > 1 {
        return Err("worker must be 0 or 1".into());
    }
    let memcheck = std::env::var(NARROW_MEMCHECK).as_deref() == Ok("1");
    let mut runner = Runner::new(Some(session))?;
    let mut executed = 0;
    let mut failed = 0;
    for (pattern, live) in narrow_cases(worker, memcheck) {
        let (record, passed) = case(&mut runner, session, pattern, live, !memcheck)?;
        println!("router_narrow_record {record}");
        executed += 1;
        failed += usize::from(!passed);
    }
    println!(
        "router_narrow_done {}",
        serde_json::json!({"worker": worker, "cases": executed, "failed": failed, "memcheck": memcheck})
    );
    if executed == 0 || failed > 0 {
        return Err(format!("{failed} of {executed} narrow cases failed"));
    }
    Ok(())
}

#[test]
fn router_narrow_cpu_plan() {
    // Both arms run the bound norm and router entries (the candidate's
    // router from the lab cubin) over the same plans; the candidate passes
    // no widened rows to either.
    let lab: std::collections::BTreeSet<String> = family::cubins()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect();
    let mut a = Arena::host();
    let i = inputs(&mut a, Pattern::Clean, 65).unwrap();
    let o = outputs(&mut a, false).unwrap();
    for rows in [CAPACITY, 2048, 65] {
        let b = step(&i, &o, rows, false).unwrap();
        let n = step(&i, &o, rows, true).unwrap();
        assert!(Runner::serving(&b[0].symbol) && Runner::serving(&b[1].symbol));
        assert_eq!(n[0].symbol, b[0].symbol);
        assert!(lab.contains(&n[1].symbol) && !Runner::serving(&n[1].symbol));
        assert_eq!((n[0].plan, n[1].plan), (b[0].plan, b[1].plan));
        // Norm: input, weight, normed, widened, ...; router: activation, ...
        assert_eq!(b[0].arguments.0[3], ptr(o.widened));
        assert_eq!(n[0].arguments.0[3], Argument::NullDevicePointer);
        assert_eq!(n[1].arguments.0[0], Argument::NullDevicePointer);
        assert_eq!(n[0].arguments.0.len(), b[0].arguments.0.len());
        assert_eq!(n[1].arguments.0[1..], b[1].arguments.0[1..]);
    }
    for worker in [0, 1] {
        assert_eq!(narrow_cases(worker, false).len(), 4);
        assert_eq!(narrow_cases(worker, true).len(), 3);
    }
    println!(
        "router_narrow_cpu_plan {}",
        serde_json::json!({"cases": 8, "compared": NARROW_COMPARED.len()})
    );
}

#[test]
#[ignore = "requires the staged lab cubin and a worker"]
fn router_narrow_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, run_narrow).unwrap();
}
