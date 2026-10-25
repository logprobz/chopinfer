use super::*;
type Readback = Vec<Vec<u8>>;

fn guards(c: &Case, rows: u32, live: u32, sticky: bool, value: &[Vec<u8>]) -> LabResult<()> {
    let active = live.min(rows) as usize;
    let n = c.width();
    for (role, stride) in [(0, n * c.bytes()), (1, n * 4), (2, n), (3, n / 32)] {
        let from = if role == 1 && !c.widen {
            0
        } else {
            active * stride
        };
        if !value[role][from..].iter().all(|b| *b == 0xa5) {
            return Err(format!("{} inactive/unused role {role} changed", c.label()));
        }
    }
    for word in (0..OFFSET).chain(OFFSET + rows as usize..OFFSET + rows as usize + 2) {
        if value[4][word * 4..word * 4 + 4] != 0x1357_2468u32.to_ne_bytes() {
            return Err("status guard changed".into());
        }
    }
    for word in OFFSET + active..OFFSET + rows as usize {
        if value[4][word * 4..word * 4 + 4] != (if sticky { 0x80u32 } else { 0 }).to_ne_bytes() {
            return Err("inactive status changed".into());
        }
    }
    Ok(())
}

pub(super) fn run(s: &Session) -> LabResult<Value> {
    let loaded = Loaded::load(s)?;
    let memcheck = std::env::var_os("CHOPINFER_SMALL_NORM_MEMCHECK").is_some();
    let patterns: &[&str] = if memcheck {
        &["finite", "nan_last", "sticky"]
    } else {
        &PATTERNS
    };
    let mut comparisons = 0usize;
    let mut negatives = 0usize;
    let mut records = vec![];
    for case in cases() {
        for rank in 0..4 {
            for rows in ROWS {
                let width = case.width();
                let input = Input {
                    values: s.allocate(width * rows as usize * case.bytes())?,
                    weight: s.allocate(width * 2)?,
                    live: s.allocate(4)?,
                };
                let outputs = (0..4)
                    .map(|_| Output::new(s, &case, rows))
                    .collect::<LabResult<Vec<_>>>()?;
                let consumers = (0..4)
                    .map(|_| Output::new(s, &case, rows))
                    .collect::<LabResult<Vec<_>>>()?;
                let graphs = [false, true]
                    .into_iter()
                    .map(|pdl| {
                        (0..4)
                            .map(|arm| {
                                s.capture(|stream| {
                                    call(
                                        &case,
                                        &loaded,
                                        arm,
                                        rows,
                                        &input,
                                        &input.values,
                                        &outputs[arm],
                                        stream,
                                        pdl,
                                    )?;
                                    // The second real norm reads the first norm's output.
                                    // Early release must not expose incomplete values.
                                    call(
                                        &case,
                                        &loaded,
                                        0,
                                        rows,
                                        &input,
                                        &outputs[arm].buffers[0],
                                        &consumers[arm],
                                        stream,
                                        pdl,
                                    )
                                })
                            })
                            .collect::<LabResult<Vec<_>>>()
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                for pattern in patterns {
                    let f32 = data::rows_of(width, rows, pattern, 0x91a2_03c4 + rank);
                    let values = if case.f32() {
                        f32
                    } else {
                        f32.chunks_exact(4)
                            .flat_map(|v| {
                                ((u32::from_ne_bytes(v.try_into().unwrap()) >> 16) as u16)
                                    .to_ne_bytes()
                            })
                            .collect()
                    };
                    upload(&input.values, &values, &s.stream)?;
                    upload(
                        &input.weight,
                        &data::weight_of(width, pattern, 0x5eed_0f0e + rank),
                        &s.stream,
                    )?;
                    for live in (0..=rows).chain([rows + 7]) {
                        upload(&input.live, &live.to_ne_bytes(), &s.stream)?;
                        let mut eager: Option<(Readback, Readback)> = None;
                        for mode in 0..3 {
                            for replay in 0..2 {
                                for out in outputs.iter().chain(&consumers) {
                                    out.poison(s, rows, *pattern == "sticky")?;
                                }
                                for arm in 0..4 {
                                    if mode == 0 {
                                        call(
                                            &case,
                                            &loaded,
                                            arm,
                                            rows,
                                            &input,
                                            &input.values,
                                            &outputs[arm],
                                            &s.stream,
                                            false,
                                        )?;
                                        call(
                                            &case,
                                            &loaded,
                                            0,
                                            rows,
                                            &input,
                                            &outputs[arm].buffers[0],
                                            &consumers[arm],
                                            &s.stream,
                                            false,
                                        )?;
                                    } else {
                                        graphs[mode - 1][arm]
                                            .launch(&s.stream)
                                            .map_err(|e| e.to_string())?;
                                    }
                                }
                                s.stream.synchronize().map_err(|e| e.to_string())?;
                                let reads = outputs
                                    .iter()
                                    .map(|o| o.read(s))
                                    .collect::<LabResult<Vec<_>>>()?;
                                let consumer_reads = consumers
                                    .iter()
                                    .map(|o| o.read(s))
                                    .collect::<LabResult<Vec<_>>>()?;
                                for arm in 0..4 {
                                    for (old, new) in [
                                        (&reads[0], &reads[arm]),
                                        (&consumer_reads[0], &consumer_reads[arm]),
                                    ] {
                                        compare(old,new).map_err(|e| format!("{} rank={rank} rows={rows} live={live} {pattern} mode={mode} replay={replay} arm={arm}: {e}",case.label()))?;
                                        guards(&case, rows, live, *pattern == "sticky", new)?;
                                    }
                                    if arm > 0 {
                                        comparisons += 1;
                                    }
                                }
                                if let Some((ref a, ref b)) = eager {
                                    compare(a, &reads[0])?;
                                    compare(b, &consumer_reads[0])?;
                                } else {
                                    eager = Some((reads[0].clone(), consumer_reads[0].clone()));
                                }
                                if live > 0
                                    && *pattern == "finite"
                                    && (reads[0][2][..width].iter().all(|b| *b == 0)
                                        || reads[0][4][OFFSET * 4..OFFSET * 4 + 4] != [0; 4])
                                {
                                    return Err(
                                        "finite pattern did not exercise a valid nonzero row"
                                            .into(),
                                    );
                                }
                                if live > 0
                                    && matches!(*pattern, "nan_last" | "sticky")
                                    && reads[0][4][OFFSET * 4..OFFSET * 4 + 4] == [0; 4]
                                {
                                    return Err("invalid/sticky row did not retain status".into());
                                }
                            }
                        }
                    }
                }
                let expected = outputs[0].read(s)?;
                for out in &outputs[1..] {
                    compare(&expected, &out.read(s)?)?;
                    for role in 0..5 {
                        let at = if role == 4 { OFFSET * 4 } else { 0 };
                        let mut changed = expected[role].clone();
                        changed[at] ^= 1;
                        upload(&out.buffers[role], &changed, &s.stream)?;
                        if compare(&expected, &out.read(s)?).is_ok() {
                            return Err("device output mutation escaped comparison".into());
                        }
                        upload(&out.buffers[role], &expected[role], &s.stream)?;
                        negatives += 1;
                    }
                }
                records.push(json!({"case":case.label(),"rank":rank,"rows":rows,"patterns":patterns.len(),"live_counts":(0..=rows).chain([rows+7]).collect::<Vec<_>>() }));
            }
        }
    }
    let expected = 3 * 4 * patterns.len() * ((2 + 2) + (4 + 2)) * 3 * 2 * 3;
    if comparisons != expected || negatives != 3 * 4 * 2 * 3 * 5 || records.len() != 24 {
        return Err(format!(
            "coverage mismatch: comparisons={comparisons}/{expected}, negatives={negatives}"
        ));
    }
    eprintln!(
        "SMALL_NORM_CORRECTNESS comparisons={comparisons} negatives={negatives} cases={} PASS",
        records.len()
    );
    Ok(
        json!({"comparisons":comparisons,"negative_controls":negatives,"cases":records,"memcheck":memcheck,"modes":["eager","graph","pdl-dependent-graph"],"replays":2}),
    )
}
