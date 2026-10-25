use super::*;
const CHAIN: usize = 32;

pub(super) fn run(s: &Session) -> LabResult<Value> {
    let loaded = Loaded::load(s)?;
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
                let values = data::rows_of(width, rows, "finite", 0x195a_dafe + rank);
                let values = if case.f32() {
                    values
                } else {
                    values
                        .chunks_exact(4)
                        .flat_map(|v| {
                            ((u32::from_ne_bytes(v.try_into().unwrap()) >> 16) as u16).to_ne_bytes()
                        })
                        .collect()
                };
                upload(&input.values, &values, &s.stream)?;
                upload(
                    &input.weight,
                    &data::weight_of(width, "finite", 0x0dd_ba11 + rank),
                    &s.stream,
                )?;
                upload(&input.live, &rows.to_ne_bytes(), &s.stream)?;
                let outputs = (0..4)
                    .map(|_| Output::new(s, &case, rows))
                    .collect::<LabResult<Vec<_>>>()?;
                for o in &outputs {
                    o.poison(s, rows, false)?;
                }
                for (cache, repeats) in [("cold", 1), ("warm", CHAIN)] {
                    let graphs = (0..4)
                        .map(|arm| {
                            s.capture(|stream| {
                                (0..repeats).try_for_each(|_| {
                                    call(
                                        &case,
                                        &loaded,
                                        arm,
                                        rows,
                                        &input,
                                        &input.values,
                                        &outputs[arm],
                                        stream,
                                        true,
                                    )
                                })
                            })
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let label = format!(
                        "small-norm:{}:rank={rank}:rows={rows}:{cache}",
                        case.label()
                    );
                    let pair = |arm: usize| {
                        kernel_harness::with_timing_label(&format!("{label}:arm={arm}"), || {
                            kernel_harness::time_ab_prepared(
                                s,
                                3,
                                15,
                                [EvictionMode::Read; 2],
                                |side, stream| {
                                    if cache == "warm" {
                                        graphs[if side == 0 { 0 } else { arm }]
                                            .launch(stream)
                                            .map_err(|e| e.to_string())?;
                                    }
                                    Ok(())
                                },
                                |side, stream| {
                                    graphs[if side == 0 { 0 } else { arm }]
                                        .launch(stream)
                                        .map_err(|e| e.to_string())
                                },
                            )
                        })
                    };
                    let aa = pair(0)?;
                    for arm in 1..4 {
                        let ab = pair(arm)?;
                        s.stream.synchronize().map_err(|e| e.to_string())?;
                        compare(&outputs[0].read(s)?, &outputs[arm].read(s)?)?;
                        records.push(json!({"case":case.label(),"rank":rank,"rows":rows,"cache":cache,"replays":repeats,
                            "warps":WARPS[arm-1],"aa_ratio":aa.paired_ratio,"ratio":ab.paired_ratio,
                            "old_us":ab.arms[0].median_ms*1000./repeats as f32,"new_us":ab.arms[1].median_ms*1000./repeats as f32}));
                    }
                }
            }
        }
    }
    if records.len() != 3 * 4 * 2 * 2 * 3 {
        return Err("timing coverage differs".into());
    }
    eprintln!("SMALL_NORM_TIMING cases={} PASS", records.len());
    Ok(json!({"cases":records,"aa_pairs":48,"ab_pairs":144,"rounds":15,"chain":CHAIN}))
}
