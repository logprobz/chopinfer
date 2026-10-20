//! Qualify packed key staging on the serving dependency cone's row intervals.
use super::*;
use model_deepseek41_contract::prefill_cone::{Item, Plan};
use std::{ops::Range, path::Path};

pub(super) fn intervals(offset: u32) -> LabResult<Vec<(u32, Range<u32>)>> {
    if ![0, 24576].contains(&offset) {
        return Err("retained probe expects offset zero or 24576".into());
    }
    let cone = Plan::new(
        &FLASH,
        &[Item {
            rows: 0..ROWS,
            prefix: offset,
        }],
        &[ROWS - 1],
        false,
        64,
    )?;
    let mut rows = (0..cone.last_source)
        .map(|id| (id, 0..ROWS))
        .collect::<Vec<_>>();
    for layer in cone.layers {
        if layer.outputs.ranges().len() != 1 {
            return Err("expected one retained interval per layer".into());
        }
        rows.push((layer.id, layer.outputs.ranges()[0].clone()));
    }
    if rows.len() != 40 {
        return Err("retained probe must cover all forty layers".into());
    }
    Ok(rows)
}

fn captured(
    path: &Path,
    manifest: &serde_json::Value,
    layer: u32,
    offset: u32,
    rows: &Range<u32>,
) -> LabResult<cache::Lists> {
    // The two window layers have identical key positions and no selected list.
    cache::layer_lists_in(
        path,
        manifest,
        if layer == 1 { 0 } else { layer },
        offset,
        false,
    )?
    .slice(rows.start, rows.end - rows.start)
}

fn manifest(path: &Path) -> LabResult<serde_json::Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

fn progress(label: &str, stage: &str, started: std::time::Instant) {
    let unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    println!(
        "PROGRESS retained_attention {label} stage={stage} elapsed_ms={} unix_ms={unix_ms}",
        started.elapsed().as_millis()
    );
}

pub fn check_inputs(path: &Path, offset: u32) -> LabResult<()> {
    let manifest = manifest(path)?;
    for (layer, rows) in intervals(offset)? {
        captured(path, &manifest, layer, offset, &rows)?;
    }
    println!("PASS retained_inputs offset={offset} layers=40 actual_selected=true origins=true");
    Ok(())
}

pub fn run(sessions: &[Session], path: &Path, offset: u32, timing: bool) -> LabResult<()> {
    exclusive(sessions)?;
    let started = std::time::Instant::now();
    let manifest = manifest(path)?;
    let spans = intervals(offset)?;
    for (rank, s) in sessions.iter().enumerate() {
        for schedule in [Schedule::Vector, Schedule::Pair] {
            for entry in entries()? {
                let group = format!(
                    "rank={rank} offset={offset} variant={} candidate={}",
                    variant(&entry),
                    schedule.name()
                );
                progress(&group, "fixture-begin", started);
                let mut rig = Rig::scheduled(s, entry.clone(), rank as u32 + 1, schedule)?;
                progress(&group, "fixture-end", started);
                if !timing {
                    oracle(s, &entry, schedule)?;
                    progress(&group, "oracle-end", started);
                }
                for (layer, span) in &spans {
                    if FLASH.shape.compress_ratios[*layer as usize] != ratio(&entry) {
                        continue;
                    }
                    // Full-width lower layers share one dispatch. Time one
                    // representative; gate every layer's actual selected list.
                    if timing && (1..20).contains(layer) && *layer != 2 {
                        continue;
                    }
                    rig.rows = span.end - span.start;
                    let label = format!(
                        "rank={rank} layer={layer} offset={offset} first={} rows={} candidate={}",
                        span.start,
                        rig.rows,
                        schedule.name()
                    );
                    progress(&label, "inputs-begin", started);
                    let lists = captured(path, &manifest, *layer, offset, span)?;
                    lists.install(s, &rig.inputs, rig.rows)?;
                    progress(&label, "inputs-end", started);
                    if timing {
                        let graphs = [0, 1]
                            .map(|arm| s.capture(|stream| rig.combined(stream, arm)))
                            .into_iter()
                            .collect::<LabResult<Vec<_>>>()?;
                        let total = kernel_harness::time_ab(
                            s,
                            1,
                            3,
                            [EvictionMode::Write; 2],
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                        )?;
                        println!(
                            "TIMING retained_attention {label} old_total_us={} candidate_total_us={} ratio={} samples=3 warmup=1 actual_selected=true",
                            total.arms[0].median_ms * 1000.0,
                            total.arms[1].median_ms * 1000.0,
                            total.paired_ratio
                        );
                    } else {
                        let graph = s.capture(|stream| {
                            rig.combined(stream, 0)?;
                            rig.combined(stream, 1)
                        })?;
                        for (generation, live) in [rig.rows, rig.rows - 1, 0, rig.rows]
                            .into_iter()
                            .enumerate()
                        {
                            rig.inputs
                                .q
                                .copy_from_bytes(
                                    &fixture::query(2, generation as u32 + 7),
                                    &s.stream,
                                )
                                .map_err(|e| e.to_string())?;
                            lists.install(s, &rig.inputs, live)?;
                            rig.poison(s)?;
                            graph.launch(&s.stream).map_err(|e| e.to_string())?;
                            rig.compare(s, live)?;
                            progress(
                                &format!("{label} generation={generation} live={live}"),
                                "replay-end",
                                started,
                            );
                        }
                        println!(
                            "PASS retained_attention {label} workspace=true output=true replay=true f64=true guard=true actual_selected=true"
                        );
                    }
                }
                if !timing {
                    progress(&group, "controls-begin", started);
                    // Different metadata on the same graph must refresh each
                    // shared tile, including invalid pages and odd final CTAs.
                    rig.rows = 65;
                    let graph = s.capture(|stream| {
                        rig.combined(stream, 0)?;
                        rig.combined(stream, 1)
                    })?;
                    for live in [0, 1, 63, 65] {
                        rig.inputs.set_case(s, 90, true, live)?;
                        rig.poison(s)?;
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        rig.compare(s, live)?;
                    }
                    for shifted in [false, true] {
                        rig.inputs.set_case(s, 91, false, 65)?;
                        rig.inputs.share_selected_pairs(s, 91, shifted)?;
                        rig.poison(s)?;
                        graph.launch(&s.stream).map_err(|e| e.to_string())?;
                        rig.compare(s, 65)?;
                    }
                    println!(
                        "PASS retained_controls rank={rank} offset={offset} variant={} candidate={} invalid_pages=true odd_rows=true shared_selected=true",
                        variant(&entry),
                        schedule.name()
                    );
                    progress(&group, "controls-end", started);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_spans_match_the_served_upper_cone_at_both_offsets() {
        for offset in [0, 24576] {
            let spans = intervals(offset).unwrap();
            assert!(spans[..20].iter().all(|(_, rows)| *rows == (0..ROWS)));
            assert_eq!(
                spans[20..]
                    .iter()
                    .map(|(_, r)| r.end - r.start)
                    .sum::<u32>(),
                25600
            );
            for (layer, rows) in &spans[20..] {
                assert_eq!(rows.end, ROWS);
                assert_eq!(rows.end - rows.start, 64 + 128 * (39 - layer));
            }
        }
        assert!(intervals(1).is_err());
    }
}
