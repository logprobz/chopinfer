//! Software profiling only. Diagnostic modes are never serving entries.
use super::*;
use sources_prefill::ProfileMode;
use std::collections::BTreeMap;

const MODES: [ProfileMode; 3] = [ProfileMode::Full, ProfileMode::Loads, ProfileMode::Resident];
const STAGES: [&str; 6] = [
    "query_setup",
    "gather_decode_stage",
    "qk",
    "softmax",
    "pv",
    "epilogue",
];
struct Probes {
    functions: Vec<Function>,
    stamps: DeviceBuffer,
    _module: Module,
}
impl Probes {
    fn new(s: &Session, rig: &Rig) -> LabResult<Self> {
        let module = s.module("latent_sources_prefill_profile_deepseek41_sm120")?;
        let plan = rig.entry.plan(ROWS).map_err(|e| e.to_string())?;
        let functions = MODES
            .into_iter()
            .map(|mode| {
                let symbol = sources_prefill::profile_symbol(&rig.entry, mode);
                let f = module.function(&symbol).map_err(|e| e.to_string())?;
                f.set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
                Ok(f)
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            functions,
            stamps: s.allocate(
                ROWS as usize * (HEADS / 8) * sources_prefill::PROFILE_WORDS as usize * 8,
            )?,
            _module: module,
        })
    }
    fn launch(
        &self,
        rig: &Rig,
        stream: &Stream,
        mode: usize,
        window: u32,
        compressed: u32,
        sample: bool,
    ) -> LabResult<()> {
        let mut args = rig.arguments(1, window, compressed);
        args.push(if sample {
            Arg::Buffer(&self.stamps)
        } else {
            Arg::NullBuffer
        });
        launch::launch_with(
            &self.functions[mode],
            stream,
            launch::config(rig.entry.plan(ROWS).map_err(|e| e.to_string())?)?,
            &args,
        )
    }
    fn samples(&self, s: &Session, rank: usize, source: &str, mode: ProfileMode) -> LabResult<()> {
        let mut bytes = vec![0; self.stamps.byte_len()];
        self.stamps
            .copy_to_bytes(&mut bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        let records = bytes
            .chunks_exact(sources_prefill::PROFILE_WORDS as usize * 8)
            .map(|row| {
                row.chunks_exact(8)
                    .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let summary = summarize(&records)?;
        println!(
            "STAGES prefill_ablation rank={rank} source={source} mode={} ctas={} cycles={}",
            mode.name(),
            records.len(),
            summary
        );
        s.receipt(
            "prefill-ablation-stages",
            BTreeMap::from([
                ("rank".into(), serde_json::json!(rank)),
                ("source".into(), serde_json::json!(source)),
                ("mode".into(), serde_json::json!(mode.name())),
                ("summary".into(), summary),
                (
                    "record_fields".into(),
                    serde_json::json!([
                        "start",
                        "end",
                        "smid",
                        "query_setup",
                        "gather_decode_stage",
                        "qk",
                        "softmax",
                        "pv",
                        "epilogue"
                    ]),
                ),
                ("records".into(), serde_json::json!(records)),
            ]),
        )?;
        Ok(())
    }
}

fn summarize(records: &[Vec<u64>]) -> LabResult<serde_json::Value> {
    if records.is_empty()
        || records
            .iter()
            .any(|r| r.len() != 9 || r[1] <= r[0] || r[3..].iter().sum::<u64>() != r[1] - r[0])
    {
        return Err("invalid or incomplete stage clock records".into());
    }
    let mut result = serde_json::Map::new();
    for (i, label) in STAGES.iter().enumerate() {
        let mut values = records.iter().map(|r| r[3 + i]).collect::<Vec<_>>();
        values.sort_unstable();
        result.insert(
            (*label).into(),
            serde_json::json!({"median":values[values.len()/2],"p95":values[values.len()*95/100]}),
        );
    }
    let mut total = records.iter().map(|r| r[1] - r[0]).collect::<Vec<_>>();
    total.sort_unstable();
    result.insert(
        "total".into(),
        serde_json::json!({"median":total[total.len()/2],"p95":total[total.len()*95/100]}),
    );
    result.insert("scope".into(),serde_json::json!("warp-zero elapsed cycles per CTA; do not sum across CTAs or treat as hardware stall counters"));
    Ok(serde_json::Value::Object(result))
}

fn full_correctness(s: &Session, rank: usize, rig: &Rig, probes: &Probes) -> LabResult<()> {
    let graph = s.capture(|stream| {
        rig.combined(stream, 0)?;
        probes.launch(rig, stream, 0, WINDOW, rig.inputs.compressed_budget, true)?;
        rig.merge(stream, 1)
    })?;
    for (generation, live) in [ROWS, 0, 1, ROWS / 2, ROWS - 1, ROWS]
        .into_iter()
        .enumerate()
    {
        rig.inputs
            .set_case(s, generation as u32 + 1, generation != 0, live)?;
        rig.poison(s)?;
        graph.launch(&s.stream).map_err(|e| e.to_string())?;
        rig.compare(s, live)?;
    }
    println!(
        "PASS prefill_ablation_full rank={rank} variant={} rows={ROWS} workspace=true output=true replay=true",
        variant(&rig.entry)
    );
    Ok(())
}

pub fn run(sessions: &[Session]) -> LabResult<()> {
    exclusive(sessions)?;
    let entries = entries()?;
    // Preserve every prior gate, then check that stamps do not change full arithmetic.
    for (rank, s) in sessions.iter().enumerate() {
        for entry in &entries {
            let rig = Rig::new(s, entry.clone(), rank as u32 + 1)?;
            correctness(s, rank, &rig)?;
            let probes = Probes::new(s, &rig)?;
            full_correctness(s, rank, &rig, &probes)?;
        }
    }
    for (rank, s) in sessions.iter().enumerate() {
        let entry = entries
            .iter()
            .find(|e| variant(e) == "mixed64")
            .unwrap()
            .clone();
        let rig = Rig::new(s, entry, rank as u32 + 1)?;
        rig.inputs.set_case(s, 120, false, ROWS)?;
        let probes = Probes::new(s, &rig)?;
        for (source, window, compressed) in [
            ("mixed", 128, 512),
            ("window", 128, 0),
            ("selected", 0, 512),
        ] {
            let original = s.capture(|stream| {
                launch::launch_with(
                    &rig.arms[0].attention,
                    stream,
                    launch::config(rig.entry.plan(ROWS).map_err(|e| e.to_string())?)?,
                    &rig.arguments(0, window, compressed),
                )
            })?;
            let timing = kernel_harness::time_named(
                s,
                &format!("ablation.{rank}.{source}.reference"),
                2,
                7,
                |stream| original.launch(stream).map_err(|e| e.to_string()),
            )?;
            println!(
                "TIMING prefill_ablation rank={rank} source={source} mode=reference rows={ROWS} median_us={}",
                timing.median_ms * 1000.0
            );
            for (mode, kind) in MODES.into_iter().enumerate() {
                let graph = s.capture(|stream| {
                    probes.launch(&rig, stream, mode, window, compressed, false)
                })?;
                let timing = kernel_harness::time_named(
                    s,
                    &format!("ablation.{rank}.{source}.{}", kind.name()),
                    2,
                    7,
                    |stream| graph.launch(stream).map_err(|e| e.to_string()),
                )?;
                println!(
                    "TIMING prefill_ablation rank={rank} source={source} mode={} rows={ROWS} median_us={}",
                    kind.name(),
                    timing.median_ms * 1000.0
                );
                probes
                    .stamps
                    .zero_at_async(0, probes.stamps.byte_len(), &s.stream)
                    .map_err(|e| e.to_string())?;
                probes.launch(&rig, &s.stream, mode, window, compressed, true)?;
                probes.samples(s, rank, source, kind)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stage_records_partition_each_cta_without_combining_unsynchronized_clocks() {
        let records = vec![
            vec![100, 121, 0, 1, 2, 3, 4, 5, 6],
            vec![10000, 10042, 7, 2, 4, 6, 8, 10, 12],
        ];
        let result = summarize(&records).unwrap();
        assert_eq!(result["gather_decode_stage"]["median"], 4);
        assert_eq!(result["total"]["median"], 42);
        let mut bad = records;
        bad[1][8] = 11;
        assert!(summarize(&bad).is_err());
        assert!(summarize(&[]).is_err());
    }
}
