//! Profile the locked paired/cache bodies before choosing another schedule.
use super::*;
use sources_prefill::profiling;
use std::collections::BTreeMap;

const PROBE: &str = "latent_sources_prefill_pair_profile_deepseek41_sm120";
const CTAS: usize = ROWS as usize / 2 * (HEADS / 8);

fn spec() -> LabResult<kernel_build::CubinSpec> {
    let native = entries()?;
    let mut all = native.clone();
    all.extend(
        native
            .iter()
            .map(cache::decoded_entry)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    );
    profiling::cubin_spec(bundle_deepseek41::sm120::latent::profile(&FLASH), &all)
        .map_err(|e| e.to_string())
}

pub fn build(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out)?;
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    if out.canonicalize()?.starts_with(repo) {
        return Err("probe snapshot must be outside repository".into());
    }
    let lock = out.join("probe-checks.lock");
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let built = kernel_build::build(
        &toolchain,
        &[spec()?],
        out,
        &lock,
        kernel_build::BuildOptions {
            lock_update: !lock.exists(),
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )?;
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("snapshot.json"))?)?;
    let modules = manifest["modules"]
        .as_array_mut()
        .ok_or("missing frozen module manifest")?;
    modules.retain(|m| m["name"] != PROBE);
    modules.extend(
        built
            .iter()
            .map(|b| serde_json::json!({"name":b.name,"sha256":b.sha256})),
    );
    std::fs::write(
        out.join("snapshot.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(())
}

pub fn snapshot(out: &Path) -> Result<kernel_harness::Cubins, Box<dyn std::error::Error>> {
    let mut specs = super::specs()?;
    specs.push(spec()?);
    super::snapshot_with(out, specs)
}

struct Probe {
    function: [Function; 2],
    stamps: DeviceBuffer,
    _module: Module,
}
impl Probe {
    fn new(s: &Session, rig: &Cached) -> LabResult<Self> {
        let module = s.module(PROBE)?;
        let function = [
            rig.rig.entry.clone(),
            cache::decoded_entry(&rig.rig.entry).map_err(|e| e.to_string())?,
        ]
        .into_iter()
        .map(|entry| {
            let f = module
                .function(&profiling::symbol(&entry))
                .map_err(|e| e.to_string())?;
            f.set_max_dynamic_shared_memory(
                Schedule::Pair
                    .plan(&entry, ROWS)
                    .map_err(|e| e.to_string())?
                    .dynamic_shared_bytes,
            )
            .map_err(|e| e.to_string())?;
            Ok(f)
        })
        .collect::<LabResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| "expected native and cached probes")?;
        let stamps = s.allocate(CTAS * profiling::WORDS * 8)?;
        rig.audit_with(&[&stamps])?;
        Ok(Self {
            function,
            stamps,
            _module: module,
        })
    }
    fn launch(&self, rig: &Cached, stream: &Stream, decoded: bool, sample: bool) -> LabResult<()> {
        let mut args = arguments(rig, 1, decoded);
        args.push(if sample {
            Arg::Buffer(&self.stamps)
        } else {
            Arg::NullBuffer
        });
        launch::launch_with(
            &self.function[usize::from(decoded)],
            stream,
            launch::config(
                Schedule::Pair
                    .plan(&rig.rig.entry, ROWS)
                    .map_err(|e| e.to_string())?,
            )?,
            &args,
        )
    }
    fn samples(&self, s: &Session, label: &str, units: u64) -> LabResult<serde_json::Value> {
        let mut bytes = vec![0; self.stamps.byte_len()];
        self.stamps
            .copy_to_bytes(&mut bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        let records = bytes
            .chunks_exact(profiling::WORDS * 8)
            .map(|row| {
                row.chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let summary = summarize(&records, CTAS, units)?;
        s.receipt(
            "prefill-paired-stages",
            BTreeMap::from([
                ("case".into(), serde_json::json!(label)),
                ("summary".into(), summary.clone()),
                (
                    "fields".into(),
                    serde_json::json!([
                        "start", "end", "smid", "setup", "gather", "qk", "softmax", "pv",
                        "epilogue", "same", "shifted", "distinct"
                    ]),
                ),
                ("records".into(), serde_json::json!(records)),
            ]),
        )?;
        Ok(summary)
    }
}

fn arguments(rig: &Cached, arm: usize, decoded: bool) -> Vec<Arg<'_>> {
    let mut args = rig
        .rig
        .arguments(arm, WINDOW, rig.rig.inputs.compressed_budget);
    if decoded {
        args[3] = rig.region(&rig.layout.window);
        args[4] = Arg::NullBuffer;
        args[8] = Arg::U64(u64::from(SLOTS) * WIDTH as u64 * 2);
        args[9] = Arg::U64(0);
        if rig.pages.is_some() {
            args[16] = rig.region(&rig.layout.compressed);
            args[19] = Arg::U64(u64::from(rig.rig.inputs.page) * WIDTH as u64 * 2);
        }
    }
    args
}
fn locked(rig: &Cached, stream: &Stream, decoded: bool) -> LabResult<()> {
    launch::launch_with(
        &rig.rig.arms[usize::from(decoded)].attention,
        stream,
        launch::config(
            Schedule::Pair
                .plan(&rig.rig.entry, ROWS)
                .map_err(|e| e.to_string())?,
        )?,
        &arguments(rig, 0, decoded),
    )
}

fn summarize(records: &[Vec<u64>], expected: usize, units: u64) -> LabResult<serde_json::Value> {
    if records.len() != expected
        || expected == 0
        || records.iter().any(|r| {
            r.len() != profiling::WORDS
                || r[1] <= r[0]
                || r[3..9].iter().sum::<u64>() != r[1] - r[0]
                || r[9..12].iter().sum::<u64>() != units
        })
    {
        return Err("incomplete paired clock partition or unit counts".into());
    }
    let total = records.iter().map(|r| r[1] - r[0]).sum::<u64>();
    let mut summary = serde_json::Map::new();
    for (i, name) in profiling::STAGES.iter().enumerate() {
        let mut values = records.iter().map(|r| r[3 + i]).collect::<Vec<_>>();
        values.sort_unstable();
        summary.insert(
            (*name).into(),
            serde_json::json!({"median":values[expected/2],"p95":values[expected*95/100],
            "fraction":values.iter().sum::<u64>() as f64/total as f64}),
        );
    }
    summary.insert("reuse".into(),serde_json::json!({"same":records.iter().map(|r|r[9]).sum::<u64>(),
        "shifted":records.iter().map(|r|r[10]).sum::<u64>(),"distinct":records.iter().map(|r|r[11]).sum::<u64>()}));
    summary.insert("ctas".into(), serde_json::json!(expected));
    summary.insert("scope".into(),serde_json::json!("CTA elapsed-cycle partition with one diagnostic softmax/PV barrier per pass; not hardware counters or additive wall milliseconds"));
    Ok(summary.into())
}

pub fn run(sessions: &[Session], lists: &Path, offset: u32, timing: bool) -> LabResult<()> {
    if ![0, 24576].contains(&offset) {
        return Err("stage probe expects offset zero or 24576".into());
    }
    exclusive(sessions)?;
    let manifest = serde_json::from_slice(&std::fs::read(lists).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    for (rank, s) in sessions.iter().enumerate() {
        for entry in entries()? {
            let variant = variant(&entry);
            let layer = match variant {
                "window" => 0,
                "mixed32" => 2,
                _ => 20,
            };
            let rig = Cached::new(s, entry.clone(), rank as u32 + 1)?;
            let probe = Probe::new(s, &rig)?;
            let data = layer_lists_in(lists, &manifest, layer, offset, false)?;
            data.install(s, &rig.rig.inputs, ROWS)?;
            if !timing {
                oracle(s, &entry, Schedule::Pair)?;
            }
            for decoded in [false, true] {
                let label = format!(
                    "rank={rank} offset={offset} variant={variant} format={}",
                    if decoded { "cached" } else { "native" }
                );
                println!(
                    "PROGRESS paired_stages {label} phase={}",
                    if timing { "time" } else { "check" }
                );
                if !timing {
                    for sample in [false, true] {
                        let graph = s.capture(|stream| {
                            if decoded {
                                rig.producer(stream)?;
                            }
                            locked(&rig, stream, decoded)?;
                            rig.rig.merge(stream, 0)?;
                            probe.launch(&rig, stream, decoded, sample)?;
                            rig.rig.merge(stream, 1)
                        })?;
                        for (generation, live) in [ROWS, ROWS - 1, 0, ROWS].into_iter().enumerate()
                        {
                            rig.rig
                                .inputs
                                .live
                                .copy_from_bytes(&live.to_le_bytes(), &s.stream)
                                .map_err(|e| e.to_string())?;
                            rig.rig
                                .inputs
                                .q
                                .copy_from_bytes(
                                    &fixture::query(2, generation as u32 + 31),
                                    &s.stream,
                                )
                                .map_err(|e| e.to_string())?;
                            rig.poison(s)?;
                            probe
                                .stamps
                                .copy_from_bytes(&vec![0; probe.stamps.byte_len()], &s.stream)
                                .map_err(|e| e.to_string())?;
                            graph.launch(&s.stream).map_err(|e| e.to_string())?;
                            rig.rig.compare(s, live)?;
                            if decoded {
                                rig.check_cache(s, live)?;
                            }
                        }
                    }
                    probe.samples(s, &label, if variant == "window" { 2 } else { 10 })?;
                    println!(
                        "PASS paired_stages {label} rows=8192 sampled=true null=true workspace=true output=true replay=true f64=true partitions=true"
                    );
                    continue;
                }
                rig.rig
                    .inputs
                    .live
                    .copy_from_bytes(&ROWS.to_le_bytes(), &s.stream)
                    .map_err(|e| e.to_string())?;
                if decoded {
                    rig.producer(&s.stream)?;
                }
                let bodies = [
                    s.capture(|stream| locked(&rig, stream, decoded))?,
                    s.capture(|stream| probe.launch(&rig, stream, decoded, false))?,
                ];
                let body =
                    kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                        bodies[arm].launch(stream).map_err(|e| e.to_string())
                    })?;
                let sampled = s.capture(|stream| probe.launch(&rig, stream, decoded, true))?;
                let sampled_body =
                    kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                        if arm == 0 {
                            bodies[0].launch(stream).map_err(|e| e.to_string())
                        } else {
                            sampled.launch(stream).map_err(|e| e.to_string())
                        }
                    })?;
                let graphs = [false, true]
                    .map(|profile| {
                        s.capture(|stream| {
                            if decoded {
                                rig.producer(stream)?;
                            }
                            if profile {
                                probe.launch(&rig, stream, decoded, false)?;
                            } else {
                                locked(&rig, stream, decoded)?;
                            }
                            rig.rig.merge(stream, usize::from(profile))
                        })
                    })
                    .into_iter()
                    .collect::<LabResult<Vec<_>>>()?;
                let total =
                    kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                    })?;
                let merge = s.capture(|stream| rig.rig.merge(stream, 0))?;
                let merge =
                    kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |_, stream| {
                        merge.launch(stream).map_err(|e| e.to_string())
                    })?;
                let producer_us = if decoded {
                    let graph = s.capture(|stream| rig.producer(stream))?;
                    kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |_, stream| {
                        graph.launch(stream).map_err(|e| e.to_string())
                    })?
                    .arms[0]
                        .median_ms
                        * 1000.0
                } else {
                    0.0
                };
                probe
                    .stamps
                    .copy_from_bytes(&vec![0; probe.stamps.byte_len()], &s.stream)
                    .map_err(|e| e.to_string())?;
                probe.launch(&rig, &s.stream, decoded, true)?;
                let summary = probe.samples(s, &label, if variant == "window" { 2 } else { 10 })?;
                println!("STAGES paired_stages {label} cycles={summary}");
                println!(
                    "TIMING paired_stages {label} rows=8192 body_us={} probe_body_us={} sampled_reference_us={} sampled_body_us={} total_us={} probe_total_us={} producer_us={producer_us} merge_us={} samples=3 warmup=1",
                    body.arms[0].median_ms * 1000.0,
                    body.arms[1].median_ms * 1000.0,
                    sampled_body.arms[0].median_ms * 1000.0,
                    sampled_body.arms[1].median_ms * 1000.0,
                    total.arms[0].median_ms * 1000.0,
                    total.arms[1].median_ms * 1000.0,
                    merge.arms[0].median_ms * 1000.0
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_probe_keeps_each_native_and_decoded_parameter_prefix() {
        let spec = spec().unwrap();
        assert_eq!(spec.entries.len(), 6);
        for reference in entries().unwrap() {
            for entry in [reference.clone(), cache::decoded_entry(&reference).unwrap()] {
                let original = entry.entry_specs().unwrap().remove(0);
                let probe = spec
                    .entries
                    .iter()
                    .find(|e| e.symbol == profiling::symbol(&entry))
                    .unwrap();
                assert_eq!(&probe.params[..probe.params.len() - 1], original.params);
                assert_eq!(probe.params.len(), original.params.len() + 1);
                assert_eq!(probe.max_threads, 256);
                assert_eq!(probe.min_blocks, 1);
                assert!(probe.body.ends_with(",stamps);"));
            }
        }
    }
    #[test]
    fn paired_clocks_reject_missing_partitions_and_wrong_reuse_counts() {
        let mut records = vec![vec![100, 121, 0, 1, 2, 3, 4, 5, 6, 1, 1, 8]; 2];
        let summary = summarize(&records, 2, 10).unwrap();
        let fraction = profiling::STAGES
            .iter()
            .map(|name| summary[name]["fraction"].as_f64().unwrap())
            .sum::<f64>();
        assert!((fraction - 1.0).abs() < 1e-12);
        assert!(summarize(&[], 0, 10).is_err());
        assert!(summarize(&records, 1, 10).is_err());
        assert!(summarize(&records, 2, 2).is_err());
        records[0][3] += 1;
        assert!(summarize(&records, 2, 10).is_err());
    }
}
