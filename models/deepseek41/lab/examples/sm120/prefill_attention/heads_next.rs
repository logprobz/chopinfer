//! Compare an exact fused epilogue with the qualified per-shape serving route.
use super::*;
use sources_prefill::{heads::next as candidate, profiling};
use std::collections::BTreeMap;

const FINISH: &str = "latent_sources_prefill_head_finish_deepseek41_sm120";
const PROFILE: &str = "latent_sources_prefill_head_profile_deepseek41_sm120";
const PAIR_PROFILE: &str = "prefill_head_current_pair_profile_sm120";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Check,
    Time,
    Profile,
}
#[derive(Clone, Debug)]
struct Case {
    name: String,
    tp: u32,
    first: u32,
    rows: u32,
    layer: u32,
    decoded: bool,
    profile: bool,
    timing: bool,
}
fn cases(offset: u32) -> Vec<Case> {
    let mut cases = Vec::new();
    for layer in [2, 20] {
        cases.push(Case {
            name: "full".into(),
            tp: 4,
            first: 0,
            rows: ROWS,
            layer,
            decoded: offset > 0,
            profile: matches!(offset, 0 | 24576),
            timing: true,
        });
    }
    if matches!(offset, 0 | 24576) {
        for (name, first) in [("partA", 0), ("partB", 4096)] {
            for layer in [2, 20] {
                cases.push(Case {
                    name: name.into(),
                    tp: 4,
                    first,
                    rows: 4096,
                    layer,
                    decoded: false,
                    profile: name == "partB" && layer == 2,
                    timing: true,
                });
            }
        }
        for layer in 20..40 {
            let rows = 64 + 128 * (39 - layer);
            cases.push(Case {
                name: format!("retained{layer}"),
                tp: 4,
                first: ROWS - rows,
                rows,
                layer,
                decoded: false,
                profile: matches!(rows, 64 | 704 | 2496),
                timing: true,
            });
        }
        for tp in [1, 2] {
            for layer in [2, 20] {
                cases.push(Case {
                    name: "topology".into(),
                    tp,
                    first: 4095,
                    rows: 129,
                    layer,
                    decoded: offset > 0,
                    profile: false,
                    timing: false,
                });
            }
        }
    }
    if offset == 24576 {
        for layer in [2, 20] {
            cases.push(Case {
                name: "unaligned".into(),
                tp: 4,
                first: 4095,
                rows: 65,
                layer,
                decoded: true,
                profile: false,
                timing: false,
            });
        }
    }
    cases
}
fn baseline(case: &Case) -> Candidate {
    if case.name == "full" {
        if case.decoded {
            Candidate::Async
        } else {
            Candidate::Shared
        }
    } else {
        Candidate::Current
    }
}
fn replace_once(text: &mut String, old: &str, new: &str) -> LabResult<()> {
    if text.matches(old).count() != 1 {
        return Err("paired clock source anchor changed".into());
    }
    *text = text.replacen(old, new, 1);
    Ok(())
}
pub fn pair_source() -> LabResult<String> {
    let mut text = profiling::source().map_err(|e| e.to_string())?;
    replace_once(
        &mut text,
        "void reuse_profile_attention(",
        "void reuse_clock_attention(",
    )?;
    candidate::add_global_timer(&mut text).map_err(|e| e.to_string())?;
    Ok(text)
}
fn pair_cubin_spec(
    profile: kernels_sm120::Profile,
    entries: &[sources::Entry],
) -> LabResult<kernel_build::CubinSpec> {
    let mut spec = profiling::cubin_spec(profile, entries).map_err(|e| e.to_string())?;
    spec.name = PAIR_PROFILE.into();
    spec.headers = vec!["#include \"latent/sources_prefill_pair_clock.cuh\"".into()];
    for entry in &mut spec.entries {
        replace_once(
            &mut entry.body,
            "::reuse_profile_attention<",
            "::reuse_clock_attention<",
        )?;
    }
    Ok(spec)
}
fn new_specs() -> LabResult<Vec<kernel_build::CubinSpec>> {
    let mut all = Vec::new();
    for tp in [1, 2, 4] {
        all.extend(topology_entries(tp)?);
    }
    all.extend(
        all.clone()
            .iter()
            .map(cache::decoded_entry)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    );
    let profile = bundle_deepseek41::sm120::latent::profile(&FLASH);
    let pair = pair_cubin_spec(profile, &topology_entries(4)?).map_err(|e| e.to_string())?;
    Ok(vec![
        candidate::cubin_spec(profile, &all, candidate::Mode::Finish).map_err(|e| e.to_string())?,
        candidate::cubin_spec(profile, &all, candidate::Mode::Profile)
            .map_err(|e| e.to_string())?,
        pair,
    ])
}
pub fn build(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    if out.canonicalize()?.starts_with(repo) {
        return Err("epilogue snapshot must be outside repository".into());
    }
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("snapshot.json"))?)?;
    for spec in new_specs()? {
        let lock = out.join(format!("{}.lock", spec.name));
        let built = kernel_build::build(
            &toolchain,
            &[spec],
            out,
            &lock,
            kernel_build::BuildOptions {
                lock_update: !lock.exists(),
                emit_cargo_directives: false,
                profile_line_info: false,
            },
        )?;
        let modules = manifest["modules"]
            .as_array_mut()
            .ok_or("missing module manifest")?;
        for b in built {
            modules.retain(|m| m["name"] != b.name);
            modules.push(serde_json::json!({"name":b.name,"sha256":b.sha256}));
        }
        std::fs::write(
            out.join("snapshot.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
    }
    Ok(())
}
pub fn snapshot(out: &Path) -> Result<kernel_harness::Cubins, Box<dyn std::error::Error>> {
    let mut specs = super::super::super::specs()?;
    specs.extend(super::super::extra_specs()?);
    specs.extend(new_specs()?);
    super::super::super::snapshot_with(out, specs)
}

struct Probe {
    function: Function,
    stamps: DeviceBuffer,
    paired: bool,
    _module: Module,
}
struct Next {
    rig: HeadRig,
    fused: Function,
    probe: Option<Probe>,
    base: Candidate,
    _module: Module,
}
impl Next {
    fn new(s: &Session, case: &Case, entry: sources::Entry, seed: u32) -> LabResult<Self> {
        let rig = HeadRig::new(
            s,
            entry,
            case.tp,
            case.rows,
            case.decoded,
            seed,
            case.name == "unaligned",
        )?;
        let e = if case.decoded {
            cache::decoded_entry(&rig.entry).map_err(|e| e.to_string())?
        } else {
            rig.entry.clone()
        };
        let module = s.module(FINISH)?;
        let fused = module
            .function(&candidate::symbol(
                &e,
                case.decoded,
                candidate::Mode::Finish,
            ))
            .map_err(|e| e.to_string())?;
        fused
            .set_max_dynamic_shared_memory(heads::SHARED_BYTES)
            .map_err(|e| e.to_string())?;
        let base = baseline(case);
        let probe = if case.profile {
            let paired = base == Candidate::Current;
            let module = s.module(if paired { PAIR_PROFILE } else { PROFILE })?;
            let symbol = if paired {
                profiling::symbol(&e)
            } else {
                candidate::symbol(&e, case.decoded, candidate::Mode::Profile)
            };
            let function = module.function(&symbol).map_err(|e| e.to_string())?;
            let plan = base
                .plan(&rig.entry, case.rows)
                .map_err(|e| e.to_string())?;
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
            let ctas = plan.grid.iter().map(|&v| v as usize).product::<usize>();
            Some(Probe {
                function,
                stamps: s.allocate(ctas * candidate::WORDS * 8)?,
                paired,
                _module: module,
            })
        } else {
            None
        };
        let result = Self {
            rig,
            fused,
            probe,
            base,
            _module: module,
        };
        result.audit()?;
        Ok(result)
    }
    fn audit(&self) -> LabResult<()> {
        self.rig.audit()?;
        if let Some(probe) = &self.probe {
            let i = &self.rig.input;
            let mut buffers = vec![
                &i.q,
                &i.row_batch,
                &i.live,
                &i.window_codes,
                &i.window_scales,
                &i.item_slots,
                &i.window_keys,
                &i.compressed_pool,
                &i.compressed_keys,
                &i.tables,
                &i.sinks,
                &i.cos_sin,
                &self.rig.arena,
            ];
            for arm in &self.rig.arms {
                buffers.extend([&arm.partial, &arm.output]);
            }
            buffers.extend(&self.rig.part_live);
            buffers.extend(self.rig.unaligned.iter());
            let first = probe
                .stamps
                .device_address_at(0, probe.stamps.byte_len())
                .map_err(|e| e.to_string())?;
            for buffer in buffers {
                let other = buffer
                    .device_address_at(0, buffer.byte_len())
                    .map_err(|e| e.to_string())?;
                if first < other + buffer.byte_len() as u64
                    && other < first + probe.stamps.byte_len() as u64
                {
                    return Err("stage stamps overlap a live attention region".into());
                }
            }
        }
        Ok(())
    }
    fn fused(
        &self,
        stream: &Stream,
        first: u32,
        rows: u32,
        live: &DeviceBuffer,
        audit: bool,
    ) -> LabResult<()> {
        if rows == 0 {
            return Ok(());
        }
        let r = &self.rig;
        let mut args = r.arguments(Candidate::Shared, 1, first, rows, live)?;
        if !audit {
            args[7] = Arg::NullBuffer;
        }
        args.extend([
            Arg::Buffer(&r.input.sinks),
            region(
                &r.input.cos_sin,
                first as usize * 64 * 4,
                rows as usize * 64 * 4,
            ),
            region(
                &r.arms[1].output,
                first as usize * r.heads * WIDTH * 2,
                rows as usize * r.heads * WIDTH * 2,
            ),
        ]);
        launch::launch_with(
            &self.fused,
            stream,
            launch::config(heads::plan(&r.entry, rows).map_err(|e| e.to_string())?)?,
            &args,
        )
    }
    fn whole(&self, stream: &Stream, audit: bool) -> LabResult<()> {
        self.fused(stream, 0, self.rig.capacity, &self.rig.input.live, audit)
    }
    fn parts(&self, stream: &Stream) -> LabResult<()> {
        let a = first_part(self.rig.capacity);
        for (part, (first, rows)) in [(0, a), (a, self.rig.capacity - a)].into_iter().enumerate() {
            self.fused(stream, first, rows, &self.rig.part_live[part], false)?;
        }
        Ok(())
    }
    fn profile(&self, stream: &Stream, sample: bool) -> LabResult<()> {
        let p = self.probe.as_ref().ok_or("missing stage probe")?;
        let r = &self.rig;
        let mut args = r.arguments(self.base, 1, 0, r.capacity, &r.input.live)?;
        args.push(if sample {
            Arg::Buffer(&p.stamps)
        } else {
            Arg::NullBuffer
        });
        launch::launch_with(
            &p.function,
            stream,
            launch::config(
                self.base
                    .plan(&r.entry, r.capacity)
                    .map_err(|e| e.to_string())?,
            )?,
            &args,
        )
    }
    fn sinks_and_rope(&self, s: &Session, first: u32) -> LabResult<()> {
        let sinks: Vec<f32> = (0..self.rig.heads)
            .map(|h| [-8.0, 0.0, 8.0, 32.0][h % 4])
            .collect::<Vec<_>>();
        self.rig
            .input
            .sinks
            .copy_from_bytes(
                &sinks
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
                &s.stream,
            )
            .map_err(|e| e.to_string())?;
        let rope = (0..=self.rig.capacity)
            .flat_map(|row| {
                let angle = (first + row) as f32 * 0.0007;
                let pair = [angle.cos(), angle.sin()];
                (0..32).flat_map(move |_| pair.into_iter().flat_map(f32::to_le_bytes))
            })
            .collect::<Vec<_>>();
        self.rig
            .input
            .cos_sin
            .copy_from_bytes(&rope, &s.stream)
            .map_err(|e| e.to_string())
    }
    fn intermediate(&self) -> bool {
        self.base == Candidate::Current && self.rig.heads == 16 && !self.rig.decoded
    }
    fn checks(&self, s: &Session) -> LabResult<()> {
        let r = &self.rig;
        let graphs = [true, false]
            .into_iter()
            .map(|audit| {
                s.capture(|stream| {
                    r.producer(stream)?;
                    r.whole(stream, self.base, 0)?;
                    self.whole(stream, audit)
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        for (generation, live) in [r.capacity, r.capacity - 1, 0, r.capacity]
            .into_iter()
            .enumerate()
        {
            r.live(s, live, 31 + generation as u32)?;
            for (graph, audit) in graphs.iter().zip([true, false]) {
                r.poison(s)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                r.compare_mode(s, live, audit)?;
            }
        }
        if r.capacity == ROWS {
            r.input
                .row_batch
                .copy_from_bytes(&(-1_i32).to_le_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
            r.live(s, 1, 47)?;
            for (graph, audit) in graphs.iter().zip([true, false]) {
                r.poison(s)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                r.compare_mode(s, 1, audit)?;
            }
            r.input
                .row_batch
                .copy_from_bytes(&0_i32.to_le_bytes(), &s.stream)
                .map_err(|e| e.to_string())?;
        }
        if self.intermediate() {
            let head = s.capture(|stream| {
                r.producer(stream)?;
                r.whole(stream, self.base, 0)?;
                r.whole(stream, Candidate::Shared, 1)
            })?;
            r.live(s, r.capacity, 53)?;
            r.poison(s)?;
            head.launch(&s.stream).map_err(|e| e.to_string())?;
            r.compare(s, r.capacity)?;
        }
        let parts = s.capture(|stream| {
            r.producer(stream)?;
            r.whole(stream, self.base, 0)?;
            self.parts(stream)
        })?;
        r.live(s, r.capacity, 51)?;
        r.poison(s)?;
        parts.launch(&s.stream).map_err(|e| e.to_string())?;
        r.compare_mode(s, r.capacity, false)?;
        if let Some(p) = &self.probe {
            for sample in [false, true] {
                let graph = s.capture(|stream| {
                    r.producer(stream)?;
                    r.whole(stream, self.base, 0)?;
                    self.profile(stream, sample)?;
                    r.merge(stream, 1, 0, r.capacity, &r.input.live)
                })?;
                for live in [r.capacity, 0] {
                    r.live(s, live, 71)?;
                    r.poison(s)?;
                    p.stamps
                        .copy_from_bytes(&vec![0; p.stamps.byte_len()], &s.stream)
                        .map_err(|e| e.to_string())?;
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                    r.compare(s, live)?;
                    if live == 0 {
                        let mut stamps = vec![0; p.stamps.byte_len()];
                        p.stamps
                            .copy_to_bytes(&mut stamps, &s.stream)
                            .map_err(|e| e.to_string())?;
                        if stamps.iter().any(|&b| b != 0) {
                            return Err("inactive stage probe wrote stamps".into());
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn timing(&self, s: &Session, label: &str) -> LabResult<()> {
        let r = &self.rig;
        r.live(s, r.capacity, 61)?;
        let base = s.capture(|stream| {
            r.producer(stream)?;
            r.whole(stream, self.base, 0)
        })?;
        let head = if self.intermediate() {
            Some(s.capture(|stream| {
                r.producer(stream)?;
                r.whole(stream, Candidate::Shared, 1)
            })?)
        } else {
            None
        };
        let fused = s.capture(|stream| {
            r.producer(stream)?;
            self.whole(stream, false)
        })?;
        let measure = |a: &runtime_cuda::GraphExec, b: &runtime_cuda::GraphExec| {
            kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                [a, b][arm].launch(stream).map_err(|e| e.to_string())
            })
        };
        let before = head.as_ref().map(|h| measure(&base, h)).transpose()?;
        let after = measure(head.as_ref().unwrap_or(&base), &fused)?;
        let current = before
            .as_ref()
            .map_or(after.arms[0].median_ms, |t| t.arms[0].median_ms)
            * 1000.0;
        let head_time = before
            .as_ref()
            .map_or(after.arms[0].median_ms, |t| t.arms[1].median_ms)
            * 1000.0;
        println!(
            "TIMING head_finish {label} current_total_us={current} head_total_us={head_time} fused_reference_us={} fused_total_us={} ab_pairs={} samples=3 warmup=1 producer_attention_merge=true eliminated_partials=true",
            after.arms[0].median_ms * 1000.0,
            after.arms[1].median_ms * 1000.0,
            1 + usize::from(self.intermediate())
        );
        Ok(())
    }
    fn stages(&self, s: &Session, label: &str) -> LabResult<()> {
        let r = &self.rig;
        let p = self.probe.as_ref().ok_or("missing profile")?;
        r.live(s, r.capacity, 61)?;
        r.producer(&s.stream)?;
        let locked =
            s.capture(|stream| r.attention(stream, self.base, 0, 0, r.capacity, &r.input.live))?;
        let null = s.capture(|stream| self.profile(stream, false))?;
        let sampled = s.capture(|stream| self.profile(stream, true))?;
        let measure = |a: &runtime_cuda::GraphExec, b: &runtime_cuda::GraphExec| {
            kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                [a, b][arm].launch(stream).map_err(|e| e.to_string())
            })
        };
        let normal = measure(&locked, &null)?;
        let sampling = measure(&locked, &sampled)?;
        let merge = s.capture(|stream| r.merge(stream, 0, 0, r.capacity, &r.input.live))?;
        let merge_time = measure(&merge, &merge)?.arms[0].median_ms * 1000.0;
        let producer_time = if r.decoded {
            let producer = s.capture(|stream| r.producer(stream))?;
            measure(&producer, &producer)?.arms[0].median_ms * 1000.0
        } else {
            0.0
        };
        p.stamps
            .copy_from_bytes(&vec![0; p.stamps.byte_len()], &s.stream)
            .map_err(|e| e.to_string())?;
        sampled.launch(&s.stream).map_err(|e| e.to_string())?;
        let mut bytes = vec![0; p.stamps.byte_len()];
        p.stamps
            .copy_to_bytes(&mut bytes, &s.stream)
            .map_err(|e| e.to_string())?;
        let records = bytes
            .chunks_exact(candidate::WORDS * 8)
            .map(|row| {
                row.chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let summary = summarize(&records, r.capacity as usize, p.paired)?;
        s.receipt(
            "prefill-head-stages",
            BTreeMap::from([
                ("case".into(), serde_json::json!(label)),
                ("records".into(), serde_json::json!(records)),
                ("summary".into(), summary.clone()),
            ]),
        )?;
        println!("STAGES head_finish {label} cycles={summary}");
        println!(
            "PROFILE head_finish {label} body_us={} null_us={} sampled_reference_us={} sampled_us={} producer_us={producer_time} merge_us={merge_time} samples=3 warmup=1",
            normal.arms[0].median_ms * 1000.0,
            normal.arms[1].median_ms * 1000.0,
            sampling.arms[0].median_ms * 1000.0,
            sampling.arms[1].median_ms * 1000.0
        );
        Ok(())
    }
}
fn summarize(records: &[Vec<u64>], expected: usize, paired: bool) -> LabResult<serde_json::Value> {
    if records.len() != expected
        || expected == 0
        || records.iter().any(|r| {
            r.len() != candidate::WORDS
                || r[1] <= r[0]
                || r[13] <= r[12]
                || r[3..9].iter().sum::<u64>() != r[1] - r[0]
                || if paired {
                    r[9..12].iter().sum::<u64>() != 10
                } else {
                    r[9] != 10
                }
        })
    {
        return Err("incomplete head stage clock partition".into());
    }
    let total = records.iter().map(|r| r[1] - r[0]).sum::<u64>();
    let mut result = serde_json::Map::new();
    for (i, name) in candidate::STAGES.iter().enumerate() {
        let mut values = records.iter().map(|r| r[3 + i]).collect::<Vec<_>>();
        values.sort_unstable();
        result.insert((*name).into(),serde_json::json!({"median":values[expected/2],"p95":values[expected*95/100],"fraction":values.iter().sum::<u64>() as f64/total as f64}));
    }
    result.insert("ctas".into(), serde_json::json!(expected));
    if paired {
        result.insert(
            "reuse".into(),
            serde_json::json!({
                "same": records.iter().map(|r| r[9]).sum::<u64>(),
                "shifted": records.iter().map(|r| r[10]).sum::<u64>(),
                "distinct": records.iter().map(|r| r[11]).sum::<u64>(),
            }),
        );
    }
    let nanos = records.iter().map(|r| r[13] - r[12]).sum::<u64>();
    result.insert(
        "clock_ghz".into(),
        serde_json::json!(total as f64 / nanos as f64),
    );
    result.insert(
        "scope".into(),
        serde_json::json!(
            "CTA elapsed cycles including barriers; not additive wall times or hardware counters"
        ),
    );
    Ok(result.into())
}
pub fn run(sessions: &[Session], lists: &Path, offset: u32, phase: Phase) -> LabResult<()> {
    if ![0, 8192, 16384, 24576].contains(&offset) {
        return Err("invalid epilogue offset".into());
    }
    if phase == Phase::Profile && !matches!(offset, 0 | 24576) {
        return Err("no stage cases at this offset".into());
    }
    exclusive(sessions)?;
    let manifest = serde_json::from_slice(&std::fs::read(lists).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    for (rank, s) in sessions.iter().enumerate() {
        for case in cases(offset).into_iter().filter(|c| {
            phase == Phase::Check
                || if phase == Phase::Profile {
                    c.profile
                } else {
                    c.timing
                }
        }) {
            let entry = topology_entries(case.tp)?
                .into_iter()
                .find(|e| {
                    variant(e)
                        == if case.layer < 20 {
                            "mixed32"
                        } else {
                            "mixed64"
                        }
                })
                .ok_or("missing measured variant")?;
            let data = layer_lists_in(lists, &manifest, case.layer, offset, false)?
                .slice(case.first, case.rows)?;
            let next = Next::new(s, &case, entry, 13 + rank as u32)?;
            data.install(s, &next.rig.input, case.rows)?;
            next.sinks_and_rope(s, offset + case.first)?;
            let label = format!(
                "rank={rank} tp={} offset={offset} layer={} variant={} case={} first={} rows={} format={} baseline={} heads={} cache_align={}",
                case.tp,
                case.layer,
                variant(&next.rig.entry),
                case.name,
                case.first,
                case.rows,
                if case.decoded { "cached" } else { "native" },
                next.base.name(),
                next.rig.heads,
                if case.name == "unaligned" { 2 } else { 16 }
            );
            println!("PROGRESS head_finish {label} phase={phase:?}");
            match phase {
                Phase::Check => {
                    next.checks(s)?;
                    println!(
                        "PASS head_finish {label} partial=true output=true audit_replays=4 timed_replays=4 poison=true partitions=2 eliminated_partials=true sink_edges=true row_rope=true cache_guards=true disjoint=true probe_replays={} empty_replays={} intermediate_replays={} actual_selected=true selected_sha256={}",
                        if case.profile { 4 } else { 0 },
                        if case.rows == ROWS { 2 } else { 0 },
                        usize::from(next.intermediate()),
                        digest(&data.selected)
                    );
                }
                Phase::Time => next.timing(s, &label)?,
                Phase::Profile => next.stages(s, &label)?,
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_clock_header_matches_the_checked_source() {
        assert_eq!(
            pair_source().unwrap(),
            include_str!(
                "../../../../../../crates/kernels-sm120/cuda/latent/sources_prefill_pair_clock.cuh"
            )
        );
        let mut broken = String::from("missing paired source");
        assert!(candidate::add_global_timer(&mut broken).is_err());
    }
    #[test]
    fn epilogue_matrix_covers_retained_layers_parts_topologies_and_timed_shapes() {
        let offsets = [0, 8192, 16384, 24576];
        assert_eq!(
            offsets.map(|o| cases(o)
                .iter()
                .filter(|c| baseline(c) == Candidate::Current && c.tp == 4 && !c.decoded)
                .count()
                * 2),
            [48, 0, 0, 48]
        );
        assert_eq!(offsets.map(|o| cases(o).len() * 2), [60, 4, 4, 64]);
        assert_eq!(
            offsets.map(|o| cases(o).iter().filter(|c| c.timing).count() * 2),
            [52, 4, 4, 52]
        );
        assert_eq!(
            offsets.map(|o| cases(o).iter().filter(|c| c.profile).count() * 2),
            [12, 0, 0, 12]
        );
        for offset in [0, 24576] {
            let retained = cases(offset)
                .into_iter()
                .filter(|c| c.name.starts_with("retained"))
                .collect::<Vec<_>>();
            assert_eq!(retained.len(), 20);
            assert_eq!(retained.iter().map(|c| c.rows).sum::<u32>(), 25600);
            assert!(retained.iter().all(|c| c.first + c.rows == ROWS
                && !c.decoded
                && baseline(c) == Candidate::Current));
        }
    }
    #[test]
    fn stage_summary_rejects_incomplete_or_misattributed_records() {
        let valid = vec![vec![10, 70, 0, 10, 10, 10, 10, 10, 10, 10, 0, 0, 100, 140]];
        assert!(summarize(&valid, 1, false).is_ok());
        assert!(summarize(&valid, 1, true).is_ok());
        assert!(summarize(&valid, 0, false).is_err());
        assert!(summarize(&[], 1, false).is_err());
        for index in [1, 3, 9] {
            let mut bad = valid.clone();
            bad[0][index] += 1;
            assert!(summarize(&bad, 1, false).is_err());
        }
    }
}
