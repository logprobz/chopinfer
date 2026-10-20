//! Compare empty-unit scheduling directly with the currently served routes.
use super::*;
use sources_prefill::heads::{empty as candidate, next};

const MODULE_EMPTY: &str = "latent_sources_prefill_head_empty_deepseek41_sm120";
const MODULE_FINISH: &str = "latent_sources_prefill_head_finish_deepseek41_sm120";

#[derive(Clone, Debug)]
struct Case {
    layer: u32,
    first: u32,
    rows: u32,
    decoded: bool,
    name: &'static str,
}
fn cases(offset: u32, small: bool) -> Vec<Case> {
    if small {
        return [2, 20]
            .into_iter()
            .flat_map(|layer| {
                [false, true].into_iter().map(move |decoded| Case {
                    layer,
                    first: 4095,
                    rows: 65,
                    decoded,
                    name: "control",
                })
            })
            .collect();
    }
    let mut result: Vec<_> = [2, 20]
        .into_iter()
        .map(|layer| Case {
            layer,
            first: 0,
            rows: ROWS,
            decoded: false,
            name: "full",
        })
        .collect();
    if offset > 0 {
        result.extend([2, 20].into_iter().map(|layer| Case {
            layer,
            first: 0,
            rows: ROWS,
            decoded: true,
            name: "cached",
        }));
    }
    result.extend((20..40).map(|layer| {
        let rows = 64 + 128 * (39 - layer);
        Case {
            layer,
            first: ROWS - rows,
            rows,
            decoded: false,
            name: "retained",
        }
    }));
    for first in [0, 4096] {
        result.extend([2, 20].into_iter().map(|layer| Case {
            layer,
            first,
            rows: 4096,
            decoded: false,
            name: "part",
        }));
    }
    result
}
fn served_fused(case: &Case, offset: u32) -> bool {
    (case.name == "full" && offset == 0)
        || case.name == "cached"
        || (matches!(case.name, "retained" | "part")
            && matches!(offset, 0 | 24576)
            && (case.rows == 4096 || case.rows >= 320))
}
fn spec() -> LabResult<kernel_build::CubinSpec> {
    let mut all = topology_entries(4)?;
    all.extend(
        all.clone()
            .iter()
            .map(cache::decoded_entry)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    );
    candidate::cubin_spec(bundle_deepseek41::sm120::latent::profile(&FLASH), &all)
        .map_err(|e| e.to_string())
}
pub fn build(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    if out.canonicalize()?.starts_with(repository) {
        return Err("snapshot must be external".into());
    }
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let lock = out.join("empty.lock");
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
    let manifest_path = out.join("snapshot.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
    let modules = manifest["modules"]
        .as_array_mut()
        .ok_or("missing modules")?;
    for b in built {
        modules.retain(|m| m["name"] != b.name);
        modules.push(serde_json::json!({"name":b.name,"sha256":b.sha256}));
    }
    std::fs::write(manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(())
}
fn fused_plan(
    entry: &sources::Entry,
    rows: u32,
) -> LabResult<Option<kernel_core::contract::LaunchPlan>> {
    if rows == 0 {
        return Ok(None);
    }
    heads::plan(entry, rows)
        .map(Some)
        .map_err(|e| e.to_string())
}

fn validate_partitions() -> LabResult<()> {
    for entry in topology_entries(4)? {
        for offset in [0, 8192, 16384, 24576] {
            for case in cases(offset, false).into_iter().chain(cases(offset, true)) {
                let first = first_part(case.rows);
                let mut covered = 0;
                for rows in [first, case.rows - first] {
                    if let Some(plan) = fused_plan(&entry, rows)? {
                        if plan.grid[0] != rows || rows == 0 {
                            return Err(
                                "partition plan does not cover its live row interval".into()
                            );
                        }
                        covered += rows;
                    }
                }
                if covered != case.rows {
                    return Err("partition plans lost query rows".into());
                }
            }
        }
    }
    Ok(())
}

pub fn snapshot(out: &Path) -> Result<kernel_harness::Cubins, Box<dyn std::error::Error>> {
    validate_partitions()?;
    let mut specs = super::super::super::specs()?;
    specs.extend(super::super::extra_specs()?);
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
    specs.push(next::cubin_spec(
        bundle_deepseek41::sm120::latent::profile(&FLASH),
        &all,
        next::Mode::Finish,
    )?);
    specs.push(spec()?);
    super::super::super::snapshot_with(out, specs)
}
struct Rig {
    head: HeadRig,
    functions: [Function; 2],
    _modules: [Module; 2],
    served: bool,
}
impl Rig {
    fn new(s: &Session, case: &Case, offset: u32) -> LabResult<Self> {
        let entry = topology_entries(4)?
            .into_iter()
            .find(|e| {
                variant(e)
                    == if case.layer < 20 {
                        "mixed32"
                    } else {
                        "mixed64"
                    }
            })
            .ok_or("missing entry")?;
        let head = HeadRig::new(
            s,
            entry,
            4,
            case.rows,
            case.decoded,
            19,
            case.name == "control" && case.decoded,
        )?;
        let entry = if case.decoded {
            cache::decoded_entry(&head.entry).map_err(|e| e.to_string())?
        } else {
            head.entry.clone()
        };
        let modules = [s.module(MODULE_FINISH)?, s.module(MODULE_EMPTY)?];
        let functions = [
            modules[0]
                .function(&next::symbol(&entry, case.decoded, next::Mode::Finish))
                .map_err(|e| e.to_string())?,
            modules[1]
                .function(&candidate::symbol(&entry, case.decoded))
                .map_err(|e| e.to_string())?,
        ];
        for f in &functions {
            f.set_max_dynamic_shared_memory(heads::SHARED_BYTES)
                .map_err(|e| e.to_string())?;
        }
        Ok(Self {
            head,
            functions,
            _modules: modules,
            served: served_fused(case, offset),
        })
    }
    fn fused(
        &self,
        stream: &Stream,
        function: usize,
        arm: usize,
        span: (u32, u32),
        live: &DeviceBuffer,
        audit: bool,
    ) -> LabResult<()> {
        let r = &self.head;
        let (first, rows) = span;
        let Some(plan) = fused_plan(&r.entry, rows)? else {
            return Ok(());
        };
        let mut args = r.arguments(Candidate::Shared, arm, first, rows, live)?;
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
                &r.arms[arm].output,
                first as usize * r.heads * WIDTH * 2,
                rows as usize * r.heads * WIDTH * 2,
            ),
        ]);
        launch::launch_with(
            &self.functions[function],
            stream,
            launch::config(plan)?,
            &args,
        )
    }
    fn whole(&self, stream: &Stream, function: usize, arm: usize, audit: bool) -> LabResult<()> {
        self.fused(
            stream,
            function,
            arm,
            (0, self.head.capacity),
            &self.head.input.live,
            audit,
        )
    }
    fn reference(&self, stream: &Stream, audit: bool) -> LabResult<()> {
        if self.served {
            self.whole(stream, 0, 0, audit)
        } else {
            self.head.whole(stream, Candidate::Current, 0)
        }
    }
    fn checks(&self, s: &Session) -> LabResult<()> {
        let r = &self.head;
        let graphs = [true, false]
            .into_iter()
            .map(|audit| {
                s.capture(|stream| {
                    r.producer(stream)?;
                    self.reference(stream, true)?;
                    self.whole(stream, 1, 1, audit)
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        for (generation, live) in [r.capacity, r.capacity - 1, 0, r.capacity]
            .into_iter()
            .enumerate()
        {
            r.live(s, live, 51 + generation as u32)?;
            for (graph, audit) in graphs.iter().zip([true, false]) {
                r.poison(s)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                r.compare_mode(s, live, audit)?;
            }
        }
        // Qualify the existing fused body independently of empty-unit skipping.
        // Its later native and retained routes have not been admitted in serving.
        for audit in [true, false] {
            let route = s.capture(|stream| {
                r.producer(stream)?;
                self.reference(stream, true)?;
                self.whole(stream, 0, 1, audit)
            })?;
            r.poison(s)?;
            route.launch(&s.stream).map_err(|e| e.to_string())?;
            r.compare_mode(s, r.capacity, audit)?;
        }
        // Invalid batches make all units empty while preserving the poisoned tail.
        r.input
            .row_batch
            .copy_from_bytes(&(-1_i32).to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        r.live(s, 1, 61)?;
        for (graph, audit) in graphs.iter().zip([true, false]) {
            r.poison(s)?;
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
            r.compare_mode(s, 1, audit)?;
        }
        r.input
            .row_batch
            .copy_from_bytes(&0_i32.to_le_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        // Force complete internal holes, then a nonempty unit, on the first row.
        // Keep later window keys so leading-empty and trailing-empty updates run.
        let holes = vec![0xff; 64 * 4];
        for unit in [0, 2, 4, 7] {
            r.input
                .compressed_keys
                .copy_from_bytes_at(unit * 64 * 4, &holes, &s.stream)
                .map_err(|e| e.to_string())?;
        }
        r.input
            .window_keys
            .copy_from_bytes(&holes, &s.stream)
            .map_err(|e| e.to_string())?;
        r.live(s, r.capacity, 71)?;
        for (graph, audit) in graphs.iter().zip([true, false]) {
            r.poison(s)?;
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
            r.compare_mode(s, r.capacity, audit)?;
        }
        let parts = s.capture(|stream| {
            r.producer(stream)?;
            self.reference(stream, true)?;
            let a = first_part(r.capacity);
            for (part, (first, rows)) in [(0, a), (a, r.capacity - a)].into_iter().enumerate() {
                self.fused(stream, 1, 1, (first, rows), &r.part_live[part], false)?;
            }
            Ok(())
        })?;
        r.poison(s)?;
        parts.launch(&s.stream).map_err(|e| e.to_string())?;
        r.compare_mode(s, r.capacity, false)?;
        if r.capacity == 65 {
            r.input.set_case(s, 83, true, r.capacity)?;
            r.live(s, r.capacity, 83)?;
            for (graph, audit) in graphs.iter().zip([true, false]) {
                r.poison(s)?;
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
                r.compare_mode(s, r.capacity, audit)?;
            }
        }
        Ok(())
    }
    fn timing(&self, s: &Session, label: &str) -> LabResult<()> {
        let r = &self.head;
        r.live(s, r.capacity, 81)?;
        let reference = s.capture(|stream| {
            r.producer(stream)?;
            self.reference(stream, false)
        })?;
        let fused = s.capture(|stream| {
            r.producer(stream)?;
            self.whole(stream, 0, 0, false)
        })?;
        let empty = s.capture(|stream| {
            r.producer(stream)?;
            self.whole(stream, 1, 1, false)
        })?;
        let measure = |a: &runtime_cuda::GraphExec, b: &runtime_cuda::GraphExec| {
            kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                [a, b][arm].launch(stream).map_err(|e| e.to_string())
            })
        };
        let route = measure(&reference, &fused)?;
        let skip = measure(&fused, &empty)?;
        println!(
            "TIMING empty_units {label} served_us={} fused_us={} skip_reference_us={} candidate_us={} samples=3 warmup=1 producer_included=true",
            route.arms[0].median_ms * 1000.0,
            route.arms[1].median_ms * 1000.0,
            skip.arms[0].median_ms * 1000.0,
            skip.arms[1].median_ms * 1000.0
        );
        Ok(())
    }
}
pub fn run(sessions: &[Session], lists: &Path, offset: u32, mode: &str) -> LabResult<()> {
    if ![0, 8192, 16384, 24576].contains(&offset) {
        return Err("invalid offset".into());
    }
    exclusive(sessions)?;
    let manifest = serde_json::from_slice(&std::fs::read(lists).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let small = mode == "empty-memcheck";
    for (rank, s) in sessions.iter().enumerate() {
        for case in cases(offset, small) {
            let data = layer_lists_in(lists, &manifest, case.layer, offset, false)?
                .slice(case.first, case.rows)?;
            let rig = Rig::new(s, &case, offset)?;
            data.install(s, &rig.head.input, case.rows)?;
            let rope = (0..=case.rows)
                .flat_map(|row| {
                    let angle = (offset + case.first + row) as f32 * 0.0007;
                    (0..32).flat_map(move |_| {
                        [angle.cos(), angle.sin()]
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                    })
                })
                .collect::<Vec<_>>();
            rig.head
                .input
                .cos_sin
                .copy_from_bytes(&rope, &s.stream)
                .map_err(|e| e.to_string())?;
            let label = format!(
                "rank={rank} offset={offset} layer={} case={} first={} rows={} decoded={} served_fused={}",
                case.layer, case.name, case.first, case.rows, case.decoded, rig.served
            );
            println!("PROGRESS empty_units {label} mode={mode}");
            if mode == "empty-time" {
                rig.timing(s, &label)?;
            } else {
                rig.checks(s)?;
                println!(
                    "PASS empty_units {label} partial=true output=true poison=true audit_replays=6 timed_replays=6 route_replays=2 adversarial_replays={} partitions=2 cache_guards=true actual_selected=true selected_sha256={}",
                    if small { 2 } else { 0 },
                    digest(&data.selected)
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
    fn empty_partition_does_not_plan_a_zero_row_launch() {
        assert_eq!(first_part(64), 64);
        validate_partitions().unwrap();
        for entry in topology_entries(4).unwrap() {
            assert!(fused_plan(&entry, 0).unwrap().is_none());
            assert_eq!(fused_plan(&entry, 64).unwrap().unwrap().grid[0], 64);
        }
    }

    #[test]
    fn routes_cover_the_served_offsets_and_every_retained_count() {
        for offset in [0, 8192, 16384, 24576] {
            let cases = cases(offset, false);
            assert_eq!(cases.len(), if offset == 0 { 26 } else { 28 });
            assert_eq!(
                cases
                    .iter()
                    .filter(|c| c.name == "retained")
                    .map(|c| c.rows)
                    .sum::<u32>(),
                25600
            );
            assert_eq!(
                cases.iter().filter(|c| served_fused(c, offset)).count(),
                if offset == 0 || offset == 24576 {
                    24
                } else {
                    2
                }
            );
        }
        assert_eq!(cases(0, true).len(), 4);
    }
}
