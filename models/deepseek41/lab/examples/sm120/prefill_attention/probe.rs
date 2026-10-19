use super::{REFERENCE, entries};
use kernel_core::backend::{Backend, Need};
use kernel_core::latent_attention::LatentAttentionQuery;
use kernel_harness::{LabResult, Session, eviction::EvictionMode, kit::cases, launch};
use kernels_sm120::{
    backend::Sm120,
    latent::{sources, sources_prefill},
};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as Arg, Module, Stream};

#[path = "ablation.rs"]
mod ablations;
#[path = "fixture.rs"]
mod fixture;
pub use ablations::run as ablation;
use fixture::{HEADS, Inputs, ROWS, SLOTS, WIDTH, WINDOW};

struct Arm {
    workspace: DeviceBuffer,
    output: DeviceBuffer,
    attention: Function,
    candidate: bool,
}
struct Rig {
    entry: sources::Entry,
    inputs: Inputs,
    arms: [Arm; 2],
    merge: Function,
    _modules: [Module; 2],
}

impl Rig {
    fn new(s: &Session, entry: sources::Entry, seed: u32) -> LabResult<Self> {
        let LatentAttentionQuery::Attention {
            page_size,
            second_fp4_keys,
            heads,
            latent,
            ..
        } = entry.query
        else {
            unreachable!()
        };
        if (heads, latent) != (HEADS as u32, WIDTH as u32) {
            return Err("expected TP4 local attention geometry".into());
        }
        let reference = s.module(REFERENCE)?;
        let candidate = s.module("latent_sources_prefill_deepseek41_sm120")?;
        let make_arm = |module: &Module, candidate: bool| -> LabResult<Arm> {
            let symbol = if candidate {
                sources_prefill::symbol(&entry)
            } else {
                entry.symbol()
            };
            let attention = module.function(&symbol).map_err(|e| e.to_string())?;
            let plan = if candidate {
                sources_prefill::plan(&entry, ROWS)
            } else {
                entry.plan(ROWS)
            }
            .map_err(|e| e.to_string())?;
            attention
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
            let resident = attention
                .max_active_clusters(launch::config(plan)?, false)
                .map_err(|e| e.to_string())?;
            println!(
                "RESOURCE prefill_attention symbol={symbol} shared={} resident_ctas={resident}",
                plan.dynamic_shared_bytes
            );
            Ok(Arm {
                workspace: s.allocate(entry.workspace_bytes(ROWS) as usize)?,
                output: s.allocate(ROWS as usize * HEADS * WIDTH * 2)?,
                attention,
                candidate,
            })
        };
        Ok(Self {
            arms: [make_arm(&reference, false)?, make_arm(&candidate, true)?],
            inputs: Inputs::new(s, page_size, second_fp4_keys, seed)?,
            merge: reference
                .function(&entry.merge_symbol())
                .map_err(|e| e.to_string())?,
            entry,
            _modules: [reference, candidate],
        })
    }

    fn arguments(&self, arm: usize, window_budget: u32, compressed_budget: u32) -> Vec<Arg<'_>> {
        let input = &self.inputs;
        let selected = &self.arms[arm];
        let LatentAttentionQuery::Attention {
            softmax_scale_bits,
            second_fp4_keys,
            ..
        } = self.entry.query
        else {
            unreachable!()
        };
        let mut args = vec![
            Arg::Buffer(&input.q),
            Arg::Buffer(&input.row_batch),
            Arg::Buffer(&input.live),
            Arg::Buffer(&input.window_codes),
            Arg::Buffer(&input.window_scales),
            Arg::Buffer(&input.item_slots),
            Arg::Buffer(&input.window_keys),
            Arg::Buffer(&selected.workspace),
            Arg::U64(u64::from(SLOTS) * WIDTH as u64),
            Arg::U64(u64::from(SLOTS) * (WIDTH / 32) as u64),
            Arg::U32(SLOTS),
            Arg::U32(1),
            Arg::U32(1),
            Arg::U32(WINDOW),
            Arg::U32(window_budget),
            Arg::U32(softmax_scale_bits),
        ];
        if second_fp4_keys {
            args.extend([
                Arg::Buffer(&input.compressed_pool),
                Arg::Buffer(&input.compressed_keys),
                Arg::Buffer(&input.tables),
                Arg::U64(u64::from(input.page) * (WIDTH / 2 + WIDTH / 16) as u64),
                Arg::U32(input.pages),
                Arg::U32(input.pages),
                Arg::U32(1),
                Arg::U32(512),
                Arg::U32(compressed_budget),
            ]);
        }
        args
    }

    fn attention(&self, stream: &Stream, arm: usize) -> LabResult<()> {
        let selected = &self.arms[arm];
        let args = self.arguments(arm, WINDOW, self.inputs.compressed_budget);
        let plan = if selected.candidate {
            sources_prefill::plan(&self.entry, ROWS)
        } else {
            self.entry.plan(ROWS)
        }
        .map_err(|e| e.to_string())?;
        launch::launch_with(&selected.attention, stream, launch::config(plan)?, &args)
    }

    fn merge(&self, stream: &Stream, arm: usize) -> LabResult<()> {
        launch::launch_with(
            &self.merge,
            stream,
            launch::config(self.entry.merge_plan(ROWS).map_err(|e| e.to_string())?)?,
            &[
                Arg::Buffer(&self.arms[arm].workspace),
                Arg::Buffer(&self.inputs.live),
                Arg::Buffer(&self.arms[arm].output),
                Arg::Buffer(&self.inputs.sinks),
                Arg::Buffer(&self.inputs.cos_sin),
            ],
        )
    }

    fn combined(&self, stream: &Stream, arm: usize) -> LabResult<()> {
        self.attention(stream, arm)?;
        self.merge(stream, arm)
    }

    fn poison(&self, s: &Session) -> LabResult<()> {
        let poison = vec![0xa5; self.arms[0].workspace.byte_len()];
        for arm in &self.arms {
            arm.workspace
                .copy_from_bytes(&poison, &s.stream)
                .map_err(|e| e.to_string())?;
            arm.output
                .copy_from_bytes(&poison[..arm.output.byte_len()], &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn compare(&self, s: &Session, live: u32) -> LabResult<()> {
        for (name, buffers, stride) in [
            (
                "workspace",
                [&self.arms[0].workspace, &self.arms[1].workspace],
                HEADS * (WIDTH + 2) * 4,
            ),
            (
                "output",
                [&self.arms[0].output, &self.arms[1].output],
                HEADS * WIDTH * 2,
            ),
        ] {
            let mut first = vec![0; buffers[0].byte_len()];
            let mut second = vec![0; buffers[1].byte_len()];
            buffers[0]
                .copy_to_bytes(&mut first, &s.stream)
                .map_err(|e| e.to_string())?;
            buffers[1]
                .copy_to_bytes(&mut second, &s.stream)
                .map_err(|e| e.to_string())?;
            if let Some(at) = first.iter().zip(&second).position(|(a, b)| a != b) {
                return Err(format!(
                    "{} {name} differs at byte {at}: old={} new={}",
                    self.entry.symbol(),
                    first[at],
                    second[at]
                ));
            }
            if first[live as usize * stride..]
                .iter()
                .any(|&byte| byte != 0xa5)
            {
                return Err(format!("{name}: inactive rows overwritten at live={live}"));
            }
        }
        Ok(())
    }
}

fn oracle(s: &Session, entry: &sources::Entry) -> LabResult<()> {
    let backend = Sm120::new().with_latent_sources(entry.clone());
    let prepared = backend
        .latent_attention_sources(&entry.query, Need::rows(4))
        .map_err(|e| e.to_string())?;
    let case = cases::latent_sources::numerical_case(entry.query, 4, prepared)?;
    let images = case.images();
    let reference = s.module(REFERENCE)?;
    let candidate = s.module("latent_sources_prefill_deepseek41_sm120")?;
    let old_lookup = |symbol: &str| reference.function(symbol).map_err(|e| e.to_string());
    let new_lookup = |symbol: &str| {
        if symbol == entry.symbol() {
            candidate
                .function(&sources_prefill::symbol(entry))
                .map_err(|e| e.to_string())
        } else {
            old_lookup(symbol)
        }
    };
    let old = case.launch(s, &old_lookup, &images, 4)?;
    let new = case.launch(s, &new_lookup, &images, 4)?;
    if old != new {
        return Err("small oracle case differs bitwise".into());
    }
    case.judge_outputs(&images, 4, &old)?;
    case.judge_outputs(&images, 4, &new)?;
    Ok(())
}

fn correctness(s: &Session, rank: usize, rig: &Rig) -> LabResult<()> {
    oracle(s, &rig.entry)?;
    let graph = s.capture(|stream| {
        rig.combined(stream, 0)?;
        rig.combined(stream, 1)
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
        "PASS prefill_attention_bitwise rank={rank} variant={} rows={ROWS} workspace=true output=true replay=true f64=true",
        variant(&rig.entry)
    );
    Ok(())
}

fn variant(entry: &sources::Entry) -> &'static str {
    match entry.query {
        LatentAttentionQuery::Attention {
            second_fp4_keys: false,
            ..
        } => "window",
        LatentAttentionQuery::Attention { page_size: 32, .. } => "mixed32",
        _ => "mixed64",
    }
}
fn ratio(entry: &sources::Entry) -> u32 {
    match variant(entry) {
        "window" => 0,
        "mixed32" => 2,
        _ => 1,
    }
}

pub fn traffic() -> LabResult<serde_json::Value> {
    fixture::traffic()
}

fn timings(s: &Session, rank: usize, rig: &Rig) -> LabResult<()> {
    for (layer, &compression) in FLASH
        .shape
        .compress_ratios
        .iter()
        .take(FLASH.layers() as usize)
        .enumerate()
    {
        if compression != ratio(&rig.entry) {
            continue;
        }
        rig.inputs
            .set_case(s, layer as u32 + rank as u32 * 40 + 100, false, ROWS)?;
        let attention = [0, 1]
            .map(|arm| s.capture(|stream| rig.attention(stream, arm)))
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let combined = [0, 1]
            .map(|arm| s.capture(|stream| rig.combined(stream, arm)))
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let label = format!("prefill_attention.rank{rank}.layer{layer}");
        let measure = |graphs: &[runtime_cuda::GraphExec]| {
            kernel_harness::time_ab(s, 2, 7, [EvictionMode::Write; 2], |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })
        };
        let attention = kernel_harness::with_timing_label(&format!("{label}.attention"), || {
            measure(&attention)
        })?;
        let total =
            kernel_harness::with_timing_label(&format!("{label}.combined"), || measure(&combined))?;
        let mut merge = Vec::new();
        for arm in 0..2 {
            merge.push(kernel_harness::with_timing_label(
                &format!("{label}.merge.{arm}"),
                || {
                    kernel_harness::time_after(
                        s,
                        || rig.attention(&s.stream, arm),
                        || rig.merge(&s.stream, arm),
                    )
                },
            )?);
        }
        let gflop = 4.0
            * f64::from(ROWS)
            * HEADS as f64
            * WIDTH as f64
            * f64::from(WINDOW + rig.inputs.compressed_budget)
            / 1e9;
        println!(
            "TIMING prefill_attention rank={rank} layer={layer} variant={} rows={ROWS} old_attention_us={} candidate_attention_us={} old_merge_us={} candidate_merge_us={} old_total_us={} candidate_total_us={} gflop={gflop} paired_ratio={}",
            variant(&rig.entry),
            attention.arms[0].median_ms * 1000.0,
            attention.arms[1].median_ms * 1000.0,
            merge[0],
            merge[1],
            total.arms[0].median_ms * 1000.0,
            total.arms[1].median_ms * 1000.0,
            total.paired_ratio
        );
    }
    Ok(())
}

fn exclusive(sessions: &[Session]) -> LabResult<()> {
    if sessions.len() != 2 {
        return Err("prefill lab requires an allocated GPU pair".into());
    }
    if [
        "CUDA_MPS_PIPE_DIRECTORY",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
        "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING",
    ]
    .iter()
    .any(|key| std::env::var_os(key).is_some_and(|v| !v.is_empty()))
    {
        return Err("prefill lab requires exclusive contexts".into());
    }
    if std::path::Path::new("/tmp/nvidia-mps/control").exists() {
        return Err("prefill lab requires exclusive contexts".into());
    }
    for session in sessions {
        let usable = session
            .context
            .usable_multiprocessors()
            .map_err(|e| e.to_string())?;
        if usable == 0 || usable != session.context.device().multiprocessor_count() as u32 {
            return Err("prefill lab refuses limited SM affinity".into());
        }
    }
    Ok(())
}

pub fn run(sessions: &[Session]) -> LabResult<()> {
    exclusive(sessions)?;
    let entries = entries()?;
    // Gate all types on both devices before emitting any timing results.
    for (rank, session) in sessions.iter().enumerate() {
        for entry in &entries {
            let rig = Rig::new(session, entry.clone(), rank as u32 + 1)?;
            correctness(session, rank, &rig)?;
        }
    }
    for (rank, session) in sessions.iter().enumerate() {
        for entry in &entries {
            let rig = Rig::new(session, entry.clone(), rank as u32 + 1)?;
            timings(session, rank, &rig)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn served_prefill_cases_cover_every_layer_and_preserve_the_merge() {
        let entries = entries().unwrap();
        let counts = entries
            .iter()
            .map(|entry| {
                FLASH
                    .shape
                    .compress_ratios
                    .iter()
                    .take(FLASH.layers() as usize)
                    .filter(|&&r| r == ratio(entry))
                    .count()
            })
            .collect::<Vec<_>>();
        assert_eq!(counts, [2, 20, 18]);
        for entry in entries {
            assert_eq!(entry.plan(ROWS).unwrap().grid, [ROWS, 2, 1]);
            assert_eq!(entry.merge_plan(ROWS).unwrap().grid, [ROWS, 16, 1]);
            assert!(
                sources_prefill::entry_spec(&entry)
                    .unwrap()
                    .body
                    .contains("sources_prefill::attention")
            );
            let backend = Sm120::new().with_latent_sources(entry.clone());
            let prepared = backend
                .latent_attention_sources(&entry.query, Need::rows(4))
                .unwrap();
            let case = cases::latent_sources::numerical_case(entry.query, 4, prepared).unwrap();
            assert_eq!(case.capacity(), 4);
            assert!(!case.images().bytes("q").unwrap().is_empty());
        }
    }
}
