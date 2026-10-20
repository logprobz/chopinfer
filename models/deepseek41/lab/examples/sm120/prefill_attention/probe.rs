use super::{REFERENCE, entries};
use kernel_core::backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{LaunchPlan, Refusal};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentSourcesBuffers};
use kernel_harness::{LabResult, Session, eviction::EvictionMode, kit::cases, launch};
use kernels_sm120::{
    backend::Sm120,
    latent::{sources, sources_prefill},
};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as Arg, Module, Stream};

#[path = "ablation.rs"]
mod ablations;
#[path = "cache.rs"]
pub(super) mod cache;
#[path = "fixture.rs"]
mod fixture;
#[path = "retained.rs"]
pub(super) mod retained;
pub use ablations::run as ablation;
use fixture::{HEADS, Inputs, ROWS, SLOTS, WIDTH, WINDOW};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Schedule {
    Stage32,
    Vector,
    Pair,
}
impl Schedule {
    fn name(self) -> &'static str {
        match self {
            Self::Stage32 => "stage32",
            Self::Vector => "vector",
            Self::Pair => "pair",
        }
    }
    fn module(self) -> &'static str {
        match self {
            Self::Stage32 => "latent_sources_prefill_deepseek41_sm120",
            Self::Vector => "latent_sources_prefill_reuse1_deepseek41_sm120",
            Self::Pair => "latent_sources_prefill_reuse2_deepseek41_sm120",
        }
    }
    fn symbol(self, entry: &sources::Entry) -> String {
        match self {
            Self::Stage32 => sources_prefill::symbol(entry),
            _ => sources_prefill::reuse_symbol(entry, if self == Self::Pair { 2 } else { 1 }),
        }
    }
    fn plan(self, entry: &sources::Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
        match self {
            Self::Stage32 => sources_prefill::plan(entry, rows),
            _ => sources_prefill::reuse_plan(entry, rows, if self == Self::Pair { 2 } else { 1 }),
        }
    }
}

struct Arm {
    workspace: DeviceBuffer,
    output: DeviceBuffer,
    attention: Function,
    candidate: bool,
}
struct Rig {
    rows: u32,
    entry: sources::Entry,
    inputs: Inputs,
    arms: [Arm; 2],
    merge: Function,
    schedule: Schedule,
    _modules: [Module; 2],
}

impl Rig {
    fn new(s: &Session, entry: sources::Entry, seed: u32) -> LabResult<Self> {
        Self::scheduled(s, entry, seed, Schedule::Stage32)
    }

    fn scheduled(
        s: &Session,
        entry: sources::Entry,
        seed: u32,
        schedule: Schedule,
    ) -> LabResult<Self> {
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
        let candidate = s.module(schedule.module())?;
        let make_arm = |module: &Module, candidate: bool| -> LabResult<Arm> {
            let symbol = if candidate {
                schedule.symbol(&entry)
            } else {
                entry.symbol()
            };
            let attention = module.function(&symbol).map_err(|e| e.to_string())?;
            let plan = if candidate {
                schedule.plan(&entry, ROWS)
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
            rows: ROWS,
            arms: [make_arm(&reference, false)?, make_arm(&candidate, true)?],
            inputs: Inputs::new(s, page_size, second_fp4_keys, seed)?,
            merge: reference
                .function(&entry.merge_symbol())
                .map_err(|e| e.to_string())?,
            entry,
            schedule,
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
            self.schedule.plan(&self.entry, self.rows)
        } else {
            self.entry.plan(self.rows)
        }
        .map_err(|e| e.to_string())?;
        launch::launch_with(&selected.attention, stream, launch::config(plan)?, &args)
    }

    fn merge(&self, stream: &Stream, arm: usize) -> LabResult<()> {
        launch::launch_with(
            &self.merge,
            stream,
            launch::config(
                self.entry
                    .merge_plan(self.rows)
                    .map_err(|e| e.to_string())?,
            )?,
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
        let checked_rows = self.rows.saturating_add(1).min(ROWS);
        let poison = vec![0xa5; self.entry.workspace_bytes(checked_rows) as usize];
        for arm in &self.arms {
            arm.workspace
                .copy_from_bytes(&poison, &s.stream)
                .map_err(|e| e.to_string())?;
            arm.output
                .copy_from_bytes(
                    &poison[..checked_rows as usize * HEADS * WIDTH * 2],
                    &s.stream,
                )
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
            let bytes = (self.rows.saturating_add(1).min(ROWS) as usize * stride)
                .min(buffers[0].byte_len());
            let mut first = vec![0; bytes];
            let mut second = vec![0; bytes];
            buffers[0]
                .copy_to_bytes(&mut first, &s.stream)
                .map_err(|e| e.to_string())?;
            buffers[1]
                .copy_to_bytes(&mut second, &s.stream)
                .map_err(|e| e.to_string())?;
            compare_host_bytes(name, &first, &second, live as usize * stride)?;
        }
        Ok(())
    }
}

fn compare_host_bytes(name: &str, first: &[u8], second: &[u8], active: usize) -> LabResult<()> {
    if first.len() != second.len() || active > first.len() {
        return Err(format!("{name}: invalid comparison extent"));
    }
    // Slice equality uses the optimized byte comparison for the passing path.
    // Locate the first differing byte only when producing a failure diagnostic.
    if first != second {
        let at = first.iter().zip(second).position(|(a, b)| a != b).unwrap();
        return Err(format!(
            "{name} differs at byte {at}: old={} new={}",
            first[at], second[at]
        ));
    }
    const POISON: [u8; 4096] = [0xa5; 4096];
    if first[active..]
        .chunks(POISON.len())
        .any(|chunk| chunk != &POISON[..chunk.len()])
    {
        return Err(format!(
            "{name}: inactive bytes overwritten after byte {active}"
        ));
    }
    Ok(())
}

// Adapt only the lab launch geometry. The reference still validates all
// buffer roles and supplies the oracle's numerical contract and merge.
struct ScheduledOp {
    reference: Prepared<LatentSourcesBuffers<View>>,
    info: OpInfo,
    entry: sources::Entry,
    schedule: Schedule,
}
impl PreparedOp<LatentSourcesBuffers<View>> for ScheduledOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &LatentSourcesBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.reference.launches(buffers, workspace, counts)?;
        for launch in &mut launches {
            if launch.symbol == self.entry.symbol() {
                launch.symbol = self.schedule.symbol(&self.entry);
                launch.plan = self.schedule.plan(&self.entry, counts.rows)?;
            }
        }
        Ok(launches)
    }
}

fn scheduled_op(
    entry: &sources::Entry,
    schedule: Schedule,
    capacity: u32,
) -> LabResult<Prepared<LatentSourcesBuffers<View>>> {
    let backend = Sm120::new().with_latent_sources(entry.clone());
    let prepared = backend
        .latent_attention_sources(&entry.query, Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    let mut info = prepared.info().clone();
    for (symbol, shared) in &mut info.symbols {
        if *symbol == entry.symbol() {
            *symbol = schedule.symbol(entry);
            *shared = schedule
                .plan(entry, capacity)
                .map_err(|e| e.to_string())?
                .dynamic_shared_bytes;
        }
    }
    Ok(Box::new(ScheduledOp {
        reference: prepared,
        info,
        entry: entry.clone(),
        schedule,
    }))
}

fn oracle(s: &Session, entry: &sources::Entry, schedule: Schedule) -> LabResult<()> {
    let backend = Sm120::new().with_latent_sources(entry.clone());
    let prepared = backend
        .latent_attention_sources(&entry.query, Need::rows(4))
        .map_err(|e| e.to_string())?;
    let case = cases::latent_sources::numerical_case(entry.query, 4, prepared)?;
    let images = case.images();
    let reference = s.module(REFERENCE)?;
    let candidate = s.module(schedule.module())?;
    let scheduled = scheduled_op(entry, schedule, 4)?;
    let new_case = cases::latent_sources::numerical_case(entry.query, 4, scheduled)?;
    let old_lookup = |symbol: &str| reference.function(symbol).map_err(|e| e.to_string());
    let new_lookup = |symbol: &str| {
        if symbol == schedule.symbol(entry) {
            candidate.function(symbol).map_err(|e| e.to_string())
        } else {
            old_lookup(symbol)
        }
    };
    let old = case.launch(s, &old_lookup, &images, 4)?;
    let new = new_case.launch(s, &new_lookup, &images, 4)?;
    if old != new {
        return Err("small oracle case differs bitwise".into());
    }
    case.judge_outputs(&images, 4, &old)?;
    new_case.judge_outputs(&images, 4, &new)?;
    Ok(())
}

fn correctness(s: &Session, rank: usize, rig: &Rig) -> LabResult<()> {
    oracle(s, &rig.entry, rig.schedule)?;
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
    if rig.schedule != Schedule::Stage32 {
        for shifted in [false, true] {
            rig.inputs.set_case(s, 91, false, ROWS - 1)?;
            rig.inputs.share_selected_pairs(s, 91, shifted)?;
            rig.poison(s)?;
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
            rig.compare(s, ROWS - 1)?;
        }
    }
    println!(
        "PASS prefill_attention_bitwise rank={rank} variant={} rows={ROWS} workspace=true output=true replay=true f64=true candidate={} reuse_cases={}",
        variant(&rig.entry),
        rig.schedule.name(),
        rig.schedule != Schedule::Stage32
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
        let attention_graphs = [0, 1]
            .map(|arm| s.capture(|stream| rig.attention(stream, arm)))
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let combined = [0, 1]
            .map(|arm| s.capture(|stream| rig.combined(stream, arm)))
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let label = format!("prefill_attention.rank{rank}.layer{layer}");
        let measure = |graphs: &[runtime_cuda::GraphExec]| {
            kernel_harness::time_ab(s, 1, 3, [EvictionMode::Write; 2], |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })
        };
        let attention = kernel_harness::with_timing_label(&format!("{label}.attention"), || {
            measure(&attention_graphs)
        })?;
        let total =
            kernel_harness::with_timing_label(&format!("{label}.combined"), || measure(&combined))?;
        let merge_graphs = [0, 1]
            .map(|arm| s.capture(|stream| rig.merge(stream, arm)))
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let merge = kernel_harness::with_timing_label(&format!("{label}.merge"), || {
            kernel_harness::time_ab_prepared(
                s,
                1,
                3,
                [EvictionMode::Write; 2],
                |arm, stream| {
                    attention_graphs[arm]
                        .launch(stream)
                        .map_err(|e| e.to_string())
                },
                |arm, stream| merge_graphs[arm].launch(stream).map_err(|e| e.to_string()),
            )
        })?;
        let gflop = 4.0
            * f64::from(ROWS)
            * HEADS as f64
            * WIDTH as f64
            * f64::from(WINDOW + rig.inputs.compressed_budget)
            / 1e9;
        println!(
            "TIMING prefill_attention rank={rank} layer={layer} variant={} rows={ROWS} old_attention_us={} candidate_attention_us={} old_merge_us={} candidate_merge_us={} old_total_us={} candidate_total_us={} gflop={gflop} paired_ratio={} candidate={} warmup=1 samples=3",
            variant(&rig.entry),
            attention.arms[0].median_ms * 1000.0,
            attention.arms[1].median_ms * 1000.0,
            merge.arms[0].median_ms * 1000.0,
            merge.arms[1].median_ms * 1000.0,
            total.arms[0].median_ms * 1000.0,
            total.arms[1].median_ms * 1000.0,
            total.paired_ratio,
            rig.schedule.name()
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

pub fn run_reuse(sessions: &[Session]) -> LabResult<()> {
    exclusive(sessions)?;
    let entries = entries()?;
    for (rank, session) in sessions.iter().enumerate() {
        for schedule in [Schedule::Vector, Schedule::Pair] {
            for entry in &entries {
                let rig = Rig::scheduled(session, entry.clone(), rank as u32 + 1, schedule)?;
                correctness(session, rank, &rig)?;
            }
        }
    }
    for (rank, session) in sessions.iter().enumerate() {
        for schedule in [Schedule::Vector, Schedule::Pair] {
            for entry in &entries {
                let rig = Rig::scheduled(session, entry.clone(), rank as u32 + 1, schedule)?;
                timings(session, rank, &rig)?;
            }
        }
    }
    Ok(())
}

// The routing decision needs one case per distinct kernel, with the winning
// paired schedule. Full all-layer experiments remain available separately.
fn phase<T>(
    name: &str,
    rank: usize,
    entry: &sources::Entry,
    work: impl FnOnce() -> LabResult<T>,
) -> LabResult<T> {
    let started = std::time::Instant::now();
    let unix_ms = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    };
    println!(
        "PHASE prefill_decision phase={name} state=begin rank={rank} variant={} unix_ms={}",
        variant(entry),
        unix_ms()
    );
    let result = work();
    println!(
        "PHASE prefill_decision phase={name} state=end rank={rank} variant={} unix_ms={} elapsed_ms={} success={}",
        variant(entry),
        unix_ms(),
        started.elapsed().as_millis(),
        result.is_ok()
    );
    result
}

pub fn decision(sessions: &[Session], selected: &str, check: bool) -> LabResult<()> {
    exclusive(sessions)?;
    let entry = entries()?
        .into_iter()
        .find(|entry| variant(entry) == selected)
        .ok_or("decision variant must be window, mixed64 or mixed32")?;
    for (rank, session) in sessions.iter().enumerate() {
        let rig = phase("fixture", rank, &entry, || {
            Rig::scheduled(session, entry.clone(), rank as u32 + 1, Schedule::Pair)
        })?;
        if check {
            phase("bitwise", rank, &entry, || correctness(session, rank, &rig))?;
        } else {
            phase("combined", rank, &entry, || {
                // Keep the dense 8192-row fixture unchanged between A/B arms.
                let graphs = [0, 1]
                    .map(|arm| session.capture(|stream| rig.combined(stream, arm)))
                    .into_iter()
                    .collect::<LabResult<Vec<_>>>()?;
                let measured = kernel_harness::with_timing_label(
                    &format!("prefill_decision.rank{rank}.{selected}.combined"),
                    || {
                        kernel_harness::time_ab(
                            session,
                            1,
                            3,
                            [EvictionMode::Write; 2],
                            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                        )
                    },
                )?;
                println!(
                    "TIMING prefill_decision rank={rank} variant={selected} rows={ROWS} old_total_us={} candidate_total_us={} paired_ratio={} candidate=pair warmup=1 samples=3",
                    measured.arms[0].median_ms * 1000.0,
                    measured.arms[1].median_ms * 1000.0,
                    measured.paired_ratio
                );
                Ok(())
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_comparison_checks_every_live_and_inactive_byte() {
        let bytes = vec![0xa5; 8195];
        for active in [0, 1, 4095, 4096, 4097, bytes.len()] {
            compare_host_bytes("test", &bytes, &bytes, active).unwrap();
            for at in [0, 4095, 4096, 4097, bytes.len() - 1] {
                let mut other = bytes.clone();
                other[at] ^= 1;
                assert!(compare_host_bytes("test", &bytes, &other, active).is_err());
                assert_eq!(
                    compare_host_bytes("test", &other, &other, active).is_err(),
                    at >= active
                );
            }
        }
        assert!(compare_host_bytes("test", &bytes, &bytes[..8194], 0).is_err());
        assert!(compare_host_bytes("test", &bytes, &bytes, bytes.len() + 1).is_err());
    }

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
            for schedule in [Schedule::Vector, Schedule::Pair] {
                let prepared = scheduled_op(&entry, schedule, 4).unwrap();
                let symbol = schedule.symbol(&entry);
                assert!(prepared.info().symbols.iter().any(|(s, bytes)| s == &symbol
                    && *bytes == schedule.plan(&entry, 4).unwrap().dynamic_shared_bytes));
                let candidate_case =
                    cases::latent_sources::numerical_case(entry.query, 4, prepared).unwrap();
                assert_eq!(candidate_case.capacity(), 4);
                assert_eq!(
                    candidate_case.images().bytes("q").unwrap(),
                    case.images().bytes("q").unwrap()
                );
            }
        }
    }
}
