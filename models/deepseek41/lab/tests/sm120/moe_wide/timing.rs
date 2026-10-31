//! Paired event timings of the reference and candidate MoE programs with an
//! in-job A/A control: each round times the reference twice and the
//! candidate once, in rotating order, after flushing L2 with a 256 MiB
//! buffer (`Flush`) so no sample starts with the previous one's lines. The complete layer is
//! the overlapped fork/join graph; each stage group is its calls alone, in
//! serial graphs, after a complete layer wrote their inputs. Both arms run
//! the same weights, routing pattern and live rows, and one bitwise
//! comparison precedes every case's timing. Above 64 rows the pair must be
//! the 8192-row expert chains against the wide direct queues
//! (`recorded::check_wide_pair`).
use super::candidate::{Lab, moe_small_l2};
use super::device::{Pool, Programs};
use super::full::{compare, setup, status};
use super::graphs::Graphs;
use super::recorded::{Arm, Weights, check_wide_pair};
use kernel_harness::{LabResult, Session, launch};
use runtime_cuda::{GraphExec, KernelArgument};
use std::path::PathBuf;

pub struct Options {
    /// Also time each routed expert launch of each arm alone
    /// (`CHOPINFER_MOE_WIDE_LAUNCHES=1`).
    launches: bool,
    ranks: Vec<u32>,
    widths: Vec<u32>,
    scale_widths: Vec<u32>,
    patterns: Vec<u32>,
    rounds: usize,
    reference: PathBuf,
    candidate: PathBuf,
    weights: PathBuf,
    lab: Option<Lab>,
}

impl Options {
    pub fn environment() -> LabResult<Self> {
        let env = |name| std::env::var(name).map_err(|_| format!("missing {name}"));
        let list = |name| -> LabResult<Vec<u32>> {
            env(name)?
                .split(',')
                .map(|r| r.parse::<u32>().map_err(|e| e.to_string()))
                .collect()
        };
        let options = Self {
            launches: std::env::var("CHOPINFER_MOE_WIDE_LAUNCHES").as_deref() == Ok("1"),
            ranks: list("CHOPINFER_MOE_WIDE_RANKS")?,
            widths: list("CHOPINFER_MOE_WIDE_ROWS")?,
            scale_widths: list("CHOPINFER_MOE_WIDE_SCALE_WIDTHS")?,
            patterns: list("CHOPINFER_MOE_WIDE_PATTERNS")?,
            rounds: env("CHOPINFER_MOE_WIDE_ROUNDS")?
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?,
            reference: env("CHOPINFER_MOE_WIDE_REFERENCE")?.into(),
            candidate: env("CHOPINFER_MOE_WIDE_CANDIDATE")?.into(),
            weights: env("CHOPINFER_MOE_WIDE_WEIGHTS")?.into(),
            lab: Lab::environment()?,
        };
        let ordered = |v: &[u32]| !v.is_empty() && v.windows(2).all(|w| w[0] < w[1]);
        if !ordered(&options.ranks)
            || options.ranks.iter().any(|&r| r > 3)
            || !ordered(&options.widths)
            || options.widths.iter().any(|&r| !(1..=192).contains(&r))
            || !ordered(&options.scale_widths)
            || options
                .scale_widths
                .iter()
                .any(|w| ![0, 1, 2, 4].contains(w))
            || !ordered(&options.patterns)
            || options.patterns.iter().any(|&p| p >= super::full::PATTERNS)
            || !(5..=200).contains(&options.rounds)
        {
            return Err("invalid timing inventory".into());
        }
        Ok(options)
    }
}

/// The stage groups timed alone, by their lowered stage names.
const GROUPS: [(&str, &[&str]); 5] = [
    ("route", &["Route(0)"]),
    ("gate", &["ExpertsGateUp(0)"]),
    ("down", &["ExpertsDown(0)"]),
    ("sum", &["RoutedSum(0)"]),
    (
        "experts",
        &["ExpertsGateUp(0)", "ExpertsDown(0)", "RoutedSum(0)"],
    ),
];

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    }
}

fn quartiles(mut values: Vec<f64>) -> [f64; 2] {
    values.sort_by(f64::total_cmp);
    let at = |q: f64| values[((values.len() - 1) as f64 * q).round() as usize];
    [at(0.25), at(0.75)]
}

/// The L2 flush before every timed sample: a 256 MiB write, or with
/// `CHOPINFER_MOE_WIDE_FLUSH=read` the lab cubin's read of the same zeroed
/// buffer. The write leaves its own dirty lines, which the timed sample then
/// writes back; the read leaves only clean lines.
struct Flush {
    buffer: runtime_cuda::DeviceBuffer,
    bytes: usize,
    read: Option<(runtime_cuda::Function, runtime_cuda::DeviceBuffer)>,
}

impl Flush {
    fn new(session: &Session, lab: Option<&Lab>) -> LabResult<Self> {
        let bytes = 256 << 20;
        let buffer = session.allocate(bytes)?;
        buffer
            .zero_at_async(0, bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        let read = match std::env::var("CHOPINFER_MOE_WIDE_FLUSH").as_deref() {
            Err(_) | Ok("write") => None,
            Ok("read") => {
                let lab = lab.ok_or("the read flush needs CHOPINFER_MOE_WIDE_LAB_CUBIN")?;
                let function = session
                    .load_cubin(&lab.cubin)?
                    .function(moe_small_l2::READ_FLUSH)
                    .map_err(|e| e.to_string())?;
                Some((function, session.allocate(256)?))
            }
            Ok(other) => return Err(format!("unknown flush {other}")),
        };
        Ok(Self {
            buffer,
            bytes,
            read,
        })
    }

    fn name(&self) -> &'static str {
        if self.read.is_some() { "read" } else { "write" }
    }

    /// The read flush's own rate in bytes per second, when it reads: a
    /// buffer twice the L2, so nearly every line comes from DRAM.
    fn read_rate(&self, session: &Session) -> LabResult<Option<f64>> {
        if self.read.is_none() {
            return Ok(None);
        }
        let start = session.context.timed_event().map_err(|e| e.to_string())?;
        let end = session.context.timed_event().map_err(|e| e.to_string())?;
        let mut samples = Vec::new();
        for round in 0..12 {
            start.record(&session.stream).map_err(|e| e.to_string())?;
            self.issue(session)?;
            end.record(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            if round >= 2 {
                let ms = f64::from(start.elapsed_ms(&end).map_err(|e| e.to_string())?);
                samples.push(self.bytes as f64 / (ms * 1e-3));
            }
        }
        Ok(Some(median(samples)))
    }

    fn issue(&self, session: &Session) -> LabResult<()> {
        match &self.read {
            None => self
                .buffer
                .zero_at_async(0, self.bytes, &session.stream)
                .map_err(|e| e.to_string()),
            Some((function, sink)) => launch::launch_with(
                function,
                &session.stream,
                launch::simple_config((1024, 1, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
                &[
                    KernelArgument::Range {
                        buffer: &self.buffer,
                        offset: 0,
                        byte_len: self.bytes,
                    },
                    KernelArgument::U64((self.bytes / 16) as u64),
                    KernelArgument::Range {
                        buffer: sink,
                        offset: 0,
                        byte_len: 4,
                    },
                ],
            ),
        }
    }
}

/// Rounds of [reference, candidate, reference] in rotating order: per round
/// the candidate over the first reference sample (B/A) and the second
/// reference over the first (A/A).
fn paired(
    session: &Session,
    flush: &Flush,
    graphs: [&GraphExec; 2],
    rounds: usize,
) -> LabResult<serde_json::Value> {
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    for round in 0..rounds + 2 {
        for step in 0..3 {
            let slot = (round + step) % 3;
            let graph = graphs[usize::from(slot == 1)];
            flush.issue(session)?;
            start.record(&session.stream).map_err(|e| e.to_string())?;
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            end.record(&session.stream).map_err(|e| e.to_string())?;
            session.stream.synchronize().map_err(|e| e.to_string())?;
            if round >= 2 {
                samples[slot].push(f64::from(
                    start.elapsed_ms(&end).map_err(|e| e.to_string())?,
                ));
            }
        }
    }
    let ratio = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| b / a).collect::<Vec<_>>();
    let ba = ratio(&samples[0], &samples[1]);
    let aa = ratio(&samples[0], &samples[2]);
    Ok(serde_json::json!({
        "reference_ms": median(samples[0].clone()),
        "candidate_ms": median(samples[1].clone()),
        "reference_again_ms": median(samples[2].clone()),
        "candidate_over_reference": median(ba.clone()),
        "candidate_over_reference_iqr": quartiles(ba),
        "aa": median(aa.clone()),
        "aa_iqr": quartiles(aa),
        "rounds": rounds,
    }))
}

/// The median of `rounds` timings of `graph` alone, each after the L2
/// flush.
fn solo(session: &Session, flush: &Flush, graph: &GraphExec, rounds: usize) -> LabResult<f64> {
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    for round in 0..rounds + 2 {
        flush.issue(session)?;
        start.record(&session.stream).map_err(|e| e.to_string())?;
        graph.launch(&session.stream).map_err(|e| e.to_string())?;
        end.record(&session.stream).map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        if round >= 2 {
            samples.push(f64::from(
                start.elapsed_ms(&end).map_err(|e| e.to_string())?,
            ));
        }
    }
    Ok(median(samples))
}

/// The device copy rate of the flush buffer's halves, read plus write
/// bytes per second: a reference for the weight-read floor.
fn copy_rate(session: &Session, flush: &Flush) -> LabResult<f64> {
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    let half = flush.bytes / 2;
    let mut samples = Vec::new();
    for round in 0..12 {
        start.record(&session.stream).map_err(|e| e.to_string())?;
        flush
            .buffer
            .copy_within_async(0, half, half, &session.stream)
            .map_err(|e| e.to_string())?;
        end.record(&session.stream).map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        if round >= 2 {
            let ms = f64::from(start.elapsed_ms(&end).map_err(|e| e.to_string())?);
            samples.push(2. * half as f64 / (ms * 1e-3));
        }
    }
    Ok(median(samples))
}

pub fn run(session: &Session, options: Options) -> LabResult<()> {
    let flush = Flush::new(session, options.lab.as_ref())?;
    println!(
        "MOE_WIDE_COPY {}",
        serde_json::json!({
            "bytes_per_second": copy_rate(session, &flush)?,
            "flush": flush.name(),
            "read_bytes_per_second": flush.read_rate(session)?,
        })
    );
    let mut cases = 0;
    let mut wide_pairs = 0;
    let mut swapped = 0;
    for &rank in &options.ranks {
        let mut pool = Pool::new(Some(session));
        let weights = Weights::load(&mut pool, &options.weights, rank)?;
        let mark = pool.mark();
        let mut programs =
            Programs::with_lab(&pool, &[], options.lab.as_ref().map(|s| s.cubin.clone()))?;
        for &rows in &options.widths {
            for &width in &options.scale_widths {
                let mut arms = [
                    Arm::load(&mut pool, &weights, &options.reference, rows, rank, width)?,
                    Arm::load(&mut pool, &weights, &options.candidate, rows, rank, width)?,
                ];
                check_wide_pair(rows, &arms[0].expert_symbols()?, &arms[1].expert_symbols()?)?;
                wide_pairs += u32::from(rows > 64);
                let twins = options.lab.as_ref().map_or(0, |s| s.apply(&mut arms[1]));
                swapped += twins;
                programs.extend(
                    &pool,
                    &arms
                        .iter()
                        .flat_map(|a| a.prefix.iter().chain(&a.calls).cloned())
                        .collect::<Vec<_>>(),
                )?;
                let graphs = Graphs::build(&pool, &programs, &arms)?;
                let groups = GROUPS
                    .iter()
                    .map(|(name, stages)| {
                        let pair = arms
                            .iter()
                            .map(|arm| {
                                let calls = stages
                                    .iter()
                                    .map(|stage| arm.scope(stage).map(<[_]>::to_vec))
                                    .collect::<LabResult<Vec<_>>>()?
                                    .concat();
                                session.capture(|s| programs.issue(&pool, &calls, s))
                            })
                            .collect::<LabResult<Vec<_>>>()?;
                        Ok((*name, pair))
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                // Each routed expert launch of each arm alone, by stage.
                let mut launches = Vec::new();
                if options.launches {
                    for (index, arm) in arms.iter().enumerate() {
                        for stage in ["ExpertsGateUp(0)", "ExpertsDown(0)", "RoutedSum(0)"] {
                            for call in arm.scope(stage)? {
                                let graph = session.capture(|s| {
                                    programs.issue(&pool, std::slice::from_ref(call), s)
                                })?;
                                launches.push((index, stage, call.symbol.clone(), graph));
                            }
                        }
                    }
                }
                for &pattern in &options.patterns {
                    for arm in &arms {
                        setup(&pool, arm, pattern, rank, 0xa5)?;
                        pool.write(arm.role("live")?, &rows.to_le_bytes())?;
                        status(&pool, arm)?;
                    }
                    graphs.run(2)?;
                    graphs.run(3)?;
                    compare(&pool, &arms, rows, pattern, rank).map_err(|e| {
                        format!("rows={rows} rank={rank} width={width} pattern={pattern}: {e}")
                    })?;
                    let active = u32::from_le_bytes(
                        pool.read(arms[0].role("scratch.active_experts")?)?[..4]
                            .try_into()
                            .unwrap(),
                    );
                    let counts = pool
                        .read(arms[0].role("scratch.route_counts")?)?
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let widest = counts.iter().copied().max().unwrap_or(0);
                    let unique = counts.iter().filter(|&&c| c > 0).count();
                    let layer = paired(
                        session,
                        &flush,
                        [graphs.graph(2), graphs.graph(3)],
                        options.rounds,
                    )?;
                    let mut record = serde_json::json!({
                        "rank": rank, "rows": rows, "scale_width": width, "pattern": pattern,
                        "active_experts": active, "widest_expert_rows": widest,
                        "unique_experts": unique, "lab_twins": twins, "flush": flush.name(),
                        "layer": layer,
                    });
                    for (name, pair) in &groups {
                        record[*name] =
                            paired(session, &flush, [&pair[0], &pair[1]], options.rounds)?;
                    }
                    if options.launches {
                        record["launches"] = launches
                            .iter()
                            .map(|(arm, stage, symbol, graph)| {
                                Ok(serde_json::json!({
                                    "arm": arm, "stage": stage, "symbol": symbol,
                                    "ms": solo(session, &flush, graph, options.rounds)?,
                                }))
                            })
                            .collect::<LabResult<Vec<_>>>()?
                            .into();
                    }
                    // The timed graphs leave the outputs of a complete layer.
                    status(&pool, &arms[0])?;
                    status(&pool, &arms[1])?;
                    graphs.run(2)?;
                    graphs.run(3)?;
                    compare(&pool, &arms, rows, pattern, rank)?;
                    println!("MOE_WIDE_TIME {record}");
                    cases += 1;
                }
                for (_, pair) in groups {
                    for graph in pair {
                        graph.release().map_err(|e| e.to_string())?;
                    }
                }
                for (_, _, _, graph) in launches {
                    graph.release().map_err(|e| e.to_string())?;
                }
                graphs.close()?;
                pool.truncate(mark)?;
            }
        }
    }
    let expected = options.ranks.len()
        * options.widths.len()
        * options.scale_widths.len()
        * options.patterns.len();
    if cases != expected {
        return Err("timed inventory differs".into());
    }
    if let Some(tag) = options.lab.as_ref().and_then(|l| l.swap.as_ref())
        && swapped == 0
    {
        return Err(format!("knob set {tag} replaced no served launch"));
    }
    println!(
        "MOE_WIDE_TIME_COMPLETE cases={cases} wide_pairs_checked={wide_pairs} lab_twins={swapped} PASS"
    );
    Ok(())
}
