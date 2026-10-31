//! Paired event timings of the reference and candidate MoE programs with an
//! in-job A/A control: each round times the reference twice and the
//! candidate once, in rotating order, after writing a 256 MiB buffer so no
//! sample starts with the previous one's lines in L2. The complete layer is
//! the overlapped fork/join graph; each stage group is its calls alone, in
//! serial graphs, after a complete layer wrote their inputs. Both arms run
//! the same weights, routing pattern and live rows, and one bitwise
//! comparison precedes every case's timing. Above 64 rows the pair must be
//! the 8192-row expert chains against the wide direct queues
//! (`recorded::check_wide_pair`).
use super::device::{Pool, Programs};
use super::full::{compare, setup, status};
use super::graphs::Graphs;
use super::recorded::{Arm, Weights, check_wide_pair};
use kernel_harness::{LabResult, Session};
use runtime_cuda::GraphExec;
use std::path::PathBuf;

pub struct Options {
    ranks: Vec<u32>,
    widths: Vec<u32>,
    scale_widths: Vec<u32>,
    patterns: Vec<u32>,
    rounds: usize,
    reference: PathBuf,
    candidate: PathBuf,
    weights: PathBuf,
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
            || options.patterns.iter().any(|&p| p > 3)
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

/// Rounds of [reference, candidate, reference] in rotating order: per round
/// the candidate over the first reference sample (B/A) and the second
/// reference over the first (A/A).
fn paired(
    session: &Session,
    flush: &runtime_cuda::DeviceBuffer,
    flush_bytes: usize,
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
            flush
                .zero_at_async(0, flush_bytes, &session.stream)
                .map_err(|e| e.to_string())?;
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

pub fn run(session: &Session, options: Options) -> LabResult<()> {
    let flush_bytes = 256 << 20;
    let flush = session.allocate(flush_bytes)?;
    let mut cases = 0;
    let mut wide_pairs = 0;
    for &rank in &options.ranks {
        let mut pool = Pool::new(Some(session));
        let weights = Weights::load(&mut pool, &options.weights, rank)?;
        let mark = pool.mark();
        let mut programs = Programs::load(&pool, &[])?;
        for &rows in &options.widths {
            for &width in &options.scale_widths {
                let arms = [
                    Arm::load(&mut pool, &weights, &options.reference, rows, rank, width)?,
                    Arm::load(&mut pool, &weights, &options.candidate, rows, rank, width)?,
                ];
                check_wide_pair(rows, &arms[0].expert_symbols()?, &arms[1].expert_symbols()?)?;
                wide_pairs += u32::from(rows > 64);
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
                    let layer = paired(
                        session,
                        &flush,
                        flush_bytes,
                        [graphs.graph(2), graphs.graph(3)],
                        options.rounds,
                    )?;
                    let mut record = serde_json::json!({
                        "rank": rank, "rows": rows, "scale_width": width, "pattern": pattern,
                        "active_experts": active, "widest_expert_rows": widest,
                        "layer": layer,
                    });
                    for (name, pair) in &groups {
                        record[*name] = paired(
                            session,
                            &flush,
                            flush_bytes,
                            [&pair[0], &pair[1]],
                            options.rounds,
                        )?;
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
    println!("MOE_WIDE_TIME_COMPLETE cases={cases} wide_pairs_checked={wide_pairs} PASS");
    Ok(())
}
