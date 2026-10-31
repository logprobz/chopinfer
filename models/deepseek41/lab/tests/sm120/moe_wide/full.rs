//! Complete routed and shared MoE pairs over all planned wide row counts.
use super::device::{Pool, Programs};
use super::graphs::{self, Graphs};
use super::recorded::{Arm, Weights};
use kernel_core::codec::f32_to_bf16;
use kernel_harness::{LabResult, Session};
use std::collections::BTreeMap;
use std::path::PathBuf;

pub struct Options {
    worker: u32,
    checked: bool,
    compact: bool,
    widths: Vec<u32>,
    ranks: Vec<u32>,
    patterns: Vec<u32>,
    reference: PathBuf,
    candidate: PathBuf,
    weights: PathBuf,
}
impl Options {
    pub fn environment() -> LabResult<Self> {
        let env = |name| std::env::var(name).map_err(|_| format!("missing {name}"));
        let worker = env("CHOPINFER_MOE_WIDE_WORKER")?
            .parse::<u32>()
            .map_err(|e| e.to_string())?;
        let list = |text: String| {
            text.split(',')
                .map(|r| r.parse::<u32>().map_err(|e| e.to_string()))
                .collect::<LabResult<Vec<_>>>()
        };
        let widths = list(env("CHOPINFER_MOE_WIDE_ROWS")?)?;
        let checked = env("CHOPINFER_MOE_WIDE_MEMCHECK")? == "1";
        // CHOPINFER_MOE_WIDE_RANKS optionally names the logical ranks; by
        // default a worker takes ranks `worker` and `worker + 2`, or only
        // `worker` under memcheck.
        let ranks = match std::env::var("CHOPINFER_MOE_WIDE_RANKS") {
            Ok(text) => list(text)?,
            Err(_) if checked => vec![worker],
            Err(_) => vec![worker, worker + 2],
        };
        // Routing patterns (`expert`), all four by default.
        let patterns = match std::env::var("CHOPINFER_MOE_WIDE_PATTERNS") {
            Ok(text) => list(text)?,
            Err(_) => (0..PATTERNS).collect(),
        };
        let ordered = |v: &[u32]| !v.is_empty() && v.windows(2).all(|w| w[0] < w[1]);
        if worker > 3
            || !ordered(&widths)
            || widths
                .iter()
                .any(|&r| r != 16 && r != 32 && !(33..=192).contains(&r))
            || !ordered(&ranks)
            || ranks.iter().any(|&r| r > 3)
            || !ordered(&patterns)
            || patterns.iter().any(|&p| p >= PATTERNS)
        {
            return Err("invalid worker, rank, pattern or ordered row inventory".into());
        }
        Ok(Self {
            worker,
            compact: std::env::var("CHOPINFER_MOE_WIDE_COMPACT").as_deref() == Ok("1"),
            checked,
            widths,
            ranks,
            patterns,
            reference: env("CHOPINFER_MOE_WIDE_REFERENCE")?.into(),
            candidate: env("CHOPINFER_MOE_WIDE_CANDIDATE")?.into(),
            weights: env("CHOPINFER_MOE_WIDE_WEIGHTS")?.into(),
        })
    }
}

/// The routing patterns: 0 sends every row to the same six experts (each
/// holds every row, above 64 for a wide step); 1 rotates rows over 24
/// experts; 2 disperses routes over every expert; 3 is seeded and mixed:
/// one expert takes every row, two take alternate rows and the other four
/// routes of a row spread over the remaining 381 experts, so one step
/// fills the wide, middle and small queues.
const PATTERNS: u32 = 4;

pub(super) fn expert(row: u32, k: u32, pattern: u32, rank: u32) -> u32 {
    let slot = match pattern {
        0 => k,
        1 => (6 * row + k) % 24,
        2 => 6 * row + k,
        3 => match k {
            0 => 0,
            1 => 1 + row % 2,
            // 97 is prime to 381, so a row's four slots are distinct.
            _ => {
                let seed = (row ^ (rank << 16))
                    .wrapping_mul(0x9e37_79b1)
                    .rotate_left(13)
                    .wrapping_mul(0x85eb_ca6b);
                3 + (seed % 381 + 97 * (k - 2)) % 381
            }
        },
        _ => unreachable!(),
    };
    (37 * slot + 11 + 17 * rank) % 384
}
pub(super) fn setup(
    pool: &Pool<'_>,
    arm: &Arm,
    pattern: u32,
    rank: u32,
    poison: u8,
) -> LabResult<()> {
    for &view in &arm.mutable {
        pool.fill(view, poison)?;
    }
    let mut narrow = Vec::new();
    let mut wide = Vec::new();
    for row in 0..arm.rows {
        for col in 0..5120 {
            let mut value = if col < 384 {
                -16.
            } else {
                ((col + 3 * row + 7 * rank) % 11) as f32 / 16. - 0.25
            };
            for k in 0..6 {
                if col == expert(row, k, pattern, rank) {
                    value = 8. - k as f32 * 0.5;
                }
            }
            let bf16 = f32_to_bf16(value);
            narrow.extend(bf16.to_le_bytes());
            wide.extend((u32::from(bf16) << 16).to_le_bytes());
        }
    }
    pool.write(arm.role("scratch.normed")?, &narrow)?;
    pool.write(arm.role("scratch.normed_wide")?, &wide)?;
    pool.write(
        arm.role("scratch.bias_select")?,
        &(0..arm.rows).map(|r| (r % 2) as u8).collect::<Vec<_>>(),
    )?;
    status(pool, arm)
}
pub(super) fn status(pool: &Pool<'_>, arm: &Arm) -> LabResult<()> {
    for name in ["scratch.row_status", "scratch.shared_row_status"] {
        pool.fill(arm.role(name)?, 0)?;
    }
    Ok(())
}

const SEMANTIC: &[&str] = &[
    "scratch.normed_codes",
    "scratch.normed_scales",
    "scratch.router_logits",
    "scratch.router_scores",
    "scratch.expert_ids",
    "scratch.expert_weights",
    "scratch.route_counts",
    "scratch.route_offsets",
    "scratch.padded_rows",
    "scratch.active_experts",
    "sorted_rows",
    "scratch.expert_intermediate",
    "scratch.intermediate_codes",
    "scratch.intermediate_scales",
    "scratch.routed",
    "scratch.shared_gate",
    "scratch.shared_up",
    "scratch.shared_codes",
    "scratch.shared_scales",
    "scratch.shared_partial",
    "scratch.partial",
    "scratch.row_status",
    "scratch.shared_row_status",
];

fn same(label: &str, a: &[u8], b: &[u8]) -> LabResult<()> {
    if a != b {
        let at = a.iter().zip(b).position(|(a, b)| a != b);
        return Err(format!(
            "{label}: bitwise mismatch at byte {at:?}, lengths {}/{}",
            a.len(),
            b.len()
        ));
    }
    Ok(())
}
fn integers(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn metadata(pool: &Pool<'_>, arm: &Arm, live: u32, pattern: u32, rank: u32) -> LabResult<()> {
    let mut counts = vec![0_u32; 384];
    let mut ids = Vec::new();
    for row in 0..live {
        for k in 0..6 {
            let e = expert(row, k, pattern, rank);
            ids.push(e);
            counts[e as usize] += 1;
        }
    }
    let actual = integers(&pool.read(arm.role("scratch.expert_ids")?)?);
    if actual[..ids.len()] != ids {
        return Err("router did not produce the requested expert union".into());
    }
    if integers(&pool.read(arm.role("scratch.route_counts")?)?) != counts {
        return Err("route counts differ from the CPU union".into());
    }
    // Each active expert's range is padded to whole 64-row tiles: one tile
    // through 64 rows, several above.
    let mut offsets = vec![0_u32; 385];
    let mut sorted = Vec::new();
    let mut active = vec![0];
    for e in 0..384 {
        offsets[e + 1] = offsets[e];
        if counts[e] == 0 {
            continue;
        }
        active.push(e as u32);
        let padded = counts[e].div_ceil(64) * 64;
        offsets[e + 1] += padded;
        let start = sorted.len();
        sorted.extend(
            ids.iter()
                .enumerate()
                .filter_map(|(r, &id)| (id == e as u32).then_some(r as u32)),
        );
        sorted.resize(start + padded as usize, u32::MAX);
    }
    active[0] = active.len() as u32 - 1;
    if integers(&pool.read(arm.role("scratch.route_offsets")?)?) != offsets
        || integers(&pool.read(arm.role("scratch.padded_rows")?)?) != [sorted.len() as u32]
    {
        return Err("route offsets or padded count differ".into());
    }
    if integers(&pool.read(arm.role("sorted_rows")?)?)[..sorted.len()] != sorted
        || integers(&pool.read(arm.role("scratch.active_experts")?)?)[..active.len()] != active
    {
        return Err("ordered routes or active expert list differ".into());
    }
    Ok(())
}
pub(super) fn compare(
    pool: &Pool<'_>,
    arms: &[Arm; 2],
    live: u32,
    pattern: u32,
    rank: u32,
) -> LabResult<Vec<u8>> {
    let mut output = Vec::new();
    for &name in SEMANTIC {
        let a = pool.read(arms[0].role(name)?)?;
        let b = pool.read(arms[1].role(name)?)?;
        same(name, &a, &b)?;
        if name == "scratch.partial" {
            output.extend(&b[..live as usize * 5120 * 4]);
            if output
                .chunks_exact(4)
                .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
            {
                return Err("non-finite live MoE output".into());
            }
        }
        if name.ends_with("row_status") && a.iter().any(|&v| v != 0) {
            return Err(format!("{name} rejected the finite fixture"));
        }
    }
    for arm in arms {
        metadata(pool, arm, live, pattern, rank)?;
    }
    pool.guards()?;
    Ok(output)
}

pub fn run(session: Option<&Session>, options: Options) -> LabResult<()> {
    let dry = session.is_none();
    let mut cases = 0;
    let mut replays = 0;
    let mut negatives = 0;
    let mut destroyed = 0;
    let mut peak = 0;
    for &rank in &options.ranks {
        let mut pool = Pool::new(session);
        let weights = Weights::load(&mut pool, &options.weights, rank)?;
        let mark = pool.mark();
        let mut programs = if dry {
            None
        } else {
            Some(Programs::load(&pool, &[])?)
        };
        let mut raw_outputs = BTreeMap::new();
        for &rows in &options.widths {
            for width in [0, 1, 2, 4] {
                let arms = [
                    Arm::load(&mut pool, &weights, &options.reference, rows, rank, width)?,
                    Arm::load(&mut pool, &weights, &options.candidate, rows, rank, width)?,
                ];
                for name in SEMANTIC {
                    if arms[0].role(name)?.bytes != arms[1].role(name)?.bytes {
                        return Err(format!("semantic extent differs: {name}"));
                    }
                }
                peak = peak.max(pool.allocated_bytes());
                if let Some(programs) = programs.as_mut() {
                    programs.extend(
                        &pool,
                        &arms
                            .iter()
                            .flat_map(|a| a.prefix.iter().chain(&a.calls).cloned())
                            .collect::<Vec<_>>(),
                    )?;
                }
                let graphs = if !dry && !options.checked {
                    Some(Graphs::build(&pool, programs.as_ref().unwrap(), &arms)?)
                } else {
                    None
                };
                for &pattern in &options.patterns {
                    let poisons: &[u8] = if options.checked || options.compact {
                        &[0xa5]
                    } else {
                        &[0xa5, 0x5a]
                    };
                    let lives = if options.checked || options.compact {
                        vec![rows, rows / 2, 0, rows]
                    } else {
                        vec![rows, rows / 2, 1, 0, rows]
                    };
                    let modes = if options.checked { 1 } else { 3 };
                    for (poison_index, &poison) in poisons.iter().enumerate() {
                        for mode in 0..modes {
                            if !dry {
                                for arm in &arms {
                                    setup(&pool, arm, pattern, rank, poison)?;
                                }
                            }
                            for (live_index, &live) in lives.iter().enumerate() {
                                if !dry {
                                    for (index, arm) in arms.iter().enumerate() {
                                        pool.write(arm.role("live")?, &live.to_le_bytes())?;
                                        status(&pool, arm)?;
                                        if mode == 0 {
                                            graphs::serial(
                                                &pool,
                                                programs.as_ref().unwrap(),
                                                arm,
                                                &session.unwrap().stream,
                                            )?;
                                        } else {
                                            graphs.as_ref().unwrap().run((mode - 1) * 2 + index)?;
                                        }
                                    }
                                    pool.sync()?;
                                    let output = compare(&pool, &arms, live, pattern, rank)
                                        .map_err(|e| format!("rows={rows} rank={rank} width={width} pattern={pattern} live={live} mode={mode} poison={poison}: {e}"))?;
                                    if mode == 0
                                        && poison_index == 0
                                        && live_index < lives.len() - 1
                                    {
                                        if width == 0 {
                                            raw_outputs.insert((rows, pattern, live), output);
                                        } else {
                                            same(
                                                "encoded/raw complete output",
                                                &raw_outputs[&(rows, pattern, live)],
                                                &output,
                                            )?;
                                        }
                                    }
                                }
                                replays += 1;
                            }
                        }
                    }
                    if !options.checked {
                        if !dry {
                            let candidate = &arms[1];
                            setup(&pool, candidate, pattern, rank, 0x5a)?;
                            pool.write(candidate.role("live")?, &rows.to_le_bytes())?;
                            graphs.as_ref().unwrap().run(4)?;
                            let bad = pool.read(candidate.role("scratch.partial")?)?;
                            let expected = &raw_outputs[&(rows, pattern, rows)];
                            if bad == *expected {
                                return Err(
                                    "missing shared join control did not alter the output".into()
                                );
                            }
                            setup(&pool, candidate, pattern, rank, 0x5a)?;
                            pool.write(candidate.role("live")?, &rows.to_le_bytes())?;
                            graphs.as_ref().unwrap().run(3)?;
                            same(
                                "restored shared join",
                                expected,
                                &pool.read(candidate.role("scratch.partial")?)?,
                            )?;
                            pool.guards()?;
                        }
                        negatives += 1;
                    }
                    cases += 1;
                    println!(
                        "MOE_FULL_{} rows={rows} rank={rank} width={width} pattern={pattern} replays={} negative={} PASS",
                        if dry { "DRY" } else { "PAIR" },
                        poisons.len() * modes * lives.len(),
                        usize::from(!options.checked)
                    );
                }
                if let Some(graphs) = graphs {
                    let count = graphs.close()?;
                    if count != 5 {
                        return Err("incomplete graph teardown".into());
                    }
                    destroyed += count;
                }
                pool.truncate(mark)?;
            }
        }
    }
    let expected = options.widths.len() * options.ranks.len() * 4 * options.patterns.len();
    if cases != expected
        || replays
            != cases
                * if options.checked {
                    4
                } else if options.compact {
                    12
                } else {
                    30
                }
        || negatives != if options.checked { 0 } else { cases }
        || (!dry && !options.checked && destroyed != cases / options.patterns.len() * 5)
    {
        return Err("full MoE executed inventory differs".into());
    }
    println!(
        "MOE_FULL_COMPLETE worker={} checked={} compact={} dry={dry} cases={cases} replays={replays} negatives={negatives} graphs_destroyed={destroyed} allocated_peak={peak} PASS",
        options.worker, options.checked, options.compact
    );
    Ok(())
}
