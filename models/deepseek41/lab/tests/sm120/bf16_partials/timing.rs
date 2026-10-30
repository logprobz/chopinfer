//! Decode and verify's served reduction against the BF16-partial transport
//! prefill serves, timed with CUDA events in graph replays on four ranks
//! under the pair-lab rules. Opt-in (`CHOPINFER_KERNEL_BENCH=1`): the
//! timings locate work and never decide adoption.
//!
//! Decode and verify reduce through the rounded all-reduce on the bundle's
//! registration (`reduce::register`, as `Kernels::new_for_kv` registers
//! it), whose composed form is the F32 reduce-scatter, all-gather and BF16
//! round. Where the next sublayer's HC mix takes the round
//! (`ReduceOps::defer_round`), a step issues only the pair; elsewhere it
//! issues all three ([`served`]). Each arm all-reduces rows of 5120 F32
//! partials in rank order:
//!
//! - `f32_deferred`: the scatter and all-gather, a served step's whole
//!   reduction where the HC mix takes the round.
//! - `f32_complete`: the scatter, all-gather and round, a served step's
//!   reduction where no HC mix takes it.
//! - `f32_control`: a second graph of the row count's primary served form
//!   on the same operands, the in-job A/A control.
//! - `bf16_partials`, the candidate: the phased BF16-partial all-reduce
//!   prefill serves.
//! - `f32_phased`, a same-transport precision reference: the phased F32
//!   rounding entries, which no decode or verify step serves
//!   (`Kernels::prefill_reduction_backend`).
//! - `f32_round`: the round alone, for reference.
//!
//! Each row count compares `bf16_partials` with the form each served step
//! of those rows issues; the primary comparison is decode's where decode
//! has the row count as a graph bucket, else verify's.
use super::{
    Fixture, Op, Plan, WIDTH, err, every, inputs, manifest, op, reduce, regular, synthetic, views,
};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    collective::AllReduceBuffers,
    contract::CommQuery,
};
use kernel_harness::{Cubins, LabResult, Session};
use kernels_sm120::{backend::Sm120, row::rounding::Rounding};
use model_deepseek41_contract::{
    geometry::FLASH,
    serving::{DECODE_BATCH, DECODE_BUCKETS, MAX_VERIFY_ROWS, VERIFY_DECODE_ROWS},
    tuning::{self, Family},
};
use runtime_cuda::GraphExec;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
    time::Instant,
};

const ROWS: [u32; 6] = [1, 8, 16, 32, 64, 128];
/// The rows every fixture holds: the most timed rows.
const ROW_CAPACITY: u32 = 128;
/// The most rows of a frozen capture; larger row counts are synthetic.
const CAPTURED_ROWS: u32 = 40;
const CALLS: usize = 32;
const WARMUP: usize = 3;
const REPLAYS: usize = 25;
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_gate_release_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

/// The F32 calls a served step issues at a sublayer boundary that has a
/// next sublayer (`successor` in `bundle/src/lower/hc_expand_mix.rs`;
/// before an Engram layer and the head every step issues the complete
/// form).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Form {
    /// The scatter and all-gather; the next HC mix rounds.
    Deferred,
    /// The scatter, all-gather and round.
    Complete,
}
impl Form {
    fn arm(self) -> Arm {
        match self {
            Self::Deferred => Arm::Deferred,
            Self::Complete => Arm::Complete,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Step {
    Decode,
    Verify,
}
impl Step {
    fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Verify => "verify",
        }
    }
}

/// The form a served `step` of `rows` rows issues, or `None` when no such
/// step has `rows` rows. This mirrors the bundle's crate-private admission
/// on native Flash TP4 (`Lowering::round_expand_with_next_mix`):
///
/// - Decode runs at its graph buckets (`DECODE_BUCKETS`), and
///   `Lowering::has_expand_mix` admits every decode phase once
///   `Kernels::new_for_kv` registers the fused HC entries, so decode
///   defers the round.
/// - Verify graphs keep their exact rows, 1 to `DECODE_BATCH *
///   MAX_VERIFY_ROWS`. Verify defers only where `HcOps::has_finish_norm`
///   admits the phase: the Flash HC table's `hc.finish_admit.verify` row
///   at the finish norm's capacity, `Shape::verify_decode_rows`, the
///   instance's verify rows (at least `DECODE_BATCH`) capped at
///   `VERIFY_DECODE_ROWS`. Every instance that can serve `rows` rows must
///   agree, so the form never depends on the instance.
fn served(step: Step, rows: u32) -> LabResult<Option<Form>> {
    match step {
        Step::Decode => Ok(DECODE_BUCKETS.contains(&rows).then_some(Form::Deferred)),
        Step::Verify => {
            let most = DECODE_BATCH * MAX_VERIFY_ROWS;
            if rows == 0 || rows > most {
                return Ok(None);
            }
            let shape = [
                ("groups", FLASH.shape.hc.streams),
                ("width", FLASH.shape.hidden),
            ];
            let admitted = |verify_rows: u32| {
                let capacity = verify_rows.clamp(DECODE_BATCH, VERIFY_DECODE_ROWS);
                tuning::select(
                    &FLASH,
                    Family::Hc,
                    "hc.finish_admit.verify",
                    "f32",
                    &shape,
                    capacity,
                    rows,
                )
                .is_some()
            };
            let deferred = admitted(rows);
            if (rows..=most).any(|verify_rows| admitted(verify_rows) != deferred) {
                return Err(format!(
                    "verify's HC fusion at {rows} rows depends on the instance's verify rows"
                ));
            }
            Ok(Some(if deferred {
                Form::Deferred
            } else {
                Form::Complete
            }))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Arm {
    Deferred,
    Complete,
    Control,
    Bf16,
    Phased,
    Round,
}
/// One cycle's visits; the second cycle visits them in reverse, so each
/// pair of arms runs A1 B1 B2 A2.
const ORDER: [Arm; 6] = [
    Arm::Deferred,
    Arm::Complete,
    Arm::Control,
    Arm::Bf16,
    Arm::Phased,
    Arm::Round,
];
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Deferred => "f32_deferred",
            Self::Complete => "f32_complete",
            Self::Control => "f32_control",
            Self::Bf16 => "bf16_partials",
            Self::Phased => "f32_phased",
            Self::Round => "f32_round",
        }
    }
    /// What the arm times, as its receipt states it.
    fn role(self) -> &'static str {
        match self {
            Self::Deferred => {
                "served F32 form where the next HC mix takes the round: \
                 the reduce-scatter and all-gather"
            }
            Self::Complete => {
                "served F32 form where no HC mix takes the round: \
                 the reduce-scatter, all-gather and round"
            }
            Self::Control => "A/A control: a second graph of the row count's primary served form",
            Self::Bf16 => "candidate: the phased BF16-partial all-reduce prefill serves",
            Self::Phased => {
                "same-transport precision reference: the phased F32 rounding entries, \
                 which no decode or verify step serves"
            }
            Self::Round => "the composed form's round alone, for reference",
        }
    }
    /// The arm's calls at a row count whose primary served form is
    /// `primary`.
    fn plan(self, primary: Form) -> Plan {
        match self {
            Self::Deferred => deferred,
            Self::Control => primary.arm().plan(primary),
            Self::Round => round,
            Self::Complete | Self::Bf16 | Self::Phased => every,
        }
    }
}

/// The composed form's pair and its round.
fn served_symbols() -> Vec<String> {
    vec![
        reduce::SCATTER_SYMBOL.into(),
        reduce::GATHER_SYMBOL.into(),
        reduce::rounding_symbol(&FLASH, Rounding::Round),
    ]
}
/// Rank `rank`'s decode and verify reduction as the bundle registers it:
/// the rounded all-reduce over the rounding rows alone, without the phased
/// F32 entries, so every row count takes the composed form.
fn served_op(rank: usize, capacity: u32) -> LabResult<Op> {
    reduce::register(Sm120::new(), &FLASH)
        .all_reduce(
            &CommQuery::AllReduceRounded {
                ranks: 4,
                elements: WIDTH as u64,
            },
            rank as u32,
            Need::rows(capacity),
        )
        .map_err(err)
}
/// The calls of a step whose HC mix takes the round: every call before
/// the round, checked as `ReduceOps::defer_round` checks them.
fn deferred(op: &Op, buffers: &AllReduceBuffers<View>, counts: Counts) -> LabResult<Vec<Launch>> {
    let round = op
        .defer_bf16_round(buffers, View::EMPTY, counts)
        .map_err(err)?
        .ok_or("the served reduction no longer defers its round")?;
    if round.width as usize != WIDTH
        || round.output != buffers.output
        || Some(round.live_rows) != buffers.live_rows
    {
        return Err("the deferred round differs from the reduction output".into());
    }
    Ok(round.launches)
}
/// The round alone: the step's calls after the deferred ones.
fn round(op: &Op, buffers: &AllReduceBuffers<View>, counts: Counts) -> LabResult<Vec<Launch>> {
    let mut calls = every(op, buffers, counts)?;
    let prefix = deferred(op, buffers, counts)?;
    if calls.len() != prefix.len() + 1 || calls[..prefix.len()] != prefix[..] {
        return Err("the deferred calls are not a prefix of the step".into());
    }
    Ok(calls.split_off(prefix.len()))
}
/// `plan`'s calls of `op` over views of the fixture's sizes, without a GPU.
fn planned(op: &Op, rows: u32, plan: Plan) -> LabResult<Vec<Launch>> {
    let state = &op.info().state;
    let row_bytes = |bytes| u64::from(ROW_CAPACITY) * WIDTH as u64 * bytes;
    let buffers = views(state.len(), |id| match id {
        1 => row_bytes(4),
        2 => row_bytes(2),
        3 => 4,
        100..1000 => row_bytes(2),
        _ => state[((id - 1000) % 100) as usize],
    });
    plan(op, &buffers, Counts::rows(rows))
}

/// The timing probe cubin that CHOPINFER_BF16_PARTIAL_PROBE names (for
/// example as `kernel-lab stage-cubins` writes it), pinned by kernel-lab's
/// lock.
fn probe() -> LabResult<Cubins> {
    let path = std::env::var_os("CHOPINFER_BF16_PARTIAL_PROBE")
        .ok_or("CHOPINFER_BF16_PARTIAL_PROBE must name the probe_sm120 cubin")?;
    let bytes = regular(Path::new(&path))?;
    let lock: Value = serde_json::from_str(include_str!(
        "../../../../../../crates/kernel-lab/kernels.lock"
    ))
    .map_err(err)?;
    let pinned = lock
        .as_array()
        .and_then(|records| records.iter().find(|r| r["name"] == "probe_sm120"))
        .ok_or("kernel-lab pins no timing probe")?;
    if pinned["cubin_sha256"] != kernel_build::digest(&bytes) {
        return Err("the timing probe differs from kernel-lab's lock".into());
    }
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    Ok(Cubins::new(
        Box::leak(Box::new([("probe_sm120", bytes)])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    ))
}

/// The checkout this test was built from.
fn tree() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}
/// The directory receipts land in when the harness writes under `root`
/// from `cwd`, refused inside `tree`: the longest existing prefix
/// resolved through its symlinks, then the rest, which may not climb.
fn external(root: &Path, cwd: &Path, tree: &Path) -> LabResult<PathBuf> {
    let absolute = cwd.join(root);
    let (base, rest) = absolute
        .ancestors()
        .find_map(|prefix| {
            let base = prefix.canonicalize().ok()?;
            Some((base, absolute.strip_prefix(prefix).ok()?.to_path_buf()))
        })
        .ok_or_else(|| format!("cannot resolve {}", absolute.display()))?;
    if rest
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!(
            "receipt directory {} climbs below a missing directory",
            absolute.display()
        ));
    }
    let resolved = base.join(rest);
    if let Ok(tree) = tree.canonicalize()
        && resolved.starts_with(&tree)
    {
        return Err(format!(
            "receipts would land inside the source tree at {}; set \
             CHOPINFER_KERNEL_RECEIPTS to a directory outside it",
            resolved.display()
        ));
    }
    Ok(resolved)
}

/// Each rank's rows: a frozen capture's first rows when `cases` has one of
/// at least `rows` rows, else the synthetic pattern, with their source.
fn timed_inputs(cases: Option<&[Value]>, rows: u32) -> LabResult<(Vec<Vec<f32>>, Value)> {
    let captured = cases.filter(|_| rows <= CAPTURED_ROWS).and_then(|cases| {
        cases
            .iter()
            .find(|case| case["rows"].as_u64().is_some_and(|n| n >= u64::from(rows)))
    });
    let Some(case) = captured else {
        return Ok((synthetic(rows as usize), json!({"source": "synthetic"})));
    };
    let values = inputs(case)?
        .into_iter()
        .map(|mut rank| {
            rank.truncate(rows as usize * WIDTH);
            rank
        })
        .collect();
    Ok((
        values,
        json!({"source": "captured", "file": case["file"], "capture_rows": case["rows"]}),
    ))
}

/// A positive finite nvidia-smi reading, or why the field has none.
fn reading(field: &str, what: &str, rank: usize) -> LabResult<f64> {
    field
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| format!("rank {rank} reports no {what} ({field:?})"))
}
/// Every rank's enforced power limit in watts and SM clock in MHz, from
/// `uuid,enforced.power.limit,clocks.sm` CSV lines, refusing a rank
/// without both.
fn parse_readings(text: &str, uuids: &[String]) -> LabResult<Vec<[f64; 2]>> {
    let rows: BTreeMap<&str, Vec<&str>> = text
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            (fields.len() == 3).then(|| (fields[0], fields[1..].to_vec()))
        })
        .collect();
    uuids
        .iter()
        .enumerate()
        .map(|(rank, uuid)| {
            let fields = rows
                .get(uuid.as_str())
                .ok_or_else(|| format!("nvidia-smi does not report rank {rank}"))?;
            Ok([
                reading(fields[0], "enforced power limit", rank)?,
                reading(fields[1], "SM clock", rank)?,
            ])
        })
        .collect()
}
/// [`parse_readings`] of one nvidia-smi query of every GPU. Recorded only:
/// the operator owns limits and clocks.
fn readings(uuids: &[String]) -> LabResult<Vec<[f64; 2]>> {
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=uuid,enforced.power.limit,clocks.sm")
        .arg("--format=csv,noheader,nounits")
        .output()
        .map_err(err)?;
    if !output.status.success() {
        return Err("nvidia-smi cannot report power limits and clocks".into());
    }
    parse_readings(&String::from_utf8(output.stdout).map_err(err)?, uuids)
}
/// The job's power samples: every rank's enforced limit and SM clock at
/// the start, before every visit and at the end, each limit compared with
/// the rank's first.
#[derive(Default)]
struct Power {
    uuids: Vec<String>,
    initial: Vec<f64>,
    samples: Vec<Value>,
    changes: Vec<String>,
}
impl Power {
    fn start(sessions: &[Session]) -> LabResult<Self> {
        let uuids = sessions
            .iter()
            .map(|s| {
                s.metadata["uuid"]
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "missing device UUID".to_owned())
            })
            .collect::<LabResult<Vec<_>>>()?;
        let mut power = Self {
            uuids,
            ..Self::default()
        };
        power.sample("start")?;
        Ok(power)
    }
    fn sample(&mut self, label: &str) -> LabResult<Value> {
        let ranks = readings(&self.uuids)?;
        Ok(self.record(label, &ranks))
    }
    /// Records one sample of every rank, noting each limit that differs
    /// from the rank's first.
    fn record(&mut self, label: &str, ranks: &[[f64; 2]]) -> Value {
        if self.initial.is_empty() {
            self.initial = ranks.iter().map(|[limit, _]| *limit).collect();
        }
        for (rank, ([limit, _], initial)) in ranks.iter().zip(&self.initial).enumerate() {
            if limit != initial {
                self.changes.push(format!(
                    "{label}: rank {rank} enforces {limit} W, first {initial} W"
                ));
            }
        }
        let ranks: Vec<_> = ranks
            .iter()
            .enumerate()
            .map(|(rank, [limit, sm])| {
                json!({"rank": rank, "enforced_power_limit_w": limit, "sm_clock_mhz": sm})
            })
            .collect();
        let record = json!({"label": label, "ranks": ranks});
        self.samples.push(record.clone());
        record
    }
}

/// Step-time acceptance's statistics of `candidate` against `baseline`
/// over the two mirrored visits, as rates: d = baseline time / candidate
/// time - 1, positive when the candidate is faster. `saved` is the mean
/// microseconds per call the candidate saves.
struct Paired {
    d: [f64; 2],
    mean: f64,
    spread: f64,
    saved: f64,
}
impl Paired {
    fn new(baseline: [f64; 2], candidate: [f64; 2]) -> Self {
        let d = [
            baseline[0] / candidate[0] - 1.0,
            baseline[1] / candidate[1] - 1.0,
        ];
        Self {
            d,
            mean: (d[0] + d[1]) / 2.0,
            spread: (d[0] - d[1]).abs(),
            saved: (baseline[0] + baseline[1] - candidate[0] - candidate[1]) / 2.0,
        }
    }
    fn json(&self) -> Value {
        json!({"d1": self.d[0], "d2": self.d[1], "mean": self.mean, "spread": self.spread,
            "saved_us_per_call": self.saved})
    }
    /// A comparison resolved past its own spread and the A/A control's
    /// `bound`, with one sign in both visits.
    fn resolved(&self, bound: f64) -> bool {
        self.d[0] * self.d[1] > 0.0 && self.mean.abs() > self.spread.max(bound)
    }
}

/// Every rank's graph of `CALLS` repetitions of `plan`'s calls.
fn graphs(
    sessions: &[Session],
    fixture: &Fixture,
    rows: u32,
    plan: Plan,
) -> LabResult<Vec<GraphExec>> {
    sessions
        .iter()
        .enumerate()
        .map(|(rank, s)| {
            s.capture(|_| {
                for _ in 0..CALLS {
                    fixture.launch_with(s, rank, rows, plan)?;
                }
                Ok(())
            })
        })
        .collect()
}

/// One visit's slowest-rank median in microseconds per call and its
/// record, with the power sample taken before it.
fn visit(
    sessions: &[Session],
    graphs: &[GraphExec],
    label: &str,
    power: &mut Power,
) -> LabResult<(f64, Value)> {
    let before = power.sample(&format!("before {label}"))?;
    let timing = kernel_harness::with_timing_label(label, || {
        kernel_harness::time_multi_graph(sessions, graphs, CALLS, WARMUP, REPLAYS)
    })?;
    let us = |ms: f32| f64::from(ms) * 1000.0 / CALLS as f64;
    let medians: Vec<f64> = timing.per_rank.iter().map(|t| us(t.median_ms)).collect();
    let slowest = medians.iter().copied().fold(0.0, f64::max);
    let record = json!({
        "label": label,
        "slowest_rank_median_us": slowest,
        "critical_path_median_us": us(timing.critical_path.median_ms),
        "critical_path_min_us": us(timing.critical_path.min_ms),
        "per_rank_median_us": medians,
        "per_rank_min_us": timing.per_rank.iter().map(|t| us(t.min_ms)).collect::<Vec<_>>(),
        "power_before": before,
    });
    Ok((slowest, record))
}

fn bench(sessions: &[Session]) -> LabResult<PathBuf> {
    if sessions.len() != 4 {
        return Err("exactly four ranks required".into());
    }
    let started = Instant::now();
    let labels: Vec<_> = sessions
        .iter()
        .map(|s| s.metadata["measurement_label"].clone())
        .collect();
    if labels.iter().any(|label| label != "exclusive-device") {
        eprintln!("DECODE_REDUCTION shared devices: these timings are diagnostic only");
    }
    let cases = match std::env::var_os("CHOPINFER_BF16_PARTIAL_INPUTS") {
        Some(_) => Some(manifest()?),
        None => {
            eprintln!(
                "DECODE_REDUCTION CHOPINFER_BF16_PARTIAL_INPUTS is unset: all rows synthetic"
            );
            None
        }
    };
    let names = served_symbols();
    let names: Vec<_> = names.iter().map(String::as_str).collect();
    let served_f32 = Fixture::with_ops(sessions, ROW_CAPACITY, &names, |rank| {
        served_op(rank, ROW_CAPACITY)
    })?;
    let bf16 = Fixture::new(sessions, true, ROW_CAPACITY)?;
    let phased = Fixture::new(sessions, false, ROW_CAPACITY)?;
    let fixture = |arm| match arm {
        Arm::Deferred | Arm::Complete | Arm::Control | Arm::Round => &served_f32,
        Arm::Bf16 => &bf16,
        Arm::Phased => &phased,
    };
    let mut power = Power::start(sessions)?;
    let mut results = Vec::new();
    for rows in ROWS {
        let steps = [Step::Decode, Step::Verify]
            .into_iter()
            .filter_map(|step| Some((step, served(step, rows).transpose()?)))
            .map(|(step, form)| form.map(|form| (step, form)))
            .collect::<LabResult<Vec<_>>>()?;
        let &(primary, form) = steps.first().ok_or("no served step has these rows")?;
        let (values, source) = timed_inputs(cases.as_deref(), rows)?;
        // Every arm's complete step first equals its exact oracle.
        served_f32.exercise(sessions, &values, rows, rows, false)?;
        bf16.exercise(sessions, &values, rows, rows, true)?;
        phased.exercise(sessions, &values, rows, rows, false)?;
        let mut launched = BTreeMap::new();
        let mut timed = BTreeMap::new();
        for arm in ORDER {
            let calls = planned(&fixture(arm).ops[0], rows, arm.plan(form))?;
            launched.insert(
                arm.name(),
                calls.into_iter().map(|c| c.symbol).collect::<Vec<_>>(),
            );
            timed.insert(arm, graphs(sessions, fixture(arm), rows, arm.plan(form))?);
        }
        let mut us = BTreeMap::<Arm, Vec<f64>>::new();
        let mut visits = Vec::new();
        for cycle in 0..2 {
            let order: Vec<Arm> = match cycle {
                0 => ORDER.to_vec(),
                _ => ORDER.iter().rev().copied().collect(),
            };
            for arm in order {
                let label = format!("rows={rows} arm={} cycle={cycle}", arm.name());
                let (slowest, record) = visit(sessions, &timed[&arm], &label, &mut power)?;
                us.entry(arm).or_default().push(slowest);
                visits.push(record);
            }
        }
        let pair = |arm: Arm| [us[&arm][0], us[&arm][1]];
        let control = Paired::new(pair(form.arm()), pair(Arm::Control));
        let bound = control.mean.abs().max(control.spread);
        let compared = |baseline: Arm, role: &str| {
            let paired = Paired::new(pair(baseline), pair(Arm::Bf16));
            let mut record = paired.json();
            record["baseline"] = json!(baseline.name());
            record["candidate"] = json!(Arm::Bf16.name());
            record["role"] = json!(role);
            record["aa_bound"] = json!(bound);
            record["resolved"] = json!(paired.resolved(bound));
            (paired, record)
        };
        let mut record = json!({
            "rows": rows,
            "live_rows": rows,
            "inputs": source,
            "exact": true,
            "served_forms": steps.iter().map(|(step, form)| (step.name(), form.arm().name()))
                .collect::<BTreeMap<_, _>>(),
            "primary_step": primary.name(),
            "primary_comparison": format!("bf16_vs_served_{}", primary.name()),
            "launched": launched,
            "slowest_rank_median_us": us.iter().map(|(arm, us)| (arm.name(), us))
                .collect::<BTreeMap<_, _>>(),
            "aa_control": control.json(),
            "aa_bound": bound,
            "bf16_vs_f32_phased": compared(Arm::Phased, "same-transport precision reference").1,
            "f32_round_us_per_call": pair(Arm::Round).iter().sum::<f64>() / 2.0,
            "visits": visits,
        });
        for &(step, served_form) in &steps {
            let role = if step == primary { "primary" } else { "served" };
            let (paired, comparison) = compared(served_form.arm(), role);
            eprintln!(
                "DECODE_REDUCTION rows={rows} step={} baseline={} inputs={} f32_us={:?} \
                 bf16_partials_us={:?} saved_us={:.3} d=[{:.4},{:.4}] mean={:.4} \
                 spread={:.4} aa_bound={bound:.4} resolved={} elapsed_s={:.0}",
                step.name(),
                served_form.arm().name(),
                record["inputs"]["source"],
                pair(served_form.arm()),
                pair(Arm::Bf16),
                paired.saved,
                paired.d[0],
                paired.d[1],
                paired.mean,
                paired.spread,
                paired.resolved(bound),
                started.elapsed().as_secs_f64(),
            );
            record[format!("bf16_vs_served_{}", step.name())] = comparison;
        }
        results.push(record);
    }
    power.sample("end")?;
    let path = sessions[0].receipt(
        "deepseek41-decode-reductions",
        BTreeMap::from([
            ("case".into(), json!("decode_reduction_timing")),
            ("geometry".into(), json!(FLASH.id)),
            ("width".into(), json!(WIDTH)),
            ("ranks".into(), json!(4)),
            ("row_capacity".into(), json!(ROW_CAPACITY)),
            ("calls_per_graph".into(), json!(CALLS)),
            ("warmup_replays".into(), json!(WARMUP)),
            ("timed_replays".into(), json!(REPLAYS)),
            ("cycle_order".into(), json!(ORDER.map(Arm::name))),
            (
                "arms".into(),
                json!(
                    ORDER
                        .map(|arm| (arm.name(), arm.role()))
                        .into_iter()
                        .collect::<BTreeMap<_, _>>()
                ),
            ),
            ("rank_measurement_labels".into(), json!(labels)),
            ("power_samples".into(), json!(power.samples)),
            ("power_limit_changes".into(), json!(power.changes)),
            (
                "power_limits_constant".into(),
                json!(power.changes.is_empty()),
            ),
            ("void".into(), json!(!power.changes.is_empty())),
            ("elapsed_s".into(), json!(started.elapsed().as_secs_f64())),
            ("rows".into(), json!(results)),
        ]),
    )?;
    eprintln!("DECODE_REDUCTION_RECEIPT {}", path.display());
    if let Some(change) = power.changes.first() {
        return Err(format!(
            "an enforced power limit changed during the job, so it is void: {change}"
        ));
    }
    Ok(path)
}

#[test]
fn timed_arms_launch_what_serving_issues() {
    assert_eq!(FLASH.shape.hidden as usize, WIDTH);
    // The form each served step issues at each timed row count.
    use Form::{Complete, Deferred};
    let expected = [
        (1, Some(Deferred), Complete),
        (8, Some(Deferred), Deferred),
        (16, Some(Deferred), Deferred),
        (32, Some(Deferred), Deferred),
        (64, None, Deferred),
        (128, None, Complete),
    ];
    assert_eq!(expected.map(|(rows, ..)| rows), ROWS);
    for (rows, decode, verify) in expected {
        assert_eq!(served(Step::Decode, rows).unwrap(), decode, "decode {rows}");
        assert_eq!(
            served(Step::Verify, rows).unwrap(),
            Some(verify),
            "verify {rows}"
        );
    }
    // Verify defers at 4, 6, 8, 12 and 16 rows and at 32 to 64; every other
    // width up to the verify graphs' 192 rows issues the complete form.
    let most = DECODE_BATCH * MAX_VERIFY_ROWS;
    for rows in 0..=most + 1 {
        let form = (1..=most).contains(&rows).then_some(
            if [4, 6, 8, 12, 16].contains(&rows) || (32..=64).contains(&rows) {
                Deferred
            } else {
                Complete
            },
        );
        assert_eq!(served(Step::Verify, rows).unwrap(), form, "verify {rows}");
    }
    let symbols = served_symbols();
    for rank in 0..4 {
        let f32_op = served_op(rank, ROW_CAPACITY).unwrap();
        let bf16 = op(rank, true, ROW_CAPACITY).unwrap();
        let phased = op(rank, false, ROW_CAPACITY).unwrap();
        for (rows, decode, verify) in expected {
            let primary = decode.unwrap_or(verify);
            let names = |op: &Op, plan: Plan| -> Vec<String> {
                planned(op, rows, plan)
                    .unwrap()
                    .into_iter()
                    .map(|call| call.symbol)
                    .collect()
            };
            assert_eq!(names(&f32_op, Arm::Deferred.plan(primary)), symbols[..2]);
            assert_eq!(names(&f32_op, Arm::Complete.plan(primary)), symbols);
            assert_eq!(names(&f32_op, Arm::Round.plan(primary)), symbols[2..]);
            assert_eq!(
                names(&f32_op, Arm::Control.plan(primary)),
                names(&f32_op, primary.arm().plan(primary))
            );
            for (op, suffix) in [(&bf16, "_bf16_v1"), (&phased, "_round_f32_v1")] {
                let launched = names(op, every);
                assert_eq!(launched.len(), 1);
                assert!(launched[0].starts_with("chop_comm_allreduce_phased_"));
                assert!(launched[0].ends_with(suffix));
            }
        }
    }
}

#[test]
fn power_limits_are_enforced_readings_against_each_ranks_first() {
    let uuids: Vec<String> = (0..4).map(|rank| format!("GPU-{rank}")).collect();
    let text = |limits: [&str; 4]| {
        (0..4)
            .map(|rank| format!("GPU-{rank}, {}, 2400\n", limits[rank]))
            .collect::<String>()
    };
    assert_eq!(
        parse_readings(&text(["500.00"; 4]), &uuids).unwrap(),
        vec![[500.0, 2400.0]; 4]
    );
    for missing in ["[N/A]", "[Not Supported]", "", "0"] {
        let limits = ["500.00", "500.00", missing, "500.00"];
        assert!(
            parse_readings(&text(limits), &uuids).is_err(),
            "{missing:?}"
        );
    }
    assert!(parse_readings("GPU-0, 500.00, 2400\n", &uuids).is_err());
    assert!(parse_readings("GPU-0, 500.00, [N/A]\n", &uuids[..1]).is_err());
    // Equal endpoints do not hide a different limit at every visit.
    let mut power = Power::default();
    power.record("start", &[[500.0, 2400.0]; 4]);
    for visit in 0..3 {
        let mut ranks = [[500.0, 2400.0]; 4];
        ranks[1][0] = 400.0;
        power.record(&format!("visit {visit}"), &ranks);
    }
    power.record("end", &[[500.0, 2400.0]; 4]);
    assert_eq!(power.changes.len(), 3);
    assert!(power.changes[0].contains("rank 1 enforces 400 W, first 500 W"));
    let mut steady = Power::default();
    for label in ["start", "visit", "end"] {
        steady.record(
            label,
            &[
                [500.0, 2400.0],
                [450.0, 2400.0],
                [500.0, 2100.0],
                [600.0, 2400.0],
            ],
        );
    }
    assert!(steady.changes.is_empty());
    assert_eq!(steady.samples.len(), 3);
}

#[test]
fn receipts_stay_outside_the_tree() {
    let scratch = std::env::temp_dir().join(format!("decode-reductions-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    let tree = scratch.join("tree");
    let outside = scratch.join("outside");
    std::fs::create_dir_all(tree.join("lab")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::os::unix::fs::symlink(&tree, scratch.join("link")).unwrap();
    let lab = tree.join("lab");
    for (root, cwd) in [
        (PathBuf::from("receipts"), lab.clone()),
        (PathBuf::from("../new/receipts"), lab.clone()),
        (tree.clone(), outside.clone()),
        (scratch.join("link/receipts"), outside.clone()),
        (outside.join("../tree/receipts"), outside.clone()),
        (outside.join("missing/../../tree"), outside.clone()),
    ] {
        assert!(external(&root, &cwd, &tree).is_err(), "{}", root.display());
    }
    let resolved = external(Path::new("receipts"), &outside, &tree).unwrap();
    assert_eq!(resolved, outside.canonicalize().unwrap().join("receipts"));
    assert!(external(&outside.join("a/b"), &lab, &tree).is_ok());
    assert!(external(&outside, &lab, &scratch.join("missing")).is_ok());
    std::fs::remove_dir_all(&scratch).unwrap();
}

#[test]
fn decode_reduction_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        eprintln!(
            "skipping the decode reduction timing: set CHOPINFER_KERNEL_BENCH=1 and \
             CHOPINFER_KERNEL_GPUS to four idle GPUs"
        );
        return;
    }
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip(
            "the decode reduction timing: set CHOPINFER_KERNEL_GPUS to four idle GPUs",
        )
        .unwrap();
        return;
    }
    // The harness writes under this root; refuse the tree before any GPU work.
    let receipts = external(
        &kernel_harness::receipt_root(),
        &std::env::current_dir().unwrap(),
        &tree(),
    )
    .unwrap();
    eprintln!("DECODE_REDUCTION receipts under {}", receipts.display());
    let result = kernel_harness::gpu_test_multi_with(probe().unwrap(), 2 << 30, bench).unwrap();
    assert!(result.is_some(), "the timing must run on four GPUs");
}
