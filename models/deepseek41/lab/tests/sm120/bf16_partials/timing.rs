//! Decode's served reduction against the BF16-partial transport prefill
//! serves, timed with CUDA events in graph replays on four ranks under the
//! pair-lab rules. Opt-in (`CHOPINFER_KERNEL_BENCH=1`): the timings locate
//! work and never decide adoption.
//!
//! Each arm all-reduces rows of 5120 F32 partials in rank order:
//!
//! - `f32_served`, the served baseline: decode's rounded all-reduce on the
//!   bundle's registration (`reduce::register`, as `Kernels::new_for_kv`
//!   registers it), whose composed form is the F32 reduce-scatter and
//!   all-gather. Decode's next HC operation performs its BF16 round
//!   (`ReduceOps::defer_round`), so the arm issues only the pair.
//! - `f32_served_control`: the same calls on the same operands in a second
//!   graph, the in-job A/A control.
//! - `bf16_partials`, the candidate: the phased BF16-partial all-reduce
//!   prefill serves.
//! - `f32_phased`, a same-transport precision reference: the phased F32
//!   rounding entries, which no decode step serves
//!   (`Kernels::prefill_reduction_backend`).
//! - `f32_round`: the round decode defers, alone. Reported beside the
//!   others, never counted as work BF16 partials remove.
//!
//! The primary comparison is `bf16_partials` against `f32_served`.
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
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::GraphExec;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Arm {
    Served,
    Control,
    Bf16,
    Phased,
    Round,
}
/// One cycle's visits; the second cycle visits them in reverse, so each
/// pair of arms runs A1 B1 B2 A2.
const ORDER: [Arm; 5] = [
    Arm::Served,
    Arm::Control,
    Arm::Bf16,
    Arm::Phased,
    Arm::Round,
];
impl Arm {
    fn name(self) -> &'static str {
        match self {
            Self::Served => "f32_served",
            Self::Control => "f32_served_control",
            Self::Bf16 => "bf16_partials",
            Self::Phased => "f32_phased",
            Self::Round => "f32_round",
        }
    }
    /// What the arm times, as its receipt states it.
    fn role(self) -> &'static str {
        match self {
            Self::Served => {
                "served baseline: decode's composed F32 reduce-scatter and all-gather, \
                 its round deferred into the next HC operation"
            }
            Self::Control => "A/A control: the served baseline's calls in a second graph",
            Self::Bf16 => "candidate: the phased BF16-partial all-reduce prefill serves",
            Self::Phased => {
                "same-transport precision reference: the phased F32 rounding entries, \
                 which no decode step serves"
            }
            Self::Round => "the served baseline's deferred round alone, not removable work",
        }
    }
    fn plan(self) -> Plan {
        match self {
            Self::Served | Self::Control => deferred,
            Self::Round => round,
            Self::Bf16 | Self::Phased => every,
        }
    }
}

/// The composed form's pair and the round decode defers.
fn served_symbols() -> Vec<String> {
    vec![
        reduce::SCATTER_SYMBOL.into(),
        reduce::GATHER_SYMBOL.into(),
        reduce::rounding_symbol(&FLASH, Rounding::Round),
    ]
}
/// Rank `rank`'s decode reduction as the bundle registers it: the rounded
/// all-reduce over the rounding rows alone, without the phased F32
/// entries, so every row count takes the composed form.
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
/// The calls decode issues: every call before the round its HC operation
/// performs, checked as `ReduceOps::defer_round` checks them.
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
/// The deferred round alone: the step's calls after the deferred ones.
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

/// Each rank's enforced power limit and SM clock, as nvidia-smi reports
/// them. Recorded only: the operator owns limits and clocks.
fn power(sessions: &[Session]) -> LabResult<Vec<Value>> {
    sessions
        .iter()
        .enumerate()
        .map(|(rank, s)| {
            let uuid = s.metadata["uuid"].as_str().ok_or("missing device UUID")?;
            let output = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=enforced.power.limit,clocks.sm")
                .arg("--format=csv,noheader,nounits")
                .arg(format!("--id={uuid}"))
                .output()
                .map_err(err)?;
            if !output.status.success() {
                return Err("nvidia-smi cannot report power limits".into());
            }
            let text = String::from_utf8(output.stdout).map_err(err)?;
            let fields: Vec<_> = text.trim().split(',').map(str::trim).collect();
            let [limit, sm] = fields[..] else {
                return Err(format!("unexpected nvidia-smi fields: {text}"));
            };
            Ok(json!({"rank": rank, "ordinal": s.ordinal,
                "enforced_power_limit_w": limit, "sm_clock_mhz": sm}))
        })
        .collect()
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

/// One visit's slowest-rank median in microseconds per call, its record,
/// and each timed replay's per-rank power limits.
fn visit(
    sessions: &[Session],
    graphs: &[GraphExec],
    label: &str,
) -> LabResult<(f64, Value, Vec<Value>)> {
    let timing = kernel_harness::with_timing_label(label, || {
        kernel_harness::time_multi_graph(sessions, graphs, CALLS, WARMUP, REPLAYS)
    })?;
    let us = |ms: f32| f64::from(ms) * 1000.0 / CALLS as f64;
    let medians: Vec<f64> = timing.per_rank.iter().map(|t| us(t.median_ms)).collect();
    let slowest = medians.iter().copied().fold(0.0, f64::max);
    let ranks = |round: &Value, field: &str| -> Value {
        round["ranks"]
            .as_array()
            .map(|ranks| {
                ranks
                    .iter()
                    .map(|r| r["gpu_state"][field].clone())
                    .collect()
            })
            .unwrap_or_default()
    };
    let limits = timing
        .gpu_rounds
        .iter()
        .map(|round| ranks(round, "power_limit_mw"))
        .collect();
    let (first, last) = (timing.gpu_rounds.first(), timing.gpu_rounds.last());
    let state = |round: Option<&Value>| {
        round.map(
            |r| json!({"power_limit_mw": ranks(r, "power_limit_mw"), "sm_mhz": ranks(r, "sm_mhz")}),
        )
    };
    let record = json!({
        "label": label,
        "slowest_rank_median_us": slowest,
        "critical_path_median_us": us(timing.critical_path.median_ms),
        "critical_path_min_us": us(timing.critical_path.min_ms),
        "per_rank_median_us": medians,
        "per_rank_min_us": timing.per_rank.iter().map(|t| us(t.min_ms)).collect::<Vec<_>>(),
        "gpu_state_first": state(first),
        "gpu_state_last": state(last),
    });
    Ok((slowest, record, limits))
}

fn bench(sessions: &[Session]) -> LabResult<std::path::PathBuf> {
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
    let served = Fixture::with_ops(sessions, ROW_CAPACITY, &names, |rank| {
        served_op(rank, ROW_CAPACITY)
    })?;
    let bf16 = Fixture::new(sessions, true, ROW_CAPACITY)?;
    let phased = Fixture::new(sessions, false, ROW_CAPACITY)?;
    let fixture = |arm| match arm {
        Arm::Served | Arm::Control | Arm::Round => &served,
        Arm::Bf16 => &bf16,
        Arm::Phased => &phased,
    };
    let power_start = power(sessions)?;
    let mut limits = BTreeSet::new();
    let mut results = Vec::new();
    for rows in ROWS {
        let (values, source) = timed_inputs(cases.as_deref(), rows)?;
        // Every arm's complete step first equals its exact oracle.
        served.exercise(sessions, &values, rows, rows, false)?;
        bf16.exercise(sessions, &values, rows, rows, true)?;
        phased.exercise(sessions, &values, rows, rows, false)?;
        let mut launched = BTreeMap::new();
        let mut timed = BTreeMap::new();
        for arm in ORDER {
            let calls = planned(&fixture(arm).ops[0], rows, arm.plan())?;
            launched.insert(
                arm.name(),
                calls.into_iter().map(|c| c.symbol).collect::<Vec<_>>(),
            );
            timed.insert(arm, graphs(sessions, fixture(arm), rows, arm.plan())?);
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
                let (slowest, record, rounds) = visit(sessions, &timed[&arm], &label)?;
                us.entry(arm).or_default().push(slowest);
                visits.push(record);
                limits.extend(rounds.iter().map(Value::to_string));
            }
        }
        let pair = |arm: Arm| [us[&arm][0], us[&arm][1]];
        let control = Paired::new(pair(Arm::Served), pair(Arm::Control));
        let bound = control.mean.abs().max(control.spread);
        let served_vs = Paired::new(pair(Arm::Served), pair(Arm::Bf16));
        let phased_vs = Paired::new(pair(Arm::Phased), pair(Arm::Bf16));
        let compared = |baseline: Arm, paired: &Paired, role: &str| {
            let mut record = paired.json();
            record["baseline"] = json!(baseline.name());
            record["candidate"] = json!(Arm::Bf16.name());
            record["role"] = json!(role);
            record["aa_bound"] = json!(bound);
            record["resolved"] = json!(paired.resolved(bound));
            record
        };
        eprintln!(
            "DECODE_REDUCTION rows={rows} inputs={} served_f32_us={:?} bf16_partials_us={:?} \
             saved_us={:.3} d=[{:.4},{:.4}] mean={:.4} spread={:.4} aa_bound={bound:.4} \
             resolved={} f32_phased_us={:?} f32_round_us={:?} elapsed_s={:.0}",
            source["source"],
            pair(Arm::Served),
            pair(Arm::Bf16),
            served_vs.saved,
            served_vs.d[0],
            served_vs.d[1],
            served_vs.mean,
            served_vs.spread,
            served_vs.resolved(bound),
            pair(Arm::Phased),
            pair(Arm::Round),
            started.elapsed().as_secs_f64(),
        );
        let medians: BTreeMap<_, _> = us.iter().map(|(arm, us)| (arm.name(), us)).collect();
        results.push(json!({
            "rows": rows,
            "live_rows": rows,
            "inputs": source,
            "exact": true,
            "launched": launched,
            "slowest_rank_median_us": medians,
            "aa_control": control.json(),
            "aa_bound": bound,
            "bf16_vs_f32_served": compared(Arm::Served, &served_vs, "primary"),
            "bf16_vs_f32_phased": compared(Arm::Phased, &phased_vs, "same-transport precision reference"),
            "f32_round_us_per_call": pair(Arm::Round).iter().sum::<f64>() / 2.0,
            "visits": visits,
        }));
    }
    let power_end = power(sessions)?;
    let enforced = |records: &[Value]| -> Vec<Value> {
        records
            .iter()
            .map(|r| r["enforced_power_limit_w"].clone())
            .collect()
    };
    let constant = limits.len() <= 1 && enforced(&power_start) == enforced(&power_end);
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
            ("primary_comparison".into(), json!("bf16_vs_f32_served")),
            ("rank_measurement_labels".into(), json!(labels)),
            ("power_start".into(), json!(power_start)),
            ("power_end".into(), json!(power_end)),
            ("power_limits_constant".into(), json!(constant)),
            ("void".into(), json!(!constant)),
            ("elapsed_s".into(), json!(started.elapsed().as_secs_f64())),
            ("rows".into(), json!(results)),
        ]),
    )?;
    eprintln!("DECODE_REDUCTION_RECEIPT {}", path.display());
    if !constant {
        return Err("a power limit changed during the job, so it is void".into());
    }
    Ok(path)
}

#[test]
fn timed_arms_launch_what_serving_issues() {
    assert_eq!(FLASH.shape.hidden as usize, WIDTH);
    let symbols = served_symbols();
    for rank in 0..4 {
        let served = served_op(rank, ROW_CAPACITY).unwrap();
        let bf16 = op(rank, true, ROW_CAPACITY).unwrap();
        let phased = op(rank, false, ROW_CAPACITY).unwrap();
        for rows in ROWS {
            let names = |op: &Op, plan: Plan| -> Vec<String> {
                planned(op, rows, plan)
                    .unwrap()
                    .into_iter()
                    .map(|call| call.symbol)
                    .collect()
            };
            assert_eq!(names(&served, deferred), symbols[..2]);
            assert_eq!(names(&served, round), symbols[2..]);
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
    let result = kernel_harness::gpu_test_multi_with(probe().unwrap(), 2 << 30, bench).unwrap();
    assert!(result.is_some(), "the timing must run on four GPUs");
}
