//! The host load a receipt records (`common.host_load`). It labels the
//! receipt's throughput and latency figures and never changes a verdict.
//!
//! The host is idle for timing evidence when, over every interval between
//! samples, the CPU idle fraction (idle plus iowait over all CPU time, so
//! niced work counts as busy) is at least [`IDLE_CPU_FRACTION`] and the
//! shares of elapsed time in which some task stalled waiting for a CPU and
//! for I/O (the "some" totals of /proc/pressure/cpu and /proc/pressure/io)
//! are each at most [`IDLE_PRESSURE_SHARE`]. Elapsed time is the host's
//! uptime, not wall time. The 1-minute load average is recorded, not
//! judged. An interval whose counters or uptime do not advance makes the
//! figures informational, as do fewer than two samples or missing pressure
//! stall information. tools/qual/common.py states why and applies the same
//! rule to the same inputs.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

pub const IDLE_CPU_FRACTION: f64 = 0.9;
pub const IDLE_PRESSURE_SHARE: f64 = 0.1;
pub const RULE: &str = "idle when, over every interval between samples, the CPU idle fraction is at least 0.9 and the CPU and I/O pressure stall shares are at most 0.1; otherwise throughput and latency are informational";
/// The judged pressure stall kinds and their names in a reason.
const PRESSURE_KINDS: [(&str, &str); 2] = [("cpu", "CPU"), ("io", "I/O")];

/// This host's wall time, uptime, load averages, cumulative idle and total
/// CPU ticks, online CPUs and cumulative pressure stall microseconds
/// (`common.host_load_sample`), the uptime and stalls null without them, or
/// null without /proc.
pub fn sample() -> Value {
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let pressure = read("/proc/pressure/cpu").zip(read("/proc/pressure/io"));
    let uptime = read("/proc/uptime");
    read("/proc/loadavg")
        .zip(read("/proc/stat"))
        .and_then(|(loadavg, stat)| {
            parse(
                &loadavg,
                &stat,
                uptime.as_deref(),
                pressure
                    .as_ref()
                    .map(|(cpu, io)| (cpu.as_str(), io.as_str())),
                now(),
            )
        })
        .unwrap_or(Value::Null)
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

fn parse(
    loadavg: &str,
    stat: &str,
    uptime: Option<&str>,
    pressure: Option<(&str, &str)>,
    at: f64,
) -> Option<Value> {
    let load = loadavg
        .split_whitespace()
        .take(3)
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    // user nice system idle iowait irq softirq steal; guest time is in user.
    let ticks = stat
        .lines()
        .next()?
        .split_whitespace()
        .skip(1)
        .take(8)
        .map(str::parse::<u64>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    if load.len() != 3 || ticks.len() != 8 {
        return None;
    }
    let cpus = stat
        .lines()
        .filter(|line| {
            line.strip_prefix("cpu")
                .and_then(|rest| rest.chars().next())
                .is_some_and(|next| next.is_ascii_digit())
        })
        .count();
    let pressure = pressure
        .and_then(|(cpu, io)| Some(json!({"cpu": some_total(cpu)?, "io": some_total(io)?})));
    let uptime = uptime
        .and_then(|text| text.split_whitespace().next())
        .filter(|seconds| {
            seconds
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b'.')
        })
        .and_then(|seconds| seconds.parse::<f64>().ok());
    Some(json!({
        "at": at,
        "uptime_seconds": uptime,
        "load_average": load,
        "cpu_idle_ticks": ticks[3] + ticks[4],
        "cpu_ticks": ticks.iter().sum::<u64>(),
        "cpus": cpus,
        "pressure_some_us": pressure,
    }))
}

/// The cumulative "some" stall microseconds of a /proc/pressure file.
fn some_total(text: &str) -> Option<u64> {
    let mut fields = text.lines().next()?.split_whitespace();
    if fields.next()? != "some" {
        return None;
    }
    let total = fields.find_map(|field| field.strip_prefix("total="))?;
    // Digits only, as int() reads the kernel's text in common.py.
    if total.is_empty() || !total.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    total.parse().ok()
}

/// A cumulative counter (`common.counter`): an integer from 0 to u64::MAX.
fn counter(value: &Value) -> Option<u64> {
    value.as_u64()
}

/// A finite JSON number (`common.seconds`).
fn seconds(value: &Value) -> Option<f64> {
    value.as_f64().filter(|number| number.is_finite())
}

/// Whether `row` has what the CPU part of the rule reads
/// (`common.judged_sample`).
fn judged_sample(row: &Value) -> bool {
    row.is_object()
        && row["load_average"]
            .as_array()
            .and_then(|load| load.first())
            .and_then(seconds)
            .is_some()
        && counter(&row["cpu_idle_ticks"]).is_some()
        && counter(&row["cpu_ticks"]).is_some()
}

/// Whether `row` also has the uptime and stalls the pressure part reads
/// (`common.timed_sample`).
fn timed_sample(row: &Value) -> bool {
    seconds(&row["uptime_seconds"]).is_some()
        && row["pressure_some_us"].is_object()
        && PRESSURE_KINDS
            .iter()
            .all(|(kind, _)| counter(&row["pressure_some_us"][kind]).is_some())
}

/// `samples` (oldest first) judged by [`RULE`], recording `cpus` online
/// CPUs (default: the latest sample's count).
pub fn judge(samples: &[Value], cpus: Option<u64>) -> Value {
    let cpus = cpus
        .or_else(|| {
            samples
                .iter()
                .rev()
                .find_map(|row| counter(&row["cpus"]).filter(|&cpus| cpus > 0))
        })
        .or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .map(|cpus| cpus.get() as u64)
        })
        .unwrap_or(1);
    let mut peak = Value::Null;
    let mut whole = Value::Null;
    let mut lowest = Value::Null;
    let mut shares = [Value::Null, Value::Null];
    let mut reasons = Vec::new();
    let pairs = samples.windows(2).collect::<Vec<_>>();
    // A counter's change between two samples, None if it went back.
    let change = |pair: &[Value], read: &dyn Fn(&Value) -> Option<u64>| {
        read(&pair[1])?.checked_sub(read(&pair[0])?)
    };
    if samples.len() < 2 {
        reasons.push("fewer than two host load samples".to_owned());
    } else if !samples.iter().all(judged_sample) {
        reasons.push("a host load sample is missing or incomplete".to_owned());
    } else {
        // Recorded, not judged.
        peak = json!(
            samples
                .iter()
                .filter_map(|row| seconds(&row["load_average"][0]))
                .fold(f64::NEG_INFINITY, f64::max)
        );
        let ticks = pairs
            .iter()
            .map(|pair| {
                Some((
                    change(pair, &|row| counter(&row["cpu_idle_ticks"]))?,
                    change(pair, &|row| counter(&row["cpu_ticks"]))?,
                ))
                .filter(|&(idle, total)| total > 0 && idle <= total)
            })
            .collect::<Option<Vec<_>>>();
        match ticks {
            None => reasons.push("CPU ticks do not advance between host load samples".to_owned()),
            Some(ticks) => {
                let (first, last) = (&samples[0], &samples[samples.len() - 1]);
                let spent = |key: &str| {
                    counter(&last[key]).unwrap_or(0) - counter(&first[key]).unwrap_or(0)
                };
                whole = json!(spent("cpu_idle_ticks") as f64 / spent("cpu_ticks") as f64);
                let low = ticks
                    .iter()
                    .map(|&(idle, total)| idle as f64 / total as f64)
                    .fold(f64::INFINITY, f64::min);
                lowest = json!(low);
                if low < IDLE_CPU_FRACTION {
                    reasons.push(format!("CPU idle fell to {low:.3}"));
                }
                if !samples.iter().all(timed_sample) {
                    reasons.push("no uptime or pressure stall information".to_owned());
                } else {
                    let elapsed = pairs
                        .iter()
                        .map(|pair| {
                            Some(
                                seconds(&pair[1]["uptime_seconds"])?
                                    - seconds(&pair[0]["uptime_seconds"])?,
                            )
                            .filter(|&span| span > 0.0)
                        })
                        .collect::<Option<Vec<_>>>();
                    let stalls = PRESSURE_KINDS
                        .iter()
                        .map(|(kind, _)| {
                            pairs
                                .iter()
                                .map(|pair| {
                                    change(pair, &|row| counter(&row["pressure_some_us"][kind]))
                                })
                                .collect::<Option<Vec<_>>>()
                        })
                        .collect::<Option<Vec<_>>>();
                    match elapsed.zip(stalls) {
                        None => reasons.push(
                            "uptime or pressure stalls do not advance between host load samples"
                                .to_owned(),
                        ),
                        Some((elapsed, stalls)) => {
                            for (((_, name), stalls), share) in
                                PRESSURE_KINDS.iter().zip(&stalls).zip(&mut shares)
                            {
                                let high = stalls
                                    .iter()
                                    .zip(&elapsed)
                                    .map(|(&stall, &span)| stall as f64 / 1e6 / span)
                                    .fold(f64::NEG_INFINITY, f64::max);
                                *share = json!(high);
                                if high > IDLE_PRESSURE_SHARE {
                                    reasons.push(format!(
                                        "{name} pressure stall share reached {high:.3}"
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let [cpu_share, io_share] = shares;
    json!({
        "rule": RULE,
        "scope": "check_host",
        "cpus": cpus,
        "samples": samples,
        "max_load_average_1m": peak,
        "cpu_idle_fraction": whole,
        "min_cpu_idle_fraction": lowest,
        "max_cpu_pressure_share": cpu_share,
        "max_io_pressure_share": io_share,
        "idle": reasons.is_empty(),
        "timing": if reasons.is_empty() { "evidence" } else { "informational" },
        "reasons": reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample `at` seconds with `idle` of `total` CPU ticks and `cpu` and
    /// `io` cumulative pressure stall microseconds (`test_host_load.sample`).
    fn row(idle: u64, total: u64, at: f64, cpu: u64, io: u64) -> Value {
        json!({"at": at, "uptime_seconds": at, "load_average": [1.0, 1.0, 1.0], "cpu_idle_ticks": idle,
               "cpu_ticks": total, "pressure_some_us": {"cpu": cpu, "io": io}})
    }

    fn reasons(rows: &[Value]) -> Value {
        judge(rows, Some(48))["reasons"].clone()
    }

    #[test]
    fn proc_is_parsed() {
        let stat = "cpu  100 20 30 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5 0 0 0 0 0\n";
        let value = parse(
            "1.25 0.90 0.50 2/900 1234\n",
            stat,
            Some("5832.17 270000.00\n"),
            Some((
                "some avg10=0.00 avg60=0.10 avg300=0.20 total=4567\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
                "some avg10=1.00 avg60=0.50 avg300=0.20 total=89012\nfull avg10=1.00 avg60=0.50 avg300=0.20 total=88000\n",
            )),
            7.0,
        )
        .unwrap();
        assert_eq!(value["load_average"], json!([1.25, 0.90, 0.50]));
        assert_eq!(
            (
                value["cpu_idle_ticks"].as_u64(),
                value["cpu_ticks"].as_u64()
            ),
            (Some(850), Some(1000))
        );
        assert_eq!(value["cpus"], 1);
        assert_eq!(value["pressure_some_us"], json!({"cpu": 4567, "io": 89012}));
        assert_eq!(value["uptime_seconds"], json!(5832.17));
        assert!(parse("", "cpu 1 2 3\n", None, None, 0.0).is_none());
        // Without pressure stall information the sample records null, and
        // its timing is informational.
        let bare = parse("1.25 0.90 0.50 2/900 1234\n", stat, None, None, 7.0).unwrap();
        assert!(bare["pressure_some_us"].is_null());
        assert!(bare["uptime_seconds"].is_null());
        let mut later = bare.clone();
        later["at"] = json!(8.0);
        later["cpu_ticks"] = json!(2000);
        later["cpu_idle_ticks"] = json!(1850);
        assert_eq!(
            reasons(&[bare, later]),
            json!(["no uptime or pressure stall information"])
        );
    }

    #[test]
    fn an_idle_host_keeps_timing_as_evidence() {
        let verdict = judge(
            &[row(0, 0, 0.0, 0, 0), row(970, 1000, 30.0, 21000, 6000)],
            Some(48),
        );
        assert_eq!(verdict["timing"], "evidence");
        assert_eq!(verdict["reasons"], json!([]));
        assert!((verdict["min_cpu_idle_fraction"].as_f64().unwrap() - 0.97).abs() < 1e-12);
        assert!((verdict["max_cpu_pressure_share"].as_f64().unwrap() - 0.0007).abs() < 1e-12);
        assert!((verdict["max_io_pressure_share"].as_f64().unwrap() - 0.0002).abs() < 1e-12);
        // The bounds are inclusive.
        assert_eq!(
            reasons(&[row(0, 0, 0.0, 0, 0), row(900, 1000, 1.0, 100_000, 100_000)]),
            json!([])
        );
        // The load average is recorded, not judged.
        let mut start = row(0, 0, 0.0, 0, 0);
        start["load_average"] = json!([11.81, 9.98, 10.0]);
        let verdict = judge(&[start, row(970, 1000, 30.0, 0, 0)], Some(48));
        assert_eq!(
            (
                verdict["timing"].clone(),
                verdict["max_load_average_1m"].clone()
            ),
            (json!("evidence"), json!(11.81))
        );
    }

    #[test]
    fn a_busy_or_unknown_host_makes_timing_informational() {
        let verdict = judge(&[row(0, 0, 0.0, 0, 0), row(80, 1000, 30.0, 0, 0)], Some(48));
        assert_eq!(verdict["timing"], "informational");
        assert_eq!(verdict["reasons"], json!(["CPU idle fell to 0.080"]));
        assert_eq!(
            reasons(&[row(0, 0, 0.0, 0, 0), row(920, 1000, 30.0, 9_000_000, 0)]),
            json!(["CPU pressure stall share reached 0.300"])
        );
        // iowait counts as idle CPU, so only the I/O pressure sees it.
        assert_eq!(
            reasons(&[row(0, 0, 0.0, 0, 0), row(990, 1000, 30.0, 0, 18_000_000)]),
            json!(["I/O pressure stall share reached 0.600"])
        );
        let rows = [
            row(0, 0, 0.0, 0, 0),
            row(990, 1000, 30.0, 0, 0),
            row(1500, 2000, 60.0, 0, 0),
            row(2490, 3000, 90.0, 0, 0),
        ];
        let verdict = judge(&rows, Some(48));
        assert_eq!(verdict["idle"], false);
        assert!((verdict["cpu_idle_fraction"].as_f64().unwrap() - 0.83).abs() < 1e-12);
        assert_eq!(
            reasons(&[
                row(0, 0, 0.0, 0, 0),
                row(990, 1000, 30.0, 0, 30_000),
                row(1980, 2000, 60.0, 0, 6_030_000)
            ]),
            json!(["I/O pressure stall share reached 0.200"])
        );
        let mut bare = row(990, 1000, 1.0, 0, 0);
        bare["pressure_some_us"] = Value::Null;
        let mut timeless = row(990, 1000, 1.0, 0, 0);
        timeless["uptime_seconds"] = Value::Null;
        for rows in [
            vec![],
            vec![row(0, 0, 0.0, 0, 0)],
            vec![Value::Null, row(990, 1000, 1.0, 0, 0)],
            vec![row(5, 10, 0.0, 0, 0), row(5, 10, 1.0, 0, 0)],
            vec![row(0, 0, 1.0, 0, 0), row(990, 1000, 1.0, 0, 0)],
            vec![row(0, 0, 0.0, 0, 0), bare],
            vec![row(0, 0, 0.0, 0, 0), timeless],
        ] {
            assert_eq!(judge(&rows, Some(48))["timing"], "informational");
        }
    }

    /// The recorded soaks judge to the reasons test_host_load.py's
    /// RecordedSoakTests expects.
    #[test]
    fn recorded_soaks_judge_as_in_common_py() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/host-load/soak-samples.json"
        );
        let soaks: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let judged = |name: &str| judge(soaks[name].as_array().unwrap(), None);
        let after_startup = judged("soak-after-startup-2026-10-17");
        assert_eq!(
            (
                after_startup["timing"].clone(),
                after_startup["reasons"].clone(),
                after_startup["max_load_average_1m"].clone()
            ),
            (json!("evidence"), json!([]), json!(10.35))
        );
        assert_eq!(
            judged("soak-unsampled-2026-10-16")["reasons"],
            json!(["no uptime or pressure stall information"])
        );
        assert_eq!(
            judged("soak-busy-cpu-2026-10-16")["reasons"],
            json!([
                "CPU idle fell to 0.804",
                "no uptime or pressure stall information"
            ])
        );
    }

    /// fixtures/host-load/judge-cases.json gives the reasons
    /// test_host_load.py's SharedCaseTests expects.
    #[test]
    fn shared_cases_judge_as_in_common_py() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/host-load/judge-cases.json"
        );
        let cases: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let cases = cases["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let verdict = judge(case["samples"].as_array().unwrap(), None);
            assert_eq!(verdict["reasons"], case["reasons"], "{}", case["name"]);
            let empty = case["reasons"].as_array().unwrap().is_empty();
            assert_eq!(
                verdict["timing"],
                if empty { "evidence" } else { "informational" },
                "{}",
                case["name"]
            );
        }
    }
}
