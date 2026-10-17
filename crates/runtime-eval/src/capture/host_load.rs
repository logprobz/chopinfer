//! The host load a receipt records (`common.host_load`). It labels the
//! receipt's throughput and latency figures and never changes a verdict.
//!
//! The host is idle for timing evidence when, over every interval between
//! samples, the CPU idle fraction (idle plus iowait over all CPU time, so
//! niced work counts as busy) is at least [`IDLE_CPU_FRACTION`] and the
//! shares of wall time in which some task stalled waiting for a CPU and for
//! I/O (the "some" totals of /proc/pressure/cpu and /proc/pressure/io) are
//! each at most [`IDLE_PRESSURE_SHARE`]. The 1-minute load average is
//! recorded, not judged. Otherwise, or without two samples or pressure stall
//! information, the figures are informational. tools/qual/common.py states
//! why and applies the same rule.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

pub const IDLE_CPU_FRACTION: f64 = 0.9;
pub const IDLE_PRESSURE_SHARE: f64 = 0.1;
pub const RULE: &str = "idle when, over every interval between samples, the CPU idle fraction is at least 0.9 and the CPU and I/O pressure stall shares are at most 0.1; otherwise throughput and latency are informational";
/// The judged pressure stall kinds and their names in a reason.
const PRESSURE_KINDS: [(&str, &str); 2] = [("cpu", "CPU"), ("io", "I/O")];

/// This host's load averages, cumulative idle and total CPU ticks, online
/// CPUs and cumulative pressure stall microseconds
/// (`common.host_load_sample`), or null without /proc.
pub fn sample() -> Value {
    let read = |path: &str| std::fs::read_to_string(path).ok();
    let pressure = read("/proc/pressure/cpu").zip(read("/proc/pressure/io"));
    read("/proc/loadavg")
        .zip(read("/proc/stat"))
        .and_then(|(loadavg, stat)| {
            parse(
                &loadavg,
                &stat,
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

fn parse(loadavg: &str, stat: &str, pressure: Option<(&str, &str)>, at: f64) -> Option<Value> {
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
    Some(json!({
        "at": at,
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
    fields
        .find_map(|field| field.strip_prefix("total="))?
        .parse()
        .ok()
}

fn number(row: &Value, key: &str) -> Option<f64> {
    row[key].as_f64()
}

fn valid(row: &Value) -> bool {
    row["load_average"][0].is_number()
        && number(row, "at").is_some()
        && number(row, "cpu_idle_ticks").is_some()
        && number(row, "cpu_ticks").is_some()
}

/// `samples` (oldest first) judged by [`RULE`], recording `cpus` online
/// CPUs (default: the latest sample's count).
pub fn judge(samples: &[Value], cpus: Option<u64>) -> Value {
    let cpus = cpus
        .or_else(|| {
            samples
                .iter()
                .rev()
                .find_map(|row| row["cpus"].as_u64().filter(|&cpus| cpus > 0))
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
    if samples.len() < 2 || !samples.iter().all(valid) {
        reasons.push("fewer than two host load samples".to_owned());
    } else {
        // Recorded, not judged.
        peak = json!(
            samples
                .iter()
                .filter_map(|row| row["load_average"][0].as_f64())
                .fold(f64::NEG_INFINITY, f64::max)
        );
        let delta = |a: &Value, b: &Value, key: &str| Some(number(b, key)? - number(a, key)?);
        let intervals = samples
            .windows(2)
            .filter(|pair| {
                delta(&pair[0], &pair[1], "cpu_ticks").is_some_and(|ticks| ticks > 0.0)
                    && delta(&pair[0], &pair[1], "at").is_some_and(|seconds| seconds > 0.0)
            })
            .collect::<Vec<_>>();
        let fraction =
            |a: &Value, b: &Value| Some(delta(a, b, "cpu_idle_ticks")? / delta(a, b, "cpu_ticks")?);
        if intervals.is_empty() {
            reasons.push("no CPU time between host load samples".to_owned());
        } else {
            whole = json!(fraction(&samples[0], &samples[samples.len() - 1]));
            let low = intervals
                .iter()
                .filter_map(|pair| fraction(&pair[0], &pair[1]))
                .fold(f64::INFINITY, f64::min);
            lowest = json!(low);
            if low < IDLE_CPU_FRACTION {
                reasons.push(format!("CPU idle fell to {low:.3}"));
            }
            let stalled = |row: &Value, kind: &str| row["pressure_some_us"][kind].as_f64();
            if !samples.iter().all(|row| {
                PRESSURE_KINDS
                    .iter()
                    .all(|(kind, _)| stalled(row, kind).is_some())
            }) {
                reasons.push("no pressure stall information".to_owned());
            } else {
                for ((kind, name), share) in PRESSURE_KINDS.iter().zip(&mut shares) {
                    let high = intervals
                        .iter()
                        .filter_map(|pair| {
                            let stall = stalled(&pair[1], kind)? - stalled(&pair[0], kind)?;
                            Some(stall / 1e6 / delta(&pair[0], &pair[1], "at")?)
                        })
                        .fold(f64::NEG_INFINITY, f64::max);
                    *share = json!(high);
                    if high > IDLE_PRESSURE_SHARE {
                        reasons.push(format!("{name} pressure stall share reached {high:.3}"));
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
        json!({"at": at, "load_average": [1.0, 1.0, 1.0], "cpu_idle_ticks": idle, "cpu_ticks": total,
               "pressure_some_us": {"cpu": cpu, "io": io}})
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
        assert!(parse("", "cpu 1 2 3\n", None, 0.0).is_none());
        // Without pressure stall information the sample records null, and
        // its timing is informational.
        let bare = parse("1.25 0.90 0.50 2/900 1234\n", stat, None, 7.0).unwrap();
        assert!(bare["pressure_some_us"].is_null());
        let mut later = bare.clone();
        later["at"] = json!(8.0);
        later["cpu_ticks"] = json!(2000);
        later["cpu_idle_ticks"] = json!(1850);
        assert_eq!(
            reasons(&[bare, later]),
            json!(["no pressure stall information"])
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
        for rows in [
            vec![],
            vec![row(0, 0, 0.0, 0, 0)],
            vec![Value::Null, row(990, 1000, 1.0, 0, 0)],
            vec![row(5, 10, 0.0, 0, 0), row(5, 10, 1.0, 0, 0)],
            vec![row(0, 0, 1.0, 0, 0), row(990, 1000, 1.0, 0, 0)],
            vec![row(0, 0, 0.0, 0, 0), bare],
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
        let after_startup = judged("qwen38-2026-10-17");
        assert_eq!(
            (
                after_startup["timing"].clone(),
                after_startup["reasons"].clone(),
                after_startup["max_load_average_1m"].clone()
            ),
            (json!("evidence"), json!([]), json!(10.35))
        );
        assert_eq!(
            judged("mimo26-2026-10-16")["reasons"],
            json!(["no pressure stall information"])
        );
        assert_eq!(
            judged("qwen38-2026-10-16")["reasons"],
            json!(["CPU idle fell to 0.804", "no pressure stall information"])
        );
    }
}
