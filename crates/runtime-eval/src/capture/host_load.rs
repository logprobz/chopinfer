//! The host load a receipt records (`common.host_load`). It labels the
//! receipt's throughput and latency figures and never changes a verdict.
//!
//! The host is idle for timing evidence when, at every sample, the 1-minute
//! load average is at most [`IDLE_LOAD_PER_CPU`] per online CPU and, over
//! every interval between samples, the CPU idle fraction (idle plus iowait
//! over all CPU time, so niced work counts as busy) is at least
//! [`IDLE_CPU_FRACTION`]. Otherwise, or without two samples, the figures are
//! informational. tools/qual/common.py states why and applies the same rule.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

pub const IDLE_LOAD_PER_CPU: f64 = 0.1;
pub const IDLE_CPU_FRACTION: f64 = 0.9;
pub const RULE: &str = "idle when every 1-minute load average is at most 0.1 per online CPU and every interval's CPU idle fraction is at least 0.9; otherwise throughput and latency are informational";

/// This host's load averages, cumulative idle and total CPU ticks and
/// online CPUs (`common.host_load_sample`), or null without /proc.
pub fn sample() -> Value {
    let read = |path| std::fs::read_to_string(path).ok();
    read("/proc/loadavg")
        .zip(read("/proc/stat"))
        .and_then(|(loadavg, stat)| parse(&loadavg, &stat, now()))
        .unwrap_or(Value::Null)
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

fn parse(loadavg: &str, stat: &str, at: f64) -> Option<Value> {
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
    Some(json!({
        "at": at,
        "load_average": load,
        "cpu_idle_ticks": ticks[3] + ticks[4],
        "cpu_ticks": ticks.iter().sum::<u64>(),
        "cpus": cpus,
    }))
}

fn number(row: &Value, key: &str) -> Option<f64> {
    row[key].as_f64()
}

fn valid(row: &Value) -> bool {
    row["load_average"][0].is_number()
        && number(row, "cpu_idle_ticks").is_some()
        && number(row, "cpu_ticks").is_some()
}

/// `samples` (oldest first) judged by [`RULE`] on `cpus` online CPUs
/// (default: the latest sample's count).
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
    let mut reasons = Vec::new();
    if samples.len() < 2 || !samples.iter().all(valid) {
        reasons.push("fewer than two host load samples".to_owned());
    } else {
        let high = samples
            .iter()
            .filter_map(|row| row["load_average"][0].as_f64())
            .fold(f64::NEG_INFINITY, f64::max);
        peak = json!(high);
        if high > IDLE_LOAD_PER_CPU * cpus as f64 {
            reasons.push(format!(
                "1-minute load average reached {high:.2} on {cpus} CPUs"
            ));
        }
        let fraction = |a: &Value, b: &Value| {
            let total = number(b, "cpu_ticks")? - number(a, "cpu_ticks")?;
            let idle = number(b, "cpu_idle_ticks")? - number(a, "cpu_idle_ticks")?;
            (total > 0.0).then_some(idle / total)
        };
        let idle = samples
            .windows(2)
            .filter_map(|pair| fraction(&pair[0], &pair[1]))
            .collect::<Vec<_>>();
        if idle.is_empty() {
            reasons.push("no CPU time between host load samples".to_owned());
        } else {
            whole = json!(fraction(&samples[0], &samples[samples.len() - 1]));
            let low = idle.iter().copied().fold(f64::INFINITY, f64::min);
            lowest = json!(low);
            if low < IDLE_CPU_FRACTION {
                reasons.push(format!("CPU idle fell to {low:.3}"));
            }
        }
    }
    json!({
        "rule": RULE,
        "scope": "check_host",
        "cpus": cpus,
        "samples": samples,
        "max_load_average_1m": peak,
        "cpu_idle_fraction": whole,
        "min_cpu_idle_fraction": lowest,
        "idle": reasons.is_empty(),
        "timing": if reasons.is_empty() { "evidence" } else { "informational" },
        "reasons": reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(load: f64, idle: u64, total: u64) -> Value {
        json!({"at": 0.0, "load_average": [load, load, load], "cpu_idle_ticks": idle, "cpu_ticks": total})
    }

    #[test]
    fn proc_is_parsed() {
        let value = parse(
            "1.25 0.90 0.50 2/900 1234\n",
            "cpu  100 20 30 800 50 0 0 0 0 0\ncpu0 1 2 3 4 5 0 0 0 0 0\n",
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
        assert!(parse("", "cpu 1 2 3\n", 0.0).is_none());
    }

    #[test]
    fn an_idle_host_keeps_timing_as_evidence() {
        let verdict = judge(&[row(1.2, 0, 0), row(1.3, 970, 1000)], Some(48));
        assert_eq!(verdict["timing"], "evidence");
        assert_eq!(verdict["reasons"], json!([]));
        assert!((verdict["min_cpu_idle_fraction"].as_f64().unwrap() - 0.97).abs() < 1e-12);
        // The bounds are inclusive.
        assert_eq!(
            judge(&[row(4.8, 0, 0), row(4.8, 900, 1000)], Some(48))["idle"],
            true
        );
    }

    #[test]
    fn a_busy_or_unknown_host_makes_timing_informational() {
        let verdict = judge(&[row(61.0, 0, 0), row(75.0, 80, 1000)], Some(48));
        assert_eq!(verdict["timing"], "informational");
        assert_eq!(
            verdict["reasons"],
            json!([
                "1-minute load average reached 75.00 on 48 CPUs",
                "CPU idle fell to 0.080"
            ])
        );
        let rows = [
            row(1.0, 0, 0),
            row(1.0, 990, 1000),
            row(1.0, 1500, 2000),
            row(1.0, 2490, 3000),
        ];
        let verdict = judge(&rows, Some(48));
        assert_eq!(verdict["idle"], false);
        assert!((verdict["cpu_idle_fraction"].as_f64().unwrap() - 0.83).abs() < 1e-12);
        for rows in [
            vec![],
            vec![row(1.0, 0, 0)],
            vec![Value::Null, row(1.0, 990, 1000)],
            vec![row(1.0, 5, 10), row(1.0, 5, 10)],
        ] {
            assert_eq!(judge(&rows, Some(48))["timing"], "informational");
        }
    }
}
