//! Paired preemption swap qualification. Capture lives in tools/qual;
//! this module owns the gates and declaration facts.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::limits::bit_equal;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Facts {
    pub budget_gib: u64,
    pub swap_every: u64,
    pub max_preemptions: u64,
    pub min_gated: u64,
    pub pressure_tokens: u64,
}

pub fn facts(value: &Value) -> Result<Facts, String> {
    let facts: Facts =
        serde_json::from_value(value.clone()).map_err(|error| format!("kv_offload: {error}"))?;
    if facts.budget_gib == 0
        || facts.swap_every == 0
        || facts.max_preemptions == 0
        || !(1..=36).contains(&facts.min_gated)
        || facts.pressure_tokens == 0
    {
        return Err("kv_offload: positive facts required, min_gated must be 1..36".into());
    }
    Ok(facts)
}

fn count(value: &Value, key: &str) -> Result<u64, String> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("kv_offload.{key}: expected a nonnegative integer"))
}

fn rows<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], String> {
    value[key]
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("kv_offload.{key}: expected request rows"))
}

fn names(rows: &[Value]) -> Result<Vec<(&str, &str)>, String> {
    rows.iter()
        .map(|row| {
            let name = row["name"].as_str().filter(|s| !s.is_empty());
            let prompt = row["prompt_sha256"].as_str().filter(|s| !s.is_empty());
            name.zip(prompt)
                .ok_or("kv_offload: missing request identity".into())
        })
        .collect()
}

fn valid_row(row: &Value) -> bool {
    if row.get("error").is_some() {
        return false;
    }
    let Some(tokens) = row["tokens"].as_array() else {
        return false;
    };
    let Some(ids) = row["token_ids"].as_array() else {
        return false;
    };
    let Some(scores) = row["logprobs"].as_array() else {
        return false;
    };
    let Some(top) = row["top"].as_array() else {
        return false;
    };
    tokens.iter().all(Value::is_string)
        && ids.iter().all(|v| v.as_u64().is_some())
        && scores.len() == tokens.len()
        && top.len() == tokens.len()
        && scores
            .iter()
            .all(|v| v.as_f64().is_some_and(f64::is_finite))
        && top.iter().all(|v| {
            v.as_object().is_some_and(|v| {
                !v.is_empty() && v.values().all(|v| v.as_f64().is_some_and(f64::is_finite))
            })
        })
        && matches!(row["finish_reason"].as_str(), Some("stop" | "length"))
        && row["usage"]["completion_tokens"].as_u64() == Some(ids.len() as u64)
        && row["usage"]["prompt_tokens"].as_u64().is_some()
        && row["usage"]["total_tokens"].as_u64().is_some()
}

fn equal(left: &Value, right: &Value) -> bool {
    [
        "token_ids",
        "tokens",
        "logprobs",
        "top",
        "finish_reason",
        "usage",
    ]
    .iter()
    .all(|field| bit_equal(&left[field], &right[field]))
}

fn fallbacks(value: &Value, pressure: bool) -> Result<(u64, u64), String> {
    let map = value["fallbacks"]
        .as_object()
        .ok_or("kv_offload: missing fallbacks")?;
    let mut all = 0;
    let mut forbidden = 0;
    for (reason, value) in map {
        let n = value.as_u64().ok_or("kv_offload: invalid fallback count")?;
        all += n;
        if !pressure
            || !matches!(
                reason.as_str(),
                "tier_full" | "capture" | "staging_capacity"
            )
        {
            forbidden += n;
        }
    }
    Ok((all, forbidden))
}

fn difference(before: &Value, after: &Value, key: &str) -> Result<u64, String> {
    count(after, key)?
        .checked_sub(count(before, key)?)
        .ok_or_else(|| format!("kv_offload: {key} counter reset"))
}

fn fallback_delta(before: &Value, after: &Value) -> Result<Value, String> {
    let start = before["fallbacks"]
        .as_object()
        .ok_or("kv_offload: missing fallbacks")?;
    let end = after["fallbacks"]
        .as_object()
        .ok_or("kv_offload: missing fallbacks")?;
    let keys: BTreeSet<_> = start.keys().chain(end.keys()).collect();
    let mut delta = BTreeMap::new();
    for key in keys {
        let a = start
            .get(key)
            .unwrap_or(&Value::from(0))
            .as_u64()
            .ok_or("invalid fallback count")?;
        let b = end
            .get(key)
            .unwrap_or(&Value::from(0))
            .as_u64()
            .ok_or("invalid fallback count")?;
        delta.insert(
            key,
            b.checked_sub(a)
                .ok_or("kv_offload: fallback counter reset")?,
        );
    }
    Ok(json!({"fallbacks": delta}))
}

fn gate(gates: &mut Vec<Value>, name: &str, value: u64, op: &str, limit: u64) {
    let passed = if op == "ge" {
        value >= limit
    } else {
        value == limit
    };
    gates.push(json!({"name": name, "value": value, "op": op, "limit": limit, "passed": passed}));
}

/// Return every gate's measured value and limit, plus informational C16 equality.
pub fn judge(args: &Value) -> Result<Value, String> {
    let policy = facts(&args["facts"])?;
    let off = &args["off"];
    let swap = &args["swap"];
    let pressure = match off["scenario"].as_str() {
        Some("plain") => false,
        Some("pressure") => true,
        _ => return Err("kv_offload: unknown scenario".into()),
    };
    if off["scenario"] != swap["scenario"]
        || off["arm"] != "off"
        || swap["arm"] != "swap"
        || count(off, "output_tokens")? != count(swap, "output_tokens")?
        || count(off, "output_tokens")? <= policy.swap_every
        || count(off, "concurrency")? != 16
        || count(swap, "concurrency")? != 16
    {
        return Err("kv_offload: incompatible arms or workload".into());
    }
    let speculative = args["speculative"]
        .as_bool()
        .ok_or("kv_offload: missing speculative fact")?;
    let mut gates = Vec::new();
    let mut reference_activity = 0;
    let mut forbidden = 0;
    let mut accounting = 0;
    let mut missed = 0;
    let mut errors = 0;
    let mut copies = 0;
    let mut gated = Vec::new();
    let mut recomputed = Vec::new();
    let mut different = Vec::new();
    let left = rows(off, "c1")?;
    let right = rows(swap, "c1")?;
    let identities = names(left)?;
    let unique: BTreeSet<_> = identities.iter().map(|(name, _)| name).collect();
    if identities != names(right)? || left.len() != 36 || unique.len() != 36 {
        return Err("kv_offload: missing, duplicate or mismatched C1 prompts".into());
    }
    for (a, b) in left.iter().zip(right) {
        errors += u64::from(!valid_row(a) || !valid_row(b));
        let observed = &b["observed"];
        for key in ["preemptions", "swap_out", "swap_in", "exact_resumes"] {
            reference_activity += count(&a["observed"], key)?;
        }
        reference_activity += fallbacks(&a["observed"], false)?.0;
        let preemptions = count(observed, "preemptions")?;
        let out = count(observed, "swap_out")?;
        let input = count(observed, "swap_in")?;
        let exact = count(observed, "exact_resumes")?;
        let (fallback, disallowed) = fallbacks(observed, pressure)?;
        forbidden += disallowed;
        copies += input;
        let same = preemptions == out && out == input && input == exact;
        let capacity = pressure && fallback > 0 && disallowed == 0;
        accounting += u64::from(preemptions > policy.max_preemptions || (!same && !capacity));
        if count(&a["usage"], "completion_tokens")? > policy.swap_every && preemptions == 0 {
            missed += 1;
        }
        if capacity {
            recomputed.push(b["name"].clone());
        } else if input > 0 && same && fallback == 0 {
            gated.push(b["name"].clone());
            if !equal(a, b) {
                different.push(b["name"].clone());
            }
        }
    }
    gate(&mut gates, "c1_requests", left.len() as u64, "eq", 36);
    gate(&mut gates, "c1_errors", errors, "eq", 0);
    gate(&mut gates, "off_activity", reference_activity, "eq", 0);
    gate(&mut gates, "c1_disallowed_fallbacks", forbidden, "eq", 0);
    gate(&mut gates, "c1_accounting_errors", accounting, "eq", 0);
    gate(&mut gates, "c1_missed_swaps", missed, "eq", 0);
    gate(
        &mut gates,
        "c1_gated",
        gated.len() as u64,
        "ge",
        policy.min_gated,
    );
    gate(
        &mut gates,
        "c1_bitwise_differences",
        different.len() as u64,
        "eq",
        0,
    );
    gate(&mut gates, "c1_swap_ins", copies, "ge", gated.len() as u64);
    // Check the measured cohort and every warm/control request for completion.
    let c16_off = rows(off, "c16")?;
    let c16_swap = rows(swap, "c16")?;
    let control = rows(off, "repeat")?;
    let mut c16_errors = 0;
    for record in [off, swap] {
        for key in ["c16", "warm", "repeat"] {
            let cohort = rows(record, key)?;
            let required = key == "c16"
                || (key == "warm" && pressure)
                || (key == "repeat" && record["arm"] == "off");
            if required && (cohort.len() != 36 || names(cohort)? != identities) {
                return Err(format!("kv_offload: missing or mismatched {key} requests"));
            }
            c16_errors += cohort.iter().filter(|row| !valid_row(row)).count() as u64;
        }
        let before = &record["before"];
        let after = &record["after"];
        // Absolute final counters catch stalls even in warmup or the C1 phase.
        for key in ["host_pages_used", "swapped_requests", "stalls"] {
            gate(
                &mut gates,
                &format!("{}_end_{key}", record["arm"].as_str().unwrap()),
                count(after, key)?,
                "eq",
                0,
            );
        }
        let disallowed = fallbacks(&fallback_delta(before, after)?, pressure)?.1;
        gate(
            &mut gates,
            &format!("{}_disallowed_fallbacks", record["arm"].as_str().unwrap()),
            disallowed,
            "eq",
            0,
        );
        if record["arm"] == "off" {
            for key in ["preemptions", "swap_out", "swap_in"] {
                gate(
                    &mut gates,
                    &format!("off_{key}"),
                    difference(before, after, key)?,
                    "eq",
                    0,
                );
            }
        }
    }
    gate(&mut gates, "c16_errors", c16_errors, "eq", 0);
    for key in ["swap_out", "swap_in"] {
        gate(
            &mut gates,
            &format!("c16_{key}"),
            difference(&swap["c16_before"], &swap["c16_after"], key)?,
            "ge",
            1,
        );
    }
    if pressure {
        gate(
            &mut gates,
            "c16_prefix_evictions",
            difference(&swap["c16_before"], &swap["c16_after"], "prefix_evictions")?,
            "ge",
            1,
        );
        if speculative {
            gate(
                &mut gates,
                "c16_verified_items",
                difference(&swap["c16_before"], &swap["c16_after"], "verified_items")?,
                "ge",
                1,
            );
        }
    }
    let identical = |other: &[Value]| {
        c16_off
            .iter()
            .zip(other)
            .filter(|(a, b)| equal(a, b))
            .count()
    };
    Ok(
        json!({"passed": gates.iter().all(|g| g["passed"] == true), "count": 36,
        "scenario": off["scenario"], "gates": gates, "gated_requests": gated,
        "recomputed_requests": recomputed, "different_requests": different,
        "c16_equality": {"swap_identical": identical(c16_swap), "control_identical": identical(control),
                         "requests": 36, "informational": true}}),
    )
}
