use runtime_eval::kv_offload::{facts, judge, judge_prefix};
use serde_json::{Value, json};

fn fixture(pressure: bool) -> Value {
    let observation = json!({"preemptions": 0, "swap_out": 0, "swap_in": 0,
        "exact_resumes": 0, "fallbacks": {}});
    let snapshot = json!({"preemptions": 0, "swap_out": 0, "swap_in": 0,
        "stalls": 0, "host_pages_used": 0, "swapped_requests": 0,
        "prefix_evictions": 0, "verified_items": 0, "fallbacks": {}});
    let rows: Vec<Value> = (0..36)
        .map(|index| {
            json!({
                "name": format!("request-{index}"), "prompt_sha256": format!("prompt-{index}"),
                "token_ids": vec![1; 32], "tokens": vec!["a"; 32], "logprobs": vec![-0.5; 32],
                "top": vec![json!({"a": -0.5, "b": -1.5}); 32], "finish_reason": "length",
                "usage": {"prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42},
                "observed": observation,
            })
        })
        .collect();
    let off = json!({"arm": "off", "scenario": if pressure {"pressure"} else {"plain"},
        "output_tokens": 96, "concurrency": 16, "c1": rows, "c16": rows, "repeat": rows,
        "warm": if pressure {rows.clone()} else {vec![]}, "before": snapshot,
        "c16_before": snapshot, "c16_after": snapshot, "after": snapshot});
    let mut swap = off.clone();
    swap["arm"] = json!("swap");
    swap["repeat"] = json!([]);
    for row in swap["c1"].as_array_mut().unwrap() {
        row["observed"] = json!({"preemptions": 2, "swap_out": 2, "swap_in": 2,
            "exact_resumes": 2, "fallbacks": {}});
    }
    for key in ["swap_out", "swap_in", "preemptions"] {
        swap["c16_before"][key] = json!(72);
        swap["c16_after"][key] = json!(144);
        swap["after"][key] = json!(144);
    }
    swap["c16_after"]["prefix_evictions"] = json!(3);
    swap["c16_after"]["verified_items"] = json!(30);
    json!({"off": off, "swap": swap, "speculative": pressure,
        "facts": {"budget_gib": 16, "swap_every": 16, "max_preemptions": 3,
                  "min_gated": 18, "pressure_tokens": 32768}})
}

fn fails(value: Value, gate: &str) {
    let result = judge(&value).unwrap();
    assert_eq!(result["passed"], false, "{result}");
    assert!(
        result["gates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == gate && row["passed"] == false),
        "{result}"
    );
}

#[test]
fn both_scenarios_pass_and_report_every_gate() {
    for pressure in [false, true] {
        let result = judge(&fixture(pressure)).unwrap();
        assert_eq!(result["passed"], true, "{result}");
        assert_eq!(result["gated_requests"].as_array().unwrap().len(), 36);
        assert_eq!(result["c16_equality"]["swap_identical"], 36);
    }
}

#[test]
fn reference_preemption_fails() {
    let mut value = fixture(false);
    value["off"]["c1"][0]["observed"]["preemptions"] = json!(1);
    fails(value, "off_activity");
}

#[test]
fn reference_copy_fails() {
    let mut value = fixture(false);
    value["off"]["after"]["swap_out"] = json!(1);
    fails(value, "off_swap_out");
}

#[test]
fn plain_fallback_fails() {
    let mut value = fixture(false);
    value["swap"]["c1"][0]["observed"]["fallbacks"] = json!({"capture": 1});
    fails(value, "c1_disallowed_fallbacks");
}

#[test]
fn pressure_capacity_fallback_is_listed_and_excluded() {
    let mut value = fixture(true);
    value["swap"]["c1"][0]["observed"]["fallbacks"] = json!({"capture": 1});
    value["swap"]["c1"][0]["observed"]["exact_resumes"] = json!(1);
    value["swap"]["c1"][0]["logprobs"][0] = json!(-0.75);
    let result = judge(&value).unwrap();
    assert_eq!(result["passed"], true);
    assert_eq!(result["recomputed_requests"], json!(["request-0"]));
    assert_eq!(result["gated_requests"].as_array().unwrap().len(), 35);
}

#[test]
fn pressure_noncapacity_fallback_fails() {
    let mut value = fixture(true);
    value["swap"]["c1"][0]["observed"]["fallbacks"] = json!({"restore": 1});
    fails(value, "c1_disallowed_fallbacks");
}

#[test]
fn every_preemption_must_resume_exactly() {
    let mut value = fixture(false);
    value["swap"]["c1"][0]["observed"]["exact_resumes"] = json!(1);
    fails(value, "c1_accounting_errors");
}

#[test]
fn logprob_bit_and_signed_zero_differences_fail() {
    for (left, right) in [
        (-0.5, f64::from_bits((-0.5_f64).to_bits() + 1)),
        (0.0, -0.0),
    ] {
        let mut value = fixture(false);
        value["off"]["c1"][0]["logprobs"][0] = json!(left);
        value["swap"]["c1"][0]["logprobs"][0] = json!(right);
        fails(value, "c1_bitwise_differences");
    }
}

#[test]
fn token_id_top_finish_and_usage_differences_fail() {
    for (key, replacement) in [
        ("token_ids", json!(vec![2; 32])),
        ("top", json!(vec![json!({"a": -0.25}); 32])),
        ("finish_reason", json!("stop")),
        (
            "usage",
            json!({"completion_tokens": 32, "prompt_tokens": 11, "total_tokens": 43}),
        ),
    ] {
        let mut value = fixture(false);
        value["swap"]["c1"][0][key] = replacement;
        fails(value, "c1_bitwise_differences");
    }
}

#[test]
fn no_swap_ins_fails() {
    let mut value = fixture(false);
    for row in value["swap"]["c1"].as_array_mut().unwrap() {
        row["observed"]["swap_in"] = json!(0);
    }
    fails(value, "c1_gated");
}

#[test]
fn too_few_gated_fails_even_with_allowed_fallbacks() {
    let mut value = fixture(true);
    for row in value["swap"]["c1"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .take(19)
    {
        row["observed"]["fallbacks"] = json!({"capture": 1});
    }
    fails(value, "c1_gated");
}

#[test]
fn request_past_swap_point_must_swap() {
    let mut value = fixture(false);
    value["swap"]["c1"][0]["observed"] = value["off"]["c1"][0]["observed"].clone();
    fails(value, "c1_missed_swaps");
}

#[test]
fn final_host_pages_swapped_requests_and_stalls_fail() {
    for key in ["host_pages_used", "swapped_requests", "stalls"] {
        let mut value = fixture(false);
        value["swap"]["after"][key] = json!(1);
        fails(value, &format!("swap_end_{key}"));
    }
}

#[test]
fn pressure_requires_evictions_and_speculation_when_declared() {
    for key in ["prefix_evictions", "verified_items"] {
        let mut value = fixture(true);
        value["swap"]["c16_after"][key] = json!(0);
        fails(value, &format!("c16_{key}"));
    }
    let mut value = fixture(true);
    value["speculative"] = json!(false);
    value["swap"]["c16_after"]["verified_items"] = json!(0);
    assert_eq!(judge(&value).unwrap()["passed"], true);
}

#[test]
fn c16_health_fails_but_inequality_is_informational() {
    let mut value = fixture(false);
    value["swap"]["c16"][0]["logprobs"][0] = json!(-0.75);
    assert_eq!(judge(&value).unwrap()["passed"], true);
    value["swap"]["c16"][0]["error"] = json!("failed");
    fails(value, "c16_errors");
}

#[test]
fn mismatched_or_duplicate_prompts_are_invalid() {
    for field in ["name", "prompt_sha256"] {
        let mut value = fixture(false);
        value["swap"]["c1"][0][field] = json!("different");
        assert!(judge(&value).unwrap_err().contains("prompts"));
    }
    let mut value = fixture(false);
    for arm in ["off", "swap"] {
        value[arm]["c1"][0]["name"] = json!("request-1");
    }
    assert!(judge(&value).is_err());
}

#[test]
fn missing_metrics_and_counter_resets_are_invalid() {
    let mut value = fixture(false);
    value["swap"]["after"]
        .as_object_mut()
        .unwrap()
        .remove("host_pages_used");
    assert!(judge(&value).is_err());
    let mut value = fixture(false);
    value["swap"]["c16_after"]["swap_in"] = json!(0);
    assert!(judge(&value).unwrap_err().contains("reset"));
}

#[test]
fn incomplete_capture_fails() {
    let mut value = fixture(false);
    value["swap"]["c1"][0]
        .as_object_mut()
        .unwrap()
        .remove("token_ids");
    fails(value, "c1_errors");
    let mut value = fixture(false);
    value["off"]["repeat"] = json!([]);
    assert!(judge(&value).is_err());
}

#[test]
fn facts_require_all_positive_typed_fields_and_bounded_coverage() {
    let valid = fixture(false)["facts"].clone();
    assert!(facts(&valid).is_ok());
    for key in [
        "budget_gib",
        "swap_every",
        "max_preemptions",
        "min_gated",
        "pressure_tokens",
    ] {
        for bad in [
            Value::Null,
            json!(0),
            json!(-1),
            json!(true),
            json!("16"),
            json!(1.5),
        ] {
            let mut value = valid.clone();
            value[key] = bad;
            assert!(facts(&value).is_err(), "{value}");
        }
        let mut value = valid.clone();
        value.as_object_mut().unwrap().remove(key);
        assert!(facts(&value).is_err());
    }
    let mut value = valid.clone();
    value["min_gated"] = json!(37);
    assert!(facts(&value).is_err());
    assert!(facts(&json!(false)).is_err());
    assert!(facts(&Value::Null).is_err());
}

/// The pressure off arm and a prefix-pressure tier arm whose C16 promoted
/// anchors from RAM and whose tier ended idle.
fn prefix_fixture() -> Value {
    let pressure = fixture(true);
    let mut off = pressure["off"].clone();
    off["passed"] = json!(true);
    off["fill"] =
        json!({"rows": [{"name": "fill-0", "prompt_sha256": "fill"}], "precondition": true});
    let mut tier = off.clone();
    tier["arm"] = json!("tier");
    tier["scenario"] = json!("prefix-pressure");
    tier["repeat"] = json!([]);
    let idle = json!({"ram_hits": 0, "ram_reused_tokens": 0, "headroom_waits": 0, "ram_evictions": 0,
        "demotions": {}, "promotions": {}, "missed_evictions": {}, "ram_fallbacks": {},
        "ram_anchors": 0, "ram_bytes": 0, "stalled_transfers": 0, "pending_micros": 0});
    let mut end = idle.clone();
    end["ram_hits"] = json!(30);
    end["ram_reused_tokens"] = json!(3000);
    end["demotions"] = json!({"ok": 40});
    end["promotions"] = json!({"ok": 30});
    end["missed_evictions"] = json!({"headroom": 4});
    end["ram_anchors"] = json!(40);
    end["ram_bytes"] = json!(1 << 30);
    for key in ["before", "c16_before"] {
        tier[key]["tier"] = idle.clone();
    }
    for key in ["c16_after", "after"] {
        tier[key]["tier"] = end.clone();
    }
    tier["c16_after"]["prefix_evictions"] = json!(3);
    tier["c16_after"]["verified_items"] = json!(30);
    tier["after"]["host_pages_used"] = json!(4096);
    json!({"off": off, "tier": tier, "speculative": true, "facts": pressure["facts"]})
}

fn prefix_fails(value: Value, gate: &str) {
    let result = judge_prefix(&value).unwrap();
    assert_eq!(result["passed"], false, "{result}");
    assert!(
        result["gates"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["name"] == gate && row["passed"] == false),
        "{result}"
    );
}

#[test]
fn prefix_pressure_passes_and_reports_the_tier() {
    let result = judge_prefix(&prefix_fixture()).unwrap();
    assert_eq!(result["passed"], true, "{result}");
    assert_eq!(result["scenario"], "prefix-pressure");
    assert_eq!(result["ram"]["promotions"], json!({"ok": 30}));
    assert_eq!(result["ram"]["missed_evictions"], json!({"headroom": 4}));
    assert_eq!(result["ram"]["end_host_pages_used"], 4096);
    assert_eq!(result["c16_equality"]["tier_identical"], 36);
}

#[test]
fn prefix_pressure_requires_every_c1_request_bitwise_equal() {
    let mut value = prefix_fixture();
    value["tier"]["c1"][35]["logprobs"][0] = json!(-0.5000000000000001);
    prefix_fails(value.clone(), "c1_bitwise_differences");
    assert_eq!(
        judge_prefix(&value).unwrap()["different_requests"],
        json!(["request-35"])
    );
    let mut value = prefix_fixture();
    value["tier"]["c1"][0]["error"] = json!("failed");
    prefix_fails(value, "c1_errors");
    let mut value = prefix_fixture();
    value["off"]["c1"][0]["observed"]["preemptions"] = json!(1);
    prefix_fails(value, "off_activity");
}

#[test]
fn prefix_pressure_requires_promotions_and_ram_hits() {
    let mut value = prefix_fixture();
    value["tier"]["after"]["tier"]["promotions"] = json!({});
    prefix_fails(value, "tier_promotions");
    let mut value = prefix_fixture();
    value["tier"]["after"]["tier"]["ram_hits"] = json!(0);
    prefix_fails(value, "tier_ram_hits");
    let mut value = prefix_fixture();
    value["tier"]["c16_after"]["prefix_evictions"] = json!(0);
    prefix_fails(value, "c16_prefix_evictions");
    let mut value = prefix_fixture();
    value["tier"]["c16_after"]["verified_items"] = json!(0);
    prefix_fails(value.clone(), "c16_verified_items");
    value["speculative"] = json!(false);
    assert_eq!(judge_prefix(&value).unwrap()["passed"], true);
}

#[test]
fn prefix_pressure_requires_an_idle_tier_and_no_stalls_at_the_end() {
    for key in ["pending_micros", "stalled_transfers"] {
        let mut value = prefix_fixture();
        value["tier"]["after"]["tier"][key] = json!(1);
        prefix_fails(value, &format!("tier_end_{key}"));
    }
    for key in ["swapped_requests", "stalls"] {
        let mut value = prefix_fixture();
        value["tier"]["after"][key] = json!(1);
        prefix_fails(value, &format!("tier_end_{key}"));
    }
    let mut value = prefix_fixture();
    value["tier"]["after"]["fallbacks"] = json!({"restore": 1});
    prefix_fails(value.clone(), "tier_disallowed_fallbacks");
    value["tier"]["after"]["fallbacks"] = json!({"tier_full": 1});
    assert_eq!(judge_prefix(&value).unwrap()["passed"], true);
}

#[test]
fn prefix_pressure_refuses_missing_tier_evidence_and_other_arms() {
    for key in [
        "missed_evictions",
        "promotions",
        "ram_hits",
        "pending_micros",
    ] {
        let mut value = prefix_fixture();
        value["tier"]["after"]["tier"]
            .as_object_mut()
            .unwrap()
            .remove(key);
        assert!(judge_prefix(&value).is_err(), "{key}");
    }
    let mut value = prefix_fixture();
    value["tier"]["after"]["tier"]["missed_evictions"] = json!({"headroom": 2});
    value["tier"]["before"]["tier"]["missed_evictions"] = json!({"headroom": 3});
    assert!(judge_prefix(&value).unwrap_err().contains("reset"));
    let pressure = fixture(true);
    for (arm, record) in [
        ("off", pressure["swap"].clone()),
        ("tier", pressure["off"].clone()),
    ] {
        let mut value = prefix_fixture();
        value[arm] = record;
        assert!(
            judge_prefix(&value).unwrap_err().contains("incompatible"),
            "{arm}"
        );
    }
    let mut value = prefix_fixture();
    value["tier"]["c16"] = json!([]);
    assert!(judge_prefix(&value).is_err());
}

#[test]
fn prefix_pressure_keeps_the_off_arms_end_gates_and_control() {
    for key in ["host_pages_used", "swapped_requests", "stalls"] {
        let mut value = prefix_fixture();
        value["off"]["after"][key] = json!(1);
        prefix_fails(value, &format!("off_end_{key}"));
    }
    for key in ["preemptions", "swap_out", "swap_in"] {
        let mut value = prefix_fixture();
        value["off"]["after"][key] = json!(1);
        prefix_fails(value, &format!("off_{key}"));
    }
    let mut value = prefix_fixture();
    value["off"]["after"]["fallbacks"] = json!({"restore": 1});
    prefix_fails(value, "off_disallowed_fallbacks");
    let mut value = prefix_fixture();
    value["off"]["repeat"][3]["error"] = json!("failed");
    prefix_fails(value, "c16_errors");
    let mut value = prefix_fixture();
    value["off"]["repeat"] = json!([]);
    assert!(judge_prefix(&value).unwrap_err().contains("repeat"));
}

#[test]
fn prefix_pressure_rejects_failed_fill_and_capture_evidence() {
    for arm in ["off", "tier"] {
        let mut value = prefix_fixture();
        value[arm]["fill"]["rows"][0]["error"] = json!("failed");
        value[arm]["passed"] = json!(false);
        let result = judge_prefix(&value).unwrap();
        assert_eq!(result["passed"], false, "{result}");
        prefix_fails(value.clone(), "fill_errors");
        prefix_fails(value, "failed_records");
    }
    let mut value = prefix_fixture();
    value["tier"].as_object_mut().unwrap().remove("passed");
    prefix_fails(value, "failed_records");
    let mut value = prefix_fixture();
    value["tier"]["fill"]["precondition"] = json!(false);
    prefix_fails(value, "tier_fill_precondition");
    for key in ["rows", "precondition"] {
        let mut value = prefix_fixture();
        value["tier"]["fill"].as_object_mut().unwrap().remove(key);
        assert!(judge_prefix(&value).is_err(), "{key}");
    }
}
