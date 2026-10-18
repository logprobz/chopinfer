//! The declaration loader and the configuration expansion: the one
//! implementation of the declaration rules.

use std::path::{Path, PathBuf};

use runtime_eval::declaration::{self, Input, Modules};
use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn vectors() -> Value {
    let bytes = std::fs::read(root().join("crates/runtime-eval/tests/data/declaration-cases.json"))
        .unwrap();
    let vectors = runtime_eval::json::parse(&bytes).unwrap();
    assert_eq!(vectors["schema"], "chopinfer-qual-declaration-vectors/v1");
    vectors
}

fn modules() -> Modules {
    Modules::read(&root()).unwrap()
}

/// The vector's base with its values set at JSON pointers, then its
/// pointers removed.
fn mutated(base: &Value, case: &Value) -> Value {
    let mut value = base.clone();
    let split = |pointer: &str| {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        (parent.to_owned(), key.to_owned())
    };
    for (pointer, item) in case["set"].as_object().unwrap() {
        let (parent, key) = split(pointer);
        match value.pointer_mut(&parent).unwrap() {
            Value::Array(items) => items[key.parse::<usize>().unwrap()] = item.clone(),
            Value::Object(map) => {
                map.insert(key, item.clone());
            }
            other => panic!("{pointer}: {other}"),
        }
    }
    for pointer in case["remove"].as_array().unwrap() {
        let (parent, key) = split(pointer.as_str().unwrap());
        match value.pointer_mut(&parent).unwrap() {
            Value::Array(items) => {
                items.remove(key.parse::<usize>().unwrap());
            }
            Value::Object(map) => {
                map.shift_remove(&key).unwrap();
            }
            other => panic!("{pointer}: {other}"),
        }
    }
    value
}

#[test]
fn each_shared_mutation_is_refused_with_its_field() {
    let vectors = vectors();
    let modules = modules();
    let cases = vectors["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let value = mutated(&vectors["base"], case);
        let result = declaration::validate(&value, &modules);
        let name = &case["name"];
        match (&case["error"], result) {
            (Value::Null, Ok(_)) => {}
            (Value::String(field), Err(error)) => {
                assert!(
                    error.to_string().starts_with(&format!("{field}: ")),
                    "{name}: {error}"
                );
            }
            (expected, result) => panic!("{name}: expected {expected}, got {result:?}"),
        }
    }
}

#[test]
fn the_base_declaration_reads_its_fields() {
    let declaration = declaration::validate(&vectors()["base"], &modules()).unwrap();
    assert_eq!(declaration.model, "example");
    let checks: Vec<_> = declaration
        .checks
        .iter()
        .map(|check| check.id.as_str())
        .collect();
    assert_eq!(checks, ["teacher", "api", "media"]);
    assert_eq!(declaration.cohort("quick").unwrap(), ["teacher", "api"]);
    assert_eq!(declaration.check("teacher").unwrap().rules.len(), 2);
    assert_eq!(
        declaration
            .check("media")
            .unwrap()
            .informational_reason
            .as_deref(),
        Some("no media fixtures yet")
    );
    let teacher = declaration.reference("teacher").unwrap();
    assert_eq!(teacher.variants.as_deref().unwrap(), ["exact", "bf16"]);
    assert_eq!(teacher.checkpoint_keys.as_deref().unwrap(), ["revision"]);
    assert_eq!(declaration.input("unused"), Some(&Input::Unused));
    assert_eq!(
        declaration.input("data"),
        Some(&Input::Environment {
            env: "EXAMPLE_DATA".into(),
            default: Some("fixtures/example".into()),
            path: "data".into()
        })
    );
    assert_eq!(declaration.checkpoint.served_model_name, "Example");
}

#[test]
fn optional_sections_may_be_absent() {
    let mut value = vectors()["base"].clone();
    let map = value.as_object_mut().unwrap();
    for key in ["references", "checkpoint", "limitations", "preview"] {
        map.shift_remove(key);
    }
    map.get_mut("checks")
        .unwrap()
        .as_object_mut()
        .unwrap()
        .retain(|id, _| id == "api");
    map.insert("cohorts".into(), json!({"quick": {"checks": ["api"]}}));
    map.insert(
        "change_classes".into(),
        json!({"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []}),
    );
    let qualification = map
        .get_mut("qualification")
        .unwrap()
        .as_object_mut()
        .unwrap();
    qualification.shift_remove("prefix");
    qualification.shift_remove("needle");
    let declaration = declaration::validate(&value, &modules()).unwrap();
    assert!(declaration.references.is_empty());
}

#[test]
fn module_metadata_is_required_and_checked() {
    let scratch = std::env::temp_dir().join(format!("runtime-eval-modules-{}", std::process::id()));
    std::fs::create_dir_all(scratch.join("tools/qual")).unwrap();
    assert!(Modules::read(&scratch).is_err());
    std::fs::write(
        scratch.join("tools/qual/modules.json"),
        b"{\"schema\": \"other\", \"modules\": {}}",
    )
    .unwrap();
    assert!(Modules::read(&scratch).is_err());
    std::fs::remove_dir_all(scratch).unwrap();
}

#[test]
fn every_shipped_declaration_validates() {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut count = 0;
    for entry in std::fs::read_dir(&models).unwrap() {
        let directory = entry.unwrap().path();
        if !directory.join(declaration::FILE_NAME).is_file() {
            continue;
        }
        let variants = declaration::variants(&directory).unwrap();
        for variant in std::iter::once(None).chain(variants.iter().map(|v| Some(v.as_str()))) {
            let loaded = declaration::load_variant(&directory, variant).unwrap();
            assert_eq!(
                loaded.root,
                std::fs::canonicalize(models.join("..")).unwrap()
            );
            assert_eq!(loaded.declaration.variant.as_deref(), variant);
            assert!(!loaded.declaration.checks.is_empty());
            count += 1;
        }
    }
    assert!(count > 0, "no model declares its qualification");
}

fn configured_fixture() -> Value {
    let mut base = vectors()["base"].clone();
    base["references"]["teacher"]["sha256"] = json!("d".repeat(64));
    let rule = |path: Value, value: Value| json!({"path": path, "op": "eq", "value": value, "source": "test"});
    let kv = rule(
        json!(["server", "health", "serving", "profile", "kv_cache_dtype"]),
        json!("fp8"),
    );
    let identity = base["qualification"]["checkpoint"]["checkpoint_identity"].clone();
    base["checks"]["teacher"]["rules"]
        .as_array_mut()
        .unwrap()
        .extend([
            kv.clone(),
            rule(
                json!(["server", "checkpoint", "checkpoint_identity"]),
                identity,
            ),
            rule(
                json!(["server", "health", "serving", "profile", "name"]),
                json!("tp4-mtp"),
            ),
        ]);
    base["checks"]["teacher_arms"] = base["checks"]["teacher"].clone();
    base["checks"]["teacher_arms"]["input_rules"] = json!({"fp8_teacher": [kv.clone()]});
    base["checks"]["prefix_exact"] = base["checks"]["teacher"].clone();
    base["checks"]["prefix_exact"]["input_rules"] = json!({"prefix_record": [kv]});
    base["checks"]["paired_teacher"] = json!({"module": "paired_teacher", "args": [],
        "timeout_seconds": 60, "receipt": "*-paired.json", "rules": [{"path": ["passed"], "op": "eq", "value": true, "source": "test"}], "constants": {"paired_teacher": {"PROFILE_POLICIES": {
        "source": "test", "value": {"tp4-mtp": {"reference_sha256": base["references"]["teacher"]["sha256"]}}}}}});
    let mut reference = base["references"]["teacher"].clone();
    reference["sha256"] = json!("f".repeat(64));
    base["configurations"] = json!({
        "kv-bf16": {"checks": ["teacher_arms", "prefix_exact"], "quick": ["teacher_arms"],
                    "serving": {"kv_cache_dtype": "bf16"}, "from_default": ["fp8_teacher"]},
        "requant": {"checks": ["teacher", "paired_teacher"], "quick": ["teacher"],
                  "serving": {"checkpoint_identity": "e".repeat(64), "profile_name": "tp4-mtp-requant"},
                  "references": {"teacher": reference}},
        "dcp2": {"checks": ["teacher", "api", "media"], "quick": ["teacher", "api"],
                 "serving": {"context_parallel_size": 2}}
    });
    base
}

const PROFILE: [&str; 4] = ["server", "health", "serving", "profile"];

/// Every rule of a check, with the input it belongs to (None: the check's own).
fn rules(check: &Value) -> Vec<(Option<&str>, &Value)> {
    let mut all: Vec<_> = check["rules"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|rule| (None, rule))
        .collect();
    for (input, rules) in check["input_rules"].as_object().into_iter().flatten() {
        all.extend(
            rules
                .as_array()
                .into_iter()
                .flatten()
                .map(|rule| (Some(input.as_str()), rule)),
        );
    }
    all
}

#[test]
fn a_configuration_expands_its_serving_facts_onto_the_base() {
    let base = configured_fixture();
    let spec = &base["configurations"]["kv-bf16"];
    let expanded = declaration::expand(&base, "kv-bf16").unwrap();
    assert_eq!(expanded["variant"], "kv-bf16");
    assert!(expanded.get("configurations").is_none());
    assert_eq!(expanded["cohorts"]["full"]["checks"], spec["checks"]);
    assert_eq!(expanded["cohorts"]["quick"]["checks"], spec["quick"]);
    let kv = |input: Option<&str>| {
        rules(&expanded["checks"]["teacher_arms"])
            .into_iter()
            .chain(rules(&expanded["checks"]["prefix_exact"]))
            .filter(|(at, _)| {
                *at == input
                    || input.is_none() && at.is_some_and(|at| at.starts_with("prefix_record"))
            })
            .filter(|(_, rule)| {
                rule["path"]
                    == json!([
                        PROFILE[0],
                        PROFILE[1],
                        PROFILE[2],
                        PROFILE[3],
                        "kv_cache_dtype"
                    ])
            })
            .map(|(_, rule)| rule["value"].clone())
            .collect::<Vec<_>>()
    };
    // The configuration's records hold its own KV dtype; an input the
    // default configuration records (from_default) keeps the default's.
    assert!(!kv(None).is_empty() && kv(None).iter().all(|value| value == "bf16"));
    assert!(
        !kv(Some("fp8_teacher")).is_empty()
            && kv(Some("fp8_teacher")).iter().all(|value| value == "fp8")
    );
    // A checkpoint configuration names its own checkpoint wherever a rule
    // pinned the base's, and its profile.
    let requant = declaration::expand(&base, "requant").unwrap();
    let identity = &base["configurations"]["requant"]["serving"]["checkpoint_identity"];
    let old = &base["qualification"]["checkpoint"]["checkpoint_identity"];
    let teacher = rules(&requant["checks"]["teacher"]);
    assert!(teacher.iter().any(|(_, rule)| rule["value"] == *identity));
    assert!(!teacher.iter().any(|(_, rule)| rule["value"] == *old));
    assert!(teacher.iter().any(|(_, rule)| rule["path"]
        == json!([PROFILE[0], PROFILE[1], PROFILE[2], PROFILE[3], "name"])
        && rule["value"] == "tp4-mtp-requant"));
    // The paired margins name the reference they apply to: its own.
    let pin = &requant["references"]["teacher"]["sha256"];
    assert_ne!(pin, &base["references"]["teacher"]["sha256"]);
    let margins = &requant["checks"]["paired_teacher"]["constants"]["paired_teacher"]["PROFILE_POLICIES"]
        ["value"];
    assert!(!margins.as_object().unwrap().is_empty());
    for policy in margins.as_object().unwrap().values() {
        assert_eq!(&policy["reference_sha256"], pin);
    }
}

#[test]
fn an_external_variant_repins_as_a_checkpoint_configuration_does() {
    let mut base = configured_fixture();
    let configured = base["configurations"]["requant"].clone();
    base.as_object_mut().unwrap().shift_remove("configurations");
    let mut reference = configured["references"]["teacher"].clone();
    reference["path"] = json!("teacher.json");
    let mut pin = base["qualification"]["checkpoint"].clone();
    pin["lock"] = json!("checkpoint-files.json");
    pin["checkpoint_identity"] = configured["serving"]["checkpoint_identity"].clone();
    let overlay = json!({"schema": "chopinfer-qual-variant/v1", "model": "example",
                         "checkpoint": "EXAMPLE_REQUANT", "checks": configured["checks"],
                         "quick": configured["quick"],
                         "serving": {"profile_name": "tp4-mtp-requant"},
                         "references": {"teacher": reference}, "qualification": {"checkpoint": pin}});
    let expanded = declaration::external(&base, "requant", &overlay, &"1".repeat(64)).unwrap();
    declaration::validate(&expanded, &modules()).unwrap();
    assert_eq!(expanded["variant"], "requant");
    assert_eq!(
        expanded["variant_source"],
        json!({"overlay": "example/requant.json", "sha256": "1".repeat(64)})
    );
    // The serving identity defaults to the pin's, which every rule that
    // held the base's now holds, beside the variant's profile.
    let identity = &configured["serving"]["checkpoint_identity"];
    let teacher = rules(&expanded["checks"]["teacher"]);
    assert!(teacher.iter().any(|(_, rule)| rule["value"] == *identity));
    assert!(!teacher.iter().any(
        |(_, rule)| rule["value"] == base["qualification"]["checkpoint"]["checkpoint_identity"]
    ));
    assert!(teacher.iter().any(|(_, rule)| rule["path"]
        == json!([PROFILE[0], PROFILE[1], PROFILE[2], PROFILE[3], "name"])
        && rule["value"] == "tp4-mtp-requant"));
    let sha256 = &configured["references"]["teacher"]["sha256"];
    let margins = &expanded["checks"]["paired_teacher"]["constants"]["paired_teacher"]["PROFILE_POLICIES"]
        ["value"];
    for policy in margins.as_object().unwrap().values() {
        assert_eq!(&policy["reference_sha256"], sha256);
    }
    // What it owns resolves under the variant root; it keeps no base reference.
    let mut owned = configured["references"]["teacher"].clone();
    owned["path"] = json!({"env": declaration::VARIANTS_ENV, "path": "example/teacher.json"});
    assert_eq!(expanded["references"], json!({"teacher": owned}));
    assert_eq!(
        expanded["qualification"]["checkpoint"]["lock"],
        "example/checkpoint-files.json"
    );
}

#[test]
fn configurations_refuse_what_they_cannot_state() {
    let base = configured_fixture();
    let refused = |value: &Value, field: &str| {
        let error = declaration::validate(value, &modules())
            .unwrap_err()
            .to_string();
        assert!(error.contains(field), "{field}: {error}");
    };
    let mut value = base.clone();
    value["configurations"]["dcp2"]["unknown"] = json!(1);
    refused(&value, "configurations.dcp2.unknown: unknown field");
    let mut value = base.clone();
    value["configurations"]["dcp2"]["checks"] = json!(["no_such_check"]);
    refused(&value, "names undeclared check no_such_check");
    let mut value = base.clone();
    value["configurations"]["default"] = value["configurations"]["dcp2"].clone();
    refused(
        &value,
        "configurations.default: must be a plain name other than default",
    );
    let mut value = base.clone();
    value["configurations"]["dcp2"]["serving"]["topology"] = json!("x");
    refused(
        &value,
        "configurations.dcp2.serving.topology: unknown field",
    );
    let mut value = base.clone();
    value["variant"] = json!("dcp2");
    refused(
        &value,
        "configurations: a variant declares no configurations",
    );
    assert!(declaration::expand(&base, "absent").is_err());
}

/// A configuration declares its own waivers, check by check; the base
/// check's waivers are the default configuration's alone.
#[test]
fn a_configuration_declares_its_own_waivers() {
    let mut base = configured_fixture();
    let waiver = json!({"category": "prose-frozen", "metric": "top1_disagreement", "ceiling": 0.12,
                        "reason": "a reason", "decision": "operator, 2026-10-15"});
    base["checks"]["teacher"]["waivers"] = json!([waiver.clone()]);
    base["configurations"]["kv-bf16"]["waivers"] = json!({"teacher_arms": [waiver.clone()]});
    declaration::validate(&base, &modules()).unwrap();
    let kv = declaration::expand(&base, "kv-bf16").unwrap();
    assert_eq!(kv["checks"]["teacher_arms"]["waivers"], json!([waiver]));
    let dcp2 = declaration::expand(&base, "dcp2").unwrap();
    assert!(dcp2["checks"]["teacher"].get("waivers").is_none());
    let refused = |value: &Value, field: &str| {
        let error = declaration::validate(value, &modules())
            .unwrap_err()
            .to_string();
        assert!(error.contains(field), "{field}: {error}");
    };
    let mut value = base.clone();
    value["configurations"]["dcp2"]["waivers"] = json!({"teacher_arms": [waiver.clone()]});
    refused(
        &value,
        "configurations.dcp2.waivers.teacher_arms: names a check the configuration does not run",
    );
    let mut value = base.clone();
    value["configurations"]["dcp2"]["waivers"] = json!({"api": [waiver]});
    refused(
        &value,
        "configurations.dcp2: checks.api.waivers.0.category: only a teacher check has categories",
    );
}

fn offload_declaration() -> Value {
    let mut value = vectors()["base"].clone();
    value["capabilities"] = json!({"text": false, "frontend": false, "api": false,
        "structured": false, "prefix_cache": false, "preemption": false,
        "long_context": false, "stability": false, "media": [], "speculation": "none",
        "kv_offload": true});
    value["kv_offload"] = json!({"budget_gib": 16, "swap_every": 16, "max_preemptions": 3,
        "min_gated": 18, "pressure_tokens": 32768});
    for scenario in ["plain", "pressure"] {
        let name = format!("offload_{scenario}");
        let off = format!("kv_offload_{scenario}_off");
        let swap = format!("kv_offload_{scenario}_swap");
        value["checks"][&name] = json!({"module": "kv_offload",
            "args": ["report", "--off", format!("{{{off}}}"), "--swap", format!("{{{swap}}}")],
            "files": [off, swap], "timeout_seconds": 600, "receipt": "*-kv-offload-*.json",
            "rules": [{"path": ["passed"], "op": "eq", "value": true, "source": "docs/serving-kv-pool.md"}]});
        value["cohorts"]["full"]["checks"]
            .as_array_mut()
            .unwrap()
            .push(json!(name));
    }
    value
}

#[test]
fn offload_capability_requires_facts_and_both_hard_full_checks() {
    let modules = modules();
    let good = offload_declaration();
    declaration::validate(&good, &modules).unwrap();
    for field in ["kv_offload", "checks"] {
        let mut value = good.clone();
        if field == "kv_offload" {
            value.as_object_mut().unwrap().remove(field);
        } else {
            value["checks"]["offload_pressure"]["informational_reason"] = json!("not measured");
        }
        assert!(
            declaration::validate(&value, &modules)
                .unwrap_err()
                .to_string()
                .contains("kv_offload")
        );
    }
    let mut value = good.clone();
    value["kv_offload"]["swap_every"] = json!(false);
    assert!(
        declaration::validate(&value, &modules)
            .unwrap_err()
            .to_string()
            .contains("kv_offload")
    );
}

#[test]
fn disabled_offload_requires_no_facts_or_checks() {
    let mut value = offload_declaration();
    value["capabilities"]["kv_offload"] = json!(false);
    value.as_object_mut().unwrap().remove("kv_offload");
    value["cohorts"]["full"]["checks"]
        .as_array_mut()
        .unwrap()
        .retain(|id| !id.as_str().unwrap().starts_with("offload_"));
    value["checks"]
        .as_object_mut()
        .unwrap()
        .retain(|id, _| !id.starts_with("offload_"));
    declaration::validate(&value, &modules()).unwrap();
}
