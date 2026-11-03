use super::*;

fn words(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).into()).collect()
}

/// Capture actual resolver output for before/after comparisons outside the tree.
/// The test exercises every shipped profile even when no capture is requested.
#[test]
fn capture_prefill_policy_equivalence() {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut profiles = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(models).unwrap().flatten() {
        let path = entry.path().join("profiles.json");
        if !path.is_file() {
            continue;
        }
        let file = crate::profiles::Profiles::read(&path).unwrap();
        for (name, profile) in &file.profiles {
            let mut sizes = std::collections::BTreeSet::from([1]);
            sizes.extend(
                file.profiles
                    .values()
                    .filter(|p| p.tensor_parallel == profile.tensor_parallel)
                    .map(|p| p.context_parallel_size.unwrap_or(1)),
            );
            for context_parallel in sizes {
                let explicit = vec![
                    "--context-parallel-size".into(),
                    context_parallel.to_string(),
                ];
                let explicit_inputs = ServingInputsV1::decode(&explicit).unwrap();
                let defaults = profile.inputs(None).unwrap().overlay(&explicit_inputs);
                let command = crate::profiles::standalone(defaults).unwrap();
                let direct = runtime_bundle::preparation::DirectProfile {
                    name: name.clone(),
                    profile: serde_json::to_value(profile).unwrap(),
                };
                let resolved = crate::profiles::direct_inputs(
                    &direct,
                    profile.tensor_parallel as u32,
                    &explicit_inputs,
                    Path::new("/direct-package"),
                )
                .unwrap();
                assert_eq!(
                    command,
                    crate::profiles::standalone(resolved.inputs).unwrap()
                );
                let shown: serde_json::Value = serde_json::from_str(
                    &crate::profiles::inspect(&[
                        "show".into(),
                        path.to_str().unwrap().into(),
                        name.clone(),
                    ])
                    .unwrap(),
                )
                .unwrap();
                profiles.insert(
                    format!("{}/{name}/cp={context_parallel}", file.model),
                    serde_json::json!({
                        "shown": shown,
                        "settings": effective_settings_v1(&command),
                        "command": format!("{command:?}"),
                        "tensor_parallel": profile.tensor_parallel,
                        "context_parallel": context_parallel,
                    }),
                );
            }
        }
    }
    assert!(!profiles.is_empty());
    let manifest = super::tests::manifest_with_rows(64, 1 << 20);
    let mut configurations = std::collections::BTreeMap::new();
    for (name, flags) in [
        ("fixed", vec![]),
        ("tbt", words(&["--prefill-policy", "tbt"])),
        ("adaptive", words(&["--prefill-policy", "adaptive"])),
    ] {
        let mut arguments = words(&["serve", "--bundle", "fixture"]);
        arguments.extend(flags);
        let HostCommandV1::Serve { serving, .. } = parse_host_command_v1(&arguments).unwrap()
        else {
            panic!("serve");
        };
        configurations.insert(
            format!("{name}/server"),
            serde_json::json!({
                "config": format!("{:?}", server_config_v1(*serving, RequestSurfaceV1::Full, None).unwrap()),
            }),
        );
        let config = engine_config_v1(
            &manifest,
            ModelVocabulary::new(128).unwrap(),
            1 << 20,
            *serving,
            None,
        )
        .unwrap();
        for grid in [false, true] {
            for cuts in [false, true] {
                let config = if grid {
                    config.with_guaranteed_prefill_chunks(2048).unwrap()
                } else {
                    config
                }
                .with_prefill_cuts(cuts);
                let result = negotiate_prefill_timing_cuts_v1(config, serving.prefill_policy);
                let value = match result {
                    Ok(config) => serde_json::json!({
                        "config": format!("{config:?}"),
                        "startup": prefill_timing_cuts_startup_v1(config),
                    }),
                    Err(_) => serde_json::json!({"refused": true}),
                };
                configurations.insert(format!("{name}/grid={grid}/cuts={cuts}"), value);
            }
        }
    }
    if let Some(path) = std::env::var_os("CHOPINFER_PREFILL_POLICY_CAPTURE") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "profiles": profiles, "configurations": configurations,
            }))
            .unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn consolidated_prefill_policy_is_accepted() {
    for value in ["fixed", "tbt", "adaptive"] {
        parse_host_command_v1(&words(&[
            "serve",
            "--bundle",
            "fixture",
            "--prefill-policy",
            value,
        ]))
        .unwrap();
    }
}

#[test]
fn policy_targets_and_negotiated_capabilities_preserve_the_schedule() {
    use runtime_scheduler::{MixedPolicy, PrefillBudget};
    let manifest = super::tests::manifest_with_rows(64, 1 << 20);
    for policy in ["fixed", "tbt", "adaptive"] {
        for target in [None, Some(1), Some(150), Some(60_000)] {
            let mut args = words(&["serve", "--bundle", "fixture", "--prefill-policy", policy]);
            if let Some(target) = target {
                args.extend(["--decode-tbt-target-ms".into(), target.to_string()]);
            }
            let parsed = parse_host_command_v1(&args);
            if policy == "fixed" && target.is_some() {
                assert!(
                    parsed
                        .unwrap_err()
                        .to_string()
                        .contains("requires --prefill-policy")
                );
                continue;
            }
            let HostCommandV1::Serve { serving, .. } = parsed.unwrap() else {
                panic!("serve");
            };
            let budget = if policy == "fixed" {
                PrefillBudget::Fixed
            } else {
                PrefillBudget::Tbt(MixedPolicy::new(target.unwrap_or(100), 512, 0).unwrap())
            };
            assert_eq!(serving.prefill_budget, budget);
            let base = engine_config_v1(
                &manifest,
                ModelVocabulary::new(128).unwrap(),
                1 << 20,
                *serving,
                None,
            )
            .unwrap();
            for grid in [false, true] {
                for cuts in [false, true] {
                    let config = if grid {
                        base.with_guaranteed_prefill_chunks(2048).unwrap()
                    } else {
                        base
                    }
                    .with_prefill_cuts(cuts);
                    let negotiated =
                        negotiate_prefill_timing_cuts_v1(config, serving.prefill_policy);
                    if policy == "adaptive" && grid && !cuts {
                        assert!(negotiated.unwrap_err().contains(
                            "--prefill-policy adaptive needs a bundle that declares prefill cuts"
                        ));
                        continue;
                    }
                    let negotiated = negotiated.unwrap();
                    assert_eq!(negotiated.prefill_budget(), budget);
                    assert_eq!(negotiated.prefill_policy_name(), Some(policy));
                    assert_eq!(
                        negotiated,
                        config
                            .with_adaptive_prefill_budget(policy == "adaptive")
                            .with_prefill_timing_cuts(policy == "adaptive" && grid)
                    );
                    assert_eq!(
                        prefill_timing_cuts_startup_v1(negotiated),
                        match (policy, grid) {
                            ("adaptive", true) => Some(ADAPTIVE_PREFILL_TIMING_CUTS_STARTUP_V1),
                            ("adaptive", false) => Some(ADAPTIVE_PREFILL_BUDGET_STARTUP_V1),
                            _ => None,
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn policy_and_target_refuse_invalid_and_duplicate_inputs() {
    for value in ["off", "strict", "target", "decode-aware", "TBT", ""] {
        let error = parse_host_command_v1(&words(&[
            "serve",
            "--bundle",
            "fixture",
            "--prefill-policy",
            value,
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("--prefill-policy must be fixed, tbt or adaptive"),
            "{error}"
        );
    }
    for value in ["0", "60001", "-1", "1.5", "many"] {
        for selection in ["--bundle", "--model"] {
            let error = parse_host_command_v1(&words(&[
                "serve",
                selection,
                "fixture",
                "--prefill-policy",
                "tbt",
                "--decode-tbt-target-ms",
                value,
            ]))
            .unwrap_err()
            .to_string();
            assert!(error.contains("--decode-tbt-target-ms"), "{error}");
        }
    }
    for flag in ["--prefill-policy", "--decode-tbt-target-ms"] {
        let value = if flag == "--prefill-policy" {
            "tbt"
        } else {
            "150"
        };
        let error = parse_host_command_v1(&words(&[
            "serve", "--bundle", "fixture", flag, value, flag, value,
        ]))
        .unwrap_err()
        .to_string();
        assert!(error.contains("more than once"), "{error}");
        let error = parse_host_command_v1(&words(&[
            "conformance-run",
            "--bundle",
            "fixture",
            flag,
            value,
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            error.contains(&format!("does not accept {flag}")),
            "{error}"
        );
    }
}

#[test]
fn retired_prefill_flags_name_the_migration() {
    for (flag, value) in [
        ("--prefill-budget", "fixed"),
        ("--prefill-timing-cuts", "off"),
        ("--prefill-itl-target-ms", "150"),
        ("--prefill-decode-steps", "2"),
        ("--mixed-min-prefill-tokens", "512"),
        ("--mixed-decode-steps", "0"),
    ] {
        for suffix in [
            vec![flag.to_owned(), value.to_owned()],
            vec![format!("{flag}={value}")],
            vec![flag.into()],
        ] {
            for prefix in [
                words(&["serve", "--bundle", "fixture"]),
                words(&["serve", "--model", "/missing-checkpoint"]),
                words(&["serve", "--bundle", "fixture", "--profile", "/missing#test"]),
            ] {
                let mut args = prefix.clone();
                args.extend(suffix.clone());
                let error = parse_host_command_v1(&args).unwrap_err().to_string();
                assert!(
                    error.contains("retired") && error.contains("--prefill-policy"),
                    "{args:?}: {error}"
                );
                for new_first in [false, true] {
                    let current = words(&["--prefill-policy", "tbt"]);
                    let mut args = prefix.clone();
                    args.extend(if new_first {
                        current.clone()
                    } else {
                        suffix.clone()
                    });
                    args.extend(if new_first { suffix.clone() } else { current });
                    let error = parse_host_command_v1(&args).unwrap_err().to_string();
                    assert!(
                        error.contains("retired") && error.contains("--prefill-policy"),
                        "{args:?}: {error}"
                    );
                }
            }
        }
    }
}

#[test]
fn policy_profile_deserialization_keeps_duplicate_field_refusals() {
    for fields in [
        r#""prefill_policy":"fixed","prefill_policy":"tbt""#,
        r#""decode_tbt_target_ms":100,"decode_tbt_target_ms":150"#,
    ] {
        let json = format!(
            r#"{{"schema":"chopinfer-serve-profiles-v2","model":"fixture","served_model_name":"fixture","profiles":{{"test":{{{fields}}}}}}}"#
        );
        let error = crate::profiles::Profiles::parse(json.as_bytes())
            .unwrap_err()
            .to_string();
        assert!(error.contains("duplicate field"), "{error}");
    }
}

#[test]
fn retired_profile_fields_including_null_name_the_migration() {
    for field in [
        "prefill_budget",
        "prefill_timing_cuts",
        "prefill_itl_target_ms",
        "prefill_decode_steps",
        "mixed_min_prefill_tokens",
        "mixed_decode_steps",
    ] {
        for value in [serde_json::Value::Null, serde_json::json!("fixed")] {
            let bytes = serde_json::to_vec(&serde_json::json!({
                "schema": "chopinfer-serve-profiles-v2", "model": "fixture", "served_model_name": "fixture",
                "profiles": {"test": {field: value}},
            })).unwrap();
            let error = crate::profiles::Profiles::parse(&bytes)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("retired") && error.contains("prefill_policy"),
                "{field}: {error}"
            );
        }
    }
}
