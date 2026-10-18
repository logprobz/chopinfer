#[path = "support/registry.rs"]
mod fixture;

use model_mimo26::checkpoint::Registry;
use model_mimo26::checkpoint_provenance::Verifier;
use model_mimo26_contract::{Tp, encoders::Encoders};
use runtime_frontend_contract::FrontendDescriptorV1;

#[test]
fn injected_registry_drives_admission_frontends_and_tokenizer_contracts() {
    let fixture = fixture::Fixture::new();
    let registry = fixture.registry();
    assert_eq!(
        registry
            .identities()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        ["mopd", "rl", "third"]
    );
    let identity = registry.identity("third").unwrap();
    let mut forged = identity.clone();
    forged.frontend = Some(serde_json::json!({"tokenizer": {"model_max_length": 1}}));
    let mut verifier = Verifier::with_registry(&fixture.root, &forged, &registry).unwrap();
    assert_eq!(verifier.identity().frontend, identity.frontend);
    assert_eq!(verifier.read_tensor("a").unwrap(), [3, 7]);
    assert_eq!(
        registry
            .verifier(&fixture.root)
            .unwrap()
            .identity()
            .repository,
        identity.repository
    );
    for set in Encoders::SETS {
        let component = registry.component_for_flash(&forged, set).unwrap();
        let descriptor = FrontendDescriptorV1::from_component_bytes(&component).unwrap();
        assert_eq!(descriptor.tokenizer.repository, "fixture/third");
        assert_eq!(descriptor.tokenizer.model_max_length, 2097152);
        assert_eq!(
            descriptor.tokenizer.definition.sha256,
            runtime_weights::sha256_bytes(b"{}")
        );
        assert_eq!(descriptor.vocabulary.stop_token_ids, [151645]);
        assert_eq!(
            runtime_frontend_contract::tokenizer_contract_v1(&descriptor).unwrap(),
            registry.tokenizer_contract_for_flash(&forged).unwrap()
        );
    }
    assert!(
        registry
            .identity("unlisted")
            .unwrap_err()
            .contains("mopd, rl, third")
    );
    forged.revision = "unlisted".into();
    assert!(
        registry
            .component_for_flash(&forged, Encoders::ALL)
            .is_err()
    );
    assert!(registry.tokenizer_contract_for_flash(&forged).is_err());
    assert!(Verifier::with_registry(&fixture.root, &forged, &registry).is_err());
}

#[test]
fn registry_refuses_duplicate_pins_missing_locks_and_invalid_overrides() {
    let mut fixture = fixture::Fixture::new();
    let third = fixture.data["checkpoints"]["third"].clone();
    fixture.data["checkpoints"]["third"] = fixture.data["checkpoints"]["mopd"].clone();
    assert!(
        Registry::parse(&fixture.data.to_string(), &fixture.locks.to_string())
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate checkpoint identity pins")
    );
    fixture.data["checkpoints"]["third"] = third;
    let pins = fixture
        .locks
        .as_object_mut()
        .unwrap()
        .remove("third")
        .unwrap();
    assert!(
        Registry::parse(&fixture.data.to_string(), &fixture.locks.to_string())
            .err()
            .unwrap()
            .to_string()
            .contains("third has no source lock")
    );
    fixture.locks["third"] = pins;
    for override_value in [
        serde_json::json!(false),
        serde_json::json!({"tokenizer": "invalid"}),
        serde_json::json!({"models": {}}),
    ] {
        fixture.data["checkpoints"]["third"]["frontend"] = override_value;
        assert!(Registry::parse(&fixture.data.to_string(), &fixture.locks.to_string()).is_err());
    }
    fixture.data["checkpoints"]["third"]["frontend"] =
        serde_json::json!({"vocabulary": {"size": 1}});
    let registry = fixture.registry();
    assert!(
        registry
            .component_for_flash(registry.identity("third").unwrap(), Encoders::ALL)
            .is_err()
    );
}

#[test]
fn fixtures_started_together_keep_their_own_directories() {
    // These tests run as threads of one process: fixtures started at once
    // must not share, remove or refuse each other's directory.
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                let fixtures: Vec<_> = (0..4).map(|_| fixture::Fixture::new()).collect();
                for fixture in &fixtures {
                    assert!(fixture.root.join("config.json").is_file());
                }
                fixtures
                    .iter()
                    .map(|fixture| fixture.root.clone())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let roots: Vec<_> = threads
        .into_iter()
        .flat_map(|thread| thread.join().unwrap())
        .collect();
    let distinct: std::collections::BTreeSet<_> = roots.iter().collect();
    assert_eq!(distinct.len(), roots.len());
}

#[test]
fn fp8_kv_calibration_belongs_to_the_registered_checkpoint() {
    let mut fixture = fixture::Fixture::new();
    let registry = fixture.registry();
    let identity = registry.identity("third").unwrap();
    assert_eq!(registry.fp8_kv_layout(identity, Tp::TP4).unwrap(), None);
    let mut forged = identity.clone();
    forged.fp8_kv_scales = Some(model_mimo26_contract::serving::GA_FP8_KV_SCALES.to_vec());
    assert_eq!(registry.fp8_kv_layout(&forged, Tp::TP4).unwrap(), None);
    for name in ["mopd", "rl"] {
        assert!(
            registry
                .fp8_kv_layout(registry.identity(name).unwrap(), Tp::TP4)
                .unwrap()
                .is_some()
        );
    }
    fixture.data["checkpoints"]["third"]["fp8_kv_scales"] = serde_json::json!([[1.0, 1.0]]);
    let error = Registry::parse(&fixture.data.to_string(), &fixture.locks.to_string())
        .err()
        .unwrap();
    assert!(error.to_string().contains("FP8 KV calibration layer count"));
}

#[test]
fn compiled_calibration_is_advertised_only_for_its_registered_identity() {
    let mut fixture = fixture::Fixture::new();
    let tp = model_mimo26_contract::Tp::TP4;
    for format in model_mimo26_contract::serving::kv_formats(tp)
        .iter()
        .filter(|f| f.fp8())
    {
        fixture.data["checkpoints"]["third"]["fp8_kv_scales"] = serde_json::json!(format.scales);
        let registry = fixture.registry();
        let identity = registry.identity("third").unwrap();
        assert_eq!(
            registry
                .fp8_kv_layout(identity, tp)
                .unwrap()
                .unwrap()
                .as_bytes(),
            format.layout_id
        );
        let mut forged = identity.clone();
        forged.fp8_kv_scales = Some(vec![[2.0, 2.0]; 9]);
        assert_eq!(
            registry.fp8_kv_layout(&forged, tp).unwrap(),
            registry.fp8_kv_layout(identity, tp).unwrap()
        );
        for name in ["mopd", "rl"] {
            assert_eq!(
                registry
                    .fp8_kv_layout(registry.identity(name).unwrap(), tp)
                    .unwrap()
                    .unwrap(),
                "mimo.ga-paged-fp8.v2"
            );
        }
    }
    fixture.data["checkpoints"]["third"]
        .as_object_mut()
        .unwrap()
        .remove("fp8_kv_scales");
    let registry = fixture.registry();
    assert_eq!(
        registry
            .fp8_kv_layout(registry.identity("third").unwrap(), tp)
            .unwrap(),
        None
    );
    fixture.data["checkpoints"]["third"]["fp8_kv_scales"] = serde_json::json!(vec![[3.0, 3.0]; 9]);
    let registry = fixture.registry();
    assert!(
        registry
            .fp8_kv_layout(registry.identity("third").unwrap(), tp)
            .unwrap_err()
            .contains("rebuild from checkpoints.json")
    );
}

#[test]
fn registry_accepts_a_different_data_only_calibration() {
    let mut fixture = fixture::Fixture::new();
    let mut scales = model_mimo26_contract::serving::GA_FP8_KV_SCALES;
    scales[0][0] = 0.125;
    fixture.data["checkpoints"]["third"]["fp8_kv_scales"] = serde_json::json!(scales);
    let registry = fixture.registry();
    assert_eq!(
        registry.identity("third").unwrap().fp8_kv_scales.as_deref(),
        Some(scales.as_slice())
    );
}

/// A checkpoint no registered lock matches is served under its local
/// identity, named by its own metadata, with BF16 KV only, once its
/// tokenizer configuration states the official one's; an edited chat
/// template or configuration is refused.
#[test]
fn unregistered_checkpoints_are_served_under_their_local_identity() {
    let mut fixture = fixture::Fixture::new();
    fixture.state_tokenizer_semantics();
    // The registered locks admit as before.
    let registry = fixture.registry();
    let verifier = registry.verifier(&fixture.root).unwrap();
    assert_eq!(verifier.identity().served_model_name, "third");
    // An index no lock pins.
    std::fs::write(
        fixture.root.join("model.safetensors.index.json"),
        br#"{"metadata":{"total_size":2,"format":"local"},"weight_map":{"a":"weight.safetensors"}}"#,
    )
    .unwrap();
    let verifier = registry.verifier(&fixture.root).unwrap();
    let identity = verifier.identity();
    let digest = verifier.source_lock_sha256();
    assert!(identity.is_local());
    assert_eq!(identity.repository, "local");
    assert_eq!(identity.revision, digest[..40]);
    assert_eq!(identity.served_model_name, format!("local-{digest}"));
    assert_eq!(verifier.config().unwrap().vocab_size, 152576);
    let local = verifier.registry();
    for tp in Tp::SUPPORTED {
        assert_eq!(local.fp8_kv_layout(identity, tp).unwrap(), None);
    }
    let component = local.component_for_flash(identity, Encoders::NONE).unwrap();
    let frontend: serde_json::Value = serde_json::from_slice(&component).unwrap();
    assert_eq!(
        frontend["models"],
        serde_json::json!([{"id": identity.served_model_name, "owned_by": "local"}])
    );
    let descriptor = FrontendDescriptorV1::from_component_bytes(&component).unwrap();
    assert_eq!(descriptor.tokenizer.repository, "local");
    assert_eq!(descriptor.tokenizer.revision, identity.revision);
    assert_eq!(
        descriptor.tokenizer.configuration.sha256,
        runtime_weights::sha256_bytes(
            &std::fs::read(fixture.root.join("tokenizer_config.json")).unwrap()
        )
    );
    // The same metadata names the same identity; other metadata, another.
    let again = registry.verifier(&fixture.root).unwrap();
    assert_eq!(
        again.identity().served_model_name,
        identity.served_model_name
    );
    std::fs::write(
        fixture.root.join("model.safetensors.index.json"),
        br#"{"metadata":{"total_size":2,"format":"other"},"weight_map":{"a":"weight.safetensors"}}"#,
    )
    .unwrap();
    let other = registry.verifier(&fixture.root).unwrap();
    assert_ne!(
        other.identity().served_model_name,
        identity.served_model_name
    );

    let refused = |configuration: serde_json::Value, template: Option<&str>| {
        std::fs::write(
            fixture.root.join("tokenizer_config.json"),
            serde_json::to_vec(&configuration).unwrap(),
        )
        .unwrap();
        let path = fixture.root.join("chat_template.jinja");
        match template {
            Some(template) => std::fs::write(&path, template).unwrap(),
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
        let error = registry.verifier(&fixture.root).err().unwrap().to_string();
        assert!(
            error.starts_with("not a supported MiMo V2.6 Flash checkpoint ("),
            "{error}"
        );
        error[error.find("local: ").unwrap()..].to_owned()
    };
    let template = fixture::TEMPLATE;
    let edited = serde_json::json!({"chat_template": "{{ edited }}", "eos_token": "<|im_end|>"});
    assert!(refused(edited, None).contains("unsupported chat template semantics"));
    let separate = serde_json::json!({"chat_template": template, "eos_token": "<|im_end|>"});
    assert!(refused(separate, Some("{{ edited }}")).contains("chat_template.jinja disagrees"));
    let added = serde_json::json!({"chat_template": template, "eos_token": "<|im_end|>", "model_max_length": 8});
    assert!(refused(added, None).contains("unsupported tokenizer_config.model_max_length"));
    let missing = serde_json::json!({"chat_template": template});
    assert!(refused(missing, None).contains("missing or unsupported tokenizer_config.eos_token"));
}

/// The registry refuses an identity that claims the local name.
#[test]
fn registered_identities_may_not_claim_the_local_name() {
    let mut fixture = fixture::Fixture::new();
    fixture.data["checkpoints"]["third"]["repository"] = "local".into();
    let error = Registry::parse(&fixture.data.to_string(), &fixture.locks.to_string())
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("names an unregistered checkpoint")
    );
}

/// The official release's tokenizer files state the semantics a local
/// identity is checked against, and build the tokenizer its local frontend
/// declares; an edited chat template is refused. Needs
/// `CHOPINFER_MIMO_CHECKPOINT`; only its metadata is read.
#[test]
fn real_tokenizer_states_the_official_semantics() {
    let Ok(checkpoint) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
        eprintln!("SKIP: set CHOPINFER_MIMO_CHECKPOINT");
        return;
    };
    let checkpoint = std::path::PathBuf::from(checkpoint);
    let fixture = fixture::Fixture::new();
    let root = fixture.root.join("local");
    std::fs::create_dir(&root).unwrap();
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ] {
        std::fs::copy(checkpoint.join(name), root.join(name)).unwrap();
    }
    let mut index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(checkpoint.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    index["metadata"]["format"] = "local".into();
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    let registry = model_mimo26::checkpoint::registry()
        .with_local(&root)
        .unwrap();
    let identity = registry.identity("local").unwrap();
    assert!(identity.served_model_name.starts_with("local-"));
    registry
        .check_tokenizer(&root, identity, &model_mimo26_contract::geometry::V26_FLASH)
        .unwrap();
    let mut configuration: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("tokenizer_config.json")).unwrap())
            .unwrap();
    configuration["chat_template"] = "{{ edited }}".into();
    std::fs::write(
        root.join("tokenizer_config.json"),
        serde_json::to_vec(&configuration).unwrap(),
    )
    .unwrap();
    let error = model_mimo26::checkpoint::registry()
        .with_local(&root)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("unsupported chat template semantics"),
        "{error}"
    );
}
