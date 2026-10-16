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
