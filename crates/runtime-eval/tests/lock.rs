//! Declared inputs read through the reference lock, as `tools/qual/refs.py`
//! authenticates them.

use std::path::{Path, PathBuf};

use runtime_eval::declaration;
use runtime_eval::json::{canonical_sha256, sha256_hex};
use runtime_eval::lock;
use serde_json::json;

fn checkout() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn the_shipped_inputs_read_as_their_lock_pins_them() {
    // Each model's declared inputs among these read as its lock pins them; a
    // model need not declare every one, but some locked model covers each.
    const KEYS: [&str; 4] = [
        "tokenizer_goldens",
        "chat_renders",
        "frontend",
        "fixture_root",
    ];
    let mut covered = [0; KEYS.len()];
    let mut locked = 0;
    for entry in std::fs::read_dir(checkout().join("models")).unwrap() {
        let model = entry.unwrap().path();
        if !model.join(lock::FILE_NAME).is_file() {
            continue;
        }
        locked += 1;
        let loaded = declaration::load(&model).unwrap();
        let model = model.display();
        assert!(lock::load(&loaded).unwrap().is_some(), "{model}");
        for (key, covered) in KEYS.iter().zip(&mut covered) {
            if loaded.declaration.input(key).is_none() {
                continue;
            }
            *covered += 1;
            let path = runtime_eval::references::input_path(&loaded, key, &|_| None)
                .unwrap_or_else(|| panic!("{model} {key}: unresolved"));
            if *key == "fixture_root" {
                lock::read(&loaded, key, &path.join("teacher-texts.json")).unwrap();
                continue;
            }
            let bytes = lock::read(&loaded, key, &path).unwrap();
            assert_eq!(bytes, std::fs::read(&path).unwrap(), "{model} {key}");
        }
    }
    assert!(locked > 0, "no shipped model has a reference lock");
    for (key, covered) in KEYS.iter().zip(covered) {
        assert!(covered > 0, "no locked model declares {key}");
    }
}

#[test]
fn a_configuration_reads_the_model_lock_with_its_own_entries() {
    let mut compared = 0;
    for entry in std::fs::read_dir(checkout().join("models")).unwrap() {
        let model = entry.unwrap().path();
        if !model.join(lock::FILE_NAME).is_file() {
            continue;
        }
        let (_, value) = runtime_eval::json::read(&model.join(lock::FILE_NAME)).unwrap();
        let base = declaration::load(&model).unwrap();
        let Some(default) =
            runtime_eval::references::input_path(&base, "teacher_reference", &|_| None)
        else {
            continue;
        };
        for (name, spec) in value["configurations"].as_object().into_iter().flatten() {
            let Some(path) = spec["entries"]["teacher_reference"]["path"].as_str() else {
                continue;
            };
            if spec["entries"]["teacher_reference"] == value["entries"]["teacher_reference"] {
                continue;
            }
            let loaded = declaration::load_variant(&model, Some(name)).unwrap();
            assert_eq!(
                lock::path(&loaded).canonicalize().unwrap(),
                model.join(lock::FILE_NAME).canonicalize().unwrap()
            );
            let input =
                runtime_eval::references::input_path(&loaded, "teacher_reference", &|_| None)
                    .unwrap();
            assert!(input.ends_with(path), "{name}: {}", input.display());
            assert_eq!(
                lock::read(&loaded, "teacher_reference", &input).unwrap(),
                std::fs::read(&input).unwrap()
            );
            assert!(lock::read(&loaded, "teacher_reference", &default).is_err());
            assert_eq!(
                lock::read(&base, "teacher_reference", &default).unwrap(),
                std::fs::read(&default).unwrap()
            );
            compared += 1;
        }
    }
    assert!(
        compared > 0,
        "no configuration overrides its teacher reference"
    );
}

/// A root whose model locks one input file and one directory member.
fn locked_root(name: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("runtime-eval-lock-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let model = root.join("models/example");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(root.join("fixtures/example")).unwrap();
    std::fs::create_dir_all(root.join("tools/qual")).unwrap();
    std::fs::copy(
        checkout().join("tools/qual/modules.json"),
        root.join("tools/qual/modules.json"),
    )
    .unwrap();
    let samples = b"{\"samples\": []}".to_vec();
    let texts = b"{\"texts\": []}".to_vec();
    std::fs::write(root.join("fixtures/example/api-samples.json"), &samples).unwrap();
    std::fs::write(root.join("fixtures/example/teacher-texts.json"), &texts).unwrap();
    let files = json!({"model.safetensors": {"sha256": "ab", "size": 1}});
    std::fs::write(model.join("checkpoint-files.json"), files.to_string()).unwrap();
    let pin = json!({"lock": "models/example/checkpoint-files.json", "repository": "org/example",
                     "revision": "abc", "served_model_name": "Example",
                     "source_lock_sha256": canonical_sha256(&files),
                     "checkpoint_identity": "4444444444444444444444444444444444444444444444444444444444444444"});
    let rule = json!({"path": ["passed"], "op": "eq", "value": true, "source": "docs/example.md"});
    let qual = json!({
        "schema": "chopinfer-qual-model/v1", "model": "example",
        "cohorts": {"quick": {"checks": ["api"]}},
        "change_classes": {"frontend": [], "quality": [], "service": [], "speculation": [], "prefix": []},
        "checks": {"api": {"module": "api", "args": [], "timeout_seconds": 60, "receipt": "*-api.json",
                           "rules": [rule]}},
        "references": {},
        "inputs": {"api_samples": "fixtures/example/api-samples.json", "fixture_root": "fixtures/example"},
        "qualification": {"teacher_reference_kind": "exact", "teacher_receipt": "teacher",
                          "teacher_capture_schema": "example.capture.v1", "verify_buckets": [2],
                          "chars_per_token": 4.0, "checkpoint": pin},
        "receipts": {"env": "RUNTIME_EVAL_TEST_UNSET", "path": "qual/example"}
    });
    std::fs::write(model.join("qual.json"), qual.to_string()).unwrap();
    let object = json!({"schema": "legacy-json-object", "production": {"producer": "test"}});
    let lock_value = json!({
        "schema": lock::SCHEMA, "model": "example",
        "objects": {sha256_hex(&samples): object, sha256_hex(&texts): object},
        "entries": {
            "api_samples": {"path": "fixtures/example/api-samples.json", "directory": false,
                            "members": {".": sha256_hex(&samples)}},
            "fixture_root": {"path": "fixtures/example", "directory": true,
                             "scope": ["teacher-texts.json"],
                             "members": {"teacher-texts.json": sha256_hex(&texts)}}
        }
    });
    std::fs::write(model.join(lock::FILE_NAME), lock_value.to_string()).unwrap();
    root
}

#[test]
fn a_changed_or_unadopted_input_is_refused() {
    let root = locked_root("refused");
    let loaded = declaration::load(&root.join("models/example")).unwrap();
    let samples = root.join("fixtures/example/api-samples.json");
    let texts = root.join("fixtures/example/teacher-texts.json");
    assert_eq!(
        lock::read(&loaded, "api_samples", &samples).unwrap(),
        b"{\"samples\": []}"
    );
    assert_eq!(
        lock::read(&loaded, "fixture_root", &texts).unwrap(),
        b"{\"texts\": []}"
    );
    let other = root.join("fixtures/example/other.json");
    std::fs::write(&other, b"{}").unwrap();
    assert_eq!(
        lock::read(&loaded, "fixture_root", &other).unwrap_err(),
        "lock entry fixture_root: member other.json is not adopted"
    );
    assert_eq!(
        lock::read(&loaded, "chat_renders", &samples).unwrap_err(),
        "lock entry chat_renders: not adopted"
    );
    std::fs::write(&samples, b"{\"samples\": [1]}").unwrap();
    assert_eq!(
        lock::read(&loaded, "api_samples", &samples).unwrap_err(),
        "lock entry api_samples: sha256 differs for ."
    );
    std::fs::write(&texts, b"{}").unwrap();
    assert_eq!(
        lock::read(&loaded, "fixture_root", &texts).unwrap_err(),
        "lock entry fixture_root: sha256 differs for teacher-texts.json"
    );
    std::fs::remove_dir_all(root).unwrap();
}
