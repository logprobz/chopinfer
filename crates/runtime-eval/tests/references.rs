//! Pinned references and their bindings, against the shipped declarations
//! and temporary reference caches.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use runtime_eval::declaration::{self, Input};
use runtime_eval::json::{canonical, sha256_hex};
use runtime_eval::references::{self, Binding, Cache, ReferenceError};

/// Every shipped declaration, in model directory order.
fn shipped() -> Vec<declaration::Loaded> {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut directories: Vec<PathBuf> = std::fs::read_dir(models)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|directory| directory.join(declaration::FILE_NAME).is_file())
        .collect();
    directories.sort();
    // Each model's declaration and its checkpoint variants'.
    let loaded: Vec<_> = directories
        .iter()
        .flat_map(|directory| {
            let variants = declaration::variants(directory).unwrap();
            std::iter::once(declaration::load(directory).unwrap()).chain(
                variants
                    .into_iter()
                    .map(|variant| declaration::load_variant(directory, Some(&variant)).unwrap()),
            )
        })
        .collect();
    assert!(!loaded.is_empty());
    loaded
}

#[test]
fn every_shipped_reference_verifies_against_its_pin() {
    let mut bound = 0;
    for loaded in shipped() {
        references::checkpoint_lock(&loaded).unwrap();
        for pin in &loaded.declaration.references {
            // A workload kept outside the repository is read only where its
            // variable is set, as crate tests that need real data do.
            if let Some(Input::Environment { env, .. }) = &pin.workload
                && std::env::var_os(env).is_none()
            {
                assert!(matches!(
                    references::pinned(&loaded, pin, None),
                    Err(ReferenceError::Stale(field, _)) if field == "workload"
                ));
                continue;
            }
            let verified = references::pinned(&loaded, pin, None).unwrap_or_else(|error| {
                panic!("{}: {}: {error}", loaded.declaration.model, pin.name)
            });
            assert_eq!(verified.sha256, pin.sha256);
            if let Binding::Bound { unchecked, .. } = &verified.binding {
                // A checkout holds the adapter and runner files.
                assert!(unchecked.is_empty());
                bound += 1;
            }
        }
    }
    assert!(bound > 0, "no shipped reference is bound");
}

#[test]
fn a_reference_with_other_bytes_or_elsewhere_is_refused() {
    // A declaration may pin no reference yet, so take the first that pins one.
    let loaded = shipped()
        .into_iter()
        .find(|loaded| !loaded.declaration.references.is_empty())
        .expect("no shipped declaration pins a reference");
    let pin = loaded.declaration.references[0].clone();
    let scratch =
        std::env::temp_dir().join(format!("runtime-eval-references-{}", std::process::id()));
    let target =
        references::declared_path(&pin.path, &scratch, &references::process_environment).unwrap();
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    assert!(matches!(
        references::pinned(&loaded, &pin, Some(&scratch)),
        Err(ReferenceError::Missing(_))
    ));
    let mut bytes = std::fs::read(
        references::declared_path(&pin.path, &loaded.root, &references::process_environment)
            .unwrap(),
    )
    .unwrap();
    bytes.push(b'\n');
    std::fs::write(&target, &bytes).unwrap();
    let refused = references::pinned(&loaded, &pin, Some(&scratch));
    assert!(
        matches!(refused, Err(ReferenceError::Unverified(_))),
        "{refused:?}"
    );
    bytes.pop();
    std::fs::write(&target, &bytes).unwrap();
    assert!(references::pinned(&loaded, &pin, Some(&scratch)).is_ok());
    std::fs::remove_dir_all(scratch).unwrap();
}

/// The first shipped bound reference whose workload is in the repository,
/// with its declaration and pin.
fn bound() -> (declaration::Loaded, declaration::ReferencePin, Value) {
    for loaded in shipped() {
        for pin in loaded.declaration.references.clone() {
            if matches!(pin.workload, Some(Input::Environment { .. })) {
                continue;
            }
            let verified = verified(&loaded, &pin, None).unwrap();
            if verified.binding != Binding::Unbound {
                return (loaded, pin, verified.value);
            }
        }
    }
    panic!("no shipped reference is bound");
}

/// The shipped reference `pin`, verified with the cache `store`.
fn verified(
    loaded: &declaration::Loaded,
    pin: &declaration::ReferencePin,
    store: Option<&Path>,
) -> Result<references::Verified, ReferenceError> {
    let path = references::declared_path(&pin.path, &loaded.root, &references::process_environment)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let checkpoint = &loaded.declaration.checkpoint;
    references::verify_pinned(
        &loaded.root,
        checkpoint,
        pin,
        &path,
        bytes,
        &|| Ok(()),
        store,
    )
}

/// Verifies `value` as the reference `pin` names, re-pinned to its bytes so
/// only its binding can refuse it.
fn repinned(
    loaded: &declaration::Loaded,
    pin: &declaration::ReferencePin,
    value: &Value,
) -> Result<references::Verified, ReferenceError> {
    let scratch = std::env::temp_dir().join(format!(
        "runtime-eval-binding-{}-{}",
        std::process::id(),
        SCRATCH.fetch_add(1, Ordering::Relaxed)
    ));
    let target =
        references::declared_path(&pin.path, &scratch, &references::process_environment).unwrap();
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();
    let bytes = serde_json::to_vec(value).unwrap();
    std::fs::write(&target, &bytes).unwrap();
    let mut edited = pin.clone();
    edited.sha256 = runtime_eval::json::sha256_hex(&bytes);
    let result = references::pinned(loaded, &edited, Some(&scratch));
    std::fs::remove_dir_all(&scratch).unwrap();
    result
}

static SCRATCH: AtomicUsize = AtomicUsize::new(0);

#[test]
fn a_bound_fixture_must_match_its_binding() {
    let (loaded, pin, mut value) = bound();
    let list = if value.get("texts").is_some() {
        "texts"
    } else {
        "cases"
    };
    value[list][0]["edited"] = true.into();
    match repinned(&loaded, &pin, &value) {
        Err(ReferenceError::Stale(field, _)) => assert_eq!(field, "assembled_fixture"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_mode_policy_modes_no_longer_gives_is_stale() {
    let (loaded, pin, value) = bound();
    // The key holds the policy digest, not the dequant rule, so another
    // rule leaves the binding consistent with itself.
    let mut changed = value.clone();
    let mode = &mut changed["binding"]["modes"]["f32"]["dequant"];
    *mode = if mode == "exact" { "rounded" } else { "exact" }.into();
    let refused = repinned(&loaded, &pin, &changed).unwrap_err();
    assert_eq!(
        refused.to_string(),
        "stale_reference: mode_policy: modes.py no longer gives the bound f32 policy"
    );
    let mut changed = value;
    changed["binding"]["modes"]["bf16"]["dequant"] = "exact".into();
    let refused = repinned(&loaded, &pin, &changed).unwrap_err();
    assert_eq!(
        refused,
        ReferenceError::Unverified("mode bf16 cannot dequantize by exact".into())
    );
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "runtime-eval-{name}-{}-{}",
        std::process::id(),
        SCRATCH.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

/// A fixture's scored items by name (`cache.items`).
fn items(value: &Value) -> serde_json::Map<String, Value> {
    if let Some(texts) = value["texts"].as_array() {
        texts
            .iter()
            .map(|item| (item["name"].as_str().unwrap().into(), item.clone()))
            .collect()
    } else if let Some(cases) = value["cases"].as_array() {
        cases
            .iter()
            .map(|item| (item["id"].as_str().unwrap().into(), item.clone()))
            .collect()
    } else {
        value["probes"].as_object().unwrap().clone()
    }
}

/// The bound entries of `value`: (mode label, item name, key, key document,
/// the scores the fixture copies), in binding order.
fn entries(value: &Value) -> Vec<(String, String, String, Value, Value)> {
    let binding = &value["binding"];
    let sections = &binding["key"];
    let items = items(value);
    let mut result = Vec::new();
    for (label, mode) in binding["modes"].as_object().unwrap() {
        for entry in mode["entries"].as_array().unwrap() {
            let name = entry[0].as_str().unwrap();
            let sequence = &binding["sequences"][entry[1].as_u64().unwrap() as usize];
            let document = json!({
                "schema": references::KEY, "model": binding["model"],
                "checkpoint": sections["checkpoint"], "sequence": sequence,
                "reference_code": sections["reference_code"], "adapter": sections["adapter"],
                "mode": {"name": mode["mode"], "policy_sha256": mode["policy_sha256"]},
                "environment": sections["environment"], "outputs": sections["outputs"],
            });
            let item = &items[name];
            let scores = if sequence["answer"]["policy"] == "greedy" {
                let mut scores = item["greedy"].clone();
                scores["answer"] = item["answer_token_ids"].clone();
                scores
            } else {
                match &item[mode["variant"].as_str().unwrap()] {
                    Value::Object(copied) => Value::Object(copied.clone()),
                    _ => json!({}),
                }
            };
            let key = entry[2].as_str().unwrap().to_owned();
            result.push((label.clone(), name.to_owned(), key, document, scores));
        }
    }
    result
}

fn entry_path(store: &Path, key: &str) -> PathBuf {
    store.join("entries").join(&key[..2]).join(key)
}

/// Writes a sealed entry as `cache.Store.put` does.
fn put(store: &Path, document: &Value, scores: &Value) {
    let data = [
        ("key.json", canonical(document)),
        ("scores.json", canonical(scores)),
    ];
    let key = sha256_hex(&data[0].1);
    let path = entry_path(store, &key);
    std::fs::create_dir_all(&path).unwrap();
    let mut files = serde_json::Map::new();
    for (name, raw) in &data {
        std::fs::write(path.join(name), raw).unwrap();
        files.insert(
            (*name).into(),
            json!({"bytes": raw.len(), "sha256": sha256_hex(raw)}),
        );
    }
    let entry = json!({"schema": references::ENTRY, "key_sha256": key, "files": files,
                       "provenance": {}});
    std::fs::write(path.join("entry.json"), canonical(&entry)).unwrap();
}

/// A cache holding every entry `value` binds, with the scores it copies,
/// and its refset.
fn cached(value: &Value, store: &Path) {
    let binding = &value["binding"];
    let mut pairs = serde_json::Map::new();
    for (label, name, key, document, mut scores) in entries(value) {
        // The cache keeps more top-k columns than the fixture copies.
        for field in ["topk_ids", "topk_logits", "topk_logprobs"] {
            for row in scores[field].as_array_mut().into_iter().flatten() {
                row.as_array_mut().unwrap().push(json!(0));
            }
        }
        put(store, &document, &scores);
        assert!(entry_path(store, &key).is_dir(), "{label}/{name}");
        let listed = pairs.entry(label).or_insert_with(|| json!([]));
        listed.as_array_mut().unwrap().push(json!([name, key]));
    }
    let refset = json!({"schema": references::REFSET, "model": binding["model"],
                        "workload_sha256": binding["workload_sha256"], "modes": pairs,
                        "assembled_fixture_sha256": binding["assembled_fixture_sha256"]});
    let bytes = canonical(&refset);
    assert_eq!(sha256_hex(&bytes), binding["refset_sha256"]);
    std::fs::create_dir_all(store.join("refsets")).unwrap();
    let refset_sha256 = binding["refset_sha256"].as_str().unwrap();
    std::fs::write(store.join(format!("refsets/{refset_sha256}.json")), bytes).unwrap();
}

/// The first copied score of a forced entry, changed in its cache entry.
fn change_a_score(value: &Value, store: &Path) -> String {
    for (label, name, key, document, mut scores) in entries(value) {
        let Some((field, _)) = scores.as_object().and_then(|copied| copied.iter().next()) else {
            continue;
        };
        let field = field.clone();
        scores[&field] = json!("changed");
        std::fs::remove_dir_all(entry_path(store, &key)).unwrap();
        put(store, &document, &scores);
        return format!(
            "stale_reference: scores: {label}/{name}: {field} differs from cache entry {key}"
        );
    }
    panic!("the fixture copies no scores");
}

fn cache_of(binding: &Binding) -> Option<Cache> {
    match binding {
        Binding::Bound { cache, .. } => cache.clone(),
        Binding::Unbound => panic!("unbound"),
    }
}

#[test]
fn the_bound_scores_are_compared_with_a_cache_store() {
    let (loaded, pin, value) = bound();
    let directory = scratch("refcache");
    let store = directory.join("refcache/v1");
    cached(&value, &store);
    let count = entries(&value)
        .iter()
        .map(|entry| entry.2.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let found = |refset_present, missing| Cache {
        root: store.display().to_string(),
        refset_present,
        entries: count,
        missing,
    };
    let binding = verified(&loaded, &pin, Some(&store)).unwrap().binding;
    assert_eq!(cache_of(&binding), Some(found(true, 0)));
    assert_eq!(
        binding.to_json()["cache"],
        json!({"root": store.display().to_string(), "refset_present": true,
               "entries": count, "missing": 0})
    );

    // Without a store the binding says so.
    let binding = verified(&loaded, &pin, None).unwrap().binding.to_json();
    let fields: Vec<&String> = binding.as_object().unwrap().keys().collect();
    // In a checkout every check runs: the binding reads as Python's.
    assert_eq!(fields, ["status", "refset_sha256", "cache"]);
    assert_eq!(binding["cache"], Value::Null);

    // A cached score unlike the fixture's is stale.
    let message = change_a_score(&value, &store);
    let refused = verified(&loaded, &pin, Some(&store)).unwrap_err();
    assert_eq!(refused.to_string(), message);

    // A file unlike its seal is refused.
    let (_, _, key, _, _) = entries(&value).remove(0);
    std::fs::write(entry_path(&store, &key).join("scores.json"), b"{}").unwrap();
    let refused = verified(&loaded, &pin, Some(&store)).unwrap_err();
    let path = entry_path(&store, &key);
    assert_eq!(
        refused,
        ReferenceError::Unverified(format!(
            "cache entry {} differs from its seal",
            path.display()
        ))
    );

    // Entries absent from this cache are counted, not refused.
    std::fs::remove_dir_all(store.join("entries")).unwrap();
    std::fs::remove_dir_all(store.join("refsets")).unwrap();
    cached(&value, &store);
    std::fs::remove_dir_all(path).unwrap();
    let refset = value["binding"]["refset_sha256"].as_str().unwrap();
    std::fs::remove_file(store.join(format!("refsets/{refset}.json"))).unwrap();
    let binding = verified(&loaded, &pin, Some(&store)).unwrap().binding;
    assert_eq!(cache_of(&binding), Some(found(false, 1)));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn the_default_store_is_the_data_directory_cache() {
    let directory = scratch("default-store");
    let data = directory.display().to_string();
    let environment = |name: &str| (name == "CHOPINFER_DATA").then(|| data.clone());
    assert_eq!(references::default_store(&|_| None), None);
    assert_eq!(references::default_store(&|_| Some(String::new())), None);
    assert_eq!(references::default_store(&environment), None);
    std::fs::create_dir_all(directory.join("refcache/v1")).unwrap();
    assert_eq!(
        references::default_store(&environment),
        Some(directory.join("refcache/v1"))
    );
    std::fs::remove_dir_all(directory).unwrap();
}

/// What `cache.verify` reports for `value` against the cache at `store`:
/// its result, or its error's message. `None` without python3.
fn python_verify(value: &Path, store: &Path, workload: &Path) -> Option<Value> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let script = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from tools.refrun import cache
value = json.loads(Path(sys.argv[2]).read_bytes())
try:
    print(json.dumps(cache.verify(value, Path(sys.argv[1]), None, cache.Store(sys.argv[3]), sys.argv[4])))
except ValueError as error:
    print(json.dumps({"error": str(error)}))
"#;
    let output = Command::new("python3")
        .arg("-B")
        .arg("-c")
        .arg(script)
        .arg(&root)
        .arg(value)
        .arg(store)
        .arg(workload)
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Some(serde_json::from_slice(&output.stdout).unwrap())
}

#[test]
fn the_cache_comparison_is_what_python_reports() {
    let (loaded, pin, value) = bound();
    let workload = match &pin.workload {
        Some(Input::Path(path)) => loaded.root.join(path),
        _ => {
            let path = references::declared_path(
                &pin.path,
                &loaded.root,
                &references::process_environment,
            )
            .unwrap();
            let stem = path.file_stem().unwrap().to_str().unwrap().to_owned();
            path.with_file_name(format!("{stem}.workload.json"))
        }
    };
    let directory = scratch("refcache-python");
    let store = directory.join("refcache/v1");
    let fixture =
        references::declared_path(&pin.path, &loaded.root, &references::process_environment)
            .unwrap();
    let agree = |what: &str| {
        let Some(python) = python_verify(&fixture, &store, &workload) else {
            eprintln!("python3 is missing; skipping");
            return;
        };
        let rust = match verified(&loaded, &pin, Some(&store)) {
            Ok(verified) => {
                let mut binding = verified.binding.to_json();
                binding.as_object_mut().unwrap().remove("unchecked");
                binding
            }
            Err(error) => json!({"error": error.to_string()}),
        };
        assert_eq!(rust, python, "{what}");
    };
    cached(&value, &store);
    agree("every entry cached");
    let (_, _, key, _, _) = entries(&value).pop().unwrap();
    std::fs::remove_dir_all(entry_path(&store, &key)).unwrap();
    let refset = value["binding"]["refset_sha256"].as_str().unwrap();
    std::fs::remove_file(store.join(format!("refsets/{refset}.json"))).unwrap();
    agree("an entry and the refset missing");
    change_a_score(&value, &store);
    agree("a score changed");
    std::fs::write(store.join(format!("refsets/{refset}.json")), b"{}").unwrap();
    std::fs::remove_dir_all(store.join("entries")).unwrap();
    agree("a refset unlike its digest");
    std::fs::remove_dir_all(directory).unwrap();
}

/// The first shipped model (by directory) whose own declaration pins one
/// checkpoint lock (no registry) and a bound teacher reference whose
/// workload is in the repository, with that declaration and the reference.
fn rehomed() -> (declaration::Loaded, Value) {
    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let mut directories: Vec<PathBuf> = std::fs::read_dir(models)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|directory| directory.join(declaration::FILE_NAME).is_file())
        .collect();
    directories.sort();
    for directory in directories {
        let loaded = declaration::load(&directory).unwrap();
        let Some(Input::Path(path)) = loaded
            .declaration
            .references
            .iter()
            .find(|pin| pin.name == "teacher" && pin.workload.is_none())
            .map(|pin| pin.path.clone())
        else {
            continue;
        };
        let (_, value) = runtime_eval::json::read(&loaded.root.join(&path)).unwrap();
        if loaded.declaration.checkpoint.registry.is_none() && value.get("binding").is_some() {
            return (loaded, value);
        }
    }
    panic!("no shipped model pins a bound teacher reference with its workload in the repository");
}

#[test]
fn an_external_variant_s_bound_reference_verifies_under_its_root() {
    let (base, mut fixture) = rehomed();
    let model = base.declaration.model.clone();
    let root = scratch("external");
    let directory = root.join(&model);
    std::fs::create_dir_all(&directory).unwrap();
    // The model's checkpoint lock as the variant's, beside its overlay, and
    // the model's teacher reference re-homed onto it: its binding names the
    // copy under the variant root, which the binding check reads.
    std::fs::copy(
        base.root.join(&base.declaration.checkpoint.lock),
        directory.join("requant.checkpoint-files.json"),
    )
    .unwrap();
    let spec = base.value["references"]["teacher"].clone();
    let source = base.root.join(spec["path"].as_str().unwrap());
    fixture["binding"]["lock"]["path"] = json!(format!("{model}/requant.checkpoint-files.json"));
    let bytes = serde_json::to_vec(&fixture).unwrap();
    std::fs::write(directory.join("requant.teacher-reference.json"), &bytes).unwrap();
    let workload = source.with_file_name(format!(
        "{}.workload.json",
        source.file_stem().unwrap().to_str().unwrap()
    ));
    std::fs::copy(
        workload,
        directory.join("requant.teacher-reference.workload.json"),
    )
    .unwrap();
    let mut pinned = spec.clone();
    pinned["path"] = json!("requant.teacher-reference.json");
    pinned["sha256"] = json!(sha256_hex(&bytes));
    let mut checkpoint = base.value["qualification"]["checkpoint"].clone();
    checkpoint["lock"] = json!("requant.checkpoint-files.json");
    let mut overlay = json!({"schema": "chopinfer-qual-variant/v1", "model": model,
                             "checkpoint": "REQUANT", "checks": ["teacher"], "quick": ["teacher"],
                             "media": false, "references": {"teacher": pinned},
                             "qualification": {"checkpoint": checkpoint}});
    let mut inputs = serde_json::Map::new();
    for (name, location) in base.value["inputs"].as_object().unwrap() {
        if *location == spec["path"] {
            inputs.insert(name.clone(), json!("requant.teacher-reference.json"));
        }
    }
    overlay["inputs"] = Value::Object(inputs);
    if let Some(capabilities) = base.value["capabilities"].as_object() {
        let narrowed: serde_json::Map<String, Value> = capabilities
            .iter()
            .filter(|(key, _)| *key != "unqualified_media")
            .map(|(key, value)| {
                let value = match key.as_str() {
                    "text" => value.clone(),
                    "media" => json!([]),
                    "speculation" => json!("none"),
                    _ => json!(false),
                };
                (key.clone(), value)
            })
            .collect();
        overlay["capabilities"] = Value::Object(narrowed);
    }
    let write = |overlay: &Value| {
        std::fs::write(
            directory.join("requant.json"),
            serde_json::to_vec(overlay).unwrap(),
        )
        .unwrap()
    };
    write(&overlay);
    let model_directory = base.root.join("models").join(&model);
    let loaded =
        declaration::load_variant_from(&model_directory, Some("requant"), Some(&root)).unwrap();
    references::checkpoint_lock(&loaded).unwrap();
    let pin = &loaded.declaration.references[0];
    let verified = references::pinned(&loaded, pin, None).unwrap();
    assert_eq!(
        verified.path,
        std::fs::canonicalize(&root)
            .unwrap()
            .join(&model)
            .join("requant.teacher-reference.json")
    );
    match &verified.binding {
        Binding::Bound { unchecked, .. } => assert!(unchecked.is_empty()),
        other => panic!("{other:?}"),
    }
    // The base's reference, kept, is bound to the model's own lock, not the pin's.
    overlay["references"] = json!({});
    overlay["keep_references"] = json!(["teacher"]);
    overlay.as_object_mut().unwrap().shift_remove("inputs");
    write(&overlay);
    let kept =
        declaration::load_variant_from(&model_directory, Some("requant"), Some(&root)).unwrap();
    let pin = &kept.declaration.references[0];
    match references::pinned(&kept, pin, None) {
        Err(ReferenceError::Stale(field, _)) => assert_eq!(field, "lock"),
        other => panic!("{other:?}"),
    }
    std::fs::remove_dir_all(root).unwrap();
}
