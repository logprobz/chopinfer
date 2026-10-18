//! External variants: an overlay kept outside the tree, expanded against a
//! model's qual.json, with its lock, checkpoint lock and references beside
//! it under the variant root.

use std::path::{Path, PathBuf};

use runtime_eval::declaration::{self, Input, VARIANTS_ENV};
use runtime_eval::json::{canonical_sha256, sha256_hex};
use runtime_eval::{lock, references};
use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn base() -> Value {
    let bytes = std::fs::read(root().join("crates/runtime-eval/tests/data/declaration-cases.json"))
        .unwrap();
    runtime_eval::json::parse(&bytes).unwrap()["base"].clone()
}

/// A tree (`TREE/models/example/qual.json`, the vectors' base) and a
/// variant root holding the overlay `requant` of it, its checkpoint lock,
/// an unbound teacher reference and the reference lock.
struct Scratch {
    directory: PathBuf,
}

impl Scratch {
    fn new(name: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "runtime-eval-external-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&directory);
        std::fs::create_dir_all(directory.join("tree/models/example")).unwrap();
        std::fs::create_dir_all(directory.join("tree/tools/qual")).unwrap();
        std::fs::create_dir_all(directory.join("variants/example")).unwrap();
        std::fs::copy(
            root().join("tools/qual/modules.json"),
            directory.join("tree/tools/qual/modules.json"),
        )
        .unwrap();
        let scratch = Self { directory };
        scratch.write_base(&base());
        let lock = json!({"config.json": {"byte_length": 2, "sha256": "a".repeat(64)}});
        scratch.write("variants/example/checkpoint-files.json", &lock);
        let reference = json!({"schema": "example.teacher.v1",
                               "provenance": {"checkpoint": {"revision": "def"}, "variants": ["exact", "bf16"]},
                               "texts": [{"name": "t", "score_from": 1, "prompt_token_ids": [1, 2]}]});
        scratch.write("variants/example/teacher.json", &reference);
        scratch.write("variants/example/requant.json", &scratch.overlay());
        scratch.write(
            "variants/example/requant.lock.json",
            &json!({"schema": "chopinfer-qual-references/v1", "model": "example",
                    "entries": {}, "objects": {}}),
        );
        scratch
    }

    fn write(&self, relative: &str, value: &Value) {
        std::fs::write(
            self.directory.join(relative),
            serde_json::to_vec_pretty(value).unwrap(),
        )
        .unwrap();
    }

    fn write_base(&self, value: &Value) {
        self.write("tree/models/example/qual.json", value);
    }

    fn model(&self) -> PathBuf {
        self.directory.join("tree/models/example")
    }

    fn variants(&self) -> PathBuf {
        self.directory.join("variants")
    }

    fn file(&self, relative: &str) -> Vec<u8> {
        std::fs::read(self.directory.join(relative)).unwrap()
    }

    /// The overlay: another checkpoint of the model, pinned by its own lock,
    /// with its own teacher reference and the teacher and api checks.
    fn overlay(&self) -> Value {
        let lock: Value =
            runtime_eval::json::parse(&self.file("variants/example/checkpoint-files.json"))
                .unwrap();
        let reference = self.file("variants/example/teacher.json");
        json!({
            "schema": "chopinfer-qual-variant/v1",
            "model": "example",
            "checkpoint": "EXAMPLE_REQUANT",
            "checks": ["teacher", "api"],
            "quick": ["teacher"],
            "references": {"teacher": {"path": "teacher.json", "sha256": sha256_hex(&reference),
                                       "schema": "example.teacher.v1", "variants": ["exact", "bf16"],
                                       "checkpoint_keys": ["revision"]}},
            "inputs": {"fixture_root": "fixtures"},
            "qualification": {"checkpoint": {
                "lock": "checkpoint-files.json", "repository": "someone/example-requant",
                "revision": "def", "served_model_name": "Example", "source_lock_sha256": canonical_sha256(&lock),
                "checkpoint_identity": "e".repeat(64)}},
        })
    }

    fn load(&self) -> Result<declaration::Loaded, declaration::DeclarationError> {
        declaration::load_variant_from(&self.model(), Some("requant"), Some(&self.variants()))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn an_external_variant_resolves_what_it_owns_beside_its_overlay() {
    let scratch = Scratch::new("resolves");
    let loaded = scratch.load().unwrap();
    let variants = std::fs::canonicalize(scratch.variants()).unwrap();
    assert_eq!(loaded.declaration.variant.as_deref(), Some("requant"));
    assert_eq!(loaded.variants.as_deref(), Some(variants.as_path()));
    assert_eq!(loaded.lock_root(), variants.as_path());
    // Its checkpoint lock is under the variant root; its references and
    // inputs are environment paths under it, whatever the process sets.
    assert_eq!(
        loaded.declaration.checkpoint.lock,
        "example/checkpoint-files.json"
    );
    let teacher = &loaded.declaration.references[0];
    assert_eq!(
        teacher.path,
        Input::Environment {
            env: VARIANTS_ENV.into(),
            default: None,
            path: "example/teacher.json".into()
        }
    );
    assert_eq!(
        loaded.environment(VARIANTS_ENV).as_deref(),
        Some(variants.to_str().unwrap())
    );
    assert_eq!(
        references::input_path(&loaded, "fixture_root", &|name| loaded.environment(name)),
        Some(variants.join("example/fixtures"))
    );
    references::checkpoint_lock(&loaded).unwrap();
    let verified = references::pinned(&loaded, teacher, None).unwrap();
    assert_eq!(verified.path, variants.join("example/teacher.json"));
    assert_eq!(
        loaded.value["variant_source"],
        json!({"overlay": "example/requant.json",
               "sha256": sha256_hex(&scratch.file("variants/example/requant.json"))})
    );
    // The lock is beside the overlay, and receipts record all three files.
    assert_eq!(
        lock::path(&loaded),
        variants.join("example/requant.lock.json")
    );
    let source = lock::variant_source(&loaded).unwrap().unwrap();
    assert_eq!(source["root"], variants.display().to_string());
    assert_eq!(source["overlay"]["path"], "example/requant.json");
    assert_eq!(
        source["lock"],
        json!({"path": "example/requant.lock.json",
               "sha256": sha256_hex(&scratch.file("variants/example/requant.lock.json"))})
    );
    assert_eq!(
        source["checkpoint_lock"],
        json!({"path": "example/checkpoint-files.json",
               "sha256": sha256_hex(&scratch.file("variants/example/checkpoint-files.json"))})
    );
    // A declaration in the tree has none.
    let tree =
        declaration::load_variant_from(&scratch.model(), None, Some(&scratch.variants())).unwrap();
    assert_eq!(lock::variant_source(&tree).unwrap(), None);
}

#[test]
fn a_profile_rule_an_external_variant_adds_cites_its_documentation() {
    let scratch = Scratch::new("profile");
    let mut base = base();
    base["checks"]["api"]["rules"].as_array_mut().unwrap().push(
        json!({"path": ["server", "health", "serving", "profile", "kv_cache_dtype"],
                     "op": "eq", "value": "fp8", "source": "docs/example.md"}),
    );
    let mut overlay = scratch.overlay();
    overlay["serving"] = json!({"profile_name": "tp4-requant"});
    let expanded = declaration::external(&base, "requant", &overlay, &"0".repeat(64)).unwrap();
    let modules = declaration::Modules::read(&root()).unwrap();
    declaration::validate(&expanded, &modules).unwrap();
    let added: Vec<_> = expanded["checks"]["api"]["rules"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|rule| rule["path"] == json!(["server", "health", "serving", "profile", "name"]))
        .collect();
    assert_eq!(
        added,
        [
            &json!({"path": ["server", "health", "serving", "profile", "name"], "op": "eq",
                  "value": "tp4-requant", "source": declaration::VARIANT_SOURCE})
        ]
    );
}

#[test]
fn an_external_variant_keeps_no_base_reference_unless_it_names_it() {
    let scratch = Scratch::new("keep");
    let loaded = scratch.load().unwrap();
    let names: Vec<_> = loaded
        .declaration
        .references
        .iter()
        .map(|pin| pin.name.as_str())
        .collect();
    assert_eq!(names, ["teacher"]);
    let mut overlay = scratch.overlay();
    overlay["keep_references"] = json!(["media"]);
    scratch.write("variants/example/requant.json", &overlay);
    let loaded = scratch.load().unwrap();
    let mut names: Vec<_> = loaded
        .declaration
        .references
        .iter()
        .map(|pin| pin.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["media", "teacher"]);
    // A kept reference is still held to the variant's checkpoint pin.
    let media = loaded
        .declaration
        .references
        .iter()
        .find(|pin| pin.name == "media")
        .unwrap();
    assert_eq!(
        media.path,
        Input::Path("fixtures/example/media.json".into())
    );
}

#[test]
fn an_external_variant_needs_its_lock() {
    let scratch = Scratch::new("lock");
    std::fs::remove_file(scratch.directory.join("variants/example/requant.lock.json")).unwrap();
    let loaded = scratch.load().unwrap();
    let error = lock::load(&loaded).unwrap_err();
    assert!(error.contains("reference lock is missing"), "{error}");
    let input = scratch
        .variants()
        .join("example/fixtures/teacher-texts.json");
    assert!(lock::read(&loaded, "fixture_root", &input).is_err());
    assert!(lock::variant_source(&loaded).is_err());
}

#[test]
fn an_external_variant_cannot_change_rules_waivers_or_serving() {
    let scratch = Scratch::new("refused");
    let base = base();
    let refused = |field: &str, value: Value, expected: &str| {
        let mut overlay = scratch.overlay();
        overlay[field] = value;
        let error = declaration::external(&base, "requant", &overlay, &"0".repeat(64))
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{field}: {error}");
    };
    let cannot = "an external variant cannot state it";
    for field in [
        "rules",
        "append",
        "waivers",
        "cohort_args",
        "environment",
        "install_features",
        "producer",
    ] {
        refused(field, json!({}), &format!("{field}: {cannot}"));
    }
    for field in [
        "serve",
        "tensor_parallel_size",
        "screens",
        "source",
        "experimental",
        "change_classes",
        "anything",
    ] {
        refused(field, json!([]), &format!("{field}: unknown field"));
    }
    refused(
        "serving",
        json!({"kv_cache_dtype": "bf16"}),
        "serving.kv_cache_dtype",
    );
    refused(
        "serving",
        json!({"checkpoint_identity": "f".repeat(64)}),
        "serving.checkpoint_identity",
    );
    refused(
        "schema",
        json!("chopinfer-qual-model/v1"),
        "schema: must be",
    );
    refused("model", json!("other"), "model: must be example");
    refused(
        "checks",
        json!(["teacher", "undeclared"]),
        "names undeclared check undeclared",
    );
    refused(
        "inputs",
        json!({"invented": "x.json"}),
        "inputs.invented: is not an input the base declares",
    );
    refused(
        "inputs",
        json!({"fixture_root": "../outside"}),
        "inputs.fixture_root: must be a relative path",
    );
    refused(
        "inputs",
        json!({"fixture_root": "/abs"}),
        "inputs.fixture_root: must be a relative path",
    );
    refused(
        "qualification",
        json!({"checkpoint": scratch.overlay()["qualification"]["checkpoint"], "prefix": {}}),
        "qualification.prefix: unknown field",
    );
    let mut overlay = scratch.overlay();
    overlay["qualification"]["checkpoint"]["registry"] = json!("models/example/registry.json");
    let error = declaration::external(&base, "requant", &overlay, &"0".repeat(64)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("qualification.checkpoint.registry"),
        "{error}"
    );
    for field in [
        "schema",
        "model",
        "checkpoint",
        "checks",
        "quick",
        "qualification",
    ] {
        let mut overlay = scratch.overlay();
        overlay.as_object_mut().unwrap().shift_remove(field);
        let error = declaration::external(&base, "requant", &overlay, &"0".repeat(64)).unwrap_err();
        assert!(
            error.to_string().contains(&format!("{field}: is required")),
            "{error}"
        );
    }
}

#[test]
fn an_external_variant_only_narrows_the_base_capabilities() {
    let scratch = Scratch::new("capabilities");
    let mut base = base();
    base["capabilities"] = json!({"text": true, "frontend": false, "api": true, "structured": false,
                                  "prefix_cache": false, "preemption": false, "long_context": false,
                                  "stability": false, "media": [], "speculation": "none",
                                  "kv_offload": false});
    let mut overlay = scratch.overlay();
    overlay["capabilities"] = base["capabilities"].clone();
    overlay["capabilities"]["api"] = json!(false);
    declaration::external(&base, "requant", &overlay, &"0".repeat(64)).unwrap();
    for (field, value) in [
        ("structured", json!(true)),
        ("media", json!(["image"])),
        ("speculation", json!("mtp")),
    ] {
        let mut wider = overlay.clone();
        wider["capabilities"][field] = value;
        let error = declaration::external(&base, "requant", &wider, &"0".repeat(64)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("capabilities.{field}: must not claim more")),
            "{error}"
        );
    }
    let mut missing = overlay.clone();
    missing["capabilities"]
        .as_object_mut()
        .unwrap()
        .shift_remove("text");
    let error = declaration::external(&base, "requant", &missing, &"0".repeat(64)).unwrap_err();
    assert!(
        error.to_string().contains("capabilities.text: is required"),
        "{error}"
    );
    // What it keeps, its checks must still cover.
    let mut uncovered = overlay.clone();
    uncovered["checks"] = json!(["api"]);
    uncovered["quick"] = json!(["api"]);
    let expanded = declaration::external(&base, "requant", &uncovered, &"0".repeat(64)).unwrap();
    let modules = declaration::Modules::read(&root()).unwrap();
    let error = declaration::validate(&expanded, &modules).unwrap_err();
    assert!(error.to_string().contains("capabilities.text"), "{error}");
}

#[test]
fn an_external_variant_cannot_take_a_name_the_tree_declares() {
    let scratch = Scratch::new("names");
    let mut value = base();
    value["configurations"] = json!({"requant": {"checks": ["teacher"], "quick": ["teacher"]}});
    let error =
        declaration::external(&value, "requant", &scratch.overlay(), &"0".repeat(64)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("is a configuration the model's qual.json declares"),
        "{error}"
    );
    let error =
        declaration::external(&base(), "default", &scratch.overlay(), &"0".repeat(64)).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("variant: must be a plain name other than default"),
        "{error}"
    );
    // An overlay beside a configuration or variant file of its name is refused.
    scratch.write_base(&value);
    let error = scratch.load().unwrap_err().to_string();
    assert!(error.contains("is a variant the tree declares"), "{error}");
    scratch.write_base(&base());
    let mut file = base();
    file["variant"] = json!("requant");
    scratch.write("tree/models/example/qual.requant.json", &file);
    let error = scratch.load().unwrap_err().to_string();
    assert!(error.contains("is a variant the tree declares"), "{error}");
    std::fs::remove_file(
        scratch
            .directory
            .join("tree/models/example/qual.requant.json"),
    )
    .unwrap();
    scratch.load().unwrap();
    // A declaration in the tree cannot claim an overlay.
    let mut claimed = base();
    claimed["variant_source"] =
        json!({"overlay": "example/requant.json", "sha256": "0".repeat(64)});
    scratch.write_base(&claimed);
    let modules = declaration::Modules::read(&root()).unwrap();
    let error = declaration::validate(&claimed, &modules).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("variant_source: names the overlay of a variant"),
        "{error}"
    );
}
