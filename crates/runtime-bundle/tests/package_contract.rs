use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use runtime_bundle::{
    BundleArtifactOriginV1, BundleBatchLimitsV1, BundleHostCompatibilityV1, BundleManifestV1,
    BundleOpaqueId, BundlePackageV1, BundleSpeculationV1, VerifiedComponentFileV1,
};
use runtime_kernel_contract::KernelResolutionErrorV1;
use runtime_weights::sha256_bytes;
use serde_json::{Value, json};

const ENTRY_SYMBOL: &str = "chop_bundle_get_api_v1";
const COMPONENT_BYTES: &[u8] = b"fixture-weight-bytes";
const TARGET_ARCHITECTURE: &str = "sm_120f";
const OTHER_ARCHITECTURE: &str = "sm_90a";
const OTHER_COMPONENT_BYTES: &[u8] = b"fixture-weight-bytes-for-sm90a";

#[test]
fn batch_limits_round_trip_and_validate() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    let original = canonical_json(&value);
    assert!(
        !String::from_utf8(original.clone())
            .unwrap()
            .contains("batch_limits")
    );
    assert_eq!(
        BundleManifestV1::from_canonical_json(&original)
            .unwrap()
            .batch_limits(),
        None
    );
    value["batch_limits"] = json!({"maximum_sequences":32, "maximum_prefill_tokens":8192});
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    let limits = manifest.batch_limits().unwrap();
    assert_eq!(
        (limits.maximum_sequences(), limits.maximum_prefill_tokens()),
        (32, 8192)
    );
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    // The package maximum is the most any package may declare.
    value["batch_limits"] = json!({"maximum_sequences":256, "maximum_prefill_tokens":8192});
    let manifest = BundleManifestV1::from_canonical_json(&canonical_json(&value)).unwrap();
    assert_eq!(
        manifest.batch_limits().unwrap().maximum_sequences(),
        BundleBatchLimitsV1::MAXIMUM_SEQUENCES
    );
    assert!(BundleBatchLimitsV1::new(257, 8192).is_err());
    for invalid in [
        json!({"maximum_sequences":257, "maximum_prefill_tokens":8192}),
        json!({"maximum_sequences":0, "maximum_prefill_tokens":8192}),
        json!({"maximum_sequences":32, "maximum_prefill_tokens":0}),
        json!({"maximum_sequences":32}),
    ] {
        value["batch_limits"] = invalid;
        assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
    }
}

/// `chop.state-per-item.v1`: the declared state bytes are per state slot, and
/// an instance serving `n` items holds `n` rounded up to a power of two
/// slots. Packages without the capability keep their bytes whatever they
/// serve.
#[test]
fn per_item_state_scales_with_the_served_slots() {
    use runtime_bundle::{
        SERVED_ITEMS_CAPABILITY_V1, STATE_PER_ITEM_CAPABILITY_V1, state_slots_v1,
    };
    assert_eq!(
        [1, 2, 3, 32, 33, 64, 100, 256].map(state_slots_v1),
        [1, 2, 4, 32, 64, 64, 128, 256]
    );
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    value["batch_limits"] = json!({"maximum_sequences":64, "maximum_prefill_tokens":8192});
    // Without the capability (with or without the served count) the state
    // stays at its declared 64 bytes.
    for capabilities in [json!([]), json!([SERVED_ITEMS_CAPABILITY_V1])] {
        value["resource_requirements"]["required_capabilities"] = capabilities;
        let manifest = BundleManifestV1::from_canonical_json(&canonical_json(&value)).unwrap();
        assert!(!manifest.state_per_item());
        for served in [None, Some(1), Some(33), Some(64), Some(1000)] {
            assert_eq!(manifest.state_bytes_v1(served).unwrap(), 64);
        }
    }
    // The capability requires the served count.
    value["resource_requirements"]["required_capabilities"] = json!([STATE_PER_ITEM_CAPABILITY_V1]);
    assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
    value["resource_requirements"]["required_capabilities"] =
        json!([SERVED_ITEMS_CAPABILITY_V1, STATE_PER_ITEM_CAPABILITY_V1]);
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    assert!(manifest.state_per_item());
    assert_eq!(manifest.state_layout().bytes(), 64);
    for (served, slots) in [(1, 1), (3, 4), (32, 32), (33, 64), (64, 64)] {
        assert_eq!(manifest.state_bytes_v1(Some(served)).unwrap(), 64 * slots);
    }
    // The count must be named and within the declared maximum.
    for served in [None, Some(0), Some(65)] {
        assert!(manifest.state_bytes_v1(served).is_err());
    }
    // Without batch limits the state has no size.
    value.as_object_mut().unwrap().remove("batch_limits");
    let manifest = BundleManifestV1::from_canonical_json(&canonical_json(&value)).unwrap();
    assert!(manifest.state_bytes_v1(Some(1)).is_err());
    // A per-slot size that overflows at the maximum is refused.
    value["batch_limits"] = json!({"maximum_sequences":256, "maximum_prefill_tokens":8192});
    value["state_layout"]["bytes"] = json!(u64::MAX / 128 / 64 * 64);
    assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
}

#[test]
fn request_state_layouts_round_trip_and_validate() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    let original = canonical_json(&value);
    assert!(
        !String::from_utf8(original.clone())
            .unwrap()
            .contains("recurrent_state_layouts")
    );
    assert_eq!(
        BundleManifestV1::from_canonical_json(&original)
            .unwrap()
            .recurrent_state_bytes_per_slot()
            .unwrap(),
        0
    );
    value["recurrent_state_layouts"] = json!([
        {"id":"conv", "bytes":128, "alignment_bytes":64, "lifetime":"request"},
        {"id":"delta", "bytes":256, "alignment_bytes":128, "lifetime":"request"}
    ]);
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    assert_eq!(manifest.recurrent_state_layouts().len(), 2);
    assert_eq!(manifest.recurrent_state_bytes_per_slot().unwrap(), 384);
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    for invalid in [
        json!([{"id":"a", "bytes":128, "alignment_bytes":3, "lifetime":"request"}]),
        json!([{"id":"a", "bytes":128, "alignment_bytes":64, "lifetime":"process"}]),
        json!([
            {"id":"a", "bytes":128, "alignment_bytes":64, "lifetime":"request"},
            {"id":"a", "bytes":128, "alignment_bytes":64, "lifetime":"request"}
        ]),
        json!([
            {"id":"a", "bytes":18446744073709551600_u64, "alignment_bytes":16, "lifetime":"request"},
            {"id":"b", "bytes":128, "alignment_bytes":64, "lifetime":"request"}
        ]),
    ] {
        value["recurrent_state_layouts"] = invalid;
        assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
    }
}

#[test]
fn speculation_manifest_round_trips_in_canonical_order() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    assert!(
        value.get("speculation").is_none(),
        "older manifests omit the new key"
    );
    value["resource_requirements"]["required_capabilities"] = json!(["chop.speculative.v1"]);
    value["speculation"] = json!({
        "maximum_draft_tokens": 7,
        "bundle_drafter": true,
        "maximum_verify_tokens": 32,
        "row_buckets": [2, 4, 8],
        "drafter_component": "weights"
    });
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    let spec = manifest.speculation().unwrap();
    assert_eq!(spec.maximum_draft_tokens(), 7);
    assert_eq!(spec.row_buckets(), &[2, 4, 8]);
    let built = BundleSpeculationV1::new(
        7,
        true,
        32,
        vec![8, 2, 4],
        Some(BundleOpaqueId::new("weights").unwrap()),
    )
    .unwrap();
    assert_eq!(built, *spec);
}

/// The optional drafting-items bound round-trips canonically, stays absent
/// from manifests without it, and refuses zero and an explicit null.
#[test]
fn speculation_drafting_item_bound_is_canonical_and_positive() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    value["resource_requirements"]["required_capabilities"] = json!(["chop.speculative.v1"]);
    value["speculation"] = json!({
        "maximum_draft_tokens": 7,
        "bundle_drafter": true,
        "maximum_verify_tokens": 32,
        "row_buckets": [2, 4, 8],
        "drafter_component": "weights"
    });
    let absent = BundleManifestV1::from_canonical_json(&canonical_json(&value)).unwrap();
    assert_eq!(absent.speculation().unwrap().maximum_draft_items(), None);
    assert!(
        !String::from_utf8(absent.to_canonical_json().unwrap())
            .unwrap()
            .contains("maximum_draft_items")
    );
    value["speculation"]["maximum_draft_items"] = json!(32);
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    let spec = manifest.speculation().unwrap();
    assert_eq!(spec.maximum_draft_items(), Some(32));
    let built = BundleSpeculationV1::new(
        7,
        true,
        32,
        vec![2, 4, 8],
        Some(BundleOpaqueId::new("weights").unwrap()),
    )
    .unwrap();
    assert!(built.clone().with_maximum_draft_items(0).is_err());
    assert_eq!(built.with_maximum_draft_items(32).unwrap(), *spec);
    for refused in [json!(0), Value::Null] {
        value["speculation"]["maximum_draft_items"] = refused.clone();
        assert!(
            BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err(),
            "{refused}"
        );
    }
}

#[test]
fn speculation_manifest_refuses_missing_capability_bad_order_and_unknown_drafter() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    value["speculation"] = json!({
        "maximum_draft_tokens": 3,
        "bundle_drafter": true,
        "maximum_verify_tokens": 8,
        "row_buckets": [2, 4],
        "drafter_component": "weights"
    });
    assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
    value["resource_requirements"]["required_capabilities"] = json!(["chop.speculative.v1"]);
    value["speculation"]["row_buckets"] = json!([4, 2]);
    assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
    value["speculation"]["row_buckets"] = json!([2, 4]);
    value["speculation"]["drafter_component"] = json!("missing");
    assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err());
}

/// A planned component: a compiled image the weight plan supplies, declared
/// by role and format without a file.
fn planned_manifest(manifest: &mut Value) {
    manifest["resource_requirements"]["required_capabilities"] =
        json!([runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1]);
    manifest["planned_components"] = json!([{
        "id": "planned-image",
        "role": "fixture.planned-image.v1",
        "format": "fixture.image.v1"
    }]);
    manifest["graph_buckets"][0]["components"] = json!(["planned-image", "weights"]);
}

/// A planned component round-trips canonically, owns no file, and may be
/// named by graph buckets and the speculation drafter; manifests without
/// one carry no such key.
#[test]
fn a_planned_component_declares_an_image_without_a_file() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    let plain = BundleManifestV1::from_canonical_json(&canonical_json(&value)).unwrap();
    assert!(plain.planned_components().is_empty());
    assert!(
        !String::from_utf8(plain.to_canonical_json().unwrap())
            .unwrap()
            .contains("planned_components")
    );
    planned_manifest(&mut value);
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    let planned = &manifest.planned_components()[0];
    assert_eq!(
        (
            planned.id().as_str(),
            planned.role().as_str(),
            planned.format().as_str()
        ),
        (
            "planned-image",
            "fixture.planned-image.v1",
            "fixture.image.v1"
        )
    );
    assert_eq!(manifest.components().len(), 1);
    value["resource_requirements"]["required_capabilities"] = json!([
        "chop.speculative.v1",
        runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1
    ]);
    value["speculation"] = json!({
        "maximum_draft_tokens": 3,
        "bundle_drafter": true,
        "maximum_verify_tokens": 8,
        "row_buckets": [2, 4],
        "drafter_component": "planned-image"
    });
    assert!(BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_ok());

    // The package directory holds no file for it and verifies as it is.
    fixture.rewrite_manifest(planned_manifest);
    let planning_host = host().with_capabilities(vec![
        BundleOpaqueId::new(runtime_bundle::WEIGHT_PLAN_CAPABILITY_V1).unwrap(),
    ]);
    let package = BundlePackageV1::verify_directory(fixture.package(), &planning_host).unwrap();
    assert_eq!(package.manifest().planned_components().len(), 1);
}

#[test]
fn a_planned_component_needs_the_plan_capability_and_its_own_id_and_role() {
    let fixture = Fixture::new();
    let refused = |change: fn(&mut Value)| {
        let mut value = fixture.manifest();
        planned_manifest(&mut value);
        change(&mut value);
        BundleManifestV1::from_canonical_json(&canonical_json(&value)).is_err()
    };
    assert!(!refused(|_| {}));
    assert!(refused(|value| {
        value["resource_requirements"]["required_capabilities"] = json!([]);
    }));
    assert!(refused(|value| {
        value["planned_components"][0]["id"] = json!("weights");
        value["graph_buckets"][0]["components"] = json!(["weights"]);
    }));
    assert!(refused(|value| {
        value["planned_components"][0]["role"] = value["components"][0]["role"].clone();
    }));
    assert!(refused(|value| {
        value["graph_buckets"][0]["components"] = json!(["weights"]);
    }));
    assert!(refused(|value| {
        value["planned_components"][0]["path"] = json!("components/planned.bin");
    }));
}

/// A named mutation applied to the on-disk manifest.
type ManifestDrift = (&'static str, fn(&mut Value));

#[test]
fn a_verified_component_read_serves_the_verified_bytes_after_the_file_changes() {
    let fixture = Fixture::new();
    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("canonical fixture package verifies");
    let component = package
        .manifest()
        .components()
        .first()
        .expect("the fixture declares a component")
        .clone();

    assert_eq!(
        package
            .read_component(&component)
            .expect("a verified component reads back"),
        COMPONENT_BYTES
    );
    assert!(
        package
            .read_native_module()
            .expect("a verified native module reads back")
            .is_some()
    );

    // The same length with different bytes, then a different length: the
    // package serves what it verified, and a new verification refuses both.
    let mut swapped = COMPONENT_BYTES.to_vec();
    swapped[0] = swapped[0].wrapping_add(1);
    fs::write(fixture.component(), &swapped).expect("fixture component is writable");
    assert_eq!(package.read_component(&component).unwrap(), COMPONENT_BYTES);
    reject(&fixture);

    fs::write(fixture.component(), b"a different length entirely").expect("component is writable");
    assert_eq!(package.read_component(&component).unwrap(), COMPONENT_BYTES);
    reject(&fixture);
}

#[cfg(unix)]
#[test]
fn direct_verified_component_open_refuses_a_final_symlink() {
    let fixture = Fixture::new();
    let target = fixture.root().join("symlink-target.bin");
    fs::write(&target, COMPONENT_BYTES).expect("the symlink target is writable");
    fs::remove_file(fixture.component()).expect("the component is removable");
    std::os::unix::fs::symlink(&target, fixture.component())
        .expect("the final component symlink is creatable");

    assert!(
        VerifiedComponentFileV1::open(
            fixture.component(),
            COMPONENT_BYTES.len() as u64,
            &sha256_bytes(COMPONENT_BYTES),
        )
        .is_err(),
        "the descriptor open must atomically refuse the final symlink"
    );
}

#[test]
fn package_component_handles_do_not_follow_directory_entry_replacement() {
    let fixture = Fixture::new();
    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("canonical fixture package verifies");
    let component = package.manifest().components()[0].clone();
    let retained = package
        .open_component_file(&component)
        .expect("every verified component keeps its verified bytes");
    let original = fixture.root().join("original-component.bin");
    fs::rename(fixture.component(), &original).expect("the original entry is retainable");
    let replacement = vec![b'x'; COMPONENT_BYTES.len()];
    fs::write(fixture.component(), &replacement).expect("the replacement entry is writable");

    let mut from_handle = vec![0; COMPONENT_BYTES.len()];
    retained
        .read_exact_range(0, &mut from_handle)
        .expect("the verified bytes stay readable");
    assert_eq!(from_handle, COMPONENT_BYTES);
    assert_eq!(package.read_component(&component).unwrap(), COMPONENT_BYTES);
}

/// A package that still divides a component into rank images is refused when
/// it is verified, and the refusal says how to serve the model instead.
#[test]
fn a_package_with_compiled_images_is_refused_naming_serve_model() {
    let fixture = Fixture::new();
    fixture.rewrite_manifest(|manifest| {
        manifest["compiled_images"] = json!([{
            "component": "weights",
            "topology": "tp2",
            "alignment_bytes": 4,
            "ranks": [
                {"rank": 0, "offset": 0, "byte_length": 8},
                {"rank": 1, "offset": 8, "byte_length": 8},
            ],
            "views": [],
        }]);
    });

    let error = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect_err("a package carrying rank images must be refused");
    assert!(error.to_string().contains("serve --model DIR"), "{error}");
}

#[test]
fn a_native_symbol_sequence_keeps_the_order_it_declares() {
    let fixture = Fixture::new();
    fixture.rewrite_manifest(|manifest| {
        manifest["native_symbol_sequences"] = json!([
            {"id": "route.decode", "symbols": ["gather", "norm", "gemm"]},
            {"id": "route.prefill", "symbols": ["norm", "gather"]},
        ]);
    });

    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("a package with native symbol sequences verifies");
    let sequences = package.manifest().native_symbol_sequences();
    assert_eq!(sequences.len(), 2);
    assert_eq!(sequences[0].id().as_str(), "route.decode");
    // The names come back in declaration order, not sorted, because the order
    // is the whole content of the record.
    assert_eq!(
        sequences[0]
            .symbols()
            .iter()
            .map(BundleOpaqueId::as_str)
            .collect::<Vec<_>>(),
        ["gather", "norm", "gemm"]
    );
    // Two sequences are two orders over names that may overlap, so one name
    // appearing in both is a declaration rather than a conflict.
    assert_eq!(
        sequences[1]
            .symbols()
            .iter()
            .map(BundleOpaqueId::as_str)
            .collect::<Vec<_>>(),
        ["norm", "gather"]
    );
}

#[test]
fn a_manifest_that_declares_no_native_symbol_sequence_carries_no_such_key() {
    let fixture = Fixture::new();
    let before = fs::read(fixture.manifest_path()).expect("fixture manifest is readable");

    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("canonical fixture package verifies");
    let manifest = package.manifest();
    assert!(manifest.native_symbol_sequences().is_empty());
    assert_eq!(
        manifest
            .to_canonical_json()
            .expect("a verified manifest recanonicalizes"),
        before,
        "a manifest that declares no sequence must stay byte-identical"
    );
    assert!(
        !String::from_utf8(before)
            .expect("fixture manifest is UTF-8")
            .contains("native_symbol_sequences"),
        "an absent sequence list must not reach the canonical JSON"
    );
}

#[test]
fn a_native_symbol_sequence_that_is_not_a_closed_declaration_is_rejected() {
    let drifts: [ManifestDrift; 5] = [
        ("a sequence with no symbol", |manifest| {
            manifest["native_symbol_sequences"][0]["symbols"] = json!([]);
        }),
        ("one name in two positions of one sequence", |manifest| {
            manifest["native_symbol_sequences"][0]["symbols"] = json!(["gather", "gather"]);
        }),
        ("two sequences under one identifier", |manifest| {
            manifest["native_symbol_sequences"][1]["id"] = json!("route.decode");
        }),
        ("sequences out of canonical order", |manifest| {
            let sequences = manifest["native_symbol_sequences"]
                .as_array_mut()
                .expect("native_symbol_sequences is an array");
            sequences.swap(0, 1);
        }),
        ("an empty sequence list written out", |manifest| {
            manifest["native_symbol_sequences"] = json!([]);
        }),
    ];
    for (label, drift) in drifts {
        let fixture = Fixture::new();
        fixture.rewrite_manifest(|manifest| {
            manifest["native_symbol_sequences"] = json!([
                {"id": "route.decode", "symbols": ["gather", "norm", "gemm"]},
                {"id": "route.prefill", "symbols": ["norm", "gather"]},
            ]);
            drift(manifest);
        });

        assert!(
            BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
            "{label} must be rejected"
        );
    }
}

#[test]
fn a_package_that_ships_no_code_cannot_declare_a_symbol_sequence() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.native_module()).expect("fixture native module is removable");
    fixture.rewrite_manifest(|manifest| {
        manifest
            .as_object_mut()
            .expect("fixture manifest is an object")
            .remove("native_module");
        manifest["native_symbol_sequences"] =
            json!([{"id": "route.decode", "symbols": ["gather"]}]);
    });

    reject(&fixture);
}

#[test]
fn two_components_cannot_claim_one_role() {
    let fixture = Fixture::new();
    fixture.rewrite_manifest(|manifest| {
        let components = manifest["components"]
            .as_array_mut()
            .expect("components is an array");
        let mut twin = components[0].clone();
        twin["id"] = json!("weights.twin");
        twin["path"] = json!("components/weights-twin.bin");
        components.push(twin);
    });
    fs::write(
        fixture.package().join("components/weights-twin.bin"),
        COMPONENT_BYTES,
    )
    .expect("the twin component is writable");
    fixture.rewrite_manifest(|manifest| {
        manifest["graph_buckets"][0]["components"] = json!(["weights", "weights.twin"]);
    });

    // A host resolves a component by role, so the pair has no defined answer.
    assert!(
        BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
        "two components sharing a role must be rejected"
    );
}

#[test]
fn package_without_a_native_module_is_valid() {
    let fixture = Fixture::new();
    fs::remove_file(fixture.native_module()).expect("fixture native module is removable");
    fixture.rewrite_manifest(|manifest| {
        manifest
            .as_object_mut()
            .expect("fixture manifest is an object")
            .remove("native_module");
    });

    verify(&fixture);
}

#[test]
fn rejects_unknown_schema_or_abi() {
    let unknown_schema = Fixture::new();
    unknown_schema.rewrite_manifest(|manifest| {
        manifest["schema"] = json!("chop-model-bundle-v2");
    });
    reject(&unknown_schema);

    let unknown_abi = Fixture::new();
    unknown_abi.rewrite_manifest(|manifest| {
        manifest["abi_version"] = json!(2);
    });
    reject(&unknown_abi);
}

#[test]
fn rejects_duplicate_component_ids_and_noncanonical_json() {
    let duplicate = Fixture::new();
    fs::write(
        duplicate.package().join("components/weights-copy.bin"),
        COMPONENT_BYTES,
    )
    .expect("duplicate fixture component is writable");
    duplicate.rewrite_manifest(|manifest| {
        let mut repeated = manifest["components"][0].clone();
        repeated["path"] = json!("components/weights-copy.bin");
        manifest["components"]
            .as_array_mut()
            .expect("fixture components are an array")
            .push(repeated);
    });
    reject(&duplicate);

    let noncanonical = Fixture::new();
    let canonical = fs::read(noncanonical.manifest_path()).expect("fixture manifest is readable");
    fs::write(
        noncanonical.manifest_path(),
        [b" ".as_slice(), canonical.as_slice()].concat(),
    )
    .expect("noncanonical fixture manifest is writable");
    reject(&noncanonical);
}

#[test]
fn rejects_component_traversal_and_symlinks() {
    let traversal = Fixture::new();
    traversal.rewrite_manifest(|manifest| {
        manifest["components"][0]["path"] = json!("../outside.bin");
    });
    reject(&traversal);

    #[cfg(unix)]
    {
        let symlink = Fixture::new();
        let outside = symlink.root().join("outside.bin");
        fs::write(&outside, COMPONENT_BYTES).expect("outside fixture component is writable");
        fs::remove_file(symlink.component()).expect("fixture component is removable");
        std::os::unix::fs::symlink(&outside, symlink.component())
            .expect("fixture component symlink is creatable");
        reject(&symlink);
    }
}

#[test]
fn rejects_missing_or_extra_files_and_digest_drift() {
    let missing = Fixture::new();
    fs::remove_file(missing.component()).expect("fixture component is removable");
    reject(&missing);

    let extra = Fixture::new();
    fs::write(extra.package().join("unexpected.bin"), b"unexpected")
        .expect("extra fixture file is writable");
    reject(&extra);

    let drift = Fixture::new();
    fs::write(drift.component(), b"mutated fixture weights")
        .expect("fixture component is writable");
    reject(&drift);
}

/// A manifest replaced after verification does not change what was verified.
///
/// The verified package keeps the manifest it was built from, and a later read
/// of the directory cannot reach back into it.
#[test]
fn a_manifest_replaced_after_verification_does_not_change_the_verified_package() {
    let fixture = Fixture::new();
    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("the fixture verifies");
    let verified = package.manifest().clone();

    let mut replaced = fixture.manifest_value();
    replaced["components"][0]["role"] = json!("fixture.replaced.v1");
    fixture.write_manifest(&replaced);

    assert_eq!(package.manifest(), &verified);
    assert_eq!(
        package.manifest().components()[0].role().as_str(),
        "fixture.weights.v1"
    );
    verify(&fixture);
    let reverified = BundlePackageV1::verify_directory(fixture.package(), &host()).unwrap();
    assert_eq!(
        reverified.manifest().components()[0].role().as_str(),
        "fixture.replaced.v1"
    );
}

#[test]
fn a_package_file_changed_after_verification_cannot_redirect_the_load() {
    for change in [PackageChange::ReplaceEntry, PackageChange::RewriteInode] {
        let fixture = Fixture::new();
        let package = BundlePackageV1::verify_directory(fixture.package(), &host())
            .expect("canonical fixture package verifies");
        let module = package
            .open_native_module()
            .expect("the declared native module opens")
            .expect("the fixture declares a native module");
        let verified = module.bytes().to_vec();

        // Either half of the race anything with write access to the install
        // root can run between a verification and a load that reads the file
        // again: swap the directory entry, or rewrite the inode it named.
        match change {
            PackageChange::ReplaceEntry => {
                let replacement = Fixture::new();
                replacement.write_native_module("fixture_bundle_other_entry");
                fs::rename(replacement.native_module(), fixture.native_module())
                    .expect("the fixture native module is replaceable");
            }
            PackageChange::RewriteInode => {
                let mut rewritten = verified.clone();
                let last = rewritten.len() - 1;
                rewritten[last] ^= 0xff;
                fs::write(fixture.native_module(), &rewritten)
                    .expect("the fixture native module is writable");
            }
        }

        assert_ne!(
            fs::read(fixture.native_module()).expect("the changed path is readable"),
            verified,
            "the fixture must actually have changed"
        );
        // This process retains the authenticated bytes. A new process must
        // reject the changed package when it verifies the directory.
        assert_eq!(
            package.read_native_module().unwrap(),
            Some(verified.clone())
        );
        assert!(BundlePackageV1::verify_directory(fixture.package(), &host()).is_err());
        assert_eq!(module.bytes(), verified.as_slice());
    }
}

#[derive(Clone, Copy)]
enum PackageChange {
    ReplaceEntry,
    RewriteInode,
}

#[test]
fn rejects_a_wrong_target_architecture_and_a_missing_native_entry_symbol() {
    let wrong_architecture = Fixture::new();
    wrong_architecture.rewrite_manifest(|manifest| {
        manifest["target"]["architectures"] = json!(["sm_999"]);
    });
    reject(&wrong_architecture);

    let missing_symbol = Fixture::new();
    missing_symbol.write_native_module("fixture_bundle_other_entry");
    missing_symbol.refresh_native_metadata();
    reject(&missing_symbol);
}

/// Declares one custom operation and its bundle implementation, anchoring the
/// implementation artifact to the package's own `weights` component.
fn declare_anchored_implementation(manifest: &mut Value) {
    manifest["operation_requirements"] = json!([{
        "key": {"operation": "fixture.projection", "variant": "decode-1"},
        "provisioning": {"kind": "bundle-custom-operation"},
        "semantic_contract": {
            "operation_abi": "fixture.projection.abi.v1",
            "operation_abi_version": 1,
            "tensors": [{
                "role": "output",
                "dimensions": [1, 4],
                "dtype": "bf16",
                "layout": "row-major.v1",
                "scale_format": "scale.none.v1",
                "scale_block_dimensions": [],
                "alignment_bytes": 16,
                "access": "write-only",
                "alias": {"kind": "disjoint"},
                "dependencies": []
            }],
            "scalar_arguments": [],
            "numerical_contract": "fixture.numerics.v1",
            "required_capabilities": []
        }
    }]);
    manifest["graph_buckets"][0]["operations"] =
        json!([{"operation": "fixture.projection", "variant": "decode-1"}]);
    manifest["implementations"] = json!([{
        "id": "bundle.projection.v1",
        "key": {"operation": "fixture.projection", "variant": "decode-1"},
        "tier": "bundle",
        "detail": {
            "artifact": {
                "origin": {"kind": "component", "component": "weights"},
                "format": "fixture.bytes.v1",
                "byte_length": COMPONENT_BYTES.len(),
                "sha256": sha256_bytes(COMPONENT_BYTES)
            },
            "symbol": "fixture_projection_v1",
            "target_architecture": TARGET_ARCHITECTURE,
            "launch": {"grid": [1, 1, 1], "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0},
            "specializations": [],
            "source": {
                "source": "fixture-source.v1",
                "revision": "0123456789abcdef",
                "path": "kernels.projection.cu",
                "byte_length": 2048,
                "sha256": "b".repeat(64)
            },
            "build": {
                "toolchain": "nvcc",
                "toolchain_version": "13.2",
                "build": "fixture-build.v1",
                "recipe_sha256": "c".repeat(64)
            }
        }
    }]);
}

/// Declares one custom operation implemented once per declared architecture.
///
/// Both records carry the same operation key and the same provider tier, and
/// each is its own exact implementation: its own component, symbol, source,
/// and build. This is what one logical bundle serving several CUDA
/// architectures declares, and it must verify while a second record for one
/// architecture stays invalid.
fn declare_per_architecture_implementations(manifest: &mut Value) {
    declare_anchored_implementation(manifest);
    manifest["target"]["architectures"] = json!(["sm_120f", OTHER_ARCHITECTURE]);
    manifest["implementations"][0]["id"] = json!("bundle.projection.sm120f.v1");
    manifest["implementations"][0]["detail"]["symbol"] = json!("fixture_projection_sm120f_v1");
    manifest["components"]
        .as_array_mut()
        .expect("components are an array")
        .push(json!({
            "id": "weights-90",
            "role": "fixture.weights-90.v1",
            "format": "fixture.bytes.v1",
            "path": "components/weights-90.bin",
            "byte_length": OTHER_COMPONENT_BYTES.len(),
            "sha256": sha256_bytes(OTHER_COMPONENT_BYTES)
        }));
    manifest["graph_buckets"][0]["components"] = json!(["weights", "weights-90"]);
    manifest["implementations"]
        .as_array_mut()
        .expect("implementations are an array")
        .push(json!({
            "id": "bundle.projection.sm90a.v1",
            "key": {"operation": "fixture.projection", "variant": "decode-1"},
            "tier": "bundle",
            "detail": {
                "artifact": {
                    "origin": {"kind": "component", "component": "weights-90"},
                    "format": "fixture.bytes.v1",
                    "byte_length": OTHER_COMPONENT_BYTES.len(),
                    "sha256": sha256_bytes(OTHER_COMPONENT_BYTES)
                },
                "symbol": "fixture_projection_sm90a_v1",
                "target_architecture": OTHER_ARCHITECTURE,
                "launch": {"grid": [1, 1, 1], "block": [32, 1, 1], "dynamic_shared_memory_bytes": 0},
                "specializations": [],
                "source": {
                    "source": "fixture-source.v1",
                    "revision": "0123456789abcdef",
                    "path": "kernels.projection.cu",
                    "byte_length": 2048,
                    "sha256": "e".repeat(64)
                },
                "build": {
                    "toolchain": "nvcc",
                    "toolchain_version": "13.2",
                    "build": "fixture-build-90.v1",
                    "recipe_sha256": "d".repeat(64)
                }
            }
        }));
}

/// The same semantic operation and provider tier may be declared once per
/// architecture, each record a different exact implementation.
#[test]
fn one_operation_and_tier_may_be_implemented_once_per_declared_architecture() {
    let fixture = Fixture::new();
    fs::write(
        fixture.package().join("components/weights-90.bin"),
        OTHER_COMPONENT_BYTES,
    )
    .expect("fixture sm_90a component is writable");
    fixture.rewrite_manifest(declare_per_architecture_implementations);

    verify(&fixture);

    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("the per-architecture package verifies");
    let implementations = package.manifest().implementations();
    let architectures: Vec<_> = implementations
        .iter()
        .map(|implementation| implementation.detail().target_architecture().as_str())
        .collect();
    assert_eq!(architectures, ["sm_120f", "sm_90a"]);
    let symbols: Vec<_> = implementations
        .iter()
        .map(|implementation| implementation.detail().symbol().as_str())
        .collect();
    assert_eq!(
        symbols,
        [
            "fixture_projection_sm120f_v1",
            "fixture_projection_sm90a_v1"
        ],
        "each architecture carries its own exact implementation"
    );

    // Resolution is exact per architecture: each declared architecture's own
    // resolver holds that architecture's implementation and nothing else.
    let manifest = package.manifest();
    let keys = manifest.semantic_keys().expect("contracts convert");
    let (_, projection) = keys.iter().next().expect("one requirement is keyed");
    for (architecture, expected) in [
        (TARGET_ARCHITECTURE, "bundle.projection.sm120f.v1"),
        (OTHER_ARCHITECTURE, "bundle.projection.sm90a.v1"),
    ] {
        let resolver = manifest
            .kernel_resolver_for_architecture(&opaque_id(architecture), [])
            .expect("the declared architecture resolves");
        let resolution = resolver
            .resolve(projection)
            .expect("the architecture's own implementation satisfies it");
        assert_eq!(resolution.tier().name(), "bundle");
        assert_eq!(resolution.implementation_id().as_str(), expected);
    }
}

/// An operation implemented for one architecture only is unsupported on the
/// other declared architecture. Nothing declared for another architecture
/// stands in for it.
#[test]
fn an_operation_implemented_for_one_architecture_is_unsupported_on_another() {
    let fixture = Fixture::new();
    fixture.rewrite_manifest(|manifest| {
        declare_anchored_implementation(manifest);
        manifest["target"]["architectures"] = json!(["sm_120f", OTHER_ARCHITECTURE]);
    });
    verify(&fixture);

    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("both architectures are declared");
    let manifest = package.manifest();
    let keys = manifest.semantic_keys().expect("contracts convert");
    let (_, projection) = keys.iter().next().expect("one requirement is keyed");
    let error = manifest
        .kernel_resolver_for_architecture(&opaque_id(OTHER_ARCHITECTURE), [])
        .expect("the declared architecture resolves")
        .resolve(projection)
        .expect_err("the sm_120f implementation must not satisfy sm_90a");

    assert_eq!(
        error,
        KernelResolutionErrorV1::Unsupported {
            operation_id: projection.operation_abi().id().clone(),
        },
        "the other architecture's implementation is not a fallback"
    );
}

fn opaque_id(value: &str) -> BundleOpaqueId {
    BundleOpaqueId::new(value).expect("the test identifier is valid")
}

/// Two records for one architecture and tier stay invalid, whatever their
/// identities: inside one architecture the exact implementation is unique.
#[test]
fn duplicate_implementations_for_one_architecture_stay_invalid() {
    let fixture = Fixture::new();
    fs::write(
        fixture.package().join("components/weights-90.bin"),
        OTHER_COMPONENT_BYTES,
    )
    .expect("fixture sm_90a component is writable");
    fixture.rewrite_manifest(|manifest| {
        declare_per_architecture_implementations(manifest);
        manifest["implementations"][1]["detail"]["target_architecture"] =
            json!(TARGET_ARCHITECTURE);
    });

    let error = BundlePackageV1::verify_directory(fixture.package(), &host()).expect_err(
        "two implementations of one operation and tier on one architecture are ambiguous",
    );
    let error = error.to_string();
    assert!(
        error.contains("duplicate") && error.contains(TARGET_ARCHITECTURE),
        "refused for the wrong reason: {error}"
    );
}

#[test]
fn a_verified_package_anchors_every_implementation_artifact_to_a_component() {
    let fixture = Fixture::new();
    fixture.rewrite_manifest(declare_anchored_implementation);
    verify(&fixture);

    let drifts: [ManifestDrift; 5] = [
        ("unknown component", |manifest| {
            manifest["implementations"][0]["detail"]["artifact"]["origin"] =
                json!({"kind": "component", "component": "absent"});
        }),
        ("format drift", |manifest| {
            manifest["implementations"][0]["detail"]["artifact"]["format"] = json!("cubin.v1");
        }),
        ("byte length drift", |manifest| {
            manifest["implementations"][0]["detail"]["artifact"]["byte_length"] =
                json!(COMPONENT_BYTES.len() + 1);
        }),
        ("digest drift", |manifest| {
            manifest["implementations"][0]["detail"]["artifact"]["sha256"] = json!("a".repeat(64));
        }),
        ("undeclared architecture", |manifest| {
            manifest["implementations"][0]["detail"]["target_architecture"] = json!("sm_90a");
        }),
    ];
    for (name, drift) in drifts {
        let fixture = Fixture::new();
        fixture.rewrite_manifest(|manifest| {
            declare_anchored_implementation(manifest);
            drift(manifest);
        });
        assert!(
            BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
            "{name} unexpectedly verified"
        );
    }
}

/// Rewrites the anchored implementation to be backed by the native module.
///
/// This is the record a bundle makes when the code that runs an operation is
/// compiled into the module the host opens, rather than carried in a component
/// the host uploads. The length and digest are the module's own.
fn back_implementation_with_native_module(manifest: &mut Value) {
    let native_module = manifest["native_module"].clone();
    manifest["implementations"][0]["detail"]["artifact"] = json!({
        "origin": {"kind": "native-module"},
        "format": "chop.bundle.native-module.v1",
        "byte_length": native_module["byte_length"],
        "sha256": native_module["sha256"]
    });
}

/// An implementation may be the package's own native module, said explicitly.
#[test]
fn a_verified_package_anchors_a_native_backed_implementation_to_its_own_module() {
    let fixture = Fixture::new();
    fixture.rewrite_manifest(|manifest| {
        declare_anchored_implementation(manifest);
        back_implementation_with_native_module(manifest);
    });
    verify(&fixture);

    let package = BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("the native-backed package verifies");
    let [implementation] = package.manifest().implementations() else {
        panic!("this package declares exactly one implementation");
    };
    let artifact = implementation.detail().artifact();
    assert_eq!(artifact.origin(), &BundleArtifactOriginV1::NativeModule);
    assert_eq!(artifact.component(), None);
    let native_module = package
        .manifest()
        .native_module()
        .expect("this package ships a native module");
    assert_eq!(artifact.byte_length(), native_module.byte_length());
    assert_eq!(artifact.sha256(), native_module.sha256());
    // The component this package also carries is not what the record describes.
    assert_ne!(artifact.sha256(), sha256_bytes(COMPONENT_BYTES));
}

/// A native-backed record must be the module this package actually ships.
#[test]
fn a_native_backed_implementation_that_is_not_this_module_is_rejected() {
    let drifts: [ManifestDrift; 4] = [
        ("no native module at all", |manifest| {
            manifest["native_module"] = Value::Null;
        }),
        ("byte length drift", |manifest| {
            let length = manifest["native_module"]["byte_length"]
                .as_u64()
                .expect("the module records a length");
            manifest["implementations"][0]["detail"]["artifact"]["byte_length"] = json!(length + 1);
        }),
        ("digest drift", |manifest| {
            manifest["implementations"][0]["detail"]["artifact"]["sha256"] = json!("a".repeat(64));
        }),
        ("another format", |manifest| {
            manifest["implementations"][0]["detail"]["artifact"]["format"] =
                json!("fixture.bytes.v1");
        }),
    ];
    for (name, drift) in drifts {
        let fixture = Fixture::new();
        fixture.rewrite_manifest(|manifest| {
            declare_anchored_implementation(manifest);
            back_implementation_with_native_module(manifest);
            drift(manifest);
        });
        assert!(
            BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
            "{name} unexpectedly verified"
        );
    }
}

/// A component-backed record must still name a component, and only one.
///
/// The origin is tagged, so the two kinds cannot be confused for each other:
/// dropping the component identifier is not a native-backed record, it is a
/// component-backed record that names nothing.
#[test]
fn a_component_backed_implementation_that_names_no_component_is_rejected() {
    for (name, origin) in [
        ("no component identifier", json!({"kind": "component"})),
        (
            "an empty component identifier",
            json!({"kind": "component", "component": ""}),
        ),
        ("no origin kind at all", json!({"component": "weights"})),
    ] {
        let fixture = Fixture::new();
        fixture.rewrite_manifest(|manifest| {
            declare_anchored_implementation(manifest);
            manifest["implementations"][0]["detail"]["artifact"]["origin"] = origin;
        });
        assert!(
            BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
            "{name} unexpectedly verified"
        );
    }
}

#[test]
fn a_verified_package_declares_only_bundle_tier_implementations() {
    // A record shipped inside a package is a bundle-provided implementation.
    // Host-optimized and generic implementations come from the host catalog,
    // whatever the requirement's provisioning asks for.
    for provisioning in [
        json!({"kind": "bundle-custom-operation"}),
        json!({"kind": "host-generic-primitive", "primitive_id": "fixture-primitive-v1"}),
    ] {
        for tier in ["host-optimized", "generic"] {
            let fixture = Fixture::new();
            let provisioning = provisioning.clone();
            fixture.rewrite_manifest(|manifest| {
                declare_anchored_implementation(manifest);
                manifest["operation_requirements"][0]["provisioning"] = provisioning;
                manifest["implementations"][0]["tier"] = json!(tier);
            });
            assert!(
                BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
                "{tier} unexpectedly verified"
            );
        }

        let fixture = Fixture::new();
        let provisioning = provisioning.clone();
        fixture.rewrite_manifest(|manifest| {
            declare_anchored_implementation(manifest);
            manifest["operation_requirements"][0]["provisioning"] = provisioning;
        });
        verify(&fixture);
    }
}

fn host() -> BundleHostCompatibilityV1 {
    BundleHostCompatibilityV1::new(TARGET_ARCHITECTURE, 1)
        .expect("fixture host compatibility is valid")
}

fn verify(fixture: &Fixture) {
    BundlePackageV1::verify_directory(fixture.package(), &host())
        .expect("fixture package verifies");
}

fn reject(fixture: &Fixture) {
    assert!(
        BundlePackageV1::verify_directory(fixture.package(), &host()).is_err(),
        "fixture package unexpectedly verified: {}",
        fixture.package().display()
    );
}

struct Fixture {
    root: TempDirectory,
    package: PathBuf,
    component: PathBuf,
    native_module: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let root = TempDirectory::new("runtime-bundle-package-contract");
        let package = root.path().join("bundle");
        let component = package.join("components/weights.bin");
        let native_module = package.join(format!("native/{}", native_module_filename()));
        fs::create_dir_all(component.parent().expect("component has a parent"))
            .expect("fixture component directory is creatable");
        fs::create_dir_all(native_module.parent().expect("native module has a parent"))
            .expect("fixture native directory is creatable");
        fs::write(&component, COMPONENT_BYTES).expect("fixture component is writable");

        let fixture = Self {
            root,
            package,
            component,
            native_module,
        };
        fixture.write_native_module(ENTRY_SYMBOL);
        let manifest = fixture.manifest_value();
        fixture.write_manifest(&manifest);
        fixture
    }

    fn root(&self) -> &Path {
        self.root.path()
    }

    fn package(&self) -> &Path {
        &self.package
    }

    fn component(&self) -> &Path {
        &self.component
    }

    fn native_module(&self) -> &Path {
        &self.native_module
    }

    fn manifest_path(&self) -> PathBuf {
        self.package.join("manifest.json")
    }

    fn manifest_value(&self) -> Value {
        json!({
            "schema": "chop-model-bundle-v1",
            "abi_version": 1,
            "id": "fixture.package.v1",
            "target": {
                "architectures": [TARGET_ARCHITECTURE],
                "topologies": [
                    {"id": "tp1", "rank_count": 1},
                    {"id": "tp2", "rank_count": 2}
                ]
            },
            "components": [{
                "id": "weights",
                "role": "fixture.weights.v1",
                "format": "fixture.bytes.v1",
                "path": "components/weights.bin",
                "byte_length": COMPONENT_BYTES.len(),
                "sha256": sha256_bytes(COMPONENT_BYTES)
            }],
            "resource_requirements": {
                "minimum_device_memory_bytes": 1,
                "required_capabilities": []
            },
            "kv_layout": {
                "id": "fixture.kv.v1",
                "page_bytes": 64,
                "alignment_bytes": 64,
                "page_tokens": 1,
                "bytes_per_token": 64,
                "segments_per_page": 1,
                "maximum_sequence_tokens": 1024,
                "rank_placement": "replicated",
                "state_lifetime": "transaction"
            },
            "state_layout": {
                "id": "fixture.state.v1",
                "bytes": 64,
                "alignment_bytes": 64,
                "lifetime": "request"
            },
            "graph_buckets": [{
                "id": "decode-1",
                "components": ["weights"],
                "operations": []
            }],
            "operation_requirements": [],
            "implementations": [],
            "native_module": self.native_manifest()
        })
    }

    fn native_manifest(&self) -> Value {
        let bytes = fs::read(&self.native_module).expect("fixture native module is readable");
        json!({
            "path": format!("native/{}", native_module_filename()),
            "byte_length": bytes.len(),
            "sha256": sha256_bytes(&bytes),
            "entry_symbol": ENTRY_SYMBOL
        })
    }

    fn manifest(&self) -> Value {
        serde_json::from_slice(
            &fs::read(self.manifest_path()).expect("fixture manifest is readable"),
        )
        .expect("fixture manifest is valid JSON")
    }

    fn rewrite_manifest(&self, mutate: impl FnOnce(&mut Value)) {
        let mut manifest = self.manifest();
        mutate(&mut manifest);
        self.write_manifest(&manifest);
    }

    fn write_manifest(&self, manifest: &Value) {
        fs::write(self.manifest_path(), canonical_json(manifest))
            .expect("fixture manifest is writable");
    }

    fn refresh_native_metadata(&self) {
        let native_module = self.native_manifest();
        self.rewrite_manifest(|manifest| {
            manifest["native_module"] = native_module;
        });
    }

    fn write_native_module(&self, symbol: &str) {
        let source = self.root().join("fixture-native-module.rs");
        fs::write(
            &source,
            format!(
                r#"
#[unsafe(no_mangle)]
pub extern "C" fn {symbol}() -> *const core::ffi::c_void {{
    static API: u8 = 0;
    &API as *const u8 as *const core::ffi::c_void
}}
"#
            ),
        )
        .expect("fixture native source is writable");

        let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let output = Command::new(compiler)
            .args(["--edition=2024", "--crate-type=cdylib"])
            .arg(&source)
            .arg("-o")
            .arg(&self.native_module)
            .output()
            .expect("fixture native compiler starts");
        assert!(
            output.status.success(),
            "fixture native compiler failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn canonical_json(value: &Value) -> Vec<u8> {
    let mut value = value.clone();
    sort_json(&mut value);
    serde_json::to_vec(&value).expect("fixture JSON serializes")
}

fn sort_json(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                sort_json(value);
            }
        }
        Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            for (_, value) in &mut entries {
                sort_json(value);
            }
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            object.extend(entries);
        }
        _ => {}
    }
}

fn native_module_filename() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "fixture.bundle.dylib"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "fixture.bundle.so"
    }
    #[cfg(windows)]
    {
        "fixture.bundle.dll"
    }
}

struct TempDirectory {
    path: PathBuf,
}

impl TempDirectory {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);

        let path = std::env::temp_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("fixture temporary directory is creatable");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn historical_schema_requires_repackaging_at_both_entry_points() {
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    value["schema"] = json!("\x72\x74\x78-model-bundle-v1");
    let expected = "unsupported bundle schema '\x72\x74\x78-model-bundle-v1'; expected 'chop-model-bundle-v1'; rebuild and repackage with the current Chopinfer compiler and matching bundle library; do not edit the manifest in place";
    assert_eq!(
        BundleManifestV1::from_canonical_json(&canonical_json(&value))
            .unwrap_err()
            .to_string(),
        expected
    );
    let manifest: BundleManifestV1 = serde_json::from_value(value).unwrap();
    assert_eq!(manifest.validate().unwrap_err().to_string(), expected);
}

fn checkpoint_fixture() -> Fixture {
    use runtime_bundle::checkpoint_binding::{ROLE, SCHEMA};
    let fixture = Fixture::new();
    let mut manifest = fixture.manifest();
    let frontend = json!({"models":[{"id":"test-model"}], "tokenizer":{"repository":"test/repository", "revision":"a".repeat(40)}});
    let marker = json!({"checkpoint_binding":ROLE});
    for (id, role, value) in [
        ("frontend", "chop.frontend.openai.v1", frontend),
        ("rank-index", "chop.bundle-metadata.test.v1", marker),
    ] {
        let bytes = serde_json::to_vec(&value).unwrap();
        let path = format!("components/{id}.json");
        fs::write(fixture.package().join(&path), &bytes).unwrap();
        manifest["components"].as_array_mut().unwrap().push(json!({"id":id,"role":role,"format":"test.json.v1","path":path,"byte_length":bytes.len(),"sha256":sha256_bytes(&bytes)}));
    }
    let mut files = serde_json::Map::new();
    for file in manifest["components"]
        .as_array()
        .unwrap()
        .iter()
        .chain(std::iter::once(&manifest["native_module"]))
    {
        files.insert(
            file["path"].as_str().unwrap().into(),
            json!({"byte_length":file["byte_length"],"sha256":file["sha256"]}),
        );
    }
    let proof = json!({"schema":SCHEMA,"repository":"test/repository","revision":"a".repeat(40),"served_model_name":"test-model","source_lock_sha256":"b".repeat(64),"files":files});
    let bytes = serde_json::to_vec(&proof).unwrap();
    fs::write(fixture.package().join("components/binding.json"), &bytes).unwrap();
    manifest["components"].as_array_mut().unwrap().push(json!({"id":"binding","role":ROLE,"format":"test.json.v1","path":"components/binding.json","byte_length":bytes.len(),"sha256":sha256_bytes(&bytes)}));
    manifest["components"]
        .as_array_mut()
        .unwrap()
        .sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    manifest["graph_buckets"][0]["components"] = manifest["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].clone())
        .collect::<Vec<_>>()
        .into();
    fixture.write_manifest(&manifest);
    fixture
}

#[test]
fn checkpoint_binding_verifies_and_rejects_relabelled_files() {
    let fixture = checkpoint_fixture();
    BundlePackageV1::verify_directory(fixture.package(), &host()).unwrap();
    // A new valid file hash in the manifest does not rewrite checkpoint evidence.
    fs::write(fixture.component(), b"foreign checkpoint").unwrap();
    fixture.rewrite_manifest(|manifest| {
        let component = manifest["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == "weights")
            .unwrap();
        component["byte_length"] = 18.into();
        component["sha256"] = sha256_bytes(b"foreign checkpoint").into();
    });
    let error = BundlePackageV1::verify_directory(fixture.package(), &host())
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("checkpoint binding differs"),
        "{error}"
    );
}

#[test]
fn checkpoint_binding_cannot_be_removed_while_rank_metadata_requires_it() {
    let fixture = checkpoint_fixture();
    fixture.rewrite_manifest(|manifest| {
        manifest["components"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c["id"] != "binding");
        manifest["graph_buckets"][0]["components"]
            .as_array_mut()
            .unwrap()
            .retain(|c| c != "binding");
    });
    fs::remove_file(fixture.package().join("components/binding.json")).unwrap();
    let error = BundlePackageV1::verify_directory(fixture.package(), &host())
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("requires one checkpoint binding"),
        "{error}"
    );
}

#[test]
fn checkpoint_binding_rejects_frontend_identity_even_with_refreshed_file_hashes() {
    let fixture = checkpoint_fixture();
    let path = fixture.package().join("components/binding.json");
    let mut proof: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    proof["revision"] = "c".repeat(40).into();
    let bytes = serde_json::to_vec(&proof).unwrap();
    fs::write(path, &bytes).unwrap();
    fixture.rewrite_manifest(|manifest| {
        let component = manifest["components"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|c| c["id"] == "binding")
            .unwrap();
        component["byte_length"] = bytes.len().into();
        component["sha256"] = sha256_bytes(&bytes).into();
    });
    let error = BundlePackageV1::verify_directory(fixture.package(), &host())
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("differs from frontend identity"),
        "{error}"
    );
}

/// A manifest that declares several topologies gives each after the first
/// its own resources. Selecting a topology serves it with them; the
/// manifest's own fields stay the first topology's.
#[test]
fn topology_resources_select_each_topology() {
    use runtime_bundle::BundleTopologyV1;
    let fixture = Fixture::new();
    let mut value = fixture.manifest();
    assert!(
        !String::from_utf8(canonical_json(&value))
            .unwrap()
            .contains("topology_resources")
    );
    let bucket = |id: &str| json!({"id": id, "components": ["weights"], "operations": []});
    let mut kv = value["kv_layout"].clone();
    kv["page_bytes"] = json!(128);
    kv["bytes_per_token"] = json!(128);
    let mut state = value["state_layout"].clone();
    state["bytes"] = json!(128);
    let tp2 = json!({
        "topology": "tp2",
        "resource_requirements": {
            "minimum_device_memory_bytes": 2,
            "required_capabilities": ["fixture.tp2.v1"]
        },
        "kv_layout": kv,
        "state_layout": state,
        "graph_buckets": [bucket("decode-1"), bucket("decode-2")],
        "batch_limits": {"maximum_sequences": 2, "maximum_prefill_tokens": 64}
    });
    value["topology_resources"] = json!([tp2]);
    let bytes = canonical_json(&value);
    let manifest = BundleManifestV1::from_canonical_json(&bytes).unwrap();
    assert_eq!(manifest.to_canonical_json().unwrap(), bytes);
    let first = manifest.for_topology("tp1").unwrap();
    let second = manifest.for_topology("tp2").unwrap();
    for (selected, id, ranks) in [(&first, "tp1", 1), (&second, "tp2", 2)] {
        let [topology] = selected.target().topologies() else {
            panic!("a selected manifest declares one topology");
        };
        assert_eq!((topology.id().as_str(), topology.rank_count()), (id, ranks));
        assert!(selected.topology_resources().is_empty());
    }
    assert_eq!(
        (
            first.kv_layout().page_bytes(),
            first.state_layout().bytes(),
            first.resource_requirements().minimum_device_memory_bytes(),
            first.graph_buckets().len(),
            first.batch_limits(),
        ),
        (64, 64, 1, 1, None)
    );
    assert_eq!(
        (
            second.kv_layout().page_bytes(),
            second.state_layout().bytes(),
            second.resource_requirements().minimum_device_memory_bytes(),
            second.graph_buckets().len(),
            second
                .batch_limits()
                .map(BundleBatchLimitsV1::maximum_sequences),
        ),
        (128, 128, 2, 2, Some(2))
    );
    assert!(manifest.for_topology("tp4").is_err());
    // Pages are declared for the one topology a selected manifest serves.
    let tp2_id = BundleOpaqueId::new("tp2").unwrap();
    assert!(manifest.declared_page_layout(&tp2_id).is_err());
    assert_eq!(
        second.declared_page_layout(&tp2_id).unwrap(),
        second
            .kv_layout()
            .declared_page_layout(&second.target().topologies()[0])
            .unwrap()
    );

    // Each topology after the first has one valid entry, in order.
    let mut first_entry = tp2.clone();
    first_entry["topology"] = json!("tp1");
    let mut undeclared = tp2.clone();
    undeclared["topology"] = json!("tp3");
    let mut unknown_component = tp2.clone();
    unknown_component["graph_buckets"][0]["components"] = json!(["missing"]);
    let mut uneven_pages = tp2.clone();
    uneven_pages["kv_layout"]["page_bytes"] = json!(100);
    for invalid in [
        json!([]),
        json!([first_entry]),
        json!([undeclared]),
        json!([tp2.clone(), tp2.clone()]),
        json!([unknown_component]),
        json!([uneven_pages]),
    ] {
        let mut candidate = value.clone();
        candidate["topology_resources"] = invalid.clone();
        assert!(
            BundleManifestV1::from_canonical_json(&canonical_json(&candidate)).is_err(),
            "{invalid}"
        );
    }

    // A component may serve only some topologies; selection keeps those, and
    // a topology's buckets may name only components that serve it. Two
    // components share a role only when no topology has both.
    let mut restricted = value.clone();
    let mut extra = restricted["components"][0].clone();
    extra["id"] = json!("weights-tp2");
    extra["path"] = json!("components/weights-tp2.bin");
    extra["role"] = json!("fixture.tp2-weights.v1");
    extra["topologies"] = json!(["tp2"]);
    restricted["components"].as_array_mut().unwrap().push(extra);
    restricted["topology_resources"][0]["graph_buckets"][1]["components"] =
        json!(["weights", "weights-tp2"]);
    let manifest_with_extra =
        BundleManifestV1::from_canonical_json(&canonical_json(&restricted)).unwrap();
    let components = |topology: &str| {
        manifest_with_extra
            .for_topology(topology)
            .unwrap()
            .components()
            .iter()
            .map(|component| component.id().as_str().to_owned())
            .collect::<Vec<_>>()
    };
    assert_eq!(components("tp1"), ["weights"]);
    assert_eq!(components("tp2"), ["weights", "weights-tp2"]);
    // A component of several topologies names the one each selection serves.
    let mut both = value.clone();
    both["components"][0]["topologies"] = json!(["tp1", "tp2"]);
    let manifest_both = BundleManifestV1::from_canonical_json(&canonical_json(&both)).unwrap();
    for topology in ["tp1", "tp2"] {
        let selected = manifest_both.for_topology(topology).unwrap();
        assert_eq!(
            selected.components()[0]
                .topologies()
                .iter()
                .map(BundleOpaqueId::as_str)
                .collect::<Vec<_>>(),
            [topology]
        );
    }
    let mut per_topology = restricted.clone();
    per_topology["components"][0]["topologies"] = json!(["tp1"]);
    per_topology["components"][1]["role"] = json!("fixture.weights.v1");
    per_topology["topology_resources"][0]["graph_buckets"] = json!([bucket("decode-1")]);
    per_topology["topology_resources"][0]["graph_buckets"][0]["components"] =
        json!(["weights-tp2"]);
    per_topology["topology_resources"][0]["batch_limits"]["maximum_sequences"] = json!(1);
    BundleManifestV1::from_canonical_json(&canonical_json(&per_topology))
        .expect("components of one role for different topologies");
    let mut first_names_it = restricted.clone();
    first_names_it["graph_buckets"][0]["components"] = json!(["weights", "weights-tp2"]);
    let mut undeclared_topology = restricted.clone();
    undeclared_topology["components"][1]["topologies"] = json!(["tp3"]);
    let mut shared_role = restricted.clone();
    shared_role["components"][1]["role"] = json!("fixture.weights.v1");
    for invalid in [first_names_it, undeclared_topology, shared_role] {
        assert!(BundleManifestV1::from_canonical_json(&canonical_json(&invalid)).is_err());
    }

    // A host checks the requirements of the topology it names, or of every
    // declared topology.
    fixture.write_manifest(&value);
    let tp2_host = |capable: bool| {
        let host = host().with_topology(BundleTopologyV1::new(tp2_id.clone(), 2).unwrap());
        if capable {
            host.with_capabilities([BundleOpaqueId::new("fixture.tp2.v1").unwrap()])
        } else {
            host
        }
    };
    assert!(BundlePackageV1::verify_directory(fixture.package(), &host()).is_err());
    assert!(BundlePackageV1::verify_directory(fixture.package(), &tp2_host(false)).is_err());
    let tp1_host = host()
        .with_topology(BundleTopologyV1::new(BundleOpaqueId::new("tp1").unwrap(), 1).unwrap());
    let package = BundlePackageV1::verify_directory(fixture.package(), &tp1_host).unwrap();
    assert_eq!(package.for_topology("tp1").unwrap().manifest(), &first);
    let package = BundlePackageV1::verify_directory(fixture.package(), &tp2_host(true)).unwrap();
    assert_eq!(package.manifest(), &manifest);
    assert_eq!(package.for_topology("tp2").unwrap().manifest(), &second);
}
