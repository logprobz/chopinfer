mod common;

use common::*;
use projection_spec::registry::Registry;
use projection_spec::{Declaration, parse};
use serde_json::Value;

#[test]
fn synthetic_declaration_keeps_its_independently_computed_digest() {
    assert_eq!(
        compact_declaration().digest(),
        include_str!("fixtures/synthetic.sha256").trim()
    );
}

fn reverse_objects(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for value in map.values_mut() {
                reverse_objects(value);
            }
            let entries: Vec<_> = std::mem::take(map).into_iter().collect();
            map.extend(entries.into_iter().rev());
        }
        Value::Array(values) => {
            for value in values {
                reverse_objects(value);
            }
        }
        _ => {}
    }
}

#[test]
fn json_key_order_whitespace_and_omitted_optional_defaults_are_stable() {
    let mut value: Value = serde_json::from_str(FIXTURE).unwrap();
    reverse_objects(&mut value);
    assert_eq!(
        compact_declaration().digest(),
        parse(
            &serde_json::to_string_pretty(&value).unwrap(),
            &geometry(),
            &Registry::default()
        )
        .unwrap()
        .digest()
    );
    for p in value["projections"].as_array_mut().unwrap() {
        if p.get("parts").is_none() {
            p["parts"] = serde_json::json!([]);
        }
        p["format"] = Value::Null;
        p["composition"] = Value::Null;
    }
    assert_eq!(
        compact_declaration().digest(),
        parse(&value.to_string(), &geometry(), &Registry::default())
            .unwrap()
            .digest()
    );
}

#[test]
fn ordered_members_and_numerical_contracts_change_identity() {
    let original = compact_declaration();
    let mut changed = original.clone();
    changed.fusion_groups[0].members.reverse();
    assert_ne!(original.digest(), changed.digest());
    changed = original.clone();
    changed.projections[0].numerical_contract = "a16.other.v1".into();
    assert_ne!(original.digest(), changed.digest());
    assert_eq!(
        serde_json::from_str::<Declaration>(&original.canonical_json())
            .unwrap()
            .digest(),
        original.digest()
    );
}

#[test]
fn every_nested_schema_object_refuses_unknown_fields() {
    for pointer in [
        "",
        "/layer_stems/0",
        "/layer_stems/0/indices",
        "/projections/0",
        "/projections/0/source",
        "/projections/0/parts/0",
        "/projections/0/kind",
        "/projections/0/split",
        "/projections/0/epilogues",
        "/use_profiles/served/0",
        "/fusion_groups/0",
        "/side_tensors/0",
        "/side_tensors/0/source",
    ] {
        let mut value: Value = serde_json::from_str(FIXTURE).unwrap();
        value
            .pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("unexpected".into(), Value::Bool(true));
        assert!(
            parse(&value.to_string(), &geometry(), &Registry::default()).is_err(),
            "accepted field in {pointer}"
        );
    }
}
