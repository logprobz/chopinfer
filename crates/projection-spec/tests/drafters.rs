mod common;
use common::*;
use projection_spec::registry::Registry;
use projection_spec::*;
use serde_json::{Value, json};

fn index(value: &Value, id: &str) -> usize {
    value["projections"]
        .as_array()
        .unwrap()
        .iter()
        .position(|p| p["id"] == id)
        .unwrap()
}

/// The synthetic drafter projection made a port-compatible copy of `out`.
fn published() -> Value {
    let mut value = serde_json::to_value(compact_declaration()).unwrap();
    let (out, draft) = (index(&value, "out"), index(&value, "draft"));
    let mut copy = value["projections"][out].clone();
    copy["id"] = json!("draft");
    copy["source"] = value["projections"][draft]["source"].clone();
    copy["publishes_as"] = json!("out");
    value["projections"][draft] = copy;
    value
}

fn parsed(value: &Value) -> Result<Declaration, Error> {
    parse(&value.to_string(), &geometry(), &Registry::default())
}

#[test]
fn absent_drafter_fields_keep_the_canonical_identity() {
    let d = compact_declaration();
    assert!(d.drafters.is_empty());
    assert!(d.projections.iter().all(|p| p.publishes_as.is_none()));
    let canonical = d.canonical_json();
    assert!(!canonical.contains("drafters") && !canonical.contains("publishes_as"));
    assert_eq!(d.published("draft"), "draft");
}

#[test]
fn a_projection_publishes_as_a_compatible_port() {
    let d = parsed(&published()).unwrap();
    assert_eq!(d.published("draft"), "out");
    assert_eq!(d.published("out"), "out");
    let draft = index(&published(), "draft");
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut value = published();
        edit(&mut value["projections"][draft]);
        parsed(&value).unwrap_err().0
    };
    for target in ["missing", "draft"] {
        assert!(refused(&|p| p["publishes_as"] = json!(target)).contains("no other declared"));
    }
    assert!(refused(&|p| p["shape"] = json!(["hidden", "intermediate"])).contains("differ"));
    assert!(refused(&|p| p["numerical_contract"] = json!("other.v1")).contains("differ"));
    // A target must publish its own port: no chains or cycles.
    let mut value = published();
    let out = index(&value, "out");
    value["projections"][out]["publishes_as"] = json!("draft");
    assert!(
        parsed(&value)
            .unwrap_err()
            .0
            .contains("publishes as another")
    );
}

#[test]
fn a_published_port_never_widens_format_admission() {
    let recipe =
        |contract: &str| json!({"numerical_contract": contract, "a16_recipe": "scaled_weight"});
    let composition =
        |formats: Value| json!({"pre": [], "post": [], "numerical_contracts": formats});
    let mut value = published();
    let (out, draft) = (index(&value, "out"), index(&value, "draft"));
    value["projections"][out]["composition"] =
        composition(json!({"bf16": recipe("bf16.v1"), "nvfp4": recipe("nvfp4.v1")}));
    value["projections"][draft]["composition"] = composition(json!({"bf16": recipe("bf16.v1")}));
    parsed(&value).unwrap();
    for formats in [
        json!({"bf16": recipe("bf16.v1"), "mxfp8": recipe("mxfp8.v1")}),
        json!({"bf16": recipe("other.v1")}),
    ] {
        value["projections"][draft]["composition"] = composition(formats);
        assert!(parsed(&value).unwrap_err().0.contains("does not publish"));
    }
    value["projections"][draft]
        .as_object_mut()
        .unwrap()
        .remove("composition");
    assert!(parsed(&value).is_err());
}

fn drafted() -> Value {
    let mut value = published();
    value["drafters"] = json!([{"id": "draft", "stems": ["drafter"], "operations": ["draft"]}]);
    value
}

#[test]
fn a_drafter_owns_its_stems_and_records_member_operations() {
    let d = parsed(&drafted()).unwrap();
    assert_eq!(d.drafter_of("draft").unwrap().id, "draft");
    assert!(d.drafter_of("out").is_none() && d.drafter_of_group("qk").is_none());
    let side = |id: &str| d.side_tensors.iter().find(|s| s.id == id).unwrap();
    assert_eq!(d.drafter_of_side(side("conv")).unwrap().id, "draft");
    assert!(d.drafter_of_side(side("norm")).is_none());
    // Omitting the drafter removes every member and still validates.
    let mut omitted = d.clone();
    omitted.omit_drafter("draft").unwrap();
    assert!(omitted.drafters.is_empty());
    assert!(omitted.projections.iter().all(|p| p.id != "draft"));
    assert!(omitted.side_tensors.iter().all(|s| s.id != "conv"));
    assert!(omitted.layer_stems.iter().all(|s| s.id != "drafter"));
    omitted.validate(&geometry(), &Registry::default()).unwrap();
    assert!(omitted.clone().omit_drafter("draft").is_err());
    // Without the drafter fields, the declaration keeps its former digest.
    let mut plain = d.clone();
    plain.drafters.clear();
    for p in &mut plain.projections {
        p.publishes_as = None;
    }
    assert_ne!(plain.digest(), d.digest());
}

#[test]
fn drafter_membership_is_closed_and_unique() {
    let refused = |drafter: Value| {
        let mut value = drafted();
        value["drafters"][0] = drafter;
        parsed(&value).unwrap_err().0
    };
    for (drafter, reason) in [
        (
            json!({"id": "d", "stems": ["missing"], "operations": []}),
            "unknown or claimed stem",
        ),
        (
            json!({"id": "d", "stems": ["drafter", "drafter"], "operations": []}),
            "unknown or claimed stem",
        ),
        (
            json!({"id": "d", "stems": [], "projections": ["missing"], "operations": []}),
            "unknown projection",
        ),
        (
            json!({"id": "d", "stems": ["drafter"], "projections": ["draft"], "operations": []}),
            "again",
        ),
        (
            json!({"id": "d", "stems": ["drafter"], "operations": ["out"]}),
            "non-member",
        ),
        (
            json!({"id": "d", "stems": ["drafter"], "operations": ["draft", "draft"]}),
            "repeated",
        ),
        (
            json!({"id": "d", "stems": [], "projections": ["q"], "operations": []}),
            "mixes drafter",
        ),
        (
            json!({"id": "", "stems": [], "operations": []}),
            "invalid or duplicate",
        ),
    ] {
        let error = refused(drafter.clone());
        assert!(error.contains(reason), "{drafter}: {error}");
    }
    let mut value = drafted();
    let mut other = value["drafters"][0].clone();
    other["id"] = json!("other");
    value["drafters"].as_array_mut().unwrap().push(other);
    assert!(parsed(&value).unwrap_err().0.contains("claimed stem"));
    value["drafters"][1] =
        json!({"id": "other", "stems": [], "projections": ["table"], "operations": []});
    parsed(&value).unwrap();
    value["drafters"][1]["projections"] = json!(["draft"]);
    assert!(parsed(&value).unwrap_err().0.contains("shares projection"));
    value["drafters"][0]["unknown"] = json!(true);
    assert!(parsed(&value).is_err());
}

#[test]
fn an_execution_alias_maps_executed_layers_to_their_source_index() {
    let mut value = drafted();
    value["layer_stems"][1]["execution"] = json!({
        "indices": ["layers"],
        "aliases": [{"execution_index": "layers", "source_index": 0}]
    });
    let d = parsed(&value).unwrap();
    let stem = &d.layer_stems[1];
    assert_eq!(stem.executed(&geometry()).unwrap(), [2]);
    assert_eq!(stem.source_index(&geometry(), 2).unwrap(), 0);
    assert_eq!(stem.source_index(&geometry(), 0).unwrap(), 0);
    assert_eq!(d.layer_stems[0].executed(&geometry()).unwrap(), [0, 1]);
    // The drafter projection runs the decoder's port after its layers.
    let g = geometry();
    assert_eq!(d.executed_at(&g, 2, "out").unwrap().id, "draft");
    assert_eq!(d.executed_at(&g, 1, "out").unwrap().id, "out");
    assert!(d.executed_at(&g, 0, "draft").is_err());
    assert_eq!(d.source_layer(&g, "draft", 2).unwrap(), 0);
    assert_eq!(d.source_layer(&g, "out", 1).unwrap(), 1);
    assert_eq!(
        d.drafter_sources(&g)
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        ["draft.0.conv", "draft.0.proj.weight"]
    );
}

#[test]
fn offered_formats_need_a_generated_weight_and_a_recipe() {
    use kernel_core::contract::WeightFormat;
    let recipe = json!({"numerical_contract": "fixture.v1", "a16_recipe": "scaled_weight"});
    let with_offers = |id: &str, offers: Value, formats: Value| {
        let mut value = serde_json::to_value(compact_declaration()).unwrap();
        let p = index(&value, id);
        value["projections"][p]["offers"] = offers;
        value["projections"][p]["composition"] =
            json!({"pre": [], "post": [], "numerical_contracts": formats});
        parsed(&value)
    };
    let both = json!({"bf16": recipe, "nvfp4": recipe, "nvfp4_weight_only": recipe});
    let d = with_offers("out", json!(["bf16", "nvfp4"]), both.clone()).unwrap();
    let out = d.projections.iter().find(|p| p.id == "out").unwrap();
    assert_eq!(
        out.offered_weights().unwrap(),
        [WeightFormat::Bf16, WeightFormat::Nvfp4]
    );
    for (id, offers, formats, reason) in [
        (
            "out",
            json!(["nvfp4", "nvfp4_weight_only"]),
            both.clone(),
            "twice",
        ),
        ("out", json!(["mxfp8"]), both, "without a numerical recipe"),
        (
            "out",
            json!(["fp8_block128"]),
            json!({"fp8_block128": recipe}),
            "no generated offer",
        ),
    ] {
        let error = with_offers(id, offers, formats).unwrap_err().0;
        assert!(error.contains(reason), "{error}");
    }
    // A projection that publishes as another declares no offers of its own.
    let mut value = published();
    let draft = index(&value, "draft");
    value["projections"][draft]["offers"] = json!(["bf16"]);
    assert!(parsed(&value).unwrap_err().0.contains("declares none"));
    // A lookup's offer uses the projection's own numerical id.
    let mut value = serde_json::to_value(compact_declaration()).unwrap();
    let table = index(&value, "table");
    value["projections"][table]["offers"] = json!(["nvfp4"]);
    parsed(&value).unwrap();
}

/// A published port's physical layout includes per-operand padding: an
/// alias whose K tile differs would read the target's entries at another
/// stride.
#[test]
fn a_published_port_keeps_the_target_operand_padding() {
    let split = json!([{"operand": "weight", "axis": "k", "tile": 8}]);
    let mut value = published();
    let (out, draft) = (index(&value, "out"), index(&value, "draft"));
    value["projections"][draft]["operand_splits"] = split.clone();
    let error = parsed(&value).unwrap_err().0;
    assert!(error.contains("physical layout differs"), "{error}");
    value["projections"][out]["operand_splits"] = split;
    parsed(&value).unwrap();
}

/// Lookup head ownership is part of the layout; the alias's own shard
/// sources are not.
#[test]
fn a_published_lookup_keeps_the_target_head_ownership() {
    let mut value = serde_json::to_value(compact_declaration()).unwrap();
    let hash = index(&value, "hash");
    let mut copy = value["projections"][hash].clone();
    copy["id"] = json!("hash-draft");
    copy["publishes_as"] = json!("hash");
    for shard in copy["lookup"]["shards"].as_array_mut().unwrap() {
        let tensor = shard["source"]["tensor"].as_str().unwrap().to_owned();
        shard["source"]["tensor"] = json!(format!("draft.{tensor}"));
    }
    value["projections"].as_array_mut().unwrap().push(copy);
    let alias = value["projections"].as_array().unwrap().len() - 1;
    parsed(&value).unwrap();
    let heads = &mut value["projections"][alias]["lookup"]["heads"];
    heads.as_array_mut().unwrap().reverse();
    let error = parsed(&value).unwrap_err().0;
    assert!(error.contains("physical layout differs"), "{error}");
}
