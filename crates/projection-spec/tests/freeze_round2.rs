mod common;
use common::*;
use projection_spec::registry::Registry;
use projection_spec::*;

#[test]
fn kernel_owned_norm_consumers_bind_dtype_without_declaring_arithmetic() {
    let mut value = serde_json::to_value(declaration()).unwrap();
    value["side_tensors"][0]["norm"] = serde_json::Value::Null;
    value["side_tensors"][0]["consumer"] = serde_json::json!({"kernel":"attention.query"});
    let d = parse(&value.to_string(), &geometry(), &Registry::default()).unwrap();
    let mut m = metadata();
    m.tensors.get_mut("layers.0.norm.weight").unwrap().dtype = "F32".into();
    let b = d
        .bind(&geometry(), &Registry::default(), &m, &Bf16Rules)
        .unwrap();
    let view = b
        .side_views
        .iter()
        .find(|v| v.tensor == "layers.0.norm.weight")
        .unwrap();
    assert_eq!(view.view.dtype, DType::F32);
    assert!(d.side_tensors[0].norm.is_none());
    value["side_tensors"][0]["consumer"]["unknown"] = serde_json::json!(true);
    assert!(parse(&value.to_string(), &geometry(), &Registry::default()).is_err());
}

#[test]
fn activation_owner_domains_intersect_rank_k_and_return_owner_relative_columns() {
    use projection_spec::inputs::ActivationSegment;
    let d = declaration();
    let source = |name: &str, origin: u64| {
        serde_json::from_value::<ActivationSegment>(serde_json::json!({
            "source":name,"origin":origin,"length":4,
            "selection":{"select":"rank_k","projection":"out","operand":"weight"}
        }))
        .unwrap()
    };
    let embedding = source("embedding", 0);
    let hidden = source("hidden", 4);
    let tp = projection_spec::derive::Topology { ranks: 2, rank: 1 };
    assert!(embedding.range(&d, &geometry(), tp).unwrap().is_empty());
    assert_eq!(hidden.range(&d, &geometry(), tp).unwrap(), 0..4);
    let mut malformed = serde_json::to_value(&hidden).unwrap();
    malformed.as_object_mut().unwrap().remove("length");
    let segment: ActivationSegment = serde_json::from_value(malformed).unwrap();
    assert!(
        segment
            .range(&d, &geometry(), tp)
            .unwrap_err()
            .0
            .contains("origin and length")
    );
}

fn parameters(v: &projection_spec::Variant, _: &Geometry) -> Result<serde_json::Value, Error> {
    Ok(v.parameters.clone())
}
static IMPLEMENTATIONS: projection_spec::registry::Provider = projection_spec::registry::Provider {
    namespace: "fixture",
    registrations: &[projection_spec::registry::Registration {
        id: "fixture/selected.v1",
        numerical_contract: "fixture.selected.v1",
        target: projection_spec::registry::Target::Projection,
        decoder: parameters,
    }],
};
#[test]
fn implementation_maps_select_by_bound_source_format_and_validate_immutable_identity() {
    use kernel_core::precision::format::FormatId;
    let mut registry = Registry::default();
    registry.register(&IMPLEMENTATIONS).unwrap();
    let variant = serde_json::json!({"id":"fixture/selected.v1","numerical_contract":"fixture.selected.v1","parameters":{},"inputs":[{"id":"x","binding":"x","dtype":"Bf16","shape":[8,16],"state":false}],"outputs":[{"id":"y","binding":"y","dtype":"Bf16","shape":[8,16],"state":false}]});
    let mut value = serde_json::to_value(declaration()).unwrap();
    value["projections"][0]["composition"] = serde_json::json!({"pre":[],"post":[],"implementations":{"bf16":variant},"numerical_contracts":{"bf16":{"numerical_contract":"fixture.selected.v1","a16_recipe":"scaled_weight"}}});
    let d = parse(&value.to_string(), &geometry(), &registry).unwrap();
    let bound = d
        .bind(&geometry(), &registry, &metadata(), &Bf16Rules)
        .unwrap();
    let composition = d.projections[0].composition.as_ref().unwrap();
    assert_eq!(
        composition
            .implementation_for(bound.operands[0].format)
            .unwrap()
            .id,
        "fixture/selected.v1"
    );
    assert!(composition.implementation_for(FormatId::Nvfp4).is_none());
    value["projections"][0]["composition"]["numerical_contracts"]["bf16"]["numerical_contract"] =
        serde_json::json!("other.v1");
    assert!(
        parse(&value.to_string(), &geometry(), &registry)
            .unwrap_err()
            .0
            .contains("immutable numerical")
    );
}

#[test]
fn lookup_shard_templates_expand_names_and_scoped_row_expressions() {
    let mut value = serde_json::to_value(declaration()).unwrap();
    value["projections"][7]["lookup"]["shards"] = serde_json::json!({"count":4,"source":{"tensor":"hash.{shard}.weight"},"rows":{"start":{"op":"product","factors":["shard",8]},"length":8}});
    let d = parse(&value.to_string(), &geometry(), &Registry::default()).unwrap();
    let mut m = metadata();
    for (index, name) in [
        "hash.0.0.weight",
        "hash.0.1.weight",
        "hash.1.0.weight",
        "hash.1.1.weight",
    ]
    .iter()
    .enumerate()
    {
        let header = m.tensors.remove(*name).unwrap();
        m.tensors.insert(format!("hash.{index}.weight"), header);
    }
    let bound = d
        .bind(&geometry(), &Registry::default(), &m, &Bf16Rules)
        .unwrap();
    assert_eq!(
        bound
            .operands
            .iter()
            .filter(|o| o.projection == "hash")
            .count(),
        4
    );
    assert!(d.canonical_json().contains("\"count\":4"));
    assert!(
        bound
            .operands
            .iter()
            .any(|o| o.imported.weight.codes.tensor == "hash.3.weight" && o.shard == Some(3))
    );
    value["projections"][7]["lookup"]["shards"]["count"] = serde_json::json!(0);
    assert!(parse(&value.to_string(), &geometry(), &Registry::default()).is_err());
}
