mod common;

use common::*;
use projection_spec::registry::{self, Provider, Registration, Registry, Target};
use projection_spec::{Variant, parse};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    chunks: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    order: Option<String>,
}

static PROVIDER: Provider = Provider {
    namespace: "fixture",
    registrations: &[
        Registration {
            numerical_contract: "fixture.numeric.v1",
            id: "fixture/chunk-regroup.v1",
            target: Target::Projection,
            decoder: registry::parameters::<Parameters>,
        },
        Registration {
            numerical_contract: "fixture.numeric.v1",
            id: "fixture/legacy-fused.v1",
            target: Target::FusionGroup,
            decoder: registry::parameters::<Parameters>,
        },
    ],
};
static FOREIGN: Provider = Provider {
    namespace: "other",
    registrations: &[Registration {
        numerical_contract: "fixture.numeric.v1",
        id: "fixture/chunk-regroup.v1",
        target: Target::Projection,
        decoder: registry::parameters::<Parameters>,
    }],
};
static DUPLICATE: Provider = Provider {
    namespace: "fixture",
    registrations: &[
        Registration {
            numerical_contract: "fixture.numeric.v1",
            id: "fixture/chunk-regroup.v1",
            target: Target::Projection,
            decoder: registry::parameters::<Parameters>,
        },
        Registration {
            numerical_contract: "fixture.numeric.v1",
            id: "fixture/chunk-regroup.v1",
            target: Target::Projection,
            decoder: registry::parameters::<Parameters>,
        },
    ],
};
static EMPTY_NAMESPACE: Provider = Provider {
    namespace: "",
    registrations: &[],
};
static BAD_ID: Provider = Provider {
    namespace: "fixture",
    registrations: &[Registration {
        numerical_contract: "fixture.numeric.v1",
        id: "missing-slash",
        target: Target::Projection,
        decoder: registry::parameters::<Parameters>,
    }],
};

fn registered() -> Registry {
    let mut registry = Registry::default();
    registry.register(&PROVIDER).unwrap();
    registry
}

#[test]
fn owned_projection_and_group_variants_resolve_with_strict_parameters() {
    let mut text: Value = serde_json::from_str(FIXTURE).unwrap();
    text["projections"][0]["composition"]["implementation"] =
        json!({"id":"fixture/chunk-regroup.v1","parameters":{"chunks":2}});
    text["fusion_groups"][0]["composition"]["implementation"] =
        json!({"id":"fixture/legacy-fused.v1","parameters":{"chunks":2}});
    complete_contracts(&mut text);
    let d = parse(&text.to_string(), &geometry(), &registered()).unwrap();
    assert_eq!(
        d.projections[0]
            .composition
            .as_ref()
            .unwrap()
            .implementation
            .as_ref()
            .unwrap()
            .parameters,
        json!({"chunks":2})
    );
    let before = parse(&text.to_string(), &geometry(), &Registry::default()).unwrap_err();
    assert!(before.0.contains("not registered"));
    text["projections"][0]["composition"]["implementation"]["parameters"]["unknown"] = json!(true);
    assert!(
        parse(&text.to_string(), &geometry(), &registered())
            .unwrap_err()
            .0
            .contains("unknown field")
    );
}

#[test]
fn providers_register_atomically_and_refuse_foreign_duplicate_or_bad_ids() {
    for provider in [&FOREIGN, &DUPLICATE, &EMPTY_NAMESPACE, &BAD_ID] {
        let mut registry = Registry::default();
        assert!(registry.register(provider).is_err());
        assert_eq!(registry.ids().count(), 0);
        registry.register(&PROVIDER).unwrap();
        assert_eq!(registry.ids().count(), 2);
    }
    let mut registry = registered();
    assert!(
        registry
            .register(&PROVIDER)
            .unwrap_err()
            .0
            .contains("already registered")
    );
    assert_eq!(registry.ids().count(), 2);
    for id in [
        "",
        "fixture",
        "/name",
        "fixture/",
        "Fixture/name",
        "fixture/a/b",
    ] {
        assert!(registry::split_id(id).is_err(), "accepted {id}");
    }
}

#[test]
fn resolve_refuses_foreign_unregistered_and_wrong_target_variants() {
    for (id, target, why) in [
        (
            "other/chunk-regroup.v1",
            Target::Projection,
            "outside declaration namespace",
        ),
        ("fixture/missing.v1", Target::Projection, "not registered"),
        (
            "fixture/legacy-fused.v1",
            Target::Projection,
            "another target",
        ),
        (
            "fixture/chunk-regroup.v1",
            Target::FusionGroup,
            "another target",
        ),
    ] {
        let error = registered()
            .resolve(
                "fixture",
                &Variant {
                    id: id.into(),
                    parameters: json!({"chunks":2}),
                    numerical_contract: "fixture.numeric.v1".into(),
                    inputs: vec![port()],
                    outputs: vec![port()],
                },
                target,
                &geometry(),
            )
            .unwrap_err();
        assert!(error.0.contains(why), "{error}");
    }
}

#[test]
fn adding_a_provider_preserves_shared_declaration_digest() {
    assert_eq!(
        compact_declaration().digest(),
        parse(FIXTURE, &geometry(), &registered()).unwrap().digest()
    );
    let mut text: Value = serde_json::from_str(FIXTURE).unwrap();
    text["projections"][0]["composition"]["implementation"] =
        json!({"id":"fixture/chunk-regroup.v1","parameters":{"chunks":2,"order":null}});
    complete_contracts(&mut text);
    let explicit = parse(&text.to_string(), &geometry(), &registered()).unwrap();
    text["projections"][0]["composition"]["implementation"]["parameters"]
        .as_object_mut()
        .unwrap()
        .remove("order");
    assert_eq!(
        explicit.digest(),
        parse(&text.to_string(), &geometry(), &registered())
            .unwrap()
            .digest()
    );
}

fn port() -> projection_spec::contracts::Port {
    projection_spec::contracts::Port {
        interpretation: None,
        id: "rows".into(),
        binding: "rows".into(),
        dtype: projection_spec::DType::Bf16,
        shape: vec![projection_spec::Dim::Literal(1)],
        state: false,
        format: None,
    }
}
fn complete_contracts(text: &mut Value) {
    for key in ["projections", "fusion_groups"] {
        for entry in text[key].as_array_mut().unwrap() {
            if let Some(c) = entry.get_mut("composition") {
                c["pre"] = json!([]);
                c["post"] = json!([]);
                let v = &mut c["implementation"];
                v["numerical_contract"] = json!("fixture.numeric.v1");
                v["inputs"] = serde_json::to_value([port()]).unwrap();
                v["outputs"] = serde_json::to_value([port()]).unwrap();
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DefaultedParameters {
    chunks: u32,
    #[serde(default)]
    order: u32,
}
static DEFAULTED: Provider = Provider {
    namespace: "fixture",
    registrations: &[Registration {
        numerical_contract: "fixture.numeric.v1",
        id: "fixture/defaulted.v1",
        target: Target::Projection,
        decoder: registry::parameters::<DefaultedParameters>,
    }],
};

#[test]
fn rust_constructors_must_supply_canonical_variant_parameters() {
    let mut r = Registry::default();
    r.register(&DEFAULTED).unwrap();
    let mut d = declaration();
    d.projections[0].composition = Some(projection_spec::contracts::Composition {
        implementations: Default::default(),
        numerical_contracts: Default::default(),
        implementation: Some(Variant {
            id: "fixture/defaulted.v1".into(),
            parameters: json!({"chunks":2}),
            numerical_contract: "fixture.numeric.v1".into(),
            inputs: vec![port()],
            outputs: vec![port()],
        }),
        ..Default::default()
    });
    assert!(
        d.validate(&geometry(), &r)
            .unwrap_err()
            .0
            .contains("canonical parameters")
    );
    let parsed = parse(&d.canonical_json(), &geometry(), &r).unwrap();
    parsed.validate(&geometry(), &r).unwrap();
    assert_eq!(
        parsed.digest(),
        parse(&parsed.canonical_json(), &geometry(), &r)
            .unwrap()
            .digest()
    );
}

#[test]
fn fusion_implementations_have_per_use_overrides_and_one_default() {
    let mut text: Value = serde_json::from_str(FIXTURE).unwrap();
    text["fusion_groups"][0]["composition"] = json!({"pre":[],"post":[],"implementation":{
        "id":"fixture/legacy-fused.v1","parameters":{"chunks":2}
    }});
    complete_contracts(&mut text);
    let default = text["fusion_groups"][0]["composition"]["implementation"].clone();
    let mut override_ = default.clone();
    override_["parameters"]["chunks"] = json!(3);
    let mut uses = text["projections"][0]["uses"].clone();
    for use_ in uses.as_array_mut().unwrap() {
        let limits = if use_["phase"] == "Verify" {
            use_["capacity"].clone()
        } else {
            Value::Null
        };
        use_["contract"] = json!({"compiled_capacity":256,"row_multiplier":1,
            "per_item_verify":limits,"aggregate_verify":limits,"output_dtype":"Bf16",
            "numerical_contract":"fixture.numeric.v1","activation":null,"exact_rows":false,
            "support":[],"implementation":override_});
    }
    text["fusion_groups"][0]
        .as_object_mut()
        .unwrap()
        .remove("use_profile");
    text["fusion_groups"][0]["uses"] = uses;
    parse(&text.to_string(), &geometry(), &registered()).unwrap();
    text["fusion_groups"][0]["uses"][0]["contract"]["implementation"] = default;
    assert!(
        parse(&text.to_string(), &geometry(), &registered())
            .unwrap_err()
            .0
            .contains("repeats the default implementation")
    );
}

#[test]
fn provider_helpers_check_ports_and_round_trip_parameters() {
    let port = |id: &str, dtype| projection_spec::contracts::Port {
        id: id.into(),
        binding: id.into(),
        dtype,
        interpretation: None,
        shape: vec![],
        state: false,
        format: None,
    };
    let mut variant = Variant {
        id: "fixture/chunk-regroup.v1".into(),
        parameters: json!({"chunks": 2}),
        numerical_contract: "fixture.numeric.v1".into(),
        inputs: vec![port("activation", projection_spec::DType::Bf16)],
        outputs: vec![port("output", projection_spec::DType::F32)],
    };
    let typed: Parameters = registry::typed(&variant).unwrap();
    assert_eq!(registry::canonical(&typed).unwrap(), json!({"chunks": 2}));
    let inputs = [("activation", projection_spec::DType::Bf16)];
    let outputs = [("output", projection_spec::DType::F32)];
    registry::require_ports(&variant, &inputs, &outputs).unwrap();
    let refusal = |variant: &Variant| {
        registry::require_ports(variant, &inputs, &outputs)
            .unwrap_err()
            .0
    };
    variant.outputs[0].binding = "other".into();
    assert!(refusal(&variant).contains("identity or dtype"));
    variant
        .outputs
        .push(port("extra", projection_spec::DType::F32));
    assert!(refusal(&variant).contains("port count"));
    assert_eq!(
        registry::require(false, "refused").unwrap_err().0,
        "refused"
    );
}
