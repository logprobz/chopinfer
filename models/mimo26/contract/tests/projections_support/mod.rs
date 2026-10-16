//! Metadata fixtures contain header facts only; no checkpoint payload is read.
use model_compiler_sdk::quantization::import::EncodingRecipe;
use model_compiler_sdk::quantization::projection::metadata::{
    HeaderCompanionRules, HeaderRole, ScalarSharing,
};
use model_contract::projections;
use model_mimo26_contract as model_contract;
use projection_spec::bind::CheckpointMetadata;
use projection_spec::{Declaration, Error, Geometry};
use std::collections::BTreeSet;

// The checkpoint header helpers every model shares, from the SDK's test
// support sources (outside its module tree).
#[path = "../../../../../crates/model-compiler-sdk/test-support/checkpoint.rs"]
pub(crate) mod checkpoint;
pub(crate) use checkpoint::fixture;

fn header_rules(d: &Declaration, g: &Geometry, m: &CheckpointMetadata) -> HeaderCompanionRules {
    let mut rules = HeaderCompanionRules::new(m, None).unwrap();
    // A BF16 source is an explicit source role when an export has a global FP8 default.
    for p in &d.projections {
        if let Some(source) = &p.source {
            let mut names = if let Some(id) = &source.stem {
                let stem = d.layer_stems.iter().find(|s| s.id == *id).unwrap();
                stem.indices
                    .resolve(g)
                    .unwrap()
                    .into_iter()
                    .map(|i| {
                        source
                            .tensor
                            .replace("{stem}", &stem.template.replace("{i}", &i.to_string()))
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![source.tensor.clone()]
            };
            if source.tensor.contains("{operand}") {
                names = names
                    .into_iter()
                    .flat_map(|name| {
                        p.operands
                            .iter()
                            .map(move |operand| name.replace("{operand}", operand))
                    })
                    .collect();
            }
            if source.tensor.contains("{expert}") {
                let experts = match &p.kind {
                    projection_spec::ProjectionKind::ExpertBank { experts, .. } => {
                        experts.resolve(g).unwrap()
                    }
                    _ => unreachable!(),
                };
                names = names
                    .into_iter()
                    .flat_map(|name| {
                        (0..experts)
                            .map(move |expert| name.replace("{expert}", &expert.to_string()))
                    })
                    .collect();
            }
            if !names.is_empty()
                && names
                    .iter()
                    .all(|name| m.tensors.get(name).is_some_and(|t| t.dtype == "BF16"))
            {
                rules.roles.insert(
                    p.id.clone(),
                    HeaderRole {
                        declared: Some(EncodingRecipe::bf16()),
                        ..Default::default()
                    },
                );
            }
        }
    }
    for id in ["ga.qkv", "swa.qkv", "mtp.qkv"] {
        rules
            .roles
            .entry(id.into())
            .or_default()
            .calibration_sharing = Some(ScalarSharing::Frame);
    }
    rules
}
pub(crate) fn bind(
    d: &Declaration,
    g: &Geometry,
    m: &CheckpointMetadata,
) -> Result<projection_spec::bind::BoundDeclaration, Error> {
    d.bind(
        g,
        &projections::registry().unwrap(),
        m,
        &header_rules(d, g, m),
    )
}
pub(crate) fn owner_scope(m: &mut CheckpointMetadata) {
    let expected = model_contract::encoders::expected_flash();
    let mut names = BTreeSet::new();
    for (name, (dtype, shape, _)) in expected {
        let tensor = &m.tensors[&name];
        assert_eq!(tensor.dtype, dtype);
        assert_eq!(tensor.shape, shape);
        names.insert(name);
    }
    m.validated_owners.insert("encoder-spec".into(), names);
}
