//! Metadata fixtures contain header facts only; no checkpoint payload is read.
use kernel_core::contract::WeightFormat;
use model_compiler_sdk::quantization::import::EncodingRecipe;
use model_compiler_sdk::quantization::projection::metadata::{
    CompanionNames, HeaderCompanionRules, HeaderRole,
};
use model_contract::projections;
use model_qwen38_contract as model_contract;
use projection_spec::bind::CheckpointMetadata;
use projection_spec::{Declaration, Error, Geometry};
use std::collections::BTreeMap;

// The checkpoint header helpers every model shares, from the SDK's test
// support sources (outside its module tree).
#[path = "../../../../../crates/model-compiler-sdk/test-support/checkpoint.rs"]
pub(crate) mod checkpoint;
pub(crate) use checkpoint::fixture;
pub(crate) mod exports;

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
    use kernel_core::precision::format::FormatId;
    let fp8 = m
        .tensors
        .iter()
        .find(|(n, _)| n.contains("ngram_embedding.shard_") && n.ends_with(".weight"))
        .is_some_and(|(_, t)| t.dtype == "F8_E4M3");
    rules.roles.insert("ngram".into(), HeaderRole {
        targets: vec!["model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding".into(),
            "ple.ple_embedding.ngram_embedding".into()],
        declared: Some(EncodingRecipe { weight: if fp8 { WeightFormat::Fp8PerTensor } else { WeightFormat::Nvfp4 }, activation: None }),
        format_companions: BTreeMap::from([
            (FormatId::Fp8PerTensor, CompanionNames { local: Some("model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding.weight_scale".into()), global: None }),
            (FormatId::Nvfp4, CompanionNames { local: None, global: Some("model.language_model.layers.{i}.ple.ple_embedding.ngram_embedding.weight_scale_2".into()) }),
        ]), ..Default::default()
    });
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
pub(crate) fn owner_scope(d: &Declaration, m: &mut CheckpointMetadata) -> Declaration {
    let d = d.clone();
    let views = model_contract::encoder::views::layout_flash();
    let present = views
        .iter()
        .filter(|v| m.tensors.contains_key(&v.name))
        .count();
    if present != 0 {
        assert_eq!(
            present,
            views.len(),
            "partial encoder owner cannot be omitted"
        );
        for v in &views {
            assert_eq!(m.tensors[&v.name].shape, v.source_shape);
            assert_eq!(m.tensors[&v.name].dtype, "BF16");
        }
        m.validated_owners.insert(
            "encoder-spec".into(),
            views.into_iter().map(|v| v.name).collect(),
        );
    }
    d
}
