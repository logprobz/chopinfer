//! Header-only imports for declarations. Source authentication and file IO stay
//! with checkpoint readers; all offsets below are relative to a named tensor.

mod parts;

use std::collections::BTreeMap;

use kernel_core::contract::DType;
use kernel_core::precision::TensorSpan;
use kernel_core::precision::format::{FormatId, registry};
use projection_spec::bind::{
    BindingRequest, CheckpointMetadata, CompanionRules, FrameCalibration, ImportedOperand,
    TensorRequirement,
};
use projection_spec::{Error, ProjectionKind};
use serde::{Deserialize, Serialize};

use crate::quantization::import::{
    EncodingRecipe, ExportMetadata, ImportOptions, LINEAR_CLASS_ROLE, OperandRole,
    ROUTED_EXPERT_ROLE, import_operand,
};

/// Companion names can use {stem}, {i}, {operand}, {expert}, {shard} and {head}.
/// Names and export target aliases describe source roles, never scale arithmetic.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompanionNames {
    pub local: Option<String>,
    pub global: Option<String>,
}

/// An explicit scalar-vector storage mapping for independent matrix frames.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarSharing {
    Shared,
    Expert,
    Operand,
    Frame,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderRole {
    #[serde(default)]
    pub targets: Vec<String>,
    pub declared: Option<EncodingRecipe>,
    #[serde(default)]
    pub companions: CompanionNames,
    #[serde(default)]
    pub format_companions: BTreeMap<FormatId, CompanionNames>,
    pub global_sharing: Option<ScalarSharing>,
    pub calibration_sharing: Option<ScalarSharing>,
}

/// The SDK depends on projection-spec, so the shared adapter lives here.
/// Roles are keyed by projection id. Ordinary exports need no role overrides.
pub struct HeaderCompanionRules {
    export: ExportMetadata,
    quantization_config: serde_json::Value,
    pub roles: BTreeMap<String, HeaderRole>,
}

fn error(e: impl std::fmt::Display) -> Error {
    Error(format!("projection metadata: {e}"))
}

fn span(name: &str, m: &CheckpointMetadata) -> Result<TensorSpan, Error> {
    let t = m
        .tensors
        .get(name)
        .ok_or_else(|| error(format!("missing tensor `{name}`")))?;
    let dtype = [
        DType::Bf16,
        DType::F16,
        DType::F32,
        DType::I32,
        DType::I64,
        DType::U8,
        DType::Fp8E4M3,
        DType::Fp8E5M2,
        DType::Ue8M0,
    ]
    .into_iter()
    .find(|d| d.checkpoint_dtype() == Some(t.dtype.as_str()))
    .ok_or_else(|| {
        error(format!(
            "tensor `{name}` has unsupported source dtype {}",
            t.dtype
        ))
    })?;
    let elements = kernel_core::precision::shape_elements(name, &t.shape).map_err(error)?;
    let byte_length = elements
        .checked_mul(kernel_core::precision::dtype_bits(dtype) / 8)
        .ok_or_else(|| error("source byte length overflows"))?;
    let span = TensorSpan {
        tensor: name.into(),
        dtype,
        shape: t.shape.clone(),
        offset: 0,
        byte_length,
    };
    span.validate().map_err(error)?;
    Ok(span)
}

fn render(text: &str, stem: &str, r: &BindingRequest<'_>) -> Result<String, Error> {
    let mut text = text.replace("{stem}", stem);
    for (key, value) in [
        ("i", r.layer),
        ("expert", r.expert),
        ("shard", r.shard),
        ("head", r.head),
    ] {
        if let Some(value) = value {
            text = text.replace(&format!("{{{key}}}"), &value.to_string());
        }
    }
    if let Some(i) = r.operand {
        let operand = r
            .projection
            .operands
            .get(i as usize)
            .ok_or_else(|| error("unknown source operand"))?;
        text = text.replace("{operand}", operand);
    }
    if text.is_empty()
        || text.len() > 4096
        || text.contains(['{', '}'])
        || text.chars().any(char::is_control)
    {
        return Err(error("source role has an unresolved or invalid template"));
    }
    Ok(text)
}

impl HeaderCompanionRules {
    pub fn new(m: &CheckpointMetadata, hf_quant_config: Option<&[u8]>) -> Result<Self, Error> {
        Self::new_with(m, hf_quant_config, &ImportOptions::default())
    }

    /// [`Self::new`] under the family's import `options`.
    pub fn new_with(
        m: &CheckpointMetadata,
        hf_quant_config: Option<&[u8]>,
        options: &ImportOptions,
    ) -> Result<Self, Error> {
        let config = (!m.quantization_config.is_null()).then_some(&m.quantization_config);
        let export =
            ExportMetadata::from_documents_with(config, hf_quant_config, options).map_err(error)?;
        Ok(Self {
            export,
            quantization_config: m.quantization_config.clone(),
            roles: BTreeMap::new(),
        })
    }
}

impl CompanionRules for HeaderCompanionRules {
    fn import(
        &self,
        r: &BindingRequest<'_>,
        m: &CheckpointMetadata,
    ) -> Result<ImportedOperand, Error> {
        if m.quantization_config != self.quantization_config {
            return Err(error(
                "quantization config changed after import rules were constructed",
            ));
        }
        let stem = r
            .tensor
            .strip_suffix(".weight")
            .ok_or_else(|| error("weight source must end in .weight"))?;
        let default = HeaderRole::default();
        let hints = self.roles.get(&r.projection.id).unwrap_or(&default);
        let mut targets = if hints.targets.is_empty() {
            vec![stem.into()]
        } else {
            hints
                .targets
                .iter()
                .map(|s| render(s, stem, r))
                .collect::<Result<Vec<_>, _>>()?
        };
        if matches!(r.projection.kind, ProjectionKind::ExpertBank { .. }) {
            targets.push(ROUTED_EXPERT_ROLE.into());
        }
        let lookup = matches!(r.projection.kind, ProjectionKind::LookupTable {});
        // A dense or expert-bank projection is a linear map: a config group
        // that targets the `Linear` class reaches it, a table does not.
        if !lookup {
            targets.push(LINEAR_CLASS_ROLE.into());
        }
        let declared = hints.declared.map(|mut recipe| {
            if lookup {
                recipe.activation = None;
            }
            recipe
        });
        let recipe = self.export.recipe(&targets, declared).map_err(error)?;
        let plugin = match r.projection.format {
            Some(id) => registry::by_id(id),
            None => registry::by_format(recipe.weight),
        }
        .map_err(error)?;
        if !plugin.accepts(recipe.weight) {
            return Err(error("declared format disagrees with export weight recipe"));
        }
        let prefix = format!("{stem}.");
        let mut tensors = m
            .tensors
            .range(prefix.clone()..)
            .take_while(|(n, _)| n.starts_with(&prefix))
            .map(|(n, _)| Ok((n.clone(), span(n, m)?)))
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        let names = hints
            .format_companions
            .get(&plugin.id())
            .unwrap_or(&hints.companions);
        for (name, suffix, needed) in [
            (
                &names.local,
                "weight_scale",
                recipe.weight != kernel_core::contract::WeightFormat::Bf16,
            ),
            (
                &names.global,
                "weight_scale_2",
                recipe.weight == kernel_core::contract::WeightFormat::Nvfp4,
            ),
        ] {
            if needed && let Some(name) = name {
                let name = render(name, stem, r)?;
                // Retain the actual tensor identity inside the alias span.
                let candidates = if suffix == "weight_scale" {
                    vec!["weight_scale_inv", "weight_scale"]
                } else {
                    vec![suffix]
                };
                for candidate in &candidates {
                    let key = format!("{stem}.{candidate}");
                    if tensors.get(&key).is_some_and(|s| s.tensor != name) {
                        return Err(error(
                            "companion alias conflicts with an existing source tensor",
                        ));
                    }
                }
                let slot = candidates
                    .iter()
                    .find(|suffix| tensors.contains_key(&format!("{stem}.{suffix}")))
                    .copied()
                    .unwrap_or(suffix);
                tensors.insert(format!("{stem}.{slot}"), span(&name, m)?);
            }
        }
        let role = OperandRole {
            operation: r.projection.id.clone(),
            weight: r.tensor.into(),
            targets,
            logical_shape: r.logical_shape.into(),
            lookup,
            shared_scale_stem: None,
            declared,
        };
        let independent_scalars = recipe.weight == kernel_core::contract::WeightFormat::Nvfp4
            && (matches!(
                hints.global_sharing,
                Some(ScalarSharing::Operand | ScalarSharing::Frame)
            ) || (!lookup
                && matches!(
                    hints.calibration_sharing,
                    Some(ScalarSharing::Operand | ScalarSharing::Frame)
                )));
        let (weight, parts, consumed, calibrations) = if r.projection.parts.is_empty()
            && !independent_scalars
        {
            let imported = import_operand(&self.export, &role, &tensors).map_err(error)?;
            plugin.companions(&imported.weight).map_err(error)?;
            if let Some(global) = &imported.weight.global_scale
                && hints.global_sharing.is_some()
            {
                let source_experts = if role.logical_shape.len() == 3 {
                    role.logical_shape[0]
                } else {
                    1
                };
                let source_operands = if r.operand.is_some() {
                    1
                } else {
                    r.projection.operand_count() as u64
                };
                for expert in 0..source_experts {
                    for operand in 0..source_operands {
                        parts::scalar(
                            &global.tensor,
                            hints.global_sharing,
                            parts::frame(r, expert, operand, 0, 1)?,
                        )?;
                    }
                }
            }
            let calibrations =
                static_calibrations(r, imported.activation.as_ref(), hints.calibration_sharing)?;
            (
                imported.weight,
                vec![],
                imported.source_tensors,
                calibrations,
            )
        } else {
            parts::import(&self.export, &role, &tensors, r, hints, plugin)?
        };
        let companions = companion_requirements(&weight, consumed, m)?;
        Ok(ImportedOperand {
            weight,
            parts,
            companions,
            calibrations,
        })
    }
}

/// The static input calibration of one bound source, resolved per source
/// frame: every expert of a stacked bank and every operand the source holds.
/// `sharing` states how a scalar vector maps to frames; without it a single
/// scalar is shared and a vector holds one scalar per expert.
pub(crate) fn static_calibrations(
    r: &BindingRequest<'_>,
    activation: Option<&kernel_core::precision::ActivationSpec>,
    sharing: Option<ScalarSharing>,
) -> Result<Vec<FrameCalibration>, Error> {
    let Some(scale) =
        activation
            .and_then(|a| a.quantizer.as_ref())
            .and_then(|q| match &q.calibration {
                kernel_core::precision::Calibration::Static(s) => Some(s),
                _ => None,
            })
    else {
        return Ok(vec![]);
    };
    let source_experts = if r.logical_shape.len() == 3 {
        r.logical_shape[0]
    } else {
        1
    };
    let source_operands = if r.operand.is_some() {
        1
    } else {
        u64::from(r.projection.operand_count())
    };
    let mut calibrations = Vec::new();
    for expert in 0..source_experts {
        for operand in 0..source_operands {
            let f = parts::frame(r, expert, operand, 0, 1)?;
            let mut resolved = scale.clone();
            resolved.tensor = parts::scalar(&scale.tensor, sharing, f)?;
            resolved.scope = kernel_core::precision::ScaleScope::Projection;
            calibrations.push(FrameCalibration {
                expert: f.expert,
                operand: f.operand,
                part: None,
                scale: resolved,
            });
        }
    }
    Ok(calibrations)
}

/// Authenticated requirements for every consumed source tensor that is not
/// the weight's codes or scales, such as a static input calibration.
pub(crate) fn companion_requirements(
    weight: &kernel_core::precision::WeightSpec,
    consumed: impl IntoIterator<Item = String>,
    m: &CheckpointMetadata,
) -> Result<Vec<TensorRequirement>, Error> {
    let standard = std::iter::once(&weight.codes)
        .chain(weight.local_scale.iter().map(|s| &s.tensor))
        .chain(weight.global_scale.iter().map(|s| &s.tensor))
        .map(|s| s.tensor.as_str())
        .collect::<Vec<_>>();
    let mut companions = Vec::new();
    for name in consumed {
        if standard.contains(&name.as_str())
            || companions
                .iter()
                .any(|c: &TensorRequirement| c.tensor == name)
        {
            continue;
        }
        let t = m
            .tensors
            .get(&name)
            .ok_or_else(|| error("consumed companion has no source header"))?;
        companions.push(TensorRequirement {
            tensor: name,
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            file: t.file.clone(),
        });
    }
    Ok(companions)
}
