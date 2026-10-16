//! Precision inputs to the existing preparation and weight/package caches.
//! This module computes identities; it owns no cache directories or publication.

use super::import::IMPORTER_VERSION;
use crate::CompilerError;
use kernel_core::precision::execution::{PrecisionDocument, PrecisionSelection};
use kernel_core::precision::{Calibration, WeightLayout};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlacementIdentity {
    pub rank: u32,
    pub source_origin: Vec<u64>,
    pub logical_shape: Vec<u64>,
    pub physical_shape: Vec<u64>,
    pub layout: WeightLayout,
    pub placement: String,
}

pub struct IdentityInputs<'a> {
    /// Authenticated contents of every consumed tensor, including source input
    /// calibration that a BF16 effective recipe deliberately does not execute.
    pub tensor_digests: &'a BTreeMap<String, String>,
    /// Include metadata absence as an explicit caller-provided identity entry.
    pub metadata_digests: &'a BTreeMap<String, String>,
    /// One record per prepared weight operand and rank, in packing order.
    pub placements: &'a [PlacementIdentity],
    /// Identity of the supplied implementation set, including absent buckets.
    pub availability_digest: &'a str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PrecisionIdentities {
    pub interpretation: String,
    pub packed_operands: String,
    pub execution: String,
}

/// Bounded human-readable choices for preparation reports. The precision
/// component records individual operations, calibration and fallback reasons.
pub fn report(document: &PrecisionDocument) -> Result<Vec<String>, CompilerError> {
    document.validate()?;
    let mut groups = BTreeMap::<String, usize>::new();
    for s in &document.operations {
        let source: Vec<_> = s.source.activations.iter().map(|a| a.format).collect();
        let effective: Vec<_> = s.effective.activations.iter().map(|a| a.format).collect();
        let weights: Vec<_> = s.source.weights.iter().map(|w| w.format).collect();
        let mut description = format!(
            "activation policy {}: weights {weights:?}, source activations {source:?}, effective activations {effective:?}, output {:?}, reuse {:?}",
            s.policy, s.effective.arithmetic.output, s.reuse
        );
        if let Some(reason) = s.fallback_reasons.first() {
            description.push_str(&format!(
                "; checkpoint recipe unavailable: {reason}; full operation/use details in the authenticated precision component"
            ));
        }
        *groups.entry(description).or_default() += 1;
    }
    Ok(groups
        .into_iter()
        .map(|(choice, count)| format!("{choice}: {count} operations"))
        .collect())
}

fn digest(value: &impl Serialize) -> Result<String, CompilerError> {
    let bytes = serde_json::to_vec(value).map_err(|e| CompilerError::validation(e.to_string()))?;
    Ok(runtime_weights::sha256_bytes(&bytes))
}

fn content<'a>(
    digests: &'a BTreeMap<String, String>,
    name: &str,
) -> Result<&'a str, CompilerError> {
    let value = digests.get(name).ok_or_else(|| {
        CompilerError::validation(format!(
            "tensor {name}: missing authenticated content identity"
        ))
    })?;
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(CompilerError::validation(format!(
            "tensor {name}: expected a canonical SHA-256 content identity"
        )));
    }
    Ok(value)
}

pub fn identities(
    selection: &PrecisionSelection,
    inputs: IdentityInputs<'_>,
) -> Result<PrecisionIdentities, CompilerError> {
    selection
        .validate()
        .map_err(|e| CompilerError::validation(e.to_string()))?;
    if inputs.placements.is_empty() || inputs.availability_digest.is_empty() {
        return Err(CompilerError::validation(
            "precision identity requires placement and supplied implementation dependencies",
        ));
    }
    let mut packed_contents = BTreeMap::new();
    for weight in &selection.source.weights {
        for tensor in std::iter::once(&weight.codes).chain(
            weight
                .local_scale
                .iter()
                .chain(weight.global_scale.iter())
                .map(|s| &s.tensor),
        ) {
            packed_contents.insert(
                tensor.tensor.as_str(),
                content(inputs.tensor_digests, &tensor.tensor)?,
            );
        }
    }
    let mut source_contents = packed_contents.clone();
    for activation in &selection.source.activations {
        if let Some(q) = &activation.quantizer
            && let Calibration::Static(scale) = &q.calibration
        {
            source_contents.insert(
                scale.tensor.tensor.as_str(),
                content(inputs.tensor_digests, &scale.tensor.tensor)?,
            );
        }
    }
    let interpretation = digest(&(
        "precision-interpretation-v1",
        IMPORTER_VERSION,
        &selection.source,
        source_contents,
        inputs.metadata_digests,
    ))?;
    // Policy and unused activation calibration do not change compatible weight
    // bytes. Encoding, transforms, scale contents and layout always participate.
    let packed_operands = digest(&(
        "precision-packed-v1",
        IMPORTER_VERSION,
        &selection.source.weights,
        packed_contents,
        inputs.placements,
    ))?;
    let execution = digest(&(
        "precision-execution-v1",
        &interpretation,
        &packed_operands,
        selection,
        inputs.availability_digest,
    ))?;
    Ok(PrecisionIdentities {
        interpretation,
        packed_operands,
        execution,
    })
}

pub fn encode_component(document: &PrecisionDocument) -> Result<Vec<u8>, CompilerError> {
    Ok(kernel_core::precision::component::encode(document)?)
}

pub fn decode_component(bytes: &[u8]) -> Result<PrecisionDocument, CompilerError> {
    Ok(kernel_core::precision::component::decode(bytes)?)
}

/// Read a standalone prepared component through a retained regular descriptor.
/// The caller must still match its source, layouts and supplied implementations.
pub fn read_component(path: &std::path::Path) -> Result<PrecisionDocument, CompilerError> {
    let failure = |error: std::io::Error| {
        CompilerError::validation(format!("read precision component: {error}"))
    };
    let source = runtime_weights::source_file::SourceFile::open(path).map_err(failure)?;
    let length = source.metadata().map_err(failure)?.len();
    if length > kernel_core::precision::component::MAX_COMPONENT_BYTES as u64 {
        return Err(CompilerError::validation(
            "precision component exceeds its byte bound",
        ));
    }
    let mut bytes = vec![0; length as usize];
    runtime_weights::read_exact_at(&source, &mut bytes, 0)
        .map_err(|error| CompilerError::validation(format!("read precision component: {error}")))?;
    source.verify().map_err(failure)?;
    decode_component(&bytes)
}
