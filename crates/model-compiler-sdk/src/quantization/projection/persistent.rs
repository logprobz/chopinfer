//! Serializable metadata for existing host ABI records and authenticated sources.
use crate::CompilerError;
use kernel_core::contract::DType;
use kernel_core::precision::TensorSpan;
use kernel_core::weight_transform::PackedSection;
use projection_spec::derive::{DerivedOperation, RankDerivation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;

fn error(reason: &str) -> CompilerError {
    CompilerError::validation(format!("persistent projection: {reason}"))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PersistentRecord {
    pub view: String,
    pub view_offset: u64,
    pub section: PackedSection,
    pub bytes: u64,
    pub recipe: RecordRecipe,
}

/// Existing transform identifiers and parameters. This descriptor introduces
/// no transform or CPU executor; consumers use the existing host record ABI.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordRecipe {
    pub transform: u32,
    pub params: Vec<u8>,
    pub sources: Vec<TensorSpan>,
    pub axis: Option<u32>,
    pub ranges: Vec<Range<u64>>,
    pub stored_shape: Vec<u64>,
}

/// Retain destination placement and exact source spans in resident identity.
/// Offset/multipart sources need provider-owned records using existing transforms.
pub fn records(operation: &DerivedOperation) -> Result<Vec<PersistentRecord>, CompilerError> {
    // Every authenticated span by tensor name, record sources first.
    let mut spans = BTreeMap::<&str, Vec<&TensorSpan>>::new();
    for span in operation
        .record_sources
        .iter()
        .chain(operation.slices.iter().flat_map(|slice| {
            std::iter::once(&slice.source.weight.codes)
                .chain(slice.source.weight.local_scale.iter().map(|s| &s.tensor))
                .chain(slice.source.weight.global_scale.iter().map(|s| &s.tensor))
        }))
    {
        spans.entry(span.tensor.as_str()).or_default().push(span);
    }
    super::records(operation)?
        .into_iter()
        .map(|r| {
            let sources = r
                .record
                .sources
                .iter()
                .zip(&r.tensors)
                .map(|(s, name)| {
                    let candidates = spans
                        .get(name.as_str())
                        .into_iter()
                        .flatten()
                        .filter(|span| {
                            span.byte_length == s.byte_length
                                && (span.dtype.checkpoint_dtype() == Some(s.dtype.as_str())
                                    || (span.dtype == DType::Ue8M0 && s.dtype == "U8"))
                        })
                        .collect::<Vec<_>>();
                    let first = candidates
                        .first()
                        .ok_or_else(|| error("record has no authenticated source view"))?;
                    if candidates
                        .iter()
                        .any(|span| span.offset != first.offset || span.dtype != first.dtype)
                    {
                        return Err(error("record has ambiguous authenticated source view"));
                    }
                    Ok(TensorSpan {
                        tensor: name.clone(),
                        dtype: first.dtype,
                        shape: s.shape.clone(),
                        offset: first.offset,
                        byte_length: s.byte_length,
                    })
                })
                .collect::<Result<Vec<_>, CompilerError>>()?;
            Ok(PersistentRecord {
                view: r.view,
                view_offset: r.view_offset,
                section: r.section,
                bytes: r.record.output_bytes().map_err(|e| error(&e.to_string()))?,
                recipe: RecordRecipe {
                    transform: r.record.transform.id(),
                    params: r
                        .record
                        .transform
                        .encode_params()
                        .map_err(|e| error(&e.to_string()))?,
                    sources,
                    axis: r.record.axis,
                    ranges: r.record.ranges,
                    stored_shape: r.record.stored_shape,
                },
            })
        })
        .collect()
}

/// Deduplicate complete resident recipes across requested buckets. Source and
/// view offsets participate in identity, including each fragment.
pub fn rank_records(rank: &RankDerivation) -> Result<Vec<PersistentRecord>, CompilerError> {
    let mut resident = BTreeMap::<String, Vec<PersistentRecord>>::new();
    for operation in rank.selected_operations() {
        let mut grouped = BTreeMap::<String, Vec<PersistentRecord>>::new();
        for record in records(operation)? {
            grouped.entry(record.view.clone()).or_default().push(record);
        }
        for (view, mut pieces) in grouped {
            pieces.sort_by_key(|r| r.view_offset);
            if let Some(previous) = resident.get(&view) {
                if *previous != pieces {
                    return Err(error(
                        "one resident view has conflicting persistent recipes",
                    ));
                }
            } else {
                resident.insert(view, pieces);
            }
        }
    }
    Ok(resident.into_values().flatten().collect())
}
