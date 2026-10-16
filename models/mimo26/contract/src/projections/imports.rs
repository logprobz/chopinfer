//! Production declaration inputs from operands admitted against checkpoint headers.
use std::collections::BTreeMap;

use model_compiler_sdk::quantization::declared::{self, Admitted};
use projection_spec::Error;
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{RankDerivation, Topology};

use crate::{Tp, catalog::TensorRecord};

fn error(e: impl std::fmt::Display) -> Error {
    Error(format!("admitted projection imports: {e}"))
}

/// The binding view of checkpoint headers.
pub fn metadata(tensors: &BTreeMap<String, TensorRecord>) -> CheckpointMetadata {
    CheckpointMetadata {
        tensors: tensors
            .iter()
            .map(|(name, t)| {
                (
                    name.clone(),
                    TensorInfo {
                        dtype: t.dtype.safetensors_name().into(),
                        shape: t.shape.clone(),
                        file: Some(t.file.clone()),
                    },
                )
            })
            .collect(),
        ..Default::default()
    }
}

/// Derive the main decoder and the optional admitted MTP or DFlash drafter
/// sources. The caller owns authenticated files; this builder cannot invent
/// checkpoint recipes.
pub fn derive(
    tp: Tp,
    rank: u32,
    tensors: &BTreeMap<String, TensorRecord>,
    admitted: &Admitted,
) -> Result<RankDerivation, Error> {
    let (declaration, geometry) = super::expanded(tp)?;
    let mut declaration = declaration.clone();
    let mtp = admitted
        .keys()
        .any(|name| name.starts_with("model.mtp.layers."));
    // The DFlash drafter's context projection reads its own `fc.weight`.
    let dflash = admitted.contains_key("fc.weight");
    declaration.projections.retain(|p| {
        if p.id.starts_with("dflash.") {
            dflash && p.id != "dflash.embedding"
        } else {
            mtp || !p.id.starts_with("mtp")
        }
    });
    declaration.side_tensors.retain(|p| {
        if p.id.starts_with("dflash.") {
            dflash
        } else {
            mtp || !p.id.starts_with("mtp")
        }
    });
    declaration.external_tensors.clear();
    let metadata = metadata(tensors);
    let mut graph = declared::derive(declared::RankInputs {
        declaration: &declaration,
        geometry,
        registry: &super::registry()?,
        variants: &super::variants::Variants(tp.geometry),
        metadata: &metadata,
        admitted,
        topology: Topology {
            ranks: tp.ranks(),
            rank,
        },
    })?;
    // The drafter's fused QKV assembly reads its members' sources; they run
    // only through it.
    graph.operations.retain(|o| {
        !o.instance.projection.starts_with("dflash.")
            || crate::capabilities::served(tp.geometry, &o.instance.projection).is_some()
    });
    // MTP adapters are complete views of source layer zero. Retain its
    // authenticated source spans instead of naming a new assembled operand.
    let sources: Vec<_> = graph
        .operations
        .iter()
        .filter(|o| !o.instance.projection.contains("-adapter."))
        .map(|o| {
            (
                o.instance.clone(),
                o.requested,
                o.logical_geometry,
                o.operation.epilogues,
                o.operation.weights.clone(),
            )
        })
        .collect();
    for operation in &mut graph.operations {
        if operation.instance.projection.contains("-adapter.") {
            let source = sources
                .iter()
                .find(|candidate| {
                    candidate.0 == operation.source_instance && candidate.1 == operation.requested
                })
                .ok_or_else(|| error("MTP adapter source use is absent"))?;
            if operation.logical_geometry != source.2 || operation.operation.epilogues != source.3 {
                return Err(error(
                    "MTP adapter changes its native source geometry or epilogues",
                ));
            }
            operation.operation.weights = source.4.clone();
            operation.operation.validate().map_err(error)?;
        }
    }
    Ok(graph)
}
