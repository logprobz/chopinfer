//! Execution aliases resolve by declared tensor names, never by matrix shape.
use super::*;
use crate::{Error, TensorSource};

fn name(
    source: &TensorSource,
    stem: &str,
    index: u64,
    b: &crate::bind::BoundOperand,
    p: &Projection,
) -> Option<String> {
    let mut name = source
        .tensor
        .replace("{stem}", stem)
        .replace("{i}", &index.to_string());
    if name.contains("{operand}") {
        name = name.replace("{operand}", p.operands.get(b.operand? as usize)?);
    }
    if name.contains("{expert}") {
        name = name.replace("{expert}", &b.expert?.to_string());
    }
    (!name.contains(['{', '}'])).then_some(name)
}

pub(super) fn source(
    declaration: &crate::Declaration,
    bound: &crate::bind::BoundDeclaration,
    geometry: &crate::Geometry,
    p: &Projection,
    stem_id: &str,
    index: u64,
) -> Result<InstanceKey, Error> {
    let stem = declaration
        .layer_stems
        .iter()
        .find(|s| s.id == stem_id)
        .ok_or_else(|| Error("alias source stem is missing".into()))?;
    let prefix = stem.template.replace("{i}", &index.to_string());
    let shards = p
        .lookup
        .as_ref()
        .map(|t| t.shards.resolve(geometry))
        .transpose()?
        .unwrap_or_default();
    let sources = p
        .source
        .iter()
        .chain(shards.iter().map(|s| &s.source))
        .collect::<Vec<_>>();
    if sources.is_empty() {
        // A pass-through assembly aliases its explicitly named source role.
        // The source may be a primitive with no assembly of its own.
        let candidates = p
            .assembly
            .iter()
            .map(|member| {
                let role = declaration
                    .projections
                    .iter()
                    .find(|s| s.id == member.projection)
                    .ok_or_else(|| Error("alias assembly source role is missing".into()))?;
                let layer = member
                    .layer
                    .as_ref()
                    .map(|l| l.resolve(geometry, Some(index)))
                    .transpose()?;
                if role.stem() != Some(stem_id) || layer != Some(index) {
                    return Err(Error(
                        "alias assembly source role disagrees with its declared source stem/layer"
                            .into(),
                    ));
                }
                Ok(InstanceKey {
                    projection: role.id.clone(),
                    layer,
                })
            })
            .collect::<Result<std::collections::BTreeSet<_>, Error>>()?;
        if candidates.len() != 1 {
            return Err(Error(
                "cross-stem assembly alias needs one explicit source role".into(),
            ));
        }
        return Ok(candidates.into_iter().next().unwrap());
    }
    let candidates = bound
        .operands
        .iter()
        .filter(|b| {
            sources.iter().any(|s| {
                name(s, &prefix, index, b, p).is_some_and(|n| n == b.imported.weight.codes.tensor)
            })
        })
        .map(|b| InstanceKey {
            projection: b.projection.clone(),
            layer: b.layer,
        })
        .collect::<std::collections::BTreeSet<_>>();
    if candidates.len() != 1 {
        return Err(Error(
            "execution alias source names do not resolve to one authenticated projection instance"
                .into(),
        ));
    }
    Ok(candidates.into_iter().next().unwrap())
}

pub(super) fn prototype<'a>(
    input: &DerivationInput<'a>,
    p: &Projection,
    source: &InstanceKey,
) -> Result<ImportedProjection, Error> {
    let selected = input.imported.get(source).ok_or_else(|| {
        Error(format!(
            "missing import for source projection `{}` layer {:?}",
            source.projection, source.layer
        ))
    })?;
    // Keys order by projection first, so its first instance starts the range.
    let own = input
        .imported
        .range(
            InstanceKey {
                projection: p.id.clone(),
                layer: None,
            }..,
        )
        .next()
        .filter(|(key, _)| key.projection == p.id)
        .map(|(_, i)| i);
    let mut imported = selected.clone();
    if source.projection != p.id {
        let own = own.ok_or_else(|| {
            Error("execution alias needs its own operation identity/recipe template".into())
        })?;
        imported.operation_id = own.operation_id.clone();
        imported.target = own.target;
        imported.family = own.family;
        imported.geometry = own.geometry;
        imported.arithmetic.numerical_contract = own.arithmetic.numerical_contract.clone();
    }
    Ok(imported)
}

/// Source formats come from authenticated headers, including assembly members.
pub(super) fn formats(bound: &BoundDeclaration, key: &InstanceKey) -> Vec<FormatId> {
    let names = bound
        .assemblies
        .iter()
        .find(|a| a.projection == key.projection && a.layer == key.layer)
        .map(|a| {
            a.tensors
                .iter()
                .flatten()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();
    super::index::operands(bound, Some((&key.projection, key.layer)), &names)
        .into_iter()
        .map(|b| b.format)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The assembly's materialization authenticates the destination mapping;
/// its primitive source key independently retains the execution alias origin.
pub(super) fn assembly_layer(
    bound: &BoundDeclaration,
    projection: &Projection,
    execution_layer: Option<u64>,
) -> Result<Option<u64>, Error> {
    let candidates: Vec<_> = bound
        .assemblies
        .iter()
        .filter(|a| a.projection == projection.id)
        .collect();
    if let Some(binding) = candidates.iter().find(|a| a.layer == execution_layer) {
        return Ok(binding.layer);
    }
    if candidates.len() == 1 {
        return Ok(candidates[0].layer);
    }
    Err(Error(
        "execution assembly needs one authenticated materialization".into(),
    ))
}
