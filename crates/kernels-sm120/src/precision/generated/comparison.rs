//! Transitional comparison inputs for retained catalogs. These synthetic spans
//! describe rank-local operands; they are never checkpoint import records.

use super::*;
use kernel_core::precision::*;

/// Build a valid portable operation for the domain advertised by a hand list.
/// Callers check model geometry independently, and later supply declaration
/// derivations at this boundary without changing generation or adoption.
pub fn operation(capability: &OperationCapability) -> Result<OperationSpec, String> {
    capability.validate()?;
    let activations = match capability.activation {
        None => vec![],
        Some(ActivationCapability::Bf16) => {
            vec![ActivationSpec::bf16(); capability.operands as usize]
        }
        _ => return Err("catalog comparison currently requires BF16 activation operands".into()),
    };
    let mut operation = domain(Domain {
        kind: capability.kind,
        target: capability.target,
        geometry: capability.geometry,
        operands: capability.operands,
        weight: capability.weight,
        local_scale: capability.local_scale_dtypes.first().copied(),
        global_scale: match capability.globals {
            GlobalScaleCapability::Absent => None,
            GlobalScaleCapability::SharedScalar => Some(ScaleScope::Tensor),
            GlobalScaleCapability::ExpertProjection => Some(ScaleScope::Expert),
        },
        activations,
        arithmetic: capability.arithmetic.clone(),
        epilogues: capability.epilogues,
        usage: capability.usage,
    })?;
    operation.id = "catalog-comparison".into();
    Ok(operation)
}

/// Rebuild retained typed schedules from an independently supplied operation.
/// This bridge keeps the old finite schedule inventory until declaration ports.
pub fn regenerate(
    operation: &OperationSpec,
    existing: &KernelOffer,
    adoptions: &[Adoption],
) -> Result<GeneratedOffer, String> {
    operation.validate().map_err(|e| e.to_string())?;
    let implementation = implementation(existing)?;
    offer(
        Request {
            operation,
            format: registry::by_format(operation.weights[0].format)
                .map_err(|e| e.to_string())?
                .id(),
            usage: existing.usage,
            implementation,
        },
        adoptions,
    )
    .map(|mut generated| {
        generated.offer.invariance = existing.invariance();
        generated
    })
}

/// Capture a retained implementation for override export and equality checks.
pub fn implementation(existing: &KernelOffer) -> Result<Implementation, String> {
    Ok(match &existing.handle.compute {
        Compute::Dense(spec) => {
            let StaticQuery::Gemm(query) = spec.key.query else {
                return Err("not a dense query".into());
            };
            Implementation::Dense {
                kernel: gemm::kernel(&spec.key).map_err(|e| e.to_string())?,
                capacity: query.m_capacity,
                knobs: Some(spec.key.knobs),
            }
        }
        Compute::Grouped(g) => Implementation::Grouped {
            phase: g.phase,
            capacity: g.tokens_capacity,
            knobs: Some(g.knobs),
            scales: g
                .packed_contract
                .ok_or("catalog grouped entry needs its registered adapter")?
                .scales,
            route_weight: moe::RouteWeight::AfterDown,
            swiglu_clamp: None,
        },
        Compute::Lookup(_) => {
            let r = existing.lookup.ok_or("lookup request is missing")?;
            Implementation::Lookup {
                heads: r.heads,
                ids: r.ids,
                scale_dtype: r.global_scale,
                shapes: r.shapes,
            }
        }
    })
}

/// Preserve a registered catalog implementation while checking its operation
/// domain and deriving identities for its existing entry bodies. Catalog
/// adapters still own arithmetic that the ordinary typed builders cannot state.
pub fn catalog(
    operation: &OperationSpec,
    capability: OperationCapability,
    specs: &[CubinSpec],
    declarations: &BTreeMap<String, serde_json::Value>,
) -> Result<(OperationCapability, Vec<Adoption>), String> {
    if !operation.uses.contains(&capability.usage) {
        return Err("catalog use is not declared by the operation".into());
    }
    let document = CapabilityDocument {
        version: 1,
        operations: vec![capability.clone()],
    };
    let inventory = CapabilityInventory::new(document.clone(), document.entries(), "0".repeat(64))?;
    inventory.bind(operation, capability.usage)?;
    let format = registry::by_format(capability.weight)
        .map_err(|e| e.to_string())?
        .id();
    let mut records = Vec::new();
    for symbol in &capability.stages {
        let matches: Vec<_> = specs
            .iter()
            .flat_map(|s| s.entries.iter().map(move |e| (s, e)))
            .filter(|(_, e)| &e.symbol == symbol)
            .collect();
        if matches.len() != 1 {
            return Err(format!(
                "catalog stage {symbol} has {} definitions",
                matches.len()
            ));
        }
        let (spec, entry) = matches[0];
        let kind = match capability.kind {
            OperationKind::Dense => "dense",
            OperationKind::Grouped { .. } => "grouped",
            OperationKind::Lookup => "lookup",
        };
        records.push(identity::record(
            format,
            kind,
            Some(capability.geometry),
            spec,
            entry,
            declarations
                .get(symbol)
                .ok_or_else(|| format!("missing normalized declaration for {symbol}"))?
                .clone(),
        ));
    }
    Ok((capability, records))
}
