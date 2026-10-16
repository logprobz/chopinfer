//! Canonical executable declarations shared by typed and retained entries.
use super::*;
use serde_json::{Value, json};

pub fn canonical(value: Value) -> Vec<u8> {
    fn ordered(value: Value) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, ordered(v)))
                    .collect::<BTreeMap<_, _>>()
                    .into_iter()
                    .collect(),
            ),
            Value::Array(values) => Value::Array(values.into_iter().map(ordered).collect()),
            value => value,
        }
    }
    serde_json::to_vec(&ordered(value)).expect("JSON value serialization")
}

pub fn entry(entry: &EntrySpec) -> Value {
    json!({"params":entry.params,"body":entry.body,"max_threads":entry.max_threads,
        "min_blocks":entry.min_blocks,"allowed_local_bytes":entry.allowed_local_bytes})
}

/// An entry's identity. A launch-shape row stage serves every geometry, so
/// it has none: neither its digest nor its name (`chop_proj_<format>_<kind>_
/// <digest>_v1`) states a shape.
pub fn record(
    format: FormatId,
    kind: &str,
    geometry: Option<MatrixGeometry>,
    spec: &CubinSpec,
    wrapper: &EntrySpec,
    declaration: Value,
) -> Adoption {
    let headers = spec.headers.iter().collect::<BTreeSet<_>>();
    let defines = spec.defines.iter().collect::<BTreeSet<_>>();
    let bytes = canonical(
        json!({"version":1,"format":format,"kind":kind,"geometry":geometry,
        "headers":headers,"defines":defines,"entry":entry(wrapper),"declaration":declaration}),
    );
    let digest = kernel_build::digest(&bytes);
    let format = serde_json::to_value(format).expect("format identifier");
    let format = format.as_str().expect("format name");
    let generated = match geometry {
        Some(g) => format!(
            "chop_proj_{format}_{kind}_n{}_k{}_{}_v1",
            g.rows,
            g.k,
            &digest[..16]
        ),
        None => format!("chop_proj_{format}_{kind}_{}_v1", &digest[..16]),
    };
    Adoption {
        symbol: wrapper.symbol.clone(),
        generated,
        entry_sha256: digest,
        placement: None,
        replaced_by: None,
    }
}

/// Whether `generated` names a launch-shape row stage: its name states no
/// geometry (`chop_proj_<format>_<kind>_<digest>_v1`), so every geometry
/// and topology serves it under that one name.
pub fn shape_free(generated: &str) -> bool {
    let Some(stem) = generated
        .strip_prefix("chop_proj_")
        .and_then(|s| s.strip_suffix("_v1"))
    else {
        return false;
    };
    let parts = stem.split('_').collect::<Vec<_>>();
    let numbered = |part: &str, prefix: char| {
        part.strip_prefix(prefix)
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    !(parts.len() >= 3
        && numbered(parts[parts.len() - 3], 'n')
        && numbered(parts[parts.len() - 2], 'k'))
}

pub fn declaration(offer: &KernelOffer, symbol: &str) -> Result<Value, String> {
    if let Some(stage) = offer
        .handle
        .row_stages()
        .into_iter()
        .find(|s| s.entry().symbol == symbol)
    {
        return Ok(stage.identity_declaration());
    }
    let counts = |m, routed_rows, split_k| RuntimeCounts {
        m,
        batch: 1,
        context: 0,
        routed_rows,
        split_k,
    };
    let capability = offer.capability().map_err(|e| e.to_string())?;
    Ok(match &offer.handle.compute {
        Compute::Dense(spec) => {
            let StaticQuery::Gemm(query) = spec.key.query else {
                return Err("not a dense query".into());
            };
            let split = gemm::required_split_k(&spec.key)
                .unwrap_or(gemm::choose_split_k(&spec.key).map_err(|e| e.to_string())?);
            json!({"query":query,"knobs":spec.key.knobs,"kernel":gemm::kernel(&spec.key).map_err(|e|e.to_string())?,
                "target":spec.key.target,"arithmetic":capability.arithmetic,
                "launches":[gemm::plan(&spec.key,counts(1,0,split)).map_err(|e|e.to_string())?,gemm::plan(&spec.key,counts(query.m_capacity,0,split)).map_err(|e|e.to_string())?]})
        }
        Compute::Grouped(g) => {
            json!({"query":g.query,"knobs":g.knobs,"phase":g.phase,"capacity":g.tokens_capacity,
            "layout":g.layout,"path":g.path,"weight_stages":g.weight_stages,"packed":g.packed_contract,
            "arithmetic":capability.arithmetic,
            "launches":[g.plan(counts(1,1,g.knobs.split_k_capacity)).map_err(|e|e.to_string())?,g.plan(counts(g.tokens_capacity,g.query.m_capacity,g.knobs.split_k_capacity)).map_err(|e|e.to_string())?]})
        }
        Compute::Lookup(_) => return Err("missing lookup row declaration".into()),
    })
}

/// Retained rows declare their grid rule, not just their wrapper and capability.
pub fn retained_row(
    entry: &crate::row::RowInstantiation,
    capability: &OperationCapability,
) -> Result<Value, String> {
    let kernel_core::contract::RowQuery::EmbeddingGather { vocab, width } = entry.query else {
        return Err("retained lookup declaration requires embedding gather".into());
    };
    let counts = |m| RuntimeCounts {
        m,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    Ok(
        json!({"query":{"embedding_gather":{"vocab":vocab,"width":width}},"width":entry.width,
        "grid":entry.grid,"threads":entry.threads,"invariance":entry.invariance,"arithmetic":capability.arithmetic,
        "launches":[entry.plan(64,counts(1)).map_err(|e|e.to_string())?,entry.plan(64,counts(64)).map_err(|e|e.to_string())?]}),
    )
}

pub fn retained_grouped(
    entry: &crate::moe::Variant,
    capability: &OperationCapability,
) -> Result<Value, String> {
    let counts = |m, routed_rows| RuntimeCounts {
        m,
        batch: 1,
        context: 0,
        routed_rows,
        split_k: entry.knobs.split_k_capacity,
    };
    let one = counts(1, 1);
    let full = counts(entry.tokens_capacity, entry.query.m_capacity);
    Ok(
        json!({"query":entry.query,"knobs":entry.knobs,"phase":entry.phase,"capacity":entry.tokens_capacity,
        "layout":entry.layout,"path":entry.path,"projection":entry.projection,"weight_stages":entry.weight_stages,
        "exact":entry.exact,"route_weight":entry.route_weight,"swiglu_clamp":entry.swiglu_clamp,
        "engine":entry.engine,"schedule":entry.schedule,"invariance":entry.invariance,"arithmetic":capability.arithmetic,
        "launches":[entry.plan(one).map_err(|e|e.to_string())?,entry.plan(full).map_err(|e|e.to_string())?],
        "reductions":[entry.reduction_plan(one).map_err(|e|e.to_string())?,entry.reduction_plan(full).map_err(|e|e.to_string())?]}),
    )
}
