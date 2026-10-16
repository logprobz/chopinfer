//! Admission of prepared operation ports and their persistent rank views.
//! Encoding footprints come from the portable layout contract; this module
//! owns Qwen's role names, geometry, fusion peers and rank partitioning.
use crate::backend::{Kernels, Selected};
use crate::{contract::*, precision, views};
use kernel_core::contract::{ActivationFormat, DType, SwigluColumns, WeightFormat};
use kernel_core::precision::capability::{CapabilityInventory, OperationCapability};
use kernel_core::precision::execution::{ActivationPolicy, PrecisionDocument, PrecisionSelection};
use kernel_core::precision::format::{FormatId, KernelFamily, registry};
use kernel_core::precision::layout::PackingPlan;
use kernel_core::precision::{
    ActivationSpec, Calibration, Clipping, GroupAxis, OperationKind, OperationSpec, QuantizerSpec,
    Rounding, ScaleConvention, ScaleScope, ScaleSpec, TensorSpan,
};
use std::collections::{BTreeMap, BTreeSet};

/// Each topology's precision component: its operations bind that
/// topology's per-rank experts and geometry.
pub fn component_id(shape: Shape) -> String {
    format!("precision.{}", shape.topology().id)
}
pub fn component_path(shape: Shape) -> String {
    format!("components/precision.{}.bin", shape.topology().id)
}
pub const COMPONENT_ROLE: &str = "chop.bundle-metadata.qwen38.precision.v1";
pub const COMPONENT_FORMAT: &str = kernel_core::precision::component::COMPONENT_FORMAT;

fn error(id: &str, reason: impl std::fmt::Display) -> String {
    format!("operation {id}: {reason}")
}

/// The MTP layer's operation IDs: its dense ports, then the operations of
/// the expert banks its drafter records, named at their source layer.
pub fn drafter_operations(shape: Shape) -> Result<Vec<String>, String> {
    if shape.mtp_layers() == 0 {
        return Ok(vec![]);
    }
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    let layer = shape.mtp_layer();
    let mut ids: Vec<String> = ports
        .layer(layer)
        .iter()
        .map(|(_, id)| id.clone())
        .collect();
    for drafter in &ports.declaration.drafters {
        for id in &drafter.operations {
            let projection = ports.projection(id).map_err(|e| e.to_string())?;
            if projection.family() == KernelFamily::GroupedExpert {
                ids.push(
                    ports
                        .operation(id, Some(u64::from(layer)))
                        .map_err(|e| e.to_string())?,
                );
            }
        }
    }
    Ok(ids)
}

/// Whether `document` records the MTP layer's operations. It records all of
/// them or none: a component of the registered compositions alone serves
/// the drafter without records, and a mixed component without them serves
/// no drafter.
pub fn records_drafter(document: &PrecisionDocument, shape: Shape) -> Result<bool, String> {
    let ids = drafter_operations(shape)?;
    let recorded = ids
        .iter()
        .filter(|id| document.operations.iter().any(|s| &s.source.id == *id))
        .count();
    if recorded == 0 {
        Ok(false)
    } else if recorded == ids.len() {
        Ok(true)
    } else {
        Err("Qwen precision component records only part of the MTP layer".into())
    }
}

/// Check every model port, then authenticate every retained binding against
/// the supplied inventory. Execution can retain the returned inventory indices.
pub fn bind(
    document: &PrecisionDocument,
    shape: Shape,
    inventory: &CapabilityInventory,
) -> Result<Vec<Vec<usize>>, String> {
    let layers = shape.layers();
    document.validate().map_err(|e| e.to_string())?;
    let operations: BTreeMap<_, _> = document
        .operations
        .iter()
        .map(|s| (s.source.id.as_str(), s))
        .collect();
    let mut compositions = BTreeMap::new();
    let get = |id: &str| {
        operations
            .get(id)
            .copied()
            .ok_or_else(|| error(id, "missing prepared model port"))
    };
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    let support =
        crate::projections::support::document(shape, document).map_err(|e| e.to_string())?;
    let uses = precision::uses().map_err(|e| e.to_string())?;
    let first = *uses.first().ok_or("Qwen declares no projection uses")?;
    let drafter = records_drafter(document, shape)?;
    // The expert banks that publish their own port, as the inventory offers
    // them; each layer executes its own bank of each.
    let banks: Vec<&str> = ports
        .declaration
        .projections
        .iter()
        .filter(|p| p.family() == KernelFamily::GroupedExpert && p.publishes_as.is_none())
        .map(|p| p.id.as_str())
        .collect();
    for index in (0..layers).chain(drafter.then_some(shape.mtp_layer())) {
        for (role, id) in ports.layer(index) {
            let (role, layer) = (*role, Some(u64::from(index)));
            let projection = ports
                .projection(role.projection_id(shape, layer.map(|l| l as u32)))
                .map_err(|e| error(id, e))?;
            let selection = get(id)?;
            let fusion = ports
                .declaration
                .fusion_groups
                .iter()
                .find(|f| f.members.contains(&projection.id));
            let members = fusion.map_or_else(
                || vec![projection.id.as_str()],
                |f| f.members.iter().map(String::as_str).collect(),
            );
            // Fetch every peer before judging the fusion, so a missing peer
            // is refused first.
            let peers = members
                .into_iter()
                .map(|member| {
                    let peer = ports
                        .layer_operation(index, member)
                        .ok_or_else(|| error(id, "fusion member is absent from the layer"))?;
                    Ok((member, get(peer)?))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let legacy = peers.iter().all(|(_, peer)| source_bf16(peer));
            let weight = selection.effective.weights[0].format;
            // The MTP layer serves only the formats its projections compose,
            // each fusion peer its own.
            if index >= layers {
                for (member, peer) in &peers {
                    let peer_id = &peer.source.id;
                    let declared = ports.projection(member).map_err(|e| error(peer_id, e))?;
                    let format = registry::by_weight(&peer.source.weights[0])
                        .map_err(|e| error(peer_id, e))?
                        .id();
                    if !declared
                        .composition
                        .as_ref()
                        .is_some_and(|c| c.numerical_contracts.contains_key(&format))
                    {
                        return Err(error(
                            peer_id,
                            format!(
                                "{} composes no {:?} MTP operation",
                                declared.id, peer.source.weights[0].format
                            ),
                        ));
                    }
                }
            }
            // Registered fusions serve every topology, as before the port.
            if !legacy {
                let format = kernel_core::precision::format::registry::by_weight(
                    &selection.source.weights[0],
                )
                .map_err(|e| error(id, e))?
                .id();
                support
                    .admit_mixed(&projection.id, first.phase, format)
                    .map_err(|e| error(id, e))?;
            }
            let geometry = if legacy {
                ports.geometry(&projection.id, fusion.is_some())
            } else {
                ports.weight_geometry(&projection.id, weight)
            }
            .map_err(|e| error(id, e))?;
            if selection.effective.kind != OperationKind::Dense
                || selection.effective.geometry != Some(geometry)
                || selection.effective.epilogues != projection.epilogues
                || selection.effective.weights.len() != projection.operands.len()
            {
                return Err(error(
                    id,
                    "prepared dense geometry or fusion differs from the declared port",
                ));
            }
            compositions.insert(
                id.clone(),
                if legacy {
                    role.legacy_implementation(shape)
                } else {
                    role.mixed_implementation_for(shape, weight)
                },
            );
        }
        for &port in &banks {
            let id = ports
                .executed_at(index, port)
                .and_then(|p| ports.operation(&p.id, Some(u64::from(index))))
                .map_err(|e| e.to_string())?;
            let s = get(&id)?;
            let usage = first;
            let capability = match s.effective.weights[0].format {
                WeightFormat::Fp8Block128x128(_) => {
                    let projection =
                        precision::legacy_expert_projection(port).ok_or_else(|| {
                            error(
                                &id,
                                format!("{port} has no registered fragment expert adapter"),
                            )
                        })?;
                    precision::legacy_expert_capability(shape, projection, usage)
                        .map_err(|e| e.0.to_string())
                }
                WeightFormat::Nvfp4 => {
                    let format =
                        kernel_core::precision::format::registry::by_weight(&s.source.weights[0])
                            .map_err(|e| error(&id, e))?
                            .id();
                    support
                        .admit_mixed(port, usage.phase, format)
                        .map_err(|e| error(&id, e))?;
                    Selected::generated_capability(shape, port, WeightFormat::Nvfp4, usage)
                }
                format => {
                    return Err(error(
                        &id,
                        format!(
                            "no supplied grouped port for {format:?} at TP{}",
                            shape.ranks()
                        ),
                    ));
                }
            }
            .map_err(|e| error(&id, e))?;
            if s.effective.kind != capability.kind
                || s.effective.geometry != Some(capability.geometry)
                || s.effective.epilogues != capability.epilogues
                || s.effective.weights.len() != usize::from(capability.operands)
            {
                return Err(error(
                    &id,
                    "prepared grouped geometry or fusion differs from the model port",
                ));
            }
            compositions.insert(id, capability.implementation);
        }
    }
    for ngram in [false, true] {
        let id = if ngram {
            views::ngram_view_for(shape)
                .trim_end_matches(".weight")
                .to_owned()
        } else {
            format!("{}.embed_tokens", views::MODEL)
        };
        let s = get(&id)?;
        let usage = first;
        let capability = if ngram && s.effective.weights[0].format == WeightFormat::Nvfp4 {
            let format = kernel_core::precision::format::registry::by_weight(&s.source.weights[0])
                .map_err(|e| error(&id, e))?
                .id();
            support
                .admit_mixed("ngram", usage.phase, format)
                .map_err(|e| error(&id, e))?;
            Selected::generated_capability(shape, "ngram", WeightFormat::Nvfp4, usage)
        } else {
            precision::legacy_lookup_capability(shape, ngram, usage).map_err(|e| e.0.to_string())
        }
        .map_err(|e| error(&id, e))?;
        if s.effective.kind != OperationKind::Lookup
            || !s.effective.activations.is_empty()
            || s.effective.weights.len() != 1
            || s.effective.matrix_geometry().map_err(|e| e.to_string())? != capability.geometry
        {
            return Err(error(
                &id,
                "prepared lookup geometry differs from the model port",
            ));
        }
        compositions.insert(id, capability.implementation);
    }
    if operations.len() != compositions.len() {
        return Err("Qwen precision component contains an unknown operation".into());
    }
    document
        .operations
        .iter()
        .map(|s| {
            if s.source.uses != uses {
                return Err(error(
                    &s.source.id,
                    "phase/capacity coverage differs from the model",
                ));
            }
            let composition = &compositions[&s.source.id];
            s.bindings
                .iter()
                .map(|b| {
                    let bound = inventory
                        .bind_matching(&s.effective, b.usage, |c| &c.implementation == composition)
                        .map_err(|e| error(&s.source.id, e))?;
                    if bound.identity != b.implementation {
                        return Err(error(
                            &s.source.id,
                            "prepared binding differs from the supplied native implementation",
                        ));
                    }
                    Ok(bound.handle)
                })
                .collect()
        })
        .collect()
}

fn source_bf16(s: &PrecisionSelection) -> bool {
    s.source
        .weights
        .iter()
        .all(|w| w.format == WeightFormat::Bf16)
        && s.source.activations.iter().all(|a| a.is_bf16())
}

#[cfg(test)]
pub fn code_layout(
    format: WeightFormat,
    layout: &str,
) -> Result<(&'static str, &'static str), String> {
    code_layout_for(Shape::flash(crate::contract::Ranks::TP4), format, layout)
}

pub fn code_layout_for(
    shape: Shape,
    format: WeightFormat,
    layout: &str,
) -> Result<(&'static str, &'static str), String> {
    let choice = crate::projections::layouts::choice(format, layout)?;
    Ok((
        crate::projections::layouts::code_dtype(format),
        if format == WeightFormat::Fp8PerTensor {
            views::ngram_layout(shape)
        } else {
            choice.codes
        },
    ))
}

pub fn local_layout(
    format: WeightFormat,
    layout: &str,
    dtype: DType,
) -> Result<(&'static str, &'static str), String> {
    let choice = crate::projections::layouts::choice(format, layout)?;
    let name = if dtype == DType::Bf16 {
        views::BF16_ROWS
    } else {
        choice
            .local_scales
            .ok_or("projection has no local scale layout")?
    };
    Ok((dtype.checkpoint_dtype().unwrap_or("U8"), name))
}

/// Call after binding the document. Fixed router, state and side-operation
/// views keep the legacy requirements; only explicit operation ports vary.
pub fn required(
    document: &PrecisionDocument,
    rank: u32,
    shape: Shape,
) -> Result<Vec<views::Required>, String> {
    let drafter = drafter_operations(shape)?;
    with_operations(
        views::required(rank, shape),
        document
            .operations
            .iter()
            .filter(|s| !drafter.contains(&s.source.id)),
        rank,
        shape,
    )
}

/// The MTP drafter's views when `document` records its layer: the fixed
/// drafter views, with each recorded operation's prepared encoding.
pub fn drafter_required(
    document: &PrecisionDocument,
    rank: u32,
    shape: Shape,
) -> Result<Vec<views::Required>, String> {
    let drafter = drafter_operations(shape)?;
    with_operations(
        views::drafter_required(shape),
        document
            .operations
            .iter()
            .filter(|s| drafter.contains(&s.source.id)),
        rank,
        shape,
    )
}

/// Checks the MTP drafter's views as [`validate_views`] checks the
/// target's: its legacy attention reads Q/K/V as contiguous weights, and
/// its declared F32 hyper-connection norms are admitted.
pub fn validate_drafter_views(
    index: &views::RankIndex,
    shape: Shape,
    document: &PrecisionDocument,
) -> Result<(), String> {
    let mut required = drafter_required(document, index.rank, shape)?;
    crate::norms::admit(index, &mut required);
    views::check(index, [shape.mtp_layer()], required, shape)
}

/// `fixed` views with each of `operations`' views in its prepared encoding.
fn with_operations<'a>(
    fixed: Vec<views::Required>,
    operations: impl IntoIterator<Item = &'a PrecisionSelection>,
    rank: u32,
    shape: Shape,
) -> Result<Vec<views::Required>, String> {
    if !shape.ranks.contains(rank) {
        return Err("rank outside the prepared topology".into());
    }
    let mut required: BTreeMap<_, _> = fixed.into_iter().map(|v| (v.name.clone(), v)).collect();
    for s in operations {
        let op = &s.effective;
        let id = &op.id;
        let weight = &op.weights[0];
        let mut geometry = op.matrix_geometry().map_err(|e| e.to_string())?;
        if id == views::ngram_view_for(shape).trim_end_matches(".weight") {
            geometry.rows = views::ngram_rows(rank, shape)?.1;
        }
        let local_scale_dtype = weight.local_scale.as_ref().map(|scale| {
            if matches!(weight.format, WeightFormat::Fp8Block128x128(_)) {
                DType::F32
            } else {
                scale.tensor.dtype
            }
        });
        let plan = PackingPlan {
            layout: weight.layout.clone(),
            format: weight.format,
            experts: if let OperationKind::Grouped { experts, .. } = op.kind {
                experts
            } else {
                1
            },
            projections: op.weights.len() as u32,
            rows: u32::try_from(geometry.rows).map_err(|_| error(id, "row overflow"))?,
            k: u32::try_from(geometry.k).map_err(|_| error(id, "K overflow"))?,
            columns: SwigluColumns::ConcatenatedHalves,
            local_scale_dtype,
        };
        plan.validate().map_err(|e| error(id, e))?;
        let sections = plan.section_bytes().map_err(|e| error(id, e))?;
        required.remove(&format!("{id}.weight_scale_inv"));
        required.remove(&format!("{id}.weight_scale"));
        let mut add = |suffix: &str, bytes, (dtype, layout_id)| {
            let name = format!("{id}.{suffix}");
            required.insert(
                name.clone(),
                views::Required {
                    name,
                    bytes,
                    dtype,
                    layout_id,
                },
            );
        };
        add(
            "weight",
            sections.codes,
            code_layout_for(shape, weight.format, &weight.layout.id)?,
        );
        if let Some(dtype) = local_scale_dtype {
            add(
                if matches!(weight.format, WeightFormat::Fp8Block128x128(_)) {
                    "weight_scale_inv"
                } else {
                    "weight_scale"
                },
                sections.local_scales,
                local_layout(weight.format, &weight.layout.id, dtype)?,
            );
        }
        if sections.global_scales != 0 {
            add(
                "weight_scale_2",
                sections.global_scales,
                ("F32", views::F32_ROWS),
            );
        }
        if s.source.activations.iter().any(|a| {
            a.quantizer
                .as_ref()
                .is_some_and(|q| matches!(q.calibration, Calibration::Static(_)))
        }) {
            let count = u64::from(plan.experts) * u64::from(plan.projections);
            add("input_scale", count * 4, ("F32", views::F32_ROWS));
        }
    }
    Ok(required.into_values().collect())
}

pub fn validate_views(
    index: &views::RankIndex,
    shape: Shape,
    document: &PrecisionDocument,
) -> Result<(), String> {
    let layers = shape.layers();
    // Only the legacy attention fusion reads Q/K/V as contiguous weights.
    let legacy: BTreeSet<_> = document
        .operations
        .iter()
        .filter(|s| {
            s.bindings[0]
                .implementation
                .implementation
                .starts_with("qwen38.legacy-dense.AttentionQ.")
        })
        .map(|s| s.source.id.as_str())
        .collect();
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    let mut full = Vec::new();
    for layer in (0..layers).filter(|&layer| shape.is_full(layer)) {
        if legacy.contains(
            ports
                .operation("full.q", Some(u64::from(layer)))
                .map_err(|e| e.to_string())?
                .as_str(),
        ) {
            full.push(layer);
        }
    }
    let layers = full.into_iter();
    let mut required = required(document, index.rank, shape)?;
    crate::norms::admit(index, &mut required);
    views::check(index, layers, required, shape)
}

/// Whether an admitted view alternative's source activations carry a
/// statically supplied scale, which the rank image stores as the
/// operation's `input_scale` view.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActivationScale {
    /// BF16 activations, or quantized ones with dynamic scales.
    Absent,
    Static,
}

/// What an admitted view alternative serves: one view of an operation port,
/// keyed by the port it publishes as, or the F32 form of a fixed norm view.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum ViewKey {
    Port {
        port: String,
        format: FormatId,
        activation_scale: ActivationScale,
        suffix: String,
    },
    F32Norm {
        name: String,
    },
}

/// One view a prepared rank image may store in place of the fixed view of
/// the same name.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize)]
pub struct AdmittedView {
    #[serde(flatten)]
    pub key: ViewKey,
    pub dtype: &'static str,
    pub layout_id: &'static str,
    pub bytes: u64,
}

/// Every view alternative a topology admits, beyond the fixed views.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Admitted {
    /// Each rank's alternatives, sorted.
    pub ranks: Vec<Vec<AdmittedView>>,
    /// The packed weight layouts of the admitted capabilities: id and
    /// version.
    pub layouts: BTreeSet<(String, u32)>,
    /// The native capabilities binding can match to an operation port of
    /// the topology, from which the rows and layouts derive. The identity
    /// hashes the rows and layouts, not these.
    pub capabilities: Vec<OperationCapability>,
}

/// The view alternatives `rank` of `shape`'s topology admits
/// ([`admitted`]).
pub fn admitted_views(rank: u32, shape: Shape) -> Result<&'static [AdmittedView], String> {
    admitted(shape)?
        .ranks
        .get(rank as usize)
        .map(Vec::as_slice)
        .ok_or_else(|| "rank outside the prepared topology".into())
}

/// Every view a prepared component can make create require at `shape`'s
/// topology, other than the fixed views, and the packed layouts it can
/// bind. For each operation port (dense ports, expert banks and lookups,
/// the drafter's included) and each capability of the native inventory
/// whose kind, geometry, operands and epilogues binding can match to that
/// port, generated or registered: the views [`with_operations`] derives for
/// the port's first-layer operation, for every local scale dtype the
/// capability accepts, without a static activation scale and, where binding
/// accepts one, with it. A static scale binds where the capability takes a
/// statically calibrated activation, or where its weight format has a BF16
/// weight reuse rule, under which the source may keep any quantized
/// activation. Also the F32 form of each fixed norm view
/// [`crate::norms::admit`] may accept. Computed once per topology.
pub fn admitted(shape: Shape) -> Result<&'static Admitted, String> {
    type Cache = BTreeMap<(usize, u32), &'static Admitted>;
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Cache>> = std::sync::OnceLock::new();
    let key = (std::ptr::from_ref(shape.geometry) as usize, shape.ranks());
    let cache = CACHE.get_or_init(Default::default);
    let lock = || cache.lock().map_err(|_| "admitted view cache".to_owned());
    if let Some(admitted) = lock()?.get(&key) {
        return Ok(admitted);
    }
    let admitted = Box::leak(Box::new(admit_views(shape)?));
    Ok(*lock()?.entry(key).or_insert(admitted))
}

fn admit_views(shape: Shape) -> Result<Admitted, String> {
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    let inventory = precision::capabilities_for(shape.geometry).map_err(|e| e.0.to_string())?;
    // Capabilities that differ only in use, arithmetic or implementation
    // admit the same views and layout.
    let mut distinct: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (index, c) in inventory.operations.iter().enumerate() {
        let content = format!(
            "{:?}",
            (
                c.kind,
                c.target,
                c.geometry,
                c.operands,
                c.weight,
                c.activation,
                c.epilogues,
                &c.layout,
                &c.local_scale_dtypes,
                c.globals,
            )
        );
        distinct.entry(content).or_default().push(index);
    }
    let mut selections = BTreeMap::new();
    let mut admitted = BTreeSet::new();
    for (projection, port) in operation_ports(ports)?.into_values() {
        let id = &first_operation(ports, projection)?;
        for indices in distinct.values() {
            let capability = &inventory.operations[indices[0]];
            if !binds(ports, projection, capability)? {
                continue;
            }
            admitted.extend(indices.iter().copied());
            let format = registry::by_format(capability.weight)
                .map_err(|e| error(id, e))?
                .id();
            let scales: Vec<_> = if capability.local_scale_dtypes.is_empty() {
                vec![None]
            } else {
                capability
                    .local_scale_dtypes
                    .iter()
                    .copied()
                    .map(Some)
                    .collect()
            };
            for scale in scales {
                let operation = capability_operation(id, capability, scale)?;
                for (activation_scale, source) in activation_scales(capability, &operation) {
                    // Uses and arithmetic do not change the views.
                    let content = format!(
                        "{port} {:?} {:?} {:?} {} {:?} {:?} {:?}",
                        operation.kind,
                        operation.geometry,
                        operation.weights,
                        id,
                        activation_scale,
                        format,
                        source.activations
                    );
                    selections.entry(content).or_insert_with(|| {
                        (
                            port.clone(),
                            format,
                            activation_scale,
                            selection(&source, &operation),
                        )
                    });
                }
            }
        }
    }
    let capabilities: Vec<_> = admitted
        .into_iter()
        .map(|index| inventory.operations[index].clone())
        .collect();
    let layouts = capabilities
        .iter()
        .map(|c| (c.layout.id.clone(), c.layout.version))
        .collect();
    let ranks = (0..shape.ranks())
        .map(|rank| {
            let mut rows = Vec::new();
            for (port, format, activation_scale, s) in selections.values() {
                let id = &s.effective.id;
                for view in with_operations(vec![], [s], rank, shape)? {
                    let suffix = view.name.strip_prefix(&format!("{id}.")).ok_or_else(|| {
                        error(id, format!("view {} names another operation", view.name))
                    })?;
                    rows.push(AdmittedView {
                        key: ViewKey::Port {
                            port: port.clone(),
                            format: *format,
                            activation_scale: *activation_scale,
                            suffix: suffix.into(),
                        },
                        dtype: view.dtype,
                        layout_id: view.layout_id,
                        bytes: view.bytes,
                    });
                }
            }
            for view in views::required(rank, shape)
                .iter()
                .chain(&views::drafter_required(shape))
                .filter_map(crate::norms::f32_form)
            {
                rows.push(AdmittedView {
                    key: ViewKey::F32Norm { name: view.name },
                    dtype: view.dtype,
                    layout_id: view.layout_id,
                    bytes: view.bytes,
                });
            }
            rows.sort();
            rows.dedup();
            Ok(rows)
        })
        .collect::<Result<_, String>>()?;
    Ok(Admitted {
        ranks,
        layouts,
        capabilities,
    })
}

/// Every projection that records operations in a prepared component, with
/// the port it publishes as: each executed layer's dense ports, the expert
/// banks and the lookups.
fn operation_ports(
    ports: &crate::projections::ports::Ports,
) -> Result<BTreeMap<&str, (&projection_spec::Projection, String)>, String> {
    let shape = ports.shape;
    let mut result = BTreeMap::new();
    for index in 0..shape.layers() + shape.mtp_layers() {
        for (role, _) in ports.layer(index) {
            let projection = ports
                .projection(role.projection_id(shape, Some(index)))
                .map_err(|e| e.to_string())?;
            result.insert(
                projection.id.as_str(),
                (projection, role.projection_id(shape, None).to_owned()),
            );
        }
    }
    for projection in &ports.declaration.projections {
        if matches!(
            projection.family(),
            KernelFamily::GroupedExpert | KernelFamily::Lookup
        ) {
            result.insert(
                projection.id.as_str(),
                (projection, ports.published(&projection.id).to_owned()),
            );
        }
    }
    Ok(result)
}

/// Whether binding can match `capability` to an operation of `projection`:
/// kind, operand count and epilogues, and the geometry the binder checks:
/// a dense port's registered (logical when fused) or padded mixed geometry,
/// a bank's local expert count, a lookup's whole table.
fn binds(
    ports: &crate::projections::ports::Ports,
    projection: &projection_spec::Projection,
    capability: &OperationCapability,
) -> Result<bool, String> {
    if capability.epilogues != projection.epilogues {
        return Ok(false);
    }
    let error = |e: projection_spec::Error| error(&projection.id, e);
    let operands = u32::from(capability.operands);
    Ok(match (projection.family(), capability.kind) {
        (KernelFamily::Dense, OperationKind::Dense) => {
            let fused = ports
                .declaration
                .fusion_groups
                .iter()
                .any(|f| f.members.contains(&projection.id));
            operands == projection.operand_count()
                && (capability.geometry == ports.geometry(&projection.id, fused).map_err(error)?
                    || capability.geometry
                        == ports
                            .weight_geometry(&projection.id, capability.weight)
                            .map_err(error)?)
        }
        (KernelFamily::GroupedExpert, OperationKind::Grouped { experts, .. }) => {
            operands == projection.operand_count() && experts == ports.shape.local_experts
        }
        (KernelFamily::Lookup, OperationKind::Lookup) => {
            let dim = |i: usize| -> Result<u64, String> {
                projection
                    .shape
                    .get(i)
                    .ok_or_else(|| format!("lookup {} has no table shape", projection.id))?
                    .resolve(&ports.geometry)
                    .map_err(error)
            };
            operands == 1
                && capability.geometry
                    == kernel_core::precision::MatrixGeometry {
                        rows: dim(0)?,
                        k: dim(1)?,
                    }
        }
        _ => false,
    })
}

/// The effective operation `id` binds to `capability`, with local scales of
/// `scale` (the format's canonical dtype when `None`).
fn capability_operation(
    id: &str,
    capability: &OperationCapability,
    scale: Option<DType>,
) -> Result<OperationSpec, String> {
    let mut operation = Selected::capability_domain(capability, scale).map_err(|e| error(id, e))?;
    operation.id = id.into();
    for weight in &mut operation.weights {
        weight.layout = capability.layout.clone();
    }
    Ok(operation)
}

/// The source operations binding accepts for `operation` on `capability`,
/// by whether their activations carry a static scale.
fn activation_scales(
    capability: &OperationCapability,
    operation: &OperationSpec,
) -> Vec<(ActivationScale, OperationSpec)> {
    use kernel_core::precision::capability::ActivationCapability;
    let mut calibrated = operation.clone();
    calibrated
        .activations
        .fill(static_activation(&operation.id));
    match capability.activation {
        None => vec![(ActivationScale::Absent, operation.clone())],
        Some(ActivationCapability::Nvfp4StaticScalar) => {
            vec![(ActivationScale::Static, calibrated)]
        }
        // Weight reuse executes BF16 activations for a source that keeps
        // any quantized activation, a statically calibrated one included.
        Some(ActivationCapability::Bf16)
            if kernel_core::precision::execution::WeightReuse::for_format(capability.weight)
                .is_some() =>
        {
            vec![
                (ActivationScale::Absent, operation.clone()),
                (ActivationScale::Static, calibrated),
            ]
        }
        Some(_) => vec![(ActivationScale::Absent, operation.clone())],
    }
}

/// The operation ID of `projection`'s instance at its first executed layer,
/// as [`crate::projections::ports::Ports::bindings`] assigns it to that
/// instance's sources: named by its first operand, or for a sharded lookup
/// the table's ID, which its shards share.
fn first_operation(
    ports: &crate::projections::ports::Ports,
    projection: &projection_spec::Projection,
) -> Result<String, String> {
    if projection.lookup.is_some() {
        return ports
            .bindings()
            .map_err(|e| e.to_string())?
            .values()
            .find(|b| b.projection == projection.id && b.shard_rows.is_some())
            .map(|b| b.operation.clone())
            .ok_or_else(|| format!("lookup {} has no shard bindings", projection.id));
    }
    let layer = projection
        .stem()
        .map(|stem| ports.first_executed(stem))
        .transpose()
        .map_err(|e| e.to_string())?
        .flatten();
    ports
        .operation(&projection.id, layer)
        .map_err(|e| e.to_string())
}

fn selection(source: &OperationSpec, effective: &OperationSpec) -> PrecisionSelection {
    PrecisionSelection {
        source: source.clone(),
        effective: effective.clone(),
        policy: ActivationPolicy::Bf16,
        reuse: None,
        fallback_reasons: vec![],
        bindings: vec![],
    }
}

/// The one activation format with a static calibration
/// (`ActivationSpec::validate`).
const STATIC_ACTIVATION: ActivationFormat = ActivationFormat::Nvfp4;

/// A statically calibrated activation of operation `id`, as the importer
/// records a supplied scalar `input_scale`.
fn static_activation(id: &str) -> ActivationSpec {
    ActivationSpec {
        format: STATIC_ACTIVATION,
        storage: DType::Bf16,
        compute: DType::Fp4E2M1,
        quantizer: Some(QuantizerSpec {
            calibration: Calibration::Static(ScaleSpec {
                tensor: TensorSpan {
                    tensor: format!("{id}.input_scale"),
                    dtype: DType::F32,
                    shape: vec![],
                    offset: 0,
                    byte_length: 4,
                },
                scope: ScaleScope::Tensor,
                convention: ScaleConvention::Multiply,
                groups: vec![],
            }),
            groups: vec![GroupAxis {
                axis: 1,
                size: 16,
                origin: 0,
            }],
            rounding: Rounding::NearestEven,
            clipping: Clipping::FiniteSaturate,
        }),
        transforms: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::DenseRole;
    use kernel_core::precision::execution::{ActivationPolicy, BoundUse};

    /// One MXFP8 `linear.qkv` port of layer 0, bound to its published TP4
    /// composition, and nothing else.
    fn mxfp8_qkv(inventory: &CapabilityInventory) -> PrecisionDocument {
        let usage = precision::uses().unwrap()[0];
        let published =
            crate::sm120::generated::publish(Shape::TP4, "linear.qkv", WeightFormat::Mxfp8, usage)
                .unwrap();
        let mut operation =
            crate::sm120::generated::domain(Shape::TP4, "linear.qkv", WeightFormat::Mxfp8, usage)
                .unwrap();
        operation.id = DenseRole::LinearQkv
            .operation_id_for(Shape::TP4, 0)
            .unwrap();
        for weight in &mut operation.weights {
            weight.layout = published.capability.layout.clone();
        }
        operation.arithmetic = published.capability.arithmetic.clone();
        let binding = inventory
            .bind_matching(&operation, usage, |c| {
                c.implementation == published.capability.implementation
            })
            .unwrap();
        let document = PrecisionDocument {
            version: 1,
            operations: vec![PrecisionSelection {
                source: operation.clone(),
                effective: operation,
                policy: ActivationPolicy::Bf16,
                reuse: None,
                fallback_reasons: vec![],
                bindings: vec![BoundUse {
                    usage,
                    implementation: binding.identity,
                }],
            }],
        };
        document.validate().unwrap();
        document
    }

    /// The pre-port binder fetched every input fusion peer before it judged
    /// the topology, so a missing peer is the first refusal even where the
    /// mixed fusion has no composition.
    /// The drafter's expert operations are its declared banks, named as the
    /// checkpoint stores the MTP layer's experts, after its dense ports.
    #[test]
    fn drafter_expert_operations_are_the_declared_banks() {
        for geometry in crate::geometry::GEOMETRIES {
            for ranks in [crate::contract::Ranks::TP2, crate::contract::Ranks::TP4] {
                let shape = Shape::new(geometry, ranks);
                let ids = drafter_operations(shape).unwrap();
                if shape.mtp_layers() == 0 {
                    assert!(ids.is_empty());
                    continue;
                }
                let ports = crate::projections::ports::get(shape).unwrap();
                let layer = shape.mtp_layer();
                let dense = ports.layer(layer).len();
                assert_eq!(
                    ids[..dense],
                    ports
                        .layer(layer)
                        .iter()
                        .map(|(_, id)| id.clone())
                        .collect::<Vec<_>>()
                );
                let stem = views::layer_stem_for(layer, shape);
                assert_eq!(
                    ids[dense..],
                    ["gate_up_proj", "down_proj"].map(|tail| format!("{stem}.mlp.experts.{tail}"))
                );
            }
        }
    }

    /// Each expert bank that publishes its own port has a registered
    /// fragment adapter with the port's epilogues and operand count.
    #[test]
    fn expert_ports_have_their_registered_fragment_adapter() {
        let usage = precision::uses().unwrap()[0];
        for shape in [Shape::TP2, Shape::TP4] {
            let ports = crate::projections::ports::get(shape).unwrap();
            let banks: Vec<_> = ports
                .declaration
                .projections
                .iter()
                .filter(|p| p.family() == KernelFamily::GroupedExpert && p.publishes_as.is_none())
                .collect();
            assert_eq!(banks.len(), 2);
            for bank in banks {
                let projection = precision::legacy_expert_projection(&bank.id).unwrap();
                let capability =
                    precision::legacy_expert_capability(shape, projection, usage).unwrap();
                assert_eq!(capability.epilogues, bank.epilogues, "{}", bank.id);
                assert_eq!(
                    u32::from(capability.operands),
                    bank.operand_count(),
                    "{}",
                    bank.id
                );
            }
        }
        assert_eq!(
            precision::legacy_expert_projection("mtp.experts.down"),
            None
        );
    }

    /// The n-gram alternative carries the table's canonical operation ID,
    /// which its shards share and which `with_operations` sizes per rank.
    #[test]
    fn the_ngram_alternative_is_the_canonical_table_operation() {
        for geometry in crate::geometry::GEOMETRIES {
            for ranks in [crate::contract::Ranks::TP2, crate::contract::Ranks::TP4] {
                let shape = Shape::new(geometry, ranks);
                let ports = crate::projections::ports::get(shape).unwrap();
                assert_eq!(
                    first_operation(ports, ports.projection("ngram").unwrap()).unwrap(),
                    views::ngram_view_for(shape).trim_end_matches(".weight")
                );
                for rank in 0..shape.ranks() {
                    let (_, rows) = views::ngram_rows(rank, shape).unwrap();
                    let width = u64::from(shape.ngram_head_width());
                    // The generated NVFP4 table packs two codes per byte;
                    // the registered FP8 table stores one.
                    for (format, bytes) in [
                        (FormatId::Nvfp4, rows * width / 2),
                        (FormatId::Fp8PerTensor, rows * width),
                    ] {
                        let weight = admitted_views(rank, shape)
                            .unwrap()
                            .iter()
                            .find(|v| {
                                matches!(&v.key, ViewKey::Port { port, format: f, suffix, .. }
                                    if port == "ngram" && *f == format && suffix == "weight")
                            })
                            .unwrap();
                        assert_eq!(
                            weight.bytes,
                            bytes,
                            "{} TP{} rank {rank} {format:?}",
                            geometry.id,
                            shape.ranks()
                        );
                    }
                }
            }
        }
    }

    /// Each projection's operation ID as [`crate::projections::ports::Ports::bindings`]
    /// assigns it to the sources at its first layer.
    fn first_operations_by_source(
        ports: &crate::projections::ports::Ports,
    ) -> Result<BTreeMap<String, String>, String> {
        let bindings = ports.bindings().map_err(|e| e.to_string())?;
        let mut first = BTreeMap::new();
        for (name, position) in ports
            .declaration
            .source_positions(&ports.geometry)
            .map_err(|e| e.to_string())?
        {
            let projection_spec::bind::SourceOwner::Projection(index) = position.owner else {
                continue;
            };
            let projection = &ports.declaration.projections[index].id;
            let operation = &bindings
                .get(&name)
                .ok_or_else(|| format!("source {name} has no binding"))?
                .operation;
            let layer = position.layer.unwrap_or(0);
            match first.get(projection) {
                Some((earliest, _)) if *earliest <= layer => {}
                _ => {
                    first.insert(projection.clone(), (layer, operation.clone()));
                }
            }
        }
        Ok(first
            .into_iter()
            .map(|(projection, (_, operation))| (projection, operation))
            .collect())
    }

    /// Each operation port's ID is the one the source bindings assign to
    /// the projection's first-layer sources.
    #[test]
    fn operation_ids_are_the_first_layer_source_bindings() {
        for geometry in crate::geometry::GEOMETRIES {
            for ranks in [crate::contract::Ranks::TP2, crate::contract::Ranks::TP4] {
                let ports = crate::projections::ports::get(Shape::new(geometry, ranks)).unwrap();
                let expected = first_operations_by_source(ports).unwrap();
                for (projection, _) in operation_ports(ports).unwrap().into_values() {
                    assert_eq!(
                        first_operation(ports, projection).unwrap(),
                        expected[&projection.id],
                        "{}",
                        projection.id
                    );
                }
            }
        }
    }

    /// Flash TP4 admits NVFP4 experts with and without a static activation
    /// scale, MXFP8 dense ports and F32 hyper-connection norms.
    #[test]
    fn flash_tp4_admits_nvfp4_expert_and_mxfp8_dense_views() {
        let shape = Shape::TP4;
        let admitted = admitted(shape).unwrap();
        let find = |port: &str, format, activation_scale, suffix: &str| {
            let key = ViewKey::Port {
                port: port.into(),
                format,
                activation_scale,
                suffix: suffix.into(),
            };
            admitted.ranks[0]
                .iter()
                .find(|v| v.key == key)
                .map(|v| (v.dtype, v.layout_id, v.bytes))
        };
        let experts = u64::from(shape.local_experts);
        let gate_up =
            experts * 2 * u64::from(shape.expert_intermediate()) * u64::from(shape.hidden());
        let nvfp4 = |activations, suffix| {
            find("main.experts.gate-up", FormatId::Nvfp4, activations, suffix)
        };
        for activations in [ActivationScale::Absent, ActivationScale::Static] {
            assert_eq!(
                nvfp4(activations, "weight"),
                Some(("U8", "nvfp4_e2m1_rows_v1", gate_up / 2))
            );
            assert_eq!(
                nvfp4(activations, "weight_scale"),
                Some(("F8_E4M3", "nvfp4_e4m3_scale_tiles_16x8_v1", gate_up / 16))
            );
            assert_eq!(
                nvfp4(activations, "weight_scale_2"),
                Some(("F32", views::F32_ROWS, experts * 2 * 4))
            );
        }
        assert_eq!(nvfp4(ActivationScale::Absent, "input_scale"), None);
        assert_eq!(
            nvfp4(ActivationScale::Static, "input_scale"),
            Some(("F32", views::F32_ROWS, experts * 2 * 4))
        );
        let geometry = crate::projections::ports::get(shape)
            .unwrap()
            .weight_geometry("linear.qkv", WeightFormat::Mxfp8)
            .unwrap();
        let mxfp8 = |suffix| {
            find(
                "linear.qkv",
                FormatId::Mxfp8,
                ActivationScale::Absent,
                suffix,
            )
        };
        assert_eq!(
            mxfp8("weight"),
            Some(("F8_E4M3", "mxfp8_e4m3_rows_v1", geometry.rows * geometry.k))
        );
        assert_eq!(
            mxfp8("weight_scale"),
            Some((
                "F8_E8M0",
                "mxfp8_ue8m0_scale_rows_v1",
                geometry.rows * geometry.k / 32
            ))
        );
        let norm = format!(
            "{}.layers.0.attn_hyper_connection.hc_norm.weight",
            views::MODEL
        );
        assert!(admitted.ranks[0].contains(&AdmittedView {
            key: ViewKey::F32Norm { name: norm },
            dtype: "F32",
            layout_id: views::F32_ROWS,
            bytes: u64::from(shape.wide()) * 4,
        }));
        for layout in [
            "moe-packed-ExpertProjection",
            "row-major",
            "fp8-fragment",
            "bf16-fragment",
        ] {
            assert!(admitted.layouts.contains(&(layout.into(), 1)), "{layout}");
        }
    }

    /// The registered FP8 block experts are admitted in their fragment
    /// layout, and under BF16 weight reuse also with a statically scaled
    /// source activation, whose `input_scale` holds one F32 per local expert
    /// and projection: 2048 and 1024 bytes at TP2, 1024 and 512 at TP4.
    #[test]
    fn registered_fp8_experts_admit_a_static_source_scale_under_reuse() {
        for (shape, gate_up, down) in [(Shape::TP2, 2048, 1024), (Shape::TP4, 1024, 512)] {
            let admitted = admitted(shape).unwrap();
            let find = |port: &str, activation_scale, suffix: &str| {
                let key = ViewKey::Port {
                    port: port.into(),
                    format: FormatId::Fp8Block128,
                    activation_scale,
                    suffix: suffix.into(),
                };
                admitted.ranks[0]
                    .iter()
                    .find(|v| v.key == key)
                    .map(|v| (v.dtype, v.layout_id, v.bytes))
            };
            for (port, bytes) in [
                ("main.experts.gate-up", gate_up),
                ("main.experts.down", down),
            ] {
                for activation_scale in [ActivationScale::Absent, ActivationScale::Static] {
                    let (dtype, layout, _) = find(port, activation_scale, "weight").unwrap();
                    assert_eq!((dtype, layout), ("F8_E4M3", views::FP8_FRAGMENT), "{port}");
                }
                assert_eq!(find(port, ActivationScale::Absent, "input_scale"), None);
                assert_eq!(
                    find(port, ActivationScale::Static, "input_scale"),
                    Some(("F32", views::F32_ROWS, bytes)),
                    "{port} TP{}",
                    shape.ranks()
                );
            }
        }
    }

    /// Every capability of the native inventory is admitted at a topology
    /// of its geometry, and the layout it binds is among that topology's
    /// packed layouts.
    #[test]
    fn every_native_capability_is_admitted_with_its_layout() {
        for geometry in crate::geometry::GEOMETRIES {
            let shapes = [crate::contract::Ranks::TP2, crate::contract::Ranks::TP4]
                .map(|ranks| Shape::new(geometry, ranks));
            let inventory = precision::capabilities_for(geometry).unwrap();
            for capability in &inventory.operations {
                assert!(
                    shapes.iter().any(|&shape| {
                        let admitted = admitted(shape).unwrap();
                        admitted.capabilities.contains(capability)
                            && admitted.layouts.contains(&(
                                capability.layout.id.clone(),
                                capability.layout.version,
                            ))
                    }),
                    "{} {:?} {:?} {:?}",
                    capability.implementation,
                    capability.kind,
                    capability.geometry,
                    capability.layout
                );
            }
        }
    }

    #[test]
    fn a_missing_fusion_peer_is_refused_before_the_topology() {
        let capabilities = crate::sm120::generated::document_for(Shape::TP4.geometry).unwrap();
        let inventory =
            CapabilityInventory::new(capabilities.clone(), capabilities.entries(), "0".repeat(64))
                .unwrap();
        let document = mxfp8_qkv(&inventory);
        let missing = DenseRole::LinearZ.operation_id_for(Shape::TP4, 0).unwrap();
        for shape in [Shape::TP2, Shape::TP4] {
            let error = bind(&document, shape, &inventory).unwrap_err();
            assert_eq!(
                error,
                format!("operation {missing}: missing prepared model port"),
                "TP{}",
                shape.ranks()
            );
        }
    }
}
