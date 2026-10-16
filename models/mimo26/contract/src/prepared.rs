//! Model-owned precision binding for the compiled serving ports and policy alternatives.
use kernel_core::contract::Refusal;
use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
use kernel_core::precision::execution::{Binding, ImplementationProvider, PrecisionDocument};
use kernel_core::precision::{ExecutionPhase, OperationSpec, OperationUse, ScaleSpec};
use model_compiler_sdk::quantization::import::companion_names;

use crate::backend::{Alternative, Kernels, Selected};
use crate::{
    Tp,
    capabilities::{self, Projection},
};

pub const COMPONENT_ROLE: &str = "chop.bundle-metadata.mimo.precision.v1";
pub const COMPONENT_FORMAT: &str = kernel_core::precision::component::COMPONENT_FORMAT;
pub fn component_id(tp: Tp) -> String {
    format!("precision.{}", crate::topology::topology_id(tp))
}
pub fn component_path(tp: Tp) -> String {
    format!("components/{}.bin", component_id(tp))
}

pub struct Provider {
    pub tp: Tp,
}

/// The compiled projection of a declared operation: its declared
/// projection's offer with the operation's output type.
fn role(tp: Tp, operation: &OperationSpec) -> Result<Projection, String> {
    let (declaration, _) = declared(tp)?;
    let (projection, _) = declaration
        .operation_instance(&operation.id)
        .map_err(|e| e.to_string())?;
    capabilities::served(tp.geometry, &projection.id)
        .and_then(|offers| {
            offers
                .iter()
                .find(|offer| offer.output() == operation.arithmetic.output)
        })
        .copied()
        .ok_or_else(|| format!("unknown declared serving port {}", operation.id))
}

impl ImplementationProvider for Provider {
    type Handle = <Selected as Kernels>::Entry;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Self::Handle>, String> {
        let projection = role(self.tp, operation)?;
        let mut capability =
            capabilities::capability(projection, self.tp, usage).map_err(|e| e.to_string())?;
        let alternative = if operation.id.ends_with(".inexact") {
            Alternative::Inexact
        } else if operation.id.ends_with(".mid") {
            Alternative::Medium
        } else {
            Alternative::Served
        };
        if alternative == Alternative::Medium
            && !projection.routed()
            && !projection.lookup()
            && (usage.phase == ExecutionPhase::Prefill
                || usage.capacity <= crate::serving::DECODE_ENTRY_ROWS)
        {
            return Err("medium policy is unavailable for this use".into());
        }
        let handle = Selected::bind(self.tp, projection, usage, alternative, &mut capability)?;
        if projection.routed() {
            capability.implementation = format!(
                "{}.{}",
                projection.implementation(self.tp),
                if alternative == Alternative::Inexact {
                    "inexact"
                } else {
                    "exact"
                }
            );
        } else if alternative == Alternative::Medium && !projection.lookup() {
            capability.implementation.push_str(".mid");
        }
        // The dependency includes every retained policy stage, independently
        // of the process environment that selects one at execution.
        let inventory = CapabilityInventory::new(
            CapabilityDocument {
                version: 1,
                operations: vec![capability.clone()],
            },
            capability.stages.iter().cloned().collect(),
            Selected::dependency(self.tp),
        )?;
        let binding = inventory.bind(operation, usage)?;
        Ok(Binding {
            identity: binding.identity,
            handle,
        })
    }
}

/// Retain both implementation policies in the component. Environment switches
/// select among these authenticated alternatives only in the model bundle.
pub fn document(
    graph: &projection_spec::derive::RankDerivation,
    tp: Tp,
    admitted: &std::collections::BTreeMap<
        String,
        model_compiler_sdk::quantization::import::ImportedOperand,
    >,
) -> Result<PrecisionDocument, String> {
    let provider = Provider { tp };
    let resolve = |operation: &OperationSpec| {
        model_compiler_sdk::quantization::resolve_operation(
            operation,
            model_compiler_sdk::quantization::ActivationPolicy::Bf16,
            &provider,
        )
        .map(|prepared| prepared.selection().clone())
        .map_err(|e| e.to_string())
    };
    let (declaration, geometry) = declared(tp)?;
    let mut operations = vec![];
    for mut source in graph.operation_specs().map_err(|e| e.to_string())? {
        model_compiler_sdk::quantization::declared::admitted_activations(
            &mut source,
            admitted,
            declaration,
            geometry,
        )
        .map_err(|e| e.to_string())?;
        let selection = resolve(&source)?;
        let projection = role(tp, &source)?;
        let alternative = if projection.routed() {
            Some(("inexact", source.uses.clone()))
        } else if projection.lookup() {
            None
        } else {
            // Every use with a catalog medium entry must bind it; a failed
            // bind refuses the component rather than dropping the policy.
            Some((
                "mid",
                medium_uses(&provider, &selection.effective, &source.uses)?,
            ))
        };
        operations.push(selection);
        if let Some((suffix, uses)) = alternative.filter(|(_, uses)| !uses.is_empty()) {
            let mut other = source.clone();
            other.id = format!("{}.{suffix}", source.id);
            other.uses = uses;
            operations.push(resolve(&other)?);
        }
    }
    let document = PrecisionDocument {
        version: 1,
        operations,
    };
    document.validate().map_err(|e| e.to_string())?;
    Ok(document)
}

/// The expanded declaration and geometry of a served topology.
fn declared(
    tp: Tp,
) -> Result<&'static (projection_spec::Declaration, projection_spec::Geometry), String> {
    crate::projections::expanded(tp).map_err(|e| e.to_string())
}

/// Every operation of the declared projections in `scope`.
fn declared_operations(
    tp: Tp,
    scope: impl Fn(&str) -> bool,
) -> Result<Vec<(String, Vec<OperationUse>)>, String> {
    let (declaration, geometry) = declared(tp)?;
    declaration
        .operations(geometry, |p| scope(&p.id))
        .map_err(|e| e.to_string())
}

/// Text ports are every declared main-decoder projection; MTP and DFlash
/// projections belong to their drafters.
fn text_scope(id: &str) -> bool {
    !id.starts_with("mtp") && !id.starts_with("dflash.")
}

fn require(
    document: &PrecisionDocument,
    operations: Vec<(String, Vec<OperationUse>)>,
) -> Result<(), String> {
    for (id, uses) in operations {
        let selection = document
            .operations
            .iter()
            .find(|s| s.source.id == id)
            .ok_or_else(|| format!("missing declared model port {id}"))?;
        if selection
            .bindings
            .iter()
            .map(|b| b.usage)
            .collect::<Vec<_>>()
            != uses
        {
            return Err(format!("model port {id} has incomplete use coverage"));
        }
    }
    Ok(())
}

/// Check that a component serves every text port, with its alternatives,
/// over every declared use, and that each operation reads the checkpoint
/// sources its projection declares. Binding identities are authenticated
/// separately against the supplied kernels.
pub fn check_ports(document: &PrecisionDocument, tp: Tp) -> Result<(), String> {
    require(document, with_alternatives(document, tp, text_scope)?)?;
    let (declaration, geometry) = declared(tp)?;
    for selection in &document.operations {
        let (projection, layer) = declaration
            .operation_instance(&selection.source.id)
            .map_err(|e| e.to_string())?;
        let expected = declaration
            .instance_sources(geometry, &projection.id, layer)
            .map_err(|e| e.to_string())?;
        if selection.source.weights.len() != expected.len()
            || selection
                .source
                .weights
                .iter()
                .zip(&expected)
                .any(|(weight, name)| weight.codes.tensor.split('@').next() != Some(name.as_str()))
        {
            return Err(format!(
                "model port {} has different source weight names",
                selection.source.id
            ));
        }
        // A weight's companions are exactly the tensors the importer reads
        // for its codes under its format, each in its own role, with the
        // dtype and extent the weight's logical shape requires.
        for operation in [&selection.source, &selection.effective] {
            for weight in &operation.weights {
                let (codes, view) = weight.codes.tensor.split_once('@').map_or(
                    (weight.codes.tensor.as_str(), String::new()),
                    |(codes, view)| (codes, format!("@{view}")),
                );
                let names = companion_names(weight.format, codes, None)
                    .map_err(|e| format!("model port {}: {e}", selection.source.id))?;
                let companion = |scale: Option<&ScaleSpec>, accepted: &[String]| match scale {
                    None => accepted.is_empty(),
                    Some(scale) => accepted
                        .iter()
                        .any(|name| scale.tensor.tensor == format!("{name}{view}")),
                };
                if !companion(weight.local_scale.as_ref(), &names.local)
                    || !companion(weight.global_scale.as_ref(), names.global.as_slice())
                {
                    return Err(format!(
                        "model port {} reads another tensor's scales",
                        selection.source.id
                    ));
                }
                weight
                    .validate()
                    .map_err(|e| format!("model port {}: {e}", selection.source.id))?;
            }
        }
    }
    Ok(())
}

/// The declared operations in `scope` with the alternatives the bundle
/// launches: inexact experts, and medium-M dense entries where the catalog
/// offers one.
fn with_alternatives(
    document: &PrecisionDocument,
    tp: Tp,
    scope: impl Fn(&str) -> bool,
) -> Result<Vec<(String, Vec<OperationUse>)>, String> {
    let provider = Provider { tp };
    let mut required = vec![];
    for (id, uses) in declared_operations(tp, scope)? {
        let selection = document
            .operations
            .iter()
            .find(|s| s.source.id == id)
            .ok_or_else(|| format!("missing declared model port {id}"))?;
        let projection = role(tp, &selection.effective)?;
        if projection.routed() {
            required.push((format!("{id}.inexact"), uses.clone()));
        } else if !projection.lookup() {
            let medium = medium_uses(&provider, &selection.effective, &uses)?;
            if !medium.is_empty() {
                required.push((format!("{id}.mid"), medium));
            }
        }
        required.push((id, uses));
    }
    Ok(required)
}

/// Check that a component serves every DFlash port the drafter launches,
/// with its medium alternatives.
pub fn check_dflash_ports(document: &PrecisionDocument, tp: Tp) -> Result<(), String> {
    require(
        document,
        with_alternatives(document, tp, |id| {
            id.starts_with("dflash.") && capabilities::served(tp.geometry, id).is_some()
        })?,
    )
}

/// Check that a component serves every declared MTP port.
pub fn check_mtp_ports(document: &PrecisionDocument, tp: Tp) -> Result<(), String> {
    require(
        document,
        declared_operations(tp, |id| id.starts_with("mtp"))?,
    )
}

/// The uses of a dense operation whose prefill-shaped entry has a medium-M
/// alternative in the catalog: decode and verify buckets above the decode
/// entries' rows.
fn medium_uses(
    provider: &Provider,
    operation: &OperationSpec,
    uses: &[OperationUse],
) -> Result<Vec<OperationUse>, String> {
    let mut medium = vec![];
    for usage in uses {
        if usage.phase == ExecutionPhase::Prefill
            || usage.capacity <= crate::serving::DECODE_ENTRY_ROWS
        {
            continue;
        }
        let bound = provider.bind(operation, *usage)?.handle;
        match Selected::has_medium(&bound) {
            None => return Err(format!("operation {} is not dense", operation.id)),
            Some(true) => medium.push(*usage),
            Some(false) => {}
        }
    }
    Ok(medium)
}

pub fn encode(
    graph: &projection_spec::derive::RankDerivation,
    tp: Tp,
    admitted: &std::collections::BTreeMap<
        String,
        model_compiler_sdk::quantization::import::ImportedOperand,
    >,
) -> Result<Vec<u8>, String> {
    kernel_core::precision::component::encode(&document(graph, tp, admitted)?)
        .map_err(|e| e.to_string())
}

pub fn usage(phase: bundle_runtime::phase::Phase) -> Result<OperationUse, Refusal> {
    use bundle_runtime::phase::Phase;
    Ok(OperationUse {
        phase: match phase {
            Phase::Prefill => ExecutionPhase::Prefill,
            Phase::Decode(_) => ExecutionPhase::Decode,
            Phase::Verify(_) => ExecutionPhase::Verify,
        },
        capacity: phase
            .capacity_in::<crate::serving::Mimo>()
            .ok_or(Refusal("invalid projection phase"))?,
        graph: phase != Phase::Prefill,
    })
}
