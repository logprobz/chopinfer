//! Compact source recipes retained by the existing preparation snapshot. Source
//! tensor contents and model placement remain in that snapshot's ordinary input
//! and rank-image records, not another checkpoint cache.
use super::*;
use kernel_core::precision::OperationUse;
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{
    Binding, ImplementationProvider, PrecisionDocument, PrecisionSelection,
};
use model_compiler_sdk::quantization::{ActivationPolicy, resolve_operation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceOperation {
    source: OperationSpec,
    /// A model-owned composition restricts equivalent matrix ports to the
    /// actual buffer/fusion contract. Grouped banks use the shared domain.
    composition: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProgram {
    version: u32,
    ranks: u32,
    #[serde(default = "flash_id", skip_serializing_if = "is_flash")]
    geometry: String,
    operations: Vec<SourceOperation>,
    /// The MTP layer's operations, recorded after the text operations when
    /// the kernels compose its encodings. The registered layout records
    /// none: its drafter keeps its fixed program.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    drafter: Vec<SourceOperation>,
}

struct Provider<'a> {
    inventory: &'a CapabilityInventory,
    composition: Option<&'a str>,
    /// What the bind attempts found (`resolve_operation` binds once per use
    /// for each operation it tries).
    attempts: std::cell::Cell<Attempts>,
}

/// A provider's bind attempts: how many were made, and whether every one
/// failed only because the inventory, which admitted the operation, has no
/// implementation of its composition in any phase.
#[derive(Clone, Copy, Default)]
struct Attempts {
    made: usize,
    all_absent: bool,
}

impl Provider<'_> {
    fn new<'a>(inventory: &'a CapabilityInventory, composition: Option<&'a str>) -> Provider<'a> {
        Provider {
            inventory,
            composition,
            attempts: std::cell::Cell::new(Attempts {
                made: 0,
                all_absent: true,
            }),
        }
    }

    /// Whether the resolution this provider served failed only for an
    /// absent composition: at least one bind was attempted, and each was
    /// an admitted operation with no capability of its composition.
    fn only_absent(&self) -> bool {
        let attempts = self.attempts.get();
        attempts.made > 0 && attempts.all_absent
    }
}

/// Whether `inventory` supplies `operation`'s target, kind, geometry,
/// operand count, weight format and epilogues in any phase.
fn supplied_anywhere(inventory: &CapabilityInventory, operation: &OperationSpec) -> bool {
    let Ok(geometry) = operation.matrix_geometry() else {
        return true;
    };
    inventory.document().operations.iter().any(|c| {
        c.target == operation.target
            && c.kind == operation.kind
            && c.geometry == geometry
            && usize::from(c.operands) == operation.weights.len()
            && c.weight == operation.weights[0].format
            && c.epilogues == operation.epilogues
    })
}

impl ImplementationProvider for Provider<'_> {
    type Handle = usize;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> std::result::Result<Binding<usize>, String> {
        let result = self.inventory.bind_matching(operation, usage, |c| {
            self.composition.is_none_or(|name| c.implementation == name)
        });
        let absent = result.is_err()
            && self.inventory.admits(operation).is_ok()
            && !supplied_anywhere(self.inventory, operation);
        let mut attempts = self.attempts.get();
        attempts.made += 1;
        attempts.all_absent &= absent;
        self.attempts.set(attempts);
        result
    }
}

impl SourceOperation {
    /// The operation resolved under `policy` against `inventory`. When it
    /// does not bind only because each attempt was a valid operation the
    /// inventory admits but supplies no implementation of in any phase (by
    /// target, kind, where a grouped kind carries the rank count's local
    /// experts, geometry, operand count, weight format and epilogues), the
    /// topology has no composition for this precision: an
    /// [`UnsuppliedTopology`] that a package leaves out. An invalid or
    /// inadmissible operation, and a supplied one that misses only some
    /// phase, capacity or composition, fail as they are.
    fn resolve(
        &self,
        policy: ActivationPolicy,
        inventory: &CapabilityInventory,
        ranks: u32,
    ) -> Result<PrecisionSelection> {
        let op = self.current();
        let provider = Provider::new(inventory, op.composition.as_deref());
        match resolve_operation(&op.source, policy, &provider) {
            Ok(prepared) => Ok(prepared.selection().clone()),
            Err(_) if provider.only_absent() => Err(UnsuppliedTopology(format!(
                "operation {}: the supplied kernels have no {:?} {:?} composition at TP{ranks}",
                op.source.id, op.source.kind, op.source.weights[0].format
            ))
            .into()),
            Err(error) => Err(error.into()),
        }
    }

    // Source snapshots from the previous provider stored its BF16 decode order.
    // Normalize only the audited plain weight formats when selecting new code.
    fn current(&self) -> Self {
        let mut op = self.clone();
        if op.source.activations.iter().all(|a| a.is_bf16())
            && matches!(
                op.source.arithmetic.decode_order,
                DecodeOrder::CodeThenScale | DecodeOrder::ScaleThenCode
            )
        {
            op.source.arithmetic.decode_order = match (op.source.weights[0].format, op.source.kind)
            {
                (WeightFormat::Nvfp4, OperationKind::Lookup) => DecodeOrder::ExactWeightToBf16,
                (WeightFormat::Nvfp4, _) => DecodeOrder::ExactLocalThenGlobal,
                (WeightFormat::Mxfp8 | WeightFormat::Mxfp4, _) => DecodeOrder::ExactScaledOperand,
                _ => op.source.arithmetic.decode_order,
            };
        }
        if let Some(name) = &mut op.composition {
            *name = name
                .replace(
                    "gemm.w-nvfp4.a-bf16.acc-f32.v1",
                    "gemm.w-nvfp4.a-bf16.acc-f32.exact-local-global-after-mma.v2",
                )
                .replace(
                    "gemm.w-mxfp8.a-bf16.acc-f32.v1",
                    "gemm.w-mxfp8.a-bf16.acc-f32.exact-scaled-operand.v2",
                )
                .replace(
                    "lookup.nvfp4.code-local-global.bf16-rne.v1",
                    "lookup.nvfp4.exact-product.bf16-rne.v2",
                );
        }
        op
    }
}

fn operation_count(shape: Shape) -> Result<usize> {
    let mut count = 2; // Token and n-gram lookup operations.
    for layer in 0..shape.layers() {
        count += DenseRole::roles_for(shape, layer)?.len() + 2; // Routed gate/up and down operations.
    }
    Ok(count)
}

/// Why a quantized checkpoint's MTP layer records no operations at
/// `shape`: the MTP tensor and format the kernels compose no operation of.
/// The registered layout's drafter keeps its fixed program.
pub fn drafter_refusal(source: &SourcePrecision, shape: Shape) -> Option<String> {
    if source.legacy_text_layout() || shape.mtp_layers() == 0 {
        return None;
    }
    drafter_operations(source, shape).err()
}

/// The MTP layer's operations: its dense ports, all-BF16 fusions bound to
/// the registered compositions, then its experts, bound by the main layers'
/// grouped compositions. An encoding without a composition is the reason
/// it records none, naming the tensor and its format.
fn drafter_operations(
    source: &SourcePrecision,
    shape: Shape,
) -> std::result::Result<Vec<SourceOperation>, String> {
    let layer = shape.mtp_layer();
    let mut operations = Vec::new();
    let mut mapped = BTreeSet::new();
    let ports = precision::ports(shape).map_err(|e| e.0.to_string())?;
    for role in DenseRole::roles_for(shape, layer).map_err(|e| e.0.to_string())? {
        let op = dense(source, layer, role, shape).map_err(|e| e.to_string())?;
        // The MTP layer serves only the formats its projections compose,
        // each fusion peer its own.
        let projection = role.projection_id(shape, Some(layer));
        let members = ports
            .declaration
            .fusion_groups
            .iter()
            .find(|f| f.members.iter().any(|m| m == projection))
            .map_or_else(
                || vec![projection],
                |f| f.members.iter().map(String::as_str).collect(),
            );
        for member in members {
            let declared = ports.projection(member).map_err(|e| e.to_string())?;
            for name in ports
                .names(member, Some(u64::from(layer)))
                .map_err(|e| e.to_string())?
            {
                let weight = &source
                    .operands
                    .get(&name)
                    .ok_or_else(|| format!("missing dense operand {name}"))?
                    .weight;
                if !composes(declared, weight) {
                    return Err(format!(
                        "tensor {name}: the MTP layer composes no {:?} {}",
                        weight.format, declared.id
                    ));
                }
            }
        }
        mapped.extend(op.source.weights.iter().map(|w| w.codes.tensor.clone()));
        operations.push(SourceOperation {
            composition: Some(role.legacy_implementation(shape)),
            source: op.source,
        });
    }
    for projection in [Projection::GateUp, Projection::Down] {
        let op = routed(source, layer, projection, shape).map_err(|e| e.to_string())?;
        let weight = &op.source.weights[0];
        // The drafter's own bank, which records this operation, composes
        // its formats.
        let declared = ports
            .operation_projection(&op.source.id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("operation {}: no declared projection", op.source.id))
            .and_then(|id| ports.projection(id).map_err(|e| e.to_string()))?;
        if !composes(declared, weight) {
            let tensor = op
                .banks
                .first()
                .and_then(|bank| bank.operand.source_tensors.first())
                .cloned()
                .unwrap_or_else(|| weight.codes.tensor.clone());
            return Err(format!(
                "tensor {tensor}: the MTP layer composes no {:?} experts",
                weight.format
            ));
        }
        for bank in &op.banks {
            mapped.extend(bank.operand.source_tensors.iter().cloned());
        }
        operations.push(SourceOperation {
            source: op.source,
            composition: None,
        });
    }
    // The drafter's routing, hyper-connections, norms and input
    // projections keep their fixed BF16 contract.
    let owned = ports.drafter_sources().map_err(|e| e.to_string())?;
    for (name, operand) in &source.operands {
        if !owned.contains(name) || mapped.contains(name) {
            continue;
        }
        if operand.weight.format != WeightFormat::Bf16
            || operand.activation.as_ref().is_some_and(|a| !a.is_bf16())
        {
            return Err(format!(
                "tensor {name}: the MTP drafter requires BF16 here, not {:?}",
                operand.weight.format
            ));
        }
    }
    Ok(operations)
}

/// Whether `declared`'s composition admits `weight`'s format.
fn composes(
    declared: &projection_spec::Projection,
    weight: &kernel_core::precision::WeightSpec,
) -> bool {
    kernel_core::precision::format::registry::by_weight(weight).is_ok_and(|plugin| {
        declared
            .composition
            .as_ref()
            .is_some_and(|c| c.numerical_contracts.contains_key(&plugin.id()))
    })
}

impl SourceProgram {
    pub fn new(source: &SourcePrecision, shape: Shape) -> Result<Self> {
        let mut operations = Vec::new();
        let mut mapped = BTreeSet::new();
        for layer in 0..shape.layers() {
            for role in DenseRole::roles_for(shape, layer)? {
                let op = dense(source, layer, role, shape)?;
                mapped.extend(op.source.weights.iter().map(|w| w.codes.tensor.clone()));
                let composition = if op.legacy {
                    role.legacy_implementation(shape)
                } else {
                    role.mixed_implementation_for(shape, op.source.weights[0].format)
                };
                operations.push(SourceOperation {
                    source: op.source,
                    composition: Some(composition),
                });
            }
            for projection in [Projection::GateUp, Projection::Down] {
                let op = routed(source, layer, projection, shape)?;
                for bank in &op.banks {
                    mapped.extend(bank.operand.source_tensors.iter().cloned());
                }
                operations.push(SourceOperation {
                    source: op.source,
                    composition: None,
                });
            }
        }
        for ngram in [false, true] {
            let op = lookup(source, ngram, shape)?;
            mapped.extend(op.table.operand.source_tensors.iter().cloned());
            let usage = op.source.uses[0];
            let composition = if ngram && op.source.weights[0].format == WeightFormat::Nvfp4 {
                precision::lookup_implementation(shape)
            } else {
                precision::legacy_lookup_capability(shape, ngram, usage)?.implementation
            };
            operations.push(SourceOperation {
                source: op.source,
                composition: Some(composition),
            });
        }
        // Routing, hyper-connections, the head and PLE side projections retain
        // their family's fixed BF16-weight contract. Importer support alone
        // cannot redirect them into unavailable quantized implementations.
        let drafter_sources = precision::ports(shape)?.drafter_sources()?;
        for (name, operand) in &source.operands {
            if drafter_sources.contains(name)
                || name.starts_with("model.visual.")
                || mapped.contains(name)
            {
                continue;
            }
            if operand.weight.format != WeightFormat::Bf16
                || operand.activation.as_ref().is_some_and(|a| !a.is_bf16())
                || !operand.weight.transforms.is_empty()
                || operand
                    .activation
                    .as_ref()
                    .is_some_and(|a| !a.transforms.is_empty())
            {
                return Err(format!("tensor {name}: this Qwen side operation requires its supplied BF16 projection contract; no implementation exists for {:?}", operand.weight.format).into());
            }
        }
        let drafter = if source.legacy_text_layout() || shape.mtp_layers() == 0 {
            vec![]
        } else {
            drafter_operations(source, shape).unwrap_or_default()
        };
        Ok(Self {
            version: 1,
            ranks: shape.ranks(),
            geometry: shape.geometry.id.into(),
            operations,
            drafter,
        })
    }

    pub fn resolve(
        &self,
        policy: ActivationPolicy,
        inventory: &CapabilityInventory,
    ) -> Result<PrecisionDocument> {
        if self.version != 1 || ![2, 4].contains(&self.ranks) {
            return Err("invalid Qwen source operation version, topology or coverage".into());
        }
        let shape = crate::placement::bundle_shape_for(
            model_qwen38_contract::geometry::GEOMETRIES
                .iter()
                .find(|g| g.id == self.geometry)
                .ok_or("unknown source geometry")?,
            self.ranks as usize,
        )?;
        if self.operations.len() != operation_count(shape)?
            || (!self.drafter.is_empty()
                && self.drafter.len()
                    != model_qwen38_contract::prepared::drafter_operations(shape)?.len())
        {
            return Err("invalid Qwen source operation version, topology or coverage".into());
        }
        let mut operations = self
            .operations
            .iter()
            .map(|op| op.resolve(policy, inventory, self.ranks))
            .collect::<Result<Vec<_>>>()?;
        // A topology without the drafter's compositions still serves text.
        let mut drafter = Vec::new();
        for op in &self.drafter {
            match op.resolve(policy, inventory, self.ranks) {
                Ok(selection) => drafter.push(selection),
                Err(error) if error.downcast_ref::<UnsuppliedTopology>().is_some() => {
                    drafter.clear();
                    break;
                }
                Err(error) => return Err(error),
            }
        }
        operations.extend(drafter);
        let document = PrecisionDocument {
            version: 1,
            operations,
        };
        document.validate()?;
        model_compiler_sdk::quantization::identity::encode_component(&document)?;
        Ok(document)
    }

    /// Checks that `document` records this checkpoint's operations: its
    /// text operations, then its MTP layer's or none (a component prepared
    /// before MTP records serves text alone).
    pub fn verify_recorded(&self, document: &PrecisionDocument) -> Result<()> {
        document.validate()?;
        let recorded = document.operations.len();
        let expected: Vec<&SourceOperation> = if recorded == self.operations.len() {
            self.operations.iter().collect()
        } else if recorded == self.operations.len() + self.drafter.len() {
            self.operations.iter().chain(&self.drafter).collect()
        } else {
            return Err(
                "recorded precision differs from the checkpoint's imported operation recipes"
                    .into(),
            );
        };
        if expected
            .into_iter()
            .map(|op| op.current().source)
            .ne(document.operations.iter().map(|selection| {
                SourceOperation {
                    source: selection.source.clone(),
                    composition: None,
                }
                .current()
                .source
            }))
        {
            return Err(
                "recorded precision differs from the checkpoint's imported operation recipes"
                    .into(),
            );
        }
        Ok(())
    }
}

/// The CPU fixture of one topology's precision in a mixed NVFP4 package:
/// the synthetic NVFP4 layer-0 routed experts (gate/up and down) at
/// `shape`, each altered by `alter`, resolved under the auto policy against
/// the supplied mixed capabilities.
#[cfg(test)]
pub(crate) fn routed_nvfp4_fixture(
    shape: Shape,
    alter: impl Fn(&mut OperationSpec),
) -> Result<Vec<PrecisionSelection>> {
    routed_nvfp4_fixture_without(shape, None, alter)
}

/// The supplied mixed capabilities, less every grouped composition of
/// `withheld`'s local experts: an inventory that composes no NVFP4 experts
/// at that rank count.
#[cfg(test)]
pub(crate) fn inventory_without(withheld: Option<Shape>) -> Result<CapabilityInventory> {
    let mut operations = model_qwen38_contract::sm120::generated::all_capabilities()?;
    if let Some(withheld) = withheld {
        operations.retain(|c| {
            !matches!(c.kind, OperationKind::Grouped { experts, .. } if experts == withheld.local_experts)
        });
    }
    let capabilities = kernel_core::precision::capability::CapabilityDocument {
        version: 1,
        operations,
    };
    Ok(CapabilityInventory::new(
        capabilities.clone(),
        capabilities.entries(),
        "0".repeat(64),
    )?)
}

/// [`routed_nvfp4_fixture`] against [`inventory_without`]`(withheld)`.
#[cfg(test)]
pub(crate) fn routed_nvfp4_fixture_without(
    shape: Shape,
    withheld: Option<Shape>,
    alter: impl Fn(&mut OperationSpec),
) -> Result<Vec<PrecisionSelection>> {
    let source = super::tests::nvfp4_source();
    let inventory = inventory_without(withheld)?;
    [Projection::GateUp, Projection::Down]
        .into_iter()
        .map(|projection| {
            let mut operation = routed(&source, 0, projection, shape)?.source;
            alter(&mut operation);
            SourceOperation {
                source: operation,
                composition: None,
            }
            .resolve(ActivationPolicy::Auto, &inventory, shape.ranks())
        })
        .collect()
}

fn flash_id() -> String {
    "flash".into()
}
fn is_flash(id: &str) -> bool {
    id == "flash"
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_compiler_sdk::native_contract::NativeModule;
    use std::path::Path;

    #[test]
    fn previous_bf16_source_recipe_binds_exact_grouped_operands() {
        let source = super::super::tests::nvfp4_source();
        let mut recipe = routed(&source, 0, Projection::GateUp, Shape::TP4)
            .unwrap()
            .source;
        recipe
            .activations
            .fill(kernel_core::precision::ActivationSpec::bf16());
        recipe.arithmetic.weight_operand = DType::Bf16;
        recipe.arithmetic.decode_order = DecodeOrder::CodeThenScale;
        let op = SourceOperation {
            source: recipe,
            composition: None,
        }
        .current();
        let capabilities = kernel_core::precision::capability::CapabilityDocument {
            version: 1,
            operations: model_qwen38_contract::sm120::generated::all_capabilities().unwrap(),
        };
        let inventory =
            CapabilityInventory::new(capabilities.clone(), capabilities.entries(), "0".repeat(64))
                .unwrap();
        let provider = Provider::new(&inventory, None);
        let resolved = resolve_operation(&op.source, ActivationPolicy::Bf16, &provider).unwrap();
        assert_eq!(
            resolved.selection().effective.arithmetic.decode_order,
            DecodeOrder::ExactLocalThenGlobal
        );
        assert!(
            resolved.selection().bindings.iter().all(|b| b
                .implementation
                .activations
                .iter()
                .all(|a| a.is_bf16()))
        );
    }

    #[test]
    fn previous_mxfp4_bf16_recipes_resolve_under_checkpoint_and_auto() {
        let source = super::super::tests::mxfp4_source();
        let mut recipe = routed(&source, 0, Projection::GateUp, Shape::TP4)
            .unwrap()
            .source;
        recipe.uses.retain(|u| u.capacity == 1);
        let inventory = super::super::sm120::mxfp4_inventory(&recipe);
        for old_order in [DecodeOrder::CodeThenScale, DecodeOrder::ScaleThenCode] {
            recipe.arithmetic.decode_order = old_order;
            let op = SourceOperation {
                source: recipe.clone(),
                composition: None,
            };
            for policy in [ActivationPolicy::Checkpoint, ActivationPolicy::Auto] {
                let selection = op.resolve(policy, &inventory, Shape::TP4.ranks()).unwrap();
                assert_eq!(
                    selection.source.arithmetic.decode_order,
                    DecodeOrder::ExactScaledOperand
                );
                assert_eq!(
                    selection.effective.arithmetic.decode_order,
                    DecodeOrder::ExactScaledOperand
                );
                assert!(selection.reuse.is_none());
                assert!(selection.fallback_reasons.is_empty());
            }
        }
    }

    /// Makes an operation's first local scale reciprocal, which validation
    /// refuses.
    pub(crate) fn reciprocal_scale(operation: &mut OperationSpec) {
        operation.weights[0]
            .local_scale
            .as_mut()
            .unwrap()
            .convention = kernel_core::precision::ScaleConvention::Reciprocal;
    }

    /// Gives an operation's first weight even-high nibble packing, which
    /// the inventory does not admit.
    pub(crate) fn even_high_nibbles(operation: &mut OperationSpec) {
        operation.weights[0].encoding = kernel_core::precision::CodeEncoding::E2M1(
            kernel_core::precision::NibbleOrder::EvenHigh,
        );
    }

    /// The supplied grouped NVFP4 kernels serve TP2's and TP4's local
    /// experts. Without TP2's, a mixed routed operation is an unsupplied
    /// topology there (which a package leaves out), and at TP4 with one of
    /// its supplied phases withdrawn it fails as a defect instead.
    #[test]
    fn routed_nvfp4_without_supplied_kernels_at_a_rank_count_is_an_unsupplied_topology() {
        let source = super::super::tests::nvfp4_source();
        let capabilities = kernel_core::precision::capability::CapabilityDocument {
            version: 1,
            operations: model_qwen38_contract::sm120::generated::all_capabilities().unwrap(),
        };
        let supplied = inventory_without(None).unwrap();
        let inventory = inventory_without(Some(Shape::TP2)).unwrap();
        for projection in [Projection::GateUp, Projection::Down] {
            let op = |shape| SourceOperation {
                source: routed(&source, 0, projection, shape).unwrap().source,
                composition: None,
            };
            for policy in [ActivationPolicy::Auto, ActivationPolicy::Bf16] {
                for shape in [Shape::TP2, Shape::TP4] {
                    op(shape).resolve(policy, &supplied, shape.ranks()).unwrap();
                }
                op(Shape::TP4).resolve(policy, &inventory, 4).unwrap();
            }
            let shape = Shape::TP2;
            let error = op(shape)
                .resolve(ActivationPolicy::Auto, &inventory, shape.ranks())
                .unwrap_err();
            let unsupplied = error
                .downcast_ref::<UnsuppliedTopology>()
                .unwrap_or_else(|| panic!("TP{}: {error}", shape.ranks()));
            assert!(
                unsupplied.0.contains(&format!("at TP{}", shape.ranks())),
                "{unsupplied}"
            );

            // An invalid or inadmissible operation at a rank count without
            // the composition still fails as itself: a reciprocal scale
            // fails validation before binding, and even-high nibble packing
            // is refused by the inventory.
            for invalid in [
                reciprocal_scale as fn(&mut OperationSpec),
                even_high_nibbles,
            ] {
                let mut source = op(Shape::TP2).source;
                invalid(&mut source);
                let error = SourceOperation {
                    source,
                    composition: None,
                }
                .resolve(ActivationPolicy::Auto, &inventory, 2)
                .unwrap_err();
                assert!(
                    error.downcast_ref::<UnsuppliedTopology>().is_none(),
                    "{error}"
                );
            }
            let tp4 = op(Shape::TP4).source;
            let geometry = tp4.matrix_geometry().unwrap();
            let mut partial = capabilities.clone();
            let first = partial
                .operations
                .iter()
                .position(|c| c.kind == tp4.kind && c.geometry == geometry)
                .unwrap();
            partial.operations.remove(first);
            assert!(
                partial
                    .operations
                    .iter()
                    .any(|c| c.kind == tp4.kind && c.geometry == geometry)
            );
            let partial =
                CapabilityInventory::new(partial.clone(), partial.entries(), "0".repeat(64))
                    .unwrap();
            let error = op(Shape::TP4)
                .resolve(ActivationPolicy::Auto, &partial, 4)
                .unwrap_err();
            assert!(
                error.downcast_ref::<UnsuppliedTopology>().is_none(),
                "{error}"
            );
        }
    }

    #[test]
    fn real_source_program_resolves_from_the_supplied_native_inventory() {
        let (Ok(root), Ok(native)) = (
            std::env::var("CHOPINFER_QWEN38_CHECKPOINT"),
            std::env::var("CHOPINFER_QWEN38_NATIVE_MODULE"),
        ) else {
            eprintln!(
                "UNRUN: set CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_NATIVE_MODULE for native-backed source resolution"
            );
            return;
        };
        let catalog = crate::catalog::Catalog::load(Path::new(&root)).unwrap();
        let source = SourceProgram::new(catalog.precision.as_ref().unwrap(), Shape::TP4).unwrap();
        let serialized = serde_json::to_vec(&source).unwrap();
        let source: SourceProgram = serde_json::from_slice(&serialized).unwrap();
        let module = NativeModule::inspect(Path::new(&native)).unwrap();
        let inventory = module.precision_inventory().unwrap();
        let auto = source.resolve(ActivationPolicy::Auto, &inventory).unwrap();
        let bf16 = source.resolve(ActivationPolicy::Bf16, &inventory).unwrap();
        assert_eq!(auto.operations.len(), 434);
        for (a, b) in auto.operations.iter().zip(&bf16.operations) {
            assert_eq!(a.effective, b.effective);
            assert_eq!(a.bindings, b.bindings);
        }
        assert!(
            source
                .resolve(ActivationPolicy::Checkpoint, &inventory)
                .is_err()
        );
        let plan = crate::placement::ImagePlan::with_precision(&catalog, None, &auto).unwrap();
        let bf16_plan = crate::placement::ImagePlan::with_precision(&catalog, None, &bf16).unwrap();
        assert_eq!(plan, bf16_plan);
        let decoded = model_compiler_sdk::quantization::identity::decode_component(
            &model_compiler_sdk::quantization::identity::encode_component(&auto).unwrap(),
        )
        .unwrap();
        assert_eq!(decoded, auto);
        model_qwen38_contract::prepared::bind(&decoded, Shape::TP4, &inventory).unwrap();
        let oracle =
            crate::oracle::weights::Checkpoint::open_prepared(Path::new(&root), &auto).unwrap();
        for name in [
            "model.language_model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.127.up_proj.weight",
            "model.language_model.layers.0.mlp.experts.128.down_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        ] {
            let weight = oracle.projection(name).unwrap();
            assert_eq!(
                vec![weight.rows as u64, weight.cols as u64],
                oracle.logical_shape(name).unwrap()
            );
            assert!(weight.data.iter().all(|v| v.is_finite()));
        }
        let table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
        let table_rows = oracle
            .prepared_lookup_rows(&[
                (format!("{table}.shard_0.weight"), 0),
                (format!("{table}.shard_127.weight"), 2_500_011),
            ])
            .unwrap();
        assert!(
            table_rows
                .iter()
                .all(|row| row.len() == 160 && row.iter().all(|v| v.is_finite()))
        );
        for index in &plan.ranks {
            let mut index = index.clone();
            index.sha256 = "0".repeat(64);
            let parsed = model_qwen38_contract::views::parse(
                &serde_json::to_vec(&index).unwrap(),
                index.rank as u32,
            )
            .unwrap();
            model_qwen38_contract::prepared::validate_views(&parsed, Shape::TP4, &auto).unwrap();
        }
        let mut altered = auto.clone();
        altered.operations[0].bindings[0].implementation.dependency = "0".repeat(64);
        assert!(model_qwen38_contract::prepared::bind(&altered, Shape::TP4, &inventory).is_err());
        let mut altered = auto.clone();
        altered.operations.remove(0);
        assert!(model_qwen38_contract::prepared::bind(&altered, Shape::TP4, &inventory).is_err());
        if catalog.precision.as_ref().unwrap().legacy_text_layout() {
            assert_eq!(
                plan,
                crate::placement::ImagePlan::new(&catalog, None).unwrap()
            );
        } else {
            assert!(plan.sources.keys().all(|name| !name.starts_with("mtp.")));
            for index in &plan.ranks {
                let views: std::collections::BTreeMap<_, _> =
                    index.views.iter().map(|v| (v.name.as_str(), v)).collect();
                let gate_up = "model.language_model.layers.0.mlp.experts.gate_up_proj";
                assert_eq!(
                    views[format!("{gate_up}.weight").as_str()].bytes,
                    128 * 2 * 640 * 2560 / 2
                );
                assert_eq!(
                    views[format!("{gate_up}.weight_scale").as_str()].bytes,
                    128 * 2 * 640 * 160
                );
                assert_eq!(
                    views[format!("{gate_up}.weight_scale_2").as_str()].bytes,
                    128 * 2 * 4
                );
                assert_eq!(
                    views[format!("{gate_up}.input_scale").as_str()].bytes,
                    128 * 2 * 4
                );
            }
        }
        eprintln!(
            "source snapshot {} bytes; precision component {} bytes; planned rank bytes {:?}",
            serialized.len(),
            model_compiler_sdk::quantization::identity::encode_component(&auto)
                .unwrap()
                .len(),
            plan.image_bytes(),
        );
    }
}

#[cfg(test)]
pub(crate) mod drafter_tests {
    use super::*;
    use kernel_core::precision::TensorSpan;
    use model_compiler_sdk::quantization::import::ExportMetadata;
    use model_compiler_sdk::quantization::source::TensorInventory;
    use std::collections::BTreeMap;

    pub(crate) use crate::mixed_layouts::Layout;

    /// The MTP layer of a mixed layout: its synthetic configuration and
    /// quantization rules, every `mtp.` tensor at its checkpoint shape, and
    /// the MTP experts in the layout's encoding (FP8 block 128 with BF16
    /// inverse scales, or weight-only NVFP4).
    pub(crate) fn mtp_source(layout: Layout) -> SourcePrecision {
        mtp_source_with(layout, &[])
    }

    /// [`mtp_source`] with the MTP projection weights `mxfp8` quantized to
    /// MXFP8 by added export rules.
    fn mtp_source_with(layout: Layout, mxfp8: &[&str]) -> SourcePrecision {
        let config = crate::config::Qwen38Config::from_bytes(
            &serde_json::to_vec(&crate::mixed_layouts::config(layout)).unwrap(),
        )
        .unwrap();
        let mut rules = crate::mixed_layouts::hf_quant_config(layout);
        for weight in mxfp8 {
            let layers = rules
                .get("quantization")
                .and_then(|q| q.get("quantized_layers"))
                .is_some();
            let rule = serde_json::json!({"quant_algo": "MXFP8", "group_size": 32});
            let module = weight.trim_end_matches(".weight").to_owned();
            if layers {
                rules["quantization"]["quantized_layers"][&module] = rule;
            } else {
                rules["quantized_layers"][&module] = rule;
            }
        }
        let metadata = ExportMetadata::from_documents(
            Some(&config.quantization_config),
            Some(&serde_json::to_vec(&rules).unwrap()),
        )
        .unwrap();
        let mut logical = crate::catalog::expected_tensors_with_config(
            config.geometry().unwrap().geometry,
            &serde_json::to_value(&config).unwrap(),
        )
        .unwrap();
        logical.retain(|name, _| name.starts_with("mtp.") && !name.ends_with("_scale_inv"));
        let mut tensors = BTreeMap::new();
        let mut add = |name: String, dtype: DType, shape: Vec<u64>| {
            let span = TensorSpan {
                byte_length: shape.iter().product::<u64>()
                    * kernel_core::precision::dtype_bits(dtype)
                    / 8,
                tensor: name.clone(),
                dtype,
                shape,
                offset: 0,
            };
            tensors.insert(name, span);
        };
        for (name, (dtype, shape)) in &logical {
            if mxfp8.contains(&name.as_str()) {
                add(name.clone(), DType::Fp8E4M3, shape.clone());
                add(
                    name.replace(".weight", ".weight_scale"),
                    DType::Ue8M0,
                    vec![shape[0], shape[1] / 32],
                );
            } else if name.contains(".mlp.experts.") && name.ends_with(".weight") {
                let (n, k) = (shape[0], shape[1]);
                if layout == Layout::Nvfp4RoutedFp8Mtp {
                    add(name.clone(), DType::Fp8E4M3, vec![n, k]);
                    add(
                        name.replace(".weight", ".weight_scale_inv"),
                        DType::Bf16,
                        vec![n.div_ceil(128), k.div_ceil(128)],
                    );
                } else {
                    add(name.clone(), DType::U8, vec![n, k / 2]);
                    add(
                        name.replace(".weight", ".weight_scale"),
                        DType::Fp8E4M3,
                        vec![n, k / 16],
                    );
                    add(
                        name.replace(".weight", ".weight_scale_2"),
                        DType::F32,
                        vec![],
                    );
                }
            } else {
                let dtype = match dtype.as_str() {
                    "BF16" => DType::Bf16,
                    "F32" => DType::F32,
                    "I64" => DType::I64,
                    other => panic!("unexpected MTP dtype {other}"),
                };
                add(name.clone(), dtype, shape.clone());
            }
        }
        logical.retain(|name, _| tensors.contains_key(name));
        let inventory = TensorInventory {
            tensors,
            spans: BTreeMap::new(),
        };
        crate::precision_source::SourcePrecision::admit(&config, metadata, inventory, logical)
            .unwrap()
    }

    /// The MTP layer of a mixed layout at `shape`, resolved under the auto
    /// policy: the records a mixed component appends to its text operations.
    pub(crate) fn drafter_records(layout: Layout, shape: Shape) -> Vec<PrecisionSelection> {
        let inventory = inventory();
        drafter_operations(&mtp_source(layout), shape)
            .unwrap()
            .iter()
            .map(|op| {
                op.resolve(ActivationPolicy::Auto, &inventory, shape.ranks())
                    .unwrap()
            })
            .collect()
    }

    fn inventory() -> CapabilityInventory {
        let mut operations = model_qwen38_contract::sm120::generated::all_capabilities().unwrap();
        operations.extend(precision::legacy_capabilities().unwrap());
        let document = kernel_core::precision::capability::CapabilityDocument {
            version: 1,
            operations,
        };
        CapabilityInventory::new(document.clone(), document.entries(), "0".repeat(64)).unwrap()
    }

    /// Both layouts' MTP layers record their seven BF16 dense ports on the
    /// registered compositions and their experts: block FP8 experts on the
    /// registered fragment entries, weight-only NVFP4 experts on the main
    /// layers' generated entries, at TP2 and TP4, under the auto and BF16
    /// policies alike.
    #[test]
    fn both_layouts_record_their_mtp_layers_at_every_topology() {
        let inventory = inventory();
        for layout in Layout::ALL {
            let source = mtp_source(layout);
            for shape in [Shape::TP2, Shape::TP4] {
                let label = format!("{} TP{}", layout.name(), shape.ranks());
                let operations = drafter_operations(&source, shape).unwrap();
                let ids: Vec<_> = operations.iter().map(|op| op.source.id.clone()).collect();
                assert_eq!(
                    ids,
                    model_qwen38_contract::prepared::drafter_operations(shape).unwrap(),
                    "{label}"
                );
                assert_eq!(ids.len(), 9, "{label}");
                for op in &operations {
                    let auto = op.resolve(ActivationPolicy::Auto, &inventory, shape.ranks());
                    let auto = auto.unwrap_or_else(|e| panic!("{label} {}: {e}", op.source.id));
                    let bf16 = op
                        .resolve(ActivationPolicy::Bf16, &inventory, shape.ranks())
                        .unwrap();
                    assert_eq!(auto.effective, bf16.effective, "{label}");
                    assert!(auto.effective.activations.iter().all(|a| a.is_bf16()));
                    let OperationKind::Grouped { experts, .. } = auto.effective.kind else {
                        assert_eq!(auto.effective.weights[0].format, WeightFormat::Bf16);
                        assert!(
                            auto.bindings[0]
                                .implementation
                                .implementation
                                .starts_with("qwen38.legacy-dense."),
                            "{label}"
                        );
                        continue;
                    };
                    assert_eq!(experts, shape.local_experts, "{label}");
                    let main = if op.source.epilogues.swiglu.is_some() {
                        "main.experts.gate-up"
                    } else {
                        "main.experts.down"
                    };
                    for binding in &auto.bindings {
                        let expected = if layout == Layout::Nvfp4RoutedFp8Mtp {
                            precision::legacy_expert_capability(
                                shape,
                                if main.ends_with("gate-up") {
                                    Projection::GateUp
                                } else {
                                    Projection::Down
                                },
                                binding.usage,
                            )
                            .unwrap()
                            .stages
                        } else {
                            model_qwen38_contract::sm120::generated::publish(
                                shape,
                                main,
                                WeightFormat::Nvfp4,
                                binding.usage,
                            )
                            .unwrap()
                            .capability
                            .stages
                            .clone()
                        };
                        assert_eq!(binding.implementation.stages, expected, "{label}");
                    }
                }
            }
        }
    }

    /// An MTP encoding without a composition records no MTP operations and
    /// names the tensor and its format: a quantized MTP attention port, a
    /// quantized MTP side tensor and a missing MTP tensor.
    #[test]
    fn mtp_encodings_without_a_composition_name_the_tensor() {
        let q = "mtp.layers.0.self_attn.q_proj.weight";
        let quantized = mtp_source_with(Layout::Mxfp8DenseNvfp4, &[q]);
        assert_eq!(quantized.operands[q].weight.format, WeightFormat::Mxfp8);
        let side = "mtp.fc_hidden.weight";
        let mut side_quantized = mtp_source(Layout::Nvfp4RoutedFp8Mtp);
        side_quantized.operands.get_mut(side).unwrap().weight.format = WeightFormat::Fp8PerTensor;
        let missing = "mtp.layers.0.self_attn.o_proj.weight";
        let mut absent = mtp_source(Layout::Mxfp8DenseNvfp4);
        absent.operands.remove(missing);
        // Alone, the block FP8 MTP layer is the registered layout's: its
        // drafter keeps the fixed program and records nothing.
        let registered = mtp_source(Layout::Nvfp4RoutedFp8Mtp);
        assert!(registered.legacy_text_layout());
        assert_eq!(drafter_refusal(&registered, Shape::TP4), None);
        for shape in [Shape::TP2, Shape::TP4] {
            for (source, expected) in [
                (
                    &quantized,
                    format!("tensor {q}: the MTP layer composes no Mxfp8 mtp.q"),
                ),
                (
                    &side_quantized,
                    format!("tensor {side}: the MTP drafter requires BF16 here, not Fp8PerTensor"),
                ),
                (&absent, format!("missing dense operand {missing}")),
            ] {
                let error = drafter_operations(source, shape).err().unwrap();
                assert_eq!(error, expected, "TP{}", shape.ranks());
                assert_eq!(drafter_refusal(source, shape), Some(expected));
            }
        }
    }

    /// The MTP layer's dense projections and their fusion peers compose BF16
    /// weights alone, so their declared compositions refuse every other
    /// format.
    #[test]
    fn mtp_dense_compositions_admit_only_bf16() {
        for shape in [Shape::TP2, Shape::TP4] {
            let ports = precision::ports(shape).unwrap();
            let layer = shape.mtp_layer();
            for role in DenseRole::roles_for(shape, layer).unwrap() {
                let declared = ports
                    .projection(role.projection_id(shape, Some(layer)))
                    .unwrap();
                let formats: Vec<_> = declared
                    .composition
                    .as_ref()
                    .unwrap()
                    .numerical_contracts
                    .keys()
                    .copied()
                    .collect();
                assert_eq!(
                    formats,
                    [kernel_core::precision::format::FormatId::Bf16],
                    "{}",
                    declared.id
                );
            }
        }
    }

    /// Every MTP dense projection whose weights are not BF16 is refused by
    /// its declared composition, naming its first tensor and format: alone,
    /// and as the fusion peer of a BF16 projection that comes before it.
    #[test]
    fn every_non_bf16_mtp_dense_projection_is_refused_naming_its_tensor() {
        for shape in [Shape::TP2, Shape::TP4] {
            let ports = precision::ports(shape).unwrap();
            let layer = shape.mtp_layer();
            for role in DenseRole::roles_for(shape, layer).unwrap() {
                let projection = role.projection_id(shape, Some(layer));
                let names = ports.names(projection, Some(u64::from(layer))).unwrap();
                let names: Vec<_> = names.iter().map(String::as_str).collect();
                let source = mtp_source_with(Layout::Mxfp8DenseNvfp4, &names);
                assert_eq!(
                    drafter_operations(&source, shape).err(),
                    Some(format!(
                        "tensor {}: the MTP layer composes no Mxfp8 {projection}",
                        names[0]
                    )),
                    "TP{}",
                    shape.ranks()
                );
            }
        }
    }

    /// The importer gives an MTP dense weight it keeps in BF16 a BF16
    /// activation, which the legacy BF16 rule also required: in both
    /// layouts' MTP layers, and under every supported algorithm as the
    /// export default (with the module also excluded), as a rule for the
    /// MTP module and as a quantization group with or without quantized
    /// inputs. Only an exclusion or the absence of a rule keeps a weight in
    /// BF16, and both give it BF16 activations.
    #[test]
    fn the_importer_gives_bf16_mtp_weights_only_bf16_activations() {
        use model_compiler_sdk::quantization::import::ActivationRecipe;
        let shape = Shape::TP4;
        let ports = precision::ports(shape).unwrap();
        let layer = shape.mtp_layer();
        let mut weights = Vec::new();
        for role in DenseRole::roles_for(shape, layer).unwrap() {
            weights.extend(
                ports
                    .names(
                        role.projection_id(shape, Some(layer)),
                        Some(u64::from(layer)),
                    )
                    .unwrap(),
            );
        }
        for layout in Layout::ALL {
            let source = mtp_source(layout);
            let name = layout.name();
            for weight in &weights {
                let operand = &source.operands[weight];
                assert_eq!(operand.weight.format, WeightFormat::Bf16, "{name} {weight}");
                assert!(
                    operand.activation.as_ref().is_some_and(|a| a.is_bf16()),
                    "{name} {weight}"
                );
            }
        }
        let algorithms = [
            ("NVFP4", Some(16)),
            ("W4A16_NVFP4", Some(16)),
            ("MXFP8", Some(32)),
            ("FP8_PB_WO", Some(128)),
            ("FP8_BLOCK_SCALES", Some(128)),
            ("FP8", None),
        ];
        let float = |bits: u32, group: Option<u32>, dynamic: bool| serde_json::json!({"dynamic": dynamic, "num_bits": bits, "type": "float", "group_size": group});
        let (mut bf16, mut quantized) = (0, 0);
        for weight in &weights {
            let module = weight.trim_end_matches(".weight");
            let mut documents = Vec::new();
            for (algorithm, group) in algorithms {
                documents.push(serde_json::json!({"quant_method": "modelopt",
                    "quant_algo": algorithm, "group_size": group}));
                documents.push(serde_json::json!({"quant_method": "modelopt",
                    "quant_algo": algorithm, "group_size": group, "ignore": [module]}));
                documents.push(serde_json::json!({"quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {module: {"quant_algo": algorithm, "group_size": group}}}));
            }
            for weights in [
                float(4, Some(16), false),
                float(8, Some(32), false),
                float(8, Some(128), false),
                float(8, None, false),
            ] {
                for inputs in [
                    None,
                    Some(float(4, Some(16), false)),
                    Some(float(4, Some(16), true)),
                    Some(float(8, Some(32), true)),
                    Some(float(8, Some(128), true)),
                ] {
                    let mut group = serde_json::json!({"weights": weights, "targets": [module]});
                    if let Some(inputs) = inputs {
                        group["input_activations"] = inputs;
                    }
                    documents.push(serde_json::json!({"quant_method": "modelopt",
                        "quant_algo": "MIXED_PRECISION", "config_groups": {"group_0": group}}));
                }
            }
            for document in documents {
                let Ok(metadata) = ExportMetadata::from_documents(Some(&document), None) else {
                    continue;
                };
                let Ok(recipe) = metadata.recipe(&[module.into()], None) else {
                    continue;
                };
                if recipe.weight == WeightFormat::Bf16 {
                    assert_eq!(
                        recipe.activation,
                        Some(ActivationRecipe::Bf16),
                        "{document}"
                    );
                    bf16 += 1;
                } else if recipe
                    .activation
                    .is_some_and(|a| a != ActivationRecipe::Bf16)
                {
                    quantized += 1;
                }
            }
        }
        // Both sides of the rule are exercised for every MTP dense weight.
        assert!(bf16 >= weights.len() * algorithms.len(), "{bf16}");
        assert!(quantized >= weights.len() * 4, "{quantized}");
    }

    /// A supplied component records the text operations and then the MTP
    /// layer's, or the text operations alone (it predates MTP records and
    /// serves text); any other prefix or count is refused.
    #[test]
    fn recorded_components_carry_all_mtp_operations_or_none() {
        let inventory = inventory();
        let shape = Shape::TP4;
        let text = super::super::text::routed_nvfp4_fixture(shape, |_| {}).unwrap();
        let drafter = drafter_operations(&mtp_source(Layout::Mxfp8DenseNvfp4), shape).unwrap();
        let program = SourceProgram {
            version: 1,
            ranks: shape.ranks(),
            geometry: shape.geometry.id.into(),
            operations: text
                .iter()
                .map(|s| SourceOperation {
                    source: s.source.clone(),
                    composition: None,
                })
                .collect(),
            drafter: drafter.clone(),
        };
        let recorded: Vec<_> = drafter
            .iter()
            .map(|op| {
                op.resolve(ActivationPolicy::Auto, &inventory, shape.ranks())
                    .unwrap()
            })
            .collect();
        let document = |operations: Vec<PrecisionSelection>| PrecisionDocument {
            version: 1,
            operations,
        };
        let full = document(text.iter().cloned().chain(recorded.clone()).collect());
        program.verify_recorded(&full).unwrap();
        program.verify_recorded(&document(text.clone())).unwrap();
        let partial = document(
            text.iter()
                .cloned()
                .chain(recorded[..recorded.len() - 1].iter().cloned())
                .collect(),
        );
        assert!(program.verify_recorded(&partial).is_err());
        let reordered = document(recorded.iter().cloned().chain(text.clone()).collect());
        assert!(program.verify_recorded(&reordered).is_err());
        // Serialization keeps a component without MTP records byte for byte.
        let mut registered = program.clone();
        registered.drafter.clear();
        let bytes = serde_json::to_string(&registered).unwrap();
        assert!(!bytes.contains("drafter"), "{bytes}");
        let round: SourceProgram =
            serde_json::from_str(&serde_json::to_string(&program).unwrap()).unwrap();
        assert_eq!(round.drafter.len(), drafter.len());
    }
}
