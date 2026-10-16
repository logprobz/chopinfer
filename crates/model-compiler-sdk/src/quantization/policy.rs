//! Resolve a complete logical operation before weight layout or package emission.
//! Providers return bindings to supplied implementations; they must not compile.

use crate::CompilerError;
use kernel_core::contract::DType;
pub use kernel_core::precision::execution::{
    ACTIVATION_POLICY_ENV, ActivationPolicy, Binding, BoundUse, ImplementationIdentity,
    ImplementationProvider, PrecisionSelection, WeightReuse,
};
use kernel_core::precision::{
    ActivationSpec, OperationKind, OperationSpec, OperationUse, ScaleConvention,
};

/// The recipe and retained phase-specific handles cannot be changed by later
/// environment variables. Serving selects a prepared use, not another precision.
pub struct PreparedOperation<H> {
    selection: PrecisionSelection,
    handles: Vec<H>,
}

impl<H> PreparedOperation<H> {
    pub fn selection(&self) -> &PrecisionSelection {
        &self.selection
    }
    pub fn handle(&self, usage: OperationUse) -> Option<&H> {
        self.selection
            .bindings
            .iter()
            .position(|b| b.usage == usage)
            .map(|i| &self.handles[i])
    }
    pub fn into_parts(self) -> (PrecisionSelection, Vec<H>) {
        (self.selection, self.handles)
    }
}

fn validate_requirements(operation: &OperationSpec) -> Result<(), CompilerError> {
    operation
        .validate()
        .map_err(|e| CompilerError::validation(format!("operation {}: {e}", operation.id)))?;
    for weight in &operation.weights {
        if !weight.transforms.is_empty() {
            return Err(CompilerError::validation(format!(
                "operation {}, tensor {}: required weight/input transform {:?} is not implemented; activation policy cannot discard it",
                operation.id, weight.codes.tensor, weight.transforms
            )));
        }
        if weight
            .local_scale
            .iter()
            .chain(weight.global_scale.iter())
            .any(|s| s.convention != ScaleConvention::Multiply)
        {
            return Err(CompilerError::validation(format!(
                "operation {}, tensor {}: reciprocal scale interpretation requires a supported importer transform",
                operation.id, weight.codes.tensor
            )));
        }
    }
    if operation
        .activations
        .iter()
        .any(|a| !a.transforms.is_empty())
    {
        return Err(CompilerError::validation(format!(
            "operation {}: required activation transform is not implemented; activation policy cannot discard it",
            operation.id
        )));
    }
    Ok(())
}

type BoundImplementations<H> = (Vec<BoundUse>, Vec<H>);

fn bind_all<P: ImplementationProvider>(
    operation: &OperationSpec,
    provider: &P,
) -> Result<BoundImplementations<P::Handle>, Vec<String>> {
    let mut bindings: Vec<BoundUse> = Vec::new();
    let mut handles = Vec::new();
    let mut reasons = Vec::new();
    for &usage in &operation.uses {
        let result = provider.bind(operation, usage).and_then(|binding| {
            let identity = &binding.identity;
            let mut arithmetic = operation.arithmetic.clone();
            arithmetic.numerical_contract = identity.arithmetic.numerical_contract.clone();
            if identity.activations != operation.activations || identity.arithmetic != arithmetic {
                return Err(
                    "supplied implementation changes the activation or numerical contract".into(),
                );
            }
            if identity.weight_layouts.len() != operation.weights.len()
                || identity.implementation.is_empty()
                || identity.dependency.is_empty()
                || identity.stages.is_empty()
                || identity.stages.iter().any(String::is_empty)
            {
                return Err("incomplete supplied implementation identity or weight layouts".into());
            }
            if bindings.first().is_some_and(|first| {
                first.implementation.weight_layouts != identity.weight_layouts
                    || first.implementation.arithmetic != identity.arithmetic
            }) {
                return Err(
                    "phase bindings require incompatible persistent weight layouts or arithmetic"
                        .into(),
                );
            }
            Ok(binding)
        });
        match result {
            Ok(binding) => {
                bindings.push(BoundUse {
                    usage,
                    implementation: binding.identity,
                });
                handles.push(binding.handle);
            }
            Err(reason) => reasons.push(format!(
                "{:?} capacity {} graph={}: {reason}",
                usage.phase, usage.capacity, usage.graph
            )),
        }
    }
    if reasons.is_empty() {
        Ok((bindings, handles))
    } else {
        Err(reasons)
    }
}

fn selected<H>(
    source: &OperationSpec,
    mut effective: OperationSpec,
    policy: ActivationPolicy,
    reuse: Option<WeightReuse>,
    fallback_reasons: Vec<String>,
    bindings: Vec<BoundUse>,
    handles: Vec<H>,
) -> Result<PreparedOperation<H>, CompilerError> {
    effective.arithmetic = bindings[0].implementation.arithmetic.clone();
    for (weight, layout) in effective
        .weights
        .iter_mut()
        .zip(&bindings[0].implementation.weight_layouts)
    {
        weight.layout = layout.clone();
    }
    let prepared = PreparedOperation {
        selection: PrecisionSelection {
            source: source.clone(),
            effective,
            policy,
            reuse,
            fallback_reasons,
            bindings,
        },
        handles,
    };
    prepared
        .selection
        .validate()
        .map_err(|e| CompilerError::validation(e.to_string()))?;
    Ok(prepared)
}

pub fn resolve_operation<P: ImplementationProvider>(
    source: &OperationSpec,
    policy: ActivationPolicy,
    provider: &P,
) -> Result<PreparedOperation<P::Handle>, CompilerError> {
    validate_requirements(source)?;
    let mut source_failure = Vec::new();
    let source_is_bf16 = source.activations.iter().all(ActivationSpec::is_bf16);
    if policy != ActivationPolicy::Bf16 || source_is_bf16 {
        match bind_all(source, provider) {
            Ok((bindings, handles)) => {
                return selected(
                    source,
                    source.clone(),
                    policy,
                    None,
                    vec![],
                    bindings,
                    handles,
                );
            }
            Err(reasons) => source_failure = reasons,
        }
    }
    if policy == ActivationPolicy::Checkpoint
        || source_is_bf16
        || source.kind == OperationKind::Lookup
    {
        return Err(refusal(source, policy, &source_failure));
    }
    let reuse = WeightReuse::for_format(source.weights[0].format).ok_or_else(|| {
        refusal(
            source,
            policy,
            &["no implemented BF16 weight-reuse policy for this source recipe".into()],
        )
    })?;
    let mut effective = source.clone();
    if effective.epilogues.output_quantization.is_some() {
        return Err(refusal(source, policy, &["BF16 override of a quantized output boundary requires an explicit supported composition".into()]));
    }
    effective.activations.fill(ActivationSpec::bf16());
    effective.arithmetic.weight_operand = DType::Bf16;
    // Weight reuse changes the arithmetic implementation as well as the MMA
    // operands. Keep output/accumulation fixed and bind one complete, explicitly
    // permitted scale policy. Never choose a different order for each bucket.
    let mut reuse_failures = vec![format!("BF16 weight reuse {reuse:?} is unavailable")];
    for order in reuse.bf16_decode_orders(source) {
        effective.arithmetic.decode_order = order;
        match bind_all(&effective, provider) {
            Ok((bindings, handles)) => {
                return selected(
                    source,
                    effective,
                    policy,
                    Some(reuse),
                    source_failure,
                    bindings,
                    handles,
                );
            }
            Err(reasons) => reuse_failures.extend(reasons),
        }
    }
    source_failure.extend(reuse_failures);
    Err(refusal(source, policy, &source_failure))
}

fn refusal(source: &OperationSpec, policy: ActivationPolicy, reasons: &[String]) -> CompilerError {
    CompilerError::validation(format!(
        "operation {}, tensors [{}], activation policy {policy}: {}",
        source.id,
        source
            .weights
            .iter()
            .map(|w| w.codes.tensor.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        reasons.join("; ")
    ))
}
