//! Package test fixtures read from the SM120 catalog offers.

use std::collections::BTreeMap;

use model_mimo26_contract::topology::Tp;

/// A small admitted component for tests of package metadata and file proofs.
pub fn fixture_precision() -> BTreeMap<String, Vec<u8>> {
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    let usage = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 1,
        graph: true,
    };
    Tp::SUPPORTED
        .into_iter()
        .map(|tp| {
            let cap = model_mimo26_contract::capabilities::capability(
                model_mimo26_contract::capabilities::Projection::Head { f32: false },
                tp,
                usage,
            )
            .unwrap();
            let mut operation =
                kernels_sm120::precision::generated::comparison::operation(&cap).unwrap();
            operation.id = "head.bf16".into();
            let bound = model_compiler_sdk::quantization::resolve_operation(
                &operation,
                model_compiler_sdk::quantization::ActivationPolicy::Bf16,
                &model_mimo26_contract::prepared::Provider { tp },
            )
            .unwrap();
            let document = kernel_core::precision::execution::PrecisionDocument {
                version: 1,
                operations: vec![bound.selection().clone()],
            };
            (
                model_mimo26_contract::prepared::component_id(tp),
                kernel_core::precision::component::encode(&document).unwrap(),
            )
        })
        .collect()
}
