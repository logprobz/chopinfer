//! The SM120 grouped offers of an MXFP4 expert source with BF16 activations,
//! which the binding tests bind without a fallback.

use super::*;
use kernel_core::contract::Target;
use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};

pub(super) fn mxfp4_inventory(op: &OperationSpec) -> CapabilityInventory {
    use kernels_sm120::{moe, precision::KernelOffer, row::precision::Nvfp4Global};
    let projection = if op.epilogues.swiglu.is_some() {
        moe::Projection::GateUp
    } else {
        moe::Projection::Down
    };
    let operations = op
        .uses
        .iter()
        .map(|&usage| {
            KernelOffer::grouped(
                moe::GroupedRequest {
                    shape: moe::Shape {
                        hidden: HIDDEN,
                        intermediate: EXPERT_INTERMEDIATE,
                        experts: Shape::TP4.local_experts,
                        top_k: TOP_K,
                    },
                    projection,
                    route_weight: moe::RouteWeight::AfterDown,
                    swiglu_clamp: None,
                    phase: if usage.phase == kernel_core::precision::ExecutionPhase::Prefill {
                        moe::Phase::Prefill
                    } else {
                        moe::Phase::Decode
                    },
                    tokens_capacity: usage.capacity,
                    weight: WeightFormat::Mxfp4,
                    activation: ActivationFormat::Bf16,
                    scales: moe::ScaleBindings {
                        weight_globals: moe::GlobalScaleLayout::Absent,
                        activation_globals: moe::GlobalScaleLayout::Absent,
                        shared_gate_up_activation: true,
                    },
                    output_quantization: None,
                    swiglu_columns: SwigluColumns::ConcatenatedHalves,
                    target: Target::Sm120f,
                    symbol: format!("exact_operand_c{}_v1", usage.capacity),
                    knobs: None,
                },
                usage,
                Nvfp4Global::Absent,
                "exact_gather_v1",
                "exact_quant_v1",
            )
            .unwrap()
            .capability()
            .unwrap()
        })
        .collect();
    let document = CapabilityDocument {
        version: 1,
        operations,
    };
    CapabilityInventory::new(document.clone(), document.entries(), "0".repeat(64)).unwrap()
}
