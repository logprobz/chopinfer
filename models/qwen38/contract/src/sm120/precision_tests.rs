//! The SM120 generated projection offers behind Qwen's precision ports
//! (`crate::precision`): mixed requests, layout copies and the decode
//! experts' packed ring instances.

use crate::contract::*;
use crate::precision::*;
use crate::sm120::generated;
use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::DecodeOrder;
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::precision::{Compute, capabilities};
use kernels_sm120::{moe, row::precision as row};
use std::collections::BTreeMap;

#[test]
fn geometry_outputs_and_encoder_kernels_follow_the_catalog() {
    for geometry in crate::geometry::GEOMETRIES {
        crate::encoder::declarations_for(geometry).unwrap();
        assert!(!crate::encoder::kernels_for(geometry).unwrap().is_empty());
        for ranks in [Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(geometry, ranks);
            for layer in 0..shape.layers() {
                let output = DenseRole::roles_for(shape, layer).unwrap()[4];
                let expected = if shape.is_full(layer) {
                    shape.attn_value
                } else {
                    shape.linear_value
                };
                assert_eq!(
                    output.logical_geometry(shape).unwrap().k,
                    u64::from(expected)
                );
                for usage in uses()
                    .unwrap()
                    .into_iter()
                    .filter(|u| u.capacity == 1 || u.phase == ExecutionPhase::Prefill)
                {
                    output.legacy_capability(shape, usage).unwrap();
                }
            }
        }
        capabilities_for(geometry).unwrap().validate().unwrap();
    }
}

#[test]
fn mixed_requests_cover_all_phases_with_bf16_operands_and_stable_layouts() {
    let offers = generated::inventory(Shape::TP4).unwrap().offers;
    let capabilities = capabilities(&offers).unwrap();
    let mut recipes = BTreeMap::new();
    for (offer, capability) in offers.iter().zip(capabilities.operations) {
        assert!(offer.handle().quantizer.is_none() && offer.handle().gather.is_none());
        assert_eq!(capability.arithmetic.weight_operand, DType::Bf16);
        let key = format!(
            "{:?}/{:?}/{:?}/{:?}",
            capability.kind, capability.geometry, capability.weight, capability.epilogues
        );
        let recipe = (capability.layout.clone(), capability.arithmetic.clone());
        if let Some(previous) = recipes.insert(key, recipe.clone()) {
            assert_eq!(previous, recipe);
        }
        if let Compute::Grouped(operation) = &offer.handle().compute {
            assert_eq!(
                operation.query.k,
                if operation.projection == moe::Projection::Down {
                    640
                } else {
                    2560
                }
            );
            assert_eq!(operation.knobs.split_k_capacity, 1);
            assert_eq!(
                capability.arithmetic.decode_order,
                DecodeOrder::ExactLocalThenGlobal
            );
        }
    }
    for capacity in [64, 128, 256] {
        assert_eq!(
            projection_capacity(OperationUse {
                phase: ExecutionPhase::Decode,
                capacity,
                graph: true
            })
            .unwrap(),
            PREFILL_TOKENS
        );
    }
    assert!(
        projection_capacity(OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: 3,
            graph: true
        })
        .is_err()
    );
}

#[test]
fn mixed_layout_copies_preserve_skinny_and_attention_output_rows() {
    // Seven layouts at every projection capacity, all one launch-shape
    // entry that takes its layout at launch.
    let copies = generated::inventory(Shape::TP4).unwrap().copies;
    assert_eq!(copies.len(), 7 * (DECODE_ENTRY_BUCKETS.len() + 1));
    let layouts: std::collections::BTreeSet<_> = copies
        .iter()
        .map(|copy| format!("{:?}", copy.layout()))
        .collect();
    assert_eq!(layouts.len(), 7);
    let symbols: std::collections::BTreeSet<_> = copies
        .iter()
        .map(|copy| copy.stage().entry().symbol.as_str())
        .collect();
    assert_eq!(symbols.len(), 1);
    let decode = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 1,
        graph: true,
    };
    let copy = |id| {
        generated::publish(Shape::TP4, id, WeightFormat::Mxfp8, decode)
            .unwrap()
            .copy
            .clone()
            .unwrap()
    };
    let shapes = |copy: &row::LayoutCopy| format!("{:?}", &copy.arguments(1, 2, 3).unwrap().0[3..]);
    let expected = |offset| {
        format!(
            "{:?}",
            [12, 16, 1, 0, 24, 1, offset].map(kernel_core::contract::Argument::U32)
        )
    };
    let (a, b) = (copy("linear.a"), copy("linear.b"));
    assert_eq!(shapes(&a), expected(0));
    assert_eq!(shapes(&b), expected(12));
    assert_eq!(
        generated::publish(Shape::TP4, "ngram", WeightFormat::Nvfp4, uses().unwrap()[0])
            .unwrap()
            .capability
            .arithmetic
            .decode_order,
        DecodeOrder::ExactWeightToBf16
    );
}

/// Decode expert entries are the kernel-lab packed ring instances
/// (`moe::template::packed_lab_instances`) under the served symbols, so
/// kernel-lab's bitwise comparison with the packed engine covers them;
/// prefill keeps the packed engine.
#[test]
fn decode_expert_entries_are_the_lab_packed_ring_instances() {
    let lab = super::moe::packed_lab_instances();
    let mut matched = 0;
    for usage in uses().unwrap() {
        for (id, projection) in [
            ("main.experts.gate-up", moe::Projection::GateUp),
            ("main.experts.down", moe::Projection::Down),
        ] {
            let published = generated::publish(Shape::TP4, id, WeightFormat::Nvfp4, usage).unwrap();
            let offer = &published.offer;
            let Compute::Grouped(operation) = &offer.handle().compute else {
                panic!("grouped expert offer");
            };
            let capacity = projection_capacity(usage).unwrap();
            assert_eq!(operation.packed_ring(), capacity <= DECODE_ENTRY_ROWS);
            if !operation.packed_ring() {
                continue;
            }
            let instance = lab
                .iter()
                .find(|i| {
                    i.template.query.weight == WeightFormat::Nvfp4
                        && i.template.projection == projection
                        && i.template.tokens_capacity == capacity
                })
                .expect("lab instance");
            assert_eq!(operation.query, instance.template.query);
            assert_eq!(operation.knobs, instance.template.knobs);
            assert_eq!(operation.packed_contract, instance.template.packed_contract);
            let served = offer.handle().entries().unwrap();
            let lab_spec = moe::cubin_spec(
                &instance.template_cubin,
                std::slice::from_ref(&instance.template),
            )
            .unwrap();
            assert_eq!(served.len(), 1);
            assert_eq!(served[0].body, lab_spec.entries[0].body);
            assert_eq!(served[0].params, lab_spec.entries[0].params);
            assert_eq!(served[0].max_threads, lab_spec.entries[0].max_threads);
            matched += 1;
        }
    }
    // Decode and verify at six buckets, two projections.
    assert_eq!(matched, 2 * 2 * DECODE_ENTRY_BUCKETS.len());
}
