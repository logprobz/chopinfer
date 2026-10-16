//! The pre-port gates of mixed precision, verbatim from 3096de63d, and
//! representative documents that drive the call sites consuming the
//! declared conditions.
use super::*;
use kernel_core::contract::ScaleMode;
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{BoundUse, PrecisionDocument, PrecisionSelection};
use model_compiler_sdk::quantization::ActivationPolicy;

/// `model_qwen38_contract::prepared::legacy_text` before the port: the MTP
/// gate of every package, drafter and bundle path.
pub fn legacy_text(document: &PrecisionDocument) -> bool {
    document.operations.iter().all(|s| match s.effective.kind {
        OperationKind::Dense => s
            .effective
            .weights
            .iter()
            .all(|w| w.format == WeightFormat::Bf16),
        OperationKind::Grouped { .. } => matches!(
            s.effective.weights[0].format,
            WeightFormat::Fp8Block128x128(_)
        ),
        OperationKind::Lookup => matches!(
            s.effective.weights[0].format,
            WeightFormat::Bf16 | WeightFormat::Fp8PerTensor
        ),
    })
}

/// The MXFP8 `linear.qkv` port of layer 0, bound to its published TP4
/// composition.
fn mxfp8_qkv() -> PrecisionSelection {
    let shape = Shape::TP4;
    let usage = precision::uses().unwrap()[0];
    let capabilities =
        model_qwen38_contract::sm120::generated::document_for(shape.geometry).unwrap();
    let inventory =
        CapabilityInventory::new(capabilities.clone(), capabilities.entries(), "0".repeat(64))
            .unwrap();
    let published = model_qwen38_contract::sm120::generated::publish(
        shape,
        "linear.qkv",
        WeightFormat::Mxfp8,
        usage,
    )
    .unwrap();
    let mut operation = model_qwen38_contract::sm120::generated::domain(
        shape,
        "linear.qkv",
        WeightFormat::Mxfp8,
        usage,
    )
    .unwrap();
    operation.id = DenseRole::LinearQkv.operation_id_for(shape, 0).unwrap();
    for weight in &mut operation.weights {
        weight.layout = published.capability.layout.clone();
    }
    operation.arithmetic = published.capability.arithmetic.clone();
    let binding = inventory
        .bind_matching(&operation, usage, |c| {
            c.implementation == published.capability.implementation
        })
        .unwrap();
    PrecisionSelection {
        source: operation.clone(),
        effective: operation,
        policy: ActivationPolicy::Bf16,
        reuse: None,
        fallback_reasons: vec![],
        bindings: vec![BoundUse {
            usage,
            implementation: binding.identity,
        }],
    }
}

/// `selections` with every weight, source and effective, in `format`.
fn reformatted(
    mut selections: Vec<PrecisionSelection>,
    format: WeightFormat,
) -> Vec<PrecisionSelection> {
    for selection in &mut selections {
        for weight in selection
            .source
            .weights
            .iter_mut()
            .chain(&mut selection.effective.weights)
        {
            weight.format = format;
        }
    }
    selections
}

/// Format mixes a checkpoint or a supplied component can present, labelled,
/// with whether the document passes validation. The supported mixes are
/// bound to their compositions; the others only change formats, as a
/// foreign or future component could.
pub fn documents() -> Vec<(&'static str, PrecisionDocument, bool)> {
    let experts = routed_nvfp4_fixture(Shape::TP4, |_| {}).unwrap();
    let qkv = mxfp8_qkv();
    let mut embedding = qkv.clone();
    for operation in [&mut embedding.source, &mut embedding.effective] {
        operation.id = format!("{}.embed_tokens", model_qwen38_contract::views::MODEL);
        operation.kind = OperationKind::Lookup;
    }
    let legacy_dense = dense(
        &super::dense::tests::fixture(false),
        0,
        DenseRole::LinearQkv,
        Shape::TP4,
    )
    .unwrap()
    .resolve(ActivationPolicy::Auto, &super::dense::tests::inventory())
    .unwrap()
    .selection()
    .clone();
    let block = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
    [
        ("no operation", vec![], false),
        ("BF16 legacy dense", vec![legacy_dense], true),
        ("MXFP8 dense", vec![qkv.clone()], true),
        ("NVFP4 experts", experts.clone(), true),
        (
            "MXFP8 dense and NVFP4 experts",
            [vec![qkv.clone()], experts.clone()].concat(),
            true,
        ),
        (
            "MXFP4 experts",
            reformatted(experts.clone(), WeightFormat::Mxfp4),
            false,
        ),
        (
            "BF16 experts",
            reformatted(experts.clone(), WeightFormat::Bf16),
            false,
        ),
        ("FP8 block experts", reformatted(experts, block), false),
        (
            "FP8 block dense",
            reformatted(vec![qkv.clone()], block),
            false,
        ),
        (
            "BF16 dense",
            reformatted(vec![qkv], WeightFormat::Bf16),
            false,
        ),
        (
            "NVFP4 embedding",
            reformatted(vec![embedding.clone()], WeightFormat::Nvfp4),
            false,
        ),
        (
            "BF16 embedding",
            reformatted(vec![embedding], WeightFormat::Bf16),
            false,
        ),
    ]
    .into_iter()
    .map(|(label, operations, valid)| {
        let document = PrecisionDocument {
            version: 1,
            operations,
        };
        assert_eq!(document.validate().is_ok(), valid, "{label}");
        (label, document, valid)
    })
    .collect()
}

/// A prepared plan of a source outside the legacy layout is never refused
/// for its topology: the declaration supplies mixed compositions at TP2 and
/// TP4.
#[test]
fn mixed_placement_admits_every_supplied_topology() {
    const REFUSAL: &str = "mixed Qwen rank-image placement requires a supplied mixed composition";
    for (label, document, valid) in documents() {
        if !valid {
            continue;
        }
        for ranks in [2, 4] {
            let catalog = crate::catalog::synthetic(ranks);
            let refused = crate::placement::ImagePlan::with_precision(&catalog, None, &document)
                .err()
                .is_some_and(|e| e.to_string().starts_with(REFUSAL));
            assert!(!refused, "{label} at TP{ranks}");
        }
    }
}
