use kernel_core::contract::{ActivationFormat, DType, Epilogues, ScaleMode, Target, WeightFormat};
use kernel_core::precision::{
    ArithmeticSpec, CodeEncoding, DecodeOrder, ExecutionPhase, NibbleOrder, OperationKind,
    OperationSpec, OperationUse, Rounding, TensorSpan, WeightLayout,
};
use model_compiler_sdk::quantization::import::{
    ActivationRecipe, EncodingRecipe, ExportMetadata, OperandRole, ROUTED_EXPERT_ROLE,
    companion_names, import_operand,
};
use model_compiler_sdk::quantization::policy::{
    Binding, ImplementationIdentity, ImplementationProvider, WeightReuse,
};
use model_compiler_sdk::quantization::{ActivationPolicy, resolve_operation};
use serde_json::json;
use std::cell::RefCell;
use std::collections::BTreeMap;

fn metadata() -> ExportMetadata {
    let mut metadata = ExportMetadata::default();
    metadata.add_document("hf_quant_config.json", &serde_json::to_vec(&json!({
        "producer": {"name": "modelopt", "version": "fixture"},
        "quantization": {"quant_algo": "MIXED_PRECISION", "group_size": 16,
            "kv_cache_quant_algo": null, "exclude_modules": ["plain*"],
            "quantized_layers": {"engine.experts": {"quant_algo": "NVFP4", "group_size": 16}}}
    })).unwrap()).unwrap();
    metadata
}

fn tensor(name: &str, dtype: DType, shape: &[u64]) -> TensorSpan {
    TensorSpan {
        tensor: name.into(),
        dtype,
        shape: shape.into(),
        offset: 0,
        byte_length: shape.iter().product::<u64>() * kernel_core::precision::dtype_bits(dtype) / 8,
    }
}

fn inputs() -> (OperandRole, BTreeMap<String, TensorSpan>) {
    let role = OperandRole {
        operation: "layer.a".into(),
        weight: "engine.experts.0.up.weight".into(),
        targets: vec!["engine.experts.0.up".into()],
        logical_shape: vec![2, 32],
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let tensors = [
        tensor(&role.weight, DType::U8, &[2, 16]),
        tensor("engine.experts.0.up.weight_scale", DType::Fp8E4M3, &[2, 2]),
        tensor("engine.experts.0.up.weight_scale_2", DType::F32, &[]),
        tensor("engine.experts.0.up.input_scale", DType::F32, &[1]),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    (role, tensors)
}

fn operation() -> OperationSpec {
    let (role, tensors) = inputs();
    let imported = import_operand(&metadata(), &role, &tensors).unwrap();
    OperationSpec {
        id: role.operation,
        kind: OperationKind::Dense,
        target: Target::Sm120f,
        geometry: None,
        weights: vec![imported.weight],
        activations: imported.activation.into_iter().collect(),
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Fp4E2M1,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: DecodeOrder::NativeBlockScaledMmaThenGlobals,
            rounding: Rounding::NearestEven,
            numerical_contract: "dense-nvfp4-source-v1".into(),
        },
        epilogues: Epilogues::default(),
        uses: vec![
            OperationUse {
                phase: ExecutionPhase::Decode,
                capacity: 1,
                graph: false,
            },
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 64,
                graph: false,
            },
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 1,
                graph: true,
            },
        ],
    }
}

#[derive(Default)]
struct Provider {
    missing: Option<OperationUse>,
    bad_bf16: bool,
    shared_input_only: bool,
    calls: RefCell<Vec<(ActivationFormat, OperationUse)>>,
}
impl ImplementationProvider for Provider {
    type Handle = (ActivationFormat, OperationUse);
    fn bind(
        &self,
        op: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Self::Handle>, String> {
        let activation = &op.activations[0];
        self.calls.borrow_mut().push((activation.format, usage));
        if self.shared_input_only && op.activations.windows(2).any(|a| a[0] != a[1]) {
            return Err(
                "fused NVFP4 implementation requires identical gate/up activation recipes".into(),
            );
        }
        if !activation.is_bf16() && self.missing == Some(usage) {
            return Err("missing supplied static-NVFP4 implementation at this capacity".into());
        }
        let mut binding = ImplementationIdentity {
            implementation: "compiled-entry-v1".into(),
            dependency: "native-source-and-lock-v1".into(),
            weight_layouts: vec![WeightLayout::row_major(); op.weights.len()],
            activations: op.activations.clone(),
            arithmetic: op.arithmetic.clone(),
            stages: vec!["compute".into()],
        };
        if self.bad_bf16 && activation.is_bf16() {
            binding.activations[0].compute = DType::Fp4E2M1;
        }
        Ok(Binding {
            identity: binding,
            handle: (activation.format, usage),
        })
    }
}

#[test]
fn bf16_reuse_records_a_complete_scale_policy_and_rejects_native_mma_tampering() {
    struct ScaleProvider(Provider);
    impl ImplementationProvider for ScaleProvider {
        type Handle = (ActivationFormat, OperationUse);
        fn bind(
            &self,
            op: &OperationSpec,
            usage: OperationUse,
        ) -> Result<Binding<Self::Handle>, String> {
            if !op.activations[0].is_bf16()
                || op.arithmetic.decode_order != DecodeOrder::CodeThenScale
            {
                return Err("only the code-first BF16 implementation is supplied".into());
            }
            self.0.bind(op, usage)
        }
    }
    let source = operation();
    let provider = ScaleProvider(Provider::default());
    let prepared = resolve_operation(&source, ActivationPolicy::Auto, &provider).unwrap();
    let selected = prepared.selection();
    assert_eq!(
        selected.source.arithmetic.decode_order,
        DecodeOrder::NativeBlockScaledMmaThenGlobals
    );
    assert_eq!(
        selected.effective.arithmetic.decode_order,
        DecodeOrder::CodeThenScale
    );
    assert_eq!(selected.bindings.len(), source.uses.len());
    selected.validate().unwrap();
    let mut tampered = selected.clone();
    tampered.effective.arithmetic.decode_order = DecodeOrder::NativeBlockScaledMmaThenGlobals;
    for binding in &mut tampered.bindings {
        binding.implementation.arithmetic = tampered.effective.arithmetic.clone();
    }
    assert!(
        tampered
            .validate()
            .unwrap_err()
            .to_string()
            .contains("BF16 scale policy")
    );
    assert!(resolve_operation(&source, ActivationPolicy::Checkpoint, &provider).is_err());
}

#[test]
fn nested_and_flat_export_schemas_normalize_without_a_model_name() {
    let mut metadata = metadata();
    metadata.add_document("quantization_config", &serde_json::to_vec(&json!({
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
        "config_groups": {"arbitrary-group-name": {"targets": ["engine.experts"],
            "weights": {"dynamic": false,"num_bits": 4,"type":"float","group_size":16},
            "input_activations": {"dynamic":false,"num_bits":4,"type":"float","group_size":16}}},
        "quantized_layers": {"encoder.linear": {"quant_algo":"MXFP8","group_size":32},
            "draft.proj": {"quant_algo":"W4A16_NVFP4","group_size":16}}
    })).unwrap()).unwrap();
    assert_eq!(
        metadata
            .recipe(&["engine.experts.19.up".into()], None)
            .unwrap(),
        EncodingRecipe {
            weight: WeightFormat::Nvfp4,
            activation: Some(ActivationRecipe::Nvfp4Static)
        }
    );
    assert_eq!(
        metadata
            .recipe(&["encoder.linear".into()], None)
            .unwrap()
            .weight,
        WeightFormat::Mxfp8
    );
    assert_eq!(
        metadata
            .recipe(&["draft.proj".into()], None)
            .unwrap()
            .activation,
        Some(ActivationRecipe::Bf16)
    );
    assert_eq!(
        metadata.recipe(&["plain.router".into()], None).unwrap(),
        EncodingRecipe::bf16()
    );
    assert_eq!(metadata.documents.len(), 2);
    assert!(
        metadata
            .recipe(&["engine.expertsx".into()], None)
            .unwrap()
            .weight
            == WeightFormat::Bf16
    );
}

#[test]
fn fp8_aliases_defer_to_explicit_activation_recipe() {
    let mut metadata = ExportMetadata::default();
    metadata.add_document("config", br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",
      "config_groups":{"x":{"weights":{"num_bits":8,"dynamic":false,"type":"float","group_size":128},
      "input_activations":{"num_bits":8,"dynamic":true,"type":"float","group_size":128},"targets":["draft"]}},
      "quantized_layers":{"draft":{"quant_algo":"FP8_PB_WO","group_size":128}}}"#).unwrap();
    metadata.add_document("hf", br#"{"producer":{"name":"modelopt","version":"fixture"},"quantization":{
      "quant_algo":"MIXED_PRECISION","quantized_layers":{"draft":{"quant_algo":"FP8_BLOCK_SCALES","group_size":128}}}}"#).unwrap();
    assert_eq!(
        metadata
            .recipe(&["draft.experts.0.up".into()], None)
            .unwrap(),
        EncodingRecipe {
            weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            activation: Some(ActivationRecipe::Fp8Dynamic)
        }
    );
}

#[test]
fn hf_block_fp8_preserves_legacy_exclusions_and_dynamic_declaration() {
    let mut metadata = ExportMetadata::default();
    metadata.add_document("config", br#"{"quant_method":"fp8","activation_scheme":"dynamic","weight_per_tensor":false,
      "act_per_tensor":false,"weight_block_size":[128,128],"modules_to_not_convert":["router","dense.*"]}"#).unwrap();
    assert_eq!(
        metadata
            .recipe(&["expert.gate".into()], None)
            .unwrap()
            .activation,
        Some(ActivationRecipe::Fp8Dynamic)
    );
    assert_eq!(
        metadata.recipe(&["dense.qkv".into()], None).unwrap(),
        EncodingRecipe::bf16()
    );
    assert_eq!(
        metadata.recipe(&["router".into()], None).unwrap(),
        EncodingRecipe::bf16()
    );
}

#[test]
fn hf_fp8_dialect_stores_routed_experts_as_declared_and_keeps_exclusions() {
    const PLAIN: &str = r#"{"quant_method":"fp8","activation_scheme":"dynamic","fmt":"e4m3",
      "weight_block_size":[128,128],"ignored_layers":["layers.0.attn.out","layers.1.moe.experts.7"]"#;
    const STORAGE: &str = r#","store_dtype":"mxfp4","mxfp4_block_size":32"#;
    let document = format!("{PLAIN}{STORAGE}}}");
    let mut metadata = ExportMetadata::default();
    metadata
        .add_document("config", document.as_bytes())
        .unwrap();
    assert_eq!(metadata.expert_storage, Some(WeightFormat::Mxfp4));
    let block = EncodingRecipe {
        weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        activation: Some(ActivationRecipe::Fp8Dynamic),
    };
    let stored = EncodingRecipe {
        weight: WeightFormat::Mxfp4,
        activation: Some(ActivationRecipe::Fp8Dynamic),
    };
    let expert = |index: u32| {
        vec![
            format!("layers.1.moe.experts.{index}.up"),
            ROUTED_EXPERT_ROLE.to_owned(),
        ]
    };
    assert_eq!(
        metadata
            .recipe(&["layers.0.attn.qkv".into()], None)
            .unwrap(),
        block
    );
    assert_eq!(
        metadata
            .recipe(&["layers.0.attn.out".into()], None)
            .unwrap(),
        EncodingRecipe::bf16()
    );
    assert_eq!(metadata.recipe(&expert(3), None).unwrap(), stored);
    // Exclusions and family declarations keep their own recipes, and the
    // role marker alone names no module.
    assert_eq!(
        metadata.recipe(&expert(7), None).unwrap(),
        EncodingRecipe::bf16()
    );
    assert_eq!(
        metadata
            .recipe(&expert(3), Some(EncodingRecipe::bf16()))
            .unwrap(),
        EncodingRecipe::bf16()
    );
    assert!(metadata.recipe(&[ROUTED_EXPERT_ROLE.into()], None).is_err());

    let role = OperandRole {
        operation: "layers.1.moe".into(),
        weight: "layers.1.moe.experts.3.up.weight".into(),
        targets: expert(3),
        logical_shape: vec![4, 64],
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let tensors: BTreeMap<_, _> = [
        tensor(&role.weight, DType::U8, &[4, 32]),
        tensor("layers.1.moe.experts.3.up.weight_scale", DType::U8, &[4, 2]),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let imported = import_operand(&metadata, &role, &tensors).unwrap();
    assert_eq!(imported.weight.format, WeightFormat::Mxfp4);
    assert_eq!(
        imported.weight.encoding,
        CodeEncoding::E2M1(NibbleOrder::EvenLow)
    );
    let scale = imported.weight.local_scale.as_ref().unwrap();
    assert_eq!(scale.tensor.dtype, DType::Ue8M0);
    assert_eq!((scale.groups[1].axis, scale.groups[1].size), (1, 32));
    assert_eq!(
        imported.activation.unwrap().format,
        ActivationFormat::Fp8Group128(ScaleMode::ExactF32)
    );
    assert_eq!(imported.source_tensors.len(), 2);
    // Without the storage declaration the same expert is block FP8, and
    // its MXFP4 companions do not satisfy that encoding.
    let mut plain = ExportMetadata::default();
    plain
        .add_document("config", format!("{PLAIN}}}").as_bytes())
        .unwrap();
    assert_eq!(plain.expert_storage, None);
    assert_eq!(plain.recipe(&expert(3), None).unwrap(), block);
    assert!(import_operand(&plain, &role, &tensors).is_err());

    // Every other spelling of these fields is refused without partial rules.
    for (field, value) in [
        (r#""fmt":"e4m3""#, r#""fmt":"e5m2""#),
        (r#""store_dtype":"mxfp4""#, r#""store_dtype":"nvfp4""#),
        (r#""mxfp4_block_size":32"#, r#""mxfp4_block_size":16"#),
        (r#","mxfp4_block_size":32"#, ""),
        (
            r#""fmt":"e4m3""#,
            r#""fmt":"e4m3","weight_per_tensor":true"#,
        ),
    ] {
        let mut refused = ExportMetadata::default();
        assert!(
            refused
                .add_document("config", document.replace(field, value).as_bytes())
                .is_err(),
            "{value}"
        );
        assert_eq!(refused, ExportMetadata::default());
    }
    let mut mixed = metadata.clone();
    let before = mixed.clone();
    assert!(
        mixed
            .add_document(
                "hf",
                br#"{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16,
          "store_dtype":"mxfp4","mxfp4_block_size":32}"#
            )
            .is_err()
    );
    assert_eq!(mixed, before);
}

#[test]
fn hf_additional_conversion_targets_preserve_defaults_and_role_precision() {
    let mut metadata = ExportMetadata::default();
    metadata
        .add_document(
            "config",
            br#"{"quant_method":"fp8","activation_scheme":"dynamic","weight_per_tensor":false,
      "act_per_tensor":false,"weight_block_size":[128,128],"modules_to_not_convert":["router"],
      "modules_to_convert":["dictionary.table"]}"#,
        )
        .unwrap();
    assert_eq!(
        metadata.recipe(&["expert.up".into()], None).unwrap().weight,
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32)
    );
    let table_recipe = EncodingRecipe {
        weight: WeightFormat::Fp8PerTensor,
        activation: Some(ActivationRecipe::Bf16),
    };
    assert_eq!(
        metadata
            .recipe(&["dictionary.table.0".into()], Some(table_recipe))
            .unwrap(),
        table_recipe
    );
    assert!(
        metadata
            .recipe(&["dictionary.table.0".into()], Some(EncodingRecipe::bf16()))
            .unwrap_err()
            .to_string()
            .contains("modules_to_convert")
    );
    metadata.exclusions.push("dictionary".into());
    assert!(
        metadata
            .recipe(&["dictionary.table.0".into()], Some(table_recipe))
            .unwrap_err()
            .to_string()
            .contains("both quantized and excluded")
    );
    let restored: ExportMetadata =
        serde_json::from_slice(&serde_json::to_vec(&metadata).unwrap()).unwrap();
    assert_eq!(restored.additional_targets, vec!["dictionary.table"]);
    let mut unsupported = ExportMetadata::default();
    assert!(
        unsupported
            .add_document(
                "mixed",
                br#"{"quant_method":"modelopt","quant_algo":"NVFP4",
      "group_size":16,"modules_to_convert":["dictionary.table"]}"#
            )
            .is_err()
    );
}

#[test]
fn companion_names_are_the_tensors_the_importer_reads() {
    let (role, tensors) = inputs();
    let imported = import_operand(&metadata(), &role, &tensors).unwrap();
    let names = companion_names(imported.weight.format, &role.weight, None).unwrap();
    assert_eq!(
        names.local,
        [imported.weight.local_scale.unwrap().tensor.tensor]
    );
    assert_eq!(
        names.global,
        Some(imported.weight.global_scale.unwrap().tensor.tensor)
    );
    let block = companion_names(
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        "a.b.weight",
        None,
    )
    .unwrap();
    assert_eq!(block.local, ["a.b.weight_scale_inv", "a.b.weight_scale"]);
    assert_eq!(block.global, None);
    let strips = companion_names(
        WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32),
        "a.b.weight",
        None,
    )
    .unwrap();
    assert_eq!(strips, block);
    let shared = companion_names(WeightFormat::Nvfp4, "a.b.weight", Some("a.c")).unwrap();
    assert_eq!(shared.local, ["a.b.weight_scale"]);
    assert_eq!(shared.global.as_deref(), Some("a.c.weight_scale_2"));
    assert_eq!(
        companion_names(WeightFormat::Bf16, "a.weight", None).unwrap(),
        Default::default()
    );
    assert!(companion_names(WeightFormat::Bf16, "a.bias", None).is_err());
}

#[test]
fn scalar_companions_are_required_and_their_sources_are_retained() {
    let (role, mut tensors) = inputs();
    // Shard ownership belongs to the inventory; this importer does not infer a
    // companion's file from the weight's file or insist on matching offsets.
    tensors
        .get_mut("engine.experts.0.up.input_scale")
        .unwrap()
        .offset = 9876;
    let imported = import_operand(&metadata(), &role, &tensors).unwrap();
    assert_eq!(imported.weight.codes.shape, [2, 16]);
    assert_eq!(imported.weight.logical_shape, [2, 32]);
    assert!(
        imported
            .source_tensors
            .contains(&"engine.experts.0.up.input_scale".into())
    );
    tensors.remove("engine.experts.0.up.input_scale");
    let error = import_operand(&metadata(), &role, &tensors)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("operation layer.a"));
    assert!(error.contains("missing companion tensor engine.experts.0.up.input_scale"));
}

#[test]
fn imported_recipes_describe_explicit_bf16_input_quantization_and_uniform_banks() {
    use kernel_core::precision::{Calibration, ScaleScope};
    let (mut role, mut tensors) = inputs();
    role.logical_shape = vec![4, 2, 32];
    let weight = tensors.get_mut(&role.weight).unwrap();
    weight.shape.insert(0, 4);
    weight.byte_length *= 4;
    let local = tensors.get_mut("engine.experts.0.up.weight_scale").unwrap();
    local.shape.insert(0, 4);
    local.byte_length *= 4;
    let imported = import_operand(&metadata(), &role, &tensors).unwrap();
    assert_eq!(
        imported.weight.global_scale.unwrap().scope,
        ScaleScope::Projection
    );
    let activation = imported.activation.unwrap();
    assert_eq!(activation.storage, DType::Bf16);
    assert_eq!(activation.compute, DType::Fp4E2M1);
    let Calibration::Static(scale) = activation.quantizer.unwrap().calibration else {
        panic!("static export")
    };
    assert_eq!(scale.scope, ScaleScope::Projection);
    for name in [
        "engine.experts.0.up.weight_scale_2",
        "engine.experts.0.up.input_scale",
    ] {
        let span = tensors.get_mut(name).unwrap();
        span.shape = vec![4];
        span.byte_length = 16;
    }
    let per_expert = import_operand(&metadata(), &role, &tensors).unwrap();
    assert_eq!(
        per_expert.weight.global_scale.unwrap().scope,
        ScaleScope::Expert
    );
    let Calibration::Static(scale) = per_expert
        .activation
        .unwrap()
        .quantizer
        .unwrap()
        .calibration
    else {
        panic!("static export")
    };
    assert_eq!(scale.scope, ScaleScope::Expert);
}

#[test]
fn hf_fp8_block_scale_source_dtype_remains_bf16_until_exact_packing() {
    let mut metadata = ExportMetadata::default();
    metadata.add_document("config",br#"{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[128,128],"weight_per_tensor":false,"act_per_tensor":false}"#).unwrap();
    let role = OperandRole {
        operation: "project".into(),
        weight: "projection.weight".into(),
        targets: vec!["projection".into()],
        logical_shape: vec![128, 256],
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let tensors = [
        tensor("projection.weight", DType::Fp8E4M3, &[128, 256]),
        tensor("projection.weight_scale_inv", DType::Bf16, &[1, 2]),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let imported = import_operand(&metadata, &role, &tensors).unwrap();
    assert_eq!(
        imported.weight.local_scale.unwrap().tensor.dtype,
        DType::Bf16
    );
    assert_eq!(imported.activation.unwrap().storage, DType::Bf16);
}

#[test]
fn separate_expert_tensors_form_a_compact_bank_without_losing_calibration() {
    use kernel_core::precision::{Calibration, ScaleScope};
    use model_compiler_sdk::quantization::{bank::concatenate_experts, payload::read_operand};
    let mut members = Vec::new();
    let mut bytes = BTreeMap::new();
    for expert in 0..4 {
        let (mut role, tensors) = inputs();
        let tensors: BTreeMap<_, _> = tensors
            .into_values()
            .map(|mut t| {
                t.tensor = t
                    .tensor
                    .replacen("experts.0", &format!("experts.{expert}"), 1);
                (t.tensor.clone(), t)
            })
            .collect();
        role.weight = role
            .weight
            .replacen("experts.0", &format!("experts.{expert}"), 1);
        role.targets = vec![role.weight.strip_suffix(".weight").unwrap().into()];
        for t in tensors.values() {
            let content = match t.dtype {
                DType::U8 => vec![0x21 + expert as u8; t.byte_length as usize],
                DType::Fp8E4M3 => vec![0x38 + expert as u8; t.byte_length as usize],
                DType::F32 => (if t.tensor.ends_with("input_scale") {
                    0.001 * (expert + 1) as f32
                } else {
                    0.125 * (expert + 1) as f32
                })
                .to_le_bytes()
                .to_vec(),
                _ => unreachable!(),
            };
            bytes.insert(t.tensor.clone(), content);
        }
        members.push(import_operand(&metadata(), &role, &tensors).unwrap());
    }
    let bank = concatenate_experts("logical.bank", &members).unwrap();
    assert_eq!(bank.operand.weight.logical_shape, [4, 2, 32]);
    assert_eq!(
        bank.operand.weight.global_scale.as_ref().unwrap().scope,
        ScaleScope::Expert
    );
    assert_eq!(bank.operand.source_tensors.len(), 16);
    assert_eq!(bank.assemblies.len(), 4);
    let Calibration::Static(scale) = &bank
        .operand
        .activation
        .as_ref()
        .unwrap()
        .quantizer
        .as_ref()
        .unwrap()
        .calibration
    else {
        panic!("static recipe")
    };
    assert_eq!(scale.scope, ScaleScope::Expert);
    assert_eq!(scale.tensor.shape, [4]);
    let loaded = read_operand(bank.operand.clone(), |span| {
        bank.read_span(span, |part| Ok(bytes[&part.tensor].clone()))
    })
    .unwrap();
    for (expert, member) in members.iter().enumerate() {
        let expected =
            read_operand(member.clone(), |span| Ok(bytes[&span.tensor].clone())).unwrap();
        let matrix = loaded.matrix(expert).unwrap();
        let source = expected.matrix(0).unwrap();
        assert_eq!(matrix.codes, source.codes);
        assert_eq!(matrix.local.unwrap().bytes, source.local.unwrap().bytes);
        assert_eq!(matrix.global.unwrap().bytes, source.global.unwrap().bytes);
    }
    let mut mixed = members.clone();
    mixed[1].weight.encoding =
        kernel_core::precision::CodeEncoding::E2M1(kernel_core::precision::NibbleOrder::EvenHigh);
    assert!(
        concatenate_experts("logical.bank", &mixed)
            .err()
            .unwrap()
            .to_string()
            .contains("expert 1")
    );
    let mut changed = members.clone();
    changed[1].activation = Some(kernel_core::precision::ActivationSpec::bf16());
    assert!(concatenate_experts("logical.bank", &changed).is_err());
    let mut corrupt = bank.clone();
    corrupt.assemblies[0].parts[0].byte_length = u64::MAX;
    let requested = corrupt.assemblies[0].tensor.clone();
    assert!(
        corrupt
            .read_span(&requested, |_| panic!(
                "refuse invalid parts before reading"
            ))
            .is_err()
    );
    // A shared source span can remain scalar without replacing values by an amax.
    let common = members[0].weight.global_scale.clone();
    for member in &mut members {
        member.weight.global_scale = common.clone();
    }
    let shared = concatenate_experts("logical.bank", &members).unwrap();
    assert_eq!(shared.operand.weight.global_scale, common);
    assert_eq!(shared.assemblies.len(), 3);
}

#[test]
fn unknown_transforms_conflicts_and_duplicate_targets_fail_transactionally() {
    for document in [
        r#"{"quant_algo":"NVFP4","group_size":16,"pre_quant_scale":"must-not-drop"}"#,
        r#"{"quant_algo":"MIXED_PRECISION","quantized_layers":{"x":{"quant_algo":"NVFP4","group_size":16},"x":{"quant_algo":"MXFP8","group_size":32}}}"#,
        r#"{"quant_algo":"MIXED_PRECISION","quantized_layers":{"engine.experts":{"quant_algo":"MXFP8","group_size":32}}}"#,
        r#"{"quant_algo":"NVFP4","group_size":32}"#,
    ] {
        let mut metadata = metadata();
        let before = metadata.clone();
        assert!(metadata.add_document("bad", document.as_bytes()).is_err());
        assert_eq!(metadata, before);
    }
    let (role, mut tensors) = inputs();
    tensors.insert(
        "engine.experts.0.up.pre_quant_scale".into(),
        tensor("engine.experts.0.up.pre_quant_scale", DType::F32, &[32]),
    );
    assert!(
        import_operand(&metadata(), &role, &tensors)
            .err()
            .unwrap()
            .to_string()
            .contains("uninterpreted execution companion")
    );
}

#[test]
fn auto_keeps_checkpoint_recipe_only_when_every_use_is_available() {
    let op = operation();
    let provider = Provider::default();
    let prepared = resolve_operation(&op, ActivationPolicy::Auto, &provider).unwrap();
    assert!(prepared.selection().reuse.is_none());
    assert_eq!(provider.calls.borrow().len(), 3);
    for &usage in &op.uses {
        assert_eq!(prepared.handle(usage).unwrap().0, ActivationFormat::Nvfp4);
    }
    prepared.selection().validate().unwrap();
    prepared
        .selection()
        .require_policy(ActivationPolicy::Checkpoint)
        .unwrap();
    assert!(
        prepared
            .selection()
            .require_policy(ActivationPolicy::Bf16)
            .is_err()
    );
}

#[test]
fn any_missing_bucket_including_prefill_graph_falls_back_as_one_operation() {
    let op = operation();
    for &missing in &op.uses {
        let provider = Provider {
            missing: Some(missing),
            ..Provider::default()
        };
        let prepared = resolve_operation(&op, ActivationPolicy::Auto, &provider).unwrap();
        assert_eq!(
            prepared.selection().reuse,
            Some(WeightReuse::PlainNvfp4Bf16V1)
        );
        assert_eq!(prepared.selection().source, op);
        assert_eq!(provider.calls.borrow().len(), 6);
        for &usage in &op.uses {
            assert_eq!(prepared.handle(usage).unwrap().0, ActivationFormat::Bf16);
        }
        assert_eq!(prepared.selection().fallback_reasons.len(), 1);
        let document = kernel_core::precision::execution::PrecisionDocument {
            version: kernel_core::precision::PRECISION_VERSION,
            operations: vec![prepared.selection().clone()],
        };
        let report = model_compiler_sdk::quantization::identity::report(&document).unwrap();
        assert_eq!(report.len(), 1);
        assert!(report[0].contains(&prepared.selection().fallback_reasons[0]));
        prepared.selection().validate().unwrap();
        assert!(
            prepared
                .selection()
                .require_policy(ActivationPolicy::Checkpoint)
                .is_err()
        );
        let error = resolve_operation(&op, ActivationPolicy::Checkpoint, &provider)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("engine.experts.0.up.weight"));
        assert!(error.contains(&format!(
            "capacity {} graph={}",
            missing.capacity, missing.graph
        )));
    }
}

#[test]
fn explicit_bf16_skips_source_binding_and_refuses_hidden_activation_conversion() {
    assert_eq!(
        ActivationPolicy::from_precedence(Some("bf16"), Some("invalid")).unwrap(),
        ActivationPolicy::Bf16
    );
    assert_eq!(
        ActivationPolicy::from_precedence(None, Some("checkpoint")).unwrap(),
        ActivationPolicy::Checkpoint
    );
    assert_eq!(
        ActivationPolicy::from_precedence(None, None).unwrap(),
        ActivationPolicy::Auto
    );
    assert!(ActivationPolicy::from_precedence(None, Some("FP4")).is_err());
    let provider = Provider::default();
    let op = operation();
    let prepared = resolve_operation(&op, ActivationPolicy::Bf16, &provider).unwrap();
    assert!(
        provider
            .calls
            .borrow()
            .iter()
            .all(|(a, _)| *a == ActivationFormat::Bf16)
    );
    prepared.selection().validate().unwrap();
    let bad = Provider {
        bad_bf16: true,
        ..Provider::default()
    };
    let error = resolve_operation(&op, ActivationPolicy::Bf16, &bad)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("changes the activation or numerical contract"));
}

#[test]
fn wire_validation_refuses_tampering_and_missing_bindings() {
    let prepared =
        resolve_operation(&operation(), ActivationPolicy::Auto, &Provider::default()).unwrap();
    let source = prepared.selection();
    let mut bad = source.clone();
    bad.bindings.pop();
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.bindings[0].implementation.activations[0].compute = DType::Bf16;
    assert!(bad.validate().is_err());
    let mut bad = source.clone();
    bad.effective.weights[0].codes.offset += 8;
    assert!(bad.validate().is_err());
    let json = serde_json::to_vec(source).unwrap();
    let parsed: kernel_core::precision::execution::PrecisionSelection =
        serde_json::from_slice(&json).unwrap();
    parsed.validate().unwrap();
    assert_eq!(source, &parsed);
}

#[test]
fn different_gate_up_calibrations_are_retained_and_only_reused_through_bf16() {
    let mut op = operation();
    let mut up = op.weights[0].clone();
    up.codes.tensor = "engine.experts.0.gate.weight".into();
    op.weights.push(up);
    let mut input = op.activations[0].clone();
    let kernel_core::precision::Calibration::Static(scale) =
        &mut input.quantizer.as_mut().unwrap().calibration
    else {
        panic!("fixture calibration");
    };
    scale.tensor.tensor = "engine.experts.0.gate.input_scale".into();
    op.activations.push(input);
    let provider = Provider {
        shared_input_only: true,
        ..Provider::default()
    };
    let error = resolve_operation(&op, ActivationPolicy::Checkpoint, &provider)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("identical gate/up activation recipes"));
    let prepared = resolve_operation(&op, ActivationPolicy::Auto, &provider).unwrap();
    assert_eq!(prepared.selection().source.activations, op.activations);
    assert_eq!(prepared.selection().effective.activations.len(), 2);
    assert!(
        prepared
            .selection()
            .effective
            .activations
            .iter()
            .all(|a| a.is_bf16())
    );
    prepared.selection().validate().unwrap();
}

#[test]
fn payload_validation_preserves_calibrated_globals_and_refuses_invalid_codes() {
    use model_compiler_sdk::quantization::payload::read_operand;
    let (role, tensors) = inputs();
    let imported = import_operand(&metadata(), &role, &tensors).unwrap();
    let read = |span: &TensorSpan| {
        Ok(match span.tensor.as_str() {
            "engine.experts.0.up.weight" => vec![0x32; 32],
            "engine.experts.0.up.weight_scale" => vec![0x38; 4],
            "engine.experts.0.up.weight_scale_2" => 0.5_f32.to_le_bytes().to_vec(),
            "engine.experts.0.up.input_scale" => 0.001_f32.to_le_bytes().to_vec(),
            _ => panic!("unexpected source"),
        })
    };
    let loaded = read_operand(imported.clone(), read).unwrap();
    assert_eq!(
        loaded.payloads()["engine.experts.0.up.input_scale"],
        0.001_f32.to_le_bytes()
    );
    assert_eq!(
        loaded
            .matrix(0)
            .unwrap()
            .bf16_operand(0, 0, DecodeOrder::ScaleThenCode)
            .unwrap(),
        0x3f00
    );
    assert!(loaded.matrix(1).is_err());
    for code in [0x80, 0x7f, 0xff] {
        let error = read_operand(imported.clone(), |span| {
            if span.tensor.ends_with(".weight_scale") {
                Ok(vec![code; 4])
            } else {
                read(span)
            }
        })
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("E4M3 scale"), "{error}");
    }
    let error = read_operand(imported, |span| {
        if span.tensor.ends_with(".input_scale") {
            Ok(0_f32.to_le_bytes().to_vec())
        } else {
            read(span)
        }
    })
    .err()
    .unwrap()
    .to_string();
    assert!(error.contains("input_scale") && error.contains("positive"));
}

#[test]
fn mx_and_packed_lookup_interpretation_reuses_the_same_importer() {
    let mut metadata = ExportMetadata::default();
    metadata.add_document("mixed", br#"{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION",
      "quantized_layers":{"project":{"quant_algo":"MXFP8","group_size":32},"table":{"quant_algo":"W4A16_NVFP4","group_size":16}}}"#).unwrap();
    let role = OperandRole {
        operation: "arbitrary-projection".into(),
        weight: "project.weight".into(),
        targets: vec!["project".into()],
        logical_shape: vec![2, 32],
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let tensors: BTreeMap<_, _> = [
        tensor("project.weight", DType::Fp8E4M3, &[2, 32]),
        tensor("project.weight_scale", DType::U8, &[2, 1]),
        tensor("table.shard_0.weight", DType::U8, &[3, 80]),
        tensor("table.shard_0.weight_scale", DType::Fp8E4M3, &[3, 10]),
        tensor("table.weight_scale_2", DType::F32, &[1]),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let imported = import_operand(&metadata, &role, &tensors).unwrap();
    assert_eq!(
        imported.weight.local_scale.unwrap().tensor.dtype,
        DType::Ue8M0
    );
    assert_eq!(imported.activation.unwrap().format, ActivationFormat::Mxfp8);
    let table = OperandRole {
        operation: "lookup".into(),
        weight: "table.shard_0.weight".into(),
        targets: vec!["table.shard_0".into()],
        logical_shape: vec![3, 160],
        lookup: true,
        shared_scale_stem: Some("table".into()),
        declared: None,
    };
    let imported = import_operand(&metadata, &table, &tensors).unwrap();
    assert!(imported.activation.is_none());
    assert_eq!(
        imported.weight.global_scale.unwrap().tensor.tensor,
        "table.weight_scale_2"
    );
}

#[test]
fn identities_separate_source_packing_policy_and_implementation_dependencies() {
    use model_compiler_sdk::quantization::identity::{
        IdentityInputs, PlacementIdentity, identities,
    };
    let op = operation();
    let provider = Provider {
        missing: Some(op.uses[0]),
        ..Provider::default()
    };
    let auto = resolve_operation(&op, ActivationPolicy::Auto, &provider).unwrap();
    let bf16 = resolve_operation(&op, ActivationPolicy::Bf16, &provider).unwrap();
    let (_, tensors) = inputs();
    let mut contents: BTreeMap<_, _> = tensors
        .keys()
        .map(|name| (name.clone(), "a".repeat(64)))
        .collect();
    let mut metadata = BTreeMap::from([("config".into(), "b".repeat(64))]);
    let mut placements = vec![PlacementIdentity {
        rank: 0,
        source_origin: vec![0, 0],
        logical_shape: vec![2, 32],
        physical_shape: vec![16, 32],
        layout: WeightLayout::row_major(),
        placement: "device".into(),
    }];
    let key = |selection,
               contents: &BTreeMap<String, String>,
               metadata: &BTreeMap<String, String>,
               placements: &[PlacementIdentity],
               native| {
        identities(
            selection,
            IdentityInputs {
                tensor_digests: contents,
                metadata_digests: metadata,
                placements,
                availability_digest: native,
            },
        )
        .unwrap()
    };
    let a = key(
        auto.selection(),
        &contents,
        &metadata,
        &placements,
        "native-v1",
    );
    let b = key(
        bf16.selection(),
        &contents,
        &metadata,
        &placements,
        "native-v1",
    );
    assert_eq!(a.interpretation, b.interpretation);
    assert_eq!(a.packed_operands, b.packed_operands);
    assert_ne!(a.execution, b.execution);
    let mut old = auto.selection().clone();
    old.effective.arithmetic.decode_order = DecodeOrder::CodeThenScale;
    for binding in &mut old.bindings {
        binding.implementation.arithmetic.decode_order = DecodeOrder::CodeThenScale;
    }
    let old_key = key(&old, &contents, &metadata, &placements, "native-v1");
    assert_eq!(a.packed_operands, old_key.packed_operands);
    assert_ne!(a.execution, old_key.execution);
    assert_eq!(
        a,
        key(
            auto.selection(),
            &contents,
            &metadata,
            &placements,
            "native-v1"
        )
    );
    contents.insert("engine.experts.0.up.input_scale".into(), "c".repeat(64));
    let calibrated = key(
        auto.selection(),
        &contents,
        &metadata,
        &placements,
        "native-v1",
    );
    assert_ne!(a.interpretation, calibrated.interpretation);
    assert_eq!(a.packed_operands, calibrated.packed_operands);
    assert_ne!(a.execution, calibrated.execution);
    contents.insert("engine.experts.0.up.weight_scale_2".into(), "d".repeat(64));
    let weight_scale = key(
        auto.selection(),
        &contents,
        &metadata,
        &placements,
        "native-v1",
    );
    assert_ne!(calibrated.packed_operands, weight_scale.packed_operands);
    let native = key(
        auto.selection(),
        &contents,
        &metadata,
        &placements,
        "native-v2",
    );
    assert_eq!(weight_scale.packed_operands, native.packed_operands);
    assert_ne!(weight_scale.execution, native.execution);
    placements[0].layout = WeightLayout {
        id: "other-consumption-layout".into(),
        version: 2,
    };
    assert_ne!(
        native.packed_operands,
        key(
            auto.selection(),
            &contents,
            &metadata,
            &placements,
            "native-v2"
        )
        .packed_operands
    );
    metadata.insert("config".into(), "e".repeat(64));
    assert_ne!(
        native.interpretation,
        key(
            auto.selection(),
            &contents,
            &metadata,
            &placements,
            "native-v2"
        )
        .interpretation
    );
    contents.remove("engine.experts.0.up.input_scale");
    assert!(
        identities(
            auto.selection(),
            IdentityInputs {
                tensor_digests: &contents,
                metadata_digests: &metadata,
                placements: &placements,
                availability_digest: "native-v1"
            }
        )
        .is_err()
    );
}

#[test]
fn bounded_precision_component_rejects_duplicate_operations_and_schema_changes() {
    use kernel_core::precision::execution::PrecisionDocument;
    use model_compiler_sdk::quantization::identity::{decode_component, encode_component};
    let op = resolve_operation(&operation(), ActivationPolicy::Bf16, &Provider::default()).unwrap();
    let mut document = PrecisionDocument {
        version: 1,
        operations: vec![op.selection().clone()],
    };
    let encoded = encode_component(&document).unwrap();
    assert_eq!(decode_component(&encoded).unwrap(), document);
    let root =
        std::env::temp_dir().join(format!("precision-component-read-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let path = root.join("component");
    std::fs::write(&path, &encoded).unwrap();
    let read = model_compiler_sdk::quantization::identity::read_component;
    assert_eq!(read(&path).unwrap(), document);
    std::os::unix::fs::symlink(&path, root.join("link")).unwrap();
    assert!(read(&root.join("link")).is_err());
    assert!(read(&root).is_err());
    std::fs::File::create(&path)
        .unwrap()
        .set_len(kernel_core::precision::component::MAX_COMPONENT_BYTES as u64 + 1)
        .unwrap();
    assert!(read(&path).unwrap_err().to_string().contains("byte bound"));
    std::fs::remove_dir_all(root).unwrap();
    assert!(encoded.len() <= kernel_core::precision::component::MAX_COMPONENT_BYTES);
    for declared in [0, u32::MAX] {
        let mut invalid = encoded.clone();
        invalid[8..12].copy_from_slice(&declared.to_le_bytes());
        assert!(decode_component(&invalid).is_err());
    }
    let mut trailing = encoded.clone();
    trailing.push(0);
    let payload = (trailing.len() - 16) as u32;
    trailing[12..16].copy_from_slice(&payload.to_le_bytes());
    assert!(decode_component(&trailing).is_err());
    let mut corrupt = encoded.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(decode_component(&corrupt).is_err());
    assert!(decode_component(&encoded[..encoded.len() - 1]).is_err());
    document.operations.push(op.selection().clone());
    assert!(
        encode_component(&document)
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    document.operations.pop();
    document.version = 2;
    assert!(encode_component(&document).is_err());
    assert!(decode_component(br#"{"version":1,"version":2,"operations":[]}"#).is_err());
    assert!(decode_component(br#"{"version":1,"operations":[],"unknown_policy":"fp4"}"#).is_err());
}

#[test]
fn authenticated_package_constraints_preserve_the_fixed_recipe() {
    use kernel_core::precision::{component, execution::PrecisionDocument};
    use model_compiler_sdk::{
        BundleDraftV1, ComponentContentV1, ComponentDraftV1, StagedContentV1,
    };
    use runtime_bundle::*;
    let root = std::env::temp_dir().join(format!("precision-package-{}", std::process::id()));
    std::fs::create_dir(&root).unwrap();
    let id = |name| BundleOpaqueId::new(name).unwrap();
    for policy in [ActivationPolicy::Checkpoint, ActivationPolicy::Bf16] {
        let selection = resolve_operation(&operation(), policy, &Provider::default()).unwrap();
        let document = PrecisionDocument {
            version: 1,
            operations: vec![selection.selection().clone()],
        };
        let bytes = component::encode(&document).unwrap();
        let component = ComponentDraftV1::new(
            id("precision"),
            id("chop.bundle-metadata.fixture.precision"),
            id(component::COMPONENT_FORMAT),
            BundleRelativePathV1::new("precision.bin").unwrap(),
            ComponentContentV1::Whole(StagedContentV1::bytes(bytes.clone())),
        );
        let draft = BundleDraftV1::new(
            id("precision.fixture.v1"),
            BundleTargetV1::new(
                vec![id("sm_120f")],
                vec![BundleTopologyV1::new(id("tp1"), 1).unwrap()],
            )
            .unwrap(),
            BundleResourceRequirementsV1::new(1, vec![]).unwrap(),
            BundleKvLayoutV1::new(
                id("fixture.kv"),
                64,
                64,
                1,
                64,
                1,
                1024,
                BundleRankPlacementV1::RankLocal,
                BundleStateLifetimeV1::Request,
            )
            .unwrap(),
            BundleStateLayoutV1::new(id("fixture.state"), 64, 64, BundleStateLifetimeV1::Request)
                .unwrap(),
        )
        .with_graph_buckets(vec![
            BundleGraphBucketV1::new(id("fixture.graph"), vec![id("precision")], vec![]).unwrap(),
        ])
        .with_components(vec![component]);
        let path = root.join(policy.to_string());
        let emitted = model_compiler_sdk::emit_bundle_directory_v1(&draft, &path).unwrap();
        let package = emitted.package();
        for constraint in [
            ActivationPolicy::Auto,
            ActivationPolicy::Checkpoint,
            ActivationPolicy::Bf16,
        ] {
            let result = model_compiler_sdk::quantization::package::check(package, constraint);
            assert_eq!(
                result.is_ok(),
                constraint == ActivationPolicy::Auto || constraint == policy
            );
            if let Ok(report) = result {
                assert!(!report.unwrap().is_empty());
            }
        }
        assert_eq!(std::fs::read(path.join("precision.bin")).unwrap(), bytes);
        // The package keeps the descriptor bytes it verified: a rewrite after
        // admission does not reach them, and a new verification refuses it.
        let mut corrupt = bytes;
        *corrupt.last_mut().unwrap() ^= 1;
        std::fs::write(path.join("precision.bin"), corrupt).unwrap();
        assert!(
            model_compiler_sdk::quantization::package::check(package, ActivationPolicy::Auto)
                .is_ok()
        );
        assert!(model_compiler_sdk::verify_emitted_package(&path).is_err());
    }
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn lookup_row_reads_keep_codes_scales_and_bounded_source_offsets() {
    use model_compiler_sdk::quantization::payload::{read_lookup_rows, read_operand};
    for (format, scale_dtype) in [
        (WeightFormat::Bf16, DType::Bf16),
        (WeightFormat::Fp8PerTensor, DType::Bf16),
        (WeightFormat::Fp8PerTensor, DType::F32),
        (WeightFormat::Nvfp4, DType::Fp8E4M3),
    ] {
        let code_dtype = match format {
            WeightFormat::Bf16 => DType::Bf16,
            WeightFormat::Nvfp4 => DType::U8,
            _ => DType::Fp8E4M3,
        };
        let codes = tensor(
            "table.weight",
            code_dtype,
            &[
                3,
                if format == WeightFormat::Nvfp4 {
                    16
                } else {
                    32
                },
            ],
        );
        let mut spans = BTreeMap::from([(codes.tensor.clone(), codes.clone())]);
        let mut payloads = BTreeMap::from([(
            codes.tensor.clone(),
            (0..codes.byte_length)
                .map(|i| {
                    if code_dtype == DType::Bf16 {
                        if i % 2 == 0 { 0x81 } else { 0x3f }
                    } else {
                        (i % 64) as u8
                    }
                })
                .collect::<Vec<_>>(),
        )]);
        if format != WeightFormat::Bf16 {
            let scale = tensor(
                "table.weight_scale",
                scale_dtype,
                if format == WeightFormat::Nvfp4 {
                    &[3, 2]
                } else {
                    &[1]
                },
            );
            let data = match scale_dtype {
                DType::Bf16 => 0.5_f32.to_bits().to_le_bytes()[2..].to_vec(),
                DType::F32 => 0.50003_f32.to_le_bytes().to_vec(),
                _ => vec![0x28, 0x38, 0x30, 0x40, 0x38, 0x28],
            };
            payloads.insert(scale.tensor.clone(), data);
            spans.insert(scale.tensor.clone(), scale);
        }
        if format == WeightFormat::Nvfp4 {
            let global = tensor("table.weight_scale_2", DType::F32, &[1]);
            payloads.insert(global.tensor.clone(), 0.31251_f32.to_le_bytes().to_vec());
            spans.insert(global.tensor.clone(), global);
        }
        let operand = import_operand(
            &ExportMetadata::default(),
            &OperandRole {
                operation: "lookup".into(),
                weight: codes.tensor.clone(),
                targets: vec!["table".into()],
                logical_shape: vec![3, 32],
                lookup: true,
                shared_scale_stem: None,
                declared: Some(EncodingRecipe {
                    weight: format,
                    activation: Some(ActivationRecipe::Bf16),
                }),
            },
            &spans,
        )
        .unwrap();
        let full =
            read_operand(operand.clone(), |span| Ok(payloads[&span.tensor].clone())).unwrap();
        let matrix = full.matrix(0).unwrap();
        for order in [DecodeOrder::ScaleThenCode, DecodeOrder::CodeThenScale] {
            let mut reads = vec![];
            let result = read_lookup_rows(&operand, &[2, 0, 2], order, |span| {
                reads.push(span.clone());
                Ok(payloads[&span.tensor]
                    [span.offset as usize..(span.offset + span.byte_length) as usize]
                    .to_vec())
            })
            .unwrap();
            for (actual, row) in result.iter().zip([2, 0, 2]) {
                assert_eq!(
                    *actual,
                    (0..32)
                        .map(|k| matrix.bf16_operand(row, k, order).unwrap())
                        .collect::<Vec<_>>()
                );
            }
            let code_reads: Vec<_> = reads.iter().filter(|s| s.tensor == codes.tensor).collect();
            assert_eq!(
                code_reads.iter().map(|s| s.offset).collect::<Vec<_>>(),
                [2 * codes.byte_length / 3, 0, 2 * codes.byte_length / 3]
            );
            assert!(
                code_reads
                    .iter()
                    .all(|s| s.byte_length == codes.byte_length / 3)
            );
        }
        assert!(
            read_lookup_rows(&operand, &[3], DecodeOrder::CodeThenScale, |_| panic!(
                "out of range rows must fail before reading"
            ))
            .is_err()
        );
    }
}

#[test]
fn lookup_kind_imports_native_nvfp4_weights_without_activation_calibration() {
    let (mut role, mut tensors) = inputs();
    tensors.remove("engine.experts.0.up.input_scale");
    role.lookup = true;
    let imported = import_operand(&metadata(), &role, &tensors).unwrap();
    assert_eq!(imported.weight.format, WeightFormat::Nvfp4);
    assert!(imported.activation.is_none());
    assert_eq!(imported.source_tensors.len(), 3);
    role.lookup = false;
    assert!(
        import_operand(&metadata(), &role, &tensors)
            .unwrap_err()
            .to_string()
            .contains("input_scale")
    );
}

#[test]
fn block32_fp8_dialect_reads_ue8m0_blocks_ceil_activations_and_fp4_experts() {
    const DOCUMENT: &str = r#"{"quant_method":"fp8","activation_scheme":"dynamic",
      "weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}"#;
    let mut metadata = ExportMetadata::default();
    metadata
        .add_document("config", DOCUMENT.as_bytes())
        .unwrap();
    let block = EncodingRecipe {
        weight: WeightFormat::Fp8Block32x32,
        activation: Some(ActivationRecipe::Mxfp8CeilDynamic),
    };
    assert_eq!(metadata.default, Some(block));
    assert_eq!(metadata.expert_storage, Some(WeightFormat::Mxfp4));
    let expert = vec![
        "layers.0.experts.3.up".to_owned(),
        ROUTED_EXPERT_ROLE.to_owned(),
    ];
    assert_eq!(
        metadata.recipe(&expert, None).unwrap(),
        EncodingRecipe {
            weight: WeightFormat::Mxfp4,
            ..block
        }
    );

    // A dense operand: E4M3 codes with one UE8M0 byte per 32x32 block, and
    // E4M3-per-32 activations whose scale is the ceil power of two.
    let dense = OperandRole {
        operation: "layers.0.attn".into(),
        weight: "layers.0.attn.query.weight".into(),
        targets: vec!["layers.0.attn.query".into()],
        logical_shape: vec![64, 96],
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let tensors: BTreeMap<_, _> = [
        tensor(&dense.weight, DType::Fp8E4M3, &[64, 96]),
        tensor("layers.0.attn.query.weight_scale", DType::Ue8M0, &[2, 3]),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let imported = import_operand(&metadata, &dense, &tensors).unwrap();
    assert_eq!(imported.weight.format, WeightFormat::Fp8Block32x32);
    assert_eq!(imported.weight.encoding, CodeEncoding::E4M3);
    let scale = imported.weight.local_scale.as_ref().unwrap();
    assert_eq!(scale.tensor.dtype, DType::Ue8M0);
    assert_eq!(
        scale
            .groups
            .iter()
            .map(|g| (g.axis, g.size))
            .collect::<Vec<_>>(),
        [(0, 32), (1, 32)]
    );
    let activation = imported.activation.unwrap();
    assert_eq!(activation.format, ActivationFormat::Mxfp8);
    let quantizer = activation.quantizer.unwrap();
    assert_eq!(
        quantizer.calibration,
        kernel_core::precision::Calibration::Dynamic(
            kernel_core::precision::DynamicScale::MXFP8_CEIL
        )
    );
    assert_eq!(quantizer.groups[0].size, 32);
    // A scale grid of the wrong extent is refused.
    let mut short = tensors.clone();
    short.insert(
        "layers.0.attn.query.weight_scale".into(),
        tensor("layers.0.attn.query.weight_scale", DType::Ue8M0, &[2, 2]),
    );
    assert!(import_operand(&metadata, &dense, &short).is_err());

    // A routed expert: E2M1 pairs with a UE8M0 scale per 32.
    let routed = OperandRole {
        operation: "layers.0.experts".into(),
        weight: "layers.0.experts.3.up.weight".into(),
        targets: expert,
        logical_shape: vec![4, 64],
        lookup: false,
        shared_scale_stem: None,
        declared: None,
    };
    let tensors: BTreeMap<_, _> = [
        tensor(&routed.weight, DType::U8, &[4, 32]),
        tensor("layers.0.experts.3.up.weight_scale", DType::Ue8M0, &[4, 2]),
    ]
    .into_iter()
    .map(|t| (t.tensor.clone(), t))
    .collect();
    let imported = import_operand(&metadata, &routed, &tensors).unwrap();
    assert_eq!(imported.weight.format, WeightFormat::Mxfp4);
    assert_eq!(
        imported.weight.encoding,
        CodeEncoding::E2M1(NibbleOrder::EvenLow)
    );
    assert_eq!(imported.activation.unwrap().format, ActivationFormat::Mxfp8);

    // Every other spelling is refused without partial rules, and the
    // block-128 dialect keeps refusing the block-32 fields.
    for (field, value) in [
        (r#""scale_fmt":"ue8m0""#, r#""scale_fmt":"e8m0""#),
        (r#","scale_fmt":"ue8m0""#, ""),
        (r#""expert_dtype":"fp4""#, r#""expert_dtype":"mxfp4""#),
        (
            r#""activation_scheme":"dynamic""#,
            r#""activation_scheme":"static""#,
        ),
        (
            r#""expert_dtype":"fp4""#,
            r#""expert_dtype":"fp4","store_dtype":"mxfp4","mxfp4_block_size":32"#,
        ),
        (
            r#""weight_block_size":[32,32]"#,
            r#""weight_block_size":[128,128]"#,
        ),
        (
            r#""weight_block_size":[32,32]"#,
            r#""weight_block_size":[32,64]"#,
        ),
    ] {
        let mut refused = ExportMetadata::default();
        assert!(
            refused
                .add_document("config", DOCUMENT.replace(field, value).as_bytes())
                .is_err(),
            "{value}"
        );
        assert_eq!(refused, ExportMetadata::default());
    }
    let mut nested = ExportMetadata::default();
    assert!(
        nested
            .add_document(
                "hf",
                br#"{"quantization":{"quant_algo":"MXFP8","group_size":32},"scale_fmt":"ue8m0"}"#
            )
            .is_err()
    );
}
