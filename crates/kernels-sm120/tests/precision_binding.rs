use kernel_core::contract::*;
use kernel_core::precision::execution::ImplementationProvider;
use kernel_core::precision::*;
use kernels_sm120::{gemm, moe, precision::*, row::precision as row};

fn tensor(name: &str, dtype: DType, shape: Vec<u64>) -> TensorSpan {
    let byte_length = shape.iter().product::<u64>() * dtype_bits(dtype) / 8;
    TensorSpan {
        tensor: name.into(),
        dtype,
        shape,
        offset: 0,
        byte_length,
    }
}

fn scalar(name: &str) -> ScaleSpec {
    ScaleSpec {
        tensor: tensor(name, DType::F32, vec![]),
        scope: ScaleScope::Tensor,
        convention: ScaleConvention::Multiply,
        groups: vec![],
    }
}

fn operation(kind: OperationKind, activation: ActivationFormat) -> OperationSpec {
    let lookup = kind == OperationKind::Lookup;
    let grouped = matches!(kind, OperationKind::Grouped { .. });
    let shape = if grouped {
        vec![4, 256, 256]
    } else if lookup {
        vec![37, 160]
    } else {
        vec![256, 256]
    };
    let rank = shape.len();
    let mut packed = shape.clone();
    *packed.last_mut().unwrap() /= 2;
    let mut scales = shape.clone();
    *scales.last_mut().unwrap() /= 16;
    let weight = WeightSpec {
        codes: tensor("matrix", DType::U8, packed),
        logical_shape: shape,
        format: WeightFormat::Nvfp4,
        encoding: CodeEncoding::E2M1(NibbleOrder::EvenLow),
        layout: WeightLayout::row_major(),
        local_scale: Some(ScaleSpec {
            tensor: tensor("local", DType::Fp8E4M3, scales),
            scope: ScaleScope::Block,
            convention: ScaleConvention::Multiply,
            groups: vec![
                GroupAxis {
                    axis: (rank - 2) as u8,
                    size: 1,
                    origin: 0,
                },
                GroupAxis {
                    axis: (rank - 1) as u8,
                    size: 16,
                    origin: 0,
                },
            ],
        }),
        global_scale: Some(scalar("weight_global")),
        transforms: vec![],
    };
    let input = if activation == ActivationFormat::Bf16 {
        ActivationSpec::bf16()
    } else {
        ActivationSpec {
            format: activation,
            storage: DType::Bf16,
            compute: DType::Fp4E2M1,
            quantizer: Some(QuantizerSpec {
                calibration: Calibration::Static(scalar("input_global")),
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
    };
    OperationSpec {
        id: "unrelated_layer".into(),
        kind,
        target: Target::Sm120f,
        geometry: None,
        weights: if grouped {
            vec![weight.clone(), weight]
        } else {
            vec![weight]
        },
        activations: if lookup {
            vec![]
        } else if grouped {
            vec![input.clone(), input]
        } else {
            vec![input]
        },
        arithmetic: ArithmeticSpec {
            weight_operand: if lookup || activation == ActivationFormat::Bf16 {
                DType::Bf16
            } else {
                DType::Fp4E2M1
            },
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: if !lookup && activation == ActivationFormat::Nvfp4 {
                DecodeOrder::NativeBlockScaledMmaThenGlobals
            } else if lookup {
                DecodeOrder::ExactWeightToBf16
            } else {
                DecodeOrder::ExactLocalThenGlobal
            },
            rounding: Rounding::NearestEven,
            numerical_contract: "source-recipe".into(),
        },
        epilogues: Epilogues {
            swiglu: grouped.then_some(SwigluColumns::ConcatenatedHalves),
            ..Epilogues::default()
        },
        uses: vec![usage()],
    }
}

fn usage() -> OperationUse {
    OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 1,
        graph: false,
    }
}

#[test]
fn refusal_reports_only_the_matching_operation_domain() {
    let offers = vec![dense(), grouped(true)];
    let available = offers
        .iter()
        .flat_map(|o| o.handle().entries().unwrap())
        .map(|e| e.symbol)
        .collect();
    let inventory = CompiledInventory::new(offers, available, "a".repeat(64)).unwrap();
    let source = operation(
        OperationKind::Grouped {
            experts: 4,
            top_k: 2,
        },
        ActivationFormat::Nvfp4,
    );
    let error = inventory.bind(&source, usage()).err().unwrap();
    assert!(
        error.contains("requested Nvfp4 activation operands"),
        "{error}"
    );
    assert!(error.contains("requires Bf16 operands"), "{error}");
    assert!(!error.contains("shape/encoding/fusion"), "{error}");
    let mut absent = source.clone();
    absent.geometry = Some(MatrixGeometry { rows: 128, k: 256 });
    let error = inventory.bind(&absent, usage()).err().unwrap();
    assert!(error.contains("N=128 K=256"), "{error}");
    assert!(error.contains("no supplied implementation"), "{error}");
}

fn dense() -> KernelOffer {
    KernelOffer::dense(
        gemm::DenseRequest {
            query: GemmQuery {
                m_capacity: 1,
                n: 256,
                k: 256,
                weight: WeightFormat::Nvfp4,
                activation: ActivationFormat::Nvfp4,
                output: DType::Bf16,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            target: Target::Sm120f,
            symbol: "consumer_dense_v1".into(),
            kernel: DenseKernel::Tiled { splits: 1 },
            knobs: None,
        },
        usage(),
        row::Nvfp4Global::Supplied,
        "consumer_quant_v1",
    )
    .unwrap()
}

fn grouped(a16: bool) -> KernelOffer {
    KernelOffer::grouped(
        moe::GroupedRequest {
            shape: moe::Shape {
                hidden: 256,
                intermediate: 256,
                experts: 4,
                top_k: 2,
            },
            projection: moe::Projection::GateUp,
            route_weight: moe::RouteWeight::AfterDown,
            swiglu_clamp: None,
            phase: moe::Phase::Decode,
            tokens_capacity: 1,
            weight: WeightFormat::Nvfp4,
            activation: if a16 {
                ActivationFormat::Bf16
            } else {
                ActivationFormat::Nvfp4
            },
            scales: moe::ScaleBindings {
                weight_globals: if a16 {
                    moe::GlobalScaleLayout::ExpertProjection
                } else {
                    moe::GlobalScaleLayout::Scalar
                },
                activation_globals: if a16 {
                    moe::GlobalScaleLayout::Absent
                } else {
                    moe::GlobalScaleLayout::Scalar
                },
                shared_gate_up_activation: true,
            },
            output_quantization: None,
            swiglu_columns: SwigluColumns::ConcatenatedHalves,
            target: Target::Sm120f,
            symbol: "consumer_grouped_v1".into(),
            knobs: None,
        },
        usage(),
        if a16 {
            row::Nvfp4Global::Absent
        } else {
            row::Nvfp4Global::Supplied
        },
        "consumer_gather_v1",
        "consumer_quant_v1",
    )
    .unwrap()
}

fn inventory(offer: KernelOffer, missing: Option<&str>) -> CompiledInventory {
    let available = offer
        .handle()
        .entries()
        .unwrap()
        .into_iter()
        .map(|e| e.symbol)
        .filter(|s| Some(s.as_str()) != missing)
        .collect();
    CompiledInventory::new(vec![offer], available, "a".repeat(64)).unwrap()
}

#[test]
fn binds_supplied_stages_and_refuses_missing_dependencies_without_compilation() {
    let op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    let binding = inventory(dense(), None).bind(&op, usage()).unwrap();
    assert_eq!(
        binding.identity.stages,
        vec!["consumer_quant_v1", "consumer_dense_v1"]
    );
    assert!(
        binding
            .identity
            .arithmetic
            .numerical_contract
            .starts_with("sm120.dense.packed.gemm.w-nvfp4")
    );
    assert_eq!(binding.identity.dependency, "a".repeat(64));
    assert_eq!(binding.handle.quantizer.unwrap().scratch_bytes(), 0);
    let error = inventory(dense(), Some("consumer_quant_v1"))
        .bind(&op, usage())
        .err()
        .unwrap();
    assert!(error.contains("missing supplied compiled entries: consumer_quant_v1"));
    let mut other = usage();
    other.graph = true;
    assert!(inventory(dense(), None).bind(&op, other).is_err());
}

#[test]
fn grouped_a4_requires_uniform_globals_and_shared_projection_calibration() {
    let kind = OperationKind::Grouped {
        experts: 4,
        top_k: 2,
    };
    let mut op = operation(kind, ActivationFormat::Nvfp4);
    let inv = inventory(grouped(false), None);
    let binding = inv.bind(&op, usage()).unwrap();
    assert_eq!(
        binding.identity.stages,
        vec![
            "consumer_gather_v1",
            "consumer_quant_v1",
            "consumer_grouped_v1"
        ]
    );
    let mut scale = scalar("expert_globals");
    scale.tensor = tensor("expert_globals", DType::F32, vec![4]);
    scale.scope = ScaleScope::Expert;
    op.weights[0].global_scale = Some(scale.clone());
    op.weights[1].global_scale = Some(scale);
    assert!(
        inv.bind(&op, usage())
            .err()
            .unwrap()
            .contains("bank-uniform")
    );
    let mut bf16 = op.clone();
    bf16.activations.fill(ActivationSpec::bf16());
    bf16.arithmetic.weight_operand = DType::Bf16;
    bf16.arithmetic.decode_order = DecodeOrder::ExactLocalThenGlobal;
    let a16 = inventory(grouped(true), None).bind(&bf16, usage()).unwrap();
    assert!(a16.handle.gather.is_none() && a16.handle.quantizer.is_none());
    op = operation(kind, ActivationFormat::Nvfp4);
    op.activations[1].quantizer.as_mut().unwrap().calibration =
        Calibration::Static(scalar("different_input"));
    assert!(
        inv.bind(&op, usage())
            .err()
            .unwrap()
            .contains("identical gate/up")
    );
}

#[test]
fn dense_binding_checks_geometry_transforms_scale_origins_and_decode_order() {
    let inv = inventory(dense(), None);
    let mut op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    op.arithmetic.decode_order = DecodeOrder::CodeThenScale;
    assert!(
        inv.bind(&op, usage())
            .err()
            .unwrap()
            .contains("NativeBlockScaledMmaThenGlobals")
    );
    op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    op.weights[0]
        .transforms
        .push(RequiredTransform::Unsupported {
            field: "rotation".into(),
            value: "hadamard".into(),
        });
    assert!(inv.bind(&op, usage()).err().unwrap().contains("transform"));
    op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    op.weights[0].encoding = CodeEncoding::E2M1(NibbleOrder::EvenHigh);
    assert!(inv.bind(&op, usage()).err().unwrap().contains("even-low"));
    op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    op.arithmetic.weight_operand = DType::Bf16;
    assert!(
        inv.bind(&op, usage())
            .err()
            .unwrap()
            .contains("weight operands")
    );
    op = operation(OperationKind::Dense, ActivationFormat::Bf16);
    assert!(inv.bind(&op, usage()).err().unwrap().contains("activation"));
}

#[test]
fn dense_capabilities_distinguish_scale_stages_and_fused_row_layouts() {
    for (weight, activation, capacity, swiglu, order) in [
        (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ActivationFormat::Bf16,
            1,
            None,
            DecodeOrder::BlockScaleAfterMma,
        ),
        (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ActivationFormat::Bf16,
            64,
            None,
            DecodeOrder::BlockScaleAfterMma,
        ),
        (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ActivationFormat::Bf16,
            64,
            Some(SwigluColumns::ConcatenatedHalves),
            DecodeOrder::BlockScaleAfterMma,
        ),
        (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ActivationFormat::Bf16,
            64,
            Some(SwigluColumns::Interleaved),
            DecodeOrder::ScaleThenCode,
        ),
        (
            WeightFormat::Fp8PerTensor,
            ActivationFormat::Bf16,
            64,
            None,
            DecodeOrder::ScaleThenCode,
        ),
        (
            WeightFormat::Fp8PerChannel,
            ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
            64,
            None,
            DecodeOrder::ScaleProductAfterBlockMma,
        ),
        (
            WeightFormat::Mxfp8,
            ActivationFormat::Bf16,
            64,
            None,
            DecodeOrder::ExactScaledOperand,
        ),
        (
            WeightFormat::Mxfp4,
            ActivationFormat::Bf16,
            64,
            None,
            DecodeOrder::ExactScaledOperand,
        ),
        (
            WeightFormat::Mxfp8,
            ActivationFormat::Mxfp8,
            64,
            None,
            DecodeOrder::NativeBlockScaledMma,
        ),
        (
            WeightFormat::Nvfp4,
            ActivationFormat::Nvfp4,
            64,
            None,
            DecodeOrder::NativeBlockScaledMmaThenGlobals,
        ),
    ] {
        let offer = KernelOffer::dense(
            gemm::DenseRequest {
                query: GemmQuery {
                    m_capacity: capacity,
                    n: 256,
                    k: 256,
                    weight,
                    activation,
                    output: DType::Bf16,
                    epilogues: Epilogues {
                        swiglu,
                        ..Default::default()
                    },
                    grouped: None,
                },
                target: Target::Sm120f,
                symbol: "scale_order_v1".into(),
                // The one-row block FP8 entry takes the fragment decode.
                kernel: if capacity <= 32 {
                    DenseKernel::FragmentDecode { splits: 1 }
                } else {
                    DenseKernel::Tiled { splits: 1 }
                },
                knobs: None,
            },
            OperationUse {
                capacity,
                ..usage()
            },
            if activation == ActivationFormat::Nvfp4 {
                row::Nvfp4Global::Supplied
            } else {
                row::Nvfp4Global::Absent
            },
            "scale_quantizer_v1",
        )
        .unwrap();
        assert_eq!(
            offer.capability().unwrap().arithmetic.decode_order,
            order,
            "{weight:?}/{activation:?}/{swiglu:?}"
        );
    }
    assert_eq!(
        grouped(false).capability().unwrap().arithmetic.decode_order,
        DecodeOrder::NativeBlockScaledMmaThenGlobals
    );
    assert_eq!(
        grouped(true).capability().unwrap().arithmetic.decode_order,
        DecodeOrder::ExactLocalThenGlobal
    );
}

#[test]
fn lookup_binding_has_integer_inputs_and_one_row_decoder() {
    let offer = KernelOffer::lookup(
        row::LookupRequest {
            vocab: 37,
            width: 160,
            heads: 1,
            capacity: 1,
            weight: WeightFormat::Nvfp4,
            global_scale: Some(DType::F32),
            ids: row::LookupIds::I32,
            shapes: row::ShapeBinding::Compiled,
        },
        usage(),
        "consumer_lookup_v1",
    )
    .unwrap();
    let op = operation(OperationKind::Lookup, ActivationFormat::Bf16);
    let bound = inventory(offer, None).bind(&op, usage()).unwrap();
    assert!(bound.identity.activations.is_empty());
    assert_eq!(bound.identity.stages, vec!["consumer_lookup_v1"]);
    assert!(matches!(bound.handle.compute, Compute::Lookup(_)));
    assert!(bound.handle.gather.is_none() && bound.handle.quantizer.is_none());
}

#[test]
fn source_geometry_and_prepared_rank_geometry_are_distinct() {
    let inv = inventory(dense(), None);
    let mut op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    let weight = &mut op.weights[0];
    weight.logical_shape[0] *= 2;
    weight.codes.shape[0] *= 2;
    weight.codes.byte_length *= 2;
    let scale = &mut weight.local_scale.as_mut().unwrap().tensor;
    scale.shape[0] *= 2;
    scale.byte_length *= 2;
    assert!(inv.bind(&op, usage()).is_err());
    op.geometry = Some(MatrixGeometry { rows: 256, k: 256 });
    assert!(inv.bind(&op, usage()).is_ok());
    op.geometry.as_mut().unwrap().k = 128;
    assert!(inv.bind(&op, usage()).is_err());
    op.geometry.as_mut().unwrap().rows = 0;
    assert!(op.validate().is_err());
}

#[test]
fn supplied_operations_form_one_audited_consumer_selection() {
    use kernel_build::{BuildContext, BuildOptions, KernelSelection, Toolchain};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };
    let dense = dense();
    let lookup = KernelOffer::lookup(
        row::LookupRequest {
            vocab: 37,
            width: 160,
            heads: 1,
            capacity: 1,
            weight: WeightFormat::Nvfp4,
            global_scale: Some(DType::F32),
            ids: row::LookupIds::I32,
            shapes: row::ShapeBinding::Compiled,
        },
        usage(),
        "consumer_lookup_v1",
    )
    .unwrap();
    let offers = vec![dense.clone(), dense, grouped(true), lookup];
    let selection = KernelSelection::new(
        BuildContext::new("independent@1", "consumer", "precision.v1"),
        selected_cubins("independent", &offers).unwrap(),
    )
    .unwrap();
    let entries: usize = selection.specs().iter().map(|s| s.entries.len()).sum();
    assert_eq!(entries, 4); // duplicate dense use shares compute and quantizer entries
    assert_eq!(
        selection
            .declarations()
            .values()
            .map(Vec::len)
            .sum::<usize>(),
        entries
    );
    assert_eq!(selection.specs().len(), 3);
    assert!(selected_cubins("independent", &[grouped(true), grouped(false)]).is_err());
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let out = std::env::temp_dir().join(format!(
        "precision-selection-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&out).unwrap();
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let built = selection
        .build(
            &toolchain,
            &out,
            &out.join("kernels.lock"),
            BuildOptions {
                lock_update: true,
                emit_cargo_directives: false,
                profile_line_info: false,
            },
        )
        .unwrap();
    assert_eq!(
        selection.audited_contract_identity(&built).unwrap(),
        selection.contract_identity().unwrap()
    );
    assert_eq!(
        selection.audited_dependency_identity(&built).unwrap().len(),
        64
    );
    let publication = publish_capabilities(&offers, &selection, &built).unwrap();
    let document: kernel_core::precision::capability::CapabilityDocument =
        serde_json::from_slice(&publication.bytes).unwrap();
    let adapted = publish_capability_document(&document, &selection, &built).unwrap();
    assert_eq!(publication.bytes, adapted.bytes);
    assert_eq!(publication.binding_dependency, adapted.binding_dependency);
    let mut undeclared = document.clone();
    undeclared.operations[0]
        .stages
        .push("absent_stage_v1".into());
    assert!(
        publish_capability_document(&undeclared, &selection, &built)
            .err()
            .unwrap()
            .contains("absent_stage_v1 has no declaration")
    );
    assert_eq!(publication.sha256, kernel_build::digest(&publication.bytes));
    let portable = kernel_core::precision::capability::CapabilityInventory::new(
        document.clone(),
        document.entries(),
        publication.binding_dependency.clone(),
    )
    .unwrap();
    let concrete = CompiledInventory::new(
        offers.clone(),
        document.entries(),
        publication.binding_dependency,
    )
    .unwrap();
    for op in [
        operation(OperationKind::Dense, ActivationFormat::Nvfp4),
        operation(
            OperationKind::Grouped {
                experts: 4,
                top_k: 2,
            },
            ActivationFormat::Bf16,
        ),
        operation(OperationKind::Lookup, ActivationFormat::Bf16),
    ] {
        assert_eq!(
            portable.bind(&op, usage()).unwrap().identity,
            concrete.bind(&op, usage()).unwrap().identity
        );
    }
    assert!(publish_capabilities(&[grouped(false)], &selection, &built).is_err());
    let mut missing = built.clone();
    missing[0].entries.pop();
    assert!(selection.audited_contract_identity(&missing).is_err());
    fs::remove_dir_all(&out).unwrap();
}

#[test]
fn composition_filter_retains_inventory_handles_and_cannot_add_support() {
    use kernel_core::precision::capability::CapabilityInventory;
    let mut document = capabilities(&[dense(), dense()]).unwrap();
    document.operations[1].implementation = "another-projection-port.v1".into();
    let inventory =
        CapabilityInventory::new(document.clone(), document.entries(), "a".repeat(64)).unwrap();
    let op = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    assert_eq!(inventory.bind(&op, usage()).unwrap().handle, 0);
    let binding = inventory
        .bind_matching(&op, usage(), |c| {
            c.implementation == "another-projection-port.v1"
        })
        .unwrap();
    assert_eq!(binding.handle, 1);
    assert_eq!(binding.identity.dependency, "a".repeat(64));
    assert!(inventory.bind_matching(&op, usage(), |_| false).is_err());
    let absent = CapabilityInventory::new(document, Default::default(), "a".repeat(64)).unwrap();
    assert!(
        absent
            .bind_matching(&op, usage(), |_| true)
            .err()
            .unwrap()
            .contains("missing")
    );
}

/// An embedding-gather row entry offers a BF16 table lookup of its
/// vocabulary and width, as itself.
#[test]
fn embedding_rows_offer_their_own_lookup_capability() {
    let entries = kernels_sm120::row::own_catalog();
    let entry = entries
        .iter()
        .find(|e| e.symbol == "chop_row_embedding_full_std_v1")
        .unwrap();
    let usage = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: 8192,
        graph: false,
    };
    let capability = catalog_lookup_capability(entry, usage).unwrap();
    assert_eq!(capability.kind, OperationKind::Lookup);
    assert_eq!(
        (capability.geometry.rows, capability.geometry.k),
        (152_576, 4096)
    );
    assert_eq!(capability.weight, WeightFormat::Bf16);
    assert!(capability.activation.is_none() && capability.local_scale_dtypes.is_empty());
    assert_eq!(capability.stages, std::slice::from_ref(&entry.symbol));
    let other = entries
        .iter()
        .find(|e| e.symbol.starts_with("chop_row_swiglu_"))
        .unwrap();
    assert!(catalog_lookup_capability(other, usage).is_err());
}

fn generated_offer(
    source: &OperationSpec,
    adoptions: &[generated::Adoption],
) -> Result<generated::GeneratedOffer, String> {
    let implementation = match source.kind {
        OperationKind::Dense => generated::Implementation::Dense {
            kernel: DenseKernel::Tiled { splits: 1 },
            capacity: 1,
            knobs: None,
        },
        OperationKind::Grouped { .. } => generated::Implementation::Grouped {
            phase: moe::Phase::Decode,
            capacity: 1,
            knobs: None,
            scales: generated::scale_bindings(source)?,
            route_weight: moe::RouteWeight::AfterDown,
            swiglu_clamp: None,
        },
        OperationKind::Lookup => generated::Implementation::Lookup {
            heads: 1,
            ids: row::LookupIds::I32,
            scale_dtype: Some(DType::F32),
            shapes: row::ShapeBinding::Compiled,
        },
    };
    generated::offer(
        generated::Request {
            operation: source,
            format: format::FormatId::Nvfp4,
            usage: source.uses[0],
            implementation,
        },
        adoptions,
    )
}

#[test]
fn generated_names_share_rank_local_bodies_and_ignore_source_names() {
    for kind in [
        OperationKind::Dense,
        OperationKind::Grouped {
            experts: 4,
            top_k: 2,
        },
        OperationKind::Lookup,
    ] {
        let source = operation(kind, ActivationFormat::Bf16);
        let first = generated_offer(&source, &[]).unwrap();
        let mut renamed = source.clone();
        renamed.id = "another_projection".into();
        for weight in &mut renamed.weights {
            weight.codes.tensor = "another_tensor".into();
        }
        let second = generated_offer(&renamed, &[]).unwrap();
        assert_eq!(first.identities, second.identities);
        assert_eq!(
            first.offer.capability().unwrap(),
            second.offer.capability().unwrap()
        );
        assert!(!first.identities.is_empty());
        assert!(
            first
                .identities
                .iter()
                .all(|i| i.generated.starts_with("chop_proj_nvfp4_"))
        );
    }
}

#[test]
fn generated_adoptions_preserve_code_and_capabilities_except_names() {
    for (activation, dynamic) in [
        (ActivationFormat::Bf16, false),
        (ActivationFormat::Nvfp4, false),
        (ActivationFormat::Nvfp4, true),
    ] {
        let mut source = operation(OperationKind::Dense, activation);
        if dynamic {
            source.activations[0]
                .quantizer
                .as_mut()
                .unwrap()
                .calibration = Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax);
        }
        let first = generated_offer(&source, &[]).unwrap();
        let aliases: Vec<_> = first
            .identities
            .iter()
            .cloned()
            .map(|mut i| {
                i.symbol = i.generated.replace("chop_proj", "chop_existing");
                i
            })
            .collect();
        let second = generated_offer(&source, &aliases).unwrap();
        let entries = second.offer.handle().entries().unwrap();
        for ((a, b), alias) in first
            .offer
            .handle()
            .entries()
            .unwrap()
            .iter()
            .zip(&entries)
            .zip(&aliases)
        {
            assert_eq!(b.symbol, alias.symbol);
            let mut expected = a.clone();
            expected.symbol.clone_from(&b.symbol);
            assert_eq!(format!("{expected:?}"), format!("{b:?}"));
        }
        let mut expected = first.offer.capability().unwrap();
        expected.stages = aliases.iter().map(|a| a.symbol.clone()).collect();
        assert_eq!(expected, second.offer.capability().unwrap());
        let mut changed = aliases.clone();
        changed[0].entry_sha256 = "0".repeat(64);
        assert!(generated_offer(&source, &changed).is_err());
        let mut duplicate = aliases.clone();
        duplicate.push(aliases[0].clone());
        assert!(generated_offer(&source, &duplicate).is_err());
    }
}

#[test]
fn generated_offers_refuse_contract_drift() {
    let mut source = operation(OperationKind::Dense, ActivationFormat::Bf16);
    source.arithmetic.decode_order = DecodeOrder::CodeThenScale;
    assert!(generated_offer(&source, &[]).is_err());
    source.arithmetic.decode_order = DecodeOrder::ExactLocalThenGlobal;
    source.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
    assert!(generated_offer(&source, &[]).is_err());
}

#[test]
fn generated_copy_adoption_preserves_exact_entry_and_footprints() {
    let source = row::CopyRows {
        row_stride: 64,
        column_stride: 1,
        column_offset: 0,
    };
    let destination = row::CopyRows {
        row_stride: 128,
        column_stride: 2,
        column_offset: 3,
    };
    let (copy, mut identity) = generated::layout_copy(
        32,
        16,
        source,
        destination,
        row::ShapeBinding::Compiled,
        &[],
    )
    .unwrap();
    identity.symbol = "chop_existing_copy_v1".into();
    let (adopted, _) = generated::layout_copy(
        32,
        16,
        source,
        destination,
        row::ShapeBinding::Compiled,
        &[identity],
    )
    .unwrap();
    assert_eq!(copy.stage().entry().body, adopted.stage().entry().body);
    assert_eq!(copy.source_bytes(), adopted.source_bytes());
    assert_eq!(copy.destination_bytes(), adopted.destination_bytes());
    assert_eq!(adopted.stage().entry().symbol, "chop_existing_copy_v1");
}

/// Capacity only sizes a row stage's grid, so uses of one copy or lookup at
/// several capacities share one identity and declaration. The body and the
/// grid rule still distinguish entries.
#[test]
fn row_stage_identities_ignore_capacity_but_not_the_grid_rule() {
    let source = row::CopyRows {
        row_stride: 64,
        column_stride: 1,
        column_offset: 0,
    };
    let destination = row::CopyRows {
        row_stride: 128,
        column_stride: 1,
        column_offset: 3,
    };
    let (one, identity) =
        generated::layout_copy(32, 1, source, destination, row::ShapeBinding::Compiled, &[])
            .unwrap();
    let (wide, same) = generated::layout_copy(
        32,
        8192,
        source,
        destination,
        row::ShapeBinding::Compiled,
        &[],
    )
    .unwrap();
    assert_eq!(identity, same);
    assert_eq!(one.stage().declaration(), wide.stage().declaration());
    assert_eq!(one.stage().plan(1).unwrap().grid, [1, 1, 1]);
    assert_eq!(wide.stage().plan(1).unwrap().grid, [8192, 1, 1]);
    let shifted = row::CopyRows {
        column_offset: 4,
        ..destination
    };
    let (_, other) =
        generated::layout_copy(32, 1, source, shifted, row::ShapeBinding::Compiled, &[]).unwrap();
    assert_ne!(other.generated, identity.generated);

    let mut lookup = operation(OperationKind::Lookup, ActivationFormat::Bf16);
    let first = generated_offer(&lookup, &[]).unwrap();
    lookup.uses[0].capacity *= 64;
    let wider = generated_offer(&lookup, &[]).unwrap();
    assert_eq!(first.identities, wider.identities);
    assert_ne!(
        first.offer.handle().row_stages()[0].plan(1).unwrap().grid,
        wider.offer.handle().row_stages()[0].plan(1).unwrap().grid
    );

    // At capacity 1 the amax and global stages plan one block each, yet
    // their grid rules (planned rows, one block) differ.
    let dynamic = row::Quantizer::new(
        256,
        1,
        OutputQuantization::Nvfp4,
        row::Nvfp4Global::TensorAmax,
        "consumer_dyn_v1",
        row::ShapeBinding::Compiled,
        row::RowResources::SM120F,
    )
    .unwrap();
    let grid = |stage: &row::RowStage| stage.declaration().requirements["grid"].clone();
    let [amax, global, _] = dynamic.stages() else {
        panic!("three tensor-amax stages");
    };
    assert_eq!(amax.plan(1).unwrap().grid, global.plan(1).unwrap().grid);
    assert_ne!(grid(amax), grid(global));
}

#[test]
fn generated_initial_adoption_requires_independent_legacy_equality() {
    let source = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    let generated = generated_offer(&source, &[]).unwrap();
    let legacy = dense();
    let aliases = generated::adopt(&generated, &legacy).unwrap();
    let adopted = generated_offer(&source, &aliases).unwrap();
    assert_eq!(
        adopted.offer.capability().unwrap(),
        legacy.capability().unwrap()
    );
    assert_eq!(
        format!("{:?}", adopted.offer.handle().entries().unwrap()),
        format!("{:?}", legacy.handle().entries().unwrap())
    );
    assert!(generated::adopt(&generated, &grouped(false)).is_err());
}

#[test]
fn generated_weight_only_plugin_refuses_native_activation() {
    let source = operation(OperationKind::Dense, ActivationFormat::Nvfp4);
    let result = generated::offer(
        generated::Request {
            operation: &source,
            format: format::FormatId::Nvfp4WeightOnly,
            usage: usage(),
            implementation: generated::Implementation::Dense {
                kernel: DenseKernel::Tiled { splits: 1 },
                capacity: 1,
                knobs: None,
            },
        },
        &[],
    );
    assert!(
        result
            .err()
            .unwrap()
            .contains("weight-only NVFP4 requires BF16")
    );
}

#[test]
fn generated_names_distinguish_reduction_schedules_and_share_equal_uses() {
    let mut source = operation(OperationKind::Dense, ActivationFormat::Bf16);
    source.weights[0].logical_shape = vec![256, 1024];
    source.weights[0].codes = tensor("matrix", DType::U8, vec![256, 512]);
    source.weights[0].local_scale.as_mut().unwrap().tensor =
        tensor("local", DType::Fp8E4M3, vec![256, 64]);
    let first = generated_offer(&source, &[]).unwrap();
    source.uses[0].phase = ExecutionPhase::Verify;
    source.uses[0].graph = true;
    let verify = generated_offer(&source, &[]).unwrap();
    assert_eq!(first.identities, verify.identities);
    assert_ne!(first.offer.usage(), verify.offer.usage());
    let split = generated::offer(
        generated::Request {
            operation: &source,
            format: format::FormatId::Nvfp4,
            usage: source.uses[0],
            implementation: generated::Implementation::Dense {
                kernel: DenseKernel::Tiled { splits: 2 },
                capacity: 1,
                knobs: None,
            },
        },
        &[],
    )
    .unwrap();
    assert_ne!(first.identities[0].generated, split.identities[0].generated);
    assert!(generated::adopt(&verify, &split.offer).is_err());
}

#[test]
fn hand_list_comparison_reconstructs_typed_domains_without_checkpoint_data() {
    let mut offers = vec![grouped(true)];
    for weight in [
        WeightFormat::Bf16,
        WeightFormat::Mxfp8,
        WeightFormat::Nvfp4,
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        WeightFormat::Fp8Rows64x128(ScaleMode::ExactF32),
    ] {
        offers.push(
            KernelOffer::dense(
                gemm::DenseRequest {
                    query: GemmQuery {
                        m_capacity: 1,
                        n: 256,
                        k: 256,
                        weight,
                        activation: ActivationFormat::Bf16,
                        output: DType::Bf16,
                        epilogues: Epilogues::default(),
                        grouped: None,
                    },
                    kernel: if matches!(
                        weight,
                        WeightFormat::Bf16
                            | WeightFormat::Fp8Block128x128(_)
                            | WeightFormat::Fp8Rows64x128(_)
                    ) {
                        DenseKernel::FragmentDecode { splits: 1 }
                    } else {
                        DenseKernel::Tiled { splits: 1 }
                    },
                    target: Target::Sm120f,
                    symbol: "retained_dense_v1".into(),
                    knobs: None,
                },
                usage(),
                row::Nvfp4Global::Absent,
                "unused_quant_v1",
            )
            .unwrap(),
        );
    }
    for weight in [WeightFormat::Bf16, WeightFormat::Nvfp4] {
        offers.push(
            KernelOffer::lookup(
                row::LookupRequest {
                    vocab: 37,
                    width: 160,
                    heads: 2,
                    capacity: 1,
                    weight,
                    global_scale: (weight == WeightFormat::Nvfp4).then_some(DType::F32),
                    ids: row::LookupIds::U64,
                    shapes: row::ShapeBinding::Compiled,
                },
                usage(),
                "retained_lookup_v1",
            )
            .unwrap(),
        );
    }
    for old in offers {
        let operation = generated::comparison::operation(&old.capability().unwrap()).unwrap();
        let fresh = generated::comparison::regenerate(&operation, &old, &[]).unwrap();
        let aliases = generated::adopt(&fresh, &old).unwrap();
        let restored = generated::comparison::regenerate(&operation, &old, &aliases).unwrap();
        assert_eq!(
            old.capability().unwrap(),
            restored.offer.capability().unwrap()
        );
        assert_eq!(
            format!("{:?}", old.handle()),
            format!("{:?}", restored.offer.handle())
        );
        // A declared domain with the format's canonical local scales
        // regenerates the same offer.
        let capability = old.capability().unwrap();
        let declared = generated::domain(generated::Domain {
            kind: capability.kind,
            target: capability.target,
            geometry: capability.geometry,
            operands: capability.operands,
            weight: capability.weight,
            local_scale: None,
            global_scale: operation.weights[0].global_scale.as_ref().map(|s| s.scope),
            activations: operation.activations.clone(),
            arithmetic: capability.arithmetic.clone(),
            epilogues: capability.epilogues,
            usage: capability.usage,
        })
        .unwrap();
        let regenerated = generated::comparison::regenerate(&declared, &old, &aliases).unwrap();
        assert_eq!(
            format!("{:?}", old.handle()),
            format!("{:?}", regenerated.offer.handle())
        );
    }
}

#[test]
fn catalog_comparison_requires_exact_stage_membership_and_operation_domain() {
    let offer = grouped(true);
    let capability = offer.capability().unwrap();
    let operation = generated::comparison::operation(&capability).unwrap();
    let specs = offer.handle().cubin_specs("retained").unwrap();
    let declarations = offer
        .handle()
        .entries()
        .unwrap()
        .into_iter()
        .map(|entry| {
            let value = generated::identity::declaration(&offer, &entry.symbol).unwrap();
            (entry.symbol, value)
        })
        .collect();
    let (actual, records) =
        generated::comparison::catalog(&operation, capability.clone(), &specs, &declarations)
            .unwrap();
    assert_eq!(actual, capability);
    assert_eq!(records.len(), actual.stages.len());
    assert_eq!(
        records,
        generated::identities(
            format::registry::by_format(capability.weight).unwrap().id(),
            &offer
        )
        .unwrap()
    );
    assert!(
        generated::comparison::catalog(&operation, capability.clone(), &specs, &Default::default())
            .is_err()
    );
    assert!(
        generated::comparison::catalog(&operation, capability.clone(), &[], &declarations).is_err()
    );
    let doubled = [specs.clone(), specs].concat();
    assert!(
        generated::comparison::catalog(&operation, capability.clone(), &doubled, &declarations)
            .is_err()
    );
    let mut other = operation;
    other.geometry = Some(MatrixGeometry { rows: 128, k: 256 });
    assert!(
        generated::comparison::catalog(&other, capability, &doubled[..1], &declarations).is_err()
    );
}

#[test]
fn adoption_history_supersedes_by_old_symbol_and_appends_in_discovery_order() {
    let source = operation(OperationKind::Dense, ActivationFormat::Bf16);
    let fresh = generated_offer(&source, &[]).unwrap();
    let mut current = fresh.identities[0].clone();
    current.symbol = "old_z_v1".into();
    let mut previous = current.clone();
    previous.generated = "chop_proj_previous_v1".into();
    previous.entry_sha256 = "0".repeat(64);
    let mut next = current.clone();
    next.symbol = "old_a_v1".into();
    let history = generated::append_adoptions(
        std::slice::from_ref(&previous),
        [current.clone(), next.clone()],
    );
    assert_eq!(history, vec![previous.clone(), current.clone(), next]);
    assert_eq!(
        generated::append_adoptions(&history, [current.clone()]),
        history
    );
    let adopted = generated_offer(&source, &[previous, current.clone()]).unwrap();
    assert_eq!(
        adopted.offer.capability().unwrap().stages,
        vec![current.symbol]
    );
}

#[test]
fn canonical_identity_ignores_json_key_and_header_order() {
    use generated::identity;
    let a: serde_json::Value = serde_json::from_str(r#"{"z":{"b":2,"a":1},"a":0}"#).unwrap();
    let b: serde_json::Value = serde_json::from_str(r#"{"a":0,"z":{"a":1,"b":2}}"#).unwrap();
    assert_eq!(identity::canonical(a), identity::canonical(b));
    let source = operation(OperationKind::Dense, ActivationFormat::Bf16);
    let offer = generated_offer(&source, &[]).unwrap().offer;
    let mut spec = offer.handle().cubin_specs("one").unwrap().remove(0);
    spec.headers.push("#include <stdint.h>".into());
    let declaration = identity::declaration(&offer, &spec.entries[0].symbol).unwrap();
    let first = identity::record(
        format::FormatId::Nvfp4,
        "dense",
        Some(source.matrix_geometry().unwrap()),
        &spec,
        &spec.entries[0],
        declaration.clone(),
    );
    spec.headers.reverse();
    let second = identity::record(
        format::FormatId::Nvfp4,
        "dense",
        Some(source.matrix_geometry().unwrap()),
        &spec,
        &spec.entries[0],
        declaration,
    );
    assert_eq!(first, second);
}

#[test]
fn retained_lookup_identity_changes_when_only_the_row_grid_changes() {
    let mut entry = kernels_sm120::row::own_catalog()
        .into_iter()
        .find(|e| {
            matches!(
                e.query,
                kernel_core::contract::RowQuery::EmbeddingGather { .. }
            )
        })
        .unwrap();
    let usage = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: 64,
        graph: false,
    };
    let specs = kernels_sm120::row::cubin_specs(&kernels_sm120::Catalogs::new(&[]).unwrap(), None);
    let identify = |entry: &kernels_sm120::row::RowInstantiation| {
        let cap = catalog_lookup_capability(entry, usage).unwrap();
        let op = generated::comparison::operation(&cap).unwrap();
        generated::comparison::catalog(
            &op,
            cap.clone(),
            &specs,
            &std::collections::BTreeMap::from([(
                entry.symbol.clone(),
                generated::identity::retained_row(entry, &cap).unwrap(),
            )]),
        )
        .unwrap()
        .1
    };
    let old = identify(&entry);
    let live = kernel_core::contract::RuntimeCounts {
        m: 64,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    assert_eq!(entry.plan(64, live).unwrap().grid[0], 64);
    entry.grid = kernels_sm120::row::Grid::One;
    assert_eq!(entry.plan(64, live).unwrap().grid[0], 1);
    assert_ne!(old, identify(&entry));
}

#[test]
fn schedule_policy_uses_portable_inputs_and_preserves_override_history() {
    use generated::schedule::{self, Override, Schedule};
    let source = operation(OperationKind::Dense, ActivationFormat::Bf16);
    let key = schedule::key(&source, source.uses[0]).unwrap();
    let old = generated_offer(&source, &[]).unwrap();
    let default = schedule::offer(&source, &key, &[], &[]).unwrap();
    assert_eq!(default.identities, old.identities);
    assert!(schedule::capture(&source, &old.offer).unwrap().is_none());
    let first = Override {
        inputs: key.clone(),
        schedule: schedule::default(&source, &key).unwrap(),
    };
    let mut tuned = first.clone();
    let Schedule::Typed {
        implementation: generated::Implementation::Dense { kernel, .. },
    } = &mut tuned.schedule
    else {
        unreachable!()
    };
    *kernel = DenseKernel::Tiled { splits: 2 };
    let history = schedule::append(std::slice::from_ref(&first), [tuned.clone()]).unwrap();
    assert_eq!(history, [first.clone(), tuned.clone()]);
    assert_eq!(
        schedule::append(&history, [tuned.clone()]).unwrap(),
        history
    );
    assert_eq!(
        schedule::resolve(&source, &key, &history).unwrap(),
        tuned.schedule
    );
    assert!(schedule::append(&[], [first, tuned]).is_err());
    let mut wrong = key.clone();
    wrong.geometry.k += 1;
    assert!(schedule::resolve(&source, &wrong, &history).is_err());
}

#[test]
fn named_schedule_overrides_require_the_declared_projection_id() {
    use generated::schedule;
    let mut source = operation(OperationKind::Dense, ActivationFormat::Bf16);
    source.id = "assembly.first".into();
    let mut inputs = schedule::key(&source, source.uses[0]).unwrap();
    assert!(
        serde_json::to_value(&inputs)
            .unwrap()
            .get("projection")
            .is_none()
    );
    inputs.projection = Some(source.id.clone());
    let tuned = schedule::Schedule::Typed {
        implementation: generated::Implementation::Dense {
            kernel: DenseKernel::Tiled { splits: 2 },
            capacity: 1,
            knobs: None,
        },
    };
    let overrides = [schedule::Override {
        inputs: inputs.clone(),
        schedule: tuned.clone(),
    }];
    assert_eq!(
        schedule::resolve(&source, &inputs, &overrides).unwrap(),
        tuned
    );
    source.id = "assembly.second".into();
    assert!(schedule::resolve(&source, &inputs, &overrides).is_err());
    inputs.projection = Some(source.id.clone());
    assert_ne!(
        schedule::resolve(&source, &inputs, &overrides).unwrap(),
        tuned
    );
}
