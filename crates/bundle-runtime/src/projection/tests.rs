use super::*;
use kernel_core::contract::{
    ActivationFormat, ArgumentList, DType, DenseKernel, GemmQuery, LaunchPlan, Target,
};
use kernel_core::precision::{
    ActivationSpec, CodeEncoding, ExecutionPhase, OperationKind, OperationSpec, TensorSpan,
    WeightSpec,
    execution::{ActivationPolicy, Binding, BoundUse, PrecisionSelection},
};
use kernels_sm120::{
    backend::ProjectionOp,
    gemm, moe,
    precision::{Compute, KernelOffer, OperationHandle},
    row::precision as row,
};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, KernelArgumentV1};

fn usage() -> OperationUse {
    OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 8,
        graph: true,
    }
}
fn span(slot: u32, bytes: u64) -> Span {
    Span::new(BundleDeviceBufferHandleV1::from_parts(slot, 1), 256, bytes)
}
fn buffers() -> (Input<'static>, Span, Scratch) {
    (
        Input {
            activation: span(1, 1 << 20),
            live_rows: span(2, 4),
            routing: Some(Routing {
                weights: span(3, 64),
                offsets: span(4, 20),
                sorted_rows: span(5, 128),
                padded_rows: span(6, 4),
                active_experts: span(7, 20),
            }),
            residual: None,
            row_offset: 0,
            shard_rows: 0,
            side: &[],
            scalars: &[],
        },
        span(8, 1 << 20),
        Scratch {
            weight_domain: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
            partial: span(9, 1 << 24),
            semaphores: span(10, 8192),
            unused: span(11, 4),
            temporary: None,
            gathered: Some(span(12, 1 << 20)),
            codes: Some(span(13, 1 << 20)),
            scales: Some(span(14, 1 << 20)),
            activation_global: Some(span(15, 4)),
            row_amax: None,
            output_codes: None,
            output_scales: None,
        },
    )
}

#[test]
fn projection_scratch_uses_the_shared_allocation_declaration() {
    use kernel_core::allocation::{Allocation, Traversal};
    let (_, _, scratch) = buffers();
    let mut original = vec![];
    scratch.visit(&mut |span| original.push(*span));
    let declaration = scratch.map(|span| Allocation::new(span.length + 32, span.length).unwrap());
    let mut cursor = 0;
    let bound = crate::scratch::bind(declaration, |bytes| {
        let expected = original[cursor];
        assert_eq!(bytes, expected.length + 32);
        cursor += 1;
        Ok(Span::new(expected.buffer, expected.offset, bytes))
    })
    .unwrap();
    let mut names = vec![];
    bound.visit_named(&mut |name, _| names.push(name.to_owned()));
    assert_eq!(
        names,
        [
            "partial",
            "semaphores",
            "unused",
            "gathered",
            "codes",
            "scales",
            "activation_global"
        ]
    );
    let mut writable = vec![];
    bound.visit_named(&mut |name, span| {
        if name != "unused" {
            writable.push(*span);
        }
    });
    assert_eq!(scratch_regions(&bound), writable);
    assert_eq!(writable.len() + 1, original.len());
    assert_eq!(cursor, original.len());
}
fn phase() -> Phase {
    Phase {
        usage: usage(),
        rows: 3,
        padded_rows: 16,
    }
}
fn dense(global: row::Nvfp4Global) -> KernelOffer {
    KernelOffer::dense(
        gemm::DenseRequest {
            query: GemmQuery {
                m_capacity: 8,
                n: 256,
                k: 1024,
                weight: WeightFormat::Nvfp4,
                activation: ActivationFormat::Nvfp4,
                output: DType::Bf16,
                epilogues: Default::default(),
                grouped: None,
            },
            kernel: DenseKernel::Tiled { splits: 2 },
            target: Target::Sm120f,
            symbol: "projection_dense_v1".into(),
            knobs: None,
        },
        usage(),
        global,
        "projection_quant_v1",
    )
    .unwrap()
}
fn tensor(name: &str, dtype: DType, shape: Vec<u64>) -> TensorSpan {
    TensorSpan {
        tensor: name.into(),
        byte_length: shape.iter().product::<u64>() * kernel_core::precision::dtype_bits(dtype) / 8,
        dtype,
        shape,
        offset: 0,
    }
}
fn offer_fixture(offer: &KernelOffer) -> (PrecisionDocument, Provider, Weights) {
    use kernel_core::precision::capability::ActivationCapability;
    use kernel_core::precision::{
        Calibration, Clipping, DynamicScale, GroupAxis, NibbleOrder, QuantizerSpec, Rounding,
        ScaleConvention, ScaleScope, ScaleSpec,
    };
    let c = offer.capability().unwrap();
    let mut logical = vec![c.geometry.rows, c.geometry.k];
    if let OperationKind::Grouped { experts, .. } = c.kind {
        logical.insert(0, u64::from(experts));
    }
    let group = if c.weight == WeightFormat::Nvfp4 {
        16
    } else {
        32
    };
    let mut codes = logical.clone();
    if c.weight == WeightFormat::Nvfp4 {
        *codes.last_mut().unwrap() /= 2;
    }
    let mut scales = logical.clone();
    *scales.last_mut().unwrap() /= group;
    let weight = WeightSpec {
        codes: tensor(
            "weight",
            match c.weight {
                WeightFormat::Nvfp4 => DType::U8,
                WeightFormat::Bf16 => DType::Bf16,
                _ => DType::Fp8E4M3,
            },
            codes,
        ),
        logical_shape: logical.clone(),
        format: c.weight,
        encoding: if c.weight == WeightFormat::Nvfp4 {
            CodeEncoding::E2M1(NibbleOrder::EvenLow)
        } else if c.weight == WeightFormat::Bf16 {
            CodeEncoding::Bf16LittleEndian
        } else {
            CodeEncoding::E4M3
        },
        layout: c.layout.clone(),
        local_scale: (c.weight != WeightFormat::Bf16).then(|| ScaleSpec {
            tensor: tensor(
                "weight_scale",
                if c.weight == WeightFormat::Nvfp4 {
                    DType::Fp8E4M3
                } else {
                    DType::Ue8M0
                },
                scales,
            ),
            scope: ScaleScope::Block,
            convention: ScaleConvention::Multiply,
            groups: vec![
                GroupAxis {
                    axis: (logical.len() - 2) as u8,
                    size: 1,
                    origin: 0,
                },
                GroupAxis {
                    axis: (logical.len() - 1) as u8,
                    size: group as u32,
                    origin: 0,
                },
            ],
        }),
        global_scale: (c.weight == WeightFormat::Nvfp4).then(|| ScaleSpec {
            tensor: tensor("weight_global", DType::F32, vec![1]),
            scope: ScaleScope::Tensor,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        }),
        transforms: vec![],
    };
    let activation = match c.activation.unwrap_or(ActivationCapability::Bf16) {
        ActivationCapability::Nvfp4StaticScalar | ActivationCapability::Nvfp4TensorAmax => {
            ActivationSpec {
                format: ActivationFormat::Nvfp4,
                storage: DType::Bf16,
                compute: DType::Fp4E2M1,
                quantizer: Some(QuantizerSpec {
                    calibration: if c.activation == Some(ActivationCapability::Nvfp4StaticScalar) {
                        Calibration::Static(ScaleSpec {
                            tensor: tensor("activation_global", DType::F32, vec![1]),
                            scope: ScaleScope::Projection,
                            convention: ScaleConvention::Multiply,
                            groups: vec![],
                        })
                    } else {
                        Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax)
                    },
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
        ActivationCapability::Mxfp8 => ActivationSpec {
            format: ActivationFormat::Mxfp8,
            storage: DType::Bf16,
            compute: DType::Fp8E4M3,
            quantizer: Some(QuantizerSpec {
                calibration: Calibration::Dynamic(DynamicScale::MxPowerOfTwo),
                groups: vec![GroupAxis {
                    axis: 1,
                    size: 32,
                    origin: 0,
                }],
                rounding: Rounding::NearestEven,
                clipping: Clipping::FiniteSaturate,
            }),
            transforms: vec![],
        },
        ActivationCapability::Bf16 => ActivationSpec::bf16(),
        _ => unreachable!(),
    };
    let source = OperationSpec {
        id: "projection".into(),
        kind: c.kind,
        target: c.target,
        weights: vec![weight; usize::from(c.operands)],
        geometry: (c.kind != OperationKind::Lookup).then_some(c.geometry),
        activations: if c.kind == OperationKind::Lookup {
            vec![]
        } else {
            vec![activation; usize::from(c.operands)]
        },
        arithmetic: c.arithmetic,
        epilogues: c.epilogues,
        uses: vec![usage()],
    };
    let inventory = kernels_sm120::precision::CompiledInventory::new(
        vec![offer.clone()],
        c.stages.iter().cloned().collect(),
        "a".repeat(64),
    )
    .unwrap();
    let bound = inventory.bind(&source, usage()).unwrap();
    let doc = PrecisionDocument {
        version: 1,
        operations: vec![PrecisionSelection {
            source: source.clone(),
            effective: source,
            policy: ActivationPolicy::Checkpoint,
            reuse: None,
            fallback_reasons: vec![],
            bindings: vec![BoundUse {
                usage: usage(),
                implementation: bound.identity.clone(),
            }],
        }],
    };
    let provider = Provider {
        identity: bound.identity,
        recipe: prepared(&bound.handle, None, global(&doc), vec![]).unwrap(),
    };
    let views = Weights {
        codes: span(16, 1 << 20),
        local_scales: (c.weight != WeightFormat::Bf16).then(|| span(17, 1 << 20)),
        global_scales: (c.weight == WeightFormat::Nvfp4).then(|| span(18, 16)),
        activation_global: (c.activation == Some(ActivationCapability::Nvfp4StaticScalar))
            .then(|| span(15, 4)),
    };
    (doc, provider, views)
}
fn handle(offer: &KernelOffer) -> Handle {
    let (doc, provider, views) = offer_fixture(offer);
    let mut store = Store::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &provider,
        &Registry::default(),
        |_| Ok(views),
    )
    .unwrap();
    let position = store.positions["projection"];
    store.handles.swap_remove(position).remove(0)
}

fn args(item: &PreparedLaunch) -> Vec<KernelArgumentV1> {
    dispatch::arguments(&item.arguments, &item.bindings).unwrap()
}

#[test]
fn supplied_global_quantization_precedes_compute_and_keeps_split_count() {
    let offer = dense(row::Nvfp4Global::Supplied);
    let h = handle(&offer);
    let (input, output, mut scratch) = buffers();
    scratch.activation_global = Some(span(99, 4));
    let calls = launch(&h, phase(), input, output, &scratch).unwrap();
    assert_eq!(
        calls.iter().map(|c| c.symbol.as_str()).collect::<Vec<_>>(),
        ["projection_quant_v1", "projection_dense_v1"]
    );
    assert_eq!(
        args(&calls[0]),
        vec![
            KernelArgumentV1::DeviceBuffer(input.activation),
            KernelArgumentV1::DeviceBuffer(scratch.codes.unwrap()),
            KernelArgumentV1::DeviceBuffer(scratch.scales.unwrap()),
            KernelArgumentV1::DeviceBuffer(h.weights().activation_global.unwrap()),
            KernelArgumentV1::DeviceBuffer(input.live_rows)
        ]
    );
    let compute = args(&calls[1]);
    assert_eq!(compute[5], KernelArgumentV1::ScalarU32(3));
    assert_eq!(compute[8], KernelArgumentV1::ScalarU32(2));
    assert_eq!(
        compute[11],
        KernelArgumentV1::DeviceBuffer(scratch.scales.unwrap())
    );
    assert_eq!(
        compute[14],
        KernelArgumentV1::DeviceBuffer(h.weights().activation_global.unwrap())
    );
    assert_eq!(calls[1].plan.workspace_bytes, h.needs().workspace_bytes);
    assert!(h.needs().workspace_bytes > 0);
}

#[test]
fn dynamic_tensor_amax_runs_both_reductions_before_quantize_and_compute() {
    let offer = dense(row::Nvfp4Global::TensorAmax);
    let h = handle(&offer);
    let (input, output, mut scratch) = buffers();
    scratch.row_amax = Some(span(19, 32));
    let calls = launch(&h, phase(), input, output, &scratch).unwrap();
    let q = offer.handle().quantizer.as_ref().unwrap();
    assert_eq!(calls.len(), 4);
    assert_eq!(
        calls.iter().take(3).map(|c| &c.symbol).collect::<Vec<_>>(),
        q.stages()
            .iter()
            .map(|s| &s.entry().symbol)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        args(&calls[0])[1],
        KernelArgumentV1::DeviceBuffer(scratch.row_amax.unwrap())
    );
    assert_eq!(
        args(&calls[1])[1],
        KernelArgumentV1::DeviceBuffer(scratch.activation_global.unwrap())
    );
    assert_eq!(h.needs().quantizer_scratch_bytes, 36);
    scratch.row_amax = Some(span(19, 31));
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
}

fn grouped_offer() -> KernelOffer {
    let mut request = moe::GroupedRequest {
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
        tokens_capacity: 8,
        weight: WeightFormat::Mxfp8,
        activation: ActivationFormat::Mxfp8,
        scales: moe::ScaleBindings::NONE,
        output_quantization: None,
        swiglu_columns: kernel_core::contract::SwigluColumns::ConcatenatedHalves,
        target: Target::Sm120f,
        symbol: "projection_grouped_v1".into(),
        knobs: None,
    };
    let mut knobs = moe::instantiate(request.clone()).unwrap().knobs;
    knobs.split_k_capacity = 2;
    request.knobs = Some(knobs);
    KernelOffer::grouped(
        request,
        usage(),
        row::Nvfp4Global::Absent,
        "projection_gather_v1",
        "projection_group_quant_v1",
    )
    .unwrap()
}

#[test]
fn grouped_gather_quantization_and_split_reduction_use_the_routed_workspace() {
    let offer = grouped_offer();
    let h = handle(&offer);
    let (input, output, mut scratch) = buffers();
    scratch.activation_global = None;
    let calls = launch(&h, phase(), input, output, &scratch).unwrap();
    assert_eq!(
        calls.iter().map(|c| c.symbol.as_str()).collect::<Vec<_>>(),
        [
            "projection_gather_v1",
            "projection_group_quant_v1",
            "projection_grouped_v1",
            "projection_grouped_reduce_v1"
        ]
    );
    let gather = args(&calls[0]);
    assert_eq!(gather[0], KernelArgumentV1::DeviceBuffer(input.activation));
    assert_eq!(
        gather[2],
        KernelArgumentV1::DeviceBuffer(scratch.gathered.unwrap())
    );
    let quant = args(&calls[1]);
    assert_eq!(quant[0], gather[2]);
    assert_eq!(
        quant[4],
        KernelArgumentV1::DeviceBuffer(input.routing.unwrap().padded_rows)
    );
    assert_eq!(
        args(&calls[2])[14],
        KernelArgumentV1::DeviceBuffer(scratch.partial)
    );
    let reduce = args(&calls[3]);
    assert_eq!(reduce[0], KernelArgumentV1::DeviceBuffer(scratch.partial));
    assert_eq!(reduce[1], KernelArgumentV1::DeviceBuffer(output));
    assert_eq!(calls[3].plan.workspace_bytes, 0);
    assert!(h.needs().workspace_bytes > 0);
}

#[test]
fn wrong_use_and_undersized_buffers_fail_before_submission() {
    let h = handle(&dense(row::Nvfp4Global::Supplied));
    let (input, output, mut scratch) = buffers();
    let mut wrong = phase();
    wrong.usage.phase = ExecutionPhase::Verify;
    assert!(launch(&h, wrong, input, output, &scratch).is_err());
    wrong = phase();
    wrong.usage.graph = false;
    assert!(launch(&h, wrong, input, output, &scratch).is_err());
    wrong = phase();
    wrong.rows = 9;
    assert!(launch(&h, wrong, input, output, &scratch).is_err());
    scratch.partial.length = h.needs().workspace_bytes - 1;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch = buffers().2;
    scratch.codes.as_mut().unwrap().length = h.needs().codes_bytes - 1;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch = buffers().2;
    scratch.scales.as_mut().unwrap().length = h.needs().scales_bytes - 1;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch = buffers().2;
    assert!(launch(&h, phase(), input, span(8, 1), &scratch).is_err());
}

struct Provider {
    identity: ImplementationIdentity,
    recipe: Recipe,
}
impl ImplementationProvider for Provider {
    type Handle = Recipe;
    fn bind(&self, _: &OperationSpec, _: OperationUse) -> Result<Binding<Recipe>, String> {
        Ok(Binding {
            identity: self.identity.clone(),
            handle: self.recipe.clone(),
        })
    }
}
/// `operation`, with `copy` into a strided output, prepared through the
/// SM120 backend's projection op for the tests' use, after the registered
/// `prefix` stages.
fn prepared(
    operation: &OperationHandle,
    copy: Option<row::LayoutCopy>,
    global: ActivationGlobal,
    prefix: Vec<String>,
) -> Result<Recipe, Refusal> {
    let op = ProjectionOp::new(
        Arc::new(operation.clone()),
        copy.map(Box::new),
        usage(),
        global,
    )?;
    Ok(Recipe::Prepared {
        op: Arc::new(op),
        prefix,
    })
}
/// The activation global recipe of the document's first operation.
fn global(doc: &PrecisionDocument) -> ActivationGlobal {
    ActivationGlobal::of(&doc.operations[0].effective).unwrap()
}
/// The BF16 tiled GEMM offer [`fixture`] binds.
fn plain_offer() -> KernelOffer {
    let spec = gemm::instantiate(gemm::DenseRequest {
        query: gemm::bf16_query(8, 256, 256, DType::Bf16),
        kernel: DenseKernel::Tiled { splits: 1 },
        target: Target::Sm120f,
        symbol: "projection_plain_v1".into(),
        knobs: None,
    })
    .unwrap();
    KernelOffer::catalog_dense(spec, usage()).unwrap()
}
fn fixture() -> (PrecisionDocument, Provider) {
    let offer = plain_offer();
    let c = offer.capability().unwrap();
    let activation = ActivationSpec::bf16();
    let identity = ImplementationIdentity {
        implementation: c.implementation.clone(),
        dependency: "a".repeat(64),
        weight_layouts: vec![c.layout.clone()],
        activations: vec![activation.clone()],
        arithmetic: c.arithmetic.clone(),
        stages: c.stages.clone(),
    };
    let source = OperationSpec {
        id: "projection".into(),
        kind: OperationKind::Dense,
        target: Target::Sm120f,
        weights: vec![WeightSpec {
            codes: TensorSpan {
                tensor: "weight".into(),
                dtype: DType::Bf16,
                shape: vec![256, 256],
                offset: 0,
                byte_length: 256 * 256 * 2,
            },
            logical_shape: vec![256, 256],
            format: WeightFormat::Bf16,
            encoding: CodeEncoding::Bf16LittleEndian,
            layout: c.layout,
            local_scale: None,
            global_scale: None,
            transforms: vec![],
        }],
        geometry: Some(c.geometry),
        activations: vec![activation],
        arithmetic: c.arithmetic,
        epilogues: Default::default(),
        uses: vec![usage()],
    };
    (
        PrecisionDocument {
            version: 1,
            operations: vec![PrecisionSelection {
                source: source.clone(),
                effective: source,
                policy: ActivationPolicy::Bf16,
                reuse: None,
                fallback_reasons: vec![],
                bindings: vec![BoundUse {
                    usage: usage(),
                    implementation: identity.clone(),
                }],
            }],
        },
        Provider {
            identity,
            recipe: prepared(offer.handle(), None, ActivationGlobal::Absent, vec![]).unwrap(),
        },
    )
}
fn weights() -> Weights {
    Weights {
        codes: span(16, 1 << 20),
        local_scales: None,
        global_scales: None,
        activation_global: None,
    }
}

/// Selections a provider supplies one recipe for share one bound use; a
/// recipe with another operation does not.
#[test]
fn uses_of_one_supplied_recipe_share_their_binding() {
    let (mut doc, provider) = fixture();
    let mut other = doc.operations[0].clone();
    other.source.id = "other".into();
    other.effective.id = "other".into();
    doc.operations.push(other);
    let bytes = kernel_core::precision::component::encode(&doc).unwrap();
    let registry = Registry::default();
    let store = Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).unwrap();
    let bound = |id| Arc::clone(&store.handle(id, usage()).unwrap().bound);
    assert!(Arc::ptr_eq(&bound("projection"), &bound("other")));

    /// A provider preparing a fresh op of its offer for every use.
    struct Copies(Provider, KernelOffer);
    impl ImplementationProvider for Copies {
        type Handle = Recipe;
        fn bind(
            &self,
            operation: &OperationSpec,
            usage: OperationUse,
        ) -> Result<Binding<Recipe>, String> {
            let mut binding = self.0.bind(operation, usage)?;
            binding.handle = prepared(self.1.handle(), None, ActivationGlobal::Absent, vec![])
                .map_err(|refusal| refusal.to_string())?;
            Ok(binding)
        }
    }
    let copies = Copies(provider, plain_offer());
    let store = Store::decode(&bytes, &copies, &registry, |_| Ok(weights())).unwrap();
    let bound = |id| Arc::clone(&store.handle(id, usage()).unwrap().bound);
    assert!(!Arc::ptr_eq(&bound("projection"), &bound("other")));
}

#[test]
fn decode_refuses_changed_dependencies_and_adapter_stages() {
    let (doc, mut provider) = fixture();
    let bytes = kernel_core::precision::component::encode(&doc).unwrap();
    let registry = Registry::default();
    let store = Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).unwrap();
    assert_eq!(
        store.handle("projection", usage()).unwrap().identity(),
        &provider.identity
    );
    assert!(store.handle("absent", usage()).is_err());
    provider.identity.dependency = "b".repeat(64);
    assert!(Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).is_err());
    provider.identity.dependency = "a".repeat(64);
    let mut operation = plain_offer().handle().clone();
    let Compute::Dense(spec) = &mut operation.compute else {
        unreachable!()
    };
    spec.symbol = "different_entry_v1".into();
    provider.recipe = prepared(&operation, None, ActivationGlobal::Absent, vec![]).unwrap();
    assert!(Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).is_err());
    assert!(Store::decode(b"invalid", &provider, &registry, |_| Ok(weights())).is_err());
}

struct Registered {
    symbol: String,
}
struct ScratchWriter {
    symbol: String,
}
impl Launcher for ScratchWriter {
    fn needs(&self) -> Needs {
        Needs {
            workspace_bytes: 64,
            semaphore_bytes: 16,
            ..Needs::default()
        }
    }
    fn writes(&self, _: Input<'_>, output: Span, scratch: &Scratch) -> Result<Vec<Span>, Refusal> {
        Ok(vec![output, scratch.partial, scratch.semaphores])
    }
    fn launch(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        Registered {
            symbol: self.symbol.clone(),
        }
        .launch(phase, input, output, scratch)
    }
}

#[test]
fn registered_writes_receive_reserved_scratch_and_refuse_aliases() {
    let (doc, mut provider) = fixture();
    provider.recipe = Recipe::Registered;
    let mut registry = Registry::default();
    registry
        .register(
            provider.identity.clone(),
            Arc::new(ScratchWriter {
                symbol: provider.identity.stages[0].clone(),
            }),
        )
        .unwrap();
    let store = Store::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &provider,
        &registry,
        |_| Ok(weights()),
    )
    .unwrap();
    let h = store.handle("projection", usage()).unwrap();
    let (mut input, output, mut scratch) = buffers();
    scratch.partial.length = 64;
    scratch.semaphores.length = 16;
    assert!(launch(h, phase(), input, output, &scratch).is_ok());
    let side = [scratch.partial];
    input.side = &side;
    assert!(launch(h, phase(), input, output, &scratch).is_err());
    input.side = &[];
    assert!(launch(h, phase(), input, scratch.partial, &scratch).is_err());
    scratch.partial.length = 63;
    assert!(launch(h, phase(), input, output, &scratch).is_err());
    scratch.partial.length = 64;
    scratch.semaphores.length = 15;
    assert!(launch(h, phase(), input, output, &scratch).is_err());
}
struct UnusedWriter;
impl Launcher for UnusedWriter {
    fn writes(&self, _: Input<'_>, output: Span, scratch: &Scratch) -> Result<Vec<Span>, Refusal> {
        Ok(vec![output, scratch.unused])
    }
    fn launch(
        &self,
        _: Phase,
        _: Input<'_>,
        _: Span,
        _: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        Ok(vec![])
    }
}

#[test]
fn registered_launchers_cannot_write_the_unused_argument_region() {
    let (doc, mut provider) = fixture();
    provider.recipe = Recipe::Registered;
    let bytes = kernel_core::precision::component::encode(&doc).unwrap();
    let store = |launcher: Arc<dyn Launcher>| {
        let mut registry = Registry::default();
        registry
            .register(provider.identity.clone(), launcher)
            .unwrap();
        Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).unwrap()
    };
    let (input, output, mut scratch) = buffers();
    let writer = store(Arc::new(UnusedWriter));
    assert_eq!(
        launch(
            writer.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch
        )
        .err(),
        Some(Refusal("registered projection writes an undeclared region"))
    );
    // A caller may point `unused` at a live region, such as a constant; it
    // takes no part in the scratch overlap refusal.
    scratch.unused = scratch.partial;
    let reader = store(Arc::new(Registered {
        symbol: provider.identity.stages[0].clone(),
    }));
    assert!(
        launch(
            reader.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch
        )
        .is_ok()
    );
}

impl Launcher for Registered {
    fn writes(&self, _input: Input<'_>, output: Span, _: &Scratch) -> Result<Vec<Span>, Refusal> {
        Ok(vec![output])
    }
    fn needs(&self) -> Needs {
        Needs {
            workspace_bytes: 64,
            ..Needs::default()
        }
    }
    fn launch(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        _: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        let mut bindings = BufferBindings::default();
        Ok(vec![PreparedLaunch {
            symbol: self.symbol.clone(),
            rows: phase.rows,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 64,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(vec![
                kernel_core::contract::Argument::DeviceBuffer(bindings.bind(input.activation)),
                kernel_core::contract::Argument::DeviceBuffer(bindings.bind(output)),
            ]),
            bindings,
        }])
    }
}
#[test]
fn registered_launchers_bind_full_identity_and_reject_unlisted_entries() {
    let (doc, mut provider) = fixture();
    provider.recipe = Recipe::Registered;
    let bytes = kernel_core::precision::component::encode(&doc).unwrap();
    let mut registry = Registry::default();
    assert!(Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).is_err());
    let launcher = Arc::new(Registered {
        symbol: provider.identity.stages[0].clone(),
    });
    registry
        .register(provider.identity.clone(), launcher.clone())
        .unwrap();
    assert!(
        registry
            .register(provider.identity.clone(), launcher)
            .is_err()
    );
    let store = Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).unwrap();
    let h = store.handle("projection", usage()).unwrap();
    assert_eq!(h.needs().workspace_bytes, 64);
    let (input, output, scratch) = buffers();
    let calls = launch(h, phase(), input, output, &scratch).unwrap();
    assert_eq!(calls[0].symbol, provider.identity.stages[0]);
    let mut bad = Registry::default();
    bad.register(
        provider.identity.clone(),
        Arc::new(Registered {
            symbol: "unlisted_entry_v1".into(),
        }),
    )
    .unwrap();
    let store = Store::decode(&bytes, &provider, &bad, |_| Ok(weights())).unwrap();
    assert!(
        launch(
            store.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch
        )
        .is_err()
    );
}

/// A registered launcher that states each launch's persistence.
struct Persistent(Registered);

impl Launcher for Persistent {
    fn writes(
        &self,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<Span>, Refusal> {
        self.0.writes(input, output, scratch)
    }
    fn needs(&self) -> Needs {
        self.0.needs()
    }
    fn launch(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        self.0.launch(phase, input, output, scratch)
    }
    fn launch_with_persistence(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<(PreparedLaunch, Option<bool>)>, Refusal> {
        Ok(self
            .launch(phase, input, output, scratch)?
            .into_iter()
            .map(|launch| (launch, Some(true)))
            .collect())
    }
}

/// A registered stage's launch carries the persistence its launcher
/// states, and none when the launcher states nothing.
#[test]
fn registered_launches_carry_the_persistence_their_launcher_states() {
    let (doc, mut provider) = fixture();
    provider.recipe = Recipe::Registered;
    let bytes = kernel_core::precision::component::encode(&doc).unwrap();
    let symbol = provider.identity.stages[0].clone();
    let (input, output, scratch) = buffers();
    let persistence = |launcher: Arc<dyn Launcher>| {
        let mut registry = Registry::default();
        registry
            .register(provider.identity.clone(), launcher)
            .unwrap();
        let store = Store::decode(&bytes, &provider, &registry, |_| Ok(weights())).unwrap();
        let h = store.handle("projection", usage()).unwrap();
        let launches = launch_with_persistence(h, phase(), input, output, &scratch).unwrap();
        let key = |launch: &PreparedLaunch| (launch.symbol.clone(), launch.rows, launch.plan);
        assert_eq!(
            launches
                .iter()
                .map(|(launch, _)| key(launch))
                .collect::<Vec<_>>(),
            launch(h, phase(), input, output, &scratch)
                .unwrap()
                .iter()
                .map(key)
                .collect::<Vec<_>>()
        );
        launches
            .into_iter()
            .map(|(_, persistent)| persistent)
            .collect::<Vec<_>>()
    };
    let registered = || Registered {
        symbol: symbol.clone(),
    };
    assert_eq!(persistence(Arc::new(registered())), [None]);
    assert_eq!(
        persistence(Arc::new(Persistent(registered()))),
        [Some(true)]
    );
}

#[test]
fn activation_globals_follow_the_recipe_not_available_spans() {
    let (input, output, mut scratch) = buffers();
    let mut dynamic = handle(&dense(row::Nvfp4Global::TensorAmax));
    scratch.row_amax = Some(span(19, 32));
    dynamic.weights.activation_global = Some(span(99, 4));
    assert!(launch(&dynamic, phase(), input, output, &scratch).is_err());

    let mut supplied = handle(&dense(row::Nvfp4Global::Supplied));
    supplied.weights.activation_global = None;
    scratch.row_amax = None;
    assert!(launch(&supplied, phase(), input, output, &scratch).is_err());
}

#[test]
fn supplied_quantizer_ignores_dynamic_scratch_from_another_projection() {
    let h = handle(&dense(row::Nvfp4Global::Supplied));
    let (input, output, mut scratch) = buffers();
    scratch.row_amax = Some(span(19, 32));
    scratch.activation_global = Some(span(99, 4));
    let calls = launch(&h, phase(), input, output, &scratch).unwrap();
    assert_eq!(
        args(&calls[0])[3],
        KernelArgumentV1::DeviceBuffer(h.weights.activation_global.unwrap())
    );
}

#[test]
fn decode_refuses_a_persistent_global_for_an_absent_recipe() {
    let (doc, provider) = fixture();
    let bytes = kernel_core::precision::component::encode(&doc).unwrap();
    let mut views = weights();
    views.activation_global = Some(span(99, 4));
    assert!(Store::decode(&bytes, &provider, &Registry::default(), |_| Ok(views)).is_err());
}

#[test]
fn absent_quantizer_ignores_global_and_amax_scratch() {
    let h = handle(&grouped_offer());
    let (input, output, mut scratch) = buffers();
    scratch.row_amax = Some(span(19, 32));
    let calls = launch(&h, phase(), input, output, &scratch).unwrap();
    assert_eq!(args(&calls[1])[3], KernelArgumentV1::NullDevicePointer);
}

#[test]
fn scratch_aliases_and_short_semaphores_are_refused() {
    let h = handle(&dense(row::Nvfp4Global::Supplied));
    let (input, output, mut scratch) = buffers();
    scratch.scales = scratch.codes;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch = buffers().2;
    scratch.codes = Some(input.activation);
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch = buffers().2;
    scratch.semaphores.length = 0;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch = buffers().2;
    scratch.row_amax = Some(scratch.partial);
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
}

#[test]
fn a_layout_copy_cannot_alias_its_output() {
    let offer = dense(row::Nvfp4Global::Supplied);
    let (mut doc, mut provider, views) = offer_fixture(&offer);
    let copy = row::LayoutCopy::new(
        256,
        8,
        row::CopyRows {
            row_stride: 256,
            column_stride: 1,
            column_offset: 0,
        },
        row::CopyRows {
            row_stride: 256,
            column_stride: 1,
            column_offset: 0,
        },
        "projection_copy_v1",
        row::ShapeBinding::Compiled,
        row::RowResources::SM120F,
    )
    .unwrap();
    provider
        .identity
        .stages
        .push(copy.stage().entry().symbol.clone());
    doc.operations[0].bindings[0].implementation = provider.identity.clone();
    provider.recipe = prepared(offer.handle(), Some(copy), global(&doc), vec![]).unwrap();
    let store = Store::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &provider,
        &Registry::default(),
        |_| Ok(views),
    )
    .unwrap();
    let (input, output, mut scratch) = buffers();
    scratch.temporary = Some(output);
    assert!(
        launch(
            store.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch
        )
        .is_err()
    );
}

fn lookup_offer() -> KernelOffer {
    KernelOffer::lookup(
        row::LookupRequest {
            vocab: 256,
            width: 128,
            heads: 2,
            capacity: 8,
            weight: WeightFormat::Bf16,
            global_scale: None,
            ids: row::LookupIds::I32,
            shapes: row::ShapeBinding::Compiled,
        },
        usage(),
        "projection_lookup_v1",
    )
    .unwrap()
}

#[test]
fn non_dense_copies_are_refused_and_destinations_cover_capacity() {
    for offer in [grouped_offer(), lookup_offer()] {
        let (doc, _, _) = offer_fixture(&offer);
        let copy = row::LayoutCopy::new(
            128,
            8,
            row::CopyRows {
                row_stride: 128,
                column_stride: 1,
                column_offset: 0,
            },
            row::CopyRows {
                row_stride: 128,
                column_stride: 1,
                column_offset: 0,
            },
            "projection_copy_v1",
            row::ShapeBinding::Compiled,
            row::RowResources::SM120F,
        )
        .unwrap();
        assert!(prepared(offer.handle(), Some(copy), global(&doc), vec![]).is_err());
        let h = handle(&offer);
        let (mut input, _, scratch) = buffers();
        input.shard_rows = 256;
        assert!(launch(&h, phase(), input, span(8, 1), &scratch).is_err());
    }
}

#[test]
fn prefix_side_regions_cannot_alias_shared_scratch() {
    let (mut doc, mut provider) = fixture();
    let symbol = provider.identity.stages[0].clone();
    provider.recipe = prepared(
        plain_offer().handle(),
        None,
        ActivationGlobal::Absent,
        vec![symbol.clone()],
    )
    .unwrap();
    provider.identity.stages.insert(0, symbol.clone());
    doc.operations[0].bindings[0].implementation = provider.identity.clone();
    let mut registry = Registry::default();
    registry
        .register(provider.identity.clone(), Arc::new(Registered { symbol }))
        .unwrap();
    let store = Store::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &provider,
        &registry,
        |_| Ok(weights()),
    )
    .unwrap();
    let (mut input, output, scratch) = buffers();
    let side = [scratch.partial];
    input.side = &side;
    assert!(
        launch(
            store.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch
        )
        .is_err()
    );
}

#[test]
fn a_split_gemm_refuses_an_empty_semaphore_region() {
    let h = handle(&dense(row::Nvfp4Global::Supplied));
    let (input, output, mut scratch) = buffers();
    scratch.semaphores.length = 0;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
}

#[test]
fn semaphore_capacity_accepts_exact_size_and_refuses_one_byte_short() {
    let h = handle(&dense(row::Nvfp4Global::Supplied));
    let (input, output, mut scratch) = buffers();
    assert_eq!(h.needs().semaphore_bytes, 16);
    scratch.semaphores.length = 16;
    assert!(launch(&h, phase(), input, output, &scratch).is_ok());
    scratch.semaphores.length = 15;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
    scratch.semaphores = Span::new(
        scratch.partial.buffer,
        scratch.partial.offset + scratch.partial.length,
        16,
    );
    assert!(launch(&h, phase(), input, output, &scratch).is_ok());
    scratch.semaphores.offset -= 1;
    assert!(launch(&h, phase(), input, output, &scratch).is_err());
}

struct RepeatedEntry {
    symbol: String,
}
impl Launcher for RepeatedEntry {
    fn writes(&self, _: Input<'_>, output: Span, _: &Scratch) -> Result<Vec<Span>, Refusal> {
        Ok(vec![output])
    }
    fn launch(
        &self,
        phase: Phase,
        input: Input<'_>,
        output: Span,
        scratch: &Scratch,
    ) -> Result<Vec<PreparedLaunch>, Refusal> {
        let entry = Registered {
            symbol: self.symbol.clone(),
        };
        let mut calls = entry.launch(phase, input, output, scratch)?;
        calls.extend(entry.launch(phase, input, output, scratch)?);
        Ok(calls)
    }
}
#[test]
fn registered_stage_order_preserves_authenticated_repeated_entries() {
    let (mut doc, mut provider) = fixture();
    let symbol = provider.identity.stages[0].clone();
    provider.identity.stages.push(symbol.clone());
    doc.operations[0].bindings[0].implementation = provider.identity.clone();
    provider.recipe = Recipe::Registered;
    let mut registry = Registry::default();
    registry
        .register(
            provider.identity.clone(),
            Arc::new(RepeatedEntry { symbol }),
        )
        .unwrap();
    let store = Store::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &provider,
        &registry,
        |_| Ok(weights()),
    )
    .unwrap();
    let (input, output, scratch) = buffers();
    let calls = launch_with_persistence(
        store.handle("projection", usage()).unwrap(),
        phase(),
        input,
        output,
        &scratch,
    )
    .unwrap();
    assert_eq!(calls.len(), 2);
    // A registered launcher states no persistence.
    assert!(calls.iter().all(|(_, persistent)| persistent.is_none()));
}

#[test]
fn registered_stage_repetitions_must_be_declared() {
    let (doc, mut provider) = fixture();
    provider.recipe = Recipe::Registered;
    let mut registry = Registry::default();
    registry
        .register(
            provider.identity.clone(),
            Arc::new(RepeatedEntry {
                symbol: provider.identity.stages[0].clone(),
            }),
        )
        .unwrap();
    let store = Store::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &provider,
        &registry,
        |_| Ok(weights()),
    )
    .unwrap();
    let (input, output, scratch) = buffers();
    assert!(
        launch(
            store.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch
        )
        .is_err()
    );
}

#[test]
fn authentication_runs_once_and_ranks_bind_only_their_views() {
    use std::cell::Cell;
    struct Counting<'a>(&'a Provider, &'a Cell<u32>);
    impl ImplementationProvider for Counting<'_> {
        type Handle = Recipe;
        fn bind(
            &self,
            operation: &OperationSpec,
            usage: OperationUse,
        ) -> Result<Binding<Recipe>, String> {
            self.1.set(self.1.get() + 1);
            self.0.bind(operation, usage)
        }
    }
    let (mut doc, mut provider) = fixture();
    provider.recipe = Recipe::Registered;
    let mut other = doc.operations[0].clone();
    other.source.id = "other".into();
    other.effective.id = "other".into();
    doc.operations.push(other);
    let mut launchers = 0;
    let registry = Registry::for_document(&doc, |_, _| {
        launchers += 1;
        Ok(Some(Arc::new(Registered {
            symbol: provider.identity.stages[0].clone(),
        })))
    })
    .unwrap();
    // Both selections share one identity, so one launcher is registered.
    assert_eq!(launchers, 1);
    let binds = Cell::new(0);
    let authenticated = Authenticated::decode(
        &kernel_core::precision::component::encode(&doc).unwrap(),
        &Counting(&provider, &binds),
        &registry,
    )
    .unwrap();
    assert_eq!(binds.get(), 2);
    let ranks = (0..4)
        .map(|rank| {
            authenticated
                .bind(|s| Ok((rank == 0 || s.source.id == "projection").then(weights)))
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(binds.get(), 2);
    assert!(ranks[0].handle("other", usage()).is_ok());
    assert_eq!(ranks[0].document().operations.len(), 2);
    assert!(ranks[1].handle("other", usage()).is_err());
    assert_eq!(ranks[1].document().operations.len(), 1);
    assert_eq!(
        ranks[3].handle("projection", usage()).unwrap().identity(),
        &provider.identity
    );
    assert!(authenticated.bind(|_| Err("missing view".into())).is_err());
    // A located use indexes the handle the lookup finds, on every rank that
    // serves its selection.
    let other = authenticated.locate("other", usage()).unwrap();
    assert_eq!(
        ranks[0].at(other).unwrap().identity(),
        ranks[0].handle("other", usage()).unwrap().identity()
    );
    assert!(ranks[1].at(other).is_err());
    let projection = authenticated.locate("projection", usage()).unwrap();
    assert_ne!(projection, other);
    assert!(ranks[3].at(projection).is_ok());
    assert!(authenticated.locate("missing", usage()).is_none());
    let mut wider = usage();
    wider.capacity += 1;
    assert!(authenticated.locate("projection", wider).is_none());
    // A decoded document authenticates alike without validating it again.
    let decoded = Authenticated::decoded(
        kernel_core::precision::component::decode(
            &kernel_core::precision::component::encode(&doc).unwrap(),
        )
        .unwrap(),
        &Counting(&provider, &binds),
        &registry,
    )
    .unwrap();
    assert_eq!(binds.get(), 4);
    assert_eq!(decoded.document(), authenticated.document());
}

/// The same offers through the SM120 backend give the launches their
/// projection op gives directly: symbols, rows, plans and resolved
/// arguments, with the same needs and stage authentication.
#[test]
fn a_backend_prepared_projection_launches_what_its_offer_does() {
    use kernel_core::backend::{Backend, Need};
    use kernels_sm120::backend::{NamedOffers, ProjectionOffer, Sm120};

    /// Binds every operation to the one offer its identity names, a named
    /// composition.
    struct Offers(ImplementationIdentity, KernelOffer);
    impl ImplementationProvider for Offers {
        type Handle = ProjectionOffer;
        fn bind(
            &self,
            _: &OperationSpec,
            _: OperationUse,
        ) -> Result<Binding<ProjectionOffer>, String> {
            Ok(Binding {
                identity: self.0.clone(),
                handle: ProjectionOffer {
                    offer: self.1.clone(),
                    copy: None,
                    dense: kernels_sm120::backend::DenseForm::Tensor,
                    live_output: false,
                },
            })
        }
    }

    let (input, output, scratch) = buffers();
    let amax = Scratch {
        row_amax: Some(span(19, 32)),
        ..scratch
    };
    let no_global = Scratch {
        activation_global: None,
        ..scratch
    };
    let mut lookup_input = input;
    lookup_input.shard_rows = 256;
    for (offer, input, scratch) in [
        (dense(row::Nvfp4Global::Supplied), input, scratch),
        (dense(row::Nvfp4Global::TensorAmax), input, amax),
        (grouped_offer(), input, no_global),
        (lookup_offer(), lookup_input, scratch),
    ] {
        let (doc, provider, views) = offer_fixture(&offer);
        let bytes = kernel_core::precision::component::encode(&doc).unwrap();
        let backend = Sm120::new().with_projections(Box::new(NamedOffers(Offers(
            provider.identity.clone(),
            offer.clone(),
        ))));
        let spec = &doc.operations[0].effective;
        let prepared = backend
            .projection(spec, usage(), Need::rows(usage().capacity))
            .unwrap();
        assert_eq!(prepared.identity, provider.identity);
        let backed = Provider {
            identity: prepared.identity,
            recipe: Recipe::Prepared {
                op: Arc::from(prepared.handle),
                prefix: vec![],
            },
        };
        let direct = Store::decode(&bytes, &provider, &Registry::default(), |_| Ok(views)).unwrap();
        let backed = Store::decode(&bytes, &backed, &Registry::default(), |_| Ok(views)).unwrap();
        let direct = direct.handle("projection", usage()).unwrap();
        let backed = backed.handle("projection", usage()).unwrap();
        assert_eq!(direct.needs(), backed.needs());
        let expected = launch_with_persistence(direct, phase(), input, output, &scratch).unwrap();
        let observed = launch_with_persistence(backed, phase(), input, output, &scratch).unwrap();
        assert_eq!(expected.len(), observed.len());
        // Every launch of an op states its persistence, alike for both.
        for ((_, expected), (_, observed)) in expected.iter().zip(&observed) {
            assert!(expected.is_some());
            assert_eq!(expected, observed);
        }
        let expected: Vec<_> = expected.into_iter().map(|(launch, _)| launch).collect();
        let observed: Vec<_> = observed.into_iter().map(|(launch, _)| launch).collect();
        for (expected, observed) in expected.iter().zip(&observed) {
            assert_eq!(expected.symbol, observed.symbol);
            assert_eq!(expected.rows, observed.rows);
            assert_eq!(expected.plan, observed.plan);
            assert_eq!(args(expected), args(observed));
        }
    }
}

/// A registered prefix before a backend-prepared recipe may write only its
/// side regions, and the op's scratch must also hold what the prefix needs.
#[test]
fn prefixes_of_prepared_recipes_keep_their_limits() {
    struct Prefix {
        symbol: String,
        needs: Needs,
        writes_output: bool,
    }
    impl Launcher for Prefix {
        fn needs(&self) -> Needs {
            self.needs
        }
        fn writes(&self, _: Input<'_>, output: Span, _: &Scratch) -> Result<Vec<Span>, Refusal> {
            Ok(if self.writes_output {
                vec![output]
            } else {
                vec![]
            })
        }
        fn launch(
            &self,
            phase: Phase,
            input: Input<'_>,
            _: Span,
            _: &Scratch,
        ) -> Result<Vec<PreparedLaunch>, Refusal> {
            let mut bindings = BufferBindings::default();
            Ok(vec![PreparedLaunch {
                symbol: self.symbol.clone(),
                rows: phase.rows,
                plan: LaunchPlan {
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1, 1, 1],
                    pdl: true,
                },
                arguments: ArgumentList(vec![kernel_core::contract::Argument::DeviceBuffer(
                    bindings.bind(input.activation),
                )]),
                bindings,
            }])
        }
    }
    let grouped = grouped_offer();
    let amax = dense(row::Nvfp4Global::TensorAmax);
    let (input, output, scratch) = buffers();
    let grouped_scratch = Scratch {
        activation_global: None,
        ..scratch
    };
    let amax_scratch = Scratch {
        row_amax: Some(span(19, 32)),
        ..scratch
    };
    let gathered = scratch.gathered.unwrap().length;
    let run = |offer: &KernelOffer, prefix: Prefix| {
        let (global, scratch) = if offer.handle().gather.is_some() {
            (ActivationGlobal::Absent, grouped_scratch)
        } else {
            (ActivationGlobal::TensorAmax, amax_scratch)
        };
        let (mut doc, mut provider, views) = offer_fixture(offer);
        let identity = provider.identity.clone();
        provider.identity.stages.insert(0, prefix.symbol.clone());
        doc.operations[0].bindings[0].implementation = provider.identity.clone();
        let op =
            ProjectionOp::new(Arc::new(offer.handle().clone()), None, usage(), global).unwrap();
        assert!(identity.stages.len() == op.info().symbols.len());
        provider.recipe = Recipe::Prepared {
            op: Arc::new(op),
            prefix: vec![prefix.symbol.clone()],
        };
        let mut registry = Registry::default();
        registry
            .register(provider.identity.clone(), Arc::new(prefix))
            .unwrap();
        let store = Store::decode(
            &kernel_core::precision::component::encode(&doc).unwrap(),
            &provider,
            &registry,
            |_| Ok(views),
        )
        .unwrap();
        launch(
            store.handle("projection", usage()).unwrap(),
            phase(),
            input,
            output,
            &scratch,
        )
        .map(|launches| launches.len())
    };
    let prefix = |needs, writes_output| Prefix {
        symbol: "projection_prefix_v1".into(),
        needs,
        writes_output,
    };
    assert!(run(&grouped, prefix(Needs::default(), false)).is_ok());
    let error = run(&grouped, prefix(Needs::default(), true)).unwrap_err();
    assert!(error.0.contains("undeclared region"), "{error}");
    let wide = Needs {
        gathered_bytes: gathered + 1,
        ..Needs::default()
    };
    let error = run(&grouped, prefix(wide, false)).unwrap_err();
    assert!(error.0.contains("does not cover"), "{error}");
    // The op's dynamic quantizer needs 32 amax bytes and the global; a
    // prefix that needs 40 in all needs 36 of amax.
    assert!(run(&amax, prefix(Needs::default(), false)).is_ok());
    let wide = Needs {
        quantizer_scratch_bytes: 40,
        ..Needs::default()
    };
    let error = run(&amax, prefix(wide, false)).unwrap_err();
    assert!(error.0.contains("does not cover"), "{error}");
}
