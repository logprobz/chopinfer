//! Fake host coverage of authenticated mixed bindings and ordinary BF16 buffers.
//! These tests do not execute arithmetic on a device.
use super::*;
use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::execution::{
    ActivationPolicy, BoundUse, PrecisionDocument, PrecisionSelection,
};
use kernel_core::precision::*;
use model_qwen38_contract::precision::{self as ports, DenseRole, Projection};

fn tensor(name: String, dtype: DType, shape: Vec<u64>) -> TensorSpan {
    TensorSpan {
        byte_length: shape.iter().product::<u64>() * dtype_bits(dtype) / 8,
        tensor: name,
        dtype,
        shape,
        offset: 0,
    }
}

fn weight(id: &str, index: u8, c: &OperationCapability) -> WeightSpec {
    let mut shape = vec![c.geometry.rows, c.geometry.k];
    let expert = if let OperationKind::Grouped { experts, .. } = c.kind {
        shape.insert(0, experts.into());
        Some(experts)
    } else {
        None
    };
    let mut codes = shape.clone();
    let (dtype, encoding) = match c.weight {
        WeightFormat::Bf16 => (DType::Bf16, CodeEncoding::Bf16LittleEndian),
        WeightFormat::Nvfp4 => {
            *codes.last_mut().unwrap() /= 2;
            (DType::U8, CodeEncoding::E2M1(NibbleOrder::EvenLow))
        }
        _ => (DType::Fp8E4M3, CodeEncoding::E4M3),
    };
    let (rn, rk, scale_dtype) = match c.weight {
        WeightFormat::Nvfp4 => (1, 16, DType::Fp8E4M3),
        WeightFormat::Mxfp8 => (1, 32, DType::Ue8M0),
        WeightFormat::Fp8Block128x128(_) => (128, 128, DType::F32),
        _ => (1, 1, DType::Bf16),
    };
    let local_scale = if c.weight == WeightFormat::Bf16 {
        None
    } else if c.weight == WeightFormat::Fp8PerTensor {
        Some(ScaleSpec {
            tensor: tensor(format!("{id}.{index}.scale"), DType::Bf16, vec![1]),
            scope: ScaleScope::Tensor,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        })
    } else {
        let mut scales = shape.clone();
        let n = scales.len();
        scales[n - 2] = scales[n - 2].div_ceil(rn);
        scales[n - 1] = scales[n - 1].div_ceil(rk);
        Some(ScaleSpec {
            tensor: tensor(format!("{id}.{index}.scale"), scale_dtype, scales),
            scope: ScaleScope::Block,
            convention: ScaleConvention::Multiply,
            groups: vec![
                GroupAxis {
                    axis: (n - 2) as u8,
                    size: rn as u32,
                    origin: 0,
                },
                GroupAxis {
                    axis: (n - 1) as u8,
                    size: rk as u32,
                    origin: 0,
                },
            ],
        })
    };
    WeightSpec {
        codes: tensor(format!("{id}.{index}.weight"), dtype, codes),
        logical_shape: shape,
        format: c.weight,
        encoding,
        layout: c.layout.clone(),
        local_scale,
        global_scale: (c.weight == WeightFormat::Nvfp4).then(|| ScaleSpec {
            tensor: tensor(
                format!("{id}.{index}.global"),
                DType::F32,
                vec![expert.map_or(1, u64::from)],
            ),
            scope: if expert.is_some() {
                ScaleScope::Expert
            } else {
                ScaleScope::Tensor
            },
            convention: ScaleConvention::Multiply,
            groups: vec![],
        }),
        transforms: vec![],
    }
}

pub(crate) fn selection(
    id: String,
    c: OperationCapability,
    inventory: &kernel_core::precision::capability::CapabilityInventory,
) -> PrecisionSelection {
    let operation = OperationSpec {
        id,
        kind: c.kind,
        target: c.target,
        weights: (0..c.operands)
            .map(|i| weight(&c.implementation, i, &c))
            .collect(),
        geometry: (c.kind != OperationKind::Lookup).then_some(c.geometry),
        activations: if c.kind == OperationKind::Lookup {
            vec![]
        } else {
            vec![ActivationSpec::bf16(); usize::from(c.operands)]
        },
        arithmetic: c.arithmetic.clone(),
        epilogues: c.epilogues,
        uses: ports::uses().unwrap(),
    };
    let bindings = operation
        .uses
        .iter()
        .map(|&usage| BoundUse {
            usage,
            implementation: inventory
                .bind_matching(&operation, usage, |entry| {
                    entry.implementation == c.implementation
                })
                .unwrap()
                .identity,
        })
        .collect();
    PrecisionSelection {
        source: operation.clone(),
        effective: operation,
        policy: ActivationPolicy::Bf16,
        reuse: None,
        fallback_reasons: vec![],
        bindings,
    }
}

fn document(nv: bool, mx: bool) -> PrecisionDocument {
    document_for(Shape::TP4, nv, mx, false)
}

pub(super) fn published(
    shape: Shape,
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> OperationCapability {
    model_qwen38_contract::sm120::generated::publish(shape, id, weight, usage)
        .unwrap()
        .capability
        .clone()
}

pub(crate) fn document_for(
    shape: Shape,
    nv: bool,
    mx: bool,
    bf16_inputs: bool,
) -> PrecisionDocument {
    document_bound(
        shape,
        nv,
        mx,
        bf16_inputs,
        &crate::precision::inventory().unwrap(),
    )
}

/// [`document_for`] bound to the compositions of `inventory`.
pub(super) fn document_bound(
    shape: Shape,
    nv: bool,
    mx: bool,
    bf16_inputs: bool,
    inventory: &kernel_core::precision::capability::CapabilityInventory,
) -> PrecisionDocument {
    let mut operations = Vec::new();
    let usage = ports::uses().unwrap()[0];
    for layer in 0..shape.layers() {
        for (index, role) in DenseRole::roles_for(shape, layer)
            .unwrap()
            .into_iter()
            .enumerate()
        {
            let capability = if mx {
                let weight = if bf16_inputs && (1..4).contains(&index) {
                    WeightFormat::Bf16
                } else {
                    WeightFormat::Mxfp8
                };
                published(shape, role.projection_id(shape, None), weight, usage)
            } else {
                role.legacy_capability(shape, usage).unwrap()
            };
            operations.push(selection(
                role.operation_id_for(shape, layer).unwrap(),
                capability,
                inventory,
            ));
        }
        for projection in [Projection::GateUp, Projection::Down] {
            let (name, id) = if projection == Projection::GateUp {
                ("gate_up_proj", "main.experts.gate-up")
            } else {
                ("down_proj", "main.experts.down")
            };
            let capability = if nv {
                published(shape, id, WeightFormat::Nvfp4, usage)
            } else {
                ports::legacy_expert_capability(shape, projection, usage).unwrap()
            };
            operations.push(selection(
                format!(
                    "{}.mlp.experts.{name}",
                    model_qwen38_contract::views::layer_stem_for(layer, shape)
                ),
                capability,
                inventory,
            ));
        }
    }
    for ngram in [false, true] {
        let capability = if mx && ngram {
            published(shape, "ngram", WeightFormat::Nvfp4, usage)
        } else {
            ports::legacy_lookup_capability(shape, ngram, usage).unwrap()
        };
        let id = if ngram {
            crate::views::ngram_view_for(shape)
                .trim_end_matches(".weight")
                .into()
        } else {
            "model.language_model.embed_tokens".into()
        };
        operations.push(selection(id, capability, inventory));
    }
    let doc = PrecisionDocument {
        version: 1,
        operations,
    };
    doc.validate().unwrap();
    doc
}

/// Every mixed symbol `host`'s instance resolved lies in a mixed module of
/// its geometry (or of every geometry) and its rank count (or of every rank
/// count), and every symbol of its bound operations resolved.
fn assert_rank_matching_mixed_modules(host: &FakeHost, document: &PrecisionDocument) {
    let shape = host.shape();
    let names: BTreeSet<_> = host.symbol_names.borrow().values().cloned().collect();
    for cubin in crate::CUBINS
        .iter()
        .filter(|c| crate::precision::mixed_module(c.name))
    {
        let matching = crate::GEOMETRY_CUBINS.iter().any(|(name, id, ranks)| {
            *name == cubin.name
                && (*id == "*" || *id == shape.geometry.id)
                && (*ranks == 0 || *ranks == shape.ranks())
        });
        for symbol in cubin.symbols {
            assert!(
                matching || !names.contains(*symbol),
                "TP{} resolved {symbol} from {}",
                shape.ranks(),
                cubin.name
            );
        }
    }
    for selection in &document.operations {
        for binding in &selection.bindings {
            for stage in &binding.implementation.stages {
                assert!(names.contains(stage), "TP{} lacks {stage}", shape.ranks());
            }
        }
    }
}

#[test]
fn prepared_mixed_ports_run_prefill_decode_and_replay_without_activation_quantizers() {
    for (nv, mx) in [(false, false), (true, false), (true, true)] {
        let host = Rc::new(FakeHost::default());
        *host.precision.borrow_mut() = Some(document(nv, mx));
        let mut instance = create(&host);
        let bindings = host.symbol_names.borrow().clone();
        run(
            &host,
            &mut instance,
            "prefill",
            step(&[(10, 2, 0, &PROMPT)]),
        );
        instance.commit().unwrap();
        for previous in [3, 4] {
            run(
                &host,
                &mut instance,
                "decode",
                step(&[(10, 2, previous, &[7])]),
            );
            instance.commit().unwrap();
        }
        assert_eq!(*host.symbol_names.borrow(), bindings);
        let names = launched(&host, 0);
        assert!(
            names
                .iter()
                .all(|n| !n.contains("quantiz") && !n.contains("pregather"))
        );
        assert_eq!(
            names
                .iter()
                .any(|n| n.starts_with("chop_qwen38_precision_nvfp4_gate_up")),
            nv
        );
        // The linear.a and linear.b copies into their fused 24-column row.
        let narrow_copies: Vec<_> = model_qwen38_contract::sm120::generated::inventory(Shape::TP4)
            .unwrap()
            .copies
            .into_iter()
            .filter(|copy| copy.layout().0 == 12)
            .map(|copy| copy.stage().entry().symbol.clone())
            .collect();
        assert!(!narrow_copies.is_empty());
        assert_eq!(names.iter().any(|n| narrow_copies.contains(n)), mx);
        assert_eq!(
            names
                .iter()
                .any(|n| n == "chop_qwen38_ngram_hash_prefill_v1"),
            mx
        );
        assert_eq!(
            names
                .iter()
                .any(|n| n == "chop_qwen38_ngram_hash_decode_v1"),
            mx
        );
        instance.retire_request(10).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.released.get(), host.acquired.get());
    }
}

/// The mixed ports at each topology: the instance resolves its mixed
/// symbols from the modules of its rank count only, and launches TP4's
/// adopted symbols or TP2's generated experts, copies and hash entries.
#[test]
fn prepared_mixed_ports_run_from_rank_matching_modules_at_each_topology() {
    for shape in [Shape::TP4, Shape::TP2] {
        for (nv, mx) in [(false, false), (true, false), (true, true)] {
            let host = Rc::new(FakeHost::at(shape));
            let document = document_for(shape, nv, mx, false);
            *host.precision.borrow_mut() = Some(document.clone());
            let mut instance = create(&host);
            let bindings = host.symbol_names.borrow().clone();
            assert_rank_matching_mixed_modules(&host, &document);
            run(
                &host,
                &mut instance,
                "prefill",
                step(&[(10, 2, 0, &PROMPT)]),
            );
            instance.commit().unwrap();
            for previous in [3, 4] {
                run(
                    &host,
                    &mut instance,
                    "decode",
                    step(&[(10, 2, previous, &[7])]),
                );
                instance.commit().unwrap();
            }
            assert_eq!(*host.symbol_names.borrow(), bindings);
            let names = launched(&host, 0);
            assert!(
                names
                    .iter()
                    .all(|n| !n.contains("quantiz") && !n.contains("pregather"))
            );
            // TP4 launches its adopted experts; TP2 its generated ones.
            let experts = if shape.ranks() == 4 {
                "chop_qwen38_precision_nvfp4_gate_up"
            } else {
                "chop_proj_nvfp4_grouped_n640_k2560_"
            };
            // The linear.a and linear.b copies of the rank's value heads into
            // their fused row.
            let narrow_copies: Vec<_> = model_qwen38_contract::sm120::generated::inventory(shape)
                .unwrap()
                .copies
                .into_iter()
                .filter(|copy| copy.layout().0 == shape.value_heads)
                .map(|copy| copy.stage().entry().symbol.clone())
                .collect();
            assert!(!narrow_copies.is_empty());
            let label = format!("TP{} nv={nv} mx={mx}", shape.ranks());
            assert_eq!(names.iter().any(|n| n.starts_with(experts)), nv, "{label}");
            assert_eq!(
                names.iter().any(|n| narrow_copies.contains(n)),
                mx,
                "{label}"
            );
            for phase in [ExecutionPhase::Prefill, ExecutionPhase::Decode] {
                let hash = model_qwen38_contract::sm120::generated::ngram_hash_symbol(shape, phase);
                assert_eq!(names.contains(&hash), mx, "{label} {hash}");
            }
            instance.retire_request(10).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.released.get(), host.acquired.get());
        }
    }
}

/// Mixed packages whose BF16 internal projections launch generated offers,
/// with prompts long enough that the prefill splits into two lanes (3072
/// tokens at TP4, 8192 at TP2): each pipelined launch's grid moves as the
/// symbol rule moves it (`sm120::overlap::compare_with_symbols`, which
/// panics on a difference), so a stored projection's own persistence cannot
/// shrink a grid the rule leaves alone (TP4's
/// `chop_qwen38_precision_bf16_n1536_k2560_linear_c8192_v1` went from 188
/// CTAs to 172 when it did).
#[test]
fn pipelined_bf16_projection_prefills_keep_the_symbol_rule_grids() {
    let _lanes = super::overlap_arm(true);
    for (shape, prompt) in [(Shape::TP4, 3072), (Shape::TP2, 8192)] {
        for nv in [false, true] {
            let host = Rc::new(FakeHost::at(shape));
            *host.precision.borrow_mut() = Some(document_for(shape, nv, true, true));
            let mut instance = create(&host);
            crate::sm120::overlap::COMPARED.with(|compared| compared.set([0; 4]));
            let tokens = super::speculation::truths(0..prompt);
            run(
                &host,
                &mut instance,
                "prefill",
                step(&[(10, 0, 0, &tokens)]),
            );
            let [all, shrunk, phased, comm] = crate::sm120::overlap::COMPARED.with(|c| c.get());
            assert!(
                all > 0 && shrunk > 0 && phased > 0 && comm > 0,
                "TP{} nv={nv}: {all} {shrunk} {phased} {comm}",
                shape.ranks()
            );
            instance.commit().unwrap();
            instance.retire_request(10).unwrap();
            instance.destroy().unwrap();
            assert_eq!(host.released.get(), host.acquired.get());
        }
    }
}

#[test]
fn altered_prepared_bindings_fail_before_device_acquisition() {
    let host = Rc::new(FakeHost::default());
    let mut doc = document(true, true);
    doc.operations[0].bindings[0].implementation.dependency = "0".repeat(64);
    *host.precision.borrow_mut() = Some(doc);
    let config = configuration_with(
        false,
        &[model_qwen38_contract::prepared::COMPONENT_ROLE
            .as_bytes()
            .to_vec()],
    );
    assert!(
        api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &config)
            .is_err()
    );
    assert!(host.symbol_names.borrow().is_empty());
    assert!(host.extents.borrow().is_empty());
    assert_eq!(host.released.get(), host.acquired.get());
}

/// The binder's first refusal, by operation ID, as the pre-port binder
/// ordered them. Per layer it fetches the four input fusion peers, then each
/// dense port, then refuses each grouped port other than FP8 block or NVFP4
/// experts; after the layers it fetches the lookups. Mixed compositions are
/// supplied at TP2 and TP4 alike.
fn first_refusal(document: &PrecisionDocument, shape: Shape) -> Option<String> {
    let operations: std::collections::BTreeMap<_, _> = document
        .operations
        .iter()
        .map(|s| (s.source.id.clone(), s))
        .collect();
    for layer in 0..shape.layers() {
        let ids: Vec<_> = DenseRole::roles_for(shape, layer)
            .unwrap()
            .into_iter()
            .map(|role| role.operation_id_for(shape, layer).unwrap())
            .collect();
        for id in ids[..4].iter().chain(&ids) {
            if !operations.contains_key(id) {
                return Some(id.clone());
            }
        }
        for name in ["gate_up_proj", "down_proj"] {
            let id = format!(
                "{}.mlp.experts.{name}",
                model_qwen38_contract::views::layer_stem_for(layer, shape)
            );
            let Some(s) = operations.get(&id) else {
                return Some(id);
            };
            match s.effective.weights[0].format {
                WeightFormat::Fp8Block128x128(_) | WeightFormat::Nvfp4 => {}
                _ => return Some(id),
            }
        }
    }
    for ngram in [false, true] {
        let id = if ngram {
            crate::views::ngram_view_for(shape)
                .trim_end_matches(".weight")
                .to_owned()
        } else {
            "model.language_model.embed_tokens".into()
        };
        if !operations.contains_key(&id) {
            return Some(id);
        }
    }
    None
}

/// Every supported format mix binds at TP2 and TP4, and a component
/// without one of its ports refuses at the port the pre-port binder
/// fetched first.
#[test]
fn prepared_refusals_keep_the_former_ports_and_order() {
    use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
    let mut operations = model_qwen38_contract::sm120::generated::all_capabilities().unwrap();
    operations.extend(ports::legacy_capabilities().unwrap());
    let capabilities = CapabilityDocument {
        version: 1,
        operations,
    };
    let inventory =
        CapabilityInventory::new(capabilities.clone(), capabilities.entries(), "0".repeat(64))
            .unwrap();
    for shape in [Shape::TP2, Shape::TP4] {
        let ngram_id = crate::views::ngram_view_for(shape)
            .trim_end_matches(".weight")
            .to_owned();
        let legacy = document_bound(shape, false, false, false, &inventory);
        let mixed = document_bound(shape, true, true, false, &inventory);
        let mut nvfp4_ngram = legacy.clone();
        let index = nvfp4_ngram
            .operations
            .iter()
            .position(|s| s.source.id == ngram_id)
            .unwrap();
        nvfp4_ngram.operations[index] = mixed
            .operations
            .iter()
            .find(|s| s.source.id == ngram_id)
            .unwrap()
            .clone();
        let mut cases = vec![
            ("FP8 legacy", legacy),
            (
                "NVFP4 experts",
                document_bound(shape, true, false, false, &inventory),
            ),
            ("MXFP8 dense, NVFP4 experts and lookup", mixed.clone()),
            (
                "MXFP8 dense and NVFP4 lookup",
                document_bound(shape, false, true, false, &inventory),
            ),
            (
                "MXFP8 fusion heads with BF16 peers",
                document_bound(shape, true, true, true, &inventory),
            ),
            ("NVFP4 lookup alone", nvfp4_ngram),
        ];
        for role in [
            DenseRole::LinearZ,
            DenseRole::LinearB,
            DenseRole::SharedDown,
        ] {
            let missing = role.operation_id_for(shape, 0).unwrap();
            let mut document = mixed.clone();
            document.operations.retain(|s| s.source.id != missing);
            cases.push(("mixed without one port", document));
        }
        for (label, document) in cases {
            let former = first_refusal(&document, shape);
            assert_eq!(
                former.is_none(),
                !label.contains("without"),
                "{label} at TP{}",
                shape.ranks()
            );
            let actual = model_qwen38_contract::prepared::bind(&document, shape, &inventory)
                .err()
                .map(|error| {
                    error
                        .strip_prefix("operation ")
                        .and_then(|rest| rest.split_once(": "))
                        .map_or(error.clone(), |(id, _)| id.to_owned())
                });
            assert_eq!(actual, former, "{label} at TP{}", shape.ranks());
        }
    }
}
