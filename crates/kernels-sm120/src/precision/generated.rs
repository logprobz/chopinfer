//! Projection offers from rank-local portable operations and explicit schedules.
//! Names contain no source tensor, declaration, model or topology identity.

use super::*;
use kernel_core::contract::{CompileKnobs, DenseKernel, SwigluColumns};
use kernel_core::precision::format::{A16Recipe, FormatId, KernelFamily, registry};
use kernel_core::precision::{Calibration, DynamicScale, ScaleScope};
use serde::{Deserialize, Serialize};

/// Schedule and addressing facts that are not inferable from a logical matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Implementation {
    Dense {
        kernel: DenseKernel,
        capacity: u32,
        knobs: Option<CompileKnobs>,
    },
    Grouped {
        phase: moe::Phase,
        capacity: u32,
        knobs: Option<CompileKnobs>,
        scales: moe::ScaleBindings,
        route_weight: moe::RouteWeight,
        swiglu_clamp: Option<moe::SwigluClamp>,
    },
    Lookup {
        heads: u32,
        ids: row::LookupIds,
        scale_dtype: Option<DType>,
        #[serde(default)]
        shapes: row::ShapeBinding,
    },
}

/// The caller has already applied TP slicing and physical padding to `operation`.
pub struct Request<'a> {
    pub operation: &'a OperationSpec,
    pub format: FormatId,
    pub usage: OperationUse,
    pub implementation: Implementation,
}

/// One immutable alias. The digest covers code, ABI, schedule and headers,
/// with a neutral symbol, so it can be checked before a legacy name is applied.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Adoption {
    pub symbol: String,
    pub generated: String,
    pub entry_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<Placement>,
    /// Retires the symbol: the entry that now serves its uses. A retired
    /// latest record has no placement and adopts nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replaced_by: Option<String>,
}

impl Adoption {
    pub fn retired(&self) -> bool {
        self.replaced_by.is_some()
    }
}

/// The latest record of each symbol that is not retired, by symbol.
pub fn active(ledger: &[Adoption]) -> BTreeMap<&str, &Adoption> {
    let mut latest = BTreeMap::new();
    for record in ledger {
        latest.insert(record.symbol.as_str(), record);
    }
    latest.retain(|_, record| !record.retired());
    latest
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub cubin: String,
    pub position: usize,
}

pub mod placement;
pub mod schedule;

#[derive(Clone, Debug)]
pub struct GeneratedOffer {
    pub offer: KernelOffer,
    pub identities: Vec<Adoption>,
}

fn fail(message: &str) -> String {
    format!("projection generation: {message}")
}

fn narrow(value: u64) -> Result<u32, String> {
    u32::try_from(value).map_err(|_| fail("matrix dimension exceeds u32"))
}

fn global(operation: &OperationSpec) -> Result<row::Nvfp4Global, String> {
    let Some(activation) = operation.activations.first() else {
        return Ok(row::Nvfp4Global::Absent);
    };
    if activation.format != ActivationFormat::Nvfp4 {
        return Ok(row::Nvfp4Global::Absent);
    }
    match activation.quantizer.as_ref().map(|q| &q.calibration) {
        Some(Calibration::Static(_)) => Ok(row::Nvfp4Global::Supplied),
        Some(Calibration::Dynamic(DynamicScale::Nvfp4TensorAmax)) => {
            Ok(row::Nvfp4Global::TensorAmax)
        }
        _ => Err(fail("unsupported activation calibration")),
    }
}

/// Generate, then bind the complete portable contract. Invalid recipes never
/// become offers merely because their shapes can instantiate a kernel.
pub fn offer(request: Request<'_>, adoptions: &[Adoption]) -> Result<GeneratedOffer, String> {
    let operation = request.operation;
    operation.validate().map_err(|e| e.to_string())?;
    if !operation.uses.contains(&request.usage) {
        return Err(fail("undeclared use"));
    }
    let plugin = registry::by_id(request.format).map_err(|e| e.to_string())?;
    for weight in &operation.weights {
        plugin.companions(weight).map_err(|e| e.to_string())?;
    }
    let geometry = operation.matrix_geometry().map_err(|e| e.to_string())?;
    let weight = operation.weights[0].format;
    let activation = operation.activations.first().map(|a| a.format);
    let global = global(operation)?;
    let family = match operation.kind {
        OperationKind::Dense => KernelFamily::Dense,
        OperationKind::Grouped { .. } => KernelFamily::GroupedExpert,
        OperationKind::Lookup => KernelFamily::Lookup,
        OperationKind::GroupedDense { .. } => {
            return Err(fail(
                "grouped dense operations run on the grouped fold GEMM contract, \
                 not a generated dense or expert kernel",
            ));
        }
    };
    let recipe = if operation.arithmetic.decode_order == DecodeOrder::BlockScaleAfterMma {
        A16Recipe::ScaleAfterBlockMma
    } else {
        A16Recipe::ScaledWeight
    };
    let order = plugin
        .decode_order(family, activation.unwrap_or(ActivationFormat::Bf16), recipe)
        .map_err(|e| e.to_string())?;
    if order != operation.arithmetic.decode_order {
        return Err(fail("plug-in arithmetic differs from the operation recipe"));
    }
    let build = |names: &[String]| -> Result<KernelOffer, String> {
        let result = match (&request.implementation, operation.kind) {
            (
                Implementation::Dense {
                    kernel,
                    capacity,
                    knobs,
                },
                OperationKind::Dense,
            ) => KernelOffer::dense(
                gemm::DenseRequest {
                    query: GemmQuery {
                        m_capacity: *capacity,
                        n: narrow(
                            geometry
                                .rows
                                .checked_mul(operation.weights.len() as u64)
                                .ok_or_else(|| fail("operand rows overflow"))?,
                        )?,
                        k: narrow(geometry.k)?,
                        weight,
                        activation: activation.ok_or_else(|| fail("missing activation"))?,
                        output: operation.arithmetic.output,
                        epilogues: operation.epilogues,
                        grouped: None,
                    },
                    kernel: *kernel,
                    target: operation.target,
                    symbol: names[0].clone(),
                    knobs: *knobs,
                },
                request.usage,
                global,
                &names[2],
            ),
            (
                Implementation::Grouped {
                    phase,
                    capacity,
                    knobs,
                    scales,
                    route_weight,
                    swiglu_clamp,
                },
                OperationKind::Grouped { experts, top_k },
            ) => {
                let gate = operation.epilogues.swiglu.is_some();
                let (hidden, intermediate) = if gate {
                    (narrow(geometry.k)?, narrow(geometry.rows)?)
                } else {
                    (narrow(geometry.rows)?, narrow(geometry.k)?)
                };
                KernelOffer::grouped(
                    moe::GroupedRequest {
                        shape: moe::Shape {
                            hidden,
                            intermediate,
                            experts,
                            top_k,
                        },
                        projection: if gate {
                            moe::Projection::GateUp
                        } else {
                            moe::Projection::Down
                        },
                        route_weight: *route_weight,
                        swiglu_clamp: *swiglu_clamp,
                        phase: *phase,
                        tokens_capacity: *capacity,
                        weight,
                        activation: activation.ok_or_else(|| fail("missing activation"))?,
                        scales: *scales,
                        output_quantization: operation.epilogues.output_quantization,
                        swiglu_columns: operation
                            .epilogues
                            .swiglu
                            .unwrap_or(SwigluColumns::ConcatenatedHalves),
                        target: operation.target,
                        symbol: names[0].clone(),
                        knobs: *knobs,
                    },
                    request.usage,
                    global,
                    &names[1],
                    &names[2],
                )
            }
            (
                Implementation::Lookup {
                    heads,
                    ids,
                    scale_dtype,
                    shapes,
                },
                OperationKind::Lookup,
            ) => KernelOffer::lookup(
                row::LookupRequest {
                    vocab: geometry.rows,
                    width: narrow(geometry.k)?,
                    heads: *heads,
                    capacity: request.usage.capacity,
                    weight,
                    global_scale: *scale_dtype,
                    ids: *ids,
                    shapes: *shapes,
                },
                request.usage,
                &names[0],
            ),
            _ => return Err(fail("implementation family differs from operation kind")),
        };
        let mut result = result.map_err(|e| e.to_string())?;
        // Calibration selects the row code before its identity and aliases are computed.
        if let Some(Calibration::Dynamic(rule)) = operation
            .activations
            .first()
            .and_then(|activation| activation.quantizer.as_ref())
            .map(|quantizer| &quantizer.calibration)
        {
            result = result
                .with_dynamic_scale(*rule, &names[2])
                .map_err(|e| e.to_string())?;
        }
        Ok(result)
    };
    let neutral = [
        "projection_compute_v1".into(),
        "projection_gather_v1".into(),
        "projection_quantize_v1".into(),
    ];
    let mut adopted = build(&neutral)?;
    let mut identities = identities(request.format, &adopted)?;
    let mut names = BTreeMap::new();
    for identity in &mut identities {
        let name = alias(identity, adoptions)?
            .unwrap_or(&identity.generated)
            .to_owned();
        names.insert(identity.symbol.clone(), name);
        identity.symbol.clone_from(&identity.generated);
    }
    match &mut adopted.handle.compute {
        Compute::Dense(spec) => spec.symbol.clone_from(&names[&spec.symbol]),
        Compute::Grouped(g) => g.symbol.clone_from(&names[&g.symbol]),
        Compute::Lookup(row) => row.rename(&names),
    }
    if let Some(gather) = &mut adopted.handle.gather {
        gather.rename(&names);
    }
    if let Some(quantizer) = &mut adopted.handle.quantizer {
        quantizer.rename(&names);
    }
    let document = capabilities(std::slice::from_ref(&adopted)).map_err(|e| e.to_string())?;
    let inventory = CapabilityInventory::new(document.clone(), document.entries(), "0".repeat(64))?;
    inventory.bind(operation, request.usage)?;
    Ok(GeneratedOffer {
        offer: adopted,
        identities,
    })
}

/// Append discoveries after the immutable historical prefix. Later records
/// supersede earlier records for the same old symbol.
pub fn append_adoptions(
    history: &[Adoption],
    incoming: impl IntoIterator<Item = Adoption>,
) -> Vec<Adoption> {
    let mut result = history.to_vec();
    for entry in incoming {
        if result.iter().rev().find(|old| old.symbol == entry.symbol) != Some(&entry) {
            result.push(entry);
        }
    }
    result
}

fn alias<'a>(identity: &Adoption, adoptions: &'a [Adoption]) -> Result<Option<&'a str>, String> {
    let mut latest = BTreeMap::new();
    for adoption in adoptions {
        if latest.insert(&adoption.symbol, adoption) == Some(adoption) {
            return Err(fail("duplicate current adoption"));
        }
    }
    let mut matches = latest
        .into_values()
        .filter(|a| !a.retired() && a.generated == identity.generated);
    let Some(adoption) = matches.next() else {
        return Ok(None);
    };
    if matches.next().is_some() || adoption.entry_sha256 != identity.entry_sha256 {
        return Err(fail("ambiguous or changed adopted entry"));
    }
    Ok(Some(&adoption.symbol))
}

pub mod identity;

/// Identity records in entry order, independent of wrapper names and Rust Debug.
pub fn identities(format: FormatId, offer: &KernelOffer) -> Result<Vec<Adoption>, String> {
    let capability = offer.capability().map_err(|e| e.to_string())?;
    let kind = match capability.kind {
        OperationKind::Dense => "dense",
        OperationKind::Grouped { .. } => "grouped",
        OperationKind::Lookup => "lookup",
        OperationKind::GroupedDense { .. } => {
            return Err(fail(
                "a generated offer never serves grouped dense operations",
            ));
        }
    };
    offer
        .handle
        .cubin_specs("projection")
        .map_err(|e| e.to_string())?
        .iter()
        .flat_map(|spec| spec.entries.iter().map(move |entry| (spec, entry)))
        .map(|(spec, entry)| {
            let launch = offer.handle.row_stages().iter().any(|s| {
                s.entry().symbol == entry.symbol && s.shapes() == row::ShapeBinding::Launch
            });
            Ok(identity::record(
                format,
                kind,
                (!launch).then_some(capability.geometry),
                spec,
                entry,
                identity::declaration(offer, &entry.symbol)?,
            ))
        })
        .collect()
}

/// Derive the grouped scale addressing directly from the imported operands.
pub fn scale_bindings(operation: &OperationSpec) -> Result<moe::ScaleBindings, String> {
    operation.validate().map_err(|e| e.to_string())?;
    let scope = operation.weights[0].global_scale.as_ref().map(|s| s.scope);
    let weight_globals = match scope {
        None => moe::GlobalScaleLayout::Absent,
        Some(ScaleScope::Tensor | ScaleScope::Projection) => moe::GlobalScaleLayout::Scalar,
        Some(ScaleScope::Expert) => moe::GlobalScaleLayout::ExpertProjection,
        _ => return Err(fail("unsupported grouped weight globals")),
    };
    Ok(moe::ScaleBindings {
        weight_globals,
        activation_globals: if global(operation)? == row::Nvfp4Global::Absent {
            moe::GlobalScaleLayout::Absent
        } else {
            moe::GlobalScaleLayout::Scalar
        },
        shared_gate_up_activation: operation.activations.windows(2).all(|a| a[0] == a[1]),
    })
}

/// Generate exact layout-copy code with the same identity rule as projections.
pub fn layout_copy(
    columns: u32,
    capacity: u32,
    source: row::CopyRows,
    destination: row::CopyRows,
    shapes: row::ShapeBinding,
    adoptions: &[Adoption],
) -> Result<(row::LayoutCopy, Adoption), String> {
    let build = |symbol: &str| {
        row::LayoutCopy::new(
            columns,
            capacity,
            source,
            destination,
            symbol,
            shapes,
            row::RowResources::SM120F,
        )
        .map_err(|e| e.to_string())
    };
    let prototype = build("projection_copy_v1")?;
    let spec = row::cubin_spec(
        "projection_copy_v1",
        std::slice::from_ref(prototype.stage()),
    )
    .map_err(|e| e.to_string())?;
    let identity = identity::record(
        FormatId::Bf16,
        "copy",
        (shapes == row::ShapeBinding::Compiled).then_some(MatrixGeometry {
            rows: u64::from(columns),
            k: 1,
        }),
        &spec,
        prototype.stage().entry(),
        prototype.stage().identity_declaration(),
    );
    let symbol = alias(&identity, adoptions)?.unwrap_or(&identity.generated);
    Ok((build(symbol)?, identity))
}

/// Prove initial aliases against an independently constructed legacy offer.
/// Cubin ownership names may differ; code, headers, defines, stages and portable
/// capabilities (apart from stage names) must match exactly.
pub fn adopt(generated: &GeneratedOffer, existing: &KernelOffer) -> Result<Vec<Adoption>, String> {
    let mut expected = generated.offer.capability().map_err(|e| e.to_string())?;
    let actual = existing.capability().map_err(|e| e.to_string())?;
    expected.stages.clone_from(&actual.stages);
    if expected != actual {
        return Err(fail("legacy capability differs from the generated offer"));
    }
    let left_declarations = generated
        .offer
        .handle
        .declarations()
        .map_err(|e| e.to_string())?;
    let right_declarations = existing.handle.declarations().map_err(|e| e.to_string())?;
    if left_declarations.len() != right_declarations.len() {
        return Err(fail("legacy stage count differs"));
    }
    for (mut expected, actual) in left_declarations.into_iter().zip(right_declarations) {
        expected.symbol.clone_from(&actual.symbol);
        if expected != actual {
            return Err(fail("legacy launch schedule or declaration differs"));
        }
    }
    let left = generated
        .offer
        .handle
        .cubin_specs("projection")
        .map_err(|e| e.to_string())?;
    let right = existing
        .handle
        .cubin_specs("projection")
        .map_err(|e| e.to_string())?;
    if left.len() != right.len() {
        return Err(fail("legacy cubin membership differs"));
    }
    for (a, b) in left.iter().zip(&right) {
        if a.headers != b.headers || a.defines != b.defines || a.entries.len() != b.entries.len() {
            return Err(fail("legacy compilation inputs differ"));
        }
        for (a, b) in a.entries.iter().zip(&b.entries) {
            let mut a = a.clone();
            a.symbol.clone_from(&b.symbol);
            if identity::entry(&a) != identity::entry(b) {
                return Err(fail("legacy entry code or ABI differs"));
            }
        }
    }
    Ok(generated
        .identities
        .iter()
        .zip(actual.stages)
        .map(|(identity, symbol)| Adoption {
            symbol,
            ..identity.clone()
        })
        .collect())
}

/// A rank-local operation domain before any checkpoint is imported. A
/// publisher derives it from its declaration to generate offers.
#[derive(Clone, Debug)]
pub struct Domain {
    pub kind: OperationKind,
    pub target: kernel_core::contract::Target,
    pub geometry: MatrixGeometry,
    pub operands: u8,
    pub weight: WeightFormat,
    /// Local scale storage; `None` selects the format's canonical dtype.
    pub local_scale: Option<DType>,
    /// Tensor scope is one F32 scalar; expert scope is one F32 per expert.
    pub global_scale: Option<ScaleScope>,
    pub activations: Vec<kernel_core::precision::ActivationSpec>,
    pub arithmetic: ArithmeticSpec,
    pub epilogues: kernel_core::contract::Epilogues,
    pub usage: OperationUse,
}

/// A valid portable operation over canonical row-major operands of the
/// domain. Its synthetic tensor spans describe shapes only; they are never
/// checkpoint import records.
pub fn domain(domain: Domain) -> Result<OperationSpec, String> {
    use kernel_core::precision::{
        CodeEncoding, GroupAxis, NibbleOrder, ScaleConvention, ScaleSpec, TensorSpan, WeightSpec,
        dtype_bits, shape_elements,
    };
    let mut shape = vec![domain.geometry.rows, domain.geometry.k];
    if let OperationKind::Grouped { experts, .. } = domain.kind {
        shape.insert(0, u64::from(experts));
    }
    let tensor = |name: &str, dtype, shape: Vec<u64>| -> Result<TensorSpan, String> {
        let elements = shape_elements(name, &shape).map_err(|e| e.to_string())?;
        let byte_length = elements
            .checked_mul(dtype_bits(dtype))
            .ok_or("fixture byte overflow")?
            .div_ceil(8);
        Ok(TensorSpan {
            tensor: name.into(),
            dtype,
            shape,
            offset: 0,
            byte_length,
        })
    };
    let packed = matches!(domain.weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp4);
    let mut codes = shape.clone();
    if packed {
        *codes.last_mut().ok_or("empty operand shape")? /= 2;
    }
    let dtype = if packed {
        DType::U8
    } else if domain.weight == WeightFormat::Bf16 {
        DType::Bf16
    } else {
        DType::Fp8E4M3
    };
    let (rows, k) = registry::by_format(domain.weight)
        .map_err(|e| e.to_string())?
        .local_scale_groups();
    let local_scale = if domain.weight == WeightFormat::Bf16 {
        None
    } else {
        let mut scales = shape.clone();
        let mut groups = vec![];
        if rows == 0 {
            scales.clear();
        } else {
            let rank = scales.len();
            scales[rank - 2] = scales[rank - 2].div_ceil(rows);
            scales[rank - 1] = if k == 0 {
                1
            } else {
                scales[rank - 1].div_ceil(k)
            };
            groups.push(GroupAxis {
                axis: (rank - 2) as u8,
                size: rows as u32,
                origin: 0,
            });
            if k != 0 {
                groups.push(GroupAxis {
                    axis: (rank - 1) as u8,
                    size: k as u32,
                    origin: 0,
                });
            }
        }
        let canonical = match domain.weight {
            WeightFormat::Nvfp4 => DType::Fp8E4M3,
            WeightFormat::Mxfp8 | WeightFormat::Mxfp4 => DType::Ue8M0,
            _ => DType::F32,
        };
        Some(ScaleSpec {
            tensor: tensor("local", domain.local_scale.unwrap_or(canonical), scales)?,
            scope: if rows == 0 {
                ScaleScope::Tensor
            } else if k == 0 {
                ScaleScope::Channel
            } else {
                ScaleScope::Block
            },
            convention: ScaleConvention::Multiply,
            groups,
        })
    };
    let global_scale = match domain.global_scale {
        None => None,
        Some(scope @ (ScaleScope::Tensor | ScaleScope::Expert)) => Some(ScaleSpec {
            tensor: tensor(
                "global",
                DType::F32,
                if scope == ScaleScope::Expert {
                    vec![shape[0]]
                } else {
                    vec![]
                },
            )?,
            scope,
            convention: ScaleConvention::Multiply,
            groups: vec![],
        }),
        Some(_) => return Err(fail("unsupported domain global scale scope")),
    };
    let weight = WeightSpec {
        codes: tensor("weight", dtype, codes)?,
        logical_shape: shape,
        format: domain.weight,
        encoding: if packed {
            CodeEncoding::E2M1(NibbleOrder::EvenLow)
        } else if dtype == DType::Bf16 {
            CodeEncoding::Bf16LittleEndian
        } else {
            CodeEncoding::E4M3
        },
        layout: WeightLayout::row_major(),
        local_scale,
        global_scale,
        transforms: vec![],
    };
    let operation = OperationSpec {
        id: "projection-domain".into(),
        kind: domain.kind,
        target: domain.target,
        weights: vec![weight; usize::from(domain.operands)],
        geometry: Some(domain.geometry),
        activations: domain.activations,
        arithmetic: domain.arithmetic,
        epilogues: domain.epilogues,
        uses: vec![domain.usage],
    };
    operation.validate().map_err(|e| e.to_string())?;
    Ok(operation)
}

pub mod comparison;
