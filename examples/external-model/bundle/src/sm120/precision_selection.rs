//! Model-owned finite construction list. The projection declaration supplies
//! every shape, kind, fused operand count, epilogue and use; the shared
//! generator builds and checks each offer. This file chooses only the
//! activation variants and schedules the example publishes; [`backend`]
//! prepares the example's operations from them.
use super::precision_contract as model;
use kernel_build::{BuildContext, KernelSelection};
use kernel_core::contract::{ActivationFormat, DType, DenseKernel, Target, WeightFormat};
use kernel_core::precision::format::{A16Recipe, FormatId, KernelFamily, registry};
use kernel_core::precision::{
    ActivationSpec, ArithmeticSpec, Calibration, Clipping, GroupAxis, MatrixGeometry,
    OperationKind, OperationUse, QuantizerSpec, Rounding, ScaleConvention, ScaleScope, ScaleSpec,
    TensorSpan,
};
use kernels_sm120::{
    backend::{DenseForm, OfferList, ProjectionOffer, Sm120},
    moe,
    precision::{
        KernelOffer,
        generated::{self, Adoption, Domain, Implementation, Request},
        selected_cubins_at,
    },
    row::precision as row,
};
use projection_spec::{Projection, ProjectionKind};
use std::collections::BTreeSet;
use std::path::Path;

/// The append-only symbol adoptions. Generated identities keep the names
/// `precision.lock` records, so the cubins stay byte-identical.
pub fn adoptions() -> Result<Vec<Adoption>, String> {
    serde_json::from_str(include_str!("../../projection-adoptions.json")).map_err(|e| e.to_string())
}

/// The adoptions of one capacity's offers. A row stage's identity omits its
/// capacity, so the `_t1` and `_t64` entries of one row body share a
/// generated identity; each capacity keeps the symbol `precision.lock`
/// records for it.
fn adoptions_at(adoptions: &[Adoption], capacity: u32) -> Vec<Adoption> {
    let suffix = format!("_t{capacity}_v1");
    adoptions
        .iter()
        .filter(|a| a.symbol.ends_with(&suffix))
        .cloned()
        .collect()
}

/// NVFP4 activations with the checkpoint's static scalar calibration.
fn calibrated_nvfp4() -> ActivationSpec {
    ActivationSpec {
        format: ActivationFormat::Nvfp4,
        storage: DType::Bf16,
        compute: DType::Fp4E2M1,
        quantizer: Some(QuantizerSpec {
            calibration: Calibration::Static(ScaleSpec {
                tensor: TensorSpan {
                    tensor: "input_scale".into(),
                    dtype: DType::F32,
                    shape: vec![],
                    offset: 0,
                    byte_length: 4,
                },
                scope: ScaleScope::Projection,
                convention: ScaleConvention::Multiply,
                groups: vec![],
            }),
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

/// The weight-global scopes offered for a projection: grouped A16 entries
/// also read one global per expert and operand. NVFP4 activations need
/// bank-uniform globals.
fn weight_globals(p: &Projection, activation: ActivationFormat) -> Vec<ScaleScope> {
    match p.kind {
        ProjectionKind::ExpertBank { .. } if activation == ActivationFormat::Bf16 => {
            vec![ScaleScope::Tensor, ScaleScope::Expert]
        }
        _ => vec![ScaleScope::Tensor],
    }
}

/// One generated offer: the declared projection's rank-local domain, with
/// the given activations and weight globals, at `usage`.
fn offer(
    p: &Projection,
    activation: ActivationFormat,
    globals: ScaleScope,
    usage: OperationUse,
    adoptions: &[Adoption],
) -> Result<KernelOffer, String> {
    let format = p.format.ok_or("example projections declare their format")?;
    if format != FormatId::Nvfp4 {
        return Err(format!(
            "projection {}: the example offers NVFP4 kernels only",
            p.id
        ));
    }
    let plugin = registry::by_id(format).map_err(|e| e.to_string())?;
    let (kind, family) = match &p.kind {
        ProjectionKind::Dense {} => (OperationKind::Dense, KernelFamily::Dense),
        ProjectionKind::ExpertBank { experts, top_k } => (
            OperationKind::Grouped {
                experts: u32::try_from(model::dim(experts)?).map_err(|e| e.to_string())?,
                top_k: u32::try_from(model::dim(top_k)?).map_err(|e| e.to_string())?,
            },
            KernelFamily::GroupedExpert,
        ),
        ProjectionKind::LookupTable {} => (OperationKind::Lookup, KernelFamily::Lookup),
        ProjectionKind::GroupedDense { .. } => {
            return Err(format!(
                "projection {}: the example offers no grouped dense kernels",
                p.id
            ));
        }
    };
    let operands = u8::try_from(p.operand_count()).map_err(|e| e.to_string())?;
    let spec = match activation {
        ActivationFormat::Nvfp4 => calibrated_nvfp4(),
        _ => ActivationSpec::bf16(),
    };
    let operation = generated::domain(Domain {
        kind,
        target: Target::Sm120f,
        geometry: MatrixGeometry {
            rows: model::dim(&p.shape[0])?,
            k: model::dim(&p.shape[1])?,
        },
        operands,
        weight: WeightFormat::Nvfp4,
        local_scale: None,
        global_scale: Some(globals),
        activations: if family == KernelFamily::Lookup {
            vec![]
        } else {
            vec![spec.clone(); usize::from(operands)]
        },
        arithmetic: ArithmeticSpec {
            weight_operand: spec.compute,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: plugin
                .decode_order(family, activation, A16Recipe::ScaledWeight)
                .map_err(|e| e.to_string())?,
            rounding: Rounding::NearestEven,
            numerical_contract: p.numerical_contract.clone(),
        },
        epilogues: p.epilogues,
        usage,
    })?;
    let implementation = match kind {
        OperationKind::Dense => Implementation::Dense {
            kernel: DenseKernel::Tiled { splits: 1 },
            capacity: usage.capacity,
            knobs: None,
        },
        OperationKind::Grouped { .. } => Implementation::Grouped {
            phase: if usage.capacity == 1 {
                moe::Phase::Decode
            } else {
                moe::Phase::Prefill
            },
            capacity: usage.capacity,
            knobs: None,
            scales: generated::scale_bindings(&operation)?,
            route_weight: moe::RouteWeight::AfterDown,
            swiglu_clamp: None,
        },
        OperationKind::Lookup => Implementation::Lookup {
            heads: 1,
            ids: row::LookupIds::I32,
            scale_dtype: Some(DType::F32),
            shapes: row::ShapeBinding::Compiled,
        },
        OperationKind::GroupedDense { .. } => {
            return Err(format!(
                "projection {}: the example offers no grouped dense kernels",
                p.id
            ));
        }
    };
    Ok(generated::offer(
        Request {
            operation: &operation,
            format,
            usage,
            implementation,
        },
        adoptions,
    )?
    .offer)
}

/// Every offer in a stable order, which is also the order of the entries in
/// each cubin: per use, the activation-quantized then the A16 projections,
/// then the lookups.
pub fn offers() -> Result<Vec<KernelOffer>, String> {
    let declaration = model::declaration()?;
    let ledger = adoptions()?;
    let (lookups, projections): (Vec<_>, Vec<_>) = declaration
        .projections
        .iter()
        .partition(|p| p.kind == ProjectionKind::LookupTable {});
    let mut offers = Vec::new();
    for usage in model::uses() {
        let adoptions = adoptions_at(&ledger, usage.capacity);
        for activation in [ActivationFormat::Nvfp4, ActivationFormat::Bf16] {
            for p in &projections {
                for globals in weight_globals(p, activation) {
                    offers.push(offer(p, activation, globals, usage, &adoptions)?);
                }
            }
        }
        for p in &lookups {
            offers.push(offer(
                p,
                ActivationFormat::Bf16,
                ScaleScope::Tensor,
                usage,
                &adoptions,
            )?);
        }
    }
    Ok(offers)
}

pub fn selection(model: &Path, cuda: &Path) -> Result<KernelSelection, String> {
    KernelSelection::new(
        BuildContext::new(
            "example-bundle@0.1.0",
            "external-model",
            "example.precision.v1",
        )
        .source_root("external-model", model)
        .source_root("chopinfer", cuda.join("../../..")),
        selected_cubins_at("external_precision", &offers()?, cuda)?,
    )
    .map_err(|e| e.to_string())
}

/// The SM120 backend over `offers`, whose compiled entries are `available`,
/// under the native `dependency` digest their publication authenticates.
/// Dense offers launch in the tensor form, over an output of their capacity.
pub fn backend(
    offers: Vec<KernelOffer>,
    available: BTreeSet<String>,
    dependency: String,
) -> Result<Sm120, String> {
    let offers = offers
        .into_iter()
        .map(|offer| ProjectionOffer {
            offer,
            copy: None,
            dense: DenseForm::Tensor,
            live_output: false,
        })
        .collect();
    let offers = OfferList::new(offers, available, dependency).map_err(|e| e.to_string())?;
    Ok(Sm120::new().with_projections(Box::new(offers)))
}
