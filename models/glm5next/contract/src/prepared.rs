//! The precision component (`chop.bundle-metadata.glm5next.precision.v1`):
//! one rank's lowered projections derived from the declaration over the
//! export's admitted operands, each resolved under the BF16 activation
//! policy and bound to the entry the selected backend serves it with
//! (`crate::backend::Kernels::bind`). The compiler writes it; the bundle
//! authenticates it at create and rebinds every use to the same entries.

use std::collections::{BTreeMap, BTreeSet};

use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
use kernel_core::precision::execution::{
    Binding, ImplementationProvider, PrecisionDocument, PrecisionSelection,
};
use kernel_core::precision::{OperationSpec, OperationUse, TensorSpan};
use model_compiler_sdk::quantization::declared::{self, Admitted};
use model_compiler_sdk::quantization::import::{
    EncodingRecipe, ExportMetadata, LINEAR_CLASS_ROLE, OperandRole, companion_names, import_operand,
};
use model_compiler_sdk::quantization::{ActivationPolicy, resolve_operation};
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{RankDerivation, Topology, hooks::VariantDeriver};

use kernel_core::weight_transform::WeightTransform;

use crate::backend::{Kernels, Selected};
use crate::catalog::{DType, checkpoint_catalog};
use crate::contract::Shape;
use crate::direct::Stored;
use crate::precision::LOWERED;
use crate::quantization::Layout;

pub const COMPONENT_ROLE: &str = "chop.bundle-metadata.glm5next.precision.v1";
pub const COMPONENT_FORMAT: &str = kernel_core::precision::component::COMPONENT_FORMAT;
/// The rank image's layout of projection weights the offers read
/// (`crate::direct`): the rank's checkpoint rows, unchanged.
const CHECKPOINT_ROWS: &str = "checkpoint_rows_v1";

/// The lowered projections derive no variant.
struct NoVariants;
impl VariantDeriver for NoVariants {}

fn code_dtype(dtype: DType) -> Result<kernel_core::contract::DType, String> {
    use kernel_core::contract::DType as Code;
    Ok(match dtype {
        DType::Bf16 => Code::Bf16,
        DType::F8E4M3 => Code::Fp8E4M3,
        DType::F32 => Code::F32,
        DType::U8 => Code::U8,
    })
}

/// The declaration restricted to the lowered projections, with its geometry.
fn lowered() -> Result<&'static (projection_spec::Declaration, projection_spec::Geometry), String> {
    use std::sync::OnceLock;
    static LOWERED_DECLARATION: OnceLock<
        Result<(projection_spec::Declaration, projection_spec::Geometry), String>,
    > = OnceLock::new();
    LOWERED_DECLARATION
        .get_or_init(|| {
            let ranks = Shape::TP4.ranks();
            let geometry = crate::projections::geometry(ranks).map_err(|e| e.to_string())?;
            // Each projection's uses from its use profile.
            let mut declaration = crate::projections::declaration(ranks)
                .and_then(|d| d.expanded_profiles(&geometry))
                .map_err(|e| e.to_string())?;
            declaration
                .projections
                .retain(|p| LOWERED.contains(&p.id.as_str()));
            declaration.side_tensors.clear();
            declaration.external_tensors.clear();
            Ok((declaration, geometry))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Every checkpoint tensor of an export stored in `layout`, described as
/// the checkpoint holds it (whole, not a rank's share), by name.
fn descriptors(layout: &Layout) -> Result<BTreeMap<String, TensorSpan>, String> {
    checkpoint_catalog(&crate::geometry::FLASH, layout)
        .iter()
        .map(|spec| {
            Ok((
                spec.name.clone(),
                TensorSpan {
                    tensor: spec.name.clone(),
                    dtype: code_dtype(spec.dtype)?,
                    shape: spec.shape.clone(),
                    offset: 0,
                    byte_length: spec.bytes(),
                },
            ))
        })
        .collect()
}

/// The lowered projections' source weights in an export stored in
/// `layout`, imported through the shared importer with their companion
/// scales, by checkpoint name.
pub fn admitted(layout: &Layout, metadata: &ExportMetadata) -> Result<Admitted, String> {
    let g = &crate::geometry::FLASH;
    let catalog = checkpoint_catalog(g, layout);
    let tensors = descriptors(layout)?;
    let mut admitted = Admitted::new();
    for spec in &catalog {
        let Some(crate::views::Binding::Projection(id)) = crate::views::binding(g, &spec.name)
        else {
            continue;
        };
        if !LOWERED.contains(&id) {
            continue;
        }
        let Some(stem) = spec.name.strip_suffix(".weight") else {
            continue;
        };
        let prefix = format!("{stem}.");
        let companions = tensors
            .range(prefix.clone()..)
            .take_while(|(name, _)| name.starts_with(&prefix))
            .map(|(name, span)| (name.clone(), span.clone()))
            .collect();
        // U8 codes hold two E2M1 values per byte along K.
        let mut logical_shape = spec.shape.clone();
        if spec.dtype == DType::U8
            && let Some(k) = logical_shape.last_mut()
        {
            *k *= 2;
        }
        let operand = import_operand(
            metadata,
            &OperandRole {
                operation: stem.into(),
                weight: spec.name.clone(),
                // A lowered projection is a `Linear` module.
                targets: vec![stem.into(), LINEAR_CLASS_ROLE.into()],
                logical_shape,
                lookup: false,
                shared_scale_stem: None,
                // The export leaves these unconverted; the catalog, which
                // the weight plan checks against the headers, stores BF16.
                declared: (spec.dtype == DType::Bf16).then(EncodingRecipe::bf16),
            },
            &companions,
        )
        .map_err(|e| format!("{}: {e}", spec.name))?;
        admitted.insert(spec.name.clone(), operand);
    }
    Ok(admitted)
}

/// Rank `rank`'s derivation of the lowered projections over the tensors
/// their admitted operands read (the binding claims every tensor it is
/// shown).
pub fn derive(layout: &Layout, admitted: &Admitted, rank: u32) -> Result<RankDerivation, String> {
    let (declaration, geometry) = lowered()?;
    let read: BTreeSet<&str> = admitted
        .values()
        .flat_map(|operand| operand.source_tensors.iter().map(String::as_str))
        .collect();
    let metadata = CheckpointMetadata {
        // A static NVFP4 input scale is ignored: activations stay BF16.
        tensors: checkpoint_catalog(&crate::geometry::FLASH, layout)
            .iter()
            .filter(|spec| read.contains(spec.name.as_str()) && !spec.ignored)
            .map(|spec| {
                (
                    spec.name.clone(),
                    TensorInfo {
                        dtype: spec.dtype.name().into(),
                        shape: spec.shape.clone(),
                        file: None,
                    },
                )
            })
            .collect(),
        ..Default::default()
    };
    declared::derive(declared::RankInputs {
        declaration,
        geometry,
        registry: &crate::projections::registry().map_err(|e| e.to_string())?,
        variants: &NoVariants,
        metadata: &metadata,
        admitted,
        topology: Topology {
            ranks: Shape::TP4.ranks(),
            rank,
        },
    })
    .map_err(|e| e.to_string())
}

/// The entry a lowered projection use binds (`backend::Kernels::Entry`).
pub type Offer = <Selected as Kernels>::Entry;

/// The entry a derived `operation` binds at `usage`, and the identity
/// its capability binds the operation to. The bundle's provider rebinds
/// through this as the compiler's does.
pub fn bind(operation: &OperationSpec, usage: OperationUse) -> Result<Binding<Offer>, String> {
    let (declaration, _) = lowered()?;
    let (projection, _) = declaration
        .operation_instance(&operation.id)
        .map_err(|e| e.to_string())?;
    let id = LOWERED
        .iter()
        .find(|&&id| id == projection.id)
        .ok_or_else(|| format!("{} is not a lowered projection", projection.id))?;
    let weight = operation
        .weights
        .first()
        .ok_or("an operation without weights")?
        .format;
    let bound = Selected::bind(id, weight, usage)?;
    let stages: BTreeSet<String> = bound.capability.stages.iter().cloned().collect();
    let inventory = CapabilityInventory::new(
        CapabilityDocument {
            version: 1,
            operations: vec![bound.capability],
        },
        stages,
        bound.dependency,
    )?;
    let binding = inventory.bind(operation, usage)?;
    Ok(Binding {
        identity: binding.identity,
        handle: bound.entry,
    })
}

struct Offers;

impl ImplementationProvider for Offers {
    type Handle = Offer;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<Offer>, String> {
        bind(operation, usage)
    }
}

/// Rank `rank`'s precision document over an export in `layout`: every
/// lowered projection operation resolved under the BF16 activation policy
/// at each declared use.
pub fn document(
    layout: &Layout,
    metadata: &ExportMetadata,
    rank: u32,
) -> Result<PrecisionDocument, String> {
    let admitted = admitted(layout, metadata)?;
    let graph = derive(layout, &admitted, rank)?;
    let (declaration, geometry) = lowered()?;
    let mut operations = Vec::new();
    for mut source in graph.operation_specs().map_err(|e| e.to_string())? {
        declared::admitted_activations(&mut source, &admitted, declaration, geometry)
            .map_err(|e| e.to_string())?;
        let prepared = resolve_operation(&source, ActivationPolicy::Bf16, &Offers)
            .map_err(|e| e.to_string())?;
        operations.push(prepared.selection().clone());
    }
    let document = PrecisionDocument {
        version: 1,
        operations,
    };
    document.validate().map_err(|e| e.to_string())?;
    Ok(document)
}

/// The component's bytes.
pub fn encode(document: &PrecisionDocument) -> Result<Vec<u8>, String> {
    kernel_core::precision::component::encode(document).map_err(|e| e.to_string())
}

/// Checks that `document`, derived from an export stored in `layout`,
/// holds exactly the lowered projections' operations, each at every use
/// its profile declares and reading its instance's declared checkpoint
/// tensors (`sources`), as the bundle requires before it binds any. The
/// bindings themselves are authenticated against the projections' kernels
/// separately (`bind`), and the rank's storage of each tensor at binding
/// (`resident`).
pub fn check(document: &PrecisionDocument, layout: &Layout) -> Result<(), String> {
    let (declaration, geometry) = lowered()?;
    let expected = declaration
        .operations(geometry, |_| true)
        .map_err(|e| e.to_string())?;
    if document.operations.len() != expected.len() {
        return Err(format!(
            "the precision component holds {} operations, the lowered projections {}",
            document.operations.len(),
            expected.len()
        ));
    }
    for (id, uses) in expected {
        let selection = document
            .operations
            .iter()
            .find(|s| s.source.id == id)
            .ok_or_else(|| format!("the precision component lacks {id}"))?;
        if selection
            .bindings
            .iter()
            .map(|b| b.usage)
            .collect::<Vec<_>>()
            != uses
        {
            return Err(format!("{id} lacks some of its declared uses"));
        }
    }
    let descriptors = descriptors(layout)?;
    for selection in &document.operations {
        sources(selection, declaration, geometry, &descriptors)?;
    }
    Ok(())
}

/// Checks that `selection`, as derived and as resolved, reads exactly the
/// checkpoint weights its projection instance declares, and for each the
/// companion scales the importer reads under its format, every tensor
/// described as the checkpoint holds it (`descriptors`: dtype, shape and
/// length of the whole tensor; a rank's share is checked at binding).
fn sources(
    selection: &PrecisionSelection,
    declaration: &projection_spec::Declaration,
    geometry: &projection_spec::Geometry,
    descriptors: &BTreeMap<String, TensorSpan>,
) -> Result<(), String> {
    let id = &selection.source.id;
    let (projection, layer) = declaration
        .operation_instance(id)
        .map_err(|e| e.to_string())?;
    let declared = declaration
        .instance_sources(geometry, &projection.id, layer)
        .map_err(|e| e.to_string())?;
    if selection.effective.id != *id {
        return Err(format!("{id} resolves as {}", selection.effective.id));
    }
    let described = |span: &TensorSpan| descriptors.get(&span.tensor) == Some(span);
    for operation in [&selection.source, &selection.effective] {
        let names: Vec<&str> = operation
            .weights
            .iter()
            .map(|weight| weight.codes.tensor.as_str())
            .collect();
        if names != declared {
            return Err(format!(
                "{id} reads {names:?}, its declaration {declared:?}"
            ));
        }
        for weight in &operation.weights {
            let codes = &weight.codes.tensor;
            if !described(&weight.codes) {
                return Err(format!("{id}: {codes} differs from the checkpoint's"));
            }
            let companions =
                companion_names(weight.format, codes, None).map_err(|e| format!("{id}: {e}"))?;
            let local = match &weight.local_scale {
                None => companions.local.is_empty(),
                Some(scale) => {
                    companions.local.contains(&scale.tensor.tensor) && described(&scale.tensor)
                }
            };
            let global = match &weight.global_scale {
                None => companions.global.is_none(),
                Some(scale) => {
                    companions.global.as_ref() == Some(&scale.tensor.tensor)
                        && described(&scale.tensor)
                }
            };
            if !local || !global {
                return Err(format!("{id} reads scales other than {codes}'s"));
            }
            weight.validate().map_err(|e| format!("{id}: {e}"))?;
        }
    }
    Ok(())
}

/// The projection and execution layer of a lowered operation's identity
/// (`linear.q.0.bf16` is linear.q's at layer 0; `head.f32` has no layer).
pub fn instance(id: &str) -> Result<(String, Option<u64>), String> {
    let (declaration, _) = lowered()?;
    let (projection, layer) = declaration
        .operation_instance(id)
        .map_err(|e| e.to_string())?;
    Ok((projection.id.clone(), layer))
}

/// One operation's rank-local storage: the views holding its codes, its
/// local scales and its global scale, by name, with the bytes its GEMM
/// reads from each.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Resident {
    pub codes: (String, u64),
    pub scales: Option<(String, u64)>,
    pub global: Option<(String, u64)>,
}

/// Checks that a rank stores `selection`'s weight as its offer reads it,
/// from the rank's verified views (`stored`, by name): the codes' share of
/// the checkpoint tensor alone in its view, copied unchanged as checkpoint
/// rows in the codes' dtype, with exactly the operation's rank-local
/// geometry (the derivation pads BF16 rows to the format's row tile, so
/// the view must hold the padded rows), and its block scales likewise over
/// that geometry's scale grid. The tensor descriptors in `selection` are
/// the whole checkpoint's (`check`); only the views are rank-local.
pub fn resident(
    selection: &PrecisionSelection,
    stored: &BTreeMap<&str, &Stored>,
) -> Result<Resident, String> {
    let id = &selection.source.id;
    let operation = &selection.effective;
    let geometry = operation
        .geometry
        .ok_or_else(|| format!("{id} has no rank-local geometry"))?;
    let [weight] = operation.weights.as_slice() else {
        return Err(format!("{id} reads {} weights", operation.weights.len()));
    };
    let view = |span: &TensorSpan, shape: Vec<u64>| -> Result<(String, u64), String> {
        let name = span.tensor.as_str();
        let entry = stored
            .get(name)
            .ok_or_else(|| format!("{id}: the rank stores no view {name}"))?;
        let [piece] = entry.pieces.as_slice() else {
            return Err(format!("{id}: {name} is packed with other tensors"));
        };
        if piece.tensor != name
            || entry.transform != WeightTransform::Copy
            || entry.layout_id != CHECKPOINT_ROWS
            || code_dtype(entry.dtype)? != span.dtype
        {
            return Err(format!(
                "{id}: {name} is stored as {} {:?} {}, not as its checkpoint rows",
                entry.layout_id, entry.dtype, piece.tensor
            ));
        }
        if entry.view.shape != shape
            || piece.shape.len() != 2
            || piece.shape[0] > shape[0]
            || piece.shape[1] != shape[1]
        {
            return Err(format!(
                "{id}: {name} stores {:?} of the {shape:?} its offer reads",
                entry.view.shape
            ));
        }
        Ok((name.into(), entry.bytes()))
    };
    // U8 codes hold two E2M1 values per byte along K.
    let packed = if weight.codes.dtype == kernel_core::contract::DType::U8 {
        2
    } else {
        1
    };
    let codes = view(&weight.codes, vec![geometry.rows, geometry.k / packed])?;
    let scales = weight
        .local_scale
        .as_ref()
        .map(|scale| {
            let mut grid = vec![geometry.rows, geometry.k];
            for group in &scale.groups {
                let axis = usize::from(group.axis);
                if axis >= grid.len() || group.origin != 0 {
                    return Err(format!("{id}: an unsupported scale grid"));
                }
                grid[axis] = grid[axis].div_ceil(u64::from(group.size));
            }
            view(&scale.tensor, grid)
        })
        .transpose()?;
    // A tensor global is one F32 every rank holds whole, as stored.
    let global = weight
        .global_scale
        .as_ref()
        .map(|scale| {
            let name = scale.tensor.tensor.as_str();
            let entry = stored
                .get(name)
                .ok_or_else(|| format!("{id}: the rank stores no view {name}"))?;
            if !entry.view.shape.is_empty()
                || entry.transform != WeightTransform::Copy
                || entry.dtype != DType::F32
                || scale.tensor.dtype != kernel_core::contract::DType::F32
            {
                return Err(format!("{id}: {name} is not stored as its F32 scalar"));
            }
            Ok((name.to_owned(), entry.bytes()))
        })
        .transpose()?;
    Ok(Resident {
        codes,
        scales,
        global,
    })
}

/// The precision component a package of the checkpoint whose `config.json`
/// is `config`, stored in `layout`, carries: rank 0's document, which is
/// every rank's (each rank binds its own views to it, `resident`). The
/// config's quantization metadata is read as admission reads it
/// (`crate::quantization::export_metadata`).
pub fn checkpoint_component(
    layout: &Layout,
    config: &serde_json::Value,
) -> Result<Vec<u8>, String> {
    let metadata = crate::quantization::export_metadata(config.get("quantization_config"))?;
    encode(&document(layout, &metadata, 0)?)
}

/// The document of component `bytes`, checked to serve every lowered
/// projection from its declared tensors of the block-FP8 checkpoint
/// (`check`).
pub fn decode(bytes: &[u8]) -> Result<PrecisionDocument, String> {
    decode_for(bytes, &crate::quantization::FP8_BLOCK128)
}

/// [`decode`] for a checkpoint stored in `layout`.
pub fn decode_for(bytes: &[u8], layout: &Layout) -> Result<PrecisionDocument, String> {
    let document = kernel_core::precision::component::decode(bytes).map_err(|e| e.to_string())?;
    check(&document, layout)?;
    Ok(document)
}

/// The component of the pinned FP8 test checkpoint
/// (`crate::direct::test_support`).
#[cfg(any(test, feature = "test-support"))]
pub fn test_component() -> Vec<u8> {
    use std::sync::OnceLock;
    static COMPONENT: OnceLock<Vec<u8>> = OnceLock::new();
    COMPONENT
        .get_or_init(|| {
            let config: serde_json::Value =
                serde_json::from_slice(include_bytes!("../fixtures/config.json"))
                    .expect("fixture config");
            checkpoint_component(&crate::quantization::FP8_BLOCK128, &config)
                .expect("fixture precision component")
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::FP8_BLOCK128;

    /// A document missing an operation, or one of an operation's uses,
    /// is refused.
    #[test]
    fn incomplete_documents_are_refused() {
        let document = kernel_core::precision::component::decode(&test_component()).unwrap();
        let mut missing = document.clone();
        missing.operations.pop();
        assert!(check(&missing, &FP8_BLOCK128).is_err());
        let mut partial = document;
        partial.operations[0].bindings.pop();
        assert!(
            check(&partial, &FP8_BLOCK128)
                .unwrap_err()
                .contains("lacks some")
        );
    }

    /// An operation that reads a tensor other than its instance's declared
    /// weight, another instance's scales of the same size, or a tensor under
    /// a descriptor other than the checkpoint's is refused, whether the
    /// derived or the resolved operation says so.
    #[test]
    fn operations_read_their_declared_checkpoint_tensors() {
        let document = kernel_core::precision::component::decode(&test_component()).unwrap();
        let edited = |id: &str, edit: &dyn Fn(&mut OperationSpec)| {
            let mut document = document.clone();
            let selection = document
                .operations
                .iter_mut()
                .find(|s| s.source.id == id)
                .unwrap();
            edit(&mut selection.source);
            edit(&mut selection.effective);
            check(&document, &FP8_BLOCK128).unwrap_err()
        };
        // A side tensor of the layer as the query's codes.
        let side = "model.language_model.layers.0.self_attn.A_log";
        let error = edited("linear.q.0.bf16", &|op| {
            op.weights[0].codes.tensor = side.into();
        });
        assert!(error.contains("its declaration"), "{error}");
        // Layer 7's scales, the same size as layer 3's.
        let scale = "model.language_model.layers.7.self_attn.q_b_proj.weight_scale_inv";
        let error = edited("latent.q-b.3.bf16", &|op| {
            op.weights[0].local_scale.as_mut().unwrap().tensor.tensor = scale.into();
        });
        assert!(error.contains("scales other than"), "{error}");
        // The declared weight under another shape.
        let error = edited("latent.q-a.3.bf16", &|op| {
            let codes = &mut op.weights[0].codes;
            codes.shape.swap(0, 1);
        });
        assert!(error.contains("differs from the checkpoint's"), "{error}");
        // Only the resolved operation reading another weight.
        let mut resolved = document.clone();
        let selection = resolved
            .operations
            .iter_mut()
            .find(|s| s.source.id == "linear.k.0.bf16")
            .unwrap();
        selection.effective.weights[0].codes.tensor =
            "model.language_model.layers.0.self_attn.q_proj.weight".into();
        assert!(
            check(&resolved, &FP8_BLOCK128)
                .unwrap_err()
                .contains("its declaration")
        );
    }

    /// Every rank stores each lowered operation's weight as its offer reads
    /// it: the BF16 projections whose rows the derivation pads to the row
    /// tile (linear.beta 16 to 64 rows, the indexer's head weights 32 to
    /// 64, the router 288 to 320, and the drafter's twins) hold the rank's
    /// rows then zero rows. Stored unpadded, or with codes in another
    /// dtype, a weight is refused.
    #[test]
    fn ranks_store_each_weight_as_its_offer_reads_it() {
        let document = kernel_core::precision::component::decode(&test_component()).unwrap();
        let (declaration, _) = lowered().unwrap();
        let projection = |id: &str| declaration.operation_instance(id).unwrap().0.id.clone();
        for rank in 0..Shape::TP4.ranks() {
            let layout = crate::direct::layout(Shape::TP4, rank).unwrap();
            let stored: BTreeMap<&str, &Stored> = layout
                .iter()
                .map(|entry| (entry.view.name.as_str(), entry))
                .collect();
            let mut padded = BTreeSet::new();
            for selection in &document.operations {
                let geometry = selection.effective.geometry.unwrap();
                let codes = &selection.effective.weights[0].codes;
                let bytes = geometry.rows * geometry.k * codes.byte_length
                    / codes.shape.iter().product::<u64>();
                let bound = resident(selection, &stored).unwrap();
                assert_eq!(bound.codes, (codes.tensor.clone(), bytes));
                assert_eq!(
                    bound.scales.is_some(),
                    selection.effective.weights[0].local_scale.is_some()
                );
                let entry = stored[codes.tensor.as_str()];
                if entry.pieces[0].shape[0] < geometry.rows {
                    padded.insert(projection(&selection.source.id));
                    // Unpadded, the rank's rows alone are refused.
                    let mut unpadded = entry.clone();
                    unpadded.view.shape = entry.pieces[0].shape.clone();
                    let mut stored = stored.clone();
                    stored.insert(codes.tensor.as_str(), &unpadded);
                    let error = resident(selection, &stored).unwrap_err();
                    assert!(error.contains("of the"), "{error}");
                }
            }
            assert_eq!(
                padded,
                BTreeSet::from(
                    [
                        "linear.beta",
                        "latent.indexer.weights",
                        "moe.router",
                        "mtp.latent.indexer.weights",
                        "mtp.moe.router",
                    ]
                    .map(String::from)
                )
            );
        }
        let layout = crate::direct::layout(Shape::TP4, 0).unwrap();
        let mut stored: BTreeMap<&str, &Stored> = layout
            .iter()
            .map(|entry| (entry.view.name.as_str(), entry))
            .collect();
        let name = "model.language_model.layers.0.self_attn.q_proj.weight";
        let mut widened = stored[name].clone();
        widened.dtype = DType::F32;
        stored.insert(name, &widened);
        let selection = document
            .operations
            .iter()
            .find(|s| s.source.id == "linear.q.0.bf16")
            .unwrap();
        assert!(
            resident(selection, &stored)
                .unwrap_err()
                .contains("not as its checkpoint rows")
        );
    }

    /// Each layout's package carries the component of its own export,
    /// which decodes only against that layout; the block-FP8 one is the
    /// pinned test component.
    #[test]
    fn each_layout_carries_its_export_s_component() {
        use crate::quantization::{NVFP4_DENSE_ROUTED, NVFP4_ROUTED};
        let config = |bytes: &[u8]| -> serde_json::Value { serde_json::from_slice(bytes).unwrap() };
        let fp8 = config(include_bytes!("../fixtures/config.json"));
        assert_eq!(
            checkpoint_component(&FP8_BLOCK128, &fp8).unwrap(),
            test_component()
        );
        for (layout, bytes) in [
            (
                &NVFP4_ROUTED,
                &include_bytes!("../fixtures/nvfp4-config.json")[..],
            ),
            (
                &NVFP4_DENSE_ROUTED,
                &include_bytes!("../fixtures/nvidia-nvfp4-config.json")[..],
            ),
        ] {
            let component = checkpoint_component(layout, &config(bytes)).unwrap();
            decode_for(&component, layout).unwrap();
            assert!(decode(&component).is_err(), "{}", layout.id);
        }
    }
}
