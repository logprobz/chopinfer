//! The architecture is declared as data in `contract/projections.json`. The
//! compiler admits each declared source through the shared importer, binds
//! the checkpoint against the declaration and derives the rank's operations
//! with the shared derivation; it chooses only the packing. Precision
//! selection is shared SDK work; the host packs the weights from the
//! checkpoint by the bundle's weight plan when it loads the package.
use example_contract::precision as model;
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{ActivationPolicy, PrecisionDocument};
use kernel_core::precision::format::KernelFamily;
use kernel_core::precision::{
    Calibration, CodeEncoding, DecodeOrder, NibbleOrder, OperationKind, OperationSpec, TensorSpan,
};
use kernel_core::weight_transform::PackedMatrixSource;
use model_compiler_sdk::native_contract::NativeModule;
use model_compiler_sdk::preparation::{self as preparation, direct::DirectEmission};
use model_compiler_sdk::quantization::{
    declared::{self, Admitted, AdmittedRules},
    import::{ExportMetadata, ImportedOperand, OperandRole, import_operand},
    packing::PackingPlan,
    requested_policy, resolve_operation,
    source::TensorInventory,
};
use model_compiler_sdk::source::SourceFiles;
use model_compiler_sdk::{ComponentDraftV1, emit_bundle_directory_v1};
use projection_spec::ProjectionKind;
use projection_spec::bind::{CheckpointMetadata, TensorInfo};
use projection_spec::derive::{self, DerivationInput, GeometryMode, Topology, hooks::NoVariants};
use runtime_bundle::preparation::{NativeProfile, PrepareDirectRequest};
use runtime_bundle::{BundleOpaqueId, WEIGHT_PLAN_CAPABILITY_V1};
use runtime_weights::sha256_bytes;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
type Result<T> = model_compiler_sdk::files::Result<T>;

/// The native module the helper packages: `libexample_bundle.so` beside the
/// helper executable, or in `../lib` of an installed `bin` directory. There
/// is no `--native-module` for direct preparation; install the module built
/// with `precision-exercise` in one of these places.
const NATIVE_MODULE: &str = "libexample_bundle.so";

/// The one serving profile a direct package names: one rank, no drafter.
/// The example bundle does not execute the precision program over HTTP; the
/// profile is what `serve --model` would take as defaults.
const PROFILES: &str = r#"{"schema":"chopinfer-serve-profiles-v2","model":"example/stateful",
"served_model_name":"example-precision","profiles":{
"tp1-ar":{"bundle_id":"example.stateful.tp1.v1","topology":"tp1","tensor_parallel":1,"devices_required":1,
 "devices":[0],"speculation":{"mode":"fixed","tokens":0,"draft_cost":"per-token","drafter":"none"},
 "max_sequences":4,"qualified_max_sequences":null,"qualification":"Synthetic precision exercise, not qualified",
 "kv_cache_dtype":"auto","prefix_cache":false,"max_batch_tokens":512,"prefill_chunk_tokens":512,"queue_depth":8,
 "kv_memory_fraction":0.9,"status":"experimental","evidence":["tests/precision_preparation.py"]}
}}"#;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    architecture: String,
    seed: u64,
}

/// An admitted checkpoint: its captured metadata and safetensors headers,
/// and the operands and operations imported from them. No tensor payload is
/// read.
pub struct Source {
    seed: u64,
    #[cfg_attr(not(test), allow(dead_code))]
    files: SourceFiles,
    inventory: TensorInventory,
    operands: Vec<ImportedOperand>,
    operations: Vec<OperationSpec>,
    /// The rank's shared derivation the operations come from.
    #[cfg_attr(not(test), allow(dead_code))]
    derivation: derive::RankDerivation,
}

pub struct Family;

/// The binding view of the checkpoint's safetensors headers.
fn checkpoint_metadata(inventory: &TensorInventory) -> Result<CheckpointMetadata> {
    let mut tensors = std::collections::BTreeMap::new();
    for (name, tensor) in &inventory.tensors {
        let dtype = tensor
            .dtype
            .checkpoint_dtype()
            .ok_or_else(|| format!("tensor {name}: unsupported source dtype"))?;
        tensors.insert(
            name.clone(),
            TensorInfo {
                dtype: dtype.into(),
                shape: tensor.shape.clone(),
                file: inventory.spans.get(name).map(|s| s.file.clone()),
            },
        );
    }
    Ok(CheckpointMetadata {
        tensors,
        ..Default::default()
    })
}

fn span(offset: &mut u64, bytes: u64) -> Result<model::Span> {
    *offset = offset
        .checked_add(model::IMAGE_ALIGNMENT - 1)
        .ok_or("image offset overflow")?
        / model::IMAGE_ALIGNMENT
        * model::IMAGE_ALIGNMENT;
    let span = model::Span {
        offset: *offset,
        bytes,
    };
    *offset = offset.checked_add(bytes).ok_or("image size overflow")?;
    if *offset > model::MAX_IMAGE_BYTES {
        return Err("precision example image exceeds its resource contract".into());
    }
    Ok(span)
}

fn profile(ranks: u32) -> Result<NativeProfile> {
    if ranks != 1 {
        return Err("precision example requires one rank".into());
    }
    Ok(NativeProfile {
        profile: "precision-tp1".into(),
        tensor_parallel_size: 1,
        weight_contract: sha256_bytes(model::WEIGHT_CONTRACT),
        package_contract: sha256_bytes(model::PACKAGE_CONTRACT),
    })
}

/// The installed native module (see [`NATIVE_MODULE`]).
fn native_module() -> Result<PathBuf> {
    let executable = std::fs::canonicalize(std::env::current_exe()?)?;
    let bin = executable
        .parent()
        .ok_or("helper has no install directory")?;
    for path in [
        bin.join(NATIVE_MODULE),
        bin.join("../lib").join(NATIVE_MODULE),
    ] {
        if path.try_exists()? {
            if !std::fs::symlink_metadata(&path)?.is_file() {
                return Err("native module must be a regular installed file".into());
            }
            return Ok(std::fs::canonicalize(path)?);
        }
    }
    Err(format!(
        "install {NATIVE_MODULE}, built with precision-exercise, beside the helper or in its ../lib"
    )
    .into())
}

/// Admits the checkpoint at `root` from its metadata and safetensors headers
/// against the projection declaration, and imports one operand per declared
/// projection.
pub fn open(root: &Path, ranks: u32) -> Result<Source> {
    profile(ranks)?;
    let files = SourceFiles::capture(
        root,
        &[
            ("config.json", 4096),
            ("model.safetensors.index.json", 1024 * 1024),
            ("hf_quant_config.json", 1024 * 1024),
        ],
        &[],
        "model.safetensors.index.json",
    )?;
    let config: Config = serde_json::from_slice(&files.inputs["config.json"])?;
    if config.architecture != model::ARCHITECTURE {
        return Err("unsupported example architecture".into());
    }
    let inventory = TensorInventory::from_captured(&files, "model.safetensors.index.json")?;
    let mut export = ExportMetadata::default();
    export.add_document(
        "hf_quant_config.json",
        &files.inputs["hf_quant_config.json"],
    )?;
    let declaration = model::declaration()?;
    let geometry = model::geometry();
    let registry = model::registry();
    // The shared importer admits every declared source from its headers
    // and the export metadata.
    let mut admitted = Admitted::new();
    let mut operands = Vec::new();
    for p in &declaration.projections {
        let [source] = &declaration
            .instance_sources(&geometry, &p.id, None)
            .map_err(|e| e.to_string())?[..]
        else {
            return Err(format!("projection {}: the example reads one source tensor", p.id).into());
        };
        let operand = import_operand(
            &export,
            &OperandRole {
                operation: p.id.clone(),
                weight: source.clone(),
                targets: vec![source.trim_end_matches(".weight").into()],
                logical_shape: model::source_shape(p)?,
                lookup: p.kind == ProjectionKind::LookupTable {},
                shared_scale_stem: None,
                declared: None,
            },
            &inventory.tensors,
        )?;
        admitted.insert(source.clone(), operand.clone());
        operands.push(operand);
    }
    // Binding claims every checkpoint tensor for a declared projection, so
    // an unknown tensor, a missing companion or a shape or format the
    // declaration does not admit is refused, naming the tensor.
    let metadata = checkpoint_metadata(&inventory)?;
    let bound = declaration
        .bind(&geometry, &registry, &metadata, &AdmittedRules(&admitted))
        .map_err(|e| e.to_string())?;
    let mut imported =
        declared::imported(declaration, &bound, &geometry, &admitted).map_err(|e| e.to_string())?;
    // Matrices and tables keep their source shape in the operation; an
    // expert bank's bank-shaped weights need the rank geometry.
    for projection in imported.values_mut() {
        if projection.family != KernelFamily::GroupedExpert {
            projection.geometry = GeometryMode::SourceShape;
        }
    }
    let graph = derive::derive(DerivationInput {
        declaration,
        bound: &bound,
        geometry: &geometry,
        registry: &registry,
        topology: Topology { ranks: 1, rank: 0 },
        imported: &imported,
        variants: &NoVariants,
    })
    .map_err(|e| e.to_string())?;
    let mut operations = graph.operation_specs().map_err(|e| e.to_string())?;
    for operation in &mut operations {
        // The checkpoint's activation recipes, then the projection's own id.
        declared::admitted_activations(operation, &admitted, declaration, &geometry)
            .map_err(|e| e.to_string())?;
        let (p, _) = declaration
            .operation_instance(&operation.id)
            .map_err(|e| e.to_string())?;
        operation.id.clone_from(&p.id);
    }
    if operations
        .iter()
        .map(|o| &o.id)
        .ne(declaration.projections.iter().map(|p| &p.id))
    {
        return Err("derived operations differ from the declared projections".into());
    }
    Ok(Source {
        seed: config.seed,
        files,
        inventory,
        operands,
        operations,
        derivation: graph,
    })
}

/// The name of `span`, after checking it is the whole checkpoint tensor:
/// the plan's records read tensors whole.
fn whole<'a>(inventory: &TensorInventory, span: &'a TensorSpan) -> Result<&'a str> {
    if inventory.tensors.get(&span.tensor) != Some(span) {
        return Err(format!(
            "tensor {}: the example reads whole tensors only",
            span.tensor
        )
        .into());
    }
    Ok(&span.tensor)
}

/// The static input calibration of an operand, when it has one.
fn calibration(operand: &ImportedOperand) -> Option<&TensorSpan> {
    operand
        .activation
        .as_ref()
        .and_then(|a| a.quantizer.as_ref())
        .and_then(|q| match &q.calibration {
            Calibration::Static(s) => Some(&s.tensor),
            _ => None,
        })
}

/// How the weight plan packs `operand` into `plan`: every `(expert,
/// projection)` reads expert `expert` of the source bank and keeps that
/// projection's rows, as the prepared-weight builder is given them.
fn packing(
    inventory: &TensorInventory,
    operand: &ImportedOperand,
    plan: PackingPlan,
    decode_order: DecodeOrder,
) -> Result<model::Packing> {
    let weight = &operand.weight;
    let shape = &weight.logical_shape;
    let (bank, rows, k) = match shape[..] {
        [bank, rows, k] => (bank, rows, k),
        [rows, k] => (1, rows, k),
        _ => return Err("example weights are matrices or banks".into()),
    };
    // Source coordinates of the first row and column, from the local scale
    // grid of one matrix (the last two logical axes).
    let origin = |axis: usize| {
        weight
            .local_scale
            .as_ref()
            .and_then(|s| s.groups.iter().find(|g| usize::from(g.axis) == axis))
            .map_or(0, |g| g.origin)
    };
    let mut sources = vec![whole(inventory, &weight.codes)?.to_owned()];
    let mut add = |span: &TensorSpan| -> Result<u32> {
        sources.push(whole(inventory, span)?.to_owned());
        Ok(sources.len() as u32 - 1)
    };
    let local = weight
        .local_scale
        .as_ref()
        .map(|s| add(&s.tensor))
        .transpose()?;
    let global = weight
        .global_scale
        .as_ref()
        .map(|s| add(&s.tensor))
        .transpose()?;
    let mut matrices = Vec::new();
    for expert in 0..plan.experts {
        for projection in 0..plan.projections {
            let first = u64::from(projection) * u64::from(plan.rows);
            matrices.push(PackedMatrixSource {
                expert,
                projection,
                format: weight.format,
                rows,
                k,
                row_origin: origin(shape.len() - 2),
                k_origin: origin(shape.len() - 1),
                nibble_order: match weight.encoding {
                    CodeEncoding::E2M1(order) => order,
                    _ => NibbleOrder::EvenLow,
                },
                codes: 0,
                local,
                local_dtype: weight.local_scale.as_ref().map(|s| s.tensor.dtype),
                global,
                source_expert: u64::from(expert),
                source_experts: bank,
                rows_kept: std::iter::once(first..first + u64::from(plan.rows)).collect(),
                k_kept: 0..u64::from(plan.k),
            });
        }
    }
    let input_scale_tensor = calibration(operand)
        .map(|span| whole(inventory, span).map(str::to_owned))
        .transpose()?;
    Ok(model::Packing {
        plan,
        decode_order,
        sources,
        matrices,
        input_scale_tensor,
    })
}

/// Selects each operation's precision under `policy` from the module's
/// capability `inventory` and lays out the weight image the plan fills.
pub fn resolve(
    source: &Source,
    policy: ActivationPolicy,
    inventory: &CapabilityInventory,
    checkpoint_identity: &str,
) -> Result<model::Exercise> {
    let mut precision = PrecisionDocument {
        version: 1,
        operations: vec![],
    };
    let mut weights = Vec::new();
    let mut offset = 0;
    for (operation, operand) in source.operations.iter().zip(&source.operands) {
        let prepared = resolve_operation(operation, policy, inventory)?;
        let selected = prepared.selection();
        let geometry = selected.effective.matrix_geometry()?;
        let plan = PackingPlan {
            layout: selected.effective.weights[0].layout.clone(),
            format: selected.effective.weights[0].format,
            experts: match operation.kind {
                OperationKind::Grouped { experts, .. } => experts,
                _ => 1,
            },
            projections: operation.weights.len() as u32,
            rows: geometry.rows as u32,
            k: geometry.k as u32,
            columns: SwigluColumns::ConcatenatedHalves,
            local_scale_dtype: Some(DType::Fp8E4M3),
        };
        plan.validate()?;
        if plan.format != WeightFormat::Nvfp4 {
            return Err("supplied example entries require NVFP4 weights".into());
        }
        let sections = plan.section_bytes()?;
        let input = calibration(operand).map_or(0, |s| s.byte_length);
        weights.push(model::WeightSpans {
            operation: operation.id.clone(),
            codes: span(&mut offset, sections.codes)?,
            local_scales: span(&mut offset, sections.local_scales)?,
            global_scales: span(&mut offset, sections.global_scales)?,
            input_scale: span(&mut offset, input)?,
            packing: packing(
                &source.inventory,
                operand,
                plan,
                selected.effective.arithmetic.decode_order,
            )?,
        });
        precision.operations.push(selected.clone());
    }
    let exercise = model::Exercise {
        version: model::EXERCISE_VERSION,
        checkpoint_identity: checkpoint_identity.into(),
        precision,
        weights,
        image_bytes: offset.next_multiple_of(model::IMAGE_ALIGNMENT),
    };
    exercise.validate()?;
    Ok(exercise)
}

impl preparation::helper::Provider for Family {
    fn name(&self) -> &str {
        "example-compile"
    }
    fn profiles(&self) -> Vec<String> {
        vec!["precision-tp1".into()]
    }
    fn probe(&self, path: &Path, ranks: u32) -> Result<()> {
        open(path, ranks).map(|_| ())
    }
    /// The package the host loads the checkpoint with: the seed, frontend
    /// and tokenizer, the precision policy and exercise documents, and the
    /// weight image as a planned component the bundle's weight plan fills.
    fn emit_direct(
        &self,
        root: &Path,
        request: &PrepareDirectRequest,
        checkpoint_identity: &str,
        output: &Path,
    ) -> Result<DirectEmission> {
        let expected = profile(request.tensor_parallel_size)?;
        let policy = requested_policy(request.activation_policy.as_deref())?
            .unwrap_or(ActivationPolicy::Auto);
        let source = open(root, request.tensor_parallel_size)?;
        let module = NativeModule::inspect(&native_module()?)?;
        module.require(&expected)?;
        let exercise = resolve(
            &source,
            policy,
            &module.precision_inventory()?,
            checkpoint_identity,
        )?;
        let extra = vec![
            super::component(
                "precision-policy",
                model::POLICY_ROLE,
                kernel_core::precision::component::COMPONENT_FORMAT,
                "precision-policy.bin",
                kernel_core::precision::component::encode(&exercise.precision)?,
            )?,
            super::component(
                "precision",
                model::PRECISION_ROLE,
                "example.precision.v2",
                "precision.json",
                serde_json::to_vec(&exercise)?,
            )?,
            ComponentDraftV1::planned(
                BundleOpaqueId::new("precision-weights")?,
                BundleOpaqueId::new(model::WEIGHTS_ROLE)?,
                BundleOpaqueId::new("example.precision.packed.v1")?,
            )?,
        ];
        let package = emit_bundle_directory_v1(
            &super::draft(
                source.seed,
                module.staged()?,
                extra,
                &[WEIGHT_PLAN_CAPABILITY_V1],
            )?,
            output,
        )?;
        Ok(DirectEmission {
            package,
            profile: preparation::direct::default_profile(PROFILES, 1, &["none"])?,
        })
    }
}

#[cfg(test)]
mod tests;
