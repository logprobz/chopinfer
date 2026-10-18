//! MiMo V2.6 Flash's weight plans for direct checkpoint loading.
//!
//! The text and drafter images are the rank derivation of the projection
//! declaration over the admitted checkpoint operands: each projection's
//! records under the served record policy ([`record_policy`]) and each side
//! tensor's rank copy. The encoder image follows the encoder contract. Views
//! come in this order:
//!
//! - text: the non-expert views by their first tensor's file and offset in
//!   the file (gate and up fused), then the expert banks by layer, each
//!   bank's experts in order;
//! - drafter: the views by their first tensor's name, DFlash's context
//!   projection and mask last;
//! - encoders: every tensor by name, identical on every rank
//!   (`encoders/image.rs`).
//!
//! Each piece, one source tensor's rank selection, becomes one record, and
//! every view starts on the 4 KiB view alignment. When the package's rank index
//! lists exactly the planned views, its offsets are used; otherwise the order
//! above is.
//!
//! A planned rank has no image file: the rank index each plan carries names
//! the plan's identity (`direct_identity`) instead of an image digest, and the
//! bundle checks it against the host's as it checks a rank file's digest.

use crate::geometry::Geometry;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use kernel_core::weight_transform::{WeightRecord, WeightTransform};
use model_compiler_sdk::weight_plan::{RequestCache, RequestFiles, RequestKey, plan_identity};
use runtime_bundle_abi::weight_plan::{
    CheckpointFileV1, NamedBytesV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use runtime_weights::DType;
use sha2::{Digest, Sha256};

use crate::catalog::TensorRecord;
pub use crate::precision::Scope;
use crate::{MimoResult, Tp};
use kernel_core::precision::format::{FormatId, KernelFamily};
use model_compiler_sdk::quantization::declared::{self, Admitted, AdmittedRules};
use model_compiler_sdk::quantization::format::RecordLayout;
use model_compiler_sdk::quantization::projection::RecordPolicy;

/// Every view of an image starts on this alignment.
pub const VIEW_ALIGNMENT: u64 = 4096;

pub const TEXT_IMAGE_ROLE: &str = "mimo.rank-images.v1";
pub const DRAFTER_IMAGE_ROLE: &str = "mimo.drafter-rank-images.v1";
/// A package without rank images names its drafter here,
/// `{"drafter": "mtp" | "dflash"}`, having no drafter index to show it.
pub const DRAFTER_KIND_ROLE: &str = "chop.bundle-metadata.mimo.drafter-kind.v1";
pub const ENCODER_IMAGE_ROLE: &str = "mimo.encoder-rank-images.v1";
/// A package serving some of the media names them here,
/// `{"media": ["image" | "video" | "audio", ...]}`, and plans only their
/// encoders; without it the package serves every medium.
pub const ENCODER_SET_ROLE: &str = "chop.bundle-metadata.mimo.encoder-set.v1";

const INDEX: &str = "model.safetensors.index.json";
/// The alignment of a planned arena: the packages' rank image alignment
/// (`mimo26-package`), which is also what a CUDA allocation guarantees. View
/// offsets within the arena keep the 4 KiB view alignment, as in the rank
/// images; the arena itself asks for no more than a package does, or the
/// host refuses a device allocation that happens to fall short of it.
pub const IMAGE_ALIGNMENT: u64 = 256;
pub const MTP_FILE: &str = "model_mtp.safetensors";
pub const DFLASH_FILE: &str = "dflash/dflash_draft_model.safetensors";
const DFLASH_CONFIG: &str = "dflash/config.json";
pub const MASK_ARCHIVE: &str = "dflash/mask_embedding.pt";
/// The checked BF16 tensor archive of the DFlash mask embedding.
pub const MASK_ARCHIVE_SHA256: &str =
    "b35b379fe0497ffdc0d6407f502622237d9346e0d03f457ec574be60f6c2cee0";
pub const TOKENIZER_FILE: &str = "audio_tokenizer/model.safetensors";

/// The rank index role of `image` at `rank`.
#[must_use]
pub fn index_role(image: &str, rank: u32) -> String {
    use bundle_runtime::index::ImageKind;
    let kind = match image {
        DRAFTER_IMAGE_ROLE => ImageKind::Drafter,
        ENCODER_IMAGE_ROLE => ImageKind::Encoder,
        _ => ImageKind::Text,
    };
    bundle_runtime::index::index_role("mimo", kind, rank)
}

/// Which drafter a package's drafter image holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Drafter {
    Mtp,
    Dflash,
}

/// One piece of a view: a transform of checkpoint tensors, at an offset
/// within the view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Piece {
    pub offset: u64,
    pub bytes: u64,
    pub transform: WeightTransform,
    /// Checkpoint tensor names, in record order.
    pub sources: Vec<String>,
    pub axis: Option<u32>,
    pub ranges: Vec<Range<u64>>,
    pub stored_shape: Vec<u64>,
}

/// One view of a rank image and the pieces that fill it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedView {
    pub name: String,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout_id: String,
    pub pieces: Vec<Piece>,
}

fn dtype_bytes(dtype: DType) -> u64 {
    match dtype {
        DType::Bf16 => 2,
        DType::F32 => 4,
        _ => 1,
    }
}

fn selected(name: &str, layers: Option<&BTreeSet<usize>>) -> bool {
    let Some(layers) = layers else {
        return true;
    };
    if name == "model.embed_tokens.weight"
        || name == "model.norm.weight"
        || name == "lm_head.weight"
    {
        return true;
    }
    name.strip_prefix("model.layers.")
        .and_then(|tail| tail.split('.').next())
        .and_then(|layer| layer.parse::<usize>().ok())
        .is_some_and(|layer| layers.contains(&layer))
}

fn tensor<'a>(
    tensors: &'a BTreeMap<String, TensorRecord>,
    name: &str,
) -> MimoResult<&'a TensorRecord> {
    tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}").into())
}

/// The served record encoding of a projection's weights: row records for
/// lookups, MXFP4 strips for routed experts, MMA fragments for GEMMs, with
/// block-FP8 GEMM scales under their alternate layout names.
pub fn record_policy(format: FormatId, family: KernelFamily) -> RecordPolicy {
    if format == FormatId::Fp8Block128 && family == KernelFamily::Dense {
        return RecordPolicy::SectionsWithLayout {
            records: RecordLayout::Fragment,
            layout: kernel_core::precision::format::LayoutVariant::AlternateBlockScaleNames,
        };
    }
    RecordPolicy::Sections(if family == KernelFamily::Lookup {
        RecordLayout::Rows
    } else if format == FormatId::Mxfp4 && family == KernelFamily::GroupedExpert {
        RecordLayout::Mxfp4Strips
    } else {
        RecordLayout::Fragment
    })
}

/// The resident view of a projection source tensor: fused gate and up
/// operands share the gate-up view, a routed bank one view of every expert.
pub fn resident(name: &str, fused: bool) -> String {
    let name = name.split('@').next().unwrap_or(name);
    let name = if fused {
        name.replace(".gate_proj.", ".gate_up_proj.")
            .replace(".up_proj.", ".gate_up_proj.")
    } else {
        name.into()
    };
    match name.split_once(".experts.") {
        Some((prefix, tail)) => match tail.split_once('.') {
            Some((expert, rest)) if expert.parse::<u64>().is_ok() => {
                format!("{prefix}.experts.{rest}")
            }
            _ => name,
        },
        None => name,
    }
}

/// The declared projections and sides an image of `scope` holds, with one
/// use each (resident records do not depend on the use) and every layer of
/// its stems materialized.
fn image_declaration(
    tp: Tp,
    scope: Scope,
) -> MimoResult<(
    projection_spec::Declaration,
    &'static projection_spec::Geometry,
)> {
    let (declaration, geometry) = crate::projections::expanded(tp)?;
    let mut declaration = declaration.clone();
    let holds = |id: &str| match scope {
        Scope::Text => !id.starts_with("mtp") && !id.starts_with("dflash."),
        Scope::Mtp => id.starts_with("mtp") && !id.contains("-adapter."),
        Scope::Dflash => id.starts_with("dflash.") && id != "dflash.embedding",
    };
    declaration.projections.retain(|p| holds(&p.id));
    declaration.side_tensors.retain(|s| holds(&s.id));
    declaration.external_tensors.clear();
    for projection in &mut declaration.projections {
        projection.uses.truncate(1);
    }
    if scope == Scope::Mtp {
        for stem in &mut declaration.layer_stems {
            stem.execution = None;
        }
    }
    Ok((declaration, geometry))
}

/// The resident tensor an assembly projection declares for `layer`.
fn assembly_name(
    declaration: &projection_spec::Declaration,
    projection: &projection_spec::Projection,
    layer: Option<u64>,
) -> Option<String> {
    let source = projection.source.as_ref()?;
    let stem = match (&source.stem, layer) {
        (Some(stem), Some(layer)) => declaration
            .layer_stems
            .iter()
            .find(|s| s.id == *stem)?
            .template
            .replace("{i}", &layer.to_string()),
        _ => String::new(),
    };
    Some(source.tensor.replace("{stem}", &stem))
}

/// Dense fragment records stream 16-row strips: every gathered row run and
/// the rank's stored rows must be whole strips.
fn check_fragment_strips(record: &WeightRecord, tensor: &str) -> MimoResult<()> {
    if matches!(record.transform, WeightTransform::Fragment(_))
        && (!record
            .stored_shape
            .first()
            .is_some_and(|rows| rows.is_multiple_of(16))
            || record
                .ranges
                .iter()
                .any(|run| !(run.end - run.start).is_multiple_of(16)))
    {
        return Err(format!("{tensor} fragment rows must align to 16").into());
    }
    Ok(())
}

/// The resident view a declared operation reads when its projection is an
/// assembly with its own tensor (the drafter's fused QKV and context K/V);
/// `None` for an operation that reads its sources' residents.
pub fn assembled_resident(tp: Tp, operation: &str) -> MimoResult<Option<String>> {
    let (declaration, _) = crate::projections::expanded(tp)?;
    let (projection, layer) = declaration.operation_instance(operation)?;
    Ok(assembly_name(declaration, projection, layer).filter(|_| !projection.assembly.is_empty()))
}

/// A planned view of one derived resident view, ordered by `key`.
struct Resident {
    key: (u8, String, u64, String),
    view: PlannedView,
}

/// One image's declaration bound to the admitted checkpoint operands; its
/// ranks derive from it independently.
struct Image<'a> {
    scope: Scope,
    tp: Tp,
    tensors: &'a BTreeMap<String, TensorRecord>,
    declaration: projection_spec::Declaration,
    geometry: &'static projection_spec::Geometry,
    metadata: projection_spec::bind::CheckpointMetadata,
    bound: projection_spec::bind::BoundDeclaration,
    imported: projection_spec::derive::ImportedProjections,
}

impl<'a> Image<'a> {
    fn new(
        scope: Scope,
        tensors: &'a BTreeMap<String, TensorRecord>,
        admitted: &Admitted,
        tp: Tp,
    ) -> MimoResult<Self> {
        let (declaration, geometry) = image_declaration(tp, scope)?;
        let metadata = crate::projections::imports::metadata(tensors);
        let registry = crate::projections::registry()?;
        let bound = declaration.bind(geometry, &registry, &metadata, &AdmittedRules(admitted))?;
        let imported = declared::imported(&declaration, &bound, geometry, admitted)?;
        Ok(Self {
            scope,
            tp,
            tensors,
            declaration,
            geometry,
            metadata,
            bound,
            imported,
        })
    }

    /// Every rank's views, derived in parallel.
    fn all_views(&self) -> MimoResult<Vec<Vec<PlannedView>>> {
        std::thread::scope(|threads| {
            let ranks: Vec<_> = (0..self.tp.count())
                .map(|rank| threads.spawn(move || self.views(rank).map_err(|e| e.to_string())))
                .collect();
            ranks
                .into_iter()
                .map(|rank| {
                    rank.join()
                        .map_err(|_| "a rank plan panicked")?
                        .map_err(Into::into)
                })
                .collect()
        })
    }

    /// The views of `rank`'s image: each projection's records under the
    /// served record policy, and each side's rank copy.
    fn views(&self, rank: usize) -> MimoResult<Vec<PlannedView>> {
        use model_compiler_sdk::quantization::projection::{bind_record_policy, records};
        use projection_spec::derive::{DerivationInput, Topology};
        let Image {
            scope,
            tp,
            tensors,
            declaration,
            geometry,
            metadata,
            bound,
            imported,
        } = self;
        let (scope, tp, geometry) = (*scope, *tp, *geometry);
        let registry = crate::projections::registry()?;
        let topology = Topology {
            ranks: tp.ranks(),
            rank: u32::try_from(rank)?,
        };
        let graph = projection_spec::derive::derive(DerivationInput {
            declaration,
            bound,
            geometry,
            registry: &registry,
            topology,
            imported,
            variants: &crate::projections::variants::Variants(tp.geometry),
        })?;
        // Members a same-layer assembly reads are resident only through it.
        let folded: BTreeSet<&str> = declaration
            .projections
            .iter()
            .flat_map(|p| &p.assembly)
            .filter(|member| {
                matches!(
                    member.layer,
                    Some(projection_spec::LayerSelector::Current(_))
                )
            })
            .map(|member| member.projection.as_str())
            .collect();
        let order = |id: &str| {
            declaration
                .projections
                .iter()
                .position(|p| p.id == id)
                .unwrap_or(usize::MAX) as u64
        };
        let mut residents: BTreeMap<String, Resident> = BTreeMap::new();
        for operation in graph.selected_operations() {
            let projection = &operation.instance.projection;
            if folded.contains(projection.as_str()) {
                continue;
            }
            let operation = if operation.variant_records.is_empty()
                && operation.source_preparation.is_none()
            {
                let format =
                    kernel_core::precision::format::registry::by_format(operation.packing.format)?
                        .id();
                &bind_record_policy(operation.clone(), record_policy(format, operation.family))?
            } else {
                operation
            };
            let declared = declaration
                .projections
                .iter()
                .find(|p| p.id == *projection)
                .ok_or("derived operation has no declaration")?;
            for record in records(operation)? {
                check_fragment_strips(&record.record, &record.tensors[0])?;
                let view = operation
                    .views
                    .iter()
                    .find(|v| v.id == record.view)
                    .ok_or("record has no derived view")?;
                let resident = residents.entry(record.view.clone()).or_insert_with(|| {
                    let first = &record.tensors[0];
                    let name = assembly_name(declaration, declared, operation.instance.layer)
                        .filter(|_| !declared.assembly.is_empty())
                        .unwrap_or_else(|| resident(first, operation.packing.projections == 2));
                    let bank = operation.packing.experts > 1;
                    let key = if bank {
                        (
                            1,
                            format!("{:08}", operation.instance.layer.unwrap_or(0)),
                            order(projection) * 4 + record.section.index() as u64,
                            String::new(),
                        )
                    } else if !declared.assembly.is_empty() && declared.stem().is_none() {
                        (2, first.clone(), 0, first.clone())
                    } else {
                        (0, first.clone(), 0, first.clone())
                    };
                    Resident {
                        key,
                        view: PlannedView {
                            name,
                            bytes: 0,
                            dtype: record.record.sources[0].dtype.clone(),
                            shape: vec![],
                            layout_id: view.layout.clone(),
                            pieces: vec![],
                        },
                    }
                });
                if resident
                    .view
                    .pieces
                    .iter()
                    .any(|p| p.offset == record.view_offset)
                {
                    continue;
                }
                let bytes = record.record.output_bytes()?;
                resident.view.pieces.push(Piece {
                    offset: record.view_offset,
                    bytes,
                    transform: record.record.transform,
                    sources: record.tensors,
                    axis: record.record.axis,
                    ranges: record.record.ranges.into_iter().collect(),
                    stored_shape: record.record.stored_shape,
                });
                if operation.packing.experts > 1 {
                    resident.key.0 = 1;
                    resident.view.shape = vec![u64::from(operation.packing.experts)];
                }
            }
        }
        let mut views = Vec::new();
        for mut resident in residents.into_values() {
            let view = &mut resident.view;
            view.pieces.sort_by_key(|p| p.offset);
            view.bytes = view.pieces.iter().map(|p| p.bytes).sum();
            // Pieces stack rows: per expert in a bank, else for the whole view.
            let experts = view.shape.first().copied().unwrap_or(1) as usize;
            let per_expert = view.pieces.len() / experts.max(1);
            let rows = view.pieces[..per_expert]
                .iter()
                .map(|p| p.stored_shape[0])
                .sum();
            let mut shape = view.shape.clone();
            shape.push(rows);
            shape.extend_from_slice(&view.pieces[0].stored_shape[1..]);
            view.shape = shape;
            views.push(resident);
        }
        for (name, record) in model_compiler_sdk::quantization::projection::side_records(
            declaration,
            bound,
            geometry,
            topology,
            metadata,
        )? {
            let bytes = record.output_bytes()?;
            let dtype = tensor(tensors, &name)?.dtype.safetensors_name().to_owned();
            // Padding past the tensor's end is inline and reads nothing.
            let sources = if record.sources.is_empty() {
                vec![]
            } else {
                vec![name.clone()]
            };
            views.push(Resident {
                key: (0, name.clone(), 0, name.clone()),
                view: PlannedView {
                    layout_id: format!("{}_row_major_v1", dtype.to_lowercase()),
                    name: name.clone(),
                    bytes,
                    dtype,
                    shape: record.stored_shape.clone(),
                    pieces: vec![Piece {
                        offset: 0,
                        bytes,
                        transform: record.transform,
                        sources,
                        axis: record.axis,
                        ranges: record.ranges.into_iter().collect(),
                        stored_shape: record.stored_shape,
                    }],
                },
            });
        }
        // The text image orders its non-bank views by checkpoint position;
        // drafters order theirs by name, unstemmed assemblies last.
        if scope == Scope::Text {
            for resident in &mut views {
                if resident.key.0 == 0 {
                    let first = tensor(tensors, &resident.key.3)?;
                    resident.key.1 = first.file.clone();
                    resident.key.2 = first.file_start;
                }
            }
        }
        views.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(views.into_iter().map(|r| r.view).collect())
    }
}

/// The text image's views of `rank`, in the order the module doc gives.
/// `layers`, when given, keeps only those decoder layers.
pub fn text_views(
    tensors: &BTreeMap<String, TensorRecord>,
    admitted: &Admitted,
    layers: Option<&BTreeSet<usize>>,
    rank: usize,
    tp: Tp,
) -> MimoResult<Vec<PlannedView>> {
    let views = Image::new(Scope::Text, tensors, admitted, tp)?.views(rank)?;
    Ok(text_layers(views, layers))
}

fn text_layers(mut views: Vec<PlannedView>, layers: Option<&BTreeSet<usize>>) -> Vec<PlannedView> {
    views.retain(|view| selected(&view.pieces[0].sources[0], layers));
    views
}

/// The drafter image's views of `rank`. `mask` is DFlash's mask embedding.
pub fn drafter_views(
    tensors: &BTreeMap<String, TensorRecord>,
    admitted: &Admitted,
    kind: Drafter,
    mask: Option<&[u8; crate::drafters::MASK_BYTES]>,
    rank: usize,
    tp: Tp,
) -> MimoResult<Vec<PlannedView>> {
    let views = Image::new(drafter_scope(kind), tensors, admitted, tp)?.views(rank)?;
    with_mask(views, kind, mask, tp)
}

fn drafter_scope(kind: Drafter) -> Scope {
    match kind {
        Drafter::Mtp => Scope::Mtp,
        Drafter::Dflash => Scope::Dflash,
    }
}

/// DFlash's drafter image ends with its mask embedding.
fn with_mask(
    mut views: Vec<PlannedView>,
    kind: Drafter,
    mask: Option<&[u8; crate::drafters::MASK_BYTES]>,
    tp: Tp,
) -> MimoResult<Vec<PlannedView>> {
    if kind == Drafter::Dflash {
        let mask = mask.ok_or("the DFlash drafter needs its mask embedding")?;
        views.push(PlannedView {
            name: "mask_embedding.weight".into(),
            bytes: mask.len() as u64,
            dtype: DType::Bf16.safetensors_name().into(),
            shape: vec![u64::from(tp.geometry.dflash().hidden_size)],
            layout_id: "bf16_row_major_v1".into(),
            pieces: vec![Piece {
                offset: 0,
                bytes: mask.len() as u64,
                transform: WeightTransform::Inline(mask.to_vec()),
                sources: vec![],
                axis: None,
                ranges: vec![],
                stored_shape: vec![],
            }],
        });
    }
    Ok(views)
}

/// Every rank's views of the request's text or drafter image: the host asks
/// rank by rank, so the first request derives all ranks in parallel.
static RANK_VIEWS: RequestCache<Vec<Vec<PlannedView>>> = RequestCache::new();

fn image_views(
    request: &WeightPlanRequestV1,
    scope: Scope,
    tensors: &BTreeMap<String, TensorRecord>,
    admitted: &Admitted,
    tp: Tp,
) -> MimoResult<Arc<Vec<Vec<PlannedView>>>> {
    let mut key = RequestKey::new(b"rank-views", request);
    for document in &request.documents {
        key.bytes(document.name.as_bytes());
        key.bytes(&document.bytes);
    }
    key.bytes(format!("{scope:?} {}", tp.ranks()).as_bytes());
    RANK_VIEWS.get_or_build(key.finish(), || {
        Image::new(scope, tensors, admitted, tp)?.all_views()
    })
}

/// The encoder image's views, the same on every rank: the tensors the
/// encoder kernels read ([`crate::encoders::prepared_flash`]). Each is filled by
/// one record per checkpoint tensor at adjacent offsets; the zeros of a
/// missing bias are a gap the executor leaves zero.
pub fn encoder_views_flash(
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<Vec<PlannedView>> {
    encoder_views_of_flash(crate::encoders::Encoders::ALL, tensors)
}

/// [`encoder_views_flash`] of the image holding `encoders` alone.
pub fn encoder_views_of_flash(
    encoders: crate::encoders::Encoders,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<Vec<PlannedView>> {
    encoder_views_for(&crate::geometry::V26_FLASH, encoders, tensors)
}

pub fn encoder_views_for(
    geometry: &'static Geometry,
    encoders: crate::encoders::Encoders,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<Vec<PlannedView>> {
    let prepared = crate::encoders::prepared_for(geometry)
        .into_iter()
        .filter(|tensor| encoders.holds(&tensor.name))
        .collect();
    prepared_views(prepared, tensors)
}

/// The media a package's encoders serve: those its encoder-set component
/// names, else all of them.
pub fn encoder_set(request: &WeightPlanRequestV1) -> MimoResult<crate::encoders::Encoders> {
    match request.component(ENCODER_SET_ROLE) {
        Some(bytes) => Ok(parse_encoder_set(bytes)?),
        None => Ok(crate::encoders::Encoders::ALL),
    }
}

/// The served media an encoder-set component names, at least one.
pub fn parse_encoder_set(bytes: &[u8]) -> Result<crate::encoders::Encoders, String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|error| format!("encoder set: {error}"))?;
    let names = value["media"]
        .as_array()
        .ok_or("the encoder-set component names no media")?
        .iter()
        .map(|name| name.as_str().ok_or("a medium's name is not a string"))
        .collect::<Result<Vec<_>, _>>()?;
    let encoders = crate::encoders::Encoders::named(names)?;
    if !encoders.any() || value.as_object().is_none_or(|fields| fields.len() != 1) {
        return Err("the encoder-set component names no media or other fields".into());
    }
    Ok(encoders)
}

/// The encoder-set component of a package serving `encoders`.
pub fn encoder_set_component(encoders: crate::encoders::Encoders) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({ "media": encoders.names() })).expect("JSON")
}

/// The views of `prepared` tensors over checkpoint `tensors`, as
/// [`encoder_views_flash`] plans them.
pub fn prepared_views(
    prepared: Vec<crate::encoders::Prepared>,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<Vec<PlannedView>> {
    use crate::encoders::Part;
    prepared
        .into_iter()
        .map(|prepared| {
            let mut pieces = Vec::with_capacity(prepared.parts.len());
            let mut offset = 0_u64;
            for part in &prepared.parts {
                let (name, transform) = match part {
                    Part::Zeros(bytes) => {
                        offset += bytes;
                        continue;
                    }
                    Part::Copy(name) => (name, WeightTransform::Copy),
                    Part::Taps(name) => (name, WeightTransform::SwapInnerAxes),
                    Part::Widened(name) => (name, WeightTransform::WidenBf16F32),
                };
                let t = tensor(tensors, name)?;
                let elements = t.shape.iter().product::<u64>();
                let (stored_shape, bytes) = match transform {
                    WeightTransform::SwapInnerAxes => {
                        let [out, channels, taps] = t.shape[..] else {
                            return Err(format!("{name} is not a convolution weight").into());
                        };
                        (vec![out, taps, channels], elements * 2)
                    }
                    WeightTransform::WidenBf16F32 => (t.shape.clone(), elements * 4),
                    _ => (t.shape.clone(), elements * dtype_bytes(t.dtype)),
                };
                pieces.push(Piece {
                    offset,
                    bytes,
                    transform,
                    sources: vec![name.clone()],
                    axis: None,
                    ranges: vec![],
                    stored_shape,
                });
                offset += bytes;
            }
            let width = if prepared.dtype == "F32" { 4 } else { 2 };
            let bytes = prepared.shape.iter().product::<u64>() * width;
            if offset != bytes {
                return Err(format!(
                    "encoder tensor {} parts differ from its shape",
                    prepared.name
                )
                .into());
            }
            Ok(PlannedView {
                layout_id: prepared.layout_id().into(),
                name: prepared.name,
                bytes,
                dtype: prepared.dtype.into(),
                shape: prepared.shape,
                pieces,
            })
        })
        .collect()
}

/// Offsets of `views` laid end to end on the view alignment, and the image
/// length.
pub fn sequential_offsets(views: &[PlannedView]) -> (Vec<u64>, u64) {
    let mut end = 0_u64;
    let offsets = views
        .iter()
        .map(|view| {
            let offset = end.div_ceil(VIEW_ALIGNMENT) * VIEW_ALIGNMENT;
            end = offset + view.bytes;
            offset
        })
        .collect();
    (offsets, end.div_ceil(VIEW_ALIGNMENT) * VIEW_ALIGNMENT)
}

/// The package index's offsets and image length, when it lists exactly
/// `views` (names, bytes, dtypes, shapes and layouts) in a valid layout.
fn package_offsets(views: &[PlannedView], index: &[u8]) -> Option<(Vec<u64>, u64)> {
    let index: serde_json::Value = serde_json::from_slice(index).ok()?;
    let image_bytes = index["image_bytes"].as_u64()?;
    let listed = index["views"].as_array()?;
    if listed.len() != views.len() || !image_bytes.is_multiple_of(VIEW_ALIGNMENT) {
        return None;
    }
    let by_name: BTreeMap<&str, &serde_json::Value> = listed
        .iter()
        .filter_map(|view| Some((view["name"].as_str()?, view)))
        .collect();
    let mut offsets = Vec::with_capacity(views.len());
    for view in views {
        let listed = by_name.get(view.name.as_str())?;
        if listed["bytes"].as_u64()? != view.bytes
            || listed["dtype"].as_str()? != view.dtype
            || listed["layout_id"].as_str()? != view.layout_id
            || listed["shape"] != serde_json::json!(view.shape)
        {
            return None;
        }
        offsets.push(listed["offset"].as_u64()?);
    }
    let mut extents: Vec<(u64, u64)> = offsets
        .iter()
        .zip(views)
        .map(|(offset, view)| (*offset, offset + view.bytes))
        .collect();
    extents.sort_unstable();
    let valid = extents
        .iter()
        .all(|(start, end)| start.is_multiple_of(VIEW_ALIGNMENT) && *end <= image_bytes)
        && extents.windows(2).all(|pair| pair[0].1 <= pair[1].0);
    valid.then_some((offsets, image_bytes))
}

/// One rank's plan of image `role` from its views at `offsets`, with the
/// rank index the bundle reads (`schema`, image file name `image`).
#[allow(clippy::too_many_arguments)]
fn assemble(
    request: &WeightPlanRequestV1,
    role: &str,
    tensors: &BTreeMap<String, TensorRecord>,
    views: &[PlannedView],
    offsets: &[u64],
    image_bytes: u64,
    schema: &str,
    image: &str,
) -> MimoResult<WeightPlanV1> {
    let files = RequestFiles::new(request);
    let mut records = Vec::new();
    for (view, offset) in views.iter().zip(offsets) {
        for piece in &view.pieces {
            records.push(WeightRecordV1 {
                arena_offset: offset + piece.offset,
                byte_length: piece.bytes,
                transform: piece.transform.id(),
                params: piece.transform.encode_params()?,
                sources: piece
                    .sources
                    .iter()
                    .map(|name| {
                        let tensor = tensor(tensors, name)?;
                        Ok(files.source(&tensor.name, &tensor.file)?)
                    })
                    .collect::<MimoResult<_>>()?,
                axis: piece.axis,
                ranges: piece.ranges.clone(),
                stored_shape: piece.stored_shape.clone(),
            });
        }
    }
    records.sort_by_key(|record| record.arena_offset);
    let mut plan = WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: role.into(),
        alignment: IMAGE_ALIGNMENT,
        arena_bytes: image_bytes,
        views: vec![],
        components: vec![],
        records,
    };
    let identity = plan_identity(&plan, request);
    let mut listed: Vec<(u64, &PlannedView)> = offsets.iter().copied().zip(views).collect();
    listed.sort_by_key(|(offset, _)| *offset);
    let index = serde_json::json!({
        "schema": schema,
        "rank": request.rank,
        "image": image,
        "image_bytes": image_bytes,
        "direct_identity": identity,
        "views": listed.iter().map(|(offset, view)| serde_json::json!({
            "name": view.name, "offset": offset, "bytes": view.bytes, "dtype": view.dtype,
            "shape": view.shape, "layout_id": view.layout_id,
        })).collect::<Vec<_>>(),
    });
    plan.components = vec![NamedBytesV1 {
        name: index_role(role, request.rank),
        bytes: serde_json::to_vec(&index)?,
    }];
    Ok(plan)
}

/// The checkpoint's tensors as the packager's catalogs read them.
pub struct Catalogs {
    pub geometry: &'static Geometry,
    /// The text model: every `lm_head.weight` and `model.` tensor of the
    /// indexed shards but the MTP layers.
    pub text: BTreeMap<String, TensorRecord>,
    /// The vision, audio and speech embedding tensors of the indexed shards,
    /// and the audio tokenizer's encoder.
    pub encoders: BTreeMap<String, TensorRecord>,
    /// Each drafter's file, whole, when present.
    pub mtp: Option<BTreeMap<String, TensorRecord>>,
    pub dflash: Option<BTreeMap<String, TensorRecord>>,
    /// The checked DFlash mask embedding.
    pub mask: Option<[u8; crate::drafters::MASK_BYTES]>,
    /// The admitted projection operands of the text model and each drafter.
    pub admitted: Admitted,
    pub mtp_admitted: Option<Admitted>,
    pub dflash_admitted: Option<Admitted>,
}

fn header_tensors(file: &CheckpointFileV1) -> MimoResult<BTreeMap<String, TensorRecord>> {
    let mut prefix = (file.data_offset - 8).to_le_bytes().to_vec();
    prefix.extend_from_slice(&file.header);
    let header =
        runtime_weights::parse_safetensors_header(&mut prefix.as_slice(), file.byte_length)
            .map_err(|error| format!("{}: {error}", file.name))?;
    header
        .tensors()
        .iter()
        .map(|(name, info)| {
            Ok((
                name.clone(),
                TensorRecord {
                    name: name.clone(),
                    dtype: info.dtype(),
                    shape: info.shape().to_vec(),
                    file: file.name.clone(),
                    file_start: header
                        .data_offset()
                        .checked_add(info.data_offsets().0)
                        .ok_or("file offset overflow")?,
                    bytes: info.byte_len(),
                },
            ))
        })
        .collect()
}

/// The DFlash mask embedding: the BF16 storage entry of the checked archive.
pub fn mask_embedding(archive: &[u8]) -> MimoResult<[u8; crate::drafters::MASK_BYTES]> {
    if archive.len() > 2 * crate::drafters::MASK_BYTES {
        return Err("mask_embedding.pt exceeds 16 KiB".into());
    }
    if format!("{:x}", Sha256::digest(archive)) != MASK_ARCHIVE_SHA256 {
        return Err(
            "mask_embedding.pt identity differs from the checked BF16 tensor archive".into(),
        );
    }
    let name = b"mask_embedding/data/0";
    for offset in 0..archive.len().saturating_sub(30) {
        if archive[offset..].starts_with(b"PK\x03\x04") {
            let method = u16::from_le_bytes([archive[offset + 8], archive[offset + 9]]);
            let name_len =
                u16::from_le_bytes([archive[offset + 26], archive[offset + 27]]) as usize;
            let extra_len =
                u16::from_le_bytes([archive[offset + 28], archive[offset + 29]]) as usize;
            let start = offset + 30;
            let data = start + name_len + extra_len;
            if method == 0 && archive.get(start..start + name_len) == Some(name.as_slice()) {
                let storage = archive
                    .get(data..data + crate::drafters::MASK_BYTES)
                    .ok_or("short mask BF16 storage")?;
                return Ok(storage.try_into().expect("checked storage length"));
            }
        }
    }
    Err("mask_embedding.pt has no BF16 storage entry".into())
}

impl Catalogs {
    /// Reads the checkpoint description: the shard index, every header, the
    /// DFlash config and mask archive. With `dflash_required` (a package
    /// that drafts with DFlash) the DFlash drafter must check; otherwise a
    /// DFlash drafter that does not check is left out, unread and unplanned.
    pub fn read(request: &WeightPlanRequestV1, dflash_required: bool) -> MimoResult<Self> {
        let geometry = request_geometry(request)?;
        let index = runtime_weights::parse_safetensors_index(
            request
                .document(INDEX)
                .ok_or("the checkpoint has no shard index")?,
        )?;
        let shards: BTreeSet<&str> = index.shards().collect();
        let mut indexed = BTreeMap::new();
        let mut total = 0_u64;
        let mut others = BTreeMap::new();
        // Shard headers parse in parallel; they are checked in file order.
        let parse = |file: &CheckpointFileV1| {
            shards
                .contains(file.name.as_str())
                .then(|| header_tensors(file).map_err(|error| error.to_string()))
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(16));
        let mut headers = std::thread::scope(|scope| {
            let parsed: Vec<_> = request
                .files
                .chunks(request.files.len().div_ceil(threads).max(1))
                .map(|files| scope.spawn(move || files.iter().map(parse).collect::<Vec<_>>()))
                .collect();
            parsed
                .into_iter()
                .flat_map(|files| files.join().expect("a header parse panicked"))
                .collect::<Vec<_>>()
        })
        .into_iter();
        for file in &request.files {
            let header = headers.next().expect("one parse per file");
            if let Some(header) = header {
                for (name, tensor) in header? {
                    if index.shard_for(&name) != Some(file.name.as_str()) {
                        return Err(format!(
                            "index ownership mismatch for {name} in {}",
                            file.name
                        )
                        .into());
                    }
                    total = total
                        .checked_add(tensor.bytes)
                        .ok_or("index size overflow")?;
                    indexed.insert(name, tensor);
                }
            } else if !file.name.contains('/') {
                return Err(format!("unindexed safetensors shard {}", file.name).into());
            } else {
                others.insert(file.name.as_str(), file);
            }
        }
        if shards
            .iter()
            .any(|shard| !request.files.iter().any(|f| f.name == *shard))
        {
            return Err("an indexed shard is missing from the checkpoint".into());
        }
        if indexed.len() != index.tensor_count() || total != index.total_size() {
            return Err("index tensor count or total size disagrees with shard headers".into());
        }
        let mut text = BTreeMap::new();
        let mut encoders = BTreeMap::new();
        let mut mtp = BTreeMap::new();
        for (name, tensor) in indexed {
            if name.starts_with("model.mtp.") {
                mtp.insert(name, tensor);
            } else if name == "lm_head.weight" || name.starts_with("model.") {
                text.insert(name, tensor);
            } else if name.starts_with("visual.")
                || name.starts_with("audio_encoder.")
                || name.starts_with("speech_embeddings.")
            {
                encoders.insert(name, tensor);
            }
        }
        if let Some(file) = others.get(TOKENIZER_FILE) {
            for (name, tensor) in header_tensors(file)? {
                if name.starts_with("encoder.")
                    && !name.contains("._codebook.cluster_size")
                    && !name.contains("._codebook.embed_avg")
                    && !name.contains("._codebook.inited")
                {
                    encoders.insert(name, tensor);
                }
            }
        }
        let dflash = others
            .get(DFLASH_FILE)
            .map(|file| -> MimoResult<_> {
                let tensors = header_tensors(file)?;
                crate::dflash::validate_config_for(
                    geometry,
                    request
                        .document(DFLASH_CONFIG)
                        .ok_or("the DFlash drafter has no config.json")?,
                )?;
                let mask = mask_embedding(
                    request
                        .document(MASK_ARCHIVE)
                        .ok_or("the DFlash drafter has no mask_embedding.pt")?,
                )?;
                let admitted = crate::precision::admit_dflash(geometry, &tensors)?;
                Ok((tensors, mask, admitted))
            })
            .transpose();
        let dflash = match dflash {
            Ok(dflash) => dflash,
            Err(error) if dflash_required => return Err(error),
            Err(_) => None,
        };
        let (dflash, mask, dflash_admitted) = match dflash {
            Some((tensors, mask, admitted)) => (Some(tensors), Some(mask), Some(admitted)),
            None => (None, None, None),
        };
        let mtp_file = request.files.iter().any(|file| file.name == MTP_FILE);
        let mtp = (mtp_file && !mtp.is_empty()).then_some(mtp);
        // The packager's precision admission: every text and MTP tensor is
        // a MiMo role or a companion the shared importer read from the
        // export metadata, in an encoding MiMo serves.
        let metadata = crate::precision::export_metadata_from_config(
            request
                .document("config.json")
                .ok_or("the checkpoint has no config.json")?,
            request.document(crate::precision::HF_QUANT_CONFIG),
        )?;
        let admitted = crate::precision::admit(
            geometry,
            &metadata,
            &text,
            &crate::precision::text_roles(geometry, |layer| geometry.is_global(layer as u32)),
        )?;
        let mtp_admitted = mtp
            .as_ref()
            .map(|mtp| {
                crate::precision::admit(
                    geometry,
                    &metadata,
                    mtp,
                    &crate::precision::mtp_roles(geometry),
                )
            })
            .transpose()?;
        Ok(Self {
            geometry,
            text,
            encoders,
            mtp,
            dflash,
            mask,
            admitted,
            mtp_admitted,
            dflash_admitted,
        })
    }

    /// Checks the encoder tensors against the contract's names, dtypes and
    /// shapes.
    fn checked_encoders(&self) -> MimoResult<&BTreeMap<String, TensorRecord>> {
        let expected = crate::encoders::expected_for(self.geometry);
        if expected.len() != self.encoders.len() {
            return Err(format!(
                "the checkpoint has {} encoder tensors, the contract {}",
                self.encoders.len(),
                expected.len()
            )
            .into());
        }
        for (name, (dtype, shape, file)) in expected {
            let tensor = self
                .encoders
                .get(&name)
                .ok_or_else(|| format!("missing encoder tensor {name}"))?;
            if tensor.dtype.safetensors_name() != dtype
                || tensor.shape != shape
                || (file == TOKENIZER_FILE) != (tensor.file == TOKENIZER_FILE)
            {
                return Err(
                    format!("encoder tensor {name} has wrong dtype, shape or source").into(),
                );
            }
        }
        Ok(&self.encoders)
    }
}

fn request_geometry(request: &WeightPlanRequestV1) -> MimoResult<&'static Geometry> {
    let config: serde_json::Value = serde_json::from_slice(
        request
            .document("config.json")
            .ok_or("the checkpoint has no config.json")?,
    )?;
    Ok(crate::geometry::match_main(&config)?)
}

/// Checks the planned text views' per-rank sizes against the rank shape.
fn check_rank_shape(views: &[PlannedView], tp: Tp) -> MimoResult<()> {
    let rows = |name: &str, dimension: usize| -> MimoResult<u64> {
        views
            .iter()
            .find(|view| view.name == name)
            .and_then(|view| view.shape.get(dimension).copied())
            .ok_or_else(|| format!("no view {name}").into())
    };
    let dense_layer = tp.geometry.dense_layer_end() - 1;
    let moe_layer = tp.geometry.dense_layer_end();

    let checks = [
        (rows("lm_head.weight", 0)?, tp.vocab()),
        (
            rows(
                &format!("model.layers.{dense_layer}.mlp.gate_up_proj.weight"),
                0,
            )?,
            2 * tp.dense_intermediate(),
        ),
        (
            rows(
                &format!("model.layers.{moe_layer}.mlp.experts.gate_up_proj.weight"),
                1,
            )?,
            2 * tp.expert_intermediate(),
        ),
        (
            rows(
                &format!("model.layers.{moe_layer}.self_attn.o_proj.weight"),
                1,
            )?,
            tp.o_proj_k_for(tp.geometry.is_global(moe_layer)),
        ),
    ];
    if checks.iter().any(|(planned, derived)| planned != derived) {
        return Err("planned per-rank sizes differ from the rank shape".into());
    }
    Ok(())
}

/// The last request's catalogs, keyed by the checkpoint and its documents.
static CATALOGS: RequestCache<Catalogs> = RequestCache::new();

fn catalogs(request: &WeightPlanRequestV1) -> MimoResult<Arc<Catalogs>> {
    let dflash_required = packaged_drafter(request)? == Some(Drafter::Dflash);
    let mut key = RequestKey::new(b"catalogs", request);
    for document in &request.documents {
        key.bytes(document.name.as_bytes());
        key.bytes(&document.bytes);
    }
    key.bytes(&[u8::from(dflash_required)]);
    CATALOGS.get_or_build(key.finish(), || Catalogs::read(request, dflash_required))
}

/// The drafter a package names: its drafter-kind component's (a package
/// without rank images) or its rank index's; none for a package that
/// names neither.
fn packaged_drafter(request: &WeightPlanRequestV1) -> MimoResult<Option<Drafter>> {
    if let Some(bytes) = request.component(DRAFTER_KIND_ROLE) {
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        return match value["drafter"].as_str() {
            Some("dflash") => Ok(Some(Drafter::Dflash)),
            Some("mtp") => Ok(Some(Drafter::Mtp)),
            _ => Err("the drafter-kind component names no known drafter".into()),
        };
    }
    let packaged = request
        .component(&index_role(DRAFTER_IMAGE_ROLE, 0))
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(bytes).ok());
    Ok(packaged.map(|index| {
        let dflash = index["views"]
            .as_array()
            .is_some_and(|views| views.iter().any(|v| v["name"] == "mask_embedding.weight"));
        if dflash {
            Drafter::Dflash
        } else {
            Drafter::Mtp
        }
    }))
}

/// The drafter a package holds: the one it names, else DFlash when the
/// checkpoint has a DFlash drafter that checks.
fn drafter_kind(request: &WeightPlanRequestV1, catalogs: &Catalogs) -> MimoResult<Drafter> {
    Ok(match packaged_drafter(request)? {
        Some(kind) => kind,
        None if catalogs.dflash.is_some() => Drafter::Dflash,
        None => Drafter::Mtp,
    })
}

/// The bundle's `chop_bundle_weight_plan_v1`: the text, drafter and encoder
/// images; other roles are declined.
pub fn plan(request: &WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String> {
    plan_layers(request, None).map_err(|error| format!("MiMo weight plan: {error}"))
}

/// [`plan`], with the text image holding only `layers` (a layer subset,
/// planned through the contract for checks and benchmarks; the bundle
/// serves only whole models, which the full contract admits).
pub fn plan_layers(
    request: &WeightPlanRequestV1,
    layers: Option<&BTreeSet<usize>>,
) -> MimoResult<Option<WeightPlanV1>> {
    // An empty role asks for the primary image: a package without rank
    // images serves the text model alone.
    let role = match request.image_role.as_str() {
        "" => TEXT_IMAGE_ROLE,
        role => role,
    };
    if ![TEXT_IMAGE_ROLE, DRAFTER_IMAGE_ROLE, ENCODER_IMAGE_ROLE].contains(&role) {
        return Ok(None);
    }
    if !request.host_views.is_empty() {
        return Err("MiMo places no views in host memory".into());
    }
    let tp = Tp::of_geometry(request_geometry(request)?, request.ranks)?;
    let catalogs = catalogs(request)?;
    plan_catalogs(request, layers, role, tp, &catalogs)
}

fn plan_catalogs(
    request: &WeightPlanRequestV1,
    layers: Option<&BTreeSet<usize>>,
    role: &str,
    tp: Tp,
    catalogs: &Catalogs,
) -> MimoResult<Option<WeightPlanV1>> {
    let rank = request.rank as usize;
    let (tensors, views, schema, image) = match role {
        TEXT_IMAGE_ROLE => {
            let views = image_views(request, Scope::Text, &catalogs.text, &catalogs.admitted, tp)?
                [rank]
                .clone();
            let views = text_layers(views, layers);
            if layers.is_none() {
                check_rank_shape(&views, tp)?;
            }
            (
                &catalogs.text,
                views,
                crate::index::SCHEMA,
                format!("rank{rank}.img"),
            )
        }
        DRAFTER_IMAGE_ROLE => {
            let kind = drafter_kind(request, catalogs)?;
            let (tensors, admitted) = match kind {
                Drafter::Dflash => catalogs
                    .dflash
                    .as_ref()
                    .zip(catalogs.dflash_admitted.as_ref()),
                Drafter::Mtp => catalogs.mtp.as_ref().zip(catalogs.mtp_admitted.as_ref()),
            }
            .ok_or("the checkpoint has no drafter of the package's kind")?;
            let views =
                image_views(request, drafter_scope(kind), tensors, admitted, tp)?[rank].clone();
            let views = with_mask(views, kind, catalogs.mask.as_ref(), tp)?;
            (
                tensors,
                views,
                crate::index::SCHEMA,
                format!("rank{rank}.img"),
            )
        }
        _ => (
            catalogs.checked_encoders()?,
            encoder_views_for(
                tp.geometry,
                encoder_set(request)?,
                catalogs.checked_encoders()?,
            )?,
            crate::encoder_index::SCHEMA,
            format!("encoder-rank{rank}.img"),
        ),
    };
    let (mut offsets, mut image_bytes) = sequential_offsets(&views);
    if let Some(packaged) = request
        .component(&index_role(role, request.rank))
        .and_then(|index| package_offsets(&views, index))
    {
        (offsets, image_bytes) = packaged;
    }
    let plan = assemble(
        request,
        role,
        tensors,
        &views,
        &offsets,
        image_bytes,
        schema,
        &image,
    )?;
    // The bundle's own admission of the planned index.
    let index = &plan.components[0].bytes;
    match role {
        TEXT_IMAGE_ROLE => {
            let parsed = crate::index::parse(index, request.rank)?;
            if layers.is_none() {
                crate::index::validate_text(&parsed, tp)?;
            }
        }
        DRAFTER_IMAGE_ROLE => {
            let parsed = crate::index::parse(index, request.rank)?;
            if crate::dflash::is_dflash(&parsed) {
                crate::dflash::validate(&parsed, tp)?;
            } else {
                crate::mtp::validate(&parsed, tp)?;
            }
        }
        _ => {
            crate::encoder_index::parse_for(tp.geometry, index, request.rank)?;
        }
    }
    Ok(Some(plan))
}

#[cfg(test)]
mod slice_tests {
    use super::*;

    /// Dense fragment pieces stream 16-row strips; a rank share that is not
    /// a whole number of strips is refused before packing.
    #[test]
    fn dense_fragments_refuse_partial_row_strips() {
        let record = |rows: std::ops::Range<u64>| WeightRecord {
            transform: WeightTransform::Fragment(
                kernel_core::transform::fragment::FragmentWeightFormat::Fp8E4M3,
            ),
            sources: vec![kernel_core::weight_transform::SourceTensor {
                dtype: "F8_E4M3".into(),
                shape: vec![128, 256],
                byte_length: 128 * 256,
            }],
            axis: Some(0),
            ranges: vec![rows.clone()],
            stored_shape: vec![rows.end - rows.start, 256],
        };
        let name = "model.layers.0.mlp.gate_proj.weight";
        assert!(check_fragment_strips(&record(32..64), name).is_ok());
        let refused = check_fragment_strips(&record(24..48), name)
            .err()
            .unwrap()
            .to_string();
        assert!(
            refused.contains("fragment rows must align to 16"),
            "{refused}"
        );
    }
}

#[cfg(test)]
mod encoder_plan_tests {
    use super::*;

    #[test]
    fn encoder_role_plans_every_listed_geometry() {
        for geometry in crate::geometry::GEOMETRIES {
            let encoders: BTreeMap<_, _> = crate::encoders::expected_for(geometry)
                .into_iter()
                .map(|(name, (dtype, shape, file))| {
                    let dtype = match dtype {
                        "BF16" => DType::Bf16,
                        "F32" => DType::F32,
                        _ => panic!("encoder dtype"),
                    };
                    let bytes =
                        shape.iter().product::<u64>() * if dtype == DType::F32 { 4 } else { 2 };
                    (
                        name.clone(),
                        TensorRecord {
                            name,
                            dtype,
                            shape,
                            file: file.into(),
                            file_start: 0,
                            bytes,
                        },
                    )
                })
                .collect();
            let files: Vec<CheckpointFileV1> = encoders
                .values()
                .map(|tensor| tensor.file.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|name| CheckpointFileV1 {
                    name,
                    ..Default::default()
                })
                .collect();
            let catalogs = Catalogs {
                geometry,
                text: BTreeMap::new(),
                encoders,
                mtp: None,
                dflash: None,
                mask: None,
                admitted: BTreeMap::new(),
                mtp_admitted: None,
                dflash_admitted: None,
            };
            for ranks in Tp::SUPPORTED {
                let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                for rank in 0..tp.ranks() {
                    let request = WeightPlanRequestV1 {
                        image_role: ENCODER_IMAGE_ROLE.into(),
                        ranks: tp.ranks(),
                        rank,
                        files: files.clone(),
                        ..Default::default()
                    };
                    let plan = plan_catalogs(&request, None, ENCODER_IMAGE_ROLE, tp, &catalogs)
                        .unwrap()
                        .unwrap();
                    let parsed =
                        crate::encoder_index::parse_for(geometry, &plan.components[0].bytes, rank)
                            .unwrap();
                    assert_eq!(
                        parsed.image_bytes,
                        crate::encoders::Encoders::ALL.image_bytes_for(geometry)
                    );
                }
            }
        }
    }
}
