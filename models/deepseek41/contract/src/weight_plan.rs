//! DeepSeek V4.1 Flash's weight plan for direct checkpoint loading: each
//! rank's text image (`serving::RANK_IMAGE_ROLE`) planned from the
//! checkpoint's headers, which the bundle exports as
//! `chop_bundle_weight_plan_v1`.
//!
//! The image is the rank derivation of the projection declaration over the
//! bound checkpoint operands (`projection_spec::derive`):
//!
//! - every projection's resident sections, under the record policy of its
//!   family ([`policy`]): a dense or grouped dense projection is one packed
//!   operand per section (block-32 FP8 resident as MXFP8 rows, E4M3 codes
//!   with one UE8M0 scale per row and 32 K), fused dense operands one
//!   packed piece each, an expert bank expert-major packed pieces, a lookup
//!   table its rows;
//! - every side tensor's rank copy (norms, the mixers, the router biases,
//!   and the attention sinks of the rank's heads).
//!
//! The Engram tables are not in the image: the host gathers their rows
//! from the mapped checkpoint (F1). The vision tower and aligner belong to
//! the encoder image.
//!
//! Views are named as the bundle binds them ([`view_name`]) and ordered by
//! name; each starts on [`VIEW_ALIGNMENT`]. Each record is one derived
//! section or piece. A planned rank has no image file: the rank index each
//! plan carries names the plan's identity (`direct_identity`) instead of an
//! image digest, and the bundle checks it as it checks a rank file's digest.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use kernel_core::precision::format::KernelFamily;
use kernel_core::weight_transform::{PackedSection, WeightTransform};
use model_compiler_sdk::quantization::declared::{self, Admitted};
use model_compiler_sdk::quantization::import::ImportedOperand;
use model_compiler_sdk::quantization::projection::metadata::HeaderCompanionRules;
use model_compiler_sdk::quantization::projection::{RecordPolicy, bind_record_policy, records};
use model_compiler_sdk::weight_plan::{RequestCache, RequestFiles, RequestKey, plan_identity};
use projection_spec::ProjectionKind;
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata};
use projection_spec::derive::{DerivationInput, Topology, hooks::VariantDeriver};
use projection_spec::registry::Registry;
use runtime_bundle_abi::weight_plan::{
    NamedBytesV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use serde_json::{Value, json};

use crate::geometry::Geometry;
use crate::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE, index_role};
use crate::{admission, projections};

/// The alignment of a planned arena, which a CUDA allocation guarantees.
pub const IMAGE_ALIGNMENT: u64 = 256;
/// Every view of the image starts on this alignment.
pub const VIEW_ALIGNMENT: u64 = 4096;

const CONFIG: &str = "config.json";
const INDEX: &str = "model.safetensors.index.json";

/// The projections whose rows the host gathers from the mapped checkpoint
/// instead of the image.
const HOST_GATHERED: [&str; 2] = ["engram.table_0", "engram.table_1"];

/// The owner that validates the vision tower's and aligner's tensors. Until
/// the encoder declaration lands, their names stand in for its validation
/// (as the checkpoint binding test does).
const ENCODER_OWNER: &str = "encoder-spec";

fn encoder_tensor(name: &str) -> bool {
    name.starts_with("vision.") || name.starts_with("aligner.") || name.starts_with("image_")
}

/// The declaration registers no variants.
struct NoVariants;
impl VariantDeriver for NoVariants {}

/// One record of a view, at `offset` within it.
#[derive(Clone, Debug, PartialEq)]
struct Piece {
    offset: u64,
    bytes: u64,
    transform: WeightTransform,
    sources: Vec<String>,
    axis: Option<u32>,
    ranges: Vec<Range<u64>>,
    stored_shape: Vec<u64>,
}

/// One view of a rank's image: what the bundle binds by `name`.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub name: String,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout_id: String,
    pieces: Vec<Piece>,
}

/// The record policy of a derived operation of `family` reading `operands`
/// source tensors: expert banks as expert-major packed pieces (a whole
/// bank's sources would pass the 64 sources one record may name), fused
/// dense operands (the shared expert's w1 and w3) as one packed piece per
/// operand, which the bundle binds as its own view, lookup tables as rows,
/// every other projection one packed operand per resident section.
pub fn policy(family: KernelFamily, operands: u32) -> Option<RecordPolicy> {
    match family {
        KernelFamily::GroupedExpert => Some(RecordPolicy::PackedPieces),
        KernelFamily::Dense if operands > 1 => Some(RecordPolicy::PackedPieces),
        KernelFamily::Lookup => Some(RecordPolicy::Sections(
            model_compiler_sdk::quantization::format::RecordLayout::Rows,
        )),
        _ => None,
    }
}

/// The view the bundle binds for `section` of the projection read from
/// `tensor` (a source tensor of the record): a weight's codes keep its
/// tensor name and its scales the companion name (`{stem}.scale`), fused
/// operands each their own; an expert bank is one view of every expert the
/// rank holds,
/// `{stem}.ffn.experts.{gate_up|down}.{weight|scale}`.
pub fn view_name(
    kind: &ProjectionKind,
    operands: u32,
    tensor: &str,
    section: PackedSection,
) -> Result<String, String> {
    let stem = tensor
        .strip_suffix(".weight")
        .ok_or_else(|| format!("{tensor}: a projection source ends in .weight"))?;
    let part = match section {
        PackedSection::Codes => "weight",
        PackedSection::LocalScales => "scale",
        PackedSection::GlobalScales => {
            return Err(format!("{tensor}: no DeepSeek weight has a global scale"));
        }
    };
    if let ProjectionKind::ExpertBank { .. } = kind {
        let (prefix, _) = stem
            .split_once(".experts.")
            .ok_or_else(|| format!("{tensor}: an expert source names its expert"))?;
        let bank = if operands == 2 { "gate_up" } else { "down" };
        return Ok(format!("{prefix}.experts.{bank}.{part}"));
    }
    // Fused dense operands (the shared expert's w1 and w3) are each their
    // own view.
    Ok(format!("{stem}.{part}"))
}

/// The checkpoint as the plan reads it: its admitted geometry, binding
/// metadata and the declaration bound to it.
struct Checkpoint {
    geometry: &'static Geometry,
    metadata: CheckpointMetadata,
    declaration: projection_spec::Declaration,
    spec: projection_spec::Geometry,
    bound: BoundDeclaration,
    /// The safetensors file holding each tensor.
    files: BTreeMap<String, String>,
}

fn document(request: &WeightPlanRequestV1, name: &str) -> Result<Value, String> {
    let bytes = request
        .document(name)
        .ok_or_else(|| format!("the checkpoint has no {name}"))?;
    serde_json::from_slice(bytes).map_err(|e| format!("{name}: {e}"))
}

impl Checkpoint {
    /// Admits the request's config, reads every header and binds the
    /// declaration of the admitted geometry to them.
    fn read(request: &WeightPlanRequestV1) -> Result<Self, String> {
        let config = document(request, CONFIG)?;
        let index = document(request, INDEX)?;
        let geometry = admission::admit(&config)?;
        let headers = request
            .files
            .iter()
            .map(|file| {
                let header: Value = serde_json::from_slice(&file.header)
                    .map_err(|e| format!("{} header: {e}", file.name))?;
                Ok((file.name.clone(), header))
            })
            .collect::<Result<BTreeMap<_, _>, String>>()?;
        let quantization = config
            .get("quantization_config")
            .cloned()
            .unwrap_or(Value::Null);
        let mut metadata = CheckpointMetadata::from_headers(&headers, Some(&index), quantization)
            .map_err(|e| e.0)?;
        let encoder: BTreeSet<_> = metadata
            .tensors
            .keys()
            .filter(|name| encoder_tensor(name))
            .cloned()
            .collect();
        metadata
            .validated_owners
            .insert(ENCODER_OWNER.into(), encoder);
        let mut files = BTreeMap::new();
        for (file, header) in &headers {
            let Value::Object(tensors) = header else {
                return Err(format!("{file} header is not an object"));
            };
            for name in tensors.keys().filter(|name| *name != "__metadata__") {
                files.insert(name.clone(), file.clone());
            }
        }
        let spec = projections::geometry(geometry);
        // Binding and derivation read every projection's expanded uses.
        let declaration = projections::declaration(geometry)
            .and_then(|declaration| declaration.expanded_profiles(&spec))
            .map_err(|e| e.0)?;
        let mut rules = HeaderCompanionRules::new(&metadata, None).map_err(|e| e.0)?;
        rules.roles = projections::header_roles(&declaration);
        let bound = declaration
            .bind(&spec, &Registry::default(), &metadata, &rules)
            .map_err(|e| e.0)?;
        Ok(Self {
            geometry,
            metadata,
            declaration,
            spec,
            bound,
            files,
        })
    }

    /// The bound operands as the derivation's admitted imports, by their
    /// codes tensor.
    fn admitted(&self) -> Admitted {
        let mut admitted = Admitted::new();
        for frame in &self.bound.operands {
            let weight = &frame.imported.weight;
            let source_tensors = std::iter::once(&weight.codes)
                .chain(weight.local_scale.iter().map(|s| &s.tensor))
                .chain(weight.global_scale.iter().map(|s| &s.tensor))
                .map(|span| span.tensor.clone())
                .collect();
            admitted.insert(
                weight.codes.tensor.clone(),
                ImportedOperand {
                    weight: weight.clone(),
                    activation: None,
                    source_tensors,
                },
            );
        }
        admitted
    }

    /// Rank `rank` of `ranks`'s views, ordered by name.
    fn views(
        &self,
        imported: &projection_spec::derive::ImportedProjections,
        ranks: u32,
        rank: u32,
    ) -> Result<Vec<View>, String> {
        let registry = Registry::default();
        let topology = Topology { ranks, rank };
        let graph = projection_spec::derive::derive(DerivationInput {
            declaration: &self.declaration,
            bound: &self.bound,
            geometry: &self.spec,
            registry: &registry,
            topology,
            imported,
            variants: &NoVariants,
        })
        .map_err(|e| e.0)?;
        let mut views = BTreeMap::<String, View>::new();
        for operation in graph.selected_operations() {
            let projection = &operation.instance.projection;
            if HOST_GATHERED.contains(&projection.as_str()) {
                continue;
            }
            let declared = self
                .declaration
                .projections
                .iter()
                .find(|p| p.id == *projection)
                .ok_or_else(|| format!("{projection}: derived without a declaration"))?;
            let at = |e: model_compiler_sdk::CompilerError| format!("{projection}: {e}");
            let operation = match policy(operation.family, declared.operand_count()) {
                Some(policy) => bind_record_policy(operation.clone(), policy).map_err(at)?,
                None => operation.clone(),
            };
            let layouts: BTreeMap<_, _> = operation
                .views
                .iter()
                .map(|view| (view.id.clone(), view.layout.clone()))
                .collect();
            for record in records(&operation).map_err(at)? {
                let name = view_name(
                    &declared.kind,
                    declared.operand_count(),
                    &record.tensors[0],
                    record.section,
                )?;
                let codes = record
                    .record
                    .sources
                    .first()
                    .ok_or_else(|| format!("{name}: a record reads no tensor"))?;
                // DeepSeek's only local scale is UE8M0.
                let dtype = match record.section {
                    PackedSection::Codes => codes.dtype.clone(),
                    _ => "F8_E8M0".to_owned(),
                };
                let view = views.entry(name.clone()).or_insert_with(|| View {
                    name: name.clone(),
                    bytes: 0,
                    dtype,
                    shape: Vec::new(),
                    layout_id: layouts.get(&record.view).cloned().unwrap_or_default(),
                    pieces: Vec::new(),
                });
                if view.pieces.iter().any(|p| p.offset == record.view_offset) {
                    continue;
                }
                let bytes = record.record.output_bytes().map_err(|e| e.to_string())?;
                view.pieces.push(Piece {
                    offset: record.view_offset,
                    bytes,
                    transform: record.record.transform,
                    sources: record.tensors,
                    axis: record.record.axis,
                    ranges: record.record.ranges,
                    stored_shape: record.record.stored_shape,
                });
            }
        }
        for view in views.values_mut() {
            view.pieces.sort_by_key(|p| p.offset);
            // A view of one operand of a fused operation starts where its
            // piece does in the operation's section.
            let start = view.pieces.first().map_or(0, |p| p.offset);
            for piece in &mut view.pieces {
                piece.offset -= start;
            }
            let mut end = 0;
            for piece in &view.pieces {
                if piece.offset != end {
                    return Err(format!(
                        "{}: pieces leave a gap or overlap at {}",
                        view.name, piece.offset
                    ));
                }
                end += piece.bytes;
            }
            view.bytes = end;
            view.shape = vec![end / element_bytes(&view.dtype)?];
        }
        let sides = model_compiler_sdk::quantization::projection::side_records(
            &self.declaration,
            &self.bound,
            &self.spec,
            topology,
            &self.metadata,
        )
        .map_err(|e| e.to_string())?;
        for (name, record) in sides {
            let info = self
                .metadata
                .tensors
                .get(&name)
                .ok_or_else(|| format!("{name}: a side tensor without a header"))?;
            let bytes = record.output_bytes().map_err(|e| e.to_string())?;
            // Padding past the tensor's end reads nothing.
            let sources = if record.sources.is_empty() {
                Vec::new()
            } else {
                vec![name.clone()]
            };
            let stored_shape = if record.stored_shape.is_empty() {
                vec![1]
            } else {
                record.stored_shape.clone()
            };
            let previous = views.insert(
                name.clone(),
                View {
                    layout_id: format!("{}_row_major_v1", info.dtype.to_lowercase()),
                    name: name.clone(),
                    bytes,
                    dtype: info.dtype.clone(),
                    shape: stored_shape.clone(),
                    pieces: vec![Piece {
                        offset: 0,
                        bytes,
                        transform: record.transform,
                        sources,
                        axis: record.axis,
                        ranges: record.ranges,
                        stored_shape,
                    }],
                },
            );
            if previous.is_some() {
                return Err(format!("{name}: a side view named as a projection view"));
            }
        }
        Ok(views.into_values().collect())
    }
}

/// The bytes of one element of the safetensors dtype `dtype`.
fn element_bytes(dtype: &str) -> Result<u64, String> {
    Ok(match dtype {
        "U8" | "I8" | "F8_E4M3" | "F8_E8M0" => 1,
        "BF16" | "F16" => 2,
        "F32" | "I32" => 4,
        "I64" => 8,
        other => return Err(format!("a view of {other} elements")),
    })
}

/// Every rank's views of one checkpoint and rank count, built once for all
/// ranks of a request, and the safetensors file holding each tensor.
struct Planned {
    ranks: Vec<Vec<View>>,
    files: BTreeMap<String, String>,
}

static PLANS: RequestCache<Planned> = RequestCache::new();

fn planned(request: &WeightPlanRequestV1) -> Result<Arc<Planned>, String> {
    let mut key = RequestKey::new(b"deepseek41.weight-plan.v1", request);
    key.optional(request.document(CONFIG));
    key.optional(request.document(INDEX));
    key.bytes(&request.ranks.to_le_bytes());
    PLANS.get_or_build(key.finish(), || {
        let checkpoint = Checkpoint::read(request)?;
        admission::check_ranks(checkpoint.geometry, request.ranks)?;
        let admitted = checkpoint.admitted();
        let imported = declared::imported(
            &checkpoint.declaration,
            &checkpoint.bound,
            &checkpoint.spec,
            &admitted,
        )
        .map_err(|e| e.0)?;
        let ranks = std::thread::scope(|threads| {
            let ranks: Vec<_> = (0..request.ranks)
                .map(|rank| {
                    let (checkpoint, imported) = (&checkpoint, &imported);
                    threads.spawn(move || checkpoint.views(imported, request.ranks, rank))
                })
                .collect();
            ranks
                .into_iter()
                .map(|rank| rank.join().map_err(|_| "a rank plan panicked")?)
                .collect::<Result<Vec<_>, String>>()
        })?;
        let files = checkpoint.files;
        // Every source the views read is in the checkpoint's files.
        for view in ranks.iter().flatten() {
            for piece in &view.pieces {
                for source in &piece.sources {
                    if !files.contains_key(source) {
                        return Err(format!("{}: {source} is in no file", view.name));
                    }
                }
            }
        }
        Ok(Planned { ranks, files })
    })
}

/// The view offsets of `views` laid out in order, each on
/// [`VIEW_ALIGNMENT`], and the arena's bytes.
fn offsets(views: &[View]) -> (Vec<u64>, u64) {
    let mut end = 0_u64;
    let offsets = views
        .iter()
        .map(|view| {
            let offset = end.next_multiple_of(VIEW_ALIGNMENT);
            end = offset + view.bytes;
            offset
        })
        .collect();
    (offsets, end.next_multiple_of(VIEW_ALIGNMENT))
}

/// The rank index of `views` at `offsets`, naming the plan's identity.
fn index(
    rank: u32,
    views: &[View],
    offsets: &[u64],
    image_bytes: u64,
    identity: &str,
) -> Result<Vec<u8>, String> {
    let views: Vec<Value> = views
        .iter()
        .zip(offsets)
        .map(|(view, offset)| {
            json!({
                "name": view.name,
                "offset": offset,
                "bytes": view.bytes,
                "dtype": view.dtype,
                "shape": view.shape,
                "layout_id": view.layout_id,
            })
        })
        .collect();
    serde_json::to_vec(&json!({
        "schema": INDEX_SCHEMA,
        "rank": rank,
        "image": format!("rank{rank}.img"),
        "image_bytes": image_bytes,
        "direct_identity": identity,
        "views": views,
    }))
    .map_err(|e| e.to_string())
}

/// The text image's plan for the request's rank, or `None` for another
/// image role. The primary image (an empty role) is the text image.
pub fn plan(request: &WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String> {
    if !(request.image_role.is_empty() || request.image_role == RANK_IMAGE_ROLE) {
        return Ok(None);
    }
    if request.rank >= request.ranks {
        return Err(format!(
            "rank {} is past the {} ranks",
            request.rank, request.ranks
        ));
    }
    let planned = planned(request)?;
    let views = &planned.ranks[request.rank as usize];
    let (offsets, arena_bytes) = offsets(views);
    let files = RequestFiles::new(request);
    let mut records = Vec::new();
    for (view, offset) in views.iter().zip(&offsets) {
        for piece in &view.pieces {
            let sources = piece
                .sources
                .iter()
                .map(|tensor| {
                    let file = planned
                        .files
                        .get(tensor)
                        .ok_or_else(|| format!("{tensor} is in no file"))?;
                    files.source(tensor, file)
                })
                .collect::<Result<Vec<_>, String>>()?;
            records.push(WeightRecordV1 {
                arena_offset: offset + piece.offset,
                byte_length: piece.bytes,
                transform: piece.transform.id(),
                params: piece.transform.encode_params().map_err(|e| e.to_string())?,
                sources,
                axis: piece.axis,
                ranges: piece.ranges.clone(),
                stored_shape: piece.stored_shape.clone(),
            });
        }
    }
    let mut plan = WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: RANK_IMAGE_ROLE.into(),
        alignment: IMAGE_ALIGNMENT,
        arena_bytes,
        views: Vec::new(),
        components: Vec::new(),
        records,
    };
    let identity = plan_identity(&plan, request);
    plan.components = vec![NamedBytesV1 {
        name: index_role(request.rank),
        bytes: index(request.rank, views, &offsets, arena_bytes, &identity)?,
    }];
    Ok(Some(plan))
}
