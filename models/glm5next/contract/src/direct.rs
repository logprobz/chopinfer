//! The direct weight plan of the text rank image (`chop.weight-plan.v1`):
//! every text tensor's share on one rank (`crate::views::rank_views`), each
//! at a 4096-aligned offset of the rank arena, and the rank index the
//! bundle reads its views from.
//!
//! Side tensors are stored as the engines read them: the KDA convolution
//! weights (the engine's `weight_q`, `weight_k` and `weight_v`) and the mHC
//! base and scale vectors in F32, widened when the checkpoint stores them
//! in BF16, `kv_b_proj` as the rank's rows `[heads, nope + v,
//! latent]` (each head's key half, then its value half, at a head stride of
//! `(nope + v) * latent` elements), the rest unchanged. The chained MoE
//! engine's weights are packed as it reads them (`chained_rows_v1`): each
//! expert's gate rows then up rows of the rank's slice, codes and block
//! scales in separate views, experts back to back; the experts' down
//! weights (the rank's columns) back to back; and the dense MLP's and
//! shared expert's gate then up halves. Other
//! projection weights are stored as checkpoint rows until the projection
//! API's layouts bind them (binding point: their layout is
//! `checkpoint_rows_v1`); a lowered projection's weight view is padded
//! with zero rows to its format's row tile, the rows its operation reads
//! (`crate::precision::stored_rows`).
//!
//! The checkpoint's quantization layout (`crate::quantization`) decides what
//! the packed expert views hold. Block FP8: codes `[experts, rows, k]` and
//! F32 `weight_scale_inv` per 128 x 128 block. NVFP4: U8 E2M1 pairs
//! `[experts, rows, k / 2]`, E4M3 `weight_scale` `[experts, rows, k / 16]`
//! and F32 `weight_scale_2` `[experts, projections]` (gate and up, or down);
//! the static `input_scale` is not stored. MXFP8: E4M3 codes and U8 E8M0
//! `weight_scale` `[experts, rows, k / 32]`.

use std::collections::HashMap;

use kernel_core::weight_transform::WeightTransform;
use model_compiler_sdk::weight_plan::{RequestFiles, plan_identity};
use runtime_bundle_abi::weight_plan::{
    NamedBytesV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use serde_json::{Value, json};

use crate::catalog::{DType, Placement, TensorSpec};
use crate::contract::Shape;
use crate::quantization::{FP8_BLOCK128, Layout};
use crate::views::{Binding, Engine, RankView, checkpoint_rank_views};

/// The compiled image role of the text rank image.
pub const RANK_IMAGE_ROLE: &str = "glm5next.rank-images.v1";
/// The schema every rank index is written under.
pub const INDEX_SCHEMA: &str = "chopinfer.glm5next.rank-image.v1";
/// View offsets and the image length are multiples of this.
pub const VIEW_ALIGNMENT: u64 = 4096;
/// The rank image's arena alignment: what a stream-ordered CUDA pool
/// guarantees every allocation (qwen38's and mimo26's images ask the same),
/// so the arena need not be its pool's first allocation.
pub const ARENA_ALIGNMENT: u64 = 256;
const CHECKPOINT_INDEX: &str = "model.safetensors.index.json";

pub fn index_role(rank: u32) -> String {
    bundle_runtime::index::index_role("glm5next", bundle_runtime::index::ImageKind::Text, rank)
}

/// How one view is stored in the rank image.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stored {
    pub view: RankView,
    pub offset: u64,
    pub transform: WeightTransform,
    pub dtype: DType,
    pub layout_id: &'static str,
    /// The checkpoint axis the rank's slice runs along, and its range.
    pub slice: Option<(u32, std::ops::Range<u64>)>,
    /// The checkpoint pieces it holds back to back: the view itself, or
    /// the members of a packed view.
    pub pieces: Vec<Piece>,
}

/// One checkpoint tensor's slice stored at `offset`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Piece {
    pub tensor: String,
    pub offset: u64,
    pub shape: Vec<u64>,
    pub slice: Option<(u32, std::ops::Range<u64>)>,
}

impl Stored {
    pub fn bytes(&self) -> u64 {
        self.view.shape.iter().product::<u64>() * self.dtype.bytes()
    }
}

/// Whether the engine reads side tensor `name` in F32: the KDA convolutions
/// and the mHC base and scale vectors, whichever dtype the checkpoint
/// stores them in.
fn f32_side(name: &str) -> bool {
    crate::quantization::SideFamily::of(name).is_some()
}

/// A packed view's members by their place in it.
type Members = Vec<((u64, u8), RankView)>;

/// The packed view a chained MoE weight or scale belongs to, and its place
/// in it: (expert, gate 0 or up 1). Gate/up views of the routed experts,
/// the dense MLP and the shared expert, and the experts' down views.
fn packed(name: &str) -> Option<(String, (u64, u8))> {
    let (stem, suffix) = [
        "weight_scale_inv",
        "weight_scale_2",
        "weight_scale",
        "weight",
    ]
    .into_iter()
    .find_map(|suffix| Some((name.strip_suffix(&format!(".{suffix}"))?, suffix)))?;
    let (base, part) = stem.rsplit_once('.')?;
    let half = match part {
        "gate_proj" => 0,
        "up_proj" => 1,
        "down_proj" => 0,
        _ => return None,
    };
    let down = part == "down_proj";
    if let Some((mlp, expert)) = base.rsplit_once(".experts.") {
        let expert: u64 = expert.parse().ok()?;
        let kind = if down { "down" } else { "gate_up" };
        return Some((format!("{mlp}.experts.{kind}.{suffix}"), (expert, half)));
    }
    if down || !(base.ends_with(".mlp") || base.ends_with(".mlp.shared_experts")) {
        return None;
    }
    Some((format!("{base}.gate_up.{suffix}"), (0, half)))
}

/// Every text view of `rank` of the block-FP8 checkpoint.
pub fn layout(shape: Shape, rank: u32) -> Result<Vec<Stored>, String> {
    checkpoint_layout(shape, &FP8_BLOCK128, rank)
}

/// Every text view of `rank` of a checkpoint stored in `quantization`,
/// placed in catalog order; a packed view sits where its first member
/// would.
pub fn checkpoint_layout(
    shape: Shape,
    quantization: &Layout,
    rank: u32,
) -> Result<Vec<Stored>, String> {
    let catalog = crate::catalog::checkpoint_catalog(shape.geometry, quantization);
    let catalog: HashMap<&str, _> = catalog
        .iter()
        .map(|spec| (spec.name.as_str(), spec))
        .collect();
    let views = checkpoint_rank_views(shape, quantization, rank)?;
    // Members of each packed view, by packed name.
    let mut groups: HashMap<String, Members> = HashMap::new();
    for view in &views {
        if let Some((name, key)) = packed(&view.name) {
            groups.entry(name).or_default().push((key, view.clone()));
        }
    }
    let slice_of = |view: &RankView| -> Result<_, String> {
        let spec = catalog
            .get(view.name.as_str())
            .ok_or_else(|| format!("{} is not in the catalog", view.name))?;
        Ok(match spec.placement {
            Placement::Split { dim } | Placement::Expert { dim, .. } => {
                Some((dim as u32, view.start..view.start + view.shape[dim]))
            }
            Placement::Replicated => None,
        })
    };
    let mut offset = 0;
    let mut stored = Vec::new();
    for view in views {
        if let Some((name, _)) = packed(&view.name) {
            let Some(mut members) = groups.remove(&name) else {
                continue;
            };
            members.sort_by_key(|(key, _)| *key);
            let experts = members
                .iter()
                .map(|((expert, _), _)| expert)
                .collect::<std::collections::BTreeSet<_>>()
                .len() as u64;
            let rows = || {
                members
                    .iter()
                    .map(|(_, member)| member.shape[0])
                    .sum::<u64>()
                    / experts
            };
            let packed_shape = if view.shape.is_empty() {
                // Scalar globals: one per expert and projection.
                vec![experts, members.len() as u64 / experts]
            } else if name.contains(".experts.") {
                vec![experts, rows(), view.shape[1]]
            } else {
                vec![rows(), view.shape[1]]
            };
            let mut at = offset;
            let mut pieces = Vec::new();
            for (_, member) in &members {
                let bytes = member.bytes();
                pieces.push(Piece {
                    tensor: member.name.clone(),
                    offset: at,
                    shape: member.shape.clone(),
                    slice: slice_of(member)?,
                });
                at += bytes;
            }
            let entry = Stored {
                view: RankView {
                    name,
                    dtype: view.dtype,
                    shape: packed_shape,
                    start: 0,
                    binding: view.binding,
                },
                offset,
                transform: WeightTransform::Copy,
                dtype: view.dtype,
                layout_id: "chained_rows_v1",
                slice: None,
                pieces,
            };
            offset = (offset + entry.bytes()).next_multiple_of(VIEW_ALIGNMENT);
            stored.push(entry);
            continue;
        }
        let slice = slice_of(&view)?;
        let (transform, dtype, layout_id) = if f32_side(&view.name) && view.dtype == DType::Bf16 {
            (
                WeightTransform::WidenBf16F32,
                DType::F32,
                "f32_row_major_v1",
            )
        } else {
            let layout_id = match (view.binding, view.dtype) {
                (Binding::Projection(_) | Binding::Embedding, _) => "checkpoint_rows_v1",
                (_, DType::F32) => "f32_row_major_v1",
                (_, DType::Bf16) => "bf16_row_major_v1",
                (_, DType::F8E4M3) => "fp8_e4m3_row_major_v1",
                (_, DType::U8) => return Err(format!("{}: no U8 side tensor layout", view.name)),
            };
            (WeightTransform::Copy, view.dtype, layout_id)
        };
        let pieces = vec![Piece {
            tensor: view.name.clone(),
            offset,
            shape: view.shape.clone(),
            slice: slice.clone(),
        }];
        // A lowered projection's weight view holds the rows its operation
        // reads: the rank's rows, then zero rows to the format's row tile.
        let mut view = view;
        if let Binding::Projection(id) = view.binding
            && view.name.ends_with(".weight")
            && let Some(rows) = view.shape.first_mut()
        {
            *rows = crate::precision::stored_rows(id, view.dtype, *rows)?;
        }
        let entry = Stored {
            view,
            offset,
            transform,
            dtype,
            layout_id,
            slice,
            pieces,
        };
        offset = (offset + entry.bytes()).next_multiple_of(VIEW_ALIGNMENT);
        stored.push(entry);
    }
    Ok(stored)
}

/// The checkpoint file of every tensor, from the index document.
fn weight_map(request: &WeightPlanRequestV1) -> Result<serde_json::Map<String, Value>, String> {
    let bytes = request
        .document(CHECKPOINT_INDEX)
        .ok_or("the checkpoint has no model.safetensors.index.json")?;
    let index = crate::admission::document(CHECKPOINT_INDEX, bytes, bytes.len())?;
    index["weight_map"]
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{CHECKPOINT_INDEX} has no weight_map"))
}

/// Checks every checkpoint tensor `layout` reads against the request's
/// safetensors headers before anything is planned: present in the file the
/// weight map names, with the admitted catalog's dtype, its complete shape
/// and data of exactly that many bytes. A header that disagrees (a
/// same-shape F16 tensor in place of BF16, say) is refused rather than
/// copied under the catalog's dtype.
fn check_headers(
    request: &WeightPlanRequestV1,
    shape: Shape,
    quantization: &Layout,
    layout: &[Stored],
    file: impl Fn(&str) -> Result<String, String>,
) -> Result<(), String> {
    let catalog: HashMap<String, TensorSpec> =
        crate::catalog::checkpoint_catalog(shape.geometry, quantization)
            .into_iter()
            .map(|spec| (spec.name.clone(), spec))
            .collect();
    let mut headers = Headers::default();
    for entry in layout {
        for piece in &entry.pieces {
            let tensor = piece.tensor.as_str();
            let spec = catalog
                .get(tensor)
                .ok_or_else(|| format!("{tensor} is not in the catalog"))?;
            let name = file(tensor)?;
            let header = headers.tensor(request, &name, tensor)?;
            let dtype = header["dtype"].as_str();
            let dims = header["shape"]
                .as_array()
                .map(|dims| dims.iter().map(Value::as_u64).collect::<Option<Vec<_>>>());
            let span = header["data_offsets"]
                .as_array()
                .and_then(|offsets| match offsets[..] {
                    [ref start, ref end] => end.as_u64()?.checked_sub(start.as_u64()?),
                    _ => None,
                });
            if dtype != Some(spec.dtype.name()) {
                return Err(format!(
                    "{tensor} is {} in {name}, the catalog {}",
                    dtype.unwrap_or("untyped"),
                    spec.dtype.name()
                ));
            }
            if dims.flatten().as_ref() != Some(&spec.shape) {
                return Err(format!(
                    "{tensor} has shape {} in {name}, the catalog {:?}",
                    header["shape"], spec.shape
                ));
            }
            if span != Some(spec.bytes()) {
                return Err(format!(
                    "{tensor} spans {} in {name}, the catalog {} bytes",
                    header["data_offsets"],
                    spec.bytes()
                ));
            }
        }
    }
    Ok(())
}

/// The safetensors headers of a request's files, each parsed once when a
/// tensor of its file is first asked for.
#[derive(Default)]
struct Headers(HashMap<String, serde_json::Map<String, Value>>);

impl Headers {
    /// The header entry of `tensor` in file `name` of `request`.
    fn tensor<'a>(
        &'a mut self,
        request: &WeightPlanRequestV1,
        name: &str,
        tensor: &str,
    ) -> Result<&'a Value, String> {
        if !self.0.contains_key(name) {
            let checkpoint = request
                .files
                .iter()
                .find(|candidate| candidate.name == name)
                .ok_or_else(|| format!("tensor {tensor}: file {name} is not in the checkpoint"))?;
            let parsed = crate::admission::document(
                &format!("{name} header"),
                &checkpoint.header,
                checkpoint.header.len(),
            )?;
            let Value::Object(parsed) = parsed else {
                return Err(format!("{name} header is not an object"));
            };
            self.0.insert(name.to_owned(), parsed);
        }
        self.0[name]
            .get(tensor)
            .ok_or_else(|| format!("{tensor} is not in the header of {name}"))
    }
}

/// The admitted layout of the checkpoint `request` describes, whose config
/// gave `matched`: its class encodings with the dtype its headers store for
/// each side tensor family (`quantization::resolve`). Preparation and the
/// load-time plan both resolve the layout here, so both plan the same one.
pub fn resolve_layout(
    request: &WeightPlanRequestV1,
    matched: &crate::admission::Matched,
) -> Result<&'static Layout, String> {
    let weight_map = weight_map(request)?;
    let mut headers = Headers::default();
    crate::quantization::resolve(matched.geometry, &matched.quantization, |tensor| {
        let name = mapped_file(&weight_map, tensor)?;
        headers.tensor(request, &name, tensor)?["dtype"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("{tensor} is untyped in {name}"))
    })
}

/// The records that store `entry`, one per piece; `file` names each
/// tensor's checkpoint file.
pub fn records(
    entry: &Stored,
    files: &RequestFiles,
    file: impl Fn(&str) -> Result<String, String>,
) -> Result<Vec<WeightRecordV1>, String> {
    entry
        .pieces
        .iter()
        .map(|piece| {
            let (axis, ranges) = match &piece.slice {
                Some((axis, range)) => (Some(*axis), vec![range.clone()]),
                None => (None, Vec::new()),
            };
            Ok(WeightRecordV1 {
                arena_offset: piece.offset,
                byte_length: piece.shape.iter().product::<u64>() * entry.dtype.bytes(),
                transform: entry.transform.id(),
                params: entry.transform.encode_params().map_err(|e| e.to_string())?,
                sources: vec![files.source(&piece.tensor, &file(&piece.tensor)?)?],
                axis,
                ranges,
                // The host gathers a scalar (an NVFP4 `weight_scale_2`) as
                // one element of shape [1].
                stored_shape: if piece.shape.is_empty() {
                    vec![1]
                } else {
                    piece.shape.clone()
                },
            })
        })
        .collect()
}

/// The rank index of `layout`, naming the plan's identity.
fn index(
    rank: u32,
    layout: &[Stored],
    image_bytes: u64,
    identity: &str,
) -> Result<Vec<u8>, String> {
    let views: Vec<Value> = layout
        .iter()
        .map(|entry| {
            // The index lists a scalar (an NVFP4 dense down projection's
            // weight_scale_2) as one element: its views need a shape.
            let shape = if entry.view.shape.is_empty() {
                vec![1]
            } else {
                entry.view.shape.clone()
            };
            json!({
                "name": entry.view.name,
                "offset": entry.offset,
                "bytes": entry.bytes(),
                "dtype": entry.dtype.name(),
                "shape": shape,
                "layout_id": entry.layout_id,
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

/// The file the weight map names for `tensor`.
fn mapped_file(
    weight_map: &serde_json::Map<String, Value>,
    tensor: &str,
) -> Result<String, String> {
    weight_map
        .get(tensor)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{tensor} is not in the weight map"))
}

/// Checks the text tensors the plan of every rank of `ranks` reads against
/// the request's headers ([`check_headers`]), as [`plan`] does at load.
/// Preparation runs it, so a checkpoint missing a text tensor, or holding
/// one of another dtype, shape or length, is refused before its package is
/// published.
pub fn check_text_headers(
    request: &WeightPlanRequestV1,
    matched: &crate::admission::Matched,
    ranks: u32,
) -> Result<(), String> {
    let shape = Shape::new(
        matched.geometry,
        bundle_runtime::geometry::Ranks::new(ranks).ok_or("rank count")?,
    );
    let quantization = resolve_layout(request, matched)?;
    let weight_map = weight_map(request)?;
    let file = |tensor: &str| mapped_file(&weight_map, tensor);
    for rank in 0..ranks {
        let layout = checkpoint_layout(shape, quantization, rank)?;
        check_headers(request, shape, quantization, &layout, file)?;
    }
    Ok(())
}

/// The plan of the text rank image for the request's rank, after the
/// checkpoint's config is admitted, in its quantization layout; `None` for
/// another image role.
pub fn plan(request: &WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String> {
    if request.image_role.as_bytes() == crate::encoder::views::IMAGE_ROLE {
        return crate::encoder_plan::plan(request).map(Some);
    }
    if !request.image_role.is_empty() && request.image_role != RANK_IMAGE_ROLE {
        return Ok(None);
    }
    let matched = crate::admission::admit_bytes(
        request
            .document("config.json")
            .ok_or("the checkpoint has no config.json")?,
    )?;
    crate::admission::check_ranks(request.ranks)?;
    let shape = Shape::new(
        matched.geometry,
        bundle_runtime::geometry::Ranks::new(request.ranks).ok_or("rank count")?,
    );
    let quantization = resolve_layout(request, &matched)?;
    let layout = checkpoint_layout(shape, quantization, request.rank)?;
    let image_bytes = layout
        .last()
        .map_or(0, |entry| entry.offset + entry.bytes())
        .next_multiple_of(VIEW_ALIGNMENT);
    let weight_map = weight_map(request)?;
    let files = RequestFiles::new(request);
    let file = |tensor: &str| mapped_file(&weight_map, tensor);
    check_headers(request, shape, quantization, &layout, file)?;
    let mut records = Vec::new();
    for entry in &layout {
        records.extend(self::records(entry, &files, file)?);
    }
    let mut plan = WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: RANK_IMAGE_ROLE.into(),
        alignment: ARENA_ALIGNMENT,
        arena_bytes: image_bytes,
        views: Vec::new(),
        components: Vec::new(),
        records,
    };
    let identity = plan_identity(&plan, request);
    plan.components = vec![NamedBytesV1 {
        name: index_role(request.rank),
        bytes: index(request.rank, &layout, image_bytes, &identity)?,
    }];
    Ok(Some(plan))
}

/// Plan requests for tests: the pinned config and one checkpoint file
/// holding every catalog tensor.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use runtime_bundle_abi::weight_plan::CheckpointFileV1;

    use super::*;
    use crate::catalog::checkpoint_catalog;

    const FILE: &str = "model.safetensors";

    /// A request for `rank` of 4 of the pinned block-FP8 checkpoint whose
    /// one file holds every catalog tensor back to back, each as its spec
    /// declares, in the header after `edit`.
    pub fn request(
        rank: u32,
        edit: impl FnOnce(&mut serde_json::Map<String, Value>),
    ) -> WeightPlanRequestV1 {
        checkpoint_request(
            &FP8_BLOCK128,
            include_bytes!("../fixtures/config.json"),
            rank,
            edit,
        )
    }

    /// A request like [`request`] for a checkpoint with `config` stored in
    /// `layout`.
    pub fn checkpoint_request(
        layout: &Layout,
        config: &[u8],
        rank: u32,
        edit: impl FnOnce(&mut serde_json::Map<String, Value>),
    ) -> WeightPlanRequestV1 {
        let tensors = checkpoint_catalog(&crate::geometry::FLASH, layout);
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        for spec in &tensors {
            header.insert(
                spec.name.clone(),
                json!({
                    "dtype": spec.dtype.name(),
                    "shape": spec.shape,
                    "data_offsets": [offset, offset + spec.bytes()],
                }),
            );
            offset += spec.bytes();
        }
        edit(&mut header);
        let header = serde_json::to_vec(&header).expect("header JSON");
        let weight_map: serde_json::Map<String, Value> = tensors
            .iter()
            .map(|spec| (spec.name.clone(), Value::from(FILE)))
            .collect();
        WeightPlanRequestV1 {
            ranks: 4,
            rank,
            files: vec![CheckpointFileV1 {
                name: FILE.into(),
                byte_length: 8 + header.len() as u64 + offset,
                data_offset: 8 + header.len() as u64,
                header,
            }],
            documents: vec![
                NamedBytesV1 {
                    name: "config.json".into(),
                    bytes: config.to_vec(),
                },
                NamedBytesV1 {
                    name: CHECKPOINT_INDEX.into(),
                    bytes: serde_json::to_vec(&json!({ "weight_map": weight_map }))
                        .expect("index JSON"),
                },
            ],
            ..Default::default()
        }
    }
}

/// The engine of a side view, for tests and the bundle's checks.
pub fn engine(entry: &Stored) -> Option<Engine> {
    match entry.view.binding {
        Binding::Side(engine) => Some(engine),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::catalog;

    fn find<'a>(layout: &'a [Stored], name: &str) -> &'a Stored {
        let full = format!("model.language_model.{name}");
        layout.iter().find(|entry| entry.view.name == full).unwrap()
    }

    #[test]
    fn kda_convolutions_widen_to_f32() {
        let layout = layout(Shape::TP4, 1).unwrap();
        let conv = find(&layout, "layers.0.self_attn.q_conv1d.weight");
        let spec = catalog(&crate::geometry::FLASH)
            .into_iter()
            .find(|spec| spec.name == conv.view.name)
            .unwrap();
        // The checkpoint holds BF16 [8192, 1, 4]; rank 1 holds channels
        // 2048..4096 as F32.
        assert_eq!(
            (spec.dtype, spec.shape.clone()),
            (DType::Bf16, vec![8192, 1, 4])
        );
        assert_eq!(conv.transform, WeightTransform::WidenBf16F32);
        assert_eq!(conv.view.shape, [2048, 1, 4]);
        assert_eq!(conv.bytes(), 2048 * 4 * 4);
        assert_eq!(conv.slice, Some((0, 2048..4096)));
        assert_eq!(engine(conv), Some(Engine::Kda));
    }

    #[test]
    fn kv_b_proj_holds_the_rank_heads_key_then_value_halves() {
        let shape = Shape::TP4;
        let p = shape.profile;
        let layout = layout(shape, 2).unwrap();
        let kv_b = find(&layout, "layers.3.self_attn.kv_b_proj.weight");
        let spec = catalog(&crate::geometry::FLASH)
            .into_iter()
            .find(|spec| spec.name == kv_b.view.name)
            .unwrap();
        // The checkpoint's rows are head-major, each head's 256 key rows
        // then its 256 value rows over the 512-wide latent, so the rank's
        // rows are its 16 heads in the same order.
        let per_head = u64::from(p.qk_nope + p.v_head);
        assert_eq!(spec.shape, [64 * per_head, u64::from(p.kv_rank)]);
        assert_eq!(kv_b.view.shape, [16 * per_head, 512]);
        assert_eq!(kv_b.slice, Some((0, 2 * 16 * per_head..3 * 16 * per_head)));
        assert_eq!(kv_b.transform, WeightTransform::Copy);
        assert_eq!(kv_b.bytes(), 16 * per_head * 512 * 2);
        assert_eq!(kv_b.layout_id, "bf16_row_major_v1");
    }

    #[test]
    fn views_are_aligned_and_disjoint() {
        let layout = layout(Shape::TP4, 0).unwrap();
        for pair in layout.windows(2) {
            assert_eq!(pair[0].offset % VIEW_ALIGNMENT, 0);
            assert!(pair[0].offset + pair[0].bytes() <= pair[1].offset);
        }
    }

    /// A request naming the pinned config and one checkpoint file holding
    /// every tensor: the plan stores every text view once, and its index
    /// parses with the shared parser.
    #[test]
    fn the_plan_places_every_text_view_and_indexes_it() {
        let request = test_support::request(3, |_| {});
        let planned = plan(&request).unwrap().unwrap();
        let layout = layout(Shape::TP4, 3).unwrap();
        let pieces: usize = layout.iter().map(|entry| entry.pieces.len()).sum();
        assert_eq!(planned.records.len(), pieces);
        assert_eq!(planned.arena_bytes % VIEW_ALIGNMENT, 0);
        let widened = planned
            .records
            .iter()
            .filter(|record| record.transform == kernel_core::weight_transform::WIDEN_BF16_F32)
            .count();
        // q, k and v of each of the 34 KDA layers.
        assert_eq!(widened, 3 * 34);
        let [component] = &planned.components[..] else {
            panic!("one rank index")
        };
        assert_eq!(component.name, index_role(3));
        let index = bundle_runtime::index::parse(&component.bytes, 3, INDEX_SCHEMA).unwrap();
        assert_eq!(index.views.len(), layout.len());
        assert_eq!(index.image_bytes, planned.arena_bytes);
        let mut other = request.clone();
        other.image_role = "glm5next.vision-images.v1".into();
        assert_eq!(plan(&other), Ok(None));
    }

    /// A header that disagrees with the catalog is refused before
    /// anything is planned: a same-shape F16 tensor in place of BF16, a
    /// transposed shape, a short data span, a missing tensor.
    #[test]
    fn the_plan_refuses_headers_that_differ_from_the_catalog() {
        const KV_B: &str = "model.language_model.layers.3.self_attn.kv_b_proj.weight";
        let refusal = |edit: fn(&mut Value)| {
            let request = test_support::request(0, |header| edit(&mut header[KV_B]));
            plan(&request).unwrap_err()
        };
        let error = refusal(|tensor| tensor["dtype"] = json!("F16"));
        assert!(
            error.contains("is F16 in model.safetensors, the catalog BF16"),
            "{error}"
        );
        let error = refusal(|tensor| tensor["shape"] = json!([512, 32768]));
        assert!(error.contains("has shape [512,32768]"), "{error}");
        let error = refusal(|tensor| tensor["shape"] = json!([32768]));
        assert!(error.contains("has shape [32768]"), "{error}");
        let error = refusal(|tensor| {
            let end = tensor["data_offsets"][1].as_u64().unwrap();
            tensor["data_offsets"][1] = json!(end - 2);
        });
        assert!(error.contains("spans"), "{error}");
        let request = test_support::request(0, |header| {
            header.remove(KV_B);
        });
        let error = plan(&request).unwrap_err();
        assert!(
            error.contains("is not in the header of model.safetensors"),
            "{error}"
        );
    }

    /// Preparation's check of every rank's text headers refuses what the
    /// plan of any rank would: a missing tensor, another dtype.
    #[test]
    fn text_headers_are_checked_for_every_rank() {
        const KV_B: &str = "model.language_model.layers.3.self_attn.kv_b_proj.weight";
        let matched =
            crate::admission::admit_bytes(include_bytes!("../fixtures/config.json")).unwrap();
        check_text_headers(&test_support::request(0, |_| {}), &matched, 4).unwrap();
        let request = test_support::request(0, |header| {
            header.remove(KV_B);
        });
        let error = check_text_headers(&request, &matched, 4).unwrap_err();
        assert!(error.contains("is not in the header"), "{error}");
        // An expert every rank reads a slice of.
        const EXPERT: &str = "model.language_model.layers.5.mlp.experts.255.down_proj.weight";
        let request = test_support::request(0, |header| header[EXPERT]["dtype"] = json!("BF16"));
        let error = check_text_headers(&request, &matched, 4).unwrap_err();
        assert!(error.contains(EXPERT), "{error}");
    }

    /// A header naming a tensor twice, or a config repeating a field, is
    /// ambiguous and refused, whichever entry comes last.
    #[test]
    fn the_plan_refuses_repeated_header_entries_and_config_fields() {
        const KV_B: &str = "model.language_model.layers.3.self_attn.kv_b_proj.weight";
        let mut request = test_support::request(0, |_| {});
        let file = &mut request.files[0];
        let header: Value = serde_json::from_slice(&file.header).unwrap();
        let entry = format!(r#"{{"{KV_B}":{},"#, header[KV_B]);
        let text = String::from_utf8(file.header.clone()).unwrap();
        file.header = text.replacen('{', &entry, 1).into_bytes();
        file.data_offset = 8 + file.header.len() as u64;
        let error = plan(&request).unwrap_err();
        assert!(
            error.contains(&format!("duplicate metadata field {KV_B}")),
            "{error}"
        );
        let mut request = test_support::request(0, |_| {});
        let config = request
            .documents
            .iter_mut()
            .find(|document| document.name == "config.json")
            .unwrap();
        let text = String::from_utf8(config.bytes.clone()).unwrap();
        config.bytes = text
            .replacen('{', r#"{"architectures": ["Other"],"#, 1)
            .into_bytes();
        let error = plan(&request).unwrap_err();
        assert!(
            error.contains("duplicate metadata field architectures"),
            "{error}"
        );
    }

    /// The chained engine reads each expert's gate rows then up rows of the
    /// rank's slice, every expert back to back, and the dense halves
    /// concatenated.
    #[test]
    fn chained_weights_pack_gate_then_up_per_expert() {
        let layout = layout(Shape::TP4, 1).unwrap();
        let experts = find(&layout, "layers.5.mlp.experts.gate_up.weight");
        assert_eq!(experts.view.shape, [288, 1024, 4096]);
        assert_eq!(experts.layout_id, "chained_rows_v1");
        let names: Vec<_> = experts.pieces[..3]
            .iter()
            .map(|p| p.tensor.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "model.language_model.layers.5.mlp.experts.0.gate_proj.weight",
                "model.language_model.layers.5.mlp.experts.0.up_proj.weight",
                "model.language_model.layers.5.mlp.experts.1.gate_proj.weight",
            ]
        );
        assert_eq!(
            experts.pieces[1].offset - experts.pieces[0].offset,
            512 * 4096
        );
        assert_eq!(experts.pieces.len(), 2 * 288);
        // Rank 1's rows 512..1024 of each expert's gate and up projections.
        assert!(
            experts
                .pieces
                .iter()
                .all(|p| p.slice == Some((0, 512..1024)) && p.shape == [512, 4096])
        );
        let scales = find(&layout, "layers.5.mlp.experts.gate_up.weight_scale_inv");
        assert_eq!(
            (scales.view.shape.clone(), scales.dtype),
            (vec![288, 8, 32], DType::F32)
        );
        assert_eq!(scales.pieces[0].slice, Some((0, 4..8)));
        // The down projections' columns 512..1024.
        let down = find(&layout, "layers.5.mlp.experts.down.weight");
        assert_eq!(down.view.shape, [288, 4096, 512]);
        assert_eq!(down.pieces[0].slice, Some((1, 512..1024)));
        let down_scales = find(&layout, "layers.5.mlp.experts.down.weight_scale_inv");
        assert_eq!(down_scales.view.shape, [288, 32, 4]);
        let dense = find(&layout, "layers.0.mlp.gate_up.weight");
        assert_eq!(dense.view.shape, [6144, 4096]);
        assert_eq!(
            dense
                .pieces
                .iter()
                .map(|p| p.slice.clone())
                .collect::<Vec<_>>(),
            [Some((0, 3072..6144)), Some((0, 3072..6144))]
        );
        let shared = find(
            &layout,
            "layers.5.mlp.shared_experts.gate_up.weight_scale_inv",
        );
        assert_eq!(shared.view.shape, [8, 32]);
        // The MTP layer's experts pack the same way; no member stays a view.
        find(&layout, "layers.45.mlp.experts.down.weight_scale_inv");
        assert!(!layout.iter().any(|e| e.view.name.contains(".gate_proj.")));
    }
}
