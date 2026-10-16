//! Qwen3.8's weight plan for direct checkpoint loading.
//!
//! The plan is the compiled rank image's own: the same catalog, placement and
//! prepared-precision templates the packager runs (`crate::placement`), read
//! from the checkpoint headers the host supplies. Each piece becomes one
//! record at the offset the packager writes it to, so the arena, the view
//! names and every byte equal the rank image's. The rank index the bundle
//! reads is the packager's runtime index for that plan.
//!
//! The vision encoder image (`encoder::views`) is planned the same way: each
//! view is its checkpoint tensor copied whole and zero-padded to the stored
//! shape, and every rank's image is the same.

use crate::catalog::{Catalog, TensorRecord};
use crate::contract::{NGRAM_HOST_VIEW, RANK_IMAGE_ROLE, index_role};
use crate::encoder::views as encoder_views;
use crate::package::RANK_IMAGE_ALIGNMENT;
use crate::placement::{ImagePlan, Piece, fragment, layout, view_dtype};
use crate::precision_packing::PackedPiece;
use kernel_core::weight_transform::WeightTransform;
use model_compiler_sdk::weight_plan::{RequestCache, RequestFiles, RequestKey, plan_identity};
use runtime_bundle_abi::weight_plan::{
    NamedBytesV1, PlannedViewV1, ViewPlacementV1, WeightPlanRequestV1, WeightPlanV1, WeightRecordV1,
};
use runtime_weights::DType;
use std::collections::BTreeMap;
use std::sync::Arc;

const INDEX: &str = "model.safetensors.index.json";
const RUNTIME_INDEX_BYTES: usize = 1 << 20;

/// The catalog and image plan of one checkpoint at one rank count.
pub struct Planned {
    pub catalog: Arc<Catalog>,
    pub plan: ImagePlan,
    /// Whether the n-gram table is in host memory.
    pub host_ngram: bool,
}

/// The last request's planning, keyed by everything but its rank: the host
/// asks once per rank for the same checkpoint.
static PLANNED: RequestCache<Planned> = RequestCache::new();

/// What a catalog is read from: the rank count, the checkpoint and the
/// documents the catalog reads.
fn source_key(request: &WeightPlanRequestV1, purpose: &[u8]) -> RequestKey {
    let mut key = RequestKey::new(purpose, request);
    key.bytes(&request.ranks.to_le_bytes());
    for name in ["config.json", INDEX, "hf_quant_config.json"] {
        key.optional(request.document(name));
    }
    key
}

/// Everything a rank's text plan reads but its rank: the catalog's sources,
/// the package components (the precision component) and the host views.
fn key(request: &WeightPlanRequestV1) -> [u8; 32] {
    let mut key = source_key(request, b"planned");
    for component in &request.components {
        key.bytes(component.name.as_bytes());
        key.bytes(&component.bytes);
    }
    for view in &request.host_views {
        key.bytes(view.as_bytes());
    }
    key.finish()
}

/// The last request's catalog, keyed by what it is read from: the host asks
/// once per image and rank for the same checkpoint and rank count.
static CATALOG: RequestCache<Catalog> = RequestCache::new();

/// The catalog of the requested checkpoint at the requested rank count, from
/// the headers and documents the host supplies.
fn catalog(request: &WeightPlanRequestV1) -> Result<Arc<Catalog>, String> {
    CATALOG.get_or_build(source_key(request, b"catalog").finish(), || {
        let ranks = request.ranks as usize;
        crate::placement::bundle_ranks(ranks).map_err(|e| e.to_string())?;
        let mut inputs = BTreeMap::new();
        for name in ["config.json", INDEX, "hf_quant_config.json"] {
            if let Some(bytes) = request.document(name) {
                inputs.insert(name.to_owned(), bytes.to_vec());
            }
        }
        let headers = request
            .files
            .iter()
            .map(|file| (file.name.clone(), (file.byte_length, file.header.clone())))
            .collect();
        let files = model_compiler_sdk::source::SourceFiles::from_memory(inputs, &headers, INDEX)
            .map_err(|e| e.to_string())?;
        Catalog::from_captured(&files, ranks).map_err(|e| e.to_string())
    })
}

/// Whether the plan places the n-gram table in host memory: where the
/// capacity rule does (`package::ngram_in_host_memory`), or when the package
/// lists the host view.
pub fn host_ngram(request: &WeightPlanRequestV1) -> Result<bool, String> {
    let shape =
        crate::placement::bundle_shape_for(catalog(request)?.geometry, request.ranks as usize)
            .map_err(|e| e.to_string())?;
    let view = std::str::from_utf8(NGRAM_HOST_VIEW).expect("ASCII view identifier");
    Ok(crate::package::ngram_in_host_memory(shape)
        || request.host_views.iter().any(|listed| listed == view))
}

/// Plans every rank of the requested checkpoint, as the packager would.
pub fn planned(request: &WeightPlanRequestV1) -> Result<Arc<Planned>, String> {
    PLANNED.get_or_build(key(request), || {
        let catalog = catalog(request)?;
        let plan = match request.component(crate::prepared::COMPONENT_ROLE) {
            Some(bytes) => {
                let document =
                    kernel_core::precision::component::decode(bytes).map_err(|e| e.to_string())?;
                ImagePlan::with_precision(&catalog, None, &document)
            }
            None => ImagePlan::new(&catalog, None),
        }
        .map_err(|e| e.to_string())?;
        Ok(Planned {
            catalog,
            plan,
            host_ngram: host_ngram(request)?,
        })
    })
}

/// The bundle's `chop_bundle_weight_plan_v1`: the text rank images (also for
/// an empty role) and the vision encoder images; other roles are declined.
pub fn plan(request: &WeightPlanRequestV1) -> Result<Option<WeightPlanV1>, String> {
    if request.image_role.as_bytes() == encoder_views::IMAGE_ROLE {
        return encoder_plan(request, &*catalog(request)?).map(Some);
    }
    if !request.image_role.is_empty() && request.image_role.as_bytes() != RANK_IMAGE_ROLE {
        return Ok(None);
    }
    let planned = planned(request)?;
    rank_plan(request, &planned).map(Some)
}

/// One rank's plan from a planned checkpoint.
pub fn rank_plan(request: &WeightPlanRequestV1, planned: &Planned) -> Result<WeightPlanV1, String> {
    let rank = request.rank as usize;
    let index = planned
        .plan
        .ranks
        .get(rank)
        .ok_or("rank outside the plan")?;
    let files = RequestFiles::new(request);
    let mut records = Vec::new();
    for (name, pieces) in &planned.plan.sources {
        let tensor = &planned.catalog.tensors[name];
        for piece in pieces.iter().filter(|piece| piece.rank == rank) {
            let layout_id = &index.views[piece.view].layout_id;
            records.push(record(&planned.catalog, &files, tensor, piece, layout_id)?);
        }
    }
    records.sort_by_key(|record| record.arena_offset);
    let mut views = Vec::new();
    if planned.host_ngram {
        let shape =
            crate::placement::bundle_shape_for(planned.catalog.geometry, planned.catalog.ranks)
                .map_err(|e| e.to_string())?;
        let table = index
            .views
            .iter()
            .find(|view| view.name == crate::views::ngram_view_for(shape))
            .ok_or("the plan has no n-gram table")?;
        views.push(PlannedViewV1 {
            id: std::str::from_utf8(NGRAM_HOST_VIEW)
                .expect("ASCII view identifier")
                .into(),
            alignment: RANK_IMAGE_ALIGNMENT,
            placement: ViewPlacementV1::Host,
            offset: table.offset,
            byte_length: table.bytes,
        });
    }
    let mut plan = WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: std::str::from_utf8(RANK_IMAGE_ROLE)
            .expect("ASCII role")
            .into(),
        alignment: RANK_IMAGE_ALIGNMENT,
        arena_bytes: index.image_bytes,
        views,
        components: vec![],
        records,
    };
    // What the plan loads, not the bytes of any image: the host serves the
    // same identity for the planned rank.
    let identity = plan_identity(&plan, request);
    plan.components = vec![NamedBytesV1 {
        name: index_role(request.rank),
        bytes: runtime_index(index, &identity)?,
    }];
    Ok(plan)
}

/// One rank's encoder image plan: each view of `encoder::views::layout`, its
/// BF16 checkpoint tensor copied whole at the view's offset, zero-padded
/// along the one axis where the stored shape is wider (the MLP to
/// `MLP_PADDED`) and as it is where the stored shape only flattens it (the
/// patch convolution). The plan and its index are the same on every rank but
/// for the rank and its identity.
pub fn encoder_plan(
    request: &WeightPlanRequestV1,
    catalog: &Catalog,
) -> Result<WeightPlanV1, String> {
    if let Some(source) = &catalog.precision {
        source.require_bf16_vision().map_err(|e| e.to_string())?;
    }
    let files = RequestFiles::new(request);
    let layout = encoder_views::layout_for(catalog.geometry);
    let mut records = Vec::with_capacity(layout.len());
    for view in &layout {
        let tensor = catalog.tensors.get(&view.name).ok_or_else(|| {
            format!(
                "the checkpoint has no vision encoder tensor {}; its images need a package \
                 with encoder images",
                view.name
            )
        })?;
        let elements = view.source_shape.iter().product::<u64>();
        if tensor.dtype != DType::Bf16
            || tensor.shape != view.source_shape
            || tensor.bytes != 2 * elements
        {
            return Err(format!(
                "vision encoder tensor {} is not BF16 of shape {:?}",
                view.name, view.source_shape
            ));
        }
        // A flattened view stores the source's bytes as they are; the
        // gather's stored shape keeps the source's rank.
        let stored_shape = if view.shape.iter().product::<u64>() == elements {
            view.source_shape.clone()
        } else {
            view.shape.clone()
        };
        records.push(WeightRecordV1 {
            arena_offset: view.offset,
            byte_length: view.bytes(),
            transform: WeightTransform::Copy.id(),
            params: WeightTransform::Copy
                .encode_params()
                .map_err(|e| e.to_string())?,
            sources: vec![files.source(&tensor.name, &tensor.file)?],
            axis: None,
            ranges: vec![],
            stored_shape,
        });
    }
    let image_bytes = encoder_views::image_bytes_for(catalog.geometry);
    let mut plan = WeightPlanV1 {
        ranks: request.ranks,
        rank: request.rank,
        image_role: std::str::from_utf8(encoder_views::IMAGE_ROLE)
            .expect("ASCII role")
            .into(),
        alignment: RANK_IMAGE_ALIGNMENT,
        arena_bytes: image_bytes,
        views: vec![],
        components: vec![],
        records,
    };
    let identity = plan_identity(&plan, request);
    // The encoder runtime index, with the plan's identity in place of an
    // image digest.
    let index = bundle_runtime::index::encoder::index(
        encoder_views::SCHEMA,
        request.rank,
        image_bytes,
        &layout,
        bundle_runtime::index::encoder::Identity::Direct(&identity),
    );
    let bytes = serde_json::to_vec(&index).map_err(|e| e.to_string())?;
    // The bundle's own admission of the planned index.
    encoder_views::parse_for(&bytes, request.rank, catalog.geometry)?;
    plan.components = vec![NamedBytesV1 {
        name: encoder_views::index_role(request.rank),
        bytes,
    }];
    Ok(plan)
}

/// The rank index the bundle reads for a planned rank: the packager's
/// runtime index, with the plan's identity in place of an image digest. A
/// planned rank has no image file, so it carries no `sha256`, and consumers
/// that need an image digest refuse it.
fn runtime_index(index: &crate::placement::RankIndex, identity: &str) -> Result<Vec<u8>, String> {
    let mut runtime = serde_json::to_value(index).map_err(|e| e.to_string())?;
    let fields = runtime
        .as_object_mut()
        .ok_or("rank index is not an object")?;
    fields.remove("sha256");
    fields.insert("direct_identity".into(), identity.into());
    let bytes = serde_json::to_vec(&runtime).map_err(|e| e.to_string())?;
    model_compiler_sdk::rank_index::compact(&bytes, RUNTIME_INDEX_BYTES).map_err(|e| e.to_string())
}

/// One piece as a record: the bytes its transform (a fragment, widening or
/// prepared packing) produces at `piece.offset`.
pub fn record(
    catalog: &Catalog,
    files: &RequestFiles,
    tensor: &TensorRecord,
    piece: &Piece,
    layout_id: &str,
) -> Result<WeightRecordV1, String> {
    if let Some(packing) = &piece.packing {
        return packed(catalog, files, piece, packing);
    }
    let source = fragment_record(tensor, piece, layout_id)?;
    Ok(WeightRecordV1 {
        arena_offset: piece.offset,
        byte_length: piece.bytes,
        transform: source.transform.id(),
        params: source
            .transform
            .encode_params()
            .map_err(|e| e.to_string())?,
        sources: vec![files.source(&tensor.name, &tensor.file)?],
        axis: source.axis,
        ranges: source.ranges,
        stored_shape: source.stored_shape,
    })
}

/// The unchanged fragment source selection used by direct loading and variants.
pub(crate) fn fragment_record(
    tensor: &TensorRecord,
    piece: &Piece,
    layout_id: &str,
) -> Result<kernel_core::weight_transform::WeightRecord, String> {
    use kernel_core::weight_transform::{SourceTensor, WeightRecord};
    let axis = piece.slices.first().ok_or("piece has no slices")?.0;
    if piece.slices.iter().any(|(other, _)| *other != axis) {
        return Err(format!("tensor {}: slices along several axes", tensor.name));
    }
    let transform = match fragment(layout_id, &piece.shape) {
        Some(layout) => WeightTransform::Fragment(layout.format),
        None if view_dtype(tensor) != tensor.dtype => WeightTransform::WidenBf16F32,
        None => WeightTransform::Copy,
    };
    debug_assert_eq!(layout(tensor), layout_id);
    Ok(WeightRecord {
        transform,
        sources: vec![SourceTensor {
            dtype: tensor.dtype.safetensors_name().into(),
            shape: tensor.shape.clone(),
            byte_length: tensor.bytes,
        }],
        axis: Some(axis as u32),
        ranges: piece.slices.iter().map(|(_, r)| r.clone()).collect(),
        stored_shape: piece.shape.clone(),
    })
}

/// A prepared precision piece: one matrix, its codes and scale companions.
fn packed(
    catalog: &Catalog,
    files: &RequestFiles,
    piece: &Piece,
    packing: &PackedPiece,
) -> Result<WeightRecordV1, String> {
    let operand = catalog
        .precision
        .as_ref()
        .and_then(|source| source.operands.get(&packing.operand))
        .ok_or_else(|| format!("tensor {}: no imported operand", packing.operand))?;
    let weight = &operand.weight;
    if weight.logical_shape.len() != 2 {
        return Err(format!(
            "tensor {}: packed pieces are matrices",
            packing.operand
        ));
    }
    let mut sources = Vec::new();
    let mut add = |name: &str| -> Result<u32, String> {
        let file = &catalog
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor {name}: not in the checkpoint"))?
            .file;
        sources.push(files.source(name, file)?);
        Ok(sources.len() as u32 - 1)
    };
    add(&weight.codes.tensor)?;
    weight
        .local_scale
        .as_ref()
        .map(|scale| add(&scale.tensor.tensor))
        .transpose()?;
    weight
        .global_scale
        .as_ref()
        .map(|scale| add(&scale.tensor.tensor))
        .transpose()?;
    for span in [Some(&weight.codes)]
        .into_iter()
        .chain([
            weight.local_scale.as_ref().map(|s| &s.tensor),
            weight.global_scale.as_ref().map(|s| &s.tensor),
        ])
        .flatten()
    {
        let tensor = &catalog.tensors[&span.tensor];
        if span.offset != 0 || span.byte_length != tensor.bytes {
            return Err(format!(
                "tensor {}: precision span differs from the whole tensor",
                span.tensor
            ));
        }
    }
    let portable = packing
        .record(weight, &piece.slices)
        .map_err(|e| e.to_string())?;
    Ok(WeightRecordV1 {
        arena_offset: piece.offset,
        byte_length: piece.bytes,
        transform: portable.transform.id(),
        params: portable
            .transform
            .encode_params()
            .map_err(|e| e.to_string())?,
        sources,
        axis: portable.axis,
        ranges: portable.ranges,
        stored_shape: portable.stored_shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::{RankIndex, ViewIndex};
    use sha2::{Digest, Sha256};

    fn index() -> RankIndex {
        RankIndex {
            schema: crate::views::SCHEMA.into(),
            rank: 1,
            image: "rank1.img".into(),
            image_bytes: 8192,
            sha256: String::new(),
            views: vec![ViewIndex {
                name: "model.language_model.norm.weight".into(),
                offset: 4096,
                bytes: 4096,
                shape: vec![2048],
                dtype: "BF16".into(),
                layout_id: crate::views::BF16_ROWS.into(),
                source_tensor_digests: BTreeMap::new(),
            }],
        }
    }

    #[test]
    fn host_ngram_lookup_uses_the_planned_ple_layer() {
        let mut geometry = crate::geometry::FLASH;
        geometry.shape.ple.layer_ids = &[3];
        let geometry = Box::leak(Box::new(geometry));
        let mut catalog = crate::catalog::synthetic(4);
        catalog.geometry = geometry;
        let mut index = index();
        index.views[0].name = crate::views::ngram_view_for(crate::contract::Shape::new(
            geometry,
            crate::contract::Ranks::TP4,
        ));
        assert_ne!(index.views[0].name, crate::views::ngram_view_flash());
        let planned = Planned {
            catalog: Arc::new(catalog),
            plan: ImagePlan {
                ranks: vec![index.clone(), index],
                sources: BTreeMap::new(),
                layers: None,
            },
            host_ngram: true,
        };
        let (_, mut request) = vision_catalog();
        request.rank = 1;
        request.image_role = std::str::from_utf8(RANK_IMAGE_ROLE).unwrap().into();
        let plan = rank_plan(&request, &planned).unwrap();
        assert_eq!(plan.views.len(), 1);
        assert_eq!(plan.views[0].placement, ViewPlacementV1::Host);
        assert_eq!(plan.views[0].offset, 4096);
        assert_eq!(plan.views[0].byte_length, 4096);
    }

    /// Every vision tensor of the encoder layout, BF16 of its source shape,
    /// the blocks in one file and the rest in another.
    fn vision_catalog() -> (Catalog, WeightPlanRequestV1) {
        let files = ["model-00001.safetensors", "model-00002.safetensors"];
        let tensors = encoder_views::layout_flash()
            .into_iter()
            .map(|view| {
                let elements = view.source_shape.iter().product::<u64>();
                let record = TensorRecord {
                    file: files[usize::from(view.name.contains(".blocks."))].into(),
                    name: view.name.clone(),
                    dtype: DType::Bf16,
                    shape: view.source_shape,
                    file_start: 0,
                    bytes: 2 * elements,
                    placement: crate::catalog::Placement::Replicated,
                    scale: crate::catalog::ScaleLayout::None,
                };
                (view.name, record)
            })
            .collect();
        let catalog = Catalog {
            geometry: &model_qwen38_contract::geometry::FLASH,
            tensors,
            ranks: 4,
            precision: None,
        };
        let request = WeightPlanRequestV1 {
            image_role: std::str::from_utf8(encoder_views::IMAGE_ROLE)
                .unwrap()
                .into(),
            ranks: 4,
            rank: 0,
            checkpoint_identity: [7; 32],
            implementation: Default::default(),
            files: files
                .iter()
                .map(|name| runtime_bundle_abi::weight_plan::CheckpointFileV1 {
                    name: (*name).into(),
                    byte_length: 0,
                    data_offset: 8,
                    header: b"{}".to_vec(),
                })
                .collect(),
            documents: vec![],
            components: vec![],
            host_views: vec![],
        };
        (catalog, request)
    }

    /// Each view is one whole-tensor copy at the view's offset, padded to
    /// the stored MLP width, and every rank plans the same image.
    #[test]
    fn encoder_plans_copy_each_tensor_whole_at_its_view() {
        let (catalog, mut request) = vision_catalog();
        let layout = encoder_views::layout_flash();
        let mut records = None;
        let mut identities = std::collections::BTreeSet::new();
        for rank in 0..4 {
            request.rank = rank;
            let plan = encoder_plan(&request, &catalog).unwrap();
            assert_eq!(plan.image_role.as_bytes(), encoder_views::IMAGE_ROLE);
            assert_eq!((plan.ranks, plan.rank), (4, rank));
            assert_eq!(plan.arena_bytes, encoder_views::image_bytes_flash());
            assert_eq!(plan.alignment, RANK_IMAGE_ALIGNMENT);
            assert!(plan.views.is_empty());
            assert_eq!(plan.records.len(), layout.len());
            for (record, view) in plan.records.iter().zip(&layout) {
                assert_eq!(record.transform, kernel_core::weight_transform::COPY);
                assert!(record.params.is_empty());
                assert_eq!(
                    (record.arena_offset, record.byte_length),
                    (view.offset, view.bytes())
                );
                assert_eq!((record.axis, record.ranges.len()), (None, 0));
                assert_eq!(record.sources.len(), 1);
                assert_eq!(record.sources[0].tensor, view.name);
                assert_eq!(
                    record.sources[0].file,
                    u32::from(view.name.contains(".blocks."))
                );
            }
            let stored = |name: &str| {
                let at = layout.iter().position(|v| v.name == name).unwrap();
                plan.records[at].stored_shape.clone()
            };
            let block = "model.visual.blocks.26.mlp";
            assert_eq!(stored(&format!("{block}.linear_fc1.weight")), [4352, 1152]);
            assert_eq!(stored(&format!("{block}.linear_fc1.bias")), [4352]);
            assert_eq!(stored(&format!("{block}.linear_fc2.weight")), [1152, 4352]);
            // The flattened patch convolution keeps its checkpoint shape.
            assert_eq!(
                stored("model.visual.patch_embed.proj.weight"),
                [1152, 3, 2, 16, 16]
            );
            assert_eq!(
                stored("model.visual.merger.linear_fc2.weight"),
                [2560, 4608]
            );
            // The records are the same on every rank.
            match &records {
                None => records = Some(plan.records.clone()),
                Some(first) => assert_eq!(*first, plan.records),
            }
            let [index] = &plan.components[..] else {
                panic!("one index component");
            };
            assert_eq!(index.name, encoder_views::index_role(rank));
            let parsed = encoder_views::parse_flash(&index.bytes, rank).unwrap();
            assert_eq!(parsed.sha256, None);
            let mut unindexed = plan.clone();
            unindexed.components.clear();
            let identity = format!(
                "{:x}",
                Sha256::digest(unindexed.identity_document(&[7; 32], &Default::default()))
            );
            assert_eq!(parsed.direct_identity.as_deref(), Some(identity.as_str()));
            identities.insert(identity);
        }
        assert_eq!(identities.len(), 4);
    }

    /// A checkpoint without the vision tower, or with a tensor of another
    /// type or shape, is refused rather than planned.
    #[test]
    fn encoder_plans_refuse_a_missing_or_different_vision_tensor() {
        let (catalog, request) = vision_catalog();
        let name = "model.visual.blocks.3.mlp.linear_fc1.weight";
        let mut missing = Catalog {
            geometry: &model_qwen38_contract::geometry::FLASH,
            tensors: catalog.tensors.clone(),
            ranks: 4,
            precision: None,
        };
        missing.tensors.remove(name);
        let error = encoder_plan(&request, &missing).unwrap_err();
        assert!(error.contains("no vision encoder tensor") && error.contains(name));
        let mut text_only = Catalog {
            geometry: &model_qwen38_contract::geometry::FLASH,
            tensors: catalog.tensors.clone(),
            ranks: 4,
            precision: None,
        };
        text_only
            .tensors
            .retain(|name, _| !name.starts_with("model.visual."));
        assert!(encoder_plan(&request, &text_only).is_err());
        for change in [
            |t: &mut TensorRecord| t.dtype = DType::F32,
            |t: &mut TensorRecord| t.shape = vec![4352, 1152],
            |t: &mut TensorRecord| t.bytes += 2,
        ] {
            let mut changed = Catalog {
                geometry: &model_qwen38_contract::geometry::FLASH,
                tensors: catalog.tensors.clone(),
                ranks: 4,
                precision: None,
            };
            change(changed.tensors.get_mut(name).unwrap());
            let error = encoder_plan(&request, &changed).unwrap_err();
            assert!(error.contains("is not BF16 of shape"), "{error}");
        }
        let mut elsewhere = request.clone();
        elsewhere.files.pop();
        let error = encoder_plan(&elsewhere, &catalog).unwrap_err();
        assert!(error.contains("is not in the checkpoint"), "{error}");
    }

    /// Other images than the text and encoder images are declined.
    #[test]
    fn plans_decline_other_image_roles() {
        let (_, mut request) = vision_catalog();
        request.image_role = "qwen38.other-images.v1".into();
        assert_eq!(plan(&request), Ok(None));
    }

    /// A planned rank has no image, so its index carries the plan's own
    /// identity and no image digest a consumer could mistake for one.
    #[test]
    fn planned_indexes_carry_the_plan_identity_not_an_image_digest() {
        let identity = "c".repeat(64);
        let bytes = runtime_index(&index(), &identity).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("sha256").is_none(), "{value}");
        assert_eq!(value["direct_identity"], identity.as_str());
        let parsed = crate::views::parse(&bytes, 1).unwrap();
        assert_eq!(parsed.sha256, None);
        assert_eq!(parsed.direct_identity.as_deref(), Some(identity.as_str()));
        assert!(parsed.image_sha256().is_err());
    }
}
