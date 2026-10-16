//! Text rank-image plan for the catalog's rank count. Vision remains
//! replicated in the catalog but is omitted here.
use crate::{
    Result,
    catalog::{Catalog, Placement, SHARED_TILE, ScaleLayout, TensorRecord, topology},
};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
pub use model_compiler_sdk::rank_index::{RankIndex, ViewIndex};
use runtime_weights::DType;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

pub const SCHEMA: &str = "chopinfer.qwen38.rank-image.v1";
pub const ALIGNMENT: u64 = 4096;
pub use model_qwen38_contract::views::{
    BF16_FRAGMENT, BF16_ROWS, F32_BLOCK_SCALES, F32_ROWS, FP8_FRAGMENT,
};
#[cfg(test)]
pub static NGRAM: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| ngram_layout(&model_qwen38_contract::geometry::FLASH));

pub fn ngram_layout(geometry: &model_qwen38_contract::geometry::Geometry) -> String {
    let width = geometry.ngram_head_width();
    format!("fp8_e4m3_ngram_head_rows_{width}_v1")
}
/// Each rank's shared expert columns (160 at TP4), zero-padded to a
/// multiple of 128 (256 at TP4). Zero
/// gate and up rows give SiLU(0) * 0 = 0, and zero down columns multiply
/// those zeros, so the padding is exact.
pub fn shared_padded(columns: u64, ranks: usize) -> Result<u64> {
    let split = topology(ranks)?
        .padded_split(columns as usize, SHARED_TILE)
        .map_err(|e| e.0)?;
    if !(columns as usize).is_multiple_of(ranks) {
        return Err("shared expert columns do not divide over the ranks".into());
    }
    Ok(split.padded_len() as u64)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Piece {
    pub rank: usize,
    pub view: usize,
    pub offset: u64,
    pub bytes: u64,
    pub slices: Vec<(usize, Range<u64>)>,
    /// The stored shape: the slices' shape, zero-padded along at most one axis.
    pub shape: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packing: Option<crate::precision_packing::PackedPiece>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImagePlan {
    /// One index per rank; the JSON is the TP4 array's at four ranks.
    pub ranks: Vec<RankIndex>,
    pub sources: BTreeMap<String, Vec<Piece>>,
    pub layers: Option<BTreeSet<usize>>,
}

pub fn selected(name: &str, layers: Option<&BTreeSet<usize>>) -> bool {
    if name.starts_with("model.visual.") {
        return false;
    }
    let Some(layers) = layers else {
        return true;
    };
    if let Some(tail) = name.strip_prefix("model.language_model.layers.") {
        return tail
            .split('.')
            .next()
            .and_then(|n| n.parse().ok())
            .is_some_and(|n| layers.contains(&n));
    }
    !name.starts_with("mtp.")
}

/// The stored dtype. Expert block scales, `A_log` and `dt_bias` widen from
/// BF16 to F32 (the MoE entries and the delta rule read `const float`);
/// every BF16 value is exactly representable in F32.
pub fn view_dtype(t: &TensorRecord) -> DType {
    if matches!(t.scale, ScaleLayout::BlockInverseScale)
        || t.name.ends_with(".A_log")
        || t.name.ends_with(".dt_bias")
    {
        DType::F32
    } else {
        t.dtype
    }
}

pub fn layout(t: &TensorRecord) -> std::borrow::Cow<'static, str> {
    if matches!(t.placement, Placement::Ngram { .. }) {
        return format!("fp8_e4m3_ngram_head_rows_{}_v1", t.shape[1]).into();
    }
    if matches!(t.scale, ScaleLayout::BlockInverseScale) {
        return F32_BLOCK_SCALES.into();
    }
    // The shared expert gate is a dot product in the MoE combine, not a GEMM.
    if t.shape.len() == 2
        && t.name.ends_with(".weight")
        && !t.name.ends_with("embed_tokens.weight")
        && !t.name.ends_with("shared_expert_gate.weight")
    {
        return if t.dtype == DType::Bf16 {
            BF16_FRAGMENT
        } else {
            FP8_FRAGMENT
        }
        .into();
    }
    match view_dtype(t) {
        DType::F32 => F32_ROWS,
        DType::Bf16 => BF16_ROWS,
        DType::I64 => "i64_row_major_v1",
        _ => "fp8_e4m3_row_major_v1",
    }
    .into()
}

pub fn fragment(layout: &str, shape: &[u64]) -> Option<FragmentWeightLayout> {
    let format = match layout {
        BF16_FRAGMENT => FragmentWeightFormat::Bf16,
        FP8_FRAGMENT => FragmentWeightFormat::Fp8E4M3,
        _ => return None,
    };
    Some(FragmentWeightLayout {
        rows: shape[shape.len() - 2] as usize,
        k: shape[shape.len() - 1] as usize,
        format,
    })
}

pub fn element_bytes(t: &TensorRecord) -> u64 {
    t.bytes / t.shape.iter().product::<u64>()
}
/// Bytes of one stored element; only widening changes the source's size.
pub fn view_element_bytes(t: &TensorRecord) -> u64 {
    if view_dtype(t) == DType::F32 {
        4
    } else {
        element_bytes(t)
    }
}

/// A main-layer shared expert projection, sharded over the ranks.
fn padded_shared(t: &TensorRecord) -> bool {
    t.name.contains(".shared_expert.") && matches!(t.placement, Placement::Axis { .. })
}

/// View order within a rank image: checkpoint names, except that each full
/// attention layer's q, k and v projections sit back to back. Every view is
/// a whole number of 16-row strips and 4096 bytes, so together they are the
/// fragment view of one `[3584, 2560]` q|k|v weight: one prefill GEMM
/// writes the packed row Q-5c's QK norm reads.
fn view_order(name: &str) -> String {
    for (i, proj) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
        if let Some(stem) = name.strip_suffix(&format!(".self_attn.{proj}.weight")) {
            return format!("{stem}.self_attn.qkv.{i}");
        }
    }
    name.into()
}
/// The bundle's per-rank sizes for a rank count it serves (2 or 4).
pub fn bundle_shape_flash(ranks: usize) -> Result<model_qwen38_contract::contract::Shape> {
    bundle_shape_for(&model_qwen38_contract::geometry::FLASH, ranks)
}

pub fn bundle_ranks(ranks: usize) -> Result<model_qwen38_contract::contract::Ranks> {
    u32::try_from(ranks)
        .ok()
        .filter(|ranks| model_qwen38_contract::contract::SERVED_RANKS.contains(ranks))
        .and_then(model_qwen38_contract::contract::Ranks::new)
        .ok_or_else(|| format!("the Qwen3.8 bundle serves 2 or 4 ranks, not {ranks}").into())
}

pub fn bundle_shape_for(
    geometry: &'static model_qwen38_contract::geometry::Geometry,
    ranks: usize,
) -> Result<model_qwen38_contract::contract::Shape> {
    u32::try_from(ranks)
        .ok()
        .and_then(model_qwen38_contract::contract::Ranks::new)
        .and_then(|ranks| model_qwen38_contract::contract::Shape::of_geometry(geometry, ranks))
        .ok_or_else(|| format!("the Qwen3.8 bundle serves 2 or 4 ranks, not {ranks}").into())
}

pub(crate) fn align(n: u64) -> u64 {
    n.div_ceil(ALIGNMENT) * ALIGNMENT
}

impl ImagePlan {
    pub fn new(catalog: &Catalog, layers: Option<&BTreeSet<usize>>) -> Result<Self> {
        if catalog
            .precision
            .as_ref()
            .is_some_and(|source| !source.legacy_text_layout())
        {
            return Err("quantized Qwen source requires prepared operation bindings before rank-image planning".into());
        }
        // Only prepared views admit F32 norms (`model_qwen38_contract::norms`).
        if let Some(span) = catalog
            .precision
            .as_ref()
            .and_then(|source| source.f32_norms().next())
        {
            return Err(format!(
                "tensor {}: F32 norm weights require prepared operation bindings",
                span.tensor
            )
            .into());
        }
        Self::build(catalog, layers, None)
    }

    pub fn with_precision(
        catalog: &Catalog,
        layers: Option<&BTreeSet<usize>>,
        precision: &kernel_core::precision::execution::PrecisionDocument,
    ) -> Result<Self> {
        precision.validate()?;
        // The legacy projection layout places as unprepared planning does,
        // and prepared views also admit F32 norms
        // (`model_qwen38_contract::norms`).
        if catalog
            .precision
            .as_ref()
            .is_some_and(|s| s.legacy_text_layout())
        {
            return Self::build(catalog, layers, None);
        }
        if model_qwen38_contract::projections::support::supplied(bundle_shape_for(
            catalog.geometry,
            catalog.ranks,
        )?)
        .is_err()
        {
            return Err(format!(
                "mixed Qwen rank-image placement requires a supplied mixed composition at TP{}",
                catalog.ranks
            )
            .into());
        }
        Self::build(catalog, layers, Some(precision))
    }

    fn build(
        catalog: &Catalog,
        layers: Option<&BTreeSet<usize>>,
        precision: Option<&kernel_core::precision::execution::PrecisionDocument>,
    ) -> Result<Self> {
        let templates = precision
            .map(|p| crate::precision_packing::templates(catalog, p))
            .transpose()?
            .unwrap_or_default();
        if layers.is_some_and(|s| {
            s.is_empty() || s.iter().any(|n| *n >= (catalog.geometry.layers() as usize))
        }) {
            return Err(format!(
                "layers must be a nonempty subset of 0..{}",
                (catalog.geometry.layers() as usize)
            )
            .into());
        }
        let ranks = catalog.ranks;
        let local = u64::from(catalog.geometry.experts()) / ranks as u64;
        let mut plan = Self {
            ranks: (0..ranks)
                .map(|rank| RankIndex {
                    schema: SCHEMA.into(),
                    rank,
                    image: format!("rank{rank}.img"),
                    image_bytes: 0,
                    sha256: String::new(),
                    views: vec![],
                })
                .collect(),
            sources: BTreeMap::new(),
            layers: layers.cloned(),
        };
        // A mixed image carries the MTP drafter's views when its precision
        // records the MTP layer's operations.
        let drafter = match precision {
            Some(p) => model_qwen38_contract::prepared::records_drafter(
                p,
                bundle_shape_for(catalog.geometry, catalog.ranks)?,
            )?,
            None => true,
        };
        let tensors: Vec<&TensorRecord> = catalog
            .tensors
            .values()
            .filter(|t| selected(&t.name, layers))
            .filter(|t| drafter || !t.name.starts_with("mtp."))
            .collect();
        for rank in 0..ranks {
            let mut groups: BTreeMap<String, Vec<(u64, &TensorRecord, Piece)>> = BTreeMap::new();
            for &t in &tensors {
                let packed = templates
                    .get(&t.name)
                    .map(|p| p.place(catalog, rank))
                    .transpose()?;
                if packed.as_ref().is_some_and(Option::is_none) {
                    continue;
                }
                let packed = packed.flatten();
                let slices = packed.as_ref().map_or_else(
                    || t.rank_slices_for(catalog.geometry, rank, ranks),
                    |p| p.slices.clone(),
                );
                if slices.is_empty() {
                    continue;
                }
                let (shape, bytes, packing) = if let Some(packed) = packed {
                    (packed.shape, packed.bytes, Some(packed.packing))
                } else {
                    let mut shape = t.shape.clone();
                    let axis = slices[0].0;
                    shape[axis] = slices.iter().map(|(_, r)| r.end - r.start).sum();
                    if padded_shared(t) {
                        let axis = usize::from(t.name.ends_with("down_proj.weight"));
                        shape[axis] = shared_padded(t.shape[axis], ranks)?;
                    }
                    let bytes = match fragment(&layout(t), &shape) {
                        Some(f) => f.packed_bytes()? as u64,
                        None => shape.iter().product::<u64>() * view_element_bytes(t),
                    };
                    (shape, bytes, None)
                };
                let (name, order) = match &t.placement {
                    Placement::Expert { local_expert, .. } => {
                        let (prefix, tail) = t.name.split_once(".experts.").ok_or("expert name")?;
                        let (_, suffix) = tail.split_once('.').ok_or("expert suffix")?;
                        let order =
                            *local_expert as u64 * 2 + u64::from(suffix.starts_with("up_proj"));
                        (
                            format!(
                                "{prefix}.experts.{}",
                                suffix.replace("gate_proj", "gate_up_proj")
                            ),
                            order,
                        )
                    }
                    Placement::Ngram { rows } => {
                        let (prefix, tail) = t.name.split_once(".shard_").ok_or("ngram name")?;
                        (
                            format!("{prefix}.{}", tail.split_once('.').ok_or("ngram suffix")?.1),
                            rows.start,
                        )
                    }
                    _ if padded_shared(t) => (
                        t.name
                            .replace(".gate_proj.", ".gate_up_proj.")
                            .replace(".up_proj.", ".gate_up_proj."),
                        u64::from(t.name.contains(".up_proj.")),
                    ),
                    _ => (t.name.clone(), 0),
                };
                // Gate and up occupy the two contiguous halves of each local expert.
                let name = if matches!(t.placement, Placement::Expert { .. }) {
                    name.replace(".up_proj.", ".gate_up_proj.")
                } else {
                    name
                };
                let name = if t.scale == ScaleLayout::BlockInverseScale
                    && name.ends_with(".weight_scale")
                {
                    format!("{name}_inv")
                } else {
                    name
                };
                groups.entry(name).or_default().push((
                    order,
                    t,
                    Piece {
                        rank,
                        view: 0,
                        offset: 0,
                        bytes,
                        slices,
                        shape,
                        packing,
                    },
                ));
            }
            let index = &mut plan.ranks[rank];
            let mut groups: Vec<_> = groups.into_iter().collect();
            groups.sort_by_cached_key(|(name, _)| view_order(name));
            for (name, mut pieces) in groups {
                pieces.sort_by_key(|p| p.0);
                let first = pieces[0].1;
                let shape = bundle_shape_for(catalog.geometry, ranks)?;
                let encoding = |tensor: &TensorRecord, piece: &Piece| -> Result<_> {
                    Ok(match &piece.packing {
                        Some(packing) => {
                            let (dtype, layout) = packing.view_for(shape)?;
                            (dtype, std::borrow::Cow::Borrowed(layout))
                        }
                        None => (view_dtype(tensor).safetensors_name(), layout(tensor)),
                    })
                };
                let (dtype, layout_id) = encoding(first, &pieces[0].2)?;
                for (_, tensor, piece) in &pieces {
                    let (dtype_other, layout_other) = encoding(tensor, piece)?;
                    if dtype != dtype_other || layout_id != layout_other {
                        return Err(format!("view {name}: contributing tensors require incompatible encodings or layouts").into());
                    }
                }
                let mut shape = pieces[0].2.shape.clone();
                match first.placement {
                    Placement::Expert { .. } => {
                        if name.contains(".gate_up_proj.") {
                            shape[0] *= 2;
                        }
                        shape.insert(0, local);
                    }
                    _ => shape[0] = pieces.iter().map(|p| p.2.shape[0]).sum(),
                }
                let offset = align(index.image_bytes);
                let bytes = pieces.iter().map(|p| p.2.bytes).sum();
                let view = index.views.len();
                let mut next = offset;
                for (_, t, mut piece) in pieces {
                    piece.offset = next;
                    piece.view = view;
                    next += piece.bytes;
                    plan.sources.entry(t.name.clone()).or_default().push(piece);
                }
                index.views.push(ViewIndex {
                    name,
                    offset,
                    bytes,
                    shape,
                    dtype: dtype.into(),
                    layout_id: layout_id.into(),
                    source_tensor_digests: BTreeMap::new(),
                });
                index.image_bytes = next;
            }
            index.image_bytes = align(index.image_bytes);
        }
        Ok(plan)
    }
    pub fn image_bytes(&self) -> Vec<u64> {
        self.ranks.iter().map(|r| r.image_bytes).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ngram_rank_rows_flash;

    /// Every rank index passes the bundle's check of the views its kernels read.
    fn bundle_views_validate(plan: &ImagePlan) {
        let shape = bundle_shape_flash(plan.ranks.len()).unwrap();
        for index in &plan.ranks {
            let mut index = index.clone();
            index.sha256 = "0".repeat(64);
            let bytes = serde_json::to_vec(&index).unwrap();
            let parsed = model_qwen38_contract::views::parse(&bytes, index.rank as u32).unwrap();
            model_qwen38_contract::views::validate(&parsed, shape).unwrap();
            assert_eq!(
                index.image_bytes,
                model_qwen38_contract::views::image_bytes(index.rank as u32, shape)
            );
        }
    }

    #[test]
    fn checkpoint_plan_passes_bundle_view_validation() {
        let Ok(root) = std::env::var("CHOPINFER_QWEN38_CHECKPOINT") else {
            eprintln!("SKIP: set CHOPINFER_QWEN38_CHECKPOINT for the bundle view check");
            return;
        };
        let catalog = Catalog::load(std::path::Path::new(&root)).unwrap();
        bundle_views_validate(&ImagePlan::new(&catalog, None).unwrap());
    }
    fn synthetic_catalog() -> Catalog {
        crate::catalog::synthetic(4)
    }

    #[test]
    fn complete_plan_preserves_placement_and_covers_each_view() {
        let catalog = synthetic_catalog();
        let plan = ImagePlan::new(&catalog, None).unwrap();
        let mut by_view = BTreeMap::<(usize, usize), Vec<&Piece>>::new();
        for p in plan.sources.values().flatten() {
            by_view.entry((p.rank, p.view)).or_default().push(p);
        }
        for (rank, index) in plan.ranks.iter().enumerate() {
            for (v, view) in index.views.iter().enumerate() {
                let mut pieces = by_view[&(rank, v)].clone();
                pieces.sort_by_key(|p| p.offset);
                let mut offset = view.offset;
                for p in pieces {
                    assert_eq!(p.offset, offset);
                    offset += p.bytes;
                }
                assert_eq!(offset, view.offset + view.bytes);
                assert!(!view.name.starts_with("model.visual."));
                if view.name.contains(".experts.gate_up_proj.weight") {
                    assert_eq!(view.shape[0], 128);
                }
                if view.layout_id == NGRAM.as_str() {
                    let rows = ngram_rank_rows_flash(rank, 4);
                    assert_eq!(view.shape, [rows.end - rows.start, 160]);
                }
            }
            assert!(index.image_bytes > 48_000_000_000 && index.image_bytes < 49_000_000_000);
        }
        let table =
            kernel_core::ngram_oracle::ngram_tables(248320, 3, 8, 20_000_000, 0, 0, 128).unwrap();
        for rank in 0..4 {
            assert_eq!(
                ngram_rank_rows_flash(rank, 4).start,
                table.offsets[rank * 4]
            );
        }
        bundle_views_validate(&plan);
        let subset = ImagePlan::new(&catalog, Some(&BTreeSet::from([0, 1, 2, 3]))).unwrap();
        assert!(subset.sources.contains_key("lm_head.weight"));
        assert!(!subset.sources.keys().any(|n| n.starts_with("mtp.")));
    }

    /// F32 norm records place as F32 row views. Their stored dtype survives
    /// the written index, which prepared validation admits and from which
    /// the bundle binds each consumer, while legacy validation refuses it.
    #[test]
    fn f32_norm_views_round_trip_through_the_rank_index() {
        let mut catalog = synthetic_catalog();
        let names = [
            "model.language_model.layers.4.mlp_hyper_connection.hc_norm.weight",
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            "model.language_model.layers.1.ple.norm_query.weight",
        ];
        for name in names {
            let record = catalog.tensors.get_mut(name).unwrap();
            assert_eq!(record.dtype, DType::Bf16);
            record.dtype = DType::F32;
            record.bytes *= 2;
        }
        let plan = ImagePlan::new(&catalog, None).unwrap();
        let shape = bundle_shape_flash(4).unwrap();
        for index in &plan.ranks {
            let view = index.views.iter().find(|v| v.name == names[0]).unwrap();
            assert_eq!(
                (view.dtype.as_str(), view.layout_id.as_str()),
                ("F32", F32_ROWS)
            );
            assert_eq!(view.bytes, 4 * 10240);
            let mut index = index.clone();
            index.sha256 = "0".repeat(64);
            let bytes = serde_json::to_vec(&index).unwrap();
            let parsed = model_qwen38_contract::views::parse(&bytes, index.rank as u32).unwrap();
            let error = model_qwen38_contract::views::validate(&parsed, shape).unwrap_err();
            assert!(error.contains("F32"), "{error}");
            let mut required = model_qwen38_contract::views::required(parsed.rank, shape);
            model_qwen38_contract::norms::admit(&parsed, &mut required);
            model_qwen38_contract::views::check(&parsed, [], required, shape).unwrap();
            let norms = model_qwen38_contract::norms::NormWeights::from_index(&parsed);
            for name in names {
                assert_eq!(norms.dtype(name), kernel_core::contract::DType::F32);
            }
            assert_eq!(
                norms.dtype("model.language_model.layers.1.ple.norm_key.weight"),
                kernel_core::contract::DType::Bf16
            );
        }
    }

    /// The per-rank shapes of docs/topology-qwen.md at 2 ranks.
    #[test]
    fn tp2_plans_have_the_documented_shapes() {
        let (ranks, q, attn, lqkv, lvalue, heads, local, shared, vocab, bytes) = (
            2,
            6144,
            3072,
            5120,
            3072,
            24,
            256,
            384,
            124_160,
            93_000_000_000..95_000_000_000,
        );
        let plan = ImagePlan::new(&crate::catalog::synthetic(ranks), None).unwrap();
        assert_eq!(plan.ranks.len(), ranks);
        for (rank, index) in plan.ranks.iter().enumerate() {
            assert_eq!(index.rank, rank);
            let views: BTreeMap<&str, &ViewIndex> =
                index.views.iter().map(|v| (v.name.as_str(), v)).collect();
            let shape = |name: &str| views[name].shape.clone();
            let full = "model.language_model.layers.3";
            let linear = "model.language_model.layers.0";
            assert_eq!(shape(&format!("{full}.self_attn.q_proj.weight")), [q, 2560]);
            assert_eq!(
                shape(&format!("{full}.self_attn.k_proj.weight")),
                [256, 2560]
            );
            assert_eq!(
                shape(&format!("{full}.self_attn.v_proj.weight")),
                [256, 2560]
            );
            assert_eq!(
                shape(&format!("{full}.self_attn.o_proj.weight")),
                [2560, attn]
            );
            assert_eq!(
                shape(&format!("{linear}.linear_attn.in_proj_qkv.weight")),
                [lqkv, 2560]
            );
            assert_eq!(
                shape(&format!("{linear}.linear_attn.conv1d.weight")),
                [lqkv, 1, 4]
            );
            assert_eq!(
                shape(&format!("{linear}.linear_attn.in_proj_z.weight")),
                [lvalue, 2560]
            );
            assert_eq!(
                shape(&format!("{linear}.linear_attn.out_proj.weight")),
                [2560, lvalue]
            );
            for name in ["in_proj_a.weight", "in_proj_b.weight"] {
                assert_eq!(
                    shape(&format!("{linear}.linear_attn.{name}")),
                    [heads, 2560]
                );
            }
            assert_eq!(shape(&format!("{linear}.linear_attn.A_log")), [heads]);
            assert_eq!(
                shape(&format!("{linear}.mlp.experts.gate_up_proj.weight")),
                [local, 1280, 2560]
            );
            assert_eq!(
                shape(&format!("{linear}.mlp.shared_expert.gate_up_proj.weight")),
                [2 * shared, 2560]
            );
            assert_eq!(
                shape(&format!("{linear}.mlp.shared_expert.down_proj.weight")),
                [2560, shared]
            );
            assert_eq!(shape("mtp.layers.0.self_attn.q_proj.weight"), [q, 2560]);
            assert_eq!(shape("lm_head.weight"), [vocab, 2560]);
            let table = index
                .views
                .iter()
                .find(|v| v.layout_id == NGRAM.as_str())
                .unwrap();
            let rows = ngram_rank_rows_flash(rank, ranks);
            assert_eq!(table.shape, [rows.end - rows.start, 160]);
            eprintln!("TP{ranks} rank {rank}: {} bytes", index.image_bytes);
            assert!(bytes.contains(&index.image_bytes));
        }
        bundle_views_validate(&plan);
    }

    #[test]
    fn mtp_layer_is_placed_like_a_main_full_layer() {
        let plan = ImagePlan::new(&synthetic_catalog(), None).unwrap();
        for index in &plan.ranks {
            let views: BTreeMap<&str, &ViewIndex> =
                index.views.iter().map(|v| (v.name.as_str(), v)).collect();
            let mut layer = 0;
            for (name, view) in &views {
                let Some(tail) = name.strip_prefix("mtp.layers.0.") else {
                    continue;
                };
                let main = views[format!("model.language_model.layers.3.{tail}").as_str()];
                assert_eq!(
                    (&view.shape, &view.dtype, &view.layout_id, view.bytes),
                    (&main.shape, &main.dtype, &main.layout_id, main.bytes),
                    "{name}"
                );
                layer += 1;
            }
            let main = views
                .keys()
                .filter(|n| n.starts_with("model.language_model.layers.3."))
                .count();
            assert_eq!(layer, main);
            let shared = views["mtp.layers.0.mlp.shared_expert.gate_up_proj.weight"];
            assert_eq!(shared.shape, [512, 2560]);
            assert_eq!(
                views["mtp.layers.0.mlp.shared_expert.down_proj.weight"].shape,
                [2560, 256]
            );
            let [q, k, v] = ["q_proj", "k_proj", "v_proj"]
                .map(|p| views[format!("mtp.layers.0.self_attn.{p}.weight").as_str()]);
            assert_eq!(
                (q.offset + q.bytes, k.offset + k.bytes),
                (k.offset, v.offset)
            );
            assert_eq!(q.shape[0] + k.shape[0] + v.shape[0], 3584);
            for name in ["mtp.fc_hidden.weight", "mtp.fc_embedding.weight"] {
                let fc = views[name];
                assert_eq!(
                    (
                        fc.shape.as_slice(),
                        fc.dtype.as_str(),
                        fc.layout_id.as_str()
                    ),
                    (&[2560, 2560][..], "BF16", BF16_FRAGMENT)
                );
            }
            assert_eq!(views["mtp.pre_fc_norm_hidden.weight"].shape, [10240]);
            assert_eq!(views["mtp.pre_fc_norm_embedding.weight"].shape, [2560]);
            assert_eq!(
                views["mtp.hyper_connection_mixer.input_mix_weight_down.weight"].shape,
                [320, 10240]
            );
            // The drafter's views are the image's tail.
            let first = index
                .views
                .iter()
                .position(|v| v.name.starts_with("mtp."))
                .unwrap();
            assert!(
                index.views[..first]
                    .iter()
                    .all(|v| !v.name.starts_with("mtp."))
            );
            assert!(
                index.views[first..]
                    .iter()
                    .all(|v| v.name.starts_with("mtp."))
            );
        }
    }
}
