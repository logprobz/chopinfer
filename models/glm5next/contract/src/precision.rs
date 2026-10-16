//! The projections glm5next lowers through the projection API and their
//! declared uses: the drafter's included, each with its use profile's
//! phases, capacities and output dtypes from the declaration, the rows the
//! rank image stores of each, and the weight format of each in the served
//! FP8 export. The shared kernels' offers for them are the SM120 module's
//! (`crate::sm120::precision`).
//!
//! The bundle lowers these projections through the projection API; the
//! routed and shared experts' gate-up and the routed down projection stay
//! on the chained MoE engine, the embedding on the row engine.

use kernel_core::contract::{ScaleMode, WeightFormat};
use kernel_core::precision::OperationUse;
use kernel_core::precision::format::registry;

use crate::contract::Shape;

/// The projections the bundle lowers through the projection API, the
/// drafter's included.
pub const LOWERED: &[&str] = &[
    "head",
    "linear.q",
    "linear.k",
    "linear.v",
    "linear.beta",
    "linear.forget-a",
    "linear.forget-b",
    "linear.gate-a",
    "linear.gate-b",
    "linear.output",
    "latent.q-a",
    "latent.q-b",
    "latent.kv-a",
    "latent.output",
    "latent.indexer.q",
    "latent.indexer.k",
    "latent.indexer.weights",
    "dense.down",
    "moe.router",
    "moe.shared.down",
    "mtp.eh",
    "mtp.latent.q-a",
    "mtp.latent.q-b",
    "mtp.latent.kv-a",
    "mtp.latent.output",
    "mtp.latent.indexer.q",
    "mtp.latent.indexer.k",
    "mtp.latent.indexer.weights",
    "mtp.moe.router",
    "mtp.moe.shared.down",
];

pub(crate) fn declared(
    id: &str,
) -> Result<
    (
        &'static projection_spec::Declaration,
        &'static projection_spec::Geometry,
        &'static projection_spec::Projection,
    ),
    String,
> {
    use std::sync::OnceLock;
    static DECLARED: OnceLock<
        Result<(projection_spec::Declaration, projection_spec::Geometry), String>,
    > = OnceLock::new();
    let (declaration, geometry) = DECLARED
        .get_or_init(|| {
            let ranks = Shape::TP4.ranks();
            Ok((
                crate::projections::declaration(ranks).map_err(|e| e.to_string())?,
                crate::projections::geometry(ranks).map_err(|e| e.to_string())?,
            ))
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let projection = declaration
        .projections
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("projection {id} is not declared"))?;
    Ok((declaration, geometry, projection))
}

/// The declared uses of projection `id` (its use profile's phases and
/// capacities), each with its output dtype.
pub fn uses(id: &str) -> Result<Vec<(OperationUse, kernel_core::contract::DType)>, String> {
    let (declaration, geometry, projection) = declared(id)?;
    let profile = match &projection.use_profile {
        Some(projection_spec::profiles::UseProfileRef::Id(profile)) => profile,
        _ => return Err(format!("projection {id} names no use profile")),
    };
    declaration
        .use_profiles
        .get(profile)
        .ok_or_else(|| format!("projection {id}: no use profile {profile}"))?
        .iter()
        .map(|used| {
            let contract = used
                .contract
                .as_ref()
                .ok_or_else(|| format!("projection {id}: a use without its contract"))?;
            Ok((
                OperationUse {
                    phase: used.phase,
                    capacity: u32::try_from(used.capacity.resolve(geometry).map_err(|e| e.0)?)
                        .map_err(|_| format!("projection {id}: capacity exceeds u32"))?,
                    graph: used.graph,
                },
                contract.output_dtype,
            ))
        })
        .collect()
}

/// The rows the rank image stores of projection `id`'s weight when the
/// rank's share holds `rows` rows in `dtype`: for a lowered projection,
/// padded to the format's row tile as [`domain`] pads the operation's
/// geometry (BF16 dense rows to 64), the rows past the share zero; else
/// the share's rows. Block FP8 rows are whole tiles already; NVFP4 views
/// are not lowered yet.
pub fn stored_rows(id: &str, dtype: crate::catalog::DType, rows: u64) -> Result<u64, String> {
    use crate::catalog::DType;
    let weight = match dtype {
        DType::Bf16 => WeightFormat::Bf16,
        DType::F8E4M3 => WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        DType::F32 | DType::U8 => return Ok(rows),
    };
    if !LOWERED.contains(&id) {
        return Ok(rows);
    }
    let (_, _, projection) = declared(id)?;
    let plugin = registry::by_format(weight).map_err(|e| e.to_string())?;
    Ok(rows.next_multiple_of(plugin.padding_tile(projection.family()).rows as u64))
}

/// The weight format each lowered projection's checkpoint rows hold in the
/// served FP8 export (`crate::views::rank_views`): block FP8 where the
/// export quantizes it, BF16 elsewhere.
pub fn checkpoint_formats(shape: Shape) -> Result<Vec<(&'static str, WeightFormat)>, String> {
    use crate::catalog::DType;
    use crate::views::Binding;
    let views = crate::views::rank_views(shape, 0)?;
    LOWERED
        .iter()
        .map(|&id| {
            let view = views
                .iter()
                .find(|view| {
                    view.binding == Binding::Projection(id) && view.name.ends_with(".weight")
                })
                .ok_or_else(|| format!("projection {id} has no weight view"))?;
            Ok((
                id,
                match view.dtype {
                    DType::F8E4M3 => WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
                    DType::Bf16 => WeightFormat::Bf16,
                    other => return Err(format!("projection {id} stores {other:?}")),
                },
            ))
        })
        .collect()
}

/// The weight format each lowered projection's rows hold in an export
/// stored in `layout` (`crate::views::checkpoint_rank_views`): block FP8,
/// NVFP4 (U8 codes) or BF16.
pub fn layout_formats(
    shape: Shape,
    layout: &crate::quantization::Layout,
) -> Result<Vec<(&'static str, WeightFormat)>, String> {
    use crate::catalog::DType;
    use crate::views::Binding;
    let views = crate::views::checkpoint_rank_views(shape, layout, 0)?;
    LOWERED
        .iter()
        .map(|&id| {
            let view = views
                .iter()
                .find(|view| {
                    view.binding == Binding::Projection(id) && view.name.ends_with(".weight")
                })
                .ok_or_else(|| format!("projection {id} has no weight view"))?;
            Ok((
                id,
                match view.dtype {
                    DType::F8E4M3 => WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
                    DType::U8 => WeightFormat::Nvfp4,
                    DType::Bf16 => WeightFormat::Bf16,
                    other => return Err(format!("projection {id} stores {other:?}")),
                },
            ))
        })
        .collect()
}

/// The router's scores per row, and the columns its GEMM writes: every
/// expert, padded to the BF16 weight's 64-row tile
/// ([`stored_rows`]), 288 of 320.
pub fn router_columns(shape: Shape) -> (u32, u32) {
    let experts = shape.geometry.shape.moe.experts;
    (experts, experts.next_multiple_of(64))
}
