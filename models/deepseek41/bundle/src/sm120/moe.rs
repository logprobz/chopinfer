//! Model-owned registrations over the shared chained engine and row quantizer.

use kernel_build::CubinSpec;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::backend::Sm120;
use kernels_sm120::moe::chained::{LocalRoute, prefill, quantized, row_bias, streamed};
use kernels_sm120::row::block_quantize::{self, Entry};
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::moe::{self, Experts};
use model_deepseek41_contract::serving::{DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS};

pub const FAMILY: &str = "routed_moe";
pub const ACTIVE_FAMILY: &str = "routed_moe_active";
pub const STREAMED_FAMILY: &str = "routed_moe_streamed";

pub const PREFILL_FAMILY: &str = "routed_moe_prefill";

/// Experimental exact prefill entries; serving keeps the measured decode
/// policy and the old prefill entries until this family is qualified.
pub fn prefill_projections(g: &Geometry) -> Result<Vec<quantized::Projection>, String> {
    projections(g, 4)?
        .into_iter()
        .filter(|p| p.row_tile == 64)
        .map(|p| p.with_prefill_tiles().map_err(|e| e.to_string()))
        .collect()
}
pub fn prefill_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        specs.push((
            PREFILL_FAMILY,
            prefill::cubin_spec(
                &format!("prefill_moe_{}_compute_sm120", g.kernel_tag),
                &prefill_projections(g)?,
            ),
        ));
    }
    specs.push((
        PREFILL_FAMILY,
        prefill::calibration_cubin_spec("prefill_moe_calibration_sm120"),
    ));
    Ok(specs)
}

/// TP4 streamed entries, with serving admission kept model-owned below.
pub fn streamed_projections(g: &Geometry) -> Result<Vec<quantized::Projection>, String> {
    let mut entries = Vec::new();
    for set in moe::expert_sets(g) {
        for tokens in [8, 64] {
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                let mut entry = quantized::Projection::new(
                    moe::projection(g, 4, set, kind, tokens)?,
                    64,
                    tokens,
                )
                .and_then(quantized::Projection::with_streamed_chains)
                .map_err(|e| e.to_string())?;
                entry.symbol = format!("{}_{}", g.kernel_tag, entry.symbol);
                entries.push(entry);
            }
        }
    }
    Ok(entries)
}

/// SM120 TP4, 250 W, 2026-10-19: all 40 eligible cases pass complete
/// graph bit parity. Combined gate/up and down beat the previous selection
/// in both banks through 64 rows. Backbone decode 1/8: 121/245 -> 36/121 us;
/// verify 6/48: 300/2166 -> 101/946 us. DSpark decode 1/32:
/// 176/505 -> 26/290 us. The DSpark Decode-1 whole graph loses despite
/// those projection wins, so serving_compact_projections overrides it.
/// The shared ceil quantizers still dominate the small whole graphs.
/// Rows above 64, prefill, other geometries and other ranks keep the old
/// entries. The backend excludes streamed entries from prefill selection.
pub fn streamed_qualified(g: &Geometry, ranks: u32) -> bool {
    g.id == model_deepseek41_contract::geometry::FLASH.id && ranks == 4
}

pub fn serving_streamed_projections(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<quantized::Projection>, String> {
    if streamed_qualified(g, ranks) {
        streamed_projections(g)
    } else {
        Ok(Vec::new())
    }
}

/// Warm SM120 TP4, 250 W, 2026-10-19, DSpark Decode-1 whole graphs:
/// first run old/compact/streamed = 261.968/259.952/269.104 us;
/// second run compact = 257.776 us and paired old/streamed =
/// 260.064/268.224 us (+3.138%). Treat the repeatable 7-8 us loss as
/// a regression and select the fastest measured compact column path.
/// Restrict both projections to Decode-1; Verify-1 still uses streamed.
/// This crossover uses the current ceil quantizers and needs remeasurement
/// when their implementation changes.
pub fn serving_compact_projections(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<quantized::Projection>, String> {
    if !streamed_qualified(g, ranks) {
        return Ok(Vec::new());
    }
    projections(g, ranks)?
        .into_iter()
        .filter(|e| {
            e.row_tile == 16
                && e.query
                    .projection
                    .gemm
                    .grouped
                    .is_some_and(|g| g.experts == 128 && g.top_k == 3)
        })
        .map(|e| {
            e.with_compact_columns()
                .and_then(|e| e.with_max_rows(1))
                .map(|e| e.with_phase(kernel_core::precision::ExecutionPhase::Decode))
                .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn streamed_routes(g: &Geometry) -> Vec<row_bias::Route> {
    moe::expert_sets(g)
        .into_iter()
        .flat_map(|set| {
            [8, 64].map(|capacity| row_bias::Route {
                local: LocalRoute {
                    experts: set.count,
                    local_experts: set.count,
                    top_k: set.top_k,
                    capacity,
                    pad: 64,
                },
            })
        })
        .collect()
}

pub fn streamed_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        specs.push((
            STREAMED_FAMILY,
            streamed::cubin_spec(
                &format!("streamed_moe_{}_compute_sm120", g.kernel_tag),
                &streamed_projections(g)?,
            ),
        ));
        specs.push((
            STREAMED_FAMILY,
            row_bias::cubin_spec(
                &format!("streamed_moe_{}_routes_sm120", g.kernel_tag),
                &[],
                &streamed_routes(g),
            )
            .map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}

/// Warm SM120 TP4, 250 W, 2026-10-19: Flash gate/up wins only at the
/// measured one-row bucket (0.162 -> 0.079 ms). At 8/32 rows it takes
/// 0.306/1.075 ms versus 0.195/0.812 ms. Active down loses even at one
/// row (0.068 versus 0.046 ms), and by 5.8x/6.3x at 8/32 rows.
/// This remains the comparison baseline, superseded by qualified streamed
/// entries in serving. Larger or unmeasured buckets, other ranks,
/// the drafter and every down projection retain the old column schedule.
/// At K=576 the four-warp schedule executes eight chains for five useful
/// ones. Its 16-column tiles gather each input four times as often as the
/// old 64-column tiles; the 376-CTA cap serializes the remaining tiles.
/// The compact-column candidate removes those costs using the locked
/// column kernel with a route-count grid bound. Only its qualified DSpark
/// Decode-1 bucket serves; the other compact buckets remain diagnostic.
pub fn serving_active_projections(
    g: &Geometry,
    ranks: u32,
) -> Result<Vec<quantized::Projection>, String> {
    if g.id != model_deepseek41_contract::geometry::FLASH.id || ranks != 4 {
        return Ok(Vec::new());
    }
    active_projections(g, ranks)?
        .into_iter()
        .filter(|e| {
            e.gate_up()
                && e.tokens_capacity == DECODE_BATCH
                && e.query.projection.gemm.grouped.unwrap().experts == g.shape.moe.experts
        })
        .map(|e| e.with_max_rows(1).map_err(|e| e.to_string()))
        .collect()
}

pub fn active_projections(g: &Geometry, ranks: u32) -> Result<Vec<quantized::Projection>, String> {
    let mut entries = Vec::new();
    for set in moe::expert_sets(g) {
        for tokens in [DECODE_BATCH, DECODE_BATCH * MAX_VERIFY_ROWS] {
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                let query = moe::projection(g, ranks, set, kind, tokens)?;
                let mut entry = quantized::Projection::new(query, 16, tokens)
                    .and_then(quantized::Projection::with_active_experts)
                    .map_err(|e| e.to_string())?;
                entry.symbol = format!("{}_{}", g.kernel_tag, entry.symbol);
                entries.push(entry);
            }
        }
    }
    Ok(entries)
}

pub fn active_routes(g: &Geometry) -> Vec<row_bias::Route> {
    routes(g)
        .into_iter()
        .filter(|r| r.local.pad == 16)
        .map(|mut r| {
            r.local.capacity = DECODE_BATCH * MAX_VERIFY_ROWS;
            r
        })
        .collect()
}

pub fn active_cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut entries = Vec::new();
        for &ranks in SUPPORTED_RANKS {
            entries.extend(active_projections(g, ranks)?);
        }
        specs.push((
            ACTIVE_FAMILY,
            quantized::active_cubin_spec(&format!("moe_{}_active_sm120", g.kernel_tag), &entries),
        ));
        specs.push((
            ACTIVE_FAMILY,
            row_bias::cubin_spec(
                &format!("moe_{}_active_routes_sm120", g.kernel_tag),
                &[],
                &active_routes(g),
            )
            .map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}

pub fn projections(g: &Geometry, ranks: u32) -> Result<Vec<quantized::Projection>, String> {
    let mut entries = Vec::new();
    for set in moe::expert_sets(g) {
        for (tile, tokens) in [(16, DECODE_BATCH), (64, PREFILL_TOKENS)] {
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                let query = moe::projection(g, ranks, set, kind, tokens)?;
                let mut entry =
                    quantized::Projection::new(query, tile, tokens).map_err(|e| e.to_string())?;
                entry.symbol = format!("{}_{}", g.kernel_tag, entry.symbol);
                entries.push(entry);
            }
        }
    }
    Ok(entries)
}

pub fn routers(g: &Geometry) -> Vec<row_bias::Router> {
    moe::expert_sets(g)
        .into_iter()
        .flat_map(|set| {
            [DECODE_BATCH, PREFILL_TOKENS].map(|capacity| row_bias::Router {
                experts: set.count,
                top_k: set.top_k,
                capacity,
            })
        })
        .collect()
}

pub fn routes(g: &Geometry) -> Vec<row_bias::Route> {
    moe::expert_sets(g)
        .into_iter()
        .flat_map(|Experts { count, top_k }| {
            [(16, DECODE_BATCH), (64, PREFILL_TOKENS)].map(|(pad, capacity)| row_bias::Route {
                local: LocalRoute {
                    experts: count,
                    local_experts: count,
                    top_k,
                    capacity,
                    pad,
                },
            })
        })
        .collect()
}

pub fn quantizers(g: &Geometry, ranks: u32) -> Result<Vec<Entry>, String> {
    model_deepseek41_contract::admission::check_ranks(g, ranks)?;
    let mut widths = vec![g.shape.hidden];
    widths.extend(
        moe::expert_sets(g)
            .iter()
            .map(|set| set.top_k * (g.shape.moe.intermediate / ranks)),
    );
    widths.sort_unstable();
    widths.dedup();
    widths
        .into_iter()
        .map(|width| {
            Entry::new(
                moe::quantizer(width),
                PREFILL_TOKENS,
                format!("chop_{}_moe_ceil_w{width}_v1", g.kernel_tag),
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut compute = Vec::new();
        let mut quant = Vec::new();
        for &ranks in SUPPORTED_RANKS {
            compute.extend(projections(g, ranks)?);
            for entry in quantizers(g, ranks)? {
                if !quant.iter().any(|e: &Entry| e.symbol == entry.symbol) {
                    quant.push(entry);
                }
            }
        }
        specs.push((
            FAMILY,
            quantized::cubin_spec(&format!("moe_{}_compute_sm120", g.kernel_tag), &compute),
        ));
        specs.push((
            FAMILY,
            row_bias::cubin_spec(
                &format!("moe_{}_routes_sm120", g.kernel_tag),
                &routers(g),
                &routes(g),
            )
            .map_err(|e| e.to_string())?,
        ));
        specs.push((
            FAMILY,
            block_quantize::cubin_spec(&format!("moe_{}_quantize_sm120", g.kernel_tag), &quant)
                .map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}

pub fn register(mut backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    for entry in quantizers(g, ranks)? {
        backend = backend.with_block_quantize(entry);
    }
    for entry in projections(g, ranks)? {
        // Capacity is the backend's first selection key. Do not let the
        // old 32-row entries shadow a qualified 64-row streamed entry.
        if !streamed_qualified(g, ranks) || entry.row_tile != 16 {
            backend = backend.with_quantized_chained_projection(entry);
        }
    }
    if !streamed_qualified(g, ranks) {
        for entry in serving_active_projections(g, ranks)? {
            backend = backend.with_quantized_chained_projection(entry);
        }
    }
    for entry in serving_streamed_projections(g, ranks)? {
        backend = backend.with_quantized_chained_projection(entry);
    }
    for entry in serving_compact_projections(g, ranks)? {
        backend = backend.with_quantized_chained_projection(entry);
    }
    if streamed_qualified(g, ranks) {
        for entry in streamed_routes(g) {
            backend = backend.with_row_bias_route(entry);
        }
    }
    for entry in routers(g) {
        backend = backend.with_row_bias_router(entry);
    }
    for entry in routes(g) {
        backend = backend.with_row_bias_route(entry);
    }
    for entry in active_routes(g) {
        backend = backend.with_row_bias_route(entry);
    }
    Ok(backend)
}
