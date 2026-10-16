//! The MXFP4 fragment expert engine: tensor-parallel routed experts (every
//! expert on every rank, its intermediate columns split) in fragment-major
//! MXFP4 tiles, decoded by per-warp rings, prefilled by a
//! persistent producer kernel, and the one-token down projection with the
//! route combine fused in. Shapes are fixed by the CUDA specialization.

use super::*;

/// Ring depth of a fragment decode entry (`fragment_ring_tiles` in
/// `mimo26_fragments.cuh`).
pub fn ring_depth(v: &Variant) -> u32 {
    let parts = if v.projection == Projection::GateUp {
        2
    } else {
        1
    };
    let tokens = v.tokens_capacity;
    let slot = parts * (512 + 128) + if tokens > 1 { tokens * (128 + 16) } else { 0 };
    let budget = 50_176 - if tokens == 1 { 8192 } else { 0 };
    let k_part_tiles = v.query.k / u32::from(v.knobs.split_k_capacity) / 64;
    k_part_tiles.min(budget / (u32::from(v.knobs.warps) * slot))
}

/// The schedule of the fragment engine for an A16 fragment entry, which its
/// template instance reproduces: the ring decode
/// (`mimo26_fragments.cuh`) or the persistent prefill.
/// `None` for entries on the packed engine.
pub fn engine_schedule(v: &Variant) -> Option<template::ScheduleKnobs> {
    use template::{PrefillKnobs, RingKnobs, ScheduleKnobs};
    let gate_up = v.projection == Projection::GateUp;
    if v.fragment_decode() {
        Some(ScheduleKnobs::Ring(RingKnobs {
            split_k: u32::from(v.knobs.split_k_capacity),
            warps: u32::from(v.knobs.warps),
            depth: ring_depth(v),
            early_row: gate_up,
            reload_rows: false,
        }))
    } else if v.fragment_prefill() {
        Some(ScheduleKnobs::Prefill(PrefillKnobs {
            weight_stages: u32::from(v.weight_stages),
            activation_stages: u32::from(v.knobs.stages),
            trim_gather: gate_up && v.query.n == 1024,
            bulk_weights: false,
        }))
    } else {
        None
    }
}

/// The routed expert shapes the A16 fragment kernels are specialized for
/// (`mimo26_fragments.cuh`): 256 experts, top 8, hidden 4096, and 1,024,
/// 512 or 256 intermediate columns per rank.
const SPECIALIZED: [Shape; 3] = [
    Shape {
        top_k: 8,
        hidden: 4096,
        experts: 256,
        intermediate: 1024,
    },
    Shape {
        top_k: 8,
        hidden: 4096,
        experts: 256,
        intermediate: 512,
    },
    Shape {
        top_k: 8,
        hidden: 4096,
        experts: 256,
        intermediate: 256,
    },
];

/// A16 fragment implementations have fixed routing and projection
/// dimensions ([`SPECIALIZED`]).
pub(super) fn supported_fragment_shape(query: &GemmQuery) -> Result<(), Refusal> {
    let group = query.grouped.expect("validated grouped GEMM");
    let projection = if query.epilogues.swiglu.is_some() {
        Projection::GateUp
    } else {
        Projection::Down
    };
    let known = SPECIALIZED.into_iter().any(|shape| {
        let (n, k) = shape.projection(projection);
        group.top_k == shape.top_k && group.experts == shape.experts && (query.n, query.k) == (n, k)
    });
    if !known {
        return Err(Refusal("unsupported specialized MXFP4 fragment shape"));
    }
    Ok(())
}

/// Routed rows per fragment prefill work item. Expert ranges stay aligned to
/// the route row block (`tile_m`); a partial tile skips its empty row groups.
pub const FRAGMENT_PREFILL_TOKEN_TILE: u16 = 128;
/// Bytes of one K64 weight stage and its two barriers: 256 rows of FP4
/// fragments and their compact scales.
pub const FRAGMENT_PREFILL_WEIGHT_STAGE_BYTES: u32 = 16 * 512 + 16 * 32 + 16;
/// Bytes of one K64 activation stage and its two barriers: 128 BF16 rows.
pub const FRAGMENT_PREFILL_ACTIVATION_STAGE_BYTES: u32 = 128 * 128 + 16;
/// Per-expert tile tables, two route slots and eight 1 KB warp staging
/// buffers, after the rings.
pub const FRAGMENT_PREFILL_FIXED_BYTES: u32 = (4 * 256 + 4) * 4 + 2 * 128 * 8 + 8 * 1024;
/// Shared memory the compiler reserves beside the dynamic allocation.
pub const FRAGMENT_PREFILL_RESERVED_BYTES: u32 = 1024 + 128;
/// Two math warpgroups and a producer warpgroup whose registers they take.
pub const FRAGMENT_PREFILL_THREADS: u32 = 3 * 128;

pub fn fragment_prefill_shared_bytes(weight_stages: u8, activation_stages: u8) -> u32 {
    u32::from(weight_stages) * FRAGMENT_PREFILL_WEIGHT_STAGE_BYTES
        + u32::from(activation_stages) * FRAGMENT_PREFILL_ACTIVATION_STAGE_BYTES
        + FRAGMENT_PREFILL_FIXED_BYTES
}

/// Both ring depths are at least two and the rings fit beside the tables.
pub fn supported_fragment_prefill(knobs: CompileKnobs, weight_stages: u8) -> Result<(), Refusal> {
    if knobs.stages < 2
        || weight_stages < 2
        || fragment_prefill_shared_bytes(weight_stages, knobs.stages)
            + FRAGMENT_PREFILL_RESERVED_BYTES
            > 101_376
    {
        return Err(Refusal("fragment prefill rings do not fit"));
    }
    Ok(())
}

/// Warps of a fragment decode CTA. Gate/up warps each take one K split
/// of a 16-column strip; down warps take one strip each over the full K.
pub(super) fn fragment_decode_warps(projection: Projection, tokens_capacity: u32) -> u8 {
    if projection == Projection::GateUp && tokens_capacity == 1 {
        16
    } else {
        8
    }
}

/// Per-warp rings of `fragment_ring_tiles` slots, as in `fragments.cuh`:
/// each slot holds a K64 tile per projection part and, above one token, its
/// activation rows. One token stages its row once after the rings.
pub(super) fn fragment_decode_shared_bytes(
    projection: Projection,
    k: u32,
    splits: u8,
    warps: u8,
    tokens_capacity: u32,
) -> u32 {
    let parts = if projection == Projection::GateUp {
        2
    } else {
        1
    };
    let (rows, row_bytes, budget) = if tokens_capacity == 1 {
        (0, k * 2, 50_176 - 8192)
    } else {
        (tokens_capacity, 0, 50_176)
    };
    let slot = parts * (512 + 128) + rows * (128 + 16);
    let depth = (k / u32::from(splits) / 64).min(budget / (u32::from(warps) * slot));
    u32::from(warps) * depth * slot + row_bytes
}

/// One token saves a combine launch by fusing it into the down projection.
pub fn fused_down_combine_plan(
    tokens_capacity: u32,
    live_tokens: u32,
) -> Result<LaunchPlan, Refusal> {
    if !DECODE_BUCKETS.contains(&tokens_capacity) || live_tokens > tokens_capacity {
        return Err(Refusal("fused down combine exceeds prepared token bucket"));
    }
    let plan = LaunchPlan {
        grid: [tokens_capacity * 256, 1, 1],
        block: [128, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn fused_down_combine_arguments(handles: [u64; 7]) -> Result<ArgumentList, Refusal> {
    let args = ArgumentList(handles.map(Argument::DeviceBuffer).to_vec());
    args.validate()?;
    Ok(args)
}
