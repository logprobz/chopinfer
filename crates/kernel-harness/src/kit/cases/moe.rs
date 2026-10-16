//! Cases of the MoE ops around the experts: the router, a rank's routes
//! (router and route metadata), the combine of each token's routed rows, a
//! rank's routed sum (with and without an addend) and the chained engine's
//! dense projections in each of their schedules, with the SM120 selection
//! the lab's cubins hold. The routed expert projections behind their routes
//! are `super::moe_routed`'s.
//!
//! Router logits are distinct multiples of 1/32 in a shuffled order per
//! token, so every key a softmax or grouped router ranks is exact in F32
//! and F64, and half the softmax tokens tie their k-th and next expert
//! exactly: the IDs compare as bits, ties included. A sigmoid-bias token
//! whose keys come within [`KEY_GAP`] of each other around its selection is
//! drawn again, since F32 and F64 sigmoids may rank them differently.
//! Weights compare within the relative error the router's F32 computation
//! allows ([`weight_roundoffs`]). Route metadata, combines of preweighted
//! rows and routed sums compare as exact bits.

use super::super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use super::encode_weight;
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::poison::WriteMap;
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, MoeQuery, RouterKind, RowQuery};
use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint};
use kernel_core::gated_mlp_oracle::{
    ENGINE_ROUNDING, clamped_silu_glu, dense_gated_mlp, local_route_metadata,
};
use kernel_core::moe::{
    CombineBuffers, ExpertCombineOp, RouteBuffers, RouteMetadata, RoutedSumBuffers, RouterBuffers,
};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch};
use kernel_core::moe_oracle::ExpertWeights;
use kernel_core::occupancy::EntryResources;
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::precision::buffers::QuantizedWeights;
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::backend::Sm120;
use kernels_sm120::moe::chained::{self, Projection, RoutedSumOutput, ServedShapes, split};
use kernels_sm120::row::RowInstantiation;
use std::cell::RefCell;
use std::rc::Rc;

/// The token rows every row-engine MoE case serves.
pub const CAPACITY: u32 = 16;

/// The token rows of the dense prefill and verify cases: past the decode
/// route's tokens, so the 64-row forms run.
pub const PREFILL_CAPACITY: u32 = 33;

/// The scaling factor of the sigmoid-bias routers the cases route with.
pub(super) const SCALE: f32 = 2.5;

/// The least gap between a sigmoid-bias token's ranked keys around its
/// selection.
const KEY_GAP: f64 = 1e-4;

/// The rank the split instances' cases launch as: with every expert local,
/// every rank keeps the routes of shard zero, all of them.
pub(super) const SPLIT_RANK: u32 = 0;

pub(super) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(super) fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A small xorshift stream from `seed`.
pub(super) fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

/// The selection bias per expert of a router's case: three levels, so the
/// bias moves experts across the selection.
pub(super) fn bias(experts: usize) -> Vec<f32> {
    (0..experts).map(|e| (e % 3) as f32 * 0.125).collect()
}

/// The ranked keys of one token, sigmoid plus bias, in F64.
fn sigmoid_keys(row: &[f32], bias: &[f32]) -> Vec<f64> {
    row.iter()
        .zip(bias)
        .map(|(&l, &b)| 1.0 / (1.0 + (-f64::from(l)).exp()) + f64::from(b))
        .collect()
}

/// Whether the top `take` keys and the next one are each at least
/// [`KEY_GAP`] apart.
fn separated(keys: &[f64], take: usize) -> bool {
    let mut sorted = keys.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[..=take.min(sorted.len() - 1)]
        .windows(2)
        .all(|pair| pair[0] - pair[1] >= KEY_GAP)
}

/// `rows` tokens of router logits over `experts`: distinct multiples of
/// 1/32 in a shuffled order. A sigmoid-bias token is drawn again until its
/// keys around the selection are [`KEY_GAP`] apart. Odd tokens then tie an
/// unselected expert to the k-th exactly: a softmax router's next one, a
/// sigmoid-bias router's first one of the same bias.
fn logits(rows: usize, experts: usize, top_k: usize, kind: RouterKind, seed: u64) -> Vec<f32> {
    let mut next = stream(seed);
    let bias = bias(experts);
    let mut out = Vec::with_capacity(rows * experts);
    for token in 0..rows {
        let row = loop {
            let mut row = (0..experts)
                .map(|e| (e as f32 - experts as f32 / 2.0) / 32.0)
                .collect::<Vec<_>>();
            for at in (1..experts).rev() {
                row.swap(at, (next() % (at as u64 + 1)) as usize);
            }
            let tie = token % 2 == 1 && top_k < experts;
            match kind {
                RouterKind::SigmoidBiasSelectNormalized { .. } => {
                    let keys = sigmoid_keys(&row, &bias);
                    if !separated(&keys, top_k) {
                        continue;
                    }
                    let mut order = (0..experts).collect::<Vec<_>>();
                    order.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));
                    let kth = order[top_k - 1];
                    if let Some(&other) = order[top_k..]
                        .iter()
                        .find(|&&other| bias[other] == bias[kth])
                        .filter(|_| tie)
                    {
                        row[other] = row[kth];
                    }
                    break row;
                }
                RouterKind::SoftmaxTopK { .. } if tie => {
                    let mut order = (0..experts).collect::<Vec<_>>();
                    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    row[order[top_k]] = row[order[top_k - 1]];
                    break row;
                }
                _ => break row,
            }
        };
        out.extend(row);
    }
    out
}

/// The router's IDs and weights over every token of `logits`.
pub(super) fn route_tokens(
    logits: &[f32],
    bias: &[f32],
    experts: usize,
    top_k: usize,
    kind: RouterKind,
) -> LabResult<(Vec<i32>, OracleOutput)> {
    let bias = bias.iter().map(|&b| f64::from(b)).collect::<Vec<_>>();
    let mut ids = Vec::new();
    let mut weights = Vec::new();
    for row in logits.chunks_exact(experts) {
        let row = row.iter().map(|&l| f64::from(l)).collect::<Vec<_>>();
        let selection =
            oracle::router(&row, Some(&bias), top_k, kind).map_err(|e| e.0.to_string())?;
        ids.extend(selection.expert_ids.iter().map(|&id| id as i32));
        weights.extend(selection.weights);
    }
    // The oracle's bound is 2^-22 K S with K = 1: S carries the weight's
    // relative error in F32 unit roundoffs, a quarter of 2^-22 each.
    let units = weight_roundoffs(kind, experts, top_k);
    let bounds = weights
        .iter()
        .map(|w: &f64| w.abs() * units / 4.0)
        .collect();
    let weights = OracleOutput::new(weights, bounds, 1).map_err(|e| e.0.to_string())?;
    Ok((ids, weights))
}

/// The relative error of a router's F32 weight, in F32 unit roundoffs, from
/// the computation the contract names (a first-order bound; every `expf`
/// within 2 ulp, 4 roundoffs, of an exact argument, since the logits are
/// multiples of 1/32 and their differences exact):
///
/// - sigmoid bias: the selected score `1 / (1 + expf(-v))` (8, as the
///   Sinkhorn oracle's sigmoid), the sum of the `top_k` selected scores
///   (their 8 and `top_k - 1` adds), the `+ 1e-20`, the division and the
///   scaling (one each);
/// - softmax renormalized: each `expf(v - max) / total` (5; the total's
///   error is common to every selected score and cancels in the ratio),
///   their sum (5 and `top_k - 1`), the division (1);
/// - softmax not renormalized: `expf` (4), the total over every expert (4
///   and `experts - 1`), the division (1);
/// - grouped: `expf(v - max)` over the selected logits (4), their sum (4 and
///   `top_k - 1`), the division (1).
fn weight_roundoffs(kind: RouterKind, experts: usize, top_k: usize) -> f64 {
    let k = top_k as f64;
    match kind {
        RouterKind::SigmoidBiasSelectNormalized { .. } => 8.0 + 8.0 + (k - 1.0) + 3.0,
        RouterKind::SoftmaxTopK { renormalize: true } => 5.0 + 5.0 + (k - 1.0) + 1.0,
        RouterKind::SoftmaxTopK { renormalize: false } => 4.0 + 4.0 + (experts as f64 - 1.0) + 1.0,
        RouterKind::GroupLimitedSelectedSoftmax { .. } => 4.0 + 4.0 + (k - 1.0) + 1.0,
    }
}

/// The case of a router over `capacity` tokens: F32 logits and a bias per
/// expert (bound for every kind; only sigmoid-bias routers read it), and
/// the contract's router as exact IDs and weights within tolerance.
pub fn router_case<'a>(
    op: &RowQuery,
    capacity: u32,
    prepared: Prepared<RouterBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RowQuery::Router {
        experts,
        top_k,
        kind,
    } = *op
    else {
        return Err("a router case needs a router".into());
    };
    let (experts, top_k) = (experts as usize, top_k as usize);
    let logits = logits(capacity as usize, experts, top_k, kind, 11);
    let bias = bias(experts);
    let roles = RouterBuffers {
        logits: Spec::input(DType::F32, (Rows::Live, experts), f32_bytes(&logits)),
        bias: Some(Spec::input(
            DType::F32,
            (Rows::Fixed(1), experts),
            f32_bytes(&bias),
        )),
        ids: Spec::output(DType::I32, (Rows::Live, top_k)),
        weights: Spec::output(DType::F32, (Rows::Live, top_k)),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let (ids, weights) = route_tokens(
            &images.f32s("logits")?,
            &images.f32s("bias")?,
            experts,
            top_k,
            kind,
        )?;
        Ok(vec![
            ("ids".into(), Expected::Bits(i32_bytes(&ids))),
            ("weights".into(), Expected::Values(weights)),
        ])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required router `op`.
pub fn router<'a>(name: String, op: RowQuery, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.moe_router(&op, need),
        move |prepared| router_case(&op, need.capacity, prepared),
    )
}

/// A rank's routes: the router, the local routes over its experts and the
/// route tile the expert op reads them in.
#[derive(Clone, Copy, Debug)]
pub struct RouteOp {
    pub router: RowQuery,
    pub routes: MoeQuery,
    pub rank: u32,
    pub route_tile: u32,
    pub usage: OperationUse,
}

/// A role of the route metadata written whole at every live count, zero
/// included (the counts, offsets and padded total).
fn whole<'a>(elements: usize) -> Spec<'a> {
    Spec::output_with(
        DType::I32,
        (Rows::Fixed(1), elements),
        Write::EveryCount(Box::new(move |_, _| {
            WriteMap::of(DType::I32).prefix(elements)
        })),
    )
    .route_metadata()
}

/// The case of rank routes over the use's capacity: the router's case
/// inputs and outputs, and the route metadata as exact bits
/// (`gated_mlp_oracle::local_route_metadata` over the router's IDs, padded
/// to the route tile): the counts, offsets and padded total written whole
/// at every live count, the sorted rows up to the padded total and the
/// active experts' count and IDs. The route tile tables are scratch the op
/// sizes.
pub fn route_case<'a>(
    op: RouteOp,
    prepared: Prepared<RouteBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RowQuery::Router {
        experts,
        top_k,
        kind,
    } = op.router
    else {
        return Err("a route case needs a router".into());
    };
    let MoeQuery::LocalRoutes { local_experts, .. } = op.routes else {
        return Err("a route case needs local routes".into());
    };
    let info = prepared.info().clone();
    if let Some((path, _)) = info.bytes.iter().find(|(path, _)| {
        !matches!(
            path.as_str(),
            "routes.sorted_rows" | "routes.active_experts" | "tile_counts" | "tile_offsets"
        )
    }) {
        return Err(format!("the kit's route case binds no {path}"));
    }
    let capacity = op.usage.capacity;
    let (experts, top_k, local) = (experts as usize, top_k as usize, local_experts as usize);
    let sorted = info
        .bytes_of("routes.sorted_rows")
        .ok_or("the op does not size its sorted rows")? as usize
        / 4;
    let active = info.bytes_of("routes.active_experts").is_some();
    let tiles = |path| {
        info.bytes_of(path)
            .map(|bytes| Spec::scratch(bytes as usize))
    };
    let logits = logits(capacity as usize, experts, top_k, kind, 13);
    let bias = bias(experts);
    // The metadata of the live tokens' routes, from the images.
    let pad = op.route_tile as usize;
    let rank = op.rank as usize;
    let metadata = Rc::new(move |images: &Images, live: u32| {
        let logits = images.f32s("logits")?;
        let (ids, weights) = route_tokens(
            &logits[..live as usize * experts],
            &images.f32s("bias")?,
            experts,
            top_k,
            kind,
        )?;
        let global = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let metadata =
            local_route_metadata(&global, top_k, local, rank, pad).map_err(|e| e.0.to_string())?;
        Ok::<_, String>((ids, weights, metadata))
    });
    let written = Rc::clone(&metadata);
    let sorted_rows = Spec::output_with(
        DType::I32,
        (Rows::Fixed(1), sorted),
        Write::Map(Box::new(move |images, live| {
            let (_, _, metadata) = written(images, live)?;
            WriteMap::of(DType::I32).prefix(metadata.num_padded_rows as usize)
        })),
    )
    // Padding is -1, all ones, which the default fill would read as
    // unwritten.
    .filled_with(&[0x80])
    .route_metadata();
    let written = Rc::clone(&metadata);
    let active_experts = active.then(|| {
        Spec::output_with(
            DType::I32,
            (Rows::Fixed(1), local + 1),
            Write::EveryCount(Box::new(move |images, live| {
                let (_, _, metadata) = written(images, live)?;
                WriteMap::of(DType::I32).prefix(metadata.active_experts.len())
            })),
        )
        .route_metadata()
    });
    let roles = RouteBuffers {
        logits: Spec::input(DType::F32, (Rows::Live, experts), f32_bytes(&logits)),
        bias: Some(Spec::input(
            DType::F32,
            (Rows::Fixed(1), experts),
            f32_bytes(&bias),
        )),
        ids: Spec::output(DType::I32, (Rows::Live, top_k)),
        weights: Spec::output(DType::F32, (Rows::Live, top_k)),
        routes: RouteMetadata {
            counts: Some(whole(local)),
            offsets: whole(local + 1),
            sorted_rows,
            padded_rows: whole(1),
            active_experts,
        },
        tile_counts: tiles("tile_counts"),
        tile_offsets: tiles("tile_offsets"),
        ticket: None,
        live_rows: Spec::live(),
    };
    let rows = capacity as usize;
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let (mut ids, weights, metadata) = metadata(images, live)?;
        ids.resize(rows * top_k, 0);
        let mut weights_all = OracleOutput {
            values: vec![0.0; rows * top_k],
            sum_abs_products: vec![0.0; rows * top_k],
            reduction_lengths: vec![1; rows * top_k],
        };
        weights_all.values[..weights.values.len()].copy_from_slice(&weights.values);
        weights_all.sum_abs_products[..weights.values.len()]
            .copy_from_slice(&weights.sum_abs_products);
        let mut sorted_rows = metadata.sorted_rows.clone();
        sorted_rows.resize(sorted, 0);
        let mut results = vec![
            ("ids".into(), Expected::Bits(i32_bytes(&ids))),
            ("weights".into(), Expected::Values(weights_all)),
            (
                "routes.counts".into(),
                Expected::Bits(i32_bytes(&metadata.counts)),
            ),
            (
                "routes.offsets".into(),
                Expected::Bits(i32_bytes(&metadata.offsets)),
            ),
            (
                "routes.sorted_rows".into(),
                Expected::Bits(i32_bytes(&sorted_rows)),
            ),
            (
                "routes.padded_rows".into(),
                Expected::Bits(i32_bytes(&[metadata.num_padded_rows])),
            ),
        ];
        if active {
            let mut experts = metadata.active_experts.clone();
            experts.resize(local + 1, 0);
            results.push((
                "routes.active_experts".into(),
                Expected::Bits(i32_bytes(&experts)),
            ));
        }
        Ok(results)
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required rank routes `op` at its use's capacity.
pub fn route<'a>(name: String, op: RouteOp) -> Required<'a> {
    Required::new(
        name,
        Need::rows(op.usage.capacity),
        move |backend, need| {
            backend.moe_route(
                &op.router,
                &op.routes,
                op.rank,
                op.route_tile,
                op.usage,
                need,
            )
        },
        move |prepared| route_case(op, prepared),
    )
}

/// The case of a combine of each token's routed rows over `capacity`
/// tokens: BF16 routed rows from the kit's generator and F32 route weights
/// when the combine applies them. Rows weighted before the combine compare
/// as the contract's bits (F32 adds in top-k order, one BF16 rounding); a
/// combine that applies the weights compares within the F64 combine's
/// tolerance over each token's `top_k` products (the contract leaves the
/// product's rounding open).
pub fn expert_combine_case<'a>(
    op: ExpertCombineOp,
    capacity: u32,
    prepared: Prepared<CombineBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (top_k, width, rows) = (op.top_k as usize, op.width as usize, capacity as usize);
    let routed = bf16_values(21, rows * top_k * width);
    let weighs = op.route_weight.is_none();
    let weights = (0..rows * top_k)
        .map(|at| 0.125 + (at % 7) as f32 * 0.0625)
        .collect::<Vec<_>>();
    let roles = CombineBuffers {
        routed: Spec::input(
            DType::Bf16,
            (Rows::Live, top_k * width),
            bf16_bytes(&routed),
        ),
        weights: weighs.then(|| Spec::input(DType::F32, (Rows::Live, top_k), f32_bytes(&weights))),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let routed = images
            .u16s("routed")?
            .into_iter()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        if !weighs {
            // Rows weighted before the combine: the contract's arithmetic,
            // F32 adds in top-k order rounded to BF16 once, as bits.
            let mut output = Vec::with_capacity(rows * width * 2);
            for token in routed.chunks_exact(top_k * width) {
                for col in 0..width {
                    let mut sum = 0.0_f32;
                    for slot in 0..top_k {
                        sum += token[slot * width + col];
                    }
                    output.extend(f32_to_bf16(sum).to_le_bytes());
                }
            }
            return Ok(vec![("output".into(), Expected::Bits(output))]);
        }
        // The combine applies the weights: within the F64 combine's
        // tolerance over each token's own top-k routes.
        let weights = images.f32s("weights")?;
        let mut values = Vec::with_capacity(rows * width);
        let mut bounds = Vec::with_capacity(rows * width);
        for (token, routes) in routed.chunks_exact(top_k * width).enumerate() {
            for col in 0..width {
                let products = (0..top_k).map(|slot| {
                    f64::from(routes[slot * width + col]) * f64::from(weights[token * top_k + slot])
                });
                values.push(products.clone().sum::<f64>());
                bounds.push(products.map(f64::abs).sum::<f64>());
            }
        }
        let output = OracleOutput::new(values, bounds, top_k).map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Values(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required combine `op`.
pub fn expert_combine<'a>(name: String, op: ExpertCombineOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.moe_expert_combine(&op, need),
        move |prepared| expert_combine_case(op, need.capacity, prepared),
    )
}

/// `rows` tokens of `top_k` distinct global expert IDs over `experts`:
/// each token routes to some of `rank`'s experts and some of others',
/// and the last token to none of the rank's.
fn global_ids(rows: usize, experts: usize, local: usize, top_k: usize, rank: usize) -> Vec<i32> {
    let mut next = stream(17);
    let mine = rank * local..(rank + 1) * local;
    // With every expert the rank's, every route is local.
    let all_mine = local == experts;
    let mut ids = Vec::with_capacity(rows * top_k);
    for token in 0..rows {
        let mut chosen = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            // Even slots route to the rank's experts, odd ones and the last
            // token's to others'.
            let local_slot = token + 1 < rows && chosen.len() % 2 == 0;
            let id = if local_slot {
                mine.start + (next() % local as u64) as usize
            } else {
                (next() % experts as u64) as usize
            };
            if !chosen.contains(&id) && (local_slot || all_mine || !mine.contains(&id)) {
                chosen.push(id);
            }
        }
        ids.extend(chosen.iter().map(|&id| id as i32));
    }
    ids
}

/// The case of a rank's routed sum over `capacity` tokens: F32 routed rows
/// from the kit's generator, the tokens' global expert IDs (some local to
/// the rank, the last token's none, unless every expert is the rank's),
/// with `addend` an F32 addend row per
/// token, and the contract's sum as exact bits (`oracle::local_routed_sum`):
/// the local routes added in top-k order in F32 from +0, then the addend
/// once, then written as F32 or rounded to BF16 once.
pub fn routed_sum_case<'a>(
    op: &MoeQuery,
    rank: u32,
    capacity: u32,
    addend: bool,
    prepared: Prepared<RoutedSumBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let MoeQuery::LocalRoutedSum {
        experts,
        local_experts,
        top_k,
        width,
        output,
        ..
    } = *op
    else {
        return Err("a routed sum case needs a local routed sum".into());
    };
    let (top_k, width, rows) = (top_k as usize, width as usize, capacity as usize);
    let local = local_experts as usize;
    let routed = bf16_values(23, rows * top_k * width)
        .into_iter()
        .enumerate()
        // F32 values with low bits set, so an add's rounding shows.
        .map(|(at, b)| bf16_to_f32(b) * (1.0 + (at % 13) as f32 / 1024.0))
        .collect::<Vec<_>>();
    let ids = global_ids(rows, experts as usize, local, top_k, rank as usize);
    // Addend values of other magnitudes than the routes', so the add
    // rounds.
    let addends = bf16_values(29, rows * width)
        .into_iter()
        .enumerate()
        .map(|(at, b)| bf16_to_f32(b) * (3.0 + (at % 7) as f32 / 512.0))
        .collect::<Vec<_>>();
    let roles = RoutedSumBuffers {
        routed: Spec::input(DType::F32, (Rows::Live, top_k * width), f32_bytes(&routed)),
        ids: Spec::input(DType::I32, (Rows::Live, top_k), i32_bytes(&ids)),
        addend: addend.then(|| Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&addends))),
        output: Spec::output(output, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let routed = images.f32s("routed")?;
        let ids = images.i32s("ids")?;
        let addend = addend.then(|| images.f32s("addend")).transpose()?;
        let sums = oracle::local_routed_sum(
            &routed,
            &ids,
            top_k,
            width,
            local,
            rank as usize,
            addend.as_deref(),
        )
        .map_err(|e| e.0.to_string())?;
        let bytes = match output {
            DType::Bf16 => sums
                .iter()
                .flat_map(|&sum| f32_to_bf16(sum).to_le_bytes())
                .collect(),
            _ => f32_bytes(&sums),
        };
        Ok(vec![("output".into(), Expected::Bits(bytes))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required routed sum `op` of rank `rank`, with an addend bound when
/// `addend`.
pub fn routed_sum<'a>(
    name: String,
    op: MoeQuery,
    rank: u32,
    addend: bool,
    need: Need,
) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.moe_routed_sum(&op, rank, need),
        move |prepared| routed_sum_case(&op, rank, need.capacity, addend, prepared),
    )
}

/// The case of a dense gated MLP projection over `capacity` tokens: BF16
/// token rows and a weight of the projection's format in the checkpoint's
/// row-major layout the op reports (BF16, or FP8 128 x 128 blocks with F32
/// scales, from the projection case's encoder), and the contract's F64
/// reference within its tolerance: for gate/up the clamped SwiGLU of the
/// concatenated halves (`gated_mlp_oracle::dense_gated_mlp`, the engine's
/// rounding), for down the plain projection.
pub fn dense_experts_case<'a>(
    op: GatedMlpProjectionQuery,
    capacity: u32,
    prepared: Prepared<GatedMlpBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let gemm = op.gemm;
    if gemm.grouped.is_some() {
        return Err("the kit's expert case covers dense projections".into());
    }
    let info = prepared.info().clone();
    match info.layout_of("weights") {
        Some(layout) if *layout != kernel_core::precision::WeightLayout::row_major() => {
            return Err(format!(
                "the kit has no packer for expert weight layout {} v{}",
                layout.id, layout.version
            ));
        }
        _ => {}
    }
    if let Some((path, _)) = info.bytes.iter().find(|(path, _)| {
        !matches!(
            path.as_str(),
            "activation" | "live_tokens" | "output" | "weights.codes" | "weights.local_scales"
        )
    }) {
        return Err(format!("the kit's dense expert case binds no {path}"));
    }
    let (n, k, rows) = (gemm.n as usize, gemm.k as usize, capacity as usize);
    let gate_up = op.projection == GatedMlpProjection::GateUp;
    let columns = if gate_up { n / 2 } else { n };
    let encoded = encode_weight(n, k, gemm.weight)?;
    let roles = GatedMlpBuffers {
        activation: Spec::input(
            DType::Bf16,
            (Rows::Live, k),
            bf16_bytes(&bf16_values(25, rows * k)),
        ),
        activation_scales: None,
        activation_global: None,
        live_tokens: Spec::live(),
        output: Spec::output(DType::Bf16, (Rows::Live, columns)),
        weights: QuantizedWeights {
            codes: Spec::weight(encoded.codes),
            local_scales: encoded.scales.map(Spec::weight),
            global_scales: None,
        },
        routing: None,
        scratch: GatedMlpScratch {
            partial: None,
            output_codes: None,
            output_scales: None,
            unused: Spec::untouched(16),
        },
    };
    let weights = encoded.decoded;
    // Every live count's reference is the capacity's (the write map selects
    // the live rows), so the oracle keeps it for the activation it last saw.
    let kept = RefCell::new(None::<(Vec<u16>, OracleOutput)>);
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let bits = images.u16s("activation")?;
        if let Some((seen, output)) = kept.borrow().as_ref()
            && *seen == bits
        {
            return Ok(vec![("output".into(), Expected::Values(output.clone()))]);
        }
        let x = bits
            .iter()
            .map(|&b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>();
        let output = if gate_up {
            let intermediate = n / 2;
            let down = vec![0.0; k * intermediate];
            let mlp = dense_gated_mlp(
                &x,
                rows,
                ExpertWeights {
                    gate_up: &weights,
                    down: &down,
                    experts: 1,
                    hidden: k,
                    intermediate,
                },
                clamped_silu_glu(op.swiglu_clamp, None),
                ENGINE_ROUNDING,
            )
            .map_err(|e| e.0.to_string())?;
            mlp.intermediate
        } else {
            let shape = GemmShape {
                m: rows,
                n,
                k,
                experts: 1,
            };
            let epilogue = GemmEpilogue {
                residual: None,
                swiglu: None,
            };
            oracle::gemm(&x, &weights, shape, &[], epilogue).map_err(|e| e.0.to_string())?
        };
        *kept.borrow_mut() = Some((bits, output.clone()));
        Ok(vec![("output".into(), Expected::Values(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required dense gated MLP projection of the chained engine's
/// `projection` at a step of `phase`.
pub fn dense_experts<'a>(
    name: String,
    projection: &Projection,
    phase: ExecutionPhase,
    need: Need,
) -> Required<'a> {
    let op = projection.query;
    let usage = OperationUse {
        phase,
        capacity: need.capacity,
        graph: false,
    };
    Required::new(
        name,
        need,
        move |backend, need| {
            backend
                .gated_mlp_projection(&op, usage, need)
                .map(|op| -> Prepared<GatedMlpBuffers<View>> { op })
        },
        move |prepared| dense_experts_case(op, need.capacity, prepared),
    )
}

/// The chained engine's dense projections at its smallest dense
/// intermediate, gate/up and down, in each dense weight format, at the
/// token capacity of the decode (`prefill` false) or prefill row tile.
fn dense_projections(prefill: bool) -> Vec<Projection> {
    let shapes = chained::TP4_288_TOP8;
    let smallest = shapes.dense_intermediates.into_iter().min().unwrap_or(0);
    let tokens = if prefill {
        shapes.prefill_tokens
    } else {
        shapes.decode_tokens
    };
    chained::catalog()
        .into_iter()
        .filter(|projection| {
            let gemm = projection.query.gemm;
            let intermediate = match projection.query.projection {
                GatedMlpProjection::GateUp => gemm.n / 2,
                GatedMlpProjection::Down => gemm.k,
            };
            gemm.grouped.is_none()
                && intermediate == smallest
                && projection.tokens_capacity == tokens
        })
        .collect()
}

/// The chained engine's served schedules of its cataloged dense contracts,
/// each form as the lab's cubins hold it: the 16-row decode forms in block
/// warps (FP8 and BF16); the 64-row prefill forms in columns and the FP8
/// gate/up also in decoded tiles. SM120 picks among a contract's forms per
/// row count (`kernels_sm120::backend`'s chained table). The routed
/// experts' forms are the split's (`chained::split::forms`).
pub(super) fn chained_schedules() -> Vec<Projection> {
    let mut forms = Vec::new();
    for projection in chained::catalog() {
        forms.push(projection.block_warps().unwrap_or(projection.clone()));
        forms.extend(
            projection
                .decoded_tiles()
                .ok()
                .filter(|_| projection.row_tile == 64),
        );
    }
    forms
}

/// The row-engine routers and combines of `catalogs`: the first entry of
/// each router query and of each combine (query, width and weighting),
/// since equal ones under two symbols are an SM120 tie.
fn catalog_rows(catalogs: &kernels_sm120::Catalogs) -> Vec<RowInstantiation> {
    let mut chosen: Vec<RowInstantiation> = Vec::new();
    for row in kernels_sm120::row::catalog(catalogs) {
        let tie = |other: &RowInstantiation| match row.query {
            RowQuery::Router { .. } => other.query == row.query,
            RowQuery::ExpertCombine { .. } => {
                other.query == row.query && other.width == row.width && other.body == row.body
            }
            _ => true,
        };
        if matches!(
            row.query,
            RowQuery::Router { .. } | RowQuery::ExpertCombine { .. }
        ) && !chosen.iter().any(tie)
        {
            chosen.push(row);
        }
    }
    chosen
}

/// Whether a combine entry adds rows weighted before it.
fn preweighted(entry: &RowInstantiation) -> bool {
    entry.body.contains(",true>(")
}

/// SM120's MoE ops over `catalogs`: their row-engine routers and
/// combines, the chained engine's served dense schedules
/// ([`chained_schedules`]), and its routed experts split over the ranks
/// (`chained::split`): every served form, the local routes, the routed sums
/// in both output forms and the F32 sum that adds a row, admitted on
/// `resources`, which ptxas reported for the cubins of the decoded-tile
/// forms and the split instances, as `kernels_sm120::backend::locked_resources`
/// reads them from their locks.
pub fn sm120(
    catalogs: &kernels_sm120::Catalogs,
    resources: Vec<(String, EntryResources)>,
) -> Sm120 {
    let mut backend = Sm120::new();
    for row in catalog_rows(catalogs) {
        backend = match row.query {
            RowQuery::Router { .. } => backend.with_router(row),
            _ => backend.with_expert_combine(row),
        };
    }
    for projection in chained_schedules() {
        backend = backend.with_chained_projection(projection);
    }
    for projection in split::forms() {
        backend = backend.with_chained_projection(projection);
    }
    backend = backend.with_entry_resources(resources);
    for route in split::routes() {
        backend = backend.with_local_route(route, split::PREFIX);
    }
    for sum in split::sums() {
        for output in [RoutedSumOutput::Bf16, RoutedSumOutput::F32] {
            backend = backend.with_routed_sum(sum.with_output(output), split::PREFIX);
        }
        let sum = sum.with_output(RoutedSumOutput::F32);
        backend = backend.with_routed_sum_add(sum, split::PREFIX);
    }
    backend
}

/// The chained engine's route and route sum ops of `shapes` at `rank`: its
/// decode and prefill routes, and its routed sum in both output forms and
/// in F32 with an addend.
fn chained_routes<'a>(shapes: ServedShapes, rank: u32, need: Need) -> Vec<Required<'a>> {
    let mut ops = Vec::new();
    let router = RowQuery::Router {
        experts: shapes.experts,
        top_k: shapes.top_k,
        kind: RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        },
    };
    for (route, phase, capacity) in [
        (
            shapes.routes()[0],
            ExecutionPhase::Decode,
            shapes.decode_tokens,
        ),
        (shapes.routes()[1], ExecutionPhase::Prefill, 64),
    ] {
        let op = RouteOp {
            router,
            routes: MoeQuery::LocalRoutes {
                experts: shapes.experts,
                local_experts: shapes.local_experts,
                top_k: shapes.top_k,
                capacity: route.capacity,
            },
            rank,
            route_tile: route.pad,
            usage: OperationUse {
                phase,
                capacity,
                graph: false,
            },
        };
        ops.push(
            self::route(
                format!(
                    "routes {} experts {} local top {} tile {} capacity {capacity}",
                    shapes.experts, shapes.local_experts, shapes.top_k, route.pad
                ),
                op,
            )
            .required(),
        );
    }
    let [sum] = shapes.sums();
    for (output, addend) in [
        (DType::Bf16, false),
        (DType::F32, false),
        (DType::F32, true),
    ] {
        let op = MoeQuery::LocalRoutedSum {
            experts: shapes.experts,
            local_experts: sum.local_experts,
            top_k: sum.top_k,
            width: sum.width,
            output,
            capacity: sum.capacity,
        };
        ops.push(
            routed_sum(
                format!(
                    "routed sum {} experts {} local top {} width {} {output:?}{}",
                    shapes.experts,
                    sum.local_experts,
                    sum.top_k,
                    sum.width,
                    if addend { " plus addend" } else { "" }
                ),
                op,
                rank,
                addend,
                need,
            )
            .required(),
        );
    }
    ops
}

/// The MoE ops of the lab's catalogs and the chained engine's served
/// shapes: every catalog router and combine at [`CAPACITY`] tokens, the
/// chained engine's decode and prefill routes of the split (rank
/// [`SPLIT_RANK`], sigmoid bias, scaled by [`SCALE`]), its routed sum in
/// both output forms and in F32 with an addend, and its dense projections
/// in each schedule: the
/// decode ones at [`CAPACITY`] tokens (block warps), and at a prefill of
/// [`PREFILL_CAPACITY`] tokens the gate/up (FP8 in decoded tiles, BF16 in
/// columns) and down (columns), and the FP8 gate/up at a verify of as
/// many (columns).
/// Every op SM120 serves from the lab's cubins is marked required.
pub fn required<'a>(catalogs: &kernels_sm120::Catalogs) -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mut ops = Vec::new();
    for row in catalog_rows(catalogs) {
        match row.query {
            RowQuery::Router {
                experts,
                top_k,
                kind,
            } => {
                ops.push(
                    router(
                        format!("router {kind:?} {experts} experts top {top_k}"),
                        row.query,
                        need,
                    )
                    .required(),
                );
            }
            RowQuery::ExpertCombine { experts, top_k } => {
                let op = ExpertCombineOp {
                    experts,
                    top_k,
                    width: row.width,
                    route_weight: preweighted(&row).then_some(RouteWeightPoint::AfterDown),
                };
                ops.push(
                    expert_combine(
                        format!(
                            "expert combine {experts} experts top {top_k} width {}{}",
                            row.width,
                            if preweighted(&row) {
                                " preweighted"
                            } else {
                                ""
                            }
                        ),
                        op,
                        need,
                    )
                    .required(),
                );
            }
            _ => {}
        }
    }
    ops.extend(chained_routes(split::SHAPES, SPLIT_RANK, need));
    let prefill = Need::rows(PREFILL_CAPACITY);
    let dense = dense_projections(false)
        .into_iter()
        .map(|projection| (projection, ExecutionPhase::Decode, need))
        .chain(
            dense_projections(true)
                .into_iter()
                .flat_map(|projection| {
                    let fp8_gate_up = projection.gate_up()
                        && projection.weights() == chained::Weights::Fp8Block128;
                    let verify = fp8_gate_up.then(|| (projection.clone(), ExecutionPhase::Verify));
                    [Some((projection, ExecutionPhase::Prefill)), verify]
                })
                .flatten()
                .filter(|(projection, _)| {
                    projection.gate_up() || projection.weights() == chained::Weights::Fp8Block128
                })
                .map(|(projection, phase)| (projection, phase, prefill)),
        );
    for (projection, phase, need) in dense {
        let gemm = projection.query.gemm;
        ops.push(
            dense_experts(
                format!(
                    "dense {:?} {:?} {}x{} {phase:?} {}",
                    projection.query.projection, gemm.weight, gemm.n, gemm.k, need.capacity
                ),
                &projection,
                phase,
                need,
            )
            .required(),
        );
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_logits_tie_softmax_tokens_and_separate_sigmoid_keys() {
        let softmax = RouterKind::SoftmaxTopK { renormalize: true };
        let rows = logits(4, 128, 8, softmax, 1);
        let ids = |row: &[f32]| {
            let row = row.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
            oracle::router(&row, None, 8, softmax).unwrap().expert_ids
        };
        // Odd tokens tie the expert after the selection to the k-th.
        let row = &rows[128..256];
        let mut sorted = row.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        assert_eq!(sorted[7], sorted[8]);
        assert_eq!(ids(row).len(), 8);
        let sigmoid = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        };
        let rows = logits(8, 288, 8, sigmoid, 2);
        for (token, row) in rows.chunks_exact(288).enumerate() {
            let keys = sigmoid_keys(row, &bias(288));
            let mut sorted = keys.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            // Odd tokens tie the k-th key exactly; the others stay apart.
            assert_eq!(sorted[7] == sorted[8], token % 2 == 1, "{token}");
            if token % 2 == 0 {
                assert!(separated(&keys, 8));
            }
        }
    }

    /// Router weights pass at their F32 rounding and fail scaled by
    /// 1.00003, for every router kind the cases run.
    #[test]
    fn weight_bounds_follow_the_selected_scores() {
        for (kind, experts) in [
            (
                RouterKind::SigmoidBiasSelectNormalized {
                    scaling_factor: SCALE,
                },
                288,
            ),
            (RouterKind::SoftmaxTopK { renormalize: true }, 128),
            (
                RouterKind::GroupLimitedSelectedSoftmax {
                    n_group: 8,
                    topk_group: 4,
                },
                128,
            ),
        ] {
            let rows = logits(16, experts, 8, kind, 13);
            let (_, weights) = route_tokens(&rows, &bias(experts), experts, 8, kind).unwrap();
            let actual = |scale: f64| {
                weights
                    .values
                    .iter()
                    .map(|&w| f64::from((w * scale) as f32))
                    .collect::<Vec<_>>()
            };
            assert!(oracle::check_numeric(&actual(1.0), &weights, DType::F32).is_ok());
            assert!(
                oracle::check_numeric(&actual(1.00003), &weights, DType::F32).is_err(),
                "{kind:?}"
            );
        }
    }

    /// The selection bias moves experts: a router that ignores it selects
    /// other IDs for some token of the route case's logits.
    #[test]
    fn the_bias_changes_the_selection() {
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        };
        let rows = logits(16, 288, 8, kind, 13);
        let (biased, _) = route_tokens(&rows, &bias(288), 288, 8, kind).unwrap();
        let (unbiased, _) = route_tokens(&rows, &[0.0; 288], 288, 8, kind).unwrap();
        assert_ne!(biased, unbiased);
        // The route metadata follows the rank and the route tile.
        let ids = biased.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let metadata = |rank, pad| local_route_metadata(&ids, 8, 72, rank, pad).unwrap();
        assert_ne!(metadata(1, 16), metadata(0, 16));
        assert_ne!(metadata(1, 16), metadata(1, 64));
    }

    /// With every expert the rank's, every token takes eight distinct
    /// experts, the last token's included.
    #[test]
    fn global_ids_with_every_expert_local_route_anywhere() {
        let ids = global_ids(4, 288, 288, 8, 0);
        for row in ids.chunks_exact(8) {
            let mut distinct = row.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), 8);
            assert!(row.iter().all(|&id| (0..288).contains(&id)));
        }
    }

    /// Each token's routes to rank 1 differ from those to rank 0, so a sum
    /// that ignores the rank, or the IDs, adds other routes.
    #[test]
    fn global_ids_mix_local_and_foreign_experts() {
        let ids = global_ids(4, 288, 72, 8, 1);
        let on = |rank: i32| ids.iter().map(|&id| id / 72 == rank).collect::<Vec<_>>();
        assert_ne!(on(1), on(0));
        for (token, row) in ids.chunks_exact(8).enumerate() {
            let local = row.iter().filter(|&&id| (72..144).contains(&id)).count();
            let mut distinct = row.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), 8);
            if token == 3 {
                assert_eq!(local, 0);
            } else {
                assert!(local > 0 && local < 8, "{row:?}");
            }
        }
    }
}
