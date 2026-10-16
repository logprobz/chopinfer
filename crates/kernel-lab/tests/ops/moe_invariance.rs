//! The grouped expert invariance adapter. Prefill entries run as the routed
//! chain they belong to, from given hidden rows and router logits: MiMo's
//! row router, tiled permutation, gate/up and down experts and combine
//! (`FullCase`), and Qwen3.8's router, permutation, experts and shared-gate
//! combine (`qwen38::invariance::Chains`); a unit is a token's combined
//! output row, so the permutation's order is free and only the rows' bits
//! count. MiMo's decode and exact expert entries run tokens with their own
//! routes, inputs spread over thirteen binades and random MXFP4 experts
//! (`exact_rows`); a unit is a token's eight expert rows, alone through the
//! bucket-1 entry. The fused router units are a token's logits, experts and
//! route weights.
//!
//! Qwen3.8's decode entries of the exact verify buckets run the decode
//! chain of their bucket (decode route, gate/up, down and combine) over
//! tokens seeded with their hidden row, router logits and shared-expert
//! row; the logits favour a few of the rank's experts, so every token has
//! local routes and tokens share experts. A unit is the token's expert IDs
//! and route weights (decode route), its local routes' intermediate rows
//! (gate/up) or weighted down rows (down); alone, a token runs the bucket-1
//! chain. The combine's rows run the combine alone over given routed rows
//! and expert IDs, alone at one row.
//!
//! Ties: the Qwen3.8 router and MiMo's fused router keep the lower expert
//! among equal scores.
//!
//! Every compared launch starts with poisoned outputs and is checked
//! afterwards against explicit write maps (`ids_written`, `route_written`,
//! `experts_written`, `rows_written`, `fused_written`): the router's logits,
//! expert IDs and weights of the live routes, the permutation's counts,
//! offsets, route slots, padded total and active list, the expert rows of
//! the routes to the rank's experts only, every live token's combined row,
//! and the fused router's split partials are written (finite where they are
//! numbers); every other byte, padding and rows past the live ones, keeps
//! the poison. Semaphores, the stream-decode split flags and inputs are
//! never poisoned.

use super::exact_rows::{Projected, mix, random_weights, routes, spread};
use super::qwen38::invariance::{Chains, Part};
use super::*;
use crate::invariance::{Adapter, Compared, Element, Units, written};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernels_sm120::invariance::{Declared, OpKind};
use model_qwen38_contract::sm120::moe as qwen;
use std::ops::Range;

/// Experts a decode token's routes draw from, so tokens share experts.
const POOL: usize = 20;
/// Experts of the Qwen3.8 chains' rank a decode token's logits favour.
const QWEN_POOL: usize = 6;
/// Experts of MiMo's fused router.
const FUSED_EXPERTS: usize = 256;

/// Where a routed chain's launch writes, for its write maps.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    /// Routes per token.
    pub(crate) top_k: usize,
    /// The global IDs of the experts the rank runs; the rank writes no
    /// expert row for a route to another expert.
    pub(crate) experts: Range<usize>,
    /// Rows each local expert's sorted range is padded to.
    pub(crate) block: usize,
    /// 256-route tiles whose expert counts a tiled permutation's count
    /// stage writes; `None` without a tiled permutation.
    pub(crate) tiles: Option<usize>,
    /// Whether the route writes the active expert list (else the launch
    /// reads a given one).
    pub(crate) active: bool,
    /// Elements of a gate/up (intermediate) row.
    pub(crate) inter: usize,
    /// Elements of a weighted down or combined (hidden) row.
    pub(crate) hidden: usize,
}

impl Layout {
    /// The rank's local index of global expert `id`, if the rank runs it.
    pub(crate) fn local(&self, id: i32) -> Option<usize> {
        usize::try_from(id)
            .ok()
            .filter(|id| self.experts.contains(id))
            .map(|id| id - self.experts.start)
    }
}

/// Every byte of the route buffers a chain launch writes, read after it.
#[derive(Clone)]
pub(crate) struct Route {
    /// Expert IDs `[token][top_k]`.
    pub(crate) ids: Vec<u8>,
    /// Route weights `[token][top_k]`, F32.
    pub(crate) weights: Vec<u8>,
    /// Routes per local expert.
    pub(crate) counts: Vec<u8>,
    /// Each local expert's first sorted row, then the padded total.
    pub(crate) offsets: Vec<u8>,
    /// Route indices by expert, each expert's range padded with -1.
    pub(crate) sorted: Vec<u8>,
    /// The padded total.
    pub(crate) padded: Vec<u8>,
    /// The active expert list (its length, then the experts with routes),
    /// where the route writes it.
    pub(crate) active: Option<Vec<u8>>,
    /// Per 256-route tile expert counts and offsets of a tiled permutation.
    pub(crate) tiles: Option<(Vec<u8>, Vec<u8>)>,
}

/// Every byte of the expert and combine outputs a chain launch writes.
pub(crate) struct Experts {
    /// Gate/up rows `[route][inter]`.
    pub(crate) inter: Vec<u8>,
    /// Weighted down rows `[route][hidden]`.
    pub(crate) routed: Vec<u8>,
    /// Combined rows `[token][hidden]`.
    pub(crate) output: Vec<u8>,
}

fn i32s(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|v| i32::from_ne_bytes([v[0], v[1], v[2], v[3]]))
        .collect()
}

/// `written`, or where the map is empty (no route to the rank's experts),
/// every byte still the poison.
fn written_or_untouched(
    label: &str,
    bytes: &[u8],
    map: &WriteMap,
    element: Element,
) -> Result<(), String> {
    if map.written_bytes() == 0 {
        poison::check(label, bytes, map, Fill::Pattern(&[POISON]))
    } else {
        written(label, bytes, map, element)
    }
}

/// Checks a router's expert IDs (`Raw(4)`) and F32 route weights of
/// `routes` live routes, from the first: written, the rest untouched.
/// Returns the IDs.
pub(crate) fn ids_written(ids: &[u8], weights: &[u8], routes: usize) -> Result<Vec<i32>, String> {
    let map = WriteMap::of(DType::I32).prefix(routes)?;
    written("expert IDs", ids, &map, Element::Raw(4))?;
    written("route weights", weights, &map, Element::F32)?;
    Ok(i32s(&ids[..routes * 4]))
}

/// Checks the route a chain launch over `tokens` tokens wrote on the
/// poison, as qwen.cuh (`router`, `route_decode`, `permute_*`), row/router.cuh
/// (`router_sigmoid_warp`, `permute_tile_*`) and gemm/fragment_router.cuh
/// (`fused_route_tail`, `route_metadata_256`) write it: each live route's
/// expert ID and weight; every local expert's route count and offset and
/// the padded total; each local route's slot at the start of its expert's
/// sorted range; where the route writes it, the active list's length and
/// its experts with routes; with a tiled permutation, the counts of the
/// tiles the count stage runs and the offsets of the live tiles. Sorted
/// padding slots must visibly store -1 over a contrasting 0x80 fill.
/// Slots past the padded total must keep that fill. Returns the IDs.
pub(crate) fn route_written(
    route: &Route,
    layout: &Layout,
    tokens: usize,
) -> Result<Vec<i32>, String> {
    let routes = tokens * layout.top_k;
    let ids = ids_written(&route.ids, &route.weights, routes)?;
    let local = layout.experts.len();
    let mut counts = vec![0_usize; local];
    for &id in &ids {
        if let Some(expert) = layout.local(id) {
            counts[expert] += 1;
        }
    }
    let raw = Element::Raw(4);
    written(
        "expert counts",
        &route.counts,
        &WriteMap::of(DType::I32).prefix(local)?,
        raw,
    )?;
    written(
        "expert offsets",
        &route.offsets,
        &WriteMap::of(DType::I32).prefix(local + 1)?,
        raw,
    )?;
    written(
        "padded rows",
        &route.padded,
        &WriteMap::of(DType::I32).prefix(1)?,
        raw,
    )?;
    let padded = counts
        .iter()
        .map(|&count| count.div_ceil(layout.block) * layout.block)
        .sum::<usize>();
    poison::check(
        "sorted rows including padding",
        &route.sorted,
        &WriteMap::of(DType::I32).prefix(padded)?,
        Fill::Pattern(&[0x80]),
    )?;
    let sorted = i32s(&route.sorted);
    let mut first = 0;
    for &count in &counts {
        let end = first + count.div_ceil(layout.block) * layout.block;
        if sorted[first..first + count]
            .iter()
            .any(|&row| row < 0 || row as usize >= routes)
        {
            return Err("sorted route slot is missing or invalid".into());
        }
        if sorted[first + count..end].iter().any(|&row| row != -1) {
            return Err("sorted padding was not written as -1".into());
        }
        first = end;
    }
    match (&route.active, layout.active) {
        (Some(active), true) => {
            let listed = 1 + counts.iter().filter(|&&count| count > 0).count();
            written(
                "active experts",
                active,
                &WriteMap::of(DType::I32).prefix(listed)?,
                raw,
            )?;
        }
        (None, false) => {}
        _ => return Err("the active list read does not match the layout".into()),
    }
    match (&route.tiles, layout.tiles) {
        (Some((tile_counts, tile_offsets)), Some(tiles)) => {
            let live = routes.div_ceil(256);
            written(
                "tile counts",
                tile_counts,
                &WriteMap::of(DType::I32).prefix(tiles * local)?,
                raw,
            )?;
            written(
                "tile offsets",
                tile_offsets,
                &WriteMap::of(DType::I32).prefix(live * local)?,
                raw,
            )?;
        }
        (None, None) => {}
        _ => return Err("the tiles read do not match the layout".into()),
    }
    Ok(ids)
}

/// Checks `rows` rows of `width` BF16 elements from the first: written and
/// finite; every later byte untouched.
pub(crate) fn rows_written(
    label: &str,
    bytes: &[u8],
    rows: usize,
    width: usize,
) -> Result<(), String> {
    written(
        label,
        bytes,
        &WriteMap::of(DType::Bf16).prefix(rows * width)?,
        Element::Bf16,
    )
}

/// Checks a chain launch's expert and combine outputs for the live routes'
/// expert IDs `ids`: the gate/up and weighted down rows of the routes to the
/// rank's experts only, written and finite, every other route's rows and
/// the rows past the live routes untouched (the grouped GEMMs write a row
/// per routed, not padded, sorted slot); and each live token's combined row.
pub(crate) fn experts_written(
    experts: &Experts,
    layout: &Layout,
    ids: &[i32],
) -> Result<(), String> {
    let local = (0..ids.len()).filter(|&route| layout.local(ids[route]).is_some());
    for (label, bytes, width) in [
        ("gate/up rows", &experts.inter, layout.inter),
        ("down rows", &experts.routed, layout.hidden),
    ] {
        let map = WriteMap::of(DType::Bf16).rows(width, local.clone())?;
        written_or_untouched(label, bytes, &map, Element::Bf16)?;
    }
    rows_written(
        "combined rows",
        &experts.output,
        ids.len() / layout.top_k,
        layout.hidden,
    )
}

/// Poisons the partial workspace of `v`'s launch over `counts` where the
/// launch writes it before reading it, as mimo_layers.rs `poison_decode`
/// does: every split partial a reduction kernel reads. An entry without a
/// workspace has none to poison; any other workspace is refused rather than
/// guessed.
fn poison_partials(
    session: &Session,
    v: &Variant,
    counts: RuntimeCounts,
    partials: &DeviceBuffer,
) -> LabResult<()> {
    let bytes = partials.byte_len();
    let range = if v.reduction_plan(counts).map_err(err)?.is_some() {
        0..bytes
    } else if v.plan(counts).map_err(err)?.workspace_bytes == 0 {
        return Ok(());
    } else {
        return Err(format!(
            "{}: a partial workspace of unknown layout",
            v.symbol
        ));
    };
    poison::fill(session, partials, &[range], &[POISON])
}

/// The route layout of MiMo's fused router (gemm/fragment_router.cuh
/// `fused_route_tail`): eight routes per token over all 256 experts, 16-row
/// expert ranges and the active list. It writes no expert rows.
fn fused_layout() -> Layout {
    Layout {
        top_k: TOP_K,
        experts: 0..FUSED_EXPERTS,
        block: 16,
        tiles: None,
        active: true,
        inter: 0,
        hidden: 0,
    }
}

/// Checks what MiMo's fused router of `bucket` tokens wrote over `tokens`
/// live tokens on the poison (gemm/fragment_router.cuh): each live token's
/// 256 F32 logits; in each of the 64 CTAs' partials (16 router strips by
/// 4 K splits, 16 rows of a `max(bucket, 8)`-token tile each) the live
/// tokens' columns, which the reducing CTA reads; and the route
/// (`route_written` on `fused_layout`). Returns the expert IDs.
fn fused_written(
    logits: &[u8],
    partials: &[u8],
    route: &Route,
    bucket: u32,
    tokens: usize,
) -> Result<Vec<i32>, String> {
    written(
        "router logits",
        logits,
        &WriteMap::of(DType::F32).prefix(tokens * FUSED_EXPERTS)?,
        Element::F32,
    )?;
    let tile = bucket.max(8) as usize;
    let rows = FUSED_EXPERTS / 16 * 4 * 16;
    let map = WriteMap::of(DType::F32)
        .byte_ranges((0..rows).map(|row| row * tile * 4..(row * tile + tokens) * 4))?;
    written("router partials", partials, &map, Element::F32)?;
    route_written(route, &fused_layout(), tokens)
}

/// The layout of MiMo's routed prefill chain over `experts` experts (every
/// one on each rank) with gate/up rows of `inter`, at row bucket
/// `capacity`: the row router, then the tiled permutation of row/router.cuh
/// (`permute_tile_count<256,8,256>` writes the counts of every launched
/// tile, `permute_tile_scan<256,64>` pads to 64 rows and writes no active
/// list; the chain gives one).
fn mimo_chain_layout(experts: usize, inter: usize, capacity: usize) -> Layout {
    Layout {
        top_k: TOP_K,
        experts: 0..experts,
        block: 64,
        tiles: Some((capacity * TOP_K).div_ceil(256)),
        active: false,
        inter,
        hidden: HIDDEN,
    }
}

/// What an entry runs as.
enum Target {
    /// A MiMo decode or exact expert entry.
    Decode(Box<Variant>),
    /// MiMo's fused router of a token bucket.
    Fused(u32),
    /// The routed prefill chain of a MiMo profile.
    Mimo(Profile),
    /// The routed prefill chain of a Qwen3.8 profile.
    Qwen(Profile),
    /// A Qwen3.8 decode entry: its profile's decode chain of its bucket,
    /// and the part of a token's outputs the entry writes.
    QwenDecode(Profile, u32, Part),
    /// The Qwen3.8 shared-expert gate combine of a profile.
    QwenCombine(Profile),
}

fn target(symbol: &str) -> LabResult<Target> {
    if let Some(v) = moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| v.symbol == symbol)
    {
        let qwen = v.profile.model() == Some(model_qwen38_contract::sm120::config::MODEL);
        return Ok(match (v.phase, qwen) {
            (Phase::Decode, true) => {
                let part = match v.projection {
                    Projection::GateUp => Part::GateUp,
                    Projection::Down => Part::Down,
                };
                Target::QwenDecode(v.profile, v.tokens_capacity, part)
            }
            (Phase::Decode, false) => Target::Decode(Box::new(v)),
            (Phase::Prefill, true) => Target::Qwen(v.profile),
            (Phase::Prefill, false) => Target::Mimo(v.profile),
        });
    }
    if let Some(tokens) = symbol
        .strip_prefix(&format!(
            "{}_t",
            model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX
        ))
        .and_then(|rest| rest.strip_suffix("_v1"))
    {
        return tokens.parse().map(Target::Fused).map_err(err);
    }
    if let Some(entry) = qwen::route_catalog()
        .into_iter()
        .find(|e| e.symbol == symbol)
    {
        return Ok(if symbol.contains("_decode_route_") {
            Target::QwenDecode(entry.profile, entry.tokens_capacity, Part::Route)
        } else if symbol.ends_with("_combine_v1") {
            Target::QwenCombine(entry.profile)
        } else {
            Target::Qwen(entry.profile)
        });
    }
    // MiMo's tiled permutation is one set of entries for every rank count.
    if symbol.starts_with("chop_row_permute_tile_") && symbol.contains("_mimo_") {
        return Ok(Target::Mimo(model_mimo26_contract::sm120::config::PROFILE));
    }
    Err(format!("{symbol}: not a grouped expert entry"))
}

/// A token's hidden row.
fn hidden(seed: u64, width: usize) -> Vec<u16> {
    (0..width)
        .map(|i| f32_to_bf16(sample(seed ^ (i as u64) << 20) * 0.12))
        .collect()
}

/// A token's router logits over `experts`, spread so its routes vary.
fn logits(seed: u64, experts: usize) -> Vec<f32> {
    (0..experts)
        .map(|e| 2.0 * sample(seed ^ 0x10_61 ^ (e as u64) << 24))
        .collect()
}

/// `count` distinct experts of `pool` drawn for `seed`.
fn draw(seed: u64, pool: std::ops::Range<usize>, count: usize) -> Vec<usize> {
    let mut drawn = Vec::new();
    let mut index = 0;
    while drawn.len() < count {
        let expert = pool.start + (mix(seed ^ index) % pool.len() as u64) as usize;
        index += 1;
        if !drawn.contains(&expert) {
            drawn.push(expert);
        }
    }
    drawn
}

/// A Qwen3.8 decode token's router logits: `logits`, with four of the
/// first `QWEN_POOL` experts of `rank` raised above the rest, so every token
/// has local routes and tokens share experts (and their 16-row tiles).
fn qwen_logits(seed: u64, rank: std::ops::Range<usize>) -> Vec<f32> {
    let mut row = logits(seed, Chains::EXPERTS);
    let pool = rank.start..rank.start + QWEN_POOL.min(rank.len());
    for (j, expert) in draw(seed ^ 0xb005_7000, pool, 4).into_iter().enumerate() {
        row[expert] += 5.0 - 0.5 * j as f32;
    }
    row
}

/// A decode token's eight distinct experts and route weights.
fn token_routes(seed: u64) -> (Vec<usize>, Vec<f32>) {
    let ids = draw(seed, 0..POOL, TOP_K);
    let weights = (0..TOP_K)
        .map(|k| 0.03 + 0.2 * ((mix(seed ^ 0x77 ^ k as u64) % 97) as f32 / 97.0))
        .collect();
    (ids, weights)
}

/// The bucket-1 entry of `v`'s projection, as a token decoded alone runs.
fn one_token(v: &Variant) -> LabResult<Variant> {
    moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|o| {
            o.profile == v.profile
                && o.projection == v.projection
                && o.phase == Phase::Decode
                && o.path == v.path
                && o.query.weight == v.query.weight
                && o.query.epilogues == v.query.epilogues
                && o.tokens_capacity == 1
                && !o.exact
        })
        .ok_or_else(|| format!("{}: no bucket-1 entry", v.symbol))
}

/// Each token's eight expert rows of `v` for tokens seeded by `seeds`, on
/// a poisoned output (and partials, `poison_partials`): every live route's
/// row must be written and finite (every route goes to a stored expert,
/// all on each rank), the rows past them untouched.
fn decode_units(session: &Session, v: &Variant, seeds: &[u64]) -> LabResult<Units> {
    let projected = Projected::new(session, v.clone())?;
    let weights = random_weights(session, v, 0x6a7e ^ v.projection as u64)?;
    let k = v.query.k as usize;
    let width = match v.projection {
        Projection::GateUp => v.query.n as usize / 2,
        Projection::Down => v.query.n as usize,
    };
    // Gate/up reads a token's hidden row; down reads its eight routed rows.
    let rows_per_token = if v.projection == Projection::GateUp {
        1
    } else {
        TOP_K
    };
    let input = seeds
        .iter()
        .flat_map(|&seed| spread(rows_per_token * k, seed))
        .collect::<Vec<_>>();
    let (mut ids, mut route_weights) = (Vec::new(), Vec::new());
    for &seed in seeds {
        let (i, w) = token_routes(seed);
        ids.extend(i);
        route_weights.extend(w);
    }
    let step = routes(session, v, &ids, &route_weights)?;
    let input = session.upload_bf16(&input)?;
    poison::poison(session, &[&projected.output])?;
    poison_partials(session, v, projected.counts, &projected.partials)?;
    projected.launch(session, &input, &weights, &step)?;
    session.stream.synchronize().map_err(err)?;
    let output = poison::read(session, &projected.output)?;
    rows_written(&v.symbol, &output, ids.len(), width)?;
    Ok(output[..ids.len() * width * 2]
        .chunks(TOP_K * width * 2)
        .map(<[u8]>::to_vec)
        .collect())
}

/// Each token's router logits, top-8 experts and route weights from MiMo's
/// fused router of `bucket` tokens over BF16 router weights `weight`
/// ([256 experts, HIDDEN]), the tokens' hidden rows `input` and the expert
/// bias. Every buffer the launch writes but its semaphores starts poisoned
/// and is checked afterwards (`fused_written`).
fn fused_router(
    session: &Session,
    bucket: u32,
    weight: &[u16],
    input: &[u16],
    bias: &[f32],
) -> LabResult<(Vec<f32>, Vec<i32>, Vec<f32>)> {
    let tokens = input.len() / HIDDEN;
    let experts = FUSED_EXPERTS;
    let gemm = RouterGemm::new(session, bucket as usize)?;
    let packed = FragmentWeightLayout {
        rows: experts,
        k: HIDDEN,
        format: FragmentWeightFormat::Bf16,
    }
    .pack(
        &weight
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .map_err(err)?
    .bytes;
    gemm.weights
        .copy_from_bytes(&packed, &session.stream)
        .map_err(err)?;
    let mut padded = input.to_vec();
    padded.resize(bucket as usize * HIDDEN, 0);
    let buffers = [
        session.upload_bf16(&padded)?,
        session.allocate(bucket as usize * experts * 4)?,
        upload_bytes(session, &i32_bytes(&[tokens as i32]))?,
        session.upload_f32(bias)?,
        session.allocate(bucket as usize * TOP_K * 4)?,
        session.allocate(bucket as usize * TOP_K * 4)?,
        session.allocate(experts * 4)?,
        session.allocate((experts + 1) * 4)?,
        session.allocate(experts * 64 * 4)?,
        session.allocate(4)?,
        session.allocate((experts + 1) * 4)?,
    ];
    let order = [
        &gemm.weights,
        &buffers[0],
        &buffers[1],
        &gemm.partials,
        &gemm.semaphores,
        &buffers[2],
        &buffers[3],
        &buffers[4],
        &buffers[5],
        &buffers[6],
        &buffers[7],
        &buffers[8],
        &buffers[9],
        &buffers[10],
    ];
    // The semaphores reset themselves; everything else the launch writes
    // starts poisoned: logits, split partials, IDs, weights and metadata.
    poison::poison(
        session,
        &[
            &buffers[1],
            &gemm.partials,
            &buffers[4],
            &buffers[5],
            &buffers[6],
            &buffers[7],
            &buffers[9],
            &buffers[10],
        ],
    )?;
    poison::fill_all(session, &buffers[8], &[0x80])?;
    let function = gemm.fused_route.as_ref().ok_or("no fused router entry")?;
    let plan =
        model_mimo26_contract::sm120::moe::fused_router_plan(bucket, tokens as u32).map_err(err)?;
    function
        .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
        .map_err(err)?;
    let args = moe::fused_router_arguments([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 1.0)
        .map_err(err)?;
    launch::launch(
        &session.context,
        function,
        &session.stream,
        plan,
        &args,
        &Bindings {
            buffers: order
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, *b))
                .collect(),
        },
    )?;
    let symbol = format!(
        "{}_t{bucket}_v1",
        model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX
    );
    kernel_lab::launched(&symbol);
    session.stream.synchronize().map_err(err)?;
    let read = |buffer: &DeviceBuffer| poison::read(session, buffer);
    let route = Route {
        ids: read(&buffers[4])?,
        weights: read(&buffers[5])?,
        counts: read(&buffers[6])?,
        offsets: read(&buffers[7])?,
        sorted: read(&buffers[8])?,
        padded: read(&buffers[9])?,
        active: Some(read(&buffers[10])?),
        tiles: None,
    };
    fused_written(
        &read(&buffers[1])?,
        &read(&gemm.partials)?,
        &route,
        bucket,
        tokens,
    )
    .map_err(|e| format!("{symbol}: {e}"))?;
    Ok((
        download_f32(session, &buffers[1], tokens * experts)?,
        download_i32(session, &buffers[4], tokens * TOP_K)?,
        download_f32(session, &buffers[5], tokens * TOP_K)?,
    ))
}

/// Each token's router logits, top-8 experts and route weights from MiMo's
/// fused router of `bucket` tokens over random router weights.
fn fused_units(session: &Session, bucket: u32, seeds: &[u64]) -> LabResult<Units> {
    let experts = FUSED_EXPERTS;
    let weight = (0..experts * HIDDEN)
        .map(|i| f32_to_bf16(sample(0x7e47 ^ (i as u64) << 8) * 0.05))
        .collect::<Vec<_>>();
    let input = seeds
        .iter()
        .flat_map(|&seed| hidden(seed, HIDDEN))
        .collect::<Vec<_>>();
    let bias = (0..experts)
        .map(|e| sample(0xc500 ^ e as u64) * 0.01)
        .collect::<Vec<_>>();
    let (logits, ids, weights) = fused_router(session, bucket, &weight, &input, &bias)?;
    Ok((0..seeds.len())
        .map(|t| {
            let mut unit = f32_bytes(&logits[t * experts..(t + 1) * experts]);
            unit.extend(i32_bytes(&ids[t * TOP_K..(t + 1) * TOP_K]));
            unit.extend(f32_bytes(&weights[t * TOP_K..(t + 1) * TOP_K]));
            unit
        })
        .collect())
}

/// MiMo's fused router of `bucket` tokens on exact score ties. Token t's
/// hidden row is one-hot at column t, so its logits are exactly column t
/// of the router weights, set to levels whose sigmoids are exactly 1, 1/2
/// and 0 in F32 (30, 0 and -200; `1 / (1 + exp(-x))` on host and device).
/// Biases on quarter levels then give scores, sigmoid plus bias, that tie
/// exactly within and across levels. A token's pattern puts every expert
/// on one level, on two, on three, or a few on the top level and the rest
/// on the next, so ties straddle the top-8 boundary; each launch takes
/// another bias spread and shifts the patterns over the tokens. The IDs
/// must be the scores in descending order, the lower expert first among
/// equal scores; the logits must be the levels exactly.
fn fused_ties(session: &Session, bucket: u32) -> LabResult<Vec<Compared>> {
    const LOGITS: [f32; 3] = [30.0, 0.0, -200.0];
    const SIGMOIDS: [f32; 3] = [1.0, 0.5, 0.0];
    let experts = FUSED_EXPERTS;
    let tokens = bucket as usize;
    let level = |pattern: usize, salt: u64, expert: usize| -> usize {
        let m = mix(salt ^ (expert as u64) << 12);
        match pattern {
            0 => 0,
            1 => (m % 2) as usize,
            2 => usize::from(m % 40 != 0),
            _ => (m % 3) as usize,
        }
    };
    let mut compared = Vec::new();
    for (launch, spread) in [1_u64, 2, 4, 3].into_iter().enumerate() {
        let bias = (0..experts)
            .map(|e| 0.25 * (mix(0xb1a5 ^ spread << 32 ^ e as u64) % spread) as f32)
            .collect::<Vec<_>>();
        let patterns = (0..tokens).map(|t| (t + launch) % 4).collect::<Vec<_>>();
        let salt = |t: usize| ((launch * 8 + t) as u64) << 40;
        let levels = (0..tokens)
            .map(|t| {
                (0..experts)
                    .map(|e| level(patterns[t], salt(t), e))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut weight = vec![0_u16; experts * HIDDEN];
        for (t, row) in levels.iter().enumerate() {
            for (e, &l) in row.iter().enumerate() {
                weight[e * HIDDEN + t] = f32_to_bf16(LOGITS[l]);
            }
        }
        let mut input = vec![0_u16; tokens * HIDDEN];
        for t in 0..tokens {
            input[t * HIDDEN + t] = f32_to_bf16(1.0);
        }
        let (logits, ids, _) = fused_router(session, bucket, &weight, &input, &bias)?;
        for (t, row) in levels.iter().enumerate() {
            let label = format!(
                "bias over {spread} levels, token {t} of {tokens} (pattern {})",
                patterns[t]
            );
            let scores = row
                .iter()
                .zip(&bias)
                .map(|(&l, &b)| SIGMOIDS[l] + b)
                .collect::<Vec<_>>();
            let mut order = (0..experts as i32).collect::<Vec<_>>();
            order.sort_by(|&a, &b| {
                scores[b as usize]
                    .total_cmp(&scores[a as usize])
                    .then(a.cmp(&b))
            });
            compared.push(Compared {
                label: format!("{label} logits"),
                got: f32_bytes(&logits[t * experts..(t + 1) * experts]),
                expected: f32_bytes(&row.iter().map(|&l| LOGITS[l]).collect::<Vec<_>>()),
            });
            compared.push(Compared {
                label: format!("{label} experts"),
                got: i32_bytes(&ids[t * TOP_K..(t + 1) * TOP_K]),
                expected: i32_bytes(&order[..TOP_K]),
            });
        }
    }
    Ok(compared)
}

impl FullCase {
    /// Poisons what the routed prefill chain writes before reading it: the
    /// router's IDs and weights, the tiled permutation's tile counts and
    /// offsets, expert counts and offsets, sorted rows and padded total,
    /// both experts' rows and partials (`poison_partials`) and the combined
    /// rows. Inputs and the given active list stay.
    fn poison_chain(&self, session: &Session) -> LabResult<()> {
        poison::poison(
            session,
            &[
                &self.ids,
                &self.topk,
                &self.tile_counts,
                &self.tile_offsets,
                &self.counts,
                &self.offsets,
                &self.padded,
                &self.gate_output,
                &self.down_output,
                &self.final_output,
            ],
        )?;
        poison::fill_all(session, &self.sorted, &[0x80])?;
        for (v, partials) in [
            (&self.gate, &self.gate_partials),
            (&self.down, &self.down_partials),
        ] {
            poison_partials(session, v, self.counts(v.knobs.split_k_capacity), partials)?;
        }
        Ok(())
    }

    /// Each row's combined output of MiMo's routed prefill chain over the
    /// given hidden rows and router logits: the row router, the tiled
    /// permutation the bundle runs, both experts and the combine, on
    /// poisoned outputs checked afterwards (`mimo_chain_layout`).
    fn chain_rows(&mut self, session: &Session, input: &[u16], logits: &[f32]) -> LabResult<Units> {
        let rows = input.len() / HIDDEN;
        if rows == 0 || rows > self.tokens {
            return Err(format!("{rows} rows past the chain's capacity"));
        }
        self.input = session.upload_bf16(input)?;
        self.logits = session.upload_f32(logits)?;
        self.live = upload_bytes(session, &i32_bytes(&[rows as i32]))?;
        // Every expert may hold routes.
        self.active = upload_bytes(
            session,
            &i32_bytes(
                &std::iter::once(self.experts as i32)
                    .chain(0..self.experts as i32)
                    .collect::<Vec<_>>(),
            ),
        )?;
        let (tokens, prepared) = (self.tokens, self.prepared_tokens);
        self.tokens = rows;
        self.prepared_tokens =
            kernels_sm120::row::smallest_bucket(rows as u32).ok_or("no row bucket")? as usize;
        let capacity = self.prepared_tokens as u32;
        let (router, _) = row_names(self.profile, self.phase);
        let launched = self
            .poison_chain(session)
            .and_then(|()| {
                launch_row(
                    session,
                    &self.row_module,
                    &format!("chop_row_router_{router}_v1"),
                    capacity,
                    vec![
                        Argument::DeviceBuffer(1),
                        Argument::DeviceBuffer(2),
                        Argument::DeviceBuffer(3),
                        Argument::DeviceBuffer(4),
                        Argument::DeviceBuffer(5),
                        Argument::U32(1.0_f32.to_bits()),
                    ],
                    &[
                        (1, &self.logits),
                        (2, &self.bias),
                        (3, &self.ids),
                        (4, &self.topk),
                        (5, &self.live),
                    ],
                    false,
                )
            })
            .and_then(|()| self.launch_tiled_route(session, capacity, false))
            .and_then(|()| self.launch_projection(session, Projection::GateUp))
            .and_then(|()| self.launch_projection(session, Projection::Down))
            .and_then(|()| self.launch_combine_row(session, capacity, false));
        self.tokens = tokens;
        self.prepared_tokens = prepared;
        launched?;
        session.stream.synchronize().map_err(err)?;
        let read = |buffer: &DeviceBuffer| poison::read(session, buffer);
        let route = Route {
            ids: read(&self.ids)?,
            weights: read(&self.topk)?,
            counts: read(&self.counts)?,
            offsets: read(&self.offsets)?,
            sorted: read(&self.sorted)?,
            padded: read(&self.padded)?,
            active: None,
            tiles: Some((read(&self.tile_counts)?, read(&self.tile_offsets)?)),
        };
        let experts = Experts {
            inter: read(&self.gate_output)?,
            routed: read(&self.down_output)?,
            output: read(&self.final_output)?,
        };
        let layout = mimo_chain_layout(
            self.experts,
            self.gate.query.n as usize / 2,
            capacity as usize,
        );
        route_written(&route, &layout, rows)
            .and_then(|ids| experts_written(&experts, &layout, &ids))
            .map_err(|e| {
                format!(
                    "{} prefill chain of {rows} rows: {e}",
                    self.profile.as_str()
                )
            })?;
        Ok(experts.output[..rows * HIDDEN * 2]
            .chunks(HIDDEN * 2)
            .map(<[u8]>::to_vec)
            .collect())
    }
}

/// Prepared chains, one per profile, sized for the largest prompt.
#[derive(Default)]
pub(crate) struct Moe {
    mimo: Vec<(Profile, FullCase)>,
    qwen: Vec<Chains>,
}

impl Moe {
    /// The Qwen3.8 chains of `profile`, created on first use.
    fn qwen(&mut self, session: &Session, profile: Profile) -> LabResult<&mut Chains> {
        let index = match self.qwen.iter().position(|c| c.profile() == profile) {
            Some(index) => index,
            None => {
                self.qwen.push(Chains::new(session, profile)?);
                self.qwen.len() - 1
            }
        };
        Ok(&mut self.qwen[index])
    }

    /// Each row's output of `target`'s chain for rows seeded by `seeds`.
    fn chain(&mut self, session: &Session, target: &Target, seeds: &[u64]) -> LabResult<Units> {
        match *target {
            Target::Mimo(profile) => {
                if !self.mimo.iter().any(|(p, _)| *p == profile) {
                    let case = FullCase::new(session, profile, Phase::Prefill, CHAIN_ROWS)?;
                    self.mimo.push((profile, case));
                }
                let (_, case) = self.mimo.iter_mut().find(|(p, _)| *p == profile).unwrap();
                let experts = case.experts;
                let input = seeds
                    .iter()
                    .flat_map(|&s| hidden(s, HIDDEN))
                    .collect::<Vec<_>>();
                let logits = seeds
                    .iter()
                    .flat_map(|&s| logits(s, experts))
                    .collect::<Vec<_>>();
                case.chain_rows(session, &input, &logits)
            }
            Target::Qwen(profile) | Target::QwenCombine(profile) => {
                let chains = self.qwen(session, profile)?;
                let x = seeds
                    .iter()
                    .flat_map(|&s| hidden(s, Chains::HIDDEN))
                    .collect::<Vec<_>>();
                let shared = seeds
                    .iter()
                    .flat_map(|&s| hidden(s ^ 0x54a4, Chains::HIDDEN))
                    .collect::<Vec<_>>();
                let logits = seeds
                    .iter()
                    .flat_map(|&s| logits(s, Chains::EXPERTS))
                    .collect::<Vec<_>>();
                chains.rows(session, &x, &logits, &shared)
            }
            _ => Err("not a prefill chain".into()),
        }
    }

    /// Each token's `part` of `profile`'s decode chain of `bucket` tokens
    /// for tokens seeded by `seeds`.
    fn qwen_decode(
        &mut self,
        session: &Session,
        profile: Profile,
        bucket: u32,
        part: Part,
        seeds: &[u64],
    ) -> LabResult<Units> {
        let chains = self.qwen(session, profile)?;
        let rank = chains.rank_experts();
        let x = seeds
            .iter()
            .flat_map(|&s| hidden(s, Chains::HIDDEN))
            .collect::<Vec<_>>();
        let shared = seeds
            .iter()
            .flat_map(|&s| hidden(s ^ 0x54a4, Chains::HIDDEN))
            .collect::<Vec<_>>();
        let logits = seeds
            .iter()
            .flat_map(|&s| qwen_logits(s, rank.clone()))
            .collect::<Vec<_>>();
        chains.decode(session, bucket, part, &x, &logits, &shared)
    }

    /// Each row's output of `profile`'s combine alone for rows seeded by
    /// `seeds`: a row's top-k expert IDs take three of the rank's experts
    /// and seven of any rank, each with a routed row of its own.
    fn qwen_combine(
        &mut self,
        session: &Session,
        profile: Profile,
        seeds: &[u64],
    ) -> LabResult<Units> {
        let chains = self.qwen(session, profile)?;
        let rank = chains.rank_experts();
        let (width, k) = (Chains::HIDDEN, Chains::TOP_K);
        let x = seeds
            .iter()
            .flat_map(|&s| hidden(s, width))
            .collect::<Vec<_>>();
        let shared = seeds
            .iter()
            .flat_map(|&s| hidden(s ^ 0x54a4, width))
            .collect::<Vec<_>>();
        let mut ids = Vec::new();
        let mut routed = Vec::new();
        for &seed in seeds {
            let mut row = draw(seed ^ 0x1d5, rank.clone(), 3);
            for expert in draw(seed ^ 0x2d5, 0..Chains::EXPERTS, 2 * k) {
                if row.len() < k && !row.contains(&expert) {
                    row.push(expert);
                }
            }
            ids.extend(row.iter().map(|&e| e as i32));
            for slot in 0..k as u64 {
                routed.extend(hidden(seed ^ (slot + 1) << 44, width));
            }
        }
        chains.combine(session, &x, &shared, &ids, &routed)
    }
}

/// Rows a chain holds: the largest prompt or batch a case runs.
const CHAIN_ROWS: usize = 512;

impl Adapter for Moe {
    fn kind(&self) -> OpKind {
        OpKind::GroupedMoe
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let target = target(&entry.symbol)?;
        let seeds = (0..u64::from(rows)).collect::<Vec<_>>();
        let mut out = Vec::new();
        let mut start = 0;
        for &end in ends {
            out.extend(self.chain(&sessions[0], &target, &seeds[start..end as usize])?);
            start = end as usize;
        }
        Ok(out)
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        match target(&entry.symbol)? {
            Target::Decode(v) => decode_units(session, &v, seeds),
            Target::Fused(bucket) => fused_units(session, bucket, seeds),
            Target::QwenDecode(profile, bucket, part) => {
                self.qwen_decode(session, profile, bucket, part, seeds)
            }
            Target::QwenCombine(profile) => self.qwen_combine(session, profile, seeds),
            target => self.chain(session, &target, seeds),
        }
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        let session = &sessions[0];
        let target = target(&entry.symbol)?;
        let mut one = |seed: u64| -> LabResult<Vec<u8>> {
            let mut units = match &target {
                Target::Decode(v) => decode_units(session, &one_token(v)?, &[seed])?,
                Target::Fused(_) => fused_units(session, 1, &[seed])?,
                // A token decoded alone runs the bucket-1 chain.
                &Target::QwenDecode(profile, _, part) => {
                    self.qwen_decode(session, profile, 1, part, &[seed])?
                }
                &Target::QwenCombine(profile) => self.qwen_combine(session, profile, &[seed])?,
                _ => return Err(format!("{}: no one-row claim", entry.symbol)),
            };
            units.pop().ok_or_else(|| "no unit".into())
        };
        seeds.iter().map(|&seed| one(seed)).collect()
    }

    /// The Qwen3.8 router on logits rows with exact ties: experts with equal
    /// logits must be taken in ascending order, highest logits first. MiMo's
    /// fused router on exact score ties (`fused_ties`).
    fn ties(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        let session = &sessions[0];
        let profile = match target(&entry.symbol)? {
            Target::Qwen(profile) => profile,
            Target::Fused(bucket) => return fused_ties(session, bucket),
            _ => return Err(format!("{}: no tie case", entry.symbol)),
        };
        let chains = self.qwen(session, profile)?;
        let experts = Chains::EXPERTS;
        // Row r: every expert on one of `levels[r]` values, so ties straddle
        // the top-k boundary.
        let levels = [1_usize, 2, 3, 7, 64];
        let rows = levels
            .iter()
            .enumerate()
            .flat_map(|(r, &n)| {
                (0..experts).map(move |e| ((mix(r as u64 ^ (e as u64) << 16) % n as u64) as f32))
            })
            .collect::<Vec<_>>();
        let ids = chains.route_ids(session, &rows)?;
        Ok(levels
            .iter()
            .enumerate()
            .map(|(r, &n)| {
                let row = &rows[r * experts..(r + 1) * experts];
                let mut order = (0..experts as i32).collect::<Vec<_>>();
                order.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
                let k = Chains::TOP_K;
                Compared {
                    label: format!("{n} logit levels"),
                    got: i32_bytes(&ids[r * k..(r + 1) * k]),
                    expected: i32_bytes(&order[..k]),
                }
            })
            .collect())
    }
}

/// A correct image of what a chain over expert IDs `ids` (token-major,
/// `top_k` per token) writes into poisoned buffers of `capacity` tokens,
/// for the CPU tests of the write checks: IDs and weights, counts, offsets,
/// route slots (padding -1), padded total, active list and tile counts and
/// offsets as `layout` says, and finite rows for the rank's routes and the
/// live tokens.
pub(crate) fn chain_image(layout: &Layout, ids: &[i32], capacity: usize) -> (Route, Experts) {
    let (k, local) = (layout.top_k, layout.experts.len());
    let routes = capacity * k;
    let put = |bytes: &mut Vec<u8>, at: usize, value: [u8; 4]| {
        bytes[at * 4..at * 4 + 4].copy_from_slice(&value);
    };
    let mut counts = vec![0_usize; local];
    for &id in ids {
        if let Some(expert) = layout.local(id) {
            counts[expert] += 1;
        }
    }
    let mut route = Route {
        ids: vec![POISON; routes * 4],
        weights: vec![POISON; routes * 4],
        counts: vec![POISON; local * 4],
        offsets: vec![POISON; (local + 1) * 4],
        sorted: vec![0x80; (routes + local * layout.block) * 4],
        padded: vec![POISON; 4],
        active: layout.active.then(|| vec![POISON; (local + 1) * 4]),
        tiles: layout.tiles.map(|tiles| {
            let bytes = (tiles.max(routes.div_ceil(256)) + 1) * local * 4;
            (vec![POISON; bytes], vec![POISON; bytes])
        }),
    };
    for (at, &id) in ids.iter().enumerate() {
        put(&mut route.ids, at, id.to_ne_bytes());
        put(&mut route.weights, at, 0.125_f32.to_ne_bytes());
    }
    let mut first = 0;
    let mut listed = Vec::new();
    for (expert, &count) in counts.iter().enumerate() {
        put(&mut route.counts, expert, (count as i32).to_ne_bytes());
        put(&mut route.offsets, expert, (first as i32).to_ne_bytes());
        let mut slot = first;
        for at in (0..ids.len()).filter(|&at| layout.local(ids[at]) == Some(expert)) {
            put(&mut route.sorted, slot, (at as i32).to_ne_bytes());
            slot += 1;
        }
        let padded = count.div_ceil(layout.block) * layout.block;
        for slot in first + count..first + padded {
            put(&mut route.sorted, slot, (-1_i32).to_ne_bytes());
        }
        if count > 0 {
            listed.push(expert as i32);
        }
        first += padded;
    }
    put(&mut route.offsets, local, (first as i32).to_ne_bytes());
    put(&mut route.padded, 0, (first as i32).to_ne_bytes());
    if let Some(active) = &mut route.active {
        put(active, 0, (listed.len() as i32).to_ne_bytes());
        for (at, &expert) in listed.iter().enumerate() {
            put(active, 1 + at, expert.to_ne_bytes());
        }
    }
    if let (Some((tile_counts, tile_offsets)), Some(tiles)) = (&mut route.tiles, layout.tiles) {
        tile_counts[..tiles * local * 4].fill(0);
        tile_offsets[..ids.len().div_ceil(256) * local * 4].fill(0);
    }
    let one = 0x3f80_u16.to_ne_bytes();
    let rows = |width: usize, count: usize, written: &dyn Fn(usize) -> bool| {
        let mut bytes = vec![POISON; count * width * 2];
        for row in (0..count).filter(|&row| written(row)) {
            bytes[row * width * 2..(row + 1) * width * 2].copy_from_slice(&one.repeat(width));
        }
        bytes
    };
    let local_route = |at: usize| at < ids.len() && layout.local(ids[at]).is_some();
    let experts = Experts {
        inter: rows(layout.inter, routes, &local_route),
        routed: rows(layout.hidden, routes, &local_route),
        output: rows(layout.hidden, capacity, &|row| row < ids.len() / k),
    };
    (route, experts)
}

/// `bytes` with element `at`, of `value.len()` bytes, set to `value`.
pub(crate) fn with(bytes: &[u8], at: usize, value: &[u8]) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    bytes[at * value.len()..(at + 1) * value.len()].copy_from_slice(value);
    bytes
}

#[test]
fn moe_write_checks_reject_unwritten_outputs() {
    let nan = 0x7fc0_u16.to_ne_bytes();
    let one = 0x3f80_u16.to_ne_bytes();

    // MiMo's prefill chain: three tokens at row bucket 64, so the count
    // stage runs two tiles but only the first holds live routes.
    let layout = mimo_chain_layout(FUSED_EXPERTS, 64, 64);
    assert_eq!(layout.tiles, Some(2));
    let ids = (0..3 * TOP_K as i32)
        .map(|route| route * 37 % 251)
        .collect::<Vec<_>>();
    let (route, experts) = chain_image(&layout, &ids, 64);
    assert_eq!(route_written(&route, &layout, 3), Ok(ids.clone()));
    assert!(experts_written(&experts, &layout, &ids).is_ok());
    let check = |route: &Route| route_written(route, &layout, 3).map(|_| ());
    // Nothing written.
    let poisoned = |bytes: &Vec<u8>| vec![POISON; bytes.len()];
    let (tile_counts, tile_offsets) = route.tiles.clone().unwrap();
    let unwritten = Route {
        ids: poisoned(&route.ids),
        weights: poisoned(&route.weights),
        counts: poisoned(&route.counts),
        offsets: poisoned(&route.offsets),
        sorted: poisoned(&route.sorted),
        padded: poisoned(&route.padded),
        active: None,
        tiles: Some((poisoned(&tile_counts), poisoned(&tile_offsets))),
    };
    assert!(check(&unwritten).is_err());
    // Only the expert IDs and weights written.
    let ids_only = Route {
        ids: route.ids.clone(),
        weights: route.weights.clone(),
        ..unwritten
    };
    assert!(check(&ids_only).is_err());
    // An expert ID or weight left poisoned, or a NaN weight.
    let ids_hole = Route {
        ids: with(&route.ids, 5, &[POISON; 4]),
        ..route.clone()
    };
    assert!(check(&ids_hole).is_err());
    let nan_weight = Route {
        weights: with(&route.weights, 2, &f32::NAN.to_ne_bytes()),
        ..route.clone()
    };
    assert!(check(&nan_weight).is_err());
    // A route past the live ones written.
    let past = Route {
        ids: with(&route.ids, 3 * TOP_K, &7_i32.to_ne_bytes()),
        ..route.clone()
    };
    assert!(check(&past).is_err());
    // A route slot left poisoned, and a padding slot written.
    let first = i32s(&route.offsets)[ids[0] as usize] as usize;
    let slot_hole = Route {
        sorted: with(&route.sorted, first, &[POISON; 4]),
        ..route.clone()
    };
    assert!(check(&slot_hole).is_err());
    let missing_padding = Route {
        sorted: with(&route.sorted, first + 1, &[0x80; 4]),
        ..route.clone()
    };
    assert!(check(&missing_padding).is_err());
    let padding = Route {
        sorted: with(&route.sorted, first + 1, &0_i32.to_ne_bytes()),
        ..route.clone()
    };
    assert!(check(&padding).is_err());
    // The second launched tile's counts left poisoned; its offsets written.
    let tile_hole = Route {
        tiles: Some((
            with(&tile_counts, 256 + 9, &[POISON; 4]),
            tile_offsets.clone(),
        )),
        ..route.clone()
    };
    assert!(check(&tile_hole).is_err());
    let tile_past = Route {
        tiles: Some((
            tile_counts.clone(),
            with(&tile_offsets, 256 + 9, &0_i32.to_ne_bytes()),
        )),
        ..route.clone()
    };
    assert!(check(&tile_past).is_err());
    // A given active list must not be read back as written.
    let active = Route {
        active: Some(vec![0; (FUSED_EXPERTS + 1) * 4]),
        ..route.clone()
    };
    assert!(check(&active).is_err());
    // Expert rows: a hole, a NaN, a row past the live routes, and a
    // combined row past the live tokens.
    let check = |experts: &Experts| experts_written(experts, &layout, &ids);
    let width = layout.inter;
    for (inter, routed, output) in [
        (
            with(&experts.inter, 2 * width + 7, &[POISON; 2]),
            experts.routed.clone(),
            experts.output.clone(),
        ),
        (
            experts.inter.clone(),
            with(&experts.routed, HIDDEN + 3, &nan),
            experts.output.clone(),
        ),
        (
            with(&experts.inter, 3 * TOP_K * width, &one),
            experts.routed.clone(),
            experts.output.clone(),
        ),
        (
            experts.inter.clone(),
            experts.routed.clone(),
            with(&experts.output, 3 * HIDDEN, &one),
        ),
        (
            experts.inter.clone(),
            experts.routed.clone(),
            with(&experts.output, HIDDEN - 1, &[POISON; 2]),
        ),
    ] {
        assert!(
            check(&Experts {
                inter,
                routed,
                output
            })
            .is_err()
        );
    }

    // MiMo's fused router of a bucket of 16, two live tokens.
    let (bucket, tokens) = (16_u32, 2);
    let ids = (0..tokens as i32 * TOP_K as i32)
        .map(|route| route * 11 % 256)
        .collect::<Vec<_>>();
    let (route, _) = chain_image(&fused_layout(), &ids, bucket as usize);
    let mut logits = vec![POISON; bucket as usize * FUSED_EXPERTS * 4];
    logits[..tokens * FUSED_EXPERTS * 4]
        .copy_from_slice(&0.5_f32.to_ne_bytes().repeat(tokens * FUSED_EXPERTS));
    let mut partials = vec![POISON; (64 * 16 * 16 + 16) * 4];
    for row in 0..64 * 16 {
        partials[row * 16 * 4..(row * 16 + tokens) * 4]
            .copy_from_slice(&0.25_f32.to_ne_bytes().repeat(tokens));
    }
    let check = |logits: &[u8], partials: &[u8], route: &Route| {
        fused_written(logits, partials, route, bucket, tokens).map(|_| ())
    };
    assert!(check(&logits, &partials, &route).is_ok());
    assert!(check(&vec![POISON; logits.len()], &partials, &route).is_err());
    assert!(
        check(
            &with(&logits, 300, &f32::NAN.to_ne_bytes()),
            &partials,
            &route
        )
        .is_err()
    );
    assert!(
        check(
            &with(&logits, 2 * FUSED_EXPERTS, &0_f32.to_ne_bytes()),
            &partials,
            &route
        )
        .is_err()
    );
    assert!(check(&logits, &with(&partials, 5 * 16 + 1, &[POISON; 4]), &route).is_err());
    assert!(
        check(
            &logits,
            &with(&partials, 5 * 16 + 2, &0_f32.to_ne_bytes()),
            &route
        )
        .is_err()
    );
    // An active expert left unlisted, or one listed past the active count.
    let active = route.active.clone().unwrap();
    let unlisted = Route {
        active: Some(with(&active, 3, &[POISON; 4])),
        ..route.clone()
    };
    assert!(check(&logits, &partials, &unlisted).is_err());
    let listed = 1 + 2 * TOP_K;
    let extra = Route {
        active: Some(with(&active, listed, &0_i32.to_ne_bytes())),
        ..route.clone()
    };
    assert!(check(&logits, &partials, &extra).is_err());

    // MiMo's decode expert rows: two tokens' sixteen rows of a bucket of 4.
    let width = 32;
    let mut rows = vec![POISON; 4 * TOP_K * width * 2];
    rows[..2 * TOP_K * width * 2].copy_from_slice(&one.repeat(2 * TOP_K * width));
    assert!(rows_written("rows", &rows, 2 * TOP_K, width).is_ok());
    assert!(rows_written("rows", &vec![POISON; rows.len()], 2 * TOP_K, width).is_err());
    assert!(rows_written("rows", &with(&rows, 40, &[POISON; 2]), 2 * TOP_K, width).is_err());
    assert!(rows_written("rows", &with(&rows, 41, &nan), 2 * TOP_K, width).is_err());
    assert!(
        rows_written(
            "rows",
            &with(&rows, 2 * TOP_K * width, &one),
            2 * TOP_K,
            width
        )
        .is_err()
    );
}

/// The fragment MXFP4 experts at every served decode bucket ignore their
/// companions. Buckets of up to 16 rows run the ring decode entries (a unit
/// is a token's eight expert rows); wider buckets run the prefill route and
/// expert chain of `FullCase::chain_rows` at the bucket's rows (a unit is a
/// token's combined row). Probe tokens run alone, beside two companion sets
/// that route elsewhere, and at two live counts of each bucket; every
/// probe's bits must match.
#[test]
fn fragment_experts_ignore_their_companions_at_decode_buckets() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let companions = |count: usize, salt: u64| {
            (0..count as u64)
                .map(|i| 0x5000 ^ salt ^ i.wrapping_mul(7919))
                .collect::<Vec<_>>()
        };
        // Probes first, then the companions.
        let steps = |probes: &[u64], bucket: usize| {
            let room = bucket - probes.len();
            [(room, 0xa), (room, 0xb), (room - room.min(3), 0xb)].map(|(count, salt)| {
                probes
                    .iter()
                    .copied()
                    .chain(companions(count, salt))
                    .collect::<Vec<_>>()
            })
        };
        let same = |label: &str, alone: &Units, got: &Units| -> LabResult<()> {
            match alone.iter().zip(got).position(|(a, b)| a != b) {
                Some(probe) => Err(format!("{label}: probe {probe} differs from alone")),
                None => Ok(()),
            }
        };
        let decode = moe::catalog(kernel_lab::catalogs())
            .into_iter()
            .filter(|v| {
                v.fragment_decode()
                    && v.path == InputPath::A16Gather
                    && !v.exact
                    && (2..=16).contains(&v.tokens_capacity)
            })
            .collect::<Vec<_>>();
        let mut profiles = decode.iter().map(|v| v.profile).collect::<Vec<_>>();
        profiles.dedup();
        let mut compared = 0;
        for v in &decode {
            let bucket = v.tokens_capacity as usize;
            let probes = &[11, 12, 13, 14][..(bucket / 2).min(4)];
            let alone = decode_units(session, v, probes)?;
            for seeds in steps(probes, bucket) {
                let got = decode_units(session, v, &seeds)?;
                same(
                    &format!("{} {} tokens", v.symbol, seeds.len()),
                    &alone,
                    &got,
                )?;
                compared += 1;
            }
        }
        for &profile in &profiles {
            let mut case = FullCase::new(session, profile, Phase::Prefill, 256)?;
            let experts = case.experts;
            let mut chain = |seeds: &[u64]| {
                let input = seeds
                    .iter()
                    .flat_map(|&s| hidden(s, HIDDEN))
                    .collect::<Vec<_>>();
                let logits = seeds
                    .iter()
                    .flat_map(|&s| logits(s, experts))
                    .collect::<Vec<_>>();
                case.chain_rows(session, &input, &logits)
            };
            let probes = [11, 12, 13, 14];
            let alone = chain(&probes)?;
            for bucket in [32, 64, 128, 256] {
                for seeds in steps(&probes, bucket) {
                    let got = chain(&seeds)?;
                    same(
                        &format!("{profile:?} chain of {} tokens", seeds.len()),
                        &alone,
                        &got,
                    )?;
                    compared += 1;
                }
            }
        }
        let expected = 3 * decode.len() + 3 * 4 * profiles.len();
        if decode.is_empty() || compared != expected {
            return Err(format!("{compared} comparisons, expected {expected}"));
        }
        Ok(())
    })
    .expect("fragment experts ignore their companions");
}
