use kernel_core::{
    codec::{SmallFloat, bf16_to_f32, f32_to_bf16},
    contract::{ArgumentList, DType, LaunchPlan, RuntimeCounts, WeightFormat},
    oracle::{self, GemmEpilogue, GemmShape, OracleOutput},
    precision::OperationUse,
    quantize::{QuantFormat, Quantized, ScaleFormat, ScaleProduct, Scales},
};
use kernel_lab::projection_cases::{self, Prepared};
use kernel_lab::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use kernels_sm120::{
    moe,
    precision::{Compute, OperationHandle},
};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::json;
use std::collections::BTreeMap;
const GUARD: u16 = 0x42c8;
const INPUT_PATTERNS: usize = 7;
const X: u64 = 1;
const W: u64 = 2;
const WS: u64 = 3;
const WG: u64 = 4;
const Y: u64 = 5;
const LIVE: u64 = 6;
const TOPK: u64 = 9;
const SORTED: u64 = 10;
const OFFSETS: u64 = 11;
const PADDED: u64 = 12;
const ACTIVE: u64 = 13;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn input(row: usize, k: usize) -> u16 {
    f32_to_bf16((((k * 17 + row % INPUT_PATTERNS * 13) % 101) as f32 - 49.0) / 173.0)
}
fn fixture_hash(expert: usize, operand: usize, row: usize, column: usize) -> u64 {
    let mut x = (expert as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (operand as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ (row as u64).wrapping_mul(0x94d049bb133111eb)
        ^ column as u64;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
fn weight_code(expert: usize, operand: usize, row: usize, column: usize) -> u8 {
    (fixture_hash(expert, operand, row, column) & 15) as u8
}
fn route(routed_row: usize, experts: usize) -> usize {
    routed_row % experts
}
#[test]
fn route_sweeps_cover_every_expert_at_every_capacity() {
    for experts in [64_usize, 256, 512] {
        for capacity in [1, 8, 16, 32, 64, 8192] {
            let top_k = 8;
            let mut seen = std::collections::BTreeSet::new();
            for offset in (0..experts.div_ceil(top_k)).step_by(capacity) {
                seen.extend((0..capacity * top_k).map(|r| route(r + offset * top_k, experts)));
            }
            assert_eq!(seen.len(), experts);
        }
    }
}

#[test]
fn grouped_weights_distinguish_columns_thirteen_rows_apart() {
    for expert in [0, 1, 63] {
        for row in [0, 1, 32, 127] {
            assert!(
                (0..128)
                    .any(|k| weight_code(expert, 0, row, k) != weight_code(expert, 0, row + 13, k))
            );
        }
    }
}

fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len().max(4))?;
    if !bytes.is_empty() {
        buffer.copy_from_bytes(bytes, &s.stream).map_err(err)?;
    }
    Ok(buffer)
}
fn integers(s: &Session, values: &[i32]) -> LabResult<DeviceBuffer> {
    upload(
        s,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

struct Compiled(Prepared);
impl Compiled {
    fn call(
        &self,
        s: &Session,
        symbol: &str,
        plan: LaunchPlan,
        arguments: ArgumentList,
    ) -> LabResult<Call> {
        Ok(Call {
            function: self.0.function(s, symbol, plan.dynamic_shared_bytes)?,
            plan,
            arguments,
        })
    }
}
struct Call {
    function: Function,
    plan: LaunchPlan,
    arguments: ArgumentList,
}
struct Weights {
    n: usize,
    k: usize,
    experts: usize,
    top_k: usize,
    swiglu: bool,
    codes: DeviceBuffer,
    scales: DeviceBuffer,
    globals: DeviceBuffer,
    reference: Vec<OracleOutput>,
}
impl Weights {
    fn new(s: &Session, handle: &OperationHandle) -> LabResult<Self> {
        let Compute::Grouped(grouped) = &handle.compute else {
            return Err("expected a grouped projection".into());
        };
        let query = grouped.query;
        let (n, k) = (query.n as usize, query.k as usize);
        let experts = query.grouped.map_or(1, |g| g.experts as usize);
        let top_k = query.grouped.map_or(1, |g| g.top_k as usize);
        let swiglu = query.epilogues.swiglu.is_some();
        let operands = if swiglu { 2 } else { 1 };
        let rows = n / operands;
        assert_eq!(query.weight, WeightFormat::Nvfp4);
        let group = 16;
        let bytes_per_row = k / 2;
        let blocks = k / group;
        let mut codes = vec![0; experts * n * bytes_per_row];
        let mut scales = vec![0; experts * n * blocks];
        let mut globals = Vec::new();
        let mut reference = Vec::new();
        let a = (0..INPUT_PATTERNS)
            .flat_map(|r| (0..k).map(move |c| f64::from(bf16_to_f32(input(r, c)))))
            .collect::<Vec<_>>();
        for expert in 0..experts {
            let mut decoded = Vec::new();
            for operand in 0..operands {
                let global = 0.0061 + (expert * 3 + operand * 5) as f32 * 0.000013;
                globals.push(global);
                let mut template_codes = vec![0; rows * bytes_per_row];
                let mut template_scales = vec![0; rows * blocks];
                for row in 0..rows {
                    for b in 0..blocks {
                        template_scales[row * blocks + b] =
                            48 + (fixture_hash(expert, operand, row, b + k) % 16) as u8;
                    }
                    for c in 0..k {
                        template_codes[row * bytes_per_row + c / 2] |=
                            weight_code(expert, operand, row, c) << (4 * (c % 2));
                    }
                }
                let (element, scale, stored_scales, global_scale) = (
                    SmallFloat::E2M1,
                    ScaleFormat::E4M3,
                    Scales::Ue4M3(template_scales.clone()),
                    global,
                );
                let format = QuantFormat {
                    element,
                    block: (1, group),
                    scale,
                    chunk_rows: None,
                };
                let template = Quantized::from_parts(
                    rows,
                    k,
                    format,
                    template_codes.clone(),
                    stored_scales,
                    global_scale,
                )
                .map_err(|e| e.to_string())?;
                // The template's decode forms code * local scale in F32.
                decoded.extend(template.decoded_with(ScaleProduct::F32));
                for row in 0..rows {
                    let physical = (expert * operands + operand) * rows + row;
                    codes[physical * bytes_per_row..(physical + 1) * bytes_per_row]
                        .copy_from_slice(
                            &template_codes[row * bytes_per_row..(row + 1) * bytes_per_row],
                        );
                    for block in 0..blocks {
                        let at = moe::packed_scale_offset(physical, block, blocks);
                        scales[at] = template_scales[row * blocks + block];
                    }
                }
            }
            reference.push(
                oracle::gemm(
                    &a,
                    &decoded,
                    GemmShape {
                        m: INPUT_PATTERNS,
                        n: rows * operands,
                        k,
                        experts: 1,
                    },
                    &[],
                    GemmEpilogue {
                        residual: None,
                        swiglu: query.epilogues.swiglu,
                    },
                )
                .map_err(err)?,
            );
        }
        Ok(Self {
            n,
            k,
            experts,
            top_k,
            swiglu,
            codes: upload(s, &codes)?,
            scales: upload(s, &scales)?,
            globals: s.upload_f32(&globals)?,
            reference,
        })
    }
    fn width(&self) -> usize {
        self.n / if self.swiglu { 2 } else { 1 }
    }
}

/// One grouped launch's inputs: the input rows (the tokens' for gate/up,
/// the routed rows' for down), and each routed row's expert and route
/// weight, token-major (`top_k` per token).
struct Routed {
    x: Vec<u16>,
    routes: Vec<usize>,
    route_weights: Vec<f32>,
}

/// The output of `handle` over `case` at `live` tokens (empty routes when
/// `empty_routes`): one captured graph replayed three times, each replay on
/// a guarded output, with equal bits each time and the guard kept past the
/// live routed rows.
fn launch_routed(
    s: &Session,
    compiled: &Compiled,
    handle: &OperationHandle,
    weight: &Weights,
    live: u32,
    case: &Routed,
    empty_routes: bool,
) -> LabResult<Vec<u16>> {
    let routed = live as usize * weight.top_k;
    let output_len = routed * weight.width() + 128;
    let permutation = oracle::expert_permute(
        if empty_routes { &[] } else { &case.routes },
        weight.experts,
        16,
    )
    .map_err(err)?;
    let offsets = permutation
        .expert_offsets
        .iter()
        .map(|v| *v as i32)
        .collect::<Vec<_>>();
    let active = (0..weight.experts)
        .filter(|e| offsets[*e] < offsets[*e + 1])
        .map(|e| e as i32)
        .collect::<Vec<_>>();
    let buffers = BTreeMap::from([
        (X, s.upload_bf16(&case.x)?),
        (Y, s.upload_bf16(&vec![GUARD; output_len])?),
        (LIVE, upload(s, &live.to_le_bytes())?),
        (TOPK, s.upload_f32(&case.route_weights)?),
        (
            SORTED,
            integers(
                s,
                &permutation
                    .forward
                    .iter()
                    .map(|r| r.map_or(-1, |r| r as i32))
                    .collect::<Vec<_>>(),
            )?,
        ),
        (OFFSETS, integers(s, &offsets)?),
        (
            PADDED,
            upload(s, &(permutation.forward.len() as u32).to_le_bytes())?,
        ),
        (
            ACTIVE,
            integers(s, &[vec![active.len() as i32], active].concat())?,
        ),
    ]);
    let mut calls = Vec::new();
    let counts = RuntimeCounts {
        m: live,
        split_k: 1,
        batch: 1,
        context: 0,
        routed_rows: routed as u32,
    };
    match &handle.compute {
        Compute::Grouped(g) => {
            calls.push(
                compiled.call(
                    s,
                    &g.symbol,
                    g.plan(counts).map_err(err)?,
                    g.arguments(
                        [
                            X, WS, WG, WS, WG, Y, TOPK, OFFSETS, SORTED, PADDED, LIVE, ACTIVE, 0,
                            0, 0,
                        ],
                        g.weight_tensor_map(W).map_err(err)?,
                    )
                    .map_err(err)?,
                )?,
            );
        }
        _ => unreachable!(),
    }
    let mut bindings = Bindings {
        buffers: buffers.iter().map(|(id, b)| (*id, b)).collect(),
    };
    bindings.buffers.extend([
        (W, &weight.codes),
        (WS, &weight.scales),
        (WG, &weight.globals),
    ]);
    let graph = s.capture(|stream| {
        for c in &calls {
            launch(
                &s.context,
                &c.function,
                stream,
                c.plan,
                &c.arguments,
                &bindings,
            )?;
        }
        Ok(())
    })?;
    let mut previous = None;
    for _ in 0..3 {
        buffers[&Y]
            .copy_from_bytes(
                &vec![GUARD; output_len]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
                &s.stream,
            )
            .map_err(err)?;
        graph.launch(&s.stream).map_err(err)?;
        let actual = s.download_bf16(&buffers[&Y], output_len)?;
        if previous.as_ref().is_some_and(|p| p != &actual) {
            return Err("graph replay changed output bits".into());
        }
        let written = if empty_routes {
            0
        } else {
            routed * weight.width()
        };
        if actual[written..].iter().any(|v| *v != GUARD) {
            return Err("projection wrote outside its live output".into());
        }
        previous = Some(actual);
    }
    Ok(previous.unwrap())
}

#[expect(
    clippy::too_many_arguments,
    reason = "one declared operation, use and live-row case"
)]
fn projection(
    s: &Session,
    compiled: &Compiled,
    handle: &OperationHandle,
    weight: &Weights,
    usage: OperationUse,
    live: u32,
    offset: usize,
    empty_routes: bool,
) -> LabResult<Vec<u16>> {
    assert!(handle.quantizer.is_none() && handle.gather.is_none());
    let down = !weight.swiglu;
    let top_k = weight.top_k;
    let m = live as usize;
    let routed = m * top_k;
    let width = weight.width();
    let input_rows = if down { routed } else { m };
    let x = (0..input_rows.max(1))
        .flat_map(|r| {
            (0..weight.k).map(move |c| input(r + if down { offset * top_k } else { offset }, c))
        })
        .collect::<Vec<_>>();
    assert!(x.iter().any(|v| {
        let f = bf16_to_f32(*v);
        SmallFloat::E4M3.decode_f32(SmallFloat::E4M3.encode_f32(f)) != f
    }));
    let routes = (0..routed)
        .map(|r| route(r + offset * top_k, weight.experts))
        .collect::<Vec<_>>();
    let route_weights = (0..routed.max(1))
        .map(|r| 0.3125 + ((r + offset * top_k) % 3) as f32 / 16.0)
        .collect::<Vec<_>>();
    let case = Routed {
        x,
        routes,
        route_weights,
    };
    let actual = launch_routed(s, compiled, handle, weight, live, &case, empty_routes)?;
    let (routes, route_weights) = (case.routes, case.route_weights);
    let mut max_error = 0_f64;
    let (mut max_error_bound, mut squared_error, mut squared_reference) = (0_f64, 0_f64, 0_f64);
    if !empty_routes {
        for r in 0..routed {
            let expert = routes[r];
            let input_row = (if !down {
                r / top_k + offset
            } else {
                r + offset * top_k
            }) % INPUT_PATTERNS;
            let mut reference = weight.reference[expert].clone();
            if down {
                let scale = f64::from(route_weights[r]);
                for v in &mut reference.values {
                    *v *= scale;
                }
                for v in &mut reference.sum_abs_products {
                    *v *= scale;
                }
            }
            let (mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0);
            for c in 0..width {
                let i = input_row * width + c;
                let value = f64::from(bf16_to_f32(actual[r * width + c]));
                let expected = reference.values[i];
                let error = (value - expected).abs();
                let bound = oracle::numeric_bound(&reference, i, DType::Bf16);
                if !value.is_finite() || error > bound {
                    return Err(format!(
                        "row {r} expert {expert} column {c}: {value} != {expected}, error {error} > {bound}"
                    ));
                }
                max_error = max_error.max(error);
                max_error_bound = max_error_bound.max(error / bound);
                squared_error += error * error;
                squared_reference += expected * expected;
                dot += value * expected;
                aa += value * value;
                bb += expected * expected;
            }
            if aa > 0.0 && bb > 0.0 && dot / (aa * bb).sqrt() < 0.9999 {
                return Err(format!(
                    "row {r} expert {expert}: cosine {} < 0.9999",
                    dot / (aa * bb).sqrt()
                ));
            }
        }
    }
    s.receipt(
        "generated-grouped-projection",
        BTreeMap::from([
            ("usage".into(), json!(usage)),
            (
                "geometry".into(),
                json!([weight.experts, weight.n, weight.k]),
            ),
            ("live".into(), json!(live)),
            ("empty_routes".into(), json!(empty_routes)),
            ("graph_replays".into(), json!(3)),
            ("correct".into(), json!(true)),
            ("max_abs_error".into(), json!(max_error)),
            ("max_error_over_bound".into(), json!(max_error_bound)),
            (
                "relative_rms".into(),
                json!(if squared_reference > 0.0 {
                    (squared_error / squared_reference).sqrt()
                } else {
                    0.0
                }),
            ),
            ("layer_cosine_minimum".into(), json!(0.9999)),
            ("bf16_activation_quantizers".into(), json!(0)),
        ]),
    )?;
    Ok(actual[..if empty_routes { 0 } else { routed * width }].to_vec())
}

#[test]
fn generated_grouped_offers_have_oracles_and_invariance() {
    kernel_lab::gpu_test(8 << 30, |s| {
        let compiled = Compiled(Prepared::read()?);
        let (offers, _, _) = kernel_lab::models::projection_inventory()?;
        let mut tested = 0;
        let mut weights = BTreeMap::new();
        let selection = projection_cases::Selection::read()?;
        for (case_index, offer) in offers.iter().enumerate() {
            if !selection.contains(case_index) {
                continue;
            }
            let Compute::Grouped(g) = &offer.handle().compute else {
                continue;
            };
            let case = projection_cases::typed(offer)?;
            let key = format!(
                "{:?}",
                (
                    case.capability.kind,
                    case.capability.geometry,
                    case.capability.weight,
                    case.capability.operands
                )
            );
            if !weights.contains_key(&key) {
                weights.insert(key.clone(), Weights::new(s, offer.handle())?);
            }
            let weight = &weights[&key];
            for live in &case.oracle_rows {
                projection(
                    s,
                    &compiled,
                    offer.handle(),
                    weight,
                    offer.usage(),
                    *live,
                    0,
                    false,
                )?;
            }
            for offset in (offer.usage().capacity as usize..weight.experts.div_ceil(weight.top_k))
                .step_by(offer.usage().capacity as usize)
            {
                projection(
                    s,
                    &compiled,
                    offer.handle(),
                    weight,
                    offer.usage(),
                    offer.usage().capacity,
                    offset,
                    false,
                )?;
            }
            projection(
                s,
                &compiled,
                offer.handle(),
                weight,
                offer.usage(),
                1,
                0,
                true,
            )?;
            if case.chunk_invariant {
                let rows = offer.usage().capacity.min(33);
                let split = rows / 2;
                let whole = projection(
                    s,
                    &compiled,
                    offer.handle(),
                    weight,
                    offer.usage(),
                    rows,
                    0,
                    false,
                )?;
                let mut chunks = projection(
                    s,
                    &compiled,
                    offer.handle(),
                    weight,
                    offer.usage(),
                    split,
                    0,
                    false,
                )?;
                chunks.extend(projection(
                    s,
                    &compiled,
                    offer.handle(),
                    weight,
                    offer.usage(),
                    rows - split,
                    split as usize,
                    false,
                )?);
                if whole != chunks {
                    return Err("generated grouped chunks differ".into());
                }
                // Over 13-binade inputs, with every token on the same experts
                // and on experts of their own: every two-chunk split around
                // the 64-row blocks, and even tokens beside other odd ones.
                let (rows, splits) =
                    projection_cases::chunk_splits(offer.usage().capacity, &[u32::from(g.knobs.tile_m)]);
                for routing in [Routing::Shared, Routing::Shifted] {
                    let tokens = (0..rows as usize).collect::<Vec<_>>();
                    let whole = grouped_rows(s, &compiled, offer.handle(), weight, &tokens, routing)?;
                    if offer.invariance().batch == kernels_sm120::invariance::Batch::Independent {
                        let beside = tokens
                            .iter()
                            .map(|&t| if t % 2 == 0 { t } else { 5000 + t })
                            .collect::<Vec<_>>();
                        let other = grouped_rows(s, &compiled, offer.handle(), weight, &beside, routing)?;
                        if (0..rows as usize).step_by(2).any(|t| other[t] != whole[t]) {
                            return Err(format!(
                                "{}: prefill tokens ({routing:?}) change with the other tokens",
                                g.symbol
                            ));
                        }
                    }
                    for &split in &splits {
                        let split = split as usize;
                        let mut chunks =
                            grouped_rows(s, &compiled, offer.handle(), weight, &tokens[..split], routing)?;
                        chunks.extend(grouped_rows(
                            s,
                            &compiled,
                            offer.handle(),
                            weight,
                            &tokens[split..],
                            routing,
                        )?);
                        if whole != chunks {
                            return Err(format!(
                                "{}: chunks split at {split} ({routing:?}) differ",
                                g.symbol
                            ));
                        }
                    }
                }
            }
            // Rows: each token of a launch of 1 to `max` tokens has the
            // routed rows of that token launched alone through the
            // operation's one-row offer (`one_row_offer`), and keeps them
            // beside other tokens' values.
            let rows_claim = matches!(
                offer.invariance().rows,
                kernels_sm120::invariance::Rows::OneRow { max } if max > 1
            );
            if let kernels_sm120::invariance::Rows::OneRow { max } = offer.invariance().rows
                && max > 1
            {
                let single = &offers[projection_cases::one_row_offer(&offers, case_index)?];
                let capacity = max.min(offer.usage().capacity) as usize;
                for routing in [Routing::Shared, Routing::Shifted] {
                    for count in 1..=capacity {
                        let tokens = (0..count).collect::<Vec<_>>();
                        let together =
                            grouped_rows(s, &compiled, offer.handle(), weight, &tokens, routing)?;
                        let beside = std::iter::once(0)
                            .chain((1..count).map(|t| 1000 + t))
                            .collect::<Vec<_>>();
                        let first =
                            grouped_rows(s, &compiled, offer.handle(), weight, &beside, routing)?;
                        if first[0] != together[0] {
                            return Err(format!(
                                "{}: token 0 of {count} ({routing:?}) changes with the other tokens",
                                g.symbol
                            ));
                        }
                        for (t, got) in together.iter().enumerate() {
                            let alone =
                                grouped_rows(s, &compiled, single.handle(), weight, &[t], routing)?;
                            if alone[0] != *got {
                                return Err(format!(
                                    "{}: token {t} of {count} ({routing:?}) differs from {} alone",
                                    g.symbol,
                                    projection_cases::compute_symbol(single)
                                ));
                            }
                        }
                    }
                }
            }
            s.receipt(
                "generated-projection",
                BTreeMap::from([
                    ("case_index".into(), json!(case_index)),
                    ("case".into(), json!(case)),
                    ("oracle".into(), json!(true)),
                    ("replay".into(), json!(true)),
                    ("one_row_rows".into(), json!(rows_claim)),
                ]),
            )?;
            tested += 1;
        }
        if tested == 0 {
            return Err("no generated grouped offers ran".into());
        }
        Ok(())
    })
    .expect("generated grouped oracle and invariance");
}

fn u64s(s: &Session, values: &[u64]) -> LabResult<DeviceBuffer> {
    upload(
        s,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn lookup(
    s: &Session,
    compiled: &Compiled,
    offer: &kernels_sm120::precision::KernelOffer,
    live: u32,
    mapped: bool,
    start: u64,
    row_origin: usize,
) -> LabResult<Vec<u16>> {
    let usage = offer.usage();
    let request = offer
        .lookup_request()
        .ok_or("missing generated lookup request")?;
    if request.weight != WeightFormat::Nvfp4
        || request.ids != kernels_sm120::row::precision::LookupIds::U64
    {
        return Err("unhandled generated lookup format or IDs".into());
    }
    let heads = request.heads as usize;
    let Compute::Lookup(lookup) = &offer.handle().compute else {
        unreachable!()
    };
    let rows = 37_u64;
    let width = request.width as usize;
    let codes = (0..rows as usize * width / 2)
        .map(|i| ((i * 37 + i / 80 * 3) % 256) as u8)
        .collect::<Vec<_>>();
    let scales = (0..rows as usize * width / 16)
        .map(|i| 40 + (i % 27) as u8)
        .collect::<Vec<_>>();
    let global = 0.0371_f32;
    let nvfp4 = QuantFormat {
        element: SmallFloat::E2M1,
        block: (1, 16),
        scale: ScaleFormat::E4M3,
        chunk_rows: None,
    };
    let stored_scales = Scales::Ue4M3(scales.clone());
    let decoded = Quantized::from_parts(
        rows as usize,
        width,
        nvfp4,
        codes.clone(),
        stored_scales,
        global,
    )
    .map_err(err)?;
    let table = if mapped {
        let mut buffer = DeviceBuffer::allocate_mapped_host(&s.stream, codes.len()).map_err(err)?;
        buffer
            .mapped_host_bytes_mut()
            .map_err(err)?
            .copy_from_slice(&codes);
        buffer
    } else {
        upload(s, &codes)?
    };
    let ids = (0..usage.capacity as usize * heads)
        .map(|i| {
            let i = i + row_origin * heads;
            match i % 7 {
                0 => start,
                1 => start + rows - 1,
                2 => start + rows,
                3 => start - 1,
                4 => u64::MAX,
                _ => start + i as u64 % rows,
            }
        })
        .collect::<Vec<_>>();
    let output_len = usage.capacity as usize * heads * width + 128;
    let buffers = BTreeMap::from([
        (W, table),
        (X, u64s(s, &ids)?),
        (WS, upload(s, &scales)?),
        (WG, s.upload_f32(&[global])?),
        (LIVE, upload(s, &live.to_le_bytes())?),
        (Y, s.upload_bf16(&vec![GUARD; output_len])?),
    ]);
    let call = compiled.call(
        s,
        &lookup.stage().entry().symbol,
        lookup.stage().plan(live).map_err(err)?,
        lookup
            .arguments(kernels_sm120::row::precision::LookupBuffers {
                table: W,
                ids: X,
                local_scales: Some(WS),
                global_scale: Some(WG),
                output: Y,
                live_rows: LIVE,
                row_offset: start,
                shard_rows: rows,
            })
            .map_err(err)?,
    )?;
    let bindings = Bindings {
        buffers: buffers.iter().map(|(id, b)| (*id, b)).collect(),
    };
    let graph = s.capture(|stream| {
        launch(
            &s.context,
            &call.function,
            stream,
            call.plan,
            &call.arguments,
            &bindings,
        )
    })?;
    let mut previous = None;
    for _ in 0..3 {
        buffers[&Y]
            .copy_from_bytes(
                &vec![GUARD; output_len]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
                &s.stream,
            )
            .map_err(err)?;
        graph.launch(&s.stream).map_err(err)?;
        let output = s.download_bf16(&buffers[&Y], output_len)?;
        for (i, &value) in output.iter().enumerate() {
            let expected = if i >= live as usize * heads * width {
                GUARD
            } else {
                let id = ids[i / width];
                if id < start || id >= start + rows {
                    0
                } else {
                    let row = (id - start) as usize;
                    let weight = decoded.decode_with(row, i % width, ScaleProduct::F32);
                    kernel_core::codec::f64_to_bf16(weight)
                }
            };
            if value != expected {
                return Err(format!("lookup element {i}: {value:#x} != {expected:#x}"));
            }
        }
        if previous.as_ref().is_some_and(|p| p != &output) {
            return Err("lookup graph replay differs".into());
        }
        previous = Some(output);
    }
    s.receipt(
        "generated-lookup",
        BTreeMap::from([
            ("usage".into(), json!(usage)),
            ("live".into(), json!(live)),
            ("mapped_host".into(), json!(mapped)),
            ("shard_start".into(), json!(start)),
            ("shard_rows".into(), json!(rows)),
            ("graph_replays".into(), json!(3)),
            ("bitwise".into(), json!(true)),
        ]),
    )?;
    Ok(previous.unwrap()[..live as usize * heads * width].to_vec())
}

#[test]
fn generated_lookup_offers_have_oracles_and_invariance() {
    kernel_lab::gpu_test(512 << 20, |s| {
        let compiled = Compiled(Prepared::read()?);
        let (offers, _, _) = kernel_lab::models::projection_inventory()?;
        let mut tested = 0;
        let selection = projection_cases::Selection::read()?;
        for (case_index, offer) in offers.into_iter().enumerate() {
            if !selection.contains(case_index) {
                continue;
            }
            let Compute::Lookup(_) = &offer.handle().compute else {
                continue;
            };
            let case = projection_cases::typed(&offer)?;
            for live in &case.oracle_rows {
                for mapped in [false, true] {
                    lookup(s, &compiled, &offer, *live, mapped, 1, 0)?;
                }
            }
            let rows = offer.usage().capacity.min(33);
            if rows > 1 {
                let split = rows / 2;
                let whole = lookup(s, &compiled, &offer, rows, false, 1, 0)?;
                let mut chunks = lookup(s, &compiled, &offer, split, false, 1, 0)?;
                chunks.extend(lookup(
                    s,
                    &compiled,
                    &offer,
                    rows - split,
                    false,
                    1,
                    split as usize,
                )?);
                if whole != chunks {
                    return Err("generated lookup chunks differ".into());
                }
            }
            s.receipt(
                "generated-projection",
                BTreeMap::from([
                    ("case_index".into(), json!(case_index)),
                    ("case".into(), json!(case)),
                    ("oracle".into(), json!(true)),
                    ("replay".into(), json!(true)),
                ]),
            )?;
            tested += 1;
        }
        if tested == 0 {
            return Err("no generated lookup offers ran".into());
        }
        Ok(())
    })
    .expect("generated lookup oracle and invariance");
}

#[test]
fn generated_layout_copies_are_exact_and_replay_invariant() {
    kernel_lab::gpu_test(512 << 20, |s| {
        let prepared = Prepared::read()?;
        let (offers, retained, copies) = kernel_lab::models::projection_inventory()?;
        if copies.is_empty() {
            return Err("no generated layout copies".into());
        }
        let selection = projection_cases::Selection::read()?;
        let mut tested = 0;
        for (offset, copy) in copies.into_iter().enumerate() {
            let case_index = offers.len() + retained.len() + offset;
            if !selection.contains(case_index) {
                continue;
            }
            let (columns, source, destination) = copy.layout();
            let capacity = copy.stage().capacity();
            let input_len = copy.source_bytes() as usize / 2;
            let output_len = copy.destination_bytes() as usize / 2 + 128;
            let input = (0..input_len)
                .map(|i| {
                    input(
                        i / source.row_stride as usize,
                        i % source.row_stride as usize,
                    )
                })
                .collect::<Vec<_>>();
            let x = s.upload_bf16(&input)?;
            let y = s.upload_bf16(&vec![GUARD; output_len])?;
            for live in [0, 1, capacity.saturating_sub(1), capacity] {
                let rows = upload(s, &live.to_le_bytes())?;
                let plan = copy.stage().plan(live).map_err(err)?;
                let function = prepared.function(
                    s,
                    &copy.stage().entry().symbol,
                    plan.dynamic_shared_bytes,
                )?;
                let arguments = copy.arguments(X, Y, LIVE).map_err(err)?;
                let bindings = Bindings {
                    buffers: BTreeMap::from([(X, &x), (Y, &y), (LIVE, &rows)]),
                };
                let mut expected = vec![GUARD; output_len];
                for r in 0..live as usize {
                    for c in 0..columns as usize {
                        expected[r * destination.row_stride as usize
                            + c * destination.column_stride as usize
                            + destination.column_offset as usize] = input[r * source.row_stride
                            as usize
                            + c * source.column_stride as usize
                            + source.column_offset as usize];
                    }
                }
                for _ in 0..2 {
                    y.copy_from_bytes(
                        &vec![GUARD; output_len]
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>(),
                        &s.stream,
                    )
                    .map_err(err)?;
                    launch(
                        &s.context, &function, &s.stream, plan, &arguments, &bindings,
                    )?;
                    if s.download_bf16(&y, output_len)? != expected {
                        return Err("generated layout copy differs".into());
                    }
                }
            }
            tested += 1;
            s.receipt(
                "generated-copy",
                BTreeMap::from([
                    ("case_index".into(), json!(case_index)),
                    ("symbol".into(), json!(copy.stage().entry().symbol)),
                    ("capacity".into(), json!(capacity)),
                    ("exact".into(), json!(true)),
                ]),
            )?;
        }
        if tested == 0 {
            return Err("no selected layout copy ran".into());
        }
        Ok(())
    })
    .expect("generated exact copies");
}

/// How an exactness case routes token `t`'s slot `j`: every token to the
/// same experts, so each expert's rows share its tiles, or each token
/// shifted to experts of its own.
#[derive(Clone, Copy, Debug)]
enum Routing {
    Shared,
    Shifted,
}

impl Routing {
    fn expert(self, token: usize, slot: usize, experts: usize, top_k: usize) -> usize {
        let spread = slot * (experts / top_k);
        match self {
            Routing::Shared => spread % experts,
            Routing::Shifted => (spread + token * 5) % experts,
        }
    }
}

/// The routed output rows of one launch of `handle` over the tokens
/// `tokens` (global ids, which seed each token's inputs across 13 binades
/// and its route weights), routed by `routing`, by token: each live
/// element written and finite.
fn grouped_rows(
    s: &Session,
    compiled: &Compiled,
    handle: &OperationHandle,
    weight: &Weights,
    tokens: &[usize],
    routing: Routing,
) -> LabResult<Vec<Vec<u16>>> {
    let (top_k, experts, width) = (weight.top_k, weight.experts, weight.width());
    let seeds: Vec<u64> = if weight.swiglu {
        tokens.iter().map(|&t| 0x7000 + t as u64).collect()
    } else {
        tokens
            .iter()
            .flat_map(|&t| (0..top_k).map(move |j| 0x9000 + (t * top_k + j) as u64))
            .collect()
    };
    let case = Routed {
        x: seeds
            .iter()
            .flat_map(|&seed| kernel_lab::wide_values(weight.k, seed))
            .map(f32_to_bf16)
            .collect(),
        routes: tokens
            .iter()
            .flat_map(|&t| (0..top_k).map(move |j| routing.expert(t, j, experts, top_k)))
            .collect(),
        route_weights: tokens
            .iter()
            .flat_map(|&t| (0..top_k).map(move |j| 0.3125 + ((t * 7 + j * 3) % 11) as f32 / 37.0))
            .collect(),
    };
    let routed = tokens.len() * top_k * width;
    let actual = launch_routed(
        s,
        compiled,
        handle,
        weight,
        tokens.len() as u32,
        &case,
        false,
    )?;
    if actual[..routed]
        .iter()
        .any(|v| *v == GUARD || !bf16_to_f32(*v).is_finite())
    {
        return Err("a live grouped element is unwritten or not finite".into());
    }
    Ok(actual[..routed]
        .chunks(width * top_k)
        .map(<[u16]>::to_vec)
        .collect())
}
