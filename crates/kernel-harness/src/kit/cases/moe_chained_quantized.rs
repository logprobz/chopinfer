//! Quantized chained MoE gates: score accuracy and exact row-bias
//! selection are separate checks, over the same poisoned replay.

use std::cell::RefCell;
use std::rc::Rc;

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{DType, MoeQuery, Refusal};
use kernel_core::moe::{RouteBuffers, RouteMetadata, RouterBuffers};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch, PreparedExperts};
use kernel_core::oracle::{OracleOutput, RouterSelection, check_numeric};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::router::{RowBiasRouteBuffers, RowBiasRouterBuffers, RowBiasRouterQuery};
use kernel_core::router_oracle::{row_bias_router, select_f32_scores, sqrt_softplus_scores};
use kernels_sm120::moe::chained::quantized::Projection;

use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec};
use crate::poison::WriteMap;

mod gate_up;
pub use gate_up::{gate_up, gate_up_case, gate_up_unit_weight_control};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_prefix(bytes: &[u8], elements: usize) -> LabResult<Vec<f32>> {
    let bytes = bytes.get(..elements * 4).ok_or("short F32 output")?;
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

/// The first judge checks F32 scores against f64 and records the exact
/// selection they imply. The following judges check ids and weights bit
/// for bit. `Case::judge` executes the returned judges in this order,
/// once per comparison, so no result is reused from an earlier replay.
fn router_expected(
    query: RowBiasRouterQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<(String, Expected)>> {
    let elements = live as usize * query.experts as usize;
    let logits = images.f32s("router.logits")?;
    let biases = [images.f32s("router.bias")?, images.f32s("alternate_bias")?];
    let selectors = images.bytes("bias_select")?[..live as usize].to_vec();
    let ideal = if live == 0 {
        vec![]
    } else {
        sqrt_softplus_scores(&logits[..elements]).map_err(|e| e.to_string())?
    };
    // Sixteen F32 unit roundoffs cover exp, log1p, add and square root
    // over the finite moderate logits of this case. The selection gate
    // has no tolerance and never substitutes the f64 scores.
    let reference = OracleOutput::new(
        ideal.clone(),
        ideal.iter().map(|v| v.abs() * 4.0).collect(),
        1,
    )
    .map_err(|e| e.to_string())?;
    let selected = Rc::new(RefCell::new(None::<Vec<RouterSelection>>));
    let scores_selected = Rc::clone(&selected);
    let scores = move |bytes: &[u8]| -> LabResult<()> {
        let scores = f32_prefix(bytes, elements)?;
        check_numeric(
            &scores.iter().copied().map(f64::from).collect::<Vec<_>>(),
            &reference,
            DType::F32,
        )
        .map_err(|e| format!("sqrt-softplus score accuracy: {e:?}"))?;
        let routes = if live == 0 {
            vec![]
        } else {
            select_f32_scores(query, &scores, [&biases[0], &biases[1]], &selectors)
                .map_err(|e| e.to_string())?
        };
        *scores_selected.borrow_mut() = Some(routes);
        Ok(())
    };
    let ids_selected = Rc::clone(&selected);
    let ids = move |bytes: &[u8]| -> LabResult<()> {
        let held = ids_selected.borrow();
        let expected = held
            .as_ref()
            .ok_or("ids were judged before the router scores")?
            .iter()
            .flat_map(|r| {
                r.expert_ids
                    .iter()
                    .flat_map(|&id| (id as i32).to_le_bytes())
            })
            .collect::<Vec<_>>();
        if bytes.get(..expected.len()) != Some(expected.as_slice()) {
            return Err("router ids differ from exact top-k over its own F32 scores".into());
        }
        Ok(())
    };
    let weights = move |bytes: &[u8]| -> LabResult<()> {
        let held = selected.borrow();
        let expected = held
            .as_ref()
            .ok_or("weights were judged before the router scores")?
            .iter()
            .flat_map(|r| r.weights.iter().flat_map(|&w| (w as f32).to_le_bytes()))
            .collect::<Vec<_>>();
        if bytes.get(..expected.len()) != Some(expected.as_slice()) {
            return Err("router weights differ from unbiased F32 normalization".into());
        }
        Ok(())
    };
    Ok(vec![
        ("scores".into(), Expected::Judge(Box::new(scores))),
        ("router.ids".into(), Expected::Judge(Box::new(ids))),
        ("router.weights".into(), Expected::Judge(Box::new(weights))),
    ])
}

pub fn router_case<'a>(
    query: RowBiasRouterQuery,
    capacity: u32,
    prepared: Prepared<RowBiasRouterBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (experts, top_k) = (query.experts as usize, query.top_k as usize);
    let logits = (0..capacity as usize)
        .flat_map(|row| {
            (0..experts).map(move |e| {
                if row % 3 == 1 {
                    0.0
                } else {
                    ((row * 13 + e * 7) % 65) as f32 / 8.0 - 4.0
                }
            })
        })
        .collect::<Vec<_>>();
    // At 2^25 the bias also creates ties after F32 addition that a
    // selector using f64 addition would break differently.
    let bias = (0..experts)
        .map(|e| if e < top_k { 33_554_432.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let alternate = (0..experts)
        .map(|e| {
            if e >= experts - top_k {
                33_554_432.0
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let roles = RowBiasRouterBuffers {
        router: RouterBuffers {
            logits: Spec::input(DType::F32, (Rows::Live, experts), f32_bytes(&logits)),
            bias: Some(Spec::input(
                DType::F32,
                (Rows::Fixed(1), experts),
                f32_bytes(&bias),
            )),
            ids: Spec::output(DType::I32, (Rows::Live, top_k)),
            weights: Spec::output(DType::F32, (Rows::Live, top_k)),
            live_rows: Spec::live(),
        },
        alternate_bias: Spec::input(DType::F32, (Rows::Fixed(1), experts), f32_bytes(&alternate)),
        bias_select: Spec::input(
            DType::U8,
            (Rows::Live, 1),
            (0..capacity).map(|r| (r % 2) as u8).collect(),
        ),
        scores: Spec::output(DType::F32, (Rows::Live, experts)),
    };
    Ok(OpCase::new(prepared, capacity, roles)?
        .oracle(move |images, live| router_expected(query, images, live)))
}

pub fn router<'a>(query: RowBiasRouterQuery, capacity: u32) -> Required<'a> {
    Required::new(
        format!(
            "sqrt-softplus row bias {} top {}",
            query.experts, query.top_k
        ),
        Need::rows(capacity),
        move |backend, need| backend.moe_row_bias_router(&query, need),
        move |prepared| router_case(query, capacity, prepared),
    )
    .required()
}

kernel_core::scratch_record! {
    #[derive(Clone, Copy, Debug)]
    pub struct RoutedBuffers {
        pub route: record(RowBiasRouteBuffers),
        pub activation: value,
        pub activation_scales: value,
        pub weights: record(QuantizedWeights),
        pub output: value,
        pub output_codes: option,
        pub output_scales: option,
        pub quantizer_status: option,
        pub quantizer_status_row_offset: scalar(u64),
        pub unit_weights: option,
        pub unused: value,
    }
}

struct Routed {
    route: Prepared<RowBiasRouteBuffers<View>>,
    down: Box<dyn PreparedExperts>,
    info: OpInfo,
}

impl Routed {
    fn new(
        route: Prepared<RowBiasRouteBuffers<View>>,
        down: Box<dyn PreparedExperts>,
    ) -> LabResult<Self> {
        let mut symbols = route.info().symbols.clone();
        symbols.extend(down.info().symbols.clone());
        let mut info = OpInfo::new(
            "kit.quantized-routed-down",
            route.info().guarantees.meet(down.info().guarantees),
            symbols,
        );
        for (path, bytes) in &route.info().bytes {
            info.bytes.push((format!("route.{path}"), *bytes));
        }
        for (path, bytes) in &down.info().bytes {
            let path = match path.as_str() {
                "live_tokens" => "route.route.live_rows".into(),
                "routing.weights" => "route.route.weights".into(),
                "scratch.output_codes" => "output_codes".into(),
                "scratch.output_scales" => "output_scales".into(),
                "scratch.quantizer_status" => "quantizer_status".into(),
                _ if path.starts_with("routing.routes.") => {
                    path.replacen("routing.routes.", "route.route.routes.", 1)
                }
                _ => path.clone(),
            };
            if let Some((_, size)) = info.bytes.iter_mut().find(|(p, _)| *p == path) {
                *size = (*size).max(*bytes);
            } else {
                info.bytes.push((path, *bytes));
            }
        }
        if route.info().workspace != 0 || down.info().workspace != 0 {
            return Err("routed case expects no unnamed workspace".into());
        }
        info.weight_layouts = down.info().weight_layouts.clone();
        Ok(Self { route, down, info })
    }
}

impl PreparedOp<RoutedBuffers<View>> for Routed {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &RoutedBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let r = b.route.route;
        let down = GatedMlpBuffers {
            activation: b.activation,
            activation_scales: Some(b.activation_scales),
            activation_global: None,
            live_tokens: r.live_rows,
            output: b.output,
            weights: b.weights,
            routing: Some(ProjectionRouting {
                weights: r.weights,
                routes: r.routes,
            }),
            scratch: GatedMlpScratch {
                quantizer_status: b.quantizer_status,
                quantizer_status_row_offset: b.quantizer_status_row_offset,
                partial: None,
                output_codes: b.output_codes,
                output_scales: b.output_scales,
                unused: b.unused,
            },
        };
        let mut launches = self.route.launches(&b.route, workspace, counts)?;
        launches.extend(self.down.launches(&down, workspace, counts)?);
        Ok(launches)
    }
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn routing_images(images: &Images) -> LabResult<Images> {
    let mut out = Images::default();
    for (from, to) in [
        ("route.route.logits", "router.logits"),
        ("route.route.bias", "router.bias"),
        ("route.alternate_bias", "alternate_bias"),
        ("route.bias_select", "bias_select"),
    ] {
        out = out.with(to, images.bytes(from)?.to_vec());
    }
    Ok(out)
}

fn selections(
    query: RowBiasRouterQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<RouterSelection>> {
    if live == 0 {
        return Ok(vec![]);
    }
    let image = routing_images(images)?;
    let logits = image.f32s("router.logits")?;
    row_bias_router(
        query,
        &logits[..live as usize * query.experts as usize],
        [&image.f32s("router.bias")?, &image.f32s("alternate_bias")?],
        &image.bytes("bias_select")?[..live as usize],
    )
    .map_err(|e| e.to_string())
}

fn route_metadata(
    query: RowBiasRouterQuery,
    pad: usize,
    images: &Images,
    live: u32,
) -> LabResult<kernel_core::gated_mlp_oracle::RouteMetadata> {
    let ids = selections(query, images, live)?
        .iter()
        .flat_map(|r| r.expert_ids.clone())
        .collect::<Vec<_>>();
    kernel_core::gated_mlp_oracle::local_route_metadata(
        &ids,
        query.top_k as usize,
        query.experts as usize,
        0,
        pad,
    )
    .map_err(|e| e.to_string())
}

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

fn down_weight_pair(expert: usize, column: usize, block: usize) -> u8 {
    let codes = [2_u8, 4, 10, 12];
    let index = (expert + column + block) % codes.len();
    codes[index] | codes[(index + 1) % codes.len()] << 4
}

fn down_activation_code(route: usize, block: usize, column: usize) -> u8 {
    [0x38, 0x40, 0xb8, 0xc0][(route + block + column % 2) % 4]
}

struct DownCache {
    images: Vec<Vec<u8>>,
    experts: Vec<usize>,
    output: Vec<f32>,
}

/// Exact dyadic inputs keep every 128-K chain and its fold representable
/// in F32. Neighboring stored rows alternate scales and signs. Entire
/// unselected experts and guard rows carry trapping scale bytes.
pub fn down_case<'a>(
    projection: &Projection,
    capacity: u32,
    prepared: Prepared<RoutedBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    if projection.gate_up() {
        return Err("down case needs a down projection".into());
    }
    // A negative control that drops a route producer must reject its
    // poisoned metadata before the next kernel consumes it as indices.
    let guards = match prepared.info().symbols.as_slice() {
        [decode, _] => vec![(decode.0.clone(), "route.route.routes.padded_rows")],
        [select, count, scatter, _] => vec![
            (select.0.clone(), "route.route.ids"),
            (count.0.clone(), "route.route.routes.counts"),
            (scatter.0.clone(), "route.route.routes.padded_rows"),
        ],
        _ => return Err("routed down case needs decode or select/count/scatter routes".into()),
    };
    let g = projection.query.projection.gemm;
    let grouped = g.grouped.ok_or("down case requires routed experts")?;
    let query = RowBiasRouterQuery {
        experts: grouped.experts,
        top_k: grouped.top_k,
        scaling_factor: 1.5,
    };
    let (experts, top_k, n, k) = (
        grouped.experts as usize,
        grouped.top_k as usize,
        g.n as usize,
        g.k as usize,
    );
    let tokens = capacity as usize;
    let pad = projection.row_tile as usize;
    let sorted = prepared
        .info()
        .bytes_of("route.route.routes.sorted_rows")
        .ok_or("sorted row allocation missing")? as usize
        / 4;
    let logits = (0..tokens)
        .flat_map(|t| (0..experts).map(move |e| ((t + e) % 5) as f32 / 16.0))
        .collect::<Vec<_>>();
    let first = (0..experts)
        .map(|e| {
            if e < top_k {
                20.0 + (top_k - e) as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let second = (0..experts)
        .map(|e| {
            if e >= experts - top_k {
                20.0 + (experts - e) as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let mut weights = vec![0xff; experts * n * k / 2 + k / 2];
    let mut scales = vec![255; experts * n * k / 32 + k / 32];
    for expert in (0..top_k).chain(experts - top_k..experts) {
        for col in 0..n {
            for block in 0..k / 32 {
                let pair = down_weight_pair(expert, col, block);
                let at = (expert * n + col) * k / 2 + block * 16;
                weights[at..at + 16].fill(pair);
                scales[(expert * n + col) * (k / 32) + block] = 125 + ((col + block) % 4) as u8;
            }
        }
    }
    let mut input = vec![0; tokens * top_k * k];
    let mut input_scales = vec![0; tokens * top_k * k / 32];
    for route in 0..tokens * top_k {
        for block in 0..k / 32 {
            for col in 0..32 {
                input[route * k + block * 32 + col] = down_activation_code(route, block, col);
            }
            input_scales[route * (k / 32) + block] = 125 + ((route + block) % 4) as u8;
        }
    }
    let sorted_rows = Spec::output_with(
        DType::I32,
        (Rows::Fixed(1), sorted),
        Write::Map(Box::new(move |images, live| {
            WriteMap::of(DType::I32)
                .prefix(route_metadata(query, pad, images, live)?.num_padded_rows as usize)
        })),
    )
    .filled_with(&[0x80])
    .route_metadata();
    let active = Spec::output_with(
        DType::I32,
        (Rows::Fixed(1), experts + 1),
        Write::EveryCount(Box::new(move |images, live| {
            WriteMap::of(DType::I32).prefix(
                route_metadata(query, pad, images, live)?
                    .active_experts
                    .len(),
            )
        })),
    )
    .route_metadata();
    let roles = RoutedBuffers {
        route: RowBiasRouteBuffers {
            route: RouteBuffers {
                logits: Spec::input(DType::F32, (Rows::Live, experts), f32_bytes(&logits)),
                bias: Some(Spec::input(
                    DType::F32,
                    (Rows::Fixed(1), experts),
                    f32_bytes(&first),
                )),
                ids: Spec::output(DType::I32, (Rows::Live, top_k)),
                weights: Spec::output(DType::F32, (Rows::Live, top_k)),
                routes: RouteMetadata {
                    counts: Some(whole(experts)),
                    offsets: whole(experts + 1),
                    sorted_rows,
                    padded_rows: whole(1),
                    active_experts: Some(active),
                },
                tile_counts: None,
                tile_offsets: None,
                ticket: None,
                live_rows: Spec::live(),
            },
            alternate_bias: Spec::input(DType::F32, (Rows::Fixed(1), experts), f32_bytes(&second)),
            bias_select: Spec::input(
                DType::U8,
                (Rows::Live, 1),
                (0..capacity).map(|r| (r % 2) as u8).collect(),
            ),
            scores: Spec::output(DType::F32, (Rows::Live, experts)),
        },
        activation: Spec::input(DType::U8, (Rows::Live, top_k * k), input),
        activation_scales: Spec::input(DType::U8, (Rows::Live, top_k * k / 32), input_scales),
        weights: QuantizedWeights {
            codes: Spec::input(DType::U8, (Rows::Fixed(1), weights.len()), weights),
            local_scales: Some(Spec::input(
                DType::U8,
                (Rows::Fixed(1), scales.len()),
                scales,
            )),
            global_scales: None,
        },
        output: Spec::output(DType::F32, (Rows::Live, top_k * n)),
        output_codes: None,
        output_scales: None,
        quantizer_status: None,
        quantizer_status_row_offset: 0,
        unit_weights: None,
        unused: Spec::scratch(4),
    };
    let cache = RefCell::new(None::<DownCache>);
    let chain_k = projection_k(k)?;
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let aliases = routing_images(images)?;
        let mut expected = router_expected(query, &aliases, live)?
            .into_iter()
            .map(|(path, result)| {
                let path = if path.starts_with("router.") {
                    path.replacen("router.", "route.route.", 1)
                } else {
                    format!("route.{path}")
                };
                (path, result)
            })
            .collect::<Vec<_>>();
        let meta = route_metadata(query, pad, images, live)?;
        let mut sorted_values = meta.sorted_rows;
        sorted_values.resize(sorted, 0);
        let mut active_values = meta.active_experts;
        active_values.resize(experts + 1, 0);
        for (path, values) in [
            ("counts", meta.counts),
            ("offsets", meta.offsets),
            ("sorted_rows", sorted_values),
            ("padded_rows", vec![meta.num_padded_rows]),
            ("active_experts", active_values),
        ] {
            expected.push((
                format!("route.route.routes.{path}"),
                Expected::Bits(i32_bytes(&values)),
            ));
        }
        let selections = selections(query, images, capacity)?;
        let a = images.bytes("activation")?;
        let a_scales = images.bytes("activation_scales")?;
        let w = images.bytes("weights.codes")?;
        let w_scales = images.bytes("weights.local_scales")?;
        let ids = selections
            .iter()
            .flat_map(|r| r.expert_ids.clone())
            .collect::<Vec<_>>();
        let operands = [a, a_scales, w, w_scales];
        let reuse = cache.borrow().as_ref().is_some_and(|c| {
            c.experts == ids
                && c.images
                    .iter()
                    .zip(operands)
                    .all(|(old, new)| old.as_slice() == new)
        });
        if !reuse {
            let mut output = vec![0.0_f32; tokens * top_k * n];
            for (token, selection) in selections.iter().enumerate() {
                for (slot, &expert) in selection.expert_ids.iter().enumerate() {
                    let route = token * top_k + slot;
                    for col in 0..n {
                        let mut x = Vec::with_capacity(k);
                        let mut weight = Vec::with_capacity(k);
                        for kk in 0..k {
                            let ac = a[route * k + kk];
                            let wc = w[(expert * n + col) * (k / 2) + kk / 2];
                            let wc = if kk % 2 == 0 { wc & 15 } else { wc >> 4 };
                            x.push(
                                kernel_core::codec::SmallFloat::E4M3.decode_f64(ac)
                                    * kernel_core::codec::decode_ue8m0_f64(
                                        a_scales[route * (k / 32) + kk / 32],
                                    ),
                            );
                            weight.push(
                                kernel_core::codec::SmallFloat::E2M1.decode_f64(wc)
                                    * kernel_core::codec::decode_ue8m0_f64(
                                        w_scales[(expert * n + col) * (k / 32) + kk / 32],
                                    ),
                            );
                        }
                        let chains =
                            kernel_core::moe_chained_oracle::partial_chains(&x, &weight, chain_k)
                                .map_err(|e| e.to_string())?;
                        output[route * n + col] = kernel_core::moe_chained_oracle::fold_f32_chains(
                            &chains.values.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                        )
                        .map_err(|e| e.to_string())?;
                    }
                }
            }
            *cache.borrow_mut() = Some(DownCache {
                images: operands.iter().map(|b| b.to_vec()).collect(),
                experts: ids,
                output,
            });
        }
        expected.push((
            "output".into(),
            Expected::Bits(f32_bytes(&cache.borrow().as_ref().unwrap().output)),
        ));
        Ok(expected)
    };
    let mut case = OpCase::new(prepared, capacity, roles)?.oracle(oracle);
    for (symbol, output) in guards {
        case = case.omission_guard(&symbol, output);
    }
    Ok(case)
}

fn projection_k(k: usize) -> LabResult<kernel_core::moe_chained::ChainedK> {
    kernel_core::moe_chained::ChainedK::new(k as u32).map_err(|e| e.to_string())
}

pub fn down<'a>(projection: Projection, capacity: u32) -> Required<'a> {
    let query = projection.query;
    let g = query.projection.gemm.grouped.expect("routed case");
    let router = RowBiasRouterQuery {
        experts: g.experts,
        top_k: g.top_k,
        scaling_factor: 1.5,
    };
    let routes = MoeQuery::LocalRoutes {
        experts: g.experts,
        local_experts: g.experts,
        top_k: g.top_k,
        capacity,
    };
    let usage = OperationUse {
        phase: if projection.row_tile == 16 {
            ExecutionPhase::Decode
        } else {
            ExecutionPhase::Prefill
        },
        capacity,
        graph: false,
    };
    Required::new(
        format!(
            "quantized chained down e{} top{} tile{}",
            g.experts, g.top_k, projection.row_tile
        ),
        Need::rows(capacity),
        move |backend, need| {
            let experts = backend.chained_mlp_projection(&query, usage, need)?;
            let route = backend.moe_row_bias_route(
                &router,
                &routes,
                0,
                experts.route_tile(),
                usage,
                need,
            )?;
            let prepared: Prepared<RoutedBuffers<View>> =
                Box::new(Routed::new(route, experts).map_err(Refused::Rule)?);
            Ok(prepared)
        },
        move |prepared| down_case(&projection, capacity, prepared),
    )
    .required()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernels_sm120::backend::Sm120;
    use kernels_sm120::moe::chained::row_bias::Router;

    #[test]
    fn down_fixture_rejects_swapped_packed_k_nibbles() {
        use kernel_core::codec::SmallFloat;
        for index in 0..4 {
            let pair = down_weight_pair(index, 0, 0);
            let low = SmallFloat::E2M1.decode_f64(pair & 15);
            let high = SmallFloat::E2M1.decode_f64(pair >> 4);
            let x = SmallFloat::E4M3.decode_f64(down_activation_code(index, 0, 0));
            let y = SmallFloat::E4M3.decode_f64(down_activation_code(index, 0, 1));
            assert_ne!(
                low * x + high * y,
                high * x + low * y,
                "the fixture must distinguish packed K neighbors"
            );
        }
    }

    #[test]
    fn own_score_cases_declare_for_both_router_geometries() {
        for (experts, top_k) in [(384, 6), (128, 3)] {
            let query = RowBiasRouterQuery {
                experts,
                top_k,
                scaling_factor: 1.5,
            };
            let backend = Sm120::new().with_row_bias_router(Router {
                experts,
                top_k,
                capacity: 32,
            });
            let op = backend.moe_row_bias_router(&query, Need::rows(32)).unwrap();
            let case = router_case(query, 32, op).unwrap();
            case.validate().unwrap();
        }
    }

    #[test]
    fn own_score_judges_reject_reversed_f32_ties_and_biased_weights() {
        let query = RowBiasRouterQuery {
            experts: 4,
            top_k: 2,
            scaling_factor: 1.5,
        };
        let logits = [0.54132485, 0.5413253, 0.0, 0.0];
        let bias = [33_554_432.0, 33_554_432.0, 0.0, 0.0];
        let scores = sqrt_softplus_scores(&logits)
            .unwrap()
            .into_iter()
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let selection = select_f32_scores(query, &scores, [&bias, &bias], &[0]).unwrap();
        assert_eq!(selection[0].expert_ids, [0, 1]);
        let images = Images::default()
            .with("router.logits", f32_bytes(&logits))
            .with("router.bias", f32_bytes(&bias))
            .with("alternate_bias", f32_bytes(&bias))
            .with("bias_select", vec![0]);
        let expected = router_expected(query, &images, 1).unwrap();
        let judges = expected
            .into_iter()
            .map(|(_, expected)| match expected {
                Expected::Judge(judge) => judge,
                _ => panic!("linked score judges"),
            })
            .collect::<Vec<_>>();
        judges[0](&f32_bytes(&scores)).unwrap();
        judges[1](
            &[0_i32, 1]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(
            judges[1](
                &[1_i32, 0]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>()
            )
            .is_err()
        );
        let correct = selection[0]
            .weights
            .iter()
            .map(|&v| v as f32)
            .collect::<Vec<_>>();
        judges[2](&f32_bytes(&correct)).unwrap();
        assert!(judges[2](&f32_bytes(&[0.5, 0.5])).is_err());
        assert!(judges[0](&f32_bytes(&[0.0; 4])).is_err());
    }
}
