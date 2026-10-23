//! F32 gate/up epilogues, then exact codes and scales over the checked
//! BF16 output. The quantizer is the shared backend operation.
use super::*;
use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64};
use kernel_core::gated_mlp::SwigluClamp;
use kernel_core::moe_chained_oracle::{
    before_down_f64, fold_f32_chains, partial_chains, quantize_before_down,
};
use kernel_core::precision::WeightLayout;
use kernel_core::precision::layout::ROW_TILES_64X128;
use kernel_core::quantize::Scales;
use kernel_core::transform::RowTileLayout;

fn decoded(codes: &[u8], scales: &[u8], row: usize, k: usize, packed: bool) -> Vec<f64> {
    (0..k)
        .map(|col| {
            let code = if packed {
                let byte = codes[row * (k / 2) + col / 2];
                if col % 2 == 0 { byte & 15 } else { byte >> 4 }
            } else {
                codes[row * k + col]
            };
            (if packed {
                SmallFloat::E2M1
            } else {
                SmallFloat::E4M3
            })
            .decode_f64(code)
                * decode_ue8m0_f64(scales[row * (k / 32) + col / 32])
        })
        .collect()
}

fn exact_dot(x: &[f64], w: &[f64]) -> LabResult<f64> {
    let chains = partial_chains(x, w, projection_k(x.len())?).map_err(|e| e.to_string())?;
    if chains.values.iter().any(|&v| f64::from(v as f32) != v) {
        return Err("gate/up fixture needs exact dyadic F32 partial chains".into());
    }
    fold_f32_chains(&chains.values.iter().map(|&v| v as f32).collect::<Vec<_>>())
        .map(f64::from)
        .map_err(|e| e.to_string())
}

fn rounded_epilogue_judge(
    gates: &[(f64, f64)],
    weights: &[f32],
    width: usize,
    clamp: Option<SwigluClamp>,
    bytes: &[u8],
) -> LabResult<Vec<f32>> {
    let values = bytes
        .chunks_exact(2)
        .map(|v| bf16_to_f32(u16::from_le_bytes([v[0], v[1]])))
        .collect::<Vec<_>>();
    if values.len() != gates.len() {
        return Err("gate/up BF16 output length differs".into());
    }
    let ideal = gates
        .iter()
        .enumerate()
        .map(|(i, &(g, u))| {
            before_down_f64(g, u, f64::from(weights[i / width]), clamp).map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    // The dyadic chain results are exact. Sixteen F32 unit roundoffs
    // cover exp, add, divide and the two products for these finite inputs;
    // rounding_admissible checks only the BF16 roundings of that interval.
    let reference = OracleOutput::new(
        ideal.clone(),
        ideal.iter().map(|v| 4.0 * v.abs()).collect(),
        1,
    )
    .map_err(|e| e.to_string())?;
    crate::conformance::rounding_admissible(DType::Bf16, &values, &reference)
        .map_err(|error| format!("gate/up BF16 oracle: {error}"))?;
    Ok(values)
}

fn weight_payload(experts: usize, n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut codes = vec![0; experts * 2 * n * k / 2];
    let mut scales = vec![127; experts * 2 * n * k / 32];
    // Keep the base and small product that exercise the F32 epilogue.
    // Every later 128-K chain and the last stored block also contribute;
    // all products and partial chains remain exact dyadic F32 values.
    let bases = [
        (14, 127),
        (10, 127),
        (2, 126),
        (2, 127),
        (6, 127),
        (7, 128),
        (15, 129),
    ];
    for expert in 0..experts {
        for row in 0..2 * n {
            let col = row % n;
            let pattern = (expert + col / 32 + usize::from(row >= n)) % bases.len();
            let (code, scale) = bases[pattern];
            let global = expert * 2 * n + row;
            codes[global * k / 2] = code;
            scales[global * k / 32] = scale;
            codes[global * k / 2 + 16] = [2, 3, 5, 7][col % 4];
            scales[global * k / 32 + 1] = 117;
            for chain in (128..k).step_by(128) {
                codes[global * k / 2 + chain / 2] = [2, 3, 4, 5][(row + chain / 128) % 4];
                scales[global * k / 32 + chain / 32] = 119 + (row % 2) as u8;
            }
            // Use the high nibble at the final K position as well as the
            // low nibbles above, so a truncated last tile loses a product.
            codes[global * k / 2 + k / 2 - 1] = 4 << 4;
            if k > 64 {
                scales[global * k / 32 + k / 32 - 1] = 124;
            }
        }
    }
    (codes, scales)
}

/// The tiles of the bank of `rows` MXFP4 rows of `k` elements the op reads
/// (`OpInfo::weight_layouts`, role `weights`): none for rows, or 64-row by
/// 128-K tiles of its codes and its scales ([`ROW_TILES_64X128`]).
fn bank_tiles(info: &OpInfo, rows: usize, k: usize) -> LabResult<Option<[RowTileLayout; 2]>> {
    match info.layout_of("weights") {
        None => Ok(None),
        Some(layout) if *layout == WeightLayout::row_major() => Ok(None),
        Some(layout) if layout.id == ROW_TILES_64X128 && layout.version == 1 => {
            let tiles = |row_bytes, tile_bytes| RowTileLayout {
                rows,
                row_bytes,
                tile_rows: 64,
                tile_bytes,
            };
            Ok(Some([tiles(k / 2, 64), tiles(k / 32, 4)]))
        }
        Some(layout) => Err(format!(
            "gate/up case cannot store its bank in {} v{}",
            layout.id, layout.version
        )),
    }
}

/// Row-major bytes of a section stored in `tiles`.
fn tile_rows(tiles: RowTileLayout, stored: &[u8]) -> LabResult<Vec<u8>> {
    if stored.len() != tiles.rows * tiles.row_bytes {
        return Err("row-tiled bank section size differs".into());
    }
    let mut rows = vec![0; stored.len()];
    for (row, out) in rows.chunks_exact_mut(tiles.row_bytes).enumerate() {
        for (tile, chunk) in out.chunks_exact_mut(tiles.tile_bytes).enumerate() {
            let at = tiles.offset(row, tile * tiles.tile_bytes);
            chunk.copy_from_slice(&stored[at..at + tiles.tile_bytes]);
        }
    }
    Ok(rows)
}

fn activation_payload(tokens: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let codes = (0..tokens)
        .flat_map(|token| {
            (0..k).map(move |col| [0x30, 0x34, 0x38, 0x3c][(token + 3 * (col % 2) + col / 32) % 4])
        })
        .collect();
    let scales = (0..tokens)
        .flat_map(|token| (0..k / 32).map(move |block| 126 + ((token / 4 + block) % 3) as u8))
        .collect();
    (codes, scales)
}

pub fn gate_up_case<'a>(
    projection: &Projection,
    capacity: u32,
    prepared: Prepared<RoutedBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    if !projection.gate_up() {
        return Err("gate/up case needs gate/up projection".into());
    }
    let g = projection.query.projection.gemm;
    let grouped = g.grouped.ok_or("gate/up requires routes")?;
    let query = RowBiasRouterQuery {
        experts: grouped.experts,
        top_k: grouped.top_k,
        scaling_factor: 1.5,
    };
    let (e, top, n, k, tokens) = (
        grouped.experts as usize,
        grouped.top_k as usize,
        projection.columns() as usize,
        g.k as usize,
        capacity as usize,
    );
    let pad = projection.row_tile as usize;
    let sorted = prepared
        .info()
        .bytes_of("route.route.routes.sorted_rows")
        .ok_or("sorted allocation missing")? as usize
        / 4;
    let compute = prepared
        .info()
        .symbols
        .iter()
        .position(|(s, _)| s == &projection.symbol)
        .ok_or("gate/up compute stage missing")?;
    let mut guards = match compute {
        1 => vec![(
            prepared.info().symbols[0].0.clone(),
            "route.route.routes.padded_rows",
        )],
        3 => vec![
            (prepared.info().symbols[0].0.clone(), "route.route.ids"),
            (
                prepared.info().symbols[1].0.clone(),
                "route.route.routes.counts",
            ),
            (
                prepared.info().symbols[2].0.clone(),
                "route.route.routes.padded_rows",
            ),
        ],
        _ => return Err("gate/up case needs decode or select/count/scatter routes".into()),
    };
    guards.push((projection.symbol.clone(), "output"));
    let logits = (0..tokens)
        .flat_map(|t| (0..e).map(move |expert| ((t + expert) % 5) as f32 / 16.0))
        .collect::<Vec<_>>();
    let bias = (0..e)
        .map(|expert| {
            if expert < top {
                20.0 + (top - expert) as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let alternate = (0..e)
        .map(|expert| {
            if expert >= e - top {
                20.0 + (e - expert) as f32
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let (mut codes, mut scales) = weight_payload(e, n, k);
    // A bank the op reads in row tiles is stored tiled; the oracle reads
    // its rows back from the stored images.
    let tiles = bank_tiles(prepared.info(), e * 2 * n, k)?;
    if let Some(tiles) = tiles {
        let mut scratch = Vec::new();
        for (bytes, tiles) in [&mut codes, &mut scales].into_iter().zip(tiles) {
            tiles
                .pack_in_place(bytes, &mut scratch)
                .map_err(|e| e.to_string())?;
        }
    }
    let (input, input_scales) = activation_payload(tokens, k);
    let unit_weights = prepared.info().bytes_of("unit_weights").map(|_| {
        Spec::input(
            DType::F32,
            (Rows::Live, top),
            f32_bytes(&vec![1.0; tokens * top]),
        )
    });
    let roles = RoutedBuffers {
        route: RowBiasRouteBuffers {
            route: RouteBuffers {
                logits: Spec::input(DType::F32, (Rows::Live, e), f32_bytes(&logits)),
                bias: Some(Spec::input(
                    DType::F32,
                    (Rows::Fixed(1), e),
                    f32_bytes(&bias),
                )),
                ids: Spec::output(DType::I32, (Rows::Live, top)),
                weights: Spec::output(DType::F32, (Rows::Live, top)),
                routes: RouteMetadata {
                    counts: Some(whole(e)),
                    offsets: whole(e + 1),
                    sorted_rows: Spec::output_with(
                        DType::I32,
                        (Rows::Fixed(1), sorted),
                        Write::Map(Box::new(move |images, live| {
                            WriteMap::of(DType::I32).prefix(
                                route_metadata(query, pad, images, live)?.num_padded_rows as usize,
                            )
                        })),
                    )
                    .filled_with(&[0x80])
                    .route_metadata(),
                    padded_rows: whole(1),
                    active_experts: Some(
                        Spec::output_with(
                            DType::I32,
                            (Rows::Fixed(1), e + 1),
                            Write::EveryCount(Box::new(move |images, live| {
                                WriteMap::of(DType::I32).prefix(
                                    route_metadata(query, pad, images, live)?
                                        .active_experts
                                        .len(),
                                )
                            })),
                        )
                        .route_metadata(),
                    ),
                },
                tile_counts: None,
                tile_offsets: None,
                ticket: None,
                live_rows: Spec::live(),
            },
            alternate_bias: Spec::input(DType::F32, (Rows::Fixed(1), e), f32_bytes(&alternate)),
            bias_select: Spec::input(
                DType::U8,
                (Rows::Live, 1),
                (0..capacity).map(|r| (r % 2) as u8).collect(),
            ),
            scores: Spec::output(DType::F32, (Rows::Live, e)),
        },
        activation: Spec::input(DType::U8, (Rows::Live, k), input),
        activation_scales: Spec::input(DType::U8, (Rows::Live, k / 32), input_scales),
        weights: QuantizedWeights {
            codes: Spec::input(DType::U8, (Rows::Fixed(1), codes.len()), codes),
            local_scales: Some(Spec::input(
                DType::U8,
                (Rows::Fixed(1), scales.len()),
                scales,
            )),
            global_scales: None,
        },
        output: Spec::output(DType::Bf16, (Rows::Live, top * n)),
        output_codes: Some(Spec::output(DType::U8, (Rows::Live, top * n))),
        output_scales: Some(Spec::output(DType::U8, (Rows::Live, top * n / 32))),
        quantizer_status: Some(crate::kit::cases::block_quantize::status_role(&vec![
            0;
            tokens
        ])),
        quantizer_status_row_offset: 0,
        unit_weights,
        unused: Spec::scratch(4),
    };
    type Cache = (Vec<Vec<u8>>, Vec<usize>, Vec<(f64, f64)>);
    let cache = RefCell::new(None::<Cache>);
    let clamp = projection.query.projection.swiglu_clamp;
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let selected = selections(query, images, capacity)?;
        let ids = selected
            .iter()
            .flat_map(|r| r.expert_ids.clone())
            .collect::<Vec<_>>();
        let operands = [
            images.bytes("activation")?,
            images.bytes("activation_scales")?,
            images.bytes("weights.codes")?,
            images.bytes("weights.local_scales")?,
        ];
        if !cache.borrow().as_ref().is_some_and(|(held, prior, _)| {
            prior == &ids && held.iter().zip(operands).all(|(a, b)| a.as_slice() == b)
        }) {
            let bank = match tiles {
                Some([codes, scales]) => [
                    tile_rows(codes, operands[2])?,
                    tile_rows(scales, operands[3])?,
                ],
                None => [operands[2].to_vec(), operands[3].to_vec()],
            };
            let mut pairs = Vec::with_capacity(tokens * top * n);
            for (token, selection) in selected.iter().enumerate() {
                let x = decoded(operands[0], operands[1], token, k, false);
                for &expert in &selection.expert_ids {
                    for col in 0..n {
                        let row = expert * 2 * n + col;
                        let gate = exact_dot(&x, &decoded(&bank[0], &bank[1], row, k, true))?;
                        let up = exact_dot(&x, &decoded(&bank[0], &bank[1], row + n, k, true))?;
                        pairs.push((gate, up));
                    }
                }
            }
            *cache.borrow_mut() = Some((operands.iter().map(|b| b.to_vec()).collect(), ids, pairs));
        }
        let live_elements = live as usize * top * n;
        let pairs = cache.borrow().as_ref().unwrap().2[..live_elements].to_vec();
        let actual_weights = Rc::new(RefCell::new(None::<Vec<f32>>));
        let mut expected = Vec::new();
        for (path, result) in router_expected(query, &routing_images(images)?, live)? {
            let result = if path == "router.weights" {
                let Expected::Judge(judge) = result else {
                    return Err("router weight judge missing".into());
                };
                let held = Rc::clone(&actual_weights);
                Expected::Judge(Box::new(move |bytes| {
                    judge(bytes)?;
                    *held.borrow_mut() = Some(f32_prefix(bytes, live as usize * top)?);
                    Ok(())
                }))
            } else {
                result
            };
            let path = if path.starts_with("router.") {
                path.replacen("router.", "route.route.", 1)
            } else {
                format!("route.{path}")
            };
            expected.push((path, result));
        }
        let meta = route_metadata(query, pad, images, live)?;
        let mut sorted_values = meta.sorted_rows;
        sorted_values.resize(sorted, 0);
        let mut active = meta.active_experts;
        active.resize(e + 1, 0);
        for (name, values) in [
            ("counts", meta.counts),
            ("offsets", meta.offsets),
            ("sorted_rows", sorted_values),
            ("padded_rows", vec![meta.num_padded_rows]),
            ("active_experts", active),
        ] {
            expected.push((
                format!("route.route.routes.{name}"),
                Expected::Bits(i32_bytes(&values)),
            ));
        }
        let quantized = Rc::new(RefCell::new(None::<(Vec<u8>, Vec<u8>)>));
        let held = Rc::clone(&quantized);
        expected.push((
            "output".into(),
            Expected::Judge(Box::new(move |bytes| {
                let weights = actual_weights.borrow();
                let weights = weights
                    .as_ref()
                    .ok_or("gate/up judged before exact route weights")?;
                let values = rounded_epilogue_judge(
                    &pairs,
                    weights,
                    n,
                    clamp,
                    bytes.get(..live_elements * 2).ok_or("short BF16 output")?,
                )?;
                if live == 0 {
                    *held.borrow_mut() = Some((vec![], vec![]));
                } else {
                    let q = quantize_before_down(&values, live as usize, top * n)
                        .map_err(|e| e.to_string())?;
                    let Scales::Ue8M0(scales) = q.scales else {
                        return Err("expected ceil UE8M0 scales".into());
                    };
                    *held.borrow_mut() = Some((q.codes, scales));
                }
                Ok(())
            })),
        ));
        for (path, scales) in [("output_codes", false), ("output_scales", true)] {
            let held = Rc::clone(&quantized);
            expected.push((
                path.into(),
                Expected::Judge(Box::new(move |bytes| {
                    let q = held.borrow();
                    let q = q.as_ref().ok_or("quantization judged before BF16 output")?;
                    let expected = if scales { &q.1 } else { &q.0 };
                    if bytes.get(..expected.len()) != Some(expected.as_slice()) {
                        return Err(format!(
                            "{path} differs from exact ceil quantization of checked BF16"
                        ));
                    }
                    Ok(())
                })),
            ));
        }
        expected.push((
            "quantizer_status".into(),
            Expected::Bits(vec![0; tokens * 4]),
        ));
        Ok(expected)
    };
    let mut case = OpCase::new(prepared, capacity, roles)?.oracle(oracle);
    for (symbol, output) in guards {
        case = case.omission_guard(&symbol, output);
    }
    Ok(case)
}

pub fn gate_up<'a>(projection: Projection, capacity: u32) -> Required<'a> {
    required(projection, capacity, false)
}

/// Device negative control: use unit weights for compute while retaining
/// the independently checked router output and the weighted f64 oracle.
pub fn gate_up_unit_weight_control<'a>(projection: Projection, capacity: u32) -> Required<'a> {
    required(projection, capacity, true)
}

struct UnitWeightControl {
    inner: Routed,
    compute: String,
}
impl PreparedOp<RoutedBuffers<View>> for UnitWeightControl {
    fn info(&self) -> &OpInfo {
        &self.inner.info
    }
    fn launches(
        &self,
        buffers: &RoutedBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.inner.launches(buffers, workspace, counts)?;
        let launch = launches
            .iter_mut()
            .find(|l| l.symbol == self.compute)
            .ok_or(Refusal("unit-weight control has no compute launch"))?;
        launch.arguments.0[8] = kernel_core::contract::Argument::DeviceBuffer(
            buffers
                .unit_weights
                .ok_or(Refusal("unit-weight control input missing"))?
                .handle,
        );
        Ok(launches)
    }
}

fn required<'a>(projection: Projection, capacity: u32, unit_weights: bool) -> Required<'a> {
    let query = projection.query;
    let g = query.projection.gemm.grouped.expect("routed gate/up");
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
    let compute = projection.symbol.clone();
    Required::new(
        format!(
            "quantized chained gate/up e{} top{} tile{} k{}",
            g.experts, g.top_k, projection.row_tile, query.k.stored
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
            let mut routed = Routed::new(route, experts).map_err(Refused::Rule)?;
            if unit_weights {
                routed.info.bytes.push((
                    "unit_weights".into(),
                    u64::from(capacity) * u64::from(g.top_k) * 4,
                ));
                Ok(Box::new(UnitWeightControl {
                    inner: routed,
                    compute: compute.clone(),
                }) as Prepared<RoutedBuffers<View>>)
            } else {
                Ok(Box::new(routed) as Prepared<RoutedBuffers<View>>)
            }
        },
        move |prepared| gate_up_case(&projection, capacity, prepared),
    )
    .required()
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::codec::f64_to_bf16;

    /// A bank the op declares in row tiles is stored in them and its rows
    /// read back exactly; rows stay rows and any other layout refuses.
    #[test]
    fn gate_up_bank_is_stored_in_the_layout_the_op_reads() {
        let (experts, n, k) = (2, 96, 256);
        let rows = experts * 2 * n;
        let mut info = OpInfo::new(
            "kit.test",
            kernel_core::properties::Invariance::any_rows(1),
            Vec::new(),
        );
        assert_eq!(bank_tiles(&info, rows, k).unwrap(), None);
        info.weight_layouts
            .push(("weights".into(), WeightLayout::row_major()));
        assert_eq!(bank_tiles(&info, rows, k).unwrap(), None);
        info.weight_layouts[0].1 = WeightLayout {
            id: ROW_TILES_64X128.into(),
            version: 1,
        };
        let tiles = bank_tiles(&info, rows, k).unwrap().unwrap();
        let (codes, scales) = weight_payload(experts, n, k);
        for (bytes, tiles) in [codes, scales].into_iter().zip(tiles) {
            let mut stored = bytes.clone();
            tiles.pack_in_place(&mut stored, &mut Vec::new()).unwrap();
            assert_ne!(stored, bytes);
            assert_eq!(tile_rows(tiles, &stored).unwrap(), bytes);
        }
        info.weight_layouts[0].1.id = "bf16-fragment".into();
        assert!(bank_tiles(&info, rows, k).is_err());
    }
    #[derive(Clone, Copy, Debug)]
    enum OperandFault {
        TokenZero,
        WeightNibbles,
        ActivationNeighbors,
        ActivationNibbles,
        FirstChain,
        MissingLastBlock,
    }

    fn rejected_outputs(k: usize, fault: OperandFault) -> usize {
        let n = 96;
        let (codes, scales) = weight_payload(3, n, k);
        let (input, input_scales) = activation_payload(32, k);
        kernels_sm120::moe::chained::quantized::admit_mxfp4_payload(&codes, &scales).unwrap();
        kernels_sm120::moe::chained::quantized::admit_activations(&input, &input_scales).unwrap();
        let mut rejected = 0;
        let clamp = Some(SwigluClamp::symmetric(10.0));
        let weight = 0.25_f32;
        for token in [0, 1, 3, 7, 31] {
            let x = decoded(&input, &input_scales, token, k, false);
            let mut wrong_input = input[token * k..(token + 1) * k].to_vec();
            match fault {
                OperandFault::ActivationNeighbors => {
                    for pair in wrong_input.chunks_exact_mut(2) {
                        pair.swap(0, 1);
                    }
                }
                OperandFault::ActivationNibbles => {
                    wrong_input.iter_mut().for_each(|b| *b = b.rotate_left(4));
                }
                _ => {}
            }
            let mutated_x = if matches!(fault, OperandFault::TokenZero) {
                decoded(&input, &input_scales, 0, k, false)
            } else {
                decoded(
                    &wrong_input,
                    &input_scales[token * k / 32..(token + 1) * k / 32],
                    0,
                    k,
                    false,
                )
            };
            for expert in 0..3 {
                for col in [0, 1, 32, 33, 64, 65] {
                    let mut correct = [0.0; 2];
                    let mut mutated = [0.0; 2];
                    for half in 0..2 {
                        let row = expert * 2 * n + half * n + col;
                        let w = decoded(&codes, &scales, row, k, true);
                        correct[half] = exact_dot(&x, &w).unwrap();
                        let mut wrong_codes = codes[row * k / 2..(row + 1) * k / 2].to_vec();
                        match fault {
                            OperandFault::WeightNibbles => {
                                wrong_codes.iter_mut().for_each(|b| *b = b.rotate_left(4));
                            }
                            OperandFault::FirstChain => wrong_codes[64..].fill(0),
                            OperandFault::MissingLastBlock => wrong_codes[(k - 32) / 2..].fill(0),
                            _ => {}
                        }
                        let mutated_w = decoded(
                            &wrong_codes,
                            &scales[row * k / 32..(row + 1) * k / 32],
                            0,
                            k,
                            true,
                        );
                        // A bit-level input mutant can leave the exact-sum
                        // fixture domain. Round its f64 partials to F32 at
                        // the defined chain boundary; keep exact_dot's
                        // stronger requirement on every original operand.
                        let partials =
                            partial_chains(&mutated_x, &mutated_w, projection_k(k).unwrap())
                                .unwrap();
                        mutated[half] = f64::from(
                            fold_f32_chains(
                                &partials
                                    .values
                                    .iter()
                                    .map(|&v| v as f32)
                                    .collect::<Vec<_>>(),
                            )
                            .unwrap(),
                        );
                    }
                    let pairs = [(correct[0], correct[1])];
                    let epilogue = |p: [f64; 2]| {
                        f64_to_bf16(before_down_f64(p[0], p[1], f64::from(weight), clamp).unwrap())
                            .to_le_bytes()
                    };
                    rounded_epilogue_judge(&pairs, &[weight], 1, clamp, &epilogue(correct))
                        .unwrap();
                    rejected += usize::from(
                        rounded_epilogue_judge(&pairs, &[weight], 1, clamp, &epilogue(mutated))
                            .is_err(),
                    );
                }
            }
        }
        rejected
    }

    #[test]
    fn fixture_rejects_wrong_rows_and_packed_positions() {
        for k in [128, 256, 5120] {
            for fault in [
                OperandFault::TokenZero,
                OperandFault::WeightNibbles,
                OperandFault::ActivationNeighbors,
                OperandFault::ActivationNibbles,
            ] {
                assert!(
                    rejected_outputs(k, fault) > 0,
                    "K={k}: {fault:?} escaped the BF16 oracle"
                );
            }
        }
    }

    #[test]
    fn fixture_rejects_truncated_chains_and_last_stored_blocks() {
        // K=128 has one chain; the multi-chain lab and native widths must
        // distinguish a reduction that stops after that chain.
        for k in [256, 5120] {
            assert!(
                rejected_outputs(k, OperandFault::FirstChain) > 0,
                "K={k}: first-chain mutant escaped"
            );
        }
        for k in [128, 256, 5120] {
            assert!(
                rejected_outputs(k, OperandFault::MissingLastBlock) > 0,
                "K={k}: last-block mutant escaped"
            );
        }
    }

    #[test]
    fn epilogue_gate_rejects_early_bf16_and_weighting_after_bf16() {
        let mut early_rejected = 0;
        let mut late_rejected = 0;
        let mut unclamped_rejected = 0;
        let mut unweighted_rejected = 0;
        for base_g in [-4.0, -1.0, 0.5, 1.0, 4.0, 12.0] {
            for base_u in [-12.0, -1.0, 1.0, 4.0, 12.0] {
                for weight in [0.23_f32, 0.25, 0.29, 0.5, 0.75] {
                    let (g, u) = (base_g + 3.0 / 1024.0, base_u + 1.5 / 1024.0);
                    let clamp = Some(SwigluClamp::symmetric(10.0));
                    let ideal = before_down_f64(g, u, f64::from(weight), clamp).unwrap();
                    let correct = f64_to_bf16(ideal).to_le_bytes();
                    rounded_epilogue_judge(&[(g, u)], &[weight], 1, clamp, &correct).unwrap();
                    let early = before_down_f64(
                        f64::from(bf16_to_f32(f64_to_bf16(g))),
                        f64::from(bf16_to_f32(f64_to_bf16(u))),
                        f64::from(weight),
                        clamp,
                    )
                    .unwrap();
                    if rounded_epilogue_judge(
                        &[(g, u)],
                        &[weight],
                        1,
                        clamp,
                        &f64_to_bf16(early).to_le_bytes(),
                    )
                    .is_err()
                    {
                        early_rejected += 1;
                    }
                    let unweighted = before_down_f64(g, u, 1.0, clamp).unwrap();
                    if rounded_epilogue_judge(
                        &[(g, u)],
                        &[weight],
                        1,
                        clamp,
                        &f64_to_bf16(unweighted).to_le_bytes(),
                    )
                    .is_err()
                    {
                        unweighted_rejected += 1;
                    }
                    let unclamped = before_down_f64(g, u, f64::from(weight), None).unwrap();
                    if rounded_epilogue_judge(
                        &[(g, u)],
                        &[weight],
                        1,
                        clamp,
                        &f64_to_bf16(unclamped).to_le_bytes(),
                    )
                    .is_err()
                    {
                        unclamped_rejected += 1;
                    }
                    let late = f64::from(weight) * f64::from(bf16_to_f32(f64_to_bf16(unweighted)));
                    if rounded_epilogue_judge(
                        &[(g, u)],
                        &[weight],
                        1,
                        clamp,
                        &f64_to_bf16(late).to_le_bytes(),
                    )
                    .is_err()
                    {
                        late_rejected += 1;
                    }
                }
            }
        }
        assert!(
            unweighted_rejected > 0,
            "fixture must expose missing route weights"
        );
        assert!(unclamped_rejected > 0, "fixture must expose missing clamps");
        assert!(
            early_rejected > 0,
            "fixture must expose premature gate/up rounding"
        );
        assert!(
            late_rejected > 0,
            "fixture must expose route weighting after BF16"
        );
    }
}
