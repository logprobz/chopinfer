//! Cases of the chained engine's routed expert projections behind their
//! routes: a rank's routes (`Backend::moe_route`) and the routed gated MLP
//! projection that reads them (`Backend::gated_mlp_projection`), prepared
//! for one use and launched as one kit op ([`Routed`]). The projection
//! reads no live count, so the case plans each live count's rows, at which
//! SM120 takes the route and the projection's schedule its tables give that
//! row count.
//!
//! Tokens route to a few chosen experts of the rank, one of them with more
//! than one 64-row tile of routes at the capacity (so tile pairs pair its
//! row tiles), and to other experts otherwise. Each selected key stands
//! well apart from the others, so the IDs and route metadata compare as
//! bits; the router weights and the projection compare within the
//! contract's tolerance over each route's own sum (the projection's output
//! by a judge over the rank's routes alone, whose references are computed
//! once per case).
//!
//! The cases ([`required_split`]) run the ops at the shapes of every expert
//! split over the ranks (`chained::split`): every expert is the rank's, so
//! every route is. Every route's output row is in the write map; the judge checks
//! the routes of [`judged`] tokens against their references (every route
//! is checked at once by the conformance run's one-row bits over the live
//! counts, which run every schedule), at
//! [`SPLIT_CAPACITY`] tokens in prefill and at the decode capacity in
//! decode.

use super::super::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use super::moe::{SCALE, SPLIT_RANK, bias, f32_bytes, i32_bytes, route_tokens, stream};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows, Write, rounding_admissible};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{DType, MoeQuery, Refusal, RouterKind, RowQuery, SwigluColumns};
use kernel_core::gated_mlp::GatedMlpProjectionQuery;
use kernel_core::gated_mlp_oracle::{RouteMetadata, clamped_silu_glu, local_route_metadata};
use kernel_core::moe::{self, RouteBuffers};
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch, PreparedExperts};
use kernel_core::oracle::{self, OracleOutput};
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::backend::TILE_PAIR_MIN_ROWS;
use kernels_sm120::moe::chained::{LocalRoute, Projection, ServedShapes, Weights, split};
use std::cell::RefCell;
use std::rc::Rc;

/// The tokens of the split prefill cases: the fewest a step runs tile pairs
/// at, so the cases run the decode forms at one token, the columns at
/// partial counts and the tile pairs at the capacity.
pub const SPLIT_CAPACITY: u32 = TILE_PAIR_MIN_ROWS;

/// Whether the split cases' judge checks token `token`'s routes: the first
/// 32 tokens and every 61st.
pub fn judged(token: usize) -> bool {
    token < 32 || token.is_multiple_of(61)
}

kernel_core::scratch_record! {
    /// A rank's routes and the routed projection that reads them, as one
    /// op: the route op's roles (`route`), whose route weights, metadata
    /// and live count the projection reads, and the projection's
    /// `activation`, `weights`, `output` and an `unused` view for the
    /// arguments it never reads.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RoutedBuffers {
        pub route: record(RouteBuffers),
        pub activation: value,
        pub weights: record(QuantizedWeights),
        pub output: value,
        pub unused: value,
    }
}

/// A route op and the routed projection prepared for the same use, its
/// route tile the projection's: the route's launches, then the
/// projection's, at each step's rows.
struct Routed {
    route: Prepared<RouteBuffers<View>>,
    experts: Box<dyn PreparedExperts>,
    info: OpInfo,
}

/// The [`RoutedBuffers`] path of a projection role the op sizes.
fn projection_path(path: &str) -> Result<String, Refused> {
    Ok(match path {
        "activation" | "output" => path.into(),
        "live_tokens" => "route.live_rows".into(),
        "routing.weights" => "route.weights".into(),
        _ if path.starts_with("weights.") => path.into(),
        _ => match path.strip_prefix("routing.routes.") {
            Some(role) => format!("route.routes.{role}"),
            None => return Err(Refused::Rule(format!("the routed case binds no {path}"))),
        },
    })
}

impl Routed {
    fn new(
        route: Prepared<RouteBuffers<View>>,
        experts: Box<dyn PreparedExperts>,
    ) -> Result<Self, Refused> {
        let (of_route, of_experts) = (route.info(), experts.info());
        let mut symbols = of_route.symbols.clone();
        for symbol in &of_experts.symbols {
            if !symbols.contains(symbol) {
                symbols.push(symbol.clone());
            }
        }
        let mut info = OpInfo::new(
            format!(
                "kit.routed[{}+{}]",
                of_route.implementation, of_experts.implementation
            ),
            of_route.guarantees.meet(of_experts.guarantees),
            symbols,
        );
        let routes = of_route
            .bytes
            .iter()
            .map(|(path, bytes)| Ok::<_, Refused>((format!("route.{path}"), *bytes)));
        let projection = of_experts
            .bytes
            .iter()
            .map(|(path, bytes)| Ok((projection_path(path)?, *bytes)));
        for sized in routes.chain(projection) {
            let (path, bytes): (String, u64) = sized?;
            match info.bytes.iter_mut().find(|(known, _)| *known == path) {
                Some((_, held)) => *held = (*held).max(bytes),
                None => info.bytes.push((path, bytes)),
            }
        }
        if of_route.workspace > 0 || of_experts.workspace > 0 {
            return Err(Refused::Rule("the routed case binds no workspace".into()));
        }
        info.weight_layouts = of_experts.weight_layouts.clone();
        Ok(Self {
            route,
            experts,
            info,
        })
    }
}

impl PreparedOp<RoutedBuffers<View>> for Routed {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RoutedBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let route = &buffers.route;
        let projection = GatedMlpBuffers {
            activation: buffers.activation,
            activation_scales: None,
            activation_global: None,
            live_tokens: route.live_rows,
            output: buffers.output,
            weights: buffers.weights,
            routing: Some(ProjectionRouting {
                weights: route.weights,
                routes: route.routes,
            }),
            scratch: GatedMlpScratch {
                quantizer_status: None,
                quantizer_status_row_offset: 0,
                partial: None,
                output_codes: None,
                output_scales: None,
                unused: buffers.unused,
            },
        };
        let mut launches = self.route.launches(route, workspace, counts)?;
        launches.extend(self.experts.launches(&projection, workspace, counts)?);
        Ok(launches)
    }
}

/// The rank's experts token `token` routes to, by local index, when the
/// rank takes one in `every` tokens: expert 3 every `every` tokens, and
/// experts 40, 41 (beside 3) and 70 at sparser tokens.
fn local_experts(token: usize, every: usize) -> &'static [usize] {
    if token % (16 * every) == 1 {
        &[40]
    } else if token % (16 * every) == 2 {
        &[3, 41]
    } else if token % (32 * every) == 3 {
        &[70]
    } else if token.is_multiple_of(every) {
        &[3]
    } else {
        &[]
    }
}

/// Router logits of `rows` tokens over `experts`, top `top_k`: each token
/// selects its [`local_experts`] of rank `rank` (of `local` each) and other
/// ranks' experts for the rest, or any other expert when every expert is
/// the rank's. Every logit is a distinct multiple of 1/32: selected ones
/// from 2 up, the others from -2 down, so a selected key (sigmoid plus at
/// most a quarter of bias) stands at least 0.006 from every other selected
/// key and 0.5 from every unselected one.
fn logits(
    rows: usize,
    experts: usize,
    local: usize,
    rank: u32,
    top_k: usize,
    every: usize,
) -> Vec<f32> {
    let mut next = stream(31);
    let mine = rank as usize * local..(rank as usize + 1) * local;
    let all_mine = local == experts;
    let mut out = Vec::with_capacity(rows * experts);
    for token in 0..rows {
        let mut chosen = local_experts(token, every)
            .iter()
            .map(|expert| mine.start + expert)
            .collect::<Vec<_>>();
        while chosen.len() < top_k {
            let id = (next() % experts as u64) as usize;
            if !chosen.contains(&id) && (all_mine || !mine.contains(&id)) {
                chosen.push(id);
            }
        }
        let mut row = vec![0.0_f32; experts];
        let mut low = 0;
        for (expert, logit) in row.iter_mut().enumerate() {
            *logit = match chosen.iter().position(|&id| id == expert) {
                Some(at) => 2.0 + 8.0 * at as f32 / 32.0,
                None => {
                    low += 1;
                    -2.0 - low as f32 / 32.0
                }
            };
        }
        out.extend(row);
    }
    out
}

/// FP8 E4M3 codes of `count` bytes with the exponent's top bit clear
/// (magnitudes at most 1.875, never NaN), and F32 block scales of `blocks`.
fn fp8_weights(count: usize, blocks: usize) -> (Vec<u8>, Vec<f32>) {
    let mut next = stream(37);
    let mut codes = Vec::with_capacity(count);
    while codes.len() < count {
        let word = next().to_le_bytes();
        let take = (count - codes.len()).min(word.len());
        codes.extend(word[..take].iter().map(|&code| code & 0xbf));
    }
    let scales = (0..blocks)
        .map(|at| (1.0 + (at % 5) as f32 / 8.0) / 64.0)
        .collect();
    (codes, scales)
}

/// The rows of expert `expert` of a weight of `n` rows of `k` per expert,
/// decoded: each FP8 code times its 128 x 128 block's scale.
fn decode_expert(codes: &[u8], scales: &[f32], expert: usize, n: usize, k: usize) -> Vec<f64> {
    let table = oracle::e4m3_table();
    let blocks = (n / 128, k / 128);
    let codes = &codes[expert * n * k..(expert + 1) * n * k];
    let scales = &scales[expert * blocks.0 * blocks.1..(expert + 1) * blocks.0 * blocks.1];
    codes
        .iter()
        .enumerate()
        .map(|(at, &code)| {
            let (row, col) = (at / k, at % k);
            table[usize::from(code)] * f64::from(scales[(row / 128) * blocks.1 + col / 128])
        })
        .collect()
}

/// Each row of `weight` (rows of `x.len()`) against `x`: the products' sum
/// and the sum of their magnitudes.
fn dots(weight: &[f64], x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    weight
        .chunks_exact(x.len())
        .map(|row| {
            row.iter().zip(x).fold((0.0, 0.0), |(sum, abs), (&w, &v)| {
                (sum + w * v, abs + (w * v).abs())
            })
        })
        .unzip()
}

/// The routes of the case's tokens and each of the rank's routes' reference
/// row, computed once over every token of the images it was given.
struct Reference {
    ids: Vec<i32>,
    weights: OracleOutput,
    /// The rank's routes, ascending: the output rows the projection
    /// writes.
    routes: Vec<usize>,
    /// The judged ones of them, each with its reference row.
    rows: Vec<(usize, OracleOutput)>,
}

/// The images a case's references were computed from (logits, activation
/// and scales) and those references.
type Kept = (Vec<u8>, Rc<Reference>);

/// The write map of a route metadata role, from the live tokens' metadata.
type MetadataMap = Rc<dyn Fn(&RouteMetadata) -> LabResult<WriteMap>>;

/// A routed projection's case: which projection, its use and the rank's
/// routes it reads.
#[derive(Clone, Copy, Debug)]
struct Shape {
    experts: usize,
    local: usize,
    rank: u32,
    /// Whether a token's routes are judged ([`judged`] when split).
    sampled: bool,
    top_k: usize,
    /// The projection's output columns and K.
    columns: usize,
    k: usize,
    gate_up: bool,
    every: usize,
}

impl Shape {
    /// The references of every token of `images`.
    fn reference(&self, op: &GatedMlpProjectionQuery, images: &Images) -> LabResult<Reference> {
        let logits = images.f32s("route.logits")?;
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        };
        let (ids, weights) = route_tokens(
            &logits,
            &images.f32s("route.bias")?,
            self.experts,
            self.top_k,
            kind,
        )?;
        let codes = images.bytes("weights.codes")?;
        let scales = images.f32s("weights.local_scales")?;
        let activation = images
            .u16s("activation")?
            .into_iter()
            .map(|b| f64::from(bf16_to_f32(b)))
            .collect::<Vec<_>>();
        let mine = self.rank as usize * self.local..(self.rank as usize + 1) * self.local;
        let n = if self.gate_up {
            2 * self.columns
        } else {
            self.columns
        };
        let mut decoded = std::collections::BTreeMap::new();
        let mut routes = Vec::new();
        let mut rows = Vec::new();
        for (route, &id) in ids.iter().enumerate() {
            let Some(expert) = usize::try_from(id)
                .ok()
                .filter(|id| mine.contains(id))
                .map(|id| id - mine.start)
            else {
                continue;
            };
            routes.push(route);
            let token = route / self.top_k;
            if self.sampled && !judged(token) {
                continue;
            }
            let weight = decoded
                .entry(expert)
                .or_insert_with(|| decode_expert(codes, &scales, expert, n, self.k));
            let reference = if self.gate_up {
                let x = &activation[token * self.k..(token + 1) * self.k];
                let (values, sums) = dots(weight, x);
                let projected = OracleOutput {
                    values,
                    sum_abs_products: sums,
                    reduction_lengths: vec![self.k; n],
                };
                oracle::silu_glu(
                    &projected,
                    n,
                    SwigluColumns::ConcatenatedHalves,
                    clamped_silu_glu(op.swiglu_clamp, None),
                )
                .map_err(|e| e.0.to_string())?
            } else {
                let h = &activation[route * self.k..(route + 1) * self.k];
                let (values, sums) = dots(weight, h);
                let w = weights.values[route];
                OracleOutput {
                    values: values.iter().map(|v| v * w).collect(),
                    sum_abs_products: sums.iter().map(|s| s * w.abs()).collect(),
                    reduction_lengths: vec![self.k; n],
                }
            };
            rows.push((route, reference));
        }
        Ok(Reference {
            ids,
            weights,
            routes,
            rows,
        })
    }
}

/// The route padding of `rows` token rows: the pad of `shapes`' chained
/// route of the least capacity that holds them, as SM120 routes them.
fn pad_at(shapes: &ServedShapes, rows: u32) -> usize {
    shapes
        .routes()
        .into_iter()
        .filter(|route| route.capacity >= rows)
        .min_by_key(|route| route.capacity)
        .map_or(0, |route: LocalRoute| route.pad as usize)
}

/// The case of rank `rank`'s routes of `shapes` and `projection`'s routed
/// op over `capacity` tokens (see the module notes).
fn routed_case<'a>(
    shapes: ServedShapes,
    rank: u32,
    projection: &Projection,
    capacity: u32,
    prepared: Prepared<RoutedBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let op = projection.query;
    let gemm = op.gemm;
    let grouped = gemm
        .grouped
        .ok_or("a routed case needs a grouped projection")?;
    if projection.weights() != Weights::Fp8Block128 {
        return Err("the routed cases decode FP8 128 x 128 weights".into());
    }
    let info = prepared.info().clone();
    let shape = Shape {
        experts: shapes.experts as usize,
        local: grouped.experts as usize,
        rank,
        sampled: grouped.experts == shapes.experts,
        top_k: grouped.top_k as usize,
        columns: projection.columns() as usize,
        k: gemm.k as usize,
        gate_up: projection.gate_up(),
        every: if projection.gate_up() { 31 } else { 4 },
    };
    let (rows, top_k, local) = (capacity as usize, shape.top_k, shape.local);
    let n = if shape.gate_up {
        2 * shape.columns
    } else {
        shape.columns
    };
    let (codes, scales) = fp8_weights(local * n * shape.k, local * (n / 128) * (shape.k / 128));
    let activation = if shape.gate_up {
        bf16_values(41, rows * shape.k)
    } else {
        bf16_values(43, rows * top_k * shape.k)
    };
    let sorted = info
        .bytes_of("route.routes.sorted_rows")
        .ok_or("the op does not size its sorted rows")? as usize
        / 4;
    let kept: Rc<RefCell<Option<Kept>>> = Rc::new(RefCell::new(None));
    // The case's references, computed again only for other images.
    let reference = {
        let kept = Rc::clone(&kept);
        Rc::new(move |images: &Images| -> LabResult<Rc<Reference>> {
            let key = [
                images.bytes("route.logits")?,
                images.bytes("activation")?,
                images.bytes("weights.local_scales")?,
            ]
            .concat();
            if let Some((seen, reference)) = kept.borrow().as_ref()
                && *seen == key
            {
                return Ok(Rc::clone(reference));
            }
            let reference = Rc::new(shape.reference(&op, images)?);
            *kept.borrow_mut() = Some((key, Rc::clone(&reference)));
            Ok(reference)
        })
    };
    // The route metadata of the live tokens at their padding.
    let metadata = {
        let reference = Rc::clone(&reference);
        Rc::new(
            move |images: &Images, live: u32| -> LabResult<RouteMetadata> {
                let reference = reference(images)?;
                let global = reference.ids[..live as usize * top_k]
                    .iter()
                    .map(|&id| id as usize)
                    .collect::<Vec<_>>();
                // Nothing is routed at live 0; any pad gives its empty metadata.
                local_route_metadata(
                    &global,
                    top_k,
                    local,
                    rank as usize,
                    pad_at(&shapes, live.max(1)),
                )
                .map_err(|e| e.0.to_string())
            },
        )
    };
    let mapped = |write: MetadataMap| {
        let metadata = Rc::clone(&metadata);
        Write::Map(Box::new(move |images: &Images, live: u32| {
            if live == 0 {
                return WriteMap::of(DType::I32).none();
            }
            write(&metadata(images, live)?)
        }))
    };
    let whole = |elements: usize| {
        Spec::output_with(
            DType::I32,
            (Rows::Fixed(1), elements),
            mapped(Rc::new(move |_| WriteMap::of(DType::I32).prefix(elements))),
        )
        .route_metadata()
    };
    let width = shape.columns;
    let (output, output_dtype) = if shape.gate_up {
        (DType::Bf16, DType::Bf16)
    } else {
        (DType::F32, DType::F32)
    };
    let routes_written = {
        let reference = Rc::clone(&reference);
        move |images: &Images, live: u32| -> LabResult<WriteMap> {
            let reference = reference(images)?;
            WriteMap::of(output).rows(
                width,
                reference
                    .routes
                    .iter()
                    .copied()
                    .filter(|&route| route < live as usize * top_k),
            )
        }
    };
    let roles = RoutedBuffers {
        route: RouteBuffers {
            logits: Spec::input(
                DType::F32,
                (Rows::Live, shape.experts),
                f32_bytes(&logits(
                    rows,
                    shape.experts,
                    local,
                    rank,
                    top_k,
                    shape.every,
                )),
            ),
            bias: Some(Spec::input(
                DType::F32,
                (Rows::Fixed(1), shape.experts),
                f32_bytes(&bias(shape.experts)),
            )),
            ids: Spec::output(DType::I32, (Rows::Live, top_k)),
            weights: Spec::output(DType::F32, (Rows::Live, top_k)),
            routes: moe::RouteMetadata {
                counts: Some(whole(local)),
                offsets: whole(local + 1),
                sorted_rows: Spec::output_with(
                    DType::I32,
                    (Rows::Fixed(1), sorted),
                    mapped(Rc::new(|metadata| {
                        WriteMap::of(DType::I32).prefix(metadata.num_padded_rows as usize)
                    })),
                )
                // Padding is -1, all ones, which the default fill would
                // read as unwritten.
                .filled_with(&[0x80])
                .route_metadata(),
                // The padded total starts at zero instead of the poison's
                // -1: the conformance run's null phase drops each launch in
                // turn, and a projection launched without the route stage
                // that writes the total must find no rows, not trap on a
                // total that is no whole number of row tiles (a sticky
                // device fault for every later case). Every positive live
                // count of the cases routes to the rank, so a total left at
                // zero still fails the bits.
                padded_rows: whole(1).filled_with(&[0]),
                active_experts: Some(
                    Spec::output_with(
                        DType::I32,
                        (Rows::Fixed(1), local + 1),
                        mapped(Rc::new(|metadata| {
                            WriteMap::of(DType::I32).prefix(metadata.active_experts.len())
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
        activation: Spec::input(
            DType::Bf16,
            (
                Rows::Live,
                if shape.gate_up {
                    shape.k
                } else {
                    top_k * shape.k
                },
            ),
            bf16_bytes(&activation),
        ),
        weights: QuantizedWeights {
            codes: Spec::weight(codes),
            local_scales: Some(Spec::weight(f32_bytes(&scales))),
            global_scales: None,
        },
        output: Spec::output_with(
            output,
            (Rows::Live, top_k * width),
            Write::Map(Box::new(routes_written)),
        ),
        unused: Spec::untouched(16),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let reference = reference(images)?;
        let metadata = metadata(images, live)?;
        let mut ids = reference.ids.clone();
        ids.resize(rows * top_k, 0);
        let mut sorted_rows = metadata.sorted_rows.clone();
        sorted_rows.resize(sorted, 0);
        let mut active = metadata.active_experts.clone();
        active.resize(local + 1, 0);
        // The rank's live routes, judged row by row.
        let live_rows = reference
            .rows
            .iter()
            .filter(|(route, _)| *route < live as usize * top_k)
            .cloned()
            .collect::<Vec<_>>();
        let judge = move |bytes: &[u8]| -> LabResult<()> {
            let element = if output_dtype == DType::Bf16 { 2 } else { 4 };
            for (route, expected) in &live_rows {
                let row = &bytes[route * width * element..(route + 1) * width * element];
                let failure = |error: String| format!("route {route}: {error}");
                if output_dtype == DType::Bf16 {
                    let actual = row
                        .chunks_exact(2)
                        .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                        .collect::<Vec<_>>();
                    rounding_admissible(DType::Bf16, &actual, expected).map_err(failure)?;
                } else {
                    let actual = row
                        .chunks_exact(4)
                        .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                        .collect::<Vec<_>>();
                    oracle::check_numeric(&actual, expected, DType::F32)
                        .map_err(|e| failure(format!("{e:?}")))?;
                }
            }
            Ok(())
        };
        Ok(vec![
            ("route.ids".into(), Expected::Bits(i32_bytes(&ids))),
            (
                "route.weights".into(),
                Expected::Values(reference.weights.clone()),
            ),
            (
                "route.routes.counts".into(),
                Expected::Bits(i32_bytes(&metadata.counts)),
            ),
            (
                "route.routes.offsets".into(),
                Expected::Bits(i32_bytes(&metadata.offsets)),
            ),
            (
                "route.routes.sorted_rows".into(),
                Expected::Bits(i32_bytes(&sorted_rows)),
            ),
            (
                "route.routes.padded_rows".into(),
                Expected::Bits(i32_bytes(&[metadata.num_padded_rows])),
            ),
            (
                "route.routes.active_experts".into(),
                Expected::Bits(i32_bytes(&active)),
            ),
            ("output".into(), Expected::Judge(Box::new(judge))),
        ])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required routes of rank `rank` of `shapes` and `projection`'s routed
/// op at a use of `capacity` tokens in `phase`.
pub fn routed<'a>(
    name: String,
    shapes: ServedShapes,
    rank: u32,
    projection: Projection,
    phase: ExecutionPhase,
    capacity: u32,
) -> Required<'a> {
    let router = RowQuery::Router {
        experts: shapes.experts,
        top_k: shapes.top_k,
        kind: RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        },
    };
    let routes = MoeQuery::LocalRoutes {
        experts: shapes.experts,
        local_experts: shapes.local_experts,
        top_k: shapes.top_k,
        capacity: shapes.prefill_tokens,
    };
    let usage = OperationUse {
        phase,
        capacity,
        graph: false,
    };
    let op = projection.query;
    Required::new(
        name,
        Need::rows(capacity),
        move |backend, need| {
            let experts = backend.gated_mlp_projection(&op, usage, need)?;
            let route =
                backend.moe_route(&router, &routes, rank, experts.route_tile(), usage, need)?;
            let routed: Prepared<RoutedBuffers<View>> = Box::new(Routed::new(route, experts)?);
            Ok(routed)
        },
        move |prepared| routed_case(shapes, rank, &projection, capacity, prepared),
    )
}

/// The routed FP8 projections of `catalog` at the row tile `row_tile`,
/// gate/up then down.
fn projections(catalog: &[Projection], row_tile: u32) -> [Projection; 2] {
    [true, false].map(|gate_up| {
        catalog
            .iter()
            .find(|p| {
                p.query.gemm.grouped.is_some()
                    && p.weights() == Weights::Fp8Block128
                    && p.gate_up() == gate_up
                    && p.row_tile == row_tile
            })
            .cloned()
            .expect("the routed FP8 projections are cataloged")
    })
}

/// [`routed`] under a name that says what it runs, marked required.
fn named<'a>(
    shapes: ServedShapes,
    rank: u32,
    projection: Projection,
    phase: ExecutionPhase,
    capacity: u32,
) -> Required<'a> {
    let gemm = projection.query.gemm;
    let split = if shapes.local_experts == shapes.experts {
        "split "
    } else {
        ""
    };
    routed(
        format!(
            "routes and routed {split}{:?} {:?} {}x{} {} {capacity}",
            projection.query.projection,
            gemm.weight,
            gemm.n,
            gemm.k,
            format!("{phase:?}").to_lowercase()
        ),
        shapes,
        rank,
        projection,
        phase,
        capacity,
    )
    .required()
}

/// The split routed cases (see the module notes), each through SM120's
/// chained selection (`super::moe`'s `sm120`): the gate/up and the down at
/// a prefill of [`SPLIT_CAPACITY`] tokens and at a decode of the split
/// shapes' decode tokens.
pub fn required_split<'a>() -> Vec<Required<'a>> {
    let shapes = split::SHAPES;
    let catalog = split::catalog();
    let prefill = projections(&catalog, 64);
    let decode = projections(&catalog, 16);
    prefill
        .into_iter()
        .map(|p| (p, ExecutionPhase::Prefill, SPLIT_CAPACITY))
        .chain(
            decode
                .into_iter()
                .map(|p| (p, ExecutionPhase::Decode, shapes.decode_tokens)),
        )
        .map(|(projection, phase, capacity)| named(shapes, SPLIT_RANK, projection, phase, capacity))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each token of a split case selects its chosen experts and other
    /// experts, top 8 in all, and one chosen expert takes more than one
    /// 64-row tile of routes at the prefill case's capacity.
    #[test]
    fn tokens_route_to_the_stated_local_experts() {
        let shapes = split::SHAPES;
        let (experts, local, top_k) = (288, 288, 8);
        assert_eq!(
            (shapes.experts, shapes.local_experts, shapes.top_k),
            (288, 288, 8)
        );
        let rows = SPLIT_CAPACITY as usize;
        let logits = logits(rows, experts, local, SPLIT_RANK, top_k, 31);
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        };
        let (ids, _) = route_tokens(&logits, &bias(experts), experts, top_k, kind).unwrap();
        let mut third = 0;
        for (token, chosen) in ids.chunks_exact(top_k).enumerate() {
            let mut distinct = chosen.to_vec();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), top_k, "{token}");
            for &expert in local_experts(token, 31) {
                assert!(chosen.contains(&(expert as i32)), "{token}: {expert}");
            }
            third += usize::from(chosen.contains(&3));
        }
        assert!(third > 64, "{third}");
    }

    /// At every positive live count of the split cases the route's padded
    /// total is never the zero it starts at: every route is the rank's.
    #[test]
    fn every_live_count_routes_to_the_rank() {
        let shapes = split::SHAPES;
        let (experts, local, top_k) = (288, 288, 8);
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: SCALE,
        };
        for capacity in [SPLIT_CAPACITY, shapes.decode_tokens] {
            let logits = logits(capacity as usize, experts, local, SPLIT_RANK, top_k, 31);
            let (ids, _) = route_tokens(&logits, &bias(experts), experts, top_k, kind).unwrap();
            for live in crate::poison::live_counts(capacity)
                .into_iter()
                .filter(|&live| live > 0)
            {
                let global = ids[..live as usize * top_k]
                    .iter()
                    .map(|&id| id as usize)
                    .collect::<Vec<_>>();
                let metadata = local_route_metadata(
                    &global,
                    top_k,
                    local,
                    SPLIT_RANK as usize,
                    pad_at(&shapes, live),
                )
                .unwrap();
                assert!(metadata.num_padded_rows > 0, "{capacity} {live}");
            }
        }
    }

    /// The decoded weights stay finite and small, and a scale block covers
    /// 128 x 128 codes.
    #[test]
    fn fp8_weights_decode_per_block() {
        let (codes, scales) = fp8_weights(2 * 256 * 256, 2 * 2 * 2);
        let decoded = decode_expert(&codes, &scales, 1, 256, 256);
        assert!(
            decoded
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 1.875 / 64.0 * 1.5)
        );
        let table = oracle::e4m3_table();
        let code = codes[256 * 256 + 129 * 256 + 130];
        assert_eq!(
            decoded[129 * 256 + 130],
            table[usize::from(code)] * f64::from(scales[4 + 3])
        );
    }
}
