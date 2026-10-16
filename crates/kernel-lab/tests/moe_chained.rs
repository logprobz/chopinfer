//! The chained expert engine (`kernels_sm120::moe::chained`) against the
//! clamped SwiGLU MLP oracle (`kernel_core::gated_mlp_oracle`) at the served
//! TP4 shapes: the dense MLP and shared expert (hidden 4096, dense
//! intermediates 3072 and 512, clamp 10) and the routed experts split over
//! the ranks (`chained::split`: all 288 experts local, top 8, intermediate
//! 512):
//!
//! - dense and shared-expert MLPs in FP8 128x128 and BF16, and from
//!   `chained::formats` the NVFP4 dense MLP at the first dense intermediate
//!   (gate/up with its gate and up globals, then the lab-only NVFP4 dense
//!   down), within the decision-0081 gate;
//! - the split's routed projections in every format and served schedule
//!   against the same gate, over host-built routes, the schedules of one
//!   projection bit for bit equal;
//! - the split's whole expert layers in every format (routes, gate/up, the
//!   down over the gate/up's intermediate, both route sums) against the
//!   oracle, poisoned through eager, relaunch, graph replay and PDL, and
//!   behind producers that hold the input or the intermediate at NaN;
//! - the split's decode and prefill routes against the f64 router on tie
//!   and bias patterns (exact IDs and metadata, weights within 8 F32
//!   epsilon); its decode route scan against the one-CTA route bit for
//!   bit, its mutants and its missing-wait control;
//! - the split's block-warp (NVFP4), warp-stream (FP8 gate/up) and
//!   tile-pair (FP8 prefill) layers against the column schedule bit for bit
//!   (the column references of `moe_chained_split_references` where the
//!   split serves no column form), the oracle, their real producers and
//!   each schedule's mutants;
//! - the 128-K accumulation gate at K = 4096 against Neumaier-compensated
//!   f64 (`|y - ref| <= 2^-22 K S`) in FP8 128x128, NVFP4, MXFP8 and BF16,
//!   with scales (BF16: values) spanning many binades;
//! - poisoned eager, relaunch, graph replay and PDL-producer launches at
//!   every `poison::live_counts` count; the decoded-tile dense FP8 prefill
//!   gate/up at every count from 1 through the prefill capacity, with the
//!   comparisons on the device (`kernel_lab::device_check`).
//!
//! Expert weights are written only for the experts the routes can select;
//! every other expert keeps the poison fill (NaN codes, NaN or trapping
//! scales), so a read of an unselected expert shows in the outputs.
//!
//! Run: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --test moe_chained`.

use kernel_core::codec::{SmallFloat, bf16_to_f32, decode_ue8m0_f64, f32_to_bf16};
use kernel_core::contract::{DType, LaunchPlan, MoeQuery, RouterKind, SwigluColumns};
use kernel_core::gated_mlp_oracle::{
    RouteSum, clamped_silu_glu, compensated_dot, local_route_metadata, rank_routes, routed_sum,
};
use kernel_core::moe_oracle::LocalRoutes;
use kernel_core::oracle::{OracleOutput, check_numeric, silu_glu};
use kernel_lab::device_check::{self, DeviceCheck};
use kernel_lab::poison::{self, POISON, Phase, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::moe::chained::{
    self, LocalRoute, Projection, RouteKernel, RoutedSum, RoutedSumOutput, Schedule, TP4_288_TOP8,
    Weights, formats, split,
};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Stream};
use std::collections::BTreeMap;

use kernel_lab::families::moe_chained_decoded::shapes as decoded_shapes;
use kernel_lab::families::moe_chained_split_accumulation as accumulation;
use kernel_lab::families::moe_chained_split_block_warps as split_block_warps;
use kernel_lab::families::moe_chained_split_references as references;
use kernel_lab::families::moe_chained_split_route_scan as route_scan_shapes;
use kernel_lab::families::moe_chained_split_tile_pairs as split_tile_pairs;
use kernel_lab::families::moe_chained_split_warp_streams as split_warp_streams;
use kernel_lab::poison as replay;

/// The served TP4 shapes: the dense MLP's and the shared expert's, and the
/// hidden, top-k and expert counts the split shares.
const SHAPES: chained::ServedShapes = TP4_288_TOP8;
const EXPERTS: usize = SHAPES.experts as usize;
const TOP_K: usize = SHAPES.top_k as usize;
const HIDDEN: usize = SHAPES.hidden as usize;
/// A quarter of the experts: the range an unsplit expert-parallel rank
/// held, which the route pools spread their experts over.
const QUARTER: usize = EXPERTS / 4;
/// Live tokens of the prefill oracle checks (the CPU oracle's cost).
const PREFILL_ORACLE_TOKENS: u32 = 128;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn zeroed(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.max(4))?;
    poison::fill_all(session, &buffer, &[0])?;
    Ok(buffer)
}

fn words<T: Copy>(bytes: &[u8], from: impl Fn([u8; 4]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(4)
        .map(|w| from(w.try_into().unwrap()))
        .collect()
}

fn bytes_of<T: Copy>(values: &[T], to: impl Fn(T) -> [u8; 4]) -> Vec<u8> {
    values.iter().flat_map(|&v| to(v)).collect()
}

fn bf16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_ne_bytes([b[0], b[1]]))
        .collect()
}

fn bf16_f64(bits: u16) -> f64 {
    f64::from(bf16_to_f32(bits))
}

/// Equal bits, or both NaN: PTX leaves F32 NaN encodings unspecified, so a
/// GPU NaN's payload need not match the host's.
fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// [`same_f32`] for BF16 bits.
fn same_bf16(a: u16, b: u16) -> bool {
    a == b || (bf16_to_f32(a).is_nan() && bf16_to_f32(b).is_nan())
}

/// A BF16 value `±(1 + m / 128) 2^e` with `e` in `exponents`.
fn bf16_value(rng: &mut Rng, exponents: std::ops::RangeInclusive<i32>) -> u16 {
    let span = (exponents.end() - exponents.start() + 1) as usize;
    let e = exponents.start() + rng.below(span) as i32;
    let value = (1.0 + rng.below(128) as f64 / 128.0) * 2_f64.powi(e);
    let sign = if rng.below(2) == 0 { 1.0 } else { -1.0 };
    f32_to_bf16((sign * value) as f32)
}

/// An E4M3 code of magnitude at most 3.75 (exponent field 0..=8), never NaN.
fn e4m3_code(rng: &mut Rng) -> u8 {
    ((rng.below(2) as u8) << 7) | ((rng.below(9) as u8) << 3) | rng.below(8) as u8
}

fn launch_with(
    stream: &Stream,
    function: &Function,
    plan: LaunchPlan,
    pdl: bool,
    args: &[KernelArgument<'_>],
) -> LabResult<()> {
    let plan = LaunchPlan { pdl, ..plan };
    function
        .launch(stream, launch::config(plan)?, args)
        .map_err(err)
}

/// One weight bank of `experts` experts [rows, k] in a format, written only
/// for `active` experts, with each active expert's decoded f64 weights.
struct Bank {
    codes: DeviceBuffer,
    block_scales: Option<DeviceBuffer>,
    local_scales: Option<DeviceBuffer>,
    globals: Option<DeviceBuffer>,
    decoded: std::collections::BTreeMap<usize, Vec<f64>>,
}

impl Bank {
    #[allow(clippy::too_many_arguments)]
    fn new(
        session: &Session,
        weights: Weights,
        experts: usize,
        rows: usize,
        k: usize,
        halves: usize,
        active: &[usize],
        rng: &mut Rng,
    ) -> LabResult<Self> {
        Self::build(
            session, weights, experts, rows, k, halves, active, rng, true,
        )
    }

    /// [`Self::new`], keeping each active expert's decoded f64 weights only
    /// where `keep_decoded` (the oracle reads them; bitwise comparisons do
    /// not, and the 288 decoded experts of a split rank take about 14 GB).
    #[allow(clippy::too_many_arguments)]
    fn build(
        session: &Session,
        weights: Weights,
        experts: usize,
        rows: usize,
        k: usize,
        halves: usize,
        active: &[usize],
        rng: &mut Rng,
        keep_decoded: bool,
    ) -> LabResult<Self> {
        let code_bytes = match weights {
            Weights::Nvfp4 => rows * k / 2,
            Weights::Bf16 => rows * k * 2,
            Weights::Fp8Block128 | Weights::Mxfp8 => rows * k,
        };
        let codes = session.allocate(experts * code_bytes)?;
        poison::fill_all(session, &codes, &[POISON])?;
        let scale_bytes = match weights {
            Weights::Fp8Block128 => rows / 128 * (k / 128) * 4,
            Weights::Nvfp4 => rows * k / 16,
            Weights::Mxfp8 => rows * k / 32,
            Weights::Bf16 => 0,
        };
        let scales = (scale_bytes > 0)
            .then(|| -> LabResult<DeviceBuffer> {
                let buffer = session.allocate(experts * scale_bytes)?;
                poison::fill_all(session, &buffer, &[POISON])?;
                Ok(buffer)
            })
            .transpose()?;
        let globals = (weights == Weights::Nvfp4)
            .then(|| -> LabResult<DeviceBuffer> {
                let buffer = session.allocate(experts * halves * 4)?;
                poison::fill_all(session, &buffer, &[POISON])?;
                Ok(buffer)
            })
            .transpose()?;
        let mut decoded = std::collections::BTreeMap::new();
        for &expert in active {
            let mut values = vec![0.0; rows * k];
            let mut code = vec![0_u8; code_bytes];
            let mut scale = vec![0_u8; scale_bytes];
            match weights {
                Weights::Fp8Block128 => {
                    let block = (0..rows / 128 * (k / 128))
                        .map(|_| {
                            (0.25 * 2_f64.powi(rng.below(3) as i32 - 1) * (1.0 + rng.unit())) as f32
                        })
                        .collect::<Vec<_>>();
                    scale.copy_from_slice(&bytes_of(&block, f32::to_ne_bytes));
                    for (i, c) in code.iter_mut().enumerate() {
                        *c = e4m3_code(rng);
                        let (row, col) = (i / k, i % k);
                        values[i] = SmallFloat::E4M3.decode_f64(*c)
                            * f64::from(block[row / 128 * (k / 128) + col / 128]);
                    }
                }
                Weights::Mxfp8 => {
                    for s in scale.iter_mut() {
                        *s = 124 + rng.below(4) as u8;
                    }
                    for (i, c) in code.iter_mut().enumerate() {
                        *c = e4m3_code(rng);
                        values[i] =
                            SmallFloat::E4M3.decode_f64(*c) * decode_ue8m0_f64(scale[i / 32]);
                    }
                }
                Weights::Nvfp4 => {
                    for s in scale.iter_mut() {
                        // UE4M3 exponent field 4..=7: 2^-3 up to 1.875.
                        *s = ((4 + rng.below(4) as u8) << 3) | rng.below(8) as u8;
                    }
                    let global = (0..halves)
                        .map(|_| (0.25 * (1.0 + rng.unit())) as f32)
                        .collect::<Vec<_>>();
                    for (i, c) in code.iter_mut().enumerate() {
                        *c = rng.below(256) as u8;
                        for nibble in 0..2 {
                            let index = 2 * i + nibble;
                            let (row, col) = (index / k, index % k);
                            let half = row / (rows / halves);
                            values[index] = SmallFloat::E2M1.decode_f64((*c >> (4 * nibble)) & 15)
                                * SmallFloat::Ue4M3.decode_f64(scale[row * (k / 16) + col / 16])
                                * f64::from(global[half]);
                        }
                    }
                    poison::write(
                        session,
                        globals.as_ref().unwrap(),
                        expert * halves * 4,
                        &bytes_of(&global, f32::to_ne_bytes),
                    )?;
                }
                Weights::Bf16 => {
                    for (i, pair) in code.chunks_exact_mut(2).enumerate() {
                        let bits = bf16_value(rng, -5..=-1);
                        pair.copy_from_slice(&bits.to_ne_bytes());
                        values[i] = bf16_f64(bits);
                    }
                }
            }
            poison::write(session, &codes, expert * code_bytes, &code)?;
            if let Some(buffer) = &scales {
                poison::write(session, buffer, expert * scale_bytes, &scale)?;
            }
            if keep_decoded {
                decoded.insert(expert, values);
            }
        }
        let (block_scales, local_scales) = match weights {
            Weights::Fp8Block128 => (scales, None),
            _ => (None, scales),
        };
        Ok(Self {
            codes,
            block_scales,
            local_scales,
            globals,
            decoded,
        })
    }

    /// One FP8 expert [rows, k] (rows from the code count) from its codes and F32 block scales.
    fn fp8(session: &Session, codes: &[u8], scales: &[f32], k: usize) -> LabResult<Self> {
        let decoded = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                SmallFloat::E4M3.decode_f64(c)
                    * f64::from(scales[i / k / 128 * (k / 128) + i % k / 128])
            })
            .collect();
        Ok(Self {
            codes: upload(session, codes)?,
            block_scales: Some(upload(session, &bytes_of(scales, f32::to_ne_bytes))?),
            local_scales: None,
            globals: None,
            decoded: [(0, decoded)].into_iter().collect(),
        })
    }

    /// One NVFP4 expert from its E2M1 pairs, UE4M3 scales and one global
    /// (a down projection's).
    fn nvfp4(
        session: &Session,
        codes: &[u8],
        scales: &[u8],
        global: f32,
        k: usize,
    ) -> LabResult<Self> {
        let decoded = (0..codes.len() * 2)
            .map(|i| {
                SmallFloat::E2M1.decode_f64((codes[i / 2] >> (4 * (i % 2))) & 15)
                    * SmallFloat::Ue4M3.decode_f64(scales[i / k * (k / 16) + i % k / 16])
                    * f64::from(global)
            })
            .collect();
        Ok(Self {
            codes: upload(session, codes)?,
            block_scales: None,
            local_scales: Some(upload(session, scales)?),
            globals: Some(upload(session, &global.to_ne_bytes())?),
            decoded: [(0, decoded)].into_iter().collect(),
        })
    }

    fn arguments<'a>(&'a self) -> [KernelArgument<'a>; 4] {
        let optional = |b: &'a Option<DeviceBuffer>| {
            b.as_ref()
                .map_or(KernelArgument::NullBuffer, KernelArgument::Buffer)
        };
        [
            KernelArgument::Buffer(&self.codes),
            optional(&self.block_scales),
            optional(&self.local_scales),
            optional(&self.globals),
        ]
    }
}

/// `values[i][col] = <inputs[i], weight_i[col]>` with compensated sums and S,
/// threads over inputs.
fn project(inputs: &[(Vec<f64>, &[f64])], n: usize, k: usize) -> OracleOutput {
    let mut values = vec![0.0; inputs.len() * n];
    let mut sums = vec![0.0; inputs.len() * n];
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get());
    let per = inputs.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for ((chunk, values), sums) in inputs
            .chunks(per)
            .zip(values.chunks_mut(per * n))
            .zip(sums.chunks_mut(per * n))
        {
            scope.spawn(move || {
                for (row, (x, w)) in chunk.iter().enumerate() {
                    for col in 0..n {
                        let (value, abs) = compensated_dot(x, &w[col * k..(col + 1) * k]).unwrap();
                        values[row * n + col] = value;
                        sums[row * n + col] = abs;
                    }
                }
            });
        }
    });
    OracleOutput::new(values, sums, k).unwrap()
}

fn gate(label: &str, actual: &[f64], reference: &OracleOutput, dtype: DType) -> LabResult<()> {
    check_numeric(actual, reference, dtype).map_err(|f| {
        format!(
            "{label}: element {} is {} against {} (bound {}, error {})",
            f.index, f.actual, f.reference, f.bound, f.error
        )
    })
}

/// The weighted routes [routes, width] summed per token in route order in
/// f64 with no cast, [tokens, width], with `routed_sum`'s S and K: the
/// contract's reference for the BF16 route sum, whose one cast the gate's
/// half BF16 ulp allows. A reference cast to BF16 itself
/// (`RouteSum::F32OneCast`) rounds a second time: a sum within its F32
/// allowance of a BF16 tie can cast to one neighbour on the device and to
/// the other in the reference, a full ulp apart.
fn exact_route_sum(
    weighted: &OracleOutput,
    routes: &LocalRoutes,
    tokens: usize,
    width: usize,
) -> LabResult<OracleOutput> {
    let mut sum = routed_sum(weighted, routes, tokens, width, RouteSum::F32OneCast)
        .map_err(|e| e.0.to_string())?;
    sum.values.fill(0.0);
    for (route, &token) in routes.tokens.iter().enumerate() {
        for col in 0..width {
            sum.values[token * width + col] += weighted.values[route * width + col];
        }
    }
    Ok(sum)
}

/// Runs without a GPU: the route sum's gate allows the one cast of the
/// exact sum and no more. Two routes sum just above the BF16 tie between 1
/// and 1 + 2^-7; a device F32 sum within its allowance below the tie casts
/// to 1, half an ulp (and the allowance's share) from the exact sum but a
/// full ulp from its BF16 cast. One ulp further fails either way.
#[test]
fn route_sum_gate_allows_the_one_cast_of_the_exact_sum() {
    let (tie, nudge) = (2_f64.powi(-8), 2_f64.powi(-20));
    // S = 16 at K = 1: an F32 allowance of 2^-18.
    let weighted = OracleOutput::new(vec![1.0, tie + nudge], vec![8.0, 8.0], 1).unwrap();
    let routes = LocalRoutes {
        tokens: vec![0, 0],
        experts: vec![0, 1],
        routes: vec![0, 1],
    };
    let exact = exact_route_sum(&weighted, &routes, 1, 1).unwrap();
    assert_eq!(exact.values, [1.0 + tie + nudge]);
    assert_eq!(exact.sum_abs_products, [16.0]);
    let device = f64::from(1.0_f32 + tie as f32 - nudge as f32);
    assert!((device - exact.values[0]).abs() <= 2_f64.powi(-18));
    let cast = bf16_f64(f32_to_bf16(device as f32));
    assert_eq!(cast, 1.0);
    gate("cast down", &[cast], &exact, DType::Bf16).unwrap();
    gate("cast up", &[1.0 + 2.0 * tie], &exact, DType::Bf16).unwrap();
    let twice = routed_sum(&weighted, &routes, 1, 1, RouteSum::F32OneCast).unwrap();
    assert_eq!(twice.values, [1.0 + 2.0 * tie]);
    assert!(gate("cast down", &[cast], &twice, DType::Bf16).is_err());
    assert!(gate("one ulp further", &[1.0 - tie], &exact, DType::Bf16).is_err());
}

/// Every projection the tests launch from a catalog, with its cubin
/// prefix: the served catalog, the split's served forms, then
/// `chained::formats`' and the lab-only companions (among them the split's
/// column-schedule decode references).
fn cataloged() -> Vec<(&'static str, Projection)> {
    let tagged =
        |prefix: &'static str, list: Vec<Projection>| list.into_iter().map(move |p| (prefix, p));
    tagged(chained::PREFIX, chained::catalog())
        .chain(tagged(chained::PREFIX, chained::block_warp_catalog()))
        .chain(tagged(split::PREFIX, split::forms()))
        .chain(tagged(
            accumulation::PREFIX,
            split::accumulation_projections(),
        ))
        .chain(tagged(
            references::PREFIX,
            references::projections().to_vec(),
        ))
        .chain(tagged(
            split_tile_pairs::PREFIX,
            split_tile_pairs::lab_projections(),
        ))
        .chain(tagged(formats::PREFIX, formats::catalog()))
        .chain(tagged(formats::LAB_PREFIX, formats::lab_projections()))
        .chain(tagged(
            decoded_shapes::PREFIX,
            decoded_shapes::projections().to_vec(),
        ))
        .collect()
}

fn projection(
    weights: Weights,
    gate_up: bool,
    grouped: bool,
    row_tile: u32,
    columns: u32,
) -> Projection {
    cataloged()
        .into_iter()
        .map(|(_, p)| p)
        .find(|p| {
            p.schedule == Schedule::Columns
                && p.weights() == weights
                && p.gate_up() == gate_up
                && p.query.gemm.grouped.is_some() == grouped
                && p.row_tile == row_tile
                && (p.gate_up() && p.columns() == columns
                    || !p.gate_up() && p.query.gemm.k == columns)
        })
        .expect("cataloged chained projection")
}

/// An entry a GPU test launches: its cubin and symbol.
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    cubin: String,
    symbol: String,
}

impl Entry {
    fn projection(prefix: &str, p: &Projection) -> Self {
        let (suffix, _) = chained::schedule_cubin(p.schedule);
        Self {
            cubin: format!("{prefix}_{}{suffix}", p.weights().tag()),
            symbol: p.symbol.clone(),
        }
    }

    /// A projection of [`cataloged`] in its cubin.
    fn of(p: &Projection) -> Self {
        let (prefix, _) = cataloged()
            .into_iter()
            .find(|(_, c)| c.symbol == p.symbol)
            .expect("cataloged chained projection");
        Self::projection(prefix, p)
    }

    /// A route or route sum entry in the route cubin of `prefix`.
    fn route_in(prefix: &str, symbol: String) -> Self {
        Self {
            cubin: format!("{prefix}_route"),
            symbol,
        }
    }

    fn load(&self, session: &Session) -> LabResult<Function> {
        session
            .module(&self.cubin)?
            .function(&self.symbol)
            .map_err(err)
    }
}

/// The split's route kernels of `route`, in launch order.
fn route_entries(route: LocalRoute) -> Vec<Entry> {
    route_entries_in(split::PREFIX, route)
}

/// [`route_entries`] of the instances of `prefix`.
fn route_entries_in(prefix: &str, route: LocalRoute) -> Vec<Entry> {
    route
        .kernels()
        .iter()
        .map(|&kernel| Entry::route_in(prefix, route.symbol(prefix, kernel)))
        .collect()
}

/// The split's compiled route sum in `output`'s form, which every
/// capacity launches.
fn sum_entry(output: RoutedSumOutput) -> Entry {
    let [sum] = split::sums();
    Entry::route_in(split::PREFIX, sum.with_output(output).symbol(split::PREFIX))
}

/// A layer's gate/up and down projections at `shapes`.
fn layer_projections(
    shapes: &chained::ServedShapes,
    weights: Weights,
    prefill: bool,
) -> [Projection; 2] {
    let row_tile = shapes.routes()[usize::from(prefill)].pad;
    [true, false].map(|gate_up| {
        // The split's catalog holds every routed projection in columns,
        // also those (BF16 at decode) whose cubins hold another schedule
        // only.
        split::catalog()
            .into_iter()
            .find(|p| {
                shapes.local_experts == split::SHAPES.local_experts
                    && p.weights() == weights
                    && p.gate_up() == gate_up
                    && p.row_tile == row_tile
            })
            .unwrap_or_else(|| {
                projection(weights, gate_up, true, row_tile, shapes.expert_intermediate)
            })
    })
}

/// A dense MLP's gate/up and down projections.
fn dense_projections(weights: Weights, intermediate: u32, prefill: bool) -> [Projection; 2] {
    let row_tile = if prefill { 64 } else { 16 };
    [true, false].map(|gate_up| projection(weights, gate_up, false, row_tile, intermediate))
}

/// Whether `p` is one of the split's routed projections (its rank holds
/// all 288 experts).
fn split_projection(p: &Projection) -> bool {
    p.query
        .gemm
        .grouped
        .is_some_and(|g| g.experts == split::SHAPES.local_experts)
}

/// `projections` in `schedule`.
fn scheduled(projections: [Projection; 2], schedule: Schedule) -> [Projection; 2] {
    projections.map(|p| match schedule {
        Schedule::Columns => p,
        Schedule::BlockWarps => p.block_warps().expect("block warps take the projection"),
        Schedule::TilePairs { min_pair_tiles } => p
            .tile_pairs(min_pair_tiles)
            .expect("tile pairs take the projection"),
        Schedule::WarpStreams => match p.warp_streams() {
            Ok(streams) => streams,
            // The split's decode down (K 512, one 512-K round) does not take
            // warp streams: its layers keep the column down, as served.
            Err(_) if !p.gate_up() && split_projection(&p) => p,
            Err(e) => panic!("warp streams refuse {}: {}", p.symbol, e.0),
        },
        Schedule::DecodedTiles => panic!("decoded tiles take gate/up only"),
    })
}

const DENSE_WEIGHTS: [Weights; 3] = [Weights::Fp8Block128, Weights::Bf16, Weights::Nvfp4];

/// The dense intermediates a dense format is cataloged at: NVFP4 at the
/// first only (`chained::formats`).
fn dense_intermediates(weights: Weights) -> Vec<u32> {
    match weights {
        Weights::Nvfp4 => vec![SHAPES.dense_intermediates[0]],
        _ => SHAPES.dense_intermediates.to_vec(),
    }
}

/// Every entry the GPU tests below launch.
fn requested_entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for weights in DENSE_WEIGHTS {
        for intermediate in dense_intermediates(weights) {
            for prefill in [false, true] {
                entries.extend(
                    dense_projections(weights, intermediate, prefill)
                        .iter()
                        .map(Entry::of),
                );
            }
        }
    }
    entries.extend(split::accumulation_projections().iter().map(Entry::of));
    entries.extend(chained::block_warp_catalog().iter().map(Entry::of));
    entries.extend(split_tile_pairs::projections().iter().map(Entry::of));
    entries.extend(decoded_shapes::projections().iter().map(Entry::of));
    entries.extend(
        split::forms()
            .iter()
            .map(|p| Entry::projection(split::PREFIX, p)),
    );
    for route in split::routes() {
        entries.extend(route_entries(route));
    }
    entries.push(sum_entry(RoutedSumOutput::Bf16));
    entries.push(sum_entry(RoutedSumOutput::F32));
    entries.extend(references::projections().iter().map(Entry::of));
    entries
}

/// Runs without a GPU: every entry a GPU test launches is in its built cubin.
#[test]
fn every_launched_entry_is_built() {
    for entry in requested_entries() {
        let built =
            kernel_lab::cubin_entries(&entry.cubin).unwrap_or_else(|e| panic!("{entry:?}: {e}"));
        assert!(
            built.contains(&entry.symbol.as_str()),
            "{entry:?} is not built"
        );
    }
}

/// Logits that keep every token's top 8 inside `pool` (global experts):
/// pool experts draw from [1, 4) on a 1/64 grid with no two equal in a row,
/// every other expert sits at -8.
fn pool_logits(tokens: usize, pool: &[usize], rng: &mut Rng) -> Vec<f32> {
    let mut logits = vec![-8.0_f32; tokens * EXPERTS];
    for token in 0..tokens {
        let mut grid = (64..256).collect::<Vec<usize>>();
        for &expert in pool {
            let at = rng.below(grid.len());
            logits[token * EXPERTS + expert] = grid.swap_remove(at) as f32 / 64.0;
        }
    }
    logits
}

/// Sixteen experts spread over all 288: four in each [`QUARTER`], at
/// offsets 0, 17, 40 and the quarter's last.
fn expert_pool() -> Vec<usize> {
    (0..EXPERTS / QUARTER)
        .flat_map(|q| [0, 17, 40, QUARTER - 1].map(|l| q * QUARTER + l))
        .collect()
}

/// Route metadata and route buffers of one capacity.
struct RouteBuffers {
    logits: DeviceBuffer,
    bias: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    counts: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live: DeviceBuffer,
    active: DeviceBuffer,
}

impl RouteBuffers {
    fn new(session: &Session, route: LocalRoute, logits: &[f32], bias: &[f32]) -> LabResult<Self> {
        let capacity = route.capacity as usize;
        let local = route.local_experts as usize;
        Ok(Self {
            logits: upload(session, &bytes_of(logits, f32::to_ne_bytes))?,
            bias: upload(session, &bytes_of(bias, f32::to_ne_bytes))?,
            ids: zeroed(session, capacity * TOP_K * 4)?,
            weights: zeroed(session, capacity * TOP_K * 4)?,
            counts: zeroed(session, local * 4)?,
            offsets: zeroed(session, (local + 1) * 4)?,
            sorted: zeroed(session, route.sorted_rows_capacity() as usize * 4)?,
            padded: zeroed(session, 4)?,
            live: zeroed(session, 4)?,
            active: zeroed(session, (local + 1) * 4)?,
        })
    }

    fn set_live(&self, session: &Session, live: u32) -> LabResult<()> {
        poison::write(session, &self.live, 0, &(live as i32).to_ne_bytes())
    }

    /// Every buffer a route writes.
    fn outputs(&self) -> [&DeviceBuffer; 7] {
        [
            &self.ids,
            &self.weights,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.padded,
            &self.active,
        ]
    }

    /// The route kernels in order.
    fn launch(
        &self,
        stream: &Stream,
        route: LocalRoute,
        functions: &[Function],
        rank: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        for (kernel, function) in route.kernels().iter().zip(functions) {
            let scale = KernelArgument::U32(2.5_f32.to_bits());
            let plan = route
                .plan(*kernel, route.capacity)
                .map_err(|e| e.0.to_string())?;
            let args = match kernel {
                RouteKernel::Decode => vec![
                    b(&self.logits),
                    b(&self.bias),
                    b(&self.ids),
                    b(&self.weights),
                    b(&self.counts),
                    b(&self.offsets),
                    b(&self.sorted),
                    b(&self.padded),
                    b(&self.live),
                    b(&self.active),
                    KernelArgument::U32(rank),
                    scale,
                ],
                RouteKernel::Select => vec![
                    b(&self.logits),
                    b(&self.bias),
                    b(&self.ids),
                    b(&self.weights),
                    b(&self.live),
                    scale,
                ],
                RouteKernel::Count => vec![
                    b(&self.ids),
                    b(&self.counts),
                    b(&self.live),
                    KernelArgument::U32(rank),
                ],
                RouteKernel::Scatter => vec![
                    b(&self.ids),
                    b(&self.counts),
                    b(&self.offsets),
                    b(&self.sorted),
                    b(&self.padded),
                    b(&self.live),
                    b(&self.active),
                    KernelArgument::U32(rank),
                ],
            };
            launch_with(stream, function, plan, pdl, &args)?;
        }
        Ok(())
    }

    fn checks<'a>(
        &'a self,
        session: &'a Session,
        route: LocalRoute,
        live: u32,
        rank: usize,
    ) -> LabResult<replay::Checks<'a>> {
        replay::Checks::new(session).route(
            replay::RouteGeometry {
                routes: live as usize * TOP_K,
                experts: rank * route.local_experts as usize
                    ..(rank + 1) * route.local_experts as usize,
                block: route.pad as usize,
                sorted_offset: 0,
            },
            replay::RouteBuffers {
                ids: &self.ids,
                weights: &self.weights,
                counts: &self.counts,
                offsets: &self.offsets,
                sorted: &self.sorted,
                padded: &self.padded,
                active: Some(&self.active),
            },
            &[0x80],
        )
    }
}

/// The ranks of `route`'s shapes: one when its rank holds every expert.
fn route_ranks(route: LocalRoute) -> usize {
    (route.experts / route.local_experts) as usize
}

fn route_functions(session: &Session, route: LocalRoute) -> LabResult<Vec<Function>> {
    route_entries(route)
        .iter()
        .map(|entry| entry.load(session))
        .collect()
}

/// GPU route outputs against the f64 router and the exact metadata.
fn check_routes(
    session: &Session,
    buffers: &RouteBuffers,
    route: LocalRoute,
    logits: &[f32],
    bias: &[f32],
    live: usize,
    rank: usize,
) -> LabResult<(Vec<i32>, Vec<f32>)> {
    let ids =
        words(&poison::read(session, &buffers.ids)?, i32::from_ne_bytes)[..live * TOP_K].to_vec();
    let weights = words(
        &poison::read(session, &buffers.weights)?,
        f32::from_ne_bytes,
    )[..live * TOP_K]
        .to_vec();
    if live > 0 {
        let logits = logits[..live * EXPERTS]
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        let bias = bias.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
        let query = MoeQuery::LocalRoutes {
            experts: EXPERTS as u32,
            local_experts: route.local_experts,
            top_k: TOP_K as u32,
            capacity: route.capacity,
        };
        let kind = RouterKind::SigmoidBiasSelectNormalized {
            scaling_factor: 2.5,
        };
        let expected =
            rank_routes(&logits, Some(&bias), kind, query, rank).map_err(|e| e.0.to_string())?;
        for (token, selection) in expected.selections.iter().enumerate() {
            for (slot, (&id, &weight)) in selection
                .expert_ids
                .iter()
                .zip(&selection.weights)
                .enumerate()
            {
                let r = token * TOP_K + slot;
                if ids[r] != id as i32 {
                    return Err(format!(
                        "route {r}: expert {} where the oracle selects {id}",
                        ids[r]
                    ));
                }
                if (f64::from(weights[r]) - weight).abs() > 8.0 * f64::from(f32::EPSILON) * weight {
                    return Err(format!("route {r}: weight {} against {weight}", weights[r]));
                }
            }
        }
    }
    let local = route.local_experts as usize;
    let metadata = local_route_metadata(
        &ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
        TOP_K,
        local,
        rank,
        route.pad as usize,
    )
    .map_err(|e| e.0.to_string())?;
    let read = |b: &DeviceBuffer, n: usize| -> LabResult<Vec<i32>> {
        Ok(words(&poison::read(session, b)?, i32::from_ne_bytes)[..n].to_vec())
    };
    let padded = metadata.num_padded_rows as usize;
    let actual = (
        read(&buffers.counts, local)?,
        read(&buffers.offsets, local + 1)?,
        read(&buffers.sorted, padded)?,
        read(&buffers.padded, 1)?[0],
        read(&buffers.active, metadata.active_experts.len())?,
    );
    let expected = (
        metadata.counts,
        metadata.offsets,
        metadata.sorted_rows,
        metadata.num_padded_rows,
        metadata.active_experts,
    );
    if actual != expected {
        return Err(format!(
            "route metadata differs at {live} live tokens on rank {rank}"
        ));
    }
    Ok((ids, weights))
}

/// Router logits and bias for `tokens` tokens with ties and reorderings.
fn tie_patterns(tokens: usize, rng: &mut Rng) -> [(Vec<f32>, Vec<f32>); 4] {
    [
        // Keys on a grid coarse enough that F32 sigmoids keep the f64 order
        // (repeats tie to the lower expert).
        (
            (0..tokens * EXPERTS)
                .map(|_| rng.below(4096) as f32 / 512.0 - 4.0)
                .collect(),
            vec![0.0; EXPERTS],
        ),
        // Every key equal: the eight lowest experts win.
        (vec![0.5; tokens * EXPERTS], vec![0.0; EXPERTS]),
        // A third of the experts share the largest key: the lowest eight of
        // them win.
        (
            (0..tokens * EXPERTS)
                .map(|i| ((i % EXPERTS) % 3) as f32)
                .collect(),
            vec![0.0; EXPERTS],
        ),
        // A bias that reorders: the largest logits lose to biased ones.
        (
            (0..tokens * EXPERTS)
                .map(|i| if (i % EXPERTS) < 8 { 6.0 } else { 0.0 })
                .collect(),
            (0..EXPERTS)
                .map(|e| if e % 37 == 5 { 1.5 } else { 0.0 })
                .collect(),
        ),
    ]
}

/// The split's decode and prefill routes (all 288 experts on rank 0) on
/// the four tie patterns (`tie_patterns`: keys on a coarse grid, every key
/// equal, a third of the experts tied at the largest key, and a bias that
/// reorders the largest logits) at every `poison::live_counts` count of the
/// decode capacity: the IDs against the f64 router exactly, every weight
/// within 8 F32 epsilon of it, and the route metadata exactly
/// (`check_routes`); poisoned eager and relaunch (decode), eager and PDL
/// (prefill), the decode and prefill routes bit for bit equal, then one
/// decode graph replayed while the live count shrinks.
#[test]
fn routes_select_exactly_and_build_exact_metadata() {
    gpu_test(1 << 30, |session| {
        let mut failures = replay::Failures::default();
        let [decode, prefill] = split::routes();
        let decode_fns = route_functions(session, decode)?;
        let prefill_fns = route_functions(session, prefill)?;
        let mut rng = Rng(7);
        for (pattern, (logits, bias)) in tie_patterns(decode.capacity as usize, &mut rng)
            .into_iter()
            .enumerate()
        {
            for rank in 0..route_ranks(decode) {
                let small = RouteBuffers::new(session, decode, &logits, &bias)?;
                let mut wide_logits = logits.clone();
                wide_logits.resize(prefill.capacity as usize * EXPERTS, 0.0);
                let wide = RouteBuffers::new(session, prefill, &wide_logits, &bias)?;
                for live in poison::live_counts(decode.capacity) {
                    let case = format!("pattern {pattern} rank {rank} live {live}");
                    let result = (|| -> LabResult<()> {
                        small.set_live(session, live)?;
                        wide.set_live(session, live)?;
                        let mut checks = small.checks(session, decode, live, rank)?;
                        checks.phase(Phase::Eager, false, || {
                            small.launch(&session.stream, decode, &decode_fns, rank as u32, false)
                        })?;
                        checks.phase(Phase::Relaunch, false, || {
                            small.launch(&session.stream, decode, &decode_fns, rank as u32, false)
                        })?;
                        let (ids, weights) = check_routes(
                            session,
                            &small,
                            decode,
                            &logits,
                            &bias,
                            live as usize,
                            rank,
                        )?;
                        let mut wide_checks = wide.checks(session, prefill, live, rank)?;
                        wide_checks.phase(Phase::Eager, false, || {
                            wide.launch(&session.stream, prefill, &prefill_fns, rank as u32, false)
                        })?;
                        wide_checks.phase(Phase::Pdl, true, || {
                            wide.launch(&session.stream, prefill, &prefill_fns, rank as u32, true)
                        })?;
                        let (wide_ids, wide_weights) = check_routes(
                            session,
                            &wide,
                            prefill,
                            &wide_logits,
                            &bias,
                            live as usize,
                            rank,
                        )?;
                        let bits = |w: &[f32]| w.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                        if ids != wide_ids || bits(&weights) != bits(&wide_weights) {
                            return Err("decode and prefill routes differ".into());
                        }
                        Ok(())
                    })();
                    failures.check(&case, result);
                }
                // One graph per route, replayed while the live count shrinks.
                let result = (|| -> LabResult<()> {
                    let graph = session.capture(|stream| {
                        small.launch(stream, decode, &decode_fns, rank as u32, true)
                    })?;
                    for live in poison::live_counts(decode.capacity) {
                        small.set_live(session, live)?;
                        let mut checks = small.checks(session, decode, live, rank)?;
                        checks.phase(Phase::Eager, false, || {
                            small.launch(&session.stream, decode, &decode_fns, rank as u32, false)
                        })?;
                        checks.phase(Phase::Graph, true, || {
                            graph.launch(&session.stream).map_err(err)
                        })?;
                    }
                    Ok(())
                })();
                failures.check(&format!("pattern {pattern} rank {rank} graph"), result);
            }
        }
        failures.finish()
    })
    .expect("split routes select exactly");
}

/// Router logits and bias of the decode route comparisons of `route`: the
/// tie patterns, the comparison pools and the spread over rank 0's local
/// experts.
fn decode_route_cases(route: LocalRoute) -> Vec<(String, Vec<f32>, Vec<f32>)> {
    let tokens = route.capacity as usize;
    let mut rng = Rng(11);
    let mut cases = tie_patterns(tokens, &mut rng)
        .into_iter()
        .enumerate()
        .map(|(pattern, (logits, bias))| (format!("pattern {pattern}"), logits, bias))
        .collect::<Vec<_>>();
    for (name, pool) in comparison_pools() {
        cases.push((
            name.to_string(),
            pool_logits(tokens, &pool, &mut rng),
            vec![0.0; EXPERTS],
        ));
    }
    cases.push((
        "spread over rank 0".into(),
        spread_logits(tokens, route.local_experts as usize),
        vec![0.0; EXPERTS],
    ));
    cases
}

/// Fill bytes for the decode route comparisons: the lab's poison, which
/// reads as -1 (the padding value) in every integer buffer, and 0x80, which
/// no route value equals, so an unwritten slot shows in one of the two.
const DECODE_ROUTE_FILLS: [u8; 2] = [poison::POISON, 0x80];

/// The route buffers' bytes after filling them with `fill` and running
/// `launch`.
fn filled_route_outputs(
    session: &Session,
    buffers: &RouteBuffers,
    fill: u8,
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<Vec<Vec<u8>>> {
    let outputs = buffers.outputs();
    for buffer in outputs {
        poison::fill_all(session, buffer, &[fill])?;
    }
    launch()?;
    outputs
        .iter()
        .map(|buffer| poison::read(session, buffer))
        .collect()
}

/// The route buffers' bytes after filling them with `fill` and running
/// `function` as the decode route on ready inputs.
fn decode_route_outputs(
    session: &Session,
    buffers: &RouteBuffers,
    function: &Function,
    rank: usize,
    fill: u8,
) -> LabResult<Vec<Vec<u8>>> {
    filled_route_outputs(session, buffers, fill, || {
        buffers.launch(
            &session.stream,
            route_scan_shapes::route(),
            std::slice::from_ref(function),
            rank as u32,
            false,
        )
    })
}

/// How long [`RouteProducer`] holds the route's inputs: several times a
/// decode route's run.
const ROUTE_PRODUCER_TAIL_NS: u64 = 100_000;

/// A PDL producer of the decode route's own inputs
/// (`probe_pdl_prefetch_sm120`): after its own dependency wait it saves the
/// router logits, overwrites them with NaN and the live count with zero,
/// releases its dependents, holds for [`ROUTE_PRODUCER_TAIL_NS`], then
/// restores the logits and publishes the live count. A route that reads
/// either before its dependency wait sees the held values. It saves the
/// logits on every launch, so a captured graph replays it.
struct RouteProducer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    _staged: DeviceBuffer,
    operand: DeviceBuffer,
}

impl RouteProducer {
    fn new(session: &Session, buffers: &RouteBuffers) -> LabResult<Self> {
        let staged = session.allocate(buffers.logits.byte_len())?;
        let operand = Producer::operand(session, &buffers.logits, &staged)?;
        Ok(Self {
            function: session
                .module("probe_pdl_prefetch_sm120")?
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction: upload(session, &[0; 16])?,
            checksum: session.allocate(128 * 4)?,
            _staged: staged,
            operand,
        })
    }

    /// Holds `buffers`' logits and live count, then publishes `live`.
    fn launch(&self, stream: &Stream, buffers: &RouteBuffers, live: u32) -> LabResult<()> {
        self.function
            .launch(
                stream,
                launch::simple_config(
                    (1, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(&self.operand),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(&buffers.live),
                    KernelArgument::U32(live),
                    KernelArgument::U64(ROUTE_PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

/// The split's decode route with warp-parallel metadata
/// (`route_local_scan.cuh`, the unmodified lab instance, all 288 experts
/// rank 0's) against the one-CTA route (`route_scan_shapes::REFERENCE`),
/// bit for bit: every buffer a route writes (IDs, weights, counts, offsets,
/// sorted rows, the padded total and the active experts) filled before each
/// run with each of `DECODE_ROUTE_FILLS` and compared whole, at every live
/// count from 0 to the capacity on every rank (the split's one), under the
/// tie patterns, the comparison pools and the spread over rank 0's
/// experts.
#[test]
fn decode_route_scan_matches_the_one_cta_route_bit_for_bit() {
    gpu_test(1 << 30, |session| {
        let route = route_scan_shapes::route();
        let module = session.module(route_scan_shapes::CUBIN)?;
        let load = |symbol: &str| module.function(symbol).map_err(err);
        let reference = load(route_scan_shapes::REFERENCE)?;
        let scan = load(&route_scan_shapes::symbol(route_scan_shapes::MUTATIONS[0]))?;
        let mut failures = replay::Failures::default();
        for (name, logits, bias) in decode_route_cases(route) {
            for rank in 0..route_ranks(route) {
                let buffers = RouteBuffers::new(session, route, &logits, &bias)?;
                for live in 0..=route.capacity {
                    let case = format!("{name} rank {rank} live {live}");
                    let result = (|| -> LabResult<()> {
                        buffers.set_live(session, live)?;
                        for fill in DECODE_ROUTE_FILLS {
                            let want =
                                decode_route_outputs(session, &buffers, &reference, rank, fill)?;
                            let got = decode_route_outputs(session, &buffers, &scan, rank, fill)?;
                            same_bytes(&format!("{case} fill {fill:#04x}"), &want, &got)?;
                        }
                        Ok(())
                    })();
                    failures.check(&case, result);
                }
            }
        }
        failures.finish()
    })
    .expect("the scan decode route matches the one-CTA route bit for bit");
}

/// The decode route comparison (`decode_route_scan_matches_the_one_cta_route_bit_for_bit`
/// on rank 0 at `poison::live_counts`) finds no difference from the one-CTA
/// route on the unmodified lab instance and a difference for each mutant:
/// routes of a group ranked from the last lane, group starts summed from the
/// last group, no padding, active experts one slot late and 32-expert chunks
/// without their carry. A mutant counts as caught only on an observed output
/// difference; operational errors fail the test.
#[test]
fn decode_route_scan_checks_reject_every_mutant() {
    gpu_test(1 << 30, |session| {
        let route = route_scan_shapes::route();
        let module = session.module(route_scan_shapes::CUBIN)?;
        let reference = module.function(route_scan_shapes::REFERENCE).map_err(err)?;
        let cases = decode_route_cases(route);
        let differences = |function: &Function| -> LabResult<Vec<String>> {
            let mut found = Vec::new();
            for (name, logits, bias) in &cases {
                let buffers = RouteBuffers::new(session, route, logits, bias)?;
                for live in poison::live_counts(route.capacity) {
                    buffers.set_live(session, live)?;
                    for fill in DECODE_ROUTE_FILLS {
                        let want = decode_route_outputs(session, &buffers, &reference, 0, fill)?;
                        let got = decode_route_outputs(session, &buffers, function, 0, fill)?;
                        if want != got {
                            found.push(format!("{name} live {live} fill {fill:#04x}"));
                        }
                    }
                }
            }
            Ok(found)
        };
        let load = |mutation: &str| {
            module
                .function(&route_scan_shapes::symbol(mutation))
                .map_err(err)
        };
        let control = differences(&load(route_scan_shapes::MUTATIONS[0])?)?;
        if !control.is_empty() {
            return Err(format!("the unmodified instance differs: {control:?}"));
        }
        let mut results = Vec::new();
        for &mutation in &route_scan_shapes::MUTATIONS[1..] {
            let found = differences(&load(mutation)?)?;
            if found.is_empty() {
                return Err(format!("the {mutation} mutant passed"));
            }
            eprintln!(
                "decode route {mutation}: outputs differ in {} of {} cases",
                found.len(),
                cases.len() * poison::live_counts(route.capacity).len() * DECODE_ROUTE_FILLS.len()
            );
            results.push(serde_json::json!({"mutation": mutation, "differs": found}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("decode_route_scan_mutants"),
        );
        receipt.insert("mutants".into(), serde_json::json!(results));
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("decode route checks reject every mutant");
}

/// The split's served decode route as a PDL consumer of its own inputs: a
/// [`RouteProducer`] holds the router logits at NaN and the live count at
/// zero through its tail, and the served route launches behind it with PDL,
/// eagerly and as one captured graph replayed twice. Every route buffer,
/// filled first with each of `DECODE_ROUTE_FILLS`, must equal the one-CTA
/// route's outputs on ready inputs, at `poison::live_counts` on every rank
/// (the split's one) under every decode route case. The missing-wait control
/// (`route_scan_shapes::NO_WAIT`: the served scan with its dependency wait
/// a no-op) must equal the reference on ready inputs and differ behind the
/// producer, as an observed output difference; operational errors fail the
/// test.
#[test]
fn decode_route_scan_waits_for_its_producer() {
    gpu_test(1 << 30, |session| {
        let route = route_scan_shapes::route();
        let reference = session
            .module(route_scan_shapes::CUBIN)?
            .function(route_scan_shapes::REFERENCE)
            .map_err(err)?;
        let control = session
            .module(route_scan_shapes::NO_WAIT_CUBIN)?
            .function(route_scan_shapes::NO_WAIT)
            .map_err(err)?;
        let served = route_functions(session, route)?;
        let mut failures = replay::Failures::default();
        let mut control_ready = Vec::new();
        let mut control_caught = Vec::new();
        let mut checks = 0;
        for (name, logits, bias) in decode_route_cases(route) {
            for rank in 0..route_ranks(route) {
                let buffers = RouteBuffers::new(session, route, &logits, &bias)?;
                let producer = RouteProducer::new(session, &buffers)?;
                for live in poison::live_counts(route.capacity) {
                    let case = format!("{name} rank {rank} live {live}");
                    let sequence = |stream: &Stream, function: &Function| -> LabResult<()> {
                        producer.launch(stream, &buffers, live)?;
                        buffers.launch(
                            stream,
                            route,
                            std::slice::from_ref(function),
                            rank as u32,
                            true,
                        )
                    };
                    let result = (|| -> LabResult<()> {
                        buffers.set_live(session, live)?;
                        let graph = session.capture(|stream| sequence(stream, &served[0]))?;
                        for fill in DECODE_ROUTE_FILLS {
                            let label = format!("{case} fill {fill:#04x}");
                            let want =
                                decode_route_outputs(session, &buffers, &reference, rank, fill)?;
                            let eager = filled_route_outputs(session, &buffers, fill, || {
                                sequence(&session.stream, &served[0])
                            })?;
                            same_bytes(&format!("{label} eager"), &want, &eager)?;
                            for replay in 0..2 {
                                let replayed =
                                    filled_route_outputs(session, &buffers, fill, || {
                                        graph.launch(&session.stream).map_err(err)
                                    })?;
                                same_bytes(&format!("{label} graph {replay}"), &want, &replayed)?;
                            }
                            checks += 3;
                            let ready =
                                decode_route_outputs(session, &buffers, &control, rank, fill)?;
                            if ready != want {
                                control_ready.push(label.clone());
                            }
                            let behind = filled_route_outputs(session, &buffers, fill, || {
                                sequence(&session.stream, &control)
                            })?;
                            if behind != want {
                                control_caught.push(label);
                            }
                        }
                        Ok(())
                    })();
                    failures.check(&case, result);
                }
            }
        }
        failures.finish()?;
        if !control_ready.is_empty() {
            return Err(format!(
                "the missing-wait control differs on ready inputs: {control_ready:?}"
            ));
        }
        if control_caught.is_empty() {
            return Err("the missing-wait control passed behind the producer".into());
        }
        let cases = decode_route_cases(route).len()
            * route_ranks(route)
            * poison::live_counts(route.capacity).len()
            * DECODE_ROUTE_FILLS.len();
        eprintln!(
            "decode route behind its producer: {checks} served comparisons equal; \
             missing-wait control differs in {} of {cases} cases",
            control_caught.len()
        );
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "case".into(),
            serde_json::json!("decode_route_scan_behind_its_producer"),
        );
        receipt.insert("served_comparisons".into(), serde_json::json!(checks));
        receipt.insert(
            "control_differs".into(),
            serde_json::json!({"cases": cases, "differing": control_caught}),
        );
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("the decode route waits for its producer");
}

/// Runs without a GPU: the reference route, every scan mutant and the
/// missing-wait control are in their cubins.
#[test]
fn decode_route_scan_lab_instances_are_built() {
    let built = kernel_lab::cubin_entries(route_scan_shapes::CUBIN).unwrap();
    assert!(built.contains(&route_scan_shapes::REFERENCE));
    let control = kernel_lab::cubin_entries(route_scan_shapes::NO_WAIT_CUBIN).unwrap();
    assert_eq!(control, [route_scan_shapes::NO_WAIT]);
    for mutation in route_scan_shapes::MUTATIONS {
        let symbol = route_scan_shapes::symbol(mutation);
        assert!(built.contains(&symbol.as_str()), "{symbol}");
    }
}

/// The local routes of `rank`, whose rank holds `local` experts, among GPU
/// route IDs: (route, token, local expert), in route order.
fn local_routes(ids: &[i32], local: usize, rank: usize) -> Vec<(usize, usize, usize)> {
    ids.iter()
        .enumerate()
        .filter(|(_, id)| (**id as usize) / local == rank && **id >= 0)
        .map(|(r, &id)| (r, r / TOP_K, id as usize % local))
        .collect()
}

/// Gate/up of `inputs` rows through `bank` experts, then the clamped unit.
fn gate_up_oracle(rows: &[(Vec<f64>, usize)], bank: &Bank, inter: usize, k: usize) -> OracleOutput {
    let inputs = rows
        .iter()
        .map(|(x, expert)| (x.clone(), bank.decoded[expert].as_slice()))
        .collect::<Vec<_>>();
    let gemm = project(&inputs, 2 * inter, k);
    silu_glu(
        &gemm,
        2 * inter,
        SwigluColumns::ConcatenatedHalves,
        clamped_silu_glu(Some(SHAPES.clamp), None),
    )
    .unwrap()
}

/// One rank's MoE layer at its shapes (the served rank's, or the split's,
/// whose rank holds every expert): route kernels, gate/up, down and the
/// route sum in its BF16 form, then its F32 form over the same routes (the
/// served form; the all-reduce rounds the ranks' F32 sums once).
struct Layer {
    shapes: chained::ServedShapes,
    rank: usize,
    route: LocalRoute,
    gate_up: Projection,
    down: Projection,
    sum: RoutedSum,
    route_fns: Vec<Function>,
    fns: [Function; 4],
    routes: RouteBuffers,
    x: DeviceBuffer,
    inter: DeviceBuffer,
    routed: DeviceBuffer,
    out: DeviceBuffer,
    out_f32: DeviceBuffer,
    gate_up_bank: Bank,
    down_bank: Bank,
    x_host: Vec<u16>,
    logits: Vec<f32>,
    bias: Vec<f32>,
}

impl Layer {
    /// A layer at `shapes` whose routes select among `pool` (global
    /// experts, at least eight), running its projections in `schedule`.
    #[allow(clippy::too_many_arguments)]
    fn pooled(
        session: &Session,
        shapes: &chained::ServedShapes,
        weights: Weights,
        prefill: bool,
        rank: usize,
        seed: u64,
        pool: &[usize],
        schedule: Schedule,
    ) -> LabResult<Self> {
        Self::routed(
            session, shapes, weights, prefill, rank, seed, pool, None, schedule,
        )
    }

    /// [`Self::pooled`] with the router logits `logits` instead of drawn
    /// ones where given (their experts must lie in `pool`); then the banks
    /// keep no decoded weights, so the layer serves bitwise comparisons
    /// only.
    #[allow(clippy::too_many_arguments)]
    fn routed(
        session: &Session,
        shapes: &chained::ServedShapes,
        weights: Weights,
        prefill: bool,
        rank: usize,
        seed: u64,
        pool: &[usize],
        logits: Option<Vec<f32>>,
        schedule: Schedule,
    ) -> LabResult<Self> {
        let keep_decoded = logits.is_none();
        let (local, inter) = (
            shapes.local_experts as usize,
            shapes.expert_intermediate as usize,
        );
        let route = shapes.routes()[usize::from(prefill)];
        let [gate_up, down] = scheduled(layer_projections(shapes, weights, prefill), schedule);
        // The compiled sum, launched at the route's capacity.
        let sum = shapes.sums()[0];
        let capacity = route.capacity as usize;
        let mut rng = Rng(seed);
        let active = pool
            .iter()
            .filter(|&&e| e / local == rank)
            .map(|&e| e % local)
            .collect::<Vec<_>>();
        let logits = match logits {
            Some(logits) => logits,
            None => pool_logits(capacity, pool, &mut rng),
        };
        // No selection bias: pool keys stay on a grid whose F32 sigmoids keep
        // the f64 order (the route test covers the bias).
        let bias = vec![0.0_f32; EXPERTS];
        let x_host = (0..capacity * HIDDEN)
            .map(|_| bf16_value(&mut rng, -4..=0))
            .collect::<Vec<_>>();
        Ok(Self {
            shapes: *shapes,
            rank,
            route,
            route_fns: route_functions(session, route)?,
            fns: [
                Entry::of(&gate_up).load(session)?,
                Entry::of(&down).load(session)?,
                sum_entry(RoutedSumOutput::Bf16).load(session)?,
                sum_entry(RoutedSumOutput::F32).load(session)?,
            ],
            routes: RouteBuffers::new(session, route, &logits, &bias)?,
            x: upload(
                session,
                &x_host
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?,
            inter: zeroed(session, capacity * TOP_K * inter * 2)?,
            routed: zeroed(session, capacity * TOP_K * HIDDEN * 4)?,
            out: zeroed(session, capacity * HIDDEN * 2)?,
            out_f32: zeroed(session, capacity * HIDDEN * 4)?,
            gate_up_bank: Bank::build(
                session,
                weights,
                local,
                2 * inter,
                HIDDEN,
                2,
                &active,
                &mut rng,
                keep_decoded,
            )?,
            down_bank: Bank::build(
                session,
                weights,
                local,
                HIDDEN,
                inter,
                1,
                &active,
                &mut rng,
                keep_decoded,
            )?,
            gate_up,
            down,
            sum,
            x_host,
            logits,
            bias,
        })
    }

    /// The experts of the layer's rank.
    fn local_experts(&self) -> usize {
        self.shapes.local_experts as usize
    }

    /// The expert intermediate of the layer's rank.
    fn expert_inter(&self) -> usize {
        self.shapes.expert_intermediate as usize
    }

    fn launch(&self, stream: &Stream, pdl: bool) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        let r = &self.routes;
        self.routes
            .launch(stream, self.route, &self.route_fns, self.rank as u32, pdl)?;
        let capacity = self.route.capacity;
        self.launch_projections(
            stream,
            [&self.gate_up, &self.down],
            [&self.fns[0], &self.fns[1]],
            capacity,
            pdl,
        )?;
        for (f, output) in [(&self.fns[2], &self.out), (&self.fns[3], &self.out_f32)] {
            launch_with(
                stream,
                f,
                self.sum.plan(capacity).map_err(|e| e.0.to_string())?,
                pdl,
                &[
                    b(&self.routed),
                    b(&r.ids),
                    b(&r.live),
                    KernelArgument::U32(self.rank as u32),
                    b(output),
                ],
            )?;
        }
        Ok(())
    }

    /// Gate/up into the intermediate, then down into the weighted routes,
    /// over the routes already in place, planned at `tokens`.
    fn launch_projections(
        &self,
        stream: &Stream,
        projections: [&Projection; 2],
        fns: [&Function; 2],
        tokens: u32,
        pdl: bool,
    ) -> LabResult<()> {
        self.launch_projection(stream, true, projections[0], fns[0], tokens, pdl)?;
        self.launch_projection(stream, false, projections[1], fns[1], tokens, pdl)
    }

    /// The gate/up (into the intermediate) or the down (from it into the
    /// weighted routes) alone, as [`Self::launch_projections`] runs it.
    fn launch_projection(
        &self,
        stream: &Stream,
        gate_up: bool,
        p: &Projection,
        f: &Function,
        tokens: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        let r = &self.routes;
        let (bank, input, weights, output) = match gate_up {
            true => (&self.gate_up_bank, &self.x, None, &self.inter),
            false => (&self.down_bank, &self.inter, Some(&r.weights), &self.routed),
        };
        let [codes, block, local, globals] = bank.arguments();
        let weights = weights.map_or(KernelArgument::NullBuffer, b);
        let args = [
            b(input),
            codes,
            block,
            local,
            globals,
            b(&r.offsets),
            b(&r.sorted),
            b(&r.padded),
            b(&r.live),
            weights,
            b(output),
        ];
        launch_with(
            stream,
            f,
            p.plan(tokens).map_err(|e| e.0.to_string())?,
            pdl,
            &args,
        )
    }

    fn checks<'a>(&'a self, session: &'a Session, live: u32) -> LabResult<replay::Checks<'a>> {
        let (rank, local, inter) = (self.rank, self.local_experts(), self.expert_inter());
        let routes = live as usize * TOP_K;
        let ids = &self.routes.ids;
        let rows = move |s: &Session| -> LabResult<Vec<usize>> {
            Ok(local_routes(&replay::ids(s, ids, routes)?, local, rank)
                .into_iter()
                .map(|(r, _, _)| r)
                .collect())
        };
        let out = WriteMap::of(DType::Bf16).prefix(live as usize * HIDDEN)?;
        let out_f32 = WriteMap::of(DType::F32).prefix(live as usize * HIDDEN)?;
        Ok(self
            .routes
            .checks(session, self.route, live, rank)?
            .dynamic(
                "gate/up intermediate",
                &self.inter,
                move |s| WriteMap::of(DType::Bf16).rows(inter, rows(s)?),
                &[POISON],
            )
            .dynamic(
                "weighted routes",
                &self.routed,
                move |s| WriteMap::of(DType::I32).rows(HIDDEN, rows(s)?),
                &[POISON],
            )
            .output("route sum", &self.out, out)
            .output("F32 route sum", &self.out_f32, out_f32))
    }

    /// Every output of a live count against the oracle.
    fn check_oracle(&self, session: &Session, live: usize) -> LabResult<()> {
        let (ids, weights) = check_routes(
            session,
            &self.routes,
            self.route,
            &self.logits,
            &self.bias,
            live,
            self.rank,
        )?;
        let inter = self.expert_inter();
        let local = local_routes(&ids, self.local_experts(), self.rank);
        let x = |token: usize| -> Vec<f64> {
            self.x_host[token * HIDDEN..(token + 1) * HIDDEN]
                .iter()
                .map(|&v| bf16_f64(v))
                .collect()
        };
        let rows = local
            .iter()
            .map(|&(_, token, expert)| (x(token), expert))
            .collect::<Vec<_>>();
        let intermediate = gate_up_oracle(&rows, &self.gate_up_bank, inter, HIDDEN);
        let inter_bits = bf16s(&poison::read(session, &self.inter)?);
        let actual_inter = local
            .iter()
            .flat_map(|&(r, _, _)| {
                inter_bits[r * inter..(r + 1) * inter]
                    .iter()
                    .map(|&v| bf16_f64(v))
            })
            .collect::<Vec<_>>();
        gate(
            &format!("{} gate/up", self.gate_up.symbol),
            &actual_inter,
            &intermediate,
            DType::Bf16,
        )?;
        // Down over the kernel's own intermediate.
        let inputs = local
            .iter()
            .enumerate()
            .map(|(i, &(_, _, expert))| {
                (
                    actual_inter[i * inter..(i + 1) * inter].to_vec(),
                    self.down_bank.decoded[&expert].as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let mut weighted = project(&inputs, HIDDEN, inter);
        for (i, &(r, _, _)) in local.iter().enumerate() {
            let w = f64::from(weights[r]);
            for col in 0..HIDDEN {
                weighted.values[i * HIDDEN + col] *= w;
                weighted.sum_abs_products[i * HIDDEN + col] *= w.abs();
            }
        }
        let routed = words(&poison::read(session, &self.routed)?, f32::from_ne_bytes);
        let actual_routed = local
            .iter()
            .flat_map(|&(r, _, _)| {
                routed[r * HIDDEN..(r + 1) * HIDDEN]
                    .iter()
                    .map(|&v| f64::from(v))
            })
            .collect::<Vec<_>>();
        gate(
            &format!("{} down", self.down.symbol),
            &actual_routed,
            &weighted,
            DType::F32,
        )?;
        // The route sum: bitwise against its F32 order, and within the gate
        // of the exact f64 sum, whose one cast the gate allows.
        let out = bf16s(&poison::read(session, &self.out)?);
        for token in 0..live {
            for col in 0..HIDDEN {
                let mut acc = 0.0_f32;
                for k in 0..TOP_K {
                    let r = token * TOP_K + k;
                    if (ids[r] as usize) / self.local_experts() == self.rank {
                        acc += routed[r * HIDDEN + col];
                    }
                }
                if !same_bf16(out[token * HIDDEN + col], f32_to_bf16(acc)) {
                    return Err(format!(
                        "route sum token {token} column {col} differs from its F32 order"
                    ));
                }
            }
        }
        let routes = LocalRoutes {
            tokens: local.iter().map(|&(_, t, _)| t).collect(),
            experts: local.iter().map(|&(_, _, e)| e).collect(),
            routes: local.iter().map(|&(r, _, _)| r).collect(),
        };
        if live > 0 && !local.is_empty() {
            let reference = exact_route_sum(&weighted, &routes, live, HIDDEN)?;
            let actual = out[..live * HIDDEN]
                .iter()
                .map(|&v| bf16_f64(v))
                .collect::<Vec<_>>();
            gate("route sum", &actual, &reference, DType::Bf16)?;
        }
        // The F32 form: bitwise the oracle's F32 order over the kernel's own
        // routes, its BF16 cast bitwise the BF16 form (NaN only as NaN), and
        // within the gate of the F32 order over the f64 routes.
        let out_f32 = words(&poison::read(session, &self.out_f32)?, f32::from_ne_bytes);
        if live > 0 {
            let s = vec![0.0; actual_routed.len()];
            let kernel_routes =
                OracleOutput::new(actual_routed, s, 1).map_err(|e| e.0.to_string())?;
            let exact = routed_sum(&kernel_routes, &routes, live, HIDDEN, RouteSum::F32)
                .map_err(|e| e.0.to_string())?;
            for (i, (&y, &r)) in out_f32.iter().zip(&exact.values).enumerate() {
                if !same_f32(y, r as f32) {
                    return Err(format!(
                        "F32 route sum token {} column {} is {y} against its F32 order {r}",
                        i / HIDDEN,
                        i % HIDDEN
                    ));
                }
                if !same_bf16(f32_to_bf16(y), out[i]) {
                    return Err(format!(
                        "route sum token {} column {} is not the BF16 of the F32 form",
                        i / HIDDEN,
                        i % HIDDEN
                    ));
                }
            }
            if !local.is_empty() {
                let reference = routed_sum(&weighted, &routes, live, HIDDEN, RouteSum::F32)
                    .map_err(|e| e.0.to_string())?;
                let actual = out_f32[..live * HIDDEN]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>();
                gate("F32 route sum", &actual, &reference, DType::F32)?;
            }
        }
        Ok(())
    }
}

/// Eager against the oracle at `oracle_live`, then poisoned relaunch, graph
/// and PDL replays at every live count.
fn run_layer(session: &Session, layer: &Layer, oracle_live: u32) -> LabResult<()> {
    let mut failures = replay::Failures::default();
    let graph = session.capture(|stream| layer.launch(stream, true))?;
    for live in poison::live_counts(layer.route.capacity) {
        let result = (|| -> LabResult<()> {
            layer.routes.set_live(session, live)?;
            let mut checks = layer.checks(session, live)?;
            checks.phase(Phase::Eager, false, || layer.launch(&session.stream, false))?;
            checks.phase(Phase::Relaunch, false, || {
                layer.launch(&session.stream, false)
            })?;
            checks.phase(Phase::Graph, true, || {
                graph.launch(&session.stream).map_err(err)
            })?;
            checks.phase(Phase::Pdl, true, || layer.launch(&session.stream, true))
        })();
        failures.check(&format!("{} live {live}", layer.gate_up.symbol), result);
    }
    let result = (|| -> LabResult<()> {
        layer.routes.set_live(session, oracle_live)?;
        layer.launch(&session.stream, false)?;
        layer.check_oracle(session, oracle_live as usize)
    })();
    failures.check(
        &format!("{} oracle at {oracle_live}", layer.gate_up.symbol),
        result,
    );
    failures.finish()
}

/// The split routed experts' gate/up or down (`chained::split`) over routes
/// built on the host: rank 0 of all 288 experts, a hot pool of 12 experts
/// takes seven routes of every token and a cold set of 20 the eighth, so
/// experts hold one route up to several row tiles.
struct SplitCase {
    projection: Projection,
    route: LocalRoute,
    live: usize,
    bank: Bank,
    input: DeviceBuffer,
    output: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live_rows: DeviceBuffer,
    weights: DeviceBuffer,
    input_host: Vec<u16>,
    ids: Vec<usize>,
    route_weights: Vec<f32>,
}

impl SplitCase {
    const HOT: [usize; 12] = [0, 1, 36, 70, 71, 72, 142, 144, 200, 214, 286, 287];

    fn cold() -> Vec<usize> {
        (0..20).map(|i| 5 + 14 * i).collect()
    }

    /// `projection` (in the column schedule) at `live` tokens.
    fn new(session: &Session, projection: &Projection, live: usize, seed: u64) -> LabResult<Self> {
        let shapes = split::SHAPES;
        let (experts, inter) = (shapes.experts as usize, shapes.expert_intermediate as usize);
        let route = split::routes()[usize::from(projection.row_tile == 64)];
        let capacity = route.capacity as usize;
        let mut rng = Rng(seed);
        let cold = Self::cold();
        let mut ids = Vec::with_capacity(live * TOP_K);
        for token in 0..live {
            let mut hot = Self::HOT.to_vec();
            for _ in 0..TOP_K - 1 {
                ids.push(hot.swap_remove(rng.below(hot.len())));
            }
            ids.push(cold[token % cold.len()]);
        }
        let route_weights = (0..live * TOP_K)
            .map(|_| (0.25 + 0.5 * rng.unit()) as f32)
            .collect::<Vec<_>>();
        let metadata = local_route_metadata(&ids, TOP_K, experts, 0, route.pad as usize)
            .map_err(|e| e.0.to_string())?;
        let active = Self::HOT.iter().chain(&cold).copied().collect::<Vec<_>>();
        let (rows, k, halves) = if projection.gate_up() {
            (2 * inter, HIDDEN, 2)
        } else {
            (HIDDEN, inter, 1)
        };
        let bank = Bank::new(
            session,
            projection.weights(),
            experts,
            rows,
            k,
            halves,
            &active,
            &mut rng,
        )?;
        let input_rows = if projection.gate_up() {
            capacity
        } else {
            capacity * TOP_K
        };
        let input_host = (0..input_rows * k)
            .map(|_| bf16_value(&mut rng, -4..=0))
            .collect::<Vec<_>>();
        let (output_columns, output_bytes) = if projection.gate_up() {
            (inter, 2)
        } else {
            (HIDDEN, 4)
        };
        let sorted = session.allocate(route.sorted_rows_capacity() as usize * 4)?;
        poison::fill_all(session, &sorted, &[POISON])?;
        poison::write(
            session,
            &sorted,
            0,
            &bytes_of(&metadata.sorted_rows, i32::to_ne_bytes),
        )?;
        Ok(Self {
            projection: projection.clone(),
            route,
            live,
            bank,
            input: upload(
                session,
                &input_host
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?,
            output: zeroed(session, capacity * TOP_K * output_columns * output_bytes)?,
            offsets: upload(session, &bytes_of(&metadata.offsets, i32::to_ne_bytes))?,
            sorted,
            padded: upload(session, &metadata.num_padded_rows.to_ne_bytes())?,
            live_rows: upload(session, &(live as i32).to_ne_bytes())?,
            weights: upload(session, &bytes_of(&route_weights, f32::to_ne_bytes))?,
            input_host,
            ids,
            route_weights,
        })
    }

    /// The poisoned output after `form` (the case's projection in some
    /// schedule, from the cubin of `prefix`) ran over the routes.
    fn run(&self, session: &Session, prefix: &str, form: &Projection) -> LabResult<Vec<u8>> {
        let function = Entry::projection(prefix, form).load(session)?;
        let b = KernelArgument::Buffer;
        let [codes, block, local, globals] = self.bank.arguments();
        let weights = match form.gate_up() {
            true => KernelArgument::NullBuffer,
            false => b(&self.weights),
        };
        let args = [
            b(&self.input),
            codes,
            block,
            local,
            globals,
            b(&self.offsets),
            b(&self.sorted),
            b(&self.padded),
            b(&self.live_rows),
            weights,
            b(&self.output),
        ];
        Ok(poisoned_outputs(session, &[&self.output], || {
            launch_with(
                &session.stream,
                &function,
                form.plan(form.tokens_capacity.min(self.route.capacity))
                    .map_err(err)?,
                false,
                &args,
            )
        })?
        .remove(0))
    }

    /// Every route's output row in `output` against the f64 oracle: the
    /// clamped SwiGLU of the gate/up, or the down times the route weight.
    fn check_oracle(&self, output: &[u8]) -> LabResult<()> {
        let inter = split::SHAPES.expert_intermediate as usize;
        let routes = self.live * TOP_K;
        let input = |row: usize, k: usize| -> Vec<f64> {
            self.input_host[row * k..(row + 1) * k]
                .iter()
                .map(|&v| bf16_f64(v))
                .collect()
        };
        if self.projection.gate_up() {
            let rows = (0..routes)
                .map(|r| (input(r / TOP_K, HIDDEN), self.ids[r]))
                .collect::<Vec<_>>();
            let reference = gate_up_oracle(&rows, &self.bank, inter, HIDDEN);
            let actual = bf16s(output)[..routes * inter]
                .iter()
                .map(|&v| bf16_f64(v))
                .collect::<Vec<_>>();
            return gate(
                &format!("{} gate/up", self.projection.symbol),
                &actual,
                &reference,
                DType::Bf16,
            );
        }
        let inputs = (0..routes)
            .map(|r| (input(r, inter), self.bank.decoded[&self.ids[r]].as_slice()))
            .collect::<Vec<_>>();
        let mut reference = project(&inputs, HIDDEN, inter);
        for (r, &w) in self.route_weights.iter().enumerate() {
            let w = f64::from(w);
            for col in 0..HIDDEN {
                reference.values[r * HIDDEN + col] *= w;
                reference.sum_abs_products[r * HIDDEN + col] *= w.abs();
            }
        }
        let actual = words(output, f32::from_ne_bytes)[..routes * HIDDEN]
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        gate(
            &format!("{} down", self.projection.symbol),
            &actual,
            &reference,
            DType::F32,
        )
    }
}

/// Every served form of the split routed projections (`chained::split`)
/// in every format, against the f64 oracle at the decode capacity and at
/// [`PREFILL_ORACLE_TOKENS`] prefill tokens (a form of fewer tokens at its
/// own), and the forms of one projection that take as many tokens bit for
/// bit equal, over the whole poisoned output.
#[test]
fn split_expert_projections_match_the_oracle_in_every_format() {
    gpu_test(10 << 30, |session| {
        let mut failures = replay::Failures::default();
        let forms = split::forms()
            .into_iter()
            .map(|p| (split::PREFIX, p))
            .collect::<Vec<_>>();
        for (seed, projection) in split::catalog().into_iter().enumerate() {
            let result = (|| -> LabResult<()> {
                let most = match projection.row_tile {
                    16 => split::SHAPES.decode_tokens,
                    _ => PREFILL_ORACLE_TOKENS,
                };
                let own = forms
                    .iter()
                    .filter(|(_, f)| {
                        f.weights() == projection.weights()
                            && f.gate_up() == projection.gate_up()
                            && f.row_tile == projection.row_tile
                    })
                    .collect::<Vec<_>>();
                if own.is_empty() {
                    return Err("no served form".into());
                }
                let mut lives = own
                    .iter()
                    .map(|(_, f)| f.tokens_capacity.min(most))
                    .collect::<Vec<_>>();
                lives.sort_unstable();
                lives.dedup();
                for live in lives {
                    let case =
                        SplitCase::new(session, &projection, live as usize, 700 + seed as u64)?;
                    let mut first: Option<Vec<u8>> = None;
                    for (prefix, form) in own.iter().filter(|(_, f)| f.tokens_capacity >= live) {
                        let output = case.run(session, prefix, form)?;
                        match &first {
                            None => {
                                case.check_oracle(&output)?;
                                first = Some(output);
                            }
                            Some(first) => same_bytes(
                                &format!("{} at {live} tokens", form.symbol),
                                std::slice::from_ref(first),
                                std::slice::from_ref(&output),
                            )?,
                        }
                    }
                }
                Ok(())
            })();
            failures.check(&projection.symbol, result);
        }
        failures.finish()
    })
    .expect("split routed projections match the oracle in every format");
}

/// The split layers of `weights` this matrix runs, as (prefill, schedule):
/// at decode the column schedule where a lab or served cubin holds it and
/// the served block warps of NVFP4 and BF16 (FP8's served warp streams run
/// in `warp_stream_layers_match_the_oracle`, its tile pairs in
/// `tile_pair_layers_match_the_oracle`); at prefill the served columns.
fn split_layer_schedules(weights: Weights) -> Vec<(bool, Schedule)> {
    let decode: &[Schedule] = match weights {
        Weights::Fp8Block128 | Weights::Mxfp8 => &[Schedule::Columns],
        Weights::Nvfp4 => &[Schedule::Columns, Schedule::BlockWarps],
        Weights::Bf16 => &[Schedule::BlockWarps],
    };
    decode
        .iter()
        .map(|&schedule| (false, schedule))
        .chain([(true, Schedule::Columns)])
        .collect()
}

/// `layer` behind real producers, against its own eager launch from
/// poisoned outputs, at every non-zero `poison::live_counts` count, each
/// sequence eager and then from one captured graph replayed twice, from
/// poisoned outputs and save areas each time:
///
/// - the routes, the producer probe holding the input at NaN, then gate/up
///   and down, all PDL: a gate/up that reads its input before its
///   dependency wait sees NaN;
/// - the routes and gate/up, then the probe holding the gate/up's
///   intermediate at NaN, then the down behind it (PDL): a down that reads
///   its intermediate before its producer finishes sees NaN.
fn layer_behind_producers(session: &Session, layer: &Layer, producer: &Producer) -> LabResult<()> {
    let fresh_x = session.allocate(layer.x.byte_len())?;
    let held_x = Producer::operand(session, &layer.x, &fresh_x)?;
    let fresh_inter = session.allocate(layer.inter.byte_len())?;
    let held_inter = Producer::operand(session, &layer.inter, &fresh_inter)?;
    let capacity = layer.route.capacity;
    let [gate_up_fn, down_fn] = [&layer.fns[0], &layer.fns[1]];
    let routes = |stream: &Stream, pdl: bool| {
        layer.routes.launch(
            stream,
            layer.route,
            &layer.route_fns,
            layer.rank as u32,
            pdl,
        )
    };
    let behind_input = |stream: &Stream| -> LabResult<()> {
        routes(stream, true)?;
        producer.launch(stream, &held_x)?;
        layer.launch_projections(
            stream,
            [&layer.gate_up, &layer.down],
            [gate_up_fn, down_fn],
            capacity,
            true,
        )
    };
    let behind_intermediate = |stream: &Stream| -> LabResult<()> {
        routes(stream, false)?;
        layer.launch_projection(stream, true, &layer.gate_up, gate_up_fn, capacity, false)?;
        producer.launch(stream, &held_inter)?;
        layer.launch_projection(stream, false, &layer.down, down_fn, capacity, true)
    };
    let outputs = [&layer.inter, &layer.routed];
    let scratch = [&layer.inter, &layer.routed, &fresh_x, &fresh_inter];
    for live in poison::live_counts(capacity)
        .into_iter()
        .filter(|&live| live > 0)
    {
        layer.routes.set_live(session, live)?;
        let eager = poisoned_outputs(session, &outputs, || layer.launch(&session.stream, false))?;
        for (name, sequence) in [
            ("input", &behind_input as &dyn Fn(&Stream) -> LabResult<()>),
            ("intermediate", &behind_intermediate),
        ] {
            let ran = poisoned_outputs(session, &scratch, || sequence(&session.stream))?;
            same_bytes(
                &format!("live {live} behind the {name} producer"),
                &eager,
                &ran[..2],
            )?;
            let graph = session.capture(|stream| sequence(stream))?;
            for replay in 0..2 {
                let replayed = poisoned_outputs(session, &scratch, || {
                    graph.launch(&session.stream).map_err(err)
                })?;
                same_bytes(
                    &format!("live {live} behind the {name} producer, graph replay {replay}"),
                    &eager,
                    &replayed[..2],
                )?;
            }
        }
    }
    Ok(())
}

/// The split's whole expert layers of `weights` (`split_layer_schedules`;
/// all 288 experts on rank 0, the oracle pool): the routes, gate/up into
/// the intermediate the down reads, and both route sums, against the f64
/// oracle and the BF16 sum against the F32 one (`check_oracle`), poisoned
/// through eager, relaunch, graph replay and PDL launches at every
/// `poison::live_counts` count (`run_layer`), then behind real producers
/// (`layer_behind_producers`).
fn split_layer_matrix(session: &Session, weights: Weights, seed: u64) -> LabResult<()> {
    let producer = Producer::new(session)?;
    let mut failures = replay::Failures::default();
    for (i, (prefill, schedule)) in split_layer_schedules(weights).into_iter().enumerate() {
        let case = format!("{weights:?} prefill {prefill} {schedule:?}");
        let result = Layer::pooled(
            session,
            &split::SHAPES,
            weights,
            prefill,
            0,
            seed + i as u64,
            &expert_pool(),
            schedule,
        )
        .and_then(|layer| {
            let oracle_live = match prefill {
                true => PREFILL_ORACLE_TOKENS,
                false => layer.route.capacity,
            };
            run_layer(session, &layer, oracle_live)?;
            layer_behind_producers(session, &layer, &producer)
        });
        failures.check(&case, result);
    }
    failures.finish()
}

#[test]
fn split_fp8_layers_match_the_oracle_and_pass_behind_their_producers() {
    gpu_test(12 << 30, |session| {
        split_layer_matrix(session, Weights::Fp8Block128, 2100)
    })
    .expect("split FP8 layers");
}

#[test]
fn split_nvfp4_layers_match_the_oracle_and_pass_behind_their_producers() {
    gpu_test(12 << 30, |session| {
        split_layer_matrix(session, Weights::Nvfp4, 2200)
    })
    .expect("split NVFP4 layers");
}

#[test]
fn split_mxfp8_layers_match_the_oracle_and_pass_behind_their_producers() {
    gpu_test(12 << 30, |session| {
        split_layer_matrix(session, Weights::Mxfp8, 2300)
    })
    .expect("split MXFP8 layers");
}

#[test]
fn split_bf16_layers_match_the_oracle_and_pass_behind_their_producers() {
    gpu_test(16 << 30, |session| {
        split_layer_matrix(session, Weights::Bf16, 2400)
    })
    .expect("split BF16 layers");
}

/// Runs without a GPU: every layer of the matrix has its projections in a
/// built cubin, and the matrix runs every format at decode and prefill.
#[test]
fn split_layer_matrix_layers_are_built() {
    for weights in [
        Weights::Fp8Block128,
        Weights::Nvfp4,
        Weights::Mxfp8,
        Weights::Bf16,
    ] {
        let schedules = split_layer_schedules(weights);
        assert!(schedules.iter().any(|(prefill, _)| *prefill));
        assert!(schedules.iter().any(|(prefill, _)| !*prefill));
        for (prefill, schedule) in schedules {
            for p in scheduled(
                layer_projections(&split::SHAPES, weights, prefill),
                schedule,
            ) {
                let entry = Entry::of(&p);
                let built = kernel_lab::cubin_entries(&entry.cubin).unwrap();
                assert!(built.contains(&entry.symbol.as_str()), "{entry:?}");
            }
        }
    }
}

/// A dense or shared-expert MLP: gate/up and down over live token rows.
struct Dense {
    gate_up: Projection,
    down: Projection,
    capacity: u32,
    fns: [Function; 2],
    x: DeviceBuffer,
    inter: DeviceBuffer,
    out: DeviceBuffer,
    live: DeviceBuffer,
    gate_up_bank: Bank,
    down_bank: Bank,
    x_host: Vec<u16>,
    intermediate: usize,
}

impl Dense {
    fn new(
        session: &Session,
        weights: Weights,
        intermediate: u32,
        prefill: bool,
        seed: u64,
    ) -> LabResult<Self> {
        Self::scheduled(
            session,
            weights,
            intermediate,
            prefill,
            seed,
            Schedule::Columns,
        )
    }

    fn scheduled(
        session: &Session,
        weights: Weights,
        intermediate: u32,
        prefill: bool,
        seed: u64,
        schedule: Schedule,
    ) -> LabResult<Self> {
        let capacity = if prefill {
            SHAPES.prefill_tokens
        } else {
            SHAPES.decode_tokens
        };
        let [gate_up, down] =
            scheduled(dense_projections(weights, intermediate, prefill), schedule);
        let mut rng = Rng(seed);
        let i = intermediate as usize;
        let x_host = (0..capacity as usize * HIDDEN)
            .map(|_| bf16_value(&mut rng, -4..=0))
            .collect::<Vec<_>>();
        Ok(Self {
            fns: [
                Entry::of(&gate_up).load(session)?,
                Entry::of(&down).load(session)?,
            ],
            x: upload(
                session,
                &x_host
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?,
            inter: zeroed(session, capacity as usize * i * 2)?,
            out: zeroed(session, capacity as usize * HIDDEN * 2)?,
            live: zeroed(session, 4)?,
            gate_up_bank: Bank::new(session, weights, 1, 2 * i, HIDDEN, 2, &[0], &mut rng)?,
            down_bank: Bank::new(session, weights, 1, HIDDEN, i, 1, &[0], &mut rng)?,
            gate_up,
            down,
            capacity,
            x_host,
            intermediate: i,
        })
    }

    fn launch(&self, stream: &Stream, pdl: bool) -> LabResult<()> {
        self.launch_projections(
            stream,
            [&self.gate_up, &self.down],
            [&self.fns[0], &self.fns[1]],
            self.capacity,
            pdl,
        )
    }

    fn launch_projections(
        &self,
        stream: &Stream,
        projections: [&Projection; 2],
        fns: [&Function; 2],
        tokens: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        let null = || KernelArgument::NullBuffer;
        for (p, f, bank, input, output) in [
            (
                projections[0],
                fns[0],
                &self.gate_up_bank,
                &self.x,
                &self.inter,
            ),
            (
                projections[1],
                fns[1],
                &self.down_bank,
                &self.inter,
                &self.out,
            ),
        ] {
            let [codes, block, local, globals] = bank.arguments();
            let args = [
                b(input),
                codes,
                block,
                local,
                globals,
                null(),
                null(),
                null(),
                b(&self.live),
                null(),
                b(output),
            ];
            launch_with(
                stream,
                f,
                p.plan(tokens).map_err(|e| e.0.to_string())?,
                pdl,
                &args,
            )?;
        }
        Ok(())
    }

    fn check_oracle(&self, session: &Session, live: usize) -> LabResult<()> {
        let i = self.intermediate;
        let rows = (0..live)
            .map(|t| {
                (
                    self.x_host[t * HIDDEN..(t + 1) * HIDDEN]
                        .iter()
                        .map(|&v| bf16_f64(v))
                        .collect(),
                    0,
                )
            })
            .collect::<Vec<_>>();
        let intermediate = gate_up_oracle(&rows, &self.gate_up_bank, i, HIDDEN);
        let inter = bf16s(&poison::read(session, &self.inter)?)[..live * i]
            .iter()
            .map(|&v| bf16_f64(v))
            .collect::<Vec<_>>();
        gate(&self.gate_up.symbol, &inter, &intermediate, DType::Bf16)?;
        let inputs = (0..live)
            .map(|t| {
                (
                    inter[t * i..(t + 1) * i].to_vec(),
                    self.down_bank.decoded[&0].as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let reference = project(&inputs, HIDDEN, i);
        let out = bf16s(&poison::read(session, &self.out)?)[..live * HIDDEN]
            .iter()
            .map(|&v| bf16_f64(v))
            .collect::<Vec<_>>();
        gate(&self.down.symbol, &out, &reference, DType::Bf16)
    }
}

#[test]
fn dense_and_shared_mlps_match_the_oracle() {
    gpu_test(2 << 30, |session| {
        let mut failures = replay::Failures::default();
        for (seed, weights) in DENSE_WEIGHTS.into_iter().enumerate() {
            for intermediate in dense_intermediates(weights) {
                for prefill in [false, true] {
                    let case = format!("{weights:?} intermediate {intermediate} prefill {prefill}");
                    let result = (|| -> LabResult<()> {
                        let dense =
                            Dense::new(session, weights, intermediate, prefill, 300 + seed as u64)?;
                        let graph = session.capture(|stream| dense.launch(stream, true))?;
                        for live in poison::live_counts(dense.capacity) {
                            poison::write(session, &dense.live, 0, &(live as i32).to_ne_bytes())?;
                            let mut checks = replay::Checks::new(session)
                                .output(
                                    "intermediate",
                                    &dense.inter,
                                    WriteMap::of(DType::Bf16)
                                        .prefix(live as usize * dense.intermediate)?,
                                )
                                .output(
                                    "output",
                                    &dense.out,
                                    WriteMap::of(DType::Bf16).prefix(live as usize * HIDDEN)?,
                                );
                            checks.phase(Phase::Eager, false, || {
                                dense.launch(&session.stream, false)
                            })?;
                            checks.phase(Phase::Relaunch, false, || {
                                dense.launch(&session.stream, false)
                            })?;
                            checks.phase(Phase::Graph, true, || {
                                graph.launch(&session.stream).map_err(err)
                            })?;
                            checks
                                .phase(Phase::Pdl, true, || dense.launch(&session.stream, true))?;
                        }
                        let live = dense.capacity.min(PREFILL_ORACLE_TOKENS);
                        poison::write(session, &dense.live, 0, &(live as i32).to_ne_bytes())?;
                        dense.launch(&session.stream, false)?;
                        dense.check_oracle(session, live as usize)
                    })();
                    failures.check(&case, result);
                }
            }
        }
        failures.finish()
    })
    .expect("chained dense MLPs match the oracle");
}

/// Rewrites the active experts of an `[n, k]` bank in `weights` with wide
/// scales and their decoded weights: FP8 one set of block scales 2^-8 to
/// 2^8 for every expert, NVFP4 UE4M3 scales per 16 K from 2^-6 to 1.875 *
/// 2^7 and a global of 1, MXFP8 UE8M0 scales per 32 K from 2^-8 to 2^8,
/// BF16 values in one binade per row and 128-K block, 2^-8 to 2^8.
fn widen(
    session: &Session,
    bank: &mut Bank,
    weights: Weights,
    (n, k): (usize, usize),
    active: &[usize],
    rng: &mut Rng,
) -> LabResult<()> {
    let all_codes = poison::read(session, &bank.codes)?;
    match weights {
        Weights::Fp8Block128 => {
            let block = (0..n / 128 * (k / 128))
                .map(|_| 2_f32.powi(rng.below(17) as i32 - 8))
                .collect::<Vec<_>>();
            for &expert in active {
                let codes = &all_codes[expert * n * k..(expert + 1) * n * k];
                let decoded = codes
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        SmallFloat::E4M3.decode_f64(c)
                            * f64::from(block[i / k / 128 * (k / 128) + i % k / 128])
                    })
                    .collect();
                bank.decoded.insert(expert, decoded);
                poison::write(
                    session,
                    bank.block_scales.as_ref().unwrap(),
                    expert * block.len() * 4,
                    &bytes_of(&block, f32::to_ne_bytes),
                )?;
            }
        }
        Weights::Nvfp4 => {
            for &expert in active {
                let codes = &all_codes[expert * n * k / 2..(expert + 1) * n * k / 2];
                // UE4M3 exponent fields 1..=14 with a 1.x mantissa: 2^-6
                // up to 1.875 * 2^7, times E2M1 magnitudes 0.5 to 6.
                let scales = (0..n * k / 16)
                    .map(|_| ((1 + rng.below(14) as u8) << 3) | rng.below(8) as u8)
                    .collect::<Vec<_>>();
                let decoded = (0..n * k)
                    .map(|i| {
                        SmallFloat::E2M1.decode_f64((codes[i / 2] >> (4 * (i % 2))) & 15)
                            * SmallFloat::Ue4M3.decode_f64(scales[i / k * (k / 16) + i % k / 16])
                    })
                    .collect();
                bank.decoded.insert(expert, decoded);
                poison::write(
                    session,
                    bank.local_scales.as_ref().unwrap(),
                    expert * scales.len(),
                    &scales,
                )?;
                poison::write(
                    session,
                    bank.globals.as_ref().unwrap(),
                    expert * 4,
                    &1.0_f32.to_ne_bytes(),
                )?;
            }
        }
        Weights::Mxfp8 => {
            for &expert in active {
                let codes = &all_codes[expert * n * k..(expert + 1) * n * k];
                let scales = (0..n * k / 32)
                    .map(|_| 119 + rng.below(17) as u8)
                    .collect::<Vec<_>>();
                let decoded = codes
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        SmallFloat::E4M3.decode_f64(c) * decode_ue8m0_f64(scales[i / 32])
                    })
                    .collect();
                bank.decoded.insert(expert, decoded);
                poison::write(
                    session,
                    bank.local_scales.as_ref().unwrap(),
                    expert * scales.len(),
                    &scales,
                )?;
            }
        }
        Weights::Bf16 => {
            for &expert in active {
                let exponents = (0..n * k / 128)
                    .map(|_| rng.below(17) as i32 - 8)
                    .collect::<Vec<_>>();
                let values = (0..n * k)
                    .map(|i| {
                        let e = exponents[i / 128];
                        bf16_value(rng, e..=e)
                    })
                    .collect::<Vec<_>>();
                bank.decoded
                    .insert(expert, values.iter().map(|&v| bf16_f64(v)).collect());
                poison::write(
                    session,
                    &bank.codes,
                    expert * n * k * 2,
                    &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
        }
    }
    Ok(())
}

/// Raw F32 sums of the grouped down `p` (route weight 1) over `bank`, whose
/// `active` experts the routes reach, against Neumaier-compensated f64
/// within 2^-22 K S. Half of each token's routes go to no expert of the
/// rank (ids past `p`'s local experts), so the projection skips them.
fn check_long_sums(
    session: &Session,
    prefix: &str,
    p: &Projection,
    bank: &Bank,
    active: [usize; 4],
    rng: &mut Rng,
) -> LabResult<()> {
    let k = p.query.gemm.k as usize;
    let n = p.columns() as usize;
    let tokens = p.tokens_capacity as usize;
    let experts = grouped_experts(p);
    let f = Entry::projection(prefix, p).load(session)?;
    // Each token routes k = 0..3 to the active experts, k = 4..7 elsewhere.
    let ids = (0..tokens * TOP_K)
        .map(|r| {
            if r % TOP_K < 4 {
                active[(r + r / TOP_K) % 4]
            } else {
                experts + r % TOP_K
            }
        })
        .collect::<Vec<_>>();
    let metadata = local_route_metadata(&ids, TOP_K, experts, 0, p.row_tile as usize)
        .map_err(|e| e.0.to_string())?;
    // Rows of alternating sign and binade: large partial sums cancel.
    let input = (0..tokens * TOP_K * k)
        .map(|i| bf16_value(rng, if (i / 128) % 2 == 0 { 4..=8 } else { -12..=-4 }))
        .collect::<Vec<_>>();
    let route_weights = vec![1.0_f32; tokens * TOP_K];
    let i32s = |v: &[i32]| bytes_of(v, i32::to_ne_bytes);
    let input_buffer = upload(
        session,
        &input
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let offsets = upload(session, &i32s(&metadata.offsets))?;
    let sorted = upload(session, &i32s(&metadata.sorted_rows))?;
    let padded = upload(session, &i32s(&[metadata.num_padded_rows]))?;
    let weights = upload(session, &bytes_of(&route_weights, f32::to_ne_bytes))?;
    let output = session.allocate(tokens * TOP_K * n * 4)?;
    poison::fill_all(session, &output, &[POISON])?;
    let b = KernelArgument::Buffer;
    let [codes, block_arg, local, globals] = bank.arguments();
    let args = [
        b(&input_buffer),
        codes,
        block_arg,
        local,
        globals,
        b(&offsets),
        b(&sorted),
        b(&padded),
        KernelArgument::NullBuffer,
        b(&weights),
        b(&output),
    ];
    launch_with(
        &session.stream,
        &f,
        p.plan(tokens as u32).map_err(|e| e.0.to_string())?,
        false,
        &args,
    )?;
    let routed = words(&poison::read(session, &output)?, f32::from_ne_bytes);
    let local = local_routes(
        &ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
        experts,
        0,
    );
    let inputs = local
        .iter()
        .map(|&(r, _, e)| {
            (
                input[r * k..(r + 1) * k]
                    .iter()
                    .map(|&v| bf16_f64(v))
                    .collect(),
                bank.decoded[&e].as_slice(),
            )
        })
        .collect::<Vec<_>>();
    let reference = project(&inputs, n, k);
    let actual = local
        .iter()
        .flat_map(|&(r, _, _)| routed[r * n..(r + 1) * n].iter().map(|&v| f64::from(v)))
        .collect::<Vec<_>>();
    gate(&p.symbol, &actual, &reference, DType::F32)
}

/// The split's K = 4096 grouped down (`split::accumulation_projections`)
/// in `weights` and `schedule`.
fn accumulation_projection(weights: Weights, schedule: Schedule) -> Projection {
    split::accumulation_projections()
        .into_iter()
        .find(|p| p.weights() == weights && p.schedule == schedule)
        .expect("the split's K = 4096 downs hold every format")
}

/// The local experts of the grouped projection `p`.
fn grouped_experts(p: &Projection) -> usize {
    p.query.gemm.grouped.expect("a grouped projection").experts as usize
}

/// Raw F32 sums at K = 4096 (route weight 1) against Neumaier-compensated
/// f64 within 2^-22 K S, over operands spanning many binades, through the
/// split's FP8 grouped down (all 288 experts the rank's).
#[test]
fn sums_at_k_4096_hold_the_128_k_accumulation_gate() {
    // The FP8 bank of 288 experts at 4096 x 4096 takes 4.5 GiB.
    gpu_test(6 << 30, |session| {
        let p = accumulation_projection(Weights::Fp8Block128, Schedule::Columns);
        let (k, n) = (p.query.gemm.k as usize, p.columns() as usize);
        let local = grouped_experts(&p);
        let mut rng = Rng(4096);
        let active = [0, 17, 40, local - 1];
        // Wide block scales: 2^-8 to 2^8.
        let mut bank = Bank::new(
            session,
            Weights::Fp8Block128,
            local,
            n,
            k,
            1,
            &active,
            &mut rng,
        )?;
        widen(
            session,
            &mut bank,
            Weights::Fp8Block128,
            (n, k),
            &active,
            &mut rng,
        )?;
        check_long_sums(session, accumulation::PREFIX, &p, &bank, active, &mut rng)
    })
    .expect("chained sums hold the 128-K accumulation gate");
}

/// [`sums_at_k_4096_hold_the_128_k_accumulation_gate`] in NVFP4, MXFP8 and
/// BF16 (the split's other K = 4096 grouped downs).
#[test]
fn sums_at_k_4096_hold_the_128_k_accumulation_gate_in_every_format() {
    // The BF16 bank of 288 experts at 4096 x 4096 takes 9 GiB.
    gpu_test(12 << 30, |session| {
        let mut failures = replay::Failures::default();
        let long = [Weights::Nvfp4, Weights::Mxfp8, Weights::Bf16]
            .map(|weights| accumulation_projection(weights, Schedule::Columns));
        for (i, p) in long.into_iter().enumerate() {
            let weights = p.weights();
            let result = (|| -> LabResult<()> {
                let (k, n) = (p.query.gemm.k as usize, p.columns() as usize);
                let local = grouped_experts(&p);
                let mut rng = Rng(4097 + i as u64);
                let active = [0, 17, 40, local - 1];
                let mut bank = Bank::new(session, weights, local, n, k, 1, &active, &mut rng)?;
                widen(session, &mut bank, weights, (n, k), &active, &mut rng)?;
                check_long_sums(session, accumulation::PREFIX, &p, &bank, active, &mut rng)
            })();
            failures.check(&format!("{weights:?}"), result);
        }
        failures.finish()
    })
    .expect("chained sums hold the 128-K accumulation gate in every format");
}

/// Large but finite FP8 operands and a gate below the point where
/// `exp(-gate)` overflows, through the shared expert's dense projections
/// (FP8, intermediate 512, decode tiles) on two tokens:
///
/// - token 0: x[0] = 2^120 against gate and up code 448 with block scale
///   2^-126. The raw product 448 * 2^120 exceeds F32; the decoded gate and up
///   are 7.
/// - token 1: x[1] = 12.75 * 2^120 against gate code -448 and up code 448:
///   gate -89.25 (no lower clamp), up 89.25 clamped to 10, so the unit is
///   about -1.6e-36, a normal BF16.
///
/// Down reads intermediate column 0 with code 448 and block scale 2^100, so
/// both results are finite and visible in the output.
#[test]
fn large_operands_and_the_silu_tail_stay_exact() {
    gpu_test(1 << 30, |session| {
        let intermediate = SHAPES.dense_intermediates[1];
        let i = intermediate as usize;
        let mut dense = Dense::new(session, Weights::Fp8Block128, intermediate, false, 7)?;
        let mut gate_up = vec![0_u8; 2 * i * HIDDEN];
        gate_up[0] = 0x7e;
        gate_up[1] = 0xfe;
        gate_up[i * HIDDEN] = 0x7e;
        gate_up[i * HIDDEN + 1] = 0x7e;
        let mut gate_up_scales = vec![1.0_f32; 2 * i / 128 * (HIDDEN / 128)];
        gate_up_scales[0] = 2_f32.powi(-126);
        gate_up_scales[i / 128 * (HIDDEN / 128)] = 2_f32.powi(-126);
        let mut down = vec![0_u8; HIDDEN * i];
        let mut down_scales = vec![1.0_f32; HIDDEN / 128 * (i / 128)];
        for row in 0..HIDDEN {
            down[row * i] = 0x7e;
        }
        for block in 0..HIDDEN / 128 {
            down_scales[block * (i / 128)] = 2_f32.powi(100);
        }
        dense.gate_up_bank = Bank::fp8(session, &gate_up, &gate_up_scales, HIDDEN)?;
        dense.down_bank = Bank::fp8(session, &down, &down_scales, i)?;
        let mut x = vec![0_u16; dense.capacity as usize * HIDDEN];
        x[0] = f32_to_bf16(2_f32.powi(120));
        x[HIDDEN + 1] = f32_to_bf16(12.75 * 2_f32.powi(120));
        dense.x_host = x;
        poison::write(
            session,
            &dense.x,
            0,
            &dense
                .x_host
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        poison::write(session, &dense.live, 0, &2_i32.to_ne_bytes())?;
        dense.launch(&session.stream, false)?;
        dense.check_oracle(session, 2)?;
        let inter = bf16s(&poison::read(session, &dense.inter)?);
        let (h0, h1) = (bf16_f64(inter[0]), bf16_f64(inter[i]));
        if !(h0 > 48.0 && h0 < 50.0 && h1 < -1e-36 && h1 > -1e-35) {
            return Err(format!("intermediate column 0 is {h0} and {h1}"));
        }
        let out = bf16s(&poison::read(session, &dense.out)?);
        let (y0, y1) = (bf16_f64(out[0]), bf16_f64(out[HIDDEN]));
        if !y0.is_finite() || y1 == 0.0 || !y1.is_finite() {
            return Err(format!("output column 0 is {y0} and {y1}"));
        }
        Ok(())
    })
    .expect("large operands and the SiLU tail stay exact");
}

/// One token whose route 0 goes to local expert 0 (its other routes go to no
/// expert of the rank) through a grouped down projection with route weight
/// 1: the raw F32 sums of `row` against expert 0, [N].
fn single_route_down(
    session: &Session,
    prefix: &str,
    p: &Projection,
    bank: &Bank,
    row: &[u16],
) -> LabResult<Vec<f32>> {
    let (k, n) = (p.query.gemm.k as usize, p.columns() as usize);
    let local = grouped_experts(p);
    let ids = (0..TOP_K)
        .map(|slot| if slot == 0 { 0 } else { local + slot })
        .collect::<Vec<_>>();
    let metadata = local_route_metadata(&ids, TOP_K, local, 0, p.row_tile as usize)
        .map_err(|e| e.0.to_string())?;
    let i32s = |v: &[i32]| bytes_of(v, i32::to_ne_bytes);
    let mut input = vec![0_u16; TOP_K * k];
    input[..k].copy_from_slice(row);
    let input = upload(
        session,
        &input
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let offsets = upload(session, &i32s(&metadata.offsets))?;
    let sorted = upload(session, &i32s(&metadata.sorted_rows))?;
    let padded = upload(session, &i32s(&[metadata.num_padded_rows]))?;
    let weights = upload(session, &bytes_of(&[1.0_f32; TOP_K], f32::to_ne_bytes))?;
    let output = session.allocate(TOP_K * n * 4)?;
    poison::fill_all(session, &output, &[POISON])?;
    let b = KernelArgument::Buffer;
    let [codes, block, local, globals] = bank.arguments();
    let args = [
        b(&input),
        codes,
        block,
        local,
        globals,
        b(&offsets),
        b(&sorted),
        b(&padded),
        KernelArgument::NullBuffer,
        b(&weights),
        b(&output),
    ];
    let f = Entry::projection(prefix, p).load(session)?;
    launch_with(
        &session.stream,
        &f,
        p.plan(1).map_err(|e| e.0.to_string())?,
        false,
        &args,
    )?;
    Ok(words(&poison::read(session, &output)?, f32::from_ne_bytes)[..n].to_vec())
}

/// Scale edges of the split operands (review round 2):
///
/// - Cancelling chains with a scale of 2^100: code 1 against +2^26 at K = 0,
///   16, 32, 48 and -2^26 at K = 64 to 112 sums to exactly zero. A scale of 1
///   or more stays out of the operand, so the chain never reaches 2^128.
/// - Zero scales against x[0] = 2^120 and code 448 (NVFP4: 6 x 448): the raw
///   product overflows F32, but a zero scale zeroes the operand, so the
///   projection is exactly zero rather than NaN.
///
/// FP8 uses the split's K = 4096 lab projection, NVFP4 the split's served
/// column down (the prefill row tile).
#[test]
fn scale_edges_cancel_and_zero_exactly() {
    gpu_test(2 << 30, |session| {
        let fp8 = accumulation_projection(Weights::Fp8Block128, Schedule::Columns);
        let nvfp4 = projection(
            Weights::Nvfp4,
            false,
            true,
            64,
            split::SHAPES.expert_intermediate,
        );
        let mut failures = replay::Failures::default();
        let cancelling = |k: usize| {
            let mut row = vec![0_u16; k];
            for (i, at) in (0..128).step_by(16).enumerate() {
                row[at] = f32_to_bf16(if i < 4 {
                    2_f32.powi(26)
                } else {
                    -(2_f32.powi(26))
                });
            }
            row
        };
        let huge = |k: usize| {
            let mut row = vec![0_u16; k];
            row[0] = f32_to_bf16(2_f32.powi(120));
            row
        };
        let zero = |label: &str, out: Vec<f32>| -> LabResult<()> {
            match out.iter().position(|&v| v != 0.0) {
                None => Ok(()),
                Some(at) => Err(format!("{label}: column {at} is {}", out[at])),
            }
        };
        for (label, cancel) in [("FP8 cancelling", true), ("FP8 zero scale", false)] {
            let result = (|| {
                let (k, n) = (fp8.query.gemm.k as usize, fp8.columns() as usize);
                let mut codes = vec![0_u8; n * k];
                let mut scales = vec![1.0_f32; n / 128 * (k / 128)];
                for r in 0..n {
                    if cancel {
                        for at in (0..128).step_by(16) {
                            codes[r * k + at] = 0x38;
                        }
                    } else {
                        codes[r * k] = 0x7e;
                    }
                }
                for block in 0..n / 128 {
                    scales[block * (k / 128)] = if cancel { 2_f32.powi(100) } else { 0.0 };
                }
                let bank = Bank::fp8(session, &codes, &scales, k)?;
                let row = if cancel { cancelling(k) } else { huge(k) };
                zero(
                    label,
                    single_route_down(session, accumulation::PREFIX, &fp8, &bank, &row)?,
                )
            })();
            failures.check(label, result);
        }
        for (label, cancel) in [("NVFP4 cancelling", true), ("NVFP4 zero global", false)] {
            let result = (|| {
                let (k, n) = (nvfp4.query.gemm.k as usize, nvfp4.columns() as usize);
                let mut codes = vec![0_u8; n * k / 2];
                let mut scales = vec![0x38_u8; n * k / 16];
                for r in 0..n {
                    if cancel {
                        for at in (0..128).step_by(16) {
                            codes[(r * k + at) / 2] = 0x02;
                        }
                    } else {
                        codes[r * k / 2] = 0x07;
                        scales[r * (k / 16)] = 0x7e;
                    }
                }
                let global = if cancel { 2_f32.powi(100) } else { 0.0 };
                let bank = Bank::nvfp4(session, &codes, &scales, global, k)?;
                let row = if cancel { cancelling(k) } else { huge(k) };
                zero(
                    label,
                    single_route_down(session, split::PREFIX, &nvfp4, &bank, &row)?,
                )
            })();
            failures.check(label, result);
        }
        failures.finish()
    })
    .expect("scale edges cancel and zero exactly");
}

/// The block-warp comparisons' route pools: the oracle pool (sixteen
/// experts spread over all 288), eight experts of the first [`QUARTER`]
/// (every route of a token lands there, two row tiles per expert at 32
/// tokens), and twelve experts in each quarter.
fn comparison_pools() -> [(&'static str, Vec<usize>); 3] {
    [
        ("sixteen spread", expert_pool()),
        ("eight", (0..8).map(|l| l * 9).collect()),
        (
            "twelve per quarter",
            (0..EXPERTS / QUARTER)
                .flat_map(|q| (0..12).map(move |l| q * QUARTER + l * 6))
                .collect(),
        ),
    ]
}

/// The routed comparison cases at `shapes`: each comparison pool on every
/// rank, then routes spread one per expert over rank 0's local experts
/// (`spread_logits`), as (name, pool, rank, spread).
fn comparison_cases(
    shapes: &chained::ServedShapes,
) -> Vec<(&'static str, Vec<usize>, usize, bool)> {
    let local = shapes.local_experts as usize;
    let mut cases = comparison_pools()
        .into_iter()
        .flat_map(|(name, pool)| {
            (0..EXPERTS / local).map(move |rank| (name, pool.clone(), rank, false))
        })
        .collect::<Vec<_>>();
    cases.push(("spread over rank 0", (0..local).collect(), 0, true));
    cases
}

/// Router logits that send token t's eight routes to rank 0's `local`
/// experts `(8 t + j) mod local`, j = 0..8, in descending order: of the
/// split's 288, 32 tokens reach 256 distinct experts with one route each.
/// Other experts sit at -8.
fn spread_logits(tokens: usize, local: usize) -> Vec<f32> {
    let mut logits = vec![-8.0_f32; tokens * EXPERTS];
    for token in 0..tokens {
        for j in 0..TOP_K {
            let expert = (TOP_K * token + j) % local;
            logits[token * EXPERTS + expert] = 3.0 - j as f32 / 64.0;
        }
    }
    logits
}

/// The bytes of `buffers` after poisoning them and running `launch`.
fn poisoned_outputs(
    session: &Session,
    buffers: &[&DeviceBuffer],
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<Vec<Vec<u8>>> {
    poison::poison(session, buffers)?;
    launch()?;
    buffers.iter().map(|b| poison::read(session, b)).collect()
}

/// The first differing element of two output sets (the column schedule's,
/// then a candidate schedule's), as `label`.
fn same_bytes(label: &str, columns: &[Vec<u8>], warps: &[Vec<u8>]) -> LabResult<()> {
    for (buffer, (a, b)) in columns.iter().zip(warps).enumerate() {
        if let Some(at) = a.iter().zip(b).position(|(x, y)| x != y) {
            return Err(format!(
                "{label}: buffer {buffer} byte {at} is {:#04x} under the candidate schedule, {:#04x} under columns",
                b[at], a[at]
            ));
        }
    }
    Ok(())
}

/// Block warps (`chained::Schedule::BlockWarps`) against the column
/// schedule they replace at decode, bit for bit, over the same weights,
/// inputs and routes: the split's NVFP4 routed layers (all 288 experts
/// the rank's, against the column references) under three route pools and
/// with routes spread one per expert over 256 of the 288 experts, and FP8
/// and BF16 dense MLPs at both intermediates, at every live count from 0
/// to 32. (The split's FP8 decode down in block warps is compared with its
/// columns in `split_expert_projections_match_the_oracle_in_every_format`.) Each
/// output buffer is poisoned before each run and compared whole, so the
/// written rows match too. Block warps run at the capacity's plan and at
/// the live count's own (the tightest grid).
#[test]
fn block_warps_match_the_column_schedule_bit_for_bit() {
    gpu_test(8 << 30, |session| {
        let mut failures = replay::Failures::default();
        // The pools, then routes spread one per expert over rank 0's local
        // experts (`comparison_cases`): the high expert-count edge of the
        // block warps' tight grid.
        let shapes = split::SHAPES;
        let weights = Weights::Nvfp4;
        let cases = comparison_cases(&shapes);
        for (seed, (pool_name, pool, rank, spread)) in cases.into_iter().enumerate() {
            {
                let local = shapes.local_experts as usize;
                let case = format!("{weights:?} layer, {pool_name}, rank {rank}");
                let result = (|| -> LabResult<()> {
                    let capacity = shapes.routes()[0].capacity as usize;
                    let layer = Layer::routed(
                        session,
                        &shapes,
                        weights,
                        false,
                        rank,
                        500 + seed as u64,
                        &pool,
                        spread.then(|| spread_logits(capacity, local)),
                        Schedule::Columns,
                    )?;
                    block_warp_layer_matches_columns(session, &layer)
                })();
                failures.check(&case, result);
            }
        }
        for (seed, weights) in [Weights::Fp8Block128, Weights::Bf16]
            .into_iter()
            .enumerate()
        {
            for intermediate in SHAPES.dense_intermediates {
                let case = format!("{weights:?} dense intermediate {intermediate}");
                let result = (|| -> LabResult<()> {
                    let dense =
                        Dense::new(session, weights, intermediate, false, 600 + seed as u64)?;
                    let warps = [&dense.gate_up, &dense.down]
                        .map(|p| p.block_warps().expect("block warps take the projection"));
                    let warp_fns = [
                        Entry::of(&warps[0]).load(session)?,
                        Entry::of(&warps[1]).load(session)?,
                    ];
                    let outputs = [&dense.inter, &dense.out];
                    for live in 0..=dense.capacity {
                        poison::write(session, &dense.live, 0, &(live as i32).to_ne_bytes())?;
                        let columns = poisoned_outputs(session, &outputs, || {
                            dense.launch(&session.stream, false)
                        })?;
                        for tokens in [dense.capacity, live.max(1)] {
                            let blocks = poisoned_outputs(session, &outputs, || {
                                dense.launch_projections(
                                    &session.stream,
                                    [&warps[0], &warps[1]],
                                    [&warp_fns[0], &warp_fns[1]],
                                    tokens,
                                    false,
                                )
                            })?;
                            same_bytes(
                                &format!("live {live}, planned at {tokens}"),
                                &columns,
                                &blocks,
                            )?;
                        }
                    }
                    Ok(())
                })();
                failures.check(&case, result);
            }
        }
        failures.finish()
    })
    .expect("block warps match the column schedule bit for bit");
}

/// A routed layer built on the column schedule against its projections in
/// block warps, bit for bit: at every live count from 0 to the capacity,
/// the whole poisoned intermediate and weighted-route outputs, with block
/// warps planned at the capacity and at the live count.
fn block_warp_layer_matches_columns(session: &Session, layer: &Layer) -> LabResult<()> {
    layer_matches_columns(session, layer, Schedule::BlockWarps)
}

/// [`block_warp_layer_matches_columns`] for the layer's projections in
/// `schedule`.
fn layer_matches_columns(session: &Session, layer: &Layer, schedule: Schedule) -> LabResult<()> {
    let warps = scheduled([layer.gate_up.clone(), layer.down.clone()], schedule);
    let warp_fns = [
        Entry::of(&warps[0]).load(session)?,
        Entry::of(&warps[1]).load(session)?,
    ];
    let outputs = [&layer.inter, &layer.routed];
    for live in 0..=layer.route.capacity {
        layer.routes.set_live(session, live)?;
        let columns = poisoned_outputs(session, &outputs, || layer.launch(&session.stream, false))?;
        for tokens in [layer.route.capacity, live.max(1)] {
            let blocks = poisoned_outputs(session, &outputs, || {
                layer.launch_projections(
                    &session.stream,
                    [&warps[0], &warps[1]],
                    [&warp_fns[0], &warp_fns[1]],
                    tokens,
                    false,
                )
            })?;
            same_bytes(
                &format!("live {live}, planned at {tokens}"),
                &columns,
                &blocks,
            )?;
        }
    }
    Ok(())
}

/// The globals `nvfp4_block_warp_scale_edges` spreads over its layers, in
/// the F32 ranges `chained::split_scale` distinguishes: exact zero, the
/// smallest subnormal and 2^-130 (subnormals, with and without a varied
/// mantissa), exactly 2^-126 (the smallest normal) and above it, 2^-101,
/// exactly 2^-100 (the split floor), above it and 2^-99, small values, 1/4,
/// [1, 2) and above (kept whole in the residual) up to exactly 2^60 and a
/// varied mantissa there.
fn edge_globals() -> Vec<f32> {
    [
        (0.0, 0),
        (1.0, -149),
        (1.0, -130),
        (1.5, -130),
        (1.0, -126),
        (1.37, -126),
        (1.0, -101),
        (1.0, -100),
        (1.63, -100),
        (1.0, -99),
        (1.0, -64),
        (1.11, -20),
        (1.0, -2),
        (1.0, 0),
        (1.5, 0),
        (1.0, 20),
        (1.0, 60),
        (1.21, 60),
    ]
    .into_iter()
    .map(|(mantissa, exponent): (f64, i32)| (mantissa * 2_f64.powi(exponent)) as f32)
    .collect()
}

/// A bank's globals, `experts` x `halves` in expert-major order: slot
/// (expert, half) takes `edge_globals()[(expert + half + rotation) % len]`,
/// so with at least `len` experts every edge value lands in every half
/// (gate/up rotation 0: gate `e`, up `e + 1`; down rotation 2), and an
/// expert's gate and up differ.
fn spread_globals(experts: usize, halves: usize, rotation: usize) -> Vec<f32> {
    let values = edge_globals();
    (0..experts * halves)
        .map(|slot| values[(slot / halves + slot % halves + rotation) % values.len()])
        .collect()
}

/// Runs without a GPU: `spread_globals` puts every edge value, bit for bit,
/// into the gate, the up and the down globals of the split's 288 local
/// experts.
#[test]
fn spread_globals_hold_every_edge_in_every_half() {
    let values = edge_globals();
    let local = split::SHAPES.local_experts as usize;
    assert!(local >= values.len());
    let gate_up = spread_globals(local, 2, 0);
    let down = spread_globals(local, 1, 2);
    let gate = gate_up.iter().step_by(2).collect::<Vec<_>>();
    let up = gate_up.iter().skip(1).step_by(2).collect::<Vec<_>>();
    for value in &values {
        for (name, half) in [("gate", &gate), ("up", &up)] {
            assert!(
                half.iter().any(|v| v.to_bits() == value.to_bits()),
                "{name} lacks {value:e}"
            );
        }
        assert!(
            down.iter().any(|v| v.to_bits() == value.to_bits()),
            "down lacks {value:e}"
        );
    }
    for exact in [2_f32.powi(-126), 2_f32.powi(-100), 2_f32.powi(60)] {
        assert!(
            values.iter().any(|v| v.to_bits() == exact.to_bits()),
            "{exact:e}"
        );
    }
    assert!(values.iter().any(|v| v.is_subnormal()));
    assert!((0..local).all(|e| gate[e].to_bits() != up[e].to_bits()));
}

/// NVFP4 block-warp scale edges (review PF6b-1), separately runnable:
///
/// - `scale_edges_cancel_and_zero_exactly`'s NVFP4 cases (cancelling chains
///   under a 2^100 global; a zero global against an F32-overflowing raw
///   product) through the split's served decode down in block warps:
///   exactly zero, and the column reference's bytes.
/// - The split's routed NVFP4 layers whose gate, up and down globals rotate
///   every `edge_globals` value through each half (`spread_globals`), block
///   warps against the column references bit for bit at every live count
///   (`block_warp_layer_matches_columns`): rank 0 with routes spread over
///   256 of its 288 local experts (every value in every half routed), and
///   the first comparison pool (its experts' subset).
#[test]
fn nvfp4_block_warp_scale_edges() {
    gpu_test(8 << 30, |session| {
        let columns = projection(
            Weights::Nvfp4,
            false,
            true,
            16,
            split::SHAPES.expert_intermediate,
        );
        let warps = columns
            .block_warps()
            .map_err(|e| format!("block warps refuse the NVFP4 down: {}", e.0))?;
        let mut failures = replay::Failures::default();
        let (k, n) = (columns.query.gemm.k as usize, columns.columns() as usize);
        for (label, cancel) in [("NVFP4 cancelling", true), ("NVFP4 zero global", false)] {
            let result = (|| -> LabResult<()> {
                let mut codes = vec![0_u8; n * k / 2];
                let mut scales = vec![0x38_u8; n * k / 16];
                let mut row = vec![0_u16; k];
                for r in 0..n {
                    if cancel {
                        for at in (0..128).step_by(16) {
                            codes[(r * k + at) / 2] = 0x02;
                        }
                    } else {
                        codes[r * k / 2] = 0x07;
                        scales[r * (k / 16)] = 0x7e;
                    }
                }
                if cancel {
                    for (i, at) in (0..128).step_by(16).enumerate() {
                        let magnitude = 2_f32.powi(26);
                        row[at] = f32_to_bf16(if i < 4 { magnitude } else { -magnitude });
                    }
                } else {
                    row[0] = f32_to_bf16(2_f32.powi(120));
                }
                let global = if cancel { 2_f32.powi(100) } else { 0.0 };
                let bank = Bank::nvfp4(session, &codes, &scales, global, k)?;
                let want = single_route_down(session, references::PREFIX, &columns, &bank, &row)?;
                let out = single_route_down(session, split::PREFIX, &warps, &bank, &row)?;
                if let Some(at) = out.iter().position(|&v| v != 0.0) {
                    return Err(format!("block warps: column {at} is {}", out[at]));
                }
                if let Some(at) = (0..n).find(|&i| out[i].to_bits() != want[i].to_bits()) {
                    return Err(format!(
                        "column {at}: block warps {:#x}, columns {:#x}",
                        out[at].to_bits(),
                        want[at].to_bits()
                    ));
                }
                Ok(())
            })();
            failures.check(label, result);
        }
        let shapes = split::SHAPES;
        let local = shapes.local_experts as usize;
        let capacity = shapes.routes()[0].capacity as usize;
        let pools = [
            ("spread over rank 0", (0..local).collect::<Vec<_>>(), true),
            (
                comparison_pools()[0].0,
                comparison_pools()[0].1.clone(),
                false,
            ),
        ];
        for (seed, (pool_name, pool, spread)) in pools.into_iter().enumerate() {
            let case = format!("NVFP4 layer, spread globals, {pool_name}");
            let result = (|| -> LabResult<()> {
                let layer = Layer::routed(
                    session,
                    &shapes,
                    Weights::Nvfp4,
                    false,
                    0,
                    800 + seed as u64,
                    &pool,
                    spread.then(|| spread_logits(capacity, local)),
                    Schedule::Columns,
                )?;
                for (halves, rotation, bank) in
                    [(2, 0, &layer.gate_up_bank), (1, 2, &layer.down_bank)]
                {
                    let globals = bank.globals.as_ref().expect("NVFP4 banks hold globals");
                    let values = spread_globals(local, halves, rotation);
                    if values.len() * 4 != globals.byte_len() {
                        return Err("globals are not local experts x halves".into());
                    }
                    poison::write(session, globals, 0, &bytes_of(&values, f32::to_ne_bytes))?;
                }
                block_warp_layer_matches_columns(session, &layer)
            })();
            failures.check(&case, result);
        }
        failures.finish()
    })
    .expect("NVFP4 block warps keep the column schedule's bits at the scale edges");
}

/// Block-warp MLPs against the oracle, with poisoned eager, relaunch,
/// graph replay and PDL-producer launches at every `poison::live_counts`
/// count (as `dense_and_shared_mlps_match_the_oracle`). The split's routed
/// block-warp forms meet the oracle in
/// `split_expert_projections_match_the_oracle_in_every_format`.
#[test]
fn block_warp_layers_and_mlps_match_the_oracle() {
    gpu_test(8 << 30, |session| {
        let mut failures = replay::Failures::default();
        for (seed, weights) in [Weights::Fp8Block128, Weights::Bf16]
            .into_iter()
            .enumerate()
        {
            for intermediate in SHAPES.dense_intermediates {
                let case = format!("{weights:?} block-warp dense intermediate {intermediate}");
                let result = (|| -> LabResult<()> {
                    let dense = Dense::scheduled(
                        session,
                        weights,
                        intermediate,
                        false,
                        800 + seed as u64,
                        Schedule::BlockWarps,
                    )?;
                    let graph = session.capture(|stream| dense.launch(stream, true))?;
                    for live in poison::live_counts(dense.capacity) {
                        poison::write(session, &dense.live, 0, &(live as i32).to_ne_bytes())?;
                        let mut checks = replay::Checks::new(session)
                            .output(
                                "intermediate",
                                &dense.inter,
                                WriteMap::of(DType::Bf16)
                                    .prefix(live as usize * dense.intermediate)?,
                            )
                            .output(
                                "output",
                                &dense.out,
                                WriteMap::of(DType::Bf16).prefix(live as usize * HIDDEN)?,
                            );
                        checks
                            .phase(Phase::Eager, false, || dense.launch(&session.stream, false))?;
                        checks.phase(Phase::Relaunch, false, || {
                            dense.launch(&session.stream, false)
                        })?;
                        checks.phase(Phase::Graph, true, || {
                            graph.launch(&session.stream).map_err(err)
                        })?;
                        checks.phase(Phase::Pdl, true, || dense.launch(&session.stream, true))?;
                    }
                    poison::write(
                        session,
                        &dense.live,
                        0,
                        &(dense.capacity as i32).to_ne_bytes(),
                    )?;
                    dense.launch(&session.stream, false)?;
                    dense.check_oracle(session, dense.capacity as usize)
                })();
                failures.check(&case, result);
            }
        }
        failures.finish()
    })
    .expect("block-warp layers and MLPs match the oracle");
}

/// The 128-K accumulation gate and the scale edges of
/// `scale_edges_cancel_and_zero_exactly` through the split's block-warp
/// K = 4096 grouped down (`split::accumulation_projections`).
#[test]
fn block_warp_sums_hold_the_128_k_accumulation_gate() {
    gpu_test(6 << 30, |session| {
        let p = accumulation_projection(Weights::Fp8Block128, Schedule::BlockWarps);
        let (k, n) = (p.query.gemm.k as usize, p.columns() as usize);
        let local = grouped_experts(&p);
        let mut failures = replay::Failures::default();
        let result = (|| -> LabResult<()> {
            let mut rng = Rng(4096);
            let active = [0, 17, 40, local - 1];
            let mut bank = Bank::new(
                session,
                Weights::Fp8Block128,
                local,
                n,
                k,
                1,
                &active,
                &mut rng,
            )?;
            widen(
                session,
                &mut bank,
                Weights::Fp8Block128,
                (n, k),
                &active,
                &mut rng,
            )?;
            check_long_sums(session, accumulation::PREFIX, &p, &bank, active, &mut rng)
        })();
        failures.check("128-K gate", result);
        for (label, cancel) in [("cancelling", true), ("zero scale", false)] {
            let result = (|| -> LabResult<()> {
                let mut codes = vec![0_u8; n * k];
                let mut scales = vec![1.0_f32; n / 128 * (k / 128)];
                let mut row = vec![0_u16; k];
                for r in 0..n {
                    if cancel {
                        for at in (0..128).step_by(16) {
                            codes[r * k + at] = 0x38;
                        }
                    } else {
                        codes[r * k] = 0x7e;
                    }
                }
                for block in 0..n / 128 {
                    scales[block * (k / 128)] = if cancel { 2_f32.powi(100) } else { 0.0 };
                }
                if cancel {
                    for (i, at) in (0..128).step_by(16).enumerate() {
                        let magnitude = 2_f32.powi(26);
                        row[at] = f32_to_bf16(if i < 4 { magnitude } else { -magnitude });
                    }
                } else {
                    row[0] = f32_to_bf16(2_f32.powi(120));
                }
                let bank = Bank::fp8(session, &codes, &scales, k)?;
                let out = single_route_down(session, accumulation::PREFIX, &p, &bank, &row)?;
                match out.iter().position(|&v| v != 0.0) {
                    None => Ok(()),
                    Some(at) => Err(format!("column {at} is {}", out[at])),
                }
            })();
            failures.check(label, result);
        }
        failures.finish()
    })
    .expect("block-warp sums hold the 128-K accumulation gate");
}

#[path = "../block_warp_mutant_shapes.rs"]
mod block_warp_mutant_shapes;

/// The PDL producer probe (`pdl_prefetch_producer`, as in
/// `tiled_chain_splits`): after its own wait it saves the activation it is
/// given, overwrites it with BF16 NaN pairs, releases its dependents, waits
/// `PRODUCER_TAIL_NS` and only then restores the activation, so a consumer
/// that reads its input before its own dependency wait sees NaN.
struct Producer {
    function: Function,
    eviction: DeviceBuffer,
    checksum: DeviceBuffer,
    live: DeviceBuffer,
    /// CTAs of a launch: one, or [`WIDE_PRODUCER_CTAS`] for the device
    /// checks at every count, which poison and restore whole activations
    /// thousands of times.
    ctas: u32,
}

/// CTAs of a wide producer: one SM in six, so the consumer's CTAs (168
/// registers over 384 threads, which share no SM with a producer CTA) find
/// the other SMs free and start during the producer's tail.
const WIDE_PRODUCER_CTAS: u32 = 32;

/// Longer than any checked projection's run.
const PRODUCER_TAIL_NS: u64 = 100_000;

impl Producer {
    fn new(session: &Session) -> LabResult<Self> {
        Self::with_ctas(session, 1)
    }

    /// The producer over [`WIDE_PRODUCER_CTAS`] CTAs: the same poison,
    /// release, tail and restore, each CTA over its stride of the words, on
    /// few enough SMs that its dependents overlap it.
    fn wide(session: &Session) -> LabResult<Self> {
        Self::with_ctas(session, WIDE_PRODUCER_CTAS)
    }

    fn with_ctas(session: &Session, ctas: u32) -> LabResult<Self> {
        Ok(Self {
            function: session
                .module("probe_pdl_prefetch_sm120")?
                .function("chop_probe_pdl_prefetch_producer_v1")
                .map_err(err)?,
            eviction: upload(session, &[0; 16])?,
            checksum: session.allocate(ctas as usize * 128 * 4)?,
            live: session.allocate(4)?,
            ctas,
        })
    }

    /// The descriptor that holds `activation` poisoned, saving it to `fresh`
    /// (`PdlOperand`: fresh, input, words, flags preserve | floating poison).
    fn operand(
        session: &Session,
        activation: &DeviceBuffer,
        fresh: &DeviceBuffer,
    ) -> LabResult<DeviceBuffer> {
        let bytes = activation.byte_len();
        let mut descriptor = Vec::with_capacity(24);
        descriptor.extend_from_slice(
            &fresh
                .device_address_at(0, bytes)
                .map_err(err)?
                .to_ne_bytes(),
        );
        descriptor.extend_from_slice(
            &activation
                .device_address_at(0, bytes)
                .map_err(err)?
                .to_ne_bytes(),
        );
        descriptor.extend_from_slice(&u32::try_from(bytes / 4).map_err(err)?.to_ne_bytes());
        descriptor.extend_from_slice(&3_u32.to_ne_bytes());
        upload(session, &descriptor)
    }

    fn launch(&self, stream: &Stream, operand: &DeviceBuffer) -> LabResult<()> {
        self.function
            .launch(
                stream,
                launch::simple_config(
                    (self.ctas, 1, 1),
                    (128, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    true,
                )?,
                &[
                    KernelArgument::Buffer(&self.eviction),
                    KernelArgument::Buffer(&self.checksum),
                    KernelArgument::U64(1),
                    KernelArgument::Buffer(operand),
                    KernelArgument::U32(1),
                    KernelArgument::Buffer(&self.live),
                    KernelArgument::U32(1),
                    KernelArgument::U64(PRODUCER_TAIL_NS),
                ],
            )
            .map_err(err)
    }
}

/// Three dense MLPs of one shape (each with its own weights and input), at
/// the decode or prefill capacity, with the producer's save areas and
/// descriptors for their inputs.
struct Mlps {
    mlps: Vec<Dense>,
    fresh: Vec<DeviceBuffer>,
    operands: Vec<DeviceBuffer>,
}

impl Mlps {
    fn new(session: &Session, weights: Weights, intermediate: u32, seed: u64) -> LabResult<Self> {
        Self::at(session, weights, intermediate, false, seed)
    }

    fn at(
        session: &Session,
        weights: Weights,
        intermediate: u32,
        prefill: bool,
        seed: u64,
    ) -> LabResult<Self> {
        let mlps = (0..3)
            .map(|i| Dense::new(session, weights, intermediate, prefill, seed + i))
            .collect::<LabResult<Vec<_>>>()?;
        let fresh = mlps
            .iter()
            .map(|d| session.allocate(d.x.byte_len()))
            .collect::<LabResult<Vec<_>>>()?;
        let operands = mlps
            .iter()
            .zip(&fresh)
            .map(|(d, f)| Producer::operand(session, &d.x, f))
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            mlps,
            fresh,
            operands,
        })
    }

    fn set_live(&self, session: &Session, live: u32) -> LabResult<()> {
        for d in &self.mlps {
            poison::write(session, &d.live, 0, &(live as i32).to_ne_bytes())?;
        }
        Ok(())
    }

    /// The intermediate and output write maps at `live` tokens.
    fn maps(&self, live: u32) -> LabResult<[WriteMap; 2]> {
        let d = &self.mlps[0];
        Ok([
            WriteMap::of(DType::Bf16).prefix(live as usize * d.intermediate)?,
            WriteMap::of(DType::Bf16).prefix(live as usize * HIDDEN)?,
        ])
    }

    /// Each MLP's intermediate and output bytes after its own launch of
    /// `projections` from poisoned outputs, checked against the write maps.
    fn separate(
        &self,
        session: &Session,
        projections: [&Projection; 2],
        fns: [&Function; 2],
        live: u32,
    ) -> LabResult<Vec<[Vec<u8>; 2]>> {
        let maps = self.maps(live)?;
        self.mlps
            .iter()
            .map(|d| {
                poison::poison(session, &[&d.inter, &d.out])?;
                d.launch_projections(&session.stream, projections, fns, d.capacity, false)?;
                session.stream.synchronize().map_err(err)?;
                let bytes = [
                    poison::read(session, &d.inter)?,
                    poison::read(session, &d.out)?,
                ];
                for (name, (b, map)) in ["intermediate", "output"]
                    .iter()
                    .zip(bytes.iter().zip(&maps))
                {
                    poison::check(
                        &format!("{} live {live} {name}", projections[0].symbol),
                        b,
                        map,
                        poison::Fill::Pattern(&[POISON]),
                    )?;
                }
                Ok(bytes)
            })
            .collect()
    }
}

/// The first differing byte of two output images, as text.
fn first_difference(label: &str, want: &[u8], got: &[u8]) -> LabResult<()> {
    match want.iter().zip(got).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(at) => Err(format!("{label}: byte {at} differs")),
    }
}

/// `tiled_chain_splits`' qualification for chained MLP projections (block
/// warps or their mutants) against `expected` (each MLP's column-schedule
/// intermediate and output at `live` tokens):
///
/// - each MLP in its own launch, byte for byte;
/// - eager, relaunched, then behind the PDL producer, which holds MLP 0's
///   input at NaN until its tail ends;
/// - one sequence without host synchronization, every launch PDL: producer,
///   MLP 0, producer, MLP 1, then MLP 2 directly behind MLP 1; run eager,
///   then from one captured graph replayed twice. Every output keeps its
///   write map and equals its expected bytes.
fn qualify_mlps(
    session: &Session,
    mlps: &Mlps,
    projections: [&Projection; 2],
    fns: [&Function; 2],
    producer: &Producer,
    live: u32,
    expected: &[[Vec<u8>; 2]],
) -> LabResult<serde_json::Value> {
    let label = |what: &str| format!("{} live {live} {what}", projections[0].symbol);
    mlps.set_live(session, live)?;
    for (i, (want, got)) in expected
        .iter()
        .zip(mlps.separate(session, projections, fns, live)?)
        .enumerate()
    {
        for (w, g) in want.iter().zip(&got) {
            first_difference(&label(&format!("separate MLP {i}")), w, g)?;
        }
    }
    let stream = &session.stream;
    let run = |i: usize, pdl: bool, stream: &Stream| {
        let d = &mlps.mlps[i];
        d.launch_projections(stream, projections, fns, d.capacity, pdl)
    };
    let [inter_map, out_map] = mlps.maps(live)?;
    let first = &mlps.mlps[0];
    let mut replay = Replay::new(session)
        .output("intermediate", &first.inter, inter_map.clone())
        .output("output", &first.out, out_map.clone())
        .scratch("saved input", &mlps.fresh[0]);
    replay.eager(|| run(0, false, stream))?;
    replay.relaunch(|| run(0, false, stream))?;
    replay.pdl(
        || producer.launch(stream, &mlps.operands[0]),
        || run(0, true, stream),
        true,
    )?;
    for (name, want) in ["intermediate", "output"].iter().zip(&expected[0]) {
        first_difference(
            &label(&format!("behind the producer {name}")),
            want,
            replay.expected(name).ok_or("missing eager output")?,
        )?;
    }
    let mut pdl = BTreeMap::new();
    replay.record(&mut pdl);
    let sequence = |stream: &Stream| -> LabResult<()> {
        producer.launch(stream, &mlps.operands[0])?;
        run(0, true, stream)?;
        producer.launch(stream, &mlps.operands[1])?;
        run(1, true, stream)?;
        run(2, true, stream)
    };
    let mut replay = Replay::new(session);
    for (i, d) in mlps.mlps.iter().enumerate() {
        replay = replay
            .output(format!("intermediate {i}"), &d.inter, inter_map.clone())
            .output(format!("output {i}"), &d.out, out_map.clone());
    }
    let mut replay = replay
        .scratch("saved input 0", &mlps.fresh[0])
        .scratch("saved input 1", &mlps.fresh[1]);
    replay.eager(|| sequence(stream))?;
    for (i, want) in expected.iter().enumerate() {
        for (name, w) in ["intermediate", "output"].iter().zip(want) {
            let got = replay
                .expected(&format!("{name} {i}"))
                .ok_or("missing eager output")?;
            first_difference(&label(&format!("sequence {name} {i}")), w, got)?;
        }
    }
    let graph = session.capture(sequence)?;
    replay.graph(&graph)?;
    replay.graph(&graph)?;
    let mut back_to_back = BTreeMap::new();
    replay.record(&mut back_to_back);
    Ok(serde_json::json!({"live": live, "behind_producer": pdl, "back_to_back": back_to_back}))
}

/// The phases of `qualify_mlps` in which `projections`' intermediate or
/// output bytes differ from `expected`, each launch from poisoned outputs
/// and save areas, read back after a host synchronization and compared
/// whole (bytes and write maps): each MLP separately; MLP 0 behind the
/// producer; the back-to-back sequence eager, then from one captured graph
/// replayed twice. Preparation, launch, capture, synchronization and
/// readback errors propagate; only a compared output difference counts.
/// `inject_failure` returns an operational error before the first launch,
/// for the control that such an error fails the test.
#[allow(clippy::too_many_arguments)]
fn mlp_mutant_differences(
    session: &Session,
    mlps: &Mlps,
    projections: [&Projection; 2],
    fns: [&Function; 2],
    producer: &Producer,
    live: u32,
    expected: &[[Vec<u8>; 2]],
    inject_failure: bool,
) -> LabResult<Vec<String>> {
    if inject_failure {
        return Err("injected operational failure before the first launch".into());
    }
    mlps.set_live(session, live)?;
    let stream = &session.stream;
    let run = |i: usize, pdl: bool, stream: &Stream| {
        let d = &mlps.mlps[i];
        d.launch_projections(stream, projections, fns, d.capacity, pdl)
    };
    let mut differing = Vec::new();
    let mut compare = |phase: &str, mlp_indices: &[usize]| -> LabResult<()> {
        stream.synchronize().map_err(err)?;
        for &i in mlp_indices {
            let d = &mlps.mlps[i];
            for (name, (buffer, want)) in ["intermediate", "output"]
                .iter()
                .zip([&d.inter, &d.out].into_iter().zip(&expected[i]))
            {
                if &poison::read(session, buffer)? != want {
                    differing.push(format!("{phase}: MLP {i} {name}"));
                }
            }
        }
        Ok(())
    };
    let outputs = |i: usize| [&mlps.mlps[i].inter, &mlps.mlps[i].out];
    for i in 0..mlps.mlps.len() {
        poison::poison(session, &outputs(i))?;
        run(i, false, stream)?;
        compare("separate", &[i])?;
    }
    poison::poison(
        session,
        &[&mlps.mlps[0].inter, &mlps.mlps[0].out, &mlps.fresh[0]],
    )?;
    producer.launch(stream, &mlps.operands[0])?;
    run(0, true, stream)?;
    compare("behind the producer", &[0])?;
    let sequence = |stream: &Stream| -> LabResult<()> {
        producer.launch(stream, &mlps.operands[0])?;
        run(0, true, stream)?;
        producer.launch(stream, &mlps.operands[1])?;
        run(1, true, stream)?;
        run(2, true, stream)
    };
    let mut scratch: Vec<&DeviceBuffer> = (0..mlps.mlps.len()).flat_map(outputs).collect();
    scratch.extend([&mlps.fresh[0], &mlps.fresh[1]]);
    let all = [0, 1, 2];
    poison::poison(session, &scratch)?;
    sequence(stream)?;
    compare("sequence", &all)?;
    let graph = session.capture(sequence)?;
    for replay in 0..2 {
        poison::poison(session, &scratch)?;
        graph.launch(stream).map_err(err)?;
        compare(&format!("graph replay {replay}"), &all)?;
    }
    Ok(differing)
}

/// Block-warp dense and shared-expert MLPs (FP8 and BF16, both
/// intermediates) pass `qualify_mlps` against the column schedule at every
/// non-zero `poison::live_counts` count.
#[test]
fn block_warp_mlps_pass_behind_a_real_producer_and_back_to_back() {
    gpu_test(4 << 30, |session| {
        let producer = Producer::new(session)?;
        let mut failures = replay::Failures::default();
        for (seed, weights) in [Weights::Fp8Block128, Weights::Bf16]
            .into_iter()
            .enumerate()
        {
            for intermediate in SHAPES.dense_intermediates {
                let case = format!("{weights:?} dense intermediate {intermediate}");
                let result = (|| -> LabResult<()> {
                    let mlps = Mlps::new(session, weights, intermediate, 900 + 10 * seed as u64)?;
                    let first = &mlps.mlps[0];
                    let columns = [&first.gate_up, &first.down];
                    let warps =
                        columns.map(|p| p.block_warps().expect("block warps take the projection"));
                    let warp_fns = [
                        Entry::of(&warps[0]).load(session)?,
                        Entry::of(&warps[1]).load(session)?,
                    ];
                    let column_fns = [&first.fns[0], &first.fns[1]];
                    for live in poison::live_counts(first.capacity)
                        .into_iter()
                        .filter(|&live| live > 0)
                    {
                        mlps.set_live(session, live)?;
                        let expected = mlps.separate(session, columns, column_fns, live)?;
                        qualify_mlps(
                            session,
                            &mlps,
                            [&warps[0], &warps[1]],
                            [&warp_fns[0], &warp_fns[1]],
                            &producer,
                            live,
                            &expected,
                        )?;
                    }
                    Ok(())
                })();
                failures.check(&case, result);
            }
        }
        failures.finish()
    })
    .expect("block-warp MLPs pass behind a real producer and back to back");
}

/// Routed block-warp layers behind their real producers: the route kernels
/// (which write the sorted rows and padded counts the projections read) and
/// the producer probe holding the input at NaN, all PDL in one sequence
/// (routes, probe, gate/up, down), equal to the column schedule's separate
/// launches at every non-zero `poison::live_counts` count; eager, then one
/// captured graph replayed twice, from poisoned outputs each time. NVFP4
/// on the split's rank 0 (all 288 experts its own, against the column
/// references), under two route pools. (The split's FP8 decode forms run
/// behind their routes and live producers in the kit's split routed
/// cases.)
#[test]
fn block_warp_layers_pass_behind_their_routes_and_a_real_producer() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let mut failures = replay::Failures::default();
        let weights = Weights::Nvfp4;
        for (seed, (pool_name, pool)) in comparison_pools().into_iter().take(2).enumerate() {
            let result = (|| -> LabResult<()> {
                let layer = Layer::pooled(
                    session,
                    &split::SHAPES,
                    weights,
                    false,
                    0,
                    950 + seed as u64,
                    &pool,
                    Schedule::Columns,
                )?;
                let warps = [&layer.gate_up, &layer.down]
                    .map(|p| p.block_warps().expect("block warps take the projection"));
                let warp_fns = [
                    Entry::of(&warps[0]).load(session)?,
                    Entry::of(&warps[1]).load(session)?,
                ];
                let fresh = session.allocate(layer.x.byte_len())?;
                let operand = Producer::operand(session, &layer.x, &fresh)?;
                let outputs = [&layer.inter, &layer.routed];
                let capacity = layer.route.capacity;
                let sequence = |stream: &Stream| -> LabResult<()> {
                    layer.routes.launch(
                        stream,
                        layer.route,
                        &layer.route_fns,
                        layer.rank as u32,
                        true,
                    )?;
                    producer.launch(stream, &operand)?;
                    layer.launch_projections(
                        stream,
                        [&warps[0], &warps[1]],
                        [&warp_fns[0], &warp_fns[1]],
                        capacity,
                        true,
                    )
                };
                for live in poison::live_counts(capacity)
                    .into_iter()
                    .filter(|&live| live > 0)
                {
                    layer.routes.set_live(session, live)?;
                    let columns = poisoned_outputs(session, &outputs, || {
                        layer.launch(&session.stream, false)
                    })?;
                    let eager =
                        poisoned_outputs(session, &[&layer.inter, &layer.routed, &fresh], || {
                            sequence(&session.stream)
                        })?;
                    same_bytes(
                        &format!("{pool_name} live {live} eager"),
                        &columns,
                        &eager[..2],
                    )?;
                    let graph = session.capture(sequence)?;
                    for replay in 0..2 {
                        let replayed = poisoned_outputs(
                            session,
                            &[&layer.inter, &layer.routed, &fresh],
                            || graph.launch(&session.stream).map_err(err),
                        )?;
                        same_bytes(
                            &format!("{pool_name} live {live} graph replay {replay}"),
                            &columns,
                            &replayed[..2],
                        )?;
                    }
                }
                Ok(())
            })();
            failures.check(&format!("{weights:?} layer, {pool_name}"), result);
        }
        failures.finish()
    })
    .expect("block-warp layers pass behind their routes and a real producer");
}

/// The checks of `qualify_mlps` pass on the unmodified lab instance of the
/// shared expert's FP8 block-warp MLP (and `mlp_mutant_differences` finds
/// no difference), and each mutant (`block_warp_mutant_shapes`) has outputs
/// that differ from the column schedule's in some phase of
/// `mlp_mutant_differences`: no or a late dependency wait (behind the
/// producer) and a reversed chain fold (every phase). A mutant counts as
/// caught only on an observed output difference; operational errors fail
/// the test, which an injected failure before the first launch shows.
#[test]
fn block_warp_checks_reject_every_mutant() {
    gpu_test(2 << 30, |session| {
        let module = session.module(block_warp_mutant_shapes::CUBIN)?;
        let producer = Producer::new(session)?;
        let projections = block_warp_mutant_shapes::projections();
        let intermediate = projections[0].columns();
        let mlps = Mlps::new(session, Weights::Fp8Block128, intermediate, 990)?;
        let first = &mlps.mlps[0];
        let live = first.capacity;
        mlps.set_live(session, live)?;
        let expected = mlps.separate(
            session,
            [&first.gate_up, &first.down],
            [&first.fns[0], &first.fns[1]],
            live,
        )?;
        let load = |mutation: &str| -> LabResult<[Function; 2]> {
            Ok([
                module
                    .function(&block_warp_mutant_shapes::symbol(mutation, true))
                    .map_err(err)?,
                module
                    .function(&block_warp_mutant_shapes::symbol(mutation, false))
                    .map_err(err)?,
            ])
        };
        let [p0, p1] = &projections;
        let control = load(block_warp_mutant_shapes::MUTATIONS[0])?;
        qualify_mlps(
            session,
            &mlps,
            [p0, p1],
            [&control[0], &control[1]],
            &producer,
            live,
            &expected,
        )
        .map_err(|e| format!("the unmodified instance failed: {e}"))?;
        let differences = |fns: &[Function; 2], inject: bool| {
            mlp_mutant_differences(
                session,
                &mlps,
                [p0, p1],
                [&fns[0], &fns[1]],
                &producer,
                live,
                &expected,
                inject,
            )
        };
        let control_differences = differences(&control, false)?;
        if !control_differences.is_empty() {
            return Err(format!(
                "the unmodified instance differs: {control_differences:?}"
            ));
        }
        if differences(&control, true).is_ok() {
            return Err("an injected operational failure did not propagate".into());
        }
        let mut results = Vec::new();
        for &mutation in &block_warp_mutant_shapes::MUTATIONS[1..] {
            let found = differences(&load(mutation)?, false)?;
            if found.is_empty() {
                return Err(format!("the {mutation} mutant passed"));
            }
            eprintln!("{mutation}: outputs differ in {found:?}");
            results.push(serde_json::json!({"mutation": mutation, "differs": found}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("block_warp_mutants"));
        receipt.insert("mutants".into(), serde_json::json!(results));
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("block-warp checks reject every mutant");
}

/// Runs without a GPU: every block-warp mutant instance is in its cubin.
#[test]
fn block_warp_mutants_are_built() {
    let built = kernel_lab::cubin_entries(block_warp_mutant_shapes::CUBIN).unwrap();
    let fp4 = kernel_lab::cubin_entries(split_block_warps::FP4_CUBIN).unwrap();
    assert_eq!(
        block_warp_mutant_shapes::MUTATIONS,
        split_block_warps::MUTATIONS
    );
    for mutation in block_warp_mutant_shapes::MUTATIONS {
        for gate_up in [true, false] {
            let symbol = block_warp_mutant_shapes::symbol(mutation, gate_up);
            assert!(built.contains(&symbol.as_str()), "{symbol}");
            let symbol = split_block_warps::fp4_symbol(mutation, gate_up);
            assert!(fp4.contains(&symbol.as_str()), "{symbol}");
        }
    }
}

/// The phases in which a routed layer's block-warp `projections` give
/// intermediate or weighted-route bytes other than the column schedule's
/// (`columns`, at the route capacity), each launch from poisoned outputs
/// and save area, read back after a host synchronization and compared
/// whole: the projections alone over the routes in place; the routes, the
/// producer holding the activation at NaN, then the projections (PDL);
/// that sequence from one captured graph replayed twice. Preparation,
/// launch, capture, synchronization and readback errors propagate; only a
/// compared output difference counts. `inject_failure` returns an
/// operational error before the first launch, for the control that such an
/// error fails the test.
#[allow(clippy::too_many_arguments)]
fn layer_mutant_differences(
    session: &Session,
    layer: &Layer,
    projections: [&Projection; 2],
    fns: [&Function; 2],
    producer: &Producer,
    operand: &DeviceBuffer,
    fresh: &DeviceBuffer,
    columns: &[Vec<u8>],
    inject_failure: bool,
) -> LabResult<Vec<String>> {
    if inject_failure {
        return Err("injected operational failure before the first launch".into());
    }
    let stream = &session.stream;
    let capacity = layer.route.capacity;
    let outputs = [&layer.inter, &layer.routed];
    let mut differing = Vec::new();
    let mut compare = |phase: &str| -> LabResult<()> {
        stream.synchronize().map_err(err)?;
        for (name, (buffer, want)) in ["intermediate", "routed"]
            .iter()
            .zip(outputs.iter().zip(columns))
        {
            if &poison::read(session, buffer)? != want {
                differing.push(format!("{phase}: {name}"));
            }
        }
        Ok(())
    };
    poison::poison(session, &outputs)?;
    layer.launch_projections(stream, projections, fns, capacity, false)?;
    compare("separate")?;
    let sequence = |stream: &Stream| -> LabResult<()> {
        layer.routes.launch(
            stream,
            layer.route,
            &layer.route_fns,
            layer.rank as u32,
            true,
        )?;
        producer.launch(stream, operand)?;
        layer.launch_projections(stream, projections, fns, capacity, true)
    };
    let scratch = [&layer.inter, &layer.routed, fresh];
    poison::poison(session, &scratch)?;
    sequence(stream)?;
    compare("behind the producer")?;
    let graph = session.capture(sequence)?;
    for replay in 0..2 {
        poison::poison(session, &scratch)?;
        graph.launch(stream).map_err(err)?;
        compare(&format!("graph replay {replay}"))?;
    }
    Ok(differing)
}

/// The split's NVFP4 routed block warps' checks (`layer_mutant_differences`,
/// all 288 experts the rank's) find no difference from the column schedule
/// (`moe_chained_split_references`) on the unmodified lab instance
/// (`split_block_warps::fp4_projections`, the served decode projections)
/// and a difference for each mutant: no or a late dependency wait (behind
/// the producer) and a reversed chain fold. A mutant counts as
/// caught only on an observed output difference; operational errors fail
/// the test, which an injected failure before the first launch shows.
#[test]
fn nvfp4_block_warp_checks_reject_every_mutant() {
    gpu_test(4 << 30, |session| {
        let module = session.module(split_block_warps::FP4_CUBIN)?;
        let producer = Producer::new(session)?;
        let layer = Layer::pooled(
            session,
            &split::SHAPES,
            Weights::Nvfp4,
            false,
            0,
            991,
            &expert_pool(),
            Schedule::Columns,
        )?;
        let warps = split_block_warps::fp4_projections();
        let own = [&layer.gate_up, &layer.down]
            .map(|p| p.block_warps().expect("block warps take the projection"));
        if own[0].symbol != warps[0].symbol || own[1].symbol != warps[1].symbol {
            return Err("the mutants' projections are not the layer's".into());
        }
        let capacity = layer.route.capacity;
        layer.routes.set_live(session, capacity)?;
        let outputs = [&layer.inter, &layer.routed];
        let columns = poisoned_outputs(session, &outputs, || layer.launch(&session.stream, false))?;
        let fresh = session.allocate(layer.x.byte_len())?;
        let operand = Producer::operand(session, &layer.x, &fresh)?;
        let load = |mutation: &str| -> LabResult<[Function; 2]> {
            Ok([
                module
                    .function(&split_block_warps::fp4_symbol(mutation, true))
                    .map_err(err)?,
                module
                    .function(&split_block_warps::fp4_symbol(mutation, false))
                    .map_err(err)?,
            ])
        };
        let differences = |fns: &[Function; 2], inject: bool| {
            layer_mutant_differences(
                session,
                &layer,
                [&warps[0], &warps[1]],
                [&fns[0], &fns[1]],
                &producer,
                &operand,
                &fresh,
                &columns,
                inject,
            )
        };
        let control = load(split_block_warps::MUTATIONS[0])?;
        let control_differences = differences(&control, false)?;
        if !control_differences.is_empty() {
            return Err(format!(
                "the unmodified instance differs: {control_differences:?}"
            ));
        }
        if differences(&control, true).is_ok() {
            return Err("an injected operational failure did not propagate".into());
        }
        let mut results = Vec::new();
        for &mutation in &split_block_warps::MUTATIONS[1..] {
            let found = differences(&load(mutation)?, false)?;
            if found.is_empty() {
                return Err(format!("the {mutation} mutant passed"));
            }
            eprintln!("NVFP4 {mutation}: outputs differ in {found:?}");
            results.push(serde_json::json!({"mutation": mutation, "differs": found}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("nvfp4_block_warp_mutants"));
        receipt.insert("mutants".into(), serde_json::json!(results));
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("NVFP4 block-warp checks reject every mutant");
}

// ----------------------------------------------------- warp streams ----

/// Warp streams (`chained::Schedule::WarpStreams`) against the column
/// schedule, bit for bit, over the same weights, inputs and routes: the
/// split's FP8 routed layers (all 288 experts on rank 0; the gate/up in
/// warp streams against the column reference, the down in columns, as
/// served, since warp streams do not take its K 512) under the three
/// comparison pools (eight experts give every expert two full 16-row tiles
/// at 32 tokens, so CTAs run several items, cross expert boundaries and
/// multiply both 8-row fragments) and with routes spread one per expert
/// over 256 of the 288 experts, at every live count from 0 to 32. Each
/// output buffer is poisoned before each run and compared whole; warp
/// streams run planned at the capacity and at the live count (one item per
/// CTA over the plan's tile bound).
#[test]
fn warp_streams_match_the_column_schedule_bit_for_bit() {
    gpu_test(8 << 30, |session| {
        let mut failures = replay::Failures::default();
        let shapes = split::SHAPES;
        let local = shapes.local_experts as usize;
        let cases = comparison_cases(&shapes);
        for (seed, (pool_name, pool, rank, spread)) in cases.into_iter().enumerate() {
            let case = format!("FP8 layer, {pool_name}, rank {rank}");
            let result = (|| -> LabResult<()> {
                let capacity = shapes.routes()[0].capacity as usize;
                let layer = Layer::routed(
                    session,
                    &shapes,
                    Weights::Fp8Block128,
                    false,
                    rank,
                    1500 + seed as u64,
                    &pool,
                    spread.then(|| spread_logits(capacity, local)),
                    Schedule::Columns,
                )?;
                layer_matches_columns(session, &layer, Schedule::WarpStreams)
            })();
            failures.check(&case, result);
        }
        failures.finish()
    })
    .expect("warp streams match the column schedule bit for bit");
}

/// The split's warp-stream layers (the gate/up in warp streams, the down in
/// columns) against the oracle, with poisoned eager, relaunch, graph replay
/// and PDL-producer launches at every `poison::live_counts` count
/// (`run_layer`), on every rank (the split's one), under the oracle pool
/// and with eight experts.
#[test]
fn warp_stream_layers_match_the_oracle() {
    gpu_test(8 << 30, |session| {
        let mut failures = replay::Failures::default();
        let shapes = split::SHAPES;
        for (pool_name, pool) in comparison_pools().into_iter().take(2) {
            for rank in 0..route_ranks(shapes.routes()[0]) {
                let result = Layer::pooled(
                    session,
                    &shapes,
                    Weights::Fp8Block128,
                    false,
                    rank,
                    1600,
                    &pool,
                    Schedule::WarpStreams,
                )
                .and_then(|layer| run_layer(session, &layer, layer.route.capacity));
                failures.check(
                    &format!("warp-stream layer, {pool_name}, rank {rank}"),
                    result,
                );
            }
        }
        failures.finish()
    })
    .expect("warp-stream layers match the oracle");
}

/// The split's routed warp-stream layers behind their real producers, as
/// `block_warp_layers_pass_behind_their_routes_and_a_real_producer`: the
/// route kernels, the producer probe holding the input at NaN, gate/up (in
/// warp streams) and down, all PDL in one sequence, equal to the column
/// schedule's separate launches at every non-zero `poison::live_counts`
/// count; eager, then one captured graph replayed twice, from poisoned
/// outputs each time. FP8, rank 0, under all three comparison pools.
#[test]
fn warp_stream_layers_pass_behind_their_routes_and_a_real_producer() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let mut failures = replay::Failures::default();
        for (seed, (pool_name, pool)) in comparison_pools().into_iter().enumerate() {
            let result = (|| -> LabResult<()> {
                let layer = Layer::pooled(
                    session,
                    &split::SHAPES,
                    Weights::Fp8Block128,
                    false,
                    0,
                    1700 + seed as u64,
                    &pool,
                    Schedule::Columns,
                )?;
                let streams = scheduled(
                    [layer.gate_up.clone(), layer.down.clone()],
                    Schedule::WarpStreams,
                );
                let stream_fns = [
                    Entry::of(&streams[0]).load(session)?,
                    Entry::of(&streams[1]).load(session)?,
                ];
                let fresh = session.allocate(layer.x.byte_len())?;
                let operand = Producer::operand(session, &layer.x, &fresh)?;
                let outputs = [&layer.inter, &layer.routed];
                let capacity = layer.route.capacity;
                let sequence = |stream: &Stream| -> LabResult<()> {
                    layer.routes.launch(
                        stream,
                        layer.route,
                        &layer.route_fns,
                        layer.rank as u32,
                        true,
                    )?;
                    producer.launch(stream, &operand)?;
                    layer.launch_projections(
                        stream,
                        [&streams[0], &streams[1]],
                        [&stream_fns[0], &stream_fns[1]],
                        capacity,
                        true,
                    )
                };
                for live in poison::live_counts(capacity)
                    .into_iter()
                    .filter(|&live| live > 0)
                {
                    layer.routes.set_live(session, live)?;
                    let columns = poisoned_outputs(session, &outputs, || {
                        layer.launch(&session.stream, false)
                    })?;
                    let eager =
                        poisoned_outputs(session, &[&layer.inter, &layer.routed, &fresh], || {
                            sequence(&session.stream)
                        })?;
                    same_bytes(
                        &format!("{pool_name} live {live} eager"),
                        &columns,
                        &eager[..2],
                    )?;
                    let graph = session.capture(sequence)?;
                    for replay in 0..2 {
                        let replayed = poisoned_outputs(
                            session,
                            &[&layer.inter, &layer.routed, &fresh],
                            || graph.launch(&session.stream).map_err(err),
                        )?;
                        same_bytes(
                            &format!("{pool_name} live {live} graph replay {replay}"),
                            &columns,
                            &replayed[..2],
                        )?;
                    }
                }
                Ok(())
            })();
            failures.check(&format!("FP8 layer, {pool_name}"), result);
        }
        failures.finish()
    })
    .expect("warp-stream layers pass behind their routes and a real producer");
}

/// The warp streams' checks (`layer_mutant_differences`, at the route
/// capacity with eight of the split's experts: two full tiles per expert)
/// find no difference from the column schedule on the unmodified lab
/// instance (`split_warp_streams`, the split's served decode gate/up,
/// beside its served column down) and a difference for each mutant: no or
/// a late dependency wait (behind the producer), a reversed chain fold and
/// a never-multiplied second fragment. A mutant counts as caught only on an observed output
/// difference; operational errors fail the test, which an injected failure
/// before the first launch shows.
#[test]
fn warp_stream_checks_reject_every_mutant() {
    gpu_test(4 << 30, |session| {
        let module = session.module(split_warp_streams::MUTANTS_CUBIN)?;
        let producer = Producer::new(session)?;
        let [_, (_, pool), _] = comparison_pools();
        let layer = Layer::pooled(
            session,
            &split::SHAPES,
            Weights::Fp8Block128,
            false,
            0,
            1800,
            &pool,
            Schedule::Columns,
        )?;
        let streams = [split_warp_streams::projection(), layer.down.clone()];
        let own = scheduled(
            [layer.gate_up.clone(), layer.down.clone()],
            Schedule::WarpStreams,
        );
        if own[0].symbol != streams[0].symbol || own[1].symbol != streams[1].symbol {
            return Err("the mutants' projections are not the layer's".into());
        }
        let capacity = layer.route.capacity;
        layer.routes.set_live(session, capacity)?;
        let outputs = [&layer.inter, &layer.routed];
        let columns = poisoned_outputs(session, &outputs, || layer.launch(&session.stream, false))?;
        let fresh = session.allocate(layer.x.byte_len())?;
        let operand = Producer::operand(session, &layer.x, &fresh)?;
        let load = |mutation: &str| -> LabResult<[Function; 2]> {
            Ok([
                module
                    .function(&split_warp_streams::symbol(mutation))
                    .map_err(err)?,
                Entry::of(&layer.down).load(session)?,
            ])
        };
        let differences = |fns: &[Function; 2], inject: bool| {
            layer_mutant_differences(
                session,
                &layer,
                [&streams[0], &streams[1]],
                [&fns[0], &fns[1]],
                &producer,
                &operand,
                &fresh,
                &columns,
                inject,
            )
        };
        let control = load(split_warp_streams::MUTATIONS[0])?;
        let control_differences = differences(&control, false)?;
        if !control_differences.is_empty() {
            return Err(format!(
                "the unmodified instance differs: {control_differences:?}"
            ));
        }
        if differences(&control, true).is_ok() {
            return Err("an injected operational failure did not propagate".into());
        }
        let mut results = Vec::new();
        for &mutation in &split_warp_streams::MUTATIONS[1..] {
            let found = differences(&load(mutation)?, false)?;
            if found.is_empty() {
                return Err(format!("the {mutation} mutant passed"));
            }
            eprintln!("warp streams {mutation}: outputs differ in {found:?}");
            results.push(serde_json::json!({"mutation": mutation, "differs": found}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("warp_stream_mutants"));
        receipt.insert("mutants".into(), serde_json::json!(results));
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("warp-stream checks reject every mutant");
}

/// Runs without a GPU: every warp-stream form the split serves is a gate/up
/// in the layout of one item per CTA (`count` = 1 in
/// `chained_warp_streams.cuh`) at every live count, so no CTA runs the
/// later items `Mutation::stale_slot` mutates and the split family builds
/// no such mutant.
#[test]
fn split_warp_streams_run_one_item_per_cta() {
    let reinstate = "reinstate the stale_slot mutant in moe_chained_split_warp_streams";
    let streams = split::forms()
        .into_iter()
        .filter(|p| p.schedule == Schedule::WarpStreams)
        .collect::<Vec<_>>();
    assert!(!streams.is_empty());
    for p in &streams {
        assert!(
            p.gate_up() && !p.warp_stream_layout().persistent,
            "{} runs the persistent warp-stream layout: {reinstate}",
            p.symbol
        );
        for tokens in 1..=p.tokens_capacity {
            let plan = p.plan(tokens).unwrap();
            let tiles = p.live_padded_rows(tokens) / p.row_tile;
            assert_eq!(
                plan.grid,
                [p.columns() / chained::WARP_STREAMS_TILE_N, tiles, 1],
                "{} at {tokens} tokens gives a CTA more than one item: {reinstate}",
                p.symbol
            );
        }
    }
    assert!(!split_warp_streams::MUTATIONS.contains(&"stale_slot"));
}

/// Runs without a GPU: every warp-stream mutant instance is in its cubin,
/// its projection is the split's served gate/up in warp streams, and warp
/// streams refuse the split's decode down (one 512-K round), which has no
/// instance.
#[test]
fn warp_stream_mutants_are_built() {
    let built = kernel_lab::cubin_entries(split_warp_streams::MUTANTS_CUBIN).unwrap();
    for mutation in split_warp_streams::MUTATIONS {
        let symbol = split_warp_streams::symbol(mutation);
        assert!(built.contains(&symbol.as_str()), "{symbol}");
    }
    assert_eq!(built.len(), split_warp_streams::MUTATIONS.len());
    let gate_up = split_warp_streams::projection();
    assert!(gate_up.gate_up());
    assert_eq!(
        gate_up.symbol,
        references::projections()[0]
            .warp_streams()
            .expect("warp streams take the split's decode gate/up")
            .symbol
    );
    let down = projection(
        Weights::Fp8Block128,
        false,
        true,
        16,
        split::SHAPES.expert_intermediate,
    );
    assert!(down.warp_streams().is_err());
}

// ------------------------------------------------------- tile pairs ----

/// The prefill live counts the tile-pair comparisons run, up to `max`: both
/// sides of the 8-row fragment and the 16-, 32- and 64-row tiles, and chunk
/// sizes.
fn prefill_counts(max: u32) -> Vec<u32> {
    [
        0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 127, 129, 255, 511, 1000, 1023, 2047,
        2048, 4095, 4096, 8191, 8192,
    ]
    .into_iter()
    .filter(|&count| count <= max)
    .collect()
}

/// FP8 128x128 weights of `experts` experts `[rows, k]` from a seed, every
/// expert written: the comparisons read every local expert and compare
/// bits only, so no decoded copy is kept.
fn fp8_bank(
    session: &Session,
    experts: usize,
    rows: usize,
    k: usize,
    seed: u64,
) -> LabResult<Bank> {
    let mut rng = Rng(seed);
    let mut codes = vec![0_u8; experts * rows * k];
    for chunk in codes.chunks_mut(8) {
        for (code, byte) in chunk.iter_mut().zip(rng.next().to_le_bytes()) {
            // Sign, mantissa and an exponent field in 0..=8: never NaN.
            *code = (byte & 0x87) | ((((byte >> 3) & 0xf) % 9) << 3);
        }
    }
    let scales = (0..experts * (rows / 128) * (k / 128))
        .map(|_| (0.25 * 2_f64.powi(rng.below(3) as i32 - 1) * (1.0 + rng.unit())) as f32)
        .collect::<Vec<_>>();
    Ok(Bank {
        codes: upload(session, &codes)?,
        block_scales: Some(upload(session, &bytes_of(&scales, f32::to_ne_bytes))?),
        local_scales: None,
        globals: None,
        decoded: BTreeMap::new(),
    })
}

/// One rank's routed FP8 prefill projections over route metadata built on
/// the host (`local_route_metadata` at the projections' route padding),
/// with weights for every local expert and buffers for `tokens` tokens: the
/// comparisons' layer, for routes given as expert IDs.
struct RoutedPrefill {
    tokens: u32,
    local: usize,
    gate_up: Projection,
    down: Projection,
    fns: [Function; 2],
    x: DeviceBuffer,
    inter: DeviceBuffer,
    routed: DeviceBuffer,
    route_weights: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    live: DeviceBuffer,
    gate_up_bank: Bank,
    down_bank: Bank,
}

impl RoutedPrefill {
    fn new(
        session: &Session,
        references: [&Projection; 2],
        tokens: u32,
        seed: u64,
    ) -> LabResult<Self> {
        let [gate_up, down] = references.map(Clone::clone);
        let mut rng = Rng(seed);
        let routes = tokens as usize * TOP_K;
        let x = (0..tokens as usize * HIDDEN)
            .flat_map(|_| bf16_value(&mut rng, -4..=0).to_ne_bytes())
            .collect::<Vec<_>>();
        let weights = (0..routes)
            .map(|_| (0.5 + 0.5 * rng.unit()) as f32)
            .collect::<Vec<_>>();
        let pad = gate_up.row_tile as usize;
        let (local, inter) = (grouped_experts(&gate_up), gate_up.columns() as usize);
        Ok(Self {
            fns: [
                Entry::of(&gate_up).load(session)?,
                Entry::of(&down).load(session)?,
            ],
            x: upload(session, &x)?,
            inter: zeroed(session, routes * inter * 2)?,
            routed: zeroed(session, routes * HIDDEN * 4)?,
            route_weights: upload(session, &bytes_of(&weights, f32::to_ne_bytes))?,
            offsets: zeroed(session, (local + 1) * 4)?,
            sorted: zeroed(session, (routes + local * (pad - 1)) * 4)?,
            padded: zeroed(session, 4)?,
            live: zeroed(session, 4)?,
            gate_up_bank: fp8_bank(session, local, 2 * inter, HIDDEN, seed ^ 0x51)?,
            down_bank: fp8_bank(session, local, HIDDEN, inter, seed ^ 0x52)?,
            gate_up,
            down,
            tokens,
            local,
        })
    }

    /// Writes the route metadata of `ids` (global experts, `[live,
    /// top_k]`) on `rank`; returns the padded row count.
    fn set_routes(&self, session: &Session, ids: &[i32], rank: usize) -> LabResult<i32> {
        let ids = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        let metadata = local_route_metadata(
            &ids,
            TOP_K,
            self.local,
            rank,
            self.gate_up.row_tile as usize,
        )
        .map_err(err)?;
        poison::write(
            session,
            &self.offsets,
            0,
            &bytes_of(&metadata.offsets, i32::to_ne_bytes),
        )?;
        if !metadata.sorted_rows.is_empty() {
            poison::write(
                session,
                &self.sorted,
                0,
                &bytes_of(&metadata.sorted_rows, i32::to_ne_bytes),
            )?;
        }
        poison::write(
            session,
            &self.padded,
            0,
            &metadata.num_padded_rows.to_ne_bytes(),
        )?;
        Ok(metadata.num_padded_rows)
    }

    /// Gate/up into the intermediate, then down into the weighted routes,
    /// planned at `tokens`.
    fn launch(
        &self,
        stream: &Stream,
        projections: [&Projection; 2],
        fns: [&Function; 2],
        tokens: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        for (p, f, bank, input, weights, output) in [
            (
                projections[0],
                fns[0],
                &self.gate_up_bank,
                &self.x,
                KernelArgument::NullBuffer,
                &self.inter,
            ),
            (
                projections[1],
                fns[1],
                &self.down_bank,
                &self.inter,
                b(&self.route_weights),
                &self.routed,
            ),
        ] {
            let [codes, block, local, globals] = bank.arguments();
            let args = [
                b(input),
                codes,
                block,
                local,
                globals,
                b(&self.offsets),
                b(&self.sorted),
                b(&self.padded),
                b(&self.live),
                weights,
                b(output),
            ];
            launch_with(
                stream,
                f,
                p.plan(tokens).map_err(|e| e.0.to_string())?,
                pdl,
                &args,
            )?;
        }
        Ok(())
    }

    fn launch_columns(&self, stream: &Stream, tokens: u32) -> LabResult<()> {
        self.launch(
            stream,
            [&self.gate_up, &self.down],
            [&self.fns[0], &self.fns[1]],
            tokens,
            false,
        )
    }
}

/// `tokens` tokens' routes, each top_k distinct experts of `pool` in a
/// random order.
fn pool_ids(tokens: usize, pool: &[usize], rng: &mut Rng) -> Vec<i32> {
    let mut ids = Vec::with_capacity(tokens * TOP_K);
    for _ in 0..tokens {
        let mut left = pool.to_vec();
        for _ in 0..TOP_K {
            ids.push(left.swap_remove(rng.below(left.len())) as i32);
        }
    }
    ids
}

/// `spread_logits`' routes: token t to rank 0's `local` experts `(8 t + j)
/// mod local`.
fn spread_ids(tokens: usize, local: usize) -> Vec<i32> {
    (0..tokens * TOP_K)
        .map(|route| (route % (TOP_K * local) % local) as i32)
        .collect()
}

/// Skewed routes: top_k distinct experts per token drawn from a Zipf law
/// of exponent 1.2 over a fixed permutation of the 288 experts, so a few
/// experts take most tokens and many take a handful.
fn skewed_ids(tokens: usize, rng: &mut Rng) -> Vec<i32> {
    let mut order = (0..EXPERTS).collect::<Vec<_>>();
    for i in (1..EXPERTS).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut cumulative = Vec::with_capacity(EXPERTS);
    let mut total = 0.0;
    for rank in 0..EXPERTS {
        total += 1.0 / ((rank + 1) as f64).powf(1.2);
        cumulative.push(total);
    }
    let mut ids = Vec::with_capacity(tokens * TOP_K);
    for _ in 0..tokens {
        let mut chosen = Vec::with_capacity(TOP_K);
        while chosen.len() < TOP_K {
            let draw = rng.unit() * total;
            let at = cumulative.partition_point(|&c| c < draw).min(EXPERTS - 1);
            if !chosen.contains(&order[at]) {
                chosen.push(order[at]);
            }
        }
        ids.extend(chosen.into_iter().map(|e| e as i32));
    }
    ids
}

/// Expert row counts around every row-pair edge: one, both sides of an
/// 8-row fragment, and both sides of one to five 64-row tiles (odd and even
/// tile counts, so hot experts with a lone last tile), then 447 to 2,048
/// rows (one expert on every token).
const STAIRCASE_ROWS: [usize; 24] = [
    1, 7, 8, 9, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 447, 449,
    1023, 1757, 2048,
];

/// Routes giving rank `rank`'s local experts (of `local` per rank) 0, 3,
/// 6, ... and the last exactly `STAIRCASE_ROWS` (scaled by `tokens / 2048`,
/// at most `tokens`) at `tokens` tokens: each token takes the experts with
/// the most rows still owed (at most eight, distinct), then fills its other
/// routes with distinct experts in turn: the other ranks', or with one rank
/// of every expert, the rank's other experts (each a few rows).
fn staircase_ids(tokens: usize, rank: usize, local: usize) -> Vec<i32> {
    let scale = tokens.div_ceil(2048);
    let mut owed = STAIRCASE_ROWS
        .iter()
        .enumerate()
        .map(|(i, &rows)| {
            let expert = if i + 1 == STAIRCASE_ROWS.len() {
                local - 1
            } else {
                3 * i
            };
            (rank * local + expert, (rows * scale).min(tokens))
        })
        .collect::<Vec<_>>();
    let stairs = owed.iter().map(|&(expert, _)| expert).collect::<Vec<_>>();
    let fillers = (0..EXPERTS)
        .filter(|e| match local < EXPERTS {
            true => e / local != rank,
            false => !stairs.contains(e),
        })
        .collect::<Vec<_>>();
    let mut next_filler = 0;
    let mut ids = Vec::with_capacity(tokens * TOP_K);
    for _ in 0..tokens {
        owed.sort_by_key(|&(expert, rows)| (std::cmp::Reverse(rows), expert));
        let mut chosen = Vec::with_capacity(TOP_K);
        for (expert, rows) in owed.iter_mut().take(TOP_K) {
            if *rows > 0 {
                *rows -= 1;
                chosen.push(*expert);
            }
        }
        while chosen.len() < TOP_K {
            chosen.push(fillers[next_filler % fillers.len()]);
            next_filler += 1;
        }
        ids.extend(chosen.into_iter().map(|e| e as i32));
    }
    assert!(
        owed.iter().all(|&(_, rows)| rows == 0),
        "staircase rows left over: {owed:?}"
    );
    ids
}

/// Router logits whose top_k selection is exactly `ids` (`[tokens, top_k]`
/// global experts, distinct per token): token t's k-th expert sits at
/// `3 - k / 64`, every other expert at -8.
fn logits_of(ids: &[i32]) -> Vec<f32> {
    let tokens = ids.len() / TOP_K;
    let mut logits = vec![-8.0_f32; tokens * EXPERTS];
    for (route, &id) in ids.iter().enumerate() {
        logits[route / TOP_K * EXPERTS + id as usize] = 3.0 - (route % TOP_K) as f32 / 64.0;
    }
    logits
}

/// The rank with the most local routes among `ids` (the lowest on ties)
/// when each rank holds `local` experts: the critical path of a step. With
/// one rank of every expert, rank 0.
fn busiest_rank(ids: &[i32], local: usize) -> usize {
    let mut counts = vec![0_usize; EXPERTS / local];
    for &id in ids {
        counts[id as usize / local] += 1;
    }
    (0..counts.len())
        .max_by_key(|&r| (counts[r], std::cmp::Reverse(r)))
        .unwrap_or(0)
}

/// The distinct experts of `ids`, ascending.
fn experts_of(ids: &[i32]) -> Vec<usize> {
    let mut experts = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
    experts.sort_unstable();
    experts.dedup();
    experts
}

/// The captured served routes (`split_tile_pairs::captured_layers`); a GPU
/// job sets the capture, CPU-only runs leave it unset. Unset, the captured
/// cases are skipped, which `CHOPINFER_KERNEL_REQUIRE=1` refuses.
fn captured() -> LabResult<Vec<split_tile_pairs::CapturedLayer>> {
    match split_tile_pairs::captured_layers()? {
        Some(layers) => Ok(layers),
        None => {
            kernel_lab::skip(&format!(
                "the captured routes: {} is unset",
                split_tile_pairs::ROUTE_CAPTURE_ENV
            ))?;
            Ok(Vec::new())
        }
    }
}

/// The variables that narrow the tile-pair checks to some thresholds or
/// case families.
const TILE_PAIR_FILTERS: [&str; 2] = [
    "CHOPINFER_MOE_TILE_PAIR_SHAPES",
    "CHOPINFER_MOE_TILE_PAIR_CASES",
];

/// Refuses [`TILE_PAIR_FILTERS`] under `CHOPINFER_KERNEL_REQUIRE=1`: strict
/// evidence runs every threshold and every case family.
fn tile_pair_filters_allowed() -> LabResult<()> {
    if kernel_lab::require_all()? {
        for name in TILE_PAIR_FILTERS {
            if std::env::var_os(name).is_some() {
                return Err(format!(
                    "{name} narrows the tile-pair checks; CHOPINFER_KERNEL_REQUIRE=1 runs them all"
                ));
            }
        }
    }
    Ok(())
}

/// Whether `projection`'s threshold is selected: unless
/// `CHOPINFER_MOE_TILE_PAIR_SHAPES` is set, every one; else those whose
/// `tp{min_pair_tiles}` it lists (comma separated; pairs are selected by
/// their gate/up), so a GPU job can sweep some thresholds only.
fn tile_pair_selected(projection: &Projection) -> bool {
    let Ok(list) = std::env::var("CHOPINFER_MOE_TILE_PAIR_SHAPES") else {
        return true;
    };
    let tag = format!("tp{}", split_tile_pairs::min_pair_tiles(projection));
    list.split(',').any(|item| item.trim() == tag)
}

/// Whether the bitwise sweep runs the case family `family` (`synthetic`,
/// `captured`, `large` or `captured-large`): unless
/// `CHOPINFER_MOE_TILE_PAIR_CASES` is set, every family; else those it lists
/// (comma separated).
fn tile_pair_family_selected(family: &str) -> bool {
    std::env::var("CHOPINFER_MOE_TILE_PAIR_CASES")
        .map(|list| list.split(',').any(|item| item.trim() == family))
        .unwrap_or(true)
}

/// Each selected tile-pair candidate pair (`split_tile_pairs`,
/// `tile_pair_selected`), loaded.
fn tile_pair_candidates(session: &Session) -> LabResult<Vec<([Projection; 2], [Function; 2])>> {
    tile_pair_filters_allowed()?;
    split_tile_pairs::candidate_pairs()
        .into_iter()
        .filter(|pair| tile_pair_selected(&pair[0]))
        .map(|pair| {
            let fns = [
                Entry::of(&pair[0]).load(session)?,
                Entry::of(&pair[1]).load(session)?,
            ];
            Ok((pair, fns))
        })
        .collect()
}

/// Every candidate of `pairs` against `layer`'s column schedule, bit for
/// bit, at each of `counts` live tokens of `ids` on `rank`: both outputs
/// poisoned before each run and compared whole; candidates planned at the
/// layer's token count and at the live count (the tightest grid).
fn compare_routed(
    session: &Session,
    layer: &RoutedPrefill,
    pairs: &[([Projection; 2], [Function; 2])],
    label: &str,
    ids: &[i32],
    rank: usize,
    counts: &[u32],
) -> LabResult<()> {
    let outputs = [&layer.inter, &layer.routed];
    for &live in counts {
        let padded = layer.set_routes(session, &ids[..live as usize * TOP_K], rank)?;
        let columns = poisoned_outputs(session, &outputs, || {
            layer.launch_columns(&session.stream, layer.tokens)
        })?;
        for (pair, fns) in pairs {
            for tokens in [layer.tokens, live.max(1)] {
                let got = poisoned_outputs(session, &outputs, || {
                    layer.launch(
                        &session.stream,
                        [&pair[0], &pair[1]],
                        [&fns[0], &fns[1]],
                        tokens,
                        false,
                    )
                })?;
                same_bytes(
                    &format!(
                        "{label} live {live} ({padded} padded rows) {} planned at {tokens}",
                        pair[0].symbol
                    ),
                    &columns,
                    &got,
                )?;
            }
        }
    }
    Ok(())
}

/// Tile pairs (`chained::Schedule::TilePairs`, every threshold of
/// `split_tile_pairs`) against the column schedule they would replace at
/// the split's prefill, bit for bit, over the same weights, inputs and
/// routes (FP8, all 288 experts on rank 0 written), each output poisoned
/// before each run and compared whole, planned at the buffers' token count
/// and at the live count:
///
/// - `synthetic`, 2,048-token buffers at every `prefill_counts` count: the
///   three route pools on every rank (the split's one), routes spread one
///   per expert over rank 0 (crowded), Zipf-skewed routes, and the
///   staircase (`STAIRCASE_ROWS`: hot experts with odd and even tile
///   counts, every fragment and tile edge, the last expert of the range
///   among them) on rank 0;
/// - `captured`: every captured layer's first 2,048 tokens on its busiest
///   rank, and every seventh layer's on every rank;
/// - `large`, 8,192-token buffers: eight hot experts on rank 0 (every token
///   on each), skewed routes and the staircase scaled by four;
/// - `captured-large`: captured 4,096-token layers, 8,192 tokens over a
///   layer and the next, and a layer's capture repeated to 8,192 tokens.
///
/// `CHOPINFER_MOE_TILE_PAIR_SHAPES` and `CHOPINFER_MOE_TILE_PAIR_CASES`
/// narrow a run to some thresholds (`tile_pair_selected`) and case families
/// (`tile_pair_family_selected`); `CHOPINFER_KERNEL_REQUIRE=1` refuses both.
/// A run that selects no family fails, as does a selected family with no
/// case under `CHOPINFER_KERNEL_REQUIRE=1` (without it, a notice).
#[test]
fn tile_pairs_match_the_column_schedule_bit_for_bit() {
    // Before any GPU work, so a strict run with a filter fails at once.
    tile_pair_filters_allowed().expect("tile-pair filters");
    gpu_test(24 << 30, |session| {
        let mut failures = replay::Failures::default();
        let references = split_tile_pairs::references();
        let local = grouped_experts(&references[0]);
        let ranks = EXPERTS / local;
        let layers = captured()?;
        let mut rng = Rng(1301);
        let routed = [&references[0], &references[1]];
        let pairs = tile_pair_candidates(session)?;
        if pairs.is_empty() {
            return Err("CHOPINFER_MOE_TILE_PAIR_SHAPES selects no threshold".into());
        }
        let started = std::time::Instant::now();
        let progress = |case: &str| {
            eprintln!("{:7.1} s: {case}", started.elapsed().as_secs_f64());
        };
        for (pair, _) in &pairs {
            progress(&format!(
                "candidates {} and {}",
                pair[0].symbol, pair[1].symbol
            ));
        }
        let families = ["synthetic", "captured", "large", "captured-large"];
        if !families.iter().any(|f| tile_pair_family_selected(f)) {
            return Err("CHOPINFER_MOE_TILE_PAIR_CASES selects no case family".into());
        }
        let mut family_cases = BTreeMap::new();
        let mut run = |layer: &RoutedPrefill, cases: Vec<(String, Vec<i32>, usize, Vec<u32>)>| {
            for (name, ids, rank, counts) in &cases {
                progress(name);
                let result = compare_routed(session, layer, &pairs, name, ids, *rank, counts);
                failures.check(&format!("FP8 layer, {name}"), result);
            }
        };
        if tile_pair_family_selected("synthetic") || tile_pair_family_selected("captured") {
            let layer = RoutedPrefill::new(session, routed, 2048, 1302)?;
            let all = prefill_counts(2048);
            let mut cases = Vec::new();
            if tile_pair_family_selected("synthetic") {
                let before = cases.len();
                for (name, pool) in comparison_pools() {
                    let ids = pool_ids(2048, &pool, &mut rng);
                    for rank in 0..ranks {
                        cases.push((
                            format!("{name}, rank {rank}"),
                            ids.clone(),
                            rank,
                            all.clone(),
                        ));
                    }
                }
                cases.push((
                    "spread over rank 0".into(),
                    spread_ids(2048, local),
                    0,
                    all.clone(),
                ));
                let skewed = skewed_ids(2048, &mut rng);
                let rank = busiest_rank(&skewed, local);
                cases.push((format!("skewed, rank {rank}"), skewed, rank, all.clone()));
                let mut stair_ranks = vec![0, ranks - 1];
                stair_ranks.dedup();
                for rank in stair_ranks {
                    cases.push((
                        format!("staircase, rank {rank}"),
                        staircase_ids(2048, rank, local),
                        rank,
                        all.clone(),
                    ));
                }
                family_cases.insert("synthetic", cases.len() - before);
            }
            if tile_pair_family_selected("captured") {
                let before = cases.len();
                for (i, captured) in layers.iter().enumerate() {
                    let ids = split_tile_pairs::adjacent_rows(&layers, i, 2048);
                    let busiest = busiest_rank(&ids, local);
                    let on = if i % 7 == 0 {
                        (0..ranks).collect()
                    } else {
                        vec![busiest]
                    };
                    for rank in on {
                        cases.push((
                            format!("captured {} 2048, rank {rank}", captured.name),
                            ids.clone(),
                            rank,
                            vec![1, 64, 1000, 2047, 2048],
                        ));
                    }
                }
                family_cases.insert("captured", cases.len() - before);
            }
            run(&layer, cases);
        }
        if tile_pair_family_selected("large") || tile_pair_family_selected("captured-large") {
            let layer = RoutedPrefill::new(session, routed, 8192, 1303)?;
            let large = vec![4095, 4096, 8191, 8192];
            let mut cases = Vec::new();
            if tile_pair_family_selected("large") {
                let before = cases.len();
                let hot = (0..8).map(|l| l * 9).collect::<Vec<_>>();
                cases.push((
                    "eight hot experts on rank 0".to_string(),
                    pool_ids(8192, &hot, &mut rng),
                    0,
                    large.clone(),
                ));
                let skewed = skewed_ids(8192, &mut rng);
                let rank = busiest_rank(&skewed, local);
                cases.push((format!("skewed, rank {rank}"), skewed, rank, large.clone()));
                cases.push((
                    "staircase scaled by four, rank 0".into(),
                    staircase_ids(8192, 0, local),
                    0,
                    vec![1000, 4096, 8192],
                ));
                family_cases.insert("large", cases.len() - before);
            }
            if tile_pair_family_selected("captured-large") {
                let before = cases.len();
                for i in (0..layers.len()).step_by(14) {
                    let ids = split_tile_pairs::adjacent_rows(&layers, i, 4096);
                    let rank = busiest_rank(&ids, local);
                    cases.push((
                        format!("captured {} 4096, rank {rank}", layers[i].name),
                        ids,
                        rank,
                        vec![4096],
                    ));
                }
                for i in (0..layers.len()).step_by(21) {
                    let ids = split_tile_pairs::adjacent_rows(&layers, i, 8192);
                    let rank = busiest_rank(&ids, local);
                    cases.push((
                        format!("captured {} and the next 8192, rank {rank}", layers[i].name),
                        ids,
                        rank,
                        vec![8191, 8192],
                    ));
                }
                for i in (7..layers.len()).step_by(14) {
                    let ids = split_tile_pairs::repeated_rows(&layers, i, 8192);
                    let rank = busiest_rank(&ids, local);
                    cases.push((
                        format!("captured {} repeated to 8192, rank {rank}", layers[i].name),
                        ids,
                        rank,
                        vec![8192],
                    ));
                }
                family_cases.insert("captured-large", cases.len() - before);
            }
            run(&layer, cases);
        }
        let empty = family_cases
            .iter()
            .filter(|(_, n)| **n == 0)
            .map(|(f, _)| *f)
            .collect::<Vec<_>>();
        eprintln!("tile-pair case families: {family_cases:?}");
        if !empty.is_empty() {
            let message = format!("case families with no case: {empty:?}");
            if kernel_lab::require_all()? {
                failures.check("case families", Err(message));
            } else {
                eprintln!("{message}");
            }
        }
        failures.finish()
    })
    .expect("tile pairs match the column schedule bit for bit");
}

/// A split prefill layer (FP8, rank `rank`, the route kernels in place)
/// whose routes are the staircase of `staircase_ids` at the prefill
/// capacity.
fn staircase_layer(session: &Session, rank: usize, seed: u64) -> LabResult<Layer> {
    let shapes = split::SHAPES;
    let capacity = shapes.routes()[1].capacity as usize;
    let ids = staircase_ids(capacity, rank, shapes.local_experts as usize);
    Layer::routed(
        session,
        &shapes,
        Weights::Fp8Block128,
        true,
        rank,
        seed,
        &experts_of(&ids),
        Some(logits_of(&ids)),
        Schedule::Columns,
    )
}

/// Tile-pair layers against the oracle, with poisoned eager, relaunch,
/// graph replay and PDL-producer launches and their write maps at every
/// `poison::live_counts` count of the prefill capacity (`run_layer`): the
/// split's FP8 layers, every threshold, the oracle pool (16 experts, about
/// 64 tiles each at 8,192 tokens) on every rank (the split's one).
#[test]
fn tile_pair_layers_match_the_oracle() {
    gpu_test(16 << 30, |session| {
        let mut failures = replay::Failures::default();
        let shapes = split::SHAPES;
        for min_pair_tiles in split_tile_pairs::MIN_PAIR_TILES {
            for rank in 0..route_ranks(shapes.routes()[1]) {
                let result = Layer::pooled(
                    session,
                    &shapes,
                    Weights::Fp8Block128,
                    true,
                    rank,
                    1310,
                    &expert_pool(),
                    Schedule::TilePairs { min_pair_tiles },
                )
                .and_then(|layer| run_layer(session, &layer, PREFILL_ORACLE_TOKENS));
                failures.check(
                    &format!("FP8 tile-pair layer tp{min_pair_tiles} rank {rank}"),
                    result,
                );
            }
        }
        failures.finish()
    })
    .expect("tile-pair layers match the oracle");
}

/// Routed tile-pair layers behind their real producers, as
/// `block_warp_layers_pass_behind_their_routes_and_a_real_producer` at the
/// prefill route: the route kernels and the producer probe holding the
/// input at NaN, all PDL in one sequence (routes, probe, gate/up, down),
/// equal to the column schedule's separate launches at every non-zero
/// `poison::live_counts` count of the prefill capacity; eager, then one
/// captured graph replayed twice, from poisoned outputs each time. The
/// split's FP8 layers, every selected threshold, the oracle pool on rank 0
/// and the staircase on rank 0.
#[test]
fn tile_pair_layers_pass_behind_their_routes_and_a_real_producer() {
    // Before any GPU work, so a strict run with a filter fails at once.
    tile_pair_filters_allowed().expect("tile-pair filters");
    gpu_test(16 << 30, |session| {
        let producer = Producer::new(session)?;
        let mut failures = replay::Failures::default();
        let layers = [
            (
                "oracle pool",
                Layer::pooled(
                    session,
                    &split::SHAPES,
                    Weights::Fp8Block128,
                    true,
                    0,
                    1320,
                    &expert_pool(),
                    Schedule::Columns,
                )?,
            ),
            ("staircase", staircase_layer(session, 0, 1321)?),
        ];
        let candidates = tile_pair_candidates(session)?;
        for (name, layer) in &layers {
            let fresh = session.allocate(layer.x.byte_len())?;
            let operand = Producer::operand(session, &layer.x, &fresh)?;
            let outputs = [&layer.inter, &layer.routed];
            let capacity = layer.route.capacity;
            let mut columns = BTreeMap::new();
            for live in poison::live_counts(capacity)
                .into_iter()
                .filter(|&live| live > 0)
            {
                layer.routes.set_live(session, live)?;
                columns.insert(
                    live,
                    poisoned_outputs(session, &outputs, || layer.launch(&session.stream, false))?,
                );
            }
            for (pair, fns) in &candidates {
                let result = (|| -> LabResult<()> {
                    let sequence = |stream: &Stream| -> LabResult<()> {
                        layer.routes.launch(
                            stream,
                            layer.route,
                            &layer.route_fns,
                            layer.rank as u32,
                            true,
                        )?;
                        producer.launch(stream, &operand)?;
                        layer.launch_projections(
                            stream,
                            [&pair[0], &pair[1]],
                            [&fns[0], &fns[1]],
                            capacity,
                            true,
                        )
                    };
                    for (&live, columns) in &columns {
                        layer.routes.set_live(session, live)?;
                        let eager = poisoned_outputs(
                            session,
                            &[&layer.inter, &layer.routed, &fresh],
                            || sequence(&session.stream),
                        )?;
                        same_bytes(&format!("live {live} eager"), columns, &eager[..2])?;
                        let graph = session.capture(sequence)?;
                        for replay in 0..2 {
                            let replayed = poisoned_outputs(
                                session,
                                &[&layer.inter, &layer.routed, &fresh],
                                || graph.launch(&session.stream).map_err(err),
                            )?;
                            same_bytes(
                                &format!("live {live} graph replay {replay}"),
                                columns,
                                &replayed[..2],
                            )?;
                        }
                    }
                    Ok(())
                })();
                failures.check(&format!("FP8 layer, {name}, {}", pair[0].symbol), result);
            }
        }
        failures.finish()
    })
    .expect("tile-pair layers pass behind their routes and a real producer");
}

/// The phases in which a routed layer's tile-pair outputs (`projections`)
/// differ from `expected` (the column schedule's intermediate and weighted
/// routes at the layer's live count), each run from poisoned outputs and
/// save area, read back after a host synchronization and compared whole:
/// the routes then the projections, eager; the routes, the producer probe
/// (holding the input at NaN) and the projections, all PDL, eager; then that
/// sequence from one captured graph replayed twice. Launch, capture,
/// synchronization and readback errors propagate; only a compared output
/// difference counts. `inject_failure` returns an operational error before
/// the first launch, for the control that such an error fails the test.
#[allow(clippy::too_many_arguments)]
fn routed_mutant_differences(
    session: &Session,
    layer: &Layer,
    projections: [&Projection; 2],
    fns: [&Function; 2],
    producer: &Producer,
    (operand, fresh): (&DeviceBuffer, &DeviceBuffer),
    expected: &[Vec<u8>],
    inject_failure: bool,
) -> LabResult<Vec<String>> {
    if inject_failure {
        return Err("injected operational failure before the first launch".into());
    }
    let stream = &session.stream;
    let capacity = layer.route.capacity;
    let outputs = [&layer.inter, &layer.routed, fresh];
    let mut differing = Vec::new();
    let mut compare = |phase: &str| -> LabResult<()> {
        stream.synchronize().map_err(err)?;
        for (name, (buffer, want)) in ["intermediate", "weighted routes"]
            .iter()
            .zip([&layer.inter, &layer.routed].into_iter().zip(expected))
        {
            if &poison::read(session, buffer)? != want {
                differing.push(format!("{phase}: {name}"));
            }
        }
        Ok(())
    };
    poison::poison(session, &outputs)?;
    layer.routes.launch(
        stream,
        layer.route,
        &layer.route_fns,
        layer.rank as u32,
        false,
    )?;
    layer.launch_projections(stream, projections, fns, capacity, false)?;
    compare("separate")?;
    let sequence = |stream: &Stream| -> LabResult<()> {
        layer.routes.launch(
            stream,
            layer.route,
            &layer.route_fns,
            layer.rank as u32,
            true,
        )?;
        producer.launch(stream, operand)?;
        layer.launch_projections(stream, projections, fns, capacity, true)
    };
    poison::poison(session, &outputs)?;
    sequence(stream)?;
    compare("behind the producer")?;
    let graph = session.capture(sequence)?;
    for replay in 0..2 {
        poison::poison(session, &outputs)?;
        graph.launch(stream).map_err(err)?;
        compare(&format!("graph replay {replay}"))?;
    }
    Ok(differing)
}

/// The unmodified lab instance of the tile-pair mutants' cubin
/// (`split_tile_pairs::mutant_projections`, threshold 2) shows no
/// difference from the column schedule in any phase of
/// `routed_mutant_differences`, on the staircase layer at the prefill
/// capacity, and each mutant (`split_tile_pairs::MUTATIONS`) differs in
/// some phase: no or a late dependency wait (behind the producer), a hot
/// expert's lone last tile written in one column tile only, and chains
/// folded every 256 K (every phase). A mutant counts as caught only on an
/// observed output difference; operational errors fail the test, which an
/// injected failure before the first launch shows.
#[test]
fn tile_pair_checks_reject_every_mutant() {
    gpu_test(16 << 30, |session| {
        let module = session.module(split_tile_pairs::MUTANTS_CUBIN)?;
        let producer = Producer::new(session)?;
        let projections = split_tile_pairs::mutant_projections();
        let layer = staircase_layer(session, 0, 1390)?;
        let fresh = session.allocate(layer.x.byte_len())?;
        let operand = Producer::operand(session, &layer.x, &fresh)?;
        layer.routes.set_live(session, layer.route.capacity)?;
        let expected = poisoned_outputs(session, &[&layer.inter, &layer.routed], || {
            layer.launch(&session.stream, false)
        })?;
        let load = |mutation: &str| -> LabResult<[Function; 2]> {
            Ok([
                module
                    .function(&split_tile_pairs::mutant_symbol(mutation, true))
                    .map_err(err)?,
                module
                    .function(&split_tile_pairs::mutant_symbol(mutation, false))
                    .map_err(err)?,
            ])
        };
        let [p0, p1] = &projections;
        let differences = |fns: &[Function; 2], inject: bool| {
            routed_mutant_differences(
                session,
                &layer,
                [p0, p1],
                [&fns[0], &fns[1]],
                &producer,
                (&operand, &fresh),
                &expected,
                inject,
            )
        };
        let control = load(split_tile_pairs::MUTATIONS[0])?;
        let control_differences = differences(&control, false)?;
        if !control_differences.is_empty() {
            return Err(format!(
                "the unmodified instance differs: {control_differences:?}"
            ));
        }
        if differences(&control, true).is_ok() {
            return Err("an injected operational failure did not propagate".into());
        }
        let mut results = Vec::new();
        for &mutation in &split_tile_pairs::MUTATIONS[1..] {
            let found = differences(&load(mutation)?, false)?;
            if found.is_empty() {
                return Err(format!("the {mutation} mutant passed"));
            }
            eprintln!("{mutation}: outputs differ in {found:?}");
            results.push(serde_json::json!({"mutation": mutation, "differs": found}));
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("tile_pair_mutants"));
        receipt.insert("mutants".into(), serde_json::json!(results));
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("tile-pair checks reject every mutant");
}

/// Runs without a GPU: every tile-pair mutant instance is in its cubin,
/// each candidate keeps its reference's contract, row tile and plan rows,
/// and the staircase routes give their experts exactly their rows.
#[test]
fn tile_pair_lab_instances_are_built() {
    let built = kernel_lab::cubin_entries(split_tile_pairs::MUTANTS_CUBIN).unwrap();
    for mutation in split_tile_pairs::MUTATIONS {
        for gate_up in [true, false] {
            let symbol = split_tile_pairs::mutant_symbol(mutation, gate_up);
            assert!(built.contains(&symbol.as_str()), "{symbol}");
        }
    }
    let references = split_tile_pairs::references();
    for candidate in split_tile_pairs::candidate_pairs() {
        for (reference, projection) in references.iter().zip(&candidate) {
            assert_eq!(projection.query, reference.query);
            assert_eq!(projection.row_tile, reference.row_tile);
            let (plan, columns) = (
                projection.plan(2048).unwrap(),
                reference.plan(2048).unwrap(),
            );
            assert_eq!(plan.grid[0], columns.grid[0]);
            assert_eq!(2 * plan.grid[1], columns.grid[1]);
            let entry = Entry::of(projection);
            assert!(
                kernel_lab::cubin_entries(&entry.cubin)
                    .unwrap()
                    .contains(&entry.symbol.as_str()),
                "{}",
                entry.symbol
            );
        }
    }
    let local = grouped_experts(&references[0]);
    assert_eq!(local, EXPERTS);
    for (tokens, rank) in [(2048, 0), (8192, 0)] {
        let ids = staircase_ids(tokens, rank, local);
        assert_eq!(ids.len(), tokens * TOP_K);
        for token in ids.chunks_exact(TOP_K) {
            assert_eq!(experts_of(token).len(), TOP_K);
        }
        let scale = tokens / 2048;
        for (i, &rows) in STAIRCASE_ROWS.iter().enumerate() {
            let at = if i + 1 == STAIRCASE_ROWS.len() {
                local - 1
            } else {
                3 * i
            };
            let expert = (rank * local + at) as i32;
            let count = ids.iter().filter(|&&id| id == expert).count();
            assert_eq!(count, (rows * scale).min(tokens), "expert {expert}");
        }
        let logits = logits_of(&ids);
        assert_eq!(logits.len(), tokens * EXPERTS);
    }
}

// ------------------------------------------------------ decoded tiles ----

/// `function` allowed the dynamic shared memory of `projection`'s plan at
/// its token capacity (decoded tiles take 99,328 bytes).
fn planned_function(function: Function, projection: &Projection) -> LabResult<Function> {
    let bytes = projection
        .plan(projection.tokens_capacity)
        .map_err(|e| e.0.to_string())?
        .dynamic_shared_bytes;
    if bytes > 0 {
        function.set_max_dynamic_shared_memory(bytes).map_err(err)?;
    }
    Ok(function)
}

/// The live counts serving can launch the decoded-tile gate/up at: a
/// prefill step of any count up to the capacity, run whole or as a B8 part
/// of up to 2,048 tokens ending anywhere. `decoded_tiles_pass_at_every_serving_count`
/// checks each.
fn decoded_serving_counts(capacity: u32) -> std::ops::RangeInclusive<u32> {
    1..=capacity
}

/// The live counts the host-side bitwise comparisons run, up to `max`:
/// zero, 1-256, 1,921-2,048, the part tails 904 and 952, 192, wave and
/// chunk sizes, and the capacity.
fn decoded_counts(max: u32) -> Vec<u32> {
    let mut counts: Vec<u32> = (0..=256)
        .chain(1921..=2048)
        .chain([
            383, 384, 385, 904, 952, 1000, 1023, 1024, 1025, 2049, 4095, 4096, 4097, 8191, 8192,
        ])
        .filter(|&count| count <= max)
        .collect();
    counts.sort_unstable();
    counts.dedup();
    counts
}

/// The live counts the decoded-tile MLPs (gate/up and the column down)
/// pass `qualify_mlps` at beyond `poison::live_counts`: verify-sized and
/// part-sized steps and the B8 part tails.
const DECODED_QUALIFIED_COUNTS: [u32; 7] = [64, 128, 192, 512, 904, 952, 2048];

/// The non-zero counts the decoded-tile MLPs are qualified at for a
/// capacity, each once.
fn decoded_qualified_counts(capacity: u32) -> Vec<u32> {
    let mut counts = Vec::new();
    for live in poison::live_counts(capacity)
        .into_iter()
        .chain(DECODED_QUALIFIED_COUNTS)
    {
        if live > 0 && live <= capacity && !counts.contains(&live) {
            counts.push(live);
        }
    }
    counts
}

/// FP8 128x128 gate/up weights `[rows, k]` whose codes span every finite
/// E4M3 magnitude (never NaN) and whose block scales include the edges of
/// `split_scale`: +0 and -0 (a zero operand), 1 and 1.5 (whole residuals),
/// 2^-100 and 2^-103 (the clamped power of two), the subnormal 2^-127 and
/// 3,000; the rest lie in [0.125, 1).
fn edge_fp8_bank(session: &Session, rows: usize, k: usize, seed: u64) -> LabResult<Bank> {
    let mut rng = Rng(seed);
    let codes = (0..rows * k)
        .map(|_| {
            loop {
                let code = rng.next() as u8;
                if code & 0x7f != 0x7f {
                    break code;
                }
            }
        })
        .collect::<Vec<_>>();
    let edges = [
        0.0,
        -0.0,
        1.0,
        1.5,
        2_f32.powi(-100),
        2_f32.powi(-103),
        2_f32.powi(-127),
        3000.0,
    ];
    let scales = (0..(rows / 128) * (k / 128))
        .map(|_| match rng.below(24) {
            i if i < edges.len() => edges[i],
            _ => (0.25 * 2_f64.powi(rng.below(3) as i32 - 1) * (1.0 + rng.unit())) as f32,
        })
        .collect::<Vec<_>>();
    Ok(Bank {
        codes: upload(session, &codes)?,
        block_scales: Some(upload(session, &bytes_of(&scales, f32::to_ne_bytes))?),
        local_scales: None,
        globals: None,
        decoded: BTreeMap::new(),
    })
}

/// `capacity` tokens of decoded-tile input: token t's values have
/// exponents in one of four ranges by t mod 4 (down to 2^-12 and up to
/// 2^4, so sums cross the clamp and the SiLU tail; with finite E4M3 codes
/// and block scales of at most 3,000 every dot product stays below about
/// 1.8e11, far from F32 overflow), with every 61st value a signed zero.
fn decoded_input(capacity: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let ranges = [-4..=0, -2..=2, 0..=4, -12..=-8];
    (0..capacity * HIDDEN)
        .map(|i| {
            if i % 61 == 0 {
                (rng.below(2) as u16) << 15
            } else {
                bf16_value(&mut rng, ranges[(i / HIDDEN) % 4].clone())
            }
        })
        .flat_map(u16::to_ne_bytes)
        .collect()
}

/// One dense FP8 prefill gate/up over the prefill capacity, for the
/// decoded-tile comparisons: the column reference and its decoded tiles
/// over one [`decoded_input`] and [`edge_fp8_bank`] weights.
struct DecodedCase {
    reference: Projection,
    candidate: Projection,
    fns: [Function; 2],
    x: DeviceBuffer,
    inter: DeviceBuffer,
    live: DeviceBuffer,
    bank: Bank,
}

impl DecodedCase {
    fn new(
        session: &Session,
        reference: &Projection,
        candidate: &Projection,
        seed: u64,
    ) -> LabResult<Self> {
        let capacity = reference.tokens_capacity as usize;
        let x = decoded_input(capacity, seed);
        Ok(Self {
            fns: [
                Entry::of(reference).load(session)?,
                planned_function(Entry::of(candidate).load(session)?, candidate)?,
            ],
            x: upload(session, &x)?,
            inter: zeroed(session, capacity * reference.columns() as usize * 2)?,
            live: zeroed(session, 4)?,
            bank: edge_fp8_bank(
                session,
                reference.query.gemm.n as usize,
                HIDDEN,
                seed ^ 0x5d,
            )?,
            reference: reference.clone(),
            candidate: candidate.clone(),
        })
    }

    /// `projection` (with `function`) planned at `tokens`.
    fn launch(
        &self,
        stream: &Stream,
        projection: &Projection,
        function: &Function,
        tokens: u32,
    ) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        let null = || KernelArgument::NullBuffer;
        let [codes, block, local, globals] = self.bank.arguments();
        launch_with(
            stream,
            function,
            projection.plan(tokens).map_err(|e| e.0.to_string())?,
            false,
            &[
                b(&self.x),
                codes,
                block,
                local,
                globals,
                null(),
                null(),
                null(),
                b(&self.live),
                null(),
                b(&self.inter),
            ],
        )
    }

    /// The column reference's intermediate at `live` tokens from a
    /// poisoned buffer, checked against its write map.
    fn reference_output(&self, session: &Session, live: u32) -> LabResult<Vec<Vec<u8>>> {
        let capacity = self.reference.tokens_capacity;
        let out = poisoned_outputs(session, &[&self.inter], || {
            self.launch(&session.stream, &self.reference, &self.fns[0], capacity)
        })?;
        poison::check(
            &format!("{} live {live}", self.reference.symbol),
            &out[0],
            &WriteMap::of(DType::Bf16).prefix(live as usize * self.reference.columns() as usize)?,
            poison::Fill::Pattern(&[POISON]),
        )?;
        Ok(out)
    }
}

/// Comparison slots one `DecodedServing::qualify_on_device` count takes.
const DEVICE_SLOTS: usize = 3 + 3 + 1 + 3 + 3 * 3;
/// Counts per receipt of the device checks.
const DEVICE_CHUNK: usize = 512;
/// The phases `DecodedServing::qualify_on_device` runs at each count.
const DEVICE_PHASES: [&str; 8] = [
    "separate",
    "planned at the capacity",
    "eager",
    "relaunch",
    "behind the producer",
    "sequence",
    "graph replay 0",
    "graph replay 1",
];

/// A dense FP8 prefill gate/up's column reference and decoded tiles for the
/// device checks at every serving count: one [`edge_fp8_bank`], three
/// [`decoded_input`]s with the producer's save areas and descriptors, and
/// per input an intermediate and the reference's intermediate.
struct DecodedServing {
    reference: Projection,
    candidate: Projection,
    reference_fn: Function,
    bank: Bank,
    inputs: Vec<DeviceBuffer>,
    fresh: Vec<DeviceBuffer>,
    operands: Vec<DeviceBuffer>,
    outputs: Vec<DeviceBuffer>,
    references: Vec<DeviceBuffer>,
    live: DeviceBuffer,
}

impl DecodedServing {
    fn new(
        session: &Session,
        reference: &Projection,
        candidate: &Projection,
        seed: u64,
    ) -> LabResult<Self> {
        let capacity = reference.tokens_capacity as usize;
        let inputs = (0..3)
            .map(|i| upload(session, &decoded_input(capacity, seed + i)))
            .collect::<LabResult<Vec<_>>>()?;
        let fresh = inputs
            .iter()
            .map(|x| session.allocate(x.byte_len()))
            .collect::<LabResult<Vec<_>>>()?;
        let operands = inputs
            .iter()
            .zip(&fresh)
            .map(|(x, f)| Producer::operand(session, x, f))
            .collect::<LabResult<Vec<_>>>()?;
        let intermediate = || session.allocate(capacity * reference.columns() as usize * 2);
        Ok(Self {
            reference_fn: Entry::of(reference).load(session)?,
            bank: edge_fp8_bank(
                session,
                reference.query.gemm.n as usize,
                HIDDEN,
                seed ^ 0x5d,
            )?,
            inputs,
            fresh,
            operands,
            outputs: (0..3).map(|_| intermediate()).collect::<LabResult<_>>()?,
            references: (0..3).map(|_| intermediate()).collect::<LabResult<_>>()?,
            live: zeroed(session, 4)?,
            reference: reference.clone(),
            candidate: candidate.clone(),
        })
    }

    /// `projection` (`function`) planned at `tokens` on input `input` into
    /// `output`.
    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        stream: &Stream,
        projection: &Projection,
        function: &Function,
        input: usize,
        output: &DeviceBuffer,
        tokens: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let b = KernelArgument::Buffer;
        let null = || KernelArgument::NullBuffer;
        let [codes, block, local, globals] = self.bank.arguments();
        launch_with(
            stream,
            function,
            projection.plan(tokens).map_err(|e| e.0.to_string())?,
            pdl,
            &[
                b(&self.inputs[input]),
                codes,
                block,
                local,
                globals,
                null(),
                null(),
                null(),
                b(&self.live),
                null(),
                b(output),
            ],
        )
    }

    /// `tiled_chain_splits`' `qualify_on_device` for the gate/up `function`
    /// (the decoded tiles or a mutant of them, planned at `live` tokens as
    /// prefill launches it) at `live` tokens, every comparison on the
    /// device over the first `checked` rows (`live` but in the
    /// unwritten-row control): the column reference's intermediate of each
    /// input into `references` from poisoned buffers (every checked BF16
    /// element written, every later byte still poison); then each input in its own launch (input 0 also planned
    /// at the capacity); input 0 eager, relaunched and behind the PDL
    /// producer; and the sequence producer, input 0, producer, input 1,
    /// input 2, every launch PDL, eager and from one graph captured before
    /// any launch, replayed twice. The intermediates and the producer's save
    /// areas are poisoned before every phase; every checked row must be
    /// written and equal the reference's, and the rest keep the poison. Returns the comparisons
    /// made and the failing ones; launch and capture errors propagate.
    fn qualify_on_device(
        &self,
        session: &Session,
        function: &Function,
        producer: &Producer,
        check: &mut DeviceCheck,
        live: u32,
        checked: u32,
    ) -> LabResult<(usize, Vec<String>)> {
        poison::write(session, &self.live, 0, &(live as i32).to_ne_bytes())?;
        let stream = &session.stream;
        let capacity = self.reference.tokens_capacity;
        let bytes = checked as usize * self.reference.columns() as usize * 2;
        let run = |input: usize, pdl: bool, stream: &Stream| {
            self.launch(
                stream,
                &self.candidate,
                function,
                input,
                &self.outputs[input],
                live,
                pdl,
            )
        };
        let sequence = |stream: &Stream| -> LabResult<()> {
            producer.launch(stream, &self.operands[0])?;
            run(0, true, stream)?;
            producer.launch(stream, &self.operands[1])?;
            run(1, true, stream)?;
            run(2, true, stream)
        };
        let graph = session.capture(sequence)?;
        let label = |what: &str| format!("{} live {live} {what}", self.candidate.symbol);
        for (input, want) in self.references.iter().enumerate() {
            check.poison(stream, &[want])?;
            self.launch(
                stream,
                &self.reference,
                &self.reference_fn,
                input,
                want,
                capacity,
                false,
            )?;
            check.compare(
                stream,
                format!("{} live {live} input {input}", self.reference.symbol),
                want,
                want,
                bytes,
                2,
            )?;
        }
        for (input, want) in self.references.iter().enumerate() {
            check.poison(stream, &[&self.outputs[input]])?;
            run(input, false, stream)?;
            check.compare(
                stream,
                label(&format!("separate input {input}")),
                want,
                &self.outputs[input],
                bytes,
                2,
            )?;
        }
        check.poison(stream, &[&self.outputs[0]])?;
        self.launch(
            stream,
            &self.candidate,
            function,
            0,
            &self.outputs[0],
            capacity,
            false,
        )?;
        check.compare(
            stream,
            label("planned at the capacity"),
            &self.references[0],
            &self.outputs[0],
            bytes,
            2,
        )?;
        for phase in ["eager", "relaunch", "behind the producer"] {
            check.poison(stream, &[&self.outputs[0], &self.fresh[0]])?;
            let behind = phase == "behind the producer";
            if behind {
                producer.launch(stream, &self.operands[0])?;
            }
            run(0, behind, stream)?;
            check.compare(
                stream,
                label(phase),
                &self.references[0],
                &self.outputs[0],
                bytes,
                2,
            )?;
        }
        let all: Vec<&DeviceBuffer> = self
            .outputs
            .iter()
            .chain([&self.fresh[0], &self.fresh[1]])
            .collect();
        for phase in ["sequence", "graph replay 0", "graph replay 1"] {
            check.poison(stream, &all)?;
            if phase == "sequence" {
                sequence(stream)?;
            } else {
                graph.launch(stream).map_err(err)?;
            }
            for (input, want) in self.references.iter().enumerate() {
                check.compare(
                    stream,
                    label(&format!("{phase} input {input}")),
                    want,
                    &self.outputs[input],
                    bytes,
                    2,
                )?;
            }
        }
        check.finish(session)
    }
}

/// The decoded-tile dense and shared-expert gate/up
/// (`decoded_shapes::projections`) pass `DecodedServing::qualify_on_device`
/// against the column schedule at every count of `decoded_serving_counts`
/// (or of the part `device_check::LIVE_COUNTS` names), behind the wide
/// producer, with a receipt per `DEVICE_CHUNK` counts.
/// `CHOPINFER_DECODED_FILTER` keeps the shapes whose candidate symbol
/// contains it.
#[test]
fn decoded_tiles_pass_at_every_serving_count() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::wide(session)?;
        let mut failures = replay::Failures::default();
        let filter = std::env::var("CHOPINFER_DECODED_FILTER").ok();
        let mut checked = 0;
        for (seed, (reference, candidate)) in decoded_shapes::references()
            .iter()
            .zip(&decoded_shapes::projections())
            .enumerate()
        {
            if filter
                .as_deref()
                .is_some_and(|f| !candidate.symbol.contains(f))
            {
                continue;
            }
            checked += 1;
            let result = (|| -> LabResult<()> {
                let case =
                    DecodedServing::new(session, reference, candidate, 1700 + 8 * seed as u64)?;
                let function = planned_function(Entry::of(candidate).load(session)?, candidate)?;
                let mut check = DeviceCheck::new(session, DEVICE_SLOTS)?;
                let counts =
                    device_check::live_counts(decoded_serving_counts(reference.tokens_capacity))?;
                let started = std::time::Instant::now();
                let mut comparisons = 0;
                let mut first = *counts.start();
                for live in counts.clone() {
                    let (compared, differ) = case
                        .qualify_on_device(session, &function, &producer, &mut check, live, live)?;
                    if !differ.is_empty() {
                        return Err(differ.join("; "));
                    }
                    comparisons += compared;
                    // A receipt per `DEVICE_CHUNK` counts, so a run cut
                    // short keeps the counts it passed.
                    if live == *counts.end() || (live - first + 1) as usize == DEVICE_CHUNK {
                        let mut receipt = BTreeMap::new();
                        receipt.insert(
                            "case".into(),
                            serde_json::json!("decoded_tile_serving_counts"),
                        );
                        receipt.insert("reference".into(), serde_json::json!(reference.symbol));
                        receipt.insert("candidate".into(), serde_json::json!(candidate.symbol));
                        receipt.insert(
                            "capacity".into(),
                            serde_json::json!(reference.tokens_capacity),
                        );
                        receipt.insert("first".into(), serde_json::json!(first));
                        receipt.insert("last".into(), serde_json::json!(live));
                        receipt.insert("phases".into(), serde_json::json!(DEVICE_PHASES));
                        receipt.insert("comparisons".into(), serde_json::json!(comparisons));
                        receipt.insert("passed".into(), serde_json::json!(true));
                        session.receipt("moe_chained", receipt)?;
                        eprintln!(
                            "{} = {} at every count {first}..={live}, {comparisons} device \
                             comparisons, {:.1} s",
                            candidate.symbol,
                            reference.symbol,
                            started.elapsed().as_secs_f64()
                        );
                        first = live + 1;
                        comparisons = 0;
                    }
                }
                Ok(())
            })();
            failures.check(&candidate.symbol, result);
        }
        if checked == 0 {
            return Err("no decoded-tile shape ran".into());
        }
        failures.finish()
    })
    .expect("decoded tiles pass at every serving count");
}

/// Decoded tiles (`chained::Schedule::DecodedTiles`, `decoded_shapes`)
/// against the column schedule they run beside, bit for bit: the dense
/// MLP's and the shared expert's FP8 prefill gate/up over 8,192-token
/// buffers, at every `decoded_counts` count, the intermediate poisoned
/// before each run and compared whole (so the written rows match too), the
/// decoded tiles planned at the capacity and at the live count. The column
/// reference runs twice at four counts and must repeat its bytes.
#[test]
fn decoded_tiles_match_the_column_schedule_bit_for_bit() {
    gpu_test(4 << 30, |session| {
        let mut failures = replay::Failures::default();
        let mut cases = Vec::new();
        for (seed, (reference, candidate)) in decoded_shapes::references()
            .iter()
            .zip(&decoded_shapes::projections())
            .enumerate()
        {
            let result = (|| -> LabResult<()> {
                let case = DecodedCase::new(session, reference, candidate, 1500 + seed as u64)?;
                let counts = decoded_counts(reference.tokens_capacity);
                for &live in &counts {
                    poison::write(session, &case.live, 0, &(live as i32).to_ne_bytes())?;
                    let columns = case.reference_output(session, live)?;
                    if [1, 129, 2048, 8192].contains(&live) {
                        let again = case.reference_output(session, live)?;
                        same_bytes(
                            &format!("{} live {live} rerun", reference.symbol),
                            &columns,
                            &again,
                        )?;
                    }
                    for tokens in [reference.tokens_capacity, live.max(1)] {
                        let got = poisoned_outputs(session, &[&case.inter], || {
                            case.launch(&session.stream, &case.candidate, &case.fns[1], tokens)
                        })?;
                        same_bytes(
                            &format!("{} live {live} planned at {tokens}", candidate.symbol),
                            &columns,
                            &got,
                        )?;
                    }
                }
                cases.push(serde_json::json!({
                    "reference": reference.symbol,
                    "candidate": candidate.symbol,
                    "live_counts": counts,
                }));
                Ok(())
            })();
            failures.check(&candidate.symbol, result);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("decoded_tile_bitwise"));
        receipt.insert("compared".into(), serde_json::json!(cases));
        receipt.insert("passed".into(), serde_json::json!(failures.is_empty()));
        session.receipt("moe_chained", receipt)?;
        failures.finish()
    })
    .expect("decoded tiles match the column schedule bit for bit");
}

/// Decoded-tile dense and shared-expert MLPs at the prefill capacity (the
/// decoded gate/up, then the column down) pass `qualify_mlps` against the
/// column schedule at every `decoded_qualified_counts` count.
#[test]
fn decoded_tile_mlps_pass_behind_a_real_producer_and_back_to_back() {
    gpu_test(8 << 30, |session| {
        let producer = Producer::new(session)?;
        let mut failures = replay::Failures::default();
        let mut qualified = Vec::new();
        for candidate in decoded_shapes::projections() {
            let intermediate = candidate.columns();
            let result = (|| -> LabResult<()> {
                let mlps = Mlps::at(session, Weights::Fp8Block128, intermediate, true, 1600)?;
                let first = &mlps.mlps[0];
                let decoded_tiles = first.gate_up.decoded_tiles().map_err(|e| e.0.to_string())?;
                if decoded_tiles.symbol != candidate.symbol {
                    return Err(format!("{} is not the MLP's gate/up", candidate.symbol));
                }
                let columns = [&first.gate_up, &first.down];
                let decoded = planned_function(Entry::of(&candidate).load(session)?, &candidate)?;
                let column_fns = [&first.fns[0], &first.fns[1]];
                for live in decoded_qualified_counts(first.capacity) {
                    mlps.set_live(session, live)?;
                    let expected = mlps.separate(session, columns, column_fns, live)?;
                    qualified.push(qualify_mlps(
                        session,
                        &mlps,
                        [&candidate, &first.down],
                        [&decoded, &first.fns[1]],
                        &producer,
                        live,
                        &expected,
                    )?);
                }
                Ok(())
            })();
            failures.check(&candidate.symbol, result);
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("decoded_tile_mlps"));
        receipt.insert("qualified".into(), serde_json::json!(qualified));
        receipt.insert("passed".into(), serde_json::json!(failures.is_empty()));
        session.receipt("moe_chained", receipt)?;
        failures.finish()
    })
    .expect("decoded-tile MLPs pass behind a real producer and back to back");
}

/// The decoded-tile mutants (`decoded_shapes::MUTATIONS`, the shared
/// expert's gate/up before the column down) each give an observed output
/// difference in some phase of `mlp_mutant_differences` and of
/// `DecodedServing::qualify_on_device` at 2,048 live tokens (the wait
/// mutants' behind the producer among them); the unmodified instance gives
/// none, a launch one token short fails every device comparison as
/// unwritten, and an injected operational failure propagates.
#[test]
fn decoded_tile_checks_reject_every_mutant() {
    gpu_test(8 << 30, |session| {
        let module = session.module(decoded_shapes::MUTANTS_CUBIN)?;
        let producer = Producer::new(session)?;
        let projection = decoded_shapes::mutant_projection();
        let mlps = Mlps::at(
            session,
            Weights::Fp8Block128,
            projection.columns(),
            true,
            1690,
        )?;
        let first = &mlps.mlps[0];
        let live = 2048;
        mlps.set_live(session, live)?;
        let expected = mlps.separate(
            session,
            [&first.gate_up, &first.down],
            [&first.fns[0], &first.fns[1]],
            live,
        )?;
        let load = |mutation: &str| -> LabResult<Function> {
            planned_function(
                module
                    .function(&decoded_shapes::mutant_symbol(mutation))
                    .map_err(err)?,
                &projection,
            )
        };
        let differences = |function: &Function, inject: bool| {
            mlp_mutant_differences(
                session,
                &mlps,
                [&projection, &first.down],
                [function, &first.fns[1]],
                &producer,
                live,
                &expected,
                inject,
            )
        };
        let control = load(decoded_shapes::MUTATIONS[0])?;
        let control_differences = differences(&control, false)?;
        if !control_differences.is_empty() {
            return Err(format!(
                "the unmodified instance differs: {control_differences:?}"
            ));
        }
        if differences(&control, true).is_ok() {
            return Err("an injected operational failure did not propagate".into());
        }
        let mut results = Vec::new();
        for &mutation in &decoded_shapes::MUTATIONS[1..] {
            let found = differences(&load(mutation)?, false)?;
            if found.is_empty() {
                return Err(format!("the {mutation} mutant passed"));
            }
            eprintln!("{mutation}: outputs differ in {found:?}");
            results.push(serde_json::json!({"mutation": mutation, "differs": found}));
        }
        // The device checks behind the wide producer of the checks at every
        // count see them too: none for the unmodified instance, an observed
        // difference for each mutant.
        let reference = decoded_shapes::references()
            .into_iter()
            .find(|r| r.query == projection.query)
            .ok_or("no column reference of the mutants' projection")?;
        let case = DecodedServing::new(session, &reference, &projection, 1790)?;
        let mut check = DeviceCheck::new(session, DEVICE_SLOTS)?;
        let wide = Producer::wide(session)?;
        let mut device = Vec::new();
        for &mutation in &decoded_shapes::MUTATIONS {
            let (_, found) =
                case.qualify_on_device(session, &load(mutation)?, &wide, &mut check, live, live)?;
            if mutation != decoded_shapes::MUTATIONS[0]
                && mutation != "long_chain"
                && !found.iter().any(|f| f.contains("behind the producer"))
            {
                return Err(format!(
                    "the {mutation} mutant shows no difference behind the producer: {found:?}"
                ));
            }
            match (mutation == decoded_shapes::MUTATIONS[0], found.is_empty()) {
                (true, false) => {
                    return Err(format!(
                        "the unmodified instance differs on the device: {found:?}"
                    ));
                }
                (false, true) => {
                    return Err(format!("the {mutation} mutant passed the device checks"));
                }
                (true, true) => {}
                (false, false) => {
                    eprintln!("{mutation} on the device: {found:?}");
                    device.push(serde_json::json!({"mutation": mutation, "differs": found}));
                }
            }
        }
        // The unwritten-row control: the reference and the decoded tiles
        // launched at one token fewer than checked leave that row's poison
        // alike; every comparison must count it unwritten.
        let (compared, short) = case.qualify_on_device(
            session,
            &load(decoded_shapes::MUTATIONS[0])?,
            &wide,
            &mut check,
            live - 1,
            live,
        )?;
        let row = projection.columns();
        if short.len() != compared
            || !short
                .iter()
                .all(|f| f.contains(&format!(" {row} live elements unwritten")))
        {
            return Err(format!(
                "a launch one token short passed {} of {compared} comparisons: {short:?}",
                compared - short.len()
            ));
        }
        eprintln!("one token short on the device: {short:?}");
        device.push(serde_json::json!({"mutation": "one_token_short", "differs": short}));
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), serde_json::json!("decoded_tile_mutants"));
        receipt.insert("mutants".into(), serde_json::json!(results));
        receipt.insert("device_checks".into(), serde_json::json!(device));
        receipt.insert("passed".into(), serde_json::json!(true));
        session.receipt("moe_chained", receipt)?;
        Ok(())
    })
    .expect("decoded-tile checks reject every mutant");
}

/// Runs without a GPU: every decoded-tile mutant instance is in its cubin,
/// each candidate keeps its reference's contract, row tile and token
/// capacity, covers it in 128-token tiles over the same column tiles, and
/// the device checks run every count from 1 through the capacity.
#[test]
fn decoded_tile_lab_instances_are_built() {
    let built = kernel_lab::cubin_entries(decoded_shapes::MUTANTS_CUBIN).unwrap();
    for mutation in decoded_shapes::MUTATIONS {
        let symbol = decoded_shapes::mutant_symbol(mutation);
        assert!(built.contains(&symbol.as_str()), "{symbol}");
    }
    let references = decoded_shapes::references();
    assert_eq!(
        references.each_ref().map(|p| p.columns()),
        SHAPES.dense_intermediates
    );
    for (reference, candidate) in references.iter().zip(&decoded_shapes::projections()) {
        assert_eq!(candidate.query, reference.query);
        assert_eq!(
            (candidate.row_tile, candidate.tokens_capacity),
            (reference.row_tile, reference.tokens_capacity)
        );
        for tokens in [1, 128, 129, 2048, 8192] {
            let (plan, columns) = (
                candidate.plan(tokens).unwrap(),
                reference.plan(tokens).unwrap(),
            );
            assert_eq!(plan.grid[0], tokens.div_ceil(128));
            assert_eq!(plan.grid[1], columns.grid[1]);
        }
        let entry = Entry::of(candidate);
        assert!(
            kernel_lab::cubin_entries(&entry.cubin)
                .unwrap()
                .contains(&entry.symbol.as_str()),
            "{}",
            entry.symbol
        );
    }
    // The device checks cover every count serving launches, 1 through the
    // capacity; the host comparisons also run zero.
    for reference in &references {
        let capacity = reference.tokens_capacity;
        assert_eq!(capacity, 8192);
        assert_eq!(
            decoded_serving_counts(capacity).collect::<Vec<_>>(),
            (1..=capacity).collect::<Vec<_>>()
        );
    }
    let counts = decoded_counts(8192);
    assert!(
        counts
            .iter()
            .all(|&c| c == 0 || decoded_serving_counts(8192).contains(&c))
    );
    let qualified = decoded_qualified_counts(8192);
    for c in [1, 64, 128, 192, 512, 904, 952, 2048, 4096, 8191, 8192] {
        assert!(qualified.contains(&c), "{c}");
    }
}

#[path = "common/invariance.rs"]
mod invariance;
#[path = "moe_chained/invariance.rs"]
mod invariance_adapter;
#[path = "moe_chained/split_invariance.rs"]
mod split_invariance;

/// The invariance adapters of this test binary, one per op kind.
fn invariance_adapters() -> Vec<Box<dyn invariance::Adapter>> {
    vec![Box::new(invariance_adapter::Chained::default())]
}

fn invariance_case(property: kernels_sm120::invariance::Property) {
    gpu_test(24 << 30, |session| {
        invariance::run(
            property,
            &mut invariance_adapters(),
            std::slice::from_ref(session),
        )
    })
    .unwrap_or_else(|error| panic!("{property:?} chained MoE invariance: {error}"));
}

#[test]
fn invariance_chained_chunk() {
    invariance_case(kernels_sm120::invariance::Property::Chunk);
}

#[test]
fn invariance_chained_rows() {
    invariance_case(kernels_sm120::invariance::Property::Rows);
}

#[test]
fn invariance_chained_batch() {
    invariance_case(kernels_sm120::invariance::Property::Batch);
}

#[test]
fn invariance_chained_ties() {
    invariance_case(kernels_sm120::invariance::Property::Ties);
}

/// This binary holds exactly the adapters the shared table places in
/// "moe_chained", and they cover every claim of their op kinds.
#[test]
fn invariance_adapters_match_the_table() {
    let held = invariance_adapters()
        .iter()
        .map(|adapter| adapter.kind())
        .collect::<Vec<_>>();
    let placed = invariance::ADAPTERS
        .iter()
        .filter(|(_, _, binary)| *binary == "moe_chained")
        .map(|(kind, _, _)| *kind)
        .collect::<Vec<_>>();
    assert_eq!(held, placed, "chained adapters against ADAPTERS");
    let uncovered = invariance::uncovered()
        .into_iter()
        .filter(|claim| {
            held.iter()
                .any(|kind| claim.contains(&format!(" {kind:?} ")))
        })
        .collect::<Vec<_>>();
    assert!(
        uncovered.is_empty(),
        "claims without an adapter: {uncovered:?}"
    );
}
