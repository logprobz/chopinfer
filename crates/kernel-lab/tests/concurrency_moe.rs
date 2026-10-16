//! Qwen3.8 Flash expert-sharded MoE on one rank: softmax
//! top-10 routing over 512 experts, this rank's 128 FP8 block experts and
//! the sigmoid-gated shared expert combine. Every local expert has distinct
//! random weights and independent uniform router logits. Routes and the
//! permutation must match the oracles exactly; gate/up, weighted down, the
//! combine and the layer are checked against the Q-4 f64 oracles.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, RouterKind, RuntimeCounts};
use kernel_core::moe_oracle::{
    ExpertMlp, ExpertMlpRounding, ExpertWeights, LocalRoutes, RouteWeightAt, expert_mlp,
};
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput, SiluGlu};

use kernel_core::transform::Transform;
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, check_output, gpu_test};

use kernels_sm120::moe::{self, Phase, Projection, Variant};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;

use std::path::PathBuf;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn upload_bytes(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    if !bytes.is_empty() {
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(err)?;
    }
    Ok(buffer)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn download_i32(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<i32>> {
    let mut bytes = vec![0; len * 4];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(err)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|v| i32::from_ne_bytes(v.try_into().unwrap()))
        .collect())
}

fn download_f32(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<f32>> {
    session.download_f32(buffer, len)
}

fn sample(seed: u64) -> f32 {
    let mut x = seed.wrapping_add(0x9e3779b97f4a7c15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    ((x >> 40) as i32 as f32 / 8_388_608.0) - 1.0
}

fn load_cubin(session: &Session, name: &str) -> LabResult<Module> {
    let path = PathBuf::from(env!("OUT_DIR")).join(format!("{name}.cubin"));
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    session.context.load_module(&bytes).map_err(err)
}

use kernel_core::contract::MoeQuery;
use kernel_core::moe_oracle::{local_expert_routes, shared_expert_gate_combine};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use model_qwen38_contract::sm120::moe as qwen;

const HANDLES: [u64; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const WEIGHT_HANDLE: u64 = 16;
const GLOBAL: usize = qwen::GLOBAL_EXPERTS as usize;
const LOCAL: usize = qwen::LOCAL_EXPERTS as usize;
const QK: usize = qwen::TOP_K as usize;
const H: usize = qwen::HIDDEN as usize;
const I: usize = qwen::INTERMEDIATE as usize;
const GATE_SEED: u64 = 0x9a7e_0000;
const DOWN_SEED: u64 = 0xd0a1_0000;
const LOGIT_SALT: u64 = 0x10_6175;

/// The rank whose experts the cases run, `CHOPINFER_QWEN_MOE_RANK` (0-3).
fn rank() -> u32 {
    std::env::var("CHOPINFER_QWEN_MOE_RANK")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&rank| rank < 4)
        .unwrap_or(1)
}

fn stream_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut bytes = Vec::with_capacity(len + 8);
    while bytes.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bytes.extend_from_slice(&state.to_le_bytes());
    }
    bytes.truncate(len);
    bytes
}

/// E4M3 codes with exponent fields 4 to 10 (magnitudes 0.125 to 15), so no
/// code is NaN and the block scales set the weight magnitude.
fn fp8_code(byte: u8) -> u8 {
    (byte & 0x87) | ((4 + (byte >> 3 & 15) % 7) << 3)
}

fn expert_codes(seed: u64, expert: usize, rows: usize, k: usize) -> Vec<u8> {
    stream_bytes(seed ^ (expert as u64) << 20, rows * k)
        .into_iter()
        .map(fp8_code)
        .collect()
}

/// BF16 `weight_scale_inv` per 128 x 128 block, as the checkpoint stores it.
fn expert_scales(seed: u64, expert: usize, rows: usize, k: usize) -> Vec<u16> {
    (0..rows / 128 * (k / 128))
        .map(|block| {
            let u = sample(seed ^ 0x5ca1e ^ (expert as u64) << 24 ^ block as u64);
            f32_to_bf16(0.0015 * (1.0 + 0.5 * u))
        })
        .collect()
}

/// The checkpoint decode of one expert, [rows, k].
fn decoded_expert(seed: u64, expert: usize, rows: usize, k: usize) -> LabResult<Vec<f64>> {
    let weight = kernel_core::moe_oracle::fp8_block_checkpoint_weight(
        &expert_codes(seed, expert, rows, k),
        rows,
        k,
        &expert_scales(seed, expert, rows, k),
    )
    .map_err(err)?;
    Ok(weight.decoded())
}

struct Bank {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
}

/// All local experts in fragment-major E4M3 tiles, and their F32 scales.
fn upload_bank(session: &Session, rows: usize, k: usize, seed: u64) -> LabResult<Bank> {
    let layout = FragmentWeightLayout {
        rows,
        k,
        format: FragmentWeightFormat::Fp8E4M3,
    };
    let codes = session.allocate(LOCAL * rows * k)?;
    let mut scales = Vec::with_capacity(LOCAL * rows / 128 * (k / 128));
    for expert in 0..LOCAL {
        let packed = layout
            .pack(&expert_codes(seed, expert, rows, k))
            .map_err(err)?;
        codes
            .copy_from_bytes_at(expert * rows * k, &packed.bytes, &session.stream)
            .map_err(err)?;
        session.stream.synchronize().map_err(err)?;
        scales.extend(
            expert_scales(seed, expert, rows, k)
                .into_iter()
                .map(bf16_to_f32),
        );
    }
    Ok(Bank {
        codes,
        scales: session.upload_f32(&scales)?,
    })
}

struct Banks {
    gate: Bank,
    down: Bank,
}

impl Banks {
    fn new(session: &Session) -> LabResult<Self> {
        Ok(Self {
            gate: upload_bank(session, 2 * I, H, GATE_SEED)?,
            down: upload_bank(session, H, I, DOWN_SEED)?,
        })
    }
}

// Independent identically distributed logits, with no popular-expert bias.
fn uniform_logits(tokens: usize) -> Vec<f32> {
    (0..tokens * GLOBAL)
        .map(|i| sample(LOGIT_SALT ^ i as u64))
        .collect()
}

fn bf16_values(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    (0..len)
        .map(|index| f32_to_bf16(scale * sample(seed ^ index as u64)))
        .collect()
}

struct Case {
    tokens: usize,
    rank: u32,
    phase: Phase,
    gate: Variant,
    down: Variant,
    gate_fn: Function,
    down_fn: Function,
    route: Module,
    padded_capacity: usize,
    logits: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    counts: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    active: DeviceBuffer,
    live: DeviceBuffer,
    tile_counts: DeviceBuffer,
    tile_offsets: DeviceBuffer,
    x: DeviceBuffer,
    inter: DeviceBuffer,
    routed: DeviceBuffer,
    shared: DeviceBuffer,
    gate_weight: DeviceBuffer,
    output: DeviceBuffer,
    dummy: DeviceBuffer,
    logits_host: Vec<f32>,
    x_host: Vec<u16>,
    shared_host: Vec<u16>,
    gate_weight_host: Vec<u16>,
}

fn qwen_variant(phase: Phase, projection: Projection, tokens: usize) -> LabResult<Variant> {
    let bucket = if phase == Phase::Prefill {
        qwen::PREFILL_CAPACITY
    } else {
        tokens as u32
    };
    moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| {
            v.fp8_fragment_weights()
                && v.phase == phase
                && v.projection == projection
                && v.tokens_capacity == bucket
        })
        .ok_or_else(|| format!("missing Qwen3.8 MoE variant for {tokens} tokens"))
}

fn gemm_function(session: &Session, v: &Variant, tokens: usize) -> LabResult<Function> {
    let cubin = moe::cubin_specs(
        kernel_lab::catalogs(),
        Some(model_qwen38_contract::sm120::config::PROFILE),
    )
    .into_iter()
    .find(|spec| spec.entries.iter().any(|entry| entry.symbol == v.symbol))
    .ok_or_else(|| format!("missing cubin for {}", v.symbol))?;
    let function = load_cubin(session, &cubin.name)?
        .function(&v.symbol)
        .map_err(err)?;
    let plan = v.plan(counts(v, tokens)).map_err(err)?;
    if plan.dynamic_shared_bytes > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(err)?;
    }
    Ok(function)
}

fn counts(v: &Variant, tokens: usize) -> RuntimeCounts {
    RuntimeCounts {
        m: tokens as u32,
        batch: 0,
        context: 0,
        routed_rows: (tokens * QK) as u32,
        split_k: v.knobs.split_k_capacity,
    }
}

fn zeroed(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.max(4))?;
    buffer
        .zero_at_async(0, bytes.max(4), &session.stream)
        .map_err(err)?;
    Ok(buffer)
}

use kernel_lab::poison::{Phase as ReplayPhase, WriteMap};

impl Case {
    fn checks<'a>(
        &'a self,
        s: &'a Session,
        operation: &str,
    ) -> LabResult<conc::replay::Checks<'a>> {
        use conc::replay::{Checks, RouteBuffers, RouteGeometry};
        let routes = self.tokens * QK;
        let experts = self.rank as usize * LOCAL..(self.rank as usize + 1) * LOCAL;
        let mut checks = Checks::new(s);
        if ["route", "full"].contains(&operation) {
            checks = checks.route(
                RouteGeometry {
                    routes,
                    experts: experts.clone(),
                    block: 16,
                    sorted_offset: 0,
                },
                RouteBuffers {
                    ids: &self.ids,
                    weights: &self.weights,
                    counts: &self.counts,
                    offsets: &self.offsets,
                    sorted: &self.sorted,
                    padded: &self.padded,
                    active: Some(&self.active),
                },
                &[0x80],
            )?;
            if !self.decode() {
                for (name, buffer) in [
                    ("tile counts", &self.tile_counts),
                    ("tile offsets", &self.tile_offsets),
                ] {
                    checks = checks.output(
                        name,
                        buffer,
                        WriteMap::of(DType::I32).prefix(routes.div_ceil(256) * LOCAL)?,
                    );
                }
            }
        }
        for (name, buffer, width, selected) in [
            (
                "gate/up",
                &self.inter,
                I,
                operation == "gate_up" || operation == "full",
            ),
            (
                "weighted down",
                &self.routed,
                H,
                operation == "down" || operation == "full",
            ),
        ] {
            if selected {
                let experts = experts.clone();
                checks = checks.dynamic(
                    name,
                    buffer,
                    move |session| {
                        conc::replay::routed_map(
                            &conc::replay::ids(session, &self.ids, routes)?,
                            experts.clone(),
                            width,
                        )
                    },
                    &[0xff],
                );
            }
        }
        if ["combine", "full"].contains(&operation) {
            checks = checks.output(
                "combine",
                &self.output,
                WriteMap::of(DType::Bf16).prefix(self.tokens * H)?,
            );
        }
        Ok(checks)
    }

    fn new(session: &Session, phase: Phase, tokens: usize) -> LabResult<Self> {
        let gate = qwen_variant(phase, Projection::GateUp, tokens)?;
        let down = qwen_variant(phase, Projection::Down, tokens)?;
        let padded_capacity = match gate.query.grouped.ok_or("missing grouped shape")?.capacity {
            kernel_core::contract::RoutedCapacity::TotalRows(rows) => rows as usize,
            kernel_core::contract::RoutedCapacity::PerExpert(_) => {
                return Err("Qwen3.8 MoE uses total row capacity".into());
            }
        };
        let routes = tokens * QK;
        let tiles = routes.div_ceil(256);
        let logits_host = uniform_logits(tokens);
        let x_host = bf16_values(0x7a11, tokens * H, 1.0);
        let shared_host = bf16_values(0x54a4, tokens * H, 0.05);
        let gate_weight_host = bf16_values(0x6a7e, H, 0.03);
        let route = load_cubin(session, qwen::ROUTE_CUBIN)?;
        Ok(Self {
            tokens,
            rank: rank(),
            phase,
            gate_fn: gemm_function(session, &gate, tokens)?,
            down_fn: gemm_function(session, &down, tokens)?,
            gate,
            down,
            route,
            padded_capacity,
            logits: session.upload_f32(&logits_host)?,
            ids: zeroed(session, routes * 4)?,
            weights: zeroed(session, routes * 4)?,
            counts: zeroed(session, LOCAL * 4)?,
            offsets: zeroed(session, (LOCAL + 1) * 4)?,
            sorted: zeroed(session, padded_capacity * 4)?,
            padded: zeroed(session, 4)?,
            active: zeroed(session, (LOCAL + 1) * 4)?,
            live: upload_bytes(session, &i32_bytes(&[tokens as i32]))?,
            tile_counts: zeroed(session, tiles * LOCAL * 4)?,
            tile_offsets: zeroed(session, tiles * LOCAL * 4)?,
            x: session.upload_bf16(&x_host)?,
            inter: zeroed(session, routes * I * 2)?,
            routed: zeroed(session, routes * H * 2)?,
            shared: session.upload_bf16(&shared_host)?,
            gate_weight: session.upload_bf16(&gate_weight_host)?,
            output: zeroed(session, tokens * H * 2)?,
            dummy: zeroed(session, 256)?,
            logits_host,
            x_host,
            shared_host,
            gate_weight_host,
        })
    }

    fn decode(&self) -> bool {
        self.phase == Phase::Decode
    }

    /// Route grids use the decode bucket, or the live prefill length.
    fn capacity(&self) -> u32 {
        if self.decode() {
            self.gate.tokens_capacity
        } else {
            self.tokens as u32
        }
    }

    fn launch_entry(
        &self,
        session: &Session,
        operation: &str,
        buffers: &[&DeviceBuffer],
        filtered: bool,
        pdl: bool,
    ) -> LabResult<()> {
        let entry = qwen::route_entry(operation).map_err(err)?;
        let function = self.route.function(&entry.symbol).map_err(err)?;
        let mut plan = entry
            .plan(self.capacity(), self.tokens as u32)
            .map_err(err)?;
        plan.pdl = pdl;
        let handles = (1..=buffers.len() as u64).collect::<Vec<_>>();
        let args = entry
            .arguments(&handles, filtered.then_some(self.rank))
            .map_err(err)?;
        let bindings = Bindings {
            buffers: handles
                .iter()
                .copied()
                .zip(buffers.iter().copied())
                .collect(),
        };
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            plan,
            &args,
            &bindings,
        )
    }

    fn launch_route(&self, session: &Session, pdl: bool) -> LabResult<()> {
        if self.decode() {
            return self.launch_entry(
                session,
                &format!("decode_route_t{}", self.gate.tokens_capacity),
                &[
                    &self.logits,
                    &self.ids,
                    &self.weights,
                    &self.counts,
                    &self.offsets,
                    &self.sorted,
                    &self.padded,
                    &self.live,
                    &self.active,
                ],
                true,
                pdl,
            );
        }
        self.launch_entry(
            session,
            "router",
            &[&self.logits, &self.ids, &self.weights, &self.live],
            false,
            pdl,
        )?;
        self.launch_entry(
            session,
            "permute_count",
            &[&self.ids, &self.tile_counts, &self.live],
            true,
            pdl,
        )?;
        self.launch_entry(
            session,
            "permute_prefix",
            &[
                &self.tile_counts,
                &self.counts,
                &self.tile_offsets,
                &self.live,
            ],
            false,
            pdl,
        )?;
        self.launch_entry(
            session,
            "permute_scan",
            &[
                &self.counts,
                &self.offsets,
                &self.sorted,
                &self.padded,
                &self.active,
            ],
            false,
            pdl,
        )?;
        self.launch_entry(
            session,
            "permute_scatter",
            &[
                &self.ids,
                &self.tile_offsets,
                &self.offsets,
                &self.sorted,
                &self.live,
            ],
            true,
            pdl,
        )
    }

    fn launch_projection(
        &self,
        session: &Session,
        banks: &Banks,
        projection: Projection,
        pdl: bool,
    ) -> LabResult<()> {
        let (v, function, bank, input, output) = match projection {
            Projection::GateUp => (&self.gate, &self.gate_fn, &banks.gate, &self.x, &self.inter),
            Projection::Down => (
                &self.down,
                &self.down_fn,
                &banks.down,
                &self.inter,
                &self.routed,
            ),
        };
        let mut buffers = BTreeMap::new();
        for (handle, buffer) in [
            input,
            &self.dummy,
            &self.dummy,
            &self.dummy,
            &bank.scales,
            output,
            &self.weights,
            &self.offsets,
            &self.sorted,
            &self.padded,
            &self.live,
            &self.active,
            &self.dummy,
            &self.dummy,
            &self.dummy,
            &bank.codes,
        ]
        .into_iter()
        .enumerate()
        {
            buffers.insert(handle as u64 + 1, buffer);
        }
        let args = v
            .arguments(HANDLES, v.weight_tensor_map(WEIGHT_HANDLE).map_err(err)?)
            .map_err(err)?;
        let mut plan = v.plan(counts(v, self.tokens)).map_err(err)?;
        plan.pdl = pdl;
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    }

    fn launch_combine(&self, session: &Session, pdl: bool) -> LabResult<()> {
        self.launch_entry(
            session,
            "combine",
            &[
                &self.routed,
                &self.ids,
                &self.shared,
                &self.x,
                &self.gate_weight,
                &self.output,
                &self.live,
            ],
            true,
            pdl,
        )
    }

    fn launch(&self, session: &Session, banks: &Banks, pdl: bool) -> LabResult<()> {
        self.launch_route(session, pdl)?;
        self.launch_projection(session, banks, Projection::GateUp, pdl)?;
        self.launch_projection(session, banks, Projection::Down, pdl)?;
        self.launch_combine(session, pdl)
    }

    fn outputs(&self, session: &Session) -> LabResult<(Vec<u16>, Vec<u16>, Vec<u16>)> {
        session.stream.synchronize().map_err(err)?;
        let routes = self.tokens * QK;
        Ok((
            session.download_bf16(&self.inter, routes * I)?,
            session.download_bf16(&self.routed, routes * H)?,
            session.download_bf16(&self.output, self.tokens * H)?,
        ))
    }
}

struct Routing {
    weights: Vec<f32>,
    local: kernel_core::moe_oracle::LocalRoutes,
    /// Oracle router weights per global route.
    oracle_weights: Vec<f64>,
    counts: Vec<usize>,
    offsets: Vec<i32>,
    sorted: Vec<i32>,
}

impl Routing {
    fn stats(&self) -> Value {
        let active = self.counts.iter().filter(|&&c| c > 0).count();
        let total = self.counts.iter().sum::<usize>();
        let max = *self.counts.iter().max().unwrap_or(&0);
        json!({"local_routes": total, "active_local_experts": active,
            "max_rows": max,
            "max_over_active_mean": if active > 0 { max as f64 * active as f64 / total as f64 } else { 0.0 },
            "padded_rows": self.sorted.len()})
    }
}

/// Router IDs must match the f64 oracle exactly and weights within F32
/// rounding; the local routes and the padded stable permutation must match
/// exactly.
fn check_routing(session: &Session, case: &Case) -> LabResult<Routing> {
    session.stream.synchronize().map_err(err)?;
    let routes = case.tokens * QK;
    let ids = download_i32(session, &case.ids, routes)?;
    let weights = download_f32(session, &case.weights, routes)?;
    let mut selections = Vec::with_capacity(case.tokens);
    let mut oracle_weights = Vec::with_capacity(routes);
    for token in 0..case.tokens {
        let scores = case.logits_host[token * GLOBAL..(token + 1) * GLOBAL]
            .iter()
            .map(|&v| f64::from(v))
            .collect::<Vec<_>>();
        let selection = oracle::router(
            &scores,
            None,
            QK,
            RouterKind::SoftmaxTopK { renormalize: true },
        )
        .map_err(err)?;
        for slot in 0..QK {
            let route = token * QK + slot;
            if ids[route] as usize != selection.expert_ids[slot] {
                return Err(format!(
                    "token {token} route {slot}: expert {} against oracle {}",
                    ids[route], selection.expert_ids[slot]
                ));
            }
            if (f64::from(weights[route]) - selection.weights[slot]).abs() > 2e-6 {
                return Err(format!(
                    "token {token} route {slot}: weight {} against oracle {}",
                    weights[route], selection.weights[slot]
                ));
            }
        }
        oracle_weights.extend_from_slice(&selection.weights);
        selections.push(selection);
    }
    let local = local_expert_routes(
        MoeQuery::LocalRoutes {
            experts: GLOBAL as u32,
            local_experts: LOCAL as u32,
            top_k: QK as u32,
            capacity: case.tokens as u32,
        },
        &selections,
        case.rank as usize,
    )
    .map_err(err)?;
    let block = qwen::ROUTE_BLOCK as usize;
    let mut counts = vec![0_usize; LOCAL];
    let mut expected_offsets = Vec::with_capacity(LOCAL + 1);
    let mut expected_sorted = Vec::new();
    for (expert, count) in counts.iter_mut().enumerate() {
        expected_offsets.push(expected_sorted.len() as i32);
        for (&route, &route_expert) in local.routes.iter().zip(&local.experts) {
            if route_expert == expert {
                expected_sorted.push(route as i32);
                *count += 1;
            }
        }
        expected_sorted.resize(expected_sorted.len().div_ceil(block) * block, -1);
    }
    expected_offsets.push(expected_sorted.len() as i32);
    let padded = download_i32(session, &case.padded, 1)?[0] as usize;
    let offsets = download_i32(session, &case.offsets, LOCAL + 1)?;
    let sorted = download_i32(session, &case.sorted, padded.min(case.padded_capacity))?;
    let gpu_counts = download_i32(session, &case.counts, LOCAL)?;
    let active = download_i32(session, &case.active, LOCAL + 1)?;
    let expected_active = (0..LOCAL)
        .filter(|&e| counts[e] > 0)
        .map(|e| e as i32)
        .collect::<Vec<_>>();
    if padded != expected_sorted.len()
        || offsets != expected_offsets
        || sorted != expected_sorted
        || gpu_counts != counts.iter().map(|&count| count as i32).collect::<Vec<_>>()
        || active[0] as usize != expected_active.len()
        || active[1..=expected_active.len()] != expected_active[..]
    {
        return Err("local routes or their permutation differ from the oracle".into());
    }
    Ok(Routing {
        weights,
        local,
        oracle_weights,
        counts,
        offsets,
        sorted: expected_sorted,
    })
}

/// Checked routes: every local route in decode; in prefill the routes of
/// three tokens and the tile-edge routes of the busiest and quietest
/// experts, which cover partial 128-row tiles and their row groups.
fn checked(case: &Case, routing: &Routing) -> (Vec<usize>, Vec<usize>) {
    if case.decode() {
        return ((0..case.tokens).collect(), routing.local.routes.clone());
    }
    let tokens = vec![0, case.tokens / 2, case.tokens - 1];
    let mut routes = routing
        .local
        .routes
        .iter()
        .copied()
        .filter(|route| tokens.contains(&(route / QK)))
        .collect::<Vec<_>>();
    let busiest = (0..LOCAL).max_by_key(|&e| routing.counts[e]).unwrap_or(0);
    let quietest = (0..LOCAL)
        .filter(|&e| routing.counts[e] > 0)
        .min_by_key(|&e| routing.counts[e])
        .unwrap_or(0);
    for expert in [busiest, quietest] {
        let first = routing.offsets[expert] as usize;
        let count = routing.counts[expert];
        for local in [
            0,
            15,
            16,
            63,
            64,
            127,
            128,
            129,
            count / 2,
            count.saturating_sub(1),
        ] {
            if local < count {
                routes.push(routing.sorted[first + local] as usize);
            }
        }
    }
    routes.sort_unstable();
    routes.dedup();
    (tokens, routes)
}

fn decoded(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn append(total: &mut OracleOutput, part: &OracleOutput) {
    total.values.extend_from_slice(&part.values);
    total
        .sum_abs_products
        .extend_from_slice(&part.sum_abs_products);
    total
        .reduction_lengths
        .extend_from_slice(&part.reduction_lengths);
}

fn empty_output() -> OracleOutput {
    OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    }
}

/// Gate/up, weighted down, combine and the layer against the f64 oracles.
fn check_numerics(
    session: &Session,
    case: &Case,
    routing: &Routing,
    receipt: &mut BTreeMap<String, Value>,
) -> LabResult<()> {
    let (inter, routed, output) = case.outputs(session)?;
    let (tokens, routes) = checked(case, routing);
    let local_expert = routing
        .local
        .routes
        .iter()
        .zip(&routing.local.experts)
        .map(|(&route, &expert)| (route, expert))
        .collect::<BTreeMap<_, _>>();
    let x = decoded(&case.x_host);
    // Per checked route: oracle rows over the GPU inputs, and the f64 chain.
    let mut gate_rows = BTreeMap::new();
    let mut down_rows = BTreeMap::new();
    let mut chain = BTreeMap::new();
    let mut by_expert = BTreeMap::<usize, Vec<usize>>::new();
    for &route in &routes {
        by_expert
            .entry(local_expert[&route])
            .or_default()
            .push(route);
    }
    for (&expert, expert_routes) in &by_expert {
        let gate_up = decoded_expert(GATE_SEED, expert, 2 * I, H)?;
        let down = decoded_expert(DOWN_SEED, expert, H, I)?;
        // The f64 chain of each route: gate/up, SwiGLU, down, route weight.
        let count = expert_routes.len();
        let rows = expert_routes
            .iter()
            .flat_map(|&route| x[(route / QK) * H..(route / QK + 1) * H].iter().copied())
            .collect::<Vec<_>>();
        let local = LocalRoutes {
            tokens: (0..count).collect(),
            experts: vec![0; count],
            routes: expert_routes.clone(),
        };
        let route_weights = expert_routes
            .iter()
            .map(|&route| routing.oracle_weights[route])
            .collect::<Vec<_>>();
        let weights = ExpertWeights {
            gate_up: &gate_up,
            down: &down,
            experts: 1,
            hidden: H,
            intermediate: I,
        };
        let mlp = ExpertMlp {
            activation: SiluGlu::UNCLAMPED,
            route_weight_at: RouteWeightAt::AfterDown,
            rounding: ExpertMlpRounding::default(),
        };
        let exact = expert_mlp(&rows, count, &local, &route_weights, weights, mlp).map_err(err)?;
        let no_epilogue = GemmEpilogue {
            residual: None,
            swiglu: None,
        };
        let down_shape = GemmShape {
            m: 1,
            n: H,
            k: I,
            experts: 1,
        };
        for (index, &route) in expert_routes.iter().enumerate() {
            let gpu_h = decoded(&inter[route * I..(route + 1) * I]);
            let mut y = oracle::gemm(&gpu_h, &down, down_shape, &[], no_epilogue).map_err(err)?;
            let weight = f64::from(routing.weights[route]);
            for (value, bound) in y.values.iter_mut().zip(&mut y.sum_abs_products) {
                *value *= weight;
                *bound *= weight.abs();
            }
            let h = &exact.intermediate;
            let at = index * I..(index + 1) * I;
            let h = OracleOutput {
                values: h.values[at.clone()].to_vec(),
                sum_abs_products: h.sum_abs_products[at.clone()].to_vec(),
                reduction_lengths: h.reduction_lengths[at].to_vec(),
            };
            gate_rows.insert(route, h);
            down_rows.insert(route, y);
            chain.insert(
                route,
                exact.weighted.values[index * H..(index + 1) * H].to_vec(),
            );
        }
    }
    let mut gate_reference = empty_output();
    let mut down_reference = empty_output();
    let mut gate_actual = Vec::new();
    let mut down_actual = Vec::new();
    for &route in &routes {
        append(&mut gate_reference, &gate_rows[&route]);
        append(&mut down_reference, &down_rows[&route]);
        gate_actual.extend(
            inter[route * I..(route + 1) * I]
                .iter()
                .map(|&b| bf16_to_f32(b)),
        );
        down_actual.extend(
            routed[route * H..(route + 1) * H]
                .iter()
                .map(|&b| bf16_to_f32(b)),
        );
    }
    let mut gate_receipt = BTreeMap::new();
    let gate_result = check_output(
        &gate_actual,
        &gate_reference,
        DType::Bf16,
        &mut gate_receipt,
    );
    receipt.insert("gate_up".into(), json!(gate_receipt));
    gate_result.map_err(|e| format!("gate/up: {e}"))?;
    let mut down_receipt = BTreeMap::new();
    let down_result = check_output(
        &down_actual,
        &down_reference,
        DType::Bf16,
        &mut down_receipt,
    );
    receipt.insert("down".into(), json!(down_receipt));
    down_result.map_err(|e| format!("weighted down: {e}"))?;

    // Combine: the Q-4 values over the GPU's routed rows. The bound adds
    // the gate logit's F32 reduction error, through sigmoid' <= 1/4.
    let gate_weight = decoded(&case.gate_weight_host);
    let mut combine_reference = empty_output();
    let mut combine_actual = Vec::new();
    let mut layer_reference = Vec::new();
    let mut layer_actual = Vec::new();
    for &token in &tokens {
        let row = &x[token * H..(token + 1) * H];
        let logit = row
            .iter()
            .zip(&gate_weight)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let logit_s = row
            .iter()
            .zip(&gate_weight)
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>();
        let shared = decoded(&case.shared_host[token * H..(token + 1) * H]);
        let mut sum = vec![0.0; H];
        let mut sum_abs = vec![0.0; H];
        let mut exact = vec![0.0; H];
        for slot in 0..QK {
            let route = token * QK + slot;
            if !local_expert.contains_key(&route) {
                continue;
            }
            for (col, &bits) in routed[route * H..(route + 1) * H].iter().enumerate() {
                let value = f64::from(bf16_to_f32(bits));
                sum[col] += value;
                sum_abs[col] += value.abs();
            }
            for (col, value) in chain[&route].iter().enumerate() {
                exact[col] += value;
            }
        }
        let query = MoeQuery::SharedExpertGateCombine {
            width: H as u32,
            capacity: case.tokens as u32,
        };
        let values = shared_expert_gate_combine(query, &sum, &shared, &[logit])
            .map_err(err)?
            .values;
        let gate = 1.0 / (1.0 + (-logit).exp());
        let bounds = (0..H)
            .map(|col| {
                sum_abs[col] + (gate * shared[col]).abs() + 0.25 * shared[col].abs() * logit_s
            })
            .collect::<Vec<_>>();
        append(
            &mut combine_reference,
            &OracleOutput::new(values, bounds, H).map_err(err)?,
        );
        combine_actual.extend(
            output[token * H..(token + 1) * H]
                .iter()
                .map(|&b| bf16_to_f32(b)),
        );
        let exact = shared_expert_gate_combine(query, &exact, &shared, &[logit])
            .map_err(err)?
            .values;
        layer_reference.extend(exact);
        layer_actual.extend(
            output[token * H..(token + 1) * H]
                .iter()
                .map(|&b| f64::from(bf16_to_f32(b))),
        );
    }
    let mut combine_receipt = BTreeMap::new();
    let combine_result = check_output(
        &combine_actual,
        &combine_reference,
        DType::Bf16,
        &mut combine_receipt,
    );
    receipt.insert("combine".into(), json!(combine_receipt));
    combine_result.map_err(|e| format!("combine: {e}"))?;
    let worst = layer_actual
        .chunks_exact(H)
        .zip(layer_reference.chunks_exact(H))
        .map(|(a, r)| {
            let dot = a.iter().zip(r).map(|(x, y)| x * y).sum::<f64>();
            let an = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let rn = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            dot / (an * rn)
        })
        .fold(1.0, f64::min);
    receipt.insert("layer_cosine_min".into(), json!(worst));
    oracle::check_layer_cosine(&layer_actual, &layer_reference, H)
        .map_err(|failure| format!("layer cosine {} at token {}", failure.actual, failure.index))?;
    receipt.insert("checked_routes".into(), json!(routes.len()));
    receipt.insert("checked_tokens".into(), json!(tokens));
    Ok(())
}

/// Graph replay, serialized launches and a repeated run must reproduce the
/// eager PDL outputs bitwise.
fn check_replay(
    session: &Session,
    case: &Case,
    banks: &Banks,
    checks: &mut conc::replay::Checks<'_>,
) -> LabResult<()> {
    let graph = session.capture(|_| case.launch(session, banks, true))?;
    let mut failures = conc::replay::Failures::default();
    for iteration in 0..2 {
        failures.check(
            &format!("graph {iteration}"),
            checks.phase(ReplayPhase::Graph, true, || {
                graph.launch(&session.stream).map_err(err)
            }),
        );
    }
    failures.check(
        "serialized",
        checks.phase(ReplayPhase::Relaunch, false, || {
            case.launch(session, banks, false)
        }),
    );
    failures.check(
        "PDL",
        checks.phase(ReplayPhase::Pdl, true, || case.launch(session, banks, true)),
    );
    failures.finish()
}

#[path = "common/concurrency.rs"]
mod conc;
#[test]
fn m3_qwen_uniform_moe() {
    if !conc::enabled("m3") {
        return;
    }
    gpu_test(8 << 30, |s| {
        conc::exclusive(s)?;
        let banks = Banks::new(s)?;
        let mut failures = conc::replay::Failures::default();
        for rows in conc::ROWS {
            let result = (|| -> LabResult<()> {
                let case = Case::new(s, Phase::Prefill, rows)?;
                let mut checks = case.checks(s, "full")?;
                checks.phase(ReplayPhase::Eager, false, || case.launch(s, &banks, true))?;
                let routing = check_routing(s, &case)?;
                let mut receipt = BTreeMap::from([
                    ("rows".into(), json!(rows)),
                    ("expert_rank".into(), json!(case.rank)),
                    (
                        "artifacts".into(),
                        json!([
                            conc::evidence::identity(qwen::ROUTE_CUBIN)?,
                            conc::evidence::identity(
                                "chop_moe_qwen38_gateup_prefill_fp8block_a16_t8192"
                            )?,
                            conc::evidence::identity(
                                "chop_moe_qwen38_down_prefill_fp8block_a16_t8192"
                            )?,
                        ]),
                    ),
                    ("distribution".into(), json!("iid uniform logits")),
                    ("routing".into(), routing.stats()),
                ]);
                check_numerics(s, &case, &routing, &mut receipt)?;
                check_replay(s, &case, &banks, &mut checks)?;
                checks.record(&mut receipt);
                let path = s.receipt("concurrency-moe-correctness", receipt)?;
                eprintln!(
                    "CONC_LAB_MOE rows={rows} expert_rank={} routing={} receipt={}",
                    case.rank,
                    routing.stats(),
                    path.display()
                );
                let before = case.outputs(s)?;
                for operation in ["route", "gate_up", "down", "combine", "full"] {
                    let run = || match operation {
                        "route" => case.launch_route(s, true),
                        "gate_up" => case.launch_projection(s, &banks, Projection::GateUp, true),
                        "down" => case.launch_projection(s, &banks, Projection::Down, true),
                        "combine" => case.launch_combine(s, true),
                        _ => case.launch(s, &banks, true),
                    };
                    let mut checks = case.checks(s, operation)?;
                    checks.phase(ReplayPhase::Eager, false, run)?;
                    conc::measure(
                        s,
                        "M3",
                        operation,
                        rows,
                        if operation == "route" || operation == "combine" {
                            qwen::ROUTE_CUBIN
                        } else if operation == "down" {
                            "chop_moe_qwen38_down_prefill_fp8block_a16_t8192"
                        } else {
                            "chop_moe_qwen38_gateup_prefill_fp8block_a16_t8192"
                        },
                        run,
                        || {
                            check_routing(s, &case)?;
                            assert_eq!(before, case.outputs(s)?);
                            Ok(())
                        },
                        |graph| {
                            checks.phase(ReplayPhase::Graph, true, || {
                                graph.launch(&s.stream).map_err(err)
                            })?;
                            let mut fields = BTreeMap::new();
                            checks.record(&mut fields);
                            Ok(fields)
                        },
                    )?;
                }

                Ok(())
            })();
            failures.check(&format!("rows={rows}"), result);
        }
        failures.finish()?;
        Ok(())
    })
    .expect("M3 uniform Qwen MoE")
    .expect("GPU required");
}
