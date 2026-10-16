//! Qwen3.8 Flash expert-sharded MoE on one rank: softmax
//! top-10 routing over 512 experts, this rank's FP8 block experts (128 at
//! TP4 and 256 at TP2, by `CHOPINFER_QWEN38_RANKS`) and the
//! sigmoid-gated shared expert combine. Every local expert has distinct
//! random weights and the router logits are skewed. Routes and the
//! permutation must match the oracles exactly; gate/up, weighted down, the
//! combine and the layer are checked against the Q-4 f64 oracles.

use super::*;
use kernel_core::contract::MoeQuery;
use kernel_core::moe_oracle::{local_expert_routes, shared_expert_gate_combine};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use model_qwen38_contract::sm120::moe as qwen;

#[path = "moe_qwen_invariance.rs"]
pub(crate) mod invariance;

const GLOBAL: usize = qwen::GLOBAL_EXPERTS as usize;
const QK: usize = qwen::TOP_K as usize;
const H: usize = qwen::HIDDEN as usize;
const I: usize = qwen::INTERMEDIATE as usize;
const GATE_SEED: u64 = 0x9a7e_0000;
const DOWN_SEED: u64 = 0xd0a1_0000;
const LOGIT_SALT: u64 = 0x10_6175;

/// The Qwen3.8 profile under test (`CHOPINFER_QWEN38_RANKS`, default TP4)
/// and the local experts of one of its ranks.
fn topology() -> LabResult<(Profile, usize)> {
    let profile = kernel_lab::qwen38_profile()?;
    let local = qwen::local_experts(profile).map_err(err)?;
    Ok((profile, local as usize))
}

/// Records the rank count in a receipt, except at TP4, whose receipts keep
/// their fields.
fn record_ranks(profile: Profile, receipt: &mut BTreeMap<String, Value>) {
    if let Some(ranks) = profile
        .ranks_of(model_qwen38_contract::sm120::config::MODEL)
        .filter(|&ranks| ranks != 4)
    {
        receipt.insert("qwen38_ranks".into(), json!(ranks));
    }
}

/// The rank whose experts the cases run, `CHOPINFER_QWEN_MOE_RANK` (below
/// the rank count of `local` experts per rank).
fn rank(local: usize) -> u32 {
    let ranks = (GLOBAL / local) as u32;
    std::env::var("CHOPINFER_QWEN_MOE_RANK")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&rank| rank < ranks)
        .unwrap_or(1)
}

/// `CHOPINFER_QWEN_MOE_TOKENS` selects one token count.
fn selected(tokens: usize) -> bool {
    std::env::var("CHOPINFER_QWEN_MOE_TOKENS")
        .ok()
        .is_none_or(|value| value == tokens.to_string())
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

/// Rows `picked` of the checkpoint decode of one expert, [picked.len(), k].
fn decoded_rows(
    seed: u64,
    expert: usize,
    rows: usize,
    k: usize,
    picked: &[usize],
) -> LabResult<Vec<f64>> {
    let weight = kernel_core::moe_oracle::fp8_block_checkpoint_weight(
        &expert_codes(seed, expert, rows, k),
        rows,
        k,
        &expert_scales(seed, expert, rows, k),
    )
    .map_err(err)?;
    let weight = &weight;
    Ok(picked
        .iter()
        .flat_map(|&row| (0..k).map(move |col| weight.decode(row, col)))
        .collect())
}

/// Lean runs (`kernel_lab::qwen38_lean`, TP2) check sampled output
/// columns: every 32nd intermediate and every 64th hidden column, both sides
/// of every 128-column boundary (the weight scale blocks and the 128 and 256
/// wide tiles), and the first and last. Full runs check every column.
fn checked_columns(lean: bool) -> (Vec<usize>, Vec<usize>) {
    if !lean {
        return ((0..I).collect(), (0..H).collect());
    }
    let blocks = |len: usize| (128..len).step_by(128).collect::<Vec<_>>();
    (
        kernel_lab::sample_indices(I, 32, &blocks(I)),
        kernel_lab::sample_indices(H, 64, &blocks(H)),
    )
}

struct Bank {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
    /// Code and F32 scale bytes of one expert.
    expert_bytes: usize,
}

/// All `local` experts in fragment-major E4M3 tiles, and their F32 scales.
fn upload_bank(
    session: &Session,
    local: usize,
    rows: usize,
    k: usize,
    seed: u64,
) -> LabResult<Bank> {
    let layout = FragmentWeightLayout {
        rows,
        k,
        format: FragmentWeightFormat::Fp8E4M3,
    };
    let codes = session.allocate(local * rows * k)?;
    let mut scales = Vec::with_capacity(local * rows / 128 * (k / 128));
    for expert in 0..local {
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
        expert_bytes: rows * k + rows / 128 * (k / 128) * 4,
    })
}

struct Banks {
    gate: Bank,
    down: Bank,
}

impl Banks {
    fn new(session: &Session, local: usize) -> LabResult<Self> {
        Ok(Self {
            gate: upload_bank(session, local, 2 * I, H, GATE_SEED)?,
            down: upload_bank(session, local, H, I, DOWN_SEED)?,
        })
    }
}

/// Global expert popularity follows a signed power law, and near-Gaussian
/// per-token noise (a sum of three uniforms) spreads the routes. On one rank
/// this gives about 40 active experts at 32 tokens and a busiest expert near
/// four times the mean at 2048, like MiMo's captured routes; receipts record
/// the resulting skew.
fn skewed_logits(tokens: usize) -> Vec<f32> {
    let popularity = (0..GLOBAL)
        .map(|expert| {
            let u = sample(LOGIT_SALT ^ expert as u64);
            0.8 * u.signum() * u.abs().powf(1.5)
        })
        .collect::<Vec<_>>();
    (0..tokens * GLOBAL)
        .map(|index| {
            let noise = (0..3)
                .map(|draw| sample(LOGIT_SALT ^ 0x8181 ^ (index as u64) << 2 ^ draw))
                .sum::<f32>();
            popularity[index % GLOBAL] + noise
        })
        .collect()
}

fn bf16_values(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    (0..len)
        .map(|index| f32_to_bf16(scale * sample(seed ^ index as u64)))
        .collect()
}

struct Case {
    profile: Profile,
    /// Local experts of one rank of `profile`.
    local: usize,
    tokens: usize,
    /// Allocation and prefill row-launch capacity, independent of live tokens.
    prepared_tokens: usize,
    rank: u32,
    phase: Phase,
    gate: Variant,
    down: Variant,
    gate_fn: Function,
    down_fn: Function,
    /// The profile's route cubin, and TP4's, which holds the router of
    /// every rank count.
    route: Module,
    router: Module,
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

fn qwen_variant(
    profile: Profile,
    phase: Phase,
    projection: Projection,
    tokens: usize,
) -> LabResult<Variant> {
    let bucket = if phase == Phase::Prefill {
        qwen::PREFILL_CAPACITY
    } else {
        tokens as u32
    };
    moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| {
            v.profile == profile
                && v.phase == phase
                && v.projection == projection
                && v.tokens_capacity == bucket
        })
        .ok_or_else(|| {
            format!(
                "missing {} MoE variant for {tokens} tokens",
                profile.as_str()
            )
        })
}

fn gemm_function(session: &Session, v: &Variant, tokens: usize) -> LabResult<Function> {
    let cubin = moe::cubin_specs(kernel_lab::catalogs(), Some(v.profile))
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

/// The route cubin of a Qwen3.8 profile; TP4's is `qwen::ROUTE_CUBIN`.
fn route_module(session: &Session, profile: Profile) -> LabResult<Module> {
    let spec = qwen::route_cubin_spec_for(profile)
        .ok_or_else(|| format!("no route cubin for {}", profile.as_str()))?;
    load_cubin(session, &spec.name)
}

impl Case {
    fn routing_checks<'a>(&'a self, session: &'a Session) -> LabResult<super::replay::Checks<'a>> {
        use super::replay::{Checks, RouteBuffers, RouteGeometry};
        let routes = self.tokens * QK;
        let first = self.rank as usize * self.local;
        let mut checks = Checks::new(session).route(
            RouteGeometry {
                routes,
                experts: first..first + self.local,
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
        // permute_count/prefix:235-314: both tile arrays write only live route tiles.
        if !self.decode() {
            for (name, buffer) in [
                ("tile counts", &self.tile_counts),
                ("tile offsets", &self.tile_offsets),
            ] {
                checks = checks.output(
                    name,
                    buffer,
                    kernel_lab::poison::WriteMap::of(DType::I32)
                        .prefix(routes.div_ceil(256) * self.local)?,
                );
            }
        }
        Ok(checks)
    }

    fn replay_checks<'a>(&'a self, session: &'a Session) -> LabResult<super::replay::Checks<'a>> {
        let routes = self.tokens * QK;
        let first = self.rank as usize * self.local;
        let mut checks = self.routing_checks(session)?;
        // moe/README.md:337: projections write original route indices for
        // local experts only. The grouped engines have no global workspace.
        for (name, buffer, width) in [
            ("gate/up", &self.inter, I),
            ("weighted down", &self.routed, H),
        ] {
            checks = checks.dynamic(
                name,
                buffer,
                move |s| {
                    super::replay::routed_map(
                        &super::replay::ids(s, &self.ids, routes)?,
                        first..first + self.local,
                        width,
                    )
                },
                &[kernel_lab::poison::POISON],
            );
        }
        Ok(checks.output(
            "combined",
            &self.output,
            kernel_lab::poison::WriteMap::of(DType::Bf16).prefix(self.tokens * H)?,
        ))
    }

    fn new(session: &Session, phase: Phase, tokens: usize) -> LabResult<Self> {
        let (profile, _) = topology()?;
        Self::for_profile(session, profile, phase, tokens)
    }

    fn for_profile(
        session: &Session,
        profile: Profile,
        phase: Phase,
        tokens: usize,
    ) -> LabResult<Self> {
        let local = qwen::local_experts(profile).map_err(err)? as usize;
        let gate = qwen_variant(profile, phase, Projection::GateUp, tokens)?;
        let down = qwen_variant(profile, phase, Projection::Down, tokens)?;
        let padded_capacity = match gate.query.grouped.ok_or("missing grouped shape")?.capacity {
            kernel_core::contract::RoutedCapacity::TotalRows(rows) => rows as usize,
            kernel_core::contract::RoutedCapacity::PerExpert(_) => {
                return Err("Qwen3.8 MoE uses total row capacity".into());
            }
        };
        let routes = tokens * QK;
        let tiles = routes.div_ceil(256);
        let logits_host = skewed_logits(tokens);
        let x_host = bf16_values(0x7a11, tokens * H, 1.0);
        let shared_host = bf16_values(0x54a4, tokens * H, 0.05);
        let gate_weight_host = bf16_values(0x6a7e, H, 0.03);
        let route = route_module(session, profile)?;
        let router = route_module(session, model_qwen38_contract::sm120::config::PROFILE)?;
        Ok(Self {
            profile,
            local,
            tokens,
            prepared_tokens: tokens,
            rank: rank(local),
            phase,
            gate_fn: gemm_function(session, &gate, tokens)?,
            down_fn: gemm_function(session, &down, tokens)?,
            gate,
            down,
            route,
            router,
            padded_capacity,
            logits: session.upload_f32(&logits_host)?,
            ids: zeroed(session, routes * 4)?,
            weights: zeroed(session, routes * 4)?,
            counts: zeroed(session, local * 4)?,
            offsets: zeroed(session, (local + 1) * 4)?,
            sorted: zeroed(session, padded_capacity * 4)?,
            padded: zeroed(session, 4)?,
            active: zeroed(session, (local + 1) * 4)?,
            live: upload_bytes(session, &i32_bytes(&[tokens as i32]))?,
            tile_counts: zeroed(session, tiles * local * 4)?,
            tile_offsets: zeroed(session, tiles * local * 4)?,
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

    /// Route grids use the prepared bucket; kernels read live tokens on device.
    fn capacity(&self) -> u32 {
        if self.decode() {
            self.gate.tokens_capacity
        } else {
            self.prepared_tokens as u32
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
        let entry = qwen::route_entry_for(self.profile, operation).map_err(err)?;
        let module = if entry.profile == self.profile {
            &self.route
        } else {
            &self.router
        };
        let function = module.function(&entry.symbol).map_err(err)?;
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
        )?;
        kernel_lab::launched(&entry.symbol);
        Ok(())
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
        )?;
        kernel_lab::launched(&v.symbol);
        Ok(())
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
            local_experts: case.local as u32,
            top_k: QK as u32,
            capacity: case.tokens as u32,
        },
        &selections,
        case.rank as usize,
    )
    .map_err(err)?;
    let block = qwen::ROUTE_BLOCK as usize;
    let local_experts = case.local;
    let mut counts = vec![0_usize; local_experts];
    let mut expected_offsets = Vec::with_capacity(local_experts + 1);
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
    let offsets = download_i32(session, &case.offsets, local_experts + 1)?;
    let sorted = download_i32(session, &case.sorted, padded.min(case.padded_capacity))?;
    let gpu_counts = download_i32(session, &case.counts, local_experts)?;
    let active = download_i32(session, &case.active, local_experts + 1)?;
    let expected_active = (0..local_experts)
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

/// The checked tokens (combine), routes (gate/up and weighted down) and
/// layer tokens (the f64 chain).
struct Checked {
    tokens: Vec<usize>,
    routes: Vec<usize>,
    layer_tokens: Vec<usize>,
}

/// Checked routes: every local route in decode; in prefill the routes of
/// three tokens and the tile-edge routes of the busiest and quietest
/// experts, which cover partial 128-row tiles and their row groups. The
/// layer chain checks the same tokens.
///
/// Lean runs check sampled columns (`checked_columns`), so they take more
/// routes and fewer chain tokens: in decode every token's combine but the
/// chain of sampled tokens (the first, every 16th and the last); in prefill
/// also every 61st row of the padded permutation and the first and last
/// routes of local experts 0, 1 and the last two (both edges of the rank's
/// expert range, the last route before each padding).
fn checked(case: &Case, routing: &Routing, lean: bool) -> Checked {
    if case.decode() {
        let tokens = (0..case.tokens).collect::<Vec<_>>();
        let layer_tokens = if lean {
            kernel_lab::sample_indices(case.tokens, 16, &[])
        } else {
            tokens.clone()
        };
        return Checked {
            tokens,
            routes: routing.local.routes.clone(),
            layer_tokens,
        };
    }
    let tokens = vec![0, case.tokens / 2, case.tokens - 1];
    let mut routes = routing
        .local
        .routes
        .iter()
        .copied()
        .filter(|route| tokens.contains(&(route / QK)))
        .collect::<Vec<_>>();
    let busiest = (0..case.local)
        .max_by_key(|&e| routing.counts[e])
        .unwrap_or(0);
    let quietest = (0..case.local)
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
    if lean {
        for index in kernel_lab::sample_indices(routing.sorted.len(), 61, &[]) {
            if let Ok(route) = usize::try_from(routing.sorted[index]) {
                routes.push(route);
            }
        }
        for expert in [0, 1, case.local - 2, case.local - 1] {
            let first = routing.offsets[expert] as usize;
            let count = routing.counts[expert];
            if count > 0 {
                routes.push(routing.sorted[first] as usize);
                routes.push(routing.sorted[first + count - 1] as usize);
            }
        }
    }
    routes.sort_unstable();
    routes.dedup();
    Checked {
        layer_tokens: tokens.clone(),
        tokens,
        routes,
    }
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

/// Gate/up, weighted down, combine and the layer against the f64 oracles,
/// on every column, or on `checked_columns` in lean runs (the combine keeps
/// every column).
fn check_numerics(
    session: &Session,
    case: &Case,
    routing: &Routing,
    lean: bool,
    receipt: &mut BTreeMap<String, Value>,
) -> LabResult<()> {
    let (inter, routed, output) = case.outputs(session)?;
    let Checked {
        tokens,
        routes,
        layer_tokens,
    } = checked(case, routing, lean);
    let (gate_cols, down_cols) = checked_columns(lean);
    // The gate/up weight rows of the checked columns: gate half, up half.
    let gate_weight_rows = gate_cols
        .iter()
        .copied()
        .chain(gate_cols.iter().map(|&col| I + col))
        .collect::<Vec<_>>();
    let all_gate_rows = (0..2 * I).collect::<Vec<_>>();
    let sampled_gate = gate_cols.len() < I;
    let local_expert = routing
        .local
        .routes
        .iter()
        .zip(&routing.local.experts)
        .map(|(&route, &expert)| (route, expert))
        .collect::<BTreeMap<_, _>>();
    // The layer tokens' local routes are all checked routes.
    let chained = |route: usize| layer_tokens.contains(&(route / QK));
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
    let swiglu = GemmEpilogue {
        residual: None,
        swiglu: Some(SwigluColumns::ConcatenatedHalves),
    };
    let no_epilogue = GemmEpilogue {
        residual: None,
        swiglu: None,
    };
    let gate_shape = |n: usize| GemmShape {
        m: 1,
        n,
        k: H,
        experts: 1,
    };
    let down_shape = GemmShape {
        m: 1,
        n: down_cols.len(),
        k: I,
        experts: 1,
    };
    for (&expert, expert_routes) in &by_expert {
        let gate_up = decoded_rows(GATE_SEED, expert, 2 * I, H, &gate_weight_rows)?;
        // The chain needs every intermediate column.
        let full_gate_up = if sampled_gate && expert_routes.iter().any(|&route| chained(route)) {
            Some(decoded_rows(GATE_SEED, expert, 2 * I, H, &all_gate_rows)?)
        } else {
            None
        };
        let down = decoded_rows(DOWN_SEED, expert, H, I, &down_cols)?;
        for &route in expert_routes {
            let token = route / QK;
            let row = &x[token * H..(token + 1) * H];
            let h = oracle::gemm(
                row,
                &gate_up,
                gate_shape(gate_weight_rows.len()),
                &[],
                swiglu,
            )
            .map_err(err)?;
            let gpu_h = decoded(&inter[route * I..(route + 1) * I]);
            let mut y = oracle::gemm(&gpu_h, &down, down_shape, &[], no_epilogue).map_err(err)?;
            let weight = f64::from(routing.weights[route]);
            for (value, bound) in y.values.iter_mut().zip(&mut y.sum_abs_products) {
                *value *= weight;
                *bound *= weight.abs();
            }
            if chained(route) {
                let exact_h = match &full_gate_up {
                    Some(full) => {
                        oracle::gemm(row, full, gate_shape(2 * I), &[], swiglu)
                            .map_err(err)?
                            .values
                    }
                    None => h.values.clone(),
                };
                let exact = oracle::gemm(&exact_h, &down, down_shape, &[], no_epilogue)
                    .map_err(err)?
                    .values
                    .into_iter()
                    .map(|v| v * routing.oracle_weights[route])
                    .collect::<Vec<_>>();
                chain.insert(route, exact);
            }
            gate_rows.insert(route, h);
            down_rows.insert(route, y);
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
            gate_cols
                .iter()
                .map(|&col| bf16_to_f32(inter[route * I + col])),
        );
        down_actual.extend(
            down_cols
                .iter()
                .map(|&col| bf16_to_f32(routed[route * H + col])),
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
        let layer = layer_tokens.contains(&token);
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
            if layer {
                for (&col, value) in down_cols.iter().zip(&chain[&route]) {
                    exact[col] += value;
                }
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
        if layer {
            let exact = shared_expert_gate_combine(query, &exact, &shared, &[logit])
                .map_err(err)?
                .values;
            layer_reference.extend(down_cols.iter().map(|&col| exact[col]));
            layer_actual.extend(
                down_cols
                    .iter()
                    .map(|&col| f64::from(bf16_to_f32(output[token * H + col]))),
            );
        }
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
    let width = down_cols.len();
    let worst = layer_actual
        .chunks_exact(width)
        .zip(layer_reference.chunks_exact(width))
        .map(|(a, r)| {
            let dot = a.iter().zip(r).map(|(x, y)| x * y).sum::<f64>();
            let an = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let rn = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            dot / (an * rn)
        })
        .fold(1.0, f64::min);
    receipt.insert("layer_cosine_min".into(), json!(worst));
    oracle::check_layer_cosine(&layer_actual, &layer_reference, width)
        .map_err(|failure| format!("layer cosine {} at token {}", failure.actual, failure.index))?;
    receipt.insert("checked_routes".into(), json!(routes.len()));
    receipt.insert("checked_tokens".into(), json!(tokens));
    if lean {
        receipt.insert("gate_up_columns".into(), json!(gate_cols.len()));
        receipt.insert("down_columns".into(), json!(down_cols.len()));
        receipt.insert("layer_tokens".into(), json!(layer_tokens));
    }
    Ok(())
}

/// Graph replay, serialized launches and a repeated run must reproduce the
/// eager PDL outputs bitwise.
fn check_replay(
    session: &Session,
    case: &Case,
    banks: &Banks,
    checks: &mut super::replay::Checks<'_>,
) -> LabResult<()> {
    let eager = case.outputs(session)?;
    if !case.decode() && std::env::var_os("CHOPINFER_MOE_PREFILL_BASELINE").is_some() {
        super::prefill_diagnostics::snapshot(
            &format!(
                "qwen38_r{}_rank{}_t{}",
                kernel_lab::qwen38_ranks()?,
                case.rank,
                case.tokens
            ),
            &[eager.0.clone(), eager.1.clone(), eager.2.clone()],
        )?;
    }
    let graph = session.capture(|_| case.launch(session, banks, true))?;
    for _ in 0..2 {
        checks.phase(super::ReplayPhase::Graph, false, || {
            graph.launch(&session.stream).map_err(err)
        })?;
    }
    if case.outputs(session)? != eager {
        return Err("graph replay differs from eager launches".into());
    }
    checks.phase(super::ReplayPhase::Relaunch, false, || {
        case.launch(session, banks, false)
    })?;
    if case.outputs(session)? != eager {
        return Err("serialized launches differ from PDL launches".into());
    }
    checks.phase(super::ReplayPhase::Pdl, true, || {
        case.launch(session, banks, true)
    })?;
    if case.outputs(session)? != eager {
        return Err("repeated launches are not deterministic".into());
    }
    let mut receipt = BTreeMap::new();
    checks.record(&mut receipt);
    receipt.insert("tokens".into(), json!(case.tokens));
    session.receipt("moe_replay", receipt)?;
    Ok(())
}

/// One case against the oracles; `replay` adds the graph, serialized and
/// repeated launch checks (`check_replay`).
fn run_case(
    session: &Session,
    banks: &Banks,
    phase: Phase,
    tokens: usize,
    replay: bool,
) -> LabResult<()> {
    let lean = kernel_lab::qwen38_lean()?;
    let case = Case::new(session, phase, tokens)?;
    let mut checks = case.replay_checks(session)?;
    checks.phase(super::ReplayPhase::Eager, false, || {
        case.launch(session, banks, true)
    })?;
    let routing = check_routing(session, &case)?;
    let mut receipt = BTreeMap::new();
    receipt.insert("engine".into(), json!("grouped_moe_qwen38"));
    receipt.insert(
        "phase".into(),
        json!(if case.decode() { "decode" } else { "prefill" }),
    );
    receipt.insert("tokens".into(), json!(tokens));
    receipt.insert("rank".into(), json!(case.rank));
    record_ranks(case.profile, &mut receipt);
    if lean {
        receipt.insert("lean".into(), json!(true));
    }
    receipt.insert("routing".into(), routing.stats());
    receipt.insert("gate_up_symbol".into(), json!(case.gate.symbol));
    receipt.insert("down_symbol".into(), json!(case.down.symbol));
    let numerics = check_numerics(session, &case, &routing, lean, &mut receipt);
    let result = if replay {
        let result = numerics.and_then(|()| check_replay(session, &case, banks, &mut checks));
        receipt.insert("graph_pdl_determinism".into(), json!(result.is_ok()));
        result
    } else {
        receipt.insert("graph_pdl_determinism".into(), json!("not run"));
        numerics
    };
    if let Err(error) = &result {
        receipt.insert("error".into(), json!(error));
    }
    let path = session.receipt("moe_qwen38", receipt)?;
    eprintln!("Qwen3.8 MoE {tokens} tokens receipt {}", path.display());
    result
}

/// Full runs check decode at 1, 8 and 32 tokens, each with `check_replay`.
/// Lean runs (TP2) check every decode bucket once, so every
/// `decode_route_t*` and decode gate/up and down entry launches, with
/// sampled columns, and `check_replay` at the largest bucket only.
#[test]
fn qwen38_moe_decode_matches_oracles() {
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let banks = Banks::new(session, topology()?.1)?;
        let lean = kernel_lab::qwen38_lean()?;
        let buckets = if lean {
            qwen::DECODE_BUCKETS.map(|bucket| bucket as usize).to_vec()
        } else {
            vec![1, 8, 32]
        };
        let last = buckets.last().copied();
        let mut failures = super::replay::Failures::default();
        for &tokens in &buckets {
            if selected(tokens) {
                let replay = !lean || Some(tokens) == last;
                failures.check(
                    &format!("decode {tokens}"),
                    run_case(session, &banks, Phase::Decode, tokens, replay),
                );
            }
        }
        failures.finish()
    })
    .expect("Qwen3.8 MoE decode");
}

/// Full runs check prefill at 512, 2048 and 8192 tokens. Lean runs (TP2)
/// check 512 tokens, which launches the router, every permute
/// entry, the t8192 gate/up and down entries and the combine, with sampled
/// columns and more sampled routes.
#[test]
fn qwen38_moe_prefill_matches_oracles() {
    // All local experts' weights and the 8192-token buffers take about 1.5 GB.
    gpu_test(4 * 1024 * 1024 * 1024, |session| {
        let banks = Banks::new(session, topology()?.1)?;
        let token_counts: &[usize] = if kernel_lab::qwen38_lean()? {
            &[512]
        } else {
            &[512, 1024, 2048, 8192]
        };
        let mut failures = super::replay::Failures::default();
        for &tokens in token_counts {
            if selected(tokens) {
                failures.check(
                    &format!("prefill {tokens}"),
                    run_case(session, &banks, Phase::Prefill, tokens, true),
                );
            }
        }
        failures.finish()
    })
    .expect("Qwen3.8 MoE prefill");
}

/// Block scales with wide ratios (review finding): four routes of local
/// expert 0, whose E4M3 codes are all 1.0, through the prefill and decode
/// down projections. The first five 128-row output blocks take the K-block
/// scales [1, 2^-126, 2^-126, 2^-126, 2^-126] (prefill used to rescale 128
/// by 2^126 and overflow), 2^100 alternating with 2^-100, zero blocks, a
/// 2^-126 first block, and 2^-20 between cancelling blocks; the others take
/// checkpoint-like scales. Each route's activation is constant per K block:
/// ones in the first block only, a zero first block, +1, -1, +1, and all
/// ones. Both phases must pass the decoded-weight oracle.
#[test]
fn qwen38_expert_block_scales_promote_like_decode() {
    const TOKENS: usize = 4;
    gpu_test(1 << 30, |session| {
        let (profile, local) = topology()?;
        let (blocks_n, blocks_k) = (H / 128, I / 128);
        let tiny = 2_f32.powi(-126);
        let special: [[f32; 5]; 5] = [
            [1.0, tiny, tiny, tiny, tiny],
            [
                2_f32.powi(100),
                2_f32.powi(-100),
                2_f32.powi(100),
                2_f32.powi(-100),
                2_f32.powi(100),
            ],
            [1.0, 0.0, 1.0, 0.0, 1.0],
            [tiny, 1.0, tiny, 1.0, tiny],
            [1.0, 1.0, 2_f32.powi(-20), 1.0, 1.0],
        ];
        assert_eq!(blocks_k, 5);
        let scales: Vec<u16> = (0..blocks_n * blocks_k)
            .map(|index| {
                let (row_block, k_block) = (index / blocks_k, index % blocks_k);
                f32_to_bf16(special.get(row_block).map_or_else(
                    || 0.0015 * (1.0 + 0.5 * sample(0x5ca1e ^ index as u64)),
                    |row| row[k_block],
                ))
            })
            .collect();
        let codes = vec![0x38_u8; H * I];
        let packed = FragmentWeightLayout {
            rows: H,
            k: I,
            format: FragmentWeightFormat::Fp8E4M3,
        }
        .pack(&codes)
        .map_err(err)?;
        let bank_codes = zeroed(session, local * H * I)?;
        bank_codes
            .copy_from_bytes_at(0, &packed.bytes, &session.stream)
            .map_err(err)?;
        session.stream.synchronize().map_err(err)?;
        let mut bank_scales = vec![0.0_f32; local * blocks_n * blocks_k];
        for (to, &from) in bank_scales.iter_mut().zip(&scales) {
            *to = bf16_to_f32(from);
        }
        let bank_scales = session.upload_f32(&bank_scales)?;
        // Slot 0 of each token routes to local expert 0.
        let per_block: [[f32; 5]; TOKENS] = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, 0.0, 0.0],
            [1.0; 5],
        ];
        let routes = TOKENS * QK;
        let mut inter = vec![0_u16; routes * I];
        for (token, values) in per_block.iter().enumerate() {
            for k in 0..I {
                inter[token * QK * I + k] = f32_to_bf16(values[k / 128]);
            }
        }
        let route_weight = 0.1_f32;
        let block = qwen::ROUTE_BLOCK as usize;
        let mut sorted: Vec<i32> = (0..TOKENS).map(|token| (token * QK) as i32).collect();
        sorted.resize(block, -1);
        let mut offsets = vec![block as i32; local + 1];
        offsets[0] = 0;
        let mut active = vec![0_i32; local + 1];
        active[0] = 1;
        let input = session.upload_bf16(&inter)?;
        let weights = session.upload_f32(&vec![route_weight; routes])?;
        let offsets = upload_bytes(session, &i32_bytes(&offsets))?;
        let sorted = upload_bytes(session, &i32_bytes(&sorted))?;
        let padded = upload_bytes(session, &i32_bytes(&[block as i32]))?;
        let live = upload_bytes(session, &i32_bytes(&[TOKENS as i32]))?;
        let active = upload_bytes(session, &i32_bytes(&active))?;
        let dummy = zeroed(session, 256)?;
        // The oracle: the checkpoint decode of the codes and scales.
        let weight = kernel_core::moe_oracle::fp8_block_checkpoint_weight(&codes, H, I, &scales)
            .map_err(err)?
            .decoded();
        let mut reference = empty_output();
        for token in 0..TOKENS {
            let row = decoded(&inter[token * QK * I..(token * QK + 1) * I]);
            let mut y = oracle::gemm(
                &row,
                &weight,
                GemmShape {
                    m: 1,
                    n: H,
                    k: I,
                    experts: 1,
                },
                &[],
                GemmEpilogue {
                    residual: None,
                    swiglu: None,
                },
            )
            .map_err(err)?;
            for (value, bound) in y.values.iter_mut().zip(&mut y.sum_abs_products) {
                *value *= f64::from(route_weight);
                *bound *= f64::from(route_weight);
            }
            append(&mut reference, &y);
        }
        for phase in [Phase::Prefill, Phase::Decode] {
            let v = qwen_variant(profile, phase, Projection::Down, TOKENS)?;
            let function = gemm_function(session, &v, TOKENS)?;
            let output = zeroed(session, routes * H * 2)?;
            let mut buffers = BTreeMap::new();
            for (handle, buffer) in [
                &input,
                &dummy,
                &dummy,
                &dummy,
                &bank_scales,
                &output,
                &weights,
                &offsets,
                &sorted,
                &padded,
                &live,
                &active,
                &dummy,
                &dummy,
                &dummy,
                &bank_codes,
            ]
            .into_iter()
            .enumerate()
            {
                buffers.insert(handle as u64 + 1, buffer);
            }
            let args = v
                .arguments(HANDLES, v.weight_tensor_map(WEIGHT_HANDLE).map_err(err)?)
                .map_err(err)?;
            launch::launch(
                &session.context,
                &function,
                &session.stream,
                v.plan(counts(&v, TOKENS)).map_err(err)?,
                &args,
                &Bindings { buffers },
            )?;
            session.stream.synchronize().map_err(err)?;
            let routed = session.download_bf16(&output, routes * H)?;
            let actual: Vec<f32> = (0..TOKENS)
                .flat_map(|token| routed[token * QK * H..(token * QK + 1) * H].iter())
                .map(|&bits| bf16_to_f32(bits))
                .collect();
            let mut receipt = BTreeMap::new();
            let result = check_output(&actual, &reference, DType::Bf16, &mut receipt);
            receipt.insert("case".into(), json!("block_scale_ratios"));
            receipt.insert("symbol".into(), json!(v.symbol));
            receipt.insert("review_route".into(), json!(actual[0]));
            record_ranks(profile, &mut receipt);
            session.receipt("moe_qwen38", receipt)?;
            result.map_err(|e| format!("{}: {e}", v.symbol))?;
            eprintln!(
                "{}: block scale ratios match the oracle (review route {})",
                v.symbol, actual[0]
            );
        }
        Ok(())
    })
    .expect("Qwen3.8 expert block scale ratios");
}

/// The shared-expert gate at sigmoid logits -150, -90, -88 and 3 on shared
/// outputs up to 2^100 (review finding): `1 / (1 + exp(-x))` overflowed
/// below about -88 and dropped products that BF16 represents. Each token's
/// gate logit is its first input element (the gate weight is one there and
/// zero elsewhere), and its routes belong to another rank, so the output is
/// the gated shared row alone.
#[test]
fn qwen38_shared_gate_keeps_sigmoid_tails() {
    const LOGITS: [f32; 4] = [-150.0, -90.0, -88.0, 3.0];
    let tokens = LOGITS.len();
    gpu_test(64 << 20, |session| {
        let (profile, local) = topology()?;
        let route = route_module(session, profile)?;
        let entry = qwen::route_entry_for(profile, "combine").map_err(err)?;
        let function = route.function(&entry.symbol).map_err(err)?;
        let rank = rank(local);
        let other = ((rank as usize + 1) % (GLOBAL / local) * local) as i32;
        let amplitudes = [
            2_f32.powi(40),
            -(2_f32.powi(60)),
            3.0 * 2_f32.powi(20),
            2_f32.powi(100),
            1.0,
        ];
        let shared: Vec<u16> = (0..tokens * H)
            .map(|i| f32_to_bf16(amplitudes[i % amplitudes.len()]))
            .collect();
        let mut x = vec![0_u16; tokens * H];
        for (token, &logit) in LOGITS.iter().enumerate() {
            x[token * H] = f32_to_bf16(logit);
        }
        let mut gate_weight = vec![0_u16; H];
        gate_weight[0] = f32_to_bf16(1.0);
        let buffers = [
            zeroed(session, tokens * QK * H * 2)?,
            upload_bytes(session, &i32_bytes(&vec![other; tokens * QK]))?,
            session.upload_bf16(&shared)?,
            session.upload_bf16(&x)?,
            session.upload_bf16(&gate_weight)?,
            zeroed(session, tokens * H * 2)?,
            upload_bytes(session, &i32_bytes(&[tokens as i32]))?,
        ];
        let handles = (1..=buffers.len() as u64).collect::<Vec<_>>();
        launch::launch(
            &session.context,
            &function,
            &session.stream,
            entry.plan(tokens as u32, tokens as u32).map_err(err)?,
            &entry.arguments(&handles, Some(rank)).map_err(err)?,
            &Bindings {
                buffers: handles.iter().copied().zip(buffers.iter()).collect(),
            },
        )?;
        session.stream.synchronize().map_err(err)?;
        let actual: Vec<f32> = session
            .download_bf16(&buffers[5], tokens * H)?
            .iter()
            .map(|&bits| bf16_to_f32(bits))
            .collect();
        let logits: Vec<f64> = LOGITS.iter().map(|&l| f64::from(l)).collect();
        let reference = shared_expert_gate_combine(
            MoeQuery::SharedExpertGateCombine {
                width: H as u32,
                capacity: tokens as u32,
            },
            &vec![0.0; tokens * H],
            &decoded(&shared),
            &logits,
        )
        .map_err(err)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("shared_gate_sigmoid_tails"));
        record_ranks(profile, &mut receipt);
        let result = check_output(&actual, &reference, DType::Bf16, &mut receipt);
        session.receipt("moe_qwen38", receipt)?;
        result.map_err(|e| format!("shared gate sigmoid tails: {e}"))
    })
    .expect("Qwen3.8 shared gate sigmoid tails");
}

/// G4: captured graphs of the full layer and of each stage, with skewed
/// routes, against the byte roofline of the active local experts (decode)
/// and the BF16 MMA roofline of the routed FLOPs (prefill).
#[test]
fn qwen38_moe_benchmarks() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen3.8 MoE benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let banks = Banks::new(session, topology()?.1)?;
        let (dram_peak, bf16_peak, _, source) = peak_rates(session);
        for (phase, tokens) in [
            (Phase::Decode, 1),
            (Phase::Decode, 2),
            (Phase::Decode, 4),
            (Phase::Decode, 8),
            (Phase::Decode, 16),
            (Phase::Decode, 32),
            (Phase::Prefill, 512),
            (Phase::Prefill, 1024),
            (Phase::Prefill, 2048),
            (Phase::Prefill, 8192),
        ] {
            if !selected(tokens) {
                continue;
            }
            let case = Case::new(session, phase, tokens)?;
            case.launch(session, &banks, true)?;
            let routing = check_routing(session, &case)?;
            let full = session.capture(|_| case.launch(session, &banks, true))?;
            let route = session.capture(|_| case.launch_route(session, true))?;
            let gate = session.capture(|_| {
                case.launch_projection(session, &banks, Projection::GateUp, true)
            })?;
            let down = session.capture(|_| {
                case.launch_projection(session, &banks, Projection::Down, true)
            })?;
            let combine = session.capture(|_| case.launch_combine(session, true))?;
            let mut us = BTreeMap::new();
            for (name, graph) in [
                ("full", &full),
                ("route", &route),
                ("gate_up", &gate),
                ("down", &down),
                ("combine", &combine),
            ] {
                let timing = time(session, 3, 9, |stream| graph.launch(stream).map_err(err))?;
                us.insert(name, f64::from(timing.median_ms) * 1000.0);
            }
            let active = routing.counts.iter().filter(|&&c| c > 0).count();
            let local_routes = routing.counts.iter().sum::<usize>();
            let gate_bytes = active * banks.gate.expert_bytes;
            let down_bytes = active * banks.down.expert_bytes;
            let floor_us = |bytes: usize| bytes as f64 / dram_peak / 1000.0;
            let flops = |n: usize, k: usize| 2.0 * local_routes as f64 * n as f64 * k as f64;
            let mma_floor_us = |flops: f64| flops / bf16_peak / 1e6;
            let percent = |floor: f64, measured: f64| 100.0 * floor / measured;
            let gate_floor = floor_us(gate_bytes).max(mma_floor_us(flops(2 * I, H)));
            let down_floor = floor_us(down_bytes).max(mma_floor_us(flops(H, I)));
            let mut receipt = BTreeMap::new();
            receipt.insert("engine".into(), json!("grouped_moe_qwen38"));
            if phase == Phase::Prefill
                && std::env::var("CHOPINFER_MOE_PREFILL_CONTEXT").as_deref() == Ok("1") {
                let mut contexts = BTreeMap::new();
                for context in ["cold", "after_route", "after_down", "after_gate"] {
                    let us = super::prefill_diagnostics::context_time(session, || match context {
                        "after_route" => case.launch_route(session, true),
                        "after_down" => case.launch_projection(session, &banks, Projection::Down, true),
                        "after_gate" => case.launch_projection(session, &banks, Projection::GateUp, true),
                        _ => Ok(()),
                    }, || case.launch_projection(session, &banks, Projection::GateUp, true))?;
                    contexts.insert(context, us);
                }
                let serialized = session.capture(|_| case.launch(session, &banks, false))?;
                let timing = time(session, 2, 5, |stream| serialized.launch(stream).map_err(err))?;
                receipt.insert("serialized_full_us".into(), json!(timing.median_ms * 1000.0));
                receipt.insert("gate_up_context_us".into(), json!(contexts));
            }
            receipt.insert(
                "phase".into(),
                json!(if case.decode() { "decode" } else { "prefill" }),
            );
            receipt.insert("tokens".into(), json!(tokens));
            receipt.insert("rank".into(), json!(case.rank));
            record_ranks(case.profile, &mut receipt);
            receipt.insert("routing".into(), routing.stats());
            receipt.insert("expert_rows".into(), json!(routing.counts));
            receipt.insert("gate_up_schedule".into(), super::prefill_diagnostics::schedule(&routing.counts, I / 128));
            receipt.insert("down_schedule".into(), super::prefill_diagnostics::schedule(&routing.counts, H / 256));
            receipt.insert("median_us".into(), json!(us));
            receipt.insert("peak_source".into(), json!(source));
            receipt.insert("dram_gb_per_second".into(), json!(dram_peak));
            receipt.insert("bf16_tflops".into(), json!(bf16_peak));
            receipt.insert("gate_up_weight_bytes".into(), json!(gate_bytes));
            receipt.insert("down_weight_bytes".into(), json!(down_bytes));
            receipt.insert(
                "gate_up_roofline_percent".into(),
                json!(percent(gate_floor, us["gate_up"])),
            );
            receipt.insert(
                "down_roofline_percent".into(),
                json!(percent(down_floor, us["down"])),
            );
            receipt.insert(
                "gemm_roofline_percent".into(),
                json!(percent(gate_floor + down_floor, us["gate_up"] + us["down"])),
            );
            receipt.insert(
                "roofline".into(),
                json!("max(active expert code and F32 scale bytes at the DRAM read peak, routed FLOPs at the BF16 MMA peak)"),
            );
            let path = session.receipt("moe_qwen38_bench", receipt.clone())?;
            eprintln!(
                "Qwen3.8 MoE {} t{tokens}: full {:.2} us, route {:.2}, gate/up {:.2} ({:.1}%), down {:.2} ({:.1}%), combine {:.2}; {} experts; {}",
                if case.decode() { "decode" } else { "prefill" },
                us["full"],
                us["route"],
                us["gate_up"],
                percent(gate_floor, us["gate_up"]),
                us["down"],
                percent(down_floor, us["down"]),
                us["combine"],
                active,
                path.display()
            );
        }
        Ok(())
    })
    .expect("Qwen3.8 MoE benchmarks");
}

/// Bitwise checks and timing of the Qwen decode path's per-warp rings at
/// small local expert counts.
#[test]
fn qwen38_moe_smalle_bitwise_and_bench() {
    if std::env::var("CHOPINFER_MOE_SMALLE").as_deref() != Ok("1") {
        kernel_lab::skip("qwen38_moe_smalle_bitwise_and_bench: set CHOPINFER_MOE_SMALLE=1")
            .unwrap();
        return;
    }
    assert!(
        kernel_lab::selected_ordinals()
            .expect("GPU selection")
            .is_some()
    );
    gpu_test(8 << 30, |session| {
        let mut failures = super::replay::Failures::default();
        if session.metadata["measurement_label"] != "exclusive-device" {
            return Err("Qwen few-expert measurements require an exclusive assigned GPU".into());
        }
        let banks = Banks::new(session, topology()?.1)?;
        for tokens in [2, 8, 16, 32] {
            let mut case = Case::new(session, Phase::Decode, tokens)?;
            let mut expert_counts = vec![8, (tokens * 8).min(case.local)];
            if tokens >= 8 {
                expert_counts.extend([21, 40]);
            }
            expert_counts.sort_unstable();
            expert_counts.dedup();
            let mut routes: Vec<Option<usize>> = expert_counts.into_iter().map(Some).collect();
            let real = std::env::var_os("CHOPINFER_MOE_QWEN_LOGITS");
            if real.is_some() {
                routes.push(None);
            }
            for requested in routes {
                let result = (|| {
                    let route_name;
                    if let Some(experts) = requested {
                        route_name = format!("e{experts}");
                        case.logits_host.fill(-8.0);
                        for token in 0..tokens {
                            // Eight local and two remote experts per token make an
                            // eight-expert local route meaningful with Qwen's top-10.
                            for k in 0..8 {
                                let expert = ((token * 8 + k) % experts * 73) % case.local;
                                case.logits_host
                                    [token * GLOBAL + case.rank as usize * case.local + expert] =
                                    8.0;
                            }
                            let remote =
                                ((case.rank as usize + 1) % (GLOBAL / case.local)) * case.local;
                            case.logits_host[token * GLOBAL + remote] = 8.0;
                            case.logits_host[token * GLOBAL + remote + 1] = 8.0;
                        }
                    } else {
                        // An existing 49-row prefill dump supplies real token routes.
                        // This is not a capture of independent C16 decode requests.
                        route_name = "real-prefill".into();
                        let bytes =
                            fs::read(real.as_ref().expect("real route path")).map_err(err)?;
                        if bytes.len() < tokens * GLOBAL * 4 || bytes.len() % (GLOBAL * 4) != 0 {
                            return Err("Qwen router dump has an invalid shape".into());
                        }
                        for (out, word) in case.logits_host.iter_mut().zip(bytes.chunks_exact(4)) {
                            *out = f32::from_le_bytes(word.try_into().unwrap());
                            if !out.is_finite() {
                                return Err("Qwen router dump contains a nonfinite logit".into());
                            }
                        }
                    }
                    case.logits
                        .copy_from_bytes(&f32_bytes(&case.logits_host), &session.stream)
                        .map_err(err)?;
                    let mut checks = case.replay_checks(session)?;
                    checks.phase(super::ReplayPhase::Eager, false, || {
                        case.launch(session, &banks, true)
                    })?;
                    let routing = check_routing(session, &case)?;
                    let experts = routing.counts.iter().filter(|&&n| n > 0).count();
                    if requested.is_some_and(|expected| experts != expected) {
                        return Err("Qwen fixture lost a requested expert".into());
                    }
                    check_replay(session, &case, &banks, &mut checks)?;
                    let (gate, down, combined) = case.outputs(session)?;
                    snapshot(
                        &format!("qwen-t{tokens}-{route_name}"),
                        &[&gate, &down, &combined],
                    )?;
                    let mut timings = BTreeMap::new();
                    if tokens >= 16 {
                        for projection in [Projection::GateUp, Projection::Down] {
                            let graph = session.capture(|_| {
                                case.launch_projection(session, &banks, projection, true)
                            })?;
                            let timing =
                                time(session, 2, 7, |stream| graph.launch(stream).map_err(err))?;
                            timings.insert(format!("{projection:?}"), timing.median_ms * 1000.0);
                        }
                    }
                    session.receipt(
                        "moe_smalle_qwen",
                        BTreeMap::from([
                            ("tokens".into(), json!(tokens)),
                            ("route".into(), json!(route_name)),
                            ("experts".into(), json!(experts)),
                            ("routing".into(), routing.stats()),
                            ("median_us".into(), json!(timings)),
                            (
                                "gate_weight_bytes".into(),
                                json!(experts * banks.gate.expert_bytes),
                            ),
                            (
                                "down_weight_bytes".into(),
                                json!(experts * banks.down.expert_bytes),
                            ),
                            (
                                "historical_snapshot_compared".into(),
                                json!(
                                    std::env::var("CHOPINFER_MOE_SMALLE_REPLAY_ONLY").as_deref()
                                        != Ok("1")
                                ),
                            ),
                            ("bitwise_replay".into(), json!(true)),
                            (
                                "cache_policy".into(),
                                json!("selected L2 eviction before each event sample"),
                            ),
                            (
                                "schedule".into(),
                                json!("per-warp rings and CTA-local split reduction"),
                            ),
                        ]),
                    )?;
                    eprintln!("SMALLE Qwen t{tokens} E{experts}");

                    Ok(())
                })();
                failures.check(
                    &format!("few experts t{tokens} requested {requested:?}"),
                    result,
                );
            }
        }
        failures.finish()
    })
    .expect("Qwen few-expert bitwise baseline and measurements");
}

#[path = "moe_qwen_boundaries.rs"]
mod boundaries;

/// Baselines are immutable files, explicitly recorded before kernel changes.
/// Candidate runs fail on a missing file or any changed BF16 output bit.
fn snapshot(name: &str, values: &[&[u16]]) -> LabResult<()> {
    use std::io::Write;
    if std::env::var("CHOPINFER_MOE_SMALLE_REPLAY_ONLY").as_deref() == Ok("1") {
        if std::env::var_os("CHOPINFER_MOE_SMALLE_GOLDEN").is_some()
            || std::env::var("CHOPINFER_MOE_SMALLE_RECORD").as_deref() == Ok("1")
        {
            return Err("replay-only mode cannot read or record historical snapshots".into());
        }
        return Ok(());
    }
    let root = std::env::var_os("CHOPINFER_MOE_SMALLE_GOLDEN").ok_or("missing golden directory")?;
    let path = PathBuf::from(root).join(format!("{name}.bin"));
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend(value.iter().flat_map(|v| v.to_le_bytes()));
    }
    if std::env::var("CHOPINFER_MOE_SMALLE_RECORD").as_deref() == Ok("1") {
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(err)?
            .write_all(&bytes)
            .map_err(err)?;
    } else {
        let expected = fs::read(&path).map_err(err)?;
        if bytes != expected {
            return Err(format!(
                "{} differs from the current-kernel baseline at byte {:?}",
                path.display(),
                bytes.iter().zip(&expected).position(|(a, b)| a != b)
            ));
        }
    }
    Ok(())
}
