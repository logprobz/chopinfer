//! Prototype: the Qwen3.8 decode MoE sublayer with the shared expert on a
//! forked graph branch.
//!
//! Serving runs one rank's MoE sublayer as a single PDL chain: router GEMM,
//! route (one CTA), routed gate/up, routed down, shared gate/up, shared down,
//! combine. The shared expert only needs the sublayer input, so it can run
//! beside the router, the one-CTA route and the routed experts. This target
//! captures the same kernels, with the same buffers, in three arrangements
//! over `CHOPINFER_QWEN_TAIL_LAYERS` layers of distinct weights (default 4):
//!
//! - `serial`: the serving order, every launch PDL;
//! - `fork`: the shared expert on a second captured stream that forks at the
//!   sublayer input and joins before the combine (the launches after the
//!   fork and the join are not PDL);
//! - `shared_first`: one chain with the shared expert first;
//!
//! plus `routed_only` (no shared expert), the lower bound for `fork`.
//!
//! Correctness: the combine output of every arrangement must equal the
//! serial one bit for bit. Every compared launch starts with poisoned
//! destinations, so a missing join cannot reuse a previous shared output. Timings (with
//! `CHOPINFER_KERNEL_BENCH=1`) replay each graph after an L2 flush and
//! record a `qwen38_moe_tail` receipt per token count
//! (`CHOPINFER_QWEN_TAIL_TOKENS`, default 1,16,32). Router logits come
//! from a prepared buffer so every arrangement routes identically;
//! `CHOPINFER_QWEN_TAIL_ROUTING=clustered` biases 96 global experts to
//! model a batch of similar prompts, `uniform` (default) routes i.i.d.
//!
//! `CHOPINFER_QWEN38_RANKS` (2 or 4, default 4) selects the topology:
//! rank 1's routed experts (256 or 128), the profile's route cubin and
//! its shared expert entries (`gemm_{tag}_own_sm120`, the shared columns
//! padded to 128). The router GEMM is TP4's at every rank count.
//!
//! Lean runs (`kernel_lab::qwen38_lean`, TP2) default to 2 layers
//! and 1,32 tokens, and every layer and token count reads one device copy
//! of the routed expert banks (TP2's are about 1.26 GB), so the bitwise
//! checks stay but the setup is short. Their receipts carry `"lean": true`.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{ArgumentList, DType, LaunchPlan, RuntimeCounts};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::gemm::fused::{Buffers, Entry, Kind, catalog};
use kernels_sm120::moe::{self, Phase, Projection, Variant};
use model_qwen38_contract::sm120::moe as qwen;
use runtime_cuda::{DeviceBuffer, Event, Function, Module, Stream};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase as ReplayPhase, WriteMap};

const HANDLES: [u64; 15] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const WEIGHT_HANDLE: u64 = 16;
const GLOBAL: usize = qwen::GLOBAL_EXPERTS as usize;
const QK: usize = qwen::TOP_K as usize;
const H: usize = qwen::HIDDEN as usize;
const I: usize = qwen::INTERMEDIATE as usize;
const LOGIT_SALT: u64 = 0x10_6175;

// Dense GEMM handles, as in qwen38_gemm.rs.
const ACTIVATION: u64 = 3;
const PARTIAL: u64 = 5;
const SEMAPHORES: u64 = 6;
const WEIGHT: u64 = 10;
const OUTPUT: u64 = 20;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
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

/// E4M3 codes with exponent fields 4 to 10, so no code is NaN.
fn fp8_code(byte: u8) -> u8 {
    (byte & 0x87) | ((4 + (byte >> 3 & 15) % 7) << 3)
}

fn bf16_values(seed: u64, len: usize, scale: f32) -> Vec<u16> {
    (0..len)
        .map(|index| f32_to_bf16(scale * sample(seed ^ index as u64)))
        .collect()
}

fn bytes_of(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn upload(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len().max(4))?;
    if !data.is_empty() {
        buffer.copy_from_bytes(data, &session.stream).map_err(err)?;
    }
    Ok(buffer)
}

/// The Qwen3.8 profile under test and the local experts of one of its ranks.
fn topology() -> LabResult<(Profile, usize)> {
    let profile = kernel_lab::qwen38_profile()?;
    let local = qwen::local_experts(profile).map_err(err)?;
    Ok((profile, local as usize))
}

fn zeroed(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.max(4))?;
    buffer
        .zero_at_async(0, bytes.max(4), &session.stream)
        .map_err(err)?;
    Ok(buffer)
}

struct Bank {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
}

/// All local experts of one projection in fragment-major E4M3 tiles, and
/// their F32 scales, on the host.
struct HostBank {
    codes: Vec<u8>,
    scales: Vec<f32>,
}

impl HostBank {
    fn new(local: usize, rows: usize, k: usize, seed: u64) -> LabResult<Self> {
        let layout = FragmentWeightLayout {
            rows,
            k,
            format: FragmentWeightFormat::Fp8E4M3,
        };
        let mut codes = Vec::with_capacity(local * rows * k);
        let mut scales = Vec::with_capacity(local * rows / 128 * (k / 128));
        for expert in 0..local {
            let raw: Vec<u8> = stream_bytes(seed ^ (expert as u64) << 20, rows * k)
                .into_iter()
                .map(fp8_code)
                .collect();
            codes.extend(layout.pack(&raw).map_err(err)?.bytes);
            scales.extend((0..rows / 128 * (k / 128)).map(|block| {
                let u = sample(seed ^ 0x5ca1e ^ (expert as u64) << 24 ^ block as u64);
                0.0015 * (1.0 + 0.5 * u)
            }));
        }
        Ok(Self { codes, scales })
    }

    /// A device copy at its own addresses: every layer reads its own bytes
    /// from DRAM, as serving's layers do, though the values repeat.
    fn upload(&self, session: &Session) -> LabResult<Bank> {
        Ok(Bank {
            codes: upload(session, &self.codes)?,
            scales: session.upload_f32(&self.scales)?,
        })
    }
}

fn qwen_variant(profile: Profile, projection: Projection, tokens: usize) -> LabResult<Variant> {
    moe::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|v| {
            v.profile == profile
                && v.phase == Phase::Decode
                && v.projection == projection
                && v.tokens_capacity == tokens as u32
        })
        .ok_or_else(|| {
            format!(
                "missing {} decode MoE variant for {tokens} tokens",
                profile.as_str()
            )
        })
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

fn moe_function(session: &Session, v: &Variant, tokens: usize) -> LabResult<Function> {
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

/// Router logits: i.i.d., or with 96 popular global experts (24 per rank on
/// average) raised so a batch of similar prompts shares most choices.
fn logits(tokens: usize, clustered: bool, layer: usize) -> Vec<f32> {
    let salt = LOGIT_SALT ^ (layer as u64) << 40;
    (0..tokens * GLOBAL)
        .map(|i| {
            let expert = i % GLOBAL;
            let hot = clustered && (expert * 2_654_435_761 + layer * 97) % 512 < 96;
            sample(salt ^ i as u64) + if hot { 2.5 } else { 0.0 }
        })
        .collect()
}

/// One layer's routed MoE destinations. Each graph call owns its buffers.
struct Routed {
    profile: Profile,
    /// Local experts of one rank of `profile`.
    local: usize,
    tokens: usize,
    gate: Variant,
    down: Variant,
    gate_fn: Function,
    down_fn: Function,
    route: Module,
    parallel_route: Module,
    route_ticket: DeviceBuffer,
    ids: DeviceBuffer,
    weights: DeviceBuffer,
    counts: DeviceBuffer,
    offsets: DeviceBuffer,
    sorted: DeviceBuffer,
    padded: DeviceBuffer,
    active: DeviceBuffer,
    live: DeviceBuffer,
    x: DeviceBuffer,
    inter: DeviceBuffer,
    routed: DeviceBuffer,
    shared: DeviceBuffer,
    gate_weight: DeviceBuffer,
    output: DeviceBuffer,
    dummy: DeviceBuffer,
}

impl Routed {
    fn new(session: &Session, tokens: usize) -> LabResult<Self> {
        let (profile, local) = topology()?;
        let gate = qwen_variant(profile, Projection::GateUp, tokens)?;
        let down = qwen_variant(profile, Projection::Down, tokens)?;
        let route_cubin = qwen::route_cubin_spec_for(profile)
            .ok_or_else(|| format!("no route cubin for {}", profile.as_str()))?
            .name;
        let padded_capacity = match gate.query.grouped.ok_or("missing grouped shape")?.capacity {
            kernel_core::contract::RoutedCapacity::TotalRows(rows) => rows as usize,
            kernel_core::contract::RoutedCapacity::PerExpert(_) => {
                return Err("Qwen3.8 MoE uses total row capacity".into());
            }
        };
        let routes = tokens * QK;
        Ok(Self {
            profile,
            local,
            tokens,
            gate_fn: moe_function(session, &gate, tokens)?,
            down_fn: moe_function(session, &down, tokens)?,
            gate,
            down,
            route: load_cubin(session, &route_cubin)?,
            parallel_route: load_cubin(
                session,
                &qwen::parallel_route_cubin_spec_for(profile)
                    .ok_or("parallel route profile")?
                    .name,
            )?,
            route_ticket: zeroed(session, 4)?,
            ids: zeroed(session, routes * 4)?,
            weights: zeroed(session, routes * 4)?,
            counts: zeroed(session, local * 4)?,
            offsets: zeroed(session, (local + 1) * 4)?,
            sorted: zeroed(session, padded_capacity * 4)?,
            padded: zeroed(session, 4)?,
            active: zeroed(session, (local + 1) * 4)?,
            live: upload(session, &(tokens as i32).to_ne_bytes())?,
            x: session.upload_bf16(&bf16_values(0x7a11, tokens * H, 1.0))?,
            inter: zeroed(session, routes * I * 2)?,
            routed: zeroed(session, routes * H * 2)?,
            shared: zeroed(session, tokens * H * 2)?,
            gate_weight: session.upload_bf16(&bf16_values(0x6a7e, H, 0.03))?,
            output: zeroed(session, tokens * H * 2)?,
            dummy: zeroed(session, 256)?,
        })
    }

    fn entry(
        &self,
        session: &Session,
        stream: &Stream,
        operation: &str,
        buffers: &[&DeviceBuffer],
        filtered: bool,
        pdl: bool,
    ) -> LabResult<()> {
        let entry = qwen::route_entry_for(self.profile, operation).map_err(err)?;
        let module = if operation.starts_with("decode_route_parallel_") {
            &self.parallel_route
        } else {
            &self.route
        };
        let function = module.function(&entry.symbol).map_err(err)?;
        let mut plan = entry
            .plan(self.gate.tokens_capacity, self.tokens as u32)
            .map_err(err)?;
        plan.pdl = pdl;
        let handles = (1..=buffers.len() as u64).collect::<Vec<_>>();
        let args = entry
            .arguments(&handles, filtered.then_some(1))
            .map_err(err)?;
        let bindings = Bindings {
            buffers: handles
                .iter()
                .copied()
                .zip(buffers.iter().copied())
                .collect(),
        };
        launch::launch(&session.context, &function, stream, plan, &args, &bindings)
    }

    fn route(
        &self,
        session: &Session,
        stream: &Stream,
        logits: &DeviceBuffer,
        pdl: bool,
    ) -> LabResult<()> {
        let parallel = qwen::use_parallel_route(self.gate.tokens_capacity);
        let operation = if parallel {
            format!("decode_route_parallel_t{}", self.gate.tokens_capacity)
        } else {
            format!("decode_route_t{}", self.gate.tokens_capacity)
        };
        let mut buffers = vec![
            logits,
            &self.ids,
            &self.weights,
            &self.counts,
            &self.offsets,
            &self.sorted,
            &self.padded,
            &self.live,
            &self.active,
        ];
        if parallel {
            buffers.push(&self.route_ticket);
        }
        self.entry(session, stream, &operation, &buffers, true, pdl)
    }

    fn projection(
        &self,
        session: &Session,
        stream: &Stream,
        bank: &Bank,
        projection: Projection,
        pdl: bool,
    ) -> LabResult<()> {
        let (v, function, input, output) = match projection {
            Projection::GateUp => (&self.gate, &self.gate_fn, &self.x, &self.inter),
            Projection::Down => (&self.down, &self.down_fn, &self.inter, &self.routed),
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
            stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    }

    fn combine(&self, session: &Session, stream: &Stream, pdl: bool) -> LabResult<()> {
        self.entry(
            session,
            stream,
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

    fn active_experts(&self, session: &Session) -> LabResult<usize> {
        session.stream.synchronize().map_err(err)?;
        let mut bytes = vec![0; self.local * 4];
        self.counts
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(err)?;
        Ok(bytes
            .chunks_exact(4)
            .filter(|c| i32::from_ne_bytes((*c).try_into().unwrap()) > 0)
            .count())
    }
}

/// One cluster decode GEMM of `gemm::fused` with its own weights; the
/// activation and first output can be bound to outside buffers.
struct Dense {
    entry: Entry,
    function: Function,
    plan: LaunchPlan,
    args: ArgumentList,
    buffers: BTreeMap<u64, DeviceBuffer>,
    weight_bytes: usize,
}

impl Dense {
    fn new(session: &Session, module: &Module, symbol: &str, m: u32, seed: u64) -> LabResult<Self> {
        let entry = catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|entry| entry.symbol == symbol)
            .ok_or_else(|| format!("missing Qwen GEMM entry {symbol}"))?;
        let q = entry.query;
        let (n, k, rows) = (q.n as usize, q.k as usize, m as usize);
        let out_n = if q.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        };
        let mut buffers = BTreeMap::new();
        let mut first = 0;
        let mut weight_bytes = 0;
        for (s, &segment) in entry.segments.iter().enumerate() {
            let segment = segment as usize;
            let raw = bf16_values(seed ^ (s as u64) << 32, segment * k, 0.02);
            let packed = FragmentWeightLayout {
                rows: segment,
                k,
                format: FragmentWeightFormat::Bf16,
            }
            .pack(&bytes_of(&raw))
            .map_err(err)?
            .bytes;
            weight_bytes += packed.len();
            buffers.insert(WEIGHT + s as u64, upload(session, &packed)?);
            first += segment;
        }
        debug_assert_eq!(first, n);
        let plan = entry.plan(m).map_err(err)?;
        let element = if q.output == DType::F32 { 4 } else { 2 };
        let columns: Vec<usize> = match entry.kind {
            Kind::ClusterDecode { .. } if out_n == n => {
                entry.segments.iter().map(|&r| r as usize).collect()
            }
            _ => vec![out_n],
        };
        let mut output_strides = [0; 4];
        for (s, &c) in columns.iter().enumerate() {
            buffers.insert(OUTPUT + s as u64, zeroed(session, rows * c * element)?);
            output_strides[s] = c as u32;
        }
        buffers.insert(
            ACTIVATION,
            session.upload_bf16(&bf16_values(seed ^ 0xac7, rows * k, 1.0))?,
        );
        buffers.insert(
            PARTIAL,
            session.allocate((plan.workspace_bytes as usize).max(4))?,
        );
        buffers.insert(
            SEMAPHORES,
            zeroed(session, (entry.semaphore_words() as usize).max(1) * 4)?,
        );
        let handles = |base: u64| std::array::from_fn(|s| base + s as u64);
        let args = entry
            .arguments(
                m,
                Buffers {
                    weights: handles(WEIGHT),
                    activation: ACTIVATION,
                    outputs: handles(OUTPUT),
                    output_strides,
                    partial: PARTIAL,
                    semaphores: SEMAPHORES,
                },
            )
            .map_err(err)?;
        let function = module.function(&entry.symbol).map_err(err)?;
        if plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(err)?;
        }
        Ok(Self {
            entry,
            function,
            plan,
            args,
            buffers,
            weight_bytes,
        })
    }

    fn add_checks<'a>(
        &'a self,
        mut checks: replay::Checks<'a>,
        output: Option<&'a DeviceBuffer>,
    ) -> LabResult<replay::Checks<'a>> {
        let element = if self.entry.query.output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        for (&id, buffer) in self.buffers.range(OUTPUT..OUTPUT + 4) {
            let buffer = if id == OUTPUT {
                output.unwrap_or(buffer)
            } else {
                buffer
            };
            checks = checks.output(
                &self.entry.symbol,
                buffer,
                WriteMap::of(element).byte_prefix(buffer.byte_len())?,
            );
        }
        let bytes = self.plan.workspace_bytes as usize;
        if bytes != 0 {
            checks = checks.scratch_map(
                &format!("{} partials", self.entry.symbol),
                &self.buffers[&PARTIAL],
                0,
                move |_| WriteMap::of(DType::F32).byte_prefix(bytes),
            );
        }
        Ok(checks)
    }

    fn launch(
        &self,
        session: &Session,
        stream: &Stream,
        activation: Option<&DeviceBuffer>,
        output: Option<&DeviceBuffer>,
        pdl: bool,
    ) -> LabResult<()> {
        let mut buffers: BTreeMap<u64, &DeviceBuffer> =
            self.buffers.iter().map(|(&id, b)| (id, b)).collect();
        if let Some(buffer) = activation {
            buffers.insert(ACTIVATION, buffer);
        }
        if let Some(buffer) = output {
            buffers.insert(OUTPUT, buffer);
        }
        let mut plan = self.plan;
        plan.pdl = pdl;
        launch::launch(
            &session.context,
            &self.function,
            stream,
            plan,
            &self.args,
            &Bindings { buffers },
        )
    }
}

struct Layer {
    routed: Routed,
    logits: DeviceBuffer,
    gate: Rc<Bank>,
    down: Rc<Bank>,
    router: Dense,
    shared_gate_up: Dense,
    shared_down: Dense,
    shared_inter: DeviceBuffer,
    fork: Event,
    join: Event,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Arrangement {
    Serial,
    Fork,
    SharedFirst,
    RoutedOnly,
}

impl Arrangement {
    fn name(self) -> &'static str {
        match self {
            Arrangement::Serial => "serial",
            Arrangement::Fork => "fork",
            Arrangement::SharedFirst => "shared_first",
            Arrangement::RoutedOnly => "routed_only",
        }
    }
}

/// The routed expert banks of each layer: a fresh device copy per layer
/// (full runs), or one copy that every layer and token count reads (lean).
enum Banks<'a> {
    Upload(&'a (HostBank, HostBank)),
    Shared(&'a (Rc<Bank>, Rc<Bank>)),
}

impl Banks<'_> {
    fn layer(&self, session: &Session) -> LabResult<(Rc<Bank>, Rc<Bank>)> {
        match self {
            Banks::Upload((gate, down)) => Ok((
                Rc::new(gate.upload(session)?),
                Rc::new(down.upload(session)?),
            )),
            Banks::Shared((gate, down)) => Ok((Rc::clone(gate), Rc::clone(down))),
        }
    }
}

struct Bench {
    layers: Vec<Layer>,
    side: Stream,
}

impl Bench {
    fn new(
        session: &Session,
        banks: &Banks,
        tokens: usize,
        layer_count: usize,
        clustered: bool,
    ) -> LabResult<Self> {
        // The router GEMM is TP4's at every rank count; the shared expert's
        // entries are the profile's.
        let module = session.module("gemm_qwen38_own_sm120")?;
        let routed = Routed::new(session, tokens)?;
        let profile = routed.profile;
        let shared_module = if profile == model_qwen38_contract::sm120::config::PROFILE {
            None
        } else {
            Some(session.module(&format!("gemm_{}_own_sm120", profile.tag()))?)
        };
        let shared_module = shared_module.as_ref().unwrap_or(&module);
        let shared_padded = model_qwen38_contract::sm120::config::profile_shape(profile)
            .ok_or("not a Qwen3.8 profile")?
            .shared_padded() as usize;
        let m = tokens as u32;
        let mut layers = Vec::new();
        for layer in 0..layer_count {
            let seed = 0x7a11_0000 ^ (layer as u64) << 8;
            let (gate, down) = banks.layer(session)?;
            layers.push(Layer {
                routed: Routed::new(session, tokens)?,
                logits: session.upload_f32(&logits(tokens, clustered, layer))?,
                gate,
                down,
                router: Dense::new(
                    session,
                    &module,
                    &format!("chop_gemm_qwen38_router_c{m}_v1"),
                    m,
                    seed ^ 1,
                )?,
                shared_gate_up: Dense::new(
                    session,
                    shared_module,
                    &format!("chop_gemm_{}_shared_gate_up_c{m}_v1", profile.tag()),
                    m,
                    seed ^ 2,
                )?,
                shared_down: Dense::new(
                    session,
                    shared_module,
                    &format!("chop_gemm_{}_shared_down_c{m}_v1", profile.tag()),
                    m,
                    seed ^ 3,
                )?,
                shared_inter: zeroed(session, tokens * shared_padded * 2)?,
                fork: session.context.event().map_err(err)?,
                join: session.context.event().map_err(err)?,
            });
        }
        Ok(Self {
            layers,
            side: session.context.stream().map_err(err)?,
        })
    }

    fn shared(
        &self,
        session: &Session,
        stream: &Stream,
        layer: &Layer,
        first_pdl: bool,
    ) -> LabResult<()> {
        let r = &layer.routed;
        layer.shared_gate_up.launch(
            session,
            stream,
            Some(&r.x),
            Some(&layer.shared_inter),
            first_pdl,
        )?;
        layer.shared_down.launch(
            session,
            stream,
            Some(&layer.shared_inter),
            Some(&r.shared),
            true,
        )
    }

    fn routed_chain(&self, session: &Session, stream: &Stream, layer: &Layer) -> LabResult<()> {
        let r = &layer.routed;
        layer
            .router
            .launch(session, stream, Some(&r.x), None, true)?;
        r.route(session, stream, &layer.logits, true)?;
        r.projection(session, stream, &layer.gate, Projection::GateUp, true)?;
        r.projection(session, stream, &layer.down, Projection::Down, true)
    }

    fn enqueue(
        &self,
        session: &Session,
        stream: &Stream,
        arrangement: Arrangement,
    ) -> LabResult<()> {
        for layer in &self.layers {
            let r = &layer.routed;
            match arrangement {
                Arrangement::Serial => {
                    self.routed_chain(session, stream, layer)?;
                    self.shared(session, stream, layer, true)?;
                    r.combine(session, stream, true)?;
                }
                Arrangement::SharedFirst => {
                    self.shared(session, stream, layer, true)?;
                    self.routed_chain(session, stream, layer)?;
                    r.combine(session, stream, true)?;
                }
                Arrangement::RoutedOnly => {
                    self.routed_chain(session, stream, layer)?;
                    r.combine(session, stream, true)?;
                }
                Arrangement::Fork => {
                    layer.fork.record(stream).map_err(err)?;
                    self.side.wait_event(&layer.fork).map_err(err)?;
                    self.shared(session, &self.side, layer, false)?;
                    layer.join.record(&self.side).map_err(err)?;
                    self.routed_chain(session, stream, layer)?;
                    stream.wait_event(&layer.join).map_err(err)?;
                    r.combine(session, stream, false)?;
                }
            }
        }
        Ok(())
    }

    fn last(&self) -> &Routed {
        &self.layers.last().expect("nonempty graph").routed
    }

    fn checks<'a>(&'a self, session: &'a Session) -> LabResult<Vec<replay::Checks<'a>>> {
        self.layers
            .iter()
            .map(|layer| {
                let r = &layer.routed;
                let experts = r.local..2 * r.local;
                let routes = r.tokens * QK;
                let geometry = replay::RouteGeometry {
                    routes,
                    experts: experts.clone(),
                    block: kernels_sm120::moe::expert_parallel::ROUTE_BLOCK as usize,
                    sorted_offset: 0,
                };
                let mut checks = replay::Checks::new(session).route(
                    geometry,
                    replay::RouteBuffers {
                        ids: &r.ids,
                        weights: &r.weights,
                        counts: &r.counts,
                        offsets: &r.offsets,
                        sorted: &r.sorted,
                        padded: &r.padded,
                        active: Some(&r.active),
                    },
                    &[0x80],
                )?;
                for (name, buffer, width) in
                    [("gate/up", &r.inter, I), ("weighted down", &r.routed, H)]
                {
                    let experts = experts.clone();
                    checks = checks.dynamic(
                        name,
                        buffer,
                        move |s| {
                            replay::routed_map(
                                &replay::ids(s, &r.ids, routes)?,
                                experts.clone(),
                                width,
                            )
                        },
                        &[0xff],
                    );
                }
                checks = checks.output(
                    "combine",
                    &r.output,
                    WriteMap::of(DType::Bf16).prefix(r.tokens * H)?,
                );
                checks = layer.router.add_checks(checks, None)?;
                checks = layer
                    .shared_gate_up
                    .add_checks(checks, Some(&layer.shared_inter))?;
                checks = layer.shared_down.add_checks(checks, Some(&r.shared))?;
                if r.gate
                    .plan(counts(&r.gate, r.tokens))
                    .map_err(err)?
                    .workspace_bytes
                    != 0
                    || r.down
                        .plan(counts(&r.down, r.tokens))
                        .map_err(err)?
                        .workspace_bytes
                        != 0
                {
                    return Err(
                        "tail grouped projection unexpectedly requires arithmetic workspace".into(),
                    );
                }
                // Route tickets, dummy protocol bindings and dense semaphores stay valid.
                Ok(checks)
            })
            .collect()
    }

    fn phase(
        &self,
        session: &Session,
        checks: &mut [replay::Checks<'_>],
        phase: ReplayPhase,
        graph: Option<&runtime_cuda::GraphExec>,
    ) -> LabResult<()> {
        session.stream.synchronize().map_err(err)?;
        self.side.synchronize().map_err(err)?;
        replay::phase_all(checks, phase, phase != ReplayPhase::Eager, || {
            if let Some(graph) = graph {
                graph.launch(&session.stream).map_err(err)
            } else {
                self.enqueue(session, &session.stream, Arrangement::Serial)
            }
        })
    }
}

fn token_counts(lean: bool) -> Vec<usize> {
    std::env::var("CHOPINFER_QWEN_TAIL_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| if lean { vec![1, 32] } else { vec![1, 16, 32] })
}

fn layer_count(lean: bool) -> usize {
    std::env::var("CHOPINFER_QWEN_TAIL_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n| (1..=16).contains(&n))
        .unwrap_or(if lean { 2 } else { 4 })
}

fn clustered() -> bool {
    std::env::var("CHOPINFER_QWEN_TAIL_ROUTING").as_deref() == Ok("clustered")
}

#[test]
fn qwen38_moe_tail_arrangements() {
    let bench = std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() == Ok("1");
    gpu_test(8 * 1024 * 1024 * 1024, |session| {
        let lean = kernel_lab::qwen38_lean()?;
        let layers = layer_count(lean);
        let (profile, local) = topology()?;
        let host = (
            HostBank::new(local, 2 * I, H, 0x9a7e_0000)?,
            HostBank::new(local, H, I, 0xd0a1_0000)?,
        );
        let shared;
        let banks = if lean {
            shared = (Rc::new(host.0.upload(session)?), Rc::new(host.1.upload(session)?));
            Banks::Shared(&shared)
        } else {
            Banks::Upload(&host)
        };
        let mut case_failures = replay::Failures::default();
        for tokens in token_counts(lean) {
            let result = (|| -> LabResult<()> {
            let setup = Bench::new(session, &banks, tokens, layers, clustered())?;
            let arrangements = [
                Arrangement::Serial,
                Arrangement::Fork,
                Arrangement::SharedFirst,
                Arrangement::RoutedOnly,
            ];
            let mut graphs = Vec::new();
            for arrangement in arrangements {
                let graph = session.capture(|stream| setup.enqueue(session, stream, arrangement))?;
                graphs.push((arrangement, graph));
            }
            let mut checks = setup.checks(session)?;
            setup.phase(session, &mut checks, ReplayPhase::Eager, None)?;
            let reference = session.download_bf16(&setup.last().output, tokens * H)?;
            if reference.iter().all(|&v| bf16_to_f32(v) == 0.0) {
                return Err(format!("{tokens} tokens: the serial combine output is all zero"));
            }
            let active = setup.last().active_experts(session)?;
            let mut failures = replay::Failures::default();
            for (arrangement, graph) in &graphs[..3] {
                failures.check(arrangement.name(), setup.phase(session, &mut checks, ReplayPhase::Graph, Some(graph)));
                let output = session.download_bf16(&setup.last().output, tokens * H)?;
                if let Some(index) = output.iter().zip(&reference).position(|(a, b)| a != b) {
                    failures.check(arrangement.name(), Err(format!("{tokens} tokens: combine differs from serial at {index}")));
                }
            }
            failures.check("eager serial relaunch", setup.phase(session, &mut checks, ReplayPhase::Relaunch, None));
            failures.check("PDL serial", setup.phase(session, &mut checks, ReplayPhase::Pdl, None));
            for (layer, check) in checks.iter().enumerate() {
                let mut fields = BTreeMap::from([("tokens".into(), json!(tokens)), ("layer".into(), json!(layer))]);
                check.record(&mut fields);
                session.receipt("moe-tail-replay", fields)?;
            }
            failures.finish()?;
            eprintln!(
                "{tokens} tokens, {layers} layers, {active} active local experts in the last layer: fork, shared_first and eager match serial bitwise"
            );
            if !bench {
                return Ok(());
            }
            let mut per_layer = BTreeMap::new();
            // Alternate the arrangements over three passes so drift in the
            // device state does not favor one of them.
            let mut samples: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
            for _ in 0..3 {
                for (arrangement, graph) in &graphs {
                    let timing = time(session, 3, 15, |stream| graph.launch(stream).map_err(err))?;
                    samples
                        .entry(arrangement.name())
                        .or_default()
                        .push(f64::from(timing.median_ms) * 1000.0 / layers as f64);
                }
            }
            for (name, mut values) in samples {
                values.sort_by(f64::total_cmp);
                per_layer.insert(name.to_string(), json!(values[1]));
                eprintln!("  {name}: {:.2} us per layer (passes {values:.2?})", values[1]);
            }
            let layer = &setup.layers[0];
            let dense_bytes = layer.router.weight_bytes
                + layer.shared_gate_up.weight_bytes
                + layer.shared_down.weight_bytes;
            let mut receipt = BTreeMap::new();
            receipt.insert("case".into(), json!("qwen38_moe_tail"));
            receipt.insert("tokens".into(), json!(tokens));
            receipt.insert("layers".into(), json!(layers));
            receipt.insert("parallel_route".into(), json!(qwen::use_parallel_route(tokens as u32)));
            if let Some(ranks) = profile.ranks_of(model_qwen38_contract::sm120::config::MODEL).filter(|&ranks| ranks != 4) {
                receipt.insert("qwen38_ranks".into(), json!(ranks));
            }
            if lean {
                receipt.insert("lean".into(), json!(true));
            }
            receipt.insert("routing".into(), json!(if clustered() { "clustered" } else { "uniform" }));
            receipt.insert("active_local_experts_last_layer".into(), json!(active));
            receipt.insert("dense_weight_bytes_per_layer".into(), json!(dense_bytes));
            receipt.insert(
                "symbols".into(),
                json!([
                    layer.router.entry.symbol,
                    layer.shared_gate_up.entry.symbol,
                    layer.shared_down.entry.symbol,
                    setup.last().gate.symbol,
                    setup.last().down.symbol,
                ]),
            );
            receipt.insert("us_per_layer_median_of_passes".into(), Value::Object(per_layer.into_iter().collect()));
            receipt.insert(
                "method".into(),
                json!("graph of all layers, L2 flushed before each replay, 3 warmups and 15 rounds per pass, 3 alternating passes"),
            );
            session.receipt("qwen38_moe_tail", receipt)?;
            Ok(())
            })();
            case_failures.check(&format!("tail tokens {tokens}"), result);
        }
        case_failures.finish()
    })
    .expect("Qwen3.8 MoE tail arrangements");
}
