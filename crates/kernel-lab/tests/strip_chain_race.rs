//! The ring-stage release race (fixed 2026-10-09) on kernel-lab's
//! `strip_chain_race` cubins (`strip_chain_race_shapes.rs`).
//!
//! The locked strip-chain and warp-tile kernels release a TMA ring stage
//! before the step's last shared reads of it return, so the producer can
//! refill the stage under a math warp; that warp's MMA chain then mixes in a
//! later chunk's operands. The served (`_ordered`) kernels fence the reads
//! first.
//!
//! - CPU (`race_instances_are_built`): every case's three entries are in the
//!   cubin, the locked entry's body is the served body but for the
//!   function's name, and the contention probe is built.
//! - GPU (`strip_chains_survive_release_race`): per case, the one-split
//!   entry's output at the capacity and at one row, from poisoned outputs,
//!   run twice and equal, is the reference. Then for
//!   `CHOPINFER_STRIP_RACE_ATTEMPTS` attempts (default 64) the locked and
//!   the served entry each run from a poisoned output and chain workspace at
//!   the capacity (even attempts) or one row (odd attempts); every other
//!   pair of attempts runs beside the contention probe on a second stream,
//!   which keeps the SMs' shared-memory and store pipes busy
//!   (`CHOPINFER_STRIP_RACE_CONTEND_NS`, default 200 us). An attempt counts
//!   when the whole output differs from the reference by a byte. The served
//!   entry must differ in no attempt of any case, and the locked entry in at
//!   least one attempt of a vocabulary-head case, so the test sees the race
//!   it guards against. The receipt records every case's counts.
//! - GPU, `CHOPINFER_KERNEL_BENCH=1` (`strip_chain_race_timing`): each
//!   case's locked and served entry timed in alternating pairs.
//!
//! Run: `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release
//! --test strip_chain_race`. `CHOPINFER_STRIP_RACE_FILTER` keeps the cases
//! whose tag contains it (the head check then applies only if a head case
//! is kept).

use kernel_core::contract::{GemmQuery, InstantiationSpec, LaunchPlan, WeightFormat};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison;
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::gemm::{self, GemmBuffers};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::families::strip_chain_race::shapes;

const WEIGHT: u64 = 1;
const ACTIVATION: u64 = 2;
const OUTPUT: u64 = 3;
const PARTIAL: u64 = 4;
const SEMAPHORES: u64 = 5;
const SCALES: u64 = 6;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[test]
fn race_instances_are_built() {
    let built = kernel_lab::cubin_entries(shapes::CUBIN).unwrap();
    let cubins = shapes::cubins();
    let bodies: BTreeMap<&str, &str> = cubins[0]
        .entries
        .iter()
        .map(|entry| (entry.symbol.as_str(), entry.body.as_str()))
        .collect();
    let cases = shapes::cases();
    assert!(cases.iter().any(|case| case.tag.starts_with("head_")));
    for case in &cases {
        for spec in [&case.one, &case.locked, &case.ordered] {
            assert!(built.contains(&spec.symbol.as_str()), "{}", spec.symbol);
        }
        let ordered = bodies[case.ordered.symbol.as_str()];
        let locked = bodies[case.locked.symbol.as_str()];
        assert!(ordered.contains("_ordered<"), "{ordered}");
        assert!(!locked.contains("_ordered<"), "{locked}");
        assert_eq!(locked, shapes::locked_body(ordered), "{}", case.tag);
    }
    let probe = kernel_lab::cubin_entries(shapes::PROBE_CUBIN).unwrap();
    assert!(probe.contains(&shapes::CONTEND));
}

/// Deterministic values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// Signed BF16 values with exponents -15 through 14.
    fn bf16_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                let bits = self.next();
                ((bits as u16 & 0x807f) | ((112 + (bits >> 16) % 30) as u16) << 7).to_le_bytes()
            })
            .collect()
    }

    /// E4M3 codes other than the NaN codes.
    fn e4m3_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .map(|_| match self.next() as u8 {
                0x7f | 0xff => 0,
                code => code,
            })
            .collect()
    }

    /// Positive F32 scales over 2^-20 through 2^3, mantissas included.
    fn scale_bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count)
            .flat_map(|_| {
                let bits = self.next();
                let exponent = (127 - 20 + bits % 24) << 23;
                f32::from_bits(exponent | (bits >> 9 & 0x7f_ffff)).to_le_bytes()
            })
            .collect()
    }
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(4))?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn function(module: &Module, spec: &InstantiationSpec) -> LabResult<Function> {
    let shared = gemm::plan(&spec.key, gemm::row_counts(&spec.key, m_capacity(spec)?))
        .map_err(err)?
        .dynamic_shared_bytes;
    let function = module.function(&spec.symbol).map_err(err)?;
    if shared >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(function)
}

fn m_capacity(spec: &InstantiationSpec) -> LabResult<u32> {
    match spec.key.query {
        kernel_core::contract::StaticQuery::Gemm(query) => Ok(query.m_capacity),
        _ => Err(format!("{} is not a GEMM", spec.symbol)),
    }
}

/// One case's operands, one output and one chain workspace and ticket array
/// that every launch shares.
struct Fixture {
    weights: DeviceBuffer,
    scales: Option<DeviceBuffer>,
    activation: DeviceBuffer,
    output: DeviceBuffer,
    workspace: DeviceBuffer,
    semaphores: DeviceBuffer,
}

impl Fixture {
    fn new(session: &Session, case: &shapes::Case) -> LabResult<Self> {
        let query: GemmQuery = case.query;
        let (n, k) = (query.n as usize, query.k as usize);
        let mut rng = Rng(0xf1c5_0001 ^ (n as u64) << 24 ^ (k as u64) << 4);
        let (weights, scales) = match query.weight {
            WeightFormat::Bf16 => (rng.bf16_bytes(n * k), None),
            WeightFormat::Fp8Block128x128(_) => (
                rng.e4m3_bytes(n * k),
                Some(rng.scale_bytes(n.div_ceil(128) * k.div_ceil(128))),
            ),
            other => return Err(format!("no fixture for {other:?} weights")),
        };
        let element = if query.output == kernel_core::contract::DType::F32 {
            4
        } else {
            2
        };
        let mut workspace = 4;
        let mut semaphores = 4;
        for spec in [&case.one, &case.locked, &case.ordered] {
            let plan = gemm::plan(&spec.key, gemm::row_counts(&spec.key, query.m_capacity))
                .map_err(err)?;
            workspace = workspace.max(plan.workspace_bytes as usize);
            semaphores = semaphores
                .max(GemmBuffers::semaphore_capacity_bytes(&spec.key).map_err(err)? as usize);
        }
        let semaphores = session.allocate(semaphores)?;
        poison::fill_all(session, &semaphores, &[0])?;
        Ok(Self {
            weights: upload(session, &weights)?,
            scales: scales.map(|s| upload(session, &s)).transpose()?,
            activation: upload(session, &rng.bf16_bytes(query.m_capacity as usize * k))?,
            output: session.allocate(query.m_capacity as usize * n * element)?,
            workspace: session.allocate(workspace)?,
            semaphores,
        })
    }

    /// Launches `spec` at `m` rows on `stream`.
    fn launch(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        m: u32,
        stream: &Stream,
    ) -> LabResult<()> {
        let counts = gemm::row_counts(&spec.key, m);
        let plan: LaunchPlan = gemm::plan(&spec.key, counts).map_err(err)?;
        let maps = gemm::tensor_maps(&spec.key, WEIGHT, ACTIVATION).map_err(err)?;
        let args = gemm::arguments(
            &spec.key,
            counts,
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: self.scales.is_some().then_some(SCALES),
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(err)?;
        let mut buffers = BTreeMap::from([
            (WEIGHT, &self.weights),
            (ACTIVATION, &self.activation),
            (OUTPUT, &self.output),
            (PARTIAL, &self.workspace),
            (SEMAPHORES, &self.semaphores),
        ]);
        if let Some(scales) = &self.scales {
            buffers.insert(SCALES, scales);
        }
        launch::launch(
            &session.context,
            function,
            stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    }

    /// `spec`'s output at `m` rows from a poisoned output and workspace,
    /// with `contention` launched on its stream just before.
    fn run(
        &self,
        session: &Session,
        spec: &InstantiationSpec,
        function: &Function,
        m: u32,
        contention: Option<(&Contention, &Stream)>,
    ) -> LabResult<Vec<u8>> {
        poison::poison(session, &[&self.output, &self.workspace])?;
        if let Some((contention, stream)) = contention {
            contention.launch(stream)?;
        }
        self.launch(session, spec, function, m, &session.stream)?;
        session.stream.synchronize().map_err(err)?;
        if let Some((_, stream)) = contention {
            stream.synchronize().map_err(err)?;
        }
        poison::read(session, &self.output)
    }
}

/// The contention probe (`strip_race.cuh`'s `contend`): two CTAs per SM.
struct Contention {
    function: Function,
    sink: DeviceBuffer,
    ns: u64,
}

impl Contention {
    fn new(session: &Session, ns: u64) -> LabResult<Self> {
        Ok(Self {
            function: session
                .module(shapes::PROBE_CUBIN)?
                .function(shapes::CONTEND)
                .map_err(err)?,
            sink: session.allocate(SINK_WORDS * 4)?,
            ns,
        })
    }

    fn launch(&self, stream: &Stream) -> LabResult<()> {
        self.function
            .launch(
                stream,
                launch::simple_config(
                    (2 * gemm::SM_COUNT, 1, 1),
                    (shapes::CONTEND_THREADS, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[
                    KernelArgument::Buffer(&self.sink),
                    KernelArgument::U64(SINK_WORDS as u64),
                    KernelArgument::U64(self.ns),
                ],
            )
            .map_err(err)
    }
}

/// Floats the contention probe stores into (64 MiB).
const SINK_WORDS: usize = 16 << 20;

fn env_u64(name: &str, default: u64) -> LabResult<u64> {
    match std::env::var(name) {
        Ok(value) => value.parse().map_err(|e| format!("{name}={value}: {e}")),
        Err(_) => Ok(default),
    }
}

/// The differing element count and first differing element of two outputs.
fn difference(element: usize, want: &[u8], got: &[u8]) -> Option<(usize, usize)> {
    let mut first = None;
    let mut count = 0;
    for (index, (w, g)) in want
        .chunks_exact(element)
        .zip(got.chunks_exact(element))
        .enumerate()
    {
        if w != g {
            count += 1;
            first.get_or_insert(index);
        }
    }
    first.map(|first| (count, first))
}

#[test]
fn strip_chains_survive_release_race() {
    gpu_test(4 << 30, |session| {
        let attempts = env_u64("CHOPINFER_STRIP_RACE_ATTEMPTS", 64)?;
        let contend_ns = env_u64("CHOPINFER_STRIP_RACE_CONTEND_NS", 200_000)?;
        let filter = std::env::var("CHOPINFER_STRIP_RACE_FILTER").ok();
        let module = session.module(shapes::CUBIN)?;
        let contention = Contention::new(session, contend_ns)?;
        let second = session.context.stream().map_err(err)?;
        let mut results = Vec::new();
        let mut ordered_failures = Vec::new();
        let (mut head_cases, mut head_locked) = (0, 0);
        for case in shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let fixture = Fixture::new(session, &case)?;
            let element = if case.query.output == kernel_core::contract::DType::F32 {
                4
            } else {
                2
            };
            let capacity = case.query.m_capacity;
            let one = function(&module, &case.one)?;
            let mut references = BTreeMap::new();
            for m in [capacity, 1] {
                let first = fixture.run(session, &case.one, &one, m, None)?;
                let again = fixture.run(session, &case.one, &one, m, None)?;
                if first != again {
                    return Err(format!("{} one-split reference varies at M={m}", case.tag));
                }
                references.insert(m, first);
            }
            let mut counts = BTreeMap::new();
            for (kind, spec) in [("locked", &case.locked), ("ordered", &case.ordered)] {
                let function = function(&module, spec)?;
                let (mut quiet, mut contended) = (0u64, 0u64);
                let mut first_difference = None;
                for attempt in 0..attempts {
                    let m = if attempt % 2 == 0 { capacity } else { 1 };
                    let busy = attempt % 4 >= 2;
                    let got = fixture.run(
                        session,
                        spec,
                        &function,
                        m,
                        busy.then_some((&contention, &second)),
                    )?;
                    if let Some((count, first)) = difference(element, &references[&m], &got) {
                        if busy {
                            contended += 1;
                        } else {
                            quiet += 1;
                        }
                        first_difference.get_or_insert_with(|| {
                            let n = case.query.n as usize;
                            json!({
                                "attempt": attempt,
                                "m": m,
                                "contended": busy,
                                "elements": count,
                                "row": first / n,
                                "column": first % n,
                            })
                        });
                    }
                }
                eprintln!(
                    "{} {kind} ({}): {} of {attempts} attempts differ ({quiet} quiet, \
                     {contended} contended)",
                    case.tag,
                    spec.symbol,
                    quiet + contended
                );
                if kind == "ordered" && quiet + contended > 0 {
                    ordered_failures.push(format!(
                        "{}: {} of {attempts} attempts differ, first {}",
                        case.tag,
                        quiet + contended,
                        first_difference.clone().unwrap_or_default()
                    ));
                }
                if kind == "locked" && case.tag.starts_with("head_") {
                    head_locked += quiet + contended;
                }
                counts.insert(
                    kind,
                    json!({
                        "symbol": spec.symbol,
                        "differing_quiet": quiet,
                        "differing_contended": contended,
                        "first_difference": first_difference,
                    }),
                );
            }
            if case.tag.starts_with("head_") {
                head_cases += 1;
            }
            results.push(json!({
                "tag": case.tag,
                "kernel": format!("{:?}", case.kernel),
                "capacity": capacity,
                "n": case.query.n,
                "k": case.query.k,
                "weight": format!("{:?}", case.query.weight),
                "reference": case.one.symbol,
                "attempts": attempts,
                "counts": counts,
            }));
        }
        if results.is_empty() {
            return Err("no strip-chain race case ran".into());
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("strip_chain_release_race"));
        receipt.insert("attempts".into(), json!(attempts));
        receipt.insert("contend_ns".into(), json!(contend_ns));
        receipt.insert("cases".into(), json!(results));
        let passed = ordered_failures.is_empty() && (head_cases == 0 || head_locked > 0);
        receipt.insert("passed".into(), json!(passed));
        session.receipt("strip_chain_race", receipt)?;
        if !ordered_failures.is_empty() {
            return Err(format!(
                "served strip chains differ: {}",
                ordered_failures.join("; ")
            ));
        }
        if head_cases > 0 && head_locked == 0 {
            return Err(format!(
                "the locked kernel never differed in {head_cases} head cases of {attempts} \
                 attempts: the test did not see the race"
            ));
        }
        Ok(())
    })
    .expect("strip chains survive the release race");
}

/// Launches per timed graph: decode entries take microseconds, prefill
/// entries milliseconds.
fn timed_launches(capacity: u32) -> usize {
    if capacity <= 32 { 20 } else { 4 }
}

/// Each case's locked and served entry at its capacity, each as one graph
/// of `timed_launches` launches, timed in alternating pairs
/// (`kernel_lab::time_ab`: two warmups, 15 rounds, each arm after an L2
/// eviction): the per-launch medians and the median of the paired
/// served/locked ratios. Set CHOPINFER_KERNEL_BENCH=1.
#[test]
fn strip_chain_race_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("strip-chain race timing: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(4 << 30, |session| {
        let filter = std::env::var("CHOPINFER_STRIP_RACE_FILTER").ok();
        let eviction = kernel_lab::eviction::EvictionMode::selected()?;
        let module = session.module(shapes::CUBIN)?;
        let mut results = Vec::new();
        for case in shapes::cases() {
            if filter.as_deref().is_some_and(|f| !case.tag.contains(f)) {
                continue;
            }
            let fixture = Fixture::new(session, &case)?;
            let capacity = case.query.m_capacity;
            let launches = timed_launches(capacity);
            let specs = [&case.locked, &case.ordered];
            let functions = specs
                .iter()
                .map(|spec| function(&module, spec))
                .collect::<LabResult<Vec<_>>>()?;
            let graphs = specs
                .iter()
                .zip(&functions)
                .map(|(spec, function)| {
                    session.capture(|stream| {
                        (0..launches).try_for_each(|_| {
                            fixture.launch(session, spec, function, capacity, stream)
                        })
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            let timing = kernel_lab::time_ab(session, 2, 15, [eviction; 2], |arm, stream| {
                graphs[arm].launch(stream).map_err(err)
            })?;
            let per = |ms: f32| f64::from(ms) * 1000.0 / launches as f64;
            let (locked_us, ordered_us) =
                (per(timing.arms[0].median_ms), per(timing.arms[1].median_ms));
            eprintln!(
                "{} timing: locked {locked_us:.2} us, served {ordered_us:.2} us, paired \
                 served/locked {:.4}",
                case.tag, timing.paired_ratio
            );
            results.push(json!({
                "tag": case.tag,
                "kernel": format!("{:?}", case.kernel),
                "capacity": capacity,
                "launches": launches,
                "locked_us": locked_us,
                "ordered_us": ordered_us,
                "paired_ordered_over_locked": timing.paired_ratio,
            }));
        }
        if results.is_empty() {
            return Err("no strip-chain race case ran".into());
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("strip_chain_race_timing"));
        receipt.insert("eviction".into(), json!(eviction.name()));
        receipt.insert("cases".into(), json!(results));
        session.receipt("bench", receipt)?;
        Ok(())
    })
    .expect("strip-chain race timing");
}
