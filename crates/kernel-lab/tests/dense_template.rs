//! The shared dense cluster decode template (`cuda/gemm/cluster_decode.cuh`)
//! against the models' own cluster decode kernels: Qwen3.8's
//! `run_cluster_decode` (`qwen38.cuh`: the `_c{M}` buckets at TP2, TP4 and
//! the HC precompute producers) and MiMo V2.6's
//! `run_fragment_decode_cluster`.
//!
//! `kernels_sm120::gemm::cluster::lab_instances` pairs every served entry's
//! template entry with its reference at the same shape, ABI and launch plan:
//! while a model serves its own kernel, the served entry and a
//! `chop_gemm_tpl_` lab instance; once it serves the template, the served
//! entry and kernel-lab's frozen `chop_gemm_eng_` copy of the old kernel.
//! Both run on identical buffers at each live count of `poison::live_counts`
//! above zero (the planners refuse an empty launch): capacity, 1, half and
//! capacity - 1 of every decode bucket from 1 to 32. Two input sets:
//!
//! - wide: BF16 weights and activations over 13 binades, or FP8 weights
//!   over every finite E4M3 code with F32 block scales over 17 binades, so a
//!   different accumulation order changes output bits;
//! - real (Qwen3.8 only): each entry family's checkpoint weights and dumped
//!   activations (`CHOPINFER_QWEN38_CHECKPOINT`,
//!   `CHOPINFER_QWEN38_LAYER_DUMPS`). Without them the test fails under
//!   `CHOPINFER_KERNEL_REQUIRE=1` and otherwise records the skip.
//!
//! Outputs (with padded Qwen row strides), the HC bottleneck and the cluster
//! workspace are poisoned before every launch, and activation rows past the
//! live count hold NaN. Each launch must write exactly its live output rows,
//! bottleneck rows and every partial slot of its plan, with no NaN in any of
//! them (`check_finite`), and the two arms' outputs, bottlenecks and partials
//! must agree bit for bit.
//!
//! The ignored `dense_template_timing` alternates the reference (A) and the
//! template (B), A B A B per repetition, and gates the median ratio of
//! adjacent pairs per instance.
//!
//! `CHOPINFER_DENSE_TEMPLATE=<substring>` selects instances by served symbol.

use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{
    Argument, ArgumentList, DType, GemmQuery, LaunchPlan, RuntimeCounts, StaticQuery,
};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Fill, POISON, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, time};
use kernels_sm120::gemm::cluster::{self, Instance, Served, Weights};
use kernels_sm120::gemm::fused::Buffers;
use kernels_sm120::gemm::{self, GemmBuffers};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::rc::Rc;

#[path = "common/accumulation.rs"]
mod accumulation;
#[path = "common/qwen_own_inputs.rs"]
mod qwen_own_inputs;

const ACTIVATION: u64 = 3;
const SCALES: u64 = 4;
const PARTIAL: u64 = 5;
const SEMAPHORES: u64 = 6;
const BOTTLENECK: u64 = 7;
/// Weight and output buffer handles are these plus the segment index.
const WEIGHT: u64 = 10;
const OUTPUT: u64 = 20;
/// Columns past every Qwen output row, which no launch may write.
const PAD: usize = 8;
/// Workspace bytes past the plan's, which no launch may write.
const SLACK: usize = 4096;
/// Synthetic activation rows per bank: more than the largest decode bucket,
/// so buckets of one family read different rows.
const POOL_ROWS: usize = 64;
/// Width of the HC bottleneck rows.
const HC_RANK: usize = 320;

fn err(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn instances() -> Vec<Instance> {
    let filter = std::env::var("CHOPINFER_DENSE_TEMPLATE").ok();
    cluster::lab_instances(kernel_lab::catalogs())
        .into_iter()
        .filter(|i| {
            filter
                .as_deref()
                .is_none_or(|f| i.served_symbol.contains(f))
        })
        .collect()
}

fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

/// The capacity, K and output type of an instance's served entry.
fn query(instance: &Instance) -> GemmQuery {
    match &instance.served {
        Served::Fused(entry) | Served::FusedHcPrecompute(entry) => entry.query,
        Served::Shared(variant) => match variant.key.query {
            StaticQuery::Gemm(query) => query,
            _ => unreachable!("GEMM catalog entries hold GEMM queries"),
        },
    }
}

/// Output rows of each weight: the Qwen segments, or the shared entry's N.
fn segment_rows(instance: &Instance) -> Vec<usize> {
    match &instance.served {
        Served::Fused(entry) | Served::FusedHcPrecompute(entry) => {
            entry.segments.iter().map(|&r| r as usize).collect()
        }
        Served::Shared(_) => vec![query(instance).n as usize],
    }
}

/// The served symbol less its bucket: instances of one family share a bank.
fn family(instance: &Instance) -> String {
    let symbol = match &instance.served {
        Served::FusedHcPrecompute(entry) => entry.symbol.clone(),
        _ => instance.served_symbol.clone(),
    };
    let stem = symbol.trim_end_matches("_v1");
    match stem.rsplit_once('_') {
        Some((head, bucket))
            if bucket.len() > 1
                && (bucket.starts_with('c') || bucket.starts_with('m'))
                && bucket[1..].parse::<u32>().is_ok() =>
        {
            head.to_owned()
        }
        _ => stem.to_owned(),
    }
}

/// The input sets a case runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Inputs {
    Wide,
    Real,
}

impl Inputs {
    fn name(self) -> &'static str {
        match self {
            Inputs::Wide => "wide",
            Inputs::Real => "real",
        }
    }
}

/// One family's weights (packed per segment), FP8 scales and a pool of
/// activation rows.
struct Bank {
    packed: Vec<DeviceBuffer>,
    scales: Option<DeviceBuffer>,
    pool: Vec<u16>,
    note: String,
}

impl Bank {
    fn wide(session: &Session, instance: &Instance, seed: u64) -> LabResult<Self> {
        let q = query(instance);
        let k = q.k as usize;
        let weights = instance.shape.weights;
        let mut packed = Vec::new();
        for (s, rows) in segment_rows(instance).into_iter().enumerate() {
            let seed = mix(seed ^ (s as u64) << 40);
            let (format, bytes) = if weights == Weights::Bf16 {
                (
                    FragmentWeightFormat::Bf16,
                    bf16_bytes(&accumulation::wide_rows(rows, k, seed)),
                )
            } else {
                // Every finite E4M3 code: 0x7f and 0xff are NaN.
                let codes = (0..rows * k)
                    .map(|i| {
                        let b = mix(seed ^ i as u64) as u8;
                        if b & 0x7f == 0x7f { b - 1 } else { b }
                    })
                    .collect::<Vec<_>>();
                (FragmentWeightFormat::Fp8E4M3, codes)
            };
            let data = FragmentWeightLayout { rows, k, format }
                .pack(&bytes)
                .map_err(err)?
                .bytes;
            packed.push(upload(session, &data)?);
        }
        let scales = match weights {
            Weights::Bf16 => None,
            format => {
                let group = if format == Weights::Fp8Rows64 {
                    64
                } else {
                    128
                };
                let len = (q.n as usize).div_ceil(group) * (k / 128);
                // Exact F32 scales from 2^-8 to 2^8 with varied significands.
                let values = (0..len).flat_map(|i| {
                    let bits = mix(seed ^ 0x5ca1e ^ (i as u64) << 8);
                    let scale = (1.0 + (bits >> 40) as f64 / (1_u64 << 24) as f64)
                        * 2_f64.powi((bits % 17) as i32 - 8);
                    (scale as f32).to_le_bytes()
                });
                Some(upload(session, &values.collect::<Vec<_>>())?)
            }
        };
        Ok(Self {
            packed,
            scales,
            pool: accumulation::wide_rows(POOL_ROWS, k, mix(seed ^ 0xac71)),
            note: "wide-range synthetic weights and activations".into(),
        })
    }

    /// Checkpoint weights and dumped activations of a Qwen family.
    fn real(
        session: &Session,
        sources: &accumulation::QwenSources,
        instance: &Instance,
    ) -> LabResult<Option<Self>> {
        let entry = match &instance.served {
            Served::Fused(entry) | Served::FusedHcPrecompute(entry) => entry,
            Served::Shared(_) => return Ok(None),
        };
        let prefix = format!("chop_gemm_{}_", entry.profile.tag());
        let name = family(instance);
        let family = name.strip_prefix(&prefix).unwrap_or(&name);
        let ranks = entry.profile.ranks().expect("a served profile") as usize;
        let Some(inputs) = qwen_own_inputs::own_inputs(sources, entry, family, ranks)? else {
            return Err(format!("{family}: no real input mapping"));
        };
        let k = entry.query.k as usize;
        let mut packed = Vec::new();
        let mut first = 0;
        for &rows in &entry.segments {
            let rows = rows as usize;
            let data = FragmentWeightLayout {
                rows,
                k,
                format: FragmentWeightFormat::Bf16,
            }
            .pack(&bf16_bytes(&inputs.weight[first * k..(first + rows) * k]))
            .map_err(err)?
            .bytes;
            packed.push(upload(session, &data)?);
            first += rows;
        }
        Ok(Some(Self {
            packed,
            scales: None,
            pool: inputs.rows,
            note: format!("{} ({})", inputs.note, inputs.origin.label()),
        }))
    }
}

/// One output buffer: columns written per row and the row stride.
struct Output {
    columns: usize,
    stride: usize,
}

/// Which entry a launch runs; both use the served launch plan and ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Reference,
    Template,
}

/// Buffers of one instance at its capacity.
struct Case<'a> {
    instance: &'a Instance,
    capacity: usize,
    k: usize,
    element: usize,
    reference: Function,
    template: Function,
    activation: DeviceBuffer,
    outputs: Vec<(Output, DeviceBuffer)>,
    bottleneck: Option<DeviceBuffer>,
    workspace: DeviceBuffer,
    semaphores: DeviceBuffer,
}

fn function(session: &Session, cubin: &str, symbol: &str, shared: u32) -> LabResult<Function> {
    let function = session.module(cubin)?.function(symbol).map_err(err)?;
    if shared >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(err)?;
    }
    Ok(function)
}

fn counts(live: u32) -> RuntimeCounts {
    RuntimeCounts {
        m: live,
        batch: live,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    }
}

impl<'a> Case<'a> {
    fn new(session: &Session, instance: &'a Instance) -> LabResult<Self> {
        let q = query(instance);
        let capacity = q.m_capacity as usize;
        let plan = Self::plan_of(instance, capacity as u32)?;
        let shared = plan.dynamic_shared_bytes;
        let element = if q.output == DType::F32 { 4 } else { 2 };
        let swiglu = q.epilogues.swiglu.is_some();
        let qwen = !matches!(instance.served, Served::Shared(_));
        let mut outputs = Vec::new();
        for rows in segment_rows(instance) {
            let columns = if swiglu { rows / 2 } else { rows };
            let stride = columns + if qwen { PAD } else { 0 };
            let buffer = session.allocate(capacity * stride * element)?;
            outputs.push((Output { columns, stride }, buffer));
        }
        let bottleneck = matches!(instance.served, Served::FusedHcPrecompute(_))
            .then(|| session.allocate(capacity * HC_RANK * 2))
            .transpose()?;
        let workspace = session.allocate(plan.workspace_bytes as usize + SLACK)?;
        let semaphores = session.allocate(4096)?;
        semaphores
            .zero_at_async(0, semaphores.byte_len(), &session.stream)
            .map_err(err)?;
        Ok(Self {
            instance,
            capacity,
            k: q.k as usize,
            element,
            reference: function(
                session,
                &instance.reference_cubin,
                &instance.reference_symbol,
                shared,
            )?,
            template: function(session, &instance.template_cubin, instance.symbol(), shared)?,
            activation: session.allocate(capacity * q.k as usize * 2)?,
            outputs,
            bottleneck,
            workspace,
            semaphores,
        })
    }

    fn plan_of(instance: &Instance, live: u32) -> LabResult<LaunchPlan> {
        match &instance.served {
            Served::Fused(entry) | Served::FusedHcPrecompute(entry) => {
                entry.plan(live).map_err(err)
            }
            Served::Shared(variant) => gemm::plan(&variant.key, counts(live)).map_err(err),
        }
    }

    fn arguments(&self, live: u32) -> LabResult<ArgumentList> {
        let mut args = match &self.instance.served {
            Served::Fused(entry) | Served::FusedHcPrecompute(entry) => {
                let mut output_strides = [0; 4];
                for (s, (output, _)) in self.outputs.iter().enumerate() {
                    output_strides[s] = output.stride as u32;
                }
                entry
                    .arguments(
                        live,
                        Buffers {
                            weights: std::array::from_fn(|s| WEIGHT + s as u64),
                            activation: ACTIVATION,
                            outputs: std::array::from_fn(|s| OUTPUT + s as u64),
                            output_strides,
                            partial: PARTIAL,
                            semaphores: SEMAPHORES,
                        },
                    )
                    .map_err(err)?
            }
            Served::Shared(variant) => {
                let key = &variant.key;
                let maps = gemm::tensor_maps(key, WEIGHT, ACTIVATION).map_err(err)?;
                gemm::arguments(
                    key,
                    counts(live),
                    maps,
                    GemmBuffers {
                        output: OUTPUT,
                        partial: PARTIAL,
                        semaphores: SEMAPHORES,
                        weight_scales: (self.instance.shape.weights != Weights::Bf16)
                            .then_some(SCALES),
                        activation_scales: None,
                        residual: None,
                        weight_global_scale: None,
                        activation_global_scale: None,
                    },
                )
                .map_err(err)?
            }
        };
        if self.bottleneck.is_some() {
            // As the bundle's dense group with a bottleneck appends it.
            args.0.push(Argument::DeviceBuffer(BOTTLENECK));
        }
        Ok(args)
    }

    /// Writes live rows of the bank's activation pool (starting at pool row
    /// `offset`) and NaN in every row past them.
    fn prepare(&self, session: &Session, bank: &Bank, live: usize, offset: usize) -> LabResult<()> {
        let k = self.k;
        let pool_rows = bank.pool.len() / k;
        let mut rows = Vec::with_capacity(self.capacity * k);
        for t in 0..live {
            let row = (offset + t) % pool_rows;
            rows.extend_from_slice(&bank.pool[row * k..(row + 1) * k]);
        }
        let mut bytes = bf16_bytes(&rows);
        bytes.resize(self.capacity * k * 2, POISON);
        poison::write(session, &self.activation, 0, &bytes)
    }

    fn launch(
        &self,
        session: &Session,
        bank: &Bank,
        arm: Arm,
        live: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let mut plan = Self::plan_of(self.instance, live)?;
        plan.pdl = pdl;
        if plan.workspace_bytes as usize + SLACK > self.workspace.byte_len() {
            return Err(format!(
                "{} needs a larger workspace",
                self.instance.served_symbol
            ));
        }
        let args = self.arguments(live)?;
        let mut buffers = BTreeMap::from([
            (ACTIVATION, &self.activation),
            (PARTIAL, &self.workspace),
            (SEMAPHORES, &self.semaphores),
        ]);
        for (s, weight) in bank.packed.iter().enumerate() {
            buffers.insert(WEIGHT + s as u64, weight);
        }
        for (s, (_, output)) in self.outputs.iter().enumerate() {
            buffers.insert(OUTPUT + s as u64, output);
        }
        if let Some(scales) = &bank.scales {
            buffers.insert(SCALES, scales);
        }
        if let Some(bottleneck) = &self.bottleneck {
            buffers.insert(BOTTLENECK, bottleneck);
        }
        let function = match arm {
            Arm::Reference => &self.reference,
            Arm::Template => &self.template,
        };
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &Bindings { buffers },
        )
    }

    /// Poisons every buffer the launch writes, launches one arm and returns
    /// the outputs, bottleneck and workspace bytes after checking their
    /// write maps and that no written value is NaN.
    fn run(
        &self,
        session: &Session,
        bank: &Bank,
        arm: Arm,
        live: usize,
    ) -> LabResult<Vec<Vec<u8>>> {
        let mut written: Vec<&DeviceBuffer> = self.outputs.iter().map(|(_, b)| b).collect();
        written.extend(&self.bottleneck);
        written.push(&self.workspace);
        poison::poison(session, &written)?;
        self.launch(session, bank, arm, live as u32, true)?;
        session.stream.synchronize().map_err(err)?;
        let label = format!(
            "{} {arm:?}",
            match arm {
                Arm::Reference => &self.instance.reference_symbol,
                Arm::Template => self.instance.symbol(),
            }
        );
        let e = self.element;
        let mut images = Vec::new();
        for (s, (output, buffer)) in self.outputs.iter().enumerate() {
            let bytes = poison::read(session, buffer)?;
            let map = WriteMap::of(if e == 4 { DType::F32 } else { DType::Bf16 })
                .ranges((0..live).map(|t| t * output.stride..t * output.stride + output.columns))?;
            poison::check(
                &format!("{label} output {s}"),
                &bytes,
                &map,
                Fill::Pattern(&[POISON]),
            )?;
            check_finite(
                &format!("{label} output {s}"),
                &bytes,
                e,
                (0..live).flat_map(|t| (0..output.columns).map(move |c| t * output.stride + c)),
            )?;
            images.push(bytes);
        }
        if let Some(buffer) = &self.bottleneck {
            let bytes = poison::read(session, buffer)?;
            let map = WriteMap::of(DType::Bf16).prefix(live * HC_RANK)?;
            poison::check(
                &format!("{label} bottleneck"),
                &bytes,
                &map,
                Fill::Pattern(&[POISON]),
            )?;
            check_finite(&format!("{label} bottleneck"), &bytes, 2, 0..live * HC_RANK)?;
            images.push(bytes);
        }
        // Every clustered CTA publishes its whole F32 partial tile
        // (cluster_decode.cuh); a single-CTA row tile uses shared memory.
        let plan = Self::plan_of(self.instance, live as u32)?;
        let bytes = poison::read(session, &self.workspace)?;
        let map = WriteMap::of(DType::F32).byte_prefix(plan.workspace_bytes as usize)?;
        poison::check(
            &format!("{label} partials"),
            &bytes,
            &map,
            Fill::Pattern(&[POISON]),
        )?;
        check_finite(
            &format!("{label} partials"),
            &bytes,
            4,
            0..plan.workspace_bytes as usize / 4,
        )?;
        images.push(bytes);
        Ok(images)
    }
}

/// Fails on the first NaN among `elements` of `bytes`, read as F32
/// (`element` 4) or BF16 (2). A written NaN such as 0x7fc0 differs from the
/// 0xff poison, so the write-map check alone passes it, and two arms that
/// both write it still compare equal.
fn check_finite(
    label: &str,
    bytes: &[u8],
    element: usize,
    elements: impl IntoIterator<Item = usize>,
) -> LabResult<()> {
    let nan = |i: usize| {
        let at = &bytes[i * element..(i + 1) * element];
        if element == 4 {
            f32::from_le_bytes(at.try_into().unwrap()).is_nan()
        } else {
            bf16_to_f32(u16::from_le_bytes([at[0], at[1]])).is_nan()
        }
    };
    match elements.into_iter().find(|&i| nan(i)) {
        Some(i) => Err(format!("{label}: element {i} is NaN")),
        None => Ok(()),
    }
}

fn first_difference(a: &[u8], b: &[u8], element: usize) -> Option<usize> {
    a.chunks(element)
        .zip(b.chunks(element))
        .position(|(x, y)| x != y)
}

/// Instances grouped by profile and family, which share a weight bank.
fn banks_of(instances: &[Instance]) -> Vec<(String, Vec<&Instance>)> {
    let mut groups: Vec<(String, Vec<&Instance>)> = Vec::new();
    for instance in instances {
        let key = format!("{}:{}", instance.profile().as_str(), family(instance));
        match groups.iter_mut().find(|g| g.0 == key) {
            Some(group) => group.1.push(instance),
            None => groups.push((key, vec![instance])),
        }
    }
    groups
}

fn live_counts(instance: &Instance) -> Vec<usize> {
    poison::live_counts(query(instance).m_capacity)
        .into_iter()
        .filter(|&live| live > 0)
        .map(|live| live as usize)
        .collect()
}

#[test]
fn dense_template_matches_served_entries() {
    let instances = instances();
    gpu_test(8 << 30, |session| {
        let sources = accumulation::QwenSources::from_env("long_4k")?;
        if sources.is_none() {
            kernel_lab::skip(
                "the real-weight cases: set CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_LAYER_DUMPS",
            )?;
        }
        let mut results = Vec::new();
        let mut failures = Vec::new();
        let mut covered = BTreeMap::new();
        for (index, (key, group)) in banks_of(&instances).into_iter().enumerate() {
            let mut banks = vec![(
                Inputs::Wide,
                Rc::new(Bank::wide(session, group[0], mix(0xde45e ^ index as u64))?),
            )];
            if let Some(sources) = &sources
                && let Some(bank) = Bank::real(session, sources, group[0])?
            {
                banks.push((Inputs::Real, Rc::new(bank)));
            }
            covered.insert(
                key.clone(),
                json!(banks.iter().map(|(inputs, bank)| json!({"inputs": inputs.name(), "note": bank.note})).collect::<Vec<_>>()),
            );
            for instance in group {
                let case = Case::new(session, instance)?;
                for (inputs, bank) in &banks {
                    for live in live_counts(instance) {
                        case.prepare(session, bank, live, index * 7 + live)?;
                        let reference = case.run(session, bank, Arm::Reference, live)?;
                        let template = case.run(session, bank, Arm::Template, live)?;
                        let names: Vec<String> = (0..case.outputs.len())
                            .map(|s| format!("output {s}"))
                            .chain(case.bottleneck.iter().map(|_| "bottleneck".into()))
                            .chain(std::iter::once("partials".into()))
                            .collect();
                        let mut differ = Vec::new();
                        for ((name, a), b) in names.iter().zip(&reference).zip(&template) {
                            if a != b {
                                let element = if name == "partials" { 4 } else { case.element };
                                differ.push(format!(
                                    "{name} (first element {:?})",
                                    first_difference(a, b, element)
                                ));
                            }
                        }
                        let s = instance.shape;
                        let result = json!({
                            "served": instance.served_symbol,
                            "reference": instance.reference_symbol,
                            "template": instance.symbol(),
                            "served_is_template": instance.serves_template(),
                            "profile": instance.profile().as_str(),
                            "capacity": case.capacity,
                            "live": live,
                            "inputs": inputs.name(),
                            "shape": {
                                "weights": format!("{:?}", s.weights),
                                "token_tile": s.token_tile,
                                "k_blocks": s.k_blocks,
                                "strips": s.strips,
                                "warp_blocks": s.warp_blocks,
                                "passes": s.passes,
                                "segments": s.segments,
                                "swiglu": s.swiglu,
                                "splits": s.splits(),
                            },
                            "compared_bytes": reference.iter().map(Vec::len).sum::<usize>(),
                            "bitwise": differ.is_empty(),
                        });
                        if !differ.is_empty() {
                            failures.push(format!(
                                "{} live {live} {}: {}",
                                instance.served_symbol,
                                inputs.name(),
                                differ.join(", ")
                            ));
                        }
                        eprintln!("{result}");
                        results.push(result);
                    }
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("instances".into(), json!(instances.len()));
        receipt.insert("executed_cases".into(), json!(results.len()));
        receipt.insert("cases".into(), json!(results));
        receipt.insert("failures".into(), json!(failures));
        receipt.insert("banks".into(), Value::Object(covered.into_iter().collect()));
        receipt.insert(
            "real_weights".into(),
            json!(if sources.is_some() { "Qwen3.8 families" } else { "skipped: CHOPINFER_QWEN38_CHECKPOINT or CHOPINFER_QWEN38_LAYER_DUMPS unset" }),
        );
        receipt.insert(
            "poisoned".into(),
            json!(["outputs and their row padding", "HC bottleneck", "cluster partials and slack", "activation rows past live (NaN)"]),
        );
        let path = session.receipt("dense_template_bitwise", receipt)?;
        eprintln!("receipt {}", path.display());
        if results.is_empty() {
            return Err("no bitwise case ran".into());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{} template cases differ:\n{}",
                failures.len(),
                failures.join("\n")
            ))
        }
    })
    .expect("dense cluster decode template bitwise equality");
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Alternates the reference (A) and the template (B), A B A B per
/// repetition, at each instance's capacity. Each timing is the median over
/// `ROUNDS` of a batch of back-to-back launches after one L2 flush, divided
/// by the batch. The events resolve about 1 us per batch on this GPU, so
/// 32 launches of a 3 us kernel moved its time in 1% steps: the batch holds
/// at least 32 launches and at least `MIN_BATCH_US` of work (from one
/// unrecorded 32-launch timing of the reference), at most `MAX_BATCH`. An
/// unrecorded repetition warms the clocks.
///
/// Each repetition gives two adjacent pairs (A1, B1) and (A2, B2). An
/// instance regresses when the median of its pair ratios B/A exceeds
/// `1 + REGRESSION`. The noise band is the median over repetitions of the
/// A1-to-A2 difference; above `NOISE_LIMIT` the instance is inconclusive and
/// the test fails asking for a rerun.
#[test]
#[ignore = "timing; run on an exclusive GPU with --ignored"]
fn dense_template_timing() {
    const REPETITIONS: usize = 7;
    const ROUNDS: usize = 15;
    const MIN_BATCH: usize = 32;
    const MAX_BATCH: usize = 1024;
    const MIN_BATCH_US: f32 = 400.0;
    const REGRESSION: f32 = 0.01;
    const NOISE_LIMIT: f32 = 0.02;
    let instances = instances();
    gpu_test(8 << 30, |session| {
        let mut rows = Vec::new();
        let (mut slower, mut inconclusive) = (Vec::new(), Vec::new());
        for (index, (_, group)) in banks_of(&instances).into_iter().enumerate() {
            let bank = Bank::wide(session, group[0], mix(0x7133 ^ index as u64))?;
            for instance in group {
                let case = Case::new(session, instance)?;
                let live = case.capacity;
                case.prepare(session, &bank, live, 0)?;
                let timed = |side: Arm, batch: usize| -> LabResult<f32> {
                    let timing = time(session, 3, ROUNDS, |_| {
                        for _ in 0..batch {
                            case.launch(session, &bank, side, live as u32, false)?;
                        }
                        Ok(())
                    })?;
                    Ok(timing.median_ms * 1000.0 / batch as f32)
                };
                let batch = ((MIN_BATCH_US / timed(Arm::Reference, MIN_BATCH)?).ceil() as usize)
                    .clamp(MIN_BATCH, MAX_BATCH);
                let arm = |side: Arm| timed(side, batch);
                for side in [Arm::Reference, Arm::Template, Arm::Reference, Arm::Template] {
                    arm(side)?;
                }
                let (mut a, mut b, mut noise, mut pairs) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..REPETITIONS {
                    let a1 = arm(Arm::Reference)?;
                    let b1 = arm(Arm::Template)?;
                    let a2 = arm(Arm::Reference)?;
                    let b2 = arm(Arm::Template)?;
                    noise.push((a1 - a2).abs() / a1.min(a2));
                    pairs.extend([b1 / a1, b2 / a2]);
                    a.extend([a1, a2]);
                    b.extend([b1, b2]);
                }
                let range = |v: &[f32]| {
                    (
                        v.iter().copied().fold(f32::MAX, f32::min),
                        v.iter().copied().fold(0.0, f32::max),
                    )
                };
                let ((a_min, a_max), (b_min, b_max)) = (range(&a), range(&b));
                let (pair_min, pair_max) = range(&pairs);
                let band = range(&noise).1;
                let (a_median, b_median) = (median(&mut a), median(&mut b));
                let band_median = median(&mut noise);
                let ratio = median(&mut pairs);
                rows.push(json!({
                    "served": instance.served_symbol,
                    "reference": instance.reference_symbol,
                    "template": instance.symbol(),
                    "profile": instance.profile().as_str(),
                    "live": live,
                    "launches_per_timing": batch,
                    "reference_us": {"median": a_median, "min": a_min, "max": a_max},
                    "template_us": {"median": b_median, "min": b_min, "max": b_max},
                    "template_over_reference": ratio,
                    "pair_ratio": {"median": ratio, "min": pair_min, "max": pair_max},
                    "a_to_a_noise": {"median": band_median, "max": band},
                }));
                eprintln!(
                    "{:<52} A {a_median:8.2} us [{a_min:.2}, {a_max:.2}]  B {b_median:8.2} us [{b_min:.2}, {b_max:.2}]  B/A {ratio:.4}  A-A noise median {:.2}% max {:.2}%",
                    instance.served_symbol,
                    100.0 * band_median,
                    100.0 * band,
                );
                if band_median > NOISE_LIMIT {
                    inconclusive.push(format!(
                        "{} A-A noise median {:.2}%",
                        instance.served_symbol,
                        100.0 * band_median
                    ));
                } else if ratio > 1.0 + REGRESSION {
                    slower.push(format!("{} median pair B/A {ratio:.4}", instance.served_symbol));
                }
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("repetitions".into(), json!(REPETITIONS));
        receipt.insert("rounds_per_timing".into(), json!(ROUNDS));
        receipt.insert("order".into(), json!("one unrecorded A B A B, then A B A B per repetition; L2 flushed before each timed batch; microseconds per launch"));
        receipt.insert(
            "batch".into(),
            json!({"min_launches": MIN_BATCH, "max_launches": MAX_BATCH, "min_batch_us": MIN_BATCH_US}),
        );
        receipt.insert("instances".into(), json!(rows));
        receipt.insert("executed_instances".into(), json!(rows.len()));
        receipt.insert(
            "gate".into(),
            json!({"median_pair_ratio_max": 1.0 + REGRESSION, "a_to_a_noise_median_max": NOISE_LIMIT}),
        );
        receipt.insert("slower".into(), json!(slower));
        receipt.insert("inconclusive_rerun".into(), json!(inconclusive));
        let path = session.receipt("dense_template_timing", receipt)?;
        eprintln!("receipt {}", path.display());
        if rows.is_empty() {
            Err("no timing instance ran".into())
        } else if !slower.is_empty() {
            Err(format!("template slower:\n{}", slower.join("\n")))
        } else if !inconclusive.is_empty() {
            Err(format!("inconclusive, rerun:\n{}", inconclusive.join("\n")))
        } else {
            Ok(())
        }
    })
    .expect("dense cluster decode template timing");
}

mod gemm_references {
    use kernel_lab::catalogs;
    include!("../gemm_references.rs");
}

/// The frozen references kernel-lab compiles (`gemm_references.rs`) keep
/// the served entries' ABI, threads and minimum blocks, one per entry of a
/// model that serves the template, and run the old kernel.
#[test]
fn references_keep_the_served_abi() {
    let instances = cluster::lab_instances(kernel_lab::catalogs());
    let specs = gemm_references::gemm_cluster_reference_specs();
    let entries: BTreeMap<_, _> = specs
        .iter()
        .flat_map(|spec| {
            spec.entries
                .iter()
                .map(move |e| (e.symbol.clone(), (spec, e)))
        })
        .collect();
    let swapped: Vec<_> = instances.iter().filter(|i| i.serves_template()).collect();
    assert_eq!(entries.len(), swapped.len());
    assert_eq!(
        swapped.len(),
        94,
        "Qwen3.8's _c{{M}} buckets and HC producers"
    );
    for instance in swapped {
        let (spec, entry) = entries[&instance.reference_symbol];
        assert_eq!(spec.name, instance.reference_cubin);
        assert_eq!(entry.params, instance.template.params);
        assert_eq!(entry.max_threads, instance.template.max_threads);
        assert_eq!(entry.min_blocks, instance.template.min_blocks);
        assert!(
            entry
                .body
                .contains("chop::gemm::qwen38::run_cluster_decode<")
        );
        assert!(instance.template.body.contains("chop::gemm::cluster::run<"));
    }
}

/// Canonical BF16 and F32 NaNs pass the poison write-map check (they differ
/// from the 0xff fill), so `check_finite` must reject them in every image a
/// case compares: outputs, the HC bottleneck and the partials.
#[test]
fn written_nans_fail_the_finite_check() {
    for (element, dtype, nan, one) in [
        (
            2,
            DType::Bf16,
            0x7fc0_u32.to_le_bytes()[..2].to_vec(),
            0x3f80_u32.to_le_bytes()[..2].to_vec(),
        ),
        (
            4,
            DType::F32,
            0x7fc0_0000_u32.to_le_bytes().to_vec(),
            1.0_f32.to_le_bytes().to_vec(),
        ),
    ] {
        let mut bytes = [one.clone(), nan, one].concat();
        bytes.extend(vec![POISON; element]);
        let map = WriteMap::of(dtype).prefix(3).unwrap();
        poison::check("image", &bytes, &map, Fill::Pattern(&[POISON]))
            .expect("the write map accepts a written NaN");
        let error = check_finite("image", &bytes, element, 0..3).unwrap_err();
        assert!(error.contains("element 1 is NaN"), "{error}");
        check_finite("image", &bytes, element, [0, 2]).expect("finite elements pass");
    }
}

/// CPU check of the case geometry the GPU tests rely on.
#[test]
fn every_instance_has_a_family_and_live_counts() {
    let instances = cluster::lab_instances(kernel_lab::catalogs());
    assert!(!instances.is_empty());
    for instance in &instances {
        let family = family(instance);
        let bucket = format!(
            "_{}{}_v1",
            if family.contains("_mimo") { 'm' } else { 'c' },
            query(instance).m_capacity
        );
        let base = match &instance.served {
            Served::FusedHcPrecompute(entry) => &entry.symbol,
            _ => &instance.served_symbol,
        };
        assert_eq!(format!("{family}{bucket}"), *base);
        let counts = live_counts(instance);
        let capacity = query(instance).m_capacity as usize;
        assert_eq!(counts[0], capacity);
        assert!(counts.contains(&1));
        assert!(counts.iter().all(|&live| (1..=capacity).contains(&live)));
    }
    let hc = instances
        .iter()
        .filter(|i| matches!(i.served, Served::FusedHcPrecompute(_)))
        .map(family)
        .collect::<Vec<_>>();
    assert_eq!(
        hc,
        [
            "chop_gemm_qwen38_hc_in",
            "chop_gemm_qwen38_hc_down",
            "chop_gemm_qwen38_hc_in",
            "chop_gemm_qwen38_hc_down"
        ]
    );
    // Every bank's instances share their weights' shape.
    for (key, group) in banks_of(&instances) {
        let rows = segment_rows(group[0]);
        let first = query(group[0]);
        for instance in group {
            let q = query(instance);
            assert_eq!(segment_rows(instance), rows, "{key}");
            assert_eq!(
                (q.n, q.k, q.weight),
                (first.n, first.k, first.weight),
                "{key}"
            );
        }
    }
}
