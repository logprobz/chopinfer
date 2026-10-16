//! Medium-M dense GEMM (`gemm::mid`) cases. Every entry is checked bit for
//! bit against the fragment prefill entry it stands in for, at the same live
//! M, eagerly, repeated and in a graph, against the f64 oracle on sampled
//! columns, and for untouched output rows past live M. FP8 weights get a
//! different scale in every 128 x 128 block, and the oracle reads their
//! exact dequantized values. The benchmark times both in 48-layer GEMM-only
//! graphs with the DRAM and MMA rooflines.
//!
//! Environment: `CHOPINFER_KERNEL_MID_FILTER` keeps references whose symbol
//! contains any comma-separated part; `CHOPINFER_KERNEL_MID_M` replaces the
//! row counts; `CHOPINFER_KERNEL_BENCH=1` enables the timing case.

use super::*;
use kernel_core::contract::{ArgumentList, LaunchPlan};
use kernels_sm120::gemm::mid;

const MID_OUTPUT: u64 = 20;
const MID_PARTIAL: u64 = 21;
const MID_SEMAPHORES: u64 = 22;
/// Rows seeded in the activation (the rest of the capacity is zero); rows
/// past a case's live M hold values the outputs must not depend on.
const SEEDED_ROWS: usize = 256;
/// Every row count the correctness case runs; the 64-token kernel takes
/// those up to 64.
const CORRECTNESS_ROWS: [u32; 5] = [33, 40, 63, 64, 65];
const BENCH_ROWS: [u32; 2] = [33, 64];
/// Shapes timed by default: every MiMo and Qwen3.8 TP4 projection a verify
/// step or a wide decode runs, one head each.
const BENCH_REFERENCES: [&str; 19] = [
    "chop_gemm_mimo_qkv_ga_prefill_v1",
    "chop_gemm_mimo_qkv_swa_prefill_v1",
    "chop_gemm_mimo_o_prefill_v1",
    "chop_gemm_mimo_mlp_gate_up_prefill_v1",
    "chop_gemm_mimo_mlp_down_prefill_v1",
    "chop_gemm_mimo_head_f32_prefill_v1",
    "chop_gemm_mimo_dflash_qkv_prefill_v1",
    "chop_gemm_mimo_dflash_o_prefill_v1",
    "chop_gemm_qwen38_lin_qkv_prefill_v1",
    "chop_gemm_qwen38_lin_z_prefill_v1",
    "chop_gemm_qwen38_out_prefill_v1",
    "chop_gemm_qwen38_attn_qkv_prefill_v1",
    "chop_gemm_qwen38_index_qk_prefill_v1",
    "chop_gemm_qwen38_router_prefill_v1",
    "chop_gemm_qwen38_hc_down_prefill_v1",
    "chop_gemm_qwen38_ple_key_prefill_v1",
    "chop_gemm_qwen38_shared_gate_up_prefill_v1",
    "chop_gemm_qwen38_shared_down_prefill_v1",
    "chop_gemm_qwen38_head_f32_prefill_v1",
];
fn filtered(symbols: impl IntoIterator<Item = String>) -> Vec<String> {
    let filter = std::env::var("CHOPINFER_KERNEL_MID_FILTER").ok();
    symbols
        .into_iter()
        .filter(|symbol| {
            filter
                .as_deref()
                .is_none_or(|f| f.split(',').any(|part| symbol.contains(part)))
        })
        .collect()
}

fn rows(default: &[u32]) -> Vec<u32> {
    std::env::var("CHOPINFER_KERNEL_MID_M")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|m| m.parse().expect("CHOPINFER_KERNEL_MID_M is a list of u32"))
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

/// A non-power-of-two amplitude for FP8 weight rows of 64-row group
/// `row_group` in 128-K block `k_block`, distinct between neighbours in
/// either direction.
fn block_amplitude(row_group: usize, k_block: usize) -> f32 {
    1.0 + ((row_group * 5 + k_block * 3) % 13) as f32 * 0.37
}

fn counts(m: u32) -> RuntimeCounts {
    RuntimeCounts {
        m,
        batch: m,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    }
}

/// One reference shape on the device: weights, scales and the activation at
/// the reference's capacity, and output, partial and semaphore buffers for
/// the prefill entry and for the medium entries.
struct Shape {
    reference: GemmVariant,
    query: GemmQuery,
    entries: Vec<mid::Entry>,
    buffers: BTreeMap<u64, DeviceBuffer>,
    /// The f64 oracle over `columns` for all seeded rows, when requested.
    oracle: Option<(Vec<usize>, OracleOutput)>,
    /// Wide-range BF16 weight [n][k] and activation [SEEDED_ROWS][k] bits.
    wide: Option<(Vec<u16>, Vec<u16>)>,
}

impl Shape {
    fn output_n(&self) -> usize {
        let n = self.query.n as usize;
        if self.query.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        }
    }

    fn output_bytes(&self) -> usize {
        if self.query.output == DType::F32 {
            4
        } else {
            2
        }
    }

    fn output_dtype(&self) -> DType {
        if self.query.output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        }
    }

    /// `wide` replaces the grid-valued activations, and BF16 weights, with
    /// values over 13 binades whose F32 sums round, so a different
    /// accumulation order changes output bits.
    fn prepare(session: &Session, reference: &str, oracle: bool, wide: bool) -> LabResult<Self> {
        let variant = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|item| item.symbol == reference)
            .ok_or_else(|| format!("missing GEMM catalog symbol {reference}"))?;
        let q = query(&variant);
        let entries = mid::catalog(kernel_lab::catalogs())
            .into_iter()
            .filter(|entry| entry.reference == reference)
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Err(format!("{reference} has no medium-M entry"));
        }
        let (n, k) = (q.n as usize, q.k as usize);
        let capacity = q.m_capacity as usize;
        let output_n = if q.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        };
        // Weight rows the oracle reads: the sampled columns, and for SwiGLU
        // their up rows.
        let columns = lean_columns(output_n);
        let weight_rows = if q.epilogues.swiglu.is_some() {
            columns
                .iter()
                .copied()
                .chain(columns.iter().map(|c| c + output_n))
                .collect::<Vec<_>>()
        } else {
            columns.clone()
        };
        let wide_weight = (wide && q.weight == WeightFormat::Bf16)
            .then(|| accumulation::probe::wide_rows(n, k, WEIGHT_SEED));
        let (weight_bytes, scales, b) = if let Some(bits) = &wide_weight {
            let b = weight_rows
                .iter()
                .flat_map(|&r| &bits[r * k..(r + 1) * k])
                .map(|&x| f64::from(bf16_to_f32(x)))
                .collect::<Vec<_>>();
            let packed = FragmentWeightLayout {
                rows: n,
                k,
                format: FragmentWeightFormat::Bf16,
            }
            .pack(
                &bits
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )
            .map_err(|error| error.to_string())?
            .bytes;
            (packed, None, b)
        } else if q.weight == WeightFormat::Bf16 {
            let cached = lean_weight(n, k, true)?;
            let b = weight_rows
                .iter()
                .flat_map(|&r| &cached.bits[r * k..(r + 1) * k])
                .map(|&x| f64::from(bf16_to_f32(x)))
                .collect::<Vec<_>>();
            (cached.data.clone(), None, b)
        } else {
            // Every 64-row group of every 128-K block gets its own
            // amplitude, so each F32 scale (per 64 or 128 rows) differs from
            // its neighbours' along N and K (the seeded values alone give
            // every block the same scale), including the gate and up halves
            // and the last 64-row N tail.
            let mut values = seeded_values(n * k, WEIGHT_SEED);
            for (index, value) in values.iter_mut().enumerate() {
                *value *= block_amplitude(index / k / 64, index % k / 128);
            }
            let mut w = weight(&values, n, k, q.weight, true)?;
            // The kernel converts each E4M3 code to BF16 exactly and applies
            // the F32 block scale to the F32 block sum, so the oracle reads
            // the exact dequantized weights.
            let b = weight_rows
                .iter()
                .flat_map(|&r| &w.decoded[r * k..(r + 1) * k])
                .copied()
                .collect::<Vec<_>>();
            w.data = FragmentWeightLayout {
                rows: n,
                k,
                format: FragmentWeightFormat::Fp8E4M3,
            }
            .pack(&w.data)
            .map_err(|error| error.to_string())?
            .bytes;
            (w.data, w.scales, b)
        };
        let wide_activation =
            wide.then(|| accumulation::probe::wide_rows(SEEDED_ROWS, k, 0x4143_5449_5641_5445));
        let mut raw = vec![0.0_f32; capacity * k];
        match &wide_activation {
            Some(bits) => raw[..SEEDED_ROWS * k]
                .iter_mut()
                .zip(bits)
                .for_each(|(value, &bits)| *value = bf16_to_f32(bits)),
            None => raw[..SEEDED_ROWS * k]
                .copy_from_slice(&seeded_values(SEEDED_ROWS * k, 0x4143_5449_5641_5445)),
        }
        let a = activation(&raw, capacity, k, ActivationFormat::Bf16, false)?;
        let oracle = if oracle {
            let a = raw[..SEEDED_ROWS * k]
                .iter()
                .map(|&v| f64::from(bf16_to_f32(f32_to_bf16(v))))
                .collect::<Vec<_>>();
            let out = oracle::gemm(
                &a,
                &b,
                GemmShape {
                    m: SEEDED_ROWS,
                    n: weight_rows.len(),
                    k,
                    experts: 1,
                },
                &[],
                GemmEpilogue {
                    residual: None,
                    swiglu: q.epilogues.swiglu,
                },
            )
            .map_err(|e| e.to_string())?;
            Some((columns, out))
        } else {
            None
        };
        let reference_plan =
            gemm::plan(&variant.key, counts(q.m_capacity.min(256))).map_err(|e| e.to_string())?;
        let semaphore_bytes = (reference_plan.workspace_bytes as usize).max(1 << 16);
        let output_bytes = SEEDED_ROWS * output_n * if q.output == DType::F32 { 4 } else { 2 };
        let mut buffers = BTreeMap::new();
        buffers.insert(WEIGHT, upload(session, &weight_bytes)?);
        buffers.insert(ACTIVATION, upload(session, &a.data)?);
        if let Some(scales) = &scales {
            buffers.insert(WEIGHT_SCALES, upload(session, scales)?);
        }
        for (output, partial, semaphores) in [
            (OUTPUT, PARTIAL, SEMAPHORES),
            (MID_OUTPUT, MID_PARTIAL, MID_SEMAPHORES),
        ] {
            buffers.insert(output, session.allocate(output_bytes)?);
            buffers.insert(
                partial,
                session.allocate(reference_plan.workspace_bytes as usize)?,
            );
            buffers.insert(semaphores, upload(session, &vec![0; semaphore_bytes])?);
        }
        Ok(Self {
            reference: variant,
            query: q,
            entries,
            buffers,
            oracle,
            wide: wide_weight.zip(wide_activation),
        })
    }

    fn reference_launch(&self, m: u32) -> LabResult<(LaunchPlan, ArgumentList)> {
        let key = &self.reference.key;
        let maps = gemm::tensor_maps(key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
        let plan = gemm::plan(key, counts(m)).map_err(|e| e.to_string())?;
        let args = gemm::arguments(
            key,
            counts(m),
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: self
                    .buffers
                    .contains_key(&WEIGHT_SCALES)
                    .then_some(WEIGHT_SCALES),
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .map_err(|e| e.to_string())?;
        Ok((plan, args))
    }

    fn mid_launch(&self, entry: &mid::Entry, m: u32) -> LabResult<(LaunchPlan, ArgumentList)> {
        let plan = entry.plan(m).map_err(|e| e.to_string())?;
        let args = entry
            .arguments(
                m,
                entry.activation_map(ACTIVATION, TensorMapL2Promotion::B128),
                mid::Buffers {
                    activation: ACTIVATION,
                    weight: WEIGHT,
                    weight_scales: self
                        .buffers
                        .contains_key(&WEIGHT_SCALES)
                        .then_some(WEIGHT_SCALES),
                    output: MID_OUTPUT,
                    partial: MID_PARTIAL,
                    semaphores: MID_SEMAPHORES,
                },
            )
            .map_err(|e| e.to_string())?;
        Ok((plan, args))
    }

    fn reference_function(&self, session: &Session) -> LabResult<Function> {
        let module = session.module(&format!("gemm_{}_sm120", self.reference.profile.tag()))?;
        let function = module
            .function(&self.reference.symbol)
            .map_err(|e| e.to_string())?;
        function
            .set_max_dynamic_shared_memory(self.reference_launch(1)?.0.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
        Ok(function)
    }

    /// Launches with the shape's buffers, or a graph layer's in their place.
    fn launch(
        &self,
        session: &Session,
        function: &Function,
        stream: &Stream,
        launch: &(LaunchPlan, ArgumentList),
        layer: Option<&GraphLayer>,
    ) -> LabResult<()> {
        let mut buffers = self
            .buffers
            .iter()
            .map(|(&id, buffer)| (id, buffer))
            .collect::<BTreeMap<_, _>>();
        if let Some(layer) = layer {
            buffers.insert(WEIGHT, &layer.weight);
            for (output, partial, semaphores) in [
                (OUTPUT, PARTIAL, SEMAPHORES),
                (MID_OUTPUT, MID_PARTIAL, MID_SEMAPHORES),
            ] {
                buffers.insert(output, &layer.output);
                buffers.insert(partial, &layer.partial);
                buffers.insert(semaphores, &layer.semaphores);
            }
        }
        launch::launch(
            &session.context,
            function,
            stream,
            launch.0,
            &launch.1,
            &Bindings { buffers },
        )
    }

    /// The first `m` output rows as F32 values.
    fn download(&self, session: &Session, buffer: &DeviceBuffer, m: u32) -> LabResult<Vec<f32>> {
        let len = m as usize * self.output_n();
        if self.query.output == DType::F32 {
            session.download_f32(buffer, len)
        } else {
            Ok(session
                .download_bf16(buffer, len)?
                .into_iter()
                .map(bf16_to_f32)
                .collect())
        }
    }

    /// The first `m` output rows of a launch on poison are written. Only the
    /// medium entries promise that rows past live M stay untouched; their
    /// replay checks enforce that with a write map.
    fn live_written(
        &self,
        session: &Session,
        buffer: &DeviceBuffer,
        m: u32,
        label: &str,
    ) -> LabResult<()> {
        let bytes = poison::read(session, buffer)?;
        let live = m as usize * self.output_n() * self.output_bytes();
        poison::check(
            label,
            &bytes[..live],
            &WriteMap::of(self.output_dtype()).byte_prefix(live)?,
            Fill::Pattern(&[POISON]),
        )
    }

    /// Poisoned checks of a medium entry at live M: rows below M written,
    /// rows past it untouched, and its F32 split partials as scratch. The
    /// self-resetting semaphores are never touched.
    fn mid_replay<'a>(&'a self, session: &'a Session, m: u32) -> LabResult<Replay<'a>> {
        let live = m as usize * self.output_n() * self.output_bytes();
        Ok(Replay::new(session)
            .output(
                "mid output",
                &self.buffers[&MID_OUTPUT],
                WriteMap::of(self.output_dtype()).byte_prefix(live)?,
            )
            .scratch("mid partials", &self.buffers[&MID_PARTIAL]))
    }

    #[allow(clippy::too_many_arguments)]
    fn bitwise(
        &self,
        session: &Session,
        label: &str,
        symbol: &str,
        m: u32,
        expected: &[f32],
        actual: &[f32],
        evidence: &Evidence,
    ) -> LabResult<()> {
        let mismatches = expected
            .iter()
            .zip(actual)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        let first = expected
            .iter()
            .zip(actual)
            .position(|(left, right)| left.to_bits() != right.to_bits());
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(label));
        receipt.insert("reference".into(), json!(self.reference.symbol));
        receipt.insert("symbol".into(), json!(symbol));
        receipt.insert("m".into(), json!(m));
        receipt.insert("elements".into(), json!(expected.len()));
        receipt.insert("mismatches".into(), json!(mismatches));
        receipt.insert("correct".into(), json!(mismatches == 0));
        evidence.record(&mut receipt);
        if let Some(index) = first {
            receipt.insert(
                "first_mismatch".into(),
                json!({"index": index, "expected": expected[index], "actual": actual[index]}),
            );
        }
        session.receipt("gemm", receipt)?;
        if mismatches != 0 {
            return Err(format!(
                "{label}: {symbol} at M={m} differs from {} in {mismatches} of {} outputs (first {first:?})",
                self.reference.symbol,
                expected.len()
            ));
        }
        Ok(())
    }

    fn oracle_check(
        &self,
        session: &Session,
        symbol: &str,
        m: u32,
        actual: &[f32],
    ) -> LabResult<()> {
        let Some((columns, oracle)) = &self.oracle else {
            return Ok(());
        };
        let output_n = self.output_n();
        let picked = (0..m as usize)
            .flat_map(|t| columns.iter().map(move |&c| actual[t * output_n + c]))
            .collect::<Vec<_>>();
        let len = picked.len();
        let reference = OracleOutput {
            values: oracle.values[..len].to_vec(),
            sum_abs_products: oracle.sum_abs_products[..len].to_vec(),
            reduction_lengths: oracle.reduction_lengths[..len].to_vec(),
        };
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("dense_gemm_mid_oracle"));
        receipt.insert("symbol".into(), json!(symbol));
        receipt.insert("reference".into(), json!(self.reference.symbol));
        receipt.insert("profile".into(), json!(self.reference.profile.as_str()));
        receipt.insert("m".into(), json!(m));
        receipt.insert("sampled_outputs".into(), json!(len));
        let gate = check_output(&picked, &reference, self.query.output, &mut receipt);
        session.receipt("gemm", receipt.clone())?;
        gate.map_err(|error| {
            format!(
                "{symbol} for {} at M={m}: {error}; worst={}",
                self.reference.symbol, receipt["worst_element"]
            )
        })
    }
}

fn mid_function(
    session: &Session,
    module: &str,
    symbol: &str,
    entry: &mid::Entry,
) -> LabResult<Function> {
    let function = session
        .module(module)?
        .function(symbol)
        .map_err(|e| e.to_string())?;
    function
        .set_max_dynamic_shared_memory(entry.kernel.dynamic_shared_bytes())
        .map_err(|e| e.to_string())?;
    Ok(function)
}

/// Every medium entry at every row count it takes: bitwise equal to its
/// reference prefill entry eagerly, when repeated, and in a graph after a
/// PDL producer; within the numerical bound of the f64 oracle.
#[test]
fn gemm_mid_matches_prefill() {
    watched_gpu_test(8 * 1024 * 1024 * 1024, 1100, |session| {
        let references = filtered(
            mid::catalog(kernel_lab::catalogs())
                .into_iter()
                .map(|entry| entry.reference)
                .fold(Vec::<String>::new(), |mut all, r| {
                    if !all.contains(&r) {
                        all.push(r);
                    }
                    all
                }),
        );
        if references.is_empty() {
            return Err("medium GEMM filter matched no reference".into());
        }
        let producer_module = session.module("probe_sm120")?;
        let producer = producer_module
            .function("chop_probe_pdl_first_v1")
            .map_err(|error| error.to_string())?;
        let producer_config = launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?;
        let stamps = session.allocate(24)?;
        let row_counts = rows(&CORRECTNESS_ROWS);
        let mut checked = 0;
        for reference in references {
            eprintln!("medium GEMM correctness: {reference}");
            let shape = Shape::prepare(session, &reference, true, false)?;
            let prefill = shape.reference_function(session)?;
            for &m in &row_counts {
                let launch = shape.reference_launch(m)?;
                poison::poison(
                    session,
                    &[&shape.buffers[&OUTPUT], &shape.buffers[&PARTIAL]],
                )?;
                shape.launch(session, &prefill, &session.stream, &launch, None)?;
                shape.live_written(
                    session,
                    &shape.buffers[&OUTPUT],
                    m,
                    &format!("{reference} reference at M={m}"),
                )?;
                let expected = shape.download(session, &shape.buffers[&OUTPUT], m)?;
                for entry in shape.entries.iter().filter(|e| m <= e.max_rows()) {
                    let symbol = entry.kernel.symbol();
                    let function = mid_function(session, &mid::cubin_name(None), &symbol, entry)?;
                    let launch = shape.mid_launch(entry, m)?;
                    let output = &shape.buffers[&MID_OUTPUT];
                    let labelled = |error: String| format!("{symbol} at M={m}: {error}");
                    let run =
                        |stream: &Stream| shape.launch(session, &function, stream, &launch, None);
                    let mut checks = shape.mid_replay(session, m)?;
                    checks.eager(|| run(&session.stream)).map_err(labelled)?;
                    let eager = shape.download(session, output, m)?;
                    shape.bitwise(
                        session,
                        "dense_gemm_mid_prefill_bitwise",
                        &symbol,
                        m,
                        &expected,
                        &eager,
                        checks.evidence(),
                    )?;
                    shape.oracle_check(session, &symbol, m, &eager)?;
                    checks.relaunch(|| run(&session.stream)).map_err(labelled)?;
                    let repeat = shape.download(session, output, m)?;
                    shape.bitwise(
                        session,
                        "dense_gemm_mid_repeat_bitwise",
                        &symbol,
                        m,
                        &expected,
                        &repeat,
                        checks.evidence(),
                    )?;
                    let graph = session.capture(|stream| {
                        producer
                            .launch(
                                stream,
                                producer_config,
                                &[KernelArgument::Buffer(&stamps), KernelArgument::U64(1000)],
                            )
                            .map_err(|error| error.to_string())?;
                        run(stream)
                    })?;
                    checks.graph(&graph).map_err(labelled)?;
                    let replay = shape.download(session, output, m)?;
                    shape.bitwise(
                        session,
                        "dense_gemm_mid_pdl_graph_bitwise",
                        &symbol,
                        m,
                        &expected,
                        &replay,
                        checks.evidence(),
                    )?;
                    checked += 1;
                }
            }
        }
        eprintln!("medium GEMM correctness: {checked} entry and row cases bitwise equal");
        Ok(())
    })
    .expect("medium GEMM entries match the prefill entries");
}

/// Every medium entry against its prefill reference on wide-range inputs,
/// eagerly and against the f64 oracle, at the rows it takes and one past.
/// For BF16 references the probe also reports the share of
/// outputs whose bits differ between one MMA chain over K and 128-K runs on
/// the same inputs: the checks see an accumulation-order change.
#[test]
fn gemm_mid_matches_prefill_wide() {
    watched_gpu_test(8 * 1024 * 1024 * 1024, 1100, |session| {
        let references = filtered(
            mid::catalog(kernel_lab::catalogs())
                .into_iter()
                .map(|entry| entry.reference)
                .fold(Vec::<String>::new(), |mut all, r| {
                    if !all.contains(&r) {
                        all.push(r);
                    }
                    all
                }),
        );
        let probe = session.module(accumulation::probe::PROBE)?;
        let row_counts = rows(&[33, 64, 65]);
        let (mut checked, mut sensitivity) = (0, Vec::new());
        for reference in references {
            let shape = Shape::prepare(session, &reference, true, true)?;
            if let Some((weight, activation)) = &shape.wide {
                let (n, k) = (shape.query.n as usize, shape.query.k as usize);
                let problem = accumulation::probe::Problem::upload(
                    session,
                    weight,
                    activation,
                    n,
                    SEEDED_ROWS,
                    k,
                )?;
                let long =
                    problem.run(session, &probe, accumulation::probe::Scheme::Chain(None))?;
                let short = problem.run(
                    session,
                    &probe,
                    accumulation::probe::Scheme::Chain(Some(128)),
                )?;
                let differ = long
                    .iter()
                    .zip(&short)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                eprintln!(
                    "{reference}: {differ} of {} outputs differ between one chain and 128-K runs",
                    long.len()
                );
                sensitivity
                    .push(json!({"reference": reference, "outputs": long.len(), "differ": differ}));
            }
            let prefill = shape.reference_function(session)?;
            for &m in &row_counts {
                let launch = shape.reference_launch(m)?;
                poison::poison(
                    session,
                    &[&shape.buffers[&OUTPUT], &shape.buffers[&PARTIAL]],
                )?;
                shape.launch(session, &prefill, &session.stream, &launch, None)?;
                shape.live_written(
                    session,
                    &shape.buffers[&OUTPUT],
                    m,
                    &format!("{reference} wide reference at M={m}"),
                )?;
                let expected = shape.download(session, &shape.buffers[&OUTPUT], m)?;
                for entry in shape.entries.iter().filter(|e| m <= e.max_rows()) {
                    let symbol = entry.kernel.symbol();
                    let function = mid_function(session, &mid::cubin_name(None), &symbol, entry)?;
                    let launch = shape.mid_launch(entry, m)?;
                    let output = &shape.buffers[&MID_OUTPUT];
                    let mut checks = shape.mid_replay(session, m)?;
                    checks
                        .eager(|| shape.launch(session, &function, &session.stream, &launch, None))
                        .map_err(|error| format!("{symbol} wide at M={m}: {error}"))?;
                    let eager = shape.download(session, output, m)?;
                    shape.bitwise(
                        session,
                        "dense_gemm_mid_prefill_wide_bitwise",
                        &symbol,
                        m,
                        &expected,
                        &eager,
                        checks.evidence(),
                    )?;
                    shape.oracle_check(session, &symbol, m, &eager)?;
                    checked += 1;
                }
            }
        }
        session.receipt(
            "gemm",
            BTreeMap::from([
                ("case".into(), json!("dense_gemm_mid_wide_sensitivity")),
                ("checked".into(), json!(checked)),
                ("chain_k_vs_128".into(), json!(sensitivity)),
            ]),
        )?;
        eprintln!("medium GEMM wide inputs: {checked} entry and row cases bitwise equal");
        Ok(())
    })
    .expect("medium GEMM entries match the prefill entries on wide inputs");
}

struct GraphTimes {
    us: f64,
}

fn graph_time(
    session: &Session,
    shape: &Shape,
    function: &Function,
    launch: &(LaunchPlan, ArgumentList),
    layers: &[GraphLayer],
) -> LabResult<GraphTimes> {
    let count = layers.len() + 1;
    let graph = session.capture(|stream| {
        for layer in 0..count {
            shape.launch(
                session,
                function,
                stream,
                launch,
                layer.checked_sub(1).map(|index| &layers[index]),
            )?;
        }
        Ok(())
    })?;
    let timing = time(session, 2, 5, |stream| {
        graph.launch(stream).map_err(|error| error.to_string())
    })?;
    Ok(GraphTimes {
        us: f64::from(timing.median_ms) * 1000.0 / count as f64,
    })
}

/// 48-layer GEMM-only graphs (12 for heads) of each reference and its
/// medium entry at 33 and 64 rows, with distinct weights, outputs and
/// workspaces per layer, against the DRAM and BF16 MMA rooflines.
#[test]
fn gemm_mid_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("medium GEMM benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    watched_gpu_test(24 * 1024 * 1024 * 1024, 1100, |session| {
        let references = filtered(BENCH_REFERENCES.map(String::from));
        let row_counts = rows(&BENCH_ROWS);
        let (dram_gbps, dram_source) = peak(session, "bf16", true);
        let (mma_tflops, mma_source) = peak(session, "bf16", false);
        for reference in references {
            let shape = Shape::prepare(session, &reference, false, false)?;
            let head = reference.contains("_head_");
            let layer_count = if head { 12 } else { 48 };
            let weight = &shape.buffers[&WEIGHT];
            let mut layers = Vec::with_capacity(layer_count - 1);
            for _ in 1..layer_count {
                let copy = session.allocate(weight.byte_len())?;
                copy.copy_range_from_async(0, weight, 0, weight.byte_len(), &session.stream)
                    .map_err(|error| error.to_string())?;
                layers.push(GraphLayer {
                    weight: copy,
                    output: session.allocate(shape.buffers[&OUTPUT].byte_len())?,
                    partial: session.allocate(shape.buffers[&PARTIAL].byte_len())?,
                    semaphores: upload(
                        session,
                        &vec![0; shape.buffers[&SEMAPHORES].byte_len()],
                    )?,
                });
            }
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let prefill = shape.reference_function(session)?;
            let q = shape.query;
            for &m in &row_counts {
                let weight_bytes = weight.byte_len() as f64;
                let dram_bytes = weight_bytes
                    + f64::from(m) * f64::from(q.k) * 2.0
                    + (m as usize * shape.output_n() * shape.output_bytes()) as f64;
                let dram_us = dram_bytes / (dram_gbps * 1e3);
                let mma_us = 2.0 * f64::from(m) * f64::from(q.n) * f64::from(q.k) / (mma_tflops * 1e6);
                let floor_us = dram_us.max(mma_us);
                let reference_us =
                    graph_time(session, &shape, &prefill, &shape.reference_launch(m)?, &layers)?.us;
                let mut line = format!(
                    "{reference} M={m}: prefill {reference_us:.2} us (floor {floor_us:.2}: DRAM {dram_us:.2}, MMA {mma_us:.2})"
                );
                let record = |symbol: &str, tokens: u32, us: f64| -> LabResult<()> {
                    let mut receipt = BTreeMap::new();
                    receipt.insert("case".into(), json!("dense_gemm_mid_48_layer_graph_benchmark"));
                    receipt.insert("reference".into(), json!(reference));
                    receipt.insert("symbol".into(), json!(symbol));
                    receipt.insert("tokens".into(), json!(tokens));
                    receipt.insert("m".into(), json!(m));
                    receipt.insert("n".into(), json!(q.n));
                    receipt.insert("k".into(), json!(q.k));
                    receipt.insert("layers".into(), json!(layer_count));
                    receipt.insert("gemm_only_us_per_layer".into(), json!(us));
                    receipt.insert("prefill_us_per_layer".into(), json!(reference_us));
                    receipt.insert("dram_roofline_us".into(), json!(dram_us));
                    receipt.insert("mma_roofline_us".into(), json!(mma_us));
                    receipt.insert("percent_of_floor".into(), json!(100.0 * floor_us / us));
                    receipt.insert("dram_peak_gbps".into(), json!(dram_gbps));
                    receipt.insert("dram_peak_source".into(), json!(dram_source));
                    receipt.insert("mma_peak_tflops".into(), json!(mma_tflops));
                    receipt.insert("mma_peak_source".into(), json!(mma_source));
                    session.receipt("bench", receipt)?;
                    Ok(())
                };
                record(&shape.reference.symbol, 128, reference_us)?;
                for entry in shape.entries.iter().filter(|e| m <= e.max_rows()) {
                    let symbol = entry.kernel.symbol();
                    let function = mid_function(session, &mid::cubin_name(None), &symbol, entry)?;
                    let us =
                        graph_time(session, &shape, &function, &shape.mid_launch(entry, m)?, &layers)?.us;
                    record(&symbol, entry.activation_box_rows(), us)?;
                    line.push_str(&format!(
                        ", t{} {us:.2} us ({:+.1}%)",
                        entry.activation_box_rows(),
                        100.0 * (us / reference_us - 1.0)
                    ));
                }
                eprintln!("{line}");
            }
        }
        Ok(())
    })
    .expect("medium GEMM benchmarks");
}
