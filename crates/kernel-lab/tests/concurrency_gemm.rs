// Fixture adapted from ops/gemm.rs; CPU oracle covers first, middle and last rows.
//! GPU-gated dense GEMM correctness and timing cases.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{
    ActivationFormat, DType, GemmQuery, Refusal, RuntimeCounts, ScaleMode, StaticQuery,
    WeightFormat,
};
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::quantize::{self, Quantized, Scales};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, check_output, gpu_test};
use kernels_sm120::gemm::{self, GemmBuffers, GemmVariant};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::collections::BTreeMap;

const WEIGHT: u64 = 1;
const ACTIVATION: u64 = 2;
const OUTPUT: u64 = 3;
const PARTIAL: u64 = 4;
const SEMAPHORES: u64 = 5;
const WEIGHT_SCALES: u64 = 6;
const ACTIVATION_SCALES: u64 = 7;
const WEIGHT_GLOBAL: u64 = 8;
const ACTIVATION_GLOBAL: u64 = 9;

fn query(variant: &GemmVariant) -> GemmQuery {
    let StaticQuery::Gemm(query) = variant.key.query else {
        unreachable!("GEMM catalog contains only GEMM keys")
    };
    query
}

fn seeded_values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let centered = ((state >> 32) % 513) as i32 - 256;
            centered as f32 / 4096.0
        })
        .collect()
}

struct Encoded {
    data: Vec<u8>,
    scales: Option<Vec<u8>>,
    global: Option<f32>,
    decoded: Vec<f64>,
}

fn encoded_quantized(quantized: Quantized, oracle: bool) -> Encoded {
    let decoded = if oracle {
        quantized.decoded()
    } else {
        Vec::new()
    };
    let global = matches!(&quantized.scales, Scales::Ue4M3(_)).then_some(quantized.global_scale);
    let scales = match quantized.scales {
        Scales::F32(values) => values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        Scales::Ue8M0(values) | Scales::Ue4M3(values) => values,
    };
    Encoded {
        data: quantized.codes,
        scales: Some(scales),
        global,
        decoded,
    }
}

fn encoded_bf16(values: &[f32], oracle: bool) -> Encoded {
    let bits = values
        .iter()
        .map(|&value| f32_to_bf16(value))
        .collect::<Vec<_>>();
    Encoded {
        data: bits.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        scales: None,
        global: None,
        decoded: if oracle {
            bits.iter()
                .map(|&bits| f64::from(bf16_to_f32(bits)))
                .collect()
        } else {
            Vec::new()
        },
    }
}

fn weight(
    values: &[f32],
    rows: usize,
    k: usize,
    format: WeightFormat,
    oracle: bool,
) -> LabResult<Encoded> {
    let quantized = match format {
        WeightFormat::Bf16 => return Ok(encoded_bf16(values, oracle)),
        WeightFormat::Fp8PerTensor => {
            quantize::fp8_weight_per_tensor(values, rows, k, ScaleMode::ExactF32)
        }
        WeightFormat::Fp8PerChannel => {
            quantize::fp8_weight_per_channel(values, rows, k, ScaleMode::ExactF32)
        }
        WeightFormat::Fp8Block128x128(mode) => quantize::fp8_weight_128x128(values, rows, k, mode),
        WeightFormat::Fp8Rows64x128(mode) => quantize::fp8_blocks(values, rows, k, 64, 128, mode),
        WeightFormat::Mxfp8 => quantize::mxfp8(values, rows, k),
        WeightFormat::Mxfp4 => quantize::mxfp4(values, rows, k),
        WeightFormat::Nvfp4 => quantize::nvfp4(values, rows, k),
        WeightFormat::Fp8Block32x32 => Err(Refusal(
            "32x32 UE8M0 fixtures require explicit codes and scales",
        )),
    }
    .map_err(|e| e.to_string())?;
    Ok(encoded_quantized(quantized, oracle))
}

fn activation(
    values: &[f32],
    rows: usize,
    k: usize,
    format: ActivationFormat,
    oracle: bool,
) -> LabResult<Encoded> {
    let quantized = match format {
        ActivationFormat::Bf16 => return Ok(encoded_bf16(values, oracle)),
        ActivationFormat::Fp8Group128(mode) => {
            quantize::fp8_activation_1x128(values, rows, k, mode)
        }
        ActivationFormat::Mxfp8 => quantize::mxfp8(values, rows, k),
        ActivationFormat::Nvfp4 => quantize::nvfp4(values, rows, k),
    }
    .map_err(|e| e.to_string())?;
    Ok(encoded_quantized(quantized, oracle))
}

fn upload(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len())?;
    buffer
        .copy_from_bytes(data, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

pub(crate) struct Case {
    variant: GemmVariant,
    counts: RuntimeCounts,
    plan: kernel_core::contract::LaunchPlan,
    args: kernel_core::contract::ArgumentList,
    buffers: BTreeMap<u64, DeviceBuffer>,
    reference: Option<OracleOutput>,
    output_len: usize,
}

struct GraphLayer {
    weight: DeviceBuffer,
    output: DeviceBuffer,
    partial: DeviceBuffer,
    semaphores: DeviceBuffer,
}

use kernel_lab::poison::{Replay, WriteMap};

fn prefill_variants() -> Vec<GemmVariant> {
    gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|v| {
            v.profile == model_qwen38_contract::sm120::config::PROFILE
                && v.symbol.contains("prefill")
        })
        .collect()
}

fn partial_map(variant: &GemmVariant, live: usize, grid: usize) -> LabResult<WriteMap> {
    let q = query(variant);
    let segments = gemm::prefill_segments(&variant.key).map_err(|e| e.to_string())? as usize;
    fragment_partial_map(
        live,
        q.n as usize,
        q.k as usize,
        q.epilogues.swiglu.is_some(),
        variant.key.knobs.tile_m as usize,
        segments,
        grid,
    )
}

// fragment_prefill.cuh:18-36,50-132,342-354,531-556. Publication is
// unconditional for all eight math warps, even when a warp has no live rows.
fn fragment_partial_map(
    live: usize,
    n: usize,
    k: usize,
    swiglu: bool,
    tokens: usize,
    declared_segments: usize,
    grid: usize,
) -> LabResult<WriteMap> {
    if ![64, 128].contains(&tokens) || grid == 0 || k < 128 || declared_segments == 0 {
        return Err("invalid fragment prefill geometry".into());
    }
    let segments = declared_segments.min(8).min(k / 128);
    let output_n = if swiglu { n / 2 } else { n };
    let tiles = output_n.div_ceil(if swiglu { 64 } else { 128 }) * live.div_ceil(tokens);
    let mut tail = if tiles >= grid { tiles % grid } else { tiles };
    if segments == 1 || tail * 10 >= grid * 9 {
        tail = 0;
    }
    let units = tail * segments;
    let ctas = grid.min(units);
    let fragments = (tokens / 32) * 4;
    let slab = 8 * fragments * 32 * 4 * 4;
    let mut ranges = Vec::new();
    for cta in 0..ctas {
        let (mut begin, end) = (cta * units / ctas, (cta + 1) * units / ctas);
        while begin < end {
            let first = begin % segments;
            let count = (segments - first).min(end - begin);
            if first == 0 && count < segments {
                let start = cta * 8 * slab;
                ranges.push(start..start + slab);
            } else if first != 0 {
                for slot in 1..=count {
                    let start = (cta * 8 + slot) * slab;
                    ranges.push(start..start + slab);
                }
            }
            begin += count;
        }
    }
    WriteMap::of(DType::F32).byte_ranges(ranges)
}

#[test]
fn fragment_partial_contract_cpu() {
    use kernel_lab::poison::{self, Fill};
    let empty = fragment_partial_map(64, 38144, 2560, false, 128, 1, 188).unwrap();
    assert_eq!(empty.written_bytes(), 0);
    let mut bytes = vec![0xff; 4];
    bytes[0] = 0;
    assert!(
        poison::check(
            "single segment scribble",
            &bytes,
            &empty,
            Fill::Pattern(&[0xff])
        )
        .is_err()
    );
    let map = fragment_partial_map(64, 128, 512, false, 64, 4, 2).unwrap();
    let slab = 8 * 8 * 32 * 4 * 4;
    assert_eq!(map.ranges(), &[0..slab, 9 * slab..11 * slab]);
    let mut bytes = vec![0xff; 16 * slab];
    for range in map.ranges() {
        bytes[range.clone()].fill(0);
    }
    assert!(poison::check("split slots", &bytes, &map, Fill::Pattern(&[0xff])).is_ok());
    bytes[8 * slab] = 0;
    assert!(poison::check("unused slot", &bytes, &map, Fill::Pattern(&[0xff])).is_err());
    bytes[8 * slab] = 0xff;
    bytes[9 * slab..9 * slab + 4].fill(0xff);
    assert!(poison::check("missing fragment", &bytes, &map, Fill::Pattern(&[0xff])).is_err());
    let mut head = false;
    for variant in prefill_variants() {
        for rows in conc::ROWS {
            let counts = RuntimeCounts {
                m: rows as u32,
                batch: rows as u32,
                split_k: gemm::choose_split_k(&variant.key).unwrap(),
                context: 0,
                routed_rows: 0,
            };
            let plan = gemm::plan(&variant.key, counts).unwrap();
            let map = partial_map(&variant, rows, plan.grid[0] as usize).unwrap();
            assert!(
                map.ranges()
                    .last()
                    .is_none_or(|r| r.end <= plan.workspace_bytes as usize)
            );
            if variant.symbol.contains("head_f32_prefill") {
                head = true;
                assert_eq!(map.written_bytes(), 0);
            }
        }
    }
    assert!(head);
}

impl Case {
    pub(crate) fn prepare(
        session: &Session,
        symbol: &str,
        m: u32,
        correctness: bool,
    ) -> LabResult<Self> {
        let variant = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|item| item.symbol == symbol)
            .ok_or_else(|| format!("missing GEMM catalog symbol {symbol}"))?;
        let q = query(&variant);
        let n = q.n as usize;
        let k = q.k as usize;
        let capacity = q.m_capacity as usize;
        if m > q.m_capacity {
            return Err(format!("M={m} exceeds {} capacity", q.m_capacity));
        }
        let raw_weight = seeded_values(n * k, 0x0047_5745_4947_4854);
        let mut raw_activation = vec![0.0_f32; capacity * k];
        raw_activation[..m as usize * k]
            .copy_from_slice(&seeded_values(m as usize * k, 0x4143_5449_5641_5445));
        let mut w = weight(&raw_weight, n, k, q.weight, correctness)?;
        let a = activation(&raw_activation, capacity, k, q.activation, correctness)?;
        if correctness && q.activation == ActivationFormat::Bf16 && q.weight != WeightFormat::Bf16 {
            for value in &mut w.decoded {
                *value = f64::from(bf16_to_f32(f32_to_bf16(*value as f32)));
            }
        }
        if gemm::fragment_weight_enabled(&variant.key) {
            let format = if q.weight == WeightFormat::Bf16 {
                FragmentWeightFormat::Bf16
            } else {
                FragmentWeightFormat::Fp8E4M3
            };
            w.data = FragmentWeightLayout { rows: n, k, format }
                .pack(&w.data)
                .map_err(|error| error.to_string())?
                .bytes;
        }
        let splits = gemm::choose_split_k(&variant.key).map_err(|e| e.to_string())?;
        let counts = RuntimeCounts {
            m,
            batch: m,
            context: 0,
            routed_rows: 0,
            split_k: splits,
        };
        let plan = gemm::plan(&variant.key, counts).map_err(|e| e.to_string())?;
        let output_n = if q.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        };
        let output_len = m as usize * output_n;
        let mut buffers = BTreeMap::new();
        buffers.insert(WEIGHT, upload(session, &w.data)?);
        buffers.insert(ACTIVATION, upload(session, &a.data)?);
        buffers.insert(
            OUTPUT,
            session.allocate(output_len * if q.output == DType::F32 { 4 } else { 2 })?,
        );
        buffers.insert(PARTIAL, session.allocate(plan.workspace_bytes as usize)?);
        // Prefill clock64 profiles store samples after the first 8 KiB.
        buffers.insert(
            SEMAPHORES,
            upload(
                session,
                &vec![0; (plan.workspace_bytes as usize).max(1 << 16)],
            )?,
        );
        if let Some(scales) = &w.scales {
            buffers.insert(WEIGHT_SCALES, upload(session, scales)?);
        }
        if let Some(scales) = &a.scales {
            buffers.insert(ACTIVATION_SCALES, upload(session, scales)?);
        }
        if let Some(global) = w.global {
            buffers.insert(WEIGHT_GLOBAL, upload(session, &global.to_ne_bytes())?);
        }
        if let Some(global) = a.global {
            buffers.insert(ACTIVATION_GLOBAL, upload(session, &global.to_ne_bytes())?);
        }
        let maps =
            gemm::tensor_maps(&variant.key, WEIGHT, ACTIVATION).map_err(|e| e.to_string())?;
        let args = gemm::arguments(
            &variant.key,
            counts,
            maps,
            GemmBuffers {
                output: OUTPUT,
                partial: PARTIAL,
                semaphores: SEMAPHORES,
                weight_scales: w.scales.as_ref().map(|_| WEIGHT_SCALES),
                activation_scales: a.scales.as_ref().map(|_| ACTIVATION_SCALES),
                residual: None,
                weight_global_scale: w.global.map(|_| WEIGHT_GLOBAL),
                activation_global_scale: a.global.map(|_| ACTIVATION_GLOBAL),
            },
        )
        .map_err(|e| e.to_string())?;
        let reference = if correctness {
            Some(
                oracle::gemm(
                    &[0, m as usize / 2, m as usize - 1]
                        .into_iter()
                        .flat_map(|r| a.decoded[r * k..(r + 1) * k].iter().copied())
                        .collect::<Vec<_>>(),
                    &w.decoded,
                    GemmShape {
                        m: 3,
                        n,
                        k,
                        experts: 1,
                    },
                    &[],
                    GemmEpilogue {
                        residual: None,
                        swiglu: q.epilogues.swiglu,
                    },
                )
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        Ok(Self {
            variant,
            counts,
            plan,
            args,
            buffers,
            reference,
            output_len,
        })
    }

    fn replay<'a>(&'a self, s: &'a Session) -> LabResult<Replay<'a>> {
        let element = if query(&self.variant).output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        let map = partial_map(
            &self.variant,
            self.counts.m as usize,
            self.plan.grid[0] as usize,
        )?;
        Ok(Replay::new(s)
            .output(
                "projection",
                &self.buffers[&OUTPUT],
                WriteMap::of(element).prefix(self.output_len)?,
            )
            .output("projection partials", &self.buffers[&PARTIAL], map))
    }

    pub(crate) fn function(&self, session: &Session) -> LabResult<Function> {
        let module = session.module(&format!("gemm_{}_sm120", self.variant.profile.tag()))?;
        let function = module
            .function(&self.variant.symbol)
            .map_err(|e| e.to_string())?;
        if self.plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(self.plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        Ok(function)
    }

    pub(crate) fn launch(
        &self,
        session: &Session,
        function: &Function,
        stream: &Stream,
    ) -> LabResult<()> {
        self.launch_layer(session, function, stream, None)
    }

    fn launch_layer(
        &self,
        session: &Session,
        function: &Function,
        stream: &Stream,
        layer: Option<&GraphLayer>,
    ) -> LabResult<()> {
        let mut buffers = self
            .buffers
            .iter()
            .map(|(&id, buffer)| (id, buffer))
            .collect::<BTreeMap<_, _>>();
        if let Some(layer) = layer {
            buffers.insert(WEIGHT, &layer.weight);
            buffers.insert(OUTPUT, &layer.output);
            buffers.insert(PARTIAL, &layer.partial);
            buffers.insert(SEMAPHORES, &layer.semaphores);
        }
        launch::launch(
            &session.context,
            function,
            stream,
            self.plan,
            &self.args,
            &Bindings { buffers },
        )
    }

    fn output(&self, session: &Session) -> LabResult<Vec<f32>> {
        let buffer = &self.buffers[&OUTPUT];
        if query(&self.variant).output == DType::F32 {
            session.download_f32(buffer, self.output_len)
        } else {
            Ok(session
                .download_bf16(buffer, self.output_len)?
                .into_iter()
                .map(bf16_to_f32)
                .collect())
        }
    }

    fn receipt(&self, session: &Session, actual: &[f32], label: &str) -> LabResult<()> {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(label));
        receipt.insert("symbol".into(), json!(self.variant.symbol));
        receipt.insert("profile".into(), json!(self.variant.profile.as_str()));
        receipt.insert("m".into(), json!(self.counts.m));
        receipt.insert("split_k".into(), json!(self.counts.split_k));
        let gate = check_output(
            &[0, self.counts.m as usize / 2, self.counts.m as usize - 1]
                .into_iter()
                .flat_map(|r| {
                    let n = self.output_len / self.counts.m as usize;
                    actual[r * n..(r + 1) * n].iter().copied()
                })
                .collect::<Vec<_>>(),
            self.reference.as_ref().ok_or("missing GEMM oracle")?,
            query(&self.variant).output,
            &mut receipt,
        );
        session.receipt("gemm", receipt.clone())?;
        gate.map_err(|error| {
            format!(
                "{}: {error}; worst={}",
                self.variant.symbol, receipt["worst_element"]
            )
        })
    }

    fn bitwise_receipt(
        &self,
        session: &Session,
        expected: &[f32],
        actual: &[f32],
        label: &str,
    ) -> LabResult<()> {
        let worst = expected
            .iter()
            .zip(actual)
            .enumerate()
            .filter(|(_, (left, right))| left.to_bits() != right.to_bits())
            .max_by(|(_, (left_a, right_a)), (_, (left_b, right_b))| {
                (**left_a - **right_a)
                    .abs()
                    .total_cmp(&(**left_b - **right_b).abs())
            });
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(label));
        receipt.insert("symbol".into(), json!(self.variant.symbol));
        receipt.insert("correct".into(), json!(worst.is_none()));
        if let Some((index, (&left, &right))) = worst {
            receipt.insert(
                "worst_element".into(),
                json!({"index": index, "eager_bits": left.to_bits(),
                    "replay_bits": right.to_bits(), "eager": left, "replay": right}),
            );
        }
        session.receipt("gemm", receipt.clone())?;
        if worst.is_some() {
            return Err(format!(
                "{label}: bitwise mismatch; worst={}",
                receipt["worst_element"]
            ));
        }
        Ok(())
    }
}

#[path = "common/concurrency.rs"]
mod conc;
#[test]
fn m3_qwen_prefill_gemms() {
    if !conc::enabled("m3") {
        return;
    }
    gpu_test(8 << 30, |s| {
        conc::exclusive(s)?;
        let symbols: Vec<_> = prefill_variants()
            .into_iter()
            .map(|v| v.symbol.clone())
            .collect();
        assert!(!symbols.is_empty());
        let mut failures = conc::replay::Failures::default();
        for rows in conc::ROWS {
            for symbol in &symbols {
                let result = (|| -> LabResult<()> {
                    let case = Case::prepare(s, symbol, rows as u32, true)?;
                    let function = case.function(s)?;
                    let mut replay = case.replay(s)?;
                    replay.eager(|| case.launch(s, &function, &s.stream))?;
                    let before = case.output(s)?;
                    case.receipt(s, &before, "M3 sampled CPU reference")?;
                    conc::measure(
                        s,
                        "M3",
                        symbol,
                        rows,
                        "gemm_qwen38_sm120",
                        || case.launch(s, &function, &s.stream),
                        || case.bitwise_receipt(s, &before, &case.output(s)?, "M3 graph replay"),
                        |graph| {
                            replay.graph(graph)?;
                            let mut fields = BTreeMap::new();
                            replay.record(&mut fields);
                            Ok(fields)
                        },
                    )?;

                    Ok(())
                })();
                failures.check(&format!("{symbol} rows={rows}"), result);
            }
        }
        failures.finish()?;
        Ok(())
    })
    .expect("M3 Qwen prefill GEMMs")
    .expect("GPU required");
}
