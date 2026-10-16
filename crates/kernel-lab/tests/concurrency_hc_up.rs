//! The Qwen3.8 GEMM kernels of `gemm::fused::catalog(kernel_lab::catalogs())` (cluster decode,
//! short K, skinny N) against the f64 GEMM oracle, with graph replay, PDL and
//! determinism, plus G4 timings.

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, LaunchPlan};
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, check_output, gpu_test};
use kernels_sm120::gemm::fused::{Buffers, Entry, Kind, catalog};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

const ACTIVATION: u64 = 3;
const PARTIAL: u64 = 5;
const SEMAPHORES: u64 = 6;
/// Weight and output buffer handles are these plus the segment index.
const WEIGHT: u64 = 10;
const OUTPUT: u64 = 20;

fn seeded(len: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16((((state >> 32) % 513) as i32 - 256) as f32 / 4096.0)
        })
        .collect()
}

fn bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn upload(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len())?;
    buffer
        .copy_from_bytes(data, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn find(symbol: &str) -> Entry {
    catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .unwrap_or_else(|| panic!("missing Qwen GEMM entry {symbol}"))
}

fn module(session: &Session) -> LabResult<Module> {
    session.module("gemm_qwen38_own_sm120")
}

struct Case {
    entry: Entry,
    m: u32,
    plan: LaunchPlan,
    args: kernel_core::contract::ArgumentList,
    buffers: BTreeMap<u64, DeviceBuffer>,
    reference: Option<OracleOutput>,
    /// Elements of each output buffer.
    output_lens: Vec<usize>,
}

use kernel_lab::poison::{Phase, WriteMap};

impl Case {
    /// `pad` extra columns end every output row; they must stay zero.
    fn prepare(
        session: &Session,
        symbol: &str,
        m: u32,
        pad: u32,
        correctness: bool,
    ) -> LabResult<Self> {
        let entry = find(symbol);
        let q = entry.query;
        let (n, k, rows) = (q.n as usize, q.k as usize, m as usize);
        // SwiGLU halves the output columns.
        let out_n = if q.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        };
        let raw = seeded(n * k, 0x0051_5745_4e38 + k as u64);
        let mut buffers = BTreeMap::new();

        let mut first = 0;
        for (s, &segment) in entry.segments.iter().enumerate() {
            let segment = segment as usize;
            let packed = FragmentWeightLayout {
                rows: segment,
                k,
                format: FragmentWeightFormat::Bf16,
            }
            .pack(&bytes(&raw[first * k..(first + segment) * k]))
            .map_err(|e| e.to_string())?
            .bytes;
            buffers.insert(WEIGHT + s as u64, upload(session, &packed)?);

            first += segment;
        }
        let activation = seeded(rows * k, 0x4143_5438 + u64::from(m));
        buffers.insert(ACTIVATION, session.upload_bf16(&activation)?);
        let plan = entry.plan(m).map_err(|e| e.to_string())?;
        // (columns, stride, first dense column) of each output buffer.
        let layout: Vec<(usize, usize, usize)> = match entry.kind {
            Kind::ClusterDecode { .. } if out_n == n => entry
                .segments
                .iter()
                .scan(0, |start, &r| {
                    let item = (r as usize, r as usize + pad as usize, *start);
                    *start += r as usize;
                    Some(item)
                })
                .collect(),
            _ => vec![(out_n, out_n + pad as usize, 0)],
        };
        let element = if q.output == DType::F32 { 4 } else { 2 };
        let mut output_strides = [0; 4];
        for (s, &(_, stride, _)) in layout.iter().enumerate() {
            buffers.insert(
                OUTPUT + s as u64,
                upload(session, &vec![0; rows * stride * element])?,
            );
            output_strides[s] = stride as u32;
        }
        buffers.insert(
            PARTIAL,
            session.allocate((plan.workspace_bytes as usize).max(4))?,
        );
        buffers.insert(
            SEMAPHORES,
            upload(
                session,
                &vec![0; (entry.semaphore_words() as usize).max(1) * 4],
            )?,
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
            .map_err(|e| e.to_string())?;
        let reference = if correctness {
            let decode = |v: &[u16]| {
                v.iter()
                    .map(|&x| f64::from(bf16_to_f32(x)))
                    .collect::<Vec<_>>()
            };
            let dense = oracle::gemm(
                &decode(&activation),
                &decode(&raw),
                GemmShape {
                    m: rows,
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
            .map_err(|e| e.to_string())?;
            // Outputs in buffer order; padding columns are exact zeros.
            let mut values = Vec::new();
            let mut bounds = Vec::new();
            for &(columns, stride, start) in &layout {
                for t in 0..rows {
                    for c in 0..stride {
                        let dense_index = t * out_n + start + c;
                        values.push(if c < columns {
                            dense.values[dense_index]
                        } else {
                            0.0
                        });
                        bounds.push(if c < columns {
                            dense.sum_abs_products[dense_index]
                        } else {
                            0.0
                        });
                    }
                }
            }
            Some(OracleOutput::new(values, bounds, k).map_err(|e| e.to_string())?)
        } else {
            None
        };
        Ok(Self {
            entry,
            m,
            plan,
            args,
            buffers,
            reference,
            output_lens: layout.iter().map(|&(_, stride, _)| rows * stride).collect(),
        })
    }

    fn checks<'a>(&'a self, s: &'a Session) -> LabResult<conc::replay::Checks<'a>> {
        let mut checks = conc::replay::Checks::new(s);
        let element = if self.entry.query.output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        for (segment, &words) in self.output_lens.iter().enumerate() {
            checks = checks.output(
                &format!("output {segment}"),
                &self.buffers[&(OUTPUT + segment as u64)],
                WriteMap::of(element).prefix(words)?,
            );
        }
        if self.plan.workspace_bytes != 0 {
            let bytes = self.plan.workspace_bytes as usize;
            checks = checks.scratch_map(
                "projection partials",
                &self.buffers[&PARTIAL],
                0,
                move |_| WriteMap::of(DType::F32).byte_prefix(bytes),
            );
        }
        Ok(checks)
    }

    fn function(&self, module: &Module) -> LabResult<Function> {
        let function = module
            .function(&self.entry.symbol)
            .map_err(|e| e.to_string())?;
        if self.plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(self.plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        Ok(function)
    }

    fn launch(&self, session: &Session, function: &Function, stream: &Stream) -> LabResult<()> {
        self.launch_weights(session, function, stream, None)
    }

    /// Launch with other copies of the weights, if given.
    fn launch_weights(
        &self,
        session: &Session,
        function: &Function,
        stream: &Stream,
        weights: Option<&[DeviceBuffer]>,
    ) -> LabResult<()> {
        let mut buffers = self
            .buffers
            .iter()
            .map(|(&id, b)| (id, b))
            .collect::<BTreeMap<_, _>>();
        for (s, buffer) in weights.into_iter().flatten().enumerate() {
            buffers.insert(WEIGHT + s as u64, buffer);
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

    /// Every output buffer, concatenated in segment order.
    fn output(&self, session: &Session) -> LabResult<Vec<f32>> {
        let mut values = Vec::new();
        for (s, &len) in self.output_lens.iter().enumerate() {
            let buffer = &self.buffers[&(OUTPUT + s as u64)];
            if self.entry.query.output == DType::F32 {
                values.extend(session.download_f32(buffer, len)?);
            } else {
                values.extend(
                    session
                        .download_bf16(buffer, len)?
                        .into_iter()
                        .map(bf16_to_f32),
                );
            }
        }
        Ok(values)
    }

    fn check(&self, session: &Session, actual: &[f32], label: &str) -> LabResult<()> {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(label));
        receipt.insert("symbol".into(), json!(self.entry.symbol));
        receipt.insert("profile".into(), json!(self.entry.profile.as_str()));
        receipt.insert("m".into(), json!(self.m));
        receipt.insert("grid".into(), json!(self.plan.grid[0]));
        let gate = check_output(
            actual,
            self.reference.as_ref().ok_or("missing oracle")?,
            self.entry.query.output,
            &mut receipt,
        );
        session.receipt("gemm_qwen38", receipt.clone())?;
        gate.map_err(|e| {
            format!(
                "{} M={}: {e}; worst={}",
                self.entry.symbol, self.m, receipt["worst_element"]
            )
        })
    }
}

#[path = "common/concurrency.rs"]
mod conc;
#[test]
fn m3_qwen_short_k_and_skinny() {
    if !conc::enabled("m3") {
        return;
    }
    gpu_test(2 << 30, |s| {
        conc::exclusive(s)?;
        let module = module(s)?;
        let mut failures = conc::replay::Failures::default();
        for rows in conc::ROWS {
            for symbol in [
                "chop_gemm_qwen38_hc_up_prefill_v1",
                "chop_gemm_qwen38_lin_ab_v1",
                "chop_gemm_qwen38_hc_inject_v1",
            ] {
                let result = (|| -> LabResult<()> {
                    let case = Case::prepare(s, symbol, rows as u32, 0, true)?;
                    let f = case.function(&module)?;
                    let mut checks = case.checks(s)?;
                    checks.phase(Phase::Eager, false, || case.launch(s, &f, &s.stream))?;
                    let before = case.output(s)?;
                    case.check(s, &before, "M3 CPU reference")?;
                    conc::measure(
                        s,
                        "M3",
                        symbol,
                        rows,
                        "gemm_qwen38_own_sm120",
                        || case.launch(s, &f, &s.stream),
                        || {
                            assert_eq!(before, case.output(s)?, "HC up graph replay");
                            Ok(())
                        },
                        |graph| {
                            checks.phase(Phase::Graph, case.plan.pdl, || {
                                graph.launch(&s.stream).map_err(|e| e.to_string())
                            })?;
                            let mut fields = BTreeMap::new();
                            checks.record(&mut fields);
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
    .expect("M3 short-K and skinny projections")
    .expect("GPU required");
}
