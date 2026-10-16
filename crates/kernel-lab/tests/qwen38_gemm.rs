//! The Qwen3.8 GEMM kernels of `gemm::fused::catalog(kernel_lab::catalogs())` (cluster decode,
//! short K, skinny N) against the f64 GEMM oracle, with graph replay, PDL and
//! determinism, plus G4 timings.
//!
//! `CHOPINFER_QWEN38_RANKS` (2 or 4, default 4) selects the topology:
//! its cluster decode and `lin_ab` entries run, with TP4's rank-invariant
//! short-K and inject entries at every topology.
//!
//! TP2 runs are lean (`kernel_lab::qwen38_lean`): only the
//! topology's own entries (every cluster decode bucket and `lin_ab`), with
//! the oracle on sampled output columns (`kernel_lab::sample_indices`).

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::SwigluColumns;
use kernel_core::contract::{DType, LaunchPlan};
use kernel_core::oracle::{self, GemmEpilogue, GemmShape, OracleOutput};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, check_output, gpu_test, time};
use kernels_sm120::Profile;
use kernels_sm120::gemm::fused::{Buffers, Entry, Kind, catalog};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "common/accumulation.rs"]
mod accumulation;
#[path = "common/qwen_own_inputs.rs"]
mod qwen_own_inputs;
use qwen_own_inputs::{OwnInputs, own_inputs};

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase, WriteMap};

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

/// The Qwen-owned GEMM cubin of each profile an entry may have: TP4's and
/// the selected topology's.
struct Modules(Vec<(Profile, Module)>);

impl Modules {
    fn load(session: &Session) -> LabResult<Self> {
        let mut modules = Vec::new();
        for profile in [
            model_qwen38_contract::sm120::config::PROFILE,
            kernel_lab::qwen38_profile()?,
        ] {
            if modules.iter().all(|(loaded, _)| *loaded != profile) {
                let name = format!("gemm_{}_own_sm120", profile.tag());
                modules.push((profile, session.module(&name)?));
            }
        }
        Ok(Self(modules))
    }

    fn get(&self, entry: &Entry) -> LabResult<&Module> {
        self.0
            .iter()
            .find(|(profile, _)| *profile == entry.profile)
            .map(|(_, module)| module)
            .ok_or_else(|| format!("no cubin loaded for {}", entry.symbol))
    }
}

/// Records the topology in a receipt when it is not TP4's.
fn record_ranks(receipt: &mut BTreeMap<String, serde_json::Value>) -> LabResult<()> {
    let ranks = kernel_lab::qwen38_ranks()?;
    if ranks != 4 {
        receipt.insert("qwen38_ranks".into(), json!(ranks));
    }
    Ok(())
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
    /// Packed fragment-major bytes of each segment's weight.
    packed: Vec<Vec<u8>>,
    /// Lean runs: the output elements (indices into `output`) the sampled
    /// oracle covers, in reference order.
    picks: Option<Vec<usize>>,
}

/// The f64 oracle of every row at the dense output `columns` only: their
/// weight rows (a SwiGLU output's gate row `c` and up row `c + half`) as a
/// smaller GEMM. Values are `[rows, columns.len()]`.
fn sampled_oracle(
    activation: &[u16],
    raw: &[u16],
    (rows, k): (usize, usize),
    columns: &[usize],
    swiglu: Option<(SwigluColumns, usize)>,
) -> LabResult<OracleOutput> {
    let decode = |v: &[u16]| {
        v.iter()
            .map(|&x| f64::from(bf16_to_f32(x)))
            .collect::<Vec<_>>()
    };
    let weight_rows = match swiglu {
        Some((SwigluColumns::ConcatenatedHalves, half)) => columns
            .iter()
            .copied()
            .chain(columns.iter().map(|c| c + half))
            .collect(),
        Some((SwigluColumns::Interleaved, _)) => {
            return Err("no Qwen GEMM interleaves its SwiGLU columns".into());
        }
        None => columns.to_vec(),
    };
    let weight = weight_rows
        .iter()
        .flat_map(|&r| decode(&raw[r * k..(r + 1) * k]))
        .collect::<Vec<_>>();
    oracle::gemm(
        &decode(activation),
        &weight,
        GemmShape {
            m: rows,
            n: weight_rows.len(),
            k,
            experts: 1,
        },
        &[],
        GemmEpilogue {
            residual: None,
            swiglu: swiglu.map(|(style, _)| style),
        },
    )
    .map_err(|e| e.to_string())
}

/// Lean output columns of an entry: every 61st, the first and last, both
/// sides of every segment start and of every 16-row strip of each segment
/// (the cluster decode tiles strips over the segments in turn; its 32- and
/// 64-row tiles and the skinny strips end on strips), and the SwiGLU half
/// (its first and last output columns). Outputs of at most 1024 columns
/// (the skinny entries, TP2's SwiGLU shared expert) all run.
fn lean_columns(entry: &Entry, out_n: usize) -> Vec<usize> {
    if out_n <= 1024 {
        return (0..out_n).collect();
    }
    let mut boundaries = Vec::new();
    let mut start = 0;
    for &rows in &entry.segments {
        let rows = rows as usize;
        boundaries.extend((start..start + rows).step_by(16));
        start += rows;
    }
    kernel_lab::sample_indices(out_n, 61, &boundaries)
}

impl Case {
    /// `pad` extra columns end every output row; they must stay zero.
    fn prepare(
        session: &Session,
        symbol: &str,
        m: u32,
        pad: u32,
        correctness: bool,
        lean: bool,
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
        let mut packed_weights = Vec::new();
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
            packed_weights.push(packed);
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
        let mut picks = None;
        let reference = if correctness && lean {
            // Sampled columns of every row, then every padding column, per
            // output buffer; `picks` holds their output indices.
            let columns = lean_columns(&entry, out_n);
            let sampled = sampled_oracle(
                &activation,
                &raw,
                (rows, k),
                &columns,
                q.epilogues.swiglu.map(|style| (style, out_n)),
            )?;
            let (mut values, mut bounds, mut indices) = (Vec::new(), Vec::new(), Vec::new());
            let mut base = 0;
            for &(width, stride, start) in &layout {
                for t in 0..rows {
                    for (j, &c) in columns.iter().enumerate() {
                        if (start..start + width).contains(&c) {
                            indices.push(base + t * stride + c - start);
                            values.push(sampled.values[t * columns.len() + j]);
                            bounds.push(sampled.sum_abs_products[t * columns.len() + j]);
                        }
                    }
                    for c in width..stride {
                        indices.push(base + t * stride + c);
                        values.push(0.0);
                        bounds.push(0.0);
                    }
                }
                base += rows * stride;
            }
            picks = Some(indices);
            Some(OracleOutput::new(values, bounds, k).map_err(|e| e.to_string())?)
        } else if correctness {
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
            packed: packed_weights,
            picks,
        })
    }

    fn output_widths(&self) -> Vec<usize> {
        let n = self.entry.query.n as usize;
        let out = if self.entry.query.epilogues.swiglu.is_some() {
            n / 2
        } else {
            n
        };
        match self.entry.kind {
            Kind::ClusterDecode { .. } if out == n => {
                self.entry.segments.iter().map(|&n| n as usize).collect()
            }
            _ => vec![out],
        }
    }

    fn replay_checks<'a>(&'a self, session: &'a Session) -> LabResult<replay::Checks<'a>> {
        let mut checks = replay::Checks::new(session);
        let element = if self.entry.query.output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        for (segment, columns) in self.output_widths().into_iter().enumerate() {
            let stride = self.output_lens[segment] / self.m as usize;
            checks = checks.output(
                &format!("output {segment}"),
                &self.buffers[&(OUTPUT + segment as u64)],
                WriteMap::of(element)
                    .ranges((0..self.m as usize).map(|row| row * stride..row * stride + columns))?,
            );
        }
        if self.plan.workspace_bytes != 0 {
            // Skinny and clustered GEMMs publish
            // complete planned F32 slabs, including padded tile lanes.
            let bytes = self.plan.workspace_bytes as usize;
            checks = checks.scratch_map("GEMM partials", &self.buffers[&PARTIAL], 0, move |_| {
                WriteMap::of(DType::F32).byte_prefix(bytes)
            });
        }
        // Self-resetting semaphores are never filled.
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
        record_ranks(&mut receipt)?;
        let sampled;
        let actual = match &self.picks {
            Some(picks) => {
                receipt.insert("lean".into(), json!(true));
                receipt.insert("sampled_outputs".into(), json!(picks.len()));
                sampled = picks.iter().map(|&i| actual[i]).collect::<Vec<_>>();
                &sampled[..]
            }
            None => actual,
        };
        // Padding is covered by the strict replay map, rather than the
        // numerical oracle's old zero initialization witness.
        let mut oracle_actual = actual.to_vec();
        if let Some(picks) = &self.picks {
            let mut base = 0;
            for (segment, columns) in self.output_widths().into_iter().enumerate() {
                let stride = self.output_lens[segment] / self.m as usize;
                for (index, &pick) in picks.iter().enumerate() {
                    if (base..base + self.output_lens[segment]).contains(&pick)
                        && (pick - base) % stride >= columns
                    {
                        oracle_actual[index] = 0.0;
                    }
                }
                base += self.output_lens[segment];
            }
        } else {
            let mut base = 0;
            for (segment, columns) in self.output_widths().into_iter().enumerate() {
                let stride = self.output_lens[segment] / self.m as usize;
                for row in 0..self.m as usize {
                    oracle_actual[base + row * stride + columns..base + (row + 1) * stride]
                        .fill(0.0);
                }
                base += self.output_lens[segment];
            }
        }
        let gate = check_output(
            &oracle_actual,
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

fn bitwise(label: &str, left: &[f32], right: &[f32]) -> LabResult<()> {
    match left
        .iter()
        .zip(right)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        None => Ok(()),
        Some(index) => Err(format!(
            "{label}: bitwise mismatch at {index}: {} vs {}",
            left[index], right[index]
        )),
    }
}

/// (symbol, M, output row padding) covering one token, partial token tiles,
/// every split count regime, and padded skinny output rows. `lin_ab` runs
/// at the selected topology (`lin_ab_case`).
const CASES: [(&str, u32, u32); 14] = [
    ("chop_gemm_qwen38_hc_up_decode_v1", 1, 0),
    ("chop_gemm_qwen38_hc_up_decode_v1", 7, 0),
    ("chop_gemm_qwen38_hc_up_decode_v1", 32, 0),
    ("chop_gemm_qwen38_hc_up_prefill_v1", 1, 0),
    ("chop_gemm_qwen38_hc_up_prefill_v1", 200, 0),
    ("chop_gemm_qwen38_hc_up_prefill_v1", 2049, 0),
    ("chop_gemm_qwen38_lin_ab_v1", 1, 0),
    ("chop_gemm_qwen38_lin_ab_v1", 13, 8),
    ("chop_gemm_qwen38_lin_ab_v1", 300, 0),
    ("chop_gemm_qwen38_lin_ab_v1", 8192, 0),
    ("chop_gemm_qwen38_hc_inject_v1", 1, 0),
    ("chop_gemm_qwen38_hc_inject_v1", 32, 0),
    ("chop_gemm_qwen38_hc_inject_v1", 777, 4),
    ("chop_gemm_qwen38_hc_inject_v1", 8192, 0),
];

/// A `CASES` row at the selected topology: `lin_ab` becomes the profile's,
/// its rows padded to the bundle's `lin_ab_stride`
/// before the case's own padding. Other rows are TP4's at every topology.
fn lin_ab_case(symbol: &str, m: u32, pad: u32) -> LabResult<(String, u32, u32)> {
    if symbol != "chop_gemm_qwen38_lin_ab_v1" {
        return Ok((symbol.to_string(), m, pad));
    }
    let profile = kernel_lab::qwen38_profile()?;
    let shape = kernel_lab::qwen38_shape()?;
    let stride_pad = shape.lin_ab_stride() - 2 * shape.linear_value_heads();
    Ok((
        format!("chop_gemm_{}_lin_ab_v1", profile.tag()),
        m,
        stride_pad + pad,
    ))
}

/// Lean `lin_ab` rows: one token (ten K splits), a partial token tile with
/// padded output rows, and 2049 rows (65 token tiles at six K splits, the
/// last tile partial).
const LEAN_LIN_AB: [(u32, u32); 3] = [(1, 0), (13, 8), (2049, 0)];

/// Every cluster decode entry of `profile` at its full bucket, and at a
/// partial one with padded output rows.
fn cluster_cases(profile: Profile) -> Vec<(String, u32, u32)> {
    catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|e| e.profile == profile)
        .filter(|e| matches!(e.kind, Kind::ClusterDecode { .. }))
        .flat_map(|e| {
            let m = e.query.m_capacity;
            let mut cases = vec![(e.symbol.clone(), m, 0)];
            if m > 1 {
                cases.push((e.symbol.clone(), m / 2 + 1, 8));
            }
            cases
        })
        .collect()
}

#[test]
fn qwen38_gemm_g1() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let modules = Modules::load(session)?;
        let lean = kernel_lab::qwen38_lean()?;
        // Lean: the rank-invariant short-K and inject rows are TP4's run.
        let mut cases = if lean {
            LEAN_LIN_AB
                .iter()
                .map(|&(m, pad)| lin_ab_case("chop_gemm_qwen38_lin_ab_v1", m, pad))
                .collect::<LabResult<Vec<_>>>()?
        } else {
            CASES
                .iter()
                .map(|&(symbol, m, pad)| lin_ab_case(symbol, m, pad))
                .collect::<LabResult<Vec<_>>>()?
        };
        cases.extend(cluster_cases(kernel_lab::qwen38_profile()?));
        let mut failures = replay::Failures::default();
        for (symbol, m, pad) in cases {
            let result = (|| -> LabResult<()> {
                let case = Case::prepare(session, &symbol, m, pad, true, lean)?;
                let function = case.function(modules.get(&case.entry)?)?;
                let mut checks = case.replay_checks(session)?;
                checks.phase(Phase::Eager, false, || {
                    case.launch(session, &function, &session.stream)
                })?;
                let eager = case.output(session)?;
                case.check(session, &eager, "qwen38_gemm")?;
                checks.phase(Phase::Relaunch, false, || {
                    case.launch(session, &function, &session.stream)
                })?;
                bitwise(
                    &format!("{symbol} M={m} repeat"),
                    &eager,
                    &case.output(session)?,
                )?;
                let graph = session.capture(|stream| case.launch(session, &function, stream))?;
                for _ in 0..2 {
                    checks.phase(Phase::Graph, false, || {
                        graph.launch(&session.stream).map_err(|e| e.to_string())
                    })?;
                }
                bitwise(
                    &format!("{symbol} M={m} graph"),
                    &eager,
                    &case.output(session)?,
                )?;
                // A PDL launch behind a producer that writes unrelated memory.
                let probe = session.module("probe_sm120")?;
                let flush = probe
                    .function("chop_probe_flush_v1")
                    .map_err(|e| e.to_string())?;
                let scratch = session.allocate(1 << 20)?;
                checks.prepare()?;
                let mut producer = replay::Checks::new(session).output(
                    "PDL producer",
                    &scratch,
                    WriteMap::of(DType::F32).byte_prefix(1 << 20)?,
                );
                producer.prepare()?;
                flush
                    .launch(
                        &session.stream,
                        launch::simple_config((64, 1, 1), (256, 1, 1), 0, (1, 1, 1), false)?,
                        &[
                            runtime_cuda::KernelArgument::Buffer(&scratch),
                            runtime_cuda::KernelArgument::U64(1 << 18),
                        ],
                    )
                    .map_err(|e| e.to_string())?;
                case.launch(session, &function, &session.stream)?;
                producer.finish(Phase::Eager, false)?;
                checks.finish(Phase::Pdl, true)?;
                let mut receipt =
                    BTreeMap::from([("symbol".into(), json!(symbol)), ("m".into(), json!(m))]);
                checks.record(&mut receipt);
                session.receipt("gemm_replay", receipt)?;
                bitwise(
                    &format!("{symbol} M={m} pdl"),
                    &eager,
                    &case.output(session)?,
                )?;
                Ok(())
            })();
            failures.check(&format!("{symbol} M={m}"), result);
        }
        failures.finish()
    })
    .expect("Qwen3.8 GEMM kernels");
}

/// Skinny entries in eager prefill phases, and prompt rows seeded by their
/// position over 13 binades so F32 sums round (`wide`): row `first + i` of
/// a 2,048-row chunk is row `i` of the window.
const SKINNY_PREFILL: [&str; 2] = [
    "chop_gemm_qwen38_lin_ab_v1",
    "chop_gemm_qwen38_hc_inject_v1",
];
const SKINNY_WINDOWS: [(u32, u32); 7] = [
    (1, 0),
    (1, 1037),
    (17, 5),
    (64, 64),
    (511, 100),
    (1024, 1024),
    (2047, 1),
];

fn wide(len: usize, seed: u64) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let binade = ((state >> 8) % 13) as i32;
            let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
            let sign = if state & 1 == 0 { 1.0 } else { -1.0 };
            f32_to_bf16(sign * (1.0 + unit) * 2_f32.powi(-binade))
        })
        .collect()
}

/// `entry`'s prefill schedule over rows [first, first + m) of `all`, as
/// output bits.
fn prefill_window(
    session: &Session,
    modules: &Modules,
    entry: &Entry,
    all: &[u16],
    first: u32,
    m: u32,
) -> LabResult<Vec<u32>> {
    let mut case = Case::prepare(session, &entry.symbol, m, 0, false, false)?;
    case.entry = entry.clone();
    let k = entry.query.k as usize;
    let rows = &all[first as usize * k..(first + m) as usize * k];
    case.buffers.insert(ACTIVATION, session.upload_bf16(rows)?);
    case.plan = entry.prefill_plan(m).map_err(|e| e.to_string())?;
    let handles = |base: u64| std::array::from_fn(|s| base + s as u64);
    case.args = entry
        .prefill_arguments(
            m,
            Buffers {
                weights: handles(WEIGHT),
                activation: ACTIVATION,
                outputs: handles(OUTPUT),
                output_strides: [entry.query.n, 0, 0, 0],
                partial: PARTIAL,
                semaphores: SEMAPHORES,
            },
        )
        .map_err(|e| e.to_string())?;
    let function = case.function(modules.get(entry)?)?;
    let mut checks = case.replay_checks(session)?;
    checks.phase(Phase::Eager, false, || {
        case.launch(session, &function, &session.stream)
    })?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(case
        .output(session)?
        .into_iter()
        .map(f32::to_bits)
        .collect())
}

/// A skinny entry's prefill rows do not depend on the step's live M or the
/// row's place in it: every window of a 2,048-row chunk equals the chunk's
/// rows bit for bit.
#[test]
fn qwen38_skinny_prefill_rows_ignore_chunking() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let modules = Modules::load(session)?;
        let mut failures = Vec::new();
        for symbol in SKINNY_PREFILL {
            let entry = find(symbol);
            let n = entry.query.n as usize;
            let all = wide(2048 * entry.query.k as usize, 0x534b_494e_4e59);
            let reference = prefill_window(session, &modules, &entry, &all, 0, 2048)?;
            for (m, first) in SKINNY_WINDOWS {
                let rows = prefill_window(session, &modules, &entry, &all, first, m)?;
                let expected = &reference[first as usize * n..(first + m) as usize * n];
                let differ = rows.iter().zip(expected).filter(|(a, b)| a != b).count();
                let mut receipt = BTreeMap::new();
                receipt.insert(
                    "case".into(),
                    json!("qwen38_skinny_prefill_rows_ignore_chunking"),
                );
                receipt.insert("symbol".into(), json!(symbol));
                receipt.insert("m".into(), json!(m));
                receipt.insert("first_row".into(), json!(first));
                receipt.insert("differ".into(), json!(differ));
                session.receipt("check", receipt)?;
                eprintln!(
                    "{symbol} M={m} first={first}: {differ} of {} differ",
                    rows.len()
                );
                if differ != 0 {
                    failures.push(format!("{symbol} M={m} first={first}: {differ}"));
                }
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "skinny prefill rows depend on chunking: {}",
                failures.join("; ")
            ))
        }
    })
    .expect("skinny prefill rows ignore chunking");
}

/// Kernel time of each skinny entry's prefill schedule at prefill chunk
/// rows for every split count from 1 to one 256-K chunk per split.
#[test]
fn qwen38_skinny_prefill_splits_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("skinny prefill split benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let modules = Modules::load(session)?;
        for symbol in SKINNY_PREFILL {
            let base = find(symbol);
            let chunks = base.query.k / 256;
            for m in [64, 512, 1024, 2048, 8192] {
                for splits in (1..=chunks).filter(|s| chunks % s == 0 || *s <= 8) {
                    let mut entry = base.clone();
                    if let Kind::Skinny {
                        ref mut prefill_splits,
                        ..
                    } = entry.kind
                    {
                        *prefill_splits = splits;
                    }
                    let mut case = Case::prepare(session, symbol, m, 0, false, false)?;
                    case.entry = entry.clone();
                    case.plan = entry.prefill_plan(m).map_err(|e| e.to_string())?;
                    let handles = |b: u64| std::array::from_fn(|s| b + s as u64);
                    case.args = entry
                        .prefill_arguments(
                            m,
                            Buffers {
                                weights: handles(WEIGHT),
                                activation: ACTIVATION,
                                outputs: handles(OUTPUT),
                                output_strides: [entry.query.n, 0, 0, 0],
                                partial: PARTIAL,
                                semaphores: SEMAPHORES,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                    let function = case.function(modules.get(&entry)?)?;
                    let timing = time(session, 3, 21, |stream| {
                        case.launch(session, &function, stream)
                    })?;
                    let us = f64::from(timing.median_ms) * 1000.0;
                    eprintln!("{symbol} M={m} splits={splits}: {us:.2} us");
                    let mut receipt = BTreeMap::new();
                    receipt.insert("case".into(), json!("qwen38_skinny_prefill_splits_bench"));
                    receipt.insert("symbol".into(), json!(symbol));
                    receipt.insert("m".into(), json!(m));
                    receipt.insert("splits".into(), json!(splits));
                    receipt.insert("median_us".into(), json!(us));
                    session.receipt("bench", receipt)?;
                }
            }
        }
        Ok(())
    })
    .expect("skinny prefill split benchmarks");
}

/// G4 timings with L2 flushed. Memory-bound rooflines use the weight,
/// activation and output bytes at the probed DRAM read rate; compute-bound
/// ones the BF16 MMA rate.
#[test]
fn qwen38_gemm_g4() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("Qwen GEMM benchmarks: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    let filter = std::env::var("CHOPINFER_KERNEL_BENCH_FILTER").ok();
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let modules = Modules::load(session)?;
        let (dram, source) = common::peak(session, "dram_read")?;
        // The selected topology's entries (TP4's include the rank-invariant
        // ones).
        let profile = kernel_lab::qwen38_profile()?;
        for entry in catalog(kernel_lab::catalogs()).into_iter().filter(|e| e.profile == profile) {
            let symbol = entry.symbol.as_str();
            if filter
                .as_ref()
                .is_some_and(|f| !symbol.contains(f.as_str()))
            {
                continue;
            }
            let bucket = [entry.query.m_capacity];
            let ms: &[u32] = if matches!(entry.kind, Kind::ClusterDecode { .. }) {
                &bucket
            } else if entry.query.m_capacity <= 32 {
                &[1, 8, 32]
            } else if symbol.contains("prefill") {
                &[512, 2048, 8192]
            } else {
                &[1, 8, 32, 512, 2048, 8192]
            };
            for &m in ms {
                let case = Case::prepare(session, symbol, m, 0, false, false)?;
                let function = case.function(modules.get(&entry)?)?;
                let timing = time(session, 3, 21, |stream| {
                    case.launch(session, &function, stream)
                })?;
                let q = entry.query;
                let weight_bytes: f64 = (0..4)
                    .filter_map(|s| case.buffers.get(&(WEIGHT + s)))
                    .map(|b| b.byte_len() as f64)
                    .sum();
                // Event timestamps here advance in steps of about 2 us, so
                // the per-call time also comes from a PDL chain of launches
                // over distinct weight copies (DRAM reads, as in a model).
                let copies = ((512e6 / weight_bytes) as usize).clamp(4, 32);
                let chain = (0..copies)
                    .map(|_| {
                        case.packed
                            .iter()
                            .map(|bytes| upload(session, bytes))
                            .collect::<LabResult<Vec<_>>>()
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let chained = time(session, 2, 7, |stream| {
                    for weights in &chain {
                        case.launch_weights(session, &function, stream, Some(weights))?;
                    }
                    Ok(())
                })?;
                drop(chain);
                let chain_us = f64::from(chained.median_ms) * 1000.0 / copies as f64;
                let bytes = weight_bytes
                    + f64::from(m) * f64::from(q.k) * 2.0
                    + f64::from(m)
                        * f64::from(q.n / (1 + u32::from(q.epilogues.swiglu.is_some())))
                        * if q.output == DType::F32 { 4.0 } else { 2.0 };
                let flops = 2.0 * f64::from(m) * f64::from(q.n) * f64::from(q.k);
                let memory_us = bytes / dram * 1e6;
                let compute_us = flops / 300.5e12 * 1e6;
                let median_us = f64::from(timing.median_ms) * 1000.0;
                let roofline_us = memory_us.max(compute_us);
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("qwen38_gemm_benchmark"));
                receipt.insert("symbol".into(), json!(symbol));
                receipt.insert("m".into(), json!(m));
                receipt.insert("n".into(), json!(q.n));
                receipt.insert("k".into(), json!(q.k));
                receipt.insert("grid".into(), json!(case.plan.grid[0]));
                record_ranks(&mut receipt)?;
                receipt.insert("median_us".into(), json!(median_us));
                receipt.insert("chain_calls".into(), json!(copies));
                receipt.insert("chain_us_per_call".into(), json!(chain_us));
                receipt.insert(
                    "chain_percent_of_roofline".into(),
                    json!(100.0 * roofline_us / chain_us),
                );
                receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
                receipt.insert("logical_bytes".into(), json!(bytes));
                receipt.insert("logical_flops".into(), json!(flops));
                receipt.insert("memory_roofline_us".into(), json!(memory_us));
                receipt.insert("compute_roofline_us".into(), json!(compute_us));
                receipt.insert(
                    "bound".into(),
                    json!(if memory_us >= compute_us {
                        "memory"
                    } else {
                        "compute"
                    }),
                );
                receipt.insert(
                    "percent_of_roofline".into(),
                    json!(100.0 * roofline_us / median_us),
                );
                receipt.insert("peak_source".into(), json!(source));
                receipt.insert(
                    "compute_peak_source".into(),
                    json!("probe:bf16 300.5 TFLOP/s"),
                );
                session.receipt("bench", receipt)?;
                eprintln!(
                    "{symbol} M={m}: {median_us:.2} us, chain {chain_us:.2} us per call, roofline {roofline_us:.2} us ({:.1}%)",
                    100.0 * roofline_us / chain_us
                );
            }
        }
        Ok(())
    })
    .expect("Qwen3.8 GEMM benchmarks");
}

mod common {
    use kernel_lab::{LabResult, Session};
    use serde_json::Value;

    /// The newest probe receipt for this GPU type, in bytes per second.
    pub fn peak(session: &Session, key: &str) -> LabResult<(f64, String)> {
        let root = kernel_lab::receipt_root().join("probes");
        let mut paths = std::fs::read_dir(root)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect::<Vec<_>>();
        paths.sort();
        for path in paths.into_iter().rev() {
            let value: Value =
                serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            if value.get("device_name") != session.metadata.get("device_name") {
                continue;
            }
            if let Some(gbps) = value
                .pointer(&format!("/{key}/gb_per_second"))
                .and_then(Value::as_f64)
                .filter(|v| v.is_finite() && *v > 0.0)
            {
                return Ok((
                    gbps * 1e9,
                    format!("probes/{}", path.file_name().unwrap().to_string_lossy()),
                ));
            }
        }
        Err(format!(
            "no measured {key} peak for this GPU type; run `kernel-lab probe` first"
        ))
    }
}

/// Dense outputs [m][out_n] of an entry, with F32 output if `f32`.
fn own_run(
    session: &Session,
    module: &Module,
    entry: &Entry,
    weights: &[DeviceBuffer],
    m: usize,
    rows: &[u16],
    f32: bool,
) -> LabResult<Vec<f32>> {
    let mut entry = entry.clone();
    if f32 {
        entry.query.output = DType::F32;
    }
    let q = entry.query;
    let (n, k) = (q.n as usize, q.k as usize);
    let out_n = if q.epilogues.swiglu.is_some() {
        n / 2
    } else {
        n
    };
    let plan = entry.plan(m as u32).map_err(|e| e.to_string())?;
    // (columns, first dense column) of each output buffer.
    let layout: Vec<(usize, usize)> = match entry.kind {
        Kind::ClusterDecode { .. } if out_n == n => entry
            .segments
            .iter()
            .scan(0, |start, &r| {
                let item = (r as usize, *start);
                *start += r as usize;
                Some(item)
            })
            .collect(),
        _ => vec![(out_n, 0)],
    };
    let element = if q.output == DType::F32 { 4 } else { 2 };
    let activation = session.upload_bf16(&rows[..m * k])?;
    let outputs = layout
        .iter()
        .map(|&(columns, _)| upload(session, &vec![0; m * columns * element]))
        .collect::<LabResult<Vec<_>>>()?;
    let partial = session.allocate((plan.workspace_bytes as usize).max(4))?;
    let semaphores = upload(
        session,
        &vec![0; (entry.semaphore_words() as usize).max(1) * 4],
    )?;
    let mut output_strides = [0; 4];
    for (s, &(columns, _)) in layout.iter().enumerate() {
        output_strides[s] = columns as u32;
    }
    let handles = |base: u64| std::array::from_fn(|s| base + s as u64);
    let args = entry
        .arguments(
            m as u32,
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
    let mut buffers = BTreeMap::from([
        (ACTIVATION, &activation),
        (PARTIAL, &partial),
        (SEMAPHORES, &semaphores),
    ]);
    for (s, weight) in weights.iter().enumerate() {
        buffers.insert(WEIGHT + s as u64, weight);
    }
    for (s, output) in outputs.iter().enumerate() {
        buffers.insert(OUTPUT + s as u64, output);
    }
    let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
    if plan.dynamic_shared_bytes >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    let mut checks = replay::Checks::new(session);
    for (segment, output) in outputs.iter().enumerate() {
        let element = if q.output == DType::F32 {
            DType::F32
        } else {
            DType::Bf16
        };
        checks = checks.output(
            &format!("accumulation output {segment}"),
            output,
            WriteMap::of(element).byte_prefix(output.byte_len())?,
        );
    }
    if plan.workspace_bytes != 0 {
        let bytes = plan.workspace_bytes as usize;
        checks = checks.scratch_map("accumulation partials", &partial, 0, move |_| {
            WriteMap::of(DType::F32).byte_prefix(bytes)
        });
    }
    checks.prepare()?;
    launch::launch(
        &session.context,
        &function,
        &session.stream,
        plan,
        &args,
        &Bindings { buffers },
    )?;
    checks.finish(Phase::Eager, false)?;
    let mut dense = vec![0.0_f32; m * out_n];
    for (&(columns, start), output) in layout.iter().zip(&outputs) {
        let values = if q.output == DType::F32 {
            session.download_f32(output, m * columns)?
        } else {
            session
                .download_bf16(output, m * columns)?
                .into_iter()
                .map(bf16_to_f32)
                .collect()
        };
        for t in 0..m {
            dense[t * out_n + start..][..columns]
                .copy_from_slice(&values[t * columns..][..columns]);
        }
    }
    Ok(dense)
}

/// The Qwen-owned GEMM entries on real weights and dumped activations, as
/// served and with F32 output, next to the probe's per-128-K scheme on the
/// same outputs (the gemm README's BF16 accumulation study). The fused
/// hc_up_mix and the HC precompute entries are left out: their outputs are
/// not plain GEMMs, and their parity tests tie them bitwise to hc_up_decode,
/// hc_in and hc_down.
#[test]
fn qwen38_gemm_accumulation() {
    gpu_test(24 << 30, |session| {
        let Some(sources) = accumulation::QwenSources::from_env("long_4k")? else {
            kernel_lab::skip(
                "Qwen GEMM accumulation: set CHOPINFER_QWEN38_CHECKPOINT and CHOPINFER_QWEN38_LAYER_DUMPS",
            )?;
            return Ok(());
        };
        let ranks = kernel_lab::qwen38_ranks()? as usize;
        let profile = kernel_lab::qwen38_profile()?;
        let modules = Modules::load(session)?;
        let probe = session.module(accumulation::PROBE)?;
        let mut rows = Vec::new();
        let mut findings = Vec::new();
        let mut weights: BTreeMap<String, (Vec<DeviceBuffer>, std::rc::Rc<OwnInputs>)> =
            BTreeMap::new();
        for entry in catalog(kernel_lab::catalogs()) {
            if entry.profile != profile
                || entry.symbol.contains("precompute")
                || entry.symbol.contains("_mix")
            {
                continue;
            }
            let Some(name) = entry
                .symbol
                .strip_prefix(&format!("chop_gemm_{}_", entry.profile.tag()))
                .and_then(|s| s.strip_suffix("_v1"))
            else {
                continue;
            };
            let (family, bucket) = match name.rsplit_once("_c") {
                Some((family, bucket)) if bucket.parse::<u32>().is_ok() => {
                    (family.to_string(), true)
                }
                _ => (name.to_string(), false),
            };
            let (n, k) = (entry.query.n as usize, entry.query.k as usize);
            if !weights.contains_key(&family) {
                let Some(inputs) = own_inputs(&sources, &entry, &family, ranks)? else {
                    findings.push(format!("{}: no input mapping", entry.symbol));
                    continue;
                };
                let mut packed = Vec::new();
                let mut first = 0;
                for &segment in &entry.segments {
                    let segment = segment as usize;
                    let data = FragmentWeightLayout {
                        rows: segment,
                        k,
                        format: FragmentWeightFormat::Bf16,
                    }
                    .pack(&bytes(&inputs.weight[first * k..(first + segment) * k]))
                    .map_err(|e| e.to_string())?
                    .bytes;
                    packed.push(upload(session, &data)?);
                    first += segment;
                }
                weights.insert(family.clone(), (packed, std::rc::Rc::new(inputs)));
            }
            let (packed, inputs) = &weights[&family];
            let module = modules.get(&entry)?;
            let decode = bucket || family == "hc_up_decode";
            let pool = inputs.rows.len() / k;
            let counts: Vec<usize> = if decode {
                vec![(entry.query.m_capacity as usize).min(pool)]
            } else {
                [64, 256, 2048].into_iter().filter(|&m| m <= pool).collect()
            };
            let out_n = if entry.query.epilogues.swiglu.is_some() {
                n / 2
            } else {
                n
            };
            let boundaries = entry
                .segments
                .iter()
                .scan(0, |start, &r| {
                    *start += r as usize;
                    Some(*start - 1)
                })
                .collect::<Vec<_>>();
            for m in counts {
                let columns = accumulation::study_columns(out_n, m, &boundaries);
                let batches = (1_usize << 17)
                    .div_ceil(m * columns.len())
                    .min(pool / m)
                    .max(1);
                let mut study = accumulation::Study::default();
                for batch in 0..batches {
                    let batch_rows = &inputs.rows[batch * m * k..(batch + 1) * m * k];
                    let served = own_run(session, module, &entry, packed, m, batch_rows, false)?;
                    let f32 = own_run(session, module, &entry, packed, m, batch_rows, true)?;
                    study.add(
                        session,
                        &probe,
                        (&inputs.weight, k),
                        entry.query.epilogues.swiglu.map(|_| out_n),
                        (batch_rows, m),
                        &columns,
                        &served,
                        &f32,
                    )?;
                }
                let (served, f32, chain) = study.finish(entry.query.output)?;
                if served.violations > 0 {
                    findings.push(format!(
                        "{} M={m}: served outputs over the contract bound",
                        entry.symbol
                    ));
                }
                let mut row = accumulation::study_row(
                    &entry.symbol,
                    m,
                    k,
                    inputs.origin,
                    inputs.note,
                    &served,
                    &f32,
                    &chain,
                );
                row["profile"] = json!(entry.profile.as_str());
                row["batches"] = json!(batches);
                rows.push(row);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("qwen38_entries"));
        receipt.insert("rows".into(), json!(rows));
        receipt.insert("findings".into(), json!(findings));
        record_ranks(&mut receipt)?;
        eprintln!(
            "receipt: {}",
            session.receipt("gemm_accumulation", receipt)?.display()
        );
        if findings.is_empty() {
            Ok(())
        } else {
            Err(findings.join("\n"))
        }
    })
    .expect("Qwen3.8-owned GEMM entries on real inputs");
}
