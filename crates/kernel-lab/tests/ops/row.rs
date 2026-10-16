use crate::pool_offsets::{self, SentinelPool};
use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::RotaryStyle;
use kernel_core::contract::{Argument, DType, OutputQuantization, RowQuery, RuntimeCounts};
use kernel_core::oracle::{NormArithmetic, NormOffset, NormScale, RmsNorm};
use kernel_core::rotary::{
    AngleRule, Axes, ComputedTable, FreqTable, RotaryArithmetic, RotaryPositions, RotaryRounding,
    RotarySlice, Rotation,
};
use kernel_core::{oracle, quantize};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, Timing, check_output, gpu_test, time};
use kernels_sm120::row::RowInstantiation;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, LaunchConfig, Module};
use serde_json::json;
use std::collections::BTreeMap;

/// HF's BF16 RMSNorm with eps 1e-6: F32 arithmetic, the normalized value
/// and the output each rounded to BF16.
const NORM: RmsNorm = RmsNorm {
    eps: 1e-6,
    offset: NormOffset::Plain,
    arithmetic: NormArithmetic::F32,
    scale: NormScale::MultiplyReciprocal,
    before_weight: Some(DType::Bf16),
    output: Some(DType::Bf16),
};

/// The device's NeoX rotation of the first `rotary` values of each head
/// before storage: HF's cos and sin (`AngleRule::HfF32AngleBf16CosSin`, or
/// a captured table) rounded to BF16, one F32 mate product and one F32 FMA,
/// and no output rounding.
pub(super) fn hf_rope_f32(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    rotary: usize,
    theta: f64,
    captured: Option<&[(f64, f64)]>,
) -> LabResult<Vec<f64>> {
    hf_rope(input, positions, head_dim, rotary, theta, captured, None)
}

/// `hf_rope_f32` of the first 64 values of each head with a computed table,
/// then BF16 storage, as the kernels store rotated Q and K.
pub(crate) fn hf_rope_bf16(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    theta: f64,
) -> LabResult<Vec<f64>> {
    hf_rope(
        input,
        positions,
        head_dim,
        64,
        theta,
        None,
        Some(DType::Bf16),
    )
}

fn hf_rope(
    input: &[f64],
    positions: &[usize],
    head_dim: usize,
    rotary: usize,
    theta: f64,
    captured: Option<&[(f64, f64)]>,
    output: Option<DType>,
) -> LabResult<Vec<f64>> {
    let rotation = Rotation {
        slice: RotarySlice {
            offset: 0,
            len: rotary,
        },
        pairing: RotaryStyle::Neox,
        inverse: false,
        rounding: RotaryRounding {
            table: Some(DType::Bf16),
            arithmetic: RotaryArithmetic::F32Fma,
            output,
        },
    };
    let computed = FreqTable::Computed(ComputedTable {
        theta,
        rule: AngleRule::HfF32AngleBf16CosSin,
        axes: Axes::Linear,
    });
    let table = captured.map_or(computed, FreqTable::Captured);
    rotation
        .apply(input, head_dim, RotaryPositions::Linear(positions), &table)
        .map_err(|e| e.to_string())
}

/// Every entry of the lab's row engine cubin, which holds all profiles.
fn catalog() -> Vec<RowInstantiation> {
    kernels_sm120::row::catalog(kernel_lab::catalogs())
}

const ROW_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/row_engine.cubin"));

/// The fill of sorted route rows, whose -1 padding is all-ones: 0x80808080
/// is neither a route index nor -1.
const ROUTE_FILL: u8 = 0x80;

#[path = "row_invariance.rs"]
pub(crate) mod invariance;
#[path = "row_live.rs"]
mod live;

struct Fixture<'a> {
    session: &'a Session,
    buffers: BTreeMap<String, DeviceBuffer>,
    scalars: BTreeMap<String, u32>,
    wide_scalars: BTreeMap<String, u64>,
}

impl<'a> Fixture<'a> {
    fn new(session: &'a Session, rows: usize) -> LabResult<Self> {
        let mut fixture = Self {
            session,
            buffers: BTreeMap::new(),
            scalars: BTreeMap::new(),
            wide_scalars: BTreeMap::new(),
        };
        fixture.u32("live_rows", &[rows as u32])?;
        fixture.u32("live_tokens", &[rows as u32])?;
        fixture.scalar("eps_bits", 1e-6_f32.to_bits());
        fixture.scalar("scale_bits", 1.0_f32.to_bits());
        Ok(fixture)
    }

    fn scalar(&mut self, name: &str, value: u32) {
        self.scalars.insert(name.into(), value);
    }

    fn scalar64(&mut self, name: &str, value: u64) {
        self.wide_scalars.insert(name.into(), value);
    }

    fn bytes(&mut self, name: &str, data: &[u8]) -> LabResult<()> {
        let buffer = self.session.allocate(data.len().max(1))?;
        buffer
            .copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.insert(name.into(), buffer);
        Ok(())
    }

    fn zero(&mut self, name: &str, bytes: usize) -> LabResult<()> {
        self.bytes(name, &vec![0; bytes])
    }

    /// An output allocated with poison, so its first launch must write it.
    fn poisoned(&mut self, name: &str, bytes: usize) -> LabResult<()> {
        self.bytes(name, &vec![POISON; bytes])
    }

    fn bf16(&mut self, name: &str, values: &[u16]) -> LabResult<()> {
        self.bytes(
            name,
            &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn f32(&mut self, name: &str, values: &[f32]) -> LabResult<()> {
        self.bytes(
            name,
            &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn i32(&mut self, name: &str, values: &[i32]) -> LabResult<()> {
        self.bytes(
            name,
            &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn u32(&mut self, name: &str, values: &[u32]) -> LabResult<()> {
        self.bytes(
            name,
            &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    }

    fn read_bytes(&self, name: &str, len: usize) -> LabResult<Vec<u8>> {
        let mut out = vec![0; len];
        self.buffers[name]
            .copy_to_bytes(&mut out, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }

    fn read_bf16(&self, name: &str, len: usize) -> LabResult<Vec<u16>> {
        Ok(self
            .read_bytes(name, len * 2)?
            .chunks_exact(2)
            .map(|x| u16::from_ne_bytes([x[0], x[1]]))
            .collect())
    }

    fn read_i32(&self, name: &str, len: usize) -> LabResult<Vec<i32>> {
        Ok(self
            .read_bytes(name, len * 4)?
            .chunks_exact(4)
            .map(|x| i32::from_ne_bytes(x.try_into().expect("i32 bytes")))
            .collect())
    }

    fn read_f32(&self, name: &str, len: usize) -> LabResult<Vec<f32>> {
        Ok(self
            .read_bytes(name, len * 4)?
            .chunks_exact(4)
            .map(|x| f32::from_ne_bytes(x.try_into().expect("f32 bytes")))
            .collect())
    }

    fn launch(
        &self,
        module: &Module,
        entry: &RowInstantiation,
        capacity: u32,
        rows: u32,
    ) -> LabResult<()> {
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        self.launch_bound(&function, entry, capacity, rows)
    }

    fn launch_bound(
        &self,
        function: &Function,
        entry: &RowInstantiation,
        capacity: u32,
        rows: u32,
    ) -> LabResult<()> {
        let mut bindings = BTreeMap::new();
        let mut handles = BTreeMap::new();
        for (index, (name, buffer)) in self.buffers.iter().enumerate() {
            let handle = index as u64 + 1;
            bindings.insert(handle, buffer);
            handles.insert(name.as_str(), handle);
        }
        let args = entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => {
                    let buffer_name = match name.as_str() {
                        "k_ring" => "k_pool",
                        "v_ring" => "v_pool",
                        other => other,
                    };
                    Ok(handles
                        .get(buffer_name)
                        .map_or(Argument::NullDevicePointer, |&handle| {
                            Argument::DeviceBuffer(handle)
                        }))
                }
                EntryParam::U32 { name } => self
                    .scalars
                    .get(name)
                    .copied()
                    .map(Argument::U32)
                    .ok_or_else(|| format!("missing scalar {name}")),
                EntryParam::U64 { name } => self
                    .wide_scalars
                    .get(name)
                    .copied()
                    .map(Argument::U64)
                    .ok_or_else(|| format!("missing u64 scalar {name}")),
                EntryParam::I32 { name } => Err(format!("unexpected i32 argument {name}")),
                EntryParam::TensorMap { name } => Err(format!("unexpected tensor map {name}")),
                EntryParam::PointerArray { name, .. } => {
                    Err(format!("unexpected pointer array {name}"))
                }
            })
            .collect::<LabResult<Vec<_>>>()?;
        let args = entry.pack_arguments(args).map_err(|e| e.to_string())?;
        let plan = entry
            .plan(
                capacity,
                RuntimeCounts {
                    m: rows,
                    batch: rows,
                    context: 0,
                    routed_rows: rows.saturating_mul(8),
                    split_k: 1,
                },
            )
            .map_err(|e| e.to_string())?;
        launch::launch(
            &self.session.context,
            function,
            &self.session.stream,
            plan,
            &args,
            &Bindings { buffers: bindings },
        )?;
        kernel_lab::launched(&entry.symbol);
        Ok(())
    }

    fn launch_shared_paged(
        &self,
        module: &Module,
        entry: &RowInstantiation,
        pool: &DeviceBuffer,
        layout: (usize, usize, usize),
        rows: u32,
    ) -> LabResult<()> {
        let (k_base, v_base, pool_bytes) = layout;
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let args = entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } if name == "k_pool" => {
                    KernelArgument::Range {
                        buffer: pool,
                        offset: k_base,
                        byte_len: pool_bytes - k_base,
                    }
                }
                EntryParam::DevicePointer { name, .. } if name == "v_pool" => {
                    KernelArgument::Range {
                        buffer: pool,
                        offset: v_base,
                        byte_len: pool_bytes - v_base,
                    }
                }
                EntryParam::DevicePointer { name, .. } => self
                    .buffers
                    .get(name)
                    .map_or(KernelArgument::NullBuffer, KernelArgument::Buffer),
                EntryParam::U32 { name } => KernelArgument::U32(self.scalars[name]),
                EntryParam::U64 { name } => KernelArgument::U64(self.wide_scalars[name]),
                other => panic!("unexpected shared-pool argument {other:?}"),
            })
            .collect::<Vec<_>>();
        let plan = entry
            .plan(
                rows,
                RuntimeCounts {
                    m: rows,
                    batch: rows,
                    context: 0,
                    routed_rows: 0,
                    split_k: 1,
                },
            )
            .map_err(|e| e.to_string())?;
        function
            .launch(&self.session.stream, launch::config(plan)?, &args)
            .map_err(|e| e.to_string())
    }

    fn check_exact<T: PartialEq + std::fmt::Debug>(
        &self,
        case: &str,
        actual: &[T],
        expected: &[T],
    ) -> LabResult<()> {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case));
        receipt.insert("correct".into(), json!(actual == expected));
        let failure = oracle::check_exact(actual, expected).err().map(|index| {
            format!(
                "{case} mismatch at {index}: actual {:?}, expected {:?}",
                actual.get(index),
                expected.get(index)
            )
        });
        if let Some(ref message) = failure {
            receipt.insert("worst_element".into(), json!(message));
        }
        self.session.receipt("row", receipt)?;
        failure.map_or(Ok(()), Err)
    }

    fn check_numeric(
        &self,
        case: &str,
        actual: &[f32],
        expected: &oracle::OracleOutput,
        dtype: DType,
    ) -> LabResult<()> {
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(case));
        let gate = check_output(actual, expected, dtype, &mut receipt);
        let worst = receipt.get("worst_element").cloned();
        self.session.receipt("row", receipt)?;
        gate.map_err(|message| format!("{case}: {message}; worst element {worst:?}"))
    }
}

fn find<'a>(entries: &'a [RowInstantiation], name: &str) -> &'a RowInstantiation {
    entries
        .iter()
        .find(|entry| entry.symbol == format!("chop_row_{name}_v1"))
        .unwrap_or_else(|| panic!("missing row entry {name}"))
}

fn case_enabled(name: &str, rows: usize) -> bool {
    let entry_matches = match std::env::var("CHOPINFER_ROW_ENTRY") {
        Ok(selected) => selected == name,
        Err(_) => true,
    };
    let rows_match = match std::env::var("CHOPINFER_ROW_ROWS") {
        Ok(selected) => selected.parse::<usize>() == Ok(rows),
        Err(_) => true,
    };
    entry_matches && rows_match
}

fn seeded_bf16(len: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16(((state >> 40) as i32 as f32 / 8_388_608.0 - 1.0) * 0.7)
        })
        .collect()
}

fn seeded_f32(len: usize, seed: u64) -> Vec<f32> {
    seeded_bf16(len, seed)
        .into_iter()
        .map(bf16_to_f32)
        .collect()
}

fn as_f64(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(bf16_to_f32(x))).collect()
}

fn module(session: &Session) -> LabResult<Module> {
    session
        .context
        .load_module(ROW_CUBIN)
        .map_err(|e| e.to_string())
}

/// A buffer an entry writes, and what its replay checks expect of it.
enum Written {
    /// Every byte in the map is written; every other byte keeps the poison.
    Output(&'static str, WriteMap),
    /// As `Output`, filled with another byte where all-ones elements are
    /// valid values (-1 route padding).
    Filled(&'static str, WriteMap, u8),
    /// Packed E2M1 codes: every byte value is valid, so the written range
    /// starts from the complement of the first launch's codes.
    Codes(&'static str, WriteMap),
    /// A KV cache: the map's rows are written without being read and start
    /// poisoned; every other byte keeps the first launch's image.
    Cache(&'static str, WriteMap),
}

impl Written {
    fn name(&self) -> &'static str {
        match self {
            Written::Output(name, _)
            | Written::Filled(name, ..)
            | Written::Codes(name, _)
            | Written::Cache(name, _) => name,
        }
    }
}

/// A relaunch, a graph replay and a PDL launch of `entry`, each on poisoned
/// outputs and caches restored to the first launch's image with their
/// written rows poisoned, must reproduce the first launch bit for bit. The
/// PDL producer delivers the entry's live count (`live_rows`, or
/// `live_tokens` for RoPE and ring writes) only after its dependents may
/// start, so an entry that reads it before waiting writes nothing.
fn replay(
    session: &Session,
    module: &Module,
    fixture: &mut Fixture<'_>,
    entry: &RowInstantiation,
    rows: usize,
    written: &[Written],
) -> LabResult<()> {
    let live = if entry.params.iter().any(
        |param| matches!(param, EntryParam::DevicePointer { name, .. } if name == "live_tokens"),
    ) {
        "live_tokens"
    } else {
        "live_rows"
    };
    let count = fixture.read_bytes(live, 4)?;
    fixture.bytes("live_source", &count)?;
    let fixture = &*fixture;
    let first = written
        .iter()
        .map(|w| poison::read(session, &fixture.buffers[w.name()]))
        .collect::<LabResult<Vec<_>>>()?;
    let mut checks = Replay::new(session);
    for (w, image) in written.iter().zip(&first) {
        let buffer = &fixture.buffers[w.name()];
        checks = match w {
            Written::Output(name, map) => checks.output(*name, buffer, map.clone()),
            Written::Filled(name, map, byte) => {
                checks.output_with(*name, buffer, map.clone(), &[*byte])
            }
            Written::Codes(name, map) => {
                checks.output_complement(*name, buffer, map.clone(), image.clone())
            }
            Written::Cache(name, map) => {
                checks.state(*name, buffer, image.clone(), map.clone(), Vec::new())
            }
        };
    }
    let symbol = &entry.symbol;
    let launch = || fixture.launch(module, entry, rows as u32, rows as u32);
    // The relaunch against the first launch, which the oracle checked.
    checks
        .eager(launch)
        .map_err(|e| format!("{symbol} determinism: {e}"))?;
    for (w, image) in written.iter().zip(&first) {
        fixture.check_exact(
            &format!("{symbol} determinism {}", w.name()),
            checks.expected(w.name()).ok_or("missing eager bytes")?,
            image,
        )?;
    }
    let graph = session.capture(|_| launch())?;
    checks
        .graph(&graph)
        .map_err(|e| format!("{symbol} graph: {e}"))?;
    fixture.check_exact(
        &format!("{symbol} graph"),
        &poison::read(session, &fixture.buffers[written[0].name()])?,
        &first[0],
    )?;
    let probe = session.module("probe_sm120")?;
    let copy = probe
        .function("chop_probe_copy_v1")
        .map_err(|e| e.to_string())?;
    checks
        .pdl(
            || {
                poison::write(session, &fixture.buffers[live], 0, &0_u32.to_ne_bytes())?;
                copy.launch(
                    &session.stream,
                    launch::simple_config(
                        (1, 1, 1),
                        (256, 1, 1),
                        0,
                        LaunchConfig::NO_CLUSTER,
                        false,
                    )?,
                    &[
                        KernelArgument::Buffer(&fixture.buffers["live_source"]),
                        KernelArgument::Buffer(&fixture.buffers[live]),
                        KernelArgument::U64(1),
                    ],
                )
                .map_err(|e| e.to_string())
            },
            launch,
            true,
        )
        .map_err(|e| format!("{symbol} PDL: {e}"))?;
    fixture.check_exact(
        &format!("{symbol} PDL"),
        &poison::read(session, &fixture.buffers[written[0].name()])?,
        &first[0],
    )
}

fn quant_reference(
    values: &[u16],
    rows: usize,
    width: usize,
    format: kernel_core::contract::OutputQuantization,
) -> LabResult<quantize::Quantized> {
    use kernel_core::contract::OutputQuantization;
    let floats = values.iter().map(|&x| bf16_to_f32(x)).collect::<Vec<_>>();
    let result = match format {
        OutputQuantization::Fp8Group128(mode) => {
            quantize::fp8_activation_1x128(&floats, rows, width, mode)
        }
        OutputQuantization::Mxfp8 => quantize::mxfp8(&floats, rows, width),
        OutputQuantization::Mxfp4 => quantize::mxfp4(&floats, rows, width),
        OutputQuantization::Nvfp4 => quantize::nvfp4(&floats, rows, width),
    };
    result.map_err(|e| e.to_string())
}

fn check_quant(
    fixture: &Fixture<'_>,
    case: &str,
    reference: &quantize::Quantized,
) -> LabResult<()> {
    fixture.check_exact(
        &format!("{case} codes"),
        &fixture.read_bytes("codes", reference.codes.len())?,
        &reference.codes,
    )?;
    match &reference.scales {
        quantize::Scales::F32(values) => {
            let actual = fixture.read_f32("scales", values.len())?;
            fixture.check_exact(
                &format!("{case} scales"),
                &actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                &values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            )
        }
        quantize::Scales::Ue8M0(values) | quantize::Scales::Ue4M3(values) => fixture.check_exact(
            &format!("{case} scales"),
            &fixture.read_bytes("scales", values.len())?,
            values,
        ),
    }
}

fn norm_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    for name in [
        "rms_mimo_hf4096",
        "rms_mimo_hf4096_residual",
        "rms_plain4096",
        "rms_plain8192",
        "rms_gemma4096",
        "rms_qk128",
    ] {
        let entry = find(entries, name);
        let width = entry.width as usize;
        let x = seeded_bf16(rows * width, 0xabc0 + width as u64);
        let weight = seeded_bf16(width, 0xdef0 + width as u64);
        let residual = seeded_bf16(rows * width, 0x1230 + width as u64);
        let fused = matches!(
            entry.query,
            RowQuery::RmsNorm {
                fused_residual: true,
                ..
            }
        );
        let mut fixture = Fixture::new(session, rows)?;
        fixture.bf16("x", &x)?;
        fixture.bf16("weight", &weight)?;
        if fused {
            fixture.bf16("residual", &residual)?;
            fixture.poisoned("residual_out", rows * width * 2)?;
        }
        fixture.poisoned("y", rows * width * 2)?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        let (kind, _) = match entry.query {
            RowQuery::RmsNorm {
                kind,
                output_quantization,
                ..
            } => (kind, output_quantization),
            _ => unreachable!(),
        };
        let x64 = as_f64(&x);
        let w64 = as_f64(&weight);
        let r64 = as_f64(&residual);
        let expected = oracle::rmsnorm(
            &x64,
            width,
            &w64,
            1e-6,
            kind,
            fused.then_some(r64.as_slice()),
        )
        .map_err(|e| e.to_string())?;
        let actual = fixture.read_bf16("y", rows * width)?;
        if kind == kernel_core::contract::RmsNormKind::HfBf16Intermediate {
            let source = if fused {
                let added = x
                    .iter()
                    .zip(&residual)
                    .map(|(&a, &b)| f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b)))
                    .collect::<Vec<_>>();
                fixture.check_exact(
                    &format!("{name} residual"),
                    &fixture.read_bf16("residual_out", rows * width)?,
                    &added,
                )?;
                added
            } else {
                x.clone()
            };
            let exact = NORM
                .apply_bf16(&source, &weight, width)
                .map_err(|e| e.to_string())?;
            fixture.check_exact(name, &actual, &exact)?;
        } else {
            fixture.check_numeric(name, &as_f32(&actual), &expected, DType::Bf16)?;
        }
        if rows == 1 {
            let mut written = vec![Written::Output(
                "y",
                WriteMap::of(DType::Bf16).prefix(rows * width)?,
            )];
            if fused {
                written.push(Written::Output(
                    "residual_out",
                    WriteMap::of(DType::Bf16).prefix(rows * width)?,
                ));
            }
            replay(session, module, &mut fixture, entry, rows, &written)?;
        }
    }
    Ok(())
}

fn as_f32(values: &[u16]) -> Vec<f32> {
    values.iter().map(|&x| bf16_to_f32(x)).collect()
}

/// Codes and scales allocated so an unwritten one shows: packed E2M1 codes,
/// where every byte is a valid pair, start from the complement of the
/// reference codes; E4M3 codes and every scale format start from poison.
fn poisoned_codes(
    fixture: &mut Fixture<'_>,
    reference: &quantize::Quantized,
    format: OutputQuantization,
) -> LabResult<()> {
    if packed_e2m1(format) {
        fixture.bytes(
            "codes",
            &reference.codes.iter().map(|byte| !byte).collect::<Vec<_>>(),
        )?;
    } else {
        fixture.poisoned("codes", reference.codes.len())?;
    }
    fixture.poisoned("scales", scale_bytes(reference))
}

fn packed_e2m1(format: OutputQuantization) -> bool {
    matches!(
        format,
        OutputQuantization::Mxfp4 | OutputQuantization::Nvfp4
    )
}

fn scale_bytes(reference: &quantize::Quantized) -> usize {
    reference.scale_rows()
        * reference.scale_cols()
        * if matches!(reference.scales, quantize::Scales::F32(_)) {
            4
        } else {
            1
        }
}

/// Every code and scale byte is written (`check_quant` compares them all).
fn quant_written(
    reference: &quantize::Quantized,
    format: OutputQuantization,
) -> LabResult<Vec<Written>> {
    let codes = WriteMap::of(DType::U8).prefix(reference.codes.len())?;
    let scale_element = if matches!(reference.scales, quantize::Scales::F32(_)) {
        DType::F32
    } else {
        DType::U8
    };
    Ok(vec![
        if packed_e2m1(format) {
            Written::Codes("codes", codes)
        } else {
            Written::Output("codes", codes)
        },
        Written::Output(
            "scales",
            WriteMap::of(scale_element).byte_prefix(scale_bytes(reference))?,
        ),
    ])
}

fn quant_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    use kernel_core::contract::OutputQuantization;
    for name in [
        "quant_fp8_exact_4096",
        "quant_fp8_pow2_4096",
        "quant_mxfp8_4096",
        "quant_mxfp4_4096",
        "quant_nvfp4_4096",
        "quant_fp8_exact_8192",
        "quant_mxfp8_8192",
        "quant_mxfp4_8192",
        "quant_nvfp4_8192",
        "quant_fp8_exact_4096_std",
        "quant_mxfp8_4096_std",
        "quant_mxfp4_4096_std",
        "quant_nvfp4_4096_std",
    ] {
        let entry = find(entries, name);
        let width = entry.width as usize;
        let input = seeded_bf16(rows * width, 0x8042 + width as u64);
        let format = match entry.query {
            RowQuery::Quantize { format, .. } => format,
            _ => unreachable!(),
        };
        let reference = quant_reference(&input, rows, width, format)?;
        let mut fixture = Fixture::new(session, rows)?;
        fixture.bf16("input", &input)?;
        poisoned_codes(&mut fixture, &reference, format)?;
        fixture.f32("global_scale", &[reference.global_scale])?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        check_quant(&fixture, name, &reference)?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                entry,
                rows,
                &quant_written(&reference, format)?,
            )?;
        }
        if format == OutputQuantization::Nvfp4 {
            let suffix = if name.ends_with("_std") { "_std" } else { "" };
            let amax = find(entries, &format!("nvfp4_row_amax_{width}{suffix}"));
            let mut prep = Fixture::new(session, rows)?;
            prep.bf16("input", &input)?;
            prep.zero("row_amax", rows * 4)?;
            prep.launch(module, amax, rows as u32, rows as u32)?;
            let actual = prep.read_f32("row_amax", rows)?;
            let expected = input
                .chunks(width)
                .map(|row| {
                    row.iter()
                        .map(|&x| bf16_to_f32(x).abs())
                        .fold(0.0_f32, f32::max)
                })
                .collect::<Vec<_>>();
            prep.check_exact("nvfp4 row amax", &actual, &expected)?;
            if width == 4096 {
                let global = find(entries, &format!("nvfp4_global_scale{suffix}"));
                prep.zero("scale", 4)?;
                prep.launch(module, global, rows as u32, rows as u32)?;
                prep.check_exact(
                    "nvfp4 global scale",
                    &prep.read_f32("scale", 1)?[0].to_bits().to_ne_bytes(),
                    &reference.global_scale.to_bits().to_ne_bytes(),
                )?;
            }
        }
    }
    Ok(())
}

fn router_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    for name in [
        "router_mimo_sigmoid256",
        "router_std_softmax128",
        "router_std_grouped128",
    ] {
        let entry = find(entries, name);
        let RowQuery::Router { experts, kind, .. } = entry.query else {
            unreachable!()
        };
        let experts = experts as usize;
        let mut logits = seeded_f32(rows * experts, 0x5060 + experts as u64);
        let mut bias = seeded_f32(experts, 0x7070 + experts as u64)
            .into_iter()
            .map(|x| x * 0.1)
            .collect::<Vec<_>>();
        bias[0] = 0.0;
        bias[1] = 0.0;
        for row in 0..rows {
            logits[row * experts] = 3.0;
            logits[row * experts + 1] = 3.0;
        }
        let mut fixture = Fixture::new(session, rows)?;
        fixture.f32("logits", &logits)?;
        fixture.f32("bias", &bias)?;
        fixture.poisoned("ids", rows * 8 * 4)?;
        fixture.poisoned("weights", rows * 8 * 4)?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        let expected = (0..rows)
            .map(|row| {
                oracle::router(
                    &logits[row * experts..(row + 1) * experts]
                        .iter()
                        .map(|&x| x as f64)
                        .collect::<Vec<_>>(),
                    Some(&bias.iter().map(|&x| x as f64).collect::<Vec<_>>()),
                    8,
                    kind,
                )
                .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        let ids = expected
            .iter()
            .flat_map(|selection| selection.expert_ids.iter().map(|&id| id as i32))
            .collect::<Vec<_>>();
        fixture.check_exact(
            &format!("{name} ids"),
            &fixture.read_i32("ids", rows * 8)?,
            &ids,
        )?;
        let weights = expected
            .iter()
            .flat_map(|selection| selection.weights.iter().copied())
            .collect::<Vec<_>>();
        let oracle_output = oracle::OracleOutput::new(weights, vec![1.0; rows * 8], 8)
            .map_err(|e| e.to_string())?;
        fixture.check_numeric(
            &format!("{name} weights"),
            &fixture.read_f32("weights", rows * 8)?,
            &oracle_output,
            DType::F32,
        )?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                entry,
                rows,
                &[
                    Written::Output("ids", WriteMap::of(DType::I32).prefix(8)?),
                    Written::Output("weights", WriteMap::of(DType::F32).prefix(8)?),
                ],
            )?;
        }
    }
    Ok(())
}

fn permute_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    for (name, experts, block) in [("mimo", 256, 32), ("std", 128, 64)] {
        let ids = (0..rows * 8)
            .map(|route| ((route * 53 + route / 8) % experts) as i32)
            .collect::<Vec<_>>();
        let oracle = oracle::expert_permute(
            &ids.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            experts,
            block,
        )
        .map_err(|e| e.to_string())?;
        let padded_capacity = rows * 8 + experts * (block - 1);
        let mut fixture = Fixture::new(session, rows)?;
        fixture.i32("ids", &ids)?;
        fixture.poisoned("counts", experts * 4)?;
        fixture.poisoned("offsets", (experts + 1) * 4)?;
        fixture.poisoned("padded_rows", 4)?;
        // -1 is written padding, so these rows start from another byte.
        fixture.bytes("sorted_rows", &vec![ROUTE_FILL; padded_capacity * 4])?;
        let count = find(entries, &format!("permute_count_{name}"));
        let scan = find(entries, &format!("permute_scan_{name}"));
        let scatter = find(entries, &format!("permute_scatter_{name}"));
        fixture.launch(module, count, rows as u32, rows as u32)?;
        fixture.launch(module, scan, rows as u32, rows as u32)?;
        fixture.launch(module, scatter, rows as u32, rows as u32)?;
        let offsets = oracle
            .expert_offsets
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        fixture.check_exact(
            &format!("{name} offsets"),
            &fixture.read_i32("offsets", experts + 1)?,
            &offsets,
        )?;
        fixture.check_exact(
            &format!("{name} padded rows"),
            &fixture.read_i32("padded_rows", 1)?,
            &[oracle.forward.len() as i32],
        )?;
        let sorted = oracle
            .forward
            .iter()
            .map(|route| route.map_or(-1, |x| x as i32))
            .collect::<Vec<_>>();
        fixture.check_exact(
            &format!("{name} sorted rows"),
            &fixture.read_i32("sorted_rows", sorted.len())?,
            &sorted,
        )?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                count,
                rows,
                &[Written::Output(
                    "counts",
                    WriteMap::of(DType::I32).prefix(experts)?,
                )],
            )?;
            replay(
                session,
                module,
                &mut fixture,
                scan,
                rows,
                &[
                    Written::Output("offsets", WriteMap::of(DType::I32).prefix(experts + 1)?),
                    Written::Output("padded_rows", WriteMap::of(DType::I32).prefix(1)?),
                ],
            )?;
            // Scatter fills the padded rows; the capacity past them is
            // untouched (router.cuh).
            replay(
                session,
                module,
                &mut fixture,
                scatter,
                rows,
                &[Written::Filled(
                    "sorted_rows",
                    WriteMap::of(DType::I32).prefix(sorted.len())?,
                    ROUTE_FILL,
                )],
            )?;
        }
        if rows <= 32 {
            let input = seeded_bf16(rows * 4096, 0xf121);
            let expected = oracle
                .forward
                .iter()
                .flat_map(|route| {
                    route.map_or_else(
                        || vec![0; 4096],
                        |r| input[r / 8 * 4096..(r / 8 + 1) * 4096].to_vec(),
                    )
                })
                .collect::<Vec<_>>();
            fixture.bf16("input", &input)?;
            fixture.poisoned("output", expected.len() * 2)?;
            let gather = find(entries, &format!("gather_{name}"));
            fixture.launch(module, gather, rows as u32, rows as u32)?;
            fixture.check_exact(
                &format!("{name} gather"),
                &fixture.read_bf16("output", expected.len())?,
                &expected,
            )?;
            if rows == 1 {
                replay(
                    session,
                    module,
                    &mut fixture,
                    gather,
                    rows,
                    &[Written::Output(
                        "output",
                        WriteMap::of(DType::Bf16).prefix(expected.len())?,
                    )],
                )?;
            }
            let quant = quant_reference(
                &expected,
                sorted.len(),
                4096,
                kernel_core::contract::OutputQuantization::Mxfp8,
            )?;
            poisoned_codes(&mut fixture, &quant, OutputQuantization::Mxfp8)?;
            let gather_mx = find(entries, &format!("gather_mxfp8_{name}"));
            fixture.launch(module, gather_mx, rows as u32, rows as u32)?;
            check_quant(&fixture, &format!("{name} gather MXFP8"), &quant)?;
        }
    }
    Ok(())
}

fn simple_row_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    for name in ["combine_mimo4096", "combine_std4096", "combine_std8192"] {
        let entry = find(entries, name);
        let width = entry.width as usize;
        let preweighted = name.contains("mimo") || name.ends_with("8192");
        let routed = seeded_bf16(rows * 8 * width, 0x45c0 + width as u64);
        let weights = (0..rows * 8)
            .map(|i| 0.08 + (i % 8) as f32 * 0.01)
            .collect::<Vec<_>>();
        let mut fixture = Fixture::new(session, rows)?;
        fixture.bf16("routed", &routed)?;
        fixture.f32("weights", &weights)?;
        fixture.poisoned("output", rows * width * 2)?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        let tokens = (0..rows * 8).map(|r| r / 8).collect::<Vec<_>>();
        let effective = if preweighted {
            vec![1.0; rows * 8]
        } else {
            weights.iter().map(|&x| x as f64).collect()
        };
        let expected = oracle::expert_combine(&as_f64(&routed), &tokens, &effective, rows, width)
            .map_err(|e| e.to_string())?;
        fixture.check_numeric(
            name,
            &as_f32(&fixture.read_bf16("output", rows * width)?),
            &expected,
            DType::Bf16,
        )?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                entry,
                rows,
                &[Written::Output(
                    "output",
                    WriteMap::of(DType::Bf16).prefix(rows * width)?,
                )],
            )?;
        }
    }
    for name in [
        "embedding_full",
        "embedding_shard",
        "embedding_shard_std",
        "embedding_qwen38",
    ] {
        let entry = find(entries, name);
        let width = entry.width as usize;
        let vocab = 64_usize;
        let offset = if name.contains("shard") { 100 } else { 0 };
        let table = seeded_bf16(vocab * width, 0x1301);
        let ids = (0..rows)
            .map(|i| (i % (vocab + if offset == 0 { 0 } else { 4 }) + offset) as i32)
            .collect::<Vec<_>>();
        let mut fixture = Fixture::new(session, rows)?;
        fixture.bf16("table", &table)?;
        fixture.i32("ids", &ids)?;
        fixture.poisoned("output", rows * width * 2)?;
        fixture.scalar("vocab_offset", offset as u32);
        fixture.scalar("shard_vocab", vocab as u32);
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        let table_f64 = as_f64(&table);
        let ids_usize = ids.iter().map(|&x| x as usize).collect::<Vec<_>>();
        let expected =
            oracle::sharded_embedding_gather(&table_f64, vocab, width, offset, &ids_usize)
                .map_err(|e| e.to_string())?;
        fixture.check_exact(
            name,
            &fixture.read_bf16("output", rows * width)?,
            &expected
                .iter()
                .map(|&x| f32_to_bf16(x as f32))
                .collect::<Vec<_>>(),
        )?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                entry,
                rows,
                &[Written::Output(
                    "output",
                    WriteMap::of(DType::Bf16).prefix(rows * width)?,
                )],
            )?;
        }
    }
    for name in [
        "swiglu_halves_4096",
        "swiglu_interleaved_4096",
        "swiglu_halves_7168_std",
    ] {
        if !entries
            .iter()
            .any(|entry| entry.symbol == format!("chop_row_{name}_v1"))
        {
            continue;
        }
        let entry = find(entries, name);
        let width = entry.width as usize;
        let input = seeded_bf16(rows * width * 2, 0x8751 + width as u64);
        let interleaved = name.contains("interleaved");
        let mut fixture = Fixture::new(session, rows)?;
        fixture.bf16("input", &input)?;
        fixture.poisoned("output", rows * width * 2)?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        let expected = oracle::swiglu(
            &as_f64(&input),
            width,
            if interleaved {
                kernel_core::contract::SwigluColumns::Interleaved
            } else {
                kernel_core::contract::SwigluColumns::ConcatenatedHalves
            },
        )
        .map_err(|e| e.to_string())?;
        fixture.check_numeric(
            name,
            &as_f32(&fixture.read_bf16("output", rows * width)?),
            &expected,
            DType::Bf16,
        )?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                entry,
                rows,
                &[Written::Output(
                    "output",
                    WriteMap::of(DType::Bf16).prefix(rows * width)?,
                )],
            )?;
        }
    }
    Ok(())
}

fn logits_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    for name in [
        "logits_argmax_full",
        "logits_top20_full",
        "logits_argmax_shard",
        "logits_top20_shard",
        "logits_argmax_shard_std",
        "logits_top20_shard_std",
        "logits_argmax_qwen38_full",
        "logits_top20_qwen38_full",
        "logits_argmax_qwen38_shard",
        "logits_top20_qwen38_shard",
    ] {
        if rows > 32 && name.contains("full") {
            continue;
        }
        let entry = find(entries, name);
        let width = entry.width as usize;
        let live_width = width - 32;
        let k = if name.contains("top20") { 20 } else { 1 };
        let mut logits = seeded_f32(rows * width, 0x99f0 + width as u64);
        for row in 0..rows {
            logits[row * width] = 2.0;
            logits[row * width + 1] = 2.0;
            for value in &mut logits[row * width + live_width..(row + 1) * width] {
                *value = 20.0;
            }
        }
        let mut fixture = Fixture::new(session, rows)?;
        fixture.f32("logits", &logits)?;
        fixture.poisoned("ids", rows * k * 4)?;
        fixture.poisoned("log_probs", rows * k * 4)?;
        fixture.u32("live_width", &[live_width as u32])?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        let expected = logits
            .chunks(width)
            .map(|row| {
                oracle::logits_topk_log_probs(
                    &row.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                    live_width,
                    k,
                )
                .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        let ids = expected
            .iter()
            .flat_map(|row| row.ids.iter().map(|&x| x as i32))
            .collect::<Vec<_>>();
        fixture.check_exact(
            &format!("{name} ids"),
            &fixture.read_i32("ids", rows * k)?,
            &ids,
        )?;
        let values = expected
            .iter()
            .flat_map(|row| row.log_probs.iter().copied())
            .collect::<Vec<_>>();
        let reference = oracle::OracleOutput::new(values, vec![1.0; rows * k], live_width)
            .map_err(|e| e.to_string())?;
        fixture.check_numeric(
            &format!("{name} log probabilities"),
            &fixture.read_f32("log_probs", rows * k)?,
            &reference,
            DType::F32,
        )?;
        if rows == 1 {
            replay(
                session,
                module,
                &mut fixture,
                entry,
                rows,
                &[
                    Written::Output("ids", WriteMap::of(DType::I32).prefix(k)?),
                    Written::Output("log_probs", WriteMap::of(DType::F32).prefix(k)?),
                ],
            )?;
        }
    }
    Ok(())
}

fn rope_case(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
    name: &str,
) -> LabResult<()> {
    if !case_enabled(name, rows) {
        return Ok(());
    }
    eprintln!("row RoPE case: rows={rows} entry={name}");
    let entry = find(entries, name);
    let std = name.contains("std_gptj");
    let ring = name.contains("swa");
    let prefill = name.contains("prefill");
    let fp8 = name.contains("fp8");
    let table = name.contains("table");
    let pos64 = name.contains("pos64");
    // MiMo's heads per rank: TP4 16 Q with 1 GA or 2 SWA KV heads, TP2
    // twice as many.
    let (qh, kvh) = if name.contains("_tp2_") {
        (32_usize, if ring { 4 } else { 2 })
    } else {
        (16, if std || ring { 2 } else { 1 })
    };
    let qk = if std { 128 } else { 192 };
    let v = 128_usize;
    let rotary = if std { 128 } else { 64 };
    let theta = if name.contains("_ga_") {
        1e7_f32
    } else {
        1e4_f32
    };
    let page_size = if name.contains("_p16_") {
        16
    } else if name.contains("_p64_") {
        64
    } else {
        32
    };
    let position_base = if rows == 1 { 127 } else { 0 };
    let positions = (0..rows).map(|i| position_base + i).collect::<Vec<_>>();
    let max_position = *positions.last().expect("nonempty positions");
    let stride = qh * qk + kvh * (qk + v);
    let input = seeded_bf16(rows * stride, 0x5511 + rows as u64);
    let pairs = rotary / 2;
    // The standard profile's device table is all F32.
    let std_table = ComputedTable {
        theta: f64::from(theta),
        rule: AngleRule::DivPowF32,
        axes: Axes::Linear,
    };
    let cos_sin = (0..=max_position)
        .flat_map(|position| {
            (0..pairs).flat_map(move |pair| {
                let (c, s) = if std {
                    std_table
                        .cos_sin([position; 3], pair, pairs)
                        .expect("F32 table pair")
                } else {
                    kernel_core::hf_rope::cos_sin(position, pair, f64::from(theta))
                };
                [c as f32, s as f32]
            })
        })
        .collect::<Vec<_>>();
    let table_pairs = cos_sin
        .chunks_exact(2)
        .map(|x| (x[0] as f64, x[1] as f64))
        .collect::<Vec<_>>();
    let style = if std {
        RotaryStyle::GptJ
    } else {
        RotaryStyle::Neox
    };
    let expanded_q = positions
        .iter()
        .flat_map(|&position| std::iter::repeat_n(position, qh))
        .collect::<Vec<_>>();
    let expanded_k = positions
        .iter()
        .flat_map(|&position| std::iter::repeat_n(position, kvh))
        .collect::<Vec<_>>();
    let mut q_input = Vec::with_capacity(rows * qh * qk);
    let mut k_input = Vec::with_capacity(rows * kvh * qk);
    let mut v_input = Vec::with_capacity(rows * kvh * v);
    for row in input.chunks(stride) {
        q_input.extend_from_slice(&row[..qh * qk]);
        k_input.extend_from_slice(&row[qh * qk..(qh + kvh) * qk]);
        v_input.extend_from_slice(&row[(qh + kvh) * qk..]);
    }
    let rotate = |values: &[u16], positions: &[usize]| {
        let captured = table.then_some(table_pairs.as_slice());
        if !std {
            return hf_rope_f32(
                &as_f64(values),
                positions,
                qk,
                rotary,
                theta as f64,
                captured,
            );
        }
        oracle::rope(
            &as_f64(values),
            positions,
            qk,
            rotary,
            theta as f64,
            style,
            captured,
        )
        .map_err(|e| e.to_string())
    };
    let q_expected = rotate(&q_input, &expanded_q)?;
    let k_expected = rotate(&k_input, &expanded_k)?;
    let multiplier = if std { 1.0_f32 } else { 0.707_f32 };
    let v_expected = as_f64(&v_input)
        .into_iter()
        .map(|x| x * multiplier as f64)
        .collect::<Vec<_>>();
    let page_count = if ring {
        1
    } else {
        max_position / page_size + 1
    };
    let cache_tokens = if ring { 128 } else { page_count * page_size };
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let cache_bytes = cache.bytes()?;
    let mut fixture = Fixture::new(session, rows)?;
    fixture.bf16("qkv", &input)?;
    fixture.poisoned("q", rows * qh * qk * 2)?;
    fixture.poisoned("k_new", rows * kvh * qk * 2)?;
    fixture.poisoned("v_new", rows * kvh * v * 2)?;
    fixture.zero("k_pool", cache_tokens * kvh * qk * cache_bytes)?;
    fixture.zero("v_pool", cache_tokens * kvh * v * cache_bytes)?;
    fixture.i32(
        "block_tables",
        &(0..page_count).map(|x| x as i32).collect::<Vec<_>>(),
    )?;
    fixture.i32("slot_ids", &[0])?;
    fixture.i32("token_batch", &vec![0; rows])?;
    if pos64 {
        fixture.bytes(
            "positions",
            &positions
                .iter()
                .flat_map(|&x| (x as i64).to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
    } else {
        fixture.i32(
            "positions",
            &positions.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        )?;
    }
    if table {
        fixture.f32("cos_sin", &cos_sin)?;
    }
    fixture.scalar("max_pages", page_count as u32);
    fixture.scalar("ring_size", 128);
    fixture.scalar64(
        "k_page_stride_bytes",
        (page_size * kvh * qk * cache_bytes) as u64,
    );
    fixture.scalar64(
        "v_page_stride_bytes",
        (page_size * kvh * v * cache_bytes) as u64,
    );
    fixture.scalar("theta_bits", theta.to_bits());
    fixture.scalar("v_multiplier_bits", multiplier.to_bits());
    fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
    fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
    fixture.launch(module, entry, rows as u32, rows as u32)?;
    let q_ref = oracle::OracleOutput::new(q_expected, vec![2.0; rows * qh * qk], 2)
        .map_err(|e| e.to_string())?;
    let k_ref = oracle::OracleOutput::new(k_expected, vec![2.0; rows * kvh * qk], 2)
        .map_err(|e| e.to_string())?;
    let v_ref = oracle::OracleOutput::new(v_expected, vec![1.0; rows * kvh * v], 1)
        .map_err(|e| e.to_string())?;
    fixture.check_numeric(
        &format!("{name} Q"),
        &as_f32(&fixture.read_bf16("q", rows * qh * qk)?),
        &q_ref,
        DType::Bf16,
    )?;
    fixture.check_numeric(
        &format!("{name} K"),
        &as_f32(&fixture.read_bf16("k_new", rows * kvh * qk)?),
        &k_ref,
        DType::Bf16,
    )?;
    fixture.check_numeric(
        &format!("{name} V"),
        &as_f32(&fixture.read_bf16("v_new", rows * kvh * v)?),
        &v_ref,
        DType::Bf16,
    )?;
    if ring && prefill {
        fixture.check_exact(
            &format!("{name} ring unchanged"),
            &fixture.read_bytes("k_pool", cache_tokens * kvh * qk * cache_bytes)?,
            &vec![0; cache_tokens * kvh * qk * cache_bytes],
        )?;
        // The ring write of the same rank count: its KV heads match.
        let tag = ["_tp2_"]
            .into_iter()
            .find(|tag| name.contains(tag))
            .map_or("_", |tag| tag);
        let write = find(
            entries,
            &format!(
                "ring_write_mimo{tag}{}_pos32",
                if fp8 { "fp8" } else { "bf16" }
            ),
        );
        fixture.bytes("last_positions", &(max_position as i64).to_ne_bytes())?;
        fixture.launch(module, write, rows as u32, rows as u32)?;
        if rows == 8 {
            // The ring write rewrites the ring rows of the chunk's positions.
            let rows_of = |width: usize| {
                cache.byte_ranges(positions.iter().map(|&position| {
                    let row = position % 128 * kvh * width * cache_bytes;
                    row..row + kvh * width * cache_bytes
                }))
            };
            replay(
                session,
                module,
                &mut fixture,
                write,
                rows,
                &[
                    Written::Cache("k_pool", rows_of(qk)?),
                    Written::Cache("v_pool", rows_of(v)?),
                ],
            )?;
        }
    }
    let new_k = fixture.read_bf16("k_new", rows * kvh * qk)?;
    let new_v = fixture.read_bf16("v_new", rows * kvh * v)?;
    if fp8 {
        let mut expected_k = vec![0_u8; cache_tokens * kvh * qk];
        let mut expected_v = vec![0_u8; cache_tokens * kvh * v];
        for (token, &position) in positions.iter().enumerate() {
            if ring && prefill && position + 128 <= max_position {
                continue;
            }
            let target = if ring { position % 128 } else { position };
            for index in 0..kvh * qk {
                let value = if prefill {
                    bf16_to_f32(new_k[token * kvh * qk + index])
                } else {
                    k_ref.values[token * kvh * qk + index] as f32
                };
                expected_k[target * kvh * qk + index] = kernel_core::kv_fp8::encode(value, 1.0);
            }
            for index in 0..kvh * v {
                let value = if prefill {
                    bf16_to_f32(new_v[token * kvh * v + index])
                } else {
                    v_ref.values[token * kvh * v + index] as f32
                };
                expected_v[target * kvh * v + index] = kernel_core::kv_fp8::encode(value, 1.0);
            }
        }
        fixture.check_exact(
            &format!("{name} K cache"),
            &fixture.read_bytes("k_pool", expected_k.len())?,
            &expected_k,
        )?;
        fixture.check_exact(
            &format!("{name} V cache"),
            &fixture.read_bytes("v_pool", expected_v.len())?,
            &expected_v,
        )?;
    } else {
        let mut expected_k = vec![0_u16; cache_tokens * kvh * qk];
        let mut expected_v = vec![0_u16; cache_tokens * kvh * v];
        for (token, &position) in positions.iter().enumerate() {
            if ring && prefill && position + 128 <= max_position {
                continue;
            }
            let target = if ring { position % 128 } else { position };
            expected_k[target * kvh * qk..(target + 1) * kvh * qk]
                .copy_from_slice(&new_k[token * kvh * qk..(token + 1) * kvh * qk]);
            expected_v[target * kvh * v..(target + 1) * kvh * v]
                .copy_from_slice(&new_v[token * kvh * v..(token + 1) * kvh * v]);
        }
        fixture.check_exact(
            &format!("{name} K cache"),
            &fixture.read_bf16("k_pool", expected_k.len())?,
            &expected_k,
        )?;
        fixture.check_exact(
            &format!("{name} V cache"),
            &fixture.read_bf16("v_pool", expected_v.len())?,
            &expected_v,
        )?;
    }
    if rows == 1 {
        // Q, the new K and V rows, and every cache variant except ring
        // prefill writes the step's cache rows (rope_kv.cuh).
        let cache_rows =
            |width: usize| {
                cache.byte_ranges(positions.iter().filter(|_| !(ring && prefill)).map(
                    |&position| {
                        let target = if ring { position % 128 } else { position };
                        let row = target * kvh * width * cache_bytes;
                        row..row + kvh * width * cache_bytes
                    },
                ))
            };
        replay(
            session,
            module,
            &mut fixture,
            entry,
            rows,
            &[
                Written::Output("q", WriteMap::of(DType::Bf16).prefix(qh * qk)?),
                Written::Output("k_new", WriteMap::of(DType::Bf16).prefix(kvh * qk)?),
                Written::Output("v_new", WriteMap::of(DType::Bf16).prefix(kvh * v)?),
                Written::Cache("k_pool", cache_rows(qk)?),
                Written::Cache("v_pool", cache_rows(v)?),
            ],
        )?;
    }
    Ok(())
}

/// MiMo TP4's Q heads and the head widths of its RoPE and ring entries.
const ROPE_Q_HEADS: usize = 16;
const ROPE_QK: usize = 192;
const ROPE_V: usize = 128;
const ROPE_V_MULTIPLIER: f32 = 0.707;

/// A row-engine launch that writes KV cache entries: a RoPE entry over
/// (sequence, position) rows, or a ring write of `k_new` and `v_new`.
struct CacheWrite<'a> {
    entry: &'a RowInstantiation,
    ring: bool,
    ring_write: bool,
    fp8: bool,
    /// The entry reads its angles from the `cos_sin` table rather than
    /// computing them. Both sources are checked against the RoPE oracle.
    table: bool,
    kv_heads: usize,
    theta: f32,
    /// Each row's sequence and position.
    rows: Vec<(usize, usize)>,
    sequences: usize,
    /// Cache entries of one sequence: its pages, or its ring slot.
    per_sequence: usize,
    /// Tokens of one entry: a page or the ring.
    tokens: usize,
}

impl CacheWrite<'_> {
    fn width(&self) -> usize {
        ROPE_Q_HEADS * ROPE_QK + self.kv_heads * (ROPE_QK + ROPE_V)
    }

    fn element(&self) -> usize {
        if self.fp8 { 1 } else { 2 }
    }

    /// Bytes of one K and one V entry.
    fn entries(&self) -> (u64, u64) {
        let row = self.tokens * self.kv_heads * self.element();
        ((row * ROPE_QK) as u64, (row * ROPE_V) as u64)
    }

    /// Launches the entry with its cache entries at `ids` (sequence-major)
    /// in `k` and `v`, whose entries are `strides` bytes apart. Returns the
    /// fixture for its row outputs.
    fn launch<'s>(
        &self,
        session: &'s Session,
        module: &Module,
        ids: &[u64],
        pools: (&DeviceBuffer, &DeviceBuffer),
        strides: (u64, u64),
    ) -> LabResult<Fixture<'s>> {
        let rows = self.rows.len();
        let mut fixture = Fixture::new(session, rows)?;
        if self.ring_write {
            fixture.bf16(
                "k_new",
                &seeded_bf16(rows * self.kv_heads * ROPE_QK, 0x4471),
            )?;
            fixture.bf16("v_new", &seeded_bf16(rows * self.kv_heads * ROPE_V, 0x4472))?;
            let mut last = vec![0_i64; self.sequences];
            for &(sequence, position) in &self.rows {
                last[sequence] = last[sequence].max(position as i64);
            }
            fixture.bytes(
                "last_positions",
                &last
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
        } else {
            fixture.bf16("qkv", &seeded_bf16(rows * self.width(), 0x5511))?;
            if self.table {
                fixture.f32("cos_sin", &self.cos_sin())?;
            }
            fixture.zero("q", rows * ROPE_Q_HEADS * ROPE_QK * 2)?;
            fixture.zero("k_new", rows * self.kv_heads * ROPE_QK * 2)?;
            fixture.zero("v_new", rows * self.kv_heads * ROPE_V * 2)?;
        }
        fixture.buffers.insert("k_pool".into(), pools.0.clone());
        fixture.buffers.insert("v_pool".into(), pools.1.clone());
        let ids = pool_offsets::i32_ids(ids);
        fixture.i32(
            if self.ring {
                "slot_ids"
            } else {
                "block_tables"
            },
            &ids,
        )?;
        fixture.i32(
            "token_batch",
            &self.rows.iter().map(|&(s, _)| s as i32).collect::<Vec<_>>(),
        )?;
        fixture.i32(
            "positions",
            &self.rows.iter().map(|&(_, p)| p as i32).collect::<Vec<_>>(),
        )?;
        fixture.scalar("max_pages", self.per_sequence as u32);
        fixture.scalar("ring_size", 128);
        fixture.scalar64("k_page_stride_bytes", strides.0);
        fixture.scalar64("v_page_stride_bytes", strides.1);
        fixture.scalar("theta_bits", self.theta.to_bits());
        fixture.scalar("v_multiplier_bits", ROPE_V_MULTIPLIER.to_bits());
        fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
        fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
        let bucket = (rows as u32).next_power_of_two();
        fixture.launch(module, self.entry, bucket, rows as u32)?;
        Ok(fixture)
    }

    /// The F32 cos/sin table over every row's position, as `rope_case`
    /// builds it.
    fn cos_sin(&self) -> Vec<f32> {
        let last = self.rows.iter().map(|&(_, p)| p).max().unwrap_or(0);
        (0..=last)
            .flat_map(|position| {
                (0..32).flat_map(move |pair| {
                    let (cos, sin) =
                        kernel_core::hf_rope::cos_sin(position, pair, f64::from(self.theta));
                    [cos as f32, sin as f32]
                })
            })
            .collect()
    }

    /// The row outputs a RoPE entry writes: Q, `k_new` and `v_new`.
    fn outputs(&self, fixture: &Fixture<'_>) -> LabResult<Vec<u8>> {
        if self.ring_write {
            return Ok(Vec::new());
        }
        let rows = self.rows.len();
        let mut all = fixture.read_bytes("q", rows * ROPE_Q_HEADS * ROPE_QK * 2)?;
        all.extend(fixture.read_bytes("k_new", rows * self.kv_heads * ROPE_QK * 2)?);
        all.extend(fixture.read_bytes("v_new", rows * self.kv_heads * ROPE_V * 2)?);
        Ok(all)
    }

    /// The low run as `rope_case` checks it. Every RoPE entry's Q, K and V
    /// meet the RoPE oracle and its cache rows hold its K and V rows (FP8:
    /// the oracle's values encoded); a ring write's cache rows hold its
    /// inputs.
    /// The K and V entries start zeroed, at dense ids `strides` bytes apart.
    fn check_low(
        &self,
        name: &str,
        fixture: &Fixture<'_>,
        pools: (&DeviceBuffer, &DeviceBuffer),
        strides: (u64, u64),
    ) -> LabResult<()> {
        let rows = self.rows.len();
        let kvh = self.kv_heads;
        let (k_rows, v_rows): (Vec<f64>, Vec<f64>) = if self.ring_write {
            (
                as_f64(&seeded_bf16(rows * kvh * ROPE_QK, 0x4471)),
                as_f64(&seeded_bf16(rows * kvh * ROPE_V, 0x4472)),
            )
        } else {
            let input = seeded_bf16(rows * self.width(), 0x5511);
            let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
            for row in input.chunks(self.width()) {
                q.extend_from_slice(&row[..ROPE_Q_HEADS * ROPE_QK]);
                k.extend_from_slice(&row[ROPE_Q_HEADS * ROPE_QK..(ROPE_Q_HEADS + kvh) * ROPE_QK]);
                v.extend_from_slice(&row[(ROPE_Q_HEADS + kvh) * ROPE_QK..]);
            }
            let expand = |heads: usize| -> Vec<usize> {
                self.rows
                    .iter()
                    .flat_map(|&(_, position)| std::iter::repeat_n(position, heads))
                    .collect()
            };
            let table: Vec<(f64, f64)> = self
                .cos_sin()
                .chunks_exact(2)
                .map(|x| (f64::from(x[0]), f64::from(x[1])))
                .collect();
            let rotate = |values: &[u16], positions: &[usize]| {
                hf_rope_f32(
                    &as_f64(values),
                    positions,
                    ROPE_QK,
                    64,
                    f64::from(self.theta),
                    self.table.then_some(table.as_slice()),
                )
            };
            let q_expected = rotate(&q, &expand(ROPE_Q_HEADS))?;
            let k_expected = rotate(&k, &expand(kvh))?;
            let v_expected: Vec<f64> = as_f64(&v)
                .into_iter()
                .map(|x| x * f64::from(ROPE_V_MULTIPLIER))
                .collect();
            for (what, actual, expected, bound, terms) in [
                (
                    "Q",
                    fixture.read_bf16("q", q_expected.len())?,
                    &q_expected,
                    2.0,
                    2,
                ),
                (
                    "K",
                    fixture.read_bf16("k_new", k_expected.len())?,
                    &k_expected,
                    2.0,
                    2,
                ),
                (
                    "V",
                    fixture.read_bf16("v_new", v_expected.len())?,
                    &v_expected,
                    1.0,
                    1,
                ),
            ] {
                let reference =
                    oracle::OracleOutput::new(expected.clone(), vec![bound; expected.len()], terms)
                        .map_err(|e| e.to_string())?;
                fixture.check_numeric(
                    &format!("{name} low ids {what}"),
                    &as_f32(&actual),
                    &reference,
                    DType::Bf16,
                )?;
            }
            // BF16 caches hold the rounded rows, FP8 caches the oracle's
            // values encoded, as `rope_case` compares them.
            if self.fp8 {
                (k_expected, v_expected)
            } else {
                (
                    as_f64(&fixture.read_bf16("k_new", rows * kvh * ROPE_QK)?),
                    as_f64(&fixture.read_bf16("v_new", rows * kvh * ROPE_V)?),
                )
            }
        };
        let mut last = vec![0; self.sequences];
        for &(sequence, position) in &self.rows {
            last[sequence] = last[sequence].max(position);
        }
        for (pool, stride, dim, values) in [
            (pools.0, strides.0, ROPE_QK, &k_rows),
            (pools.1, strides.1, ROPE_V, &v_rows),
        ] {
            let row_bytes = kvh * dim * self.element();
            for (index, &(sequence, position)) in self.rows.iter().enumerate() {
                if self.ring_write && position + 128 <= last[sequence] {
                    continue;
                }
                let (entry, token) = if self.ring {
                    (sequence, position % self.tokens)
                } else {
                    (
                        sequence * self.per_sequence + position / self.tokens,
                        position % self.tokens,
                    )
                };
                let row = &values[index * kvh * dim..(index + 1) * kvh * dim];
                let expected: Vec<u8> = if self.fp8 {
                    row.iter()
                        .map(|&x| kernel_core::kv_fp8::encode(x as f32, 1.0))
                        .collect()
                } else {
                    row.iter()
                        .flat_map(|&x| f32_to_bf16(x as f32).to_ne_bytes())
                        .collect()
                };
                pool_offsets::same(
                    name,
                    &format!("low-id cache row of row {index}"),
                    &expected,
                    &pool_offsets::read(
                        fixture.session,
                        pool,
                        entry as u64 * stride + (token * row_bytes) as u64,
                        row_bytes as u64,
                    )?,
                )?;
            }
        }
        Ok(())
    }
}

/// `case` at dense low ids, checked against the oracle, then with its
/// entries at high ids in sentinel pools: the row outputs and every entry
/// must match bit for bit and no other pool byte may change.
fn cache_write_high_ids(
    session: &Session,
    module: &Module,
    name: &str,
    case: &CacheWrite<'_>,
) -> LabResult<()> {
    let n = case.sequences * case.per_sequence;
    let (k_entry, v_entry) = case.entries();
    let element = case.element() as u64;
    // Paged K and V share the TP4 serving page of nine GA layers; a ring
    // slot is its own stride.
    let strides = if case.ring {
        (k_entry, v_entry)
    } else {
        let page = 9 * (k_entry + v_entry);
        (page, page)
    };
    let (high, groups) = if case.ring {
        let k = pool_offsets::high_ids(strides.0, element, n);
        let v = pool_offsets::high_ids(strides.1, element, n);
        let mut high = Vec::new();
        for i in 0..n {
            for id in [k.a[i], k.b[i], v.a[i], v.b[i]] {
                if !high.contains(&id) {
                    high.push(id);
                }
            }
        }
        high.truncate(n);
        (high, json!({"k": k.describe(), "v": v.describe()}))
    } else {
        let ids = pool_offsets::high_ids(strides.0, element, n.div_ceil(2));
        (ids.alternating(n), json!({"pages": ids.describe()}))
    };
    let low_ids: Vec<u64> = (0..n as u64).collect();
    let low_k = session.allocate(n * strides.0 as usize)?;
    let low_v = session.allocate(n * strides.1 as usize)?;
    for buffer in [&low_k, &low_v] {
        buffer
            .zero_at_async(0, buffer.byte_len(), &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let entries = high.iter().copied().max().unwrap_or(0) + 1;
    let mut high_k = SentinelPool::new(session, entries * strides.0)?;
    let mut high_v = SentinelPool::new(session, entries * strides.1)?;
    for (&from, &to) in low_ids.iter().zip(&high) {
        high_k.copy_from(session, to * strides.0, &low_k, from * strides.0, k_entry)?;
        high_v.copy_from(session, to * strides.1, &low_v, from * strides.1, v_entry)?;
    }
    let low = case.launch(session, module, &low_ids, (&low_k, &low_v), strides)?;
    case.check_low(name, &low, (&low_k, &low_v), strides)?;
    let high_run = case.launch(
        session,
        module,
        &high,
        (&high_k.buffer, &high_v.buffer),
        strides,
    )?;
    pool_offsets::same(
        name,
        "row outputs",
        &case.outputs(&low)?,
        &case.outputs(&high_run)?,
    )?;
    for (&from, &to) in low_ids.iter().zip(&high) {
        pool_offsets::same(
            name,
            &format!("K entry {to}"),
            &pool_offsets::read(session, &low_k, from * strides.0, k_entry)?,
            &high_k.read(session, to * strides.0, k_entry)?,
        )?;
        pool_offsets::same(
            name,
            &format!("V entry {to}"),
            &pool_offsets::read(session, &low_v, from * strides.1, v_entry)?,
            &high_v.read(session, to * strides.1, v_entry)?,
        )?;
    }
    let scanned = high_k.check_untouched(session, name)? + high_v.check_untouched(session, name)?;
    let highest = high.iter().copied().max().unwrap_or(0);
    pool_offsets::receipt(
        session,
        "row_rope",
        name,
        json!({
            "rows": case.rows.len(),
            "sequences": case.sequences,
            "groups": groups,
            "highest_k_offset": highest * strides.0,
            "highest_v_offset": highest * strides.1,
            "outputs_bitwise": true,
            "entries_bitwise": true,
            "sentinel_bytes_checked": scanned,
        }),
    )
}

/// RoPE KV writes and ring writes past the 32-bit lines (`pool_offsets`):
/// the paged GA entries over pages alternating between groups A and B, the
/// SWA ring's decode entries and ring writes with one sequence in each
/// group of the K and of the V slot stride, and the single-sequence ring
/// tail at each of those slots. Each low run is checked against the RoPE
/// oracle; the high run's Q, `k_new`, `v_new` and cache entries must match
/// it bit for bit.
#[test]
fn row_rope_high_ids_match_low_ids() {
    gpu_test(pool_offsets::need_bytes(22 << 30), |session| {
        let module = module(session)?;
        let entries = catalog();
        let rope = |name: &str, ring: bool, fp8: bool| CacheWrite {
            entry: find(&entries, name),
            ring,
            ring_write: false,
            fp8,
            table: name.contains("_table_"),
            kv_heads: if ring { 2 } else { 1 },
            theta: if ring { 1e4 } else { 1e7 },
            rows: Vec::new(),
            sequences: 4,
            per_sequence: if ring { 1 } else { 5 },
            tokens: if ring { 128 } else { 64 },
        };
        // Paged: four sequences of 96 rows from position 64 s + 17, over
        // their first five pages. Both table and computed low runs must
        // meet the oracle before checking the high-address writes.
        for (name, fp8) in [
            ("rope_mimo_ga_bf16_table_p64_pos32", false),
            ("rope_mimo_ga_fp8_table_p64_pos32", true),
            ("rope_mimo_ga_bf16_computed_p64_pos32", false),
            ("rope_mimo_ga_fp8_computed_p64_pos32", true),
        ] {
            let mut case = rope(name, false, fp8);
            case.rows = (0..4)
                .flat_map(|s| (0..96).map(move |i| (s, 64 * s + 17 + i)))
                .collect();
            cache_write_high_ids(session, &module, name, &case)?;
        }
        // Ring decode: one row per sequence.
        for (name, fp8) in [
            ("rope_mimo_swa_decode_table_p32_pos32", false),
            ("rope_mimo_swa_decode_fp8_table_p32_pos32", true),
            ("rope_mimo_swa_decode_computed_p32_pos32", false),
            ("rope_mimo_swa_decode_fp8_computed_p32_pos32", true),
        ] {
            let mut case = rope(name, true, fp8);
            case.rows = (0..4).map(|s| (s, 1000 + 37 * s)).collect();
            cache_write_high_ids(session, &module, name, &case)?;
        }
        // Ring writes: 160 prefill rows per sequence keep their last 128.
        for (name, fp8) in [
            ("ring_write_mimo_bf16_pos32", false),
            ("ring_write_mimo_fp8_pos32", true),
        ] {
            let mut case = rope(name, true, fp8);
            case.ring_write = true;
            case.rows = (0..4).flat_map(|s| (0..160).map(move |p| (s, p))).collect();
            cache_write_high_ids(session, &module, name, &case)?;
        }
        // The single-sequence tail of 300 rows at each ring id.
        let name = "ring_write_mimo_bf16_single_sequence_pos32";
        let mut tail = rope(name, true, false);
        tail.ring_write = true;
        tail.sequences = 1;
        tail.rows = (0..300).map(|p| (0, p)).collect();
        let (k_slot, v_slot) = tail.entries();
        let (k, v) = (
            pool_offsets::high_ids(k_slot, 2, 1),
            pool_offsets::high_ids(v_slot, 2, 1),
        );
        for slot in [k.a[0], k.b[0], v.a[0], v.b[0]] {
            single_slot_tail(session, &module, name, &tail, slot)?;
        }
        Ok(())
    })
    .expect("RoPE KV and ring writes at high page and slot ids");
}

/// The single-sequence ring tail with its slot at `slot` against slot 0.
fn single_slot_tail(
    session: &Session,
    module: &Module,
    name: &str,
    case: &CacheWrite<'_>,
    slot: u64,
) -> LabResult<()> {
    let (k_slot, v_slot) = case.entries();
    let low_k = session.allocate(k_slot as usize)?;
    let low_v = session.allocate(v_slot as usize)?;
    for buffer in [&low_k, &low_v] {
        buffer
            .zero_at_async(0, buffer.byte_len(), &session.stream)
            .map_err(|e| e.to_string())?;
    }
    let mut high_k = SentinelPool::new(session, (slot + 1) * k_slot)?;
    let mut high_v = SentinelPool::new(session, (slot + 1) * v_slot)?;
    high_k.copy_from(session, slot * k_slot, &low_k, 0, k_slot)?;
    high_v.copy_from(session, slot * v_slot, &low_v, 0, v_slot)?;
    let label = format!("{name} slot {slot}");
    let low = case.launch(session, module, &[0], (&low_k, &low_v), (k_slot, v_slot))?;
    case.check_low(&label, &low, (&low_k, &low_v), (k_slot, v_slot))?;
    case.launch(
        session,
        module,
        &[slot],
        (&high_k.buffer, &high_v.buffer),
        (k_slot, v_slot),
    )?;
    pool_offsets::same(
        &label,
        "K slot",
        &pool_offsets::read(session, &low_k, 0, k_slot)?,
        &high_k.read(session, slot * k_slot, k_slot)?,
    )?;
    pool_offsets::same(
        &label,
        "V slot",
        &pool_offsets::read(session, &low_v, 0, v_slot)?,
        &high_v.read(session, slot * v_slot, v_slot)?,
    )?;
    let scanned =
        high_k.check_untouched(session, &label)? + high_v.check_untouched(session, &label)?;
    pool_offsets::receipt(
        session,
        "row_rope",
        &label,
        json!({
            "rows": case.rows.len(),
            "slot": slot,
            "k_offset": slot * k_slot,
            "v_offset": slot * v_slot,
            "entries_bitwise": true,
            "sentinel_bytes_checked": scanned,
        }),
    )
}

fn fused_norm_quant_cases(
    session: &Session,
    module: &Module,
    entries: &[RowInstantiation],
    rows: usize,
) -> LabResult<()> {
    for name in [
        "rms_mimo_hf4096_fp8_exact",
        "rms_mimo_fused_fp8_exact",
        "rms_mimo_fused_fp8_pow2",
        "rms_mimo_fused_mxfp8",
        "rms_mimo_fused_nvfp4",
    ] {
        let entry = find(entries, name);
        let width = 4096;
        let x = seeded_bf16(rows * width, 0x9801);
        let residual = seeded_bf16(rows * width, 0x9802);
        let weight = seeded_bf16(width, 0x9803);
        let added = x
            .iter()
            .zip(&residual)
            .map(|(&a, &b)| f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b)))
            .collect::<Vec<_>>();
        let (format, fused_residual) = match entry.query {
            RowQuery::RmsNorm {
                output_quantization: Some(format),
                fused_residual,
                ..
            } => (format, fused_residual),
            _ => unreachable!(),
        };
        let y = NORM
            .apply_bf16(if fused_residual { &added } else { &x }, &weight, width)
            .map_err(|e| e.to_string())?;
        let reference = quant_reference(&y, rows, width, format)?;
        let mut fixture = Fixture::new(session, rows)?;
        fixture.bf16("x", &x)?;
        fixture.bf16("residual", &residual)?;
        fixture.bf16("weight", &weight)?;
        fixture.poisoned("residual_out", rows * width * 2)?;
        fixture.poisoned("y", rows * width * 2)?;
        poisoned_codes(&mut fixture, &reference, format)?;
        fixture.f32("global_scale", &[reference.global_scale])?;
        fixture.launch(module, entry, rows as u32, rows as u32)?;
        if fused_residual {
            fixture.check_exact(
                &format!("{name} residual"),
                &fixture.read_bf16("residual_out", rows * width)?,
                &added,
            )?;
        }
        fixture.check_exact(
            &format!("{name} y"),
            &fixture.read_bf16("y", rows * width)?,
            &y,
        )?;
        check_quant(&fixture, name, &reference)?;
        if rows == 1 {
            let mut written = quant_written(&reference, format)?;
            written.push(Written::Output(
                "y",
                WriteMap::of(DType::Bf16).prefix(rows * width)?,
            ));
            if fused_residual {
                written.push(Written::Output(
                    "residual_out",
                    WriteMap::of(DType::Bf16).prefix(rows * width)?,
                ));
            }
            replay(session, module, &mut fixture, entry, rows, &written)?;
        }
    }
    Ok(())
}

#[test]
fn row_norm_and_quant_gpu() {
    gpu_test(512 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        for rows in [1, 8, 32, 512, 2048] {
            norm_cases(session, &module, &entries, rows)?;
            quant_cases(session, &module, &entries, rows)?;
            if rows <= 32 {
                fused_norm_quant_cases(session, &module, &entries, rows)?;
            }
        }
        Ok(())
    })
    .expect("row norms and quantizers");
}

#[test]
fn row_router_permute_combine_gpu() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        for rows in [1, 8, 32, 512, 2048] {
            router_cases(session, &module, &entries, rows)?;
            permute_cases(session, &module, &entries, rows)?;
            simple_row_cases(session, &module, &entries, rows)?;
        }
        Ok(())
    })
    .expect("row router, permutation, combine, embedding and SwiGLU");
}

#[test]
fn row_prefill_tiled_permute_gpu() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let count = find(&entries, "permute_tile_count_mimo_b64");
        let prefix = find(&entries, "permute_tile_prefix_mimo_b64");
        let scan = find(&entries, "permute_tile_scan_mimo_b64");
        let scatter = find(&entries, "permute_tile_scatter_mimo_b64");
        for rows in [2048, 8192] {
            for skewed in [false, true] {
                let ids = (0..rows * 8)
                    .map(|route| {
                        if skewed {
                            7
                        } else {
                            ((route * 53 + route / 8) % 256) as i32
                        }
                    })
                    .collect::<Vec<_>>();
                let expected = oracle::expert_permute(
                    &ids.iter().map(|&id| id as usize).collect::<Vec<_>>(),
                    256,
                    64,
                )
                .map_err(|e| e.to_string())?;
                let mut fixture = Fixture::new(session, rows)?;
                fixture.i32("ids", &ids)?;
                fixture.zero("tile_counts", 256 * 256 * 4)?;
                fixture.zero("tile_offsets", 256 * 256 * 4)?;
                fixture.zero("counts", 256 * 4)?;
                fixture.zero("offsets", 257 * 4)?;
                fixture.zero("padded_rows", 4)?;
                fixture.zero("sorted_rows", (rows * 8 + 256 * 63) * 4)?;
                for entry in [count, prefix, scan, scatter] {
                    fixture.launch(&module, entry, rows as u32, rows as u32)?;
                }
                let label = format!("tiled prefill rows={rows} skewed={skewed}");
                fixture.check_exact(
                    &format!("{label} offsets"),
                    &fixture.read_i32("offsets", 257)?,
                    &expected
                        .expert_offsets
                        .iter()
                        .map(|&value| value as i32)
                        .collect::<Vec<_>>(),
                )?;
                fixture.check_exact(
                    &format!("{label} sorted rows"),
                    &fixture.read_i32("sorted_rows", expected.forward.len())?,
                    &expected
                        .forward
                        .iter()
                        .map(|route| route.map_or(-1, |value| value as i32))
                        .collect::<Vec<_>>(),
                )?;
            }
        }
        Ok(())
    })
    .expect("prefill tile routing is stable and bit-exact");
}

#[test]
fn row_prefill_single_sequence_ring_tail_gpu() {
    gpu_test(256 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let full = find(&entries, "ring_write_mimo_bf16_pos32");
        let tail = find(&entries, "ring_write_mimo_bf16_single_sequence_pos32");
        for rows in [2048, 8192] {
            let mut fixture = Fixture::new(session, rows)?;
            fixture.bf16("k_new", &seeded_bf16(rows * 2 * 192, 0x4471))?;
            fixture.bf16("v_new", &seeded_bf16(rows * 2 * 128, 0x4472))?;
            fixture.zero("k_pool", 128 * 2 * 192 * 2)?;
            fixture.zero("v_pool", 128 * 2 * 128 * 2)?;
            fixture.i32("slot_ids", &[0])?;
            fixture.i32("token_batch", &vec![0; rows])?;
            fixture.i32("positions", &(0..rows as i32).collect::<Vec<_>>())?;
            fixture.bytes("last_positions", &((rows - 1) as i64).to_ne_bytes())?;
            fixture.scalar("ring_size", 128);
            fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
            fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
            fixture.launch(&module, full, rows as u32, rows as u32)?;
            let expected_k = fixture.read_bytes("k_pool", 128 * 2 * 192 * 2)?;
            let expected_v = fixture.read_bytes("v_pool", 128 * 2 * 128 * 2)?;
            fixture.zero("k_pool", expected_k.len())?;
            fixture.zero("v_pool", expected_v.len())?;
            fixture.launch(&module, tail, rows as u32, rows as u32)?;
            fixture.check_exact(
                &format!("ring tail K rows={rows}"),
                &fixture.read_bytes("k_pool", expected_k.len())?,
                &expected_k,
            )?;
            fixture.check_exact(
                &format!("ring tail V rows={rows}"),
                &fixture.read_bytes("v_pool", expected_v.len())?,
                &expected_v,
            )?;
        }
        Ok(())
    })
    .expect("single-sequence ring tail matches full ring write");
}

/// Fills a fixture for one of the row entries eager MiMo prefill launches
/// over `rows` live rows spread across three items, with the argument names
/// the bundle binds (null residuals, tables and cos/sin where it passes
/// null).
fn prefill_row_fixture<'a>(
    session: &'a Session,
    name: &str,
    rows: usize,
) -> LabResult<Fixture<'a>> {
    let mut fixture = Fixture::new(session, rows)?;
    let items = rows.min(3);
    let batch = (0..rows)
        .map(|row| (row * items / rows) as i32)
        .collect::<Vec<_>>();
    // Each item continues a prefix: item i starts at 700 * i + 5.
    let mut positions = vec![0_i32; rows];
    let mut last = vec![0_i64; items];
    for row in 0..rows {
        let item = batch[row] as usize;
        let first = batch
            .iter()
            .position(|&b| b as usize == item)
            .expect("item row");
        positions[row] = (700 * item + 5 + row - first) as i32;
        last[item] = i64::from(positions[row]);
    }
    let seed = rows as u64;
    if name == "embedding_full" {
        let vocab = 97_usize;
        fixture.bf16("table", &seeded_bf16(vocab * 4096, 0x9101))?;
        fixture.i32(
            "ids",
            &(0..rows)
                .map(|row| ((row * 31) % vocab) as i32)
                .collect::<Vec<_>>(),
        )?;
        fixture.zero("output", rows * 4096 * 2)?;
        fixture.scalar("vocab_offset", 0);
        fixture.scalar("shard_vocab", vocab as u32);
    } else if name == "rms_mimo_hf4096" {
        fixture.bf16("x", &seeded_bf16(rows * 4096, 0x9102 + seed))?;
        fixture.bf16("weight", &seeded_bf16(4096, 0x9103))?;
        fixture.zero("y", rows * 4096 * 2)?;
    } else if name == "router_mimo_sigmoid256" {
        fixture.f32("logits", &seeded_f32(rows * 256, 0x9104 + seed))?;
        fixture.f32(
            "bias",
            &seeded_f32(256, 0x9105)
                .into_iter()
                .map(|x| x * 0.1)
                .collect::<Vec<_>>(),
        )?;
        fixture.zero("ids", rows * 8 * 4)?;
        fixture.zero("weights", rows * 8 * 4)?;
    } else if name == "combine_mimo4096" {
        fixture.bf16("routed", &seeded_bf16(rows * 8 * 4096, 0x9106 + seed))?;
        fixture.f32(
            "weights",
            &(0..rows * 8)
                .map(|i| 0.08 + (i % 8) as f32 * 0.01)
                .collect::<Vec<_>>(),
        )?;
        fixture.zero("output", rows * 4096 * 2)?;
    } else if name.starts_with("rope_mimo") || name.starts_with("ring_write_mimo") {
        let ring = name.contains("swa") || name.starts_with("ring_write");
        let fp8 = name.contains("fp8");
        let (qh, kvh) = if name.contains("_tp2_") {
            (32_usize, if ring { 4 } else { 2 })
        } else {
            (16, if ring { 2 } else { 1 })
        };
        let (qk, v) = (192_usize, 128_usize);
        let cache_bytes = if fp8 { 1 } else { 2 };
        let page = 64_usize;
        let max_pages = (700 * items + rows) / page + 1;
        let cache_tokens = if ring {
            128 * items
        } else {
            items * max_pages * page
        };
        fixture.zero("k_pool", cache_tokens * kvh * qk * cache_bytes)?;
        fixture.zero("v_pool", cache_tokens * kvh * v * cache_bytes)?;
        fixture.i32("token_batch", &batch)?;
        fixture.i32("positions", &positions)?;
        fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
        fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
        fixture.scalar("ring_size", 128);
        if ring {
            // Slots in another order than items.
            fixture.i32(
                "slot_ids",
                &(0..items)
                    .map(|item| ((item + items - 1) % items) as i32)
                    .collect::<Vec<_>>(),
            )?;
        }
        if name.starts_with("ring_write") {
            fixture.bf16("k_new", &seeded_bf16(rows * kvh * qk, 0x9107 + seed))?;
            fixture.bf16("v_new", &seeded_bf16(rows * kvh * v, 0x9108 + seed))?;
            fixture.bytes(
                "last_positions",
                &last
                    .iter()
                    .flat_map(|p| p.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
        } else {
            let stride = qh * qk + kvh * (qk + v);
            fixture.bf16("qkv", &seeded_bf16(rows * stride, 0x9109 + seed))?;
            fixture.zero("q", rows * qh * qk * 2)?;
            fixture.zero("k_new", rows * kvh * qk * 2)?;
            fixture.zero("v_new", rows * kvh * v * 2)?;
            if !ring {
                fixture.i32(
                    "block_tables",
                    &(0..items * max_pages)
                        .map(|page| page as i32)
                        .collect::<Vec<_>>(),
                )?;
            }
            fixture.scalar("max_pages", max_pages as u32);
            fixture.scalar64(
                "k_page_stride_bytes",
                (page * kvh * qk * cache_bytes) as u64,
            );
            fixture.scalar64("v_page_stride_bytes", (page * kvh * v * cache_bytes) as u64);
            fixture.scalar("theta_bits", if ring { 1e4_f32 } else { 1e7_f32 }.to_bits());
            fixture.scalar("v_multiplier_bits", 0.707_f32.to_bits());
        }
    } else {
        return Err(format!("no prefill grid fixture for {name}"));
    }
    Ok(fixture)
}

fn snapshot(fixture: &Fixture<'_>) -> LabResult<BTreeMap<String, Vec<u8>>> {
    fixture
        .buffers
        .iter()
        .map(|(name, buffer)| Ok((name.clone(), fixture.read_bytes(name, buffer.byte_len())?)))
        .collect()
}

/// Eager MiMo prefill launches its row kernels at the smallest row bucket
/// holding the live rows instead of the 8,192-row capacity
/// (`Phase::row_grid`). Every buffer, inputs included, must match the
/// capacity launch byte for byte. The sizes include odd counts, bucket
/// edges and the two halves pipelined prefill gives 1,908 (960 + 948) and
/// 3,743 (1,920 + 1,823) tokens.
#[test]
fn row_prefill_live_bucket_grids_gpu() {
    gpu_test(1 << 30, |session| {
        let module = module(session)?;
        let entries = catalog();
        let mut names = vec![
            "embedding_full".to_string(),
            "rms_mimo_hf4096".into(),
            "router_mimo_sigmoid256".into(),
            "combine_mimo4096".into(),
        ];
        for tag in ["", "_tp2"] {
            names.push(format!("rope_mimo{tag}_swa_prefill_computed_p32_pos32"));
            names.push(format!("rope_mimo{tag}_ga_bf16_computed_p64_pos32"));
            names.push(format!("rope_mimo{tag}_ga_fp8_computed_p64_pos32"));
            names.push(format!("ring_write_mimo{tag}_bf16_pos32"));
        }
        let mut checked = 0;
        for rows in [
            1_usize, 7, 50, 63, 64, 65, 511, 691, 948, 960, 1023, 1025, 1823, 1908, 1920, 3743,
        ] {
            let bucket = kernels_sm120::row::smallest_bucket(rows as u32).ok_or("row bucket")?;
            for name in &names {
                if !entries
                    .iter()
                    .any(|entry| entry.symbol == format!("chop_row_{name}_v1"))
                    || !case_enabled(name, rows)
                {
                    continue;
                }
                let entry = find(&entries, name);
                let mut outputs = Vec::new();
                for capacity in [8192, bucket] {
                    let fixture = prefill_row_fixture(session, name, rows)?;
                    fixture.launch(&module, entry, capacity, rows as u32)?;
                    outputs.push((fixture, capacity));
                }
                let (full, _) = &outputs[0];
                let (live, _) = &outputs[1];
                let (expected, actual) = (snapshot(full)?, snapshot(live)?);
                for (buffer, bytes) in &expected {
                    live.check_exact(
                        &format!("{name} rows={rows} grid={bucket} {buffer}"),
                        &actual[buffer],
                        bytes,
                    )?;
                }
                checked += 1;
            }
        }
        if checked == 0 {
            return Err("no prefill row grid case ran".into());
        }
        eprintln!("prefill row grids: {checked} entry and size cases bitwise equal");
        Ok(())
    })
    .expect("prefill row kernels at the live bucket grid match the capacity grid");
}

#[test]
fn row_prefill_fused_fp8_graph_gpu() {
    gpu_test(1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let entry = find(&entries, "rms_mimo_hf4096_fp8_exact");
        for rows in [2048, 8192] {
            let mut fixture = Fixture::new(session, rows)?;
            fixture.bf16("x", &seeded_bf16(rows * 4096, 0x9511))?;
            fixture.bf16("weight", &seeded_bf16(4096, 0x9512))?;
            fixture.poisoned("y", rows * 4096 * 2)?;
            fixture.poisoned("codes", rows * 4096)?;
            fixture.poisoned("scales", rows * 32 * 4)?;
            fixture.f32("global_scale", &[1.0])?;
            let mut checks = Replay::new(session)
                .output(
                    "y",
                    &fixture.buffers["y"],
                    WriteMap::of(DType::Bf16).prefix(rows * 4096)?,
                )
                .output(
                    "codes",
                    &fixture.buffers["codes"],
                    WriteMap::of(DType::U8).prefix(rows * 4096)?,
                )
                .output(
                    "scales",
                    &fixture.buffers["scales"],
                    WriteMap::of(DType::F32).prefix(rows * 32)?,
                );
            checks.eager(|| fixture.launch(&module, entry, rows as u32, rows as u32))?;
            let eager_y = fixture.read_bytes("y", rows * 4096 * 2)?;
            let eager_codes = fixture.read_bytes("codes", rows * 4096)?;
            let eager_scales = fixture.read_bytes("scales", rows * 32 * 4)?;
            let graph =
                session.capture(|_| fixture.launch(&module, entry, rows as u32, rows as u32))?;
            checks.graph(&graph)?;
            fixture.check_exact(
                &format!("FP8 prefill norm y rows={rows}"),
                &fixture.read_bytes("y", eager_y.len())?,
                &eager_y,
            )?;
            fixture.check_exact(
                &format!("FP8 prefill norm codes rows={rows}"),
                &fixture.read_bytes("codes", eager_codes.len())?,
                &eager_codes,
            )?;
            fixture.check_exact(
                &format!("FP8 prefill norm scales rows={rows}"),
                &fixture.read_bytes("scales", eager_scales.len())?,
                &eager_scales,
            )?;
            let timing = time(session, 1, 5, |stream| {
                graph.launch(stream).map_err(|e| e.to_string())
            })?;
            let mut receipt = BTreeMap::new();
            receipt.insert("symbol".into(), json!(entry.symbol));
            receipt.insert("tokens".into(), json!(rows));
            receipt.insert("graph_median_us".into(), json!(timing.median_ms * 1000.0));
            receipt.insert(
                "graph_bitwise_eager".into(),
                json!(checks.evidence().has(poison::Phase::Graph)),
            );
            checks.record(&mut receipt);
            receipt.insert(
                "timing_environment".into(),
                json!("shared-device, L2 flushed"),
            );
            session.receipt("row_prefill_fp8", receipt)?;
        }
        Ok(())
    })
    .expect("prefill FP8 norm graph matches eager");
}

#[test]
fn row_rope_paged_ring_gpu() {
    gpu_test(512 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        for rows in [1, 8, 32, 512, 2048] {
            rope_case(
                session,
                &module,
                &entries,
                rows,
                "rope_mimo_ga_bf16_table_p32_pos32",
            )?;
            rope_case(
                session,
                &module,
                &entries,
                rows,
                "rope_std_gptj_table_p32_pos32",
            )?;
            rope_case(
                session,
                &module,
                &entries,
                rows,
                if rows > 32 {
                    "rope_mimo_swa_prefill_table_p32_pos32"
                } else {
                    "rope_mimo_swa_decode_table_p32_pos32"
                },
            )?;
        }
        rope_case(
            session,
            &module,
            &entries,
            1,
            "rope_mimo_ga_fp8_table_p16_pos64",
        )?;
        rope_case(
            session,
            &module,
            &entries,
            8,
            "rope_mimo_ga_fp8_table_p32_pos32",
        )?;
        rope_case(
            session,
            &module,
            &entries,
            1,
            "rope_mimo_ga_bf16_table_p64_pos64",
        )?;
        rope_case(
            session,
            &module,
            &entries,
            1,
            "rope_mimo_swa_decode_fp8_computed_p32_pos32",
        )?;
        rope_case(
            session,
            &module,
            &entries,
            8,
            "rope_mimo_swa_prefill_fp8_table_p32_pos32",
        )?;
        // TP2 per-rank heads.
        {
            let tag = "tp2";
            for rows in [1, 8, 512] {
                rope_case(
                    session,
                    &module,
                    &entries,
                    rows,
                    &format!("rope_mimo_{tag}_ga_bf16_table_p64_pos32"),
                )?;
                rope_case(
                    session,
                    &module,
                    &entries,
                    rows,
                    &if rows > 32 {
                        format!("rope_mimo_{tag}_swa_prefill_table_p32_pos32")
                    } else {
                        format!("rope_mimo_{tag}_swa_decode_table_p32_pos32")
                    },
                )?;
            }
            // Computed-angle comparison entries, including FP8 global pool
            // and FP8 ring writes. Serving selects table-backed entries.
            for (rows, name) in [
                (1, "ga_bf16_computed_p64_pos32"),
                (8, "ga_bf16_computed_p64_pos32"),
                (1, "ga_fp8_computed_p64_pos32"),
                (8, "ga_fp8_computed_p64_pos32"),
                (1, "swa_decode_computed_p32_pos32"),
                (8, "swa_decode_computed_p32_pos32"),
                (1, "swa_decode_fp8_computed_p32_pos32"),
                (8, "swa_prefill_computed_p32_pos32"),
                (8, "swa_prefill_fp8_table_p32_pos32"),
            ] {
                rope_case(
                    session,
                    &module,
                    &entries,
                    rows,
                    &format!("rope_mimo_{tag}_{name}"),
                )?;
            }
        }
        Ok(())
    })
    .expect("row RoPE and KV writes");
}

/// Compare the MiMo computed-angle global entry and its table twin at position
/// 511 against the HF F32-angle oracle used by `rope_case`. Both use pinned
/// HF inverse frequencies, an F32 angle product and accurate trig rounded to
/// BF16. Serving selects the table-backed entry.
#[test]
fn row_rope_computed_angle_position_511_gpu() {
    gpu_test(64 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let (qh, qk, v, rotary, theta) = (16_usize, 192_usize, 128_usize, 64_usize, 1e7_f32);
        let position = 511_usize;
        let pages = position / 64 + 1;
        let input = seeded_bf16(qh * qk + qk + v, 0x9a0 + position as u64);
        let exact = (0..=position)
            .flat_map(|position| {
                (0..rotary / 2).flat_map(move |pair| {
                    let (cos, sin) =
                        kernel_core::hf_rope::cos_sin(position, pair, f64::from(theta));
                    [cos as f32, sin as f32]
                })
            })
            .collect::<Vec<_>>();
        let expected = hf_rope_f32(
            &as_f64(&input[..qh * qk]),
            &[position; 16],
            qk,
            rotary,
            f64::from(theta),
            None,
        )?;
        let reference = oracle::OracleOutput::new(expected, vec![2.0; qh * qk], 2)
            .map_err(|e| e.to_string())?;
        let mut results = Vec::new();
        for name in [
            "rope_mimo_ga_bf16_table_p64_pos32",
            "rope_mimo_ga_bf16_computed_p64_pos32",
        ] {
            let mut fixture = Fixture::new(session, 1)?;
            fixture.bf16("qkv", &input)?;
            fixture.poisoned("q", qh * qk * 2)?;
            fixture.poisoned("k_new", qk * 2)?;
            fixture.poisoned("v_new", v * 2)?;
            fixture.zero("k_pool", pages * 64 * qk * 2)?;
            fixture.zero("v_pool", pages * 64 * v * 2)?;
            fixture.i32("block_tables", &(0..pages as i32).collect::<Vec<_>>())?;
            fixture.i32("slot_ids", &[0])?;
            fixture.i32("token_batch", &[0])?;
            fixture.i32("positions", &[position as i32])?;
            fixture.f32("cos_sin", &exact)?;
            fixture.scalar("max_pages", pages as u32);
            fixture.scalar("ring_size", 128);
            fixture.scalar64("k_page_stride_bytes", (64 * qk * 2) as u64);
            fixture.scalar64("v_page_stride_bytes", (64 * v * 2) as u64);
            fixture.scalar("theta_bits", theta.to_bits());
            fixture.scalar("v_multiplier_bits", 0.707_f32.to_bits());
            fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
            fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
            fixture.launch(&module, find(&entries, name), 1, 1)?;
            results.push(fixture.check_numeric(
                &format!("{name} Q at position {position}"),
                &as_f32(&fixture.read_bf16("q", qh * qk)?),
                &reference,
                DType::Bf16,
            ));
        }
        let failures = results
            .into_iter()
            .filter_map(Result::err)
            .collect::<Vec<_>>();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("computed-angle RoPE meets the oracle at position 511");
}

#[test]
fn row_rope_shared_paged_pool_gpu() {
    gpu_test(64 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        for fp8 in [false, true] {
            for rows in [1_usize, 2] {
                let name = if fp8 {
                    "rope_mimo_ga_fp8_table_p16_pos64"
                } else {
                    "rope_mimo_ga_bf16_table_p16_pos64"
                };
                let entry = find(&entries, name);
                let qk = 192;
                let v = 128;
                let qh = 16;
                let kvh = 1;
                let page_tokens = 16;
                let element_bytes = if fp8 { 1 } else { 2 };
                let k_block = page_tokens * kvh * qk * element_bytes;
                let v_block = page_tokens * kvh * v * element_bytes;
                let layer_bytes = k_block + v_block;
                // Unequal FP8 scales, so a swapped pair cannot pass.
                let (k_scale, v_scale) = if fp8 { (0.5_f32, 0.25_f32) } else { (1.0, 1.0) };
                let page_bytes = 3 * layer_bytes;
                let pool_bytes = 2 * page_bytes;
                let k_base = layer_bytes;
                let v_base = layer_bytes + k_block;
                let positions = [15_i64, 16][..rows].to_vec();
                let row_width = qh * qk + kvh * (qk + v);
                let mut input = vec![0_u16; rows * row_width];
                let mut expected = vec![0xa5_u8; pool_bytes];
                for token in 0..rows {
                    let k = (token + 1) as f32;
                    let value = 2.0 * k;
                    let page = if token == 0 { 1 } else { 0 };
                    let offset = positions[token] as usize % page_tokens;
                    for d in 0..qk {
                        input[token * row_width + qh * qk + d] = f32_to_bf16(k);
                        let address =
                            page * page_bytes + k_base + (offset * qk + d) * element_bytes;
                        if fp8 {
                            expected[address] = kernel_core::kv_fp8::encode(k, k_scale);
                        } else {
                            expected[address..address + 2]
                                .copy_from_slice(&f32_to_bf16(k).to_ne_bytes());
                        }
                    }
                    for d in 0..v {
                        input[token * row_width + (qh + kvh) * qk + d] = f32_to_bf16(value);
                        let address = page * page_bytes + v_base + (offset * v + d) * element_bytes;
                        if fp8 {
                            expected[address] = kernel_core::kv_fp8::encode(value, v_scale);
                        } else {
                            expected[address..address + 2]
                                .copy_from_slice(&f32_to_bf16(value).to_ne_bytes());
                        }
                    }
                }
                let mut fixture = Fixture::new(session, rows)?;
                fixture.bf16("qkv", &input)?;
                fixture.zero("q", rows * qh * qk * 2)?;
                fixture.zero("k_new", rows * qk * 2)?;
                fixture.zero("v_new", rows * v * 2)?;
                fixture.i32("block_tables", &[1, 0])?;
                fixture.i32("slot_ids", &[0])?;
                fixture.i32("token_batch", &vec![0; rows])?;
                fixture.bytes(
                    "positions",
                    &positions
                        .iter()
                        .flat_map(|x| x.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
                fixture.f32(
                    "cos_sin",
                    &(0..=16)
                        .flat_map(|_| (0..32).flat_map(|_| [1.0_f32, 0.0]))
                        .collect::<Vec<_>>(),
                )?;
                fixture.scalar("max_pages", 2);
                fixture.scalar("ring_size", 0);
                fixture.scalar64("k_page_stride_bytes", page_bytes as u64);
                fixture.scalar64("v_page_stride_bytes", page_bytes as u64);
                fixture.scalar("theta_bits", 1e7_f32.to_bits());
                fixture.scalar("v_multiplier_bits", 1.0_f32.to_bits());
                fixture.scalar("k_scale_bits", k_scale.to_bits());
                fixture.scalar("v_scale_bits", v_scale.to_bits());
                let pool = session.allocate(pool_bytes)?;
                pool.copy_from_bytes(
                    &expected.iter().map(|_| 0xa5_u8).collect::<Vec<_>>(),
                    &session.stream,
                )
                .map_err(|e| e.to_string())?;
                fixture.launch_shared_paged(
                    &module,
                    entry,
                    &pool,
                    (k_base, v_base, pool_bytes),
                    rows as u32,
                )?;
                let mut actual = vec![0; pool_bytes];
                pool.copy_to_bytes(&mut actual, &session.stream)
                    .map_err(|e| e.to_string())?;
                fixture.check_exact(
                    &format!("{name} shared pool rows={rows}"),
                    &actual,
                    &expected,
                )?;
            }
        }
        Ok(())
    })
    .expect("strided multi-layer paged KV write");
}

#[test]
fn row_mimo_preweighted_combine_gpu() {
    gpu_test(64 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let entry = find(&entries, "combine_mimo4096");
        let width = 4096;
        let weights = [0.05_f32, 0.10, 0.15, 0.20, 0.05, 0.10, 0.15, 0.20];
        let down_values = seeded_bf16(8 * width, 0x7711);
        let weighted_down = down_values
            .iter()
            .enumerate()
            .map(|(i, &value)| f32_to_bf16(bf16_to_f32(value) * weights[i / width]))
            .collect::<Vec<_>>();
        let mut fixture = Fixture::new(session, 1)?;
        fixture.bf16("routed", &weighted_down)?;
        fixture.f32("weights", &weights)?;
        fixture.zero("output", width * 2)?;
        fixture.launch(&module, entry, 1, 1)?;
        let expected =
            oracle::expert_combine(&as_f64(&weighted_down), &[0; 8], &[1.0; 8], 1, width)
                .map_err(|e| e.to_string())?;
        let exact = (0..width)
            .map(|column| {
                let mut sum = 0.0_f32;
                for route in 0..8 {
                    sum += bf16_to_f32(weighted_down[route * width + column]);
                }
                f32_to_bf16(sum)
            })
            .collect::<Vec<_>>();
        let actual = fixture.read_bf16("output", width)?;
        fixture.check_exact("MiMo ordered preweighted combine", &actual, &exact)?;
        fixture.check_numeric(
            "MiMo weighted down then combine",
            &as_f32(&actual),
            &expected,
            DType::Bf16,
        )
    })
    .expect("MiMo weighted down output is combined once");
}

#[test]
fn row_logits_gpu() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        for rows in [1, 8, 32, 512, 2048] {
            logits_cases(session, &module, &entries, rows)?;
        }
        Ok(())
    })
    .expect("row logits reductions");
}

/// All-equal and strongly skewed rows at narrow and full live widths,
/// launched repeatedly: every warp must normalize by the row maximum, which
/// the CTA's shared reduction slot holds only until the sum reuses it.
#[test]
fn row_logits_shared_maximum_gpu() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let rows = 32;
        for name in [
            "logits_top20_full",
            "logits_top20_shard",
            "logits_top20_shard_std",
            "logits_top20_qwen38_full",
            "logits_top20_qwen38_shard",
            "logits_argmax_shard",
        ] {
            let entry = find(&entries, name);
            let width = entry.width as usize;
            let k = if name.contains("top20") { 20 } else { 1 };
            for live_width in [1, 31, 32, 33, width] {
                if live_width < k {
                    continue;
                }
                for skewed in [false, true] {
                    let mut logits = vec![20.0_f32; rows * width];
                    for row in 0..rows {
                        for (c, value) in logits[row * width..row * width + live_width]
                            .iter_mut()
                            .enumerate()
                        {
                            *value = match (skewed, c == (row * 7) % live_width) {
                                (false, _) => 0.0,
                                (true, true) => 30.0,
                                (true, false) => -((c % 97) as f32) / 4.0,
                            };
                        }
                    }
                    let mut fixture = Fixture::new(session, rows)?;
                    fixture.f32("logits", &logits)?;
                    fixture.poisoned("ids", rows * k * 4)?;
                    fixture.poisoned("log_probs", rows * k * 4)?;
                    fixture.u32("live_width", &[live_width as u32])?;
                    let expected = logits
                        .chunks(width)
                        .map(|row| {
                            oracle::logits_topk_log_probs(
                                &row.iter().map(|&x| x as f64).collect::<Vec<_>>(),
                                live_width,
                                k,
                            )
                            .map_err(|e| e.to_string())
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let ids = expected
                        .iter()
                        .flat_map(|row| row.ids.iter().map(|&x| x as i32))
                        .collect::<Vec<_>>();
                    let reference = oracle::OracleOutput::new(
                        expected
                            .iter()
                            .flat_map(|row| row.log_probs.iter().copied())
                            .collect(),
                        vec![1.0; rows * k],
                        live_width,
                    )
                    .map_err(|e| e.to_string())?;
                    let case = format!(
                        "{name} w{live_width} {}",
                        if skewed { "skewed" } else { "equal" }
                    );
                    let mut first = None;
                    for _ in 0..20 {
                        let outputs = [&fixture.buffers["ids"], &fixture.buffers["log_probs"]];
                        poison::poison(session, &outputs)?;
                        fixture.launch(&module, entry, rows as u32, rows as u32)?;
                        for output in outputs {
                            poison::check(
                                &case,
                                &poison::read(session, output)?,
                                &WriteMap::of(DType::F32).prefix(rows * k)?,
                                poison::Fill::Pattern(&[POISON]),
                            )?;
                        }
                        let log_probs = fixture.read_f32("log_probs", rows * k)?;
                        let bits = log_probs.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                        if first.get_or_insert_with(|| bits.clone()) != &bits {
                            return Err(format!("{case} log probabilities vary across launches"));
                        }
                        fixture.check_exact(
                            &format!("{case} ids"),
                            &fixture.read_i32("ids", rows * k)?,
                            &ids,
                        )?;
                        fixture.check_numeric(
                            &format!("{case} log probabilities"),
                            &log_probs,
                            &reference,
                            DType::F32,
                        )?;
                    }
                }
            }
        }
        Ok(())
    })
    .expect("row logits normalize by one maximum");
}

/// The Qwen3.8 gather over its whole 248,320 x 2560 BF16 table, first,
/// last and special rows copied bit for bit. The table is 1.27 GB, so its
/// offsets stay below 2^31 bytes; no package's embedding table crosses that
/// line, and `pool_offsets` records the product's width.
#[test]
fn row_qwen38_full_vocab_embedding_gpu() {
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let entry = find(&entries, "embedding_qwen38");
        let (vocab, width) = (248_320_usize, 2560_usize);
        let word = |row: usize, column: usize| {
            ((row.wrapping_mul(2_654_435_761) ^ column.wrapping_mul(40_503)) >> 3) as u16
        };
        let table = (0..vocab * width)
            .flat_map(|i| word(i / width, i % width).to_ne_bytes())
            .collect::<Vec<_>>();
        let ids = [0_i32, 1, 248_319, 248_046, 248_044, 131_072, 209_715, 7];
        let mut fixture = Fixture::new(session, ids.len())?;
        fixture.bytes("table", &table)?;
        drop(table);
        fixture.i32("ids", &ids)?;
        fixture.zero("output", ids.len() * width * 2)?;
        fixture.scalar("vocab_offset", 0);
        fixture.scalar("shard_vocab", vocab as u32);
        fixture.launch(&module, entry, 8, ids.len() as u32)?;
        let expected = ids
            .iter()
            .flat_map(|&id| (0..width).map(move |c| word(id as usize, c)))
            .collect::<Vec<_>>();
        fixture.check_exact(
            "embedding_qwen38 full vocabulary",
            &fixture.read_bf16("output", ids.len() * width)?,
            &expected,
        )
    })
    .expect("Qwen3.8 full-vocabulary embedding gather");
}

fn measured_bandwidth_peak(session: &Session) -> LabResult<(f64, String)> {
    let root = kernel_lab::receipt_root().join("probes");
    let mut receipts = std::fs::read_dir(root)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
        .collect::<Vec<_>>();
    receipts.sort_by_key(|(modified, _)| std::cmp::Reverse(*modified));
    for (_, path) in receipts {
        if let Ok(bytes) = std::fs::read(&path)
            && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
            && json.get("ordinal").and_then(|x| x.as_i64()) == Some(i64::from(session.ordinal))
            && json.get("uuid") == session.metadata.get("uuid")
            && let Some(value) = json
                .pointer("/dram_read/gb_per_second")
                .and_then(|x| x.as_f64())
            && value.is_finite()
            && value > 0.0
        {
            return Ok((value * 1e9, path.display().to_string()));
        }
    }
    Err("no matching GPU DRAM read probe receipt; run `kernel-lab probe` first".into())
}

fn measure_launch_floor(session: &Session) -> LabResult<Timing> {
    let module = session.module("probe_sm120")?;
    let function = module
        .function("chop_probe_sm120_v1")
        .map_err(|e| e.to_string())?;
    let output = session.allocate(4)?;
    time(session, 1, 5, |stream| {
        function
            .launch(
                stream,
                launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
                &[KernelArgument::Buffer(&output)],
            )
            .map_err(|e| e.to_string())
    })
}

fn row_bottleneck(name: &str) -> &'static str {
    if name.starts_with("rms_mimo_hf") {
        "exact HF output requires an ordered 4096-value FP32 square sum"
    } else if name.starts_with("router_") {
        "eight dependent top-k selections and sigmoid or softmax math"
    } else if name.starts_with("permute_count_") {
        "256 expert reductions scan the same L2-resident route list"
    } else if name.starts_with("rope_") {
        "rotation, head scheduling and scattered KV writes"
    } else if name.starts_with("logits_") {
        "FP32 logsumexp exponentials and row-wide maximum reduction"
    } else if name.starts_with("swiglu_") {
        "sigmoid exponential per output element"
    } else if name.starts_with("combine_") {
        "eight ordered contributions per output element"
    } else if name.starts_with("embedding_") {
        "repeated table rows are served from L2"
    } else {
        "launch floor and memory transfer"
    }
}

#[test]
fn row_bench_gpu() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("row benchmark: set CHOPINFER_KERNEL_BENCH=1").unwrap();
        return;
    }
    gpu_test(2 * 1024 * 1024 * 1024, |session| {
        let module = module(session)?;
        let entries = catalog();
        let (peak, peak_source) = measured_bandwidth_peak(session)?;
        let launch_floor_us = f64::from(measure_launch_floor(session)?.median_ms) * 1000.0;
        for rows in [1, 8, 32, 512, 2048, 8192] {
            for name in [
                "rms_mimo_hf4096",
                "quant_mxfp8_4096",
                "router_mimo_sigmoid256",
                "combine_mimo4096",
                "embedding_shard",
                "swiglu_halves_4096",
                "logits_argmax_shard",
                "permute_count_mimo",
                "rope_mimo_ga_bf16_computed_p32_pos32",
                "rope_mimo_ga_bf16_table_p32_pos32",
                "rope_mimo_swa_prefill_computed_p32_pos32",
                "rope_mimo_swa_prefill_table_p32_pos32",
                "rms_plain4096",
                "quant_mxfp8_8192",
                "router_std_softmax128",
                "combine_std8192",
                "embedding_qwen38",
                "logits_argmax_qwen38_shard",
            ] {
                if !case_enabled(name, rows) {
                    continue;
                }
                eprintln!("row bench case: rows={rows} entry={name}");
                let entry = find(&entries, name);
                if rows == 8192 && entry.profile == kernels_sm120::Profile::StdDenseMoe {
                    continue;
                }
                let width = entry.width as usize;
                let mut fixture = Fixture::new(session, rows)?;
                let bytes = match entry.query {
                    RowQuery::RmsNorm { .. } => {
                        fixture.bf16("x", &seeded_bf16(rows * width, 0x1311))?;
                        fixture.bf16("weight", &seeded_bf16(width, 0x1312))?;
                        fixture.zero("y", rows * width * 2)?;
                        (rows * width * 4) as f64
                    }
                    RowQuery::Quantize { .. } => {
                        fixture.bf16("input", &seeded_bf16(rows * width, 0x1313))?;
                        fixture.zero("codes", rows * width)?;
                        fixture.zero("scales", rows * width / 32)?;
                        (rows * width * 3) as f64
                    }
                    RowQuery::Router { .. } => {
                        fixture.f32("logits", &seeded_f32(rows * width, 0x1314))?;
                        fixture.f32("bias", &vec![0.0; width])?;
                        fixture.zero("ids", rows * 8 * 4)?;
                        fixture.zero("weights", rows * 8 * 4)?;
                        (rows * width * 4 + rows * 64) as f64
                    }
                    RowQuery::ExpertCombine { .. } => {
                        fixture.bf16("routed", &seeded_bf16(rows * 8 * width, 0x1315))?;
                        fixture.f32("weights", &vec![0.125; rows * 8])?;
                        fixture.zero("output", rows * width * 2)?;
                        (rows * width * 18) as f64
                    }
                    RowQuery::EmbeddingGather { .. } => {
                        fixture.bf16("table", &seeded_bf16(64 * width, 0x1316))?;
                        fixture.i32(
                            "ids",
                            &(0..rows).map(|i| (i % 64) as i32).collect::<Vec<_>>(),
                        )?;
                        fixture.zero("output", rows * width * 2)?;
                        fixture.scalar("vocab_offset", 0);
                        fixture.scalar("shard_vocab", 64);
                        (rows * width * 4) as f64
                    }
                    RowQuery::Swiglu { .. } => {
                        fixture.bf16("input", &seeded_bf16(rows * width * 2, 0x1317))?;
                        fixture.zero("output", rows * width * 2)?;
                        (rows * width * 6) as f64
                    }
                    RowQuery::LogitsArgmax { .. } | RowQuery::LogitsTopK { .. } => {
                        fixture.f32("logits", &seeded_f32(rows * width, 0x1318))?;
                        fixture.zero("ids", rows * 4)?;
                        fixture.zero("log_probs", rows * 4)?;
                        fixture.u32("live_width", &[width as u32])?;
                        (rows * width * 4 + rows * 8) as f64
                    }
                    RowQuery::ExpertPermute { .. } => {
                        fixture.i32(
                            "ids",
                            &(0..rows * 8).map(|i| (i % 256) as i32).collect::<Vec<_>>(),
                        )?;
                        fixture.zero("counts", 256 * 4)?;
                        (rows * 8 * 4 * 256 + 256 * 4) as f64
                    }
                    RowQuery::Rope { .. } => {
                        let ring = name.contains("swa");
                        let kvh = if ring { 2 } else { 1 };
                        let qk = 192;
                        let v = 128;
                        let pages = rows.div_ceil(32);
                        fixture.bf16(
                            "qkv",
                            &seeded_bf16(rows * (16 * qk + kvh * (qk + v)), 0x1319),
                        )?;
                        fixture.zero("q", rows * 16 * qk * 2)?;
                        fixture.zero("k_new", rows * kvh * qk * 2)?;
                        fixture.zero("v_new", rows * kvh * v * 2)?;
                        fixture.zero(
                            "k_pool",
                            (if ring { 128 } else { pages * 32 }) * kvh * qk * 2,
                        )?;
                        fixture.zero(
                            "v_pool",
                            (if ring { 128 } else { pages * 32 }) * kvh * v * 2,
                        )?;
                        fixture.i32(
                            "block_tables",
                            &(0..pages).map(|i| i as i32).collect::<Vec<_>>(),
                        )?;
                        fixture.i32("slot_ids", &[0])?;
                        fixture.i32("token_batch", &vec![0; rows])?;
                        fixture.i32(
                            "positions",
                            &(0..rows).map(|i| i as i32).collect::<Vec<_>>(),
                        )?;
                        if name.contains("table") {
                            let table = ComputedTable {
                                theta: if ring { 1e4 } else { 1e7 },
                                rule: AngleRule::DivPowF32,
                                axes: Axes::Linear,
                            };
                            let cos_sin = (0..rows)
                                .flat_map(|position| {
                                    table.row([position; 3], 32).expect("F32 table row")
                                })
                                .flat_map(|(cos, sin)| [cos as f32, sin as f32])
                                .collect::<Vec<_>>();
                            fixture.f32("cos_sin", &cos_sin)?;
                        }
                        fixture.scalar("max_pages", pages as u32);
                        fixture.scalar("ring_size", 128);
                        fixture.scalar64("k_page_stride_bytes", (32 * kvh * qk * 2) as u64);
                        fixture.scalar64("v_page_stride_bytes", (32 * kvh * v * 2) as u64);
                        fixture.scalar(
                            "theta_bits",
                            (if ring { 1e4_f32 } else { 1e7_f32 }).to_bits(),
                        );
                        fixture.scalar("v_multiplier_bits", 0.707_f32.to_bits());
                        fixture.scalar("k_scale_bits", 1.0_f32.to_bits());
                        fixture.scalar("v_scale_bits", 1.0_f32.to_bits());
                        (rows * (16 * qk + kvh * (qk + v)) * 4) as f64
                    }
                    _ => unreachable!(),
                };
                let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
                let timing = time(session, 1, 3, |_| {
                    fixture.launch_bound(&function, entry, rows as u32, rows as u32)
                })?;
                let achieved = bytes / (f64::from(timing.median_ms) * 1e-3);
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!(name));
                receipt.insert("tokens".into(), json!(rows));
                receipt.insert("median_us".into(), json!(timing.median_ms * 1000.0));
                receipt.insert("min_us".into(), json!(timing.min_ms * 1000.0));
                receipt.insert("rounds".into(), json!(timing.rounds));
                receipt.insert("achieved_bytes_per_s".into(), json!(achieved));
                receipt.insert("percent_peak".into(), json!(100.0 * achieved / peak));
                receipt.insert("peak_source".into(), json!(peak_source));
                receipt.insert("logical_bytes".into(), json!(bytes));
                receipt.insert("roofline_us".into(), json!(bytes / peak * 1e6));
                receipt.insert("launch_floor_us".into(), json!(launch_floor_us));
                receipt.insert("limiting_work".into(), json!(row_bottleneck(name)));
                session.receipt("bench", receipt)?;
            }
        }
        Ok(())
    })
    .expect("row benchmarks");
}

#[rustfmt::skip] #[path = "rope_positions.rs"] pub(crate) mod rope_positions;

pub(crate) fn generated_lookup_oracle(
    session: &Session,
    capability: &kernel_core::precision::capability::OperationCapability,
) -> LabResult<()> {
    let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|entry| entry.symbol == capability.stages[0])
        .ok_or("missing generated lookup entry")?;
    let RowQuery::EmbeddingGather { width, .. } = entry.query else {
        return Err("not an embedding entry".into());
    };
    let width = width as usize;
    let spec = kernels_sm120::row::cubin_specs(kernel_lab::catalogs(), None)
        .into_iter()
        .find(|s| s.entries.iter().any(|e| e.symbol == entry.symbol))
        .ok_or("missing lookup cubin")?;
    let module = session.module(&spec.name)?;
    let table = seeded_bf16(64 * width, 0x1316);
    let mut fixture = Fixture::new(session, capability.usage.capacity as usize)?;
    fixture.bf16("table", &table)?;
    let capacity = capability.usage.capacity;
    let ids = (0..capacity).map(|i| (i % 64) as i32).collect::<Vec<_>>();
    fixture.i32("ids", &ids)?;
    fixture.scalar("vocab_offset", 0);
    fixture.scalar("shard_vocab", 64);
    for live in [0, 1, capacity] {
        fixture.u32("live_rows", &[live])?;
        let mut previous = None;
        for _ in 0..2 {
            fixture.poisoned("output", (capacity as usize + 1) * width * 2)?;
            fixture.launch(&module, &entry, capacity, live)?;
            let actual = fixture.read_bf16("output", (capacity as usize + 1) * width)?;
            for (i, value) in actual.iter().enumerate() {
                let expected = if i < live as usize * width {
                    table[ids[i / width] as usize * width + i % width]
                } else {
                    u16::from_ne_bytes([POISON; 2])
                };
                if *value != expected {
                    return Err(format!("generated lookup differs at {i}"));
                }
            }
            if previous.as_ref().is_some_and(|old| old != &actual) {
                return Err("generated lookup replay differs".into());
            }
            previous = Some(actual);
        }
    }
    Ok(())
}
