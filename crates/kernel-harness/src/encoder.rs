//! Encoder conformance: each kernel entry a declared encoder stage selects,
//! run from its serving cubin against the stage's component oracle
//! (`kernel_core::multimodal_oracle`, `kernel_core::encoder_oracle`).
//!
//! A consumer passes its declarations, the catalog of its encoder entries,
//! its embedded cubins ([`cases`]) and a table of variant IDs and runners
//! ([`run`]): the runners here, or its own for a variant they do not cover.

use crate::launch::{self, Bindings};
use crate::poison::{Replay, WriteMap};
use crate::{LabResult, Session, check_output};
use encoder_spec::Declaration;
use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, MultimodalQuery};
use kernel_core::encoder_oracle as vision;
use kernel_core::multimodal_oracle as oracle;
use kernel_core::oracle::{GemmShape, OracleOutput, accumulation_bound, check_exact};
use kernels_sm120::encoder::{self, Entry, Registered};
use runtime_cuda::{DeviceBuffer, Module};
use std::collections::BTreeMap;
use std::ops::Range;

/// One kernel entry an encoder declaration selects, with what its runner
/// needs: the stage's variant ID and declared JSON, and the serving cubins
/// that hold the entry.
pub struct Case {
    /// The stage's variant ID, such as `linear.pairs.v1`.
    pub variant: String,
    pub entry: Entry,
    /// Whether the entry is the first GEMM over the payload, behind any
    /// column padding: its runner also checks it at full capacity.
    pub input_linear: bool,
    /// The declared stage.
    pub stage: serde_json::Value,
    /// The serving cubin of a name ([`encoder::ROWS_GEMM_CUBIN`] or
    /// [`encoder::ATTENTION_CUBIN`]), such as a bundle's embedded one.
    pub cubins: fn(&str) -> Option<&'static [u8]>,
}

impl Case {
    /// The serving cubin `name` that holds this case's entry.
    pub fn cubin(&self, name: &str) -> LabResult<&'static [u8]> {
        (self.cubins)(name).ok_or_else(|| format!("missing serving cubin {name}"))
    }
}

/// Runs a case on a session; [`Runner`]s check one kind of entry against
/// its component oracle.
pub type Runner = fn(&Session, &Case) -> LabResult<()>;

/// The cases `declarations` select through their kernels (each with the
/// kernels it registers), one per entry symbol in declaration order, with
/// each entry from `catalog` and its cubins from `cubins`.
pub fn cases(
    declarations: &[(&Declaration, &dyn Registered)],
    catalog: &[Entry],
    cubins: fn(&str) -> Option<&'static [u8]>,
) -> LabResult<Vec<Case>> {
    let mut cases: Vec<Case> = Vec::new();
    for &(declaration, registered) in declarations {
        let json: serde_json::Value =
            serde_json::from_str(&declaration.canonical_json()).map_err(|e| e.to_string())?;
        for (slot, kernel) in
            encoder::kernels(declaration, registered).map_err(|e| e.0.to_string())?
        {
            let symbol = kernel.entry()?.symbol;
            // The first GEMM over the payload, behind any column padding.
            let input_linear = slot.part == "linear"
                && declaration.stages[..slot.stage]
                    .iter()
                    .all(|stage| stage.op.id() == "pad.zero-columns.v1");
            if let Some(known) = cases.iter_mut().find(|c| c.entry.symbol == symbol) {
                if known.variant != kernel.variant {
                    return Err(format!("{symbol}: two variants select one entry"));
                }
                known.input_linear |= input_linear;
                continue;
            }
            let entry = catalog
                .iter()
                .find(|e| e.symbol == symbol)
                .ok_or_else(|| format!("{symbol}: not in the catalog"))?
                .clone();
            cases.push(Case {
                variant: kernel.variant,
                entry,
                input_linear,
                stage: json["stages"][slot.stage].clone(),
                cubins,
            });
        }
    }
    Ok(cases)
}

/// Runs every case of variant `id` with its runner from `table` on a
/// session of the selected GPU ([`crate::gpu_test`], skipped without one),
/// and fails with every case's error. Refuses a variant with no runner or
/// no case.
pub fn run(id: &str, cases: &[Case], table: &[(&str, Runner)]) -> LabResult<()> {
    let runner = table
        .iter()
        .find(|(variant, _)| *variant == id)
        .ok_or_else(|| format!("no runner for {id}"))?
        .1;
    let cases = cases.iter().filter(|c| c.variant == id).collect::<Vec<_>>();
    if cases.is_empty() {
        return Err(format!("no cases for {id}"));
    }
    crate::gpu_test(4 << 30, |session| {
        let mut misses = Vec::new();
        for case in &cases {
            eprintln!("{id} {}", case.entry.symbol);
            if let Err(error) = runner(session, case) {
                eprintln!("{id}: {error}");
                misses.push(error);
            }
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(misses.join("; "))
        }
    })
    .map(|_| ())
}

struct Lab<'a> {
    entry: &'a Entry,
    session: &'a Session,
    rows: Module,
    attention: Module,
}

impl<'a> Lab<'a> {
    fn new(session: &'a Session, case: &'a Case) -> LabResult<Self> {
        Ok(Self {
            session,
            entry: &case.entry,
            rows: session.load_cubin(case.cubin(encoder::ROWS_GEMM_CUBIN)?)?,
            attention: session.load_cubin(case.cubin(encoder::ATTENTION_CUBIN)?)?,
        })
    }

    /// Launches one entry with buffers bound by parameter name and scalars
    /// by name. Unbound pointers are null.
    fn run(
        &self,
        name: &str,
        live: u32,
        buffers: &[(&str, &DeviceBuffer)],
        scalars: &[(&str, u64)],
    ) -> LabResult<()> {
        let entry = self.entry;
        assert_eq!(name, entry.symbol);
        let module = if entry.is_attention() {
            &self.attention
        } else {
            &self.rows
        };
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        let plan = entry.plan(live).map_err(|e| e.to_string())?;
        if plan.dynamic_shared_bytes >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        let arguments = entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } => Ok(buffers
                    .iter()
                    .position(|(key, _)| key == name)
                    .map_or(Argument::NullDevicePointer, |index| {
                        Argument::DeviceBuffer(index as u64 + 1)
                    })),
                EntryParam::U64 { name } => scalars
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|&(_, value)| Argument::U64(value))
                    .ok_or_else(|| format!("{}: missing scalar {name}", entry.symbol)),
                EntryParam::U32 { name } => scalars
                    .iter()
                    .find(|(key, _)| key == name)
                    .map(|&(_, value)| Argument::U32(value as u32))
                    .ok_or_else(|| format!("{}: missing scalar {name}", entry.symbol)),
                _ => Err(format!("{}: unexpected ABI kind", entry.symbol)),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let arguments = entry.pack_arguments(arguments).map_err(|e| e.to_string())?;
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .enumerate()
                .map(|(index, (_, buffer))| (index as u64 + 1, *buffer))
                .collect(),
        };
        launch::launch(
            &self.session.context,
            &function,
            &self.session.stream,
            plan,
            &arguments,
            &bindings,
        )
    }

    fn bytes(&self, bytes: &[u8]) -> LabResult<DeviceBuffer> {
        let buffer = self.session.allocate(bytes.len().max(1))?;
        buffer
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    }
    fn bf16(&self, values: &[u16]) -> LabResult<DeviceBuffer> {
        self.session.upload_bf16(values)
    }
    fn f32(&self, values: &[f32]) -> LabResult<DeviceBuffer> {
        self.session.upload_f32(values)
    }
    fn i32(&self, values: &[i32]) -> LabResult<DeviceBuffer> {
        self.bytes(
            &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    }
    fn count(&self, value: usize) -> LabResult<DeviceBuffer> {
        self.bytes(&(value as u32).to_ne_bytes())
    }
    /// Output storage of `len` elements of `width` bytes, poisoned with NaN
    /// (or -1 for codes), so an unwritten element fails.
    fn output(&self, len: usize, width: usize) -> LabResult<DeviceBuffer> {
        self.bytes(&vec![0xff; len * width])
    }
    fn read(&self, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u16>> {
        self.session.download_bf16(buffer, len)
    }
    fn read_f32(&self, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<f32>> {
        self.session.download_f32(buffer, len)
    }
}

/// BF16 values in [-scale, scale] from a fixed xorshift stream.
fn seeded(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16((((state >> 40) % 513) as f32 / 256.0 - 1.0) * scale)
        })
        .collect()
}
/// F32 values in [-scale, scale] with all 24 bits populated.
fn seeded_f32(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}
fn decoded(values: &[u16]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(bf16_to_f32(x))).collect()
}
fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&x| f64::from(x)).collect()
}
/// The values the kernels multiply for F32 operands: their BF16 pairs.
fn pair_values(values: &[f32]) -> Vec<f64> {
    values
        .iter()
        .map(|&x| oracle::bf16_pair(f64::from(x)))
        .collect()
}
fn split(x: f32) -> [u16; 2] {
    let hi = f32_to_bf16(x);
    [hi, f32_to_bf16(x - bf16_to_f32(hi))]
}
fn rows_of<T: Copy>(values: &[T], width: usize, rows: &[usize]) -> Vec<T> {
    rows.iter()
        .flat_map(|&row| values[row * width..(row + 1) * width].iter().copied())
        .collect()
}
fn check(name: &str, actual: &[u16], expected: &OracleOutput) -> LabResult<()> {
    let actual = actual.iter().map(|&x| bf16_to_f32(x)).collect::<Vec<_>>();
    let mut receipt = BTreeMap::new();
    check_output(&actual, expected, DType::Bf16, &mut receipt)
        .map_err(|e| format!("{name}: {e}; {receipt:?}"))
}
fn check_f32(name: &str, actual: &[f32], expected: &OracleOutput) -> LabResult<()> {
    if actual.iter().any(|x| !x.is_finite()) {
        return Err(format!("{name}: unwritten or non-finite output"));
    }
    let mut receipt = BTreeMap::new();
    check_output(actual, expected, DType::F32, &mut receipt)
        .map_err(|e| format!("{name}: {e}; {receipt:?}"))
}
fn concat(parts: Vec<OracleOutput>) -> OracleOutput {
    let mut out = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    for part in parts {
        out.values.extend(part.values);
        out.sum_abs_products.extend(part.sum_abs_products);
        out.reduction_lengths.extend(part.reduction_lengths);
    }
    out
}
/// Runs `work` over `count` items in contiguous ranges, one per CPU.
fn parallel<T: Send>(count: usize, work: impl Fn(Range<usize>) -> Vec<T> + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let chunk = count.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles = (0..count)
            .step_by(chunk)
            .map(|start| {
                let work = &work;
                scope.spawn(move || work(start..(start + chunk).min(count)))
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("oracle thread"))
            .collect()
    })
}
/// The L2 distance between F32 results and a reference.
fn distance(actual: &[f32], reference: &[f64]) -> f64 {
    actual
        .iter()
        .zip(reference)
        .map(|(&x, y)| (f64::from(x) - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn rotary(session: &Session, case: &Case) -> LabResult<()> {
    let lab = Lab::new(session, case)?;
    let name = case.entry.symbol.as_str();
    let m = 300.min(case.entry.capacity() as usize);
    let live = lab.count(m)?;
    let query = case.entry.query;
    let MultimodalQuery::RopeQkvPairs {
        q_heads,
        kv_heads,
        head_dim,
        rotary_dim,
        ..
    } = query
    else {
        unreachable!("rope entries carry rope queries")
    };
    let (q_width, kv_width) = (
        (q_heads * head_dim) as usize,
        (kv_heads * head_dim) as usize,
    );
    let width = q_width + 2 * kv_width;
    let pairs = rotary_dim as usize / 2;
    let input = seeded_f32(m * width, 0x7720 ^ width as u64, 6.0);
    let angles = seeded_f32(m * pairs, 0x7721, 3.0);
    let cos = angles.iter().map(|x| x.cos()).collect::<Vec<_>>();
    let sin = angles.iter().map(|x| x.sin()).collect::<Vec<_>>();
    let (rows, cos_buffer, sin_buffer) = (lab.f32(&input)?, lab.f32(&cos)?, lab.f32(&sin)?);
    lab.run(
        name,
        m as u32,
        &[
            ("rows", &rows),
            ("cos", &cos_buffer),
            ("sin", &sin_buffer),
            ("live_rows", &live),
        ],
        &[],
    )?;
    let planes = lab.read(&rows, m * 2 * width)?;
    let [q, k, _] = oracle::rope_qkv_pairs(query, &widen(&input), &widen(&cos), &widen(&sin))
        .map_err(|e| e.to_string())?;
    // Plane offsets in a packed row: q hi 0, k hi 2 q_width; each
    // low plane right after its high plane.
    for (tensor, start, tensor_width, expected) in
        [(0, 0, q_width, &q), (1, 2 * q_width, kv_width, &k)]
    {
        let actual = planes
            .chunks_exact(2 * width)
            .flat_map(|row| {
                row[start..][..tensor_width]
                    .iter()
                    .zip(&row[start + tensor_width..][..tensor_width])
                    .map(|(&hi, &lo)| f64::from(bf16_to_f32(hi)) + f64::from(bf16_to_f32(lo)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for (index, (&value, &reference)) in actual.iter().zip(&expected.values).enumerate() {
            let bound = accumulation_bound(
                expected.reduction_lengths[index],
                expected.sum_abs_products[index],
            ) + 2_f64.powi(-16) * reference.abs();
            if (value - reference).abs() > bound {
                return Err(format!(
                    "{name} tensor {tensor} element {index}: {value} vs {reference}"
                ));
            }
        }
    }
    // Check each high/low plane at its GQA width, including V.
    for row in planes.chunks_exact(2 * width) {
        for (start, width) in [
            (0, q_width),
            (2 * q_width, kv_width),
            (2 * q_width + 2 * kv_width, kv_width),
        ] {
            for (&hi, &lo) in row[start..][..width]
                .iter()
                .zip(&row[start + width..][..width])
            {
                let hi = f64::from(bf16_to_f32(hi));
                let unit = if hi == 0.0 {
                    0.0
                } else {
                    2_f64.powi(hi.abs().log2().floor() as i32 - 7)
                };
                if f64::from(bf16_to_f32(lo)).abs() > unit / 2.0 {
                    return Err(format!("{name}: low half exceeds half a BF16 unit"));
                }
            }
        }
    }
    let v = input
        .chunks_exact(width)
        .flat_map(|row| row[q_width + kv_width..].iter().map(|&x| split(x)))
        .collect::<Vec<_>>();
    let actual = planes
        .chunks_exact(2 * width)
        .flat_map(|row| {
            let hi = &row[2 * q_width + 2 * kv_width..][..kv_width];
            let lo = &row[2 * q_width + 3 * kv_width..][..kv_width];
            hi.iter()
                .zip(lo)
                .map(|(&hi, &lo)| [hi, lo])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    check_exact(&actual, &v).map_err(|index| format!("{name}: v pairs differ at {index}"))?;

    Ok(())
}
pub fn gemm(session: &Session, case: &Case) -> LabResult<()> {
    let lab = Lab::new(session, case)?;
    let name = case.entry.symbol.as_str();
    let mut counts = vec![300.min(case.entry.capacity() as usize)];
    if case.input_linear {
        counts.push(case.entry.capacity() as usize);
    }
    for m in counts {
        let live = lab.count(m)?;
        let MultimodalQuery::GemmPairs(query) = case.entry.query else {
            unreachable!("filtered on pair GEMM queries")
        };
        let (n, k) = (query.n as usize, query.k as usize);
        let a = seeded_f32(m * k, 0x7730 ^ k as u64, 1.0);
        let b = seeded(n * k, 0x7731 ^ n as u64, 2.0 / (k as f32).sqrt());
        let bias = seeded(n, 0x7732, 0.25);
        let (a_buffer, b_buffer, bias_buffer) = (lab.f32(&a)?, lab.bf16(&b)?, lab.bf16(&bias)?);
        let f32_output = query.output == DType::F32;
        let residual = query
            .epilogues
            .residual_add
            .then(|| seeded_f32(m * n, 0x7733, 4.0));
        let output = match (&residual, f32_output) {
            (Some(residual), _) => lab.f32(residual)?,
            (None, true) => lab.output(m * n, 4)?,
            (None, false) => lab.output(m * n, 2)?,
        };
        lab.run(
            name,
            m as u32,
            &[
                ("activation", &a_buffer),
                ("weight", &b_buffer),
                ("bias", &bias_buffer),
                ("output", &output),
                ("live_rows", &live),
            ],
            &[],
        )?;
        let sampled = [0, 1, 127, 128, 200, m / 2 + 3, m - 2, m - 1];
        let rows = pair_values(&rows_of(&a, k, &sampled));
        let sampled_residual = residual.as_ref().map(|x| widen(&rows_of(x, n, &sampled)));
        let zero_bias = vec![0.0; n];
        let expected = gemm_rows(
            &rows,
            &decoded(&b),
            (n, k),
            if query.epilogues.bias {
                decoded(&bias)
            } else {
                zero_bias
            },
            query.epilogues.gelu,
            sampled_residual.as_deref(),
        );
        if f32_output {
            let full = lab.read_f32(&output, m * n)?;
            if full.iter().any(|x| !x.is_finite()) {
                return Err(format!("{name}: unwritten output"));
            }
            let actual = rows_of(&full, n, &sampled);
            check_f32(name, &actual, &expected)?;
            let rounded = rows_of(&a, k, &sampled)
                .iter()
                .map(|&x| f64::from(bf16_to_f32(f32_to_bf16(x))))
                .collect::<Vec<_>>();
            let bf16 = gemm_rows(
                &rounded,
                &decoded(&b),
                (n, k),
                if query.epilogues.bias {
                    decoded(&bias)
                } else {
                    vec![0.0; n]
                },
                query.epilogues.gelu,
                sampled_residual.as_deref(),
            );
            if distance(&actual, &expected.values) * 16.0 > distance(&actual, &bf16.values) {
                return Err(format!("{name}: not closer to pairs than BF16"));
            }
        } else {
            let full = lab.read(&output, m * n)?;
            if full.iter().any(|&x| !bf16_to_f32(x).is_finite()) {
                return Err(format!("{name}: unwritten or non-finite BF16 output"));
            }
            eprintln!(
                "{name}: checked all {} BF16 outputs for finiteness",
                full.len()
            );
            check(name, &rows_of(&full, n, &sampled), &expected)?;
        }
    }
    Ok(())
}
/// The f64 encoder GEMM oracle over the sampled rows `a`.
fn gemm_rows(
    a: &[f64],
    b: &[f64],
    (n, k): (usize, usize),
    bias: Vec<f64>,
    gelu: Option<kernel_core::contract::GeluMode>,
    residual: Option<&[f64]>,
) -> OracleOutput {
    let m = a.len() / k;
    concat(parallel(m, |rows| {
        vec![
            oracle::gemm_bias_epilogue(
                &a[rows.start * k..rows.end * k],
                b,
                GemmShape {
                    m: rows.len(),
                    n,
                    k,
                    experts: 1,
                },
                Some(&bias),
                gelu,
                None,
                residual.map(|x| &x[rows.start * n..rows.end * n]),
            )
            .expect("GEMM oracle"),
        ]
    }))
}

/// Packed pair rows: q hi, q lo, k hi, k lo, v hi, v lo per token.
fn packed_pairs(q: &[f32], k: &[f32], v: &[f32], q_width: usize, kv_width: usize) -> Vec<u16> {
    let tokens = q.len() / q_width;
    let mut rows = Vec::with_capacity(tokens * 2 * (q_width + 2 * kv_width));
    for token in 0..tokens {
        for (tensor, width) in [(q, q_width), (k, kv_width), (v, kv_width)] {
            let pairs = tensor[token * width..][..width]
                .iter()
                .map(|&x| split(x))
                .collect::<Vec<_>>();
            rows.extend(pairs.iter().map(|pair| pair[0]));
            rows.extend(pairs.iter().map(|pair| pair[1]));
        }
    }
    rows
}

pub fn rows(session: &Session, case: &Case) -> LabResult<()> {
    let lab = Lab::new(session, case)?;
    let name = case.entry.symbol.as_str();
    let m = 300.min(case.entry.capacity() as usize);
    let live = lab.count(m)?;
    match case.entry.query {
        MultimodalQuery::PermuteF32 { width, .. } => {
            let width = width as usize;
            let input = seeded_f32(m * width, 0x7703, 4.0);
            let mut indices = (0..m as u32).collect::<Vec<_>>();
            indices.reverse();
            indices.swap(3, 200);
            let (input_buffer, index_buffer) = (
                lab.f32(&input)?,
                lab.i32(&indices.iter().map(|&x| x as i32).collect::<Vec<_>>())?,
            );
            let output = lab.output(m * width, 4)?;
            lab.run(
                name,
                m as u32,
                &[
                    ("input", &input_buffer),
                    ("indices", &index_buffer),
                    ("output", &output),
                    ("live_rows", &live),
                ],
                &[],
            )?;
            let expected = rows_of(
                &input,
                width,
                &indices.iter().map(|&x| x as usize).collect::<Vec<_>>(),
            );
            check_exact(&lab.read_f32(&output, m * width)?, &expected)
                .map_err(|index| format!("{name} differs at {index}"))?;
        }
        MultimodalQuery::GatherRowsF32 { width, group, .. } => {
            // A space-to-depth merge over odd grids, whose last cells run
            // past the grid: their blocks must be zeros, every other block
            // its patch row exactly, and the row past the live cells must
            // keep its poison.
            let (width, group) = (width as usize, group as usize);
            let factor = group.isqrt();
            if factor * factor != group {
                return Err(format!(
                    "{name}: a gather group of {group} is not a square cell"
                ));
            }
            let capacity = case.entry.capacity() as usize;
            for [h, w] in [[7_u64, 5], [31, 51], [1, 1]] {
                let patches = (h * w) as usize;
                let indices =
                    encoder_spec::geometry::space_to_depth_indices([1, h, w], factor as u64)?;
                let cells = indices.len() / group;
                if cells + 1 > capacity {
                    continue;
                }
                let input = seeded_f32(patches * width, 0x7708 ^ h ^ (w << 8), 4.0);
                let (input_buffer, index_buffer) = (lab.f32(&input)?, lab.i32(&indices)?);
                let output = lab.output((cells + 1) * group * width, 4)?;
                let live = lab.count(cells)?;
                lab.run(
                    name,
                    cells as u32,
                    &[
                        ("input", &input_buffer),
                        ("indices", &index_buffer),
                        ("output", &output),
                        ("live_rows", &live),
                    ],
                    &[],
                )?;
                let expected = oracle::gather_rows(&widen(&input), width, &indices)
                    .map_err(|e| e.to_string())?;
                let actual = lab.read_f32(&output, (cells + 1) * group * width)?;
                let (written, past) = actual.split_at(cells * group * width);
                // A gather moves bits: each value, a hole's +0 included, is
                // the oracle's bit for bit, and past the live cells the
                // poison's exact bits remain.
                if written.len() != expected.values.len() {
                    return Err(format!("{name} {h} x {w}: the oracle's rows differ"));
                }
                if let Some(index) = written
                    .iter()
                    .zip(&expected.values)
                    .position(|(x, &e)| x.to_bits() != (e as f32).to_bits())
                {
                    return Err(format!("{name} {h} x {w} differs at {index}"));
                }
                if let Some(index) = past.iter().position(|x| x.to_bits() != u32::MAX) {
                    return Err(format!(
                        "{name} {h} x {w}: wrote past its live cells at {index}"
                    ));
                }
            }
        }
        MultimodalQuery::SwigluF32 { hidden, .. } => {
            let hidden = hidden as usize;
            let input = seeded_f32(m * 2 * hidden, 0x7704 ^ hidden as u64, 12.0);
            let input_buffer = lab.f32(&input)?;
            let output = lab.output(m * hidden, 4)?;
            lab.run(
                name,
                m as u32,
                &[
                    ("input", &input_buffer),
                    ("output", &output),
                    ("live_rows", &live),
                ],
                &[],
            )?;
            let expected =
                oracle::swiglu_rows(&widen(&input), hidden).map_err(|e| e.to_string())?;
            check_f32(name, &lab.read_f32(&output, m * hidden)?, &expected)?;
        }
        MultimodalQuery::ResidualAddF32 { width, .. } => {
            let width = width as usize;
            let input = seeded_f32(m * width, 0x7705, 64.0);
            let residual = seeded_f32(m * width, 0x7706, 1.0);
            let (input_buffer, residual_buffer) = (lab.f32(&input)?, lab.f32(&residual)?);
            // In place, as the carry adds into the residual stream.
            lab.run(
                name,
                m as u32,
                &[
                    ("input", &input_buffer),
                    ("residual", &residual_buffer),
                    ("output", &input_buffer),
                    ("live_rows", &live),
                ],
                &[],
            )?;
            let expected = oracle::residual_add(&widen(&input), &widen(&residual))
                .map_err(|e| e.to_string())?;
            check_f32(name, &lab.read_f32(&input_buffer, m * width)?, &expected)?;
        }
        MultimodalQuery::Im2col1dF32 {
            channels,
            kernel,
            stride,
            pad_left,
            pad_right,
            ..
        } => {
            let (channels, kernel, stride, pad) = (
                channels as usize,
                kernel as usize,
                stride as usize,
                [pad_left as usize, pad_right as usize],
            );
            let frames = 601;
            let live_frames = lab.count(frames)?;
            let input = seeded_f32(frames * channels, 0x7707 ^ channels as u64, 3.0);
            let input_buffer = lab.f32(&input)?;
            let rows = (frames + pad[0] + pad[1] - kernel) / stride + 1;
            let output = lab.output((rows + 1) * kernel * channels, 4)?;
            lab.run(
                name,
                rows as u32,
                &[
                    ("input", &input_buffer),
                    ("output", &output),
                    ("live_frames", &live_frames),
                ],
                &[],
            )?;
            let expected =
                oracle::im2col1d_f32(&widen(&input), channels, kernel, stride, pad[0], pad[1])
                    .map_err(|e| e.to_string())?;
            let actual = lab.read_f32(&output, (rows + 1) * kernel * channels)?;
            let (written, past) = actual.split_at(rows * kernel * channels);
            check_exact(&widen(written), &expected.values)
                .map_err(|index| format!("{name} differs at {index}"))?;
            if past.iter().any(|x| !x.is_nan()) {
                return Err(format!("{name}: wrote past its output rows"));
            }
        }
        MultimodalQuery::PadColumnsF32 { input, output, .. } => {
            let (input, output) = (input as usize, output as usize);
            // One row past the live rows, which must keep its poison.
            let rows = seeded_f32((m + 1) * input, 0x7708 ^ input as u64, 4.0);
            let input_buffer = lab.f32(&rows)?;
            let padded = lab.output((m + 1) * output, 4)?;
            let launch = |_: &runtime_cuda::Stream| {
                lab.run(
                    name,
                    m as u32,
                    &[
                        ("input", &input_buffer),
                        ("output", &padded),
                        ("live_rows", &live),
                    ],
                    &[],
                )
            };
            let replay = Replay::new(session).output(
                "padded rows",
                &padded,
                WriteMap::of(DType::F32).prefix(m * output)?,
            );
            let actual = replayed(session, replay, &launch, &padded, m * output)?;
            let expected = vision::pad_columns(&widen(&rows[..m * input]), input, output)
                .map_err(|e| e.to_string())?;
            check_exact(&widen(&actual), &expected.values)
                .map_err(|index| format!("{name} differs at {index}"))?;
        }
        MultimodalQuery::SwigluClampedF32 {
            hidden, limit_bits, ..
        } => {
            let hidden = hidden as usize;
            let limit = f32::from_bits(limit_bits);
            // Gates and ups up to twice the limit on both sides.
            let input = seeded_f32((m + 1) * 2 * hidden, 0x7709 ^ hidden as u64, 2.0 * limit);
            let input_buffer = lab.f32(&input)?;
            let output = lab.output((m + 1) * hidden, 4)?;
            let launch = |_: &runtime_cuda::Stream| {
                lab.run(
                    name,
                    m as u32,
                    &[
                        ("input", &input_buffer),
                        ("output", &output),
                        ("live_rows", &live),
                    ],
                    &[("limit_bits", u64::from(limit_bits))],
                )
            };
            let replay = Replay::new(session).output(
                "clamped rows",
                &output,
                WriteMap::of(DType::F32).prefix(m * hidden)?,
            );
            let actual = replayed(session, replay, &launch, &output, m * hidden)?;
            let expected = vision::clamped_swiglu_rows(
                &widen(&input[..m * 2 * hidden]),
                hidden,
                f64::from(limit),
            )
            .map_err(|e| e.to_string())?;
            check_f32(name, &actual, &expected)?;
        }
        MultimodalQuery::GeluRowsF32 { width, .. } => {
            let width = width as usize;
            let input = seeded_f32((m + 1) * width, 0x770a ^ width as u64, 12.0);
            let buffer = lab.f32(&input)?;
            // In place, as the projector runs it.
            let launch = |_: &runtime_cuda::Stream| {
                lab.run(
                    name,
                    m as u32,
                    &[
                        ("input", &buffer),
                        ("output", &buffer),
                        ("live_rows", &live),
                    ],
                    &[],
                )
            };
            let initial = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let replay = Replay::new(session).state(
                "gelu rows",
                &buffer,
                initial,
                WriteMap::of(DType::F32).none()?,
                std::iter::once(0..m * width * 4).collect(),
            );
            let actual = replayed(session, replay, &launch, &buffer, m * width)?;
            let expected =
                vision::gelu_rows_f32(&widen(&input[..m * width])).map_err(|e| e.to_string())?;
            check_f32(name, &actual, &expected)?;
        }
        _ => return Err(format!("no row oracle for {}", case.variant)),
    }
    Ok(())
}

/// Runs `launch` eager, relaunched and from a graph under `replay`'s
/// poison and restore rules, and returns the eager result's first `len`
/// F32 elements of `output`, which every replay must equal bitwise.
fn replayed(
    session: &Session,
    mut replay: Replay<'_>,
    launch: &dyn Fn(&runtime_cuda::Stream) -> LabResult<()>,
    output: &DeviceBuffer,
    len: usize,
) -> LabResult<Vec<f32>> {
    replay.eager(|| launch(&session.stream))?;
    let eager = session.download_f32(output, len)?;
    replay.relaunch(|| launch(&session.stream))?;
    same_bits(&session.download_f32(output, len)?, &eager)?;
    let graph = session.capture(launch)?;
    replay.graph(&graph)?;
    same_bits(&session.download_f32(output, len)?, &eager)?;
    let mut fields = BTreeMap::new();
    replay.record(&mut fields);
    session.receipt("encoder-rows-replay", fields)?;
    Ok(eager)
}

/// The q/k RMSNorm attention's two entries: the per-head norm in place on
/// the packed qkv rows, and the pair attention.
pub fn qk_rms_attention(session: &Session, case: &Case) -> LabResult<()> {
    let MultimodalQuery::QkRmsNormF32 {
        q_heads,
        kv_heads,
        head_dim,
        eps_bits,
        ..
    } = case.entry.query
    else {
        return attention(session, case);
    };
    let (q, kv, d) = (q_heads as usize, kv_heads as usize, head_dim as usize);
    let (width, normed) = ((q + 2 * kv) * d, (q + kv) * d);
    let lab = Lab::new(session, case)?;
    let name = case.entry.symbol.as_str();
    let m = 300.min(case.entry.capacity() as usize);
    let live = lab.count(m)?;
    // Heads of different scales, from tiny to large, and one row past the
    // live rows, which keeps its bytes.
    let mut rows = seeded_f32((m + 1) * width, 0x770b, 3.0);
    for (index, value) in rows.iter_mut().enumerate() {
        *value *= [1e-3, 1.0, 40.0][(index / d) % 3];
    }
    let (q_weight, k_weight) = (seeded(d, 0x770c, 1.5), seeded(d, 0x770d, 1.5));
    let (buffer, q_buffer, k_buffer) =
        (lab.f32(&rows)?, lab.bf16(&q_weight)?, lab.bf16(&k_weight)?);
    let launch = |_: &runtime_cuda::Stream| {
        lab.run(
            name,
            m as u32,
            &[
                ("rows", &buffer),
                ("q_weight", &q_buffer),
                ("k_weight", &k_buffer),
                ("live_rows", &live),
            ],
            &[("eps_bits", u64::from(eps_bits))],
        )
    };
    let initial = rows.iter().flat_map(|v| v.to_ne_bytes()).collect();
    // q and k of every live row are rewritten; v and the extra row keep
    // their bytes.
    let replay = Replay::new(session).state(
        "qkv rows",
        &buffer,
        initial,
        WriteMap::of(DType::F32).none()?,
        (0..m)
            .map(|row| row * width * 4..(row * width + normed) * 4)
            .collect(),
    );
    let actual = replayed(session, replay, &launch, &buffer, m * width)?;
    let expected = vision::qk_rms_norm(
        &widen(&rows[..m * width]),
        q,
        kv,
        d,
        &decoded(&q_weight),
        &decoded(&k_weight),
        f64::from(f32::from_bits(eps_bits)),
    )
    .map_err(|e| e.to_string())?;
    check_f32(name, &actual, &expected)
}
pub fn norm(session: &Session, case: &Case) -> LabResult<()> {
    let (width, eps_bits, layer) = match case.entry.query {
        MultimodalQuery::LayerNormF32 {
            width, eps_bits, ..
        } => (width as usize, eps_bits, true),
        MultimodalQuery::RmsNormF32 {
            width, eps_bits, ..
        } => (width as usize, eps_bits, false),
        _ => return Err("expected F32 norm".into()),
    };
    let lab = Lab::new(session, case)?;
    let name = &case.entry.symbol;
    let m = 300.min(case.entry.capacity() as usize);
    let live = lab.count(m)?;
    let input = seeded_f32(m * width, 0x7701 ^ width as u64, 30.0);
    let weight = seeded(width, 0x7702, 2.0);
    let bias = seeded(width, 0x7703, 0.25);
    let (x, w, b) = (lab.f32(&input)?, lab.bf16(&weight)?, lab.bf16(&bias)?);
    let output = lab.output(m * width, 4)?;
    lab.run(
        name,
        m as u32,
        &[
            ("input", &x),
            ("weight", &w),
            ("bias", &b),
            ("output", &output),
            ("live_rows", &live),
        ],
        &[("eps_bits", u64::from(eps_bits))],
    )?;
    let eps = f64::from(f32::from_bits(eps_bits));
    let expected = if layer {
        oracle::layernorm_affine(
            &widen(&input),
            width,
            &decoded(&weight),
            &decoded(&bias),
            eps,
        )
    } else {
        oracle::rms_norm(&widen(&input), width, &decoded(&weight), eps)
    }
    .map_err(|e| e.to_string())?;
    check_f32(name, &lab.read_f32(&output, m * width)?, &expected)
}
fn same_bits(actual: &[f32], expected: &[f32]) -> LabResult<()> {
    check_exact(
        &actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        &expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    )
    .map_err(|i| format!("graph replay differs at {i}"))
}
pub fn position(session: &Session, case: &Case) -> LabResult<()> {
    let MultimodalQuery::InterpolatedPositionAddF32 {
        width,
        table_rows,
        capacity,
    } = case.entry.query
    else {
        return Err("expected position query".into());
    };
    let (width, rows) = (width as usize, capacity as usize);
    let table_shape = case.stage["table"]
        .as_array()
        .ok_or("position table shape missing")?;
    let (th, tw) = (
        table_shape[0].as_u64().unwrap() as usize,
        table_shape[1].as_u64().unwrap() as usize,
    );
    assert_eq!(th * tw, table_rows as usize);
    let height = (rows as f64).sqrt() as usize;
    let grid_width = rows.div_ceil(height);
    let axis = |p: usize, size: usize, table: usize| {
        let x = if size == 1 {
            0.0
        } else {
            p as f32 * (table - 1) as f32 / (size - 1) as f32
        };
        let low = (x.floor() as usize).min(table - 1);
        (low, (low + 1).min(table - 1), x - low as f32)
    };
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for r in 0..rows {
        let (h0, h1, dh) = axis(r / grid_width, height, th);
        let (w0, w1, dw) = axis(r % grid_width, grid_width, tw);
        for (h, w, v) in [
            (h0, w0, (1.0 - dh) * (1.0 - dw)),
            (h0, w1, (1.0 - dh) * dw),
            (h1, w0, dh * (1.0 - dw)),
            (h1, w1, dh * dw),
        ] {
            indices.push((h * tw + w) as i32);
            weights.push(v);
        }
    }
    let lab = Lab::new(session, case)?;
    let name = &case.entry.symbol;
    let hidden = seeded_f32(rows * width, 0x51570001, 2.0);
    let table = seeded(th * tw * width, 0x51570002, 0.5);
    let (buffer, t, i, w, live) = (
        lab.f32(&hidden)?,
        lab.bf16(&table)?,
        lab.i32(&indices)?,
        lab.f32(&weights)?,
        lab.count(rows)?,
    );
    let launch = |_: &runtime_cuda::Stream| {
        lab.run(
            name,
            rows as u32,
            &[
                ("input", &buffer),
                ("table", &t),
                ("indices", &i),
                ("weights", &w),
                ("output", &buffer),
                ("live_rows", &live),
            ],
            &[],
        )
    };
    let initial = hidden.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut replay = Replay::new(session).state(
        "position hidden",
        &buffer,
        initial,
        WriteMap::of(DType::F32).none()?,
        std::iter::once(0..rows * width * 4).collect(),
    );
    replay.eager(|| launch(&session.stream))?;
    let actual = lab.read_f32(&buffer, rows * width)?;
    if actual.iter().any(|x| !x.is_finite()) {
        return Err("position: unwritten output".into());
    }
    // Preserve the original grid-boundary probes as well as capacity samples.
    let mut sampled = [
        0,
        1,
        255,
        256,
        4097,
        32_768,
        40_000,
        rows / 4,
        rows / 2,
        rows - 2,
        rows - 1,
    ]
    .into_iter()
    .filter(|&row| row < rows)
    .collect::<Vec<_>>();
    sampled.sort_unstable();
    sampled.dedup();
    eprintln!("{name}: position grid {height}x{grid_width}, sampled rows {sampled:?}");
    let expected = oracle::interpolated_position_add_f32(
        &widen(&rows_of(&hidden, width, &sampled)),
        &decoded(&table),
        width,
        &rows_of(&indices, 4, &sampled),
        &widen(&rows_of(&weights, 4, &sampled)),
    )
    .map_err(|e| e.to_string())?;
    check_f32(name, &rows_of(&actual, width, &sampled), &expected)?;
    buffer
        .copy_from_f32(&hidden, &session.stream)
        .map_err(|e| e.to_string())?;
    let graph = session.capture(launch)?;
    replay.graph(&graph)?;
    same_bits(&lab.read_f32(&buffer, rows * width)?, &actual)?;
    let mut fields = BTreeMap::new();
    replay.record(&mut fields);
    session.receipt("encoder-position-replay", fields)?;
    Ok(())
}
pub fn layout(session: &Session, case: &Case) -> LabResult<()> {
    let MultimodalQuery::EmbeddingOverride { width, .. } = case.entry.query else {
        return Err("expected embedding override".into());
    };
    let width = width as usize;
    let lab = Lab::new(session, case)?;
    let activation = seeded(5 * width, 0x51570008, 1.0);
    let arena = seeded(3 * width, 0x51570009, 1.0);
    let source = [-1_i64, 2, 0, -1, 1];
    let (a, b, s, live) = (
        lab.bf16(&activation)?,
        lab.bf16(&arena)?,
        lab.bytes(
            &source
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?,
        lab.count(5)?,
    );
    lab.run(
        &case.entry.symbol,
        5,
        &[
            ("activation", &a),
            ("arena", &b),
            ("source", &s),
            ("live_rows", &live),
        ],
        &[("arena_rows", 3)],
    )?;
    check(
        &case.entry.symbol,
        &lab.read(&a, 5 * width)?,
        &oracle::embedding_override(&decoded(&activation), &decoded(&arena), width, &source)
            .map_err(|e| e.to_string())?,
    )
}
pub fn attention(session: &Session, case: &Case) -> LabResult<()> {
    let lab = Lab::new(session, case)?;
    let name = case.entry.symbol.as_str();
    let mut misses = Vec::new();
    for (lengths, long, constant) in [
        (vec![1000_usize, 333, 5, 64, 129], false, false),
        (vec![1000, 2333, 5, 64], false, false),
        (vec![4], false, true),
        (
            vec![(case.entry.capacity() as usize).min(64_800)],
            true,
            false,
        ),
    ] {
        let checked = (|| -> LabResult<()> {
            let tokens: usize = lengths.iter().sum();
            let mut cu_seqlens = vec![0_i32];
            for &length in &lengths {
                cu_seqlens.push(cu_seqlens.last().unwrap() + length as i32);
            }
            let cu_buffer = lab.i32(&cu_seqlens)?;
            let live = lab.count(tokens)?;
            let query = case.entry.query;
            let MultimodalQuery::ContiguousAttentionPairs {
                q_heads,
                kv_heads,
                head_dim,
                key0_sink,
                ..
            } = query
            else {
                unreachable!("attention entries carry pair attention queries")
            };
            let (q_width, kv_width) = (
                (q_heads * head_dim) as usize,
                (kv_heads * head_dim) as usize,
            );
            let mut q = seeded_f32(tokens * q_width, 0x7740 ^ q_width as u64, 2.0);
            let mut k = seeded_f32(tokens * kv_width, 0x7741 ^ kv_width as u64, 2.0);
            let mut v = seeded_f32(tokens * kv_width, 0x7742 ^ kv_width as u64, 2.0);
            let sinks = seeded_f32(q_heads as usize, 0x7743, 3.0);
            let scale = (head_dim as f32).powf(-0.5);
            if constant {
                q.fill(0.0);
                k.fill(0.0);
                v.fill(1.0 + 1.0 / 256.0);
                for t in 0..tokens {
                    for h in 0..q_heads as usize {
                        q[t * q_width + h * head_dim as usize] = 1.0;
                    }
                    for h in 0..kv_heads as usize {
                        k[t * kv_width + h * head_dim as usize] =
                            if t == 0 { 0.0 } else { -1.0 / 64.0 };
                    }
                }
            }
            let qkv = lab.bf16(&packed_pairs(&q, &k, &v, q_width, kv_width))?;
            let sink_buffer = lab.f32(&sinks)?;
            let output = lab.output(tokens * q_width, 4)?;
            let mut buffers = vec![
                ("qkv", &qkv),
                ("cu_seqlens", &cu_buffer),
                ("output", &output),
                ("live_tokens", &live),
            ];
            if key0_sink {
                buffers.push(("sinks", &sink_buffer));
            }
            let launch = |_: &runtime_cuda::Stream| {
                lab.run(
                    name,
                    tokens as u32,
                    &buffers,
                    &[
                        ("live_sequences", lengths.len() as u64),
                        ("scale_bits", u64::from(scale.to_bits())),
                    ],
                )
            };
            let mut replay = Replay::new(session).output(
                "encoder attention",
                &output,
                WriteMap::of(DType::F32).prefix(tokens * q_width)?,
            );
            replay.eager(|| launch(&session.stream))?;
            let eager = lab.read_f32(&output, tokens * q_width)?;
            let sink_values = key0_sink.then(|| widen(&sinks));
            if eager.iter().any(|x| !x.is_finite()) {
                return Err(format!("{name}: unwritten attention output"));
            }
            let rows = if long {
                (0..128)
                    .map(|i| i * (tokens - 1) / 127)
                    .chain([1, 127, 128, 32_767, 32_768])
                    .filter(|&r| r < tokens)
                    .collect::<Vec<_>>()
            } else {
                (0..tokens).collect::<Vec<_>>()
            };
            eprintln!(
                "{name}: attention lengths {lengths:?}, {} oracle rows",
                rows.len()
            );
            if long {
                eprintln!("{name}: long attention sampled rows {rows:?}");
            }
            let actual = rows_of(&eager, q_width, &rows);
            let reference = |q: &[f64], k: &[f64], v: &[f64]| {
                concat(parallel(rows.len(), |range| {
                    let selected = &rows[range];
                    vec![
                        oracle::contiguous_attention_rows(
                            query,
                            selected,
                            &rows_of(q, q_width, selected),
                            k,
                            v,
                            &cu_seqlens,
                            f64::from(scale),
                            sink_values.as_deref(),
                        )
                        .expect("attention oracle"),
                    ]
                }))
            };
            let expected = reference(&pair_values(&q), &pair_values(&k), &pair_values(&v));
            check_f32(name, &actual, &expected)?;
            let rounded = |x: &[f32]| {
                x.iter()
                    .map(|&x| f64::from(bf16_to_f32(f32_to_bf16(x))))
                    .collect::<Vec<_>>()
            };
            if !constant {
                let bf16 = reference(&rounded(&q), &rounded(&k), &rounded(&v));
                let (to_pairs, to_bf16) = (
                    distance(&actual, &expected.values),
                    distance(&actual, &bf16.values),
                );
                eprintln!("{name}: L2 distance {to_pairs:.3e} to pairs, {to_bf16:.3e} to BF16");
                if to_pairs * 16.0 > to_bf16 {
                    return Err(format!("{name}: not closer to the pairs than to BF16"));
                }
            }
            let graph = session.capture(launch)?;
            replay.graph(&graph)?;
            same_bits(&lab.read_f32(&output, tokens * q_width)?, &eager)?;
            let mut fields = BTreeMap::new();
            replay.record(&mut fields);
            session.receipt("encoder-attention-replay", fields)?;
            Ok(())
        })();
        if let Err(error) = checked {
            eprintln!("{name} lengths={lengths:?}: {error}");
            misses.push(error);
        }
    }
    if misses.is_empty() {
        Ok(())
    } else {
        Err(misses.join("; "))
    }
}
