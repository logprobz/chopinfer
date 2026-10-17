//! The latent attention engine's (`kernels_sm120::latent`) checks against
//! the kernel-core oracles (`latent_attention_oracle`) at the lab's served
//! shape (`kernel_lab::latent_shapes`), over one checked cubin
//! ([`Kernels`]): the lab's latent cubin, or a bundle's embedded cubin of
//! the same entries under the bundle's symbol names. `tests/latent_attention.rs`
//! lists what they cover.
#![allow(dead_code, reason = "each test binary runs a subset of the checks")]

use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernel_core::latent_attention_oracle::{
    LatentAttentionInput, LatentAttentionResult, LatentPaging, LatentPartials, head_batched_gemm,
    latent_attention, latent_kv_write,
};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_lab::latent_shapes::{self, GEOMETRY};
use kernel_lab::poison::{CopyProducer, POISON, Replay, WriteMap};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::latent::{Entry, Query};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module, Stream};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::rc::Rc;

const HEADS: usize = GEOMETRY.heads as usize;
const LATENT: usize = GEOMETRY.latent as usize;
const PAGE: usize = GEOMETRY.page_size as usize;
const MAX_KEYS: usize = GEOMETRY.max_keys as usize;
const POOL: usize = 4;
const TOP_POOLS: usize = 512;
/// Bytes between consecutive pages that no kernel may touch.
const PAGE_GAP: usize = 256;
/// Rows of one KV write launch; the write entry's capacity.
const WRITE_ROWS: usize = 8192;
/// Rows each attention case checks against the oracle.
const SAMPLED_ROWS: usize = 16;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1_u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
    fn bf16(&mut self, len: usize, scale: f64) -> Vec<u16> {
        (0..len)
            .map(|_| f32_to_bf16((self.normal() * scale) as f32))
            .collect()
    }
    /// Values spanning 2^-8..2^8 in magnitude with random signs, so long
    /// sums cancel far below their sum of magnitudes.
    fn cancelling(&mut self, len: usize) -> Vec<u16> {
        (0..len)
            .map(|_| {
                let magnitude = 2_f64.powf(16.0 * self.uniform() - 8.0);
                let sign = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
                f32_to_bf16((sign * magnitude) as f32)
            })
            .collect()
    }
}

fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&x| f64::from(bf16_to_f32(x))).collect()
}
fn bytes_of<T: Copy, const N: usize>(values: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| to(v)).collect()
}
fn u16_bytes(values: &[u16]) -> Vec<u8> {
    bytes_of(values, u16::to_ne_bytes)
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    bytes_of(values, i32::to_ne_bytes)
}
fn as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}
fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(16))?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}
fn zeros(session: &Session, bytes: usize) -> LabResult<DeviceBuffer> {
    upload(session, &vec![0; bytes.max(16)])
}
fn download(session: &Session, buffer: &DeviceBuffer, bytes: usize) -> LabResult<Vec<u8>> {
    let mut out = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut out, &session.stream)
        .map_err(|e| e.to_string())?;
    out.truncate(bytes);
    Ok(out)
}

fn dtype_name(kv: LatentKvFormat) -> &'static str {
    match kv {
        LatentKvFormat::Bf16 => "bf16",
        LatentKvFormat::Fp8E4M3Block128 => "fp8",
        LatentKvFormat::Fp8E4M3Ue8M0Block32 | LatentKvFormat::Fp4E2M1E4M3Block16 => {
            unreachable!("native latent formats need their own conformance cases")
        }
    }
}

/// A checked cubin, its entries and the producer's module: the lab's
/// latent cubin with its catalog, or a bundle's embedded latent cubin with
/// that bundle's entries, whose symbols carry its own profile tag. The
/// checks find entries by kernel name (`{tag}_{kernel}`), which the two
/// share.
pub struct Kernels {
    /// The checked cubin's name, recorded in every receipt.
    cubin: String,
    module: Module,
    entries: Vec<Entry>,
    /// The lab cubin, for the PDL producer, when another cubin is checked.
    lab: Option<Module>,
    /// The symbols launched so far, main kernels and merges.
    launched: Rc<RefCell<BTreeSet<String>>>,
}

impl Kernels {
    /// The lab's latent cubin with every catalog entry and the producer.
    pub fn lab(session: &Session) -> LabResult<Self> {
        Ok(Self {
            cubin: latent_shapes::cubin_name(),
            module: session.module(&latent_shapes::cubin_name())?,
            entries: latent_shapes::catalog(),
            lab: None,
            launched: Rc::default(),
        })
    }

    /// Another cubin, `name` with bytes `cubin`, holding `entries` (each
    /// with its merge); the producer comes from the lab cubin.
    pub fn embedded(
        session: &Session,
        name: &str,
        cubin: &[u8],
        entries: Vec<Entry>,
    ) -> LabResult<Self> {
        Ok(Self {
            cubin: name.to_owned(),
            module: session.load_cubin(cubin)?,
            entries,
            lab: Some(session.module(&latent_shapes::cubin_name())?),
            launched: Rc::default(),
        })
    }

    fn find(&self, suffix: &str) -> LabResult<Entry> {
        self.entries
            .iter()
            .find(|entry| entry.name.ends_with(suffix))
            .cloned()
            .ok_or_else(|| format!("{}: no latent entry {suffix}", self.cubin))
    }

    /// Every symbol of the entries, main kernels and merges, that no check
    /// launched.
    pub fn unlaunched(&self) -> Vec<String> {
        let launched = self.launched.borrow();
        self.entries
            .iter()
            .flat_map(|entry| std::iter::once(entry.symbol()).chain(entry.merge_symbol()))
            .filter(|symbol| !launched.contains(symbol))
            .collect()
    }
}

/// An entry and its loaded kernels.
struct Kernel {
    entry: Entry,
    main: Function,
    merge: Option<Function>,
    launched: Rc<RefCell<BTreeSet<String>>>,
}

impl Kernel {
    fn load(kernels: &Kernels, suffix: &str) -> LabResult<Self> {
        let entry = kernels.find(suffix)?;
        let module = &kernels.module;
        let main = module
            .function(&entry.symbol())
            .map_err(|e| e.to_string())?;
        let shared = entry.dynamic_shared_bytes();
        if shared > 0 {
            main.set_max_dynamic_shared_memory(shared)
                .map_err(|e| e.to_string())?;
        }
        let merge = entry
            .merge_symbol()
            .map(|symbol| module.function(&symbol).map_err(|e| e.to_string()))
            .transpose()?;
        Ok(Self {
            entry,
            main,
            merge,
            launched: Rc::clone(&kernels.launched),
        })
    }

    fn record(&self, symbol: String) {
        self.launched.borrow_mut().insert(symbol);
    }

    /// Launches the main kernel and, when the bucket splits, the merge, each
    /// with `pdl`, binding parameters with [`bind`].
    fn launch(
        &self,
        stream: &Stream,
        prepared: u32,
        buffers: &BTreeMap<&str, &DeviceBuffer>,
        scalars: &BTreeMap<&str, u64>,
        pdl: bool,
    ) -> LabResult<()> {
        let args = |params: &[EntryParam]| bind(params, buffers, scalars);
        let mut plan = self.entry.plan(prepared).map_err(|e| e.0.to_string())?;
        plan.pdl = pdl;
        self.main
            .launch(stream, launch::config(plan)?, &args(&self.entry.params())?)
            .map_err(|e| e.to_string())?;
        self.record(self.entry.symbol());
        if let (Some(merge), Some(mut plan)) = (
            &self.merge,
            self.entry
                .merge_plan(prepared)
                .map_err(|e| e.0.to_string())?,
        ) {
            plan.pdl = pdl;
            merge
                .launch(
                    stream,
                    launch::config(plan)?,
                    &args(&self.entry.merge_params())?,
                )
                .map_err(|e| e.to_string())?;
            self.record(self.entry.merge_symbol().ok_or("a merge has a symbol")?);
        }
        Ok(())
    }
}

/// Kernel arguments for `params`: pointers by name from `buffers` (a
/// missing output binds null), scalars from `scalars`.
fn bind<'a>(
    params: &[EntryParam],
    buffers: &BTreeMap<&str, &'a DeviceBuffer>,
    scalars: &BTreeMap<&str, u64>,
) -> LabResult<Vec<KernelArgument<'a>>> {
    params
        .iter()
        .map(|param| match param {
            EntryParam::DevicePointer { name, .. } => Ok(buffers
                .get(name.as_str())
                .map_or(KernelArgument::NullBuffer, |&b| KernelArgument::Buffer(b))),
            EntryParam::U32 { name } => scalars
                .get(name.as_str())
                .map(|&v| KernelArgument::U32(v as u32))
                .ok_or_else(|| format!("missing scalar {name}")),
            EntryParam::U64 { name } => scalars
                .get(name.as_str())
                .map(|&v| KernelArgument::U64(v))
                .ok_or_else(|| format!("missing scalar {name}")),
            other => Err(format!("unexpected parameter {other:?}")),
        })
        .collect()
}

// ---------------------------------------------------------------- cache ----

/// Sequences of the given lengths written through the KV write kernel into
/// pages scattered over one pool, with `PAGE_GAP` bytes between pages.
struct Cache {
    kv: LatentKvFormat,
    latent: Vec<Vec<u16>>,
    tables: Vec<i32>,
    max_pages: usize,
    page_stride: usize,
    pool: Vec<u8>,
    d_pool: DeviceBuffer,
    d_tables: DeviceBuffer,
}

impl Cache {
    fn paging(&self) -> LatentPaging<'_> {
        LatentPaging {
            block_tables: &self.tables,
            max_pages: self.max_pages,
            page_stride: self.page_stride,
        }
    }

    fn write_query(kv: LatentKvFormat) -> LatentAttentionQuery {
        LatentAttentionQuery::KvWrite {
            latent: GEOMETRY.latent,
            page_size: GEOMETRY.page_size,
            kv,
            capacity: WRITE_ROWS as u32,
        }
    }

    /// Writes every token with the kernel, then checks the pool bit for bit
    /// against the oracle's write of the same rows.
    fn new(
        session: &Session,
        kernels: &Kernels,
        kv: LatentKvFormat,
        lengths: &[usize],
        rng: &mut Rng,
        mut latent_of: impl FnMut(&mut Rng, usize) -> Vec<u16>,
    ) -> LabResult<Self> {
        let max_pages = lengths.iter().map(|l| l.div_ceil(PAGE)).max().unwrap_or(1);
        let pages = lengths.iter().map(|l| l.div_ceil(PAGE)).sum::<usize>();
        // Page ids in a shuffled order, so neighbors in a table are not
        // neighbors in the pool.
        let mut ids = (0..pages as i32).collect::<Vec<_>>();
        for i in (1..ids.len()).rev() {
            ids.swap(i, rng.below(i + 1));
        }
        let mut tables = vec![-1; lengths.len() * max_pages];
        let mut next = 0;
        for (s, &len) in lengths.iter().enumerate() {
            for page in 0..len.div_ceil(PAGE) {
                tables[s * max_pages + page] = ids[next];
                next += 1;
            }
        }
        let page_stride = kv.page_bytes(GEOMETRY.latent, GEOMETRY.page_size) as usize + PAGE_GAP;
        let initial = vec![0x5a_u8; pages * page_stride];
        let latent = lengths
            .iter()
            .map(|&len| latent_of(rng, len * LATENT))
            .collect::<Vec<_>>();
        let d_pool = upload(session, &initial)?;
        let d_tables = upload(session, &i32_bytes(&tables))?;
        let mut cache = Cache {
            kv,
            latent,
            tables,
            max_pages,
            page_stride,
            pool: initial,
            d_pool,
            d_tables,
        };
        let write = Kernel::load(kernels, &format!("kv_write_{}", dtype_name(kv)))?;
        for (s, &len) in lengths.iter().enumerate() {
            for start in (0..len).step_by(WRITE_ROWS) {
                let rows = (len - start).min(WRITE_ROWS);
                let positions = (start as i32..(start + rows) as i32).collect::<Vec<_>>();
                let batch = vec![s as i32; rows];
                let bits = cache.latent[s][start * LATENT..(start + rows) * LATENT].to_vec();
                let d_latent = upload(session, &u16_bytes(&bits))?;
                let d_batch = upload(session, &i32_bytes(&batch))?;
                let d_positions = upload(session, &i32_bytes(&positions))?;
                let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
                let buffers = BTreeMap::from([
                    ("latent", &d_latent),
                    ("row_batch", &d_batch),
                    ("positions", &d_positions),
                    ("live_rows", &d_live),
                    ("block_tables", &cache.d_tables),
                    ("pool", &cache.d_pool),
                ]);
                let scalars = BTreeMap::from([
                    ("max_pages", cache.max_pages as u64),
                    ("page_stride_bytes", cache.page_stride as u64),
                ]);
                write.launch(&session.stream, rows as u32, &buffers, &scalars, false)?;
                let paging = LatentPaging {
                    block_tables: &cache.tables,
                    max_pages: cache.max_pages,
                    page_stride: cache.page_stride,
                };
                latent_kv_write(
                    Self::write_query(kv),
                    &bits,
                    &batch,
                    &positions,
                    paging,
                    &mut cache.pool,
                )
                .map_err(|e| e.0.to_string())?;
            }
        }
        let written = download(session, &cache.d_pool, cache.pool.len())?;
        if let Some(offset) = written.iter().zip(&cache.pool).position(|(a, b)| a != b) {
            return Err(format!(
                "{} KV write: pool byte {offset} is {:#04x}, the oracle wrote {:#04x}",
                dtype_name(kv),
                written[offset],
                cache.pool[offset]
            ));
        }
        Ok(cache)
    }
}

// ----------------------------------------------------------- key lists ----

/// The key list of a row at `position`: every complete pool when there are
/// at most 512 of them, otherwise 512 distinct pools chosen at random (in
/// ascending order), each as its four positions, then the incomplete tail
/// (the row's own position unless it completes a pool), then -1. With
/// `holes`, about one listed entry in nine becomes -1.
fn key_list(rng: &mut Rng, position: usize, holes: bool) -> Vec<i32> {
    let visible = position + 1;
    let complete = visible / POOL;
    let pools: Vec<usize> = if complete <= TOP_POOLS {
        (0..complete).collect()
    } else {
        let mut chosen = (0..complete).collect::<Vec<_>>();
        for i in 0..TOP_POOLS {
            let j = i + rng.below(complete - i);
            chosen.swap(i, j);
        }
        chosen.truncate(TOP_POOLS);
        chosen.sort_unstable();
        chosen
    };
    let mut list = pools
        .iter()
        .flat_map(|&pool| pool * POOL..(pool + 1) * POOL)
        .chain(complete * POOL..visible)
        .map(|p| p as i32)
        .collect::<Vec<_>>();
    if holes {
        for entry in &mut list {
            if rng.below(9) == 0 {
                *entry = -1;
            }
        }
    }
    list.resize(MAX_KEYS, -1);
    list
}

/// One attention launch's rows: `rows_per_item` consecutive positions
/// ending at each sequence's last token, in sequence order. With holes and
/// at least two rows, the middle row's list is all holes.
struct Step {
    rows: usize,
    /// The all-hole row.
    empty: Option<usize>,
    row_batch: Vec<i32>,
    positions: Vec<usize>,
    lists: Vec<i32>,
    q: Vec<u16>,
}

impl Step {
    fn new(rng: &mut Rng, lengths: &[usize], rows_per_item: usize, holes: bool) -> Self {
        let mut step = Step {
            rows: 0,
            empty: None,
            row_batch: Vec::new(),
            positions: Vec::new(),
            lists: Vec::new(),
            q: Vec::new(),
        };
        for (s, &len) in lengths.iter().enumerate() {
            for position in len.saturating_sub(rows_per_item)..len {
                step.rows += 1;
                step.row_batch.push(s as i32);
                step.positions.push(position);
                step.lists.extend(key_list(rng, position, holes));
            }
        }
        if holes && step.rows >= 2 {
            let row = step.rows / 2;
            step.lists[row * MAX_KEYS..(row + 1) * MAX_KEYS].fill(-1);
            step.empty = Some(row);
        }
        step.q = rng.bf16(step.rows * HEADS * LATENT, 1.0);
        step
    }

    /// Up to `SAMPLED_ROWS` rows: the first, the last, the all-hole row,
    /// the rows at positions 2050 to 2052 when present, then evenly spaced
    /// ones.
    fn sampled(&self) -> Vec<usize> {
        let mut rows = vec![0, self.rows - 1];
        rows.extend(self.empty);
        for (row, &p) in self.positions.iter().enumerate() {
            if (2050..=2052).contains(&p) && !rows.contains(&row) {
                rows.push(row);
            }
        }
        let stride = (self.rows / SAMPLED_ROWS).max(1);
        for row in (0..self.rows).step_by(stride) {
            if rows.len() < SAMPLED_ROWS && !rows.contains(&row) {
                rows.push(row);
            }
        }
        rows.sort_unstable();
        rows.dedup();
        rows
    }

    /// The step restricted to `rows`, for the oracle.
    fn subset(&self, rows: &[usize]) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
        let width = HEADS * LATENT;
        let q = rows
            .iter()
            .flat_map(|&r| decode(&self.q[r * width..(r + 1) * width]))
            .collect();
        let batch = rows.iter().map(|&r| self.row_batch[r]).collect();
        let lists = rows
            .iter()
            .flat_map(|&r| self.lists[r * MAX_KEYS..(r + 1) * MAX_KEYS].to_vec())
            .collect();
        (q, batch, lists)
    }
}

// -------------------------------------------------------------- report ----

/// One receipt per case holding every check's verdict and error summary.
pub struct Report {
    cubin: String,
    case: String,
    checks: Vec<Value>,
    failures: Vec<String>,
}

impl Report {
    pub fn new(kernels: &Kernels, case: impl Into<String>) -> Self {
        Self {
            cubin: kernels.cubin.clone(),
            case: case.into(),
            checks: Vec::new(),
            failures: Vec::new(),
        }
    }

    /// The decision-0081 gate against `reference`, with the worst element's
    /// share of its bound.
    fn gate(&mut self, name: &str, actual: &[f64], reference: &OracleOutput, dtype: DType) {
        let result = check_numeric(actual, reference, dtype);
        let bound_use = actual
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let bound = kernel_core::oracle::numeric_bound(reference, i, dtype);
                let error = (a - reference.values[i]).abs();
                if error == 0.0 { 0.0 } else { error / bound }
            })
            .fold(0.0, f64::max);
        self.checks.push(json!({
            "name": name,
            "correct": result.is_ok(),
            "elements": actual.len(),
            "max_error_over_bound": bound_use,
            "worst": result.as_ref().err().map(|w| json!({
                "index": w.index, "actual": w.actual, "reference": w.reference,
                "bound": w.bound, "error": w.error,
            })),
        }));
        if let Err(worst) = result {
            self.failures.push(format!("{name}: {worst:?}"));
        }
    }

    fn exact(&mut self, name: &str, result: Result<(), String>) {
        self.checks.push(
            json!({"name": name, "correct": result.is_ok(), "detail": result.as_ref().err()}),
        );
        if let Err(error) = result {
            self.failures.push(format!("{name}: {error}"));
        }
    }

    pub fn finish(self, session: &Session, extra: BTreeMap<String, Value>) -> LabResult<()> {
        let mut data = extra;
        data.insert("cubin".into(), json!(self.cubin));
        data.insert("case".into(), json!(self.case));
        data.insert("gate".into(), json!("G1"));
        data.insert("correct".into(), json!(self.failures.is_empty()));
        data.insert("checks".into(), json!(self.checks));
        session.receipt("latent_attention", data)?;
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(format!("{}: {}", self.case, self.failures.join("; ")))
        }
    }
}

/// The kernel's partial `[rows, heads]` maxima and denominators and
/// `[rows, heads, latent]` numerators.
struct Partial {
    maximum: Vec<f32>,
    denominator: Vec<f32>,
    numerator: Vec<f32>,
}

impl Partial {
    fn rows(&self, rows: &[usize]) -> Partial {
        let pick = |values: &[f32], width: usize| {
            rows.iter()
                .flat_map(|&r| values[r * width..(r + 1) * width].to_vec())
                .collect()
        };
        Partial {
            maximum: pick(&self.maximum, HEADS),
            denominator: pick(&self.denominator, HEADS),
            numerator: pick(&self.numerator, HEADS * LATENT),
        }
    }
}

/// Every head of `row` holds exactly the empty partial: maximum -inf,
/// denominator +0 and a +0 numerator.
fn empty_partial(partial: &Partial, row: usize) -> Result<(), String> {
    for pair in row * HEADS..(row + 1) * HEADS {
        let numerator = &partial.numerator[pair * LATENT..(pair + 1) * LATENT];
        if partial.maximum[pair].to_bits() != f32::NEG_INFINITY.to_bits()
            || partial.denominator[pair].to_bits() != 0
            || numerator.iter().any(|x| x.to_bits() != 0)
        {
            return Err(format!(
                "row {row} pair {pair}: maximum {:#010x}, denominator {:#010x}",
                partial.maximum[pair].to_bits(),
                partial.denominator[pair].to_bits()
            ));
        }
    }
    Ok(())
}

/// Every element of `row` of a normalized output is +0.
fn zero_row(out: &[u16], row: usize) -> Result<(), String> {
    match out[row * HEADS * LATENT..(row + 1) * HEADS * LATENT]
        .iter()
        .position(|&x| x != 0)
    {
        None => Ok(()),
        Some(i) => Err(format!("row {row} element {i} is not +0")),
    }
}

/// Gates a kernel partial against the oracle's: the maximum within one
/// score's bound, then the denominator and numerator against the oracle
/// rescaled to the kernel's maximum. Empty lists must give exactly -inf, 0
/// and 0.
fn gate_partial(report: &mut Report, name: &str, actual: &Partial, reference: &LatentPartials) {
    let pairs = reference.maximum.values.len();
    let mut empty = Ok(());
    let mut maximum = (Vec::new(), OracleOutput::new(vec![], vec![], 1).unwrap());
    let mut denominator = maximum.clone();
    let mut numerator = maximum.clone();
    let push = |into: &mut (Vec<f64>, OracleOutput), actual: f64, r: &OracleOutput, i, f: f64| {
        into.0.push(actual);
        into.1.values.push(r.values[i] * f);
        into.1.sum_abs_products.push(r.sum_abs_products[i] * f);
        into.1.reduction_lengths.push(r.reduction_lengths[i]);
    };
    for pair in 0..pairs {
        let m_ref = reference.maximum.values[pair];
        let m = f64::from(actual.maximum[pair]);
        if m_ref == f64::NEG_INFINITY {
            let zeros = actual.numerator[pair * LATENT..(pair + 1) * LATENT]
                .iter()
                .all(|&x| x.to_bits() == 0);
            if m != f64::NEG_INFINITY || actual.denominator[pair].to_bits() != 0 || !zeros {
                empty = Err(format!("pair {pair} has no keys but a nonempty partial"));
            }
            continue;
        }
        push(&mut maximum, m, &reference.maximum, pair, 1.0);
        let factor = (m_ref - m).exp();
        push(
            &mut denominator,
            f64::from(actual.denominator[pair]),
            &reference.denominator,
            pair,
            factor,
        );
        for d in pair * LATENT..(pair + 1) * LATENT {
            push(
                &mut numerator,
                f64::from(actual.numerator[d]),
                &reference.numerator,
                d,
                factor,
            );
        }
    }
    report.exact(&format!("{name} empty partials"), empty);
    report.gate(
        &format!("{name} maximum"),
        &maximum.0,
        &maximum.1,
        DType::F32,
    );
    report.gate(
        &format!("{name} denominator"),
        &denominator.0,
        &denominator.1,
        DType::F32,
    );
    report.gate(
        &format!("{name} numerator"),
        &numerator.0,
        &numerator.1,
        DType::F32,
    );
}

// ----------------------------------------------------------- attention ----

/// Device operands of one attention step.
struct StepBuffers {
    q: DeviceBuffer,
    lists: DeviceBuffer,
    row_batch: DeviceBuffer,
    live: DeviceBuffer,
    workspace: DeviceBuffer,
    out: DeviceBuffer,
    out_max: DeviceBuffer,
    out_den: DeviceBuffer,
    out_num: DeviceBuffer,
}

impl StepBuffers {
    /// Outputs hold `prepared` rows filled with [`POISON`], so a write past
    /// the live rows or a missed write shows.
    fn new(
        session: &Session,
        step: &Step,
        lists: &[i32],
        workspace: u64,
        prepared: u32,
    ) -> LabResult<Self> {
        let pairs = prepared as usize * HEADS;
        let poisoned = |bytes: usize| upload(session, &vec![POISON; bytes]);
        Ok(Self {
            q: upload(session, &u16_bytes(&step.q))?,
            lists: upload(session, &i32_bytes(lists))?,
            row_batch: upload(session, &i32_bytes(&step.row_batch))?,
            live: upload(session, &(step.rows as u32).to_ne_bytes())?,
            workspace: zeros(session, workspace as usize)?,
            out: poisoned(pairs * LATENT * 2)?,
            out_max: poisoned(pairs * 4)?,
            out_den: poisoned(pairs * 4)?,
            out_num: poisoned(pairs * LATENT * 4)?,
        })
    }

    /// Every output byte past the `live` rows still holds its poison.
    fn tails(&self, session: &Session, live: usize) -> Result<(), String> {
        for (name, buffer, row_bytes) in [
            ("out", &self.out, HEADS * LATENT * 2),
            ("out_max", &self.out_max, HEADS * 4),
            ("out_den", &self.out_den, HEADS * 4),
            ("out_num", &self.out_num, HEADS * LATENT * 4),
        ] {
            let bytes = download(session, buffer, buffer.byte_len())?;
            if let Some(offset) = bytes[live * row_bytes..].iter().position(|&b| b != POISON) {
                return Err(format!(
                    "{name}: byte {} past the live rows was written",
                    live * row_bytes + offset
                ));
            }
        }
        Ok(())
    }

    fn launch(
        &self,
        stream: &Stream,
        kernel: &Kernel,
        cache: &Cache,
        prepared: u32,
        pdl: bool,
    ) -> LabResult<()> {
        let buffers = BTreeMap::from([
            ("q", &self.q),
            ("pool", &cache.d_pool),
            ("block_tables", &cache.d_tables),
            ("row_batch", &self.row_batch),
            ("key_lists", &self.lists),
            ("live_rows", &self.live),
            ("workspace", &self.workspace),
            ("out", &self.out),
            ("out_max", &self.out_max),
            ("out_den", &self.out_den),
            ("out_num", &self.out_num),
        ]);
        let scalars = BTreeMap::from([
            ("max_pages", cache.max_pages as u64),
            ("page_stride_bytes", cache.page_stride as u64),
            ("budget", MAX_KEYS as u64),
            ("splits", u64::from(kernel.entry.splits())),
            ("scale_bits", u64::from(GEOMETRY.softmax_scale.to_bits())),
        ]);
        kernel.launch(stream, prepared, &buffers, &scalars, pdl)
    }

    fn normalized(&self, session: &Session, rows: usize) -> LabResult<Vec<u16>> {
        Ok(as_u16(&download(
            session,
            &self.out,
            rows * HEADS * LATENT * 2,
        )?))
    }

    fn partial(&self, session: &Session, rows: usize) -> LabResult<Partial> {
        let pairs = rows * HEADS;
        Ok(Partial {
            maximum: as_f32(&download(session, &self.out_max, pairs * 4)?),
            denominator: as_f32(&download(session, &self.out_den, pairs * 4)?),
            numerator: as_f32(&download(session, &self.out_num, pairs * LATENT * 4)?),
        })
    }
}

fn attention_query(kernel: &Kernel) -> LatentAttentionQuery {
    match kernel.entry.query {
        Query::Latent(query) => query,
        Query::HeadGemm(_) => unreachable!("an attention entry"),
    }
}

/// The oracle over `rows` of `step` with key lists `lists`.
fn oracle(
    kernel: &Kernel,
    cache: &Cache,
    step: &Step,
    lists: &[i32],
    rows: &[usize],
) -> LabResult<LatentAttentionResult> {
    let (q, batch, _) = step.subset(rows);
    let lists = rows
        .iter()
        .flat_map(|&r| lists[r * MAX_KEYS..(r + 1) * MAX_KEYS].to_vec())
        .collect::<Vec<_>>();
    latent_attention(
        attention_query(kernel),
        LatentAttentionInput {
            q: &q,
            row_batch: &batch,
            key_lists: &lists,
            budget: MAX_KEYS,
            paging: cache.paging(),
            pool: &cache.pool,
        },
    )
    .map_err(|e| e.0.to_string())
}

/// Elements `rows` of a row-major `[rows, width]` vector.
fn pick<T: Copy>(values: &[T], rows: &[usize], width: usize) -> Vec<T> {
    rows.iter()
        .flat_map(|&r| values[r * width..(r + 1) * width].to_vec())
        .collect()
}

struct Case {
    name: String,
    kv: LatentKvFormat,
    phase: &'static str,
    lengths: Vec<usize>,
    rows_per_item: usize,
    /// The graph bucket the launch is planned for, at least the live rows.
    prepared: Option<u32>,
}

/// Runs both output modes of one case against the oracle, then merges
/// partials over 2 and 4 fake context-parallel ranks.
fn attention_case(session: &Session, kernels: &Kernels, case: &Case, seed: u64) -> LabResult<()> {
    let mut rng = Rng::new(seed);
    let started = std::time::Instant::now();
    let cache = Cache::new(
        session,
        kernels,
        case.kv,
        &case.lengths,
        &mut rng,
        |rng: &mut Rng, n| rng.bf16(n, 1.0),
    )?;
    let step = Step::new(&mut rng, &case.lengths, case.rows_per_item, true);
    let prepared = case.prepared.unwrap_or(step.rows as u32);
    let dtype = dtype_name(case.kv);
    let normalized = Kernel::load(kernels, &format!("attn_{dtype}_{}", case.phase))?;
    let partial = Kernel::load(kernels, &format!("attn_{dtype}_{}_partial", case.phase))?;
    let workspace = normalized.entry.workspace_bytes(prepared);
    let buffers = StepBuffers::new(session, &step, &step.lists, workspace, prepared)?;
    let mut report = Report::new(kernels, case.name.clone());
    let sampled = step.sampled();

    buffers.launch(&session.stream, &normalized, &cache, prepared, false)?;
    buffers.launch(&session.stream, &partial, &cache, prepared, false)?;
    let out = buffers.normalized(session, step.rows)?;
    let parts = buffers.partial(session, step.rows)?;
    report.exact(
        "prepared rows past the live count stay poisoned",
        buffers.tails(session, step.rows),
    );
    if let Some(row) = step.empty {
        // An all-hole row, written over poison: +0 output and the exact
        // empty partial, through the split merge when the bucket splits.
        report.exact("all-hole row normalized", zero_row(&out, row));
        report.exact("all-hole row partial", empty_partial(&parts, row));
    }
    let LatentAttentionResult::Normalized(reference) =
        oracle(&normalized, &cache, &step, &step.lists, &sampled)?
    else {
        unreachable!()
    };
    report.gate(
        "normalized",
        &decode(&pick(&out, &sampled, HEADS * LATENT)),
        &reference,
        DType::Bf16,
    );
    let LatentAttentionResult::Partial(reference_parts) =
        oracle(&partial, &cache, &step, &step.lists, &sampled)?
    else {
        unreachable!()
    };
    gate_partial(
        &mut report,
        "partial",
        &parts.rows(&sampled),
        &reference_parts,
    );

    // One rank: the contract's merge of the kernel's own partial is the
    // normalized output bit for bit.
    let single = merge_f32(&[&parts], step.rows * HEADS);
    report.exact(
        "one-rank merge is the normalized output",
        match single.iter().zip(&out).position(|(a, b)| a != b) {
            None => Ok(()),
            Some(i) => Err(format!(
                "element {i}: merge {:#06x}, normalized {:#06x}",
                single[i], out[i]
            )),
        },
    );

    // Two and four fake ranks own whole pages; each attends to its own keys
    // (the others are holes), and the merged result passes the same gate.
    for ranks in [2, 4] {
        let mut rank_parts = Vec::new();
        for rank in 0..ranks {
            let lists = step
                .lists
                .iter()
                .map(|&p| {
                    if p >= 0 && (p as usize / PAGE) % ranks == rank {
                        p
                    } else {
                        -1
                    }
                })
                .collect::<Vec<_>>();
            let rank_buffers = StepBuffers::new(session, &step, &lists, workspace, prepared)?;
            rank_buffers.launch(&session.stream, &partial, &cache, prepared, false)?;
            let rank_part = rank_buffers.partial(session, step.rows)?;
            // A rank that owns none of a row's keys writes the exact empty
            // partial over poison.
            let empty_rows = (0..step.rows)
                .filter(|&r| {
                    lists[r * MAX_KEYS..(r + 1) * MAX_KEYS]
                        .iter()
                        .all(|&p| p < 0)
                })
                .collect::<Vec<_>>();
            report.exact(
                &format!("{ranks}-rank rank {rank} empty partials"),
                empty_rows
                    .iter()
                    .try_for_each(|&r| empty_partial(&rank_part, r)),
            );
            report.checks.push(json!({
                "name": format!("{ranks}-rank rank {rank} empty rows"),
                "rows": empty_rows.len(),
            }));
            report.exact(
                &format!("{ranks}-rank rank {rank} tails"),
                rank_buffers.tails(session, step.rows),
            );
            rank_parts.push(rank_part);
        }
        let merged = merge_f32(&rank_parts.iter().collect::<Vec<_>>(), step.rows * HEADS);
        if let Some(row) = step.empty {
            report.exact(
                &format!("{ranks}-rank all-empty merge"),
                zero_row(&merged, row),
            );
        }
        report.gate(
            &format!("{ranks}-rank merge"),
            &decode(&pick(&merged, &sampled, HEADS * LATENT)),
            &reference,
            DType::Bf16,
        );
        let differing = merged.iter().zip(&out).filter(|(a, b)| a != b).count();
        report.checks.push(json!({
            "name": format!("{ranks}-rank merge vs one rank"),
            "differing_elements": differing,
        }));
    }
    let extra = BTreeMap::from([
        ("kv".to_owned(), json!(dtype)),
        ("phase".to_owned(), json!(case.phase)),
        ("rows".to_owned(), json!(step.rows)),
        ("prepared".to_owned(), json!(prepared)),
        ("splits".to_owned(), json!(normalized.entry.splits())),
        ("lengths".to_owned(), json!(case.lengths)),
        ("sampled_rows".to_owned(), json!(sampled)),
        ("seconds".to_owned(), json!(started.elapsed().as_secs_f64())),
    ]);
    report.finish(session, extra)
}

/// The context-parallel merge (`ContextParallelQuery::PartialMerge`) in F32
/// on the host: `M = max m_r`, `a_r = exp(m_r - M)` (0 for an empty rank),
/// `L` and `O` FMA sums from zero in rank order, BF16 `O / L`.
fn merge_f32(ranks: &[&Partial], pairs: usize) -> Vec<u16> {
    let mut out = vec![0_u16; pairs * LATENT];
    for pair in 0..pairs {
        let top = ranks
            .iter()
            .map(|r| r.maximum[pair])
            .fold(f32::NEG_INFINITY, f32::max);
        if top == f32::NEG_INFINITY {
            continue;
        }
        let weights = ranks
            .iter()
            .map(|r| {
                if r.maximum[pair] == f32::NEG_INFINITY {
                    0.0
                } else {
                    (r.maximum[pair] - top).exp()
                }
            })
            .collect::<Vec<_>>();
        let total = ranks
            .iter()
            .zip(&weights)
            .fold(0.0_f32, |l, (r, &a)| a.mul_add(r.denominator[pair], l));
        for d in 0..LATENT {
            let index = pair * LATENT + d;
            let o = ranks
                .iter()
                .zip(&weights)
                .fold(0.0_f32, |o, (r, &a)| a.mul_add(r.numerator[index], o));
            out[index] = f32_to_bf16(o / total);
        }
    }
    out
}

pub fn oracle_cases(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let mut failures = Vec::new();
    let mut seed = 0x1a7e_0000;
    for kv in [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16] {
        let dtype = dtype_name(kv);
        let cases = [
            ("decode_b1", "decode", vec![50_000], 1, None),
            (
                "decode_b8",
                "decode",
                vec![1, 7, 2051, 2052, 3000, 10_000, 65, 4096],
                1,
                None,
            ),
            // Eight live rows planned in a 32-row bucket.
            (
                "decode_b8_bucket32",
                "decode",
                vec![300, 2050, 5, 9000, 2053, 64, 128, 777],
                1,
                Some(32),
            ),
            (
                "decode_b32",
                "decode",
                (0..32).map(|i| 1 + i * 257).collect(),
                1,
                None,
            ),
            (
                "verify_k3",
                "verify",
                vec![4, 2049, 2052, 4100, 70, 8000, 2051, 12],
                4,
                None,
            ),
            ("verify_k5", "verify", vec![6, 2050, 3333, 20_000], 6, None),
            ("prefill_512", "prefill", vec![8192], 512, None),
            ("prefill_8192", "prefill", vec![8192], 8192, None),
        ];
        for (name, phase, lengths, rows_per_item, prepared) in cases {
            seed += 1;
            let case = Case {
                name: format!("{name}_{dtype}"),
                kv,
                phase,
                lengths,
                rows_per_item,
                prepared,
            };
            if let Err(error) = attention_case(session, kernels, &case, seed) {
                failures.push(error);
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

// ------------------------------------------------- batch invariance ----

/// One row of an invariance launch: its sequence, key list and queries.
#[derive(Clone)]
struct RowInput {
    batch: i32,
    position: usize,
    list: Vec<i32>,
    q: Vec<u16>,
}

/// An invariance launch's rows and its (row, probe) pairs.
type ProbeLaunch<'a> = (Vec<&'a RowInput>, Vec<(usize, usize)>);

/// A step of `rows` in order.
fn step_of(rows: &[&RowInput]) -> Step {
    Step {
        rows: rows.len(),
        empty: None,
        row_batch: rows.iter().map(|row| row.batch).collect(),
        positions: rows.iter().map(|row| row.position).collect(),
        lists: rows.iter().flat_map(|row| row.list.clone()).collect(),
        q: rows.iter().flat_map(|row| row.q.clone()).collect(),
    }
}

/// One row's output bytes: the BF16 row, or the partial's maxima,
/// denominators and numerators.
fn row_bytes(
    buffers: &StepBuffers,
    session: &Session,
    rows: usize,
    row: usize,
    partial: bool,
) -> LabResult<Vec<u8>> {
    if !partial {
        let out = buffers.normalized(session, rows)?;
        return Ok(u16_bytes(
            &out[row * HEADS * LATENT..(row + 1) * HEADS * LATENT],
        ));
    }
    let parts = buffers.partial(session, rows)?.rows(&[row]);
    let f32_bytes = |values: &[f32]| bytes_of(values, f32::to_ne_bytes);
    Ok([
        f32_bytes(&parts.maximum),
        f32_bytes(&parts.denominator),
        f32_bytes(&parts.numerator),
    ]
    .concat())
}

/// The schedule decode and verify ran before their splits were fixed, as the
/// reference: one split per CTA over all 64 splits, which every bucket of
/// up to 8 rows took. Its kernel is the prefill entry's (`attention`, the
/// same traits; `sass_same.sh` showed the old decode and verify kernels
/// equal it), launched with 64 splits, then the entry's own merge.
fn launch_split_per_cta(
    session: &Session,
    kernels: &Kernels,
    buffers: &StepBuffers,
    cache: &Cache,
    entry: &Kernel,
    partial: bool,
    prepared: u32,
) -> LabResult<()> {
    let suffix = if partial { "_partial" } else { "" };
    let reference = Kernel::load(
        kernels,
        &format!("attn_{}_prefill{suffix}", dtype_name(cache.kv)),
    )?;
    let splits = entry.entry.splits();
    let mut plan = reference
        .entry
        .plan(prepared)
        .map_err(|e| e.0.to_string())?;
    plan.grid[2] = splits;
    plan.pdl = false;
    let buffer_map = BTreeMap::from([
        ("q", &buffers.q),
        ("pool", &cache.d_pool),
        ("block_tables", &cache.d_tables),
        ("row_batch", &buffers.row_batch),
        ("key_lists", &buffers.lists),
        ("live_rows", &buffers.live),
        ("workspace", &buffers.workspace),
        ("out", &buffers.out),
        ("out_max", &buffers.out_max),
        ("out_den", &buffers.out_den),
        ("out_num", &buffers.out_num),
    ]);
    let scalars = BTreeMap::from([
        ("max_pages", cache.max_pages as u64),
        ("page_stride_bytes", cache.page_stride as u64),
        ("budget", MAX_KEYS as u64),
        ("splits", u64::from(splits)),
        ("scale_bits", u64::from(GEOMETRY.softmax_scale.to_bits())),
    ]);
    let args = |params: &[EntryParam]| bind(params, &buffer_map, &scalars);
    reference
        .main
        .launch(
            &session.stream,
            launch::config(plan)?,
            &args(&reference.entry.params())?,
        )
        .map_err(|e| e.to_string())?;
    reference.record(reference.entry.symbol());
    let (Some(merge), Some(mut merge_plan)) = (
        &entry.merge,
        entry
            .entry
            .merge_plan(prepared)
            .map_err(|e| e.0.to_string())?,
    ) else {
        return Err("a decode or verify entry has a merge".into());
    };
    merge_plan.pdl = false;
    merge
        .launch(
            &session.stream,
            launch::config(merge_plan)?,
            &args(&entry.entry.merge_params())?,
        )
        .map_err(|e| e.to_string())?;
    entry.record(entry.entry.merge_symbol().ok_or("a merge has a symbol")?);
    Ok(())
}

/// Buckets the invariance case runs every decode and verify probe row in:
/// one split per CTA up to 8 rows, then 2, 4 (30 and 32 rows, either side of
/// an eager 30-row catch-up), 8 (36 rows) and, at the entries' capacity, 32.
const INVARIANCE_BUCKETS: [usize; 8] = [1, 8, 16, 30, 32, 36, 64, 256];

/// Rows of the prefill launches that hold the probe rows: one alone, a
/// catch-up pass either side of the 32-row verify limit, and a whole
/// prefill chunk of prompt rows.
const PREFILL_ROWS: [usize; 4] = [1, 30, 36, 2048];

/// A row's decode and verify output does not depend on its bucket or on
/// the rows beside it. Four probe rows: a full 2,051-key row, a 70-key row,
/// and the full row's keys owned by group rank 1 of 2 and rank 3 of 4 (whole
/// pages; the others holes, as `localize_keys` leaves them). Each runs
/// alone (bucket 1) and among other neighbours at buckets 8 to 256,
/// in both output modes of the decode and verify entries; every run's bytes
/// equal the reference schedule's ([`launch_split_per_cta`]) at bucket 8
/// with yet other neighbours. The prefill entries, which prompt rows and
/// the drafter's widest prefill-path passes run, give each probe the bytes
/// it has alone in launches of 30 and 36 rows and inside a 2048-row chunk.
pub fn batch_invariance(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let mut failures = Vec::new();
    for (seed, kv) in [
        (0x1a7e_1000, LatentKvFormat::Fp8E4M3Block128),
        (0x1a7e_2000, LatentKvFormat::Bf16),
    ] {
        if let Err(error) = invariance_case(session, kernels, kv, seed) {
            failures.push(error);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

fn invariance_case(
    session: &Session,
    kernels: &Kernels,
    kv: LatentKvFormat,
    seed: u64,
) -> LabResult<()> {
    let mut rng = Rng::new(seed);
    let started = std::time::Instant::now();
    // Sequence 0 holds the full row (512 pools and a 3-token tail), 1 the
    // short row, the rest the neighbours.
    let mut lengths = vec![10_003, 70];
    lengths.extend((0..62).map(|_| 1 + rng.below(6000)));
    let cache = Cache::new(
        session,
        kernels,
        kv,
        &lengths,
        &mut rng,
        |rng: &mut Rng, n| rng.bf16(n, 1.0),
    )?;
    let row = |rng: &mut Rng, batch: usize, holes: bool| {
        let position = lengths[batch] - 1;
        RowInput {
            batch: batch as i32,
            position,
            list: key_list(rng, position, holes),
            q: rng.bf16(HEADS * LATENT, 1.0),
        }
    };
    let full = row(&mut rng, 0, false);
    let owned = |rng: &mut Rng, ranks: usize, rank: usize| RowInput {
        list: full
            .list
            .iter()
            .map(|&p| {
                if p >= 0 && (p as usize / PAGE) % ranks == rank {
                    p
                } else {
                    -1
                }
            })
            .collect(),
        q: rng.bf16(HEADS * LATENT, 1.0),
        ..full.clone()
    };
    let probes = vec![
        ("full", full.clone()),
        ("short", row(&mut rng, 1, false)),
        ("rank 1 of 2", owned(&mut rng, 2, 1)),
        ("rank 3 of 4", owned(&mut rng, 4, 3)),
    ];
    if probes[0].1.list.iter().filter(|&&p| p >= 0).count() != MAX_KEYS
        || probes[1].1.list.iter().filter(|&&p| p >= 0).count() != 70
    {
        return Err("the probe rows do not hold 2,051 and 70 keys".into());
    }
    let neighbours = (0..256)
        .map(|i| row(&mut rng, 2 + i % 62, i % 3 == 0))
        .collect::<Vec<_>>();
    // Bucket `bucket` with the probes at rows `at` and neighbours from
    // `first` on everywhere else.
    let arrange = |bucket: usize, at: &[usize], first: usize| {
        let mut next = first;
        (0..bucket)
            .map(|r| match at.iter().position(|&a| a == r) {
                Some(p) => &probes[p].1,
                None => {
                    next += 1;
                    &neighbours[(next - 1) % neighbours.len()]
                }
            })
            .collect::<Vec<_>>()
    };
    let dtype = dtype_name(kv);
    let mut report = Report::new(kernels, format!("batch_invariance_{dtype}"));
    let mut groups = BTreeMap::new();
    for phase in ["decode", "verify"] {
        for suffix in ["", "_partial"] {
            let entry = Kernel::load(kernels, &format!("attn_{dtype}_{phase}{suffix}"))?;
            let partial = !suffix.is_empty();
            let name = format!("{phase}{suffix}");
            // The reference: bucket 8, probes at rows 1, 3, 5 and 7.
            let at = [1, 3, 5, 7];
            let step = step_of(&arrange(8, &at, 200));
            let buffers = StepBuffers::new(
                session,
                &step,
                &step.lists,
                entry.entry.workspace_bytes(8),
                8,
            )?;
            launch_split_per_cta(session, kernels, &buffers, &cache, &entry, partial, 8)?;
            let mut expected = Vec::new();
            for &r in &at {
                expected.push(row_bytes(&buffers, session, 8, r, partial)?);
            }
            for bucket in INVARIANCE_BUCKETS {
                groups.insert(bucket.to_string(), json!(entry.entry.groups(bucket as u32)));
                // Each launch's rows and its (row, probe) pairs: every probe
                // alone at bucket 1, otherwise spread over the bucket among
                // neighbours that differ per bucket.
                let launches: Vec<ProbeLaunch> = if bucket == 1 {
                    (0..probes.len())
                        .map(|p| (vec![&probes[p].1], vec![(0, p)]))
                        .collect()
                } else {
                    let at = [0, bucket / 2, bucket / 2 - 1, bucket - 1];
                    let pairs = at.iter().copied().zip(0..).collect();
                    vec![(arrange(bucket, &at, bucket * 3), pairs)]
                };
                for (rows, pairs) in launches {
                    let step = step_of(&rows);
                    let prepared = bucket as u32;
                    let buffers = StepBuffers::new(
                        session,
                        &step,
                        &step.lists,
                        entry.entry.workspace_bytes(prepared),
                        prepared,
                    )?;
                    buffers.launch(&session.stream, &entry, &cache, prepared, false)?;
                    for (r, probe) in pairs {
                        let actual = row_bytes(&buffers, session, step.rows, r, partial)?;
                        let probe_name = probes[probe].0;
                        report.exact(
                            &format!("{name} {probe_name} at bucket {bucket} row {r}"),
                            match actual
                                .iter()
                                .zip(&expected[probe])
                                .position(|(a, b)| a != b)
                            {
                                None => Ok(()),
                                Some(i) => Err(format!(
                                    "byte {i} is {:#04x}, the bucket-8 reference {:#04x}",
                                    actual[i], expected[probe][i]
                                )),
                            },
                        );
                    }
                }
            }
        }
    }
    // Prefill: one split per row, so each probe alone is the reference.
    for suffix in ["", "_partial"] {
        let entry = Kernel::load(kernels, &format!("attn_{dtype}_prefill{suffix}"))?;
        let partial = !suffix.is_empty();
        let mut expected = Vec::new();
        for rows in PREFILL_ROWS {
            let launches: Vec<ProbeLaunch> = if rows == 1 {
                (0..probes.len())
                    .map(|p| (vec![&probes[p].1], vec![(0, p)]))
                    .collect()
            } else {
                let at = [0, rows / 2, rows / 2 - 1, rows - 1];
                let pairs = at.iter().copied().zip(0..).collect();
                vec![(arrange(rows, &at, rows * 5), pairs)]
            };
            for (launch, pairs) in launches {
                let step = step_of(&launch);
                let prepared = rows as u32;
                let buffers = StepBuffers::new(
                    session,
                    &step,
                    &step.lists,
                    entry.entry.workspace_bytes(prepared),
                    prepared,
                )?;
                buffers.launch(&session.stream, &entry, &cache, prepared, false)?;
                for (r, probe) in pairs {
                    let actual = row_bytes(&buffers, session, step.rows, r, partial)?;
                    if rows == 1 {
                        expected.push(actual);
                        continue;
                    }
                    let probe_name = probes[probe].0;
                    report.exact(
                        &format!("prefill{suffix} {probe_name} in {rows} rows at row {r}"),
                        match actual
                            .iter()
                            .zip(&expected[probe])
                            .position(|(a, b)| a != b)
                        {
                            None => Ok(()),
                            Some(i) => Err(format!(
                                "byte {i} is {:#04x}, alone {:#04x}",
                                actual[i], expected[probe][i]
                            )),
                        },
                    );
                }
            }
        }
    }
    let extra = BTreeMap::from([
        ("kv".to_owned(), json!(dtype)),
        ("buckets".to_owned(), json!(INVARIANCE_BUCKETS)),
        ("prefill_rows".to_owned(), json!(PREFILL_ROWS)),
        ("groups_per_bucket".to_owned(), json!(groups)),
        (
            "splits".to_owned(),
            json!(kernels_sm120::latent::MAX_SPLITS),
        ),
        (
            "probes".to_owned(),
            json!(probes.iter().map(|(name, _)| *name).collect::<Vec<_>>()),
        ),
        ("seconds".to_owned(), json!(started.elapsed().as_secs_f64())),
    ]);
    report.finish(session, extra)
}

// -------------------------------------------------- 128-K accumulation ----

/// Latent rows and queries spanning 2^-8..2^8 with random signs: the
/// partial maximum is one 512-wide score and the numerator sums 2051 keys,
/// against Neumaier-compensated f64 within 2^-22 K S.
pub fn attention_accumulation(
    session: &Session,
    kernels: &Kernels,
    report: &mut Report,
) -> LabResult<()> {
    for kv in [LatentKvFormat::Bf16, LatentKvFormat::Fp8E4M3Block128] {
        let mut rng = Rng::new(0x128c);
        let lengths = vec![2051, 9000, 2051, 40_000];
        let cache = Cache::new(
            session,
            kernels,
            kv,
            &lengths,
            &mut rng,
            |rng: &mut Rng, n| rng.cancelling(n),
        )?;
        let mut step = Step::new(&mut rng, &lengths, 1, false);
        // Small queries keep the scores of cancelling keys within a few
        // units of each other, so many keys carry weight.
        step.q = rng
            .cancelling(step.rows * HEADS * LATENT)
            .into_iter()
            .map(|x| f32_to_bf16(bf16_to_f32(x) / 1024.0))
            .collect();
        let kernel = Kernel::load(kernels, &format!("attn_{}_decode_partial", dtype_name(kv)))?;
        let prepared = step.rows as u32;
        let buffers = StepBuffers::new(
            session,
            &step,
            &step.lists,
            kernel.entry.workspace_bytes(prepared),
            prepared,
        )?;
        buffers.launch(&session.stream, &kernel, &cache, prepared, false)?;
        let parts = buffers.partial(session, step.rows)?;
        let rows = (0..step.rows).collect::<Vec<_>>();
        let LatentAttentionResult::Partial(reference) =
            oracle(&kernel, &cache, &step, &step.lists, &rows)?
        else {
            unreachable!()
        };
        gate_partial(report, dtype_name(kv), &parts, &reference);
    }
    Ok(())
}

/// The F32 projections reduce K = 256 and 512 of cancelling operands,
/// against Neumaier-compensated f64 within 2^-22 K S.
pub fn projection_accumulation(
    session: &Session,
    kernels: &Kernels,
    report: &mut Report,
) -> LabResult<()> {
    // Both projections with F32 output on cancelling operands, bound in
    // place in one up-projection tensor [heads * 512, 512].
    let mut rng = Rng::new(0x128d);
    let rows = 33;
    let stride = 2 * GEOMETRY.nope as usize * LATENT;
    let up = rng.cancelling(HEADS * stride);
    let up_f64 = decode(&up);
    for (suffix, k, n, offset) in [
        ("absorb_key_f32", GEOMETRY.nope as usize, LATENT, 0),
        (
            "absorb_value_f32",
            LATENT,
            GEOMETRY.v_dim as usize,
            GEOMETRY.nope as usize * LATENT,
        ),
    ] {
        let kernel = Kernel::load(kernels, suffix)?;
        let x = rng.cancelling(rows * HEADS * k);
        let d_x = upload(session, &u16_bytes(&x))?;
        let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
        let d_y = zeros(session, rows * HEADS * n * 4)?;
        let weights = upload(session, &u16_bytes(&up[offset..]))?;
        gemm_launch(
            session, &kernel, &d_x, &weights, &d_live, &d_y, stride, rows, false,
        )?;
        let y = as_f32(&download(session, &d_y, rows * HEADS * n * 4)?);
        let Query::HeadGemm(query) = kernel.entry.query else {
            unreachable!()
        };
        let reference = head_batched_gemm(query, &decode(&x), &up_f64[offset..], stride, rows)
            .map_err(|e| e.0.to_string())?;
        report.gate(
            suffix,
            &y.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &reference,
            DType::F32,
        );
    }
    Ok(())
}

// ------------------------------------------------------- projections ----

#[allow(clippy::too_many_arguments)]
fn gemm_launch(
    session: &Session,
    kernel: &Kernel,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    live: &DeviceBuffer,
    y: &DeviceBuffer,
    head_stride: usize,
    prepared: usize,
    pdl: bool,
) -> LabResult<()> {
    let buffers = BTreeMap::from([("x", x), ("w", w), ("live_rows", live), ("y", y)]);
    let scalars = BTreeMap::from([("head_stride", head_stride as u64)]);
    kernel.launch(&session.stream, prepared as u32, &buffers, &scalars, pdl)
}

/// The served BF16 projections against the oracle at decode, verify and
/// prefill row counts, reading both halves of one up-projection tensor in
/// place.
pub fn projections(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let mut report = Report::new(kernels, "projections");
    let mut rng = Rng::new(0x9e0);
    let stride = 2 * GEOMETRY.nope as usize * LATENT;
    let up = rng.bf16(HEADS * stride, 0.05);
    let up_f64 = decode(&up);
    for (suffix, k, n, offset) in [
        ("absorb_key", GEOMETRY.nope as usize, LATENT, 0),
        (
            "absorb_value",
            LATENT,
            GEOMETRY.v_dim as usize,
            GEOMETRY.nope as usize * LATENT,
        ),
    ] {
        let kernel = Kernel::load(kernels, suffix)?;
        let Query::HeadGemm(query) = kernel.entry.query else {
            unreachable!()
        };
        for rows in [1, 7, 32, 192, 1000, 8192] {
            let x = rng.bf16(rows * HEADS * k, 1.0);
            let d_x = upload(session, &u16_bytes(&x))?;
            let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
            let d_y = zeros(session, rows * HEADS * n * 2)?;
            let weights = upload(session, &u16_bytes(&up[offset..]))?;
            gemm_launch(
                session, &kernel, &d_x, &weights, &d_live, &d_y, stride, rows, false,
            )?;
            let y = as_u16(&download(session, &d_y, rows * HEADS * n * 2)?);
            let sampled = (0..rows).step_by((rows / 16).max(1)).chain([rows - 1]);
            let sampled = {
                let mut s = sampled.collect::<Vec<_>>();
                s.dedup();
                s
            };
            let x_rows = pick(&x, &sampled, HEADS * k);
            let reference = head_batched_gemm(
                query,
                &decode(&x_rows),
                &up_f64[offset..],
                stride,
                sampled.len(),
            )
            .map_err(|e| e.0.to_string())?;
            report.gate(
                &format!("{suffix} rows={rows}"),
                &decode(&pick(&y, &sampled, HEADS * n)),
                &reference,
                DType::Bf16,
            );
        }
    }
    report.finish(session, BTreeMap::new())
}

// ------------------------------------------------------------- poison ----

/// Rows `rows` of a `[rows, width]` buffer of `element`.
fn prefix(rows: usize, width: usize, element: DType) -> LabResult<WriteMap> {
    WriteMap::of(element).prefix(rows * width)
}

/// The pool bytes a KV write of `positions` of sequence `sequence` writes:
/// each token's row, and for FP8 its scales.
fn written_pool(cache: &Cache, sequence: usize, positions: Range<usize>) -> Vec<Range<usize>> {
    let token = match cache.kv {
        LatentKvFormat::Bf16 => LATENT * 2,
        LatentKvFormat::Fp8E4M3Block128 => LATENT,
        LatentKvFormat::Fp8E4M3Ue8M0Block32 | LatentKvFormat::Fp4E2M1E4M3Block16 => {
            unreachable!("native latent formats need their own conformance cases")
        }
    };
    let mut ranges = Vec::new();
    for p in positions {
        let page = cache.tables[sequence * cache.max_pages + p / PAGE] as usize;
        let base = page * cache.page_stride;
        let slot = p % PAGE;
        ranges.push(base + slot * token..base + (slot + 1) * token);
        if cache.kv == LatentKvFormat::Fp8E4M3Block128 {
            let scales = base + PAGE * LATENT + slot * (LATENT / 128) * 4;
            ranges.push(scales..scales + (LATENT / 128) * 4);
        }
    }
    ranges
}

/// Poisoned eager, relaunch, graph and PDL-producer runs: each writes
/// exactly its contract's bytes (the outputs of the live rows of a larger
/// prepared bucket, a split workspace only within the live rows' partials,
/// the KV write only its tokens' codes and scales) and reproduces the eager
/// bits. Every phase first has the PDL copy producer write a real input
/// (the queries, the latent rows or the activations) over its poison, so a
/// consumer that reads it before its PDL wait diverges from eager.
pub fn poisoned_replay(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let pdl = std::env::var("CHOPINFER_LATENT_NO_PDL").as_deref() != Ok("1");
    let producer = CopyProducer::new(kernels.lab.as_ref().unwrap_or(&kernels.module))?;
    let mut evidence = BTreeMap::new();
    let mut rng = Rng::new(0x9015);
    let stream = &session.stream;

    // Attention with its merge: decode FP8 normalized (split), verify
    // BF16 partial (split) and prefill FP8 partial (one split), each
    // with an all-hole row.
    for (suffix, kv, lengths, rows_per_item) in [
        (
            "attn_fp8_decode",
            LatentKvFormat::Fp8E4M3Block128,
            vec![10, 2052, 7000, 64, 1, 3000, 2051, 129],
            1,
        ),
        (
            "attn_bf16_verify_partial",
            LatentKvFormat::Bf16,
            vec![2049, 5000, 4, 300],
            4,
        ),
        (
            "attn_fp8_prefill_partial",
            LatentKvFormat::Fp8E4M3Block128,
            vec![3000],
            512,
        ),
    ] {
        let cache = Cache::new(
            session,
            kernels,
            kv,
            &lengths,
            &mut rng,
            |rng: &mut Rng, n| rng.bf16(n, 1.0),
        )?;
        let step = Step::new(&mut rng, &lengths, rows_per_item, true);
        let empty = step.empty.ok_or("a poison step has an all-hole row")?;
        let kernel = Kernel::load(kernels, suffix)?;
        // Plan for a larger bucket so CTAs past the live rows run; the
        // outputs hold the bucket's rows and only the live ones may change.
        let prepared = (step.rows as u32 + 3).min(kernel.entry.capacity());
        let splits = kernel.entry.splits() as usize;
        let workspace = kernel.entry.workspace_bytes(prepared);
        let buffers = StepBuffers::new(session, &step, &step.lists, workspace, prepared)?;
        let staged_q = upload(session, &u16_bytes(&step.q))?;
        let q_bytes = step.q.len() * 2;
        let partial = suffix.ends_with("_partial");
        let mut replay = Replay::new(session).state(
            "q",
            &buffers.q,
            download(session, &buffers.q, buffers.q.byte_len())?,
            WriteMap::of(DType::Bf16).byte_prefix(q_bytes)?,
            vec![],
        );
        replay = if partial {
            replay
                .output(
                    "out_max",
                    &buffers.out_max,
                    prefix(step.rows, HEADS, DType::F32)?,
                )
                .output(
                    "out_den",
                    &buffers.out_den,
                    prefix(step.rows, HEADS, DType::F32)?,
                )
                .output(
                    "out_num",
                    &buffers.out_num,
                    prefix(step.rows, HEADS * LATENT, DType::F32)?,
                )
        } else {
            replay.output(
                "out",
                &buffers.out,
                prefix(step.rows, HEADS * LATENT, DType::Bf16)?,
            )
        };
        if splits > 1 {
            // Split partials of the live rows: [row][split][head][2 + latent].
            replay = replay.scratch_within(
                "workspace",
                &buffers.workspace,
                std::iter::once(0..step.rows * splits * HEADS * (LATENT + 2) * 4).collect(),
            );
        }
        let run =
            |stream: &Stream, pdl: bool| buffers.launch(stream, &kernel, &cache, prepared, pdl);
        let chain = |stream: &Stream, pdl: bool| {
            producer.copy(stream, &staged_q, &buffers.q)?;
            run(stream, pdl)
        };
        replay.eager(|| chain(stream, false))?;
        // The all-hole row's eager bits: +0, or the exact empty partial.
        let row_result = if partial {
            let image = |name| as_f32(replay.expected(name).unwrap_or_default());
            empty_partial(
                &Partial {
                    maximum: image("out_max"),
                    denominator: image("out_den"),
                    numerator: image("out_num"),
                },
                empty,
            )
        } else {
            zero_row(&as_u16(replay.expected("out").unwrap_or_default()), empty)
        };
        row_result.map_err(|e| format!("{suffix} all-hole row: {e}"))?;
        replay.relaunch(|| chain(stream, false))?;
        let graph = session.capture(|stream| chain(stream, pdl))?;
        replay.graph(&graph)?;
        replay.pdl(
            || producer.copy(stream, &staged_q, &buffers.q),
            || run(stream, pdl),
            pdl,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        receipt.insert("splits".into(), json!(splits));
        receipt.insert("prepared".into(), json!(prepared));
        receipt.insert("rows".into(), json!(step.rows));
        evidence.insert(suffix.to_owned(), json!(receipt));
    }

    // The KV write: in-place pool state whose written bytes are the new
    // tokens' rows and scales; pages, gaps and other tokens keep theirs.
    // The producer writes the latent rows.
    for kv in [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16] {
        let lengths = vec![700, 130];
        let cache = Cache::new(
            session,
            kernels,
            kv,
            &lengths,
            &mut rng,
            |rng: &mut Rng, n| rng.bf16(n, 1.0),
        )?;
        // Rewrite positions 600..700 of sequence 0 and 0..130 of 1 with
        // new rows, and give one padded row position -1.
        let mut positions = (600..700).chain(0..130).collect::<Vec<i32>>();
        let mut batch = [vec![0; 100], vec![1; 130]].concat();
        positions.push(-1);
        batch.push(0);
        let rows = positions.len();
        let latent = u16_bytes(&rng.bf16(rows * LATENT, 1.0));
        let staged = upload(session, &latent)?;
        let d_latent = upload(session, &latent)?;
        let d_batch = upload(session, &i32_bytes(&batch))?;
        let d_positions = upload(session, &i32_bytes(&positions))?;
        let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
        let mut written = written_pool(&cache, 0, 600..700);
        written.extend(written_pool(&cache, 1, 0..130));
        // Four-byte elements: no code group, BF16 pair or scale is all
        // ones.
        let element = DType::I32;
        let kernel = Kernel::load(kernels, &format!("kv_write_{}", dtype_name(kv)))?;
        let buffers = BTreeMap::from([
            ("latent", &d_latent),
            ("row_batch", &d_batch),
            ("positions", &d_positions),
            ("live_rows", &d_live),
            ("block_tables", &cache.d_tables),
            ("pool", &cache.d_pool),
        ]);
        let scalars = BTreeMap::from([
            ("max_pages", cache.max_pages as u64),
            ("page_stride_bytes", cache.page_stride as u64),
        ]);
        let run = |stream: &Stream, pdl: bool| kernel.launch(stream, 256, &buffers, &scalars, pdl);
        let chain = |stream: &Stream, pdl: bool| {
            producer.copy(stream, &staged, &d_latent)?;
            run(stream, pdl)
        };
        let mut replay = Replay::new(session)
            .state(
                "pool",
                &cache.d_pool,
                cache.pool.clone(),
                WriteMap::of(element).byte_ranges(written)?,
                vec![],
            )
            .state(
                "latent",
                &d_latent,
                download(session, &d_latent, d_latent.byte_len())?,
                WriteMap::of(DType::Bf16).byte_prefix(latent.len())?,
                vec![],
            );
        replay.eager(|| chain(stream, false))?;
        replay.relaunch(|| chain(stream, false))?;
        let graph = session.capture(|stream| chain(stream, pdl))?;
        replay.graph(&graph)?;
        replay.pdl(
            || producer.copy(stream, &staged, &d_latent),
            || run(stream, pdl),
            pdl,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        evidence.insert(format!("kv_write_{}", dtype_name(kv)), json!(receipt));
    }

    // Both projections, planned for 64 rows with 40 live; the producer
    // writes the activations (the weights may stream before the wait).
    let stride = 2 * GEOMETRY.nope as usize * LATENT;
    let up = rng.bf16(HEADS * stride, 0.05);
    for (suffix, k, n, offset) in [
        ("absorb_key", GEOMETRY.nope as usize, LATENT, 0),
        (
            "absorb_value",
            LATENT,
            GEOMETRY.v_dim as usize,
            GEOMETRY.nope as usize * LATENT,
        ),
    ] {
        let kernel = Kernel::load(kernels, suffix)?;
        let (rows, prepared) = (40, 64);
        let x = u16_bytes(&rng.bf16(prepared * HEADS * k, 1.0));
        let staged = upload(session, &x)?;
        let d_x = upload(session, &x)?;
        let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
        let d_y = zeros(session, prepared * HEADS * n * 2)?;
        let weights = upload(session, &u16_bytes(&up[offset..]))?;
        let run = |pdl: bool| {
            gemm_launch(
                session, &kernel, &d_x, &weights, &d_live, &d_y, stride, prepared, pdl,
            )
        };
        let chain = |pdl: bool| {
            producer.copy(stream, &staged, &d_x)?;
            run(pdl)
        };
        let mut replay = Replay::new(session)
            .state(
                "x",
                &d_x,
                download(session, &d_x, d_x.byte_len())?,
                WriteMap::of(DType::Bf16).byte_prefix(x.len())?,
                vec![],
            )
            .output("y", &d_y, prefix(rows, HEADS * n, DType::Bf16)?);
        replay.eager(|| chain(false))?;
        replay.relaunch(|| chain(false))?;
        let graph = session.capture(|_| chain(pdl))?;
        replay.graph(&graph)?;
        replay.pdl(|| producer.copy(stream, &staged, &d_x), || run(pdl), pdl)?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        evidence.insert(suffix.to_owned(), json!(receipt));
    }
    let mut data = BTreeMap::new();
    data.insert("cubin".into(), json!(kernels.cubin));
    data.insert("case".into(), json!("poison"));
    data.insert("pdl".into(), json!(pdl));
    data.insert("entries".into(), json!(evidence));
    data.insert("correct".into(), json!(true));
    session.receipt("latent_attention", data)?;
    Ok(())
}

/// The split merge folds crafted partials in ascending split order, checked
/// against an independent ascending F32 FMA expectation bit for bit. Head 0
/// has equal maxima and numerators [2^25, -2^25, 1, 0] in its first column:
/// ascending order gives O = 1 and BF16 O / L = 0.25, a reversed fold 0.
/// Head 1 has an empty split, head 2 only empty splits (maximum -inf,
/// denominator +0, numerator +0, BF16 +0), the rest random partials with
/// equal maxima, so every weight is exactly 1 or 0.
pub fn split_merge_order(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let mut rng = Rng::new(0x5e7);
    let splits = 4_usize;
    let width = LATENT + 2;
    let mut workspace = vec![0.0_f32; splits * HEADS * width];
    for split in 0..splits {
        for head in 0..HEADS {
            let at = (split * HEADS + head) * width;
            let empty = head == 2 || (head == 1 && split == 1);
            if empty {
                workspace[at] = f32::NEG_INFINITY;
                continue;
            }
            workspace[at + 1] = 1.0 + rng.uniform() as f32;
            for d in 0..LATENT {
                workspace[at + 2 + d] = rng.normal() as f32;
            }
            if head == 0 {
                workspace[at + 1] = 1.0;
                workspace[at + 2] = [2_f32.powi(25), -(2_f32.powi(25)), 1.0, 0.0][split];
            }
        }
    }
    // The expectation: maxima are 0 or -inf, so a_s is 1 or 0 exactly.
    let mut maximum = vec![f32::NEG_INFINITY; HEADS];
    let mut denominator = vec![0.0_f32; HEADS];
    let mut numerator = vec![0.0_f32; HEADS * LATENT];
    for head in 0..HEADS {
        for split in 0..splits {
            let at = (split * HEADS + head) * width;
            maximum[head] = maximum[head].max(workspace[at]);
        }
        for split in 0..splits {
            let at = (split * HEADS + head) * width;
            let a: f32 = if workspace[at] == f32::NEG_INFINITY {
                0.0
            } else {
                1.0
            };
            denominator[head] = a.mul_add(workspace[at + 1], denominator[head]);
            for d in 0..LATENT {
                let o = &mut numerator[head * LATENT + d];
                *o = a.mul_add(workspace[at + 2 + d], *o);
            }
        }
    }
    let normalized = numerator
        .iter()
        .enumerate()
        .map(|(i, &o)| {
            let l = denominator[i / LATENT];
            f32_to_bf16(if l == 0.0 { 0.0 } else { o / l })
        })
        .collect::<Vec<_>>();
    assert_eq!(normalized[0], f32_to_bf16(0.25));

    let mut report = Report::new(kernels, "split_merge_order");
    let d_workspace = upload(
        session,
        &workspace
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>(),
    )?;
    let d_live = upload(session, &1_u32.to_ne_bytes())?;
    for suffix in ["attn_bf16_decode", "attn_bf16_decode_partial"] {
        let kernel = Kernel::load(kernels, suffix)?;
        let merge = kernel.merge.as_ref().ok_or("a decode entry has a merge")?;
        let plan = kernel
            .entry
            .merge_plan(1)
            .map_err(|e| e.0.to_string())?
            .ok_or("one decode row splits")?;
        let poisoned = |bytes: usize| upload(session, &vec![POISON; bytes]);
        let (out, out_max, out_den, out_num) = (
            poisoned(HEADS * LATENT * 2)?,
            poisoned(HEADS * 4)?,
            poisoned(HEADS * 4)?,
            poisoned(HEADS * LATENT * 4)?,
        );
        let buffers = BTreeMap::from([
            ("workspace", &d_workspace),
            ("live_rows", &d_live),
            ("out", &out),
            ("out_max", &out_max),
            ("out_den", &out_den),
            ("out_num", &out_num),
        ]);
        let scalars = BTreeMap::from([("splits", splits as u64)]);
        merge
            .launch(
                &session.stream,
                launch::config(plan)?,
                &bind(&kernel.entry.merge_params(), &buffers, &scalars)?,
            )
            .map_err(|e| e.to_string())?;
        kernel.record(kernel.entry.merge_symbol().ok_or("a merge has a symbol")?);
        let bits_equal = |name: &str, actual: Vec<u32>, expected: Vec<u32>| match actual
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
        {
            None if actual.len() == expected.len() => Ok(()),
            None => Err(format!("{name}: length {}", actual.len())),
            Some(i) => Err(format!(
                "{name} element {i}: {:#010x}, expected {:#010x}",
                actual[i], expected[i]
            )),
        };
        let f32_bits = |bytes: Vec<u8>| as_f32(&bytes).iter().map(|v| v.to_bits()).collect();
        let expect = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect();
        if suffix.ends_with("_partial") {
            report.exact(
                "partial maximum",
                bits_equal(
                    "maximum",
                    f32_bits(download(session, &out_max, HEADS * 4)?),
                    expect(&maximum),
                ),
            );
            report.exact(
                "partial denominator",
                bits_equal(
                    "denominator",
                    f32_bits(download(session, &out_den, HEADS * 4)?),
                    expect(&denominator),
                ),
            );
            report.exact(
                "partial numerator",
                bits_equal(
                    "numerator",
                    f32_bits(download(session, &out_num, HEADS * LATENT * 4)?),
                    expect(&numerator),
                ),
            );
        } else {
            let actual = as_u16(&download(session, &out, HEADS * LATENT * 2)?);
            report.exact(
                "normalized",
                bits_equal(
                    "normalized",
                    actual.iter().map(|&v| u32::from(v)).collect(),
                    normalized.iter().map(|&v| u32::from(v)).collect(),
                ),
            );
        }
    }
    report.finish(session, BTreeMap::new())
}

/// The KV write at widths that are not powers of two (and the narrowest and
/// widest) matches the oracle bit for bit over the whole pool, so no lane
/// writes another block's codes or a scale past its token. One row holds an
/// all-zero block (scale 1).
pub fn kv_write_widths(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let mut rng = Rng::new(0x3840);
    let mut report = Report::new(kernels, "kv_write_widths");
    for width in latent_shapes::WRITE_WIDTHS {
        let latent = width as usize;
        for kv in [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16] {
            let lengths = [70_usize, 130];
            let max_pages = 130_usize.div_ceil(PAGE);
            let pages = lengths.iter().map(|l| l.div_ceil(PAGE)).sum::<usize>();
            let mut ids = (0..pages as i32).collect::<Vec<_>>();
            for i in (1..ids.len()).rev() {
                ids.swap(i, rng.below(i + 1));
            }
            let mut tables = vec![-1; 2 * max_pages];
            let mut next = 0;
            for (s, &len) in lengths.iter().enumerate() {
                for page in 0..len.div_ceil(PAGE) {
                    tables[s * max_pages + page] = ids[next];
                    next += 1;
                }
            }
            let page_stride = kv.page_bytes(width, GEOMETRY.page_size) as usize + PAGE_GAP;
            let mut pool = vec![0x5a_u8; pages * page_stride];
            let rows = 200;
            let batch = [vec![0; 70], vec![1; 130]].concat();
            let positions = (0..70).chain(0..130).collect::<Vec<i32>>();
            let mut bits = rng.bf16(rows * latent, 1.0);
            bits[5 * latent..5 * latent + 128].fill(0);
            let d_pool = upload(session, &pool)?;
            let d_tables = upload(session, &i32_bytes(&tables))?;
            let d_latent = upload(session, &u16_bytes(&bits))?;
            let d_batch = upload(session, &i32_bytes(&batch))?;
            let d_positions = upload(session, &i32_bytes(&positions))?;
            let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
            let name = format!("kv_write_{}_w{width}", dtype_name(kv));
            let kernel = Kernel::load(kernels, &name)?;
            let buffers = BTreeMap::from([
                ("latent", &d_latent),
                ("row_batch", &d_batch),
                ("positions", &d_positions),
                ("live_rows", &d_live),
                ("block_tables", &d_tables),
                ("pool", &d_pool),
            ]);
            let scalars = BTreeMap::from([
                ("max_pages", max_pages as u64),
                ("page_stride_bytes", page_stride as u64),
            ]);
            kernel.launch(&session.stream, rows as u32, &buffers, &scalars, false)?;
            let written = download(session, &d_pool, pool.len())?;
            latent_kv_write(
                LatentAttentionQuery::KvWrite {
                    latent: width,
                    page_size: GEOMETRY.page_size,
                    kv,
                    capacity: WRITE_ROWS as u32,
                },
                &bits,
                &batch,
                &positions,
                LatentPaging {
                    block_tables: &tables,
                    max_pages,
                    page_stride,
                },
                &mut pool,
            )
            .map_err(|e| e.0.to_string())?;
            report.exact(
                &name,
                match written.iter().zip(&pool).position(|(a, b)| a != b) {
                    None => Ok(()),
                    Some(offset) => Err(format!(
                        "pool byte {offset} is {:#04x}, the oracle wrote {:#04x}",
                        written[offset], pool[offset]
                    )),
                },
            );
        }
    }
    report.finish(session, BTreeMap::new())
}

// ------------------------------------------- context-parallel attention ----

/// The context-parallel entries a whole-step check launches beside the
/// latent entries: the lab's `context_parallel` cubin and its catalog.
pub struct ContextParallelKernels {
    pub module: Module,
    pub entries: Vec<kernels_sm120::context_parallel::Entry>,
}

impl ContextParallelKernels {
    fn function(
        &self,
        kind: kernels_sm120::context_parallel::Kind,
        list_width: u32,
    ) -> LabResult<(kernels_sm120::context_parallel::Entry, Function)> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.kind == kind && entry.geometry.list_width == list_width)
            .cloned()
            .ok_or_else(|| format!("no context-parallel entry {kind:?}"))?;
        let function = self
            .module
            .function(&entry.symbol())
            .map_err(|e| e.to_string())?;
        Ok((entry, function))
    }
}

/// `step` with the query `q` of another head block.
fn with_query(step: &Step, q: &[u16]) -> Step {
    Step {
        rows: step.rows,
        empty: step.empty,
        row_batch: step.row_batch.clone(),
        positions: step.positions.clone(),
        lists: step.lists.clone(),
        q: q.to_vec(),
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    bytes_of(values, f32::to_ne_bytes)
}

/// The rank-major gathered queries `[N][rows][16][512]` the query
/// all-gather delivers, staged as one device buffer.
struct GatheredQueries {
    buffer: DeviceBuffer,
    rows: usize,
}

impl GatheredQueries {
    fn new(session: &Session, blocks: &[Step]) -> LabResult<Self> {
        let bits = blocks
            .iter()
            .flat_map(|block| block.q.iter().copied())
            .collect::<Vec<_>>();
        Ok(Self {
            buffer: upload(session, &u16_bytes(&bits))?,
            rows: blocks.first().map_or(0, |block| block.rows),
        })
    }

    /// Source block `source`'s byte offset: `query_block_offset` elements
    /// of two bytes.
    fn offset(&self, source: usize) -> usize {
        kernels_sm120::context_parallel::query_block_offset(
            self.rows as u32,
            HEADS as u32,
            LATENT as u32,
            source as u32,
        ) as usize
            * 2
    }

    /// Bytes of one source block, `[rows][16][512]` BF16.
    fn block_bytes(&self) -> usize {
        self.rows * HEADS * LATENT * 2
    }
}

impl StepBuffers {
    /// [`StepBuffers::launch`] with the query bound to the `[rows, 16, 512]`
    /// view at byte `offset` of `queries`, the partial attention launch for
    /// one source block of the gathered queries.
    fn launch_from_view(
        &self,
        stream: &Stream,
        kernel: &Kernel,
        cache: &Cache,
        prepared: u32,
        queries: &GatheredQueries,
        offset: usize,
    ) -> LabResult<()> {
        let buffers = BTreeMap::from([
            ("pool", &cache.d_pool),
            ("block_tables", &cache.d_tables),
            ("row_batch", &self.row_batch),
            ("key_lists", &self.lists),
            ("live_rows", &self.live),
            ("workspace", &self.workspace),
            ("out", &self.out),
            ("out_max", &self.out_max),
            ("out_den", &self.out_den),
            ("out_num", &self.out_num),
        ]);
        let scalars = BTreeMap::from([
            ("max_pages", cache.max_pages as u64),
            ("page_stride_bytes", cache.page_stride as u64),
            ("budget", MAX_KEYS as u64),
            ("splits", u64::from(kernel.entry.splits())),
            ("scale_bits", u64::from(GEOMETRY.softmax_scale.to_bits())),
        ]);
        let args = |params: &[EntryParam]| -> LabResult<Vec<KernelArgument<'_>>> {
            let mut args = bind(params, &buffers, &scalars)?;
            for (arg, param) in args.iter_mut().zip(params) {
                if matches!(param, EntryParam::DevicePointer { name, .. } if name == "q") {
                    *arg = KernelArgument::Range {
                        buffer: &queries.buffer,
                        offset,
                        byte_len: queries.block_bytes(),
                    };
                }
            }
            Ok(args)
        };
        let plan = kernel.entry.plan(prepared).map_err(|e| e.0.to_string())?;
        kernel
            .main
            .launch(
                stream,
                launch::config(plan)?,
                &args(&kernel.entry.params())?,
            )
            .map_err(|e| e.to_string())?;
        kernel.record(kernel.entry.symbol());
        if let (Some(merge), Some(plan)) = (
            &kernel.merge,
            kernel
                .entry
                .merge_plan(prepared)
                .map_err(|e| e.0.to_string())?,
        ) {
            merge
                .launch(
                    stream,
                    launch::config(plan)?,
                    &args(&kernel.entry.merge_params())?,
                )
                .map_err(|e| e.to_string())?;
            kernel.record(kernel.entry.merge_symbol().ok_or("a merge has a symbol")?);
        }
        Ok(())
    }
}

/// Launches the partial merge over `partials` (one segment per group rank,
/// in group-rank order) read through `layout`, returning `rows` merged rows.
fn merge_on_device(
    session: &Session,
    cp: &ContextParallelKernels,
    ranks: u32,
    partials: &[f32],
    rows: usize,
    layout: kernels_sm120::context_parallel::MergeLayout,
) -> LabResult<Vec<u16>> {
    let (entry, function) = cp.function(
        kernels_sm120::context_parallel::Kind::PartialMerge { ranks },
        GEOMETRY.max_keys,
    )?;
    entry
        .check_merge(rows as u32, layout)
        .map_err(|e| e.0.to_string())?;
    let d_partials = upload(session, &f32_bytes(partials))?;
    let live = upload(session, &(rows as u32).to_ne_bytes())?;
    let out = upload(session, &vec![POISON; rows * HEADS * LATENT * 2])?;
    let plan = entry.plan(rows as u32).map_err(|e| e.0.to_string())?;
    function
        .launch(
            &session.stream,
            launch::config(plan)?,
            &[
                KernelArgument::Buffer(&d_partials),
                KernelArgument::Buffer(&live),
                KernelArgument::Buffer(&out),
                KernelArgument::U32(layout.rank_rows),
                KernelArgument::U32(layout.input_heads),
                KernelArgument::U32(layout.head_offset),
            ],
        )
        .map_err(|e| e.to_string())?;
    Ok(as_u16(&download(session, &out, rows * HEADS * LATENT * 2)?))
}

/// One whole context-parallel attention step at group size `ranks`, against
/// the single-rank result of every group head.
///
/// The group's `16 * ranks` heads are `ranks` query blocks `[rows, 16,
/// 512]`, staged as the query all-gather delivers them: one rank-major
/// buffer, each launch reading its source block's view at
/// `query_block_offset` (checked to matter by a wrong offset). Each group rank holds only
/// its own pages (`ContextParallelPagesV1`): its KV is written by the KV
/// write entry at its local positions into its own pool and tables. It
/// localizes the step's key lists with `localize_keys{ranks}` (checked bit
/// for bit against the host reference), then runs the existing partial
/// attention entry once per query block over its own keys. The partials are
/// exchanged on the host as the row exchange delivers them (one packed
/// block per source) and also laid out block-major as the all-gather
/// delivers them; each owner merges its own block with
/// `partial_merge{ranks}` through both layouts, which must agree bit for
/// bit. Every merged block passes the oracle's gate on sampled rows, as the
/// single-rank normalized output of the same block does; all-hole rows are
/// +0, and the differing elements against the single-rank output are
/// recorded.
fn context_parallel_case(
    session: &Session,
    kernels: &Kernels,
    cp: &ContextParallelKernels,
    case: &Case,
    ranks: usize,
    seed: u64,
) -> LabResult<()> {
    use kernels_sm120::context_parallel::{Kind as CpKind, MergeLayout};
    let mut rng = Rng::new(seed);
    let started = std::time::Instant::now();
    let cache = Cache::new(
        session,
        kernels,
        case.kv,
        &case.lengths,
        &mut rng,
        |rng: &mut Rng, n| rng.bf16(n, 1.0),
    )?;
    let step = Step::new(&mut rng, &case.lengths, case.rows_per_item, true);
    let rows = step.rows;
    let prepared = rows as u32;
    let dtype = dtype_name(case.kv);
    let normalized = Kernel::load(kernels, &format!("attn_{dtype}_{}", case.phase))?;
    let partial = Kernel::load(kernels, &format!("attn_{dtype}_{}_partial", case.phase))?;
    let workspace = normalized.entry.workspace_bytes(prepared);
    let blocks = (0..ranks)
        .map(|_| with_query(&step, &rng.bf16(rows * HEADS * LATENT, 1.0)))
        .collect::<Vec<_>>();
    let mut report = Report::new(kernels, format!("{}_cp{ranks}", case.name));
    let sampled = step.sampled();

    // The query all-gather's output: one rank-major buffer. Every launch
    // below reads its source block from it.
    let queries = GatheredQueries::new(session, &blocks)?;

    // The single-rank result of every block, over every key.
    let mut single = Vec::new();
    for (source, block) in blocks.iter().enumerate() {
        let buffers = StepBuffers::new(session, block, &block.lists, workspace, prepared)?;
        buffers.launch_from_view(
            &session.stream,
            &normalized,
            &cache,
            prepared,
            &queries,
            queries.offset(source),
        )?;
        single.push(buffers.normalized(session, rows)?);
    }
    // The view's offset matters: with several rows, a block offset of
    // `source * H * width` (the row factor missing) reads another block's
    // rows and gives a different output.
    if rows > 1 {
        let buffers = StepBuffers::new(session, &blocks[1], &step.lists, workspace, prepared)?;
        buffers.launch_from_view(
            &session.stream,
            &normalized,
            &cache,
            prepared,
            &queries,
            HEADS * LATENT * 2,
        )?;
        let wrong = buffers.normalized(session, rows)?;
        report.exact(
            "a block offset without the row factor changes the output",
            if wrong == single[1] {
                Err("the wrong view gave block 1's output".to_owned())
            } else {
                Ok(())
            },
        );
    }

    // Each group rank: its own pages, localized lists and one partial per
    // query block.
    let pages =
        runtime_kv::ContextParallelPagesV1::new(ranks, PAGE, ranks).map_err(|e| e.to_string())?;
    let (localize, localize_function) = cp.function(
        CpKind::LocalizeKeys {
            ranks: ranks as u32,
        },
        GEOMETRY.max_keys,
    )?;
    let d_lists = upload(session, &i32_bytes(&step.lists))?;
    let d_live = upload(session, &(rows as u32).to_ne_bytes())?;
    let mut parts: Vec<Vec<Partial>> = Vec::new();
    for rank in 0..ranks {
        let local_lengths = case
            .lengths
            .iter()
            .map(|&len| pages.rank_tokens(rank, len))
            .collect::<Vec<_>>();
        let mut local = case.lengths.iter().enumerate().map(|(s, &len)| {
            (0..pages.rank_tokens(rank, len))
                .flat_map(|k| {
                    let position = pages.rank_position(rank, k);
                    cache.latent[s][position * LATENT..(position + 1) * LATENT].to_vec()
                })
                .collect::<Vec<u16>>()
        });
        let local_cache = Cache::new(
            session,
            kernels,
            case.kv,
            &local_lengths,
            &mut rng,
            |_: &mut Rng, _| local.next().unwrap_or_default(),
        )?;
        let d_local = upload(session, &vec![POISON; rows * MAX_KEYS * 4])?;
        localize_function
            .launch(
                &session.stream,
                launch::config(localize.plan(prepared).map_err(|e| e.0.to_string())?)?,
                &[
                    KernelArgument::Buffer(&d_lists),
                    KernelArgument::Buffer(&d_live),
                    KernelArgument::Buffer(&d_local),
                    KernelArgument::U32(rank as u32),
                ],
            )
            .map_err(|e| e.to_string())?;
        let local_lists = download(session, &d_local, rows * MAX_KEYS * 4)?
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<_>>();
        let expected = kernel_core::context_parallel_oracle::localize_keys(
            localize.query(),
            rank as u32,
            &step.lists,
        )
        .map_err(|e| e.0.to_string())?;
        report.exact(
            &format!("rank {rank} localized lists"),
            match local_lists.iter().zip(&expected).position(|(a, e)| a != e) {
                None => Ok(()),
                Some(i) => Err(format!("entry {i}: {} vs {}", local_lists[i], expected[i])),
            },
        );
        let mut rank_parts = Vec::new();
        for (source, block) in blocks.iter().enumerate() {
            let buffers = StepBuffers::new(session, block, &local_lists, workspace, prepared)?;
            buffers.launch_from_view(
                &session.stream,
                &partial,
                &local_cache,
                prepared,
                &queries,
                queries.offset(source),
            )?;
            rank_parts.push(buffers.partial(session, rows)?);
        }
        parts.push(rank_parts);
    }

    // Each owner merges its own block through both delivery layouts.
    for owner in 0..ranks {
        let exchanged = (0..ranks)
            .flat_map(|rank| {
                let p = &parts[rank][owner];
                [
                    p.maximum.clone(),
                    p.denominator.clone(),
                    p.numerator.clone(),
                ]
                .concat()
            })
            .collect::<Vec<_>>();
        let merged = merge_on_device(
            session,
            cp,
            ranks as u32,
            &exchanged,
            rows,
            MergeLayout::exchanged(rows as u32, HEADS as u32),
        )?;
        let gathered = (0..ranks)
            .flat_map(|rank| {
                let blocks = &parts[rank];
                blocks
                    .iter()
                    .flat_map(|p| p.maximum.clone())
                    .chain(blocks.iter().flat_map(|p| p.denominator.clone()))
                    .chain(blocks.iter().flat_map(|p| p.numerator.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let merged_gathered = merge_on_device(
            session,
            cp,
            ranks as u32,
            &gathered,
            rows,
            MergeLayout::gathered(rows as u32, HEADS as u32, ranks as u32, owner as u32),
        )?;
        report.exact(
            &format!("owner {owner} exchanged and gathered merges agree"),
            match merged
                .iter()
                .zip(&merged_gathered)
                .position(|(a, b)| a != b)
            {
                None => Ok(()),
                Some(i) => Err(format!("element {i}")),
            },
        );
        if let Some(row) = step.empty {
            report.exact(
                &format!("owner {owner} all-hole row"),
                zero_row(&merged, row),
            );
        }
        let LatentAttentionResult::Normalized(reference) =
            oracle(&normalized, &cache, &blocks[owner], &step.lists, &sampled)?
        else {
            unreachable!("a normalized entry")
        };
        report.gate(
            &format!("owner {owner} single rank"),
            &decode(&pick(&single[owner], &sampled, HEADS * LATENT)),
            &reference,
            DType::Bf16,
        );
        report.gate(
            &format!("owner {owner} context-parallel merge"),
            &decode(&pick(&merged, &sampled, HEADS * LATENT)),
            &reference,
            DType::Bf16,
        );
        let differing = merged
            .iter()
            .zip(&single[owner])
            .filter(|(a, b)| a != b)
            .count();
        report.checks.push(json!({
            "name": format!("owner {owner} merge vs single rank"),
            "differing_elements": differing,
            "elements": merged.len(),
        }));
    }
    let extra = BTreeMap::from([
        ("kv".to_owned(), json!(dtype)),
        ("phase".to_owned(), json!(case.phase)),
        ("ranks".to_owned(), json!(ranks)),
        ("rows".to_owned(), json!(rows)),
        ("lengths".to_owned(), json!(case.lengths)),
        ("sampled_rows".to_owned(), json!(sampled)),
        ("seconds".to_owned(), json!(started.elapsed().as_secs_f64())),
    ]);
    report.finish(session, extra)
}

/// Whole context-parallel attention steps at group sizes 2 and 4 for FP8
/// and BF16 KV: decode batches with short and long items (some ranks own
/// none of a short item's pages), verify items of 4 rows and a 256-row
/// prefill chunk.
pub fn context_parallel_steps(
    session: &Session,
    kernels: &Kernels,
    cp: &ContextParallelKernels,
) -> LabResult<()> {
    let mut failures = Vec::new();
    let mut seed = 0xc0_0000;
    for kv in [LatentKvFormat::Fp8E4M3Block128, LatentKvFormat::Bf16] {
        let dtype = dtype_name(kv);
        let cases = [
            (
                "decode_b6",
                "decode",
                vec![70, 300, 2051, 2100, 5000, 129],
                1,
            ),
            ("verify_k3", "verify", vec![260, 2052, 4100, 70], 4),
            ("prefill_256", "prefill", vec![3000], 256),
        ];
        for (name, phase, lengths, rows_per_item) in cases {
            for ranks in [2, 4] {
                seed += 1;
                let case = Case {
                    name: format!("{name}_{dtype}"),
                    kv,
                    phase,
                    lengths: lengths.clone(),
                    rows_per_item,
                    prepared: None,
                };
                if let Err(error) = context_parallel_case(session, kernels, cp, &case, ranks, seed)
                {
                    failures.push(error);
                }
            }
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

// ------------------------------------------------- drafter timing ----

/// The drafter's latent attention before and after PERF-B14b, timed at its
/// row counts over FP8 pages: graphs of eight launches of the one-split
/// prefill entry (one CTA per row walks the row's 65 units of 32 keys)
/// against eight of the decode entry's 64 fixed splits with their merge,
/// each launch PDL after the previous. A step's rows are the last positions
/// of one sequence of 3,000 or 32,768 tokens, each with a full key list.
/// Both arms' normalized outputs of every row are compared with the oracle
/// on the sampled rows (the decision-0081 gate), and the elements where the
/// two arms differ are counted (the splits round differently).
pub fn drafter_attention_timing(session: &Session, kernels: &Kernels) -> LabResult<()> {
    const LAUNCHES: usize = 8;
    let kv = LatentKvFormat::Fp8E4M3Block128;
    let one = Kernel::load(kernels, "attn_fp8_prefill")?;
    let split = Kernel::load(kernels, "attn_fp8_decode")?;
    let mut rows_out = Vec::new();
    let mut report = Report::new(kernels, "drafter_attention_timing");
    for context in [3000_usize, 32_768] {
        let mut rng = Rng::new(0xd7af_0000 + context as u64);
        let cache = Cache::new(session, kernels, kv, &[context], &mut rng, |rng, n| {
            rng.bf16(n, 1.0)
        })?;
        for rows in [1_usize, 2, 4, 8, 16, 32, 36, 64, 192] {
            let step = Step::new(&mut rng, &[context], rows, false);
            let prepared = rows as u32;
            let workspace = split.entry.workspace_bytes(prepared);
            let buffers = [
                StepBuffers::new(session, &step, &step.lists, workspace, prepared)?,
                StepBuffers::new(session, &step, &step.lists, workspace, prepared)?,
            ];
            for (kernel, buffers) in [&one, &split].into_iter().zip(&buffers) {
                buffers.launch(&session.stream, kernel, &cache, prepared, false)?;
            }
            let outputs = [
                buffers[0].normalized(session, rows)?,
                buffers[1].normalized(session, rows)?,
            ];
            let sampled = step.sampled();
            let LatentAttentionResult::Normalized(reference) =
                oracle(&one, &cache, &step, &step.lists, &sampled)?
            else {
                unreachable!()
            };
            for (arm, out) in ["one_split", "fixed_splits"].iter().zip(&outputs) {
                report.gate(
                    &format!("{arm} {context} tokens {rows} rows"),
                    &decode(&pick(out, &sampled, HEADS * LATENT)),
                    &reference,
                    DType::Bf16,
                );
            }
            let differing = outputs[0]
                .iter()
                .zip(&outputs[1])
                .filter(|(a, b)| a != b)
                .count();
            let kernels = [&one, &split];
            let graphs = [0, 1].map(|arm| {
                session.capture(|stream| {
                    for _ in 0..LAUNCHES {
                        buffers[arm].launch(stream, kernels[arm], &cache, prepared, true)?;
                    }
                    Ok(())
                })
            });
            let [a, b] = graphs;
            let graphs = [a?, b?];
            let paired = kernel_lab::time_ab(
                session,
                3,
                15,
                [kernel_lab::eviction::EvictionMode::Write; 2],
                |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
            )?;
            let per = |ms: f32| f64::from(ms) * 1e3 / LAUNCHES as f64;
            let row = json!({
                "context": context,
                "rows": rows,
                "one_split_us": per(paired.arms[0].median_ms),
                "fixed_splits_us": per(paired.arms[1].median_ms),
                "fixed_over_one_split": paired.paired_ratio,
                "groups": split.entry.groups(prepared),
                "elements_differing": differing,
            });
            eprintln!("latent drafter attention timing: {row}");
            rows_out.push(row);
        }
    }
    report.finish(
        session,
        BTreeMap::from([("timing".to_owned(), json!(rows_out))]),
    )
}

// ------------------------------------------------- groups timing ----

/// The verify entry's attention at each CTA grouping of its 64 fixed splits
/// against the planner's grouping, at 3K- and 32K-token contexts and 1 to
/// 64 rows: every grouping's merged output equals the planner's bit for bit
/// (the grouping only decides which CTA computes a split), and each
/// grouping's attention launches are timed against the planner's (graphs of
/// eight PDL launches, attention only). The split merge and the column
/// merge (`latent::merge_columns`, from its lab cubin) are timed on the same
/// workspace, after checking their outputs are equal.
pub fn attention_groups_timing(session: &Session, kernels: &Kernels) -> LabResult<()> {
    use kernels_sm120::latent::merge_columns;
    const LAUNCHES: usize = 8;
    let kv = LatentKvFormat::Fp8E4M3Block128;
    let kernel = Kernel::load(kernels, "attn_fp8_verify")?;
    let columns_entry = merge_columns::Entry::of(&kernel.entry).map_err(|e| e.0.to_string())?;
    let columns_module = session.module(&merge_columns::cubin_name(latent_shapes::PROFILE, ""))?;
    let columns = columns_module
        .function(&columns_entry.symbol())
        .map_err(|e| e.to_string())?;
    let merge = kernel
        .merge
        .as_ref()
        .ok_or("the verify entry has a merge")?;
    let mut rows_out = Vec::new();
    let report = Report::new(kernels, "attention_groups_timing");
    for context in [3000_usize, 32_768] {
        let mut rng = Rng::new(0x9a0b_0000 + context as u64);
        let cache = Cache::new(session, kernels, kv, &[context], &mut rng, |rng, n| {
            rng.bf16(n, 1.0)
        })?;
        for rows in [1_usize, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64] {
            let step = Step::new(&mut rng, &[context], rows, false);
            let prepared = rows as u32;
            let served = kernel.entry.groups(prepared);
            let workspace = kernel.entry.workspace_bytes(prepared);
            let buffers = StepBuffers::new(session, &step, &step.lists, workspace, prepared)?;
            let pointers = BTreeMap::from([
                ("q", &buffers.q),
                ("pool", &cache.d_pool),
                ("block_tables", &cache.d_tables),
                ("row_batch", &buffers.row_batch),
                ("key_lists", &buffers.lists),
                ("live_rows", &buffers.live),
                ("workspace", &buffers.workspace),
                ("out", &buffers.out),
                ("out_max", &buffers.out_max),
                ("out_den", &buffers.out_den),
                ("out_num", &buffers.out_num),
            ]);
            let scalars = BTreeMap::from([
                ("max_pages", cache.max_pages as u64),
                ("page_stride_bytes", cache.page_stride as u64),
                ("budget", MAX_KEYS as u64),
                ("splits", u64::from(kernel.entry.splits())),
                ("scale_bits", u64::from(GEOMETRY.softmax_scale.to_bits())),
            ]);
            let main_args = bind(&kernel.entry.params(), &pointers, &scalars)?;
            let merge_args = bind(&kernel.entry.merge_params(), &pointers, &scalars)?;
            let attend = |stream: &Stream, groups: u32| -> LabResult<()> {
                let mut plan = kernel.entry.plan(prepared).map_err(|e| e.0.to_string())?;
                plan.grid[2] = groups;
                kernel
                    .main
                    .launch(stream, launch::config(plan)?, &main_args)
                    .map_err(|e| e.to_string())
            };
            let merge_plan = kernel
                .entry
                .merge_plan(prepared)
                .map_err(|e| e.0.to_string())?
                .ok_or("the verify entry merges")?;
            let columns_plan = columns_entry.plan(prepared).map_err(|e| e.0.to_string())?;
            let merged = |columns_merge: bool| -> LabResult<Vec<u16>> {
                if columns_merge {
                    columns
                        .launch(&session.stream, launch::config(columns_plan)?, &merge_args)
                        .map_err(|e| e.to_string())?;
                } else {
                    merge
                        .launch(&session.stream, launch::config(merge_plan)?, &merge_args)
                        .map_err(|e| e.to_string())?;
                }
                session.stream.synchronize().map_err(|e| e.to_string())?;
                buffers.normalized(session, rows)
            };
            attend(&session.stream, served)?;
            let reference = merged(false)?;
            if merged(true)? != reference {
                return Err(format!(
                    "{context} tokens {rows} rows: the column merge differs from the split merge"
                ));
            }
            let graph = |groups: u32| {
                session.capture(|stream| {
                    for _ in 0..LAUNCHES {
                        attend(stream, groups)?;
                    }
                    Ok(())
                })
            };
            let served_graph = graph(served)?;
            let per = |ms: f32| f64::from(ms) * 1e3 / LAUNCHES as f64;
            let mut groupings = Vec::new();
            for groups in [64_u32, 32, 16, 8, 4, 2, 1] {
                attend(&session.stream, groups)?;
                if merged(false)? != reference {
                    return Err(format!(
                        "{context} tokens {rows} rows: {groups} groups change the output"
                    ));
                }
                let candidate = graph(groups)?;
                let graphs = [&served_graph, &candidate];
                let paired = kernel_lab::time_ab(
                    session,
                    3,
                    15,
                    [kernel_lab::eviction::EvictionMode::Write; 2],
                    |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                )?;
                groupings.push(json!({
                    "groups": groups,
                    "ctas": rows as u32 * groups,
                    "attention_us": per(paired.arms[1].median_ms),
                    "over_served": paired.paired_ratio,
                }));
            }
            let merge_graphs = [false, true].map(|columns_merge| {
                session.capture(|stream| {
                    for _ in 0..LAUNCHES {
                        if columns_merge {
                            columns
                                .launch(stream, launch::config(columns_plan)?, &merge_args)
                                .map_err(|e| e.to_string())?;
                        } else {
                            merge
                                .launch(stream, launch::config(merge_plan)?, &merge_args)
                                .map_err(|e| e.to_string())?;
                        }
                    }
                    Ok(())
                })
            });
            let [a, b] = merge_graphs;
            let merge_graphs = [a?, b?];
            let merges = kernel_lab::time_ab(
                session,
                3,
                15,
                [kernel_lab::eviction::EvictionMode::Write; 2],
                |arm, stream| merge_graphs[arm].launch(stream).map_err(|e| e.to_string()),
            )?;
            let row = json!({
                "context": context,
                "rows": rows,
                "served_groups": served,
                "groupings": groupings,
                "split_merge_us": per(merges.arms[0].median_ms),
                "column_merge_us": per(merges.arms[1].median_ms),
            });
            eprintln!("latent attention groups timing: {row}");
            rows_out.push(row);
        }
    }
    report.finish(
        session,
        BTreeMap::from([("timing".to_owned(), json!(rows_out))]),
    )
}

// ------------------------------------------------ drafter accuracy ----

/// The BF16 value nearest `x` (ties to even), rounded once from f64.
fn bf16_nearest(x: f64) -> u16 {
    let guess = f32_to_bf16(x as f32);
    if !x.is_finite() {
        return guess;
    }
    let candidates = [guess.wrapping_sub(1), guess, guess.wrapping_add(1)];
    let mut best = guess;
    let mut best_error = f64::INFINITY;
    for c in candidates {
        let v = f64::from(bf16_to_f32(c));
        if !v.is_finite() {
            continue;
        }
        let error = (v - x).abs();
        if error < best_error || (error == best_error && c & 1 == 0) {
            best = c;
            best_error = error;
        }
    }
    best
}

/// Error statistics of outputs against f64 oracle values.
#[derive(Clone, Copy, Debug, Default)]
struct Errors {
    elements: u64,
    max_abs: f64,
    sum_abs: f64,
    max_rel: f64,
    sum_rel: f64,
    rel_elements: u64,
    /// BF16 outputs other than the correctly rounded oracle value.
    misrounded: u64,
    /// Non-finite outputs (or oracle values), kept out of the error sums
    /// (`f64::max` would drop a NaN) and gated to zero.
    nonfinite: u64,
}

impl Errors {
    fn add(&mut self, actual: f64, reference: f64, rounded: Option<bool>) {
        self.elements += 1;
        if rounded == Some(false) {
            self.misrounded += 1;
        }
        if !actual.is_finite() || !reference.is_finite() {
            self.nonfinite += 1;
            return;
        }
        let error = (actual - reference).abs();
        self.max_abs = self.max_abs.max(error);
        self.sum_abs += error;
        if reference != 0.0 {
            let rel = error / reference.abs();
            self.max_rel = self.max_rel.max(rel);
            self.sum_rel += rel;
            self.rel_elements += 1;
        }
    }

    fn merge(&mut self, other: &Errors) {
        self.elements += other.elements;
        self.max_abs = self.max_abs.max(other.max_abs);
        self.sum_abs += other.sum_abs;
        self.max_rel = self.max_rel.max(other.max_rel);
        self.sum_rel += other.sum_rel;
        self.rel_elements += other.rel_elements;
        self.misrounded += other.misrounded;
        self.nonfinite += other.nonfinite;
    }

    fn mean_abs(&self) -> f64 {
        self.sum_abs / (self.elements - self.nonfinite).max(1) as f64
    }

    fn mean_rel(&self) -> f64 {
        self.sum_rel / self.rel_elements.max(1) as f64
    }

    fn json(&self) -> Value {
        json!({
            "elements": self.elements,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs(),
            "max_rel": self.max_rel,
            "mean_rel": self.mean_rel(),
            "nonzero_reference_elements": self.rel_elements,
            "misrounded_bf16": self.misrounded,
            "nonfinite": self.nonfinite,
        })
    }
}

/// One BF16 unit in the last place at `x` (the subnormal step below the
/// normal range).
fn bf16_ulp(x: f64) -> f64 {
    let exponent = if x == 0.0 {
        -126
    } else {
        (x.abs().log2().floor() as i32).max(-126)
    };
    2_f64.powi(exponent - 7)
}

/// The arms of the drafter accuracy comparison, in output order.
const ACCURACY_ARMS: [&str; 2] = ["one split", "fixed splits"];

/// The paired verdict on one case's normalized (BF16) outputs of both arms
/// against the f64 oracle, `width` outputs per (row, head) vector. Fails
/// when either arm has an output outside the decision-0081 bound
/// (`check_numeric`, which also refuses a non-finite output or a length
/// mismatch), or when the fixed splits' worst error exceeds one split's by
/// more than one BF16 rounding step: the max absolute error by more than
/// one BF16 ulp of the largest reference magnitude, the max relative error
/// over significant elements (at least 2^-8 of their vector's largest
/// reference magnitude; smaller ones are near-cancellation zeros whose
/// relative error only the absolute bound bounds) by more than 2^-7.
fn paired_bf16_verdict(
    reference: &OracleOutput,
    width: usize,
    one: &[u16],
    fixed: &[u16],
) -> (Value, Vec<String>) {
    let mut failures = Vec::new();
    let mut arms = Vec::new();
    for (name, out) in ACCURACY_ARMS.into_iter().zip([one, fixed]) {
        let values = out
            .iter()
            .map(|&bits| f64::from(bf16_to_f32(bits)))
            .collect::<Vec<_>>();
        if let Err(worst) = check_numeric(&values, reference, DType::Bf16) {
            failures.push(format!("{name}: outside the oracle bound: {worst:?}"));
        }
        let (mut max_abs, mut max_rel, mut bound_use) = (0_f64, 0_f64, 0_f64);
        for (vector, (actual, expected)) in values
            .chunks(width)
            .zip(reference.values.chunks(width))
            .enumerate()
        {
            let largest = expected.iter().fold(0_f64, |m, r| m.max(r.abs()));
            for (i, (&a, &r)) in actual.iter().zip(expected).enumerate() {
                // A NaN error counts as infinite, so it cannot hide in a max.
                let error = if (a - r).is_nan() {
                    f64::INFINITY
                } else {
                    (a - r).abs()
                };
                max_abs = max_abs.max(error);
                if r != 0.0 && r.abs() >= largest * 2_f64.powi(-8) {
                    max_rel = max_rel.max(error / r.abs());
                }
                let bound =
                    kernel_core::oracle::numeric_bound(reference, vector * width + i, DType::Bf16);
                bound_use = bound_use.max(error / bound);
            }
        }
        arms.push((max_abs, max_rel, bound_use));
    }
    let largest = reference.values.iter().fold(0_f64, |m, r| m.max(r.abs()));
    let (abs_slack, rel_slack) = (bf16_ulp(largest), 2_f64.powi(-7));
    let [(one_abs, one_rel, _), (fixed_abs, fixed_rel, _)] = [arms[0], arms[1]];
    // Within the slack, comparably: a NaN on either side fails.
    let within = |fixed: f64, limit: f64| {
        matches!(
            fixed.partial_cmp(&limit),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        )
    };
    if !within(fixed_abs, one_abs + abs_slack) {
        failures.push(format!(
            "fixed splits max abs error {fixed_abs:e} exceeds one split's {one_abs:e} by more \
             than one BF16 ulp ({abs_slack:e})"
        ));
    }
    if !within(fixed_rel, one_rel + rel_slack) {
        failures.push(format!(
            "fixed splits max relative error {fixed_rel:e} exceeds one split's {one_rel:e} by \
             more than 2^-7"
        ));
    }
    let arm = |(max_abs, max_rel, bound_use): (f64, f64, f64)| {
        json!({
            "max_abs": max_abs,
            "max_rel_significant": max_rel,
            "max_error_over_bound": bound_use,
        })
    };
    let summary = json!({
        "one_split": arm(arms[0]),
        "fixed_splits": arm(arms[1]),
        "max_abs_slack": abs_slack,
        "max_rel_slack": rel_slack,
        "passed": failures.is_empty(),
    });
    (summary, failures)
}

/// Negative and positive controls of [`paired_bf16_verdict`] and [`Errors`]
/// on a synthetic oracle (CPU): one misrounded output passes; one gross
/// error, one NaN, one infinity or a short output fails; a NaN is counted
/// as non-finite, not dropped from the maxima.
pub fn accuracy_verdict_controls() -> Result<(), String> {
    let width = 16;
    let values = (0..64)
        .map(|i| (f64::from(i) * 0.37 - 11.0) * 2_f64.powi(i % 5 - 2) + 1.0 / 3.0)
        .collect::<Vec<_>>();
    // An attention-like reduction: K = 2048 keys, so one BF16 rounding
    // step lies within the decision-0081 bound and 1023 far outside it.
    let sums = values
        .iter()
        .map(|v| 16.0 * v.abs() + 1.0)
        .collect::<Vec<_>>();
    let reference = OracleOutput::new(values.clone(), sums, 2048).map_err(|e| e.0.to_string())?;
    let one = values.iter().map(|&v| bf16_nearest(v)).collect::<Vec<_>>();
    let with = |index: usize, bits: u16| {
        let mut out = one.clone();
        out[index] = bits;
        out
    };
    // The exact value's other BF16 neighbour, as a misrounding kernel gives.
    let side = |bits: u16| (f64::from(bf16_to_f32(bits)) - values[7]).signum();
    let other = [one[7].wrapping_add(1), one[7].wrapping_sub(1)]
        .into_iter()
        .find(|&bits| side(bits) != side(one[7]))
        .ok_or("no other neighbour")?;
    let misrounded = with(7, other);
    let gross = with(9, f32_to_bf16(bf16_to_f32(one[9]) + 1023.0));
    let cases: [(&str, Vec<u16>, bool); 5] = [
        ("one misrounded output", misrounded, true),
        ("gross error of 1023", gross, false),
        ("NaN output", with(3, 0x7fc0), false),
        ("infinite output", with(5, 0x7f80), false),
        ("short output", one[..63].to_vec(), false),
    ];
    for (name, fixed, pass) in cases {
        let (summary, failures) = paired_bf16_verdict(&reference, width, &one, &fixed);
        if failures.is_empty() != pass {
            return Err(format!(
                "{name}: expected {}, got {failures:?} ({summary})",
                if pass { "a pass" } else { "a failure" }
            ));
        }
    }
    let mut errors = Errors::default();
    errors.add(1.0, 1.0, Some(true));
    errors.add(f64::NAN, 2.0, Some(false));
    if errors.nonfinite != 1 || errors.max_abs.is_nan() || errors.elements != 2 {
        return Err(format!(
            "a NaN output was not counted as non-finite: {errors:?}"
        ));
    }
    Ok(())
}

/// One accuracy case of the drafter's attention: its data and key lists.
struct AccuracyCase {
    name: &'static str,
    context: usize,
    rows: usize,
    holes: bool,
    /// Latent rows: `rng` and the element count.
    latent: fn(&mut Rng, usize) -> Vec<u16>,
    /// Multiplies every query element.
    query_scale: f64,
    /// Rewrites each row's key list (entries `[0, MAX_KEYS)`).
    lists: Option<fn(&mut [i32])>,
}

fn normal_latent(rng: &mut Rng, n: usize) -> Vec<u16> {
    rng.bf16(n, 1.0)
}

fn cancelling_latent(rng: &mut Rng, n: usize) -> Vec<u16> {
    rng.cancelling(n)
}

/// Each 128-wide block of a latent row at its own scale, 2^-12 to 2^9, so
/// FP8 pages hold block scales far apart and BF16 sums mix magnitudes.
fn block_scaled_latent(rng: &mut Rng, n: usize) -> Vec<u16> {
    let values = rng.bf16(n, 1.0);
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let block = (i % LATENT) / 128;
            let scale = 2_f64.powi(7 * block as i32 - 12);
            f32_to_bf16((f64::from(bf16_to_f32(v)) * scale) as f32)
        })
        .collect()
}

/// Only the last 40 listed keys stay, so most fixed splits hold holes.
fn late_keys(list: &mut [i32]) {
    for entry in &mut list[..MAX_KEYS - 40] {
        *entry = -1;
    }
}

/// One listed key, in the middle of the list.
fn one_key(list: &mut [i32]) {
    let keep = list[1000];
    list.fill(-1);
    list[1000] = keep;
}

fn accuracy_cases() -> Vec<AccuracyCase> {
    let case = |name, context, holes, latent, query_scale, lists| AccuracyCase {
        name,
        context,
        rows: 8,
        holes,
        latent,
        query_scale,
        lists,
    };
    vec![
        // Representative: the served shapes' data.
        case(
            "random_3k",
            3000,
            false,
            normal_latent as fn(&mut Rng, usize) -> Vec<u16>,
            1.0,
            None,
        ),
        case("random_holes_32k", 32_768, true, normal_latent, 1.0, None),
        // Adversarial.
        case("peaky_scores", 3000, false, normal_latent, 16.0, None),
        case("flat_scores", 3000, false, normal_latent, 1.0 / 64.0, None),
        case(
            "cancelling_values",
            3000,
            false,
            cancelling_latent,
            0.25,
            None,
        ),
        case(
            "block_scales",
            3000,
            false,
            block_scaled_latent,
            0.0625,
            None,
        ),
        case(
            "late_keys",
            3000,
            false,
            normal_latent,
            1.0,
            Some(late_keys as fn(&mut [i32])),
        ),
        case("one_key", 3000, false, normal_latent, 1.0, Some(one_key)),
    ]
}

/// The drafter's latent attention before and after PERF-B14b against the
/// f64 oracle (Accuracy decision v2): for FP8 and BF16 pages and every
/// case of [`accuracy_cases`], the one-split prefill entry (before) and the
/// decode entry's 64 fixed splits with their merge (after) run the same
/// step, and every output element of every row is compared with the oracle:
/// max and mean absolute and relative error, and the BF16 outputs other
/// than the correctly rounded oracle value. The partial (context-parallel)
/// forms' F32 `o / l` are compared the same way, without the rounding
/// count. Fails when, in any case, either arm's normalized output or
/// partial (`gate_partial`) lies outside the decision-0081 bound or is not
/// finite, or the fixed splits' normalized max absolute or relative error
/// exceeds one split's by more than one BF16 rounding step
/// ([`paired_bf16_verdict`]); or when, per page format over all cases, the
/// fixed splits misround more BF16 outputs than one split or any output is
/// not finite. The F32 partials' error ratios are reported, not compared
/// (lab run 20261009T113942Z: about 1% higher mean absolute error with
/// fixed splits, from the split merge's rescales; context-parallel drafter
/// passes keep one split, `Lowering::attention_phase`).
pub fn drafter_attention_accuracy(session: &Session, kernels: &Kernels) -> LabResult<()> {
    let mut report = Report::new(kernels, "drafter_attention_accuracy");
    let mut cases_out = Vec::new();
    let mut verdicts = Vec::new();
    for (seed, kv) in [
        (0xacc0_0000_u64, LatentKvFormat::Fp8E4M3Block128),
        (0xacc1_0000, LatentKvFormat::Bf16),
    ] {
        let dtype = dtype_name(kv);
        let arms = [
            Kernel::load(kernels, &format!("attn_{dtype}_prefill"))?,
            Kernel::load(kernels, &format!("attn_{dtype}_decode"))?,
        ];
        let partial_arms = [
            Kernel::load(kernels, &format!("attn_{dtype}_prefill_partial"))?,
            Kernel::load(kernels, &format!("attn_{dtype}_decode_partial"))?,
        ];
        let mut totals = [Errors::default(); 2];
        let mut partial_totals = [Errors::default(); 2];
        for (index, case) in accuracy_cases().iter().enumerate() {
            let mut rng = Rng::new(seed + index as u64);
            let cache = Cache::new(session, kernels, kv, &[case.context], &mut rng, case.latent)?;
            let mut step = Step::new(&mut rng, &[case.context], case.rows, case.holes);
            for q in &mut step.q {
                *q = f32_to_bf16((f64::from(bf16_to_f32(*q)) * case.query_scale) as f32);
            }
            if let Some(rewrite) = case.lists {
                for row in step.lists.chunks_mut(MAX_KEYS) {
                    rewrite(row);
                }
            }
            let rows = (0..step.rows).collect::<Vec<_>>();
            let prepared = step.rows as u32;
            let workspace = arms[1].entry.workspace_bytes(prepared);
            let LatentAttentionResult::Normalized(reference) =
                oracle(&arms[0], &cache, &step, &step.lists, &rows)?
            else {
                unreachable!()
            };
            let LatentAttentionResult::Partial(reference_parts) =
                oracle(&partial_arms[0], &cache, &step, &step.lists, &rows)?
            else {
                unreachable!()
            };
            let mut errors = [Errors::default(); 2];
            let mut outputs = [Vec::new(), Vec::new()];
            for (arm, kernel) in arms.iter().enumerate() {
                let buffers = StepBuffers::new(session, &step, &step.lists, workspace, prepared)?;
                buffers.launch(&session.stream, kernel, &cache, prepared, false)?;
                let out = buffers.normalized(session, step.rows)?;
                for (&bits, &r) in out.iter().zip(&reference.values) {
                    let rounded = bits == bf16_nearest(r);
                    errors[arm].add(f64::from(bf16_to_f32(bits)), r, Some(rounded));
                }
                totals[arm].merge(&errors[arm]);
                outputs[arm] = out;
            }
            let (paired, failures) =
                paired_bf16_verdict(&reference, LATENT, &outputs[0], &outputs[1]);
            report.exact(
                &format!(
                    "{dtype} {} normalized: oracle bound and paired max errors",
                    case.name
                ),
                if failures.is_empty() {
                    Ok(())
                } else {
                    Err(failures.join("; "))
                },
            );
            let mut partial_errors = [Errors::default(); 2];
            for (arm, kernel) in partial_arms.iter().enumerate() {
                let buffers = StepBuffers::new(session, &step, &step.lists, workspace, prepared)?;
                buffers.launch(&session.stream, kernel, &cache, prepared, false)?;
                let parts = buffers.partial(session, step.rows)?;
                gate_partial(
                    &mut report,
                    &format!("{dtype} {} {} partial", case.name, ACCURACY_ARMS[arm]),
                    &parts,
                    &reference_parts,
                );
                for (pair, &l) in parts.denominator.iter().enumerate() {
                    for d in 0..LATENT {
                        let o = f64::from(parts.numerator[pair * LATENT + d]);
                        let actual = if l == 0.0 { 0.0 } else { o / f64::from(l) };
                        partial_errors[arm].add(actual, reference.values[pair * LATENT + d], None);
                    }
                }
                partial_totals[arm].merge(&partial_errors[arm]);
            }
            cases_out.push(json!({
                "kv": dtype,
                "case": case.name,
                "context": case.context,
                "rows": case.rows,
                "one_split": errors[0].json(),
                "fixed_splits": errors[1].json(),
                "paired": paired,
                "one_split_partial": partial_errors[0].json(),
                "fixed_splits_partial": partial_errors[1].json(),
            }));
            eprintln!(
                "drafter attention accuracy {dtype} {}: one split mean abs {:.3e} max abs {:.3e} \
                 misrounded {}; fixed splits mean abs {:.3e} max abs {:.3e} misrounded {}; \
                 partial mean abs {:.3e} / {:.3e}",
                case.name,
                errors[0].mean_abs(),
                errors[0].max_abs,
                errors[0].misrounded,
                errors[1].mean_abs(),
                errors[1].max_abs,
                errors[1].misrounded,
                partial_errors[0].mean_abs(),
                partial_errors[1].mean_abs(),
            );
        }
        for (form, [one, fixed]) in [("normalized", totals), ("partial", partial_totals)] {
            // BF16 outputs: v2's count against the correctly rounded oracle
            // value. F32 partials: the mean absolute error, reported.
            let gated = form == "normalized";
            let no_less_accurate = if gated {
                fixed.misrounded <= one.misrounded
            } else {
                fixed.mean_abs() <= one.mean_abs()
            };
            let verdict = json!({
                "kv": dtype,
                "form": form,
                "one_split": one.json(),
                "fixed_splits": fixed.json(),
                "mean_abs_ratio": fixed.mean_abs() / one.mean_abs(),
                "mean_rel_ratio": fixed.mean_rel() / one.mean_rel(),
                "gated": gated,
                "fixed_splits_no_less_accurate": no_less_accurate,
            });
            eprintln!("drafter attention accuracy total: {verdict}");
            report.exact(
                &format!("{dtype} {form}: every output finite"),
                if one.nonfinite + fixed.nonfinite == 0 {
                    Ok(())
                } else {
                    Err(format!("{verdict}"))
                },
            );
            if gated {
                report.exact(
                    &format!("{dtype} {form}: fixed splits misround no more BF16 outputs"),
                    if no_less_accurate {
                        Ok(())
                    } else {
                        Err(format!("{verdict}"))
                    },
                );
            }
            verdicts.push(verdict);
        }
    }
    report.finish(
        session,
        BTreeMap::from([
            ("cases".to_owned(), json!(cases_out)),
            ("totals".to_owned(), json!(verdicts)),
        ]),
    )
}
