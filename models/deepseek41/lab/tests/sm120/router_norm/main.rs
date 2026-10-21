//! Qualification of the router-in-norm prefill (../../../router_norm.md):
//! the served MoE-input norm, zero-component prepare and router projection
//! against the certifying norm and the certified projection, bit for bit on
//! every norm output and the router logits, over synthetic rows. GPU tests
//! are ignored by default and read the staged lab cubin named by
//! CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS; with
//! CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE=1 they run on the lab's host
//! emulation instead (a check of the job, not evidence).
#[path = "../../../sm120/router_norm.rs"]
#[allow(dead_code)]
mod family;

mod emulate;
mod wide;

use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{LabResult, Session};
use kernels_sm120::f32_projection::{certified, zero_prefill as zero};
use kernels_sm120::row::rmsnorm_prefill::Mutation;
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use std::collections::BTreeMap;

pub const CUBINS: &str = "CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS";
pub const EMULATE: &str = "CHOPINFER_DEEPSEEK41_ROUTER_NORM_EMULATE";
pub const MEMCHECK: &str = "CHOPINFER_DEEPSEEK41_ROUTER_NORM_MEMCHECK";
pub const K: u32 = 5120;
pub const N: u32 = 384;
pub const CAPACITY: u32 = 8192;
pub const LIVES: [u32; 6] = [1, 65, 513, 2048, 8191, 8192];
pub const MEMCHECK_LIVES: [u32; 3] = [1, 65, 2048];
pub const TIMED_ROWS: [u32; 3] = [8192, 2048, 512];

pub fn emulated() -> bool {
    std::env::var(EMULATE).as_deref() == Ok("1")
}

/// Device buffers by handle, host buffers under emulation, or handles alone
/// for a plan.
pub struct Arena<'a> {
    pub session: Option<&'a Session>,
    pub host: Option<BTreeMap<u64, Vec<u8>>>,
    pub next: u64,
    pub buffers: BTreeMap<u64, DeviceBuffer>,
}
impl<'a> Arena<'a> {
    pub fn device(session: &'a Session) -> Self {
        Self {
            session: Some(session),
            host: None,
            next: 0,
            buffers: BTreeMap::new(),
        }
    }
    pub fn host() -> Self {
        Self {
            session: None,
            host: Some(BTreeMap::new()),
            next: 0,
            buffers: BTreeMap::new(),
        }
    }
    pub fn bytes(&mut self, bytes: &[u8]) -> LabResult<View> {
        self.next += 1;
        if let Some(session) = self.session {
            let b = session.allocate(bytes.len().max(1))?;
            b.copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            self.buffers.insert(self.next, b);
        } else if let Some(host) = &mut self.host {
            host.insert(self.next, bytes.to_vec());
        }
        Ok(View {
            handle: self.next,
            bytes: bytes.len() as u64,
        })
    }
    pub fn filled(&mut self, bytes: u64, value: u8) -> LabResult<View> {
        self.bytes(&vec![value; bytes as usize])
    }
    pub fn read(&self, v: View) -> LabResult<Vec<u8>> {
        if let Some(host) = &self.host {
            return Ok(host[&v.handle].clone());
        }
        let mut out = vec![0; v.bytes as usize];
        self.buffers[&v.handle]
            .copy_to_bytes(&mut out, &self.session.unwrap().stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
    pub fn write(&mut self, v: View, bytes: &[u8]) -> LabResult<()> {
        if let Some(host) = &mut self.host {
            host.insert(v.handle, bytes.to_vec());
            return Ok(());
        }
        self.buffers[&v.handle]
            .copy_from_bytes(bytes, &self.session.unwrap().stream)
            .map_err(|e| e.to_string())
    }
}

/// Serving symbols from the bundle's embedded cubins, lab symbols from the
/// staged lab cubin; under emulation, the host model of each launch.
pub struct Runner<'a> {
    pub session: Option<&'a Session>,
    lab: Option<Module>,
    embedded: BTreeMap<&'static str, Module>,
    functions: BTreeMap<String, Function>,
}
impl<'a> Runner<'a> {
    pub fn new(session: Option<&'a Session>) -> LabResult<Self> {
        let lab = match session {
            Some(session) => {
                let dir = std::path::PathBuf::from(
                    std::env::var(CUBINS).map_err(|e| format!("{CUBINS}: {e}"))?,
                );
                let bytes = std::fs::read(dir.join(format!("{}.cubin", family::CUBIN)))
                    .map_err(|e| e.to_string())?;
                Some(session.load_cubin(&bytes)?)
            }
            None => None,
        };
        Ok(Self {
            session,
            lab,
            embedded: BTreeMap::new(),
            functions: BTreeMap::new(),
        })
    }
    pub fn serving(symbol: &str) -> bool {
        bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).is_some()
    }
    pub fn ensure(&mut self, ops: &[Launch]) -> LabResult<()> {
        let Some(session) = self.session else {
            return Ok(());
        };
        for op in ops {
            if self.functions.contains_key(&op.symbol) {
                continue;
            }
            let function = if let Some((name, bytes)) =
                bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
            {
                if !self.embedded.contains_key(name) {
                    self.embedded.insert(name, session.load_cubin(bytes)?);
                }
                self.embedded[name].function(&op.symbol)
            } else {
                self.lab.as_ref().unwrap().function(&op.symbol)
            }
            .map_err(|e| format!("{}: {e}", op.symbol))?;
            if op.plan.dynamic_shared_bytes > 0 {
                function
                    .set_max_dynamic_shared_memory(op.plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            self.functions.insert(op.symbol.clone(), function);
        }
        Ok(())
    }
    pub fn issue(&self, a: &Arena<'_>, ops: &[Launch], stream: &Stream) -> LabResult<()> {
        let bindings = Bindings {
            buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
        };
        for op in ops {
            launch(
                &self.session.unwrap().context,
                &self.functions[&op.symbol],
                stream,
                op.plan,
                &op.arguments,
                &bindings,
            )?;
        }
        Ok(())
    }
    pub fn run(&mut self, a: &mut Arena<'_>, ops: &[Launch]) -> LabResult<()> {
        if self.session.is_none() {
            return emulate::run(a, ops);
        }
        self.ensure(ops)?;
        let session = self.session.unwrap();
        self.issue(a, ops, &session.stream)?;
        session.stream.synchronize().map_err(|e| e.to_string())
    }
    /// One graph replay of `ops` (a plain run under emulation).
    pub fn replay(&mut self, a: &mut Arena<'_>, ops: &[Launch]) -> LabResult<()> {
        let Some(session) = self.session else {
            return emulate::run(a, ops);
        };
        self.ensure(ops)?;
        let graph = session.capture(|s| self.issue(a, ops, s))?;
        graph.launch(&session.stream).map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())
    }
}

pub fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
pub fn widen(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}
/// `exact_f32::tensor_operand`.
pub fn tensor_operand(x: f32) -> bool {
    let bits = x.to_bits() & 0x7fff_ffff;
    bits == 0 || ((97 << 23)..(158 << 23)).contains(&bits)
}
pub fn u32s(v: &[u8]) -> Vec<u32> {
    v.chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pattern {
    /// Pseudo-random rows; every 61st row from the 11th all zero.
    Clean,
    /// Clean, plus every 37th row from the 3rd with one 1e15 element (the
    /// rest normalize below the tensor domain), every 53rd from the 7th
    /// with a NaN, and the [`split_row`]s.
    Outliers,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Prior {
    Clear,
    /// Every 97th row from the 5th already flagged, as an earlier stage
    /// leaves it.
    Flagged,
}

pub struct Rng(u64);
impl Rng {
    pub fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Roughly normal, from the sum of four uniforms.
    pub fn normal(&mut self) -> f32 {
        (0..4)
            .map(|_| (self.step() >> 40) as f32 / (1u64 << 24) as f32)
            .sum::<f32>()
            * 1.732
            - 3.464
    }
}

/// Outlier rows whose odd elements are scaled by 2^-40, so they normalize
/// below the tensor domain while the even ones keep many significant
/// terms: every 41st row from the 9th that is not a zero, 1e15 or NaN row.
/// Only these uncertified rows are certain to sum differently on the
/// tensor chain than on the SIMT path; a 1e15 row has one significant
/// term and a NaN row is NaN on both.
pub fn split_row(r: usize) -> bool {
    r % 41 == 9 && r % 61 != 11 && r % 37 != 3 && r % 53 != 7
}

pub fn input_rows(pattern: Pattern) -> Vec<u16> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut rows = vec![0u16; (CAPACITY * K) as usize];
    for r in 0..CAPACITY as usize {
        let row = &mut rows[r * K as usize..(r + 1) * K as usize];
        if r % 61 == 11 {
            continue;
        }
        for x in row.iter_mut() {
            *x = bf16(rng.normal());
        }
        if pattern == Pattern::Outliers {
            if split_row(r) {
                for x in row.iter_mut().skip(1).step_by(2) {
                    *x = bf16(widen(*x) * (-40f32).exp2());
                }
            }
            if r % 37 == 3 {
                row[r % K as usize] = bf16(1e15);
            }
            if r % 53 == 7 {
                row[(r * 7) % K as usize] = 0x7fc0;
            }
        }
    }
    rows
}
pub fn norm_weight() -> Vec<u16> {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    (0..K).map(|_| bf16(1.0 + 0.25 * rng.normal())).collect()
}
/// Router weights; experts 80 to 95 are below the tensor domain, so those
/// columns take the SIMT path in both arms.
pub fn router_weight() -> Vec<u16> {
    let mut rng = Rng(0x1405_7b7e_f767_814f);
    let mut w = vec![0u16; (N * K) as usize];
    for f in 0..N as usize {
        for k in 0..K as usize {
            let scale = if (80..96).contains(&f) { 1e-20 } else { 0.02 };
            w[f * K as usize + k] = bf16(scale * rng.normal());
        }
    }
    w
}
/// `exact_f32::classify_weights`: one flag per 16 features.
pub fn weight_domain(w: &[u16]) -> Vec<u32> {
    (0..N.div_ceil(16) as usize)
        .map(|g| {
            u32::from((g * 16..(g * 16 + 16).min(N as usize)).all(|f| {
                w[f * K as usize..(f + 1) * K as usize]
                    .iter()
                    .all(|&x| tensor_operand(widen(x)))
            }))
        })
        .collect()
}
pub fn prior_status(prior: Prior) -> Vec<u32> {
    (0..CAPACITY)
        .map(|r| u32::from(prior == Prior::Flagged && r >= 5 && (r - 5) % 97 == 0))
        .collect()
}

/// One arm's outputs.
#[derive(Clone, Copy)]
pub struct Outputs {
    pub normed: View,
    pub widened: View,
    pub codes: View,
    pub scales: View,
    pub status: View,
    pub logits: View,
    /// The served zero workspace, or the certified arm's flags.
    pub rows: View,
}
impl Outputs {
    pub fn compared(&self) -> [View; 6] {
        [
            self.normed,
            self.widened,
            self.codes,
            self.scales,
            self.status,
            self.logits,
        ]
    }
}
pub const COMPARED: [&str; 6] = ["normed", "widened", "codes", "scales", "status", "logits"];

pub struct Inputs {
    pub input: View,
    pub norm_weight: View,
    pub weight: View,
    pub weight_domain: View,
    pub live: View,
}

pub fn outputs(a: &mut Arena<'_>, served: bool) -> LabResult<Outputs> {
    let rows = u64::from(CAPACITY);
    let e = family::router();
    Ok(Outputs {
        normed: a.filled(rows * u64::from(K) * 2, 0xa5)?,
        widened: a.filled(rows * u64::from(K) * 4, 0xa5)?,
        codes: a.filled(rows * u64::from(K), 0xa5)?,
        scales: a.filled(rows * u64::from(K / 32), 0xa5)?,
        status: a.filled(rows * 4, 0)?,
        logits: a.filled(rows * u64::from(N) * 4, 0xa5)?,
        rows: if served {
            a.filled(zero::workspace_bytes(&e, CAPACITY), 0xa5)?
        } else {
            a.filled(rows * 4, 0xa5)?
        },
    })
}

fn ptr(v: View) -> Argument {
    Argument::DeviceBuffer(v.handle)
}

fn norm_launch(
    entry: &kernels_sm120::row::rmsnorm_prefill::Entry,
    i: &Inputs,
    o: &Outputs,
    rows: u32,
) -> LabResult<Launch> {
    let mut arguments = entry
        .reference
        .arguments(RmsNormQuantizeBuffers {
            input: i.input.handle,
            weight: i.norm_weight.handle,
            normed: o.normed.handle,
            widened: Some(o.widened.handle),
            codes: o.codes.handle,
            scales: o.scales.handle,
            live_rows: i.live.handle,
            status: o.status.handle,
            status_row_offset: 0,
        })
        .map_err(|e| e.to_string())?;
    if entry.certify {
        arguments.0.push(ptr(o.rows));
    }
    Ok(Launch {
        symbol: entry.reference.symbol.clone(),
        rows,
        plan: entry.plan(rows).map_err(|e| e.to_string())?,
        arguments,
        collective: false,
        persistent: false,
    })
}

fn make(symbol: String, rows: u32, plan: LaunchPlan, arguments: Vec<Argument>) -> Launch {
    Launch {
        symbol,
        rows,
        plan,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    }
}

/// The served arm: the served norm, the zero-component prepare and its
/// 32-row projection, as `router_prefill` launches them.
pub fn served(i: &Inputs, o: &Outputs, rows: u32) -> LabResult<Vec<Launch>> {
    let e = family::router();
    Ok(vec![
        norm_launch(&family::served_norm(), i, o, rows)?,
        make(
            zero::prepare_symbol(&e),
            rows,
            LaunchPlan {
                grid: [rows, 1, 1],
                block: [128, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1; 3],
                pdl: true,
            },
            vec![
                ptr(o.widened),
                ptr(i.live),
                Argument::U32(rows),
                ptr(o.rows),
            ],
        ),
        make(
            zero::symbol(&e, 32),
            rows,
            zero::plan(&e, 32, rows).map_err(|e| e.to_string())?,
            vec![
                ptr(o.widened),
                ptr(i.weight),
                ptr(o.rows),
                ptr(o.logits),
                ptr(i.live),
                Argument::U32(rows),
                ptr(i.weight_domain),
            ],
        ),
    ])
}

/// The candidate arm: the certifying norm (or its `certify_all` mutation)
/// and the bundle's certified projection (64-expert tiles).
pub fn candidate(i: &Inputs, o: &Outputs, rows: u32, mutation: Mutation) -> LabResult<Vec<Launch>> {
    let e = family::router();
    Ok(vec![
        norm_launch(&family::certifying_norm(mutation), i, o, rows)?,
        make(
            family::served_router_symbol(),
            rows,
            certified::wide_plan(&e, family::TILE_ROWS, family::SERVED_FEATURES, rows)
                .map_err(|e| e.to_string())?,
            vec![
                ptr(o.widened),
                ptr(o.normed),
                ptr(o.rows),
                ptr(i.weight),
                ptr(o.logits),
                ptr(i.live),
                Argument::U32(rows),
                ptr(i.weight_domain),
            ],
        ),
    ])
}

/// The served zero workspace's `single` flags of a step of `rows` rows.
pub fn served_single(workspace: &[u8], rows: u32) -> Vec<u32> {
    let components = (u64::from(rows) * 4).next_multiple_of(16);
    let single = components + u64::from(rows) * 3 * u64::from(K) * 2;
    u32s(&workspace[single as usize..(single + u64::from(rows) * 4) as usize])
}

pub fn inputs(a: &mut Arena<'_>, pattern: Pattern, live: u32) -> LabResult<Inputs> {
    let w = router_weight();
    let input: Vec<u8> = input_rows(pattern)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let nw: Vec<u8> = norm_weight().iter().flat_map(|x| x.to_le_bytes()).collect();
    let wb: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let domain: Vec<u8> = weight_domain(&w)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    Ok(Inputs {
        input: a.bytes(&input)?,
        norm_weight: a.bytes(&nw)?,
        weight: a.bytes(&wb)?,
        weight_domain: a.bytes(&domain)?,
        live: a.bytes(&live.to_le_bytes())?,
    })
}

/// Reset an arm's outputs: poison, then the prior status.
pub fn reset(a: &mut Arena<'_>, o: &Outputs, prior: Prior) -> LabResult<()> {
    for v in [o.normed, o.widened, o.codes, o.scales, o.logits, o.rows] {
        a.write(v, &vec![0xa5; v.bytes as usize])?;
    }
    let status: Vec<u8> = prior_status(prior)
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    a.write(o.status, &status)
}

pub fn snapshot(a: &Arena<'_>, o: &Outputs) -> LabResult<Vec<Vec<u8>>> {
    o.compared().iter().map(|v| a.read(*v)).collect()
}

/// Every check of one case; returns its record.
pub fn qualify(
    runner: &mut Runner<'_>,
    a: &mut Arena<'_>,
    pattern: Pattern,
    prior: Prior,
    live: u32,
) -> LabResult<serde_json::Value> {
    let i = inputs(a, pattern, live)?;
    let s = outputs(a, true)?;
    let c = outputs(a, false)?;
    let rows = CAPACITY;
    let served_ops = served(&i, &s, rows)?;
    let candidate_ops = candidate(&i, &c, rows, Mutation::None)?;
    let mutant_ops = candidate(&i, &c, rows, Mutation::CertifyAll)?;
    let mut failures = Vec::new();
    reset(a, &s, prior)?;
    reset(a, &c, prior)?;
    runner.run(a, &served_ops)?;
    runner.run(a, &candidate_ops)?;
    let reference = snapshot(a, &s)?;
    let first = snapshot(a, &c)?;
    let mut differing = BTreeMap::new();
    for (name, (x, y)) in COMPARED.iter().zip(reference.iter().zip(&first)) {
        let count = x.iter().zip(y).filter(|(p, q)| p != q).count();
        if count > 0 {
            failures.push(format!("{name} differs in {count} bytes"));
        }
        differing.insert(*name, count);
    }
    // Rows past the live count stay poisoned in both arms.
    let logits = u32s(&first[5]);
    if logits[(live * N) as usize..]
        .iter()
        .any(|&x| x != 0xa5a5_a5a5)
    {
        failures.push("the candidate wrote logits past the live rows".into());
    }
    // Flags: the served prepare's `single` and a host model of the norm's
    // widened rows.
    let flags = u32s(&a.read(c.rows)?)[..live as usize].to_vec();
    let single = served_single(&a.read(s.rows)?, rows)[..live as usize].to_vec();
    let widened = a.read(c.widened)?;
    let model: Vec<u32> = (0..live as usize)
        .map(|r| {
            u32::from(
                widened[r * K as usize * 4..(r + 1) * K as usize * 4]
                    .chunks_exact(4)
                    .all(|b| tensor_operand(f32::from_le_bytes(b.try_into().unwrap()))),
            )
        })
        .collect();
    if flags != single {
        failures.push("certified flags differ from the served prepare's single flags".into());
    }
    if flags != model {
        failures.push("certified flags differ from the host model".into());
    }
    let uncertified = flags.iter().filter(|&&f| f == 0).count();
    // Repeat, then a graph replay, each from poisoned outputs.
    reset(a, &c, prior)?;
    runner.run(a, &candidate_ops)?;
    let repeat = snapshot(a, &c)? == first;
    reset(a, &c, prior)?;
    runner.replay(a, &candidate_ops)?;
    let replay = snapshot(a, &c)? == first;
    if !repeat || !replay {
        failures.push(format!("repeat {repeat} replay {replay}"));
    }
    // Negative control: certifying every row never changes a certified
    // row's logits, and changes some row's logits exactly when a live
    // split row exists. Uncertified 1e15 and NaN rows may keep their bits.
    let split = if pattern == Pattern::Outliers {
        (0..live as usize).filter(|&r| split_row(r)).collect()
    } else {
        Vec::new()
    };
    if split.iter().any(|&r| flags[r] != 0) {
        failures.push("a split row was certified".into());
    }
    reset(a, &c, prior)?;
    runner.run(a, &mutant_ops)?;
    let mutant = a.read(c.logits)?;
    let row_bytes = (N * 4) as usize;
    let changed: Vec<usize> = (0..live as usize)
        .filter(|&r| {
            mutant[r * row_bytes..(r + 1) * row_bytes]
                != first[5][r * row_bytes..(r + 1) * row_bytes]
        })
        .collect();
    let changed_certified = changed.iter().filter(|&&r| flags[r] != 0).count();
    let mutant_differs = !changed.is_empty();
    if mutant[live as usize * row_bytes..] != first[5][live as usize * row_bytes..] {
        failures.push("certify_all wrote logits past the live rows".into());
    }
    if changed_certified > 0 {
        failures.push(format!(
            "certify_all changed {changed_certified} certified rows"
        ));
    }
    let split_live = !split.is_empty();
    if mutant_differs != split_live {
        failures.push(format!(
            "certify_all differs {mutant_differs} with {} live split rows",
            split.len()
        ));
    }
    reset(a, &c, prior)?;
    runner.run(a, &candidate_ops)?;
    if snapshot(a, &c)? != first {
        failures.push("the candidate differs after the negative control".into());
    }
    Ok(serde_json::json!({
        "pattern": format!("{pattern:?}").to_lowercase(),
        "prior": format!("{prior:?}").to_lowercase(),
        "live": live,
        "differing_bytes": differing,
        "certified_rows": live as usize - uncertified,
        "uncertified_rows": uncertified,
        "repeat_bitwise": repeat,
        "graph_bitwise": replay,
        "split_rows": split.len(),
        "mutant_differs": mutant_differs,
        "mutant_changed_rows": changed.len(),
        "mutant_changed_certified_rows": changed_certified,
        "failures": failures,
        "passed": failures.is_empty(),
        "emulated": runner.session.is_none(),
    }))
}

pub fn cases(memcheck: bool) -> Vec<(Pattern, Prior, u32)> {
    let mut out = Vec::new();
    for pattern in [Pattern::Clean, Pattern::Outliers] {
        for prior in [Prior::Clear, Prior::Flagged] {
            if memcheck && prior == Prior::Flagged {
                continue;
            }
            for &live in if memcheck {
                &MEMCHECK_LIVES[..]
            } else {
                &LIVES[..]
            } {
                out.push((pattern, prior, live));
            }
        }
    }
    out
}

fn bitwise(session: Option<&Session>) -> LabResult<()> {
    let memcheck = std::env::var(MEMCHECK).as_deref() == Ok("1");
    let mut runner = Runner::new(session)?;
    let mut executed = 0;
    let mut failed = 0;
    for (pattern, prior, live) in cases(memcheck) {
        let mut a = match session {
            Some(s) => Arena::device(s),
            None => Arena::host(),
        };
        let record = qualify(&mut runner, &mut a, pattern, prior, live)?;
        executed += 1;
        failed += usize::from(!record["passed"].as_bool().unwrap());
        println!("router_norm_record {record}");
    }
    println!(
        "router_norm_cases {{\"executed\":{executed},\"failed\":{failed},\"memcheck\":{memcheck}}}"
    );
    if executed == 0 || failed > 0 {
        return Err(format!("{failed} of {executed} cases failed"));
    }
    Ok(())
}

fn timing(session: &Session) -> LabResult<()> {
    let mut runner = Runner::new(Some(session))?;
    let mut executed = 0;
    for rows in TIMED_ROWS {
        let mut a = Arena::device(session);
        let i = inputs(&mut a, Pattern::Clean, rows)?;
        let s = outputs(&mut a, true)?;
        let c = outputs(&mut a, false)?;
        let reference = served(&i, &s, rows)?;
        let candidate = candidate(&i, &c, rows, Mutation::None)?;
        runner.ensure(&reference)?;
        runner.ensure(&candidate)?;
        let graphs = [&reference, &candidate]
            .iter()
            .map(|ops| session.capture(|st| runner.issue(&a, ops, st)))
            .collect::<LabResult<Vec<_>>>()?;
        let result = timing::pair(session, |arm, st| {
            graphs[arm].launch(st).map_err(|e| e.to_string())
        })?;
        executed += 1;
        println!(
            "router_norm_timing {}",
            serde_json::json!({"rows": rows, "timing": result})
        );
    }
    println!("router_norm_timed {{\"executed\":{executed}}}");
    Ok(())
}

mod timing;

#[test]
fn router_norm_cpu_plan() {
    // Every launch names an entry the lab cubin or the bundle declares, the
    // certified projection keeps the served grid, and the host emulation
    // passes a small case and catches the negative control.
    let lab: std::collections::BTreeSet<String> = family::cubins()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect();
    let mut a = Arena::host();
    let i = inputs(&mut a, Pattern::Outliers, 65).unwrap();
    let s = outputs(&mut a, true).unwrap();
    let c = outputs(&mut a, false).unwrap();
    for rows in [8192, 2048, 65] {
        let ops = served(&i, &s, rows).unwrap();
        let cand = candidate(&i, &c, rows, Mutation::None).unwrap();
        for op in &ops {
            assert!(Runner::serving(&op.symbol), "{}", op.symbol);
        }
        // The lab's certifying norm is its own entry; the certified
        // projection is also the bundle's served entry (`sm120::router_norm`),
        // which the runner then loads from the bundle.
        for (op, served) in cand.iter().zip([false, true]) {
            assert!(lab.contains(&op.symbol), "{}", op.symbol);
            assert_eq!(Runner::serving(&op.symbol), served, "{}", op.symbol);
        }
        // The zero path's rows and block over the 64-expert feature grid.
        assert_eq!(ops[2].plan.block, cand[1].plan.block);
        assert_eq!(ops[2].plan.grid[1], cand[1].plan.grid[1]);
        assert_eq!(cand[1].plan.grid[0], N.div_ceil(family::SERVED_FEATURES));
        assert_eq!(ops[0].plan, cand[0].plan);
        // The certifying norm takes the served arguments (over its own
        // outputs) and the flags.
        assert_eq!(cand[0].arguments.0.len(), ops[0].arguments.0.len() + 1);
        assert_eq!(cand[0].arguments.0[..2], ops[0].arguments.0[..2]);
        assert_eq!(cand[0].arguments.0.last(), Some(&ptr(c.rows)));
    }
    let mut runner = Runner::new(None).unwrap();
    let mut a = Arena::host();
    let record = qualify(&mut runner, &mut a, Pattern::Outliers, Prior::Flagged, 65).unwrap();
    assert!(record["passed"].as_bool().unwrap(), "{record}");
    assert!(record["uncertified_rows"].as_u64().unwrap() > 0, "{record}");
    assert!(record["mutant_differs"].as_bool().unwrap());
    println!(
        "router_norm_cpu_plan {{\"lab_symbols\":{},\"cases\":{}}}",
        lab.len(),
        cases(false).len()
    );
}

#[test]
fn emulation_catches_a_faulty_certified_arm() {
    // A candidate that ignores the flags (every row certified) fails a case
    // with an uncertified live row.
    let mut runner = Runner::new(None).unwrap();
    let mut a = Arena::host();
    let i = inputs(&mut a, Pattern::Outliers, 65).unwrap();
    let s = outputs(&mut a, true).unwrap();
    let c = outputs(&mut a, false).unwrap();
    reset(&mut a, &s, Prior::Clear).unwrap();
    reset(&mut a, &c, Prior::Clear).unwrap();
    runner
        .run(&mut a, &served(&i, &s, CAPACITY).unwrap())
        .unwrap();
    runner
        .run(
            &mut a,
            &candidate(&i, &c, CAPACITY, Mutation::CertifyAll).unwrap(),
        )
        .unwrap();
    assert_ne!(a.read(s.logits).unwrap(), a.read(c.logits).unwrap());
    assert_eq!(a.read(s.normed).unwrap(), a.read(c.normed).unwrap());
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS"]
fn router_norm_bitwise_gpu() {
    if emulated() {
        bitwise(None).unwrap();
        return;
    }
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, |s| bitwise(Some(s)))
        .unwrap();
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_DEEPSEEK41_ROUTER_NORM_CUBINS"]
fn router_norm_timing_gpu() {
    if emulated() {
        println!("router_norm_timed {{\"executed\":0,\"emulated\":true}}");
        return;
    }
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, timing).unwrap();
}
