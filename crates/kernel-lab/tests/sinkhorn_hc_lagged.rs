//! The lagged Sinkhorn hyper-connection engine
//! (`kernels_sm120::sinkhorn_hc::lagged`) at the lab's 4 x 5120 lagged
//! mixer, on the GPU. Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test
//! -p kernel-lab --release --test sinkhorn_hc_lagged -- --test-threads=1`;
//! without a GPU only the CPU contract tests run.
//!
//! The F32-output weighted head is also checked at 4 x 5120: exact FMA
//! bits, an f64 bound, live rows below capacity, poisoned unwritten rows,
//! eager/graph/PDL replay and chunk/rows/batch properties. Device mutants
//! that round to BF16 or reverse the stream order must fail its exact gate.
//!
//! One unit is the chain F32 expand -> lagged mix-collapse -> F32 expand ->
//! weighted head, as a sublayer boundary and the model head run it, at
//! decode (1, 32), verify (6 x 32) and prefill (8192) buckets:
//!
//! - the probe's raw projection against Neumaier-compensated f64 within
//!   2^-22 K S at K = 20,480 (the 128-K accumulation gate) and its rstd
//!   within its bound;
//! - the collapsed row bitwise equal to the exact collapse with the given
//!   weights, and within the f64 bound;
//! - pre and post within their F32 bounds, from the streams and tightly
//!   from the kernel's own logits; the comb by its parts: the probe's
//!   Sinkhorn input cells within their bound and the F32 comb bitwise equal
//!   to the exact normalizations of those cells;
//! - the per-sublayer twin's probe (same mixer, per-sublayer placement) on
//!   the same streams: projection, rstd and cells bitwise equal, its BF16
//!   post and comb the lagged F32 values rounded once, and its collapsed row
//!   the exact collapse with the lagged entry's stored pre;
//! - the served mix-collapse bitwise equal to the probe, written into
//!   freshly poisoned buffers that alone fail the comparison;
//! - both F32 expands and the head bitwise equal to their exact forms;
//! - the served chain replayed on poisoned buffers (eager, relaunch, graph,
//!   PDL) at every live count `poison::live_counts` lists, also on the
//!   extreme common-offset input (comb bases at 1e8, zero comb scale).

use kernel_core::backend::{Backend, Need};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, LaunchPlan};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::sinkhorn_hyper_connection::{
    SinkhornHyperConnectionQuery as Query, StreamHeadMode,
};
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_lab::kit::{LabModules, Modules, OpCase, Outcome, cases::sinkhorn_lagged as kit};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::lagged::{self, Entry, LAB, Op};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 5120;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const BUCKETS: [usize; 4] = [1, 32, 6 * 32, 8192];

fn f32_head_case(entry: Entry, capacity: u32) -> LabResult<OpCase<'static>> {
    let query = entry.query;
    let prepared = Sm120::new()
        .with_sinkhorn_lagged([entry])
        .sinkhorn_weighted_head(&query, Need::rows(capacity))
        .map_err(err)?;
    Ok(kit::head_case(query, capacity, prepared)?.live_counts(vec![
        0,
        1,
        3,
        capacity - 1,
        capacity,
    ]))
}

fn f32_head_reference(
    query: Query,
    images: &kernel_lab::conformance::Images,
    live: usize,
) -> LabResult<OracleOutput> {
    oracle::weighted_head_reference(
        query,
        &decode(&images.u16s("x")?[..live * FLAT]),
        &images.f32s("pre")?[..live * N],
    )
    .map_err(err)
}

#[test]
fn weighted_head_f32_cpu_contract() {
    let entry = lagged::head_f32_entry(lagged::LAB_PROFILE, LAB);
    for (cubin, entry) in std::iter::once((lagged::head_f32_cubin(), entry)).chain(
        lagged::HEAD_F32_MUTANTS.map(|(suffix, _)| {
            (
                lagged::head_f32_mutant_cubin(suffix),
                lagged::head_f32_mutant(suffix),
            )
        }),
    ) {
        assert!(
            kernel_lab::cubin_entries(&cubin)
                .unwrap()
                .contains(&entry.symbol.as_str())
        );
        kernel_lab::cubin(&cubin).unwrap();
        let case = f32_head_case(entry, 8).unwrap();
        case.validate().unwrap();
        let images = case.select(&case.images(), &[0, 3, 7]).unwrap();
        let query = lagged::head_f32_entry(lagged::LAB_PROFILE, LAB).query;
        assert_eq!(
            f32_head_reference(query, &images, 3).unwrap().values.len(),
            3 * H
        );
        assert_eq!(
            case.row_roles(kernel_core::properties::Scope::Entry)
                .unwrap()
                .outputs,
            vec![("output".into(), H * 4)]
        );
    }
}

#[test]
fn weighted_head_f32_matches_bits_bound_poison_and_properties() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = lagged::head_f32_entry(lagged::LAB_PROFILE, LAB);
        let query = entry.query;
        let case = f32_head_case(entry, 8)?;
        let functions = LabModules.load(session)?;
        let probes = LabModules.probes(session)?;
        let checks = case.checks(
            Need::rows(8),
            Some(session),
            Some(&*functions),
            probes.as_ref(),
        );
        let mut receipt = BTreeMap::new();
        for check in checks {
            match check.outcome {
                Outcome::Passed(value) => {
                    receipt.insert(check.name, value);
                }
                other => return Err(format!("F32 weighted head {}: {other:?}", check.name)),
            }
        }
        // Inspect actual F32 device values against the independent f64
        // bound as well as the pinned chain, using three rows in eight.
        let images = case.select(&case.images(), &[0, 3, 7])?;
        let outputs = case.launch(session, &*functions, &images, 3)?;
        case.judge_outputs(&images, 3, &outputs)?;
        let actual = outputs["output"]
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
            .collect::<Vec<_>>();
        let reference = f32_head_reference(query, &images, 3)?;
        gate(
            "F32 weighted head",
            &actual[..3 * H],
            &reference,
            DType::F32,
        )?;
        receipt.insert(
            "f64_worst_over_bound".into(),
            json!(worst_ratio(&actual[..3 * H], &reference, DType::F32)),
        );
        session.receipt("sinkhorn_weighted_head_f32", receipt)?;
        Ok(())
    })
    .expect(
        "F32 weighted head bits, f64 bound, poisoned live rows and chunk/rows/batch properties",
    );
}

#[test]
fn weighted_head_f32_rejects_device_rounding_and_order_mutants() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let functions = LabModules.load(session)?;
        let good = f32_head_case(lagged::head_f32_entry(lagged::LAB_PROFILE, LAB), 8)?;
        let images = good.select(&good.images(), &[0, 3, 7])?;
        let expected = good.launch(session, &*functions, &images, 3)?;
        good.judge_outputs(&images, 3, &expected)?;
        let mut receipt = BTreeMap::new();
        for (suffix, _) in lagged::HEAD_F32_MUTANTS {
            let case = f32_head_case(lagged::head_f32_mutant(suffix), 8)?;
            // Launch checks all writes, finite values and unwritten poison
            // before the exact oracle judges the mutant's output.
            let actual = case.launch(session, &*functions, &images, 3)?;
            let failure = case
                .judge_outputs(&images, 3, &actual)
                .expect_err("a device head mutant must fail its exact oracle");
            if !failure.contains("output byte") || !failure.contains("the reference") {
                return Err(format!(
                    "{suffix} failed outside the exact oracle: {failure}"
                ));
            }
            let differing = actual["output"][..3 * H * 4]
                .chunks_exact(4)
                .zip(expected["output"][..3 * H * 4].chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            if differing == 0 {
                return Err(format!("{suffix} has no device witness"));
            }
            eprintln!(
                "F32 head {suffix}: {differing} of {} device elements differ",
                3 * H
            );
            receipt.insert(
                suffix.into(),
                json!({"differing": differing, "failure": failure}),
            );
        }
        session.receipt("sinkhorn_weighted_head_f32_mutants", receipt)?;
        Ok(())
    })
    .expect("F32 weighted head device negative controls");
}

// Buffer handles. The chain is
//   X1 = expand_f32(X0, Y0, POST0, COMB0)
//   COLLAPSED, PRE, POST, COMB (, PROJECTED, RSTD, CELL_INPUT) =
//       lagged_mix_collapse(X1, PRE_IN)
//   X2 = expand_f32(X1, Y1, POST, COMB)
//   HEAD = weighted_head(X2, PRE)
// and the twin probe writes T_COLLAPSED, T_POST, T_COMB, T_PROJECTED,
// T_RSTD and T_CELLS from X1.
const X0: u64 = 0;
const Y0: u64 = 1;
const POST0: u64 = 2;
const COMB0: u64 = 3;
const X1: u64 = 4;
const FN: u64 = 5;
const BASE: u64 = 6;
const SCALE: u64 = 7;
const PRE_IN: u64 = 8;
const COLLAPSED: u64 = 9;
const PRE: u64 = 10;
const POST: u64 = 11;
const COMB: u64 = 12;
const PROJECTED: u64 = 13;
const RSTD: u64 = 14;
const CELL_INPUT: u64 = 15;
const Y1: u64 = 16;
const X2: u64 = 17;
const HEAD: u64 = 18;
const LIVE: u64 = 19;
const T_COLLAPSED: u64 = 20;
const T_POST: u64 = 21;
const T_COMB: u64 = 22;
const T_PROJECTED: u64 = 23;
const T_RSTD: u64 = 24;
const T_CELLS: u64 = 25;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn entry(op: Op) -> Entry {
    lagged::lab_catalog()
        .into_iter()
        .find(|entry| entry.op == op)
        .expect("a catalog entry")
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Deterministic BF16 bits: random signs and mantissas over the binades
/// `low..=high`, with zeros, negative zeros and a subnormal among them.
fn bf16_values(len: usize, seed: u64, low: i32, high: i32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            let state = xorshift(&mut state);
            match i % 509 {
                0 => 0x0000,
                1 => 0x8000,
                2 => 0x0001,
                _ => {
                    let span = (high - low + 1) as u64;
                    let exponent = (low + ((state >> 8) % span) as i32 + 127) as u16;
                    let sign = ((state >> 40) & 1) as u16;
                    (sign << 15) | (exponent << 7) | ((state >> 24) & 0x7f) as u16
                }
            }
        })
        .collect()
}

/// Deterministic F32 values with full mantissas over the binades
/// `low..=high`, random signs.
fn f32_values(len: usize, seed: u64, low: i32, high: i32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            let state = xorshift(&mut state);
            let span = (high - low + 1) as u64;
            let exponent = (low + (state % span) as i32 + 127) as u32;
            let sign = ((state >> 40) & 1) as u32;
            f32::from_bits((sign << 31) | (exponent << 23) | ((state >> 9) as u32 & 0x7f_ffff))
        })
        .collect()
}

/// Non-negative F32 values in [0, scale) with full mantissas.
fn unit_values(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| ((xorshift(&mut state) >> 40) as f32 / 16_777_216.0) * scale)
        .collect()
}

fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn decode_f32(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn round_bf16(values: &[f32]) -> Vec<u16> {
    values.iter().map(|&v| f32_to_bf16(v)).collect()
}

/// The served mix-collapse's outputs: collapsed, then pre, post and comb.
struct MixOutputs {
    collapsed: Vec<u16>,
    f32s: [Vec<f32>; 3],
}

impl MixOutputs {
    /// Bitwise equal to `other`.
    fn matches(&self, other: &MixOutputs) -> LabResult<()> {
        exact("served collapsed", &self.collapsed, &other.collapsed)?;
        for ((label, served), probe) in ["served pre", "served post", "served comb"]
            .into_iter()
            .zip(&self.f32s)
            .zip(&other.f32s)
        {
            exact_f32(label, served, probe)?;
        }
        Ok(())
    }
}

/// The comb by its parts: the probe's Sinkhorn input cells within their
/// bound from the kernel's own logits, then the F32 comb bitwise equal to
/// the exact F32 normalizations of those cells.
fn check_comb(cells: &[f32], bounds: &OracleOutput, comb: &[f32]) -> LabResult<()> {
    gate(
        "Sinkhorn input cells",
        &decode_f32(cells),
        bounds,
        DType::F32,
    )?;
    let mix = LAB.mix;
    let expected = cells
        .chunks_exact(CELLS)
        .flat_map(|row| {
            oracle::sinkhorn_f32(
                mix.order,
                row,
                N,
                mix.iterations,
                f32::from_bits(mix.eps_bits),
            )
        })
        .collect::<Vec<_>>();
    exact_f32("comb from the probe's cells", comb, &expected)
}

struct Inputs {
    x0: Vec<u16>,
    y0: Vec<u16>,
    post0: Vec<f32>,
    comb0: Vec<f32>,
    y1: Vec<u16>,
    pre_in: Vec<f32>,
    weights: Vec<f32>,
    base: Vec<f32>,
    scale: [f32; 3],
}

impl Inputs {
    fn new(rows: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let base = (0..MIXES)
            .map(|_| ((xorshift(&mut state) >> 40) % 2048) as f32 / 1024.0 - 1.0)
            .collect();
        Self {
            x0: bf16_values(rows * FLAT, seed + 1, -4, 2),
            y0: bf16_values(rows * H, seed + 2, -6, 1),
            post0: unit_values(rows * N, seed + 3, 2.0),
            comb0: unit_values(rows * CELLS, seed + 4, 1.0),
            y1: bf16_values(rows * H, seed + 5, -6, 1),
            pre_in: unit_values(rows * N, seed + 6, 1.25),
            // Logits of order one at K = 20,480.
            weights: f32_values(MIXES * FLAT, seed + 7, -10, -6),
            base,
            scale: [0.5, 0.75, 2.0],
        }
    }

    /// Every comb base at the F32 value 1e8 with a zero comb scale: all
    /// softmax cells are 1/4 + eps, where an affine allowance proportional
    /// to the logit would hide a wrong comb.
    fn extreme(rows: usize, seed: u64) -> Self {
        let mut inputs = Self::new(rows, seed);
        inputs.base[2 * N..].fill(1e8);
        inputs.scale[2] = 0.0;
        inputs
    }
}

struct Fixture {
    rows: usize,
    buffers: BTreeMap<u64, DeviceBuffer>,
    _modules: [Module; 3],
    expand: (Entry, Function),
    head: (Entry, Function),
    served: (Entry, Function),
    probe: (Entry, Function),
    twin: (kernels_sm120::sinkhorn_hc::Entry, Function),
}

impl Fixture {
    fn new(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Self> {
        let served_module = session.module(&lagged::served_cubin())?;
        let probe_module = session.module(&lagged::probe_cubin())?;
        let twin_module = session.module(&lagged::twin_cubin())?;
        let load = |module: &Module, op: Op| -> LabResult<(Entry, Function)> {
            let entry = entry(op);
            let function = module.function(&entry.symbol).map_err(err)?;
            Ok((entry, function))
        };
        let expand = load(&served_module, Op::Expand)?;
        let head = load(&served_module, Op::Head)?;
        let served = load(&served_module, Op::MixCollapse { probe: false })?;
        let probe = load(&probe_module, Op::MixCollapse { probe: true })?;
        let twin_entry = lagged::twin_probe();
        let twin_function = twin_module.function(&twin_entry.symbol).map_err(err)?;
        let mut buffers = BTreeMap::new();
        buffers.insert(X0, session.upload_bf16(&inputs.x0)?);
        buffers.insert(Y0, session.upload_bf16(&inputs.y0)?);
        buffers.insert(POST0, session.upload_f32(&inputs.post0)?);
        buffers.insert(COMB0, session.upload_f32(&inputs.comb0)?);
        buffers.insert(Y1, session.upload_bf16(&inputs.y1)?);
        buffers.insert(PRE_IN, session.upload_f32(&inputs.pre_in)?);
        buffers.insert(FN, session.upload_f32(&inputs.weights)?);
        buffers.insert(BASE, session.upload_f32(&inputs.base)?);
        buffers.insert(SCALE, session.upload_f32(&inputs.scale)?);
        for (handle, bytes) in [
            (X1, rows * FLAT * 2),
            (COLLAPSED, rows * H * 2),
            (PRE, rows * N * 4),
            (POST, rows * N * 4),
            (COMB, rows * CELLS * 4),
            (PROJECTED, rows * MIXES * 4),
            (RSTD, rows * 4),
            (CELL_INPUT, rows * CELLS * 4),
            (X2, rows * FLAT * 2),
            (HEAD, rows * H * 2),
            (LIVE, 4),
            (T_COLLAPSED, rows * H * 2),
            (T_POST, rows * N * 2),
            (T_COMB, rows * CELLS * 2),
            (T_PROJECTED, rows * MIXES * 4),
            (T_RSTD, rows * 4),
            (T_CELLS, rows * CELLS * 4),
        ] {
            buffers.insert(handle, session.allocate(bytes)?);
        }
        let fixture = Self {
            rows,
            buffers,
            _modules: [served_module, probe_module, twin_module],
            expand,
            head,
            served,
            probe,
            twin: (twin_entry, twin_function),
        };
        fixture.set_live(session, rows)?;
        Ok(fixture)
    }

    fn set_live(&self, session: &Session, live: usize) -> LabResult<()> {
        poison::write(
            session,
            &self.buffers[&LIVE],
            0,
            &(live as u32).to_ne_bytes(),
        )
    }

    fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self.buffers.iter().map(|(&k, v)| (k, v)).collect(),
        }
    }

    fn run(
        &self,
        session: &Session,
        function: &Function,
        mut plan: LaunchPlan,
        args: kernel_core::contract::ArgumentList,
        pdl: bool,
    ) -> LabResult<()> {
        plan.pdl = pdl;
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &self.bindings(),
        )
    }

    fn launch(
        &self,
        session: &Session,
        (entry, function): &(Entry, Function),
        handles: &[u64],
        pdl: bool,
    ) -> LabResult<()> {
        let plan = entry.plan(self.rows as u32, 0).map_err(err)?;
        let args = entry
            .pack_arguments(handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect())
            .map_err(err)?;
        self.run(session, function, plan, args, pdl)
    }

    fn expand(&self, session: &Session, inputs: [u64; 4], out: u64, pdl: bool) -> LabResult<()> {
        let [x, y, post, comb] = inputs;
        self.launch(session, &self.expand, &[x, y, post, comb, out, LIVE], pdl)
    }

    fn mix(&self, session: &Session, probe: bool, pdl: bool) -> LabResult<()> {
        let mut handles = vec![X1, FN, BASE, SCALE, PRE_IN, COLLAPSED, PRE, POST, COMB];
        if probe {
            handles.extend([PROJECTED, RSTD, CELL_INPUT]);
        }
        handles.push(LIVE);
        let kernel = if probe { &self.probe } else { &self.served };
        self.launch(session, kernel, &handles, pdl)
    }

    fn twin(&self, session: &Session) -> LabResult<()> {
        let (entry, function) = &self.twin;
        let plan = entry.plan(self.rows as u32, 0).map_err(err)?;
        let handles = [
            X1,
            FN,
            BASE,
            SCALE,
            T_COLLAPSED,
            T_POST,
            T_COMB,
            T_PROJECTED,
            T_RSTD,
            T_CELLS,
            LIVE,
        ];
        let args = entry
            .pack_arguments(handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect())
            .map_err(err)?;
        self.run(session, function, plan, args, false)
    }

    /// The first expand, then the rest of the chain, with PDL on every
    /// launch after the first when `pdl`.
    fn producer(&self, session: &Session) -> LabResult<()> {
        self.expand(session, [X0, Y0, POST0, COMB0], X1, false)
    }

    fn dependents(&self, session: &Session, probe: bool, pdl: bool) -> LabResult<()> {
        self.mix(session, probe, pdl)?;
        self.expand(session, [X1, Y1, POST, COMB], X2, pdl)?;
        self.launch(session, &self.head, &[X2, PRE, HEAD, LIVE], pdl)
    }

    fn chain(&self, session: &Session, probe: bool, pdl: bool) -> LabResult<()> {
        self.producer(session)?;
        self.dependents(session, probe, pdl)
    }

    fn bf16(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<u16>> {
        session.download_bf16(&self.buffers[&handle], self.rows * per_row)
    }

    fn f32(&self, session: &Session, handle: u64, per_row: usize) -> LabResult<Vec<f32>> {
        session.download_f32(&self.buffers[&handle], self.rows * per_row)
    }

    fn mix_outputs(&self, session: &Session) -> LabResult<MixOutputs> {
        Ok(MixOutputs {
            collapsed: self.bf16(session, COLLAPSED, H)?,
            f32s: [
                self.f32(session, PRE, N)?,
                self.f32(session, POST, N)?,
                self.f32(session, COMB, CELLS)?,
            ],
        })
    }
}

/// `check_numeric` with a labelled error.
fn gate(label: &str, actual: &[f64], reference: &OracleOutput, dtype: DType) -> LabResult<()> {
    check_numeric(actual, reference, dtype).map_err(|f| {
        format!(
            "{label}: element {} is {}, reference {}, error {} over bound {}",
            f.index, f.actual, f.reference, f.error, f.bound
        )
    })
}

/// The largest error over its gate bound, for the receipt.
fn worst_ratio(actual: &[f64], reference: &OracleOutput, dtype: DType) -> f64 {
    (0..actual.len())
        .map(|i| {
            let bound = kernel_core::oracle::numeric_bound(reference, i, dtype);
            (actual[i] - reference.values[i]).abs() / bound
        })
        .fold(0.0, f64::max)
}

fn exact(label: &str, actual: &[u16], expected: &[u16]) -> LabResult<()> {
    kernel_core::oracle::check_exact(actual, expected).map_err(|at| {
        format!(
            "{label}: element {at} is {:#06x}, expected {:#06x}",
            actual.get(at).copied().unwrap_or_default(),
            expected.get(at).copied().unwrap_or_default()
        )
    })
}

fn exact_f32(label: &str, actual: &[f32], expected: &[f32]) -> LabResult<()> {
    let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    kernel_core::oracle::check_exact(&bits(actual), &bits(expected)).map_err(|at| {
        format!(
            "{label}: element {at} is {:?}, expected {:?}",
            actual.get(at),
            expected.get(at)
        )
    })
}

/// Runs `run` over row chunks on every core, in row order.
fn by_rows<T: Send>(rows: usize, run: impl Fn(std::ops::Range<usize>) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(8, |v| v.get());
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let handles = (0..rows)
            .step_by(per)
            .map(|start| {
                let run = &run;
                scope.spawn(move || run(start..(start + per).min(rows)))
            })
            .collect::<Vec<_>>();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
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

fn query(rows: usize) -> Query {
    Query::MixCollapse {
        mix: LAB.mix,
        capacity: rows as u32,
    }
}

fn twin_query(rows: usize) -> Query {
    Query::MixCollapse {
        mix: lagged::TWIN.mix,
        capacity: rows as u32,
    }
}

/// Every oracle comparison of one bucket; returns its receipt row.
fn oracle_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    fixture.chain(session, true, false)?;
    fixture.twin(session)?;
    session.stream.synchronize().map_err(err)?;
    let x1 = fixture.bf16(session, X1, FLAT)?;
    let probe = fixture.mix_outputs(session)?;
    let [pre, post, comb] = &probe.f32s;
    let collapsed = &probe.collapsed;
    let x2 = fixture.bf16(session, X2, FLAT)?;
    let head = fixture.bf16(session, HEAD, H)?;
    let projected = fixture.f32(session, PROJECTED, MIXES)?;
    let rstd = fixture.f32(session, RSTD, 1)?;
    let cells = fixture.f32(session, CELL_INPUT, CELLS)?;

    // The F32 expands and the head: exact.
    let expand_query = Query::ExpandF32 {
        streams: N as u32,
        width: H as u32,
        capacity: rows as u32,
    };
    let head_query = Query::Head {
        streams: N as u32,
        width: H as u32,
        mode: StreamHeadMode::SigmoidWeighted,
        capacity: rows as u32,
    };
    let expected = oracle::expand_f32(
        expand_query,
        &inputs.x0,
        &inputs.y0,
        &inputs.post0,
        &inputs.comb0,
    )
    .map_err(err)?;
    exact("first expand", &x1, &expected)?;
    let expected = oracle::expand_f32(expand_query, &x1, &inputs.y1, post, comb).map_err(err)?;
    exact("second expand", &x2, &expected)?;
    let expected = oracle::weighted_head(head_query, &x2, pre).map_err(err)?;
    exact("weighted head", &head, &expected)?;

    // The collapse with the given weights: exact.
    let expected = oracle::collapse(query(rows), &x1, &inputs.pre_in).map_err(err)?;
    exact("collapsed", collapsed, &expected)?;

    // The long reductions, then the nonlinear stage.
    let x = decode(&x1);
    let weights = decode_f32(&inputs.weights);
    let base = decode_f32(&inputs.base);
    let scale = inputs.scale.map(f64::from);
    let mix_weights = MixWeights {
        projection: &weights,
        base: &base,
        scale: &scale,
    };
    let rows_of = |r: std::ops::Range<usize>| &x[r.start * FLAT..r.end * FLAT];
    let projection = concat(by_rows(rows, |r| {
        oracle::projection(query(r.len()), rows_of(r.clone()), mix_weights).unwrap()
    }));
    let scales = concat(by_rows(rows, |r| {
        oracle::rstd(query(r.len()), rows_of(r.clone())).unwrap()
    }));
    let projected64 = decode_f32(&projected);
    let rstd64 = decode_f32(&rstd);
    gate(
        "projection (128-K accumulation)",
        &projected64,
        &projection,
        DType::F32,
    )?;
    gate("rstd", &rstd64, &scales, DType::F32)?;
    let logits = projected
        .iter()
        .enumerate()
        .map(|(i, &p)| p * rstd[i / MIXES])
        .collect::<Vec<_>>();
    let tight = by_rows(rows, |r| {
        oracle::lagged_mix_stage_from_kernel_logits(
            query(r.len()),
            &logits[r.start * MIXES..r.end * MIXES],
            &inputs.base,
            &inputs.scale,
        )
        .unwrap()
    });
    let full = by_rows(rows, |r| {
        oracle::lagged_mix_collapse(
            query(r.len()),
            rows_of(r.clone()),
            &inputs.pre_in[r.start * N..r.end * N],
            mix_weights,
        )
        .unwrap()
    });
    gate(
        "collapsed from the streams",
        &decode(collapsed),
        &concat(full.iter().map(|f| f.collapsed.clone()).collect()),
        DType::Bf16,
    )?;
    let (pre64, post64) = (decode_f32(pre), decode_f32(post));
    let mut ratios = BTreeMap::new();
    for (label, p, q) in [
        (
            "from the kernel's logits",
            concat(tight.iter().map(|t| t.pre.clone()).collect()),
            concat(tight.iter().map(|t| t.post.clone()).collect()),
        ),
        (
            "from the streams",
            concat(full.iter().map(|f| f.pre.clone()).collect()),
            concat(full.iter().map(|f| f.post.clone()).collect()),
        ),
    ] {
        gate(&format!("pre {label}"), &pre64, &p, DType::F32)?;
        gate(&format!("post {label}"), &post64, &q, DType::F32)?;
        ratios.insert(
            label,
            json!({
                "pre": worst_ratio(&pre64, &p, DType::F32),
                "post": worst_ratio(&post64, &q, DType::F32),
            }),
        );
    }
    let cell_bounds = concat(tight.iter().map(|t| t.cells.clone()).collect());
    check_comb(&cells, &cell_bounds, comb)?;
    if check_comb(&cells, &cell_bounds, &vec![0.0; comb.len()]).is_ok() {
        return Err("a zero comb passed the comb check".into());
    }
    let reference = concat(full.iter().map(|f| f.comb.clone()).collect());
    let comb_deviation = decode_f32(comb)
        .iter()
        .zip(&reference.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    // The per-sublayer twin on the same streams: the same long reductions
    // and Sinkhorn, its stored values the lagged ones rounded once, and its
    // collapse the lagged entry's stored pre's.
    let bits_f32 = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for (label, twin, lagged) in [
        ("twin projection", T_PROJECTED, &projected),
        ("twin rstd", T_RSTD, &rstd),
        ("twin cells", T_CELLS, &cells),
    ] {
        let twin = fixture.f32(session, twin, lagged.len() / rows)?;
        kernel_core::oracle::check_exact(&bits_f32(&twin), &bits_f32(lagged))
            .map_err(|at| format!("{label} differs at {at}"))?;
    }
    exact(
        "twin post",
        &fixture.bf16(session, T_POST, N)?,
        &round_bf16(post),
    )?;
    exact(
        "twin comb",
        &fixture.bf16(session, T_COMB, CELLS)?,
        &round_bf16(comb),
    )?;
    let expected = oracle::collapse(twin_query(rows), &x1, pre).map_err(err)?;
    exact(
        "twin collapsed",
        &fixture.bf16(session, T_COLLAPSED, H)?,
        &expected,
    )?;

    // The served entry writes the probe's bits into freshly poisoned
    // buffers; the poisoned buffers alone must fail the comparison.
    let outputs = [COLLAPSED, PRE, POST, COMB].map(|handle| &fixture.buffers[&handle]);
    poison::poison(session, &outputs)?;
    if fixture.mix_outputs(session)?.matches(&probe).is_ok() {
        return Err("unwritten served outputs matched the probe".into());
    }
    fixture.mix(session, false, false)?;
    session.stream.synchronize().map_err(err)?;
    fixture.mix_outputs(session)?.matches(&probe)?;
    Ok(json!({
        "rows": rows,
        "projection_worst_over_bound": worst_ratio(&projected64, &projection, DType::F32),
        "rstd_worst_over_bound": worst_ratio(&rstd64, &scales, DType::F32),
        "worst_over_bound": ratios,
        "cells_worst_over_bound": worst_ratio(&decode_f32(&cells), &cell_bounds, DType::F32),
        "comb_max_abs_vs_f64": comb_deviation,
    }))
}

/// The served chain on poisoned buffers at every live count of `rows`.
fn replay_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    let graph = session.capture(|_| fixture.chain(session, false, true))?;
    let mut evidence = Vec::new();
    for live in poison::live_counts(rows as u32) {
        let live = live as usize;
        fixture.set_live(session, live)?;
        let buffer = |handle| &fixture.buffers[&handle];
        let bf16 = |elements: usize| WriteMap::of(DType::Bf16).prefix(live * elements);
        let f32 = |elements: usize| WriteMap::of(DType::F32).prefix(live * elements);
        let mut replay = Replay::new(session)
            .output("x1", buffer(X1), bf16(FLAT)?)
            .output("collapsed", buffer(COLLAPSED), bf16(H)?)
            .output("pre", buffer(PRE), f32(N)?)
            .output("post", buffer(POST), f32(N)?)
            .output("comb", buffer(COMB), f32(CELLS)?)
            .output("x2", buffer(X2), bf16(FLAT)?)
            .output("head", buffer(HEAD), bf16(H)?);
        replay.eager(|| fixture.chain(session, false, false))?;
        replay.relaunch(|| fixture.chain(session, false, false))?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.producer(session),
            || fixture.dependents(session, false, true),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        evidence.push(json!({"live": live, "replay": receipt}));
    }
    Ok(json!({"rows": rows, "live_counts": evidence}))
}

#[test]
fn sinkhorn_hc_lagged_cpu_contract() {
    for (cubin, op) in [
        (lagged::served_cubin(), Op::MixCollapse { probe: false }),
        (lagged::served_cubin(), Op::Expand),
        (lagged::served_cubin(), Op::Head),
        (lagged::probe_cubin(), Op::MixCollapse { probe: true }),
    ] {
        let symbol = entry(op).symbol;
        assert!(
            kernel_lab::cubin_entries(&cubin)
                .unwrap()
                .contains(&symbol.as_str()),
            "{cubin} lacks {symbol}"
        );
        kernel_lab::cubin(&cubin).unwrap();
        for rows in BUCKETS {
            let plan = entry(op).plan(rows as u32, rows as u32).unwrap();
            assert!(plan.pdl);
        }
    }
    let twin = lagged::twin_probe().symbol;
    assert!(
        kernel_lab::cubin_entries(&lagged::twin_cubin())
            .unwrap()
            .contains(&twin.as_str())
    );
    // A served launch that writes nothing leaves the poison pattern, which
    // fails the comparison with any probe output.
    let probe = MixOutputs {
        collapsed: vec![0x3f80; H],
        f32s: [vec![0.5; N], vec![1.5; N], vec![0.25; CELLS]],
    };
    let unwritten = MixOutputs {
        collapsed: vec![0xffff; H],
        f32s: [
            vec![f32::from_bits(u32::MAX); N],
            vec![f32::from_bits(u32::MAX); N],
            vec![f32::from_bits(u32::MAX); CELLS],
        ],
    };
    assert!(unwritten.matches(&probe).is_err());
    probe.matches(&probe).unwrap();
    let inputs = Inputs::new(1, 1);
    assert_eq!(inputs.x0.len(), FLAT);
    assert_eq!(inputs.weights.len(), MIXES * FLAT);
    assert!(inputs.weights.iter().all(|w| w.is_finite() && *w != 0.0));
}

#[test]
fn sinkhorn_hc_lagged_matches_the_oracles_at_lab_shapes() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| oracle_bucket(session, rows, &Inputs::new(rows, 0x5eed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = oracle_bucket(session, 32, &Inputs::extreme(32, 0xe0))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_lagged_oracle"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_lagged_oracle", receipt)?;
        Ok(())
    })
    .expect("lagged Sinkhorn hyper-connection oracle comparison");
}

#[test]
fn sinkhorn_hc_lagged_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| replay_bucket(session, rows, &Inputs::new(rows, 0xfeed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = replay_bucket(session, 32, &Inputs::extreme(32, 0xfe))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_lagged_poison"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_lagged_poison", receipt)?;
        Ok(())
    })
    .expect("lagged Sinkhorn hyper-connection poisoned replays");
}
