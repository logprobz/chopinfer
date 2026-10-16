//! The Sinkhorn hyper-connection engine (`kernels_sm120::sinkhorn_hc`) at
//! the served 4 x 4096 mixer, on the GPU. Run with
//! `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab --release --test
//! sinkhorn_hc -- --test-threads=1`; without a GPU only the CPU contract
//! test runs.
//!
//! One unit is the chain expand -> mix-collapse -> expand -> head, as a
//! sublayer boundary and the model head run it, at decode (1, 32), verify
//! (6 x 32) and prefill (8192) buckets:
//!
//! - the probe's raw projection against Neumaier-compensated f64 within
//!   2^-22 K S at K = 16,384 (the 128-K accumulation gate) and its rstd
//!   within its bound;
//! - post, comb and collapsed against the f64 oracle, within the gate from
//!   the streams and tightly from the kernel's own F32 logits;
//! - the served mix-collapse bitwise equal to the probe, written into
//!   freshly poisoned buffers that alone fail the comparison;
//! - expand and head bitwise equal to their exact F32 forms;
//! - the served chain replayed on poisoned buffers (eager, relaunch, graph,
//!   PDL) at every live count `poison::live_counts` lists, also on the
//!   extreme common-offset input (comb bases at 1e8, zero comb scale).

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType};
use kernel_core::oracle::{OracleOutput, check_numeric};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::poison::{self, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::sinkhorn_hc::{self, Entry, Op, SERVED};
use runtime_cuda::{DeviceBuffer, Function, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const N: usize = 4;
const H: usize = 4096;
const FLAT: usize = N * H;
const MIXES: usize = (2 + N) * N;
const CELLS: usize = N * N;
const BUCKETS: [usize; 4] = [1, 32, 6 * 32, 8192];

// Buffer handles. The chain is
//   X1 = expand(X0, Y0, POST0, COMB0)
//   COLLAPSED, POST, COMB (, PROJECTED, RSTD, CELL_INPUT) = mix_collapse(X1)
//   X2 = expand(X1, Y1, POST, COMB)
//   HEAD = head(X2)
const X0: u64 = 0;
const Y0: u64 = 1;
const POST0: u64 = 2;
const COMB0: u64 = 3;
const X1: u64 = 4;
const FN: u64 = 5;
const BASE: u64 = 6;
const SCALE: u64 = 7;
const COLLAPSED: u64 = 8;
const POST: u64 = 9;
const COMB: u64 = 10;
const PROJECTED: u64 = 11;
const RSTD: u64 = 12;
const Y1: u64 = 13;
const X2: u64 = 14;
const HEAD: u64 = 15;
const LIVE: u64 = 16;
const CELL_INPUT: u64 = 17;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn served_cubin() -> String {
    format!("sinkhorn_hc_{}_sm120", sinkhorn_hc::SERVED_PROFILE.tag())
}

fn probe_cubin() -> String {
    format!(
        "sinkhorn_hc_{}_probe_sm120",
        sinkhorn_hc::SERVED_PROFILE.tag()
    )
}

fn entry(op: Op) -> Entry {
    sinkhorn_hc::lab_catalog()
        .into_iter()
        .find(|entry| entry.op == op)
        .expect("a catalog entry")
}

/// Deterministic BF16 bits: random signs and mantissas over the binades
/// `low..=high`, with zeros, negative zeros and a subnormal among them.
fn bf16_values(len: usize, seed: u64, low: i32, high: i32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
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

/// Non-negative BF16 bits in [0, scale).
fn unit_values(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16(((state >> 40) % 4096) as f32 / 4096.0 * scale)
        })
        .collect()
}

fn decode(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn decode_f32(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

/// The served mix-collapse's collapsed, post and comb equal the probe's
/// bit for bit.
fn served_matches_probe(served: &[&[u16]; 3], probe: &[&[u16]; 3]) -> LabResult<()> {
    for ((label, served), probe) in ["served collapsed", "served post", "served comb"]
        .into_iter()
        .zip(served)
        .zip(probe)
    {
        exact(label, served, probe)?;
    }
    Ok(())
}

/// The comb by its parts: the probe's Sinkhorn input cells within their
/// bound from the kernel's own logits, then the comb bitwise equal to the
/// exact F32 normalizations of those cells, rounded once to BF16.
fn check_comb(cells: &[f32], bounds: &OracleOutput, comb: &[u16]) -> LabResult<()> {
    gate(
        "Sinkhorn input cells",
        &decode_f32(cells),
        bounds,
        DType::F32,
    )?;
    let mix = SERVED.mix;
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
        .map(f32_to_bf16)
        .collect::<Vec<_>>();
    exact("comb from the probe's cells", comb, &expected)
}

struct Inputs {
    x0: Vec<u16>,
    y0: Vec<u16>,
    post0: Vec<u16>,
    comb0: Vec<u16>,
    y1: Vec<u16>,
    weights: Vec<u16>,
    base: Vec<f32>,
    scale: [f32; 3],
}

impl Inputs {
    fn new(rows: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let base = (0..MIXES)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) % 2048) as f32 / 1024.0 - 1.0
            })
            .collect();
        Self {
            x0: bf16_values(rows * FLAT, seed + 1, -4, 2),
            y0: bf16_values(rows * H, seed + 2, -6, 1),
            post0: unit_values(rows * N, seed + 3, 2.0),
            comb0: unit_values(rows * CELLS, seed + 4, 1.0),
            y1: bf16_values(rows * H, seed + 5, -6, 1),
            // Logits of order one at K = 16,384.
            weights: bf16_values(MIXES * FLAT, seed + 6, -10, -6),
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
    _modules: [Module; 2],
    expand: (Entry, Function),
    head: (Entry, Function),
    served: (Entry, Function),
    probe: (Entry, Function),
}

impl Fixture {
    fn new(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Self> {
        let served_module = session.module(&served_cubin())?;
        let probe_module = session.module(&probe_cubin())?;
        let load = |module: &Module, op: Op| -> LabResult<(Entry, Function)> {
            let entry = entry(op);
            let function = module.function(&entry.symbol).map_err(err)?;
            Ok((entry, function))
        };
        let expand = load(&served_module, Op::Expand)?;
        let head = load(&served_module, Op::Head)?;
        let served = load(&served_module, Op::MixCollapse { probe: false })?;
        let probe = load(&probe_module, Op::MixCollapse { probe: true })?;
        let mut buffers = BTreeMap::new();
        buffers.insert(X0, session.upload_bf16(&inputs.x0)?);
        buffers.insert(Y0, session.upload_bf16(&inputs.y0)?);
        buffers.insert(POST0, session.upload_bf16(&inputs.post0)?);
        buffers.insert(COMB0, session.upload_bf16(&inputs.comb0)?);
        buffers.insert(Y1, session.upload_bf16(&inputs.y1)?);
        buffers.insert(FN, session.upload_bf16(&inputs.weights)?);
        buffers.insert(BASE, session.upload_f32(&inputs.base)?);
        buffers.insert(SCALE, session.upload_f32(&inputs.scale)?);
        for (handle, bytes) in [
            (X1, rows * FLAT * 2),
            (COLLAPSED, rows * H * 2),
            (POST, rows * N * 2),
            (COMB, rows * CELLS * 2),
            (PROJECTED, rows * MIXES * 4),
            (RSTD, rows * 4),
            (CELL_INPUT, rows * CELLS * 4),
            (X2, rows * FLAT * 2),
            (HEAD, rows * H * 2),
            (LIVE, 4),
        ] {
            buffers.insert(handle, session.allocate(bytes)?);
        }
        let fixture = Self {
            rows,
            buffers,
            _modules: [served_module, probe_module],
            expand,
            head,
            served,
            probe,
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

    fn launch(
        &self,
        session: &Session,
        (entry, function): &(Entry, Function),
        handles: &[u64],
        pdl: bool,
    ) -> LabResult<()> {
        let mut plan = entry.plan(self.rows as u32, 0).map_err(err)?;
        plan.pdl = pdl;
        let args = entry
            .pack_arguments(handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect())
            .map_err(err)?;
        launch::launch(
            &session.context,
            function,
            &session.stream,
            plan,
            &args,
            &self.bindings(),
        )
    }

    fn expand(&self, session: &Session, inputs: [u64; 4], out: u64, pdl: bool) -> LabResult<()> {
        let [x, y, post, comb] = inputs;
        self.launch(session, &self.expand, &[x, y, post, comb, out, LIVE], pdl)
    }

    fn mix(&self, session: &Session, probe: bool, pdl: bool) -> LabResult<()> {
        if probe {
            let handles = [
                X1, FN, BASE, SCALE, COLLAPSED, POST, COMB, PROJECTED, RSTD, CELL_INPUT, LIVE,
            ];
            self.launch(session, &self.probe, &handles, pdl)
        } else {
            let handles = [X1, FN, BASE, SCALE, COLLAPSED, POST, COMB, LIVE];
            self.launch(session, &self.served, &handles, pdl)
        }
    }

    /// The first expand, then the rest of the chain, with PDL on every
    /// launch after the first when `pdl`.
    fn producer(&self, session: &Session) -> LabResult<()> {
        self.expand(session, [X0, Y0, POST0, COMB0], X1, false)
    }

    fn dependents(&self, session: &Session, probe: bool, pdl: bool) -> LabResult<()> {
        self.mix(session, probe, pdl)?;
        self.expand(session, [X1, Y1, POST, COMB], X2, pdl)?;
        self.launch(session, &self.head, &[X2, HEAD, LIVE], pdl)
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
        mix: SERVED.mix,
        capacity: rows as u32,
    }
}

/// Every oracle comparison of one bucket; returns its receipt row.
fn oracle_bucket(session: &Session, rows: usize, inputs: &Inputs) -> LabResult<Value> {
    let fixture = Fixture::new(session, rows, inputs)?;
    fixture.chain(session, true, false)?;
    session.stream.synchronize().map_err(err)?;
    let x1 = fixture.bf16(session, X1, FLAT)?;
    let post = fixture.bf16(session, POST, N)?;
    let comb = fixture.bf16(session, COMB, CELLS)?;
    let collapsed = fixture.bf16(session, COLLAPSED, H)?;
    let x2 = fixture.bf16(session, X2, FLAT)?;
    let head = fixture.bf16(session, HEAD, H)?;
    let projected = fixture.f32(session, PROJECTED, MIXES)?;
    let rstd = fixture.f32(session, RSTD, 1)?;
    let cells = fixture.f32(session, CELL_INPUT, CELLS)?;

    // Expand and head: exact.
    let expand_query = Query::Expand {
        streams: N as u32,
        width: H as u32,
        capacity: rows as u32,
    };
    let head_query = Query::Head {
        streams: N as u32,
        width: H as u32,
        mode: kernel_core::sinkhorn_hyper_connection::StreamHeadMode::MeanOfStreams,
        capacity: rows as u32,
    };
    let expected = oracle::expand(
        expand_query,
        &inputs.x0,
        &inputs.y0,
        &inputs.post0,
        &inputs.comb0,
    )
    .map_err(err)?;
    exact("first expand", &x1, &expected)?;
    let expected = oracle::expand(expand_query, &x1, &inputs.y1, &post, &comb).map_err(err)?;
    exact("second expand", &x2, &expected)?;
    exact("head", &head, &oracle::head(head_query, &x2).map_err(err)?)?;

    // The long reductions, then the nonlinear stage from the kernel's logits.
    let x = decode(&x1);
    let weights = decode(&inputs.weights);
    let base = inputs
        .base
        .iter()
        .map(|&v| f64::from(v))
        .collect::<Vec<_>>();
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
    let projected64 = projected.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let rstd64 = rstd.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    gate(
        "projection (128-K accumulation)",
        &projected64,
        &projection,
        DType::F32,
    )?;
    gate("rstd", &rstd64, &scales, DType::F32)?;
    // Post and collapsed from the kernel's own F32 logits (tight) and from
    // the streams (within the projection gate); the comb by its parts.
    let logits = projected
        .iter()
        .enumerate()
        .map(|(i, &p)| p * rstd[i / MIXES])
        .collect::<Vec<_>>();
    let tight = by_rows(rows, |r| {
        oracle::mix_stage_from_kernel_logits(
            query(r.len()),
            rows_of(r.clone()),
            &logits[r.start * MIXES..r.end * MIXES],
            &inputs.base,
            &inputs.scale,
        )
        .unwrap()
    });
    let full = by_rows(rows, |r| {
        oracle::mix_collapse(query(r.len()), rows_of(r.clone()), mix_weights).unwrap()
    });
    let (collapsed64, post64, comb64) = (decode(&collapsed), decode(&post), decode(&comb));
    let mut ratios = BTreeMap::new();
    for (label, c, p) in [
        (
            "from the kernel's logits",
            concat(tight.iter().map(|t| t.collapsed.clone()).collect()),
            concat(tight.iter().map(|t| t.post.clone()).collect()),
        ),
        (
            "from the streams",
            concat(full.iter().map(|f| f.collapsed.clone()).collect()),
            concat(full.iter().map(|f| f.post.clone()).collect()),
        ),
    ] {
        gate(&format!("collapsed {label}"), &collapsed64, &c, DType::Bf16)?;
        gate(&format!("post {label}"), &post64, &p, DType::Bf16)?;
        ratios.insert(
            label,
            json!({
                "collapsed": worst_ratio(&collapsed64, &c, DType::Bf16),
                "post": worst_ratio(&post64, &p, DType::Bf16),
            }),
        );
    }
    let cell_bounds = concat(tight.iter().map(|t| t.cells.clone()).collect());
    check_comb(&cells, &cell_bounds, &comb)?;
    if check_comb(&cells, &cell_bounds, &vec![0; comb.len()]).is_ok() {
        return Err("a zero comb passed the comb check".into());
    }
    // Measured only: no useful finite bound covers 20 Sinkhorn iterations.
    let reference = concat(full.iter().map(|f| f.comb.clone()).collect());
    let comb_deviation = comb64
        .iter()
        .zip(&reference.values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    // The served entry writes the probe's bits into freshly poisoned
    // buffers; the poisoned buffers alone must fail the comparison.
    let probe = [collapsed.as_slice(), &post, &comb];
    let outputs = [COLLAPSED, POST, COMB].map(|handle| &fixture.buffers[&handle]);
    poison::poison(session, &outputs)?;
    let served = |fixture: &Fixture| -> LabResult<[Vec<u16>; 3]> {
        Ok([
            fixture.bf16(session, COLLAPSED, H)?,
            fixture.bf16(session, POST, N)?,
            fixture.bf16(session, COMB, CELLS)?,
        ])
    };
    let unwritten = served(&fixture)?;
    if served_matches_probe(&unwritten.each_ref().map(Vec::as_slice), &probe).is_ok() {
        return Err("unwritten served outputs matched the probe".into());
    }
    fixture.mix(session, false, false)?;
    session.stream.synchronize().map_err(err)?;
    served_matches_probe(&served(&fixture)?.each_ref().map(Vec::as_slice), &probe)?;
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
        let mut replay = Replay::new(session)
            .output(
                "x1",
                buffer(X1),
                WriteMap::of(DType::Bf16).prefix(live * FLAT)?,
            )
            .output(
                "collapsed",
                buffer(COLLAPSED),
                WriteMap::of(DType::Bf16).prefix(live * H)?,
            )
            .output(
                "post",
                buffer(POST),
                WriteMap::of(DType::Bf16).prefix(live * N)?,
            )
            .output(
                "comb",
                buffer(COMB),
                WriteMap::of(DType::Bf16).prefix(live * CELLS)?,
            )
            .output(
                "x2",
                buffer(X2),
                WriteMap::of(DType::Bf16).prefix(live * FLAT)?,
            )
            .output(
                "head",
                buffer(HEAD),
                WriteMap::of(DType::Bf16).prefix(live * H)?,
            );
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
fn sinkhorn_hc_cpu_contract() {
    for (cubin, op) in [
        (served_cubin(), Op::MixCollapse { probe: false }),
        (served_cubin(), Op::Expand),
        (served_cubin(), Op::Head),
        (probe_cubin(), Op::MixCollapse { probe: true }),
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
    // A served launch that writes nothing leaves the poison pattern, which
    // fails the comparison with any probe output.
    let probe = [vec![0x3f80_u16; H], vec![0x3f00; N], vec![0x3e80; CELLS]];
    let probe = probe.each_ref().map(Vec::as_slice);
    let unwritten = [vec![0xffff_u16; H], vec![0xffff; N], vec![0xffff; CELLS]];
    assert!(served_matches_probe(&unwritten.each_ref().map(Vec::as_slice), &probe).is_err());
    served_matches_probe(&probe, &probe).unwrap();
    let inputs = Inputs::new(1, 1);
    assert_eq!(inputs.x0.len(), FLAT);
    assert_eq!(inputs.weights.len(), MIXES * FLAT);
}

#[test]
fn sinkhorn_hc_matches_the_oracles_at_served_shapes() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| oracle_bucket(session, rows, &Inputs::new(rows, 0x5eed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = oracle_bucket(session, 32, &Inputs::extreme(32, 0xe0))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_oracle"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_oracle", receipt)?;
        Ok(())
    })
    .expect("Sinkhorn hyper-connection oracle comparison");
}

#[test]
fn sinkhorn_hc_replays_on_poisoned_buffers() {
    kernel_lab::gpu_test(2 << 30, |session| {
        let mut rows = BUCKETS
            .iter()
            .map(|&rows| replay_bucket(session, rows, &Inputs::new(rows, 0xfeed + rows as u64)))
            .collect::<LabResult<Vec<_>>>()?;
        let mut extreme = replay_bucket(session, 32, &Inputs::extreme(32, 0xfe))?;
        extreme["extreme_common_offset"] = json!(1e8);
        rows.push(extreme);
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("sinkhorn_hc_poison"));
        receipt.insert("buckets".into(), json!(rows));
        session.receipt("sinkhorn_hc_poison", receipt)?;
        Ok(())
    })
    .expect("Sinkhorn hyper-connection poisoned replays");
}
