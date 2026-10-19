//! Exact lagged split comparisons, f64 gates, poisoned replay and row status.
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType};
use kernel_core::oracle::check_numeric;
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm::RmsNormBuffers;
use kernel_core::sinkhorn_hyper_connection::{LaggedMixCollapseBuffers, MixScratch, MixerWeights};
use kernel_core::sinkhorn_hyper_connection_oracle::{self as oracle, MixWeights};
use kernel_harness::conformance::Probes;
use kernel_harness::kit::{Outcome, cases::sinkhorn_lagged as kit};
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{self, CopyProducer, LiveProducer, Replay, WriteMap};
use kernel_harness::{LabResult, Session};
use kernel_lab_families::{
    sinkhorn_hc_lagged_boundary as boundary, sinkhorn_hc_lagged_split as family,
};
use kernels_sm120::backend::Sm120;
use kernels_sm120::sinkhorn_hc::{Geometry, lagged, lagged_split};
use runtime_cuda::{DeviceBuffer, Module};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

const H: usize = 5120;
const N: usize = 4;
const FLAT: usize = H * N;
const MIXES: usize = 24;
const CHAINS: usize = FLAT / 128;
const WIDTHS: [usize; 7] = [H, N, N, N * N, MIXES, 1, N * N];
const TYPES: [DType; 7] = [
    DType::Bf16,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
    DType::F32,
];
const NAMES: [&str; 7] = [
    "collapsed",
    "pre",
    "post",
    "comb",
    "projection",
    "rstd",
    "cells",
];

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn floats(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn bf16(v: &[u8]) -> Vec<u16> {
    v.chunks_exact(2)
        .map(|v| u16::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().copied().map(f64::from).collect()
}
fn reference(probe: bool) -> lagged::Entry {
    lagged::lab_catalog()
        .into_iter()
        .find(|e| e.op == lagged::Op::MixCollapse { probe })
        .unwrap()
}
fn backend(capacity: u32) -> Sm120 {
    Sm120::new()
        .with_sinkhorn_lagged(lagged::catalog())
        .with_sinkhorn_lagged_split([lagged_split::entry(
            lagged::LAB_PROFILE,
            Geometry {
                capacity,
                ..lagged::LAB
            },
            false,
        )])
}

#[test]
fn cpu_split_routing_scratch_and_oracle_declaration() {
    let backend = backend(32);
    let query = reference(false).query;
    let op = backend
        .sinkhorn_lagged_mix_collapse(
            &query,
            Need::rows(128).with_exact(Invariance::any_rows(128)),
        )
        .unwrap();
    assert_eq!(
        op.info().bytes_of("scratch.partials"),
        Some(32 * MIXES as u64 * CHAINS as u64 * 4)
    );
    assert_eq!(op.info().bytes_of("scratch.tickets"), Some(128));
    let view = |handle| View {
        handle,
        bytes: u64::MAX,
    };
    let buffers = LaggedMixCollapseBuffers {
        x: view(1),
        mixer: MixerWeights {
            fn_: view(2),
            base: view(3),
            scale: view(4),
        },
        pre_in: view(5),
        collapsed: view(6),
        pre: view(7),
        post: view(8),
        comb: view(9),
        scratch: MixScratch {
            partials: Some(view(10)),
            tickets: Some(view(11)),
        },
        live_rows: view(12),
    };
    for rows in [1, 2, 31, 32, 33, 128] {
        let launch = op
            .launches(&buffers, view(0), Counts::rows(rows))
            .unwrap()
            .remove(0);
        assert_eq!(
            launch.symbol,
            if rows <= 32 {
                family::entry(false).symbol()
            } else {
                reference(false).symbol
            }
        );
        assert_eq!(launch.arguments.0.len(), if rows <= 32 { 12 } else { 10 });
        assert_eq!(
            launch.plan.grid,
            if rows <= 32 {
                [20, rows, 1]
            } else {
                [rows, 1, 1]
            }
        );
    }
    let missing = LaggedMixCollapseBuffers {
        scratch: MixScratch {
            partials: None,
            tickets: None,
        },
        ..buffers
    };
    assert!(op.launches(&missing, view(0), Counts::rows(32)).is_err());
    assert!(op.launches(&missing, view(0), Counts::rows(33)).is_ok());
    let prepared = backend
        .sinkhorn_lagged_mix_collapse(&query, Need::rows(4))
        .unwrap();
    kit::mix_collapse_case(query, 4, prepared)
        .unwrap()
        .validate()
        .unwrap();
}

struct Inputs {
    x: Vec<u16>,
    weights: Vec<f32>,
    base: Vec<f32>,
    scale: Vec<f32>,
    pre: Vec<f32>,
}
impl Inputs {
    fn new(rows: usize, special: u32) -> Self {
        let mut x = kernel_harness::kit::bf16_values(42, rows * FLAT);
        for (i, value) in x.iter_mut().enumerate() {
            match i % 509 {
                0 => *value = 0,
                1 => *value = 0x8000,
                2 => *value = 1,
                3 => *value = 0x8001,
                _ => {}
            }
        }
        let mut v = Self {
            x,
            weights: kernel_harness::kit::bf16_values(41, MIXES * FLAT)
                .into_iter()
                .map(|v| bf16_to_f32(v) / 128.0 + 2_f32.powi(-22))
                .collect(),
            base: (0..MIXES).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect(),
            scale: vec![0.5, 0.25, 1.0],
            pre: (0..rows * N)
                .map(|i| 0.0625 + (i % 7) as f32 * 0.125 + 2_f32.powi(-20))
                .collect(),
        };
        match special {
            1 => {
                v.base[2 * N..].fill(1e8);
                v.scale[2] = 0.0;
            }
            2 => {
                for (row, value) in [0x7fc1, 0x7f80, 0xff80, 0x7f7f].into_iter().enumerate() {
                    v.x[row * FLAT + row] = value;
                }
                v.pre[4 * N] = f32::NAN;
                v.pre[5 * N] = f32::INFINITY;
                v.pre[6 * N] = f32::NEG_INFINITY;
            }
            3 => {
                v.weights[0] = f32::NAN;
                v.weights[FLAT] = f32::INFINITY;
            }
            _ => {}
        }
        v
    }
}

struct Fixture {
    buffers: Vec<DeviceBuffer>,
    rows: usize,
}
impl Fixture {
    fn new(session: &Session, rows: usize, input: &Inputs) -> LabResult<Self> {
        let mut buffers = vec![
            session.upload_bf16(&input.x)?,
            session.upload_f32(&input.weights)?,
            session.upload_f32(&input.base)?,
            session.upload_f32(&input.scale)?,
            session.upload_f32(&input.pre)?,
        ];
        buffers.push(session.allocate(4)?); // live
        for _ in 0..2 {
            for (width, ty) in WIDTHS.into_iter().zip(TYPES) {
                buffers
                    .push(session.allocate(rows * width * if ty == DType::Bf16 { 2 } else { 4 })?);
            }
        }
        buffers.push(session.allocate(rows * MIXES * CHAINS * 4)?); // partials: 20
        buffers.push(session.upload_f32(&vec![0.0; rows])?); // tickets: 21
        buffers.push(session.upload_bf16(&vec![f32_to_bf16(1.0); H])?); // norm weights: 22
        for _ in 0..2 {
            buffers.push(session.allocate(rows * H * 2)?); // norm: 23, 27
            buffers.push(session.allocate(rows * H)?); // codes
            buffers.push(session.allocate(rows * H / 32)?); // scales
            buffers.push(session.allocate(rows * 4)?); // status
        }
        Ok(Self { buffers, rows })
    }
    fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64, b))
                .collect(),
        }
    }
    fn live(&self, session: &Session, live: usize) -> LabResult<()> {
        self.buffers[5]
            .copy_from_bytes(&(live as u32).to_le_bytes(), &session.stream)
            .map_err(err)
    }
    fn fire(
        &self,
        session: &Session,
        module: &Module,
        which: u32,
        probe: bool,
        pdl: bool,
    ) -> LabResult<()> {
        let output = if which == 0 { 6 } else { 13 };
        let mut values: Vec<_> = (0..5)
            .chain(output..output + if probe { 7 } else { 4 })
            .map(Argument::DeviceBuffer)
            .collect();
        let (symbol, mut plan, args) = if which == 0 {
            let entry = reference(probe);
            values.push(Argument::DeviceBuffer(5));
            (
                entry.symbol.clone(),
                entry.plan(self.rows as u32, 0).map_err(err)?,
                entry.pack_arguments(values).map_err(err)?,
            )
        } else {
            let entry = if which == 2 {
                family::mutant()
            } else {
                family::entry(probe)
            };
            values.extend([20, 21, 5].map(Argument::DeviceBuffer));
            (
                entry.symbol(),
                entry.plan(self.rows as u32, 0).map_err(err)?,
                entry.pack_arguments(values).map_err(err)?,
            )
        };
        plan.pdl = pdl;
        launch(
            &session.context,
            &module.function(&symbol).map_err(err)?,
            &session.stream,
            plan,
            &args,
            &self.bindings(),
        )
    }
    fn outputs(&self, session: &Session, first: usize, count: usize) -> LabResult<Vec<Vec<u8>>> {
        (first..first + count)
            .map(|i| poison::read(session, &self.buffers[i]))
            .collect()
    }
    fn boundary(&self, session: &Session, module: &Module, which: usize) -> LabResult<()> {
        let out = 23 + which * 4;
        let norm = boundary::norm();
        let args = norm
            .arguments(RmsNormBuffers {
                x: (6 + which * 7) as u64,
                weight: 22,
                y: out as u64,
                residual: None,
                live_rows: 5,
                status: None,
                status_row_offset: 0,
            })
            .map_err(err)?;
        launch(
            &session.context,
            &module.function(boundary::NORM).map_err(err)?,
            &session.stream,
            norm.plan(self.rows as u32).map_err(err)?,
            &args,
            &self.bindings(),
        )?;
        let quant = boundary::quantize();
        let args = quant
            .arguments(
                out as u64,
                (out + 1) as u64,
                Some((out + 2) as u64),
                5,
                (out + 3) as u64,
                0,
            )
            .map_err(err)?;
        launch(
            &session.context,
            &module.function(&quant.symbol).map_err(err)?,
            &session.stream,
            quant.plan(self.rows as u32).map_err(err)?,
            &args,
            &self.bindings(),
        )
    }
}

fn equal(actual: &[Vec<u8>], expected: &[Vec<u8>]) -> LabResult<()> {
    for ((name, a), b) in NAMES.iter().zip(actual).zip(expected) {
        if let Some(i) = a.iter().zip(b).position(|(a, b)| a != b) {
            return Err(format!("{name} byte {i}: {} != {}", a[i], b[i]));
        }
        if a.len() != b.len() {
            return Err(format!("{name}: length mismatch"));
        }
    }
    Ok(())
}

fn oracle_check(input: &Inputs, output: &[Vec<u8>], rows: usize) -> LabResult<()> {
    let query = reference(true).query;
    let x: Vec<_> = input.x.iter().map(|&v| f64::from(bf16_to_f32(v))).collect();
    let weights = widen(&input.weights);
    let base = widen(&input.base);
    let scale = widen(&input.scale);
    let weights = MixWeights {
        projection: &weights,
        base: &base,
        scale: &scale,
    };
    let check = |actual: Vec<f64>, expected, dtype| {
        check_numeric(&actual, &expected, dtype).map_err(|e| format!("f64 gate: {e:?}"))
    };
    check(
        widen(&floats(&output[4])),
        oracle::projection(query, &x, weights).map_err(err)?,
        DType::F32,
    )?;
    check(
        widen(&floats(&output[5])),
        oracle::rstd(query, &x).map_err(err)?,
        DType::F32,
    )?;
    let expected = oracle::lagged_mix_collapse(query, &x, &input.pre, weights).map_err(err)?;
    check(
        bf16(&output[0])
            .into_iter()
            .map(|v| f64::from(bf16_to_f32(v)))
            .collect(),
        expected.collapsed,
        DType::Bf16,
    )?;
    check(widen(&floats(&output[1])), expected.pre, DType::F32)?;
    check(widen(&floats(&output[2])), expected.post, DType::F32)?;
    check(widen(&floats(&output[3])), expected.comb, DType::F32)?;
    assert_eq!(
        bf16(&output[0]),
        oracle::collapse(query, &input.x, &input.pre).map_err(err)?
    );
    let projected = floats(&output[4]);
    let rstd = floats(&output[5]);
    let logits: Vec<_> = projected
        .iter()
        .enumerate()
        .map(|(i, &v)| v * rstd[i / MIXES])
        .collect();
    let stage =
        oracle::lagged_mix_stage_from_kernel_logits(query, &logits, &input.base, &input.scale)
            .map_err(err)?;
    check(widen(&floats(&output[6])), stage.cells, DType::F32)?;
    check(widen(&floats(&output[1])), stage.pre, DType::F32)?;
    check(widen(&floats(&output[2])), stage.post, DType::F32)?;
    let cells = floats(&output[6]);
    let comb: Vec<_> = cells
        .chunks_exact(N * N)
        .flat_map(|c| oracle::sinkhorn_f32(lagged::LAB.mix.order, c, N, 20, 1e-6))
        .collect();
    assert_eq!(bytes(&comb), output[3]);
    assert_eq!(output[0].len(), rows * H * 2);
    Ok(())
}

fn load(session: &Session, name: &str) -> LabResult<Module> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_LAGGED_SPLIT_CUBINS")
            .ok_or("prebuilt lagged split cubins required")?,
    );
    session.load_cubin(&std::fs::read(root.join(format!("{name}.cubin"))).map_err(err)?)
}

fn run_bucket(session: &Session, rows: usize, special: u32) -> LabResult<Value> {
    let input = Inputs::new(rows, special);
    let fixture = Fixture::new(session, rows, &input)?;
    let old = load(session, &lagged::probe_cubin())?;
    let new = load(session, family::CUBIN)?;
    let norm = load(session, boundary::CUBIN)?;
    let counts = if rows == 32 || rows == 128 {
        vec![0, 1, rows / 2, rows - 1, rows]
    } else {
        vec![rows]
    };
    let mut records = vec![];
    for live in counts {
        fixture.live(session, live)?;
        poison::poison(
            session,
            &(6..20).map(|i| &fixture.buffers[i]).collect::<Vec<_>>(),
        )?;
        fixture.fire(session, &old, 0, true, false)?;
        let mut expected = fixture.outputs(session, 6, 7)?;
        for (i, output) in expected.iter_mut().enumerate() {
            let size = if TYPES[i] == DType::Bf16 { 2 } else { 4 };
            output[live * WIDTHS[i] * size..].fill(0xa5);
        }
        let served_old = load(session, &lagged::served_cubin())?;
        poison::poison(
            session,
            &(6..10).map(|i| &fixture.buffers[i]).collect::<Vec<_>>(),
        )?;
        for i in 6..10 {
            fixture.buffers[i]
                .copy_from_bytes(&vec![0xa5; fixture.buffers[i].byte_len()], &session.stream)
                .map_err(err)?;
        }
        fixture.fire(session, &served_old, 0, false, false)?;
        equal(&fixture.outputs(session, 6, 4)?, &expected[..4])?;
        let graph = session.capture(|_| fixture.fire(session, &new, 1, true, true))?;
        let mut replay = Replay::new(session);
        for i in 0..7 {
            replay = replay.output_with(
                NAMES[i],
                &fixture.buffers[13 + i],
                WriteMap::of(TYPES[i]).prefix(live * WIDTHS[i])?,
                &[0xa5],
            );
        }
        replay = replay
            .output_with(
                "partials",
                &fixture.buffers[20],
                WriteMap::of(DType::F32).prefix(live * MIXES * CHAINS)?,
                &[0xa5],
            )
            .state(
                "tickets",
                &fixture.buffers[21],
                vec![0; rows * 4],
                WriteMap::of(DType::I32).none()?,
                vec![],
            );
        replay.eager(|| fixture.fire(session, &new, 1, true, false))?;
        equal(&fixture.outputs(session, 13, 7)?, &expected)?;
        replay.relaunch(|| fixture.fire(session, &new, 1, true, false))?;
        replay.graph(&graph)?;
        replay.pdl(
            || fixture.fire(session, &old, 0, true, false),
            || fixture.fire(session, &new, 1, true, true),
            true,
        )?;
        equal(&fixture.outputs(session, 13, 7)?, &expected)?;
        if special <= 1 && live == rows && [1, 32, 128].contains(&rows) {
            oracle_check(&input, &expected, rows)?;
        }
        poison::poison(
            session,
            &(13..17).map(|i| &fixture.buffers[i]).collect::<Vec<_>>(),
        )?;
        for i in 13..17 {
            fixture.buffers[i]
                .copy_from_bytes(&vec![0xa5; fixture.buffers[i].byte_len()], &session.stream)
                .map_err(err)?;
        }
        fixture.fire(session, &new, 1, false, false)?;
        equal(&fixture.outputs(session, 13, 4)?, &expected[..4])?;
        // The mixer and BF16 norm have no status ABI. The unchanged next
        // quantizer preserves prior flags and flags non-finite normed rows.
        let prior: Vec<u32> = (0..rows)
            .map(|r| if r % 3 == 1 { 0x40 } else { 0 })
            .collect();
        let prior_bytes: Vec<_> = prior.iter().flat_map(|v| v.to_le_bytes()).collect();
        for first in [23, 27] {
            poison::poison(
                session,
                &(first..first + 3)
                    .map(|i| &fixture.buffers[i])
                    .collect::<Vec<_>>(),
            )?;
            fixture.buffers[first + 3]
                .copy_from_bytes(&prior_bytes, &session.stream)
                .map_err(err)?;
        }
        fixture.boundary(session, &norm, 0)?;
        fixture.boundary(session, &norm, 1)?;
        let a = fixture.outputs(session, 23, 4)?;
        let b = fixture.outputs(session, 27, 4)?;
        assert_eq!(
            a, b,
            "norm/quantizer/status rows={rows} live={live} special={special}"
        );
        let values: Vec<_> = bf16(&a[0]).into_iter().map(bf16_to_f32).collect();
        let status = boundary::quantize()
            .query
            .row_status(&values[..live * H], &prior[..live])
            .map_err(err)?;
        let mut status_bytes: Vec<_> = status.iter().flat_map(|v| v.to_le_bytes()).collect();
        status_bytes.extend_from_slice(&prior_bytes[live * 4..]);
        assert_eq!(a[3], status_bytes);
        let mut record = BTreeMap::new();
        replay.record(&mut record);
        records.push(json!({"live":live,"replay":record,"row_status_exact":true}));
    }
    if rows == 1 && special == 0 {
        let mutant = load(session, family::MUTANT)?;
        fixture.fire(session, &mutant, 2, true, false)?;
        assert_ne!(
            fixture.outputs(session, 17, 1)?,
            fixture.outputs(session, 10, 1)?,
            "reversed-chain mutant escaped projection bits"
        );
    }
    Ok(json!({"rows":rows,"special":special,"runs":records}))
}

#[test]
fn lagged_split_bitwise_oracle_status_on_gpu() {
    let ran = kernel_harness::gpu_test(512 << 20, |session| {
        let root = PathBuf::from(
            std::env::var_os("CHOPINFER_LAGGED_SPLIT_CUBINS").ok_or("prebuilt cubins required")?,
        );
        kernel_build::report::BuildReport::read(&root.join("kernel-build-report.json"))
            .and_then(|r| r.verify_current())
            .map_err(err)?;
        let mut cases = vec![];
        for rows in (1..=33).chain([128]) {
            cases.push(run_bucket(session, rows, 0)?);
        }
        for special in 1..=3 {
            cases.push(run_bucket(session, 32, special)?);
        }
        session.receipt(
            "lagged-split-exact",
            BTreeMap::from([("cases".into(), json!(cases))]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required GPU comparison skipped");
    }
}

#[test]
fn lagged_split_properties_on_gpu() {
    let ran = kernel_harness::gpu_test(512 << 20, |session| {
        let query = reference(false).query;
        let need = Need::rows(4).with_exact(Invariance::any_rows(4));
        let prepared = backend(32)
            .sinkhorn_lagged_mix_collapse(&query, need)
            .map_err(err)?;
        let case = kit::mix_collapse_case(query, 4, prepared)?;
        let old = load(session, &lagged::served_cubin())?;
        let new = load(session, family::CUBIN)?;
        let probes = load(session, kernels_sm120::probes::CUBIN)?;
        let probes = Probes {
            live: LiveProducer::with_function(
                session,
                probes.function(kernels_sm120::probes::LIVE).map_err(err)?,
            )?,
            copy: Some(CopyProducer::with_function(
                probes.function(kernels_sm120::probes::COPY).map_err(err)?,
            )),
        };
        let lookup = |symbol: &str| {
            new.function(symbol)
                .or_else(|_| old.function(symbol))
                .map_err(err)
        };
        let checks = case.checks(need, Some(session), Some(&lookup), Some(&probes));
        assert!(!checks.is_empty());
        for check in &checks {
            assert!(
                matches!(check.outcome, Outcome::Passed(_)),
                "{}: {:?}",
                check.name,
                check.outcome
            );
        }
        session.receipt(
            "lagged-split-properties",
            BTreeMap::from([(
                "checks".into(),
                json!(checks.iter().map(|c| &c.name).collect::<Vec<_>>()),
            )]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required GPU properties skipped");
    }
}

#[test]
fn lagged_split_times_on_gpu() {
    let ran = kernel_harness::gpu_test(512 << 20, |session| {
        let old = load(session, &lagged::served_cubin())?;
        let new = load(session, family::CUBIN)?;
        let mut times = vec![];
        for rows in [1, 32, 128] {
            let fixture = Fixture::new(session, rows, &Inputs::new(rows, 0))?;
            fixture.live(session, rows)?;
            fixture.fire(session, &old, 0, false, false)?;
            for i in 13..17 {
                fixture.buffers[i]
                    .copy_from_bytes(&vec![0xa5; fixture.buffers[i].byte_len()], &session.stream)
                    .map_err(err)?;
            }
            fixture.fire(session, &new, 1, false, false)?;
            equal(
                &fixture.outputs(session, 13, 4)?,
                &fixture.outputs(session, 6, 4)?,
            )?;
            let a = kernel_harness::time_named(session, "lagged one-CTA", 2, 15, |_| {
                fixture.fire(session, &old, 0, false, false)
            })?;
            let b = kernel_harness::time_named(session, "lagged split", 2, 15, |_| {
                fixture.fire(session, &new, 1, false, false)
            })?;
            equal(
                &fixture.outputs(session, 13, 4)?,
                &fixture.outputs(session, 6, 4)?,
            )?;
            times.push(json!({"rows":rows,"one_cta_ms":a.median_ms,"split_ms":b.median_ms}));
        }
        session.receipt(
            "lagged-split-timing",
            BTreeMap::from([("buckets".into(), json!(times))]),
        )?;
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required timings skipped");
    }
}
