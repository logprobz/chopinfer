use super::{
    ConfidenceHead, ConfidenceHeadBuffers, Counts, MarkovStep, MarkovStepBuffers, Need, OwnOps,
    Sm120Own, View, cubin, view,
};
use kernel_core::backend::Launch;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::oracle::{OracleOutput, check_numeric, max_bound_use};
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::poison::{LiveProducer, Replay, Rewrites, WriteMap};
use kernel_harness::{LabResult, Session};
use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES, Geometry};
use model_deepseek41_contract::own_ops::dspark::{confidence_reference, markov_reference};
use runtime_cuda::{DeviceBuffer, Stream};
use serde_json::json;
use std::collections::BTreeMap;

#[path = "probe.rs"]
mod probe;

#[path = "parallel.rs"]
mod parallel;

// More than two eight-row passes of the tiled Markov step.
const ROWS: u32 = 19;

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    buffer
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}

fn bf16(len: usize, seed: u64) -> Vec<u16> {
    kernel_harness::wide_values(len, seed)
        .into_iter()
        .map(|v| f32_to_bf16(v * 0.125))
        .collect()
}
fn b16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn b32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn chain(x: &[u16], w: &[u16]) -> f32 {
    chain_f32(&x.iter().copied().map(bf16_to_f32).collect::<Vec<_>>(), w)
}

fn chain_f32(x: &[f32], w: &[u16]) -> f32 {
    let mut sum = 0.0_f32;
    for (x, w) in x.chunks(128).zip(w.chunks(128)) {
        let mut partial = 0.0_f32;
        for (&x, &w) in x.iter().zip(w) {
            partial = x.mul_add(bf16_to_f32(w), partial);
        }
        sum += partial;
    }
    sum
}

struct Case {
    geometry: &'static str,
    case: &'static str,
    hidden_width: Option<u32>,
    launch: Launch,
    images: [Vec<u8>; 3],
    stride: usize,
    width: usize,
    in_place: bool,
    oracle: Vec<OracleOutput>,
    exact: Vec<Vec<f32>>,
}

fn run(session: &Session, case: Case) -> LabResult<()> {
    let module = session.load_cubin(if case.hidden_width.is_some() {
        super::parallel_cubin()
    } else if case.launch.symbol == crate::markov_tiled::SYMBOL {
        super::tiled_cubin()
    } else {
        cubin()
    })?;
    let producer_module = session.load_cubin(probe::cubin())?;
    let producer = LiveProducer::with_function(
        session,
        producer_module
            .function(probe::SYMBOL)
            .map_err(|e| e.to_string())?,
    )?;
    let inputs: Vec<_> = case
        .images
        .iter()
        .map(|image| upload(session, image))
        .collect::<LabResult<_>>()?;
    let staged: Vec<_> = case
        .images
        .iter()
        .map(|image| upload(session, image))
        .collect::<LabResult<_>>()?;
    let rewrites = Rewrites::new(session, &staged.iter().zip(&inputs).collect::<Vec<_>>())?;
    let live = session.allocate(4)?;
    let allocated = session.allocate(ROWS as usize * case.stride * 4)?;
    let output = if case.in_place {
        &inputs[2]
    } else {
        &allocated
    };
    let bindings = Bindings {
        buffers: BTreeMap::from([
            (1, &inputs[0]),
            (2, &inputs[1]),
            (3, &inputs[2]),
            (4, output),
            (5, &live),
        ]),
    };
    let function = module
        .function(&case.launch.symbol)
        .map_err(|e| e.to_string())?;
    let invoke = |stream: &Stream| {
        launch(
            &session.context,
            &function,
            stream,
            case.launch.plan,
            &case.launch.arguments,
            &bindings,
        )
    };
    let graph = session.capture(invoke)?;
    for n in [0, 1, 3, 8, 9, ROWS] {
        live.copy_from_bytes(&n.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        let ranges: Vec<_> = (0..n as usize)
            .map(|r| r * case.stride..r * case.stride + case.width)
            .collect();
        let mut replay = if case.in_place {
            Replay::new(session).state(
                "head",
                output,
                case.images[2].clone(),
                WriteMap::of(DType::F32).none()?,
                ranges.iter().map(|r| r.start * 4..r.end * 4).collect(),
            )
        } else {
            Replay::new(session).output("head", output, WriteMap::of(DType::F32).ranges(ranges)?)
        };
        replay.eager(|| invoke(&session.stream))?;
        replay.relaunch(|| invoke(&session.stream))?;
        for _ in 0..3 {
            replay.graph(&graph)?;
        }
        replay.pdl(
            || producer.launch_rewriting(session, &live, &rewrites),
            || invoke(&session.stream),
            case.launch.plan.pdl,
        )?;
        let actual = session.download_f32(output, ROWS as usize * case.stride)?;
        let mut worst = 0.0_f64;
        for r in 0..n as usize {
            let got = &actual[r * case.stride..][..case.width];
            check_numeric(
                &got.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                &case.oracle[r],
                DType::F32,
            )
            .map_err(|e| format!("{} row {r}: {e:?}", case.launch.symbol))?;
            for (col, (&got, &expected)) in got.iter().zip(&case.exact[r]).enumerate() {
                if got.to_bits() != expected.to_bits() {
                    return Err(format!(
                        "{} row {r} col {col}: {got} != chain128 {expected}",
                        case.launch.symbol
                    ));
                }
            }
            worst = worst.max(max_bound_use(got, &case.oracle[r], DType::F32));
        }
        let mut receipt = BTreeMap::from([
            ("geometry".into(), json!(case.geometry)),
            ("case".into(), json!(case.case)),
            ("hidden_width".into(), json!(case.hidden_width)),
            ("op".into(), json!(case.launch.symbol)),
            ("live_rows".into(), json!(n)),
            ("prepared_rows".into(), json!(ROWS)),
            ("stride".into(), json!(case.stride)),
            ("width".into(), json!(case.width)),
            ("in_place".into(), json!(case.in_place)),
            ("max_bound_use".into(), json!(worst)),
            ("chain128_bitwise".into(), json!(true)),
            ("graph_replays".into(), json!(3)),
        ]);
        replay.record(&mut receipt);
        session.receipt("deepseek41-dspark-head", receipt)?;
    }
    Ok(())
}

fn markov(
    session: &Session,
    geometry: &'static Geometry,
    vocab: u32,
    in_place: bool,
    cancellation: bool,
) -> LabResult<()> {
    let rank = geometry.shape.dspark.markov_rank;
    let op = MarkovStep {
        rank,
        vocab_rows: vocab,
        logits_stride: 5 * vocab,
    };
    let mut x = bf16((ROWS * rank) as usize, 11);
    let mut weight = bf16((vocab * rank) as usize, 23);
    let base = if cancellation {
        vec![1.25; (ROWS * op.logits_stride) as usize]
    } else {
        kernel_harness::wide_values((ROWS * op.logits_stride) as usize, 41)
    };
    if cancellation {
        x.fill(f32_to_bf16(1.0));
        weight.fill(f32_to_bf16(1.0));
        for w in weight.chunks_exact_mut(rank as usize) {
            w[0] = f32_to_bf16(33554432.0);
            w[128] = f32_to_bf16(-33554432.0);
        }
    }
    let images = [b16(&x), b16(&weight), b32(&base)];
    let buffers = MarkovStepBuffers {
        embedding: view(1, images[0].len() as u64),
        weight: view(2, images[1].len() as u64),
        base: view(3, images[2].len() as u64),
        output: view(if in_place { 3 } else { 4 }, images[2].len() as u64),
        live_rows: view(5, 4),
    };
    let launch = Sm120Own::new(geometry)
        .dspark_markov_step(&op, Need::rows(ROWS))
        .map_err(|e| e.to_string())?
        .launches(&buffers, View::EMPTY, Counts::rows(ROWS))
        .map_err(|e| e.to_string())?
        .remove(0);
    let mut oracle = Vec::new();
    let mut exact = Vec::new();
    for r in 0..ROWS as usize {
        let x = &x[r * rank as usize..][..rank as usize];
        let base = &base[r * op.logits_stride as usize..][..vocab as usize];
        oracle.push(markov_reference(&op, x, &weight, base).map_err(|e| e.to_string())?);
        exact.push(
            weight
                .chunks_exact(rank as usize)
                .zip(base)
                .map(|(w, base)| chain(x, w) + base)
                .collect(),
        );
    }
    run(
        session,
        Case {
            geometry: geometry.id,
            case: if cancellation {
                "markov_chain128_control"
            } else {
                "markov_random"
            },
            hidden_width: None,
            launch,
            images,
            stride: op.logits_stride as usize,
            width: vocab as usize,
            in_place,
            oracle,
            exact,
        },
    )
}

fn confidence(
    session: &Session,
    geometry: &'static Geometry,
    rounding_control: bool,
) -> LabResult<()> {
    let (hidden, rank) = (geometry.shape.hidden, geometry.shape.dspark.markov_rank);
    let op = ConfidenceHead {
        hidden,
        markov_rank: rank,
        hidden_stride: hidden * 5,
        score_stride: 5,
    };
    let mut x = kernel_harness::wide_values((ROWS * op.hidden_stride) as usize, 31);
    let mut markov = bf16((ROWS * rank) as usize, 47);
    let mut weight = bf16((hidden + rank) as usize, 67);
    if rounding_control {
        x.fill(0.0);
        markov.fill(0);
        weight.fill(0);
        weight[0] = f32_to_bf16(1.0);
        for row in 0..ROWS as usize {
            x[row * op.hidden_stride as usize] = f32::from_bits(0x3f80_8000); // Exactly 1.00390625.
        }
    }
    let images = [b32(&x), b16(&markov), b16(&weight)];
    let buffers = ConfidenceHeadBuffers {
        hidden: view(1, images[0].len() as u64),
        markov: view(2, images[1].len() as u64),
        weight: view(3, images[2].len() as u64),
        scores: view(4, u64::from(ROWS * 5 * 4)),
        live_rows: view(5, 4),
    };
    let launch = Sm120Own::new(geometry)
        .dspark_confidence(&op, Need::rows(ROWS))
        .map_err(|e| e.to_string())?
        .launches(&buffers, View::EMPTY, Counts::rows(ROWS))
        .map_err(|e| e.to_string())?
        .remove(0);
    let mut oracle = Vec::new();
    let mut exact = Vec::new();
    for r in 0..ROWS as usize {
        let x = &x[r * op.hidden_stride as usize..][..hidden as usize];
        let m = &markov[r * rank as usize..][..rank as usize];
        oracle.push(confidence_reference(&op, x, m, &weight).map_err(|e| e.to_string())?);
        let input = x
            .iter()
            .copied()
            .chain(m.iter().copied().map(bf16_to_f32))
            .collect::<Vec<_>>();
        exact.push(vec![chain_f32(&input, &weight)]);
    }
    run(
        session,
        Case {
            geometry: geometry.id,
            case: if rounding_control {
                "confidence_hidden_rounding_control"
            } else {
                "confidence_random"
            },
            hidden_width: Some(hidden),
            launch,
            images,
            stride: 5,
            width: 1,
            in_place: false,
            oracle,
            exact,
        },
    )
}

fn oracle_geometry() -> LabResult<&'static Geometry> {
    let id = std::env::var("CHOPINFER_DEEPSEEK41_HEAD_GEOMETRY")
        .unwrap_or_else(|_| "synthetic_small".into());
    let geometry = GEOMETRIES
        .iter()
        .find(|g| g.id == id)
        .ok_or_else(|| format!("head test geometry {id} was not compiled"))?;
    if id == "synthetic_small" && geometry.layers() != 12 {
        return Err("DSpark synthetic head checks require the 12-layer fixture".into());
    }
    Ok(geometry)
}

#[test]
#[ignore = "requires CHOPINFER_DEEPSEEK41_HEAD_GEOMETRY selected by the staged job"]
fn head_cpu_oracle_geometry_is_compiled() {
    let geometry = oracle_geometry().unwrap();
    eprintln!(
        "DSPARK_HEAD_GEOMETRY {}",
        json!({
            "id": geometry.id,
            "layers": geometry.layers(),
            "hidden": geometry.shape.hidden,
            "markov_rank": geometry.shape.dspark.markov_rank,
            "vocab_rows_tp4": geometry.shape.vocab / 4,
        })
    );
}

#[test]
fn head_gpu_markov_and_confidence_match_oracles_and_replay() {
    kernel_harness::gpu_test(512 << 20, |session| {
        let geometry = oracle_geometry()?;
        for alias in [false, true] {
            markov(session, geometry, 131, alias, false)?;
            markov(session, geometry, 1, alias, true)?;
        }
        markov(session, geometry, geometry.shape.vocab / 4, false, false)?;
        confidence(session, geometry, false)?;
        if geometry.id != FLASH.id {
            confidence(session, &FLASH, false)?;
        }
        confidence(session, &FLASH, true)
    })
    .unwrap();
}
