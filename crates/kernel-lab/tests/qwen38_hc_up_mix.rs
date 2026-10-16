//! The fused decode hyper-connection up mix against the three entries it
//! replaces (bottleneck row, short-K up projection, stream mix row): the
//! same inputs must give the same BF16 mix bit for bit at every live count,
//! eagerly and in PDL graphs, and dead rows stay untouched. Both are within
//! the bounds of the contract's F64 chain (`own_ops::hc`).

use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType};
use kernel_core::transform::Transform;
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, gpu_test};
use kernels_sm120::gemm::fused as gemm;
use model_qwen38_contract::{
    own_ops::hc,
    sm120::{gemm as model_gemm, row},
};
use runtime_cuda::{DeviceBuffer, Function};
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase, WriteMap};

const ROWS: usize = 32;
const WIDTH: usize = 2560;
const WIDE: usize = 4 * WIDTH;
const RANK: usize = 320;
const SENTINEL: u16 = 0xffff;

const WEIGHT: u64 = 1;
const DOWN: u64 = 2;
const NORMED: u64 = 3;
const LIVE: u64 = 4;
const BOTTLENECK: u64 = 5;
const UP: u64 = 6;
const EXPECTED: u64 = 7;
const FUSED: u64 = 8;

#[path = "common/accumulation.rs"]
mod accumulation;

fn seeded(len: usize, seed: u64, scale: f32) -> Vec<u16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16((((state >> 32) % 513) as i32 - 256) as f32 * scale)
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

fn function(session: &Session, cubin: &[u8], symbol: &str, shared: u32) -> LabResult<Function> {
    let function = session
        .context
        .load_module(cubin)
        .map_err(|e| e.to_string())?
        .function(symbol)
        .map_err(|e| e.to_string())?;
    if shared >= 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(shared)
            .map_err(|e| e.to_string())?;
    }
    Ok(function)
}

#[test]
fn qwen38_hc_up_mix_matches_the_unfused_chain() {
    gpu_test(256 << 20, |session| {
        let rows_cubin: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/row_qwen38.cubin"));
        let gemm_cubin: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/gemm_qwen38_own_sm120.cubin"));
        let row_entry = |name: &str| {
            row::hyper_connection_rows()
                .into_iter()
                .find(|e| e.symbol == format!("chop_row_qwen38_{name}_v1"))
                .unwrap()
        };
        let (bottleneck, mix) = (row_entry("hc_bottleneck"), row_entry("hc_mix"));
        let up = gemm::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|e| e.symbol == "chop_gemm_qwen38_hc_up_decode_v1")
            .unwrap();
        let fused_plan = gemm::hc_up_mix_plan(model_gemm::HC, ROWS as u32).map_err(|e| e.to_string())?;
        let functions = [
            function(session, rows_cubin, &bottleneck.symbol, 0)?,
            function(
                session,
                gemm_cubin,
                &up.symbol,
                up.plan(ROWS as u32)
                    .map_err(|e| e.to_string())?
                    .dynamic_shared_bytes,
            )?,
            function(session, rows_cubin, &mix.symbol, 0)?,
            function(
                session,
                gemm_cubin,
                model_gemm::HC_UP_MIX,
                fused_plan.dynamic_shared_bytes,
            )?,
        ];

        // Grid values, then values over 13 binades whose F32 sums round, so a
        // different accumulation order in either path changes output bits.
        let lives = [1_usize, 2, 3, 8, 16, 31, 32];
        let mut sensitivity = None;
        let mut failures = replay::Failures::default();
        for wide in [false, true] {
        let inputs = |len: usize, seed: u64, scale: f32| {
            if wide {
                accumulation::wide_rows(len, 1, seed)
            } else {
                seeded(len, seed, scale)
            }
        };
        let weight = inputs(WIDE * RANK, 0x5550_4d49, 1.0 / 4096.0);
        let packed = FragmentWeightLayout {
            rows: WIDE,
            k: RANK,
            format: FragmentWeightFormat::Bf16,
        }
        .pack(&bytes(&weight))
        .map_err(|e| e.to_string())?
        .bytes;
        let mut buffers = BTreeMap::new();
        buffers.insert(WEIGHT, upload(session, &packed)?);
        let (down, normed) = (
            inputs(ROWS * RANK, 7, 1.0 / 64.0),
            inputs(ROWS * WIDE, 11, 1.0 / 128.0),
        );
        buffers.insert(DOWN, upload(session, &bytes(&down))?);
        buffers.insert(NORMED, upload(session, &bytes(&normed))?);
        buffers.insert(LIVE, upload(session, &[0; 4])?);
        buffers.insert(BOTTLENECK, upload(session, &vec![0; ROWS * RANK * 2])?);
        buffers.insert(UP, upload(session, &vec![0; ROWS * WIDE * 2])?);
        for id in [EXPECTED, FUSED, FUSED + 1, FUSED + 2] {
            buffers.insert(id, upload(session, &bytes(&vec![SENTINEL; ROWS * WIDTH]))?);
        }
        let bindings = Bindings {
            buffers: buffers.iter().map(|(&k, v)| (k, v)).collect(),
        };
        let run = |index: usize, plan, args| {
            launch::launch(
                &session.context,
                &functions[index],
                &session.stream,
                plan,
                &args,
                &bindings,
            )
        };
        let fused = |id: u64, pdl: bool| -> LabResult<()> {
            let mut plan = fused_plan;
            plan.pdl = pdl;
            run(
                3,
                plan,
                gemm::hc_up_mix_arguments(WEIGHT, DOWN, NORMED, id, LIVE)
                    .map_err(|e| e.to_string())?,
            )
        };
        let compare = |live: usize, label: &str| -> LabResult<()> {
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let expected = session.download_bf16(&buffers[&EXPECTED], ROWS * WIDTH)?;
            let actual = session.download_bf16(&buffers[&FUSED], ROWS * WIDTH)?;
            if actual[..live * WIDTH] != expected[..live * WIDTH] {
                let first = (0..live * WIDTH)
                    .find(|&i| actual[i] != expected[i])
                    .unwrap();
                return Err(format!(
                    "{label}: element {first} is {:#06x}, unfused {:#06x}",
                    actual[first], expected[first]
                ));
            }
            if actual[live * WIDTH..].iter().any(|&v| v != SENTINEL) {
                return Err(format!("{label}: a dead row was written"));
            }
            Ok(())
        };
        for live in lives {
            let result = (|| -> LabResult<()> {
            buffers[&LIVE]
                .copy_from_bytes(&(live as u32).to_ne_bytes(), &session.stream)
                .map_err(|e| e.to_string())?;
            let mut unfused = replay::Checks::new(session)
                .output("bottleneck", &buffers[&BOTTLENECK], WriteMap::of(DType::Bf16).prefix(live * RANK)?)
                .output("up", &buffers[&UP], WriteMap::of(DType::Bf16).prefix(live * WIDE)?)
                .output("mixed", &buffers[&EXPECTED], WriteMap::of(DType::Bf16).prefix(live * WIDTH)?);
            unfused.prepare()?;
            let row_args = |entry: &kernels_sm120::row::hyper_connection::Entry, ids: &[u64]| -> LabResult<_> {
                entry
                    .pack_arguments(ids.iter().map(|&id| Argument::DeviceBuffer(id)).collect())
                    .map_err(|e| e.to_string())
            };
            run(
                0,
                bottleneck
                    .plan(ROWS as u32, live as u32)
                    .map_err(|e| e.to_string())?,
                row_args(&bottleneck, &[DOWN, BOTTLENECK, LIVE])?,
            )?;
            run(
                1,
                up.plan(live as u32).map_err(|e| e.to_string())?,
                up.arguments(
                    live as u32,
                    gemm::Buffers {
                        weights: [WEIGHT, 0, 0, 0],
                        activation: BOTTLENECK,
                        outputs: [UP, 0, 0, 0],
                        ..Default::default()
                    },
                )
                .map_err(|e| e.to_string())?,
            )?;
            run(
                2,
                mix.plan(ROWS as u32, live as u32)
                    .map_err(|e| e.to_string())?,
                row_args(&mix, &[UP, NORMED, EXPECTED, LIVE])?,
            )?;
            unfused.finish(Phase::Eager, false)?;
            let mut calls = (FUSED..FUSED + 3).map(|id| Ok(replay::Checks::new(session)
                .output("fused mix", &buffers[&id], WriteMap::of(DType::Bf16).prefix(live * WIDTH)?)))
                .collect::<LabResult<Vec<_>>>()?;
            replay::phase_all(&mut calls, Phase::Eager, false, || {
                (FUSED..FUSED + 3).try_for_each(|id| fused(id, false))
            })?;
            compare(live, &format!("eager live {live}"))?;
            // Both are the contract's chain within its bounds.
            let op = hc::HcUpMix {
                width: WIDTH as u32,
                streams: 4,
                rank: RANK as u32,
                input: hc::UpMixInput::Down,
            };
            let fused_rows = session.download_bf16(&buffers[&FUSED], ROWS * WIDTH)?;
            hc::hc_up_mix(&op, &weight, &down, &normed, live as u32)
                .map_err(|e| format!("up mix oracle: {}", e.0))?
                .check_bf16(&fused_rows[..live * WIDTH])
                .map_err(|failure| format!("live {live} outside the oracle: {failure:?}"))?;
            let expected = session.download_bf16(&buffers[&EXPECTED], ROWS * WIDTH)?;
            for id in FUSED + 1..FUSED + 3 {
                assert_eq!(session.download_bf16(&buffers[&id], ROWS * WIDTH)?, expected, "fused call {id}");
            }
            let graph = session.capture(|_| (FUSED..FUSED + 3).try_for_each(|id| fused(id, true)))?;
            for iteration in 0..2 {
                replay::phase_all(&mut calls, Phase::Graph, true, || graph.launch(&session.stream).map_err(|e| e.to_string()))?;
                compare(live, &format!("graph live {live} replay {iteration}"))?;
            }
            replay::phase_all(&mut calls, Phase::Pdl, true, || (FUSED..FUSED + 3).try_for_each(|id| fused(id, true)))?;
            for call in calls {
                let mut fields = BTreeMap::from([("live".into(), json!(live)), ("wide".into(), json!(wide))]);
                call.record(&mut fields);
                session.receipt("hc-up-mix-replay", fields)?;
            }
                Ok(())
            })();
            failures.check(&format!("live {live} wide {wide}"), result);
        }
        if wide {
            // The up projection's own inputs: the last unfused run's bottleneck.
            session.stream.synchronize().map_err(|e| e.to_string())?;
            let rows = session.download_bf16(&buffers[&BOTTLENECK], ROWS * RANK)?;
            let probe = session.module(accumulation::PROBE)?;
            let problem = accumulation::Problem::upload(session, &weight, &rows, WIDE, ROWS, RANK)?;
            let long = problem.run(session, &probe, accumulation::Scheme::Chain(None))?;
            let short = problem.run(session, &probe, accumulation::Scheme::Chain(Some(128)))?;
            let differ = long.iter().zip(&short).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            eprintln!("HC up, wide inputs: {differ} of {} outputs differ between one chain and 128-K runs", long.len());
            sensitivity = Some(json!({"outputs": long.len(), "differ": differ}));
        }
        }
        failures.finish()?;
        session.receipt(
            "qwen38-hc-up-mix-g1",
            BTreeMap::from([
                ("symbol".into(), json!(model_gemm::HC_UP_MIX)),
                ("lives".into(), json!(lives)),
                (
                    "bitwise_equal_to".into(),
                    json!([bottleneck.symbol, up.symbol, mix.symbol]),
                ),
                ("graph_determinism_pdl".into(), json!(true)),
                ("wide_inputs".into(), json!(true)),
                ("wide_chain_320_vs_128".into(), json!(sensitivity)),
            ]),
        )?;
        Ok(())
    })
    .expect("Qwen HC up mix G1");
}
