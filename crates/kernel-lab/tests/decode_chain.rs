//! Real-weight MiMo decode sequence, PDL and graph replay.

#[path = "mimo_layers.rs"]
mod mimo_layers;

use kernel_lab::poison::{Evidence, Phase};
use kernel_lab::{LabResult, Session};
use mimo_layers::execution;
use runtime_cuda::GraphExec;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

use kernel_core::contract::{AttentionPhase, RuntimeCounts};
use kernels_sm120::{attention, gemm, moe, row};

#[test]
fn decode_chain_has_pdl_plans_for_b1_b8_b16_and_b32() {
    for (layer, attention_name, qkv_name) in [
        (5, "mimo_ga_bf16_decode", "mimo_qkv_ga"),
        (1, "mimo_swa_decode", "mimo_qkv_swa"),
    ] {
        for batch in [1, 8, 16, 32] {
            let counts = RuntimeCounts {
                m: batch,
                batch,
                context: 68,
                routed_rows: batch * 8,
                split_k: 1,
            };
            let attention = attention::catalog(kernel_lab::catalogs())
                .into_iter()
                .find(|entry| entry.name == attention_name)
                .unwrap();
            assert_eq!(attention.query.phase, AttentionPhase::Decode);
            assert!(attention.query.sink == (layer == 1));
            for name in [qkv_name, "mimo_o", "mimo_router"] {
                let symbol = format!("chop_gemm_{name}_m{batch}_v1");
                let entry = gemm::catalog(kernel_lab::catalogs())
                    .into_iter()
                    .find(|entry| entry.symbol == symbol)
                    .unwrap();
                let mut live = counts;
                live.split_k = gemm::choose_split_k(&entry.key).unwrap();
                assert!(gemm::plan(&entry.key, live).unwrap().pdl);
            }
            for name in ["rms_mimo_hf4096", "combine_mimo4096"] {
                let symbol = format!("chop_row_{name}_v1");
                let entry = row::catalog(kernel_lab::catalogs())
                    .into_iter()
                    .find(|entry| entry.symbol == symbol)
                    .unwrap();
                assert!(entry.plan(batch, counts).unwrap().pdl);
            }
            // Buckets above 16 rows (`serving::DECODE_EXPERT_ROWS`) route
            // as prefill does and run the prefill entries, as served.
            let (phase, tokens) = if batch > 16 {
                (moe::Phase::Prefill, 8192)
            } else {
                assert!(moe::decode_route_plan(batch, batch).unwrap().pdl);
                (moe::Phase::Decode, batch)
            };
            for projection in [moe::Projection::GateUp, moe::Projection::Down] {
                let entry = moe::catalog(kernel_lab::catalogs())
                    .into_iter()
                    .find(|entry| {
                        entry.profile == model_mimo26_contract::sm120::config::PROFILE
                            && entry.phase == phase
                            && entry.tokens_capacity == tokens
                            && entry.projection == projection
                            && entry.path == moe::InputPath::A16Gather
                    })
                    .unwrap();
                let mut live = counts;
                live.split_k = entry.knobs.split_k_capacity;
                assert!(entry.plan(live).unwrap().pdl);
                if let Some(reduction) = entry.reduction_plan(live).unwrap() {
                    assert!(reduction.pdl);
                }
            }
        }
    }
}

#[test]
fn decode_chain_graph_replay() {
    let Ok(checkpoint) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
        kernel_lab::skip("MiMo decode chain: set CHOPINFER_MIMO_CHECKPOINT").unwrap();
        return;
    };
    let Some(ordinals) = kernel_lab::selected_ordinals().expect("GPU gate") else {
        kernel_lab::skip("MiMo decode chain: set CHOPINFER_KERNEL_GPUS to idle GPUs").unwrap();
        return;
    };
    assert_eq!(
        ordinals.len(),
        4,
        "MiMo decode chain requires four GPU ordinals"
    );
    let images = execution::images(Path::new(&checkpoint)).expect("TP4 layer images");
    let selected_layer = std::env::var("CHOPINFER_MIMO_LAYER").ok().map(|value| {
        value
            .parse::<usize>()
            .expect("CHOPINFER_MIMO_LAYER is an integer")
    });
    let selected_batch = std::env::var("CHOPINFER_MIMO_BATCH").ok().map(|value| {
        value
            .parse::<usize>()
            .expect("CHOPINFER_MIMO_BATCH is an integer")
    });
    assert!(selected_layer.is_none_or(|layer| [1, 5].contains(&layer)));
    assert!(selected_batch.is_none_or(|batch| [1, 8, 16, 32].contains(&batch)));
    kernel_lab::gpu_test_multi(4 * 1024 * 1024 * 1024, |sessions| {
        for layer in [1, 5]
            .into_iter()
            .filter(|layer| selected_layer.is_none_or(|selected| selected == *layer))
        {
            for batch in [1, 8, 16, 32]
                .into_iter()
                .filter(|batch| selected_batch.is_none_or(|selected| selected == *batch))
            {
                run_case(sessions, &images, layer, batch)?;
            }
        }
        Ok(())
    })
    .expect("MiMo decode chain");
}

fn graph_capture(sessions: &[Session], ranks: &[execution::Rank]) -> LabResult<Vec<GraphExec>> {
    let count = execution::stage_count(ranks, false)?;
    sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for stage in 0..count {
                    execution::launch_one(sessions, ranks, false, stage, rank, true)?;
                }
                Ok(())
            })
        })
        .collect()
}

fn replay(sessions: &[Session], graphs: &[GraphExec]) -> LabResult<()> {
    for (session, graph) in sessions.iter().zip(graphs) {
        graph.launch(&session.stream).map_err(|e| e.to_string())?;
    }
    execution::synchronize(sessions)
}

/// Every rank's FINAL rows and route active-expert list after a decode pass
/// on poisoned buffers, each checked against its write map first.
fn decode_outputs(
    sessions: &[Session],
    ranks: &[execution::Rank],
    phase: &str,
) -> LabResult<Vec<(Vec<u16>, Vec<i32>)>> {
    sessions
        .iter()
        .zip(ranks)
        .enumerate()
        .map(|(rank, (session, fixture))| {
            let label = format!(
                "layer {} batch {} rank {rank} {phase}",
                fixture.layer, fixture.batch
            );
            Ok((
                execution::decode_final(session, fixture, &label)?,
                execution::decode_active(session, fixture, &label)?,
            ))
        })
        .collect()
}

/// A pass's FINAL rows and active-expert lists, written as their maps say,
/// equal eager's bit for bit on every rank.
fn exact_outputs(
    sessions: &[Session],
    ranks: &[execution::Rank],
    expected: &[(Vec<u16>, Vec<i32>)],
    phase: &str,
) -> LabResult<()> {
    let outputs = decode_outputs(sessions, ranks, phase)?;
    for (rank, ((fixture, (actual, active)), (reference, eager_active))) in
        ranks.iter().zip(&outputs).zip(expected).enumerate()
    {
        if let Some(index) = actual.iter().zip(reference).position(|(a, b)| a != b) {
            return Err(format!(
                "layer {} batch {} rank {rank} {phase} differs at {index}: {} != {}",
                fixture.layer, fixture.batch, actual[index], reference[index]
            ));
        }
        if let Some(index) = active.iter().zip(eager_active).position(|(a, b)| a != b) {
            return Err(format!(
                "layer {} batch {} rank {rank} {phase} active expert word {index}: {} != {}",
                fixture.layer, fixture.batch, active[index], eager_active[index]
            ));
        }
    }
    Ok(())
}

fn time_eager(sessions: &[Session], ranks: &[execution::Rank], pdl: bool) -> LabResult<f32> {
    let mut samples = Vec::new();
    for iteration in 0..4 {
        let starts = sessions
            .iter()
            .map(|session| session.context.timed_event().map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let ends = sessions
            .iter()
            .map(|session| session.context.timed_event().map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        for (session, start) in sessions.iter().zip(&starts) {
            start.record(&session.stream).map_err(|e| e.to_string())?;
        }
        execution::launch_all(sessions, ranks, false, pdl)?;
        for (session, end) in sessions.iter().zip(&ends) {
            end.record(&session.stream).map_err(|e| e.to_string())?;
        }
        execution::synchronize(sessions)?;
        if iteration > 0 {
            samples.push(
                starts
                    .iter()
                    .zip(&ends)
                    .map(|(start, end)| start.elapsed_ms(end).map_err(|e| e.to_string()))
                    .collect::<LabResult<Vec<_>>>()?
                    .into_iter()
                    .fold(0.0_f32, f32::max),
            );
        }
    }
    samples.sort_by(f32::total_cmp);
    Ok(samples[1])
}

fn kernel_breakdown(
    sessions: &[Session],
    ranks: &[execution::Rank],
) -> LabResult<Vec<serde_json::Value>> {
    let count = execution::stage_count(ranks, false)?;
    let mut events = Vec::with_capacity(count);
    for stage in 0..count {
        let mut by_rank = Vec::with_capacity(ranks.len());
        for (rank, session) in sessions.iter().enumerate() {
            let start = session.context.timed_event().map_err(|e| e.to_string())?;
            let end = session.context.timed_event().map_err(|e| e.to_string())?;
            start.record(&session.stream).map_err(|e| e.to_string())?;
            execution::launch_one(sessions, ranks, false, stage, rank, false)?;
            end.record(&session.stream).map_err(|e| e.to_string())?;
            by_rank.push((start, end));
        }
        events.push(by_rank);
    }
    execution::synchronize(sessions)?;
    Ok(events
        .iter()
        .enumerate()
        .map(|(stage, by_rank)| {
            let elapsed_ms = by_rank
                .iter()
                .map(|(start, end)| start.elapsed_ms(end).expect("timed kernel event"))
                .fold(0.0_f32, f32::max);
            json!({"kernel": ranks[0].decode[stage].name, "critical_path_ms": elapsed_ms})
        })
        .collect())
}

fn run_case(
    sessions: &[Session],
    images: &execution::Images,
    layer: usize,
    batch: usize,
) -> LabResult<()> {
    let ranks = execution::ranks(sessions, images, layer, batch, false)?;
    let input = execution::seeded_hidden(batch, 0xdc65_0000 ^ ((layer as u64) << 8) ^ batch as u64);
    for (session, rank) in sessions.iter().zip(&ranks) {
        rank.set_inputs(session, &input, 64, false)?;
    }
    // Every compared pass runs on buffers poisoned first; the timing-only
    // launches in between compare nothing and stay unpoisoned.
    let mut evidence = Evidence::new(execution::decode_poisoned());
    execution::poison_decode(sessions, &ranks)?;
    execution::launch_all(sessions, &ranks, false, false)?;
    execution::synchronize(sessions)?;
    let eager = decode_outputs(sessions, &ranks, "eager")?;
    execution::poison_decode(sessions, &ranks)?;
    execution::launch_all(sessions, &ranks, false, true)?;
    execution::synchronize(sessions)?;
    exact_outputs(sessions, &ranks, &eager, "PDL")?;
    evidence.passed(Phase::Pdl, true);
    let graphs = graph_capture(sessions, &ranks)?;
    execution::poison_decode(sessions, &ranks)?;
    replay(sessions, &graphs)?;
    exact_outputs(sessions, &ranks, &eager, "graph")?;
    evidence.passed(Phase::Graph, false);
    let breakdown = kernel_breakdown(sessions, &ranks)?;
    let eager_ms = time_eager(sessions, &ranks, false)?;
    let pdl_ms = time_eager(sessions, &ranks, true)?;
    let graph_timing = kernel_lab::time_multi_graph(sessions, &graphs, 1, 1, 3)?;
    execution::poison_decode(sessions, &ranks)?;
    replay(sessions, &graphs)?;
    exact_outputs(sessions, &ranks, &eager, "timed graph")?;
    evidence.passed(Phase::Graph, false);
    let mut receipt = BTreeMap::new();
    receipt.insert("layer".into(), json!(layer));
    receipt.insert("batch".into(), json!(batch));
    receipt.insert(
        "ordinals".into(),
        json!(sessions.iter().map(|s| s.ordinal).collect::<Vec<_>>()),
    );
    receipt.insert("eager_ms".into(), json!(eager_ms));
    receipt.insert("pdl_chained_ms".into(), json!(pdl_ms));
    receipt.insert(
        "graph_replay_ms".into(),
        json!(graph_timing.critical_path.median_ms),
    );
    // Only phases that ran on poisoned buffers and matched eager's bits.
    evidence.record(&mut receipt);
    receipt.insert(
        "graph_replay_bitwise_eager".into(),
        json!(evidence.has(Phase::Graph)),
    );
    receipt.insert("pdl_bitwise_eager".into(), json!(evidence.has(Phase::Pdl)));
    receipt.insert("per_kernel".into(), json!(breakdown));
    receipt.insert("timing_environment".into(), json!("shared-device"));
    sessions[0].receipt("decode_chain", receipt)?;
    Ok(())
}
