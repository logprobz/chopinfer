//! Real-weight MiMo prefill graph replay and in-graph profiling.

#[path = "mimo_layers.rs"]
mod mimo_layers;

use kernel_lab::{LabResult, Session};
use mimo_layers::execution;
use runtime_cuda::GraphExec;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

#[path = "common/prefill_local_context.rs"]
mod local_context;

#[path = "common/prefill_later_context.rs"]
mod later_context;

#[path = "common/prefill_global_attention.rs"]
mod global_attention;

#[path = "common/prefill_poison.rs"]
mod prefill_poison;
use kernel_lab::poison::Phase;

fn capture(sessions: &[Session], ranks: &[execution::Rank]) -> LabResult<Vec<GraphExec>> {
    let count = execution::stage_count(ranks, true)?;
    sessions
        .iter()
        .enumerate()
        .map(|(rank, session)| {
            session.capture(|_| {
                for stage in 0..count {
                    execution::launch_one(sessions, ranks, true, stage, rank, true)?;
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

fn check_outputs(
    sessions: &[Session],
    ranks: &[execution::Rank],
    expected: &[Vec<u16>],
    rows: usize,
) -> LabResult<()> {
    for (rank_index, ((session, rank), reference)) in
        sessions.iter().zip(ranks).zip(expected).enumerate()
    {
        let actual = execution::output(session, rank, rows)?;
        if let Some(index) = actual.iter().zip(reference).position(|(a, b)| a != b) {
            return Err(format!(
                "prefill layer {} tokens {rows} rank {rank_index} graph differs at {index}: {} != {}",
                rank.layer, actual[index], reference[index]
            ));
        }
    }
    Ok(())
}

#[test]
fn prefill_chain_graph_replay() {
    let Ok(checkpoint) = std::env::var("CHOPINFER_MIMO_CHECKPOINT") else {
        kernel_lab::skip("prefill chain: set CHOPINFER_MIMO_CHECKPOINT").unwrap();
        return;
    };
    let Some(ordinals) = kernel_lab::selected_ordinals().expect("GPU gate") else {
        kernel_lab::skip("prefill chain: set CHOPINFER_KERNEL_GPUS").unwrap();
        return;
    };
    assert_eq!(ordinals.len(), 4, "prefill chain needs four GPU ordinals");
    let layer = std::env::var("CHOPINFER_MIMO_PREFILL_LAYER")
        .unwrap_or_else(|_| "1".into())
        .parse::<usize>()
        .expect("prefill layer integer");
    let rows = std::env::var("CHOPINFER_MIMO_PREFILL_TOKENS")
        .unwrap_or_else(|_| "2048".into())
        .parse::<usize>()
        .expect("prefill token integer");
    assert!([1, 5].contains(&layer) && [2048, 8192].contains(&rows));
    let images = execution::images(Path::new(&checkpoint)).expect("TP4 layer images");
    kernel_lab::gpu_test_multi(4 * 1024 * 1024 * 1024, |sessions| {
        let ranks = execution::ranks_with_prefill_rows(sessions, &images, layer, rows)?;
        let input = execution::seeded_hidden(rows, 0x6f18_0000 ^ ((layer as u64) << 8));
        for (session, rank) in sessions.iter().zip(&ranks) {
            rank.set_inputs(session, &input, 0, true)?;
        }
        let mut units = sessions
            .iter()
            .zip(&ranks)
            .map(|(s, rank)| {
                prefill_poison::unit(s, rank, rows, 0..rank.prefill.len(), false, None)
            })
            .collect::<LabResult<Vec<_>>>()?;
        prefill_poison::collective(sessions, &mut units, Phase::Eager, false, || {
            execution::launch_all(sessions, &ranks, true, false)
        })?;
        execution::synchronize(sessions)?;
        let eager = sessions
            .iter()
            .zip(&ranks)
            .map(|(session, rank)| execution::output(session, rank, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let graphs = capture(sessions, &ranks)?;
        prefill_poison::collective(sessions, &mut units, Phase::Graph, true, || {
            replay(sessions, &graphs)
        })?;
        check_outputs(sessions, &ranks, &eager, rows)?;
        let timing = kernel_lab::time_multi_graph(sessions, &graphs, 1, 1, 3)?;
        prefill_poison::collective(sessions, &mut units, Phase::Graph, true, || {
            replay(sessions, &graphs)
        })?;
        check_outputs(sessions, &ranks, &eager, rows)?;
        let mut receipt = BTreeMap::new();
        receipt.insert(
            "poisoned_ranks".into(),
            json!(units.iter().map(|unit| unit.evidence()).collect::<Vec<_>>()),
        );
        receipt.insert("layer".into(), json!(layer));
        receipt.insert("tokens".into(), json!(rows));
        receipt.insert(
            "graph_replay_ms".into(),
            json!(timing.critical_path.median_ms),
        );
        receipt.insert("graph_replay_bitwise_eager".into(), json!(true));
        receipt.insert("timing_environment".into(), json!("shared-device"));
        receipt.insert(
            "stages".into(),
            json!(
                ranks[0]
                    .prefill
                    .iter()
                    .map(|stage| &stage.name)
                    .collect::<Vec<_>>()
            ),
        );
        sessions[0].receipt("prefill_chain", receipt)
    })
    .expect("MiMo prefill chain");
}
