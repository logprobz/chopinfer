//! Assigned-device checks for predecessor timing and aggregate telemetry.
use kernel_lab::eviction::{Eviction, EvictionMode};
use kernel_lab::{
    LabResult, gpu_test_multi, launch, time_after, time_multi_graph, time_named, with_timing_label,
};
use runtime_cuda::{KernelArgument, LaunchConfig};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn target_arms_and_peer_telemetry() {
    gpu_test_multi(2 << 30, |sessions| -> LabResult<()> {
        if std::env::var("CHOPINFER_KERNEL_EVICTION_AB").as_deref() != Ok("1")
            || EvictionMode::selected()? != EvictionMode::Write
        {
            return Err(
                "this regression requires paired timing with the write arm selected: set \
                 CHOPINFER_KERNEL_EVICTION_AB=1 and CHOPINFER_KERNEL_EVICTION=write; a \
                 multi-GPU run fails without them"
                    .into(),
            );
        }
        let session = &sessions[0];
        let prefix = Eviction::new(session, EvictionMode::Write)?;
        let outputs = sessions
            .iter()
            .map(|s| s.allocate(4))
            .collect::<LabResult<Vec<_>>>()?;
        let modules = sessions
            .iter()
            .map(|s| s.module("probe_sm120"))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .map(|m| m.function("chop_probe_sm120_v1").map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let config =
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?;
        let prefix_ms = time_named(session, "prefix_calibration", 1, 3, |stream| {
            prefix.run(stream)
        })?
        .median_ms;
        let target_us = with_timing_label("predecessor_target", || {
            time_after(
                session,
                || prefix.run(&session.stream),
                || {
                    functions[0]
                        .launch(
                            &session.stream,
                            config,
                            &[KernelArgument::Buffer(&outputs[0])],
                        )
                        .map_err(|e| e.to_string())
                },
            )
        })?;
        if target_us >= prefix_ms * 500.0 {
            return Err("target-only interval includes too much of the bulk-write prefix".into());
        }
        let graphs = sessions
            .iter()
            .enumerate()
            .map(|(rank, s)| {
                s.capture(|stream| {
                    functions[rank]
                        .launch(stream, config, &[KernelArgument::Buffer(&outputs[rank])])
                        .map_err(|e| e.to_string())
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let graph_timing = with_timing_label("aggregate_probe", || {
            time_multi_graph(sessions, &graphs, 1, 1, 3)
        })?;
        if graph_timing.gpu_rounds.len() != 3
            || graph_timing.gpu_rounds.iter().any(|round| {
                round["ranks"]
                    .as_array()
                    .is_none_or(|ranks| ranks.len() != sessions.len())
            })
        {
            return Err("graph timing lost participating rank snapshots".into());
        }
        for (s, output) in sessions.iter().zip(&outputs) {
            let mut bytes = [0; 4];
            output
                .copy_to_bytes(&mut bytes, &s.stream)
                .map_err(|e| e.to_string())?;
            if u32::from_ne_bytes(bytes) != 120 {
                return Err("probe target did not execute".into());
            }
        }
        let receipt = session.receipt(
            "timing_regression",
            BTreeMap::from([
                ("target_us".into(), json!(target_us)),
                ("prefix_ms".into(), json!(prefix_ms)),
            ]),
        )?;
        let data: serde_json::Value =
            serde_json::from_slice(&std::fs::read(receipt).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let records = data["timing_rounds"]
            .as_array()
            .ok_or("missing timing records")?;
        let target = records
            .iter()
            .find(|record| {
                record["kind"] == "paired_summary" && record["operation"] == "predecessor_target"
            })
            .ok_or("missing target-only paired summary")?;
        let selected_ms = target["arm_median_ms"][0]
            .as_f64()
            .ok_or("missing write-arm median")?;
        if (selected_ms * 1000.0 - f64::from(target_us)).abs() > 1e-3 {
            return Err("returned predecessor timing belongs to another eviction arm".into());
        }
        let aggregate = records
            .iter()
            .filter(|record| record["kind"] == "multi_graph")
            .collect::<Vec<_>>();
        if aggregate.len() != 3
            || aggregate.iter().any(|record| {
                record["ranks"]
                    .as_array()
                    .is_none_or(|ranks| ranks.len() != sessions.len())
            })
        {
            return Err("rank-zero receipt lost graph peer telemetry".into());
        }
        Ok(())
    })
    .expect("target-only timing and peer telemetry");
}
