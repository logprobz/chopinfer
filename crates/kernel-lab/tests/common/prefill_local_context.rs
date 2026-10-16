//! Single-rank real-weight compute context, with no collective launches.

use super::*;

fn local_range(
    session: &Session,
    rank: &execution::Rank,
    range: std::ops::Range<usize>,
    live_rows: usize,
    ctas: u32,
    pdl: bool,
) -> LabResult<()> {
    for index in range {
        if !rank.prefill[index].name.starts_with("chop_comm_") {
            execution::launch_local_prefill(session, rank, index, live_rows, ctas, pdl)?;
        }
    }
    Ok(())
}

fn outputs(session: &Session, rank: &execution::Rank, rows: usize) -> LabResult<Vec<Vec<u16>>> {
    ["gate_up", "down", "moe_combine"]
        .into_iter()
        .map(|name| execution::intermediate(session, rank, name, rows))
        .collect()
}

/// The checkpoint the single-rank real-weight prefill tests plan from.
pub(super) fn checkpoint() -> String {
    std::env::var("CHOPINFER_MIMO_CHECKPOINT").expect("MiMo checkpoint")
}

#[test]
fn prefill_single_rank_context() {
    if std::env::var("CHOPINFER_MIMO_PREFILL_LOCAL_CONTEXT").as_deref() != Ok("1") {
        kernel_lab::skip(
            "single-rank prefill context: set CHOPINFER_MIMO_PREFILL_LOCAL_CONTEXT=1 and CHOPINFER_MIMO_CHECKPOINT",
        )
        .unwrap();
        return;
    }
    let checkpoint = checkpoint();
    let ordinals = kernel_lab::selected_ordinals()
        .expect("GPU selection")
        .expect("one assigned GPU");
    assert_eq!(ordinals.len(), 1, "local context must use one GPU");
    let images = execution::images(Path::new(&checkpoint)).expect("TP4 layer images");
    // Rank 1's planned identity and the digest of the index its plan produced.
    let direct_identity = images.indexes[1].direct_identity.clone();
    let index_sha256 = images.index_sha256(1);
    kernel_lab::gpu_test(8 << 30, |session| {
        let mut failures = prefill_poison::Failures::default();
        for layer in [1, 5] {
            // Capacity matches the serving trace's routing grids. Live rows
            // remain one half-chunk; all buffers retain their declared size.
            let rank = execution::Rank::build(session, &images, 1, layer, 1, true, 8192, false)?;
            let find = |prefix: &str| rank.prefill.iter().position(|s| s.name.starts_with(prefix))
                .ok_or_else(|| format!("missing local prefill stage {prefix}"));
            let gate = find("chop_moe_mimo_gateup_prefill_")?;
            let down = find("chop_moe_mimo_down_prefill_")?;
            let end = find("chop_row_combine_mimo4096_")? + 1;
            find("chop_row_permute_tile_count_mimo_")?;
            for rows in [1024, 2048] {
                let result = (|| -> LabResult<()> {

                    let seed = 0x6f18_0000 ^ ((layer as u64) << 8);
                    rank.set_inputs(session, &execution::seeded_hidden(rows, seed), 0, true)?;
                    let mut baseline = prefill_poison::unit(session, &rank, rows, 0..end, true, None)?;
                    baseline.phase(Phase::Eager, false, || local_range(session, &rank, 0..end, rows, 188, false))?;
                    let reference_188 = outputs(session, &rank, rows)?;
                    let routes_188 = execution::router_choices(session, &rank, rows)?;
                    let fixed_inputs = [33,35,36,37,38,39,40].into_iter().map(|id| {
                        Ok((id, kernel_lab::poison::read(session, &rank.buffers[&id])?))
                    }).collect::<LabResult<Vec<_>>>()?;
                    for ctas in [188, 172] {
                        let result = (|| -> LabResult<()> {

                            // The preceding 188-CTA replay leaves fixed MoE inputs.
                            // Check only MoE's grid here: upstream stream-K GEMMs
                            // can change their reduction partition with the grid.
                            for (id, initial) in &fixed_inputs {
                                kernel_lab::poison::write(session, &rank.buffers[id], 0, initial)?;
                            }
                            let mut moe = prefill_poison::unit(session, &rank, rows, gate..end, true, None)?;
                            moe.phase(Phase::Eager, false, || local_range(session, &rank, gate..end, rows, ctas, false))?;
                            if outputs(session, &rank, rows)? != reference_188 {
                                return Err(format!("layer {layer} rows {rows} grid {ctas}: fixed-input MoE differs bitwise"));
                            }
                            let mut unit = prefill_poison::unit(session, &rank, rows, 0..end, true, None)?;
                            unit.phase(Phase::Eager, false, || local_range(session, &rank, 0..end, rows, ctas, false))?;
                            let reference = outputs(session, &rank, rows)?;
                            if reference.iter().flatten().any(|v| v & 0x7f80 == 0x7f80)
                                || reference[0].iter().all(|v| v & 0x7fff == 0) {
                                return Err("local real-weight prefill produced nonfinite or empty gate output".into());
                            }
                            let routes = execution::router_choices(session, &rank, rows)?;
                            let changed_ids = routes.iter().flatten().zip(routes_188.iter().flatten())
                                .filter(|(a, b)| a != b).count();
                            let changed_gate = reference[0].iter().zip(&reference_188[0])
                                .filter(|(a, b)| a != b).count();
                            let mut counts = vec![0usize; 256];
                            for &expert in routes.iter().flatten() {
                                let count = counts.get_mut(expert).ok_or("router expert outside 0..256")?;
                                *count += 1;
                            }
                            let graph = session.capture(|_| local_range(session, &rank, 0..end, rows, ctas, true))?;
                            unit.phase(Phase::Graph, true, || graph.launch(&session.stream).map_err(|e| e.to_string()))?;
                            unit.phase(Phase::Pdl, true, || local_range(session, &rank, 0..end, rows, ctas, true))?;
                            if outputs(session, &rank, rows)? != reference
                                || execution::router_choices(session, &rank, rows)? != routes {
                                return Err(format!("layer {layer} rows {rows} grid {ctas}: local PDL output differs bitwise"));
                            }
                            let mut timings = BTreeMap::new();
                            for (name, prefix, target) in [
                                ("gate_up_cold", 0..0, gate),
                                ("gate_up_after_local_prefix", 0..gate, gate),
                                ("down_cold", 0..0, down),
                                ("down_after_local_prefix_and_gate", 0..down, down),
                            ] {
                                let us = kernel_lab::time_after(session,
                                    || local_range(session, &rank, prefix.clone(), rows, ctas, true),
                                    || execution::launch_local_prefill(session, &rank, target, rows, ctas, true))?;
                                timings.insert(name, us);
                            }
                            let pdl = kernel_lab::time(session, 2, 5, |stream| graph.launch(stream).map_err(|e| e.to_string()))?;
                            let serial = session.capture(|_| local_range(session, &rank, 0..end, rows, ctas, false))?;
                            let serial_time = kernel_lab::time(session, 2, 5, |stream| serial.launch(stream).map_err(|e| e.to_string()))?;
                            unit.phase(Phase::Graph, false, || serial.launch(&session.stream).map_err(|e| e.to_string()))?;
                            if outputs(session, &rank, rows)? != reference {
                                return Err("local serialized replay differs bitwise".into());
                            }
                            timings.insert("local_full_pdl", pdl.median_ms * 1000.0);
                            timings.insert("local_full_serialized", serial_time.median_ms * 1000.0);
                            let stages: Vec<_> = rank.prefill[..end].iter().filter(|s| !s.name.starts_with("chop_comm_"))
                                .map(|s| json!({"symbol": s.name, "planned_grid": s.plan.grid})).collect();
                            session.receipt("moe_prefill_local_context", BTreeMap::from([
                                ("poisoned_unit".into(), unit.evidence()),
                                ("fixed_input_unit".into(), moe.evidence()),
                                ("layer".into(), json!(layer)), ("rank".into(), json!(1)),
                                ("tokens".into(), json!(rows)), ("capacity".into(), json!(8192)),
                                ("dense_gemm_live_m".into(), json!(rows)),
                                ("persistent_ctas".into(), json!(ctas)), ("seed".into(), json!(seed)),
                                ("rank_direct_identity".into(), json!(direct_identity)),
                                ("planned_rank_index_sha256".into(), json!(index_sha256)),
                                ("rank_weights_read".into(), images.read_record(1)),
                                ("expert_rows".into(), json!(counts)), ("median_us".into(), json!(timings)),
                                ("stages".into(), json!(stages)), ("bitwise_graph_and_pdl_at_same_grid".into(), json!(true)),
                                ("bitwise_moe_grid_with_fixed_inputs".into(), json!(true)),
                                ("upstream_grid_changed_route_ids".into(), json!(changed_ids)),
                                ("upstream_grid_changed_gate_elements".into(), json!(changed_gate)),
                                ("serving_reproduction".into(), json!(false)),
                                ("limitations".into(), json!("Seeded hidden states and real rank-1 weights. Local partials replace all-reduced tensors before residual/norm. No peer traffic, concurrent communication or second-half overlap.")),
                            ]))?;

                            Ok(())
                                            })();
                        failures.check(&format!("layer={layer} rows={rows} ctas={ctas}"), result);
                    }

                    Ok(())
                                })();
                failures.check(&format!("layer={layer} rows={rows}"), result);
            }
        }
        failures.finish()
    }).expect("single-rank prefill context");
}
