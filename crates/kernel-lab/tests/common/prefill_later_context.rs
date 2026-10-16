//! Later-layer controls with real captured residuals and routes.

use super::*;
use kernel_core::codec::bf16_to_f32;
use serde_json::Value;
use sha2::{Digest, Sha256};

fn read_fixture(root: &Path, case: &Value, role: &str, bytes: usize) -> LabResult<Vec<u8>> {
    let entry = &case["files"][role];
    let file = entry["file"].as_str().ok_or("missing replay filename")?;
    if Path::new(file).components().count() != 1 {
        return Err("replay file must be a basename".into());
    }
    let data = std::fs::read(root.join(file)).map_err(|e| e.to_string())?;
    if data.len() != bytes || entry["sha256"] != format!("{:x}", Sha256::digest(&data)) {
        return Err(format!("replay shape/hash mismatch for {role}"));
    }
    Ok(data)
}

fn bf16(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

fn permute<T: Copy>(input: &[T], order: &[usize], width: usize) -> Vec<T> {
    order
        .iter()
        .flat_map(|&row| input[row * width..(row + 1) * width].iter().copied())
        .collect()
}

fn output_bits(session: &Session, rank: &execution::Rank, rows: usize) -> LabResult<Vec<Vec<u16>>> {
    ["gate_up", "down", "moe_combine"]
        .into_iter()
        .map(|name| execution::intermediate(session, rank, name, rows))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_sequence(
    session: &Session,
    rank: &execution::Rank,
    end: usize,
    rows: usize,
    ctas: u32,
    pdl: bool,
    replay: Option<&execution::PrefillReplayBindings>,
) -> LabResult<()> {
    for stage in 0..end {
        if !rank.prefill[stage].name.starts_with("chop_comm_") {
            execution::launch_local_prefill_replay(session, rank, stage, rows, ctas, pdl, replay)?;
        }
    }
    Ok(())
}

#[test]
fn prefill_later_layer_context() {
    let Ok(root) = std::env::var("CHOPINFER_MIMO_LATER_REPLAY") else {
        kernel_lab::skip(
            "later-layer prefill context: set CHOPINFER_MIMO_LATER_REPLAY to captured layer data",
        )
        .unwrap();
        return;
    };
    let layer = std::env::var("CHOPINFER_MIMO_PREFILL_LAYER")
        .expect("layer")
        .parse::<usize>()
        .expect("integer layer");
    let rows = std::env::var("CHOPINFER_MIMO_PREFILL_TOKENS")
        .expect("tokens")
        .parse::<usize>()
        .expect("integer tokens");
    let mode = std::env::var("CHOPINFER_MIMO_LATER_MODE").expect("replay mode");
    assert!([1, 5, 21, 30, 47].contains(&layer) && [1024, 2048].contains(&rows));
    assert!(["captured", "reordered", "relocated", "warm", "seeded"].contains(&mode.as_str()));
    assert_eq!(
        kernel_lab::selected_ordinals()
            .expect("GPU selection")
            .expect("assigned GPU")
            .len(),
        1
    );
    let root = Path::new(&root).join(format!("layer{layer:02}"));
    let manifest_bytes = std::fs::read(root.join("manifest.json")).expect("replay manifest");
    let manifest: Value = serde_json::from_slice(&manifest_bytes).expect("manifest JSON");
    assert_eq!(manifest["layer"], layer);
    let case = manifest["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|case| case["tokens"] == rows)
        .expect("requested token case");
    let captured_input = bf16(&read_fixture(&root, case, "input", rows * 4096 * 2).expect("input"));
    let residual = bf16(&read_fixture(&root, case, "residual", rows * 4096 * 2).expect("residual"));
    let ids: Vec<i32> = read_fixture(&root, case, "ids", rows * 8 * 4)
        .expect("IDs")
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    assert!(
        ids.chunks_exact(8)
            .all(|row| row.iter().all(|id| (0..256).contains(id))
                && row.iter().collect::<std::collections::BTreeSet<_>>().len() == 8)
    );
    // Every layer, so view offsets are those of the served image.
    let checkpoint = std::env::var("CHOPINFER_MIMO_CHECKPOINT").expect("MiMo checkpoint");
    let images = execution::layer_images(Path::new(&checkpoint), None).expect("TP4 model image");
    let gate_offset = images.indexes[1]
        .view(&format!(
            "model.layers.{layer}.mlp.experts.gate_up_proj.weight"
        ))
        .expect("gate weight view")
        .offset;
    kernel_lab::gpu_test(8 << 30, |session| {
        let rank = execution::Rank::build(session, &images, 1, layer, 1, true, 8192, false)?;
        let end = rank.prefill.iter().position(|s| s.name.starts_with("chop_row_combine_mimo4096_"))
            .ok_or("missing MoE combine")? + 1;
        let ctas = 172;
        let seeded = mode == "seeded";
        let input = if seeded { execution::seeded_hidden(rows, 0x6f18_0000 ^ ((layer as u64) << 8)) }
                    else { captured_input.clone() };
        rank.set_inputs(session, &input, 0, true)?;
        let canonical = execution::PrefillReplayBindings::new(session, &rank, &residual, &ids, false)?;
        let mut canonical_unit = prefill_poison::unit(session, &rank, rows, 0..end, true, (!seeded).then_some(ids.as_slice()))?;
        canonical_unit.phase(Phase::Eager, false, || run_sequence(session, &rank, end, rows, ctas, false, (!seeded).then_some(&canonical)))?;
        let reference = output_bits(session, &rank, rows)?;
        let original_addresses = canonical.weight_addresses(&rank)?;
        let mut order: Vec<_> = (0..rows).collect();
        if mode == "reordered" { order.sort_by_key(|&row| ids[row * 8]); }
        rank.set_inputs(session, &permute(&input, &order, 4096), 0, true)?;
        let replay = execution::PrefillReplayBindings::new(session, &rank,
            &permute(&residual, &order, 4096), &permute(&ids, &order, 8), mode == "relocated")?;
        let bindings = (!seeded).then_some(&replay);
        let effective_ids = permute(&ids, &order, 8);
        let mut unit = prefill_poison::unit(session, &rank, rows, 0..end, true, (!seeded).then_some(effective_ids.as_slice()))?;
        unit.phase(Phase::Eager, false, || run_sequence(session, &rank, end, rows, ctas, false, bindings))?;
        let graph = session.capture(|_| run_sequence(session, &rank, end, rows, ctas, true, bindings))?;
        unit.phase(Phase::Graph, true, || graph.launch(&session.stream).map_err(|e| e.to_string()))?;
        unit.phase(Phase::Pdl, true, || run_sequence(session, &rank, end, rows, ctas, true, bindings))?;
        let check = |record_hashes: bool| -> LabResult<Vec<String>> {
            let actual = output_bits(session, &rank, rows)?;
            for ((expected, actual), width) in reference.iter().zip(&actual).zip([8 * 512, 8 * 4096, 4096]) {
                if *actual != permute(expected, &order, width) {
                    return Err(format!("layer {layer} {rows} {mode}: replay differs bitwise"));
                }
            }
            if actual[0].iter().all(|v| v & 0x7fff == 0) || actual.iter().flatten().any(|v| v & 0x7f80 == 0x7f80) {
                return Err("nonfinite or empty real-layer outputs".into());
            }
            if !record_hashes { return Ok(Vec::new()); }
            Ok(actual.iter().map(|values| {
                let mut digest = Sha256::new();
                for value in values { digest.update(value.to_le_bytes()); }
                format!("{:x}", digest.finalize())
            }).collect())
        };
        check(false)?;
        let natural_ids = execution::router_choices(session, &rank, rows)?;
        let used_ids: Vec<usize> = if seeded { natural_ids.iter().flatten().copied().collect() }
            else { permute(&ids, &order, 8).iter().map(|&x| x as usize).collect() };
        let natural_mismatches = natural_ids.iter().flatten().zip(&used_ids).filter(|(a,b)| a != b).count();
        let mut counts = vec![0usize; 256];
        for &expert in &used_ids { counts[expert] += 1; }
        let normalized = execution::intermediate(session, &rank, "post_norm", rows)?;
        let sumsq: f64 = normalized.iter().map(|&x| f64::from(bf16_to_f32(x)).powi(2)).sum();
        let absmax = normalized.iter().map(|&x| bf16_to_f32(x).abs()).fold(0.0_f32, f32::max);
        // Events surround the entire layer, never prefix/gate or gate/down.
        // The warm control runs the same full layer before the target graph.
        let us = if mode == "warm" {
            kernel_lab::time_after(session,
                || run_sequence(session, &rank, end, rows, ctas, true, bindings),
                || run_sequence(session, &rank, end, rows, ctas, true, bindings))?
        } else {
            kernel_lab::time(session, 2, 5, |stream| graph.launch(stream).map_err(|e| e.to_string()))?.median_ms * 1000.0
        };
        unit.phase(Phase::Graph, true, || graph.launch(&session.stream).map_err(|e| e.to_string()))?;
        let output_hashes = check(true)?;
        let weight_addresses = replay.weight_addresses(&rank)?;
        if mode == "relocated" && weight_addresses == original_addresses {
            return Err("weight relocation did not change addresses".into());
        }
        let stages: Vec<_> = rank.prefill[..end].iter()
            .filter(|stage| !stage.name.starts_with("chop_comm_"))
            .map(|stage| json!({"symbol": stage.name, "planned_grid": stage.plan.grid})).collect();
        session.receipt("moe_prefill_later_context", BTreeMap::from([
            ("poisoned_unit".into(), unit.evidence()),
            ("canonical_unit".into(), canonical_unit.evidence()),
            ("rank".into(), json!(1)),
            ("attention_global".into(), json!(execution::GLOBAL_LAYERS.contains(&layer))),
            ("stages".into(), json!(stages)),
            ("layer".into(), json!(layer)), ("tokens".into(), json!(rows)),
            ("mode".into(), json!(mode)), ("persistent_ctas".into(), json!(ctas)),
            ("capacity".into(), json!(8192)), ("dense_gemm_live_m".into(), json!(rows)),
            ("layer_graph_median_us".into(), json!(us)),
            ("bitwise_graph_and_control".into(), json!(true)),
            ("gate_down_combine_sha256".into(), json!(output_hashes)),
            ("expert_rows".into(), json!(counts)),
            ("natural_router_id_mismatches".into(), json!(natural_mismatches)),
            ("post_norm_rms".into(), json!((sumsq / normalized.len() as f64).sqrt())),
            ("post_norm_abs_max".into(), json!(absmax)),
            ("weight_addresses".into(), json!(weight_addresses)),
            ("original_weight_addresses".into(), json!(original_addresses)),
            ("rank_direct_identity".into(), json!(images.indexes[1].direct_identity)),
            ("gate_planned_rank_offset".into(), json!(gate_offset)),
            ("planned_rank_index_sha256".into(), json!(images.index_sha256(1))),
            ("rank_weights_read".into(), images.read_record(1)),
            ("gate_cubin_sha256".into(), json!(format!("{:x}", Sha256::digest(kernel_lab::cubin("chop_moe_mimo_gateup_prefill_mxfp4_a16_t8192")?)))),
            ("manifest_sha256".into(), json!(format!("{:x}", Sha256::digest(&manifest_bytes)))),
            ("captured_unique_rows".into(), manifest["unique_rows"].clone()),
            ("repeated_rows".into(), json!(!seeded)),
            ("serving_reproduction".into(), json!(false)),
            ("limitations".into(), json!("Single rank with separate weight allocations, no peer traffic or half overlap. Captured mode repeats 144 real rows and substitutes captured TP4 residuals before norm; route consumers use captured IDs with fixed 1/8 down weights. Prefix attention runs on the repeated hidden sequence. Seeded mode uses local partials and natural routes.")),
        ]))
    }).expect("later-layer prefill control");
}
