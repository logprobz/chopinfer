//! Site schemas reject choices that a consumer would silently ignore.
use kernel_build::tuning::{Row, TuningError};

pub(super) fn validate(row: &Row) -> Result<(), TuningError> {
    let (schedules, counts, flags): (&[&str], &[&str], &[&str]) = match row.key.op.as_str() {
        "dense_fold.attention_kv_cluster.decode"
        | "dense_fold.attention_o_a_cluster.decode"
        | "dense_fold.attention_o_a_status.verify"
        | "dense_fold.attention_o_b_cluster.decode"
        | "dense_fold.attention_o_b_status.verify"
        | "dense_fold.attention_q_a_cluster.decode"
        | "dense_fold.attention_q_a_status.verify" => (
            &["cluster_fused"],
            &["splits", "stages", "tile_n"],
            &["distributed"],
        ),
        "dense_fold.direct.decode" => (&["direct"], &["k_warps", "n_warps"], &[]),
        "dense_fold.fold.decode"
        | "dense_fold.split.decode"
        | "dense_fold.wide.verify"
        | "reuse.fold.decode" => (
            &["dense_decode"],
            &["finish", "row_groups", "stages", "tile_n"],
            &["weight_prefetch"],
        ),
        "dense_fold.status.decode" => (
            &["status"],
            &["slices", "stages", "tile_n", "tile_rows", "warps"],
            &[],
        ),
        "dense_fold.status_blocks.decode" => (&["status"], &[], &[]),
        "dense_fold.status_tails.decode" => (&["status"], &["tile_rows"], &[]),
        "dense_fold.status_wide.verify" | "dense_fold.status_wide_tails.decode" => {
            (&["local_wide"], &["stages", "tile_n", "tile_rows"], &[])
        }
        "dense_fold.stream.decode" => (&["stream"], &[], &[]),
        "dense_prefill.compact.prefill"
        | "dense_prefill.matrix.prefill"
        | "dense_prefill.dspark_main.prefill" => (
            &["prefill_tiles"],
            &["stages", "tile_rows"],
            &["matrix_fetch"],
        ),
        "dense_prefill.tma_scales.prefill" => (&["tma_scales"], &[], &[]),
        "f32.band.prefill" | "f32.exact.prefill" => (&["tiled"], &["tile"], &[]),
        "f32.blocked.prefill" => (&["blocked"], &["tile"], &[]),
        "f32.bounded_reuse.verify" => (&["bounded_reuse"], &[], &[]),
        "f32.cached.verify"
        | "indexer.cached.prefill"
        | "norm.ceil_cached_bf16.decode"
        | "norm.ceil_cached_f32.decode" => (&["cached"], &[], &[]),
        "f32.head_last.decode" => (&["last"], &[], &[]),
        "f32.router_norm.prefill" => (&["certified"], &[], &[]),
        "f32.rows.verify" => (
            &["rows"],
            &[
                "features",
                "plane_k",
                "rows",
                "slices",
                "stages",
                "warp_features",
                "warp_rows",
            ],
            &[],
        ),
        "f32.router_zero.prefill" => (&["zero"], &[], &[]),
        "f32.wide.prefill" => (&["wide"], &["tile"], &[]),
        "hc.decode_mix.decode"
        | "hc.decode_mix.verify"
        | "hc.draft_mix.decode"
        | "hc.draft_mix.prefill"
        | "hc.draft_mix.verify"
        | "hc.finish_admit.decode"
        | "hc.finish_admit.verify"
        | "hc.wide_mix.verify"
        | "norm.input_admit.prefill"
        | "norm.query_admit.prefill"
        | "norm.small_admit.decode" => (&["admit"], &[], &[]),
        "hc.expand_mix.prefill" | "hc.lagged_mix.prefill" => {
            (&["pipeline"], &["chains", "tile"], &[])
        }
        "hc.finish_chain.verify" | "hc.round_chain.decode" => (&["chain"], &["warps"], &[]),
        "hc.finish_cols.verify" => (&["cols"], &["tile", "warps"], &["transposed"]),
        "hc.finish_pair.verify" => (&["pair"], &[], &[]),
        "hc.finish_rows.verify" => (&["rows"], &["tile"], &[]),
        "hc.finish_twin.verify" => (&["twin"], &["warps"], &[]),
        "hc.lagged_warp.decode" => (&["warp"], &[], &[]),
        "indexer.candidate_scores.prefill" | "indexer.compressed_scores.prefill" => {
            (&["tiled"], &[], &[])
        }
        "indexer.hierarchy.decode" => (&["hierarchical"], &[], &[]),
        "indexer.query.verify" | "latent.query.decode" | "latent.query.verify" => {
            (&["parallel"], &[], &[])
        }
        "latent.cache.decode" | "latent.cache.verify" => {
            (&["selected_cache"], &["head_group"], &[])
        }
        "latent.paired.prefill" => (&["paired"], &[], &[]),
        "latent.sources.decode" | "latent.sources.verify" => (&["independent_heads"], &[], &[]),
        "latent.staging.decode" | "latent.staging.verify" => {
            (&["staged"], &["head_group", "stage_keys"], &["packed"])
        }
        "moe.absorb.decode" | "moe.absorb.verify" => (&["absorb"], &[], &[]),
        "moe.active_gate_up.decode"
        | "moe.active_gate_up.prefill"
        | "moe.active_gate_up.verify" => (&["active"], &[], &[]),
        "moe.compact_down.decode" | "moe.compact_gate_up.decode" => (&["compact"], &[], &[]),
        "moe.direct_admit.decode" => (&["direct"], &["order"], &[]),
        "moe.direct_admit.verify" => (&["direct", "gap", "wide"], &["order"], &[]),
        "moe.direct_wide.verify" => (&["direct"], &[], &[]),
        "moe.direct_drafter.verify" => (&["direct"], &["order"], &[]),
        "moe.gate_finish.decode" | "moe.gate_finish.verify" => (&["gate_finish"], &[], &[]),
        "moe.gate_grid.decode" | "moe.gate_grid.verify" => (&["gate_grid"], &[], &[]),
        "moe.gate_resident.decode" | "moe.gate_resident.prefill" | "moe.gate_resident.verify" => {
            (&["gate_resident"], &[], &[])
        }
        "moe.grouped_down.prefill" => (&["grouped"], &[], &[]),
        "moe.pipeline_gate_up.prefill" => (&["pipeline"], &[], &[]),
        "moe.prefill_down.prefill" | "moe.prefill_gate_up.prefill" => (&["prefill"], &[], &[]),
        "moe.route.decode" | "moe.streamed.decode" | "moe.streamed.verify" => {
            (&["streamed"], &[], &[])
        }
        "moe.router_finish.decode" | "moe.router_finish.verify" => (&["router_finish"], &[], &[]),
        "moe.router_gap.decode" | "moe.router_gap.verify" => (&["router_gap"], &[], &[]),
        "moe.router_r2.decode" | "moe.router_r2.verify" => (&["router_r2"], &[], &[]),
        "moe.router_wide.verify" => (&["router_wide"], &[], &[]),
        "moe.small.decode" | "moe.small.prefill" | "moe.small.verify" => (&["small"], &[], &[]),
        "moe.strip2.decode" | "moe.strip2.prefill" | "moe.strip2.verify" => (&["strip2"], &[], &[]),
        "moe.wide.verify" => (&["wide"], &[], &[]),
        "norm.activation_capacity.decode"
        | "norm.ceil_capacity_bf16.decode"
        | "norm.ceil_capacity_f32.decode"
        | "norm.ceil_registered_bf16.decode"
        | "norm.ceil_registered_f32.decode" => (&["parallel"], &["capacity"], &[]),
        "norm.ceil_prefill_bf16.prefill" | "norm.ceil_prefill_f32.prefill" => {
            (&["prefill"], &["warps"], &[])
        }
        "norm.input_cached.decode"
        | "norm.input_small.decode"
        | "norm.query_cached.decode"
        | "norm.query_small.decode" => (&["cached"], &["warps"], &[]),
        "norm.input_strategy.prefill" => (&["cached", "recompute"], &[], &[]),
        "norm.query_strategy.prefill" => (&["cached", "reread"], &[], &[]),
        "reuse.head.decode" => (&["reuse"], &[], &[]),
        _ => return Err(TuningError(format!("{}: unsupported tuning site", row.key))),
    };
    if !schedules.contains(&row.schedule.as_str()) {
        return Err(TuningError(format!(
            "{}: unsupported schedule {} (expected {})",
            row.key,
            row.schedule,
            schedules.join(" or ")
        )));
    }
    let folds = row.schedule == "dense_decode"
        && row.knobs.get("row_groups").and_then(|k| k.count()) == Some(2);
    for &name in counts.iter().chain(folds.then_some("folds").as_slice()) {
        if row.knobs.get(name).and_then(|k| k.count()).is_none() {
            return Err(TuningError(format!(
                "{}: knob {name} needs a count",
                row.key
            )));
        }
    }
    for &name in flags {
        if row.knobs.get(name).and_then(|k| k.flag()).is_none() {
            return Err(TuningError(format!(
                "{}: knob {name} needs a flag",
                row.key
            )));
        }
    }
    if row.knobs.len() != counts.len() + flags.len() + usize::from(folds) {
        return Err(TuningError(format!(
            "{}: unsupported tuning knobs",
            row.key
        )));
    }
    validate_values(row)
}

// Check catalog choices here; kernels still validate shapes and launch resources.
fn validate_values(row: &Row) -> Result<(), TuningError> {
    let count = |name: &str| row.knobs.get(name).and_then(|k| k.count()).unwrap_or(0);
    let flag = |name: &str| row.knobs.get(name).and_then(|k| k.flag()).unwrap_or(false);
    let [first, last] = row.key.rows.unwrap_or([1, row.key.bucket]);
    let compact_tile = |rows| {
        if rows <= 16 {
            16
        } else if rows <= 128 {
            32
        } else {
            64
        }
    };
    let valid = match row.key.op.as_str() {
        "dense_fold.fold.decode"
        | "dense_fold.split.decode"
        | "dense_fold.wide.verify"
        | "reuse.fold.decode" => {
            let merge = count("finish") == 1
                && count("tile_n") == 16
                && count("row_groups") == 1
                && count("stages") == 2
                && !flag("weight_prefetch");
            let ticket = count("finish") == 0
                && [32, 64].contains(&count("tile_n"))
                && [1, 2].contains(&count("row_groups"))
                && count("stages") == 3
                && (!flag("weight_prefetch") || count("row_groups") == 1);
            match row.key.op.as_str() {
                "dense_fold.split.decode" => merge,
                "reuse.fold.decode" => {
                    ticket && count("row_groups") == 2 && [2, 4, 8].contains(&count("folds"))
                }
                // Wide entries keep one row group and the ticket finish.
                "dense_fold.wide.verify" => ticket && count("row_groups") == 1,
                _ => (merge || ticket) && count("row_groups") == 1,
            }
        }
        "dense_fold.direct.decode" => {
            matches!((count("n_warps"), count("k_warps")), (4, 1) | (1, 4))
        }
        "dense_fold.status.decode" => [16, 32].contains(&count("tile_rows")),
        "dense_fold.status_tails.decode" => [16, 32, 64].contains(&count("tile_rows")),
        "dense_fold.status_wide.verify" | "dense_fold.status_wide_tails.decode" => {
            count("tile_rows") == 32
                && count("stages") >= 3
                && (8..=64).contains(&count("tile_n"))
                && count("tile_n") % 8 == 0
        }
        "dense_prefill.compact.prefill" | "dense_prefill.dspark_main.prefill" => {
            !flag("matrix_fetch")
                && count("tile_rows") == compact_tile(first)
                && count("tile_rows") == compact_tile(last)
                && [2, 3].contains(&count("stages"))
        }
        "dense_prefill.matrix.prefill" => {
            flag("matrix_fetch")
                && matches!(
                    (count("tile_rows"), count("stages")),
                    (32, 3) | (64, 2 | 3) | (128, 2)
                )
        }
        "f32.band.prefill" | "f32.exact.prefill" => [32, 64].contains(&count("tile")),
        "f32.blocked.prefill" => count("tile") == 32,
        "f32.wide.prefill" => count("tile") == 64,
        // Above the decode bucket, at the bound itself; each tile holds
        // whole 128-K chains and a ring of at least two planes.
        "f32.rows.verify" => {
            row.key.bucket == crate::verify_bounds::F32_ROWS
                && first > crate::serving::DECODE_BATCH
                && [32, 64, 128].contains(&count("plane_k"))
                && count("stages") >= 2
                && count("slices") >= 1
        }
        "hc.expand_mix.prefill" => matches!((count("tile"), count("chains")), (4, 4) | (8, 8)),
        "hc.lagged_mix.prefill" => count("tile") == 8 && [4, 8].contains(&count("chains")),
        "hc.finish_chain.verify" | "hc.round_chain.decode" => [4, 8].contains(&count("warps")),
        "hc.finish_rows.verify" => [2, 4, 8].contains(&count("tile")),
        "hc.finish_cols.verify" => {
            [2, 4, 8].contains(&count("tile")) && [8, 16, 32].contains(&count("warps"))
        }
        "hc.finish_twin.verify" => [8, 16, 32].contains(&count("warps")),
        "latent.cache.decode" => count("head_group") == 0,
        "latent.cache.verify" => [0, 1, 2, 4, 8].contains(&count("head_group")),
        "latent.staging.decode" | "latent.staging.verify" => {
            if flag("packed") {
                [32, 64].contains(&count("stage_keys"))
                    && [0, 2, 4, 8].contains(&count("head_group"))
            } else {
                count("stage_keys") == 64 && count("head_group") == 0
            }
        }
        "norm.activation_capacity.decode"
        | "norm.ceil_capacity_bf16.decode"
        | "norm.ceil_capacity_f32.decode"
        | "norm.ceil_registered_bf16.decode"
        | "norm.ceil_registered_f32.decode" => {
            (1..=crate::verify_bounds::CEIL).contains(&count("capacity"))
        }
        "norm.ceil_prefill_bf16.prefill" | "norm.ceil_prefill_f32.prefill" => {
            [4, 8, 16].contains(&count("warps"))
        }
        "norm.input_cached.decode"
        | "norm.input_small.decode"
        | "norm.query_cached.decode"
        | "norm.query_small.decode" => [8, 16, 32].contains(&count("warps")),
        _ if row.schedule == "cluster_fused" => {
            !flag("distributed")
                && [16, 32, 64].contains(&count("tile_n"))
                && (2..=8).contains(&count("splits"))
                && (2..=8).contains(&count("stages"))
        }
        _ => true,
    };
    if !valid {
        return Err(TuningError(format!(
            "{}: unsupported knob values for {}",
            row.key, row.schedule
        )));
    }
    Ok(())
}
