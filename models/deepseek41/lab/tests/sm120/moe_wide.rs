//! Complete wide MoE pairs keep explicit reference schedules and guarded spans.
mod device;
mod full;
mod graphs;
mod recorded;
mod router;
mod timing;

#[test]
#[ignore = "requires frozen programs, weights and an allocated GPU pair"]
fn full_gpu() {
    let options = full::Options::environment().unwrap();
    if std::env::var("CHOPINFER_MOE_WIDE_DRY_RUN").as_deref() == Ok("1") {
        full::run(None, options).unwrap();
        return;
    }
    let ran = kernel_harness::gpu_test(4_u64 << 30, |s| full::run(Some(s), options)).unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}

#[test]
#[ignore = "requires frozen programs, weights and an allocated GPU"]
fn timing_gpu() {
    let options = timing::Options::environment().unwrap();
    let ran = kernel_harness::gpu_test(8_u64 << 30, |s| timing::run(s, options)).unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}

/// Above 64 rows the pair check accepts only the 8192-row chains as the
/// reference and the wide direct queues as the candidate: a served export
/// in either role, a swapped pair or a capacity below the rows fails.
#[test]
fn wide_pair_rejects_a_served_candidate() {
    let chain: Vec<Vec<&str>> = vec![
        vec![
            "deepseek41_chop_moe_chain_mxfp4_a8ceil_before_gateup_e384k6_n1152_k5120_r64_t8192_clamp4120000041200000_lanes_v1",
            "chop_deepseek41_moe_ceil_w3456_v1",
        ],
        vec!["deepseek41_chop_moe_chain_mxfp4_a8ceil_before_down_e384k6_n5120_k576_r64_t8192_v1"],
        vec!["chop_moe_deepseek41_sum_add_f32_k6_w5120_l384_t8192_v1"],
    ];
    let wide = |gate: &'static str, down: &'static str| -> Vec<Vec<&'static str>> {
        vec![
            vec![
                "chop_deepseek41_moe_wide_t192_e384_k6_pack_v1",
                gate,
                "chop_deepseek41_moe_bandwidth_direct_t192_e384_k6_gate_finish_v1",
            ],
            vec![down],
            vec!["chop_deepseek41_moe_floor_t192_e384_k6_down_finish_v1"],
        ]
    };
    let raw = wide(
        "chop_deepseek41_moe_absorb_lanes_t192_e384_k6_gate_small_v1",
        "chop_deepseek41_moe_strip2_t192_e384_k6_down_small_v1",
    );
    let encoded = wide(
        "chop_deepseek41_moe_absorb_lanes_t192_e384_k6_gate_small_scale_tiles_w1_v1",
        "chop_deepseek41_moe_strip2_t192_e384_k6_down_small_scale_tiles_w1_v1",
    );
    let check = recorded::check_wide_pair;
    for rows in [65, 128, 192] {
        check(rows, &chain, &raw).unwrap();
        check(rows, &chain, &encoded).unwrap();
        assert!(check(rows, &chain, &chain).is_err());
        assert!(check(rows, &raw, &raw).is_err());
        assert!(check(rows, &raw, &chain).is_err());
    }
    assert!(check(193, &chain, &raw).is_err());
    let mut renamed = raw.clone();
    renamed[1][0] = "chop_deepseek41_moe_strip2_t256_e384_k6_down_small_v1";
    assert!(check(128, &chain, &renamed).is_err());
    let mut short = raw.clone();
    short[0].pop();
    assert!(check(128, &chain, &short).is_err());
    check(64, &chain, &chain).unwrap();
}

#[test]
fn router64_cpu_inventory() {
    router::cpu_inventory().unwrap();
}

#[test]
#[ignore = "requires an allocated GPU pair"]
fn router64_gpu() {
    let worker: u32 = std::env::var("CHOPINFER_MOE_WIDE_WORKER")
        .expect("set worker 0 or 1")
        .parse()
        .unwrap();
    assert!(worker < 2);
    let checked = std::env::var("CHOPINFER_MOE_WIDE_MEMCHECK").as_deref() == Ok("1");
    if std::env::var("CHOPINFER_MOE_WIDE_DRY_RUN").as_deref() == Ok("1") {
        router::cpu_worker(worker, checked).unwrap();
        return;
    }
    let ran = kernel_harness::gpu_test(256 << 20, |s| router::run(s, worker, checked)).unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
