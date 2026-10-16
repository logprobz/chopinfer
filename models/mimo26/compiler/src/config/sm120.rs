//! The engine domain the supported geometries admit agrees with the shared
//! SM120 kernels' specializations: the route, the expert fragments, the
//! comm epilogue and the drafter rows.

#[test]
fn engine_domain_agrees_with_shared_specializations() {
    use model_mimo26_contract::{geometry, serving, sm120};
    let rules: serde_json::Value = serde_json::from_str(geometry::SUPPORTED).unwrap();
    let domain = &rules["engine_domain"];
    let only = |field: &str| {
        let values = domain[field]["values"].as_array().unwrap();
        assert_eq!(values.len(), 1, "{field}");
        values[0].as_u64().unwrap() as u32
    };
    let (experts, top_k, hidden) = (
        only("n_routed_experts"),
        only("num_experts_per_tok"),
        only("hidden_size"),
    );
    let route = include_str!("../../../../../crates/kernels-sm120/cuda/moe/route.cuh");
    assert!(route.contains(&format!(
        "static_assert({experts} == {} * 32);",
        experts / 32
    )));
    assert!(route.contains("static_assert(Threads == 256);"));
    assert!(route.contains(&format!(
        "float scores[{}], unbiased[{}];",
        experts / 32,
        experts / 32
    )));
    assert!(route.contains(&format!("for (int k = 0; k < {top_k}; ++k)")));
    let max_tokens = kernels_sm120::moe::sigmoid_route::ROUTE_MAX_TOKENS;
    assert_eq!(max_tokens * top_k, 256);
    assert!(route.contains(&format!("if (tokens > {max_tokens}) __trap();")));
    assert!(serving::DECODE_ENTRY_ROWS <= max_tokens);
    assert!(kernels_sm120::moe::sigmoid_route::decode_route_plan(max_tokens + 1, 1).is_err());
    assert!(sm120::moe::fused_router_plan(max_tokens + 1, 1).is_err());

    let fragments = include_str!("../../../../../crates/kernels-sm120/src/moe/mxfp4_fragment.rs");
    let specialized = fragments
        .split("const SPECIALIZED: [Shape; 3] = [")
        .nth(1)
        .unwrap()
        .split("];\n")
        .next()
        .unwrap();
    let specialized: String = specialized.split_whitespace().collect();
    for profile in sm120::PROFILES {
        let shape = profile.rank_shape().unwrap();
        let intermediate = only("moe_intermediate_size") / shape.ranks();
        assert!(specialized.contains(&format!(
            "Shape{{top_k:{top_k},hidden:{hidden},experts:{experts},intermediate:{intermediate},}}"
        )));
    }
    assert_eq!(hidden, kernels_sm120::comm::FUSED_EPILOGUE.width);
    assert_eq!(
        kernels_sm120::comm::input::ROUTED_COMBINE_TOP8.policy,
        format!("chop::comm::input::RoutedCombine<{top_k}>")
    );

    let dflash = include_str!("../../../../../crates/kernels-sm120/cuda/row/dflash.cuh");
    assert!(dflash.contains(&format!("head * {}", only("dflash.head_dim"))));
    assert!(dflash.contains(&format!(
        "rotated<{}, true, Table>",
        only("dflash.rotary_dim")
    )));
    assert!(dflash.contains(&format!(
        "position, {}.f, cos_sin",
        only("dflash.rope_theta")
    )));
    let rotary = include_str!("../../../../../crates/kernels-sm120/cuda/common/rotary.cuh");
    for field in ["rotary_dim", "dflash.rotary_dim"] {
        assert!(rotary.contains(&format!("if constexpr (Rotary == {})", only(field))));
    }
    for field in ["rope_theta", "swa_rope_theta"] {
        let thetas = domain[field]["values"].as_array().unwrap();
        assert_eq!(thetas.len(), 2);
        assert!(rotary.contains(&format!(
            "if (theta == {}.f || theta == {}.f)",
            thetas[0].as_u64().unwrap(),
            thetas[1].as_u64().unwrap()
        )));
    }
}
