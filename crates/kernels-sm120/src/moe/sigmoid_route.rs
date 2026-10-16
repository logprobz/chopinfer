//! The sigmoid-bias top-8 routes of 256 tensor-parallel experts
//! (`moe/route.cuh`): the one-CTA decode route and the router GEMV with the
//! route fused into its tail (`gemm/fragment_router.cuh`). The serving model
//! names the cubins and entries.

use super::*;

/// Most tokens of one decode route or fused router tail: eight routes
/// each for one 256-thread CTA (`route_decode_mimo_body` in moe/route.cuh,
/// `fused_route_tail` in gemm/fragment_router.cuh).
pub const ROUTE_MAX_TOKENS: u32 = 32;
pub fn fused_router_arguments(
    handles: [u64; 14],
    scaling_factor: f32,
) -> Result<ArgumentList, Refusal> {
    if !scaling_factor.is_finite() || scaling_factor <= 0.0 {
        return Err(Refusal("router scale must be finite and positive"));
    }
    let mut values = handles.map(Argument::DeviceBuffer).to_vec();
    values.push(Argument::U32(scaling_factor.to_bits()));
    let args = ArgumentList(values);
    args.validate()?;
    Ok(args)
}

/// The fused router GEMV and route tail cubin `prefix`, with an entry
/// `{prefix}_t{tokens}_v1` per decode bucket.
pub fn fused_router_cubin_spec(prefix: &str) -> CubinSpec {
    CubinSpec {
        name: prefix.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: Vec::new(),
        headers: vec!["#include \"gemm/fragment_router.cuh\"".into()],
        defines: Vec::new(),
        entries: [1_u32, 2, 4, 8, 16, 32]
            .into_iter()
            .map(|tokens| EntrySpec::new(
                format!("{prefix}_t{tokens}_v1"),
                vec![
                    ptr("weight", "const uint8_t"),
                    ptr("activation", "const uint16_t"),
                    ptr("logits", "float"),
                    ptr("partials", "float"),
                    ptr("semaphores", "uint32_t"),
                    ptr("live_tokens", "const int32_t"),
                    ptr("bias", "const float"),
                    ptr("topk_ids", "int32_t"),
                    ptr("topk_weights", "float"),
                    ptr("counts", "int32_t"),
                    ptr("offsets", "int32_t"),
                    ptr("sorted_rows", "int32_t"),
                    ptr("num_padded_rows", "int32_t"),
                    ptr("active_experts", "int32_t"),
                    EntryParam::U32 { name: "scale_bits".into() },
                ],
                format!("chop::gemm::run_fragment_router_route<{}>(weight,activation,logits,partials,semaphores,live_tokens,bias,topk_ids,topk_weights,counts,offsets,sorted_rows,num_padded_rows,active_experts,__uint_as_float(scale_bits));", tokens.max(8)),
                256,
                1,
            ))
            .collect(),
    }
}

/// One CTA selects top-8 and builds stable, padded expert ranges for decode.
pub fn decode_route_plan(tokens_capacity: u32, live_tokens: u32) -> Result<LaunchPlan, Refusal> {
    // One route per thread of a 256-thread CTA; the kernel traps above it.
    if tokens_capacity > ROUTE_MAX_TOKENS || live_tokens > ROUTE_MAX_TOKENS {
        return Err(Refusal(
            "sigmoid decode route holds at most 32 tokens (ROUTE_MAX_TOKENS)",
        ));
    }
    if !DECODE_BUCKETS.contains(&tokens_capacity) || live_tokens > tokens_capacity {
        return Err(Refusal("decode route exceeds prepared token bucket"));
    }
    let plan = LaunchPlan {
        grid: [1, 1, 1],
        block: [tokens_capacity.saturating_mul(32).clamp(256, 1024), 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn decode_route_arguments(
    handles: [u64; 10],
    scaling_factor: f32,
) -> Result<ArgumentList, Refusal> {
    if !scaling_factor.is_finite() || scaling_factor <= 0.0 {
        return Err(Refusal("route scale must be finite and positive"));
    }
    let mut values = handles.map(Argument::DeviceBuffer).to_vec();
    values.push(Argument::U32(scaling_factor.to_bits()));
    let args = ArgumentList(values);
    args.validate()?;
    Ok(args)
}

/// The decode route cubin `name` with its entry `symbol`.
pub fn decode_route_cubin_spec(name: &str, symbol: &str) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![PathBuf::from("common"), PathBuf::from("moe")],
        headers: vec!["#include \"moe/moe.cuh\"".into()],
        defines: Vec::new(),
        entries: vec![EntrySpec::new(
            symbol.into(),
            vec![
                ptr("logits", "const float"),
                ptr("bias", "const float"),
                ptr("ids", "int32_t"),
                ptr("weights", "float"),
                ptr("counts", "int32_t"),
                ptr("offsets", "int32_t"),
                ptr("sorted_rows", "int32_t"),
                ptr("num_padded_rows", "int32_t"),
                ptr("live_tokens", "const int32_t"),
                ptr("active_experts", "int32_t"),
                EntryParam::U32 {
                    name: "scale_bits".into(),
                },
            ],
            "chop::moe::route_decode_mimo(logits,bias,ids,weights,counts,offsets,sorted_rows,num_padded_rows,live_tokens,active_experts,__uint_as_float(scale_bits));".into(),
            1024,
            1,
        )],
    }
}
