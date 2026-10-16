//! Launch planning and entry bodies for contiguous BF16-pair attention.

use kernel_build::EntryParam;
use kernel_core::contract::{LaunchPlan, MultimodalQuery, Refusal, Validate};

/// Threads per CTA: eight warps, each owning one m16 tile of grouped rows.
pub const THREADS: u32 = 256;

/// Grouped query rows per CTA.
const ROWS: u32 = 128;

/// Query tokens one CTA covers: 128 grouped rows of one KV head.
pub fn tokens_per_cta(query: MultimodalQuery) -> Result<u32, Refusal> {
    let MultimodalQuery::ContiguousAttentionPairs {
        q_heads, kv_heads, ..
    } = query
    else {
        return Err(Refusal("expected contiguous attention query"));
    };
    query.validate()?;
    super::kernel::limits(query)?;
    let group = q_heads / kv_heads;
    if !16u32.is_multiple_of(group) {
        return Err(Refusal(
            "contiguous attention packs whole GQA groups in m16 tiles",
        ));
    }
    Ok(ROWS / group)
}

/// Key rows per shared tile and its padded row bytes (`varlen.cuh` Traits).
const KEYS: u32 = 64;
fn row_bytes(head_dim: u32) -> u32 {
    (head_dim.div_ceil(16) * 16 + 8) * 2
}

/// The launch for a prepared capacity. The grid covers the whole capacity;
/// CTAs past the device live token count return at once, so one plan serves
/// every live count and graph replay. Pair entries keep two stages of four
/// key tiles (k hi, k lo, v hi, v lo) in dynamic shared memory.
pub fn plan(query: MultimodalQuery) -> Result<LaunchPlan, Refusal> {
    let tokens = tokens_per_cta(query)?;
    let MultimodalQuery::ContiguousAttentionPairs {
        kv_heads,
        head_dim,
        capacity,
        ..
    } = query
    else {
        unreachable!("tokens_per_cta checked the variant");
    };
    let dynamic_shared_bytes = 2 * 4 * KEYS * row_bytes(head_dim);
    let plan = LaunchPlan {
        grid: [capacity.div_ceil(tokens), kv_heads, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// The C++ traits type for a query.
pub fn traits(query: MultimodalQuery) -> Result<String, Refusal> {
    tokens_per_cta(query)?;
    let MultimodalQuery::ContiguousAttentionPairs {
        q_heads,
        kv_heads,
        head_dim,
        causal,
        window_left,
        window_right,
        key0_sink,
        ..
    } = query
    else {
        unreachable!("tokens_per_cta checked the variant");
    };
    let window = |value: Option<u32>| value.map_or(-1, i64::from);
    Ok(format!(
        "chop::attention::varlen::Traits<{head_dim},{q_heads},{kv_heads},{causal},{},{},{key0_sink}>",
        window(window_left),
        window(window_right),
    ))
}

/// The ABI of the BF16 pair entries: packed pair rows `[T, 2 * (q_heads + 2
/// * kv_heads) * head_dim]` BF16 and F32 output.
pub fn pairs_params() -> Vec<EntryParam> {
    let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    vec![
        ptr("qkv", "const uint16_t"),
        ptr("cu_seqlens", "const int32_t"),
        ptr("sinks", "const float"),
        ptr("output", "float"),
        ptr("live_tokens", "const uint32_t"),
        EntryParam::U32 {
            name: "live_sequences".into(),
        },
        EntryParam::U32 {
            name: "scale_bits".into(),
        },
    ]
}

/// The entry body for a query.
pub fn body(query: MultimodalQuery) -> Result<String, Refusal> {
    Ok(format!(
        "chop::attention::varlen::attention_pairs<{}>(\
         reinterpret_cast<const __nv_bfloat16*>(qkv),\
         cu_seqlens,sinks,output,live_tokens,live_sequences,scale_bits);",
        traits(query)?
    ))
}
