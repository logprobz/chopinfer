//! Qwen's checkpoint admission to the SM120 engines: the shapes its
//! catalogs instantiate (QSA blocks and rings, rotary sections, attention
//! instances, GQA padding and fold scratch, expert counts, hyper-connection
//! and n-gram rows, cluster decode warp blocks), checked in a fixed order
//! between the geometry rules' engine-domain phases.

use crate::backend::{EngineConfig, engine_refusal};
use crate::geometry::Geometry;
use kernels_sm120::attention::qualification;

/// The geometry rules' engine-domain phases, in the order [`admit`] checks
/// them, each after the relations its refusals must not precede.
pub const ENGINE_PHASES: [&str; 5] = [
    model_compiler_sdk::geometry::FIRST_ENGINE_PHASE,
    "after_qsa_context",
    "after_linear_groups",
    "after_row_widths",
    "after_row_shapes",
];

fn served_ranks() -> impl Iterator<Item = usize> {
    crate::contract::SERVED_RANKS
        .into_iter()
        .map(|ranks| ranks as usize)
}

fn served_rank_counts() -> String {
    let mut ranks = served_ranks().collect::<Vec<_>>();
    ranks.sort_unstable();
    ranks
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" and ")
}

/// Admit `config` to the SM120 engines: the first refusal in check order,
/// calling `limits` at each of [`ENGINE_PHASES`] between the checks.
/// `geometry` is the matched geometry, if any, whose compiled keep count
/// and ID the checks read.
pub fn admit(
    config: &EngineConfig,
    geometry: Option<&'static Geometry>,
    limits: &mut dyn FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let [
        first,
        after_qsa_context,
        after_linear_groups,
        after_row_widths,
        after_row_shapes,
    ] = ENGINE_PHASES;
    limits(first)?;
    let vision = &config.vision;
    let side = vision.num_position_embeddings.isqrt();
    if side * side != vision.num_position_embeddings {
        return Err(engine_refusal(
            "vision_config.num_position_embeddings",
            "position table must be square",
            vision.num_position_embeddings,
        ));
    }
    if vision.num_heads == 0
        || vision.hidden_size % vision.num_heads != 0
        || ![64, 72].contains(&(vision.hidden_size / vision.num_heads))
    {
        return Err(engine_refusal(
            "vision_config.hidden_size",
            "vision attention requires an integral head dimension of 64 or 72",
            format!("{} / {} heads", vision.hidden_size, vision.num_heads),
        ));
    }
    let text = &config.text;
    let block = text.indexer_compress_ratio;
    let ring = crate::contract::QSA_RING as usize;
    let page = crate::contract::PAGE_TOKENS as usize;
    if block < 2 || block > ring || ring % block != 0 || page % block != 0 {
        return Err(engine_refusal(
            "text_config.indexer_compress_ratio",
            "QSA block must be at least 2 and divide the ring and page sizes",
            block,
        ));
    }
    if block + crate::contract::MAX_VERIFY_ROWS as usize - 1 > ring {
        return Err(engine_refusal(
            "text_config.indexer_compress_ratio",
            "QSA block plus verify rows minus one must fit QSA_RING",
            block,
        ));
    }
    if block != qualification::BLOCK as usize {
        return Err(engine_refusal(
            "text_config.indexer_compress_ratio",
            &format!("qualified QSA block is {}", qualification::BLOCK),
            block,
        ));
    }
    if text.max_position_embeddings == 0
        || text.max_position_embeddings % block != 0
        || text.max_position_embeddings / block > 65_536
    {
        return Err(engine_refusal(
            "text_config.max_position_embeddings",
            "QSA context must be positive, divisible by block, and at most 65536 blocks",
            text.max_position_embeddings,
        ));
    }
    limits(after_qsa_context)?;
    let eps = text.rms_norm_eps as f32;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(engine_refusal(
            "text_config.rms_norm_eps",
            "epsilon must be finite and positive in f32",
            text.rms_norm_eps,
        ));
    }
    for (path, theta) in [
        ("text_config.rope_parameters.rope_theta", text.rope_theta),
        ("text_config.mtp.rope_theta", text.mtp_rope_theta),
    ] {
        if !theta.is_finite() || theta <= 0.0 || !(theta as f32).is_finite() || theta as f32 <= 0.0
        {
            return Err(engine_refusal(
                path,
                "RoPE theta must be finite and positive in f32",
                theta,
            ));
        }
    }
    let rotary = text.head_dim as f64 * text.rope_partial_rotary_factor;
    if !qualification::ROTARY_SECTIONS
        .iter()
        .any(|(width, _)| f64::from(*width) == rotary)
    {
        return Err(engine_refusal(
            "text_config.rope_parameters.partial_rotary_factor",
            &format!(
                "gated attention and sparse indexer require rotary dim in {:?}",
                qualification::ROTARY_SECTIONS
                    .iter()
                    .map(|(width, _)| *width)
                    .collect::<Vec<_>>()
            ),
            text.rope_partial_rotary_factor,
        ));
    }
    if text.num_key_value_heads == 0 || text.num_attention_heads % text.num_key_value_heads != 0 {
        return Err(engine_refusal(
            "text_config.num_key_value_heads",
            "sparse attention requires an integral GQA group at most 16",
            text.num_key_value_heads,
        ));
    }
    // Direct dense decode is selected at TP4. Check its scratch limit
    // before other topology limits, so widening those keeps this refusal.
    let dense_rank = crate::contract::Shape::TP4.ranks() as usize;
    let dense_q = text.num_attention_heads / dense_rank;
    let dense_kv = (text.num_key_value_heads / dense_rank).max(1);
    if text.head_dim == 128 && dense_q == 16 * dense_kv {
        return Err(engine_refusal(
            "text_config.num_attention_heads",
            "dense paged decode at head dim 128 requires local GQA group at most 8: group 16 exceeds fold scratch",
            format!(
                "TP{dense_rank} group 16 from {} Q heads / {} KV heads",
                text.num_attention_heads, text.num_key_value_heads
            ),
        ));
    }
    if text.num_key_value_heads > 2 {
        return Err(engine_refusal(
            "text_config.num_key_value_heads",
            "attention layout requires one KV head per rank at every served rank count",
            text.num_key_value_heads,
        ));
    }
    if text.linear_num_key_heads == 0
        || text.linear_num_value_heads % text.linear_num_key_heads != 0
        || text.linear_num_value_heads / text.linear_num_key_heads > 3
    {
        return Err(engine_refusal(
            "text_config.linear_num_value_heads",
            "linear value heads must be a multiple of key heads with group at most 3",
            text.linear_num_value_heads,
        ));
    }
    limits(after_linear_groups)?;
    if text.shared_expert_intermediate_size == 0 {
        return Err(engine_refusal(
            "text_config.shared_expert_intermediate_size",
            "shared expert width must be positive before padding",
            0,
        ));
    }
    for rank in served_ranks() {
        if text.linear_num_value_heads / rank > 32 {
            const RESTRICTION: &str = "skinny linear projection requires local value heads \
                in 1..=32 at every served rank count";
            return Err(engine_refusal(
                "text_config.linear_num_value_heads",
                RESTRICTION,
                text.linear_num_value_heads,
            ));
        }
        let local = text.num_experts / rank;
        if !(32..=256).contains(&local) || !local.is_power_of_two() {
            const RESTRICTION: &str = "native expert prefill requires power-of-two local experts \
                in 32..=256 at every served rank count";
            return Err(engine_refusal(
                "text_config.num_experts",
                RESTRICTION,
                text.num_experts,
            ));
        }
        let kv = text.num_key_value_heads;
        if !(kv.is_multiple_of(rank) || rank.is_multiple_of(kv)) {
            return Err(engine_refusal(
                "text_config.num_key_value_heads",
                &format!(
                    "KV heads must split or replicate evenly at served rank counts {}",
                    served_rank_counts()
                ),
                kv,
            ));
        }
        let local_q = text.num_attention_heads / rank;
        let local_kv = (kv / rank).max(1);
        let group = local_q / local_kv;
        if local_q % local_kv != 0 || group > 16 {
            return Err(engine_refusal(
                "text_config.num_key_value_heads",
                "sparse attention requires an integral GQA group at most 16",
                kv,
            ));
        }
        // The catalog adds direct dense entries only at TP4. QSA serves
        // every rank count and pads its groups to whole n8 tiles (12 to 16
        // at TP2). Direct dense prefill pads non-divisors below 8 to 8,
        // but leaves groups 9..15 unpadded, failing its m16 static assert.
        let direct_dense = rank == crate::contract::Shape::TP4.ranks() as usize;
        let padded_group = if !direct_dense {
            group.div_ceil(8) * 8
        } else if 16 % group == 0 {
            group
        } else {
            group.max(8)
        };
        if ![1, 2, 4, 8, 16].contains(&padded_group) {
            return Err(engine_refusal(
                "text_config.num_attention_heads",
                "dense prefill requires padded local GQA group in {1,2,4,8,16} to fill m16 tiles; local groups 9..15 are not padded",
                format!("TP{rank} group {group} padded to {padded_group}"),
            ));
        }
        // Both KV formats are compiled for every direct dense profile.
        if direct_dense && text.head_dim == 256 && [2, 4].contains(&padded_group) {
            return Err(engine_refusal(
                "text_config.num_attention_heads",
                "dense FP8 prefill at head dim 256 requires local GQA group in {1,3,5,6,7,8,16}: groups 2 and 4 spill and fail the zero-stack audit",
                format!(
                    "TP{rank} group {group} from {} Q heads / {kv} KV heads",
                    text.num_attention_heads
                ),
            ));
        }
        if direct_dense && !group.is_multiple_of(2) {
            return Err(engine_refusal(
                "text_config.num_attention_heads",
                "dense verify requires an even local GQA group: eight query rows must fill m16 tiles",
                format!(
                    "TP{rank} group {group} from {} Q heads",
                    text.num_attention_heads
                ),
            ));
        }
        if local_q.div_ceil(8) + local_kv > 8 {
            return Err(engine_refusal(
                "text_config.num_attention_heads",
                "gated attention requires ceil(local Q heads / 8) + local KV heads <= 8",
                text.num_attention_heads,
            ));
        }
    }
    let sections = qualification::ROTARY_SECTIONS
        .iter()
        .find(|(width, _)| *width == rotary as u32)
        .ok_or_else(|| {
            engine_refusal(
                "text_config.rope_parameters.partial_rotary_factor",
                "rotary width has no qualified M-RoPE sections",
                rotary,
            )
        })?
        .1;
    if text.mrope_section != sections.map(|s| s as usize) {
        return Err(engine_refusal(
            "text_config.rope_parameters.mrope_section",
            &format!(
                "qualified M-RoPE sections at rotary {} are {sections:?}",
                rotary as u32
            ),
            format!("{:?}", text.mrope_section),
        ));
    }
    // Runtime budget overrides do not change the compiled keep count.
    let kept = geometry.map_or(text.indexer_budget / block, |g| g.top_blocks() as usize);
    if !qualification::KEPT_BLOCKS
        .iter()
        .any(|&n| n as usize == kept)
    {
        return Err(engine_refusal(
            "text_config.indexer_budget",
            &format!(
                "qualified kept-block counts are {:?}",
                qualification::KEPT_BLOCKS
            ),
            format!("{kept} blocks"),
        ));
    }
    let covered = qualification::HEAD_SHAPES
        .iter()
        .find(|s| s.head_dim as usize == text.head_dim)
        .ok_or_else(|| {
            engine_refusal(
                "text_config.head_dim",
                "head dimension has no qualified attention instances",
                text.head_dim,
            )
        })?;
    for rank in served_ranks() {
        let local_q = text.num_attention_heads / rank;
        if rank == dense_rank && !covered.dense_groups.contains(&(local_q as u32)) {
            return Err(engine_refusal(
                "text_config.num_attention_heads",
                &format!(
                    "TP{rank} group {local_q} at head dim {} has no qualified dense prefill/verify instance; qualified groups are {:?}",
                    text.head_dim, covered.dense_groups
                ),
                format!("TP{rank} group {local_q}"),
            ));
        }
        if !covered.rows.contains(&(local_q as u32))
            || !covered.sparse_groups.contains(&(local_q as u32))
        {
            return Err(engine_refusal(
                "text_config.num_attention_heads",
                "local gated rows and QSA group must occur in the attention qualification table",
                format!("TP{rank} Q{local_q} at head dim {}", text.head_dim),
            ));
        }
    }
    let ngram_heads = text
        .ngram_size
        .checked_sub(1)
        .and_then(|n| n.checked_mul(text.heads_per_ngram))
        .ok_or("invalid text_config.ngram_size")?;
    if ngram_heads == 0 || served_ranks().any(|rank| ngram_heads % rank != 0) {
        return Err(engine_refusal(
            "text_config.heads_per_ngram",
            &format!(
                "total ngram heads must be positive and divisible by served rank counts {}",
                served_rank_counts()
            ),
            text.heads_per_ngram,
        ));
    }
    if text.hidden_size == 0 || !text.hidden_size.is_multiple_of(256) {
        return Err(engine_refusal(
            "text_config.hidden_size",
            "hyper-connection rows require a positive width divisible by 256",
            text.hidden_size,
        ));
    }
    if text.ple_embed_dim == 0 || !text.ple_embed_dim.is_multiple_of(ngram_heads * 16) {
        return Err(engine_refusal(
            "text_config.ple_embed_dim",
            "ngram rows require positive head widths divisible by 16",
            text.ple_embed_dim,
        ));
    }
    limits(after_row_widths)?;
    limits(after_row_shapes)?;
    for ranks in served_ranks() {
        for (role, k) in [
            ("lin_in", text.hidden_size),
            ("attn_in", text.hidden_size),
            ("router", text.hidden_size),
            ("shared_gate_up", text.hidden_size),
            ("mtp_fc", text.hidden_size),
            ("hc_in", text.hc_count * text.hidden_size),
            ("hc_down", text.hc_count * text.hidden_size),
            ("ple_key", text.ple_embed_dim),
            ("ple_value", text.ple_embed_dim),
            (
                "out",
                text.linear_num_value_heads / ranks * text.linear_value_head_dim,
            ),
            ("attn_out", text.num_attention_heads / ranks * text.head_dim),
            (
                "shared_down",
                (text.shared_expert_intermediate_size / ranks).next_multiple_of(128),
            ),
        ] {
            for buckets in [[1, 2, 4], [8, 16, 32]] {
                if !(1..=8).any(|blocks| {
                    buckets.into_iter().all(|bucket| {
                        u32::try_from(k).ok().is_some_and(|k| {
                            kernels_sm120::gemm::fused::cluster_decode_shape(k, bucket, blocks)
                                .is_some()
                        })
                    })
                }) {
                    let id = geometry.map_or("unmatched", |g| g.id);
                    return Err(format!(
                        "geometry {id} role {role}: cluster decode K={k} has no legal warp-block count in 1..=8 for buckets {buckets:?} (at most 8 CTAs and 101376 shared bytes)"
                    ));
                }
            }
        }
    }
    if eps.to_bits() != 1e-6_f32.to_bits() {
        return Err(engine_refusal(
            "text_config.rms_norm_eps",
            "hyper-connection rows require f32 epsilon 1e-6",
            text.rms_norm_eps,
        ));
    }

    Ok(())
}
