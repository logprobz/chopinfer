//! Native latent attention queries from the admitted geometry. Key lists
//! carry visibility; the drafter's full block follows its main-KV window.

use crate::admission::check_ranks;
use crate::geometry::Geometry;
use crate::serving::{DECODE_BATCH, MAX_VERIFY_ROWS, PAGE_TOKENS, PREFILL_TOKENS};
use kernel_core::contract::{AttentionPhase, Validate};
use kernel_core::latent_attention::{
    LATENT_WINDOW_SLOTS, LatentAttentionOutput, LatentAttentionQuery, LatentKvFormat,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keys {
    Window,
    Compressed,
    DraftBlock,
}

pub fn query(
    g: &Geometry,
    ranks: u32,
    phase: AttentionPhase,
    keys: Keys,
) -> Result<LatentAttentionQuery, String> {
    check_ranks(g, ranks)?;
    let s = &g.shape;
    if s.attention.window != LATENT_WINDOW_SLOTS {
        return Err("native latent attention requires a 128-slot window".into());
    }
    if keys == Keys::DraftBlock && (s.dspark.stages == 0 || phase != AttentionPhase::Verify) {
        return Err("draft latent attention requires a full verify block".into());
    }
    let extra = match keys {
        Keys::Window => 0,
        Keys::Compressed => s.indexer.top_k,
        Keys::DraftBlock => s.dspark.block,
    };
    let capacity = match (phase, keys) {
        (_, Keys::DraftBlock) => DECODE_BATCH * s.dspark.block,
        (AttentionPhase::Prefill, _) => PREFILL_TOKENS,
        (AttentionPhase::Decode, _) => DECODE_BATCH,
        (AttentionPhase::Verify, _) => DECODE_BATCH * MAX_VERIFY_ROWS,
    };
    Ok(LatentAttentionQuery::Attention {
        heads: s.attention.heads / ranks,
        latent: s.attention.head_dim,
        max_keys: s
            .attention
            .window
            .checked_add(extra)
            .ok_or("latent key budget overflows")?,
        softmax_scale_bits: (s.attention.head_dim as f32).powf(-0.5).to_bits(),
        page_size: PAGE_TOKENS as u32,
        kv: LatentKvFormat::Fp8E4M3Ue8M0Block32,
        phase,
        output: LatentAttentionOutput::Normalized,
        sink: true,
        tail_rope_dims: s.attention.rope_dim,
        second_fp4_keys: keys == Keys::Compressed,
        capacity,
    })
}

/// Compressed source geometry is measured in stored positions, not tokens.
pub fn compressed_query(
    g: &Geometry,
    ranks: u32,
    phase: AttentionPhase,
    ratio: u32,
) -> Result<LatentAttentionQuery, String> {
    if !matches!(ratio, 1 | 2) || !g.shape.compress_ratios.contains(&ratio) {
        return Err("native compressed attention requires an admitted ratio of 1 or 2".into());
    }
    let mut query = query(g, ranks, phase, Keys::Compressed)?;
    if let LatentAttentionQuery::Attention { page_size, .. } = &mut query {
        *page_size = PAGE_TOKENS as u32 / ratio;
    }
    Ok(query)
}

/// Every attention shape used by this geometry at one rank count.
pub fn queries(g: &Geometry, ranks: u32) -> Result<Vec<(Keys, LatentAttentionQuery)>, String> {
    let mut out = Vec::new();
    for keys in [Keys::Window, Keys::Compressed] {
        let present = g.shape.compress_ratios.iter().any(|&ratio| {
            (keys == Keys::Window && ratio == 0) || (keys == Keys::Compressed && ratio != 0)
        });
        if present {
            for phase in [
                AttentionPhase::Prefill,
                AttentionPhase::Decode,
                AttentionPhase::Verify,
            ] {
                out.push((keys, query(g, ranks, phase, keys)?));
            }
        }
    }
    if g.shape.compress_ratios.contains(&2) {
        for phase in [
            AttentionPhase::Prefill,
            AttentionPhase::Decode,
            AttentionPhase::Verify,
        ] {
            out.push((Keys::Compressed, compressed_query(g, ranks, phase, 2)?));
        }
    }
    if g.dspark_stages() != 0 {
        out.push((
            Keys::DraftBlock,
            query(g, ranks, AttentionPhase::Verify, Keys::DraftBlock)?,
        ));
    }
    Ok(out)
}

/// Select storage without changing the source visibility or attention math.
pub fn with_storage(
    mut query: LatentAttentionQuery,
    storage: crate::state::KvStorage,
) -> Result<LatentAttentionQuery, String> {
    let LatentAttentionQuery::Attention { kv, .. } = &mut query else {
        return Err("latent comparison storage requires an attention query".into());
    };
    *kv = match storage {
        crate::state::KvStorage::Native => LatentKvFormat::Fp8E4M3Ue8M0Block32,
        crate::state::KvStorage::Bf16 => LatentKvFormat::Bf16,
    };
    query.validate().map_err(|e| e.to_string())?;
    Ok(query)
}
