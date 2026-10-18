//! The geometry and table choices around native latent attention.
use crate::geometry::Geometry;
use crate::own_ops::rope::RopeTable;
use kernel_core::contract::Validate;
use kernel_core::latent_glue::{
    LatentQueryRope, LatentWindowKeys, LatentWindowWrite, WindowAddressing,
};

pub fn query_rope(g: &Geometry, ranks: u32) -> Result<LatentQueryRope, String> {
    crate::admission::check_ranks(g, ranks)?;
    let a = g.shape.attention;
    let query = LatentQueryRope {
        heads: a.heads / ranks,
        latent: a.head_dim,
        rotary: a.rope_dim,
    };
    query.validate().map_err(|e| e.to_string())?;
    Ok(query)
}

pub fn ring(g: &Geometry) -> WindowAddressing {
    WindowAddressing::Circular {
        slots: crate::state::ring_cells(crate::state::Widths::of(g)) as u32,
    }
}

pub fn window_write(g: &Geometry, addressing: WindowAddressing) -> LatentWindowWrite {
    window_write_for_storage(g, addressing, crate::state::KvStorage::Native)
}

pub fn window_write_for_storage(
    g: &Geometry,
    addressing: WindowAddressing,
    storage: crate::state::KvStorage,
) -> LatentWindowWrite {
    LatentWindowWrite {
        kv: window_format(storage),
        epsilon_bits: (g.shape.eps as f32).to_bits(),
        latent: g.shape.attention.head_dim,
        rotary: g.shape.attention.rope_dim,
        addressing,
    }
}

pub fn window_keys(g: &Geometry, addressing: WindowAddressing) -> LatentWindowKeys {
    LatentWindowKeys {
        window: g.shape.attention.window,
        addressing,
    }
}

/// Ratio-zero layers and draft stages use the window table. Every other
/// attention layer uses the compressed table for both Q and window KV.
/// The caller asks `OwnOps::rope_cos_sin` with `RopeRows::Positions` and
/// binds its F32 row-packed output to both shared glue operations.
pub fn rope_table(g: &Geometry, layer: u32) -> Result<RopeTable, String> {
    if layer >= g.layers() + g.dspark_stages() {
        return Err("RoPE layer outside geometry".into());
    }
    Ok(if g.compress_ratio(layer) == 0 {
        RopeTable::Window
    } else {
        RopeTable::Compressed
    })
}

/// The query-down norm keeps F32 until either query projection quantizes it.
pub fn query_norm(g: &Geometry) -> kernel_core::rmsnorm::RmsNormRequirement {
    kernel_core::rmsnorm::RmsNormRequirement {
        input: kernel_core::contract::DType::F32,
        output: kernel_core::contract::DType::F32,
        ..crate::rows::norm(g, g.shape.attention.q_rank)
    }
}

/// Gather the prior window before a prefill writes its staged chunk.
pub fn window_stage(
    g: &Geometry,
    staged_slots: u32,
) -> Result<kernel_core::latent_window_stage::LatentWindowStage, String> {
    window_stage_for_storage(g, staged_slots, crate::state::KvStorage::Native)
}

pub fn window_stage_for_storage(
    g: &Geometry,
    staged_slots: u32,
    storage: crate::state::KvStorage,
) -> Result<kernel_core::latent_window_stage::LatentWindowStage, String> {
    let query = kernel_core::latent_window_stage::LatentWindowStage {
        kv: window_format(storage),
        latent: g.shape.attention.head_dim,
        window: g.shape.attention.window,
        ring_slots: ring(g).slots(),
        staged_slots,
    };
    query.validate().map_err(|e| e.to_string())?;
    Ok(query)
}

/// Use with gathered prefix cells; the ordinary staged lists assume every
/// cell in their interval was populated.
pub fn staged_window_keys(
    g: &Geometry,
    staged_slots: u32,
) -> Result<kernel_core::latent_staged_window_keys::LatentStagedWindowKeys, String> {
    let query = kernel_core::latent_staged_window_keys::LatentStagedWindowKeys {
        window: g.shape.attention.window,
        ring_slots: ring(g).slots(),
        staged_slots,
    };
    query.validate().map_err(|e| e.to_string())?;
    Ok(query)
}

fn window_format(
    storage: crate::state::KvStorage,
) -> kernel_core::latent_attention::LatentKvFormat {
    match storage {
        crate::state::KvStorage::Native => {
            kernel_core::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32
        }
        crate::state::KvStorage::Bf16 => kernel_core::latent_attention::LatentKvFormat::Bf16,
    }
}

#[cfg(test)]
mod bf16_tests {
    use super::*;
    use crate::state::KvStorage;
    use kernel_core::latent_attention::LatentKvFormat;
    #[test]
    fn bf16_writer_and_staging_share_the_attention_layout() {
        for g in crate::geometry::GEOMETRIES {
            assert_eq!(
                window_write_for_storage(g, ring(g), KvStorage::Bf16).kv,
                LatentKvFormat::Bf16
            );
            assert_eq!(
                window_stage_for_storage(g, crate::serving::PREFILL_TOKENS + 127, KvStorage::Bf16)
                    .unwrap()
                    .kv,
                LatentKvFormat::Bf16
            );
            assert_eq!(
                window_write(g, ring(g)).kv,
                LatentKvFormat::Fp8E4M3Ue8M0Block32
            );
        }
    }
}
