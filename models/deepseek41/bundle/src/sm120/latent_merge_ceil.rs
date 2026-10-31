//! The `attention_merge_ceil` candidate's cubin: each served decode and
//! verify latent merge with normalized output, followed in the same CTA by
//! WO_a's row-sliced input quantizer groups over the merged head
//! (`latent/sources_merge_ceil.cuh`). The merge keeps its plan, its
//! arguments and every BF16 output; the groups keep the sliced quantizer's
//! arithmetic, codes, scales and status bits (`fold_fused::sliced`). Flash
//! TP4 only; the serving side is `latent_merge_ceil_serving`.
use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::{LatentAttentionOutput, LatentAttentionQuery};
use kernels_sm120::latent::sources::{self, Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "latent_merge_ceil";
pub const CUBIN: &str = "latent_merge_ceil_deepseek41_sm120";

/// The latent entries whose merge has a quantizing twin: Flash TP4's served
/// decode and verify entries with normalized output, in both KV layouts,
/// one per merge symbol.
pub fn entries() -> Vec<Entry> {
    let mut served =
        super::latent_scheduled::serving_entries(&FLASH, 4).expect("served latent entries");
    served.extend(super::latent::bf16_entries(&FLASH, 4).expect("BF16 latent entries"));
    let mut entries: Vec<Entry> = Vec::new();
    for entry in served {
        let merged = matches!(
            entry.query,
            LatentAttentionQuery::Attention {
                phase: AttentionPhase::Decode | AttentionPhase::Verify,
                output: LatentAttentionOutput::Normalized,
                ..
            }
        );
        if merged
            && !entries
                .iter()
                .any(|e| e.merge_symbol() == entry.merge_symbol())
        {
            entries.push(entry);
        }
    }
    entries
}

/// Whether `geometry` at `ranks` has the quantizing merges: Flash TP4,
/// whose WO_a status folds quantize their input with the sliced quantizer.
pub fn serves(geometry: &Geometry, ranks: u32) -> bool {
    geometry.id == FLASH.id && ranks == 4
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| serves(g, 4)) {
        return Ok(Vec::new());
    }
    let floor = kernels_sm120::gemm::mxfp8_fold_cluster_fused::floor_bits();
    Ok(vec![(
        FAMILY,
        sources::merge_ceil_cubin_spec(CUBIN, &entries(), floor).map_err(|e| e.to_string())?,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_served_decode_and_verify_merge_has_one_twin() {
        let entries = entries();
        let spec = &cubins(&[FLASH]).unwrap()[0].1;
        assert_eq!(spec.name, CUBIN);
        assert_eq!(spec.entries.len(), entries.len());
        for (entry, twin) in entries.iter().zip(&spec.entries) {
            assert_eq!(twin.symbol, entry.merge_ceil_symbol());
            assert!(twin.symbol.ends_with("merge_ceil_v1"), "{}", twin.symbol);
            // The twin's ABI is the merge's, then codes, scales, status and
            // the status row offset.
            let merge = entry.merge_params().unwrap();
            assert_eq!(twin.params.len(), merge.len() + 4);
            assert_eq!(
                format!("{:?}", &twin.params[..merge.len()]),
                format!("{merge:?}")
            );
        }
        for symbol in [
            "chop_latent_sources_deepseek41_h16_d512_mixed_normalized_s1_r64_verify_heads_merge_v1",
            "chop_latent_sources_deepseek41_h16_d512_mixed_normalized_s1_r64_verify_p32_heads_merge_v1",
            "chop_latent_sources_deepseek41_h16_d512_window_normalized_s1_r64_verify_heads_merge_v1",
            "chop_latent_sources_deepseek41_h16_d512_window_normalized_s1_r64_verify_draft_block_merge_v1",
        ] {
            assert!(
                entries.iter().any(|e| e.merge_symbol() == symbol),
                "{symbol}"
            );
        }
        assert!(cubins(&[]).unwrap().is_empty());
        assert!(serves(&FLASH, 4) && !serves(&FLASH, 2));
    }
}
