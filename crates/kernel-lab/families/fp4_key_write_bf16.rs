//! Exact BF16 comparison index keys and 64-token page geometries.

use kernel_build::CubinSpec;
use kernel_core::compressed_kv::{CompressedKvStorage, CompressedKvWriteQuery};
use kernel_core::pooled_indexer::{IndexerCandidates, IndexerKeyFormat, PositionIndexerQuery};
use kernels_sm120::gated_pooling::key_write::{Entry, cubin_spec};

pub const FAMILY: &str = "fp4_key_write_bf16";
pub const CUBIN: &str = "fp4_key_write_bf16_v1";

pub fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for ratio in [1, 2] {
        let page = 64 / ratio;
        let query = PositionIndexerQuery {
            heads: 32, head_dim: 128, compression_ratio: ratio,
            tail_rope_dims: 64, key_norm_eps_bits: 1e-20f32.to_bits(),
            max_top_k: 512, page_size: page, candidates: IndexerCandidates::All, capacity: 12,
        };
        entries.push(Entry::index_with_storage(&query, IndexerKeyFormat::Bf16,
            &format!("chop_fp4_index_bf16_r{ratio}_p{page}_v1")).unwrap());
        // The ratio-one native entries and latent BF16 entry already exist.
        if ratio == 2 {
            entries.push(Entry::index(&query, "chop_fp4_index_r2_p32_v1").unwrap());
            for (tag, storage) in [("packed", CompressedKvStorage::Packed), ("bf16", CompressedKvStorage::Bf16)] {
                entries.push(Entry::compressed(&CompressedKvWriteQuery {
                    width: 512, tail_rope_dims: 64, compression_ratio: ratio,
                    page_size: page, storage, capacity: 12,
                }, &format!("chop_fp4_compressed_{tag}_r2_p32_v1")).unwrap());
            }
        }
    }
    entries
}

pub fn all_entries() -> Vec<Entry> {
    crate::fp4_key_write::entries().into_iter().chain(entries()).collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[CUBIN], vec![cubin_spec(CUBIN, &entries()).unwrap()])
}
