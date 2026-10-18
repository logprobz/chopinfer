//! Paged trained FP4 index keys and compressed KV, plus BF16 comparison KV.

use kernel_build::CubinSpec;
use kernel_core::compressed_kv::{CompressedKvStorage, CompressedKvWriteQuery};
use kernel_core::pooled_indexer::{IndexerCandidates, PositionIndexerQuery};
use kernels_sm120::gated_pooling::key_write::{Entry, cubin_spec};

pub const FAMILY: &str = "fp4_key_write";
pub const CUBIN: &str = "fp4_key_write_v1";

pub fn entries() -> Vec<Entry> {
    let mut entries = Vec::new();
    for ratio in [1, 2] {
        entries.push(
            Entry::index(
                &PositionIndexerQuery {
                    heads: 32,
                    head_dim: 128,
                    compression_ratio: ratio,
                    tail_rope_dims: 64,
                    key_norm_eps_bits: 1e-20f32.to_bits(),
                    max_top_k: 512,
                    page_size: 64,
                    candidates: IndexerCandidates::All,
                    capacity: 12,
                },
                &format!("chop_fp4_index_r{ratio}_v1"),
            )
            .expect("lab index write"),
        );
        for (name, storage) in [
            ("packed", CompressedKvStorage::Packed),
            ("bf16", CompressedKvStorage::Bf16),
        ] {
            entries.push(
                Entry::compressed(
                    &CompressedKvWriteQuery {
                        width: 512,
                        tail_rope_dims: 64,
                        compression_ratio: ratio,
                        page_size: 64,
                        storage,
                        capacity: 12,
                    },
                    &format!("chop_fp4_compressed_{name}_r{ratio}_v1"),
                )
                .expect("lab compressed write"),
            );
        }
    }
    entries
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(
        &[CUBIN],
        vec![cubin_spec(CUBIN, &entries()).expect("lab FP4 key writes")],
    )
}
