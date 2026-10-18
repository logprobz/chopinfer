//! Prepare the actual native attention catalog at every geometry and TP
//! shape, including staged DSpark blocks. No device is opened here.

#[path = "../../../bundle/src/sm120/latent.rs"]
#[allow(dead_code)]
mod sm120;

use kernel_core::backend::{Backend, Need};
use kernel_core::contract::{AttentionPhase, Validate};
use kernel_core::latent_attention::LatentAttentionQuery;
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::GEOMETRIES;
use model_deepseek41_contract::latent::{self, Keys};
use model_deepseek41_contract::serving::{DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS};

#[test]
fn kernel_latent_prepares_all_required_shapes() {
    let mut count = 0;
    assert!(!GEOMETRIES.is_empty());
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = sm120::register(kernels_sm120::backend::Sm120::new(), g, ranks).unwrap();
            let queries = latent::queries(g, ranks).unwrap();
            assert_eq!(
                queries.len(),
                7 + 3 * usize::from(g.shape.compress_ratios.contains(&2))
            );
            if g.shape.compress_ratios.contains(&2) {
                let q = latent::compressed_query(g, ranks, AttentionPhase::Decode, 2).unwrap();
                assert!(matches!(
                    q,
                    LatentAttentionQuery::Attention {
                        page_size: 32,
                        second_fp4_keys: true,
                        ..
                    }
                ));
                backend.latent_attention_sources(&q, Need::rows(1)).unwrap();
            }
            for (keys, query) in queries {
                query.validate().unwrap();
                let LatentAttentionQuery::Attention {
                    heads,
                    latent,
                    max_keys,
                    softmax_scale_bits,
                    capacity,
                    phase,
                    ..
                } = query
                else {
                    unreachable!()
                };
                assert_eq!(heads, g.shape.attention.heads / ranks);
                assert_eq!(latent, g.shape.attention.head_dim);
                assert_eq!(softmax_scale_bits, (latent as f32).powf(-0.5).to_bits());
                let (extra, rows) = if keys == Keys::DraftBlock {
                    (g.shape.dspark.block, DECODE_BATCH * g.shape.dspark.block)
                } else {
                    (
                        if keys == Keys::Compressed {
                            g.shape.indexer.top_k
                        } else {
                            0
                        },
                        match phase {
                            AttentionPhase::Prefill => PREFILL_TOKENS,
                            AttentionPhase::Decode => DECODE_BATCH,
                            AttentionPhase::Verify => DECODE_BATCH * MAX_VERIFY_ROWS,
                        },
                    )
                };
                assert_eq!(capacity, rows);
                assert_eq!(max_keys, g.shape.attention.window + extra);
                let prepared = backend
                    .latent_attention_sources(&query, Need::rows(rows))
                    .unwrap();
                assert_eq!(prepared.info().symbols.len(), 2);
                assert!(prepared.info().bytes_of("workspace").unwrap() > 0);
                count += 1;
            }
        }
    }
    assert_eq!(
        count,
        GEOMETRIES
            .iter()
            .map(|g| 21 + 9 * usize::from(g.shape.compress_ratios.contains(&2)))
            .sum::<usize>()
    );
}

#[test]
fn kernel_latent_bf16_prepares_all_required_shapes() {
    let mut prepared = 0;
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let backend = sm120::register(kernels_sm120::backend::Sm120::new(), g, ranks).unwrap();
            for (_, query) in latent::queries(g, ranks).unwrap() {
                let query =
                    latent::with_storage(query, model_deepseek41_contract::state::KvStorage::Bf16)
                        .unwrap();
                assert!(matches!(
                    query,
                    LatentAttentionQuery::Attention {
                        kv: kernel_core::latent_attention::LatentKvFormat::Bf16,
                        ..
                    }
                ));
                backend
                    .latent_attention_sources(&query, Need::rows(4))
                    .unwrap();
                prepared += 1;
            }
        }
    }
    assert_eq!(
        prepared,
        GEOMETRIES
            .iter()
            .map(|g| 21 + 9 * usize::from(g.shape.compress_ratios.contains(&2)))
            .sum::<usize>()
    );
}
