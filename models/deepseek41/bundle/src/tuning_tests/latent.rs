use super::dense_prefill::{assert_untuned, key};
use kernel_core::{contract::AttentionPhase, latent_attention::LatentAttentionQuery};
use model_deepseek41_contract::{geometry::FLASH, tuning::Family};
use std::collections::BTreeSet;

#[test]
fn pin_untuned_served_keys() {
    let mut domain = BTreeSet::new();
    for ranks in [1, 2, 4] {
        for source in crate::sm120::latent::entries(&FLASH, ranks).unwrap() {
            if source.name.ends_with("_draft_block") {
                continue;
            }
            let LatentAttentionQuery::Attention {
                phase, capacity, ..
            } = source.query
            else {
                unreachable!()
            };
            let phase = match phase {
                AttentionPhase::Decode => "decode",
                AttentionPhase::Verify => "verify",
                AttentionPhase::Prefill => "prefill",
            };
            for site in ["sources", "cache", "staging"] {
                domain.insert(key(
                    &format!("latent.{site}.{phase}"),
                    "fp8",
                    &[("ranks", ranks)],
                    capacity,
                ));
            }
        }
        domain.insert(key(
            "latent.query.decode",
            "f32",
            &[("ranks", ranks)],
            crate::sm120::latent_scheduled::QUERY_ROWS,
        ));
        for capacity in 32..=64 {
            domain.insert(key(
                "latent.query.verify",
                "f32",
                &[("ranks", ranks)],
                capacity,
            ));
        }
        domain.insert(key(
            "latent.paired.prefill",
            "fp8",
            &[("ranks", ranks)],
            8192,
        ));
    }
    assert_eq!(domain.len(), 132);
    assert_untuned(
        Family::Latent,
        domain,
        UNTUNED,
        &[
            ("independent_heads", &[]),
            ("selected_cache", &["head_group"]),
            ("staged", &["head_group", "packed", "stage_keys"]),
            ("parallel", &[]),
            ("paired", &[]),
        ],
    );
}

const UNTUNED: &[&str] = &[
    "latent.cache.decode fp8 ranks=1 buckets=32 rows=all",
    "latent.cache.decode fp8 ranks=2 buckets=32 rows=all",
    "latent.cache.decode fp8 ranks=4 buckets=32 rows=3,5..7,9..32",
    "latent.cache.prefill fp8 ranks=1 buckets=8192 rows=all",
    "latent.cache.prefill fp8 ranks=2 buckets=8192 rows=all",
    "latent.cache.prefill fp8 ranks=4 buckets=8192 rows=all",
    "latent.cache.verify fp8 ranks=1 buckets=192 rows=all",
    "latent.cache.verify fp8 ranks=2 buckets=192 rows=all",
    "latent.cache.verify fp8 ranks=4 buckets=192 rows=1..3,5,7..15,17..31,65..192",
    "latent.paired.prefill fp8 ranks=1 buckets=8192 rows=all",
    "latent.paired.prefill fp8 ranks=2 buckets=8192 rows=all",
    "latent.query.decode f32 ranks=1 buckets=48 rows=all",
    "latent.query.decode f32 ranks=2 buckets=48 rows=all",
    "latent.query.verify f32 ranks=1 buckets=32..64 rows=all",
    "latent.query.verify f32 ranks=2 buckets=32..64 rows=all",
    "latent.query.verify f32 ranks=4 buckets=32..48 rows=all",
    "latent.query.verify f32 ranks=4 buckets=49..64 rows=1..48",
    "latent.sources.decode fp8 ranks=1 buckets=32 rows=all",
    "latent.sources.decode fp8 ranks=2 buckets=32 rows=all",
    "latent.sources.prefill fp8 ranks=1 buckets=8192 rows=all",
    "latent.sources.prefill fp8 ranks=2 buckets=8192 rows=all",
    "latent.sources.prefill fp8 ranks=4 buckets=8192 rows=all",
    "latent.sources.verify fp8 ranks=1 buckets=192 rows=all",
    "latent.sources.verify fp8 ranks=2 buckets=192 rows=all",
    "latent.staging.decode fp8 ranks=1 buckets=32 rows=all",
    "latent.staging.decode fp8 ranks=2 buckets=32 rows=all",
    "latent.staging.decode fp8 ranks=4 buckets=32 rows=1..7,9..32",
    "latent.staging.prefill fp8 ranks=1 buckets=8192 rows=all",
    "latent.staging.prefill fp8 ranks=2 buckets=8192 rows=all",
    "latent.staging.prefill fp8 ranks=4 buckets=8192 rows=all",
    "latent.staging.verify fp8 ranks=1 buckets=192 rows=all",
    "latent.staging.verify fp8 ranks=2 buckets=192 rows=all",
    "latent.staging.verify fp8 ranks=4 buckets=192 rows=1..31,33..192",
];
