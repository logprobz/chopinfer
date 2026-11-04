use super::dense_prefill::{assert_untuned, key};
use model_deepseek41_contract::tuning::Family;
use std::collections::BTreeSet;

#[test]
fn pin_untuned_served_keys() {
    let mut domain = BTreeSet::new();
    for ranks in [1, 2, 4] {
        domain.insert(key(
            "indexer.hierarchy.decode",
            "fp4",
            &[("ranks", ranks)],
            32,
        ));
        // 32 to 64 rows, and the widest wide verify bucket.
        for capacity in (32..=64).chain([192]) {
            domain.insert(key(
                "indexer.query.verify",
                "f32",
                &[("ranks", ranks)],
                capacity,
            ));
        }
    }
    for (site, capacity) in [
        ("candidate_scores", 512),
        ("compressed_scores", 4096),
        ("long_candidate_scores", 512),
        ("long_compressed_scores", 4096),
    ] {
        domain.insert(key(
            &format!("indexer.{site}.prefill"),
            "fp4",
            &[("top_k", 512)],
            capacity,
        ));
    }
    for width in [65536, 131072] {
        for capacity in 1..=4096 {
            domain.insert(key(
                "indexer.cached.prefill",
                "f32",
                &[("width", width), ("top_k", 512)],
                capacity,
            ));
        }
    }
    assert_eq!(domain.len(), 8301);
    assert_untuned(
        Family::Indexer,
        domain,
        UNTUNED,
        &[
            ("hierarchical", &[]),
            ("parallel", &[]),
            ("cached", &[]),
            ("tiled", &[]),
            ("tile4", &["ctas"]),
        ],
    );
}

const UNTUNED: &[&str] = &[
    "indexer.cached.prefill f32 top_k=512,width=131072 buckets=1..31 rows=all",
    "indexer.cached.prefill f32 top_k=512,width=131072 buckets=32..4096 rows=1..31",
    "indexer.cached.prefill f32 top_k=512,width=65536 buckets=1..31 rows=all",
    "indexer.cached.prefill f32 top_k=512,width=65536 buckets=32..4096 rows=1..31",
    "indexer.candidate_scores.prefill fp4 top_k=512 buckets=512 rows=1..511",
    "indexer.compressed_scores.prefill fp4 top_k=512 buckets=4096 rows=1..4095",
    "indexer.long_candidate_scores.prefill fp4 top_k=512 buckets=512 rows=1..31",
    "indexer.long_compressed_scores.prefill fp4 top_k=512 buckets=4096 rows=1..31,4033..4096",
    "indexer.query.verify f32 ranks=1 buckets=32..64,192 rows=all",
    "indexer.query.verify f32 ranks=2 buckets=32..64,192 rows=all",
    "indexer.query.verify f32 ranks=4 buckets=192 rows=1..64",
    "indexer.query.verify f32 ranks=4 buckets=32 rows=all",
    "indexer.query.verify f32 ranks=4 buckets=33..64 rows=1..32",
];
