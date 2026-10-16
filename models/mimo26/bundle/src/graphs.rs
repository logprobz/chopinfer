//! MiMo's graph buckets over the shared graph cache.

pub use bundle_runtime::graphs::{GraphKey, pieces};

use crate::contract::Mimo;

pub type GraphCache = bundle_runtime::graphs::GraphCache<Mimo>;

pub fn decode_bucket(live: u32) -> Option<u32> {
    bundle_runtime::graphs::decode_bucket::<Mimo>(live)
}

pub fn verify_bucket(live: u32) -> Option<u32> {
    bundle_runtime::graphs::verify_bucket::<Mimo>(live)
}

pub use model_mimo26_contract::graphs::{buckets_through, decode_buckets};

#[test]
fn sdk_decode_buckets_match_the_bundle() {
    assert_eq!(
        model_compiler_sdk::package_draft::decode_buckets(crate::contract::DECODE_BATCH)
            .collect::<Vec<_>>(),
        crate::graphs::decode_buckets().collect::<Vec<_>>()
    );
}
