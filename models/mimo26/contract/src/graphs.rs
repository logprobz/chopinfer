//! Package graph buckets.
use crate::serving::Mimo;

/// Every decode graph bucket of the package maximum, smallest first.
pub fn decode_buckets() -> impl Iterator<Item = u32> {
    bundle_runtime::geometry::Items::package::<Mimo>().buckets()
}

/// Every decode graph bucket of an instance serving `sequences`, smallest
/// first: a package declares these for its maximum.
pub fn buckets_through(sequences: u32) -> impl Iterator<Item = u32> {
    bundle_runtime::geometry::Items::new(sequences)
        .expect("a served count from 1 to MAX_ITEMS")
        .buckets()
}
