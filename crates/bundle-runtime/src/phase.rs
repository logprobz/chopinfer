//! The phase a step's program runs in, and the token capacity its stages
//! are prepared for.
//!
//! A decode phase runs one token per item over a graph bucket of items, a
//! verify phase several rows per item over a bucket of tokens, both
//! replayed for every live count in their bucket; a prefill phase runs
//! eagerly. Which kernel entry serves a stage in a phase, and how a verify
//! row's attention runs, stay with the bundle.

use crate::geometry::BundleGeometry;
use crate::graphs::{decode_bucket, verify_bucket};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    /// One token per item, over this many items.
    Decode(u32),
    /// Several rows per item, over this many tokens in all.
    Verify(u32),
    Prefill,
}

impl Phase {
    /// The prepared token capacity of token-indexed stages at `G`'s package
    /// limits: the decode or verify graph bucket, or the prefill capacity.
    /// None for a live count no bucket holds.
    pub fn capacity_in<G: BundleGeometry>(self) -> Option<u32> {
        match self {
            Self::Decode(live) => decode_bucket::<G>(live),
            Self::Verify(live) => verify_bucket::<G>(live),
            Self::Prefill => Some(G::PREFILL_TOKENS),
        }
    }

    /// Whether token-indexed stages run graph buckets.
    pub const fn token_bucketed(self) -> bool {
        matches!(self, Self::Decode(_) | Self::Verify(_))
    }

    /// Whether token-indexed stages run the prefill kernels: prefill, and
    /// graph buckets above the `decode_rows` the decode kernels hold. Their
    /// GEMMs take the bucket's rows, so a graph still replays for every
    /// live count in its bucket.
    pub fn prefill_kernels_in<G: BundleGeometry>(self, decode_rows: u32) -> bool {
        match self {
            Self::Decode(_) | Self::Verify(_) => self
                .capacity_in::<G>()
                .is_some_and(|capacity| capacity > decode_rows),
            Self::Prefill => true,
        }
    }

    /// The grid rows of a row kernel over `live` rows. Eager prefill
    /// launches the `smallest` row bucket that holds them, within the
    /// prefill capacity: row kernels return for rows at or past the live
    /// count, so the empty CTAs of the capacity grid need not launch.
    /// Graph phases keep their bucket, so a replay covers every live count
    /// in it.
    pub fn row_grid_in<G: BundleGeometry>(
        self,
        live: u32,
        smallest: impl FnOnce(u32) -> Option<u32>,
    ) -> Option<u32> {
        match self {
            Self::Prefill => smallest(live).filter(|&rows| rows <= G::PREFILL_TOKENS),
            _ => self.capacity_in::<G>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Phase;
    use crate::geometry::tests::Tp4;

    #[test]
    fn graph_phases_prepare_their_bucket_and_prefill_its_capacity() {
        assert_eq!(Phase::Decode(3).capacity_in::<Tp4>(), Some(4));
        assert_eq!(Phase::Decode(33).capacity_in::<Tp4>(), None);
        assert_eq!(Phase::Verify(20).capacity_in::<Tp4>(), Some(32));
        assert_eq!(Phase::Prefill.capacity_in::<Tp4>(), Some(8192));
        assert!(!Phase::Decode(32).prefill_kernels_in::<Tp4>(32));
        assert!(Phase::Decode(32).prefill_kernels_in::<Tp4>(16));
        assert!(Phase::Prefill.prefill_kernels_in::<Tp4>(32));
        let buckets = |live: u32| Some(live.next_power_of_two().max(64));
        assert_eq!(Phase::Prefill.row_grid_in::<Tp4>(100, buckets), Some(128));
        assert_eq!(Phase::Prefill.row_grid_in::<Tp4>(9000, buckets), None);
        assert_eq!(Phase::Decode(5).row_grid_in::<Tp4>(5, buckets), Some(8));
    }
}
