//! Visibility of a latent window ring followed by a bidirectional block.
//!
//! These are physical row indices into `[main ring | block]`, not absolute
//! token positions for a paged cache. Every query in the block sees the same
//! keys. The ring contains only committed main rows, projected before the
//! block runs; provisional block rows occupy separate storage.

use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};

/// A main-KV window plus every position of the current block.
///
/// Main row `p` is stored at `p % window_size`. Before the ring fills, only
/// its initialized prefix is visible. Once full, all physical slots are
/// visible in slot order, including after wrap. Block keys always start at
/// `window_size`, even while part of the ring is uninitialized.
///
/// A scheduler may emit a prefix of the proposals, but must still compute
/// all `block_size` positions: later block keys affect earlier queries.
/// This contract has no causal mask or per-query sliding window.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WindowBlockVisibility {
    /// Capacity of the committed main-KV ring, in rows.
    pub window_size: u32,
    /// Full block width, independent of the emitted proposal count.
    pub block_size: u32,
}

impl Validate for WindowBlockVisibility {
    fn validate(&self) -> Result<(), Refusal> {
        if self.window_size == 0 || self.block_size == 0 {
            return Err(Refusal("window and block sizes must be nonzero"));
        }
        if u64::from(self.window_size) + u64::from(self.block_size) > i32::MAX as u64 {
            return Err(Refusal("window plus block key-list width must fit I32"));
        }
        Ok(())
    }
}

impl NumericalContract for WindowBlockVisibility {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(
            "attention.visibility.window-block-bidirectional.v1".to_owned(),
        ))
    }
}

impl WindowBlockVisibility {
    /// Exact key indices for `query`, a position within the full block.
    ///
    /// `main_rows` counts the contiguous main rows projected into the ring,
    /// including the latest committed row. After projecting row `p`, pass
    /// `p + 1`, not `p`. At least one main row must be present. Catch-up may
    /// advance this count by several rows between blocks.
    ///
    /// The list is `0..min(window_size, main_rows)` followed by
    /// `window_size..window_size + block_size`, with no duplicates or holes.
    /// Its order is part of the contract, including after ring wrap. The
    /// caller uses the same list for every block query of this item; batch
    /// items use their own `main_rows` counts. No cache is read or written.
    pub fn key_indices(
        self,
        main_rows: u64,
        query: u32,
    ) -> Result<impl Iterator<Item = i32> + Clone, Refusal> {
        self.validate()?;
        if main_rows == 0 {
            return Err(Refusal("block visibility needs a projected main row"));
        }
        if query >= self.block_size {
            return Err(Refusal("block query is outside the full block"));
        }
        let main_end = main_rows.min(u64::from(self.window_size)) as i32;
        let block_start = self.window_size as i32;
        let block_end = (self.window_size + self.block_size) as i32;
        Ok((0..main_end).chain(block_start..block_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_id_and_refusals() {
        let visibility = WindowBlockVisibility {
            window_size: 8,
            block_size: 5,
        };
        assert_eq!(
            visibility.numerical_contract_id().unwrap().as_str(),
            "attention.visibility.window-block-bidirectional.v1"
        );
        assert!(visibility.key_indices(0, 0).is_err());
        assert!(visibility.key_indices(1, 5).is_err());
        assert!(visibility.key_indices(1, u32::MAX).is_err());
        for (window_size, block_size, message) in [
            (0, 5, "window and block sizes must be nonzero"),
            (8, 0, "window and block sizes must be nonzero"),
            (
                i32::MAX as u32,
                1,
                "window plus block key-list width must fit I32",
            ),
            (
                u32::MAX,
                u32::MAX,
                "window plus block key-list width must fit I32",
            ),
        ] {
            let invalid = WindowBlockVisibility {
                window_size,
                block_size,
            };
            assert_eq!(invalid.validate(), Err(Refusal(message)));
            assert_eq!(invalid.numerical_contract_id(), Err(Refusal(message)));
            assert_eq!(invalid.key_indices(1, 0).err(), Some(Refusal(message)));
        }
    }

    #[test]
    fn large_counts_do_not_wrap_or_allocate() {
        let visibility = WindowBlockVisibility {
            window_size: i32::MAX as u32 - 5,
            block_size: 5,
        };
        visibility.validate().unwrap();
        let mut keys = visibility.key_indices(u64::MAX, 4).unwrap();
        assert_eq!(keys.next(), Some(0));
        assert_eq!(keys.last(), Some(i32::MAX - 1));
    }
}
