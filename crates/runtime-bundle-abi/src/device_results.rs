//! Fixed-width records for device acceptance and bounded result collection.
//!
//! These records do not opt a bundle into asynchronous execution. A result
//! is usable only after its completion dependency and all sampler rows have
//! been validated. Request and slot membership are checked by the caller.

mod sink;
pub use sink::*;
mod cycle;
pub use cycle::*;
mod views;
pub use views::*;

use crate::BundleStatusV1;

pub const BUNDLE_ACCEPTANCE_INPUT_BYTES_V1: usize = 40;
pub const BUNDLE_ACCEPTANCE_RESULT_BYTES_V1: usize = 32;
pub const BUNDLE_ACCEPTANCE_NO_DRAFT_V1: u32 = u32::MAX;
pub const BUNDLE_ACCEPTANCE_NO_FAILURE_V1: u32 = u32::MAX;
pub const BUNDLE_ACCEPTANCE_TERMINAL_V1: u32 = 1;
pub const BUNDLE_ACCEPTANCE_REJECTED_V1: u32 = 2;

/// One item's device acceptance inputs. All offsets count elements.
/// `stop_offset` indexes terminal stop IDs, not the sampler's masked IDs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleAcceptanceInputV1 {
    pub cycle_epoch: u64,
    pub remaining_tokens: u64,
    pub first_sample_row: u32,
    pub draft_item_index: u32,
    pub draft_tokens: u32,
    pub stop_offset: u32,
    pub stop_count: u32,
    pub reserved: u32,
}

impl BundleAcceptanceInputV1 {
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, BundleStatusV1> {
        if bytes.len() != BUNDLE_ACCEPTANCE_INPUT_BYTES_V1 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("bounded word"));
        Ok(Self {
            cycle_epoch: u64::from_le_bytes(bytes[..8].try_into().expect("bounded epoch")),
            remaining_tokens: u64::from_le_bytes(bytes[8..16].try_into().expect("bounded budget")),
            first_sample_row: word(16),
            draft_item_index: word(20),
            draft_tokens: word(24),
            stop_offset: word(28),
            stop_count: word(32),
            reserved: word(36),
        })
    }

    /// Validate every range before a reducer can index its device inputs.
    pub fn validate(
        self,
        sample_rows: u32,
        draft_items: u32,
        draft_stride: u32,
        stop_ids: u32,
    ) -> Result<(), BundleStatusV1> {
        let draft_ok = if self.draft_tokens == 0 {
            self.draft_item_index == BUNDLE_ACCEPTANCE_NO_DRAFT_V1
        } else {
            self.draft_item_index < draft_items && self.draft_tokens <= draft_stride
        };
        if self.cycle_epoch == 0
            || self.remaining_tokens == 0
            || self.reserved != 0
            || !draft_ok
            || self
                .draft_tokens
                .checked_add(1)
                .and_then(|rows| self.first_sample_row.checked_add(rows))
                .is_none_or(|end| end > sample_rows)
            || self
                .stop_offset
                .checked_add(self.stop_count)
                .is_none_or(|end| end > stop_ids)
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(())
    }

    #[must_use]
    pub fn to_le_bytes(self) -> [u8; BUNDLE_ACCEPTANCE_INPUT_BYTES_V1] {
        let mut bytes = [0; BUNDLE_ACCEPTANCE_INPUT_BYTES_V1];
        bytes[..8].copy_from_slice(&self.cycle_epoch.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.remaining_tokens.to_le_bytes());
        for (target, word) in bytes[16..].chunks_exact_mut(4).zip([
            self.first_sample_row,
            self.draft_item_index,
            self.draft_tokens,
            self.stop_offset,
            self.stop_count,
            self.reserved,
        ]) {
            target.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// An acceptance reducer's record, distinct from the unchanged 256-byte
/// sampled-row records. `accepted_drafts` never includes the bonus row.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleAcceptanceResultV1 {
    pub cycle_epoch: u64,
    pub accepted_drafts: u32,
    pub emitted_rows: u32,
    pub flags: u32,
    pub status: u32,
    pub failure_row: u32,
    pub reserved: u32,
}

/// A checked result still needs full sampler validation and the model's
/// status check. An empty distribution authorizes no token publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceAcceptanceV1 {
    Prefix {
        accepted_drafts: u32,
        emitted_rows: u32,
        terminal: bool,
        rejected: bool,
    },
    EmptyDistribution {
        row: u32,
    },
}

impl BundleAcceptanceResultV1 {
    pub fn from_le_bytes(bytes: &[u8]) -> Result<Self, BundleStatusV1> {
        if bytes.len() != BUNDLE_ACCEPTANCE_RESULT_BYTES_V1 {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let word = |at| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("bounded word"));
        Ok(Self {
            cycle_epoch: u64::from_le_bytes(bytes[..8].try_into().expect("bounded epoch")),
            accepted_drafts: word(8),
            emitted_rows: word(12),
            flags: word(16),
            status: word(20),
            failure_row: word(24),
            reserved: word(28),
        })
    }

    #[must_use]
    pub fn to_le_bytes(self) -> [u8; BUNDLE_ACCEPTANCE_RESULT_BYTES_V1] {
        let mut bytes = [0; BUNDLE_ACCEPTANCE_RESULT_BYTES_V1];
        bytes[..8].copy_from_slice(&self.cycle_epoch.to_le_bytes());
        for (target, word) in bytes[8..].chunks_exact_mut(4).zip([
            self.accepted_drafts,
            self.emitted_rows,
            self.flags,
            self.status,
            self.failure_row,
            self.reserved,
        ]) {
            target.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// Check identity, bounds and the count relationships of sequential
    /// acceptance. This does not replace validation of every sampled row.
    pub fn validate(
        self,
        input: BundleAcceptanceInputV1,
    ) -> Result<DeviceAcceptanceV1, BundleStatusV1> {
        if input.cycle_epoch == 0 || self.cycle_epoch != input.cycle_epoch {
            return Err(BundleStatusV1::StaleHandle);
        }
        if self.reserved != 0
            || self.flags & !(BUNDLE_ACCEPTANCE_TERMINAL_V1 | BUNDLE_ACCEPTANCE_REJECTED_V1) != 0
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if self.status == 2 {
            return if self.failure_row <= input.draft_tokens {
                Ok(DeviceAcceptanceV1::EmptyDistribution {
                    row: self.failure_row,
                })
            } else {
                Err(BundleStatusV1::InvalidArgument)
            };
        }
        if self.status != 0 {
            return Err(BundleStatusV1::HostError);
        }
        let terminal = self.flags & BUNDLE_ACCEPTANCE_TERMINAL_V1 != 0;
        let rejected = self.flags & BUNDLE_ACCEPTANCE_REJECTED_V1 != 0;
        let last_matched = self.accepted_drafts == self.emitted_rows;
        let last_unmatched = self.accepted_drafts.checked_add(1) == Some(self.emitted_rows);
        if self.failure_row != BUNDLE_ACCEPTANCE_NO_FAILURE_V1
            || self.emitted_rows == 0
            || u64::from(self.emitted_rows) > input.remaining_tokens
            || self.accepted_drafts > input.draft_tokens
            || (!last_matched && !last_unmatched)
            || (last_matched
                && (rejected
                    || (!terminal && u64::from(self.emitted_rows) != input.remaining_tokens)))
            || (last_unmatched && rejected != (self.emitted_rows <= input.draft_tokens))
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(DeviceAcceptanceV1::Prefix {
            accepted_drafts: self.accepted_drafts,
            emitted_rows: self.emitted_rows,
            terminal,
            rejected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    fn input(k: u32, budget: u64) -> BundleAcceptanceInputV1 {
        BundleAcceptanceInputV1 {
            cycle_epoch: 19,
            remaining_tokens: budget,
            first_sample_row: 0,
            draft_item_index: if k == 0 {
                BUNDLE_ACCEPTANCE_NO_DRAFT_V1
            } else {
                0
            },
            draft_tokens: k,
            stop_offset: 0,
            stop_count: 1,
            reserved: 0,
        }
    }
    fn result(accepted: u32, emitted: u32, flags: u32) -> BundleAcceptanceResultV1 {
        BundleAcceptanceResultV1 {
            cycle_epoch: 19,
            accepted_drafts: accepted,
            emitted_rows: emitted,
            flags,
            status: 0,
            failure_row: BUNDLE_ACCEPTANCE_NO_FAILURE_V1,
            reserved: 0,
        }
    }

    #[test]
    fn wire_layout_and_endianness_are_fixed() {
        assert_eq!(size_of::<BundleAcceptanceInputV1>(), 40);
        assert_eq!(size_of::<BundleAcceptanceResultV1>(), 32);
        assert_eq!(offset_of!(BundleAcceptanceInputV1, first_sample_row), 16);
        assert_eq!(offset_of!(BundleAcceptanceInputV1, reserved), 36);
        assert_eq!(offset_of!(BundleAcceptanceResultV1, flags), 16);
        assert_eq!(offset_of!(BundleAcceptanceResultV1, failure_row), 24);
        let value = result(3, 4, BUNDLE_ACCEPTANCE_REJECTED_V1);
        assert_eq!(
            BundleAcceptanceResultV1::from_le_bytes(&value.to_le_bytes()).unwrap(),
            value
        );
        assert_eq!(&input(5, 6).to_le_bytes()[8..16], &6_u64.to_le_bytes());
        assert!(BundleAcceptanceResultV1::from_le_bytes(&[0; 31]).is_err());
        assert!(BundleAcceptanceResultV1::from_le_bytes(&[0; 33]).is_err());
    }

    #[test]
    fn validates_bonus_matched_stop_mismatching_stop_and_token_limit() {
        for (k, budget, accepted, emitted, flags) in [
            (0, 1, 0, 1, 0),
            (5, 6, 5, 6, 0),
            (5, 6, 1, 1, BUNDLE_ACCEPTANCE_TERMINAL_V1),
            (
                5,
                6,
                0,
                1,
                BUNDLE_ACCEPTANCE_TERMINAL_V1 | BUNDLE_ACCEPTANCE_REJECTED_V1,
            ),
            (5, 6, 0, 1, BUNDLE_ACCEPTANCE_REJECTED_V1),
            (5, 2, 2, 2, 0),
        ] {
            let input = input(k, budget);
            input.validate(k + 1, 1, 5, 1).unwrap();
            assert!(matches!(
                result(accepted, emitted, flags).validate(input),
                Ok(DeviceAcceptanceV1::Prefix { .. })
            ));
        }
    }

    #[test]
    fn refuses_stale_epochs_overflow_ranges_and_impossible_counts() {
        let original = input(5, 6);
        for changed in [
            BundleAcceptanceInputV1 {
                first_sample_row: u32::MAX,
                ..original
            },
            BundleAcceptanceInputV1 {
                draft_tokens: u32::MAX,
                ..original
            },
            BundleAcceptanceInputV1 {
                stop_offset: u32::MAX,
                ..original
            },
            BundleAcceptanceInputV1 {
                draft_item_index: 1,
                ..original
            },
            BundleAcceptanceInputV1 {
                remaining_tokens: 0,
                ..original
            },
        ] {
            assert!(changed.validate(6, 1, 5, 1).is_err());
        }
        assert_eq!(
            BundleAcceptanceResultV1 {
                cycle_epoch: 18,
                ..result(0, 1, 2)
            }
            .validate(original),
            Err(BundleStatusV1::StaleHandle)
        );
        for bad in [
            result(0, 0, 0),
            result(1, 3, 2),
            result(1, 1, 0),
            result(5, 6, 2),
            result(6, 6, 1),
            result(0, 1, 4),
        ] {
            assert!(bad.validate(original).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn empty_distribution_does_not_read_unspecified_counts() {
        let empty = BundleAcceptanceResultV1 {
            status: 2,
            failure_row: 2,
            accepted_drafts: u32::MAX,
            emitted_rows: u32::MAX,
            ..result(0, 1, 0)
        };
        assert_eq!(
            empty.validate(input(5, 6)),
            Ok(DeviceAcceptanceV1::EmptyDistribution { row: 2 })
        );
        for status in [1, 3, u32::MAX] {
            assert_eq!(
                BundleAcceptanceResultV1 { status, ..empty }.validate(input(5, 6)),
                Err(BundleStatusV1::HostError)
            );
        }
    }
}
