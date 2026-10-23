//! Bit-exact expansion of selected matrices in a `scale-row-offset-v1` bank.
//!
//! The oracle is [`super::scale_offset::decode_bank_matrix`]. Matrix
//! `expert * projections + projection` writes `rows * groups` consecutive
//! bytes at that matrix's position in the raw row-major bank. Every byte,
//! including 253 through 255, is preserved without interpreting its code.

use super::scale_offset::{TABLE_ALIGNMENT, fixed_bytes};
use crate::contract::{Refusal, Validate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaleExpandQuery {
    pub width: u32,
    pub rows: u32,
    pub groups: u32,
    pub projections: u32,
    pub experts: u32,
}

impl ScaleExpandQuery {
    /// Minimum encoded bank size, with no exceptions. Actual blocks and
    /// their sorted, unique exception indices must be validated at load.
    pub fn minimum_bank_bytes(&self) -> Result<u64, Refusal> {
        let fixed = fixed_bytes(self.rows.into(), self.groups.into(), self.width)
            .map_err(|_| Refusal("invalid scale expansion matrix shape or offset width"))?;
        if self.experts == 0 || self.projections == 0 {
            return Err(Refusal("scale expansion needs experts and projections"));
        }
        let matrices = u64::from(self.experts) * u64::from(self.projections);
        let bytes = matrices
            .checked_add(1)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_next_multiple_of(TABLE_ALIGNMENT))
            .and_then(|table| matrices.checked_mul(fixed)?.checked_add(table))
            .ok_or(Refusal("scale expansion bank size overflow"))?;
        if bytes > u64::from(u32::MAX) {
            return Err(Refusal("scale expansion encoded bank exceeds u32 offsets"));
        }
        Ok(bytes)
    }

    pub fn output_bytes(&self) -> Result<u64, Refusal> {
        self.validate()?;
        // A validated bank bounds the matrix count and each matrix's size.
        Ok(u64::from(self.experts)
            * u64::from(self.projections)
            * u64::from(self.rows)
            * u64::from(self.groups))
    }

    pub fn active_bytes(&self) -> Result<u64, Refusal> {
        self.validate()?;
        Ok((u64::from(self.experts) + 1) * 4)
    }
}

impl Validate for ScaleExpandQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.minimum_bank_bytes().map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> ScaleExpandQuery {
        ScaleExpandQuery {
            width: 1,
            rows: 65,
            groups: 18,
            projections: 2,
            experts: 7,
        }
    }

    #[test]
    fn scale_expand_sizes_follow_the_codec() {
        for width in [1, 2, 4] {
            let q = ScaleExpandQuery { width, ..query() };
            let block = fixed_bytes(65, 18, width).unwrap();
            let bank = super::super::scale_offset::bank_layout(&[block; 14]).unwrap();
            assert_eq!(q.minimum_bank_bytes().unwrap(), bank.bytes);
            assert_eq!(q.output_bytes().unwrap(), 14 * 65 * 18);
            assert_eq!(q.active_bytes().unwrap(), 32);
            let high = ScaleExpandQuery {
                width,
                rows: 64,
                groups: 1 << 18,
                projections: 1,
                experts: 257,
            };
            assert_eq!(high.output_bytes().unwrap(), 257 * (1_u64 << 24));
            assert!(high.minimum_bank_bytes().unwrap() <= u64::from(u32::MAX));
        }
    }

    #[test]
    fn scale_expand_rejects_unrepresentable_banks() {
        for q in [
            ScaleExpandQuery {
                width: 3,
                ..query()
            },
            ScaleExpandQuery { rows: 0, ..query() },
            ScaleExpandQuery {
                groups: 0,
                ..query()
            },
            ScaleExpandQuery {
                experts: 0,
                ..query()
            },
            ScaleExpandQuery {
                projections: 0,
                ..query()
            },
            ScaleExpandQuery {
                rows: 1 << 24,
                ..query()
            },
            ScaleExpandQuery {
                experts: u32::MAX,
                ..query()
            },
            ScaleExpandQuery {
                experts: u32::MAX,
                projections: u32::MAX,
                ..query()
            },
        ] {
            assert!(q.validate().is_err(), "{q:?}");
            assert!(q.output_bytes().is_err());
            assert!(q.active_bytes().is_err());
        }
        assert!(
            ScaleExpandQuery {
                width: 1,
                rows: 1,
                groups: 1 << 24,
                projections: 1,
                experts: 1,
            }
            .validate()
            .is_ok()
        );
    }
}
