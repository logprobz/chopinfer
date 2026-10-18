//! Tail RoPE and trained FP4 storage of normalized compressed KV rows.
//!
//! Input is the F32 normalized latent before RoPE. Rotate adjacent pairs
//! of the last `tail_rope_dims` channels using an F32 cos/sin table at
//! compressed position times `compression_ratio`. Round the result to BF16
//! before quantization, including before amax. Each 16-channel block uses
//! `s = E4M3(max(amax, 6 * 2^-9) / 6)` with F32 division and RNE, then
//! `q = E2M1(clamp(x / s, -6, 6))` with F32 division and RNE. Scale zero is
//! impossible, including for an all-zero input block. There is no tensor
//! global scale. Both storage forms hold the same decoded values.
//!
//! The caller runs every consumer of the unrotated latent before this op.
//! Negative compressed positions and negative page IDs are holes. Distinct
//! live rows address distinct cache slots, and page-table entries are below
//! the declared page count. Republish on incomplete groups is model state,
//! not a side effect of a write that emits no row.

use crate::attention::BlockTables;
use crate::contract::{NumericalContract, NumericalContractId, Refusal, Validate};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CompressedKvStorage {
    /// Packed low-nibble-first E2M1 codes and a separate E4M3 scale plane.
    Packed,
    /// The same E2M1 times E4M3 values stored as BF16, not unquantized KV.
    Bf16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct CompressedKvWriteQuery {
    pub width: u32,
    pub tail_rope_dims: u32,
    pub compression_ratio: u32,
    /// Compressed positions per page.
    pub page_size: u32,
    pub storage: CompressedKvStorage,
    pub capacity: u32,
}

impl CompressedKvWriteQuery {
    /// Minimum code/value and scale bytes per page, checked for overflow.
    pub fn page_bytes(&self) -> Result<(u64, u64), Refusal> {
        let elements = u64::from(self.width) * u64::from(self.page_size);
        match self.storage {
            CompressedKvStorage::Packed => Ok((elements / 2, elements / 16)),
            CompressedKvStorage::Bf16 => Ok((
                elements
                    .checked_mul(2)
                    .ok_or(Refusal("compressed BF16 page size overflows"))?,
                0,
            )),
        }
    }
}

impl Validate for CompressedKvWriteQuery {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0
            || !self.width.is_multiple_of(16)
            || self.tail_rope_dims == 0
            || !self.tail_rope_dims.is_multiple_of(2)
            || self.tail_rope_dims > self.width
            || self.compression_ratio == 0
            || self.page_size == 0
            || self.capacity == 0
        {
            return Err(Refusal("invalid compressed KV write geometry"));
        }
        self.page_bytes()?;
        Ok(())
    }
}

impl NumericalContract for CompressedKvWriteQuery {
    fn numerical_contract_id(&self) -> Result<NumericalContractId, Refusal> {
        self.validate()?;
        Ok(NumericalContractId::new(
            "kv.f32-tail-rope-bf16-input-e2m1-e4m3-block16.v1".into(),
        ))
    }
}

crate::scratch_record! {
    /// Codes or BF16 values in `values`, and packed storage's E4M3 scales.
    /// Byte strides address separate pages and may include padding. BF16
    /// storage ignores `scales` and requires no scale allocation.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CompressedKvPages {
        pub values: value,
        pub scales: option,
        pub pages: scalar(u32),
        pub value_page_stride: scalar(u64),
        pub scale_page_stride: scalar(u64),
        pub page_tables: record(BlockTables),
    }
}

crate::scratch_record! {
    /// F32 normalized `latent[rows][width]`; I32 compressed positions and
    /// item IDs; F32 `cos_sin[tokens][tail_rope_dims / 2][2]`. The latent is
    /// read-only, so index-key consumers retain its unrotated form.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct CompressedKvWriteBuffers {
        /// Sticky shared quantizer flags, one U32 per launch row. The caller
        /// clears and checks this plane. A valid addressed row with any flag
        /// writes zero codes and finite scale 1. Invalid positions, items,
        /// table rows and pages are holes: cache and status stay untouched.
        pub status: value,
        pub status_row_offset: scalar(u64),
        pub latent: value,
        pub cos_sin: value,
        pub positions: value,
        pub row_batch: value,
        pub live_rows: value,
        pub cache: record(CompressedKvPages),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(storage: CompressedKvStorage) -> CompressedKvWriteQuery {
        CompressedKvWriteQuery {
            width: 512,
            tail_rope_dims: 64,
            compression_ratio: 2,
            page_size: 64,
            storage,
            capacity: 16,
        }
    }

    #[test]
    fn packed_and_comparison_storage_keep_one_quantized_numerical_contract() {
        let packed = query(CompressedKvStorage::Packed);
        let bf16 = query(CompressedKvStorage::Bf16);
        assert_eq!(packed.page_bytes().unwrap(), (16384, 2048));
        assert_eq!(bf16.page_bytes().unwrap(), (65536, 0));
        assert_eq!(
            packed.numerical_contract_id().unwrap(),
            bf16.numerical_contract_id().unwrap()
        );
    }

    #[test]
    fn refuses_invalid_geometry_and_overflowing_comparison_pages() {
        let q = query(CompressedKvStorage::Packed);
        for bad in [
            CompressedKvWriteQuery { width: 0, ..q },
            CompressedKvWriteQuery {
                tail_rope_dims: 63,
                ..q
            },
            CompressedKvWriteQuery {
                compression_ratio: 0,
                ..q
            },
            CompressedKvWriteQuery { page_size: 0, ..q },
            CompressedKvWriteQuery { capacity: 0, ..q },
            CompressedKvWriteQuery {
                width: u32::MAX - 15,
                page_size: u32::MAX,
                storage: CompressedKvStorage::Bf16,
                ..q
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
