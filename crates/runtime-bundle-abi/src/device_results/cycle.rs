//! Engine-owned cycle metadata lent to an opted-in bundle during execute.
use super::BundleAcceptanceInputV1;
use crate::{BundleByteViewV1, BundleStatusV1, ResultSinkLimitsV1};
use std::collections::BTreeSet;

pub const DEVICE_CYCLE_ITEM_BYTES_V1: usize = 56;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceCycleItemV1 {
    pub request_id: u64,
    pub slot_generation: u64,
    pub acceptance: BundleAcceptanceInputV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeviceCycleMetadataV1 {
    pub cycle_epoch: u64,
    pub items: Vec<DeviceCycleItemV1>,
    pub termination_stop_ids: Vec<u32>,
}

impl DeviceCycleMetadataV1 {
    pub fn validate(&self, limits: ResultSinkLimitsV1) -> Result<(), BundleStatusV1> {
        if self.cycle_epoch == 0
            || self.items.is_empty()
            || self.items.len() as u64 > limits.maximum_rows
            || self.items.len() > 1 << 20
            || self.termination_stop_ids.len() > 1 << 20
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut rows = 0_u32;
        let mut ids = BTreeSet::new();
        let mut draft_indices = BTreeSet::new();
        for item in &self.items {
            if !ids.insert(item.request_id)
                || item.slot_generation == 0
                || item.acceptance.cycle_epoch != self.cycle_epoch
                || item.acceptance.first_sample_row != rows
                || (item.acceptance.draft_tokens != 0
                    && !draft_indices.insert(item.acceptance.draft_item_index))
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
            rows = item
                .acceptance
                .draft_tokens
                .checked_add(1)
                .and_then(|n| rows.checked_add(n))
                .ok_or(BundleStatusV1::InvalidArgument)?;
        }
        if u64::from(rows) > limits.maximum_rows {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let draft_items = self
            .items
            .iter()
            .filter(|item| item.acceptance.draft_tokens != 0)
            .count() as u32;
        let draft_stride = self
            .items
            .iter()
            .map(|item| item.acceptance.draft_tokens)
            .max()
            .unwrap_or(0);
        for item in &self.items {
            item.acceptance.validate(
                rows,
                draft_items,
                draft_stride,
                self.termination_stop_ids.len() as u32,
            )?;
            let start = item.acceptance.stop_offset as usize;
            let end = start + item.acceptance.stop_count as usize;
            let stops = &self.termination_stop_ids[start..end];
            if stops.windows(2).any(|pair| pair[0] >= pair[1])
                || stops
                    .iter()
                    .any(|id| u64::from(*id) >= limits.maximum_logits_per_row)
            {
                return Err(BundleStatusV1::InvalidArgument);
            }
        }
        Ok(())
    }

    pub(crate) fn encode_items(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.items.len() * DEVICE_CYCLE_ITEM_BYTES_V1);
        for item in &self.items {
            bytes.extend_from_slice(&item.request_id.to_le_bytes());
            bytes.extend_from_slice(&item.slot_generation.to_le_bytes());
            bytes.extend_from_slice(&item.acceptance.to_le_bytes());
        }
        bytes
    }

    pub(crate) fn decode(
        epoch: u64,
        items: &[u8],
        stops: &[u8],
        limits: ResultSinkLimitsV1,
    ) -> Result<Self, BundleStatusV1> {
        if items.len() % DEVICE_CYCLE_ITEM_BYTES_V1 != 0
            || stops.len() % 4 != 0
            || items.len() / DEVICE_CYCLE_ITEM_BYTES_V1 > (1 << 20)
            || items.len() as u64 / DEVICE_CYCLE_ITEM_BYTES_V1 as u64 > limits.maximum_rows
            || stops.len() > (1 << 20) * 4
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let mut decoded = Vec::with_capacity(items.len() / DEVICE_CYCLE_ITEM_BYTES_V1);
        for row in items.chunks_exact(DEVICE_CYCLE_ITEM_BYTES_V1) {
            decoded.push(DeviceCycleItemV1 {
                request_id: u64::from_le_bytes(row[..8].try_into().expect("bounded ID")),
                slot_generation: u64::from_le_bytes(
                    row[8..16].try_into().expect("bounded generation"),
                ),
                acceptance: BundleAcceptanceInputV1::from_le_bytes(&row[16..])?,
            });
        }
        let result = Self {
            cycle_epoch: epoch,
            items: decoded,
            termination_stop_ids: stops
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("bounded stop")))
                .collect(),
        };
        result.validate(limits)?;
        Ok(result)
    }
}

/// The pointed-to byte arrays remain immutable until the enclosing execute
/// returns. The bundle-side wrapper copies them before exposing safe records.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDeviceCycleViewV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub cycle_epoch: u64,
    /// 56-byte LE records: request ID, slot generation, 40-byte acceptance input.
    pub items: BundleByteViewV1,
    pub termination_stop_ids: BundleByteViewV1,
}

impl BundleDeviceCycleViewV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            reserved: 0,
            cycle_epoch: 0,
            items: BundleByteViewV1::empty(),
            termination_stop_ids: BundleByteViewV1::empty(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata() -> DeviceCycleMetadataV1 {
        DeviceCycleMetadataV1 {
            cycle_epoch: 7,
            items: (0..2)
                .map(|index| DeviceCycleItemV1 {
                    request_id: index as u64,
                    slot_generation: 1,
                    acceptance: BundleAcceptanceInputV1 {
                        cycle_epoch: 7,
                        remaining_tokens: 8,
                        first_sample_row: index * 3,
                        draft_item_index: index,
                        draft_tokens: 2,
                        stop_offset: 0,
                        stop_count: 2,
                        reserved: 0,
                    },
                })
                .collect(),
            termination_stop_ids: vec![1, 2],
        }
    }
    #[test]
    fn metadata_round_trip_and_reused_or_stale_membership() {
        let limits = ResultSinkLimitsV1::new(6, 8, 48);
        let valid = metadata();
        let stops = [1_u32.to_le_bytes(), 2_u32.to_le_bytes()].concat();
        assert_eq!(
            DeviceCycleMetadataV1::decode(7, &valid.encode_items(), &stops, limits).unwrap(),
            valid
        );
        let mut bad = valid.clone();
        bad.items[1].acceptance.draft_item_index = 0;
        assert!(bad.validate(limits).is_err());
        bad = valid.clone();
        bad.items[1].request_id = 0;
        assert!(bad.validate(limits).is_err());
        bad = valid.clone();
        bad.items[0].slot_generation = 0;
        assert!(bad.validate(limits).is_err());
        bad = valid.clone();
        bad.items[0].acceptance.cycle_epoch = 6;
        assert!(bad.validate(limits).is_err());
        bad = valid.clone();
        bad.items[1].acceptance.first_sample_row = 2;
        assert!(bad.validate(limits).is_err());
        bad = valid.clone();
        bad.termination_stop_ids.reverse();
        assert!(bad.validate(limits).is_err());
        assert!(
            DeviceCycleMetadataV1::decode(7, &valid.encode_items()[..100], &stops, limits).is_err()
        );
        assert!(
            DeviceCycleMetadataV1::decode(
                7,
                &valid.encode_items(),
                &stops,
                ResultSinkLimitsV1::new(1, 8, 8)
            )
            .is_err()
        );
    }
}
