//! Checked device spans for sampling and acceptance result hooks.
use crate::{
    BundleAbiError, BundleContextHandleV1, BundleDeviceBufferViewV1, BundleEventHandleV1,
    BundleLogitsViewV1, BundleStatusV1, validate_device_buffer_view_v1,
    validate_execute_logits_view_v1,
};

pub const DEVICE_RESULTS_CAPABILITY_V1: &str = "chop.device-results.v1";
/// Select device results by default for the declaring topology's preferred KV layout.
pub const DEVICE_RESULTS_DEFAULT_CAPABILITY_V1: &str = "chop.device-results-default.v1";

/// An explicit zero keeps the legacy path; bundles without a qualified
/// default still require an explicit one.
pub fn device_results_selected_v1(default: bool, value: Option<&std::ffi::OsStr>) -> bool {
    value.map_or(default, |value| value == "1")
}

#[cfg(test)]
mod default_tests {
    use super::device_results_selected_v1;
    use std::ffi::OsStr;

    #[test]
    fn explicit_zero_restores_legacy_selection_for_every_default() {
        for qualified in [false, true] {
            assert_eq!(device_results_selected_v1(qualified, None), qualified);
            assert!(!device_results_selected_v1(
                qualified,
                Some(OsStr::new("0"))
            ));
            assert!(device_results_selected_v1(qualified, Some(OsStr::new("1"))));
        }
    }
}
pub const DEVICE_RESULTS_HOST_VALUES_V1: u32 = 0;
pub const DEVICE_RESULTS_DEVICE_FEEDBACK_V1: u32 = 1;
pub const DEVICE_SAMPLE_RECORD_BYTES_V1: u64 = 256;

/// A borrowed span and the event that makes its contents readable. The
/// caller retains both until all device consumers and collection finish.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleReadyDeviceViewV1 {
    pub context: BundleContextHandleV1,
    pub rank: u32,
    pub reserved: u32,
    pub buffer: BundleDeviceBufferViewV1,
    pub ready: BundleEventHandleV1,
}

impl BundleReadyDeviceViewV1 {
    pub fn validate(self, minimum_bytes: u64) -> Result<(), BundleStatusV1> {
        validate_device_buffer_view_v1(self.buffer).map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !self.context.has_generation()
            || !self.buffer.buffer.has_generation()
            || !self.ready.has_generation()
            || self.reserved != 0
            || minimum_bytes == 0
            || self.buffer.length < minimum_bytes
            || self.buffer.offset.checked_add(self.buffer.length).is_none()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(())
    }
}

/// Device proposals for one scheduled item. Confidence is optional and
/// contains raw F32 bits; it is never used to decide target acceptance.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDeviceDraftsV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub cycle_epoch: u64,
    pub request_id: u64,
    pub slot_generation: u64,
    pub requested_tokens: u32,
    pub computed_tokens: u32,
    pub token_stride_bytes: u64,
    pub proposals: BundleReadyDeviceViewV1,
    /// Empty when absent; otherwise one raw F32 word per computed token.
    pub confidence: BundleDeviceBufferViewV1,
}

impl BundleDeviceDraftsV1 {
    pub fn validate(self) -> Result<(), BundleStatusV1> {
        if self.struct_size < std::mem::size_of::<Self>() as u32
            || self.flags != 0
            || self.cycle_epoch == 0
            || self.slot_generation == 0
            || self.requested_tokens == 0
            || self.requested_tokens > self.computed_tokens
            || self.computed_tokens > 1 << 20
            || self.token_stride_bytes < 4
            || self.token_stride_bytes % 4 != 0
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let bytes = u64::from(self.computed_tokens - 1)
            .checked_mul(self.token_stride_bytes)
            .and_then(|bytes| bytes.checked_add(4))
            .ok_or(BundleStatusV1::InvalidArgument)?;
        self.proposals.validate(bytes)?;
        if self.confidence.struct_size != std::mem::size_of::<BundleDeviceBufferViewV1>() as u32
            || self.confidence.reserved != 0
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if self.confidence.buffer.raw == 0 {
            if self.confidence.offset != 0 || self.confidence.length != 0 {
                return Err(BundleStatusV1::InvalidArgument);
            }
        } else if !self.confidence.buffer.has_generation()
            || self.confidence.length < u64::from(self.computed_tokens) * 4
            || self
                .confidence
                .offset
                .checked_add(self.confidence.length)
                .is_none()
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        Ok(())
    }
}

/// Logits with an explicit dependency, without a context-wide host fence.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleReadyLogitsViewV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub cycle_epoch: u64,
    pub logits: BundleLogitsViewV1,
    pub ready: BundleEventHandleV1,
}

impl BundleReadyLogitsViewV1 {
    pub fn validate(self) -> Result<(), BundleAbiError> {
        if self.struct_size != std::mem::size_of::<Self>() as u32
            || self.flags != 0
            || self.cycle_epoch == 0
            || !self.ready.has_generation()
        {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
        }
        validate_execute_logits_view_v1(self.logits)
    }
}

/// Submit the ordinary sampler into caller-owned storage, then record done.
/// The host validates all three contexts and retains the allocations/events.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDeviceSamplingRequestV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub logits: BundleReadyLogitsViewV1,
    pub output: BundleDeviceBufferViewV1,
    pub done: BundleEventHandleV1,
}

impl BundleDeviceSamplingRequestV1 {
    pub fn validate(self) -> Result<(), BundleAbiError> {
        self.logits.validate()?;
        validate_device_buffer_view_v1(self.output)?;
        if self.struct_size < std::mem::size_of::<Self>() as u32
            || self.reserved != 0
            || !self.output.buffer.has_generation()
            || self.output.offset % 8 != 0
            || !self.done.has_generation()
            || self.done == self.logits.ready
            || self.output.buffer == self.logits.logits.buffer.buffer
            || self.output.offset.checked_add(self.output.length).is_none()
            || self.output.length
                < u64::from(self.logits.logits.row_count) * DEVICE_SAMPLE_RECORD_BYTES_V1
        {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
        }
        Ok(())
    }
}

/// One final packet containing every acceptance record and every computed
/// sampler row. Each offset is relative to packet.buffer, not the allocation.
/// Unused sample suffixes remain present for full-view validity checks.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDeviceCompletionV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub cycle_epoch: u64,
    pub item_count: u32,
    pub sample_rows: u32,
    pub acceptance_offset: u64,
    pub samples_offset: u64,
    pub packet: BundleReadyDeviceViewV1,
}

impl BundleDeviceCompletionV1 {
    pub fn validate(self) -> Result<(), BundleStatusV1> {
        if self.struct_size < std::mem::size_of::<Self>() as u32
            || self.flags != 0
            || self.cycle_epoch == 0
            || self.item_count == 0
            || self.sample_rows == 0
            || self.item_count > self.sample_rows
            || self.acceptance_offset % 8 != 0
            || self.samples_offset % 8 != 0
        {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let acceptance_end = self
            .acceptance_offset
            .checked_add(u64::from(self.item_count) * 32)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let samples_end = self
            .samples_offset
            .checked_add(u64::from(self.sample_rows) * DEVICE_SAMPLE_RECORD_BYTES_V1)
            .ok_or(BundleStatusV1::InvalidArgument)?;
        if self.acceptance_offset < samples_end && self.samples_offset < acceptance_end {
            return Err(BundleStatusV1::InvalidArgument);
        }
        self.packet.validate(acceptance_end.max(samples_end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BundleDeviceBufferHandleV1;
    fn ready(bytes: u64) -> BundleReadyDeviceViewV1 {
        BundleReadyDeviceViewV1 {
            context: BundleContextHandleV1::from_parts(1, 1),
            rank: 0,
            reserved: 0,
            buffer: BundleDeviceBufferViewV1 {
                buffer: BundleDeviceBufferHandleV1::from_parts(1, 2),
                offset: 0,
                length: bytes,
                ..BundleDeviceBufferViewV1::empty()
            },
            ready: BundleEventHandleV1::from_parts(1, 3),
        }
    }
    #[test]
    fn packet_requires_complete_disjoint_bounded_sections() {
        for (items, rows) in [(1, 1), (8, 48), (32, 192)] {
            let header = u64::from(items) * 32;
            let packet = BundleDeviceCompletionV1 {
                struct_size: std::mem::size_of::<BundleDeviceCompletionV1>() as u32,
                flags: 0,
                cycle_epoch: 3,
                item_count: items,
                sample_rows: rows,
                acceptance_offset: 0,
                samples_offset: header,
                packet: ready(header + u64::from(rows) * 256),
            };
            packet.validate().unwrap();
            for bad in [
                BundleDeviceCompletionV1 {
                    samples_offset: header - 8,
                    ..packet
                },
                BundleDeviceCompletionV1 {
                    acceptance_offset: u64::MAX - 7,
                    ..packet
                },
                BundleDeviceCompletionV1 {
                    sample_rows: rows + 1,
                    ..packet
                },
                BundleDeviceCompletionV1 {
                    cycle_epoch: 0,
                    ..packet
                },
                BundleDeviceCompletionV1 {
                    struct_size: 4,
                    ..packet
                },
            ] {
                assert!(bad.validate().is_err());
            }
        }
    }
    #[test]
    fn proposals_require_checked_stride_and_complete_confidence() {
        let draft = BundleDeviceDraftsV1 {
            struct_size: std::mem::size_of::<BundleDeviceDraftsV1>() as u32,
            flags: 0,
            cycle_epoch: 2,
            request_id: 1,
            slot_generation: 3,
            requested_tokens: 3,
            computed_tokens: 5,
            token_stride_bytes: 4,
            proposals: ready(20),
            confidence: BundleDeviceBufferViewV1::empty(),
        };
        draft.validate().unwrap();
        assert!(
            BundleDeviceDraftsV1 {
                token_stride_bytes: u64::MAX - 3,
                ..draft
            }
            .validate()
            .is_err()
        );
        assert!(
            BundleDeviceDraftsV1 {
                computed_tokens: 6,
                ..draft
            }
            .validate()
            .is_err()
        );
        assert!(
            BundleDeviceDraftsV1 {
                slot_generation: 0,
                ..draft
            }
            .validate()
            .is_err()
        );
        assert!(
            BundleDeviceDraftsV1 {
                confidence: BundleDeviceBufferViewV1 {
                    length: 4,
                    ..BundleDeviceBufferViewV1::empty()
                },
                ..draft
            }
            .validate()
            .is_err()
        );
    }
}
