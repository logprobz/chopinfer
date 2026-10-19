//! Copy committed ring slots into per-item draft storage without changing bytes.

use kernel_core::backend::Need;
use kernel_core::contract::Refusal;

pub const CONTRACT: &str = "deepseek41.dspark.ring-gather.bytes.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RingGather {
    pub bytes: u32,
    pub source_stride: u32,
    pub destination_stride: u32,
    pub slots: u32,
}

impl RingGather {
    pub fn validate(self, need: Need) -> Result<(), Refusal> {
        if self.bytes == 0
            || self.slots == 0
            || need.capacity == 0
            || self.source_stride < self.bytes
            || self.destination_stride < self.bytes
        {
            return Err(Refusal(
                "ring gather has invalid widths, strides or capacity",
            ));
        }
        Ok(())
    }

    pub fn source_bytes(self) -> u64 {
        u64::from(self.slots - 1) * u64::from(self.source_stride) + u64::from(self.bytes)
    }

    pub fn destination_bytes(self, items: u32) -> u64 {
        u64::from(items - 1) * u64::from(self.destination_stride) + u64::from(self.bytes)
    }
}

kernel_core::scratch_record! {
    /// For item i below live_items, copy bytes from source_slots[i] to i.
    /// Out-of-range source slots, inactive items and stride gaps are untouched.
    /// Source, destination and metadata must occupy disjoint byte ranges.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RingGatherBuffers {
        pub source: value,
        pub source_slots: value,
        pub live_items: value,
        pub destination: value,
    }
}

pub fn oracle(
    op: RingGather,
    source: &[u8],
    slots: &[u32],
    live: u32,
    destination: &mut [u8],
) -> Result<(), Refusal> {
    op.validate(Need::rows(live.max(1)))?;
    if live == 0 {
        return Ok(());
    }
    if (source.len() as u64) < op.source_bytes()
        || slots.len() < live as usize
        || (destination.len() as u64) < op.destination_bytes(live)
    {
        return Err(Refusal("ring gather oracle has a short buffer"));
    }
    for (item, &slot) in slots[..live as usize].iter().enumerate() {
        if slot >= op.slots {
            continue;
        }
        let from = slot as usize * op.source_stride as usize;
        let to = item * op.destination_stride as usize;
        destination[to..to + op.bytes as usize]
            .copy_from_slice(&source[from..from + op.bytes as usize]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reordered_slots_preserve_tail_holes_and_inactive_items() {
        let op = RingGather {
            bytes: 8,
            source_stride: 12,
            destination_stride: 16,
            slots: 4,
        };
        let source: Vec<_> = (0..48).collect();
        let mut destination = vec![0xa5; 80];
        oracle(op, &source, &[3, 1, u32::MAX, 4, 0], 4, &mut destination).unwrap();
        assert_eq!(&destination[..8], &source[36..44]);
        assert_eq!(&destination[16..24], &source[12..20]);
        assert!(
            destination[8..16]
                .iter()
                .chain(&destination[24..])
                .all(|&v| v == 0xa5)
        );
        assert!(oracle(op, &source[..43], &[0], 1, &mut destination).is_err());
        assert!(
            RingGather {
                destination_stride: 7,
                ..op
            }
            .validate(Need::rows(1))
            .is_err()
        );
    }
}
