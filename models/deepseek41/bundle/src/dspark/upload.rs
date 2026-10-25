//! Coalesce drafter metadata without changing any consumer's byte encoding.
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

#[derive(Clone, Copy)]
pub(crate) struct Layout {
    pub inputs: (u64, u64),
    pub source_slots: (u64, u64),
    pub first: (u64, u64),
    pub last: (u64, u64),
    pub item_slots: (u64, u64),
    pub predecessor_ids: (u64, u64),
    pub live_items: (u64, u64),
    /// One `kernel_core::draft_noise::NoiseRecord` per item.
    pub noise: (u64, u64),
    pub keys: (u64, u64),
    pub bytes: u64,
}

impl Layout {
    /// Reserve the unused bytes after the live-item word without moving
    /// any existing plane or growing the allocation. At most 8 * 3 rows.
    pub fn head_indices(&self) -> (u64, u64) {
        (self.live_items.0 + 16, 24 * 4)
    }
    pub fn head_rows(&self) -> (u64, u64) {
        (self.live_items.0 + 4, 4)
    }

    pub fn new(inputs: u64, items: u64, rows: u64, cells: u64) -> Result<Self, String> {
        if inputs == 0 || items == 0 || rows == 0 || cells == 0 {
            return Err("DSpark metadata layout is empty".into());
        }
        let words = items
            .checked_mul(4)
            .ok_or("DSpark metadata items overflow")?;
        let records = items
            .checked_mul(kernel_core::draft_noise::NOISE_BYTES as u64)
            .ok_or("DSpark metadata items overflow")?;
        let keys = rows
            .checked_mul(cells)
            .and_then(|n| n.checked_mul(4))
            .ok_or("DSpark key metadata overflow")?;
        let mut end = inputs;
        let mut range = |bytes| -> Result<(u64, u64), String> {
            let start = end
                .checked_add(127)
                .ok_or("DSpark metadata alignment overflow")?
                / 128
                * 128;
            end = start
                .checked_add(bytes)
                .ok_or("DSpark metadata length overflow")?;
            Ok((start, bytes))
        };
        Ok(Self {
            inputs: (0, inputs),
            source_slots: range(words)?,
            first: range(words)?,
            last: range(words)?,
            item_slots: range(words)?,
            predecessor_ids: range(words)?,
            live_items: range(4)?,
            noise: range(records)?,
            keys: range(keys)?,
            bytes: end,
        })
    }
}

/// Only the supplied metadata planes are semantically live. Padding between
/// them is initialized, so a larger allocation never exposes host memory.
pub(crate) fn pack(arena: Span, updates: &[(Span, Vec<u8>)]) -> Result<(Span, Vec<u8>), String> {
    let mut intervals = Vec::new();
    let mut first = arena.length;
    let mut end = 0;
    for (span, bytes) in updates {
        let start = span
            .offset
            .checked_sub(arena.offset)
            .ok_or("DSpark metadata precedes arena")?;
        let stop = start
            .checked_add(bytes.len() as u64)
            .ok_or("DSpark metadata range overflow")?;
        if span.buffer != arena.buffer
            || bytes.len() as u64 > span.length
            || start
                .checked_add(span.length)
                .is_none_or(|end| end > arena.length)
            || stop > arena.length
        {
            return Err("DSpark metadata exceeds its arena".into());
        }
        if !bytes.is_empty() {
            first = first.min(start);
            end = end.max(stop);
            intervals.push((start, stop));
        }
    }
    intervals.sort_unstable();
    if intervals.is_empty() || intervals.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err("DSpark metadata is empty or overlapping".into());
    }
    let length = usize::try_from(end - first).map_err(|_| "DSpark metadata exceeds host size")?;
    let mut output = vec![0; length];
    for (span, bytes) in updates.iter().filter(|(_, bytes)| !bytes.is_empty()) {
        let start = (span.offset - arena.offset - first) as usize;
        output[start..start + bytes.len()].copy_from_slice(bytes);
    }
    Ok((super::device::slice(arena, first, end - first)?, output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn packed_metadata_preserves_every_plane_and_initializes_gaps() {
        let arena = Span::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 256, 1024);
        let a = Span::new(arena.buffer, 384, 128);
        let b = Span::new(arena.buffer, 768, 64);
        let bytes = [0_u32, u32::MAX, 123]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let (span, packed) = pack(arena, &[(b, vec![1, 2, 3]), (a, bytes.clone())]).unwrap();
        assert_eq!(span.offset, a.offset);
        assert_eq!(span.length, b.offset + 3 - a.offset);
        assert_eq!(&packed[..bytes.len()], bytes);
        assert!(packed[bytes.len()..384].iter().all(|&byte| byte == 0));
        assert_eq!(&packed[384..], [1, 2, 3]);
        assert!(pack(arena, &[(a, vec![0; 129])]).is_err());
        assert!(pack(arena, &[(a, vec![0; 4]), (a, vec![1; 4])]).is_err());
        assert!(pack(arena, &[]).is_err());
        assert!(
            pack(
                Span {
                    offset: 512,
                    ..arena
                },
                &[(a, vec![0; 4])]
            )
            .is_err()
        );
    }

    #[test]
    fn layout_keeps_keys_last_and_checks_overflow() {
        let layout = Layout::new(4096, 32, 160, 133).unwrap();
        let ranges = [
            layout.inputs,
            layout.source_slots,
            layout.first,
            layout.last,
            layout.item_slots,
            layout.predecessor_ids,
            layout.live_items,
            layout.noise,
            layout.keys,
        ];
        assert!(ranges.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0));
        assert!(ranges.iter().all(|&(start, _)| start % 128 == 0));
        assert_eq!(layout.bytes, layout.keys.0 + layout.keys.1);
        assert_eq!(layout.noise.1, 32 * 32);
        assert_eq!(layout.head_rows(), (layout.live_items.0 + 4, 4));
        assert_eq!(layout.head_indices(), (layout.live_items.0 + 16, 96));
        assert!(layout.head_indices().0 + layout.head_indices().1 <= layout.noise.0);
        assert!(Layout::new(u64::MAX, 32, 160, 133).is_err());
        assert!(Layout::new(4096, u64::MAX, 160, 133).is_err());
    }
}
