//! Retain disjoint spans of raw feature rows without interpreting their bits.
use kernel_core::backend::Need;
use kernel_core::contract::Refusal;

pub const CONTRACT: &str = "deepseek41.dspark.retain-rows.bytes.v1";
pub const MAX_SEGMENTS: u32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainRows {
    pub row_bytes: u32,
    pub source_rows: u32,
    pub destination_rows: u32,
}

/// Three little-endian U32 words per segment in device metadata. A caller
/// validates the complete list before uploading any metadata or launching.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Segment {
    pub source: u32,
    pub destination: u32,
    pub rows: u32,
}

impl RetainRows {
    pub fn validate(self, need: Need) -> Result<(), Refusal> {
        if self.row_bytes == 0
            || self.source_rows == 0
            || self.destination_rows == 0
            || need.capacity == 0
            || need.capacity > MAX_SEGMENTS
        {
            return Err(Refusal(
                "retained-row copy has invalid geometry or capacity",
            ));
        }
        Ok(())
    }

    pub fn source_bytes(self) -> u64 {
        u64::from(self.source_rows) * u64::from(self.row_bytes)
    }
    pub fn destination_bytes(self) -> u64 {
        u64::from(self.destination_rows) * u64::from(self.row_bytes)
    }

    /// Reject every invalid range and overlapping destination before writing.
    /// Repeated source rows are allowed because all source bytes are immutable.
    pub fn encode(self, segments: &[Segment]) -> Result<Vec<u8>, Refusal> {
        let count =
            u32::try_from(segments.len()).map_err(|_| Refusal("too many retention segments"))?;
        self.validate(Need::rows(count))?;
        for (index, segment) in segments.iter().enumerate() {
            let end = segment
                .destination
                .checked_add(segment.rows)
                .filter(|&end| end <= self.destination_rows)
                .ok_or(Refusal("retention destination is out of bounds"))?;
            if segment.rows == 0
                || segment
                    .source
                    .checked_add(segment.rows)
                    .is_none_or(|end| end > self.source_rows)
            {
                return Err(Refusal("retention source or row count is invalid"));
            }
            for previous in &segments[..index] {
                if segment.destination < previous.destination + previous.rows
                    && previous.destination < end
                {
                    return Err(Refusal("retention destinations overlap"));
                }
            }
        }
        Ok(segments
            .iter()
            .flat_map(|segment| [segment.source, segment.destination, segment.rows])
            .flat_map(u32::to_le_bytes)
            .collect())
    }
}

kernel_core::scratch_record! {
    /// Metadata is `RetainRows::encode` output, with Counts::rows segments.
    /// Source, destination and metadata byte ranges must be disjoint and
    /// remain live until all consumers on the stream finish.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RetainRowsBuffers {
        pub source: value,
        pub segments: value,
        pub destination: value,
    }
}

pub fn oracle(
    op: RetainRows,
    source: &[u8],
    segments: &[Segment],
    destination: &mut [u8],
) -> Result<(), Refusal> {
    op.encode(segments)?;
    if (source.len() as u64) < op.source_bytes()
        || (destination.len() as u64) < op.destination_bytes()
    {
        return Err(Refusal("retained-row oracle has a short buffer"));
    }
    for segment in segments {
        let from = segment.source as usize * op.row_bytes as usize;
        let to = segment.destination as usize * op.row_bytes as usize;
        let bytes = segment.rows as usize * op.row_bytes as usize;
        destination[to..to + bytes].copy_from_slice(&source[from..from + bytes]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reordered_spans_preserve_every_byte_and_untouched_row() {
        let op = RetainRows {
            row_bytes: 19,
            source_rows: 8,
            destination_rows: 12,
        };
        let source: Vec<_> = (0..op.source_bytes()).map(|i| (i * 37) as u8).collect();
        let segments = [
            Segment {
                source: 5,
                destination: 10,
                rows: 2,
            },
            Segment {
                source: 0,
                destination: 2,
                rows: 3,
            },
            Segment {
                source: 5,
                destination: 0,
                rows: 1,
            },
        ];
        let mut output = vec![0xa5; op.destination_bytes() as usize + 32];
        oracle(op, &source, &segments, &mut output).unwrap();
        for row in 0..12 {
            let wanted = match row {
                0 => Some(5),
                2..=4 => Some(row - 2),
                10..=11 => Some(row - 5),
                _ => None,
            };
            let bytes = &output[row * 19..(row + 1) * 19];
            match wanted {
                Some(from) => assert_eq!(bytes, &source[from * 19..(from + 1) * 19]),
                None => assert!(bytes.iter().all(|&byte| byte == 0xa5)),
            }
        }
        assert!(output[12 * 19..].iter().all(|&byte| byte == 0xa5));
        assert_eq!(
            &op.encode(&segments).unwrap()[..12],
            &[5, 0, 0, 0, 10, 0, 0, 0, 2, 0, 0, 0]
        );
    }

    #[test]
    fn invalid_later_segments_refuse_before_the_first_write() {
        let op = RetainRows {
            row_bytes: 32,
            source_rows: 8,
            destination_rows: 12,
        };
        let first = Segment {
            source: 0,
            destination: 2,
            rows: 3,
        };
        for bad in [
            Segment {
                source: 8,
                destination: 8,
                rows: 1,
            },
            Segment {
                source: u32::MAX,
                destination: 8,
                rows: 2,
            },
            Segment {
                source: 0,
                destination: u32::MAX,
                rows: 2,
            },
            Segment {
                source: 0,
                destination: 4,
                rows: 1,
            },
            Segment {
                source: 0,
                destination: 8,
                rows: 0,
            },
        ] {
            let mut output = vec![0xa5; op.destination_bytes() as usize];
            assert!(
                oracle(
                    op,
                    &vec![0; op.source_bytes() as usize],
                    &[first, bad],
                    &mut output
                )
                .is_err()
            );
            assert!(output.iter().all(|&byte| byte == 0xa5));
        }
        assert!(op.encode(&[]).is_err());
        assert!(op.encode(&vec![first; MAX_SEGMENTS as usize + 1]).is_err());
    }
}
