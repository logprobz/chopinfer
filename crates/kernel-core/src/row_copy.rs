//! Row copies: ops that move element bytes between row layouts and compute
//! nothing. The stream broadcast ([`StreamBroadcast`]) copies each row into
//! every stream of a `[rows, streams, width]` destination; the column slice
//! ([`ColumnSlice`]) copies `width` columns from `offset` of each source
//! row of `stride` elements into a dense `[rows, width]` destination.
//!
//! Every field of the contracts is math: the element type, whose bytes are
//! copied without numeric interpretation (a NaN or a negative zero keeps
//! its bits), the row widths, the streams, and the slice's source stride
//! and offset. How a backend moves the bytes (vector widths, threads,
//! grids, the word size it copies in) is its own choice, and every choice
//! writes the same bytes.
//!
//! Both ops read `live_rows`, the device count of live rows: they write the
//! destination elements of the live rows and leave every other destination
//! byte as it was, so rows past the live count and, for a slice, nothing
//! past each row's `width` columns change. Source and destination do not
//! overlap. The oracles are [`broadcast_streams`] and [`column_slice`].
//!
//! Each contract states `alignment_bytes`, a power of two, which the host
//! guarantees: it binds the source and destination views at multiples of
//! that many bytes (and the live count at its U32 alignment). A backend
//! whose kernel needs a coarser alignment refuses the op when it prepares
//! it; views stay opaque, so no launch checks an address.

use crate::contract::{DType, Refusal, Validate};

/// The bytes of one element of `dtype`, for the types whose elements are
/// whole bytes; `None` for sub-byte codes, which no row copy moves.
pub const fn element_bytes(dtype: DType) -> Option<u32> {
    match dtype {
        DType::U8 | DType::Fp8E4M3 | DType::Fp8E5M2 | DType::Ue8M0 | DType::Ue4M3 => Some(1),
        DType::Bf16 | DType::F16 => Some(2),
        DType::F32 | DType::I32 => Some(4),
        DType::I64 => Some(8),
        DType::Fp6E2M3 | DType::Fp6E3M2 | DType::Fp4E2M1 => None,
    }
}

/// The bytes of `elements` elements of `dtype`.
fn bytes(dtype: DType, elements: Option<u64>) -> Result<u64, Refusal> {
    let bytes = element_bytes(dtype).ok_or(Refusal("a row copy moves elements of whole bytes"))?;
    elements
        .and_then(|elements| elements.checked_mul(u64::from(bytes)))
        .ok_or(Refusal("row copy sizes overflow u64"))
}

fn row_bytes(dtype: DType, elements: u32, rows: u64) -> Result<u64, Refusal> {
    bytes(dtype, rows.checked_mul(u64::from(elements)))
}

/// The bytes of `rows` rows of `row` bytes.
fn rows_bytes(row: Result<u64, Refusal>, rows: u32) -> Result<u64, Refusal> {
    row?.checked_mul(u64::from(rows))
        .ok_or(Refusal("row copy sizes overflow u64"))
}

fn check_alignment(alignment_bytes: u32) -> Result<(), Refusal> {
    if !alignment_bytes.is_power_of_two() {
        return Err(Refusal("a row copy's view alignment is a power of two"));
    }
    Ok(())
}

/// Each source row of `width` elements copied into every one of `streams`
/// streams: destination row `r`, stream `s` is source row `r`
/// (`[rows, width]` into `[rows, streams, width]`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct StreamBroadcast {
    pub dtype: DType,
    pub width: u32,
    pub streams: u32,
    /// The alignment of every view the host binds, in bytes.
    pub alignment_bytes: u32,
}

impl StreamBroadcast {
    /// The bytes of one source row.
    pub fn source_row_bytes(&self) -> Result<u64, Refusal> {
        row_bytes(self.dtype, self.width, 1)
    }

    /// The bytes of one destination row: every stream's copy.
    pub fn destination_row_bytes(&self) -> Result<u64, Refusal> {
        row_bytes(self.dtype, self.width, u64::from(self.streams))
    }

    /// The bytes `rows` rows read.
    pub fn source_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        rows_bytes(self.source_row_bytes(), rows)
    }

    /// The bytes `rows` rows write over.
    pub fn destination_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        rows_bytes(self.destination_row_bytes(), rows)
    }
}

impl Validate for StreamBroadcast {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 || self.streams == 0 {
            return Err(Refusal("a stream broadcast needs columns and streams"));
        }
        check_alignment(self.alignment_bytes)?;
        self.destination_row_bytes().map(|_| ())
    }
}

/// Columns `offset..offset + width` of each source row of `stride`
/// elements, copied into a dense destination row of `width` elements
/// (`[rows, stride]` into `[rows, width]`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ColumnSlice {
    pub dtype: DType,
    pub width: u32,
    pub stride: u32,
    pub offset: u32,
    /// The alignment of every view the host binds, in bytes.
    pub alignment_bytes: u32,
}

impl ColumnSlice {
    /// The bytes of one source row.
    pub fn source_row_bytes(&self) -> Result<u64, Refusal> {
        row_bytes(self.dtype, self.stride, 1)
    }

    /// The bytes of one destination row.
    pub fn destination_row_bytes(&self) -> Result<u64, Refusal> {
        row_bytes(self.dtype, self.width, 1)
    }

    /// The bytes `rows` rows read: whole rows up to the last, whose columns
    /// past the slice are not read.
    pub fn source_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        if rows == 0 {
            return Ok(0);
        }
        let elements = u64::from(rows - 1) * u64::from(self.stride)
            + u64::from(self.offset)
            + u64::from(self.width);
        bytes(self.dtype, Some(elements))
    }

    /// The bytes `rows` rows write over.
    pub fn destination_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        rows_bytes(self.destination_row_bytes(), rows)
    }
}

impl Validate for ColumnSlice {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 {
            return Err(Refusal("a column slice needs columns"));
        }
        if u64::from(self.offset) + u64::from(self.width) > u64::from(self.stride) {
            return Err(Refusal("a column slice lies within its source row"));
        }
        check_alignment(self.alignment_bytes)?;
        self.source_row_bytes().map(|_| ())
    }
}

crate::scratch_record! {
    /// A row copy's buffers: the `source` rows, the `destination` rows it
    /// writes, and `live_rows`, the device U32 count of rows it copies,
    /// which the host keeps at most the launch's rows. Rows are dense at
    /// the op's row bytes (`source_row_bytes`, `destination_row_bytes`).
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RowCopyBuffers {
        pub source: value,
        pub destination: value,
        pub live_rows: value,
    }
}

/// The stream broadcast of `op` over `live_rows` rows of `source` into
/// `destination`, whose other bytes it leaves.
pub fn broadcast_streams(
    op: &StreamBroadcast,
    source: &[u8],
    destination: &mut [u8],
    live_rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let row = op.source_row_bytes()? as usize;
    if (source.len() as u64) < op.source_bytes(live_rows)?
        || (destination.len() as u64) < op.destination_bytes(live_rows)?
    {
        return Err(Refusal("row copy buffers do not hold the live rows"));
    }
    for (r, from) in source
        .chunks_exact(row)
        .take(live_rows as usize)
        .enumerate()
    {
        for s in 0..op.streams as usize {
            let at = (r * op.streams as usize + s) * row;
            destination[at..at + row].copy_from_slice(from);
        }
    }
    Ok(())
}

/// The column slice of `op` over `live_rows` rows of `source` into
/// `destination`, whose other bytes it leaves.
pub fn column_slice(
    op: &ColumnSlice,
    source: &[u8],
    destination: &mut [u8],
    live_rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let stride = op.source_row_bytes()? as usize;
    let row = op.destination_row_bytes()? as usize;
    if (source.len() as u64) < op.source_bytes(live_rows)?
        || (destination.len() as u64) < op.destination_bytes(live_rows)?
    {
        return Err(Refusal("row copy buffers do not hold the live rows"));
    }
    let element = row / op.width as usize;
    let offset = op.offset as usize * element;
    for r in 0..live_rows as usize {
        let from = r * stride + offset;
        destination[r * row..(r + 1) * row].copy_from_slice(&source[from..from + row]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const POISON: u8 = 0xa5;

    fn pattern(count: usize) -> Vec<u8> {
        (0..count).map(|i| (i * 7 + 3) as u8).collect()
    }

    #[test]
    fn element_bytes_cover_whole_byte_types_only() {
        assert_eq!(element_bytes(DType::Bf16), Some(2));
        assert_eq!(element_bytes(DType::F32), Some(4));
        assert_eq!(element_bytes(DType::Fp8E4M3), Some(1));
        assert_eq!(element_bytes(DType::I64), Some(8));
        assert_eq!(element_bytes(DType::Fp4E2M1), None);
    }

    #[test]
    fn contracts_check_their_math_only() {
        let broadcast = StreamBroadcast {
            dtype: DType::Bf16,
            width: 3,
            streams: 5,
            alignment_bytes: 2,
        };
        assert_eq!(broadcast.validate(), Ok(()));
        // No vector width or least alignment: odd widths and narrow views
        // are the backend's to refuse.
        assert_eq!(broadcast.destination_row_bytes(), Ok(30));
        assert_eq!(broadcast.destination_bytes(3), Ok(90));
        // Sizes past u64 are refused, not wrapped.
        let huge = StreamBroadcast {
            dtype: DType::I64,
            width: 1 << 30,
            streams: 1 << 30,
            alignment_bytes: 16,
        };
        assert_eq!(huge.validate(), Ok(()));
        assert!(huge.destination_bytes(2).is_err());
        for wrong in [
            StreamBroadcast {
                width: 0,
                ..broadcast
            },
            StreamBroadcast {
                streams: 0,
                ..broadcast
            },
            StreamBroadcast {
                dtype: DType::Fp6E2M3,
                ..broadcast
            },
            StreamBroadcast {
                dtype: DType::I64,
                width: u32::MAX,
                streams: u32::MAX,
                alignment_bytes: 16,
            },
            StreamBroadcast {
                alignment_bytes: 0,
                ..broadcast
            },
            StreamBroadcast {
                alignment_bytes: 12,
                ..broadcast
            },
        ] {
            assert!(wrong.validate().is_err(), "{wrong:?}");
        }
        let slice = ColumnSlice {
            dtype: DType::F32,
            width: 288,
            stride: 320,
            offset: 32,
            alignment_bytes: 16,
        };
        assert_eq!(slice.validate(), Ok(()));
        assert_eq!(slice.destination_bytes(2), Ok(2 * 288 * 4));
        assert_eq!(slice.source_bytes(2), Ok((320 + 32 + 288) * 4));
        assert_eq!(slice.source_bytes(0), Ok(0));
        for wrong in [
            ColumnSlice { width: 0, ..slice },
            ColumnSlice {
                offset: 33,
                ..slice
            },
            ColumnSlice {
                offset: u32::MAX,
                ..slice
            },
            ColumnSlice {
                dtype: DType::Fp4E2M1,
                ..slice
            },
            ColumnSlice {
                alignment_bytes: 3,
                ..slice
            },
        ] {
            assert!(wrong.validate().is_err(), "{wrong:?}");
        }
    }

    #[test]
    fn the_broadcast_oracle_copies_live_rows_into_every_stream() {
        let op = StreamBroadcast {
            dtype: DType::Bf16,
            width: 3,
            streams: 2,
            alignment_bytes: 2,
        };
        let source = pattern(4 * 6);
        let mut destination = vec![POISON; 4 * 12];
        broadcast_streams(&op, &source, &mut destination, 3).unwrap();
        for row in 0..4 {
            for stream in 0..2 {
                let at = (row * 2 + stream) * 6;
                let got = &destination[at..at + 6];
                if row < 3 {
                    assert_eq!(got, &source[row * 6..row * 6 + 6]);
                } else {
                    assert!(got.iter().all(|&b| b == POISON));
                }
            }
        }
        let mut untouched = vec![POISON; 4 * 12];
        broadcast_streams(&op, &source, &mut untouched, 0).unwrap();
        assert!(untouched.iter().all(|&b| b == POISON));
        assert!(broadcast_streams(&op, &source, &mut [0; 4 * 12 - 1], 4).is_err());
        assert!(broadcast_streams(&op, &source[1..], &mut destination, 4).is_err());
    }

    #[test]
    fn the_slice_oracle_copies_only_the_live_columns() {
        let op = ColumnSlice {
            dtype: DType::Bf16,
            width: 2,
            stride: 5,
            offset: 3,
            alignment_bytes: 2,
        };
        let source = pattern(3 * 10);
        let mut destination = vec![POISON; 3 * 4 + 2];
        column_slice(&op, &source, &mut destination, 2).unwrap();
        assert_eq!(&destination[..4], &source[6..10]);
        assert_eq!(&destination[4..8], &source[16..20]);
        assert!(destination[8..].iter().all(|&b| b == POISON));
        // The last row's columns past the slice are not read.
        let short = &source[..10 + 10];
        let mut again = vec![POISON; 8];
        column_slice(&op, short, &mut again, 2).unwrap();
        assert_eq!(&again[..], &destination[..8]);
        assert!(column_slice(&op, &short[..19], &mut again, 2).is_err());
        assert!(column_slice(&op, &source, &mut [0; 7], 2).is_err());
    }
}
