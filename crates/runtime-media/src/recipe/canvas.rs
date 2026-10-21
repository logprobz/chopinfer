//! Patch rows kept as the 8-bit canvas they are cut from.
//!
//! Each value of an 8-bit-canvas recipe is a function of one 8-bit sample
//! and its channel, so the canvas and one 256-entry table per channel
//! determine every row. A prepared image keeps them, 3 bytes per pixel, in
//! place of its F32 rows (12 bytes per pixel for each temporal patch), and
//! its payload is written from them when it is needed, bit for bit the bytes
//! of the rows.

use super::PatchGrid;
use crate::core::try_with_capacity;
use crate::{MediaError, MediaResult};

/// How rows are cut from the canvas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanvasLayout {
    /// The order of [`super::patch_grid`] for one image repeated over
    /// `temporal_patch` frames: merge blocks in reading order, the patches
    /// of each block in reading order, and each patch channel-major with the
    /// frames inside the channel.
    MergedPatches {
        patch: u32,
        temporal_patch: u32,
        merge: u32,
    },
    /// Patches in reading order, each channel-major.
    Patches { patch: u32 },
}

/// How a payload stores each value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadElement {
    /// Little-endian F32.
    F32,
    /// Little-endian BF16, rounded to nearest with ties to even.
    Bf16,
}

impl PayloadElement {
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            PayloadElement::F32 => 4,
            PayloadElement::Bf16 => 2,
        }
    }
}

/// The BF16 bits of `value`, rounded to nearest with ties to even.
#[must_use]
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The bytes a canvas keeps besides its samples: the three value tables.
const TABLE_BYTES: usize = 3 * 256 * 4;

/// An image's patch rows as an interleaved RGB canvas and the value of each
/// sample per channel.
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasRows {
    height: u32,
    width: u32,
    canvas: Vec<u8>,
    table: [[f32; 256]; 3],
    layout: CanvasLayout,
    tokens: u64,
}

impl CanvasRows {
    /// Checks that `layout` tiles a `height x width` canvas and that the
    /// rows fit in memory.
    pub(crate) fn new(
        height: u32,
        width: u32,
        canvas: Vec<u8>,
        table: [[f32; 256]; 3],
        layout: CanvasLayout,
        tokens: u64,
    ) -> MediaResult<Self> {
        let (patch, block) = match layout {
            CanvasLayout::MergedPatches {
                patch,
                temporal_patch,
                merge,
            } => {
                if temporal_patch == 0 {
                    return Err(MediaError::new("invalid frame or patch count"));
                }
                (patch, patch.checked_mul(merge))
            }
            CanvasLayout::Patches { patch } => (patch, Some(patch)),
        };
        let block = block
            .filter(|block| patch != 0 && *block != 0)
            .ok_or_else(|| MediaError::new("patch geometry overflow"))?;
        if height % block != 0 || width % block != 0 {
            return Err(MediaError::new("image does not fit patch grid"));
        }
        if canvas.len() as u64 != u64::from(height) * u64::from(width) * 3 {
            return Err(MediaError::new(
                "RGB byte count differs from image geometry",
            ));
        }
        let rows = Self {
            height,
            width,
            canvas,
            table,
            layout,
            tokens,
        };
        rows.values()
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| MediaError::new("patch allocation overflow"))?;
        Ok(rows)
    }

    fn patch(&self) -> u32 {
        match self.layout {
            CanvasLayout::MergedPatches { patch, .. } | CanvasLayout::Patches { patch } => patch,
        }
    }

    fn frames(&self) -> usize {
        match self.layout {
            CanvasLayout::MergedPatches { temporal_patch, .. } => temporal_patch as usize,
            CanvasLayout::Patches { .. } => 1,
        }
    }

    /// The patch grid `[t, h, w]`: one temporal step of `h x w` patches.
    #[must_use]
    pub fn grid(&self) -> [u32; 3] {
        [1, self.height / self.patch(), self.width / self.patch()]
    }

    #[must_use]
    pub fn columns(&self) -> usize {
        3 * self.frames() * self.patch() as usize * self.patch() as usize
    }

    #[must_use]
    pub fn tokens(&self) -> u64 {
        self.tokens
    }

    fn values(&self) -> Option<usize> {
        let [_, h, w] = self.grid();
        (h as usize)
            .checked_mul(w as usize)
            .and_then(|patches| patches.checked_mul(self.columns()))
    }

    /// Bytes of the payload these rows write as `element`.
    #[must_use]
    pub fn payload_len(&self, element: PayloadElement) -> usize {
        self.values().unwrap_or(0) * element.bytes()
    }

    /// Bytes this form keeps: the canvas and its tables.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.canvas.len() + TABLE_BYTES
    }

    /// Calls `patch` with the top-left sample of each patch, in row order.
    fn for_each_patch(&self, mut patch: impl FnMut(usize, usize)) {
        let p = self.patch() as usize;
        let [_, grid_h, grid_w] = self.grid().map(|side| side as usize);
        match self.layout {
            CanvasLayout::MergedPatches { merge, .. } => {
                let m = merge as usize;
                for bh in 0..grid_h / m {
                    for bw in 0..grid_w / m {
                        for mh in 0..m {
                            for mw in 0..m {
                                patch((bh * m + mh) * p, (bw * m + mw) * p);
                            }
                        }
                    }
                }
            }
            CanvasLayout::Patches { .. } => {
                for by in 0..grid_h {
                    for bx in 0..grid_w {
                        patch(by * p, bx * p);
                    }
                }
            }
        }
    }

    /// Calls `value` with each sample's table entry of the patch at `(y, x)`
    /// in row order: per channel, each frame repeats the patch's samples.
    fn for_each_sample<T: Copy>(
        &self,
        table: &[[T; 256]; 3],
        y: usize,
        x: usize,
        mut value: impl FnMut(T),
    ) {
        let p = self.patch() as usize;
        let width = self.width as usize;
        for (c, values) in table.iter().enumerate() {
            for _ in 0..self.frames() {
                for row in y..y + p {
                    let start = (row * width + x) * 3;
                    for pixel in self.canvas[start..start + p * 3].chunks_exact(3) {
                        value(values[usize::from(pixel[c])]);
                    }
                }
            }
        }
    }

    /// The F32 rows, as [`PatchGrid::rows`].
    pub fn rows(&self) -> MediaResult<Vec<f32>> {
        let mut rows = try_with_capacity(self.values().unwrap_or(0))?;
        self.for_each_patch(|y, x| self.for_each_sample(&self.table, y, x, |v| rows.push(v)));
        Ok(rows)
    }

    /// The patch grid these rows make.
    pub fn into_grid(self) -> MediaResult<PatchGrid> {
        Ok(PatchGrid {
            grid: self.grid(),
            columns: self.columns(),
            rows: self.rows()?,
            tokens: self.tokens,
        })
    }

    fn element_table<const N: usize>(
        &self,
        encode: impl Fn(f32) -> [u8; N],
    ) -> [[[u8; N]; 256]; 3] {
        self.table.map(|values| values.map(&encode))
    }

    /// Appends the payload, [`Self::payload_len`] bytes, to `out`.
    pub fn write_payload(&self, element: PayloadElement, out: &mut Vec<u8>) {
        out.reserve(self.payload_len(element));
        self.write_patches(element, out, |_| {});
    }

    /// Passes the payload to `chunk` in order, in pieces of at least
    /// `chunk_bytes` (the last may be shorter), without holding it whole.
    pub fn for_each_payload_chunk(
        &self,
        element: PayloadElement,
        chunk_bytes: usize,
        chunk: &mut dyn FnMut(&[u8]),
    ) {
        let chunk_bytes = chunk_bytes.max(1);
        let mut buffer = Vec::with_capacity(chunk_bytes + self.columns() * element.bytes());
        self.write_patches(element, &mut buffer, |buffer| {
            if buffer.len() >= chunk_bytes {
                chunk(buffer);
                buffer.clear();
            }
        });
        if !buffer.is_empty() {
            chunk(&buffer);
        }
    }

    /// Appends each patch's bytes to `out`, calling `written` after each.
    fn write_patches(
        &self,
        element: PayloadElement,
        out: &mut Vec<u8>,
        written: impl FnMut(&mut Vec<u8>),
    ) {
        match element {
            PayloadElement::F32 => {
                self.write_with(&self.element_table(f32::to_le_bytes), out, written);
            }
            PayloadElement::Bf16 => self.write_with(
                &self.element_table(|value| bf16_bits(value).to_le_bytes()),
                out,
                written,
            ),
        }
    }

    fn write_with<const N: usize>(
        &self,
        table: &[[[u8; N]; 256]; 3],
        out: &mut Vec<u8>,
        mut written: impl FnMut(&mut Vec<u8>),
    ) {
        self.for_each_patch(|y, x| {
            self.for_each_sample(table, y, x, |bytes| out.extend_from_slice(&bytes));
            written(out);
        });
    }
}

/// The table of `value(channel, sample)` for every 8-bit sample.
pub(crate) fn value_table(value: impl Fn(usize, u8) -> f32) -> [[f32; 256]; 3] {
    let mut table = [[0.0; 256]; 3];
    for (c, values) in table.iter_mut().enumerate() {
        for (v, entry) in values.iter_mut().enumerate() {
            *entry = value(c, v as u8);
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas(height: u32, width: u32) -> Vec<u8> {
        (0..height * width * 3)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect()
    }

    fn table() -> [[f32; 256]; 3] {
        value_table(|c, v| (f32::from(v) - 10.0 * c as f32) / 3.7)
    }

    #[test]
    fn layouts_must_tile_the_canvas() {
        let merged = CanvasLayout::MergedPatches {
            patch: 2,
            temporal_patch: 2,
            merge: 2,
        };
        assert!(CanvasRows::new(6, 8, canvas(6, 8), table(), merged, 6).is_err());
        assert!(CanvasRows::new(8, 8, canvas(8, 4), table(), merged, 4).is_err());
        let rows = CanvasRows::new(8, 4, canvas(8, 4), table(), merged, 2).unwrap();
        assert_eq!((rows.grid(), rows.columns()), ([1, 4, 2], 24));
        assert_eq!(rows.payload_len(PayloadElement::F32), 8 * 24 * 4);
        assert_eq!(rows.retained_bytes(), 8 * 4 * 3 + TABLE_BYTES);
    }

    #[test]
    fn payloads_are_the_bytes_of_the_rows_in_any_chunking() {
        for layout in [
            CanvasLayout::MergedPatches {
                patch: 2,
                temporal_patch: 2,
                merge: 2,
            },
            CanvasLayout::Patches { patch: 4 },
        ] {
            let rows = CanvasRows::new(8, 12, canvas(8, 12), table(), layout, 1).unwrap();
            let values = rows.rows().unwrap();
            assert_eq!(values.len() * 4, rows.payload_len(PayloadElement::F32));
            for (element, bytes) in [
                (
                    PayloadElement::F32,
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                ),
                (
                    PayloadElement::Bf16,
                    values
                        .iter()
                        .flat_map(|v| bf16_bits(*v).to_le_bytes())
                        .collect(),
                ),
            ] {
                let mut written = vec![7];
                rows.write_payload(element, &mut written);
                assert_eq!(written[1..], bytes[..]);
                for chunk_bytes in [1, 5, 96, 1 << 20] {
                    let mut chunked = Vec::new();
                    rows.for_each_payload_chunk(element, chunk_bytes, &mut |chunk| {
                        assert!(!chunk.is_empty());
                        chunked.extend_from_slice(chunk);
                    });
                    assert_eq!(chunked, bytes, "{layout:?} {element:?} {chunk_bytes}");
                }
            }
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
    }
}
