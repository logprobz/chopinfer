//! Reversible host-side weight and scale layouts for TMA and register MMA.

pub mod fragment;

use crate::contract::Refusal;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Layout {
    RowMajor {
        rows: usize,
        bytes_per_row: usize,
    },
    Sharded {
        rows: usize,
        bytes_per_row: usize,
        ranks: usize,
        axis: ShardAxis,
    },
    NibbleSwapped {
        elements: usize,
    },
    ScaleTile(ScaleTileLayout),
    Mxfp4MoeFragments(Mxfp4MoeFragmentLayout),
    Mxfp4MoeFragmentScales(Mxfp4MoeFragmentLayout),
    FragmentMajor(fragment::FragmentWeightLayout),
    RowTiles(RowTileLayout),
    QuadLanes(QuadLaneLayout),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Packed {
    pub bytes: Vec<u8>,
    pub layout: Layout,
}

/// A transform carries enough geometry to invert its own byte mapping.
pub trait Transform {
    fn pack(&self, source: &[u8]) -> Result<Packed, Refusal>;
    fn unpack(&self, packed: &Packed) -> Result<Vec<u8>, Refusal>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShardAxis {
    Rows,
    KBytes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkedScaleGrid {
    pub chunks: usize,
    pub rows_per_chunk: usize,
    pub k_blocks: usize,
    pub block_rows: usize,
}

impl ChunkedScaleGrid {
    pub fn scale_rows_per_chunk(self) -> usize {
        self.rows_per_chunk.div_ceil(self.block_rows)
    }
    pub fn total_scale_rows(self) -> usize {
        self.chunks * self.scale_rows_per_chunk()
    }

    /// Explicit chunk address. `row / block_rows` is wrong at partial tails.
    pub fn scale_row(self, weight_row: usize) -> Result<usize, Refusal> {
        if self.chunks == 0
            || self.rows_per_chunk == 0
            || self.block_rows == 0
            || weight_row >= self.chunks * self.rows_per_chunk
        {
            return Err(Refusal("weight row outside chunked scale grid"));
        }
        Ok(
            weight_row / self.rows_per_chunk * self.scale_rows_per_chunk()
                + (weight_row % self.rows_per_chunk) / self.block_rows,
        )
    }

    pub fn split_chunks(self, scales: &[u8]) -> Result<Vec<Vec<u8>>, Refusal> {
        if self.k_blocks == 0 || scales.len() != self.total_scale_rows() * self.k_blocks {
            return Err(Refusal("chunked scale grid byte count mismatch"));
        }
        let chunk_bytes = self.scale_rows_per_chunk() * self.k_blocks;
        Ok(scales
            .chunks_exact(chunk_bytes)
            .map(<[u8]>::to_vec)
            .collect())
    }

    /// A flat scale grid includes each partial tail as its own row block.
    /// `weight_row_to_scale_row` maps every source row into that flat grid.
    pub fn flatten(self, scales: &[u8]) -> Result<FlatScaleGrid, Refusal> {
        self.split_chunks(scales)?;
        let mut map = Vec::with_capacity(self.chunks * self.rows_per_chunk);
        for row in 0..self.chunks * self.rows_per_chunk {
            map.push(self.scale_row(row)?);
        }
        Ok(FlatScaleGrid {
            values: scales.to_vec(),
            weight_row_to_scale_row: map,
            k_blocks: self.k_blocks,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FlatScaleGrid {
    pub values: Vec<u8>,
    pub weight_row_to_scale_row: Vec<usize>,
    pub k_blocks: usize,
}

/// Scale bytes for one warp-level block-scaled MMA tile. A tile has at most
/// 128 scales, assigned to 32 lanes. Each lane loads one little-endian u32:
/// byte j of lane l holds logical scale (j*32+l), padding absent positions
/// with zero. Logical scales are row-major [tile_rows, k_blocks_per_tile].
/// This layout is parameterized for 1X, 2X and 4X MMA scale vectors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaleTileLayout {
    pub rows: usize,
    pub k_blocks: usize,
    pub tile_rows: usize,
    pub k_blocks_per_tile: usize,
}

impl ScaleTileLayout {
    fn check(self) -> Result<(), Refusal> {
        let tile = self
            .tile_rows
            .checked_mul(self.k_blocks_per_tile)
            .ok_or(Refusal("scale tile overflow"))?;
        if self.rows == 0
            || self.k_blocks == 0
            || self.tile_rows == 0
            || self.k_blocks_per_tile == 0
            || tile > 128
        {
            return Err(Refusal("scale tile needs 1 to 128 scales"));
        }
        Ok(())
    }

    fn tiles(self) -> (usize, usize) {
        (
            self.rows.div_ceil(self.tile_rows),
            self.k_blocks.div_ceil(self.k_blocks_per_tile),
        )
    }
}

impl Transform for ScaleTileLayout {
    fn pack(&self, source: &[u8]) -> Result<Packed, Refusal> {
        self.check()?;
        if source.len() != self.rows * self.k_blocks {
            return Err(Refusal("scale tile source size mismatch"));
        }
        let (row_tiles, k_tiles) = self.tiles();
        let mut bytes = vec![0_u8; row_tiles * k_tiles * 128];
        for rt in 0..row_tiles {
            for kt in 0..k_tiles {
                let tile_base = (rt * k_tiles + kt) * 128;
                for row in 0..self.tile_rows {
                    for kb in 0..self.k_blocks_per_tile {
                        let src_row = rt * self.tile_rows + row;
                        let src_k = kt * self.k_blocks_per_tile + kb;
                        if src_row >= self.rows || src_k >= self.k_blocks {
                            continue;
                        }
                        let logical = row * self.k_blocks_per_tile + kb;
                        let lane = logical % 32;
                        let byte = logical / 32;
                        bytes[tile_base + lane * 4 + byte] =
                            source[src_row * self.k_blocks + src_k];
                    }
                }
            }
        }
        Ok(Packed {
            bytes,
            layout: Layout::ScaleTile(*self),
        })
    }

    fn unpack(&self, packed: &Packed) -> Result<Vec<u8>, Refusal> {
        self.check()?;
        let (row_tiles, k_tiles) = self.tiles();
        if packed.layout != Layout::ScaleTile(*self)
            || packed.bytes.len() != row_tiles * k_tiles * 128
        {
            return Err(Refusal("scale tile packed layout mismatch"));
        }
        let mut source = vec![0_u8; self.rows * self.k_blocks];
        for rt in 0..row_tiles {
            for kt in 0..k_tiles {
                let tile_base = (rt * k_tiles + kt) * 128;
                for row in 0..self.tile_rows {
                    for kb in 0..self.k_blocks_per_tile {
                        let dst_row = rt * self.tile_rows + row;
                        let dst_k = kt * self.k_blocks_per_tile + kb;
                        if dst_row >= self.rows || dst_k >= self.k_blocks {
                            continue;
                        }
                        let logical = row * self.k_blocks_per_tile + kb;
                        source[dst_row * self.k_blocks + dst_k] =
                            packed.bytes[tile_base + (logical % 32) * 4 + logical / 32];
                    }
                }
            }
        }
        Ok(source)
    }
}

/// Expert-major rows, then 16-row strips and 64-K tiles. Each lane owns
/// 16 consecutive FP4 bytes. Four lanes share one compact UE8M0 scale word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mxfp4MoeFragmentLayout {
    pub experts: usize,
    pub rows_per_expert: usize,
    pub k: usize,
}

impl Mxfp4MoeFragmentLayout {
    fn geometry(self) -> Result<(usize, usize, usize), Refusal> {
        if self.experts == 0
            || self.rows_per_expert == 0
            || !self.rows_per_expert.is_multiple_of(16)
            || self.k == 0
            || !self.k.is_multiple_of(64)
        {
            return Err(Refusal("MXFP4 MoE fragments require 16 rows and 64 K"));
        }
        let strips = self
            .experts
            .checked_mul(self.rows_per_expert / 16)
            .ok_or(Refusal("MXFP4 MoE fragment size overflow"))?;
        let k_tiles = self.k / 64;
        let tiles = strips
            .checked_mul(k_tiles)
            .ok_or(Refusal("MXFP4 MoE fragment size overflow"))?;
        tiles
            .checked_mul(512)
            .ok_or(Refusal("MXFP4 MoE fragment size overflow"))?;
        Ok((strips, k_tiles, tiles))
    }

    fn code_index(
        self,
        strip: usize,
        k_tile: usize,
        lane: usize,
        step: usize,
        pair: usize,
    ) -> usize {
        let row = strip * 16 + (lane >> 2) + if pair & 1 != 0 { 8 } else { 0 };
        let column = k_tile * 64 + step * 16 + 2 * (lane & 3) + if pair >= 2 { 8 } else { 0 };
        row * (self.k / 2) + column / 2
    }

    pub fn pack_codes(self, source: &[u8]) -> Result<Packed, Refusal> {
        let (_, k_tiles, tiles) = self.geometry()?;
        let len = tiles
            .checked_mul(512)
            .ok_or(Refusal("MXFP4 MoE fragment size overflow"))?;
        if source.len() != len {
            return Err(Refusal("MXFP4 MoE code source size mismatch"));
        }
        let mut bytes = vec![0; len];
        for strip in 0..self.experts * (self.rows_per_expert / 16) {
            for kt in 0..k_tiles {
                let base = (strip * k_tiles + kt) * 512;
                for lane in 0..32 {
                    for step in 0..4 {
                        for pair in 0..4 {
                            bytes[base + lane * 16 + step * 4 + pair] =
                                source[self.code_index(strip, kt, lane, step, pair)];
                        }
                    }
                }
            }
        }
        Ok(Packed {
            bytes,
            layout: Layout::Mxfp4MoeFragments(self),
        })
    }

    pub fn pack_scales(self, source: &[u8]) -> Result<Packed, Refusal> {
        let (strips, k_tiles, tiles) = self.geometry()?;
        let source_len = strips
            .checked_mul(16)
            .and_then(|v| v.checked_mul(self.k / 32))
            .ok_or(Refusal("MXFP4 MoE scale size overflow"))?;
        if source.len() != source_len {
            return Err(Refusal("MXFP4 MoE scale source size mismatch"));
        }
        let mut bytes = vec![0; tiles * 32];
        for strip in 0..strips {
            for kt in 0..k_tiles {
                let base = (strip * k_tiles + kt) * 32;
                for g in 0..8 {
                    for block in 0..2 {
                        for high_row in 0..2 {
                            let row = strip * 16 + g + high_row * 8;
                            bytes[base + g * 4 + block * 2 + high_row] =
                                source[row * (self.k / 32) + kt * 2 + block];
                        }
                    }
                }
            }
        }
        Ok(Packed {
            bytes,
            layout: Layout::Mxfp4MoeFragmentScales(self),
        })
    }
}

/// Row-major bytes cut into strips of `tile_rows` rows, each strip stored as
/// its tiles of `tile_bytes` columns in column order and every tile
/// row-major. A strip covers the same bytes in both orders, so the
/// permutation never leaves it. With 64 rows by 128 MXFP4 elements (64 code
/// bytes, 4 UE8M0 scale bytes) a row block of 64 rows is one contiguous
/// tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowTileLayout {
    pub rows: usize,
    pub row_bytes: usize,
    pub tile_rows: usize,
    pub tile_bytes: usize,
}

impl RowTileLayout {
    /// The layout's bytes, when it holds whole tiles.
    fn bytes(self) -> Result<usize, Refusal> {
        if self.rows == 0
            || self.row_bytes == 0
            || self.tile_rows == 0
            || self.tile_bytes == 0
            || !self.rows.is_multiple_of(self.tile_rows)
            || !self.row_bytes.is_multiple_of(self.tile_bytes)
        {
            return Err(Refusal("row tiles need whole tiles of rows and bytes"));
        }
        self.rows
            .checked_mul(self.row_bytes)
            .ok_or(Refusal("row tile size overflow"))
    }

    /// Where byte `column` of row `row` is stored.
    #[must_use]
    pub const fn offset(self, row: usize, column: usize) -> usize {
        (row / self.tile_rows) * self.tile_rows * self.row_bytes
            + (column / self.tile_bytes) * self.tile_rows * self.tile_bytes
            + (row % self.tile_rows) * self.tile_bytes
            + column % self.tile_bytes
    }

    /// Tiles row-major `bytes` in place, one strip at a time through
    /// `scratch`, which holds one strip afterwards.
    pub fn pack_in_place(self, bytes: &mut [u8], scratch: &mut Vec<u8>) -> Result<(), Refusal> {
        if bytes.len() != self.bytes()? {
            return Err(Refusal("row tile source size mismatch"));
        }
        let strip = self.tile_rows * self.row_bytes;
        for out in bytes.chunks_exact_mut(strip) {
            scratch.clear();
            scratch.extend_from_slice(out);
            match self.tile_bytes {
                4 => tile_strip::<4>(self.tile_rows, self.row_bytes, scratch, out),
                64 => tile_strip::<64>(self.tile_rows, self.row_bytes, scratch, out),
                width => {
                    for (row, source) in scratch.chunks_exact(self.row_bytes).enumerate() {
                        for (tile, chunk) in source.chunks_exact(width).enumerate() {
                            let at = (tile * self.tile_rows + row) * width;
                            out[at..at + width].copy_from_slice(chunk);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// One strip's row-major `rows` in `source` stored as `N`-byte-wide tiles
/// in `out`; a constant width keeps each copy a few register moves.
fn tile_strip<const N: usize>(rows: usize, row_bytes: usize, source: &[u8], out: &mut [u8]) {
    for (row, source) in source.chunks_exact(row_bytes).enumerate() {
        for (tile, chunk) in source.chunks_exact(N).enumerate() {
            let at = (tile * rows + row) * N;
            out[at..at + N].copy_from_slice(chunk);
        }
    }
}

impl Transform for RowTileLayout {
    fn pack(&self, source: &[u8]) -> Result<Packed, Refusal> {
        let mut bytes = source.to_vec();
        self.pack_in_place(&mut bytes, &mut Vec::new())?;
        Ok(Packed {
            bytes,
            layout: Layout::RowTiles(*self),
        })
    }

    fn unpack(&self, packed: &Packed) -> Result<Vec<u8>, Refusal> {
        if packed.layout != Layout::RowTiles(*self) || packed.bytes.len() != self.bytes()? {
            return Err(Refusal("row tile packed layout mismatch"));
        }
        let mut source = vec![0; packed.bytes.len()];
        for row in 0..self.rows {
            for column in 0..self.row_bytes {
                source[row * self.row_bytes + column] = packed.bytes[self.offset(row, column)];
            }
        }
        Ok(source)
    }
}

/// Packed MXFP4 code bytes in 64-byte blocks (one row's 128-K block, as row
/// tiles store it), each block reordered so the bytes one lane of a quad
/// converts for its m16n8k16 A fragments are adjacent. Byte
/// `16 * g + 4 * q + t` (32-K group `g`, word `q` of the group, lane `t`) is
/// stored at `16 * t + 4 * g + q`. Lane `t`'s 16-byte load at `16 * t` then
/// holds group `g`'s four bytes in word `g`, byte `q` of the word. The
/// permutation stays inside each block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuadLaneLayout {
    pub bytes: usize,
}

impl QuadLaneLayout {
    /// Bytes per block: 128 MXFP4 codes.
    pub const BLOCK: usize = 64;
    /// Where byte `within` of a block is stored in it.
    const ORDER: [u8; 64] = {
        let mut order = [0; 64];
        let mut at = 0;
        while at < 64 {
            order[at] = Self::offset(at) as u8;
            at += 1;
        }
        order
    };

    /// Where source byte `at` is stored.
    #[must_use]
    pub const fn offset(at: usize) -> usize {
        let within = at % Self::BLOCK;
        at - within + (within % 4) * 16 + (within / 16) * 4 + (within / 4) % 4
    }

    fn check(self, bytes: &[u8]) -> Result<(), Refusal> {
        if self.bytes == 0 || !self.bytes.is_multiple_of(Self::BLOCK) {
            return Err(Refusal("quad lane order needs whole 64-byte blocks"));
        }
        if bytes.len() != self.bytes {
            return Err(Refusal("quad lane order size mismatch"));
        }
        Ok(())
    }

    /// Reorders source-order `bytes` in place.
    pub fn pack_in_place(self, bytes: &mut [u8]) -> Result<(), Refusal> {
        self.check(bytes)?;
        for block in bytes.chunks_exact_mut(Self::BLOCK) {
            let source: [u8; Self::BLOCK] = block.try_into().unwrap();
            for (byte, &to) in source.iter().zip(&Self::ORDER) {
                block[usize::from(to)] = *byte;
            }
        }
        Ok(())
    }

    /// Restores the source order of packed `bytes` in place.
    pub fn unpack_in_place(self, bytes: &mut [u8]) -> Result<(), Refusal> {
        self.check(bytes)?;
        for block in bytes.chunks_exact_mut(Self::BLOCK) {
            let packed: [u8; Self::BLOCK] = block.try_into().unwrap();
            for (byte, &from) in block.iter_mut().zip(&Self::ORDER) {
                *byte = packed[usize::from(from)];
            }
        }
        Ok(())
    }
}

impl Transform for QuadLaneLayout {
    fn pack(&self, source: &[u8]) -> Result<Packed, Refusal> {
        let mut bytes = source.to_vec();
        self.pack_in_place(&mut bytes)?;
        Ok(Packed {
            bytes,
            layout: Layout::QuadLanes(*self),
        })
    }

    fn unpack(&self, packed: &Packed) -> Result<Vec<u8>, Refusal> {
        if packed.layout != Layout::QuadLanes(*self) {
            return Err(Refusal("quad lane packed layout mismatch"));
        }
        let mut source = packed.bytes.clone();
        self.unpack_in_place(&mut source)?;
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mxfp4_moe_fragment_tiles_round_trip_every_byte_and_lane() {
        for experts in [1, 2, 3] {
            for rows_per_expert in [16, 32] {
                for k in [64, 128, 256] {
                    let layout = Mxfp4MoeFragmentLayout {
                        experts,
                        rows_per_expert,
                        k,
                    };
                    let code_len = experts * rows_per_expert * k / 2;
                    let codes = (0..code_len).map(|i| i as u8).collect::<Vec<_>>();
                    let packed = layout.pack_codes(&codes).unwrap();
                    let k_tiles = k / 64;
                    for strip in 0..experts * rows_per_expert / 16 {
                        for kt in 0..k_tiles {
                            for lane in 0..32 {
                                for step in 0..4 {
                                    for pair in 0..4 {
                                        assert_eq!(
                                            packed.bytes[(strip * k_tiles + kt) * 512
                                                + lane * 16
                                                + step * 4
                                                + pair],
                                            codes[layout.code_index(strip, kt, lane, step, pair)]
                                        );
                                    }
                                }
                            }
                        }
                    }
                    let scale_len = experts * rows_per_expert * k / 32;
                    for seed in 0..=255_u8 {
                        let scales = (0..scale_len)
                            .map(|i| (i as u8).wrapping_add(seed))
                            .collect::<Vec<_>>();
                        let packed_scales = layout.pack_scales(&scales).unwrap();
                        assert_eq!(
                            packed_scales.bytes.len(),
                            experts * rows_per_expert * k / 32
                        );
                        for strip in 0..experts * rows_per_expert / 16 {
                            for kt in 0..k_tiles {
                                for g in 0..8 {
                                    for block in 0..2 {
                                        for high_row in 0..2 {
                                            let row = strip * 16 + g + high_row * 8;
                                            assert_eq!(
                                                packed_scales.bytes[(strip * k_tiles + kt) * 32
                                                    + g * 4
                                                    + block * 2
                                                    + high_row],
                                                scales[row * (k / 32) + kt * 2 + block]
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn mxfp4_moe_fragment_layout_refuses_bad_geometry_and_scale_size() {
        let bad = Mxfp4MoeFragmentLayout {
            experts: 1,
            rows_per_expert: 15,
            k: 64,
        };
        assert!(bad.pack_codes(&[]).is_err());
        let layout = Mxfp4MoeFragmentLayout {
            experts: 2,
            rows_per_expert: 16,
            k: 64,
        };
        assert!(layout.pack_codes(&[0; 511]).is_err());
        let scales = layout.pack_scales(&[127; 64]).unwrap();
        assert_eq!(scales.bytes.len(), 64);
    }
    /// Each 64-row strip of a 1152-row expert (gate then up) at K 5120 is
    /// stored as forty 4096-byte code tiles and forty 256-byte scale tiles,
    /// the order the tiled routed kernels address: byte `b` of tile
    /// `strip * 40 + block` is row `b / width` of the strip, byte
    /// `block * width + b % width` of that row.
    #[test]
    fn row_tiles_store_each_strip_as_its_k_blocks() {
        for (row_bytes, width) in [(2560, 64), (160, 4)] {
            let layout = RowTileLayout {
                rows: 1152,
                row_bytes,
                tile_rows: 64,
                tile_bytes: width,
            };
            let source = (0..1152 * row_bytes)
                .map(|i| (i * 131 + i / 7919) as u8)
                .collect::<Vec<_>>();
            let packed = layout.pack(&source).unwrap();
            let tile = 64 * width;
            for (at, byte) in packed.bytes.iter().enumerate() {
                let (strip, block) = (at / tile / 40, at / tile % 40);
                let row = strip * 64 + at % tile / width;
                assert_eq!(*byte, source[row * row_bytes + block * width + at % width]);
            }
            assert_eq!(layout.unpack(&packed).unwrap(), source);
        }
    }

    #[test]
    fn row_tiles_round_trip_every_width_in_reused_scratch() {
        let mut scratch = vec![0xa5; 3];
        for (rows, tile_rows) in [(4, 4), (12, 4), (128, 64)] {
            for (row_bytes, tile_bytes) in [(8, 4), (12, 3), (128, 64), (64, 64), (6, 1)] {
                let layout = RowTileLayout {
                    rows,
                    row_bytes,
                    tile_rows,
                    tile_bytes,
                };
                let source = (0..rows * row_bytes)
                    .map(|i| (i * 37 + 11) as u8)
                    .collect::<Vec<_>>();
                let mut bytes = source.clone();
                layout.pack_in_place(&mut bytes, &mut scratch).unwrap();
                for row in 0..rows {
                    for column in 0..row_bytes {
                        assert_eq!(
                            bytes[layout.offset(row, column)],
                            source[row * row_bytes + column]
                        );
                    }
                }
                let packed = layout.pack(&source).unwrap();
                assert_eq!(packed.bytes, bytes);
                assert_eq!(layout.unpack(&packed).unwrap(), source);
            }
        }
    }

    #[test]
    fn row_tiles_refuse_partial_tiles_and_other_sizes() {
        let layout = RowTileLayout {
            rows: 128,
            row_bytes: 128,
            tile_rows: 64,
            tile_bytes: 64,
        };
        let mut scratch = Vec::new();
        for bad in [
            RowTileLayout { rows: 96, ..layout },
            RowTileLayout {
                row_bytes: 96,
                ..layout
            },
            RowTileLayout {
                tile_bytes: 0,
                ..layout
            },
        ] {
            let mut bytes = vec![0; bad.rows * bad.row_bytes];
            assert!(bad.pack_in_place(&mut bytes, &mut scratch).is_err());
        }
        assert!(
            layout
                .pack_in_place(&mut [0; 128 * 128 - 1], &mut scratch)
                .is_err()
        );
        let packed = layout.pack(&[0; 128 * 128]).unwrap();
        let other = RowTileLayout {
            tile_rows: 32,
            ..layout
        };
        assert!(other.unpack(&packed).is_err());
    }

    /// Lane `t`'s word `g` holds bytes `16 * g + 4 * q + t` of each block, `q`
    /// in byte order, and the inverse restores every block.
    #[test]
    fn quad_lanes_give_each_lane_its_fragment_bytes_and_invert() {
        let layout = QuadLaneLayout { bytes: 3 * 64 };
        let source = (0..3 * 64).map(|i| (i * 89 + 7) as u8).collect::<Vec<_>>();
        let packed = layout.pack(&source).unwrap();
        for block in 0..3 {
            for lane in 0..4 {
                for group in 0..4 {
                    for word_byte in 0..4 {
                        assert_eq!(
                            packed.bytes[block * 64 + lane * 16 + group * 4 + word_byte],
                            source[block * 64 + group * 16 + word_byte * 4 + lane]
                        );
                    }
                }
            }
        }
        let mut order = (0..64).map(QuadLaneLayout::offset).collect::<Vec<_>>();
        order.sort_unstable();
        assert_eq!(order, (0..64).collect::<Vec<_>>());
        assert_eq!(layout.unpack(&packed).unwrap(), source);
        let mut bytes = packed.bytes.clone();
        layout.unpack_in_place(&mut bytes).unwrap();
        assert_eq!(bytes, source);
        assert_ne!(packed.bytes, source);
        for bad in [0, 65] {
            assert!(QuadLaneLayout { bytes: bad }.pack(&vec![0; bad]).is_err());
        }
        assert!(layout.pack(&source[..128]).is_err());
        assert!(QuadLaneLayout { bytes: 128 }.unpack(&packed).is_err());
    }

    #[test]
    fn scale_tiles_round_trip() {
        let src: Vec<u8> = (0..128).collect();
        let tile = ScaleTileLayout {
            rows: 8,
            k_blocks: 16,
            tile_rows: 8,
            k_blocks_per_tile: 16,
        };
        assert_eq!(tile.unpack(&tile.pack(&src).unwrap()).unwrap(), src);
        assert_eq!(tile.pack(&src).unwrap().bytes[4], src[1]);
    }
    #[test]
    fn chunked_qkv_scale_rows() {
        for (rows, grid_rows, expected) in [(3392, 108, 27), (3712, 116, 29)] {
            let layout = ChunkedScaleGrid {
                chunks: 4,
                rows_per_chunk: rows,
                k_blocks: 32,
                block_rows: 128,
            };
            assert_eq!(layout.scale_rows_per_chunk(), expected);
            assert_eq!(layout.total_scale_rows(), grid_rows);
            assert_eq!(layout.scale_row(rows).unwrap(), expected);
            assert_eq!(layout.scale_row(rows - 1).unwrap(), expected - 1);
            let source = vec![7; grid_rows * 32];
            assert_eq!(layout.split_chunks(&source).unwrap().concat(), source);
            let flat = layout.flatten(&source).unwrap();
            assert_eq!(flat.weight_row_to_scale_row[rows], expected);
        }
    }
}

#[cfg(test)]
mod checkpoint_tests;
