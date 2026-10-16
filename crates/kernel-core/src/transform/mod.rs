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
