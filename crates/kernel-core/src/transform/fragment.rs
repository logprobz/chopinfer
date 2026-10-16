//! Dense weight tiles in the exact per-lane MMA A-fragment order.

use super::{Layout, Packed, Transform};
use crate::contract::Refusal;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FragmentWeightFormat {
    Bf16,
    Fp8E4M3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FragmentWeightLayout {
    pub rows: usize,
    pub k: usize,
    pub format: FragmentWeightFormat,
}

impl FragmentWeightLayout {
    pub const fn tile_k(self) -> usize {
        match self.format {
            FragmentWeightFormat::Bf16 => 16,
            FragmentWeightFormat::Fp8E4M3 => 32,
        }
    }

    pub const fn element_bytes(self) -> usize {
        match self.format {
            FragmentWeightFormat::Bf16 => 2,
            FragmentWeightFormat::Fp8E4M3 => 1,
        }
    }

    pub fn packed_bytes(self) -> Result<usize, Refusal> {
        self.check()?;
        self.rows
            .div_ceil(16)
            .checked_mul(self.k / self.tile_k())
            .and_then(|tiles| tiles.checked_mul(512))
            .ok_or(Refusal("fragment-major weight size overflow"))
    }

    fn check(self) -> Result<(), Refusal> {
        if self.rows == 0 || self.k == 0 || !self.k.is_multiple_of(self.tile_k()) {
            return Err(Refusal("fragment-major K must align to its MMA tile"));
        }
        Ok(())
    }

    fn positions(self, lane: usize) -> impl Iterator<Item = (usize, usize)> {
        let g = lane >> 2;
        let t = lane & 3;
        let steps = self.tile_k() / 16;
        (0..steps).flat_map(move |step| {
            let col = step * 16 + 2 * t;
            [
                (g, col),
                (g, col + 1),
                (g + 8, col),
                (g + 8, col + 1),
                (g, col + 8),
                (g, col + 9),
                (g + 8, col + 8),
                (g + 8, col + 9),
            ]
        })
    }
}

impl Transform for FragmentWeightLayout {
    fn pack(&self, source: &[u8]) -> Result<Packed, Refusal> {
        self.check()?;
        let source_bytes = self
            .rows
            .checked_mul(self.k)
            .and_then(|elements| elements.checked_mul(self.element_bytes()))
            .ok_or(Refusal("fragment-major source size overflow"))?;
        if source.len() != source_bytes {
            return Err(Refusal("fragment-major source size mismatch"));
        }
        let tile_k = self.tile_k();
        let element_bytes = self.element_bytes();
        let k_tiles = self.k / tile_k;
        let mut bytes = vec![0_u8; self.packed_bytes()?];
        for strip in 0..self.rows.div_ceil(16) {
            for kt in 0..k_tiles {
                let tile_base = (strip * k_tiles + kt) * 512;
                for lane in 0..32 {
                    for (item, (local_row, local_k)) in self.positions(lane).enumerate() {
                        let row = strip * 16 + local_row;
                        if row >= self.rows {
                            continue;
                        }
                        let source_offset = (row * self.k + kt * tile_k + local_k) * element_bytes;
                        let target_offset = tile_base + lane * 16 + item * element_bytes;
                        bytes[target_offset..target_offset + element_bytes]
                            .copy_from_slice(&source[source_offset..source_offset + element_bytes]);
                    }
                }
            }
        }
        Ok(Packed {
            bytes,
            layout: Layout::FragmentMajor(*self),
        })
    }

    fn unpack(&self, packed: &Packed) -> Result<Vec<u8>, Refusal> {
        let packed_bytes = self.packed_bytes()?;
        if packed.layout != Layout::FragmentMajor(*self) || packed.bytes.len() != packed_bytes {
            return Err(Refusal("fragment-major packed layout mismatch"));
        }
        let tile_k = self.tile_k();
        let element_bytes = self.element_bytes();
        let k_tiles = self.k / tile_k;
        let mut source = vec![0_u8; self.rows * self.k * element_bytes];
        for strip in 0..self.rows.div_ceil(16) {
            for kt in 0..k_tiles {
                let tile_base = (strip * k_tiles + kt) * 512;
                for lane in 0..32 {
                    for (item, (local_row, local_k)) in self.positions(lane).enumerate() {
                        let row = strip * 16 + local_row;
                        if row >= self.rows {
                            continue;
                        }
                        let target_offset = (row * self.k + kt * tile_k + local_k) * element_bytes;
                        let source_offset = tile_base + lane * 16 + item * element_bytes;
                        source[target_offset..target_offset + element_bytes].copy_from_slice(
                            &packed.bytes[source_offset..source_offset + element_bytes],
                        );
                    }
                }
            }
        }
        Ok(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_fragment_coordinate_round_trips() {
        for format in [FragmentWeightFormat::Bf16, FragmentWeightFormat::Fp8E4M3] {
            let tile_k = FragmentWeightLayout {
                rows: 1,
                k: 16,
                format,
            }
            .tile_k();
            for rows in 1..=33 {
                for k_tiles in 1..=5 {
                    let layout = FragmentWeightLayout {
                        rows,
                        k: k_tiles * tile_k,
                        format,
                    };
                    let source = (0..rows * layout.k * layout.element_bytes())
                        .map(|index| (index.wrapping_mul(113) ^ (index >> 3)) as u8)
                        .collect::<Vec<_>>();
                    let packed = layout.pack(&source).unwrap();
                    assert_eq!(packed.bytes.len(), layout.packed_bytes().unwrap());
                    assert_eq!(layout.unpack(&packed).unwrap(), source, "{layout:?}");
                    for strip in 0..rows.div_ceil(16) {
                        for kt in 0..k_tiles {
                            for lane in 0..32 {
                                for (item, (lr, lk)) in layout.positions(lane).enumerate() {
                                    let packed_offset = ((strip * k_tiles + kt) * 32 + lane) * 16
                                        + item * layout.element_bytes();
                                    let row = strip * 16 + lr;
                                    let expected = if row < rows {
                                        let offset = (row * layout.k + kt * tile_k + lk)
                                            * layout.element_bytes();
                                        &source[offset..offset + layout.element_bytes()]
                                    } else {
                                        &[0_u8; 2][..layout.element_bytes()]
                                    };
                                    assert_eq!(
                                        &packed.bytes
                                            [packed_offset..packed_offset + layout.element_bytes()],
                                        expected,
                                        "{layout:?} strip={strip} kt={kt} lane={lane} item={item}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_wrong_geometry_and_tag() {
        let layout = FragmentWeightLayout {
            rows: 17,
            k: 32,
            format: FragmentWeightFormat::Fp8E4M3,
        };
        assert!(layout.pack(&[0; 16]).is_err());
        let mut packed = layout.pack(&vec![0; 17 * 32]).unwrap();
        packed.layout = Layout::RowMajor {
            rows: 17,
            bytes_per_row: 32,
        };
        assert!(layout.unpack(&packed).is_err());
        assert!(
            FragmentWeightLayout { k: 17, ..layout }
                .packed_bytes()
                .is_err()
        );
    }
}
