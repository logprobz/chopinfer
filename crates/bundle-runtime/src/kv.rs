//! Element formats of the serving KV pool.
//!
//! A bundle lists the KV layouts it can serve. The host declares one of them,
//! by ID and page size, when it creates an instance; the bundle then sizes
//! its planes by the format's element and passes each paged layer's K and V
//! scales to the KV-write and attention kernels. FP8 E4M3 planes store
//! `E4M3(value / scale)`, and the kernels multiply the scale back in.

use kernel_core::contract::DType;

/// One serving KV layout a bundle supports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvFormat {
    pub layout_id: &'static [u8],
    pub page_bytes: u64,
    /// `Bf16` or `Fp8E4M3`, the element of the K and V planes.
    pub element: DType,
    /// Per paged layer ordinal, the K and V scales. Empty means 1.0 on every
    /// layer, which is also the only scale a BF16 format may have.
    pub scales: &'static [[f32; 2]],
}

impl KvFormat {
    pub const fn fp8(&self) -> bool {
        matches!(self.element, DType::Fp8E4M3)
    }

    /// Bytes of one K or V element.
    pub const fn element_bytes(&self) -> u64 {
        if self.fp8() { 1 } else { 2 }
    }

    /// The K and V scales of paged layer `ordinal` as the F32 bits the
    /// kernels take.
    pub fn scale_bits(&self, ordinal: u64) -> [u32; 2] {
        let [k, v] = usize::try_from(ordinal)
            .ok()
            .and_then(|ordinal| self.scales.get(ordinal))
            .copied()
            .unwrap_or([1.0, 1.0]);
        [k.to_bits(), v.to_bits()]
    }

    /// Whether the scales are usable for `layers` paged layers.
    pub fn valid(&self, layers: u64) -> bool {
        let element = matches!(self.element, DType::Bf16 | DType::Fp8E4M3);
        let scales = if self.fp8() {
            self.scales.is_empty()
                || (self.scales.len() as u64 == layers
                    && self
                        .scales
                        .iter()
                        .flatten()
                        .all(|scale| scale.is_finite() && *scale > 0.0))
        } else {
            self.scales.is_empty()
        };
        element && scales && self.page_bytes > 0 && !self.layout_id.is_empty()
    }
}

/// The format among `formats` that the host declared by ID and page bytes.
pub fn select<'a>(
    formats: &'a [KvFormat],
    layout_id: &[u8],
    page_bytes: u64,
) -> Option<&'a KvFormat> {
    formats
        .iter()
        .find(|format| format.layout_id == layout_id && format.page_bytes == page_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FORMATS: [KvFormat; 2] = [
        KvFormat {
            layout_id: b"test.paged-bf16.v1",
            page_bytes: 4096,
            element: DType::Bf16,
            scales: &[],
        },
        KvFormat {
            layout_id: b"test.paged-fp8.v1",
            page_bytes: 2048,
            element: DType::Fp8E4M3,
            scales: &[[1.0, 0.25], [0.5, 2.0]],
        },
    ];

    #[test]
    fn declared_layout_selects_its_element_and_scales() {
        assert!(select(&FORMATS, b"test.paged-bf16.v1", 2048).is_none());
        assert!(select(&FORMATS, b"test.paged-fp16.v1", 4096).is_none());
        let bf16 = select(&FORMATS, b"test.paged-bf16.v1", 4096).unwrap();
        assert_eq!((bf16.fp8(), bf16.element_bytes()), (false, 2));
        assert_eq!(bf16.scale_bits(7), [1.0_f32.to_bits(); 2]);
        let fp8 = select(&FORMATS, b"test.paged-fp8.v1", 2048).unwrap();
        assert_eq!((fp8.fp8(), fp8.element_bytes()), (true, 1));
        assert_eq!(fp8.scale_bits(1), [0.5_f32.to_bits(), 2.0_f32.to_bits()]);
        assert!(FORMATS.iter().all(|format| format.valid(2)));
        assert!(!fp8.valid(3));
        let scaled_bf16 = KvFormat {
            scales: &[[1.0, 1.0]],
            ..FORMATS[0]
        };
        assert!(!scaled_bf16.valid(1));
        let zero = KvFormat {
            scales: &[[1.0, 0.0]],
            ..FORMATS[1]
        };
        assert!(!zero.valid(1));
    }
}
