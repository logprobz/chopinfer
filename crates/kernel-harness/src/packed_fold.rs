//! Bitwise parity of an op's launches on row-major E4M3 weight codes with
//! its packed entries' launches on the same codes stored as
//! `mxfp8-fold-packed-8x128-v1` (`kernels_sm120::gemm::mxfp8_fold_packed`).
//!
//! A packed entry changes only where it reads weight codes, so every byte it
//! writes (outputs, status, workspace and fold state) must equal the
//! row-major entry's. [`codes`] makes every four-byte word of a matrix
//! distinct, so a packed read from the wrong strip, group, fold or lane
//! changes some output. Correctness against an oracle stays with the
//! row-major entry's own conformance cases.

use crate::launch::{Bindings, launch};
use crate::{LabResult, Session};
use kernel_core::backend::Launch;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;
use runtime_cuda::{DeviceBuffer, Function};

/// One view the launches name, with the bytes it holds before the first.
pub struct Image {
    pub handle: u64,
    /// Row-major bytes; the row-major run binds these.
    pub bytes: Vec<u8>,
    /// For weight codes, the packed bytes the packed run binds instead.
    pub packed: Option<Vec<u8>>,
}

impl Image {
    pub fn data(handle: u64, bytes: Vec<u8>) -> Self {
        Self {
            handle,
            bytes,
            packed: None,
        }
    }

    /// Weight codes of `layout`, from [`codes`].
    pub fn codes(handle: u64, layout: Mxfp8FoldLayout, seed: u64) -> LabResult<Self> {
        let bytes = codes(layout.packed_bytes().map_err(|e| e.0)?, seed);
        let packed = layout.pack(&bytes).map_err(|e| e.0)?;
        Ok(Self {
            handle,
            bytes,
            packed: Some(packed),
        })
    }
}

fn mix(seed: u64, index: u64) -> u64 {
    let mut x = seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The finite E4M3 code (never 0x7f or 0xff) numbered `digit` below 254.
fn finite(digit: u64) -> u8 {
    let digit = (digit % 254) as u8;
    if digit < 127 { digit } else { digit + 1 }
}

/// Finite E4M3 codes. The first three bytes of each four-byte word number
/// it in base 254 and the last is pseudorandom, so the first 254^3 words
/// (65 MB) are all distinct and later ones repeat only by chance.
pub fn codes(bytes: usize, seed: u64) -> Vec<u8> {
    (0..bytes.div_ceil(4) as u64)
        .flat_map(|word| {
            let number = word % (254 * 254 * 254);
            [
                finite(number),
                finite(number / 254),
                finite(number / (254 * 254)),
                finite(mix(seed, word)),
            ]
        })
        .take(bytes)
        .collect()
}

/// UE8M0 scales from 2^-10 to 2^6, which keep products finite.
pub fn scales(bytes: usize, seed: u64) -> Vec<u8> {
    (0..bytes as u64)
        .map(|i| 117 + (mix(seed, i) % 17) as u8)
        .collect()
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len().max(1))?;
    crate::poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

fn download(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; buffer.byte_len()];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    bytes.truncate(len);
    Ok(bytes)
}

/// Runs `launches` on fresh buffers holding `images` (packed codes when
/// `packed`) and returns every buffer's final bytes.
pub fn run(
    session: &Session,
    function: &dyn Fn(&str) -> LabResult<Function>,
    launches: &[Launch],
    images: &[Image],
    packed: bool,
) -> LabResult<Vec<Vec<u8>>> {
    let initial = |image: &'_ Image| -> Vec<u8> {
        match &image.packed {
            Some(bytes) if packed => bytes.clone(),
            _ => image.bytes.clone(),
        }
    };
    let buffers = images
        .iter()
        .map(|image| upload(session, &initial(image)))
        .collect::<LabResult<Vec<_>>>()?;
    let bindings = Bindings {
        buffers: images
            .iter()
            .zip(&buffers)
            .map(|(image, buffer)| (image.handle, buffer))
            .collect(),
    };
    for call in launches {
        let function = function(&call.symbol)?;
        if call.plan.dynamic_shared_bytes > 48 << 10 {
            function
                .set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
                .map_err(|e| e.to_string())?;
        }
        launch(
            &session.context,
            &function,
            &session.stream,
            call.plan,
            &call.arguments,
            &bindings,
        )?;
    }
    session.stream.synchronize().map_err(|e| e.to_string())?;
    images
        .iter()
        .zip(&buffers)
        .map(|(image, buffer)| download(session, buffer, image.bytes.len()))
        .collect()
}

/// Runs `row_major` on row-major codes and `packed` on packed codes, then
/// requires every data view's bytes to match and every code view to be left
/// as it was. Errs at the first differing byte.
pub fn compare(
    session: &Session,
    function: &dyn Fn(&str) -> LabResult<Function>,
    row_major: &[Launch],
    packed: &[Launch],
    images: &[Image],
) -> LabResult<()> {
    let expected = run(session, function, row_major, images, false)?;
    let actual = run(session, function, packed, images, true)?;
    for ((image, expected), actual) in images.iter().zip(&expected).zip(&actual) {
        if let Some(codes) = &image.packed {
            if *expected != image.bytes || actual != codes {
                return Err(format!("weight codes {} were written", image.handle));
            }
            continue;
        }
        if let Some(at) = expected.iter().zip(actual).position(|(a, b)| a != b) {
            let differing = expected.iter().zip(actual).filter(|(a, b)| a != b).count();
            return Err(format!(
                "view {} byte {at}: row-major {:#04x}, packed {:#04x} ({differing} of {} bytes differ)",
                image.handle,
                expected[at],
                actual[at],
                expected.len()
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_finite_and_every_word_is_distinct() {
        let bytes = codes(1 << 20, 7);
        assert_eq!(bytes.len(), 1 << 20);
        assert!(bytes.iter().all(|b| b & 0x7f != 0x7f));
        let mut words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
            .collect();
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), 1 << 18);
        assert_ne!(codes(64, 1), codes(64, 2));
        assert_eq!(codes(6, 3).len(), 6);
        assert!(scales(4096, 5).iter().all(|s| (117..=133).contains(s)));
    }
}
