//! Shared GPU test helpers: a deterministic generator, device upload and
//! download, and native-endian byte packing.
#![allow(dead_code)] // Shared by test binaries that use different parts.

use kernel_lab::{LabResult, Session};
use runtime_cuda::DeviceBuffer;

fn err(e: impl ToString) -> String {
    e.to_string()
}

/// A deterministic generator (xorshift64*).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// Spanning 2^-8..2^8 with a random sign.
    pub fn wide(&mut self) -> f32 {
        let exponent = (self.next() % 17) as i32 - 8;
        let sign = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        sign * (1.0 + 0.5 * self.unit().abs()) * 2_f32.powi(exponent)
    }
}

/// A device copy of `data`, at least 16 bytes long.
pub fn upload(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len().max(16))?;
    buffer.copy_from_bytes(data, &session.stream).map_err(err)?;
    Ok(buffer)
}

/// The first `len` bytes of `buffer`, after the stream's earlier work.
pub fn download(session: &Session, buffer: &DeviceBuffer, len: usize) -> LabResult<Vec<u8>> {
    let mut out = vec![0; len];
    buffer
        .copy_to_bytes_at_async(0, &mut out, &session.stream)
        .map_err(err)?;
    session.stream.synchronize().map_err(err)?;
    Ok(out)
}

pub fn bytes<T: Copy, const N: usize>(values: &[T], to: fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| to(v)).collect()
}

pub fn bytes_u16(values: &[u16]) -> Vec<u8> {
    bytes(values, u16::to_ne_bytes)
}

pub fn bytes_i32(values: &[i32]) -> Vec<u8> {
    bytes(values, i32::to_ne_bytes)
}

pub fn bytes_f32(values: &[f32]) -> Vec<u8> {
    bytes(values, f32::to_ne_bytes)
}
