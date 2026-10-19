use kernel_core::codec::f32_to_bf16;
use kernel_harness::{LabResult, Session, kit::bf16_bytes};
use runtime_cuda::DeviceBuffer;

pub const ROWS: u32 = 8192;
pub const HEADS: usize = 16;
pub const WIDTH: usize = 512;
pub const WINDOW: u32 = 128;
pub const SLOTS: u32 = ROWS + WINDOW - 1;
pub const CONTEXT: u32 = 32768;

pub fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len().max(4))?;
    buffer
        .copy_from_bytes(bytes, &s.stream)
        .map_err(|e| e.to_string())?;
    Ok(buffer)
}
fn integers(values: impl Iterator<Item = i32>) -> Vec<u8> {
    values.flat_map(i32::to_le_bytes).collect()
}
pub fn query(rows: u32, seed: u32) -> Vec<u8> {
    bf16_bytes(
        &(0..rows as usize * HEADS * WIDTH)
            .map(|i| {
                let x = (i as u32)
                    .wrapping_mul(1664525)
                    .wrapping_add(seed.wrapping_mul(1013904223));
                let value = if i % 127 == 0 {
                    -0.0
                } else {
                    ((x >> 17) % 511) as f32 / 256.0 - 1.0
                };
                f32_to_bf16(value * [0.015625, 1.0, 8.0, 64.0][(i / WIDTH) % 4])
            })
            .collect::<Vec<_>>(),
    )
}

pub struct Inputs {
    pub q: DeviceBuffer,
    pub row_batch: DeviceBuffer,
    pub live: DeviceBuffer,
    pub window_codes: DeviceBuffer,
    pub window_scales: DeviceBuffer,
    pub item_slots: DeviceBuffer,
    pub window_keys: DeviceBuffer,
    pub compressed_pool: DeviceBuffer,
    pub compressed_keys: DeviceBuffer,
    pub tables: DeviceBuffer,
    pub sinks: DeviceBuffer,
    pub cos_sin: DeviceBuffer,
    pub page: u32,
    pub pages: u32,
    pub compressed_budget: u32,
}

impl Inputs {
    pub fn new(s: &Session, page: u32, second: bool, seed: u32) -> LabResult<Self> {
        let pages = CONTEXT / page;
        let stride = page as usize * (WIDTH / 2 + WIDTH / 16);
        let codes = [0x00, 0x80, 0x38, 0xb8, 0x40, 0xc0, 0x18, 0xb4];
        let window_codes = (0..SLOTS as usize * WIDTH)
            .map(|i| codes[(i * 17 + i / WIDTH + seed as usize) % codes.len()])
            .collect::<Vec<_>>();
        let window_scales = (0..SLOTS as usize * (WIDTH / 32))
            .map(|i| (121 + (i * 7 + i / 16) % 8) as u8)
            .collect::<Vec<_>>();
        let mut compressed_pool = vec![0u8; pages as usize * stride];
        for (p, data) in compressed_pool.chunks_exact_mut(stride).enumerate() {
            let (codes, scales) = data.split_at_mut(page as usize * WIDTH / 2);
            for (i, value) in codes.iter_mut().enumerate() {
                *value = (((i + p) % 16) | (((i * 7 + p + 3) % 16) << 4)) as u8;
            }
            for (i, value) in scales.iter_mut().enumerate() {
                *value = [0x20, 0x28, 0x30, 0x38, 0x40, 0x48][(i + p) % 6];
            }
        }
        let inputs = Self {
            q: upload(s, &query(ROWS, seed))?,
            row_batch: upload(s, &vec![0; ROWS as usize * 4])?,
            live: upload(s, &ROWS.to_le_bytes())?,
            window_codes: upload(s, &window_codes)?,
            window_scales: upload(s, &window_scales)?,
            item_slots: upload(s, &0_i32.to_le_bytes())?,
            window_keys: upload(s, &vec![0; ROWS as usize * WINDOW as usize * 4])?,
            compressed_pool: upload(s, &compressed_pool)?,
            compressed_keys: upload(s, &vec![0; ROWS as usize * 512 * 4])?,
            tables: upload(s, &integers((0..pages as i32).rev()))?,
            sinks: s.upload_f32(&(0..HEADS).map(|i| i as f32 * 0.125).collect::<Vec<_>>())?,
            cos_sin: s.upload_f32(&[0.6, 0.8].repeat(ROWS as usize * 32))?,
            page,
            pages,
            compressed_budget: if second { 512 } else { 0 },
        };
        inputs.set_case(s, seed, false, ROWS)?;
        Ok(inputs)
    }

    pub fn set_case(&self, s: &Session, seed: u32, adversarial: bool, live: u32) -> LabResult<()> {
        let write = |buffer: &DeviceBuffer, bytes: &[u8]| {
            buffer
                .copy_from_bytes(bytes, &s.stream)
                .map_err(|e| e.to_string())
        };
        // Row zero is deliberately all holes in adversarial cases. Change a
        // live nonempty query too, so replay checks cannot reuse old Q bits.
        write(&self.q, &query(2, seed))?;
        write(&self.live, &live.to_le_bytes())?;
        write(
            &self.item_slots,
            &(if adversarial && seed % 2 == 1 {
                -1_i32
            } else {
                0_i32
            })
            .to_le_bytes(),
        )?;
        write(
            &self.row_batch,
            &integers((0..ROWS).map(|r| {
                if adversarial && r % 31 == 2 {
                    -1
                } else if adversarial && r % 31 == 3 {
                    1
                } else {
                    0
                }
            })),
        )?;
        write(
            &self.window_keys,
            &integers((0..ROWS).flat_map(|row| {
                (0..WINDOW).map(move |key| {
                    if adversarial && (row % 31 == 0 || key % 7 == 6) {
                        -1
                    } else if adversarial && key == 63 {
                        SLOTS as i32
                    } else {
                        (row + key) as i32
                    }
                })
            })),
        )?;
        write(
            &self.compressed_keys,
            &integers((0..ROWS).flat_map(|row| {
                (0..512).map(move |key| {
                    if adversarial && (row % 31 == 0 || key % 5 == 4) {
                        -1
                    } else if adversarial && key == 64 {
                        CONTEXT as i32
                    } else {
                        ((key * 47 + row * 13 + seed) % (CONTEXT - ROWS + row + 1)) as i32
                    }
                })
            })),
        )?;
        write(
            &self.tables,
            &integers((0..self.pages).map(|p| {
                if adversarial && p % 29 == 0 {
                    -1
                } else if adversarial && p % 29 == 1 {
                    self.pages as i32
                } else {
                    (self.pages - 1 - p) as i32
                }
            })),
        )?;
        Ok(())
    }
}
