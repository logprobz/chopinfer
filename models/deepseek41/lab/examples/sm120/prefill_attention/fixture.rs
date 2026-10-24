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
    query_with_heads(rows, HEADS, seed)
}
pub fn query_with_heads(rows: u32, heads: usize, seed: u32) -> Vec<u8> {
    bf16_bytes(
        &(0..rows as usize * heads * WIDTH)
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
    pub capacity: u32,
    pub heads: usize,
    /// Window state slots, items and compressed sequences: one per packed
    /// prompt.
    pub items: u32,
}

impl Inputs {
    pub fn share_selected_pairs(&self, s: &Session, seed: u32, shifted: bool) -> LabResult<()> {
        let keys = integers((0..ROWS).flat_map(|row| {
            (0..512).map(move |key| {
                compressed_key(row & !1, key + u32::from(shifted) * (row % 2), seed, false)
            })
        }));
        self.compressed_keys
            .copy_from_bytes(&keys, &s.stream)
            .map_err(|e| e.to_string())
    }

    pub fn new(s: &Session, page: u32, second: bool, seed: u32) -> LabResult<Self> {
        Self::with_shape(s, page, second, seed, HEADS, ROWS)
    }
    pub fn with_shape(
        s: &Session,
        page: u32,
        second: bool,
        seed: u32,
        heads: usize,
        rows: u32,
    ) -> LabResult<Self> {
        if rows < 2 || ![16, 32, 64].contains(&heads) {
            return Err("unsupported lab head/row shape".into());
        }
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
            q: upload(s, &query_with_heads(rows, heads, seed))?,
            row_batch: upload(s, &vec![0; rows as usize * 4])?,
            live: upload(s, &rows.to_le_bytes())?,
            window_codes: upload(s, &window_codes)?,
            window_scales: upload(s, &window_scales)?,
            item_slots: upload(s, &0_i32.to_le_bytes())?,
            window_keys: upload(s, &vec![0; rows as usize * WINDOW as usize * 4])?,
            compressed_pool: upload(s, &compressed_pool)?,
            compressed_keys: upload(s, &vec![0; rows as usize * 512 * 4])?,
            tables: upload(s, &integers((0..pages as i32).rev()))?,
            sinks: s.upload_f32(&(0..heads).map(|i| i as f32 * 0.125).collect::<Vec<_>>())?,
            cos_sin: s.upload_f32(&[0.6, 0.8].repeat(rows as usize * 32))?,
            page,
            pages,
            compressed_budget: if second { 512 } else { 0 },
            capacity: rows,
            heads,
            items: 1,
        };
        inputs.set_case(s, seed, false, rows)?;
        Ok(inputs)
    }

    /// Prompts of `lengths` rows packed in order into one prefill step, as
    /// the host packs them: each item has its own window ring, state slot,
    /// block table and keys at its own positions. Adversarial cases add
    /// holes, out-of-range keys and pages, an item without window state and
    /// rows without an item.
    pub fn packed(
        s: &Session,
        page: u32,
        second: bool,
        seed: u32,
        lengths: &[u32],
        adversarial: bool,
    ) -> LabResult<Self> {
        let rows = lengths.iter().sum::<u32>();
        if lengths.is_empty() || lengths.contains(&0) || rows > ROWS {
            return Err("packed prompts must be nonempty and fit the chunk".into());
        }
        let items = lengths.len() as u32;
        let pages = CONTEXT / page;
        let stride = page as usize * (WIDTH / 2 + WIDTH / 16);
        let codes = [0x00, 0x80, 0x38, 0xb8, 0x40, 0xc0, 0x18, 0xb4];
        let window_codes = (0..items as usize * SLOTS as usize * WIDTH)
            .map(|i| codes[(i * 17 + i / WIDTH + seed as usize) % codes.len()])
            .collect::<Vec<_>>();
        let window_scales = (0..items as usize * SLOTS as usize * (WIDTH / 32))
            .map(|i| (121 + (i * 7 + i / 16 + i / (SLOTS as usize * 16)) % 8) as u8)
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
        // (item, row within the item) of every packed row.
        let owners = lengths
            .iter()
            .enumerate()
            .flat_map(|(item, &length)| (0..length).map(move |local| (item as u32, local)))
            .collect::<Vec<_>>();
        let row_batch = integers(owners.iter().enumerate().map(|(row, &(item, _))| {
            if adversarial && row % 37 == 5 {
                -1
            } else {
                item as i32
            }
        }));
        let item_slots = integers((0..items).map(|item| {
            if adversarial && item == items / 2 && items > 1 {
                -1
            } else {
                item as i32
            }
        }));
        let window_keys = integers(owners.iter().flat_map(|&(_, local)| {
            (0..WINDOW).map(move |key| window_key(local, key, adversarial))
        }));
        let compressed_keys = integers(owners.iter().flat_map(|&(item, local)| {
            (0..512).map(move |key| compressed_key(local, key, seed + 97 * item, adversarial))
        }));
        // A distinct page order per item; adversarial tables add holes and
        // out-of-range pages.
        let tables = integers((0..items).flat_map(|item| {
            (0..pages).map(move |p| {
                if adversarial && p % 29 == item % 29 {
                    -1
                } else if adversarial && p % 31 == 1 {
                    pages as i32
                } else {
                    ((p * 5 + item * 37) % pages) as i32
                }
            })
        }));
        Ok(Self {
            q: upload(s, &query_with_heads(rows.max(2), HEADS, seed))?,
            row_batch: upload(s, &row_batch)?,
            live: upload(s, &rows.to_le_bytes())?,
            window_codes: upload(s, &window_codes)?,
            window_scales: upload(s, &window_scales)?,
            item_slots: upload(s, &item_slots)?,
            window_keys: upload(s, &window_keys)?,
            compressed_pool: upload(s, &compressed_pool)?,
            compressed_keys: upload(s, &compressed_keys)?,
            tables: upload(s, &tables)?,
            sinks: s.upload_f32(&(0..HEADS).map(|i| i as f32 * 0.125).collect::<Vec<_>>())?,
            cos_sin: s.upload_f32(&[0.6, 0.8].repeat(rows.max(2) as usize * 32))?,
            page,
            pages,
            compressed_budget: if second { 512 } else { 0 },
            capacity: rows,
            heads: HEADS,
            items,
        })
    }

    pub fn set_case(&self, s: &Session, seed: u32, adversarial: bool, live: u32) -> LabResult<()> {
        let write = |buffer: &DeviceBuffer, bytes: &[u8]| {
            buffer
                .copy_from_bytes(bytes, &s.stream)
                .map_err(|e| e.to_string())
        };
        // Row zero is deliberately all holes in adversarial cases. Change a
        // live nonempty query too, so replay checks cannot reuse old Q bits.
        write(&self.q, &query_with_heads(2, self.heads, seed))?;
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
            &integers((0..self.capacity).map(|r| {
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
            &integers(
                (0..self.capacity)
                    .flat_map(|row| (0..WINDOW).map(move |key| window_key(row, key, adversarial))),
            ),
        )?;
        write(
            &self.compressed_keys,
            &integers((0..self.capacity).flat_map(|row| {
                (0..512).map(move |key| compressed_key(row, key, seed, adversarial))
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

fn window_key(row: u32, key: u32, adversarial: bool) -> i32 {
    if adversarial && (row % 31 == 0 || key % 7 == 6) {
        -1
    } else if adversarial && key == 63 {
        SLOTS as i32
    } else {
        (row + key) as i32
    }
}
fn compressed_key(row: u32, key: u32, seed: u32, adversarial: bool) -> i32 {
    if adversarial && row % 31 == 0 {
        -1
    } else if adversarial && key == 64 {
        CONTEXT as i32
    } else if adversarial && key % 5 == 4 {
        -1
    } else {
        ((key * 47 + row * 13 + seed) % (CONTEXT - ROWS + row + 1)) as i32
    }
}

// Unique encoded payload bytes, excluding caches, transaction granularity,
// repeated scale-load instructions and metadata. Logical positions suffice:
// the profile fixture's reversed physical page table is a bijection.
fn unions(rows: u32, seed: u32, group: u32) -> (u64, u64) {
    use std::collections::BTreeSet;
    let mut totals = (0, 0);
    for first in (0..rows).step_by(group as usize) {
        let mut window = BTreeSet::new();
        let mut compressed = BTreeSet::new();
        for row in first..(first + group).min(rows) {
            for key in 0..WINDOW {
                window.insert(window_key(row, key, false));
            }
            for key in 0..512 {
                compressed.insert(compressed_key(row, key, seed, false));
            }
        }
        totals.0 += window.len() as u64;
        totals.1 += compressed.len() as u64;
    }
    totals
}

pub fn traffic() -> LabResult<serde_json::Value> {
    let window_bytes = WIDTH as u64 + WIDTH as u64 / 32;
    let compressed_bytes = WIDTH as u64 / 2 + WIDTH as u64 / 16;
    let groups = [1, 8, 16, 64, ROWS].into_iter().map(|group| {
        let (window, compressed) = unions(ROWS, 120, group);
        serde_json::json!({
            "query_rows_per_group":group,
            "window_unique_rows_summed_over_groups":window,
            "compressed_unique_rows_summed_over_groups":compressed,
            "encoded_bytes_with_head_reuse":window * window_bytes + compressed * compressed_bytes,
            "encoded_bytes_with_two_head_groups":2 * (window * window_bytes + compressed * compressed_bytes),
        })
    }).collect::<Vec<_>>();
    Ok(serde_json::json!({
        "model":"unique encoded payload, not measured DRAM/L2 traffic",
        "rows":ROWS,"heads":HEADS,"width":WIDTH,"window_keys":WINDOW,"compressed_keys":512,"seed":120,
        "q_bytes":u64::from(ROWS) * HEADS as u64 * WIDTH as u64 * 2,
        "partial_write_bytes":u64::from(ROWS) * HEADS as u64 * (WIDTH as u64 + 2) * 4,
        "baseline_gather_passes":1,"stage32_gather_passes":2,"reuse":groups,
        "window_union_64_query_rows":{"positions":191,"bf16_bytes":191 * WIDTH * 2,
            "native_bytes":191 * window_bytes},
        "ideal_shared_tile_bytes":{"baseline_stores":u64::from(ROWS) * 2 * 640 * WIDTH as u64 * 2,
            "stage32_stores":u64::from(ROWS) * 2 * 640 * WIDTH as u64 * 4,
            "qk_pv_loads_each_variant":u64::from(ROWS) * 2 * 640 * WIDTH as u64 * 4},
    }))
}

#[cfg(test)]
mod traffic_tests {
    use super::*;
    #[test]
    fn window_union_counts_overlap_without_claiming_cache_bytes() {
        assert_eq!(unions(64, 120, 1).0, 64 * 128);
        assert_eq!(unions(64, 120, 64).0, 191);
        assert_eq!(unions(64, 120, 8).0, 8 * 135);
        assert_eq!(unions(1, 120, 1), (128, 512));
        assert_eq!(window_key(0, 0, true), -1);
        assert_eq!(compressed_key(1, 64, 120, true), CONTEXT as i32);
    }
}
