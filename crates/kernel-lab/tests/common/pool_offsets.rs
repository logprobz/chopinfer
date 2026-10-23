//! Pool offsets past the 32-bit lines.
//!
//! Kernels scale page, slot, checkpoint, entry and row ids into byte or
//! element offsets of pools that serving sizes from free device memory. The
//! other kernel-lab cases pack those ids densely from zero, so a product
//! computed in 32 bits passes them and fails only once a large pool hands out
//! high ids. The `*_high_ids_*` tests run each family twice: on a dense low
//! pool, checked by the family's existing oracle or reference, and with its
//! live entries at the ids [`high_ids`] picks, in a pool whose every other
//! byte holds [`SENTINEL`]. Group A ids are the first whose byte offset
//! exceeds `i32::MAX`, so a signed 32-bit product turns them negative. Group
//! B ids are the first at or past 2^32 elements of the pool's element type,
//! so their offsets truncated to 32 bits, signed or unsigned, in bytes or in
//! elements, land in the pool's first entries, which hold the sentinel. The
//! high run must reproduce the low run bit for bit, and a kernel that writes
//! the pool must leave every other byte at the sentinel. Each case writes a
//! `pool_offsets` receipt.
//!
//! # Audit
//!
//! Every kernel product that scales an id into a pool offset, as compiled on
//! 2026-09-28. The maximum is the largest offset under the package limits:
//! up to 256 items, MiMo contexts of 1,048,576 tokens and Qwen contexts of
//! 262,144, a KV pool of 0.9 of free device memory
//! (`docs/numerics.md#kv-offload-and-prefix-state`) and at most 512 linear-state entries per layer
//! plane (`docs/numerics.md#recurrent-state`). No product needed a fix.
//!
//! Attention engine (`kernels-sm120/cuda/attention`):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | `cache_byte_offset`, paged | `u64(u32(table[b * max_pages + p])) * page_stride_bytes` | u64, bytes | the pool, tens of GB on a 96 GB device | `attention_high_ids_match_low_ids`, `qwen38_high_ids_match_low_ids` |
//! | block-table index | `u64(batch) * max_pages + position / page` | u64, elements | 256 x 16,384 entries | same |
//! | `cache_byte_offset` and `cache_token`, ring | `(u64(slots[b]) * page + pos % page) * kv_heads + head`, times `dim * bytes` | u64, bytes | 256 slots: 25 MB per SWA layer, 256 MB DFlash | `attention_high_ids_match_low_ids` |
//! | `decode` current row (`CurrentRow`, `GatedRow`) | offsets from `cache_byte_offset` | u64, bytes | as paged and ring | `attention_high_ids_rope_fold_matches_low_ids`, `qwen38_high_ids_match_low_ids` |
//! | `decode_window` `offset` | `((u64 slot * page + pos % page) * kv_heads + head) * u64(dim) * 2` | u64, bytes | as ring | `attention_high_ids_match_low_ids` |
//! | `prefill` paged `issue_tma` | i32 page coordinate of a 4D map whose outer extent is the pool's pages (u32) and stride the u64 page stride (below 2^40); TMA forms 64-bit addresses | i32 coordinate | page ids up to `i32::MAX` | `attention_high_ids_match_low_ids`, `qwen38_high_ids_match_low_ids` |
//! | `prefill` ring `issue_ring` | `(u64(slot) * page + pos % page) * kv_heads + head`, times `dim` | u64, elements | as ring | `attention_high_ids_match_low_ids` |
//! | `verify` own rows | `((u64(item.start) + t) * kv_heads + head) * dim * 2` | u64, bytes | 256 tokens | none needed |
//! | `decode_partial`, `verify_partial`, `partial_index`, `prefill_record` | row, head and split products | u64, elements | the planned workspace | none needed |
//! | `qk_norm_rope_kv` (`gated.cuh`) | `u64(u32(table[..])) * k_page_stride_bytes` | u64, bytes | as paged | `qwen38_high_ids_match_low_ids` |
//! | sparse `attention` (`sparse.cuh`) | `u64(u32(block_tables[..])) * k_page_stride_bytes`; `selected[u64(row) * top + i]` | u64, bytes | as paged | `qsa_high_ids_match_low_ids` |
//! | `block_key_row` (`sparse_indexer.cuh`) | `u64(u32(table[..])) * page_stride_bytes + u64(b % page_blocks) * dim * 2` | u64, bytes | as paged | `qsa_high_ids_match_low_ids` |
//! | `indexer_key_write` raw-key ring | `u64(u32(slot_ids[b])) * ring * dim`, positions `* ring * 3` | u64, elements | 256 slots: 1 MiB | none needed |
//! | `block_scores`, `block_topk` | `u64(row) * score_stride + b` | u64, elements | 2,048 rows x 65,536 F32: 537 MB | `qsa_high_ids_match_low_ids` |
//! | `varlen` (encoders) | `(u64(token) * heads + head) * d` | u64, elements | 65,536 tokens | none needed |
//!
//! Row engine (`kernels-sm120/cuda/row`):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | `rope_kv`, paged | `u64(page) * k_page_stride_bytes`, then `(cache_token * KvHeads + head) * Qk + d` within the page | u64, bytes | as paged | `row_rope_high_ids_match_low_ids` |
//! | `rope_kv` ring, `ring_write` | `u64(slot_ids[b]) * ring_size + pos % ring_size`, then `(cache_token * KvHeads + head) * Qk + d` | u64, elements | 256 slots | `row_rope_high_ids_match_low_ids` |
//! | `embedding` | `u64(local) * Width + c` | u64, elements | 248,319 x 2,560 BF16: 1.27 GB (MiMo 1.25 GB), below 2^31 bytes | `row_qwen38_full_vocab_embedding_gpu` |
//! | `ngram_row` (`qwen38_ple.cuh`) | `(row - table_start) * 160 + c`, row u64 | u64, bytes | one rank's table: 12 or 24 GiB at TP4 or TP2 | `ngram_high_ids_g1` |
//! | `ngram_decode`, `ple_conv_decode` | `item * entry` (u32 times u64) | u64, words and elements | 256 items x 92,224 elements: 47 MB | none needed |
//! | `logits_topk`, DFlash `argmax_partial` | `u64(row) * Capacity + c` | u64, elements | 8,191 x 248,320 F32: 8.1 GB | none needed |
//! | norms, SwiGLU, router, permute, gather, quantize, combine | `u64(row) * Width`, `u64(token) * Experts` | u64, elements | 8,192 rows | none needed |
//! | `scale_expand` raw matrix bank | `u64(matrix) * rows * groups` | u64, bytes | encoded bank below 4 GiB, expanded bank can exceed 4 GiB | `scale_expand_high_matrix_offsets_on_gpu` in kernel-lab-families (2 GiB and 4 GiB offsets) |
//!
//! Linear engine (`kernels-sm120/cuda/linear`):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | `conv_prefill`, `conv_decode`, `conv_verify` | `u64(u32(slot_ids[s])) * History * Channels`, likewise `checkpoint_ids` | u64, elements | 512 entries x 30,720 B at TP2: 15.7 MB; any i32 id by contract | `linear_high_ids_match_low_ids` |
//! | `gated_delta_recurrent` (decode, verify) | `u64(u32(slot_ids[seq])) * ValueHeads * HeadElements`, likewise `checkpoint_ids` | u64, elements | 512 x 1,572,864 B at TP2: 805 MB | `linear_high_ids_match_low_ids` |
//! | `gated_delta_scan` (prefill) | `(u64(u32(slot_ids[seq])) * ValueHeads + head) * Dim * Dim` | u64, elements | as decode | `linear_high_ids_match_low_ids` |
//! | delta prefill workspace | `u64(chunk) * kBytes` | u64, bytes | 384 chunks x 1.1 MB | none needed |
//!
//! Per-channel-decay delta rule (`kernels-sm120/cuda/linear/channel_decay_*.cuh`,
//! added 2026-10-06):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | `short_conv_prefill`, `short_conv_decode`, `short_conv_verify` | `u64(u32(slot_ids[s])) * History * Channels`, likewise `anchor_ids` and `checkpoint_ids` | u64, elements | 512 entries x 36,864 B: 18.9 MB; any i32 id by contract | `channel_decay_high_ids_match_low_ids` |
//! | `channel_delta_recurrent` (decode, verify) | `u64(u32(slot_ids[seq])) * Heads * HeadElements`, likewise `checkpoint_ids` | u64, elements | 512 x 1,048,576 B: 537 MB | `channel_decay_high_ids_match_low_ids` |
//! | `channel_delta_scan` (prefill) | `u64(u32(slot_ids[seq])) * Heads * HeadElements`, likewise `anchor_ids` | u64, elements | as decode | `channel_decay_high_ids_match_low_ids` |
//! | prefill workspace | `(u64(u32(chunk)) * heads + head) * floats` | u64, elements | 640 chunks x 2.37 MB: 1.5 GB | none needed |
//!
//! MoE and GEMM engines (`kernels-sm120/cuda/moe`, `cuda/gemm`):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | `moe.cuh` `run`, `weight_value` | `expert * N + column`, then `u64(strip) * strips_k * 512` or an i32 TMA coordinate | i32 rows, then u64 bytes | 1,048,575 rows (MiMo down, 256 x 4,096) | none needed |
//! | `moe.cuh` FP8 block scales | `(expert * n_blocks + column / 128) * k_blocks + k / 128` | i32, elements | 24,575 (standard profile only) | none needed |
//! | `packed_scale_offset`, `scale_offset` | `(row / 16) * k_tiles` in i32, then u64 times 128 | i32, then u64 bytes | 1,048,575, then 33.5 MB | none needed |
//! | `qwen.cuh`, `mimo26_*.cuh` weight strips | `expert * N + ...` in i32, then `u64(strip) * Ktiles * 512` | u64, bytes | 1.07 GB | none needed |
//! | routed rows | `u64(routed) * OutN + column` | u64, elements | 81,920 routes: 419 MB | none needed |
//! | GEMM weights | `u64(strip) * k_stages * strip_stage_bytes` | u64, bytes | 636 MB (Qwen TP2 head) | none needed |
//! | GEMM outputs | `u64(row) * row_stride + column` | u64, elements | 8,192 x 124,160 F32: 4.07 GB | none needed |
//!
//! FP4 compressed-key writers (`kernels-sm120/cuda/gated_pooling/key_write.cuh`):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | index and compressed code/value planes | `u64(page) * code_stride` | u64, bytes | allocated pool | `fp4_key_write_offsets::fp4_planes_cross_both_32_bit_boundaries` |
//! | index and compressed scale planes | `u64(page) * scale_stride` | u64, bytes | allocated pool | same, independent code/scale strides and low-address sentinels |
//!
//! Sampling, comm and multimodal rows use u64 row products; their remaining
//! 32-bit products are tile and route counters of at most 1,048,576.
//!
//! Bundle-owned kernels (`models/*/bundle/cuda`):
//!
//! | Kernel | Product | Width, unit | Maximum | Test |
//! |---|---|---|---|---|
//! | `mimo_ring` `backup`, `restore` | `u64(slot_ids[item]) * kRingTokens + pos % kRingTokens`; `blockIdx.y * kLayer` | u64, bytes | 256 slots; 6.5 GB of ring state | `ring_entries_of_other_slot_counts_address_their_planes` (offsets past 2^31) |
//! | `mimo_retain` | `u64(slot) * capacity + pos % capacity`, times `history_stride` | u64, elements | slots checked against `slots` | none needed |
//! | `mimo_dflash` `cache_write` | `(u64(slot) * 2048 + pos % 2048) * kEntry` | u64, elements | slot below 256 (traps) | none needed |
//! | `ple_stage` | `u64(slot_ids[item]) * kPleEntryBytes` | u64, bytes | 256 x 184,448 B: 47 MB | `ple_state_high_ids_match_low_ids` |
//! | `copy_entries` | `u64(blockIdx.y) * plane_stride + u64(ids[item]) * entry_bytes` | u64, bytes | 36 planes x 512 x 1,572,864 B: 29 GB | `ple_state_high_ids_match_low_ids`; planes by the wide `qwen38_bundle_rows` cases |
//! | `prefix_copy` | `((at / entry) * pool_entries + index) * entry`, `pages[at / page_bytes] * page_bytes` | u64, bytes | 29 GB | `qwen38_bundle_rows` prefix copies |
//! | `ple_checkpoints_indexed` | `u64(checkpoint_ids[item * 8 + row]) * kPleEntryBytes` | u64, bytes | i32 pool id | `indexed_checkpoint_entries_match_copy_bits` |
//! | `prefix_copy<true>` PLE | `u64(ple_entry) * kPleEntryBytes` | u64, bytes | u32 pool id | `prefix_copy_uses_independent_entry_indices` |
//! | `ple_checkpoints`, `mtp_gather`, `ngram_verify`, `ple_conv_verify` | `(u64(item) * 8 + row) * kPleEntryBytes`, `u64(source) * kWide`, `item * entry` | u64 | 47 MB | none needed |
//!
//! Linear-state ids above 512 are reachable only through the kernel
//! contract, which takes any i32 id; the tests cover that contract.
#![allow(dead_code)]

use kernel_lab::{LabResult, Session};
use runtime_cuda::DeviceBuffer;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Real page-one strides used by the FP4 writer test. The extra 4096
/// bytes leave a visible low-address target for unsigned truncation.
pub const FP4_PLANE_STRIDES: [u64; 2] = [(1_u64 << 31) + 4096, (1_u64 << 32) + 4096];

/// Every pool byte outside the live entries: NaN in BF16, FP8 E4M3 and F32.
pub const SENTINEL: u8 = 0xff;

/// Bytes of one host transfer when scanning a pool.
const CHUNK: usize = 256 << 20;

/// The ids of one pool whose entries lie past the 32-bit lines.
#[derive(Clone, Debug)]
pub struct HighIds {
    /// Bytes from one id's entry to the next.
    pub stride: u64,
    /// Bytes of the pool's element type: the unit of an element offset.
    pub element: u64,
    /// The first ids whose byte offset exceeds `i32::MAX`.
    pub a: Vec<u64>,
    /// The first ids whose offset is at least 2^32 elements.
    pub b: Vec<u64>,
}

/// `count` ids in each group for entries of `stride` bytes and elements of
/// `element` bytes. The pool's first `count + 1` entries stay free: every
/// 32-bit truncation of a group B offset (and of any byte within its entry)
/// lands there.
pub fn high_ids(stride: u64, element: u64, count: usize) -> HighIds {
    assert!(stride > 0 && element > 0 && count > 0);
    let first_a = i32::MAX as u64 / stride + 1;
    let first_b = ((1_u64 << 32) * element).div_ceil(stride);
    let count = count as u64;
    assert!(
        first_a > count + 1 && first_a + count <= first_b,
        "stride {stride} leaves no room for {count} ids per group"
    );
    assert!(
        first_b + count - 1 <= i32::MAX as u64,
        "ids must fit a signed 32-bit table"
    );
    HighIds {
        stride,
        element,
        a: (first_a..first_a + count).collect(),
        b: (first_b..first_b + count).collect(),
    }
}

impl HighIds {
    /// Entries a pool needs to hold every id.
    pub fn entries(&self) -> u64 {
        self.b
            .last()
            .copied()
            .unwrap_or(0)
            .max(self.a.last().copied().unwrap_or(0))
            + 1
    }

    /// Bytes of a pool of `entries()` entries.
    pub fn pool_bytes(&self) -> u64 {
        self.entries() * self.stride
    }

    /// Ids for `n` entries, alternating group A and group B.
    pub fn alternating(&self, n: usize) -> Vec<u64> {
        assert!(n.div_ceil(2) <= self.a.len() && n / 2 <= self.b.len());
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    self.a[i / 2]
                } else {
                    self.b[i / 2]
                }
            })
            .collect()
    }

    /// The groups' first ids and byte offsets, for receipts.
    pub fn describe(&self) -> Value {
        json!({
            "stride_bytes": self.stride,
            "element_bytes": self.element,
            "group_a": [self.a[0], self.a[self.a.len() - 1]],
            "group_b": [self.b[0], self.b[self.b.len() - 1]],
            "group_a_first_offset": self.a[0] * self.stride,
            "group_b_first_offset": self.b[0] * self.stride,
            "pool_bytes": self.pool_bytes(),
        })
    }
}

/// Ids as the signed 32-bit values of block tables and slot lists.
pub fn i32_ids(ids: &[u64]) -> Vec<i32> {
    ids.iter()
        .map(|&id| i32::try_from(id).expect("an id past i32::MAX"))
        .collect()
}

/// A device pool that holds `SENTINEL` everywhere except its live entries.
pub struct SentinelPool {
    pub buffer: DeviceBuffer,
    /// Byte ranges holding live data, `[start, end)`.
    live: Vec<(u64, u64)>,
}

impl SentinelPool {
    /// Allocates `bytes` holding the sentinel.
    pub fn new(session: &Session, bytes: u64) -> LabResult<Self> {
        let len = usize::try_from(bytes).map_err(|e| e.to_string())?;
        let buffer = session.allocate(len)?;
        poison(session, &buffer)?;
        Ok(Self {
            buffer,
            live: Vec::new(),
        })
    }

    pub fn bytes(&self) -> u64 {
        self.buffer.byte_len() as u64
    }

    /// Marks `[offset, offset + len)` as live without writing it.
    pub fn mark(&mut self, offset: u64, len: u64) {
        self.live.push((offset, offset + len));
    }

    /// Writes host bytes at `offset` and marks them live.
    pub fn put(&mut self, session: &Session, offset: u64, bytes: &[u8]) -> LabResult<()> {
        self.buffer
            .copy_from_bytes_at(offset as usize, bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        // The host bytes must outlive the copy.
        session.stream.synchronize().map_err(|e| e.to_string())?;
        self.mark(offset, bytes.len() as u64);
        Ok(())
    }

    /// Copies `len` device bytes from `source` at `source_offset` to
    /// `offset`, in stream order, and marks them live.
    pub fn copy_from(
        &mut self,
        session: &Session,
        offset: u64,
        source: &DeviceBuffer,
        source_offset: u64,
        len: u64,
    ) -> LabResult<()> {
        self.buffer
            .copy_range_from_async(
                offset as usize,
                source,
                source_offset as usize,
                len as usize,
                &session.stream,
            )
            .map_err(|e| e.to_string())?;
        self.mark(offset, len);
        Ok(())
    }

    /// `len` bytes at `offset`.
    pub fn read(&self, session: &Session, offset: u64, len: u64) -> LabResult<Vec<u8>> {
        read(session, &self.buffer, offset, len)
    }

    /// Every byte outside the live ranges still holds the sentinel. Returns
    /// the bytes scanned.
    pub fn check_untouched(&self, session: &Session, case: &str) -> LabResult<u64> {
        let mut live = self.live.clone();
        live.sort_unstable();
        let mut gaps = Vec::new();
        let mut at = 0;
        for (from, to) in live {
            if from > at {
                gaps.push((at, from));
            }
            at = at.max(to);
        }
        if at < self.bytes() {
            gaps.push((at, self.bytes()));
        }
        let mut host = vec![0_u8; CHUNK];
        let mut scanned = 0;
        for (from, to) in gaps {
            let mut start = from;
            while start < to {
                let len = (to - start).min(CHUNK as u64) as usize;
                let chunk = &mut host[..len];
                self.buffer
                    .copy_to_bytes_at_async(start as usize, chunk, &session.stream)
                    .map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                if let Some(position) = chunk.iter().position(|&byte| byte != SENTINEL) {
                    return Err(format!(
                        "{case}: pool byte {} outside every live entry changed to {:#04x}",
                        start + position as u64,
                        chunk[position]
                    ));
                }
                scanned += len as u64;
                start += len as u64;
            }
        }
        Ok(scanned)
    }
}

/// Fills all of `buffer` with the sentinel on the device: one uploaded
/// chunk, then doubling copies within the allocation. Outputs and
/// workspaces are poisoned before a high-id run, so a result the run never
/// wrote cannot pass as the low-id run's.
pub fn poison(session: &Session, buffer: &DeviceBuffer) -> LabResult<()> {
    let len = buffer.byte_len();
    if len == 0 {
        return Ok(());
    }
    let first = len.min(64 << 20);
    buffer
        .copy_from_bytes(&vec![SENTINEL; first], &session.stream)
        .map_err(|e| e.to_string())?;
    let mut filled = first;
    while filled < len {
        let step = filled.min(len - filled);
        buffer
            .copy_within_async(0, filled, step, &session.stream)
            .map_err(|e| e.to_string())?;
        filled += step;
    }
    session.stream.synchronize().map_err(|e| e.to_string())
}

/// `len` bytes of `buffer` at `offset`.
pub fn read(session: &Session, buffer: &DeviceBuffer, offset: u64, len: u64) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0_u8; len as usize];
    buffer
        .copy_to_bytes_at_async(offset as usize, &mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// Free device bytes a test declares: its largest case's pools plus room for
/// the low-id fixture and workspaces.
pub fn need_bytes(pool_bytes: u64) -> u64 {
    pool_bytes + (3 << 30)
}

/// `Ok` when `actual` equals `expected` byte for byte, else the first
/// differing index.
pub fn same(case: &str, what: &str, expected: &[u8], actual: &[u8]) -> LabResult<()> {
    if expected.len() != actual.len() {
        return Err(format!(
            "{case}: {what} has {} bytes, the low-id run {}",
            actual.len(),
            expected.len()
        ));
    }
    match expected.iter().zip(actual).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(index) => Err(format!(
            "{case}: {what} differs from the low-id run at byte {index}: {:#04x} against {:#04x}",
            actual[index], expected[index]
        )),
    }
}

/// Writes a `pool_offsets` receipt for one case.
pub fn receipt(session: &Session, family: &str, case: &str, fields: Value) -> LabResult<()> {
    let mut data: BTreeMap<String, Value> =
        serde_json::from_value(fields).map_err(|e| e.to_string())?;
    data.insert("family".into(), json!(family));
    data.insert("case".into(), json!(case));
    data.insert("sentinel".into(), json!(SENTINEL));
    data.insert("correct".into(), json!(true));
    session.receipt("pool_offsets", data).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every (entry stride, element bytes, ids per group) the high-id cases
    /// use, at each rank count they can run.
    fn geometries() -> Vec<(&'static str, u64, u64, usize)> {
        let mut all = vec![
            // MiMo TP4 GA page of nine layers, K and V planes.
            ("mimo ga bf16 page", 368_640, 2, 128),
            ("mimo ga fp8 page", 184_320, 1, 128),
            // MiMo TP4 SWA ring slot, K then V, BF16 and FP8.
            ("mimo swa k slot", 98_304, 2, 8),
            ("mimo swa v slot", 65_536, 2, 8),
            ("mimo swa fp8 k slot", 49_152, 1, 8),
            ("mimo swa fp8 v slot", 32_768, 1, 8),
            // MiMo TP4 DFlash ring slot.
            ("mimo dflash slot", 1_048_576, 2, 8),
            // Qwen page of twelve layers, K and V planes.
            ("qwen bf16 page", 786_432, 2, 128),
            ("qwen fp8 page", 393_216, 1, 128),
            // Qwen serving page with block keys, BF16 and FP8 KV.
            ("qwen qsa bf16 page", 835_584, 2, 512),
            ("qwen qsa fp8 page", 442_368, 2, 512),
            // PLE state entry and n-gram row.
            ("ple entry", 184_448, 1, 8),
            ("ngram row", 160, 1, 4096),
        ];
        // Linear conv and delta-rule entries at TP2 and TP4.
        for (conv, delta) in [(30_720, 1_572_864), (15_360, 786_432)] {
            all.push(("linear conv entry", conv, 2, 64));
            all.push(("linear delta entry", delta, 4, 64));
        }
        // Per-channel-decay conv and delta-rule entries of the served shape.
        all.push(("channel decay conv entry", 36_864, 2, 18));
        all.push(("channel decay delta entry", 1_048_576, 4, 18));
        all
    }

    #[test]
    fn boundary_ids_cross_both_lines() {
        for (name, stride, element, count) in geometries() {
            let ids = high_ids(stride, element, count);
            let head = (count as u64 + 1) * stride;
            // Group A starts just past i32::MAX bytes.
            assert!(ids.a[0] * stride > i32::MAX as u64, "{name}");
            assert!((ids.a[0] - 1) * stride <= i32::MAX as u64, "{name}");
            // Group B starts at 2^32 elements.
            assert!(ids.b[0] * stride >= (1_u64 << 32) * element, "{name}");
            assert!((ids.b[0] - 1) * stride < (1_u64 << 32) * element, "{name}");
            for &id in &ids.b {
                for within in [0, stride - 1] {
                    let offset = id * stride + within;
                    // Bytes, unsigned and signed.
                    let bytes = u64::from(offset as u32);
                    assert!(bytes < head, "{name}: u32 bytes of id {id}");
                    assert!((offset as u32 as i32) >= 0, "{name}: i32 bytes of id {id}");
                    // Elements, unsigned and signed.
                    let elements = offset / element;
                    assert!(
                        u64::from(elements as u32) * element < head,
                        "{name}: u32 elements of id {id}"
                    );
                    assert!(
                        (elements as u32 as i32) >= 0,
                        "{name}: i32 elements of id {id}"
                    );
                }
            }
            // The head of the pool never holds a live id.
            assert!(ids.a[0] * stride >= head, "{name}");
        }
    }

    #[test]
    fn fp4_plane_strides_cross_signed_and_unsigned_boundaries() {
        let [signed, unsigned] = FP4_PLANE_STRIDES;
        assert!(signed > i32::MAX as u64 && signed < u32::MAX as u64);
        assert!((signed as u32 as i32) < 0);
        assert!(unsigned > u32::MAX as u64);
        assert_eq!(unsigned as u32, 4096);
    }

    #[test]
    fn alternating_ids_cover_both_groups() {
        let ids = high_ids(368_640, 2, 4);
        assert_eq!(
            ids.alternating(5),
            vec![ids.a[0], ids.b[0], ids.a[1], ids.b[1], ids.a[2]]
        );
        assert_eq!(ids.entries(), ids.b[3] + 1);
    }
}
