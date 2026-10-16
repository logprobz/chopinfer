//! MiMo's own kernels as contracts: the ring backup, restore and prefix
//! copy, the retained-row history, the DFlash drafter's row kernels, and
//! the audio encoder's quantizer and embedding sum. No other model runs
//! them, so they live here rather than in kernel-core (an op moves there
//! when a second model needs it).
//!
//! Each op is a contract type whose fields are the math the model needs
//! (layouts, ring addressing, skip rules), a role record of its buffers and
//! a CPU oracle that computes exactly what the kernel computes, bit for bit.
//! A backend prepares each op through [`OwnOps`] or refuses it; SM120 choices
//! (threads, vector widths, grids, compiled constants) are its refusals, not
//! contract rules.

use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::contract::Refusal;

pub mod dflash;
pub mod encoder;
pub mod retain;
pub mod ring;

use dflash::{
    CacheWrite, CacheWriteBuffers, CopyRows, CopyRowsBuffers, Drafts, DraftsBuffers, MaskRows,
    MaskRowsBuffers,
};
use encoder::{EmbeddingSum, EmbeddingSumBuffers, Rvq, RvqBuffers};
use retain::{GatherRows, GatherRowsBuffers, RetainRows, RetainRowsBuffers};
use ring::{
    RingBackup, RingBackupBuffers, RingPrefixCopy, RingPrefixCopyBuffers, RingRestore,
    RingRestoreBuffers,
};

/// A backend of MiMo's own kernels. Each method prepares one op or refuses
/// it; every method refuses by default, so a backend implements the ops it
/// has kernels for. Each method's doc says which [`kernel_core::backend::Counts`]
/// its launches read and which device tables bound what: a per-item table
/// holds `Counts::items` entries, at least one, and the host keeps every
/// item index below that. Each role record states the alignment its views
/// need.
pub trait OwnOps: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

    // Ring: the sliding-window ring's shadow and prefix snapshots.

    /// Saves the ring entries a phase is about to overwrite.
    ///
    /// Launches read `Counts::rows`, the items the launch covers (at most
    /// the prepared capacity), and record `rows * span` rows. `slot_ids`
    /// entries of live items with a positive `q_len` are below the ring's
    /// slot count.
    fn ring_backup(
        &self,
        op: &RingBackup,
        need: Need,
    ) -> Result<Prepared<RingBackupBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Restores a verify step's rejected ring positions from the shadow.
    ///
    /// Launches read `Counts::rows`, the items the launch covers, and record
    /// `rows * span` rows. `slot_ids` entries of live items with a positive
    /// count are below the ring's slot count.
    fn ring_restore(
        &self,
        op: &RingRestore,
        need: Need,
    ) -> Result<Prepared<RingRestoreBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Copies one slot's prefix snapshot between the ring (and a drafter
    /// history) and pool pages.
    ///
    /// Launches read no counts and record one row; the slot is a scalar of
    /// the buffers, below the ring's slot count. `pages` entries are pool
    /// pages, distinct.
    fn ring_prefix_copy(
        &self,
        op: &RingPrefixCopy,
        need: Need,
    ) -> Result<Prepared<RingPrefixCopyBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Retained rows: the drafters' circular histories of target rows.

    /// Writes every live row into its slot's history cell.
    ///
    /// Launches read `Counts::rows` and `Counts::items`. `token_batch`
    /// entries index the per-item tables (`slot_ids`, `q_start`,
    /// `prefix_len`), and no two live rows share a cell.
    fn retain_rows(
        &self,
        op: &RetainRows,
        need: Need,
    ) -> Result<Prepared<RetainRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Reads every live row back from its slot's history cell.
    ///
    /// Launches read `Counts::rows` and `Counts::items`. `token_batch`
    /// entries index `slot_ids`.
    fn gather_rows(
        &self,
        op: &GatherRows,
        need: Need,
    ) -> Result<Prepared<GatherRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // DFlash: the drafter's row kernels.

    /// Copies (and scales) a column span of every live row.
    ///
    /// Launches read `Counts::rows`.
    fn dflash_copy_rows(
        &self,
        op: &CopyRows,
        need: Need,
    ) -> Result<Prepared<CopyRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Writes the mask embedding into every draft row.
    ///
    /// Launches read `Counts::rows` and `Counts::items`. `token_batch`
    /// entries index `prefix_len`.
    fn dflash_mask_rows(
        &self,
        op: &MaskRows,
        need: Need,
    ) -> Result<Prepared<MaskRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Writes one layer's context K and scaled V into the circular cache.
    ///
    /// Launches read `Counts::rows`, the rows the K and V rows and per-row
    /// tables hold, and `Counts::items`; they grid and record
    /// `Counts::padded_rows` where it is larger (an eager prefill's
    /// capacity, whose blocks past the live rows return). `token_batch`
    /// entries index the per-item tables (`slot_ids`, `last_positions`),
    /// slots are below the cache's slot count, and no two kept rows share a
    /// cell.
    fn dflash_cache_write(
        &self,
        op: &CacheWrite,
        need: Need,
    ) -> Result<Prepared<CacheWriteBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Takes each draft row's best (score, id) over the ranks.
    ///
    /// Launches read `Counts::rows`, the rows of each rank's gathered pairs,
    /// and `Counts::items`. `token_batch` entries index `prefix_len`.
    fn dflash_drafts(
        &self,
        op: &Drafts,
        need: Need,
    ) -> Result<Prepared<DraftsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Encoder: the audio tokenizer's registered stages, which the shared
    // encoder runtime launches as the model's ops.

    /// Quantizes every live row against the stacked codebooks.
    ///
    /// Launches read `Counts::rows`. `codebook_offsets` entries are
    /// ascending and within the codebooks' rows.
    fn encoder_rvq(&self, op: &Rvq, need: Need) -> Result<Prepared<RvqBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Sums every live row's table rows.
    ///
    /// Launches read `Counts::rows`.
    fn encoder_embedding_sum(
        &self,
        op: &EmbeddingSum,
        need: Need,
    ) -> Result<Prepared<EmbeddingSumBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }
}

/// Entry `index` of a device table, or a refusal naming the table.
fn entry<T: Copy>(table: &[T], index: usize, name: &'static str) -> Result<T, Refusal> {
    table.get(index).copied().ok_or(Refusal(name))
}

/// `bytes` bytes of `buffer` from `at`, or a refusal.
fn bytes_at(buffer: &[u8], at: u64, bytes: u64) -> Result<&[u8], Refusal> {
    let at = usize::try_from(at).map_err(|_| Refusal("an oracle offset overflows"))?;
    let end = at
        .checked_add(usize::try_from(bytes).map_err(|_| Refusal("an oracle length overflows"))?)
        .ok_or(Refusal("an oracle offset overflows"))?;
    buffer
        .get(at..end)
        .ok_or(Refusal("an oracle buffer is too small"))
}

/// Copies `bytes` bytes from `source[from..]` to `target[to..]`.
fn copy(target: &mut [u8], to: u64, source: &[u8], from: u64, bytes: u64) -> Result<(), Refusal> {
    let source = bytes_at(source, from, bytes)?;
    let at = to as usize;
    target
        .get_mut(at..at + source.len())
        .ok_or(Refusal("an oracle buffer is too small"))?
        .copy_from_slice(source);
    Ok(())
}

/// The BF16 value `bits` times `scale`, as the kernels scale: the product in
/// F32, rounded to the nearest BF16 (ties to even), NaN to the canonical
/// `0x7fff` (`cvt.rn.bf16x2.f32`).
fn scale_bf16(bits: u16, scale: f32) -> u16 {
    let product = kernel_core::codec::bf16_to_f32(bits) * scale;
    if product.is_nan() {
        0x7fff
    } else {
        kernel_core::codec::f32_to_bf16(product)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Nothing;

    impl OwnOps for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    fn unsupported<T>(result: Result<T, Refused>) {
        assert!(matches!(result, Err(Refused::Unsupported)));
    }

    #[test]
    fn every_method_refuses_by_default() {
        let tp = crate::Tp::new(4).unwrap();
        let need = Need::rows(32);
        let backend = Nothing;
        assert_eq!(backend.name(), "nothing");
        let layout = ring::RingLayout::of(tp, 32);
        unsupported(backend.ring_backup(&RingBackup { layout, span: 1 }, need));
        unsupported(backend.ring_restore(&RingRestore { layout, span: 8 }, need));
        unsupported(backend.ring_prefix_copy(
            &RingPrefixCopy {
                layout,
                drafter_bytes: 0,
                page_bytes: 368_640,
                pages: 18,
                direction: ring::Direction::Capture,
            },
            need,
        ));
        let history = retain::History {
            stride: 4096,
            capacity: 136,
            slots: 32,
        };
        unsupported(backend.retain_rows(
            &RetainRows {
                width: 4096,
                input_stride: 4096,
                history,
            },
            need,
        ));
        unsupported(backend.gather_rows(
            &GatherRows {
                width: 4096,
                output_stride: 4096,
                history,
            },
            need,
        ));
        unsupported(backend.dflash_copy_rows(
            &CopyRows {
                width: 256,
                input_stride: Some(1536),
                output_stride: 256,
                scale_bits: 1.0_f32.to_bits(),
            },
            need,
        ));
        unsupported(backend.dflash_mask_rows(&MaskRows { hidden: 4096 }, need));
        unsupported(backend.dflash_cache_write(&CacheWrite::of(tp, 32, 1536, 1536), need));
        unsupported(backend.dflash_drafts(&Drafts { ranks: 4 }, need));
        let rvq = encoder::Rvq {
            dim: 1024,
            stages: 20,
            max_codes: 1024,
            codebook_rows: 4608,
        };
        unsupported(backend.encoder_rvq(&rvq, need));
        let sum = encoder::EmbeddingSum {
            tables: 20,
            vocab: 1280,
            width: 1024,
        };
        unsupported(backend.encoder_embedding_sum(&sum, need));
    }

    #[test]
    fn scaling_rounds_the_f32_product_to_nearest_even() {
        // 1.0 * 0.5 is exact.
        assert_eq!(scale_bf16(0x3f80, 0.5), 0x3f00);
        // 1.0078125 (0x3f81) * 0.612 = 0.616781... rounds to 0x3f1e.
        assert_eq!(
            scale_bf16(0x3f81, 0.612),
            kernel_core::codec::f32_to_bf16(1.0078125 * 0.612)
        );
        // NaN products are the canonical NaN.
        assert_eq!(scale_bf16(0x7fc1, 0.612), 0x7fff);
        assert_eq!(scale_bf16(0x3f80, f32::NAN), 0x7fff);
        assert_eq!(scale_bf16(0x7f80, 0.0), 0x7fff);
    }
}
