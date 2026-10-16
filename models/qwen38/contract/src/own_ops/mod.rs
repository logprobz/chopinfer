//! Qwen's own kernels as contracts: the PLE residual, state staging,
//! checkpoints, anchors and convolution items, the state entry and prefix
//! snapshot copies, the n-gram item forms, the MTP drafter's rows and draft
//! argmax, the fused hyper-connection up mix and the dense projection
//! groups. No other model runs them, so they live here rather than in
//! kernel-core (an op moves there when a second model needs it, as the
//! stream expansion moved to the shared row copies).
//!
//! Each op is a contract type whose fields are the math the model needs
//! (layouts, row and item addressing, skip rules), a role record of its
//! buffers and a CPU oracle. Copies, hashes, sums rounded once and argmaxes
//! have bitwise oracles that compute exactly what the kernel computes;
//! reductions through `expf`, `rsqrtf` or an MMA accumulation have an F64
//! reference with a per-element bound ([`Bounded`]). A backend prepares each
//! op through [`OwnOps`] or refuses it; SM120 choices (threads, vector
//! widths, grids, copy chunks, compiled constants) are its refusals, not
//! contract rules.

use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::Refusal;
use kernel_core::oracle::CheckFailure;
use kernel_core::precision::OperationUse;

pub mod copies;
pub mod dense;
pub mod hc;
pub mod mtp;
pub mod ngram;
pub mod ple;

use copies::{CopyEntries, CopyEntriesBuffers, PrefixCopy, PrefixCopyBuffers};
use dense::{DenseGroup, DenseGroupBuffers};
use hc::{HcUpMix, HcUpMixBuffers};
use mtp::{
    DraftArgmaxFinal, DraftArgmaxFinalBuffers, DraftArgmaxPartial, DraftArgmaxPartialBuffers,
    MtpCombine, MtpCombineBuffers, MtpGather, MtpGatherBuffers, MtpNorms, MtpNormsBuffers,
};
use ngram::{Ngram, NgramBuffers};
use ple::{
    PleAnchor, PleAnchorBuffers, PleCheckpoints, PleCheckpointsBuffers, PleConv, PleConvBuffers,
    PleResidual, PleResidualBuffers, PleStage, PleStageBuffers,
};

/// A backend of Qwen's own kernels. Each method prepares one op or refuses
/// it; every method refuses by default, so a backend implements the ops it
/// has kernels for. Each method's doc says which
/// [`kernel_core::backend::Counts`] its launches read, the rows each launch
/// records, and which device tables bound what: a per-item table holds
/// `Counts::items` entries, at least one, and the host keeps every item
/// index below that. Each role record states the alignment its views need.
pub trait OwnOps: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

    // PLE: the per-layer embedding's residual and its per-item state.

    /// Adds the gated PLE rows and the convolution rows to X.
    ///
    /// Launches read `Counts::rows` and record them.
    fn ple_residual(
        &self,
        op: &PleResidual,
        need: Need,
    ) -> Result<Prepared<PleResidualBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Stages each live item's PLE state entry for the step.
    ///
    /// Launches read `Counts::rows`, which they record, and
    /// `Counts::items`. `slot_ids` entries of continuing items are below the
    /// pool's entries.
    fn ple_stage(
        &self,
        op: &PleStage,
        need: Need,
    ) -> Result<Prepared<PleStageBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Writes the PLE state after each verify row into the entries the
    /// checkpoint ids name.
    ///
    /// Launches read `Counts::rows`, the verify step's rows (its token
    /// bucket, at most the prepared capacity), which `input` and `tokens`
    /// hold, and `Counts::items`, at most those rows; they record the op's
    /// verify rows, whatever the step's: rows 256 of 32 items record 8.
    /// Each item's `cu_rows` and `q_len` entries address its rows in
    /// `input` and `tokens`; `q_len` is at most the verify rows.
    fn ple_checkpoints(
        &self,
        op: &PleCheckpoints,
        need: Need,
    ) -> Result<Prepared<PleCheckpointsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Writes each anchored prefill item's PLE state at its anchor into a
    /// spare entry.
    ///
    /// Launches read `Counts::rows`, which they record, and
    /// `Counts::items`. Non-negative `anchor_slots` entries are below the
    /// spare entries, and an anchor lies within its item's rows.
    fn ple_anchor(
        &self,
        op: &PleAnchor,
        need: Need,
    ) -> Result<Prepared<PleAnchorBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The PLE convolution of each item over its staged history.
    ///
    /// A prefill op launches one item, its views offset to the item, over
    /// `Counts::rows` rows, which it records. A decode op covers every
    /// bucket item, one row or none each, over `Counts::rows` items, which
    /// it records. A verify op reads `Counts::rows`, the verify step's rows
    /// (at most the prepared capacity), which `input` and `output` hold,
    /// and `Counts::items`, at most those rows, and records the op's verify
    /// rows (rows 256 of 32 items record 8); `q_len` is at most the verify
    /// rows.
    fn ple_conv(
        &self,
        op: &PleConv,
        need: Need,
    ) -> Result<Prepared<PleConvBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Copies of per-item state.

    /// Copies state entries between planes of entries, per item.
    ///
    /// Launches read `Counts::items` and record one row. Source and
    /// destination ids are below the planes' entries, and no destination
    /// is written twice or read by another item.
    fn copy_entries(
        &self,
        op: &CopyEntries,
        need: Need,
    ) -> Result<Prepared<CopyEntriesBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Copies one slot's prefix snapshot between its state and pool pages.
    ///
    /// Launches read no counts and record one row; the slot, pool entry and
    /// PLE entry are scalars of the buffers. `pages` entries are distinct
    /// pages of the pool.
    fn prefix_copy(
        &self,
        op: &PrefixCopy,
        need: Need,
    ) -> Result<Prepared<PrefixCopyBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // The hashed n-gram embedding over items.

    /// The n-gram hash rows (and, for a gather form, the embedding rows) of
    /// every item.
    ///
    /// A hash prefill op launches one item, its views offset to the item,
    /// over `Counts::rows` rows, which it records. Decode ops cover every
    /// bucket item, one row or none each, over `Counts::rows` items, which
    /// they record. Verify ops read `Counts::rows`, the verify step's rows
    /// (at most the prepared capacity), which `tokens`, `row_ids` and the
    /// output hold, and `Counts::items`, at most those rows, and record the
    /// op's verify rows (rows 256 of 32 items record 8); `q_len` is at most
    /// the verify rows.
    fn ngram(&self, op: &Ngram, need: Need) -> Result<Prepared<NgramBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // The MTP drafter's rows.

    /// Gathers the drafter's input rows and tokens.
    ///
    /// Launches read `Counts::rows` and record them. Non-negative `sources`
    /// are rows of `last`, negative ones name retained entries;
    /// non-negative `token_sources` are entries of `drafts`.
    fn mtp_gather(
        &self,
        op: &MtpGather,
        need: Need,
    ) -> Result<Prepared<MtpGatherBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The drafter's two input norms.
    ///
    /// Launches read `Counts::rows` and record them.
    fn mtp_norms(
        &self,
        op: &MtpNorms,
        need: Need,
    ) -> Result<Prepared<MtpNormsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Adds the token row to every stream of the drafter's hidden rows.
    ///
    /// Launches read `Counts::rows` and record them.
    fn mtp_combine(
        &self,
        op: &MtpCombine,
        need: Need,
    ) -> Result<Prepared<MtpCombineBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Each head row's best logit of this rank's shard.
    ///
    /// Launches read `Counts::rows`, the head rows, and record them.
    fn draft_argmax_partial(
        &self,
        op: &DraftArgmaxPartial,
        need: Need,
    ) -> Result<Prepared<DraftArgmaxPartialBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Each head row's draft from every rank's best, scattered to the
    /// drafts and the verify tokens.
    ///
    /// Launches read `Counts::rows`, the head rows, and record them.
    /// `slots` entries index `drafts` and are distinct; non-negative
    /// `targets` entries index `token_ids` and are distinct.
    fn draft_argmax_final(
        &self,
        op: &DraftArgmaxFinal,
        need: Need,
    ) -> Result<Prepared<DraftArgmaxFinalBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // The hyper-connection.

    /// The up projection of the bottleneck and the stream mix, fused.
    ///
    /// Launches read `Counts::rows` and record them.
    fn hc_up_mix(
        &self,
        op: &HcUpMix,
        need: Need,
    ) -> Result<Prepared<HcUpMixBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // Dense projections.

    /// A dense group of steps of `usage`: an eager prefill or a graph
    /// bucket, whose rows a backend may schedule differently.
    ///
    /// Launches read `Counts::rows` and record them.
    fn dense_group(
        &self,
        op: &DenseGroup,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<DenseGroupBuffers<View>>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }
}

/// How a launch over items addresses them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ItemForm {
    /// One item per launch, its views offset to the item, whose row count
    /// is the one `q_len` entry.
    One,
    /// Every bucket item at once: item `i` has `q_len[i]` rows, zero or
    /// one, at row `i`, and its state is the `i`-th entry.
    Decode,
    /// Every live verify item at once: item `z` has `q_len[z]` rows, at most
    /// `rows`, from row `cu_rows[z]`, and its state is the `z`-th entry.
    Verify { rows: u32 },
}

/// One item a launch writes: its index, first row and rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ItemRows {
    pub item: u32,
    pub start: u64,
    pub rows: u32,
}

/// The items with rows that one launch of `form` over `items` grid items
/// writes: a one-item launch its item (`q_len[0]` rows, at most `items`,
/// the launch's rows); a decode launch each item with its row; a verify
/// launch each of the first `live_items` items with rows. `cu_rows` and
/// `live_items` are the verify form's.
pub fn item_rows(
    form: ItemForm,
    q_len: &[i32],
    cu_rows: Option<&[i32]>,
    live_items: Option<u32>,
    items: u32,
) -> Result<Vec<ItemRows>, Refusal> {
    let count = |item: u32| -> Result<u32, Refusal> {
        u32::try_from(entry(q_len, item as usize, "q_len")?)
            .map_err(|_| Refusal("an item has a negative row count"))
    };
    match form {
        ItemForm::One => {
            let rows = count(0)?;
            if rows > items {
                return Err(Refusal("an item has more rows than its launch"));
            }
            Ok((rows > 0)
                .then_some(ItemRows {
                    item: 0,
                    start: 0,
                    rows,
                })
                .into_iter()
                .collect())
        }
        ItemForm::Decode => {
            let mut written = Vec::new();
            for item in 0..items {
                match count(item)? {
                    0 => {}
                    1 => written.push(ItemRows {
                        item,
                        start: u64::from(item),
                        rows: 1,
                    }),
                    _ => return Err(Refusal("a decode item has more than one row")),
                }
            }
            Ok(written)
        }
        ItemForm::Verify { rows } => {
            let cu_rows = cu_rows.ok_or(Refusal("a verify launch needs cu_rows"))?;
            let live = live_items.ok_or(Refusal("a verify launch needs live items"))?;
            let mut written = Vec::new();
            for item in 0..items.min(live) {
                let count = verify_rows(q_len, item, rows)?;
                if count > 0 {
                    let start = u64::try_from(entry(cu_rows, item as usize, "cu_rows")?)
                        .map_err(|_| Refusal("a verify item starts before row 0"))?;
                    written.push(ItemRows {
                        item,
                        start,
                        rows: count,
                    });
                }
            }
            Ok(written)
        }
    }
}

/// The rows of verify item `item`, at most `most`; a negative count has
/// none.
fn verify_rows(q_len: &[i32], item: u32, most: u32) -> Result<u32, Refusal> {
    let rows = entry(q_len, item as usize, "q_len")?.max(0) as u32;
    if rows > most {
        return Err(Refusal("a verify item has more rows than the op serves"));
    }
    Ok(rows)
}

/// A numeric reference: per element its F64 value and the most a kernel's
/// stored value may differ from it.
#[derive(Clone, Debug, PartialEq)]
pub struct Bounded {
    pub values: Vec<f64>,
    pub bounds: Vec<f64>,
}

impl Bounded {
    /// Refuses BF16 `actual` values unless each is finite and within its
    /// bound of the reference, naming the element with the largest excess.
    pub fn check_bf16(&self, actual: &[u16]) -> Result<(), CheckFailure> {
        if actual.len() != self.values.len() || self.values.len() != self.bounds.len() {
            return Err(CheckFailure {
                index: actual.len().min(self.values.len()),
                actual: f64::NAN,
                reference: f64::NAN,
                bound: 0.0,
                error: f64::INFINITY,
            });
        }
        let mut worst: Option<(f64, CheckFailure)> = None;
        for (index, (&bits, (&reference, &bound))) in actual
            .iter()
            .zip(self.values.iter().zip(&self.bounds))
            .enumerate()
        {
            let value = f64::from(bf16_to_f32(bits));
            let error = (value - reference).abs();
            if value.is_finite() && reference.is_finite() && error <= bound {
                continue;
            }
            let excess = if (error - bound).is_nan() {
                f64::INFINITY
            } else {
                error - bound
            };
            if worst.as_ref().is_none_or(|(most, _)| excess > *most) {
                let failure = CheckFailure {
                    index,
                    actual: value,
                    reference,
                    bound,
                    error,
                };
                worst = Some((excess, failure));
            }
        }
        worst.map_or(Ok(()), |(_, failure)| Err(failure))
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

/// The `bytes` bytes of `buffer` from `at`, writable, or a refusal.
fn bytes_at_mut(buffer: &mut [u8], at: u64, bytes: u64) -> Result<&mut [u8], Refusal> {
    let at = usize::try_from(at).map_err(|_| Refusal("an oracle offset overflows"))?;
    let end = at
        .checked_add(usize::try_from(bytes).map_err(|_| Refusal("an oracle length overflows"))?)
        .ok_or(Refusal("an oracle offset overflows"))?;
    buffer
        .get_mut(at..end)
        .ok_or(Refusal("an oracle buffer is too small"))
}

/// Copies `bytes` bytes from `source[from..]` to `target[to..]`.
fn copy(target: &mut [u8], to: u64, source: &[u8], from: u64, bytes: u64) -> Result<(), Refusal> {
    let source = bytes_at(source, from, bytes)?;
    bytes_at_mut(target, to, bytes)?.copy_from_slice(source);
    Ok(())
}

/// The F32 sum of BF16 values `terms` in order, rounded once to BF16, as
/// the own kernels add rows (`f32_to_bf16` of `qwen38_bundle.cuh`, whose
/// NaN keeps its payload's top bits and sets the quiet bit).
fn sum_bf16(terms: &[u16]) -> u16 {
    let sum = terms
        .iter()
        .fold(None, |sum: Option<f32>, &bits| {
            let value = bf16_to_f32(bits);
            Some(sum.map_or(value, |sum| sum + value))
        })
        .unwrap_or(0.0);
    kernel_core::codec::f32_to_bf16(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Shape;

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
        let shape = Shape::TP4;
        let need = Need::rows(32);
        let backend = Nothing;
        assert_eq!(backend.name(), "nothing");
        let layout = ple::PleLayout::of(shape);
        unsupported(backend.ple_residual(&PleResidual::of(shape), need));
        unsupported(backend.ple_stage(&PleStage { layout }, need));
        unsupported(backend.ple_checkpoints(&PleCheckpoints::of(shape), need));
        unsupported(backend.ple_anchor(&PleAnchor { layout }, need));
        unsupported(backend.ple_conv(&PleConv::of(shape, ItemForm::Decode), need));
        unsupported(backend.copy_entries(
            &CopyEntries {
                entry_bytes: layout.entry_bytes(),
                planes: 1,
                plane_stride: 0,
            },
            need,
        ));
        unsupported(backend.prefix_copy(
            &PrefixCopy {
                snapshot: copies::Snapshot::of(shape, 64, 32, 12, 32, false),
                page_bytes: 835_584,
                pages: 35,
                direction: copies::Direction::Capture,
            },
            need,
        ));
        unsupported(backend.ngram(
            &Ngram::of(shape, 1, ngram::NgramWrites::Gather, ItemForm::Decode),
            need,
        ));
        unsupported(backend.mtp_gather(&MtpGather::of(shape), need));
        unsupported(backend.mtp_norms(&MtpNorms::of(shape), need));
        unsupported(backend.mtp_combine(&MtpCombine::of(shape), need));
        unsupported(backend.draft_argmax_partial(&DraftArgmaxPartial::of(shape, 0), need));
        unsupported(backend.draft_argmax_final(&DraftArgmaxFinal::of(shape), need));
        unsupported(backend.hc_up_mix(&HcUpMix::of(shape, hc::UpMixInput::Down), need));
        unsupported(backend.dense_group(
            &DenseGroup::of(shape, dense::Projection::Router),
            OperationUse {
                phase: kernel_core::precision::ExecutionPhase::Prefill,
                capacity: 8192,
                graph: false,
            },
            need,
        ));
    }

    #[test]
    fn launches_over_items_cover_the_items_with_rows() {
        let rows = |found: Vec<ItemRows>| {
            found
                .into_iter()
                .map(|r| (r.item, r.start, r.rows))
                .collect::<Vec<_>>()
        };
        let decode = item_rows(ItemForm::Decode, &[1, 0, 1], None, None, 3).unwrap();
        assert_eq!(rows(decode), [(0, 0, 1), (2, 2, 1)]);
        assert!(item_rows(ItemForm::Decode, &[2], None, None, 1).is_err());
        let verify = ItemForm::Verify { rows: 8 };
        let found = item_rows(verify, &[3, 0, 8, 2], Some(&[0, 3, 3, 11]), Some(3), 4).unwrap();
        assert_eq!(rows(found), [(0, 0, 3), (2, 3, 8)]);
        assert!(item_rows(verify, &[9], Some(&[0]), Some(1), 1).is_err());
        let one = item_rows(ItemForm::One, &[5], None, None, 5).unwrap();
        assert_eq!(rows(one), [(0, 0, 5)]);
        assert!(item_rows(ItemForm::One, &[6], None, None, 5).is_err());
    }

    #[test]
    fn row_sums_round_once_in_f32_order() {
        // 1 + 2^-9 + 2^-9: each BF16 rounding alone would drop the 2^-9s.
        let small = kernel_core::codec::f32_to_bf16(2_f32.powi(-9));
        assert_eq!(
            sum_bf16(&[0x3f80, small, small]),
            kernel_core::codec::f32_to_bf16(1.0 + 2_f32.powi(-8))
        );
        // NaN keeps its payload's top bits and is quiet.
        assert_eq!(sum_bf16(&[0x7f81, 0x3f80]), 0x7fc1);
    }

    #[test]
    fn bounded_references_name_the_worst_element() {
        let reference = Bounded {
            values: vec![1.0, 2.0, 3.0],
            bounds: vec![0.0, 0.01, 0.1],
        };
        let bits = |values: [f32; 3]| values.map(kernel_core::codec::f32_to_bf16);
        assert!(reference.check_bf16(&bits([1.0, 2.0, 3.0])).is_ok());
        let failure = reference.check_bf16(&bits([1.0, 2.25, 3.5])).unwrap_err();
        assert_eq!(failure.index, 2);
        assert!(reference.check_bf16(&bits([1.0, f32::NAN, 3.0])).is_err());
        assert!(reference.check_bf16(&bits([1.0, 2.0, 3.0])[..2]).is_err());
    }
}
