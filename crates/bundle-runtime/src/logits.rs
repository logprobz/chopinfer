//! Device logits views in the original mixed-batch item order.

use std::collections::BTreeMap;

use runtime_bundle_abi::{
    BUNDLE_LOGITS_ELEMENT_V1_BF16, BUNDLE_LOGITS_ELEMENT_V1_F32, BundleAbiError,
    BundleContextHandleV1, BundleDeviceBufferViewV1, BundleLogitsViewV1, DeviceBufferSpanV1,
    ExecuteResultSinkV1, validate_execute_logits_view_v1,
};

use crate::geometry::ModelLimits;
use crate::inputs::PhaseInputs;

mod exchange;
pub mod shards;
pub use exchange::{ExchangeRowsBuffers, ExchangeRowsState, exchange_rows};

/// Rows the host samples in one call.
const MAX_VIEW_ROWS: u32 = 64;

#[derive(Clone, Copy)]
pub struct PhaseOutput<'a> {
    pub inputs: &'a PhaseInputs,
    pub buffer: DeviceBufferSpanV1,
    pub element_type: u32,
    /// The phase's logits rows are packed in item order at rows 0..count
    /// rather than at their token positions.
    pub compact: bool,
}

/// Ranks that hold blocks of a phase's buffer rows: row `r` lives in
/// `ranks[r / block]`'s context and buffer, at the offset it would have in
/// the phase's own buffer.
#[derive(Clone, Copy)]
pub struct RowOwners<'a> {
    pub block: u32,
    pub ranks: &'a [(BundleContextHandleV1, DeviceBufferSpanV1)],
}

/// Every logits row of both phases as host views over full-vocabulary rows,
/// in the step's original item order.
pub fn ordered_views(
    limits: ModelLimits,
    context: BundleContextHandleV1,
    prefill: PhaseOutput<'_>,
    decode: PhaseOutput<'_>,
) -> Result<Vec<BundleLogitsViewV1>, BundleAbiError> {
    ordered_views_owned(limits, context, prefill, decode, None)
}

/// As [`ordered_views`], with each decode row's view on the rank that owns
/// it when `decode_owners` is set; one view never spans two ranks.
pub fn ordered_views_owned(
    limits: ModelLimits,
    context: BundleContextHandleV1,
    prefill: PhaseOutput<'_>,
    decode: PhaseOutput<'_>,
    decode_owners: Option<RowOwners<'_>>,
) -> Result<Vec<BundleLogitsViewV1>, BundleAbiError> {
    for inputs in [prefill.inputs, decode.inputs] {
        if inputs.item_order.len() != inputs.emits_row.len()
            || inputs.item_order.len() != inputs.prompt_rows.len()
            || inputs.item_order.len() != inputs.q_start.len()
            || inputs.item_order.len() != inputs.q_len.len()
            || inputs.q_len.iter().any(|&length| length <= 0)
        {
            return Err(BundleAbiError::InvalidLogitsView);
        }
    }
    let mut output_items = [prefill.inputs, decode.inputs]
        .into_iter()
        .flat_map(|inputs| {
            inputs.item_order.iter().enumerate().map(|(row, &item)| {
                let count = if inputs.prompt_rows[row] || every_row(inputs, row) {
                    inputs.q_len[row] as u32
                } else {
                    u32::from(inputs.emits_row[row])
                };
                (item, count)
            })
        })
        .collect::<Vec<_>>();
    output_items.sort_unstable_by_key(|&(item, _)| item);
    if output_items.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(BundleAbiError::InvalidLogitsView);
    }
    let mut slot = 0_u32;
    let mut first_slot = BTreeMap::new();
    for (item, count) in output_items {
        first_slot.insert(item, slot);
        slot = slot
            .checked_add(count)
            .ok_or(BundleAbiError::InvalidLogitsView)?;
    }
    let width = limits.vocab();
    let mut rows = Vec::new();
    for (phase, owners) in [(prefill, None), (decode, decode_owners)] {
        let element_bytes = match phase.element_type {
            BUNDLE_LOGITS_ELEMENT_V1_F32 => 4,
            BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2,
            _ => return Err(BundleAbiError::InvalidLogitsView),
        };
        let mut packed = 0_u64;
        for (phase_row, &item) in phase.inputs.item_order.iter().enumerate() {
            let prompt = phase.inputs.prompt_rows[phase_row];
            // Prompt-scored and verify items emit a row per input token.
            let all_rows = prompt || every_row(phase.inputs, phase_row);
            if !all_rows && !phase.inputs.emits_row[phase_row] {
                continue;
            }
            let start = u64::try_from(phase.inputs.q_start[phase_row])
                .map_err(|_| BundleAbiError::InvalidLogitsView)?;
            let length = u64::try_from(phase.inputs.q_len[phase_row])
                .map_err(|_| BundleAbiError::InvalidLogitsView)?;
            let row_count = if all_rows {
                u32::try_from(length).map_err(|_| BundleAbiError::InvalidLogitsView)?
            } else {
                1
            };
            let row = if phase.compact {
                let row = packed;
                packed += u64::from(row_count);
                row
            } else if all_rows {
                start
            } else {
                start
                    .checked_add(length)
                    .and_then(|end| end.checked_sub(1))
                    .ok_or(BundleAbiError::InvalidLogitsView)?
            };
            if row_count == 0 {
                return Err(BundleAbiError::InvalidLogitsView);
            }
            let offset = row
                .checked_mul(u64::from(width) * element_bytes)
                .and_then(|delta| phase.buffer.offset.checked_add(delta))
                .ok_or(BundleAbiError::InvalidLogitsView)?;
            let end = offset
                .checked_add(u64::from(width) * element_bytes * u64::from(row_count))
                .ok_or(BundleAbiError::InvalidLogitsView)?;
            if phase
                .buffer
                .offset
                .checked_add(phase.buffer.length)
                .is_none_or(|limit| end > limit)
            {
                return Err(BundleAbiError::InvalidLogitsView);
            }
            let item = *first_slot
                .get(&item)
                .ok_or(BundleAbiError::InvalidLogitsView)?;
            // An owned row names its owner's context and buffer. An item
            // whose rows cross an owner block (verify items of different
            // lengths) splits into one view per owner.
            let mut segments = Vec::new();
            match owners {
                None => segments.push((0, row_count, context, 0, phase.buffer.buffer, offset)),
                Some(owners) => {
                    let block = u64::from(owners.block.max(1));
                    let mut first = 0_u32;
                    while first < row_count {
                        let at = row + u64::from(first);
                        let owner = u32::try_from(at / block)
                            .map_err(|_| BundleAbiError::InvalidLogitsView)?;
                        let rows_here = ((owner as u64 + 1) * block - at)
                            .min(u64::from(row_count - first))
                            as u32;
                        let &(owner_context, owner_buffer) = owners
                            .ranks
                            .get(owner as usize)
                            .ok_or(BundleAbiError::InvalidLogitsView)?;
                        if owner_buffer.length != phase.buffer.length {
                            return Err(BundleAbiError::InvalidLogitsView);
                        }
                        let segment_offset =
                            offset + u64::from(first) * u64::from(width) * element_bytes;
                        segments.push((
                            first,
                            rows_here,
                            owner_context,
                            owner,
                            owner_buffer.buffer,
                            segment_offset - phase.buffer.offset + owner_buffer.offset,
                        ));
                        first += rows_here;
                    }
                }
            }
            for (segment, segment_rows, view_context, view_rank, buffer, offset) in segments {
                let mut view = BundleLogitsViewV1::empty();
                view.context = view_context;
                view.rank = view_rank;
                view.element_type = phase.element_type;
                view.buffer = BundleDeviceBufferViewV1 {
                    struct_size: std::mem::size_of::<BundleDeviceBufferViewV1>() as u32,
                    reserved: 0,
                    buffer,
                    offset,
                    length: u64::from(width) * element_bytes * u64::from(segment_rows),
                };
                view.row_stride_elements = u64::from(width);
                view.row_count = segment_rows;
                view.live_width = width;
                let item = item
                    .checked_add(segment)
                    .ok_or(BundleAbiError::InvalidLogitsView)?;
                view.first_item_index = item;
                for first in (0..segment_rows).step_by(MAX_VIEW_ROWS as usize) {
                    let mut chunk = view;
                    chunk.row_count = (segment_rows - first).min(MAX_VIEW_ROWS);
                    chunk.first_item_index = item
                        .checked_add(first)
                        .ok_or(BundleAbiError::InvalidLogitsView)?;
                    chunk.buffer.offset =
                        offset + u64::from(first) * u64::from(width) * element_bytes;
                    chunk.buffer.length =
                        u64::from(chunk.row_count) * u64::from(width) * element_bytes;
                    validate_execute_logits_view_v1(chunk)?;
                    rows.push(chunk);
                }
            }
        }
    }
    rows.sort_by_key(|view| view.first_item_index);
    if rows
        .windows(2)
        .any(|pair| pair[0].first_item_index + pair[0].row_count > pair[1].first_item_index)
    {
        return Err(BundleAbiError::InvalidLogitsView);
    }
    let scores_prompt = [prefill.inputs, decode.inputs]
        .iter()
        .any(|inputs| inputs.prompt_rows.iter().any(|&row| row));
    if scores_prompt {
        return Ok(rows);
    }
    Ok(merge_contiguous(rows))
}

/// Whether `row`'s item is a verify item, which owns one output slot per
/// input row.
fn every_row(inputs: &PhaseInputs, row: usize) -> bool {
    inputs.verify_rows.get(row).copied().unwrap_or(false)
}

/// The host samples one view per call, so output rows that follow each other
/// both in item order and in one buffer share a view. A step that scores
/// prompt rows keeps each item's bounded views separate.
fn merge_contiguous(rows: Vec<BundleLogitsViewV1>) -> Vec<BundleLogitsViewV1> {
    let mut merged: Vec<BundleLogitsViewV1> = Vec::with_capacity(rows.len());
    for view in rows {
        if let Some(last) = merged.last_mut()
            && last.context == view.context
            && last.rank == view.rank
            && last.buffer.buffer == view.buffer.buffer
            && last.element_type == view.element_type
            && last.row_stride_elements == view.row_stride_elements
            && last.live_width == view.live_width
            && last.first_item_index + last.row_count == view.first_item_index
            && last.buffer.offset + last.buffer.length == view.buffer.offset
            && last.row_count + view.row_count <= MAX_VIEW_ROWS
        {
            last.row_count += view.row_count;
            last.buffer.length += view.buffer.length;
            continue;
        }
        merged.push(view);
    }
    merged
}

pub fn emit(
    limits: ModelLimits,
    sink: &mut ExecuteResultSinkV1,
    context: BundleContextHandleV1,
    prefill: PhaseOutput<'_>,
    decode: PhaseOutput<'_>,
) -> Result<(), BundleAbiError> {
    for view in ordered_views(limits, context, prefill, decode)? {
        sink.emit_device_view(view)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::geometry::tests::LIMITS;
    use runtime_bundle_abi::{BundleContextHandleV1, BundleDeviceBufferHandleV1};

    use super::*;

    const WIDTH: u32 = LIMITS.vocab();
    const PREFILL_LOGIT_BYTES: u64 = 4;
    const DECODE_LOGIT_BYTES: u64 = 2;

    #[test]
    fn runtime_model_limits_set_each_logits_width() {
        let context = BundleContextHandleV1::from_parts(1, 1);
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let inputs = PhaseInputs {
            item_order: vec![0],
            emits_row: vec![true],
            prompt_rows: vec![false],
            q_start: vec![0],
            q_len: vec![1],
            ..PhaseInputs::default()
        };
        let empty = PhaseInputs::default();
        let output = |inputs| PhaseOutput {
            inputs,
            buffer: DeviceBufferSpanV1 {
                buffer,
                offset: 0,
                length: 64,
            },
            element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
            compact: true,
        };
        for vocab in [8, 16, 8] {
            let limits = ModelLimits::new(vocab, 128, 64).unwrap();
            let views = ordered_views(limits, context, output(&inputs), output(&empty)).unwrap();
            assert_eq!(views.len(), 1);
            assert_eq!(views[0].live_width, vocab);
            assert_eq!(views[0].row_stride_elements, u64::from(vocab));
            assert_eq!(views[0].buffer.length, u64::from(vocab) * 4);
        }
    }

    #[test]
    fn scored_prefill_views_respect_the_abi_row_limit() {
        let context = BundleContextHandleV1::from_parts(1, 1);
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let row_bytes = u64::from(WIDTH) * PREFILL_LOGIT_BYTES;
        for count in [64_u32, 65, 128, 130, 8192] {
            let prefill = PhaseInputs {
                item_order: vec![0],
                emits_row: vec![true],
                prompt_rows: vec![true],
                q_start: vec![0],
                q_len: vec![count as i32],
                ..PhaseInputs::default()
            };
            let decode = PhaseInputs::default();
            let output = |inputs| PhaseOutput {
                inputs,
                buffer: DeviceBufferSpanV1::new(buffer, 256, u64::from(count) * row_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: true,
            };
            let views = ordered_views(LIMITS, context, output(&prefill), output(&decode)).unwrap();
            let mut first = 0;
            for view in views {
                validate_execute_logits_view_v1(view).unwrap();
                assert_eq!(view.first_item_index, first);
                assert_eq!(view.row_count, (count - first).min(64));
                assert_eq!(view.buffer.offset, 256 + u64::from(first) * row_bytes);
                assert_eq!(view.buffer.length, u64::from(view.row_count) * row_bytes);
                first += view.row_count;
            }
            assert_eq!(first, count);
        }
    }

    #[test]
    fn owned_decode_rows_are_viewed_on_their_ranks() {
        // Five decode items in a bucket of eight rows, blocks of two rows.
        let decode = PhaseInputs {
            item_order: vec![0, 1, 2, 3, 4],
            emits_row: vec![true; 5],
            prompt_rows: vec![false; 5],
            q_start: vec![0, 1, 2, 3, 4],
            q_len: vec![1; 5],
            ..PhaseInputs::default()
        };
        let prefill = PhaseInputs::default();
        let row_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let buffer = |rank| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, 10 + rank),
                512,
                8 * row_bytes,
            )
        };
        let ranks = (0..4)
            .map(|rank| (BundleContextHandleV1::from_parts(1, 1 + rank), buffer(rank)))
            .collect::<Vec<_>>();
        let output = |inputs, buffer| PhaseOutput {
            inputs,
            buffer,
            element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
            compact: false,
        };
        let views = ordered_views_owned(
            LIMITS,
            ranks[0].0,
            output(&prefill, buffer(0)),
            output(&decode, buffer(0)),
            Some(RowOwners {
                block: 2,
                ranks: &ranks,
            }),
        )
        .unwrap();
        // Rows merge within a rank and never across ranks.
        assert_eq!(views.len(), 3);
        for (view, (rank, first, rows)) in views.iter().zip([(0, 0, 2), (1, 2, 2), (2, 4, 1)]) {
            validate_execute_logits_view_v1(*view).unwrap();
            assert_eq!(view.rank, rank);
            assert_eq!(view.context, ranks[rank as usize].0);
            assert_eq!(view.buffer.buffer, ranks[rank as usize].1.buffer);
            assert_eq!(view.first_item_index, first);
            assert_eq!(view.row_count, rows);
            assert_eq!(view.buffer.offset, 512 + u64::from(first) * row_bytes);
        }
        // Without owners every row stays on rank zero in one view.
        let single = ordered_views(
            LIMITS,
            ranks[0].0,
            output(&prefill, buffer(0)),
            output(&decode, buffer(0)),
        )
        .unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!((single[0].rank, single[0].row_count), (0, 5));
        // A missing owner is refused.
        assert!(
            ordered_views_owned(
                LIMITS,
                ranks[0].0,
                output(&prefill, buffer(0)),
                output(&decode, buffer(0)),
                Some(RowOwners {
                    block: 1,
                    ranks: &ranks,
                }),
            )
            .is_err()
        );
    }

    #[test]
    fn owned_verify_items_split_at_owner_blocks() {
        // Verify items of 3, 6 and 2 rows over blocks of four rows: the
        // second item crosses from rank 0 through rank 2.
        let decode = PhaseInputs {
            item_order: vec![0, 1, 2],
            emits_row: vec![true; 3],
            prompt_rows: vec![false; 3],
            verify_rows: vec![true; 3],
            q_start: vec![0, 3, 9],
            q_len: vec![3, 6, 2],
            ..PhaseInputs::default()
        };
        let prefill = PhaseInputs::default();
        let row_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let buffer = |rank| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, 10 + rank),
                256,
                16 * row_bytes,
            )
        };
        let ranks = (0..4)
            .map(|rank| (BundleContextHandleV1::from_parts(1, 1 + rank), buffer(rank)))
            .collect::<Vec<_>>();
        let output = |inputs, buffer| PhaseOutput {
            inputs,
            buffer,
            element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
            compact: false,
        };
        let views = ordered_views_owned(
            LIMITS,
            ranks[0].0,
            output(&prefill, buffer(0)),
            output(&decode, buffer(0)),
            Some(RowOwners {
                block: 4,
                ranks: &ranks,
            }),
        )
        .unwrap();
        let shape = views
            .iter()
            .map(|view| {
                validate_execute_logits_view_v1(*view).unwrap();
                assert_eq!(view.context, ranks[view.rank as usize].0);
                assert_eq!(view.buffer.buffer, ranks[view.rank as usize].1.buffer);
                assert_eq!(
                    view.buffer.offset,
                    256 + u64::from(view.first_item_index) * row_bytes
                );
                (view.rank, view.first_item_index, view.row_count)
            })
            .collect::<Vec<_>>();
        assert_eq!(shape, [(0, 0, 4), (1, 4, 4), (2, 8, 3)]);
    }

    #[test]
    fn owned_rows_split_one_block_per_rank_at_every_rank_count() {
        // Eight decode rows split evenly across the ranks.
        let decode = PhaseInputs {
            item_order: (0..8).collect(),
            emits_row: vec![true; 8],
            prompt_rows: vec![false; 8],
            q_start: (0..8).collect(),
            q_len: vec![1; 8],
            ..PhaseInputs::default()
        };
        let prefill = PhaseInputs::default();
        let row_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let buffer = |rank| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, 10 + rank),
                0,
                8 * row_bytes,
            )
        };
        let output = |inputs, buffer| PhaseOutput {
            inputs,
            buffer,
            element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
            compact: false,
        };
        for count in crate::geometry::SUPPORTED_RANKS {
            let ranks = (0..count)
                .map(|rank| (BundleContextHandleV1::from_parts(1, 1 + rank), buffer(rank)))
                .collect::<Vec<_>>();
            let block = 8 / count;
            let views = ordered_views_owned(
                LIMITS,
                ranks[0].0,
                output(&prefill, buffer(0)),
                output(&decode, buffer(0)),
                Some(RowOwners {
                    block,
                    ranks: &ranks,
                }),
            )
            .unwrap();
            assert_eq!(views.len(), count as usize, "{count} ranks");
            for (rank, view) in (0..count).zip(&views) {
                assert_eq!(view.rank, rank);
                assert_eq!(view.context, ranks[rank as usize].0);
                assert_eq!(view.buffer.buffer, ranks[rank as usize].1.buffer);
                assert_eq!(
                    (view.first_item_index, view.row_count),
                    (rank * block, block)
                );
            }
        }
    }

    #[test]
    fn mixed_phase_views_follow_original_items_and_skip_unrequested_rows() {
        let prefill = PhaseInputs {
            item_order: vec![1, 3],
            emits_row: vec![true, false],
            prompt_rows: vec![false, false],
            q_start: vec![0, 2],
            q_len: vec![2, 3],
            ..PhaseInputs::default()
        };
        let decode = PhaseInputs {
            item_order: vec![0, 4],
            emits_row: vec![true, true],
            prompt_rows: vec![false, false],
            q_start: vec![0, 1],
            q_len: vec![1, 1],
            ..PhaseInputs::default()
        };
        let context = BundleContextHandleV1::from_parts(1, 1);
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let prefill_bytes = u64::from(WIDTH) * PREFILL_LOGIT_BYTES;
        let decode_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let views = ordered_views(
            LIMITS,
            context,
            PhaseOutput {
                inputs: &prefill,
                buffer: DeviceBufferSpanV1::new(buffer, 256, prefill_bytes * 5),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: false,
            },
            PhaseOutput {
                inputs: &decode,
                buffer: DeviceBufferSpanV1::new(buffer, 1_024_000, decode_bytes * 2),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
                compact: false,
            },
        )
        .unwrap();
        assert_eq!(
            views
                .iter()
                .map(|view| view.first_item_index)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(views[0].buffer.offset, 1_024_000);
        assert_eq!(views[0].element_type, BUNDLE_LOGITS_ELEMENT_V1_BF16);
        assert_eq!(views[1].element_type, BUNDLE_LOGITS_ELEMENT_V1_F32);
        assert_eq!(views[1].buffer.offset, 256 + prefill_bytes);
        assert_eq!(views[2].buffer.offset, 1_024_000 + decode_bytes);
    }

    #[test]
    fn contiguous_decode_rows_share_one_view() {
        let decode = PhaseInputs {
            item_order: vec![0, 1, 3, 4],
            emits_row: vec![true, true, true, true],
            prompt_rows: vec![false; 4],
            q_start: vec![0, 1, 2, 3],
            q_len: vec![1; 4],
            ..PhaseInputs::default()
        };
        let prefill = PhaseInputs {
            item_order: vec![2],
            emits_row: vec![true],
            prompt_rows: vec![false],
            q_start: vec![0],
            q_len: vec![7],
            ..PhaseInputs::default()
        };
        let context = BundleContextHandleV1::from_parts(1, 1);
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let decode_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let views = ordered_views(
            LIMITS,
            context,
            PhaseOutput {
                inputs: &prefill,
                buffer: DeviceBufferSpanV1::new(buffer, 1 << 30, 7 * decode_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
                compact: false,
            },
            PhaseOutput {
                inputs: &decode,
                buffer: DeviceBufferSpanV1::new(buffer, 512, 4 * decode_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
                compact: false,
            },
        )
        .unwrap();
        // The prefill item sits between decode items 1 and 3, so the decode
        // rows form two runs.
        assert_eq!(
            views
                .iter()
                .map(|view| (view.first_item_index, view.row_count))
                .collect::<Vec<_>>(),
            [(0, 2), (2, 1), (3, 2)]
        );
        assert_eq!(views[0].buffer.offset, 512);
        assert_eq!(views[0].buffer.length, 2 * decode_bytes);
        assert_eq!(views[2].buffer.offset, 512 + 2 * decode_bytes);
        assert_eq!(views[2].buffer.length, 2 * decode_bytes);
        for view in &views {
            validate_execute_logits_view_v1(*view).unwrap();
        }
    }

    #[test]
    fn verify_items_own_a_slot_per_row_and_share_merged_views() {
        let decode = PhaseInputs {
            decode: true,
            verify: true,
            item_order: vec![0, 2, 3],
            verify_rows: vec![true, false, true],
            emits_row: vec![true, true, true],
            prompt_rows: vec![false; 3],
            q_start: vec![0, 4, 5],
            q_len: vec![4, 1, 3],
            ..PhaseInputs::default()
        };
        let prefill = PhaseInputs {
            item_order: vec![1],
            emits_row: vec![true],
            prompt_rows: vec![false],
            q_start: vec![0],
            q_len: vec![6],
            ..PhaseInputs::default()
        };
        let context = BundleContextHandleV1::from_parts(1, 1);
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let decode_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let views = ordered_views(
            LIMITS,
            context,
            PhaseOutput {
                inputs: &prefill,
                buffer: DeviceBufferSpanV1::new(buffer, 1 << 30, 6 * decode_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
                compact: false,
            },
            PhaseOutput {
                inputs: &decode,
                buffer: DeviceBufferSpanV1::new(buffer, 0, 8 * decode_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
                compact: false,
            },
        )
        .unwrap();
        // Item 0 owns slots 0..4, the prefill item slot 4, the decode item
        // slot 5 and item 3 slots 6..9. The decode and last verify rows are
        // contiguous in both slots and buffer, so they share a view.
        assert_eq!(
            views
                .iter()
                .map(|view| (view.first_item_index, view.row_count, view.buffer.offset))
                .collect::<Vec<_>>(),
            [
                (0, 4, 0),
                (4, 1, (1 << 30) + 5 * decode_bytes),
                (5, 4, 4 * decode_bytes)
            ]
        );
    }

    #[test]
    fn flagged_prefill_emits_each_prompt_row_in_flattened_item_order() {
        let prefill = PhaseInputs {
            item_order: vec![2, 0],
            emits_row: vec![true, false],
            prompt_rows: vec![true, true],
            q_start: vec![0, 3],
            q_len: vec![3, 2],
            ..PhaseInputs::default()
        };
        let decode = PhaseInputs {
            item_order: vec![1],
            emits_row: vec![true],
            prompt_rows: vec![false],
            q_start: vec![0],
            q_len: vec![1],
            ..PhaseInputs::default()
        };
        let context = BundleContextHandleV1::from_parts(1, 1);
        let buffer = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let prefill_bytes = u64::from(WIDTH) * PREFILL_LOGIT_BYTES;
        let decode_bytes = u64::from(WIDTH) * DECODE_LOGIT_BYTES;
        let rows = ordered_views(
            LIMITS,
            context,
            PhaseOutput {
                inputs: &prefill,
                buffer: DeviceBufferSpanV1::new(buffer, 0, 5 * prefill_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                compact: false,
            },
            PhaseOutput {
                inputs: &decode,
                buffer: DeviceBufferSpanV1::new(buffer, 10 * prefill_bytes, decode_bytes),
                element_type: BUNDLE_LOGITS_ELEMENT_V1_BF16,
                compact: false,
            },
        )
        .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| (row.first_item_index, row.row_count))
                .collect::<Vec<_>>(),
            [(0, 2), (2, 1), (3, 3)]
        );
        assert_eq!(rows[0].buffer.offset, 3 * prefill_bytes);
        assert_eq!(rows[1].buffer.offset, 10 * prefill_bytes);
        assert_eq!(rows[2].buffer.offset, 0);
    }
}
