//! Retained target rows for lazily caught-up drafters.
//!
//! Each slot keeps a circular history of committed target rows, indexed by
//! sequence position. Decode and verify phases write every row into its
//! history cell inside the target graph (`chop_mimo_retain_rows_v1`), so a
//! commit only advances the committed range. A history holds `window` rows a
//! drafter may read plus [`SLACK_ROWS`], so the rows of an open step never
//! overwrite a committed row inside the window: cancelled and rejected rows
//! lie past the committed end and are rewritten by the next step.

#[cfg(test)]
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
#[cfg(test)]
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::topology::Tp;

#[cfg(test)]
use crate::contract::MAX_VERIFY_ROWS;
#[cfg(test)]
use crate::dispatch::BufferBindings;
#[cfg(test)]
use crate::pipeline::PreparedLaunch;
#[cfg(test)]
use crate::program_lower::RankStage;
#[cfg(test)]
use crate::selection::MimoPhase as _;

/// The retained-row kernel's entry: one per geometry, its TP4 topology's
/// own kernel `retain_rows`, which every rank count launches.
pub fn retain_rows_symbol(tp: Tp) -> String {
    geometry_entry(tp, "retain_rows")
}

/// The entry of the own kernel `name` that `tp`'s geometry compiles once for
/// every rank count, as the selected backend names it at the TP4 topology.
fn geometry_entry(tp: Tp, name: &str) -> String {
    let tp4 = Tp::of_geometry(tp.geometry, 4).expect("every served geometry serves TP4");
    crate::topology::bundle_symbol(tp4, name)
}

/// History rows past the window: the most rows one decode-phase item writes.
pub use model_mimo26_contract::resources::RETAINED_SLACK_ROWS as SLACK_ROWS;

/// Rows per slot of a history that serves a `window`-row drafter window.
#[must_use]
pub const fn capacity(window: u64) -> u64 {
    model_mimo26_contract::resources::retained_capacity(window)
}

/// The history row of `slot`'s sequence position `position`.
#[must_use]
pub const fn cell(slot: u32, position: u64, capacity: u64) -> u64 {
    slot as u64 * capacity + position % capacity
}

#[cfg(test)]
/// A history buffer: `slots` slots of `capacity` rows, `stride` BF16 values
/// apart.
#[derive(Clone, Copy, Debug)]
pub struct HistoryBuffer {
    pub span: DeviceBufferSpanV1,
    pub stride: u32,
    pub capacity: u64,
    pub slots: u32,
}

#[cfg(test)]
/// Copies `width` BF16 columns of every live row of a decode or verify phase
/// from `input` (row stride `input_stride`) into its history cell.
/// `history` starts at the column the copy writes in slot 0's first row.
pub fn retain_rows(
    bound: &RankStage<'_>,
    input: DeviceBufferSpanV1,
    input_stride: u32,
    history: HistoryBuffer,
    width: u32,
) -> Result<PreparedLaunch, Refusal> {
    if !bound.phase.token_bucketed() {
        return Err(Refusal("retained rows are written by decode phases only"));
    }
    let rows = u64::from(bound.phase.capacity()?);
    let HistoryBuffer {
        span: history,
        stride: history_stride,
        capacity,
        slots: slot_count,
    } = history;
    let slots = u64::from(slot_count);
    let capacity_u32 = u32::try_from(capacity).map_err(|_| Refusal("retained row capacity"))?;
    if width == 0
        || capacity <= SLACK_ROWS
        || !width.is_multiple_of(8)
        || !input_stride.is_multiple_of(8)
        || !history_stride.is_multiple_of(8)
        || width > input_stride
        || width > history_stride
        || input.offset % 16 != 0
        || history.offset % 16 != 0
        || input.length < ((rows - 1) * u64::from(input_stride) + u64::from(width)) * 2
        || history.length
            < ((slots * capacity - 1) * u64::from(history_stride) + u64::from(width)) * 2
    {
        return Err(Refusal("retained row geometry"));
    }
    let mut bindings = BufferBindings::default();
    let inputs = bound.inputs;
    let mut buffer = |span| Argument::DeviceBuffer(bindings.bind(span));
    let arguments = vec![
        buffer(input),
        buffer(history),
        buffer(inputs.slot_ids),
        buffer(inputs.token_batch),
        buffer(inputs.q_start),
        buffer(inputs.prefix_len),
        buffer(inputs.live_tokens),
        Argument::U32(width),
        Argument::U32(input_stride),
        Argument::U32(history_stride),
        Argument::U32(capacity_u32),
        Argument::U32(slot_count),
    ];
    Ok(PreparedLaunch {
        symbol: retain_rows_symbol(bound.tp),
        rows: rows as u32,
        plan: LaunchPlan {
            grid: [rows as u32, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments: ArgumentList(arguments),
        bindings,
    })
}

/// The history gather kernel's entry, one per geometry like
/// [`retain_rows_symbol`].
pub fn gather_rows_symbol(tp: Tp) -> String {
    geometry_entry(tp, "gather_rows")
}

#[cfg(test)]
/// Copies `width` BF16 columns of every live row's history cell into
/// `output` (row stride `output_stride`): row r is item `token_batch[r]`'s
/// sequence position `positions[r]`. A catch-up reads its committed rows this
/// way, one launch per rank for all its items.
pub fn gather_rows(
    bound: &RankStage<'_>,
    history: HistoryBuffer,
    output: DeviceBufferSpanV1,
    output_stride: u32,
    width: u32,
) -> Result<PreparedLaunch, Refusal> {
    let rows = u64::from(if bound.phase.token_bucketed() {
        bound.phase.capacity()?
    } else {
        bound.live_tokens
    });
    let HistoryBuffer {
        span: history,
        stride: history_stride,
        capacity,
        slots: slot_count,
    } = history;
    let slots = u64::from(slot_count);
    let capacity_u32 = u32::try_from(capacity).map_err(|_| Refusal("gathered row capacity"))?;
    if rows == 0
        || width == 0
        || capacity <= SLACK_ROWS
        || !width.is_multiple_of(8)
        || !output_stride.is_multiple_of(8)
        || !history_stride.is_multiple_of(8)
        || width > output_stride
        || width > history_stride
        || output.offset % 16 != 0
        || history.offset % 16 != 0
        || output.length < ((rows - 1) * u64::from(output_stride) + u64::from(width)) * 2
        || history.length
            < ((slots * capacity - 1) * u64::from(history_stride) + u64::from(width)) * 2
    {
        return Err(Refusal("gathered row geometry"));
    }
    let mut bindings = BufferBindings::default();
    let inputs = bound.inputs;
    let mut buffer = |span| Argument::DeviceBuffer(bindings.bind(span));
    let arguments = vec![
        buffer(history),
        buffer(output),
        buffer(inputs.slot_ids),
        buffer(inputs.token_batch),
        buffer(inputs.positions),
        buffer(inputs.live_tokens),
        Argument::U32(width),
        Argument::U32(history_stride),
        Argument::U32(output_stride),
        Argument::U32(capacity_u32),
        Argument::U32(slot_count),
    ];
    Ok(PreparedLaunch {
        symbol: gather_rows_symbol(bound.tp),
        rows: rows as u32,
        plan: LaunchPlan {
            grid: [rows as u32, 1, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments: ArgumentList(arguments),
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host model of one slot's history: the target writes every row of an
    /// open step; commit keeps a prefix of them, cancel keeps none.
    struct History {
        cells: Vec<Option<u64>>,
        window: u64,
        end: u64,
    }

    impl History {
        fn new(window: u64) -> Self {
            Self {
                cells: vec![None; capacity(window) as usize],
                window,
                end: 0,
            }
        }

        /// Writes rows `end..end + count`; the value of a row names its
        /// position and the attempt that wrote it.
        fn step(&mut self, count: u64, attempt: u64) {
            let capacity = capacity(self.window);
            for position in self.end..self.end + count {
                self.cells[cell(0, position, capacity) as usize] = Some(position << 8 | attempt);
            }
        }

        fn commit(&mut self, kept: u64) {
            self.end += kept;
        }

        /// Checks every committed position the window covers.
        fn check(&self, attempts: &[u64]) {
            let capacity = capacity(self.window);
            for position in self.end.saturating_sub(self.window)..self.end {
                assert_eq!(
                    self.cells[cell(0, position, capacity) as usize],
                    Some(position << 8 | attempts[position as usize]),
                    "position {position} at end {}",
                    self.end
                );
            }
        }
    }

    #[test]
    fn open_steps_never_overwrite_the_committed_window() {
        for window in [128, 1023] {
            let mut history = History::new(window);
            // attempts[p]: the attempt whose write of p was committed.
            let mut attempts = Vec::new();
            let mut state = 0x9e37_79b9_u64;
            for attempt in 0..6000_u64 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let count = (state >> 33) % u64::from(MAX_VERIFY_ROWS) + 1;
                history.step(count, attempt & 0xff);
                match (state >> 20) % 4 {
                    // Cancel, then the next attempt rewrites the same rows.
                    0 => {}
                    // Partial commit of a verify item.
                    1 => {
                        let kept = (state >> 40) % count + 1;
                        attempts.extend((0..kept).map(|_| attempt & 0xff));
                        history.commit(kept);
                    }
                    _ => {
                        attempts.extend((0..count).map(|_| attempt & 0xff));
                        history.commit(count);
                    }
                }
                history.check(&attempts);
            }
            assert!(history.end > 3 * capacity(window));
        }
    }

    #[test]
    fn a_slot_owns_capacity_rows_indexed_by_position() {
        assert_eq!(capacity(128), 136);
        assert_eq!(cell(0, 135, 136), 135);
        assert_eq!(cell(0, 136, 136), 0);
        assert_eq!(cell(3, 140, 136), 3 * 136 + 4);
    }

    /// The retain and gather entries are the geometry-wide own kernels at
    /// every rank count: the TP4 entry retagged with the geometry's kernel
    /// tag, as the bundle launched them before the backend named them.
    #[test]
    fn retain_entries_are_the_geometry_wide_own_kernels() {
        let mut checked = 0;
        for tp in crate::topology::all() {
            assert_eq!(
                super::retain_rows_symbol(tp),
                crate::topology::tagged("chop_mimo_retain_rows_v1", tp)
            );
            assert_eq!(
                super::gather_rows_symbol(tp),
                crate::topology::tagged("chop_mimo_gather_rows_v1", tp)
            );
            checked += 1;
        }
        assert!(checked >= 2);
    }
}
