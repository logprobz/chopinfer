//! Host bookkeeping of the context each request's DFlash cache holds.
//!
//! Commits retain raw target features for the bounded attention window.
//! Projection into K/V happens only before drafting and reads committed
//! features. Pending or rejected target rows never enter the context cache.
//! The projection cursor belongs to the device runtime; this state tracks
//! the contiguous committed target positions across commit and cancel.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH;

use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use crate::contract::CONTEXT_TOKENS;
#[cfg(test)]
use crate::contract::DECODE_ENTRY_ROWS;
use crate::contract::{MAX_VERIFY_ROWS, PREFILL_TOKENS};

/// Raw target rows needed by the drafter attention window.
#[cfg(test)]
pub const HISTORY_ROWS: u64 = V26_FLASH.dflash().sliding_window as u64 - 1;
/// Retained rows per slot, including the rows an open step writes.
#[cfg(test)]
pub const HISTORY_CAPACITY: u64 = crate::retain::capacity(HISTORY_ROWS);

pub fn history_rows(tp: crate::topology::Tp) -> u64 {
    tp.geometry.dflash().sliding_window as u64 - 1
}
pub fn history_capacity(tp: crate::topology::Tp) -> u64 {
    crate::retain::capacity(history_rows(tp))
}

/// One item of a step: `count` rows from position `start`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rows {
    pub request: u64,
    pub slot: u32,
    pub start: u64,
    pub count: u32,
    pub verify: bool,
    /// The step drafts this verify item's tokens from the cache.
    pub drafted: bool,
}

/// Committed cache contents of one request: positions `start..end`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Context {
    pub slot: u32,
    pub start: u64,
    pub end: u64,
}

pub struct State {
    tp: crate::topology::Tp,
    committed: BTreeMap<u64, Context>,
    pending: Option<Vec<Rows>>,
    /// The instance's served sequences: slots are below it, and a step holds
    /// at most two rows per item (a prefill chunk and a decode or verify
    /// row).
    slots: u32,
}

/// 32 served sequences.
#[cfg(test)]
impl Default for State {
    fn default() -> Self {
        Self::new(DECODE_ENTRY_ROWS)
    }
}

impl State {
    /// The bookkeeping of an instance serving `slots` sequences.
    #[cfg(test)]
    pub fn new(slots: u32) -> Self {
        Self::for_topology(crate::topology::Tp::TP4, slots)
    }

    pub fn for_topology(tp: crate::topology::Tp, slots: u32) -> Self {
        Self {
            tp,
            committed: BTreeMap::new(),
            pending: None,
            slots,
        }
    }

    pub fn context(&self, request: u64) -> Option<Context> {
        self.committed.get(&request).copied()
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Opens a step. A known request continues at its committed end in the
    /// same slot; an unknown one starts wherever its first rows are, but it
    /// cannot draft until the cache holds its context.
    pub fn prepare(&mut self, rows: Vec<Rows>) -> Result<(), &'static str> {
        if self.pending.is_some() || rows.is_empty() || rows.len() > 2 * self.slots as usize {
            return Err("invalid DFlash transaction");
        }
        let mut requests = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for row in &rows {
            let rows = if row.verify {
                2..=MAX_VERIFY_ROWS
            } else {
                1..=PREFILL_TOKENS
            };
            if !requests.insert(row.request)
                || !slots.insert(row.slot)
                || row.slot >= self.slots
                || !rows.contains(&row.count)
                || row.drafted && !row.verify
                || row
                    .start
                    .checked_add(u64::from(row.count))
                    .is_none_or(|end| end > self.tp.geometry.context_tokens() as u64)
            {
                return Err("invalid DFlash rows");
            }
            match self.committed.get(&row.request) {
                Some(context) if context.slot != row.slot => {
                    return Err("DFlash cache slot changed without retirement");
                }
                Some(context) if context.end != row.start => {
                    return Err("DFlash context is not contiguous");
                }
                Some(_) => {}
                None if row.drafted => return Err("DFlash cannot draft without context"),
                None => {
                    if self.committed.values().any(|old| old.slot == row.slot) {
                        return Err("DFlash cache slot is still held");
                    }
                }
            }
        }
        self.pending = Some(rows);
        Ok(())
    }

    /// Commits the open step. `kept` names the verify items in item order,
    /// as `commit_accepted_v1` does; other items commit every row.
    pub fn commit(&mut self, kept: &[(u64, u32)]) -> Result<(), &'static str> {
        let pending = self.pending.as_ref().ok_or("no DFlash transaction")?;
        let verify = pending.iter().filter(|row| row.verify).collect::<Vec<_>>();
        if verify.len() != kept.len()
            || verify.iter().zip(kept).any(|(row, &(request, count))| {
                request != row.request || count == 0 || count > row.count
            })
        {
            return Err("invalid DFlash kept rows");
        }
        let mut kept = kept.iter();
        for row in self.pending.take().ok_or("no DFlash transaction")? {
            let count = if row.verify {
                kept.next().ok_or("invalid DFlash kept rows")?.1
            } else {
                row.count
            };
            let start = self
                .committed
                .get(&row.request)
                .map_or(row.start, |context| context.start);
            self.committed.insert(
                row.request,
                Context {
                    slot: row.slot,
                    start,
                    end: row.start + u64::from(count),
                },
            );
        }
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<(), &'static str> {
        self.pending
            .take()
            .map(|_| ())
            .ok_or("no DFlash transaction")
    }

    /// Restores a complete retained window with no projected K/V cursor.
    pub fn restore(&mut self, request: u64, slot: u32, end: u64) -> Result<(), &'static str> {
        if self.pending.is_some()
            || self.committed.contains_key(&request)
            || self.committed.values().any(|context| context.slot == slot)
            || slot >= self.slots
            || end == 0
            || end > self.tp.geometry.context_tokens() as u64
        {
            return Err("invalid DFlash prefix restore");
        }
        self.committed.insert(
            request,
            Context {
                slot,
                start: end.saturating_sub(history_rows(self.tp)),
                end,
            },
        );
        Ok(())
    }

    pub fn retire(&mut self, request: u64) -> Result<(), &'static str> {
        if self.pending.is_some() {
            return Err("cannot retire during a DFlash transaction");
        }
        self.committed.remove(&request);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(start: u64, count: u32, verify: bool) -> Rows {
        Rows {
            request: 41,
            slot: 2,
            start,
            count,
            verify,
            drafted: verify,
        }
    }

    fn context(start: u64, end: u64) -> Option<Context> {
        Some(Context {
            slot: 2,
            start,
            end,
        })
    }

    #[test]
    fn restored_windows_draft_cancel_commit_and_retire() {
        let mut state = State::default();
        for end in [0, CONTEXT_TOKENS + 1] {
            assert!(state.restore(41, 2, end).is_err());
        }
        assert!(state.restore(41, DECODE_ENTRY_ROWS, 1).is_err());
        let mut wide = State::new(256);
        wide.restore(41, 255, 3).unwrap();
        assert!(wide.restore(42, 256, 3).is_err());
        for end in [3, 2045] {
            state.restore(41, 2, end).unwrap();
            let original = context(end.saturating_sub(HISTORY_ROWS), end);
            assert_eq!(state.context(41), original);
            assert!(state.restore(41, 3, end).is_err());
            assert!(state.restore(42, 2, end).is_err());
            state.prepare(vec![rows(end, 4, true)]).unwrap();
            assert!(state.restore(42, 3, end).is_err());
            assert!(state.commit(&[(41, 0)]).is_err());
            state.cancel().unwrap();
            assert_eq!(state.context(41), original);
            state.prepare(vec![rows(end, 4, true)]).unwrap();
            state.commit(&[(41, 2)]).unwrap();
            assert_eq!(
                state.context(41),
                context(end.saturating_sub(HISTORY_ROWS), end + 2)
            );
            state.retire(41).unwrap();
            assert_eq!(state.context(41), None);
        }
    }

    #[test]
    fn prefill_chunks_and_zero_draft_decodes_extend_the_context() {
        let mut state = State::default();
        for (start, count) in [(0, 2048), (2048, 1000), (3048, 1)] {
            state.prepare(vec![rows(start, count, false)]).unwrap();
            state.commit(&[]).unwrap();
        }
        assert_eq!(state.context(41), context(0, 3049));
        // A chunk must continue at the committed end.
        assert!(state.prepare(vec![rows(3050, 1, false)]).is_err());
        assert!(state.prepare(vec![rows(3048, 1, false)]).is_err());
        state.prepare(vec![rows(3049, 8, true)]).unwrap();
        state.commit(&[(41, 8)]).unwrap();
        assert_eq!(state.context(41), context(0, 3057));
    }

    #[test]
    fn verify_commits_keep_only_the_kept_rows_and_cancel_keeps_the_previous_end() {
        let mut state = State::default();
        state.prepare(vec![rows(0, 3, false)]).unwrap();
        state.commit(&[]).unwrap();
        state.prepare(vec![rows(3, 4, true)]).unwrap();
        assert!(state.commit(&[(41, 0)]).is_err());
        assert!(state.commit(&[(41, 5)]).is_err());
        assert!(state.commit(&[(99, 2)]).is_err());
        assert!(state.commit(&[]).is_err());
        state.commit(&[(41, 2)]).unwrap();
        assert_eq!(state.context(41), context(0, 5));
        state.prepare(vec![rows(5, 8, true)]).unwrap();
        assert!(state.retire(41).is_err());
        state.cancel().unwrap();
        assert!(state.cancel().is_err());
        assert_eq!(state.context(41), context(0, 5));
        // The cancelled step runs again from the same position.
        state.prepare(vec![rows(5, 8, true)]).unwrap();
        state.commit(&[(41, 1)]).unwrap();
        assert_eq!(state.context(41), context(0, 6));
    }

    #[test]
    fn drafting_needs_context_and_slots_follow_retirement() {
        let mut state = State::default();
        assert_eq!(
            state.prepare(vec![rows(0, 2, true)]),
            Err("DFlash cannot draft without context")
        );
        // An undrafted request may start past zero, for example after a
        // prefix restore; drafting then reads whatever context it has.
        state.prepare(vec![rows(512, 1, false)]).unwrap();
        state.commit(&[]).unwrap();
        assert_eq!(state.context(41), context(512, 513));
        let mut other = rows(0, 4, false);
        other.request = 42;
        assert_eq!(
            state.prepare(vec![other]),
            Err("DFlash cache slot is still held")
        );
        let mut moved = rows(513, 2, true);
        moved.slot = 3;
        assert!(state.prepare(vec![moved]).is_err());
        state.retire(41).unwrap();
        assert_eq!(state.context(41), None);
        state.prepare(vec![other]).unwrap();
        state.commit(&[]).unwrap();
        assert_eq!(
            state.context(42),
            Some(Context {
                slot: 2,
                start: 0,
                end: 4
            })
        );
    }

    #[test]
    fn malformed_steps_are_refused_before_any_change() {
        let mut state = State::default();
        assert!(state.prepare(Vec::new()).is_err());
        assert!(
            state
                .prepare(vec![rows(0, 1, false), rows(1, 1, false)])
                .is_err()
        );
        assert!(state.prepare(vec![rows(0, 1, true)]).is_err());
        assert!(state.prepare(vec![rows(0, 9, true)]).is_err());
        assert!(
            state
                .prepare(vec![rows(0, PREFILL_TOKENS + 1, false)])
                .is_err()
        );
        let mut drafted = rows(0, 2, false);
        drafted.drafted = true;
        assert!(state.prepare(vec![drafted]).is_err());
        let mut far = rows(CONTEXT_TOKENS - 1, 2, false);
        far.request = 7;
        assert!(state.prepare(vec![far]).is_err());
        assert!(state.commit(&[]).is_err());
        state.prepare(vec![rows(0, 1, false)]).unwrap();
        assert!(state.prepare(vec![rows(1, 1, false)]).is_err());
    }
}
