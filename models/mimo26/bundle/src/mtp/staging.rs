//! Committed target rows waiting for MTP catch-up.

use std::collections::{BTreeMap, BTreeSet};

use crate::contract::{DRAFT_ITEMS, MAX_VERIFY_ROWS, PREFILL_TOKENS};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TargetRows {
    pub request: u64,
    pub slot: u32,
    pub start: u64,
    pub tokens: Vec<u32>,
    pub verify: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatchUp {
    pub slot: u32,
    /// MTP uses the target hidden row's position, with the next token.
    pub start: u64,
    pub tokens: Vec<u32>,
}

pub struct Staging {
    tp: crate::topology::Tp,
    committed: BTreeMap<u64, TargetRows>,
    pending: Option<Vec<TargetRows>>,
    materialized: BTreeMap<u64, u64>,
}

#[cfg(test)]
impl Default for Staging {
    fn default() -> Self {
        Self::for_topology(crate::topology::Tp::TP4)
    }
}
impl Staging {
    pub fn for_topology(tp: crate::topology::Tp) -> Self {
        Self {
            tp,
            committed: Default::default(),
            pending: None,
            materialized: Default::default(),
        }
    }

    /// Catch-up reads only target rows kept by the previous transaction.
    /// Its last row pairs with the current step's bonus input.
    pub fn catch_up(
        &self,
        request: u64,
        start: u64,
        bonus: u32,
    ) -> Result<Option<CatchUp>, &'static str> {
        let Some(previous) = self.committed.get(&request) else {
            return if start == 0 {
                Ok(None)
            } else {
                Err("MTP target history is missing")
            };
        };
        if previous.start + previous.tokens.len() as u64 != start {
            return Err("MTP target history is not contiguous");
        }
        let first = self
            .materialized
            .get(&request)
            .copied()
            .unwrap_or(0)
            .max(previous.start);
        let skip = (first - previous.start) as usize;
        let mut tokens = previous.tokens[skip + 1..].to_vec();
        tokens.push(bonus);
        Ok(Some(CatchUp {
            slot: previous.slot,
            start: first,
            tokens,
        }))
    }

    /// Device taps write a separate pending buffer until prefix commit.
    pub fn prepare(&mut self, rows: Vec<TargetRows>) -> Result<(), &'static str> {
        if self.pending.is_some() || rows.is_empty() || rows.len() > DRAFT_ITEMS as usize {
            return Err("invalid MTP transaction");
        }
        let mut requests = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for row in &rows {
            if !requests.insert(row.request)
                || !slots.insert(row.slot)
                || row.slot >= DRAFT_ITEMS
                || row.tokens.is_empty()
                || row.tokens.len() > PREFILL_TOKENS as usize
                || row.verify && !(2..=MAX_VERIFY_ROWS as usize).contains(&row.tokens.len())
                || row
                    .start
                    .checked_add(row.tokens.len() as u64)
                    .is_none_or(|end| end > self.tp.geometry.context_tokens() as u64)
                || row
                    .tokens
                    .iter()
                    .any(|&token| token >= self.tp.geometry.vocab())
            {
                return Err("invalid MTP target rows");
            }
            match self.committed.get(&row.request) {
                Some(previous) if previous.start + previous.tokens.len() as u64 != row.start => {
                    return Err("MTP target history is not contiguous");
                }
                None if row.start != 0 => return Err("MTP target history is missing"),
                _ => {}
            }
            if self
                .committed
                .values()
                .any(|old| old.slot == row.slot && old.request != row.request)
                || self
                    .committed
                    .get(&row.request)
                    .is_some_and(|old| old.slot != row.slot)
            {
                return Err("MTP ring slot changed without retirement");
            }
        }
        self.pending = Some(rows);
        Ok(())
    }

    /// Validate every kept count before changing any sequence. Records are
    /// supplied in verify-item order, as in commit_accepted_v1.
    pub fn commit(&mut self, kept: &[(u64, u32)]) -> Result<(), &'static str> {
        self.commit_materialized(kept, &[])
    }

    pub fn commit_materialized(
        &mut self,
        kept: &[(u64, u32)],
        caught: &[(u64, u64)],
    ) -> Result<(), &'static str> {
        let pending = self.pending.as_ref().ok_or("no MTP transaction")?;
        let verify = pending.iter().filter(|row| row.verify).collect::<Vec<_>>();
        if verify.len() != kept.len()
            || verify.iter().zip(kept).any(|(row, &(request, count))| {
                request != row.request || count == 0 || count as usize > row.tokens.len()
            })
        {
            return Err("invalid MTP kept rows");
        }
        for mut row in self.pending.take().ok_or("no MTP transaction")? {
            if let Some(&(_, count)) = kept.iter().find(|&&(request, _)| request == row.request) {
                row.tokens.truncate(count as usize);
            }
            if let Some(previous) = self.committed.remove(&row.request) {
                let mut tokens = previous.tokens;
                tokens.extend(row.tokens);
                row.tokens = tokens;
                row.start = previous.start;
            }
            let skip = row
                .tokens
                .len()
                .saturating_sub(self.tp.geometry.window() as usize);
            row.start += skip as u64;
            row.tokens.drain(..skip);
            self.committed.insert(row.request, row);
        }
        for &(request, end) in caught {
            self.materialized.insert(request, end);
        }
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<(), &'static str> {
        self.pending.take().map(|_| ()).ok_or("no MTP transaction")
    }

    /// Slot and end of the retained committed history, at any step boundary.
    #[must_use]
    pub fn committed_boundary(&self, request: u64) -> Option<(u32, u64)> {
        self.committed
            .get(&request)
            .filter(|_| self.pending.is_none())
            .map(|row| (row.slot, row.start + row.tokens.len() as u64))
    }

    /// Restores the retained token suffix, leaving K/V to be rebuilt on draft.
    pub fn restore(&mut self, request: u64, slot: u32, tokens: &[u32]) -> Result<(), &'static str> {
        if self.pending.is_some()
            || self.committed.contains_key(&request)
            || self.committed.values().any(|row| row.slot == slot)
            || slot >= DRAFT_ITEMS
            || tokens.is_empty()
            || tokens.len() as u64 > self.tp.geometry.context_tokens() as u64
            || tokens
                .iter()
                .any(|&token| token >= self.tp.geometry.vocab())
        {
            return Err("invalid MTP prefix restore");
        }
        let start = tokens
            .len()
            .saturating_sub(self.tp.geometry.window() as usize);
        self.materialized.remove(&request);
        self.committed.insert(
            request,
            TargetRows {
                request,
                slot,
                start: start as u64,
                tokens: tokens[start..].to_vec(),
                verify: false,
            },
        );
        Ok(())
    }

    pub fn retire(&mut self, request: u64) -> Result<(), &'static str> {
        if self.pending.is_some() {
            return Err("cannot retire during MTP transaction");
        }
        self.committed.remove(&request);
        self.materialized.remove(&request);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(start: u64, tokens: &[u32], verify: bool) -> TargetRows {
        TargetRows {
            request: 41,
            slot: 2,
            start,
            tokens: tokens.to_vec(),
            verify,
        }
    }

    #[test]
    fn catch_up_uses_next_tokens_and_only_the_kept_target_prefix() {
        let mut state = Staging::default();
        state.prepare(vec![rows(0, &[10, 11, 12], false)]).unwrap();
        state.commit_materialized(&[], &[(41, 2)]).unwrap();
        assert_eq!(
            state.catch_up(41, 3, 13).unwrap(),
            Some(CatchUp {
                slot: 2,
                start: 2,
                tokens: vec![13]
            })
        );
        state
            .prepare(vec![rows(3, &[13, 14, 15, 16], true)])
            .unwrap();
        assert!(state.commit(&[(41, 0)]).is_err());
        assert!(state.commit(&[(99, 2)]).is_err());
        state.commit_materialized(&[(41, 2)], &[(41, 3)]).unwrap();
        assert_eq!(
            state.catch_up(41, 5, 90).unwrap(),
            Some(CatchUp {
                slot: 2,
                start: 3,
                tokens: vec![14, 90]
            })
        );
        assert!(state.catch_up(41, 7, 90).is_err());
    }

    #[test]
    fn cancelled_rows_and_reused_slots_cannot_enter_catch_up() {
        let mut state = Staging::default();
        state.prepare(vec![rows(0, &[10], false)]).unwrap();
        state.commit(&[]).unwrap();
        state.prepare(vec![rows(1, &[11, 12], true)]).unwrap();
        assert!(state.retire(41).is_err());
        state.cancel().unwrap();
        assert_eq!(state.catch_up(41, 1, 99).unwrap().unwrap().tokens, [99]);
        let mut replacement = rows(0, &[20], false);
        replacement.request = 42;
        assert!(state.prepare(vec![replacement.clone()]).is_err());
        state.retire(41).unwrap();
        state.prepare(vec![replacement]).unwrap();
        state.commit(&[]).unwrap();
        assert!(state.catch_up(41, 1, 99).is_err());
    }

    #[test]
    fn a_restored_committed_boundary_continues_like_the_captured_one() {
        let mut captured = Staging::default();
        captured
            .prepare(vec![rows(0, &[1, 2, 3, 4], false)])
            .unwrap();
        captured.commit(&[]).unwrap();
        assert_eq!(captured.committed_boundary(41), Some((2, 4)));

        let mut restored = Staging::default();
        restored.restore(41, 2, &[1, 2, 3, 4]).unwrap();
        assert_eq!(
            restored.catch_up(41, 4, 9).unwrap(),
            captured.catch_up(41, 4, 9).unwrap()
        );
        assert!(
            restored.restore(41, 3, &[1, 2, 3, 4]).is_err(),
            "a known request"
        );
        assert!(
            restored.restore(42, 2, &[1, 2, 3, 4]).is_err(),
            "a slot in use"
        );
        restored.prepare(vec![rows(4, &[9, 10], true)]).unwrap();
        assert_eq!(restored.committed_boundary(41), None);
        restored.commit(&[(41, 2)]).unwrap();
        assert_eq!(
            restored.committed_boundary(41),
            Some((2, 6)),
            "verify commits expose only their kept rows"
        );
    }

    #[test]
    fn verify_boundaries_keep_only_committed_hidden_rows_for_restore() {
        for kept in 1..=4 {
            let mut state = Staging::default();
            let mut tokens = (0..130).collect::<Vec<_>>();
            state.prepare(vec![rows(0, &tokens, false)]).unwrap();
            state.commit(&[]).unwrap();
            state
                .prepare(vec![rows(130, &[130, 131, 132, 133], true)])
                .unwrap();
            state.commit(&[(41, kept)]).unwrap();
            tokens.extend(130..130 + kept);
            assert_eq!(state.committed_boundary(41), Some((2, tokens.len() as u64)));
            let mut restored = Staging::default();
            restored.restore(42, 3, &tokens).unwrap();
            let original = state
                .catch_up(41, tokens.len() as u64, 999)
                .unwrap()
                .unwrap();
            let copy = restored
                .catch_up(42, tokens.len() as u64, 999)
                .unwrap()
                .unwrap();
            assert_eq!((original.start, original.tokens), (copy.start, copy.tokens));
            assert_eq!(copy.slot, 3);
        }
    }

    #[test]
    fn restored_history_is_bounded_and_rejects_invalid_or_pending_state() {
        let tokens = (0..300).collect::<Vec<_>>();
        let mut state = Staging::default();
        assert!(state.restore(41, 2, &[]).is_err());
        assert!(state.restore(41, 2, &[state.tp.geometry.vocab()]).is_err());
        assert!(state.restore(41, DRAFT_ITEMS, &[1]).is_err());
        state.restore(41, 2, &tokens).unwrap();
        assert_eq!(state.committed_boundary(41), Some((2, 300)));
        let caught = state.catch_up(41, 300, 900).unwrap().unwrap();
        assert_eq!(caught.start, 172);
        assert_eq!(caught.tokens, (173..300).chain([900]).collect::<Vec<_>>());
        state.prepare(vec![rows(300, &[900, 901], true)]).unwrap();
        assert!(state.restore(42, 3, &[1]).is_err());
        state.cancel().unwrap();
        assert_eq!(state.catch_up(41, 300, 900).unwrap().unwrap(), caught);
        state.retire(41).unwrap();
        state.restore(41, 2, &[10]).unwrap();
        assert_eq!(state.catch_up(41, 1, 11).unwrap().unwrap().tokens, [11]);
    }

    #[test]
    fn chunked_prefill_and_zero_draft_decode_keep_history_current() {
        let mut state = Staging::default();
        for (start, tokens) in [(0, vec![1, 2]), (2, vec![3, 4]), (4, vec![5])] {
            state.prepare(vec![rows(start, &tokens, false)]).unwrap();
            state.commit(&[]).unwrap();
        }
        assert_eq!(
            state.catch_up(41, 5, 6).unwrap().unwrap().tokens,
            [2, 3, 4, 5, 6]
        );
        assert!(state.prepare(vec![rows(6, &[7], false)]).is_err());
    }
    #[test]
    fn lazy_history_wraps_and_cancel_preserves_the_materialization_boundary() {
        let mut state = Staging::default();
        for start in 0..300 {
            state
                .prepare(vec![rows(start, &[start as u32], false)])
                .unwrap();
            state.commit(&[]).unwrap();
        }
        let caught = state.catch_up(41, 300, 300).unwrap().unwrap();
        assert_eq!(caught.start, 172);
        assert_eq!(caught.tokens, (173..=300).collect::<Vec<_>>());
        state
            .prepare(vec![rows(300, &[300, 301, 302, 303], true)])
            .unwrap();
        assert!(state.commit_materialized(&[(41, 0)], &[(41, 300)]).is_err());
        state.cancel().unwrap();
        assert_eq!(state.catch_up(41, 300, 300).unwrap().unwrap(), caught);
        state
            .prepare(vec![rows(300, &[300, 301, 302, 303], true)])
            .unwrap();
        state.commit_materialized(&[(41, 2)], &[(41, 300)]).unwrap();
        let next = state.catch_up(41, 302, 900).unwrap().unwrap();
        assert_eq!(next.start, 300);
        assert_eq!(next.tokens, [301, 900]);
        assert_eq!(state.committed[&41].tokens.len(), 128);
        state.retire(41).unwrap();
        assert!(state.materialized.is_empty());
        state.prepare(vec![rows(0, &[7], false)]).unwrap();
        state.commit(&[]).unwrap();
        assert_eq!(state.catch_up(41, 1, 8).unwrap().unwrap().tokens, [8]);
    }
}
