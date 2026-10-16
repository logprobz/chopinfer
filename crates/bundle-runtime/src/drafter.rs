//! MTP drafter bookkeeping: which drafter rows each step runs, where their
//! inputs come from, and what commit keeps.
//!
//! Drafter row `p` of a request reads the target's hidden row of token `p`
//! and the token at `p + 1`, sits at position `p`, and predicts token
//! `p + 2`. Its KV lives in the request's pages beside the target's, so
//! every row up to the last committed token but one must run once, in
//! order: the slot's `frontier` counts the rows that have.
//!
//! A slot keeps the target hidden rows of its tokens from the frontier on in
//! `RETAINED_ROWS` retained rows, by position. Each step:
//!
//! - a drafting verify item (committed length `c`, bonus token `t_c`) first
//!   runs rows `frontier..c`; the last reads the bonus and drafts, then
//!   one chained row per further draft reads the drafter's own hidden row
//!   and the previous draft. Commit moves the frontier to `c` and retains
//!   the hidden rows of the kept rows.
//! - a continuing prefill item first runs the rows its earlier steps left
//!   (reading retained rows and ending with the chunk's first token) before
//!   the target, then every row its chunk completes after the target, and
//!   retains its last hidden row. So no run exceeds a prefill chunk's rows.
//! - a decode item runs nothing until its retained rows would overflow,
//!   then catches up before the target runs.
//!
//! Rows past the frontier that a step writes (chained drafts, a cancelled
//! step) are overwritten when the frontier reaches them, because a drafter
//! row reads only KV (and any selection keys) of earlier positions.
//!
//! A bundle names its drafter's constants and the token its row at position
//! 0 embeds in a [`Params`] type.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::inputs::PhaseInputs;

/// A bundle's drafter constants.
pub trait Params: Copy + Debug + Default + Eq + Send + Sync + 'static {
    /// Slots of a default drafter: the items one step drafts for at most.
    const SLOTS: u32;
    /// Rows of one verify item at most.
    const MAX_VERIFY_ROWS: u32;
    /// Target hidden rows of each slot the drafter has not consumed yet, by
    /// position modulo this many: a verify item's accepted rows, or a decode
    /// item's rows until the drafter catches up. At least `MAX_VERIFY_ROWS`.
    const RETAINED_ROWS: u32;
    /// Rows one drafter run takes at most: a whole prefill chunk, or the
    /// retained rows of every item.
    const DRAFTER_ROWS: u64;

    /// The token id the drafter row at position 0 embeds, given the token
    /// it would embed otherwise. By default that token; a drafter that
    /// embeds no token there returns an id that gathers zeros.
    fn position_zero_token(token: u32) -> u32 {
        token
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftError {
    /// The slot's drafter state is too far behind for this step.
    History,
    /// A step or commit this bookkeeping cannot follow.
    Step,
}

/// The drafter state of the request a slot holds; the step ledger
/// (`crate::ledger`) says which request that is and how many tokens it
/// committed.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Slot {
    /// Drafter rows that ran and were committed.
    frontier: u64,
    /// The last committed tokens by position modulo their count.
    tokens: Box<[u32]>,
}

impl Slot {
    fn new(history: usize) -> Self {
        Self {
            frontier: 0,
            tokens: vec![0; history].into_boxed_slice(),
        }
    }

    fn token(&self, position: u64) -> u32 {
        self.tokens[position as usize % self.tokens.len()]
    }

    fn set_token(&mut self, position: u64, token: u32) {
        let history = self.tokens.len();
        self.tokens[position as usize % history] = token;
    }
}

/// Where a drafter row's target hidden row comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    /// Retained entry `slot * RETAINED_ROWS + position % RETAINED_ROWS`.
    Retained(u32),
    /// Row of the hidden rows the previous program left: the target
    /// phase's, or the previous drafter run's.
    Last(u32),
}

impl Source {
    /// The code of the drafter's hidden-row gather.
    pub fn code(self) -> i32 {
        match self {
            Self::Retained(entry) => -1 - entry as i32,
            Self::Last(row) => row as i32,
        }
    }
}

/// One item of a drafter run: its rows are consecutive positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunItem {
    pub request: u64,
    pub slot: u32,
    pub first: u64,
    /// Per row: its hidden-row source and its token, or the drafting item
    /// whose latest draft is the token.
    pub rows: Vec<(Source, Token)>,
    /// The request's page table and committed-or-appended length, as the
    /// target phase received them.
    pub pages: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Token {
    Known(u32),
    Draft(u32),
}

/// One draft row: its run row, drafting item and the verify token it fills.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftRow {
    pub row: u32,
    pub item: u32,
    pub target: i32,
}

/// One drafter run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Run<P> {
    pub items: Vec<RunItem>,
    pub drafts: Vec<DraftRow>,
    params: PhantomData<P>,
}

impl<P: Params> Run<P> {
    pub fn rows(&self) -> u32 {
        self.items.iter().map(|item| item.rows.len() as u32).sum()
    }

    /// The run as target-shaped phase inputs: tokens (drafts read as 0 and
    /// are written on the device; the row at position 0 reads
    /// `P::position_zero_token`), positions, items, pages and the draft
    /// rows' logits.
    pub fn inputs(&self) -> PhaseInputs {
        let mut phase = PhaseInputs {
            decode: true,
            ..PhaseInputs::default()
        };
        for (index, item) in self.items.iter().enumerate() {
            let count = item.rows.len() as i32;
            let first = item.first as i32;
            phase.item_order.push(index);
            phase.request_ids.push(item.request);
            phase.emits_row.push(false);
            phase.prompt_rows.push(false);
            phase.verify_rows.push(false);
            phase.slot_ids.push(item.slot as i32);
            phase.q_start.push(phase.token_ids.len() as i32);
            phase.q_len.push(count);
            phase.prefix_len.push(first);
            phase.seq_lens.push(first + count);
            phase.last_positions.push(i64::from(first + count - 1));
            phase.block_tables.push(item.pages.clone());
            for (offset, &(_, token)) in item.rows.iter().enumerate() {
                let position = first + offset as i32;
                let token = match token {
                    Token::Known(token) => token,
                    Token::Draft(_) => 0,
                };
                phase.token_ids.push(if position == 0 {
                    P::position_zero_token(token)
                } else {
                    token
                });
                phase.positions.push(position);
                phase.token_batch.push(index as i32);
            }
        }
        phase
    }

    /// Per row, the hidden-row gather's source and token codes.
    pub fn codes(&self) -> (Vec<i32>, Vec<i32>) {
        self.items
            .iter()
            .flat_map(|item| &item.rows)
            .map(|&(source, token)| {
                (
                    source.code(),
                    match token {
                        Token::Known(_) => -1,
                        Token::Draft(item) => item as i32,
                    },
                )
            })
            .unzip()
    }
}

/// A step's drafter work.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Plan<P> {
    /// Runs before the target: catch-up and the first draft, then one run
    /// per further draft.
    pub before: Vec<Run<P>>,
    /// The run after the prefill phase: every row its items complete.
    pub prefill: Option<Run<P>>,
    /// Per drafting item: request, verify phase item and drafts.
    pub drafting: Vec<(u64, usize, u32)>,
    /// What commit does per item.
    commits: Vec<Commit>,
}

impl<P: Params> Plan<P> {
    pub fn is_empty(&self) -> bool {
        self.before.is_empty() && self.prefill.is_none()
    }

    /// What commit makes of the step, which the open step keeps.
    pub fn pending(&self) -> Pending {
        Pending(self.commits.clone())
    }
}

/// An open step's commit of every item, until the ledger closes the step.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Pending(Vec<Commit>);

impl Pending {
    /// Replaces the input tokens of `request` (a drafting item's drafts
    /// arrive after planning).
    pub fn set_tokens(&mut self, request: u64, tokens: &[u32]) -> Result<(), DraftError> {
        let commit = self
            .0
            .iter_mut()
            .find(|commit| commit.request == request)
            .ok_or(DraftError::Step)?;
        if commit.tokens.len() != tokens.len() {
            return Err(DraftError::Step);
        }
        commit.tokens = tokens.to_vec();
        Ok(())
    }
}

/// What commit makes of one item.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Commit {
    request: u64,
    slot: u32,
    /// Committed tokens before the step, and this step's input tokens.
    length: u64,
    tokens: Vec<u32>,
    /// The frontier after the step's rows, or `None` for a verify item,
    /// whose frontier becomes its length before the step.
    frontier: Option<u64>,
    /// Whether this is a verify item that keeps some of its rows.
    verify: bool,
    /// The phase (0 prefill, 1 decode) and staged entry of each input row's
    /// target hidden row, when commit retains it.
    staged: (usize, u32),
    /// Whether every row's hidden row is staged (decode) or only the last
    /// (prefill).
    every_row: bool,
}

/// One retained-row copy commit makes: staged entry of a phase into a
/// retained entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Retain {
    pub phase: usize,
    pub staged: u32,
    pub retained: u32,
}

#[derive(Clone, Debug)]
pub struct Drafter<P> {
    slots: Vec<Slot>,
    params: PhantomData<P>,
}

impl<P: Params> Default for Drafter<P> {
    fn default() -> Self {
        Self::new(P::SLOTS)
    }
}

const fn retained<P: Params>(slot: u32, position: u64) -> u32 {
    slot * P::RETAINED_ROWS + (position % P::RETAINED_ROWS as u64) as u32
}

impl<P: Params> Drafter<P> {
    /// Committed tokens a slot remembers: enough for every retained row's
    /// next token.
    const HISTORY: usize = 2 * P::RETAINED_ROWS as usize;

    pub fn new(slots: u32) -> Self {
        // Every verify item's kept rows fit its retained rows.
        const { assert!(P::RETAINED_ROWS >= P::MAX_VERIFY_ROWS) };
        Self {
            slots: vec![Slot::new(Self::HISTORY); slots as usize],
            params: PhantomData,
        }
    }

    /// The state of a slot whose request continues at `length` tokens, which
    /// the ledger checked.
    fn continuing(&self, slot: u32, length: u64) -> Result<&Slot, DraftError> {
        let state = self.slots.get(slot as usize).ok_or(DraftError::History)?;
        if state.frontier > length {
            return Err(DraftError::History);
        }
        Ok(state)
    }

    /// Plans the drafter rows of a step whose items the ledger placed.
    /// `drafts[i]` is the number of drafts decode item `i` asks the bundle
    /// for (zero for a decode item or a verify item of one row).
    pub fn plan(
        &self,
        prefill: &PhaseInputs,
        decode: &PhaseInputs,
        drafts: &[u32],
    ) -> Result<Plan<P>, DraftError> {
        if drafts.len() != decode.request_ids.len() {
            return Err(DraftError::Step);
        }
        let mut plan = Plan::default();
        // Catch-up rows and the first draft row, before the target.
        let mut first = Run::default();
        let mut last_rows = BTreeMap::new();
        for (index, &count) in drafts.iter().enumerate() {
            let slot = decode.slot_ids[index] as u32;
            let request = decode.request_ids[index];
            let length = decode.prefix_len[index] as u64;
            let start = decode.q_start[index] as usize;
            let rows = decode.q_len[index] as usize;
            let tokens = decode.token_ids[start..start + rows].to_vec();
            let state = self.continuing(slot, length)?;
            let bonus = tokens[0];
            let catch_up = if count > 0 {
                if rows != count as usize + 1 || state.frontier >= length {
                    return Err(DraftError::Step);
                }
                state.frontier..length
            } else if length + rows as u64 - state.frontier > u64::from(P::RETAINED_ROWS) {
                state.frontier..length
            } else {
                length..length
            };
            if length - state.frontier > u64::from(P::RETAINED_ROWS) {
                return Err(DraftError::History);
            }
            let caught = !catch_up.is_empty();
            if caught {
                let rows = catch_up
                    .clone()
                    .map(|position| {
                        let next = position + 1;
                        let token = if next == length {
                            bonus
                        } else {
                            state.token(next)
                        };
                        (
                            Source::Retained(retained::<P>(slot, position)),
                            Token::Known(token),
                        )
                    })
                    .collect::<Vec<_>>();
                if count > 0 {
                    let row = first.rows() + rows.len() as u32 - 1;
                    first.drafts.push(DraftRow {
                        row,
                        item: plan.drafting.len() as u32,
                        target: start as i32 + 1,
                    });
                    last_rows.insert(plan.drafting.len(), row);
                }
                first.items.push(RunItem {
                    request,
                    slot,
                    first: catch_up.start,
                    rows,
                    pages: decode.block_tables[index].clone(),
                });
            }
            if count > 0 {
                plan.drafting.push((request, index, count));
            }
            plan.commits.push(Commit {
                request,
                slot,
                length,
                tokens,
                frontier: if count > 0 {
                    None
                } else if caught {
                    Some(length)
                } else {
                    Some(state.frontier)
                },
                verify: decode.verify_rows[index],
                staged: (1, start as u32),
                every_row: true,
            });
        }
        // A continuing prefill first catches up the rows its earlier steps
        // left, before the target: they read retained rows and end with the
        // chunk's first token. After the target, the drafter runs every row
        // the chunk itself completes, at most the chunk's rows minus one.
        let mut run = Run::default();
        for index in 0..prefill.request_ids.len() {
            let slot = prefill.slot_ids[index] as u32;
            let request = prefill.request_ids[index];
            let length = prefill.prefix_len[index] as u64;
            let start = prefill.q_start[index] as usize;
            let count = prefill.q_len[index] as usize;
            let tokens = prefill.token_ids[start..start + count].to_vec();
            let end = length + count as u64 - 1;
            if length > 0 {
                let state = self.continuing(slot, length)?;
                if length - state.frontier > u64::from(P::RETAINED_ROWS) {
                    return Err(DraftError::History);
                }
                let rows = (state.frontier..length)
                    .map(|position| {
                        let next = position + 1;
                        let token = if next < length {
                            state.token(next)
                        } else {
                            tokens[0]
                        };
                        (
                            Source::Retained(retained::<P>(slot, position)),
                            Token::Known(token),
                        )
                    })
                    .collect::<Vec<_>>();
                if !rows.is_empty() {
                    first.items.push(RunItem {
                        request,
                        slot,
                        first: state.frontier,
                        rows,
                        pages: prefill.block_tables[index].clone(),
                    });
                }
            }
            let rows = (length..end)
                .map(|position| {
                    (
                        Source::Last((start as u64 + position - length) as u32),
                        Token::Known(tokens[(position + 1 - length) as usize]),
                    )
                })
                .collect::<Vec<_>>();
            if !rows.is_empty() {
                run.items.push(RunItem {
                    request,
                    slot,
                    first: length,
                    rows,
                    pages: prefill.block_tables[index].clone(),
                });
            }
            plan.commits.push(Commit {
                request,
                slot,
                length,
                tokens,
                frontier: Some(end),
                verify: false,
                staged: (0, index as u32),
                every_row: false,
            });
        }
        if !first.items.is_empty() {
            plan.before.push(first);
        }
        // One chained row per further draft.
        let depth = plan
            .drafting
            .iter()
            .map(|&(_, _, count)| count)
            .max()
            .unwrap_or(0);
        for draft in 1..depth {
            let mut run = Run::default();
            let mut rows = BTreeMap::new();
            for (item, &(request, index, count)) in plan.drafting.iter().enumerate() {
                if count <= draft {
                    continue;
                }
                let row = run.rows();
                let length = decode.prefix_len[index] as u64;
                run.items.push(RunItem {
                    request,
                    slot: decode.slot_ids[index] as u32,
                    first: length - 1 + u64::from(draft),
                    rows: vec![(Source::Last(last_rows[&item]), Token::Draft(item as u32))],
                    pages: decode.block_tables[index].clone(),
                });
                run.drafts.push(DraftRow {
                    row,
                    item: item as u32,
                    target: decode.q_start[index] + 1 + draft as i32,
                });
                rows.insert(item, row);
            }
            last_rows = rows;
            plan.before.push(run);
        }
        if !run.items.is_empty() {
            plan.prefill = Some(run);
        }
        // Every run fits the drafter's buffers, or the step is refused
        // before the target runs.
        if plan
            .before
            .iter()
            .chain(&plan.prefill)
            .any(|run| u64::from(run.rows()) > P::DRAFTER_ROWS)
        {
            return Err(DraftError::Step);
        }
        Ok(plan)
    }

    /// Commits a step's `pending` work with each verify item's kept rows and
    /// returns the retained-row copies to make.
    pub fn commit(
        &mut self,
        pending: &Pending,
        kept: impl Fn(u64) -> Option<u32>,
    ) -> Result<Vec<Retain>, DraftError> {
        let mut copies = Vec::new();
        for commit in &pending.0 {
            let kept_rows = if commit.verify {
                kept(commit.request)
                    .map(|rows| rows as usize)
                    .ok_or(DraftError::Step)?
            } else {
                commit.tokens.len()
            };
            if kept_rows == 0 || kept_rows > commit.tokens.len() {
                return Err(DraftError::Step);
            }
            let state = &mut self.slots[commit.slot as usize];
            if commit.length == 0 {
                *state = Slot::new(Self::HISTORY);
            }
            for (offset, &token) in commit.tokens[..kept_rows].iter().enumerate() {
                state.set_token(commit.length + offset as u64, token);
            }
            let length = commit.length + kept_rows as u64;
            let frontier = commit.frontier.unwrap_or(commit.length);
            // Retain the hidden row of every committed token from the
            // frontier on that this step computed.
            let (phase, staged) = commit.staged;
            for position in frontier.max(commit.length)..length {
                let offset = (position - commit.length) as u32;
                let staged = if commit.every_row {
                    staged + offset
                } else if position + 1 == length {
                    staged
                } else {
                    return Err(DraftError::Step);
                };
                copies.push(Retain {
                    phase,
                    staged,
                    retained: retained::<P>(commit.slot, position),
                });
            }
            state.frontier = frontier;
        }
        Ok(copies)
    }

    /// The frontier of the committed state `slot` holds, for a prefix
    /// snapshot.
    pub fn frontier(&self, slot: u32) -> u64 {
        self.slots[slot as usize].frontier
    }

    /// Makes `slot`, whose retained rows now hold a snapshot's, the state
    /// after committing `tokens` with `frontier` drafter rows run: the
    /// retained rows cover positions `frontier..tokens.len()`.
    pub fn restore(&mut self, slot: u32, tokens: &[u32], frontier: u64) -> Result<(), DraftError> {
        let length = tokens.len() as u64;
        // Row `length - 1` always waits for the next token, so the frontier
        // is below the length; the retained rows cover the rest.
        if slot as usize >= self.slots.len()
            || frontier >= length
            || length - frontier > u64::from(P::RETAINED_ROWS)
        {
            return Err(DraftError::History);
        }
        let mut state = Slot {
            frontier,
            ..Slot::new(Self::HISTORY)
        };
        for (position, &token) in tokens
            .iter()
            .enumerate()
            .skip(tokens.len().saturating_sub(Self::HISTORY))
        {
            state.set_token(position as u64, token);
        }
        self.slots[slot as usize] = state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Six retained rows, so no test leans on a particular bundle's count.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct Test;

    impl Params for Test {
        const SLOTS: u32 = 32;
        const MAX_VERIFY_ROWS: u32 = 6;
        const RETAINED_ROWS: u32 = 6;
        const DRAFTER_ROWS: u64 = CHUNK as u64;
    }

    /// A prefill chunk's rows, which one drafter run holds.
    const CHUNK: u32 = 64;
    const RETAINED_ROWS: u32 = Test::RETAINED_ROWS;

    /// A drafter that embeds no token at position 0.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    struct NoToken;

    impl Params for NoToken {
        const SLOTS: u32 = Test::SLOTS;
        const MAX_VERIFY_ROWS: u32 = Test::MAX_VERIFY_ROWS;
        const RETAINED_ROWS: u32 = Test::RETAINED_ROWS;
        const DRAFTER_ROWS: u64 = Test::DRAFTER_ROWS;

        fn position_zero_token(_: u32) -> u32 {
            u32::MAX
        }
    }

    type Drafter = super::Drafter<Test>;

    fn phase(decode: bool, items: &[(u64, u32, u64, &[u32])], verify: &[bool]) -> PhaseInputs {
        let mut phase = PhaseInputs {
            decode,
            ..PhaseInputs::default()
        };
        for (index, &(request, slot, previous, tokens)) in items.iter().enumerate() {
            phase.item_order.push(index);
            phase.request_ids.push(request);
            phase.slot_ids.push(slot as i32);
            phase.q_start.push(phase.token_ids.len() as i32);
            phase.q_len.push(tokens.len() as i32);
            phase.prefix_len.push(previous as i32);
            phase
                .verify_rows
                .push(verify.get(index).copied().unwrap_or(false));
            phase.block_tables.push(vec![slot as i32]);
            phase.token_ids.extend_from_slice(tokens);
        }
        phase
    }

    fn empty() -> PhaseInputs {
        PhaseInputs::default()
    }

    #[test]
    fn a_prefill_runs_every_completed_row_and_retains_its_last_row() {
        let mut drafter = Drafter::default();
        let prefill = phase(false, &[(7, 3, 0, &[10, 11, 12, 13])], &[]);
        let plan = drafter.plan(&prefill, &empty(), &[]).unwrap();
        assert!(plan.before.is_empty());
        let run = plan.prefill.as_ref().unwrap();
        // Rows 0..3 pair the chunk's hidden rows with the next tokens.
        assert_eq!(
            run.items[0].rows,
            vec![
                (Source::Last(0), Token::Known(11)),
                (Source::Last(1), Token::Known(12)),
                (Source::Last(2), Token::Known(13)),
            ]
        );
        assert_eq!(
            drafter.commit(&plan.pending(), |_| None).unwrap(),
            [Retain {
                phase: 0,
                staged: 0,
                retained: 3 * RETAINED_ROWS + 3
            }]
        );
        assert_eq!(drafter.frontier(3), 3);
        // A second chunk catches the retained row 3 up before the target,
        // then runs its own row after it.
        let chunk = phase(false, &[(7, 3, 4, &[14, 15])], &[]);
        let plan = drafter.plan(&chunk, &empty(), &[]).unwrap();
        assert_eq!(
            plan.before[0].items[0].rows,
            [(Source::Retained(3 * RETAINED_ROWS + 3), Token::Known(14))]
        );
        assert_eq!(
            plan.prefill.as_ref().unwrap().items[0].rows,
            [(Source::Last(0), Token::Known(15))]
        );
    }

    /// The row at position 0 embeds the parameters' token there, whichever
    /// run holds it; every later row embeds its next token.
    #[test]
    fn the_row_at_position_zero_embeds_the_parameters_token() {
        fn inputs<P: Params>() -> [Vec<u32>; 3] {
            let mut drafter = super::Drafter::<P>::default();
            let prefill = phase(false, &[(7, 3, 0, &[10, 11, 12, 13])], &[]);
            let plan = drafter.plan(&prefill, &empty(), &[]).unwrap();
            let first = plan.prefill.as_ref().unwrap().inputs();
            assert_eq!(first.positions, [0, 1, 2]);
            drafter.commit(&plan.pending(), |_| None).unwrap();
            let chunk = phase(false, &[(7, 3, 4, &[14, 15])], &[]);
            let plan = drafter.plan(&chunk, &empty(), &[]).unwrap();
            [
                first.token_ids,
                plan.before[0].inputs().token_ids,
                plan.prefill.as_ref().unwrap().inputs().token_ids,
            ]
        }
        assert_eq!(inputs::<Test>(), [vec![11, 12, 13], vec![14], vec![15]]);
        assert_eq!(
            inputs::<NoToken>(),
            [vec![u32::MAX, 12, 13], vec![14], vec![15]]
        );
    }

    /// Every run fits the drafter's rows: the rows a continuing prefill left
    /// run before the target, so a full chunk after a slot's retained rows
    /// runs its own rows minus one after it.
    #[test]
    fn a_full_chunk_after_outstanding_rows_fits_the_drafter() {
        let mut drafter = Drafter::default();
        let prompt = [1, 2, 3];
        let plan = drafter
            .plan(&phase(false, &[(7, 0, 0, &prompt)], &[]), &empty(), &[])
            .unwrap();
        drafter.commit(&plan.pending(), |_| None).unwrap();
        let full = u64::from(RETAINED_ROWS) + 2;
        for length in 3..full {
            let decode = phase(true, &[(7, 0, length, &[length as u32])], &[]);
            let plan = drafter.plan(&empty(), &decode, &[0]).unwrap();
            assert!(plan.is_empty());
            drafter.commit(&plan.pending(), |_| None).unwrap();
        }
        let chunk = vec![5; CHUNK as usize];
        let plan = drafter
            .plan(&phase(false, &[(7, 0, full, &chunk)], &[]), &empty(), &[])
            .unwrap();
        assert_eq!(plan.before[0].rows(), RETAINED_ROWS);
        assert_eq!(plan.prefill.as_ref().unwrap().rows(), CHUNK - 1);
        // A chunk whose completed rows exceed one run is refused.
        let long = vec![5; CHUNK as usize + 2];
        assert_eq!(
            drafter.plan(&phase(false, &[(8, 1, 0, &long)], &[]), &empty(), &[]),
            Err(DraftError::Step)
        );
    }

    #[test]
    fn drafting_catches_up_drafts_chains_and_commits_the_kept_rows() {
        let mut drafter = Drafter::default();
        let prefill = phase(false, &[(7, 1, 0, &[10, 11, 12])], &[]);
        let plan = drafter.plan(&prefill, &empty(), &[]).unwrap();
        drafter.commit(&plan.pending(), |_| None).unwrap();
        // Frontier 2, length 3: the bonus token 13 and three drafts.
        let verify = phase(true, &[(7, 1, 3, &[13, 0, 0, 0])], &[true]);
        let plan = drafter.plan(&empty(), &verify, &[3]).unwrap();
        assert_eq!(plan.before.len(), 3);
        assert_eq!(
            plan.before[0].items[0].rows,
            vec![(Source::Retained(RETAINED_ROWS + 2), Token::Known(13))]
        );
        assert_eq!(
            plan.before[0].drafts,
            [DraftRow {
                row: 0,
                item: 0,
                target: 1
            }]
        );
        let chained = &plan.before[2];
        assert_eq!(chained.items[0].first, 4);
        assert_eq!(
            chained.items[0].rows,
            vec![(Source::Last(0), Token::Draft(0))]
        );
        assert_eq!(chained.drafts[0].target, 3);
        assert_eq!(chained.codes(), (vec![0], vec![0]));
        assert_eq!(
            plan.before[0].codes(),
            (vec![-1 - (RETAINED_ROWS as i32 + 2)], vec![-1])
        );
        let mut pending = plan.pending();
        pending.set_tokens(7, &[13, 20, 21, 22]).unwrap();
        assert_eq!(pending.set_tokens(7, &[13]), Err(DraftError::Step));
        assert_eq!(
            pending.set_tokens(8, &[13, 20, 21, 22]),
            Err(DraftError::Step)
        );
        // Two rows kept: the bonus and the first draft.
        let copies = drafter
            .commit(&pending, |request| (request == 7).then_some(2))
            .unwrap();
        assert_eq!(
            copies,
            [
                Retain {
                    phase: 1,
                    staged: 0,
                    retained: RETAINED_ROWS + 3
                },
                Retain {
                    phase: 1,
                    staged: 1,
                    retained: RETAINED_ROWS + 4
                }
            ]
        );
        assert_eq!(drafter.frontier(1), 3);
        // The next step catches up row 3 (token 20) and drafts from row 4
        // with its bonus.
        let verify = phase(true, &[(7, 1, 5, &[30, 0])], &[true]);
        let plan = drafter.plan(&empty(), &verify, &[1]).unwrap();
        assert_eq!(
            plan.before[0].items[0].rows,
            vec![
                (Source::Retained(RETAINED_ROWS + 3), Token::Known(20)),
                (Source::Retained(RETAINED_ROWS + 4), Token::Known(30)),
            ]
        );
        assert_eq!(plan.before[0].drafts[0].row, 1);
    }

    #[test]
    fn decode_items_catch_up_only_when_retained_rows_would_overflow() {
        let mut drafter = Drafter::default();
        let prefill = phase(false, &[(7, 0, 0, &[1, 2])], &[]);
        let plan = drafter.plan(&prefill, &empty(), &[]).unwrap();
        drafter.commit(&plan.pending(), |_| None).unwrap();
        let mut length = 2;
        for step in 0..9 {
            let decode = phase(true, &[(7, 0, length, &[40 + step])], &[]);
            let plan = drafter.plan(&empty(), &decode, &[0]).unwrap();
            // Frontier 1: the decode steps before the last retained row
            // fill the retained rows, so the next one catches up first.
            let catching = step == RETAINED_ROWS - 1;
            assert_eq!(plan.before.is_empty(), !catching, "step {step}");
            if catching {
                let rows = &plan.before[0].items[0].rows;
                assert_eq!(rows.len(), RETAINED_ROWS as usize);
                assert_eq!(rows[rows.len() - 1].1, Token::Known(40 + step));
                assert!(plan.before[0].drafts.is_empty());
            }
            drafter.commit(&plan.pending(), |_| None).unwrap();
            length += 1;
        }
        assert_eq!(drafter.frontier(0), u64::from(RETAINED_ROWS) + 1);
    }

    /// A step's plan changes nothing until its commit: a cancelled step
    /// plans again as before. A new request's commit starts its slot over,
    /// and a restore fills it from a snapshot.
    #[test]
    fn only_commit_changes_a_slot_and_a_new_request_starts_it_over() {
        let mut drafter = Drafter::default();
        let prefill = phase(false, &[(7, 2, 0, &[1, 2, 3])], &[]);
        let plan = drafter.plan(&prefill, &empty(), &[]).unwrap();
        drafter.commit(&plan.pending(), |_| None).unwrap();
        let verify = phase(true, &[(7, 2, 3, &[4, 0, 0])], &[true]);
        let plan = drafter.plan(&empty(), &verify, &[2]).unwrap();
        assert_eq!(drafter.plan(&empty(), &verify, &[2]).unwrap(), plan);
        assert_eq!(
            drafter.commit(&plan.pending(), |_| None),
            Err(DraftError::Step)
        );
        // Another request prefills the slot from scratch.
        let fresh = phase(false, &[(8, 2, 0, &[5, 6])], &[]);
        let plan = drafter.plan(&fresh, &empty(), &[]).unwrap();
        drafter.commit(&plan.pending(), |_| None).unwrap();
        assert_eq!(drafter.frontier(2), 1);
        // A snapshot's frontier is below its length, within the retained
        // rows.
        assert_eq!(drafter.restore(2, &[1, 2, 3], 3), Err(DraftError::History));
        assert_eq!(drafter.restore(2, &[0; 12], 3), Err(DraftError::History));
        drafter.restore(2, &[1, 2, 3, 4], 2).unwrap();
        assert_eq!(drafter.frontier(2), 2);
        let decode = phase(true, &[(9, 2, 4, &[5])], &[]);
        assert!(drafter.plan(&empty(), &decode, &[0]).unwrap().is_empty());
    }

    /// A restore keeps the last `2 * RETAINED_ROWS` tokens: the catch-up
    /// after it reads them back by position.
    #[test]
    fn a_restore_keeps_the_tokens_the_retained_rows_read() {
        let mut drafter = Drafter::default();
        let tokens = (100..120).collect::<Vec<u32>>();
        let frontier = 20 - u64::from(RETAINED_ROWS);
        drafter.restore(0, &tokens, frontier).unwrap();
        let verify = phase(true, &[(7, 0, 20, &[120, 0])], &[true]);
        let plan = drafter.plan(&empty(), &verify, &[1]).unwrap();
        let rows = &plan.before[0].items[0].rows;
        assert_eq!(plan.before[0].items[0].first, frontier);
        let known = rows
            .iter()
            .map(|&(_, token)| match token {
                Token::Known(token) => token,
                Token::Draft(_) => unreachable!(),
            })
            .collect::<Vec<_>>();
        assert_eq!(known, (115..121).collect::<Vec<u32>>());
    }
}
