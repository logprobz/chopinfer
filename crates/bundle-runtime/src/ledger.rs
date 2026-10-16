//! One transaction ledger for per-request state.
//!
//! The ledger maps each request to the slot that holds its committed state
//! and the tokens that state covers, and runs one step at a time over it:
//! `begin` opens a step of items, `commit` ends it keeping each verify
//! item's accepted rows, and `cancel` ends it keeping nothing. It holds no
//! bytes. What a commit or cancel copies follows each buffer's declaration
//! ([`crate::state::Rollback`], [`crate::state::Cancel`]); the ledger
//! returns the items so the bundle can do it.
//!
//! A new request's slot is the first free one ([`Placement::Ledger`]) or the
//! one the host names ([`Placement::Host`]). A step claims new requests'
//! slots when it opens, and a cancel frees them. When a declared buffer
//! invalidates on cancel, a cancel frees every slot the step touched: those
//! requests retire and start again.

use std::sync::Arc;

use crate::state::{Cancel, RequestState, Rollback};

/// Why the ledger refused a call. Nothing changes when it does.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LedgerError {
    /// No slot is free.
    NoSlot,
    /// The request continues state no slot holds for it.
    UnknownRequest,
    /// The slot holds another request's state, or the request holds another
    /// slot.
    ForeignSlot,
    /// The request's committed state covers other tokens.
    Tokens,
    /// A step is open (one that holds the request, for a retire).
    Open,
    /// No step is open.
    NotOpen,
    /// The step's items are malformed.
    InvalidStep,
    /// Commit named other verify items or rows than the step ran.
    Kept,
    /// A foreign or expired commit plan, indicating a caller sequencing error.
    StalePlan,
}

/// Who picks a new request's slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Placement {
    /// The first free slot.
    Ledger,
    /// The slot each item names.
    Host,
}

/// One item of a step: `rows` rows of `request` from token `start`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Item {
    pub request: u64,
    /// The host's slot, under [`Placement::Host`].
    pub slot: Option<u32>,
    pub start: u64,
    pub rows: u32,
    /// Whether commit names the rows it keeps of this item.
    pub verify: bool,
}

/// An item of an open step, in its slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Placed {
    pub request: u64,
    pub slot: u32,
    pub start: u64,
    pub rows: u32,
    pub verify: bool,
    /// Whether the step claimed the slot for a new request.
    pub new: bool,
}

/// An open step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Step {
    pub items: Vec<Placed>,
    /// Per anchor item, the (request, tokens) whose state the step keeps at
    /// its anchor.
    pub anchors: Vec<(u64, u64)>,
}

impl Step {
    /// Item `item`'s index among the step's verify items, which take
    /// checkpoints in that order.
    pub fn verify_index(&self, item: usize) -> Option<u32> {
        self.items
            .get(item)
            .filter(|placed| placed.verify)
            .map(|_| {
                self.items[..item]
                    .iter()
                    .filter(|placed| placed.verify)
                    .count() as u32
            })
    }
}

/// How a committed item ended: `kept` of its `rows` rows from `start`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Committed {
    pub request: u64,
    pub slot: u32,
    pub start: u64,
    pub rows: u32,
    pub kept: u32,
}

/// What the declared state asks of the ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rules {
    /// A buffer invalidates the entries a cancelled step touched.
    pub invalidate_on_cancel: bool,
    /// The fewest positions of any buffer that restores a rejected tail. A
    /// partial commit of an item with more rows has overwritten an accepted
    /// position with a rejected one, and the shadow holds only the cell's
    /// value from before the step.
    pub tail_window: Option<u64>,
}

impl Rules {
    pub fn of(state: &RequestState) -> Self {
        Self {
            invalidate_on_cancel: state
                .buffers
                .iter()
                .any(|buffer| buffer.cancel == Cancel::Invalidate),
            tail_window: state
                .buffers
                .iter()
                .filter(|buffer| buffer.rollback == Rollback::RestoreTail)
                .filter_map(|buffer| buffer.scope.positions())
                .min(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Holder {
    request: u64,
    tokens: u64,
}

#[derive(Debug)]
pub struct Ledger {
    identity: Arc<()>,
    generation: u64,
    slots: Vec<Option<Holder>>,
    open: Option<Step>,
    /// Per anchor item, the (request, tokens) whose state the last committed
    /// step kept at its anchor, until a step opens or the request retires.
    anchors: Vec<Option<(u64, u64)>>,
    placement: Placement,
    rules: Rules,
    /// The most tokens a request's state covers.
    context: u64,
}

/// Validated ordered rows for one ledger's open step. Applying consumes it.
///
/// ```compile_fail
/// use bundle_runtime::ledger::{Ledger, Placement, Rules};
/// let mut ledger = Ledger::new(1, Placement::Ledger, Rules {
///     invalidate_on_cancel: false, tail_window: None,
/// }, 64);
/// let plan = ledger.plan_commit(&[]).unwrap();
/// ledger.apply_commit(plan).unwrap();
/// ledger.apply_commit(plan).unwrap(); // The first apply moved the plan.
/// ```
#[derive(Debug)]
pub struct CommitPlan {
    identity: Arc<()>,
    generation: u64,
    committed: Vec<Committed>,
}

impl CommitPlan {
    pub fn items(&self) -> &[Committed] {
        &self.committed
    }
}

impl Clone for Ledger {
    fn clone(&self) -> Self {
        Self {
            identity: Arc::new(()),
            generation: 0,
            slots: self.slots.clone(),
            open: self.open.clone(),
            anchors: self.anchors.clone(),
            placement: self.placement,
            rules: self.rules,
            context: self.context,
        }
    }
}

impl Ledger {
    fn advance_generation(&mut self) {
        if let Some(next) = self.generation.checked_add(1) {
            self.generation = next;
        } else {
            // Retained plans keep the old identity alive, so wrap cannot revive them.
            self.identity = Arc::new(());
            self.generation = 0;
        }
    }

    /// Exercise a declared tail-window refusal in fake-host bundle tests.
    #[cfg(feature = "test-support")]
    pub fn set_test_tail_window(&mut self, window: Option<u64>) {
        self.rules.tail_window = window;
        self.advance_generation();
    }

    pub fn new(slots: u32, placement: Placement, rules: Rules, context: u64) -> Self {
        Self {
            identity: Arc::new(()),
            generation: 0,
            slots: vec![None; slots as usize],
            open: None,
            anchors: Vec::new(),
            placement,
            rules,
            context,
        }
    }

    pub fn slot_of(&self, request: u64) -> Option<u32> {
        self.slots
            .iter()
            .position(|slot| slot.is_some_and(|holder| holder.request == request))
            .map(|slot| slot as u32)
    }

    /// The request whose state `slot` holds, and the tokens it covers.
    pub fn holder(&self, slot: u32) -> Option<(u64, u64)> {
        self.slots
            .get(slot as usize)
            .copied()
            .flatten()
            .map(|holder| (holder.request, holder.tokens))
    }

    pub fn step(&self) -> Option<&Step> {
        self.open.as_ref()
    }

    /// The slot each item would run in, without opening the step.
    pub fn preview(&self, items: &[Item]) -> Result<Vec<u32>, LedgerError> {
        Ok(self
            .place(items)?
            .iter()
            .map(|placed| placed.slot)
            .collect())
    }

    fn place(&self, items: &[Item]) -> Result<Vec<Placed>, LedgerError> {
        if self.open.is_some() {
            return Err(LedgerError::Open);
        }
        if items.is_empty() || items.len() > self.slots.len() {
            return Err(LedgerError::InvalidStep);
        }
        let mut taken = self.slots.iter().map(Option::is_some).collect::<Vec<_>>();
        let mut placed = Vec::<Placed>::with_capacity(items.len());
        for item in items {
            if item.rows == 0
                || placed.iter().any(|other| other.request == item.request)
                || item
                    .start
                    .checked_add(u64::from(item.rows))
                    .is_none_or(|end| end > self.context)
            {
                return Err(LedgerError::InvalidStep);
            }
            let (slot, new) = match (self.slot_of(item.request), self.placement, item.slot) {
                (Some(held), _, named) => {
                    if named.is_some_and(|named| named != held) {
                        return Err(LedgerError::ForeignSlot);
                    }
                    if self.holder(held).map(|(_, tokens)| tokens) != Some(item.start) {
                        return Err(LedgerError::Tokens);
                    }
                    (held, false)
                }
                (None, Placement::Host, Some(named)) => {
                    if item.start != 0 {
                        return Err(LedgerError::UnknownRequest);
                    }
                    match taken.get_mut(named as usize) {
                        Some(taken) if !*taken => *taken = true,
                        Some(_) => return Err(LedgerError::ForeignSlot),
                        None => return Err(LedgerError::InvalidStep),
                    }
                    (named, true)
                }
                (None, Placement::Ledger, None) => {
                    if item.start != 0 {
                        return Err(LedgerError::UnknownRequest);
                    }
                    let free = taken
                        .iter()
                        .position(|taken| !taken)
                        .ok_or(LedgerError::NoSlot)?;
                    taken[free] = true;
                    (free as u32, true)
                }
                (None, _, _) => return Err(LedgerError::InvalidStep),
            };
            placed.push(Placed {
                request: item.request,
                slot,
                start: item.start,
                rows: item.rows,
                verify: item.verify,
                new,
            });
        }
        Ok(placed)
    }

    /// Opens a step of `items`, claiming new requests' slots. The step
    /// rewrites the anchor spares, so the last step's anchors end.
    pub fn begin(
        &mut self,
        items: &[Item],
        anchors: Vec<(u64, u64)>,
    ) -> Result<&Step, LedgerError> {
        let items = self.place(items)?;
        for placed in items.iter().filter(|placed| placed.new) {
            self.slots[placed.slot as usize] = Some(Holder {
                request: placed.request,
                tokens: 0,
            });
        }
        self.anchors.clear();
        self.advance_generation();
        Ok(self.open.insert(Step { items, anchors }))
    }

    /// Validates a commit without changing slots, anchors or the open step.
    fn validate_commit(&self, kept: &[(u64, u32)]) -> Result<(), LedgerError> {
        let step = self.open.as_ref().ok_or(LedgerError::NotOpen)?;
        let mut verify = step.items.iter().filter(|placed| placed.verify);
        if kept.len() != verify.clone().count()
            || kept.iter().any(|&(request, rows)| {
                verify.next().is_none_or(|placed| {
                    placed.request != request
                        || rows == 0
                        || rows > placed.rows
                        || rows < placed.rows
                            && self
                                .rules
                                .tail_window
                                .is_some_and(|window| u64::from(placed.rows) > window)
                })
            })
        {
            return Err(LedgerError::Kept);
        }
        Ok(())
    }

    /// Ends the open step. `kept` names, in item order, every verify item
    /// and the rows it keeps (one through its rows, and all of them when it
    /// has more rows than the tail window); every other item keeps all of
    /// its rows. Returns how each item ended, in item order.
    pub fn commit(&mut self, kept: &[(u64, u32)]) -> Result<Vec<Committed>, LedgerError> {
        let plan = self.plan_commit(kept)?;
        self.apply_commit(plan)
    }

    /// Resolves accepted rows without changing the open step or ownership.
    /// Bundles call this before advancing device or drafter state.
    pub fn plan_commit(&self, kept: &[(u64, u32)]) -> Result<CommitPlan, LedgerError> {
        self.validate_commit(kept)?;
        let step = self.open.as_ref().expect("checked above");
        let mut accepted = kept.iter();
        let committed = step
            .items
            .iter()
            .map(|placed| Committed {
                request: placed.request,
                slot: placed.slot,
                start: placed.start,
                rows: placed.rows,
                kept: if placed.verify {
                    accepted.next().expect("one per verify item").1
                } else {
                    placed.rows
                },
            })
            .collect();
        Ok(CommitPlan {
            identity: Arc::clone(&self.identity),
            generation: self.generation,
            committed,
        })
    }

    /// Rejects foreign or expired plans before changing any ledger state.
    pub fn apply_commit(&mut self, plan: CommitPlan) -> Result<Vec<Committed>, LedgerError> {
        if !Arc::ptr_eq(&self.identity, &plan.identity)
            || self.generation != plan.generation
            || self.open.is_none()
        {
            return Err(LedgerError::StalePlan);
        }
        let step = self.open.take().expect("checked above");
        for item in &plan.committed {
            self.slots[item.slot as usize] = Some(Holder {
                request: item.request,
                tokens: item.start + u64::from(item.kept),
            });
        }
        self.anchors = step.anchors.into_iter().map(Some).collect();
        Ok(plan.committed)
    }

    /// Ends the open step keeping nothing: frees the slots it claimed, or
    /// every slot it touched when the state invalidates on cancel. Returns
    /// the step, whose touched entries the bundle puts back.
    pub fn cancel(&mut self) -> Result<Step, LedgerError> {
        let step = self.open.take().ok_or(LedgerError::NotOpen)?;
        for placed in &step.items {
            if placed.new || self.rules.invalidate_on_cancel {
                self.slots[placed.slot as usize] = None;
            }
        }
        Ok(step)
    }

    /// The slot a restore of `request` at `tokens` fills: the host's `slot`,
    /// or the first free one. Nothing changes.
    pub fn check_restore(
        &self,
        request: u64,
        tokens: u64,
        slot: Option<u32>,
    ) -> Result<u32, LedgerError> {
        if self.open.is_some() {
            return Err(LedgerError::Open);
        }
        if tokens == 0 || tokens > self.context {
            return Err(LedgerError::Tokens);
        }
        if self.slot_of(request).is_some() {
            return Err(LedgerError::ForeignSlot);
        }
        match (self.placement, slot) {
            (Placement::Host, Some(slot)) => match self.slots.get(slot as usize) {
                Some(None) => Ok(slot),
                _ => Err(LedgerError::ForeignSlot),
            },
            (Placement::Ledger, None) => self
                .slots
                .iter()
                .position(Option::is_none)
                .map(|slot| slot as u32)
                .ok_or(LedgerError::NoSlot),
            _ => Err(LedgerError::InvalidStep),
        }
    }

    /// Makes a slot `request`'s state after `tokens` committed tokens, for
    /// a prefix restore that fills it; returns the slot.
    pub fn restore(
        &mut self,
        request: u64,
        tokens: u64,
        slot: Option<u32>,
    ) -> Result<u32, LedgerError> {
        let slot = self.check_restore(request, tokens, slot)?;
        self.slots[slot as usize] = Some(Holder { request, tokens });
        Ok(slot)
    }

    /// Forgets a finished request and its anchors; returns the slot it
    /// freed, if it held one. The open step must not hold it.
    pub fn retire(&mut self, request: u64) -> Result<Option<u32>, LedgerError> {
        if self
            .open
            .as_ref()
            .is_some_and(|step| step.items.iter().any(|placed| placed.request == request))
        {
            return Err(LedgerError::Open);
        }
        for anchor in &mut self.anchors {
            if anchor.is_some_and(|(owner, _)| owner == request) {
                *anchor = None;
            }
        }
        let slot = self.slot_of(request);
        if let Some(slot) = slot {
            self.slots[slot as usize] = None;
        }
        Ok(slot)
    }

    /// The slot whose committed state is `request`'s after exactly `tokens`
    /// tokens: what a capture copies. No step may be open.
    pub fn committed(&self, request: u64, tokens: u64) -> Result<u32, LedgerError> {
        if self.open.is_some() {
            return Err(LedgerError::Open);
        }
        let slot = self.slot_of(request).ok_or(LedgerError::UnknownRequest)?;
        match self.holder(slot) {
            Some((_, held)) if held == tokens => Ok(slot),
            _ => Err(LedgerError::Tokens),
        }
    }

    /// The anchor item whose spares hold `request`'s state at `tokens`,
    /// kept by the last committed step. No step may be open.
    pub fn anchor(&self, request: u64, tokens: u64) -> Result<u32, LedgerError> {
        if self.open.is_some() {
            return Err(LedgerError::Open);
        }
        self.anchors
            .iter()
            .position(|anchor| *anchor == Some((request, tokens)))
            .map(|anchor| anchor as u32)
            .ok_or(LedgerError::UnknownRequest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    struct Observable {
        slots: Vec<Option<Holder>>,
        open: Option<Step>,
        anchors: Vec<Option<(u64, u64)>>,
        placement: Placement,
        rules: Rules,
        context: u64,
    }

    fn observable(ledger: &Ledger) -> Observable {
        Observable {
            slots: ledger.slots.clone(),
            open: ledger.open.clone(),
            anchors: ledger.anchors.clone(),
            placement: ledger.placement,
            rules: ledger.rules,
            context: ledger.context,
        }
    }

    #[test]
    fn plans_refuse_foreign_cloned_and_expired_ledgers_without_mutation() {
        let mut ledger = Ledger::new(3, Placement::Ledger, rules(false), 64);
        ledger.begin(&[item(1, 0, 4, true)], vec![(1, 2)]).unwrap();
        let mut other = ledger.clone();
        let before = observable(&other);
        assert_eq!(
            other.apply_commit(ledger.plan_commit(&[(1, 2)]).unwrap()),
            Err(LedgerError::StalePlan)
        );
        assert_eq!(observable(&other), before);
        let stale = ledger.plan_commit(&[(1, 2)]).unwrap();
        ledger.cancel().unwrap();
        ledger.begin(&[item(1, 0, 4, true)], vec![(1, 2)]).unwrap();
        let before = observable(&ledger);
        assert_eq!(ledger.apply_commit(stale), Err(LedgerError::StalePlan));
        assert_eq!(observable(&ledger), before);
        let duplicate = ledger.plan_commit(&[(1, 2)]).unwrap();
        ledger.commit(&[(1, 2)]).unwrap();
        let before = observable(&ledger);
        assert_eq!(ledger.apply_commit(duplicate), Err(LedgerError::StalePlan));
        assert_eq!(observable(&ledger), before);
    }

    #[test]
    fn planned_rows_match_independent_ordered_expectations() {
        for placement in [Placement::Ledger, Placement::Host] {
            for invalidate in [false, true] {
                for accepted in 1..=4 {
                    let mut ledger = Ledger::new(3, placement, rules(invalidate), 64);
                    let mut items = [
                        item(8, 0, 3, false),
                        item(7, 0, 4, true),
                        item(9, 0, 1, false),
                    ];
                    if placement == Placement::Host {
                        for (slot, item) in items.iter_mut().enumerate() {
                            item.slot = Some(slot as u32);
                        }
                    }
                    ledger.begin(&items, vec![(7, 2)]).unwrap();
                    let before = observable(&ledger);
                    for refused in [&[][..], &[(8, 1)][..], &[(7, 0)][..], &[(7, 5)][..]] {
                        assert_eq!(ledger.plan_commit(refused).unwrap_err(), LedgerError::Kept);
                        assert_eq!(observable(&ledger), before);
                    }
                    let plan = ledger.plan_commit(&[(7, accepted)]).unwrap();
                    let expected = vec![
                        Committed {
                            request: 8,
                            slot: 0,
                            start: 0,
                            rows: 3,
                            kept: 3,
                        },
                        Committed {
                            request: 7,
                            slot: 1,
                            start: 0,
                            rows: 4,
                            kept: accepted,
                        },
                        Committed {
                            request: 9,
                            slot: 2,
                            start: 0,
                            rows: 1,
                            kept: 1,
                        },
                    ];
                    assert_eq!(plan.items(), expected);
                    assert_eq!(observable(&ledger), before);
                    let mut wrapper = ledger.clone();
                    assert_eq!(ledger.apply_commit(plan).unwrap(), expected);
                    assert_eq!(wrapper.commit(&[(7, accepted)]).unwrap(), expected);
                    assert_eq!(observable(&ledger), observable(&wrapper));
                }
            }
        }
    }

    #[test]
    fn committed_plans_stay_stale_after_begin_and_generation_wrap() {
        let mut ledger = Ledger::new(1, Placement::Ledger, rules(false), 64);
        ledger.begin(&[item(1, 0, 4, true)], Vec::new()).unwrap();
        let stale = ledger.plan_commit(&[(1, 2)]).unwrap();
        ledger.commit(&[(1, 2)]).unwrap();
        ledger.begin(&[item(1, 2, 4, true)], Vec::new()).unwrap();
        let before = observable(&ledger);
        assert_eq!(ledger.apply_commit(stale), Err(LedgerError::StalePlan));
        assert_eq!(observable(&ledger), before);
        ledger.generation = 0;
        let stale = ledger.plan_commit(&[(1, 2)]).unwrap();
        ledger.generation = u64::MAX;
        ledger.advance_generation();
        let before = observable(&ledger);
        assert_eq!(ledger.apply_commit(stale), Err(LedgerError::StalePlan));
        assert_eq!(observable(&ledger), before);
        assert_eq!(ledger.commit(&[(1, 2)]).unwrap()[0].kept, 2);
    }

    #[test]
    fn retained_plan_cannot_bind_to_recreated_ledger() {
        let mut ledger = Ledger::new(1, Placement::Ledger, rules(false), 64);
        ledger.begin(&[item(1, 0, 4, true)], Vec::new()).unwrap();
        let plan = ledger.plan_commit(&[(1, 2)]).unwrap();
        drop(ledger);
        let mut ledger = Ledger::new(1, Placement::Ledger, rules(false), 64);
        ledger.begin(&[item(1, 0, 4, true)], Vec::new()).unwrap();
        let before = observable(&ledger);
        assert_eq!(ledger.apply_commit(plan), Err(LedgerError::StalePlan));
        assert_eq!(observable(&ledger), before);
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn changing_tail_window_expires_previously_valid_plans() {
        let mut ledger = Ledger::new(1, Placement::Ledger, rules(false), 64);
        ledger.begin(&[item(1, 0, 4, true)], Vec::new()).unwrap();
        let stale = ledger.plan_commit(&[(1, 2)]).unwrap();
        ledger.set_test_tail_window(Some(2));
        let before = observable(&ledger);
        assert_eq!(ledger.apply_commit(stale), Err(LedgerError::StalePlan));
        assert_eq!(observable(&ledger), before);
        assert_eq!(ledger.commit(&[(1, 2)]), Err(LedgerError::Kept));
        assert_eq!(observable(&ledger), before);
    }

    fn item(request: u64, start: u64, rows: u32, verify: bool) -> Item {
        Item {
            request,
            slot: None,
            start,
            rows,
            verify,
        }
    }

    fn rules(invalidate_on_cancel: bool) -> Rules {
        Rules {
            invalidate_on_cancel,
            tail_window: None,
        }
    }

    #[test]
    fn rules_follow_the_declaration() {
        let mut state = crate::state::synthetic();
        // The slot buffer invalidates on cancel; the eight-position window
        // restores rejected tails.
        assert_eq!(
            Rules::of(&state),
            Rules {
                invalidate_on_cancel: true,
                tail_window: Some(8),
            }
        );
        state.buffers[0].rollback = Rollback::Truncate;
        state.buffers[1].cancel = Cancel::None;
        assert_eq!(Rules::of(&state), rules(false));
    }

    /// An eight-position window: a verify item of nine rows overwrote its
    /// accepted first position with its ninth, so only a commit of all nine
    /// rows keeps the window right.
    #[test]
    fn a_partial_commit_longer_than_the_tail_window_is_refused() {
        let window = Rules {
            invalidate_on_cancel: false,
            tail_window: Some(8),
        };
        let mut ledger = Ledger::new(2, Placement::Ledger, window, 1 << 20);
        ledger
            .begin(&[item(1, 0, 9, true), item(2, 0, 8, true)], Vec::new())
            .unwrap();
        for kept in [[(1, 1), (2, 8)], [(1, 8), (2, 1)]] {
            assert_eq!(ledger.commit(&kept), Err(LedgerError::Kept));
        }
        // Nothing changed: the step is open and the slots claimed.
        assert_eq!(ledger.step().map(|step| step.items.len()), Some(2));
        assert_eq!(
            (ledger.holder(0), ledger.holder(1)),
            (Some((1, 0)), Some((2, 0)))
        );
        // An item within the window may keep a prefix.
        let committed = ledger.commit(&[(1, 9), (2, 1)]).unwrap();
        assert_eq!((committed[0].kept, committed[1].kept), (9, 1));
        assert_eq!(
            (ledger.holder(0), ledger.holder(1)),
            (Some((1, 9)), Some((2, 1)))
        );
    }

    fn hosted(slot: u32, request: u64, start: u64, rows: u32) -> Item {
        Item {
            slot: Some(slot),
            ..item(request, start, rows, false)
        }
    }

    #[test]
    fn commit_keeps_each_verify_items_accepted_rows() {
        let mut ledger = Ledger::new(4, Placement::Ledger, rules(false), 1 << 20);
        let prefill = [item(10, 0, 7, false), item(11, 0, 3, false)];
        assert_eq!(ledger.preview(&prefill), Ok(vec![0, 1]));
        assert_eq!(ledger.holder(0), None);
        ledger.begin(&prefill, Vec::new()).unwrap();
        assert_eq!(ledger.holder(0), Some((10, 0)));
        assert_eq!(ledger.commit(&[]).unwrap()[1].kept, 3);
        assert_eq!(
            (ledger.holder(0), ledger.holder(1)),
            (Some((10, 7)), Some((11, 3)))
        );
        // A verify step: a decode item keeps all of its row, the two verify
        // items what commit names, in item order.
        let step = [
            item(11, 3, 4, true),
            item(12, 0, 1, false),
            item(10, 7, 4, true),
        ];
        ledger.begin(&step, Vec::new()).unwrap();
        let open = ledger.step().unwrap();
        assert_eq!(
            (
                open.verify_index(0),
                open.verify_index(1),
                open.verify_index(2)
            ),
            (Some(0), None, Some(1))
        );
        // Out of order, missing, extra or out-of-range rows change nothing.
        for kept in [
            &[(10, 2), (11, 1)][..],
            &[(11, 1)],
            &[(11, 1), (10, 2), (12, 1)],
            &[(11, 0), (10, 2)],
            &[(11, 5), (10, 2)],
        ] {
            assert_eq!(ledger.commit(kept), Err(LedgerError::Kept));
        }
        assert_eq!(ledger.commit(&[]), Err(LedgerError::Kept));
        let committed = ledger.commit(&[(11, 1), (10, 4)]).unwrap();
        assert_eq!(
            committed,
            [
                Committed {
                    request: 11,
                    slot: 1,
                    start: 3,
                    rows: 4,
                    kept: 1
                },
                Committed {
                    request: 12,
                    slot: 2,
                    start: 0,
                    rows: 1,
                    kept: 1
                },
                Committed {
                    request: 10,
                    slot: 0,
                    start: 7,
                    rows: 4,
                    kept: 4
                },
            ]
        );
        assert_eq!(ledger.committed(11, 4), Ok(1));
        assert_eq!(ledger.committed(11, 7), Err(LedgerError::Tokens));
        assert_eq!(ledger.commit(&[]), Err(LedgerError::NotOpen));
    }

    #[test]
    fn cancel_frees_claimed_slots_or_invalidates_every_touched_one() {
        for invalidate in [false, true] {
            let mut ledger = Ledger::new(3, Placement::Ledger, rules(invalidate), 1 << 20);
            ledger.begin(&[item(1, 0, 4, false)], Vec::new()).unwrap();
            ledger.commit(&[]).unwrap();
            ledger
                .begin(&[item(1, 4, 2, true), item(2, 0, 5, false)], Vec::new())
                .unwrap();
            assert_eq!(ledger.committed(1, 4), Err(LedgerError::Open));
            let step = ledger.cancel().unwrap();
            assert_eq!(step.items.len(), 2);
            assert_eq!(ledger.slot_of(2), None);
            // The continuing request keeps its committed state unless the
            // state invalidates on cancel.
            let kept = (!invalidate).then_some((1, 4));
            assert_eq!(ledger.holder(0), kept);
            assert_eq!(ledger.cancel(), Err(LedgerError::NotOpen));
            // An aborted step leaves the next one free to start over.
            let next = [item(2, 0, 1, false)];
            assert_eq!(ledger.preview(&next), Ok(vec![u32::from(!invalidate)]));
        }
    }

    #[test]
    fn steps_must_continue_their_requests_committed_state() {
        let mut ledger = Ledger::new(2, Placement::Ledger, rules(false), 64);
        ledger.begin(&[item(1, 0, 8, false)], Vec::new()).unwrap();
        assert_eq!(
            ledger.begin(&[item(2, 0, 1, false)], Vec::new()).err(),
            Some(LedgerError::Open)
        );
        ledger.commit(&[]).unwrap();
        let refused = |ledger: &Ledger, items: &[Item]| ledger.preview(items).unwrap_err();
        assert_eq!(
            refused(&ledger, &[item(1, 7, 1, false)]),
            LedgerError::Tokens
        );
        assert_eq!(
            refused(&ledger, &[item(2, 3, 1, false)]),
            LedgerError::UnknownRequest
        );
        assert_eq!(
            refused(&ledger, &[item(1, 8, 57, false)]),
            LedgerError::InvalidStep
        );
        assert_eq!(
            refused(&ledger, &[item(1, 8, 0, false)]),
            LedgerError::InvalidStep
        );
        assert_eq!(
            refused(&ledger, &[item(2, 0, 1, false), item(2, 0, 1, false)]),
            LedgerError::InvalidStep
        );
        assert_eq!(
            refused(&ledger, &[item(2, 0, 1, false), item(3, 0, 1, false)]),
            LedgerError::NoSlot
        );
        assert_eq!(refused(&ledger, &[]), LedgerError::InvalidStep);
        assert_eq!(
            refused(&ledger, &[hosted(1, 2, 0, 1)]),
            LedgerError::InvalidStep
        );
    }

    #[test]
    fn the_host_names_new_requests_slots() {
        let mut ledger = Ledger::new(4, Placement::Host, rules(true), 1 << 20);
        ledger
            .begin(&[hosted(2, 7, 0, 5), hosted(0, 8, 0, 3)], Vec::new())
            .unwrap();
        ledger.commit(&[]).unwrap();
        assert_eq!((ledger.slot_of(7), ledger.slot_of(8)), (Some(2), Some(0)));
        let refused = |ledger: &Ledger, items: &[Item]| ledger.preview(items).unwrap_err();
        // Another request's slot, another slot for a held request, a slot
        // past the ledger, a new request without a slot.
        assert_eq!(
            refused(&ledger, &[hosted(2, 9, 0, 1)]),
            LedgerError::ForeignSlot
        );
        assert_eq!(
            refused(&ledger, &[hosted(1, 7, 5, 1)]),
            LedgerError::ForeignSlot
        );
        assert_eq!(
            refused(&ledger, &[hosted(4, 9, 0, 1)]),
            LedgerError::InvalidStep
        );
        assert_eq!(
            refused(&ledger, &[item(9, 0, 1, false)]),
            LedgerError::InvalidStep
        );
        assert_eq!(
            refused(&ledger, &[hosted(1, 9, 0, 1), hosted(1, 10, 0, 1)]),
            LedgerError::ForeignSlot
        );
        assert_eq!(ledger.preview(&[hosted(2, 7, 5, 2)]), Ok(vec![2]));
        // Restores fill the named free slot.
        assert_eq!(
            ledger.check_restore(9, 4, Some(2)),
            Err(LedgerError::ForeignSlot)
        );
        assert_eq!(
            ledger.check_restore(7, 4, Some(1)),
            Err(LedgerError::ForeignSlot)
        );
        assert_eq!(
            ledger.check_restore(9, 4, None),
            Err(LedgerError::InvalidStep)
        );
        assert_eq!(ledger.restore(9, 4, Some(1)), Ok(1));
        assert_eq!(ledger.holder(1), Some((9, 4)));
    }

    #[test]
    fn restore_retire_and_anchors_follow_the_open_step() {
        let mut ledger = Ledger::new(2, Placement::Ledger, rules(false), 100);
        assert_eq!(ledger.restore(5, 0, None), Err(LedgerError::Tokens));
        assert_eq!(ledger.restore(5, 101, None), Err(LedgerError::Tokens));
        assert_eq!(ledger.restore(5, 64, None), Ok(0));
        assert_eq!(ledger.restore(5, 64, None), Err(LedgerError::ForeignSlot));
        assert_eq!(ledger.preview(&[item(5, 64, 2, false)]), Ok(vec![0]));
        ledger
            .begin(&[item(5, 64, 2, false), item(6, 0, 9, false)], vec![(6, 9)])
            .unwrap();
        assert_eq!(ledger.anchor(6, 9), Err(LedgerError::Open));
        assert_eq!(ledger.retire(6), Err(LedgerError::Open));
        assert_eq!(ledger.restore(7, 3, None), Err(LedgerError::Open));
        // A request outside the step may retire.
        assert_eq!(ledger.retire(99), Ok(None));
        ledger.commit(&[]).unwrap();
        assert_eq!(ledger.anchor(6, 9), Ok(0));
        assert_eq!(ledger.anchor(6, 8), Err(LedgerError::UnknownRequest));
        // The next step rewrites the spares; a retire forgets them.
        ledger.begin(&[item(6, 9, 1, false)], Vec::new()).unwrap();
        ledger.cancel().unwrap();
        assert_eq!(ledger.anchor(6, 9), Err(LedgerError::UnknownRequest));
        ledger
            .begin(&[item(6, 9, 1, false)], vec![(5, 66)])
            .unwrap();
        ledger.commit(&[]).unwrap();
        assert_eq!(ledger.anchor(5, 66), Ok(0));
        assert_eq!(ledger.retire(5), Ok(Some(0)));
        assert_eq!(ledger.anchor(5, 66), Err(LedgerError::UnknownRequest));
        assert_eq!(ledger.committed(5, 66), Err(LedgerError::UnknownRequest));
        assert_eq!(ledger.committed(6, 10), Ok(1));
    }
}
