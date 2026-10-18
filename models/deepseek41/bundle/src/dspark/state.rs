//! Transactional bookkeeping, independent of device launches and weights.
//!
//! Before any continuing target step, including K=0, each stage catches up
//! from retained committed features. Draft KV lives outside the main ring.
//! Cancellation therefore leaves committed history and completed projections
//! valid. Only accepted target rows are copied from staging on commit.
//!
//! A slot retains `window + MAX_VERIFY_ROWS` raw rows. A larger accepted
//! prefill chunk projects every row directly from target staging during
//! commit, then retains its last window. This avoids losing unprojected rows
//! or allocating an entire prefill chunk per served slot.

use std::collections::{BTreeMap, BTreeSet};

pub const BLOCK_ROWS: u32 = model_deepseek41_contract::serving::MAX_DRAFT_TOKENS;
pub const MAX_VERIFY_ROWS: u32 = model_deepseek41_contract::serving::MAX_VERIFY_ROWS;
pub type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Config {
    pub slots: u32,
    pub stages: u32,
    pub window: u32,
    pub max_prefill_rows: u32,
    pub context_tokens: u64,
    /// Rows available in the RoPE tables, including any reserved block tail.
    pub position_capacity: u64,
}

impl Config {
    pub fn validate(self) -> Result<()> {
        if self.slots == 0
            || self.stages == 0
            || self.window == 0
            || self.max_prefill_rows == 0
            || self.context_tokens == 0
            || self.context_tokens > self.position_capacity
            || self.window.checked_add(MAX_VERIFY_ROWS).is_none()
        {
            return Err("invalid DSpark lifecycle geometry");
        }
        Ok(())
    }

    pub fn history_capacity(self) -> u64 {
        u64::from(self.window) + u64::from(MAX_VERIFY_ROWS)
    }

    /// Row of the raw history allocation; features within a row remain in
    /// target-layer order. Slot bounds are checked by step admission.
    pub fn history_cell(self, slot: u32, position: u64) -> u64 {
        u64::from(slot) * self.history_capacity() + position % self.history_capacity()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Prefill {
        rows: u32,
    },
    /// K=0 is an AR step; K>0 computes all five proposals and emits K.
    Decode {
        drafts: u32,
    },
    /// Verification tokens supplied by the host, including the bonus row.
    /// These rows participate in retention without invoking this drafter.
    Verify {
        rows: u32,
    },
}

impl Kind {
    pub fn target_rows(self) -> Result<u32> {
        match self {
            Self::Prefill { rows } if rows > 0 => Ok(rows),
            Self::Decode { drafts } if drafts <= BLOCK_ROWS => Ok(drafts + 1),
            Self::Verify { rows } if (2..=MAX_VERIFY_ROWS).contains(&rows) => Ok(rows),
            _ => Err("invalid DSpark target rows"),
        }
    }

    pub fn drafts(self) -> u32 {
        match self {
            Self::Prefill { .. } | Self::Verify { .. } => 0,
            Self::Decode { drafts } => drafts,
        }
    }

    pub fn is_verify(self) -> bool {
        matches!(self, Self::Verify { .. }) || self.drafts() > 0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rows {
    pub request: u64,
    pub slot: u32,
    pub start: u64,
    pub kind: Kind,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Context {
    pub slot: u32,
    pub retained_start: u64,
    pub end: u64,
    projected: Vec<u64>,
    lease: u64,
}

impl Context {
    pub fn projected(&self) -> &[u64] {
        &self.projected
    }
}

/// One bounded projection pass from retained raw rows. The lease prevents
/// a completion from an earlier slot owner or prefix restore being applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Projection {
    pub request: u64,
    pub slot: u32,
    pub stage: u32,
    pub start: u64,
    pub count: u32,
    lease: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftBlock {
    pub request: u64,
    pub slot: u32,
    /// Number of committed main rows, not the newest row's position.
    pub main_rows: u64,
    pub compute_rows: u32,
    pub emit_rows: u32,
    pub verify_rows: u32,
}

/// Raw features copied for a prefix snapshot. The snapshot contains no
/// derived stage KV; restore rebuilds each stage from this window.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub start: u64,
    pub end: u64,
    pub window: u32,
    pub stages: u32,
}

/// Accepted work after the whole batch's acceptance counts validate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitItem {
    pub rows: Rows,
    /// Retained target input rows, including the input/bonus row. Without
    /// early termination, zero accepted proposals keep one row and all five
    /// keep six. Stop tokens and token limits can omit that extra row, so
    /// this is not an exact accepted-proposal count.
    pub kept: u32,
    /// Project all `kept` rows from staging through every stage, in position
    /// order, before retaining their last window and releasing staging.
    pub project_from_staging: bool,
    pub retain_offset: u32,
}

impl CommitItem {
    pub fn retain_count(self) -> u32 {
        self.kept - self.retain_offset
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommitPlan {
    transaction: u64,
    items: Vec<CommitItem>,
}

impl CommitPlan {
    pub fn items(&self) -> &[CommitItem] {
        &self.items
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Prepared,
    Executing,
    Executed,
    Committing,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Pending {
    transaction: u64,
    rows: Vec<Rows>,
    phase: Phase,
    commit: Option<CommitPlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct State {
    config: Config,
    committed: BTreeMap<u64, Context>,
    pending: Option<Pending>,
    serial: u64,
    poisoned: bool,
}

impl State {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            committed: BTreeMap::new(),
            pending: None,
            serial: 0,
            poisoned: false,
        })
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn context(&self, request: u64) -> Option<&Context> {
        self.committed.get(&request)
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// A failed device operation may have partially changed history or a
    /// ring. The instance must poison instead of cancelling and retrying it.
    pub fn poison(&mut self) {
        self.poisoned = true;
    }

    fn usable(&self) -> Result<()> {
        if self.poisoned {
            return Err("DSpark state is poisoned");
        }
        Ok(())
    }

    fn idle(&self) -> Result<()> {
        self.usable()?;
        if self.pending.is_some() {
            return Err("DSpark transaction is open");
        }
        Ok(())
    }

    fn next_serial(&self) -> Result<u64> {
        self.serial
            .checked_add(1)
            .ok_or("DSpark transaction counter overflow")
    }

    pub fn prepare(&mut self, rows: Vec<Rows>) -> Result<()> {
        self.idle()?;
        if rows.is_empty() || rows.len() > self.config.slots as usize {
            return Err("invalid DSpark batch size");
        }
        let mut requests = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for row in &rows {
            let count = row.kind.target_rows()?;
            if !requests.insert(row.request)
                || !slots.insert(row.slot)
                || row.slot >= self.config.slots
                || matches!(row.kind, Kind::Prefill { rows } if rows > self.config.max_prefill_rows)
                || row
                    .start
                    .checked_add(u64::from(count))
                    .is_none_or(|end| end > self.config.context_tokens)
            {
                return Err("invalid DSpark batch rows");
            }
            if row.kind.drafts() > 0
                && row
                    .start
                    .checked_add(u64::from(BLOCK_ROWS))
                    .is_none_or(|end| end > self.config.position_capacity)
            {
                return Err("DSpark full block exceeds position capacity");
            }
            match self.committed.get(&row.request) {
                Some(context) if context.slot != row.slot || context.end != row.start => {
                    return Err("DSpark request is not contiguous in its slot");
                }
                Some(_) => {}
                None => {
                    if row.start != 0 || row.kind.is_verify() {
                        return Err("DSpark request needs prefill or prefix restore");
                    }
                    if self
                        .committed
                        .values()
                        .any(|context| context.slot == row.slot)
                    {
                        return Err("DSpark slot is still held");
                    }
                }
            }
        }
        let transaction = self.next_serial()?;
        self.pending = Some(Pending {
            transaction,
            rows,
            phase: Phase::Prepared,
            commit: None,
        });
        self.serial = transaction;
        Ok(())
    }

    fn can_project(&self) -> Result<()> {
        self.usable()?;
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.phase != Phase::Prepared)
        {
            return Err("DSpark catch-up must precede execution");
        }
        Ok(())
    }

    /// The next pass for one stage, including during an AR-only interval.
    /// Execute it on every rank before acknowledging it with `projected`.
    pub fn projection(
        &self,
        request: u64,
        stage: u32,
        max_rows: u32,
    ) -> Result<Option<Projection>> {
        self.can_project()?;
        let context = self
            .committed
            .get(&request)
            .ok_or("DSpark context missing")?;
        let start = *context
            .projected
            .get(stage as usize)
            .ok_or("DSpark stage out of range")?;
        if max_rows == 0 || start < context.retained_start || start > context.end {
            return Err("DSpark projection history is unavailable");
        }
        let count = (context.end - start).min(u64::from(max_rows)) as u32;
        Ok((count > 0).then_some(Projection {
            request,
            slot: context.slot,
            stage,
            start,
            count,
            lease: context.lease,
        }))
    }

    pub fn projected(&mut self, pass: Projection) -> Result<()> {
        self.can_project()?;
        let context = self
            .committed
            .get_mut(&pass.request)
            .ok_or("DSpark context missing")?;
        let cursor = context
            .projected
            .get_mut(pass.stage as usize)
            .ok_or("DSpark stage out of range")?;
        let end = pass
            .start
            .checked_add(u64::from(pass.count))
            .ok_or("DSpark projection overflow")?;
        if pass.lease != context.lease
            || pass.slot != context.slot
            || pass.start != *cursor
            || pass.start < context.retained_start
            || pass.count == 0
            || end > context.end
        {
            return Err("stale or noncontiguous DSpark projection");
        }
        *cursor = end;
        Ok(())
    }

    /// The returned blocks always compute five positions. The execution
    /// layer stages only the first K IDs for the K+1 target verify rows.
    pub fn begin_execution(&mut self) -> Result<Vec<DraftBlock>> {
        self.usable()?;
        let pending = self.pending.as_ref().ok_or("no DSpark transaction")?;
        if pending.phase != Phase::Prepared {
            return Err("DSpark execution already started");
        }
        for row in &pending.rows {
            if self.committed.get(&row.request).is_some_and(|context| {
                context
                    .projected
                    .iter()
                    .any(|&cursor| cursor != context.end)
            }) {
                return Err("DSpark stages need catch-up before execution");
            }
        }
        let blocks = pending
            .rows
            .iter()
            .filter(|row| row.kind.drafts() > 0)
            .map(|row| DraftBlock {
                request: row.request,
                slot: row.slot,
                main_rows: row.start,
                compute_rows: BLOCK_ROWS,
                emit_rows: row.kind.drafts(),
                verify_rows: row.kind.drafts() + 1,
            })
            .collect();
        self.pending.as_mut().expect("checked pending").phase = Phase::Executing;
        Ok(blocks)
    }

    /// Called after synchronous execute has finished target verification and
    /// staged every row's raw target features, including rejected rows.
    pub fn executed(&mut self) -> Result<()> {
        self.usable()?;
        let pending = self.pending.as_mut().ok_or("no DSpark transaction")?;
        if pending.phase != Phase::Executing {
            return Err("DSpark execute is not running");
        }
        pending.phase = Phase::Executed;
        Ok(())
    }

    /// Validate the entire acceptance vector before issuing any copies.
    /// `kept` lists verification items in batch order, with counts including the
    /// bonus row, as in the host's commit_accepted callback. AR and prefill
    /// items retain all their target rows and have no entry in `kept`.
    pub fn begin_commit(&mut self, kept: &[(u64, u32)]) -> Result<CommitPlan> {
        self.usable()?;
        let pending = self.pending.as_ref().ok_or("no DSpark transaction")?;
        if pending.phase != Phase::Executed {
            return Err("DSpark target rows have not completed");
        }
        let verify = pending.rows.iter().filter(|row| row.kind.is_verify());
        if verify.clone().count() != kept.len()
            || verify.zip(kept).any(|(row, &(request, count))| {
                request != row.request
                    || count == 0
                    || count > row.kind.target_rows().expect("validated target rows")
            })
        {
            return Err("invalid DSpark accepted target rows");
        }
        let mut accepted = kept.iter();
        let items = pending
            .rows
            .iter()
            .map(|&rows| {
                let count = if rows.kind.is_verify() {
                    accepted.next().expect("validated acceptance vector").1
                } else {
                    rows.kind.target_rows().expect("validated target rows")
                };
                let project_from_staging = u64::from(count) > self.config.history_capacity();
                CommitItem {
                    rows,
                    kept: count,
                    project_from_staging,
                    retain_offset: if project_from_staging {
                        count - self.config.window
                    } else {
                        0
                    },
                }
            })
            .collect();
        let plan = CommitPlan {
            transaction: pending.transaction,
            items,
        };
        let pending = self.pending.as_mut().expect("checked pending");
        pending.phase = Phase::Committing;
        pending.commit = Some(plan.clone());
        Ok(plan)
    }

    /// Publish only after all ranks finish every projection and retained-row
    /// copy in the plan. A device failure after begin_commit must poison the
    /// instance; cancellation cannot roll back those physical writes.
    pub fn complete_commit(&mut self, plan: CommitPlan) -> Result<()> {
        self.usable()?;
        let pending = self.pending.as_ref().ok_or("no DSpark transaction")?;
        if pending.phase != Phase::Committing || pending.commit.as_ref() != Some(&plan) {
            return Err("stale DSpark commit plan");
        }
        for item in plan.items {
            let row = item.rows;
            let context = self
                .committed
                .entry(row.request)
                .or_insert_with(|| Context {
                    slot: row.slot,
                    retained_start: 0,
                    end: 0,
                    projected: vec![0; self.config.stages as usize],
                    lease: plan.transaction,
                });
            context.end = row.start + u64::from(item.kept);
            if item.project_from_staging {
                context.projected.fill(context.end);
                context.retained_start = context.end - u64::from(self.config.window);
            } else {
                context.retained_start = context
                    .retained_start
                    .max(context.end.saturating_sub(self.config.history_capacity()));
            }
        }
        self.pending = None;
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<()> {
        self.usable()?;
        let pending = self.pending.as_ref().ok_or("no DSpark transaction")?;
        if pending.phase == Phase::Committing {
            return Err("cannot cancel a DSpark commit after copies begin");
        }
        self.pending = None;
        Ok(())
    }

    /// Catch up all stages before capturing, so no accepted rows disappear
    /// when the snapshot keeps only the raw window. Preemption copies this
    /// snapshot successfully before retiring the request and releasing its slot.
    pub fn snapshot(&self, request: u64) -> Result<Snapshot> {
        self.idle()?;
        let context = self
            .committed
            .get(&request)
            .ok_or("DSpark context missing")?;
        if context
            .projected
            .iter()
            .any(|&cursor| cursor != context.end)
        {
            return Err("DSpark stages need catch-up before snapshot");
        }
        Ok(Snapshot {
            start: context.end.saturating_sub(u64::from(self.config.window)),
            end: context.end,
            window: self.config.window,
            stages: self.config.stages,
        })
    }

    /// Validate before copying a snapshot into the slot. The instance holds
    /// exclusive lifecycle access until `restore` publishes the completed copy.
    pub fn validate_restore(&self, request: u64, slot: u32, snapshot: Snapshot) -> Result<()> {
        self.idle()?;
        if slot >= self.config.slots
            || snapshot.end == 0
            || snapshot.end > self.config.context_tokens
            || snapshot.window != self.config.window
            || snapshot.stages != self.config.stages
            || snapshot.start != snapshot.end.saturating_sub(u64::from(self.config.window))
            || self.committed.contains_key(&request)
            || self.committed.values().any(|context| context.slot == slot)
        {
            return Err("invalid DSpark prefix restore");
        }
        self.next_serial()?;
        Ok(())
    }

    /// Called after validation and after every rank restores exactly the
    /// snapshot's raw window. Derived rings and cursors from the prior slot
    /// owner are not reused. A copy failure poisons the instance instead.
    pub fn restore(&mut self, request: u64, slot: u32, snapshot: Snapshot) -> Result<()> {
        self.validate_restore(request, slot, snapshot)?;
        let lease = self.next_serial()?;
        self.committed.insert(
            request,
            Context {
                slot,
                retained_start: snapshot.start,
                end: snapshot.end,
                projected: vec![snapshot.start; self.config.stages as usize],
                lease,
            },
        );
        self.serial = lease;
        Ok(())
    }

    pub fn retire(&mut self, request: u64) -> Result<()> {
        self.idle()?;
        self.committed.remove(&request);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
