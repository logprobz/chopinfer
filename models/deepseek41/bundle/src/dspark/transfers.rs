//! Row ranges for lowering DSpark's host lifecycle to device operations.
//!
//! Offsets are in feature or KV rows, not bytes. The model's bound layouts
//! supply each row's stride and native KV format. Submit projection chunks
//! in order, and acknowledge the state pass only after every rank finishes
//! every chunk. A chunk never crosses a source wrap or a destination wrap,
//! so its ring writes are disjoint even on a parallel device launch.

use super::state::{CommitItem, Config, Kind, Projection, Result, RowProjection, Snapshot};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    Retained { row: u64 },
    Staged { row: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionChunk {
    pub request: u64,
    pub slot: u32,
    pub stage: u32,
    pub position: u64,
    pub rows: u32,
    pub source: Source,
    /// Row within this stage's `[slot][window]` KV allocation.
    pub ring_row: u64,
}

/// Retained rows of one request gathered into contiguous staging rows for
/// a batched launch. A segment never crosses a history wrap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GatherSegment {
    pub request: u64,
    pub slot: u32,
    pub position: u64,
    pub rows: u32,
    pub history_row: u64,
    /// Row of the launch's contiguous staging input.
    pub staged_row: u64,
}

/// One launch through every stage over gathered rows of several requests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchLaunch {
    pub segments: Vec<GatherSegment>,
    pub rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetainCopy {
    pub staged_row: u64,
    pub history_row: u64,
    pub rows: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotCopy {
    pub history_row: u64,
    pub snapshot_row: u64,
    pub rows: u32,
}

fn range(config: Config, slot: u32, start: u64, rows: u32) -> Result<()> {
    config.validate()?;
    if slot >= config.slots
        || rows == 0
        || start
            .checked_add(u64::from(rows))
            .is_none_or(|end| end > config.context_tokens)
    {
        return Err("invalid DSpark transfer range");
    }
    Ok(())
}

fn commit(config: Config, item: CommitItem) -> Result<()> {
    range(config, item.rows.slot, item.rows.start, item.kept)?;
    let direct = u64::from(item.kept) > config.history_capacity();
    if item.kept > item.rows.kind.target_rows()?
        || matches!(item.rows.kind, Kind::Prefill { rows } if rows > config.max_prefill_rows || rows != item.kept)
        || item.project_from_staging != direct
        || item.retain_offset != if direct { item.kept - config.window } else { 0 }
    {
        return Err("invalid DSpark feature commit");
    }
    Ok(())
}

/// Lower a state-issued retained-feature projection. `max_rows` is the
/// prepared projection's capacity; ring and history boundaries may split
/// it further. This does not acknowledge or advance the state cursor.
pub fn retained_projection(
    config: Config,
    pass: Projection,
    max_rows: u32,
) -> Result<Vec<ProjectionChunk>> {
    range(config, pass.slot, pass.start, pass.count)?;
    if pass.stage >= config.stages
        || max_rows == 0
        || u64::from(pass.count) > config.history_capacity()
    {
        return Err("invalid DSpark retained projection");
    }
    let mut chunks = Vec::new();
    let mut position = pass.start;
    let end = pass.start + u64::from(pass.count);
    let window = u64::from(config.window);
    let history = config.history_capacity();
    while position < end {
        let count = (end - position)
            .min(u64::from(max_rows))
            .min(window - position % window)
            .min(history - position % history);
        chunks.push(ProjectionChunk {
            request: pass.request,
            slot: pass.slot,
            stage: pass.stage,
            position,
            rows: count as u32,
            source: Source::Retained {
                row: config.history_cell(pass.slot, position),
            },
            ring_row: u64::from(pass.slot) * window + position % window,
        });
        position += count;
    }
    Ok(chunks)
}

/// Pack batched passes, in order, into launches of at most `max_rows` rows.
/// A pass holds at most one window of one slot, so its ring writes are
/// disjoint, and different slots own disjoint rings.
pub fn batched_projection(
    config: Config,
    passes: &[RowProjection],
    max_rows: u32,
) -> Result<Vec<BatchLaunch>> {
    if max_rows == 0 {
        return Err("invalid DSpark batched projection");
    }
    let history = config.history_capacity();
    let mut launches: Vec<BatchLaunch> = Vec::new();
    for (index, pass) in passes.iter().enumerate() {
        range(config, pass.slot, pass.start, pass.count)?;
        if pass.count > config.window
            || passes[..index]
                .iter()
                .any(|other| other.request == pass.request || other.slot == pass.slot)
        {
            return Err("invalid DSpark batched projection");
        }
        let mut position = pass.start;
        let end = pass.start + u64::from(pass.count);
        while position < end {
            if launches.last().is_none_or(|launch| launch.rows == max_rows) {
                launches.push(BatchLaunch {
                    segments: Vec::new(),
                    rows: 0,
                });
            }
            let launch = launches.last_mut().expect("open launch");
            let count = (end - position)
                .min(u64::from(max_rows - launch.rows))
                .min(history - position % history);
            launch.segments.push(GatherSegment {
                request: pass.request,
                slot: pass.slot,
                position,
                rows: count as u32,
                history_row: config.history_cell(pass.slot, position),
                staged_row: u64::from(launch.rows),
            });
            launch.rows += count as u32;
            position += count;
        }
    }
    Ok(launches)
}

/// Project every accepted row of a large prefill chunk before staging is
/// released. `staged_row` is the first row of this item in the target's
/// packed tap buffer. Each stage receives the same ordered raw rows.
pub fn staged_projection(
    config: Config,
    item: CommitItem,
    stage: u32,
    staged_row: u64,
    max_rows: u32,
) -> Result<Vec<ProjectionChunk>> {
    staged_projection_from(config, item, stage, staged_row, max_rows, 0)
}

/// The final retained prefill rows produce exactly the final projected ring.
/// Earlier projected rows are overwritten before any drafter reads them.
pub fn staged_retained_projection(
    config: Config,
    item: CommitItem,
    stage: u32,
    staged_row: u64,
    max_rows: u32,
) -> Result<Vec<ProjectionChunk>> {
    let first = item.rows.start + u64::from(item.retain_offset);
    // Keep whole original dispatches. Cutting one can choose a different
    // main-projection kernel at the 64-row boundary.
    Ok(
        staged_projection(config, item, stage, staged_row, max_rows)?
            .into_iter()
            .filter(|chunk| chunk.position + u64::from(chunk.rows) > first)
            .collect(),
    )
}

fn staged_projection_from(
    config: Config,
    item: CommitItem,
    stage: u32,
    staged_row: u64,
    max_rows: u32,
    first: u64,
) -> Result<Vec<ProjectionChunk>> {
    commit(config, item)?;
    if !item.project_from_staging || stage >= config.stages || max_rows == 0 {
        return Err("invalid DSpark staged projection");
    }
    staged_row
        .checked_add(u64::from(item.kept))
        .ok_or("DSpark staged row overflow")?;
    let window = u64::from(config.window);
    let mut offset = first;
    let mut chunks = Vec::new();
    while offset < u64::from(item.kept) {
        let position = item.rows.start + offset;
        let count = (u64::from(item.kept) - offset)
            .min(u64::from(max_rows))
            .min(window - position % window);
        chunks.push(ProjectionChunk {
            request: item.rows.request,
            slot: item.rows.slot,
            stage,
            position,
            rows: count as u32,
            source: Source::Staged {
                row: staged_row + offset,
            },
            ring_row: u64::from(item.rows.slot) * window + position % window,
        });
        offset += count;
    }
    Ok(chunks)
}

/// Retain only accepted raw features, splitting copies at the history wrap.
/// A direct projection completes before these copies release the staging
/// rows that did not fit the retained window.
pub fn retain_copies(config: Config, item: CommitItem, staged_row: u64) -> Result<Vec<RetainCopy>> {
    commit(config, item)?;
    staged_row
        .checked_add(u64::from(item.kept))
        .ok_or("DSpark staged row overflow")?;
    let mut offset = u64::from(item.retain_offset);
    let mut copies = Vec::new();
    while offset < u64::from(item.kept) {
        let position = item.rows.start + offset;
        let count = (u64::from(item.kept) - offset)
            .min(config.history_capacity() - position % config.history_capacity());
        copies.push(RetainCopy {
            staged_row: staged_row + offset,
            history_row: config.history_cell(item.rows.slot, position),
            rows: count as u32,
        });
        offset += count;
    }
    Ok(copies)
}

/// The same raw-row mapping for capture and restore. Snapshot rows are
/// compact and chronological. Slot ownership and cursor readiness are
/// validated by the lifecycle before any of these copies are issued.
pub fn snapshot_copies(config: Config, slot: u32, snapshot: Snapshot) -> Result<Vec<SnapshotCopy>> {
    config.validate()?;
    if slot >= config.slots
        || snapshot.end == 0
        || snapshot.end > config.context_tokens
        || snapshot.window != config.window
        || snapshot.stages != config.stages
        || snapshot.start != snapshot.end.saturating_sub(u64::from(config.window))
    {
        return Err("invalid DSpark snapshot transfers");
    }
    let mut position = snapshot.start;
    let mut copies = Vec::new();
    while position < snapshot.end {
        let count = (snapshot.end - position)
            .min(config.history_capacity() - position % config.history_capacity());
        copies.push(SnapshotCopy {
            history_row: config.history_cell(slot, position),
            snapshot_row: position - snapshot.start,
            rows: count as u32,
        });
        position += count;
    }
    Ok(copies)
}

#[cfg(test)]
mod tests;
