//! Bind raw-feature row plans to checked device spans.
//!
//! Each BF16 row concatenates the geometry's target attention inputs in
//! declaration order. Provisional rows and retained rows occupy disjoint
//! spans. All copies of a plan are bound before any copy is submitted.

use super::state::{Config, Result};
use super::transfers::{
    GatherSegment, ProjectionChunk, RetainCopy, SnapshotCopy, Source, StagedSegment,
};
use model_deepseek41_contract::geometry::Geometry;
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Copy {
    pub source: Span,
    pub destination: Span,
}

/// Rows of retained state and target staging use the same raw BF16 layout.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    config: Config,
    row_bytes: u64,
}

impl Layout {
    pub fn new(geometry: &Geometry, config: Config) -> Result<Self> {
        config.validate()?;
        if config.window != geometry.shape.attention.window
            || config.stages != geometry.dspark_stages()
            || config.context_tokens > geometry.context_tokens()
        {
            return Err("DSpark feature layout disagrees with its geometry");
        }
        let row_bytes = (geometry.shape.dspark.targets.len() as u64)
            .checked_mul(u64::from(geometry.shape.hidden))
            .and_then(|n| n.checked_mul(2))
            .filter(|&n| n != 0)
            .ok_or("invalid DSpark feature width")?;
        Ok(Self { config, row_bytes })
    }

    pub fn row_bytes(self) -> u64 {
        self.row_bytes
    }

    pub fn retained_bytes(self) -> Result<u64> {
        self.config
            .history_capacity()
            .checked_mul(u64::from(self.config.slots))
            .and_then(|n| n.checked_mul(self.row_bytes))
            .ok_or("DSpark retained feature size overflow")
    }

    /// Bind the declared retained-state view and the target's packed taps.
    /// A zero-length staging view is valid during prefix capture or restore.
    pub fn bind(self, retained: Span, staged: Span) -> Result<Features> {
        self.validate_span(retained)?;
        self.validate_span(staged)?;
        if retained.length != self.retained_bytes()? {
            return Err("DSpark retained feature span has the wrong slot stride");
        }
        disjoint(retained, staged)?;
        Ok(Features {
            layout: self,
            retained,
            staged,
        })
    }

    fn validate_span(self, span: Span) -> Result<()> {
        if !span.offset.is_multiple_of(2)
            || !span.length.is_multiple_of(self.row_bytes)
            || span.offset.checked_add(span.length).is_none()
        {
            return Err("invalid DSpark raw feature span");
        }
        Ok(())
    }

    fn rows(self, span: Span, row: u64, count: u32) -> Result<Span> {
        if count == 0 {
            return Err("empty DSpark feature transfer");
        }
        let start = row
            .checked_mul(self.row_bytes)
            .ok_or("DSpark feature offset overflow")?;
        let bytes = u64::from(count)
            .checked_mul(self.row_bytes)
            .ok_or("DSpark feature length overflow")?;
        if start.checked_add(bytes).is_none_or(|end| end > span.length) {
            return Err("DSpark feature transfer exceeds its bound span");
        }
        let offset = span
            .offset
            .checked_add(start)
            .ok_or("DSpark feature address overflow")?;
        Ok(Span::new(span.buffer, offset, bytes))
    }

    fn history_rows(self, row: u64, count: u32) -> Result<()> {
        if row % self.config.history_capacity() + u64::from(count) > self.config.history_capacity()
        {
            return Err("DSpark feature transfer crosses a history wrap");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Features {
    layout: Layout,
    retained: Span,
    staged: Span,
}

impl Features {
    /// The exact committed or staged rows of one ordered projection chunk.
    pub fn projection_source(self, chunk: ProjectionChunk) -> Result<Span> {
        let c = self.layout.config;
        let window = u64::from(c.window);
        if chunk.stage >= c.stages
            || chunk.slot >= c.slots
            || chunk.rows == 0
            || chunk
                .position
                .checked_add(u64::from(chunk.rows))
                .is_none_or(|end| end > c.context_tokens)
            || chunk.ring_row != u64::from(chunk.slot) * window + chunk.position % window
            || u64::from(chunk.rows) > window - chunk.position % window
        {
            return Err("invalid DSpark projection chunk binding");
        }
        match chunk.source {
            Source::Retained { row } => {
                if row != c.history_cell(chunk.slot, chunk.position) {
                    return Err("DSpark projection reads another slot or position");
                }
                self.layout.history_rows(row, chunk.rows)?;
                self.layout.rows(self.retained, row, chunk.rows)
            }
            Source::Staged { row } => self.layout.rows(self.staged, row, chunk.rows),
        }
    }

    pub fn retain(self, plan: &[RetainCopy]) -> Result<Vec<Copy>> {
        plan.iter()
            .map(|copy| {
                self.layout.history_rows(copy.history_row, copy.rows)?;
                Ok(Copy {
                    source: self.layout.rows(self.staged, copy.staged_row, copy.rows)?,
                    destination: self
                        .layout
                        .rows(self.retained, copy.history_row, copy.rows)?,
                })
            })
            .collect()
    }

    /// Gather a commit segment without overwriting target staging or the
    /// raw history that remains live until retention finishes.
    pub fn gather_staged(self, segment: StagedSegment, destination: Span) -> Result<Copy> {
        if !matches!(segment.source.source, Source::Staged { .. }) {
            return Err("DSpark commit gather reads retained history");
        }
        self.layout.validate_span(destination)?;
        disjoint(self.staged, destination)?;
        disjoint(self.retained, destination)?;
        Ok(Copy {
            source: self.projection_source(segment.source)?,
            destination: self.layout.rows(
                destination,
                u64::from(segment.destination_row),
                segment.source.rows,
            )?,
        })
    }

    /// Copy one segment's retained rows into the batched launch's staging
    /// input. Catch-up precedes target staging, so these rows are free.
    pub fn gather(self, segment: GatherSegment) -> Result<Copy> {
        let c = self.layout.config;
        if segment.slot >= c.slots
            || segment.history_row != c.history_cell(segment.slot, segment.position)
        {
            return Err("DSpark gather reads another slot or position");
        }
        self.layout
            .history_rows(segment.history_row, segment.rows)?;
        Ok(Copy {
            source: self
                .layout
                .rows(self.retained, segment.history_row, segment.rows)?,
            destination: self
                .layout
                .rows(self.staged, segment.staged_row, segment.rows)?,
        })
    }

    /// The contiguous staging input of a batched launch of `rows` rows.
    pub fn batch_source(self, rows: u32) -> Result<Span> {
        self.layout.rows(self.staged, 0, rows)
    }

    /// Compact chronological rows copied out after projection cursors catch up.
    pub fn capture(self, plan: &[SnapshotCopy], snapshot: Span) -> Result<Vec<Copy>> {
        self.snapshot(plan, snapshot, false)
    }

    /// Bind only after validating the new slot lease. All stage cursors reset
    /// after these copies complete, so old derived KV is never reused.
    pub fn restore(self, plan: &[SnapshotCopy], snapshot: Span) -> Result<Vec<Copy>> {
        self.snapshot(plan, snapshot, true)
    }

    fn snapshot(self, plan: &[SnapshotCopy], snapshot: Span, restore: bool) -> Result<Vec<Copy>> {
        self.layout.validate_span(snapshot)?;
        if snapshot.length / self.layout.row_bytes > u64::from(self.layout.config.window) {
            return Err("DSpark snapshot exceeds one raw feature window");
        }
        disjoint(self.retained, snapshot)?;
        plan.iter()
            .map(|copy| {
                self.layout.history_rows(copy.history_row, copy.rows)?;
                let history = self
                    .layout
                    .rows(self.retained, copy.history_row, copy.rows)?;
                let saved = self.layout.rows(snapshot, copy.snapshot_row, copy.rows)?;
                Ok(if restore {
                    Copy {
                        source: saved,
                        destination: history,
                    }
                } else {
                    Copy {
                        source: history,
                        destination: saved,
                    }
                })
            })
            .collect()
    }
}

fn disjoint(a: Span, b: Span) -> Result<()> {
    // Callers validate each span's checked end before this comparison.
    if a.buffer == b.buffer
        && a.length != 0
        && b.length != 0
        && a.offset < b.offset + b.length
        && b.offset < a.offset + a.length
    {
        return Err("DSpark raw feature roles overlap");
    }
    Ok(())
}

#[cfg(test)]
mod tests;
