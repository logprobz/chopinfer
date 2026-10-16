//! Logical TP selection, with padding carried separately from source ownership.

use kernel_core::precision::MatrixGeometry;
use kernel_core::topology::TensorParallel;
use std::ops::Range;

use super::{OperandSlice, SourceMatrix, Topology};
use crate::{Error, Geometry, Projection, ProjectionKind, SliceAxis, SplitClass};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Selection {
    pub rows: Vec<Range<u64>>,
    pub k: Range<u64>,
    pub experts: Range<u32>,
    pub logical: MatrixGeometry,
    pub physical: MatrixGeometry,
}

pub(super) fn topology(t: Topology) -> Result<TensorParallel, Error> {
    let tp = TensorParallel::new(t.ranks as usize).map_err(|e| Error(e.to_string()))?;
    if t.rank >= t.ranks {
        return Err(Error("rank is outside topology".into()));
    }
    Ok(tp)
}

pub(crate) fn pad(n: u64, tile: u64) -> Result<u64, Error> {
    if n == 0 || tile == 0 {
        return Err(Error(
            "matrix dimension and padding tile must be positive".into(),
        ));
    }
    n.checked_add(tile - 1)
        .and_then(|v| (v / tile).checked_mul(tile))
        .ok_or_else(|| Error("matrix padding overflows".into()))
}

fn shard(tp: TensorParallel, total: u64, tile: u64, rank: u32) -> Result<(Range<u64>, u64), Error> {
    let total = usize::try_from(total)
        .map_err(|_| Error("split dimension exceeds address space".into()))?;
    let tile =
        usize::try_from(tile).map_err(|_| Error("split tile exceeds address space".into()))?;
    let split = tp
        .padded_split(total, tile)
        .map_err(|e| Error(e.to_string()))?;
    let owned = split
        .rank(rank as usize)
        .map_err(|e| Error(e.to_string()))?;
    if owned.live.is_empty() {
        return Err(Error("rank owns no live matrix elements".into()));
    }
    Ok((
        owned.live.start as u64..owned.live.end as u64,
        owned.padded_len() as u64,
    ))
}

fn heads(
    tp: TensorParallel,
    count: u64,
    width: u64,
    start: u64,
    rank: u32,
) -> Result<Range<u64>, Error> {
    let count =
        usize::try_from(count).map_err(|_| Error("head count exceeds address space".into()))?;
    let owned = tp
        .kv_heads(count)
        .map_err(|e| Error(e.to_string()))?
        .rank(rank as usize)
        .map_err(|e| Error(e.to_string()))?;
    let first = (owned.live.start as u64)
        .checked_mul(width)
        .and_then(|v| v.checked_add(start))
        .ok_or_else(|| Error("head selection overflows".into()))?;
    let last = (owned.live.end as u64)
        .checked_mul(width)
        .and_then(|v| v.checked_add(start))
        .ok_or_else(|| Error("head selection overflows".into()))?;
    Ok(first..last)
}

/// Select rank ownership before format-specific padding or record emission.
pub fn select(p: &Projection, g: &Geometry, t: Topology, operand: u32) -> Result<Selection, Error> {
    let tp = topology(t)?;
    let total_rows = p.shape[0].resolve(g)?;
    let total_k = p.shape[1].resolve(g)?;
    let expert_count = match &p.kind {
        ProjectionKind::ExpertBank { experts, .. } => u32::try_from(experts.resolve(g)?)
            .map_err(|_| Error("expert count exceeds u32".into()))?,
        _ => 1,
    };
    let mut rows = std::iter::once(0..total_rows).collect::<Vec<_>>();
    let mut k = 0..total_k;
    let mut experts = 0..expert_count;
    let mut physical = MatrixGeometry {
        rows: total_rows,
        k: total_k,
    };
    match &p.split {
        SplitClass::Replicate {} => {}
        SplitClass::Column { tile } => {
            let (owned, n) = shard(tp, total_rows, tile.resolve(g)?, t.rank)?;
            rows = vec![owned];
            physical.rows = n;
        }
        SplitClass::Row { tile } => {
            let (owned, n) = shard(tp, total_k, tile.resolve(g)?, t.rank)?;
            k = owned;
            physical.k = n;
        }
        SplitClass::ExpertSharded { axis, tile } => match axis {
            SliceAxis::Rows => {
                let (owned, n) = shard(tp, total_rows, tile.resolve(g)?, t.rank)?;
                rows = vec![owned];
                physical.rows = n;
            }
            SliceAxis::K => {
                let (owned, n) = shard(tp, total_k, tile.resolve(g)?, t.rank)?;
                k = owned;
                physical.k = n;
            }
        },
        SplitClass::ExpertOwned { tile } => {
            let split = tp
                .split(expert_count as usize)
                .map_err(|e| Error(e.to_string()))?;
            let owned = split
                .rank(t.rank as usize)
                .map_err(|e| Error(e.to_string()))?;
            if pad(owned.live_len() as u64, tile.resolve(g)?)? != owned.live_len() as u64 {
                return Err(Error(
                    "padding virtual experts requires a registered implementation".into(),
                ));
            }
            experts = owned.live.start as u32..owned.live.end as u32;
        }
        SplitClass::HeadGrouped {
            heads: count,
            head_dim,
            tile,
        } => {
            rows.clear();
            let width = head_dim.resolve(g)?;
            if p.parts.is_empty() {
                rows.push(heads(tp, count.resolve(g)?, width, 0, t.rank)?);
            } else {
                let mut offset = 0;
                for part in &p.parts {
                    if part.head_runs.is_empty() {
                        rows.push(heads(
                            tp,
                            part.heads.as_ref().unwrap_or(count).resolve(g)?,
                            width,
                            offset,
                            t.rank,
                        )?);
                    } else {
                        for run in &part.head_runs {
                            let start = run
                                .rows
                                .start
                                .resolve_offset(g)?
                                .checked_add(offset)
                                .ok_or_else(|| Error("head run offset overflows".into()))?;
                            rows.push(heads(
                                tp,
                                run.heads.resolve(g)?,
                                run.head_dim.resolve(g)?,
                                start,
                                t.rank,
                            )?);
                        }
                    }
                    offset = offset
                        .checked_add(part.rows.resolve(g)?)
                        .ok_or_else(|| Error("part offset overflows".into()))?;
                }
            }
            let live = rows.iter().try_fold(0u64, |n, r| {
                n.checked_add(r.end - r.start)
                    .ok_or_else(|| Error("head row count overflows".into()))
            })?;
            physical.rows = pad(live, tile.resolve(g)?)?;
        }
        SplitClass::SegmentedHeads {
            segments,
            order,
            tile,
        } => {
            rows = crate::segments::select(segments, order, g, t)?;
            physical.rows = pad(rows.iter().map(|r| r.end - r.start).sum(), tile.resolve(g)?)?;
        }
        SplitClass::VocabSharded { tile } => {
            if let Some(table) = &p.lookup {
                let owned = tp
                    .padded_split(table.heads.len(), 1)
                    .map_err(|e| Error(e.to_string()))?
                    .rank(t.rank as usize)
                    .map_err(|e| Error(e.to_string()))?;
                let mut selected = Vec::new();
                for head in &table.heads[owned.live.clone()] {
                    let start = head.vocabulary.start.resolve_offset(g)?;
                    selected.push(
                        start
                            ..start
                                .checked_add(head.vocabulary.length.resolve(g)?)
                                .ok_or_else(|| Error("vocabulary range overflows".into()))?,
                    );
                }
                let padding_rank = t.ranks - 1;
                let last = table
                    .heads
                    .last()
                    .ok_or_else(|| Error("lookup has no heads".into()))?;
                let padding_start = last
                    .vocabulary
                    .start
                    .resolve_offset(g)?
                    .checked_add(last.vocabulary.length.resolve(g)?)
                    .ok_or_else(|| Error("vocabulary padding overflows".into()))?;
                if t.rank == padding_rank && padding_start < total_rows {
                    selected.push(padding_start..total_rows);
                }
                if selected.is_empty() {
                    return Err(Error("rank owns no lookup heads or padding".into()));
                }
                rows = selected;
                physical.rows = pad(rows.iter().map(|r| r.end - r.start).sum(), tile.resolve(g)?)?;
            } else {
                let (owned, n) = shard(tp, total_rows, tile.resolve(g)?, t.rank)?;
                rows = vec![owned];
                physical.rows = n;
            }
        }
    }
    let logical = MatrixGeometry {
        rows: rows.iter().map(|r| r.end - r.start).sum(),
        k: k.end - k.start,
    };
    for split in &p.operand_splits {
        if p.operand_index(&split.operand) == Some(operand.into()) {
            match split.axis {
                SliceAxis::Rows => physical.rows = pad(logical.rows, split.tile.resolve(g)?)?,
                SliceAxis::K => physical.k = pad(logical.k, split.tile.resolve(g)?)?,
            }
        }
    }
    Ok(Selection {
        rows,
        k,
        experts,
        logical,
        physical,
    })
}

pub(super) fn slices(
    sources: &[SourceMatrix],
    selection: &Selection,
    format: kernel_core::precision::format::FormatId,
    operand: u32,
) -> Result<Vec<OperandSlice>, Error> {
    let mut result = Vec::new();
    for source in sources
        .iter()
        .filter(|s| s.operand == operand && selection.experts.contains(&s.expert))
    {
        let mut rows = Vec::new();
        let mut target_rows = Vec::new();
        let mut offset = 0;
        for owned in &selection.rows {
            let start = owned.start.max(source.target_rows.start);
            let end = owned.end.min(source.target_rows.end);
            if start < end {
                target_rows.push(offset + start - owned.start..offset + end - owned.start);
                rows.push(
                    source.source_rows.start + start - source.target_rows.start
                        ..source.source_rows.start + end - source.target_rows.start,
                );
            }
            offset += owned.end - owned.start;
        }
        let start = selection.k.start.max(source.target_k.start);
        let end = selection.k.end.min(source.target_k.end);
        if rows.is_empty() || start >= end {
            continue;
        }
        let k = source.source_k.start + start - source.target_k.start
            ..source.source_k.start + end - source.target_k.start;
        result.push(OperandSlice {
            source: source.clone(),
            format,
            rows,
            k,
            target_rows,
            target_k: start - selection.k.start..end - selection.k.start,
            expert: source.expert - selection.experts.start,
            operand,
            geometry: selection.physical,
        });
    }
    if result.is_empty() {
        return Err(Error(
            "rank selection has no imported source matrices".into(),
        ));
    }
    Ok(result)
}
