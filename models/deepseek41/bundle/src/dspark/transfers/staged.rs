//! Pack original staged dispatches without changing their row identities.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StagedSegment {
    pub source: ProjectionChunk,
    pub destination_row: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedBatch {
    pub stage: u32,
    pub segments: Vec<StagedSegment>,
    pub rows: u32,
}

/// Keep stages and each slot's positions ordered, splitting only at capacity.
/// Original chunks still name their source and ring-wrap boundaries so the
/// feature binder can validate every source before any copy is submitted.
pub fn staged_batches(
    config: Config,
    chunks: &[ProjectionChunk],
    capacity: u32,
) -> Result<Vec<StagedBatch>> {
    config.validate()?;
    if capacity == 0 || capacity > config.max_prefill_rows {
        return Err("invalid DSpark staged batch capacity");
    }
    let mut result: Vec<StagedBatch> = Vec::new();
    let mut stage = None;
    let mut slots = BTreeMap::new();
    let mut requests = BTreeMap::new();
    for &chunk in chunks {
        range(config, chunk.slot, chunk.position, chunk.rows)?;
        let Source::Staged { row } = chunk.source else {
            return Err("staged batch reads retained history");
        };
        if chunk.stage >= config.stages
            || stage.is_some_and(|prior| chunk.stage < prior)
            || chunk.ring_row
                != u64::from(chunk.slot) * u64::from(config.window)
                    + chunk.position % u64::from(config.window)
            || u64::from(chunk.rows)
                > u64::from(config.window) - chunk.position % u64::from(config.window)
            || row.checked_add(u64::from(chunk.rows)).is_none()
        {
            return Err("invalid DSpark staged batch source");
        }
        if stage != Some(chunk.stage) {
            slots.clear();
            requests.clear();
            stage = Some(chunk.stage);
        }
        if slots
            .get(&chunk.slot)
            .is_some_and(|&(request, end)| request != chunk.request || chunk.position < end)
            || requests
                .get(&chunk.request)
                .is_some_and(|&slot| slot != chunk.slot)
        {
            return Err("DSpark staged batch changes a slot or repeats positions");
        }
        slots.insert(
            chunk.slot,
            (chunk.request, chunk.position + u64::from(chunk.rows)),
        );
        requests.insert(chunk.request, chunk.slot);
        let mut offset = 0;
        while offset < chunk.rows {
            if result
                .last()
                .is_none_or(|batch| batch.stage != chunk.stage || batch.rows == capacity)
            {
                result.push(StagedBatch {
                    stage: chunk.stage,
                    segments: Vec::new(),
                    rows: 0,
                });
            }
            let batch = result.last_mut().expect("open staged batch");
            let rows = (chunk.rows - offset).min(capacity - batch.rows);
            batch.segments.push(StagedSegment {
                source: ProjectionChunk {
                    position: chunk.position + u64::from(offset),
                    rows,
                    source: Source::Staged {
                        row: row + u64::from(offset),
                    },
                    ring_row: chunk.ring_row + u64::from(offset),
                    ..chunk
                },
                destination_row: batch.rows,
            });
            batch.rows += rows;
            offset += rows;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dspark::state::Rows;

    fn config() -> Config {
        Config {
            slots: 16,
            stages: 3,
            window: 128,
            max_prefill_rows: 8192,
            context_tokens: 1 << 20,
            position_capacity: 1 << 20,
        }
    }
    fn chunks() -> Vec<ProjectionChunk> {
        let c = config();
        let mut chunks = Vec::new();
        for stage in 0..3 {
            let mut offset = 0;
            for (request, slot, start, rows) in [
                (7, 15, 0, 192),
                (9, 3, 57, 193),
                (11, 8, 126, 255),
                (12, 1, 128, 256),
            ] {
                let item = CommitItem {
                    rows: Rows {
                        request,
                        slot,
                        start,
                        kind: Kind::Prefill { rows },
                    },
                    kept: rows,
                    project_from_staging: true,
                    retain_offset: rows - c.window,
                };
                chunks
                    .extend(staged_retained_projection(c, item, stage, offset, c.window).unwrap());
                offset += u64::from(rows) + 17;
            }
        }
        chunks
    }
    fn values(chunk: ProjectionChunk) -> impl Iterator<Item = (u32, u32, u64, u64, u64)> {
        let Source::Staged { row } = chunk.source else {
            panic!("staged input")
        };
        (0..u64::from(chunk.rows)).map(move |i| {
            (
                chunk.stage,
                chunk.slot,
                chunk.request,
                chunk.position + i,
                row + i,
            )
        })
    }

    #[test]
    fn staged_batches_preserve_rows_and_final_rings_across_capacity_boundaries() {
        let c = config();
        let chunks = chunks();
        let original: Vec<_> = chunks.iter().copied().flat_map(values).collect();
        let mut expected = BTreeMap::new();
        for &(stage, slot, request, position, source) in &original {
            expected.insert(
                (stage, slot, position % u64::from(c.window)),
                (request, position, source),
            );
        }
        for capacity in [31, 64, 65, 127, 128, 129, 512, 8192] {
            let batches = staged_batches(c, &chunks, capacity).unwrap();
            let actual: Vec<_> = batches
                .iter()
                .flat_map(|b| b.segments.iter().flat_map(|s| values(s.source)))
                .collect();
            assert_eq!(actual, original);
            let mut rings = BTreeMap::new();
            for batch in &batches {
                assert!((1..=capacity).contains(&batch.rows));
                assert_eq!(
                    batch.rows,
                    batch.segments.iter().map(|s| s.source.rows).sum::<u32>()
                );
                let mut last = BTreeMap::new();
                let mut destination = 0;
                for segment in &batch.segments {
                    assert_eq!(segment.destination_row, destination);
                    destination += segment.source.rows;
                    last.insert(
                        segment.source.slot,
                        segment.source.position + u64::from(segment.source.rows) - 1,
                    );
                }
                for segment in &batch.segments {
                    for (stage, slot, request, position, source) in values(segment.source) {
                        if position + u64::from(c.window) > last[&slot] {
                            rings.insert(
                                (stage, slot, position % u64::from(c.window)),
                                (request, position, source),
                            );
                        }
                    }
                }
            }
            assert_eq!(rings, expected, "capacity={capacity}");
            if capacity == 8192 {
                assert_eq!(batches.len(), 3);
            }
        }
    }

    #[test]
    fn staged_batches_refuse_aliases_wraps_and_reordered_stages() {
        let c = config();
        let chunks = chunks();
        for capacity in [0, 8193] {
            assert!(staged_batches(c, &chunks, capacity).is_err());
        }
        let first = chunks[0];
        for bad in [
            ProjectionChunk {
                source: Source::Retained { row: 0 },
                ..first
            },
            ProjectionChunk {
                source: Source::Staged { row: u64::MAX },
                ..first
            },
            ProjectionChunk { rows: 0, ..first },
            ProjectionChunk { rows: 129, ..first },
            ProjectionChunk { slot: 16, ..first },
            ProjectionChunk {
                ring_row: first.ring_row + 1,
                ..first
            },
            ProjectionChunk { stage: 3, ..first },
        ] {
            assert!(staged_batches(c, &[bad], 8192).is_err());
        }
        assert!(staged_batches(c, &[first, first], 8192).is_err());
        assert!(staged_batches(c, &[ProjectionChunk { stage: 1, ..first }, first], 8192).is_err());
        let mut other = first;
        other.request += 1;
        other.position += 128;
        assert!(staged_batches(c, &[first, other], 8192).is_err());
    }
}
