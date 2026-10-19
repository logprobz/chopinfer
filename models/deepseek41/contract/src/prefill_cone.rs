//! Backward row dependencies for a prefill ending in complete request state.
//!
//! Earlier chunks already retain exact windows. This plan keeps each item's
//! complete physical ring and raw drafter history at its current chunk end.
//! The final compressed-key source still consumes every input row. Optional
//! outward alignment adds producers before their consumers are planned.

use std::ops::Range;

use crate::geometry::Geometry;
use crate::serving::{MAX_VERIFY_ROWS, PREFILL_TOKENS};
use crate::state::RING_HEADROOM;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Rows(Vec<Range<u32>>);

impl Rows {
    pub fn new(ranges: impl IntoIterator<Item = Range<u32>>) -> Self {
        let mut ranges: Vec<_> = ranges.into_iter().filter(|r| r.start < r.end).collect();
        ranges.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u32>> = Vec::new();
        for range in ranges {
            match merged.last_mut() {
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }
        Self(merged)
    }

    pub fn ranges(&self) -> &[Range<u32>] {
        &self.0
    }

    pub fn count(&self) -> u32 {
        self.0.iter().map(|r| r.end - r.start).sum()
    }

    pub fn contains(&self, row: u32) -> bool {
        self.0.iter().any(|range| range.contains(&row))
    }

    pub fn union(&self, other: &Self) -> Self {
        Self::new(self.0.iter().chain(&other.0).cloned())
    }

    fn aligned(&self, alignment: u32, limit: u32) -> Self {
        Self::new(self.0.iter().map(|range| {
            range.start / alignment * alignment
                ..range
                    .end
                    .div_ceil(alignment)
                    .saturating_mul(alignment)
                    .min(limit)
        }))
    }

    fn window(&self, items: &[Item], radius: u32) -> Self {
        Self::new(items.iter().flat_map(|item| {
            self.0.iter().filter_map(move |range| {
                let first = range.start.max(item.rows.start);
                let end = range.end.min(item.rows.end);
                (first < end).then(|| first.saturating_sub(radius).max(item.rows.start)..end)
            })
        }))
    }

    /// Bound every dispatch without joining disjoint intervals. Callers use
    /// a multiple of their alignment as `capacity`.
    pub fn parts(&self, capacity: u32) -> Vec<Range<u32>> {
        assert!(capacity > 0);
        self.0
            .iter()
            .flat_map(|range| {
                (range.start..range.end)
                    .step_by(capacity as usize)
                    .map(|start| start..start.saturating_add(capacity).min(range.end))
            })
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Item {
    /// This item's rows in the original packed prefill.
    pub rows: Range<u32>,
    /// Absolute sequence position of its first packed row.
    pub prefix: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layer {
    pub id: u32,
    /// HC attention prefix, input norm, quantizer and window-KV branch.
    pub keys: Rows,
    /// Complete attention and MLP outputs, with their HC prefixes recomputed
    /// locally so transient inputs always begin at local row zero.
    pub outputs: Rows,
    /// Raw attention-input stream means required by the drafter.
    pub features: Rows,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub rows: u32,
    pub last_source: u32,
    pub layers: Vec<Layer>,
    /// Original packed rows, in logits emission order.
    pub logits: Vec<u32>,
    /// Rows packed for a prefill's raw-feature commit, at their original
    /// staging offsets. No skipped feature may be packed or projected.
    pub features: Rows,
}

/// Preserve the original drafter projection dispatch shapes at a prompt end
/// that splits a 128-row ring block. The overwritten prefix of that final
/// block must also exist; changing a dispatch's M can select another GEMM.
pub fn feature_offset(prefix: u32, rows: u32, window: u32) -> u32 {
    if rows <= window + MAX_VERIFY_ROWS {
        return 0;
    }
    let retained = prefix + rows - window;
    (retained / window * window).saturating_sub(prefix)
}

impl Plan {
    pub fn new(
        g: &Geometry,
        items: &[Item],
        logits: &[u32],
        dspark: bool,
        alignment: u32,
    ) -> Result<Self, &'static str> {
        let source = *g
            .shape
            .csa
            .kv_sources
            .last()
            .ok_or("prefill has no compressed source")?;
        if g.shape.attention.window == 0
            || alignment == 0
            || alignment > PREFILL_TOKENS
            || !alignment.is_power_of_two()
            || g.shape.engram.layers.iter().any(|&layer| layer > source)
        {
            return Err("unsupported prefill dependency geometry or alignment");
        }
        let mut rows = 0;
        for item in items {
            if item.rows.start != rows
                || item.rows.end <= rows
                || item.rows.end > PREFILL_TOKENS
                || item
                    .prefix
                    .checked_add(item.rows.end - rows)
                    .is_none_or(|end| end > g.shape.context)
            {
                return Err("prefill items must be contiguous bounded packed rows");
            }
            rows = item.rows.end;
        }
        if rows == 0
            || logits.iter().any(|&row| row >= rows)
            || !logits.windows(2).all(|pair| pair[0] < pair[1])
        {
            return Err("invalid prefill logits rows");
        }
        let window = g.shape.attention.window;
        let ring = window
            .checked_add(RING_HEADROOM as u32)
            .ok_or("prefill ring overflow")?;
        let captured =
            Rows::new(items.iter().map(|item| {
                item.rows.end.saturating_sub(ring).max(item.rows.start)..item.rows.end
            }));
        let features = if dspark {
            Rows::new(items.iter().map(|item| {
                let count = item.rows.end - item.rows.start;
                item.rows.start + feature_offset(item.prefix, count, window)..item.rows.end
            }))
        } else {
            Rows::default()
        };
        let mut needed = Rows::new(logits.iter().map(|&row| row..row + 1));
        let mut layers = Vec::new();
        for id in (source..g.layers()).rev() {
            let outputs = needed.aligned(alignment, rows);
            let keys = outputs
                .window(items, window - 1)
                .union(&captured)
                .aligned(alignment, rows);
            let features = if g.shape.dspark.targets.contains(&id) {
                features.clone()
            } else {
                Rows::default()
            };
            needed = keys.union(&features);
            layers.push(Layer {
                id,
                keys,
                outputs,
                features,
            });
        }
        layers.reverse();
        Ok(Self {
            rows,
            last_source: source,
            layers,
            logits: logits.to_vec(),
            features,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;
    use std::collections::BTreeMap;

    fn fake(items: &[Item], plan: &Plan, sparse: bool) -> (Vec<u64>, BTreeMap<(u32, u32), u64>) {
        let mut hidden: BTreeMap<_, _> = (0..plan.rows)
            .map(|p| (p, u64::from(p + 1) * 1234567))
            .collect();
        let mut snapshots = BTreeMap::new();
        for layer in &plan.layers {
            let current_keys: BTreeMap<_, _> = (0..plan.rows)
                .filter(|&p| !sparse || layer.keys.contains(p))
                .map(|p| (p, hidden[&p].wrapping_mul(33) ^ u64::from(layer.id)))
                .collect();
            for item in items {
                for p in item.rows.end.saturating_sub(136).max(item.rows.start)..item.rows.end {
                    snapshots.insert((layer.id, p), current_keys[&p]);
                }
            }
            for range in layer.features.ranges() {
                for p in range.clone() {
                    snapshots.insert((layer.id + 100, p), hidden[&p]);
                }
            }
            let mut next = BTreeMap::new();
            for item in items {
                for p in item
                    .rows
                    .clone()
                    .filter(|&p| !sparse || layer.outputs.contains(p))
                {
                    let first = p.saturating_sub(127).max(item.rows.start);
                    let mut sum = hidden[&p];
                    for k in first..=p {
                        sum = sum.wrapping_add(current_keys[&k]);
                    }
                    // Exact prior-chunk ring values are external inputs.
                    let absolute = item.prefix + p - item.rows.start;
                    for k in absolute.saturating_sub(127)..item.prefix {
                        sum = sum.wrapping_add(u64::from(k).wrapping_mul(97) ^ u64::from(layer.id));
                    }
                    next.insert(p, sum);
                }
            }
            hidden = next;
        }
        (plan.logits.iter().map(|p| hidden[p]).collect(), snapshots)
    }

    #[test]
    fn exact_and_aligned_plans_never_read_a_skipped_row() {
        for (items, logits) in [
            (
                vec![Item {
                    rows: 0..8192,
                    prefix: 0,
                }],
                vec![8191],
            ),
            (
                vec![Item {
                    rows: 0..2048,
                    prefix: 2048,
                }],
                vec![],
            ),
            (
                vec![
                    Item {
                        rows: 0..17,
                        prefix: 529,
                    },
                    Item {
                        rows: 17..4097,
                        prefix: 8192,
                    },
                ],
                vec![16, 4096],
            ),
        ] {
            for alignment in [1, 64] {
                let plan = Plan::new(&FLASH, &items, &logits, true, alignment).unwrap();
                assert_eq!(fake(&items, &plan, true), fake(&items, &plan, false));
                for layer in &plan.layers {
                    assert!(
                        layer
                            .keys
                            .parts(4096)
                            .iter()
                            .all(|r| r.end - r.start <= 4096)
                    );
                }
            }
        }
    }

    #[test]
    fn exact_single_chunk_counts_match_the_independent_oracle() {
        let plan = Plan::new(
            &FLASH,
            &[Item {
                rows: 0..8192,
                prefix: 0,
            }],
            &[8191],
            true,
            1,
        )
        .unwrap();
        assert_eq!(plan.last_source, 20);
        assert_eq!(
            plan.layers.iter().map(|l| l.outputs.count()).sum::<u32>(),
            24302
        );
        assert_eq!(
            plan.layers[0].outputs.ranges(),
            std::slice::from_ref(&(5770..8192))
        );
        assert_eq!(plan.features.ranges(), std::slice::from_ref(&(8064..8192)));
        assert_eq!(
            plan.layers.last().unwrap().outputs.ranges(),
            std::slice::from_ref(&(8191..8192))
        );
    }

    #[test]
    fn small_chunks_keep_every_raw_feature_and_all_prompt_scores_keep_every_output() {
        for count in [1, 64, 128, 134, 135] {
            let logits = (0..count).collect::<Vec<_>>();
            let plan = Plan::new(
                &FLASH,
                &[Item {
                    rows: 0..count,
                    prefix: 0,
                }],
                &logits,
                true,
                64,
            )
            .unwrap();
            assert_eq!(plan.features.count(), count - feature_offset(0, count, 128));
            assert!(plan.layers.iter().all(|l| l.outputs.count() == count));
        }
    }
}
