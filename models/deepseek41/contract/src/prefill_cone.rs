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
    /// Absolute complete-state endpoint of each packed item. These are the
    /// actual scheduler cuts, which need not lie on the configured grid.
    pub endpoints: Vec<u32>,
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
        Self::with_anchors(g, items, &[], logits, dspark, alignment)
    }

    /// A plan that also keeps complete ring state at each anchored item's
    /// anchor, an absolute position strictly inside the item: the anchor's
    /// physical ring cells root the cone like those of the item's end.
    /// `anchors` is empty or has one entry per item. Raw drafter history at
    /// an anchor is not planned, so DSpark refuses anchors.
    pub fn with_anchors(
        g: &Geometry,
        items: &[Item],
        anchors: &[Option<u32>],
        logits: &[u32],
        dspark: bool,
        alignment: u32,
    ) -> Result<Self, &'static str> {
        if !anchors.is_empty() && anchors.len() != items.len() {
            return Err("prefill anchors must name every item");
        }
        if dspark && anchors.iter().any(Option::is_some) {
            return Err("DSpark keeps no raw feature history at an anchor");
        }
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
        let mut anchored = Vec::new();
        for (item, anchor) in items.iter().zip(anchors) {
            let Some(anchor) = *anchor else {
                continue;
            };
            let end = item.prefix + (item.rows.end - item.rows.start);
            if anchor <= item.prefix || anchor >= end {
                return Err("a prefill anchor must lie strictly inside its item");
            }
            let row = item.rows.start + (anchor - item.prefix);
            anchored.push(row.saturating_sub(ring).max(item.rows.start)..row);
        }
        let captured = Rows::new(
            items
                .iter()
                .map(|item| item.rows.end.saturating_sub(ring).max(item.rows.start)..item.rows.end)
                .chain(anchored),
        );
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
            endpoints: items
                .iter()
                .map(|item| item.prefix + (item.rows.end - item.rows.start))
                .collect(),
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
        fake_anchored(items, &[], plan, sparse)
    }

    /// `fake`, also snapshotting each anchored item's ring at its anchor.
    fn fake_anchored(
        items: &[Item],
        anchors: &[Option<u32>],
        plan: &Plan,
        sparse: bool,
    ) -> (Vec<u64>, BTreeMap<(u32, u32), u64>) {
        let mut hidden: BTreeMap<_, _> = (0..plan.rows)
            .map(|p| (p, u64::from(p + 1) * 1234567))
            .collect();
        let mut snapshots = BTreeMap::new();
        let ends: Vec<_> = items
            .iter()
            .enumerate()
            .flat_map(|(at, item)| {
                let anchor = anchors.get(at).copied().flatten();
                let anchor = anchor.map(|a| (item.rows.start + a - item.prefix, 1000));
                [(item.rows.end, 0)]
                    .into_iter()
                    .chain(anchor)
                    .map(move |(end, tag)| (item, end, tag))
            })
            .collect();
        for layer in &plan.layers {
            let current_keys: BTreeMap<_, _> = (0..plan.rows)
                .filter(|&p| !sparse || layer.keys.contains(p))
                .map(|p| (p, hidden[&p].wrapping_mul(33) ^ u64::from(layer.id)))
                .collect();
            for &(item, end, tag) in &ends {
                for p in end.saturating_sub(136).max(item.rows.start)..end {
                    snapshots.insert((layer.id + tag, p), current_keys[&p]);
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

    /// One step that keeps its rounded prompt end as an anchor plans the
    /// anchor's ring like a chunk end: no planned row reads a skipped one,
    /// and the anchor's and the end's snapshots and the logits match the
    /// full schedule. The anchored cone is the union of both closures.
    #[test]
    fn anchored_plans_keep_the_anchor_ring_and_never_read_a_skipped_row() {
        for (items, anchors, logits) in [
            (
                vec![Item {
                    rows: 0..789,
                    prefix: 0,
                }],
                vec![Some(768)],
                vec![788],
            ),
            (
                vec![Item {
                    rows: 0..4171,
                    prefix: 0,
                }],
                vec![Some(4160)],
                vec![4170],
            ),
            (
                vec![Item {
                    rows: 0..1808,
                    prefix: 8192,
                }],
                vec![Some(9984)],
                vec![1807],
            ),
            (
                vec![
                    Item {
                        rows: 0..70,
                        prefix: 0,
                    },
                    Item {
                        rows: 70..100,
                        prefix: 8256,
                    },
                    Item {
                        rows: 100..2148,
                        prefix: 2048,
                    },
                ],
                vec![Some(64), None, Some(4032)],
                vec![69, 99, 2147],
            ),
        ] {
            for alignment in [1, 64] {
                let plan = Plan::with_anchors(&FLASH, &items, &anchors, &logits, false, alignment)
                    .unwrap();
                assert_eq!(
                    fake_anchored(&items, &anchors, &plan, true),
                    fake_anchored(&items, &anchors, &plan, false)
                );
                let plain = Plan::new(&FLASH, &items, &logits, false, alignment).unwrap();
                for (anchored, plain) in plan.layers.iter().zip(&plain.layers) {
                    assert_eq!(anchored.keys.union(&plain.keys), anchored.keys);
                    assert_eq!(anchored.outputs.union(&plain.outputs), anchored.outputs);
                }
            }
        }
        let item = Item {
            rows: 0..789,
            prefix: 0,
        };
        let plan = |anchors: &[Option<u32>], dspark| {
            Plan::with_anchors(
                &FLASH,
                std::slice::from_ref(&item),
                anchors,
                &[788],
                dspark,
                64,
            )
        };
        // The anchored single step plans the same rows as the cut step at
        // 768 and its 21-row tail together (`prefill_cone.py --cuts 768`).
        let anchored = plan(&[Some(768)], false).unwrap();
        assert_eq!(
            anchored
                .layers
                .iter()
                .map(|l| l.outputs.count())
                .sum::<u32>(),
            13412
        );
        assert_eq!(plan(&[None], false).unwrap(), plan(&[], false).unwrap());
        for refused in [
            vec![Some(0)],
            vec![Some(789)],
            vec![Some(800)],
            vec![None, None],
        ] {
            assert!(plan(&refused, false).is_err(), "{refused:?}");
        }
        assert!(plan(&[Some(768)], true).is_err());
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

    /// Plans every chunk of the oracle's guaranteed schedule for `tokens`
    /// with extra `cuts`, compares each layer's rows with it and checks the
    /// sparse plan against full fake kernels. Returns the chunks checked.
    fn matches_the_python_oracle(tokens: u32, cuts: &[u32]) -> usize {
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../prefill_cone.py");
        let output = std::process::Command::new("python3")
            .arg("-B")
            .arg(script)
            .args(["--contract", "guaranteed", "--chunk", "8192", "--tokens"])
            .arg(tokens.to_string())
            .arg("--cuts")
            .args(cuts.iter().map(u32::to_string))
            .output()
            .expect("the Python dependency oracle starts");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let cases: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let mut checked = 0;
        for case in cases.as_array().unwrap() {
            let dspark = case["dspark"].as_bool().unwrap();
            for chunk in case["chunks"].as_array().unwrap() {
                let first = chunk["start"].as_u64().unwrap() as u32;
                let end = chunk["end"].as_u64().unwrap() as u32;
                let item = Item {
                    rows: 0..end - first,
                    prefix: first,
                };
                let logits = if end == tokens {
                    vec![end - first - 1]
                } else {
                    Vec::new()
                };
                let plan =
                    Plan::new(&FLASH, std::slice::from_ref(&item), &logits, dspark, 64).unwrap();
                assert_eq!(plan.endpoints, [end]);
                for (layer, expected) in plan.layers.iter().zip(chunk["layers"].as_array().unwrap())
                {
                    assert_eq!(layer.id as u64, expected["layer"].as_u64().unwrap());
                    for (name, rows) in [
                        ("keys", &layer.keys),
                        ("outputs", &layer.outputs),
                        ("features", &layer.features),
                    ] {
                        let ranges: Vec<[u32; 2]> =
                            rows.ranges().iter().map(|r| [r.start, r.end]).collect();
                        assert_eq!(
                            serde_json::to_value(ranges).unwrap(),
                            expected[name],
                            "end={end}, dspark={dspark}, layer={}, {name}",
                            layer.id
                        );
                    }
                }
                assert_eq!(
                    fake(std::slice::from_ref(&item), &plan, true),
                    fake(std::slice::from_ref(&item), &plan, false)
                );
                checked += 1;
            }
        }
        checked
    }

    #[test]
    fn scheduler_endpoint_cones_match_the_python_oracle() {
        assert_eq!(matches_the_python_oracle(8257, &[64, 192, 4032, 8064]), 12);
    }

    /// Timing cuts end grid chunks at granule multiples the TBT cap
    /// reaches: repeated 512-row cuts to the grid and past it, and small
    /// 64-row cuts before the rounded prompt-end anchor.
    #[test]
    fn timing_cut_cones_match_the_python_oracle() {
        let mut repeated: Vec<u32> = (1..16).map(|n| n * 512).collect();
        repeated.extend([8704, 8960]);
        assert_eq!(matches_the_python_oracle(9000, &repeated), 2 * 19);
        assert_eq!(
            matches_the_python_oracle(1500, &[64, 128, 192, 704, 1216, 1472]),
            2 * 7
        );
    }

    /// Packed timing-cut chunks of several requests, from off-granule and
    /// grid starts, never read a skipped row.
    #[test]
    fn packed_timing_cuts_never_read_a_skipped_row() {
        let items = [
            Item {
                rows: 0..495,
                prefix: 529,
            },
            Item {
                rows: 495..1007,
                prefix: 7168,
            },
            Item {
                rows: 1007..1071,
                prefix: 8192,
            },
            Item {
                rows: 1071..1107,
                prefix: 1472,
            },
        ];
        for dspark in [false, true] {
            let plan = Plan::new(&FLASH, &items, &[1106], dspark, 64).unwrap();
            assert_eq!(plan.endpoints, [1024, 7680, 8256, 1508]);
            assert_eq!(fake(&items, &plan, true), fake(&items, &plan, false));
        }
    }
}
