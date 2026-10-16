//! Committed target-row geometry, aggregated without per-step log I/O.

use std::collections::BTreeMap;
use std::fmt::Write;

use crate::{BatchPhase, ModelBatchItem};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StepRows {
    pub(crate) prefill_tokens: usize,
    pub(crate) prefill_items: usize,
    pub(crate) decode_rows: usize,
    pub(crate) verify_items: usize,
    pub(crate) verify_rows: usize,
    pub(crate) row_capacity: Option<usize>,
}

impl StepRows {
    /// Execution items already contain the rounded, realized draft lengths.
    pub(crate) fn from_items(items: &[ModelBatchItem], capacity: Option<usize>) -> Self {
        let mut rows = Self::default();
        for item in items {
            match item.phase() {
                BatchPhase::Prefill { tokens } => {
                    rows.prefill_tokens += tokens;
                    rows.prefill_items += 1;
                }
                BatchPhase::Decode => rows.decode_rows += 1,
                BatchPhase::Verify { draft_tokens } => {
                    rows.verify_items += 1;
                    rows.verify_rows += draft_tokens + 1;
                }
            }
        }
        // A mixed batch may execute several programs. Do not invent one cap.
        rows.row_capacity = capacity.filter(|&capacity| {
            rows.prefill_items == 0 && rows.live_rows() > 0 && capacity >= rows.live_rows()
        });
        rows
    }

    pub(crate) const fn live_rows(self) -> usize {
        self.decode_rows + self.verify_rows
    }
}

/// Bound both memory and exported series even for an executor with arbitrary
/// capacities. Dropped observations are explicit, so receipts can reject an
/// incomplete histogram. Existing bins keep counting after the bound is hit.
const MAX_BINS: usize = 4096;

#[derive(Debug, Default)]
pub(crate) struct StepRowHistogram {
    bins: BTreeMap<(usize, usize, Option<usize>), u64>,
    dropped: u64,
}

impl StepRowHistogram {
    pub(crate) fn observe(&mut self, rows: StepRows) {
        if rows.prefill_items > 0 || rows.live_rows() == 0 {
            return;
        }
        let key = (rows.decode_rows, rows.verify_rows, rows.row_capacity);
        if let Some(count) = self.bins.get_mut(&key) {
            *count = count.saturating_add(1);
        } else if self.bins.len() < MAX_BINS {
            self.bins.insert(key, 1);
        } else {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    pub(crate) fn render(&self, output: &mut String, labels: &str) {
        writeln!(output, "# TYPE chop_step_rows_total counter").unwrap();
        for ((decode, verify, capacity), count) in &self.bins {
            let capacity = capacity.map_or_else(|| "unknown".to_owned(), |rows| rows.to_string());
            writeln!(
                output,
                "chop_step_rows_total{{{labels},decode_rows=\"{decode}\",verify_rows=\"{verify}\",row_capacity=\"{capacity}\"}} {count}"
            )
            .unwrap();
        }
        crate::compatibility::counter(output, "chop_step_rows_dropped_total", labels, self.dropped);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_uses_target_rows_and_refuses_an_invalid_or_mixed_capacity() {
        let item = |phase| ModelBatchItem::new(crate::RequestId::new(1), phase, vec![0], true);
        let mut items = vec![
            item(BatchPhase::Verify { draft_tokens: 2 }),
            item(BatchPhase::Decode),
        ];
        let rows = StepRows::from_items(&items, Some(4));
        assert_eq!(
            (
                rows.verify_items,
                rows.verify_rows,
                rows.decode_rows,
                rows.live_rows()
            ),
            (1, 3, 1, 4)
        );
        assert_eq!(rows.row_capacity, Some(4));
        assert_eq!(StepRows::from_items(&items, Some(2)).row_capacity, None);
        items.push(item(BatchPhase::Prefill { tokens: 7 }));
        let mixed = StepRows::from_items(&items, Some(16));
        assert_eq!((mixed.prefill_tokens, mixed.prefill_items), (7, 1));
        assert_eq!(mixed.row_capacity, None);
    }

    #[test]
    fn histogram_separates_ar_mixed_verify_and_unknown_capacities() {
        let mut histogram = StepRowHistogram::default();
        let verify = StepRows {
            decode_rows: 1,
            verify_items: 2,
            verify_rows: 6,
            row_capacity: Some(8),
            ..StepRows::default()
        };
        histogram.observe(verify);
        histogram.observe(verify);
        histogram.observe(StepRows {
            prefill_items: 1,
            ..verify
        });
        histogram.observe(StepRows::default());
        histogram.observe(StepRows {
            row_capacity: None,
            ..verify
        });
        histogram.observe(StepRows {
            decode_rows: 7,
            row_capacity: Some(8),
            ..StepRows::default()
        });
        let mut output = String::new();
        histogram.render(&mut output, "engine=\"0\"");
        assert!(output.contains("decode_rows=\"1\",verify_rows=\"6\",row_capacity=\"8\"} 2"));
        assert!(output.contains("decode_rows=\"1\",verify_rows=\"6\",row_capacity=\"unknown\"} 1"));
        assert!(output.contains("decode_rows=\"7\",verify_rows=\"0\",row_capacity=\"8\"} 1"));
        assert_eq!(histogram.bins.len(), 3);
    }

    #[test]
    fn histogram_overflow_is_visible_and_existing_bins_keep_counting() {
        let mut histogram = StepRowHistogram::default();
        for decode_rows in 1..=MAX_BINS + 2 {
            histogram.observe(StepRows {
                decode_rows,
                ..StepRows::default()
            });
        }
        histogram.observe(StepRows {
            decode_rows: 1,
            ..StepRows::default()
        });
        assert_eq!(histogram.bins.len(), MAX_BINS);
        assert_eq!(histogram.bins[&(1, 0, None)], 2);
        assert_eq!(histogram.dropped, 2);
    }
}
