//! Shapes that startup warmup drives through the engine so that a bundle
//! captures selected verify graphs before serving.
//!
//! Most graph keys round rows and items to powers of two. Exact graphs
//! retain both counts. Warmup requests pin their draft
//! tokens (`EngineRequest::with_pinned_draft_tokens`), so one decode step of
//! a wave reaches one (row bucket, item bucket) shape.
use std::collections::BTreeMap;

/// What verify warmup needs to know of the served speculation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyWarmup {
    /// Rows one verify item may take, its drafts plus one.
    pub item_rows: Vec<usize>,
    /// Most draft tokens of one item.
    pub maximum_draft_tokens: usize,
    /// Most decode and verify rows of one step.
    pub step_rows: usize,
    /// Most items that draft in one step, when limited.
    pub drafting_items: Option<usize>,
    /// Graph keys retain exact row and item counts instead of rounded buckets.
    pub exact_graphs: bool,
    /// Verify widths wider than `step_rows` that are also warmed, each at
    /// every concurrency wave whose items reach it with even drafts: the
    /// balanced plan's listed widths above the rows warmed exhaustively.
    pub widths: Vec<usize>,
    /// Verify steps with an allocation, which warmup reaches with unpinned
    /// requests because pinned drafts are never allocated.
    pub allocated: Vec<AllocatedShape>,
}

/// A verify step with an allocation and the wave that reaches it: `items`
/// fresh unpinned requests, whose plan verifies `width` rows of
/// `reserved_rows`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocatedShape {
    pub items: usize,
    pub width: usize,
    pub reserved_rows: usize,
}

/// One verify graph shape and a wave that reaches it: the wave's requests
/// pin these draft tokens, zero for a request that only decodes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyShape {
    pub row_bucket: usize,
    pub item_bucket: usize,
    pub drafts: Vec<usize>,
}

/// Graph buckets are powers of two, of rows and of items alike.
fn bucket(count: usize) -> usize {
    count.max(1).next_power_of_two()
}

/// Rounded keys use the smallest wave that reaches each shape. Exact keys
/// cover every reachable width at the concurrency waves, the widest shape
/// where the row budget can draft for every request, and each reachable
/// drafter count. Other exact keys
/// remain lazy. A step's rows are one
/// per decoding request plus its drafts; at least one request verifies, the
/// rows stay within `step_rows`, and at most `drafting_items` draft.
pub fn verify_shapes(sequences: usize, warmup: &VerifyWarmup) -> Vec<VerifyShape> {
    let mut extras: Vec<usize> = warmup
        .item_rows
        .iter()
        .filter(|&&rows| rows > 1 && rows - 1 <= warmup.maximum_draft_tokens)
        .map(|rows| rows - 1)
        .collect();
    extras.sort_unstable();
    extras.dedup();
    let mut shapes = BTreeMap::new();
    let waves = super::waves(sequences);
    for requests in 1..=sequences {
        let Some(room) = warmup.step_rows.checked_sub(requests) else {
            break;
        };
        let drafting = warmup.drafting_items.unwrap_or(requests).min(requests);
        // fewest[e]: the fewest drafting requests whose drafts sum to e,
        // with the last draft count that reaches it.
        let mut fewest: Vec<Option<(usize, usize)>> = vec![None; room + 1];
        fewest[0] = Some((0, 0));
        for total in 1..=room {
            fewest[total] = extras
                .iter()
                .filter(|&&extra| extra <= total)
                .filter_map(|&extra| fewest[total - extra].map(|(count, _)| (count + 1, extra)))
                .min();
        }
        let widest = fewest
            .iter()
            .enumerate()
            .rev()
            .find(|(_, reach)| reach.is_some_and(|(count, _)| count <= drafting))
            .map(|(total, _)| total)
            .unwrap_or(0);
        for (total, reach) in fewest.iter().enumerate().skip(1) {
            let Some((count, _)) = *reach else {
                continue;
            };
            if count > drafting {
                continue;
            }
            if warmup.exact_graphs && !waves.contains(&requests) {
                let all_fit = extras.first().is_some_and(|extra| {
                    requests
                        .checked_mul(extra + 1)
                        .is_some_and(|rows| rows <= warmup.step_rows)
                });
                if !all_fit || total != widest {
                    continue;
                }
            }
            // Keyed by items first, so the waves grow.
            let key = if warmup.exact_graphs {
                (requests, requests + total)
            } else {
                (bucket(requests), bucket(requests + total))
            };
            shapes.entry(key).or_insert_with(|| {
                let mut drafts = vec![0; requests];
                let (mut left, mut index) = (total, 0);
                while left > 0 {
                    let (_, extra) = fewest[left].expect("reachable");
                    drafts[index] = extra;
                    left -= extra;
                    index += 1;
                }
                drafts
            });
        }
    }
    // The width witnesses above minimize drafting requests. Add one wave
    // per reachable drafter count so its graph is warmed as well.
    if warmup.exact_graphs
        && let Some(&extra) = extras.first()
    {
        let limit = sequences.min(warmup.drafting_items.unwrap_or(sequences));
        for requests in 1..=limit {
            let Some(rows) = requests.checked_mul(extra + 1) else {
                break;
            };
            if rows > warmup.step_rows {
                break;
            }
            shapes.insert((requests, rows), vec![extra; requests]);
        }
    }
    // A listed wider width at each wave that reaches it with drafts as
    // even as the per-item depths allow.
    if warmup.exact_graphs {
        for &width in warmup.widths.iter().filter(|&&w| w > warmup.step_rows) {
            for &requests in &waves {
                let Some(extra) = width.checked_sub(requests).filter(|&extra| extra > 0) else {
                    continue;
                };
                let (base, more) = (extra / requests, extra % requests);
                let drafts: Vec<usize> = (0..requests)
                    .map(|item| base + usize::from(item < more))
                    .collect();
                let drafting = drafts.iter().filter(|&&d| d > 0).count();
                if drafts.iter().all(|d| *d == 0 || extras.contains(d))
                    && warmup.drafting_items.is_none_or(|limit| drafting <= limit)
                {
                    shapes.insert((requests, width), drafts);
                }
            }
        }
    }
    shapes
        .into_iter()
        .map(|((item_bucket, row_bucket), drafts)| VerifyShape {
            row_bucket,
            item_bucket,
            drafts,
        })
        .collect()
}

/// Prompt lengths whose first drafting step catches a drafter up on about
/// three quarters of each row bucket up to `step_rows`.
pub fn context_ladder(step_rows: usize, limit: usize) -> Vec<usize> {
    std::iter::successors(Some(2_usize), |rows| rows.checked_mul(2))
        .take_while(|&rows| rows <= bucket(step_rows))
        .map(|rows| (rows * 3 / 4).max(1))
        .filter(|&prompt| prompt + 2 < limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dflash(step_rows: usize, drafting_items: Option<usize>) -> VerifyWarmup {
        VerifyWarmup {
            item_rows: vec![2, 4, 8],
            maximum_draft_tokens: 7,
            step_rows,
            drafting_items,
            exact_graphs: false,
            widths: Vec::new(),
            allocated: Vec::new(),
        }
    }

    fn keys(shapes: &[VerifyShape]) -> Vec<(usize, usize)> {
        shapes
            .iter()
            .map(|shape| (shape.row_bucket, shape.item_bucket))
            .collect()
    }

    #[test]
    fn exact_widths_cover_concurrency_waves_and_every_drafter_count() {
        for (sequences, count) in [(1, 5), (8, 67), (16, 97), (32, 97)] {
            let warmup = VerifyWarmup {
                item_rows: (2..=6).collect(),
                maximum_draft_tokens: 5,
                step_rows: 32,
                drafting_items: None,
                exact_graphs: true,
                widths: Vec::new(),
                allocated: Vec::new(),
            };
            let shapes = verify_shapes(sequences, &warmup);
            assert_eq!(shapes.len(), count);
            for requests in super::super::waves(sequences) {
                for rows in requests + 1..=(requests * 6).min(32) {
                    assert!(keys(&shapes).contains(&(rows, requests)));
                }
            }
            for drafting in 1..=sequences.min(16) {
                assert!(shapes.iter().any(|s| s.drafts == vec![1; drafting]));
            }
            for requests in 1..=sequences.min(16) {
                assert!(keys(&shapes).contains(&((requests * 6).min(32), requests)));
            }
            for shape in shapes {
                assert_eq!(shape.item_bucket, shape.drafts.len());
                assert_eq!(
                    shape.row_bucket,
                    shape.drafts.len() + shape.drafts.iter().sum::<usize>()
                );
                assert!(shape.row_bucket <= 32);
            }
        }
    }

    #[test]
    fn exact_selection_bounds_the_complete_adaptive_geometry_set() {
        let warmup = VerifyWarmup {
            item_rows: (2..=6).collect(),
            maximum_draft_tokens: 5,
            step_rows: 32,
            drafting_items: None,
            exact_graphs: true,
            widths: Vec::new(),
            allocated: Vec::new(),
        };
        let selected = verify_shapes(32, &warmup);
        let mut complete = std::collections::BTreeSet::new();
        // Independent reachable-sum enumeration permits every per-item
        // depth, including zero for AR items in a mixed step.
        let mut sums = std::collections::BTreeSet::from([0]);
        for items in 1..=32 {
            sums = sums
                .iter()
                .flat_map(|sum| (0..=5).map(move |draft| sum + draft))
                .filter(|sum| items + sum <= 32)
                .collect();
            for &drafts in sums.iter().filter(|&&sum| sum != 0) {
                complete.insert((items + drafts, items));
            }
        }
        assert_eq!(complete.len(), 426);
        assert_eq!(selected.len(), 97);
        assert!(keys(&selected).iter().all(|key| complete.contains(key)));
        // Three target pieces per rank would require 1278 handles for the
        // complete set, before any draft, AR or catch-up graph.
        assert_eq!(complete.len() * 3, 1278);
        assert_eq!(selected.len() * 3, 291);
    }

    /// Listed widths above the exhaustive rows warm once per wave that
    /// reaches them with even drafts, and nothing else changes.
    #[test]
    fn listed_wide_widths_warm_at_the_waves_that_reach_them() {
        let narrow = VerifyWarmup {
            item_rows: (2..=6).collect(),
            maximum_draft_tokens: 5,
            step_rows: 32,
            drafting_items: None,
            exact_graphs: true,
            widths: Vec::new(),
            allocated: Vec::new(),
        };
        let wide = VerifyWarmup {
            widths: vec![16, 32, 48, 64],
            ..narrow.clone()
        };
        let before = verify_shapes(32, &narrow);
        let after = verify_shapes(32, &wide);
        let added: Vec<_> = after.iter().filter(|s| !before.contains(s)).collect();
        assert_eq!(after.len(), before.len() + 5);
        assert_eq!(
            added
                .iter()
                .map(|s| (s.row_bucket, s.item_bucket))
                .collect::<Vec<_>>(),
            [(48, 8), (48, 16), (64, 16), (48, 32), (64, 32)]
        );
        assert_eq!(added[0].drafts, vec![5; 8]);
        assert_eq!(added[2].drafts, vec![3; 16]);
        assert_eq!(added[3].drafts, [vec![1; 16], vec![0; 16]].concat());
        for shape in added {
            assert_eq!(
                shape.row_bucket,
                shape.drafts.len() + shape.drafts.iter().sum::<usize>()
            );
        }
    }

    #[test]
    fn exact_widths_respect_sparse_depths_and_drafter_limits() {
        let warmup = VerifyWarmup {
            exact_graphs: true,
            ..dflash(16, Some(2))
        };
        let shapes = verify_shapes(4, &warmup);
        let mut expected = std::collections::BTreeSet::new();
        for requests in [1, 2, 4] {
            for first in [0, 1, 3, 7] {
                for second in [0, 1, 3, 7] {
                    let drafted = usize::from(first != 0) + usize::from(second != 0);
                    let rows = requests + first + second;
                    if drafted > 0 && drafted <= requests && rows <= 16 {
                        expected.insert((requests, rows));
                    }
                }
            }
        }
        // Non-wave concurrency also warms its widest reachable shape.
        expected.insert((3, 3 + 7 + 3));
        assert_eq!(
            keys(&shapes),
            expected
                .into_iter()
                .map(|(items, rows)| (rows, items))
                .collect::<Vec<_>>()
        );
        assert!(
            shapes
                .iter()
                .all(|s| s.drafts.iter().filter(|&&d| d > 0).count() <= 2)
        );
    }

    #[test]
    fn four_sequences_reach_every_shape_with_the_smallest_wave() {
        let shapes = verify_shapes(4, &dflash(32, None));
        assert_eq!(
            keys(&shapes),
            [
                (2, 1),
                (4, 1),
                (8, 1),
                (4, 2),
                (8, 2),
                (16, 2),
                (4, 4),
                (8, 4),
                (16, 4),
                (32, 4)
            ]
        );
        // Rows 4 with four item slots: three requests, one drafting one.
        let four = &shapes[6];
        assert_eq!(four.drafts, [1, 0, 0]);
        for shape in &shapes {
            let rows = shape.drafts.len() + shape.drafts.iter().sum::<usize>();
            assert_eq!(bucket(rows), shape.row_bucket, "{shape:?}");
            assert_eq!(bucket(shape.drafts.len()), shape.item_bucket);
            assert!(shape.drafts.iter().all(|d| [0, 1, 3, 7].contains(d)));
            assert!(rows <= 32 && shape.drafts.iter().any(|&d| d > 0));
        }
    }

    /// Every (rows, items) a step of up to 32 requests with 2, 4 or 8 rows
    /// per verify item can take, by brute force over the drafting counts.
    #[test]
    fn thirty_two_sequences_match_a_brute_force_enumeration() {
        for drafting in [None, Some(4)] {
            let mut expected = std::collections::BTreeSet::new();
            for requests in 1..=32_usize {
                let limit = drafting.unwrap_or(requests).min(requests);
                for one in 0..=limit {
                    for three in 0..=limit - one {
                        for seven in 0..=limit - one - three {
                            let rows = requests + one + 3 * three + 7 * seven;
                            if one + three + seven > 0 && rows <= 256 {
                                expected.insert((bucket(requests), bucket(rows)));
                            }
                        }
                    }
                }
            }
            let shapes = verify_shapes(32, &dflash(256, drafting));
            assert_eq!(
                keys(&shapes),
                expected
                    .into_iter()
                    .map(|(items, rows)| (rows, items))
                    .collect::<Vec<_>>(),
                "{drafting:?}"
            );
            for shape in &shapes {
                let drafting_now = shape.drafts.iter().filter(|&&d| d > 0).count();
                assert!(drafting.is_none_or(|limit| drafting_now <= limit));
            }
        }
        // 32 sequences: 3 + 3 + 4 + 4 + 4 + 4 shapes by item bucket.
        assert_eq!(verify_shapes(32, &dflash(256, None)).len(), 22);
    }

    #[test]
    fn the_context_ladder_covers_each_row_bucket() {
        assert_eq!(context_ladder(256, 4096), [1, 3, 6, 12, 24, 48, 96, 192]);
        assert_eq!(context_ladder(100, 4096), [1, 3, 6, 12, 24, 48, 96]);
        assert_eq!(context_ladder(256, 50), [1, 3, 6, 12, 24]);
    }
}
