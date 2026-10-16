//! Shapes that startup warmup drives through the engine so that a bundle
//! captures every verify graph before serving.
//!
//! Graph buckets are powers of two, of rows and of items alike, as the
//! shared bundle runtime keys its graphs. Warmup requests pin their draft
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

/// Every verify shape a step of at most `sequences` decoding requests can
/// take, each with the smallest wave that reaches it. A step's rows are one
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
        for (total, reach) in fewest.iter().enumerate().skip(1) {
            let Some((count, _)) = *reach else {
                continue;
            };
            if count > drafting {
                continue;
            }
            // Keyed by item bucket first, so the waves grow.
            let key = (bucket(requests), bucket(requests + total));
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
        }
    }

    fn keys(shapes: &[VerifyShape]) -> Vec<(usize, usize)> {
        shapes
            .iter()
            .map(|shape| (shape.row_bucket, shape.item_bucket))
            .collect()
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
