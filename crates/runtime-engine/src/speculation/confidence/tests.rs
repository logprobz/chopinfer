use super::*;

fn candidate(probabilities: &[f64]) -> ConfidenceCandidate<'_> {
    ConfidenceCandidate {
        probabilities,
        maximum_draft_tokens: probabilities.len(),
    }
}

fn budget(maximum_rows: usize) -> ConfidenceBudget {
    ConfidenceBudget {
        maximum_rows,
        fixed_micros: 0.0,
    }
}

#[test]
fn products_and_global_budget_choose_prefixes_across_sequences() {
    let candidates = [candidate(&[0.9, 0.1, 1.0]), candidate(&[0.8, 0.8, 0.8])];
    let plan = allocate(
        &candidates,
        budget(5),
        &[VerifyBucketCost {
            rows: 8,
            micros: 10.0,
        }],
    )
    .unwrap();
    assert_eq!(plan.drafts, [1, 2]);
    assert_eq!((plan.rows, plan.bucket_rows), (5, 8));
    assert!((plan.expected_tokens - 4.34).abs() < 1e-12);
}

#[test]
fn a_bucket_jump_is_evaluated_after_filling_the_bucket() {
    let candidates = [candidate(&[1.0; 5]), candidate(&[1.0; 5])];
    let costs = [
        VerifyBucketCost {
            rows: 2,
            micros: 2.0,
        },
        VerifyBucketCost {
            rows: 8,
            micros: 5.0,
        },
    ];
    let plan = allocate(&candidates, budget(8), &costs).unwrap();
    assert_eq!(plan.drafts, [5, 1]);
    assert_eq!(plan.rows, 8);
    assert_eq!(
        allocate(&candidates, budget(4), &costs).unwrap().drafts,
        [0, 0]
    );
}

#[test]
fn exact_costs_preserve_the_winning_non_power_of_two_width() {
    let costs: Vec<_> = (1..=6)
        .map(|rows| VerifyBucketCost {
            rows,
            micros: if rows <= 3 { 10.0 } else { 100.0 },
        })
        .collect();
    let plan = allocate(&[candidate(&[0.9; 5])], budget(6), &costs).unwrap();
    assert_eq!((plan.rows, plan.bucket_rows, plan.drafts), (3, 3, vec![2]));
}

#[test]
fn fixed_drafter_cost_is_charged_once_and_caps_never_grow() {
    let mut candidates = [candidate(&[1.0; 5]), candidate(&[])];
    candidates[0].maximum_draft_tokens = 1;
    let costs = [
        VerifyBucketCost {
            rows: 2,
            micros: 1.0,
        },
        VerifyBucketCost {
            rows: 3,
            micros: 3.0,
        },
    ];
    assert_eq!(
        allocate(&candidates, budget(8), &costs).unwrap().drafts,
        [0, 0]
    );
    assert_eq!(
        allocate(
            &candidates,
            ConfidenceBudget {
                maximum_rows: 8,
                fixed_micros: 10.0
            },
            &costs
        )
        .unwrap()
        .drafts,
        [1, 0]
    );
}

#[test]
fn no_spare_rows_and_zero_confidence_choose_no_drafts() {
    for count in [1, 8, 32] {
        let costs = [VerifyBucketCost {
            rows: count * 6,
            micros: 10.0,
        }];
        assert_eq!(
            allocate(&vec![candidate(&[1.0; 5]); count], budget(count), &costs)
                .unwrap()
                .drafts,
            vec![0; count]
        );
        assert_eq!(
            allocate(
                &vec![candidate(&[0.0; 5]); count],
                budget(count * 6),
                &costs
            )
            .unwrap()
            .drafts,
            vec![0; count]
        );
    }
}

#[test]
fn malformed_confidence_and_costs_fail_before_any_selection() {
    let costs = [VerifyBucketCost {
        rows: 8,
        micros: 10.0,
    }];
    for probability in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        assert_eq!(
            allocate(&[candidate(&[probability])], budget(2), &costs),
            Err(ConfidenceError::InvalidProbability)
        );
    }
    assert_eq!(
        allocate(
            &[ConfidenceCandidate {
                probabilities: &[],
                maximum_draft_tokens: 1
            }],
            budget(2),
            &costs
        ),
        Err(ConfidenceError::InvalidCandidate)
    );
    assert_eq!(
        allocate(&[candidate(&[1.0; 17])], budget(32), &costs),
        Err(ConfidenceError::InvalidCandidate)
    );
    for value in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert_eq!(
            allocate(
                &[candidate(&[1.0])],
                budget(2),
                &[VerifyBucketCost {
                    rows: 2,
                    micros: value
                }]
            ),
            Err(ConfidenceError::InvalidCosts)
        );
    }
    assert_eq!(
        allocate(
            &[candidate(&[1.0])],
            budget(2),
            &[VerifyBucketCost {
                rows: 1,
                micros: 1.0
            }]
        ),
        Err(ConfidenceError::MissingCost)
    );
    assert_eq!(
        allocate(&[candidate(&[1.0])], budget(2), &[costs[0], costs[0]]),
        Err(ConfidenceError::InvalidCosts)
    );
    assert_eq!(
        allocate(&[], budget(2), &costs),
        Err(ConfidenceError::InvalidBudget)
    );
    assert_eq!(
        allocate(&[candidate(&[])], budget(0), &costs),
        Err(ConfidenceError::InvalidBudget)
    );
}

#[test]
fn greedy_prefix_allocation_matches_exhaustive_search() {
    // Independently enumerate every feasible vector for all combinations of
    // three conditional probabilities, including zero and ties.
    for code in 0..729usize {
        let mut code = code;
        let probabilities: Vec<[f64; 2]> = (0..3)
            .map(|_| {
                std::array::from_fn(|_| {
                    let p = [0.0, 0.5, 1.0][code % 3];
                    code /= 3;
                    p
                })
            })
            .collect();
        let candidates: Vec<_> = probabilities.iter().map(|p| candidate(p)).collect();
        for maximum in 3..=9 {
            let costs = [
                VerifyBucketCost {
                    rows: 4,
                    micros: 4.0,
                },
                VerifyBucketCost {
                    rows: 9,
                    micros: 7.0,
                },
            ];
            let plan = allocate(&candidates, budget(maximum), &costs).unwrap();
            let mut best: f64 = 0.0;
            for a in 0..=2 {
                for b in 0..=2 {
                    for c in 0..=2 {
                        let widths = [a, b, c];
                        let rows = 3 + widths.iter().sum::<usize>();
                        if rows > maximum {
                            continue;
                        }
                        let mut expected = 3.0;
                        for (p, &k) in probabilities.iter().zip(&widths) {
                            for end in 1..=k {
                                expected += p[..end].iter().product::<f64>();
                            }
                        }
                        best = best.max(expected / if rows <= 4 { 4.0 } else { 7.0 });
                    }
                }
            }
            assert_eq!(plan.expected_tokens / plan.estimated_micros, best);
        }
    }
}
