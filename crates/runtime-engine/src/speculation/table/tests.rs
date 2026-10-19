use super::*;

const ALL: [bool; MAX_DRAFT_TOKENS + 1] = [true; MAX_DRAFT_TOKENS + 1];

fn costs(entries: &[(u16, u32, u32, u32, u32)]) -> StepCosts {
    StepCosts::new(
        &entries
            .iter()
            .map(|&(items, base_us, step_us, drafting_us, row_us)| StepCost {
                items,
                base_us,
                step_us,
                drafting_us,
                row_us,
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

/// Survival products of conditional per-position acceptance.
fn survival(conditional: &[f64]) -> Vec<f64> {
    conditional
        .iter()
        .scan(1.0, |product, &rate| {
            *product *= rate;
            Some(*product)
        })
        .collect()
}

fn candidates(survival: &[Vec<f64>]) -> Vec<TableCandidate<'_>> {
    survival
        .iter()
        .map(|survival| TableCandidate {
            maximum_draft_tokens: survival.len(),
            survival,
        })
        .collect()
}

#[test]
fn tables_parse_validate_and_interpolate_between_concurrencies() {
    let table = StepCosts::parse("1:16000:0:4000:800, 8:32000:100:2000:600").unwrap();
    assert_eq!(table.entries().len(), 2);
    assert_eq!(table.estimate(1, 0, 1), 16000.0);
    assert_eq!(table.estimate(1, 1, 6), 16000.0 + 4000.0 + 5.0 * 800.0);
    // The step cost counts once when any item drafts, never without.
    assert_eq!(table.estimate(8, 0, 8), 32000.0);
    assert_eq!(table.estimate(8, 1, 9), 32000.0 + 100.0 + 2000.0 + 600.0);
    assert_eq!(
        table.estimate(8, 6, 32),
        32000.0 + 100.0 + 6.0 * 2000.0 + 24.0 * 600.0
    );
    // Three sevenths of the way in items, so is every coefficient.
    let weight = 3.0 / 7.0;
    let mid = 16000.0 + weight * 16000.0;
    let step = weight * 100.0;
    let drafting = 4000.0 - weight * 2000.0;
    let row = 800.0 - weight * 200.0;
    let expected = mid + step + 2.0 * drafting + 3.0 * row;
    assert!((table.estimate(4, 2, 7) - expected).abs() < 1e-9);
    // Beyond the table the nearest entry applies.
    assert_eq!(table.estimate(32, 0, 32), 32000.0);
    for bad in [
        "",
        "1:2:3:4",
        "x:1:1:1:1",
        "0:1:1:1:1",
        "2:1:1:1:1,1:1:1:1:1",
        "1:0:1:1:1",
    ] {
        assert!(StepCosts::parse(bad).is_err(), "{bad}");
    }
    // Equal tables share one copy.
    assert!(std::ptr::eq(
        table.entries(),
        StepCosts::parse("1:16000:0:4000:800,8:32000:100:2000:600")
            .unwrap()
            .entries()
    ));
    let many = (1..=MAX_STEP_COSTS + 1)
        .map(|items| format!("{items}:1:1:1:1"))
        .collect::<Vec<_>>();
    assert_eq!(
        StepCosts::parse(&many.join(",")),
        Err(StepCostsError::TooMany {
            maximum: MAX_STEP_COSTS
        })
    );
}

#[test]
fn c1_drafts_its_full_block_when_rows_are_cheap_and_none_when_ar_is_faster() {
    // Round 4 C1: a K=0 step about 16 ms, a full K=5 verify about 24 ms.
    let table = costs(&[(1, 15900, 0, 4400, 780)]);
    let accepting = [survival(&[0.88, 0.78, 0.70, 0.69, 0.66])];
    let plan = plan(&candidates(&accepting), &table, 5, usize::MAX, &ALL);
    assert_eq!(plan.drafts, [5]);
    // A drafter that rarely matches loses to the AR step.
    let rejecting = [survival(&[0.1, 0.1, 0.1, 0.1, 0.1])];
    let plan = super::plan(&candidates(&rejecting), &table, 5, usize::MAX, &ALL);
    assert_eq!(plan.drafts, [0]);
    assert_eq!(plan.expected_tokens, 1.0);
    assert_eq!(plan.estimated_us, 15900.0);
    // So does any drafter when the drafting pass costs more than the step.
    let slow = costs(&[(1, 15900, 0, 80000, 780)]);
    assert_eq!(
        super::plan(&candidates(&accepting), &slow, 5, usize::MAX, &ALL).drafts,
        [0]
    );
}

#[test]
fn drafting_cost_concentrates_drafts_and_row_cost_spreads_them() {
    let acceptance = (0..8)
        .map(|_| survival(&[0.8, 0.79, 0.78, 0.84, 0.46]))
        .collect::<Vec<_>>();
    // Each drafting item costs far more than a row: few items, full depth.
    let per_item = costs(&[(8, 30000, 0, 6000, 300)]);
    let plan = plan(&candidates(&acceptance), &per_item, 24, usize::MAX, &ALL);
    assert!(plan.drafts.iter().all(|&k| k == 0 || k >= 4), "{plan:?}");
    assert!(
        plan.drafts.iter().filter(|&&k| k > 0).count() <= 5,
        "{plan:?}"
    );
    // Drafting passes are cheap and rows are not: every item drafts, and
    // the 24-draft budget is not spent on deep positions.
    let per_row = costs(&[(8, 30000, 0, 100, 1500)]);
    let plan = super::plan(&candidates(&acceptance), &per_row, 24, usize::MAX, &ALL);
    assert!(
        plan.drafts.iter().all(|&k| (1..=3).contains(&k)),
        "{plan:?}"
    );
}

#[test]
fn a_per_step_drafting_cost_is_shared_by_every_drafting_item() {
    // Fit to the deferred-catch-up candidate at C8: one drafter launch per
    // step, little per drafting item, rows about as dear as items.
    let table = costs(&[(8, 22300, 4400, 700, 780)]);
    let acceptance = (0..8)
        .map(|_| survival(&[0.8, 0.79, 0.78, 0.84, 0.46]))
        .collect::<Vec<_>>();
    let plan = plan(&candidates(&acceptance), &table, 24, usize::MAX, &ALL);
    assert!(plan.drafts.iter().all(|&k| k >= 2), "{plan:?}");
    assert!(plan.expected_tokens / plan.estimated_us > 8.0 / 22300.0);
    // The same costs per item, without the shared step, favour fewer items.
    let per_item = costs(&[(8, 22300, 0, 4400 + 700, 780)]);
    let alone = super::plan(&candidates(&acceptance), &per_item, 24, usize::MAX, &ALL);
    assert!(
        alone.drafts.iter().filter(|&&k| k > 0).count()
            < plan.drafts.iter().filter(|&&k| k > 0).count(),
        "{alone:?} {plan:?}"
    );
}

/// Exhaustive search over every executable assignment of small batches.
fn brute(
    candidates: &[TableCandidate<'_>],
    table: &StepCosts,
    remaining: usize,
    maximum_drafting: usize,
    executable: &[bool; MAX_DRAFT_TOKENS + 1],
) -> f64 {
    let items = candidates.len();
    let mut best = items as f64 / table.estimate(items, 0, items);
    let mut counts = vec![0; items];
    loop {
        let total: usize = counts.iter().sum();
        let drafting = counts.iter().filter(|&&k| k > 0).count();
        if total <= remaining
            && drafting <= maximum_drafting
            && counts.iter().all(|&k| executable[k])
        {
            let expected = items as f64
                + counts
                    .iter()
                    .zip(candidates)
                    .map(|(&k, c)| c.survival[..k].iter().sum::<f64>())
                    .sum::<f64>();
            best = best.max(expected / table.estimate(items, drafting, items + total));
        }
        let mut index = 0;
        loop {
            if index == items {
                return best;
            }
            counts[index] += 1;
            if counts[index] <= candidates[index].maximum_draft_tokens {
                break;
            }
            counts[index] = 0;
            index += 1;
        }
    }
}

#[test]
fn plans_match_an_exhaustive_search() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |bound: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % bound
    };
    for case in 0..400 {
        let items = 1 + next(4) as usize;
        let survival = (0..items)
            .map(|_| {
                let depth = next(5) as usize;
                let rates = (0..depth)
                    .map(|_| next(101) as f64 / 100.0)
                    .collect::<Vec<_>>();
                self::survival(&rates)
            })
            .collect::<Vec<_>>();
        let table = costs(&[(
            items as u16,
            1000 + next(30000) as u32,
            next(8000) as u32,
            next(4000) as u32,
            next(3000) as u32,
        )]);
        let remaining = next(13) as usize;
        let maximum_drafting = 1 + next(4) as usize;
        let mut executable = ALL;
        if case % 3 == 0 {
            executable[3] = false;
        }
        let candidates = candidates(&survival);
        let plan = plan(
            &candidates,
            &table,
            remaining,
            maximum_drafting,
            &executable,
        );
        let total: usize = plan.drafts.iter().sum();
        assert!(total <= remaining);
        assert!(plan.drafts.iter().filter(|&&k| k > 0).count() <= maximum_drafting);
        assert!(plan.drafts.iter().all(|&k| executable[k]));
        let rate = plan.expected_tokens / plan.estimated_us;
        let best = brute(
            &candidates,
            &table,
            remaining,
            maximum_drafting,
            &executable,
        );
        assert!(
            (rate - best).abs() <= 1e-12 * best,
            "case {case}: {plan:?} rate {rate} best {best}"
        );
    }
}

#[test]
fn equal_candidates_and_repeated_inputs_give_one_plan() {
    let acceptance = (0..4)
        .map(|_| survival(&[0.9, 0.9, 0.9]))
        .collect::<Vec<_>>();
    let table = costs(&[(4, 20000, 0, 5000, 500)]);
    let first = plan(&candidates(&acceptance), &table, 6, usize::MAX, &ALL);
    for _ in 0..8 {
        assert_eq!(
            plan(&candidates(&acceptance), &table, 6, usize::MAX, &ALL),
            first
        );
    }
    assert!(first.drafts.iter().any(|&k| k > 0));
}
