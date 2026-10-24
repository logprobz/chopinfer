//! Common draft depths within a batch, with acceptance-ranked spare rows,
//! up to the depth a schedule gives the batch's concurrency.

use super::{MAX_DRAFT_TOKENS, PlanCandidate, SpeculationPolicy};

impl SpeculationPolicy {
    /// Reserve a common executable depth without ranking partial increments
    /// by acceptance history. Keep every draft the balanced plan needs.
    pub(super) fn reserve_balanced(
        &self,
        candidates: &[PlanCandidate<'_>],
        remaining: usize,
        planned: &[usize],
    ) -> Vec<usize> {
        let mut reserved = planned.to_vec();
        for depth in 1..=MAX_DRAFT_TOKENS {
            if !self.executable[depth] {
                continue;
            }
            let next = candidates
                .iter()
                .zip(planned)
                .map(|(candidate, &drafts)| {
                    self.executable_at_most(depth.min(candidate.maximum_draft_tokens))
                        .max(drafts)
                })
                .collect::<Vec<_>>();
            if next.iter().sum::<usize>() > remaining {
                break;
            }
            reserved = next;
        }
        reserved
    }

    /// No item verifies more than `limit` drafts; `None` leaves the request
    /// and batch limits alone, and `Some(0)` plans no drafts.
    pub(super) fn plan_balanced(
        &self,
        candidates: &[PlanCandidate<'_>],
        mut remaining: usize,
        limit: Option<usize>,
    ) -> Vec<usize> {
        let mut drafts = vec![0; candidates.len()];
        for depth in 1..=limit.unwrap_or(MAX_DRAFT_TOKENS).min(MAX_DRAFT_TOKENS) {
            if !self.executable[depth] {
                continue;
            }
            let next = candidates
                .iter()
                .map(|candidate| self.executable_at_most(depth.min(candidate.maximum_draft_tokens)))
                .collect::<Vec<_>>();
            let needed: usize = next.iter().zip(&drafts).map(|(n, d)| n - d).sum();
            if needed <= remaining {
                drafts = next;
                remaining -= needed;
                continue;
            }
            // Every uncapped item already has the deepest common executable
            // prefix. Rank the next increments by expected accepted tokens
            // per extra row. This is prefix survival, not conditional
            // acceptance of the last position alone.
            let mut order = next
                .iter()
                .zip(&drafts)
                .enumerate()
                .filter(|(_, (n, d))| n > d)
                .map(|(item, (&n, &d))| {
                    let gain: f64 = (d..n)
                        .map(|position| candidates[item].acceptance.survival(position))
                        .sum();
                    (item, gain / (n - d) as f64)
                })
                .collect::<Vec<_>>();
            order.sort_by(|(a, gain_a), (b, gain_b)| {
                gain_b.total_cmp(gain_a).then_with(|| a.cmp(b))
            });
            for (item, _) in order {
                let increment = next[item] - drafts[item];
                if increment <= remaining {
                    drafts[item] = next[item];
                    remaining -= increment;
                }
            }
            break;
        }
        drafts
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        AcceptanceEstimate, DraftDepths, DraftSource, SpeculationConfig, SpeculationMode,
        VerifyWidths,
    };
    use super::*;
    use runtime_scheduler::DraftBudget;

    fn scheduled(depths: DraftDepths) -> SpeculationPolicy {
        SpeculationPolicy::new(
            SpeculationConfig::new(
                5,
                32,
                SpeculationMode::Balanced(depths),
                DraftSource::Bundle,
            )
            .unwrap(),
        )
    }

    fn policy() -> SpeculationPolicy {
        scheduled(DraftDepths::default())
    }

    fn landed(depths: DraftDepths, widths: VerifyWidths) -> SpeculationPolicy {
        SpeculationPolicy::new(
            SpeculationConfig::new(
                5,
                32,
                SpeculationMode::Balanced(depths),
                DraftSource::Bundle,
            )
            .unwrap()
            .with_verify_widths(widths),
        )
    }

    fn candidate(acceptance: &AcceptanceEstimate, maximum: usize) -> PlanCandidate<'_> {
        PlanCandidate {
            maximum_draft_tokens: maximum,
            acceptance,
            fresh: false,
            remaining_tokens: 128,
        }
    }

    #[test]
    fn common_depth_fills_the_batch_before_deepening_any_request() {
        let estimate = AcceptanceEstimate::initial();
        for (items, depth) in [(1, 5), (8, 3), (16, 1), (32, 0)] {
            let candidates = vec![candidate(&estimate, 5); items];
            let drafts = policy().plan(&candidates, DraftBudget::new(32 - items, 32 - items));
            assert_eq!(drafts, vec![depth; items]);
        }
    }

    /// A schedule caps each concurrency's depth from its entry on: depth 0
    /// serves C16 without drafts, depth 2 keeps C8 shallower than its rows
    /// allow, and spare rows still fill only up to the cap.
    #[test]
    fn a_depth_schedule_caps_each_concurrency_and_zero_drafts_nothing() {
        let estimate = AcceptanceEstimate::initial();
        let depths = DraftDepths::parse("1:3,8:2,16:0").unwrap();
        for (items, expected) in [
            (1, vec![3]),
            (8, vec![2; 8]),
            (16, vec![0; 16]),
            (32, vec![0; 32]),
        ] {
            let candidates = vec![candidate(&estimate, 5); items];
            let drafts =
                scheduled(depths).plan(&candidates, DraftBudget::new(32 - items, 32 - items));
            assert_eq!(drafts, expected, "{items}");
        }
        // Seven items under the C1 entry: three each fill 21 of 25 spare
        // rows, and no item goes deeper than three.
        let candidates = vec![candidate(&estimate, 5); 7];
        assert_eq!(
            scheduled(depths).plan(&candidates, DraftBudget::new(25, 25)),
            vec![3; 7]
        );
        // The full schedule is the unscheduled plan wherever rows bind first.
        let full = DraftDepths::parse("1:5,8:3,16:1").unwrap();
        for items in [1, 4, 8, 12, 16, 24] {
            let candidates = vec![candidate(&estimate, 5); items];
            let budget = DraftBudget::new(32 - items, 32 - items);
            assert_eq!(
                scheduled(full).plan(&candidates, budget),
                policy().plan(&candidates, budget),
                "{items}"
            );
        }
    }

    /// With this schedule and these prepared widths, each concurrency
    /// verifies the widest width its limits reach, filled by acceptance:
    /// two items 8 rows, three to five items 16, six to eight 32, and one
    /// item keeps its 6. No item goes deeper than its schedule allows, and
    /// C16 still drafts nothing.
    #[test]
    fn plans_land_on_the_widest_prepared_width_the_limits_reach() {
        let estimate = AcceptanceEstimate::initial();
        let depths = DraftDepths::parse("1:5,8:3,16:0").unwrap();
        let widths = VerifyWidths::parse("6,8,16,32").unwrap();
        for (items, rows, expected) in [
            (1, 6, vec![5]),
            (2, 8, vec![3, 3]),
            (3, 16, vec![5, 4, 4]),
            (4, 16, vec![3; 4]),
            (5, 16, vec![3, 2, 2, 2, 2]),
            (6, 32, vec![5, 5, 4, 4, 4, 4]),
            (7, 32, vec![4, 4, 4, 4, 3, 3, 3]),
            (8, 32, vec![3; 8]),
            (16, 16, vec![0; 16]),
        ] {
            let candidates = vec![candidate(&estimate, 5); items];
            let budget = DraftBudget::new(32 - items, 32 - items);
            let drafts = landed(depths, widths).plan(&candidates, budget);
            assert_eq!(drafts, expected, "{items}");
            assert_eq!(items + drafts.iter().sum::<usize>(), rows, "{items}");
        }
        // Ten items at depth 1 reach 20 rows and land on 16: six draft.
        let shallow = DraftDepths::parse("1:5,8:1").unwrap();
        let candidates = vec![candidate(&estimate, 5); 10];
        let drafts = landed(shallow, widths).plan(&candidates, DraftBudget::new(22, 22));
        assert_eq!(drafts, [vec![1; 6], vec![0; 4]].concat());
        // Rows already on a width, or with no width between the decode rows
        // and the limits, keep the unlanded plan.
        for items in [1, 6, 7, 8, 12, 20] {
            let candidates = vec![candidate(&estimate, 5); items];
            let budget = DraftBudget::new(32 - items, 32 - items);
            assert_eq!(
                landed(depths, widths).plan(&candidates, budget),
                scheduled(depths).plan(&candidates, budget),
                "{items}"
            );
        }
    }

    /// With confidence, the plan's rows stay the step's width while every
    /// item reserves drafts up to its limits within the token capacity; a
    /// plan without drafts reserves nothing, and a capacity the plan already
    /// fills leaves the reservations at the plan.
    #[test]
    fn plans_with_confidence_reserve_drafts_and_keep_the_width() {
        use super::super::VerifyAllocation;
        let estimate = AcceptanceEstimate::initial();
        let allocation = VerifyAllocation::new(1, &[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap();
        let allocated = |capacity| {
            SpeculationPolicy::new(
                SpeculationConfig::new(
                    5,
                    capacity,
                    SpeculationMode::Balanced(DraftDepths::parse("1:5,8:3,16:0").unwrap()),
                    DraftSource::Bundle,
                )
                .unwrap()
                .with_verify_widths(VerifyWidths::parse("6,8,16,32").unwrap())
                .with_verify_allocation(Some(allocation)),
            )
        };
        for (items, reserved, width) in [
            (1, vec![5], None),
            (4, vec![5; 4], Some(16)),
            (6, vec![5; 6], Some(32)),
            (8, vec![5; 8], Some(32)),
            (16, vec![0; 16], None),
        ] {
            let mut policy = allocated(128);
            let candidates = vec![candidate(&estimate, 5); items];
            let spare = 128 - items;
            assert_eq!(
                policy.plan(&candidates, DraftBudget::new(spare, spare)),
                reserved
            );
            assert_eq!(policy.take_allocation().map(|a| a.width), width, "{items}");
            assert_eq!(policy.take_allocation(), None);
        }
        // A step whose verified drafts are decided in advance keeps the
        // plan unallocated: one drafting item, or a floor of one draft that
        // fills 16 items' 32 rows. A floor of zero leaves them to rank.
        let ranked = |floor, depths: &str, items| {
            let allocation =
                VerifyAllocation::new(floor, &[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap();
            let mut policy = SpeculationPolicy::new(
                SpeculationConfig::new(
                    5,
                    64,
                    SpeculationMode::Balanced(DraftDepths::parse(depths).unwrap()),
                    DraftSource::Bundle,
                )
                .unwrap()
                .with_verify_widths(VerifyWidths::parse("6,8,16,32").unwrap())
                .with_verify_allocation(Some(allocation)),
            );
            let candidates = vec![candidate(&estimate, 5); items];
            let spare = 64 - items;
            let drafts = policy.plan(&candidates, DraftBudget::new(spare, spare));
            (drafts, policy.take_allocation().map(|a| a.width))
        };
        assert_eq!(ranked(1, "1:3", 1), (vec![3], None));
        assert_eq!(ranked(1, "1:1", 16), (vec![1; 16], None));
        assert_eq!(ranked(0, "1:1", 16), (vec![3; 16], Some(32)));
        assert_eq!(ranked(1, "1:3", 8), (vec![5; 8], Some(32)));
        let mut narrow = allocated(32);
        let candidates = vec![candidate(&estimate, 5); 8];
        assert_eq!(
            narrow.plan(&candidates, DraftBudget::new(24, 24)),
            vec![3; 8]
        );
        assert_eq!(narrow.take_allocation(), None);
        // Request caps bound reservations; a host drafter never reserves.
        let mut policy = allocated(128);
        let capped = [5, 1, 0, 5].map(|cap| candidate(&estimate, cap));
        // The caps reach 15 rows, which land on width 8.
        assert_eq!(
            policy.plan(&capped, DraftBudget::new(124, 124)),
            [5, 1, 0, 5]
        );
        assert_eq!(policy.take_allocation().map(|a| a.width), Some(8));
        let mut host = SpeculationPolicy::new(
            SpeculationConfig::new(
                5,
                128,
                SpeculationMode::Balanced(DraftDepths::parse("1:5,8:3").unwrap()),
                DraftSource::Host,
            )
            .unwrap()
            .with_verify_allocation(Some(allocation)),
        );
        let candidates = vec![candidate(&estimate, 5); 8];
        assert_eq!(
            host.plan(&candidates, DraftBudget::new(120, 120)),
            vec![3; 8]
        );
        assert_eq!(host.take_allocation(), None);
    }

    /// Two engines with a depth schedule and verify widths that verify the
    /// same drafts but time their steps differently, one about 20 ms a step
    /// and the other anywhere from 30 to 120 ms, plan the same drafts at
    /// every step and concurrency, and neither trains a step-cost or
    /// realized table.
    #[test]
    fn balanced_plans_never_read_step_times() {
        let depths = DraftDepths::parse("1:5,8:3,16:1").unwrap();
        let widths = VerifyWidths::parse("6,8,16,32").unwrap();
        let mut engines = [landed(depths, widths), landed(depths, widths)];
        let mut estimates: [Vec<AcceptanceEstimate>; 2] =
            std::array::from_fn(|_| vec![AcceptanceEstimate::initial(); 32]);
        for step in 0..400_usize {
            let items = [1, 8, 16, 7, 12, 32, 3, 5][step / 40 % 8];
            let mut plans = Vec::new();
            for (engine, estimates) in engines.iter_mut().zip(&estimates) {
                let candidates = estimates[..items]
                    .iter()
                    .map(|estimate| candidate(estimate, 5))
                    .collect::<Vec<_>>();
                let spare = 32 - items;
                plans.push(engine.plan(&candidates, DraftBudget::new(spare, spare)));
            }
            assert_eq!(plans[0], plans[1], "step {step} at {items} items");
            let drafts = &plans[0];
            for (index, (engine, estimates)) in engines.iter_mut().zip(&mut estimates).enumerate() {
                for (item, &count) in drafts.iter().enumerate() {
                    if count > 0 {
                        let accepted = (step * 31 + item * 17) % (count + 1);
                        engine.observe(&mut estimates[item], count, accepted);
                    }
                }
                let micros = if index == 0 {
                    20_000 + (step % 7) as u64 * 100
                } else {
                    30_000 + (step * 7919 % 90_000) as u64
                };
                engine.observe_step(drafts, step % 3, micros);
            }
        }
        for engine in &engines {
            assert!(engine.costs.micros.is_empty());
            assert!(engine.block_steps.steps.is_empty());
            assert!(engine.realized.is_empty());
            assert!(engine.fresh_costs.is_empty());
        }
    }

    #[test]
    fn spare_rows_follow_prefix_survival_and_stable_ties() {
        let mut estimates: [AcceptanceEstimate; 3] =
            std::array::from_fn(|_| AcceptanceEstimate::initial());
        estimates[0].positions[..3].copy_from_slice(&[0.9, 0.9, 0.1]);
        estimates[1].positions[..3].copy_from_slice(&[0.6, 0.6, 0.9]);
        estimates[2].positions[..3].copy_from_slice(&[0.8, 0.8, 0.2]);
        let candidates = estimates
            .iter()
            .map(|e| candidate(e, 5))
            .collect::<Vec<_>>();
        assert_eq!(
            policy().plan(&candidates, DraftBudget::new(7, 7)),
            [2, 3, 2]
        );
        let equal = vec![candidate(&estimates[0], 5); 3];
        assert_eq!(policy().plan(&equal, DraftBudget::new(7, 7)), [3, 2, 2]);
    }

    #[test]
    fn request_caps_and_both_budgets_bound_the_common_prefix() {
        let estimate = AcceptanceEstimate::initial();
        let candidates = [5, 1, 0, 5].map(|cap| candidate(&estimate, cap));
        for budget in [DraftBudget::new(7, 20), DraftBudget::new(20, 7)] {
            assert_eq!(policy().plan(&candidates, budget), [3, 1, 0, 3]);
        }
        assert!(policy().plan(&[], DraftBudget::new(4, 4)).is_empty());
    }

    #[test]
    fn balanced_depths_respect_executable_prefixes() {
        let estimate = AcceptanceEstimate::initial();
        let candidates = vec![candidate(&estimate, 5); 3];
        let mut policy = policy().with_draft_rounding(|n| match n {
            0 => 0,
            1..=2 => 1,
            _ => 3,
        });
        assert_eq!(policy.plan(&candidates, DraftBudget::new(8, 8)), [3, 3, 1]);
        assert_eq!(policy.plan(&candidates, DraftBudget::new(9, 9)), [3, 3, 3]);
    }

    #[test]
    fn balanced_plans_keep_the_declared_drafter_item_rotation() {
        let estimate = AcceptanceEstimate::initial();
        let candidates = vec![candidate(&estimate, 5); 8];
        let mut policy = SpeculationPolicy::new(policy().config.with_maximum_draft_items(Some(3)));
        let mut seen = [false; 8];
        for _ in 0..3 {
            let drafts = policy.plan(&candidates, DraftBudget::new(24, 24));
            assert_eq!(drafts.iter().filter(|&&n| n > 0).count(), 3);
            for (seen, count) in seen.iter_mut().zip(drafts) {
                *seen |= count > 0;
            }
        }
        assert!(seen.into_iter().all(|seen| seen));
    }
}
