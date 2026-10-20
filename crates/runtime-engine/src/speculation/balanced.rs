//! Common draft depths within a batch, with acceptance-ranked spare rows.

use super::{MAX_DRAFT_TOKENS, PlanCandidate, SpeculationPolicy};

impl SpeculationPolicy {
    pub(super) fn plan_balanced(
        &self,
        candidates: &[PlanCandidate<'_>],
        mut remaining: usize,
    ) -> Vec<usize> {
        let mut drafts = vec![0; candidates.len()];
        for depth in 1..=MAX_DRAFT_TOKENS {
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
    use super::super::{AcceptanceEstimate, DraftSource, SpeculationConfig, SpeculationMode};
    use super::*;
    use runtime_scheduler::DraftBudget;

    fn policy() -> SpeculationPolicy {
        SpeculationPolicy::new(
            SpeculationConfig::new(5, 32, SpeculationMode::Balanced, DraftSource::Bundle).unwrap(),
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
