//! Optional snapshots of policy decisions, before the step trains its costs.

use super::{Plan, SpeculationMode, SpeculationPolicy};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlanTrace {
    mode: &'static str,
    pub(super) reason: &'static str,
    items: usize,
    spare_rows: usize,
    planned_drafts: usize,
    planned_rows: usize,
    planned_bucket_rows: usize,
    ar_estimate_us: Option<f64>,
    candidate_expected_tokens: Option<f64>,
    candidate_estimate_us: Option<f64>,
    realized_ar_tokens_per_us: Option<f64>,
    realized_draft_tokens_per_us: Option<f64>,
    realized_draft_steps: u32,
    probe_interval: u16,
    probe_rechecks: u8,
    idle_plans: u16,
    cooldown_plans: u8,
    trial_steps: u32,
    trial_wait_plans: u16,
    resume_gain: f64,
    /// Per measured row bucket: rows, time, drafting sequences averaged and
    /// the fitted cost of one more drafting sequence, if measured.
    measured_verify_costs: Vec<(usize, f64, f64, Option<f64>)>,
}

impl PlanTrace {
    pub(crate) fn record(&self) -> serde_json::Value {
        serde_json::json!({
            "mode": self.mode,
            "reason": self.reason,
            "items": self.items,
            "spare_rows": self.spare_rows,
            "planned_drafts": self.planned_drafts,
            "planned_rows": self.planned_rows,
            "planned_bucket_rows": self.planned_bucket_rows,
            "ar_estimate_us": self.ar_estimate_us,
            "candidate_expected_tokens": self.candidate_expected_tokens,
            "candidate_estimate_us": self.candidate_estimate_us,
            "realized_ar_tokens_per_us": self.realized_ar_tokens_per_us,
            "realized_draft_tokens_per_us": self.realized_draft_tokens_per_us,
            "realized_draft_steps": self.realized_draft_steps,
            "probe_interval": self.probe_interval,
            "probe_rechecks": self.probe_rechecks,
            "idle_plans": self.idle_plans,
            "cooldown_plans": self.cooldown_plans,
            "trial_steps": self.trial_steps,
            "trial_wait_plans": self.trial_wait_plans,
            "resume_gain": self.resume_gain,
            "measured_verify_costs": self.measured_verify_costs.iter().map(|&(rows, micros, items, item_micros)| {
                serde_json::json!({"rows": rows, "micros": micros, "drafting_items": items,
                                   "micros_per_drafting_item": item_micros})
            }).collect::<Vec<_>>(),
        })
    }
}

impl SpeculationPolicy {
    pub(crate) fn with_diagnostics(mut self, enabled: bool) -> Self {
        self.trace_enabled = enabled;
        self
    }

    pub(crate) fn clear_diagnostics(&mut self) {
        self.decision = None;
    }

    pub(crate) fn decision(&self) -> Option<PlanTrace> {
        self.decision.clone()
    }

    pub(super) fn begin_diagnostics(&mut self, items: usize, spare_rows: usize) {
        if !self.trace_enabled {
            return;
        }
        let state = self.zero_draft.get(items).copied().unwrap_or_default();
        let realized = self.realized.get(&items).copied().unwrap_or_default();
        self.decision = Some(PlanTrace {
            mode: match self.config.mode {
                SpeculationMode::Fixed => "fixed",
                SpeculationMode::Balanced(_) => "balanced",
                SpeculationMode::Adaptive => "adaptive",
                SpeculationMode::Table(_) => "table",
            },
            reason: "not_selected",
            items,
            spare_rows,
            planned_drafts: 0,
            planned_rows: items,
            planned_bucket_rows: items,
            ar_estimate_us: self
                .costs
                .measured(items)
                .then(|| self.costs.estimate(items)),
            candidate_expected_tokens: None,
            candidate_estimate_us: None,
            realized_ar_tokens_per_us: realized.ar,
            realized_draft_tokens_per_us: realized.drafting,
            realized_draft_steps: realized.drafting_steps,
            probe_interval: state.probe_interval,
            probe_rechecks: state.rechecks,
            idle_plans: state.idle,
            cooldown_plans: state.remaining,
            trial_steps: state.trial,
            trial_wait_plans: state.trial_wait,
            resume_gain: if state.disabled {
                super::RESUME_GAIN
            } else {
                1.0
            },
            measured_verify_costs: self
                .block_steps
                .steps
                .range((items, 0)..(items + 1, 0))
                .map(|(&(_, rows), step)| {
                    let prior = self.costs.estimate(1) * super::PRIOR_SEQUENCE_COST;
                    let item = self.block_steps.sequence_cost(items, rows, prior);
                    (rows, step.micros, step.sequences, item)
                })
                .collect(),
        });
    }

    pub(super) fn trace_reason(&mut self, reason: &'static str) {
        if let Some(trace) = &mut self.decision {
            trace.reason = reason;
        }
    }

    pub(super) fn trace_candidate(&mut self, plan: &Plan) {
        if let Some(trace) = &mut self.decision
            && plan.drafts.iter().any(|&k| k > 0)
            && trace
                .candidate_estimate_us
                .zip(trace.candidate_expected_tokens)
                .is_none_or(|(cost, expected)| plan.rate() > expected / cost)
        {
            trace.candidate_expected_tokens = Some(plan.expected);
            trace.candidate_estimate_us = Some(plan.cost);
        }
    }

    pub(super) fn finish_diagnostics(&mut self, drafts: &[usize]) {
        if !self.trace_enabled {
            return;
        }
        let rows = drafts.len() + drafts.iter().sum::<usize>();
        let bucket = self.bucket(rows);
        if let Some(trace) = &mut self.decision {
            trace.planned_drafts = rows - drafts.len();
            trace.planned_rows = rows;
            trace.planned_bucket_rows = bucket;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        AcceptanceEstimate, DraftCostModel, DraftSource, PlanCandidate, Realized, SpeculationConfig,
    };
    use super::*;
    use runtime_scheduler::DraftBudget;

    fn policy() -> SpeculationPolicy {
        SpeculationPolicy::new(
            SpeculationConfig::new(5, 32, SpeculationMode::Adaptive, DraftSource::Bundle)
                .unwrap()
                .with_draft_cost_model(DraftCostModel::PerBlock),
        )
    }

    fn candidates(estimate: &AcceptanceEstimate) -> Vec<PlanCandidate<'_>> {
        vec![
            PlanCandidate {
                maximum_draft_tokens: 5,
                acceptance: estimate,
                fresh: false,
                remaining_tokens: 128,
            };
            8
        ]
    }

    fn baseline(policy: &mut SpeculationPolicy) {
        policy.observe_step(&[0; 8], 0, 20_000);
        policy.observe_step(&[0; 8], 0, 20_000);
    }

    #[test]
    fn zero_draft_trace_distinguishes_initial_estimated_realized_and_cooldown_rules() {
        let estimate = AcceptanceEstimate::initial();
        let candidates = candidates(&estimate);
        let budget = DraftBudget::new(24, 24);
        let mut p = policy().with_diagnostics(true);
        assert_eq!(p.plan(&candidates, budget), [0; 8]);
        assert_eq!(p.decision().unwrap().reason, "await_ar_baseline");
        baseline(&mut p);
        for rows in 9..=32 {
            p.block_steps.record((8, rows), 1_000_000.0, 8, 1.0);
        }
        assert_eq!(p.plan(&candidates, budget), [0; 8]);
        let trace = p.decision().unwrap().record();
        assert_eq!(trace["reason"], "estimated_slower");
        assert_eq!(trace["ar_estimate_us"], 20_000.0);
        assert_eq!(trace["measured_verify_costs"].as_array().unwrap().len(), 24);
        p.realized.insert(
            8,
            Realized {
                ar: Some(0.0004),
                drafting: Some(0.0002),
                drafting_weight: 1.0,
                drafting_steps: super::super::REALIZED_STEPS,
            },
        );
        assert_eq!(p.plan(&candidates, budget), [0; 8]);
        assert_eq!(p.decision().unwrap().reason, "realized_slower");
        assert_eq!(p.plan(&candidates, budget), [0; 8]);
        let trace = p.decision().unwrap().record();
        assert_eq!(trace["reason"], "cooldown");
        assert_eq!(trace["cooldown_plans"], super::super::ZERO_DRAFT_RECHECK);
        p.clear_diagnostics();
        assert!(p.decision().is_none());
    }

    #[test]
    fn diagnostics_leave_plans_and_learning_unchanged() {
        let estimate = AcceptanceEstimate::initial();
        let candidates = candidates(&estimate);
        let mut plain = policy();
        let mut traced = policy().with_diagnostics(true);
        for step in 0..160 {
            let budget = DraftBudget::new(24, 24);
            let drafts = plain.plan(&candidates, budget);
            assert_eq!(traced.plan(&candidates, budget), drafts);
            let accepted = drafts.iter().sum::<usize>() / 2;
            let micros = 20_000 + drafts.iter().sum::<usize>() as u64 * 2_000 + step % 3;
            for p in [&mut plain, &mut traced] {
                p.step_accepted = accepted;
                p.observe_step(&drafts, 0, micros);
            }
            assert_eq!(plain.costs, traced.costs);
            assert_eq!(plain.block_steps, traced.block_steps);
            assert_eq!(plain.zero_draft, traced.zero_draft);
            assert_eq!(plain.realized, traced.realized);
        }
    }
}
