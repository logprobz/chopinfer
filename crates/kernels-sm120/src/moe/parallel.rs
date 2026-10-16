//! Model-neutral geometry and scoring for the shared parallel route engine.
use super::*;

/// Bytes of the arrival count every parallel route launch takes. It must be
/// zeroed at create, bound by no other launch, and used by one stream: the
/// route resets it on election and cannot detect a small stale count
/// (route_parallel.cuh).
pub const PARALLEL_ROUTE_TICKET_BYTES: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteScoring {
    Softmax,
    SigmoidBias,
    GroupedSoftmax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParallelRoute {
    pub experts: u32,
    pub local_experts: u32,
    pub top_k: u32,
    pub padding: u32,
    pub capacity: u32,
    pub groups: u32,
    pub take_groups: u32,
    pub scoring: RouteScoring,
}

impl ParallelRoute {
    pub fn plan(self, live: u32) -> Result<LaunchPlan, Refusal> {
        let Self {
            experts,
            local_experts,
            top_k,
            padding,
            capacity,
            groups,
            take_groups,
            ..
        } = self;
        if experts == 0
            || experts % 32 != 0
            || !(32..=512).contains(&local_experts)
            || !local_experts.is_power_of_two()
            || experts % local_experts != 0
            || !(1..=32).contains(&top_k)
            || top_k > experts
            || padding == 0
            || !(1..=256).contains(&capacity)
            || live > capacity
            || groups == 0
            || experts % groups != 0
            || take_groups == 0
            || take_groups > groups
            || top_k > experts / groups * take_groups
        {
            return Err(Refusal("unsupported parallel route geometry or live count"));
        }
        let wide = capacity > 32;
        let route_groups = (capacity * top_k).div_ceil(32);
        let parts = if wide { 1024 / local_experts } else { 1 };
        // The builder's static shared arrays, within the 48 KiB static limit.
        let static_shared = route_groups * local_experts * 2
            + parts * local_experts * 4
            + local_experts * 8
            + local_experts / 32 * 8
            + 4;
        if static_shared > 48 * 1024 {
            return Err(Refusal("parallel route metadata exceeds shared memory"));
        }
        let plan = LaunchPlan {
            grid: [capacity.div_ceil(if wide { 8 } else { 1 }), 1, 1],
            block: [
                if wide {
                    1024
                } else {
                    (capacity * top_k).next_multiple_of(32).max(local_experts)
                },
                1,
                1,
            ],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// Body arguments have the common route ABI. Bias and scale expressions
    /// are supplied by the instantiation, without model names in the planner.
    pub fn body(self, rank: &str, bias: &str, scale: &str) -> Result<String, Refusal> {
        self.plan(self.capacity)?;
        let policy = match self.scoring {
            RouteScoring::Softmax => "chop::moe::Softmax{}".into(),
            RouteScoring::SigmoidBias => format!("chop::moe::SigmoidBias{{{bias},{scale}}}"),
            RouteScoring::GroupedSoftmax => "chop::moe::GroupedSoftmax{}".into(),
        };
        Ok(format!(
            "chop::moe::route_parallel<{},{},{},{},{},{},{}>(logits,ids,weights,counts,offsets,sorted_rows,num_padded_rows,live_tokens,active_experts,{rank},ticket,{policy});",
            self.experts,
            self.top_k,
            self.local_experts,
            self.capacity,
            self.padding,
            self.groups,
            self.take_groups
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_route_geometry_covers_shards_groups_and_wide_rows() {
        for (experts, top_k) in [(256, 8), (512, 10)] {
            for local_experts in [64, 128, 256] {
                for capacity in [16, 32, 33, 64, 128, 256] {
                    let route = ParallelRoute {
                        experts,
                        local_experts,
                        top_k,
                        padding: 16,
                        capacity,
                        groups: 8,
                        take_groups: 4,
                        scoring: RouteScoring::SigmoidBias,
                    };
                    for live in [0, 1, capacity] {
                        let plan = route.plan(live).unwrap();
                        assert_eq!(
                            plan.grid[0],
                            capacity.div_ceil(if capacity > 32 { 8 } else { 1 })
                        );
                    }
                    assert!(route.plan(capacity + 1).is_err());
                    assert!(
                        route
                            .body("0", "bias", "scale")
                            .unwrap()
                            .contains("SigmoidBias{bias,scale}")
                    );
                }
            }
        }
        // 80 route groups of 512 local experts need 80 KiB of counts.
        let oversized = ParallelRoute {
            experts: 512,
            local_experts: 512,
            top_k: 10,
            padding: 16,
            capacity: 256,
            groups: 1,
            take_groups: 1,
            scoring: RouteScoring::Softmax,
        };
        assert!(oversized.plan(0).is_err());
        assert!(
            ParallelRoute {
                capacity: 64,
                ..oversized
            }
            .plan(0)
            .is_ok()
        );
    }
}
