//! Sqrt-softplus routing on the shared local-route engine, selecting a
//! separate bias vector for each token and recording its unbiased scores.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::properties::{Invariance, Ties};
use kernel_core::router::RowBiasRouterQuery;

use super::{LocalRoute, ROW_THREADS, RouteKernel};
use crate::moe::ptr;

fn score(experts: u32) -> String {
    format!(
        "chop::moe::SqrtSoftplusRowBias<{experts}>{{bias,alternate_bias,bias_select,scores,__uint_as_float(scale_bits)}}"
    )
}

fn extra_params() -> [EntryParam; 3] {
    [
        ptr("alternate_bias", "const float"),
        ptr("bias_select", "const uint8_t"),
        ptr("scores", "float"),
    ]
}

#[derive(Clone, Debug, PartialEq)]
pub struct Router {
    pub experts: u32,
    pub top_k: u32,
    pub capacity: u32,
}

impl Validate for Router {
    fn validate(&self) -> Result<(), Refusal> {
        RowBiasRouterQuery {
            experts: self.experts,
            top_k: self.top_k,
            scaling_factor: 1.0,
        }
        .validate()?;
        if !self.experts.is_multiple_of(32)
            || self.experts > 1024
            || self.top_k > 32
            || self.capacity == 0
            || self.capacity > i32::MAX as u32 / self.top_k
        {
            return Err(Refusal(
                "row-bias router geometry exceeds warp selection limits",
            ));
        }
        Ok(())
    }
}

impl Router {
    pub fn symbol(&self) -> String {
        format!(
            "chop_moe_sqrt_softplus_row_bias_e{}k{}_t{}_v1",
            self.experts, self.top_k, self.capacity
        )
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if rows > self.capacity {
            return Err(Refusal("row-bias router launch exceeds token capacity"));
        }
        let plan = LaunchPlan {
            grid: [self.capacity.div_ceil(8), 1, 1],
            block: [ROW_THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn invariance(&self) -> Invariance {
        if self.validate().is_err() {
            return Invariance::NONE;
        }
        Invariance::any_rows(self.capacity).with_ties(Ties::LowerIndex)
    }

    pub fn entry(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let mut params = vec![ptr("logits", "const float"), ptr("bias", "const float")];
        params.extend(extra_params());
        params.extend([
            ptr("ids", "int32_t"),
            ptr("weights", "float"),
            ptr("live_tokens", "const int32_t"),
            EntryParam::U32 {
                name: "scale_bits".into(),
            },
        ]);
        Ok(EntrySpec::new(
            self.symbol(),
            params,
            format!(
                "chop::moe::local::route_select_rows<{},{},{}>(logits,ids,weights,live_tokens,{});",
                self.experts,
                self.top_k,
                self.capacity,
                score(self.experts)
            ),
            ROW_THREADS,
            1,
        ))
    }
}

/// The existing local-route metadata layout and schedules, with only the
/// scoring policy and its required row-bias arguments changed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Route {
    pub local: LocalRoute,
}

impl Validate for Route {
    fn validate(&self) -> Result<(), Refusal> {
        self.local.validate()?;
        Router {
            experts: self.local.experts,
            top_k: self.local.top_k,
            capacity: self.local.capacity,
        }
        .validate()?;
        self.local
            .capacity
            .checked_mul(self.local.top_k)
            .and_then(|r| {
                self.local
                    .local_experts
                    .checked_mul(self.local.pad - 1)
                    .and_then(|p| r.checked_add(p))
            })
            .filter(|&r| r <= i32::MAX as u32)
            .ok_or(Refusal("row-bias route metadata overflows"))?;
        Ok(())
    }
}

impl Route {
    pub fn prefix(&self) -> &'static str {
        "chop_moe_sqrt_softplus_row_bias_route"
    }

    pub fn entries(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let local = self.local;
        let mut entries = local.entries(self.prefix())?;
        for (&kernel, entry) in local.kernels().iter().zip(&mut entries) {
            let scoring = score(local.experts);
            match kernel {
                RouteKernel::Decode => {
                    entry.params.splice(2..2, extra_params());
                    entry.body = format!(
                        "chop::moe::local::route_decode_scan<{},{},{},{},{}>(logits,ids,weights,counts,offsets,sorted_rows,num_padded_rows,live_tokens,active_experts,rank,{scoring});",
                        local.experts, local.top_k, local.local_experts, local.capacity, local.pad
                    );
                }
                RouteKernel::Select => {
                    entry.params.splice(2..2, extra_params());
                    entry.body = format!(
                        "chop::moe::local::route_select_rows<{},{},{}>(logits,ids,weights,live_tokens,{scoring});",
                        local.experts, local.top_k, local.capacity
                    );
                }
                RouteKernel::Count | RouteKernel::Scatter => {}
            }
        }
        Ok(entries)
    }
}

pub fn cubin_spec(name: &str, routers: &[Router], routes: &[Route]) -> Result<CubinSpec, Refusal> {
    let mut entries = routers
        .iter()
        .map(Router::entry)
        .collect::<Result<Vec<_>, _>>()?;
    for route in routes {
        entries.extend(route.entries()?);
    }
    Ok(super::spec(name.into(), "route_row_bias.cuh", entries))
}
