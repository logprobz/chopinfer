//! Metadata for branches chosen during model lowering, outside kernel launches.
use super::Lowering;
use crate::{
    ops::Lowered,
    program::{Phase, Stage, Sublayer},
};
use kernel_core::route::{RouteClass, branch};

impl Lowering<'_> {
    pub(super) fn tag_routes(&self, phase: Phase, stage: Stage, launches: &mut [Lowered]) {
        if !kernel_core::route::Metadata::enabled() {
            return;
        }
        if matches!(stage, Stage::Route(_))
            && let Some(moe) = &self.ops.moe
        {
            for item in launches.iter_mut() {
                item.launch
                    .bindings
                    .set_route(moe.router_route(phase, &item.launch.symbol));
            }
        }
        if matches!(stage, Stage::Project(_, "moe.router"))
            && let Some(moe) = &self.ops.moe
            && moe.direct(phase).is_some_and(|op| op.fuses_router())
        {
            for item in launches
                .iter_mut()
                .filter(|item| item.launch.bindings.route().is_none())
            {
                item.launch
                    .bindings
                    .set_route(moe.router_route(phase, &item.launch.symbol));
            }
        }
        if !phase.prefill() {
            return;
        }
        if matches!(stage, Stage::Project(_, "moe.router")) && self.router_in_norm(phase) {
            for item in launches.iter_mut() {
                let tag = branch(
                    &item.launch.symbol,
                    RouteClass::Fast,
                    "f32_projection",
                    "router-in-norm",
                    [crate::Kernels::router_in_norm_route_rows()],
                );
                item.launch.bindings.set_route(Some(tag));
            }
        }
        if self.ops.norm_prefill.is_some()
            && matches!(stage, Stage::InputNorm(..) | Stage::QuantizeInput(..))
        {
            let selected = crate::ops::norm_prefill::qualified(self.shape.geometry, phase);
            let admitted = (64..=2624)
                .step_by(128)
                .chain([8192])
                .map(|r| r..=r)
                .collect::<Vec<_>>();
            for item in launches.iter_mut() {
                // Widening is retained by both paths and is not a displaced norm.
                if !selected && item.launch.symbol != crate::Kernels::norm_route_symbol(self.shape)
                {
                    continue;
                }
                let tag = branch(
                    &item.launch.symbol,
                    if selected {
                        RouteClass::Fast
                    } else {
                        RouteClass::Baseline
                    },
                    "prefill_norm",
                    if selected { "fused" } else { "unfused" },
                    admitted.clone(),
                );
                item.launch.bindings.set_route(Some(tag));
            }
        }
        if !self.prefill_hc_fusion {
            return;
        }
        let boundary = match stage {
            Stage::HcExpand(layer, sublayer) => Some((layer, sublayer)),
            Stage::HcMix(layer, Sublayer::Mlp) => Some((layer, Sublayer::Attention)),
            Stage::HcMix(layer, Sublayer::Attention) => {
                layer.checked_sub(1).map(|l| (l, Sublayer::Mlp))
            }
            _ => None,
        };
        let Some((layer, sublayer)) = boundary else {
            return;
        };
        if super::hc_expand_mix::successor(self.shape.geometry, layer, sublayer).is_none() {
            return;
        }
        let Some(admitted) = crate::Kernels::prefill_hc_route_rows(self.shape) else {
            return;
        };
        let selected = self
            .ops
            .hc
            .as_ref()
            .is_some_and(|hc| hc.has_prefill_expand_mix(phase));
        for item in launches {
            let tag = branch(
                &item.launch.symbol,
                if selected {
                    RouteClass::Fast
                } else {
                    RouteClass::Baseline
                },
                "hc_prefill_expand_mix",
                if selected { "fused" } else { "separate" },
                admitted.clone(),
            );
            item.launch.bindings.set_route(Some(tag));
        }
    }
}
