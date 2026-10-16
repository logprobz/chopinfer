//! The routed experts' gate/up and down projections of an authenticated
//! expert use, which MiMo's registered launcher launches
//! (`crate::projection`).

use kernel_core::backend::{Backend, Counts, Need, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::gated_mlp::GatedMlpProjectionQuery;
use kernel_core::moe::RouteMetadata;
use kernel_core::moe_experts::{GatedMlpBuffers, GatedMlpScratch, PreparedExperts};
use kernel_core::precision::OperationUse;
use kernel_core::precision::buffers::{ProjectionRouting, QuantizedWeights};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::Bind;
use crate::program_lower::Lowered;

/// One expert projection prepared for each use the component binds it to.
pub struct Experts(Vec<(OperationUse, Box<dyn PreparedExperts>)>);

impl Experts {
    /// `op` on `backend` for each of `uses`, with what each needs.
    pub fn prepare(
        backend: &dyn Backend,
        op: &GatedMlpProjectionQuery,
        uses: &[(OperationUse, Need)],
    ) -> Result<Self, Refused> {
        uses.iter()
            .map(|&(usage, need)| {
                let prepared = backend.gated_mlp_projection(op, usage, need)?;
                if prepared.info().workspace > 0 {
                    return Err(Refused::Rule(
                        "MiMo binds no backend workspace for its ops".into(),
                    ));
                }
                Ok((usage, prepared))
            })
            .collect::<Result<_, Refused>>()
            .map(Self)
    }

    /// The launches of the projection of `rows` token rows at `usage`.
    pub fn launch(
        &self,
        usage: OperationUse,
        rows: u32,
        views: &ExpertViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .0
            .iter()
            .find(|(prepared, _)| *prepared == usage)
            .ok_or(Refusal(
                "MiMo has no prepared expert projection for this use",
            ))?;
        let mut bind = Bind::default();
        let buffers = GatedMlpBuffers {
            activation: bind.view(views.activation),
            activation_scales: None,
            activation_global: None,
            live_tokens: bind.view(views.live_tokens),
            output: bind.view(views.output),
            weights: QuantizedWeights {
                codes: bind.view(views.weight),
                local_scales: Some(bind.view(views.weight_scales)),
                global_scales: None,
            },
            routing: Some(ProjectionRouting {
                weights: bind.view(views.route_weights),
                routes: RouteMetadata {
                    counts: None,
                    offsets: bind.view(views.offsets),
                    sorted_rows: bind.view(views.sorted_rows),
                    padded_rows: bind.view(views.padded_rows),
                    active_experts: Some(bind.view(views.active_experts)),
                },
            }),
            scratch: GatedMlpScratch {
                partial: Some(bind.view(views.partials)),
                output_codes: None,
                output_scales: None,
                unused: bind.view(views.unused),
            },
        };
        Ok(bind.launches(op.launches(&buffers, View::EMPTY, Counts::rows(rows))?))
    }
}

/// The views of an expert projection: its BF16 activation rows, MXFP4
/// weight codes and scales, output rows, the step's routes, the K-split
/// partials, and a live region the entries read for operands they do not
/// use.
pub struct ExpertViews {
    pub activation: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub weight_scales: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub route_weights: DeviceBufferSpanV1,
    pub offsets: DeviceBufferSpanV1,
    pub sorted_rows: DeviceBufferSpanV1,
    pub padded_rows: DeviceBufferSpanV1,
    pub active_experts: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
    pub partials: DeviceBufferSpanV1,
    pub unused: DeviceBufferSpanV1,
}
