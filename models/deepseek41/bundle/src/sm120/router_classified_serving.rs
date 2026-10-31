//! The `router_prepare_fusion` candidate's launch (`router_classified`):
//! one layer's router finish prepare and router producer, at 1 to 32 rows,
//! as the producer that classifies its own rows, with the producer's
//! arguments and bindings followed by the prepare's ticket.
use bundle_runtime::pipeline::PreparedLaunch;
use kernel_core::contract::{Argument, Refusal};
use kernels_sm120::f32_projection::classified_decode;

use super::router_classified;
use crate::ops::Lowered;

/// The prepare's arguments: input, live count, rows, workspace, ticket.
const PREPARE_ARGUMENTS: usize = 5;
/// The producer's arguments: activation, weight, workspace, live count,
/// rows, weight domain.
const PRODUCER_ARGUMENTS: usize = 6;

impl crate::Kernels {
    /// Whether `shape` folds the router finish's prepare into the router
    /// producer at `rows` rows: the candidate option, on a topology that
    /// has the producer, through the 32 rows of the fused finish.
    pub(crate) fn router_prepare_fusion(shape: &crate::shape::Shape, rows: u32) -> bool {
        shape.router_prepare_fusion
            && router_classified::serves(shape.geometry, shape.ranks())
            && rows <= 32
    }

    /// Whether `producer` is the router producer the classifying one
    /// replaces: the backbone router's; the drafter's router keeps its
    /// prepare.
    pub(crate) fn router_classifies(producer: &str) -> bool {
        producer == router_classified::entry().stage_symbol("_decode")
    }

    /// `prepare`, one layer's router finish prepare, and `producer`, the
    /// router's cached decode producer of the same rows, as the producer
    /// that classifies its own rows and resets the prepare's ticket.
    pub(crate) fn classified_router(
        prepare: Vec<Lowered>,
        producer: Lowered,
    ) -> Result<Lowered, Refusal> {
        let [prepare] = <[Lowered; 1]>::try_from(prepare)
            .map_err(|_| Refusal("the classifying router needs the one finish prepare"))?;
        let entry = router_classified::entry();
        let (p, d) = (&prepare.launch, &producer.launch);
        let rows = d.rows;
        let view = |launch: &PreparedLaunch, at: usize| match launch.arguments.0.get(at) {
            Some(Argument::DeviceBuffer(handle)) => launch.bindings.get(*handle).ok(),
            _ => None,
        };
        if !p.symbol.contains("_router_finish_")
            || !p.symbol.ends_with("_prepare_v1")
            || d.symbol != entry.stage_symbol("_decode")
            || p.rows != rows
            || p.arguments.0.len() != PREPARE_ARGUMENTS
            || d.arguments.0.len() != PRODUCER_ARGUMENTS
            || p.arguments.0[2] != Argument::U32(rows)
            || d.arguments.0[4] != Argument::U32(rows)
            || prepare.collective
            || producer.collective
        {
            return Err(Refusal(
                "the classifying router replaces the served finish prepare and router producer",
            ));
        }
        let same_start =
            |a: Option<runtime_bundle_abi::DeviceBufferSpanV1>,
             b: Option<runtime_bundle_abi::DeviceBufferSpanV1>| {
                matches!((a, b), (Some(a), Some(b)) if a.buffer == b.buffer && a.offset == b.offset)
            };
        // Input and activation, live counts, and the workspace whose words
        // the prepare classified into and the producer read.
        if !same_start(view(p, 0), view(d, 0))
            || !same_start(view(p, 1), view(d, 3))
            || !same_start(view(p, 3), view(d, 2))
        {
            return Err(Refusal(
                "the classifying router needs the prepare's rows, live count and workspace",
            ));
        }
        let arrived = view(p, 4).ok_or(Refusal("the finish prepare binds no ticket"))?;
        let mut bindings = d.bindings.clone();
        let mut arguments = d.arguments.clone();
        arguments
            .0
            .push(Argument::DeviceBuffer(bindings.bind(arrived)));
        Ok(Lowered {
            collective: false,
            launch: PreparedLaunch {
                symbol: classified_decode::symbol(&entry),
                rows,
                plan: classified_decode::plan(&entry, rows)?,
                arguments,
                bindings,
            },
        })
    }
}
