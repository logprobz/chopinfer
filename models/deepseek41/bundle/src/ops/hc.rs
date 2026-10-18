//! The hyper-connection ops (`model_deepseek41_contract::hc`): each
//! sublayer's lagged mix-collapse, the F32 expand of its output into the
//! streams, and the head's unrounded F32 collapse by the last mixer's
//! `pre`, each prepared for a prefill's rows.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::sinkhorn_hyper_connection::{
    LaggedMixCollapseBuffers, MixScratch, MixerWeights, StreamExpandBuffers,
    WeightedStreamHeadBuffers,
};
use model_deepseek41_contract::hc;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;

/// The role paths of the mixer scratch a backend may list
/// (`OpInfo::bytes`).
const PARTIALS: &str = "scratch.partials";
const TICKETS: &str = "scratch.tickets";

pub(crate) struct HcOps {
    mix: Prepared<LaggedMixCollapseBuffers<View>>,
    expand: Prepared<StreamExpandBuffers<View>>,
    head: Prepared<WeightedStreamHeadBuffers<View>>,
}

/// The mixer scratch the mix-collapse lists, in bytes: its partials,
/// rewritten by every launch, and its tickets, zeroed once at create.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MixScratchBytes {
    pub partials: Option<u64>,
    pub tickets: Option<u64>,
}

/// A step's spans of a lagged mix-collapse: the streams and the sublayer's
/// mixer weights (`fn`, base, scale) it reads with the previous `pre`, and
/// the collapsed row, `pre`, `post` and `comb` it writes.
pub(crate) struct MixViews {
    pub x: DeviceBufferSpanV1,
    pub mixer: [DeviceBufferSpanV1; 3],
    pub pre_in: DeviceBufferSpanV1,
    pub collapsed: DeviceBufferSpanV1,
    pub pre: DeviceBufferSpanV1,
    pub post: DeviceBufferSpanV1,
    pub comb: DeviceBufferSpanV1,
    pub partials: Option<DeviceBufferSpanV1>,
    pub tickets: Option<DeviceBufferSpanV1>,
    pub live_rows: DeviceBufferSpanV1,
}

/// A step's spans of an expand: the streams the sublayer read, its reduced
/// output, the mixer's `post` and `comb`, and the streams it writes.
pub(crate) struct ExpandViews {
    pub x: DeviceBufferSpanV1,
    pub y: DeviceBufferSpanV1,
    pub post: DeviceBufferSpanV1,
    pub comb: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

/// A step's spans of the head's collapse.
pub(crate) struct HeadViews {
    pub x: DeviceBufferSpanV1,
    pub pre: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl HcOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<HcOps, Refused> {
        // Every phase mixes and expands its sublayers and collapses the
        // head's rows.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let g = shape.geometry;
        Ok(HcOps {
            mix: asks.ask("hc.mix_collapse", need, |need| {
                backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
            })?,
            expand: asks.ask("hc.expand", need, |need| {
                backend.sinkhorn_expand(&hc::expand(g, need.capacity), need)
            })?,
            head: asks.ask("hc.head", need, |need| {
                backend.sinkhorn_weighted_head(&hc::head(g, need.capacity), need)
            })?,
        })
    }

    /// The mixer scratch every rank allocates for the mix-collapse.
    pub(crate) fn scratch(&self) -> MixScratchBytes {
        let bytes = |role: &str| {
            self.mix
                .info()
                .bytes
                .iter()
                .find(|(path, _)| path == role)
                .map(|&(_, bytes)| bytes)
        };
        MixScratchBytes {
            partials: bytes(PARTIALS),
            tickets: bytes(TICKETS),
        }
    }

    pub(crate) fn mix(
        &self,
        rows: u32,
        views: &MixViews,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let mut bind = Bind::default();
        let mut view = |span| bind.view(span);
        let buffers = LaggedMixCollapseBuffers {
            x: view(views.x),
            mixer: MixerWeights {
                fn_: view(views.mixer[0]),
                base: view(views.mixer[1]),
                scale: view(views.mixer[2]),
            },
            pre_in: view(views.pre_in),
            collapsed: view(views.collapsed),
            pre: view(views.pre),
            post: view(views.post),
            comb: view(views.comb),
            scratch: MixScratch {
                partials: views.partials.map(&mut view),
                tickets: views.tickets.map(&mut view),
            },
            live_rows: view(views.live_rows),
        };
        bind.launch(&self.mix, &buffers, Counts::rows(rows))
    }

    pub(crate) fn expand(
        &self,
        rows: u32,
        views: &ExpandViews,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let mut bind = Bind::default();
        let mut view = |span| bind.view(span);
        let buffers = StreamExpandBuffers {
            x: view(views.x),
            y: view(views.y),
            post: view(views.post),
            comb: view(views.comb),
            output: view(views.output),
            live_rows: view(views.live_rows),
        };
        bind.launch(&self.expand, &buffers, Counts::rows(rows))
    }

    pub(crate) fn head(
        &self,
        rows: u32,
        views: &HeadViews,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let mut bind = Bind::default();
        let mut view = |span| bind.view(span);
        let buffers = WeightedStreamHeadBuffers {
            x: view(views.x),
            pre: view(views.pre),
            output: view(views.output),
            live_rows: view(views.live_rows),
        };
        bind.launch(&self.head, &buffers, Counts::rows(rows))
    }
}
