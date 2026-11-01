//! The hyper-connection ops (`model_deepseek41_contract::hc`): each
//! sublayer's lagged mix-collapse, the F32 expand of its output into the
//! streams, and the head's unrounded F32 collapse by the last mixer's
//! `pre`, each prepared for a prefill's rows.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::sinkhorn_expand_mix::{
    SinkhornExpandMixBuffers, SinkhornExpandMixQuery, SinkhornRoundExpandMixBuffers,
    SinkhornRoundNormBuffers, SinkhornRoundNormQuery,
};
use kernel_core::sinkhorn_hyper_connection::{
    LaggedMixCollapseBuffers, MixScratch, MixerWeights, StreamExpandBuffers,
    WeightedStreamHeadBuffers,
};
use model_deepseek41_contract::dspark_options::SMALL_DRAFT_ROWS;
use model_deepseek41_contract::hc;
use model_deepseek41_contract::serving::{DECODE_BATCH, PREFILL_TOKENS};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::{program::Phase, shape::Shape};

/// The role paths of the mixer scratch a backend may list
/// (`OpInfo::bytes`).
const PARTIALS: &str = "scratch.partials";
const TICKETS: &str = "scratch.tickets";

fn baseline_finish_selected(phase: Phase, finish_rows: u32) -> bool {
    matches!(
        phase,
        Phase::Decode(1 | 8 | 16 | 32)
            | Phase::Verify {
                rows: 4 | 6 | 8 | 12 | 16 | 32,
                ..
            }
    ) || baseline_wide_selected(phase, finish_rows)
}

fn baseline_wide_selected(phase: Phase, finish_rows: u32) -> bool {
    matches!(phase, Phase::Verify { rows: 33..=64, .. }) && phase.rows() <= finish_rows
}

fn baseline_decode_selected(phase: Phase) -> bool {
    matches!(
        phase,
        Phase::Decode(1 | 8)
            | Phase::Verify {
                rows: 4 | 6 | 8 | 12 | 16 | 32,
                ..
            }
    )
}

fn baseline_draft_selected(phase: Phase) -> bool {
    phase.rows() > DECODE_BATCH && phase.rows() <= SMALL_DRAFT_ROWS
}

fn table_selected(
    g: &model_deepseek41_contract::geometry::Geometry,
    phase: Phase,
    ops: [&str; 3],
    capacity: u32,
) -> bool {
    use model_deepseek41_contract::tuning::{self, Family};
    let op = match phase {
        Phase::Decode(_) => ops[0],
        Phase::Verify { .. } => ops[1],
        Phase::Prefill { .. } => ops[2],
    };
    tuning::select(
        g,
        Family::Hc,
        op,
        "f32",
        &[("groups", g.shape.hc.streams), ("width", g.shape.hidden)],
        capacity,
        phase.rows(),
    )
    .is_some()
}
fn finish_selected(
    g: &model_deepseek41_contract::geometry::Geometry,
    phase: Phase,
    capacity: u32,
) -> bool {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return baseline_finish_selected(phase, capacity);
    }
    table_selected(
        g,
        phase,
        [
            "hc.finish_admit.decode",
            "hc.finish_admit.verify",
            "hc.finish_admit.prefill",
        ],
        capacity,
    )
}
/// The drafter's verify rows the drafter table admits to the finish route
/// (`hc.draft_admit.verify`, the `dspark_hc_draft_rows` candidate), within
/// the prepared finish rows.
fn draft_finish_selected(
    g: &model_deepseek41_contract::geometry::Geometry,
    phase: Phase,
    finish_rows: u32,
) -> bool {
    g.id == model_deepseek41_contract::geometry::FLASH.id
        && matches!(phase, Phase::Verify { .. })
        && phase.rows() <= finish_rows
        && model_deepseek41_contract::tuning::select(
            g,
            model_deepseek41_contract::tuning::Family::Hc,
            "hc.draft_admit.verify",
            "f32",
            &[("groups", g.shape.hc.streams), ("width", g.shape.hidden)],
            model_deepseek41_contract::verify_bounds::HC_FINISH,
            phase.rows(),
        )
        .is_some()
}
fn wide_selected(
    g: &model_deepseek41_contract::geometry::Geometry,
    phase: Phase,
    capacity: u32,
) -> bool {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return baseline_wide_selected(phase, capacity);
    }
    table_selected(
        g,
        phase,
        [
            "hc.wide_mix.decode",
            "hc.wide_mix.verify",
            "hc.wide_mix.prefill",
        ],
        capacity,
    )
}
fn decode_selected(g: &model_deepseek41_contract::geometry::Geometry, phase: Phase) -> bool {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return baseline_decode_selected(phase);
    }
    table_selected(
        g,
        phase,
        [
            "hc.decode_mix.decode",
            "hc.decode_mix.verify",
            "hc.decode_mix.prefill",
        ],
        32,
    )
}
fn draft_selected(g: &model_deepseek41_contract::geometry::Geometry, phase: Phase) -> bool {
    if g.id != model_deepseek41_contract::geometry::FLASH.id {
        return baseline_draft_selected(phase);
    }
    table_selected(
        g,
        phase,
        [
            "hc.draft_mix.decode",
            "hc.draft_mix.verify",
            "hc.draft_mix.prefill",
        ],
        SMALL_DRAFT_ROWS,
    )
}

/// One wide verify bucket's fused finish norm (`Shape::wide_verify_rows`,
/// the `wide_verify` candidate): the boundaries of verify steps above the
/// decode route up to `rows` rows (`sm120::hc_wide`). A mixer that follows
/// no fused boundary keeps the one-CTA entry: the bounded split would take
/// three times the decode route's mixer partials for three launches per
/// rank and step.
struct Wide {
    rows: u32,
    finish: Prepared<SinkhornRoundNormBuffers<View>>,
}

pub(crate) struct HcOps {
    geometry: &'static model_deepseek41_contract::geometry::Geometry,
    mix: Prepared<LaggedMixCollapseBuffers<View>>,
    prefill_mix: Prepared<LaggedMixCollapseBuffers<View>>,
    decode_mix: Prepared<LaggedMixCollapseBuffers<View>>,
    /// Decode and verify rows above `DECODE_BATCH` up to a drafter
    /// forward's small-row limit (`SMALL_DRAFT_ROWS`), on the bounded split.
    draft_mix: Prepared<LaggedMixCollapseBuffers<View>>,
    verify_mix: Option<Prepared<LaggedMixCollapseBuffers<View>>>,
    finish_rows: u32,
    /// A drafter's operations with the `dspark_hc_draft_rows` candidate,
    /// whose finish route also admits the drafter table's rows.
    draft_rows: bool,
    expand: Prepared<StreamExpandBuffers<View>>,
    head: Prepared<WeightedStreamHeadBuffers<View>>,
    fused: Option<Prepared<SinkhornExpandMixBuffers<View>>>,
    /// The measured prefill rows' expansion and next mixer in one launch.
    prefill_fused: Option<Prepared<SinkhornExpandMixBuffers<View>>>,
    round_fused: Option<Prepared<SinkhornRoundExpandMixBuffers<View>>>,
    finish_norm: Option<Prepared<SinkhornRoundNormBuffers<View>>>,
    /// The target's wide verify buckets, narrowest first.
    wide: Vec<Wide>,
}

/// The mixer scratch the mix-collapse lists, in bytes: its partials,
/// rewritten by every launch, and its tickets, zeroed once at create.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct MixScratchBytes {
    pub partials: Option<u64>,
    pub tickets: Option<u64>,
}

impl MixScratchBytes {
    /// The most of each role `self` and `other` list.
    pub(crate) fn most(self, other: Self) -> Self {
        Self {
            partials: self.partials.max(other.partials),
            tickets: self.tickets.max(other.tickets),
        }
    }
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
    /// `target` prepares the wide verify buckets, which no drafter forward
    /// reaches.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
        target: bool,
    ) -> Result<HcOps, Refused> {
        // Every phase mixes and expands its sublayers and collapses the
        // head's rows.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let g = shape.geometry;
        let finish_rows =
            shape.verify_rows_within(model_deepseek41_contract::verify_bounds::HC_FINISH);
        Ok(HcOps {
            geometry: g,
            finish_rows,
            draft_rows: !target && shape.dspark_hc_draft_rows,
            wide: if target {
                Self::prepare_wide(shape, asks)?
            } else {
                Vec::new()
            },
            finish_norm: {
                let capacity = finish_rows;
                let need = exact::need(capacity, exact::every_phase(capacity));
                let query = SinkhornRoundNormQuery {
                    mixer: SinkhornExpandMixQuery { mix: hc::mix(g) },
                    norm: model_deepseek41_contract::rows::norm_quantizers(g)[0],
                };
                match asks.ask("hc.round_expand_mix_norm", need, |need| {
                    backend.sinkhorn_round_expand_mix_norm(&query, need)
                }) {
                    Ok(op) => Some(op),
                    Err(Refused::Unsupported) => None,
                    Err(e) => return Err(e),
                }
            },
            mix: asks.ask("hc.mix_collapse", need, |need| {
                backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
            })?,
            prefill_mix: asks.ask("hc.prefill_mix", need, |need| {
                backend.sinkhorn_lagged_prefill(&hc::mix_collapse(g, need.capacity), need)
            })?,
            decode_mix: {
                let need = exact::need(32, exact::every_phase(32));
                asks.ask("hc.decode_mix", need, |need| {
                    backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
                })?
            },
            draft_mix: {
                let need = exact::need(SMALL_DRAFT_ROWS, exact::every_phase(SMALL_DRAFT_ROWS));
                asks.ask("hc.draft_mix", need, |need| {
                    backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
                })?
            },
            verify_mix: if finish_rows > DECODE_BATCH {
                let need = exact::need(finish_rows, exact::every_phase(finish_rows));
                Some(asks.ask("hc.verify_mix", need, |need| {
                    backend.sinkhorn_lagged_mix_collapse(&hc::mix_collapse(g, need.capacity), need)
                })?)
            } else {
                None
            },
            expand: asks.ask("hc.expand", need, |need| {
                backend.sinkhorn_expand(&hc::expand(g, need.capacity), need)
            })?,
            head: asks.ask("hc.head", need, |need| {
                backend.sinkhorn_weighted_head(&hc::head(g, need.capacity), need)
            })?,
            fused: {
                let capacity = model_deepseek41_contract::serving::DECODE_BATCH;
                let need = exact::need(capacity, exact::every_phase(capacity));
                match asks.ask("hc.expand_mix", need, |need| {
                    backend.sinkhorn_expand_mix(&SinkhornExpandMixQuery { mix: hc::mix(g) }, need)
                }) {
                    Ok(op) => Some(op),
                    Err(Refused::Unsupported) => None,
                    Err(e) => return Err(e),
                }
            },
            prefill_fused: match asks.ask("hc.prefill_expand_mix", need, |need| {
                backend.sinkhorn_expand_mix(&SinkhornExpandMixQuery { mix: hc::mix(g) }, need)
            }) {
                Ok(op) => Some(op),
                Err(Refused::Unsupported) => None,
                Err(e) => return Err(e),
            },
            round_fused: {
                let capacity = model_deepseek41_contract::serving::DECODE_BATCH;
                let need = exact::need(capacity, exact::every_phase(capacity));
                match asks.ask("hc.round_expand_mix", need, |need| {
                    backend.sinkhorn_round_expand_mix(
                        &SinkhornExpandMixQuery { mix: hc::mix(g) },
                        need,
                    )
                }) {
                    Ok(op) => Some(op),
                    Err(Refused::Unsupported) => None,
                    Err(e) => return Err(e),
                }
            },
        })
    }

    /// The fused finish norm of each wide verify bucket the instance
    /// reaches, from the wide boundary's own backend
    /// (`Kernels::wide_hc_backend`).
    fn prepare_wide(shape: &Shape, asks: &Asks) -> Result<Vec<Wide>, Refused> {
        let buckets = shape.wide_verify_rows();
        let Some(&widest) = buckets.last() else {
            return Ok(Vec::new());
        };
        let g = shape.geometry;
        let wide_backend = crate::Kernels::wide_hc_backend(g, widest);
        let query = SinkhornRoundNormQuery {
            mixer: SinkhornExpandMixQuery { mix: hc::mix(g) },
            norm: model_deepseek41_contract::rows::norm_quantizers(g)[0],
        };
        buckets
            .into_iter()
            .map(|rows| {
                let need = exact::need(rows, exact::every_phase(rows));
                Ok(Wide {
                    rows,
                    finish: asks.ask(
                        format!("hc.round_expand_mix_norm wide rows {rows}"),
                        need,
                        |need| wide_backend.sinkhorn_round_expand_mix_norm(&query, need),
                    )?,
                })
            })
            .collect()
    }

    /// The wide verify bucket that serves `phase`: a verify step above the
    /// decode route's rows, in the narrowest bucket that holds it.
    fn wide_at(&self, phase: Phase) -> Option<&Wide> {
        if !matches!(phase, Phase::Verify { .. }) || phase.rows() <= self.finish_rows {
            return None;
        }
        self.wide.iter().find(|wide| phase.rows() <= wide.rows)
    }

    /// The wide verify finish norm that serves `phase`, where
    /// `hc.finish_admit.verify` admits its rows.
    fn wide_finish(&self, phase: Phase) -> Option<&Prepared<SinkhornRoundNormBuffers<View>>> {
        self.wide_at(phase)
            .filter(|_| crate::Kernels::wide_hc_admitted("hc.finish_admit.verify", phase.rows()))
            .map(|wide| &wide.finish)
    }

    pub(crate) fn has_expand_mix(&self) -> bool {
        self.fused.is_some()
    }
    pub(crate) fn expand_mix(
        &self,
        rows: u32,
        views: SinkhornExpandMixBuffers<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let op = self.fused.as_ref().ok_or(kernel_core::contract::Refusal(
            "fused expansion mixer unavailable",
        ))?;
        let mut bind = Bind::default();
        let b = SinkhornExpandMixBuffers {
            x: bind.view(views.x),
            y: bind.view(views.y),
            post_in: bind.view(views.post_in),
            comb_in: bind.view(views.comb_in),
            expanded: bind.view(views.expanded),
            fn_: bind.view(views.fn_),
            base: bind.view(views.base),
            scale: bind.view(views.scale),
            pre_in: bind.view(views.pre_in),
            collapsed: bind.view(views.collapsed),
            pre: bind.view(views.pre),
            post: bind.view(views.post),
            comb: bind.view(views.comb),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch_within(op, &b, workspace, Counts::rows(rows))
    }

    /// The prefill rows the backend's expansion mixer launches: it refuses
    /// every row count it has not measured, so ask it with distinct views.
    pub(crate) fn has_prefill_expand_mix(&self, phase: Phase) -> bool {
        let Some(op) = self.prefill_fused.as_ref().filter(|_| phase.prefill()) else {
            return false;
        };
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let probe = SinkhornExpandMixBuffers {
            x: view(1),
            y: view(2),
            post_in: view(3),
            comb_in: view(4),
            expanded: view(5),
            fn_: view(6),
            base: view(7),
            scale: view(8),
            pre_in: view(9),
            collapsed: view(10),
            pre: view(11),
            post: view(12),
            comb: view(13),
            live_rows: view(14),
        };
        op.launches(&probe, View::EMPTY, Counts::rows(phase.rows()))
            .is_ok()
    }
    pub(crate) fn prefill_expand_mix(
        &self,
        rows: u32,
        views: SinkhornExpandMixBuffers<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let op = self
            .prefill_fused
            .as_ref()
            .ok_or(kernel_core::contract::Refusal(
                "prefill expansion mixer unavailable",
            ))?;
        let mut bind = Bind::default();
        let b = SinkhornExpandMixBuffers {
            x: bind.view(views.x),
            y: bind.view(views.y),
            post_in: bind.view(views.post_in),
            comb_in: bind.view(views.comb_in),
            expanded: bind.view(views.expanded),
            fn_: bind.view(views.fn_),
            base: bind.view(views.base),
            scale: bind.view(views.scale),
            pre_in: bind.view(views.pre_in),
            collapsed: bind.view(views.collapsed),
            pre: bind.view(views.pre),
            post: bind.view(views.post),
            comb: bind.view(views.comb),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(op, &b, Counts::rows(rows))
    }

    pub(crate) fn has_round_expand_mix(&self) -> bool {
        self.round_fused.is_some()
    }
    pub(crate) fn round_expand_mix(
        &self,
        rows: u32,
        views: SinkhornRoundExpandMixBuffers<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        let op = self
            .round_fused
            .as_ref()
            .ok_or(kernel_core::contract::Refusal(
                "fused expansion mixer unavailable",
            ))?;
        let mut bind = Bind::default();
        let b = SinkhornRoundExpandMixBuffers {
            x: bind.view(views.x),
            sum: bind.view(views.sum),
            rounded: bind.view(views.rounded),
            post_in: bind.view(views.post_in),
            comb_in: bind.view(views.comb_in),
            expanded: bind.view(views.expanded),
            fn_: bind.view(views.fn_),
            base: bind.view(views.base),
            scale: bind.view(views.scale),
            pre_in: bind.view(views.pre_in),
            collapsed: bind.view(views.collapsed),
            pre: bind.view(views.pre),
            post: bind.view(views.post),
            comb: bind.view(views.comb),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch_within(op, &b, workspace, Counts::rows(rows))
    }

    // TP4, 250 W, 2026-10-21, 16-row boundaries of real captured rows
    // (hc_tail16, hc16-r1): at Decode-16 the tail is bitwise against the
    // rounded expansion mixer and input norm it replaces, memcheck clean,
    // and saves 1.07..1.12 ms per step cold (0.75..0.85 hot), 77 of 77 sites.
    pub(crate) fn has_finish_norm(&self, phase: Phase) -> bool {
        (finish_selected(self.geometry, phase, self.finish_rows)
            || self.draft_rows && draft_finish_selected(self.geometry, phase, self.finish_rows))
            && self.finish_norm.is_some()
            || self.wide_finish(phase).is_some()
    }

    pub(crate) fn round_expand_mix_norm(
        &self,
        phase: Phase,
        views: SinkhornRoundNormBuffers<DeviceBufferSpanV1>,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, kernel_core::contract::Refusal> {
        use kernel_core::allocation::Traversal;
        if !self.has_finish_norm(phase) {
            return Err(kernel_core::contract::Refusal(
                "unqualified HC finish norm phase",
            ));
        }
        let mut bind = Bind::default();
        // Shared collapsed/live roles and the allowed post/comb aliases
        // must retain one logical view when composing the two validators.
        let mut seen = Vec::new();
        let b = views.map(|span| {
            if let Some((_, view)) = seen.iter().find(|(old, _)| *old == span) {
                return *view;
            }
            let view = bind.view(span);
            seen.push((span, view));
            view
        });
        let op = self
            .wide_finish(phase)
            .or(self.finish_norm.as_ref())
            .unwrap();
        bind.launch_within(op, &b, workspace, Counts::rows(phase.rows()))
    }

    /// The mixer scratch every rank allocates for the mix-collapse.
    pub(crate) fn scratch(&self) -> MixScratchBytes {
        let mut scratch = self.scratch_of(&[&self.mix]);
        if let Some(op) = &self.verify_mix {
            scratch = scratch.most(self.scratch_of(&[op]));
        }
        scratch
    }

    /// The mixer scratch a drafter's ops bind: [`Self::scratch`] or the
    /// draft-rows mixer's, whichever is more.
    pub(crate) fn draft_scratch(&self) -> MixScratchBytes {
        self.scratch().most(self.scratch_of(&[&self.draft_mix]))
    }

    fn scratch_of(&self, mixers: &[&Prepared<LaggedMixCollapseBuffers<View>>]) -> MixScratchBytes {
        let bytes = |role: &str| {
            mixers
                .iter()
                .flat_map(|op| op.info().bytes.iter())
                .filter(|(path, _)| path == role)
                .map(|&(_, bytes)| bytes)
                .max()
        };
        // A one-launch finish/norm boundary elects its tail with the same
        // zeroed tickets; every use runs on the main stream and leaves them
        // zeroed, so one allocation covers both.
        let tail = self.finish_norm.as_ref().and_then(|op| {
            op.info()
                .bytes
                .iter()
                .find(|(path, _)| path == TICKETS)
                .map(|&(_, bytes)| bytes)
        });
        MixScratchBytes {
            partials: bytes(PARTIALS),
            tickets: bytes(TICKETS).max(tail),
        }
    }

    fn mix_for_phase(&self, phase: Phase) -> &Prepared<LaggedMixCollapseBuffers<View>> {
        if phase.prefill() {
            &self.prefill_mix
        } else if wide_selected(self.geometry, phase, self.finish_rows)
            && let Some(op) = &self.verify_mix
        {
            op
        } else if decode_selected(self.geometry, phase) {
            &self.decode_mix
        } else if draft_selected(self.geometry, phase) {
            &self.draft_mix
        } else {
            &self.mix
        }
    }

    pub(crate) fn mix(
        &self,
        phase: Phase,
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
        bind.launch(
            self.mix_for_phase(phase),
            &buffers,
            Counts::rows(phase.rows()),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    #[test]
    fn prefill_mixer_selects_only_measured_buckets_and_keeps_operands() {
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let buffers = LaggedMixCollapseBuffers {
            x: view(1),
            mixer: MixerWeights {
                fn_: view(2),
                base: view(3),
                scale: view(4),
            },
            pre_in: view(5),
            collapsed: view(6),
            pre: view(7),
            post: view(8),
            comb: view(9),
            scratch: MixScratch {
                partials: Some(view(11)),
                tickets: Some(view(12)),
            },
            live_rows: View {
                bytes: 4,
                ..view(10)
            },
        };
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default(), true).unwrap();
            for rows in [
                33, 63, 64, 65, 191, 192, 193, 320, 2239, 2240, 2241, 2495, 2496, 2497, 2623, 2624,
                2625, 8150, 8191, 8192,
            ] {
                let old = ops
                    .mix
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                for phase in [Phase::Prefill { rows }, Phase::Verify { rows, items: 1 }] {
                    let new = ops
                        .mix_for_phase(phase)
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    let chains = if phase.prefill() {
                        match rows {
                            64 | 192 => Some(8),
                            2240 | 2496 | 2624 | 8192 => Some(4),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if let Some(chains) = chains {
                        assert_eq!(new.len(), 1);
                        assert_eq!(new[0].arguments, old[0].arguments);
                        assert_eq!(new[0].rows, old[0].rows);
                        assert_eq!(
                            new[0].symbol,
                            format!(
                                "chop_sinkhorn_hc_deepseek41_prefill_mix_pipeline_r8_c{chains}_v1"
                            )
                        );
                        assert_eq!(new[0].plan.grid, [rows.div_ceil(8), 1, 1]);
                        assert_eq!(new[0].plan.block, [256, 1, 1]);
                        assert_eq!(new[0].plan.dynamic_shared_bytes, 0);
                    } else if !phase.prefill() && rows <= SMALL_DRAFT_ROWS {
                        // A drafter forward's rows take the bounded split.
                        assert_eq!(new.len(), 1);
                        assert!(new[0].symbol.ends_with("_lagged_mix_collapse_split_v1"));
                        assert_eq!(new[0].plan.grid[1], rows);
                        assert_eq!(new[0].arguments.0.len(), old[0].arguments.0.len() + 2);
                    } else {
                        assert_eq!(new, old, "{phase:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn prefill_expand_mix_serves_only_measured_flash_rows() {
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let buffers = SinkhornExpandMixBuffers {
            x: view(1),
            y: view(2),
            post_in: view(3),
            comb_in: view(4),
            expanded: view(5),
            fn_: view(6),
            base: view(7),
            scale: view(8),
            pre_in: view(9),
            collapsed: view(10),
            pre: view(11),
            post: view(12),
            comb: view(13),
            live_rows: View {
                bytes: 4,
                ..view(14)
            },
        };
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default(), true).unwrap();
            let op = ops.prefill_fused.as_ref().unwrap();
            assert_eq!(op.info().workspace, 0);
            for rows in [1, 63, 64, 65, 192, 320, 2496, 4096, 8191, 8192] {
                let tile = match rows {
                    64 | 320 => Some("r4_c4"),
                    8192 => Some("r8_c8"),
                    _ => None,
                };
                for phase in [
                    Phase::Prefill { rows },
                    Phase::Verify { rows, items: 1 },
                    Phase::Decode(rows),
                ] {
                    assert_eq!(
                        ops.has_prefill_expand_mix(phase),
                        phase.prefill() && tile.is_some()
                    );
                }
                let launches = op.launches(&buffers, View::EMPTY, Counts::rows(rows));
                let Some(tile) = tile else {
                    assert!(launches.is_err(), "{rows}");
                    continue;
                };
                let launches = launches.unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(
                    launches[0].symbol,
                    format!("chop_deepseek41_hc_prefill_expand_mix_{tile}_v1")
                );
                assert_eq!(launches[0].arguments.0.len(), 14);
                assert_eq!(launches[0].plan.block, [256, 1, 1]);
            }
        }
    }

    #[test]
    fn verify_tail_selection_does_not_change_standalone_producer_buckets() {
        use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops = HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default(), true).unwrap();
        let tail = ops.finish_norm.as_ref().unwrap();
        let spans: std::collections::BTreeMap<_, _> = tail
            .info()
            .bytes
            .iter()
            .enumerate()
            .map(|(i, (name, bytes))| {
                (
                    name.as_str(),
                    View {
                        handle: i as u64 + 1,
                        bytes: *bytes,
                    },
                )
            })
            .collect();
        let v = |name: &str| spans[name];
        let hc = SinkhornRoundExpandMixBuffers {
            x: v("hc.x"),
            sum: v("hc.sum"),
            rounded: v("hc.rounded"),
            post_in: v("hc.post_in"),
            comb_in: v("hc.comb_in"),
            expanded: v("hc.expanded"),
            fn_: v("hc.fn_"),
            base: v("hc.base"),
            scale: v("hc.scale"),
            pre_in: v("hc.pre_in"),
            collapsed: v("hc.collapsed"),
            pre: v("hc.pre"),
            post: v("hc.post"),
            comb: v("hc.comb"),
            live_rows: v("hc.live_rows"),
        };
        let workspace = View {
            handle: 100,
            bytes: tail.info().workspace,
        };
        for rows in [1, 2, 4, 6, 8, 12, 16, 32] {
            let standalone = ops
                .round_fused
                .as_ref()
                .unwrap()
                .launches(&hc, workspace, Counts::rows(rows))
                .unwrap();
            // TP4 serves the chain at rows 1-8 (`hc_round_chain::register`).
            let expected = match rows {
                1..=4 => "chop_deepseek41_hc_round_chain_w8_v1",
                5..=8 => "chop_deepseek41_hc_round_chain_w4_v1",
                _ => "chop_deepseek41_hc_round_expand_mix_partials_v1",
            };
            assert_eq!(standalone[0].symbol, expected);
            for (widened, tickets) in [
                None,
                Some(View {
                    handle: 101,
                    bytes: 32 * 5120 * 4,
                }),
            ]
            .into_iter()
            .flat_map(|w| {
                [
                    (w, None),
                    (
                        w,
                        Some(View {
                            handle: 102,
                            bytes: 32 * 4,
                        }),
                    ),
                ]
            }) {
                let buffers = SinkhornRoundNormBuffers {
                    hc,
                    norm: RmsNormQuantizeBuffers {
                        input: hc.collapsed,
                        weight: v("norm.weight"),
                        normed: v("norm.normed"),
                        widened,
                        codes: v("norm.codes"),
                        scales: v("norm.scales"),
                        live_rows: hc.live_rows,
                        status: v("norm.status"),
                        status_row_offset: 0,
                    },
                    tickets,
                };
                // No bucket needs the tickets (the one-launch boundary's).
                let chain = tail
                    .launches(&buffers, workspace, Counts::rows(rows))
                    .unwrap();
                // Measured verify buckets keep the producer and run the tail
                // on two CTAs per row with the tail's own arguments.
                if let Some(&(_, warps)) = crate::sm120::hc_finish_norm::twin_buckets()
                    .iter()
                    .find(|(bucket, _)| *bucket == rows)
                {
                    let twin = crate::sm120::hc_finish_twin::served();
                    assert_eq!(chain.len(), 2);
                    assert!(chain[0].symbol.ends_with("hc_round_chain_w4_v1"));
                    assert_eq!(chain[1].symbol, twin.symbol(warps));
                    assert_eq!(chain[1].plan, twin.plan(rows, rows, warps).unwrap());
                    assert_eq!(chain[1].arguments.0.len(), 18);
                    continue;
                }
                // The column pair's measured buckets run the column-block
                // producer and split finish over both launches' own arguments.
                if let Some(b) = crate::sm120::hc_cols::decode_buckets()
                    .iter()
                    .find(|b| b.rows == rows)
                {
                    let (producer, finish) = (
                        crate::sm120::hc_cols::producer(),
                        crate::sm120::hc_cols::finish(),
                    );
                    assert_eq!(chain.len(), 2);
                    assert_eq!(chain[0].symbol, producer.symbol(b.tile, b.transposed));
                    assert_eq!(
                        chain[0].plan,
                        producer.plan(rows, rows, b.tile, b.transposed).unwrap()
                    );
                    assert_eq!(chain[0].arguments.0.len(), 9);
                    assert_eq!(chain[1].symbol, finish.symbol(b.warps));
                    assert_eq!(chain[1].plan, finish.plan(rows, rows, b.warps).unwrap());
                    assert_eq!(chain[1].arguments.0.len(), 18);
                    continue;
                }
                if matches!(rows, 4 | 6 | 12) {
                    assert_eq!(chain.len(), 2);
                    assert!(chain[1].symbol.ends_with("hc_finish_norm_w32_v1"));
                    assert!(chain[0].symbol.ends_with(match rows {
                        4 => "hc_round_chain_w8_v1",
                        _ => "hc_round_chain_w4_v1",
                    }));
                } else {
                    // The tail keeps its own measured chain buckets
                    // (`hc_finish_norm::register_pair`), so the standalone
                    // small-row chain does not reach it.
                    assert_eq!(chain.len(), 3);
                    assert_eq!(
                        chain[0].symbol,
                        "chop_deepseek41_hc_round_expand_mix_partials_v1"
                    );
                }
            }
        }
    }
    /// With the `wide_verify` option at the package's 192 verify rows,
    /// verify steps of 65 to 192 rows take the column-block pair that
    /// serves 33 to 64 rows, over the arguments a 64-row step binds. Every
    /// mixer, every step of at most 64 rows, every prefill and the
    /// drafter's ops keep their launches and the mixer scratch its bytes,
    /// and without the option every wider step keeps the unfused boundary.
    #[test]
    fn wide_verify_buckets_take_the_decode_boundary_above_64_rows() {
        use kernel_core::rmsnorm_quantize::RmsNormQuantizeBuffers;
        let view = |handle| View {
            handle,
            bytes: u64::MAX / 2,
        };
        let mixer = LaggedMixCollapseBuffers {
            x: view(1),
            mixer: MixerWeights {
                fn_: view(2),
                base: view(3),
                scale: view(4),
            },
            pre_in: view(5),
            collapsed: view(6),
            pre: view(7),
            post: view(8),
            comb: view(9),
            scratch: MixScratch {
                partials: Some(view(11)),
                tickets: Some(view(12)),
            },
            live_rows: View {
                bytes: 4,
                ..view(10)
            },
        };
        let hc = SinkhornRoundExpandMixBuffers {
            x: view(21),
            sum: view(22),
            rounded: view(23),
            post_in: view(24),
            comb_in: view(25),
            expanded: view(26),
            fn_: view(27),
            base: view(28),
            scale: view(29),
            pre_in: view(30),
            collapsed: view(31),
            pre: view(32),
            post: view(24),
            comb: view(25),
            live_rows: View {
                bytes: 4,
                ..view(33)
            },
        };
        let finish = SinkhornRoundNormBuffers {
            hc,
            norm: RmsNormQuantizeBuffers {
                input: hc.collapsed,
                weight: view(41),
                normed: view(42),
                widened: Some(view(43)),
                codes: view(44),
                scales: view(45),
                live_rows: hc.live_rows,
                status: view(46),
                status_row_offset: 0,
            },
            tickets: None,
        };
        let workspace = view(50);
        let shape = Shape::new(&FLASH, Ranks::TP4)
            .unwrap()
            .with_verify_rows(192);
        let wide_shape = shape.with_wide_verify(true);
        let (kernels, wide_kernels) = (
            crate::Kernels::new(&shape).unwrap(),
            crate::Kernels::new(&wide_shape).unwrap(),
        );
        let prepare = |kernels: &crate::Kernels, shape: &Shape, target| {
            HcOps::prepare(kernels.backend.as_ref(), shape, &Asks::default(), target).unwrap()
        };
        let off = prepare(&kernels, &shape, true);
        let on = prepare(&wide_kernels, &wide_shape, true);
        let drafter = prepare(&wide_kernels, &wide_shape, false);
        assert!(off.wide.is_empty() && drafter.wide.is_empty());
        assert_eq!(
            on.wide.iter().map(|w| w.rows).collect::<Vec<_>>(),
            [128, 192]
        );
        assert_eq!(off.scratch().partials, Some(64 * 15360));
        assert_eq!(on.scratch(), off.scratch());
        assert_eq!(drafter.scratch(), off.scratch());
        let mix_at = |ops: &HcOps, phase: Phase| {
            ops.mix_for_phase(phase)
                .launches(&mixer, View::EMPTY, Counts::rows(phase.rows()))
                .unwrap()
        };
        let finish_at = |ops: &HcOps, phase: Phase| {
            ops.has_finish_norm(phase).then(|| {
                ops.wide_finish(phase)
                    .or(ops.finish_norm.as_ref())
                    .unwrap()
                    .launches(&finish, workspace, Counts::rows(phase.rows()))
                    .unwrap()
            })
        };
        let at_64 = Phase::Verify { rows: 64, items: 1 };
        let finish_64 = finish_at(&on, at_64).unwrap();
        let [producer, split] = [&finish_64[0].symbol, &finish_64[1].symbol];
        assert_eq!(producer, "chop_deepseek41_hc_round_cols_r4_tx_v1");
        assert_eq!(split, "chop_deepseek41_hc_finish_split_w32_v1");
        for rows in 1..=192 {
            let phases = [Phase::Prefill { rows }, Phase::Verify { rows, items: 1 }]
                .into_iter()
                .chain((rows <= DECODE_BATCH).then_some(Phase::Decode(rows)));
            for phase in phases {
                for ops in [&on, &drafter] {
                    assert_eq!(mix_at(ops, phase), mix_at(&off, phase), "{phase:?}");
                    if rows <= 64 || phase.prefill() || ops.wide.is_empty() {
                        assert_eq!(finish_at(ops, phase), finish_at(&off, phase), "{phase:?}");
                        continue;
                    }
                    assert!(!off.has_finish_norm(phase));
                    let fused = finish_at(ops, phase).unwrap();
                    assert_eq!(fused.len(), 2);
                    for (new, old) in fused.iter().zip(&finish_64) {
                        assert_eq!((&new.symbol, &new.arguments), (&old.symbol, &old.arguments));
                        assert_eq!(new.plan.block, old.plan.block);
                        assert_eq!(
                            new.plan.workspace_bytes,
                            old.plan.workspace_bytes / 64 * u64::from(rows)
                        );
                    }
                    assert_eq!(fused[0].plan.grid, [40, rows.div_ceil(4), 1]);
                    assert_eq!(fused[1].plan.grid, [2, rows, 1]);
                }
            }
        }
    }

    #[test]
    fn warp_mixer_is_limited_to_qualified_decode_and_verify_buckets() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                HcOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default(), true).unwrap();
            for rows in [1, 2, 4, 6, 8, 12, 16, 32] {
                for phase in [
                    Phase::Decode(rows),
                    Phase::Prefill { rows },
                    Phase::Verify { rows, items: 1 },
                ] {
                    let selected = ops
                        .mix_for_phase(phase)
                        .info()
                        .symbols
                        .iter()
                        .any(|(s, _)| s.ends_with("hc_mixer_probe_warp_v1"));
                    let qualified = matches!(
                        phase,
                        Phase::Decode(1 | 8)
                            | Phase::Verify {
                                rows: 4 | 6 | 8 | 12 | 16 | 32,
                                ..
                            }
                    );
                    assert_eq!(selected, qualified, "{phase:?}");
                    assert_eq!(
                        ops.has_finish_norm(phase),
                        qualified || matches!(phase, Phase::Decode(16 | 32))
                    );
                }
            }
            assert_eq!(ops.scratch().partials, Some(32 * 15360));
            assert_eq!(ops.scratch().tickets, Some(32 * 4));
            assert_eq!(
                ops.draft_scratch().partials,
                Some(u64::from(SMALL_DRAFT_ROWS) * 15360)
            );
            assert_eq!(
                ops.draft_scratch().tickets,
                Some(u64::from(SMALL_DRAFT_ROWS) * 4)
            );
        }
    }
}
