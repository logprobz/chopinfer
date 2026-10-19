//! One rank's lowering of a phase's stages into launches through the
//! prepared ops (`crate::ops`), over the step's inputs and the rank's
//! buffers.
//!
//! A stage whose family the instance lacks, or whose lowering is not
//! written yet, lowers to `None`: execute refuses such a step unless the
//! instance runs the lowered stages alone (the fake-host tests).

mod compressor;
mod dense;
pub mod dspark;
mod engram;
mod exact_f32;
mod hc_expand_mix;
mod indexer;
mod latent;
mod logits;
mod moe;
mod norm_quantize;

pub use compressor::{TailUpload, tail_upload};

use bundle_runtime::state::Extent;
use bundle_runtime::upload::DeviceInputs;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use crate::buffers::{RankBuffers, Scratch, rows};
use crate::ops::{
    EmbeddingViews, ExpandViews, HeadViews, Lowered, MixViews, NormViews, Ops, PeerViews,
    ReduceViews, StreamViews, WidenViews,
};
use crate::program::{Phase, Stage, Sublayer};
use crate::shape::Shape;

/// What one rank lowers a phase with.
pub struct Lowering<'a> {
    pub shape: &'a Shape,
    pub ops: &'a Ops,
    pub buffers: &'a RankBuffers,
    /// Every rank's buffers in rank order, this rank's included: the
    /// collectives bind their peers' views.
    pub peers: &'a [RankBuffers],
    pub inputs: &'a DeviceInputs,
    pub rank: u32,
    pub kv: KvExtent,
    /// The rows the head scores: a prefill's logits rows, or a decode or
    /// verify phase's rows.
    pub head_rows: u32,
}

/// The serving KV pages and request state a rank's stages address: the
/// pages' storage and count, what the state buffers are sized for, and the
/// block tables' rows (one per item slot) and entries per row.
#[derive(Clone, Copy, Debug)]
pub struct KvExtent {
    pub storage: KvStorage,
    pub pages: u32,
    pub extent: Extent,
    pub sequences: u32,
    pub max_pages: u32,
}

/// The checkpoint name of layer `layer`'s tensor `name`.
fn layer_view(layer: u32, name: &str) -> String {
    format!("layers.{layer}.{name}")
}

/// The position of `(layer, sublayer)` among the sublayers.
pub(crate) const fn sublayer_index(layer: u32, sublayer: Sublayer) -> u32 {
    2 * layer
        + match sublayer {
            Sublayer::Attention => 0,
            Sublayer::Mlp => 1,
        }
}

/// The streams sublayer `index` reads: the two stream buffers alternate,
/// the first sublayer reading the expanded embedding's.
pub(crate) fn streams(scratch: &Scratch<DeviceBufferSpanV1>, index: u32) -> DeviceBufferSpanV1 {
    match index % 2 {
        0 => scratch.streams,
        _ => scratch.expanded,
    }
}

/// The `pre` sublayer `index`'s mixer writes.
pub(crate) fn pre(scratch: &Scratch<DeviceBufferSpanV1>, index: u32) -> DeviceBufferSpanV1 {
    match index % 2 {
        0 => scratch.pre,
        _ => scratch.pre_odd,
    }
}

/// The `pre` sublayer `index` collapses with: the previous mixer's, or the
/// one-hot `pre` before the first sublayer.
fn pre_in(scratch: &Scratch<DeviceBufferSpanV1>, index: u32) -> DeviceBufferSpanV1 {
    match index {
        0 => scratch.first_pre,
        _ => pre(scratch, index - 1),
    }
}

/// The checkpoint prefix of a sublayer's mixer weights.
const fn mixer(sublayer: Sublayer) -> &'static str {
    match sublayer {
        Sublayer::Attention => "hc_attn",
        Sublayer::Mlp => "hc_ffn",
    }
}

impl Lowering<'_> {
    /// `stage`'s launches in `phase`, or `None` while it is not lowered.
    pub fn stage(&self, phase: Phase, stage: Stage) -> Result<Option<Vec<Lowered>>, String> {
        let mut lowered = self.stage_inner(phase, stage)?;
        if matches!(
            stage,
            Stage::SharedGateUp(_) | Stage::SharedActivation(_) | Stage::SharedDown(_)
        ) && let Some(launches) = &mut lowered
        {
            // Relocate at the stage boundary so split, paired and future
            // shared kernels all receive the same independent workspace.
            for lowered in launches {
                crate::shared_overlap::bind_shared(&mut lowered.launch, self.buffers)?;
            }
        }
        Ok(lowered)
    }

    fn stage_inner(&self, phase: Phase, stage: Stage) -> Result<Option<Vec<Lowered>>, String> {
        match stage {
            Stage::Embedding | Stage::ExpandStreams => self.row_stage(phase, stage),
            Stage::FinalNorm => self.final_norm(phase),
            Stage::InputNorm(layer, sublayer) if matches!(phase, Phase::Decode(_)) => {
                self.fused_input_norm(phase, layer, sublayer)
            }
            Stage::QuantizeInput(..) if matches!(phase, Phase::Decode(_)) => Ok(Some(Vec::new())),
            // The norm, then its rows widened for the exact F32 projections.
            Stage::InputNorm(layer, sublayer) => {
                let Some(mut launches) = self.row_stage(phase, stage)? else {
                    return Ok(None);
                };
                let Some(widen) = self.widen_normed(phase, layer, sublayer)? else {
                    return Ok(None);
                };
                launches.extend(widen);
                Ok(Some(launches))
            }
            Stage::EmbeddingReduce => self.embedding_reduce(phase),
            Stage::HcMix(layer, sublayer) if self.mix_was_fused(phase, layer, sublayer) => {
                Ok(Some(Vec::new()))
            }
            Stage::HcMix(layer, sublayer) => self.mix(phase, layer, sublayer),
            Stage::HcExpand(layer, sublayer) => self.expand(phase, layer, sublayer),
            Stage::HeadCollapse => self.head(phase),
            Stage::Reduce(layer, sublayer) => {
                if let Some(fused) = self.round_expand_with_next_mix(phase, layer, sublayer)? {
                    Ok(Some(fused))
                } else {
                    self.reduce(phase, |scratch| scratch.reduced)
                }
            }
            Stage::QuantizeInput(..) => self.quantize_input(phase),
            Stage::Route(layer) => self.route(phase, layer),
            Stage::ExpertsGateUp(layer) => self.experts(phase, layer, true),
            Stage::ExpertsDown(layer) => self.experts(phase, layer, false),
            Stage::RoutedSum(_) => self.routed_sum(phase),
            Stage::SharedGateUp(layer) => self.shared_gate_up(phase, layer),
            Stage::SharedActivation(layer) => self.shared_activation(phase, layer),
            Stage::SharedDown(layer) => self.dense(phase, layer, "moe.shared-down"),
            Stage::RopeRows => self.rope_rows(phase),
            Stage::WindowKeys => self.window_keys(phase),
            Stage::QueryNorm(layer) => self.query_norm(phase, layer),
            Stage::QueryRope(layer) => self.query_rope(phase, layer),
            Stage::WindowKv(layer) => self.window_kv(phase, layer),
            Stage::LatentAttention(layer) => self.latent_attention(phase, layer),
            Stage::WindowRingWrite(layer) => self.window_ring_write(phase, layer),
            Stage::Project(layer, id) => match self.dense(phase, layer, id)? {
                Some(launches) => Ok(Some(launches)),
                None => self.exact_f32(phase, layer, id),
            },
            Stage::IndexQuery(layer) => self.index_query(phase, layer),
            Stage::IndexSelect(layer) => self.index_select(phase, layer),
            Stage::Compress(layer) => self.compress(phase, layer),
            Stage::IndexKeyWrite(layer) => self.compressed_write(phase, layer, true),
            Stage::CompressedKvWrite(layer) => self.compressed_write(phase, layer, false),
            // Only the DSpark drafter reads the retained feature, and no
            // instance loads the drafter yet (create refuses its state).
            Stage::DsparkFeature(_) => Ok(Some(Vec::new())),
            Stage::EngramRows(layer) => self.engram_rows(phase, layer),
            Stage::EngramGate(layer) => self.engram_gate(phase, layer),
            Stage::GatherHeadRows => self.gather_head_rows(phase),
            Stage::LmHead => self.lm_head(phase),
            Stage::GatherLogits => self.gather_logits(phase),
        }
    }

    /// Peer gathering is measured only at two ranks.
    fn replicated_embedding(&self, phase: Phase) -> bool {
        matches!(phase, Phase::Decode(_))
            && self.shape.ranks() == 2
            && self
                .ops
                .rows
                .as_ref()
                .is_some_and(|r| r.has_replicated_embedding())
    }

    fn row_stage(&self, phase: Phase, stage: Stage) -> Result<Option<Vec<Lowered>>, String> {
        let error = |refusal: kernel_core::contract::Refusal| format!("{stage:?}: {refusal}");
        let Some(row_ops) = &self.ops.rows else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let live_rows = self.inputs.live_tokens;
        let norm = |weight: String| -> Result<Option<Vec<Lowered>>, String> {
            let views = NormViews {
                x: rows_of(scratch.collapsed),
                weight: self.buffers.view(&weight)?,
                y: rows_of(scratch.normed),
                live_rows,
                status: None,
            };
            row_ops
                .norm(self.shape.dims().hidden, phase.rows(), phase.rows(), &views)
                .map(Some)
                .map_err(error)
        };
        match stage {
            Stage::Embedding if self.replicated_embedding(phase) => {
                let views = kernel_core::replicated_embedding::ReplicatedEmbeddingBuffers {
                    tables: self
                        .peers
                        .iter()
                        .map(|p| p.view("embed.weight"))
                        .collect::<Result<_, _>>()?,
                    ids: self.inputs.token_ids,
                    shard: rows_of(scratch.embedding),
                    wide: rows_of(scratch.partial),
                    output: rows_of(scratch.embedded),
                    expanded: rows_of(scratch.streams),
                    live_rows,
                };
                row_ops
                    .replicated_embedding(self.rank, phase.rows(), views)
                    .map(Some)
                    .map_err(error)
            }
            Stage::ExpandStreams if self.replicated_embedding(phase) => Ok(Some(Vec::new())),
            Stage::Embedding => {
                let views = EmbeddingViews {
                    table: self.buffers.view("embed.weight")?,
                    ids: self.inputs.token_ids,
                    output: rows_of(scratch.embedding),
                    live_rows,
                };
                row_ops
                    .embedding(self.rank, phase.rows(), &views)
                    .map(Some)
                    .map_err(error)
            }
            Stage::ExpandStreams => {
                let views = StreamViews {
                    source: rows_of(scratch.embedded),
                    destination: rows_of(scratch.streams),
                    live_rows,
                };
                row_ops
                    .expand_streams(phase.rows(), &views)
                    .map(Some)
                    .map_err(error)
            }
            Stage::InputNorm(layer, Sublayer::Attention) => {
                norm(layer_view(layer, "attn_norm.weight"))
            }
            Stage::InputNorm(layer, Sublayer::Mlp) => norm(layer_view(layer, "ffn_norm.weight")),
            _ => Ok(None),
        }
    }

    /// The final norm of the F32 collapse into the F32 rows the LM head
    /// reads, flagging the phase's row status.
    fn final_norm(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(row_ops) = &self.ops.rows else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let views = NormViews {
            x: rows_of(scratch.head_collapsed),
            weight: self.buffers.view("norm.weight")?,
            y: rows_of(scratch.normed_wide),
            live_rows: self.inputs.live_tokens,
            status: Some(rows_of(scratch.row_status)),
        };
        row_ops
            .final_norm(phase.rows(), phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("final norm: {refusal}"))
    }

    /// Sublayer `(layer, sublayer)`'s lagged mix-collapse: its streams
    /// collapsed with the previous mixer's `pre`, and its own `pre`, `post`
    /// and `comb`.
    fn mix(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.hc else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let index = sublayer_index(layer, sublayer);
        let weight = |name: &str| {
            self.buffers
                .view(&layer_view(layer, &format!("{}_{name}", mixer(sublayer))))
        };
        let views = MixViews {
            x: rows_of(streams(scratch, index)),
            mixer: [weight("fn")?, weight("base")?, weight("scale")?],
            pre_in: rows_of(pre_in(scratch, index)),
            collapsed: rows_of(scratch.collapsed),
            pre: rows_of(pre(scratch, index)),
            post: rows_of(scratch.post),
            comb: rows_of(scratch.comb),
            partials: self.buffers.mix_partials,
            tickets: self.buffers.mix_tickets,
            live_rows: self.inputs.live_tokens,
        };
        ops.mix(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("mix-collapse: {refusal}"))
    }

    /// Sublayer `(layer, sublayer)`'s reduced output expanded with its
    /// streams into the other stream buffer.
    fn expand(
        &self,
        phase: Phase,
        layer: u32,
        sublayer: Sublayer,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if let Some(launches) = self.expand_with_next_mix(phase, layer, sublayer)? {
            return Ok(Some(launches));
        }
        let Some(ops) = &self.ops.hc else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let index = sublayer_index(layer, sublayer);
        let views = ExpandViews {
            x: rows_of(streams(scratch, index)),
            y: rows_of(scratch.reduced),
            post: rows_of(scratch.post),
            comb: rows_of(scratch.comb),
            output: rows_of(streams(scratch, index + 1)),
            live_rows: self.inputs.live_tokens,
        };
        ops.expand(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("expand: {refusal}"))
    }

    /// The last sublayer's streams collapsed by its mixer's `pre` into the
    /// F32 rows the final norm reads.
    fn head(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.hc else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let end = 2 * self.shape.geometry.layers();
        let views = HeadViews {
            x: rows_of(streams(scratch, end)),
            pre: rows_of(pre(scratch, end - 1)),
            output: rows_of(scratch.head_collapsed),
            live_rows: self.inputs.live_tokens,
        };
        ops.head(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("head collapse: {refusal}"))
    }

    /// The embedding's shard rows widened into the partials, then reduced
    /// into the embedding.
    fn embedding_reduce(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        if self.replicated_embedding(phase) {
            return Ok(Some(Vec::new()));
        }
        let Some(ops) = &self.ops.reduce else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let views = WidenViews {
            input: rows(scratch.embedding, phase.rows()),
            output: rows(scratch.partial, phase.rows()),
            live_rows: self.inputs.live_tokens,
        };
        let mut launches = ops
            .widen(phase.rows(), &views)
            .map_err(|refusal| format!("embedding widening: {refusal}"))?;
        let Some(reduce) = self.reduce(phase, |scratch| scratch.embedded)? else {
            return Ok(None);
        };
        launches.extend(reduce);
        Ok(Some(launches))
    }

    /// The all-reduce of the rank partials into `output`'s rows on every
    /// rank.
    fn reduce(
        &self,
        phase: Phase,
        output: impl Fn(&Scratch<DeviceBufferSpanV1>) -> DeviceBufferSpanV1,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.reduce else {
            return Ok(None);
        };
        let views = self.reduce_views(phase, output);
        ops.reduce(self.rank, phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("all-reduce: {refusal}"))
    }

    fn reduce_views(
        &self,
        phase: Phase,
        output: impl Fn(&Scratch<DeviceBufferSpanV1>) -> DeviceBufferSpanV1,
    ) -> ReduceViews {
        let rows_of = |span| rows(span, phase.rows());
        ReduceViews {
            input: rows_of(self.buffers.scratch.partial),
            output: rows_of(output(&self.buffers.scratch)),
            live_rows: self.inputs.live_tokens,
            peers: self
                .peers
                .iter()
                .map(|peer| PeerViews {
                    output: rows_of(output(&peer.scratch)),
                    state: peer.comm.clone(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::allocation::Traversal;
    use kernel_core::backend::{
        Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View,
    };
    use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
    use kernel_core::properties::Invariance;
    use kernel_core::sinkhorn_hyper_connection::{
        LaggedMixCollapseBuffers, SinkhornHyperConnectionQuery, StreamExpandBuffers,
        WeightedStreamHeadBuffers,
    };
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::own_ops::OwnOps;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use crate::buffers::StateSpans;
    use crate::program::stages;

    /// An op that launches one entry, `symbol`, over every view it binds
    /// in role order.
    struct Echo(&'static str, OpInfo);

    impl<B: Traversal<View>> PreparedOp<B> for Echo {
        fn info(&self) -> &OpInfo {
            &self.1
        }

        fn launches(&self, buffers: &B, _: View, counts: Counts) -> Result<Vec<Launch>, Refusal> {
            let mut arguments = Vec::new();
            buffers.visit(&mut |view: &View| arguments.push(Argument::DeviceBuffer(view.handle)));
            Ok(vec![Launch {
                symbol: self.0.into(),
                rows: counts.rows,
                plan: LaunchPlan {
                    grid: [1; 3],
                    block: [32, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1; 3],
                    pdl: false,
                },
                arguments: ArgumentList(arguments),
                collective: false,
                persistent: false,
            }])
        }
    }

    fn echo<B: Traversal<View>>(symbol: &'static str, need: Need) -> Prepared<B> {
        Box::new(Echo(
            symbol,
            OpInfo::new(symbol, Invariance::any_rows(need.capacity), vec![]),
        ))
    }

    /// Kernels with the hyper-connection ops alone.
    struct HcOnly;

    impl Backend for HcOnly {
        fn name(&self) -> &str {
            "hc only"
        }

        fn sinkhorn_lagged_mix_collapse(
            &self,
            _: &SinkhornHyperConnectionQuery,
            need: Need,
        ) -> Result<Prepared<LaggedMixCollapseBuffers<View>>, Refused> {
            Ok(echo("mix", need))
        }

        fn sinkhorn_expand(
            &self,
            _: &SinkhornHyperConnectionQuery,
            need: Need,
        ) -> Result<Prepared<StreamExpandBuffers<View>>, Refused> {
            Ok(echo("expand", need))
        }

        fn sinkhorn_weighted_head(
            &self,
            _: &SinkhornHyperConnectionQuery,
            need: Need,
        ) -> Result<Prepared<WeightedStreamHeadBuffers<View>>, Refused> {
            Ok(echo("head", need))
        }
    }

    impl OwnOps for HcOnly {
        fn name(&self) -> &str {
            "hc only"
        }
    }

    /// Rank `rank`'s buffers, every span its own buffer, with each layer's
    /// mixer weights.
    pub(super) fn buffers(shape: &Shape, rank: u32) -> RankBuffers {
        let mut next = 0;
        let mut span = |bytes| {
            next += 1;
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(next, rank + 1),
                0,
                bytes,
            )
        };
        let scratch =
            bundle_runtime::scratch::bind(Scratch::of(shape), |bytes| Ok(span(bytes))).unwrap();
        let mut views = std::collections::BTreeMap::new();
        for layer in 0..shape.geometry.layers() {
            for prefix in ["hc_attn", "hc_ffn"] {
                for name in ["fn", "base", "scale"] {
                    views.insert(layer_view(layer, &format!("{prefix}_{name}")), span(64));
                }
            }
        }
        RankBuffers {
            scratch,
            state: StateSpans {
                buffers: Vec::new(),
            },
            views,
            kv_pool: span(64),
            comm: Vec::new(),
            logits: None,
            sorted_rows: span(64),
            workspace: span(64),
            shared_workspace: span(64),
            mix_partials: None,
            mix_tickets: None,
            rope_compressed: span(64),
        }
    }

    /// Each sublayer collapses its streams with the previous mixer's `pre`
    /// (the one-hot `pre` first) and expands into the other stream buffer,
    /// and the head collapses the last streams by the last mixer's `pre`.
    #[test]
    fn mixers_alternate_the_streams_and_lag_their_pre() {
        let shape = Shape::new(&FLASH, Ranks::TP1).unwrap();
        let kernels = crate::Kernels {
            backend: Box::new(HcOnly),
            own: Box::new(HcOnly),
            gather_state: [0; 2],
        };
        let ops = Ops::prepare(&kernels, &shape, KvStorage::Native);
        assert!(ops.hc.is_some() && ops.rows.is_none());
        let buffers = buffers(&shape, 0);
        let live = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(999, 1), 0, 4);
        let inputs = DeviceInputs {
            live_tokens: live,
            ..DeviceInputs::within(
                bundle_runtime::upload::PhaseArena {
                    arena: live,
                    block_tables: live,
                },
                &bundle_runtime::upload::InputLayout::decode_within(
                    bundle_runtime::geometry::Items::new(1).unwrap(),
                ),
            )
        };
        let peers = [buffers.clone()];
        let lowering = Lowering {
            shape: &shape,
            ops: &ops,
            buffers: &buffers,
            peers: &peers,
            inputs: &inputs,
            rank: 0,
            kv: KvExtent {
                storage: KvStorage::Native,
                pages: 1,
                extent: Extent {
                    served: 1,
                    bucket: 1,
                    verify_rows: 0,
                },
                sequences: 1,
                max_pages: 1,
            },
            head_rows: 1,
        };
        let phase = Phase::Decode(1);
        let scratch = &buffers.scratch;
        let one = |span| rows(span, 1);
        // Each launch's views, as spans, by entry.
        let mut launches: Vec<(String, Vec<DeviceBufferSpanV1>)> = Vec::new();
        for stage in stages(shape.geometry, phase) {
            for lowered in lowering.stage(phase, stage).unwrap().unwrap_or_default() {
                let spans = lowered
                    .launch
                    .arguments
                    .0
                    .iter()
                    .map(|argument| match argument {
                        Argument::DeviceBuffer(handle) => {
                            lowered.launch.bindings.get(*handle).unwrap()
                        }
                        other => panic!("{other:?}"),
                    })
                    .collect();
                launches.push((lowered.launch.symbol, spans));
            }
        }
        let sublayers = 2 * shape.geometry.layers() as usize;
        assert_eq!(launches.len(), 2 * sublayers + 1);
        let [even, odd] = [one(scratch.streams), one(scratch.expanded)];
        let [pre_even, pre_odd] = [one(scratch.pre), one(scratch.pre_odd)];
        for index in 0..sublayers {
            let (mix, expand) = (&launches[2 * index], &launches[2 * index + 1]);
            let (x, other) = match index % 2 {
                0 => (even, odd),
                _ => (odd, even),
            };
            let pre = [pre_even, pre_odd][index % 2];
            let pre_in = match index {
                0 => one(scratch.first_pre),
                _ => [pre_even, pre_odd][(index - 1) % 2],
            };
            // x, fn, base, scale, pre_in, collapsed, pre, post, comb,
            // live rows.
            assert_eq!(mix.0, "mix");
            assert_eq!(
                [mix.1[0], mix.1[4], mix.1[5], mix.1[6], mix.1[9]],
                [x, pre_in, one(scratch.collapsed), pre, live],
                "sublayer {index}"
            );
            // x, y, post, comb, output, live rows.
            assert_eq!(expand.0, "expand");
            assert_eq!(
                [
                    expand.1[0],
                    expand.1[1],
                    expand.1[2],
                    expand.1[3],
                    expand.1[4]
                ],
                [x, one(scratch.reduced), mix.1[7], mix.1[8], other],
                "sublayer {index}"
            );
        }
        let head = launches.last().unwrap();
        assert_eq!(head.0, "head");
        assert_eq!(head.1[..3], [even, pre_odd, one(scratch.head_collapsed)]);
    }
}
