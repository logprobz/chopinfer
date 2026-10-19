//! The full-block vocabulary projection and five sequential greedy Markov
//! steps. Token IDs stay on the device until Engram staging reads the block.

use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::collective::{AllReduceBuffers, GatherBuffers, RankViews};
use kernel_core::contract::{ArgmaxLayout, CommQuery, DType, Refusal, RowQuery};
use kernel_core::precision::buffers::ProjectionBuffers;
use kernel_core::rmsnorm::RmsNormBuffers;
use kernel_core::row_copy::{GatherByIndex, GatherByIndexBuffers};
use kernel_core::rows::{
    ArgmaxFinalBuffers, ArgmaxPairs, ArgmaxPartialBuffers, ArgmaxScatter, EmbeddingBuffers,
};
use kernel_core::sinkhorn_hyper_connection::{
    SinkhornHyperConnectionQuery, StreamHeadMode, WeightedStreamHeadBuffers,
};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{
    ConfidenceHead, ConfidenceHeadBuffers, MarkovStep, MarkovStepBuffers,
};
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};
use model_deepseek41_contract::serving::{DECODE_BATCH, PREFILL_TOKENS};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

use super::ops::slice;
use super::{BLOCK_ROWS, DraftMode};
use crate::ops::exact_f32;
use crate::ops::{Bind, Lowered};
use crate::shape::Shape;

pub struct Heads {
    shape: Shape,
    collapse: Prepared<WeightedStreamHeadBuffers<View>>,
    norm: Prepared<RmsNormBuffers<View>>,
    vocabulary: Prepared<ProjectionBuffers<View>>,
    embedding: Vec<Prepared<EmbeddingBuffers<View>>>,
    widen: Prepared<WidenRowsBuffers<View>>,
    reduce: Vec<Prepared<AllReduceBuffers<View>>>,
    logits_row: Prepared<GatherByIndexBuffers<View>>,
    markov: Prepared<MarkovStepBuffers<View>>,
    confidence: Prepared<ConfidenceHeadBuffers<View>>,
    partial: Vec<Prepared<ArgmaxPartialBuffers<View>>>,
    gather: Vec<Prepared<GatherBuffers<View>>>,
    select: Prepared<ArgmaxFinalBuffers<View>>,
}

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub struct HeadViews {
    /// Final drafter streams (BF16) and the last mixer's F32 pre weights.
    pub streams: Span,
    pub pre: Span,
    /// Output of the unrounded F32 collapse. Preserve it for confidence.
    pub hidden_f32: Span,
    pub norm_weight: Span,
    pub normalized_f32: Span,
    pub vocabulary_weight: Span,
    pub projection_state: exact_f32::ProjectionState,
    pub logits: Span,
    pub live_rows: Span,
    pub status: Span,
}

pub struct Peer {
    pub embedding: Span,
    pub reduce_state: Vec<Span>,
    pub gathered: Span,
    pub gather_state: Vec<Span>,
}

#[derive(Clone, Copy)]
pub struct PositionViews {
    pub predecessor_ids: Span,
    pub embedding_weight: Span,
    pub embedding_partial: Span,
    pub embedding_wide: Span,
    pub embedding: Span,
    pub block_logits: Span,
    pub position_rows: Span,
    pub position_logits: Span,
    pub markov_weight: Span,
    pub hidden_f32: Span,
    pub confidence_weight: Span,
    pub confidence: Span,
    /// Two planes of DECODE_BATCH scores and IDs, initialized before use.
    pub pairs: Span,
    pub gathered: Span,
    pub live_width: Span,
    pub live_items: Span,
    pub draft_ids: Span,
}

impl Heads {
    pub fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: Shape,
        mode: DraftMode,
    ) -> Result<Self, Refused> {
        mode.validate().map_err(Refused::Rule)?;
        let g = shape.geometry;
        let need = Need::rows(DECODE_BATCH);
        let every = Need::rows(PREFILL_TOKENS);
        let dim = g.shape.dspark.markov_rank;
        let width = shape.vocab_rows();
        let collapse = backend.sinkhorn_weighted_head(
            &SinkhornHyperConnectionQuery::Head {
                streams: g.shape.hc.streams,
                width: g.shape.hidden,
                mode: StreamHeadMode::SigmoidWeightedF32,
                capacity: PREFILL_TOKENS,
            },
            every,
        )?;
        let norm = backend.rmsnorm(&model_deepseek41_contract::rows::final_norm(g), every)?;
        let vocabulary = backend
            .projection(
                &exact_f32::operation(
                    "dspark.head",
                    "head.weight",
                    exact_f32::query(width, g.shape.hidden),
                )?,
                exact_f32::USE,
                Need::rows(DECODE_BATCH * BLOCK_ROWS),
            )?
            .handle;
        let embedding = (0..shape.ranks())
            .map(|rank| {
                backend.embedding_gather(
                    &RowQuery::EmbeddingGather {
                        vocab: width,
                        width: dim,
                    },
                    rank * width,
                    need,
                )
            })
            .collect::<Result<_, _>>()?;
        let widen = own.widen_rows(&WidenRows { width: dim }, need)?;
        let reduce = (0..shape.ranks())
            .map(|rank| {
                backend.all_reduce(
                    &CommQuery::AllReduceRounded {
                        ranks: shape.ranks(),
                        elements: u64::from(dim),
                    },
                    rank,
                    need,
                )
            })
            .collect::<Result<_, _>>()?;
        let logits_row = backend.gather_by_index(
            &GatherByIndex {
                dtype: DType::F32,
                width,
                alignment_bytes: 16,
            },
            need,
        )?;
        let markov = own.dspark_markov_step(
            &MarkovStep {
                rank: dim,
                vocab_rows: width,
                logits_stride: width,
            },
            need,
        )?;
        let confidence = own.dspark_confidence(
            &ConfidenceHead {
                hidden: g.shape.hidden,
                markov_rank: dim,
                hidden_stride: BLOCK_ROWS * g.shape.hidden,
                score_stride: BLOCK_ROWS,
            },
            need,
        )?;
        let partial = (0..shape.ranks())
            .map(|rank| {
                backend.sharded_argmax_partial(
                    &RowQuery::ShardedArgmaxPartial {
                        vocab_per_rank: width,
                        ranks: shape.ranks(),
                        logits: DType::F32,
                    },
                    rank,
                    need,
                )
            })
            .collect::<Result<_, _>>()?;
        let gather = (0..shape.ranks())
            .map(|rank| {
                backend.all_gather(
                    &CommQuery::AllGather {
                        ranks: shape.ranks(),
                        elements_per_rank: 2 * u64::from(DECODE_BATCH),
                        dtype: DType::F32,
                    },
                    rank,
                    None,
                    need,
                )
            })
            .collect::<Result<_, _>>()?;
        let select = backend.sharded_argmax_final(
            &RowQuery::ShardedArgmaxFinal {
                ranks: shape.ranks(),
                pairs: ArgmaxLayout::RankMajor {
                    stride: 2 * DECODE_BATCH,
                },
            },
            need,
        )?;
        Ok(Self {
            shape,
            collapse,
            norm,
            vocabulary,
            embedding,
            widen,
            reduce,
            logits_row,
            markov,
            confidence,
            partial,
            gather,
            select,
        })
    }

    fn infos(&self) -> impl Iterator<Item = &kernel_core::backend::OpInfo> {
        [
            self.collapse.info(),
            self.norm.info(),
            self.vocabulary.info(),
            self.widen.info(),
            self.logits_row.info(),
            self.markov.info(),
            self.confidence.info(),
            self.select.info(),
        ]
        .into_iter()
        .chain(self.embedding.iter().map(|op| op.info()))
        .chain(self.reduce.iter().map(|op| op.info()))
        .chain(self.partial.iter().map(|op| op.info()))
        .chain(self.gather.iter().map(|op| op.info()))
    }

    pub(crate) fn symbols(&self) -> impl Iterator<Item = &(String, u32)> {
        self.infos().flat_map(|info| &info.symbols)
    }

    pub(crate) fn workspace(&self) -> u64 {
        self.infos().map(|info| info.workspace).max().unwrap_or(0)
    }

    /// Per-rank persistent collective regions, zeroed once and kept
    /// separate for embedding reduction and candidate gathering.
    pub fn state(&self) -> (&[u64], &[u64]) {
        (&self.reduce[0].info().state, &self.gather[0].info().state)
    }

    pub fn vocabulary(&self, rows: u32, v: &HeadViews) -> Result<Vec<Lowered>, Refusal> {
        // These writes must not overwrite the raw hidden that every
        // confidence position reads after the vocabulary projection.
        for other in [v.streams, v.pre, v.normalized_f32, v.logits] {
            let end = |span: Span| {
                span.offset
                    .checked_add(span.length)
                    .ok_or(Refusal("DSpark head span overflows"))
            };
            let (hidden_end, other_end) = (end(v.hidden_f32)?, end(other)?);
            if v.hidden_f32.buffer == other.buffer
                && v.hidden_f32.offset < other_end
                && other.offset < hidden_end
            {
                return Err(Refusal(
                    "DSpark collapsed hidden must have separate storage",
                ));
            }
        }
        let mut bind = Bind::default();
        let buffers = WeightedStreamHeadBuffers {
            x: bind.view(v.streams),
            pre: bind.view(v.pre),
            output: bind.view(v.hidden_f32),
            live_rows: bind.view(v.live_rows),
        };
        let mut result = bind.launch(&self.collapse, &buffers, Counts::rows(rows))?;
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
            x: bind.view(v.hidden_f32),
            weight: bind.view(v.norm_weight),
            residual: None,
            y: bind.view(v.normalized_f32),
            live_rows: bind.view(v.live_rows),
            status: Some(bind.view(v.status)),
            status_row_offset: 0,
        };
        result.extend(bind.launch(&self.norm, &buffers, Counts::rows(rows))?);
        result.extend(exact_f32::project(
            &self.vocabulary,
            rows,
            v.normalized_f32,
            v.vocabulary_weight,
            v.logits,
            v.live_rows,
            v.projection_state,
        )?);
        Ok(result)
    }

    /// Append positions 0..5 in order on every rank, with the predecessor
    /// IDs initialized to each item's bonus ID. All ranks enter every
    /// collective. The final argmax writes both the next predecessor and
    /// this position of the item-major five-ID output, with lower-ID ties.
    pub fn position(
        &self,
        rank: u32,
        position: u32,
        items: u32,
        v: &PositionViews,
        peers: &[Peer],
    ) -> Result<Vec<Lowered>, String> {
        if rank >= self.shape.ranks()
            || position >= BLOCK_ROWS
            || items == 0
            || items > DECODE_BATCH
            || peers.len() != self.shape.ranks() as usize
        {
            return Err("DSpark Markov position is outside the prepared shape".into());
        }
        let counts = Counts::rows(items);
        let mut result = Vec::new();
        let run = |value: Result<Vec<Lowered>, Refusal>| value.map_err(|e| e.to_string());
        let mut bind = Bind::default();
        let buffers = EmbeddingBuffers {
            table: bind.view(v.embedding_weight),
            ids: bind.view(v.predecessor_ids),
            output: bind.view(v.embedding_partial),
            live_rows: bind.view(v.live_items),
            table_rows: self.shape.vocab_rows(),
        };
        result.extend(run(bind.launch(
            &self.embedding[rank as usize],
            &buffers,
            counts,
        ))?);
        let mut bind = Bind::default();
        let buffers = WidenRowsBuffers {
            input: bind.view(v.embedding_partial),
            output: bind.view(v.embedding_wide),
            live_rows: bind.view(v.live_items),
        };
        result.extend(run(bind.launch(&self.widen, &buffers, counts))?);
        let mut bind = Bind::default();
        let buffers = AllReduceBuffers {
            input: bind.view(v.embedding_wide),
            output: bind.view(v.embedding),
            live_rows: Some(bind.view(v.live_items)),
            ranks: peers
                .iter()
                .map(|p| RankViews {
                    output: bind.view(p.embedding),
                    state: p.reduce_state.iter().map(|&s| bind.view(s)).collect(),
                })
                .collect(),
        };
        result.extend(run(bind.launch(
            &self.reduce[rank as usize],
            &buffers,
            counts,
        ))?);
        let mut bind = Bind::default();
        let buffers = GatherByIndexBuffers {
            source: bind.view(v.block_logits),
            row_indices: bind.view(v.position_rows),
            count: bind.view(v.live_items),
            destination: bind.view(v.position_logits),
        };
        result.extend(run(bind.launch(&self.logits_row, &buffers, counts))?);
        let mut bind = Bind::default();
        let output = bind.view(v.position_logits);
        let buffers = MarkovStepBuffers {
            embedding: bind.view(v.embedding),
            weight: bind.view(v.markov_weight),
            base: output,
            output,
            live_rows: bind.view(v.live_items),
        };
        result.extend(run(bind.launch(&self.markov, &buffers, counts))?);
        let hidden_offset = u64::from(position) * u64::from(self.shape.geometry.shape.hidden) * 4;
        let score_offset = u64::from(position) * 4;
        let mut bind = Bind::default();
        let buffers = ConfidenceHeadBuffers {
            hidden: bind.view(slice(
                v.hidden_f32,
                hidden_offset,
                v.hidden_f32
                    .length
                    .checked_sub(hidden_offset)
                    .ok_or("short DSpark F32 hidden span")?,
            )?),
            markov: bind.view(v.embedding),
            weight: bind.view(v.confidence_weight),
            scores: bind.view(slice(
                v.confidence,
                score_offset,
                v.confidence
                    .length
                    .checked_sub(score_offset)
                    .ok_or("short DSpark confidence span")?,
            )?),
            live_rows: bind.view(v.live_items),
        };
        result.extend(run(bind.launch(&self.confidence, &buffers, counts))?);
        let plane = u64::from(DECODE_BATCH) * 4;
        let mut bind = Bind::default();
        let buffers = ArgmaxPartialBuffers {
            logits: bind.view(v.position_logits),
            live_width: bind.view(v.live_width),
            live_rows: bind.view(v.live_items),
            pairs: ArgmaxPairs {
                scores: bind.view(slice(v.pairs, 0, plane)?),
                ids: bind.view(slice(v.pairs, plane, plane)?),
                peer: false,
            },
        };
        result.extend(run(bind.launch(
            &self.partial[rank as usize],
            &buffers,
            counts,
        ))?);
        let mut bind = Bind::default();
        let buffers = GatherBuffers {
            input: bind.view(v.pairs),
            ranks: peers
                .iter()
                .map(|p| RankViews {
                    output: bind.view(p.gathered),
                    state: p.gather_state.iter().map(|&s| bind.view(s)).collect(),
                })
                .collect(),
        };
        result.extend(run(bind.launch(
            &self.gather[rank as usize],
            &buffers,
            counts,
        ))?);
        let mut bind = Bind::default();
        let buffers = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: bind.view(v.gathered),
                ids: bind.view(slice(
                    v.gathered,
                    plane,
                    v.gathered
                        .length
                        .checked_sub(plane)
                        .ok_or("short DSpark gathered pairs")?,
                )?),
                peer: false,
            },
            output: Some(bind.view(v.predecessor_ids)),
            scatter: vec![ArgmaxScatter {
                rows: bind.view(v.position_rows),
                output: bind.view(v.draft_ids),
            }],
            live_rows: bind.view(v.live_items),
        };
        result.extend(run(bind.launch(&self.select, &buffers, counts))?);
        Ok(result)
    }
}
