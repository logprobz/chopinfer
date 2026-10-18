//! The indexer's ops: each index query's F32 tail rotation at the token's
//! position (the compressed RoPE table, as the index keys) and its MXFP4
//! fake quantization, whose only BF16 rounding is the quantizer input; and
//! the selections. A Full source scores every visible key and selects its
//! top-k (dense Position v3), the candidate source also publishing each
//! row's top blocks as block IDs; a Reindex source scores and selects
//! inside those blocks (compact Position v3). The dense scores are
//! `[rows][position capacity]` F32, so the selections run in row blocks.

use kernel_core::attention::BlockTables;
use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::pooled_indexer::{PositionKeyPages, PositionQueryBuffers, PositionSelectBuffers};
use kernel_core::position_blocks::{
    CandidateBlockIds, CandidateBlockIdsBuffers, PositionBlockSelectBuffers,
};
use model_deepseek41_contract::indexer;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;

pub(crate) struct IndexerOps {
    /// The query preparation every index source shares: the heads, width
    /// and rotation are the same at each.
    query: Prepared<PositionQueryBuffers<View>>,
}

/// A step's spans of a query preparation: the F32 index query, the
/// compressed RoPE table, each row's position, the prepared BF16 query,
/// and the rank's backend workspace.
pub(crate) struct QueryViews {
    pub q: DeviceBufferSpanV1,
    pub cos_sin: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
    /// The phase's row status (`Scratch::row_status`).
    pub status: DeviceBufferSpanV1,
}

impl IndexerOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<IndexerOps, Refused> {
        let g = shape.geometry;
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let source = g.shape.csa.index_sources[0];
        let query = indexer::selection(g, source, PREFILL_TOKENS)
            .ok_or_else(|| Refused::Rule(format!("{} has no index source", g.id)))?;
        let query = asks.ask("indexer.query", every, |need| {
            backend.position_indexer_query(&query, need)
        })?;
        Ok(IndexerOps { query })
    }

    /// The preparation of `rows` index queries over a RoPE table of
    /// `table_rows` positions.
    pub(crate) fn prepare_query(
        &self,
        rows: u32,
        table_rows: u32,
        views: &QueryViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = PositionQueryBuffers {
            q: bind.view(views.q),
            cos_sin: bind.view(views.cos_sin),
            positions: bind.view(views.positions),
            live_rows: bind.view(views.live_rows),
            output: bind.view(views.output),
            quantizer_status: bind.view(views.status),
            quantizer_status_row_offset: 0,
        };
        let counts = Counts::rows(rows).with_positions(table_rows);
        bind.launch_within(&self.query, &buffers, views.workspace, counts)
    }
}

/// Rows of one dense selection launch: its scores are 256 MiB at a
/// 2^20-position capacity.
pub(crate) const FULL_BLOCK: u32 = 64;

/// Rows of one compact selection launch: its scores are 256 MiB.
pub(crate) const REINDEX_BLOCK: u32 = 4096;

/// The selections of an instance: each Full source ratio's dense
/// selection, the candidate source's block-ID producer and each Reindex
/// source ratio's compact selection.
pub(crate) struct SelectOps {
    full: Vec<(u32, Prepared<PositionSelectBuffers<View>>)>,
    producer: Prepared<CandidateBlockIdsBuffers<View>>,
    reindex: Vec<(u32, Prepared<PositionBlockSelectBuffers<View>>)>,
}

/// A source's index keys in the serving pages, as one row block's
/// selection reads them.
pub(crate) type KeyPages = PositionKeyPages<DeviceBufferSpanV1>;

/// The candidate source's retained block IDs of a row block: the IDs, each
/// row's tag and visible count, and its source status.
#[derive(Clone, Copy)]
pub(crate) struct CandidateViews {
    pub ids: DeviceBufferSpanV1,
    pub row_tags: DeviceBufferSpanV1,
    pub visible_positions: DeviceBufferSpanV1,
    pub status: DeviceBufferSpanV1,
}

/// One row block's spans of a selection: the prepared index queries, the
/// F32 head weights, the keys, each row's position and item, the block's
/// live count, the scores, and the selection's status and output.
pub(crate) struct SelectViews {
    pub q: DeviceBufferSpanV1,
    pub head_weights: DeviceBufferSpanV1,
    pub keys: KeyPages,
    pub positions: DeviceBufferSpanV1,
    pub row_batch: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub scores: DeviceBufferSpanV1,
    pub status: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
}

fn key_pages(bind: &mut Bind, keys: &KeyPages) -> PositionKeyPages<View> {
    PositionKeyPages {
        codes: bind.view(keys.codes),
        scales: bind.view(keys.scales),
        pages: keys.pages,
        code_page_stride: keys.code_page_stride,
        scale_page_stride: keys.scale_page_stride,
        page_tables: BlockTables {
            tables: bind.view(keys.page_tables.tables),
            sequences: keys.page_tables.sequences,
            max_pages: keys.page_tables.max_pages,
        },
    }
}

fn record(bind: &mut Bind, record: CandidateViews, width: u32) -> CandidateBlockIds<View> {
    CandidateBlockIds {
        ids: bind.view(record.ids),
        row_tags: bind.view(record.row_tags),
        visible_positions: bind.view(record.visible_positions),
        status: bind.view(record.status),
        position_capacity: width,
    }
}

impl SelectOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<SelectOps, Refused> {
        let g = shape.geometry;
        let mut full = Vec::new();
        let mut reindex = Vec::new();
        for &layer in g.shape.csa.index_sources {
            let ratio = g.compress_ratio(layer);
            let width = indexer::position_capacity(g, layer)
                .ok_or_else(|| Refused::Rule(format!("layer {layer} selects nothing")))?;
            match indexer::block_selection(g, layer, REINDEX_BLOCK) {
                None if !full.iter().any(|&(r, _)| r == ratio) => {
                    let query = indexer::selection(g, layer, FULL_BLOCK)
                        .ok_or_else(|| Refused::Rule(format!("layer {layer} selects nothing")))?;
                    let need = exact::need(FULL_BLOCK, exact::every_phase(FULL_BLOCK));
                    let op = asks.ask(format!("indexer.select r{ratio}"), need, |need| {
                        backend.position_indexer(&query, query.max_top_k, width, need)
                    })?;
                    full.push((ratio, op));
                }
                Some(query) if !reindex.iter().any(|&(r, _)| r == ratio) => {
                    let need = exact::need(REINDEX_BLOCK, exact::every_phase(REINDEX_BLOCK));
                    let op = asks.ask(format!("indexer.reindex r{ratio}"), need, |need| {
                        backend.position_block_indexer(&query, query.indexer.max_top_k, width, need)
                    })?;
                    reindex.push((ratio, op));
                }
                _ => {}
            }
        }
        let source = g.shape.candidates.source;
        let width = indexer::position_capacity(g, source)
            .ok_or_else(|| Refused::Rule(format!("layer {source} selects nothing")))?;
        let need = exact::need(FULL_BLOCK, exact::every_phase(FULL_BLOCK));
        let producer = asks.ask("indexer.candidate_block_ids", need, |need| {
            backend.indexer_candidate_block_ids(&indexer::candidates(g, FULL_BLOCK), width, need)
        })?;
        Ok(SelectOps {
            full,
            producer,
            reindex,
        })
    }

    /// A Full source's dense selection of a block of `rows` rows over keys
    /// of `width` positions.
    pub(crate) fn select(
        &self,
        ratio: u32,
        rows: u32,
        width: u32,
        views: &SelectViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .full
            .iter()
            .find(|(of, _)| *of == ratio)
            .ok_or(Refusal("DeepSeek V4.1 fully selects at no such ratio"))?;
        let mut bind = Bind::default();
        let buffers = PositionSelectBuffers {
            q: bind.view(views.q),
            head_weights: bind.view(views.head_weights),
            keys: key_pages(&mut bind, &views.keys),
            positions: bind.view(views.positions),
            row_batch: bind.view(views.row_batch),
            live_rows: bind.view(views.live_rows),
            candidate_mask: None,
            position_capacity: width,
            scores: bind.view(views.scores),
            status: bind.view(views.status),
            output: bind.view(views.output),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// The block IDs of a block of `rows` rows from its dense `scores`,
    /// each row's visible count and current tag, and the selection's
    /// `status`, into the retained `output`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn produce(
        &self,
        rows: u32,
        width: u32,
        scores: DeviceBufferSpanV1,
        visible_positions: DeviceBufferSpanV1,
        row_tags: DeviceBufferSpanV1,
        status: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
        output: CandidateViews,
        workspace: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = CandidateBlockIdsBuffers {
            scores: bind.view(scores),
            visible_positions: bind.view(visible_positions),
            row_tags: bind.view(row_tags),
            source_status: bind.view(status),
            live_rows: bind.view(live_rows),
            output: record(&mut bind, output, width),
        };
        bind.launch_within(&self.producer, &buffers, workspace, Counts::rows(rows))
    }

    /// A Reindex source's compact selection of a block of `rows` rows over
    /// keys of `width` positions, inside the retained `candidates`.
    pub(crate) fn reindex(
        &self,
        ratio: u32,
        rows: u32,
        width: u32,
        views: &SelectViews,
        row_tags: DeviceBufferSpanV1,
        candidates: CandidateViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .reindex
            .iter()
            .find(|(of, _)| *of == ratio)
            .ok_or(Refusal("DeepSeek V4.1 reindexes at no such ratio"))?;
        let mut bind = Bind::default();
        let buffers = PositionBlockSelectBuffers {
            q: bind.view(views.q),
            head_weights: bind.view(views.head_weights),
            keys: key_pages(&mut bind, &views.keys),
            positions: bind.view(views.positions),
            row_batch: bind.view(views.row_batch),
            row_tags: bind.view(row_tags),
            live_rows: bind.view(views.live_rows),
            candidates: record(&mut bind, candidates, width),
            scores: bind.view(views.scores),
            status: bind.view(views.status),
            output: bind.view(views.output),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }
}
