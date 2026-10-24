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
use kernel_core::pooled_indexer::{
    IndexerKeyFormat, PositionKeyPages, PositionQueryBuffers, PositionSelectBuffers,
};
use kernel_core::position_blocks::{
    CandidateBlockIds, CandidateBlockIdsBuffers, PositionBlockSelectBuffers,
};
use model_deepseek41_contract::indexer;
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::shape::Shape;

pub(crate) mod prefill;

pub(crate) struct IndexerOps {
    /// The query preparation every index source shares: the heads, width
    /// and rotation are the same at each.
    query: Prepared<PositionQueryBuffers<View>>,
    wide_query: Option<Prepared<PositionQueryBuffers<View>>>,
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
        target: bool,
    ) -> Result<IndexerOps, Refused> {
        let g = shape.geometry;
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let source = g.shape.csa.index_sources[0];
        let query_spec = indexer::selection(g, source, PREFILL_TOKENS)
            .ok_or_else(|| Refused::Rule(format!("{} has no index source", g.id)))?;
        let query = asks.ask("indexer.query", every, |need| {
            backend.position_indexer_query(&query_spec, need)
        })?;
        let rows = shape.verify_decode_rows();
        let wide_query = if target && rows > 32 {
            Some(asks.ask(
                "indexer.query wide verify",
                exact::need(rows, exact::verify(rows)),
                |need| crate::Kernels::wide_index_query(shape, &query_spec, need),
            )?)
        } else {
            None
        };
        Ok(IndexerOps { query, wide_query })
    }

    /// The preparation of `rows` index queries over a RoPE table of
    /// `table_rows` positions.
    pub(crate) fn prepare_query(
        &self,
        phase: crate::program::Phase,
        table_rows: u32,
        views: &QueryViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
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
        let op = if matches!(phase, crate::program::Phase::Verify { rows: 33..=64, .. }) {
            self.wide_query.as_ref()
        } else {
            None
        }
        .unwrap_or(&self.query);
        bind.launch_within(op, &buffers, views.workspace, counts)
    }
}

/// Rows of one full-capacity selection launch: its scores are 256 MiB at a
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
    pub(crate) prefill: Option<prefill::Factory>,
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
    pub workspace: DeviceBufferSpanV1,
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
        scales: if keys.scales.length == 0 {
            View::EMPTY
        } else {
            bind.view(keys.scales)
        },
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
        storage: KvStorage,
        asks: &Asks,
    ) -> Result<SelectOps, Refused> {
        let g = shape.geometry;
        let storage = match storage {
            KvStorage::Native => IndexerKeyFormat::Fp4,
            KvStorage::Bf16 => IndexerKeyFormat::Bf16,
        };
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
                        backend.position_indexer_with_storage(
                            &query,
                            storage,
                            shape.runtime.index_topk,
                            width,
                            need,
                        )
                    })?;
                    full.push((ratio, op));
                }
                Some(query) if !reindex.iter().any(|&(r, _)| r == ratio) => {
                    let need = exact::need(REINDEX_BLOCK, exact::every_phase(REINDEX_BLOCK));
                    let op = asks.ask(format!("indexer.reindex r{ratio}"), need, |need| {
                        backend.position_block_indexer_with_storage(
                            &query,
                            storage,
                            shape.runtime.index_topk,
                            width,
                            need,
                        )
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
            let query = indexer::candidates(g, FULL_BLOCK);
            if shape.runtime.candidate_topk_blocks == query.top_k_blocks {
                backend.indexer_candidate_block_ids(&query, width, need)
            } else {
                backend.indexer_candidate_block_ids_runtime(
                    &query,
                    shape.runtime.candidate_topk_blocks,
                    width,
                    need,
                )
            }
        })?;
        Ok(SelectOps {
            full,
            producer,
            reindex,
            prefill: prefill::Factory::new(shape, storage),
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
        Self::select_prepared(op, rows, width, views)
    }

    fn select_prepared(
        op: &Prepared<PositionSelectBuffers<View>>,
        rows: u32,
        width: u32,
        views: &SelectViews,
    ) -> Result<Vec<Lowered>, Refusal> {
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
        bind.launch_within(op, &buffers, views.workspace, Counts::rows(rows))
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
        Self::produce_prepared(
            &self.producer,
            rows,
            width,
            scores,
            visible_positions,
            row_tags,
            status,
            live_rows,
            output,
            workspace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn produce_prepared(
        op: &Prepared<CandidateBlockIdsBuffers<View>>,
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
        bind.launch_within(op, &buffers, workspace, Counts::rows(rows))
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
        bind.launch_within(op, &buffers, views.workspace, Counts::rows(rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;
    #[test]
    fn wide_index_query_preserves_drafter_prefill_and_default_rows() {
        use crate::program::Phase;
        use bundle_runtime::geometry::Ranks;
        use model_deepseek41_contract::geometry::FLASH;
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap().with_verify_rows(64);
        let kernels = crate::Kernels::new(&shape).unwrap();
        let span = |handle, bytes| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(handle), 0, bytes)
        };
        for target in [false, true] {
            let ops =
                IndexerOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default(), target)
                    .unwrap();
            let views = QueryViews {
                q: span(1, 65 * 4096 * 4),
                cos_sin: span(2, 8 * 64 * 4),
                positions: span(3, 65 * 4),
                live_rows: span(4, 4),
                output: span(5, 65 * 4096 * 2),
                status: span(6, 65 * 4),
                workspace: span(7, ops.query.info().workspace),
            };
            for phase in [Phase::Prefill { rows: 64 }, Phase::Decode(32)]
                .into_iter()
                .chain([32, 33, 48, 64, 65].map(|rows| Phase::Verify { items: 16, rows }))
            {
                let calls = ops.prepare_query(phase, 8, &views).unwrap();
                let fused = phase.rows() <= 32
                    || target && matches!(phase, Phase::Verify { rows: 33..=64, .. });
                assert_eq!(calls.len(), if fused { 1 } else { 2 });
                assert_eq!(
                    calls[0].launch.symbol,
                    if fused {
                        "chop_position_query_deepseek41_parallel_v1"
                    } else {
                        "chop_position_query_deepseek41_rotate_v3"
                    }
                );
            }
        }
    }

    #[test]
    fn bf16_index_pages_do_not_bind_an_absent_scale_plane() {
        let values =
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 256, 16384);
        let tables = DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(2, 1), 0, 4);
        let keys = KeyPages {
            codes: values,
            scales: DeviceBufferSpanV1::new(values.buffer, values.offset, 0),
            pages: 1,
            code_page_stride: 204800,
            scale_page_stride: 0,
            page_tables: BlockTables {
                tables,
                sequences: 1,
                max_pages: 1,
            },
        };
        let mut bind = Bind::default();
        let b = key_pages(&mut bind, &keys);
        assert_eq!(b.scales, View::EMPTY);
        assert_eq!(b.scale_page_stride, 0);
        assert_eq!(bind.0.get(b.codes.handle).unwrap(), values);
        assert_eq!(bind.0.get(b.page_tables.tables.handle).unwrap(), tables);
        assert!(bind.0.get(3).is_err(), "only values and tables are bound");
    }
}
