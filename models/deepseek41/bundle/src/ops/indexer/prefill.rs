//! Eager-only preparation of the qualified guarded indexer at a bounded stride.
use super::*;
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub(crate) fn enabled(shape: &Shape, storage: KvStorage) -> bool {
    shape.geometry == &FLASH && storage == KvStorage::Native && shape.runtime.indexer_mma
}

pub(crate) struct Factory {
    backend: Box<dyn Backend>,
    geometry: &'static Geometry,
    top_k: u32,
    candidate_top_k: u32,
}

pub(crate) struct BatchOps {
    select: Prepared<PositionSelectBuffers<View>>,
    producer: Option<Prepared<CandidateBlockIdsBuffers<View>>>,
}

impl Factory {
    pub(super) fn new(shape: &Shape, storage: IndexerKeyFormat) -> Option<Self> {
        if storage != IndexerKeyFormat::Fp4 || !enabled(shape, KvStorage::Native) {
            return None;
        }
        Some(Self {
            backend: crate::Kernels::prefill_indexer_backend(shape),
            geometry: shape.geometry,
            top_k: shape.runtime.index_topk,
            candidate_top_k: shape.runtime.candidate_topk_blocks,
        })
    }

    pub(crate) fn batches(
        &self,
        layer: u32,
        positions: &[i32],
        score_bytes: u64,
    ) -> Result<Vec<indexer::prefill::Batch>, Refusal> {
        if indexer::block_selection(self.geometry, layer, 1).is_some() {
            return Err(Refusal("bounded prefill requires a full index source"));
        }
        let max_rows = if layer == self.geometry.shape.candidates.source {
            512
        } else {
            4096
        };
        indexer::prefill::batches(
            positions,
            self.geometry.compress_ratio(layer),
            indexer::position_capacity(self.geometry, layer)
                .ok_or(Refusal("no full index source"))?,
            score_bytes.min(indexer::prefill::SCORE_BYTES),
            max_rows,
        )
    }

    pub(crate) fn prepare(
        &self,
        layer: u32,
        rows: u32,
        width: u32,
        workspace: u64,
    ) -> Result<BatchOps, String> {
        if indexer::block_selection(self.geometry, layer, rows).is_some() {
            return Err("bounded prefill requires a full index source".into());
        }
        let query = indexer::selection(self.geometry, layer, rows).ok_or("no index source")?;
        let need = exact::need(rows, exact::every_phase(rows));
        let select = self
            .backend
            .position_indexer(&query, self.top_k, width, need)
            .map_err(|e| e.to_string())?;
        let producer = if layer == self.geometry.shape.candidates.source {
            let query = indexer::candidates(self.geometry, rows);
            Some(
                if self.candidate_top_k == query.top_k_blocks {
                    self.backend
                        .indexer_candidate_block_ids(&query, width, need)
                } else {
                    self.backend.indexer_candidate_block_ids_runtime(
                        &query,
                        self.candidate_top_k,
                        width,
                        need,
                    )
                }
                .map_err(|e| e.to_string())?,
            )
        } else {
            None
        };
        if select.info().workspace > workspace
            || producer
                .as_ref()
                .is_some_and(|op| op.info().workspace > workspace)
        {
            return Err("bounded prefill exceeds the existing indexer workspace".into());
        }
        Ok(BatchOps { select, producer })
    }
}

impl BatchOps {
    pub(crate) fn select(
        &self,
        rows: u32,
        width: u32,
        views: &SelectViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        SelectOps::select_prepared(&self.select, rows, width, views)
    }

    pub(crate) fn produce(
        &self,
        rows: u32,
        width: u32,
        views: &SelectViews,
        visible: DeviceBufferSpanV1,
        tags: DeviceBufferSpanV1,
        output: CandidateViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        SelectOps::produce_prepared(
            self.producer
                .as_ref()
                .ok_or(Refusal("no prefill candidate producer"))?,
            rows,
            width,
            views.scores,
            visible,
            tags,
            views.status,
            views.live_rows,
            output,
            views.workspace,
        )
    }
}
