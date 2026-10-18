//! Lowering of the indexer's stages through `crate::ops::indexer`. Each
//! selection runs in row blocks: a dense block's scores are `[FULL_BLOCK,
//! position capacity]`, so the candidate source publishes a block's IDs
//! before the next block reuses the scores.

use kernel_core::attention::BlockTables;
use model_deepseek41_contract::indexer;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, rows};
use crate::buffers::{block_live, index_selected, index_status, scratch_rows};
use crate::ops::Lowered;
use crate::ops::indexer::{CandidateViews, FULL_BLOCK, QueryViews, REINDEX_BLOCK, SelectViews};
use crate::program::Phase;

/// Rows `first..first + rows` of `span`, `row` bytes each.
fn slice(span: DeviceBufferSpanV1, row: u64, first: u32, rows: u32) -> DeviceBufferSpanV1 {
    DeviceBufferSpanV1::new(
        span.buffer,
        span.offset + u64::from(first) * row,
        u64::from(rows) * row,
    )
}

/// Rows `first..first + rows` of `span`, a buffer of `scratch_rows()` rows.
fn block(span: DeviceBufferSpanV1, first: u32, rows: u32) -> DeviceBufferSpanV1 {
    slice(span, span.length / u64::from(scratch_rows()), first, rows)
}

impl Lowering<'_> {
    /// Layer `layer`'s index query: the F32 `wq_b` rows rotated at each
    /// row's position and fake-quantized to MXFP4.
    pub(super) fn index_query(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let (Some(ops), Some(compressor)) = (&self.ops.indexer, &self.ops.compressor) else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let views = QueryViews {
            q: rows(scratch.index_query, phase.rows()),
            cos_sin: self.buffers.rope_compressed,
            positions: self.inputs.positions,
            live_rows: self.inputs.live_tokens,
            output: rows(scratch.index_prepared, phase.rows()),
            workspace: self.buffers.workspace,
            status: rows(scratch.row_status, phase.rows()),
        };
        ops.prepare_query(phase.rows(), compressor.table_rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("index query of layer {layer}: {refusal}"))
    }

    /// Layer `layer`'s selection of each row's top-k compressed positions:
    /// a Full source over every visible key, a Reindex source inside the
    /// candidate source's blocks. BF16 pages have no index-key reader.
    pub(super) fn index_select(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.select else {
            return Ok(None);
        };
        if self.kv.storage != KvStorage::Native {
            return Ok(None);
        }
        let g = self.shape.geometry;
        let error = |refusal: kernel_core::contract::Refusal| {
            format!("index selection of layer {layer}: {refusal}")
        };
        let source = g
            .shape
            .csa
            .index_sources
            .iter()
            .position(|&l| l == layer)
            .ok_or_else(|| format!("layer {layer} is no index source"))?;
        let width = indexer::position_capacity(g, layer)
            .ok_or_else(|| format!("layer {layer} selects nothing"))?;
        let keys = self
            .source(layer)?
            .native_index_pages(BlockTables {
                tables: self.inputs.block_tables,
                sequences: self.kv.sequences,
                max_pages: self.kv.max_pages,
            })
            .map_err(error)?;
        let ratio = g.compress_ratio(layer);
        let scratch = &self.buffers.scratch;
        let top_k = g.shape.indexer.top_k;
        let reindex = indexer::block_selection(g, layer, REINDEX_BLOCK).is_some();
        let size = if reindex { REINDEX_BLOCK } else { FULL_BLOCK };
        let status = index_status(scratch.index_status, source, phase.rows());
        let selected = index_selected(scratch.index_selected, source, phase.rows(), top_k);
        let mut launches = Vec::new();
        for (index, first) in (0..phase.rows()).step_by(size as usize).enumerate() {
            let n = size.min(phase.rows() - first);
            let live_rows = block_live(scratch.block_live, size, index as u32);
            let status = slice(status, 4, first, n);
            let candidates = CandidateViews {
                ids: block(scratch.candidate_ids, first, n),
                row_tags: block(scratch.candidate_tags, first, n),
                visible_positions: block(scratch.candidate_visible, first, n),
                status: block(scratch.candidate_status, first, n),
            };
            let views = SelectViews {
                q: block(scratch.index_prepared, first, n),
                head_weights: block(scratch.index_weights, first, n),
                keys,
                positions: slice(self.inputs.positions, 4, first, n),
                row_batch: slice(self.inputs.token_batch, 4, first, n),
                live_rows,
                scores: scratch.index_scores,
                status,
                output: slice(selected, u64::from(top_k) * 4, first, n),
            };
            let row_tags = block(scratch.row_tags, first, n);
            if reindex {
                launches.extend(
                    ops.reindex(ratio, n, width, &views, row_tags, candidates)
                        .map_err(error)?,
                );
                continue;
            }
            launches.extend(ops.select(ratio, n, width, &views).map_err(error)?);
            if layer == g.shape.candidates.source {
                launches.extend(
                    ops.produce(
                        n,
                        width,
                        scratch.index_scores,
                        block(scratch.index_visible, first, n),
                        row_tags,
                        status,
                        live_rows,
                        candidates,
                        self.buffers.workspace,
                    )
                    .map_err(error)?,
                );
            }
        }
        Ok(Some(launches))
    }
}
