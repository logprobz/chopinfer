//! Lowering of the indexer's stages through `crate::ops::indexer`. Native
//! eager full-index batches use a bounded score stride. Every source finishes
//! selection and candidate publication before the next batch reuses scores.

use kernel_core::attention::BlockTables;
use model_deepseek41_contract::indexer;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, rows};
use crate::buffers::{block_live, index_selected, index_status, scratch_rows};
use crate::indexer_workloads::Route;
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
            status: self.carried(phase, scratch.row_status)?,
        };
        ops.prepare_query(phase.rows(), compressor.table_rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("index query of layer {layer}: {refusal}"))
    }

    /// Layer `layer`'s selection of each row's top-k compressed positions:
    /// a Full source over every visible key, a Reindex source inside the
    /// candidate source's blocks, using the selected KV storage layout.
    pub(super) fn index_select(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        self.index_select_inner(phase, layer, None, Route::Accepted)
    }

    pub(crate) fn index_select_workload(
        &self,
        phase: Phase,
        layer: u32,
        route: Route,
    ) -> Result<Option<Vec<Lowered>>, String> {
        self.index_select_inner(phase, layer, None, route)
    }

    pub(super) fn index_select_prefill(
        &self,
        phase: Phase,
        layer: u32,
        positions: &[i32],
    ) -> Result<Option<Vec<Lowered>>, String> {
        if !phase.prefill() || positions.len() != phase.rows() as usize {
            return Err("prefill index metadata does not match its local query rows".into());
        }
        self.index_select_inner(phase, layer, Some(positions), Route::Accepted)
    }

    fn index_select_inner(
        &self,
        phase: Phase,
        layer: u32,
        positions: Option<&[i32]>,
        route: Route,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(ops) = &self.ops.select else {
            return Ok(None);
        };
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
            .index_pages(BlockTables {
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
        let factory = ops
            .prefill
            .as_ref()
            .filter(|_| !reindex && positions.is_some());
        let batches = if let Some(factory) = factory {
            factory
                .batches(layer, positions.unwrap(), scratch.index_scores.length)
                .map_err(error)?
        } else {
            (0..phase.rows())
                .step_by(size as usize)
                .map(|first| indexer::prefill::Batch {
                    rows: first..(first + size).min(phase.rows()),
                    width,
                })
                .collect()
        };
        let origin = self.prefill_row_origin.filter(|_| phase.prefill());
        let status = self.carried(
            phase,
            index_status(scratch.index_status, source, scratch_rows()),
        )?;
        let selected = self.carried(
            phase,
            index_selected(scratch.index_selected, source, scratch_rows(), top_k),
        )?;
        let mut launches = Vec::new();
        let hierarchy =
            crate::Kernels::hierarchical_indexer_enabled(self.shape, self.kv.storage, phase)
                || route != Route::Accepted
                    && matches!(phase, Phase::Verify { rows: 32, items: 8 });
        // The fused publisher and the early-stop selectors were qualified
        // beside the bounded scoring tiles, under the same admission.
        let fused_candidates = phase.prefill()
            && crate::Kernels::prefill_tile_enabled(self.shape, self.kv.storage, positions);
        let select = |mut calls| {
            if phase.prefill()
                && crate::Kernels::prefill_tile_enabled(self.shape, self.kv.storage, positions)
            {
                calls = crate::Kernels::prefill_tile_scores(calls, reindex).map_err(error)?;
                calls = crate::Kernels::early_prefill_select(calls).map_err(error)?;
            }
            if route == Route::Combined && !reindex {
                calls = crate::Kernels::tiled_indexer_scores(calls).map_err(error)?;
                calls = crate::Kernels::cached_prefill_indexer_select(calls).map_err(error)?;
            }
            if hierarchy {
                crate::Kernels::hierarchical_indexer_select(
                    calls,
                    scratch.index_scores,
                    phase.rows(),
                    width,
                )
                .map_err(error)
            } else {
                Ok(calls)
            }
        };
        // A phase has at most two batch sizes (full blocks and a final tail).
        // Keep their prepared ops local to this eager lowering, never cached
        // across a changed position upload or a decode graph replay.
        let mut prepared = std::collections::BTreeMap::new();
        for batch in batches {
            let first = batch.rows.start;
            let n = batch.rows.end - first;
            let width = batch.width;
            let global_first = origin
                .unwrap_or(0)
                .checked_add(first)
                .ok_or("index row origin overflows")?;
            let live_rows = if factory.is_some() {
                crate::buffers::index_prefill_count(scratch.block_live, n)
                    .map_err(|e| e.to_string())?
            } else if origin.is_some() && reindex {
                // A micro-batch is at most one reindex block. Its stable
                // count differs from the full chunk's block count.
                if phase.rows() > REINDEX_BLOCK {
                    return Err("prefill part exceeds one reindex block".into());
                }
                self.inputs.live_tokens
            } else {
                if !global_first.is_multiple_of(size) {
                    return Err("prefill row origin splits an index block".into());
                }
                block_live(scratch.block_live, size, global_first / size)
            };
            let status = slice(status, 4, first, n);
            let candidates = CandidateViews {
                ids: block(scratch.candidate_ids, global_first, n),
                row_tags: block(scratch.candidate_tags, global_first, n),
                visible_positions: block(scratch.candidate_visible, global_first, n),
                status: block(scratch.candidate_status, global_first, n),
            };
            let views = SelectViews {
                workspace: self.buffers.workspace,
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
            let row_tags = block(scratch.row_tags, global_first, n);
            if reindex {
                launches.extend(select(
                    ops.reindex(ratio, n, width, &views, row_tags, candidates)
                        .map_err(error)?,
                )?);
                continue;
            }
            if let Some(factory) = factory {
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    prepared.entry((n, width))
                {
                    entry.insert(factory.prepare(
                        layer,
                        n,
                        width,
                        self.buffers.workspace.length,
                    )?);
                }
                let op = &prepared[&(n, width)];
                launches.extend(select(op.select(n, width, &views).map_err(error)?)?);
                if layer == g.shape.candidates.source {
                    let calls = op
                        .produce(
                            n,
                            width,
                            &views,
                            block(scratch.index_visible, global_first, n),
                            row_tags,
                            candidates,
                        )
                        .map_err(error)?;
                    launches.extend(if fused_candidates {
                        crate::Kernels::cached_prefill_candidates(calls).map_err(error)?
                    } else {
                        calls
                    });
                }
                continue;
            }
            launches.extend(select(ops.select(ratio, n, width, &views).map_err(error)?)?);
            if layer == g.shape.candidates.source {
                launches.extend(
                    ops.produce(
                        n,
                        width,
                        scratch.index_scores,
                        block(scratch.index_visible, global_first, n),
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
