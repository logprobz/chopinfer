//! Lowering of the latent attention and its glue (`crate::ops::latent`)
//! over each layer's window ring in the request state and its compressed
//! source in the serving pages.
//!
//! A decode or verify step writes its window keys into the ring before its
//! queries read them. A prefill stages each layer's window: it gathers each
//! item's retained ring prefix into the item's staged block and writes its
//! chunk's keys after it, lists each row's staged cells, attends over the
//! staged window, and writes its keys into the ring after the attention.

use bundle_runtime::state::Extent;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::latent_glue;
use model_deepseek41_contract::own_ops::rope::RopeTable;
use model_deepseek41_contract::state::{self, KvStorage, Widths, ring_cells};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view, rows};
use crate::buffers::index_selected;
use crate::ops::exact::Kind;
use crate::ops::latent::{
    AttentionViews, CompressedViews, QueryRopeViews, Ring, StageViews, WindowWriteViews,
    rope_tables, staged,
};
use crate::ops::{Lowered, NormViews};
use crate::program::Phase;

/// The request state buffer that holds the window rings.
const RING_BUFFER: &str = "window.ring";

/// Layer `layer`'s window ring in the ring buffer `span` of an instance of
/// `extent`: the buffer's layers are `[layer][plane][slot]`, codes then
/// scales.
pub(crate) fn ring(
    g: &Geometry,
    storage: KvStorage,
    extent: Extent,
    span: DeviceBufferSpanV1,
    layer: u32,
) -> Result<Ring, String> {
    let declaration = state::request_state(g, storage, false);
    let buffer = declaration
        .buffers
        .iter()
        .find(|buffer| buffer.name == RING_BUFFER)
        .ok_or("the request state has no window rings")?;
    // Codes then scales, or the BF16 layout's one plane of values.
    let (codes, scales) = match buffer.planes[..] {
        [codes, scales] => (codes, Some(scales)),
        [values] => (values, None),
        _ => return Err("the window ring keeps neither code and scale planes nor values".into()),
    };
    let entries = buffer.entries(extent);
    let base = span.offset + u64::from(layer) * buffer.layer_stride(entries);
    let scale_stride = scales.unwrap_or(0);
    if layer >= g.layers() || base + entries * (codes + scale_stride) > span.offset + span.length {
        return Err(format!("layer {layer} has no window ring"));
    }
    Ok(Ring {
        codes: DeviceBufferSpanV1::new(span.buffer, base, entries * codes),
        scales: scales.map(|scales| {
            DeviceBufferSpanV1::new(span.buffer, base + entries * codes, entries * scales)
        }),
        code_stride: codes,
        scale_stride,
        slots: entries as u32,
        cells: ring_cells(Widths::of(g)) as u32,
        window: g.shape.attention.window,
    })
}

impl Lowering<'_> {
    /// Layer `layer`'s window ring on this rank.
    pub(super) fn ring(&self, layer: u32) -> Result<Ring, String> {
        let span = self
            .buffers
            .state
            .named(RING_BUFFER)
            .ok_or("the request state has no window rings")?;
        ring(
            self.shape.geometry,
            self.kv.storage,
            self.kv.extent,
            span,
            layer,
        )
    }

    /// The step rows' RoPE rows of `table`.
    fn rope(&self, phase: Phase, table: RopeTable) -> Result<DeviceBufferSpanV1, String> {
        let scratch = &self.buffers.scratch;
        self.carried(
            phase,
            match table {
                RopeTable::Window => scratch.rope_window,
                RopeTable::Compressed => scratch.rope_compressed_rows,
            },
        )
    }

    /// The step rows' RoPE rows of the table layer `layer` rotates with.
    fn layer_rope(&self, phase: Phase, layer: u32) -> Result<DeviceBufferSpanV1, String> {
        let table = latent_glue::rope_table(self.shape.geometry, layer)?;
        self.rope(phase, table)
    }

    /// Each table's rows at the step's positions.
    pub(super) fn rope_rows(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(glue) = &self.ops.glue else {
            return Ok(None);
        };
        let mut launches = Vec::new();
        for table in rope_tables(self.shape) {
            launches.extend(
                glue.rope_rows(
                    table,
                    phase.rows(),
                    self.inputs.positions,
                    self.inputs.live_tokens,
                    self.rope(phase, table)?,
                )
                .map_err(|refusal| format!("RoPE rows of the {table:?} table: {refusal}"))?,
            );
        }
        Ok(Some(launches))
    }

    /// `q_norm` of layer `layer`'s query down projection into F32 rows.
    pub(super) fn query_norm(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        if matches!(phase, Phase::Decode(_)) {
            return self.fused_query_norm(phase, layer);
        }
        let Some(glue) = &self.ops.glue else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let views = NormViews {
            x: rows(scratch.query_a, phase.rows()),
            weight: self
                .buffers
                .view(&layer_view(layer, "attn.q_norm.weight"))?,
            y: rows(scratch.query_normed, phase.rows()),
            live_rows: self.inputs.live_tokens,
            status: Some(self.carried(phase, scratch.row_status)?),
        };
        glue.query_norm(phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("q_norm of layer {layer}: {refusal}"))
    }

    /// Layer `layer`'s queries' tail RoPE into the BF16 attention query;
    /// at a one-row decode step where the twin serves
    /// ([`Self::query_window_twin`]), with the layer's window write in the
    /// same launch.
    pub(super) fn query_rope(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(glue) = &self.ops.glue else {
            return Ok(None);
        };
        let scratch = &self.buffers.scratch;
        let views = QueryRopeViews {
            query: rows(scratch.query, phase.rows()),
            native_q: rows(scratch.native_q, phase.rows()),
            cos_sin: self.layer_rope(phase, layer)?,
            live_rows: self.inputs.live_tokens,
            status: self.carried(phase, scratch.row_status)?,
        };
        if self.query_window_twin(phase, layer) {
            let window = self.ring_write_views(phase, layer, false)?.0;
            return glue
                .query_window(&views, &window)
                .map(Some)
                .map_err(|refusal| {
                    format!("query RoPE and window write of layer {layer}: {refusal}")
                });
        }
        glue.query_rope(phase, &views)
            .map(Some)
            .map_err(|refusal| format!("query RoPE of layer {layer}: {refusal}"))
    }

    /// Whether layer `layer`'s query RoPE also writes its window keys in
    /// `phase`: a one-row decode step whose dual fold wrote the KV
    /// projection before the query chain, where the twin serves.
    fn query_window_twin(&self, phase: Phase, layer: u32) -> bool {
        matches!(phase, Phase::Decode(1))
            && crate::Kernels::query_window_serves(self.shape)
            && self
                .ops
                .dense
                .as_ref()
                .and_then(|dense| dense.dual.as_ref())
                .is_some_and(|dual| dual.eligible(self.shape, phase, layer))
    }

    /// A prefill's staged window on this rank.
    fn staged(&self) -> Ring {
        let scratch = &self.buffers.scratch;
        staged(
            self.shape,
            self.kv.storage,
            scratch.stage_codes,
            scratch.stage_scales,
        )
    }

    /// A prefill's staging spans: the phase's items' slots, chunk bounds
    /// and staged block starts, and each slot's ring snapshot.
    fn stage_views(&self) -> StageViews {
        let scratch = &self.buffers.scratch;
        StageViews {
            item_slots: self.inputs.slot_ids,
            first: self.inputs.prefix_len,
            last: scratch.stage_last,
            base: scratch.stage_base,
            live_items: self.inputs.live_batch,
            ring_last: scratch.ring_last,
            positions: self.inputs.positions,
            row_items: self.inputs.token_batch,
            live_rows: self.inputs.live_tokens,
            items: self.kv.sequences,
            slots: self.kv.extent.bucket,
        }
    }

    /// The window write of layer `layer`'s rows into its ring, or with
    /// `staged` into a prefill's staged window, in the instance's KV
    /// storage.
    fn ring_write(
        &self,
        phase: Phase,
        layer: u32,
        staged: bool,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let Some(glue) = &self.ops.glue else {
            return Ok(None);
        };
        let (views, first) = self.ring_write_views(phase, layer, staged)?;
        glue.window_write(phase, &views, first)
            .map(Some)
            .map_err(|refusal| format!("window write of layer {layer}: {refusal}"))
    }

    /// The views of [`Self::ring_write`], and with `staged` each item's
    /// staged block start.
    fn ring_write_views(
        &self,
        phase: Phase,
        layer: u32,
        staged: bool,
    ) -> Result<(WindowWriteViews, Option<DeviceBufferSpanV1>), String> {
        let scratch = &self.buffers.scratch;
        let rows_of = |span| rows(span, phase.rows());
        let stage = self.stage_views();
        let (ring, row_slots, last_positions, first) = match staged {
            false => (
                self.ring(layer)?,
                self.carried(phase, scratch.window_rows)?,
                scratch.window_last,
                None,
            ),
            true => (
                self.staged(),
                self.inputs.token_batch,
                stage.last,
                Some(stage.base),
            ),
        };
        let views = WindowWriteViews {
            input: rows_of(scratch.kv),
            weight: self
                .buffers
                .view(&layer_view(layer, "attn.kv_norm.weight"))?,
            latent: rows_of(scratch.window_latent),
            codes: rows_of(scratch.window_codes),
            scales: rows_of(scratch.window_scales),
            cos_sin: self.layer_rope(phase, layer)?,
            positions: self.inputs.positions,
            row_slots,
            last_positions,
            ring,
            live_rows: self.inputs.live_tokens,
            workspace: self.buffers.workspace,
            status: self.carried(phase, scratch.row_status)?,
        };
        Ok((views, first))
    }

    /// Layer `layer`'s window keys: a decode or verify step's into the
    /// ring; a prefill's retained ring prefix gathered into its staged
    /// window, then its chunk's keys after it.
    pub(super) fn window_kv(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        match phase {
            Phase::Prefill { .. } => {
                let Some(write) = self.ring_write(phase, layer, true)? else {
                    return Ok(None);
                };
                // A part starting at zero gathers the original ring prefix.
                // Later parts retain that staged prefix and the earlier rows.
                let mut launches = if self.prefill_row_origin.is_some_and(|first| first > 0) {
                    Vec::new()
                } else {
                    let Some(prefix) = self.prefill_window_prefix(layer)? else {
                        return Ok(None);
                    };
                    prefix
                };
                launches.extend(write);
                Ok(Some(launches))
            }
            // The query RoPE's launch has written the keys.
            Phase::Decode(_) if self.query_window_twin(phase, layer) => Ok(Some(Vec::new())),
            Phase::Decode(_) | Phase::Verify { .. } => self.ring_write(phase, layer, false),
        }
    }

    /// Gather the original chunk's retained ring prefix once before any
    /// window writes, including when the first retained row is nonzero.
    /// WindowKv gathers automatically at origin zero; callers starting at
    /// a later origin issue this first with the original item metadata.
    pub fn prefill_window_prefix(&self, layer: u32) -> Result<Option<Vec<Lowered>>, String> {
        let Some(glue) = &self.ops.glue else {
            return Ok(None);
        };
        glue.stage(self.ring(layer)?, self.staged(), &self.stage_views())
            .map(Some)
            .map_err(|refusal| format!("window staging of layer {layer}: {refusal}"))
    }

    /// A prefill's window keys of layer `layer` into its ring, after its
    /// attention read the window.
    pub(super) fn window_ring_write(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        match phase {
            Phase::Prefill { .. } => self.ring_write(phase, layer, false),
            Phase::Decode(_) | Phase::Verify { .. } => Err(format!(
                "layer {layer} writes its ring before its attention"
            )),
        }
    }

    /// Each row's window key list over the ring, or a prefill's over its
    /// staged window.
    pub(super) fn window_keys(&self, phase: Phase) -> Result<Option<Vec<Lowered>>, String> {
        let Some(glue) = &self.ops.glue else {
            return Ok(None);
        };
        if phase.prefill() {
            return glue
                .staged_keys(
                    phase.rows(),
                    &self.stage_views(),
                    self.carried(phase, self.buffers.scratch.window_keys)?,
                )
                .map(Some)
                .map_err(|refusal| format!("staged window key lists: {refusal}"));
        }
        glue.window_keys(
            phase.rows(),
            self.inputs.positions,
            rows(self.buffers.scratch.window_keys, phase.rows()),
            self.inputs.live_tokens,
        )
        .map(Some)
        .map_err(|refusal| format!("window key lists: {refusal}"))
    }

    /// Layer `layer`'s attention over its window, a prefill's staged one,
    /// and, at a compressing layer, its index source's selection of its KV
    /// source's latents.
    pub(super) fn latent_attention(
        &self,
        phase: Phase,
        layer: u32,
    ) -> Result<Option<Vec<Lowered>>, String> {
        self.latent_attention_cached(phase, layer, None)
    }

    pub(super) fn latent_attention_cached(
        &self,
        phase: Phase,
        layer: u32,
        offset: Option<crate::ops::latent::CacheRange>,
    ) -> Result<Option<Vec<Lowered>>, String> {
        let g = self.shape.geometry;
        // A window-only layer has no sources; a compressing layer reads its
        // KV source at that source's ratio.
        let sources = model_deepseek41_contract::indexer::sources(g, layer);
        let ratio = sources.map_or(0, |sources| g.compress_ratio(sources.keys));
        let Some(ops) = self.ops.attention.iter().find(|ops| ops.ratio == ratio) else {
            return Ok(None);
        };
        let mut compressed = match sources {
            None => None,
            Some(sources) => {
                let source = self.source(layer)?;
                let index = g
                    .shape
                    .csa
                    .index_sources
                    .iter()
                    .position(|&l| l == sources.selection)
                    .ok_or_else(|| format!("layer {layer}'s index source selects nothing"))?;
                // The pool runs from the first page's latents to the last
                // page's scales, or its values in the BF16 layout.
                let plane = source.latent;
                let last = plane.scales.unwrap_or(plane.values);
                let end = last.offset + last.length;
                let top_k = g.shape.indexer.top_k;
                Some(CompressedViews {
                    pool: DeviceBufferSpanV1::new(
                        plane.values.buffer,
                        plane.values.offset,
                        end - plane.values.offset,
                    ),
                    pages: plane.pages,
                    page_stride: plane.page_stride,
                    tables: self.inputs.block_tables,
                    sequences: self.kv.sequences,
                    max_pages: self.kv.max_pages,
                    selected: self.carried(
                        phase,
                        index_selected(
                            self.buffers.scratch.index_selected,
                            index,
                            crate::buffers::scratch_rows(),
                            top_k,
                        ),
                    )?,
                    top_k,
                })
            }
        };
        let scratch = &self.buffers.scratch;
        // A prefill's items read their own staged blocks.
        let (mut ring, item_slots) = match phase {
            Phase::Prefill { .. } => (self.staged(), scratch.stage_items),
            Phase::Decode(_) | Phase::Verify { .. } => (self.ring(layer)?, self.inputs.slot_ids),
        };
        let decoded_cache = self.buffers.prefill_cache.filter(|_| {
            phase.prefill()
                && phase.rows() == 8192
                && self.prefill_row_origin.is_none()
                && offset.is_some_and(|r| {
                    matches!(r.offset, 0 | 8192 | 16384 | 24576)
                        && (ratio == 0 || (r.offset > 0 && r.pages > 0))
                })
        });
        if let Some(cache) = decoded_cache {
            // One live item uses staged block zero. Compressed page numbers
            // are rebased together with the native pool pointer, preserving
            // every selected address without decoding the entire serving pool.
            ring.slots = 1;
            if let Some(c) = &mut compressed {
                let range = offset.ok_or("missing decoded page range")?;
                if range.pages == 0
                    || range
                        .first_page
                        .checked_add(range.pages)
                        .is_none_or(|end| end > c.pages)
                {
                    return Err("decoded page range exceeds the serving pool".into());
                }
                let tail = c.pool.length - u64::from(c.pages - 1) * c.page_stride;
                c.pool = DeviceBufferSpanV1::new(
                    c.pool.buffer,
                    c.pool.offset + u64::from(range.first_page) * c.page_stride,
                    u64::from(range.pages - 1) * c.page_stride + tail,
                );
                c.pages = range.pages;
                c.tables = cache.tables;
                c.sequences = 1;
            }
        }
        let views = AttentionViews {
            q: rows(scratch.native_q, phase.rows()),
            row_batch: self.inputs.token_batch,
            ring,
            item_slots,
            items: if decoded_cache.is_some() {
                1
            } else {
                self.kv.sequences
            },
            window_keys: self.carried(phase, scratch.window_keys)?,
            compressed,
            sinks: self.buffers.view(&layer_view(layer, "attn.attn_sink"))?,
            cos_sin: self.layer_rope(phase, layer)?,
            output: rows(scratch.attention, phase.rows()),
            workspace: self.buffers.workspace,
            decoded_cache: decoded_cache.map(|cache| cache.data),
            // A step without a cache range (packed prompts, or one prompt
            // that is not a consecutive full chunk) takes the fused shared
            // heads at every row count: each row reads its own item's slot,
            // block table and keys, and the epilogue is the reference's
            // single-split merge. An admitted chunk keeps its measured rule.
            native_shared_heads: phase.prefill()
                && offset.is_none_or(|range| {
                    qualified_native_finish(
                        phase.rows(),
                        self.prefill_row_origin,
                        range.offset,
                        ratio,
                    )
                }),
            selected_cache: scratch.selected_cache,
            live_rows: self.inputs.live_tokens,
        };
        ops.attend(Kind::from(phase), phase.rows(), &views)
            .map(Some)
            .map_err(|refusal| format!("attention of layer {layer}: {refusal}"))
    }
}

/// `offset` comes from the admitted single-item, consecutive full chunk.
/// Parts keep that origin while their query, RoPE and output views are local.
fn qualified_native_finish(rows: u32, first: Option<u32>, offset: u32, ratio: u32) -> bool {
    if rows == 8192 && first.is_none() {
        return offset == 0;
    }
    if !matches!(offset, 0 | 8192 | 16384 | 24576) {
        return false;
    }
    (matches!(offset, 0 | 24576) && rows == 4096 && matches!(first, Some(0 | 4096)))
        || (ratio == 1
            && (64..=2496).contains(&rows)
            && rows % 128 == 64
            && first.and_then(|first| first.checked_add(rows)) == Some(8192))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::state::Holder;
    use model_deepseek41_contract::geometry::FLASH;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    #[test]
    fn native_finish_admits_only_measured_offsets_counts_and_origins() {
        for offset in [0, 8192, 16384, 24576, 32768] {
            for ratio in [1, 2] {
                assert_eq!(
                    qualified_native_finish(8192, None, offset, ratio),
                    offset == 0
                );
                for first in [0, 4096] {
                    assert_eq!(
                        qualified_native_finish(4096, Some(first), offset, ratio),
                        matches!(offset, 0 | 24576)
                    );
                }
                for rows in (64..=2560).step_by(64) {
                    assert_eq!(
                        qualified_native_finish(rows, Some(8192 - rows), offset, ratio),
                        ratio == 1
                            && matches!(offset, 0 | 8192 | 16384 | 24576)
                            && (64..=2496).contains(&rows)
                            && rows % 128 == 64
                    );
                }
            }
        }
        for (rows, first) in [
            (320, None),
            (320, Some(0)),
            (4096, Some(1)),
            (448, Some(u32::MAX)),
        ] {
            assert!(!qualified_native_finish(rows, first, 0, 1));
        }
    }

    /// Each layer's ring views address exactly the pieces the request
    /// state declares for a slot's ring: its codes, then its scales, or
    /// the BF16 layout's one plane of values with no scales.
    #[test]
    fn ring_views_address_the_declared_ring_pieces() {
        let extent = Extent {
            served: 3,
            bucket: 4,
            verify_rows: 6,
        };
        let latent = u64::from(FLASH.shape.attention.head_dim);
        for (storage, cell) in [(KvStorage::Native, latent), (KvStorage::Bf16, 2 * latent)] {
            let declaration = state::request_state(&FLASH, storage, false);
            let index = declaration
                .buffers
                .iter()
                .position(|buffer| buffer.name == RING_BUFFER)
                .unwrap();
            let bytes = declaration.buffers[index].bytes(extent);
            let span =
                DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, 1), 4096, bytes);
            let planes = declaration.buffers[index].planes.len();
            for slot in 0..extent.bucket {
                let pieces: Vec<_> = declaration
                    .pieces(extent, Holder::Slot(slot))
                    .unwrap()
                    .into_iter()
                    .filter(|piece| piece.buffer == index)
                    .collect();
                assert_eq!(pieces.len(), planes * FLASH.layers() as usize);
                for layer in 0..FLASH.layers() {
                    let ring = ring(&FLASH, storage, extent, span, layer).unwrap();
                    assert_eq!((ring.slots, ring.cells, ring.window), (4, 136, 128));
                    assert_eq!(ring.code_stride, 136 * cell, "{storage:?}");
                    let mut views = vec![(ring.codes, ring.code_stride)];
                    views.extend(ring.scales.map(|scales| (scales, ring.scale_stride)));
                    assert_eq!(views.len(), planes, "{storage:?}");
                    for (plane, (view, stride)) in views.into_iter().enumerate() {
                        let piece = &pieces[planes * layer as usize + plane];
                        let offset = view.offset - span.offset + u64::from(slot) * stride;
                        assert_eq!(
                            (piece.offset, piece.length),
                            (offset, stride),
                            "{storage:?} layer {layer}"
                        );
                    }
                }
            }
            assert!(ring(&FLASH, storage, extent, span, FLASH.layers()).is_err());
        }
    }
}
