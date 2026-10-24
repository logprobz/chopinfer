//! The latent attention's ops. The glue around it: `q_norm` of the query
//! down projection into F32 rows, each step's RoPE rows at its positions,
//! the queries' tail RoPE into the BF16 attention query, the window write
//! (`kv_norm`, the tail RoPE and the ceil E4M3/32 quantization of the
//! window key into the layer's ring) and the window key lists. Then the
//! native attention over the ring and, at a compressing layer, the
//! compressed positions its index source selected, with the sink and the
//! output's tail rotated back.
//!
//! The ring has `window + RING_HEADROOM` cells, so a decode or verify step
//! writes its rows before its queries read them: no row overwrites a cell
//! another query of the step sees. A prefill chunk is wider than the
//! headroom, so a prefill stages its window: each item's retained ring
//! prefix is gathered into the item's staged block and its chunk's rows
//! written after it; its key lists mask the prefix cells the ring no longer
//! holds; it attends over the staged block; then it writes its rows into the
//! ring. The gather and the lists read the same ring snapshot: each slot's
//! last position before the step.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::contract::{AttentionPhase, Refusal};
use kernel_core::latent_attention::{
    BlockTables, LatentAttentionQuery, LatentCompressedBuffers, LatentKvCache,
    LatentSourcesBuffers, LatentWindowBuffers, LatentWindowStorage,
};
use kernel_core::latent_glue::{
    LatentQueryRopeBuffers, LatentWindowKeyBuffers, LatentWindowWriteBuffers, WindowAddressing,
};
use kernel_core::latent_sources_cache::{
    Buffers as SelectedCacheBuffers, Layout as SelectedCacheLayout,
};
use kernel_core::latent_staged_window_keys::LatentStagedWindowKeyBuffers;
use kernel_core::latent_window_stage::LatentWindowStageBuffers;
use kernel_core::rmsnorm::RmsNormBuffers;
use model_deepseek41_contract::latent::{self, Keys};
use model_deepseek41_contract::latent_glue;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::rope::{
    RopeCosSin, RopeCosSinBuffers, RopeRows, RopeTable,
};
use model_deepseek41_contract::serving::{DECODE_BATCH, PREFILL_TOKENS};
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::exact::{self, Kind};
use super::row_buckets::RowBuckets;
use super::{Asks, Bind, Lowered, NormViews};
use crate::shape::Shape;

/// The RoPE tables a geometry's layers rotate with: the window table at
/// its ratio-0 layers, the compressed table at every other.
pub(crate) fn rope_tables(shape: &Shape) -> Vec<RopeTable> {
    let g = shape.geometry;
    let mut tables = Vec::new();
    for layer in 0..g.layers() {
        let table = match g.compress_ratio(layer) {
            0 => RopeTable::Window,
            _ => RopeTable::Compressed,
        };
        if !tables.contains(&table) {
            tables.push(table);
        }
    }
    tables
}

/// The glue's ops, each prepared for a prefill's rows.
pub(crate) struct GlueOps {
    query_norm: Prepared<RmsNormBuffers<View>>,
    query_rope: Prepared<LatentQueryRopeBuffers<View>>,
    query_rope_small: Prepared<LatentQueryRopeBuffers<View>>,
    query_rope_wide: Option<Prepared<LatentQueryRopeBuffers<View>>>,
    /// The circular write into the ring and the circular key lists.
    window_write: RowBuckets<LatentWindowWriteBuffers<View>>,
    decode_window_write: Prepared<LatentWindowWriteBuffers<View>>,
    window_keys: Prepared<LatentWindowKeyBuffers<View>>,
    /// A prefill's gather of the retained ring prefix into its staged
    /// window, its write into the staged window and its staged key lists.
    stage: Prepared<LatentWindowStageBuffers<View>>,
    staged_write: Prepared<LatentWindowWriteBuffers<View>>,
    staged_keys: Prepared<LatentStagedWindowKeyBuffers<View>>,
    /// Each table's rows at a step's positions.
    rope_rows: Vec<(RopeTable, Prepared<RopeCosSinBuffers<View>>)>,
}

/// A step's spans of the query RoPE: the F32 queries, rotated in place,
/// the BF16 attention query and the layer's RoPE rows.
pub(crate) struct QueryRopeViews {
    pub query: DeviceBufferSpanV1,
    pub native_q: DeviceBufferSpanV1,
    pub cos_sin: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    /// The phase's row status, which a nonfinite query or rotation flags.
    pub status: DeviceBufferSpanV1,
}

/// A layer's window ring on one rank: its code and scale planes, each
/// slot's cells at its stride, the cells of a slot and the visible window.
/// A BF16 ring is one plane of BF16 values in `codes`, with no scales.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Ring {
    pub codes: DeviceBufferSpanV1,
    pub scales: Option<DeviceBufferSpanV1>,
    pub code_stride: u64,
    pub scale_stride: u64,
    pub slots: u32,
    pub cells: u32,
    pub window: u32,
}

/// A prefill's staged window on one rank: one block of `cells` cells an
/// item, read as a ring of `slots` sequences, one per item.
pub(crate) fn staged(
    shape: &Shape,
    storage: KvStorage,
    codes: DeviceBufferSpanV1,
    scales: Option<DeviceBufferSpanV1>,
) -> Ring {
    let cells = crate::buffers::staged_slots(shape);
    let latent = u64::from(shape.geometry.shape.attention.head_dim);
    Ring {
        codes,
        scales: if storage == KvStorage::Native {
            scales
        } else {
            None
        },
        code_stride: u64::from(cells) * latent * if storage == KvStorage::Bf16 { 2 } else { 1 },
        scale_stride: if storage == KvStorage::Bf16 {
            0
        } else {
            u64::from(cells) * latent / 32
        },
        slots: DECODE_BATCH,
        cells,
        window: shape.geometry.shape.attention.window,
    }
}

/// A prefill's spans of the staging: each item's state slot, chunk first
/// and last position and staged block start, the live item count, each
/// state slot's last ring position before the step, each row's position
/// and item, the live row count, and the items and state slots.
pub(crate) struct StageViews {
    pub item_slots: DeviceBufferSpanV1,
    pub first: DeviceBufferSpanV1,
    pub last: DeviceBufferSpanV1,
    pub base: DeviceBufferSpanV1,
    pub live_items: DeviceBufferSpanV1,
    pub ring_last: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub row_items: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub items: u32,
    pub slots: u32,
}

/// A step's spans of the window write: the F32 window key rows, the
/// `kv_norm` weight, the F32 normalized and rotated rows and their codes
/// and scales, the layer's RoPE rows, each row's position and state slot,
/// each slot's last position in the step, and the layer's ring.
pub(crate) struct WindowWriteViews {
    pub input: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub latent: DeviceBufferSpanV1,
    pub codes: DeviceBufferSpanV1,
    pub scales: DeviceBufferSpanV1,
    pub cos_sin: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub row_slots: DeviceBufferSpanV1,
    pub last_positions: DeviceBufferSpanV1,
    pub ring: Ring,
    pub live_rows: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
    /// The phase's row status (`Scratch::row_status`).
    pub status: DeviceBufferSpanV1,
}

impl GlueOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: &Shape,
        asks: &Asks,
        storage: KvStorage,
        target: bool,
    ) -> Result<GlueOps, Refused> {
        let g = shape.geometry;
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(shape.verify_rows));
        let query_norm = latent_glue::query_norm(g);
        let query_norm = asks.ask("glue.q_norm", every, |need| {
            backend.rmsnorm(&query_norm, need)
        })?;
        let query = latent_glue::query_rope(g, shape.ranks()).map_err(Refused::Rule)?;
        let query_rope = asks.ask("glue.query_rope", every, |need| {
            backend.latent_query_rope(&query, need)
        })?;
        let small = crate::Kernels::latent_query_small_rows();
        let query_rope_small = asks.ask(
            "glue.query_rope small rows",
            exact::need(small, exact::every_phase(small)),
            |need| backend.latent_query_rope(&query, need),
        )?;
        let wide_rows = shape.verify_decode_rows();
        let query_rope_wide = if target && wide_rows > small {
            Some(asks.ask(
                "glue.query_rope wide verify",
                exact::need(wide_rows, exact::verify(wide_rows)),
                |need| crate::Kernels::wide_query_rope(shape, &query, need),
            )?)
        } else {
            None
        };
        let ring = latent_glue::ring(g);
        let write = latent_glue::window_write_for_storage(g, ring, storage);
        let window_write = RowBuckets::prepare(asks, "glue.window_write", every, shape, |need| {
            backend.latent_window_write(&write, need)
        })?;
        let decode_window_write = asks.ask(
            "glue.decode_window_write",
            exact::need(DECODE_BATCH, exact::every_phase(DECODE_BATCH)),
            |need| backend.latent_window_write_fused(&write, need),
        )?;
        let keys = latent_glue::window_keys(g, ring);
        let window_keys = asks.ask("glue.window_keys", every, |need| {
            backend.latent_window_keys(&keys, need)
        })?;
        // A prefill's staging: its items, then its rows.
        let cells = crate::buffers::staged_slots(shape);
        let stage =
            latent_glue::window_stage_for_storage(g, cells, storage).map_err(Refused::Rule)?;
        let prefill = |capacity| exact::need(capacity, exact::PREFILL);
        let stage = asks.ask("glue.window_stage", prefill(DECODE_BATCH), |need| {
            backend.latent_window_stage(&stage, need)
        })?;
        let write = latent_glue::window_write_for_storage(
            g,
            WindowAddressing::Staged { slots: cells },
            storage,
        );
        let staged_write = asks.ask(
            "glue.staged_window_write",
            prefill(PREFILL_TOKENS),
            |need| backend.latent_window_write(&write, need),
        )?;
        let keys = latent_glue::staged_window_keys(g, cells).map_err(Refused::Rule)?;
        let staged_keys = asks.ask("glue.staged_window_keys", prefill(PREFILL_TOKENS), |need| {
            backend.latent_staged_window_keys(&keys, need)
        })?;
        let rope_rows = rope_tables(shape)
            .into_iter()
            .map(|table| {
                let op = asks.ask(format!("glue.rope_rows {table:?}"), every, |need| {
                    own.rope_cos_sin(
                        &RopeCosSin {
                            table,
                            rows: RopeRows::Positions,
                        },
                        need,
                    )
                })?;
                Ok((table, op))
            })
            .collect::<Result<_, Refused>>()?;
        Ok(GlueOps {
            query_norm,
            query_rope,
            query_rope_small,
            query_rope_wide,
            window_write,
            decode_window_write,
            window_keys,
            stage,
            staged_write,
            staged_keys,
            rope_rows,
        })
    }

    /// `q_norm` of `rows` query-down rows into F32 rows.
    pub(crate) fn query_norm(&self, rows: u32, views: &NormViews) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = RmsNormBuffers {
            status: views.status.map(|status| bind.view(status)),
            status_row_offset: 0,
            x: bind.view(views.x),
            weight: bind.view(views.weight),
            residual: None,
            y: bind.view(views.y),
            live_rows: bind.view(views.live_rows),
        };
        let counts = Counts {
            padded_rows: rows,
            ..Counts::rows(rows)
        };
        bind.launch(&self.query_norm, &buffers, counts)
    }

    /// Table `table`'s rows at `rows` rows' `positions` into `output`.
    pub(crate) fn rope_rows(
        &self,
        table: RopeTable,
        rows: u32,
        positions: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .rope_rows
            .iter()
            .find(|(of, _)| *of == table)
            .ok_or(Refusal("DeepSeek V4.1 rotates with no such table"))?;
        let mut bind = Bind::default();
        let buffers = RopeCosSinBuffers {
            positions: Some(bind.view(positions)),
            live_rows: Some(bind.view(live_rows)),
            first: 0,
            output: bind.view(output),
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// The tail RoPE of `rows` rows of queries into the BF16 query.
    pub(crate) fn query_rope(
        &self,
        phase: crate::program::Phase,
        views: &QueryRopeViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
        let mut bind = Bind::default();
        let buffers = query_rope_buffers(&mut bind, views);
        let op = if matches!(phase, crate::program::Phase::Verify { rows: 49..=64, .. }) {
            self.query_rope_wide.as_ref()
        } else {
            None
        }
        .unwrap_or_else(|| self.query_rope_op(rows));
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    fn query_rope_op(&self, rows: u32) -> &Prepared<LatentQueryRopeBuffers<View>> {
        if rows <= crate::Kernels::latent_query_small_rows() {
            &self.query_rope_small
        } else {
            &self.query_rope
        }
    }

    /// A one-row decode step's query RoPE and window write into the ring
    /// as one launch (`crate::Kernels::query_window_launch`), from the
    /// served launches of both over one binding.
    pub(crate) fn query_window(
        &self,
        rope: &QueryRopeViews,
        window: &WindowWriteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        if (rope.live_rows, rope.status) != (window.live_rows, window.status) {
            return Err(Refusal(
                "the query window twin needs one live count and status row for both halves",
            ));
        }
        let counts = Counts::rows(1);
        let mut bind = Bind::default();
        let rope_buffers = query_rope_buffers(&mut bind, rope);
        let shared = (rope_buffers.live_rows, rope_buffers.status);
        let window_buffers = window_write_buffers(&mut bind, window, None, Some(shared));
        if self.decode_window_write.info().workspace != 0 {
            return Err(Refusal("the one-row window write takes no workspace"));
        }
        let rope = self
            .query_rope_op(1)
            .launches(&rope_buffers, View::EMPTY, counts)?;
        let window = self
            .decode_window_write
            .launches(&window_buffers, View::EMPTY, counts)?;
        let launch = crate::Kernels::query_window_launch(rope, window)?;
        Ok(bind.lowered_launches(vec![launch]))
    }

    /// The window write of `rows` rows into the ring, or with `first`,
    /// each item's staged block start, into a prefill's staged window
    /// (`row_slots` then names each row's item and `last_positions` each
    /// item's last position).
    pub(crate) fn window_write(
        &self,
        phase: crate::program::Phase,
        views: &WindowWriteViews,
        first: Option<DeviceBufferSpanV1>,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
        let mut bind = Bind::default();
        let buffers = window_write_buffers(&mut bind, views, first, None);
        let op = match first {
            None if matches!(phase, crate::program::Phase::Decode(_)) => &self.decode_window_write,
            None => self.window_write.at_phase(phase),
            Some(_) => &self.staged_write,
        };
        bind.launch_within(op, &buffers, views.workspace, Counts::rows(rows))
    }

    /// A prefill's gather of each item's retained ring prefix into its
    /// staged block.
    pub(crate) fn stage(
        &self,
        ring: Ring,
        staged: Ring,
        views: &StageViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = LatentWindowStageBuffers {
            ring: ring_buffers(&mut bind, ring, Some((views.item_slots, views.items)), None),
            staged: ring_buffers(&mut bind, staged, None, None),
            first_positions: bind.view(views.first),
            last_positions: bind.view(views.ring_last),
            live_items: bind.view(views.live_items),
        };
        bind.launch(&self.stage, &buffers, Counts::rows(views.items))
    }

    /// A prefill's `rows` rows' key lists over its staged window: each
    /// row's visible cells oldest first, a prefix cell the ring snapshot
    /// no longer holds a hole.
    pub(crate) fn staged_keys(
        &self,
        rows: u32,
        views: &StageViews,
        key_lists: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = LatentStagedWindowKeyBuffers {
            positions: bind.view(views.positions),
            row_items: bind.view(views.row_items),
            chunk_first: bind.view(views.first),
            chunk_last: bind.view(views.last),
            item_slots: bind.view(views.item_slots),
            ring_last: bind.view(views.ring_last),
            items: views.items,
            sequences: views.slots,
            key_lists: bind.view(key_lists),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(&self.staged_keys, &buffers, Counts::rows(rows))
    }

    /// The window key lists of `rows` rows at `positions`: each row's
    /// visible ring cells, oldest first.
    pub(crate) fn window_keys(
        &self,
        rows: u32,
        positions: DeviceBufferSpanV1,
        key_lists: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
    ) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = LatentWindowKeyBuffers {
            positions: bind.view(positions),
            row_batch: None,
            first_positions: None,
            sequences: 0,
            key_lists: bind.view(key_lists),
            live_rows: bind.view(live_rows),
        };
        bind.launch(&self.window_keys, &buffers, Counts::rows(rows))
    }
}

/// The query RoPE's buffers over `views`.
fn query_rope_buffers(bind: &mut Bind, views: &QueryRopeViews) -> LatentQueryRopeBuffers<View> {
    LatentQueryRopeBuffers {
        query: bind.view(views.query),
        native_q: bind.view(views.native_q),
        cos_sin: bind.view(views.cos_sin),
        live_rows: bind.view(views.live_rows),
        status: bind.view(views.status),
        status_row_offset: 0,
    }
}

/// The window write's buffers over `views`, staged from `first` when given,
/// with the live count and status views `shared` when another op of the
/// launch has bound them.
fn window_write_buffers(
    bind: &mut Bind,
    views: &WindowWriteViews,
    first: Option<DeviceBufferSpanV1>,
    shared: Option<(View, View)>,
) -> LatentWindowWriteBuffers<View> {
    let window = ring_buffers(bind, views.ring, None, None);
    LatentWindowWriteBuffers {
        input: bind.view(views.input),
        weight: bind.view(views.weight),
        latent: bind.view(views.latent),
        cos_sin: bind.view(views.cos_sin),
        positions: bind.view(views.positions),
        row_batch: bind.view(views.row_slots),
        first_positions: first.map(|first| bind.view(first)),
        last_positions: bind.view(views.last_positions),
        window,
        quantized_codes: bind.view(views.codes),
        quantized_scales: bind.view(views.scales),
        live_rows: shared.map_or_else(|| bind.view(views.live_rows), |(live, _)| live),
        status: shared.map_or_else(|| bind.view(views.status), |(_, status)| status),
        status_row_offset: 0,
    }
}

/// The ring's views as the attention, the window write and the staging
/// read them: the ring as staged storage of its cells, each item's slot in
/// `item_slots` (the attention's and the gather's; the write addresses
/// slots by row), and the key lists (the attention's).
fn ring_buffers(
    bind: &mut Bind,
    ring: Ring,
    items: Option<(DeviceBufferSpanV1, u32)>,
    key_lists: Option<DeviceBufferSpanV1>,
) -> LatentWindowBuffers<View> {
    let (item_slots, items) = match items {
        Some((item_slots, items)) => (bind.view(item_slots), items),
        None => (View::EMPTY, 0),
    };
    let (key_lists, lists) = match key_lists {
        Some(key_lists) => (bind.view(key_lists), ring.window),
        None => (View::EMPTY, 0),
    };
    LatentWindowBuffers {
        codes: bind.view(ring.codes),
        scales: ring.scales.map_or(View::EMPTY, |scales| bind.view(scales)),
        item_slots,
        items,
        sequences: ring.slots,
        code_stride: ring.code_stride,
        scale_stride: ring.scale_stride,
        storage: LatentWindowStorage::Staged { slots: ring.cells },
        key_lists,
        key_stride: lists,
        budget: lists,
    }
}

/// The compressed source a layer's attention reads, as the attention binds
/// it: the source's latent plane in the serving pages, the block tables,
/// and its index source's selection.
pub(crate) struct CompressedViews {
    pub pool: DeviceBufferSpanV1,
    pub pages: u32,
    pub page_stride: u64,
    pub tables: DeviceBufferSpanV1,
    pub sequences: u32,
    pub max_pages: u32,
    pub selected: DeviceBufferSpanV1,
    pub top_k: u32,
}

/// One eager prefill's bounded physical-page span.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CacheRange {
    pub offset: u32,
    pub first_page: u32,
    /// Zero retains the paired mixed path when physical pages are too scattered.
    pub pages: u32,
}

/// A step's spans of one layer's attention.
pub(crate) struct AttentionViews {
    pub q: DeviceBufferSpanV1,
    pub row_batch: DeviceBufferSpanV1,
    pub ring: Ring,
    pub item_slots: DeviceBufferSpanV1,
    pub items: u32,
    pub window_keys: DeviceBufferSpanV1,
    pub compressed: Option<CompressedViews>,
    pub sinks: DeviceBufferSpanV1,
    pub cos_sin: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
    pub decoded_cache: Option<DeviceBufferSpanV1>,
    /// An admitted native full chunk, part or retained tail uses fused heads.
    pub native_shared_heads: bool,
    pub selected_cache: Option<DeviceBufferSpanV1>,
    pub live_rows: DeviceBufferSpanV1,
}

/// The attention of the layers whose compressed source has one ratio (0
/// for the window-only layers), one op per phase kind.
pub(crate) struct AttentionOps {
    pub ratio: u32,
    ops: Vec<(Kind, Prepared<LatentSourcesBuffers<View>>)>,
    cached: Option<Prepared<model_deepseek41_contract::own_ops::prefill_cache::Buffers>>,
    shared_heads: Option<Prepared<LatentSourcesBuffers<View>>>,
    selected: Vec<(Kind, Prepared<SelectedCacheBuffers<View>>)>,
}

/// The attention query of the layers of compress ratio `ratio` in `phase`:
/// a compressed source's page holds `PAGE_TOKENS / ratio` positions.
pub(crate) fn attention_query(
    shape: &Shape,
    ratio: u32,
    phase: AttentionPhase,
) -> Result<LatentAttentionQuery, String> {
    match ratio {
        0 => latent::query(shape.geometry, shape.ranks(), phase, Keys::Window),
        _ => latent::compressed_query(shape.geometry, shape.ranks(), phase, ratio),
    }
}

impl AttentionOps {
    /// The attention of the layers of compress ratio `ratio` over KV in
    /// `storage`: the BF16 layout reads the same operands stored as BF16
    /// values (`latent::with_storage`).
    pub(crate) fn prepare(
        kernels: &crate::Kernels,
        shape: &Shape,
        ratio: u32,
        storage: KvStorage,
        asks: &Asks,
    ) -> Result<AttentionOps, Refused> {
        let ops = [
            (Kind::Prefill, AttentionPhase::Prefill),
            (Kind::Decode, AttentionPhase::Decode),
            (Kind::Verify, AttentionPhase::Verify),
        ]
        .into_iter()
        .map(|(kind, phase)| {
            let query = attention_query(shape, ratio, phase)
                .and_then(|query| latent::with_storage(query, storage))
                .map_err(Refused::Rule)?;
            let LatentAttentionQuery::Attention { capacity, .. } = query else {
                unreachable!("an attention query");
            };
            let need = exact::need(capacity, exact::of_phase(kind, capacity));
            let op = asks.ask(format!("attention r{ratio} {kind:?}"), need, |need| {
                kernels.prepare_latent_sources(shape.geometry, shape.ranks(), &query, need)
            })?;
            Ok((kind, op))
        })
        .collect::<Result<_, Refused>>()?;
        let cached =
            if crate::Kernels::prefill_cache_eligible(shape.geometry, shape.ranks(), storage) {
                let query = attention_query(shape, ratio, AttentionPhase::Prefill)
                    .map_err(Refused::Rule)?;
                let need = exact::need(PREFILL_TOKENS, exact::PREFILL);
                Some(
                    asks.ask(format!("attention.cached r{ratio}"), need, |need| {
                        kernels.prepare_prefill_cache(&query, need)
                    })?,
                )
            } else {
                None
            };
        let shared_heads = if ratio > 0 && cached.is_some() {
            let query =
                attention_query(shape, ratio, AttentionPhase::Prefill).map_err(Refused::Rule)?;
            let need = exact::need(PREFILL_TOKENS, exact::PREFILL);
            Some(
                asks.ask(format!("attention.shared-heads r{ratio}"), need, |need| {
                    kernels.prepare_prefill_heads(&query, need)
                })?,
            )
        } else {
            None
        };
        let mut selected = Vec::new();
        if crate::Kernels::selected_cache_bytes(
            shape.geometry,
            shape.ranks(),
            storage,
            shape.verify_decode_rows(),
        )
        .is_some()
        {
            for (kind, phase) in [
                (Kind::Decode, AttentionPhase::Decode),
                (Kind::Verify, AttentionPhase::Verify),
            ] {
                let query = attention_query(shape, ratio, phase).map_err(Refused::Rule)?;
                let LatentAttentionQuery::Attention { capacity, .. } = query else {
                    unreachable!()
                };
                let need = exact::need(capacity, exact::of_phase(kind, capacity));
                let op = asks.ask(
                    format!("attention.selected r{ratio} {kind:?}"),
                    need,
                    |need| {
                        kernels.prepare_selected_cache(
                            shape.geometry,
                            &query,
                            need,
                            shape.verify_decode_rows(),
                        )
                    },
                )?;
                selected.push((kind, op));
            }
        }
        Ok(AttentionOps {
            ratio,
            ops,
            cached,
            shared_heads,
            selected,
        })
    }

    /// The partial results of the widest prepared phase. Attention names
    /// this role `workspace` in `OpInfo::bytes`; it is separate from the
    /// backend's unnamed `OpInfo::workspace` scratch.
    pub(crate) fn workspace(&self) -> u64 {
        self.ops
            .iter()
            .filter_map(|(_, op)| op.info().bytes_of("workspace"))
            .max()
            .unwrap_or(0)
    }

    /// One layer's attention over `rows` rows of a phase of `kind`.
    pub(crate) fn attend(
        &self,
        kind: Kind,
        rows: u32,
        views: &AttentionViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let (_, op) = self
            .ops
            .iter()
            .find(|(of, _)| *of == kind)
            .ok_or(Refusal("DeepSeek V4.1 attends in no such phase"))?;
        let mut bind = Bind::default();
        let window = ring_buffers(
            &mut bind,
            views.ring,
            Some((views.item_slots, views.items)),
            Some(views.window_keys),
        );
        let compressed = views.compressed.as_ref().map(|c| LatentCompressedBuffers {
            cache: LatentKvCache {
                pool: bind.view(c.pool),
                pages: c.pages,
                page_stride: c.page_stride,
            },
            block_tables: BlockTables {
                tables: bind.view(c.tables),
                sequences: c.sequences,
                max_pages: c.max_pages,
            },
            key_lists: bind.view(c.selected),
            key_stride: c.top_k,
            budget: c.top_k,
        });
        let buffers = LatentSourcesBuffers {
            q: bind.view(views.q),
            row_batch: bind.view(views.row_batch),
            window,
            compressed,
            sinks: Some(bind.view(views.sinks)),
            cos_sin: Some(bind.view(views.cos_sin)),
            output: Some(bind.view(views.output)),
            partial: None,
            workspace: bind.view(views.workspace),
            live_rows: bind.view(views.live_rows),
        };
        let phase = match kind {
            Kind::Prefill => AttentionPhase::Prefill,
            Kind::Decode => AttentionPhase::Decode,
            Kind::Verify => AttentionPhase::Verify,
        };
        if let Some((_, op)) = self
            .selected
            .iter()
            .find(|(of, _)| *of == kind)
            .filter(|_| crate::Kernels::selected_cache_rows(phase, rows))
        {
            let arena = views
                .selected_cache
                .ok_or(Refusal("selected cache allocation is absent"))?;
            let layout = SelectedCacheLayout::new(
                op.info().bytes_of("decoded").unwrap(),
                op.info().bytes_of("valid").unwrap(),
            )?;
            let mut live = vec![
                views.q,
                views.row_batch,
                views.ring.codes,
                views.item_slots,
                views.window_keys,
                views.sinks,
                views.cos_sin,
                views.output,
                views.workspace,
                views.live_rows,
            ];
            live.extend(views.ring.scales);
            if let Some(c) = &views.compressed {
                live.extend([c.pool, c.tables, c.selected]);
            }
            live.extend(views.decoded_cache);
            if layout.bytes > arena.length || live.iter().any(|v| overlaps(arena, *v)) {
                return Err(Refusal(
                    "selected cache exceeds its allocation or overlaps live operands",
                ));
            }
            let mut region = |r: std::ops::Range<u64>| {
                bind.view(DeviceBufferSpanV1::new(
                    arena.buffer,
                    arena.offset + r.start,
                    r.end - r.start,
                ))
            };
            let cached = SelectedCacheBuffers {
                native: buffers,
                decoded: region(layout.decoded),
                valid: region(layout.valid),
            };
            return bind.launch(op, &cached, Counts::rows(rows));
        }
        if let (Some(op), Some(arena)) = (&self.cached, views.decoded_cache) {
            let page = 64 / self.ratio.max(1);
            let pages = views.compressed.as_ref().map_or(1, |c| c.pages);
            let layout = crate::Kernels::prefill_cache_layout(views.ring.cells, page, pages)?;
            if layout.bytes > arena.length {
                return Err(Refusal(
                    "decoded prefill cache exceeds its owning allocation",
                ));
            }
            let mut live = vec![
                views.q,
                views.row_batch,
                views.ring.codes,
                views.item_slots,
                views.window_keys,
                views.sinks,
                views.cos_sin,
                views.output,
                views.workspace,
                views.live_rows,
            ];
            live.extend(views.ring.scales);
            if let Some(c) = &views.compressed {
                live.extend([c.pool, c.tables, c.selected]);
            }
            if live.iter().any(|v| overlaps(arena, *v)) {
                return Err(Refusal(
                    "decoded prefill cache overlaps a live attention region",
                ));
            }
            let mut region = |r: std::ops::Range<u64>| {
                bind.view(DeviceBufferSpanV1::new(
                    arena.buffer,
                    arena.offset + r.start,
                    r.end - r.start,
                ))
            };
            let cached = model_deepseek41_contract::own_ops::prefill_cache::Buffers {
                native: buffers,
                window: region(layout.window),
                compressed: region(layout.compressed),
                marks: region(layout.marks),
            };
            bind.launch(op, &cached, Counts::rows(rows))
        } else {
            let op = if views.native_shared_heads {
                self.shared_heads.as_ref().unwrap_or(op)
            } else {
                op
            };
            bind.launch_within(op, &buffers, views.workspace, Counts::rows(rows))
        }
    }
}

fn overlaps(a: DeviceBufferSpanV1, b: DeviceBufferSpanV1) -> bool {
    a.buffer == b.buffer
        && a.offset < b.offset.saturating_add(b.length)
        && b.offset < a.offset.saturating_add(a.length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::state::KvStorage;

    #[test]
    fn wide_query_routing_keeps_the_drafter_entry() {
        use crate::program::Phase;
        use model_deepseek41_contract::moe::Experts;
        use runtime_bundle_abi::BundleDeviceBufferHandleV1;
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap().with_verify_rows(64);
        let kernels = crate::Kernels::new(&shape).unwrap();
        let span = |handle, bytes| {
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(handle), 0, bytes)
        };
        let views = QueryRopeViews {
            query: span(1, 60 * 16 * 512 * 4),
            native_q: span(2, 60 * 16 * 512 * 2),
            cos_sin: span(3, 60 * 64 * 4),
            live_rows: span(4, 4),
            status: span(5, 60 * 4),
        };
        for (experts, symbol) in [
            (
                Experts {
                    count: FLASH.shape.moe.experts,
                    top_k: FLASH.shape.moe.top_k,
                },
                "chop_deepseek41_latent_query_rope_heads_v1",
            ),
            (
                Experts {
                    count: FLASH.shape.dspark.experts,
                    top_k: FLASH.shape.dspark.top_k,
                },
                "chop_deepseek41_latent_query_rope_v1",
            ),
        ] {
            let ops = super::super::Ops::prepare_set(&kernels, &shape, KvStorage::Native, experts);
            let glue = ops
                .glue
                .as_ref()
                .unwrap_or_else(|| panic!("{:?}", ops.missing));
            let calls = glue
                .query_rope(
                    Phase::Verify {
                        items: 12,
                        rows: 60,
                    },
                    &views,
                )
                .unwrap();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].launch.symbol, symbol);
        }
    }

    /// A ratio-2 source's page holds half the positions of a ratio-1
    /// source's, and the window-only attention keeps the window's.
    #[test]
    fn attention_pages_hold_the_sources_positions() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        for (ratio, page) in [(0, 64), (1, 64), (2, 32)] {
            let query = attention_query(&shape, ratio, AttentionPhase::Decode).unwrap();
            let LatentAttentionQuery::Attention {
                page_size,
                second_fp4_keys,
                heads,
                ..
            } = query
            else {
                unreachable!()
            };
            assert_eq!((page_size, second_fp4_keys, heads), (page, ratio > 0, 16));
        }
    }

    /// SM120 prepares the glue and the attention at every source ratio.
    #[test]
    fn sm120_attends_at_the_sources_it_registers() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        let ops =
            super::super::Ops::prepare(&kernels, &shape, KvStorage::Native, Default::default());
        assert!(ops.glue.is_some(), "{:?}", ops.missing);
        let ratios: Vec<u32> = ops.attention.iter().map(|ops| ops.ratio).collect();
        assert_eq!(ratios, [0, 1, 2], "{:?}", ops.missing);
        assert!(
            !ops.missing
                .iter()
                .any(|missing| missing.contains("latent attention")),
            "{:?}",
            ops.missing
        );
    }

    #[test]
    fn full_budget_workspace_covers_attention_roles() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let kernels = crate::Kernels::new(&shape).unwrap();
        for storage in [KvStorage::Native, KvStorage::Bf16] {
            let ops = super::super::Ops::prepare(&kernels, &shape, storage, Default::default());
            assert!(ops.missing.is_empty(), "{:?}", ops.missing);
            for attention in &ops.attention {
                for (kind, op) in &attention.ops {
                    let required = op.info().bytes_of("workspace").unwrap();
                    assert!(
                        ops.workspace() >= required,
                        "{storage:?} r{} {kind:?}: allocated {}, attention requires {required}",
                        attention.ratio,
                        ops.workspace(),
                    );
                }
            }
        }
    }
}
