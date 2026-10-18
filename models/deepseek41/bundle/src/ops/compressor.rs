//! The compressor ops of the KV sources: the ratio-2 sources' F32 gated
//! pooling of each completed pair with its RMSNorm (the F32 tail of an
//! incomplete pair in the item's state), each compressed latent's tail
//! RoPE and FP4 write into the serving pages, and each index key's RMSNorm,
//! RoPE and MXFP4 write. Ratio-1 sources normalize their F32 latent with
//! an F32 RMSNorm (F32 in and out, the weight applied in F32). The writes read the compressed RoPE table, which create fills
//! once over the served positions (`RopeTable::Compressed`).

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::compressed_kv::{
    CompressedKvPages, CompressedKvStorage, CompressedKvWriteBuffers, CompressedKvWriteQuery,
};
use kernel_core::contract::{DType, RecurrentPhase, Refusal, RmsNormKind};
use kernel_core::gated_pooling::{GatedPoolingBuffers, GatedPoolingQuery, GatedPoolingState};
use kernel_core::latent_attention::BlockTables;
use kernel_core::pooled_indexer::{
    IndexerCandidates, IndexerKeyFormat, PositionIndexerQuery, PositionKeyPages,
    PositionKeyWriteBuffers,
};
use kernel_core::rmsnorm::{RmsNormBuffers, RmsNormRequirement};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::rope::{
    RopeCosSin, RopeCosSinBuffers, RopeRows, RopeTable,
};
use model_deepseek41_contract::serving::{MAX_VERIFY_ROWS, PAGE_TOKENS, PREFILL_TOKENS};
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, NormViews, exact};
use crate::program::Phase;
use crate::shape::Shape;

/// The gated pooling of each kind of phase.
fn pooling(shape: &Shape, phase: RecurrentPhase) -> GatedPoolingQuery {
    let g = shape.geometry;
    GatedPoolingQuery {
        group: 2,
        width: g.shape.attention.head_dim,
        norm_eps_bits: (g.shape.eps as f32).to_bits(),
        phase,
        capacity: PREFILL_TOKENS,
    }
}

/// A ratio-1 source's RMSNorm: F32 rows of the head dimension, normalized
/// and weighted in F32 into F32 rows.
fn ratio_one_norm(shape: &Shape) -> RmsNormRequirement {
    let g = shape.geometry;
    RmsNormRequirement {
        input: DType::F32,
        output: DType::F32,
        ..RmsNormRequirement::bf16_backend_chosen(
            g.shape.attention.head_dim,
            g.shape.eps as f32,
            RmsNormKind::Plain,
            false,
        )
    }
}

/// The index key write of a source of `ratio`.
pub(crate) fn index_keys(shape: &Shape, ratio: u32) -> PositionIndexerQuery {
    let g = shape.geometry;
    PositionIndexerQuery {
        heads: g.shape.indexer.heads,
        head_dim: g.shape.indexer.dim,
        compression_ratio: ratio,
        tail_rope_dims: g.shape.attention.rope_dim,
        key_norm_eps_bits: (g.shape.eps as f32).to_bits(),
        max_top_k: g.shape.indexer.top_k,
        page_size: PAGE_TOKENS as u32 / ratio,
        candidates: IndexerCandidates::All,
        capacity: PREFILL_TOKENS,
    }
}

/// The compressed latent write of a source of `ratio` in `storage`.
fn latent_write(shape: &Shape, ratio: u32, storage: KvStorage) -> CompressedKvWriteQuery {
    let g = shape.geometry;
    CompressedKvWriteQuery {
        width: g.shape.attention.head_dim,
        tail_rope_dims: g.shape.attention.rope_dim,
        compression_ratio: ratio,
        page_size: PAGE_TOKENS as u32 / ratio,
        storage: match storage {
            KvStorage::Native => CompressedKvStorage::Packed,
            KvStorage::Bf16 => CompressedKvStorage::Bf16,
        },
        capacity: PREFILL_TOKENS,
    }
}

/// The phases of each kind, in [`kind_of`] order.
const PHASES: [RecurrentPhase; 3] = [
    RecurrentPhase::Prefill,
    RecurrentPhase::Decode,
    RecurrentPhase::Verify {
        max_rows: MAX_VERIFY_ROWS,
    },
];

fn kind_of(phase: Phase) -> usize {
    match phase {
        Phase::Prefill { .. } => 0,
        Phase::Decode(_) => 1,
        Phase::Verify { .. } => 2,
    }
}

/// The writes of the sources of one ratio: the latents', and the index
/// keys'. Both physical layouts store the same native products.
struct Writes {
    ratio: u32,
    latent: Prepared<CompressedKvWriteBuffers<View>>,
    keys: Prepared<PositionKeyWriteBuffers<View>>,
}

pub(crate) struct CompressorOps {
    /// Each kind's pooling; none without a ratio-2 source.
    pooling: Option<[Prepared<GatedPoolingBuffers<View>>; 3]>,
    /// A ratio-1 source's F32 RMSNorm; none without one.
    ratio_one: Option<Prepared<RmsNormBuffers<View>>>,
    /// Each source ratio's writes.
    writes: Vec<Writes>,
    /// The compressed RoPE table's rows, filled once at create.
    rope: Prepared<RopeCosSinBuffers<View>>,
    /// The table's rows: every served position.
    table_rows: u32,
}

/// A step's spans of a pooling: the F32 key and gate rows, the norm
/// weight, the step's positions and items, the layer's tail entries and
/// this step's entry ids, the F32 pooled rows with their compressed
/// positions, and the rank's backend workspace.
pub(crate) struct PoolingViews {
    pub key: DeviceBufferSpanV1,
    pub gate: DeviceBufferSpanV1,
    pub norm_weight: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub row_batch: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub tail: DeviceBufferSpanV1,
    pub tail_entries: u32,
    pub slot_ids: DeviceBufferSpanV1,
    pub checkpoint_ids: Option<DeviceBufferSpanV1>,
    pub output: DeviceBufferSpanV1,
    pub output_positions: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
}

/// One cache plane in the serving pages through the step's tables.
#[derive(Clone, Copy)]
pub(crate) struct PlaneViews {
    pub values: DeviceBufferSpanV1,
    pub scales: Option<DeviceBufferSpanV1>,
    pub pages: u32,
    pub page_stride: u64,
    pub tables: DeviceBufferSpanV1,
    pub sequences: u32,
    pub max_pages: u32,
}

impl PlaneViews {
    fn tables(&self, bind: &mut Bind) -> BlockTables<View> {
        BlockTables {
            tables: bind.view(self.tables),
            sequences: self.sequences,
            max_pages: self.max_pages,
        }
    }
}

/// A step's spans of a write: the F32 latent or projected index-key rows,
/// the index key's norm weight, the compressed RoPE table, each row's
/// compressed position (negative for none) and item, the plane, and the
/// rank's backend workspace.
pub(crate) struct WriteViews {
    pub rows: DeviceBufferSpanV1,
    pub norm_weight: Option<DeviceBufferSpanV1>,
    pub cos_sin: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub row_batch: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub plane: PlaneViews,
    pub workspace: DeviceBufferSpanV1,
    /// The phase's row status (`Scratch::row_status`).
    pub status: DeviceBufferSpanV1,
}

impl CompressorOps {
    pub(crate) fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: &Shape,
        storage: KvStorage,
        asks: &Asks,
    ) -> Result<CompressorOps, Refused> {
        let g = shape.geometry;
        let mut ratios: Vec<u32> = g
            .shape
            .csa
            .kv_sources
            .iter()
            .map(|&layer| g.compress_ratio(layer))
            .collect();
        ratios.sort_unstable();
        ratios.dedup();
        let every = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let pooling = match ratios.contains(&2) {
            true => {
                let [prefill, decode, verify] = PHASES.map(|phase| {
                    let need =
                        exact::need(PREFILL_TOKENS, exact::of_phase(kind(phase), PREFILL_TOKENS));
                    asks.ask(format!("compressor.pooling {phase:?}"), need, |need| {
                        backend.gated_pooling(&pooling(shape, phase), need)
                    })
                });
                Some([prefill?, decode?, verify?])
            }
            false => None,
        };
        let ratio_one = match ratios.contains(&1) {
            true => Some(asks.ask("compressor.ratio_one_norm", every, |need| {
                backend.rmsnorm(&ratio_one_norm(shape), need)
            })?),
            false => None,
        };
        let writes = ratios
            .iter()
            .map(|&ratio| {
                let latent =
                    asks.ask(format!("compressor.latent_write r{ratio}"), every, |need| {
                        backend.compressed_kv_write(&latent_write(shape, ratio, storage), need)
                    })?;
                let keys = match storage {
                    KvStorage::Native => asks.ask(
                        format!("compressor.index_key_write r{ratio}"),
                        every,
                        |need| backend.position_indexer_key_write(&index_keys(shape, ratio), need),
                    )?,
                    KvStorage::Bf16 => asks.ask(
                        format!("compressor.index_key_write.bf16 r{ratio}"),
                        every,
                        |need| {
                            backend.position_indexer_key_write_with_storage(
                                &index_keys(shape, ratio),
                                IndexerKeyFormat::Bf16,
                                need,
                            )
                        },
                    )?,
                };
                Ok(Writes {
                    ratio,
                    latent,
                    keys,
                })
            })
            .collect::<Result<_, Refused>>()?;
        let table_rows = u32::try_from(g.context_tokens())
            .map_err(|_| Refused::Rule(format!("{} serves too many positions", g.id)))?;
        let rope = asks.ask(
            "compressor.rope_table",
            kernel_core::backend::Need::rows(table_rows),
            |need| {
                own.rope_cos_sin(
                    &RopeCosSin {
                        table: RopeTable::Compressed,
                        rows: RopeRows::Range,
                    },
                    need,
                )
            },
        )?;
        Ok(CompressorOps {
            pooling,
            ratio_one,
            writes,
            rope,
            table_rows,
        })
    }

    /// The compressed RoPE table's rows: every served position.
    pub(crate) fn table_rows(&self) -> u32 {
        self.table_rows
    }

    /// The compressed RoPE table's bytes.
    pub(crate) fn table_bytes(&self) -> u64 {
        RopeCosSin::output_bytes(self.table_rows)
    }

    /// The launches that fill the compressed RoPE table `output` at create.
    pub(crate) fn fill_table(&self, output: DeviceBufferSpanV1) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = RopeCosSinBuffers {
            positions: None,
            live_rows: None,
            first: 0,
            output: bind.view(output),
        };
        bind.launch(&self.rope, &buffers, Counts::rows(self.table_rows))
    }

    /// The pooling of `phase`'s rows of its `items` items.
    /// A ratio-1 source's RMSNorm of `rows` F32 latent rows.
    pub(crate) fn normalize(&self, rows: u32, views: &NormViews) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .ratio_one
            .as_ref()
            .ok_or(Refusal("DeepSeek V4.1 has no ratio-1 KV source"))?;
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
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    pub(crate) fn pool(
        &self,
        phase: Phase,
        items: u32,
        views: &PoolingViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = self
            .pooling
            .as_ref()
            .map(|ops| &ops[kind_of(phase)])
            .ok_or(Refusal("DeepSeek V4.1 pools no ratio-2 source"))?;
        let mut bind = Bind::default();
        let buffers = GatedPoolingBuffers {
            key: bind.view(views.key),
            gate: bind.view(views.gate),
            norm_weight: bind.view(views.norm_weight),
            positions: bind.view(views.positions),
            row_batch: bind.view(views.row_batch),
            live_rows: bind.view(views.live_rows),
            tail: GatedPoolingState {
                state: bind.view(views.tail),
                entries: views.tail_entries,
                slot_ids: bind.view(views.slot_ids),
                checkpoint_ids: views.checkpoint_ids.map(|span| bind.view(span)),
            },
            output: bind.view(views.output),
            output_positions: bind.view(views.output_positions),
        };
        let counts = Counts::rows(phase.rows()).with_items(items, 0);
        bind.launch_within(op, &buffers, views.workspace, counts)
    }

    fn write_ops(&self, ratio: u32) -> Result<&Writes, Refusal> {
        self.writes
            .iter()
            .find(|writes| writes.ratio == ratio)
            .ok_or(Refusal("DeepSeek V4.1 writes no source of this ratio"))
    }

    /// Whether a source of `ratio` has a writer for the selected storage.
    pub(crate) fn writes_index_keys(&self, ratio: u32) -> bool {
        self.write_ops(ratio).is_ok()
    }

    /// The compressed latents' write of a source of `ratio`.
    pub(crate) fn write_latent(
        &self,
        ratio: u32,
        rows: u32,
        views: &WriteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = &self.write_ops(ratio)?.latent;
        let mut bind = Bind::default();
        let plane = &views.plane;
        let buffers = CompressedKvWriteBuffers {
            latent: bind.view(views.rows),
            cos_sin: bind.view(views.cos_sin),
            positions: bind.view(views.positions),
            row_batch: bind.view(views.row_batch),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: 0,
            cache: CompressedKvPages {
                values: bind.view(plane.values),
                scales: plane.scales.map(|span| bind.view(span)),
                pages: plane.pages,
                value_page_stride: plane.page_stride,
                scale_page_stride: plane.page_stride,
                page_tables: plane.tables(&mut bind),
            },
        };
        let counts = Counts::rows(rows).with_positions(self.table_rows);
        bind.launch_within(op, &buffers, views.workspace, counts)
    }

    /// The index keys' write of a source of `ratio`.
    pub(crate) fn write_index_keys(
        &self,
        ratio: u32,
        rows: u32,
        views: &WriteViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let op = &self.write_ops(ratio)?.keys;
        let mut bind = Bind::default();
        let plane = &views.plane;

        let norm_weight = views
            .norm_weight
            .ok_or(Refusal("the index key write reads its norm weight"))?;
        let buffers = PositionKeyWriteBuffers {
            key: bind.view(views.rows),
            norm_weight: bind.view(norm_weight),
            cos_sin: bind.view(views.cos_sin),
            positions: bind.view(views.positions),
            row_batch: bind.view(views.row_batch),
            live_rows: bind.view(views.live_rows),
            status: bind.view(views.status),
            status_row_offset: 0,
            keys: PositionKeyPages {
                codes: bind.view(plane.values),
                scales: plane
                    .scales
                    .map(|scales| bind.view(scales))
                    .unwrap_or(View::EMPTY),
                pages: plane.pages,
                code_page_stride: plane.page_stride,
                scale_page_stride: if plane.scales.is_some() {
                    plane.page_stride
                } else {
                    0
                },
                page_tables: plane.tables(&mut bind),
            },
        };
        let counts = Counts::rows(rows).with_positions(self.table_rows);
        bind.launch_within(op, &buffers, views.workspace, counts)
    }
}

fn kind(phase: RecurrentPhase) -> exact::Kind {
    match phase {
        RecurrentPhase::Prefill => exact::Kind::Prefill,
        RecurrentPhase::Decode => exact::Kind::Decode,
        RecurrentPhase::Verify { .. } => exact::Kind::Verify,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::contract::Argument;
    use model_deepseek41_contract::geometry::FLASH;
    use model_deepseek41_contract::state;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    fn span(id: u32, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(id, 1),
            offset,
            length,
        )
    }

    #[test]
    fn both_storage_modes_bind_index_and_latent_streams_at_k2_offsets() {
        for rank in [1, 2, 4] {
            let shape = Shape::new(&FLASH, Ranks::new(rank).unwrap()).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            for storage in [KvStorage::Native, KvStorage::Bf16] {
                let asks = Asks::default();
                let ops = CompressorOps::prepare(
                    kernels.backend.as_ref(),
                    kernels.own.as_ref(),
                    &shape,
                    storage,
                    &asks,
                )
                .unwrap();
                let state = state::request_state(&FLASH, storage, false);
                let pool = span(100, 4096, 3 * state.page_bytes());
                for (source, &layer) in FLASH.shape.csa.kv_sources.iter().enumerate() {
                    let binding =
                        crate::compressor::source_binding(&FLASH, layer, storage, pool, 3, 65)
                            .unwrap()
                            .unwrap();
                    if storage == KvStorage::Bf16 {
                        assert_eq!(state.page_bytes(), 204800);
                        assert_eq!(
                            binding.index.values.offset - pool.offset,
                            [32768, 73728, 114688, 188416][source]
                        );
                        assert_eq!(
                            binding.latent.values.offset - pool.offset,
                            [0, 40960, 81920, 122880][source]
                        );
                    }
                    for (index, plane, width) in
                        [(true, binding.index, 128), (false, binding.latent, 512)]
                    {
                        let views = WriteViews {
                            rows: span(1, 0, 3 * width * 4),
                            norm_weight: index.then(|| span(2, 0, width * 2)),
                            cos_sin: span(3, 0, ops.table_bytes()),
                            positions: span(4, 0, 12),
                            row_batch: span(5, 0, 12),
                            live_rows: span(6, 0, 4),
                            plane: PlaneViews {
                                values: plane.values,
                                scales: plane.scales,
                                pages: 3,
                                page_stride: plane.page_stride,
                                tables: span(7, 0, 12),
                                sequences: 1,
                                max_pages: 3,
                            },
                            workspace: span(8, 0, 0),
                            status: span(9, 0, 12),
                        };
                        let launches = if index {
                            ops.write_index_keys(binding.ratio, 3, &views)
                        } else {
                            ops.write_latent(binding.ratio, 3, &views)
                        }
                        .unwrap();
                        assert_eq!(launches.len(), 1);
                        let call = &launches[0].launch;
                        let value_argument = if index { 7 } else { 6 };
                        let Argument::DeviceBuffer(handle) = call.arguments.0[value_argument]
                        else {
                            panic!("value pointer")
                        };
                        assert_eq!(call.bindings.get(handle).unwrap(), plane.values);
                        let strides = call
                            .arguments
                            .0
                            .iter()
                            .filter_map(|a| {
                                if let Argument::U64(v) = a {
                                    Some(*v)
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(strides[0], state.page_bytes());
                        if index && storage == KvStorage::Bf16 {
                            assert!(call.symbol.contains("index_key_bf16"));
                            assert_eq!(strides, [state.page_bytes(), 0, 0]);
                            assert_eq!(call.arguments.0.len(), 17);
                        }
                        assert_eq!(plane.positions_per_page, 64 / binding.ratio);
                    }
                }
            }
        }
    }
}
