//! RoPE with the KV write of global (paged) and sliding-window (ring)
//! layers, the ring write, and DFlash's Q and K RoPE alone.

use kernel_core::attention::KvCache;
use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::contract::{DType, Refusal, RotarySource, RotaryStyle, RowQuery};
use kernel_core::rope_kv::{
    KvWrite, RingWriteBuffers, RingWriteOp, RopeBuffers, RopeKvBuffers, RopeKvOp, RopeOp,
};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, need, no_workspace, one};
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase, RowBuckets};
use crate::switches::Switches;
use crate::topology::Tp;

type RopeKv = Prepared<RopeKvBuffers<View>>;

/// The RoPE ops of an instance; every rank launches the same.
#[derive(Default)]
pub(super) struct Ropes {
    /// Global layers' RoPE and paged KV write.
    paged: ByPhase<RopeKv>,
    /// Sliding-window layers' RoPE, which stores into the rings at decode
    /// and leaves K and V rows to the ring write otherwise.
    ring: ByPhase<RopeKv>,
    /// The sliding-window rows' ring write after attention (prefill and
    /// verify).
    ring_write: ByPhase<Prepared<RingWriteBuffers<View>>>,
    /// DFlash's Q and K RoPE alone, of a DFlash instance.
    drafter: [ByPhase<Prepared<RopeBuffers<View>>>; 2],
    /// The rotated dimensions of the target's layers and of DFlash's, whose
    /// tables hold `rotary / 2` cos and sin pairs per position.
    rotary: [u32; 2],
}

/// The rotation of `tp`'s layers, global or sliding-window: NeoX over the
/// first `rotary` dimensions of each head, from the instance's F32 table.
fn rotation(tp: Tp, global: bool) -> RowQuery {
    let g = tp.geometry;
    RowQuery::Rope {
        head_dim: tp.qk_dim(global) as u32,
        rotary_dim: g.rotary(),
        theta: if global {
            g.rope_theta()
        } else {
            g.swa_rope_theta()
        } as f32,
        style: RotaryStyle::Neox,
        source: RotarySource::CosSinTable,
        fused_paged_kv_write: global,
    }
}

/// The RoPE KV write of `tp`'s global (paged, of `kv` pages) or
/// sliding-window layers (rings, storing when `store`).
fn rope_kv(tp: Tp, global: bool, kv: DType, store: bool) -> RopeKvOp {
    RopeKvOp {
        rope: rotation(tp, global),
        q_heads: tp.q_heads_for(global) as u32,
        kv_heads: tp.kv_heads(global) as u32,
        v_dim: tp.value_dim(global) as u32,
        value_scale: tp.geometry.semantics.attention.value_scale as f32,
        positions: DType::I32,
        cache: if global {
            KvWrite::Paged {
                page_size: crate::contract::PAGE_TOKENS as u32,
                max_context: tp.geometry.context_tokens(),
                kv_dtype: kv,
            }
        } else {
            KvWrite::Ring {
                ring_size: tp.geometry.window(),
                kv_dtype: DType::Bf16,
                store,
            }
        },
    }
}

/// The ring write of the sliding-window layers' K and V rows.
fn ring_write(tp: Tp) -> RingWriteOp {
    RingWriteOp {
        query: RowQuery::RingWrite {
            qk_dim: tp.qk_dim(false) as u32,
            v_dim: tp.value_dim(false) as u32,
            kv_heads: tp.kv_heads(false) as u32,
            ring_size: tp.geometry.window(),
            kv_dtype: DType::Bf16,
        },
        positions: DType::I32,
        single_sequence: false,
    }
}

/// DFlash's RoPE of its Q (`q`) or K heads, on their own.
fn drafter_rope(tp: Tp, q: bool) -> RopeOp {
    let dflash = tp.geometry.dflash();
    RopeOp {
        rope: RowQuery::Rope {
            head_dim: dflash.head_dim,
            rotary_dim: dflash.rotary_dim,
            theta: dflash.rope_theta as f32,
            style: RotaryStyle::Neox,
            source: RotarySource::CosSinTable,
            fused_paged_kv_write: false,
        },
        heads: if q {
            tp.dflash_q_heads()
        } else {
            tp.dflash_kv_heads()
        } as u32,
        positions: DType::I32,
    }
}

impl Ropes {
    /// The RoPE ops of `tp` over `kv` global pages, with DFlash's when
    /// `dflash`.
    pub(super) fn prepare(
        backend: &dyn Backend,
        tp: Tp,
        switches: Switches,
        kv: DType,
        dflash: bool,
    ) -> Result<Self, Refused> {
        let drafter = |q: bool| {
            ByPhase::prepare(|phase| {
                if !dflash {
                    return Ok(None);
                }
                let need = Need::rows(phase.capacity()?);
                no_workspace(backend.rope(&drafter_rope(tp, q), need)?).map(Some)
            })
        };
        Ok(Self {
            paged: ByPhase::prepare(|phase| {
                no_workspace(
                    backend.rope_kv_write(&rope_kv(tp, true, kv, true), need(phase, switches)?)?,
                )
                .map(Some)
            })?,
            ring: ByPhase::prepare(|phase| {
                let store = matches!(phase, Phase::Decode(_));
                no_workspace(
                    backend
                        .rope_kv_write(&rope_kv(tp, false, kv, store), need(phase, switches)?)?,
                )
                .map(Some)
            })?,
            ring_write: ByPhase::prepare(|phase| {
                if matches!(phase, Phase::Decode(_)) {
                    return Ok(None);
                }
                no_workspace(backend.ring_write(&ring_write(tp), need(phase, switches)?)?).map(Some)
            })?,
            drafter: [drafter(true)?, drafter(false)?],
            rotary: [tp.geometry.rotary(), tp.geometry.dflash().rotary_dim],
        })
    }
}

/// The counts of a row kernel over a `phase` step's `live_tokens` rows,
/// which its per-row buffers hold, gridded at the phase's row bucket
/// (`MimoPhase::row_grid`), which launches record as their rows.
fn row_counts(
    phase: Phase,
    switches: Switches,
    buckets: RowBuckets,
    live_tokens: u32,
) -> Result<Counts, Refusal> {
    Ok(Counts {
        padded_rows: phase.row_grid(live_tokens, switches, buckets)?,
        ..Counts::rows(live_tokens)
    })
}

/// The positions of `table`, an F32 `[position][rotary / 2][cos, sin]`
/// table.
pub(super) fn table_rows(table: DeviceBufferSpanV1, rotary: u32) -> Result<u32, Refusal> {
    u32::try_from(table.length / (u64::from(rotary / 2) * 8))
        .map_err(|_| Refusal("MiMo RoPE table rows overflow"))
}

/// The views a RoPE KV write and the ring write read and write.
pub struct RopeViews {
    /// The F32 cos and sin table of the layer's rotation.
    pub cos_sin: DeviceBufferSpanV1,
    pub qkv: DeviceBufferSpanV1,
    pub q: DeviceBufferSpanV1,
    pub k_new: DeviceBufferSpanV1,
    pub v_new: DeviceBufferSpanV1,
    pub token_batch: DeviceBufferSpanV1,
    pub positions: DeviceBufferSpanV1,
    pub live_tokens: DeviceBufferSpanV1,
}

/// A layer's KV cache: its K and V planes of `slots` slots (pages or
/// rings) at their strides, and the FP8 scale bits.
pub struct CacheViews {
    pub k: DeviceBufferSpanV1,
    pub v: DeviceBufferSpanV1,
    pub slots: u32,
    pub strides: [u64; 2],
    pub scale_bits: [u32; 2],
}

impl CacheViews {
    pub(super) fn bind(&self, bind: &mut Bind) -> KvCache<View> {
        KvCache {
            k: bind.view(self.k),
            v: bind.view(self.v),
            slots: self.slots,
            k_slot_stride: self.strides[0],
            v_slot_stride: self.strides[1],
            k_scale: f32::from_bits(self.scale_bits[0]),
            v_scale: f32::from_bits(self.scale_bits[1]),
        }
    }
}

/// Where a RoPE KV write stores: a global layer's pages through the items'
/// block tables, or a sliding-window layer's rings through the items'
/// slot ids.
pub enum Target {
    Paged { block_tables: DeviceBufferSpanV1 },
    Ring { slot_ids: DeviceBufferSpanV1 },
}

impl super::Ops {
    /// The RoPE and KV write of a `phase` step's `live_tokens` rows (bucket
    /// `bucket` of `COMM_BUCKETS`) into `target`. It grids the phase's row
    /// bucket (`MimoPhase::row_grid`), which it records as its rows.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_kv(
        &self,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        views: &RopeViews,
        cache: &CacheViews,
        target: Target,
    ) -> Result<Lowered, Refusal> {
        let mut bind = Bind::default();
        let (op, block_tables, slot_ids) = match target {
            Target::Paged { block_tables } => (
                self.ropes.paged.get(phase, bucket)?,
                Some(bind.view(block_tables)),
                None,
            ),
            Target::Ring { slot_ids } => (
                self.ropes.ring.get(phase, bucket)?,
                None,
                Some(bind.view(slot_ids)),
            ),
        };
        let buffers = RopeKvBuffers {
            qkv: bind.view(views.qkv),
            q: bind.view(views.q),
            k_new: Some(bind.view(views.k_new)),
            v_new: Some(bind.view(views.v_new)),
            cache: cache.bind(&mut bind),
            block_tables,
            slot_ids,
            token_batch: bind.view(views.token_batch),
            positions: bind.view(views.positions),
            cos_sin: Some(bind.view(views.cos_sin)),
            live_rows: bind.view(views.live_tokens),
        };
        let counts = Counts {
            positions: table_rows(views.cos_sin, self.ropes.rotary[0])?,
            ..row_counts(phase, switches, self.row_buckets, live_tokens)?
        };
        one(bind.lower(op.as_ref(), &buffers, counts)?)
    }

    /// The ring write of a `phase` step's sliding-window K and V rows
    /// (bucket `bucket` of `COMM_BUCKETS`) into the items' rings, after
    /// attention read them. It grids the phase's row bucket.
    #[allow(clippy::too_many_arguments)]
    pub fn ring_write(
        &self,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        views: &RopeViews,
        cache: &CacheViews,
        slot_ids: DeviceBufferSpanV1,
        last_positions: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let op = self.ropes.ring_write.get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RingWriteBuffers {
            k_new: bind.view(views.k_new),
            v_new: bind.view(views.v_new),
            cache: cache.bind(&mut bind),
            slot_ids: bind.view(slot_ids),
            token_batch: bind.view(views.token_batch),
            positions: bind.view(views.positions),
            last_positions: bind.view(last_positions),
            live_rows: bind.view(views.live_tokens),
        };
        let counts = row_counts(phase, switches, self.row_buckets, live_tokens)?;
        one(bind.lower(op.as_ref(), &buffers, counts)?)
    }

    /// DFlash's RoPE of its Q (`q`) or K heads of a `phase` step's
    /// `live_tokens` rows (bucket `bucket` of `COMM_BUCKETS`), from `input`
    /// into `output`. It grids the phase's capacity, which it records as
    /// its rows.
    #[allow(clippy::too_many_arguments)]
    pub fn drafter_rope(
        &self,
        q: bool,
        phase: Phase,
        bucket: usize,
        live: u32,
        input: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        positions: DeviceBufferSpanV1,
        cos_sin: DeviceBufferSpanV1,
        live_tokens: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let op = self.ropes.drafter[usize::from(!q)].get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RopeBuffers {
            input: bind.view(input),
            output: bind.view(output),
            positions: bind.view(positions),
            cos_sin: Some(bind.view(cos_sin)),
            live_rows: bind.view(live_tokens),
        };
        let counts = Counts {
            padded_rows: phase.capacity()?,
            positions: table_rows(cos_sin, self.ropes.rotary[1])?,
            ..Counts::rows(live)
        };
        one(bind.lower(op.as_ref(), &buffers, counts)?)
    }
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{Argument, RuntimeCounts};
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, resolved};
    use super::*;
    use crate::dispatch::BufferBindings;
    use crate::ops::{Ops, Plan};
    use crate::pipeline::PreparedLaunch;
    use crate::program_lower::launches;
    use crate::row_plan::{GaRopeViews, SwaRopeViews};

    /// A span of its own buffer, large enough for any RoPE.
    fn span(buffer: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            1 << 34,
        )
    }

    fn prepared(tp: Tp, switches: Switches, kv: DType) -> Ops {
        Ops::prepare(
            &*crate::sm120::backend(tp, switches, true).unwrap(),
            &Plan {
                tp,
                switches,
                kv,
                items: crate::contract::DECODE_ENTRY_ROWS,
                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                state: &|_| [u64::MAX; 3],
                gathers: &any_gathers,
                dflash: Some([u64::MAX; 2]),
                own: crate::ops::OwnPlan::new(
                    &crate::sm120::own_ops::Sm120Own::new(tp),
                    crate::contract::DECODE_ENTRY_ROWS,
                ),
            },
        )
        .unwrap()
    }

    /// Every switch setting that moves a RoPE launch: the prefill row grids.
    fn settings() -> [Switches; 2] {
        [false, true].map(|capacity_row_grids| Switches {
            capacity_row_grids,
            ..Switches::configured()
        })
    }

    /// The live row counts compared at a phase of `capacity` rows: every
    /// row grid's least and most rows.
    fn counts(capacity: u32) -> Vec<u32> {
        let mut counts = vec![1, capacity];
        let mut bucket = 1;
        while bucket < capacity {
            counts.extend([bucket, bucket + 1]);
            bucket *= 2;
        }
        counts.sort_unstable();
        counts.dedup();
        counts
    }

    /// A span of its own buffer of `rows` I32 rows, as a step's per-row
    /// inputs hold only its rows.
    fn per_row(buffer: u32, rows: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            u64::from(rows) * 4,
        )
    }

    fn views(tables: [DeviceBufferSpanV1; 2], global: bool, live: u32) -> RopeViews {
        RopeViews {
            cos_sin: tables[usize::from(!global)],
            qkv: span(10),
            q: span(11),
            k_new: span(12),
            v_new: span(13),
            token_batch: per_row(14, live),
            positions: per_row(15, live),
            live_tokens: span(16),
        }
    }

    /// The prepared RoPE KV writes and ring write launch what MiMo's row
    /// builders planned (`row_plan::rope_ga`, `rope_swa`, `ring_write_swa`):
    /// symbol, rows, plan and resolved arguments, for both page formats and
    /// row grid settings, at every phase, bucket and row grid, at TP2 and
    /// TP4.
    #[test]
    fn rope_kv_writes_launch_what_the_row_builders_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let g = tp.geometry;
            let global = (0..g.layers()).find(|&layer| g.is_global(layer)).unwrap();
            let sliding = (0..g.layers()).find(|&layer| !g.is_global(layer)).unwrap();
            let ring = crate::ring::RingGeometry::new(tp, 32);
            let ring_state = span(20);
            let (k_ring, v_ring) = crate::ring::cache_views(ring_state, 0, ring).unwrap();
            let window = u64::from(g.window());
            let rings = CacheViews {
                k: k_ring,
                v: v_ring,
                slots: ring.slots,
                strides: [window * ring.k_entry, window * ring.v_entry],
                scale_bits: [1.0_f32.to_bits(); 2],
            };
            let tables = [span(30), span(31)];
            for kv in [DType::Bf16, DType::Fp8E4M3] {
                let fp8 = kv == DType::Fp8E4M3;
                // A pool of whole pages.
                let pool = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(21, 1),
                    256,
                    128 * crate::topology::ga_page_bytes(tp, fp8),
                );
                let pages = crate::ga_pool::layer_pages(pool, global, fp8, tp).unwrap();
                let scale_bits = [0.5_f32.to_bits(), 0.25_f32.to_bits()];
                let paged = CacheViews {
                    k: pages.k,
                    v: pages.v,
                    slots: u32::try_from(pages.page_capacity).unwrap(),
                    strides: [pages.k_page_stride_bytes, pages.v_page_stride_bytes],
                    scale_bits,
                };
                for switches in settings() {
                    let ops = prepared(tp, switches, kv);
                    for (phase, bucket) in phases() {
                        for live in counts(phase.capacity().unwrap()) {
                            let planned = crate::row_plan::rope_ga(
                                global,
                                phase,
                                switches,
                                kv,
                                scale_bits,
                                live,
                                tp,
                                GaRopeViews {
                                    cos_sin: tables[0],
                                    qkv: span(10),
                                    q: span(11),
                                    k_new: span(12),
                                    v_new: span(13),
                                    pages: crate::ga_pool::layer_pages(pool, global, fp8, tp)
                                        .unwrap(),
                                    block_tables: span(17),
                                    token_batch: per_row(14, live),
                                    positions: per_row(15, live),
                                    live_tokens: span(16),
                                },
                            )
                            .unwrap();
                            let lowered = ops
                                .rope_kv(
                                    phase,
                                    bucket,
                                    switches,
                                    live,
                                    &views(tables, true, live),
                                    &paged,
                                    Target::Paged {
                                        block_tables: span(17),
                                    },
                                )
                                .unwrap();
                            assert_eq!(
                                resolved(&launches(vec![lowered])),
                                resolved(&[planned]),
                                "{tp:?} {kv:?} {phase:?} {live}"
                            );
                            let swa = || SwaRopeViews {
                                cos_sin: tables[1],
                                qkv: span(10),
                                q: span(11),
                                k_new: span(12),
                                v_new: span(13),
                                ring_state,
                                ring,
                                slot_ids: span(18),
                                token_batch: per_row(14, live),
                                positions: per_row(15, live),
                                live_tokens: span(16),
                            };
                            let planned = crate::row_plan::rope_swa(
                                sliding,
                                phase,
                                switches,
                                live,
                                swa(),
                                tp,
                            )
                            .unwrap();
                            let lowered = ops
                                .rope_kv(
                                    phase,
                                    bucket,
                                    switches,
                                    live,
                                    &views(tables, false, live),
                                    &rings,
                                    Target::Ring { slot_ids: span(18) },
                                )
                                .unwrap();
                            assert_eq!(
                                resolved(&launches(vec![lowered])),
                                resolved(&[planned]),
                                "{tp:?} {phase:?} {live} ring"
                            );
                            let lowered = ops.ring_write(
                                phase,
                                bucket,
                                switches,
                                live,
                                &views(tables, false, live),
                                &rings,
                                span(18),
                                span(19),
                            );
                            if matches!(phase, Phase::Decode(_)) {
                                assert!(lowered.is_err());
                                continue;
                            }
                            let planned = crate::row_plan::ring_write_swa(
                                sliding,
                                phase,
                                switches,
                                live,
                                swa(),
                                span(19),
                                tp,
                            )
                            .unwrap();
                            assert_eq!(
                                resolved(&launches(vec![lowered.unwrap()])),
                                resolved(&[planned]),
                                "{tp:?} {phase:?} {live} ring write"
                            );
                            compared += 3;
                        }
                    }
                }
            }
        }
        assert!(compared > 1000, "{compared}");
    }

    /// DFlash's Q and K RoPE as MiMo's DFlash program planned it: its
    /// table entry over the phase's capacity, input and output at the
    /// heads' width.
    fn planned_drafter_rope(
        tp: Tp,
        q: bool,
        phase: Phase,
        live: u32,
        [input, output, positions, table, live_tokens]: [DeviceBufferSpanV1; 5],
    ) -> PreparedLaunch {
        let name = if q {
            "dflash_q_rope_table_pos32"
        } else {
            "dflash_k_rope_table_pos32"
        };
        let crate::sm120::selection::Selected::Row(entry) =
            crate::sm120::selection::row_for(tp, name).unwrap()
        else {
            unreachable!()
        };
        let width = if q {
            tp.dflash_q_heads()
        } else {
            tp.dflash_kv_heads()
        } * u64::from(tp.geometry.dflash().head_dim);
        let width = width as u32;
        let mut bindings = BufferBindings::default();
        let arguments = vec![
            Argument::DeviceBuffer(bindings.bind(input)),
            Argument::DeviceBuffer(bindings.bind(output)),
            Argument::DeviceBuffer(bindings.bind(positions)),
            Argument::DeviceBuffer(bindings.bind(live_tokens)),
            Argument::U32(width),
            Argument::U32(width),
            Argument::DeviceBuffer(bindings.bind(table)),
        ];
        let capacity = phase.capacity().unwrap();
        let (plan, arguments) = crate::sm120::launch::row::plan(
            &entry,
            capacity,
            RuntimeCounts {
                m: live,
                batch: 1,
                context: 1,
                routed_rows: 8,
                split_k: 1,
            },
            arguments,
        )
        .unwrap();
        PreparedLaunch {
            symbol: entry.symbol,
            rows: capacity,
            plan,
            arguments,
            bindings,
        }
    }

    /// The prepared DFlash RoPE launches what MiMo's DFlash program planned
    /// for its Q and K heads at every phase and bucket, at TP2 and TP4.
    #[test]
    fn drafter_ropes_launch_what_the_dflash_program_planned() {
        for tp in Tp::SUPPORTED {
            let switches = Switches::configured();
            let ops = prepared(tp, switches, DType::Bf16);
            for (phase, bucket) in phases() {
                for live in counts(phase.capacity().unwrap()) {
                    // Positions hold the step's rows.
                    let spans = [span(1), span(2), per_row(3, live), span(4), span(5)];
                    for q in [true, false] {
                        let [input, output, positions, table, live_tokens] = spans;
                        let lowered = ops
                            .drafter_rope(
                                q,
                                phase,
                                bucket,
                                live,
                                input,
                                output,
                                positions,
                                table,
                                live_tokens,
                            )
                            .unwrap();
                        assert_eq!(
                            resolved(&launches(vec![lowered])),
                            resolved(&[planned_drafter_rope(tp, q, phase, live, spans)]),
                            "{tp:?} {phase:?} {live} q {q}"
                        );
                    }
                }
            }
        }
    }
}
