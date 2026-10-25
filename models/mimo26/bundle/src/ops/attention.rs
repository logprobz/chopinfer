//! Attention of the global (paged) and sliding-window (ring) layers, and
//! DFlash's block attention.

use kernel_core::attention::{AttentionBuffers, AttentionFold, AttentionOp, FoldInputs};
use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::contract::{AttentionLayout, AttentionPhase, DType, Refusal, RotarySource};
use kernel_core::properties::{Invariance, Rows};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, CacheViews, need, no_workspace};
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase};
use crate::switches::Switches;
use crate::target::Target;
use crate::topology::Tp;

type Attention = Prepared<AttentionBuffers<View>>;

/// The attention ops of an instance; every rank launches the same.
#[derive(Default)]
pub(super) struct Attentions {
    /// Global layers' attention over the instance's pages.
    paged: ByPhase<Attention>,
    /// Sliding-window layers' attention over the rings.
    ring: ByPhase<Attention>,
    /// DFlash's block attention, of a DFlash instance's verify buckets.
    block: ByPhase<Attention>,
    /// The rotated dimensions of a folded decode, whose table holds
    /// `rotary / 2` cos and sin pairs per position.
    rotary: u32,
}

/// The rows one item of a `phase` step attends from: every packed token of
/// a prefill, one decode token, a verify item's tokens.
fn item_rows(phase: Phase) -> u32 {
    match phase {
        Phase::Prefill => crate::contract::PREFILL_TOKENS,
        Phase::Decode(_) => 1,
        Phase::Verify(_) => crate::contract::MAX_VERIFY_ROWS,
    }
}

fn attention_phase(phase: Phase) -> AttentionPhase {
    match phase {
        Phase::Prefill => AttentionPhase::Prefill,
        Phase::Decode(_) => AttentionPhase::Decode,
        Phase::Verify(_) => AttentionPhase::Verify,
    }
}

/// Whether an instance of `items` items (its largest bucket) runs `phase`
/// steps: decode buckets of at most its items, verify buckets of at most
/// its items' verify rows.
fn served(phase: Phase, items: u32) -> Result<bool, Refusal> {
    let rows = phase.capacity()?;
    Ok(match phase {
        Phase::Prefill => true,
        Phase::Decode(_) => rows <= items,
        Phase::Verify(_) => rows <= items.saturating_mul(crate::contract::MAX_VERIFY_ROWS),
    })
}

/// What a `phase` step needs of attention: the phase's rows, a row's bits
/// in any chunk for prefill, and for exact verify rows each row's one-row
/// bits, which claim nothing about the other rows.
fn attention_need(phase: Phase, switches: Switches) -> Result<Need, Refusal> {
    let mut need = need(phase, switches)?;
    if need.exact.rows != Rows::None && matches!(phase, Phase::Verify(_)) {
        need.exact = Invariance {
            rows: need.exact.rows,
            ..Invariance::NONE
        };
    }
    Ok(need)
}

impl Attentions {
    /// The attention ops of a `tp` instance of `items` items (its largest
    /// bucket) over `kv` global pages, with DFlash's block attention when
    /// `dflash`, at the queries `target`'s catalog states.
    pub(super) fn prepare(
        backend: &dyn Backend,
        target: &dyn Target,
        tp: Tp,
        switches: Switches,
        kv: DType,
        items: u32,
        dflash: bool,
    ) -> Result<Self, Refused> {
        let g = tp.geometry;
        let op = |global: bool, phase: Phase| -> Result<Option<Attention>, Refused> {
            if !served(phase, items)? {
                return Ok(None);
            }
            let (layout, kv) = if global {
                (AttentionLayout::Paged, kv)
            } else {
                (AttentionLayout::Ring, DType::Bf16)
            };
            let mut query = target.attention_query(layout, attention_phase(phase), kv)?;
            // Prefill steps plan for every served item, and at least the
            // catalog's 32; decode and verify for every served item.
            query.max_batch_capacity = match phase {
                Phase::Prefill => items.max(crate::contract::DECODE_ENTRY_ROWS),
                Phase::Decode(_) | Phase::Verify(_) => items,
            };
            // Decode folds the RoPE and KV write in unless switched off.
            let fold = (matches!(phase, Phase::Decode(_)) && switches.fold_rope).then(|| {
                AttentionFold::Neox {
                    rotary: g.rotary(),
                    theta: if global {
                        g.rope_theta()
                    } else {
                        g.swa_rope_theta()
                    } as f32,
                    v_multiplier: g.semantics.attention.value_scale as f32,
                    source: RotarySource::CosSinTable,
                }
            });
            let op = AttentionOp {
                query,
                item_rows: item_rows(phase),
                fold,
                plan_at_capacity: false,
            };
            no_workspace(backend.attention(&op, attention_need(phase, switches)?)?).map(Some)
        };
        Ok(Self {
            paged: ByPhase::prepare(|phase| op(true, phase))?,
            ring: ByPhase::prepare(|phase| op(false, phase))?,
            block: ByPhase::prepare(|phase| {
                if !dflash || !matches!(phase, Phase::Verify(_)) || !served(phase, items)? {
                    return Ok(None);
                }
                // Draft blocks plan for every drafting item of the largest
                // bucket, and at least the catalog's 32, whatever items a
                // step drafts, so one graph serves them all.
                let mut query = target.block_attention_query()?;
                query.max_batch_capacity = items.max(crate::contract::DECODE_ENTRY_ROWS);
                let op = AttentionOp {
                    query,
                    item_rows: item_rows(phase),
                    fold: None,
                    plan_at_capacity: true,
                };
                let need = Need::rows(phase.capacity()?);
                no_workspace(backend.attention(&op, need)?).map(Some)
            })?,
            rotary: g.rotary(),
        })
    }
}

/// The step views an attention reads and writes.
pub struct AttentionViews {
    /// The query rows, or the packed QKV rows of a folded decode.
    pub q: DeviceBufferSpanV1,
    pub k_new: DeviceBufferSpanV1,
    pub v_new: DeviceBufferSpanV1,
    /// The items' rings (sliding-window and block attention) or block
    /// tables (global attention).
    pub slot_ids: Option<DeviceBufferSpanV1>,
    pub block_tables: Option<DeviceBufferSpanV1>,
    pub seq_lens: DeviceBufferSpanV1,
    pub q_start: DeviceBufferSpanV1,
    pub q_len: DeviceBufferSpanV1,
    pub prefix_len: DeviceBufferSpanV1,
    pub live_batch: DeviceBufferSpanV1,
    /// The per-head sink logits of a layer that has them.
    pub sink: Option<DeviceBufferSpanV1>,
    /// The F32 cos and sin table a folded decode rotates by.
    pub cos_sin: Option<DeviceBufferSpanV1>,
    pub output: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
}

/// Which of an attention's launches a stage takes: the partial, or the
/// merge (none when the partial writes the output directly).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Part {
    Partial,
    Merge,
}

/// Which attention of a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layer {
    Global,
    Sliding,
    /// DFlash's block attention.
    Block,
}

impl super::Ops {
    /// `part` of a `phase` step's attention (bucket `bucket` of
    /// `COMM_BUCKETS`) in `layer` over `cache`: `live_tokens` query rows of
    /// `items` items, whose keys reach `context`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        layer: Layer,
        part: Part,
        phase: Phase,
        bucket: usize,
        live_tokens: u32,
        items: u32,
        context: u32,
        views: &AttentionViews,
        cache: &CacheViews,
    ) -> Result<Option<Lowered>, Refusal> {
        let ops = &self.attentions;
        let op = match layer {
            Layer::Global => &ops.paged,
            Layer::Sliding => &ops.ring,
            Layer::Block => &ops.block,
        }
        .get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = AttentionBuffers {
            q: bind.view(views.q),
            cache: cache.bind(&mut bind),
            block_tables: views.block_tables.map(|span| bind.view(span)),
            slot_ids: views.slot_ids.map(|span| bind.view(span)),
            seq_lens: Some(bind.view(views.seq_lens)),
            q_start: Some(bind.view(views.q_start)),
            q_len: Some(bind.view(views.q_len)),
            prefix_len: Some(bind.view(views.prefix_len)),
            live_batch: bind.view(views.live_batch),
            k_new: Some(bind.view(views.k_new)),
            v_new: Some(bind.view(views.v_new)),
            sink: views.sink.map(|span| bind.view(span)),
            rope: views.cos_sin.map(|span| FoldInputs {
                cos_sin: Some(bind.view(span)),
                q_norm: None,
                k_norm: None,
                rope_positions: None,
            }),
            output: bind.view(views.output),
            workspace: bind.view(views.workspace),
        };
        let positions = match views.cos_sin {
            Some(table) => super::rope::table_rows(table, ops.rotary)?,
            None => 0,
        };
        // Verify rows that run on their own grid the bucket's rows, as the
        // builders planned them, when a step (an eager MTP catch-up) holds
        // fewer.
        let padded_rows = match (layer, phase) {
            (Layer::Global | Layer::Sliding, Phase::Verify(_)) => phase.capacity()?,
            _ => 0,
        };
        let counts = Counts {
            items,
            context,
            positions,
            padded_rows,
            ..Counts::rows(live_tokens)
        };
        let mut launches = bind.lower(op.as_ref(), &buffers, counts)?;
        Ok(match part {
            Part::Partial if !launches.is_empty() => Some(launches.remove(0)),
            Part::Partial => return Err(Refusal("an attention launched nothing")),
            Part::Merge => launches.into_iter().nth(1),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;

    use bundle_runtime::geometry::Items;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, resolved};
    use super::*;
    use crate::ops::{Ops, Plan};
    use crate::program::Stage;
    use crate::program_lower::{DeviceInputs, launches};
    use crate::scratch::RankScratch;

    const BIG: u64 = 1 << 36;

    /// A span of its own buffer.
    fn span(buffer: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(buffer, 1), 256, BIG)
    }

    fn inputs() -> DeviceInputs {
        DeviceInputs {
            token_ids: span(10),
            positions: span(11),
            token_batch: span(12),
            block_tables: span(13),
            slot_ids: span(14),
            q_start: span(15),
            q_len: span(16),
            prefix_len: span(17),
            seq_lens: span(18),
            live_batch: span(19),
            last_positions: span(20),
            live_tokens: span(21),
            logit_rows: span(22),
            row_indices: span(23),
        }
    }

    /// One rank's scratch, each buffer its own, with the RoPE tables.
    fn scratch(tp: Tp) -> RankScratch {
        let next = Cell::new(100);
        let mut scratch = RankScratch::allocate(0, tp, |bytes| {
            next.set(next.get() + 1);
            Ok(DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(next.get(), 1),
                0,
                bytes,
            ))
        })
        .unwrap();
        scratch.rope_tables = Some([span(30), span(31)]);
        scratch
    }

    /// Every switch setting that moves an attention launch.
    fn settings() -> Vec<Switches> {
        let mut settings = Vec::new();
        for exact_verify in [false, true] {
            for exact_attention_grouped in [false, true] {
                for verify_attention_rows in [false, true] {
                    for fold_rope in [false, true] {
                        settings.push(Switches {
                            exact_verify,
                            exact_attention_grouped,
                            verify_attention_rows,
                            fold_rope,
                            ..Switches::configured()
                        });
                    }
                }
            }
        }
        settings
    }

    /// The step shapes compared at a phase: (the step's phase, live tokens,
    /// live items, context). Verify steps run their bucket's rows, or, as
    /// an eager MTP catch-up does, fewer rows of the bucket
    /// (`Phase::Verify(live)`).
    fn steps(phase: Phase, items: u32) -> Vec<(Phase, u32, u32, u32)> {
        let rows = phase.capacity().unwrap();
        let contexts = [1, 300, 5_000, 70_000, 1_048_576];
        let mut steps = Vec::new();
        for context in contexts {
            match phase {
                Phase::Prefill => {
                    for (tokens, batch) in [(1, 1), (64, 3), (700, 1), (8192, 8), (8192, items)] {
                        steps.push((phase, tokens, batch.min(items), context.max(tokens)));
                    }
                }
                Phase::Decode(_) => steps.push((phase, rows, rows, context)),
                Phase::Verify(_) => {
                    let mut live = vec![rows];
                    live.extend(
                        [rows / 2 + 1, rows - 1]
                            .into_iter()
                            .filter(|&n| n > rows / 2),
                    );
                    live.dedup();
                    for live in live {
                        // A verify item holds at most MAX_VERIFY_ROWS rows.
                        let fewest = live.div_ceil(crate::contract::MAX_VERIFY_ROWS);
                        for batch in [fewest, live.min(items)] {
                            steps.push((Phase::Verify(live), live, batch, context));
                        }
                    }
                }
            }
        }
        steps
    }

    /// The prepared attentions launch what MiMo's attention builders
    /// planned (`attention_plan::paged` and `ring`) for each stage, the
    /// partial and the merge: symbol, rows, plan and resolved arguments, for
    /// both page formats, every switch setting that moves an attention, 32
    /// and 256 served items, at every phase, bucket and step shape, at TP2
    /// and TP4.
    #[test]
    fn attentions_launch_what_the_attention_builders_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let g = tp.geometry;
            let global = (0..g.layers()).find(|&layer| g.is_global(layer)).unwrap();
            let sliding = (0..g.layers()).find(|&layer| !g.is_global(layer)).unwrap();
            let scratch = scratch(tp);
            let workspace = span(40);
            let ring_state = span(41);
            let views = BTreeMap::from([(
                format!("model.layers.{sliding}.self_attn.attention_sink_bias"),
                span(42),
            )]);
            for kv in [DType::Bf16, DType::Fp8E4M3] {
                let fp8 = kv == DType::Fp8E4M3;
                let pool = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(43, 1),
                    256,
                    128 * crate::topology::ga_page_bytes(tp, fp8),
                );
                let scale_bits = [0.5_f32.to_bits(), 0.25_f32.to_bits()];
                let pages = crate::ga_pool::layer_pages(pool, global, fp8, tp).unwrap();
                let paged = CacheViews {
                    k: pages.k,
                    v: pages.v,
                    slots: u32::try_from(pages.page_capacity).unwrap(),
                    strides: [pages.k_page_stride_bytes, pages.v_page_stride_bytes],
                    scale_bits,
                };
                for items in [32, 256] {
                    let served_items = Items::new(items).unwrap();
                    let ring = crate::ring::RingGeometry::new(tp, items);
                    let (k, v) = crate::ring::cache_views(ring_state, 0, ring).unwrap();
                    let window = u64::from(g.window());
                    let rings = CacheViews {
                        k,
                        v,
                        slots: ring.slots,
                        strides: [window * ring.k_entry, window * ring.v_entry],
                        scale_bits: [1.0_f32.to_bits(); 2],
                    };
                    for switches in settings() {
                        let backend = crate::sm120::backend(tp, switches, true).unwrap();
                        let ops = Ops::prepare(
                            &*backend,
                            &Plan {
                                tp,
                                switches,
                                kv,
                                items,
                                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                                state: &|_| [u64::MAX; 3],
                                gathers: &any_gathers,
                                dflash: None,
                                own: crate::ops::OwnPlan::new(
                                    &crate::sm120::own_ops::Sm120Own::new(tp),
                                    items,
                                ),
                            },
                        )
                        .unwrap();
                        for (phase, bucket) in phases() {
                            if !served(phase, items).unwrap() {
                                continue;
                            }
                            for (phase, live_tokens, live_items, context) in steps(phase, items) {
                                let fold = crate::attention_plan::folds_rope(phase, switches);
                                let ops_items = match phase {
                                    Phase::Decode(rows) => rows,
                                    _ => live_items,
                                };
                                for (layer, global) in [(global, true), (sliding, false)] {
                                    let tables = scratch.rope_tables.unwrap();
                                    let step = AttentionViews {
                                        q: if fold { scratch.qkv } else { scratch.q },
                                        k_new: scratch.k_new,
                                        v_new: scratch.v_new,
                                        slot_ids: (!global).then_some(inputs().slot_ids),
                                        block_tables: global.then_some(inputs().block_tables),
                                        seq_lens: inputs().seq_lens,
                                        q_start: inputs().q_start,
                                        q_len: inputs().q_len,
                                        prefix_len: inputs().prefix_len,
                                        live_batch: inputs().live_batch,
                                        sink: (!global).then_some(span(42)),
                                        cos_sin: fold.then_some(tables[usize::from(!global)]),
                                        output: scratch.attention,
                                        workspace,
                                    };
                                    for (part, stage) in [
                                        (Part::Partial, Stage::AttentionPartial(layer)),
                                        (Part::Merge, Stage::AttentionMerge(layer)),
                                    ] {
                                        let planned = if global {
                                            crate::attention_plan::paged(
                                                stage,
                                                tp,
                                                served_items,
                                                phase,
                                                switches,
                                                kv,
                                                scale_bits,
                                                live_tokens,
                                                live_items,
                                                context,
                                                inputs(),
                                                &scratch,
                                                pool,
                                                workspace,
                                            )
                                        } else {
                                            crate::attention_plan::ring(
                                                stage,
                                                tp,
                                                ring,
                                                served_items,
                                                phase,
                                                switches,
                                                live_tokens,
                                                live_items,
                                                context,
                                                inputs(),
                                                &scratch,
                                                ring_state,
                                                workspace,
                                                &views,
                                            )
                                        }
                                        .unwrap_or_else(|error| {
                                            panic!(
                                                "{error:?} {tp:?} {items} {phase:?} \
                                                 {live_tokens} {live_items} {context}"
                                            )
                                        });
                                        let lowered = ops
                                            .attention(
                                                if global {
                                                    Layer::Global
                                                } else {
                                                    Layer::Sliding
                                                },
                                                part,
                                                phase,
                                                bucket,
                                                live_tokens,
                                                ops_items,
                                                context,
                                                &step,
                                                if global { &paged } else { &rings },
                                            )
                                            .unwrap();
                                        assert_eq!(
                                            resolved(&launches(lowered.into_iter().collect())),
                                            resolved(&planned.into_iter().collect::<Vec<_>>()),
                                            "{tp:?} {kv:?} {items} {switches:?} {phase:?} \
                                             {live_tokens} {live_items} {context} {stage:?}"
                                        );
                                        compared += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(compared > 10_000, "{compared}");
    }

    /// MiMo's DFlash block attention before it lowered through the backend:
    /// the catalog's block variant over the layer's circular cache planes,
    /// planned for the instance's items, at least 32, and the largest
    /// context.
    fn planned_block(
        tp: Tp,
        slots: u32,
        live_tokens: u32,
        [k_plane, v_plane]: [DeviceBufferSpanV1; 2],
        step: &AttentionViews,
    ) -> Vec<crate::pipeline::PreparedLaunch> {
        use crate::dispatch::BufferBindings;
        use crate::sm120::launch::attention::{Buffers, mimo_variants, plan_for_bucket};
        let variant = *mimo_variants(tp)
            .iter()
            .find(|variant| variant.name.ends_with("dflash_block"))
            .unwrap();
        let mut bindings = BufferBindings::default();
        let q = bindings.bind(step.q);
        let k_cache = bindings.bind(k_plane);
        let v_cache = bindings.bind(v_plane);
        let sink = bindings.bind(step.sink.unwrap());
        let (k_map, v_map) = variant.cache_maps(slots, k_cache, v_cache).unwrap();
        let items = slots.max(crate::contract::DECODE_ENTRY_ROWS);
        let variant = if items > variant.query.max_batch_capacity {
            variant.with_batch_capacity(items).unwrap()
        } else {
            variant
        };
        let buffers = Buffers {
            q,
            k_cache,
            v_cache,
            k_page_stride_bytes: crate::sm120::launch::tight_page_stride(
                &variant,
                variant.query.qk_dim,
            ),
            v_page_stride_bytes: crate::sm120::launch::tight_page_stride(
                &variant,
                variant.query.v_dim,
            ),
            k_map: Some(k_map),
            v_map: Some(v_map),
            block_tables: None,
            slot_ids: Some(bindings.bind(step.slot_ids.unwrap())),
            seq_lens: Some(bindings.bind(step.seq_lens)),
            q_start: Some(bindings.bind(step.q_start)),
            q_len: Some(bindings.bind(step.q_len)),
            prefix_len: Some(bindings.bind(step.prefix_len)),
            live_batch: bindings.bind(step.live_batch),
            k_new: Some(bindings.bind(step.k_new)),
            v_new: Some(bindings.bind(step.v_new)),
            sink: Some(sink),
            output: bindings.bind(step.output),
            workspace: bindings.bind(step.workspace),
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let planned = plan_for_bucket(
            variant,
            items,
            kernel_core::contract::RuntimeCounts {
                m: live_tokens,
                batch: items,
                context: tp.geometry.context_tokens(),
                routed_rows: live_tokens.saturating_mul(8),
                split_k: 1,
            },
            buffers,
        )
        .unwrap();
        let mut result = vec![crate::pipeline::PreparedLaunch {
            symbol: planned.partial_symbol,
            rows: live_tokens,
            plan: planned.partial,
            arguments: planned.partial_args,
            bindings: bindings.clone(),
        }];
        if !planned.direct_output {
            result.push(crate::pipeline::PreparedLaunch {
                symbol: planned.merge_symbol,
                rows: live_tokens,
                plan: planned.merge,
                arguments: planned.merge_args,
                bindings,
            });
        }
        result
    }

    /// The prepared DFlash block attention launches what MiMo's DFlash
    /// program planned, partial and merge, over every layer's cache planes
    /// and per-item arrays of the served items as the program binds them,
    /// at every verify bucket and live row count an 8-, 16-, 32- and
    /// 256-item instance serves, at TP2 and TP4.
    #[test]
    fn block_attentions_launch_what_the_dflash_program_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            let scratch = scratch(tp);
            // The verify inputs' per-item arrays hold the served items.
            let step = |sink, items: u32| {
                let per_item = |buffer| {
                    DeviceBufferSpanV1::new(
                        BundleDeviceBufferHandleV1::from_parts(buffer, 1),
                        256,
                        4 * u64::from(items),
                    )
                };
                AttentionViews {
                    q: scratch.q,
                    k_new: scratch.k_new,
                    v_new: scratch.v_new,
                    slot_ids: Some(per_item(14)),
                    block_tables: None,
                    seq_lens: per_item(18),
                    q_start: per_item(15),
                    q_len: per_item(16),
                    prefix_len: per_item(17),
                    live_batch: inputs().live_batch,
                    sink: Some(sink),
                    cos_sin: None,
                    output: scratch.attention,
                    workspace: span(40),
                }
            };
            for items in [8, 16, 32, 256] {
                let switches = Switches::configured();
                let backend = crate::sm120::backend(tp, switches, true).unwrap();
                let ops = Ops::prepare(
                    &*backend,
                    &Plan {
                        tp,
                        switches,
                        kv: DType::Bf16,
                        items,
                        target: &crate::sm120::own_ops::Sm120Own::new(tp),
                        state: &|_| [u64::MAX; 3],
                        gathers: &any_gathers,
                        dflash: Some([u64::MAX; 2]),
                        own: crate::ops::OwnPlan::new(
                            &crate::sm120::own_ops::Sm120Own::new(tp),
                            items,
                        ),
                    },
                )
                .unwrap();
                let cache = DeviceBufferSpanV1::new(
                    BundleDeviceBufferHandleV1::from_parts(44, 1),
                    4096,
                    crate::dflash::program::cache_bytes(tp, items),
                );
                for layer in 0..tp.geometry.dflash().num_hidden_layers {
                    let (k, v) =
                        crate::dflash::program::cache_planes(cache, layer, tp, items).unwrap();
                    // As the program binds them: each slot's cache follows
                    // the last.
                    let stride = k.length / u64::from(items);
                    let planes = CacheViews {
                        k,
                        v,
                        slots: items,
                        strides: [stride, stride],
                        scale_bits: [1.0_f32.to_bits(); 2],
                    };
                    let step = step(span(50 + layer), items);
                    for (phase, bucket) in phases() {
                        if !matches!(phase, Phase::Verify(_)) || !served(phase, items).unwrap() {
                            continue;
                        }
                        let rows = phase.capacity().unwrap();
                        let mut lives = vec![1, rows / 2 + 1, rows];
                        lives.dedup();
                        for live_tokens in lives {
                            let mut lowered = Vec::new();
                            for part in [Part::Partial, Part::Merge] {
                                lowered.extend(
                                    ops.attention(
                                        Layer::Block,
                                        part,
                                        phase,
                                        bucket,
                                        live_tokens,
                                        items,
                                        tp.geometry.context_tokens(),
                                        &step,
                                        &planes,
                                    )
                                    .unwrap(),
                                );
                            }
                            assert_eq!(
                                resolved(&launches(lowered)),
                                resolved(&planned_block(tp, items, live_tokens, [k, v], &step)),
                                "{tp:?} {items} {phase:?} {live_tokens} layer {layer}"
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
        assert!(compared > 50, "{compared}");
    }
}
