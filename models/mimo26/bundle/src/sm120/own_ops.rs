//! MiMo's own kernels on SM120: [`OwnOps`] over the bundle's cubins.
//!
//! Each prepared op launches the entry the bundle builds for its topology,
//! with the plan and arguments today's builders give for the same inputs
//! (`row_plan` ring builders, `prefix::prefix_copy`, `retain`,
//! `dflash::program`, `mtp::program`). Ops are refused where an entry's
//! compiled constants (ring geometry, drafter heads, hidden widths, rank
//! count) or its 16-byte vectors do not fit the contract.
//!
//! Views carry no offset, so a host binds each view at the alignment its role
//! record states (16 bytes for vector-loaded roles), as the builders require
//! of their spans. Guarantees are the catalog declaration of the symbol an
//! op launches, none for a symbol without one.

use kernel_core::backend::{Counts, Launch, Need, Prepared, Refused, View};
use kernel_core::contract::{Argument, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernels_sm120::backend::own::{self, OwnKernels, buffer, launch, rows, rule};
use model_mimo26_contract::own_ops::OwnOps;
use model_mimo26_contract::own_ops::dflash::{
    CacheWrite, CacheWriteBuffers, CopyRows, CopyRowsBuffers, Drafts, DraftsBuffers, MaskRows,
    MaskRowsBuffers,
};
use model_mimo26_contract::own_ops::encoder::{EmbeddingSum, EmbeddingSumBuffers, Rvq, RvqBuffers};
use model_mimo26_contract::own_ops::retain::{
    GatherRows, GatherRowsBuffers, RetainRows, RetainRowsBuffers, row_bytes,
};
use model_mimo26_contract::own_ops::ring::{
    Direction, RingBackup, RingBackupBuffers, RingLayout, RingPrefixCopy, RingPrefixCopyBuffers,
    RingRestore, RingRestoreBuffers,
};
use model_mimo26_contract::sm120::encoders::rows as encoder_rows;

use crate::topology::Tp;

/// Threads of a retained-row, row copy or mask rows block.
const ROW_THREADS: u32 = 256;
/// Threads of a prefix copy block; each moves one 16-byte vector.
const PREFIX_THREADS: u32 = 256;
/// Threads of a drafts block: one row each.
const DRAFT_THREADS: u32 = 32;

/// MiMo's own kernels as the bundle compiles them for one topology.
#[derive(Clone, Copy, Debug)]
pub struct Sm120Own {
    tp: Tp,
}

impl Sm120Own {
    pub fn new(tp: Tp) -> Sm120Own {
        Sm120Own { tp }
    }

    /// The topology the kernels are compiled for.
    pub fn tp(&self) -> Tp {
        self.tp
    }
}

/// A launch of a bundle-owned entry: PDL, no shared memory or cluster.
fn plan(grid: [u32; 3], threads: u32) -> LaunchPlan {
    own::pdl_plan(grid, threads)
}

/// MiMo's own kernels: their implementation names, the bundle's cubins and
/// the declarations of what they keep.
const OWN: OwnKernels = OwnKernels {
    prefix: "sm120.mimo",
    embedded,
    declared,
    check: own::geometry,
};

/// Whether the bundle's cubins hold `symbol`.
fn embedded(symbol: &str) -> bool {
    crate::CUBINS
        .iter()
        .any(|cubin| cubin.symbols.contains(&symbol))
}

/// Whether BF16 widths and strides keep every row on 16-byte vectors.
fn vectors(values: &[u32]) -> bool {
    values.iter().all(|value| value.is_multiple_of(8))
}

/// The entries of a launch's per-item tables (`Counts::items`): at least
/// one, which each table's view must hold.
fn items(counts: Counts) -> Result<u64, Refusal> {
    if counts.items == 0 {
        return Err(Refusal("an own op over items needs their count"));
    }
    Ok(u64::from(counts.items))
}

/// The invariance the catalog declares for `symbol`, none when it has no
/// declaration: a declaration covers its own symbol only.
fn declared(symbol: &str) -> Invariance {
    own::declared_in(&crate::sm120::declared_kernels(), symbol)
}

/// Prepares the op launching `symbol` for some rows, with the guarantees
/// its declaration claims, after checking the need.
fn prepare<B: 'static>(
    name: &str,
    symbol: String,
    need: Need,
    planner: impl Fn(&B, Counts) -> Result<Launch, Refusal> + Send + Sync + 'static,
) -> Result<Prepared<B>, Refused> {
    let info = OWN.info(name, vec![(symbol, 0)], need)?;
    OWN.prepare_one(info, need, planner)
}

impl Sm120Own {
    /// Refuses a ring other than the one the ring entries are compiled for:
    /// the topology's window, layers and entries, at a slot count with an
    /// entry, serving at most its slots per launch.
    fn ring(&self, layout: RingLayout, need: Need) -> Result<(), Refused> {
        layout.validate()?;
        if layout != RingLayout::of(self.tp, layout.slots) {
            return Err(rule(
                "the ring entries are compiled for the topology's ring",
            ));
        }
        if need.capacity == 0 || need.capacity > layout.slots {
            return Err(rule("a ring launch covers at most the ring's slots"));
        }
        Ok(())
    }

    /// The ring backup or restore of `name`: one block per (item, recent
    /// position, layer), a thread per 16-byte vector of a K and V entry.
    fn ring_copy<B: 'static>(
        &self,
        name: &'static str,
        layout: RingLayout,
        span: u32,
        need: Need,
        views: fn(&B) -> [View; 5],
    ) -> Result<Prepared<B>, Refused> {
        self.ring(layout, need)?;
        let threads = crate::topology::ring_threads(self.tp);
        // The widest launch: every item of the capacity at the full span.
        need.capacity
            .checked_mul(span)
            .map(|blocks| plan([blocks, layout.layers, 1], threads))
            .ok_or(Refusal("a ring launch's blocks overflow"))
            .and_then(|widest| widest.validate_geometry())?;
        let name = format!("ring_{name}");
        let symbol = super::ring_entries::ring_symbol(self.tp, &name, layout.slots);
        let capacity = need.capacity;
        let planned = symbol.clone();
        prepare(&name, symbol, need, move |buffers: &B, counts| {
            let items = rows(counts, capacity)?;
            let blocks = items
                .checked_mul(span)
                .ok_or(Refusal("a ring launch's blocks overflow"))?;
            let [state, slot_ids, first, count, live_items] = views(buffers);
            state.covers(2 * layout.working())?;
            for table in [slot_ids, first, count] {
                table.covers(4 * u64::from(items))?;
            }
            live_items.covers(4)?;
            Ok(launch(
                &planned,
                blocks,
                plan([blocks, layout.layers, 1], threads),
                vec![
                    buffer(state),
                    buffer(slot_ids),
                    buffer(first),
                    buffer(count),
                    buffer(live_items),
                    Argument::U32(span),
                ],
            ))
        })
    }
}

impl OwnOps for Sm120Own {
    fn name(&self) -> &str {
        "sm120"
    }

    fn ring_backup(
        &self,
        op: &RingBackup,
        need: Need,
    ) -> Result<Prepared<RingBackupBuffers<View>>, Refused> {
        op.validate()?;
        self.ring_copy(
            "backup",
            op.layout,
            op.span,
            need,
            |b: &RingBackupBuffers<View>| {
                [b.state, b.slot_ids, b.q_len, b.prefix_len, b.live_items]
            },
        )
    }

    fn ring_restore(
        &self,
        op: &RingRestore,
        need: Need,
    ) -> Result<Prepared<RingRestoreBuffers<View>>, Refused> {
        op.validate()?;
        self.ring_copy(
            "restore",
            op.layout,
            op.span,
            need,
            |b: &RingRestoreBuffers<View>| [b.state, b.slot_ids, b.starts, b.counts, b.live_items],
        )
    }

    fn ring_prefix_copy(
        &self,
        op: &RingPrefixCopy,
        need: Need,
    ) -> Result<Prepared<RingPrefixCopyBuffers<View>>, Refused> {
        op.validate()?;
        let layout = op.layout;
        self.ring(layout, Need::rows(1))?;
        if !op.page_bytes.is_multiple_of(16) || !op.drafter_bytes.is_multiple_of(16) {
            return Err(rule("the prefix copy moves 16-byte vectors"));
        }
        let pages =
            u32::try_from(op.pages).map_err(|_| rule("a prefix snapshot has too many pages"))?;
        // One thread per 16-byte vector of the pieces.
        let blocks = u32::try_from((op.bytes() / 16).div_ceil(u64::from(PREFIX_THREADS)))
            .map_err(|_| rule("a prefix snapshot needs too many blocks"))?;
        plan([blocks, 1, 1], PREFIX_THREADS).validate_geometry()?;
        let symbol = super::ring_entries::prefix_copy_symbol(self.tp, layout.slots);
        let planned = symbol.clone();
        let op = *op;
        prepare(
            "ring_prefix_copy",
            symbol,
            need,
            move |b: &RingPrefixCopyBuffers<View>, _| {
                if b.slot >= layout.slots {
                    return Err(Refusal("a prefix copy slot is outside the ring"));
                }
                b.state.covers(layout.working())?;
                b.pool.covers(op.page_bytes)?;
                b.pages.covers(u64::from(pages) * 8)?;
                let drafter = match (b.drafter, op.drafter_bytes) {
                    (None, 0) => Argument::NullDevicePointer,
                    (Some(view), bytes) if bytes > 0 && view.bytes == bytes => buffer(view),
                    _ => {
                        return Err(Refusal(
                            "a prefix copy's drafter piece does not match its op",
                        ));
                    }
                };
                Ok(launch(
                    &planned,
                    // A snapshot copy moves one slot's state and has no token rows.
                    1,
                    plan([blocks, 1, 1], PREFIX_THREADS),
                    vec![
                        buffer(b.state),
                        drafter,
                        Argument::U64(op.drafter_bytes),
                        buffer(b.pool),
                        buffer(b.pages),
                        Argument::U32(pages),
                        Argument::U64(op.page_bytes),
                        Argument::U32(b.slot),
                        Argument::U32(u32::from(op.direction == Direction::Capture)),
                    ],
                ))
            },
        )
    }

    fn retain_rows(
        &self,
        op: &RetainRows,
        need: Need,
    ) -> Result<Prepared<RetainRowsBuffers<View>>, Refused> {
        op.validate()?;
        if !vectors(&[op.width, op.input_stride, op.history.stride]) {
            return Err(rule("retained rows move 16-byte vectors"));
        }
        let symbol = crate::retain::retain_rows_symbol(self.tp);
        let (planned, op, capacity) = (symbol.clone(), *op, need.capacity);
        prepare(
            "retain_rows",
            symbol,
            need,
            move |b: &RetainRowsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.input.covers(row_bytes(rows, op.input_stride, op.width))?;
                b.history.covers(op.history.bytes(op.width))?;
                let items = items(counts)?;
                for table in [b.slot_ids, b.q_start, b.prefix_len] {
                    table.covers(4 * items)?;
                }
                b.token_batch.covers(4 * u64::from(rows))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1], ROW_THREADS),
                    vec![
                        buffer(b.input),
                        buffer(b.history),
                        buffer(b.slot_ids),
                        buffer(b.token_batch),
                        buffer(b.q_start),
                        buffer(b.prefix_len),
                        buffer(b.live_rows),
                        Argument::U32(op.width),
                        Argument::U32(op.input_stride),
                        Argument::U32(op.history.stride),
                        Argument::U32(op.history.capacity),
                        Argument::U32(op.history.slots),
                    ],
                ))
            },
        )
    }

    fn gather_rows(
        &self,
        op: &GatherRows,
        need: Need,
    ) -> Result<Prepared<GatherRowsBuffers<View>>, Refused> {
        op.validate()?;
        if !vectors(&[op.width, op.output_stride, op.history.stride]) {
            return Err(rule("gathered rows move 16-byte vectors"));
        }
        let symbol = crate::retain::gather_rows_symbol(self.tp);
        let (planned, op, capacity) = (symbol.clone(), *op, need.capacity);
        prepare(
            "gather_rows",
            symbol,
            need,
            move |b: &GatherRowsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.history.covers(op.history.bytes(op.width))?;
                b.output
                    .covers(row_bytes(rows, op.output_stride, op.width))?;
                b.slot_ids.covers(4 * items(counts)?)?;
                b.token_batch.covers(4 * u64::from(rows))?;
                b.positions.covers(4 * u64::from(rows))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1], ROW_THREADS),
                    vec![
                        buffer(b.history),
                        buffer(b.output),
                        buffer(b.slot_ids),
                        buffer(b.token_batch),
                        buffer(b.positions),
                        buffer(b.live_rows),
                        Argument::U32(op.width),
                        Argument::U32(op.history.stride),
                        Argument::U32(op.output_stride),
                        Argument::U32(op.history.capacity),
                        Argument::U32(op.history.slots),
                    ],
                ))
            },
        )
    }

    fn dflash_copy_rows(
        &self,
        op: &CopyRows,
        need: Need,
    ) -> Result<Prepared<CopyRowsBuffers<View>>, Refused> {
        op.validate()?;
        if !vectors(&[op.width, op.input_stride.unwrap_or(0), op.output_stride]) {
            return Err(rule("the row copy moves 16-byte vectors"));
        }
        let symbol = crate::topology::bundle_symbol(self.tp, "dflash_copy_rows");
        let (planned, op, capacity) = (symbol.clone(), *op, need.capacity);
        prepare(
            "dflash_copy_rows",
            symbol,
            need,
            move |b: &CopyRowsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                let input = match (b.input, op.input_stride) {
                    (Some(view), Some(stride)) => {
                        view.covers(row_bytes(rows, stride, op.width))?;
                        buffer(view)
                    }
                    (None, None) => Argument::NullDevicePointer,
                    _ => return Err(Refusal("a row copy's input does not match its op")),
                };
                b.output
                    .covers(row_bytes(rows, op.output_stride, op.width))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1], ROW_THREADS),
                    vec![
                        input,
                        buffer(b.output),
                        buffer(b.live_rows),
                        Argument::U32(op.width),
                        Argument::U32(op.input_stride.unwrap_or(0)),
                        Argument::U32(op.output_stride),
                        Argument::U32(op.scale_bits),
                    ],
                ))
            },
        )
    }

    fn dflash_mask_rows(
        &self,
        op: &MaskRows,
        need: Need,
    ) -> Result<Prepared<MaskRowsBuffers<View>>, Refused> {
        op.validate()?;
        if op.hidden != self.tp.geometry.dflash().hidden_size {
            return Err(rule(
                "the mask rows entry is compiled for the drafter's hidden width",
            ));
        }
        let symbol = crate::topology::bundle_symbol(self.tp, "dflash_mask_rows");
        let (planned, row, capacity) = (symbol.clone(), u64::from(op.hidden) * 2, need.capacity);
        prepare(
            "dflash_mask_rows",
            symbol,
            need,
            move |b: &MaskRowsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.activation.covers(u64::from(rows) * row)?;
                b.mask.covers(row)?;
                b.prefix_len.covers(4 * items(counts)?)?;
                b.positions.covers(4 * u64::from(rows))?;
                b.token_batch.covers(4 * u64::from(rows))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1], ROW_THREADS),
                    vec![
                        buffer(b.activation),
                        buffer(b.mask),
                        buffer(b.positions),
                        buffer(b.prefix_len),
                        buffer(b.token_batch),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn dflash_cache_write(
        &self,
        op: &CacheWrite,
        need: Need,
    ) -> Result<Prepared<CacheWriteBuffers<View>>, Refused> {
        op.validate()?;
        let compiled = CacheWrite::of(self.tp, op.slots, op.k_stride, op.v_stride);
        if (CacheWrite {
            v_scale_bits: op.v_scale_bits,
            ..compiled
        }) != *op
            || op.slots > crate::contract::DECODE_BATCH
        {
            return Err(rule(
                "the cache write entry is compiled for the drafter's cache",
            ));
        }
        if !vectors(&[op.k_stride, op.v_stride]) {
            return Err(rule("the cache write moves 16-byte vectors"));
        }
        let symbol = crate::topology::bundle_symbol(self.tp, "dflash_cache_write");
        let (planned, op, capacity) = (symbol.clone(), *op, need.capacity);
        // A thread per 16 bytes of the K entry, then of the V entry.
        let threads = op.entry() / 4;
        prepare(
            "dflash_cache_write",
            symbol,
            need,
            move |b: &CacheWriteBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                // A padded grid's blocks past the live rows return.
                let grid = rows.max(counts.padded_rows);
                if grid > capacity {
                    return Err(Refusal(
                        "an own op launches from one row to its prepared capacity",
                    ));
                }
                b.k.covers(row_bytes(rows, op.k_stride, op.entry()))?;
                b.v.covers(row_bytes(rows, op.v_stride, op.entry()))?;
                b.k_cache.covers(op.plane_bytes())?;
                b.v_cache.covers(op.plane_bytes())?;
                let items = items(counts)?;
                b.slot_ids.covers(4 * items)?;
                b.last_positions.covers(8 * items)?;
                b.token_batch.covers(4 * u64::from(rows))?;
                b.positions.covers(4 * u64::from(rows))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    grid,
                    plan([grid, 1, 1], threads),
                    vec![
                        buffer(b.k),
                        buffer(b.v),
                        buffer(b.k_cache),
                        buffer(b.v_cache),
                        buffer(b.slot_ids),
                        buffer(b.token_batch),
                        buffer(b.positions),
                        buffer(b.last_positions),
                        buffer(b.live_rows),
                        Argument::U32(op.k_stride),
                        Argument::U32(op.v_stride),
                        Argument::U32(op.v_scale_bits),
                    ],
                ))
            },
        )
    }

    fn dflash_drafts(
        &self,
        op: &Drafts,
        need: Need,
    ) -> Result<Prepared<DraftsBuffers<View>>, Refused> {
        op.validate()?;
        if op.ranks != self.tp.ranks() {
            return Err(rule(
                "the drafts entry is compiled for the topology's ranks",
            ));
        }
        let symbol = crate::topology::bundle_symbol(self.tp, "dflash_drafts");
        let (planned, ranks, capacity) = (symbol.clone(), op.ranks, need.capacity);
        prepare(
            "dflash_drafts",
            symbol,
            need,
            move |b: &DraftsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.gathered
                    .covers(u64::from(ranks) * 2 * u64::from(rows) * 4)?;
                for table in [b.token_ids, b.positions, b.token_batch] {
                    table.covers(4 * u64::from(rows))?;
                }
                b.prefix_len.covers(4 * items(counts)?)?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows.div_ceil(DRAFT_THREADS), 1, 1], DRAFT_THREADS),
                    vec![
                        buffer(b.gathered),
                        buffer(b.token_ids),
                        buffer(b.positions),
                        buffer(b.prefix_len),
                        buffer(b.token_batch),
                        buffer(b.live_rows),
                        Argument::U32(rows),
                    ],
                ))
            },
        )
    }

    fn encoder_rvq(&self, op: &Rvq, need: Need) -> Result<Prepared<RvqBuffers<View>>, Refused> {
        encoder_rvq(op, need)
    }

    fn encoder_embedding_sum(
        &self,
        op: &EmbeddingSum,
        need: Need,
    ) -> Result<Prepared<EmbeddingSumBuffers<View>>, Refused> {
        encoder_embedding_sum(op, need)
    }
}

/// The rows an encoder kernel grids: one block of `rows::THREADS` per row
/// of the prepared capacity, each reading the live count on the device (the
/// bundle's `mimo_encoder` cubin, entries named as `sm120::encoders::rows` names
/// them).
fn encoder_capacity(need: Need) -> Result<u32, Refused> {
    if need.capacity == 0 {
        return Err(rule("an encoder kernel serves at least one row"));
    }
    Ok(need.capacity)
}

fn encoder_rvq(op: &Rvq, need: Need) -> Result<Prepared<RvqBuffers<View>>, Refused> {
    op.validate()?;
    let capacity = encoder_capacity(need)?;
    let symbol = format!(
        "chop_mimo_enc_rvq{}_f32_d{}_codes{}_v1",
        op.stages, op.dim, op.max_codes
    );
    let (planned, op) = (symbol.clone(), *op);
    prepare(
        "encoder_rvq",
        symbol,
        need,
        move |b: &RvqBuffers<View>, counts| {
            let rows = rows(counts, capacity)?;
            let (dim, stages) = (u64::from(op.dim), u64::from(op.stages));
            b.input.covers(u64::from(rows) * dim * 4)?;
            b.codebooks.covers(u64::from(op.codebook_rows) * dim * 4)?;
            b.codebook_offsets.covers((stages + 1) * 4)?;
            b.codes.covers(u64::from(rows) * stages * 4)?;
            b.live_rows.covers(4)?;
            Ok(launch(
                &planned,
                rows,
                plan([capacity, 1, 1], encoder_rows::THREADS),
                vec![
                    buffer(b.input),
                    buffer(b.codebooks),
                    buffer(b.codebook_offsets),
                    buffer(b.codes),
                    buffer(b.live_rows),
                ],
            ))
        },
    )
}

fn encoder_embedding_sum(
    op: &EmbeddingSum,
    need: Need,
) -> Result<Prepared<EmbeddingSumBuffers<View>>, Refused> {
    op.validate()?;
    let capacity = encoder_capacity(need)?;
    let symbol = format!(
        "chop_mimo_enc_sum{}_f32_vocab{}_w{}_v1",
        op.tables, op.vocab, op.width
    );
    let (planned, op) = (symbol.clone(), *op);
    prepare(
        "encoder_embedding_sum",
        symbol,
        need,
        move |b: &EmbeddingSumBuffers<View>, counts| {
            let rows = rows(counts, capacity)?;
            let (tables, width) = (u64::from(op.tables), u64::from(op.width));
            b.tables.covers(tables * u64::from(op.vocab) * width * 2)?;
            b.codes.covers(u64::from(rows) * tables * 4)?;
            b.output.covers(u64::from(rows) * width * 4)?;
            b.live_rows.covers(4)?;
            Ok(launch(
                &planned,
                rows,
                plan([capacity, 1, 1], encoder_rows::THREADS),
                vec![
                    buffer(b.tables),
                    buffer(b.codes),
                    buffer(b.output),
                    buffer(b.live_rows),
                ],
            ))
        },
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bundle_runtime::phase::Phase;
    use bundle_runtime::state::{Extent, Holder};
    use model_mimo26_contract::own_ops::retain::History;
    use model_mimo26_contract::state::{Drafter, request_state};
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    use super::*;
    use crate::contract::{KV_FORMATS, MAX_VERIFY_ROWS, Mimo};
    use crate::pipeline::PreparedLaunch;
    use crate::program_lower::RankStage;
    use crate::ring::RingGeometry;
    use crate::scratch::RankScratch;
    use crate::upload::DeviceInputs;
    use kernel_core::contract::ArgumentList;

    /// Room for any test buffer: a full prefill of 4096-wide BF16 rows.
    const BIG: u64 = 1 << 34;

    /// A span of its own buffer `id`.
    fn span(id: u32, bytes: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, id), 0, bytes)
    }

    /// The view a span binds as: its buffer and offset name the handle.
    fn view(span: DeviceBufferSpanV1) -> View {
        View {
            handle: u64::from(span.buffer.slot()) << 32 | span.offset,
            bytes: span.length,
        }
    }

    /// A builder's launch as a prepared op's: each bound span becomes its
    /// view's handle.
    fn built(launch: PreparedLaunch) -> Launch {
        let arguments = launch
            .arguments
            .0
            .iter()
            .map(|argument| match argument {
                Argument::DeviceBuffer(id) => {
                    Argument::DeviceBuffer(view(launch.bindings.get(*id).unwrap()).handle)
                }
                other => other.clone(),
            })
            .collect();
        Launch {
            symbol: launch.symbol,
            rows: launch.rows,
            plan: launch.plan,
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }
    }

    /// The counts of a launch over `rows` rows and as many items.
    fn counts(rows: u32) -> Counts {
        Counts::rows(rows).with_items(rows, 0)
    }

    fn one<B>(prepared: &Prepared<B>, buffers: &B, rows: u32) -> Launch {
        let mut launches = prepared
            .launches(buffers, View::EMPTY, counts(rows))
            .unwrap();
        assert_eq!(launches.len(), 1);
        launches.remove(0)
    }

    fn inputs() -> DeviceInputs {
        DeviceInputs {
            token_ids: span(10, BIG),
            positions: span(11, BIG),
            token_batch: span(12, BIG),
            block_tables: span(13, BIG),
            slot_ids: span(14, BIG),
            q_start: span(15, BIG),
            q_len: span(16, BIG),
            prefix_len: span(17, BIG),
            seq_lens: span(18, BIG),
            live_batch: span(19, 4),
            last_positions: span(20, BIG),
            live_tokens: span(21, 4),
            logit_rows: span(22, BIG),
            row_indices: span(23, BIG),
        }
    }

    /// What a rank stage borrows.
    struct Rank {
        scratch: RankScratch,
        views: BTreeMap<String, DeviceBufferSpanV1>,
        /// The instance's prepared ops, which the MTP pair's norm launches.
        ops: crate::ops::Ops,
    }

    impl Rank {
        fn new(tp: Tp) -> Rank {
            Rank::serving(tp, crate::contract::DECODE_ENTRY_ROWS)
        }

        /// A rank of an instance serving `items` items, which drafts with
        /// DFlash, and with MTP where it serves that many.
        fn serving(tp: Tp, items: u32) -> Rank {
            Rank {
                scratch: RankScratch::allocate(0, tp, |bytes| Ok(span(30, bytes))).unwrap(),
                views: BTreeMap::new(),
                ops: crate::ops::Ops::prepare(
                    &*crate::sm120::backend(tp, crate::switches::Switches::configured(), true)
                        .unwrap(),
                    &crate::ops::Plan {
                        tp,
                        switches: crate::switches::Switches::configured(),
                        kv: kernel_core::contract::DType::Bf16,
                        items,
                        target: &crate::sm120::own_ops::Sm120Own::new(tp),
                        state: &|_| [u64::MAX; 3],
                        gathers: &|_| crate::ops::GatherRegions {
                            rows: [u64::MAX; 2],
                            exchange: Some([u64::MAX; 2]),
                        },
                        dflash: Some([u64::MAX; 2]),
                        own: crate::ops::OwnPlan {
                            mtp: items <= crate::contract::DRAFT_ITEMS,
                            ..crate::ops::OwnPlan::new(&Sm120Own::new(tp), items)
                        },
                    },
                )
                .unwrap(),
            }
        }

        /// Rank `rank`'s stage of `phase` over `live` rows, in an instance
        /// of `items` items.
        fn stage(&self, tp: Tp, rank: u32, phase: Phase, live: u32, items: u32) -> RankStage<'_> {
            RankStage {
                services: None,
                tp,
                rank,
                phase,
                live_tokens: live,
                live_batch: items,
                context: 1,
                ga_kv: kernel_core::contract::DType::Bf16,
                ga_kv_scales: &[],
                views: &self.views,
                projections: None,
                inputs: inputs(),
                scratch: &self.scratch,
                activation: span(40, BIG),
                normalized: span(41, BIG),
                workspace: span(42, BIG),
                local_logits: span(43, BIG),
                scored_prompt_logits: false,
                logit_rows: &[],
                logits_gather: crate::ops::LogitsBuffers {
                    local: span(43, BIG),
                    shard: crate::topology::vocab_per_rank(tp),
                    outputs: vec![span(44, BIG); tp.count()],
                    flags: vec![span(45, BIG); tp.count()],
                    epochs: vec![span(46, 4); tp.count()],
                },
                exchange_logits: false,
                hidden_inboxes: vec![span(47, BIG); tp.count()],
                hidden_flags: vec![span(48, BIG); tp.count()],
                hidden_epochs: vec![span(49, 4); tp.count()],
                hidden_gathered: vec![span(50, BIG); tp.count()],
                ring_state: span(51, BIG),
                ring: RingGeometry::new(tp, items),
                items: bundle_runtime::geometry::Items::new(items).unwrap(),
                switches: crate::switches::Switches::configured(),
                ops: &self.ops,
                bucket: crate::scratch::COMM_BUCKETS
                    .iter()
                    .position(|&rows| Some(rows) == phase.capacity_in::<Mimo>())
                    .unwrap(),
                kv_pool: span(52, BIG),
            }
        }
    }

    fn capacity(phase: Phase) -> u32 {
        phase.capacity_in::<Mimo>().unwrap()
    }

    /// Decode, verify and prefill: (phase, live rows).
    fn phases() -> [(Phase, u32); 4] {
        [
            (Phase::Decode(5), 5),
            (Phase::Decode(32), 32),
            (Phase::Verify(24), 24),
            (Phase::Prefill, 1908),
        ]
    }

    /// An own op prepared for no rows is refused, as every model's own ops
    /// refuse it.
    #[test]
    fn an_own_op_without_a_capacity_is_refused() {
        for tp in Tp::SUPPORTED {
            let mask = MaskRows {
                hidden: tp.geometry.dflash().hidden_size,
            };
            let own = Sm120Own::new(tp);
            assert!(own.dflash_mask_rows(&mask, Need::rows(1)).is_ok());
            assert!(matches!(
                own.dflash_mask_rows(&mask, Need::rows(0)),
                Err(Refused::Rule(_))
            ));
        }
    }

    #[test]
    fn ring_backup_and_restore_launch_what_the_builders_plan() {
        for tp in Tp::SUPPORTED {
            let own = Sm120Own::new(tp);
            for slots in [32, 8, 256] {
                let ring = RingGeometry::new(tp, slots);
                let layout = RingLayout::of(tp, slots);
                let state = span(1, 2 * ring.working);
                let tables = [span(2, 1024), span(3, 1024), span(4, 1024), span(5, 4)];
                for (phase, items) in [
                    (Phase::Decode(5), 5),
                    (Phase::Verify(24), 3),
                    (Phase::Prefill, 2),
                ] {
                    let builder = crate::row_plan::ring_backup(
                        tp,
                        ring,
                        phase,
                        items,
                        crate::row_plan::RingBackupViews {
                            ring_state: state,
                            slot_ids: tables[0],
                            q_len: tables[1],
                            prefix_len: tables[2],
                            live_batch: tables[3],
                        },
                    )
                    .unwrap();
                    let span = match phase {
                        Phase::Decode(_) => 1,
                        Phase::Verify(_) => MAX_VERIFY_ROWS,
                        Phase::Prefill => layout.tokens,
                    };
                    let prepared = own
                        .ring_backup(&RingBackup { layout, span }, Need::rows(slots))
                        .unwrap();
                    // Every slot count's backup is declared a copy.
                    assert_eq!(prepared.info().guarantees, Invariance::COPY);
                    let buffers = RingBackupBuffers {
                        state: view(state),
                        slot_ids: view(tables[0]),
                        q_len: view(tables[1]),
                        prefix_len: view(tables[2]),
                        live_items: view(tables[3]),
                    };
                    let rows = builder.rows / span;
                    assert_eq!(one(&prepared, &buffers, rows), built(builder));
                    // The state holds both halves.
                    let short = RingBackupBuffers {
                        state: View {
                            bytes: 2 * ring.working - 1,
                            ..buffers.state
                        },
                        ..buffers
                    };
                    assert!(
                        prepared
                            .launches(&short, View::EMPTY, counts(rows))
                            .is_err()
                    );
                    assert!(
                        prepared
                            .launches(&buffers, View::EMPTY, counts(slots + 1))
                            .is_err()
                    );
                }
                let restore = own
                    .ring_restore(
                        &RingRestore {
                            layout,
                            span: MAX_VERIFY_ROWS,
                        },
                        Need::rows(slots),
                    )
                    .unwrap();
                for items in [1, 3, slots] {
                    let builder = crate::row_plan::ring_restore(
                        tp,
                        ring,
                        items,
                        MAX_VERIFY_ROWS,
                        crate::row_plan::RingRestoreViews {
                            ring_state: state,
                            slot_ids: tables[0],
                            starts: tables[1],
                            counts: tables[2],
                            live_items: tables[3],
                        },
                    )
                    .unwrap();
                    let buffers = RingRestoreBuffers {
                        state: view(state),
                        slot_ids: view(tables[0]),
                        starts: view(tables[1]),
                        counts: view(tables[2]),
                        live_items: view(tables[3]),
                    };
                    assert_eq!(one(&restore, &buffers, items), built(builder));
                }
                // More items than slots, and another topology's ring.
                assert!(
                    own.ring_backup(&RingBackup { layout, span: 1 }, Need::rows(slots + 1))
                        .is_err()
                );
                let other = Tp::SUPPORTED
                    .into_iter()
                    .find(|&other| other != tp)
                    .unwrap();
                assert!(
                    own.ring_backup(
                        &RingBackup {
                            layout: RingLayout::of(other, slots),
                            span: 1
                        },
                        Need::rows(slots)
                    )
                    .is_err()
                );
            }
            // No entry serves a ring of 12 slots.
            assert!(
                own.ring_restore(
                    &RingRestore {
                        layout: RingLayout::of(tp, 12),
                        span: 8
                    },
                    Need::rows(12)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn prefix_copies_launch_what_the_builder_plans() {
        for tp in Tp::SUPPORTED {
            let own = Sm120Own::new(tp);
            for (slots, slot) in [(32, 5), (256, 200)] {
                let ring = RingGeometry::new(tp, slots);
                for (fp8, drafter) in [
                    (false, Drafter::None),
                    (false, Drafter::Mtp),
                    (true, Drafter::Dflash),
                ] {
                    // The MTP drafter serves at most its draft items.
                    if drafter == Drafter::Mtp && slots > crate::contract::DRAFT_ITEMS {
                        continue;
                    }
                    let state = request_state(tp, &KV_FORMATS[usize::from(fp8)], drafter);
                    let extent = Extent {
                        served: slots,
                        bucket: slots,
                        verify_rows: MAX_VERIFY_ROWS,
                    };
                    let table =
                        bundle_runtime::snapshot::Table::new(&state, extent, Holder::Slot(slot))
                            .unwrap();
                    let drafter_bytes = table
                        .descriptors
                        .iter()
                        .filter(|descriptor| descriptor.buffer == 1)
                        .map(|descriptor| descriptor.length)
                        .sum::<u64>();
                    let spans = crate::prefix::PrefixCopyViews {
                        ring_state: span(1, ring.working),
                        drafter: (drafter_bytes > 0).then(|| span(2, drafter_bytes)),
                        pool: span(3, 256 * table.page_bytes),
                        pages: span(4, 2048),
                    };
                    let buffers = RingPrefixCopyBuffers {
                        state: view(spans.ring_state),
                        drafter: spans.drafter.map(view),
                        pool: view(spans.pool),
                        pages: view(spans.pages),
                        slot,
                    };
                    for to_pages in [true, false] {
                        let op = RingPrefixCopy {
                            layout: RingLayout::of(tp, slots),
                            drafter_bytes,
                            page_bytes: table.page_bytes,
                            pages: table.pages(),
                            direction: if to_pages {
                                Direction::Capture
                            } else {
                                Direction::Restore
                            },
                        };
                        let prepared = own.ring_prefix_copy(&op, Need::rows(1)).unwrap();
                        let builder = crate::prefix::prefix_copy(
                            &table,
                            slot,
                            table.pages(),
                            to_pages,
                            crate::prefix::PrefixCopyViews {
                                drafter: spans.drafter,
                                ..spans
                            },
                            ring,
                        )
                        .unwrap();
                        assert_eq!(one(&prepared, &buffers, 1), built(builder));
                        let outside = RingPrefixCopyBuffers {
                            slot: slots,
                            ..buffers
                        };
                        assert!(prepared.launches(&outside, View::EMPTY, counts(1)).is_err());
                    }
                }
            }
        }
    }

    /// The history the MTP drafter retains whole target rows in, and the
    /// DFlash tap shard history, at `items` slots.
    fn histories(tp: Tp, items: u32) -> [(History, u32); 2] {
        [
            (
                History {
                    stride: tp.geometry.hidden(),
                    capacity: crate::retain::capacity(128) as u32,
                    slots: items,
                },
                tp.geometry.hidden(),
            ),
            (
                History {
                    stride: crate::dflash::program::fc_shard(tp) as u32,
                    capacity: crate::dflash::state::history_capacity(tp) as u32,
                    slots: items,
                },
                2048,
            ),
        ]
    }

    #[test]
    fn retained_and_gathered_rows_launch_what_the_builders_plan() {
        for tp in Tp::SUPPORTED {
            let own = Sm120Own::new(tp);
            let rank = Rank::new(tp);
            for items in [32, 256] {
                for (history, width) in histories(tp, items) {
                    let spans = (span(60, BIG), span(61, history.bytes(width)));
                    let buffer = crate::retain::HistoryBuffer {
                        span: spans.1,
                        stride: history.stride,
                        capacity: u64::from(history.capacity),
                        slots: history.slots,
                    };
                    let stride = tp.geometry.hidden();
                    for (phase, live) in phases() {
                        let stage = rank.stage(tp, 0, phase, live, items);
                        let gather = own
                            .gather_rows(
                                &GatherRows {
                                    width,
                                    output_stride: history.stride,
                                    history,
                                },
                                Need::rows(capacity(phase)),
                            )
                            .unwrap();
                        let builder = crate::retain::gather_rows(
                            &stage,
                            buffer,
                            spans.0,
                            history.stride,
                            width,
                        )
                        .unwrap();
                        let buffers = GatherRowsBuffers {
                            history: view(spans.1),
                            output: view(spans.0),
                            slot_ids: view(stage.inputs.slot_ids),
                            token_batch: view(stage.inputs.token_batch),
                            positions: view(stage.inputs.positions),
                            live_rows: view(stage.inputs.live_tokens),
                        };
                        let rows = builder.rows;
                        assert_eq!(one(&gather, &buffers, rows), built(builder));
                        let short = GatherRowsBuffers {
                            history: View {
                                bytes: history.bytes(width) - 2,
                                ..buffers.history
                            },
                            ..buffers
                        };
                        assert!(gather.launches(&short, View::EMPTY, counts(rows)).is_err());
                        if phase == Phase::Prefill {
                            continue;
                        }
                        let retain = own
                            .retain_rows(
                                &RetainRows {
                                    width,
                                    input_stride: stride,
                                    history,
                                },
                                Need::rows(capacity(phase)),
                            )
                            .unwrap();
                        let builder =
                            crate::retain::retain_rows(&stage, spans.0, stride, buffer, width)
                                .unwrap();
                        let buffers = RetainRowsBuffers {
                            input: view(spans.0),
                            history: view(spans.1),
                            slot_ids: view(stage.inputs.slot_ids),
                            token_batch: view(stage.inputs.token_batch),
                            q_start: view(stage.inputs.q_start),
                            prefix_len: view(stage.inputs.prefix_len),
                            live_rows: view(stage.inputs.live_tokens),
                        };
                        assert_eq!(one(&retain, &buffers, builder.rows), built(builder));
                    }
                }
            }
            // Widths off the 16-byte vectors.
            let (history, _) = histories(tp, 32)[0];
            assert!(
                own.retain_rows(
                    &RetainRows {
                        width: 12,
                        input_stride: 4096,
                        history
                    },
                    Need::rows(32)
                )
                .is_err()
            );
        }
    }

    /// Every phase with its live rows: prefill, and each decode and verify
    /// bucket at its capacity.
    fn every_phase() -> Vec<(Phase, u32)> {
        crate::scratch::COMM_BUCKETS
            .iter()
            .flat_map(|&rows| {
                [Phase::Decode(rows), Phase::Verify(rows)]
                    .into_iter()
                    .filter(move |phase| phase.capacity_in::<Mimo>() == Some(rows))
                    .map(move |phase| (phase, rows))
            })
            .chain([(Phase::Prefill, 1908)])
            .collect()
    }

    /// The instance's writes into the drafters' retained histories (MTP's
    /// hidden rows, every rank's DFlash tap columns) and DFlash's gathers
    /// launch what the retain builders plan at every phase and bucket, at
    /// TP2 and TP4, for 32 and 256 served items; views off their 16-byte
    /// alignment are refused, and so is a shape the instance did not
    /// prepare.
    #[test]
    fn drafter_histories_launch_what_the_retain_builders_plan() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            for items in [32, 256] {
                let rank = Rank::serving(tp, items);
                let (input, history, output) = (span(60, BIG), span(61, BIG), span(62, BIG));
                let shard = crate::dflash::program::fc_shard(tp) as u32;
                let dflash = crate::retain::HistoryBuffer {
                    span: history,
                    stride: shard,
                    capacity: crate::dflash::state::history_capacity(tp),
                    slots: items,
                };
                let hidden = tp.geometry.hidden();
                let mtp = crate::retain::HistoryBuffer {
                    span: history,
                    stride: hidden,
                    capacity: crate::retain::capacity(tp.geometry.window() as u64),
                    slots: crate::contract::DRAFT_ITEMS,
                };
                for (phase, live) in every_phase() {
                    let stage = rank.stage(tp, 0, phase, live, items);
                    let mut retains = vec![];
                    if items <= crate::contract::DRAFT_ITEMS {
                        retains.push((crate::mtp::program::retain(tp), hidden, mtp, hidden));
                    }
                    for index in 0..tp.ranks() {
                        for tap in 0..5 {
                            if let Some((_, _, width)) =
                                crate::dflash::program::tap_columns(index, tap, tp)
                            {
                                let op = crate::dflash::program::retain(tp, items, width as u32);
                                retains.push((op, op.input_stride, dflash, width as u32));
                            }
                        }
                    }
                    for (op, stride, buffer, width) in retains {
                        let lowered = stage.ops.retain_rows(
                            &op,
                            phase,
                            stage.bucket,
                            input,
                            history,
                            &stage.inputs,
                        );
                        if phase == Phase::Prefill {
                            assert!(lowered.is_err());
                            continue;
                        }
                        let builder =
                            crate::retain::retain_rows(&stage, input, stride, buffer, width)
                                .unwrap();
                        assert_eq!(
                            built(lowered.unwrap().launch),
                            built(builder),
                            "{tp:?} {items} {phase:?} {op:?}"
                        );
                        compared += 1;
                        let off = DeviceBufferSpanV1 { offset: 8, ..input };
                        assert!(
                            stage
                                .ops
                                .retain_rows(&op, phase, stage.bucket, off, history, &stage.inputs)
                                .is_err()
                        );
                    }
                    let op = crate::dflash::program::gather(tp, items);
                    let builder =
                        crate::retain::gather_rows(&stage, dflash, output, shard, shard).unwrap();
                    let lowered = stage
                        .ops
                        .gather_rows(
                            &op,
                            phase,
                            stage.bucket,
                            live,
                            history,
                            output,
                            &stage.inputs,
                        )
                        .unwrap();
                    assert_eq!(
                        built(lowered.launch),
                        built(builder),
                        "{tp:?} {items} {phase:?} gather"
                    );
                    compared += 1;
                    let other = crate::dflash::program::gather(tp, items / 2);
                    assert!(
                        stage
                            .ops
                            .gather_rows(
                                &other,
                                phase,
                                stage.bucket,
                                live,
                                history,
                                output,
                                &stage.inputs
                            )
                            .is_err()
                    );
                }
            }
        }
        assert!(compared > 100, "{compared}");
    }

    /// The instance's DFlash row copies (every rank's tap columns, the
    /// context's zeroed residual, a block's scaled V), mask rows, context
    /// cache writes and drafts launch what the DFlash program's builders
    /// plan at every phase and bucket, TP2 and TP4, for 32 and 256 served
    /// items; the block's ops serve verify buckets only, and a copy view
    /// shorter than the phase's capacity or off its alignment is refused.
    #[test]
    fn dflash_rows_launch_from_the_instance_ops() {
        use crate::dflash::program as dflash;
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            for items in [32, 256] {
                let rank = Rank::serving(tp, items);
                let kv = (tp.dflash_kv_heads() * 128) as u32;
                let shard = dflash::fc_shard(tp) as u32;
                let hidden = tp.geometry.dflash().hidden_size;
                let (input, output) = (span(60, BIG), span(61, BIG));
                for (phase, live) in every_phase() {
                    let stage = rank.stage(tp, 0, phase, live, items);
                    let verify = matches!(phase, Phase::Verify(_));
                    let copy = |op: &CopyRows, input, output| {
                        stage.ops.dflash_copy_rows(
                            op,
                            phase,
                            stage.bucket,
                            stage.switches,
                            live,
                            input,
                            output,
                            stage.inputs.live_tokens,
                        )
                    };
                    let mut copies = vec![
                        (dflash::zero_residual(tp), None, hidden, hidden, 1.0, true),
                        (
                            dflash::scale_v(tp),
                            Some((tp.dflash_q_heads() * 128) as u32 + 2 * kv),
                            kv,
                            kv,
                            tp.geometry.dflash().dflash_config.attention_value_scale as f32,
                            verify,
                        ),
                    ];
                    for index in 0..tp.ranks() {
                        for tap in 0..5 {
                            if let Some((_, _, width)) = dflash::tap_columns(index, tap, tp) {
                                let width = width as u32;
                                let stride = tp.geometry.dflash().target_hidden_size;
                                let op = dflash::copy_tap(tp, width);
                                copies.push((op, Some(stride), shard, width, 1.0, true));
                            }
                        }
                    }
                    for (op, stride, output_stride, width, scale, served) in copies {
                        let input = stride.map(|_| input);
                        let lowered = copy(&op, input, output);
                        if !served {
                            assert!(lowered.is_err());
                            continue;
                        }
                        let builder = dflash::copy_rows(
                            &stage,
                            input.zip(stride),
                            output,
                            output_stride,
                            width,
                            scale,
                        )
                        .unwrap();
                        assert_eq!(
                            built(lowered.unwrap().launch),
                            built(builder),
                            "{tp:?} {items} {phase:?} {op:?}"
                        );
                        compared += 1;
                        let capacity = phase.capacity_in::<Mimo>().unwrap();
                        let short = DeviceBufferSpanV1 {
                            length: row_bytes(capacity, output_stride, width) - 2,
                            ..output
                        };
                        assert!(copy(&op, input, short).is_err());
                        let off = DeviceBufferSpanV1 {
                            offset: 8,
                            ..output
                        };
                        assert!(copy(&op, input, off).is_err());
                    }
                    let cache = span(64, BIG);
                    let buffers = dflash::RankBuffers {
                        views: BTreeMap::new(),
                        cache,
                        taps: span(65, BIG),
                        pairs: span(66, BIG),
                        gathered: span(67, BIG),
                        gather_flags: span(68, BIG),
                        gather_epoch: span(69, 4),
                        vocab_width: span(70, 4),
                    };
                    let (k, v) = (span(71, BIG), span(72, BIG));
                    let write = dflash::cache_write(tp, items);
                    assert_eq!((write.k_stride, write.v_stride), (kv, 10 * kv));
                    for layer in [0, 4] {
                        let planes = dflash::cache_planes(cache, layer, tp, items).unwrap();
                        let write_rows = |inputs: &crate::upload::DeviceInputs, k, v| {
                            stage.ops.dflash_cache_write(
                                &write,
                                phase,
                                stage.bucket,
                                live,
                                k,
                                v,
                                planes.into(),
                                inputs,
                            )
                        };
                        let builder = dflash::cache_write_builder(
                            &stage,
                            &buffers,
                            layer,
                            (k, write.k_stride),
                            (v, write.v_stride),
                        )
                        .unwrap();
                        assert_eq!(
                            built(write_rows(&stage.inputs, k, v).unwrap().launch),
                            built(builder)
                        );
                        compared += 1;
                        // A prefill (a catch-up's packed inputs) holds only its
                        // live rows, and grids the capacity all the same.
                        let rows = if phase == Phase::Prefill {
                            live
                        } else {
                            phase.capacity_in::<Mimo>().unwrap()
                        };
                        let held = |view: DeviceBufferSpanV1, bytes: u64| DeviceBufferSpanV1 {
                            length: bytes,
                            ..view
                        };
                        let packed = crate::upload::DeviceInputs {
                            positions: held(stage.inputs.positions, 4 * u64::from(rows)),
                            token_batch: held(stage.inputs.token_batch, 4 * u64::from(rows)),
                            ..stage.inputs
                        };
                        let (k, v) = (
                            held(k, row_bytes(rows, write.k_stride, kv)),
                            held(v, row_bytes(rows, write.v_stride, kv)),
                        );
                        let lowered = write_rows(&packed, k, v).unwrap().launch;
                        assert_eq!(lowered.plan.grid[0], phase.capacity_in::<Mimo>().unwrap());
                        let short = crate::upload::DeviceInputs {
                            positions: held(stage.inputs.positions, 4 * u64::from(rows) - 4),
                            ..packed
                        };
                        assert!(write_rows(&short, k, v).is_err());
                    }
                    let mask = span(73, BIG);
                    let lowered = stage.ops.dflash_mask_rows(
                        &dflash::mask(tp),
                        phase,
                        stage.bucket,
                        stage.activation,
                        mask,
                        &stage.inputs,
                    );
                    let drafted = stage.ops.dflash_drafts(
                        &dflash::drafts(tp),
                        phase,
                        stage.bucket,
                        buffers.gathered,
                        &stage.inputs,
                    );
                    if !verify {
                        assert!(lowered.is_err() && drafted.is_err());
                        continue;
                    }
                    let builder = dflash::mask_rows(&stage, mask).unwrap();
                    assert_eq!(built(lowered.unwrap().launch), built(builder));
                    let rows = phase.capacity_in::<Mimo>().unwrap();
                    let builder = dflash::drafts_builder(&stage, buffers.gathered, rows);
                    assert_eq!(built(drafted.unwrap().launch), built(builder));
                    compared += 2;
                }
            }
        }
        assert!(compared > 300, "{compared}");
    }

    #[test]
    fn dflash_rows_launch_what_the_builders_plan() {
        for tp in Tp::SUPPORTED {
            let own = Sm120Own::new(tp);
            let rank = Rank::new(tp);
            let kv = (tp.dflash_kv_heads() * 128) as u32;
            for items in [32, 256] {
                for (phase, live) in phases() {
                    let stage = rank.stage(tp, 0, phase, live, items);
                    let need = Need::rows(capacity(phase));
                    // A tap's shard copy, a scaled V copy and a zeroing copy.
                    for (input, output_stride, width, scale) in [
                        (Some((span(60, BIG), tp.geometry.hidden())), 5120, 2048, 1.0),
                        (Some((span(61, BIG), 3 * kv)), kv, kv, 0.612),
                        (None, tp.geometry.hidden(), tp.geometry.hidden(), 1.0),
                    ] {
                        let op = CopyRows {
                            width,
                            input_stride: input.map(|(_, stride)| stride),
                            output_stride,
                            scale_bits: f32::to_bits(scale),
                        };
                        let prepared = own.dflash_copy_rows(&op, need).unwrap();
                        let output = span(62, BIG);
                        let builder = crate::dflash::program::copy_rows(
                            &stage,
                            input,
                            output,
                            output_stride,
                            width,
                            scale,
                        )
                        .unwrap();
                        let buffers = CopyRowsBuffers {
                            input: input.map(|(span, _)| view(span)),
                            output: view(output),
                            live_rows: view(stage.inputs.live_tokens),
                        };
                        assert_eq!(one(&prepared, &buffers, builder.rows), built(builder));
                        let mismatched = CopyRowsBuffers {
                            input: buffers.input.xor(Some(view(span(63, BIG)))),
                            ..buffers
                        };
                        assert!(
                            prepared
                                .launches(&mismatched, View::EMPTY, counts(1))
                                .is_err()
                        );
                    }
                    let cache = span(64, BIG);
                    let dflash = crate::dflash::program::RankBuffers {
                        views: BTreeMap::new(),
                        cache,
                        taps: span(65, BIG),
                        pairs: span(66, BIG),
                        gathered: span(67, BIG),
                        gather_flags: span(68, BIG),
                        gather_epoch: span(69, 4),
                        vocab_width: span(70, 4),
                    };
                    let (k, v) = ((span(71, BIG), 3 * kv), (span(72, BIG), kv));
                    let write = own
                        .dflash_cache_write(&CacheWrite::of(tp, items, k.1, v.1), need)
                        .unwrap();
                    for layer in [0, 4] {
                        let builder = crate::dflash::program::cache_write_builder(
                            &stage, &dflash, layer, k, v,
                        )
                        .unwrap();
                        let (k_cache, v_cache) =
                            crate::dflash::program::cache_planes(cache, layer, tp, items).unwrap();
                        let buffers = CacheWriteBuffers {
                            k: view(k.0),
                            v: view(v.0),
                            k_cache: view(k_cache),
                            v_cache: view(v_cache),
                            slot_ids: view(stage.inputs.slot_ids),
                            token_batch: view(stage.inputs.token_batch),
                            positions: view(stage.inputs.positions),
                            last_positions: view(stage.inputs.last_positions),
                            live_rows: view(stage.inputs.live_tokens),
                        };
                        assert_eq!(one(&write, &buffers, builder.rows), built(builder));
                    }
                    if !matches!(phase, Phase::Verify(_)) {
                        continue;
                    }
                    let mask = span(73, BIG);
                    let builder = crate::dflash::program::mask_rows(&stage, mask).unwrap();
                    let prepared = own
                        .dflash_mask_rows(
                            &MaskRows {
                                hidden: tp.geometry.dflash().hidden_size,
                            },
                            need,
                        )
                        .unwrap();
                    let buffers = MaskRowsBuffers {
                        activation: view(stage.activation),
                        mask: view(mask),
                        positions: view(stage.inputs.positions),
                        prefix_len: view(stage.inputs.prefix_len),
                        token_batch: view(stage.inputs.token_batch),
                        live_rows: view(stage.inputs.live_tokens),
                    };
                    assert_eq!(one(&prepared, &buffers, builder.rows), built(builder));
                    let rows = capacity(phase);
                    let builder =
                        crate::dflash::program::drafts_builder(&stage, dflash.gathered, rows);
                    let prepared = own
                        .dflash_drafts(&Drafts { ranks: tp.ranks() }, need)
                        .unwrap();
                    let buffers = DraftsBuffers {
                        gathered: view(dflash.gathered),
                        token_ids: view(stage.inputs.token_ids),
                        positions: view(stage.inputs.positions),
                        prefix_len: view(stage.inputs.prefix_len),
                        token_batch: view(stage.inputs.token_batch),
                        live_rows: view(stage.inputs.live_tokens),
                    };
                    assert_eq!(one(&prepared, &buffers, rows), built(builder));
                }
            }
            // Entries are compiled for the topology's drafter and ranks.
            let need = Need::rows(32);
            assert!(
                own.dflash_mask_rows(&MaskRows { hidden: 2048 }, need)
                    .is_err()
            );
            assert!(
                own.dflash_drafts(
                    &Drafts {
                        ranks: tp.ranks() + 2
                    },
                    need
                )
                .is_err()
            );
            let wide = CacheWrite {
                kv_heads: 8,
                ..CacheWrite::of(tp, 32, 3 * kv, kv)
            };
            assert!(own.dflash_cache_write(&wide, need).is_err());
            assert!(
                own.dflash_cache_write(&CacheWrite::of(tp, 257, 3 * kv, kv), need)
                    .is_err()
            );
        }
    }

    /// The instance slices each rank's EH shard of its normalized rows
    /// through the shared column slice, one CTA per row of the phase's
    /// capacity, and the MTP pair launches that slice last.
    #[test]
    fn mtp_eh_slices_launch_the_shared_slice_over_the_capacity() {
        for tp in Tp::SUPPORTED {
            let rank = Rank::new(tp);
            for (phase, live) in every_phase() {
                for index in 0..tp.ranks() {
                    let stage = rank.stage(tp, index, phase, live, 32);
                    let packed = span(80, BIG);
                    let instance = stage.ops.mtp_eh_slice(
                        index,
                        phase,
                        stage.bucket,
                        stage.normalized,
                        packed,
                        stage.inputs.live_tokens,
                    );
                    let slice = built(instance.unwrap().launch);
                    let rows = capacity(phase);
                    assert_eq!(
                        (slice.symbol.as_str(), slice.rows),
                        (kernels_sm120::row::copy::SLICE_SYMBOL, rows)
                    );
                    assert_eq!(
                        (slice.plan.grid, slice.plan.block, slice.plan.pdl),
                        ([rows, 1, 1], [256, 1, 1], true)
                    );
                    let buffers = [stage.normalized, packed, stage.inputs.live_tokens]
                        .map(|span| Argument::DeviceBuffer(view(span).handle));
                    assert_eq!(slice.arguments.0[..3], buffers);
                    let shard = tp.mtp_eh_slice(index as usize);
                    let (width, stride, offset) = (shard.width, shard.stride, shard.offset);
                    assert_eq!(
                        slice.arguments.0[3..],
                        [width, stride, 1, offset, width, 1, 0].map(Argument::U32)
                    );
                    if !crate::mtp::program::eh_shard(tp, index).0 {
                        // Embedding ranks lower the embedding first, which
                        // needs projections.
                        continue;
                    }
                    let mut launches =
                        crate::mtp::program::pair(&stage, span(81, BIG), span(82, BIG), packed)
                            .unwrap();
                    assert_eq!(built(launches.pop().unwrap().launch), slice);
                }
            }
        }
    }

    #[test]
    fn prepared_ops_refuse_exactness_they_do_not_claim() {
        let own = Sm120Own::new(Tp::TP4);
        let need = Need::rows(32).with_exact(Invariance::any_rows(32));
        let layout = RingLayout::of(Tp::TP4, 32);
        assert!(
            own.ring_backup(&RingBackup { layout, span: 1 }, need)
                .is_ok()
        );
        assert!(matches!(
            own.ring_restore(&RingRestore { layout, span: 8 }, need),
            Err(Refused::Inexact { .. })
        ));
    }

    #[test]
    fn ring_launches_refuse_grids_past_cuda_limits() {
        let own = Sm120Own::new(Tp::TP4);
        let layout = RingLayout::of(Tp::TP4, 8);
        // One item of span 2^31 is past the grid limit; two items of span
        // 2^31 overflow; two of 2^30 reach 2^31.
        for (capacity, span) in [(1, 1 << 31), (2, 1 << 31), (2, 1 << 30)] {
            assert!(
                own.ring_backup(&RingBackup { layout, span }, Need::rows(capacity))
                    .is_err()
            );
            assert!(
                own.ring_restore(&RingRestore { layout, span }, Need::rows(capacity))
                    .is_err()
            );
        }
        // The widest grid that fits.
        assert!(
            own.ring_backup(
                &RingBackup {
                    layout,
                    span: (1 << 30) - 1
                },
                Need::rows(2)
            )
            .is_ok()
        );
    }

    #[test]
    fn guarantees_are_the_launched_symbols_own_declaration() {
        for tp in Tp::SUPPORTED {
            let own = Sm120Own::new(tp);
            let copy = Need::rows(8).with_exact(Invariance::COPY);
            let backup = |slots| RingBackup {
                layout: RingLayout::of(tp, slots),
                span: 1,
            };
            // Every slot count's backup is declared; the restore, which no
            // exact step launches, is not.
            for slots in [32, 8] {
                assert!(own.ring_backup(&backup(slots), copy).is_ok());
                assert!(matches!(
                    own.ring_restore(
                        &RingRestore {
                            layout: RingLayout::of(tp, slots),
                            span: 8
                        },
                        copy
                    ),
                    Err(Refused::Inexact { .. })
                ));
            }
        }
    }

    /// The declarations name the ring backup the bundle embeds for every
    /// topology and every slot count an instance may hold, and nothing else.
    #[test]
    fn every_ring_backup_is_declared_a_copy() {
        let declared = crate::sm120::declared_kernels();
        for tp in crate::topology::all() {
            for items in 1..=crate::contract::DECODE_BATCH {
                let slots = bundle_runtime::geometry::Items::new(items)
                    .unwrap()
                    .capacity();
                let symbol = crate::sm120::ring_entries::ring_symbol(tp, "ring_backup", slots);
                assert!(
                    declared
                        .iter()
                        .any(|d| d.symbol == symbol && d.invariance == Invariance::COPY),
                    "{symbol}"
                );
            }
        }
        for declared in &declared {
            assert!(embedded(&declared.symbol), "{}", declared.symbol);
            assert!(
                declared.symbol.contains("_ring_backup"),
                "{}",
                declared.symbol
            );
        }
    }

    /// Every per-item table is checked against `Counts::items`: views of
    /// exactly the bytes a one-row, one-item launch reads pass, one byte
    /// less is refused, and so are zero items.
    #[test]
    fn per_item_tables_are_checked() {
        let tp = Tp::TP4;
        let own = Sm120Own::new(tp);
        let need = Need::rows(1);
        let exact = |bytes| View { handle: 7, bytes };
        fn check<B: Copy>(prepared: &Prepared<B>, buffers: B, shorten: &[fn(&mut B)]) {
            let one = Counts::rows(1).with_items(1, 0);
            assert!(prepared.launches(&buffers, View::EMPTY, one).is_ok());
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
            for shorten in shorten {
                let mut short = buffers;
                shorten(&mut short);
                assert!(prepared.launches(&short, View::EMPTY, one).is_err());
            }
        }
        let (history, width) = histories(tp, 32)[0];
        let row = u64::from(width) * 2;
        let retain = own
            .retain_rows(
                &RetainRows {
                    width,
                    input_stride: width,
                    history,
                },
                need,
            )
            .unwrap();
        let buffers = RetainRowsBuffers {
            input: exact(row),
            history: exact(history.bytes(width)),
            slot_ids: exact(4),
            token_batch: exact(4),
            q_start: exact(4),
            prefix_len: exact(4),
            live_rows: exact(4),
        };
        check(
            &retain,
            buffers,
            &[
                |b| b.slot_ids.bytes -= 1,
                |b| b.q_start.bytes -= 1,
                |b| b.prefix_len.bytes -= 1,
                |b| b.token_batch.bytes -= 1,
            ],
        );
        let gather = own
            .gather_rows(
                &GatherRows {
                    width,
                    output_stride: width,
                    history,
                },
                need,
            )
            .unwrap();
        let buffers = GatherRowsBuffers {
            history: exact(history.bytes(width)),
            output: exact(row),
            slot_ids: exact(4),
            token_batch: exact(4),
            positions: exact(4),
            live_rows: exact(4),
        };
        check(
            &gather,
            buffers,
            &[
                |b| b.slot_ids.bytes -= 1,
                |b| b.token_batch.bytes -= 1,
                |b| b.positions.bytes -= 1,
            ],
        );
        let hidden = tp.geometry.dflash().hidden_size;
        let mask = own.dflash_mask_rows(&MaskRows { hidden }, need).unwrap();
        let buffers = MaskRowsBuffers {
            activation: exact(u64::from(hidden) * 2),
            mask: exact(u64::from(hidden) * 2),
            positions: exact(4),
            prefix_len: exact(4),
            token_batch: exact(4),
            live_rows: exact(4),
        };
        check(
            &mask,
            buffers,
            &[|b| b.prefix_len.bytes -= 1, |b| b.positions.bytes -= 1],
        );
        let op = CacheWrite::of(tp, 32, 256, 256);
        let write = own.dflash_cache_write(&op, need).unwrap();
        let entry = u64::from(op.entry()) * 2;
        let buffers = CacheWriteBuffers {
            k: exact(entry),
            v: exact(entry),
            k_cache: exact(op.plane_bytes()),
            v_cache: exact(op.plane_bytes()),
            slot_ids: exact(4),
            token_batch: exact(4),
            positions: exact(4),
            last_positions: exact(8),
            live_rows: exact(4),
        };
        check(
            &write,
            buffers,
            &[
                |b| b.slot_ids.bytes -= 1,
                |b| b.last_positions.bytes -= 1,
                |b| b.token_batch.bytes -= 1,
            ],
        );
        let drafts = own
            .dflash_drafts(&Drafts { ranks: tp.ranks() }, need)
            .unwrap();
        let buffers = DraftsBuffers {
            gathered: exact(u64::from(tp.ranks()) * 8),
            token_ids: exact(4),
            positions: exact(4),
            prefix_len: exact(4),
            token_batch: exact(4),
            live_rows: exact(4),
        };
        check(
            &drafts,
            buffers,
            &[|b| b.prefix_len.bytes -= 1, |b| b.gathered.bytes -= 1],
        );
    }
}

#[cfg(test)]
mod encoder_tests {
    use encoder_spec::components::StageOp;
    use kernel_core::backend::Counts;
    use kernel_core::contract::ArgumentList;
    use model_mimo26_contract::encoders::spec::{self, Rvq20, Sum20};
    use model_mimo26_contract::geometry::GEOMETRIES;

    use super::*;

    fn view(handle: u64) -> View {
        View {
            handle,
            bytes: u64::MAX,
        }
    }

    /// Every registered stage of every geometry's declarations launches
    /// what the encoder runtime launched for its `registered` slot: the
    /// `sm120::encoders::rows` kernel, one block of `THREADS` per row of the
    /// declaration's capacity, PDL, its buffers in the kernel's parameter
    /// order, at any live row count.
    #[test]
    fn encoder_kernels_launch_the_rows_kernels() {
        let own = Sm120Own::new(Tp::new(4).unwrap());
        let mut seen = 0;
        for geometry in GEOMETRIES {
            for declaration in spec::declarations_for(geometry).unwrap() {
                let inputs = declaration.inputs().unwrap();
                for (stage, item) in declaration.stages.iter().enumerate() {
                    let StageOp::Registered(registered) = &item.op else {
                        continue;
                    };
                    let kernel = encoder_rows::stage_kernel(declaration, stage)
                        .unwrap()
                        .unwrap();
                    let capacity = declaration.input.capacity_rows;
                    let need = Need::rows(capacity);
                    let value = registered.component.to_value();
                    // Views numbered by the kernel's parameter order.
                    let handle = |name: &str| {
                        kernel
                            .params
                            .iter()
                            .position(|&(param, _)| param == name)
                            .unwrap() as u64
                            + 1
                    };
                    let launches = |rows: u32| match registered.component.id().as_str() {
                        "mimo26:quantize.rvq20-f32.v1" => {
                            let rvq: Rvq20 = serde_json::from_value(value.clone()).unwrap();
                            let op = Rvq::of(rvq.params(), inputs[stage].width().unwrap());
                            let buffers = RvqBuffers {
                                input: view(handle("input")),
                                codebooks: view(handle("codebooks")),
                                codebook_offsets: view(handle("codebook_offsets")),
                                codes: view(handle("codes")),
                                live_rows: view(handle("live_rows")),
                            };
                            own.encoder_rvq(&op, need).unwrap().launches(
                                &buffers,
                                View::EMPTY,
                                Counts::rows(rows),
                            )
                        }
                        _ => {
                            let sum: Sum20 = serde_json::from_value(value.clone()).unwrap();
                            let buffers = EmbeddingSumBuffers {
                                tables: view(handle("tables")),
                                codes: view(handle("codes")),
                                output: view(handle("output")),
                                live_rows: view(handle("live_rows")),
                            };
                            own.encoder_embedding_sum(&EmbeddingSum::of(sum.params()), need)
                                .unwrap()
                                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        }
                    };
                    for rows in [1, capacity / 3, capacity] {
                        let expected = Launch {
                            symbol: kernel.symbol.clone(),
                            rows,
                            plan: plan([kernel.capacity, 1, 1], encoder_rows::THREADS),
                            arguments: ArgumentList(
                                (1..=kernel.params.len() as u64)
                                    .map(Argument::DeviceBuffer)
                                    .collect(),
                            ),
                            collective: false,
                            persistent: false,
                        };
                        assert_eq!(launches(rows).unwrap(), [expected], "{}", kernel.symbol);
                    }
                    assert!(launches(capacity + 1).is_err());
                    assert!(launches(0).is_err());
                    seen += 1;
                }
            }
        }
        assert!(seen >= 2);
    }

    #[test]
    fn encoder_kernels_refuse_small_views_and_unbuilt_entries() {
        let own = Sm120Own::new(Tp::new(4).unwrap());
        let op = Rvq {
            dim: 1024,
            stages: 20,
            max_codes: 1024,
            codebook_rows: 4608,
        };
        let need = Need::rows(8192);
        let rvq = own.encoder_rvq(&op, need).unwrap();
        let mut buffers = RvqBuffers {
            input: view(1),
            codebooks: view(2),
            codebook_offsets: view(3),
            codes: view(4),
            live_rows: view(5),
        };
        assert!(rvq.launches(&buffers, View::EMPTY, Counts::rows(8)).is_ok());
        buffers.codebooks.bytes = 4608 * 1024 * 4 - 1;
        assert!(
            rvq.launches(&buffers, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        let unbuilt = Rvq { dim: 1000, ..op };
        assert!(own.encoder_rvq(&unbuilt, need).is_err());
        assert!(own.encoder_rvq(&op, Need::rows(0)).is_err());
        let sum = EmbeddingSum {
            tables: 20,
            vocab: 7,
            width: 1024,
        };
        assert!(own.encoder_embedding_sum(&sum, need).is_err());
    }
}
