//! Qwen's own kernels on SM120: [`OwnOps`] over the bundle's cubins.
//!
//! Each prepared op launches the entry the bundle builds for its shape:
//! the bundle and speculation entries at 256 threads with PDL, the row
//! catalog's PLE convolution and n-gram items, the prefix snapshot copy,
//! the fused up mix, and the dense groups (`dense.rs`). Ops are refused
//! where an entry's compiled constants (widths, streams, the PLE entry,
//! verify rows, heads, ranks, snapshot pieces) or its 16-byte vectors do
//! not fit the contract.
//!
//! Views carry no offset, so a host binds each view at the alignment its
//! role record states. Guarantees are the declaration of the symbol an op
//! launches (the bundle's own, `model_qwen38_contract::owned`, or the
//! catalog's), none for a symbol without one.

use std::sync::OnceLock;

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, Refused, View};
use kernel_core::contract::{Argument, LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;
use kernels_sm120::backend::own::{self, OwnKernels, buffer, bytes, launch, rows, rule};
use kernels_sm120::row::hyper_connection::Entry as RowEntry;
use model_qwen38_contract::own_ops::copies::{
    CopyEntries, CopyEntriesBuffers, PrefixCopy, PrefixCopyBuffers, Snapshot,
};
use model_qwen38_contract::own_ops::hc::{HcUpMix, HcUpMixBuffers, UpMixInput};
use model_qwen38_contract::own_ops::mtp::{
    DraftArgmaxFinal, DraftArgmaxFinalBuffers, DraftArgmaxPartial, DraftArgmaxPartialBuffers,
    MtpCombine, MtpCombineBuffers, MtpGather, MtpGatherBuffers, MtpNorms, MtpNormsBuffers,
};
use model_qwen38_contract::own_ops::ngram::{Ngram, NgramBuffers, NgramWrites};
use model_qwen38_contract::own_ops::ple::{
    PleAnchor, PleAnchorBuffers, PleCheckpoints, PleCheckpointsBuffers, PleConv, PleConvBuffers,
    PleLayout, PleResidual, PleResidualBuffers, PleStage, PleStageBuffers,
};
use model_qwen38_contract::own_ops::{ItemForm, OwnOps};

use super::entries;
use crate::contract::{MAX_VERIFY_ROWS, Shape};
use crate::launch;
use model_qwen38_contract::sm120::owned;

#[path = "dense.rs"]
mod dense;

/// Threads of an own-kernel block.
const THREADS: u32 = 256;
/// Bytes one state copy block moves (`kChunk`).
const COPY_CHUNK: u64 = 16 * 1024;
/// Channels of one PLE convolution block.
const CONV_CHANNELS: u32 = 256;

/// Qwen's own kernels as the bundle compiles them for one shape. The padded
/// up mix layout is chosen at creation, as the instance's switch says.
#[derive(Clone, Copy, Debug)]
pub struct Sm120Own {
    shape: Shape,
    padded_mix: bool,
}

impl Sm120Own {
    pub fn new(shape: Shape, padded_mix: bool) -> Sm120Own {
        Sm120Own { shape, padded_mix }
    }

    /// The shape the kernels are compiled for.
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// `symbol` with the shape's kernel tag.
    fn tagged(&self, symbol: &str) -> String {
        launch::tagged(symbol, self.shape)
    }

    /// The rank-count variant of `symbol` with the shape's kernel tag.
    fn per_rank(&self, symbol: &str) -> String {
        launch::tagged(&launch::per_rank(symbol, self.shape), self.shape)
    }
}

/// A launch of a bundle-owned entry: 256 threads, PDL, no shared memory or
/// cluster.
fn plan(grid: [u32; 3]) -> LaunchPlan {
    own::pdl_plan(grid, THREADS)
}

/// Qwen's own kernels: their implementation names, the bundle's cubins and
/// the declarations of what they keep.
const OWN: OwnKernels = OwnKernels {
    prefix: "sm120.qwen",
    embedded,
    declared,
    check: own::plan,
};

/// Whether the bundle's cubins hold `symbol`.
fn embedded(symbol: &str) -> bool {
    crate::CUBINS
        .iter()
        .any(|cubin| cubin.symbols.contains(&symbol))
}

/// The row catalog's entry `symbol` of a Qwen profile.
fn row_entry(symbol: &str) -> Result<RowEntry, Refused> {
    static ENTRIES: OnceLock<Vec<RowEntry>> = OnceLock::new();
    ENTRIES
        .get_or_init(model_qwen38_contract::sm120::row::hyper_connection_rows)
        .iter()
        .find(|entry| {
            entry.symbol == symbol
                && model_qwen38_contract::sm120::geometry::owner(entry.profile).is_some()
        })
        .cloned()
        .ok_or(rule("the row catalog has no entry for this own op"))
}

/// The invariance the bundle or the catalog declares for `symbol`, none
/// when it has no declaration: a declaration covers its own symbol only.
fn declared(symbol: &str) -> Invariance {
    static DECLARED: OnceLock<Vec<kernels_sm120::invariance::Declared>> = OnceLock::new();
    let declared = DECLARED.get_or_init(|| {
        let mut all = model_qwen38_contract::owned::declared();
        all.extend(kernels_sm120::invariance::declarations(
            model_qwen38_contract::sm120::catalogs(),
        ));
        all
    });
    own::declared_in(declared, symbol)
}

/// The items of a launch over items: from one to its rows (or, for an op
/// over items only, its capacity).
fn items(counts: Counts, most: u32) -> Result<u32, Refusal> {
    if counts.items == 0 || counts.items > most {
        return Err(Refusal(
            "an own op over items needs from one item to its rows",
        ));
    }
    Ok(counts.items)
}

/// Prepares the op launching `symbol`, with the guarantees its declaration
/// claims, after checking the need.
fn prepare<B: 'static>(
    name: &str,
    symbol: &str,
    need: Need,
    planner: impl Fn(&B, Counts) -> Result<Launch, Refusal> + Send + Sync + 'static,
) -> Result<Prepared<B>, Refused> {
    prepare_info(name, symbol, need, |_| {}, planner)
}

/// [`prepare`], with `extend` adding what the op reports beyond its
/// symbol and guarantees.
fn prepare_info<B: 'static>(
    name: &str,
    symbol: &str,
    need: Need,
    extend: impl FnOnce(&mut OpInfo),
    planner: impl Fn(&B, Counts) -> Result<Launch, Refusal> + Send + Sync + 'static,
) -> Result<Prepared<B>, Refused> {
    let mut info = OWN.info(name, vec![(symbol.to_owned(), 0)], need)?;
    extend(&mut info);
    OWN.prepare_one(info, need, planner)
}

/// A launch of row-catalog entry `entry` over `rows` rows, every one live.
fn row_launch(entry: &RowEntry, rows: u32, arguments: Vec<Argument>) -> Result<Launch, Refusal> {
    let plan = entry.plan(rows, rows)?;
    let arguments = entry.pack_arguments(arguments)?;
    Ok(Launch {
        symbol: entry.symbol.clone(),
        rows,
        plan,
        arguments,
        collective: false,
        persistent: false,
    })
}

/// Entry chunks of one PLE state entry copy.
fn chunks(layout: PleLayout) -> u32 {
    layout.entry_bytes().div_ceil(COPY_CHUNK) as u32
}

impl Sm120Own {
    /// Refuses a PLE layout other than the compiled one.
    fn layout(&self, layout: PleLayout) -> Result<(), Refused> {
        layout.validate()?;
        if layout != PleLayout::of(self.shape) {
            return Err(rule("the PLE entries are compiled for the shape's layout"));
        }
        Ok(())
    }

    /// Refuses verify items of other than the compiled verify rows.
    fn verify(rows: u32) -> Result<(), Refused> {
        if rows != MAX_VERIFY_ROWS {
            return Err(rule("the verify entries are compiled for the verify rows"));
        }
        Ok(())
    }
}

impl OwnOps for Sm120Own {
    fn name(&self) -> &str {
        "sm120"
    }

    fn ple_residual(
        &self,
        op: &PleResidual,
        need: Need,
    ) -> Result<Prepared<PleResidualBuffers<View>>, Refused> {
        op.validate()?;
        if *op != PleResidual::of(self.shape) {
            return Err(rule("the PLE residual is compiled for the shape's rows"));
        }
        let symbol = self.tagged(owned::PLE_RESIDUAL);
        let (planned, capacity) = (symbol.clone(), need.capacity);
        let row = u64::from(op.hidden) * u64::from(op.streams) * 2;
        let streams = op.streams;
        prepare(
            "ple_residual",
            &symbol,
            need,
            move |b: &PleResidualBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                for view in [b.x, b.gated, b.conv, b.out] {
                    view.covers(bytes(rows, row))?;
                }
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, streams, 1]),
                    vec![
                        buffer(b.x),
                        buffer(b.gated),
                        buffer(b.conv),
                        buffer(b.out),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn ple_stage(
        &self,
        op: &PleStage,
        need: Need,
    ) -> Result<Prepared<PleStageBuffers<View>>, Refused> {
        op.validate()?;
        self.layout(op.layout)?;
        let symbol = self.tagged(owned::PLE_STAGE);
        let (planned, capacity, layout) = (symbol.clone(), need.capacity, op.layout);
        prepare(
            "ple_stage",
            &symbol,
            need,
            move |b: &PleStageBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                let items = items(counts, rows)?;
                b.pool.covers(layout.entry_bytes())?;
                b.staging.covers(bytes(items, layout.entry_bytes()))?;
                for table in [b.slot_ids, b.prefix_len] {
                    table.covers(bytes(items, 4))?;
                }
                b.live_items.covers(4)?;
                // A per-item state copy, staged for the step's rows.
                Ok(launch(
                    &planned,
                    rows,
                    plan([items, chunks(layout), 1]),
                    vec![
                        buffer(b.pool),
                        buffer(b.staging),
                        buffer(b.slot_ids),
                        buffer(b.prefix_len),
                        buffer(b.live_items),
                    ],
                ))
            },
        )
    }

    fn ple_checkpoints(
        &self,
        op: &PleCheckpoints,
        need: Need,
    ) -> Result<Prepared<PleCheckpointsBuffers<View>>, Refused> {
        op.validate()?;
        self.layout(op.layout)?;
        Self::verify(op.verify_rows)?;
        // The launched entry names each checkpoint's pool entry.
        let symbol = self.tagged(owned::PLE_CHECKPOINTS_INDEXED);
        let (planned, capacity, layout, verify) =
            (symbol.clone(), need.capacity, op.layout, op.verify_rows);
        prepare(
            "ple_checkpoints",
            &symbol,
            need,
            move |b: &PleCheckpointsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                let items = items(counts, rows)?;
                b.input.covers(bytes(rows, layout.row_bytes()))?;
                b.tokens.covers(bytes(rows, 4))?;
                b.staged.covers(bytes(items, layout.entry_bytes()))?;
                for table in [b.cu_rows, b.q_len] {
                    table.covers(bytes(items, 4))?;
                }
                b.checkpoint_ids.covers(bytes(items * verify, 4))?;
                b.live_items.covers(4)?;
                Ok(launch(
                    &planned,
                    verify,
                    plan([verify, chunks(layout), items]),
                    vec![
                        buffer(b.input),
                        buffer(b.tokens),
                        buffer(b.staged),
                        buffer(b.pool),
                        buffer(b.cu_rows),
                        buffer(b.q_len),
                        buffer(b.live_items),
                        buffer(b.checkpoint_ids),
                        Argument::U64(b.pool.bytes / layout.entry_bytes()),
                    ],
                ))
            },
        )
    }

    fn ple_anchor(
        &self,
        op: &PleAnchor,
        need: Need,
    ) -> Result<Prepared<PleAnchorBuffers<View>>, Refused> {
        op.validate()?;
        self.layout(op.layout)?;
        let symbol = self.tagged(owned::PLE_ANCHOR);
        let (planned, capacity, layout) = (symbol.clone(), need.capacity, op.layout);
        prepare(
            "ple_anchor",
            &symbol,
            need,
            move |b: &PleAnchorBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                let items = items(counts, rows)?;
                b.input.covers(bytes(rows, layout.row_bytes()))?;
                b.tokens.covers(bytes(rows, 4))?;
                b.staged.covers(bytes(items, layout.entry_bytes()))?;
                b.anchors.covers(layout.entry_bytes())?;
                for table in [b.cu_rows, b.prefix_len, b.anchor_slots, b.anchor_ends] {
                    table.covers(bytes(items, 4))?;
                }
                b.live_items.covers(4)?;
                // A per-item state copy at the step's anchors, for its rows.
                Ok(launch(
                    &planned,
                    rows,
                    plan([chunks(layout), items, 1]),
                    vec![
                        buffer(b.input),
                        buffer(b.tokens),
                        buffer(b.staged),
                        buffer(b.anchors),
                        buffer(b.cu_rows),
                        buffer(b.prefix_len),
                        buffer(b.anchor_slots),
                        buffer(b.anchor_ends),
                        buffer(b.live_items),
                    ],
                ))
            },
        )
    }

    fn ple_conv(
        &self,
        op: &PleConv,
        need: Need,
    ) -> Result<Prepared<PleConvBuffers<View>>, Refused> {
        op.validate()?;
        if *op != PleConv::of(self.shape, op.items) {
            return Err(rule(
                "the PLE convolution is compiled for the shape's taps and layout",
            ));
        }
        let layout = op.layout;
        let (capacity, row, entry) = (need.capacity, layout.row_bytes(), layout.entry_bytes());
        let (history, taps) = (layout.conv_bytes(), op.taps);
        // Every form checks the views it reads at the launch's rows and
        // items; the one-item form's views start at its item's.
        let check = move |b: &PleConvBuffers<View>, rows: u32, items: u32, one: bool| {
            for view in [b.input, b.output] {
                view.covers(bytes(rows, row))?;
            }
            b.weight
                .covers(u64::from(layout.wide) * u64::from(taps) * 2)?;
            // The history rows of each item's entry, `entry` bytes apart.
            let states = bytes(if one { 0 } else { items - 1 }, entry) + history;
            for view in [b.state, b.state_out] {
                view.covers(states)?;
            }
            b.q_len.covers(bytes(items, 4))
        };
        match op.items {
            ItemForm::Verify { rows: verify } => {
                Self::verify(verify)?;
                let symbol = self.tagged(owned::PLE_CONV_VERIFY);
                let planned = symbol.clone();
                let blocks = layout.wide.div_ceil(CONV_CHANNELS);
                prepare(
                    "ple_conv_verify",
                    &symbol,
                    need,
                    move |b: &PleConvBuffers<View>, counts| {
                        let rows = rows(counts, capacity)?;
                        let items = items(counts, rows)?;
                        check(b, rows, items, false)?;
                        let (Some(cu_rows), Some(live_items)) = (b.cu_rows, b.live_items) else {
                            return Err(Refusal("a verify convolution needs its item tables"));
                        };
                        cu_rows.covers(bytes(items, 4))?;
                        live_items.covers(4)?;
                        Ok(launch(
                            &planned,
                            verify,
                            plan([verify, blocks, items]),
                            vec![
                                buffer(b.input),
                                buffer(b.weight),
                                buffer(b.state),
                                buffer(b.output),
                                buffer(b.state_out),
                                buffer(cu_rows),
                                buffer(b.q_len),
                                buffer(live_items),
                                Argument::U64(entry / 2),
                            ],
                        ))
                    },
                )
            }
            ItemForm::Decode | ItemForm::One => {
                let decode = op.items == ItemForm::Decode;
                let entry_row = row_entry(&self.tagged(if decode {
                    entries::PLE_CONV_ITEMS
                } else {
                    entries::PLE_CONV_PREFILL
                }))?;
                entry_row.plan(capacity, capacity)?;
                let symbol = entry_row.symbol.clone();
                prepare(
                    if decode {
                        "ple_conv_items"
                    } else {
                        "ple_conv_prefill"
                    },
                    &symbol,
                    need,
                    move |b: &PleConvBuffers<View>, counts| {
                        let rows = rows(counts, capacity)?;
                        if b.cu_rows.is_some() || b.live_items.is_some() {
                            return Err(Refusal("only verify convolutions take item tables"));
                        }
                        let mut arguments = vec![
                            buffer(b.input),
                            buffer(b.weight),
                            buffer(b.state),
                            buffer(b.output),
                            buffer(b.state_out),
                            buffer(b.q_len),
                        ];
                        if decode {
                            // One row per bucket item.
                            check(b, rows, rows, false)?;
                            arguments.push(Argument::U64(entry / 2));
                        } else {
                            check(b, rows, 1, true)?;
                        }
                        row_launch(&entry_row, rows, arguments)
                    },
                )
            }
        }
    }

    fn copy_entries(
        &self,
        op: &CopyEntries,
        need: Need,
    ) -> Result<Prepared<CopyEntriesBuffers<View>>, Refused> {
        op.validate()?;
        if !op.entry_bytes.is_multiple_of(16) || !op.plane_stride.is_multiple_of(16) {
            return Err(rule("entry copies move 16-byte vectors"));
        }
        let symbol = self.tagged(owned::COPY_ENTRIES);
        let (planned, capacity, op) = (symbol.clone(), need.capacity, *op);
        let blocks = u32::try_from(op.entry_bytes.div_ceil(COPY_CHUNK))
            .map_err(|_| rule("an entry copy needs too many blocks"))?;
        let span = u64::from(op.planes - 1) * op.plane_stride + op.entry_bytes;
        prepare(
            "copy_entries",
            &symbol,
            need,
            move |b: &CopyEntriesBuffers<View>, counts| {
                let items = items(counts, capacity)?;
                for view in [b.src, b.dst] {
                    view.covers(span)?;
                }
                for table in [b.src_ids, b.dst_ids] {
                    table.covers(bytes(items, 4))?;
                }
                b.live_items.covers(4)?;
                // A per-item state copy: no token rows of its own.
                Ok(launch(
                    &planned,
                    1,
                    plan([items, op.planes, blocks]),
                    vec![
                        buffer(b.src),
                        buffer(b.dst),
                        buffer(b.src_ids),
                        buffer(b.dst_ids),
                        buffer(b.live_items),
                        Argument::U64(op.entry_bytes),
                        Argument::U64(op.plane_stride),
                    ],
                ))
            },
        )
    }

    fn prefix_copy(
        &self,
        op: &PrefixCopy,
        need: Need,
    ) -> Result<Prepared<PrefixCopyBuffers<View>>, Refused> {
        op.validate()?;
        let snapshot = op.snapshot;
        let drafted = snapshot.retained_bytes > 0;
        let compiled = Snapshot::of(
            self.shape,
            snapshot.pool_entries,
            snapshot.ple_entries,
            snapshot.ring_layers,
            snapshot.ring_slots,
            drafted,
        );
        if snapshot != compiled {
            return Err(rule("the prefix copy is compiled for the shape's pieces"));
        }
        if !op.page_bytes.is_multiple_of(16)
            || snapshot
                .pieces(model_qwen38_contract::own_ops::copies::Held {
                    slot: 0,
                    index: 0,
                    ple_entry: 0,
                })
                .iter()
                .any(|piece| !piece.length.is_multiple_of(16))
        {
            return Err(rule("the prefix copy moves 16-byte vectors"));
        }
        // One thread per 16-byte vector of the pieces.
        let blocks = u32::try_from((snapshot.bytes() / 16).div_ceil(u64::from(THREADS)))
            .map_err(|_| rule("a prefix snapshot needs too many blocks"))?;
        plan([blocks, 1, 1]).validate()?;
        let symbol = self.per_rank(owned::PREFIX_COPY_INDEXED);
        let (planned, op) = (symbol.clone(), *op);
        prepare(
            "prefix_copy",
            &symbol,
            need,
            move |b: &PrefixCopyBuffers<View>, _| {
                use model_qwen38_contract::own_ops::copies::{Buffer, Direction};
                let slots = b.slot + 1;
                b.conv_pool
                    .covers(snapshot.buffer_bytes(Buffer::Conv, slots))?;
                b.delta_pool
                    .covers(snapshot.buffer_bytes(Buffer::Delta, slots))?;
                b.ple_pool
                    .covers(snapshot.buffer_bytes(Buffer::Ple, slots))?;
                b.ring.covers(snapshot.buffer_bytes(Buffer::Ring, slots))?;
                b.pool.covers(op.page_bytes)?;
                b.pages.covers(op.pages * 8)?;
                let retained = match (b.retained, drafted) {
                    (None, false) => Argument::NullDevicePointer,
                    (Some(view), true) => {
                        view.covers(snapshot.buffer_bytes(Buffer::Retained, slots))?;
                        buffer(view)
                    }
                    _ => {
                        return Err(Refusal("a prefix copy's retained rows do not match its op"));
                    }
                };
                // A snapshot copy between slots and pages: no token rows of
                // its own.
                Ok(launch(
                    &planned,
                    1,
                    plan([blocks, 1, 1]),
                    vec![
                        buffer(b.conv_pool),
                        buffer(b.delta_pool),
                        buffer(b.ple_pool),
                        buffer(b.ring),
                        retained,
                        buffer(b.pool),
                        buffer(b.pages),
                        Argument::U32(snapshot.ring_layers),
                        Argument::U64(op.page_bytes),
                        Argument::U32(b.slot),
                        Argument::U32(b.index),
                        Argument::U32(u32::from(op.direction == Direction::Capture)),
                        Argument::U32(snapshot.pool_entries),
                        Argument::U32(snapshot.ring_slots),
                        Argument::U32(b.ple_entry),
                        Argument::U32(snapshot.ple_entries),
                    ],
                ))
            },
        )
    }

    fn ngram(&self, op: &Ngram, need: Need) -> Result<Prepared<NgramBuffers<View>>, Refused> {
        op.validate()?;
        let shape = self.shape;
        if *op != Ngram::of(shape, op.rank, op.writes, op.items) || op.rank >= shape.ranks() {
            return Err(rule(
                "the n-gram entries are compiled for the shape's heads",
            ));
        }
        let gather = op.writes == NgramWrites::Gather;
        let local = op.local_heads;
        let heads = u64::from(op.heads());
        let (capacity, op) = (need.capacity, *op);
        let width = u64::from(op.head_width);
        let history = op.entry_words * 4;
        // The views every form reads at `rows` rows of `items` items.
        let check = move |b: &NgramBuffers<View>,
                          rows: u32,
                          items: u32|
              -> Result<Vec<Argument>, Refusal> {
            b.tokens.covers(bytes(rows, 4))?;
            // Two words of each item's history, `entry_words` apart; the
            // one-item form's views start at its item's.
            let histories = if op.items == ItemForm::One {
                8
            } else {
                bytes(items - 1, history) + 8
            };
            for view in [b.history, b.history_out] {
                view.covers(histories)?;
            }
            b.multipliers.covers(u64::from(op.orders) * 8)?;
            for view in [b.sizes, b.offsets] {
                view.covers(heads * 8)?;
            }
            b.row_ids.covers(bytes(rows, u64::from(local) * 8))?;
            b.q_len.covers(bytes(items, 4))?;
            let mut arguments = vec![
                buffer(b.tokens),
                buffer(b.history),
                buffer(b.multipliers),
                buffer(b.sizes),
                buffer(b.offsets),
            ];
            match (gather, b.table, b.scale, b.output) {
                (true, Some(table), Some(scale), Some(output)) => {
                    table.covers(b.table_rows * width)?;
                    scale.covers(2)?;
                    output.covers(bytes(rows, u64::from(local) * width * 2))?;
                    arguments.extend([buffer(table), buffer(scale), buffer(output)]);
                }
                (false, None, None, None) => {}
                _ => {
                    return Err(Refusal(
                        "only n-gram gathers take a table, a scale and output rows",
                    ));
                }
            }
            arguments.extend([buffer(b.row_ids), buffer(b.history_out)]);
            Ok(arguments)
        };
        let tail = move |b: &NgramBuffers<View>| {
            [
                Argument::U32(op.rank),
                Argument::U64(b.table_start),
                Argument::U64(b.table_rows),
            ]
        };
        let one_item = |b: &NgramBuffers<View>| -> Result<(), Refusal> {
            if b.cu_rows.is_some() || b.live_items.is_some() {
                return Err(Refusal("only verify n-grams take item tables"));
            }
            Ok(())
        };
        match (op.items, op.writes) {
            (ItemForm::One, NgramWrites::Gather) => Err(rule(
                "prefill n-gram gathers are the row family's, not an own op",
            )),
            (ItemForm::Verify { rows: verify }, writes) => {
                Self::verify(verify)?;
                let symbol = match writes {
                    NgramWrites::Gather => self.per_rank(owned::NGRAM_VERIFY),
                    NgramWrites::Hash => self.hash(kernel_core::precision::ExecutionPhase::Verify),
                };
                let planned = symbol.clone();
                prepare(
                    "ngram_verify",
                    &symbol,
                    need,
                    move |b: &NgramBuffers<View>, counts| {
                        let rows = rows(counts, capacity)?;
                        let items = items(counts, rows)?;
                        let mut arguments = check(b, rows, items)?;
                        let (Some(cu_rows), Some(live_items)) = (b.cu_rows, b.live_items) else {
                            return Err(Refusal("a verify n-gram needs its item tables"));
                        };
                        cu_rows.covers(bytes(items, 4))?;
                        live_items.covers(4)?;
                        arguments.extend([buffer(cu_rows), buffer(b.q_len), buffer(live_items)]);
                        arguments.extend(tail(b));
                        arguments.push(Argument::U64(op.entry_words));
                        Ok(launch(
                            &planned,
                            verify,
                            plan([verify, local, items]),
                            arguments,
                        ))
                    },
                )
            }
            (ItemForm::Decode, NgramWrites::Gather) => {
                let entry_row = row_entry(&self.per_rank(entries::NGRAM_DECODE))?;
                entry_row.plan(capacity, capacity)?;
                let symbol = entry_row.symbol.clone();
                prepare(
                    "ngram_decode",
                    &symbol,
                    need,
                    move |b: &NgramBuffers<View>, counts| {
                        let rows = rows(counts, capacity)?;
                        one_item(b)?;
                        // One row per bucket item.
                        let mut arguments = check(b, rows, rows)?;
                        arguments.push(buffer(b.q_len));
                        arguments.extend(tail(b));
                        arguments.push(Argument::U64(op.entry_words));
                        row_launch(&entry_row, rows, arguments)
                    },
                )
            }
            (form, NgramWrites::Hash) => {
                let decode = form == ItemForm::Decode;
                let symbol = self.hash(if decode {
                    kernel_core::precision::ExecutionPhase::Decode
                } else {
                    kernel_core::precision::ExecutionPhase::Prefill
                });
                let planned = symbol.clone();
                prepare(
                    if decode {
                        "ngram_hash_decode"
                    } else {
                        "ngram_hash_prefill"
                    },
                    &symbol,
                    need,
                    move |b: &NgramBuffers<View>, counts| {
                        let rows = rows(counts, capacity)?;
                        one_item(b)?;
                        let mut arguments = check(b, rows, if decode { rows } else { 1 })?;
                        arguments.push(buffer(b.q_len));
                        arguments.extend(tail(b));
                        if decode {
                            arguments.push(Argument::U64(op.entry_words));
                        }
                        Ok(launch(&planned, rows, plan([rows, local, 1]), arguments))
                    },
                )
            }
        }
    }

    fn mtp_gather(
        &self,
        op: &MtpGather,
        need: Need,
    ) -> Result<Prepared<MtpGatherBuffers<View>>, Refused> {
        op.validate()?;
        if *op != MtpGather::of(self.shape) {
            return Err(rule("the MTP gather is compiled for the shape's rows"));
        }
        let symbol = self.tagged(owned::MTP_GATHER);
        let (planned, capacity, row) = (symbol.clone(), need.capacity, u64::from(op.wide) * 2);
        prepare(
            "mtp_gather",
            &symbol,
            need,
            move |b: &MtpGatherBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                for view in [b.last, b.retained] {
                    view.covers(row)?;
                }
                b.x.covers(bytes(rows, row))?;
                for table in [b.sources, b.token_sources, b.token_ids] {
                    table.covers(bytes(rows, 4))?;
                }
                for view in [b.drafts, b.live_rows] {
                    view.covers(4)?;
                }
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1]),
                    vec![
                        buffer(b.last),
                        buffer(b.retained),
                        buffer(b.sources),
                        buffer(b.token_sources),
                        buffer(b.drafts),
                        buffer(b.x),
                        buffer(b.token_ids),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn mtp_norms(
        &self,
        op: &MtpNorms,
        need: Need,
    ) -> Result<Prepared<MtpNormsBuffers<View>>, Refused> {
        op.validate()?;
        if *op != MtpNorms::of(self.shape) {
            return Err(rule("the MTP norms are compiled for the shape's rows"));
        }
        let symbol = self.tagged(owned::MTP_NORMS);
        let hidden = u64::from(op.hidden) * 2;
        let wide = hidden * u64::from(op.streams);
        let (planned, capacity) = (symbol.clone(), need.capacity);
        prepare(
            "mtp_norms",
            &symbol,
            need,
            move |b: &MtpNormsBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                for view in [b.x, b.x_normed] {
                    view.covers(bytes(rows, wide))?;
                }
                for view in [b.embed, b.embed_normed] {
                    view.covers(bytes(rows, hidden))?;
                }
                b.hidden_weight.covers(wide)?;
                b.embed_weight.covers(hidden)?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1]),
                    vec![
                        buffer(b.x),
                        buffer(b.hidden_weight),
                        buffer(b.embed),
                        buffer(b.embed_weight),
                        buffer(b.x_normed),
                        buffer(b.embed_normed),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn mtp_combine(
        &self,
        op: &MtpCombine,
        need: Need,
    ) -> Result<Prepared<MtpCombineBuffers<View>>, Refused> {
        op.validate()?;
        if *op != MtpCombine::of(self.shape) {
            return Err(rule("the MTP combine is compiled for the shape's rows"));
        }
        let symbol = self.tagged(owned::MTP_COMBINE);
        let hidden = u64::from(op.hidden) * 2;
        let (planned, capacity, streams) = (symbol.clone(), need.capacity, op.streams);
        prepare(
            "mtp_combine",
            &symbol,
            need,
            move |b: &MtpCombineBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                for view in [b.hidden, b.x] {
                    view.covers(bytes(rows, hidden * u64::from(streams)))?;
                }
                b.token.covers(bytes(rows, hidden))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, streams, 1]),
                    vec![
                        buffer(b.hidden),
                        buffer(b.token),
                        buffer(b.x),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn draft_argmax_partial(
        &self,
        op: &DraftArgmaxPartial,
        need: Need,
    ) -> Result<Prepared<DraftArgmaxPartialBuffers<View>>, Refused> {
        op.validate()?;
        if *op != DraftArgmaxPartial::of(self.shape, op.rank) || op.rank >= self.shape.ranks() {
            return Err(rule("the draft argmax is compiled for the shape's shards"));
        }
        let symbol = self.per_rank(owned::DRAFT_ARGMAX_PARTIAL);
        let (planned, capacity, op) = (symbol.clone(), need.capacity, *op);
        prepare(
            "draft_argmax_partial",
            &symbol,
            need,
            move |b: &DraftArgmaxPartialBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.logits
                    .covers(bytes(rows, u64::from(op.shard_vocab) * 2))?;
                b.best.covers(bytes(rows, 16))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1]),
                    vec![
                        buffer(b.logits),
                        buffer(b.best),
                        buffer(b.live_rows),
                        Argument::U32(op.rank),
                    ],
                ))
            },
        )
    }

    fn draft_argmax_final(
        &self,
        op: &DraftArgmaxFinal,
        need: Need,
    ) -> Result<Prepared<DraftArgmaxFinalBuffers<View>>, Refused> {
        op.validate()?;
        if *op != DraftArgmaxFinal::of(self.shape) {
            return Err(rule("the draft argmax is compiled for the shape's ranks"));
        }
        let symbol = self.per_rank(owned::DRAFT_ARGMAX_FINAL);
        let (planned, capacity, ranks) = (symbol.clone(), need.capacity, op.ranks);
        prepare(
            "draft_argmax_final",
            &symbol,
            need,
            move |b: &DraftArgmaxFinalBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.bests.covers(bytes(rows, u64::from(ranks) * 16))?;
                for table in [b.slots, b.targets] {
                    table.covers(bytes(rows, 4))?;
                }
                for view in [b.drafts, b.token_ids, b.live_rows] {
                    view.covers(4)?;
                }
                Ok(launch(
                    &planned,
                    rows,
                    plan([rows, 1, 1]),
                    vec![
                        buffer(b.bests),
                        buffer(b.slots),
                        buffer(b.targets),
                        buffer(b.drafts),
                        buffer(b.token_ids),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn hc_up_mix(
        &self,
        op: &HcUpMix,
        need: Need,
    ) -> Result<Prepared<HcUpMixBuffers<View>>, Refused> {
        op.validate()?;
        if *op != HcUpMix::of(self.shape, op.input) {
            return Err(rule(
                "the up mix is compiled for the shape's hyper-connection",
            ));
        }
        let hc = super::catalog::geometry_owner(self.shape).hc;
        let (symbol, padded) = match op.input {
            UpMixInput::Down => (hc.up_mix_symbol, false),
            UpMixInput::Bottleneck if self.padded_mix => {
                (kernels_sm120::gemm::fused::HC_PADDED_MIX, true)
            }
            UpMixInput::Bottleneck => (kernels_sm120::gemm::fused::HC_PRECOMPUTED_MIX, false),
        };
        let planner = move |rows: u32| {
            if padded {
                kernels_sm120::gemm::fused::hc_padded_mix_plan(hc, rows)
            } else {
                kernels_sm120::gemm::fused::hc_up_mix_plan(hc, rows)
            }
        };
        // The dynamic shared memory does not depend on the rows.
        let shared = planner(need.capacity)?.dynamic_shared_bytes;
        let row = |width: u32| u64::from(width) * 2;
        let (capacity, op) = (need.capacity, *op);
        let wide = op.width * op.streams;
        prepare_info(
            "hc_up_mix",
            symbol,
            need,
            |info| {
                info.symbols = vec![(symbol.to_owned(), shared)];
                // The up weight in BF16 fragments, as the packager writes it.
                info.weight_layouts = vec![(
                    "weight".into(),
                    kernel_core::precision::WeightLayout {
                        id: "bf16-fragment".into(),
                        version: 1,
                    },
                )];
            },
            move |b: &HcUpMixBuffers<View>, counts| {
                let rows = rows(counts, capacity)?;
                b.weight.covers(u64::from(wide) * row(op.rank))?;
                b.input.covers(bytes(rows, row(op.rank)))?;
                b.normed.covers(bytes(rows, row(wide)))?;
                b.output.covers(bytes(rows, row(op.width)))?;
                b.live_rows.covers(4)?;
                Ok(launch(
                    symbol,
                    rows,
                    planner(rows)?,
                    vec![
                        buffer(b.weight),
                        buffer(b.input),
                        buffer(b.normed),
                        buffer(b.output),
                        buffer(b.live_rows),
                    ],
                ))
            },
        )
    }

    fn dense_group(
        &self,
        op: &model_qwen38_contract::own_ops::dense::DenseGroup,
        usage: kernel_core::precision::OperationUse,
        need: Need,
    ) -> Result<Prepared<model_qwen38_contract::own_ops::dense::DenseGroupBuffers<View>>, Refused>
    {
        dense::prepare(self.shape, op, usage, need)
    }
}

impl Sm120Own {
    /// The hash-only n-gram entry of `phase`.
    fn hash(&self, phase: kernel_core::precision::ExecutionPhase) -> String {
        self.tagged(&model_qwen38_contract::sm120::generated::ngram_hash_symbol(
            self.shape, phase,
        ))
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    use super::*;
    use crate::contract::{MAX_VERIFY_TOKENS, PREFILL_GRANULE, PREFILL_TOKENS, Ranks};

    /// Room for any test buffer.
    const BIG: u64 = 1 << 40;

    /// A span of its own buffer `id`.
    fn span(id: u32) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, id), 0, BIG)
    }

    /// `length` bytes of buffer `id` from `offset`.
    fn part(id: u32, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, id),
            offset,
            length,
        )
    }

    /// The view a span binds as: its buffer and offset name the handle.
    fn view(span: DeviceBufferSpanV1) -> View {
        View {
            handle: u64::from(span.buffer.slot()) << 40 | span.offset,
            bytes: span.length,
        }
    }

    /// Every shape the bundle serves: each geometry at TP2 and TP4.
    fn shapes() -> Vec<Shape> {
        model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .flat_map(|geometry| {
                [Ranks::TP2, Ranks::TP4]
                    .into_iter()
                    .filter_map(move |ranks| Shape::of_geometry(geometry, ranks))
            })
            .collect()
    }

    fn step(rows: u32, items: u32) -> Counts {
        Counts::rows(rows).with_items(items, 0)
    }

    /// The op's buffers over the same spans, with `cu_rows` and `live` for
    /// verify forms.
    fn ngram_buffers(
        gather: bool,
        [tokens, history, ids, history_out, output]: [DeviceBufferSpanV1; 5],
        side: &[DeviceBufferSpanV1; 12],
        q_len: DeviceBufferSpanV1,
        verify: bool,
        (table_start, table_rows): (u64, u64),
    ) -> NgramBuffers<View> {
        NgramBuffers {
            tokens: view(tokens),
            history: view(history),
            multipliers: view(side[2]),
            sizes: view(side[3]),
            offsets: view(side[4]),
            table: gather.then_some(view(side[10])),
            scale: gather.then_some(view(side[11])),
            output: gather.then_some(view(output)),
            row_ids: view(ids),
            history_out: view(history_out),
            cu_rows: verify.then_some(view(side[7])),
            q_len: view(q_len),
            live_items: verify.then_some(view(side[8])),
            table_start,
            table_rows,
        }
    }

    /// The n-gram reads and writes two history words per item and the
    /// convolution the history rows of each item's entry: a view that ends
    /// right after the last item's is enough, one byte less is refused.
    #[test]
    fn item_histories_need_what_the_kernels_read() {
        let shape = Shape::TP4;
        let own = Sm120Own::new(shape, false);
        let (entry, conv) = (shape.ple_entry_bytes(), shape.ple_conv_bytes());
        let graph = Serving::new(shape, 4, true);
        let side = graph.ngram_side(0);
        let shard = model_qwen38_contract::views::ngram_rows(0, shape).unwrap();
        let ngram = own
            .ngram(
                &Ngram::of(shape, 0, NgramWrites::Hash, ItemForm::Decode),
                Need::rows(4),
            )
            .unwrap();
        let conv_items = own
            .ple_conv(&PleConv::of(shape, ItemForm::Decode), Need::rows(4))
            .unwrap();
        for items in [1, 3] {
            for (bytes, fits) in [
                (u64::from(items - 1) * entry + 8, true),
                (u64::from(items - 1) * entry + 7, false),
            ] {
                let history = part(11, conv, bytes);
                let spans = [
                    side[0],
                    history,
                    side[9],
                    history,
                    graph.scratch.ngram_local,
                ];
                let buffers = ngram_buffers(false, spans, &side, side[6], false, shard);
                assert_eq!(
                    ngram
                        .launches(&buffers, View::EMPTY, Counts::rows(items))
                        .is_ok(),
                    fits,
                    "n-gram history of {bytes} bytes at {items} items"
                );
            }
            for (bytes, fits) in [
                (u64::from(items - 1) * entry + conv, true),
                (u64::from(items - 1) * entry + conv - 1, false),
            ] {
                let state = view(part(12, 0, bytes));
                let buffers = PleConvBuffers {
                    input: view(span(1)),
                    weight: view(span(2)),
                    state,
                    output: view(span(4)),
                    state_out: state,
                    cu_rows: None,
                    q_len: view(span(7)),
                    live_items: None,
                };
                assert_eq!(
                    conv_items
                        .launches(&buffers, View::EMPTY, Counts::rows(items))
                        .is_ok(),
                    fits,
                    "convolution state of {bytes} bytes at {items} items"
                );
            }
        }
    }

    /// Each op claims what its launched symbol's declaration does, and
    /// refuses a need beyond it.
    #[test]
    fn guarantees_are_the_launched_symbols_own_declaration() {
        for shape in shapes() {
            let own = Sm120Own::new(shape, false);
            let need = Need::rows(MAX_VERIFY_TOKENS);
            let layout = PleLayout::of(shape);
            let guarantees = [
                (
                    own.ple_residual(&PleResidual::of(shape), need)
                        .unwrap()
                        .info()
                        .guarantees,
                    Invariance::any_rows(PREFILL_TOKENS),
                ),
                (
                    own.ple_stage(&PleStage { layout }, need)
                        .unwrap()
                        .info()
                        .guarantees,
                    Invariance::COPY,
                ),
                (
                    own.ple_checkpoints(&PleCheckpoints::of(shape), need)
                        .unwrap()
                        .info()
                        .guarantees,
                    Invariance::COPY,
                ),
                (
                    own.ple_anchor(&PleAnchor { layout }, need)
                        .unwrap()
                        .info()
                        .guarantees,
                    Invariance {
                        chunk: kernel_core::properties::Chunk::Granule(PREFILL_GRANULE),
                        anchors: kernel_core::properties::Anchors::Spare,
                        ..Invariance::NONE
                    },
                ),
                (
                    own.mtp_gather(&MtpGather::of(shape), need)
                        .unwrap()
                        .info()
                        .guarantees,
                    Invariance::NONE,
                ),
            ];
            for (index, (got, expected)) in guarantees.into_iter().enumerate() {
                assert_eq!(got, expected, "{shape:?} op {index}");
            }
            // The verify convolution and n-gram give verify rows the decode
            // rows' bits; the decode forms claim what their catalog entries
            // declare.
            let verify = ItemForm::Verify {
                rows: MAX_VERIFY_ROWS,
            };
            let exact =
                Need::rows(MAX_VERIFY_TOKENS).with_exact(Invariance::one_row(MAX_VERIFY_ROWS));
            let conv = own.ple_conv(&PleConv::of(shape, verify), exact).unwrap();
            assert_eq!(conv.info().guarantees, Invariance::one_row(MAX_VERIFY_ROWS));
            for rank in 0..shape.ranks() {
                let ngram = own
                    .ngram(&Ngram::of(shape, rank, NgramWrites::Gather, verify), exact)
                    .unwrap();
                assert_eq!(
                    ngram.info().guarantees,
                    Invariance::one_row(MAX_VERIFY_ROWS)
                );
            }
            for form in [ItemForm::Decode, ItemForm::One] {
                let prepared = own
                    .ple_conv(&PleConv::of(shape, form), Need::rows(1))
                    .unwrap();
                let symbol = &prepared.info().symbols[0].0;
                assert_eq!(prepared.info().guarantees, declared(symbol));
            }
            // An unmet need is refused: the gather declares nothing.
            assert!(matches!(
                own.mtp_gather(&MtpGather::of(shape), exact),
                Err(Refused::Inexact { .. })
            ));
            // The hash entries serve what serving asks of them: the verify
            // rows' one-row bits, and prefill chunks at the granule with
            // the PLE anchor deriving their history; decode asks nothing,
            // and its entry declares nothing.
            let granule = Invariance {
                chunk: kernel_core::properties::Chunk::Granule(PREFILL_GRANULE),
                anchors: kernel_core::properties::Anchors::Companion,
                ..Invariance::NONE
            };
            for rank in 0..shape.ranks() {
                let hash = |items, need| {
                    own.ngram(&Ngram::of(shape, rank, NgramWrites::Hash, items), need)
                };
                let prepared = hash(verify, exact).unwrap();
                assert_eq!(
                    prepared.info().guarantees,
                    Invariance::one_row(MAX_VERIFY_ROWS)
                );
                let prepared = hash(
                    ItemForm::One,
                    Need::rows(PREFILL_TOKENS).with_exact(granule),
                )
                .unwrap();
                assert!(prepared.info().guarantees.provides(granule));
                // Its history advances only at an item's last row, so a
                // spare entry at anchors is the PLE anchor's (Companion),
                // never the hash's own: Spare needs are refused.
                for anchors in [
                    kernel_core::properties::Anchors::Spare,
                    kernel_core::properties::Anchors::SpareAfter {
                        rows: PREFILL_GRANULE,
                    },
                ] {
                    assert!(matches!(
                        hash(
                            ItemForm::One,
                            Need::rows(PREFILL_TOKENS)
                                .with_exact(Invariance { anchors, ..granule })
                        ),
                        Err(Refused::Inexact { .. })
                    ));
                }
                assert!(matches!(
                    hash(
                        ItemForm::Decode,
                        Need::rows(4).with_exact(Invariance::one_row(4))
                    ),
                    Err(Refused::Inexact { .. })
                ));
            }
        }
    }

    /// Ops other than the compiled ones, launches past capacity and views
    /// short of what a launch touches are refused.
    #[test]
    fn mismatched_ops_rows_and_views_are_refused() {
        let [a, b, c, d, e] = [1, 2, 3, 4, 5].map(span);
        let shape = Shape::TP4;
        let own = Sm120Own::new(shape, false);
        let need = Need::rows(32);
        let residual = PleResidual::of(shape);
        assert!(
            own.ple_residual(
                &PleResidual {
                    hidden: residual.hidden / 2,
                    ..residual
                },
                need
            )
            .is_err()
        );
        let layout = PleLayout::of(shape);
        assert!(
            own.ple_stage(
                &PleStage {
                    layout: PleLayout {
                        conv_rows: layout.conv_rows + 1,
                        ..layout
                    }
                },
                need
            )
            .is_err()
        );
        assert!(
            own.ple_checkpoints(
                &PleCheckpoints {
                    verify_rows: 4,
                    ..PleCheckpoints::of(shape)
                },
                need
            )
            .is_err()
        );
        assert!(
            own.ple_conv(&PleConv::of(shape, ItemForm::Verify { rows: 4 }), need)
                .is_err()
        );
        assert!(
            own.draft_argmax_final(&DraftArgmaxFinal::of(Shape::TP2), need)
                .is_err()
        );
        assert!(
            own.draft_argmax_partial(&DraftArgmaxPartial::of(shape, 4), need)
                .is_err()
        );
        assert!(own.ple_residual(&residual, Need::rows(0)).is_err());
        let prepared = own.ple_residual(&residual, need).unwrap();
        let buffers = PleResidualBuffers {
            x: view(a),
            gated: view(b),
            conv: view(c),
            out: view(d),
            live_rows: view(e),
        };
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        let row = u64::from(shape.wide()) * 2;
        let short = PleResidualBuffers {
            out: view(part(4, 0, 3 * row - 1)),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(3))
                .is_err()
        );
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(2))
                .is_ok()
        );
        // Item ops need items, at most their rows.
        let stage = own.ple_stage(&PleStage { layout }, need).unwrap();
        let staged = PleStageBuffers {
            pool: view(a),
            staging: view(b),
            slot_ids: view(c),
            prefix_len: view(d),
            live_items: view(e),
        };
        assert!(
            stage
                .launches(&staged, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        assert!(stage.launches(&staged, View::EMPTY, step(4, 5)).is_err());
        let few = PleStageBuffers {
            staging: view(part(2, 0, 3 * layout.entry_bytes())),
            ..staged
        };
        assert!(stage.launches(&few, View::EMPTY, step(4, 4)).is_err());
        assert!(stage.launches(&few, View::EMPTY, step(4, 3)).is_ok());
    }

    /// One rank's buffers as an instance serving `items` items lays them
    /// out: its scratch (`RankScratch::allocate`, each buffer its own
    /// allocation of exactly its bytes) and a graph (decode and verify) or
    /// prefill phase's input and side arenas.
    struct Serving {
        shape: Shape,
        scratch: crate::scratch::RankScratch,
        inputs: crate::upload::DeviceInputs,
        side: crate::side::SideInputs,
        graph: bool,
    }

    impl Serving {
        fn new(shape: Shape, items: u32, graph: bool) -> Serving {
            let items = crate::contract::Items::new(items).unwrap();
            let served = crate::scratch::Served::new(items);
            let mut next = 1000;
            let mut allocate = |bytes: u64| {
                next += 1;
                Ok(part(next, 0, bytes))
            };
            let speculation = crate::scratch::Speculation {
                verify_rows: MAX_VERIFY_ROWS,
                drafter: true,
            };
            let scratch = crate::scratch::RankScratch::allocate(
                &crate::sm120::comm_bytes(shape, served),
                0,
                speculation,
                shape.kv_formats()[1].page_bytes,
                shape,
                served,
                &mut allocate,
            )
            .unwrap();
            let layout = if graph {
                crate::upload::InputLayout::verify_within::<crate::contract::Qwen>(items)
            } else {
                crate::upload::InputLayout::within(
                    PREFILL_TOKENS,
                    crate::contract::SCORED_ROWS,
                    items,
                )
            };
            let side = crate::side::SideLayout::for_shape(
                shape,
                if graph { served.rows } else { PREFILL_TOKENS },
                served.items,
            );
            let arena = crate::upload::PhaseArena {
                arena: allocate(layout.end).unwrap(),
                block_tables: allocate(BIG).unwrap(),
            };
            let inputs = crate::upload::DeviceInputs::within(arena, &layout);
            let side = crate::side::SideInputs::within(allocate(side.end).unwrap(), &side);
            Serving {
                shape,
                scratch,
                inputs,
                side,
                graph,
            }
        }

        /// `span` from row `row` of `row_bytes`-byte rows on (`lower::at`).
        fn at(span: DeviceBufferSpanV1, row: u64, row_bytes: u64) -> DeviceBufferSpanV1 {
            let skip = (row * row_bytes).min(span.length);
            DeviceBufferSpanV1::new(span.buffer, span.offset + skip, span.length - skip)
        }

        /// The n-gram stage's side spans (`lower/projection.rs`
        /// `projection_ngram`) and the rank's table shard, multipliers,
        /// sizes, offsets and scale at their weights' sizes.
        fn ngram_side(&self, rank: u32) -> [DeviceBufferSpanV1; 12] {
            let s = &self.scratch;
            let staging = s.ple_staging[usize::from(self.graph)];
            let conv = self.shape.ple_conv_bytes();
            let (_, shard_rows) =
                model_qwen38_contract::views::ngram_rows(rank, self.shape).unwrap();
            let heads = u64::from(self.shape.ngram_heads());
            [
                self.inputs.token_ids,
                Self::at(staging.input, 1, conv),
                part(901, 0, u64::from(self.shape.ngram_size()) * 8),
                part(902, 0, heads * 8),
                part(903, 0, heads * 8),
                Self::at(staging.output, 1, conv),
                self.inputs.q_len,
                self.side.cu_rows,
                self.inputs.live_batch,
                s.ngram_rows,
                part(
                    904,
                    0,
                    shard_rows * u64::from(self.shape.ngram_head_width()),
                ),
                part(905, 0, 2),
            ]
        }
    }
}
