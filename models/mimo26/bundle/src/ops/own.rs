//! MiMo's own kernels through `OwnOps` (`model_mimo26_contract::own_ops`):
//! the sliding-window ring's backups and restore, the prefix snapshot
//! copies, the drafters' retained and gathered rows, DFlash's row copies,
//! mask rows, context cache writes and drafts.

use kernel_core::backend::{Counts, Need, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use model_mimo26_contract::own_ops::OwnOps;
use model_mimo26_contract::own_ops::dflash::{
    CacheWrite, CacheWriteBuffers, CopyRows, CopyRowsBuffers, Drafts, DraftsBuffers, MaskRows,
    MaskRowsBuffers,
};
use model_mimo26_contract::own_ops::retain::{
    GatherRows, GatherRowsBuffers, RetainRows, RetainRowsBuffers, row_bytes,
};
use model_mimo26_contract::own_ops::ring::{
    Direction, RingBackup, RingBackupBuffers, RingLayout, RingPrefixCopy, RingPrefixCopyBuffers,
    RingRestore, RingRestoreBuffers,
};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Bind, ByPhase, need, one};
use crate::contract::MAX_VERIFY_ROWS;
use crate::program_lower::Lowered;
use crate::selection::{MimoPhase as _, Phase};
use crate::switches::Switches;
use crate::topology::Tp;
use crate::upload::DeviceInputs;

/// What an instance prepares its own kernels' ops for.
#[derive(Clone, Copy)]
pub struct OwnPlan<'a> {
    /// The bundle's own kernels.
    pub ops: &'a dyn OwnOps,
    /// The capture of a slot's prefix snapshot (`prefix::copy_op`), whose
    /// restore differs only in its direction, when the instance's request
    /// state fits the copy kernel.
    pub prefix: Option<RingPrefixCopy>,
    /// The sequences the instance serves (`Items::get`): one slot each of
    /// DFlash's retained context features.
    pub served: u32,
    /// Whether the instance drafts with MTP.
    pub mtp: bool,
}

impl<'a> OwnPlan<'a> {
    /// The own ops on `ops` of an instance serving `served` sequences,
    /// without prefix snapshots or MTP.
    pub fn new(ops: &'a dyn OwnOps, served: u32) -> Self {
        Self {
            ops,
            prefix: None,
            served,
            mtp: false,
        }
    }
}

type Backup = Prepared<RingBackupBuffers<View>>;
type PrefixCopy = Prepared<RingPrefixCopyBuffers<View>>;
type Retain = Prepared<RetainRowsBuffers<View>>;
type Gather = Prepared<GatherRowsBuffers<View>>;

/// The own kernels' ops of an instance, which every rank launches alike.
#[derive(Default)]
pub(super) struct Own {
    /// The ring's slots: one per item of the largest bucket.
    slots: u32,
    /// Each phase's backup of the ring cells it overwrites.
    backup: ByPhase<Backup>,
    /// A verify step's restore of its rejected ring cells.
    restore: Option<Prepared<RingRestoreBuffers<View>>>,
    /// The prefix snapshot copy, then its capture and restore.
    prefix: Option<(RingPrefixCopy, [PrefixCopy; 2])>,
    /// The drafters' writes of decode and verify rows into their retained
    /// histories (MTP's hidden rows, each rank's DFlash tap columns), by
    /// shape and phase.
    retains: Vec<(RetainRows, ByPhase<Retain>)>,
    /// DFlash's gathers of retained context features, by shape and phase.
    gathers: Vec<(GatherRows, ByPhase<Gather>)>,
    /// DFlash's row copies (each rank's tap columns, the context's zeroed
    /// residual, a draft block's scaled V), mask rows, context cache writes
    /// and drafts, by shape and phase.
    copies: Vec<(CopyRows, ByPhase<Prepared<CopyRowsBuffers<View>>>)>,
    masks: Vec<(MaskRows, ByPhase<Prepared<MaskRowsBuffers<View>>>)>,
    writes: Vec<(CacheWrite, ByPhase<Prepared<CacheWriteBuffers<View>>>)>,
    drafts: Vec<(Drafts, ByPhase<Prepared<DraftsBuffers<View>>>)>,
}

/// The positions of an item's ring a `phase` step's backup saves: the one
/// a decode row writes, the verify rows, or a prefill's whole window.
fn span(tp: Tp, phase: Phase) -> u32 {
    match phase {
        Phase::Decode(_) => 1,
        Phase::Verify(_) => MAX_VERIFY_ROWS,
        Phase::Prefill => tp.geometry.window(),
    }
}

impl Own {
    /// The own ops of a `tp` instance of `slots` ring slots.
    pub(super) fn prepare(
        plan: &OwnPlan<'_>,
        tp: Tp,
        switches: Switches,
        slots: u32,
        dflash: bool,
    ) -> Result<Self, Refused> {
        let own = plan.ops;
        let layout = RingLayout::of(tp, slots);
        // A step backs up at most its bucket's items and the ring's slots,
        // with the phase's exactness: prefill's and an exact verify
        // bucket's.
        let backup = ByPhase::prepare(|phase| {
            let need = need(phase, switches)?;
            let need = Need {
                capacity: need.capacity.min(slots),
                ..need
            };
            let op = RingBackup {
                layout,
                span: span(tp, phase),
            };
            own.ring_backup(&op, need).map(Some)
        })?;
        let restore = own.ring_restore(
            &RingRestore {
                layout,
                span: MAX_VERIFY_ROWS,
            },
            Need::rows(slots),
        )?;
        let prefix = plan
            .prefix
            .map(|capture| {
                let copy = |direction| {
                    own.ring_prefix_copy(
                        &RingPrefixCopy {
                            direction,
                            ..capture
                        },
                        Need::rows(1),
                    )
                };
                Ok::<_, Refused>((
                    capture,
                    [copy(Direction::Capture)?, copy(Direction::Restore)?],
                ))
            })
            .transpose()?;
        // MTP retains each row's hidden state, DFlash each rank's columns
        // of every tap its fc shard covers, at decode and verify buckets;
        // DFlash gathers them back at every phase.
        let mut retained = vec![];
        if plan.mtp {
            retained.push(crate::mtp::program::retain(tp));
        }
        let (mut gathered, mut taps, mut zeros, mut scaled) = (vec![], vec![], vec![], vec![]);
        let (mut masks, mut writes, mut drafts) = (vec![], vec![], vec![]);
        if dflash {
            use crate::dflash::program as dflash;
            let tapped = tp.geometry.dflash().dflash_config.target_layer_ids.len();
            for rank in 0..tp.ranks() {
                for tap in 0..tapped {
                    if let Some((_, _, width)) = dflash::tap_columns(rank, tap, tp) {
                        retained.push(dflash::retain(tp, plan.served, width as u32));
                        taps.push(dflash::copy_tap(tp, width as u32));
                    }
                }
            }
            gathered.push(dflash::gather(tp, plan.served));
            // The context's zeroed residual at every phase, and a draft
            // block's scaled V, mask rows and drafts at verify buckets; the
            // context cache writes at every phase.
            zeros.push(dflash::zero_residual(tp));
            scaled.push(dflash::scale_v(tp));
            masks.push(dflash::mask(tp));
            writes.push(dflash::cache_write(tp, slots));
            drafts.push(dflash::drafts(tp));
        }
        let copy = |op: &CopyRows, need| own.dflash_copy_rows(op, need);
        let mut copies = by_shape(taps, every, copy)?;
        copies.extend(by_shape(zeros, every, copy)?);
        copies.extend(by_shape(scaled, verify, copy)?);
        Ok(Self {
            slots,
            backup,
            restore: Some(restore),
            prefix,
            retains: by_shape(retained, bucketed, |op, need| own.retain_rows(op, need))?,
            gathers: by_shape(gathered, every, |op, need| own.gather_rows(op, need))?,
            copies,
            masks: by_shape(masks, verify, |op, need| own.dflash_mask_rows(op, need))?,
            writes: by_shape(writes, every, |op, need| own.dflash_cache_write(op, need))?,
            drafts: by_shape(drafts, verify, |op, need| own.dflash_drafts(op, need))?,
        })
    }

    /// The counts of an own op over `rows` rows of a step whose per-item
    /// tables hold the instance's largest bucket of items.
    fn counts(&self, rows: u32) -> Counts {
        Counts::rows(rows).with_items(self.slots, 0)
    }

    /// Refuses a ring of other slots than the ops were prepared for.
    fn ring(&self, slots: u32) -> Result<(), Refusal> {
        if slots != self.slots {
            return Err(Refusal("MiMo's ring ops serve the instance's ring"));
        }
        Ok(())
    }
}

/// The ring and inputs a backup reads.
pub struct RingBackupViews {
    /// Both halves of the ring state of `slots` slots.
    pub ring_state: DeviceBufferSpanV1,
    pub slots: u32,
    pub slot_ids: DeviceBufferSpanV1,
    pub q_len: DeviceBufferSpanV1,
    pub prefix_len: DeviceBufferSpanV1,
    pub live_batch: DeviceBufferSpanV1,
}

/// The ring and inputs a restore reads: per item, the slot, first rejected
/// position and rejected count, then the item count.
pub struct RingRestoreViews {
    /// Both halves of the ring state of `slots` slots.
    pub ring_state: DeviceBufferSpanV1,
    pub slots: u32,
    pub slot_ids: DeviceBufferSpanV1,
    pub starts: DeviceBufferSpanV1,
    pub counts: DeviceBufferSpanV1,
    pub live_items: DeviceBufferSpanV1,
}

/// Which phases an own op serves.
type Serves = fn(Phase) -> bool;

fn every(_: Phase) -> bool {
    true
}

fn bucketed(phase: Phase) -> bool {
    phase.token_bucketed()
}

fn verify(phase: Phase) -> bool {
    matches!(phase, Phase::Verify(_))
}

/// An op of each distinct one of `shapes` at every phase `serves` admits,
/// for the phase's rows.
fn by_shape<O: PartialEq, P>(
    shapes: Vec<O>,
    serves: Serves,
    prepare: impl Fn(&O, Need) -> Result<P, Refused>,
) -> Result<Vec<(O, ByPhase<P>)>, Refused> {
    let mut ops: Vec<(O, ByPhase<P>)> = Vec::new();
    for shape in shapes {
        if ops.iter().any(|(prepared, _)| *prepared == shape) {
            continue;
        }
        let prepared = ByPhase::prepare(|phase| {
            if !serves(phase) {
                return Ok(None);
            }
            prepare(&shape, Need::rows(phase.capacity()?)).map(Some)
        })?;
        ops.push((shape, prepared));
    }
    Ok(ops)
}

/// The prepared op of `op`'s shape at `phase` (bucket `bucket`).
fn shaped<'a, O: PartialEq, P>(
    ops: &'a [(O, ByPhase<P>)],
    op: &O,
    phase: Phase,
    bucket: usize,
) -> Result<&'a P, Refusal> {
    ops.iter()
        .find(|(prepared, _)| prepared == op)
        .ok_or(Refusal("MiMo prepared no own op of this shape"))?
        .1
        .get(phase, bucket)
}

impl super::Ops {
    /// The backup of the ring cells a `phase` step (bucket `bucket` of
    /// `COMM_BUCKETS`) of `live_batch` items overwrites. Decode items
    /// write one position, so a decode bucket covers the items of its
    /// live rows' bucket; verify covers its items' bucket and prefill its
    /// items.
    pub fn ring_backup(
        &self,
        phase: Phase,
        bucket: usize,
        live_batch: u32,
        views: &RingBackupViews,
    ) -> Result<Lowered, Refusal> {
        let own = &self.own;
        own.ring(views.slots)?;
        let items = match phase {
            Phase::Decode(live) => crate::graphs::decode_bucket(live),
            Phase::Verify(_) => crate::graphs::decode_bucket(live_batch),
            Phase::Prefill => Some(live_batch),
        }
        .ok_or(Refusal("MiMo ring backup item count"))?;
        let op = own.backup.get(phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RingBackupBuffers {
            state: bind.view(views.ring_state),
            slot_ids: bind.view(views.slot_ids),
            q_len: bind.view(views.q_len),
            prefix_len: bind.view(views.prefix_len),
            live_items: bind.view(views.live_batch),
        };
        one(bind.lower(op.as_ref(), &buffers, Counts::rows(items))?)
    }

    /// The restore of `items` items' rejected ring cells after a verify
    /// step, each at most the verify rows.
    pub fn ring_restore(&self, items: u32, views: &RingRestoreViews) -> Result<Lowered, Refusal> {
        let own = &self.own;
        own.ring(views.slots)?;
        let op = own
            .restore
            .as_ref()
            .ok_or(Refusal("MiMo prepared no ring restore"))?;
        let mut bind = Bind::default();
        let buffers = RingRestoreBuffers {
            state: bind.view(views.ring_state),
            slot_ids: bind.view(views.slot_ids),
            starts: bind.view(views.starts),
            counts: bind.view(views.counts),
            live_items: bind.view(views.live_items),
        };
        one(bind.lower(op.as_ref(), &buffers, Counts::rows(items))?)
    }

    /// The copy `op` of `slot`'s prefix snapshot between the ring and
    /// drafter state and the snapshot pages: the op the instance prepared
    /// for its request state, in either direction.
    pub fn prefix_copy(
        &self,
        op: &RingPrefixCopy,
        slot: u32,
        views: &crate::prefix::PrefixCopyViews,
    ) -> Result<Lowered, Refusal> {
        let (capture, copies) = self
            .own
            .prefix
            .as_ref()
            .ok_or(Refusal("MiMo prepared no prefix snapshot copy"))?;
        if (RingPrefixCopy {
            direction: op.direction,
            ..*capture
        }) != *op
        {
            return Err(Refusal(
                "MiMo's prefix copy serves its request state's snapshots",
            ));
        }
        let copy = &copies[usize::from(op.direction == Direction::Restore)];
        let mut bind = Bind::default();
        let buffers = RingPrefixCopyBuffers {
            state: bind.view(views.ring_state),
            drafter: views.drafter.map(|span| bind.view(span)),
            pool: bind.view(views.pool),
            pages: bind.view(views.pages),
            slot,
        };
        one(bind.lower(copy.as_ref(), &buffers, Counts::rows(1))?)
    }
    /// The write of `op`'s columns of every row of a decode or verify
    /// `phase` step (bucket `bucket`) from `input` into its cell of the
    /// retained `history`, each view at the first column it copies.
    pub fn retain_rows(
        &self,
        op: &RetainRows,
        phase: Phase,
        bucket: usize,
        input: DeviceBufferSpanV1,
        history: DeviceBufferSpanV1,
        inputs: &DeviceInputs,
    ) -> Result<Lowered, Refusal> {
        let own = &self.own;
        let prepared = shaped(&own.retains, op, phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = RetainRowsBuffers {
            input: bind.aligned(input, 16)?,
            history: bind.aligned(history, 16)?,
            slot_ids: bind.view(inputs.slot_ids),
            token_batch: bind.view(inputs.token_batch),
            q_start: bind.view(inputs.q_start),
            prefix_len: bind.view(inputs.prefix_len),
            live_rows: bind.view(inputs.live_tokens),
        };
        let counts = own.counts(phase.capacity()?);
        one(bind.lower(prepared.as_ref(), &buffers, counts)?)
    }

    /// The gather of `op`'s columns of every row of a `phase` step (bucket
    /// `bucket`) from its cell of the retained `history` into `output`, each
    /// view at the first column it copies. A decode or verify step gathers
    /// its bucket's rows, a prefill step its `live_tokens` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_rows(
        &self,
        op: &GatherRows,
        phase: Phase,
        bucket: usize,
        live_tokens: u32,
        history: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        inputs: &DeviceInputs,
    ) -> Result<Lowered, Refusal> {
        let own = &self.own;
        let prepared = shaped(&own.gathers, op, phase, bucket)?;
        let rows = if phase.token_bucketed() {
            phase.capacity()?
        } else {
            live_tokens
        };
        let mut bind = Bind::default();
        let buffers = GatherRowsBuffers {
            history: bind.aligned(history, 16)?,
            output: bind.aligned(output, 16)?,
            slot_ids: bind.view(inputs.slot_ids),
            token_batch: bind.view(inputs.token_batch),
            positions: bind.view(inputs.positions),
            live_rows: bind.view(inputs.live_tokens),
        };
        one(bind.lower(prepared.as_ref(), &buffers, own.counts(rows))?)
    }
    /// DFlash's copy `op` of `live_tokens` rows of a `phase` step (bucket
    /// `bucket`) from `input` (none zeroes them) into `output`, each view at
    /// its first copied column and holding the phase's capacity of rows. It
    /// grids the phase's row bucket (`MimoPhase::row_grid`).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_copy_rows(
        &self,
        op: &CopyRows,
        phase: Phase,
        bucket: usize,
        switches: Switches,
        live_tokens: u32,
        input: Option<DeviceBufferSpanV1>,
        output: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
    ) -> Result<Lowered, Refusal> {
        let prepared = shaped(&self.own.copies, op, phase, bucket)?;
        let capacity = phase.capacity()?;
        let holds = |span: DeviceBufferSpanV1, stride: u32| {
            span.length >= row_bytes(capacity, stride, op.width)
        };
        if input.is_some_and(|span| !op.input_stride.is_some_and(|stride| holds(span, stride)))
            || !holds(output, op.output_stride)
        {
            return Err(Refusal("DFlash row copy geometry"));
        }
        let mut bind = Bind::default();
        let buffers = CopyRowsBuffers {
            input: input.map(|span| bind.aligned(span, 16)).transpose()?,
            output: bind.aligned(output, 16)?,
            live_rows: bind.view(live_rows),
        };
        let counts = self
            .own
            .counts(phase.row_grid(live_tokens, switches, self.row_buckets)?);
        one(bind.lower(prepared.as_ref(), &buffers, counts)?)
    }

    /// DFlash's mask embedding `mask` into the draft rows of a verify
    /// `phase` step's `activation` (bucket `bucket`), over its capacity.
    pub fn dflash_mask_rows(
        &self,
        op: &MaskRows,
        phase: Phase,
        bucket: usize,
        activation: DeviceBufferSpanV1,
        mask: DeviceBufferSpanV1,
        inputs: &DeviceInputs,
    ) -> Result<Lowered, Refusal> {
        let prepared = shaped(&self.own.masks, op, phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = MaskRowsBuffers {
            activation: bind.view(activation),
            mask: bind.view(mask),
            positions: bind.view(inputs.positions),
            prefix_len: bind.view(inputs.prefix_len),
            token_batch: bind.view(inputs.token_batch),
            live_rows: bind.view(inputs.live_tokens),
        };
        let counts = self.own.counts(phase.capacity()?);
        one(bind.lower(prepared.as_ref(), &buffers, counts)?)
    }

    /// DFlash's write `op` of a `phase` step's context K and V rows (bucket
    /// `bucket`) into one layer's cache planes, gridding its capacity. A
    /// decode or verify step's rows and tables hold its bucket, a prefill
    /// step's (a catch-up's packed inputs) its `live_tokens` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_cache_write(
        &self,
        op: &CacheWrite,
        phase: Phase,
        bucket: usize,
        live_tokens: u32,
        k: DeviceBufferSpanV1,
        v: DeviceBufferSpanV1,
        [k_cache, v_cache]: [DeviceBufferSpanV1; 2],
        inputs: &DeviceInputs,
    ) -> Result<Lowered, Refusal> {
        let prepared = shaped(&self.own.writes, op, phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = CacheWriteBuffers {
            k: bind.aligned(k, 16)?,
            v: bind.aligned(v, 16)?,
            k_cache: bind.aligned(k_cache, 16)?,
            v_cache: bind.aligned(v_cache, 16)?,
            slot_ids: bind.view(inputs.slot_ids),
            token_batch: bind.view(inputs.token_batch),
            positions: bind.view(inputs.positions),
            last_positions: bind.view(inputs.last_positions),
            live_rows: bind.view(inputs.live_tokens),
        };
        let capacity = phase.capacity()?;
        let rows = if phase.token_bucketed() {
            capacity
        } else {
            live_tokens
        };
        let counts = Counts {
            padded_rows: capacity,
            ..self.own.counts(rows)
        };
        one(bind.lower(prepared.as_ref(), &buffers, counts)?)
    }

    /// DFlash's drafts of a verify `phase` step (bucket `bucket`): each
    /// draft row's best of every rank's `gathered` pairs into its input
    /// token, over its capacity.
    pub fn dflash_drafts(
        &self,
        op: &Drafts,
        phase: Phase,
        bucket: usize,
        gathered: DeviceBufferSpanV1,
        inputs: &DeviceInputs,
    ) -> Result<Lowered, Refusal> {
        let prepared = shaped(&self.own.drafts, op, phase, bucket)?;
        let mut bind = Bind::default();
        let buffers = DraftsBuffers {
            gathered: bind.view(gathered),
            token_ids: bind.view(inputs.token_ids),
            positions: bind.view(inputs.positions),
            prefix_len: bind.view(inputs.prefix_len),
            token_batch: bind.view(inputs.token_batch),
            live_rows: bind.view(inputs.live_tokens),
        };
        let counts = self.own.counts(phase.capacity()?);
        one(bind.lower(prepared.as_ref(), &buffers, counts)?)
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::state::{Extent, Holder};
    use model_mimo26_contract::state::{Drafter, request_state};
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::super::tests::{any_gathers, phases, resolved};
    use super::*;
    use crate::contract::KV_FORMATS;
    use crate::ops::{Ops, Plan};
    use crate::program_lower::launches;
    use crate::ring::RingGeometry;
    use crate::sm120::own_ops::Sm120Own;

    /// A span of its own buffer of `bytes`.
    fn span(buffer: u32, bytes: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(buffer, 1),
            256,
            bytes,
        )
    }

    /// The ops of a `tp` instance of `slots` items with `switches`, whose
    /// prefix snapshots copy `prefix`.
    fn prepared(tp: Tp, switches: Switches, slots: u32, prefix: Option<RingPrefixCopy>) -> Ops {
        Ops::prepare(
            &*crate::sm120::backend(tp, switches, true).unwrap(),
            &Plan {
                tp,
                switches,
                kv: kernel_core::contract::DType::Bf16,
                items: slots,
                target: &crate::sm120::own_ops::Sm120Own::new(tp),
                state: &|_| [u64::MAX; 3],
                gathers: &any_gathers,
                dflash: None,
                own: OwnPlan {
                    prefix,
                    ..OwnPlan::new(&Sm120Own::new(tp), slots)
                },
            },
        )
        .unwrap()
    }

    /// The ring backups of every phase and bucket and the restore launch
    /// what the ring builders planned, at TP2 and TP4, for rings of 8, 32
    /// and 256 slots, with and without exact verify rows; a step of more
    /// items than the ring holds is refused by both, and so is a ring of
    /// other slots than the instance's.
    #[test]
    fn ring_backups_and_restores_launch_what_the_ring_builders_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
            for slots in [8, 32, 256] {
                let ring = RingGeometry::new(tp, slots);
                let state = span(1, 2 * ring.working);
                let tables = [span(2, 1024), span(3, 1024), span(4, 1024), span(5, 4)];
                for exact_verify in [false, true] {
                    let switches = Switches {
                        exact_verify,
                        ..Switches::configured()
                    };
                    let ops = prepared(tp, switches, slots, None);
                    let views = RingBackupViews {
                        ring_state: state,
                        slots,
                        slot_ids: tables[0],
                        q_len: tables[1],
                        prefix_len: tables[2],
                        live_batch: tables[3],
                    };
                    for (bucketed, bucket) in phases() {
                        let capacity = bucketed.capacity().unwrap();
                        let counts = [1, capacity / 2 + 1, capacity.min(slots), capacity];
                        for count in counts {
                            // Decode phases carry their live rows; verify and
                            // prefill steps back up their items.
                            let (phase, items) = match bucketed {
                                Phase::Decode(_) => (Phase::Decode(count), count),
                                _ => (bucketed, count),
                            };
                            let lowered = ops.ring_backup(phase, bucket, items, &views);
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
                            );
                            match (lowered, builder) {
                                (Ok(lowered), Ok(builder)) => {
                                    assert_eq!(
                                        resolved(&launches(vec![lowered])),
                                        resolved(&[builder]),
                                        "{tp:?} {slots} {phase:?} {items}"
                                    );
                                    compared += 1;
                                }
                                (Err(_), Err(_)) => assert!(items > slots),
                                (lowered, builder) => panic!(
                                    "{tp:?} {slots} {phase:?} {items}: {:?} {:?}",
                                    lowered.err(),
                                    builder.err()
                                ),
                            }
                        }
                    }
                    let other = RingBackupViews {
                        slots: slots * 2,
                        ..views
                    };
                    assert!(ops.ring_backup(Phase::Prefill, 0, 1, &other).is_err());
                    let views = RingRestoreViews {
                        ring_state: state,
                        slots,
                        slot_ids: tables[0],
                        starts: tables[1],
                        counts: tables[2],
                        live_items: tables[3],
                    };
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
                        assert_eq!(
                            resolved(&launches(vec![ops.ring_restore(items, &views).unwrap()])),
                            resolved(&[builder]),
                        );
                        compared += 1;
                    }
                    assert!(ops.ring_restore(slots + 1, &views).is_err());
                    let other = RingRestoreViews {
                        slots: slots / 2,
                        ..views
                    };
                    assert!(ops.ring_restore(1, &other).is_err());
                }
            }
        }
        assert!(compared > 500, "{compared}");
    }

    /// The prefix copies launch what the copy builder planned, capture and
    /// restore, for the request state of every drafter, at TP2 and TP4 and
    /// rings of 32 and 256 slots, at a slot other than the one the op was
    /// prepared from; another snapshot's copy and an instance without one
    /// are refused.
    #[test]
    fn prefix_copies_launch_what_the_copy_builder_planned() {
        let mut compared = 0;
        for tp in Tp::SUPPORTED {
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
                    let table = |slot| {
                        bundle_runtime::snapshot::Table::new(&state, extent, Holder::Slot(slot))
                            .unwrap()
                    };
                    let capture =
                        crate::prefix::copy_op(&table(0), 0, ring, Direction::Capture).unwrap();
                    let switches = Switches::configured();
                    let ops = prepared(tp, switches, slots, Some(capture));
                    let table = table(slot);
                    let views = crate::prefix::PrefixCopyViews {
                        ring_state: span(1, ring.working),
                        drafter: (capture.drafter_bytes > 0)
                            .then(|| span(2, capture.drafter_bytes)),
                        pool: span(3, 256 * table.page_bytes),
                        pages: span(4, 2048),
                    };
                    for (to_pages, direction) in
                        [(true, Direction::Capture), (false, Direction::Restore)]
                    {
                        let op = crate::prefix::copy_op(&table, slot, ring, direction).unwrap();
                        let builder = crate::prefix::prefix_copy(
                            &table,
                            slot,
                            table.pages(),
                            to_pages,
                            crate::prefix::PrefixCopyViews { ..views },
                            ring,
                        )
                        .unwrap();
                        assert_eq!(
                            resolved(&launches(vec![ops.prefix_copy(&op, slot, &views).unwrap()])),
                            resolved(&[builder]),
                            "{tp:?} {slots} {drafter:?} {direction:?}"
                        );
                        compared += 1;
                        let other = RingPrefixCopy {
                            pages: op.pages + 1,
                            ..op
                        };
                        assert!(ops.prefix_copy(&other, slot, &views).is_err());
                        assert!(ops.prefix_copy(&op, slots, &views).is_err());
                    }
                    let none = prepared(tp, switches, slots, None);
                    assert!(none.prefix_copy(&capture, slot, &views).is_err());
                }
            }
        }
        assert_eq!(compared, 20);
    }
}
