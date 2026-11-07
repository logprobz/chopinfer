//! Prefix-cache capture and restore of a request's window rings and
//! compressor tails: the slot's committed entries of every captured buffer,
//! copied piece by piece between the slot and the snapshot's serving pages
//! with device copies (`bundle_runtime::snapshot`), each rank its own
//! copy, and the shared header (`HeaderLayout::V1`) on rank 0. Under the
//! `prefix_snapshot_copy_kernel` candidate each rank instead uploads the
//! snapshot's page and segment table and launches the shared snapshot copy
//! kernel once ([`SnapshotKernel`]), with the same checks, header actions
//! and drain. The KV
//! sources' compressed latents and index keys live in the shared pages
//! already. A ring holds its last window of positions; a tail the open
//! group's row. The Engram history needs no pages: a restore names every
//! prefix token, and the history takes the last of them. A capture at an
//! anchor kept inside a prompt-ending item copies that anchor's spares
//! instead of a slot (`execute::anchors`), DSpark's raw feature history
//! included; a restore rebuilds the drafter's rings from those rows.

use bundle_runtime::prefix::{
    Header, HeaderLayout, PrefixInstance, Stamp, layout_digest, may_restore,
};
use bundle_runtime::resources::status;
use bundle_runtime::snapshot::{
    Action, BoundLaunch, HeaderAction, HeldBuffers, HostRankStreams, KernelCopy, Pages, Table,
    copy_prefix,
};
use bundle_runtime::state::{Extent, Holder, RequestState};
use kernel_core::backend::{Backend, Need, Prepared, View};
use kernel_core::snapshot_copy::{Direction, SnapshotCopy, SnapshotCopyBuffers};
use runtime_bundle_abi::{
    BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1, PrefixStateRequestV1,
};

use super::Instance;

/// The shared header: the tokens, the capturing request, the rank count
/// and the declaration's layout digest.
const HEADER: HeaderLayout = HeaderLayout::V1;

/// The shared snapshot copy kernel of an instance's snapshots: one op a
/// direction, prepared for the snapshot's pages, the views its pieces name
/// and the most pieces a slot's or an anchor item's table holds, and each
/// rank's table buffer.
pub(crate) struct SnapshotKernel {
    capture: (SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>),
    restore: (SnapshotCopy, Prepared<SnapshotCopyBuffers<View>>),
    table_bytes: u64,
    /// Each rank's table buffer, which acquire allocates.
    pub(super) tables: Vec<DeviceBufferSpanV1>,
}

impl SnapshotKernel {
    /// The kernel of `state`'s snapshots in an instance of `extent`, on
    /// `backend`; why not, when the kernel cannot serve a slot's or an
    /// anchor item's table, so the instance keeps the device copies.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        state: &RequestState,
        extent: Extent,
    ) -> Result<Self, String> {
        HEADER
            .check(state)
            .map_err(|refusal| refusal.0.to_owned())?;
        let anchored = state.buffers.iter().any(|buffer| buffer.spare_bytes() > 0);
        let holders = [Some(Holder::Slot(0)), anchored.then_some(Holder::Anchor(0))];
        let tables = holders
            .into_iter()
            .flatten()
            .map(|holder| Table::new(state, extent, holder).map_err(|error| format!("{error:?}")))
            .collect::<Result<Vec<_>, _>>()?;
        // Each declared buffer, or each spare, a piece names is one view.
        let views = |table: &Table| {
            table
                .descriptors
                .iter()
                .map(|descriptor| (descriptor.buffer, descriptor.spare))
                .collect::<std::collections::BTreeSet<_>>()
                .len()
        };
        let slot = &tables[0];
        if tables
            .iter()
            .any(|table| views(table) != views(slot) || table.pages() != slot.pages())
        {
            return Err("slot and anchor snapshots differ in views or pages".into());
        }
        let segments = tables
            .iter()
            .map(|table| table.descriptors.len())
            .max()
            .unwrap_or(0);
        let segments = u32::try_from(segments).map_err(|_| "too many snapshot pieces")?;
        let op = |direction| -> Result<SnapshotCopy, String> {
            Ok(SnapshotCopy {
                direction,
                views: u32::try_from(views(slot)).map_err(|_| "too many snapshot views")?,
                page_bytes: slot.page_bytes,
                pages: u32::try_from(slot.pages()).map_err(|_| "too many snapshot pages")?,
            })
        };
        let prepare = |direction| -> Result<_, String> {
            let op = op(direction)?;
            let prepared = backend
                .snapshot_copy(&op, Need::rows(segments))
                .map_err(|refused| refused.to_string())?;
            Ok((op, prepared))
        };
        let capture = prepare(Direction::ToPages)?;
        let restore = prepare(Direction::FromPages)?;
        let table_bytes = capture
            .0
            .table_bytes(segments)
            .map_err(|refusal| refusal.0.to_owned())?;
        Ok(Self {
            capture,
            restore,
            table_bytes,
            tables: Vec::new(),
        })
    }

    /// Bytes of one rank's table buffer.
    pub(crate) fn table_bytes(&self) -> u64 {
        self.table_bytes
    }
}

impl Instance {
    /// What both sides of a snapshot must agree on.
    fn stamp(&self) -> Stamp {
        Stamp {
            ranks: u64::from(self.config.shape.ranks()),
            digest: layout_digest(&self.state),
        }
    }

    /// The table of `holder`'s snapshot: a slot's, or an anchor item's
    /// spares.
    fn snapshot_table(&self, holder: Holder) -> Result<Table, BundleStatusV1> {
        HEADER
            .check(&self.state)
            .map_err(|_| BundleStatusV1::InternalError)?;
        Table::new(&self.state, self.extent, holder).map_err(|_| BundleStatusV1::InternalError)
    }

    /// Copies `holder`'s state to the snapshot `pages` (with `header`) or
    /// back into a slot on every rank, and returns once every rank's
    /// copies are done, with the header bytes rank 0 read on a restore.
    /// Every rank's copies are planned before any rank starts; a drain that
    /// cannot be confirmed is INTERNAL_ERROR, which keeps the host from
    /// reusing the pages.
    fn copy_snapshot(
        &self,
        holder: Holder,
        pages: &[u64],
        header: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, BundleStatusV1> {
        let table = self.snapshot_table(holder)?;
        let to_pages = header.is_some();
        // Rank `rank`'s held state, its view of the snapshot's pages (once
        // they are checked) and its header action.
        let rank_of = |rank: usize, buffers: &crate::buffers::RankBuffers| {
            let held = HeldBuffers {
                buffers: buffers
                    .state
                    .buffers
                    .iter()
                    .map(|&(_, span)| Some(span))
                    .collect(),
                spares: self.anchor_spares.get(rank).map_or_else(
                    || vec![None; buffers.state.buffers.len()],
                    |spares| spares.buffers.clone(),
                ),
            };
            let paged = Pages {
                pool: buffers.kv_pool,
                ids: pages,
                page_bytes: self.config.page_bytes,
            };
            paged
                .check(&table, self.kv_pages)
                .map_err(|_| BundleStatusV1::InvalidArgument)?;
            let action = match (rank, &header) {
                (0, Some(bytes)) => HeaderAction::Write(bytes.clone()),
                (0, None) => HeaderAction::Read,
                _ => HeaderAction::None,
            };
            Ok((held, paged, action))
        };
        let Some(kernel) = &self.snapshot_kernel else {
            let plans = self
                .buffers
                .iter()
                .enumerate()
                .map(|(rank, buffers)| {
                    let (held, paged, action) = rank_of(rank, buffers)?;
                    table
                        .actions::<()>(&held, &paged, to_pages, action)
                        .map_err(|_| BundleStatusV1::InternalError)
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.submit_snapshot(plans, &|_, _, ()| Ok(()));
        };
        let (op, prepared) = if to_pages {
            &kernel.capture
        } else {
            &kernel.restore
        };
        let plans = self
            .buffers
            .iter()
            .enumerate()
            .map(|(rank, buffers)| {
                let (held, paged, action) = rank_of(rank, buffers)?;
                let copy = KernelCopy {
                    op: *op,
                    prepared: prepared.as_ref(),
                    table: *kernel
                        .tables
                        .get(rank)
                        .ok_or(BundleStatusV1::InternalError)?,
                };
                table
                    .kernel_actions(&held, &paged, to_pages, action, &copy)
                    .map_err(|refusal| {
                        self.log(&format!("prefix snapshot copy kernel: {}", refusal.0));
                        BundleStatusV1::InternalError
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.submit_snapshot(plans, &|rank, stream, launch: BoundLaunch| {
            launch
                .submit(&self.services, stream, &self.symbols[rank])
                .map_err(bundle_runtime::dispatch::status)
        })
    }

    /// Submits every rank's snapshot `plans`, launching through `launch`,
    /// and returns once every rank that received work is drained, with the
    /// bytes rank 0 read. A drain that cannot be confirmed is
    /// INTERNAL_ERROR.
    fn submit_snapshot<L>(
        &self,
        plans: Vec<Vec<Action<L>>>,
        launch: &dyn Fn(usize, BundleStreamHandleV1, L) -> Result<(), BundleStatusV1>,
    ) -> Result<Vec<u8>, BundleStatusV1> {
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        let rank_streams = HostRankStreams {
            services: &self.services,
            streams: &streams,
            launch,
            host_error: &status,
            refusal: &|_| BundleStatusV1::InternalError,
            unconfirmed: &|| BundleStatusV1::InternalError,
        };
        let mut reads = copy_prefix(&rank_streams, plans)?;
        Ok(std::mem::take(&mut reads[0]))
    }
}

impl PrefixInstance for Instance {
    fn state_pages(&self) -> Result<u64, BundleStatusV1> {
        Ok(self.snapshot_table(Holder::Slot(0))?.pages())
    }

    /// Writes a request's committed state into snapshot pages. The host
    /// captures after a commit, so no step is open.
    fn capture(&mut self, request: &PrefixStateRequestV1) -> Result<(), BundleStatusV1> {
        if !request.token_ids.is_empty() || request.tokens == 0 {
            return Err(BundleStatusV1::InvalidState);
        }
        // A capture at an anchor the last step kept inside a prompt-ending
        // item copies that anchor item's spares (`execute::anchors`).
        let holder = match self.ledger.anchor(request.request_id, request.tokens) {
            Ok(anchor) => Holder::Anchor(anchor),
            Err(_) => Holder::Slot(
                self.ledger
                    .committed(request.request_id, request.tokens)
                    .map_err(|_| BundleStatusV1::InvalidState)?,
            ),
        };
        self.dspark_catch_up(request.request_id)?;
        if let Some(device) = &self.dspark {
            let snapshot = device
                .runtime
                .as_ref()
                .ok_or(BundleStatusV1::InvalidState)?
                .state()
                .snapshot(request.request_id)
                .map_err(|_| BundleStatusV1::InvalidState)?;
            // The drafter keeps only raw rows in a snapshot, and an anchor's
            // spare holds those of its anchor, inside the slot's history.
            let kept = match holder {
                Holder::Slot(_) => snapshot.end == request.tokens,
                Holder::Anchor(_) => request.tokens < snapshot.end,
            };
            if !kept {
                return Err(BundleStatusV1::InvalidState);
            }
        }
        let header = Header {
            tokens: request.tokens,
            frontier: None,
            owner: Some(request.request_id),
            cursor: None,
        }
        .encode(&HEADER, self.stamp())
        .map_err(|_| BundleStatusV1::InternalError)?;
        self.copy_snapshot(holder, &request.pages, Some(header))?;
        #[cfg(feature = "diagnostic-taps")]
        self.capture_profile_boundary(request.request_id, request.tokens, "snapshot")
            .map_err(|error| {
                self.log(&error);
                BundleStatusV1::HostError
            })?;
        Ok(())
    }

    /// Gives an unseen request the host's state slot, filled from a
    /// snapshot, as if it had committed `token_ids`. A failed copy or a
    /// header that does not match leaves the slot free.
    fn restore(
        &mut self,
        request: &PrefixStateRequestV1,
        slot: Option<u64>,
    ) -> Result<(), BundleStatusV1> {
        if request.token_ids.is_empty() || request.token_ids.len() as u64 != request.tokens {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // The host places every request: a restore names its slot.
        let slot = slot
            .and_then(|slot| u32::try_from(slot).ok())
            .ok_or(BundleStatusV1::InvalidArgument)?;
        self.ledger
            .check_restore(request.request_id, request.tokens, Some(slot))
            .map_err(|_| BundleStatusV1::InvalidState)?;
        self.dspark_validate_restore(request.request_id, slot, request.tokens)?;
        let read = self
            .copy_snapshot(Holder::Slot(slot), &request.pages, None)
            .inspect_err(|_| {
                self.dspark_poison();
            })?;
        let header = Header::decode(&read, &HEADER, self.stamp(), request.tokens)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !may_restore(&self.state, &header, request.request_id) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // The Engram history: the prefix's last tokens.
        if self.dspark.is_some() {
            self.dspark_restored(request.request_id, slot, &request.token_ids)
                .inspect_err(|_| {
                    self.dspark_poison();
                })?;
        } else if let Some(engram) = &mut self.engram {
            engram
                .restore(request.request_id, &request.token_ids)
                .map_err(|error| {
                    self.log(&format!("Engram restore: {error}"));
                    BundleStatusV1::InvalidArgument
                })?;
        }
        let restored = self
            .ledger
            .restore(request.request_id, request.tokens, Some(slot))
            .map(|_| ())
            .map_err(|_| BundleStatusV1::InternalError);
        // The KV-state taps read the restored state before its first step.
        #[cfg(feature = "diagnostic-taps")]
        if restored.is_ok() {
            self.taps.restored(request.request_id);
        }
        if restored.is_err() {
            self.dspark_forget(request.request_id);
            if let Some(engram) = &mut self.engram {
                engram.forget(request.request_id);
            }
        }
        restored
    }
}
