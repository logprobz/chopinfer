//! Prefix-cache capture and restore of a request's linear, PLE, ring and
//! drafter state.

use bundle_runtime::prefix::{Header, Stamp, may_restore};
use bundle_runtime::snapshot::{Action, HeaderAction, HostRankStreams, Pages, Table, copy_prefix};
use bundle_runtime::state::{Extent, Holder, RequestState};
use runtime_bundle_abi::PrefixStateRequestV1;

use super::*;
use model_qwen38_contract::own_ops::copies::{Direction, PrefixCopy, PrefixCopyBuffers};

use crate::prefix::{HEADER, Held, PrefixCopyViews, check_copy, prefix_copy_op};

/// The header layout holds neither the rank count nor a digest.
const STAMP: Stamp = Stamp {
    ranks: 0,
    digest: 0,
};

impl Instance {
    /// What the instance keeps per request, as the contract declares it.
    fn request_state(&self) -> RequestState {
        model_qwen38_contract::state::request_state(
            self.shape,
            &self.layout.kv,
            self.speculation.drafter,
        )
    }

    /// What the instance holds state for: a slot per served sequence, a
    /// ring of the largest bucket's items, and the verify rows the host
    /// enabled.
    fn state_extent(&self) -> Extent {
        Extent {
            served: self.served.slots,
            bucket: self.ring().slots as u32,
            verify_rows: self.speculation.verify_rows,
        }
    }

    /// The table of `holder`'s snapshot.
    fn prefix_table(&self, state: &RequestState, holder: Holder) -> Result<Table, BundleStatusV1> {
        let mapped = match holder {
            Holder::Slot(slot) => vec![
                (
                    crate::prefix::CONV,
                    self.linear_entries.committed(slot) as u64,
                ),
                (
                    crate::prefix::DELTA,
                    self.linear_entries.committed(slot) as u64,
                ),
                (crate::prefix::PLE, self.ple_entries.committed(slot) as u64),
            ],
            Holder::Anchor(_) => Vec::new(),
        };
        Table::with_entries(state, self.state_extent(), holder, &mapped)
            .map_err(|_| BundleStatusV1::InternalError)
    }

    /// What `holder`'s snapshot copies hold, with its committed linear and
    /// PLE entries.
    fn held(&self, holder: Holder, linear_entry: u64, ple_entry: u64) -> Held {
        Held::of(
            holder,
            self.state_extent(),
            self.speculation,
            self.layout.kv_layers,
            linear_entry,
            ple_entry,
        )
    }

    /// The snapshot copies of a slot's and an anchor item's state, each
    /// way (restore, capture), or `None` where the instance's state has no
    /// snapshot table.
    pub(super) fn prefix_copies(&self) -> Option<[[PrefixCopy; 2]; 2]> {
        let state = self.request_state();
        let copies = |holder| -> Option<[PrefixCopy; 2]> {
            let table = self.prefix_table(&state, holder).ok()?;
            let held = self.held(holder, 0, 0);
            Some(
                [Direction::Restore, Direction::Capture]
                    .map(|direction| prefix_copy_op(self.shape, &table, held, direction)),
            )
        };
        Some([copies(Holder::Slot(0))?, copies(Holder::Anchor(0))?])
    }

    pub(super) fn prefix_state_pages(&self) -> Result<u64, BundleStatusV1> {
        let pool = self.kv_views.first().ok_or(BundleStatusV1::InvalidState)?;
        if Some(pool.length) != self.kv_page_capacity.checked_mul(self.layout.kv.page_bytes) {
            return Err(BundleStatusV1::InvalidState);
        }
        Ok(self
            .prefix_table(&self.request_state(), Holder::Slot(0))?
            .pages())
    }

    /// Where `rank`'s copy kernel reads and writes `holder`'s state: a
    /// slot's pools, PLE entries, live ring and retained rows, or the
    /// anchor items' spares.
    fn prefix_views(&self, rank: usize, holder: Holder, pages: u64) -> PrefixCopyViews {
        let scratch = &self.scratch[rank];
        let pages = DeviceBufferSpanV1::new(
            scratch.prefix_pages.buffer,
            scratch.prefix_pages.offset,
            pages * 8,
        );
        let (ple_pool, ring, retained) = match holder {
            Holder::Slot(_) => (
                scratch.ple_pool,
                // Only the live half; the saved half belongs to open steps.
                self.live_ring(rank),
                scratch.drafter.map(|drafter| drafter.retained),
            ),
            Holder::Anchor(_) => (
                scratch.anchor_ple,
                // The first layers: the ring's.
                DeviceBufferSpanV1::new(
                    scratch.anchor_ring.buffer,
                    scratch.anchor_ring.offset,
                    Ring {
                        shape: self.shape,
                        layers: self.layout.kv_layers,
                        slots: u64::from(ANCHOR_ITEMS),
                    }
                    .bytes(),
                ),
                scratch.drafter.map(|drafter| drafter.anchor_retained),
            ),
        };
        PrefixCopyViews {
            conv_pool: scratch.conv_pool,
            delta_pool: scratch.delta_pool,
            ple_pool,
            ring,
            retained,
            pool: self.kv_views[rank],
            pages,
        }
    }

    /// Copies `holder`'s state to the snapshot pages, or back, on every
    /// rank, with the header on rank 0, and returns once the copies are
    /// complete. A capture writes `header`; a restore returns the header
    /// bytes it read.
    ///
    /// Each rank uploads the page table and launches one copy kernel on its
    /// stream, behind the commit that wrote the slot. The host reuses the
    /// pages on its own stream as soon as this returns, even when it fails:
    /// every launch is built before any rank submits, and every rank that
    /// received work is drained on every exit (`snapshot::copy_prefix`).
    /// When a drain fails, a copy may still access the pages, so that is
    /// INTERNAL_ERROR, which poisons the instance and keeps the host from
    /// reusing them.
    fn copy_prefix(
        &self,
        state: &RequestState,
        holder: Holder,
        pages: &[u64],
        header: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, BundleStatusV1> {
        let table = self.prefix_table(state, holder)?;
        let held = self.held(
            holder,
            match holder {
                Holder::Slot(slot) => self.linear_entries.committed(slot) as u64,
                Holder::Anchor(_) => 0,
            },
            match holder {
                Holder::Slot(slot) => self.ple_entries.committed(slot) as u64,
                Holder::Anchor(_) => 0,
            },
        );
        let to_pages = header.is_some();
        let direction = if to_pages {
            Direction::Capture
        } else {
            Direction::Restore
        };
        let copy = self
            .copies
            .as_ref()
            .ok_or(BundleStatusV1::InternalError)?
            .prefix(
                matches!(holder, Holder::Anchor(_)),
                &prefix_copy_op(self.shape, &table, held, direction),
            )
            .map_err(|_| BundleStatusV1::InternalError)?;
        let ids = pages
            .iter()
            .flat_map(|page| page.to_le_bytes())
            .collect::<Vec<_>>();
        let plans = (0..self.shape.ranks.count())
            .map(|rank| {
                let paged = Pages {
                    pool: self.kv_views[rank],
                    ids: pages,
                    page_bytes: self.layout.kv.page_bytes,
                };
                paged
                    .check(&table, self.kv_page_capacity)
                    .map_err(|_| BundleStatusV1::InvalidArgument)?;
                let views = self.prefix_views(rank, holder, pages.len() as u64);
                let table_span = views.pages;
                let launch = check_copy(self.shape, &table, held, &views).and_then(|()| {
                    let mut bind = crate::ops::Bind::default();
                    let buffers = PrefixCopyBuffers {
                        conv_pool: bind.view(views.conv_pool),
                        delta_pool: bind.view(views.delta_pool),
                        ple_pool: bind.view(views.ple_pool),
                        ring: bind.view(views.ring),
                        retained: views.retained.map(|retained| bind.view(retained)),
                        pool: bind.view(views.pool),
                        pages: bind.view(views.pages),
                        slot: held.slot,
                        index: held.index as u32,
                        ple_entry: held.ple_index as u32,
                    };
                    bind.launch(copy, &buffers, kernel_core::backend::Counts::rows(1))
                });
                let mut launch = launch.map_err(|_| BundleStatusV1::InternalError)?;
                let launch = match launch.len() {
                    1 => launch.remove(0),
                    _ => return Err(BundleStatusV1::InternalError),
                };
                let header = match (rank, &header) {
                    (0, Some(bytes)) => HeaderAction::Write(bytes.clone()),
                    (0, None) => HeaderAction::Read,
                    _ => HeaderAction::None,
                };
                let mut actions = vec![
                    Action::Upload {
                        to: table_span,
                        bytes: ids.clone(),
                    },
                    Action::Launch(launch),
                ];
                actions.extend(
                    paged
                        .header_actions(&table, header)
                        .map_err(|_| BundleStatusV1::InternalError)?,
                );
                Ok(actions)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let streams = HostRankStreams {
            services: &self.services,
            streams: &self.streams,
            launch: &|rank, stream, launch| {
                crate::pipeline::launch_all(&self.services, stream, &self.symbols[rank], &[launch])
                    .map_err(crate::dispatch::status)
            },
            host_error: &status,
            refusal: &|_| BundleStatusV1::InternalError,
            // An unconfirmed drain keeps the pages and poisons the instance.
            unconfirmed: &|| BundleStatusV1::InternalError,
        };
        let mut reads = copy_prefix(&streams, plans)?;
        Ok(std::mem::take(&mut reads[0]))
    }

    /// Writes a request's committed state into snapshot pages.
    ///
    /// The host captures after a commit (a prefill chunk, or the step before
    /// a preemption), so no step is open, the live ring is the committed one
    /// and a verify step's rejected ring rows are rewritten before any row
    /// reads them. A capture at the anchor the last step kept inside a
    /// prompt-ending item copies that anchor item's spares instead: its
    /// linear entries, PLE entry and ring, and with the drafter the X of the
    /// row before the anchor, which is the drafter's frontier there.
    pub(super) fn capture_prefix(
        &mut self,
        request: &PrefixStateRequestV1,
    ) -> Result<(), BundleStatusV1> {
        if !request.token_ids.is_empty() || request.tokens == 0 {
            return Err(BundleStatusV1::InvalidState);
        }
        let (holder, frontier) =
            if let Ok(anchor) = self.ledger.anchor(request.request_id, request.tokens) {
                (
                    Holder::Anchor(anchor),
                    self.drafter.as_ref().map(|_| request.tokens - 1),
                )
            } else {
                let slot = self
                    .ledger
                    .committed(request.request_id, request.tokens)
                    .map_err(|_| BundleStatusV1::InvalidState)?;
                let frontier = self
                    .drafter
                    .as_ref()
                    .map(|drafter| drafter.state.frontier(slot));
                (Holder::Slot(slot), frontier)
            };
        let header = Header {
            tokens: request.tokens,
            frontier,
            owner: Some(request.request_id),
            cursor: None,
        }
        .encode(&HEADER, STAMP)
        .map_err(|_| BundleStatusV1::InternalError)?;
        let state = self.request_state();
        self.copy_prefix(&state, holder, &request.pages, Some(header))
            .map(|_| ())
    }

    /// Gives an unseen request the host's `state_slot`, filled from a
    /// snapshot, as if it had committed `token_ids`. A failed copy or a
    /// snapshot that does not match leaves the slot free; its next owner
    /// starts fresh or restores it whole.
    ///
    /// The drafter's K, V and block keys live in the shared pages, and a
    /// continuation first runs the rows from the frontier on: row `tokens -
    /// 1` at least, which reads the continuation's own next token. The
    /// package declares that steps rewrite rows of the last page
    /// (`rewrites_last_page` in its batch limits), so the host copies the
    /// page holding the anchor's last row before an append. A frontier that
    /// lags into an earlier page (a decode or verify boundary) would write a
    /// full page an anchor shares with every hit, so such a snapshot is
    /// restored only by the request that captured it: a preemption's
    /// private anchor (`prefix::may_restore`).
    pub(super) fn restore_prefix(
        &mut self,
        request: &PrefixStateRequestV1,
        state_slot: u64,
    ) -> Result<(), BundleStatusV1> {
        if request.token_ids.is_empty() || request.token_ids.len() as u64 != request.tokens {
            return Err(BundleStatusV1::InvalidArgument);
        }
        let slot = u32::try_from(state_slot).map_err(|_| BundleStatusV1::InvalidArgument)?;
        self.ledger
            .check_restore(request.request_id, request.tokens, Some(slot))
            .map_err(|_| BundleStatusV1::InvalidState)?;
        let state = self.request_state();
        let read = self.copy_prefix(&state, Holder::Slot(slot), &request.pages, None)?;
        let header = Header::decode(&read, &HEADER, STAMP, request.tokens)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !may_restore(&state, &header, request.request_id) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if let (Some(drafter), Some(frontier)) = (self.drafter.as_mut(), header.frontier) {
            drafter
                .state
                .restore(slot, &request.token_ids, frontier)
                .map_err(|_| BundleStatusV1::InvalidArgument)?;
        }
        self.ledger
            .restore(request.request_id, request.tokens, Some(slot))
            .map(|_| ())
            .map_err(|_| BundleStatusV1::InternalError)
    }
}
