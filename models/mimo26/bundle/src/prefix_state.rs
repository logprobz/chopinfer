//! Prefix-cache capture and restore of a request's ring slot and drafter state.

use bundle_runtime::prefix::{Header, Stamp};
use bundle_runtime::snapshot::{
    Action, HeldBuffers, HostRankStreams, Pages, Table, agreed, copy_prefix,
};
use bundle_runtime::state::{Extent, Holder, RequestState};
use kernel_core::contract::Refusal;
use model_mimo26_contract::own_ops::ring::{Direction, RingPrefixCopy};
use runtime_bundle_abi::PrefixStateRequestV1;

use super::*;
use crate::prefix::{Drafter, KERNEL_BUFFERS, PrefixCopyViews, copy_op};
use crate::prefix_projected::HEADER;

/// The declared buffer the DFlash projected context lives in, which device
/// copies move.
const PROJECTED: usize = 2;

/// What an instance serving `items` holds state for: a slot per sequence,
/// in buffers of the largest bucket's entries.
pub(super) fn state_extent(items: Items) -> Extent {
    Extent {
        served: items.get(),
        bucket: items.capacity(),
        verify_rows: MAX_VERIFY_ROWS,
    }
}

impl Instance {
    fn prefix_page_bytes(&self) -> Result<u64, BundleAbiError> {
        let pool = self
            .kv_views
            .first()
            .ok_or(BundleAbiError::InvalidRange("MiMo serving KV pool"))?;
        if self.kv_page_capacity == 0 || !pool.length.is_multiple_of(self.kv_page_capacity) {
            return Err(BundleAbiError::InvalidRange("MiMo serving KV page size"));
        }
        Ok(pool.length / self.kv_page_capacity)
    }

    fn prefix_drafter(&self) -> Drafter {
        if self.dflash.is_some() {
            Drafter::Dflash
        } else if self.mtp.is_some() {
            Drafter::Mtp
        } else {
            Drafter::None
        }
    }

    /// What the instance keeps per request, as the contract declares it.
    pub(super) fn request_state(&self) -> RequestState {
        model_mimo26_contract::state::request_state(self.tp, &self.kv, self.prefix_drafter())
    }

    /// The capture of a slot's prefix snapshot the instance's ops copy, when
    /// its request state's pieces fit the copy kernel; a snapshot copy
    /// refuses as it would without it otherwise (`prefix::copy_op`).
    pub(super) fn prefix_copy_op(&self) -> Option<RingPrefixCopy> {
        let table = self.prefix_table(&self.request_state(), 0).ok()?;
        copy_op(&table, 0, self.ring, Direction::Capture).ok()
    }

    /// The table of `slot`'s snapshot.
    fn prefix_table(&self, state: &RequestState, slot: u32) -> Result<Table, BundleAbiError> {
        Table::new(state, state_extent(self.items), Holder::Slot(slot))
            .map_err(|_| BundleAbiError::InvalidRange("MiMo prefix snapshot table"))
    }

    pub(super) fn prefix_state_pages(&self) -> Result<u64, BundleAbiError> {
        let table = self.prefix_table(&self.request_state(), 0)?;
        if table.page_bytes != self.prefix_page_bytes()? {
            return Err(BundleAbiError::InvalidRange("MiMo serving KV page size"));
        }
        Ok(table.pages())
    }

    /// Where `rank` holds each declared buffer the snapshot copies: the
    /// working half of the ring (the shadow belongs to open transactions),
    /// the drafter's history and the DFlash projected context.
    fn prefix_held(&self, rank: usize) -> HeldBuffers {
        let state = self.state_views[rank];
        let ring = DeviceBufferSpanV1::new(state.buffer, state.offset, state.length / 2);
        let history = self
            .mtp_prefix_span(rank)
            .or_else(|| self.dflash_prefix_span(rank));
        HeldBuffers {
            buffers: vec![Some(ring), history, self.dflash_projected_span(rank)],
            spares: Vec::new(),
        }
    }

    /// One rank's snapshot copy: the page table upload and, for a DFlash
    /// capture, the header's into the rank's metadata span; the copy kernel
    /// over the ring and drafter history; device copies of the projected
    /// context and of the header between the span and the pages; and, for a
    /// DFlash restore, the span's read back.
    fn prefix_actions(
        &self,
        rank: usize,
        table: &Table,
        slot: u32,
        pages: &[u64],
        to_pages: bool,
        header: Option<&[u8]>,
    ) -> Result<Vec<Action<crate::pipeline::PreparedLaunch>>, BundleAbiError> {
        let invalid = |refusal: Refusal| BundleAbiError::InvalidRange(refusal.0);
        let buffer = self.restore_inputs[rank];
        let layout = RestoreLayout::new(self.items);
        let ids = pages
            .iter()
            .flat_map(|page| page.to_le_bytes())
            .collect::<Vec<_>>();
        let table_span = DeviceBufferSpanV1::new(
            buffer.buffer,
            buffer.offset + layout.prefix_pages(),
            ids.len() as u64,
        );
        let metadata = DeviceBufferSpanV1::new(
            buffer.buffer,
            buffer.offset + layout.prefix_metadata(),
            table.header_bytes,
        );
        let held = self.prefix_held(rank);
        let paged = Pages {
            pool: self.kv_views[rank],
            ids: pages,
            page_bytes: self.prefix_page_bytes()?,
        };
        paged.check(table, self.kv_page_capacity).map_err(invalid)?;
        let drafter = table
            .descriptors
            .iter()
            .find(|descriptor| descriptor.buffer == KERNEL_BUFFERS.end - 1)
            .map(|descriptor| {
                let base = held.buffers[descriptor.buffer]
                    .ok_or(BundleAbiError::InvalidRange("MiMo drafter prefix state"))?;
                if descriptor
                    .offset
                    .checked_add(descriptor.length)
                    .is_none_or(|end| end > base.length)
                {
                    return Err(BundleAbiError::InvalidRange("MiMo prefix state range"));
                }
                Ok(DeviceBufferSpanV1::new(
                    base.buffer,
                    base.offset + descriptor.offset,
                    descriptor.length,
                ))
            })
            .transpose()?;
        let direction = if to_pages {
            Direction::Capture
        } else {
            Direction::Restore
        };
        let op = copy_op(table, slot, self.ring, direction).map_err(invalid)?;
        if op.pages != pages.len() as u64 {
            return Err(BundleAbiError::InvalidRange(
                "MiMo prefix snapshot geometry",
            ));
        }
        let launch = self
            .ops
            .prefix_copy(
                &op,
                slot,
                &PrefixCopyViews {
                    ring_state: held.buffers[0].expect("the ring is held"),
                    drafter,
                    pool: self.kv_views[rank],
                    pages: table_span,
                },
            )
            .map_err(invalid)?
            .launch;
        let mut actions = vec![Action::Upload {
            to: table_span,
            bytes: ids,
        }];
        if let Some(bytes) = header {
            actions.push(Action::Upload {
                to: metadata,
                bytes: bytes.to_vec(),
            });
        }
        actions.push(Action::Launch(launch));
        if table.header_bytes > 0 {
            let copies = table
                .copies(PROJECTED..PROJECTED + 1, &held, &paged, to_pages)
                .map_err(invalid)?
                .into_iter()
                .chain(
                    paged
                        .copies(metadata, table.header_at(), to_pages)
                        .map_err(invalid)?,
                );
            actions.extend(copies.map(|(from, to)| Action::Copy { from, to }));
            if !to_pages {
                actions.push(Action::Read { from: metadata });
            }
        }
        Ok(actions)
    }

    /// Copies `slot`'s state to the snapshot pages, or back, on every rank,
    /// and returns once the copies are complete, with the DFlash header a
    /// restore read back.
    ///
    /// Each rank uploads the page table and launches one copy kernel on its
    /// stream, behind the step that wrote the slot: a pipelined prefill's
    /// compute stream already waits for its all-reduce stream before the step
    /// ends. The host reuses snapshot pages on its own stream, so every rank
    /// is synchronized once, after all ranks launched (`snapshot::copy_prefix`).
    fn copy_prefix(
        &self,
        slot: u32,
        pages: &[u64],
        to_pages: bool,
        header: Option<Header>,
    ) -> Result<Option<Vec<u8>>, BundleAbiError> {
        let invalid = |refusal: Refusal| BundleAbiError::InvalidRange(refusal.0);
        if pages.iter().any(|page| *page >= self.kv_page_capacity)
            || pages.len() as u64 > PREFIX_PAGES_BYTES / 8
        {
            return Err(BundleAbiError::InvalidRange("MiMo prefix snapshot page"));
        }
        if (self.dflash.is_some() && to_pages) != header.is_some() {
            return Err(BundleAbiError::InvalidRange(
                "missing DFlash snapshot header",
            ));
        }
        let stamp = Stamp {
            ranks: self.tp.ranks().into(),
            digest: 0,
        };
        let encoded = header
            .map(|header| header.encode(&HEADER, stamp))
            .transpose()
            .map_err(invalid)?;
        let state = self.request_state();
        let table = self.prefix_table(&state, slot)?;
        if (0..self.tp.count()).any(|rank| {
            self.dflash_projected_span(rank).is_some_and(|cache| {
                cache.length != state.buffers[PROJECTED].bytes(state_extent(self.items))
            })
        }) {
            return Err(BundleAbiError::InvalidRange(
                "DFlash projected cache slot count",
            ));
        }
        let plans = (0..self.tp.count())
            .map(|rank| {
                self.prefix_actions(rank, &table, slot, pages, to_pages, encoded.as_deref())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let launch = |rank: usize, stream, launch| {
            crate::pipeline::launch_all(&self.services, stream, &self.symbols[rank], &[launch])
                .map_err(|_| BundleAbiError::InvalidRange("MiMo prefix snapshot launch"))
        };
        let streams = HostRankStreams {
            services: &self.services,
            streams: &self.streams[..self.tp.count()],
            launch: &launch,
            host_error: &std::convert::identity,
            refusal: &|refusal: Refusal| BundleAbiError::InvalidRange(refusal.0),
            // Unconfirmed work poisons the client and keeps pages from reuse.
            unconfirmed: &|| BundleAbiError::Status(BundleStatusV1::InternalError),
        };
        let reads = copy_prefix(&streams, plans)?;
        if reads.iter().all(Vec::is_empty) {
            return Ok(None);
        }
        agreed(&reads)
            .map(<[u8]>::to_vec)
            .map(Some)
            .ok_or(BundleAbiError::InvalidRange(
                "DFlash snapshot headers disagree across ranks",
            ))
    }

    /// Writes a request's committed ring slot and retained drafter rows into pages.
    ///
    /// Capture follows a prefill, decode or verify commit. The ring holds
    /// the committed window after rejected verify rows have been restored;
    /// each drafter must cover the same committed boundary.
    pub(super) fn capture_prefix(
        &mut self,
        request: &PrefixStateRequestV1,
    ) -> Result<(), BundleStatusV1> {
        if self.ledger.step().is_some() || !request.token_ids.is_empty() {
            return Err(BundleStatusV1::InvalidState);
        }
        let slot = match self.ledger.committed(request.request_id, request.tokens) {
            Ok(slot) => slot,
            Err(LedgerError::UnknownRequest) => return Err(BundleStatusV1::NotFound),
            Err(_) => return Err(BundleStatusV1::InvalidState),
        };
        if request.tokens == 0
            || !self.mtp_prefix_ready(request.request_id, slot, request.tokens)
            || !self.dflash_prefix_ready(request.request_id, slot, request.tokens)
        {
            return Err(BundleStatusV1::InvalidState);
        }
        self.copy_prefix(
            slot,
            &request.pages,
            true,
            self.dflash_snapshot_header(request.request_id, request.tokens),
        )
        .map(|_| ())
        .map_err(status)
    }

    /// Gives an unseen request a ring slot filled from a snapshot, as if it
    /// had committed `token_ids`.
    pub(super) fn restore_prefix(
        &mut self,
        request: &PrefixStateRequestV1,
    ) -> Result<(), BundleStatusV1> {
        if request.token_ids.is_empty() {
            return Err(BundleStatusV1::InvalidArgument);
        }
        if self.ledger.step().is_some() || request.token_ids.len() as u64 != request.tokens {
            return Err(BundleStatusV1::InvalidState);
        }
        let slot = self
            .ledger
            .restore(request.request_id, request.tokens, None)
            .map_err(ledger_status)?;
        let restored = self
            .copy_prefix(slot, &request.pages, false, None)
            .and_then(|metadata| match self.prefix_drafter() {
                Drafter::Mtp => {
                    self.mtp_prefix_restore(request.request_id, slot, &request.token_ids)
                }
                Drafter::Dflash => {
                    let bytes = metadata.ok_or(BundleAbiError::InvalidRange(
                        "missing DFlash restored header",
                    ))?;
                    let stamp = Stamp {
                        ranks: self.tp.ranks().into(),
                        digest: 0,
                    };
                    let header = Header::decode(&bytes, &HEADER, stamp, request.tokens)
                        .map_err(|e| BundleAbiError::InvalidRange(e.0))?;
                    self.dflash_prefix_restore(
                        request.request_id,
                        slot,
                        request.tokens,
                        header.cursor,
                    )
                }
                Drafter::None => Ok(()),
            });
        if let Err(error) = restored {
            // The slot is refilled by its next owner's prefill or restore.
            let _ = self.ledger.retire(request.request_id);
            return Err(status(error));
        }
        Ok(())
    }
}
