//! Prefix-cache capture and restore of a request's window rings and
//! compressor tails: the slot's committed entries of every captured buffer,
//! copied piece by piece between the slot and the snapshot's serving pages
//! with device copies (`bundle_runtime::snapshot`), each rank its own
//! copy, and the shared header (`HeaderLayout::V1`) on rank 0. The KV
//! sources' compressed latents and index keys live in the shared pages
//! already. A ring holds its last window of positions; a tail the open
//! group's row. The Engram history needs no pages: a restore names every
//! prefix token, and the history takes the last of them.

use bundle_runtime::prefix::{
    Header, HeaderLayout, PrefixInstance, Stamp, layout_digest, may_restore,
};
use bundle_runtime::resources::status;
use bundle_runtime::snapshot::{
    HeaderAction, HeldBuffers, HostRankStreams, Pages, Table, copy_prefix,
};
use bundle_runtime::state::Holder;
use runtime_bundle_abi::{BundleStatusV1, PrefixStateRequestV1};

use super::Instance;

/// The shared header: the tokens, the capturing request, the rank count
/// and the declaration's layout digest.
const HEADER: HeaderLayout = HeaderLayout::V1;

impl Instance {
    /// What both sides of a snapshot must agree on.
    fn stamp(&self) -> Stamp {
        Stamp {
            ranks: u64::from(self.config.shape.ranks()),
            digest: layout_digest(&self.state),
        }
    }

    /// The table of slot `slot`'s snapshot.
    fn snapshot_table(&self, holder: Holder) -> Result<Table, BundleStatusV1> {
        HEADER
            .check(&self.state)
            .map_err(|_| BundleStatusV1::InternalError)?;
        Table::new(&self.state, self.extent, holder).map_err(|_| BundleStatusV1::InternalError)
    }

    /// Copies `slot`'s state to the snapshot `pages` (with `header`) or
    /// back into the slot on every rank, and returns once every rank's
    /// copies are done, with the header bytes rank 0 read on a restore.
    /// Every rank's copies are planned before any rank starts; a drain that
    /// cannot be confirmed is INTERNAL_ERROR, which keeps the host from
    /// reusing the pages.
    fn copy_snapshot(
        &self,
        slot: u32,
        pages: &[u64],
        header: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, BundleStatusV1> {
        let table = self.snapshot_table(Holder::Slot(slot))?;
        let to_pages = header.is_some();
        let plans = self
            .buffers
            .iter()
            .enumerate()
            .map(|(rank, buffers)| {
                let held = HeldBuffers {
                    buffers: buffers
                        .state
                        .buffers
                        .iter()
                        .map(|&(_, span)| Some(span))
                        .collect(),
                    spares: vec![None; buffers.state.buffers.len()],
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
                table
                    .actions::<()>(&held, &paged, to_pages, action)
                    .map_err(|_| BundleStatusV1::InternalError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let streams = self
            .ranks
            .iter()
            .map(|&(_, _, stream)| stream)
            .collect::<Vec<_>>();
        let rank_streams = HostRankStreams {
            services: &self.services,
            streams: &streams,
            launch: &|_, _, ()| Ok(()),
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
        let slot = self
            .ledger
            .committed(request.request_id, request.tokens)
            .map_err(|_| BundleStatusV1::InvalidState)?;
        self.dspark_catch_up(request.request_id)?;
        if let Some(device) = &self.dspark {
            let snapshot = device
                .runtime
                .as_ref()
                .ok_or(BundleStatusV1::InvalidState)?
                .state()
                .snapshot(request.request_id)
                .map_err(|_| BundleStatusV1::InvalidState)?;
            if snapshot.end != request.tokens {
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
        self.copy_snapshot(slot, &request.pages, Some(header))?;
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
            .copy_snapshot(slot, &request.pages, None)
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
        if restored.is_err() {
            self.dspark_forget(request.request_id);
            if let Some(engram) = &mut self.engram {
                engram.forget(request.request_id);
            }
        }
        restored
    }
}
