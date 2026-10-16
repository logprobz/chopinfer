//! Prefix-cache capture and restore of a request's KDA state, indexer tail
//! and, with the MTP drafter, its retained target hidden rows: the slot's
//! committed entries of every declared buffer (or, for a capture at the
//! anchor the last step kept inside a prompt-ending item, that anchor
//! item's spares), copied
//! piece by piece between the slot and the snapshot's serving pages with
//! device copies (`bundle_runtime::snapshot`; a piece's copies over
//! consecutive pages joined into one, `joined`), each rank its own
//! shards, and the shared header (`HeaderLayout::V1`) on rank 0, which also
//! records the drafter's frontier. The latent KV and pooled keys, the
//! drafter's layer's included, live in the shared pages already.

use bundle_runtime::prefix::{
    Header, HeaderLayout, PrefixInstance, Stamp, layout_digest, may_restore,
};
use bundle_runtime::resources::status;
use bundle_runtime::snapshot::{
    Action, HeaderAction, HeldBuffers, HostRankStreams, Pages, Table, copy_prefix,
};
use bundle_runtime::state::Holder;
use runtime_bundle_abi::{BundleStatusV1, DeviceBufferSpanV1, PrefixStateRequestV1};

use super::Instance;

/// The shared header: the tokens, the capturing request, the rank count
/// and the declaration's layout digest.
const HEADER: HeaderLayout = HeaderLayout::V1;

impl Instance {
    /// What both sides of a snapshot must agree on.
    fn stamp(&self) -> Stamp {
        Stamp {
            ranks: u64::from(self.config.shape.ranks()),
            digest: layout_digest(&self.buffers[0].state.state),
        }
    }

    /// The table of slot `slot`'s snapshot.
    pub(super) fn snapshot_table(&self, slot: u32) -> Result<Table, BundleStatusV1> {
        self.holder_table(Holder::Slot(slot))
    }

    /// The table of `holder`'s snapshot: a slot's, or an anchor item's
    /// spares.
    fn holder_table(&self, holder: Holder) -> Result<Table, BundleStatusV1> {
        let pools = &self
            .buffers
            .first()
            .ok_or(BundleStatusV1::InvalidState)?
            .state;
        HEADER
            .check(&pools.state)
            .map_err(|_| BundleStatusV1::InternalError)?;
        Table::new(&pools.state, pools.extent, holder).map_err(|_| BundleStatusV1::InternalError)
    }

    /// Copies `holder`'s state to the snapshot `pages` (with `header`)
    /// or back into a slot on every rank, and returns once every rank's
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
        let table = self.holder_table(holder)?;
        let to_pages = header.is_some();
        let plans = self
            .buffers
            .iter()
            .enumerate()
            .map(|(rank, buffers)| {
                let held = HeldBuffers {
                    buffers: buffers.state.spans.iter().copied().map(Some).collect(),
                    spares: buffers.state.spares.clone(),
                };
                let paged = Pages {
                    pool: buffers.kv_pool,
                    ids: pages,
                    page_bytes: self.config.format.page_bytes,
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
                    .map(joined)
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

/// `actions` with each device copy that continues the one before it on
/// both sides (a piece across consecutive snapshot pages) joined into it:
/// the same bytes in fewer copies. Each copy costs the host about 44 us
/// on the GPU while the device idles (a capture's 644 copies took 28.9 ms
/// of 0.9 ms of device time), and a snapshot of the default layout spans
/// 167 copies per rank where its 79 pieces would do. A copy whose source
/// and destination would overlap once joined stays apart.
fn joined<L>(actions: Vec<Action<L>>) -> Vec<Action<L>> {
    let continues = |span: DeviceBufferSpanV1, next: DeviceBufferSpanV1| {
        span.buffer == next.buffer && span.offset.checked_add(span.length) == Some(next.offset)
    };
    let disjoint = |a: DeviceBufferSpanV1, b: DeviceBufferSpanV1| {
        a.buffer != b.buffer || a.offset + a.length <= b.offset || b.offset + b.length <= a.offset
    };
    let mut joined: Vec<Action<L>> = Vec::with_capacity(actions.len());
    for action in actions {
        if let (
            Some(Action::Copy { from, to }),
            Action::Copy {
                from: next,
                to: into,
            },
        ) = (joined.last_mut(), &action)
            && continues(*from, *next)
            && continues(*to, *into)
        {
            let from_joined =
                DeviceBufferSpanV1::new(from.buffer, from.offset, from.length + next.length);
            let to_joined = DeviceBufferSpanV1::new(to.buffer, to.offset, to.length + into.length);
            if disjoint(from_joined, to_joined) {
                (*from, *to) = (from_joined, to_joined);
                continue;
            }
        }
        joined.push(action);
    }
    joined
}

impl PrefixInstance for Instance {
    fn state_pages(&self) -> Result<u64, BundleStatusV1> {
        Ok(self.snapshot_table(0)?.pages())
    }

    /// Writes a request's committed state into snapshot pages. The host
    /// captures after a commit, so no step is open. A capture at the anchor
    /// the last step kept inside a prompt-ending item copies that anchor
    /// item's spares instead, with the drafter's frontier at the pool
    /// before the anchor (`crate::drafter::anchor_frontier`).
    fn capture(&mut self, request: &PrefixStateRequestV1) -> Result<(), BundleStatusV1> {
        if !request.token_ids.is_empty() || request.tokens == 0 {
            return Err(BundleStatusV1::InvalidState);
        }
        let (holder, frontier) =
            if let Ok(anchor) = self.ledger.anchor(request.request_id, request.tokens) {
                let frontier = self
                    .drafter
                    .as_ref()
                    .map(|_| crate::drafter::anchor_frontier(self.config.shape, request.tokens));
                (Holder::Anchor(anchor), frontier)
            } else {
                let slot = self
                    .ledger
                    .committed(request.request_id, request.tokens)
                    .map_err(|_| BundleStatusV1::InvalidState)?;
                let frontier = self.drafter.as_ref().map(|drafter| drafter.frontier(slot));
                (Holder::Slot(slot), frontier)
            };
        let header = Header {
            tokens: request.tokens,
            frontier,
            owner: Some(request.request_id),
            cursor: None,
        }
        .encode(&HEADER, self.stamp())
        .map_err(|_| BundleStatusV1::InternalError)?;
        self.copy_snapshot(holder, &request.pages, Some(header))
            .map(|_| ())
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
        let read = self.copy_snapshot(Holder::Slot(slot), &request.pages, None)?;
        let header = Header::decode(&read, &HEADER, self.stamp(), request.tokens)
            .map_err(|_| BundleStatusV1::InvalidArgument)?;
        if !may_restore(&self.buffers[0].state.state, &header, request.request_id) {
            return Err(BundleStatusV1::InvalidArgument);
        }
        // The drafter continues from the snapshot's frontier: its retained
        // rows cover the rest, and its history the restored tokens.
        if let Some(drafter) = &mut self.drafter {
            let frontier = header.frontier.ok_or(BundleStatusV1::InvalidArgument)?;
            drafter
                .restore(slot, &request.token_ids, frontier)
                .map_err(|_| BundleStatusV1::InvalidArgument)?;
        }
        self.ledger
            .restore(request.request_id, request.tokens, Some(slot))
            .map(|_| ())
            .map_err(|_| BundleStatusV1::InternalError)
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::snapshot::Action;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    use super::joined;

    fn span(buffer: u32, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, buffer),
            offset,
            length,
        )
    }

    fn copy(from: DeviceBufferSpanV1, to: DeviceBufferSpanV1) -> Action<()> {
        Action::Copy { from, to }
    }

    fn spans(actions: &[Action<()>]) -> Vec<Option<(u64, u64, u64)>> {
        actions
            .iter()
            .map(|action| match action {
                Action::Copy { from, to } => Some((from.offset, to.offset, from.length)),
                _ => None,
            })
            .collect()
    }

    /// Copies that continue each other on both sides join; one that
    /// continues on one side only, follows another action, or would
    /// overlap its own destination once joined stays apart.
    #[test]
    fn copies_that_continue_on_both_sides_join() {
        let actions = vec![
            copy(span(1, 0, 16), span(2, 64, 16)),
            copy(span(1, 16, 32), span(2, 80, 32)),
            copy(span(1, 48, 16), span(2, 128, 16)),
            copy(span(1, 64, 16), span(2, 144, 16)),
            Action::Upload {
                to: span(2, 160, 4),
                bytes: vec![0; 4],
            },
            copy(span(1, 80, 16), span(2, 160, 16)),
            copy(span(3, 0, 16), span(3, 16, 16)),
            copy(span(3, 16, 16), span(3, 32, 16)),
        ];
        assert_eq!(
            spans(&joined(actions)),
            [
                Some((0, 64, 48)),
                Some((48, 128, 32)),
                None,
                Some((80, 160, 16)),
                Some((0, 16, 16)),
                Some((16, 32, 16)),
            ]
        );
    }
}
