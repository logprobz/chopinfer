//! Bound graphs across target pieces, drafts and catch-up without releasing
//! request state or mapped-gather resources.
use super::Instance;
use crate::indexer_workloads::Route;
use bundle_runtime::graphs::{GraphKey, GraphReservation};
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1};

/// Graph handles per rank. A TP4 startup warms 131 target keys of three
/// pieces per rank, 24 catch-up and 6 draft widths: 423 handles per rank,
/// which 384 could not hold, so the widest 16-item verify keys ran eagerly.
/// The rest leaves room for completion tails that serving adds lazily.
/// Workloads at 546 handles per rank held about 1.19 GiB of capture
/// allocations with at least 3.10 GiB free on each GPU.
const HANDLES_PER_RANK: usize = 512;
const ENV: &str = "CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK";

impl Instance {
    pub(crate) fn configure_dspark_graphs(&self) -> Result<(), BundleAbiError> {
        if self.dspark.is_none() {
            return Ok(());
        }
        let per_rank = match std::env::var_os(ENV) {
            None => HANDLES_PER_RANK,
            Some(value) => value
                .to_str()
                .and_then(|v| v.parse::<usize>().ok())
                .ok_or_else(|| {
                    self.log("CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK requires an integer");
                    BundleAbiError::Status(BundleStatusV1::InvalidArgument)
                })?,
        };
        let capacity = per_rank
            .checked_mul(self.ranks.len())
            .ok_or(BundleAbiError::InvalidRange("DSpark graph capacity"))?;
        self.graphs.set_capacity(capacity)?;
        self.graphs.trace(&self.services, "capacity", None, 0, 0);
        Ok(())
    }

    pub(crate) fn reserve_dspark_target(
        &self,
        (bucket, items): (u32, u32),
        route: Route,
    ) -> Option<GraphReservation> {
        self.dspark.as_ref()?;
        let missing = (0..self.ranks.len())
            .map(|rank| {
                let key = GraphKey {
                    rank: rank as u32,
                    bucket,
                    items,
                };
                self.indexer_graph_missing(route, key)
            })
            .sum();
        // Completion tails can reach hundreds of exact keys. Keep the
        // warmed keys when full and lower every rank of a new shape eagerly.
        let admission = self.graphs.reserve_retaining(missing);
        if admission.is_none() {
            self.graphs.trace(
                &self.services,
                "target-eager",
                Some(GraphKey {
                    rank: 0,
                    bucket,
                    items,
                }),
                missing,
                0,
            );
        }
        admission
    }

    /// Only an execution boundary can retire the ledger. Prefix catch-up
    /// may request capture capacity too, but never retires graphs itself.
    pub(crate) fn maintain_dspark_graphs(&mut self) -> Result<(), BundleAbiError> {
        let Some(device) = &self.dspark else {
            return Ok(());
        };
        let stats = self.graphs.stats();
        if !stats.pressure || stats.live == 0 {
            return Ok(());
        }
        if self.collectives.is_poisoned()
            || self.ledger.step().is_some()
            || device.cycle.pending.is_some()
            || device.cycle.gathering
            || device.results.metadata.is_some()
            || device
                .runtime
                .as_ref()
                .is_none_or(|r| r.state().is_pending() || r.state().is_poisoned())
        {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        // Results have been collected and the prior transaction committed or
        // cancelled. Establish completion of every lane before releasing any
        // handle. No graph lookup or lowering for this step has happened yet.
        crate::instance::execute::dma::drain_streams(self)?;
        self.graphs
            .trace(&self.services, "eviction-begin", None, 0, 0);
        self.graphs.release_all(&self.services)
    }
}
