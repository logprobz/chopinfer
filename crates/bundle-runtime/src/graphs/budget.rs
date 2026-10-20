//! Shared capture admission. Reservations cover a whole rank group before
//! lowering can omit cached programs or a rank begins capturing.
use super::{GraphCache, Ledger};
use runtime_bundle_abi::{BundleAbiError, BundleStatusV1};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Default)]
pub(super) struct Budget {
    pub capacity: Option<usize>,
    pub reserved: usize,
    pub peak: usize,
    pub captured: u64,
    pub capture_failures: u64,
    pub released: u64,
    pub release_failures: u64,
    pub denied: u64,
    pub pressure: bool,
}

/// Counts the shared ledger once, including failed-cleanup handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphStats {
    pub capacity: Option<usize>,
    pub live: usize,
    pub reserved: usize,
    pub peak: usize,
    pub epoch: u64,
    pub captured: u64,
    pub capture_failures: u64,
    pub released: u64,
    pub release_failures: u64,
    pub denied: u64,
    pub pressure: bool,
}

/// Unused capacity returns to the shared ledger when this value drops.
/// Successful captures consume it even if a later capture fails.
#[derive(Debug)]
pub struct GraphReservation {
    pub(super) ledger: Rc<RefCell<Ledger>>,
    pub(super) epoch: u64,
    pub(super) remaining: usize,
}

impl GraphReservation {
    pub(super) fn validate<G>(
        &self,
        cache: &GraphCache<G>,
        needed: usize,
    ) -> Result<(), BundleAbiError> {
        if !Rc::ptr_eq(&self.ledger, &cache.ledger)
            || self.epoch != cache.ledger.borrow().epoch
            || self.remaining < needed
        {
            return Err(BundleAbiError::InvalidRange("graph capture reservation"));
        }
        Ok(())
    }
}

impl Drop for GraphReservation {
    fn drop(&mut self) {
        self.ledger.borrow_mut().budget.reserved -= self.remaining;
    }
}

impl<G> GraphCache<G> {
    /// Bound every cache sharing this ledger. Unconfigured ledgers stay
    /// unbounded. A live reservation or a smaller-than-live cap is refused.
    pub fn set_capacity(&self, capacity: usize) -> Result<(), BundleAbiError> {
        let mut ledger = self.ledger.borrow_mut();
        if ledger.budget.reserved != 0 || ledger.graphs.len() > capacity {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidState));
        }
        ledger.budget.capacity = Some(capacity);
        Ok(())
    }

    /// Admit a complete capture group across ranks and sibling caches.
    /// None means capacity pressure, before any capture or launch starts.
    pub fn reserve(&self, handles: usize) -> Option<GraphReservation> {
        let mut ledger = self.ledger.borrow_mut();
        let occupied = ledger.graphs.len().checked_add(ledger.budget.reserved);
        let admitted = occupied.and_then(|n| n.checked_add(handles));
        if admitted.is_none_or(|n| ledger.budget.capacity.is_some_and(|limit| n > limit)) {
            ledger.budget.denied += 1;
            ledger.budget.pressure = true;
            return None;
        }
        ledger.budget.reserved += handles;
        Some(GraphReservation {
            ledger: Rc::clone(&self.ledger),
            epoch: ledger.epoch,
            remaining: handles,
        })
    }

    pub fn stats(&self) -> GraphStats {
        let ledger = self.ledger.borrow();
        let budget = &ledger.budget;
        GraphStats {
            capacity: budget.capacity,
            live: ledger.graphs.len(),
            reserved: budget.reserved,
            peak: budget.peak,
            epoch: ledger.epoch,
            captured: budget.captured,
            capture_failures: budget.capture_failures,
            released: budget.released,
            release_failures: budget.release_failures,
            denied: budget.denied,
            pressure: budget.pressure,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Label {
    pub family: &'static str,
    pub width: Option<u32>,
    pub segment: u32,
}

impl Default for Label {
    fn default() -> Self {
        Self {
            family: "target",
            width: None,
            segment: 0,
        }
    }
}

impl<G> GraphCache<G> {
    /// Name the program represented by an opaque key in capacity telemetry.
    pub fn label(&mut self, family: &'static str, width: Option<u32>, segment: u32) {
        self.label = Label {
            family,
            width,
            segment,
        };
    }

    /// Capacity diagnostics use the host's bounded routine-log path.
    /// Unbounded caches retain their existing logging behavior.
    pub fn trace(
        &self,
        services: &runtime_bundle_abi::HostServicesV1,
        event: &str,
        key: Option<super::GraphKey>,
        pieces: usize,
        micros: u128,
    ) {
        let stats = self.stats();
        if stats.capacity.is_none() {
            return;
        }
        let message = serde_json::json!({
            "schema": "chopinfer.graph-cache.v1", "event": event,
            "family": self.label.family, "segment": self.label.segment,
            "rank": key.map(|k| k.rank), "width": self.label.width.or(key.map(|k| k.bucket)),
            "items": key.map(|k| k.items), "pieces": pieces, "micros": micros,
            "capacity": stats.capacity, "live": stats.live, "reserved": stats.reserved,
            "peak": stats.peak, "epoch": stats.epoch, "captured": stats.captured,
            "capture_failures": stats.capture_failures, "released": stats.released,
            "release_failures": stats.release_failures, "denied": stats.denied,
            "pressure": stats.pressure
        })
        .to_string();
        let _ = services.log(message.as_bytes());
    }
}
