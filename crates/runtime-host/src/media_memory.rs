//! Host memory for media items. Decoding media and preparing encoder
//! inputs holds several buffers, so each image and audio item reserves a
//! conservative bound on them before it decodes, against its request's
//! budget and the process's. A prepared item's payload then keeps its bytes
//! reserved in a lease that travels with the payload through the queue and
//! the engine and drops with its last copy, once prefill has passed the item
//! and no cache keeps it. What a
//! preparation reserved beyond its payloads is released when the request's
//! [`RequestMediaMemoryV1`] drops, on success, error or unwind.
//!
//! Payloads a cache keeps for requests to come hold leases too. When a
//! reservation does not fit, the pool asks each registered [`ReclaimV1`] to
//! drop an idle payload, one at a time, before it waits.

use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};
use std::time::Duration;

use runtime_server::{AdmissionError, CancellationSignal, MediaPayloadSourceV1, MediaPayloadV1};

/// Bytes every preparation and every live payload of the process may hold
/// together: about three of the largest images a 16 MiP recipe prepares
/// from a decode at the host's pixel bound, or about forty of their
/// payloads.
pub(crate) const PROCESS_MEDIA_PREPARATION_BYTES_V1: u64 = 8 << 30;
/// Bytes one request's preparation may hold at once: the item being prepared
/// and the payloads of the items before it.
pub(crate) const REQUEST_MEDIA_PREPARATION_BYTES_V1: u64 = 4 << 30;

/// A holder of payloads no request needs, which it gives up on demand.
pub(crate) trait ReclaimV1: Send + Sync {
    /// Drops one idle payload, least recently used first; false when it
    /// holds none.
    fn reclaim_one(&self) -> bool;
}

pub(crate) struct MediaMemoryV1 {
    limit: u64,
    used: Mutex<u64>,
    available: Condvar,
    reclaimers: Mutex<Vec<Weak<dyn ReclaimV1>>>,
}

impl MediaMemoryV1 {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            used: Mutex::new(0),
            available: Condvar::new(),
            reclaimers: Mutex::new(Vec::new()),
        }
    }

    /// Lets the pool drop `reclaimer`'s idle payloads when a reservation
    /// does not fit. The pool holds it weakly.
    pub(crate) fn register(&self, reclaimer: Weak<dyn ReclaimV1>) {
        self.reclaimers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(reclaimer);
    }

    /// Asks the registered holders to drop one idle payload. Called without
    /// the `used` lock, since a dropped payload gives its bytes back here.
    fn reclaim_one(&self) -> bool {
        let reclaimers = {
            let mut reclaimers = self
                .reclaimers
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            reclaimers.retain(|reclaimer| reclaimer.strong_count() > 0);
            reclaimers.clone()
        };
        reclaimers
            .iter()
            .filter_map(Weak::upgrade)
            .any(|reclaimer| reclaimer.reclaim_one())
    }

    /// The pool every backend of this process shares.
    pub(crate) fn process() -> Arc<Self> {
        static PROCESS: OnceLock<Arc<MediaMemoryV1>> = OnceLock::new();
        Arc::clone(PROCESS.get_or_init(|| Arc::new(Self::new(PROCESS_MEDIA_PREPARATION_BYTES_V1))))
    }

    #[cfg(test)]
    pub(crate) fn in_use(&self) -> u64 {
        *self
            .used
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Takes `bytes` if they fit now.
    fn try_take(&self, bytes: u64) -> bool {
        let mut used = self
            .used
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match used.checked_add(bytes).filter(|total| *total <= self.limit) {
            Some(total) => {
                *used = total;
                true
            }
            None => false,
        }
    }

    fn take(&self, bytes: u64) -> bool {
        loop {
            if self.try_take(bytes) {
                return true;
            }
            if !self.reclaim_one() {
                return false;
            }
        }
    }

    fn wait(&self, bytes: u64, cancellation: &CancellationSignal) -> Result<(), AdmissionError> {
        loop {
            if cancellation.is_cancelled() {
                return Err(AdmissionError::Rejected(
                    "media preparation cancelled".into(),
                ));
            }
            if self.try_take(bytes) {
                return Ok(());
            }
            if self.reclaim_one() {
                continue;
            }
            let used = self
                .used
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if used
                .checked_add(bytes)
                .is_some_and(|total| total <= self.limit)
            {
                continue;
            }
            let _ = self
                .available
                .wait_timeout(used, Duration::from_millis(50))
                .unwrap_or_else(|poison| poison.into_inner());
        }
    }

    fn give(&self, bytes: u64) {
        let mut used = self
            .used
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *used = used.saturating_sub(bytes);
        self.available.notify_all();
    }
}

/// Payload bytes reserved until their last copy drops.
struct LeasedBytes {
    bytes: Vec<u8>,
    pool: Arc<MediaMemoryV1>,
}

impl AsRef<[u8]> for LeasedBytes {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for LeasedBytes {
    fn drop(&mut self) {
        self.pool.give(self.bytes.len() as u64);
    }
}

/// A payload kept in another form than its bytes, with the bytes that form
/// holds reserved until its last copy drops.
struct LeasedSource<S> {
    source: S,
    retained: u64,
    pool: Arc<MediaMemoryV1>,
}

impl<S: MediaPayloadSourceV1> MediaPayloadSourceV1 for LeasedSource<S> {
    fn len(&self) -> usize {
        self.source.len()
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        self.source.write_to(out);
    }
}

impl<S> Drop for LeasedSource<S> {
    fn drop(&mut self) {
        self.pool.give(self.retained);
    }
}

/// One request's reservations against the process pool.
pub(crate) struct RequestMediaMemoryV1 {
    pool: Arc<MediaMemoryV1>,
    limit: u64,
    /// Preparation bytes not yet turned into payload leases.
    held: u64,
    /// Payload bytes this request's items lease.
    retained: u64,
    upfront: bool,
}

impl RequestMediaMemoryV1 {
    #[cfg(test)]
    pub(crate) fn new(pool: Arc<MediaMemoryV1>, limit: u64) -> Self {
        Self {
            pool,
            limit,
            held: 0,
            retained: 0,
            upfront: false,
        }
    }

    /// Acquire the full preparation allowance before reading media. Waiting
    /// requests own no decoded payloads, so partial preparations cannot deadlock.
    pub(crate) fn wait(
        pool: Arc<MediaMemoryV1>,
        limit: u64,
        cancellation: &CancellationSignal,
    ) -> Result<Self, AdmissionError> {
        let limit = limit.min(pool.limit);
        pool.wait(limit, cancellation)?;
        Ok(Self {
            pool,
            limit,
            held: 0,
            retained: 0,
            upfront: true,
        })
    }

    /// Checks one media item's preparation against its request's allowance before
    /// decoding. Production requests already reserved that allowance in `wait`;
    /// tests can also exercise incremental reservations on an isolated pool.
    pub(crate) fn reserve(&mut self, bytes: u64) -> Result<(), AdmissionError> {
        let needed = self
            .held
            .saturating_add(self.retained)
            .saturating_add(bytes);
        if needed > self.limit {
            return Err(AdmissionError::Rejected(format!(
                "the request's media need {needed} bytes of host memory to prepare; \
                 the limit is {}",
                self.limit
            )));
        }
        if !self.upfront && !self.pool.take(bytes) {
            return Err(AdmissionError::Unavailable(
                "media preparation memory is in use by other requests; retry later".to_owned(),
            ));
        }
        self.held += bytes;
        Ok(())
    }

    /// Turns the `reserved` preparation bytes of an item into a lease on its
    /// payload `bytes`: the rest is released now and the payload's length
    /// stays reserved until its last copy drops. A payload larger than its
    /// reservation (video reserves nothing before) reserves the
    /// difference first and may be refused like [`Self::reserve`].
    pub(crate) fn retain(
        &mut self,
        reserved: u64,
        bytes: Vec<u8>,
    ) -> Result<MediaPayloadV1, AdmissionError> {
        let kept = bytes.len() as u64;
        self.lease(reserved, kept)?;
        Ok(MediaPayloadV1::new(LeasedBytes {
            bytes,
            pool: Arc::clone(&self.pool),
        }))
    }

    /// [`Self::retain`] for a payload `source` keeps in `kept` bytes, such
    /// as an image's 8-bit canvas: those bytes, not the payload's length,
    /// stay reserved until its last copy drops.
    pub(crate) fn retain_source(
        &mut self,
        reserved: u64,
        source: impl MediaPayloadSourceV1 + 'static,
        kept: u64,
    ) -> Result<MediaPayloadV1, AdmissionError> {
        self.lease(reserved, kept)?;
        Ok(MediaPayloadV1::from_source(LeasedSource {
            source,
            retained: kept,
            pool: Arc::clone(&self.pool),
        }))
    }

    /// Moves `kept` of an item's `reserved` preparation bytes into a lease
    /// and releases the rest.
    fn lease(&mut self, reserved: u64, kept: u64) -> Result<(), AdmissionError> {
        if kept > reserved {
            self.reserve(kept - reserved)?;
        }
        let reserved = reserved.max(kept).min(self.held);
        self.held -= reserved;
        if !self.upfront {
            self.pool.give(reserved - kept.min(reserved));
        }
        self.retained += kept;
        Ok(())
    }
}

impl Drop for RequestMediaMemoryV1 {
    fn drop(&mut self) {
        self.pool.give(if self.upfront {
            self.limit - self.retained
        } else {
            self.held
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_share_the_process_budget_and_release_on_drop() {
        let pool = Arc::new(MediaMemoryV1::new(100));
        let mut first = RequestMediaMemoryV1::new(Arc::clone(&pool), 80);
        first.reserve(60).unwrap();
        // The request's own budget refuses as a client error.
        assert!(matches!(
            first.reserve(30),
            Err(AdmissionError::Rejected(_))
        ));
        // Another request finds the process budget held: unavailable.
        let mut second = RequestMediaMemoryV1::new(Arc::clone(&pool), 80);
        assert!(matches!(
            second.reserve(50),
            Err(AdmissionError::Unavailable(_))
        ));
        assert_eq!(pool.in_use(), 60);
        // A prepared image keeps its payload's bytes in a lease that
        // outlives the request's preparation.
        let payload = first.retain(60, vec![7; 10]).unwrap();
        assert_eq!(pool.in_use(), 10);
        second.reserve(50).unwrap();
        // The request's budget still counts its leased payloads.
        assert!(matches!(
            first.reserve(71),
            Err(AdmissionError::Rejected(_))
        ));
        first.reserve(30).unwrap();
        assert_eq!(pool.in_use(), 90);
        drop(first);
        drop(second);
        assert_eq!(pool.in_use(), 10);
        let copy = payload.clone();
        drop(payload);
        assert_eq!((pool.in_use(), copy.len()), (10, 10));
        drop(copy);
        assert_eq!(pool.in_use(), 0);
        // A payload without a preparation reservation reserves its length.
        let mut third = RequestMediaMemoryV1::new(Arc::clone(&pool), 80);
        let audio = third.retain(0, vec![1; 40]).unwrap();
        assert!(matches!(
            third.retain(0, vec![1; 41]),
            Err(AdmissionError::Rejected(_))
        ));
        drop(third);
        assert_eq!(pool.in_use(), 40);
        drop(audio);
        assert_eq!(pool.in_use(), 0);
    }
}
