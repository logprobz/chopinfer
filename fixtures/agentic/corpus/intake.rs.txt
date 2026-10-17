//! Bounded request intake: what a request may hold before it is admitted.
//!
//! Generation capacity is taken only after a request is parsed and its
//! prompts are prepared, so without a bound of its own every open connection
//! could buffer a whole body and tokenize it. Intake charges each body's
//! bytes, as they arrive, against one budget shared by all connections, and
//! gives every body one deadline to arrive in. The charge is held until the
//! request is admitted or refused.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::http::StatusCode;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::{Instant, timeout_at};
use tokio_stream::StreamExt;

use crate::error::ApiError;

/// The least a request is charged, so the budget bounds how many requests are
/// in intake at once as well as their bytes.
const MINIMUM_CHARGE_BYTES: usize = 16 * 1024;

#[derive(Debug)]
pub(crate) struct Intake {
    budget: Arc<Semaphore>,
    body_read_timeout: Duration,
    capacity: usize,
}

/// Bytes one request holds against the intake budget until it is dropped.
pub(crate) struct IntakeCharge {
    _permit: OwnedSemaphorePermit,
    used_bytes: usize,
    fixed: bool,
}

impl IntakeCharge {
    /// Charges expanded stored history against the same intake reservation.
    pub(crate) fn extend(&mut self, intake: &Intake, bytes: usize) -> Result<(), ApiError> {
        let total = self
            .used_bytes
            .checked_add(bytes)
            .ok_or_else(|| ApiError::capacity("expanded request size overflowed"))?;
        if total > self._permit.num_permits() && self.fixed {
            return Err(ApiError::invalid(
                "expanded request exceeds the intake byte limit",
                None,
            ));
        }
        if total > self._permit.num_permits() {
            self._permit
                .merge(intake.charge(total - self._permit.num_permits())?);
        }
        self.used_bytes = total;
        Ok(())
    }
}

impl Intake {
    pub(crate) fn new(budget_bytes: usize, body_read_timeout: Duration) -> Self {
        Self {
            budget: Arc::new(Semaphore::new(budget_bytes.min(Semaphore::MAX_PERMITS))),
            body_read_timeout,
            capacity: budget_bytes.min(Semaphore::MAX_PERMITS),
        }
    }

    pub(crate) fn prometheus(&self) -> String {
        let available = self.budget.available_permits();
        format!(
            "# TYPE chop_intake_budget_bytes gauge\nchop_intake_budget_bytes {}\n# TYPE chop_intake_used_bytes gauge\nchop_intake_used_bytes {}\n",
            self.capacity,
            self.capacity.saturating_sub(available)
        )
    }

    /// Reads one body of at most `limit` bytes, charging it as it arrives.
    /// Waiting, it first reserves `reserve` bytes ([`reservation`]).
    pub(crate) async fn read(
        &self,
        body: Body,
        limit: usize,
        wait: bool,
        reserve: usize,
    ) -> Result<(Bytes, IntakeCharge), ApiError> {
        // Reserve the whole body and expansion allowance atomically before
        // reading. Partial-body holders must not deadlock waiting on each other.
        let mut charged = if wait {
            reserve.max(MINIMUM_CHARGE_BYTES).min(self.capacity)
        } else {
            MINIMUM_CHARGE_BYTES
        };
        let mut charge = if wait {
            Arc::clone(&self.budget)
                .acquire_many_owned(charged as u32)
                .await
                .map_err(|_| ApiError::unavailable("server intake is closed"))?
        } else {
            self.charge(charged)?
        };
        let deadline = Instant::now() + self.body_read_timeout;
        let mut bytes = Vec::new();
        let mut stream = body.into_data_stream();
        loop {
            let chunk = match timeout_at(deadline, stream.next()).await {
                Err(_) => return Err(body_timeout()),
                Ok(None) => break,
                Ok(Some(Err(_))) => {
                    return Err(ApiError::invalid("request body could not be read", None));
                }
                Ok(Some(Ok(chunk))) => chunk,
            };
            let total = bytes.len() + chunk.len();
            if total > limit {
                return Err(ApiError::invalid(
                    "request body exceeds the byte limit",
                    None,
                ));
            }
            if total > charged {
                charge.merge(self.charge(total - charged)?);
                charged = total;
            }
            bytes.extend_from_slice(&chunk);
        }
        let used_bytes = bytes.len();
        Ok((
            Bytes::from(bytes),
            IntakeCharge {
                _permit: charge,
                used_bytes,
                fixed: wait,
            },
        ))
    }

    fn charge(&self, bytes: usize) -> Result<OwnedSemaphorePermit, ApiError> {
        u32::try_from(bytes)
            .ok()
            .and_then(|bytes| Arc::clone(&self.budget).try_acquire_many_owned(bytes).ok())
            .ok_or_else(|| ApiError::capacity("server request intake capacity is exhausted"))
    }
}

/// What a waiting request reserves before its body arrives. An endpoint that
/// expands the body (only stored Responses history does) reserves `limit`
/// three times over. Any other reserves its declared `Content-Length`, or
/// `limit` without one, so a large limit does not make every small request
/// wait for room.
pub(crate) fn reservation(headers: &axum::http::HeaderMap, limit: usize, expands: bool) -> usize {
    if expands {
        return limit.saturating_mul(3);
    }
    declared_length(headers).map_or(limit, |length| length.min(limit))
}

/// The `Content-Length` a request declares, when it is a plain count.
fn declared_length(headers: &axum::http::HeaderMap) -> Option<usize> {
    headers
        .get(axum::http::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn body_timeout() -> ApiError {
    ApiError {
        status: StatusCode::REQUEST_TIMEOUT,
        message: "request body did not arrive in time".to_owned(),
        kind: "RequestTimeoutError",
        code: None,
        parameter: None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Intake, reservation};
    use axum::http::{HeaderMap, HeaderValue, header::CONTENT_LENGTH};
    use std::time::Duration;

    /// Only an expanding endpoint reserves three times its limit; any other
    /// reserves its declared length, or its limit when it declares none.
    #[test]
    fn only_an_expanding_endpoint_reserves_beyond_its_limit() {
        let limit = 256 * 1024;
        let mut declared = HeaderMap::new();
        declared.insert(CONTENT_LENGTH, HeaderValue::from_static("100"));
        assert_eq!(reservation(&declared, limit, false), 100);
        assert_eq!(reservation(&HeaderMap::new(), limit, false), limit);
        assert_eq!(reservation(&declared, limit, true), 3 * limit);
        assert_eq!(reservation(&HeaderMap::new(), limit, true), 3 * limit);
        // A declared length over the limit reserves only the limit; reading
        // refuses the body.
        declared.insert(CONTENT_LENGTH, HeaderValue::from_static("999999999"));
        assert_eq!(reservation(&declared, limit, false), limit);
    }

    /// A waiting read reserves at least the minimum charge and at most the
    /// budget.
    #[tokio::test]
    async fn a_waiting_read_reserves_between_the_minimum_and_the_budget() {
        let intake = Intake::new(1 << 20, Duration::from_secs(1));
        let body = || axum::body::Body::from(vec![b' '; 100]);
        for (reserve, used) in [
            (100, 16 * 1024),
            (300 * 1024, 300 * 1024),
            (4 << 20, 1 << 20),
        ] {
            let (_, charge) = intake
                .read(body(), 256 * 1024, true, reserve)
                .await
                .unwrap();
            assert!(
                intake
                    .prometheus()
                    .contains(&format!("chop_intake_used_bytes {used}\n")),
                "{reserve}"
            );
            drop(charge);
        }
    }
}
