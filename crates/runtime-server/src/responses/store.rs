use std::collections::VecDeque;
use std::sync::Mutex;

use serde_json::Value;

use crate::{CancellationSignal, error::ApiError};

pub(crate) struct Store {
    entries: Mutex<VecDeque<Entry>>,
    maximum_entries: usize,
    maximum_bytes: usize,
    available: tokio::sync::Notify,
    admission: tokio::sync::Mutex<()>,
}
struct Entry {
    id: String,
    response: Value,
    history: Vec<Value>,
    cancel: CancellationSignal,
    active: bool,
    bytes: usize,
}
impl Store {
    pub fn new(maximum_entries: usize, maximum_bytes: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::new()),
            maximum_entries,
            maximum_bytes,
            available: tokio::sync::Notify::new(),
            admission: tokio::sync::Mutex::new(()),
        }
    }
    pub fn enabled(&self) -> bool {
        self.maximum_entries > 0 && self.maximum_bytes > 0
    }
    pub fn history(
        &self,
        id: &str,
        intake: &crate::intake::Intake,
        charge: &mut crate::intake::IntakeCharge,
    ) -> Result<Vec<Value>, ApiError> {
        let entries = self.entries.lock().expect("response store");
        let entry = entries
            .iter()
            .find(|entry| entry.id == id && (!entry.active || entry.response["background"] == true))
            .ok_or_else(|| not_found(id))?;
        if entry.active {
            return Err(super::request::invalid(
                "previous response is still in progress",
                "previous_response_id",
            ));
        }
        charge.extend(intake, json_bytes(&entry.history))?;
        Ok(entry.history.clone())
    }
    pub fn get(&self, id: &str) -> Result<Value, ApiError> {
        self.entries
            .lock()
            .expect("response store")
            .iter()
            .find(|entry| entry.id == id && (!entry.active || entry.response["background"] == true))
            .map(|entry| entry.response.clone())
            .ok_or_else(|| not_found(id))
    }
    pub async fn reserve_waiting(
        &self,
        response: &Value,
        history: &[Value],
        cancellation: CancellationSignal,
        output_limit: usize,
        drain: Option<&CancellationSignal>,
    ) -> Result<(), ApiError> {
        // Tokio's mutex grants turns in polling order. Hold the turn while
        // storage is full so new arrivals cannot steal a released entry.
        let _turn = match drain {
            Some(drain) => tokio::select! {
                biased;
                () = cancellation.cancelled() => return Err(ApiError::unavailable("response cancelled while waiting for storage")),
                () = drain.cancelled() => return Err(ApiError::unavailable("server drain deadline expired")),
                turn = self.admission.lock() => turn,
            },
            None => self
                .admission
                .try_lock()
                .map_err(|_| ApiError::capacity("response store capacity is exhausted"))?,
        };
        loop {
            if cancellation.is_cancelled() {
                return Err(ApiError::unavailable(
                    "response cancelled while waiting for storage",
                ));
            }
            let available = self.available.notified();
            tokio::pin!(available);
            available.as_mut().enable();
            match self.reserve(response, history, cancellation.clone(), output_limit) {
                Ok(()) => return Ok(()),
                Err(error)
                    if error.status == axum::http::StatusCode::TOO_MANY_REQUESTS
                        && drain.is_some() => {}
                Err(error) => return Err(error),
            }
            tokio::select! {
                () = &mut available => {},
                () = cancellation.cancelled() => return Err(ApiError::unavailable("response cancelled while waiting for storage")),
                () = drain.expect("waiting is enabled").cancelled() => return Err(ApiError::unavailable("server drain deadline expired")),
            }
        }
    }

    fn reserve(
        &self,
        response: &Value,
        history: &[Value],
        cancel: CancellationSignal,
        output_limit: usize,
    ) -> Result<(), ApiError> {
        // Reserve output space before admitting a stored response. Completed
        // entries can be evicted; active background work is never evicted.
        let bytes = json_bytes(&response)
            .saturating_add(json_bytes(&history))
            .saturating_add(output_limit.saturating_mul(8));
        let mut entries = self.entries.lock().expect("response store");
        if bytes > self.maximum_bytes {
            return Err(ApiError::invalid(
                "response exceeds the configured store byte budget",
                Some("input"),
            ));
        }
        while entries.len() >= self.maximum_entries
            || entries
                .iter()
                .map(|entry| entry.bytes)
                .sum::<usize>()
                .saturating_add(bytes)
                > self.maximum_bytes
        {
            let Some(index) = entries.iter().position(|entry| !entry.active) else {
                return Err(ApiError::capacity("response store capacity is exhausted"));
            };
            entries.remove(index);
        }
        entries.push_back(Entry {
            id: response["id"].as_str().expect("response id").to_owned(),
            response: response.clone(),
            history: history.to_vec(),
            cancel,
            active: true,
            bytes,
        });
        Ok(())
    }
    pub fn finish(&self, id: &str, response: Value, output_history: Vec<Value>) {
        let mut entries = self.entries.lock().expect("response store");
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) {
            if entry.response["status"] != "cancelled" {
                entry.response = response;
                entry.history.extend(output_history);
            }
            entry.active = false;
            self.available.notify_waiters();
            entry.bytes = json_bytes(&entry.response).saturating_add(json_bytes(&entry.history));
            // Serialization and item overhead can exceed the reservation for
            // many tiny items. Drop that record instead of exceeding the bound.
            if entries.iter().map(|entry| entry.bytes).sum::<usize>() > self.maximum_bytes {
                entries.retain(|entry| entry.id != id);
            }
        }
    }
    pub fn abandon(&self, id: &str) {
        let mut entries = self.entries.lock().expect("response store");
        entries.retain(|entry| entry.id != id || entry.response["status"] == "cancelled");
        self.available.notify_waiters();
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == id) {
            entry.active = false;
            self.available.notify_waiters();
        }
    }
    pub fn cancel(&self, id: &str) -> Result<Value, ApiError> {
        let mut entries = self.entries.lock().expect("response store");
        let entry = entries
            .iter_mut()
            .find(|entry| entry.id == id)
            .ok_or_else(|| not_found(id))?;
        if entry.active && entry.response["background"] != true {
            return Err(not_found(id));
        }
        if !entry.active
            || entry.response["background"] != true
            || entry.response["status"] == "cancelled"
        {
            let mut error =
                super::request::invalid("Cannot cancel a synchronous response.", "response_id");
            error.kind = "invalid_request_error";
            return Err(error);
        }
        entry.response["status"] = Value::String("cancelled".to_owned());
        entry.cancel.cancel();
        Ok(entry.response.clone())
    }
}
fn json_bytes(value: &impl serde::Serialize) -> usize {
    serde_json::to_vec(value).expect("JSON value").len()
}
fn not_found(id: &str) -> ApiError {
    ApiError {
        status: axum::http::StatusCode::NOT_FOUND,
        message: format!("Response with id '{id}' not found."),
        kind: "invalid_request_error",
        code: None,
        parameter: Some("response_id".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{future::Future, pin::Pin, task::Poll};

    async fn poll_once<F: Future>(future: Pin<&mut F>) -> Poll<F::Output> {
        let mut future = future;
        std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx))).await
    }

    #[tokio::test]
    async fn storage_waiters_keep_order_when_a_new_request_arrives_and_a_waiter_leaves() {
        let store = Store::new(1, 65536);
        let drain = CancellationSignal::new();
        let held = json!({"id":"held"});
        store
            .reserve(&held, &[], CancellationSignal::new(), 1024)
            .unwrap();
        let a = json!({"id":"a"});
        let b = json!({"id":"b"});
        let c = json!({"id":"c"});
        let mut first =
            Box::pin(store.reserve_waiting(&a, &[], CancellationSignal::new(), 1024, Some(&drain)));
        let mut second =
            Box::pin(store.reserve_waiting(&b, &[], CancellationSignal::new(), 1024, Some(&drain)));
        assert!(poll_once(first.as_mut()).await.is_pending());
        assert!(poll_once(second.as_mut()).await.is_pending());
        store.abandon("held");
        let mut newcomer =
            Box::pin(store.reserve_waiting(&c, &[], CancellationSignal::new(), 1024, Some(&drain)));
        assert!(
            poll_once(newcomer.as_mut()).await.is_pending(),
            "new request overtook existing waiters"
        );
        assert!(matches!(
            poll_once(first.as_mut()).await,
            Poll::Ready(Ok(()))
        ));
        store.abandon("a");
        assert!(
            poll_once(newcomer.as_mut()).await.is_pending(),
            "new request overtook the second waiter"
        );
        drop(second);
        assert!(matches!(
            poll_once(newcomer.as_mut()).await,
            Poll::Ready(Ok(()))
        ));
    }
}
