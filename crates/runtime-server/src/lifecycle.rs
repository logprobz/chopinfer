//! Admission drain and worker supervision for a process-owned listener.
use std::{future::Future, io, sync::Arc, time::Duration};

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use tokio::{
    net::TcpListener,
    sync::{Semaphore, oneshot},
    time::{sleep, timeout},
};

use crate::{
    BackendReadiness, CancellationSignal, GenerationBackend, MetricsSnapshotProvider,
    MonotonicClock, ServerConfig, ServerFrontendV1, error::ApiError,
};

pub(crate) async fn admission(
    State(state): State<Arc<crate::ServerState>>,
    request: Request,
    next: Next,
) -> Response {
    if state.in_flight.is_closed() && request.uri().path() != "/metrics" {
        state
            .admission_metrics
            .response(axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let mut response = draining_response("shutdown");
        if readiness_probe(&request) {
            response
                .extensions_mut()
                .insert(crate::request_log::ReadinessAnswer);
        }
        return response;
    }
    next.run(request).await
}

/// A request the router answers with readiness: `/health` and `/ping` with
/// the methods their routes take.
fn readiness_probe(request: &Request) -> bool {
    use axum::http::Method;
    let method = request.method();
    match request.uri().path() {
        "/health" => method == Method::GET || method == Method::HEAD,
        "/ping" => method == Method::GET || method == Method::HEAD || method == Method::POST,
        _ => false,
    }
}

pub(crate) fn draining_response(request_id: &str) -> Response {
    let mut refusal = ApiError::unavailable("server is draining");
    // A drain refusal is part of a stop, not a fault: its line is a warning.
    refusal.log.detail_mut().shutdown = true;
    let mut response = refusal.into_response(request_id);
    response.headers_mut().insert(
        axum::http::header::RETRY_AFTER,
        axum::http::HeaderValue::from_static("1"),
    );
    response
}

/// Refuses new work with 503 and Retry-After on shutdown while admitted responses drain.
/// Health becomes unready before the listener closes on completion or deadline.
/// The caller owns backend teardown and must drop its async runtime after return
/// to cancel any HTTP connections still blocked on a slow client.
#[allow(clippy::too_many_arguments)]
pub async fn serve_with_shutdown(
    listener: TcpListener,
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics: Arc<dyn MetricsSnapshotProvider>,
    shutdown: impl Future<Output = ()>,
    drain_timeout: Duration,
) -> io::Result<()> {
    serve_with_shutdown_and_http_options(
        listener,
        backend,
        frontend,
        config,
        clock,
        metrics,
        shutdown,
        drain_timeout,
        crate::HttpOptions::default(),
    )
    .await
}

/// Serves with deployment HTTP policy and the same drain lifecycle.
#[allow(clippy::too_many_arguments)]
pub async fn serve_with_shutdown_and_http_options(
    listener: TcpListener,
    backend: Arc<dyn GenerationBackend>,
    frontend: Arc<dyn ServerFrontendV1>,
    config: ServerConfig,
    clock: Arc<dyn MonotonicClock>,
    metrics: Arc<dyn MetricsSnapshotProvider>,
    shutdown: impl Future<Output = ()>,
    drain_timeout: Duration,
    options: crate::HttpOptions,
) -> io::Result<()> {
    let in_flight = Arc::new(Semaphore::new(config.max_in_flight()));
    let drain_expired = CancellationSignal::new();
    let frontend = crate::aliases::wrap(frontend, &options.served_model_names);
    let (router, state) = crate::router_with_lifecycle(
        Arc::clone(&backend),
        frontend,
        config,
        clock,
        metrics,
        Arc::clone(&in_flight),
        drain_expired.clone(),
        &options,
    )
    .map_err(io::Error::other)?;
    let interval = crate::summary_interval(&options);
    let router = crate::logged(options.layer(router), &state);
    let log = Arc::clone(&state.request_log);
    // Every return writes what is queued: the lines of requests the stop
    // ended included.
    let _flush = FlushOnDrop(Arc::clone(&log));
    let (stop, stopped) = oneshot::channel();
    let server = crate::transport::serve(
        listener,
        router,
        config,
        Arc::clone(&state.transport),
        Arc::clone(&log),
        async {
            let _ = stopped.await;
        },
    );
    tokio::pin!(server);
    let summaries = crate::summarize(Arc::clone(&state), interval);
    tokio::pin!(summaries);
    let failed = async {
        loop {
            if let BackendReadiness::Terminated(reason) = backend.readiness() {
                // No handler will answer the requests the worker held.
                log.sweep(&crate::request_log::worker_terminated_detail());
                return io::Error::other(reason);
            }
            sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::pin!(failed);
    tokio::select! {
        result = &mut server => return result,
        error = &mut failed => return Err(error),
        () = &mut summaries => unreachable!("summaries run until dropped"),
        () = shutdown => {}
    }
    // Closing admission is atomic with permit acquisition. Requests still
    // parsing when the signal arrives cannot slip into the engine afterwards.
    in_flight.close();
    runtime_log::info!(
        "shutdown: draining for up to {:.3}s",
        drain_timeout.as_secs_f64()
    );
    let drained = async {
        while in_flight.available_permits() != config.max_in_flight() {
            sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::select! {
        error = &mut failed => return Err(error),
        result = &mut server => return result,
        result = timeout(drain_timeout, drained) => {
            if result.is_err() {
                // Requests that end from here on ended by shutdown, not by
                // their clients.
                log.begin_shutdown();
                // Notify response tasks while the runtime and connections are
                // still alive, so cancellation can produce terminal responses.
                drain_expired.cancel();
            }
        }
    }
    let _ = stop.send(());
    // Bound terminal response delivery for slow clients. The worker observes
    // request cancellation and owns native resource release after return.
    let _ = timeout(Duration::from_millis(100), &mut server).await;
    // A request still open has no handler left to report it.
    log.begin_shutdown();
    log.sweep(&crate::request_log::shutdown_detail());
    Ok(())
}

/// Waits briefly for the request log's queued lines when the server returns.
struct FlushOnDrop(Arc<crate::request_log::RequestLog>);

impl Drop for FlushOnDrop {
    fn drop(&mut self) {
        self.0.flush();
    }
}
