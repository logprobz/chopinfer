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
        return draining_response("shutdown");
    }
    next.run(request).await
}

pub(crate) fn draining_response(request_id: &str) -> Response {
    let mut response = ApiError::unavailable("server is draining").into_response(request_id);
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
    let (router, transport) = crate::router_with_lifecycle(
        Arc::clone(&backend),
        frontend,
        config,
        clock,
        metrics,
        Arc::clone(&in_flight),
        drain_expired.clone(),
        options.serving_identity.as_ref(),
    )
    .map_err(io::Error::other)?;
    let router = options.layer(router);
    let _refusal_log_flush = crate::error::RefusalLogFlushGuard;
    let (stop, stopped) = oneshot::channel();
    let server = crate::transport::serve(listener, router, config, transport, async {
        let _ = stopped.await;
    });
    tokio::pin!(server);
    let failed = async {
        loop {
            if let BackendReadiness::Terminated(reason) = backend.readiness() {
                return io::Error::other(reason);
            }
            sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::pin!(failed);
    tokio::select! {
        result = &mut server => return result,
        error = &mut failed => return Err(error),
        () = shutdown => {}
    }
    // Closing admission is atomic with permit acquisition. Requests still
    // parsing when the signal arrives cannot slip into the engine afterwards.
    in_flight.close();
    eprintln!(
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
    Ok(())
}
