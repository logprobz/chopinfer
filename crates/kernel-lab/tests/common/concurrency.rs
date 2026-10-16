use kernel_lab::{LabResult, Session};
use serde_json::{Value, json};
use std::collections::BTreeMap;
#[allow(dead_code)]
#[path = "evidence.rs"]
pub mod evidence;
pub use kernel_lab::poison as replay;
pub const ROWS: [usize; 3] = [64, 128, 256];
pub fn enabled(measurement: &str) -> bool {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        return false;
    }
    let selected = std::env::var("CHOPINFER_KERNEL_BENCH_VARIANT").unwrap_or_default();
    if !selected.is_empty() && selected != measurement {
        return false;
    }
    assert!(
        kernel_lab::selected_ordinals()
            .expect("GPU selection")
            .is_some(),
        "benchmark requires GPU selection"
    );
    true
}
pub fn exclusive(s: &Session) -> LabResult<()> {
    if std::env::var("CHOPINFER_KERNEL_SHARED_DEVICE").as_deref() == Ok("1") {
        Ok(())
    } else {
        evidence::exclusive(s)
    }
}
#[allow(clippy::too_many_arguments)]
pub fn record(
    s: &Session,
    measurement: &str,
    operation: &str,
    rows: usize,
    cubin: &str,
    median_us: f64,
    min_us: f64,
    mut extra: BTreeMap<String, Value>,
) -> LabResult<()> {
    extra.extend(BTreeMap::from([
        ("measurement".into(), json!(measurement)),
        (
            "measurement_label".into(),
            s.metadata["measurement_label"].clone(),
        ),
        ("gpu_ordinal".into(), json!(s.ordinal)),
        ("gpu_uuid".into(), s.metadata["uuid"].clone()),
        ("operation".into(), json!(operation)),
        ("rows".into(), json!(rows)),
        ("median_us_per_call".into(), json!(median_us)),
        ("min_us_per_call".into(), json!(min_us)),
        ("artifact".into(), evidence::identity(cubin)?),
        ("correct".into(), json!(true)),
    ]));
    exclusive(s)?;
    if std::env::var("CHOPINFER_KERNEL_SHARED_DEVICE").as_deref() == Ok("1") {
        s.receipt("concurrency", extra.clone())?;
    } else {
        evidence::receipt(s, "concurrency", extra.clone())?;
    }
    eprintln!(
        "CONC_LAB {}",
        serde_json::to_string(&extra).map_err(|e| e.to_string())?
    );
    Ok(())
}
/// Prepare every rank before any peer can launch or publish into its buffers.
#[allow(dead_code)]
pub fn collective_phase(
    sessions: &[Session],
    replays: &mut [kernel_lab::poison::Replay<'_>],
    phase: kernel_lab::poison::Phase,
    pdl: bool,
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<()> {
    kernel_lab::poison::synchronize(sessions)?;
    for replay in replays.iter() {
        replay.prepare(phase)?;
    }
    launch()?;
    kernel_lab::poison::synchronize(sessions)?;
    let mut failures = replay::Failures::default();
    for (rank, replay) in replays.iter_mut().enumerate() {
        failures.check(
            &format!("rank {rank} {}", phase.name()),
            replay.finish(phase, pdl),
        );
    }
    failures.finish()
}

#[allow(dead_code, clippy::too_many_arguments)]
pub fn measure(
    s: &Session,
    measurement: &str,
    operation: &str,
    rows: usize,
    cubin: &str,
    launch: impl Fn() -> LabResult<()>,
    verify: impl Fn() -> LabResult<()>,
    mut correctness: impl FnMut(&runtime_cuda::GraphExec) -> LabResult<BTreeMap<String, Value>>,
) -> LabResult<()> {
    exclusive(s)?;
    // time() flushes L2 before each graph. One call avoids reusing weights
    // within a graph, which would understate a layer's DRAM traffic.
    let graph = s.capture(|_| launch())?;
    let t = kernel_lab::time(s, 3, 9, |stream| {
        graph.launch(stream).map_err(|e| e.to_string())
    })?;
    verify()?;
    // Correctness fills and restoration stay outside the event interval.
    let mut replay = correctness(&graph)?;
    verify()?;
    replay.extend(BTreeMap::from([
        ("calls_per_graph".into(), json!(1)),
        ("cache_policy".into(), json!("L2 flushed before each graph")),
        ("rounds".into(), json!(9)),
    ]));
    record(
        s,
        measurement,
        operation,
        rows,
        cubin,
        f64::from(t.median_ms) * 1000.0,
        f64::from(t.min_ms) * 1000.0,
        replay,
    )
}
