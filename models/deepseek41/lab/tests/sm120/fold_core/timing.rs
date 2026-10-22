use super::*;

/// Five interleaved rounds, three samples per arm and one replay per sample.
/// Keep every sample so a subsequent crossover decision can inspect noise.
pub fn pair(
    session: &Session,
    cold: bool,
    work: impl Fn(usize, &Stream) -> LabResult<()>,
) -> LabResult<serde_json::Value> {
    arms(session, cold, 2, work)
}
/// `pair` over `count` arms, rotated in the same order.
pub fn arms(
    session: &Session,
    cold: bool,
    count: usize,
    work: impl Fn(usize, &Stream) -> LabResult<()>,
) -> LabResult<serde_json::Value> {
    let eviction = if cold {
        Some(kernel_harness::eviction::Eviction::new(
            session,
            kernel_harness::eviction::EvictionMode::Read,
        )?)
    } else {
        None
    };
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    for _ in 0..3 {
        for arm in 0..count {
            work(arm, &session.stream)?;
        }
    }
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    let mut medians = vec![Vec::new(); count];
    for round in 0..5 {
        let mut values = vec![Vec::new(); count];
        for sample in 0..3 {
            for step in 0..count {
                let arm = (round + sample + step) % count;
                if let Some(e) = &eviction {
                    e.run(&session.stream)?;
                }
                start.record(&session.stream).map_err(|e| e.to_string())?;
                for _ in 0..1 {
                    work(arm, &session.stream)?;
                }
                end.record(&session.stream).map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                values[arm].push(start.elapsed_ms(&end).map_err(|e| e.to_string())?);
            }
        }
        for arm in 0..count {
            medians[arm].push(median(&values[arm]));
        }
        samples.push(values);
    }
    Ok(
        serde_json::json!({"ms":medians.iter().map(|v| median(v)).collect::<Vec<_>>(), "round_samples_ms":samples, "replays_per_sample":1}),
    )
}
fn median(values: &[f32]) -> f32 {
    let mut values = values.to_vec();
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}
