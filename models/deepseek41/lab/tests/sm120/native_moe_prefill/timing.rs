//! Paired CUDA-event timing: five interleaved rounds, three samples per arm
//! and one graph replay per sample; every sample is kept.
use super::*;

pub fn pair(
    session: &Session,
    work: impl Fn(usize, &Stream) -> LabResult<()>,
) -> LabResult<serde_json::Value> {
    let start = session.context.timed_event().map_err(|e| e.to_string())?;
    let end = session.context.timed_event().map_err(|e| e.to_string())?;
    for _ in 0..3 {
        for arm in 0..2 {
            work(arm, &session.stream)?;
        }
    }
    session.stream.synchronize().map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    let mut medians = [Vec::new(), Vec::new()];
    let mut ratios = Vec::new();
    for round in 0..5 {
        let mut values = [Vec::new(), Vec::new()];
        for sample in 0..3 {
            for step in 0..2 {
                let arm = (round + sample + step) % 2;
                start.record(&session.stream).map_err(|e| e.to_string())?;
                work(arm, &session.stream)?;
                end.record(&session.stream).map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                values[arm].push(start.elapsed_ms(&end).map_err(|e| e.to_string())?);
            }
        }
        let m = [median(&values[0]), median(&values[1])];
        ratios.push(m[1] / m[0]);
        for arm in 0..2 {
            medians[arm].push(m[arm]);
        }
        samples.push(values);
    }
    Ok(serde_json::json!({
        "arms": ["reference", "candidate"],
        "ms": medians.map(|v| median(&v)),
        "median_round_ratio": median(&ratios),
        "round_samples_ms": samples,
        "replays_per_sample": 1,
    }))
}

pub fn median(values: &[f32]) -> f32 {
    let mut values = values.to_vec();
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// Paired timing of two launch lists, each captured as one graph.
pub fn graphs(
    runner: &mut Runner<'_>,
    a: &Arena<'_>,
    reference: &[Launch],
    candidate: &[Launch],
) -> LabResult<serde_json::Value> {
    runner.ensure(reference)?;
    runner.ensure(candidate)?;
    let session = runner.session;
    let graphs = [reference, candidate]
        .iter()
        .map(|ops| session.capture(|s| runner.issue(a, ops, s)))
        .collect::<LabResult<Vec<_>>>()?;
    pair(session, |i, s| {
        graphs[i].launch(s).map_err(|e| e.to_string())
    })
}
