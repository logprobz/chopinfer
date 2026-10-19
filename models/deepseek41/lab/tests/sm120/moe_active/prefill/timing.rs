use super::*;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};

/// Five interleaved rounds, five samples per arm and two replays per sample.
/// Keep every sample so a subsequent crossover decision can inspect noise.
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
    for round in 0..5 {
        let mut values = [Vec::new(), Vec::new()];
        for sample in 0..5 {
            for step in 0..2 {
                let arm = (round + sample + step) % 2;
                start.record(&session.stream).map_err(|e| e.to_string())?;
                for _ in 0..2 {
                    work(arm, &session.stream)?;
                }
                end.record(&session.stream).map_err(|e| e.to_string())?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                values[arm].push(start.elapsed_ms(&end).map_err(|e| e.to_string())? / 2.0);
            }
        }
        for arm in 0..2 {
            medians[arm].push(median(&values[arm]));
        }
        samples.push(values);
    }
    Ok(
        serde_json::json!({"ms":medians.map(|v| median(&v)), "round_samples_ms":samples, "replays_per_sample":2}),
    )
}
fn median(values: &[f32]) -> f32 {
    let mut values = values.to_vec();
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

pub fn calibration(session: &Session) -> LabResult<serde_json::Value> {
    let mut a = Arena {
        session: Some(session),
        next: 0,
        buffers: BTreeMap::new(),
    };
    let output = a.filled(752 * 256 * 4, 0)?;
    let operands = a.bytes(
        &[0x3f00_3f80_u32; 8]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    let memory = a.filled(512 << 20, 91)?;
    let bindings = Bindings {
        buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
    };
    let mut result = serde_json::Map::new();
    for (name, symbol, args, work) in [
        (
            "bf16_tflops",
            grouped::MMA,
            vec![
                Argument::DeviceBuffer(output.handle),
                Argument::DeviceBuffer(operands.handle),
                Argument::U32(1024),
            ],
            752.0 * 8.0 * 8.0 * 1024.0 * 4096.0 / 1e9,
        ),
        (
            "dram_gb_s",
            grouped::READ,
            vec![
                Argument::DeviceBuffer(output.handle),
                Argument::DeviceBuffer(memory.handle),
                Argument::U64((512 << 20) / 16),
            ],
            (512_u64 << 20) as f64 / 1e6,
        ),
        (
            "l2_gb_s",
            grouped::READ,
            vec![
                Argument::DeviceBuffer(output.handle),
                Argument::DeviceBuffer(memory.handle),
                Argument::U64((32 << 20) / 16),
            ],
            (32_u64 << 20) as f64 / 1e6,
        ),
    ] {
        let (_, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol)
            .ok_or("missing throughput calibration")?;
        let function = session
            .load_cubin(bytes)?
            .function(symbol)
            .map_err(|e| e.to_string())?;
        let plan = LaunchPlan {
            grid: [752, 1, 1],
            block: [256, 1, 1],
            cluster: [1; 3],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            pdl: true,
        };
        let args = ArgumentList(args);
        let graph =
            session.capture(|s| launch(&session.context, &function, s, plan, &args, &bindings))?;
        let measurement = pair(session, |_, s| graph.launch(s).map_err(|e| e.to_string()))?;
        let ms = measurement["ms"][0]
            .as_f64()
            .unwrap()
            .min(measurement["ms"][1].as_f64().unwrap());
        result.insert(name.into(), serde_json::json!(work / ms));
        result.insert(format!("{name}_measurement"), measurement);
    }
    result.insert(
        "mma".into(),
        serde_json::json!("dense BF16 m16n8k16, eight independent streams"),
    );
    result.insert("required_power_limit_w".into(), serde_json::json!(250));
    Ok(result.into())
}
