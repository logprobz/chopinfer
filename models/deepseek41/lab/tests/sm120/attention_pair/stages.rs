//! Isolated timings of latent query candidates and window preparation kernels.
use super::*;
use fixture::{Rng, f32s, u32s};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernels_sm120::latent::glue::Kernel;

fn call(
    spec: EntrySpec,
    plan: LaunchPlan,
    rows: u32,
    roles: &[(&str, Argument)],
) -> LabResult<Launch> {
    let arguments = spec
        .params
        .iter()
        .map(|p| {
            let name = match p {
                EntryParam::DevicePointer { name, .. }
                | EntryParam::U32 { name }
                | EntryParam::U64 { name } => name,
                _ => return Err("unsupported stage argument".into()),
            };
            roles
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, arg)| arg.clone())
                .ok_or_else(|| format!("unbound stage argument {name}"))
        })
        .collect::<LabResult<Vec<_>>>()?;
    Ok(Launch {
        symbol: spec.symbol,
        rows,
        plan,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    })
}

fn check_query(
    session: &Session,
    modules: &Modules,
    old: &Launch,
    new: &Launch,
    buffers: &BTreeMap<&str, runtime_cuda::DeviceBuffer>,
    data: &BTreeMap<&str, Vec<u8>>,
    bindings: &kernel_harness::launch::Bindings<'_>,
) -> LabResult<()> {
    let rows = data["status"].len() / 4;
    for mode in 0..6 {
        let mut input = data["query"].clone();
        let mut table = data["row_table"].clone();
        let mut status = data["status"].clone();
        let live = u32s([if mode == 5 { 0 } else { rows as u32 }]);
        let poison = vec![0xa5; data["native_q"].len()];
        match mode {
            1 => {
                let at = (fixture::HEADS - 1) * 512 * 4;
                input[at..at + 4].copy_from_slice(&f32::NAN.to_ne_bytes());
            }
            2 => {
                for col in [448, 449] {
                    input[col * 4..col * 4 + 4].copy_from_slice(&f32::MAX.to_ne_bytes());
                }
            }
            3 => status[..4].copy_from_slice(&2_u32.to_ne_bytes()),
            4 => table[..4].copy_from_slice(&f32::NAN.to_ne_bytes()),
            _ => (),
        }
        let mut outputs = Vec::new();
        for call in [old, new] {
            for (name, bytes) in [
                ("query", &input),
                ("row_table", &table),
                ("status", &status),
                ("live", &live),
                ("native_q", &poison),
            ] {
                buffers[name]
                    .copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            modules.launch(session, &session.stream, call, bindings)?;
            let mut result = Vec::new();
            for name in ["query", "native_q", "status"] {
                let mut bytes = vec![0; data[name].len()];
                buffers[name]
                    .copy_to_bytes(&mut bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
                result.push(bytes);
            }
            outputs.push(result);
        }
        if outputs[0] != outputs[1] {
            return Err(format!("query schedule changed mode {mode}"));
        }
        if (1..=4).contains(&mode) {
            let row_bytes = fixture::HEADS * 512 * 4;
            if outputs[0][0][..row_bytes] != input[..row_bytes]
                || outputs[0][1][..row_bytes / 2].iter().any(|&b| b != 0)
                || outputs[0][2][..4] == [0; 4]
            {
                return Err(format!(
                    "query failure protocol did not fire for mode {mode}"
                ));
            }
        }
        if mode == 5 && outputs[0][1] != poison {
            return Err("zero-live query wrote output".into());
        }
    }
    for name in ["query", "row_table", "status", "live", "native_q"] {
        buffers[name]
            .copy_from_bytes(&data[name], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn measure(session: &Session, modules: &Modules, case: Case) -> LabResult<()> {
    let rows = case.rows();
    let g = geometry();
    let mut rng = Rng(0x78a1_5f2e_4492_771d ^ case.context as u64);
    let latent_quant = glue::quantizers().remove(0);
    let query_values = f32s((0..rows * fixture::HEADS * fixture::WIDTH).map(|_| rng.value()));
    let norm_values = f32s((0..rows * fixture::WIDTH).map(|_| rng.value()));
    let table = |pairs: usize| f32s((0..pairs).map(|i| if i % 2 == 0 { 0.8 } else { 0.6 }));
    let data = BTreeMap::from([
        ("query_original", query_values.clone()),
        ("query", query_values),
        (
            "native_q",
            vec![0; rows * fixture::HEADS * fixture::WIDTH * 2],
        ),
        ("row_table", table(rows * 64)),
        ("norm_input", norm_values),
        ("norm_values", vec![0; rows * 512 * 4]),
        (
            "norm_weight",
            (0..512).flat_map(|_| 0x3f80_u16.to_ne_bytes()).collect(),
        ),
        ("live", u32s([rows as u32])),
        ("status", vec![0; rows * 4]),
        (
            "window_codes",
            vec![0; latent_quant.output_bytes(rows as u32) as usize],
        ),
        (
            "window_scales",
            vec![0; latent_quant.scale_bytes(rows as u32) as usize],
        ),
        (
            "positions",
            u32s((0..rows).map(|row| (case.context - 1 + row % case.per_sequence()) as u32)),
        ),
        (
            "row_batch",
            u32s((0..rows).map(|r| (r / case.per_sequence()) as u32)),
        ),
        ("first_positions", u32s(vec![0; case.sequences])),
        (
            "last_positions",
            u32s(vec![
                (case.context + case.per_sequence() - 2) as u32;
                case.sequences
            ]),
        ),
        ("window_keys", vec![0; rows * 128 * 4]),
        ("ring_codes", vec![0; case.sequences * fixture::SLOTS * 512]),
        ("ring_scales", vec![0; case.sequences * fixture::SLOTS * 16]),
    ]);
    let mut buffers = BTreeMap::new();
    for (&name, bytes) in &data {
        let buffer = session.allocate(bytes.len().max(16))?;
        buffer
            .copy_from_bytes(bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        buffers.insert(name, buffer);
    }
    let handle = |name: &str| data.keys().position(|&key| key == name).unwrap() as u64 + 1;
    let ptr = |name| Argument::DeviceBuffer(handle(name));
    let bindings = kernel_harness::launch::Bindings {
        buffers: buffers
            .iter()
            .map(|(&name, buffer)| (handle(name), buffer))
            .collect(),
    };
    let entries = glue::entries();
    let entry = |kernel| entries.iter().find(|e| e.kernel == kernel).unwrap();
    let query_rope = entry(Kernel::QueryRope);
    let mut calls = vec![(
        "query_rope",
        call(
            query_rope.spec(),
            query_rope
                .plan(rows as u32, 16)
                .map_err(|e| e.to_string())?,
            rows as u32,
            &[
                ("values", ptr("query")),
                ("native_q", ptr("native_q")),
                ("cos_sin", ptr("row_table")),
                ("live_rows", ptr("live")),
                ("heads", Argument::U32(16)),
                ("status", ptr("status")),
                ("status_row_offset", Argument::U64(0)),
            ],
        )?,
    )];
    let window = entry(Kernel::WindowRotate);
    let write = model_deepseek41_contract::latent_glue::window_write_for_storage(
        g,
        model_deepseek41_contract::latent_glue::ring(g),
        model_deepseek41_contract::state::KvStorage::Native,
    );
    calls.push((
        "window_norm_rope",
        call(
            window.spec(),
            window.plan(rows as u32, 1).map_err(|e| e.to_string())?,
            rows as u32,
            &[
                ("input", ptr("norm_input")),
                ("weight", ptr("norm_weight")),
                ("values", ptr("norm_values")),
                ("cos_sin", ptr("row_table")),
                ("live_rows", ptr("live")),
                ("eps_bits", Argument::U32(write.epsilon_bits)),
                ("status", ptr("status")),
                ("status_row_offset", Argument::U64(0)),
            ],
        )?,
    ));
    calls.push((
        "window_quantize",
        call(
            latent_quant.spec(),
            latent_quant.plan(rows as u32).map_err(|e| e.to_string())?,
            rows as u32,
            &[
                ("input", ptr("norm_values")),
                ("output", ptr("window_codes")),
                ("scales", ptr("window_scales")),
                ("live_rows", ptr("live")),
                ("status", ptr("status")),
                ("status_row_offset", Argument::U64(0)),
            ],
        )?,
    ));
    let keys = entry(Kernel::WindowKeys);
    calls.push((
        "window_keys",
        call(
            keys.spec(),
            keys.plan(rows as u32, 1).map_err(|e| e.to_string())?,
            rows as u32,
            &[
                ("positions", ptr("positions")),
                ("row_batch", ptr("row_batch")),
                ("first_positions", ptr("first_positions")),
                ("output", ptr("window_keys")),
                ("live_rows", ptr("live")),
                ("sequences", Argument::U32(case.sequences as u32)),
                ("window", Argument::U32(128)),
                ("slots", Argument::U32(fixture::SLOTS as u32)),
                ("circular", Argument::U32(1)),
            ],
        )?,
    ));
    let store = entry(Kernel::WindowStore);
    calls.push((
        "window_store",
        call(
            store.spec(),
            store.plan(rows as u32, 1).map_err(|e| e.to_string())?,
            rows as u32,
            &[
                ("input_codes", ptr("window_codes")),
                ("input_scales", ptr("window_scales")),
                ("positions", ptr("positions")),
                ("row_batch", ptr("row_batch")),
                ("first_positions", ptr("first_positions")),
                ("last_positions", ptr("last_positions")),
                ("codes", ptr("ring_codes")),
                ("scales", ptr("ring_scales")),
                ("live_rows", ptr("live")),
                ("sequences", Argument::U32(case.sequences as u32)),
                ("slots", Argument::U32(fixture::SLOTS as u32)),
                ("circular", Argument::U32(1)),
                ("code_stride", Argument::U64((fixture::SLOTS * 512) as u64)),
                ("scale_stride", Argument::U64((fixture::SLOTS * 16) as u64)),
                ("status", ptr("status")),
                ("status_row_offset", Argument::U64(0)),
            ],
        )?,
    ));
    let mut parallel = calls[0].1.clone();
    let query_candidate = candidate::query();
    parallel.symbol = query_candidate.symbol.clone();
    parallel.plan = query_candidate
        .plan(rows as u32, 16)
        .map_err(|e| e.to_string())?;
    check_query(
        session,
        modules,
        &calls[0].1,
        &parallel,
        &buffers,
        &data,
        &bindings,
    )?;
    println!(
        "ATTENTION_QUERY_OK {}",
        json!({"context": case.context, "rows": rows, "verify": case.verify, "ratio": case.ratio, "modes": 6})
    );
    calls.insert(1, ("query_rope_candidate", parallel));
    for (stage, call) in calls {
        let timing = kernel_harness::time_prepared(
            session,
            2,
            9,
            kernel_harness::eviction::EvictionMode::Write,
            |stream| {
                if stage.starts_with("query_rope") {
                    buffers["query"]
                        .copy_range_from_async(
                            0,
                            &buffers["query_original"],
                            0,
                            data["query"].len(),
                            stream,
                        )
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            },
            |stream| modules.launch(session, stream, &call, &bindings),
        )?;
        println!(
            "ATTENTION_STAGE {}",
            json!({
                "stage": stage, "context": case.context, "sequences": case.sequences,
                "rows": rows, "verify": case.verify, "ratio": case.ratio,
                "median_us": timing.median_ms * 1000.0,
            })
        );
    }
    let mut status = vec![0; rows * 4];
    buffers["status"]
        .copy_to_bytes(&mut status, &session.stream)
        .map_err(|e| e.to_string())?;
    if status.iter().any(|&value| value != 0) {
        return Err("query or window timing produced a row error".into());
    }
    Ok(())
}
