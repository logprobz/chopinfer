//! Diagnostic-only eager observer. All ranks enqueue each launch before readback.
use super::{Instance, Run};
use crate::program::{Phase, Stage};
use bundle_runtime::pipeline::{self, PreparedLaunch};
use bundle_runtime::upload::DeviceInputs;
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::contract::{Argument, LaunchPlan};
use runtime_bundle_abi::{
    BundleAbiError, BundleStatusV1, BundleStreamHandleV1, DeviceBufferSpanV1,
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

#[derive(Clone)]
struct Source {
    name: String,
    specs: Vec<EntrySpec>,
    plan: LaunchPlan,
    workspace_bytes: u64,
}
impl Source {
    fn native(&self) -> &str {
        &self.specs[0].symbol
    }
    fn merge(&self) -> &str {
        &self.specs[1].symbol
    }
    fn cached(&self) -> &str {
        &self.specs[2].symbol
    }
    fn producer(&self) -> &str {
        &self.specs[3].symbol
    }
}

static COUNTS: OnceLock<Mutex<BTreeMap<String, usize>>> = OnceLock::new();
const ENV: &str = "CHOPINFER_ATTENTION_CAPTURE_DIR";
pub(super) fn enabled() -> bool {
    std::env::var_os(ENV).is_some()
}
fn armed() -> Option<(PathBuf, Value)> {
    let root = PathBuf::from(std::env::var_os(ENV)?);
    if !root.is_absolute() {
        return None;
    }
    let path = root.join("armed.json");
    if !fs::symlink_metadata(&path).ok()?.is_file() {
        return None;
    }
    let arm: Value = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
    let nonce = std::env::var("CHOPINFER_ATTENTION_CAPTURE_NONCE").ok()?;
    let name = arm["group"].as_str()?;
    if arm["nonce"] != nonce
        || nonce.len() != 32
        || name.is_empty()
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return None;
    }
    Some((root, arm))
}
pub(super) fn selected(run: &Run) -> bool {
    let Some((_, arm)) = armed() else {
        return false;
    };
    matches!(run.phase, Phase::Verify { rows: 2..=32, .. })
        && u64::from(run.inputs.live_batch()) == arm["items"].as_u64().unwrap_or(0)
        && run.phase.rows() == (if arm["items"] == 1 { 4 } else { 32 })
        && run.inputs.live_tokens() == run.phase.rows()
}
fn fail(message: impl ToString) -> String {
    message.to_string()
}
fn document(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(fail)?;
    fs::write(path, bytes).map_err(fail)
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn blob(root: &Path, data: &[u8]) -> Result<Value, String> {
    let sha = kernel_build::digest(data);
    let name = format!("blobs/{sha}.bin");
    let path = root.join(&name);
    if path.exists() {
        if fs::read(&path).map_err(fail)? != data {
            return Err("capture hash collision".into());
        }
    } else {
        fs::write(&path, data).map_err(fail)?;
    }
    Ok(json!({"file":name,"bytes":data.len(),"sha256":sha}))
}
fn span(call: &PreparedLaunch, index: usize) -> Result<DeviceBufferSpanV1, String> {
    let Some(Argument::DeviceBuffer(id)) = call.arguments.0.get(index) else {
        return Err("capture expected a buffer".into());
    };
    call.bindings.get(*id).map_err(|e| format!("{e:?}"))
}
fn read(
    instance: &Instance,
    stream: BundleStreamHandleV1,
    view: DeviceBufferSpanV1,
    length: u64,
) -> Result<Vec<u8>, String> {
    if length > view.length {
        return Err("capture exceeds declared span".into());
    }
    let mut data = vec![0; length as usize];
    instance
        .services
        .copy_device_to_host(
            stream,
            &mut data,
            DeviceBufferSpanV1::new(view.buffer, view.offset, length),
        )
        .map_err(|e| format!("{e:?}"))?;
    instance
        .services
        .synchronize_stream(stream)
        .map_err(|e| format!("{e:?}"))?;
    Ok(data)
}
fn gather(
    instance: &Instance,
    stream: BundleStreamHandleV1,
    view: DeviceBufferSpanV1,
    starts: &[u64],
    width: u64,
) -> Result<Vec<u8>, String> {
    let mut data = vec![0; starts.len() * width as usize];
    for (i, &start) in starts.iter().enumerate() {
        if start.checked_add(width).is_none_or(|end| end > view.length) {
            return Err("capture page exceeds declared span".into());
        }
        instance
            .services
            .copy_device_to_host(
                stream,
                &mut data[i * width as usize..(i + 1) * width as usize],
                DeviceBufferSpanV1::new(view.buffer, view.offset + start, width),
            )
            .map_err(|e| format!("{e:?}"))?;
    }
    instance
        .services
        .synchronize_stream(stream)
        .map_err(|e| format!("{e:?}"))?;
    Ok(data)
}
fn parameter_name(param: &EntryParam) -> Result<&str, String> {
    match param {
        EntryParam::DevicePointer { name, .. }
        | EntryParam::U32 { name }
        | EntryParam::U64 { name } => Ok(name),
        _ => Err("unsupported attention parameter".into()),
    }
}
fn scalar(call: &PreparedLaunch, index: usize) -> Result<u64, String> {
    match call.arguments.0.get(index) {
        Some(Argument::U32(v)) => Ok(u64::from(*v)),
        Some(Argument::U64(v)) => Ok(*v),
        _ => Err("capture expected scalar".into()),
    }
}
#[allow(clippy::too_many_arguments)]
fn page_map(
    batch: &[u32],
    keys: &[u32],
    tables: &[u32],
    live: usize,
    budget: usize,
    stride: usize,
    sequences: usize,
    max_pages: usize,
    page: usize,
    pages: u32,
) -> BTreeSet<u32> {
    let mut used = BTreeSet::new();
    for (row, &batch) in batch.iter().take(live).enumerate() {
        if batch as usize >= sequences {
            continue;
        }
        for &key in &keys[row * stride..row * stride + budget] {
            let logical = key as usize / page;
            if key == u32::MAX || logical >= max_pages {
                continue;
            }
            let physical = tables[batch as usize * max_pages + logical];
            if physical < pages {
                used.insert(physical);
            }
        }
    }
    used
}

fn capture_inputs(
    instance: &Instance,
    stream: BundleStreamHandleV1,
    call: &PreparedLaunch,
    source: &Source,
    root: &Path,
) -> Result<Value, String> {
    let params = &source.specs[0].params;
    if source.plan.grid[0] != call.rows {
        return Err("capture row plan differs".into());
    }
    if call.arguments.0.len() < params.len() {
        return Err("attention source arity mismatch".into());
    }
    let mut scalars = BTreeMap::new();
    let mut at = BTreeMap::new();
    for (i, param) in params.iter().enumerate() {
        let name = parameter_name(param)?;
        if !matches!(param, EntryParam::DevicePointer { .. }) {
            scalars.insert(name.to_owned(), scalar(call, i)?);
        } else {
            at.insert(name, i);
        }
    }
    let original_scalars = scalars.clone();
    let rows = u64::from(call.rows);
    let copy = |name: &str, length: u64| read(instance, stream, span(call, at[name])?, length);
    let mut data = BTreeMap::<String, Vec<u8>>::new();
    for (name, length) in [
        ("q", rows * 16 * 512 * 2),
        ("row_batch", rows * 4),
        ("live_rows", 4),
        ("window_item_slots", scalars["window_items"] * 4),
        ("window_keys", rows * scalars["window_key_stride"] * 4),
    ] {
        data.insert(name.into(), copy(name, length)?);
    }
    let batch = words(&data["row_batch"]);
    let live = words(&data["live_rows"])[0] as usize;
    if live > call.rows as usize {
        return Err("capture live rows exceed launch rows".into());
    }
    let original_slots = words(&data["window_item_slots"]);
    let used_slots = batch
        .iter()
        .take(live)
        .filter_map(|&b| original_slots.get(b as usize))
        .copied()
        .filter(|&slot| u64::from(slot) < scalars["window_sequences"])
        .collect::<BTreeSet<_>>();
    let slots = used_slots
        .iter()
        .enumerate()
        .map(|(i, &slot)| (slot, i as u32))
        .collect::<BTreeMap<_, _>>();
    let code_width = scalars["window_slots"] * 512;
    let scale_width = scalars["window_slots"] * 16;
    for (name, stride, width) in [
        ("window_codes", "window_code_stride", code_width),
        ("window_scales", "window_scale_stride", scale_width),
    ] {
        let starts = used_slots
            .iter()
            .map(|&s| u64::from(s) * scalars[stride])
            .collect::<Vec<_>>();
        let mut content = gather(instance, stream, span(call, at[name])?, &starts, width)?;
        if content.is_empty() {
            content.resize(width as usize, 0);
        }
        data.insert(name.into(), content);
        scalars.insert(stride.into(), width);
    }
    data.insert(
        "window_item_slots".into(),
        bytes(
            &original_slots
                .iter()
                .map(|s| slots.get(s).copied().unwrap_or(u32::MAX))
                .collect::<Vec<_>>(),
        ),
    );
    scalars.insert("window_sequences".into(), used_slots.len().max(1) as u64);
    let mut original_tables = Vec::new();
    let mut selected_pages = BTreeSet::new();
    let page = if source.name.contains("_p32") { 32 } else { 64 };
    if at.contains_key("compressed_pool") {
        let keys = copy(
            "compressed_keys",
            rows * scalars["compressed_key_stride"] * 4,
        )?;
        original_tables = copy(
            "block_tables",
            scalars["compressed_sequences"] * scalars["max_pages"] * 4,
        )?;
        let tables = words(&original_tables);
        selected_pages = page_map(
            &batch,
            &words(&keys),
            &tables,
            live,
            scalars["compressed_budget"] as usize,
            scalars["compressed_key_stride"] as usize,
            scalars["compressed_sequences"] as usize,
            scalars["max_pages"] as usize,
            page,
            scalars["compressed_pages"] as u32,
        );
        let map = selected_pages
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as u32))
            .collect::<BTreeMap<_, _>>();
        let starts = selected_pages
            .iter()
            .map(|&p| u64::from(p) * scalars["page_stride"])
            .collect::<Vec<_>>();
        let width = page as u64 * 288;
        let mut pool = gather(
            instance,
            stream,
            span(call, at["compressed_pool"])?,
            &starts,
            width,
        )?;
        if pool.is_empty() {
            pool.resize(width as usize, 0);
        }
        data.insert("compressed_pool".into(), pool);
        data.insert("compressed_keys".into(), keys);
        data.insert(
            "block_tables".into(),
            bytes(
                &tables
                    .iter()
                    .map(|p| map.get(p).copied().unwrap_or(u32::MAX))
                    .collect::<Vec<_>>(),
            ),
        );
        scalars.insert("page_stride".into(), width);
        scalars.insert(
            "compressed_pages".into(),
            selected_pages.len().max(1) as u64,
        );
    }
    let mut buffers = serde_json::Map::new();
    for (name, data) in data {
        buffers.insert(name, blob(root, &data)?);
    }
    Ok(
        json!({"source_name":source.name,"rows":call.rows,"live_rows":live,"page":page,"scalars":scalars,
        "original_scalars":original_scalars,"buffers":buffers,"original_window_items":blob(root,&bytes(&original_slots))?,
        "original_tables":blob(root,&original_tables)?,"physical_window_slots":used_slots,"physical_pages":selected_pages,
        "served_first_symbol":call.symbol,"source_plan":{"grid":source.plan.grid,
        "block":source.plan.block},"numerical_splits":8}),
    )
}

pub(super) fn launch(
    instance: &mut Instance,
    run: &Run,
    streams: &[BundleStreamHandleV1],
    programs: &[Vec<PreparedLaunch>],
    collectives: &[Vec<bool>],
) -> Result<(), BundleAbiError> {
    inner(instance, run, streams, programs, collectives).map_err(|message| {
        instance.log(&format!("attention capture: {message}"));
        instance.collectives.poison();
        BundleAbiError::Status(BundleStatusV1::InternalError)
    })
}
fn inner(
    instance: &mut Instance,
    run: &Run,
    streams: &[BundleStreamHandleV1],
    programs: &[Vec<PreparedLaunch>],
    collectives: &[Vec<bool>],
) -> Result<(), String> {
    if instance.collectives.is_poisoned()
        || instance.config.shape.ranks() != 4
        || streams.len() != 4
        || programs.len() != 4
        || run.stages.len() != 4
        || programs.iter().any(|p| p.len() != programs[0].len())
        || run
            .stages
            .iter()
            .any(|s| s != &run.stages[0] || s.len() != programs[0].len())
    {
        return Err("capture requires complete aligned TP4 programs".into());
    }
    pipeline::check_collectives(programs, collectives)?;
    if collectives.iter().any(|c| c != &collectives[0]) {
        return Err("capture collective indexes differ".into());
    }
    let (root, arm) = armed().ok_or("capture arm disappeared")?;
    let group = arm["group"].as_str().ok_or("group missing")?;
    let input = DeviceInputs::within(instance.arenas[run.arena][0], &run.layout);
    let rows = u64::from(run.phase.rows());
    let positions = words(&read(instance, streams[0], input.positions, rows * 4)?);
    let eligible = positions.iter().all(|&p| {
        u64::from(p) >= arm["context"].as_u64().unwrap_or(u64::MAX)
            && u64::from(p) < arm["context"].as_u64().unwrap_or(0) + 512
    });
    let ordinal = if eligible {
        let mut counts = COUNTS
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .map_err(fail)?;
        let count = counts.entry(group.into()).or_default();
        let old = *count;
        *count += 1;
        (old < arm["quota"].as_u64().unwrap_or(0).min(2) as usize).then_some(old)
    } else {
        None
    };
    let directory = ordinal.map(|n| root.join(format!("{group}-{n:02}")));
    if let Some(dir) = &directory {
        fs::create_dir(dir).map_err(fail)?;
        fs::create_dir_all(root.join("blobs")).map_err(fail)?;
    }
    let sources = crate::Kernels::attention_capture_sources(
        instance.config.shape.geometry,
        run.phase.rows(),
    )?
    .into_iter()
    .map(|(name, specs, plan, workspace_bytes)| Source {
        name,
        specs,
        plan,
        workspace_bytes,
    })
    .collect::<Vec<_>>();
    let mut pending: Option<(u32, Source, Value)> = None;
    let mut records = Vec::new();
    for index in 0..programs[0].len() {
        let stage = run.stages[0][index];
        let call = &programs[0][index];
        if instance.dspark.is_some()
            && let Stage::EngramRows(layer) = stage
        {
            let mapped = instance
                .dspark
                .as_ref()
                .unwrap()
                .cycle
                .layers
                .iter()
                .position(|entry| entry.layer == layer)
                .ok_or("missing mapped layer")?;
            instance
                .wait_dspark_rows(mapped)
                .map_err(|e| format!("{e:?}"))?;
        }
        let _unwinding = pipeline::PoisonOnUnwind(&instance.collectives);
        if directory.is_some()
            && pending.is_none()
            && let Stage::LatentAttention(layer @ (0 | 2 | 20)) = stage
            && let Some(source) = sources
                .iter()
                .find(|s| call.symbol == s.native() || call.symbol == s.producer())
        {
            let mut record = capture_inputs(instance, streams[0], call, source, &root)?;
            record["layer"] = json!(layer);
            pending = Some((layer, source.clone(), record));
        }
        for rank in 0..4 {
            pipeline::launch_all(
                &instance.services,
                streams[rank],
                &instance.symbols[rank],
                std::slice::from_ref(&programs[rank][index]),
            )
            .map_err(|e| format!("{e:?}"))?;
        }
        if let Some((_, source, record)) = &mut pending {
            if call.symbol == source.native() || call.symbol == source.cached() {
                let workspace = read(instance, streams[0], span(call, 7)?, source.workspace_bytes)?;
                record["expected_workspace"] = blob(&root, &workspace)?;
                record["served_body_symbol"] = json!(call.symbol);
            }
            if call.symbol == source.merge() {
                for (i, name, length) in [
                    (2, "expected_output", rows * 16 * 512 * 2),
                    (3, "sinks", 16 * 4),
                    (4, "cos_sin", rows * 64 * 4),
                ] {
                    let data = read(instance, streams[0], span(call, i)?, length)?;
                    if name == "expected_output" {
                        record[name] = blob(&root, &data)?;
                    } else {
                        record["buffers"][name] = blob(&root, &data)?;
                    }
                }
                record["served_merge_symbol"] = json!(call.symbol);
                records.push(pending.take().unwrap().2);
            }
        }
    }
    if let Some(dir) = directory {
        if pending.is_some() || records.len() != 3 {
            return Err("incomplete attention capture".into());
        }
        for &stream in streams {
            instance
                .services
                .synchronize_stream(stream)
                .map_err(|e| format!("{e:?}"))?;
        }
        document(
            &dir.join("capture.json"),
            &json!({"schema":"chopinfer.verify-attention-capture.v1","group":group,"arm":arm,
            "rows":rows,"items":run.inputs.live_batch(),"phase":"verify","rank":0,"ranks":4,"execution":"eager-current-exact",
            "request_ids":run.inputs.request_ids,"positions":positions,"token_ids":words(&read(instance,streams[0],input.token_ids,rows*4)?),
            "token_batch":words(&read(instance,streams[0],input.token_batch,rows*4)?),"slot_ids":run.inputs.slot_ids,"q_start":run.inputs.q_start,"q_len":run.inputs.q_len,
            "nonce":std::env::var("CHOPINFER_ATTENTION_CAPTURE_NONCE").map_err(fail)?,"snapshot_sha256":std::env::var("CHOPINFER_ATTENTION_CAPTURE_SNAPSHOT_SHA256").map_err(fail)?,"records":records}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn page_selection_respects_rows_tables_and_invalid_entries() {
        let batch = [0, 1, u32::MAX, 0];
        let keys = [0, 64, 0, 64, 0, 64, 0, 0];
        let tables = [8, 2, u32::MAX, 5, 99, 3];
        assert_eq!(
            page_map(&batch, &keys, &tables, 3, 2, 2, 2, 3, 64, 9),
            BTreeSet::from([8, 2, 5])
        );
        assert!(page_map(&batch, &keys, &tables, 0, 2, 2, 2, 3, 64, 9).is_empty());
    }
}
