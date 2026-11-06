//! Compare the unpartitioned route retained by other topologies with the
//! adopted Flash TP4 dense prefill tail partition. The unpartitioned route
//! uses the generic fold at 85 and 127 rows, then a split tail or padded TMA
//! above 128 rows.
//! Run the ignored test on two idle GPUs, with a 295-second process timeout.
//!
//! `compact_live_rows_pair` compares that served partition with the compact
//! prefill entries (stages 2 and 3, the bundle's packed cubins) at the live
//! prefill rows, which the compact route does not serve today: a bitwise
//! pre-check of every output and status byte, clean and with a poisoned
//! final row, then three visits of an A/A control and both A/B pairs in
//! rotating order, on two idle GPUs within the same timeout.
use bundle_deepseek41::sm120::{self, mxfp8_fold as registration};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    contract::Argument,
    mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers,
};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode};
use kernels_sm120::{
    backend::Sm120,
    gemm::{mxfp8_fold::Entry, mxfp8_prefill as tiled, mxfp8_prefill_compact::Entry as Compact},
};
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Stream, TensorMap};
use serde_json::json;
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

const ROWS: [u32; 7] = [85, 127, 789, 831, 853, 1557, 1599];

struct Arena<'a> {
    session: &'a Session,
    views: Vec<(DeviceBuffer, usize, usize)>,
}
impl Arena<'_> {
    fn bytes(&mut self, data: &[u8]) -> LabResult<View> {
        let buffer = self.session.allocate(data.len())?;
        buffer
            .copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.views.push((buffer, 0, data.len()));
        Ok(View {
            handle: self.views.len() as u64,
            bytes: data.len() as u64,
        })
    }
    fn filled(&mut self, bytes: u64, value: u8) -> LabResult<View> {
        self.bytes(&vec![value; bytes as usize])
    }
    fn range(&mut self, source: View, offset: u64, bytes: u64) -> LabResult<View> {
        if offset
            .checked_add(bytes)
            .is_none_or(|end| end > source.bytes)
        {
            return Err("tail view exceeds its source".into());
        }
        let (buffer, base, _) = &self.views[source.handle as usize - 1];
        self.views
            .push((buffer.clone(), base + offset as usize, bytes as usize));
        Ok(View {
            handle: self.views.len() as u64,
            bytes,
        })
    }
    fn write(&self, view: View, data: &[u8]) -> LabResult<()> {
        let (buffer, offset, bytes) = &self.views[view.handle as usize - 1];
        if data.len() != *bytes {
            return Err("probe upload must fill its view".into());
        }
        buffer
            .copy_from_bytes_at(*offset, data, &self.session.stream)
            .map_err(|e| e.to_string())
    }
    fn read(&self, view: View) -> LabResult<Vec<u8>> {
        let (buffer, offset, bytes) = &self.views[view.handle as usize - 1];
        Ok(kernel_harness::poison::read(self.session, buffer)?[*offset..offset + bytes].to_vec())
    }
}

fn pattern(bytes: u64, salt: u32, scale: bool) -> Vec<u8> {
    (0..bytes)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(747796405)
                .wrapping_add(salt)
                .rotate_left(7);
            if scale {
                119 + (x % 9) as u8
            } else {
                (8 + x % 96) as u8 | ((x >> 8) as u8 & 128)
            }
        })
        .collect()
}

fn backend() -> LabResult<Sm120> {
    let b = registration::register(Sm120::new(), &FLASH, 4)?;
    let b = sm120::projection_reuse::register(b, &FLASH, 4)?;
    let b = sm120::prefill_matrix::register(b, &FLASH, 4)?;
    let b = sm120::prefill_tma_scales::register(b, &FLASH, 4)?;
    Ok(b.with_packed_fold_weights())
}

// These are the same registrations DenseOps prepares: register supplies the
// exact-row folds and compact entries, then reuse, matrix and staged scales
// wrap them, followed by packed weight conversion. Other served registrations
// (stream, direct, wide verify) use different backend methods or query bounds.
// Need's exactness requirements filter entries. These prefill and exact-row
// registrations provide the serving invariance requirements, so the relaxed
// lab Need selects the same entries. DenseOps' pins compare the served asks
// and each tail's route against the unpartitioned project: the generic fold
// at 85 and 127 rows, then a split tail or padded TMA above 128 rows.
// The served lowering and this probe both inspect the selected baseline symbol
// with prefill_tail_partition before retaining its full launches. Thus neither
// reselects the full part at a rounded width, which could choose another kernel.
fn calls(
    a: &mut Arena<'_>,
    backend: &Sm120,
    e: &Entry,
    rows: u32,
    b: Buffers<View>,
    partition_tail: bool,
) -> LabResult<Vec<Launch>> {
    let q = e.query;
    let op = if rows >= tiled::TILE {
        backend.mxfp8_prefill_gemm(&q, Need::rows(q.m_capacity))
    } else {
        backend.mxfp8_fold_gemm(&q, Need::rows(q.m_capacity))
    }
    .map_err(|e| e.to_string())?;
    let mut result = op
        .launches(&b, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    let partition = partition_tail
        .then(|| {
            registration::prefill_tail_partition(
                rows,
                &result.last().expect("nonempty baseline").symbol,
            )
        })
        .flatten();
    let Some(partition) = partition else {
        return Ok(result);
    };
    result.truncate(partition.full_launches);
    if partition.padded {
        result[1].plan.grid[1] = partition.full_rows / tiled::TILE;
    }
    let mut offset = partition.full_rows;
    while offset < rows {
        let count = (rows - offset).min(64);
        let output_stride = u64::from(q.n)
            * u64::from(q.groups)
            * if q.output == kernel_core::contract::DType::F32 {
                4
            } else {
                2
            };
        let buffers = Buffers {
            activations: a.range(
                b.activations,
                u64::from(offset) * u64::from(q.activation_row_stride),
                u64::from(count) * u64::from(q.activation_row_stride),
            )?,
            activation_scales: a.range(
                b.activation_scales,
                u64::from(offset) * u64::from(q.activation_row_stride) / 32,
                u64::from(count) * u64::from(q.activation_row_stride) / 32,
            )?,
            output: a.range(
                b.output,
                u64::from(offset) * output_stride,
                u64::from(count) * output_stride,
            )?,
            status_row_offset: b.status_row_offset + u64::from(offset),
            ..b
        };
        let q = registration::routed_query(q, count);
        let op = backend
            .mxfp8_fold_gemm(&q, Need::rows(q.m_capacity))
            .map_err(|e| e.to_string())?;
        let scratch = a.filled(op.info().workspace, 0xa5)?;
        result.extend(
            op.launches(&buffers, scratch, Counts::rows(count))
                .map_err(|e| e.to_string())?,
        );
        offset += count;
    }
    Ok(result)
}

struct Calls {
    launches: Vec<Launch>,
    functions: BTreeMap<String, Function>,
    maps: BTreeMap<(usize, usize), TensorMap>,
}
impl Calls {
    fn new(a: &Arena<'_>, launches: Vec<Launch>) -> LabResult<Self> {
        let mut modules = BTreeMap::new();
        let mut functions = BTreeMap::new();
        let mut maps = BTreeMap::new();
        for (i, call) in launches.iter().enumerate() {
            let (name, cubin) = sm120::fold_cubin_for_symbol(&call.symbol)
                .ok_or_else(|| format!("missing fold cubin: {}", call.symbol))?;
            if !modules.contains_key(name) {
                modules.insert(name, a.session.load_cubin(cubin)?);
            }
            let f = modules[name]
                .function(&call.symbol)
                .map_err(|e| e.to_string())?;
            if call.plan.dynamic_shared_bytes > 0 {
                f.set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            functions.insert(call.symbol.clone(), f);
            for (j, arg) in call.arguments.0.iter().enumerate() {
                if let Argument::TensorMap(spec) = arg {
                    let (buffer, offset, bytes) = &a.views[spec.base_buffer as usize - 1];
                    maps.insert(
                        (i, j),
                        a.session
                            .context
                            .tensor_map(
                                buffer,
                                *offset,
                                *bytes,
                                kernel_harness::launch::descriptor(spec)?,
                            )
                            .map_err(|e| e.to_string())?,
                    );
                }
            }
        }
        Ok(Self {
            launches,
            functions,
            maps,
        })
    }
    fn issue(&self, a: &Arena<'_>, stream: &Stream) -> LabResult<()> {
        for (i, call) in self.launches.iter().enumerate() {
            let args = call
                .arguments
                .0
                .iter()
                .enumerate()
                .map(|(j, arg)| {
                    Ok(match arg {
                        Argument::DeviceBuffer(id) => {
                            let (buffer, offset, byte_len) = &a.views[*id as usize - 1];
                            KernelArgument::Range {
                                buffer,
                                offset: *offset,
                                byte_len: *byte_len,
                            }
                        }
                        Argument::U32(v) => KernelArgument::U32(*v),
                        Argument::U64(v) => KernelArgument::U64(*v),
                        Argument::I32(v) => KernelArgument::I32(*v),
                        Argument::NullDevicePointer => KernelArgument::NullBuffer,
                        Argument::TensorMap(_) => KernelArgument::TensorMap(&self.maps[&(i, j)]),
                        Argument::DevicePointerArray(_) => {
                            return Err("unexpected pointer array in dense fold".to_string());
                        }
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            kernel_harness::launch::launch_with(
                &self.functions[&call.symbol],
                stream,
                kernel_harness::launch::config(call.plan)?,
                &args,
            )?;
        }
        Ok(())
    }
}

fn probe() -> LabResult<Cubins> {
    let path = std::env::var_os("CHOPINFER_KERNEL_PROBE_CUBIN")
        .ok_or("set CHOPINFER_KERNEL_PROBE_CUBIN to the staged probe_sm120.cubin")?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let locks: Vec<serde_json::Value> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../crates/kernel-lab/kernels.lock"
    )))
    .map_err(|e| e.to_string())?;
    let digest = locks
        .iter()
        .find(|r| r["name"] == "probe_sm120")
        .and_then(|r| r["cubin_sha256"].as_str())
        .ok_or("missing locked timing probe")?;
    if kernel_build::digest(&bytes) != digest {
        return Err("timing probe differs from its lock".into());
    }
    Ok(Cubins::new(
        Box::leak(vec![("probe_sm120", &*Box::leak(bytes.into_boxed_slice()))].into_boxed_slice()),
        &[],
    ))
}

fn gpu_snapshot(session: &Session) -> LabResult<serde_json::Value> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=enforced.power.limit,clocks.sm",
            "--format=csv,noheader,nounits",
            "--id",
            session.metadata["uuid"]
                .as_str()
                .ok_or("missing device identity")?,
        ])
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("GPU telemetry query failed".into());
    }
    let text = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    let values = text
        .trim()
        .split(',')
        .map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    if values.len() != 2 || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("GPU power and clock telemetry is unavailable".into());
    }
    Ok(json!({"enforced_power_w":values[0], "sm_mhz":values[1]}))
}

#[test]
#[ignore = "pair-lab: two idle GPUs, CUDA events, A/A control, at most five minutes"]
fn prefill_tail_folds_pair() {
    assert_eq!(
        kernel_harness::selected_ordinals()
            .unwrap()
            .expect("select two idle GPUs")
            .len(),
        2
    );
    let start = Instant::now();
    let backend = backend().unwrap();
    kernel_harness::gpu_test_multi_with(probe().unwrap(), 2 << 30, |sessions| {
        let job_start = sessions.iter().map(gpu_snapshot).collect::<LabResult<Vec<_>>>()?;
        for (session, initial) in sessions.iter().zip(&job_start) {
            let limit = initial["enforced_power_w"].as_f64().ok_or("missing enforced power limit")?;
            let snapshot = || -> LabResult<serde_json::Value> {
                let state = gpu_snapshot(session)?;
                if state["enforced_power_w"].as_f64() != Some(limit) {
                    return Err("power limit changed; pair-lab job is void".into());
                }
                Ok(state)
            };
            let mut results = Vec::new();
            for e in registration::prefill_entries(&FLASH, 4)? {
                let mut arena = Arena { session, views: Vec::new() };
                let q = e.query;
                let size = e.bytes(1);
                let weights = kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout {
                    groups: q.groups as usize, n: q.n as usize, k: q.k as usize,
                }.pack(&pattern(size.weights, 731, false)).map_err(|e| e.to_string())?;
                let weights = arena.bytes(&weights)?;
                let weight_scales = arena.bytes(&pattern(size.weight_scales, 381, true))?;
                for rows in ROWS {
                    let keep = arena.views.len();
                    let size = e.bytes(rows);
                    let mut status = vec![0; size.status as usize + 24];
                    status[..12].fill(0xa5);
                    status[size.status as usize + 12..].fill(0xa5);
                    let mut scales = pattern(size.activation_scales, 33, true);
                    let mut b = Buffers {
                        activations: arena.bytes(&pattern(size.activations, 173, false))?,
                        activation_scales: arena.bytes(&scales)?,
                        weights, weight_scales,
                        output: arena.filled(size.output, 0xff)?,
                        live_rows: arena.bytes(&rows.to_le_bytes())?,
                        status: arena.bytes(&status)?,
                        status_row_offset: 3,
                    };
                    let baseline_output = b.output;
                    let baseline_status = b.status;
                    let baseline_calls = calls(&mut arena, &backend, &e, rows, b, false)?;
                    b.output = arena.filled(size.output, 0xff)?;
                    b.status = arena.bytes(&status)?;
                    let partition_calls = calls(&mut arena, &backend, &e, rows, b, true)?;
                    let plans = [Calls::new(&arena, baseline_calls)?, Calls::new(&arena, partition_calls)?];
                    for plan in &plans { plan.issue(&arena, &session.stream)?; }
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    let expected = arena.read(baseline_output)?;
                    if expected.chunks_exact(4).any(|word| !f32::from_le_bytes(word.try_into().unwrap()).is_finite())
                        || expected.iter().all(|byte| *byte == 0)
                        || arena.read(baseline_status)? != status
                        || expected != arena.read(b.output)? || arena.read(baseline_status)? != arena.read(b.status)? {
                        return Err(format!("bitwise tail mismatch n={} k={} groups={} rows={rows}", q.n, q.k, q.groups));
                    }
                    // Poison the final row, in the second fold when the tail
                    // exceeds 64. A wrong status_row_offset must move this bit
                    // or miss the zeroed output, including at the nonzero base.
                    let poison_row = rows - 1;
                    let scale_at = poison_row as usize * q.activation_row_stride as usize / 32;
                    let clean_scale = scales[scale_at];
                    scales[scale_at] = 255;
                    arena.write(b.activation_scales, &scales)?;
                    for output in [baseline_output, b.output] {
                        arena.write(output, &vec![0xff; size.output as usize])?;
                    }
                    for plan in &plans { plan.issue(&arena, &session.stream)?; }
                    session.stream.synchronize().map_err(|e| e.to_string())?;
                    let mut poisoned_status = status.clone();
                    let status_at = (b.status_row_offset as usize + poison_row as usize) * 4;
                    poisoned_status[status_at..status_at + 4].copy_from_slice(
                        &kernel_core::mxfp8_gemm::NON_FINITE_INPUT.to_le_bytes());
                    let mut poisoned_output = expected.clone();
                    let output_stride = size.output as usize / rows as usize;
                    poisoned_output[poison_row as usize * output_stride..].fill(0);
                    for (output, flags) in [(baseline_output, baseline_status), (b.output, b.status)] {
                        if arena.read(flags)? != poisoned_status || arena.read(output)? != poisoned_output {
                            return Err(format!("poison row/status offset mismatch n={} k={} groups={} rows={rows} row={poison_row}", q.n, q.k, q.groups));
                        }
                    }
                    // Time the clean operands only, after both exactness gates.
                    scales[scale_at] = clean_scale;
                    arena.write(b.activation_scales, &scales)?;
                    arena.write(baseline_status, &status)?;
                    arena.write(b.status, &status)?;
                    let label = format!("prefill-tail-n{}-k{}-g{}-r{rows}", q.n, q.k, q.groups);
                    let mut visits = Vec::new();
                    for visit in 0..3 {
                        if start.elapsed() >= Duration::from_secs(280) { return Err("pair-lab time budget exhausted; no complete result".into()); }
                        // Alternate the control/comparison order each visit. Both A/A arms
                        // issue the very same baseline plan over the very same operands.
                        let visit_start = snapshot()?;
                        let mut pairs = Vec::new();
                        for step in 0..2 {
                            let control = (visit + step) % 2 == 0;
                            let timing = kernel_harness::with_timing_label(&format!("{label}-{}", if control { "aa" } else { "ab" }), || {
                                kernel_harness::time_ab(session, 1, 3, [EvictionMode::Read; 2], |arm, stream| plans[if control { 0 } else { arm }].issue(&arena, stream))
                            })?;
                            pairs.push((control, timing));
                        }
                        let aa = &pairs.iter().find(|(control, _)| *control).unwrap().1;
                        let ab = &pairs.iter().find(|(control, _)| !*control).unwrap().1;
                        let spread = (1.0 - aa.paired_ratio).abs() * 100.0;
                        let gain = (1.0 - ab.paired_ratio) * 100.0;
                        println!("device={} {label} visit={visit} baseline_ms={} partition_ms={} gain_pct={gain} aa_spread_pct={spread}", session.ordinal, ab.arms[0].median_ms, ab.arms[1].median_ms);
                        if spread > 3.0 { println!("{label}: wide A/A spread ({spread}%); rerun on an idle host"); }
                        let visit_end = snapshot()?;
                        visits.push(json!({"gpu_start":visit_start,"gpu_end":visit_end,"baseline_ms":ab.arms[0].median_ms,"partition_ms":ab.arms[1].median_ms,"gain_pct":gain,"aa_spread_pct":spread,"gain_exceeds_control":gain > spread}));
                    }
                    results.push(json!({"n":q.n,"k":q.k,"groups":q.groups,"rows":rows,"bitwise":true,"poison_row":poison_row,"status_row_offset":b.status_row_offset,"poison_status_checked":true,"visits":visits}));
                    drop(plans);
                    arena.views.truncate(keep);
                }
            }
            session.receipt("deepseek41-prefill-tail-folds", BTreeMap::from([("results".into(), json!(results)), ("gpu_start".into(), initial.clone()), ("gpu_end".into(), snapshot()?)]))?;
        }
        let job_end = sessions.iter().map(gpu_snapshot).collect::<LabResult<Vec<_>>>()?;
        for (before, after) in job_start.iter().zip(&job_end) {
            if before["enforced_power_w"] != after["enforced_power_w"] {
                return Err("power limit changed across the two-device job; result is void".into());
            }
        }
        sessions[0].receipt("deepseek41-prefill-tail-folds-job", BTreeMap::from([
            ("gpu_start".into(), json!(job_start)), ("gpu_end".into(), json!(job_end)),
        ]))?;
        Ok(())
    }).unwrap().expect("the two-GPU probe must execute");
    assert!(start.elapsed() < Duration::from_secs(295));
}

/// The live prefill rows the compact pair covers: the profiled unaligned
/// counts and their 64- or 128-row neighbors.
const LIVE_ROWS: [u32; 7] = [1500, 1536, 2300, 2304, 2880, 3000, 3072];

/// The bundle's two compact entries (stages 2 and 3) over `e`.
fn compact_entries(e: &Entry) -> LabResult<[Compact; 2]> {
    let mut found: Vec<Compact> = registration::compact::catalog()?
        .into_iter()
        .filter(|c| c.base.symbol == e.symbol)
        .collect();
    found.sort_by_key(|c| c.stages);
    found
        .try_into()
        .map_err(|_| format!("{}: expected compact stages 2 and 3", e.symbol))
}

/// The compact candidate's launches at `rows`, prepared at the query's
/// capacity over packed weights as the served route is.
fn compact_calls(c: &Compact, rows: u32, b: Buffers<View>) -> LabResult<Vec<Launch>> {
    Sm120::new()
        .with_mxfp8_prefill(c.base.clone())
        .with_mxfp8_prefill_compact(c.clone(), rows)
        .with_packed_fold_weights()
        .mxfp8_prefill_gemm(&c.base.query, Need::rows(c.base.query.m_capacity))
        .map_err(|e| e.to_string())?
        .launches(&b, View::EMPTY, Counts::rows(rows))
        .map_err(|e| e.to_string())
}

#[test]
fn compact_candidates_cover_every_tp4_prefill_projection_at_the_live_rows() {
    let view = |handle| View {
        handle,
        bytes: 1 << 40,
    };
    let b = Buffers {
        activations: view(1),
        activation_scales: view(2),
        weights: view(3),
        weight_scales: view(4),
        output: view(5),
        live_rows: view(6),
        status: view(7),
        status_row_offset: 3,
    };
    let entries = registration::prefill_entries(&FLASH, 4).unwrap();
    assert!(entries.len() >= 8);
    for e in &entries {
        for c in compact_entries(e).unwrap() {
            for rows in LIVE_ROWS {
                let calls = compact_calls(&c, rows, b).unwrap();
                assert!(
                    calls
                        .iter()
                        .any(|call| call.symbol.contains("_compact_m64_")
                            && call.symbol.ends_with("_packed_v1")),
                    "{} rows={rows}: {:?}",
                    e.symbol,
                    calls.iter().map(|call| &call.symbol).collect::<Vec<_>>()
                );
                for call in &calls {
                    call.plan.validate_geometry().unwrap();
                    assert_eq!(call.plan.workspace_bytes, 0);
                    assert!(
                        sm120::fold_cubin_for_symbol(&call.symbol).is_some(),
                        "{}",
                        call.symbol
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "pair-lab: two idle GPUs, CUDA events, A/A control, at most five minutes"]
fn compact_live_rows_pair() {
    assert_eq!(
        kernel_harness::selected_ordinals()
            .unwrap()
            .expect("select two idle GPUs")
            .len(),
        2
    );
    let start = Instant::now();
    let backend = backend().unwrap();
    kernel_harness::gpu_test_multi_with(probe().unwrap(), 4 << 30, |sessions| {
        let job_start = sessions.iter().map(gpu_snapshot).collect::<LabResult<Vec<_>>>()?;
        for (session, initial) in sessions.iter().zip(&job_start) {
            let limit = initial["enforced_power_w"]
                .as_f64()
                .ok_or("missing enforced power limit")?;
            let snapshot = || -> LabResult<serde_json::Value> {
                let state = gpu_snapshot(session)?;
                if state["enforced_power_w"].as_f64() != Some(limit) {
                    return Err("power limit changed; pair-lab job is void".into());
                }
                Ok(state)
            };
            let mut results = Vec::new();
            for e in registration::prefill_entries(&FLASH, 4)? {
                let compacts = compact_entries(&e)?;
                let mut arena = Arena {
                    session,
                    views: Vec::new(),
                };
                let q = e.query;
                let size = e.bytes(1);
                let weights = kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout {
                    groups: q.groups as usize,
                    n: q.n as usize,
                    k: q.k as usize,
                }
                .pack(&pattern(size.weights, 731, false))
                .map_err(|e| e.to_string())?;
                let weights = arena.bytes(&weights)?;
                let weight_scales = arena.bytes(&pattern(size.weight_scales, 381, true))?;
                for rows in LIVE_ROWS {
                    let keep = arena.views.len();
                    let size = e.bytes(rows);
                    let mut status = vec![0; size.status as usize + 24];
                    status[..12].fill(0xa5);
                    status[size.status as usize + 12..].fill(0xa5);
                    let mut scales = pattern(size.activation_scales, 33, true);
                    let input = Buffers {
                        activations: arena.bytes(&pattern(size.activations, 173, false))?,
                        activation_scales: arena.bytes(&scales)?,
                        weights,
                        weight_scales,
                        output: View::EMPTY,
                        live_rows: arena.bytes(&rows.to_le_bytes())?,
                        status: View::EMPTY,
                        status_row_offset: 3,
                    };
                    // Arm 0 is the served partition; arms 1 and 2 the compact
                    // stages, each writing its own output and status.
                    let mut arms = Vec::new();
                    for arm in 0..3 {
                        let b = Buffers {
                            output: arena.filled(size.output, 0xff)?,
                            status: arena.bytes(&status)?,
                            ..input
                        };
                        let launches = if arm == 0 {
                            calls(&mut arena, &backend, &e, rows, b, true)?
                        } else {
                            compact_calls(&compacts[arm - 1], rows, b)?
                        };
                        arms.push((b, Calls::new(&arena, launches)?));
                    }
                    let run = |arena: &Arena<'_>| -> LabResult<Vec<(Vec<u8>, Vec<u8>)>> {
                        for (_, plan) in &arms {
                            plan.issue(arena, &session.stream)?;
                        }
                        session.stream.synchronize().map_err(|e| e.to_string())?;
                        arms.iter()
                            .map(|(b, _)| Ok((arena.read(b.output)?, arena.read(b.status)?)))
                            .collect()
                    };
                    let label = format!("compact-n{}-k{}-g{}-r{rows}", q.n, q.k, q.groups);
                    let clean = run(&arena)?;
                    let (expected, expected_status) = &clean[0];
                    if expected
                        .chunks_exact(4)
                        .any(|word| !f32::from_le_bytes(word.try_into().unwrap()).is_finite())
                        || expected.iter().all(|byte| *byte == 0)
                        || *expected_status != status
                    {
                        return Err(format!("{label}: the served partition's clean output is invalid"));
                    }
                    if let Some(arm) = clean.iter().position(|outputs| outputs != &clean[0]) {
                        return Err(format!("{label}: compact stage {} differs from the served partition", arm + 1));
                    }
                    // A nonfinite scale in the final row must set its status
                    // and zero its output in every arm alike.
                    let poison_row = rows - 1;
                    let scale_at = poison_row as usize * q.activation_row_stride as usize / 32;
                    let clean_scale = scales[scale_at];
                    scales[scale_at] = 255;
                    arena.write(input.activation_scales, &scales)?;
                    for (b, _) in &arms {
                        arena.write(b.output, &vec![0xff; size.output as usize])?;
                    }
                    let poisoned = run(&arena)?;
                    let mut poisoned_status = status.clone();
                    let status_at = (input.status_row_offset as usize + poison_row as usize) * 4;
                    poisoned_status[status_at..status_at + 4]
                        .copy_from_slice(&kernel_core::mxfp8_gemm::NON_FINITE_INPUT.to_le_bytes());
                    let mut poisoned_output = expected.clone();
                    let output_stride = size.output as usize / rows as usize;
                    poisoned_output[poison_row as usize * output_stride..].fill(0);
                    if poisoned.iter().any(|(output, flags)| *output != poisoned_output || *flags != poisoned_status) {
                        return Err(format!("{label}: poisoned final row differs"));
                    }
                    scales[scale_at] = clean_scale;
                    arena.write(input.activation_scales, &scales)?;
                    for (b, _) in &arms {
                        arena.write(b.status, &status)?;
                    }
                    let mut visits = Vec::new();
                    for visit in 0..3 {
                        if start.elapsed() >= Duration::from_secs(280) {
                            return Err("pair-lab time budget exhausted; no complete result".into());
                        }
                        // Rotate the control and the two comparisons each visit. The
                        // A/A arms issue the very same served plan over the same operands.
                        let visit_start = snapshot()?;
                        let mut timings = BTreeMap::new();
                        for step in 0..3 {
                            let which = (visit + step) % 3;
                            let name = ["aa", "ab-s2", "ab-s3"][which];
                            let timing = kernel_harness::with_timing_label(&format!("{label}-{name}"), || {
                                kernel_harness::time_ab(session, 1, 3, [EvictionMode::Read; 2], |arm, stream| {
                                    let plan = if which == 0 || arm == 0 { 0 } else { which };
                                    arms[plan].1.issue(&arena, stream)
                                })
                            })?;
                            timings.insert(name, timing);
                        }
                        let spread = (1.0 - timings["aa"].paired_ratio).abs() * 100.0;
                        let mut record = json!({"gpu_start": visit_start, "aa_spread_pct": spread,
                            "served_ms": timings["ab-s2"].arms[0].median_ms});
                        for (name, stages) in [("ab-s2", 2), ("ab-s3", 3)] {
                            let ab = &timings[name];
                            let gain = (1.0 - ab.paired_ratio) * 100.0;
                            println!(
                                "device={} {label} visit={visit} stages={stages} served_ms={} compact_ms={} gain_pct={gain} aa_spread_pct={spread}",
                                session.ordinal, ab.arms[0].median_ms, ab.arms[1].median_ms
                            );
                            record[format!("s{stages}")] = json!({"compact_ms": ab.arms[1].median_ms,
                                "served_ms": ab.arms[0].median_ms, "gain_pct": gain, "gain_exceeds_control": gain > spread});
                        }
                        if spread > 3.0 {
                            println!("{label}: wide A/A spread ({spread}%); rerun on an idle host");
                        }
                        record["gpu_end"] = snapshot()?;
                        visits.push(record);
                    }
                    results.push(json!({"n": q.n, "k": q.k, "groups": q.groups, "rows": rows, "bitwise": true,
                        "poison_row": poison_row, "served_symbols": arms[0].1.launches.iter().map(|c| &c.symbol).collect::<Vec<_>>(),
                        "compact_symbols": [&arms[1].1.launches.iter().map(|c| &c.symbol).collect::<Vec<_>>(),
                                            &arms[2].1.launches.iter().map(|c| &c.symbol).collect::<Vec<_>>()],
                        "visits": visits}));
                    drop(arms);
                    arena.views.truncate(keep);
                }
            }
            session.receipt(
                "deepseek41-compact-live-rows",
                BTreeMap::from([
                    ("results".into(), json!(results)),
                    ("gpu_start".into(), initial.clone()),
                    ("gpu_end".into(), snapshot()?),
                ]),
            )?;
        }
        let job_end = sessions.iter().map(gpu_snapshot).collect::<LabResult<Vec<_>>>()?;
        for (before, after) in job_start.iter().zip(&job_end) {
            if before["enforced_power_w"] != after["enforced_power_w"] {
                return Err("power limit changed across the two-device job; result is void".into());
            }
        }
        sessions[0].receipt(
            "deepseek41-compact-live-rows-job",
            BTreeMap::from([("gpu_start".into(), json!(job_start)), ("gpu_end".into(), json!(job_end))]),
        )?;
        Ok(())
    })
    .unwrap()
    .expect("the two-GPU probe must execute");
    assert!(start.elapsed() < Duration::from_secs(295));
}
