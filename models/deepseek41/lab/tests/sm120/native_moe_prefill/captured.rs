//! Criterion 2 on real operands: the published r7 capture (inputs, routes
//! and the exact producer's outputs) with the sealed rank banks. Each case
//! first reproduces the captured exact intermediate, codes, scales and
//! sampled down rows bit for bit, which authenticates its banks; then the
//! checks of `checks::qualify` run, and with CHOPINFER_NATIVE_MOE_TIMING=1
//! the 8192-row chunks are timed. Worker w takes ranks w and w + 2.
use super::checks::{Host, qualify};
use super::pipeline::{Inputs, build};
use super::*;
use kernel_core::moe::RouteMetadata;
use kernel_core::precision::buffers::QuantizedWeights;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const READY: &str = "CHOPINFER_NATIVE_MOE_READY";
pub const BANKS: &str = "CHOPINFER_NATIVE_MOE_BANKS";
pub const WORKER: &str = "CHOPINFER_NATIVE_MOE_WORKER";
/// The declared layers (native_moe_prefill.md, criterion 2).
pub const LAYERS: [u32; 4] = [0, 13, 14, 39];
/// The declared chunks: (prompt, first position, rows).
pub const CHUNKS: [(&str, u64, usize); 3] = [
    ("input-0", 0, 8192),
    ("input-1", 8192, 65),
    ("input-2", 8192, 257),
];

pub fn json(path: &Path) -> LabResult<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
        .map_err(|e| e.to_string())
}

fn checked(path: &Path, expected: &Value) -> LabResult<Vec<u8>> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() || Some(meta.len()) != expected["bytes"].as_u64() {
        return Err(format!("invalid capture file {}", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if expected["sha256"] != kernel_build::digest(&bytes) {
        return Err(format!("digest mismatch {}", path.display()));
    }
    Ok(bytes)
}

/// One captured bank: its directory, manifest files and metadata, and the
/// rank-zero bank of the same step (replicated inputs and routes).
pub struct Capture {
    pub prompt: String,
    pub path: PathBuf,
    pub files: Value,
    pub meta: Value,
    pub rank0: (PathBuf, Value),
}
impl Capture {
    pub fn read(&self, name: &str) -> LabResult<Vec<u8>> {
        checked(&self.path.join(name), &self.files[name])
    }
    pub fn common(&self, name: &str) -> LabResult<Vec<u8>> {
        checked(&self.rank0.0.join(name), &self.rank0.1[name])
    }
    pub fn layer(&self) -> u32 {
        self.meta["layer"].as_u64().unwrap() as u32
    }
    pub fn rank(&self) -> u32 {
        self.meta["rank"].as_u64().unwrap() as u32
    }
    pub fn rows(&self) -> usize {
        self.meta["rows"].as_u64().unwrap() as usize
    }
    pub fn position(&self) -> u64 {
        self.meta["positions"][0].as_u64().unwrap()
    }
    pub fn samples(&self) -> Vec<usize> {
        self.meta["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect()
    }
    /// The captured per-token row hashes of a tensor.
    pub fn row_hashes(&self, name: &str) -> LabResult<Vec<String>> {
        let bytes = self.read(&format!("{name}.rows.sha256"))?;
        Ok(std::str::from_utf8(&bytes)
            .map_err(|e| e.to_string())?
            .lines()
            .map(str::to_string)
            .collect())
    }
}

/// The declared cases of `worker` from the published manifest.
pub fn captures(manifest: &Value, worker: u32) -> LabResult<Vec<Capture>> {
    if manifest["schema"] != "chopinfer.native-down-inputs.v1"
        || manifest["chunk"]["sha256_rows_equal"] != true
    {
        return Err("capture prerequisite not qualified".into());
    }
    let all = manifest["captures"].as_array().ok_or("missing captures")?;
    let by_path: BTreeMap<&str, &Value> = all
        .iter()
        .map(|v| (v["path"].as_str().unwrap_or(""), v))
        .collect();
    let mut out = Vec::new();
    for v in all {
        let path = PathBuf::from(v["path"].as_str().ok_or("missing path")?);
        let meta: Value =
            serde_json::from_slice(&checked(&path.join("down.json"), &v["files"]["down.json"])?)
                .map_err(|e| e.to_string())?;
        let (layer, rank) = (
            meta["layer"].as_u64().ok_or("layer")? as u32,
            meta["rank"].as_u64().ok_or("rank")? as u32,
        );
        let prompt = v["prompt"].as_str().ok_or("prompt")?.to_string();
        let key = (
            prompt.as_str(),
            meta["positions"][0].as_u64().ok_or("position")?,
            meta["rows"].as_u64().ok_or("rows")? as usize,
        );
        if rank % 2 != worker || !LAYERS.contains(&layer) || !CHUNKS.contains(&key) {
            continue;
        }
        if meta["experts"] != 384
            || meta["top_k"] != 6
            || meta["hidden"] != 5120
            || meta["intermediate"] != 576
        {
            return Err("invalid capture geometry".into());
        }
        let r0 = path.parent().unwrap().join(format!("moe-L{layer:02}-r0"));
        let r0_files = by_path
            .get(r0.to_str().ok_or("non-UTF8 path")?)
            .ok_or("missing rank-zero bank")?["files"]
            .clone();
        out.push(Capture {
            prompt,
            path,
            files: v["files"].clone(),
            meta,
            rank0: (r0, r0_files),
        });
    }
    out.sort_by_key(|c| (c.layer(), c.rank(), c.prompt.clone(), c.position()));
    let expected = LAYERS.len() * 2 * CHUNKS.len();
    if out.len() != expected {
        return Err(format!(
            "expected {expected} declared cases, found {}",
            out.len()
        ));
    }
    Ok(out)
}

/// The sealed banks: BANKS.json names each file's bytes and digest.
pub struct Banks {
    root: PathBuf,
    manifest: Value,
}
impl Banks {
    pub fn open(root: &Path) -> LabResult<Self> {
        Ok(Self {
            root: root.into(),
            manifest: json(&root.join("BANKS.json"))?,
        })
    }
    pub fn read(&self, layer: u32, rank: u32, gate: bool) -> LabResult<[Vec<u8>; 2]> {
        let part = if gate { "gate" } else { "down" };
        let file = |name: &str| format!("L{layer:02}-r{rank}-{part}-{name}.bin");
        Ok([
            checked(
                &self.root.join(file("weight")),
                &self.manifest["files"][file("weight")],
            )?,
            checked(
                &self.root.join(file("scale")),
                &self.manifest["files"][file("scale")],
            )?,
        ])
    }
    pub fn expected(&self, layer: u32, rank: u32, name: &str) -> Value {
        self.manifest["files"][format!("L{layer:02}-r{rank}-down-{name}.bin")].clone()
    }
}

/// `bytes` at the front of a zeroed buffer of `capacity` bytes.
fn padded(a: &mut Arena<'_>, bytes: &[u8], capacity: usize) -> LabResult<View> {
    let mut all = bytes.to_vec();
    all.resize(capacity.max(bytes.len()), 0);
    let v = a.bytes(&all)?;
    Ok(v)
}

fn authenticate(
    a: &Arena<'_>,
    c: &Capture,
    out: &super::pipeline::Outputs,
    rows: usize,
) -> LabResult<Vec<String>> {
    let mut failures = Vec::new();
    for (view, name, width) in [
        (out.intermediate, "intermediate.bf16", 6 * 576 * 2),
        (out.codes, "codes.e4m3", 6 * 576),
        (out.scales, "scales.ue8m0", 6 * 18),
    ] {
        let bytes = a.read(view)?;
        let hashes = c.row_hashes(name)?;
        if hashes.len() != rows
            || bytes
                .chunks_exact(width)
                .take(rows)
                .zip(&hashes)
                .any(|(r, h)| kernel_build::digest(r) != *h)
        {
            failures.push(format!("exact {name} differs from the capture"));
        }
    }
    let down = a.read(out.down)?;
    let width = 6 * 5120 * 4;
    let samples = c.read("down.f32.samples")?;
    let got: Vec<u8> = c
        .samples()
        .iter()
        .flat_map(|&t| down[t * width..(t + 1) * width].to_vec())
        .collect();
    if got != samples {
        failures.push("exact down differs from the captured samples".into());
    }
    Ok(failures)
}

pub fn run(session: &Session) -> LabResult<()> {
    let worker: u32 = std::env::var(WORKER)
        .map_err(|e| e.to_string())?
        .parse()
        .map_err(|_| "worker must be 0 or 1")?;
    if worker > 1 {
        return Err("worker must be 0 or 1".into());
    }
    let ready = json(Path::new(&std::env::var(READY).map_err(|e| e.to_string())?))?;
    let manifest_path = PathBuf::from(ready["path"].as_str().ok_or("missing manifest path")?);
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| e.to_string())?;
    if ready["sha256"] != kernel_build::digest(&manifest_bytes) {
        return Err("capture manifest digest mismatch".into());
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let cases = captures(&manifest, worker)?;
    let banks = Banks::open(Path::new(&std::env::var(BANKS).map_err(|e| e.to_string())?))?;
    let mut runner = Runner::new(session)?;
    let set = family::serving::target_set();
    let capacity = super::synthetic::CAPACITY as usize;
    let top = set.top_k as usize;
    let mut executed = 0;
    let mut failed = 0;
    let started = std::time::Instant::now();
    for layer in LAYERS {
        for rank in [worker, worker + 2] {
            let mut a = Arena::new(Some(session));
            let host_gate = banks.read(layer, rank, true)?;
            let host_down = banks.read(layer, rank, false)?;
            let gate = QuantizedWeights {
                codes: a.bytes(&host_gate[0])?,
                local_scales: Some(a.bytes(&host_gate[1])?),
                global_scales: None,
            };
            let down = QuantizedWeights {
                codes: a.bytes(&host_down[0])?,
                local_scales: Some(a.bytes(&host_down[1])?),
                global_scales: None,
            };
            let keep = a.next;
            for c in cases
                .iter()
                .filter(|c| c.layer() == layer && c.rank() == rank)
            {
                // The down bank must be the one the capture's producer read.
                for name in ["weight", "scale"] {
                    if c.meta["weights"][name] != banks.expected(layer, rank, name) {
                        return Err(format!(
                            "layer {layer} rank {rank}: sealed down {name} is not the captured bank"
                        ));
                    }
                }
                let rows = c.rows();
                let codes = c.common("input.e4m3")?;
                let scales = c.common("input.ue8m0")?;
                let ids = ints(&c.common("ids.i32")?);
                let route_weights = c.common("route_weights.f32")?;
                if codes.len() != rows * 5120
                    || scales.len() != rows * 160
                    || ids.len() != rows * top
                {
                    return Err("captured input extent mismatch".into());
                }
                let metadata = |name: &str| c.read(name);
                let routes = RouteMetadata {
                    counts: Some(padded(
                        &mut a,
                        &metadata("counts.i32")?,
                        (set.count as usize + 1) * 4,
                    )?),
                    offsets: padded(
                        &mut a,
                        &metadata("offsets.i32")?,
                        (set.count as usize + 1) * 4,
                    )?,
                    sorted_rows: padded(
                        &mut a,
                        &metadata("sorted_rows.i32")?,
                        (capacity * top + set.count as usize * 64) * 4,
                    )?,
                    padded_rows: padded(&mut a, &metadata("padded_rows.i32")?, 4)?,
                    active_experts: Some(padded(
                        &mut a,
                        &metadata("active.i32")?,
                        (set.count as usize + 1) * 4,
                    )?),
                };
                let inputs = Inputs {
                    codes: padded(&mut a, &codes, capacity * 5120)?,
                    scales: padded(&mut a, &scales, capacity * 160)?,
                    live: a.bytes(&(rows as u32).to_le_bytes())?,
                    route_weights: padded(&mut a, &route_weights, capacity * top * 4)?,
                    routes,
                    gate,
                    down,
                };
                let p = build(&mut a, set, capacity as u32, rows as u32, &inputs)?;
                runner.run(&a, &p.exact.launches())?;
                let auth = authenticate(&a, c, &p.exact.out, rows)?;
                let shared = c.read("shared.f32.samples")?;
                let shared: Vec<Vec<f32>> = floats(&shared)
                    .chunks_exact(5120)
                    .map(<[f32]>::to_vec)
                    .collect();
                let sampled = c.samples();
                if sampled != samples(rows) || shared.len() != sampled.len() {
                    return Err("captured sample rows differ from the declared sampling".into());
                }
                // The captured ordered sum reproduces from the captured
                // routes and shared rows (the report-only sum's model).
                let down_samples = floats(&c.read("down.f32.samples")?);
                let sums = floats(&c.read("sum.f32.samples")?);
                let mut sum_model = true;
                for (s, row) in shared.iter().enumerate() {
                    let routes: Vec<&[f32]> = (0..top)
                        .map(|k| &down_samples[(s * top + k) * 5120..(s * top + k + 1) * 5120])
                        .collect();
                    sum_model &=
                        super::oracle::ordered_sum(&routes, row) == sums[s * 5120..(s + 1) * 5120];
                }
                let host = Host {
                    codes,
                    scales,
                    ids,
                    gate: host_gate.clone(),
                    down: host_down.clone(),
                    shared,
                };
                let report = qualify(
                    &mut runner,
                    &mut a,
                    &p,
                    &inputs,
                    &host,
                    rows,
                    super::synthetic::oracle_enabled(),
                )?;
                let mut out = report.json;
                let passed = report.passed && auth.is_empty();
                out["authentication_failures"] = serde_json::json!(auth);
                out["ordered_sum_model_reproduces_capture"] = serde_json::json!(sum_model);
                if super::synthetic::timing() && rows == capacity {
                    out["timing"] = serde_json::json!({
                        "gate_up": super::timing::graphs(&mut runner, &a, &p.exact.gate, &p.native.gate)?,
                        "down": super::timing::graphs(&mut runner, &a, &p.exact.down, &p.native.down)?,
                        "both": super::timing::graphs(&mut runner, &a, &p.exact.launches(), &p.native.launches())?,
                        "tiles": tile_timing(&mut runner, &mut a, &p, &inputs)?,
                    });
                }
                for (key, value) in [
                    ("layer", serde_json::json!(layer)),
                    ("rank", serde_json::json!(rank)),
                    ("prompt", serde_json::json!(c.prompt)),
                    ("position", serde_json::json!(c.position())),
                    ("rows", serde_json::json!(rows)),
                    ("worker", serde_json::json!(worker)),
                    ("passed", serde_json::json!(passed)),
                    (
                        "elapsed_seconds",
                        serde_json::json!(started.elapsed().as_secs_f64()),
                    ),
                ] {
                    out[key] = value;
                }
                println!("native_moe_captured {out}");
                executed += 1;
                failed += usize::from(!passed);
                a.buffers.retain(|&h, _| h <= keep);
            }
        }
    }
    println!(
        "native_moe_captured_done {}",
        serde_json::json!({"worker": worker, "cases": executed, "failed": failed})
    );
    if executed != LAYERS.len() * 2 * CHUNKS.len() || failed > 0 {
        return Err(format!("{failed} of {executed} captured cases failed"));
    }
    Ok(())
}

/// Each lab tile of both projections against the exact serving op.
fn tile_timing(
    runner: &mut Runner<'_>,
    a: &mut Arena<'_>,
    p: &super::pipeline::Pipeline,
    i: &Inputs,
) -> LabResult<Value> {
    use super::pipeline::{native_buffers, queue};
    let mut out = Vec::new();
    for kind in family::KINDS {
        let gate = kind == GatedMlpProjection::GateUp;
        let shape = family::shape(p.set, kind);
        for tile in family::tiles(kind) {
            let q = queue(a, &shape, tile)?;
            let n = p.native.out;
            let buffers = if gate {
                native_buffers(i, true, i.codes, i.scales, n.intermediate, q, None)
            } else {
                native_buffers(i, false, n.codes, n.scales, n.down, q, None)
            };
            let mut ops = native_ops(&shape, tile, p.rows, &buffers, Variant::Serving)?;
            let reference = if gate {
                ops.extend(quantizer(&p.native.gate)?);
                &p.exact.gate
            } else {
                &p.exact.down
            };
            out.push(serde_json::json!({
                "kind": format!("{kind:?}"),
                "tile": format!("{tile:?}"),
                "serving_tile": *tile == family::serving::tile(kind),
                "timing": super::timing::graphs(runner, a, reference, &ops)?,
            }));
        }
    }
    Ok(serde_json::json!(out))
}

#[test]
#[ignore = "requires the capture named by CHOPINFER_NATIVE_MOE_READY and banks named by CHOPINFER_NATIVE_MOE_BANKS"]
fn native_moe_captured_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 16 << 30, run).unwrap();
}

#[test]
#[ignore = "requires the capture named by CHOPINFER_NATIVE_MOE_READY and banks named by CHOPINFER_NATIVE_MOE_BANKS"]
fn native_moe_captured_cpu_admission() {
    // The declared cases, the sealed banks' identity against the capture,
    // and the ordered-sum model, without a GPU.
    let ready = json(Path::new(&std::env::var(READY).unwrap())).unwrap();
    let bytes = std::fs::read(ready["path"].as_str().unwrap()).unwrap();
    assert_eq!(ready["sha256"], kernel_build::digest(&bytes));
    let manifest: Value = serde_json::from_slice(&bytes).unwrap();
    let banks = Banks::open(Path::new(&std::env::var(BANKS).unwrap())).unwrap();
    let mut seen = 0;
    for worker in 0..2 {
        for c in captures(&manifest, worker).unwrap() {
            for name in ["weight", "scale"] {
                assert_eq!(
                    c.meta["weights"][name],
                    banks.expected(c.layer(), c.rank(), name)
                );
            }
            assert_eq!(c.samples(), samples(c.rows()));
            seen += 1;
        }
    }
    assert_eq!(seen, LAYERS.len() * 4 * CHUNKS.len());
    println!(
        "native_moe_captured_admission {}",
        serde_json::json!({"cases": seen})
    );
}
