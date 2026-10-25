//! Qualify small-row norm admissions using only the frozen serving cubins.
use bundle_deepseek41::sm120;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::row::rmsnorm_quantize_cached::Entry;
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
mod context;
mod data;
mod matrix;
mod timing;

const ROWS: [u32; 2] = [2, 4];
const WARPS: [u32; 3] = [8, 16, 32];
const OFFSET: usize = 3;
const PROBES: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
const PATTERNS: [&str; 16] = [
    "finite",
    "spread",
    "wide",
    "zero",
    "denormal",
    "maximum",
    "nan_first",
    "nan_last",
    "inf_first",
    "inf_last",
    "boundaries",
    "sticky",
    "nan_block",
    "inf_block",
    "weight_nan",
    "weight_inf",
];

#[derive(Clone)]
struct Case {
    cached: Entry,
    widen: bool,
}
impl Case {
    fn width(&self) -> usize {
        self.cached.reference.query.norm.width as usize
    }
    fn f32(&self) -> bool {
        self.cached.reference.query.norm.input == kernel_core::contract::DType::F32
    }
    fn bytes(&self) -> usize {
        if self.f32() { 4 } else { 2 }
    }
    fn label(&self) -> String {
        format!(
            "{}:widen={}",
            if self.f32() { "query" } else { "input" },
            self.widen
        )
    }
    fn symbols(&self) -> Vec<String> {
        std::iter::once(self.cached.reference.symbol.clone())
            .chain(self.cached.symbols())
            .collect()
    }
}
fn cases() -> Vec<Case> {
    vec![
        Case {
            cached: sm120::norm_quantize_cached::entry(&FLASH),
            widen: false,
        },
        Case {
            cached: sm120::norm_quantize_cached::entry(&FLASH),
            widen: true,
        },
        Case {
            cached: sm120::norm_query_cached::entry(&FLASH),
            widen: false,
        },
    ]
}
fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_SMALL_NORM_ARTIFACTS")
        .expect("frozen norm artifact directory required")
        .into()
}
type Images = BTreeMap<String, (Vec<u8>, Vec<String>)>;
fn images() -> LabResult<Images> {
    let mut images = Images::new();
    for case in cases() {
        for symbol in case.symbols() {
            let (name, bytes) = sm120::fold_cubin_for_symbol(&symbol)
                .ok_or_else(|| format!("serving cubin missing: {symbol}"))?;
            let entry = images
                .entry(name.into())
                .or_insert_with(|| (bytes.to_vec(), vec![]));
            if !entry.1.contains(&symbol) {
                entry.1.push(symbol);
            }
        }
    }
    for symbol in context::symbols() {
        let (name, bytes) =
            sm120::fold_cubin_for_symbol(&symbol).ok_or("context serving module missing")?;
        let entry = images
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), vec![]));
        if !entry.1.contains(&symbol) {
            entry.1.push(symbol);
        }
    }
    let locks: Vec<Value> = [
        include_str!("../../../../bundle/kernels.lock.d/norm_quantize.lock"),
        include_str!("../../../../bundle/kernels.lock.d/norm_quantize_cached.lock"),
        include_str!("../../../../bundle/kernels.lock.d/norm_query_cached.lock"),
        include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_expand_mix.lock"),
        include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_chain.lock"),
        include_str!("../../../../bundle/kernels.lock.d/mxfp8_fold_fused.lock"),
    ]
    .into_iter()
    .flat_map(|s| serde_json::from_str::<Vec<Value>>(s).unwrap())
    .collect();
    for (name, (bytes, symbols)) in &images {
        let record = locks
            .iter()
            .find(|r| r["name"] == *name)
            .ok_or("missing serving lock")?;
        if record["cubin_sha256"] != kernel_build::digest(bytes) {
            return Err(format!("frozen serving module differs from lock: {name}"));
        }
        for symbol in symbols {
            if !record["entries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["symbol"] == *symbol)
            {
                return Err(format!("unlocked serving symbol: {symbol}"));
            }
        }
    }
    Ok(images)
}
fn verify_artifacts() -> LabResult<()> {
    let saved: Value = serde_json::from_slice(
        &std::fs::read(root().join("modules.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let current = images()?;
    if saved.as_array().map(Vec::len) != Some(current.len()) {
        return Err("module count differs".into());
    }
    for (name, (bytes, symbols)) in current {
        let record = saved
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("module absent")?;
        if record["sha256"] != kernel_build::digest(&bytes)
            || record["symbols"] != json!(symbols)
            || std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?
                != bytes
        {
            return Err(format!("serving module changed after freeze: {name}"));
        }
    }
    Ok(())
}
#[test]
fn frozen_serving_entries_keep_the_reference_abi_and_plans() {
    assert_eq!(images().unwrap().len(), 6);
    for case in cases() {
        assert_eq!(case.cached.early_release, case.f32());
        for rows in ROWS {
            let old = case.cached.reference.plan(rows).unwrap();
            for warps in WARPS {
                let new = case.cached.plan(rows, rows, warps).unwrap();
                assert_eq!(new.grid, [1, rows, 1]);
                assert_eq!(new.block, [warps * 32, 1, 1]);
                assert_eq!(
                    (new.pdl, new.workspace_bytes),
                    (old.pdl, old.workspace_bytes)
                );
            }
        }
        let specs = case.cached.cubin_spec("norm").unwrap();
        assert!(
            specs
                .entries
                .iter()
                .all(|e| e.params == case.cached.reference.spec().params)
        );
    }
}
#[test]
#[ignore = "requires an external directory named by CHOPINFER_SMALL_NORM_ARTIFACTS"]
fn freeze_serving_modules() {
    let root = root().canonicalize().unwrap();
    let checkout = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap();
    assert!(!root.starts_with(checkout));
    let mut saved = vec![];
    for (name, (bytes, symbols)) in images().unwrap() {
        std::fs::write(root.join(format!("{name}.cubin")), &bytes).unwrap();
        saved.push(json!({"name":name,"sha256":kernel_build::digest(&bytes),"symbols":symbols}));
    }
    std::fs::write(
        root.join("modules.json"),
        serde_json::to_vec_pretty(&saved).unwrap(),
    )
    .unwrap();
    verify_artifacts().unwrap();
    eprintln!("SMALL_NORM_FROZEN modules={} PASS", saved.len());
}
fn probes() -> Cubins {
    let bytes = std::fs::read(root().join("probe_sm120.cubin")).expect("frozen timing probe");
    let lock: Vec<Value> = serde_json::from_str(include_str!(
        "../../../../../../crates/kernel-lab/kernels.lock"
    ))
    .unwrap();
    assert!(
        lock.iter().any(
            |r| r["name"] == "probe_sm120" && r["cubin_sha256"] == kernel_build::digest(&bytes)
        )
    );
    let bytes = Box::leak(bytes.into_boxed_slice());
    Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[("probe_sm120", PROBES)],
    )
}
struct Loaded {
    functions: BTreeMap<String, Function>,
    _modules: Vec<Module>,
}
impl Loaded {
    fn load(s: &Session) -> LabResult<Self> {
        verify_artifacts()?;
        let mut functions = BTreeMap::new();
        let mut modules = vec![];
        for (_, (bytes, symbols)) in images()? {
            let m = s.load_cubin(&bytes)?;
            for symbol in symbols {
                let function = m.function(&symbol).map_err(|e| e.to_string())?;
                let shared = context::dynamic_shared(&symbol);
                if shared > 0 {
                    function
                        .set_max_dynamic_shared_memory(shared)
                        .map_err(|e| e.to_string())?;
                }
                functions.insert(symbol.clone(), function);
            }
            modules.push(m);
        }
        Ok(Self {
            functions,
            _modules: modules,
        })
    }
}
fn upload(buffer: &DeviceBuffer, bytes: &[u8], stream: &Stream) -> LabResult<()> {
    buffer
        .copy_from_bytes(bytes, stream)
        .map_err(|e| e.to_string())
}
fn download(buffer: &DeviceBuffer, size: usize, stream: &Stream) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; size];
    buffer
        .copy_to_bytes(&mut bytes, stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}
struct Input {
    values: DeviceBuffer,
    weight: DeviceBuffer,
    live: DeviceBuffer,
}
struct Output {
    buffers: [DeviceBuffer; 5],
    lengths: [usize; 5],
}
impl Output {
    fn new(s: &Session, c: &Case, rows: u32) -> LabResult<Self> {
        let n = c.width() * rows as usize;
        let lengths = [
            n * c.bytes(),
            n * 4,
            n,
            n / 32,
            (rows as usize + OFFSET + 2) * 4,
        ];
        let buffers = lengths
            .into_iter()
            .map(|n| s.allocate(n))
            .collect::<LabResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| "five output buffers")?;
        Ok(Self { buffers, lengths })
    }
    fn poison(&self, s: &Session, rows: u32, sticky: bool) -> LabResult<()> {
        for i in 0..4 {
            upload(&self.buffers[i], &vec![0xa5; self.lengths[i]], &s.stream)?;
        }
        let mut flags = vec![0x1357_2468u32; rows as usize + OFFSET + 2];
        flags[OFFSET..OFFSET + rows as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.buffers[4],
            &flags
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
            &s.stream,
        )
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .zip(self.lengths)
            .map(|(b, n)| download(b, n, &s.stream))
            .collect()
    }
}
#[allow(
    clippy::too_many_arguments,
    reason = "the frozen ABI call keeps operands, launch mode and selected arm explicit"
)]
fn call(
    c: &Case,
    k: &Loaded,
    arm: usize,
    rows: u32,
    i: &Input,
    input: &DeviceBuffer,
    o: &Output,
    stream: &Stream,
    pdl: bool,
) -> LabResult<()> {
    let symbols = c.symbols();
    let mut plan = if arm == 0 {
        c.cached.reference.plan(rows)
    } else {
        c.cached.plan(rows, rows, WARPS[arm - 1])
    }
    .map_err(|e| e.to_string())?;
    plan.pdl = pdl;
    let args = [
        A::Buffer(input),
        A::Buffer(&i.weight),
        A::Buffer(&o.buffers[0]),
        if c.widen {
            A::Buffer(&o.buffers[1])
        } else {
            A::NullBuffer
        },
        A::Buffer(&o.buffers[2]),
        A::Buffer(&o.buffers[3]),
        A::Buffer(&i.live),
        A::Buffer(&o.buffers[4]),
        A::U64(OFFSET as u64),
        A::U32(c.cached.reference.query.norm.epsilon_bits),
    ];
    k.functions[&symbols[arm]]
        .launch(stream, launch::config(plan)?, &args)
        .map_err(|e| e.to_string())
}
fn compare(a: &[Vec<u8>], b: &[Vec<u8>]) -> LabResult<()> {
    for (role, (a, b)) in ["normed", "widened", "codes", "scales", "status"]
        .into_iter()
        .zip(a.iter().zip(b))
    {
        if a != b {
            return Err(format!(
                "{role} mismatch at {:?}",
                a.iter().zip(b).position(|(a, b)| a != b)
            ));
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_SMALL_NORM_ARTIFACTS"]
fn frozen_norms_bitwise_all_live_rows_and_controls_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix::run(s)?;
        s.receipt(
            "small-norm-correctness",
            value
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    });
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_SMALL_NORM_ARTIFACTS"]
fn frozen_norms_paired_timing_with_aa_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = timing::run(s)?;
        s.receipt(
            "small-norm-timing",
            value
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    });
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
