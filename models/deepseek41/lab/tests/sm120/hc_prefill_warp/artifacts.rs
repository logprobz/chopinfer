//! Frozen module identity and launch-attribute checks, shared by CPU and GPU runs.
use super::catalog;
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_core::contract::LaunchPlan;
use kernel_harness::{Cubins, LabResult, Session};
use runtime_cuda::{Function, Module};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

pub const PROBE_SYMBOLS: [&str; 3] = [
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
    "chop_probe_gate_v1",
];
pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_HC_WARP_ARTIFACTS")
        .expect("frozen HC artifact directory required")
        .into()
}
fn locks() -> Vec<Value> {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/kernels.lock.d/hc_prefill_warp.lock"
    )))
    .unwrap()
}
fn probe_lock() -> Value {
    let records: Vec<Value> = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../crates/kernel-lab/kernels.lock"
    )))
    .unwrap();
    records
        .into_iter()
        .find(|r| r["name"] == "probe_sm120")
        .unwrap()
}
pub fn expected() -> BTreeMap<String, Value> {
    let mut map = BTreeMap::new();
    for record in locks() {
        map.insert(record["name"].as_str().unwrap().to_owned(), json!({
            "sha256":record["cubin_sha256"], "symbols":record["entries"].as_array().unwrap().iter().map(|e|e["symbol"].clone()).collect::<Vec<_>>(),
            "resources":record["resources"],
        }));
    }
    for symbol in [
        catalog::old_mix(64, false).0.symbol,
        catalog::old_mix(8192, false).0.symbol,
        catalog::old_expand(8192).0.symbol,
        kernels_sm120::probes::COPY.into(),
        kernels_sm120::probes::LIVE.into(),
    ] {
        let (name, bytes) = fold_cubin_for_symbol(&symbol).expect("current reference is embedded");
        let record = map
            .entry(name.into())
            .or_insert_with(|| json!({"sha256":kernel_build::digest(bytes),"symbols":[]}));
        record["symbols"]
            .as_array_mut()
            .unwrap()
            .push(json!(symbol));
    }
    map.insert(
        "probe_sm120".into(),
        json!({"sha256":probe_lock()["cubin_sha256"],"symbols":PROBE_SYMBOLS}),
    );
    map
}
pub fn export() -> LabResult<()> {
    let dir = root();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let canonical = dir.canonicalize().map_err(|e| e.to_string())?;
    if canonical.starts_with(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .map_err(|e| e.to_string())?,
    ) {
        return Err("frozen artifacts must stay outside the repository".into());
    }
    for (name, record) in expected() {
        if name == "probe_sm120" {
            let source = std::env::var_os("CHOPINFER_HC_WARP_TIMING_PROBE")
                .ok_or("original kernel-lab timing probe path required")?;
            std::fs::copy(source, dir.join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        } else if let Some((_, bytes)) =
            fold_cubin_for_symbol(record["symbols"][0].as_str().unwrap())
        {
            std::fs::write(dir.join(format!("{name}.cubin")), bytes).map_err(|e| e.to_string())?;
        }
    }
    std::fs::write(
        dir.join("modules.json"),
        serde_json::to_vec_pretty(&expected()).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(
        dir.join("coverage.json"),
        serde_json::to_vec_pretty(&super::coverage()).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    verify()
}
pub fn verify() -> LabResult<()> {
    let dir = root();
    let coverage: Value = serde_json::from_slice(
        &std::fs::read(dir.join("coverage.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if coverage != super::coverage() {
        return Err("frozen HC coverage differs from the compiled test".into());
    }
    let actual: BTreeMap<String, Value> = serde_json::from_slice(
        &std::fs::read(dir.join("modules.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if actual != expected() {
        return Err("frozen HC module inventory differs".into());
    }
    for (name, record) in actual {
        let bytes = std::fs::read(dir.join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        if json!(kernel_build::digest(&bytes)) != record["sha256"] {
            return Err(format!("frozen HC module digest differs: {name}"));
        }
    }
    Ok(())
}
pub fn cubins() -> LabResult<Cubins> {
    verify()?;
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    for (name, record) in expected() {
        let code =
            std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        let name: &'static str = Box::leak(name.into_boxed_str());
        let symbols: Vec<&'static str> = record["symbols"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                let s: &'static str = Box::leak(v.as_str().unwrap().to_owned().into_boxed_str());
                s
            })
            .collect();
        bytes.push((name, &*Box::leak(code.into_boxed_slice())));
        entries.push((name, &*Box::leak(symbols.into_boxed_slice())));
    }
    Ok(Cubins::new(
        Box::leak(bytes.into_boxed_slice()),
        Box::leak(entries.into_boxed_slice()),
    ))
}
pub fn limits() -> BTreeMap<String, u32> {
    catalog::candidates(false)
        .into_iter()
        .chain(catalog::candidates(true))
        .chain([catalog::mix_control(), catalog::expand_control()])
        .map(|e| (e.reference.symbol.clone(), e.shared_bytes()))
        .collect()
}
pub fn check(symbol: &str, plan: LaunchPlan, configured: &BTreeMap<String, u32>) -> LabResult<()> {
    let attribute = configured.get(symbol).copied().unwrap_or(48 * 1024);
    if plan.dynamic_shared_bytes > attribute {
        return Err(format!(
            "{symbol}: dynamic shared {} exceeds MAX_DYNAMIC_SHARED_SIZE_BYTES={attribute}",
            plan.dynamic_shared_bytes
        ));
    }
    if plan.cluster != [1, 1, 1] || !matches!(plan.block, [128 | 256, 1, 1]) {
        return Err(format!("invalid HC launch geometry {plan:?}"));
    }
    kernel_harness::launch::config(plan)?;
    Ok(())
}
pub struct Loaded {
    pub functions: BTreeMap<String, Function>,
    pub configured: BTreeMap<String, u32>,
    _modules: Vec<Module>,
}
impl Loaded {
    pub fn new(s: &Session) -> LabResult<Self> {
        let mut modules = Vec::new();
        let mut functions = BTreeMap::new();
        let mut configured = BTreeMap::new();
        let limits = limits();
        for (name, record) in expected() {
            let m = s.module(&name)?;
            for symbol in record["symbols"].as_array().unwrap() {
                let symbol = symbol.as_str().unwrap();
                let f = m
                    .function(symbol)
                    .map_err(|e| format!("resolve {name}:{symbol}: {e}"))?;
                if let Some(&bytes) = limits.get(symbol).filter(|&&n| n > 48 * 1024) {
                    f.set_max_dynamic_shared_memory(bytes).map_err(|e| {
                        format!("{symbol}: MAX_DYNAMIC_SHARED_SIZE_BYTES={bytes}: {e}")
                    })?;
                    configured.insert(symbol.into(), bytes);
                }
                functions.insert(symbol.into(), f);
            }
            modules.push(m);
        }
        Ok(Self {
            functions,
            configured,
            _modules: modules,
        })
    }
}
