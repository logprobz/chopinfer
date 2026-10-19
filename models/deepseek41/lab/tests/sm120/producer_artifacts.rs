//! Frozen modules for the expansion and mixer comparison.
use super::norm_quantize::PROBE_SYMBOLS;
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_harness::{Cubins, LabResult, Session};
use runtime_cuda::{Function, Module};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

pub(super) fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_PRODUCER_ARTIFACTS")
        .expect("frozen producer artifacts required")
        .into()
}
fn modules() -> BTreeMap<String, (Vec<u8>, Vec<String>)> {
    let mut modules = BTreeMap::<String, (Vec<u8>, Vec<String>)>::new();
    for symbol in super::hc_fused::symbols() {
        let (name, bytes) = fold_cubin_for_symbol(&symbol).expect("served producer symbol");
        let entry = modules
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), Vec::new()));
        if !entry.1.contains(&symbol) {
            entry.1.push(symbol);
        }
    }
    modules
}
pub(super) fn verify() -> LabResult<()> {
    let records: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root().join("modules.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut expected = modules();
    let probe = std::fs::read(root().join("probe_sm120.cubin")).map_err(|e| e.to_string())?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../crates/kernel-lab/kernels.lock"
    ))
    .unwrap();
    let pinned = lock
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "probe_sm120")
        .unwrap();
    if pinned["cubin_sha256"] != kernel_build::digest(&probe) {
        return Err("original timing probe digest differs".into());
    }
    expected.insert(
        "probe_sm120".into(),
        (probe, PROBE_SYMBOLS.iter().map(|s| (*s).into()).collect()),
    );
    let records = records.as_array().ok_or("module array required")?;
    if records.len() != expected.len() {
        return Err("producer module set differs".into());
    }
    for (name, (embedded, symbols)) in expected {
        let bytes =
            std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        let record = records
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("missing producer module")?;
        if bytes != embedded
            || record["sha256"] != kernel_build::digest(&bytes)
            || record["symbols"] != json!(symbols)
        {
            return Err(format!("producer module identity differs: {name}"));
        }
        for symbol in symbols {
            if !bytes
                .windows(symbol.len() + 1)
                .any(|s| s[..symbol.len()] == *symbol.as_bytes() && s[symbol.len()] == 0)
            {
                return Err(format!("missing symbol {symbol}"));
            }
        }
    }
    Ok(())
}
pub(super) struct Loaded {
    pub functions: Vec<Function>,
    _modules: Vec<Module>,
}
pub(super) fn load(s: &Session, symbols: &[String]) -> LabResult<Loaded> {
    let mut modules = Vec::new();
    let mut functions = Vec::new();
    for symbol in symbols {
        let (name, _) = fold_cubin_for_symbol(symbol).ok_or("unregistered producer symbol")?;
        let module = s.load_cubin(
            &std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?,
        )?;
        functions.push(module.function(symbol).map_err(|e| e.to_string())?);
        modules.push(module);
    }
    Ok(Loaded {
        functions,
        _modules: modules,
    })
}
pub(super) fn timing_cubins() -> LabResult<Cubins> {
    verify()?;
    let bytes = Box::leak(
        std::fs::read(root().join("probe_sm120.cubin"))
            .map_err(|e| e.to_string())?
            .into_boxed_slice(),
    );
    Ok(Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    ))
}
pub(super) fn random(count: usize, mut seed: u64, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed as u32 as f32) / (u32::MAX as f32) - 0.5) * scale
        })
        .collect()
}
pub(super) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
pub(super) fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| kernel_core::codec::f32_to_bf16(v).to_ne_bytes())
        .collect()
}

#[test]
#[ignore = "requires an external frozen artifact directory"]
fn frozen_modules_are_complete() {
    if std::env::var_os("CHOPINFER_PRODUCER_EXPORT").is_some() {
        let dir = root();
        std::fs::create_dir_all(&dir).unwrap();
        assert!(
            !dir.canonicalize().unwrap().starts_with(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../..")
                    .canonicalize()
                    .unwrap()
            )
        );
        let mut images = modules();
        let probe = std::fs::read(
            std::env::var_os("CHOPINFER_PRODUCER_TIMING_PROBE")
                .expect("original probe path required"),
        )
        .unwrap();
        images.insert(
            "probe_sm120".into(),
            (probe, PROBE_SYMBOLS.iter().map(|s| (*s).into()).collect()),
        );
        let mut records = Vec::new();
        for (name, (bytes, symbols)) in images {
            std::fs::write(dir.join(format!("{name}.cubin")), &bytes).unwrap();
            records
                .push(json!({"name":name,"sha256":kernel_build::digest(&bytes),"symbols":symbols}));
        }
        std::fs::write(
            dir.join("modules.json"),
            serde_json::to_vec_pretty(&records).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("coverage.json"),
            serde_json::to_vec_pretty(&json!({"HC":super::hc_fused::coverage()})).unwrap(),
        )
        .unwrap();
    }
    verify().unwrap();
    eprintln!("PRODUCER_MODULES {} PASS", modules().len() + 1);
}
