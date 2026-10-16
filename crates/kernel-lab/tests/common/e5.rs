//! Counter-free diagnostics are external lab cubins, never catalog entries.
use kernel_lab::{LabResult, Session};
use runtime_cuda::{DeviceBuffer, Module};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// External derivatives need their generated source and disassembly bound
/// to the exact cubin. A serving lock cannot identify a diagnostic artifact.
pub fn artifact() -> LabResult<Value> {
    let cubin =
        std::env::var_os("CHOPINFER_E5_DIAGNOSTIC_CUBIN").ok_or("missing diagnostic cubin")?;
    let cubin = std::path::PathBuf::from(cubin);
    let bytes = std::fs::read(&cubin).map_err(|e| e.to_string())?;
    artifact_bytes(&cubin, &bytes)
}

fn artifact_bytes(cubin: &std::path::Path, cubin_bytes: &[u8]) -> LabResult<Value> {
    let manifest = std::env::var_os("CHOPINFER_E5_DIAGNOSTIC_MANIFEST")
        .ok_or("diagnostic cubin requires its source/SASS manifest")?;
    let manifest = std::path::PathBuf::from(manifest);
    let report: Value =
        serde_json::from_slice(&std::fs::read(&manifest).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let evidence = manifest
        .parent()
        .ok_or("diagnostic manifest has no parent")?;
    for (field, bytes) in [
        ("sha256", cubin_bytes.to_vec()),
        (
            "source_sha256",
            std::fs::read(evidence.join("diagnostics.cu")).map_err(|e| e.to_string())?,
        ),
        (
            "sass_sha256",
            std::fs::read(evidence.join("sass.txt")).map_err(|e| e.to_string())?,
        ),
    ] {
        if report[field].as_str() != Some(digest(&bytes).as_str()) {
            return Err(format!("diagnostic {field} differs from its manifest"));
        }
    }
    let inputs = report["inputs"]
        .as_object()
        .ok_or("missing diagnostic source inputs")?;
    if inputs.is_empty() || report["diagnostic_only"] != true {
        return Err("diagnostic manifest must identify its source inputs".into());
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (name, expected) in inputs {
        let path = std::path::Path::new(name);
        if !path
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
        {
            return Err("diagnostic source input must be repository-relative".into());
        }
        let actual = digest(&std::fs::read(root.join(path)).map_err(|e| e.to_string())?);
        if expected.as_str() != Some(actual.as_str()) {
            return Err(format!("diagnostic source input changed: {name}"));
        }
    }
    let ordered = inputs.iter().collect::<std::collections::BTreeMap<_, _>>();
    let closure = digest(&serde_json::to_vec(&ordered).map_err(|e| e.to_string())?);
    if report["source_closure_sha256"].as_str() != Some(closure.as_str()) {
        return Err("diagnostic source closure differs from its manifest".into());
    }
    let sass = std::process::Command::new("cuobjdump")
        .arg("--dump-sass")
        .arg(cubin)
        .output()
        .map_err(|e| e.to_string())?;
    if !sass.status.success()
        || report["sass_sha256"].as_str() != Some(digest(&sass.stdout).as_str())
    {
        return Err("loaded diagnostic cubin SASS differs from its manifest".into());
    }
    Ok(
        json!({"cubin_sha256": report["sha256"], "source_sha256": report["source_sha256"],
        "source_closure_sha256": closure, "sass_sha256": report["sass_sha256"],
        "provenance_status": "verified_report_artifacts", "diagnostic_only": true}),
    )
}

pub fn module(s: &Session) -> LabResult<Option<Module>> {
    let Some(path) = std::env::var_os("CHOPINFER_E5_DIAGNOSTIC_CUBIN") else {
        return Ok(None);
    };
    let path = std::path::PathBuf::from(path)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !path.starts_with("/data") {
        return Err("E5 cubins must stay under /data".into());
    }
    let identity = artifact()?;
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if identity["cubin_sha256"].as_str() != Some(digest(&bytes).as_str()) {
        return Err("diagnostic cubin changed before module loading".into());
    }
    s.context
        .load_module(&bytes)
        .map(Some)
        .map_err(|e| e.to_string())
}

pub fn trace(
    s: &Session,
    buffer: &DeviceBuffer,
    ctas: usize,
    stride: usize,
    stages: &[(usize, String)],
) -> LabResult<Value> {
    let mut bytes = vec![0; ctas * stride * 8];
    buffer
        .copy_to_bytes(&mut bytes, &s.stream)
        .map_err(|e| e.to_string())?;
    let stamps = bytes
        .chunks_exact(8)
        .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut raw = Vec::with_capacity(ctas);
    let mut durations = vec![Vec::with_capacity(ctas); stages.len() - 1];
    for cta in stamps.chunks_exact(stride) {
        let row = stages.iter().map(|(i, _)| cta[*i]).collect::<Vec<_>>();
        if row.contains(&0) || row.windows(2).any(|w| w[1] < w[0]) || row.last() == row.first() {
            return Err("missing or unordered E5 globaltimer stamps".into());
        }
        for (i, pair) in row.windows(2).enumerate() {
            durations[i].push(pair[1] - pair[0]);
        }
        raw.push(row);
    }
    let intervals = durations
        .iter_mut()
        .enumerate()
        .map(|(i, values)| {
            values.sort_unstable();
            json!({"from": stages[i].1, "to": stages[i + 1].1,
                "median_ns": values[values.len() / 2],
                "p90_ns": values[values.len() * 9 / 10], "max_ns": values.last()})
        })
        .collect::<Vec<_>>();
    Ok(
        json!({"clock": "globaltimer", "observer": "thread 0 of each CTA",
        "stages": stages.iter().map(|(_, s)| s).collect::<Vec<_>>(),
        "intervals": intervals, "cta_stamps_ns": raw,
        "interpretation": "instrumented control-flow intervals; loads/stores may still be outstanding"}),
    )
}
