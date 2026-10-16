use kernel_lab::{LabResult, Session};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn identity(name: &str) -> LabResult<Value> {
    let mut locks: Vec<Value> =
        serde_json::from_str(include_str!("../../kernels.lock")).map_err(|e| e.to_string())?;
    let owned: Vec<Value> = serde_json::from_str(include_str!(concat!(
        env!("OUT_DIR"),
        "/hc-prefill/kernels.lock"
    )))
    .map_err(|e| e.to_string())?;
    locks.extend(owned);
    let lock = locks
        .iter()
        .find(|v| v["name"] == name)
        .ok_or("missing cubin in kernels.lock")?;
    Ok(json!({"cubin": name, "cubin_sha256": lock["cubin_sha256"],
        "source_closure_sha256": lock["source_closure_sha256"], "nvcc_sha256": lock["nvcc_sha256"]}))
}

pub fn exclusive(session: &Session) -> LabResult<()> {
    if session.metadata["measurement_label"] != "exclusive-device" {
        return Err(
            "Benchmark evidence requires exclusive GPU access; shared-device timing is refused"
                .into(),
        );
    }
    Ok(())
}

pub fn peak(session: &Session, key: &str) -> LabResult<(f64, String)> {
    let root = kernel_lab::receipt_root().join("probes");
    let mut paths = std::fs::read_dir(root)
        .map_err(|e| e.to_string())?
        .map(|e| e.map(|e| e.path()).map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    paths.sort();
    for path in paths.into_iter().rev() {
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let value: Value =
            serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if value.get("device_name") != session.metadata.get("device_name")
            || value["measurement_label"] != "exclusive-device"
        {
            continue;
        }
        if let Some(gbps) = value
            .pointer(&format!("/{key}/gb_per_second"))
            .and_then(Value::as_f64)
            && gbps.is_finite()
            && gbps > 0.0
        {
            return Ok((
                gbps * 1e9,
                format!("probes/{}", path.file_name().unwrap().to_string_lossy()),
            ));
        }
    }
    Err(format!(
        "no measured {key} peak for this GPU type; run `kernel-lab probe` first"
    ))
}

pub fn receipt(session: &Session, kind: &str, data: BTreeMap<String, Value>) -> LabResult<()> {
    let path = session.receipt(kind, data)?;
    let recorded: Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if recorded["measurement_label"] != "exclusive-device" {
        return Err(
            "another GPU process appeared during G4; receipt is shared-device and does not qualify"
                .into(),
        );
    }
    Ok(())
}
