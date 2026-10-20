use super::{NORM, WIDTH, quant, registration};
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_harness::{Cubins, LabResult};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_QUERY_PREFILL")
        .expect("staged directory required")
        .into()
}
fn read(path: &Path) -> LabResult<Vec<u8>> {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(0x20000)
        .open(path)
        .map_err(|e| e.to_string())?;
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() {
        return Err("artifact must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes)
}
pub fn stage() -> LabResult<()> {
    let dir = root();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut images = BTreeMap::<String, (Vec<u8>, Vec<String>)>::new();
    for symbol in [NORM.to_owned(), quant().symbol].into_iter().chain(
        registration::entries()?
            .into_iter()
            .map(|e| e.reference.symbol),
    ) {
        let (name, bytes) = fold_cubin_for_symbol(&symbol).ok_or("embedded entry missing")?;
        let image = images
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), Vec::new()));
        if !image.1.contains(&symbol) {
            image.1.push(symbol);
        }
    }
    for (name, symbols) in [
        (
            "query_prefill_controls_sm120",
            registration::controls()?
                .into_iter()
                .map(|e| e.reference.symbol)
                .collect(),
        ),
        (
            "probe_sm120",
            vec![
                "chop_probe_gate_v1".into(),
                "chop_probe_flush_v1".into(),
                "chop_probe_read_v1".into(),
            ],
        ),
    ] {
        images.insert(
            name.into(),
            (read(&dir.join(format!("{name}.cubin")))?, symbols),
        );
    }
    let mut manifest = Vec::new();
    for (name, (bytes, symbols)) in images {
        fs::write(dir.join(format!("{name}.cubin")), &bytes).map_err(|e| e.to_string())?;
        manifest
            .push(json!({"name":name, "sha256":kernel_build::digest(&bytes), "symbols":symbols}));
    }
    fs::write(
        dir.join("modules.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .map_err(|e| e.to_string())
}
pub fn verify() -> LabResult<()> {
    let modules: Vec<Value> =
        serde_json::from_slice(&read(&root().join("modules.json"))?).map_err(|e| e.to_string())?;
    if modules.len() != 5 {
        return Err("expected five modules".into());
    }
    let mut symbols = std::collections::BTreeSet::new();
    for module in modules {
        let name = module["name"].as_str().ok_or("module name missing")?;
        if name.contains('/') {
            return Err("invalid module name".into());
        }
        let bytes = read(&root().join(format!("{name}.cubin")))?;
        if module["sha256"] != kernel_build::digest(&bytes) {
            return Err(format!("module digest differs: {name}"));
        }
        for symbol in module["symbols"].as_array().ok_or("symbols missing")? {
            let symbol = symbol.as_str().ok_or("symbol missing")?;
            if !symbols.insert(symbol.to_owned()) {
                return Err("duplicate symbol".into());
            }
            if let Some((embedded_name, embedded)) = fold_cubin_for_symbol(symbol)
                && (embedded_name != name || embedded != bytes)
            {
                return Err(format!("served module differs: {symbol}"));
            }
        }
    }
    if symbols.len() != 9 {
        return Err("expected nine used entries".into());
    }
    Ok(())
}
pub fn module_for(symbol: &str) -> LabResult<Vec<u8>> {
    if let Some((name, _)) = fold_cubin_for_symbol(symbol) {
        read(&root().join(format!("{name}.cubin")))
    } else {
        read(&root().join("query_prefill_controls_sm120.cubin"))
    }
}
pub fn probes() -> Cubins {
    let bytes = Box::leak(
        read(&root().join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[(
            "probe_sm120",
            &[
                "chop_probe_gate_v1",
                "chop_probe_flush_v1",
                "chop_probe_read_v1",
            ],
        )],
    )
}
pub struct Case {
    pub layer: u32,
    pub input: Vec<u8>,
    pub weight: Vec<u8>,
}
pub fn corpus() -> LabResult<Vec<Case>> {
    let manifest = read(&root().join("corpus.json"))?;
    let value: Value = serde_json::from_slice(&manifest).map_err(|e| e.to_string())?;
    if value["schema"] != "chopinfer.prefill-query-corpus.v1" || value["rows"] != 8192 {
        return Err("invalid query corpus".into());
    }
    let mut cases = Vec::new();
    for layer in [0, 39] {
        let mut buffers = Vec::new();
        for (role, size) in [("input", 8192 * WIDTH * 4), ("weight", WIDTH * 2)] {
            let records = value["tensors"]
                .as_array()
                .ok_or("tensor records missing")?;
            let matches = records
                .iter()
                .filter(|t| t["layer"] == layer && t["role"] == role)
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err("missing or duplicate query tensor".into());
            }
            let tensor = matches[0];
            let name = tensor["file"].as_str().ok_or("tensor filename missing")?;
            if name.contains('/') || name == ".." {
                return Err("invalid tensor filename".into());
            }
            let bytes = read(&root().join(name))?;
            if bytes.len() != size || tensor["sha256"] != kernel_build::digest(&bytes) {
                return Err(format!("query tensor digest/size differs: {name}"));
            }
            buffers.push(bytes);
        }
        cases.push(Case {
            layer,
            input: buffers.remove(0),
            weight: buffers.remove(0),
        });
    }
    Ok(cases)
}
