use super::catalog;
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_harness::{Cubins, LabResult};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};
pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_HC_PREFILL_FUSED")
        .expect("staged directory required")
        .into()
}
pub fn read(name: &str) -> LabResult<Vec<u8>> {
    if name.contains('/') || name == ".." {
        return Err("invalid artifact name".into());
    }
    let p = root().join(name);
    if p.ancestors().any(|p| p.is_symlink()) || !p.is_file() {
        return Err("regular artifact required".into());
    }
    fs::read(p).map_err(|e| e.to_string())
}
pub fn symbols() -> Vec<String> {
    let mut symbols: Vec<_> = catalog::ROWS
        .into_iter()
        .flat_map(|r| [catalog::expand(r).0.symbol, catalog::mix(r).0.symbol])
        .collect();
    symbols.push(catalog::norm().reference.symbol);
    symbols.sort();
    symbols.dedup();
    symbols
}
pub fn stage() -> LabResult<()> {
    let mut images = BTreeMap::<String, (Vec<u8>, Vec<String>)>::new();
    for symbol in symbols() {
        let (name, bytes) = fold_cubin_for_symbol(&symbol)
            .ok_or_else(|| format!("baseline entry missing: {symbol}"))?;
        let item = images
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), vec![]));
        item.1.push(symbol);
    }
    images.insert(
        catalog::MODULE.into(),
        (
            read(&format!("{}.cubin", catalog::MODULE))?,
            catalog::entries()
                .into_iter()
                .chain(catalog::controls())
                .map(|e| e.symbol)
                .collect(),
        ),
    );
    images.insert(
        "probe_sm120".into(),
        (
            read("probe_sm120.cubin")?,
            [
                "chop_probe_gate_v1",
                "chop_probe_flush_v1",
                "chop_probe_read_v1",
            ]
            .map(String::from)
            .to_vec(),
        ),
    );
    let mut records = vec![];
    for (name, (bytes, symbols)) in images {
        fs::write(root().join(format!("{name}.cubin")), &bytes).map_err(|e| e.to_string())?;
        records.push(json!({"name":name,"sha256":kernel_build::digest(&bytes),"symbols":symbols}));
    }
    fs::write(
        root().join("modules.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .map_err(|e| e.to_string())
}
pub fn modules() -> LabResult<Vec<Value>> {
    serde_json::from_slice(&read("modules.json")?).map_err(|e| e.to_string())
}
pub fn verify() -> LabResult<()> {
    let expected: std::collections::BTreeSet<_> = symbols()
        .into_iter()
        .chain(
            catalog::entries()
                .into_iter()
                .chain(catalog::controls())
                .map(|e| e.symbol),
        )
        .chain(
            [
                "chop_probe_gate_v1",
                "chop_probe_flush_v1",
                "chop_probe_read_v1",
            ]
            .map(String::from),
        )
        .collect();
    let mut actual = std::collections::BTreeSet::new();
    for m in modules()? {
        let name = m["name"].as_str().ok_or("module name missing")?;
        let bytes = read(&format!("{name}.cubin"))?;
        if m["sha256"] != kernel_build::digest(&bytes) {
            return Err("module digest differs".into());
        }
        for symbol in m["symbols"].as_array().ok_or("symbol list missing")? {
            let symbol = symbol.as_str().ok_or("symbol missing")?;
            if !actual.insert(symbol.to_owned()) {
                return Err("duplicate symbol".into());
            }
            if let Some((n, b)) = fold_cubin_for_symbol(symbol)
                && (n != name || b != bytes)
            {
                return Err("served baseline bytes differ".into());
            }
        }
    }
    if actual != expected {
        return Err("used symbol inventory differs".into());
    }
    corpus()?;
    Ok(())
}
pub fn module_for(symbol: &str) -> LabResult<Vec<u8>> {
    for m in modules()? {
        if m["symbols"].as_array().unwrap().iter().any(|s| s == symbol) {
            return read(&format!("{}.cubin", m["name"].as_str().unwrap()));
        }
    }
    Err(format!("unstaged symbol: {symbol}"))
}
pub fn probes() -> Cubins {
    let bytes = Box::leak(read("probe_sm120.cubin").unwrap().into_boxed_slice());
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
pub struct Site {
    pub id: String,
    pub inputs: BTreeMap<String, Vec<u8>>,
}
pub fn corpus() -> LabResult<Vec<Site>> {
    let value: Value = serde_json::from_slice(&read("corpus.json")?).map_err(|e| e.to_string())?;
    if value["schema"] != "chopinfer.hc-prefill-fused-corpus.v1" || value["seed_rows"] != 8 {
        return Err("invalid seeded corpus".into());
    }
    let mut sites = vec![];
    for case in value["cases"].as_array().ok_or("sites missing")? {
        let mut inputs = BTreeMap::new();
        for (role, size) in [
            ("fn", 24 * 20480 * 4),
            ("base", 24 * 4),
            ("scale", 12),
            ("norm_weight", 5120 * 2),
            ("streams", 8 * 20480 * 2),
            ("sum", 8 * 5120 * 4),
            ("pre_in", 128),
            ("post_in", 128),
            ("comb_in", 512),
        ] {
            let rec = &case["inputs"][role];
            let data = read(rec["file"].as_str().ok_or("operand missing")?)?;
            if data.len() != size
                || rec["bytes"] != size
                || rec["sha256"] != kernel_build::digest(&data)
            {
                return Err(format!("operand digest or size differs: {role}"));
            }
            inputs.insert(role.into(), data);
        }
        sites.push(Site {
            id: case["id"].as_str().ok_or("site id missing")?.into(),
            inputs,
        });
    }
    if sites.iter().map(|s| s.id.as_str()).collect::<Vec<_>>() != ["site0", "site1", "site2"] {
        return Err("expected three ordered sites".into());
    }
    Ok(sites)
}
