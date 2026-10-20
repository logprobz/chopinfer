use super::{corpus, device, registration};
use kernel_harness::{Cubins, LabResult};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};
type Images = BTreeMap<String, (Vec<u8>, Vec<String>)>;
const PROBES: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_gate_release_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_HC_ROWS_ARTIFACTS")
        .expect("artifact directory required")
        .into()
}
fn exports(lock: &str, name: &str, bytes: &[u8]) -> LabResult<Vec<String>> {
    let rows: Vec<Value> = serde_json::from_str(lock).map_err(|e| e.to_string())?;
    let row = rows
        .iter()
        .find(|r| r["name"] == name && r["cubin_sha256"] == kernel_build::digest(bytes))
        .ok_or("module differs from lock")?;
    row["entries"]
        .as_array()
        .ok_or("entry array required")?
        .iter()
        .map(|e| {
            e["symbol"]
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "entry symbol required".into())
        })
        .collect()
}
pub fn images() -> LabResult<Images> {
    use bundle_deepseek41::sm120 as b;
    let g = &model_deepseek41_contract::geometry::FLASH;
    let mut result = Images::new();
    for (symbol, lock) in [
        (
            b::hc_round_expand_mix::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_expand_mix.lock"),
        ),
        (
            b::hc_round_chain::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_chain.lock"),
        ),
        (
            b::norm_quantize::entries(g)?[0].symbol.clone(),
            include_str!("../../../../bundle/kernels.lock.d/norm_quantize.lock"),
        ),
        (
            b::norm_quantize_cached::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/norm_quantize_cached.lock"),
        ),
        (
            b::hc_finish_norm::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_norm.lock"),
        ),
    ] {
        let (name, bytes) = b::fold_cubin_for_symbol(&symbol).ok_or("reference cubin missing")?;
        result.insert(name.into(), (bytes.to_vec(), exports(lock, name, bytes)?));
    }
    let candidate = std::fs::read(
        std::env::var_os("CHOPINFER_HC_ROWS_CUBIN").ok_or("candidate path required")?,
    )
    .map_err(|e| e.to_string())?;
    let symbols = exports(
        include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_rows.lock"),
        registration::CUBIN,
        &candidate,
    )?;
    let (_, embedded) = b::fold_cubin_for_symbol(&registration::entry().symbols()[0])
        .ok_or("served row producer missing")?;
    if embedded != candidate {
        return Err("served row producer differs from frozen qualification".into());
    }
    result.insert(registration::CUBIN.into(), (candidate, symbols));
    let probe = std::fs::read(
        std::env::var_os("CHOPINFER_HC_ROWS_PROBE").ok_or("kernel-lab probe required")?,
    )
    .map_err(|e| e.to_string())?;
    let symbols = exports(
        include_str!("../../../../../../crates/kernel-lab/kernels.lock"),
        "probe_sm120",
        &probe,
    )?;
    result.insert("probe_sm120".into(), (probe, symbols));
    Ok(result)
}
pub fn verify() -> LabResult<()> {
    let saved: Value = serde_json::from_slice(
        &std::fs::read(root().join("modules.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let images = images()?;
    if saved.as_array().map(Vec::len) != Some(images.len()) {
        return Err("module count differs".into());
    }
    for (name, (bytes, symbols)) in images {
        let row = saved
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == name)
            .ok_or("unfrozen module")?;
        if row["symbols"] != json!(symbols)
            || row["sha256"] != kernel_build::digest(&bytes)
            || std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?
                != bytes
        {
            return Err(format!("module changed: {name}"));
        }
    }
    Ok(())
}
pub fn timing_cubins() -> LabResult<Cubins> {
    verify()?;
    let data = Box::leak(
        std::fs::read(root().join("probe_sm120.cubin"))
            .map_err(|e| e.to_string())?
            .into_boxed_slice(),
    );
    Ok(Cubins::new(
        Box::leak(Box::new([("probe_sm120", &data[..])])),
        &[("probe_sm120", PROBES)],
    ))
}
pub fn cases() -> LabResult<(Vec<corpus::Case>, String)> {
    let (cases, complete, sha) = corpus::load()?;
    if !complete {
        return Err("complete captured corpus required".into());
    }
    let selected: Vec<_> = cases
        .into_iter()
        .filter(|c| matches!(c.rows, 8 | 32))
        .collect();
    if selected.len() != 308
        || [8, 32]
            .iter()
            .any(|&rows| selected.iter().filter(|c| c.rows == rows).count() != 154)
    {
        return Err("C8 and Verify-32 real coverage differs".into());
    }
    for case in &selected {
        case.eps_bits()?;
        let bytes = case.record["outputs"]["partials"]["bytes"]
            .as_u64()
            .ok_or("captured partial extent is missing")?;
        if ![3840, 4096]
            .map(|words| case.rows as u64 * words * 4)
            .contains(&bytes)
        {
            return Err("captured partial row layout is invalid".into());
        }
        for tile in [2, 4, 8] {
            registration::entry()
                .plan(case.rows as u32, case.rows as u32, tile)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok((selected, sha))
}
#[test]
#[ignore = "requires staged CHOPINFER_HC_ROWS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn frozen_modules_and_cases_on_cpu() {
    let export = std::env::var("CHOPINFER_HC_ROWS_EXPORT").as_deref() == Ok("1");
    let images = images().unwrap();
    if export {
        std::fs::create_dir_all(root()).unwrap();
        let mut modules = Vec::new();
        for (name, (bytes, symbols)) in &images {
            std::fs::write(root().join(format!("{name}.cubin")), bytes).unwrap();
            modules
                .push(json!({"name":name,"sha256":kernel_build::digest(bytes),"symbols":symbols}));
        }
        std::fs::write(
            root().join("modules.json"),
            serde_json::to_vec_pretty(&modules).unwrap(),
        )
        .unwrap();
    }
    verify().unwrap();
    let (cases, sha) = cases().unwrap();
    let value = json!({"cases":cases.iter().map(|c|json!({"id":c.id,"rows":c.rows})).collect::<Vec<_>>(),"corpus_sha256":sha,"variants":device::VARIANTS,"widens":[false,true],"matrix_reports":cases.len()*device::VARIANTS.len()*2,"timing_records":cases.len()*device::VARIANTS.len()*4,"rounds":7,"warmup":2,"scope":"C32 AR replay uses real Verify-32 operands; no 32-row AR capture is claimed"});
    if export {
        std::fs::write(
            root().join("coverage.json"),
            serde_json::to_vec_pretty(&value).unwrap(),
        )
        .unwrap();
    }
    let saved: Value =
        serde_json::from_slice(&std::fs::read(root().join("coverage.json")).unwrap()).unwrap();
    assert_eq!(saved, value);
    println!(
        "HC_ROWS_ARTIFACTS modules={} symbols={} cases={} PASS",
        images.len(),
        images.values().map(|(_, s)| s.len()).sum::<usize>(),
        cases.len()
    );
}
