use super::{corpus, device, registration, twin};
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
/// Executed sites per row count: two captured groups of 77 boundaries.
pub const SITES: usize = 154;
/// Captured row counts with a chain-producer pair: C1 and C8 decode, Verify-6.
const QUALIFIABLE: [usize; 3] = [1, 6, 8];
/// The row counts this run qualifies, `CHOPINFER_HC_COMPLETION_ROWS` as
/// increasing comma-separated values; C1 and C8 when unset.
pub fn rows() -> LabResult<Vec<usize>> {
    let Ok(text) = std::env::var("CHOPINFER_HC_COMPLETION_ROWS") else {
        return Ok(vec![1, 8]);
    };
    let rows = text
        .split(',')
        .map(|v| v.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    if rows.is_empty()
        || rows.windows(2).any(|p| p[0] >= p[1])
        || rows.iter().any(|r| !QUALIFIABLE.contains(r))
    {
        return Err("invalid HC completion row selection".into());
    }
    Ok(rows)
}
pub const ROUNDS: usize = 7;
pub const WARMUP: usize = 2;
pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_HC_COMPLETION_ARTIFACTS")
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
/// A frozen candidate module: its exports must be the registration's, and
/// an embedded copy must be the frozen bytes.
fn candidate(
    variable: &str,
    lock: &str,
    name: &str,
    expected: Vec<String>,
) -> LabResult<(Vec<u8>, Vec<String>)> {
    let bytes = std::fs::read(std::env::var_os(variable).ok_or("candidate path required")?)
        .map_err(|e| e.to_string())?;
    let symbols = exports(lock, name, &bytes)?;
    if symbols != expected {
        return Err("candidate exports differ from the registration".into());
    }
    if let Some((_, embedded)) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&expected[0])
        && embedded != bytes
    {
        return Err("embedded candidate differs from frozen qualification".into());
    }
    Ok((bytes, symbols))
}
/// The served producer and tail modules embedded in the bundle, the frozen
/// candidate modules, and the kernel-lab timing probe.
pub fn images() -> LabResult<Images> {
    use bundle_deepseek41::sm120 as b;
    let g = &model_deepseek41_contract::geometry::FLASH;
    let mut result = Images::new();
    for (symbol, lock) in [
        (
            b::hc_round_chain::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_chain.lock"),
        ),
        (
            b::hc_finish_norm::entry(g).symbols()[1].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_norm.lock"),
        ),
    ] {
        let (name, bytes) = b::fold_cubin_for_symbol(&symbol).ok_or("reference cubin missing")?;
        result.insert(name.into(), (bytes.to_vec(), exports(lock, name, bytes)?));
    }
    result.insert(
        registration::CUBIN.into(),
        candidate(
            "CHOPINFER_HC_COMPLETION_CUBIN",
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_completion.lock"),
            registration::CUBIN,
            registration::entries()
                .iter()
                .map(|(e, w)| e.symbol(*w))
                .collect(),
        )?,
    );
    result.insert(
        twin::CUBIN.into(),
        candidate(
            "CHOPINFER_HC_FINISH_TWIN_CUBIN",
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_twin.lock"),
            twin::CUBIN,
            twin::entries().iter().map(|(e, w)| e.symbol(*w)).collect(),
        )?,
    );
    let probe = std::fs::read(
        std::env::var_os("CHOPINFER_HC_COMPLETION_PROBE").ok_or("kernel-lab probe required")?,
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
/// Every eligible real boundary of the selected row counts in the
/// authenticated corpus.
pub fn cases() -> LabResult<(Vec<corpus::Case>, String)> {
    let (cases, complete, sha) = corpus::load()?;
    if !complete {
        return Err("complete captured corpus required".into());
    }
    let rows = rows()?;
    let selected: Vec<_> = cases
        .into_iter()
        .filter(|c| rows.contains(&c.rows))
        .collect();
    if selected.len() != rows.len() * SITES
        || rows
            .iter()
            .any(|&r| selected.iter().filter(|c| c.rows == r).count() != SITES)
    {
        return Err("selected real coverage differs".into());
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
    }
    Ok((selected, sha))
}
/// The two smallest-index sites of each row count, for memcheck.
pub fn memcheck_cases() -> LabResult<Vec<corpus::Case>> {
    let (cases, _) = cases()?;
    let mut picked = Vec::new();
    for rows in rows()? {
        picked.extend(
            cases
                .iter()
                .filter(|c| c.rows == rows)
                .take(2)
                .map(|c| corpus::Case {
                    id: c.id.clone(),
                    rows: c.rows,
                    root: c.root.clone(),
                    record: c.record.clone(),
                }),
        );
    }
    Ok(picked)
}
pub fn coverage(cases: &[corpus::Case], sha: &str) -> LabResult<Value> {
    let variants = device::VARIANTS.len();
    let rows = rows()?;
    Ok(
        json!({"cases":cases.iter().map(|c|json!({"id":c.id,"rows":c.rows,"widening":c.widening()})).collect::<Vec<_>>(),
        "corpus_sha256":sha,"rows":rows,"variants":device::VARIANTS,"timing_variants":device::TIMING_VARIANTS,
        "controls":device::controls().iter().map(device::Candidate::symbol).collect::<Vec<_>>(),
        "configurations":["served","other"],
        "matrix_reports":cases.len()*variants*2,"timing_records":cases.len()*device::TIMING_VARIANTS.len()*2,
        "memcheck_reports":2*rows.len()*(variants+1),"rounds":ROUNDS,"warmup":WARMUP,
        "scope":"Real captured HC boundaries of the selected row counts (C1 and C8 decode, Verify-6). The served configuration uses the site's widening policy with post and comb aliased to their inputs; the other configuration flips widening with separate coefficient buffers. Timing pairs the served pair with itself (A/A) and with every variant."}),
    )
}
#[test]
#[ignore = "requires staged CHOPINFER_HC_COMPLETION_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn frozen_modules_and_cases_on_cpu() {
    let export = std::env::var("CHOPINFER_HC_COMPLETION_EXPORT").as_deref() == Ok("1");
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
    let value = coverage(&cases, &sha).unwrap();
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
    assert_eq!(memcheck_cases().unwrap().len(), 2 * rows().unwrap().len());
    println!(
        "HC_COMPLETION_ARTIFACTS modules={} symbols={} cases={} PASS",
        images.len(),
        images.values().map(|(_, s)| s.len()).sum::<usize>(),
        cases.len()
    );
}
