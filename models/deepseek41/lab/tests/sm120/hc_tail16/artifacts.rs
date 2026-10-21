use super::{cases, corpus, device};
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
/// The bundle lock families of every module either arm launches.
const LOCKS: [&str; 4] = [
    include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_expand_mix.lock"),
    include_str!("../../../../bundle/kernels.lock.d/norm_quantize.lock"),
    include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_chain.lock"),
    include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_norm.lock"),
];
/// Cases per kind: one per site of both C8 decode forwards, and two per
/// site of the Verify-32 forwards' first 16 rows.
pub const SITES: usize = 77;
pub const ROUNDS: usize = 7;
pub const WARMUP: usize = 2;
pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_HC_TAIL16_ARTIFACTS")
        .expect("artifact directory required")
        .into()
}
fn exports(locks: &[&str], name: &str, bytes: &[u8]) -> LabResult<Vec<String>> {
    let rows = locks
        .iter()
        .map(|lock| serde_json::from_str::<Vec<Value>>(lock).map_err(|e| e.to_string()))
        .collect::<LabResult<Vec<_>>>()?;
    let row = rows
        .iter()
        .flatten()
        .find(|r| r["name"] == name && r["cubin_sha256"] == kernel_build::digest(bytes))
        .ok_or_else(|| format!("module {name} differs from its lock"))?;
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
/// The bundle modules holding every symbol either arm launches, checked
/// against their locks, and the kernel-lab timing probe.
pub fn images() -> LabResult<Images> {
    let mut result = Images::new();
    for symbol in device::symbols()? {
        let (name, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&symbol)
            .ok_or_else(|| format!("{symbol}: bundle cubin missing"))?;
        let symbols = exports(&LOCKS, name, bytes)?;
        if !symbols.contains(&symbol) {
            return Err(format!("{symbol}: not exported by {name}"));
        }
        result.insert(name.into(), (bytes.to_vec(), symbols));
    }
    let probe = std::fs::read(
        std::env::var_os("CHOPINFER_HC_TAIL16_PROBE").ok_or("kernel-lab probe required")?,
    )
    .map_err(|e| e.to_string())?;
    let symbols = exports(
        &[include_str!(
            "../../../../../../crates/kernel-lab/kernels.lock"
        )],
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
/// The first case of each kind, for memcheck.
pub fn memcheck_cases(composites: Vec<cases::Composite>) -> Vec<cases::Composite> {
    let mut picked = Vec::new();
    for c in composites {
        if !picked.iter().any(|p: &cases::Composite| p.kind == c.kind) {
            picked.push(c);
        }
    }
    picked
}
pub fn coverage(cases: &[corpus::Case], composites: &[cases::Composite], sha: &str) -> Value {
    json!({"cases":composites.iter().map(|c|json!({"id":c.id,"kind":c.kind,"widening":cases::widening(cases,c),
        "rows":c.rows.iter().map(|&(i,r)|json!([cases[i].id,r])).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "corpus_sha256":sha,"configurations":["served","other"],
        "matrix_reports":composites.len()*2,"timing_records":composites.len()*4,"memcheck_reports":2*2,
        "rounds":ROUNDS,"warmup":WARMUP,
        "scope":"Real captured rows of one site per 16-row case: both captured C8 decode forwards, or the first 16 rows of a captured Verify-32 forward. The served configuration uses the site's widening policy with post and comb aliased to their inputs; the other configuration flips widening with separate coefficient buffers."})
}
#[test]
#[ignore = "requires staged CHOPINFER_HC_TAIL16_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn frozen_modules_and_cases_on_cpu() {
    let export = std::env::var("CHOPINFER_HC_TAIL16_EXPORT").as_deref() == Ok("1");
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
    let (corpus, composites, sha) = cases::load().unwrap();
    assert_eq!(
        composites
            .iter()
            .map(|c| c.kind)
            .fold(BTreeMap::new(), |mut m, k| {
                *m.entry(k).or_insert(0) += 1;
                m
            }),
        BTreeMap::from([("decode8x2", SITES), ("verify32", 2 * SITES)])
    );
    let value = coverage(&corpus, &composites, &sha);
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
    assert_eq!(memcheck_cases(composites).len(), 2);
    println!(
        "HC_TAIL16_ARTIFACTS modules={} symbols={} cases={} PASS",
        images.len(),
        images.values().map(|(_, s)| s.len()).sum::<usize>(),
        value["cases"].as_array().unwrap().len()
    );
}
