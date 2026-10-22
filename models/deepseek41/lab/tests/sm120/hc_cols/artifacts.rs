use super::{cases, cols, corpus, device, twin};
use bundle_deepseek41::sm120 as b;
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
pub const ROUNDS: usize = 7;
pub const WARMUP: usize = 2;
/// Timed sites: every seventh of the 77, at 16 rows (both C8 decode
/// forwards) and 32 rows (the first Verify-32 forward).
pub const TIMING_STRIDE: usize = 7;
pub fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_HC_COLS_ARTIFACTS")
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
    name: &str,
    expected: Vec<String>,
) -> LabResult<(Vec<u8>, Vec<String>)> {
    let bytes = std::fs::read(std::env::var_os(variable).ok_or("candidate path required")?)
        .map_err(|e| e.to_string())?;
    let symbols = exports(
        include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_cols.lock"),
        name,
        &bytes,
    )?;
    if symbols != expected {
        return Err("candidate exports differ from the registration".into());
    }
    if let Some((_, embedded)) = b::fold_cubin_for_symbol(&expected[0])
        && embedded != bytes
    {
        return Err("embedded candidate differs from frozen qualification".into());
    }
    Ok((bytes, symbols))
}
/// The served producer and tail modules and the served two-CTA tail module
/// embedded in the bundle, the frozen column-pair modules, and the
/// kernel-lab timing probe.
pub fn images() -> LabResult<Images> {
    let g = &model_deepseek41_contract::geometry::FLASH;
    let mut result = Images::new();
    for (symbol, lock) in [
        (
            b::hc_round_chain::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_chain.lock"),
        ),
        (
            b::hc_round_rows::entry(g).symbols()[0].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_round_rows.lock"),
        ),
        (
            b::hc_finish_norm::entry(g).symbols()[1].clone(),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_norm.lock"),
        ),
        (
            twin::served().symbol(32),
            include_str!("../../../../bundle/kernels.lock.d/sinkhorn_hc_finish_twin.lock"),
        ),
    ] {
        let (name, bytes) = b::fold_cubin_for_symbol(&symbol).ok_or("served cubin missing")?;
        result.insert(name.into(), (bytes.to_vec(), exports(lock, name, bytes)?));
    }
    if result[twin::CUBIN].1
        != twin::entries()
            .iter()
            .map(|(e, w)| e.symbol(*w))
            .collect::<Vec<_>>()
    {
        return Err("served two-CTA tail exports differ from the registration".into());
    }
    result.insert(
        cols::PRODUCER_CUBIN.into(),
        candidate(
            "CHOPINFER_HC_COLS_PRODUCER_CUBIN",
            cols::PRODUCER_CUBIN,
            cols::producer_entries()
                .iter()
                .map(|(e, tile, tx)| e.symbol(*tile, *tx))
                .collect(),
        )?,
    );
    result.insert(
        cols::FINISH_CUBIN.into(),
        candidate(
            "CHOPINFER_HC_COLS_FINISH_CUBIN",
            cols::FINISH_CUBIN,
            cols::finish_entries()
                .iter()
                .map(|(e, w)| e.symbol(*w))
                .collect(),
        )?,
    );
    let probe = std::fs::read(
        std::env::var_os("CHOPINFER_HC_COLS_PROBE").ok_or("kernel-lab probe required")?,
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
/// Every case: 77 decode8x2 and 154 verify32h at 16 rows, 154 verify32 at
/// 32 rows.
pub fn cases() -> LabResult<(Vec<corpus::Case>, Vec<cases::Composite>, String)> {
    let (corpus, composites, sha) = cases::load()?;
    for (kind, _) in cases::KINDS {
        let n = composites.iter().filter(|c| c.kind == kind).count();
        if n != if kind == "decode8x2" { 1 } else { 2 } * cases::SITES {
            return Err("case coverage differs".into());
        }
    }
    Ok((corpus, composites, sha))
}
/// The first case of each of `kinds`, for memcheck and racecheck.
pub fn sanitizer_cases<'a>(
    all: &'a [cases::Composite],
    kinds: &[String],
) -> Vec<&'a cases::Composite> {
    let mut seen = std::collections::BTreeSet::new();
    all.iter()
        .filter(|c| kinds.iter().any(|k| k == c.kind) && seen.insert(c.kind))
        .collect()
}
/// `CHOPINFER_HC_COLS_SANITIZER_KINDS` as comma-separated case kinds;
/// every kind when unset.
pub fn sanitizer_kinds() -> LabResult<Vec<String>> {
    let Ok(text) = std::env::var("CHOPINFER_HC_COLS_SANITIZER_KINDS") else {
        return Ok(cases::KINDS.iter().map(|(k, _)| (*k).to_owned()).collect());
    };
    let kinds = text.split(',').map(str::to_owned).collect::<Vec<_>>();
    if kinds.is_empty()
        || kinds
            .iter()
            .any(|k| !cases::KINDS.iter().any(|(kind, _)| kind == k))
        || kinds
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != kinds.len()
    {
        return Err("invalid sanitizer case kinds".into());
    }
    Ok(kinds)
}
/// Every seventh site at 16 rows (decode8x2) and 32 rows (the first
/// Verify-32 forward).
pub fn timing_cases(all: &[cases::Composite]) -> Vec<&cases::Composite> {
    all.iter()
        .filter(|c| {
            c.site % TIMING_STRIDE == 0
                && (c.kind == "decode8x2" || c.id.starts_with("verify32:0:"))
        })
        .collect()
}
/// The first case at each row count, for Nsight Compute.
pub fn profile_cases(all: &[cases::Composite]) -> Vec<&cases::Composite> {
    all.iter()
        .filter(|c| c.site == 0 && (c.kind == "decode8x2" || c.id.starts_with("verify32:0:")))
        .collect()
}
pub fn coverage(
    corpus: &[corpus::Case],
    composites: &[cases::Composite],
    sha: &str,
) -> LabResult<Value> {
    let variants = device::variants();
    let ids = |list: Vec<&cases::Composite>| list.iter().map(|c| c.id.clone()).collect::<Vec<_>>();
    let timing = ids(timing_cases(composites));
    let every = cases::KINDS
        .iter()
        .map(|(k, _)| (*k).to_owned())
        .collect::<Vec<_>>();
    let sanitizer = ids(sanitizer_cases(composites, &every));
    let profile = ids(profile_cases(composites));
    Ok(
        json!({"cases":composites.iter().map(|c|json!({"id":c.id,"kind":c.kind,"rows":c.size(),"site":c.site,
            "widening":cases::widening(corpus, c)})).collect::<Vec<_>>(),
        "corpus_sha256":sha,"variants":variants,
        "controls":device::controls().iter().map(|c| c.symbols()).collect::<Vec<_>>(),
        "configurations":["served","other"],
        "timing_cases":timing,"sanitizer_cases":sanitizer,"profile_cases":profile,
        "matrix_reports":composites.len()*variants.len()*2,
        "timing_records":timing.len()*(variants.len()+1)*2,
        "sanitizer_reports":sanitizer.len()*variants.len(),
        "profile_arms":profile.len()*(variants.len()+1),
        "rounds":ROUNDS,"warmup":WARMUP,
        "scope":"Real captured HC boundaries at 16 rows (two C8 decode forwards of a site, or the first 16 rows of a Verify-32 forward) and 32 rows (a Verify-32 forward). The served configuration uses the site's widening policy with post and comb aliased to their inputs; the other configuration flips widening with separate coefficient buffers. Every output role, the chain sums and (for candidates that keep the served producer) the RMS lane words are compared bit for bit with the served boundary; the column pair must leave the lane words untouched. Timing pairs the served boundary with itself (A/A) and with every candidate."}),
    )
}
#[test]
#[ignore = "requires staged CHOPINFER_HC_COLS_ARTIFACTS and CHOPINFER_HC_GRAM_CORPUS"]
fn frozen_modules_and_cases_on_cpu() {
    let export = std::env::var("CHOPINFER_HC_COLS_EXPORT").as_deref() == Ok("1");
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
    let (corpus, composites, sha) = cases().unwrap();
    let value = coverage(&corpus, &composites, &sha).unwrap();
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
    let staged = images
        .values()
        .flat_map(|(_, s)| s.iter().cloned())
        .collect::<std::collections::BTreeSet<_>>();
    for symbol in device::symbols().unwrap() {
        assert!(staged.contains(&symbol), "{symbol} is not staged");
    }
    assert_eq!(value["timing_cases"].as_array().unwrap().len(), 22);
    assert_eq!(value["sanitizer_cases"].as_array().unwrap().len(), 3);
    assert_eq!(value["profile_cases"].as_array().unwrap().len(), 2);
    println!(
        "HC_COLS_ARTIFACTS modules={} symbols={} cases={} PASS",
        images.len(),
        images.values().map(|(_, s)| s.len()).sum::<usize>(),
        composites.len()
    );
}
