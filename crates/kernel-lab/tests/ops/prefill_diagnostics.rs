//! Opt-in prefill snapshots and schedule accounting. No kernel changes.

use super::*;
use sha2::{Digest, Sha256};
use std::io::Write;

const CANDIDATE_NAMES: [&str; 2] = [
    "chop_moe_mimo_gateup_prefill_mxfp4_a16_t8192",
    "chop_moe_qwen38_gateup_prefill_fp8block_a16_t8192",
];

fn candidate_specs() -> Vec<kernel_build::CubinSpec> {
    moe::cubin_specs(kernel_lab::catalogs(), None)
        .into_iter()
        .filter(|spec| CANDIDATE_NAMES.contains(&spec.name.as_str()))
        .collect()
}

/// Compile only the two candidate gate/up entries against a scratch CUDA tree.
/// This CPU check writes its audit lock and identities outside the repository.
#[test]
fn candidate_cubins_build() {
    let Some(cuda) = std::env::var_os("CHOPINFER_MOE_PREFILL_CANDIDATE_CUDA") else {
        kernel_lab::skip("candidate_cubins_build: set CHOPINFER_MOE_PREFILL_CANDIDATE_CUDA and CHOPINFER_MOE_PREFILL_CANDIDATE_OUT").unwrap();
        return;
    };
    let out = PathBuf::from(
        std::env::var_os("CHOPINFER_MOE_PREFILL_CANDIDATE_OUT")
            .expect("candidate output directory"),
    );
    let specs: Vec<_> = candidate_specs()
        .into_iter()
        .map(|mut spec| {
            spec.sources_root = PathBuf::from(&cuda);
            spec
        })
        .collect();
    assert_eq!(specs.len(), 2);
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA compiler");
    assert!(!toolchain.skipped, "candidate audit requires CUDA 13.2");
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out,
        &out.join("kernels.lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit candidate gate/up cubins");
    assert_eq!(built.len(), 2);
    for cubin in built {
        println!(
            "candidate {} {} {}",
            cubin.name,
            cubin.sha256,
            cubin.path.display()
        );
    }
}

/// A prefill experiment must preserve every other MoE cubin byte for byte.
#[test]
fn candidate_unrelated_cubins_stay_identical() {
    let Some(cuda) = std::env::var_os("CHOPINFER_MOE_PREFILL_CANDIDATE_CUDA") else {
        kernel_lab::skip("candidate_unrelated_cubins_stay_identical: set the candidate CUDA source and output directories").unwrap();
        return;
    };
    let out = PathBuf::from(
        std::env::var_os("CHOPINFER_MOE_PREFILL_CANDIDATE_OUT")
            .expect("candidate output directory"),
    )
    .join("unchanged");
    let specs: Vec<_> = moe::cubin_specs(kernel_lab::catalogs(), None)
        .into_iter()
        .filter(|spec| !CANDIDATE_NAMES.contains(&spec.name.as_str()))
        .map(|mut spec| {
            spec.sources_root = PathBuf::from(&cuda);
            spec
        })
        .collect();
    assert!(!specs.is_empty());
    let locked: Vec<Value> = serde_json::from_str(include_str!("../../kernels.lock"))
        .expect("current production lab lock");
    let toolchain = kernel_build::Toolchain::detect().expect("CUDA compiler");
    assert!(!toolchain.skipped, "candidate audit requires CUDA 13.2");
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out,
        &out.join("kernels.lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("compile and audit unrelated MoE cubins");
    assert_eq!(built.len(), specs.len());
    let changed: Vec<_> = built
        .iter()
        .filter(|cubin| {
            let baseline = locked
                .iter()
                .find(|entry| entry["name"] == cubin.name)
                .expect("unrelated cubin missing from baseline lock");
            baseline["cubin_sha256"] != cubin.sha256
        })
        .map(|cubin| &cubin.name)
        .collect();
    assert!(changed.is_empty(), "unrelated cubins changed: {changed:?}");
    println!("{} unrelated MoE cubins stayed byte-identical", built.len());
}

/// Opt-in, audited gate/up candidates. All other entries retain their cubins.
/// Record the exact loaded bytes and source identity separately from timings.
pub(super) fn candidate_cubin(session: &Session, name: &str) -> LabResult<Option<Module>> {
    let Some(out) = std::env::var_os("CHOPINFER_MOE_PREFILL_CANDIDATE_OUT") else {
        return Ok(None);
    };
    if !CANDIDATE_NAMES.contains(&name) {
        return Ok(None);
    }
    let out = PathBuf::from(out);
    let path = out.join(format!("{name}.cubin"));
    let identity: Value =
        serde_json::from_slice(&fs::read(out.join(format!("{name}.identity.json"))).map_err(err)?)
            .map_err(err)?;
    let bytes = fs::read(&path).map_err(err)?;
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    if identity["name"] != name
        || identity["cubin_sha256"] != sha256
        || identity["line_info"] != false
    {
        return Err("candidate cubin differs from its uninstrumented audit identity".into());
    }
    let spec = candidate_specs()
        .into_iter()
        .find(|spec| spec.name == name)
        .ok_or("candidate spec missing")?;
    let toolchain = kernel_build::Toolchain::detect().map_err(err)?;
    kernel_build::audit_cubin(&toolchain, &spec, &path).map_err(err)?;
    let mut receipt = BTreeMap::new();
    receipt.insert("candidate_cubin".into(), json!(name));
    receipt.insert("cubin_sha256".into(), json!(sha256));
    receipt.insert(
        "source_closure_sha256".into(),
        identity["source_closure_sha256"].clone(),
    );
    receipt.insert("path".into(), json!(path));
    session.receipt("moe_prefill_candidate", receipt)?;
    session.context.load_module(&bytes).map(Some).map_err(err)
}

pub(super) use kernel_lab::time_after as context_time;

/// Save all live output elements once, or compare against a saved baseline.
/// The caller includes the model, route fixture and token count in `name`.
pub(super) fn snapshot(name: &str, outputs: &[Vec<u16>]) -> LabResult<()> {
    let Some(root) = std::env::var_os("CHOPINFER_MOE_PREFILL_BASELINE") else {
        return Ok(());
    };
    let root = PathBuf::from(root);
    let mode = std::env::var("CHOPINFER_MOE_PREFILL_BASELINE_MODE").map_err(err)?;
    if !matches!(mode.as_str(), "record" | "compare") {
        return Err("baseline mode must be record or compare".into());
    }
    for (stage, output) in outputs.iter().enumerate() {
        let path = root.join(format!("{name}-{stage}.bf16"));
        let bytes: Vec<_> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
        if mode == "record" {
            fs::create_dir_all(&root).map_err(err)?;
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(err)?
                .write_all(&bytes)
                .map_err(err)?;
        } else {
            let expected = fs::read(&path).map_err(err)?;
            if expected != bytes {
                return Err(format!(
                    "prefill output differs bitwise: {}",
                    path.display()
                ));
            }
        }
    }
    eprintln!("prefill baseline {mode}: {name}, {} stages", outputs.len());
    Ok(())
}

/// Mirror the persistent full-first, partial-group-sorted snake schedule.
/// Group counts are a work proxy, not measured cycles or a latency prediction.
pub(super) fn schedule(counts: &[usize], feature_tiles: usize) -> Value {
    schedule_grid(counts, feature_tiles, 188)
}

pub(super) fn schedule_grid(counts: &[usize], feature_tiles: usize, ctas: usize) -> Value {
    assert!(ctas > 0);
    let mut groups = Vec::new();
    let mut partial = Vec::new();
    for (expert, &rows) in counts.iter().enumerate() {
        groups.extend(std::iter::repeat_n(8usize, rows / 128 * feature_tiles));
        if rows % 128 != 0 {
            partial.push(((rows % 128).div_ceil(16), expert));
        }
    }
    partial.sort_by_key(|&(groups, expert)| (8 - groups, expert));
    for (groups_in_tile, _) in partial {
        groups.extend(std::iter::repeat_n(groups_in_tile, feature_tiles));
    }
    let mut cta_groups = vec![0usize; ctas];
    let mut cta_items = vec![0usize; ctas];
    for (work, &groups) in groups.iter().enumerate() {
        let round = work / ctas;
        let cta = if round % 2 == 0 {
            work % ctas
        } else {
            ctas - 1 - work % ctas
        };
        cta_groups[cta] += groups;
        cta_items[cta] += 1;
    }
    json!({"ctas": ctas, "items": groups.len(), "rounds": groups.len().div_ceil(ctas),
        "last_round_items": if groups.is_empty() { 0 } else { (groups.len() - 1) % ctas + 1 },
        "row_groups_per_cta": cta_groups,
        "items_per_cta": cta_items,
        "useful_rows": counts.iter().sum::<usize>() * feature_tiles,
        "mma_rows": groups.iter().sum::<usize>() * 16})
}

/// Optional token-major F32 logits. A replay directory has t1024.f32 and
/// t2048.f32 plus a provenance manifest maintained by its producer.
pub(super) fn logits(tokens: usize, experts: usize) -> LabResult<Option<Vec<f32>>> {
    let Some(root) = std::env::var_os("CHOPINFER_MOE_PREFILL_LOGITS") else {
        return Ok(None);
    };
    let bytes = fs::read(PathBuf::from(root).join(format!("t{tokens}.f32"))).map_err(err)?;
    if bytes.len() != tokens * experts * 4 {
        return Err("prefill replay logits have the wrong shape".into());
    }
    let values: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("prefill replay logits must be finite".into());
    }
    Ok(Some(values))
}
