//! Qualify shared-head staging separately from its asynchronous loaders.
use super::*;
use sources_prefill::heads;

const TOPOLOGY_REFERENCE: &str = "prefill_head_topology_reference_sm120";
const TOPOLOGY_CACHE: &str = "prefill_head_topology_cache_sm120";
#[path = "heads_run.rs"]
mod execution;
pub use execution::next;
pub use execution::run;

fn topology_entries(tp: u32) -> LabResult<Vec<sources::Entry>> {
    let entries = bundle_deepseek41::sm120::latent::entries(&FLASH, tp)?
        .into_iter()
        .filter(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: kernel_core::contract::AttentionPhase::Prefill,
                    second_fp4_keys: true,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    if entries.len() != 2 {
        return Err("expected two mixed prefill entries per topology".into());
    }
    Ok(entries)
}
fn extra_specs() -> LabResult<Vec<kernel_build::CubinSpec>> {
    let native = [1, 2, 4]
        .into_iter()
        .map(topology_entries)
        .collect::<LabResult<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let profile = bundle_deepseek41::sm120::latent::profile(&FLASH);
    let mut reference = sources::cubin_spec(profile, &native).map_err(|e| e.to_string())?;
    reference.name = TOPOLOGY_REFERENCE.into();
    reference.headers = vec![format!("#include \"{}\"", sources_prefill::HEADER)];
    reference.entries.extend(
        sources_prefill::reuse_cubin_spec(profile, &native, 2)
            .map_err(|e| e.to_string())?
            .entries,
    );
    let mut decoded = cache::cubin_spec(profile, &native).map_err(|e| e.to_string())?;
    decoded.name = TOPOLOGY_CACHE.into();
    let mut all = native.clone();
    all.extend(
        native
            .iter()
            .map(cache::decoded_entry)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    );
    let decoded_entries = all[native.len()..].to_vec();
    Ok(vec![
        reference,
        decoded,
        heads::cubin_spec(profile, &all).map_err(|e| e.to_string())?,
        heads::asynchronous::cubin_spec(profile, &decoded_entries, false)
            .map_err(|e| e.to_string())?,
        heads::asynchronous::cubin_spec(profile, &decoded_entries, true)
            .map_err(|e| e.to_string())?,
    ])
}

pub fn snapshot(out: &Path) -> Result<kernel_harness::Cubins, Box<dyn std::error::Error>> {
    let mut specs = super::specs()?;
    specs.extend(extra_specs()?);
    super::snapshot_with(out, specs)
}

pub fn build(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    std::fs::create_dir_all(out)?;
    if out.canonicalize()?.starts_with(repo) {
        return Err("head snapshot must be outside repository".into());
    }
    let toolchain = kernel_build::Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("snapshot.json"))?)?;
    for spec in extra_specs()? {
        let lock = out.join(format!("{}.lock", spec.name));
        let built = kernel_build::build(
            &toolchain,
            &[spec],
            out,
            &lock,
            kernel_build::BuildOptions {
                lock_update: !lock.exists(),
                emit_cargo_directives: false,
                profile_line_info: false,
            },
        )?;
        let modules = manifest["modules"]
            .as_array_mut()
            .ok_or("missing frozen module manifest")?;
        for b in built {
            modules.retain(|m| m["name"] != b.name);
            modules.push(serde_json::json!({"name":b.name,"sha256":b.sha256}));
        }
        std::fs::write(
            out.join("snapshot.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_heads_keep_every_topology_partial_and_merge_abi() {
        for tp in [1, 2, 4] {
            for entry in topology_entries(tp).unwrap() {
                for e in [entry.clone(), cache::decoded_entry(&entry).unwrap()] {
                    let profile = bundle_deepseek41::sm120::latent::profile(&FLASH);
                    let spec = heads::cubin_spec(profile, std::slice::from_ref(&e)).unwrap();
                    assert_eq!(spec.entries[0].params, e.params().unwrap());
                    for pipeline in [false, true] {
                        let spec = heads::asynchronous::cubin_spec(
                            profile,
                            std::slice::from_ref(&e),
                            pipeline,
                        );
                        if e.name.ends_with("_decoded") {
                            assert_eq!(spec.unwrap().entries[0].params, e.params().unwrap());
                        } else {
                            assert!(spec.is_err());
                        }
                    }
                    for rows in [1, 63, 128, 4096, 8192] {
                        let plan = heads::plan(&e, rows).unwrap();
                        assert_eq!(plan.grid, [rows, 4 / tp, 1]);
                        assert_eq!(plan.block, [256, 1, 1]);
                        assert_eq!(plan.dynamic_shared_bytes, 83200);
                        assert_eq!(plan.workspace_bytes, e.workspace_bytes(rows));
                    }
                }
            }
        }
    }
}
