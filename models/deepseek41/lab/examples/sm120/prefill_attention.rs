//! Snapshot lab for the served TP4 prefill attention geometry.
use bundle_deepseek41::sm120::latent;
use kernel_build::{BuildOptions, CubinSpec, Toolchain};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernel_harness::Cubins;
use kernels_sm120::latent::{sources, sources_prefill};
use model_deepseek41_contract::geometry::FLASH;
use std::path::Path;

#[path = "prefill_attention/probe.rs"]
mod probe;
#[path = "prefill_attention/timing.rs"]
mod timing;

const REFERENCE: &str = "prefill_attention_reference_sm120";

fn entries() -> Result<Vec<sources::Entry>, String> {
    let entries = latent::entries(&FLASH, 4)?
        .into_iter()
        .filter(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    if entries.len() != 3 {
        return Err("expected window and two mixed prefill entries".into());
    }
    Ok(entries)
}

fn specs() -> Result<Vec<CubinSpec>, String> {
    let entries = entries()?;
    let mut old =
        sources::cubin_spec(latent::profile(&FLASH), &entries).map_err(|e| e.to_string())?;
    old.name = REFERENCE.into();
    Ok(vec![
        old,
        sources_prefill::cubin_spec(latent::profile(&FLASH), &entries)
            .map_err(|e| e.to_string())?,
        timing::spec(),
    ])
}

fn build(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(out)?;
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()?;
    if out.canonicalize()?.starts_with(repository) {
        return Err("snapshot must be outside the repository".into());
    }
    let lock = out.join("checks.lock");
    let toolchain = Toolchain::detect()?;
    if toolchain.skipped {
        return Err("CUDA compiler required".into());
    }
    let built = kernel_build::build(
        &toolchain,
        &specs()?,
        out,
        &lock,
        BuildOptions {
            lock_update: !lock.exists(),
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )?;
    let modules = built
        .iter()
        .map(|b| serde_json::json!({"name":b.name,"sha256":b.sha256}))
        .collect::<Vec<_>>();
    std::fs::write(
        out.join("snapshot.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"modules":modules, "rows":8192, "heads":16, "width":512,
                "layers": &FLASH.shape.compress_ratios[..FLASH.layers() as usize]}),
        )?,
    )?;
    Ok(())
}

fn snapshot(out: &Path) -> Result<Cubins, Box<dyn std::error::Error>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("snapshot.json"))?)?;
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    for spec in specs()? {
        let record = manifest["modules"]
            .as_array()
            .ok_or("missing modules")?
            .iter()
            .find(|r| r["name"] == spec.name)
            .ok_or("missing module")?;
        let code = std::fs::read(out.join(format!("{}.cubin", spec.name)))?;
        if record["sha256"] != kernel_build::digest(&code) {
            return Err("cubin digest differs".into());
        }
        let name: &'static str = Box::leak(spec.name.into_boxed_str());
        let symbols = spec
            .entries
            .into_iter()
            .map(|e| &*Box::leak(e.symbol.into_boxed_str()))
            .collect::<Vec<_>>();
        bytes.push((name, &*Box::leak(code.into_boxed_slice())));
        entries.push((name, &*Box::leak(symbols.into_boxed_slice())));
    }
    Ok(Cubins::new(
        Box::leak(bytes.into_boxed_slice()),
        Box::leak(entries.into_boxed_slice()),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mode = args
        .first()
        .ok_or("usage: prefill_attention build|check|run OUT")?;
    let out = Path::new(args.get(1).ok_or("missing OUT")?);
    if mode == "build" {
        return build(out);
    }
    let cubins = snapshot(out)?;
    if mode == "check" {
        cubins.bytes("probe_sm120")?;
        println!("PASS prefill_attention_snapshot modules=3 entries=13 probe_sm120=true");
        return Ok(());
    }
    if mode != "run" {
        return Err("unknown mode".into());
    }
    if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, probe::run)?.is_none() {
        return Err("GPU test did not execute".into());
    }
    Ok(())
}
