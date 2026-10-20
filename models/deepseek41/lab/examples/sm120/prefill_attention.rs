//! Snapshot lab for the served TP4 prefill attention geometry.
use bundle_deepseek41::sm120::latent;
use kernel_build::{BuildOptions, CubinSpec, Toolchain};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernel_harness::Cubins;
use kernels_sm120::latent::{sources, sources_prefill};
use model_deepseek41_contract::geometry::FLASH;
use std::path::Path;

#[cfg(test)]
#[path = "prefill_attention/head_sharing_design.rs"]
mod head_sharing_design;
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
        sources_prefill::profile_cubin_spec(latent::profile(&FLASH), &entries)
            .map_err(|e| e.to_string())?,
        sources_prefill::reuse_cubin_spec(latent::profile(&FLASH), &entries, 1)
            .map_err(|e| e.to_string())?,
        sources_prefill::reuse_cubin_spec(latent::profile(&FLASH), &entries, 2)
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
        .ok_or("usage: prefill_attention build|check|run|run-reuse|decision-check|decision-time|ablation|traffic OUT [VARIANT]")?;
    let out = Path::new(args.get(1).ok_or("missing OUT")?);
    if mode == "heads-next-source" {
        use sources_prefill::heads::next::{Mode, source};
        for (mode, file) in [
            (Mode::Profile, "sources_prefill_heads_profile.cuh"),
            (Mode::Finish, "sources_prefill_heads_finish.cuh"),
        ] {
            std::fs::write(out.join(file), source(mode).map_err(|e| e.to_string())?)?;
        }
        std::fs::write(
            out.join("sources_prefill_pair_clock.cuh"),
            probe::cache::heads::next::pair_source().map_err(|e| e.to_string())?,
        )?;
        return Ok(());
    }
    if mode == "build" {
        return build(out);
    }
    if mode == "cache-build" {
        return probe::cache::build(out);
    }
    if mode == "heads-next-build" {
        return probe::cache::heads::next::build(out);
    }
    if mode == "heads-next-snapshot" {
        probe::cache::heads::next::snapshot(out)?;
        println!("PASS head_finish_snapshot hashes=true probe_sm120=true");
        return Ok(());
    }
    if ["heads-next-check", "heads-next-time", "heads-next-profile"].contains(&mode.as_str()) {
        use probe::cache::heads::next::{Phase, run, snapshot};
        if args.len() != 4 {
            return Err("usage: heads-next-mode OUT LISTS OFFSET".into());
        }
        let phase = match mode.as_str() {
            "heads-next-check" => Phase::Check,
            "heads-next-time" => Phase::Time,
            _ => Phase::Profile,
        };
        let cubins = snapshot(out)?;
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            run(
                sessions,
                Path::new(&args[2]),
                args[3]
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?,
                phase,
            )
        })?
        .is_none()
        {
            return Err("epilogue GPU phase did not execute".into());
        }
        return Ok(());
    }
    if mode == "heads-build" {
        return probe::cache::heads::build(out);
    }
    if mode == "heads-snapshot" {
        probe::cache::heads::snapshot(out)?;
        println!("PASS head_staging_snapshot hashes=true probe_sm120=true");
        return Ok(());
    }
    if ["heads-check", "heads-time"].contains(&mode.as_str()) {
        if args.len() != 4 {
            return Err("usage: heads-mode OUT LISTS OFFSET".into());
        }
        let cubins = probe::cache::heads::snapshot(out)?;
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            probe::cache::heads::run(
                sessions,
                Path::new(&args[2]),
                args[3]
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?,
                mode == "heads-time",
            )
        })?
        .is_none()
        {
            return Err("head staging GPU phase did not execute".into());
        }
        return Ok(());
    }
    if mode == "stages-build" {
        return probe::cache::stages::build(out);
    }
    if mode == "stages-source" {
        print!("{}", sources_prefill::profiling::source()?);
        return Ok(());
    }
    if mode == "stages-snapshot" {
        probe::cache::stages::snapshot(out)?;
        println!("PASS paired_stages_snapshot hashes=true probe_sm120=true");
        return Ok(());
    }
    if ["stages-check", "stages-time"].contains(&mode.as_str()) {
        if args.len() != 4 {
            return Err("usage: stages-mode OUT LISTS OFFSET".into());
        }
        let cubins = probe::cache::stages::snapshot(out)?;
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            probe::cache::stages::run(
                sessions,
                Path::new(&args[2]),
                args[3]
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?,
                mode == "stages-time",
            )
        })?
        .is_none()
        {
            return Err("paired profiling GPU phase did not execute".into());
        }
        return Ok(());
    }
    if mode == "cache-check" {
        probe::cache::snapshot(out)?;
        println!("PASS prefill_cache_snapshot hashes=true");
        return Ok(());
    }
    if ["split-table", "split-inputs", "split-check", "split-time"].contains(&mode.as_str()) {
        if args.len() != 4 {
            return Err("usage: split-mode OUT LISTS OFFSET".into());
        }
        let lists = Path::new(&args[2]);
        let offset = args[3].parse()?;
        if mode == "split-table" {
            println!(
                "{}",
                serde_json::to_string_pretty(&probe::cache::split::table(offset)?)?
            );
            return Ok(());
        }
        if mode == "split-inputs" {
            return probe::cache::split::check_inputs(lists, offset).map_err(Into::into);
        }
        let cubins = probe::cache::snapshot(out)?;
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            probe::cache::split::run(sessions, lists, offset, mode == "split-time")
        })?
        .is_none()
        {
            return Err("split GPU phase did not execute".into());
        }
        return Ok(());
    }
    if ["retained-inputs", "retained-check", "retained-time"].contains(&mode.as_str()) {
        if args.len() != 4 {
            return Err("usage: retained-mode OUT LISTS OFFSET".into());
        }
        let lists = Path::new(&args[2]);
        let offset = args[3].parse()?;
        if mode == "retained-inputs" {
            return probe::retained::check_inputs(lists, offset).map_err(Into::into);
        }
        let cubins = snapshot(out)?;
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            probe::retained::run(sessions, lists, offset, mode == "retained-time")
        })?
        .is_none()
        {
            return Err("retained GPU phase did not execute".into());
        }
        return Ok(());
    }
    if ["cache-sample-check", "cache-list-check"].contains(&mode.as_str()) {
        if args.len() != 5 {
            return Err("usage: cache-list-check OUT LISTS VARIANT OFFSET".into());
        }
        probe::cache::check_lists(
            Path::new(&args[2]),
            &args[3],
            args[4].parse()?,
            mode == "cache-sample-check",
        )?;
        return Ok(());
    }
    if ["cache-sample", "cache-bitwise", "cache-time"].contains(&mode.as_str()) {
        if args.len() != 5 {
            return Err("usage: cache-mode OUT LISTS VARIANT OFFSET".into());
        }
        let lists = Path::new(&args[2]);
        let variant = &args[3];
        let offset = args[4].parse::<u32>()?;
        let cubins = probe::cache::snapshot(out)?;
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            probe::cache::run(
                sessions,
                lists,
                variant,
                offset,
                mode == "cache-sample",
                mode == "cache-time",
            )
        })?
        .is_none()
        {
            return Err("cache GPU phase did not execute".into());
        }
        return Ok(());
    }
    let cubins = snapshot(out)?;
    if mode == "check" {
        cubins.bytes("probe_sm120")?;
        let specs = specs()?;
        println!(
            "PASS prefill_attention_snapshot modules={} entries={} probe_sm120=true",
            specs.len(),
            specs.iter().map(|s| s.entries.len()).sum::<usize>()
        );
        return Ok(());
    }
    if mode == "traffic" {
        println!("{}", serde_json::to_string_pretty(&probe::traffic()?)?);
        return Ok(());
    }
    if mode == "ablation" {
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, probe::ablation)?.is_none() {
            return Err("GPU ablation did not execute".into());
        }
        return Ok(());
    }
    if mode == "decision-check" || mode == "decision-time" {
        let selected = args.get(2).ok_or("missing decision variant")?;
        if args.len() != 3 || !["window", "mixed64", "mixed32"].contains(&selected.as_str()) {
            return Err("invalid decision arguments".into());
        }
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, |sessions| {
            probe::decision(sessions, selected, mode == "decision-check")
        })?
        .is_none()
        {
            return Err("GPU decision phase did not execute".into());
        }
        return Ok(());
    }
    if mode == "run-reuse" {
        if kernel_harness::gpu_test_multi_with(cubins, 4 << 30, probe::run_reuse)?.is_none() {
            return Err("GPU reuse test did not execute".into());
        }
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
