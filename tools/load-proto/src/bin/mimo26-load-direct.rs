//! CPU check and benchmark of MiMo V2.6 Flash's direct loading path.
//!
//! ```text
//! mimo26-load-direct digest --checkpoint DIR --image text|drafter|encoders
//!                           [--package DIR --package-index] [--ranks N] [--layers 0,1,5]
//! mimo26-load-direct verify --checkpoint DIR --image ... --reference LEDGER [...as digest]
//! mimo26-load-direct bench  --checkpoint DIR --image ... [--package DIR --package-index]
//!                           [--ranks N] [--cold] [--digest]
//! mimo26-load-direct plan   --checkpoint DIR --image ... [--package DIR --package-index] [--ranks N]
//! mimo26-load-direct evict  --checkpoint DIR
//! mimo26-load-direct create --checkpoint DIR --bundle-dir DIR [--topology ID] [--ranks N]
//!                           [--allocation-alignment BYTES]
//! common: [--threads N] [--chunk-mib M] [--direct-reads 0|1] [--read-ahead 0|1]
//!         [--module libbundle_mimo26.so]
//! ```
//!
//! The plans come from the bundle's planner
//! (`model_mimo26_contract::weight_plan`, or the built module's export with
//! `--module`) and run through the host's executor (`runtime_host::direct`),
//! every rank at once, `--threads` workers per rank, into one reused chunk
//! per rank as the pinned upload does. By default the plan sees no package
//! components, so its layout is derived from the checkpoint alone;
//! `--package DIR --package-index` passes that package's bundle-metadata
//! components (its drafter kind, for example) as the host does: those of
//! its topology of `--ranks` ranks. `--layers`
//! plans a text image of only those decoder layers through the contract.
//! `digest` streams every rank and prints the load's ledger entry
//! ([`qwen38_load_proto::ledger`]); `verify --reference` compares the load
//! with a ledger entry (or a `bench --digest` report) of an earlier build,
//! which proves the device bytes unchanged. `create` loads a package
//! directory with a direct load of the checkpoint and runs the
//! bundle's create on the host's fake device with sparse allocations, so the
//! planned images, indexes and identities meet the bundle's own admission
//! and upload without a GPU or reading tensor bytes (stderr carries one
//! `startup weight_upload` line per uploaded image; the fake device does not
//! run MiMo's later device setup, so create itself then stops). Its
//! allocations are only as aligned as CUDA guarantees (256 bytes by
//! default), so an arena alignment above that is refused as a real device
//! may refuse it. Each command prints one JSON document.
use qwen38_load_proto::{
    GIB, MIB, allowed_cpus, cached_bytes, cpu_seconds, evict, ledger, package_components,
    parse_args, peak_rss_bytes,
};
use runtime_bundle_abi::weight_plan::WeightPlanV1;
use runtime_host::direct::{DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1};
use sha2::Digest;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use model_mimo26_contract::weight_plan::{
    DRAFTER_IMAGE_ROLE, ENCODER_IMAGE_ROLE, TEXT_IMAGE_ROLE, index_role,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

/// Every safetensors file of the checkpoint and its immediate subdirectories.
fn safetensors(root: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if !entry.file_name().to_string_lossy().starts_with('.') {
                for inner in std::fs::read_dir(entry.path())? {
                    let inner = inner?;
                    if inner
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".safetensors")
                    {
                        paths.push(inner.path());
                    }
                }
            }
        } else if entry
            .file_name()
            .to_string_lossy()
            .ends_with(".safetensors")
        {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("missing command")?;
    let options = parse_args(args).map_err(|e| e.to_string())?;
    let path = |key: &str| options.get(key).map(PathBuf::from);
    let number = |key: &str, default: usize| -> Result<usize> {
        options.get(key).map_or(Ok(default), |v| Ok(v.parse()?))
    };
    let checkpoint_root = path("checkpoint").ok_or("missing --checkpoint")?;
    if command == "evict" {
        let before = cached_bytes();
        evict(&safetensors(&checkpoint_root)?)?;
        println!(
            "{}",
            serde_json::json!({"command": "evict", "cached_before": before, "cached_after": cached_bytes()})
        );
        return Ok(());
    }
    if command == "create" {
        return create(
            &checkpoint_root,
            &path("bundle-dir").ok_or("create needs --bundle-dir")?,
            options
                .get("topology")
                .map_or("mimo-v26-flash-tp4", String::as_str),
            number("ranks", 4)?,
            number("allocation-alignment", 256)? as u64,
        );
    }
    if options.contains_key("images") {
        return Err(
            "--images is gone: rank images are no longer written; verify compares with a digest ledger (--reference)".into(),
        );
    }
    let role = match options.get("image").map(String::as_str) {
        Some("text") => TEXT_IMAGE_ROLE,
        Some("drafter") => DRAFTER_IMAGE_ROLE,
        Some("encoders") => ENCODER_IMAGE_ROLE,
        _ => return Err("--image must be text, drafter or encoders".into()),
    };
    let ranks = number("ranks", 4)?;
    let chunk = number("chunk-mib", 64)? * MIB;
    let layers: Option<BTreeSet<usize>> = options
        .get("layers")
        .map(|list| list.split(',').map(str::parse).collect())
        .transpose()?;
    let opened = Instant::now();
    let checkpoint = Arc::new(DirectCheckpointV1::open(&checkpoint_root)?);
    let open_seconds = opened.elapsed().as_secs_f64();
    let with_index = options.contains_key("package-index");
    let components = match (path("package"), with_index) {
        (Some(package), true) => package_components(&package, ranks as u32)?,
        (None, false) => Vec::new(),
        (Some(_), false) => {
            return Err("--package passes its components with --package-index".into());
        }
        (None, true) => return Err("--package-index needs --package".into()),
    };
    // The module's digest is in the planned identities, as the host states it.
    let mut bundle_module = [0_u8; 32];
    let library = path("module")
        .map(|module| -> Result<_> {
            let bytes = std::fs::read(&module)?;
            bundle_module = sha2::Sha256::digest(&bytes).into();
            Ok(runtime_bundle_abi::BundleLibraryV1::open_verified_image(
                &bytes,
                &module.display().to_string(),
            )?)
        })
        .transpose()?;
    let implementation = runtime_host::direct::implementation_v1(bundle_module);
    if library.is_some() && layers.is_some() {
        return Err("--layers plans through the contract, not the module".into());
    }
    let planning = Instant::now();
    let plans: Vec<WeightPlanV1> = (0..ranks as u32)
        .map(|rank| -> Result<WeightPlanV1> {
            let request = checkpoint.request(
                role,
                ranks as u32,
                rank,
                implementation,
                components.clone(),
                vec![],
            );
            match &library {
                Some(library) => library.weight_plan(&request)?,
                None => model_mimo26_contract::weight_plan::plan_layers(&request, layers.as_ref())?,
            }
            .ok_or_else(|| format!("the planner declined {role}").into())
        })
        .collect::<Result<_>>()?;
    let plan_seconds = planning.elapsed().as_secs_f64();
    let records: usize = plans.iter().map(|p| p.records.len()).sum();
    let planned_indexes: Vec<serde_json::Value> = plans
        .iter()
        .map(|plan| {
            let component = plan
                .components
                .iter()
                .find(|c| c.name == index_role(role, plan.rank))
                .ok_or("plan has no rank index")?;
            Ok(serde_json::from_slice(&component.bytes)?)
        })
        .collect::<Result<_>>()?;
    // A planned rank has no image: its index names the plan's identity, the
    // one the host serves for it, and no image digest.
    let identities_ok = plans.iter().zip(&planned_indexes).all(|(plan, index)| {
        let identity = format!(
            "{:x}",
            sha2::Sha256::digest(plan.identity_document(&checkpoint.identity(), &implementation))
        );
        index.get("sha256").is_none() && index["direct_identity"] == identity.as_str()
    });
    let mut ledger_ranks: Vec<serde_json::Value> = plans.iter().map(ledger::rank_entry).collect();
    let mut transforms = BTreeMap::<u32, (u64, u64)>::new();
    for record in plans.iter().flat_map(|plan| &plan.records) {
        let entry = transforms.entry(record.transform).or_default();
        entry.0 += 1;
        entry.1 += record.byte_length;
    }
    let transforms: Vec<_> = transforms
        .iter()
        .map(|(id, (count, bytes))| serde_json::json!({"id": id, "records": count, "bytes": bytes}))
        .collect();
    if command == "plan" {
        // A package without rank images asks for the primary image with an
        // empty role: the text image.
        let primary = if role == TEXT_IMAGE_ROLE && layers.is_none() {
            let request = checkpoint.request(
                "",
                ranks as u32,
                0,
                implementation,
                components.clone(),
                vec![],
            );
            let primary = match &library {
                Some(library) => library.weight_plan(&request)?,
                None => model_mimo26_contract::weight_plan::plan_layers(&request, None)?,
            };
            Some(primary.as_ref() == plans.first())
        } else {
            None
        };
        // The host checks every record against the checkpoint headers.
        PlannedImageV1::new(
            checkpoint.clone(),
            plans,
            None,
            implementation,
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                ..DirectLoadOptionsV1::for_ranks(ranks)
            },
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "command": "plan", "image_role": role, "ranks": ranks, "records": records,
                "identities_ok": identities_ok, "primary_is_text": primary,
                "plan_seconds": plan_seconds, "transforms": transforms,
                "checkpoint_identity": checkpoint.identity_hex(),
                "image_bytes": planned_indexes.iter().map(|i| i["image_bytes"].clone()).collect::<Vec<_>>(),
                "views": planned_indexes.iter().map(|i| i["views"].as_array().map_or(0, Vec::len)).collect::<Vec<_>>(),
            }))?
        );
        return Ok(());
    }
    // The host's defaults: direct reads of cold ranges, read-ahead without them.
    let direct_reads = options
        .get("direct-reads")
        .map_or(DirectLoadOptionsV1::for_ranks(ranks).direct_reads, |v| {
            v != "0"
        });
    let read_ahead = options
        .get("read-ahead")
        .map_or(!direct_reads, |v| v != "0");
    let mut threads = number("threads", 0)?;
    if threads == 0 {
        threads = DirectLoadOptionsV1::for_ranks(ranks).threads_per_rank;
    }
    let cold = options.contains_key("cold");
    let reference = path("reference")
        .map(|path| -> Result<serde_json::Value> {
            let document: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
            Ok(document.get("ledger").cloned().unwrap_or(document))
        })
        .transpose()?;
    if reference.is_some() && command != "verify" {
        return Err("--reference is a verify option".into());
    }
    match command.as_str() {
        "verify" if reference.is_none() => {
            return Err("verify needs --reference LEDGER".into());
        }
        "verify" | "bench" | "digest" => {}
        other => return Err(format!("unknown command {other}").into()),
    }
    let digest = options.contains_key("digest") || command == "verify" || command == "digest";
    let cached_before = cached_bytes();
    if cold {
        evict(&safetensors(&checkpoint_root)?)?;
    }
    let cached_start = cached_bytes();
    let preparing = Instant::now();
    let image = PlannedImageV1::new(
        checkpoint.clone(),
        plans,
        None,
        implementation,
        DirectLoadOptionsV1 {
            threads_per_rank: threads,
            read_ahead,
            direct_reads,
            ..DirectLoadOptionsV1::for_ranks(ranks)
        },
    )?;
    let prepare_seconds = preparing.elapsed().as_secs_f64();
    let (user0, system0) = cpu_seconds();
    let started = Instant::now();
    let results: Vec<Result<serde_json::Value>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..ranks)
            .map(|rank| {
                let image = &image;
                scope.spawn(move || -> Result<serde_json::Value> {
                    let arena = image.arena_bytes(rank);
                    let base = image.description().rank_images()[rank].offset;
                    let mut buffer = vec![0_u8; chunk];
                    let mut planned_digest = digest.then(sha2::Sha256::default);
                    let rank_started = Instant::now();
                    let mut offset = 0_u64;
                    while offset < arena {
                        let count = (arena - offset).min(chunk as u64) as usize;
                        image.read(base + offset, &mut buffer[..count])?;
                        if let Some(d) = planned_digest.as_mut() {
                            d.update(&buffer[..count]);
                        }
                        offset += count as u64;
                    }
                    Ok(serde_json::json!({
                        "rank": rank, "bytes": arena,
                        "sha256": planned_digest.map(|d| format!("{:x}", d.finalize())),
                        "seconds": rank_started.elapsed().as_secs_f64(),
                    }))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err("rank thread panicked".into()))
            })
            .collect()
    });
    let wall = started.elapsed().as_secs_f64();
    let (user1, system1) = cpu_seconds();
    let mut per_rank = Vec::new();
    let mut failures = Vec::new();
    for result in results {
        match result {
            Ok(rank) => per_rank.push(rank),
            Err(error) => failures.push(error.to_string()),
        }
    }
    let stats = image.stats();
    let bytes: u64 = (0..ranks).map(|rank| image.arena_bytes(rank)).sum();
    let ledger = digest.then(|| {
        for (entry, rank) in ledger_ranks.iter_mut().zip(&per_rank) {
            entry["bytes_sha256"] = rank["sha256"].clone();
        }
        serde_json::json!({
            "format": ledger::FORMAT, "image_role": role, "ranks": ranks, "layers": layers,
            "checkpoint_identity": checkpoint.identity_hex(),
            "planner": if library.is_some() { "bundle export" } else { "contract" },
            "package_index": with_index,
            "per_rank": ledger_ranks,
        })
    });
    let reference_differences = reference.as_ref().map(|reference| {
        let current = ledger.clone().unwrap_or_default();
        let mut differences = ledger::compare(reference, &current);
        differences.extend(failures.iter().cloned());
        if !identities_ok {
            differences.push("a planned rank index does not name the served identity".into());
        }
        differences
    });
    let equal = reference_differences
        .as_ref()
        .map(|differences| differences.is_empty());
    let report = serde_json::json!({
        "command": command,
        "image_role": role,
        "equal": equal,
        "reference_differences": reference_differences,
        "ledger": ledger,
        "failures": failures,
        "package_index": with_index,
        "identities_ok": identities_ok,
        "layers": layers,
        "checkpoint_identity": checkpoint.identity_hex(),
        "planner": if library.is_some() { "bundle export" } else { "contract" },
        "ranks": ranks, "records": records, "transforms": transforms,
        "threads_per_rank": threads, "read_ahead": read_ahead, "direct_reads": direct_reads,
        "chunk_bytes": chunk, "cold": cold,
        "open_seconds": open_seconds, "plan_seconds": plan_seconds, "prepare_seconds": prepare_seconds,
        "wall_seconds": wall,
        "image_bytes": bytes, "image_gib_per_second": bytes as f64 / GIB / wall,
        "source_bytes": stats.source_bytes, "source_reads": stats.source_reads,
        "advised_bytes": stats.advised_bytes, "direct_bytes": stats.direct_bytes,
        "read_thread_seconds": stats.read_seconds, "produce_thread_seconds": stats.produce_seconds,
        "shared_thread_seconds": stats.shared_seconds, "rank_read_seconds": stats.rank_seconds,
        "cpu_user_seconds": user1 - user0, "cpu_system_seconds": system1 - system0,
        "average_busy_cpus": (user1 - user0 + system1 - system0) / wall,
        "cached_bytes": {"before_evict": cached_before, "at_start": cached_start, "at_end": cached_bytes()},
        "load_average": std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
        "allowed_cpus": allowed_cpus(), "peak_rss_bytes": peak_rss_bytes(),
        "per_rank": per_rank,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if (command == "verify" && equal != Some(true)) || (command == "digest" && !failures.is_empty())
    {
        std::process::exit(2);
    }
    Ok(())
}

/// Runs the bundle's create over a direct load of `checkpoint` on the fake
/// device, as serving would up to the first device work, at 32 sequences or
/// the topology's package maximum if smaller.
fn create(
    checkpoint: &Path,
    bundle: &Path,
    topology: &str,
    ranks: usize,
    allocation_alignment: u64,
) -> Result<()> {
    runtime_host::direct::select_checkpoint_v1(checkpoint)?;
    let manifest = runtime_bundle::BundleManifestV1::from_canonical_json(&std::fs::read(
        bundle.join("manifest.json"),
    )?)?;
    let host = runtime_bundle::BundleHostCompatibilityV1::new(
        "sm_120f",
        runtime_bundle::MODEL_BUNDLE_V1_ABI_VERSION,
    )?
    .with_capabilities(manifest.capabilities_of_every_topology().cloned().chain(
        std::iter::once(runtime_bundle::BundleOpaqueId::new(
            runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
        )?),
    ));
    let started = Instant::now();
    let loaded = runtime_host::load_selected_topology_bundle_v1(
        &runtime_host::BundleSelectionV1::Directory(bundle.to_path_buf()),
        &host,
        topology,
    )?;
    let mut device = runtime_host::fake::FakeHostDeviceV1::default();
    device.use_sparse_probe_allocations(true);
    device.set_allocation_alignment(allocation_alignment);
    // An RTX PRO 6000's 96 GiB per rank.
    device.set_device_memory_bytes(96 << 30);
    let sequences = loaded
        .manifest()
        .batch_limits()
        .map_or(32, |limits| limits.maximum_sequences().min(32));
    // As `serve` creates it: serving KV, DFlash's block speculation (7
    // drafts, 8 verify rows per item) and `sequences` sequences.
    let result = loaded.create_instance_with_serving_options(
        device,
        topology,
        (0..ranks as i32).collect(),
        Some(runtime_host::ServingKvPolicyV1 {
            maximum_pages: Some(4096),
            ..Default::default()
        }),
        runtime_bundle_abi::BundleRawSpeculationV1 {
            maximum_draft_tokens: 7,
            maximum_verify_tokens: sequences * 8,
            ..runtime_bundle_abi::BundleRawSpeculationV1::disabled()
        },
        Some(sequences),
        None,
    );
    let (logs, refused) = match &result {
        Ok((backend, _)) => backend.with_device(|device| {
            (
                device
                    .logs()
                    .iter()
                    .map(|line| String::from_utf8_lossy(line).into_owned())
                    .collect::<Vec<_>>(),
                device.refused_launches().len(),
            )
        }),
        Err(_) => (Vec::new(), 0),
    };
    let report = serde_json::json!({
        "command": "create", "bundle": bundle, "topology": topology,
        "sequences": sequences, "created": result.is_ok(),
        "error": result.as_ref().err().map(ToString::to_string),
        "bundle_logs": logs, "refused_launches": refused,
        "seconds": started.elapsed().as_secs_f64(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if result.is_err() {
        std::process::exit(2);
    }
    Ok(())
}
