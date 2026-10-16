//! CPU check and benchmark of the direct loading path (step A2).
//!
//! ```text
//! qwen38-load-direct bench  --checkpoint DIR [--image ...] [--package DIR] [--ranks N] [--cold]
//! qwen38-load-direct plan   --checkpoint DIR [--image ...] [--package DIR] [--ranks N]
//! qwen38-load-direct digest --checkpoint DIR [--image ...] [--package DIR] [--ranks N]
//!                           [--view-digests FILE [--view-rows PATTERN[,PATTERN...]]]
//! qwen38-load-direct verify --checkpoint DIR [--image ...] [--package DIR] [--ranks N]
//!                           --reference LEDGER
//! qwen38-load-direct evict  --checkpoint DIR
//! qwen38-load-direct create --checkpoint DIR --bundle-dir DIR [--topology ID] [--ranks N]
//!                           [--allocation-alignment BYTES]
//! qwen38-load-direct precision --checkpoint DIR --prepare-module FILE --output FILE
//!                              [--activation-policy auto|checkpoint|bf16]
//! qwen38-load-direct packbench [--gib N]
//! bench [--digest] also prints each rank's SHA-256, the host-side digest a
//! device arena digest is compared with. `digest` streams every rank and
//! prints the load's ledger entry ([`qwen38_load_proto::ledger`]): per rank
//! the arena length, plan and component digests and the bytes' SHA-256;
//! `digest --view-digests FILE` also writes each rank's per-view digests
//! (with each expert's and, for views named by `--view-rows`, each row's),
//! so views can be compared across rank counts.
//! `verify --reference` compares the load with a ledger entry (or a `bench
//! --digest` report) of an earlier build, which proves the bytes unchanged.
//! common: [--threads N] [--chunk-mib M] [--direct-reads 0|1] [--read-ahead 0|1]
//!         [--module libbundle_qwen38.so] [--layers 0,3,47] (bench, plan, digest and verify)
//! ```
//!
//! The plans come from the bundle's own planner
//! (`model_qwen38_contract::weight_plan::plan`) and run through the host's
//! executor (`runtime_host::direct`), all ranks at once, `--threads` workers
//! per rank, into one reused chunk per rank as the pinned upload does.
//! `--package` passes the package's bundle metadata (its precision
//! component) as the host does; `--prepare-module FILE` instead resolves the
//! checkpoint's precision against a built bundle module, as the package
//! emitter does. `plan` checks that each rank's records tile every view of
//! its rank index and produces nothing. `--image encoders` plans the vision
//! encoder image instead of the text rank images (default `text`).
//! `--layers` plans only the selected decoder layers (and every tensor
//! outside the layers but the MTP drafter's) through the contract's
//! `rank_plan`, so a subset of a load can be digested or compared with a
//! ledger without the whole of it. `create` loads a package directory
//! with a direct load of the checkpoint and runs the bundle's
//! create on the host's fake device with sparse allocations only as aligned
//! as CUDA guarantees (256 bytes by default), so every planned image, index
//! and identity meets the bundle's admission and upload without a GPU or
//! reading tensor bytes. Each command prints one JSON document on stdout.
use model_qwen38_contract::{contract, views::ngram_view_for};
use qwen38_load_proto::{
    GIB, MIB, allowed_cpus, cached_bytes, cpu_seconds, ledger, parse_args, peak_rss_bytes,
};
use runtime_bundle_abi::weight_plan::NamedBytesV1;
use runtime_host::direct::{DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1};
use runtime_host::page_cache::{CacheAdviceV1, advise_v1};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

/// The package's bundle-metadata components that serve `topology`, by
/// role, sorted by role.
fn package_components(root: &Path, topology: &str) -> Result<Vec<NamedBytesV1>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    let mut components = Vec::new();
    for component in manifest["components"]
        .as_array()
        .ok_or("manifest components")?
    {
        let component_role = component["role"].as_str().ok_or("component role")?;
        let serves = component["topologies"]
            .as_array()
            .is_none_or(|topologies| topologies.iter().any(|t| t == topology));
        if serves && component_role.starts_with("chop.bundle-metadata.") {
            let path = component["path"].as_str().ok_or("component path")?;
            components.push(NamedBytesV1 {
                name: component_role.into(),
                bytes: std::fs::read(root.join(path))?,
            });
        }
    }
    components.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(components)
}

/// Views of a rank index as `(name, offset, bytes, dtype, shape, layout)`.
fn views(index: &serde_json::Value) -> Result<Vec<serde_json::Value>> {
    let mut views: Vec<serde_json::Value> = index["views"]
        .as_array()
        .ok_or("index views")?
        .iter()
        .map(|view| {
            serde_json::json!([
                view["name"],
                view["offset"],
                view["bytes"],
                view["dtype"],
                view["shape"],
                view["layout_id"]
            ])
        })
        .collect();
    views.sort_by_key(|view| view[1].as_u64());
    Ok(views)
}

/// Digests of one rank's views, for comparing views across rank counts:
/// each view's SHA-256; each leading-index slice's (each expert's) for a
/// view of three or more axes; each n-gram head's rows' for the rank's
/// n-gram table views; and each row's first eight SHA-256 bytes for a
/// matrix view whose name contains one of the `--view-rows` patterns.
struct ViewDigests {
    views: Vec<ViewDigest>,
    next: usize,
}

struct ViewDigest {
    view: serde_json::Value,
    offset: u64,
    bytes: u64,
    /// Each unit's bytes, in order, and whether a unit's digest is
    /// truncated to eight bytes; `None` digests the view whole only.
    units: Option<(Vec<u64>, bool)>,
    whole: sha2::Sha256,
    current: sha2::Sha256,
    filled: u64,
    digests: Vec<String>,
}

/// The rank's n-gram table views: their name stem, and the rows of each
/// head the rank owns (the last rank's last head with the table padding).
struct NgramHeads<'a> {
    stem: &'a str,
    rows: &'a [u64],
}

impl ViewDigests {
    fn new(
        index: &serde_json::Value,
        rows: &[String],
        heads: Option<NgramHeads<'_>>,
    ) -> Result<Self> {
        let views = views(index)?
            .into_iter()
            .map(|view| -> Result<ViewDigest> {
                let name = view[0].as_str().ok_or("view name")?;
                let bytes = view[2].as_u64().ok_or("view bytes")?;
                let shape: Vec<u64> = serde_json::from_value(view[4].clone())?;
                let leading = shape.first().copied().unwrap_or(0);
                let row = (leading > 0 && bytes % leading == 0).then(|| bytes / leading);
                // An n-gram table view (codes or flat scales) holds whole rows
                // of the rank's heads in order.
                let table = heads.as_ref().and_then(|heads| {
                    let rows: u64 = heads.rows.iter().sum();
                    (name.starts_with(&format!("{}.", heads.stem))
                        && rows > 0
                        && leading >= rows
                        && bytes % rows == 0)
                        .then(|| heads.rows.iter().map(|r| r * (bytes / rows)).collect())
                });
                let units = match (row, table) {
                    (_, Some(table)) => Some((table, false)),
                    (Some(row), None) if shape.len() >= 3 => {
                        Some((vec![row; leading as usize], false))
                    }
                    (Some(row), None)
                        if shape.len() == 2 && rows.iter().any(|r| name.contains(r.as_str())) =>
                    {
                        Some((vec![row; leading as usize], true))
                    }
                    _ => None,
                };
                Ok(ViewDigest {
                    offset: view[1].as_u64().ok_or("view offset")?,
                    bytes,
                    view,
                    units,
                    whole: sha2::Sha256::default(),
                    current: sha2::Sha256::default(),
                    filled: 0,
                    digests: Vec::new(),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { views, next: 0 })
    }

    /// Feeds `data`, the image bytes from arena offset `at`, in order.
    fn update(&mut self, mut at: u64, mut data: &[u8]) {
        use sha2::Digest;
        while !data.is_empty() {
            while self
                .views
                .get(self.next)
                .is_some_and(|v| v.offset + v.bytes <= at)
            {
                self.next += 1;
            }
            let Some(view) = self.views.get_mut(self.next) else {
                return;
            };
            if at < view.offset {
                let skip = ((view.offset - at) as usize).min(data.len());
                at += skip as u64;
                data = &data[skip..];
                continue;
            }
            let count = ((view.offset + view.bytes - at) as usize).min(data.len());
            let (mut head, tail) = data.split_at(count);
            view.whole.update(head);
            if let Some((units, truncated)) = &view.units {
                while !head.is_empty() {
                    let unit = units[view.digests.len()];
                    let take = ((unit - view.filled) as usize).min(head.len());
                    view.current.update(&head[..take]);
                    view.filled += take as u64;
                    head = &head[take..];
                    if view.filled == unit {
                        let digest = format!("{:x}", std::mem::take(&mut view.current).finalize());
                        view.digests.push(if *truncated {
                            digest[..16].to_owned()
                        } else {
                            digest
                        });
                        view.filled = 0;
                    }
                }
            }
            at += count as u64;
            data = tail;
        }
    }

    fn finish(self) -> serde_json::Value {
        use sha2::Digest;
        self.views
            .into_iter()
            .map(|v| {
                serde_json::json!({
                    "name": v.view[0], "offset": v.offset, "bytes": v.bytes, "dtype": v.view[3],
                    "shape": v.view[4], "layout_id": v.view[5],
                    "sha256": format!("{:x}", v.whole.finalize()),
                    "units": v.units.map(|_| v.digests),
                })
            })
            .collect()
    }
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("missing command")?;
    let options = parse_args(args).map_err(|e| e.to_string())?;
    if command == "packbench" {
        let gib: f64 = options.get("gib").map_or(Ok(2.0), |v| v.parse())?;
        return packbench(gib);
    }
    let path = |key: &str| options.get(key).map(PathBuf::from);
    let number = |key: &str, default: usize| -> Result<usize> {
        options.get(key).map_or(Ok(default), |v| Ok(v.parse()?))
    };
    let checkpoint_root = path("checkpoint").ok_or("missing --checkpoint")?;
    let ranks = number("ranks", 4)?;
    let chunk = number("chunk-mib", 64)? * MIB;
    let opened = Instant::now();
    let checkpoint = Arc::new(DirectCheckpointV1::open(&checkpoint_root)?);
    let open_seconds = opened.elapsed().as_secs_f64();
    if command == "evict" {
        let before = cached_bytes();
        evict(&checkpoint_root)?;
        println!(
            "{}",
            serde_json::json!({"command": "evict", "cached_before": before, "cached_after": cached_bytes()})
        );
        return Ok(());
    }
    let layers: Option<BTreeSet<usize>> = options
        .get("layers")
        .map(|list| list.split(',').map(str::parse).collect())
        .transpose()?;
    if layers.is_some() && matches!(command.as_str(), "create" | "precision") {
        return Err(format!(
            "{command} loads whole images; --layers applies to verify, bench and plan"
        )
        .into());
    }
    if command == "create" {
        let shape = model_qwen38_contract::placement::bundle_shape_flash(ranks)
            .map_err(|e| e.to_string())?;
        return create(
            &checkpoint_root,
            &path("bundle-dir").ok_or("create needs --bundle-dir")?,
            options
                .get("topology")
                .map_or(shape.topology().id, String::as_str),
            ranks,
            CreateOptions {
                allocation_alignment: number("allocation-alignment", 256)? as u64,
                device_memory_bytes: number("device-memory-bytes", 96 << 30)? as u64,
                speculative_tokens: number("speculative-tokens", 0)? as u32,
                verify_tokens: number("verify-tokens", 0)? as u32,
                maximum_sequences: number("max-sequences", 32)? as u32,
                kv_fraction_millionths: options
                    .get("kv-fraction-millionths")
                    .map(|v| v.parse())
                    .transpose()?,
            },
        );
    }
    // The text rank images (`rank{r}`) or the vision encoder's
    // (`encoder-rank{r}`), and each rank's index role.
    let encoders = match options.get("image").map_or("text", String::as_str) {
        "text" => false,
        "encoders" => true,
        other => return Err(format!("--image must be text or encoders, not {other}").into()),
    };
    let role = std::str::from_utf8(if encoders {
        model_qwen38_contract::encoder::views::IMAGE_ROLE
    } else {
        contract::RANK_IMAGE_ROLE
    })?;
    let index_role = |rank: u32| {
        if encoders {
            model_qwen38_contract::encoder::views::index_role(rank)
        } else {
            model_qwen38_contract::contract::index_role(rank)
        }
    };
    let shape =
        model_qwen38_contract::placement::bundle_shape_flash(ranks).map_err(|e| e.to_string())?;
    let topology = shape.topology().id;
    let mut components = path("package")
        .map(|root| package_components(&root, topology))
        .transpose()?
        .unwrap_or_default();
    // The activation policy preparation resolves the precision with.
    let policy = model_compiler_sdk::quantization::requested_policy(
        options.get("activation-policy").map(String::as_str),
    )
    .map_err(|e| e.to_string())?
    .unwrap_or_default();
    if command == "precision" {
        // The precision component a package of this checkpoint records,
        // resolved against the module, as preparation resolves it.
        let module = path("prepare-module").ok_or("precision needs --prepare-module")?;
        let output = path("output").ok_or("precision needs --output")?;
        let catalog = model_qwen38::catalog::Catalog::load_ranks(&checkpoint_root, ranks)
            .map_err(|e| e.to_string())?;
        let module = model_compiler_sdk::native_contract::NativeModule::inspect(&module)
            .map_err(|e| e.to_string())?;
        let precision = model_qwen38::preparation::resolve_catalog(&catalog, policy, &module)
            .map_err(|e| e.to_string())?;
        let bytes =
            kernel_core::precision::component::encode(&precision).map_err(|e| format!("{e:?}"))?;
        std::fs::write(&output, &bytes)?;
        println!(
            "{}",
            serde_json::json!({"command": "precision", "bytes": bytes.len(), "output": output})
        );
        return Ok(());
    }
    if let Some(module) = path("prepare-module") {
        let catalog = model_qwen38::catalog::Catalog::load_ranks(&checkpoint_root, ranks)
            .map_err(|e| e.to_string())?;
        let module = model_compiler_sdk::native_contract::NativeModule::inspect(&module)
            .map_err(|e| e.to_string())?;
        let precision = model_qwen38::preparation::resolve_catalog(&catalog, policy, &module)
            .map_err(|e| e.to_string())?;
        components.push(NamedBytesV1 {
            name: model_qwen38_contract::prepared::COMPONENT_ROLE.into(),
            bytes: kernel_core::precision::component::encode(&precision)
                .map_err(|e| format!("{e:?}"))?,
        });
        components.sort_by(|a, b| a.name.cmp(&b.name));
    }
    // With --module, plans come through the built bundle's export and the
    // host's loader, as serving asks for them.
    // The module's digest is in the planned identities, as the host states it.
    let mut bundle_module = [0_u8; 32];
    let library = path("module")
        .map(|module| -> Result<_> {
            let bytes = std::fs::read(&module)?;
            bundle_module = <sha2::Sha256 as sha2::Digest>::digest(&bytes).into();
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
    if encoders && layers.is_some() {
        return Err("--layers selects text layers, not encoder images".into());
    }
    let planning = Instant::now();
    let mut layered = None;
    let plans = (0..ranks as u32)
        .map(|rank| {
            let request = checkpoint.request(
                role,
                ranks as u32,
                rank,
                implementation,
                components.clone(),
                Vec::new(),
            );
            if let Some(layers) = &layers {
                if layered.is_none() {
                    layered = Some(plan_layers(&checkpoint_root, &request, layers)?);
                }
                let planned = layered.as_ref().expect("planned above");
                return Ok(model_qwen38_contract::weight_plan::rank_plan(
                    &request, planned,
                )?);
            }
            match &library {
                Some(library) => library.weight_plan(&request),
                None => model_qwen38_contract::weight_plan::plan(&request),
            }?
            .ok_or_else(|| format!("the bundle declined {role}").into())
        })
        .collect::<Result<Vec<_>>>()?;
    let plan_seconds = planning.elapsed().as_secs_f64();
    let records: usize = plans.iter().map(|p| p.records.len()).sum();
    let planned_indexes: Vec<serde_json::Value> = plans
        .iter()
        .map(|plan| {
            let component = plan
                .components
                .iter()
                .find(|c| c.name == index_role(plan.rank))
                .ok_or("plan has no rank index")?;
            Ok(serde_json::from_slice(&component.bytes)?)
        })
        .collect::<Result<_>>()?;
    let mut ledger_ranks: Vec<serde_json::Value> = plans.iter().map(ledger::rank_entry).collect();
    if command == "plan" {
        let mut coverage = Vec::new();
        for (plan, index) in plans.iter().zip(&planned_indexes) {
            coverage.push(tiles(plan, index)?);
        }
        // The host checks every record against the checkpoint headers.
        PlannedImageV1::new(
            checkpoint.clone(),
            plans,
            None,
            implementation,
            DirectLoadOptionsV1 {
                threads_per_rank: 1,
                read_ahead: false,
                direct_reads: false,
                shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
            },
        )?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "command": "plan", "ranks": ranks, "records": records, "plan_seconds": plan_seconds,
                "checkpoint_identity": checkpoint.identity_hex(), "per_rank": coverage,
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
    match command.as_str() {
        "verify" if reference.is_none() => {
            return Err("verify needs --reference LEDGER".into());
        }
        "verify" | "bench" | "digest" => {}
        _ if reference.is_some() => return Err("--reference is a verify option".into()),
        other => return Err(format!("unknown command {other}").into()),
    }
    let digest = options.contains_key("digest") || command == "digest" || reference.is_some();
    // `--view-digests FILE` writes each rank's view digests to FILE.
    let view_digests = path("view-digests");
    if view_digests.is_some() && command != "digest" {
        return Err("--view-digests is a digest option".into());
    }
    let view_rows: Vec<String> = options
        .get("view-rows")
        .map(|list| list.split(',').map(str::to_owned).collect())
        .unwrap_or_default();
    // The text image's n-gram table views, digested per head.
    let ngram_view = ngram_view_for(shape);
    let ngram_stem = ngram_view.trim_end_matches(".weight");
    let ngram_heads: Vec<Vec<u64>> = if encoders {
        Vec::new()
    } else {
        let tables = contract::ngram_tables_for(shape.geometry).map_err(|e| format!("{e:?}"))?;
        let local = shape.ngram_local_heads as usize;
        (0..shape.ranks() as usize)
            .map(|rank| {
                let mut rows: Vec<u64> = tables.sizes[rank * local..(rank + 1) * local].to_vec();
                if rank + 1 == shape.ranks() as usize {
                    let last = rows.len() - 1;
                    rows[last] = tables.rows - tables.offsets[(rank + 1) * local - 1];
                }
                rows
            })
            .collect()
    };
    let (view_digests, view_rows, planned_indexes, ngram_heads) =
        (&view_digests, &view_rows, &planned_indexes, &ngram_heads);
    let cached_before = cached_bytes();
    if cold {
        evict(&checkpoint_root)?;
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
                    let mut digest = digest.then(sha2::Sha256::default);
                    let mut view_digests = view_digests
                        .is_some()
                        .then(|| {
                            let heads = ngram_heads.get(rank).map(|rows| NgramHeads {
                                stem: ngram_stem,
                                rows,
                            });
                            ViewDigests::new(&planned_indexes[rank], view_rows, heads)
                        })
                        .transpose()?;
                    let rank_started = Instant::now();
                    let mut offset = 0_u64;
                    while offset < arena {
                        let count = (arena - offset).min(chunk as u64) as usize;
                        image.read(base + offset, &mut buffer[..count])?;
                        if let Some(digest) = digest.as_mut() {
                            sha2::Digest::update(digest, &buffer[..count]);
                        }
                        if let Some(views) = view_digests.as_mut() {
                            views.update(offset, &buffer[..count]);
                        }
                        offset += count as u64;
                    }
                    Ok(serde_json::json!({
                        "rank": rank, "bytes": arena,
                        "sha256": digest.map(|d| format!("{:x}", sha2::Digest::finalize(d))),
                        "seconds": rank_started.elapsed().as_secs_f64(),
                        "views": view_digests.map(ViewDigests::finish),
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
    let mut ranks_json = Vec::new();
    let mut failures = Vec::new();
    for result in results {
        match result {
            Ok(rank) => ranks_json.push(rank),
            Err(error) => failures.push(error.to_string()),
        }
    }
    if let Some(output) = view_digests {
        let per_rank: Vec<_> = ranks_json
            .iter_mut()
            .map(|rank| rank["views"].take())
            .collect();
        std::fs::write(
            output,
            serde_json::to_vec(&serde_json::json!({
                "checkpoint_identity": checkpoint.identity_hex(), "ranks": ranks,
                "view_rows": view_rows, "per_rank": per_rank,
            }))?,
        )?;
    }
    for rank in &mut ranks_json {
        if let Some(rank) = rank.as_object_mut() {
            rank.remove("views");
        }
    }
    let stats = image.stats();
    let bytes: u64 = (0..ranks).map(|rank| image.arena_bytes(rank)).sum();
    let ledger = digest.then(|| {
        for (entry, rank) in ledger_ranks.iter_mut().zip(&ranks_json) {
            entry["bytes_sha256"] = rank["sha256"].clone();
        }
        serde_json::json!({
            "format": ledger::FORMAT, "image_role": role, "ranks": ranks,
            "checkpoint_identity": checkpoint.identity_hex(),
            "planner": if library.is_some() { "bundle export" } else { "contract" },
            "per_rank": ledger_ranks,
        })
    });
    let reference_differences = reference.as_ref().map(|reference| {
        let current = ledger.clone().unwrap_or_default();
        let mut differences = ledger::compare(reference, &current);
        differences.extend(failures.iter().cloned());
        differences
    });
    let report = serde_json::json!({
        "command": command,
        "equal": reference_differences
            .as_ref()
            .is_some_and(|differences| differences.is_empty()),
        "reference_differences": reference_differences,
        "ledger": ledger,
        "failures": failures,
        "checkpoint_identity": checkpoint.identity_hex(),
        "planner": if library.is_some() { "bundle export" } else { "contract" },
        "image_role": role,
        "layers": layers,
        "ranks": ranks, "records": records, "threads_per_rank": threads, "read_ahead": read_ahead, "direct_reads": direct_reads,
        "chunk_bytes": chunk, "cold": cold,
        "open_seconds": open_seconds, "plan_seconds": plan_seconds, "prepare_seconds": prepare_seconds,
        "wall_seconds": wall,
        "image_bytes": bytes, "image_gib_per_second": bytes as f64 / GIB / wall,
        "source_bytes": stats.source_bytes, "source_reads": stats.source_reads,
        "advised_bytes": stats.advised_bytes, "direct_bytes": stats.direct_bytes,
        "read_thread_seconds": stats.read_seconds, "produce_thread_seconds": stats.produce_seconds,
        "shared_thread_seconds": stats.shared_seconds,
        "shared_retained_peak_bytes": stats.retained_peak_bytes, "rank_read_seconds": stats.rank_seconds,
        "cpu_user_seconds": user1 - user0, "cpu_system_seconds": system1 - system0,
        "average_busy_cpus": (user1 - user0 + system1 - system0) / wall,
        "cached_bytes": {"before_evict": cached_before, "at_start": cached_start, "at_end": cached_bytes()},
        "allowed_cpus": allowed_cpus(), "peak_rss_bytes": peak_rss_bytes(),
        "per_rank": ranks_json,
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if (report["command"] == "verify" && report["equal"] != true)
        || (report["command"] == "digest" && !failures.is_empty())
    {
        std::process::exit(2);
    }
    Ok(())
}

/// Runs the bundle's create over a direct load of `checkpoint` on the fake
/// device, as serving would, with serving KV, 32 sequences and
/// no speculation.
/// The serving options `create` runs the bundle's create with. By default
/// a probe: speculation off, 32 sequences and at most 4096 KV pages; with
/// `--kv-fraction-millionths` the KV pool takes that share of the free
/// memory left after create's allocations, as `serve` sizes it.
struct CreateOptions {
    allocation_alignment: u64,
    device_memory_bytes: u64,
    speculative_tokens: u32,
    verify_tokens: u32,
    maximum_sequences: u32,
    kv_fraction_millionths: Option<u32>,
}

fn create(
    checkpoint: &Path,
    bundle: &Path,
    topology: &str,
    ranks: usize,
    options: CreateOptions,
) -> Result<()> {
    let allocation_alignment = options.allocation_alignment;
    runtime_host::direct::select_checkpoint_v1(checkpoint)?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(bundle.join("manifest.json"))?)?;
    let capabilities = manifest["resource_requirements"]["required_capabilities"]
        .as_array()
        .ok_or("manifest capabilities")?
        .iter()
        .map(|c| runtime_bundle::BundleOpaqueId::new(c.as_str().unwrap_or_default()))
        .chain(std::iter::once(runtime_bundle::BundleOpaqueId::new(
            runtime_bundle_abi::REQUEST_RETIREMENT_CAPABILITY_V1,
        )))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let host = runtime_bundle::BundleHostCompatibilityV1::new(
        "sm_120f",
        runtime_bundle::MODEL_BUNDLE_V1_ABI_VERSION,
    )?
    .with_capabilities(capabilities);
    let started = Instant::now();
    let loaded = runtime_host::load_selected_topology_bundle_v1(
        &runtime_host::BundleSelectionV1::Directory(bundle.to_path_buf()),
        &host,
        topology,
    )?;
    let mut device = runtime_host::fake::FakeHostDeviceV1::default();
    device.use_sparse_probe_allocations(true);
    device.set_allocation_alignment(allocation_alignment);
    // An RTX PRO 6000's 96 GiB per rank, and its report of 46 resident
    // 8-CTA clusters on 188 SMs for the communication entries.
    device.set_device_memory_bytes(options.device_memory_bytes);
    device.report_cluster_residency(runtime_bundle_abi::BundleRawClusterResidencyV1 {
        multiprocessor_count: 188,
        usable_multiprocessors: 188,
        max_active_clusters: 46,
        flags: runtime_bundle_abi::BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED,
        ..runtime_bundle_abi::BundleRawClusterResidencyV1::unreported()
    });
    // Launches are recorded and reported as ordered, so create's graph
    // setup runs to its end without a device.
    device.accept_launches(true);
    let result = loaded.create_instance_with_serving_options(
        device,
        topology,
        (0..ranks as i32).collect(),
        Some(match options.kv_fraction_millionths {
            Some(fraction_millionths) => runtime_host::ServingKvPolicyV1 {
                fraction_millionths,
                maximum_pages: None,
                ..Default::default()
            },
            None => runtime_host::ServingKvPolicyV1 {
                maximum_pages: Some(4096),
                ..Default::default()
            },
        }),
        {
            let mut speculation = runtime_bundle_abi::BundleRawSpeculationV1::disabled();
            if options.speculative_tokens != 0 {
                speculation.maximum_draft_tokens = options.speculative_tokens;
                speculation.maximum_verify_tokens = options.verify_tokens;
            }
            speculation
        },
        Some(options.maximum_sequences),
        None,
    );
    let (logs, refused, buffers) = match &result {
        Ok((backend, _)) => backend.with_device(|device| {
            (
                device
                    .logs()
                    .iter()
                    .map(|line| String::from_utf8_lossy(line).into_owned())
                    .collect::<Vec<_>>(),
                device.refused_launches().len(),
                device.buffer_objects().len(),
            )
        }),
        Err(_) => (Vec::new(), 0, 0),
    };
    let report = serde_json::json!({
        "command": "create", "bundle": bundle, "topology": topology,
        "created": result.is_ok(),
        "error": result.as_ref().err().map(ToString::to_string),
        "bundle_logs": logs, "refused_launches": refused, "device_buffers": buffers,
        "seconds": started.elapsed().as_secs_f64(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if result.is_err() {
        std::process::exit(2);
    }
    Ok(())
}

/// The planning of a `qwen38-materialize --layers` run: the catalog and
/// precision the contract's planner uses, with the image plan of `layers`,
/// and its n-gram placement. The catalog comes from the checkpoint's own
/// config and index, the files the host describes to the planner, so the
/// full 48-layer plan is never built.
fn plan_layers(
    root: &Path,
    request: &runtime_bundle_abi::weight_plan::WeightPlanRequestV1,
    layers: &BTreeSet<usize>,
) -> Result<model_qwen38_contract::weight_plan::Planned> {
    use model_qwen38_contract::placement::ImagePlan;
    let catalog = Arc::new(model_qwen38_contract::catalog::Catalog::load_ranks(
        root,
        request.ranks as usize,
    )?);
    let plan = match request.component(model_qwen38_contract::prepared::COMPONENT_ROLE) {
        Some(bytes) => ImagePlan::with_precision(
            &catalog,
            Some(layers),
            &kernel_core::precision::component::decode(bytes).map_err(|e| format!("{e:?}"))?,
        )?,
        None => ImagePlan::new(&catalog, Some(layers))?,
    };
    Ok(model_qwen38_contract::weight_plan::Planned {
        catalog,
        plan,
        host_ngram: host_ngram(request, layers)?,
    })
}

/// Whether a layer subset's plan places the n-gram table in host memory:
/// where the contract's planner does (`weight_plan::host_ngram`), and only
/// if the subset holds the table, which sits under the PLE layer.
fn host_ngram(
    request: &runtime_bundle_abi::weight_plan::WeightPlanRequestV1,
    layers: &BTreeSet<usize>,
) -> Result<bool> {
    Ok(model_qwen38_contract::weight_plan::host_ngram(request)?
        && layers.contains(&(model_qwen38_contract::contract::PLE_LAYER as usize)))
}

/// Checks that the records of one rank tile each view of its index exactly.
fn tiles(
    plan: &runtime_bundle_abi::weight_plan::WeightPlanV1,
    index: &serde_json::Value,
) -> Result<serde_json::Value> {
    let views = views(index)?;
    let mut records = plan.records.iter().peekable();
    let (mut transforms, mut covered) = (BTreeMap::<u32, (u64, u64, u64)>::new(), 0_u64);
    for view in &views {
        let (name, offset, bytes) = (
            view[0].as_str().unwrap_or_default(),
            view[1].as_u64().ok_or("view offset")?,
            view[2].as_u64().ok_or("view bytes")?,
        );
        let mut at = offset;
        while let Some(record) = records.next_if(|r| r.arena_offset < offset + bytes) {
            if record.arena_offset != at {
                return Err(
                    format!("rank {} view {name}: gap or overlap at {at}", plan.rank).into(),
                );
            }
            at += record.byte_length;
            let entry = transforms.entry(record.transform).or_default();
            entry.0 += 1;
            entry.1 += record.byte_length;
            entry.2 = entry.2.max(record.byte_length);
        }
        if at != offset + bytes {
            return Err(format!("rank {} view {name}: records end at {at}", plan.rank).into());
        }
        covered += bytes;
    }
    if records.next().is_some() {
        return Err(format!("rank {}: a record lies outside every view", plan.rank).into());
    }
    // Packed operand records larger than a staging chunk, by layout.
    let mut large = BTreeMap::<String, (u64, u64)>::new();
    for record in &plan.records {
        if record.transform == kernel_core::weight_transform::PACKED_OPERAND
            && record.byte_length > 64 << 20
        {
            let operand = kernel_core::weight_transform::PackedOperand::decode(&record.params)?;
            let key = format!(
                "{} {:?} {:?} rows {} matrices {}",
                operand.plan.layout.id,
                operand.plan.format,
                operand.section,
                operand.plan.rows,
                operand.matrices.len()
            );
            let entry = large.entry(key).or_default();
            entry.0 += 1;
            entry.1 = entry.1.max(record.byte_length);
        }
    }
    Ok(serde_json::json!({
        "rank": plan.rank, "arena_bytes": plan.arena_bytes, "views": views.len(),
        "view_bytes": covered, "host_views": plan.views.len(),
        "large_packed_records": large.iter().map(|(key, (count, largest))| serde_json::json!({"kind": key, "records": count, "largest": largest})).collect::<Vec<_>>(),
        "transforms": transforms.iter().map(|(id, (count, bytes, largest))| serde_json::json!({"id": id, "records": count, "bytes": bytes, "largest": largest})).collect::<Vec<_>>(),
    }))
}

/// One thread's fragment packing throughput, the reference packer against
/// `weight_transform`'s, over the served matrices' shapes; the outputs are
/// compared as they are produced.
fn packbench(gib: f64) -> Result<()> {
    use kernel_core::transform::Transform;
    use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
    let mut rows = Vec::new();
    for (label, format, matrix_rows, k) in [
        (
            "fp8 expert 640x2560",
            FragmentWeightFormat::Fp8E4M3,
            640,
            2560,
        ),
        (
            "fp8 expert 2560x640",
            FragmentWeightFormat::Fp8E4M3,
            2560,
            640,
        ),
        ("bf16 2560x2560", FragmentWeightFormat::Bf16, 2560, 2560),
        (
            "bf16 lm_head 62080x2560 (strips of 4096 rows)",
            FragmentWeightFormat::Bf16,
            4096,
            2560,
        ),
    ] {
        let layout = FragmentWeightLayout {
            rows: matrix_rows,
            k,
            format,
        };
        let bytes = matrix_rows * k * layout.element_bytes();
        let source: Vec<u8> = (0..bytes)
            .map(|i| (i.wrapping_mul(2_654_435_761) >> 11) as u8)
            .collect();
        let repeats = ((gib * GIB) / bytes as f64).ceil().max(1.0) as usize;
        let started = Instant::now();
        let mut reference = Vec::new();
        for _ in 0..repeats.div_ceil(8) {
            reference = layout.pack(&source).map_err(|e| e.0)?.bytes;
        }
        let reference_seconds = started.elapsed().as_secs_f64() / repeats.div_ceil(8) as f64;
        let started = Instant::now();
        let mut fast = Vec::new();
        for _ in 0..repeats {
            fast = kernel_core::weight_transform::fragment::pack(layout, &source)?;
        }
        let fast_seconds = started.elapsed().as_secs_f64() / repeats as f64;
        if fast != reference {
            return Err(format!("{label}: packers differ").into());
        }
        rows.push(serde_json::json!({
            "matrix": label, "bytes": bytes, "equal": true,
            "reference_gib_per_core_second": bytes as f64 / GIB / reference_seconds,
            "whole_segment_gib_per_core_second": bytes as f64 / GIB / fast_seconds,
            "speedup": reference_seconds / fast_seconds,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "command": "packbench", "allowed_cpus": allowed_cpus(), "rows": rows,
        }))?
    );
    Ok(())
}

/// Drops the page cache of every safetensors file of the checkpoint.
fn evict(root: &Path) -> Result<()> {
    let mut names: Vec<_> = std::fs::read_dir(root)?
        .map(|entry| entry.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    names.sort();
    let mut files = BTreeMap::new();
    for name in names.into_iter().filter(|n| n.ends_with(".safetensors")) {
        let file = runtime_weights::checkpoint::CheckpointFile::open(root, &name)?;
        files.insert(name, file);
    }
    for file in files.values() {
        advise_v1(file, 0, 0, CacheAdviceV1::DontNeed)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{NgramHeads, ViewDigests};
    use sha2::Digest;

    fn hex(bytes: &[u8]) -> String {
        format!("{:x}", sha2::Sha256::digest(bytes))
    }

    /// Views fed in uneven chunks across a gap digest whole, per leading
    /// index of a bank, per n-gram head and per row of a named matrix.
    #[test]
    fn view_digests_split_banks_heads_and_named_rows() {
        let index = serde_json::json!({"views": [
            {"name": "bank", "offset": 0, "bytes": 12, "dtype": "U8", "shape": [2, 3, 2],
             "layout_id": "l"},
            {"name": "table.weight", "offset": 16, "bytes": 8, "dtype": "U8", "shape": [4, 2],
             "layout_id": "l"},
            {"name": "named.rows", "offset": 24, "bytes": 4, "dtype": "U8", "shape": [2, 2],
             "layout_id": "l"},
            {"name": "other", "offset": 28, "bytes": 4, "dtype": "U8", "shape": [2, 2],
             "layout_id": "l"},
            {"name": "table.weight_scale", "offset": 32, "bytes": 8, "dtype": "U8", "shape": [8],
             "layout_id": "l"},
            {"name": "table.weight_scale_2", "offset": 40, "bytes": 4, "dtype": "F32",
             "shape": [1], "layout_id": "l"},
        ]});
        let image: Vec<u8> = (0..44).collect();
        let heads = NgramHeads {
            stem: "table",
            rows: &[1, 3],
        };
        let mut digests = ViewDigests::new(&index, &["rows".into()], Some(heads)).unwrap();
        let mut at = 0;
        for chunk in image.chunks(5) {
            digests.update(at, chunk);
            at += chunk.len() as u64;
        }
        let views = digests.finish();
        assert_eq!(views[0]["sha256"], hex(&image[..12]));
        assert_eq!(
            views[0]["units"],
            serde_json::json!([hex(&image[..6]), hex(&image[6..12])])
        );
        // The n-gram table's heads of one and three rows.
        assert_eq!(
            views[1]["units"],
            serde_json::json!([hex(&image[16..18]), hex(&image[18..24])])
        );
        let rows: Vec<_> = (0..2)
            .map(|r| hex(&image[24 + 2 * r..26 + 2 * r])[..16].to_owned())
            .collect();
        assert_eq!(views[2]["units"], serde_json::json!(rows));
        assert_eq!(views[3]["sha256"], hex(&image[28..32]));
        assert!(views[3]["units"].is_null());
        // Flat per-row scales follow the heads; a scalar stays whole.
        assert_eq!(
            views[4]["units"],
            serde_json::json!([hex(&image[32..34]), hex(&image[34..40])])
        );
        assert!(views[5]["units"].is_null());
    }
}
