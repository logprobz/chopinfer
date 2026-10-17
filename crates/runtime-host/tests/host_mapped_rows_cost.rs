//! The cost of host-mapped gathers (`chop.host-mapped-gather.v1`) over a
//! real checkpoint's row tables, through the host's CUDA services: one
//! mapping per table for every rank, and per rank one `gather_mapped_rows`
//! call per plane (codes and scales) of random rows into that rank's device
//! buffer on its own stream, as a bundle without cross-rank dedupe issues
//! them.
//!
//! For each step size it drops the mappings' resident pages
//! (`MADV_DONTNEED`) and evicts the tables' clean pages from the page cache
//! (`POSIX_FADV_DONTNEED`), then reads with `mincore` how many stayed and
//! labels the first pass by what it saw: "cold" when no page was resident,
//! "partly-warm" otherwise. It gathers once so and once warm (the same rows
//! again), and records each rank's call time (the CPU gather and the copies
//! queued), page faults, resident pages, distinct rows and the time until
//! every copy is done. Rank 0's first pass faults the pages the other ranks
//! then find resident, so the later ranks' times are what a per-step dedupe
//! across ranks would save. Every rank's device bytes are checked against
//! the mapped rows.
//!
//! Opt-in, for one GPU job that has the host to itself:
//! - `CHOPINFER_HOST_ROWS_COST`: a new directory for `result.json`;
//! - `CHOPINFER_HOST_ROWS_CHECKPOINT`: the checkpoint directory;
//! - `CHOPINFER_HOST_ROWS_TABLES`: `codes:scales` tensor name pairs,
//!   comma-separated, each pair one table's two planes with equal row
//!   counts;
//! - `CHOPINFER_HOST_ROWS_PER_TOKEN` (default 24): rows each token gathers
//!   from each table;
//! - `CHOPINFER_KERNEL_GPUS`: the ranks' GPUs, comma-separated ordinals;
//!   without it the test plans and maps the tables for four ranks and stops.
//!
//! The measurement needs the CUDA backend (`--release --features
//! linux-cuda-host`). The same sequence runs on the host-memory device over a
//! small synthetic checkpoint in every build.
#![cfg(target_os = "linux")]
#![allow(clippy::single_range_in_vec_init)]

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use kernel_core::weight_transform::WeightTransform;
use runtime_bundle_abi::weight_plan::{
    PlannedViewV1, RecordSourceV1, ViewPlacementV1, WeightPlanV1, WeightRecordV1,
};
use runtime_bundle_abi::{
    DeviceBufferSpanV1, HostServicesTableV1, MappedRowGatherV1, RankArenaUploadV1,
};
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use runtime_host::fake::FakeHostDeviceV1;
use runtime_host::host_rows::{MappedRowsV1, RowAdviceV1};
use runtime_host::page_cache::{CacheAdviceV1, advise_v1};
use runtime_host::{
    HostDeviceBackendV1, HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1,
};
use serde_json::{Value, json};

const ROLE: &str = "host-rows-cost.v1";
const ALIGNMENT: u64 = 4096;
/// Tokens per measured step.
const TOKENS: [u64; 4] = [32, 512, 4096, 16384];

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// One plane of a table: its tensor, file and shape.
#[derive(Clone, Debug)]
struct Plane {
    tensor: String,
    file: u32,
    file_name: String,
    data_offset: u64,
    rows: u64,
    row_bytes: u64,
    shape: Vec<u64>,
}

impl Plane {
    fn bytes(&self) -> u64 {
        self.rows * self.row_bytes
    }
}

/// Finds `tensor` in the checkpoint's headers.
fn plane(checkpoint: &DirectCheckpointV1, tensor: &str) -> Plane {
    let request = checkpoint.request(ROLE, 1, 0, implementation_v1([0; 32]), vec![], vec![]);
    for (index, file) in request.files.iter().enumerate() {
        let header: Value = serde_json::from_slice(&file.header).unwrap();
        let Some(entry) = header.get(tensor) else {
            continue;
        };
        let shape: Vec<u64> = serde_json::from_value(entry["shape"].clone()).unwrap();
        let offsets: Vec<u64> = serde_json::from_value(entry["data_offsets"].clone()).unwrap();
        let bytes = offsets[1] - offsets[0];
        assert!(
            shape.len() >= 2 && shape[0] > 0 && bytes % shape[0] == 0,
            "{tensor}"
        );
        return Plane {
            tensor: tensor.into(),
            file: index as u32,
            file_name: file.name.clone(),
            data_offset: file.data_offset + offsets[0],
            rows: shape[0],
            row_bytes: bytes / shape[0],
            shape,
        };
    }
    panic!("tensor {tensor} is in no checkpoint file");
}

/// Every rank's plan: each plane mapped whole at an aligned offset, then one
/// page of device bytes no record covers.
fn plans(planes: &[Plane], ranks: u32) -> Vec<WeightPlanV1> {
    let mut views = Vec::new();
    let mut records = Vec::new();
    let mut at = 0;
    for plane in planes {
        views.push(PlannedViewV1 {
            id: plane.tensor.clone(),
            alignment: ALIGNMENT,
            placement: ViewPlacementV1::Mapped { prewarm: false },
            offset: at,
            byte_length: plane.bytes(),
        });
        records.push(WeightRecordV1 {
            arena_offset: at,
            byte_length: plane.bytes(),
            transform: WeightTransform::Copy.id(),
            params: vec![],
            sources: vec![RecordSourceV1 {
                file: plane.file,
                tensor: plane.tensor.clone(),
            }],
            axis: None,
            ranges: vec![],
            stored_shape: plane.shape.clone(),
        });
        at = (at + plane.bytes()).next_multiple_of(ALIGNMENT);
    }
    (0..ranks)
        .map(|rank| WeightPlanV1 {
            ranks,
            rank,
            image_role: ROLE.into(),
            alignment: ALIGNMENT,
            arena_bytes: at + ALIGNMENT,
            views: views.clone(),
            components: vec![],
            records: records.clone(),
        })
        .collect()
}

/// This process's (minor, major) page faults so far.
fn faults() -> (u64, u64) {
    // SAFETY-REVIEWED: getrusage writes one rusage record this frame owns.
    #[allow(unsafe_code)]
    let usage = unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, &mut usage), 0);
        usage
    };
    (usage.ru_minflt as u64, usage.ru_majflt as u64)
}

/// `count` row ids below `rows`, uniform, from `seed` (splitmix64).
fn row_ids(count: u64, rows: u64, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            u32::try_from((mixed ^ (mixed >> 31)) % rows).expect("row ids fit u32")
        })
        .collect()
}

/// The rows `ids` of `table`, row after row: what a gather writes.
fn expected_rows(table: &[u8], row_bytes: usize, ids: &[u32]) -> Vec<u8> {
    ids.iter()
        .flat_map(|&id| &table[id as usize * row_bytes..(id as usize + 1) * row_bytes])
        .copied()
        .collect()
}

/// Asks the kernel to evict the tables from memory: first this process's
/// resident pages of each mapping (`MADV_DONTNEED`), which a file advice
/// alone leaves mapped, then the page cache's clean pages of the file range
/// (`POSIX_FADV_DONTNEED`). Advice may leave pages (dirty ones, or ones
/// another process maps), so this answers how many pages of all mappings
/// are resident afterwards.
fn evict(root: &Path, planes: &[Plane], mappings: &[Arc<MappedRowsV1>]) -> u64 {
    for rows in mappings {
        rows.advise(RowAdviceV1::DontNeed).unwrap();
    }
    // The advice drops only pages wholly inside its range, so it covers each
    // plane's pages whole, its neighbours' shared end pages included.
    const PAGE: u64 = 4096;
    for plane in planes {
        let file = std::fs::File::open(root.join(&plane.file_name)).unwrap();
        let start = plane.data_offset - plane.data_offset % PAGE;
        let end = (plane.data_offset + plane.bytes()).next_multiple_of(PAGE);
        advise_v1(&file, start, end - start, CacheAdviceV1::DontNeed).unwrap();
    }
    for rows in mappings {
        rows.advise(RowAdviceV1::Random).unwrap();
    }
    mappings
        .iter()
        .map(|rows| rows.resident_pages().unwrap().0)
        .sum()
}

/// What one measurement reads: the checkpoint, its tables and the steps.
struct Run<'a> {
    root: &'a Path,
    tables: &'a str,
    per_token: u64,
    tokens: &'a [u64],
}

/// Plans and maps the tables for `ordinals.len()` ranks (four without
/// GPUs), then, with ranks, measures every step on `device` and answers the
/// result.
fn measure<D: HostDeviceBackendV1 + 'static>(
    run: &Run,
    ordinals: Option<Vec<i32>>,
    device: impl FnOnce() -> D,
) -> Option<Value> {
    let Run {
        root,
        tables,
        per_token,
        tokens,
    } = *run;
    let ranks = ordinals
        .as_ref()
        .map_or(4, |ordinals| ordinals.len() as u32);
    let started = Instant::now();
    let checkpoint = Arc::new(DirectCheckpointV1::open(root).unwrap());
    let pairs: Vec<(Plane, Plane)> = tables
        .split(',')
        .map(|pair| {
            let (codes, scales) = pair.split_once(':').expect("codes:scales");
            let pair = (plane(&checkpoint, codes), plane(&checkpoint, scales));
            assert_eq!(pair.0.rows, pair.1.rows, "{codes} and {scales} rows");
            pair
        })
        .collect();
    let planes: Vec<Plane> = pairs
        .iter()
        .flat_map(|(c, s)| [c.clone(), s.clone()])
        .collect();
    let image = PlannedImageV1::new(
        Arc::clone(&checkpoint),
        plans(&planes, ranks),
        None,
        implementation_v1([0; 32]),
        DirectLoadOptionsV1 {
            threads_per_rank: 1,
            read_ahead: false,
            direct_reads: false,
            shared_bytes: DirectLoadOptionsV1::SHARED_BYTES_DEFAULT,
        },
    )
    .unwrap();
    let map_seconds = started.elapsed().as_secs_f64();
    // One mapping per plane, whatever the rank count.
    let shared = planes.iter().all(|plane| {
        let first = image.mapped_view(0, plane.tensor.as_bytes()).unwrap();
        (1..ranks).all(|rank| {
            Arc::ptr_eq(
                first,
                image.mapped_view(rank, plane.tensor.as_bytes()).unwrap(),
            )
        })
    });
    let mapped_bytes = image.mapped_bytes();
    let mappings: Vec<_> = planes
        .iter()
        .map(|plane| Arc::clone(image.mapped_view(0, plane.tensor.as_bytes()).unwrap()))
        .collect();
    let resident = |mappings: &[Arc<MappedRowsV1>]| -> Value {
        mappings
            .iter()
            .map(|rows| {
                let (resident, pages) = rows.resident_pages().unwrap();
                json!({"resident_pages": resident, "pages": pages})
            })
            .collect()
    };

    // Without GPUs the plan and the mappings are all it checks.
    let Some(ordinals) = ordinals else {
        eprintln!(
            "host rows cost: mapped {mapped_bytes} bytes in {map_seconds:.3}s, one mapping per \
             plane: {shared}; skipped the rest: CHOPINFER_KERNEL_GPUS is not set"
        );
        assert!(shared, "ranks did not share one mapping per plane");
        return None;
    };
    let image = Arc::new(image);
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals.clone()).unwrap())
        .with_planned_compiled_image(Arc::clone(&image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(device(), config));
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let host = table.caller();
    let handles: Vec<_> = ordinals
        .iter()
        .enumerate()
        .map(|(rank, &ordinal)| {
            let context = host.select_context(rank as u32, ordinal).unwrap();
            let pool = host.select_memory_pool(context).unwrap();
            let stream = host.create_stream(context, 0).unwrap();
            let event = host.create_event(context).unwrap();
            (context, pool, stream, event)
        })
        .collect();
    let compiled = host.open_compiled_image(ROLE.as_bytes()).unwrap();
    let uploads: Vec<_> = handles
        .iter()
        .enumerate()
        .map(|(rank, &(context, pool, stream, _))| RankArenaUploadV1 {
            context,
            pool,
            stream,
            image: compiled,
            rank: rank as u32,
        })
        .collect();
    let arenas = host.upload_rank_arenas(&uploads).unwrap();
    let setup_seconds = started.elapsed().as_secs_f64();
    let memory = backend.host_memory_v1();

    let mut steps = Vec::new();
    for (step, &tokens) in tokens.iter().enumerate() {
        let count = tokens * per_token;
        let ids: Vec<Vec<u32>> = pairs
            .iter()
            .enumerate()
            .map(|(index, (codes, _))| row_ids(count, codes.rows, (step * 16 + index) as u64 + 1))
            .collect();
        let distinct: Vec<u64> = ids
            .iter()
            .map(|ids| {
                let mut unique = ids.clone();
                unique.sort_unstable();
                unique.dedup();
                unique.len() as u64
            })
            .collect();
        // Each rank's destination: each table's codes rows, then its scales
        // rows.
        let mut regions = Vec::new();
        let mut bytes = 0;
        for ((codes, scales), ids) in pairs.iter().zip(&ids) {
            for plane in [codes, scales] {
                regions.push((plane, ids, bytes, count * plane.row_bytes));
                bytes += count * plane.row_bytes;
            }
        }
        let device: Vec<_> = handles
            .iter()
            .map(|&(_, pool, stream, _)| {
                host.allocate_device_buffer(pool, stream, bytes, 256)
                    .unwrap()
            })
            .collect();
        let resident_before_cold = evict(root, &planes, &mappings);
        let evicted = resident(&mappings);
        let mut passes = Vec::new();
        let first = if resident_before_cold == 0 {
            "cold"
        } else {
            "partly-warm"
        };
        for pass in [first, "warm"] {
            let (minor, major) = faults();
            let begun = Instant::now();
            let mut rank_seconds = Vec::new();
            for (rank, &(_, _, stream, event)) in handles.iter().enumerate() {
                let called = Instant::now();
                for &(plane, ids, offset, length) in &regions {
                    host.gather_mapped_rows(
                        stream,
                        DeviceBufferSpanV1::new(device[rank], offset, length),
                        MappedRowGatherV1 {
                            image: compiled,
                            rank: rank as u32,
                            view_id: plane.tensor.as_bytes(),
                            row_bytes: plane.row_bytes,
                            rows: ids,
                        },
                    )
                    .unwrap();
                }
                host.record_event(event, stream).unwrap();
                rank_seconds.push(called.elapsed().as_secs_f64());
            }
            let gather_seconds = begun.elapsed().as_secs_f64();
            let (minor_after, major_after) = faults();
            for &(_, _, _, event) in &handles {
                host.synchronize_event(event).unwrap();
            }
            let done_seconds = begun.elapsed().as_secs_f64();
            passes.push(json!({
                "pass": pass, "rank_call_seconds": rank_seconds,
                "gather_seconds": gather_seconds, "done_seconds": done_seconds,
                "minor_faults": minor_after - minor, "major_faults": major_after - major,
                "resident_after": resident(&mappings),
            }));
        }
        // Every rank's device bytes are the mapped rows.
        let mut expected = Vec::with_capacity(bytes as usize);
        for &(plane, ids, _, _) in &regions {
            let table = image.mapped_view(0, plane.tensor.as_bytes()).unwrap();
            expected.extend(expected_rows(table.bytes(), plane.row_bytes as usize, ids));
        }
        for (rank, &(_, _, stream, _)) in handles.iter().enumerate() {
            let mut read = vec![0_u8; bytes as usize];
            host.copy_device_to_host(
                stream,
                &mut read,
                DeviceBufferSpanV1::new(device[rank], 0, bytes),
            )
            .unwrap();
            assert!(
                read == expected,
                "rank {rank} device rows differ at {tokens} tokens"
            );
        }
        // The host releases in reverse acquisition order.
        for buffer in device.into_iter().rev() {
            host.release_device_buffer(buffer).unwrap();
        }
        let summary = json!({
            "tokens": tokens, "rows_per_table": count, "distinct_rows": distinct,
            "bytes_per_rank": bytes, "resident_after_eviction": evicted,
            "resident_pages_before_first_pass": resident_before_cold,
            "passes": passes, "verified_ranks": ranks,
            "gather_staging_bytes": backend.host_memory_v1().gather_staging_bytes,
        });
        eprintln!("host rows cost: {summary}");
        steps.push(summary);
    }
    for arena in arenas.into_iter().rev() {
        host.release_rank_arena(arena).unwrap();
    }
    host.release_compiled_image(compiled).unwrap();
    for (context, pool, stream, event) in handles.into_iter().rev() {
        host.release_event(event).unwrap();
        host.release_stream(stream).unwrap();
        host.release_memory_pool(pool).unwrap();
        host.release_context(context).unwrap();
    }
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let result = json!({
        "ranks": ranks, "rows_per_token": per_token, "cpu_threads": threads,
        "tables": pairs.iter().map(|(c, s)| json!({
            "codes": c.tensor, "scales": s.tensor, "rows": c.rows,
            "code_row_bytes": c.row_bytes, "scale_row_bytes": s.row_bytes,
        })).collect::<Vec<_>>(),
        "mapped_bytes": mapped_bytes, "one_mapping_per_plane": shared,
        "map_seconds": map_seconds, "setup_seconds": setup_seconds,
        "pinned_bytes_after_setup": memory.pinned_bytes,
        "mapped_view_bytes_after_setup": memory.mapped_view_bytes,
        "steps": steps,
    });
    assert!(shared, "ranks did not share one mapping per plane");
    Some(result)
}

#[test]
fn host_mapped_rows_cost() {
    let (Some(out), Some(root), Some(tables)) = (
        env("CHOPINFER_HOST_ROWS_COST"),
        env("CHOPINFER_HOST_ROWS_CHECKPOINT"),
        env("CHOPINFER_HOST_ROWS_TABLES"),
    ) else {
        eprintln!("skipped: CHOPINFER_HOST_ROWS_COST, _CHECKPOINT or _TABLES is not set");
        return;
    };
    let (out, root) = (PathBuf::from(out), PathBuf::from(root));
    assert!(!out.exists(), "{} exists", out.display());
    let run = Run {
        root: &root,
        tables: &tables,
        per_token: env("CHOPINFER_HOST_ROWS_PER_TOKEN").map_or(24, |v| v.parse().unwrap()),
        tokens: &TOKENS,
    };
    let ordinals: Option<Vec<i32>> = env("CHOPINFER_KERNEL_GPUS").map(|ordinals| {
        ordinals
            .split(',')
            .map(|ordinal| ordinal.trim().parse().expect("GPU ordinal"))
            .collect()
    });
    #[cfg(feature = "linux-cuda-host")]
    let result = measure(&run, ordinals, || {
        runtime_host::cuda::CudaHostDeviceV1::new().unwrap()
    });
    #[cfg(not(feature = "linux-cuda-host"))]
    let result = {
        assert!(
            ordinals.is_none(),
            "measuring on GPUs needs --features linux-cuda-host"
        );
        measure(&run, None, FakeHostDeviceV1::new)
    };
    let Some(result) = result else {
        return;
    };
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
}

/// The measurement's whole sequence, releases included, on the host-memory
/// device over two small tables: every rank's device bytes are the gathered
/// rows at each step, and each first pass is labelled by the residency it
/// observed, which eviction does not promise.
#[test]
fn the_measurement_runs_on_the_host_memory_device() {
    let root = std::env::temp_dir().join(format!("host-rows-cost-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let rows = 3000_u64;
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, row_bytes) in [
        ("a.codes", 64),
        ("a.scales", 2),
        ("b.codes", 64),
        ("b.scales", 2),
    ] {
        let start = payload.len() as u64;
        payload.extend((0..rows * row_bytes).map(|i| (i * 7 + start + 1) as u8));
        header.insert(
            name.into(),
            json!({"dtype": "U8", "shape": [rows, row_bytes],
                   "data_offsets": [start, payload.len() as u64]}),
        );
    }
    let header = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(payload);
    // Written back, so the eviction can drop its clean pages where the file
    // system holds them in the page cache.
    let mut written = std::fs::File::create(root.join("model.safetensors")).unwrap();
    std::io::Write::write_all(&mut written, &file).unwrap();
    written.sync_all().unwrap();
    drop(written);
    let run = Run {
        root: &root,
        tables: "a.codes:a.scales,b.codes:b.scales",
        per_token: 3,
        tokens: &[1, 5, 40],
    };
    let result = measure(&run, Some(vec![0, 1, 2, 3]), FakeHostDeviceV1::new).unwrap();
    assert_eq!(result["one_mapping_per_plane"], true);
    for step in result["steps"].as_array().unwrap() {
        let cold = step["resident_pages_before_first_pass"] == 0;
        let label = if cold { "cold" } else { "partly-warm" };
        assert_eq!(step["passes"][0]["pass"], label, "{step}");
        assert_eq!(step["passes"][1]["pass"], "warm");
        assert_eq!(
            step["passes"][0]["rank_call_seconds"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }
    assert_eq!(result["steps"].as_array().unwrap().len(), 3);
    assert_eq!(result["steps"][2]["verified_ranks"], 4);
    assert_eq!(result["pinned_bytes_after_setup"], 0);
    // Each rank's stream holds one staging chunk of the host-memory device.
    assert_eq!(result["steps"][2]["gather_staging_bytes"], 4 * 4096);
    let _ = std::fs::remove_dir_all(&root);
}
