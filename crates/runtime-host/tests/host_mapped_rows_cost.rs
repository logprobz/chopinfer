//! The cost of host-mapped rows (`chop.host-mapped-rows.v1`) over a real
//! checkpoint's row tables, through the host's CUDA services: one mapping
//! per table for every rank, a CPU gather of random rows of both planes
//! (codes and scales) into rank 0's staging buffer, its copy into the other
//! ranks' staging buffers, and each rank's device copy on its own stream.
//!
//! For each step size it evicts the tables' pages from the page cache
//! (`POSIX_FADV_DONTNEED`, clean pages only), gathers once cold and once warm
//! (the same rows again), and records wall time, page faults, resident pages,
//! distinct rows and copy time. The device bytes are checked against the
//! mapped bytes.
//!
//! Opt-in, for one GPU job that has the host to itself:
//! - `CHOPINFER_HOST_ROWS_COST`: a new directory for `result.json`;
//! - `CHOPINFER_HOST_ROWS_CHECKPOINT`: the checkpoint directory;
//! - `CHOPINFER_HOST_ROWS_TABLES`: `codes:scales` tensor name pairs,
//!   comma-separated, each pair one table's two planes with equal row
//!   counts;
//! - `CHOPINFER_HOST_ROWS_PER_TOKEN` (default 24): rows each token gathers
//!   from each table;
//! - `CHOPINFER_HOST_ROWS_THREADS` (default 8): gather threads;
//! - `CHOPINFER_KERNEL_GPUS`: the ranks' GPUs, comma-separated ordinals;
//!   without it the test plans and maps the tables for four ranks and stops.
//!
//! It needs the CUDA backend: `--release --features linux-cuda-host`.
#![cfg(all(target_os = "linux", feature = "linux-cuda-host"))]
#![allow(clippy::single_range_in_vec_init)]

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use kernel_core::weight_transform::WeightTransform;
use runtime_bundle_abi::weight_plan::{
    PlannedViewV1, RecordSourceV1, ViewPlacementV1, WeightPlanV1, WeightRecordV1,
};
use runtime_bundle_abi::{DeviceBufferSpanV1, HostServicesTableV1, RankArenaUploadV1};
use runtime_host::cuda::CudaHostDeviceV1;
use runtime_host::direct::{
    DirectCheckpointV1, DirectLoadOptionsV1, PlannedImageV1, implementation_v1,
};
use runtime_host::page_cache::{CacheAdviceV1, advise_v1};
use runtime_host::{HostInstanceConfigV1, HostServicesV1Backend, HostTopologyV1};
use serde_json::{Value, json};

const ROLE: &str = "host-rows-cost.v1";
const ALIGNMENT: u64 = 4096;
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
            placement: ViewPlacementV1::Mapped,
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
fn row_ids(count: u64, rows: u64, seed: u64) -> Vec<u64> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (mixed ^ (mixed >> 31)) % rows
        })
        .collect()
}

/// Gathers `ids`' rows of `table` into `out`, row after row, on `threads`
/// threads.
fn gather(table: &[u8], row_bytes: usize, ids: &[u64], out: &mut [u8], threads: usize) {
    let per = ids.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (ids, out) in ids.chunks(per).zip(out.chunks_mut(per * row_bytes)) {
            scope.spawn(move || {
                for (id, row) in ids.iter().zip(out.chunks_mut(row_bytes)) {
                    let at = *id as usize * row_bytes;
                    row.copy_from_slice(&table[at..at + row_bytes]);
                }
            });
        }
    });
}

fn evict(root: &Path, planes: &[Plane]) {
    for plane in planes {
        let file = std::fs::File::open(root.join(&plane.file_name)).unwrap();
        advise_v1(
            &file,
            plane.data_offset,
            plane.bytes(),
            CacheAdviceV1::DontNeed,
        )
        .unwrap();
    }
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
    let per_token: u64 = env("CHOPINFER_HOST_ROWS_PER_TOKEN").map_or(24, |v| v.parse().unwrap());
    let threads: usize = env("CHOPINFER_HOST_ROWS_THREADS").map_or(8, |v| v.parse().unwrap());
    let ordinals: Option<Vec<i32>> = env("CHOPINFER_KERNEL_GPUS").map(|ordinals| {
        ordinals
            .split(',')
            .map(|ordinal| ordinal.trim().parse().expect("GPU ordinal"))
            .collect()
    });
    let ranks = ordinals
        .as_ref()
        .map_or(4, |ordinals| ordinals.len() as u32);

    let started = Instant::now();
    let checkpoint = Arc::new(DirectCheckpointV1::open(&root).unwrap());
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
    let resident = |mappings: &[Arc<runtime_host::host_rows::MappedRowsV1>]| -> Value {
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
        return;
    };
    let config = HostInstanceConfigV1::new(HostTopologyV1::new(ordinals.clone()).unwrap())
        .with_planned_compiled_image(Arc::new(image))
        .unwrap();
    let backend = Rc::new(HostServicesV1Backend::new(
        CudaHostDeviceV1::new().unwrap(),
        config,
    ));
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
    for (step, tokens) in TOKENS.into_iter().enumerate() {
        let count = tokens * per_token;
        let ids: Vec<Vec<u64>> = pairs
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
        // The staging layout: each table's codes rows, then its scales rows.
        let layout: Vec<(u64, u64)> = pairs
            .iter()
            .map(|(codes, scales)| (count * codes.row_bytes, count * scales.row_bytes))
            .collect();
        let bytes: u64 = layout.iter().map(|(c, s)| c + s).sum();
        let staging: Vec<_> = handles
            .iter()
            .map(|&(_, pool, stream, _)| {
                host.allocate_staging_buffer(pool, stream, bytes, 256)
                    .unwrap()
            })
            .collect();
        let device: Vec<_> = handles
            .iter()
            .map(|&(_, pool, stream, _)| {
                host.allocate_device_buffer(pool, stream, bytes, 256)
                    .unwrap()
            })
            .collect();
        evict(&root, &planes);
        let evicted = resident(&mappings);
        let mut passes = Vec::new();
        for pass in ["cold", "warm"] {
            let (minor, major) = faults();
            let begun = Instant::now();
            host.with_staging_buffer(staging[0], |out| {
                let mut at = 0_usize;
                for (((codes, scales), ids), (code_bytes, scale_bytes)) in
                    pairs.iter().zip(&ids).zip(&layout)
                {
                    for (plane, region) in [(codes, *code_bytes), (scales, *scale_bytes)] {
                        let region = region as usize;
                        host.with_mapped_view(compiled, 0, plane.tensor.as_bytes(), |rows| {
                            gather(
                                rows,
                                plane.row_bytes as usize,
                                ids,
                                &mut out[at..at + region],
                                threads,
                            );
                        })
                        .unwrap();
                        at += region;
                    }
                }
            })
            .unwrap();
            let gather_seconds = begun.elapsed().as_secs_f64();
            let (minor_after, major_after) = faults();
            let begun = Instant::now();
            host.with_staging_buffer(staging[0], |first| {
                for &other in &staging[1..] {
                    host.with_staging_buffer(other, |out| out.copy_from_slice(first))
                        .unwrap();
                }
            })
            .unwrap();
            let replicate_seconds = begun.elapsed().as_secs_f64();
            let begun = Instant::now();
            for (rank, &(_, _, stream, event)) in handles.iter().enumerate() {
                host.copy_device_to_device(
                    stream,
                    DeviceBufferSpanV1::new(device[rank], 0, bytes),
                    DeviceBufferSpanV1::new(staging[rank], 0, bytes),
                )
                .unwrap();
                host.record_event(event, stream).unwrap();
            }
            for &(_, _, _, event) in &handles {
                host.synchronize_event(event).unwrap();
            }
            let copy_seconds = begun.elapsed().as_secs_f64();
            passes.push(json!({
                "pass": pass, "gather_seconds": gather_seconds,
                "minor_faults": minor_after - minor, "major_faults": major_after - major,
                "replicate_seconds": replicate_seconds, "copy_seconds": copy_seconds,
                "resident_after": resident(&mappings),
            }));
        }
        // Every rank's device bytes are the mapped rows.
        let mut expected = vec![0_u8; bytes as usize];
        let mut at = 0_usize;
        for (((codes, scales), ids), (code_bytes, scale_bytes)) in
            pairs.iter().zip(&ids).zip(&layout)
        {
            for (plane, region) in [(codes, *code_bytes), (scales, *scale_bytes)] {
                let region = region as usize;
                host.with_mapped_view(compiled, 0, plane.tensor.as_bytes(), |rows| {
                    gather(
                        rows,
                        plane.row_bytes as usize,
                        ids,
                        &mut expected[at..at + region],
                        threads,
                    );
                })
                .unwrap();
                at += region;
            }
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
        for buffer in device.into_iter().chain(staging) {
            host.release_device_buffer(buffer).unwrap();
        }
        let summary = json!({
            "tokens": tokens, "rows_per_table": count, "distinct_rows": distinct,
            "staging_bytes_per_rank": bytes, "resident_after_eviction": evicted,
            "passes": passes, "verified_ranks": ranks,
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
    let result = json!({
        "ranks": ranks, "rows_per_token": per_token, "threads": threads,
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
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    assert!(shared, "ranks did not share one mapping per plane");
}
