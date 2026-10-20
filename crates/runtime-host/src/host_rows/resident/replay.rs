//! Explicit CPU measurement replay through the normal resident copy path.
use super::*;
use std::io::Read;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Replay {
    planes: Vec<Range>,
    ids: PathBuf,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Range {
    offset: u64,
    length: u64,
}

pub(crate) fn run(groups: &[Replay]) -> Result<serde_json::Value, String> {
    if groups.is_empty() || groups.len() > 8 {
        return Err("row replay requires one to eight groups".into());
    }
    let started = std::time::Instant::now();
    let mappings = super::super::REGISTRY
        .lock()
        .map_err(|_| "mapped row registry is poisoned")?
        .values()
        .filter_map(Weak::upgrade)
        .filter(|mapping| mapping.key.identity.is_some())
        .collect::<Vec<_>>();
    let mut ids = 0;
    let mut copies = 0;
    for group in groups {
        let planes = bind(group, &mappings)?;
        let count = replay(group, &planes)?;
        ids += count;
        copies += count * planes.len() as u64;
    }
    Ok(serde_json::json!({
        "groups":groups.len(), "ids":ids, "plane_copies":copies,
        "seconds":started.elapsed().as_secs_f64(),
    }))
}

fn bind(group: &Replay, mappings: &[Arc<MappedRowsV1>]) -> Result<Vec<Arc<MappedRowsV1>>, String> {
    if group.planes.is_empty() || group.planes.len() > MAX_PLANES {
        return Err("row replay has an unsupported plane count".into());
    }
    let mut planes = Vec::new();
    for range in &group.planes {
        let found = mappings
            .iter()
            .filter(|mapping| {
                mapping.key.identity.is_some()
                    && mapping.key.offset == range.offset
                    && mapping.key.length == range.length
            })
            .collect::<Vec<_>>();
        if found.len() != 1 {
            return Err("row replay mapping range is absent or ambiguous".into());
        }
        let mapping = found[0];
        let plane = mapping
            .resident
            .get()
            .ok_or("row replay mapping has no resident cache")?;
        if planes.iter().any(|known| Arc::ptr_eq(known, mapping)) {
            return Err("row replay repeats a mapping".into());
        }
        if let Some(first) = planes.first()
            && !Arc::ptr_eq(
                &first.resident.get().expect("bound plane").cache,
                &plane.cache,
            )
        {
            return Err("row replay planes belong to different resident sets".into());
        }
        planes.push(Arc::clone(mapping));
    }
    Ok(planes)
}

fn replay(group: &Replay, planes: &[Arc<MappedRowsV1>]) -> Result<u64, String> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&group.ids)
        .map_err(|error| format!("open row replay IDs: {error}"))?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    let length = metadata.len();
    if !metadata.is_file() || length == 0 || length > 256 << 20 || !length.is_multiple_of(4) {
        return Err("row replay IDs require a bounded regular file of u32 words".into());
    }
    // A shard has its own LRU clock. FIFO queues preserve its input order;
    // separate shards can fault and copy concurrently without changing it.
    const WORKERS: usize = 8;
    std::thread::scope(|scope| -> Result<(), String> {
        let mut senders = Vec::new();
        let mut workers = Vec::new();
        for _ in 0..WORKERS {
            let (send, receive) = std::sync::mpsc::sync_channel::<Vec<u32>>(2);
            senders.push(send);
            workers.push(scope.spawn(move || -> Result<(), String> {
                let width = planes
                    .iter()
                    .map(|p| p.resident.get().unwrap().width)
                    .max()
                    .unwrap();
                let mut output = vec![0; width];
                while let Ok(rows) = receive.recv() {
                    for row in rows {
                        for mapping in planes {
                            let plane = mapping.resident.get().expect("bound plane");
                            if u64::from(row) >= mapping.len() / plane.width as u64 {
                                return Err(
                                    "row replay ID is outside its authenticated mapping".into()
                                );
                            }
                            plane.copy(mapping.bytes(), row, 0, &mut output[..plane.width]);
                        }
                    }
                }
                Ok(())
            }));
        }
        let read = (|| -> Result<(), String> {
            let mut reader = file.take(length);
            let mut remaining = length;
            let mut chunk = vec![0; 64 << 10];
            while remaining > 0 {
                let count = remaining.min(chunk.len() as u64) as usize;
                reader
                    .read_exact(&mut chunk[..count])
                    .map_err(|error| format!("read row replay IDs: {error}"))?;
                let mut batches: [Vec<u32>; WORKERS] = std::array::from_fn(|_| Vec::new());
                for word in chunk[..count].chunks_exact(4) {
                    let row = u32::from_le_bytes(word.try_into().expect("u32 row ID"));
                    let shard = planes[0].resident.get().unwrap().cache.bucket(row).0;
                    batches[shard % WORKERS].push(row);
                }
                for (sender, batch) in senders.iter().zip(batches) {
                    sender
                        .send(batch)
                        .map_err(|_| "row replay worker stopped")?;
                }
                remaining -= count as u64;
            }
            Ok(())
        })();
        drop(senders);
        let mut copied = Ok(());
        for worker in workers {
            let result = worker
                .join()
                .map_err(|_| "row replay worker panicked".to_owned())
                .and_then(std::convert::identity);
            if copied.is_ok() {
                copied = result;
            }
        }
        copied?;
        read
    })?;
    Ok(length / 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_replay_preserves_each_shards_serial_lru_state() {
        let registry = super::super::tests::registry(2);
        let data = super::super::tests::source(4000, 264);
        let parallel = super::super::tests::mapping("parallel-replay", &data, 28);
        let serial = super::super::tests::mapping("serial-replay", &data, 29);
        for mapping in [&parallel, &serial] {
            attach(&registry, vec![(Arc::clone(mapping), 264)]).unwrap();
        }
        let ids = (0..40_000_u32)
            .map(|i| (i * 7919 + i / 7) % 4000)
            .collect::<Vec<_>>();
        let path = std::env::temp_dir().join(format!("parallel-replay-{}", std::process::id()));
        std::fs::write(
            &path,
            ids.iter()
                .flat_map(|id| id.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let group = Replay {
            planes: vec![],
            ids: path.clone(),
        };
        assert_eq!(
            replay(&group, &[Arc::clone(&parallel)]).unwrap(),
            ids.len() as u64
        );
        for row in ids {
            super::super::tests::copy(&serial, 264, row, 0, 264);
        }
        let state = |mapping: &MappedRowsV1| {
            let cache = &mapping.resident.get().unwrap().cache;
            let mut result = Vec::new();
            for shard in &cache.shards {
                let mut shard = shard.lock().unwrap();
                let bytes = shard.memory.bytes();
                for bucket in 0..cache.buckets_per_shard {
                    let mut entries = (0..WAYS)
                        .map(|way| {
                            let at = bucket * cache.bucket_bytes + way * ENTRY_META;
                            (word(bytes, at + 8), word(bytes, at), word(bytes, at + 16))
                        })
                        .collect::<Vec<_>>();
                    entries.sort_unstable();
                    result.push(entries);
                }
            }
            result
        };
        assert_eq!(state(&parallel), state(&serial));
        assert_eq!(
            parallel
                .resident
                .get()
                .unwrap()
                .cache
                .contention
                .load(Ordering::Relaxed),
            0
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    #[ignore = "requires a staged trace manifest named by CHOPINFER_ROW_REPLAY_MANIFEST"]
    fn measure_staged_row_replay() {
        #[derive(serde::Deserialize)]
        struct InputPlane {
            file: PathBuf,
            offset: u64,
            length: u64,
            width: usize,
            identity: [u8; 32],
        }
        #[derive(serde::Deserialize)]
        struct Input {
            sets: Vec<Vec<InputPlane>>,
            replay: Vec<Replay>,
        }
        let input: Input = serde_json::from_slice(
            &std::fs::read(std::env::var_os("CHOPINFER_ROW_REPLAY_MANIFEST").unwrap()).unwrap(),
        )
        .unwrap();
        let started = std::time::Instant::now();
        let mut mappings = Vec::new();
        for set in input.sets {
            let mut registered = Vec::new();
            for plane in set {
                let file = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                    .open(plane.file)
                    .unwrap();
                let mapping = super::super::super::map_authenticated_rows_v1(
                    &file,
                    plane.offset,
                    plane.length,
                    plane.identity,
                )
                .unwrap();
                mappings.push(Arc::clone(&mapping));
                registered.push((mapping, plane.width));
            }
            super::super::register_group(registered).unwrap();
        }
        let allocation_seconds = started.elapsed().as_secs_f64();
        let warming = std::time::Instant::now();
        // Match the measurement job's sequential source warmer. Serving keeps
        // random advice; this is not a timing of the loader's prewarm policy.
        for mapping in &mappings {
            mapping
                .advise(super::super::super::RowAdviceV1::Normal)
                .unwrap();
        }
        let windows = mappings
            .iter()
            .flat_map(|mapping| {
                mapping
                    .windows(64 << 20)
                    .into_iter()
                    .map(move |(start, length)| (mapping, start, length))
            })
            .collect::<Vec<_>>();
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let workers = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        while let Some((mapping, start, length)) =
                            windows.get(next.fetch_add(1, Ordering::Relaxed))
                        {
                            mapping.prewarm_window(*start, *length).unwrap();
                        }
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        for mapping in &mappings {
            mapping
                .advise(super::super::super::RowAdviceV1::Random)
                .unwrap();
        }
        let warming_seconds = warming.elapsed().as_secs_f64();
        let result = run(&input.replay).unwrap();
        let snapshot = super::super::measurement_snapshot();
        assert_eq!(
            snapshot["sets"].as_array().unwrap().len(),
            input.replay.len()
        );
        println!(
            "ROW_REPLAY_MEASUREMENT {}",
            serde_json::json!({
                "allocation_seconds":allocation_seconds,"warming_seconds":warming_seconds,
                "warming_advice":"normal_then_random",
                "replay":result,"cache":snapshot,
            })
        );
        drop(mappings);
    }

    #[test]
    fn replay_preserves_plane_bytes_and_refuses_unbound_inputs() {
        let registry = super::super::tests::registry(1);
        let mapping =
            super::super::tests::mapping("replay", &super::super::tests::source(100, 264), 27);
        attach(&registry, vec![(Arc::clone(&mapping), 264)]).unwrap();
        let path = std::env::temp_dir().join(format!("row-replay-{}", std::process::id()));
        let ids: Vec<u8> = [17_u32, 3, 17]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        std::fs::write(&path, &ids).unwrap();
        let group = Replay {
            planes: vec![Range {
                offset: mapping.file_offset(),
                length: mapping.len(),
            }],
            ids: path.clone(),
        };
        let planes = bind(&group, &[Arc::clone(&mapping)]).unwrap();
        assert_eq!(replay(&group, &planes).unwrap(), 3);
        let cache = &mapping.resident.get().unwrap().cache;
        assert_eq!(cache.misses.load(Ordering::Relaxed), 2);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
        super::super::tests::copy(&mapping, 264, 17, 0, 264);
        assert_eq!(cache.hits.load(Ordering::Relaxed), 2);
        assert!(bind(&group, &[]).is_err());
        assert!(bind(&group, &[Arc::clone(&mapping), Arc::clone(&mapping)]).is_err());
        std::fs::write(&path, 100_u32.to_le_bytes()).unwrap();
        assert!(replay(&group, &planes).unwrap_err().contains("outside"));
        std::fs::write(&path, [1]).unwrap();
        assert!(replay(&group, &planes).unwrap_err().contains("u32 words"));
        let link = path.with_extension("link");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(
            replay(
                &Replay {
                    ids: link.clone(),
                    planes: vec![]
                },
                &planes
            )
            .is_err()
        );
        std::fs::remove_file(link).unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
