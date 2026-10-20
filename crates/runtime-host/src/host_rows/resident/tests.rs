use super::*;
use std::fs::File;

#[path = "../gather/cache_tests.rs"]
mod gather_cache;

pub(super) fn registry(sets: usize) -> Registry {
    let set_bytes = HEADER_BYTES + SHARDS * super::super::page_bytes_v1();
    Registry {
        config: Config {
            set_bytes,
            total_bytes: set_bytes * sets,
        },
        budget: Arc::new(Budget {
            limit: set_bytes * sets,
            used: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
        }),
        caches: Mutex::new(BTreeMap::new()),
        fallback_logged: AtomicBool::new(false),
        allocation_unavailable: AtomicBool::new(false),
    }
}
pub(super) fn mapping(label: &str, bytes: &[u8], identity: u8) -> Arc<MappedRowsV1> {
    let path = std::env::temp_dir().join(format!("resident-{label}-{}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    let file = File::open(&path).unwrap();
    let mapping =
        super::super::map_authenticated_rows_v1(&file, 0, bytes.len() as u64, [identity; 32])
            .unwrap();
    std::fs::remove_file(path).unwrap();
    mapping
}
pub(super) fn source(rows: usize, width: usize) -> Vec<u8> {
    (0..rows * width)
        .map(|i| ((i * 31) ^ (i / 251) ^ (i / 4093)) as u8)
        .collect()
}
pub(super) fn copy(
    mapping: &MappedRowsV1,
    width: usize,
    row: u32,
    within: usize,
    length: usize,
) -> Vec<u8> {
    let mut output = vec![0xa5; length];
    mapping
        .resident_plane(width)
        .unwrap()
        .copy(mapping.bytes(), row, within, &mut output);
    assert_eq!(
        output,
        mapping.bytes()[row as usize * width + within..][..length]
    );
    output
}

#[test]
fn bulk_hits_return_unfilled_positions_and_match_single_row_copies() {
    let registry = registry(2);
    let codes = mapping("batch-codes", &source(256, 256), 21);
    let scales = mapping("batch-scales", &source(256, 8), 21);
    attach(
        &registry,
        vec![(Arc::clone(&codes), 256), (Arc::clone(&scales), 8)],
    )
    .unwrap();
    for row in [3, 7, 19] {
        copy(&codes, 256, row, 0, 256);
    }
    copy(&scales, 8, 7, 0, 8);
    let ids = [19, 5, 7, 3, 5, 200, 19];
    for (mapping, width, expected_misses) in [
        (&codes, 256, vec![1, 4, 5]),
        (&scales, 8, vec![0, 1, 3, 4, 5, 6]),
    ] {
        let plane = mapping.resident_plane(width).unwrap();
        let cache = &plane.cache;
        let before = [
            cache.hits.load(Ordering::Relaxed),
            cache.misses.load(Ordering::Relaxed),
            cache.insertions.load(Ordering::Relaxed),
            cache.evictions.load(Ordering::Relaxed),
        ];
        let mut actual = vec![0xa5; ids.len() * width];
        let misses = plane.copy_resident_rows(&ids, &mut actual);
        assert_eq!(misses, expected_misses);
        assert_eq!(
            cache.hits.load(Ordering::Relaxed) - before[0],
            (ids.len() - misses.len()) as u64
        );
        assert_eq!(
            [
                cache.misses.load(Ordering::Relaxed),
                cache.insertions.load(Ordering::Relaxed),
                cache.evictions.load(Ordering::Relaxed),
            ],
            before[1..]
        );
        for &position in &misses {
            assert!(
                actual[position * width..][..width]
                    .iter()
                    .all(|&b| b == 0xa5)
            );
            // The existing single-row path still owns miss fetch and insertion.
            plane.copy(
                mapping.bytes(),
                ids[position],
                0,
                &mut actual[position * width..][..width],
            );
        }
        let expected = ids
            .iter()
            .flat_map(|&row| copy(mapping, width, row, 0, width))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!(plane.copy_resident_rows(&ids, &mut actual).is_empty());
        assert_eq!(actual, expected);
        let shard = cache.bucket(ids[0]).0;
        let held = cache.shards[shard].lock().unwrap();
        let mut poisoned = vec![0xa5; ids.len() * width];
        let contended = plane.copy_resident_rows(&ids, &mut poisoned);
        assert_eq!(
            contended,
            ids.iter()
                .enumerate()
                .filter_map(|(position, &row)| (cache.bucket(row).0 == shard).then_some(position))
                .collect::<Vec<_>>()
        );
        drop(held);
        for position in contended {
            plane.copy(
                mapping.bytes(),
                ids[position],
                0,
                &mut poisoned[position * width..][..width],
            );
        }
        assert_eq!(poisoned, expected);
    }
}

#[test]
fn code_and_scale_planes_keep_exact_bits_and_share_one_charge() {
    let registry = registry(2);
    let codes = mapping("codes", &source(1000, 256), 1);
    let scales = mapping("scales", &source(1000, 8), 1);
    attach(
        &registry,
        vec![
            (Arc::clone(&codes), 256),
            (Arc::clone(&scales), 8),
            (Arc::clone(&codes), 256),
        ],
    )
    .unwrap();
    let cache = &codes.resident.get().unwrap().cache;
    assert!(Arc::ptr_eq(cache, &scales.resident.get().unwrap().cache));
    assert_eq!(
        registry.budget.used.load(Ordering::Relaxed),
        registry.config.set_bytes as u64
    );
    for row in [0, 17, 999, 17, 999, 0] {
        copy(&codes, 256, row, 0, 256);
        copy(&scales, 8, row, 0, 8);
    }
    assert_eq!(cache.misses.load(Ordering::Relaxed), 6);
    assert_eq!(cache.hits.load(Ordering::Relaxed), 6);
    attach(&registry, vec![(scales, 8), (Arc::clone(&codes), 256)]).unwrap();
    assert_eq!(
        registry.budget.used.load(Ordering::Relaxed),
        registry.config.set_bytes as u64
    );
}

#[test]
fn fetched_planes_publish_exact_bytes_and_the_next_batch_hits() {
    let registry = registry(1);
    let codes = mapping("fetched-codes", &source(1000, 256), 30);
    let scales = mapping("fetched-scales", &source(1000, 8), 30);
    attach(
        &registry,
        vec![(Arc::clone(&codes), 256), (Arc::clone(&scales), 8)],
    )
    .unwrap();
    let rows = [0, 17, 999, 17];
    let cache = &codes.resident.get().unwrap().cache;
    for mapping in [&codes, &scales, &codes, &scales] {
        let width = mapping.resident.get().unwrap().width;
        let mut out = vec![0xa5; rows.len() * width];
        super::super::fetch::copy_rows(mapping, width, &rows, &mut out).unwrap();
        for (position, &row) in rows.iter().enumerate() {
            assert_eq!(
                &out[position * width..][..width],
                &mapping.bytes()[row as usize * width..][..width]
            );
        }
    }
    assert_eq!(cache.copies.load(Ordering::Relaxed), 16);
    assert_eq!(cache.misses.load(Ordering::Relaxed), 6);
    assert_eq!(cache.hits.load(Ordering::Relaxed), 10);
    assert_eq!(cache.evictions.load(Ordering::Relaxed), 0);
}

#[test]
fn partial_plane_copy_never_publishes_unwritten_bytes() {
    let registry = registry(1);
    let mapping = mapping("partial", &source(100, 97), 2);
    attach(&registry, vec![(Arc::clone(&mapping), 97)]).unwrap();
    let cache = &mapping.resident.get().unwrap().cache;
    copy(&mapping, 97, 12, 3, 40);
    assert_eq!(cache.insertions.load(Ordering::Relaxed), 0);
    copy(&mapping, 97, 12, 0, 97);
    copy(&mapping, 97, 12, 45, 52);
    assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
}

#[test]
fn repeated_misses_evict_but_never_mix_rows() {
    let registry = registry(1);
    let mapping = mapping("evict", &source(4000, 264), 3);
    attach(&registry, vec![(Arc::clone(&mapping), 264)]).unwrap();
    let cache = &mapping.resident.get().unwrap().cache;
    let bucket = cache.bucket(0);
    let collisions = (0..4000)
        .filter(|&r| cache.bucket(r) == bucket)
        .take(WAYS + 4)
        .collect::<Vec<_>>();
    assert_eq!(collisions.len(), WAYS + 4);
    for _ in 0..3 {
        for &row in &collisions {
            copy(&mapping, 264, row, 0, 264);
        }
    }
    assert!(cache.evictions.load(Ordering::Relaxed) > 0);
}

#[test]
fn contended_cache_falls_back_without_waiting() {
    let registry = registry(1);
    let mapping = mapping("contention", &source(10, 8), 4);
    attach(&registry, vec![(Arc::clone(&mapping), 8)]).unwrap();
    let cache = &mapping.resident.get().unwrap().cache;
    let (shard, _) = cache.bucket(5);
    let _held = cache.shards[shard].lock().unwrap();
    copy(&mapping, 8, 5, 0, 8);
    assert_eq!(cache.contention.load(Ordering::Relaxed), 2);
}

#[test]
fn budget_prefers_first_set_and_releases_with_last_mapping() {
    let registry = registry(1);
    let first = mapping("first", &source(10, 8), 5);
    let next = mapping("next", &source(10, 8), 6);
    attach(&registry, vec![(Arc::clone(&first), 8)]).unwrap();
    attach(&registry, vec![(Arc::clone(&next), 8)]).unwrap();
    assert!(first.resident.get().is_some());
    assert!(next.resident.get().is_none());
    assert_eq!(registry.budget.skipped.load(Ordering::Relaxed), 1);
    drop(first);
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
    attach(&registry, vec![(next, 8)]).unwrap();
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
}

#[test]
fn independent_files_and_authentication_do_not_share_cached_bytes() {
    let registry = registry(2);
    let a = mapping("identity-a", &[1; 800], 7);
    let b = mapping("identity-b", &[2; 800], 7);
    attach(&registry, vec![(Arc::clone(&a), 8)]).unwrap();
    attach(&registry, vec![(Arc::clone(&b), 8)]).unwrap();
    assert!(!Arc::ptr_eq(
        &a.resident.get().unwrap().cache,
        &b.resident.get().unwrap().cache
    ));
    assert_eq!(copy(&a, 8, 3, 0, 8), vec![1; 8]);
    assert_eq!(copy(&b, 8, 3, 0, 8), vec![2; 8]);
}

#[test]
fn incompatible_group_is_refused_before_cache_publication() {
    let registry = registry(1);
    let a = mapping("mismatch-a", &source(10, 8), 8);
    let b = mapping("mismatch-b", &source(11, 8), 9);
    assert!(attach(&registry, vec![(Arc::clone(&a), 8), (b, 8)]).is_err());
    assert!(a.resident.get().is_none());
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
    assert!(attach(&registry, vec![(a, 0)]).is_err());
}

#[test]
fn concurrent_rank_copies_match_source_under_replacement() {
    let registry = registry(1);
    let mapping = mapping("ranks", &source(1000, 97), 9);
    attach(&registry, vec![(Arc::clone(&mapping), 97)]).unwrap();
    std::thread::scope(|scope| {
        for rank in 0..4 {
            let mapping = &mapping;
            scope.spawn(move || {
                for i in 0..4000 {
                    copy(mapping, 97, (i * 37 + rank * 19) % 1000, 0, 97);
                }
            });
        }
    });
    assert_eq!(
        mapping
            .resident
            .get()
            .unwrap()
            .cache
            .copies
            .load(Ordering::Relaxed),
        16000
    );
}

#[test]
fn mapped_identity_separates_checkpoint_authentication() {
    let path = std::env::temp_dir().join(format!("resident-auth-{}", std::process::id()));
    std::fs::write(&path, [0; 100]).unwrap();
    let file = File::open(&path).unwrap();
    let a = super::super::map_authenticated_rows_v1(&file, 0, 100, [1; 32]).unwrap();
    let b = super::super::map_authenticated_rows_v1(&file, 0, 100, [2; 32]).unwrap();
    assert!(!Arc::ptr_eq(&a, &b));
    let c = super::super::map_authenticated_rows_v1(&file, 0, 100, [1; 32]).unwrap();
    assert!(Arc::ptr_eq(&a, &c));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn measurement_reset_discards_tags_and_keeps_the_same_allocation_and_bytes() {
    let registry = registry(1);
    let mapping = mapping("measurement-reset", &source(100, 8), 23);
    attach(&registry, vec![(Arc::clone(&mapping), 8)]).unwrap();
    let cache = &mapping.resident.get().unwrap().cache;
    let charge = registry.budget.used.load(Ordering::Relaxed);
    let first = copy(&mapping, 8, 17, 0, 8);
    assert_eq!(copy(&mapping, 8, 17, 0, 8), first);
    assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
    cache.reset_for_measurement().unwrap();
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), charge);
    assert_eq!(cache.copies.load(Ordering::Relaxed), 0);
    assert_eq!(copy(&mapping, 8, 17, 0, 8), first);
    assert_eq!(cache.misses.load(Ordering::Relaxed), 1);
    assert_eq!(cache.hits.load(Ordering::Relaxed), 0);
}

#[test]
fn measurement_page_table_release_keeps_resident_rows_and_counters() {
    let registry = registry(1);
    let mapping = mapping("measurement-page-tables", &source(1000, 264), 24);
    attach(&registry, vec![(Arc::clone(&mapping), 264)]).unwrap();
    let cache = &mapping.resident.get().unwrap().cache;
    let first = copy(&mapping, 264, 17, 0, 264);
    let before = cache.measurement_snapshot();
    super::super::release_page_tables(&[Arc::clone(&mapping)]).unwrap();
    assert_eq!(cache.measurement_snapshot(), before);
    assert_eq!(copy(&mapping, 264, 17, 0, 264), first);
    assert_eq!(cache.hits.load(Ordering::Relaxed), 1);
    assert_eq!(cache.misses.load(Ordering::Relaxed), 1);
}

#[test]
fn cache_is_on_by_default_and_disabled_budgets_are_not_parsed() {
    let default = Config::parse(|_| None).unwrap().unwrap();
    assert_eq!((default.set_bytes, default.total_bytes), (1 << 30, 2 << 30));
    assert!(
        Config::parse(|name| match name {
            "CHOPINFER_MAPPED_HOT_ROWS" => Some("0".into()),
            "CHOPINFER_MAPPED_HOT_ROWS_BYTES" => Some("invalid".into()),
            _ => None,
        })
        .unwrap()
        .is_none()
    );
    let config = Config::parse(|name| (name == "CHOPINFER_MAPPED_HOT_ROWS").then(|| "1".into()))
        .unwrap()
        .unwrap();
    assert_eq!((config.set_bytes, config.total_bytes), (1 << 30, 2 << 30));
    assert!(
        Config::parse(|name| Some(
            if name == "CHOPINFER_MAPPED_HOT_ROWS" {
                "1"
            } else {
                "invalid"
            }
            .into()
        ))
        .is_err()
    );
}

#[test]
fn allocation_failure_returns_the_entire_reservation() {
    let registry = registry(1);
    let mapping = mapping("allocation", &source(10, 8), 10);
    let charge = registry.budget.reserve(registry.config.set_bytes).unwrap();
    let mut calls = 0;
    let result = Cache::allocate(vec![(mapping.key, 8)], charge, |bytes| {
        calls += 1;
        if calls == 3 {
            Err(std::io::Error::other("injected lock failure"))
        } else {
            LockedBytes::new(bytes)
        }
    });
    assert!(result.is_err());
    assert_eq!(calls, 3);
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
    assert!(mapping.resident.get().is_none());
}

#[test]
fn failed_lock_serves_exact_uncached_rows_without_retrying_allocation() {
    let registry = registry(1);
    let mapping = mapping("lock-fallback", &source(10, 8), 21);
    attach_with(&registry, vec![(Arc::clone(&mapping), 8)], |key, charge| {
        Cache::allocate(key, charge, |_| {
            Err(std::io::Error::from_raw_os_error(libc::ENOMEM))
        })
    })
    .unwrap();
    assert!(registry.fallback_logged.load(Ordering::Relaxed));
    assert!(registry.allocation_unavailable.load(Ordering::Relaxed));
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
    assert!(mapping.resident.get().is_none());
    attach_with(&registry, vec![(Arc::clone(&mapping), 8)], |_, _| {
        panic!("unavailable allocation must not be retried")
    })
    .unwrap();
    let plane = gather_plane(Arc::clone(&mapping), 8, 0);
    let mut output = vec![0xa5; 24];
    let ids = [2_u32, 8, 2]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    crate::device::mapped_gather::gather_plane_v1(&plane, &ids, &mut output).unwrap();
    let expected = [2, 8, 2]
        .into_iter()
        .flat_map(|row| mapping.bytes()[row * 8..][..8].to_vec())
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn unauthenticated_mapping_cannot_enter_cache() {
    let registry = registry(1);
    let path = std::env::temp_dir().join(format!("resident-unauth-{}", std::process::id()));
    std::fs::write(&path, [0; 100]).unwrap();
    let file = File::open(&path).unwrap();
    let mapping = super::super::map_rows_v1(&file, 0, 100).unwrap();
    std::fs::remove_file(path).unwrap();
    attach(&registry, vec![(Arc::clone(&mapping), 10)]).unwrap();
    assert!(mapping.resident.get().is_none());
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
}

fn gather_plane(mapping: Arc<MappedRowsV1>, width: usize, offset: usize) -> MappedGatherPlaneV1 {
    use crate::device::{DeviceObjectV1, DeviceRangeV1};
    MappedGatherPlaneV1 {
        mapping,
        row_bytes: width,
        index_offset: offset,
        stream: DeviceObjectV1::new(1),
        destination: DeviceRangeV1 {
            object: DeviceObjectV1::new(2),
            offset: 0,
            length: (width * 3) as u64,
        },
    }
}

#[test]
fn served_async_gathers_fill_reuse_and_refuse_bad_ids_before_writes() {
    let registry = registry(1);
    let codes = mapping("async-code", &source(10, 256), 11);
    let scales = mapping("async-scale", &source(10, 8), 11);
    let planes = vec![
        gather_plane(Arc::clone(&codes), 256, 0),
        gather_plane(Arc::clone(&scales), 8, 0),
    ];
    register_checked(
        &registry,
        &planes,
        HOST_RESERVE_BYTES + registry.config.total_bytes as u64,
    )
    .unwrap();
    let input = [1_u32, 3, 1]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    for _ in 0..2 {
        for plane in &planes {
            let mut out = vec![0xa5; plane.row_bytes * 3];
            crate::device::mapped_gather::gather_plane_v1(plane, &input, &mut out).unwrap();
            assert_eq!(
                out,
                [1, 3, 1]
                    .into_iter()
                    .flat_map(|row| plane.mapping.bytes()[row * plane.row_bytes..]
                        [..plane.row_bytes]
                        .to_vec())
                    .collect::<Vec<_>>()
            );
        }
    }
    let cache = &codes.resident.get().unwrap().cache;
    assert_eq!(cache.hits.load(Ordering::Relaxed), 8);
    assert_eq!(cache.misses.load(Ordering::Relaxed), 4);
    let before = cache.copies.load(Ordering::Relaxed);
    let input = [1_u32, 3, 10]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    let mut out = vec![0xa5; 768];
    assert!(crate::device::mapped_gather::gather_plane_v1(&planes[0], &input, &mut out).is_err());
    assert_eq!(out, vec![0xa5; 768]);
    assert_eq!(cache.copies.load(Ordering::Relaxed), before);
}

#[test]
fn synchronous_chunked_prefill_fills_the_same_cache_as_verify() {
    use super::super::gather::{GatherProbe, ReadAhead, gather_span_v1};
    struct Clock;
    impl GatherProbe for Clock {
        fn now(&self) -> std::time::Duration {
            std::time::Duration::ZERO
        }
        fn read_ahead(&self, _: &[u32]) {
            panic!("read-ahead disabled");
        }
    }
    let registry = registry(1);
    let mapping = mapping("prefill", &source(100, 97), 12);
    attach(&registry, vec![(Arc::clone(&mapping), 97)]).unwrap();
    let rows = [5_u32, 9, 11, 12, 13, 5, 9];
    let mut out = vec![0xa5; rows.len() * 97];
    let ahead = ReadAhead::disabled(&Clock).with_resident(mapping.resident_plane(97));
    for (i, chunk) in out.chunks_mut(200).enumerate() {
        gather_span_v1(mapping.bytes(), 97, &rows, i * 200, chunk, &ahead);
    }
    let expected = rows
        .iter()
        .flat_map(|&r| mapping.bytes()[r as usize * 97..][..97].to_vec())
        .collect::<Vec<_>>();
    assert_eq!(out, expected);
    let cache = &mapping.resident.get().unwrap().cache;
    let before = cache.hits.load(Ordering::Relaxed);
    copy(&mapping, 97, 5, 0, 97);
    assert!(cache.hits.load(Ordering::Relaxed) > before);
}

#[test]
fn host_reserve_falls_back_before_allocating() {
    let registry = registry(1);
    let mapping = mapping("reserve", &source(10, 8), 13);
    let planes = [gather_plane(Arc::clone(&mapping), 8, 0)];
    register_checked(&registry, &planes, HOST_RESERVE_BYTES).unwrap();
    assert!(registry.fallback_logged.load(Ordering::Relaxed));
    assert!(mapping.resident.get().is_none());
    assert_eq!(registry.budget.used.load(Ordering::Relaxed), 0);
}
