//! Large gathers use mapped bytes while small gathers retain resident hits.
use super::*;
use crate::host_rows::gather::{GatherProbe, ReadAhead, gather_span_v1};

struct Clock;
impl GatherProbe for Clock {
    fn now(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
    fn read_ahead(&self, _: &[u32]) {
        panic!("read-ahead disabled");
    }
}

#[test]
fn large_gathers_bypass_resident_metadata_but_preserve_every_byte() {
    let cache_large =
        std::env::var_os("CHOPINFER_MAPPED_LARGE_GATHER_CACHE").is_some_and(|value| value == "1");
    let registry = registry(1);
    let codes = mapping("large-gather-codes", &source(100, 256), 31);
    let scales = mapping("large-gather-scales", &source(100, 8), 31);
    attach(
        &registry,
        vec![(Arc::clone(&codes), 256), (Arc::clone(&scales), 8)],
    )
    .unwrap();
    for (mapping, width) in [(&codes, 256), (&scales, 8)] {
        let cache = &mapping.resident.get().unwrap().cache;
        for count in [24, 192, 768, 6144, 8191, 8192, 8193, 65535, 65536, 196608] {
            let rows = (0..count)
                .map(|i| ((i * 37) % 100) as u32)
                .collect::<Vec<_>>();
            let expected = rows
                .iter()
                .flat_map(|&row| {
                    mapping.bytes()[row as usize * width..][..width]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let ahead = ReadAhead::disabled(&Clock).with_resident(mapping.resident_plane(width));
            let before = cache.copies.load(Ordering::Relaxed);
            let mut out = vec![0xa5; expected.len()];
            gather_span_v1(mapping.bytes(), width, &rows, 0, &mut out, &ahead);
            assert_eq!(out, expected);
            let copies = cache.copies.load(Ordering::Relaxed) - before;
            assert_eq!(
                copies,
                if count < 8192 || cache_large {
                    count as u64
                } else {
                    0
                },
                "rows={count}"
            );
            if count >= 8192 && !cache_large {
                // Upload-slot and worker pieces must use the whole gather's
                // threshold, including pieces that split a row.
                out.fill(0x5a);
                for (i, chunk) in out.chunks_mut(10007).enumerate() {
                    gather_span_v1(mapping.bytes(), width, &rows, i * 10007, chunk, &ahead);
                }
                assert_eq!(out, expected);
                assert_eq!(cache.copies.load(Ordering::Relaxed), before);
            }
        }
        let hits = cache.hits.load(Ordering::Relaxed);
        copy(mapping, width, 5, 0, width);
        copy(mapping, width, 5, 0, width);
        assert!(
            cache.hits.load(Ordering::Relaxed) > hits,
            "small gathers still cache"
        );
    }
}

#[test]
fn asynchronous_gathers_preserve_large_row_cache_admission() {
    let cache_large =
        std::env::var_os("CHOPINFER_MAPPED_LARGE_GATHER_CACHE").is_some_and(|value| value == "1");
    let registry = registry(1);
    for width in [8, 256] {
        let mapping = mapping(
            &format!("async-large-gather-{width}"),
            &source(100, width),
            32,
        );
        attach(&registry, vec![(Arc::clone(&mapping), width)]).unwrap();
        let cache = &mapping.resident.get().unwrap().cache;
        let plane = gather_plane(Arc::clone(&mapping), width, 0);
        for count in [24, 8191, 8192, 8193, 65_535, 65_536, 196_608, 192] {
            let rows: Vec<u32> = (0..count).map(|i| ((i * 37) % 100) as u32).collect();
            let input: Vec<u8> = rows.iter().flat_map(|row| row.to_le_bytes()).collect();
            let expected: Vec<u8> = rows
                .iter()
                .flat_map(|&row| {
                    mapping.bytes()[row as usize * width..][..width]
                        .iter()
                        .copied()
                })
                .collect();
            let mut output = vec![0xa5; expected.len()];
            let before = cache.copies.load(Ordering::Relaxed);
            crate::device::mapped_gather::gather_plane_v1(&plane, &input, &mut output).unwrap();
            assert_eq!(output, expected);
            let copies = cache.copies.load(Ordering::Relaxed) - before;
            if count >= 8192 && !cache_large {
                assert_eq!(copies, 0, "large asynchronous gather used resident lookup");
            } else {
                assert!(
                    copies > 0,
                    "small or explicitly cached gather bypassed lookup"
                );
            }
        }
    }
}

#[test]
fn prefill_sized_gathers_match_resident_and_descriptor_reads() {
    let registry = registry(1);
    for width in [8, 97, 256] {
        let mapping = mapping(
            &format!("prefill-route-pair-{width}"),
            &source(257, width),
            39,
        );
        attach(&registry, vec![(Arc::clone(&mapping), width)]).unwrap();
        let plane = mapping.resident_plane(width).unwrap();
        let cache = &mapping.resident.get().unwrap().cache;
        let mut fanout = crate::host_rows::fetch::LargeFanout::default();
        for count in [8192, 8193, 43600] {
            let rows: Vec<_> = (0..count).map(|i| ((i * 37 + 11) % 257) as u32).collect();
            let mut resident = vec![0xa5; count * width];
            for (&row, bytes) in rows.iter().zip(resident.chunks_mut(width)) {
                plane.copy(mapping.bytes(), row, 0, bytes);
            }
            let before = cache.copies.load(Ordering::Relaxed);
            let large = fanout.rows(&mapping, width, &rows, 3).unwrap();
            assert_eq!(*large, resident);
            for _ in 0..3 {
                let later = fanout.rows(&mapping, width, &rows, 3).unwrap();
                assert!(Arc::ptr_eq(&large, &later));
                assert_eq!(*later, resident);
            }
            assert_eq!(cache.copies.load(Ordering::Relaxed), before);
        }
    }
}
