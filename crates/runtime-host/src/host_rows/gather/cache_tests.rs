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
        for count in [24, 192, 768, 6144, 65535, 65536, 196608] {
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
                if count < 65536 || cache_large {
                    count as u64
                } else {
                    0
                },
                "rows={count}"
            );
            if count >= 65536 && !cache_large {
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
