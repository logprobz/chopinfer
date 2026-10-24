use super::*;
use kernels_sm120::f32_projection::PrefillBand;

fn seed() -> Entry {
    let original = entry(WeightFormat::Bf16);
    Entry::new(
        GemmQuery {
            m_capacity: 8192,
            n: 64,
            k: 512,
            ..original.query()
        },
        OperationUse {
            capacity: 8192,
            ..original.usage()
        },
        "band_route",
    )
    .unwrap()
    .with_split_decode()
}
fn band(lo: u32, hi: u32, features: u32) -> PrefillBand {
    PrefillBand { lo, hi, features }
}

fn launches(e: &Entry, rows: u32) -> Vec<kernel_core::backend::Launch> {
    let op = backend(vec![e.clone()])
        .projection(&operation(e), e.usage(), Need::rows(8192))
        .unwrap()
        .handle;
    let mut b = buffers(e);
    b.scratch.weight_domain = Some(View {
        handle: 10,
        bytes: e.weight_domain_bytes(),
    });
    op.launches(
        &b,
        View {
            handle: 9,
            bytes: op.info().workspace,
        },
        Counts::rows(rows),
    )
    .unwrap()
}

#[test]
fn inclusive_bands_preserve_exact_wide_blocked_and_tiled_entries() {
    let exact = seed()
        .with_prefill_buckets(vec![(512, 32), (1024, 64), (2048, 32)])
        .unwrap()
        .with_blocked_prefill_buckets(vec![(512, 32), (1024, 32)])
        .unwrap()
        .with_wide_prefill_buckets(vec![(1024, 64)])
        .unwrap();
    let routed = exact
        .clone()
        .with_prefill_bands(vec![band(257, 4096, 64), band(4097, 8192, 32)])
        .unwrap();
    for rows in [512, 1024, 2048] {
        assert_eq!(routed.entry_for_rows(rows), exact.entry_for_rows(rows));
    }
    assert_eq!(routed.entry_for_rows(512).blocked_prefill_rows(), 32);
    assert_eq!(routed.entry_for_rows(1024).wide_prefill_features(), 64);
    for rows in [257, 319, 511, 513, 1023, 1025, 4095, 4096] {
        assert_eq!(routed.entry_for_rows(rows).prefill_features(), 64);
    }
    for rows in [4097, 7754, 8191, 8192] {
        assert_eq!(routed.entry_for_rows(rows).prefill_features(), 32);
    }
    for rows in [1, 32, 33, 128, 256] {
        assert_eq!(launches(&routed, rows), launches(&exact, rows));
    }
    let source = |e: &Entry| {
        kernel_build::generate(
            &f32_projection::cubin_spec("band_source", std::slice::from_ref(e)).unwrap(),
        )
        .unwrap()
        .translation_unit
    };
    assert_eq!(source(&exact), source(&routed));
}

#[test]
fn bands_are_canonical_disjoint_bounded_and_above_decode() {
    let valid = vec![band(257, 319, 32), band(320, 640, 64), band(1024, 8192, 32)];
    let mut reversed = valid.clone();
    reversed.reverse();
    let a = seed().with_prefill_bands(valid).unwrap();
    assert_eq!(a, seed().with_prefill_bands(reversed).unwrap());
    for rows in [641, 789, 1023] {
        assert_eq!(launches(&a, rows), launches(&seed(), rows));
    }
    for invalid in [
        vec![band(256, 320, 32)],
        vec![band(300, 257, 32)],
        vec![band(257, 8193, 32)],
        vec![band(257, 320, 16)],
        vec![band(257, 320, 32), band(320, 640, 64)],
        vec![band(257, 320, 32), band(257, 320, 32)],
    ] {
        assert!(seed().with_prefill_bands(invalid).is_err());
    }
    assert!(
        entry(WeightFormat::Bf16)
            .with_prefill_bands(vec![])
            .is_err()
    );
    assert!(
        a.with_prefill_bands(vec![])
            .unwrap()
            .prefill_bands()
            .is_empty()
    );
}

#[test]
fn band_workspace_covers_capacities_inside_a_band_without_reserving_later_rows() {
    let plain = seed();
    let e = plain
        .clone()
        .with_prefill_bands(vec![band(257, 640, 32), band(1024, 8192, 64)])
        .unwrap();
    for capacity in [256, 257, 319, 500, 640, 789, 1023, 1024, 1803, 7754, 8192] {
        let last_prefill = match capacity {
            0..=256 => 0,
            257..=640 => capacity,
            641..=1023 => 640,
            _ => capacity,
        };
        let bytes = u64::from(last_prefill).div_ceil(4) * 16 + u64::from(last_prefill) * 512 * 6;
        let op = backend(vec![e.clone()])
            .projection(&operation(&e), e.usage(), Need::rows(capacity))
            .unwrap()
            .handle;
        assert_eq!(
            op.info().workspace,
            bytes.max(plain.decode_workspace_bytes(capacity)),
            "capacity {capacity}"
        );
        let mut b = buffers(&e);
        b.scratch.weight_domain = Some(View {
            handle: 10,
            bytes: e.weight_domain_bytes(),
        });
        let work = View {
            handle: 9,
            bytes: op.info().workspace,
        };
        let launches = op.launches(&b, work, Counts::rows(capacity)).unwrap();
        if (257..=640).contains(&capacity) || capacity >= 1024 {
            assert_eq!(launches.len(), 2);
            assert!(launches[1].symbol.contains("_tiles_n"));
            assert!(
                op.launches(
                    &b,
                    View {
                        bytes: e.prefill_workspace_bytes(capacity) - 1,
                        ..work
                    },
                    Counts::rows(capacity)
                )
                .is_err()
            );
        }
    }
}

#[test]
fn band_only_exports_and_shared_memory_are_required() {
    let e = seed()
        .with_prefill_bands(vec![band(257, 640, 32), band(641, 8192, 64)])
        .unwrap();
    let op = backend(vec![e.clone()])
        .projection(&operation(&e), e.usage(), Need::rows(8192))
        .unwrap()
        .handle;
    let tiles = [e.entry_for_rows(320), e.entry_for_rows(789)];
    for (tile, shared) in tiles.iter().zip([65536, 81920]) {
        assert!(op.info().symbols.contains(&(tile.symbol().into(), shared)));
        let available = e
            .runtime_symbols()
            .into_iter()
            .filter(|s| s != tile.symbol())
            .collect();
        let missing = Sm120::new().with_f32_projections(
            F32ProjectionOffers::new(vec![e.clone()], available, "0".repeat(64)).unwrap(),
        );
        assert!(
            missing
                .projection(&operation(&e), e.usage(), Need::rows(8192))
                .is_err()
        );
    }
}

#[test]
fn canonical_band_duplicates_collapse_but_different_routes_refuse() {
    let a = seed()
        .with_prefill_bands(vec![band(257, 640, 32), band(641, 8192, 64)])
        .unwrap();
    let b = seed()
        .with_prefill_bands(vec![band(641, 8192, 64), band(257, 640, 32)])
        .unwrap();
    assert!(
        backend(vec![a.clone(), b])
            .projection(&operation(&a), a.usage(), Need::rows(8192))
            .is_ok()
    );
    let other = seed()
        .with_prefill_bands(vec![band(257, 640, 64), band(641, 8192, 64)])
        .unwrap();
    assert!(
        backend(vec![a.clone(), other])
            .projection(&operation(&a), a.usage(), Need::rows(8192))
            .is_err()
    );
}
