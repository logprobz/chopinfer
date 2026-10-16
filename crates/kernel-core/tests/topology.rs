use kernel_core::topology::{MAX_RANKS, RankSplit, TensorParallel};

const RANKS: [usize; 5] = [1, 2, 4, 8, 16];

fn check_partition(split: RankSplit, tile: usize) {
    let mut seen = vec![0; split.total()];
    let mut source_end = 0;
    let mut storage_end = 0;
    for rank in 0..split.ranks() {
        let shard = split.rank(rank).unwrap();
        assert_eq!(shard.live.start, source_end);
        assert_eq!(shard.padded.start, storage_end);
        assert!(shard.live.end <= split.total());
        assert!(shard.live_len() <= shard.padded_len());
        assert_eq!(shard.padded_len(), split.padded_len());
        assert_eq!(shard.padded_len() % tile, 0);
        assert_eq!(split.duplication_group(rank).unwrap(), rank..rank + 1);
        for index in shard.live.clone() {
            seen[index] += 1;
            assert_eq!(split.owner(index), Some(rank));
            assert_eq!(split.owners(index), Some(rank..rank + 1));
        }
        source_end = shard.live.end;
        storage_end = shard.padded.end;
    }
    assert_eq!(source_end, split.total());
    assert_eq!(storage_end, split.ranks() * split.padded_len());
    assert!(seen.iter().all(|count| *count == 1));
    assert_eq!(split.owner(split.total()), None);
    assert_eq!(split.owners(usize::MAX), None);
    assert!(split.rank(split.ranks()).is_err());
    assert!(split.duplication_group(usize::MAX).is_err());
}

#[test]
fn padded_partitions_cover_once_for_all_small_dimensions_and_tiles() {
    for ranks in RANKS {
        let tp = TensorParallel::new(ranks).unwrap();
        for total in 1..=256 {
            for tile in (1..=17).chain([64, 128, 256]) {
                check_partition(tp.padded_split(total, tile).unwrap(), tile);
            }
        }
    }
}

#[test]
fn equal_splits_refuse_remainders_and_match_unpadded_partitions() {
    for ranks in RANKS {
        let tp = TensorParallel::new(ranks).unwrap();
        assert_eq!(tp.ranks(), ranks);
        for total in 1..=256 {
            if total % ranks == 0 {
                let split = tp.split(total).unwrap();
                assert_eq!(split, tp.padded_split(total, 1).unwrap());
                check_partition(split, 1);
                for rank in 0..ranks {
                    let shard = split.rank(rank).unwrap();
                    assert_eq!(shard.live, shard.padded);
                    assert_eq!(shard.live_len(), total / ranks);
                }
            } else {
                assert!(tp.split(total).is_err());
            }
        }
    }
}

#[test]
fn kv_coverage_overlap_and_owners_agree_for_all_small_dimensions() {
    for ranks in RANKS {
        let tp = TensorParallel::new(ranks).unwrap();
        for total in 1..=256 {
            let split = tp.kv_heads(total).unwrap();
            if total >= ranks {
                check_partition(split, 1);
                continue;
            }
            for head in 0..total {
                let holders: Vec<_> = (0..ranks)
                    .filter(|rank| split.rank(*rank).unwrap().live.contains(&head))
                    .collect();
                let group = split.owners(head).unwrap();
                assert!(!holders.is_empty());
                assert_eq!(holders, group.clone().collect::<Vec<_>>());
                assert_eq!(split.owner(head), holders.first().copied());
                for rank in group.clone() {
                    assert_eq!(split.duplication_group(rank).unwrap(), group);
                }
            }
            for rank in 0..ranks {
                let shard = split.rank(rank).unwrap();
                assert_eq!(shard.live, rank * total / ranks..rank * total / ranks + 1);
                assert_eq!(shard.padded, rank..rank + 1);
            }
            assert_eq!(split.owners(total), None);
            assert_eq!(split.owner(usize::MAX), None);
            assert!(split.rank(ranks).is_err());
            assert!(split.duplication_group(ranks).is_err());
        }
    }
}

#[test]
fn uneven_splits_keep_tile_padding_out_of_source_ranges() {
    let tp = TensorParallel::new(4).unwrap();
    let split = tp.padded_split(10, 4).unwrap();
    for (rank, live) in [0..3, 3..6, 6..9, 9..10].into_iter().enumerate() {
        let shard = split.rank(rank).unwrap();
        assert_eq!(shard.live, live);
        assert_eq!(shard.padded, rank * 4..(rank + 1) * 4);
    }
    let split = tp.padded_split(2, 128).unwrap();
    for (rank, live) in [0..1, 1..2, 2..2, 2..2].into_iter().enumerate() {
        let shard = split.rank(rank).unwrap();
        assert_eq!(shard.live, live);
        assert_eq!(shard.padded_len(), 128);
    }
    assert_eq!(split.owner(2), None);
}

#[test]
fn kv_duplication_groups_include_pairs_quartets_and_unequal_groups() {
    for (ranks, heads, expected) in [
        (4, 2, vec![0..2, 2..4]),
        (8, 2, vec![0..4, 4..8]),
        (16, 2, vec![0..8, 8..16]),
        (8, 4, vec![0..2, 2..4, 4..6, 6..8]),
        (8, 3, vec![0..3, 3..6, 6..8]),
    ] {
        let split = TensorParallel::new(ranks).unwrap().kv_heads(heads).unwrap();
        for (head, group) in expected.into_iter().enumerate() {
            assert_eq!(split.owners(head), Some(group.clone()));
            for rank in group.clone() {
                assert_eq!(split.rank(rank).unwrap().live, head..head + 1);
                assert_eq!(split.duplication_group(rank).unwrap(), group);
            }
        }
    }
}

// Frozen TP4 placements from models/mimo26/compiler/src/placement.rs:181-254 and
// config.rs:90-102. MiMo shards each expert's intermediate, not expert IDs.
// Keep these literals independent of the new split formulas and avoid model
// dependencies in kernel-core. The fused QKV chunks are source rows only;
// regrouping those chunks for other topologies belongs to the model package.
#[test]
fn mimo_tp4_matches_current_source_placements() {
    let tp = TensorParallel::new(4).unwrap();
    for (total, boundaries) in [
        (64, [0, 16, 32, 48, 64]), // Q heads and attention sinks
        (4, [0, 1, 2, 3, 4]),      // GA KV heads
        (8, [0, 2, 4, 6, 8]),      // SWA KV heads
        (152_576, [0, 38_144, 76_288, 114_432, 152_576]),
        (16_384, [0, 4096, 8192, 12_288, 16_384]), // Dense intermediate
        (2048, [0, 512, 1024, 1536, 2048]),        // Expert intermediate
        (13_568, [0, 3392, 6784, 10_176, 13_568]), // GA QKV rows
        (14_848, [0, 3712, 7424, 11_136, 14_848]), // SWA QKV rows
        (116, [0, 29, 58, 87, 116]),               // SWA QKV scale rows
        (8192, [0, 2048, 4096, 6144, 8192]),       // O/MTP projection columns
        (1024, [0, 256, 512, 768, 1024]),          // Packed expert down K bytes
    ] {
        let split = tp.split(total).unwrap();
        for rank in 0..4 {
            let shard = split.rank(rank).unwrap();
            assert_eq!(shard.live, boundaries[rank]..boundaries[rank + 1]);
            assert_eq!(shard.padded, shard.live);
        }
    }
    for heads in [4, 8] {
        assert_eq!(tp.kv_heads(heads).unwrap(), tp.split(heads).unwrap());
    }
}

// Frozen from models/qwen38/compiler/src/catalog.rs:248-270, :321-362 and
// placement.rs:21-24, :209-212: axis splits, duplicated KV, linear QKV,
// expert ownership and shared-expert padding.
#[test]
fn qwen_tp4_matches_current_source_and_padded_placements() {
    let tp = TensorParallel::new(4).unwrap();
    for (total, boundaries) in [
        (24, [0, 6, 12, 18, 24]),
        (16, [0, 4, 8, 12, 16]),   // Linear key heads
        (48, [0, 12, 24, 36, 48]), // Linear value heads
        (512, [0, 128, 256, 384, 512]),
        (248_320, [0, 62_080, 124_160, 186_240, 248_320]),
        (12_288, [0, 3072, 6144, 9216, 12_288]), // Gated Q rows
        (6144, [0, 1536, 3072, 4608, 6144]),     // O projection columns
    ] {
        let split = tp.split(total).unwrap();
        for rank in 0..4 {
            let shard = split.rank(rank).unwrap();
            assert_eq!(shard.live, boundaries[rank]..boundaries[rank + 1]);
            assert_eq!(shard.padded, shard.live);
        }
    }
    let kv = tp.kv_heads(2).unwrap();
    for (rank, head) in [0, 0, 1, 1].into_iter().enumerate() {
        let shard = kv.rank(rank).unwrap();
        assert_eq!(shard.live, head..head + 1);
        // K and V each have 256 rows per head.
        assert_eq!(
            shard.live.start * 256..shard.live.end * 256,
            head * 256..(head + 1) * 256
        );
    }
    let experts = tp.split(512).unwrap();
    for expert in 0..512 {
        let owner = experts.owner(expert).unwrap();
        assert_eq!(owner, expert / 128);
        assert_eq!(
            expert - experts.rank(owner).unwrap().live.start,
            expert % 128
        );
    }
    let shared = tp.padded_split(640, 128).unwrap();
    for (rank, live) in [0..160, 160..320, 320..480, 480..640]
        .into_iter()
        .enumerate()
    {
        let shard = shared.rank(rank).unwrap();
        assert_eq!(shard.live, live);
        assert_eq!(shard.live_len(), 160);
        assert_eq!(shard.padded_len(), 256);
    }
    let expected = [
        [0..512, 2048..2560, 4096..5632],
        [512..1024, 2560..3072, 5632..7168],
        [1024..1536, 3072..3584, 7168..8704],
        [1536..2048, 3584..4096, 8704..10_240],
    ];
    for (rank, ranges) in expected.into_iter().enumerate() {
        for ((offset, total), expected) in [(0, 2048), (2048, 2048), (4096, 6144)]
            .into_iter()
            .zip(ranges)
        {
            let live = tp.split(total).unwrap().rank(rank).unwrap().live;
            assert_eq!(offset + live.start..offset + live.end, expected);
        }
    }
}

#[test]
fn model_dimensions_cover_all_rank_counts_including_tp16() {
    // Columns are TP1, TP2, TP4, TP8, TP16. Qwen Q heads need padding at TP16.
    for (total, lengths) in [
        (64, [64, 32, 16, 8, 4]),
        (16_384, [16_384, 8192, 4096, 2048, 1024]),
        (2048, [2048, 1024, 512, 256, 128]),
        (152_576, [152_576, 76_288, 38_144, 19_072, 9536]),
        (24, [24, 12, 6, 3, 2]),
        (16, [16, 8, 4, 2, 1]),
        (48, [48, 24, 12, 6, 3]),
        (512, [512, 256, 128, 64, 32]),
        (248_320, [248_320, 124_160, 62_080, 31_040, 15_520]),
    ] {
        for (ranks, length) in RANKS.into_iter().zip(lengths) {
            let split = TensorParallel::new(ranks)
                .unwrap()
                .padded_split(total, 1)
                .unwrap();
            assert_eq!(split.padded_len(), length);
            check_partition(split, 1);
        }
    }
    for (heads, local, copies) in [
        (4, [4, 2, 1, 1, 1], [1, 1, 1, 2, 4]),
        (8, [8, 4, 2, 1, 1], [1, 1, 1, 1, 2]),
        (2, [2, 1, 1, 1, 1], [1, 1, 2, 4, 8]),
    ] {
        for ((ranks, local), copies) in RANKS.into_iter().zip(local).zip(copies) {
            let split = TensorParallel::new(ranks).unwrap().kv_heads(heads).unwrap();
            for rank in 0..ranks {
                assert_eq!(split.rank(rank).unwrap().live_len(), local);
                assert_eq!(split.duplication_group(rank).unwrap().len(), copies);
            }
        }
    }
    for ((ranks, live), padded) in RANKS
        .into_iter()
        .zip([640, 320, 160, 80, 40])
        .zip([640, 384, 256, 128, 128])
    {
        let split = TensorParallel::new(ranks)
            .unwrap()
            .padded_split(640, 128)
            .unwrap();
        check_partition(split, 128);
        for rank in 0..ranks {
            let shard = split.rank(rank).unwrap();
            assert_eq!(shard.live_len(), live);
            assert_eq!(shard.padded_len(), padded);
        }
    }
}

#[test]
fn invalid_topologies_dimensions_tiles_and_ranks_are_refused() {
    assert_eq!(MAX_RANKS, 16);
    for ranks in [0, 3, 5, 6, 7, 9, 15, 17, 32, usize::MAX] {
        assert!(TensorParallel::new(ranks).is_err());
    }
    for ranks in RANKS {
        let tp = TensorParallel::new(ranks).unwrap();
        assert!(tp.split(0).is_err());
        assert!(tp.padded_split(0, 1).is_err());
        assert!(tp.padded_split(1, 0).is_err());
        assert!(tp.kv_heads(0).is_err());
        let split = tp.split(ranks).unwrap();
        assert!(split.rank(ranks).is_err());
        assert!(split.rank(usize::MAX).is_err());
    }
}

#[test]
fn endpoint_overflow_is_refused_without_rejecting_representable_ranges() {
    let tp1 = TensorParallel::new(1).unwrap();
    let split = tp1.split(usize::MAX).unwrap();
    assert_eq!(split.rank(0).unwrap().live, 0..usize::MAX);
    assert_eq!(split.owner(usize::MAX - 1), Some(0));
    assert_eq!(split.owner(usize::MAX), None);
    assert!(tp1.padded_split(usize::MAX, 2).is_err());
    for ranks in [2, 4, 8, 16] {
        let tp = TensorParallel::new(ranks).unwrap();
        assert!(tp.padded_split(usize::MAX, 1).is_err());
        assert!(tp.kv_heads(usize::MAX).is_err());
        assert!(tp.padded_split(ranks, usize::MAX).is_err());
        let total = usize::MAX - (ranks - 1);
        let split = tp.split(total).unwrap();
        assert_eq!(split.rank(ranks - 1).unwrap().live.end, total);
        assert_eq!(split.owner(total - 1), Some(ranks - 1));
        assert_eq!(split.owners(total), None);
    }
}
