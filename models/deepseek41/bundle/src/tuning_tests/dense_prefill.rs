use kernel_build::tuning::Key;
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn key(op: &str, weights: &str, shape: &[(&str, u32)], bucket: u32) -> Key {
    Key {
        op: op.into(),
        weights: weights.into(),
        shape: shape
            .iter()
            .map(|&(name, value)| (name.into(), value))
            .collect(),
        bucket,
        rows: None,
    }
}

fn ranges(values: impl IntoIterator<Item = u32>) -> String {
    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for value in values.into_iter().collect::<BTreeSet<_>>() {
        if let Some((_, hi)) = ranges.last_mut()
            && *hi + 1 == value
        {
            *hi = value;
        } else {
            ranges.push((value, value));
        }
    }
    ranges
        .iter()
        .map(|&(lo, hi)| {
            if lo == hi {
                lo.to_string()
            } else {
                format!("{lo}..{hi}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn assert_untuned(
    family: Family,
    domain: BTreeSet<Key>,
    expected: &[&str],
    schedules: &[(&str, &[&str])],
) {
    assert!(!domain.is_empty());
    let mut covered: BTreeMap<Key, Vec<[u32; 2]>> = BTreeMap::new();
    for row in table(&FLASH, family).unwrap().rows() {
        let mut key = row.key.clone();
        key.rows = None;
        assert!(domain.contains(&key), "unserved tuning key: {key}");
        let knobs = schedules
            .iter()
            .find(|(name, _)| *name == row.schedule)
            .unwrap_or_else(|| panic!("unknown schedule: {}", row.schedule))
            .1;
        assert_eq!(
            row.knobs.keys().map(String::as_str).collect::<Vec<_>>(),
            knobs
        );
        covered
            .entry(key)
            .or_default()
            .push(row.key.rows.unwrap_or([1, row.key.bucket]));
    }
    let mut missing: BTreeMap<(String, String), Vec<u32>> = BTreeMap::new();
    for key in domain {
        let mut next = 1;
        let mut gaps = Vec::new();
        for &[lo, hi] in covered.get(&key).into_iter().flatten() {
            if next < lo {
                gaps.push([next, lo - 1]);
            }
            next = hi + 1;
        }
        if next <= key.bucket {
            gaps.push([next, key.bucket]);
        }
        if gaps.is_empty() {
            continue;
        }
        let gap = if gaps == [[1, key.bucket]] {
            "all".into()
        } else {
            gaps.iter()
                .map(|&[lo, hi]| {
                    if lo == hi {
                        lo.to_string()
                    } else {
                        format!("{lo}..{hi}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let shape = key
            .shape
            .iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(",");
        missing
            .entry((format!("{} {} {shape}", key.op, key.weights), gap))
            .or_default()
            .push(key.bucket);
    }
    let mut actual: Vec<_> = missing
        .into_iter()
        .map(|((identity, gaps), buckets)| {
            format!("{identity} buckets={} rows={gaps}", ranges(buckets))
        })
        .collect();
    actual.sort();
    assert_eq!(
        actual,
        expected.iter().map(|s| s.to_string()).collect::<Vec<_>>()
    );
}

#[test]
fn pin_untuned_served_keys() {
    let mut domain = BTreeSet::new();
    for ranks in [1, 2, 4] {
        for entry in crate::sm120::mxfp8_fold::prefill_entries(&FLASH, ranks).unwrap() {
            let q = entry.query;
            if (q.n, q.k, q.groups) == (5120, 15360, 1) {
                domain.insert(key(
                    "dense_prefill.dspark_main.prefill",
                    "mxfp8",
                    &[
                        ("n", q.n),
                        ("k", q.k),
                        ("groups", q.groups),
                        ("ranks", ranks),
                    ],
                    q.m_capacity,
                ));
            }
            for site in ["compact", "matrix", "tma_scales"] {
                let mut shape = vec![("n", q.n), ("k", q.k), ("groups", q.groups)];
                if site != "compact" {
                    shape.push(("ranks", ranks));
                }
                domain.insert(key(
                    &format!("dense_prefill.{site}.prefill"),
                    "mxfp8",
                    &shape,
                    q.m_capacity,
                ));
            }
        }
    }
    assert_eq!(domain.len(), 83);
    assert_untuned(
        Family::DensePrefill,
        domain,
        UNTUNED,
        &[
            ("prefill_tiles", &["matrix_fetch", "stages", "tile_rows"]),
            ("tma_scales", &[]),
        ],
    );
}

const UNTUNED: &[&str] = &[
    "dense_prefill.compact.prefill mxfp8 groups=1,k=1152,n=5120 buckets=8192 rows=1..127,129..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=1280,n=16384 buckets=8192 rows=all",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=1280,n=32768 buckets=8192 rows=1..127,129..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=1280,n=4096 buckets=8192 rows=1..127,129..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=1280,n=8192 buckets=8192 rows=1..127,129..511,513..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=15360,n=5120 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=2048,n=5120 buckets=8192 rows=1..127,129..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=2304,n=5120 buckets=8192 rows=1..127,129..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=4096,n=5120 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=5120,n=1152 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=5120,n=1280 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=5120,n=2304 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=5120,n=512 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=5120,n=576 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=576,n=5120 buckets=8192 rows=1..127,129..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=6144,n=25600 buckets=8192 rows=1..127,129..511,513..2047,2049..8192",
    "dense_prefill.compact.prefill mxfp8 groups=1,k=8192,n=5120 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=2,k=4096,n=1024 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=4,k=4096,n=1024 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.compact.prefill mxfp8 groups=8,k=4096,n=1024 buckets=8192 rows=1..127,129..511,513..2047,2049..4095,4097..8191",
    "dense_prefill.dspark_main.prefill mxfp8 groups=1,k=15360,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.dspark_main.prefill mxfp8 groups=1,k=15360,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.dspark_main.prefill mxfp8 groups=1,k=15360,n=5120,ranks=4 buckets=8192 rows=1..128,512..8192",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1152,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1280,n=16384,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1280,n=32768,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1280,n=4096,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1280,n=4096,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1280,n=4096,ranks=4 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=1280,n=8192,ranks=4 buckets=8192 rows=1..191,193..447,449..959,961..1471,1473..2495,2497..4095,4097..8191",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=15360,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=15360,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=15360,n=5120,ranks=4 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=2048,n=5120,ranks=4 buckets=8192 rows=1..191,193..447,449..959,961..1471,1473..2495,2497..4095,4097..8191",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=2304,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=4096,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=1152,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=1280,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=1280,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=1280,ranks=4 buckets=8192 rows=1..191,193..447,449..959,961..1471,1473..2495,2497..4095,4097..8191",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=2304,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=512,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=512,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=512,ranks=4 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=5120,n=576,ranks=4 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=576,n=5120,ranks=4 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=6144,n=25600,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=6144,n=25600,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=6144,n=25600,ranks=4 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=1,k=8192,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=2,k=4096,n=1024,ranks=4 buckets=8192 rows=1..191,193..447,449..959,961..1471,1473..2495,2497..4095,4097..8191",
    "dense_prefill.matrix.prefill mxfp8 groups=4,k=4096,n=1024,ranks=2 buckets=8192 rows=all",
    "dense_prefill.matrix.prefill mxfp8 groups=8,k=4096,n=1024,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1152,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1280,n=16384,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1280,n=32768,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1280,n=4096,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1280,n=4096,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1280,n=4096,ranks=4 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=1280,n=8192,ranks=4 buckets=8192 rows=1..4095,4097..8191",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=15360,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=15360,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=15360,n=5120,ranks=4 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=2048,n=5120,ranks=4 buckets=8192 rows=1..4095,4097..8191",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=2304,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=4096,n=5120,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=1152,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=1280,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=1280,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=1280,ranks=4 buckets=8192 rows=1..4095,4097..8191",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=2304,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=512,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=512,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=512,ranks=4 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=5120,n=576,ranks=4 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=576,n=5120,ranks=4 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=6144,n=25600,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=6144,n=25600,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=6144,n=25600,ranks=4 buckets=8192 rows=1..8191",
    "dense_prefill.tma_scales.prefill mxfp8 groups=1,k=8192,n=5120,ranks=1 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=2,k=4096,n=1024,ranks=4 buckets=8192 rows=1..4095,4097..8191",
    "dense_prefill.tma_scales.prefill mxfp8 groups=4,k=4096,n=1024,ranks=2 buckets=8192 rows=all",
    "dense_prefill.tma_scales.prefill mxfp8 groups=8,k=4096,n=1024,ranks=1 buckets=8192 rows=all",
];
