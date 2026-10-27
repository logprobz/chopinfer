use crate::sm120::mxfp8_fold;
use kernel_build::tuning::Key;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{self, Family},
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) type Coverage = BTreeMap<Key, BTreeSet<u32>>;

pub(super) fn fold_shape(q: Mxfp8FoldGemmQuery, paired: bool) -> BTreeMap<String, u32> {
    [
        ("n", q.n),
        ("k", q.k),
        ("groups", q.groups),
        ("paired", u32::from(paired)),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v))
    .collect()
}
pub(super) fn cover(
    coverage: &mut Coverage,
    op: &str,
    weights: &str,
    shape: BTreeMap<String, u32>,
    bucket: u32,
) {
    coverage
        .entry(Key {
            op: op.into(),
            weights: weights.into(),
            shape,
            bucket,
            rows: None,
        })
        .or_insert_with(|| (1..=bucket).collect());
}
fn served() -> Coverage {
    let mut keys = Coverage::new();
    // Decode and verify use this registry, as do qualified short prefills.
    // Capacity is the registry's 64-row bound; exact-row offers are retargeted
    // only after selection. Paired offers exist only for shared gate/up.
    for ranks in [1, 2, 4] {
        for registration in mxfp8_fold::decode_registrations(&FLASH, ranks, 0).unwrap() {
            let q = registration.entry.query;
            let paired = registration.entry.paired;
            for site in ["fold", "split", "stream", "direct"] {
                cover(
                    &mut keys,
                    &format!("dense_fold.{site}.decode"),
                    "mxfp8",
                    fold_shape(q, site != "split" && paired),
                    64,
                );
            }
        }
    }
    let shape: BTreeMap<String, u32> = [("ranks".into(), 4)].into();
    for (site, phase, bucket) in [
        ("status", "decode", 32),
        ("status_tails", "decode", 64),
        ("status_wide_tails", "decode", 64),
        ("status_blocks", "decode", 64),
        ("status_wide", "verify", 64),
    ] {
        cover(
            &mut keys,
            &format!("dense_fold.{site}.{phase}"),
            "mxfp8",
            shape.clone(),
            bucket,
        );
    }
    for id in crate::sm120::fold_verify::IDS {
        let site = id.replace(['.', '-'], "_");
        cover(
            &mut keys,
            &format!("dense_fold.{site}_cluster.decode"),
            "mxfp8",
            shape.clone(),
            32,
        );
        cover(
            &mut keys,
            &format!("dense_fold.{site}_status.verify"),
            "mxfp8",
            shape.clone(),
            64,
        );
    }
    keys
}
fn ranges(rows: &[u32]) -> String {
    let mut parts = Vec::new();
    let mut at = 0;
    while at < rows.len() {
        let first = rows[at];
        while at + 1 < rows.len() && rows[at + 1] == rows[at] + 1 {
            at += 1;
        }
        let last = rows[at];
        parts.push(if first == last {
            first.to_string()
        } else {
            format!("{first}..={last}")
        });
        at += 1;
    }
    parts.join(",")
}
pub(super) fn assert_untuned(family: Family, coverage: Coverage, expected: &str) {
    let table = tuning::table(&FLASH, family).unwrap();
    let mut missing = Vec::new();
    for (key, rows) in &coverage {
        let shape = key
            .shape
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        let absent = rows
            .iter()
            .copied()
            .filter(|&rows| {
                tuning::select(
                    &FLASH,
                    family,
                    &key.op,
                    &key.weights,
                    &shape,
                    key.bucket,
                    rows,
                )
                .is_none()
            })
            .collect::<Vec<_>>();
        if !absent.is_empty() {
            missing.push(format!("{key}: {}", ranges(&absent)));
        }
    }
    for row in table.rows() {
        let mut key = row.key.clone();
        key.rows = None;
        assert!(coverage.contains_key(&key), "uncovered table key {key}");
    }
    let actual = missing.join("\n");
    println!("{} missing keys:\n{}\nEND MISSING", family.name(), actual);
    assert_eq!(
        actual,
        expected.trim(),
        "untuned {} keys changed",
        family.name()
    );
}
#[test]
fn pin_untuned_served_keys() {
    assert_untuned(Family::DenseFold, served(), UNTUNED);
}
const UNTUNED: &str = r#"
dense_fold.attention_kv_cluster.decode mxfp8 ranks=4 t32: 1..=15,17..=31
dense_fold.attention_kv_status.verify mxfp8 ranks=4 t64: 1..=64
dense_fold.attention_o_a_cluster.decode mxfp8 ranks=4 t32: 1..=15,17..=31
dense_fold.attention_o_a_status.verify mxfp8 ranks=4 t64: 1..=32
dense_fold.attention_o_b_cluster.decode mxfp8 ranks=4 t32: 1..=15,17..=31
dense_fold.attention_o_b_status.verify mxfp8 ranks=4 t64: 1..=32
dense_fold.attention_q_a_cluster.decode mxfp8 ranks=4 t32: 1..=15,17..=31
dense_fold.attention_q_a_status.verify mxfp8 ranks=4 t64: 1..=32
dense_fold.direct.decode mxfp8 groups=1,k=576,n=5120,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=1152,n=5120,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=1280,n=4096,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=1280,n=8192,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=1280,n=16384,paired=0 t64: 2..=64
dense_fold.direct.decode mxfp8 groups=1,k=1280,n=32768,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=2048,n=5120,paired=0 t64: 3..=64
dense_fold.direct.decode mxfp8 groups=1,k=2304,n=5120,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=4096,n=5120,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=512,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=576,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=576,paired=1 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=1152,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=1152,paired=1 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=1280,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=2304,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=5120,n=2304,paired=1 t64: 3..=64
dense_fold.direct.decode mxfp8 groups=1,k=6144,n=25600,paired=0 t64: 1..=31,33..=63
dense_fold.direct.decode mxfp8 groups=1,k=8192,n=5120,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=1,k=15360,n=5120,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=2,k=4096,n=1024,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=4,k=4096,n=1024,paired=0 t64: 1..=64
dense_fold.direct.decode mxfp8 groups=8,k=4096,n=1024,paired=0 t64: 2..=64
dense_fold.fold.decode mxfp8 groups=1,k=1152,n=5120,paired=0 t64: 3..=8,17..=21,28,33..=47,49..=63
dense_fold.fold.decode mxfp8 groups=1,k=1280,n=4096,paired=0 t64: 1,3,5..=9,12,14..=16
dense_fold.fold.decode mxfp8 groups=1,k=1280,n=8192,paired=0 t64: 4,13..=15,17..=64
dense_fold.fold.decode mxfp8 groups=1,k=1280,n=16384,paired=0 t64: 5..=6,12..=64
dense_fold.fold.decode mxfp8 groups=1,k=1280,n=32768,paired=0 t64: 1..=64
dense_fold.fold.decode mxfp8 groups=1,k=2048,n=5120,paired=0 t64: 33..=64
dense_fold.fold.decode mxfp8 groups=1,k=2304,n=5120,paired=0 t64: 2,5..=6,8,12..=13,15..=64
dense_fold.fold.decode mxfp8 groups=1,k=4096,n=5120,paired=0 t64: 1..=27,29..=31,33..=64
dense_fold.fold.decode mxfp8 groups=1,k=5120,n=1152,paired=0 t64: 22,57..=58,60,64
dense_fold.fold.decode mxfp8 groups=1,k=5120,n=1152,paired=1 t64: 17..=23,25..=34,36,38..=41,43,45..=46,52..=64
dense_fold.fold.decode mxfp8 groups=1,k=5120,n=2304,paired=0 t64: 18..=27,33..=50,52..=55,58
dense_fold.fold.decode mxfp8 groups=1,k=5120,n=2304,paired=1 t64: 1..=64
dense_fold.fold.decode mxfp8 groups=1,k=6144,n=25600,paired=0 t64: 1..=64
dense_fold.fold.decode mxfp8 groups=1,k=8192,n=5120,paired=0 t64: 1..=33,36,38,40,42..=43,45..=64
dense_fold.fold.decode mxfp8 groups=1,k=15360,n=5120,paired=0 t64: 1..=16,22,26..=64
dense_fold.split.decode mxfp8 groups=1,k=576,n=5120,paired=0 t64: 29..=64
dense_fold.split.decode mxfp8 groups=1,k=1152,n=5120,paired=0 t64: 3..=8,17..=64
dense_fold.split.decode mxfp8 groups=1,k=1280,n=4096,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=1280,n=8192,paired=0 t64: 3..=4,13..=15,17..=64
dense_fold.split.decode mxfp8 groups=1,k=1280,n=16384,paired=0 t64: 5..=6,12..=64
dense_fold.split.decode mxfp8 groups=1,k=1280,n=32768,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=2048,n=5120,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=2304,n=5120,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=4096,n=5120,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=5120,n=512,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=5120,n=576,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=5120,n=1152,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=5120,n=1280,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=5120,n=2304,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=6144,n=25600,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=8192,n=5120,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=1,k=15360,n=5120,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=2,k=4096,n=1024,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=4,k=4096,n=1024,paired=0 t64: 1..=64
dense_fold.split.decode mxfp8 groups=8,k=4096,n=1024,paired=0 t64: 1..=64
dense_fold.status.decode mxfp8 ranks=4 t32: 3,5..=7,9..=15,17..=31
dense_fold.status_blocks.decode mxfp8 ranks=4 t64: 1..=4,6..=9,11..=14,16..=19,21..=24,26..=29,31..=34,36..=39,41..=44,46..=49,51..=54,56..=59,61..=64
dense_fold.status_tails.decode mxfp8 ranks=4 t64: 1..=2,4,8,16,32..=34,36..=39,41..=44,46..=49,51..=54,56..=59,61..=64
dense_fold.status_wide.verify mxfp8 ranks=4 t64: 1..=32
dense_fold.status_wide_tails.decode mxfp8 ranks=4 t64: 1..=34,36..=39,41..=44,46..=49,51..=54,56..=59,61..=64
dense_fold.stream.decode mxfp8 groups=1,k=576,n=5120,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=1152,n=5120,paired=0 t64: 3,5..=64
dense_fold.stream.decode mxfp8 groups=1,k=1280,n=4096,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=1280,n=8192,paired=0 t64: 3,5,7,9..=64
dense_fold.stream.decode mxfp8 groups=1,k=1280,n=16384,paired=0 t64: 3,5,7,9..=64
dense_fold.stream.decode mxfp8 groups=1,k=1280,n=32768,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=2048,n=5120,paired=0 t64: 3,5..=64
dense_fold.stream.decode mxfp8 groups=1,k=2304,n=5120,paired=0 t64: 3,5,7,9..=64
dense_fold.stream.decode mxfp8 groups=1,k=4096,n=5120,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=512,paired=0 t64: 1..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=576,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=576,paired=1 t64: 1..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=1152,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=1152,paired=1 t64: 3..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=1280,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=2304,paired=0 t64: 1..=64
dense_fold.stream.decode mxfp8 groups=1,k=5120,n=2304,paired=1 t64: 3..=64
dense_fold.stream.decode mxfp8 groups=1,k=6144,n=25600,paired=0 t64: 3,5,7,9..=15,17..=64
dense_fold.stream.decode mxfp8 groups=1,k=8192,n=5120,paired=0 t64: 3,5,7,9..=15,17..=31,33..=64
dense_fold.stream.decode mxfp8 groups=1,k=15360,n=5120,paired=0 t64: 3..=64
dense_fold.stream.decode mxfp8 groups=2,k=4096,n=1024,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=4,k=4096,n=1024,paired=0 t64: 2..=64
dense_fold.stream.decode mxfp8 groups=8,k=4096,n=1024,paired=0 t64: 1..=64
"#;

#[test]
fn local_wide_rows_keep_the_qualified_launches() {
    let table = tuning::table(&FLASH, Family::DenseFold).unwrap();
    let candidate = crate::sm120::fold_verify_local::entry().unwrap();
    let mut checked = 0;
    for row in table
        .rows()
        .iter()
        .filter(|row| row.schedule == "local_wide")
    {
        assert_eq!(
            row.knobs.keys().map(String::as_str).collect::<Vec<_>>(),
            ["stages", "tile_n", "tile_rows"]
        );
        assert_eq!(
            (
                candidate.entry().stages,
                candidate.entry().tile_n,
                candidate.entry().base.query.m_capacity
            ),
            (4, 48, 32)
        );
        let [lo, hi] = row.key.rows.unwrap_or([1, row.key.bucket]);
        for rows in lo..=hi {
            assert!(crate::sm120::fold_verify_local::selected(&row.key.op, rows));
            let plan = candidate.plan(rows).unwrap();
            let mut original = candidate.entry().plan(32).unwrap();
            original.grid[1] = rows.div_ceil(32);
            assert_eq!(plan, original);
        }
        checked += 1;
    }
    assert_eq!(checked, 7);
}

#[test]
fn cluster_rows_keep_cta_local_storage() {
    let table = tuning::table(&FLASH, Family::DenseFold).unwrap();
    let mut checked = 0;
    for row in table
        .rows()
        .iter()
        .filter(|row| row.schedule == "cluster_fused")
    {
        assert_eq!(
            row.knobs.keys().map(String::as_str).collect::<Vec<_>>(),
            ["distributed", "splits", "stages", "tile_n"]
        );
        assert_eq!(row.knobs["distributed"].flag(), Some(false));
        let (site, phase) = row
            .key
            .op
            .strip_prefix("dense_fold.")
            .unwrap()
            .rsplit_once('.')
            .unwrap();
        let id = site
            .trim_end_matches("_cluster")
            .trim_end_matches("_status")
            .replace("attention_", "attention.")
            .replace('_', "-");
        let [lo, hi] = row.key.rows.unwrap_or([1, row.key.bucket]);
        for rows in lo..=hi {
            let entry = if phase == "decode" {
                crate::sm120::fold_cluster::served(&id, rows).unwrap()
            } else {
                crate::sm120::fold_verify::served(&id, rows)
                    .unwrap()
                    .cluster
            };
            assert!(!entry.distributed);
            assert_eq!(entry.tile_n, row.knobs["tile_n"].count().unwrap());
            assert_eq!(entry.splits, row.knobs["splits"].count().unwrap());
            assert_eq!(entry.stages, row.knobs["stages"].count().unwrap());
        }
        checked += 1;
    }
    assert_eq!(checked, 11);
}

#[test]
fn direct_rows_keep_qualified_warp_layouts() {
    let table = tuning::table(&FLASH, Family::DenseFold).unwrap();
    let mut layouts = BTreeMap::new();
    for row in table.rows().iter().filter(|row| row.schedule == "direct") {
        assert_eq!(
            row.knobs.keys().map(String::as_str).collect::<Vec<_>>(),
            ["k_warps", "n_warps"]
        );
        let warps = (
            row.knobs["n_warps"].count().unwrap(),
            row.knobs["k_warps"].count().unwrap(),
        );
        assert!(matches!(warps, (4, 1) | (1, 4)));
        *layouts.entry(warps).or_insert(0) += 1;
    }
    assert_eq!(layouts, [((4, 1), 3), ((1, 4), 3)].into());
}
