//! Served query domains and the keys that retain their baseline schedule.
use kernel_build::tuning::Key;
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{self, Family},
};

fn add(
    out: &mut std::collections::BTreeSet<Key>,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
) {
    out.insert(Key {
        op: op.into(),
        weights: weights.into(),
        shape: shape.iter().map(|&(k, v)| (k.into(), v)).collect(),
        bucket,
        rows: None,
    });
}
fn dtype(dtype: kernel_core::contract::DType) -> &'static str {
    match dtype {
        kernel_core::contract::DType::Bf16 => "bf16",
        kernel_core::contract::DType::F32 => "f32",
        _ => panic!("norm table dtype"),
    }
}
fn domains() -> std::collections::BTreeSet<Key> {
    use crate::sm120::{ceil_parallel, ceil_prefill};
    use model_deepseek41_contract::{
        admission::SUPPORTED_RANKS,
        rows,
        serving::{PREFILL_TOKENS, VERIFY_DECODE_ROWS},
    };
    let mut out = std::collections::BTreeSet::new();
    for (site, query) in ["input", "query"]
        .into_iter()
        .zip(rows::norm_quantizers(&FLASH))
    {
        let shape = [("width", query.norm.width)];
        let weights = dtype(query.norm.weight);
        add(
            &mut out,
            &format!("norm.{site}_cached.decode"),
            weights,
            &shape,
            VERIFY_DECODE_ROWS,
        );
        for &ranks in SUPPORTED_RANKS {
            add(
                &mut out,
                &format!("norm.{site}_small.decode"),
                weights,
                &[("ranks", ranks), ("width", query.norm.width)],
                4,
            );
        }
        for kind in ["strategy", "admit"] {
            add(
                &mut out,
                &format!("norm.{site}_{kind}.prefill"),
                weights,
                &shape,
                PREFILL_TOKENS,
            );
        }
    }
    add(
        &mut out,
        "norm.small_admit.decode",
        "bf16",
        &[("ranks", 4)],
        4,
    );
    for &ranks in SUPPORTED_RANKS {
        for entry in ceil_parallel::baseline_quantizers(&FLASH, ranks).unwrap() {
            add(
                &mut out,
                &format!("norm.ceil_capacity_{}.decode", dtype(entry.query.input)),
                "none",
                &[("ranks", ranks), ("width", entry.query.width)],
                entry.capacity,
            );
        }
        for entry in ceil_parallel::registered_quantizers(&FLASH, ranks).unwrap() {
            let shape = [("ranks", ranks), ("width", entry.query.width)];
            add(
                &mut out,
                &format!("norm.ceil_registered_{}.decode", dtype(entry.query.input)),
                "none",
                &shape,
                VERIFY_DECODE_ROWS,
            );
            add(
                &mut out,
                &format!("norm.ceil_cached_{}.decode", dtype(entry.query.input)),
                "none",
                &shape,
                entry.capacity,
            );
        }
        add(
            &mut out,
            "norm.activation_capacity.decode",
            "none",
            &[("ranks", ranks)],
            VERIFY_DECODE_ROWS,
        );
        for entry in ceil_prefill::baseline_quantizers(&FLASH, ranks).unwrap() {
            add(
                &mut out,
                &format!("norm.ceil_prefill_{}.prefill", dtype(entry.query.input)),
                "none",
                &[("ranks", ranks), ("width", entry.query.width)],
                entry.capacity,
            );
        }
    }
    out
}

const UNTUNED: &[&str] = &[
    "norm.ceil_cached_bf16.decode none ranks=1,width=13824 t32: 1..=32",
    "norm.ceil_cached_bf16.decode none ranks=1,width=32768 t60: 1..=60",
    "norm.ceil_cached_bf16.decode none ranks=2,width=3456 t60: 1..=60",
    "norm.ceil_cached_bf16.decode none ranks=2,width=16384 t60: 1..=60",
    "norm.ceil_cached_bf16.decode none ranks=4,width=1728 t64: 1..=64",
    "norm.ceil_cached_bf16.decode none ranks=4,width=3456 t32: 1..=32",
    "norm.ceil_cached_f32.decode none ranks=1,width=512 t60: 1..=60",
    "norm.ceil_cached_f32.decode none ranks=1,width=1280 t60: 1..=60",
    "norm.ceil_cached_f32.decode none ranks=2,width=512 t60: 1..=60",
    "norm.ceil_cached_f32.decode none ranks=2,width=1280 t60: 1..=60",
    "norm.ceil_cached_f32.decode none ranks=4,width=512 t64: 1..=64",
    "norm.ceil_cached_f32.decode none ranks=4,width=1280 t64: 1..=64",
    "norm.ceil_cached_f32.decode none ranks=4,width=2048 t64: 1..=64",
    "norm.input_admit.prefill bf16 width=5120 t8192: 1..=63,65..=191,193..=319,321..=447,449..=575,577..=703,705..=831,833..=959,961..=1087,1089..=1215,1217..=1343,1345..=1471,1473..=1599,1601..=1727,1729..=1855,1857..=1983,1985..=2111,2113..=2239,2241..=2367,2369..=2495,2497..=2623,2625..=8191",
    "norm.input_cached.decode bf16 width=5120 t64: 2..=7,9..=64",
    "norm.input_small.decode bf16 ranks=1,width=5120 t4: 1..=4",
    "norm.input_small.decode bf16 ranks=2,width=5120 t4: 1..=4",
    "norm.input_small.decode bf16 ranks=4,width=5120 t4: 1,3",
    "norm.query_admit.prefill bf16 width=1280 t8192: 1..=8191",
    "norm.query_cached.decode bf16 width=1280 t64: 2..=5,7,9..=31,33..=64",
    "norm.query_small.decode bf16 ranks=1,width=1280 t4: 1..=4",
    "norm.query_small.decode bf16 ranks=2,width=1280 t4: 1..=4",
    "norm.query_small.decode bf16 ranks=4,width=1280 t4: 1,3",
    "norm.small_admit.decode bf16 ranks=4 t4: 1,3",
];

#[test]
fn pin_untuned_served_keys() {
    let family = Family::Norm;
    let table = tuning::table(&FLASH, family).unwrap();
    assert!(!table.rows().is_empty());
    let domains = domains();
    let mut missing = Vec::new();
    let mut cases = 0;
    for key in &domains {
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for rows in 1..=key.bucket {
            if table
                .select(&key.op, &key.weights, &key.shape, key.bucket, rows)
                .is_none()
            {
                if let Some((_, hi)) = ranges.last_mut()
                    && *hi + 1 == rows
                {
                    *hi = rows;
                } else {
                    ranges.push((rows, rows));
                }
            }
            cases += 1;
        }
        assert!(
            table
                .select(&key.op, &key.weights, &key.shape, key.bucket, 0)
                .is_none()
        );
        assert!(
            table
                .select(
                    &key.op,
                    &key.weights,
                    &key.shape,
                    key.bucket,
                    key.bucket + 1
                )
                .is_none()
        );
        if !ranges.is_empty() {
            let rows = ranges
                .into_iter()
                .map(|(lo, hi)| {
                    if lo == hi {
                        lo.to_string()
                    } else {
                        format!("{lo}..={hi}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            missing.push(format!("{key}: {rows}"));
        }
    }
    assert_eq!(missing, UNTUNED, "untuned served keys changed");
    for row in table.rows() {
        assert!(
            domains.contains(&Key {
                rows: None,
                ..row.key.clone()
            }),
            "table row is outside the pinned served domains: {}",
            row.key
        );
    }
    assert!(cases > 0);
    eprintln!(
        "{}: {cases} served lookup cases, {} untuned keys",
        family.name(),
        missing.len()
    );
}
