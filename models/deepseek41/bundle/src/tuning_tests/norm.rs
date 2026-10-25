use crate::{
    Kernels,
    program::Phase,
    sm120::{
        ceil_cached, ceil_parallel, ceil_prefill, norm_prefill, norm_quantize_cached,
        norm_query_cached, query_prefill,
    },
};
use kernel_build::tuning::{Key, Knob, Knobs, Row, Table, Target};
use kernel_core::contract::DType;
use model_deepseek41_contract::{
    geometry::FLASH,
    rows,
    tuning::{Family, table},
};

fn add(
    t: &mut Table,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
    range: Option<[u32; 2]>,
    schedule: &str,
    knobs: Knobs,
) {
    t.upsert(Row {
        key: Key {
            op: format!("norm.{op}"),
            weights: weights.into(),
            shape: shape.iter().map(|&(k, v)| (k.into(), v)).collect(),
            bucket,
            rows: range,
        },
        schedule: schedule.into(),
        knobs,
    })
    .unwrap();
}

fn dtype(input: DType) -> &'static str {
    match input {
        DType::Bf16 => "bf16",
        DType::F32 => "f32",
        _ => unreachable!(),
    }
}

fn generated() -> Table {
    let mut t = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    for (index, (site, buckets)) in [
        ("input_cached.decode", norm_quantize_cached::buckets()),
        ("query_cached.decode", norm_query_cached::BUCKETS.to_vec()),
    ]
    .into_iter()
    .enumerate()
    {
        let width = rows::norm_quantizers(&FLASH)[index].norm.width;
        for (rows, warps) in buckets {
            add(
                &mut t,
                site,
                "f32",
                &[("width", width)],
                64,
                Some([rows, rows]),
                "cached",
                [("warps".into(), Knob::Count(warps))].into(),
            );
        }
        for (rows, warps) in Kernels::small_norm_buckets(index == 1) {
            let site = if index == 0 {
                "input_small.decode"
            } else {
                "query_small.decode"
            };
            add(
                &mut t,
                site,
                "f32",
                &[("ranks", 4), ("width", width)],
                4,
                Some([rows, rows]),
                "cached",
                [("warps".into(), Knob::Count(warps))].into(),
            );
        }
    }
    for (site, width, cached) in [
        (
            "input_strategy.prefill",
            rows::norm_quantizers(&FLASH)[0].norm.width,
            norm_prefill::selected().unwrap().cache_groups,
        ),
        (
            "query_strategy.prefill",
            rows::norm_quantizers(&FLASH)[1].norm.width,
            query_prefill::selected().unwrap().cache_input,
        ),
    ] {
        add(
            &mut t,
            site,
            "f32",
            &[("width", width)],
            8192,
            None,
            if cached { "cached" } else { "reread" },
            Knobs::new(),
        );
    }
    for rows in 1..=8192 {
        let phase = Phase::Prefill { rows };
        for (index, (site, chosen)) in [
            (
                "input_admit.prefill",
                crate::ops::norm_prefill::qualified(&FLASH, phase),
            ),
            (
                "query_admit.prefill",
                crate::ops::query_prefill::eligible(&FLASH, phase),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            if chosen {
                add(
                    &mut t,
                    site,
                    "f32",
                    &[("width", rows::norm_quantizers(&FLASH)[index].norm.width)],
                    8192,
                    Some([rows, rows]),
                    "admit",
                    Knobs::new(),
                );
            }
        }
    }
    for rows in 1..=4 {
        if crate::ops::norm_quantize::small_selected(Phase::Decode(rows)) {
            add(
                &mut t,
                "small_admit.decode",
                "f32",
                &[("ranks", 4)],
                4,
                Some([rows, rows]),
                "admit",
                Knobs::new(),
            );
        }
    }
    for ranks in [1, 2, 4] {
        for entry in ceil_parallel::baseline_quantizers(&FLASH, ranks).unwrap() {
            let shape = [("ranks", ranks), ("width", entry.query.width)];
            let capacity = ceil_parallel::capacity(&FLASH, ranks, &entry);
            add(
                &mut t,
                "ceil_capacity.decode",
                dtype(entry.query.input),
                &shape,
                entry.capacity,
                None,
                "parallel",
                [("capacity".into(), Knob::Count(capacity))].into(),
            );
        }
        for entry in ceil_parallel::registered_quantizers(&FLASH, ranks).unwrap() {
            let shape = [("ranks", ranks), ("width", entry.query.width)];
            add(
                &mut t,
                "ceil_registered.decode",
                dtype(entry.query.input),
                &shape,
                64,
                None,
                "parallel",
                [("capacity".into(), Knob::Count(entry.capacity))].into(),
            );
            if ceil_cached::cached(&entry).is_some() {
                add(
                    &mut t,
                    "ceil_cached.decode",
                    dtype(entry.query.input),
                    &shape,
                    entry.capacity,
                    None,
                    "cached",
                    Knobs::new(),
                );
            }
        }
        let entry = ceil_parallel::activation(&FLASH, ranks).unwrap();
        add(
            &mut t,
            "activation_capacity.decode",
            "bf16",
            &[("ranks", ranks)],
            64,
            None,
            "parallel",
            [("capacity".into(), Knob::Count(entry.capacity))].into(),
        );
        for entry in ceil_prefill::baseline_quantizers(&FLASH, ranks).unwrap() {
            add(
                &mut t,
                "ceil_prefill.prefill",
                dtype(entry.query.input),
                &[("ranks", ranks), ("width", entry.query.width)],
                entry.capacity,
                None,
                "prefill",
                [(
                    "warps".into(),
                    Knob::Count(ceil_prefill::selected_warps(entry.query.width)),
                )]
                .into(),
            );
        }
    }
    t
}

#[test]
#[ignore = "one-time migration generator; writes to the external evidence directory"]
fn generate() {
    let out = std::path::PathBuf::from(
        std::env::var_os("CHOPINFER_TUNING_OUT").expect("external output directory"),
    );
    assert!(
        !out.canonicalize().unwrap().starts_with(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .canonicalize()
                .unwrap()
        )
    );
    std::fs::write(out.join("norm.json"), generated().render()).unwrap();
}

#[test]
fn differential() {
    let expected = generated();
    let actual = table(&FLASH, Family::Norm).unwrap();
    assert_eq!(actual.render(), expected.render());
    let mut cases = 0;
    for row in expected.rows() {
        let key = &row.key;
        for rows in 0..=key.bucket + 1 {
            assert_eq!(
                actual.select(&key.op, &key.weights, &key.shape, key.bucket, rows),
                expected.select(&key.op, &key.weights, &key.shape, key.bucket, rows)
            );
            cases += 1;
        }
    }
    assert!(cases > 100_000);
    eprintln!("Norm differential: {cases} cases");
}
