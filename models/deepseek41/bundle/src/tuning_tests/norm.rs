use crate::sm120::{ceil_cached, ceil_parallel, ceil_prefill};
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
            op: if op.starts_with("ceil_") {
                let (site, phase) = op.rsplit_once('.').unwrap();
                format!("norm.{site}_{weights}.{phase}")
            } else {
                format!("norm.{op}")
            },
            weights: if op.starts_with("ceil_") || op.starts_with("activation_") {
                "none"
            } else {
                "bf16"
            }
            .into(),
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
        ("input_cached.decode", vec![(1, 32), (8, 32)]),
        (
            "query_cached.decode",
            vec![(1, 16), (6, 16), (8, 32), (32, 32)],
        ),
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
        for (rows, warps) in vec![
            (2, if index == 1 { 16 } else { 32 }),
            (4, if index == 1 { 16 } else { 32 }),
        ] {
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
            true,
        ),
        (
            "query_strategy.prefill",
            rows::norm_quantizers(&FLASH)[1].norm.width,
            false,
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
        for (index, (site, chosen)) in [
            (
                "input_admit.prefill",
                rows == 8192 || ((64..=2624).contains(&rows) && (rows - 64) % 128 == 0),
            ),
            ("query_admit.prefill", rows == 8192),
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
        if matches!(rows, 2 | 4) {
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
            let capacity = legacy_capacity(ranks, &entry);
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
        for entry in legacy_registered(ranks) {
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
        let capacity = if ranks == 4 { 64 } else { 60 };
        add(
            &mut t,
            "activation_capacity.decode",
            "bf16",
            &[("ranks", ranks)],
            64,
            None,
            "parallel",
            [("capacity".into(), Knob::Count(capacity))].into(),
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
    for count in 0..=8193 {
        for phase in [
            crate::program::Phase::Decode(count),
            crate::program::Phase::Verify {
                rows: count,
                items: 1,
            },
            crate::program::Phase::Prefill { rows: count },
        ] {
            assert_eq!(
                crate::ops::norm_prefill::qualified(&FLASH, phase),
                phase.prefill()
                    && (count == 8192 || ((64..=2624).contains(&count) && (count - 64) % 128 == 0))
            );
            assert_eq!(
                crate::ops::query_prefill::eligible(&FLASH, phase),
                phase.prefill() && count == 8192
            );
            assert_eq!(
                crate::ops::norm_quantize::small_selected(phase),
                matches!(phase, crate::program::Phase::Decode(2 | 4))
            );
        }
    }
    for ranks in [1, 2, 4] {
        for entry in ceil_parallel::baseline_quantizers(&FLASH, ranks).unwrap() {
            assert_eq!(
                ceil_parallel::capacity(&FLASH, ranks, &entry),
                legacy_capacity(ranks, &entry)
            );
        }
        let actual = ceil_parallel::registered_quantizers(&FLASH, ranks).unwrap();
        let old = legacy_registered(ranks);
        assert_eq!(actual.len(), old.len());
        for (a, b) in actual.iter().zip(&old) {
            assert_eq!((&a.symbol, a.capacity), (&b.symbol, b.capacity));
        }
        assert_eq!(
            ceil_parallel::activation(&FLASH, ranks).unwrap().capacity,
            if ranks == 4 { 64 } else { 60 }
        );
    }
}

fn legacy_capacity(ranks: u32, entry: &kernels_sm120::row::block_quantize::Entry) -> u32 {
    if ranks == 4 {
        return 64;
    }
    let intermediate = FLASH.shape.moe.top_k * (FLASH.shape.moe.intermediate / ranks);
    let draft = FLASH.shape.dspark.top_k * (FLASH.shape.moe.intermediate / ranks);
    if entry.symbol == format!("chop_{}_moe_ceil_w{intermediate}_v1", FLASH.kernel_tag)
        && intermediate != draft
    {
        32
    } else {
        60
    }
}
fn legacy_registered(ranks: u32) -> Vec<kernels_sm120::row::block_quantize::Entry> {
    let intermediate = FLASH.shape.moe.top_k * (FLASH.shape.moe.intermediate / ranks);
    let draft = FLASH.shape.dspark.top_k * (FLASH.shape.moe.intermediate / ranks);
    ceil_parallel::baseline_quantizers(&FLASH, ranks)
        .unwrap()
        .into_iter()
        .map(|mut e| {
            e.capacity = legacy_capacity(ranks, &e);
            if e.query == model_deepseek41_contract::moe::quantizer(intermediate)
                && intermediate != draft
            {
                e.capacity = 32;
            }
            if e.symbol == "chop_deepseek41_latent_ceil32_v1" {
                e.symbol = format!("chop_{}_latent_ceil32_v1", FLASH.kernel_tag);
            }
            e.symbol = format!("{}_parallel_v1", e.symbol.strip_suffix("_v1").unwrap());
            e.parallel().unwrap()
        })
        .collect()
}
