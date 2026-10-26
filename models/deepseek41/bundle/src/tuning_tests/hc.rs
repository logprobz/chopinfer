use crate::{program::Phase, sm120::hc_prefill_expand_mix};
use kernel_build::tuning::{Key, Knob, Knobs, Row, Table, Target};
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};

fn shape() -> Vec<(&'static str, u32)> {
    vec![
        ("groups", FLASH.shape.hc.streams),
        ("width", FLASH.shape.hidden),
    ]
}

fn add(
    table: &mut Table,
    site: &str,
    phase: &str,
    shape: &[(&str, u32)],
    bucket: u32,
    schedule: &str,
    cases: impl IntoIterator<Item = (u32, Knobs)>,
) {
    let mut cases = cases.into_iter().collect::<Vec<_>>();
    cases.sort_by_key(|(rows, _)| *rows);
    let mut ranges: Vec<(u32, u32, Knobs)> = Vec::new();
    for (rows, knobs) in cases {
        if let Some((_, hi, old)) = ranges.last_mut()
            && *hi + 1 == rows
            && old == &knobs
        {
            *hi = rows;
        } else {
            ranges.push((rows, rows, knobs));
        }
    }
    for (lo, hi, knobs) in ranges {
        table
            .upsert(Row {
                key: Key {
                    op: format!("hc.{site}.{phase}"),
                    weights: "f32".into(),
                    shape: shape.iter().map(|&(k, v)| (k.into(), v)).collect(),
                    bucket,
                    rows: (lo != 1 || hi != bucket).then_some([lo, hi]),
                },
                schedule: schedule.into(),
                knobs,
            })
            .unwrap();
    }
}

fn counts(entries: impl IntoIterator<Item = (u32, u32)>, knob: &str) -> Vec<(u32, Knobs)> {
    entries
        .into_iter()
        .map(|(rows, value)| (rows, [(knob.into(), Knob::Count(value))].into()))
        .collect()
}

fn generated() -> Table {
    let mut t = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    let plain = |rows: Vec<u32>| {
        rows.into_iter()
            .map(|r| (r, Knobs::new()))
            .collect::<Vec<_>>()
    };
    add(
        &mut t,
        "lagged_warp",
        "decode",
        &shape(),
        32,
        "warp",
        plain(vec![1, 4, 6, 8, 12, 16, 32]),
    );
    for ranks in [1, 2, 4] {
        let mut s = shape();
        s.push(("ranks", ranks));
        add(
            &mut t,
            "round_chain",
            "decode",
            &s,
            64,
            "chain",
            counts(
                if ranks == 4 {
                    vec![
                        (1, 8),
                        (2, 8),
                        (3, 8),
                        (4, 8),
                        (5, 4),
                        (6, 4),
                        (7, 4),
                        (8, 4),
                    ]
                } else {
                    vec![(1, 8), (8, 4)]
                },
                "warps",
            ),
        );
    }
    add(
        &mut t,
        "finish_pair",
        "verify",
        &shape(),
        64,
        "pair",
        plain(
            [1, 4, 6, 8, 12, 16, 32]
                .into_iter()
                .chain(33..=64)
                .collect(),
        ),
    );
    add(
        &mut t,
        "finish_chain",
        "verify",
        &shape(),
        64,
        "chain",
        counts(
            [(1, 8), (4, 8), (6, 4), (8, 4), (12, 4), (16, 4), (32, 4)]
                .into_iter()
                .chain((33..=64).map(|r| (r, 4))),
            "warps",
        ),
    );
    add(
        &mut t,
        "finish_rows",
        "verify",
        &shape(),
        64,
        "rows",
        counts([(32, 2)], "tile"),
    );
    add(
        &mut t,
        "finish_twin",
        "verify",
        &shape(),
        64,
        "twin",
        counts([(6, 32)], "warps"),
    );
    add(
        &mut t,
        "finish_cols",
        "verify",
        &shape(),
        64,
        "cols",
        [1, 8, 16, 32]
            .into_iter()
            .chain(33..=64)
            .map(|rows| kernels_sm120::sinkhorn_hc::finish_split::Bucket {
                rows,
                tile: if rows == 1 { 2 } else { 4 },
                warps: 32,
                transposed: true,
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|b| {
                (
                    b.rows,
                    [
                        ("tile".into(), Knob::Count(b.tile)),
                        ("warps".into(), Knob::Count(b.warps)),
                        ("transposed".into(), Knob::Flag(b.transposed)),
                    ]
                    .into(),
                )
            }),
    );
    for (site, buckets) in [
        (
            "lagged_mix",
            vec![
                ((8, 8), vec![64, 192]),
                ((8, 4), vec![2240, 2496, 2624, 8192]),
            ],
        ),
        (
            "expand_mix",
            hc_prefill_expand_mix::BUCKETS
                .iter()
                .map(|&(pair, rows)| (pair, rows.to_vec()))
                .collect(),
        ),
    ] {
        add(
            &mut t,
            site,
            "prefill",
            &shape(),
            8192,
            "pipeline",
            buckets.into_iter().flat_map(|((tile, chains), rows)| {
                rows.into_iter().map(move |r| {
                    (
                        r,
                        [
                            ("tile".into(), Knob::Count(tile)),
                            ("chains".into(), Knob::Count(chains)),
                        ]
                        .into(),
                    )
                })
            }),
        );
    }
    for capacity in 32..=64 {
        for phase in ["decode", "verify", "prefill"] {
            for (index, site) in ["finish_admit", "wide_mix", "decode_mix", "draft_mix"]
                .into_iter()
                .enumerate()
            {
                // These two mixers have fixed prepared capacities; only the
                // finish and wide mixer queries use the instance's verify bound.
                if (site == "decode_mix" && capacity != 32)
                    || (site == "draft_mix" && capacity != 60)
                {
                    continue;
                }
                add(
                    &mut t,
                    site,
                    phase,
                    &shape(),
                    capacity,
                    "admit",
                    (1..=capacity).filter_map(|rows| {
                        let p = match phase {
                            "decode" => Phase::Decode(rows),
                            "verify" => Phase::Verify { rows, items: 1 },
                            _ => Phase::Prefill { rows },
                        };
                        crate::ops::Ops::migration_hc_choices(p, capacity)[index]
                            .then(|| (rows, Knobs::new()))
                    }),
                );
            }
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
    std::fs::write(out.join("hc.json"), generated().render()).unwrap();
}

#[test]
fn differential() {
    let expected = generated();
    let actual = table(&FLASH, Family::Hc).unwrap();
    assert_eq!(actual.render(), expected.render());
    let mut cases = 0;
    for row in expected.rows() {
        for rows in 0..=row.key.bucket + 1 {
            let key = &row.key;
            assert_eq!(
                actual.select(&key.op, &key.weights, &key.shape, key.bucket, rows),
                expected.select(&key.op, &key.weights, &key.shape, key.bucket, rows)
            );
            cases += 1;
        }
    }
    assert!(cases > 50_000);
    eprintln!("HC differential: {cases} cases");
    for capacity in 32..=64 {
        for rows in 0..=8193 {
            for phase in [
                Phase::Decode(rows),
                Phase::Verify { rows, items: 1 },
                Phase::Prefill { rows },
            ] {
                assert_eq!(
                    crate::ops::Ops::migration_hc_actual(phase, capacity),
                    crate::ops::Ops::migration_hc_choices(phase, capacity),
                    "{phase:?} capacity={capacity}"
                );
            }
        }
    }
}
