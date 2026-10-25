use crate::{
    program::Phase,
    sm120::{hc_cols, hc_finish_norm as tail, hc_prefill_expand_mix, hc_round_chain},
};
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
        plain(tail::migration_mixer_buckets()),
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
            counts(hc_round_chain::buckets(ranks), "warps"),
        );
    }
    add(
        &mut t,
        "finish_pair",
        "verify",
        &shape(),
        64,
        "pair",
        plain(tail::pair_buckets()),
    );
    add(
        &mut t,
        "finish_chain",
        "verify",
        &shape(),
        64,
        "chain",
        counts(tail::chain_buckets(), "warps"),
    );
    add(
        &mut t,
        "finish_rows",
        "verify",
        &shape(),
        64,
        "rows",
        counts(tail::row_buckets(), "tile"),
    );
    add(
        &mut t,
        "finish_twin",
        "verify",
        &shape(),
        64,
        "twin",
        counts(tail::TWIN_BUCKETS, "warps"),
    );
    add(
        &mut t,
        "finish_cols",
        "verify",
        &shape(),
        64,
        "cols",
        hc_cols::buckets().into_iter().map(|b| {
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
        ("lagged_mix", tail::migration_prefill_buckets()),
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
}
