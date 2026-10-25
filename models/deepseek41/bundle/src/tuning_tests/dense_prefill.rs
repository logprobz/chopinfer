use crate::sm120::{mxfp8_fold, prefill_matrix, prefill_tma_scales};
use kernel_build::tuning::{Key, Knob, Row, Table, Target};
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};
use std::collections::BTreeMap;

fn cases(mut visit: impl FnMut(Key, Option<(&'static str, BTreeMap<String, Knob>)>)) {
    for ranks in [1, 2, 4] {
        for entry in mxfp8_fold::prefill_entries(&FLASH, ranks).unwrap() {
            let q = entry.query;
            for rows in 1..=q.m_capacity {
                for site in ["compact", "matrix", "tma_scales"] {
                    let mut shape: BTreeMap<_, _> = [
                        ("n".into(), q.n),
                        ("k".into(), q.k),
                        ("groups".into(), q.groups),
                    ]
                    .into();
                    if site != "compact" {
                        shape.insert("ranks".into(), ranks);
                    }
                    let choice = match site {
                        "compact" => mxfp8_fold::compact::stages(&FLASH, q, rows)
                            .map(|s| ("compact", [("stages".into(), Knob::Count(s))].into())),
                        "matrix" => (ranks == 4)
                            .then(|| prefill_matrix::choice(q, rows))
                            .flatten()
                            .map(|(m, s)| {
                                (
                                    "matrix",
                                    [
                                        ("tile_rows".into(), Knob::Count(m)),
                                        ("stages".into(), Knob::Count(s)),
                                    ]
                                    .into(),
                                )
                            }),
                        _ => prefill_tma_scales::choice(&FLASH, ranks, q, rows)
                            .then(|| ("tma_scales", BTreeMap::new())),
                    };
                    visit(
                        Key {
                            op: format!("dense_prefill.{site}.prefill"),
                            weights: "mxfp8".into(),
                            shape,
                            bucket: q.m_capacity,
                            rows: Some([rows, rows]),
                        },
                        choice,
                    );
                }
            }
        }
    }
}

fn generated() -> Table {
    let mut result = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    cases(|key, choice| {
        if let Some((schedule, knobs)) = choice {
            result
                .upsert(Row {
                    key,
                    schedule: schedule.into(),
                    knobs,
                })
                .unwrap();
        }
    });
    result
}

#[test]
#[ignore = "one-time selector migration generator"]
fn generate() {
    let out = std::path::PathBuf::from(
        std::env::var_os("C9_TUNING_OUT").expect("external output directory"),
    );
    std::fs::write(out.join("dense_prefill.json"), generated().render()).unwrap();
}

#[test]
fn differential() {
    let committed = table(&FLASH, Family::DensePrefill).unwrap();
    assert_eq!(committed.render(), generated().render());
    let mut count = 0;
    cases(|key, expected| {
        let actual = committed.select(
            &key.op,
            &key.weights,
            &key.shape,
            key.bucket,
            key.rows.unwrap()[0],
        );
        assert_eq!(
            actual.map(|r| (r.schedule.as_str(), r.knobs.clone())),
            expected,
            "{key}"
        );
        count += 1;
    });
    let expected: u32 = [1, 2, 4]
        .into_iter()
        .flat_map(|ranks| mxfp8_fold::prefill_entries(&FLASH, ranks).unwrap())
        .map(|entry| 3 * entry.query.m_capacity)
        .sum();
    assert!(expected > 0);
    assert_eq!(count, expected);
    eprintln!("dense_prefill differential cases: {count}");
}
