use crate::sm120::{dspark_projection, f32_prefill, f32_split, head_last};
use kernel_build::tuning::{Key, Knob, Row, Table, Target};
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};
use std::collections::BTreeMap;

fn cases(mut visit: impl FnMut(Key, Option<(&'static str, u32)>)) {
    for ranks in [1, 2, 4] {
        for entry in f32_split::entries(&FLASH, ranks).unwrap() {
            let q = entry.query();
            let prefill = f32_prefill::serving_entry(&FLASH, entry.clone()).unwrap();
            let last = head_last::head(&FLASH, ranks, entry.clone()).unwrap();
            let reuse = dspark_projection::route(&FLASH, ranks, entry).unwrap();
            for rows in 1..=q.m_capacity {
                let exact = f32_prefill::BUCKETS
                    .iter()
                    .find(|b| (b.n, b.k, b.rows) == (q.n, q.k, rows))
                    .map(|b| ("tiled", b.features));
                let band = prefill
                    .prefill_bands()
                    .iter()
                    .find(|b| (b.lo..=b.hi).contains(&rows))
                    .map(|b| ("tiled", b.features));
                let blocked = prefill
                    .blocked_prefill_buckets()
                    .iter()
                    .find(|&&(r, _)| r == rows)
                    .map(|&(_, f)| ("blocked", f));
                let wide = prefill
                    .wide_prefill_buckets()
                    .iter()
                    .find(|&&(r, _)| r == rows)
                    .map(|&(_, f)| ("wide", f));
                for (site, phase, choice, ranked) in [
                    ("exact", "prefill", exact, false),
                    ("band", "prefill", band, false),
                    ("blocked", "prefill", blocked, false),
                    ("wide", "prefill", wide, false),
                    (
                        "head_last",
                        "decode",
                        last.decode_last(rows).map(|_| ("last", 0)),
                        true,
                    ),
                    (
                        "bounded_reuse",
                        "verify",
                        reuse
                            .uses_bounded_reuse(rows)
                            .then_some(("bounded_reuse", 0)),
                        true,
                    ),
                ] {
                    let mut shape: BTreeMap<_, _> = [("n".into(), q.n), ("k".into(), q.k)].into();
                    if ranked {
                        shape.insert("ranks".into(), ranks);
                    }
                    visit(
                        Key {
                            op: format!("f32.{site}.{phase}"),
                            weights: "bf16".into(),
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
    for ranks in [1, 2, 4] {
        for rows in 1..=8192 {
            let shape: BTreeMap<String, u32> =
                [("experts".into(), 384), ("ranks".into(), ranks)].into();
            for (site, selected) in [
                (
                    "router_zero",
                    crate::sm120::f32_zero::selected(&FLASH, ranks, 384)
                        && (crate::sm120::f32_zero::ROWS[0]..=8192).contains(&rows),
                ),
                (
                    "router_norm",
                    crate::sm120::router_norm::applies(&FLASH, ranks, rows),
                ),
            ] {
                visit(
                    Key {
                        op: format!("f32.{site}.prefill"),
                        weights: "bf16".into(),
                        shape: shape.clone(),
                        bucket: 8192,
                        rows: Some([rows, rows]),
                    },
                    selected.then_some((
                        if site == "router_zero" {
                            "zero"
                        } else {
                            "certified"
                        },
                        0,
                    )),
                );
            }
        }
        for capacity in 32..=64 {
            let shape = crate::shape::Shape::new(
                &FLASH,
                bundle_runtime::geometry::Ranks::new(ranks).unwrap(),
            )
            .unwrap()
            .with_verify_rows(capacity);
            for candidate in crate::sm120::f32_verify::entries().unwrap() {
                let q = candidate.entry().query();
                let wide = crate::ops::exact_f32::cached::prepare(
                    &shape,
                    &crate::ops::Asks::default(),
                    "tuning",
                    q.n,
                    q.k,
                    true,
                )
                .unwrap();
                for rows in 1..=capacity {
                    let selected = wide.as_ref().is_some_and(|wide| {
                        wide.for_phase(crate::program::Phase::Verify { rows, items: 1 })
                    });
                    visit(
                        Key {
                            op: "f32.cached.verify".into(),
                            weights: "bf16".into(),
                            shape: [
                                ("n".into(), q.n),
                                ("k".into(), q.k),
                                ("ranks".into(), ranks),
                            ]
                            .into(),
                            bucket: capacity,
                            rows: Some([rows, rows]),
                        },
                        selected.then_some(("cached", 0)),
                    );
                }
            }
        }
    }
}

fn generated() -> Table {
    let mut cells = BTreeMap::new();
    cases(|key, choice| {
        if let Some(choice) = choice {
            cells.insert(key, choice);
        }
    });
    let mut result = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    let mut pending: Option<Row> = None;
    for (key, (schedule, tile)) in cells {
        let knobs = if tile == 0 {
            BTreeMap::new()
        } else {
            [("tile".into(), Knob::Count(tile))].into()
        };
        if let Some(row) = pending.as_mut() {
            let [_, hi] = row.key.rows.unwrap();
            if row.key.op == key.op
                && row.key.shape == key.shape
                && row.key.bucket == key.bucket
                && row.schedule == schedule
                && row.knobs == knobs
                && hi + 1 == key.rows.unwrap()[0]
            {
                row.key.rows.as_mut().unwrap()[1] += 1;
                continue;
            }
        }
        if let Some(row) = pending.take() {
            result.upsert(row).unwrap();
        }
        pending = Some(Row {
            key,
            schedule: schedule.into(),
            knobs,
        });
    }
    if let Some(row) = pending {
        result.upsert(row).unwrap();
    }
    result
}

#[test]
#[ignore = "one-time selector migration generator"]
fn generate() {
    let out = std::path::PathBuf::from(
        std::env::var_os("C9_TUNING_OUT").expect("external output directory"),
    );
    std::fs::write(out.join("f32.json"), generated().render()).unwrap();
}

#[test]
fn differential() {
    let committed = table(&FLASH, Family::F32).unwrap();
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
            actual.map(|r| (
                r.schedule.as_str(),
                r.knobs.get("tile").and_then(|k| k.count()).unwrap_or(0)
            )),
            expected,
            "{key}"
        );
        count += 1;
    });
    assert!(count > 800_000);
    eprintln!("f32 differential cases: {count}");
}
