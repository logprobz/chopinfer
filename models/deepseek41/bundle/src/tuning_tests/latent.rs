use crate::sm120::{latent_cache, latent_decode, latent_scheduled};
use kernel_build::tuning::{Key, Knob, Row, Table, Target};
use kernel_core::{
    backend::{Backend, Need},
    contract::AttentionPhase,
    latent_attention::LatentAttentionQuery,
};
use kernels_sm120::{backend::Sm120, latent::sources::Schedule};
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{Family, table},
};
use std::collections::BTreeMap;

fn cases(mut visit: impl FnMut(Key, Option<(&'static str, BTreeMap<String, Knob>)>)) {
    for ranks in [1, 2, 4] {
        for source in latent_scheduled::serving_entries(&FLASH, ranks).unwrap() {
            if source.name.ends_with("_draft_block") {
                continue;
            }
            let LatentAttentionQuery::Attention {
                phase, capacity, ..
            } = source.query
            else {
                unreachable!()
            };
            let phase_name = match phase {
                AttentionPhase::Decode => "decode",
                AttentionPhase::Verify => "verify",
                AttentionPhase::Prefill => "prefill",
            };
            for rows in 1..=capacity {
                let cache = latent_scheduled::qualified(&FLASH, ranks)
                    && latent_cache::selected(phase, rows);
                let staged = latent_decode::selection(phase).filter(|&(r, _, _, _)| {
                    r == rows && latent_scheduled::qualified(&FLASH, ranks)
                });
                for (site, choice) in [
                    (
                        "sources",
                        (source.schedule == Schedule::IndependentHeads)
                            .then(|| ("independent_heads", BTreeMap::new())),
                    ),
                    (
                        "cache",
                        cache.then(|| {
                            (
                                "selected_cache",
                                [(
                                    "head_group".into(),
                                    Knob::Count(latent_cache::head_group(phase, rows)),
                                )]
                                .into(),
                            )
                        }),
                    ),
                    (
                        "staging",
                        staged.map(|(_, keys, group, packed)| {
                            (
                                "staged",
                                [
                                    ("stage_keys".into(), Knob::Count(keys)),
                                    ("head_group".into(), Knob::Count(group)),
                                    ("packed".into(), Knob::Flag(packed)),
                                ]
                                .into(),
                            )
                        }),
                    ),
                ] {
                    visit(
                        Key {
                            op: format!("latent.{site}.{phase_name}"),
                            weights: "fp8".into(),
                            shape: [("ranks".into(), ranks)].into(),
                            bucket: capacity,
                            rows: Some([rows, rows]),
                        },
                        choice,
                    );
                }
            }
        }
        let query = model_deepseek41_contract::latent_glue::query_rope(&FLASH, ranks).unwrap();
        let backend = latent_scheduled::register_query(Sm120::new(), &FLASH, ranks);
        for rows in 1..=latent_scheduled::QUERY_ROWS {
            let selected = backend.latent_query_rope(&query, Need::rows(rows)).is_ok();
            visit(
                Key {
                    op: "latent.query.decode".into(),
                    weights: "f32".into(),
                    shape: [("ranks".into(), ranks)].into(),
                    bucket: latent_scheduled::QUERY_ROWS,
                    rows: Some([rows, rows]),
                },
                selected.then(|| ("parallel", BTreeMap::new())),
            );
        }
        for capacity in 32..=64 {
            let shape = crate::shape::Shape::new(
                &FLASH,
                bundle_runtime::geometry::Ranks::new(ranks).unwrap(),
            )
            .unwrap()
            .with_verify_rows(capacity);
            let selected =
                crate::Kernels::wide_query_rope(&shape, &query, Need::rows(capacity)).is_ok();
            for rows in 1..=capacity {
                visit(
                    Key {
                        op: "latent.query.verify".into(),
                        weights: "f32".into(),
                        shape: [("ranks".into(), ranks)].into(),
                        bucket: capacity,
                        rows: Some([rows, rows]),
                    },
                    (selected && rows >= 49).then(|| ("parallel", BTreeMap::new())),
                );
            }
        }
        // The full-capacity paired op serves all launch rows; source and KV
        // layout admission remain in its provider.
        for rows in 1..=8192 {
            visit(
                Key {
                    op: "latent.paired.prefill".into(),
                    weights: "fp8".into(),
                    shape: [("ranks".into(), ranks)].into(),
                    bucket: 8192,
                    rows: Some([rows, rows]),
                },
                latent_scheduled::paired_prefill_enabled(&FLASH, ranks, 8192)
                    .then(|| ("paired", BTreeMap::new())),
            );
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
    let flush = |mut row: Row, table: &mut Table| {
        if row.key.rows == Some([1, row.key.bucket]) {
            row.key.rows = None;
        }
        table.upsert(row).unwrap();
    };
    for (key, (schedule, knobs)) in cells {
        if let Some(row) = pending.as_mut() {
            if row.key.op == key.op
                && row.key.shape == key.shape
                && row.key.bucket == key.bucket
                && row.schedule == schedule
                && row.knobs == knobs
                && row.key.rows.unwrap()[1] + 1 == key.rows.unwrap()[0]
            {
                row.key.rows.as_mut().unwrap()[1] += 1;
                continue;
            }
        }
        if let Some(row) = pending.take() {
            flush(row, &mut result);
        }
        pending = Some(Row {
            key,
            schedule: schedule.into(),
            knobs,
        });
    }
    if let Some(row) = pending {
        flush(row, &mut result);
    }
    result
}
#[test]
#[ignore = "one-time selector migration generator"]
fn generate() {
    let out = std::path::PathBuf::from(
        std::env::var_os("C9_TUNING_OUT").expect("external output directory"),
    );
    std::fs::write(out.join("latent.json"), generated().render()).unwrap();
}
#[test]
fn differential() {
    let committed = table(&FLASH, Family::Latent).unwrap();
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
    assert!(count > 200_000);
    eprintln!("latent differential cases: {count}");
}
