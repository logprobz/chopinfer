use crate::{Kernels, program::Phase, shape::Shape};
use bundle_runtime::geometry::Ranks;
use kernel_build::tuning::{Key, Row, Table, Target};
use kernel_core::backend::Need;
use model_deepseek41_contract::{
    geometry::FLASH,
    state::KvStorage,
    tuning::{Family, table},
};
use std::collections::BTreeMap;

fn cases(mut visit: impl FnMut(Key, Vec<bool>, &'static str)) {
    for ranks in [1, 2, 4] {
        let shape = Shape::new(&FLASH, Ranks::new(ranks).unwrap()).unwrap();
        visit(
            Key {
                op: "indexer.hierarchy.decode".into(),
                weights: "fp4".into(),
                shape: [("ranks".into(), ranks)].into(),
                bucket: 32,
                rows: None,
            },
            (1..=32)
                .map(|rows| {
                    Kernels::hierarchical_indexer_enabled(
                        &shape,
                        KvStorage::Native,
                        Phase::Decode(rows),
                    )
                })
                .collect(),
            "hierarchical",
        );
        let query = model_deepseek41_contract::indexer::selection(
            &FLASH,
            FLASH.shape.csa.index_sources[0],
            8192,
        )
        .unwrap();
        for capacity in 32..=64 {
            let shape = Shape::new(&FLASH, Ranks::new(ranks).unwrap())
                .unwrap()
                .with_verify_rows(capacity);
            let enabled = Kernels::wide_index_query(&shape, &query, Need::rows(capacity)).is_ok();
            visit(
                Key {
                    op: "indexer.query.verify".into(),
                    weights: "f32".into(),
                    shape: [("ranks".into(), ranks)].into(),
                    bucket: capacity,
                    rows: None,
                },
                (1..=capacity).map(|rows| enabled && rows >= 33).collect(),
                "parallel",
            );
        }
    }
    for (compact, ratio2, site, capacity) in [
        (false, false, "candidate_scores", 512),
        (false, true, "compressed_scores", 4096),
        (true, false, "compressed_scores", 4096),
        (true, true, "compressed_scores", 4096),
    ] {
        visit(
            Key {
                op: format!("indexer.{site}.prefill"),
                weights: "fp4".into(),
                shape: [("top_k".into(), 512)].into(),
                bucket: capacity,
                rows: None,
            },
            (1..=capacity)
                .map(|rows| Kernels::prefill_tile_selected(compact, ratio2, rows))
                .collect(),
            "tiled",
        );
    }
    for (ratio, width) in [(1, 131072), (2, 65536)] {
        for capacity in 1..=4096 {
            visit(
                Key {
                    op: "indexer.cached.prefill".into(),
                    weights: "f32".into(),
                    shape: [("width".into(), width), ("top_k".into(), 512)].into(),
                    bucket: capacity,
                    rows: None,
                },
                (1..=capacity)
                    .map(|rows| Kernels::cached_prefill_indexer_enabled(ratio, width, rows, 512))
                    .collect(),
                "cached",
            );
        }
    }
}
fn generated() -> (Table, std::collections::BTreeSet<Key>) {
    let mut domain = std::collections::BTreeSet::new();
    let mut out = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    cases(|key, selected, schedule| {
        domain.insert(key.clone());
        let mut start = None;
        for (index, chosen) in selected.iter().copied().chain([false]).enumerate() {
            if chosen {
                start.get_or_insert(index as u32 + 1);
            } else if let Some(lo) = start.take() {
                let mut key = key.clone();
                key.rows = if lo == 1 && index as u32 == key.bucket {
                    None
                } else {
                    Some([lo, index as u32])
                };
                out.upsert(Row {
                    key,
                    schedule: schedule.into(),
                    knobs: BTreeMap::new(),
                })
                .unwrap();
            }
        }
    });
    (out, domain)
}
#[test]
#[ignore = "one-time selector migration generator"]
fn generate() {
    let out = std::path::PathBuf::from(
        std::env::var_os("C9_TUNING_OUT").expect("external output directory"),
    );
    let (table, domain) = generated();
    std::fs::write(out.join("indexer.json"), table.render()).unwrap();
    let domain: Vec<_> = domain.into_iter().map(|key| serde_json::json!({"op":key.op,"weights":key.weights,"shape":key.shape,"bucket":key.bucket})).collect();
    std::fs::write(
        out.join("indexer-domain.json"),
        serde_json::to_string_pretty(&domain).unwrap(),
    )
    .unwrap();
}
#[test]
fn differential() {
    let committed = table(&FLASH, Family::Indexer).unwrap();
    assert_eq!(committed.render(), generated().0.render());
    let mut count = 0;
    cases(|key, expected, schedule| {
        for (index, selected) in expected.into_iter().enumerate() {
            let rows = index as u32 + 1;
            let actual = committed.select(&key.op, &key.weights, &key.shape, key.bucket, rows);
            assert_eq!(
                actual.map(|r| r.schedule.as_str()),
                selected.then_some(schedule),
                "{key} rows={rows}"
            );
            count += 1;
        }
    });
    assert!(count > 16_000_000);
    eprintln!("indexer differential cases: {count}");
}
