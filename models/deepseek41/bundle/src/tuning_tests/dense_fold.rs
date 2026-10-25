use crate::sm120::{
    fold_cluster, fold_fused, fold_tails, fold_verify, mxfp8_fold, mxfp8_fold_direct,
    mxfp8_fold_stream,
};
use kernel_build::tuning::{Key, Knob, Knobs, Row, Table, Target};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{self, Family},
};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Choice(pub String, pub Knobs);
pub(super) fn choice(schedule: &str, knobs: &[(&str, u32)]) -> Choice {
    Choice(
        schedule.into(),
        knobs
            .iter()
            .map(|&(k, v)| (k.into(), Knob::Count(v)))
            .collect(),
    )
}
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
pub(super) struct Case {
    pub key: Key,
    pub rows: u32,
    pub expected: Option<Choice>,
}
pub(super) fn case(
    op: &str,
    weights: &str,
    shape: BTreeMap<String, u32>,
    bucket: u32,
    rows: u32,
    expected: Option<Choice>,
) -> Case {
    Case {
        key: Key {
            op: op.into(),
            weights: weights.into(),
            shape,
            bucket,
            rows: None,
        },
        rows,
        expected,
    }
}
fn status(op: &str, bucket: u32, rows: u32, expected: Option<Choice>) -> Case {
    case(
        op,
        "mxfp8",
        [("ranks".into(), 4)].into(),
        bucket,
        rows,
        expected,
    )
}
fn cluster_choice(e: kernels_sm120::gemm::mxfp8_fold_cluster::Entry) -> Choice {
    choice(
        "cluster",
        &[
            ("tile_n", e.tile_n),
            ("splits", e.splits),
            ("stages", e.stages),
        ],
    )
}
fn cases() -> Vec<Case> {
    let mut result = Vec::new();
    // These registration sites serve decode and verify, and the TP4 short
    // prefill chain reuses their exact-row offers. Enumerate each caller.
    for ranks in [1, 2, 4] {
        for phase in ["decode", "verify", "prefill"] {
            for registration in mxfp8_fold::decode_registrations(&FLASH, ranks, 0).unwrap() {
                if registration.usage.capacity != 32 {
                    continue;
                }
                let q = registration.entry.query;
                let paired = registration.entry.paired;
                for rows in 1..=64 {
                    let _caller = (&registration.projection, ranks, phase, rows);
                    let selected = match mxfp8_fold::decode_path(&FLASH, q, rows, paired) {
                        mxfp8_fold::DecodePath::Split => Some(choice("split", &[])),
                        mxfp8_fold::DecodePath::Prefetch => Some(choice("prefetch", &[])),
                        mxfp8_fold::DecodePath::Ticket => None,
                    };
                    result.push(case(
                        "dense_fold.fold.decode",
                        "mxfp8",
                        fold_shape(q, paired),
                        64,
                        rows,
                        selected,
                    ));
                    result.push(case(
                        "dense_fold.split.decode",
                        "mxfp8",
                        fold_shape(q, false),
                        64,
                        rows,
                        (mxfp8_fold::small_query(&FLASH, q, rows).m_capacity == 32)
                            .then(|| choice("split", &[])),
                    ));
                    result.push(case(
                        "dense_fold.stream.decode",
                        "mxfp8",
                        fold_shape(q, paired),
                        64,
                        rows,
                        mxfp8_fold_stream::qualified(&FLASH, q, rows, paired)
                            .then(|| choice("stream", &[])),
                    ));
                    result.push(case(
                        "dense_fold.direct.decode",
                        "mxfp8",
                        fold_shape(q, paired),
                        64,
                        rows,
                        mxfp8_fold_direct::selected(&FLASH, q, rows, paired)
                            .map(|s| choice(s.name(), &[])),
                    ));
                }
            }
        }
    }
    let shape = crate::shape::Shape::new(&FLASH, bundle_runtime::geometry::Ranks::TP4).unwrap();
    let wide_shape = shape.with_verify_rows(64);
    let wide_query = crate::ops::dense::projections(&shape)
        .into_iter()
        .find(|r| r.id == "attention.q-b")
        .unwrap()
        .query;
    for rows in 1..=64 {
        let wide = rows > 32
            && crate::Kernels::prepare_local_fold(
                &wide_shape,
                "attention.q-b",
                mxfp8_fold::routed_query(wide_query, rows),
                kernel_core::backend::Need::rows(rows),
            )
            .unwrap()
            .is_some();
        result.push(status(
            "dense_fold.status_wide.verify",
            64,
            rows,
            wide.then(|| choice("local", &[])),
        ));
        if rows <= 32 {
            result.push(status(
                "dense_fold.status.decode",
                32,
                rows,
                fold_fused::tile(rows).map(|tile| {
                    let q = fold_fused::sliced(rows).unwrap();
                    let w = fold_fused::wide(rows).unwrap();
                    choice(
                        "status",
                        &[
                            ("tile_rows", tile),
                            ("slices", q.slices),
                            ("warps", q.warps),
                            ("tile_n", w.tile_n),
                            ("stages", w.stages),
                        ],
                    )
                }),
            ));
        }
        result.push(status(
            "dense_fold.status_tails.decode",
            64,
            rows,
            fold_tails::tile(rows).map(|tile| choice("status", &[("tile_rows", tile)])),
        ));
        result.push(status(
            "dense_fold.status_wide_tails.decode",
            64,
            rows,
            fold_tails::wide_rows(rows).then(|| choice("local", &[])),
        ));
        result.push(status(
            "dense_fold.status_blocks.decode",
            64,
            rows,
            crate::Kernels::drafter_status_fold(&shape, "attention.q-a", rows)
                .then(|| choice("status", &[])),
        ));
        for id in fold_verify::IDS {
            let site = id.replace(['.', '-'], "_");
            if rows <= 32 {
                result.push(status(
                    &format!("dense_fold.{site}_cluster.decode"),
                    32,
                    rows,
                    fold_cluster::served(id, rows).map(cluster_choice),
                ));
            }
            result.push(status(
                &format!("dense_fold.{site}_status.verify"),
                64,
                rows,
                fold_verify::served(id, rows).map(|e| cluster_choice(e.cluster)),
            ));
        }
    }
    result
}

pub(super) fn render(cases: Vec<Case>) -> String {
    let mut groups: BTreeMap<Key, BTreeMap<u32, Option<Choice>>> = BTreeMap::new();
    for c in cases {
        let rows = groups.entry(c.key).or_default();
        if let Some(old) = rows.insert(c.rows, c.expected.clone()) {
            assert_eq!(old, c.expected);
        }
    }
    let mut table = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    for (key, choices) in groups {
        let mut at = 1;
        while at <= key.bucket {
            let Some(Some(selected)) = choices.get(&at) else {
                at += 1;
                continue;
            };
            let first = at;
            while at < key.bucket && choices.get(&(at + 1)) == Some(&Some(selected.clone())) {
                at += 1;
            }
            let mut key = key.clone();
            key.rows = if first == 1 && at == key.bucket {
                None
            } else {
                Some([first, at])
            };
            table
                .upsert(Row {
                    key,
                    schedule: selected.0.clone(),
                    knobs: selected.1.clone(),
                })
                .unwrap();
            at += 1;
        }
    }
    table.render()
}
pub(super) fn generate(family: Family, cases: Vec<Case>) {
    let text = render(cases);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../contract/tuning/flash/{}.json", family.name()));
    if std::env::var_os("CHOPINFER_WRITE_TUNING").is_some() {
        std::fs::write(path, text).unwrap();
    } else {
        assert_eq!(std::fs::read_to_string(path).unwrap(), text);
    }
}
#[test]
#[ignore = "one-time extraction; set CHOPINFER_WRITE_TUNING to write, otherwise check bytes"]
fn generate_dense_fold() {
    generate(Family::DenseFold, cases());
}
pub(super) fn differential(family: Family, cases: Vec<Case>) {
    let count = cases.len();
    let other = model_deepseek41_contract::geometry::Geometry {
        id: "untuned",
        ..FLASH
    };
    for c in cases {
        let shape = c
            .key
            .shape
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect::<Vec<_>>();
        let selected = tuning::select(
            &FLASH,
            family,
            &c.key.op,
            &c.key.weights,
            &shape,
            c.key.bucket,
            c.rows,
        )
        .map(|r| Choice(r.schedule.clone(), r.knobs.clone()));
        assert_eq!(selected, c.expected, "{} at {}", c.key, c.rows);
        assert!(
            tuning::select(
                &other,
                family,
                &c.key.op,
                &c.key.weights,
                &shape,
                c.key.bucket,
                c.rows
            )
            .is_none()
        );
        assert!(
            tuning::select(
                &FLASH,
                family,
                &c.key.op,
                &c.key.weights,
                &shape,
                c.key.bucket,
                0
            )
            .is_none()
        );
        assert!(
            tuning::select(
                &FLASH,
                family,
                &c.key.op,
                &c.key.weights,
                &shape,
                c.key.bucket,
                c.key.bucket + 1
            )
            .is_none()
        );
    }
    println!("{}: {count} differential cases passed", family.name());
}
#[test]
fn dense_fold_matches_current_selectors() {
    differential(Family::DenseFold, cases());
}
