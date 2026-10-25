use super::dense_fold::{Case, case, choice, fold_shape, generate};
use crate::sm120::{f32_head, mxfp8_fold, projection_reuse};
use model_deepseek41_contract::{geometry::FLASH, tuning::Family};

fn cases() -> Vec<Case> {
    let mut result = Vec::new();
    for ranks in [1, 2, 4] {
        for phase in ["decode", "verify", "prefill"] {
            for registration in mxfp8_fold::decode_registrations(&FLASH, ranks, 0).unwrap() {
                if registration.usage.capacity != 32 || registration.entry.paired {
                    continue;
                }
                let q = registration.entry.query;
                for rows in 1..=64 {
                    let _caller = (&registration.projection, ranks, phase, rows);
                    result.push(case(
                        "reuse.fold.decode",
                        "mxfp8",
                        fold_shape(q, false),
                        64,
                        rows,
                        projection_reuse::choice(q, rows).map(|(tile, folds)| {
                            choice("reuse", &[("tile_n", tile), ("folds", folds)])
                        }),
                    ));
                }
            }
            let e = projection_reuse::head(
                &FLASH,
                ranks,
                f32_head::decode_entry(&FLASH, ranks).unwrap(),
            )
            .unwrap();
            let q = e.query();
            for rows in 1..=q.m_capacity {
                result.push(case(
                    "reuse.head.decode",
                    "bf16",
                    [
                        ("n".into(), q.n),
                        ("k".into(), q.k),
                        ("ranks".into(), ranks),
                    ]
                    .into(),
                    q.m_capacity,
                    rows,
                    e.uses_decode_reuse(rows).then(|| choice("reuse", &[])),
                ));
            }
        }
    }
    result
}
#[test]
#[ignore = "one-time extraction; set CHOPINFER_WRITE_TUNING to write, otherwise check bytes"]
fn generate_reuse() {
    generate(Family::Reuse, cases());
}
