use super::dense_prefill::{assert_untuned, key};
use model_deepseek41_contract::{geometry::FLASH, tuning::Family};
use std::collections::BTreeSet;

#[test]
fn pin_untuned_served_keys() {
    let mut domain = BTreeSet::new();
    for ranks in [1, 2, 4] {
        for entry in crate::sm120::f32_split::entries(&FLASH, ranks).unwrap() {
            let q = entry.query();
            for (site, phase, ranked) in [
                ("exact", "prefill", false),
                ("band", "prefill", false),
                ("blocked", "prefill", false),
                ("wide", "prefill", false),
                ("head_last", "decode", true),
                ("bounded_reuse", "verify", true),
            ] {
                let mut shape = vec![("n", q.n), ("k", q.k)];
                if ranked {
                    shape.push(("ranks", ranks));
                }
                domain.insert(key(
                    &format!("f32.{site}.{phase}"),
                    "bf16",
                    &shape,
                    q.m_capacity,
                ));
            }
        }
        for site in ["router_zero", "router_norm"] {
            domain.insert(key(
                &format!("f32.{site}.prefill"),
                "bf16",
                &[("experts", 384), ("ranks", ranks)],
                8192,
            ));
        }
        // The row tiles serve Flash TP4's head, router and compressor
        // above the decode bucket, and its head within it.
        if ranks == 4 {
            let s = &FLASH.shape;
            for n in [s.vocab / 4, s.moe.experts, s.attention.head_dim] {
                domain.insert(key(
                    "f32.rows.verify",
                    "bf16",
                    &[("n", n), ("k", s.hidden), ("ranks", ranks)],
                    model_deepseek41_contract::verify_bounds::F32_ROWS,
                ));
            }
            domain.insert(key(
                "f32.rows.decode",
                "bf16",
                &[("n", s.vocab / 4), ("k", s.hidden), ("ranks", ranks)],
                model_deepseek41_contract::serving::DECODE_BATCH,
            ));
        }
        for candidate in crate::sm120::f32_verify::entries().unwrap() {
            let q = candidate.entry().query();
            for capacity in 32..=64 {
                domain.insert(key(
                    "f32.cached.verify",
                    "bf16",
                    &[("n", q.n), ("k", q.k), ("ranks", ranks)],
                    capacity,
                ));
            }
        }
    }
    assert_eq!(domain.len(), 573);
    assert_untuned(
        Family::F32,
        domain,
        UNTUNED,
        &[
            ("tiled", &["tile"]),
            ("blocked", &["tile"]),
            ("wide", &["tile"]),
            ("last", &[]),
            ("bounded_reuse", &[]),
            ("cached", &[]),
            ("zero", &[]),
            ("certified", &[]),
            (
                "rows",
                &[
                    "features",
                    "plane_k",
                    "rows",
                    "slices",
                    "stages",
                    "warp_features",
                    "warp_rows",
                ],
            ),
        ],
    );
}

const UNTUNED: &[&str] = &[
    "f32.band.prefill bf16 k=512,n=128 buckets=8192 rows=1..256",
    "f32.band.prefill bf16 k=5120,n=128 buckets=8192 rows=all",
    "f32.band.prefill bf16 k=5120,n=129280 buckets=8192 rows=all",
    "f32.band.prefill bf16 k=5120,n=32 buckets=8192 rows=1..256",
    "f32.band.prefill bf16 k=5120,n=32320 buckets=8192 rows=all",
    "f32.band.prefill bf16 k=5120,n=384 buckets=8192 rows=1..256,320..8192",
    "f32.band.prefill bf16 k=5120,n=512 buckets=8192 rows=1..256",
    "f32.band.prefill bf16 k=5120,n=64640 buckets=8192 rows=all",
    "f32.blocked.prefill bf16 k=512,n=128 buckets=8192 rows=1..2047,2049..8190",
    "f32.blocked.prefill bf16 k=5120,n=128 buckets=8192 rows=1..511,513..2047,2049..8190",
    "f32.blocked.prefill bf16 k=5120,n=129280 buckets=8192 rows=all",
    "f32.blocked.prefill bf16 k=5120,n=32 buckets=8192 rows=1..447,449..511,513..959,961..1471,1473..1983,1985..2047,2049..2495,2497..8190",
    "f32.blocked.prefill bf16 k=5120,n=32320 buckets=8192 rows=all",
    "f32.blocked.prefill bf16 k=5120,n=384 buckets=8192 rows=1..319,321..447,449..511,513..575,577..703,705..831,833..959,961..1087,1089..1215,1217..1343,1345..1471,1473..1599,1601..1727,1729..1855,1857..1983,1985..2047,2049..2111,2113..2239,2241..2367,2369..2495,2497..4095,4097..8190",
    "f32.blocked.prefill bf16 k=5120,n=512 buckets=8192 rows=1..511,513..2047,2049..4095,4097..8190",
    "f32.blocked.prefill bf16 k=5120,n=64640 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=512,n=128,ranks=1 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=512,n=128,ranks=2 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=512,n=128,ranks=4 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=128,ranks=1 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=128,ranks=2 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=128,ranks=4 buckets=8192 rows=1..34,36..39,41..44,46..49,51..54,56..59,61..8192",
    "f32.bounded_reuse.verify bf16 k=5120,n=129280,ranks=1 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=32,ranks=1 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=32,ranks=2 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=32,ranks=4 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=32320,ranks=4 buckets=8192 rows=1..34,36..39,41..44,46..49,51..54,56..59,61..8192",
    "f32.bounded_reuse.verify bf16 k=5120,n=384,ranks=1 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=384,ranks=2 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=384,ranks=4 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=512,ranks=1 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=512,ranks=2 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=512,ranks=4 buckets=8192 rows=all",
    "f32.bounded_reuse.verify bf16 k=5120,n=64640,ranks=2 buckets=8192 rows=all",
    "f32.cached.verify bf16 k=512,n=128,ranks=1 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=512,n=128,ranks=2 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=512,n=128,ranks=4 buckets=32 rows=all",
    "f32.cached.verify bf16 k=512,n=128,ranks=4 buckets=33..64 rows=1..32",
    "f32.cached.verify bf16 k=5120,n=32,ranks=1 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=32,ranks=2 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=32,ranks=4 buckets=32 rows=all",
    "f32.cached.verify bf16 k=5120,n=32,ranks=4 buckets=33..64 rows=1..32",
    "f32.cached.verify bf16 k=5120,n=32320,ranks=1 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=32320,ranks=2 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=32320,ranks=4 buckets=32 rows=all",
    "f32.cached.verify bf16 k=5120,n=32320,ranks=4 buckets=33..64 rows=1..32",
    "f32.cached.verify bf16 k=5120,n=384,ranks=1 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=384,ranks=2 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=384,ranks=4 buckets=32 rows=all",
    "f32.cached.verify bf16 k=5120,n=384,ranks=4 buckets=33..64 rows=1..32",
    "f32.cached.verify bf16 k=5120,n=512,ranks=1 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=512,ranks=2 buckets=32..64 rows=all",
    "f32.cached.verify bf16 k=5120,n=512,ranks=4 buckets=32 rows=all",
    "f32.cached.verify bf16 k=5120,n=512,ranks=4 buckets=33..64 rows=1..32",
    "f32.exact.prefill bf16 k=512,n=128 buckets=8192 rows=1..256,258..510,514..2046,2050..4094,4097..8190",
    "f32.exact.prefill bf16 k=5120,n=128 buckets=8192 rows=1..256,258..510,514..2046,2050..4094,4097..8190",
    "f32.exact.prefill bf16 k=5120,n=129280 buckets=8192 rows=all",
    "f32.exact.prefill bf16 k=5120,n=32 buckets=8192 rows=1..256,258..510,514..2046,2050..4094,4097..8190",
    "f32.exact.prefill bf16 k=5120,n=32320 buckets=8192 rows=all",
    "f32.exact.prefill bf16 k=5120,n=384 buckets=8192 rows=1..256,258..510,514..2046,2050..4094,4097..8190",
    "f32.exact.prefill bf16 k=5120,n=512 buckets=8192 rows=1..256,258..510,514..2046,2050..4094,4097..8190",
    "f32.exact.prefill bf16 k=5120,n=64640 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=512,n=128,ranks=1 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=512,n=128,ranks=2 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=512,n=128,ranks=4 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=128,ranks=1 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=128,ranks=2 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=128,ranks=4 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=129280,ranks=1 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=32,ranks=1 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=32,ranks=2 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=32,ranks=4 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=32320,ranks=4 buckets=8192 rows=2..7,9..31,33..8192",
    "f32.head_last.decode bf16 k=5120,n=384,ranks=1 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=384,ranks=2 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=384,ranks=4 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=512,ranks=1 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=512,ranks=2 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=512,ranks=4 buckets=8192 rows=all",
    "f32.head_last.decode bf16 k=5120,n=64640,ranks=2 buckets=8192 rows=all",
    "f32.router_norm.prefill bf16 experts=384,ranks=1 buckets=8192 rows=all",
    "f32.router_norm.prefill bf16 experts=384,ranks=2 buckets=8192 rows=all",
    "f32.router_norm.prefill bf16 experts=384,ranks=4 buckets=8192 rows=1..8191",
    "f32.router_zero.prefill bf16 experts=384,ranks=1 buckets=8192 rows=all",
    "f32.router_zero.prefill bf16 experts=384,ranks=2 buckets=8192 rows=all",
    "f32.router_zero.prefill bf16 experts=384,ranks=4 buckets=8192 rows=1..319",
    "f32.rows.verify bf16 k=5120,n=32320,ranks=4 buckets=192 rows=1..32",
    "f32.rows.verify bf16 k=5120,n=384,ranks=4 buckets=192 rows=1..64",
    "f32.rows.verify bf16 k=5120,n=512,ranks=4 buckets=192 rows=1..32",
    "f32.wide.prefill bf16 k=512,n=128 buckets=8192 rows=all",
    "f32.wide.prefill bf16 k=5120,n=128 buckets=8192 rows=1..8191",
    "f32.wide.prefill bf16 k=5120,n=129280 buckets=8192 rows=all",
    "f32.wide.prefill bf16 k=5120,n=32 buckets=8192 rows=all",
    "f32.wide.prefill bf16 k=5120,n=32320 buckets=8192 rows=all",
    "f32.wide.prefill bf16 k=5120,n=384 buckets=8192 rows=1..8191",
    "f32.wide.prefill bf16 k=5120,n=512 buckets=8192 rows=1..8191",
    "f32.wide.prefill bf16 k=5120,n=64640 buckets=8192 rows=all",
];
