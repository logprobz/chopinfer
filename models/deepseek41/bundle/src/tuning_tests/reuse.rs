use super::dense_fold::{Coverage, assert_untuned, cover, fold_shape};
use crate::sm120::{f32_head, mxfp8_fold};
use model_deepseek41_contract::{geometry::FLASH, tuning::Family};

fn served() -> Coverage {
    let mut keys = Coverage::new();
    for ranks in [1, 2, 4] {
        for registration in mxfp8_fold::decode_registrations(&FLASH, ranks, 0).unwrap() {
            if registration.entry.paired {
                continue;
            }
            cover(
                &mut keys,
                "reuse.fold.decode",
                "mxfp8",
                fold_shape(registration.entry.query, false),
                64,
            );
        }
        let q = f32_head::query(&FLASH, ranks);
        cover(
            &mut keys,
            "reuse.head.decode",
            "bf16",
            [
                ("n".into(), q.n),
                ("k".into(), q.k),
                ("ranks".into(), ranks),
            ]
            .into(),
            q.m_capacity,
        );
    }
    keys
}
#[test]
fn pin_untuned_served_keys() {
    assert_untuned(Family::Reuse, served(), UNTUNED);
}
const UNTUNED: &str = r#"
reuse.fold.decode mxfp8 groups=1,k=576,n=5120,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=1152,n=5120,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=1280,n=4096,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=1280,n=8192,paired=0 t64: 1..=8,13..=64
reuse.fold.decode mxfp8 groups=1,k=1280,n=16384,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=1280,n=32768,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=2048,n=5120,paired=0 t64: 1..=16,24,33..=64
reuse.fold.decode mxfp8 groups=1,k=2304,n=5120,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=4096,n=5120,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=5120,n=512,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=5120,n=576,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=5120,n=1152,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=5120,n=1280,paired=0 t64: 1..=16,22,25..=64
reuse.fold.decode mxfp8 groups=1,k=5120,n=2304,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=6144,n=25600,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=8192,n=5120,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=1,k=15360,n=5120,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=2,k=4096,n=1024,paired=0 t64: 1..=16,33..=64
reuse.fold.decode mxfp8 groups=4,k=4096,n=1024,paired=0 t64: 1..=64
reuse.fold.decode mxfp8 groups=8,k=4096,n=1024,paired=0 t64: 1..=64
reuse.head.decode bf16 k=5120,n=32320,ranks=4 t8192: 1..=8,33..=8192
reuse.head.decode bf16 k=5120,n=64640,ranks=2 t8192: 1..=8192
reuse.head.decode bf16 k=5120,n=129280,ranks=1 t8192: 1..=8192
"#;
