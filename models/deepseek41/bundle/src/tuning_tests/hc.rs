//! Served query domains and the keys that retain their baseline schedule.
use kernel_build::tuning::Key;
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{self, Family},
};

fn add(
    out: &mut std::collections::BTreeSet<Key>,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
) {
    out.insert(Key {
        op: op.into(),
        weights: weights.into(),
        shape: shape.iter().map(|&(k, v)| (k.into(), v)).collect(),
        bucket,
        rows: None,
    });
}
fn domains() -> std::collections::BTreeSet<Key> {
    use model_deepseek41_contract::{
        admission::SUPPORTED_RANKS,
        dspark_options::SMALL_DRAFT_ROWS,
        serving::{DECODE_BATCH, PREFILL_TOKENS, VERIFY_DECODE_ROWS},
    };
    let mut out = std::collections::BTreeSet::new();
    let shape = [
        ("groups", FLASH.shape.hc.streams),
        ("width", FLASH.shape.hidden),
    ];
    add(
        &mut out,
        "hc.lagged_warp.decode",
        "f32",
        &shape,
        DECODE_BATCH,
    );
    for &ranks in SUPPORTED_RANKS {
        add(
            &mut out,
            "hc.round_chain.decode",
            "f32",
            &[
                ("groups", FLASH.shape.hc.streams),
                ("ranks", ranks),
                ("width", FLASH.shape.hidden),
            ],
            VERIFY_DECODE_ROWS,
        );
    }
    for site in [
        "finish_pair",
        "finish_chain",
        "finish_rows",
        "finish_twin",
        "finish_cols",
    ] {
        add(
            &mut out,
            &format!("hc.{site}.verify"),
            "f32",
            &shape,
            VERIFY_DECODE_ROWS,
        );
    }
    for site in ["lagged_mix", "expand_mix"] {
        add(
            &mut out,
            &format!("hc.{site}.prefill"),
            "f32",
            &shape,
            PREFILL_TOKENS,
        );
    }
    for phase in ["decode", "verify", "prefill"] {
        for capacity in DECODE_BATCH..=VERIFY_DECODE_ROWS {
            for site in ["finish_admit", "wide_mix"] {
                add(
                    &mut out,
                    &format!("hc.{site}.{phase}"),
                    "f32",
                    &shape,
                    capacity,
                );
            }
        }
        add(
            &mut out,
            &format!("hc.decode_mix.{phase}"),
            "f32",
            &shape,
            DECODE_BATCH,
        );
        add(
            &mut out,
            &format!("hc.draft_mix.{phase}"),
            "f32",
            &shape,
            SMALL_DRAFT_ROWS,
        );
    }
    out
}

const UNTUNED: &[&str] = &[
    "hc.decode_mix.decode f32 groups=4,width=5120 t32: 2..=7,9..=32",
    "hc.decode_mix.prefill f32 groups=4,width=5120 t32: 1..=32",
    "hc.decode_mix.verify f32 groups=4,width=5120 t32: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.draft_mix.decode f32 groups=4,width=5120 t60: 1..=32",
    "hc.draft_mix.prefill f32 groups=4,width=5120 t60: 1..=32",
    "hc.draft_mix.verify f32 groups=4,width=5120 t60: 1..=32",
    "hc.expand_mix.prefill f32 groups=4,width=5120 t8192: 1..=63,65..=319,321..=8191",
    "hc.finish_admit.decode f32 groups=4,width=5120 t32: 2..=7,9..=15,17..=31",
    "hc.finish_admit.decode f32 groups=4,width=5120 t33: 2..=7,9..=15,17..=31,33",
    "hc.finish_admit.decode f32 groups=4,width=5120 t34: 2..=7,9..=15,17..=31,33..=34",
    "hc.finish_admit.decode f32 groups=4,width=5120 t35: 2..=7,9..=15,17..=31,33..=35",
    "hc.finish_admit.decode f32 groups=4,width=5120 t36: 2..=7,9..=15,17..=31,33..=36",
    "hc.finish_admit.decode f32 groups=4,width=5120 t37: 2..=7,9..=15,17..=31,33..=37",
    "hc.finish_admit.decode f32 groups=4,width=5120 t38: 2..=7,9..=15,17..=31,33..=38",
    "hc.finish_admit.decode f32 groups=4,width=5120 t39: 2..=7,9..=15,17..=31,33..=39",
    "hc.finish_admit.decode f32 groups=4,width=5120 t40: 2..=7,9..=15,17..=31,33..=40",
    "hc.finish_admit.decode f32 groups=4,width=5120 t41: 2..=7,9..=15,17..=31,33..=41",
    "hc.finish_admit.decode f32 groups=4,width=5120 t42: 2..=7,9..=15,17..=31,33..=42",
    "hc.finish_admit.decode f32 groups=4,width=5120 t43: 2..=7,9..=15,17..=31,33..=43",
    "hc.finish_admit.decode f32 groups=4,width=5120 t44: 2..=7,9..=15,17..=31,33..=44",
    "hc.finish_admit.decode f32 groups=4,width=5120 t45: 2..=7,9..=15,17..=31,33..=45",
    "hc.finish_admit.decode f32 groups=4,width=5120 t46: 2..=7,9..=15,17..=31,33..=46",
    "hc.finish_admit.decode f32 groups=4,width=5120 t47: 2..=7,9..=15,17..=31,33..=47",
    "hc.finish_admit.decode f32 groups=4,width=5120 t48: 2..=7,9..=15,17..=31,33..=48",
    "hc.finish_admit.decode f32 groups=4,width=5120 t49: 2..=7,9..=15,17..=31,33..=49",
    "hc.finish_admit.decode f32 groups=4,width=5120 t50: 2..=7,9..=15,17..=31,33..=50",
    "hc.finish_admit.decode f32 groups=4,width=5120 t51: 2..=7,9..=15,17..=31,33..=51",
    "hc.finish_admit.decode f32 groups=4,width=5120 t52: 2..=7,9..=15,17..=31,33..=52",
    "hc.finish_admit.decode f32 groups=4,width=5120 t53: 2..=7,9..=15,17..=31,33..=53",
    "hc.finish_admit.decode f32 groups=4,width=5120 t54: 2..=7,9..=15,17..=31,33..=54",
    "hc.finish_admit.decode f32 groups=4,width=5120 t55: 2..=7,9..=15,17..=31,33..=55",
    "hc.finish_admit.decode f32 groups=4,width=5120 t56: 2..=7,9..=15,17..=31,33..=56",
    "hc.finish_admit.decode f32 groups=4,width=5120 t57: 2..=7,9..=15,17..=31,33..=57",
    "hc.finish_admit.decode f32 groups=4,width=5120 t58: 2..=7,9..=15,17..=31,33..=58",
    "hc.finish_admit.decode f32 groups=4,width=5120 t59: 2..=7,9..=15,17..=31,33..=59",
    "hc.finish_admit.decode f32 groups=4,width=5120 t60: 2..=7,9..=15,17..=31,33..=60",
    "hc.finish_admit.decode f32 groups=4,width=5120 t61: 2..=7,9..=15,17..=31,33..=61",
    "hc.finish_admit.decode f32 groups=4,width=5120 t62: 2..=7,9..=15,17..=31,33..=62",
    "hc.finish_admit.decode f32 groups=4,width=5120 t63: 2..=7,9..=15,17..=31,33..=63",
    "hc.finish_admit.decode f32 groups=4,width=5120 t64: 2..=7,9..=15,17..=31,33..=64",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t32: 1..=32",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t33: 1..=33",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t34: 1..=34",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t35: 1..=35",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t36: 1..=36",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t37: 1..=37",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t38: 1..=38",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t39: 1..=39",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t40: 1..=40",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t41: 1..=41",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t42: 1..=42",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t43: 1..=43",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t44: 1..=44",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t45: 1..=45",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t46: 1..=46",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t47: 1..=47",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t48: 1..=48",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t49: 1..=49",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t50: 1..=50",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t51: 1..=51",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t52: 1..=52",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t53: 1..=53",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t54: 1..=54",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t55: 1..=55",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t56: 1..=56",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t57: 1..=57",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t58: 1..=58",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t59: 1..=59",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t60: 1..=60",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t61: 1..=61",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t62: 1..=62",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t63: 1..=63",
    "hc.finish_admit.prefill f32 groups=4,width=5120 t64: 1..=64",
    "hc.finish_admit.verify f32 groups=4,width=5120 t32: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t33: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t34: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t35: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t36: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t37: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t38: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t39: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t40: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t41: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t42: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t43: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t44: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t45: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t46: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t47: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t48: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t49: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t50: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t51: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t52: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t53: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t54: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t55: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t56: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t57: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t58: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t59: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t60: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t61: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t62: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t63: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_admit.verify f32 groups=4,width=5120 t64: 1..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_chain.verify f32 groups=4,width=5120 t64: 2..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_cols.verify f32 groups=4,width=5120 t64: 2..=7,9..=15,17..=31",
    "hc.finish_pair.verify f32 groups=4,width=5120 t64: 2..=3,5,7,9..=11,13..=15,17..=31",
    "hc.finish_rows.verify f32 groups=4,width=5120 t64: 1..=31,33..=64",
    "hc.finish_twin.verify f32 groups=4,width=5120 t64: 1..=5,7..=64",
    "hc.lagged_mix.prefill f32 groups=4,width=5120 t8192: 1..=63,65..=191,193..=2239,2241..=2495,2497..=2623,2625..=8191",
    "hc.lagged_warp.decode f32 groups=4,width=5120 t32: 2..=3,5,7,9..=11,13..=15,17..=31",
    "hc.round_chain.decode f32 groups=4,ranks=1,width=5120 t64: 2..=7,9..=64",
    "hc.round_chain.decode f32 groups=4,ranks=2,width=5120 t64: 2..=7,9..=64",
    "hc.round_chain.decode f32 groups=4,ranks=4,width=5120 t64: 9..=64",
    "hc.wide_mix.decode f32 groups=4,width=5120 t32: 1..=32",
    "hc.wide_mix.decode f32 groups=4,width=5120 t33: 1..=33",
    "hc.wide_mix.decode f32 groups=4,width=5120 t34: 1..=34",
    "hc.wide_mix.decode f32 groups=4,width=5120 t35: 1..=35",
    "hc.wide_mix.decode f32 groups=4,width=5120 t36: 1..=36",
    "hc.wide_mix.decode f32 groups=4,width=5120 t37: 1..=37",
    "hc.wide_mix.decode f32 groups=4,width=5120 t38: 1..=38",
    "hc.wide_mix.decode f32 groups=4,width=5120 t39: 1..=39",
    "hc.wide_mix.decode f32 groups=4,width=5120 t40: 1..=40",
    "hc.wide_mix.decode f32 groups=4,width=5120 t41: 1..=41",
    "hc.wide_mix.decode f32 groups=4,width=5120 t42: 1..=42",
    "hc.wide_mix.decode f32 groups=4,width=5120 t43: 1..=43",
    "hc.wide_mix.decode f32 groups=4,width=5120 t44: 1..=44",
    "hc.wide_mix.decode f32 groups=4,width=5120 t45: 1..=45",
    "hc.wide_mix.decode f32 groups=4,width=5120 t46: 1..=46",
    "hc.wide_mix.decode f32 groups=4,width=5120 t47: 1..=47",
    "hc.wide_mix.decode f32 groups=4,width=5120 t48: 1..=48",
    "hc.wide_mix.decode f32 groups=4,width=5120 t49: 1..=49",
    "hc.wide_mix.decode f32 groups=4,width=5120 t50: 1..=50",
    "hc.wide_mix.decode f32 groups=4,width=5120 t51: 1..=51",
    "hc.wide_mix.decode f32 groups=4,width=5120 t52: 1..=52",
    "hc.wide_mix.decode f32 groups=4,width=5120 t53: 1..=53",
    "hc.wide_mix.decode f32 groups=4,width=5120 t54: 1..=54",
    "hc.wide_mix.decode f32 groups=4,width=5120 t55: 1..=55",
    "hc.wide_mix.decode f32 groups=4,width=5120 t56: 1..=56",
    "hc.wide_mix.decode f32 groups=4,width=5120 t57: 1..=57",
    "hc.wide_mix.decode f32 groups=4,width=5120 t58: 1..=58",
    "hc.wide_mix.decode f32 groups=4,width=5120 t59: 1..=59",
    "hc.wide_mix.decode f32 groups=4,width=5120 t60: 1..=60",
    "hc.wide_mix.decode f32 groups=4,width=5120 t61: 1..=61",
    "hc.wide_mix.decode f32 groups=4,width=5120 t62: 1..=62",
    "hc.wide_mix.decode f32 groups=4,width=5120 t63: 1..=63",
    "hc.wide_mix.decode f32 groups=4,width=5120 t64: 1..=64",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t32: 1..=32",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t33: 1..=33",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t34: 1..=34",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t35: 1..=35",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t36: 1..=36",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t37: 1..=37",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t38: 1..=38",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t39: 1..=39",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t40: 1..=40",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t41: 1..=41",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t42: 1..=42",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t43: 1..=43",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t44: 1..=44",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t45: 1..=45",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t46: 1..=46",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t47: 1..=47",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t48: 1..=48",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t49: 1..=49",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t50: 1..=50",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t51: 1..=51",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t52: 1..=52",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t53: 1..=53",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t54: 1..=54",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t55: 1..=55",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t56: 1..=56",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t57: 1..=57",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t58: 1..=58",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t59: 1..=59",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t60: 1..=60",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t61: 1..=61",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t62: 1..=62",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t63: 1..=63",
    "hc.wide_mix.prefill f32 groups=4,width=5120 t64: 1..=64",
    "hc.wide_mix.verify f32 groups=4,width=5120 t32: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t33: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t34: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t35: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t36: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t37: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t38: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t39: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t40: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t41: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t42: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t43: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t44: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t45: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t46: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t47: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t48: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t49: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t50: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t51: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t52: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t53: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t54: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t55: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t56: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t57: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t58: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t59: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t60: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t61: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t62: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t63: 1..=32",
    "hc.wide_mix.verify f32 groups=4,width=5120 t64: 1..=32",
];

#[test]
fn pin_untuned_served_keys() {
    let family = Family::Hc;
    let table = tuning::table(&FLASH, family).unwrap();
    assert!(!table.rows().is_empty());
    let domains = domains();
    let mut missing = Vec::new();
    let mut cases = 0;
    for key in &domains {
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for rows in 1..=key.bucket {
            if table
                .select(&key.op, &key.weights, &key.shape, key.bucket, rows)
                .is_none()
            {
                if let Some((_, hi)) = ranges.last_mut()
                    && *hi + 1 == rows
                {
                    *hi = rows;
                } else {
                    ranges.push((rows, rows));
                }
            }
            cases += 1;
        }
        assert!(
            table
                .select(&key.op, &key.weights, &key.shape, key.bucket, 0)
                .is_none()
        );
        assert!(
            table
                .select(
                    &key.op,
                    &key.weights,
                    &key.shape,
                    key.bucket,
                    key.bucket + 1
                )
                .is_none()
        );
        if !ranges.is_empty() {
            let rows = ranges
                .into_iter()
                .map(|(lo, hi)| {
                    if lo == hi {
                        lo.to_string()
                    } else {
                        format!("{lo}..={hi}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            missing.push(format!("{key}: {rows}"));
        }
    }
    assert_eq!(missing, UNTUNED, "untuned served keys changed");
    for row in table.rows() {
        assert!(
            domains.contains(&Key {
                rows: None,
                ..row.key.clone()
            }),
            "table row is outside the pinned served domains: {}",
            row.key
        );
    }
    assert!(cases > 0);
    eprintln!(
        "{}: {cases} served lookup cases, {} untuned keys",
        family.name(),
        missing.len()
    );
}
