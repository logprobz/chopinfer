//! The `phased_decode_reduce` candidate on verify steps: at every width
//! whose HC boundaries take no round every hidden all-reduce launches the
//! phased entry of its rows, and at the widths that defer the round only
//! the complete ones do.
use super::*;
use model_deepseek41_contract::execution_options::{
    DSPARK_GRAPH_HANDLES_PER_RANK, PHASED_DECODE_REDUCE,
};

const ONE_SHOT: &str = "chop_comm_allreduce_phased_oneshot_round_f32_v1";
const TWO_SHOT: &str = "chop_comm_allreduce_phased_round_f32_v1";
const SCATTER: &str = "chop_comm_reduce_scatter_f32_v1";
const ROUND: &str = "chop_row_rounding_round_deepseek41_w5120_v1";
/// Each rank's hidden all-reduces per target step, and those whose round
/// a deferring width's HC boundaries take.
const REDUCTIONS: usize = 81;
const DEFERRED: usize = 77;
/// (items, rows, whether the width defers the round): complete widths in
/// the one-shot's rows, the two-shot's and the two-shot with workers, and
/// deferring ones (`decode-reductions.md`).
const STEPS: [(u32, u32, bool); 9] = [
    (1, 2, false),
    (1, 3, false),
    (1, 4, true),
    (1, 5, false),
    (4, 24, false),
    (8, 31, false),
    (8, 33, true),
    (16, 64, true),
    (32, 128, false),
];

#[test]
fn verify_reductions_take_the_phased_entries_where_complete() {
    for enabled in [false, true] {
        let (host, mut instance) = create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            32,
            192,
            &[
                (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
                (PHASED_DECODE_REDUCE, enabled.into()),
            ],
        );
        for slot in 0..32 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        for (count, rows, deferred) in STEPS {
            let (target, _, accepted) =
                super::verify_rows::verify_step(&mut instance, &host, count, rows);
            let launched = |symbol: &str| {
                target
                    .iter()
                    .filter(|line| line.split_whitespace().nth(2) == Some(symbol))
                    .count()
            };
            let complete = 4 * if deferred {
                REDUCTIONS - DEFERRED
            } else {
                REDUCTIONS
            };
            let phased = if enabled { complete } else { 0 };
            let (one_shot, two_shot) = if rows <= 4 { (phased, 0) } else { (0, phased) };
            let case = format!("{count} items, {rows} rows, option {enabled}");
            assert_eq!(launched(ONE_SHOT), one_shot, "{case}");
            assert_eq!(launched(TWO_SHOT), two_shot, "{case}");
            assert_eq!(launched(SCATTER), 4 * REDUCTIONS - phased, "{case}");
            assert_eq!(launched(ROUND), complete - phased, "{case}");
            instance.commit_accepted(&accepted).unwrap();
        }
        for request in 1..=32 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// With `wide_verify` as well, a wide verify step's boundaries take the
/// round, so only its complete reductions launch the phased entry.
#[test]
fn wide_verify_keeps_its_deferred_rounds() {
    use model_deepseek41_contract::execution_options::WIDE_VERIFY;
    let (host, mut instance) = create_with_options(
        Shape::new(&FLASH, Ranks::TP4).unwrap(),
        32,
        192,
        &[
            (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
            (PHASED_DECODE_REDUCE, true.into()),
            (WIDE_VERIFY, true.into()),
        ],
    );
    for slot in 0..32 {
        execute(
            &mut instance,
            &host,
            &super::super::prompt(slot + 1, slot, 0, 3, true),
        );
        instance.commit().unwrap();
    }
    for (count, rows) in [(16, 65), (32, 128), (32, 192)] {
        let (target, _, accepted) =
            super::verify_rows::verify_step(&mut instance, &host, count, rows);
        let launched = |symbol: &str| {
            target
                .iter()
                .filter(|line| line.split_whitespace().nth(2) == Some(symbol))
                .count()
        };
        assert_eq!(launched(TWO_SHOT), 4 * (REDUCTIONS - DEFERRED), "{rows}");
        assert_eq!(launched(SCATTER), 4 * DEFERRED, "{rows}");
        assert_eq!(launched(ONE_SHOT) + launched(ROUND), 0, "{rows}");
        instance.commit_accepted(&accepted).unwrap();
    }
    for request in 1..=32 {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
