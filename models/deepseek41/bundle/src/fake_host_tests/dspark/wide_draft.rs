//! The `wide_draft` candidate: with `wide_verify`, a drafter step of more
//! than twelve drafting items runs them in the fewest forwards the widest
//! wide verify bucket holds, and a forward above 64 block rows folds its
//! dense projections at the wide verify buckets and projects the vocabulary
//! in the target head's exact F32 row tiles. Forwards of up to twelve items
//! keep their programs.
use super::*;
use model_deepseek41_contract::execution_options::{
    DSPARK_GRAPH_HANDLES_PER_RANK, WIDE_DRAFT, WIDE_VERIFY,
};

const RING_GATHER: &str = "chop_deepseek41_dspark_ring_gather_v1";
const HEAD_TILES: &str = "chop_deepseek41_verify_f32_n32320_k5120_rows_f256_r32_w32x32_s2_k64_v1";

/// An instance of 32 items at `verify_rows` verify rows, eager, with
/// `wide_verify` and, if `wide`, `wide_draft`.
fn create_wide(verify_rows: u32, wide: bool) -> (Rc<Host>, BundleInstanceV1) {
    create_with_options(
        Shape::new(&FLASH, Ranks::TP4).unwrap(),
        32,
        verify_rows,
        &[
            (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
            (WIDE_VERIFY, true.into()),
            (WIDE_DRAFT, wide.into()),
        ],
    )
}

/// The items of each drafter forward of a step, in order: the grid of its
/// ring gathers, six on each of the four ranks.
fn forwards(step: &[String]) -> Vec<u32> {
    let hash = crate::sm120::engram_hash_rows::SYMBOL;
    let end = step.iter().position(|line| line.contains(hash)).unwrap();
    let grids: Vec<u32> = step[..end]
        .iter()
        .filter(|line| {
            line.starts_with("launch ") && line.split_whitespace().nth(2) == Some(RING_GATHER)
        })
        .map(|line| {
            let (_, rest) = line.split_once("grid_x: ").unwrap();
            rest[..rest.find(',').unwrap()].parse().unwrap()
        })
        .collect();
    assert_eq!(grids.len() % 24, 0);
    grids
        .chunks(24)
        .map(|forward| {
            assert!(forward.iter().all(|&items| items == forward[0]));
            forward[0]
        })
        .collect()
}

/// A step of `count` drafting items runs the fewest near-equal forwards
/// within twelve items, or within the widest bucket's items with
/// `wide_draft`: 25 at 128 verify rows and 32 at 192.
#[test]
fn wide_drafts_run_the_fewest_forwards_the_widest_bucket_holds() {
    for (verify_rows, wide, steps) in [
        (
            192,
            false,
            &[(8, vec![8]), (16, vec![8, 8]), (32, vec![11, 11, 10])][..],
        ),
        (192, true, &[(8, vec![8]), (16, vec![16]), (32, vec![32])]),
        (
            128,
            true,
            &[(8, vec![8]), (16, vec![16]), (32, vec![16, 16])],
        ),
    ] {
        let (host, mut instance) = create_wide(verify_rows, wide);
        for slot in 0..32 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
        for (count, items) in steps {
            let (_, step, accepted) =
                super::verify_rows::verify_step(&mut instance, &host, *count, 2 * count);
            assert_eq!(
                &forwards(&step),
                items,
                "{count} items at {verify_rows} rows, wide {wide}"
            );
            instance.commit_accepted(&accepted).unwrap();
        }
        for request in 1..=32 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Every forward above twelve items runs the wide dense folds and the
/// head's row tiles, and no 8192-row fold or split head route; forwards
/// of up to twelve items lower as without the option.
#[test]
fn wide_drafter_forwards_run_the_wide_buckets() {
    for (verify_rows, most) in [(192, 32), (128, 25)] {
        let (_, mut narrow) = create_wide(verify_rows, false);
        let (_, mut wide) = create_wide(verify_rows, true);
        for items in 1..=most {
            let lowered = |instance: &BundleInstanceV1| {
                crate::instance::dspark_drafter_launches(instance.handle(), items)
                    .unwrap_or_else(|e| panic!("{items} items: {e}"))
            };
            let (before, after) = (lowered(&narrow), lowered(&wide));
            let what = format!("{items} items at {verify_rows} rows");
            if items <= 12 {
                assert_eq!(before, after, "{what}");
                continue;
            }
            for program in &after {
                let symbols: Vec<_> = program.iter().map(|op| op.0.as_str()).collect();
                assert!(
                    !symbols.iter().any(|s| s.contains("_mxfp8_fold_m8192_")),
                    "{what}: an 8192-row fold"
                );
                assert!(
                    !symbols
                        .iter()
                        .any(|s| s.starts_with("chop_deepseek41_head_f32_")),
                    "{what}: the head's split route"
                );
                assert_eq!(
                    symbols.iter().filter(|s| **s == HEAD_TILES).count(),
                    1,
                    "{what}"
                );
                let folds = program
                    .iter()
                    .filter(|op| op.0.contains("_dense_decode_") && op.0.contains("_rows256_"))
                    .filter(|op| op.0.ends_with("_packed_v1"));
                assert_eq!(folds.clone().count(), 8 * 3, "{what}: wide folds");
                assert!(folds.clone().all(|op| op.1 == items * 5), "{what}");
            }
        }
        narrow.destroy().unwrap();
        wide.destroy().unwrap();
    }
}
