//! The drafter's complete program at every item count one forward may run.
use super::*;

/// Every item count up to 16 (80 block rows) lowers one complete drafter
/// forward. With `CHOPINFER_DRAFTER_JOURNALS` set to a directory, each
/// count's rank 0 program is written there as `items-N.txt`, one launch per
/// line: symbol, rows, grid, block, dynamic shared bytes, collective.
#[test]
fn drafter_programs_lower_one_forward_at_every_item_count() {
    let (_host, mut instance) =
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 192);
    let directory = std::env::var_os("CHOPINFER_DRAFTER_JOURNALS").map(std::path::PathBuf::from);
    for items in 1..=16 {
        let programs = crate::instance::dspark_drafter_launches(instance.handle(), items)
            .unwrap_or_else(|e| panic!("{items} items: {e}"));
        assert_eq!(programs.len(), 4);
        assert!(programs.iter().all(|rank| rank.len() == programs[0].len()));
        for (rank, program) in programs.iter().enumerate() {
            let status_folds = program
                .iter()
                .filter(|(symbol, ..)| {
                    symbol.starts_with("chop_deepseek41_mxfp8_fused_n")
                        || symbol.starts_with("chop_deepseek41_verify_mxfp8_")
                })
                .collect::<Vec<_>>();
            if items <= 12 {
                assert_eq!(
                    status_folds.len(),
                    4 * FLASH.dspark_stages() as usize,
                    "items={items} rank={rank}: query and WO folds must reach every stage"
                );
                assert!(status_folds.iter().all(|(_, rows, ..)| *rows == items * 5));
                assert!(
                    status_folds
                        .iter()
                        .all(|(symbol, ..)| !symbol.contains("n512_k5120"))
                );
            } else {
                assert!(status_folds.is_empty(), "unqualified draft width {items}");
            }
        }
        if let Some(directory) = &directory {
            let text: String = programs[0]
                .iter()
                .map(|(symbol, rows, grid, block, shared, collective)| {
                    format!(
                        "{symbol} rows={rows} grid={}x{}x{} block={}x{}x{} smem={shared} coll={}\n",
                        grid[0],
                        grid[1],
                        grid[2],
                        block[0],
                        block[1],
                        block[2],
                        u8::from(*collective)
                    )
                })
                .collect();
            std::fs::write(directory.join(format!("items-{items}.txt")), text).unwrap();
        }
    }
    instance.destroy().unwrap();
}

#[test]
fn used_depth_program_preserves_backbone_and_removes_only_unused_heads() {
    let (_host, mut instance) =
        create_with_limits(Shape::new(&FLASH, Ranks::TP4).unwrap(), 32, 192);
    let directory =
        std::env::var_os("CHOPINFER_DRAFTER_DEPTH_JOURNALS").map(std::path::PathBuf::from);
    for items in 3..=8 {
        let full = crate::instance::dspark_drafter_launches(instance.handle(), items).unwrap();
        let compact =
            crate::instance::dspark_drafter_launches_at_depth(instance.handle(), items, 3).unwrap();
        for (rank, (a, b)) in full.iter().zip(&compact).enumerate() {
            let collapse = a
                .iter()
                .rposition(|op| op.0.ends_with("weighted_head_f32_v1"))
                .unwrap();
            assert_eq!(
                &a[..collapse + 2],
                &b[..collapse + 2],
                "backbone, collapse and norm: items={items} rank={rank}"
            );
            for (program, positions) in [(a, 5), (b, 3)] {
                for symbol in [
                    "chop_deepseek41_dspark_markov_tiled_v1",
                    "chop_deepseek41_dspark_confidence_parallel_v1",
                    "chop_dspark_argmax_final_v1",
                ] {
                    assert_eq!(
                        program.iter().filter(|op| op.0 == symbol).count(),
                        positions
                    );
                }
            }
            for op in &b[collapse + 2..] {
                if op.0.starts_with("chop_deepseek41_head_f32_") {
                    assert_eq!(op.1, items * 3);
                    assert!(!op.0.contains("prefill"));
                }
            }
        }
        if let Some(directory) = &directory {
            for (label, program) in [("full", &full), ("compact", &compact)] {
                std::fs::write(
                    directory.join(format!("items-{items}-{label}.json")),
                    serde_json::to_vec_pretty(program).unwrap(),
                )
                .unwrap();
            }
        }
    }
    for items in [1, 2, 9] {
        assert!(
            crate::instance::dspark_drafter_launches_at_depth(instance.handle(), items, 3).is_err()
        );
    }
    instance.destroy().unwrap();
}
