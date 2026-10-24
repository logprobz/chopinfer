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
