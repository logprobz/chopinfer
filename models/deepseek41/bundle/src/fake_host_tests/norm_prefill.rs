use super::*;

#[test]
fn prefill_norm_fuses_only_qualified_row_counts() {
    for rows in [64, 65, 320, 8192] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, rows, true),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow();
        let launches: Vec<_> = lines.iter().filter(|l| l.starts_with("launch ")).collect();
        let count = |symbol: &str| launches.iter().filter(|l| l.contains(symbol)).count();
        let qualified = rows != 65;
        let calls = 2 * FLASH.layers() as usize * shape.ranks() as usize;
        assert_eq!(
            count("chop_prefill_norm_ceil_deepseek41_cached_v1"),
            if qualified { calls } else { 0 },
            "rows={rows}"
        );
        assert_eq!(count("chop_prefill_norm_ceil_deepseek41_recompute_v1"), 0);
        assert_eq!(
            count("chop_row_rmsnorm_deepseek41_w5120_v1"),
            if qualified { 0 } else { calls }
        );
        assert_eq!(
            count("chop_deepseek41_moe_ceil_w5120_v1"),
            if qualified { 0 } else { calls }
        );
        for launch in launches
            .iter()
            .filter(|l| l.contains("chop_prefill_norm_ceil_deepseek41_cached_v1"))
        {
            assert!(launch.contains(&format!("grid_x: {rows},")), "{launch}");
            assert!(launch.contains("block_x: 256,"), "{launch}");
            assert!(launch.contains("shared_memory_bytes: 0,"), "{launch}");
        }
        drop(lines);
        let start = host.device.journal.borrow().len();
        step(
            &mut instance,
            &decode(&[(7, 0, rows, 9)]),
            &mut Views::default(),
        );
        assert!(
            !host.device.journal.borrow()[start..]
                .iter()
                .any(|l| l.starts_with("launch ") && l.contains("chop_prefill_norm_ceil_"))
        );
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
