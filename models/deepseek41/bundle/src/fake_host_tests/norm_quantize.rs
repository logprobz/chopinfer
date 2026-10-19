use super::*;

/// The journal prints each bound span even when two planner IDs name the
/// same storage. Compare physical ranges, not planner handles.
fn spans(line: &str) -> Vec<(u64, u64, u64)> {
    line.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|part| {
            let (buffer, part) = part.split_once(" }, offset: ").unwrap();
            let (offset, part) = part.split_once(", length: ").unwrap();
            let length = part.split_once(" }").unwrap().0;
            (
                buffer.parse().unwrap(),
                offset.parse().unwrap(),
                length.parse().unwrap(),
            )
        })
        .collect()
}
fn overlap(a: (u64, u64, u64), b: (u64, u64, u64)) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

#[test]
fn concurrent_router_and_shared_expert_workspace_ranges_are_disjoint() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    step(
        &mut instance,
        &prompt(7, 0, 0, 3, true),
        &mut Views::default(),
    );
    let start = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, 3, 9)]),
        &mut Views::default(),
    );
    let lines = host.device.journal.borrow()[start..].to_vec();
    // The paired gate/up admission and router row classifier are the first
    // writers in the two branches of every captured MoE fork. Their workspace
    // views bound every downstream partial/flag use in that branch.
    let side: Vec<_> = lines
        .iter()
        .filter(|l| {
            l.starts_with("launch ")
                && (l.contains("_pair_v1_admit_v1") || l.contains("_pair_f32_v2_initialize_v2"))
        })
        .map(|l| spans(l)[2])
        .collect();
    let router_tag = format!(
        "_n{}_k{}_v1_cached_rows_v1",
        FLASH.shape.moe.experts, FLASH.shape.hidden
    );
    let compute: Vec<_> = lines
        .iter()
        .filter(|l| l.starts_with("launch ") && l.contains(&router_tag))
        .map(|l| *spans(l).last().unwrap())
        .collect();
    assert_eq!(side.len(), FLASH.layers() as usize * shape.ranks() as usize);
    assert_eq!(compute.len(), side.len());
    assert!(lines.iter().any(|l| l.starts_with("wait ")));
    let conflicts: Vec<_> = side
        .iter()
        .zip(&compute)
        .enumerate()
        .filter(|(_, (a, b))| overlap(**a, **b))
        .map(|(i, _)| i)
        .collect();
    instance.destroy().unwrap();
    assert!(
        conflicts.is_empty(),
        "concurrent backend scratch overlaps at fork indices {conflicts:?}"
    );
    // The predicate catches overlapping subviews even with unequal offsets.
    assert!(overlap((1, 256, 512), (1, 512, 512)));
    assert!(!overlap((1, 256, 256), (1, 512, 512)));
}

#[test]
fn decode_journal_fuses_norm_quantizers_and_widens() {
    for kv in KV_LAYOUTS {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, kv);
        let mut instance = create(&host, 4);
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        let start = host.device.journal.borrow().len();
        step(
            &mut instance,
            &decode(&[(7, 0, 3, 9)]),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[start..].to_vec();
        let launches: Vec<_> = lines.iter().filter(|l| l.starts_with("launch ")).collect();
        let count = |symbol: &str| launches.iter().filter(|l| l.contains(symbol)).count();
        let ranks = shape.ranks() as usize;
        let layers = FLASH.layers() as usize;
        assert_eq!(layers, 40);
        assert_eq!(
            count("chop_norm_quantize_deepseek41_input_v1"),
            2 * layers * ranks
        );
        assert_eq!(
            count("chop_norm_quantize_deepseek41_query_v1"),
            layers * ranks
        );
        assert_eq!(count("chop_row_rmsnorm_deepseek41_w5120_v1"), 0);
        assert_eq!(count("chop_deepseek41_q_norm_f32_v1"), 0);
        // Only the embedding's widening remains. Later indexer projections
        // retain their quantization after other producers can set status.
        assert_eq!(count("chop_row_rounding_widen_deepseek41_w5120_v1"), ranks);
        let widened = (0..FLASH.layers())
            .flat_map(|layer| {
                [
                    crate::program::Sublayer::Attention,
                    crate::program::Sublayer::Mlp,
                ]
                .map(move |s| (layer, s))
            })
            .filter(|&(layer, s)| crate::ops::exact_f32::widens(&FLASH, layer, s))
            .count();
        assert_eq!(widened, 48);
        assert_eq!(count("chop_deepseek41_window_fused_"), layers * ranks);
        assert_eq!(count("chop_deepseek41_latent_window_rope_v1"), 0);
        let saved = 5 * layers + widened;
        assert_eq!(saved, 248);
        assert_eq!(launches.len() % ranks, 0);
        eprintln!(
            "NORM_LAUNCHES {:?} rows=1 after={} removed={saved}",
            kv.storage,
            launches.len() / ranks
        );
        if let Some(dir) = std::env::var_os("CHOPINFER_NORM_JOURNALS") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join(format!("decode.{:?}.journal", kv.storage)),
                lines.join("\n"),
            )
            .unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
