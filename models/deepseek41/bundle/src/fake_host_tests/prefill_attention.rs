//! Native TP4 prefill fuses the mixed layers' attention and output (shared
//! heads) at every row count and packing of prompts outside the cache's
//! admitted chunks, which keep their measured rule; paired attention covers
//! the window layers.
use super::*;

#[test]
fn fused_prefill_writes_output_without_partial_or_merge_launches() {
    let trace = super::prefill_cache::composed_trace(0, false, true, false);
    let fused = trace
        .iter()
        .filter(|line| line.starts_with("launch ") && line.contains("_head_shared_finish_v1"))
        .collect::<Vec<_>>();
    // Cold full mixed32 layers and every retained mixed64 count of 64 to
    // 2496 rows, the two small tails included. The later native chunk stays
    // unchanged.
    assert_eq!(fused.len(), 4 * (18 + 40));
    for line in fused {
        assert!(line.contains("shared_memory_bytes: 83200,"));
        assert!(line.contains("NullDevicePointer, ScalarU64("));
        assert!(line.contains("block_x: 256,"));
    }
}

#[test]
fn retained_prefill_fuses_every_measured_count_and_small_tail() {
    let trace = super::prefill_cache::composed_trace(0, false, true, false);
    let attention = trace
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && line.contains("h16_d512_mixed_normalized_s1_r64_prefill_reuse2_v1")
        })
        .collect::<Vec<_>>();
    // Retained native-key attention and its merge are fused at every
    // measured count of 64 to 2496 rows, so neither launches separately.
    assert!(attention.is_empty(), "{attention:?}");
    for rows in (64..=2496).step_by(128) {
        let merge_grid = format!("grid_x: {rows}, grid_y: 16,");
        assert_eq!(
            trace
                .iter()
                .filter(|s| s.starts_with("launch ")
                    && s.contains("h16_d512_mixed_normalized_s1_r64_prefill_merge_v1")
                    && s.contains(&merge_grid))
                .count(),
            0,
            "retained rows={rows}"
        );
    }
}

#[test]
fn native_prefill_fuses_mixed_and_pairs_window_layers_at_every_row_count() {
    // Native TP4 fuses every mixed layer's prefill rows with shared heads
    // and pairs the window layers' at every row count; other layouts and
    // topologies keep the reference.
    for (ranks, layout, rows, paired) in [
        (Ranks::TP4, KV_LAYOUTS[0], 8192, true),
        (Ranks::TP4, KV_LAYOUTS[0], 8191, true),
        (Ranks::TP4, KV_LAYOUTS[1], 8192, false),
        (Ranks::TP2, KV_LAYOUTS[0], 8192, false),
        (Ranks::TP4, KV_LAYOUTS[0], 1, true),
        (Ranks::TP4, KV_LAYOUTS[0], 3, true),
        (Ranks::TP4, KV_LAYOUTS[0], 64, true),
        (Ranks::TP4, KV_LAYOUTS[0], 65, true),
        (Ranks::TP4, KV_LAYOUTS[0], 789, true),
        (Ranks::TP4, KV_LAYOUTS[0], 2560, true),
        (Ranks::TP4, KV_LAYOUTS[0], 4095, true),
        (Ranks::TP4, KV_LAYOUTS[0], 4096, true),
        (Ranks::TP4, KV_LAYOUTS[0], 4097, true),
        (Ranks::TP4, KV_LAYOUTS[1], 64, false),
        (Ranks::TP2, KV_LAYOUTS[0], 789, false),
        (Ranks::TP4, KV_LAYOUTS[1], 4096, false),
    ] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, layout);
        let record = crate::config::tests::record(&shape, layout, 1, false);
        let mut instance = api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap();
        instance
            .execute(&prompt(7, 0, 0, rows, true), &mut Views::default())
            .unwrap();
        let lines = host.device.journal.borrow();
        assert_launches_fit_configured_shared(&lines, &shape.topology());
        let attention: Vec<_> = lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains("chop_latent_sources_")
                    && line.contains("_prefill")
                    && !line.contains("_merge_")
            })
            .collect();
        assert_eq!(attention.len(), shape.ranks() as usize * 40);
        for line in &attention {
            let shared = paired && line.contains("_mixed_");
            let paired = paired && !shared;
            assert_eq!(line.contains("_head_shared_finish_v1"), shared, "{line}");
            assert_eq!(line.contains("_reuse2_v1"), paired, "{line}");
            if shared {
                assert!(
                    line.contains(&format!("grid_x: {rows}, grid_y: 1, grid_z: 1,")),
                    "{line}"
                );
                assert!(line.contains("block_x: 256,"), "{line}");
                assert!(line.contains("shared_memory_bytes: 83200,"), "{line}");
            }
            if paired {
                assert!(
                    line.contains(&format!(
                        "grid_x: {}, grid_y: 2, grid_z: 1,",
                        rows.div_ceil(2)
                    )),
                    "{line}"
                );
                assert!(line.contains("block_x: 256,"), "{line}");
                assert!(line.contains("shared_memory_bytes: 85504,"), "{line}");
            }
        }
        let merges: Vec<_> = lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains("chop_latent_sources_")
                    && line.contains("_prefill")
                    && line.contains("_merge_")
            })
            .collect();
        assert_eq!(
            merges.len(),
            attention
                .iter()
                .filter(|line| !line.contains("_finish_v1"))
                .count()
        );
        assert!(merges.iter().all(|line| !line.contains("reuse")));
        drop(lines);
        instance.commit().unwrap();
        let from = host.device.journal.borrow().len();
        instance
            .execute(&decode(&[(7, 0, rows, 1)]), &mut Views::default())
            .unwrap();
        assert!(
            host.device.journal.borrow()[from..]
                .iter()
                .all(|line| !line.contains("reuse2"))
        );
        instance.commit().unwrap();
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// One prefill step of `prompts`: (request, state slot, tokens), each from
/// its first token, packed in order.
fn packed(prompts: &[(u64, u64, u64)]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let mut items = Vec::new();
    let mut appends = Vec::new();
    for &(request, slot, count) in prompts {
        let inputs = (0..count).map(|token| token as u32).collect();
        items.push(BatchItemV1::new(request, false, count as u32, true, inputs));
        appends.push(BatchAppendV1::new(
            0,
            count,
            count,
            (0..count.div_ceil(64))
                .map(|page| slot * 16 + page)
                .collect(),
            None,
        ));
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
        .with_state_slots(prompts.iter().map(|&(_, slot, _)| slot).collect())
        .encode()
        .unwrap()
}

/// A prefill step that packs prompts of served lengths fuses every mixed
/// layer's rows with shared heads, one CTA per row and sixteen heads
/// without a merge, and pairs every window layer's rows, one CTA per two
/// rows with one reference merge.
#[test]
fn packed_prompts_fuse_mixed_and_pair_window_attention_rows() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 4, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    let prompts = [(7, 0, 554), (9, 1, 560), (11, 2, 568)];
    let rows: u64 = prompts.iter().map(|&(_, _, count)| count).sum();
    instance
        .execute(&packed(&prompts), &mut Views::default())
        .unwrap();
    let lines = host.device.journal.borrow();
    assert_launches_fit_configured_shared(&lines, &shape.topology());
    let prefill = |merge: bool| {
        lines
            .iter()
            .filter(|line| {
                line.starts_with("launch ")
                    && line.contains("chop_latent_sources_")
                    && line.contains("_prefill")
                    && line.contains("_merge_") == merge
            })
            .collect::<Vec<_>>()
    };
    let attention = prefill(false);
    assert_eq!(attention.len(), shape.ranks() as usize * 40);
    let paired = format!("grid_x: {}, grid_y: 2, grid_z: 1,", rows.div_ceil(2));
    let fused = format!("grid_x: {rows}, grid_y: 1, grid_z: 1,");
    for line in &attention {
        if line.contains("_mixed_") {
            assert!(line.contains("_head_shared_finish_v1"), "{line}");
            assert!(line.contains(&fused), "{line}");
            assert!(line.contains("shared_memory_bytes: 83200,"), "{line}");
        } else {
            assert!(line.contains("_reuse2_v1"), "{line}");
            assert!(line.contains(&paired), "{line}");
            assert!(line.contains("shared_memory_bytes: 85504,"), "{line}");
        }
    }
    let windows = attention
        .iter()
        .filter(|line| !line.contains("_mixed_"))
        .count();
    assert_eq!(windows, shape.ranks() as usize * 2);
    let merges = prefill(true);
    assert_eq!(merges.len(), windows);
    assert!(
        merges
            .iter()
            .all(|line| !line.contains("reuse") && line.contains("_window_"))
    );
    drop(lines);
    instance.commit().unwrap();
    for (request, _, _) in prompts {
        instance.retire_request(request).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
