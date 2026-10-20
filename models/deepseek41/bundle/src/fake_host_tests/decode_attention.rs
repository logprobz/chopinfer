//! Serving launches use only the measured source rows and their ordered merge.
use super::*;

fn check_sources(lines: &[String], rows: u32, verify: bool, selected: bool, ranks: u32) {
    let sources: Vec<_> = lines
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && line.contains("chop_latent_sources_")
                && !line.contains("_merge_")
        })
        .collect();
    assert_eq!(sources.len(), ranks as usize * 40);
    for line in sources {
        let symbol = line.split_whitespace().nth(2).unwrap();
        assert_eq!(
            symbol.contains("_stage64_h0_b0_profile0_v1"),
            selected,
            "{line}"
        );
        assert!(
            symbol.contains(if verify { "_verify" } else { "_decode" }),
            "{line}"
        );
        assert!(line.contains(&format!("grid_x: {rows},")), "{line}");
        assert!(line.contains("grid_z: 8,"), "{line}");
        if selected {
            assert!(line.contains("grid_y: 2,"), "{line}");
            assert!(line.contains("block_x: 128,"), "{line}");
            assert!(line.contains("shared_memory_bytes: 75008,"), "{line}");
            assert!(line.ends_with("NullDevicePointer]"), "{line}");
        }
    }
    let merges: Vec<_> = lines
        .iter()
        .filter(|line| {
            line.starts_with("launch ")
                && line.contains("chop_latent_sources_")
                && line.contains("_merge_")
        })
        .collect();
    assert_eq!(merges.len(), ranks as usize * 40);
    assert!(merges.iter().all(|line| !line.contains("_stage")));
}

#[test]
fn measured_decode_attention_serves_exact_graph_buckets() {
    crate::sm120::latent_cache::with_reference(measured_staging_reference);
}

fn measured_staging_reference() {
    crate::sm120::latent_decode::with_reference(|| run_case(Ranks::TP4, KV_LAYOUTS[0], true));
    for (ranks, layout) in [
        (Ranks::TP4, KV_LAYOUTS[0]),
        (Ranks::TP4, KV_LAYOUTS[1]),
        (Ranks::TP2, KV_LAYOUTS[0]),
    ] {
        run_case(ranks, layout, false);
    }
}

fn run_case(ranks: Ranks, layout: KvLayout, reference: bool) {
    let shape = Shape::new(&FLASH, ranks).unwrap();
    let host = FakeHost::new(shape, layout);
    if layout.storage == KvStorage::Native {
        host.speculation.set(Some((5, 48)));
    }
    let mut instance = create(&host, 32);
    let qualified = ranks == Ranks::TP4 && layout.storage == KvStorage::Native;
    assert_eq!(
        host.device
            .module_cubins
            .borrow()
            .values()
            .any(|cubin| *cubin == crate::sm120::latent_decode::CUBIN),
        qualified
    );
    let mut ends = [1_u64; 32];
    for slot in 0..32 {
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, 1, true),
            &mut Views::default(),
        );
    }
    for rows in [1, 8, 16, 32] {
        let from = host.device.journal.borrow().len();
        let items = (0..rows)
            .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        check_sources(
            &host.device.journal.borrow()[from..],
            rows,
            false,
            qualified && !reference && rows == 8,
            shape.ranks(),
        );
        for end in &mut ends[..rows as usize] {
            *end += 1;
        }
    }
    // BF16 KV admits autoregressive requests only. The query-selection test
    // separately verifies that BF16 queries never select a staging entry.
    for rows in [8_u32, 16, 32]
        .into_iter()
        .filter(|_| layout.storage == KvStorage::Native)
    {
        let items_count = (rows / 4).min(8);
        let width = rows / items_count;
        let tokens = vec![7; width as usize];
        let items = (0..items_count)
            .map(|slot| {
                (
                    u64::from(slot) + 1,
                    u64::from(slot),
                    ends[slot as usize],
                    tokens.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let from = host.device.journal.borrow().len();
        instance
            .execute(&verify(&items), &mut Views::default())
            .unwrap_or_else(|error| {
                panic!(
                    "verify {rows}: {error:?}, {:?}, {:?}",
                    instance.last_execute_refusal(),
                    host.logs.borrow()
                )
            });
        check_sources(
            &host.device.journal.borrow()[from..],
            rows.next_power_of_two(),
            true,
            qualified && !reference && rows == 32,
            shape.ranks(),
        );
        instance
            .commit_accepted(
                &(1..=items_count)
                    .map(|id| kept(u64::from(id), width))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        for end in &mut ends[..items_count as usize] {
            *end += u64::from(width);
        }
    }
    // The served contract caps a verification step at 32 rows. Reject a
    // wider request before launching, even when the host advertises 48.
    let tokens = [7; 6];
    let too_wide = (0..8)
        .map(|slot| (slot as u64 + 1, slot as u64, ends[slot], tokens.as_slice()))
        .collect::<Vec<_>>();
    let from = host.device.journal.borrow().len();
    assert!(matches!(
        instance.execute(&verify(&too_wide), &mut Views::default()),
        Err(BundleAbiError::Status(BundleStatusV1::Unsupported))
    ));
    assert_eq!(host.device.journal.borrow().len(), from);
    assert_launches_fit_configured_shared(&host.device.journal.borrow(), &shape.topology());
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_ATTENTION_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let layout = format!("{:?}", layout.storage).to_lowercase();
        let arm = if reference { "reference" } else { "serving" };
        let name = format!(
            "{}.{}.decode-buckets.{arm}.journal",
            shape.topology(),
            layout
        );
        std::fs::write(dir.join(&name), host.device.journal.borrow().join("\n")).unwrap();
        export_handles(&host.device, &dir, &name);
    }
}

pub(super) fn export_handles(device: &FakeDevice, dir: &std::path::Path, name: &str) {
    let metadata = serde_json::json!({
        "buffers": device.buffer_order.borrow().clone(),
        "ranks": device.allocation_ranks.borrow().clone(),
        "modules": device.module_cubins.borrow().clone(),
    });
    std::fs::write(
        dir.join(format!("{name}.handles.json")),
        metadata.to_string(),
    )
    .unwrap();
}
