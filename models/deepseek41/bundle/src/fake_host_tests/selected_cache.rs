//! The qualified cache changes only source launches, keeping every ordered merge.
use super::*;

fn sources(line: &str) -> bool {
    line.starts_with("launch ")
        && line.contains("chop_latent_sources_")
        && !line.contains("_merge_")
}

fn run_case(ranks: Ranks, layout: KvLayout, cached: bool) -> Vec<String> {
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
            .any(|name| *name == crate::sm120::latent_cache::CUBIN),
        qualified
    );
    for slot in 0..32 {
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, 1, true),
            &mut Views::default(),
        );
    }
    let mut ends = [1_u64; 32];
    let check = |lines: &[String], rows: u32, verify: bool| {
        let selected = qualified
            && cached
            && if verify {
                matches!(rows, 4 | 6 | 32)
            } else {
                matches!(rows, 1 | 8)
            };
        let count = 40 * ranks.get() as usize;
        let calls: Vec<_> = lines.iter().filter(|l| sources(l)).collect();
        assert_eq!(calls.len(), count * if selected { 2 } else { 1 });
        assert_eq!(
            calls
                .iter()
                .filter(|l| l.contains("_cache_decode_control0_"))
                .count(),
            if selected { count } else { 0 }
        );
        assert_eq!(
            calls
                .iter()
                .filter(|l| l.contains("_cached_profile0_"))
                .count(),
            if selected { count } else { 0 }
        );
        assert!(
            calls
                .iter()
                .all(|l| !l.contains("_control1_") && !l.contains("_cached_profile1_"))
        );
        for line in calls
            .into_iter()
            .filter(|l| l.contains("_cached_profile0_"))
        {
            assert!(line.contains(&format!("grid_x: {rows},")), "{line}");
            assert!(
                line.contains("grid_z: 8,") && line.contains("shared_memory_bytes: 73984,"),
                "{line}"
            );
            assert!(
                line.contains(if verify { "_verify" } else { "_decode" }),
                "{line}"
            );
        }
    };
    for rows in [1_u32, 2, 4, 8, 16, 32] {
        let from = host.device.journal.borrow().len();
        let items = (0..rows)
            .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        check(&host.device.journal.borrow()[from..], rows, false);
        for end in &mut ends[..rows as usize] {
            *end += 1;
        }
    }
    if layout.storage == KvStorage::Native {
        for (items_count, width) in [(1_u32, 4_u32), (1, 6), (2, 4), (4, 4), (8, 4)] {
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
                .unwrap_or_else(|e| panic!("{e:?}: {:?}", host.logs.borrow()));
            check(
                &host.device.journal.borrow()[from..],
                items_count * width,
                true,
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
    }
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let lines = host.device.journal.borrow().clone();
    assert_launches_fit_configured_shared(&lines, &shape.topology());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_ATTENTION_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let name = format!(
            "{}.{:?}.selected-cache.{}.journal",
            shape.topology(),
            layout.storage,
            if cached { "serving" } else { "reference" }
        );
        std::fs::write(dir.join(&name), lines.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &dir, &name);
    }
    lines
}

#[test]
fn selected_cache_changes_only_measured_source_launches() {
    let reference =
        crate::sm120::latent_cache::with_reference(|| run_case(Ranks::TP4, KV_LAYOUTS[0], false));
    let serving = run_case(Ranks::TP4, KV_LAYOUTS[0], true);
    let retained = |lines: Vec<String>| {
        lines
            .into_iter()
            .filter(|l| l.starts_with("launch ") && !sources(l))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        retained(reference),
        retained(serving),
        "all merges and non-source launches must retain their exact bindings"
    );
    run_case(Ranks::TP2, KV_LAYOUTS[0], false);
    run_case(Ranks::TP4, KV_LAYOUTS[1], false);
}

#[test]
fn selected_cache_journal_reference_lifecycle() {
    // This companion receipt audits pins with the new catalog and allocation
    // present, while the selected source route is disabled.
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    crate::sm120::latent_cache::with_reference(|| journal_steps(&host));
    let lines = host.device.journal.borrow().clone();
    assert_eq!(host.acquired.get(), host.released.get());
    assert!(lines.iter().any(|line| sources(line)));
    assert!(
        lines
            .iter()
            .filter(|line| sources(line))
            .all(|line| !line.contains("_cached_profile"))
    );
    assert_launches_fit_configured_shared(&lines, &shape.topology());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_ATTENTION_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let name = "deepseek41-flash-tp4.native.cache-disabled.journal";
        std::fs::write(dir.join(name), lines.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &dir, name);
    }
}
