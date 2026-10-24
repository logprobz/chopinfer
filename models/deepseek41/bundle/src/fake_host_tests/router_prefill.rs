//! Journal admission of the prefill-only router specialization.
use super::*;
#[test]
fn router_prefill_journal_preserves_decode_and_configures_shared_memory() {
    for drafter in [false, true] {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.drafter.set(drafter);
        if drafter {
            host.speculation.set(Some((5, 32)));
        }
        let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
        let services = HostServicesTableV1::new(Rc::new(dspark::MappedHost::new(Rc::clone(&host))));
        let mut instance = api().create(services, &record).unwrap();
        for (id, rows) in [
            (7, 8192),
            (8, 320),
            (9, 321),
            (10, 446),
            (11, 789),
            (12, 1803),
            (13, 7754),
            (14, 319),
            (15, 17),
        ] {
            let from = host.device.journal.borrow().len();
            instance
                .execute(&prompt(id, 0, 0, rows, true), &mut Views::default())
                .unwrap();
            instance.commit().unwrap();
            instance.retire_request(id).unwrap();
            let lines = host.device.journal.borrow();
            let zero = lines[from..]
                .iter()
                .filter(|l| {
                    l.starts_with("launch ")
                        && (l.contains("_zero_m32_v1")
                            || l.contains("_certified_m32_f64_normed_v1"))
                })
                .count();
            let in_norm = lines[from..]
                .iter()
                .filter(|l| l.starts_with("launch ") && l.contains("_certified_m32_f64_normed_v1"))
                .count();
            assert_eq!(in_norm > 0, rows == 8192, "rows {rows}: router in the norm");
            assert_eq!(
                zero > 0,
                rows >= 320,
                "rows {rows}: expected prefill router route"
            );
            if matches!(rows, 319 | 320 | 321 | 446 | 789 | 1803 | 7754) {
                for (n, k, tile) in [
                    (512, 5120, if rows < 1024 { 32 } else { 64 }),
                    (32, 5120, 64),
                    (128, 512, if rows < 4096 { 32 } else { 64 }),
                ] {
                    let symbol = format!("chop_deepseek41_dense_f32_m8192_n{n}_k{k}_v1");
                    let projected: Vec<_> = lines[from..]
                        .iter()
                        .filter(|line| line.starts_with("launch ") && line.contains(&symbol))
                        .collect();
                    assert!(
                        projected.iter().any(|line| line
                            .contains(&format!("{symbol}_tiles_n{tile}_v1 LaunchConfig"))),
                        "rows {rows}: missing {n}x{k} tile {tile}"
                    );
                    assert!(
                        projected
                            .iter()
                            .all(|line| !line.contains(&format!("{symbol} LaunchConfig"))),
                        "rows {rows}: {n}x{k} retained the baseline route"
                    );
                }
            }
            assert_launches_fit_configured_shared(&lines, &shape.topology());
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        if let Some(root) = std::env::var_os("CHOPINFER_ROUTER_JOURNAL_DIR") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(
                root.join(if drafter {
                    "dspark.journal"
                } else {
                    "ar.journal"
                }),
                host.device.journal.borrow().join("\n"),
            )
            .unwrap();
        }
    }
}

fn spans(line: &str) -> Vec<(u64, u64, u64)> {
    line.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|part| {
            let (buffer, rest) = part.split_once(" }, offset: ").unwrap();
            let (offset, rest) = rest.split_once(", length: ").unwrap();
            let length = rest.split_once(" }").unwrap().0;
            (
                buffer.parse().unwrap(),
                offset.parse().unwrap(),
                length.parse().unwrap(),
            )
        })
        .collect()
}

/// At 8192 rows each MoE layer's input norm certifies its rows and the
/// router's certified projection (64-expert tiles) reads that norm's BF16
/// rows and certificate; neither writes or reads widened rows, no
/// zero-component prepare or projection runs, and the
/// attention input norms keep the served entry.
#[test]
fn router_runs_in_the_moe_input_norm_at_8192_rows() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 1, false);
    let mut instance = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .unwrap();
    instance
        .execute(&prompt(7, 0, 0, 8192, true), &mut Views::default())
        .unwrap();
    let journal = host.device.journal.borrow();
    assert_launches_fit_configured_shared(&journal, &shape.topology());
    let launches: Vec<_> = journal
        .iter()
        .filter(|l| l.starts_with("launch "))
        .collect();
    let norm = "chop_prefill_norm_ceil_deepseek41_cached_certified_v1 ";
    let router = "chop_deepseek41_dense_f32_m8192_n384_k5120_v1_certified_m32_f64_normed_v1 ";
    assert!(
        !launches
            .iter()
            .any(|l| l.contains("_zero_prepare_v1") || l.contains("_zero_m32_v1"))
    );
    let mut pairs = 0;
    for (at, line) in launches.iter().enumerate() {
        if !line.contains(router) {
            continue;
        }
        // The nearest earlier norm launch on its stream is the certifying one.
        let stream = line.split_whitespace().nth(1).unwrap();
        let produced = launches[..at]
            .iter()
            .rev()
            .find(|l| {
                l.split_whitespace().nth(1) == Some(stream) && l.contains("chop_prefill_norm_ceil_")
            })
            .unwrap();
        assert!(produced.contains(norm), "{produced}");
        let n = spans(produced);
        let r = spans(line);
        // Norm: input, weight, normed, (no widened), codes, scales, live,
        // status, certified. Router: (no widened), normed, certified,
        // weight, output, live, domain.
        assert_eq!((n.len(), r.len()), (8, 6));
        assert_eq!([r[0], r[1], r[4]], [n[2], n[7], n[5]]);
        assert_eq!(r[1].2, 8192 * 4);
        for l in [produced, line] {
            assert_eq!(l.matches("NullDevicePointer").count(), 1, "{l}");
        }
        pairs += 1;
    }
    assert_eq!(pairs, 4 * 40);
    let certifying = launches.iter().filter(|l| l.contains(norm)).count();
    assert_eq!(certifying, 4 * 40);
    drop(journal);
    instance.commit().unwrap();
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
