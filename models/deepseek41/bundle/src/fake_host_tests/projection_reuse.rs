//! Compare routed and reference graphs with the same loaded module catalog.
use super::*;

fn projection(line: &str) -> bool {
    line.starts_with("launch ")
        && (line.contains("chop_deepseek41_mxfp8_")
            || line.contains("head_f32_") && line.contains("_decode_v1"))
}
fn run_case(reference: bool) -> Vec<String> {
    crate::sm120::head_last::with_reference(|| run_without_last(reference))
}
fn run_without_last(reference: bool) -> Vec<String> {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    host.speculation.set(Some((5, 48)));
    let mut instance = create(&host, 32);
    for slot in 0..32 {
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, 1, true),
            &mut Views::default(),
        );
    }
    let mut ends = [1u64; 32];
    let mut records = Vec::new();
    let mut captured = std::collections::BTreeSet::new();
    for (rows, width) in [
        (1u32, 1u32),
        (8, 1),
        (16, 1),
        (32, 1),
        (6, 6),
        (8, 4),
        (24, 4),
        (32, 4),
        (32, 2),
    ] {
        let count = rows / width;
        let from = host.device.journal.borrow().len();
        if width == 1 {
            let items = (0..count)
                .map(|i| (u64::from(i) + 1, u64::from(i), ends[i as usize], 7))
                .collect::<Vec<_>>();
            step(&mut instance, &decode(&items), &mut Views::default());
        } else {
            let tokens = vec![7; width as usize];
            let items = (0..count)
                .map(|i| {
                    (
                        u64::from(i) + 1,
                        u64::from(i),
                        ends[i as usize],
                        tokens.as_slice(),
                    )
                })
                .collect::<Vec<_>>();
            instance
                .execute(&verify(&items), &mut Views::default())
                .unwrap();
            instance
                .commit_accepted(
                    &(1..=count)
                        .map(|i| kept(u64::from(i), width))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let lines = host.device.journal.borrow()[from..].to_vec();
        let changed = lines
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains("_reuse32_"))
            .count();
        // One verify graph serves every item count of its rows: a repeated
        // key replays the graph checked at its capture.
        let first = captured.insert((rows, width > 1));
        if !first {
            assert_eq!(changed, 0, "rows={rows} width={width}");
            assert!(lines.iter().any(|line| line.starts_with("replay ")));
        } else if reference || rows <= 8 {
            assert_eq!(changed, 0, "rows={rows}");
        } else {
            assert!(changed >= 4, "row bucket must reach the reuse head");
        }
        if first && !reference && rows == 32 {
            assert_eq!(
                changed,
                4 * (40 * 2 + 1),
                "the WO_a admit/compute pair and one head per rank (WO_b at 32 rows runs its \
                 row-local ring)"
            );
        }
        records.push(format!("case rows={rows} width={width}"));
        records.extend(lines);
        for end in &mut ends[..count as usize] {
            *end += u64::from(width);
        }
    }
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_PROJECTION_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let name = if reference {
            "projection-reference.journal"
        } else {
            "projection-serving.journal"
        };
        std::fs::write(dir.join(name), records.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &dir, name);
    }
    records
}

#[test]
fn reuse_preserves_non_projection_launches_and_graph_bindings() {
    let before = crate::sm120::projection_reuse::with_reference(|| run_case(true));
    let after = run_case(false);
    let retained = |lines: Vec<String>| {
        let mut all = false;
        let mut kept = Vec::new();
        for line in lines {
            if let Some(case) = line.strip_prefix("case rows=") {
                all = case
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<u32>()
                    .unwrap()
                    <= 8;
                kept.push(line);
            } else if line.starts_with("launch ") && (all || !projection(&line)) {
                kept.push(line);
            }
        }
        kept
    };
    assert_eq!(
        retained(before),
        retained(after),
        "all other kernels, head classifiers, reducers and their spans remain exact"
    );
}

#[test]
fn modules_retain_the_gpu_qualified_bytes() {
    for (name, expected) in [
        (
            "mxfp8_fold_reuse_deepseek41_sm120",
            "640d2fa0b820958c2ecd2ec674481785b01eea2a03d73271c783f66300641b85",
        ),
        (
            "f32_decode_reuse_deepseek41_sm120",
            "5822024aceddb2e6064580c326b8281518289b2884cbf93b07200bc8982727de",
        ),
    ] {
        let module = crate::CUBINS.iter().find(|c| c.name == name).unwrap();
        assert_eq!(kernel_build::digest(module.bytes), expected);
    }
}

#[test]
fn reuse_modules_load_only_for_flash_tp4() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        for cubin in [
            crate::sm120::projection_reuse::HEAD_CUBIN,
            crate::sm120::projection_reuse::FOLD_CUBIN,
        ] {
            assert_eq!(crate::Kernels::loads(cubin, &shape), ranks == Ranks::TP4);
        }
    }
}
