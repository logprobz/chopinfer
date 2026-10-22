//! Audited head row routing with the same inventory on both comparison arms.
use super::*;

#[test]
fn qualified_module_and_row_policy_are_explicit() {
    use crate::sm120::{f32_head, head_last};
    let module = crate::CUBINS
        .iter()
        .find(|m| m.name == head_last::CUBIN)
        .unwrap();
    assert_eq!(
        kernel_build::digest(module.bytes),
        "0e37e17f5aaffaa527f04b6fe94c6fe3e6f58c7339930d6d7482a2059e557b00"
    );
    for shape in crate::shape::topologies() {
        let enabled = head_last::enabled(shape.geometry, shape.ranks());
        assert_eq!(crate::Kernels::loads(head_last::CUBIN, &shape), enabled);
        let entry = f32_head::entry(shape.geometry, shape.ranks())
            .unwrap()
            .with_split_decode();
        let entry = head_last::head(shape.geometry, shape.ranks(), entry).unwrap();
        for rows in 1..=256 {
            assert_eq!(
                entry.decode_last(rows).is_some(),
                enabled && [1, 8, 32].contains(&rows)
            );
        }
    }
}

fn run_case(reference: bool) -> Vec<String> {
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
    let mut cases = 0;
    let mut captured = std::collections::BTreeSet::new();
    for (rows, width) in [
        (1u32, 1u32),
        (2, 1),
        (6, 1),
        (8, 1),
        (9, 1),
        (16, 1),
        (24, 1),
        (31, 1),
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
        let lines = host.device.journal.borrow();
        // AR pads to its graph capacity; verification retains the exact
        // row width, with one graph for every item count of those rows. A
        // repeated capacity replays the already checked graph.
        let capacity = if width == 1 {
            rows.next_power_of_two()
        } else {
            rows
        };
        let first = captured.insert((capacity, width > 1));
        let selected = !reference && first && [1, 8, 32].contains(&capacity);
        assert_eq!(
            lines[from..]
                .iter()
                .filter(|line| line.starts_with("launch ") && line.contains("_last_m"))
                .count(),
            if selected { 8 } else { 0 },
            "rows={rows} width={width}"
        );
        if !first {
            assert!(lines[from..].iter().any(|line| line.starts_with("replay ")));
        }
        for end in &mut ends[..count as usize] {
            *end += u64::from(width);
        }
        cases += 1;
    }
    assert_eq!(cases, 14);
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    let lines = host.device.journal.borrow().clone();
    assert_launches_fit_configured_shared(&lines, &shape.topology());
    if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_HEAD_LAST_JOURNAL_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        let name = if reference {
            "head-reference.journal"
        } else {
            "head-serving.journal"
        };
        std::fs::write(dir.join(name), lines.join("\n")).unwrap();
        super::decode_attention::export_handles(&host.device, &dir, name);
    }
    lines
}

#[test]
fn measured_heads_preserve_other_launches_and_graph_dependencies() {
    let before = crate::sm120::head_last::with_reference(|| run_case(true));
    let after = run_case(false);
    let retained = |lines: Vec<String>| {
        lines
            .into_iter()
            .filter(|line| {
                line.starts_with("launch ") && !line.contains("_head_f32_m")
                    || line.starts_with("record ")
                    || line.starts_with("wait ")
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        retained(before),
        retained(after),
        "all other arithmetic and cross-stream dependencies remain unchanged"
    );
}
