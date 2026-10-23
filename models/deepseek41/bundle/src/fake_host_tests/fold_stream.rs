//! Fold scratch aliases ordered layers, while weights and lanes stay disjoint.
use super::shared_overlap::journal_spans;
use super::*;
use crate::shared_overlap::{Settings, with_settings_in_tests};

#[test]
fn packed_folds_share_scratch_across_layers_and_keep_lanes_disjoint() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = with_settings_in_tests(Settings::ALL, || create(&host, 4));
        audit_allocations(&host, &instance, false);
        let initial = host.device.journal.borrow().clone();
        let mut initialized = BTreeMap::new();
        let mut persistent = std::collections::BTreeSet::new();
        for line in initial.iter().filter(|l| {
            l.starts_with("launch ")
                && (l.contains("mxfp8_stream_") || l.contains("mxfp8_direct_"))
                && l.contains("_prepare_v1")
        }) {
            let symbol = line
                .split_whitespace()
                .nth(2)
                .unwrap()
                .strip_suffix("_prepare_v1")
                .unwrap()
                .to_owned();
            let spans = journal_spans(line);
            let (packed, state) = match spans.as_slice() {
                [_, packed, state] => (vec![*packed], *state),
                [_, _, a, b, state] => (vec![*a, *b], *state),
                _ => panic!("unexpected packed initializer: {line}"),
            };
            assert!(line.contains("flags: 0,"), "prepare must be stream ordered");
            assert!(
                line.ends_with(&format!("ScalarU32({})]", state.2 / 4)),
                "prepare must zero the full shared state: {line}"
            );
            assert!(
                initialized
                    .insert(packed[0], (symbol, packed.clone(), state))
                    .is_none()
            );
            persistent.insert(state.0);
            for span in packed {
                assert!(persistent.insert(span.0));
            }
        }
        assert!(!initialized.is_empty());
        assert!(
            initial
                .iter()
                .rposition(|line| line.starts_with("synchronize "))
                .unwrap()
                > initial
                    .iter()
                    .rposition(|line| line.contains("_prepare_v1 "))
                    .unwrap()
        );
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(7, 0, 0, 3, true),
            &mut Views::default(),
        );
        step(
            &mut instance,
            &prompt(8, 1, 0, 3, true),
            &mut Views::default(),
        );
        step(
            &mut instance,
            &decode(&[(7, 0, 3, 9)]),
            &mut Views::default(),
        );
        step(
            &mut instance,
            &decode(&[(7, 0, 4, 9), (8, 1, 3, 9)]),
            &mut Views::default(),
        );
        step(
            &mut instance,
            &decode(&[(7, 0, 5, 9)]),
            &mut Views::default(),
        );
        let lines = host.device.journal.borrow()[from..].to_vec();
        let mut workspaces = BTreeMap::new();
        let mut observed = 0;
        for line in lines.iter().filter(|l| {
            l.starts_with("launch ") && (l.contains("mxfp8_stream_") || l.contains("mxfp8_direct_"))
        }) {
            assert!(!line.contains("_prepare_v1"), "initializer entered a step");
            let symbol = line.split_whitespace().nth(2).unwrap();
            let spans = journal_spans(line);
            let state = spans[spans.len() - 3];
            let workspace = (!symbol.contains("mxfp8_direct_")).then(|| spans[spans.len() - 4]);
            let (owner, packed, initialized_state) = initialized
                .get(&spans[2])
                .expect("uninitialized packed weights");
            assert_eq!(state.0, initialized_state.0);
            assert_eq!(state.1, initialized_state.1);
            assert!(state.2 <= initialized_state.2);
            let shape = |symbol: &str| {
                symbol
                    .split('_')
                    .filter(|part| {
                        part.len() > 1
                            && matches!(part.as_bytes()[0], b'n' | b'k' | b'g')
                            && part[1..].bytes().all(|b| b.is_ascii_digit())
                    })
                    .collect::<Vec<_>>()
                    .join("_")
            };
            assert_eq!(shape(owner), shape(symbol));
            assert_eq!(spans[2], packed[0]);
            if packed.len() == 2 {
                assert_eq!(spans[4], packed[1]);
            }
            if let Some(workspace) = workspace {
                assert!(!persistent.contains(&workspace.0));
                if let Some(old) = workspaces.insert(workspace.0, state.0) {
                    assert_eq!(old, state.0, "workspace shared by distinct fold operations");
                }
            }
            observed += 1;
        }
        assert!(observed > shape.ranks() as usize * FLASH.layers() as usize);
        let (forks, collisions) = super::shared_overlap::audit_scratch(&lines);
        if ranks != Ranks::TP1 {
            assert!(forks > 0);
        }
        assert_eq!(collisions, 0);
        instance.retire_request(7).unwrap();
        instance.retire_request(8).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        println!(
            "{}",
            serde_json::json!({"topology":shape.topology(),"initialized_operations":initialized.len(),"observed_launches":observed,"shared_workspaces":workspaces.len(),"reinitialized_in_steps":0,"unordered_aliases":0})
        );
    }
}

pub(super) fn audit_allocations(host: &Rc<Host>, instance: &BundleInstanceV1, drafter: bool) {
    use std::collections::BTreeSet;
    let mut ranks = 0;
    for views in crate::instance::fold_views_for_test(instance.handle()) {
        let mut groups = BTreeMap::new();
        let mut unique = BTreeSet::new();
        let mut packed = BTreeSet::new();
        let mut before = [0_u64; 3];
        let mut after = [0_u64; 3];
        let mut reused = 0;
        let mut lanes = [BTreeSet::new(), BTreeSet::new()];
        let mut images = [BTreeSet::new(), BTreeSet::new()];
        for (name, span) in &views {
            let mut parts = name.splitn(3, '.');
            let image = parts.next().unwrap();
            parts.next().unwrap().parse::<u32>().unwrap();
            let projection = parts.next().unwrap();
            let role = name.rsplit('.').next().unwrap();
            let index = match role {
                "packed0" | "packed1" => 0,
                "state" => 1,
                "workspace" => 2,
                _ => panic!("unexpected fold role {role}"),
            };
            let allocated = host.device.allocation_ranks.borrow()[&span.buffer.slot()].1;
            assert_eq!(allocated, span.length, "{name}");
            before[index] += allocated;
            if index == 0 {
                assert!(
                    packed.insert(span.buffer.slot()),
                    "packed weights alias: {name}"
                );
                after[0] += allocated;
                continue;
            }
            if let Some(old) = groups.insert((image, projection), *span) {
                assert_eq!(old, *span, "layers must share {image}.{projection}");
                reused += 1;
            } else {
                assert!(
                    unique.insert(span.buffer.slot()),
                    "distinct fold operations alias: {name}"
                );
                after[index] += allocated;
            }
            lanes[usize::from(projection.starts_with("ffn.shared_experts."))]
                .insert(span.buffer.slot());
            images[usize::from(image == "mtp")].insert(span.buffer.slot());
        }
        assert!(reused > 0);
        assert!(!lanes[0].is_empty() && !lanes[1].is_empty());
        assert!(lanes[0].is_disjoint(&lanes[1]));
        assert!(images[0].is_disjoint(&images[1]));
        assert_eq!(!images[1].is_empty(), drafter);
        assert!(packed.is_disjoint(&unique));
        assert_eq!(before[0], after[0]);
        assert!(before[1] > after[1] && before[2] > after[2]);
        if host.shape.ranks() == 4 {
            let (expected_before, expected_after) = if drafter {
                (
                    [2_211_119_104, 317_184, 360_526_336],
                    [2_211_119_104, 27_904, 92_817_664],
                )
            } else {
                (
                    [2_015_887_360, 297_984, 342_095_872],
                    [2_015_887_360, 20_992, 83_396_864],
                )
            };
            assert_eq!(before, expected_before, "per-layer allocation baseline");
            assert_eq!(after, expected_after, "actual shared allocations per rank");
        }
        println!(
            "{}",
            serde_json::json!({"fold_bytes_before":before,"fold_bytes_after":after,"saved":before.iter().sum::<u64>() - after.iter().sum::<u64>(),"drafter":drafter})
        );
        ranks += 1;
    }
    assert!(ranks > 0);
}
