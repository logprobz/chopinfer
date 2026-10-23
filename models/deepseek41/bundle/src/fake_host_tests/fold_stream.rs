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
        // The folds read the planned packed codes: create packs nothing.
        assert!(
            !initial
                .iter()
                .any(|l| l.starts_with("launch ") && l.contains("_prepare_v1")),
            "create launched a fold prepare"
        );
        // Each state span is zeroed on its rank stream before create's last
        // synchronize; layers sharing a span zero it once.
        let last_sync = initial
            .iter()
            .rposition(|line| line.starts_with("synchronize "))
            .unwrap();
        let mut zeroed = BTreeMap::new();
        for line in &initial {
            if line.starts_with("zero ") {
                let spans = journal_spans(line);
                let [span] = spans[..] else {
                    panic!("unexpected zero: {line}")
                };
                *zeroed.entry(span).or_insert(0) += 1;
            }
        }
        let mut states = BTreeMap::new();
        let mut persistent = std::collections::BTreeSet::new();
        for views in crate::instance::fold_views_for_test(instance.handle()) {
            for (name, span) in views {
                persistent.insert(span.buffer.raw);
                if name.ends_with(".state") {
                    states.insert((span.buffer.raw, span.offset, span.length), name);
                }
            }
        }
        assert!(!states.is_empty());
        for (span, name) in &states {
            assert_eq!(zeroed.get(span), Some(&1), "{name} is not zeroed once");
            let at = initial
                .iter()
                .position(|l| l.starts_with("zero ") && journal_spans(l) == [*span])
                .unwrap();
            assert!(at < last_sync, "{name} is zeroed after create synchronizes");
        }
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
            let initialized = states
                .keys()
                .find(|s| s.0 == state.0 && s.1 == state.1)
                .expect("state no create zeroed");
            assert!(state.2 <= initialized.2);
            // The packed bank is the weight's planned view, never fold storage.
            assert!(!persistent.contains(&spans[2].0), "{line}");
            if let Some(workspace) = workspace {
                assert!(states.keys().all(|s| s.0 != workspace.0));
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
            serde_json::json!({"topology":shape.topology(),"zeroed_states":states.len(),"observed_launches":observed,"shared_workspaces":workspaces.len(),"reinitialized_in_steps":0,"unordered_aliases":0})
        );
    }
}

pub(super) fn audit_allocations(host: &Rc<Host>, instance: &BundleInstanceV1, drafter: bool) {
    use std::collections::BTreeSet;
    let mut ranks = 0;
    for views in crate::instance::fold_views_for_test(instance.handle()) {
        let mut groups = BTreeMap::new();
        let mut unique = BTreeSet::new();
        let mut before = [0_u64; 2];
        let mut after = [0_u64; 2];
        let mut reused = 0;
        let mut lanes = [BTreeSet::new(), BTreeSet::new()];
        let mut images = [BTreeSet::new(), BTreeSet::new()];
        for (name, span) in &views {
            let mut parts = name.splitn(3, '.');
            let image = parts.next().unwrap();
            parts.next().unwrap().parse::<u32>().unwrap();
            let projection = parts.next().unwrap();
            let role = name.rsplit('.').next().unwrap();
            // The folds hold no packed copies: they bind the planned views.
            let index = match role {
                "state" => 0,
                "workspace" => 1,
                _ => panic!("unexpected fold role {role}"),
            };
            let allocated = host.device.allocation_ranks.borrow()[&span.buffer.slot()].1;
            assert_eq!(allocated, span.length, "{name}");
            before[index] += allocated;
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
        assert!(before[0] > after[0] && before[1] > after[1]);
        if host.shape.ranks() == 4 {
            // State and workspace; the packed copies these baselines held
            // (2_015_887_360 bytes per rank, 2_094_530_560 with the drafter)
            // are gone.
            let (expected_before, expected_after) = if drafter {
                ([298_752, 347_011_840], [21_760, 88_312_832])
            } else {
                ([297_984, 342_095_872], [20_992, 83_396_864])
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
