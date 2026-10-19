//! Persistent fold storage is initialized at create and private across operations.
use super::shared_overlap::journal_spans;
use super::*;
use crate::shared_overlap::{Settings, with_settings_in_tests};

#[test]
fn packed_folds_initialize_once_and_own_disjoint_state_and_workspace() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = with_settings_in_tests(Settings::ALL, || create(&host, 4));
        let initial = host.device.journal.borrow().clone();
        let mut initialized = BTreeMap::new();
        let mut persistent = std::collections::BTreeSet::new();
        for line in initial.iter().filter(|l| {
            l.starts_with("launch ") && l.contains("mxfp8_stream_") && l.contains("_prepare_v1")
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
            assert!(
                initialized
                    .insert(state, (symbol, packed.clone()))
                    .is_none()
            );
            for span in packed.into_iter().chain([state]) {
                assert!(persistent.insert(span.0));
            }
        }
        assert!(!initialized.is_empty());
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
        for line in lines
            .iter()
            .filter(|l| l.starts_with("launch ") && l.contains("mxfp8_stream_"))
        {
            assert!(!line.contains("_prepare_v1"), "initializer entered a step");
            let symbol = line.split_whitespace().nth(2).unwrap();
            let spans = journal_spans(line);
            let state = spans[spans.len() - 3];
            let workspace = spans[spans.len() - 4];
            let (owner, packed) = initialized.get(&state).expect("uninitialized fold state");
            assert_eq!(owner, symbol);
            assert_eq!(spans[2], packed[0]);
            if packed.len() == 2 {
                assert_eq!(spans[4], packed[1]);
            }
            assert!(!persistent.contains(&workspace.0));
            if let Some(old) = workspaces.insert(workspace.0, state.0) {
                assert_eq!(old, state.0, "workspace shared by distinct fold operations");
            }
            observed += 1;
        }
        assert!(observed > shape.ranks() as usize * FLASH.layers() as usize);
        instance.retire_request(7).unwrap();
        instance.retire_request(8).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        println!(
            "{}",
            serde_json::json!({"topology":shape.topology(),"initialized_operations":initialized.len(),"observed_launches":observed,"private_workspaces":workspaces.len(),"reinitialized_in_steps":0,"aliases":0})
        );
    }
}
