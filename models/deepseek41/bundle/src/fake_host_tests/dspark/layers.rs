//! The target can consume one layer while the next layer is still gathering.
use super::*;

#[test]
fn layer_gathers_join_only_at_their_own_consumers() {
    for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let (host, mut instance) = create(shape);
        execute(
            &mut instance,
            &host,
            &super::super::prompt(7, 0, 0, 3, true),
        );
        instance.commit().unwrap();
        for cycle in 0..2 {
            let begin = host.journal.borrow().len();
            execute(&mut instance, &host, &verify(7, 0, 3 + cycle, 5));
            let logs = host.logs.borrow();
            let timing: serde_json::Value = serde_json::from_str(
                logs.iter()
                    .rev()
                    .find(|s| s.contains("\"phase\":\"dspark-engram\""))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(timing["layers"][0]["layer"], 1);
            assert_eq!(timing["layers"][1]["layer"], 14);
            assert_eq!(timing["layers"].as_array().unwrap().len(), 2);
            assert_eq!(timing["bundle_readbacks"], 3);
            drop(logs);
            if cycle == 1 {
                let journal = host.journal.borrow();
                let step = &journal[begin..];
                let at = step
                    .iter()
                    .position(|s| s.starts_with("async mapped ids "))
                    .unwrap();
                let target = &step[at..];
                assert_eq!(
                    target
                        .iter()
                        .filter(|s| s.starts_with("async mapped ids "))
                        .count(),
                    2,
                    "each Engram layer needs its own transfer"
                );
                for (layer, preceding) in [(1, 1), (14, 2)] {
                    let first = target
                        .iter()
                        .position(|s| s.contains(&format!("layers.{layer}.engram.embed.weight")))
                        .unwrap();
                    assert_eq!(
                        target[..first]
                            .iter()
                            .filter(|s| s.starts_with("replay "))
                            .count(),
                        preceding * ranks.count(),
                        "layer {layer} joined before its target prefix ran"
                    );
                }
                assert_eq!(
                    target.iter().filter(|s| s.starts_with("replay ")).count(),
                    3 * ranks.count()
                );
                assert_eq!(
                    target
                        .iter()
                        .filter(|s| s.starts_with("async mapped complete"))
                        .count(),
                    2
                );
                assert!(!target.iter().any(|s| s.starts_with("synchronize ")));
            }
            instance.commit_accepted(&[kept(7, 1)]).unwrap();
        }
        instance.retire_request(7).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
