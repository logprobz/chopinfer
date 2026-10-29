use super::*;

thread_local! { static ENABLED: Cell<bool> = const { Cell::new(false) }; }

pub(super) fn profiles(default: &'static str) -> &'static str {
    if !ENABLED.get() {
        return default;
    }
    static CANDIDATE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CANDIDATE.get_or_init(|| {
        let mut profiles: serde_json::Value = serde_json::from_str(default).unwrap();
        profiles["profiles"]["tp4-dspark"]["model_options"]["dspark_main_compact"] = "true".into();
        serde_json::to_string(&profiles).unwrap()
    })
}

pub(super) fn servings(mut values: Vec<conformance::Serving>) -> Vec<conformance::Serving> {
    if ENABLED.get() {
        for value in &mut values {
            if value.profile == "tp4-dspark" {
                value
                    .model_options
                    .insert("dspark_main_compact".into(), "true".into());
            }
        }
    }
    values
}

fn allowlist() -> String {
    let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml"));
    let mut removed = 0;
    let sections = source
        .split("[[allow]]")
        .filter(|section| {
            let remove = section
                .lines()
                .any(|line| line.starts_with("entry = \"dspark_main/candidate-off/"));
            removed += usize::from(remove);
            !remove
        })
        .collect::<Vec<_>>();
    assert_eq!(removed, 1);
    sections.join("[[allow]]")
}

fn candidate_main_tp4_dspark_route_sweep() {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.set(false);
        }
    }
    ENABLED.set(true);
    let _reset = Reset;
    conformance::route_guard::sweep::<Served>("tp4-dspark", &allowlist(), env!("CARGO_PKG_NAME"));
}

#[test]
fn candidate_sweep_narrows_only_the_main_projection_allowance() {
    let changed = allowlist();
    assert!(!changed.contains("entry = \"dspark_main/candidate-off/"));
    assert!(changed.contains("entry = \"dspark_main/generic-below-range/"));
    assert!(changed.contains("entry = \"dspark_main/generic-above-range/"));
    assert!(profiles(include_str!("../../../../profiles.json")).contains("tp4-dspark"));
}

mod route_sweep {
    #[test]
    fn candidate_main_tp4_dspark() {
        super::candidate_main_tp4_dspark_route_sweep();
    }
}
