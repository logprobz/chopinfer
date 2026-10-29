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

fn candidate_main_tp4_dspark_route_sweep() {
    if std::env::var("CHOPINFER_ROUTE_SWEEP").is_ok_and(|tier| tier != "quick") {
        println!("tp4-dspark: route sweep covers the quick tier only");
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.set(false);
        }
    }
    ENABLED.set(true);
    let _reset = Reset;
    conformance::route_guard::sweep::<Served>(
        "tp4-dspark",
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
        env!("CARGO_PKG_NAME"),
    );
}

#[test]
fn candidate_tags_add_only_the_compact_range() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    assert!(crate::sm120::dspark_main::route_tags(&shape, false).is_empty());
    let tags = crate::sm120::dspark_main::route_tags(&shape, true);
    assert!(!tags.is_empty());
    for tag in tags {
        assert_eq!(tag.family, "dspark_main");
        assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
        assert_eq!(tag.wrappers, ["compact"]);
        assert_eq!(tag.admitted, [129..=511]);
    }
}

mod route_sweep {
    #[test]
    fn candidate_main_tp4_dspark() {
        super::candidate_main_tp4_dspark_route_sweep();
    }
}
