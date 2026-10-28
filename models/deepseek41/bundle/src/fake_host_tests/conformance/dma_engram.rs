use super::*;

thread_local! { pub(super) static ENABLED: Cell<bool> = const { Cell::new(false) }; }

pub(super) fn profiles(default: &'static str) -> &'static str {
    if !ENABLED.get() {
        return default;
    }
    static CANDIDATE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CANDIDATE.get_or_init(|| {
        let mut profiles: serde_json::Value = serde_json::from_str(default).unwrap();
        for profile in profiles["profiles"].as_object_mut().unwrap().values_mut() {
            profile["model_options"]["prefill_dma_engram"] = "true".into();
        }
        serde_json::to_string(&profiles).unwrap()
    })
}

fn sweep(profile: &str) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.set(false);
        }
    }
    ENABLED.set(true);
    let _reset = Reset;
    conformance::route_guard::sweep::<Served>(
        profile,
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml")),
        env!("CARGO_PKG_NAME"),
    );
}

#[test]
fn route_sweep_dma_engram_ar() {
    sweep("tp4-ar");
}

#[test]
fn route_sweep_dma_engram_dspark() {
    sweep("tp4-dspark");
}
