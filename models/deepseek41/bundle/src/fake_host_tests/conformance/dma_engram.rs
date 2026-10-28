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
    conformance::route_guard::sweep::<Served>(profile, &allowlist(), env!("CARGO_PKG_NAME"));
}

// Keep the default-off allowances in the package's production allowlist.
// Candidate sweeps remove exactly the two fallbacks the option eliminates.
fn allowlist() -> String {
    let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml"));
    let mut removed = 0;
    let result = source
        .split("[[allow]]")
        .filter(|section| {
            let remove = section
                .lines()
                .any(|line| line.trim() == "entry = \"engram/sync/prefill_schedule\"");
            removed += usize::from(remove);
            !remove
        })
        .collect::<Vec<_>>()
        .join("[[allow]]");
    assert_eq!(
        removed, 2,
        "candidate must remove only both DMA fallback allowances"
    );
    result
}

#[test]
fn candidate_allowlist_keeps_other_entries() {
    let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/route-allowlist.toml"));
    let candidate = allowlist();
    assert!(!candidate.contains("engram/sync/prefill_schedule"));
    assert_eq!(
        candidate.matches("[[allow]]").count() + 2,
        source.matches("[[allow]]").count()
    );
}

mod route_sweep {
    #[test]
    fn dma_engram_ar() {
        super::sweep("tp4-ar");
    }

    #[test]
    fn dma_engram_dspark() {
        super::sweep("tp4-dspark");
    }
}
