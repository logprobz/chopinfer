//! Resource evidence for the linear engine's admitted dimension.
use kernel_core::contract::{RecurrentPhase, RecurrentQuery};
use kernels_sm120::{
    Profile, ProfileNames,
    invariance::Invariance,
    linear::{Entry, Stage},
};

#[test]
fn scan_dimension_256_exceeds_shared_memory() {
    let mut entry = Entry {
        profile: Profile::synthetic(ProfileNames {
            name: "linear-resource-probe",
            tag: "linear_resource_probe",
        }),
        symbol: String::new(),
        params: Vec::new(),
        invariance: Invariance::NONE,
        stage: Stage::Scan,
        query: RecurrentQuery::GatedDeltaRule {
            key_heads: 1,
            value_heads: 1,
            key_dim: 128,
            value_dim: 128,
            phase: RecurrentPhase::Prefill,
            capacity: 64,
        },
    };
    assert_eq!(entry.dynamic_shared_bytes(), 100_136);
    if let RecurrentQuery::GatedDeltaRule {
        key_dim, value_dim, ..
    } = &mut entry.query
    {
        *key_dim = 256;
        *value_dim = 256;
    }
    assert_eq!(entry.dynamic_shared_bytes(), 157_480);
    assert!(entry.dynamic_shared_bytes() > 101_376);
    entry.stage = Stage::Prepare;
    assert_eq!(entry.dynamic_shared_bytes(), 104_224);
}
