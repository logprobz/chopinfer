#![forbid(unsafe_code)]

//! The boundary this crate exists to hold.
//!
//! A model compiler links this SDK, not the other way round. If the SDK ever
//! gained a model, a kernel catalog, or a toolchain dependency, the emitter
//! would stop being the one thing every model can compile through, and these
//! two checks are what say so at build time rather than at review time.

use std::collections::BTreeSet;

const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// The complete dependency set: contracts, digests, JSON and no-follow file access.
#[test]
fn the_emitter_depends_on_no_model_and_no_toolchain() {
    let declared = dependency_names();

    assert_eq!(
        declared,
        BTreeSet::from([
            "kernel-core".to_owned(),
            "projection-spec".to_owned(),
            "runtime-bundle".to_owned(),
            "runtime-bundle-abi".to_owned(),
            "runtime-weights".to_owned(),
            "serde".to_owned(),
            "serde_json".to_owned(),
            "sha2".to_owned(),
        ]),
        "the emitter's dependencies changed; a model, kernel catalog, or device \
         dependency here would put model knowledge inside the SDK"
    );
}

fn dependency_names() -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_dependencies = false;
    for line in CARGO_TOML.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_dependencies = line.contains("dependencies]");
            continue;
        }
        if !in_dependencies || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, _) = line
            .split_once('=')
            .expect("every dependency line assigns a requirement");
        // `name.workspace = true` and `name = { ... }` both name one crate.
        let name = key.trim().split('.').next().expect("a dependency is named");
        names.insert(name.to_owned());
    }
    names
}
