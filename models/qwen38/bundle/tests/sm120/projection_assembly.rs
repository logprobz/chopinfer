#[allow(dead_code)]
#[path = "../../src/sm120/build.rs"]
mod construction;
use construction::projection_selection as assembly;

#[test]
fn generated_cubins_preserve_every_locked_entry_and_wrapper() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (baseline, _, geometry_of) = construction::selection_inputs(root);
    let assembled = assembly::assemble(&baseline, &geometry_of).unwrap();
    let locked: serde_json::Value =
        serde_json::from_str(include_str!("../../kernels.lock")).unwrap();
    assert_eq!(assembled.specs().len(), locked.as_array().unwrap().len());
    for spec in assembled.specs() {
        let old = baseline
            .specs()
            .iter()
            .find(|s| s.name == spec.name)
            .unwrap();
        let a = kernel_build::generate(old).unwrap();
        let b = kernel_build::generate(spec).unwrap();
        assert_eq!(a.translation_unit, b.translation_unit, "{}", spec.name);
        assert_eq!(a.header, b.header, "{}", spec.name);
        let expected = locked
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["name"] == spec.name)
            .unwrap();
        let entries = spec.entries.iter().map(|e|serde_json::json!({"symbol":e.symbol,"params":e.params,"max_threads":e.max_threads,"min_blocks":e.min_blocks,"allowed_local_bytes":e.allowed_local_bytes})).collect::<Vec<_>>();
        assert_eq!(
            serde_json::json!(entries),
            expected["entries"],
            "{}",
            spec.name
        );
        assert_eq!(spec.defines, old.defines);
    }
    assert_eq!(assembled.declarations(), baseline.declarations());
    let mut ledger = model_qwen38_contract::sm120::generated::ledger()
        .unwrap()
        .to_vec();
    let (entries, identities) = assembly::replacements(&baseline, &geometry_of, &ledger).unwrap();
    ledger
        .last_mut()
        .unwrap()
        .placement
        .as_mut()
        .unwrap()
        .position += 1;
    assert!(
        kernels_sm120::precision::generated::placement::assemble(
            &baseline,
            &entries,
            &identities,
            &ledger
        )
        .is_err()
    );
}

#[test]
fn unadopted_geometries_keep_catalog_entries_without_generated_schedules() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (baseline, _, geometry_of) = construction::selection_inputs(root);
    // New geometries have catalog entries but no adoption or schedule records.
    let (entries, identities) = assembly::replacements(&baseline, &geometry_of, &[]).unwrap();
    assert!(
        entries.is_empty(),
        "unadopted entries must use their catalog schedules"
    );
    assert!(identities.is_empty());
    let assembled = kernels_sm120::precision::generated::placement::assemble(
        &baseline,
        &entries,
        &identities,
        &[],
    )
    .unwrap();
    assert_eq!(assembled.specs().len(), baseline.specs().len());
    for (actual, expected) in assembled.specs().iter().zip(baseline.specs()) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.defines, expected.defines);
        let actual = kernel_build::generate(actual).unwrap();
        let expected = kernel_build::generate(expected).unwrap();
        assert_eq!(actual.translation_unit, expected.translation_unit);
        assert_eq!(actual.header, expected.header);
    }
    assert_eq!(assembled.declarations(), baseline.declarations());
}

#[test]
fn adopted_cubins_require_geometry_ownership() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (baseline, _, mut geometry_of) = construction::selection_inputs(root);
    let ledger = model_qwen38_contract::sm120::generated::ledger().unwrap();
    let cubin = &ledger.last().unwrap().placement.as_ref().unwrap().cubin;
    geometry_of.remove(cubin).unwrap();
    let error = assembly::assemble(&baseline, &geometry_of).err().unwrap();
    assert_eq!(
        error,
        format!("missing geometry owner for adopted cubin {cubin}")
    );
}
