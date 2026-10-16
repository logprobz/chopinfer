include!("../../bundle/sm120/build.rs");

#[test]
fn every_geometry_builds_the_encoder_entries_its_runtime_selects() {
    let _generate: fn() = main;
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bundle");
    for owner in geometry::MODELS {
        let mut owner = *owner;
        owner.geometry.encoders = true;
        let execution = model_mimo26_contract::geometry::GEOMETRIES
            .iter()
            .find(|g| g.id == owner.geometry.id)
            .unwrap();
        let (specs, _) = geometry_specs(&manifest, owner);
        let symbols = specs
            .iter()
            .flat_map(|cubin| &cubin.spec().entries)
            .map(|entry| entry.symbol.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        for entry in model_mimo26_contract::sm120::encoders::kernels_for(execution).unwrap() {
            assert!(
                symbols.contains(entry.symbol.as_str()),
                "{}: missing {}",
                execution.id,
                entry.symbol
            );
        }
        for entry in model_mimo26_contract::sm120::encoders::rows::kernels_for(execution).unwrap() {
            assert!(
                symbols.contains(entry.symbol.as_str()),
                "{}: missing {}",
                execution.id,
                entry.symbol
            );
        }
    }
}
