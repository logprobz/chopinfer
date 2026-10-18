//! The cubins the build script compiles (`src/sm120/build.rs`), assembled
//! for the compiled geometries: each geometry's own cubins carry its kernel
//! tag, and every cubin a geometry needs is the same whichever other
//! geometries the build compiles, so a geometry added as data only appends
//! lock records (the bundle-stage geometry recipe runs this target with an
//! added geometry).

#[allow(dead_code)]
#[path = "../../src/sm120/build.rs"]
mod construction;

use std::collections::BTreeMap;
use std::path::Path;

use construction::{Selection, selection};
use model_deepseek41_contract::geometry::GEOMETRIES;

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The specs of a selection by cubin name, with their lock family.
fn by_name(selected: &Selection) -> BTreeMap<&str, (&str, &kernel_build::CubinSpec)> {
    let mut specs = BTreeMap::new();
    for (family, spec) in &selected.specs {
        let previous = specs.insert(spec.name.as_str(), (*family, spec));
        assert!(previous.is_none(), "cubin {} is selected twice", spec.name);
    }
    specs
}

/// Each geometry's own cubins (those that serve it alone) carry its kernel
/// tag and are selected; no two geometries own one cubin.
#[test]
fn each_geometry_s_own_cubins_carry_its_kernel_tag() {
    let all = selection(manifest(), GEOMETRIES);
    let specs = by_name(&all);
    let mut owner = BTreeMap::new();
    for (cubin, id) in &all.owners {
        let geometry = GEOMETRIES.iter().find(|g| g.id == *id).unwrap();
        assert!(
            cubin.contains(geometry.kernel_tag),
            "{cubin} serves {id} alone without its kernel tag {}",
            geometry.kernel_tag
        );
        assert!(
            specs.contains_key(cubin.as_str()),
            "{cubin} is not compiled"
        );
        assert!(owner.insert(cubin, id).is_none(), "{cubin} has two owners");
    }
}

/// The build names the geometry of every cubin it selects for one
/// geometry alone, so an instance of another geometry never loads it: a
/// cubin is a geometry's own when that geometry's build selects it and no
/// other geometry's does, and those are exactly the cubins the build lists
/// as serving one geometry. It needs two compiled geometries, as the
/// geometry recipe and the synthetic build have.
#[test]
fn the_build_names_the_geometry_of_each_geometry_s_own_cubin() {
    if GEOMETRIES.len() < 2 {
        eprintln!("one compiled geometry: ownership needs another to compare: skipped");
        return;
    }
    let mut selected = BTreeMap::<String, Vec<&str>>::new();
    for geometry in GEOMETRIES {
        let alone = selection(manifest(), std::slice::from_ref(geometry));
        for (_, spec) in &alone.specs {
            selected
                .entry(spec.name.clone())
                .or_default()
                .push(geometry.id);
        }
    }
    let mut derived: Vec<(String, &str)> = selected
        .into_iter()
        .filter(|(_, ids)| ids.len() == 1)
        .map(|(cubin, ids)| (cubin, ids[0]))
        .collect();
    derived.sort();
    let mut listed = selection(manifest(), GEOMETRIES).owners;
    listed.sort();
    assert_eq!(listed, derived);
}

/// Every cubin a geometry's build selects is the one the build of every
/// compiled geometry selects, in the same lock family with the same
/// generated source, entries and defines; every cubin of the full build is
/// one some geometry's build selects.
#[test]
fn a_geometry_s_cubins_do_not_depend_on_the_other_geometries() {
    let all = selection(manifest(), GEOMETRIES);
    let full = by_name(&all);
    let mut covered = std::collections::BTreeSet::new();
    for geometry in GEOMETRIES {
        let alone = selection(manifest(), std::slice::from_ref(geometry));
        for (name, (family, spec)) in by_name(&alone) {
            let at = format!("{} {name}", geometry.id);
            let Some((full_family, full_spec)) = full.get(name) else {
                panic!("{at}: not in the build of every geometry");
            };
            assert_eq!(family, *full_family, "{at}: lock family");
            let (a, b) = (
                kernel_build::generate(spec).unwrap(),
                kernel_build::generate(full_spec).unwrap(),
            );
            assert_eq!(a.translation_unit, b.translation_unit, "{at}: source");
            assert_eq!(a.header, b.header, "{at}: header");
            assert_eq!(
                format!("{:?}", spec.entries),
                format!("{:?}", full_spec.entries),
                "{at}: entries"
            );
            assert_eq!(spec.defines, full_spec.defines, "{at}: defines");
            covered.insert(name.to_owned());
        }
    }
    let missing: Vec<_> = full
        .keys()
        .filter(|name| !covered.contains(**name))
        .collect();
    assert!(
        missing.is_empty(),
        "built for no single geometry: {missing:?}"
    );
    eprintln!("{} cubins over {} geometries", full.len(), GEOMETRIES.len());
}
