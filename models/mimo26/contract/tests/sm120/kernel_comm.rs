use kernel_core::contract::StaticQuery;
use kernels_sm120::comm::{CATALOG_RANKS, catalog, own_catalog_for};
use model_mimo26_contract::sm120::{
    catalogs,
    comm::{shapes_for, shapes_for_geometry},
    geometry::MODELS,
};

#[test]
fn catalog_covers_rank_shapes() {
    let specs = catalog(catalogs());
    let expected = CATALOG_RANKS
        .into_iter()
        .map(|ranks| 22 * MODELS.len() + own_catalog_for(ranks).unwrap().len())
        .sum::<usize>();
    assert_eq!(specs.len(), expected);
    // The query's rank count keys every entry and sizes the vocabulary
    // shard; TP4's entries are the former catalog's.
    for ranks in CATALOG_RANKS {
        for owner in MODELS {
            let entries = shapes_for_geometry(*owner, ranks).unwrap();
            assert_eq!(entries.len(), 22);
            for entry in &entries {
                assert!(
                    specs
                        .iter()
                        .any(|spec| spec.instantiation == entry.instantiation)
                );
                let StaticQuery::Comm {
                    family,
                    ranks: keyed,
                    elements_capacity,
                    ..
                } = &entry.instantiation.key.query
                else {
                    panic!("comm catalog key");
                };
                assert_eq!(*keyed, ranks);
                if family.starts_with(&format!("{}_vocab", owner.geometry.kernel_tag)) {
                    assert!(
                        elements_capacity
                            .is_multiple_of(u64::from(owner.geometry.vocab()) / u64::from(ranks))
                    );
                }
            }
        }
    }
    assert!(shapes_for(4).unwrap().iter().any(|entry| matches!(
        entry.instantiation.key.query,
        StaticQuery::Comm {
            family: "mimo_vocab",
            ranks: 4,
            elements_capacity: 38_144,
            ..
        }
    )));
    assert!(shapes_for(0).is_err() && shapes_for(9).is_err() && shapes_for(3).is_err());
}
