//! Exercise production encoder admission using declared cubin entries, without CUDA.
use model_mimo26_contract::{geometry, topology::Tp};
use std::sync::LazyLock;

#[path = "../../../bundle/src/encoder_availability.rs"]
mod encoder_availability;

struct Cubin {
    name: &'static str,
    symbols: Vec<&'static str>,
}

fn intern(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

static CUBINS: LazyLock<Vec<Cubin>> = LazyLock::new(|| {
    let mut cubins = Vec::new();
    let mut seen = std::collections::BTreeMap::new();
    for g in geometry::GEOMETRIES.iter().filter(|g| g.encoders) {
        let specs = kernels_sm120::encoder::cubin_specs(
            model_mimo26_contract::sm120::encoders::kernels_for(g).unwrap(),
        )
        .unwrap();
        for mut spec in specs {
            model_mimo26_contract::sm120::encoders::deduplicate(
                &mut spec.entries,
                &mut seen,
                |entry| &entry.symbol,
            );
            let symbols: Vec<_> = spec
                .entries
                .into_iter()
                .map(|entry| intern(entry.symbol))
                .collect();
            if symbols.is_empty() {
                continue;
            }
            cubins.push(Cubin {
                name: intern(if g.id == geometry::V26_FLASH.id {
                    spec.name
                } else {
                    format!("{}_{}", g.kernel_tag, spec.name)
                }),
                symbols,
            });
        }
        let mut entries = model_mimo26_contract::sm120::encoders::rows::kernels_for(g).unwrap();
        model_mimo26_contract::sm120::encoders::deduplicate(&mut entries, &mut seen, |entry| {
            &entry.symbol
        });
        let symbols: Vec<_> = entries
            .into_iter()
            .map(|entry| intern(entry.symbol))
            .collect();
        if symbols.is_empty() {
            continue;
        }
        cubins.push(Cubin {
            name: intern(if g.id == geometry::V26_FLASH.id {
                model_mimo26_contract::sm120::encoders::rows::CUBIN.into()
            } else {
                format!("{}_encoder", g.kernel_tag)
            }),
            symbols,
        });
    }
    cubins
});

static GEOMETRY_CUBINS: LazyLock<Vec<(&'static str, &'static str, u32)>> = LazyLock::new(|| {
    CUBINS
        .iter()
        .flat_map(|cubin| {
            model_mimo26_contract::sm120::encoders::cubin_members(&cubin.symbols)
                .unwrap()
                .into_iter()
                .map(|id| (cubin.name, id, 0))
                .collect::<Vec<_>>()
        })
        .collect()
});

#[test]
fn encoder_symbols_are_embedded_once() {
    let mut seen = std::collections::BTreeSet::new();
    for cubin in CUBINS.iter() {
        for symbol in &cubin.symbols {
            assert!(seen.insert(symbol), "duplicate encoder symbol {symbol}");
        }
    }
    assert!(!seen.is_empty());
}

#[test]
fn every_geometry_and_rank_has_all_required_encoder_entries() {
    for g in geometry::GEOMETRIES {
        for rank in Tp::SUPPORTED {
            let tp = Tp::of_geometry(g, rank.ranks()).unwrap();
            assert_eq!(
                encoder_availability::encoders_available(tp),
                g.encoders,
                "{tp:?}"
            );
        }
    }
}

#[test]
fn shared_audio_uses_flash_cubin_membership() {
    if std::env::var_os("CHOPINFER_MIMO26_GEOMETRY_DIRS").is_none() {
        return;
    }
    let g = geometry::GEOMETRIES
        .iter()
        .find(|g| g.id == "shared-audio")
        .expect("shared-audio gate fixture");
    for rank in Tp::SUPPORTED {
        let tp = Tp::of_geometry(g, rank.ranks()).unwrap();
        assert!(encoder_availability::cubin_serves(
            model_mimo26_contract::sm120::encoders::rows::CUBIN,
            tp
        ));
        assert!(encoder_availability::encoders_available(tp));
    }
}
