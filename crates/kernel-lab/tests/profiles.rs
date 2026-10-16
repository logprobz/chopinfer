//! Model-owned profile names and rank shapes in the composed lab catalog.

use kernels_sm120::Profile;
use model_mimo26_contract::sm120::config as mimo26;
use model_qwen38_contract::sm120::config as qwen38;

#[test]
fn profiles_keep_their_names_and_order() {
    let names = kernel_lab::catalogs()
        .profiles()
        .into_iter()
        .map(|profile| (profile.as_str(), profile.tag()))
        .collect::<Vec<_>>();
    assert_eq!(
        names[..5],
        [
            ("std-dense-moe", "std"),
            ("mimo-v26-flash", "mimo"),
            ("mimo-v26-flash-tp2", "mimo_tp2"),
            ("qwen38-flash", "qwen38"),
            ("qwen38-flash-tp2", "qwen38_tp2"),
        ]
    );
    assert_eq!(
        names[5..],
        kernel_lab::lab_shapes::PROFILES.map(|p| (p.as_str(), p.tag()))
    );
    for catalog in kernel_lab::MODELS {
        if catalog.name == "lab-shapes" {
            for profile in catalog.profiles {
                assert!(profile.model().is_none());
                assert!(profile.ranks().is_none());
                assert!(profile.rank_shape().is_none());
            }
            continue;
        }
        if catalog.profiles.is_empty() {
            // A catalog without profiles holds no per-profile entries: only
            // its projection inventory and declarations.
            assert!((catalog.gemm)().is_empty(), "{}", catalog.name);
            assert!((catalog.gemm_fused)().is_empty(), "{}", catalog.name);
            assert!((catalog.gemm_mid)().is_empty(), "{}", catalog.name);
            assert!((catalog.moe)().is_empty(), "{}", catalog.name);
            assert!((catalog.moe_routes)().is_empty(), "{}", catalog.name);
            assert!((catalog.attention)().is_empty(), "{}", catalog.name);
            assert!((catalog.gated)().is_empty(), "{}", catalog.name);
            assert!((catalog.sparse)().is_empty(), "{}", catalog.name);
            assert!((catalog.pooled_indexer)().is_empty(), "{}", catalog.name);
            assert!((catalog.linear)().is_empty(), "{}", catalog.name);
            assert!(
                (catalog.hyper_connection_rows)().is_empty(),
                "{}",
                catalog.name
            );
            continue;
        }
        let model = catalog.profiles[0].model().expect("model profile");
        for &profile in catalog.profiles {
            let ranks = profile.ranks().expect("model profile has ranks");
            assert_eq!(model.try_at(ranks), Some(profile));
            assert_eq!(profile, model.at(ranks));
            assert_eq!(
                (profile.model(), profile.ranks()),
                (Some(model), Some(ranks))
            );
            assert_eq!(profile.ranks_of(model), Some(ranks));
            assert_eq!(model.try_at(16), None);
            let shape = profile.rank_shape().unwrap();
            assert_eq!(shape.ranks(), ranks);
            shape.validate().unwrap();
        }
    }
    assert_eq!(mimo26::PROFILE.ranks_of(qwen38::MODEL), None);
    assert_eq!(Profile::StdDenseMoe.ranks_of(qwen38::MODEL), None);
    assert_eq!(Profile::StdDenseMoe.rank_shape(), None);
}
