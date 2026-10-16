use std::collections::BTreeMap;

use model_mimo26_contract::sm120::{PROFILES, config, moe};
use model_mimo26_contract::{Tp, geometry::GEOMETRIES, serving, state};

fn values(defines: Vec<(String, String)>) -> BTreeMap<String, u64> {
    defines
        .into_iter()
        .map(|(key, value)| (key, value.trim_end_matches("ull").parse().unwrap()))
        .collect()
}

#[test]
fn cuda_ring_dimensions_match_request_state() {
    for profile in PROFILES {
        let shape = profile.rank_shape().unwrap();
        let owner = model_mimo26_contract::sm120::geometry::owner(profile).unwrap();
        let g = GEOMETRIES
            .iter()
            .find(|g| g.id == owner.geometry.id)
            .unwrap();
        let tp = Tp::of_geometry(g, shape.ranks()).unwrap();
        let defines = values(config::ring_defines_for(*g, shape));
        let value = |name: &str| {
            *defines
                .get(name)
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        assert_eq!(value("CHOP_MIMO_RING_TOKENS"), u64::from(g.window()));
        assert_eq!(
            value("CHOP_MIMO_RING_SLOTS"),
            u64::from(serving::DECODE_ENTRY_ROWS)
        );
        assert_eq!(value("CHOP_MIMO_SWA_LAYERS"), g.sliding_layers() as u64);
        let key = value("CHOP_MIMO_SWA_KV_HEADS") * value("CHOP_MIMO_SWA_QK_DIM") * 2;
        let val = value("CHOP_MIMO_SWA_KV_HEADS") * value("CHOP_MIMO_SWA_V_DIM") * 2;
        assert_eq!(key, tp.kv_heads(false) * tp.qk_dim(false) * 2);
        assert_eq!(val, tp.kv_heads(false) * tp.value_dim(false) * 2);
        let request = state::request_state(tp, &serving::kv_formats(tp)[0], state::Drafter::None);
        let ring = request
            .buffers
            .iter()
            .find(|buffer| buffer.name == "swa.ring")
            .unwrap();
        let slots = value("CHOP_MIMO_RING_SLOTS") as u32;
        let extent = bundle_runtime::state::Extent {
            served: slots,
            bucket: slots,
            verify_rows: 0,
        };
        assert_eq!(
            2 * value("CHOP_MIMO_RING_WORKING_BYTES"),
            ring.bytes(extent)
        );
        assert!(value("CHOP_MIMO_RING_THREADS") >= (key + val) / 16);
        assert_eq!(
            value("CHOP_MIMO_PREFIX_THREADS"),
            u64::from(config::PREFIX_THREADS)
        );
    }
}

#[test]
fn cuda_drafter_and_expert_dimensions_match_catalogs() {
    for profile in PROFILES {
        let shape = profile.rank_shape().unwrap();
        let owner = model_mimo26_contract::sm120::geometry::owner(profile).unwrap();
        let g = GEOMETRIES
            .iter()
            .find(|g| g.id == owner.geometry.id)
            .unwrap();
        let tp = Tp::of_geometry(g, shape.ranks()).unwrap();
        let draft = values(config::dflash_defines_for(*g, shape));
        assert_eq!(
            draft["CHOP_MIMO_DFLASH_HIDDEN"],
            u64::from(g.dflash().hidden_size)
        );
        assert_eq!(draft["CHOP_MIMO_DFLASH_KV_HEADS"], tp.dflash_kv_heads());
        assert_eq!(
            draft["CHOP_MIMO_DFLASH_HEAD_DIM"],
            u64::from(g.dflash().head_dim)
        );
        let slots = u32::try_from(draft["CHOP_MIMO_DFLASH_SLOTS"]).unwrap();
        let items = bundle_runtime::geometry::Items::new(slots).unwrap();
        assert_eq!(items.capacity(), slots);
        assert!(bundle_runtime::geometry::Items::new(slots + 1).is_none());
        let request = state::request_state(tp, &serving::kv_formats(tp)[0], state::Drafter::Dflash);
        let cache = request
            .buffers
            .iter()
            .find(|buffer| buffer.name == "dflash.context")
            .unwrap();
        assert_eq!(cache.entries, bundle_runtime::state::Entries::Bucket);
        let extent = bundle_runtime::state::Extent {
            served: slots,
            bucket: items.capacity(),
            verify_rows: 0,
        };
        assert_eq!(
            cache.bytes(extent),
            u64::from(slots) * cache.layers * cache.planes.iter().sum::<u64>()
        );
        assert_eq!(
            cache.planes,
            vec![
                draft["CHOP_MIMO_DFLASH_CACHE_TOKENS"]
                    * tp.dflash_kv_heads()
                    * draft["CHOP_MIMO_DFLASH_HEAD_DIM"]
                    * 2;
                2
            ]
        );
    }
    let defines = values(moe::fused_down_combine_cubin_spec().defines);
    assert_eq!(defines["CHOP_MIMO_HIDDEN"], u64::from(moe::SHAPE.hidden));
    assert_eq!(
        defines["CHOP_MIMO_EXPERT_INTERMEDIATE"],
        u64::from(moe::SHAPE.intermediate)
    );
    assert_eq!(defines["CHOP_MIMO_EXPERTS"], u64::from(moe::SHAPE.experts));
    assert_eq!(defines["CHOP_MIMO_TOP_K"], u64::from(moe::SHAPE.top_k));
}
