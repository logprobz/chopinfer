//! The contract's per-rank sizes equal `kernel_core`'s rank-shape derivation
//! from MiMo's configuration values at every served rank count, until the
//! contract reads them from it (`tp.rs` feeds the packing revision, so its
//! copies stay until the rank-image path retires).

use crate::Tp;
use crate::sm120::config::{DENSE, DFLASH, DFLASH_CONTEXT, GA, HIDDEN, MOE, SWA};
use crate::sm120::profile;
use crate::topology::{kernel_tag, row_infix, topology_id};

#[test]
fn contract_sizes_are_the_rank_shapes() {
    for tp in Tp::SUPPORTED {
        let shape = profile(tp).rank_shape().expect("a served profile");
        shape.validate().unwrap();
        let (ga, swa, dflash) = (
            shape.validate_attention(GA).unwrap(),
            shape.validate_attention(SWA).unwrap(),
            shape.validate_attention(DFLASH).unwrap(),
        );
        assert_eq!(tp.q_heads(), u64::from(ga.q_heads), "{tp:?}");
        assert_eq!(tp.kv_heads(true), u64::from(ga.kv_heads), "{tp:?}");
        assert_eq!(tp.kv_heads(false), u64::from(swa.kv_heads), "{tp:?}");
        assert_eq!(tp.vocab(), u64::from(shape.vocab_shard()), "{tp:?}");
        assert_eq!(
            tp.dense_intermediate(),
            u64::from(shape.dense(DENSE).intermediate)
        );
        assert_eq!(
            tp.expert_intermediate(),
            u64::from(shape.experts(MOE).intermediate)
        );
        assert_eq!(tp.mtp_eh_k(), u64::from(shape.split(2 * HIDDEN)), "{tp:?}");
        assert_eq!(tp.dflash_q_heads(), u64::from(dflash.q_heads), "{tp:?}");
        assert_eq!(tp.dflash_kv_heads(), u64::from(dflash.kv_heads), "{tp:?}");
        assert_eq!(
            tp.dflash_fc_k(),
            u64::from(shape.split(DFLASH_CONTEXT)),
            "{tp:?}"
        );
    }
}

/// The package identities are a pinned table; the kernel names, which the
/// topology derives from its geometry, are the profile's at every geometry.
#[test]
fn identities_follow_the_profiles() {
    for geometry in crate::geometry::GEOMETRIES {
        for ranks in Tp::SUPPORTED {
            let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
            assert_eq!(kernel_tag(tp), profile(tp).tag(), "{tp:?}");
        }
    }
    for tp in Tp::SUPPORTED {
        let profile = profile(tp);
        let pinned = if tp == Tp::TP4 {
            format!("{}-tp4", profile.as_str())
        } else {
            profile.as_str().to_owned()
        };
        assert_eq!(topology_id(tp), pinned);
    }
    assert_eq!(Tp::SUPPORTED.map(row_infix), ["_tp2", ""]);
}
