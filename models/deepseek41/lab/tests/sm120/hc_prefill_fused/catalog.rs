//! Lab-only fusion and the entries selected by the current prefill backend.
use kernel_build::CubinSpec;
use kernel_core::{
    contract::LaunchPlan,
    sinkhorn_expand_mix::{SinkhornExpandMixQuery, SinkhornRoundNormQuery},
};
use kernels_sm120::{
    Profile, ProfileNames,
    sinkhorn_hc::{
        Geometry, lagged, lagged_tiled,
        prefill_fused::{self, Entry, Mutation},
        prefill_warp,
    },
};
use model_deepseek41_contract::{geometry::FLASH, hc, rows, serving::PREFILL_TOKENS};

pub const FAMILY: &str = "hc_prefill_fused";
pub const MODULE: &str = "hc_prefill_fused_sm120";
pub fn entries() -> Vec<Entry> {
    [1, 4, 8]
        .into_iter()
        .map(|tile_rows| Entry {
            query: SinkhornRoundNormQuery {
                mixer: SinkhornExpandMixQuery {
                    mix: hc::mix(&FLASH),
                },
                norm: rows::norm_quantizers(&FLASH)[0],
            },
            capacity: PREFILL_TOKENS,
            tile_rows,
            symbol: format!("chop_hc_prefill_fused_r{tile_rows}_v1"),
            mutation: Mutation::None,
        })
        .collect()
}
pub fn controls() -> Vec<Entry> {
    [
        (Mutation::ReverseExpansion, "reverse_expansion"),
        (Mutation::RoundBeforeWeight, "round_before_weight"),
        (Mutation::ClearStatus, "clear_status"),
    ]
    .into_iter()
    .map(|(mutation, name)| {
        let mut e = entries().remove(1);
        e.mutation = mutation;
        e.symbol = format!("chop_hc_prefill_fused_{name}_v1");
        e
    })
    .collect()
}
pub fn specs() -> Vec<CubinSpec> {
    vec![
        prefill_fused::cubin_spec(
            MODULE,
            &entries().into_iter().chain(controls()).collect::<Vec<_>>(),
        )
        .unwrap(),
    ]
}
fn profile() -> Profile {
    Profile::synthetic(ProfileNames {
        name: "deepseek41",
        tag: FLASH.kernel_tag,
    })
}
fn geometry() -> Geometry {
    Geometry {
        mix: hc::mix(&FLASH),
        capacity: PREFILL_TOKENS,
    }
}
pub fn expand(rows: u32) -> (lagged::Entry, LaunchPlan) {
    let e = lagged::entries(profile(), geometry(), false)
        .into_iter()
        .find(|e| e.op == lagged::Op::Expand)
        .unwrap();
    let p = e.plan(rows, 0).unwrap();
    (e, p)
}
pub fn mix(rows: u32) -> (lagged::Entry, LaunchPlan) {
    if [64, 192, 2240, 2496, 2624, 8192].contains(&rows) {
        let e = prefill_warp::pipeline(
            profile(),
            geometry(),
            8,
            if rows < 256 { 8 } else { 4 },
            false,
        );
        let p = e.plan(rows, 0).unwrap();
        (e.reference, p)
    } else if rows >= 256 {
        let e = lagged_tiled::entry(profile(), geometry(), 8, 256, false);
        let p = e.plan(rows, 0).unwrap();
        (e.reference, p)
    } else {
        let e = lagged::entries(profile(), geometry(), false)
            .into_iter()
            .find(|e| e.op == lagged::Op::MixCollapse { probe: false })
            .unwrap();
        let p = e.plan(rows, 0).unwrap();
        (e, p)
    }
}
pub fn norm() -> kernels_sm120::row::rmsnorm_prefill::Entry {
    bundle_deepseek41::sm120::norm_prefill::entries()
        .unwrap()
        .pop()
        .unwrap()
}
pub const ROWS: [u32; 8] = [1, 7, 64, 192, 320, 2496, 8191, 8192];
