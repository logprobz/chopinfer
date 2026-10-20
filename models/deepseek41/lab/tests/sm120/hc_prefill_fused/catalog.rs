//! Lab-only fusion and the entries selected by the current prefill backend.
use kernel_build::CubinSpec;
use kernel_core::{
    contract::LaunchPlan,
    sinkhorn_expand_mix::{SinkhornExpandMixQuery, SinkhornRoundNormQuery},
};
use kernels_sm120::{
    Profile, ProfileNames,
    sinkhorn_hc::{
        Geometry, lagged, lagged_tiled, prefill_expand_mix,
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
pub const EXPAND_MIX_FAMILY: &str = "hc_prefill_expand_mix";
pub const EXPAND_MIX_MODULE: &str = "hc_prefill_expand_mix_sm120";
/// Expansion then the unchanged prefill pipeline in one launch, at the
/// served pipeline's tile and two other in-flight footprints.
pub fn expand_mix_entries() -> Vec<prefill_expand_mix::Entry> {
    [(4, 4), (8, 4), (8, 8)]
        .into_iter()
        .map(|(tile_rows, chains)| prefill_expand_mix::Entry {
            query: SinkhornExpandMixQuery {
                mix: hc::mix(&FLASH),
            },
            capacity: PREFILL_TOKENS,
            tile_rows,
            chains,
            symbol: format!("chop_hc_prefill_expand_mix_r{tile_rows}_c{chains}_v1"),
            mutation: prefill_expand_mix::Mutation::None,
        })
        .collect()
}
pub fn expand_mix_controls() -> Vec<prefill_expand_mix::Entry> {
    use prefill_expand_mix::Mutation as M;
    [
        (M::ReverseExpansion, "reverse_expansion"),
        (M::FusedPost, "fused_post"),
        (M::RoundMixed, "round_mixed"),
    ]
    .into_iter()
    .map(|(mutation, name)| {
        let mut e = expand_mix_entries().remove(1);
        e.mutation = mutation;
        e.symbol = format!("chop_hc_prefill_expand_mix_{name}_v1");
        e
    })
    .collect()
}
pub fn expand_mix_specs() -> Vec<CubinSpec> {
    vec![
        prefill_expand_mix::cubin_spec(
            EXPAND_MIX_MODULE,
            &expand_mix_entries()
                .into_iter()
                .chain(expand_mix_controls())
                .collect::<Vec<_>>(),
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
/// A lab arm against the served chain: the earlier one-kernel composition
/// (expansion, mixer and norm), or expansion then the unchanged mixer in
/// one launch followed by the served norm.
#[derive(Clone, Debug)]
pub enum Candidate {
    Fused(Entry),
    ExpandMix(prefill_expand_mix::Entry),
}
impl Candidate {
    pub fn symbol(&self) -> &str {
        match self {
            Candidate::Fused(e) => &e.symbol,
            Candidate::ExpandMix(e) => &e.symbol,
        }
    }
    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, kernel_core::contract::Refusal> {
        match self {
            Candidate::Fused(e) => e.plan(rows, live),
            Candidate::ExpandMix(e) => e.plan(rows, live),
        }
    }
    /// The case-label token of the arm's tile.
    pub fn label(&self) -> String {
        match self {
            Candidate::Fused(e) => format!("r{}", e.tile_rows),
            Candidate::ExpandMix(e) => format!("r{}c{}", e.tile_rows, e.chains),
        }
    }
    /// Launches of the candidate arm for one boundary.
    pub fn launches(&self) -> u32 {
        match self {
            Candidate::Fused(_) => 1,
            Candidate::ExpandMix(_) => 2,
        }
    }
    /// A mutation that only shows over a sticky prior row status.
    pub fn needs_sticky(&self) -> bool {
        matches!(self, Candidate::Fused(e) if e.mutation == Mutation::ClearStatus)
    }
}
/// The matrix's arms and controls, and the staged module that holds them.
pub fn candidates() -> Vec<Candidate> {
    expand_mix_entries()
        .into_iter()
        .map(Candidate::ExpandMix)
        .collect()
}
pub fn candidate_controls() -> Vec<Candidate> {
    expand_mix_controls()
        .into_iter()
        .map(Candidate::ExpandMix)
        .collect()
}
pub const CANDIDATE_MODULE: &str = EXPAND_MIX_MODULE;
/// The served input norm's epsilon, which both arms pass.
pub fn norm_eps_bits() -> u32 {
    rows::norm_quantizers(&FLASH)[0].norm.epsilon_bits
}
