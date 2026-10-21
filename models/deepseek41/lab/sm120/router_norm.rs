//! The router-in-norm lab cubin: Flash's served MoE-input prefill norm with
//! its router certificate (`rmsnorm_prefill` with `certify`), served and
//! certifying every row (a lab mutation), and the served router entry's
//! certified projection (`f32_projection::certified`, 32-row tiles), which
//! replace the served norm, the zero-component prepare and its projection.
use bundle_deepseek41::sm120::{f32_zero, norm_prefill};
use kernels_sm120::f32_projection::certified;
use kernels_sm120::row::rmsnorm_prefill::{self, Mutation};

pub const FAMILY: &str = "router_norm_lab";
pub const CUBIN: &str = "router_norm_lab_deepseek41_sm120";
/// The served router's tile rows (`f32_zero`).
pub const TILE_ROWS: u32 = 32;

/// The served cached norm: the MoE input's.
pub fn served_norm() -> rmsnorm_prefill::Entry {
    norm_prefill::entries()
        .expect("served prefill norm entries")
        .into_iter()
        .find(|e| e.cache_groups)
        .expect("the cached prefill norm")
}

/// The served norm, certifying: `none` or `certify_all`.
pub fn certifying_norm(mutation: Mutation) -> rmsnorm_prefill::Entry {
    let mut e = served_norm();
    let suffix = match mutation {
        Mutation::None => "certified",
        Mutation::CertifyAll => "certify_all",
        other => panic!("{other:?} is not a certifying mutation"),
    };
    e.reference.symbol = format!(
        "{}_{suffix}_v1",
        e.reference
            .symbol
            .trim_end_matches("_v1")
            .replacen("chop_", "chop_lab_", 1)
    );
    e.certify = true;
    e.mutation = mutation;
    e.validate().expect("certifying prefill norm");
    e
}

pub fn router() -> kernels_sm120::f32_projection::Entry {
    f32_zero::entry()
}

pub fn certified_symbol() -> String {
    certified::symbol(&router(), TILE_ROWS)
}

pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    let mut spec =
        certified::cubin_spec(CUBIN, &[router()], &[TILE_ROWS]).expect("certified router");
    let norms = rmsnorm_prefill::cubin_spec(
        "unused".into(),
        &[
            certifying_norm(Mutation::None),
            certifying_norm(Mutation::CertifyAll),
        ],
    )
    .expect("certifying norms");
    spec.headers.extend(norms.headers);
    spec.entries.extend(norms.entries);
    vec![spec]
}
