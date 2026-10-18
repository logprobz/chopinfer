//! Symbols shared by DSpark head preparation and cubin construction.

pub const FAMILY: &str = "dspark_heads";
pub const CUBIN: &str = "deepseek41_dspark_heads_sm120";
pub const MARKOV: &str = "chop_deepseek41_dspark_markov_v1";
pub const CONFIDENCE: &str = "chop_deepseek41_dspark_confidence_v1";
pub const THREADS: u32 = 128;

pub fn rounding_symbol(
    g: &model_deepseek41_contract::geometry::Geometry,
    kind: kernels_sm120::row::rounding::Rounding,
) -> String {
    let kind = match kind {
        kernels_sm120::row::rounding::Rounding::Widen => "widen",
        kernels_sm120::row::rounding::Rounding::Round => "round",
    };
    format!("chop_{}_dspark_markov_{kind}_v1", g.kernel_tag)
}
