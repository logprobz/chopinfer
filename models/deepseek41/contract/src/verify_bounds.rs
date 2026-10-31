//! The widest verify step each bounded Flash TP4 decode-route family
//! serves. Every bound starts at `VERIFY_DECODE_ROWS`; a family that admits
//! wider steps raises its own bound and leaves the others in place. An
//! instance serves each bound capped by its verify rows. Bounds of
//! registered entries and tuning keys are the bound itself.
use crate::serving::VERIFY_DECODE_ROWS;

/// The draft-rows op of a row-bucketed row op: decode and verify rows
/// above the widest decode bucket.
pub const ROW_BUCKETS: u32 = VERIFY_DECODE_ROWS;
/// The parallel and cached decode ceil quantizers and the decode SwiGLU
/// quantizer, their registered capacities and capacity knobs.
pub const CEIL: u32 = VERIFY_DECODE_ROWS;
/// The fused input and query norm quantizers.
pub const NORM_QUANTIZE: u32 = VERIFY_DECODE_ROWS;
/// The HC round and expand-mix fusion and its finish norm.
pub const HC_FINISH: u32 = VERIFY_DECODE_ROWS;
/// The HC split mixer of wide verify steps.
pub const HC_SPLIT: u32 = VERIFY_DECODE_ROWS;
/// The selected-key latent cache.
pub const LATENT_CACHE: u32 = VERIFY_DECODE_ROWS;
/// The wide latent query rotation.
pub const LATENT_QUERY: u32 = VERIFY_DECODE_ROWS;
/// The wide parallel index query.
pub const INDEX_QUERY: u32 = VERIFY_DECODE_ROWS;
/// The cached exact F32 verify projections.
pub const F32_CACHED: u32 = VERIFY_DECODE_ROWS;
/// The verify dense folds.
pub const DENSE_FOLD: u32 = VERIFY_DECODE_ROWS;
/// Engram's wide verify matrix fold.
pub const ENGRAM_FOLD: u32 = VERIFY_DECODE_ROWS;
/// The direct MoE buckets and their router finish.
pub const MOE_DIRECT: u32 = VERIFY_DECODE_ROWS;
/// The streamed MoE tuning keys.
pub const MOE_STREAMED: u32 = VERIFY_DECODE_ROWS;

#[cfg(test)]
mod tests {
    use super::*;

    /// Splitting the bounds moved no family: each is still the bounded
    /// decode route's rows.
    #[test]
    fn every_family_keeps_the_decode_route_rows() {
        for bound in [
            ROW_BUCKETS,
            CEIL,
            NORM_QUANTIZE,
            HC_FINISH,
            HC_SPLIT,
            LATENT_CACHE,
            LATENT_QUERY,
            INDEX_QUERY,
            F32_CACHED,
            DENSE_FOLD,
            ENGRAM_FOLD,
            MOE_DIRECT,
            MOE_STREAMED,
        ] {
            assert_eq!(bound, 64);
        }
    }
}
