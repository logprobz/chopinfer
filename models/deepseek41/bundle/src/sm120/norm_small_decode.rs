//! Qualified cached norms for the Flash TP4 target's small decode buckets.
use kernel_core::backend::Backend;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;

impl crate::Kernels {
    pub(crate) fn small_decode_norm_backend(
        shape: &crate::shape::Shape,
    ) -> Option<Box<dyn Backend>> {
        if shape.geometry.id != FLASH.id || shape.ranks() != 4 {
            return None;
        }
        let input = super::norm_quantize_cached::entry(shape.geometry);
        let query = super::norm_query_cached::entry(shape.geometry);
        Some(Box::new(
            Sm120::new()
                .with_rmsnorm_quantize(input.reference.clone())
                .with_rmsnorm_quantize(query.reference.clone())
                .with_rmsnorm_quantize_cached(input, Self::small_norm_buckets(false))
                .with_rmsnorm_quantize_cached(query, Self::small_norm_buckets(true)),
        ))
    }

    pub(crate) fn small_norm_buckets(query: bool) -> Vec<(u32, u32)> {
        use model_deepseek41_contract::tuning::{self, Family};
        let (op, width) = if query {
            ("norm.query_small.decode", FLASH.shape.attention.q_rank)
        } else {
            ("norm.input_small.decode", FLASH.shape.hidden)
        };
        (1..=4)
            .filter_map(|rows| {
                let row = tuning::select(
                    &FLASH,
                    Family::Norm,
                    op,
                    "bf16",
                    &[("ranks", 4), ("width", width)],
                    4,
                    rows,
                )?;
                Some((rows, row.knobs["warps"].count().expect("small norm warps")))
            })
            .collect()
    }
}
