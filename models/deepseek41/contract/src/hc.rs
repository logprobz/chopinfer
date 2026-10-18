//! The hyper-connection contracts the bundle prepares: lagged Sinkhorn
//! mixing over the residual streams (`Block.hc_mixes`, `hc_pre` and
//! `hc_post`, `model.py:950-1000`) and the head's collapse by the last
//! mixer's `pre` (`model.py:1271`). Each sublayer collapses its streams
//! with the `pre` the previous sublayer's mixer computed (the first with
//! the one-hot `pre` on stream 0, `make_identity_pre_mix`,
//! `model.py:1162-1166`), and its `post` and `comb` expand its
//! output in F32 before the streams round once to BF16.

use kernel_core::contract::DType;
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, SinkhornHyperConnectionQuery, SinkhornMix, SinkhornOrder,
    StreamHeadMode,
};

use crate::geometry::Geometry;

/// `g`'s mixer: F32 `fn`, eps and iterations from the geometry, the RMS
/// statistic at the model's norm eps (`Block.norm_eps`).
pub fn mix(g: &Geometry) -> SinkhornMix {
    SinkhornMix {
        streams: g.shape.hc.streams,
        width: g.shape.hidden,
        iterations: g.shape.hc.sinkhorn_iterations,
        eps_bits: (g.shape.hc.eps as f32).to_bits(),
        norm_eps_bits: (g.shape.eps as f32).to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::LaggedSinglePass,
        fn_dtype: DType::F32,
    }
}

/// A sublayer's lagged mix-collapse over up to `capacity` rows.
pub fn mix_collapse(g: &Geometry, capacity: u32) -> SinkhornHyperConnectionQuery {
    SinkhornHyperConnectionQuery::MixCollapse {
        mix: mix(g),
        capacity,
    }
}

/// A sublayer's expand of its output by the F32 `post` and `comb`.
pub fn expand(g: &Geometry, capacity: u32) -> SinkhornHyperConnectionQuery {
    SinkhornHyperConnectionQuery::ExpandF32 {
        streams: g.shape.hc.streams,
        width: g.shape.hidden,
        capacity,
    }
}

/// The collapse before the final norm, by the last mixer's `pre`: F32
/// rows, unrounded, which the F32 final norm reads (`rows::final_norm`).
pub fn head(g: &Geometry, capacity: u32) -> SinkhornHyperConnectionQuery {
    SinkhornHyperConnectionQuery::Head {
        streams: g.shape.hc.streams,
        width: g.shape.hidden,
        mode: StreamHeadMode::SigmoidWeightedF32,
        capacity,
    }
}

/// The one-hot `pre` the first sublayer collapses with: 1 on stream 0.
pub fn first_pre(g: &Geometry) -> Vec<f32> {
    let mut pre = vec![0.0; g.shape.hc.streams as usize];
    pre[0] = 1.0;
    pre
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::FLASH;
    use kernel_core::contract::Validate;

    #[test]
    fn flash_mixes_four_streams_lagged_in_f32() {
        let m = mix(&FLASH);
        m.validate().unwrap();
        assert_eq!((m.streams, m.width, m.iterations), (4, 5120, 20));
        assert_eq!(f32::from_bits(m.eps_bits), 1e-6);
        assert_eq!(f32::from_bits(m.norm_eps_bits), 1e-20);
        assert_eq!(m.mixes(), FLASH.hc_mix());
        assert_eq!(m.flat_width(), FLASH.hc_width());
        for query in [
            mix_collapse(&FLASH, 64),
            expand(&FLASH, 64),
            head(&FLASH, 64),
        ] {
            query.validate().unwrap();
        }
        assert_eq!(first_pre(&FLASH), [1.0, 0.0, 0.0, 0.0]);
    }
}
