//! Reuse qualified exact-row folds for short Flash TP4 prefills.
use crate::shape::Shape;
use model_deepseek41_contract::geometry::FLASH;

pub const MAX_ROWS: u32 = 64;

pub fn enabled(shape: &Shape) -> bool {
    shape.geometry == &FLASH && shape.ranks() == 4
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{
        Asks,
        dense::{DenseOps, GemmViews, projections},
    };
    use crate::program::Phase;
    use bundle_runtime::geometry::Ranks;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    #[test]
    fn threshold_keeps_unqualified_topologies_on_full_folds() {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let shape = Shape::new(&FLASH, ranks).unwrap();
            let kernels = crate::Kernels::new(&shape).unwrap();
            let ops =
                DenseOps::prepare(kernels.backend.as_ref(), &shape, &Asks::default()).unwrap();
            assert_eq!(ops.short_prefill, ranks == Ranks::TP4);
            for d in projections(&shape) {
                for rows in 1..=65 {
                    let q = d.query;
                    let span = |id, bytes| {
                        DeviceBufferSpanV1::new(
                            BundleDeviceBufferHandleV1::from_parts(id, 1),
                            128,
                            bytes,
                        )
                    };
                    let views = GemmViews {
                        stream: None,
                        activations: span(1, u64::from(rows) * u64::from(q.activation_row_stride)),
                        activation_scales: span(
                            2,
                            u64::from(rows) * u64::from(q.activation_row_stride) / 32,
                        ),
                        weights: span(3, u64::from(q.n) * u64::from(q.k) * u64::from(q.groups)),
                        weight_scales: span(
                            4,
                            u64::from(q.n) * u64::from(q.k) * u64::from(q.groups) / 32,
                        ),
                        output: span(
                            5,
                            u64::from(rows) * u64::from(q.n) * u64::from(q.groups) * 4,
                        ),
                        live_rows: span(6, 4),
                        status: span(7, u64::from(rows) * 4),
                    };
                    let new = ops
                        .project(d.id, Phase::Prefill { rows }, &views, span(99, 1 << 30))
                        .unwrap();
                    if ranks == Ranks::TP4 && rows <= MAX_ROWS {
                        assert!(new.len() >= 2);
                        assert!(
                            new.iter()
                                .all(|c| !c.launch.symbol.contains("mxfp8_fold_m8192"))
                        );
                    } else {
                        assert_eq!(new.len(), 1);
                        assert!(new[0].launch.symbol.contains("mxfp8_fold_m8192"));
                    }
                }
            }
        }
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        assert!(enabled(&shape));
    }
}
