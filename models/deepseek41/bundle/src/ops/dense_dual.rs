//! One target decode launch for the query and window-key projections.
use super::dense::GemmViews;
use super::{Asks, Bind, Lowered, exact, fold_stream};
use crate::{program::Phase, shape::Shape};
use kernel_core::backend::{Counts, Prepared, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldStreamBuffers};

pub(crate) struct Buffers {
    pub projections: [Mxfp8FoldStreamBuffers<View>; 2],
    pub work: [View; 2],
}

pub(crate) struct Op {
    entry: Prepared<Buffers>,
    kv_initializer: fold_stream::Op,
}
impl Op {
    pub(crate) fn prepare(shape: &Shape, asks: &Asks) -> Result<Option<Self>, Refused> {
        let need = exact::need(1, exact::verify(1));
        let Some(op) = crate::Kernels::prepare_dense_dual(shape, need)? else {
            return Ok(None);
        };
        asks.record("dense.query-kv decode 1", need, op.entry.info());
        Ok(Some(op))
    }
    pub(crate) fn new(entry: Prepared<Buffers>, kv_initializer: fold_stream::Op) -> Self {
        Self {
            entry,
            kv_initializer,
        }
    }
    pub(crate) fn initializer(&self, weight: &str) -> Option<&fold_stream::Op> {
        (weight.starts_with("layers.") && weight.ends_with(".attn.wkv.weight"))
            .then_some(&self.kv_initializer)
    }
    pub(crate) fn eligible(&self, shape: &Shape, phase: Phase, layer: u32) -> bool {
        matches!(phase, Phase::Decode(1)) && layer < shape.geometry.layers()
    }
    pub(crate) fn launch(&self, views: &[GemmViews; 2]) -> Result<Vec<Lowered>, Refusal> {
        if (
            views[0].activations,
            views[0].activation_scales,
            views[0].live_rows,
            views[0].status,
        ) != (
            views[1].activations,
            views[1].activation_scales,
            views[1].live_rows,
            views[1].status,
        ) {
            return Err(Refusal(
                "dual projections require identical physical input spans",
            ));
        }
        let mut bind = Bind::default();
        let activations = bind.view(views[0].activations);
        let activation_scales = bind.view(views[0].activation_scales);
        let live_rows = bind.view(views[0].live_rows);
        let status = bind.view(views[0].status);
        let mut buffers = Vec::new();
        let mut work = Vec::new();
        for v in views {
            let storage = v
                .stream
                .ok_or(Refusal("dual fold has no initialized weight storage"))?;
            let first = Mxfp8FoldGemmBuffers {
                activations,
                activation_scales,
                weights: bind.view(v.weights),
                weight_scales: bind.view(v.weight_scales),
                output: bind.view(v.output),
                live_rows,
                status,
                status_row_offset: 0,
            };
            buffers.push(Mxfp8FoldStreamBuffers {
                first,
                second: None,
                packed_weights: [bind.view(storage.packed[0]), View::EMPTY],
                state: bind.view(storage.state),
            });
            work.push(bind.view(storage.workspace));
        }
        let buffers = buffers.try_into().unwrap();
        let work = work.try_into().unwrap();
        let launches = self.entry.launches(
            &Buffers {
                projections: buffers,
                work,
            },
            View::EMPTY,
            Counts::rows(1),
        )?;
        Ok(launches
            .into_iter()
            .map(|launch| bind.lowered(launch))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    #[test]
    fn limit_combined_folds_to_one_row_target_decode() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let op = Op::prepare(&shape, &Asks::default()).unwrap().unwrap();
        for layer in [0, 39, 40, 41] {
            for phase in [
                Phase::Decode(1),
                Phase::Decode(8),
                Phase::Verify { rows: 1, items: 1 },
                Phase::Verify { rows: 32, items: 8 },
                Phase::Prefill { rows: 1 },
            ] {
                assert_eq!(
                    op.eligible(&shape, phase, layer),
                    phase == Phase::Decode(1) && layer < 40
                );
            }
        }
        assert!(op.initializer("layers.0.attn.wkv.weight").is_some());
        assert!(op.initializer("mtp.0.attn.wkv.weight").is_none());
        assert!(op.initializer("layers.0.attn.wq_a.weight").is_none());
        // The dual fold reads the planned packed WKV codes.
        assert_eq!(
            op.kv_initializer.info().bytes_of("packed_weights.0"),
            Some(512 * 5120)
        );
        for ranks in [Ranks::TP1, Ranks::TP2] {
            assert!(
                Op::prepare(&Shape::new(&FLASH, ranks).unwrap(), &Asks::default())
                    .unwrap()
                    .is_none()
            );
        }
        crate::sm120::dense_dual::with_reference(|| {
            assert!(Op::prepare(&shape, &Asks::default()).unwrap().is_none())
        });
    }
}

#[cfg(test)]
mod binding_tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
    #[test]
    fn bind_identical_spans_once_and_refuse_different_inputs() {
        let span =
            |raw| DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(raw), 0, 1 << 30);
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let op = Op::prepare(&shape, &Asks::default()).unwrap().unwrap();
        let mut views = std::array::from_fn(|i| {
            let base = 10 + i as u64 * 10;
            GemmViews {
                stream: Some(fold_stream::Storage {
                    packed: [
                        span(base),
                        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(0), 0, 0),
                    ],
                    state: span(base + 1),
                    workspace: span(base + 2),
                }),
                activations: span(1),
                activation_scales: span(2),
                live_rows: span(3),
                status: span(4),
                weights: span(base + 3),
                weight_scales: span(base + 4),
                output: span(base + 5),
            }
        });
        let calls = op.launch(&views).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].launch.rows, 1);
        assert_eq!(calls[0].launch.arguments.0.len(), 18);
        views[1].activation_scales = span(31);
        assert!(op.launch(&views).is_err());
    }
}
