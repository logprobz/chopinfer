//! The chained MoE's weights in the checkpoint's layout
//! (`model_glm5next_contract::quantization`): each chained projection's
//! format by layer and class, and the views its launch binds in that
//! format, as the rank image stores them (`chained_rows_v1`): codes, then
//! FP8's F32 block scales, NVFP4's UE4M3 and MXFP8's UE8M0 local scales,
//! and NVFP4's F32 globals (gate and up per expert, one per expert for
//! down, `[1, 2]` for a dense gate/up).

use kernel_core::contract::{ScaleMode, WeightFormat};
use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::quantization::{Encoding, Layout};
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Lowering, layer_view};

/// Which chained projection of a layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Class {
    /// The dense MLP of the first layers.
    Dense,
    /// The shared expert.
    Shared,
    /// The routed experts (the MTP layer's after the main layers').
    Experts,
}

/// The format of `class`'s weights at `layer` in an export stored in
/// `layout`: the shared expert in the projections' encoding, the routed
/// experts of the MTP layer (after the main layers) in its own.
pub(crate) fn weight_format(
    layout: &Layout,
    shape: Shape,
    layer: u32,
    class: Class,
) -> WeightFormat {
    let encoding = match class {
        Class::Dense => layout.dense,
        Class::Shared => layout.projections,
        Class::Experts if layer < shape.layers() => layout.routed,
        Class::Experts => layout.mtp_routed,
    };
    match encoding {
        Encoding::Fp8Block128 => WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        Encoding::Nvfp4 => WeightFormat::Nvfp4,
        Encoding::Mxfp8 => WeightFormat::Mxfp8,
        Encoding::Bf16 => WeightFormat::Bf16,
    }
}

impl Lowering<'_> {
    /// The format of `class`'s weights at `layer` in the instance's layout.
    pub(super) fn chained_weights(&self, layer: u32, class: Class) -> WeightFormat {
        weight_format(self.ops.plan.layout, self.shape, layer, class)
    }

    /// The codes, block scales, local scales and globals a chained
    /// projection over the packed views `name` of `layer` reads in
    /// `weights`, in the engine's argument order; `None` where the format
    /// reads none.
    pub(super) fn chained_views(
        &self,
        layer: u32,
        name: &str,
        weights: WeightFormat,
    ) -> Result<[Option<DeviceBufferSpanV1>; 4], String> {
        let view = |suffix: &str| {
            self.buffers
                .view(&layer_view(layer, &format!("{name}.{suffix}")))
                .map(Some)
        };
        Ok(match weights {
            WeightFormat::Fp8Block128x128(_) => {
                [view("weight")?, view("weight_scale_inv")?, None, None]
            }
            WeightFormat::Nvfp4 => [
                view("weight")?,
                None,
                view("weight_scale")?,
                view("weight_scale_2")?,
            ],
            WeightFormat::Mxfp8 => [view("weight")?, None, view("weight_scale")?, None],
            WeightFormat::Bf16 => [view("weight")?, None, None, None],
            other => return Err(format!("{name} of layer {layer} has no {other:?} views")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::quantization::{FP8_BLOCK128, NVFP4_DENSE_ROUTED, NVFP4_ROUTED};

    #[test]
    fn each_layout_names_its_chained_formats() {
        let shape = Shape::TP4;
        let mtp = shape.layers();
        let of = |layout| {
            [
                weight_format(layout, shape, 0, Class::Dense),
                weight_format(layout, shape, 3, Class::Shared),
                weight_format(layout, shape, 3, Class::Experts),
                weight_format(layout, shape, mtp, Class::Shared),
                weight_format(layout, shape, mtp, Class::Experts),
            ]
        };
        use WeightFormat::*;
        let fp8 = Fp8Block128x128(ScaleMode::ExactF32);
        assert_eq!(of(&FP8_BLOCK128), [fp8; 5]);
        assert_eq!(of(&NVFP4_ROUTED), [Bf16, Bf16, Nvfp4, Bf16, Mxfp8]);
        assert_eq!(of(&NVFP4_DENSE_ROUTED), [Nvfp4, Bf16, Nvfp4, Bf16, Bf16]);
    }

    /// Each layout's records run its chained projections in their formats
    /// and its projections at their logical shapes: NVIDIA's NVFP4 dense
    /// down projection reads K = 3072 per rank, not its 1,536 packed bytes,
    /// and no scale companion counts as a weight.
    #[test]
    fn records_follow_the_checkpoint_layout() {
        use crate::launch::{Op, records_for};
        use crate::program::{Phase, Stage};
        use kernel_core::contract::{DType, WeightFormat};
        use kernel_core::latent_attention::LatentKvFormat;
        let shape = Shape::TP4;
        for layout in [&FP8_BLOCK128, &NVFP4_ROUTED, &NVFP4_DENSE_ROUTED] {
            let records =
                records_for(shape, layout, LatentKvFormat::Bf16, Phase::Decode(8)).unwrap();
            let mut gated = 0;
            let mut dense_down = 0;
            for record in &records {
                match (record.stage, record.op) {
                    (Stage::DenseGateUp(layer), Op::GatedMlp(query)) => {
                        assert_eq!(
                            query.gemm.weight,
                            weight_format(layout, shape, layer, Class::Dense)
                        );
                        gated += 1;
                    }
                    (Stage::SharedGateUp(layer), Op::GatedMlp(query)) => {
                        assert_eq!(
                            query.gemm.weight,
                            weight_format(layout, shape, layer, Class::Shared)
                        );
                        gated += 1;
                    }
                    (
                        Stage::ExpertsGateUp(layer) | Stage::ExpertsDown(layer),
                        Op::GatedMlp(query),
                    ) => {
                        assert_eq!(
                            query.gemm.weight,
                            weight_format(layout, shape, layer, Class::Experts)
                        );
                        gated += 1;
                    }
                    (Stage::Project(_, "dense.down"), Op::Gemm(gemm)) => {
                        assert_eq!((gemm.n, gemm.k, gemm.output), (4096, 3072, DType::F32));
                        let expected = match layout.dense {
                            Encoding::Nvfp4 => WeightFormat::Nvfp4,
                            Encoding::Bf16 => WeightFormat::Bf16,
                            _ => weight_format(layout, shape, 0, Class::Dense),
                        };
                        assert_eq!(gemm.weight, expected, "{}", layout.id);
                        dense_down += 1;
                    }
                    _ => {}
                }
            }
            assert!(gated > 0 && dense_down == 3, "{}", layout.id);
        }
    }
}
