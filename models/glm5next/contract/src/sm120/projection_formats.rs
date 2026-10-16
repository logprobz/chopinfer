//! The lowered projections' offers in the NVFP4 exports' layouts that the
//! block-FP8 export's offers (`crate::sm120::precision`) do not hold: the BF16
//! rows of projections the FP8 export quantizes (latent attention, the
//! shared expert's down projection, the dense MLP's down projection in
//! local-inference-lab's export) and NVIDIA's NVFP4 dense down projections.
//! They build in their own cubin under their own lock family, so the FP8
//! projections' cubins and lock stay as they are.

use std::collections::BTreeMap;

use kernel_build::CubinSpec;
use kernel_core::contract::InstantiationSpec;
use kernels_sm120::precision::Compute;

use super::precision::{copies, instantiations, offer};
use crate::contract::Shape;
use crate::precision::{layout_formats, uses};
use crate::quantization::{FP8_BLOCK128, LAYOUTS};

/// The lock family of [`CUBIN`] (`bundle/kernels.lock.d`).
pub const FAMILY: &str = "projection_formats";
/// The GEMM cubin of the NVFP4 layouts' added offers.
pub const CUBIN: &str = "projection_formats_glm5next_sm120";

/// Every distinct GEMM entry the NVFP4 layouts' lowered projections run at
/// their declared uses and the block-FP8 export's do not
/// (`crate::sm120::precision::instantiations`), in symbol order. Refuses a layout
/// whose uses need a layout copy the FP8 projections' copies lack.
pub fn instantiations_added(shape: Shape) -> Result<Vec<InstantiationSpec>, String> {
    let served: BTreeMap<String, InstantiationSpec> = instantiations(shape)?
        .into_iter()
        .map(|spec| (spec.symbol.clone(), spec))
        .collect();
    let copied: Vec<String> = copies(shape)?
        .iter()
        .map(|copy| copy.stage().entry().symbol.clone())
        .collect();
    let mut added = BTreeMap::new();
    for layout in LAYOUTS.iter().filter(|layout| **layout != FP8_BLOCK128) {
        for (id, weight) in layout_formats(shape, layout)? {
            for (usage, _) in uses(id)? {
                let generated = offer(id, weight, usage)?;
                let Compute::Dense(spec) = &generated.offer.handle().compute else {
                    return Err(format!("projection {id} is not a dense GEMM"));
                };
                if !served.contains_key(&spec.symbol) {
                    added.insert(spec.symbol.clone(), spec.clone());
                }
                if let Some(copy) = super::precision::copy(id, weight, usage)?
                    && !copied.contains(&copy.stage().entry().symbol)
                {
                    return Err(format!(
                        "projection {id} in the {} layout needs a layout copy the FP8 \
                         projections do not build",
                        layout.id
                    ));
                }
            }
        }
    }
    Ok(added.into_values().collect())
}

/// [`CUBIN`]: the added GEMM entries.
pub fn cubin_specs(shape: Shape) -> Result<Vec<CubinSpec>, String> {
    let mut gemm = kernels_sm120::gemm::cubin_spec(&instantiations_added(shape)?)
        .map_err(|r| r.0.to_string())?;
    gemm.name = CUBIN.into();
    Ok(vec![gemm])
}

#[cfg(test)]
mod tests {
    use kernel_core::contract::{ScaleMode, WeightFormat};

    use super::*;
    use crate::quantization::{NVFP4_DENSE_ROUTED, NVFP4_ROUTED};

    #[test]
    fn the_nvfp4_layouts_add_bf16_rows_and_nvidia_s_nvfp4_dense_down() {
        let shape = Shape::TP4;
        let fp8 = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
        let formats = |layout| layout_formats(shape, layout).unwrap();
        assert_eq!(
            formats(&FP8_BLOCK128),
            crate::precision::checkpoint_formats(shape).unwrap()
        );
        for layout in [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED] {
            assert!(formats(layout).iter().all(|(_, weight)| *weight != fp8));
        }
        let nvfp4 = |layout| {
            formats(layout)
                .into_iter()
                .filter(|(_, weight)| *weight == WeightFormat::Nvfp4)
                .map(|(id, _)| id)
                .collect::<Vec<_>>()
        };
        assert!(nvfp4(&NVFP4_ROUTED).is_empty());
        assert_eq!(nvfp4(&NVFP4_DENSE_ROUTED), ["dense.down"]);
        let added = instantiations_added(shape).unwrap();
        let served: Vec<String> = instantiations(shape)
            .unwrap()
            .into_iter()
            .map(|spec| spec.symbol)
            .collect();
        assert!(!added.is_empty());
        assert!(added.iter().all(|spec| !served.contains(&spec.symbol)));
        assert!(added.iter().any(|spec| spec.symbol.contains("nvfp4")));
        let [spec] = cubin_specs(shape).unwrap().try_into().unwrap();
        assert_eq!(spec.name, CUBIN);
        assert_eq!(spec.entries.len(), added.len());
    }
}
