//! MiMo's encoders on the shared encoder runtime
//! (`bundle_runtime::encoder`): both declarations share one buffer
//! plan and are lowered under `f32-pairs.v1`, staged. The instance's target
//! creates the runtime (`crate::target::Target::encoders`), preparing every
//! slot on its encoder kernels and MiMo's own encoder ops; this module holds
//! the encoders' buffer plan and block taps.

use encoder_spec::buffers::BufferPlan;
use encoder_spec::components::StageOp;
use encoder_spec::policy::Precision;
use model_mimo26_contract::encoders::Encoders;
use runtime_bundle_sdk::taps::{TapCompare, TapDType, TapSharding, TapSpec};

/// The block taps of both encoders' transformers (`Encoder::block_taps`),
/// as the runtime names them: each block's output, and each transformer's
/// first block's input, rows of the transformer's width that every rank
/// holds alike as F32 under `f32-pairs.v1`. The declaration loader rejects
/// the retired `bf16-legacy.v1` policy.
pub fn block_taps_for(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> Vec<TapSpec> {
    let Ok(declarations) = Encoders::ALL.declarations_for(geometry) else {
        return Vec::new();
    };
    let mut specs = Vec::new();
    for declaration in declarations {
        let tower = match declaration.input.recipes.first().map(|recipe| recipe.tag()) {
            Some(tag) if tag.starts_with("audio.") => "audio",
            _ => "vision",
        };
        let dtype = match declaration.precision {
            Precision::F32Pairs => TapDType::F32,
            Precision::Bf16Legacy => TapDType::Bf16,
        };
        let mut block = 0;
        for stage in &declaration.stages {
            let StageOp::Transformer(transformer) = &stage.op else {
                continue;
            };
            let params = transformer.params();
            let tap = |site: &str, block: u32| TapSpec {
                name: format!("enc.{tower}.B{block:02}.{site}"),
                layer: None,
                detail: false,
                cols: params.width,
                dtype,
                sharding: TapSharding::Replicated,
                compare: TapCompare::Dense,
            };
            specs.push(tap("in", block));
            for _ in 0..params.blocks {
                specs.push(tap("out", block));
                block += 1;
            }
        }
    }
    specs
}

/// The buffer plan the encoders serving `media` allocate on every rank.
#[cfg(test)]
pub fn plan(media: Encoders) -> Result<BufferPlan, String> {
    plan_for(&model_mimo26_contract::geometry::V26_FLASH, media)
}
pub fn plan_for(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    media: Encoders,
) -> Result<BufferPlan, String> {
    let declarations = media.declarations_for(geometry).map_err(|error| error.0)?;
    let declarations: Vec<_> = declarations.iter().collect();
    model_mimo26_contract::encoders::plan_declarations(&declarations).map_err(|error| error.0)
}

#[cfg(test)]
pub fn block_taps() -> Vec<TapSpec> {
    block_taps_for(&model_mimo26_contract::geometry::V26_FLASH)
}
