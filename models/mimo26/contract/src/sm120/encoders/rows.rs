//! MiMo's own encoder row kernels: the residual quantizer and the 20-table
//! embedding sum of its `f32-pairs.v1` audio stages, a discrete tokenizer
//! and speech tables no other model has (ENCODER-FRAMEWORK 3.11). Their
//! bodies are in models/mimo26/bundle/cuda/mimo_encoder.cuh; the bundle
//! builds the kernels its declarations run into its cubin `CUBIN` and
//! resolves each such registered stage's `registered` slot to them. The
//! shared encoder catalog never names them.

use encoder_spec::components::{Boundary, Component, StageOp};
use encoder_spec::{Declaration, Error};

use crate::encoders::spec::{Rvq20, Sum20};

/// The cubin holding the kernels.
pub const CUBIN: &str = "mimo_encoder";
/// The header holding their templates, in models/mimo26/bundle/cuda.
pub const HEADER: &str = "mimo_encoder.cuh";
/// Threads per CTA; every kernel grids one CTA per row of its capacity.
pub const THREADS: u32 = 256;

/// One kernel: its symbol, the entry body calling its template, its device
/// pointer parameters (name and C++ type) in ABI order, and the rows it
/// grids for.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowKernel {
    pub symbol: String,
    pub body: String,
    pub params: Vec<(&'static str, &'static str)>,
    pub capacity: u32,
}

/// Decodes a registered variant from the parameters it was written with.
fn typed<T: serde::de::DeserializeOwned>(component: &dyn Component) -> Result<T, String> {
    serde_json::from_value(component.to_value())
        .map_err(|error| format!("`{}`: {error}", component.id()))
}

/// The kernel of registered stage `component` reading `input` in runs of
/// up to `capacity` rows, or `None` for a variant the shared catalog runs:
/// `chop::mimo_encoder::rvq_f32` for `mimo26:quantize.rvq20-f32.v1` and
/// `chop::mimo_encoder::embedding_sum_f32` for `mimo26:embed.sum20-f32.v1`.
pub fn kernel(
    component: &dyn Component,
    input: &Boundary,
    capacity: u32,
) -> Result<Option<RowKernel>, String> {
    let live = ("live_rows", "const uint32_t");
    Ok(Some(match component.id().as_str() {
        "mimo26:quantize.rvq20-f32.v1" => {
            let rvq: Rvq20 = typed(component)?;
            let codebooks = &rvq.params().codebooks;
            let dim = input.width()?;
            let max_codes = codebooks.iter().copied().max().unwrap_or(0);
            let stages = codebooks.len();
            RowKernel {
                symbol: format!("chop_mimo_enc_rvq{stages}_f32_d{dim}_codes{max_codes}_v1"),
                body: format!(
                    "chop::mimo_encoder::rvq_f32<{dim},{stages},{max_codes}>(input,codebooks,codebook_offsets,codes,live_rows);"
                ),
                params: vec![
                    ("input", "const float"),
                    ("codebooks", "const float"),
                    ("codebook_offsets", "const int32_t"),
                    ("codes", "int32_t"),
                    live,
                ],
                capacity,
            }
        }
        "mimo26:embed.sum20-f32.v1" => {
            let sum: Sum20 = typed(component)?;
            let params = sum.params();
            let (tables, vocab, width) = (20, params.vocab, params.width);
            RowKernel {
                symbol: format!("chop_mimo_enc_sum{tables}_f32_vocab{vocab}_w{width}_v1"),
                body: format!(
                    "chop::mimo_encoder::embedding_sum_f32<{tables},{vocab},{width}>(tables,codes,output,live_rows);"
                ),
                params: vec![
                    ("tables", "const uint16_t"),
                    ("codes", "const int32_t"),
                    ("output", "float"),
                    live,
                ],
                capacity,
            }
        }
        _ => return Ok(None),
    }))
}

/// The kernel of stage `stage` of `declaration`, if MiMo owns it.
pub fn stage_kernel(declaration: &Declaration, stage: usize) -> Result<Option<RowKernel>, Error> {
    let fail = |error: String| Error(format!("{} stages[{stage}]: {error}", declaration.name));
    let Some(StageOp::Registered(registered)) = declaration.stages.get(stage).map(|s| &s.op) else {
        return Ok(None);
    };
    let inputs = declaration.inputs()?;
    kernel(
        registered.component.as_ref(),
        &inputs[stage],
        declaration.input.capacity_rows,
    )
    .map_err(fail)
}

/// Every kernel MiMo's declarations run, each symbol once, in order of
/// first use.
pub fn kernels_flash() -> Result<Vec<RowKernel>, Error> {
    kernels_for(&crate::geometry::V26_FLASH)
}

pub fn kernels_for(geometry: &'static crate::geometry::Geometry) -> Result<Vec<RowKernel>, Error> {
    let mut kernels: Vec<RowKernel> = Vec::new();
    for declaration in crate::encoders::spec::declarations_for(geometry)? {
        for stage in 0..declaration.stages.len() {
            let Some(kernel) = stage_kernel(declaration, stage)? else {
                continue;
            };
            match kernels
                .iter_mut()
                .find(|known| known.symbol == kernel.symbol)
            {
                Some(known) if known.body != kernel.body || known.params != kernel.params => {
                    return Err(Error(format!("{} stands for two kernels", kernel.symbol)));
                }
                Some(known) => known.capacity = known.capacity.max(kernel.capacity),
                None => kernels.push(kernel),
            }
        }
    }
    Ok(kernels)
}
