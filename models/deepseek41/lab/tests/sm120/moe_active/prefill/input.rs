//! Include input quantization and the serving exact F32 router projection.
use super::*;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::buffers::{ProjectionBuffers, ProjectionScratch};
use kernel_core::precision::quantizer::BlockQuantizeBuffers;

pub struct Source {
    pub bf16: View,
    pub wide: View,
    pub router: QuantizedWeights<View>,
}

pub fn source(a: &mut Arena<'_>, case: Case) -> LabResult<Source> {
    let mut bf16 = Vec::new();
    let mut wide = Vec::new();
    let mut weights = Vec::new();
    if a.session.is_some() {
        let logits = case.logits();
        for row in 0..case.rows as usize {
            for col in 0..5120 {
                let value = if col < case.set.count as usize {
                    let at = (row * case.set.count as usize + col) * 4;
                    f32::from_le_bytes(logits[at..at + 4].try_into().unwrap())
                } else {
                    ((row * 37 + col * 19) % 997) as f32 / 251.0 - 1.5
                };
                let bits = f32_to_bf16(value);
                bf16.extend(bits.to_le_bytes());
                wide.extend(bf16_to_f32(bits).to_le_bytes());
            }
        }
        // An identity prefix controls expert order. The dense common suffix
        // starts at a 128-K boundary and adds the same offset to every logit;
        // it avoids measuring a router with nearly all weights zero.
        for n in 0..case.set.count as usize {
            for k in 0..5120 {
                let value = if k >= case.set.count as usize {
                    (k as i32 % 11 - 5) as f32 / 128.0
                } else if n == k {
                    1.0
                } else {
                    0.0
                };
                weights.extend(f32_to_bf16(value).to_le_bytes());
            }
        }
    }
    let mut data = |bytes: Vec<u8>, count: u64| {
        if a.session.is_some() {
            a.bytes(&bytes)
        } else {
            a.filled(count, 0)
        }
    };
    Ok(Source {
        bf16: data(bf16, u64::from(case.rows) * 5120 * 2)?,
        wide: data(wide, u64::from(case.rows) * 5120 * 4)?,
        router: QuantizedWeights {
            codes: data(weights, u64::from(case.set.count) * 5120 * 2)?,
            local_scales: None,
            global_scales: None,
        },
    })
}

pub fn prepend(
    a: &mut Arena<'_>,
    case: Case,
    source: &Source,
    inputs: Inputs,
    arm: &mut Arm,
) -> LabResult<()> {
    let backend = registration::quantizers(&FLASH, 4)?
        .into_iter()
        .fold(Sm120::new(), Sm120::with_block_quantize);
    let quant = backend
        .block_quantize(&moe::quantizer(5120), Need::rows(case.rows))
        .map_err(|e| e.to_string())?;
    let mut prefix = quant
        .launches(
            &BlockQuantizeBuffers {
                input: source.bf16,
                output: inputs.input,
                scales: Some(inputs.scales),
                live_rows: inputs.live,
                status: arm.status,
                status_row_offset: 0,
            },
            View::EMPTY,
            Counts::rows(case.rows),
        )
        .map_err(|e| e.to_string())?;
    let usage = OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity: 8192,
        graph: false,
    };
    let entry = kernels_sm120::f32_projection::Entry::new(
        GemmQuery {
            m_capacity: 8192,
            n: case.set.count,
            k: 5120,
            weight: WeightFormat::Bf16,
            activation: ActivationFormat::F32,
            output: DType::F32,
            epilogues: Epilogues::default(),
            grouped: None,
        },
        usage,
        "prefill_lab_router",
    )
    .map_err(|e| e.to_string())?;
    let backend = bundle_deepseek41::sm120::register_f32_projections(Sm120::new(), &FLASH, 4)?;
    let op = backend
        .projection(
            &kernel_harness::kit::cases::f32_projection::operation(&entry),
            usage,
            Need::rows(case.rows),
        )
        .map_err(|e| e.to_string())?
        .handle;
    let workspace = if op.info().workspace == 0 {
        View::EMPTY
    } else {
        a.filled(op.info().workspace, 0)?
    };
    let buffers = ProjectionBuffers {
        activation: source.wide,
        live_rows: inputs.live,
        output: inputs.logits,
        weights: source.router,
        residual: None,
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            weight_domain: None,
            partial: View::EMPTY,
            semaphores: View::EMPTY,
            unused: View::EMPTY,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
            quantizer_status: None,
            quantizer_status_row_offset: 0,
        },
        row_offset: 0,
        shard_rows: 0,
    };
    prefix.extend(
        op.launches(&buffers, workspace, Counts::rows(case.rows))
            .map_err(|e| e.to_string())?,
    );
    prefix.append(&mut arm.launches);
    arm.launches = prefix;
    arm.outputs
        .extend([inputs.input, inputs.scales, inputs.logits]);
    arm.poison
        .extend([inputs.input, inputs.scales, inputs.logits]);
    if workspace != View::EMPTY {
        arm.poison.push(workspace);
    }
    Ok(())
}
