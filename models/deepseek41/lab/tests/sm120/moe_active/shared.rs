use super::*;
use kernel_core::gated_mlp::quantized::SiluMulQuantizeBuffers;
use kernel_core::moe::RoutedSumBuffers;
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;
use kernels_sm120::moe::chained::{RoutedSum, RoutedSumOutput};

pub fn weights(a: &mut Arena<'_>) -> LabResult<[QuantizedWeights<View>; 3]> {
    let mut make = |n: usize, k: usize, seed: usize| -> LabResult<QuantizedWeights<View>> {
        let codes = if a.session.is_some() {
            let row_major = (0..n * k)
                .map(|i| ((i.wrapping_mul(seed) % 96) as u8) | if i % 3 == 0 { 128 } else { 0 })
                .collect::<Vec<_>>();
            let packed =
                kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout { groups: 1, n, k }
                    .pack(&row_major)
                    .map_err(|e| e.to_string())?;
            a.bytes(&packed)?
        } else {
            a.filled((n * k) as u64, 0)?
        };
        Ok(QuantizedWeights {
            codes,
            local_scales: Some(a.filled((n * k / 32) as u64, 124)?),
            global_scales: None,
        })
    };
    Ok([
        make(576, 5120, 17)?,
        make(576, 5120, 29)?,
        make(5120, 576, 43)?,
    ])
}

/// The serving shared folds always use their 8192-capacity declarations.
/// Keep their three separate projections, fused activation/quantization,
/// sticky row status and final routed sum exactly as production binds them.
pub fn append(
    a: &mut Arena<'_>,
    case: Case,
    inputs: Inputs,
    arm: &mut Arm,
    weights: [QuantizedWeights<View>; 3],
) -> LabResult<()> {
    let fold = bundle_deepseek41::sm120::mxfp8_fold::entries(&FLASH, 4)?;
    let mut backend = Sm120::new().with_packed_fold_weights();
    for e in &fold {
        backend = backend.with_mxfp8_fold(e.clone());
    }
    let query = moe::shared_activation(&FLASH, 4);
    let activation = kernels_sm120::row::silu_mul_quantize::Entry::new(
        query,
        8192,
        format!(
            "chop_{}_shared_swiglu_w{}_v1",
            FLASH.kernel_tag, query.width
        ),
    )
    .map_err(|e| e.to_string())?;
    backend = backend.with_silu_mul_quantize(activation);
    let mut outputs = Vec::new();
    for bytes in [
        case.rows * 576 * 4,
        case.rows * 576 * 4,
        case.rows * 576,
        case.rows * 18,
        case.rows * 5120 * 4,
        case.rows * 5120 * 4,
    ] {
        outputs.push(a.filled(u64::from(bytes), 0)?);
    }
    let [gate, up, codes, scales, partial, combined] = outputs[..] else {
        unreachable!()
    };
    let counts = Counts::rows(case.rows);
    let need = Need::rows(case.rows);
    let projection = |n,
                      k,
                      input,
                      input_scales,
                      weight: QuantizedWeights<View>,
                      output|
     -> LabResult<Vec<Launch>> {
        let entry = fold
            .iter()
            .find(|e| {
                e.query.n == n
                    && e.query.k == k
                    && e.query.groups == 1
                    && e.query.m_capacity == 8192
            })
            .ok_or("missing shared fold")?;
        let op = backend
            .mxfp8_fold_gemm(&entry.query, need)
            .map_err(|e| e.to_string())?;
        op.launches(
            &Mxfp8FoldGemmBuffers {
                activations: input,
                activation_scales: input_scales,
                weights: weight.codes,
                weight_scales: weight.local_scales.unwrap(),
                output,
                live_rows: inputs.live,
                status: arm.status,
                status_row_offset: 0,
            },
            View::EMPTY,
            counts,
        )
        .map_err(|e| e.to_string())
    };
    arm.side.extend(projection(
        576,
        5120,
        inputs.input,
        inputs.scales,
        weights[0],
        gate,
    )?);
    arm.side.extend(projection(
        576,
        5120,
        inputs.input,
        inputs.scales,
        weights[1],
        up,
    )?);
    let activation = backend
        .silu_mul_quantize(&query, need)
        .map_err(|e| e.to_string())?;
    arm.side.extend(
        activation
            .launches(
                &SiluMulQuantizeBuffers {
                    gate,
                    up,
                    codes,
                    scales,
                    live_rows: inputs.live,
                    status: arm.status,
                    status_row_offset: 0,
                },
                View::EMPTY,
                counts,
            )
            .map_err(|e| e.to_string())?,
    );
    arm.side
        .extend(projection(5120, 576, codes, scales, weights[2], partial)?);
    let prefix = if case.set.count == FLASH.shape.moe.experts {
        format!("chop_moe_{}", FLASH.kernel_tag)
    } else {
        format!("chop_{}_dspark_moe", FLASH.kernel_tag)
    };
    let sum_entry = RoutedSum {
        top_k: case.set.top_k,
        width: 5120,
        local_experts: case.set.count,
        capacity: 8192,
        output: RoutedSumOutput::F32,
    };
    backend = backend
        .with_routed_sum(sum_entry, prefix.clone())
        .with_routed_sum_add(sum_entry, prefix);
    let sum = backend
        .moe_routed_sum(
            &MoeQuery::LocalRoutedSum {
                experts: case.set.count,
                local_experts: case.set.count,
                top_k: case.set.top_k,
                width: 5120,
                capacity: case.rows,
                output: kernel_core::contract::DType::F32,
            },
            0,
            need,
        )
        .map_err(|e| e.to_string())?;
    arm.sum = sum
        .launches(
            &RoutedSumBuffers {
                routed: arm.outputs[5],
                ids: arm.outputs[0],
                addend: Some(partial),
                output: combined,
                live_rows: inputs.live,
            },
            View::EMPTY,
            counts,
        )
        .map_err(|e| e.to_string())?;
    arm.poison.extend(outputs.iter().copied());
    arm.outputs.extend(outputs);
    Ok(())
}
