//! Pair-pooling cases with an observable F32 intermediate. Small F32-only
//! perturbations around +/-8 exercise the softmax and remain visible
//! through the F32 norm. The raw pool and norm have separate f64 bounds.

pub mod steps;

use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Rows, Write};
use crate::kit::{OpCase, Required, Spec, bf16_bytes};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, RecurrentPhase, Refusal};
use kernel_core::gated_pooling::{GatedPoolingBuffers, GatedPoolingQuery, GatedPoolingState};
use kernel_core::gated_pooling_oracle;
use kernel_core::oracle::OracleOutput;
use kernels_sm120::gated_pooling::Entry;

pub const CAPACITY: u32 = 6;
const POSITIONS: [i32; 6] = [1, 0, 5, 2, -1, 9];
const ENTRIES: usize = 13;

kernel_core::scratch_record! {
    #[derive(Clone)]
    pub struct ObservedBuffers {
        pub op: record(GatedPoolingBuffers),
        pub pooled: value,
    }
}

struct Observed {
    op: Prepared<GatedPoolingBuffers<View>>,
    info: OpInfo,
}

impl PreparedOp<ObservedBuffers<View>> for Observed {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &ObservedBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut launches = self.op.launches(&b.op, b.pooled, counts)?;
        // The same workspace is now an explicit observed output role.
        for launch in &mut launches {
            launch.plan.workspace_bytes = 0;
        }
        Ok(launches)
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn output_map(dtype: DType, width: usize, bytes: usize) -> Write<'static> {
    Write::Map(Box::new(move |images, live| {
        let positions = images.i32s("op.positions")?;
        WriteMap::of(dtype).byte_ranges(
            (0..live as usize)
                .filter(|&row| positions[row] >= 0 && positions[row] % 2 == 1)
                .map(|row| row * width * bytes..(row + 1) * width * bytes),
        )
    }))
}

/// One decode row per item. The source slot IDs are deliberately sparse,
/// and their high-ID replay moves the full raw F32 state banks.
pub fn case(
    query: GatedPoolingQuery,
    prepared: Prepared<GatedPoolingBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    if query.phase != RecurrentPhase::Decode || query.group != 2 {
        return Err("the pooling decode case needs pair pooling".into());
    }
    let width = query.width as usize;
    let rows = CAPACITY as usize;
    let state_elements = 2 * width;
    let slots: Vec<i32> = (0..rows).map(|row| (2 * row + 1) as i32).collect();
    let base = |d: usize| if d % 3 == 0 { -8.0 } else { 8.0 };
    let mut keys = vec![0.0; rows * width];
    let mut gates = keys.clone();
    let mut state = vec![31.0; ENTRIES * state_elements];
    // Displaced source entries must remain inside the oracle's rounding
    // domain too. Their distinct low bits still exercise the F32 pool,
    // and exact state destinations witness a stale slot-ID write.
    for entry in 0..ENTRIES {
        for d in 0..width {
            state[entry * state_elements + d] = base(d) + (entry % 3) as f32 / 512.0;
        }
    }
    for row in 0..rows {
        let at = slots[row] as usize * state_elements;
        for d in 0..width {
            keys[row * width + d] = base(d) + ((d % 5) as f32 - 2.0) / 512.0;
            gates[row * width + d] = ((row + d) % 11) as f32 / 4.0 - 1.0;
            state[at + d] = base(d) + ((d % 3) as f32 - 1.0) / 256.0;
            state[at + width + d] = (d % 7) as f32 / 4.0 - 0.5;
        }
    }
    let weights: Vec<u16> = (0..width)
        .map(|d| f32_to_bf16(0.5 + (d % 5) as f32 / 4.0))
        .collect();
    let state_write = Write::Map(Box::new(move |images: &Images, live| {
        let positions = images.i32s("op.positions")?;
        let slots = images.i32s("op.tail.slot_ids")?;
        WriteMap::of(DType::F32).byte_ranges(
            (0..live as usize)
                .filter(|&row| positions[row] >= 0 && positions[row] % 2 == 0)
                .map(|row| {
                    let at = slots[row] as usize * state_elements * 4;
                    at..at + state_elements * 4
                }),
        )
    }));
    let roles = ObservedBuffers {
        op: GatedPoolingBuffers {
            key: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&keys)),
            gate: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&gates)),
            norm_weight: Spec::input(DType::Bf16, (Rows::Fixed(1), width), bf16_bytes(&weights)),
            positions: Spec::ids((Rows::Live, 1), POSITIONS.to_vec(), &[], Holes::LastLive),
            row_batch: Spec::input(
                DType::I32,
                (Rows::Live, 1),
                i32_bytes(&(0..CAPACITY as i32).collect::<Vec<_>>()),
            ),
            live_rows: Spec::live(),
            tail: GatedPoolingState {
                state: Spec::state(
                    DType::F32,
                    (Rows::Fixed(ENTRIES), state_elements),
                    f32_bytes(&state),
                    state_write,
                ),
                entries: ENTRIES as u32,
                slot_ids: Spec::ids(
                    (Rows::Fixed(rows), 1),
                    slots,
                    &[("op.tail.state", state_elements * 4)],
                    Holes::Never,
                ),
                checkpoint_ids: None,
            },
            output: Spec::output_with(
                DType::F32,
                (Rows::Live, width),
                output_map(DType::F32, width, 4),
            ),
            output_positions: Spec::output(DType::I32, (Rows::Live, 1)).filled_with(&[0x5a]),
        },
        pooled: Spec::output_with(
            DType::F32,
            (Rows::Live, width),
            output_map(DType::F32, width, 4),
        ),
    };
    let mut info = prepared.info().clone();
    info.workspace = 0;
    let observed = Box::new(Observed { op: prepared, info });
    Ok(OpCase::with_counts(observed, CAPACITY, roles, |rows| {
        Counts::rows(rows).with_items(CAPACITY, 0)
    })?
    .oracle(move |images, live| oracle(query, images, live))
    .live_counts(vec![0, 1, 2, 5, 6])
    .no_write(
        "op.tail.state",
        1,
        "the first row completes its incoming group and has no live tail",
    ))
}

fn oracle(
    query: GatedPoolingQuery,
    images: &Images,
    live: u32,
) -> LabResult<Vec<(String, Expected)>> {
    let width = query.width as usize;
    let keys = images.f32s("op.key")?;
    let gates = images.f32s("op.gate")?;
    let incoming = images.f32s("op.tail.state")?;
    let mut state = images.bytes("op.tail.state")?.to_vec();
    let slots = images.i32s("op.tail.slot_ids")?;
    let positions = images.i32s("op.positions")?;
    let weights: Vec<f64> = images
        .u16s("op.norm_weight")?
        .into_iter()
        .map(|w| f64::from(bf16_to_f32(w)))
        .collect();
    let elements = CAPACITY as usize * width;
    let mut pooled =
        OracleOutput::new(vec![0.0; elements], vec![0.0; elements], 2).map_err(|e| e.0)?;
    let mut output =
        OracleOutput::new(vec![0.0; elements], vec![0.0; elements], width).map_err(|e| e.0)?;
    let mut output_positions = vec![-1; CAPACITY as usize];
    for row in 0..live as usize {
        let position = positions[row];
        if position < 0 {
            continue;
        }
        let source = slots[row] as usize * 2 * width;
        let range = row * width..(row + 1) * width;
        if position % 2 == 0 {
            let data = f32_bytes(&keys[range.clone()]);
            let scores = f32_bytes(&gates[range]);
            state[source * 4..(source + width) * 4].copy_from_slice(&data);
            state[(source + width) * 4..(source + 2 * width) * 4].copy_from_slice(&scores);
            continue;
        }
        output_positions[row] = position / 2;
        let group_keys: Vec<f32> = incoming[source..source + width]
            .iter()
            .chain(&keys[range.clone()])
            .copied()
            .collect();
        let group_gates: Vec<f32> = incoming[source + width..source + 2 * width]
            .iter()
            .chain(&gates[range.clone()])
            .copied()
            .collect();
        let reference =
            gated_pooling_oracle::pool_without_position(query, &group_keys, &group_gates)
                .map_err(|e| e.0)?;
        let norm = gated_pooling_oracle::norm_after_pool(
            query,
            &reference
                .values
                .iter()
                .map(|&v| v as f32)
                .collect::<Vec<_>>(),
            &weights,
        )
        .map_err(|e| e.0)?;
        for d in 0..width {
            let at = row * width + d;
            pooled.values[at] = reference.values[d];
            pooled.sum_abs_products[at] = reference.sum_abs_products[d];
            output.values[at] = norm.values[d];
            output.sum_abs_products[at] = norm.sum_abs_products[d];
            output.reduction_lengths[at] = norm.reduction_lengths[d];
        }
    }
    Ok(vec![
        ("pooled".into(), Expected::Values(pooled)),
        ("op.output".into(), Expected::Values(output)),
        (
            "op.output_positions".into(),
            Expected::Bits(i32_bytes(&output_positions)),
        ),
        ("op.tail.state".into(), Expected::Bits(state)),
    ])
}

pub fn family(entries: &[Entry]) -> Vec<Required<'static>> {
    entries
        .iter()
        .filter(|e| e.query().phase == RecurrentPhase::Decode)
        .map(|entry| {
            let query = entry.query();
            Required::new(
                format!("F32 pair pooling width {}", query.width),
                Need::rows(CAPACITY),
                move |backend, need| backend.gated_pooling(&query, need),
                move |prepared| case(query, prepared),
            )
            .required()
        })
        .collect()
}

/// Stateless ratio-one normalization with F32 values that BF16 cannot
/// preserve. The F32 output is bounded directly against the f64 oracle.
pub fn ratio_one_case(
    requirement: kernel_core::rmsnorm::RmsNormRequirement,
    capacity: u32,
    prepared: Prepared<kernel_core::rmsnorm::RmsNormBuffers<View>>,
) -> LabResult<OpCase<'static>> {
    use kernel_core::rmsnorm::RmsNormBuffers;
    if [requirement.input, requirement.weight, requirement.output]
        != [DType::F32, DType::Bf16, DType::F32]
        || requirement.fused_residual
    {
        return Err(
            "ratio-one case requires F32 values and BF16 weights without a residual".into(),
        );
    }
    let width = requirement.width as usize;
    let x: Vec<f32> = (0..capacity as usize * width)
        .map(|i| {
            let value = 1.0 + (i % 17) as f32 / 8.0 + ((i * 7) % 31) as f32 / 4096.0;
            if i % 3 == 0 { -value } else { value }
        })
        .collect();
    let weights: Vec<u16> = (0..width)
        .map(|i| f32_to_bf16(0.5 + (i % 7) as f32 / 4.0))
        .collect();
    let roles = RmsNormBuffers {
        status: None,
        status_row_offset: 0,
        x: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&x)),
        weight: Spec::input(DType::Bf16, (Rows::Fixed(1), width), bf16_bytes(&weights)),
        residual: None,
        y: Spec::output(DType::F32, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    Ok(
        OpCase::new(prepared, capacity, roles)?.oracle(move |images, _| {
            let input: Vec<f64> = images.f32s("x")?.into_iter().map(f64::from).collect();
            let weight: Vec<f64> = images
                .u16s("weight")?
                .into_iter()
                .map(|b| f64::from(bf16_to_f32(b)))
                .collect();
            let reference = requirement
                .oracle(&input, &weight, capacity as usize)
                .map_err(|e| e.0)?;
            Ok(vec![("y".into(), Expected::Values(reference))])
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::backend::Backend;
    use kernels_sm120::backend::Sm120;

    #[test]
    fn pooling_cases_validate_f64_pool_and_exact_tail_witnesses() {
        let mut checked = 0;
        for width in [128, 512] {
            let query = GatedPoolingQuery {
                group: 2,
                width,
                norm_eps_bits: 1e-20f32.to_bits(),
                phase: RecurrentPhase::Decode,
                capacity: CAPACITY,
            };
            let entry = Entry::new(query, "chop_pool_case").unwrap();
            let backend = Sm120::new().with_gated_pooling([entry]);
            let prepared = backend.gated_pooling(&query, Need::rows(CAPACITY)).unwrap();
            case(query, prepared).unwrap().validate().unwrap();
            checked += 1;
        }
        assert_eq!(checked, 2);
    }
}
