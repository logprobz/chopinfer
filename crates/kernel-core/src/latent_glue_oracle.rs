//! Independent numerical and storage references for latent attention glue.

use crate::codec::{bf16_to_f32, f32_to_bf16};
use crate::contract::{Refusal, Validate};
use crate::latent_attention::LatentKvFormat;
use crate::latent_attention_oracle::{EncodedLatent, encode_latent_row};
use crate::latent_glue::{LatentQueryRope, LatentWindowKeys, LatentWindowWrite, WindowAddressing};
use crate::oracle::OracleOutput;

fn shape(query: LatentQueryRope, input: &[f32], table: &[f32]) -> Result<usize, Refusal> {
    query.validate()?;
    let width = query.heads as usize * query.latent as usize;
    if !input.len().is_multiple_of(width)
        || table.len() != input.len() / width * query.rotary as usize
        || input.iter().chain(table).any(|v| !v.is_finite())
    {
        return Err(Refusal(
            "latent tail rotation operand shape or value mismatch",
        ));
    }
    Ok(input.len() / width)
}

/// Ideal rotation of the supplied F32 operands, with an f64 two-product
/// bound. The prefix's values are unchanged, without an arithmetic step.
pub fn query_rope_f64(
    query: LatentQueryRope,
    input: &[f32],
    table: &[f32],
) -> Result<OracleOutput, Refusal> {
    let rows = shape(query, input, table)?;
    let (heads, width, tail) = (
        query.heads as usize,
        query.latent as usize,
        query.rotary as usize,
    );
    let mut values = input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let mut magnitudes = values.iter().map(|v| v.abs()).collect::<Vec<_>>();
    for row in 0..rows {
        for head in 0..heads {
            for pair in 0..tail / 2 {
                let at = (row * heads + head) * width + width - tail + 2 * pair;
                let (a, b) = (f64::from(input[at]), f64::from(input[at + 1]));
                let (c, s) = (
                    f64::from(table[row * tail + pair * 2]),
                    f64::from(table[row * tail + pair * 2 + 1]),
                );
                values[at] = a * c - b * s;
                values[at + 1] = a * s + b * c;
                magnitudes[at] = (a * c).abs() + (b * s).abs();
                magnitudes[at + 1] = (a * s).abs() + (b * c).abs();
            }
        }
    }
    OracleOutput::new(values, magnitudes, 2)
}

/// The contract's separate F32 products and add/subtract, then the one
/// BF16 Q boundary. No input value or table entry rounds to BF16 first.
pub fn query_rope_f32(
    query: LatentQueryRope,
    input: &[f32],
    table: &[f32],
) -> Result<(Vec<f32>, Vec<u16>), Refusal> {
    let rows = shape(query, input, table)?;
    let (heads, width, tail) = (
        query.heads as usize,
        query.latent as usize,
        query.rotary as usize,
    );
    let mut rotated = input.to_vec();
    for row in 0..rows {
        for head in 0..heads {
            for pair in 0..tail / 2 {
                let at = (row * heads + head) * width + width - tail + 2 * pair;
                let (a, b) = (input[at], input[at + 1]);
                let (c, s) = (
                    table[row * tail + pair * 2],
                    table[row * tail + pair * 2 + 1],
                );
                rotated[at] = a * c - b * s;
                rotated[at + 1] = a * s + b * c;
            }
        }
    }
    if rotated.iter().any(|v| !v.is_finite()) {
        return Err(Refusal("latent tail rotation overflows F32"));
    }
    let bf16 = rotated.iter().map(|&v| f32_to_bf16(v)).collect();
    Ok((rotated, bf16))
}

/// Independent f64 RMSNorm over the exact BF16 operands.
pub fn window_norm_f64(
    query: LatentWindowWrite,
    input: &[u16],
    weight: &[u16],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    crate::oracle::rmsnorm(
        &input
            .iter()
            .map(|&v| f64::from(bf16_to_f32(v)))
            .collect::<Vec<_>>(),
        query.latent as usize,
        &weight
            .iter()
            .map(|&v| f64::from(bf16_to_f32(v)))
            .collect::<Vec<_>>(),
        f64::from(f32::from_bits(query.epsilon_bits)),
        crate::contract::RmsNormKind::Plain,
        None,
    )
}

/// The pinned F32 norm order for the bitwise cache boundary. Each lane
/// owns adjacent pairs, traversing later 512-column spans in order.
pub fn window_norm_f32(
    query: LatentWindowWrite,
    input: &[u16],
    weight: &[u16],
) -> Result<Vec<f32>, Refusal> {
    query.validate()?;
    let width = query.latent as usize;
    if !input.len().is_multiple_of(width) || weight.len() != width {
        return Err(Refusal("window norm operand shape mismatch"));
    }
    let mut output = Vec::with_capacity(input.len());
    for row in input.chunks_exact(width) {
        let mut lanes = [0.0_f32; 256];
        for (lane, sum) in lanes.iter_mut().enumerate() {
            for start in (2 * lane..width).step_by(512) {
                for &v in &row[start..start + 2] {
                    let v = bf16_to_f32(v);
                    *sum = v.mul_add(v, *sum);
                }
            }
        }
        let reduce = |v: &mut [f32]| {
            for d in [16, 8, 4, 2, 1] {
                for i in 0..d {
                    v[i] += v[i + d];
                }
            }
            v[0]
        };
        let mut warps = [0.0_f32; 32];
        for (i, warp) in lanes.chunks_exact_mut(32).enumerate() {
            warps[i] = reduce(warp);
        }
        let mean = reduce(&mut warps) / width as f32;
        let inv = 1.0 / (mean + f32::from_bits(query.epsilon_bits)).sqrt();
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(&x, &w)| (bf16_to_f32(x) * inv) * bf16_to_f32(w)),
        );
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Err(Refusal("window norm overflows F32"));
    }
    Ok(output)
}

/// The native cache encoder is the bitwise oracle for the shared row
/// quantizer. It receives precisely the post-rotation BF16 values.
pub fn window_rows(
    query: LatentWindowWrite,
    input: &[u16],
    weight: &[u16],
    table: &[f32],
) -> Result<(Vec<f32>, Vec<EncodedLatent>), Refusal> {
    query.validate()?;
    let (rotated, rounded) = query_rope_f32(
        LatentQueryRope {
            heads: 1,
            latent: query.latent,
            rotary: query.rotary,
        },
        &window_norm_f32(query, input, weight)?,
        table,
    )?;
    let encoded = rounded
        .chunks_exact(query.latent as usize)
        .map(|row| encode_latent_row(LatentKvFormat::Fp8E4M3Ue8M0Block32, row))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((rotated, encoded))
}

/// One causal row, with absolute positions increasing before their storage
/// mapping. Staging must contain the entire visible interval, including
/// any retained prefix before a continuation chunk.
pub fn window_keys(
    query: LatentWindowKeys,
    position: i32,
    first: i32,
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    if position < 0 {
        return Err(Refusal("window query positions must be nonnegative"));
    }
    let end = i64::from(position) + 1;
    let start = (end - i64::from(query.window)).max(0);
    let mut keys = (start..end)
        .map(|p| query.addressing.slot(p as i32, first).map(|v| v as i32))
        .collect::<Result<Vec<_>, _>>()?;
    keys.resize(query.window as usize, -1);
    Ok(keys)
}

/// Strict builder gate. A reader may safely skip an out-of-range slot,
/// but a builder emitting one is wrong. A valid stale slot is also wrong:
/// range checks alone do not establish visibility or oldest-first order.
pub fn check_window_keys(
    query: LatentWindowKeys,
    position: i32,
    first: i32,
    actual: &[i32],
) -> Result<(), Refusal> {
    query.validate()?;
    if actual.len() != query.window as usize
        || actual
            .iter()
            .any(|&slot| slot != -1 && (slot < 0 || slot as u32 >= query.addressing.slots()))
    {
        return Err(Refusal("window-key oracle: outside slot domain"));
    }
    if actual != window_keys(query, position, first)? {
        return Err(Refusal(
            "window-key oracle: visible positions or order differ",
        ));
    }
    Ok(())
}

/// Write through independent code/scale sequence strides. The oracle
/// validates every destination before changing either plane. Padding and
/// all slots absent from the write set preserve their original bytes.
#[allow(clippy::too_many_arguments)]
pub fn store_window_rows(
    query: LatentWindowWrite,
    encoded: &[EncodedLatent],
    positions: &[i32],
    row_batch: &[u32],
    first_positions: &[i32],
    last_positions: &[i32],
    codes: &mut [u8],
    scales: &mut [u8],
    code_sequence_stride: usize,
    scale_sequence_stride: usize,
) -> Result<(), Refusal> {
    query.validate()?;
    let width = query.latent as usize;
    let slots = query.addressing.slots() as usize;
    let sequences = first_positions.len();
    if encoded.len() != positions.len()
        || positions.len() != row_batch.len()
        || sequences == 0
        || last_positions.len() != sequences
        || code_sequence_stride < slots * width
        || scale_sequence_stride < slots * width / 32
        || code_sequence_stride
            .checked_mul(sequences.saturating_sub(1))
            .and_then(|n| n.checked_add(slots * width))
            .is_none_or(|n| codes.len() < n)
        || scale_sequence_stride
            .checked_mul(sequences.saturating_sub(1))
            .and_then(|n| n.checked_add(slots * width / 32))
            .is_none_or(|n| scales.len() < n)
    {
        return Err(Refusal("native window write storage or row shape mismatch"));
    }
    let mut destinations = Vec::with_capacity(encoded.len());
    let mut unique = std::collections::BTreeSet::new();
    for (row, (&position, &batch)) in positions.iter().zip(row_batch).enumerate() {
        let sequence = batch as usize;
        if sequence >= sequences
            || position < 0
            || position > last_positions[sequence]
            || encoded[row].values.len() != width
            || encoded[row].scale_bytes.len() != width / 32
        {
            return Err(Refusal("native window write row lies outside its sequence"));
        }
        if matches!(query.addressing, WindowAddressing::Circular { .. })
            && i64::from(position) + slots as i64 <= i64::from(last_positions[sequence])
        {
            continue;
        }
        let slot = query.addressing.slot(position, first_positions[sequence])? as usize;
        if !unique.insert((sequence, slot)) {
            return Err(Refusal("native window write repeats a destination"));
        }
        destinations.push((
            row,
            sequence * code_sequence_stride + slot * width,
            sequence * scale_sequence_stride + slot * width / 32,
        ));
    }
    for (row, code, scale) in destinations {
        codes[code..code + width].copy_from_slice(&encoded[row].values);
        scales[scale..scale + width / 32].copy_from_slice(&encoded[row].scale_bytes);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::bf16_to_f32;
    use crate::precision::quantizer::BlockQuantization;
    use crate::quantize::Scales;

    fn query() -> LatentQueryRope {
        LatentQueryRope {
            heads: 2,
            latent: 512,
            rotary: 64,
        }
    }
    fn operands(heads: usize) -> (Vec<f32>, Vec<f32>) {
        let x = (0..2 * heads * 512)
            .map(|i| ((i % 31) as f32 - 15.0) / 7.0 + 0.001)
            .collect();
        let table = (0..2 * 32)
            .flat_map(|i| {
                let angle = (i as f64 + 0.3) * 0.271;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .collect();
        (x, table)
    }

    #[test]
    fn query_preserves_f32_until_its_explicit_native_boundary() {
        let (input, table) = operands(2);
        let (rotated, native) = query_rope_f32(query(), &input, &table).unwrap();
        for row in 0..4 {
            assert_eq!(
                &rotated[row * 512..row * 512 + 448],
                &input[row * 512..row * 512 + 448]
            );
        }
        assert_eq!(
            native,
            rotated.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>()
        );
        let early = input
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        assert_ne!(native, query_rope_f32(query(), &early, &table).unwrap().1);
        let narrowed_table = table
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        assert_ne!(
            native,
            query_rope_f32(query(), &input, &narrowed_table).unwrap().1
        );
        let ideal = query_rope_f64(query(), &input, &table).unwrap();
        crate::oracle::check_numeric(
            &rotated.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &ideal,
            crate::contract::DType::F32,
        )
        .unwrap();
    }

    #[test]
    fn window_quantizer_matches_native_encoder_after_rotation() {
        let (input, table) = operands(1);
        let op = LatentWindowWrite {
            epsilon_bits: 1e-20_f32.to_bits(),
            latent: 512,
            rotary: 64,
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let bf16 = input.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>();
        let weight = (0..512)
            .map(|i| f32_to_bf16(0.7 + (i % 19) as f32 / 23.0))
            .collect::<Vec<_>>();
        let normed = window_norm_f32(op, &bf16, &weight).unwrap();
        crate::oracle::check_numeric(
            &normed.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &window_norm_f64(op, &bf16, &weight).unwrap(),
            crate::contract::DType::F32,
        )
        .unwrap();
        let (rotated, encoded) = window_rows(op, &bf16, &weight, &table).unwrap();
        let quantized = BlockQuantization::E4M3_CEIL32
            .apply(&rotated, 2, 512)
            .unwrap();
        let Scales::Ue8M0(scales) = quantized.scales else {
            panic!("UE8M0 output")
        };
        assert_eq!(
            quantized.codes,
            encoded
                .iter()
                .flat_map(|r| r.values.clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            scales,
            encoded
                .iter()
                .flat_map(|r| r.scale_bytes.clone())
                .collect::<Vec<_>>()
        );
        let pre_rope = BlockQuantization::E4M3_CEIL32
            .apply(&normed, 2, 512)
            .unwrap();
        assert_ne!(pre_rope.codes, quantized.codes);
        let rounded = normed
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        let early = query_rope_f32(
            LatentQueryRope {
                heads: 1,
                latent: 512,
                rotary: 64,
            },
            &rounded,
            &table,
        )
        .unwrap()
        .0;
        assert_ne!(
            BlockQuantization::E4M3_CEIL32
                .apply(&early, 2, 512)
                .unwrap()
                .codes,
            quantized.codes
        );
        assert_ne!(
            window_rows(op, &bf16, &vec![f32_to_bf16(1.0); 512], &table)
                .unwrap()
                .1,
            encoded
        );
    }

    #[test]
    fn causal_keys_keep_absolute_order_across_ring_wrap_and_staging() {
        let ring = LatentWindowKeys {
            window: 128,
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        assert_eq!(&window_keys(ring, 1, 0).unwrap()[..4], &[0, 1, -1, -1]);
        let keys = window_keys(ring, 136, 0).unwrap();
        assert_eq!(&keys[..3], &[9, 10, 11]);
        assert_eq!(keys[127], 0);
        let wrong = LatentWindowKeys {
            addressing: WindowAddressing::Circular { slots: 128 },
            ..ring
        };
        assert_ne!(keys, window_keys(wrong, 136, 0).unwrap());
        assert!(window_keys(ring, -1, 0).is_err());
        let staged = LatentWindowKeys {
            addressing: WindowAddressing::Staged { slots: 160 },
            ..ring
        };
        assert_eq!(
            window_keys(staged, 256, 128).unwrap(),
            (1..=128).collect::<Vec<_>>()
        );
        assert!(window_keys(staged, 255, 129).is_err());
        assert!(window_keys(ring, -2, 0).is_err());
        assert_eq!(window_keys(ring, i32::MAX, 0).unwrap().len(), 128);
    }

    #[test]
    fn key_builder_rejects_out_of_range_and_stale_slots_without_wrapping_them() {
        let query = LatentWindowKeys {
            window: 128,
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        for p in [0, 1, 127, 128, 135, 136, 137, 272, 273, 70_001, i32::MAX] {
            let keys = window_keys(query, p, 0).unwrap();
            check_window_keys(query, p, 0, &keys).unwrap();
            let valid = keys
                .iter()
                .copied()
                .filter(|&v| v >= 0)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(valid.len(), keys.iter().filter(|&&v| v >= 0).count());
            let mut outside = keys.clone();
            outside[0] = 136;
            assert!(
                check_window_keys(query, p, 0, &outside)
                    .unwrap_err()
                    .0
                    .contains("outside slot domain")
            );
            outside[0] = i32::MAX;
            assert!(check_window_keys(query, p, 0, &outside).is_err());
            let mut stale = keys;
            // This physical cell's latest resident is just older than the
            // visible window, or uninitialized while the sequence fills.
            stale[0] = (i64::from(p) - 128).rem_euclid(136) as i32;
            assert!(
                check_window_keys(query, p, 0, &stale)
                    .unwrap_err()
                    .0
                    .contains("visible positions")
            );
        }
    }

    #[test]
    fn ring_seeding_uses_independent_planes_and_preserves_poisoned_padding() {
        let op = LatentWindowWrite {
            epsilon_bits: 1e-20_f32.to_bits(),
            latent: 512,
            rotary: 64,
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let encoded = [1.0, 2.0, 4.0].map(|value| {
            encode_latent_row(
                LatentKvFormat::Fp8E4M3Ue8M0Block32,
                &[f32_to_bf16(value); 512],
            )
            .unwrap()
        });
        let (cs, ss) = (136 * 512 + 64, 136 * 16 + 32);
        let mut codes = vec![0xa5; 2 * cs];
        let mut scales = vec![0xff; 2 * ss];
        store_window_rows(
            op,
            &encoded,
            &[0, 136, 137],
            &[1, 1, 1],
            &[0, 0],
            &[0, 137],
            &mut codes,
            &mut scales,
            cs,
            ss,
        )
        .unwrap();
        assert!(codes[..cs].iter().all(|&v| v == 0xa5));
        assert!(scales[..ss].iter().all(|&v| v == 0xff));
        assert_eq!(&codes[cs..cs + 512], &encoded[1].values);
        assert_eq!(&scales[ss..ss + 16], &encoded[1].scale_bytes);
        assert_eq!(&codes[cs + 512..cs + 1024], &encoded[2].values);
        assert!(codes[cs + 1024..].iter().all(|&v| v == 0xa5));
        assert!(scales[ss + 32..].iter().all(|&v| v == 0xff));
        let before = codes.clone();
        assert!(
            store_window_rows(
                op,
                &encoded[..2],
                &[136, 136],
                &[1, 1],
                &[0, 0],
                &[0, 137],
                &mut codes,
                &mut scales,
                cs,
                ss
            )
            .is_err()
        );
        assert_eq!(codes, before);
    }
}
