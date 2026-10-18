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

/// Independent f64 RMSNorm over exact F32 projection rows and BF16 weights.
pub fn window_norm_f64(
    query: LatentWindowWrite,
    input: &[f32],
    weight: &[u16],
) -> Result<OracleOutput, Refusal> {
    query.validate()?;
    crate::oracle::rmsnorm(
        &input.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
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
    input: &[f32],
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
        let total = reduce(&mut warps);
        let mean = total / width as f32;
        let inv = 1.0 / (mean + f32::from_bits(query.epsilon_bits)).sqrt();
        if !total.is_finite() || !mean.is_finite() || !inv.is_finite() {
            return Err(Refusal("window norm reduction or reciprocal overflows F32"));
        }
        output.extend(
            row.iter()
                .zip(weight)
                .map(|(&x, &w)| (x * inv) * bf16_to_f32(w)),
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
    input: &[f32],
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
/// mapping. Missing staged positions and negative query positions are holes.
pub fn window_keys(
    query: LatentWindowKeys,
    position: i32,
    first: i32,
) -> Result<Vec<i32>, Refusal> {
    query.validate()?;
    if position < 0 {
        return Ok(vec![-1; query.window as usize]);
    }
    let end = i64::from(position) + 1;
    let start = (end - i64::from(query.window)).max(0);
    let mut keys = (start..end)
        .map(|p| {
            query
                .addressing
                .slot(p as i32, first)
                .map_or(-1, |v| v as i32)
        })
        .collect::<Vec<_>>();
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

/// Exact BF16 bytes in native channel order. This is a decoder, not a second
/// quantizer. The native product admission is retained, including finite,
/// exact and normal-or-zero BF16 products.
pub fn exact_window_bf16(encoded: &EncodedLatent, width: usize) -> Result<Vec<u8>, Refusal> {
    use crate::latent_attention::LatentKvFormat;
    use crate::latent_attention_oracle::decode_native_page;
    if encoded.values.len() != width || encoded.scale_bytes.len() != width / 32 {
        return Err(Refusal("native window encoded row shape mismatch"));
    }
    let mut page = encoded.values.clone();
    page.extend_from_slice(&encoded.scale_bytes);
    let values = decode_native_page(LatentKvFormat::Fp8E4M3Ue8M0Block32, width, 1, &page, 0)?;
    let mut bytes = Vec::with_capacity(width * 2);
    for value in values {
        let bits = crate::codec::f32_to_bf16(value as f32);
        if bits & 0x7fff != 0 && bits & 0x7f80 == 0 {
            return Err(Refusal("window native product must be zero or normal BF16"));
        }
        bytes.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(bytes)
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
    let bf16 = query.kv == crate::latent_attention::LatentKvFormat::Bf16;
    let stored_width = crate::latent_glue::window_storage_width(query.kv, query.latent)? as usize;
    let slots = query.addressing.slots() as usize;
    let sequences = first_positions.len();
    if encoded.len() != positions.len()
        || positions.len() != row_batch.len()
        || sequences == 0
        || last_positions.len() != sequences
        || code_sequence_stride < slots * stored_width
        || (!bf16 && scale_sequence_stride < slots * width / 32)
        || code_sequence_stride
            .checked_mul(sequences.saturating_sub(1))
            .and_then(|n| n.checked_add(slots * stored_width))
            .is_none_or(|n| codes.len() < n)
        || (!bf16
            && scale_sequence_stride
                .checked_mul(sequences.saturating_sub(1))
                .and_then(|n| n.checked_add(slots * width / 32))
                .is_none_or(|n| scales.len() < n))
    {
        return Err(Refusal("native window write storage or row shape mismatch"));
    }
    let mut destinations = Vec::with_capacity(encoded.len());
    let mut unique = std::collections::BTreeSet::new();
    for (row, (&position, &batch)) in positions.iter().zip(row_batch).enumerate() {
        let sequence = batch as usize;
        if sequence >= sequences || position < 0 || position > last_positions[sequence] {
            continue;
        }
        if encoded[row].values.len() != width || encoded[row].scale_bytes.len() != width / 32 {
            return Err(Refusal("native window encoded row shape mismatch"));
        }
        if matches!(query.addressing, WindowAddressing::Circular { .. })
            && i64::from(position) + slots as i64 <= i64::from(last_positions[sequence])
        {
            continue;
        }
        let Ok(slot) = query.addressing.slot(position, first_positions[sequence]) else {
            continue;
        };
        let slot = slot as usize;
        if !unique.insert((sequence, slot)) {
            return Err(Refusal("native window write repeats a destination"));
        }
        destinations.push((
            row,
            sequence * code_sequence_stride + slot * stored_width,
            if bf16 {
                0
            } else {
                sequence * scale_sequence_stride + slot * width / 32
            },
        ));
    }
    // Validate every exact product before committing any destination.
    let products = if bf16 {
        destinations
            .iter()
            .map(|&(row, _, _)| exact_window_bf16(&encoded[row], width))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    for (index, (row, code, scale)) in destinations.into_iter().enumerate() {
        if bf16 {
            codes[code..code + stored_width].copy_from_slice(&products[index]);
            continue;
        }
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
    fn projection_stays_f32_until_the_trained_quantizer() {
        let op = LatentWindowWrite {
            kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let mut input = vec![1.0; 512];
        input[0] = 1.00375;
        let mut weight = vec![f32_to_bf16(1.0); 512];
        weight[0] = f32_to_bf16(1.0625);
        let table = [1.0, 0.0].repeat(32);
        let (_, kept) = window_rows(op, &input, &weight, &table).unwrap();
        let narrowed = input
            .iter()
            .map(|&v| bf16_to_f32(f32_to_bf16(v)))
            .collect::<Vec<_>>();
        let (_, early) = window_rows(op, &narrowed, &weight, &table).unwrap();
        assert_eq!((kept[0].values[0], kept[0].scale_bytes[0]), (0x79, 119));
        assert_eq!((early[0].values[0], early[0].scale_bytes[0]), (0x78, 119));
        let ideal = window_norm_f64(op, &input, &weight).unwrap();
        crate::oracle::check_numeric(
            &window_norm_f32(op, &input, &weight)
                .unwrap()
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>(),
            &ideal,
            crate::contract::DType::F32,
        )
        .unwrap();
    }

    #[test]
    fn window_quantizer_matches_native_encoder_after_rotation() {
        let (input, table) = operands(1);
        let op = LatentWindowWrite {
            kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
            epsilon_bits: 1e-20_f32.to_bits(),
            latent: 512,
            rotary: 64,
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let weight = (0..512)
            .map(|i| f32_to_bf16(0.7 + (i % 19) as f32 / 23.0))
            .collect::<Vec<_>>();
        let normed = window_norm_f32(op, &input, &weight).unwrap();
        crate::oracle::check_numeric(
            &normed.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &window_norm_f64(op, &input, &weight).unwrap(),
            crate::contract::DType::F32,
        )
        .unwrap();
        let (rotated, encoded) = window_rows(op, &input, &weight, &table).unwrap();
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
        let tiny = vec![1e-11; 512];
        let exact_eps = window_rows(op, &tiny, &weight, &table[..64]).unwrap().1;
        let wrong_eps = LatentWindowWrite {
            kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
            epsilon_bits: 1e-6_f32.to_bits(),
            ..op
        };
        assert_ne!(
            exact_eps,
            window_rows(wrong_eps, &tiny, &weight, &table[..64])
                .unwrap()
                .1
        );
        let zero = window_rows(op, &[0.0; 512], &weight, &table[..64])
            .unwrap()
            .1;
        let positive_zero =
            encode_latent_row(LatentKvFormat::Fp8E4M3Ue8M0Block32, &[0; 512]).unwrap();
        assert_eq!(zero[0].scale_bytes, positive_zero.scale_bytes);
        assert!(zero[0].values.iter().all(|v| v & 0x7f == 0));
        // Rotation can create negative zero. Clearing its sign is a mutant.
        assert_ne!(zero[0].values, positive_zero.values);
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
            window_rows(op, &input, &vec![f32_to_bf16(1.0); 512], &table)
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
        assert_eq!(window_keys(ring, -1, 0).unwrap(), vec![-1; 128]);
        let staged = LatentWindowKeys {
            addressing: WindowAddressing::Staged { slots: 160 },
            ..ring
        };
        assert_eq!(
            window_keys(staged, 256, 128).unwrap(),
            (1..=128).collect::<Vec<_>>()
        );
        assert_eq!(window_keys(staged, 255, 129).unwrap()[0], -1);
        assert_eq!(window_keys(ring, -2, 0).unwrap(), vec![-1; 128]);
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
            kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
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

/// Rotated F32 rows, native BF16 rows and their sticky status words.
pub type QueryStatusRows = (Vec<f32>, Vec<u16>, Vec<u32>);
/// Normalized F32 rows, encoded cache rows and their sticky status words.
pub type WindowStatusRows = (Vec<f32>, Vec<EncodedLatent>, Vec<u32>);

/// Sticky status reference. Failed query rows preserve every input bit and
/// produce positive-zero native Q, including when another head raised the flag.
pub fn query_rope_status(
    query: LatentQueryRope,
    input: &[f32],
    table: &[f32],
    prior: &[u32],
) -> Result<QueryStatusRows, Refusal> {
    use crate::precision::quantizer::{NONFINITE_INPUT, UNREPRESENTABLE_OUTPUT};
    query.validate()?;
    let width = query.heads as usize * query.latent as usize;
    let tail = query.rotary as usize;
    if input.len() != prior.len() * width || table.len() != prior.len() * tail {
        return Err(Refusal("query status operand shape mismatch"));
    }
    let mut rotated = input.to_vec();
    let mut native = vec![0; input.len()];
    let mut status = prior.to_vec();
    for row in 0..prior.len() {
        let at = row * width..(row + 1) * width;
        let angles = &table[row * tail..(row + 1) * tail];
        let mut values = input[at.clone()].to_vec();
        let mut bits = vec![0; width];
        let mut current = 0;
        for head in 0..query.heads as usize {
            for column in (0..query.latent as usize).step_by(2) {
                let pair = head * query.latent as usize + column;
                let (a, b) = (input[at.start + pair], input[at.start + pair + 1]);
                let mut invalid = !a.is_finite() || !b.is_finite();
                let (x, y) = if column >= query.latent as usize - tail {
                    let angle = column - (query.latent as usize - tail);
                    let (c, s) = (angles[angle], angles[angle + 1]);
                    invalid |= !c.is_finite() || !s.is_finite();
                    (a * c - b * s, a * s + b * c)
                } else {
                    (a, b)
                };
                values[pair] = x;
                values[pair + 1] = y;
                bits[pair] = f32_to_bf16(x);
                bits[pair + 1] = f32_to_bf16(y);
                current |= if invalid {
                    NONFINITE_INPUT
                } else if !x.is_finite()
                    || !y.is_finite()
                    || !bf16_to_f32(bits[pair]).is_finite()
                    || !bf16_to_f32(bits[pair + 1]).is_finite()
                {
                    UNREPRESENTABLE_OUTPUT
                } else {
                    0
                };
            }
        }
        status[row] = prior[row] | current;
        if status[row] == 0 {
            rotated[at.clone()].copy_from_slice(&values);
            native[at].copy_from_slice(&bits);
        }
    }
    Ok((rotated, native, status))
}

/// The same norm/rotation boundary followed by the shared quantizer's
/// independent status oracle. The caller skips storage for nonzero flags.
pub fn window_rows_status(
    query: LatentWindowWrite,
    input: &[f32],
    weight: &[u16],
    table: &[f32],
    prior: &[u32],
) -> Result<WindowStatusRows, Refusal> {
    use crate::contract::DType;
    use crate::precision::quantizer::{
        BlockQuantization, BlockQuantizeQuery, NONFINITE_INPUT, QuantizeOutput,
        UNREPRESENTABLE_OUTPUT,
    };
    query.validate()?;
    let width = query.latent as usize;
    let tail = query.rotary as usize;
    if input.len() != prior.len() * width
        || table.len() != prior.len() * tail
        || weight.len() != width
    {
        return Err(Refusal("window status operand shape mismatch"));
    }
    let quantizer = BlockQuantizeQuery {
        width: query.latent,
        input: DType::F32,
        format: BlockQuantization::E4M3_CEIL32,
        output: QuantizeOutput::Codes,
    };
    let mut latent = vec![0.0; input.len()];
    let mut status = prior.to_vec();
    let mut encoded = Vec::new();
    for row in 0..prior.len() {
        let at = row * width..(row + 1) * width;
        let angles = &table[row * tail..(row + 1) * tail];
        let mut current = 0;
        let values = if input[at.clone()]
            .iter()
            .chain(angles)
            .any(|v| !v.is_finite())
            || weight.iter().any(|&v| !bf16_to_f32(v).is_finite())
        {
            current = NONFINITE_INPUT;
            None
        } else {
            let result = window_norm_f32(query, &input[at.clone()], weight).and_then(|norm| {
                query_rope_f32(
                    LatentQueryRope {
                        heads: 1,
                        latent: query.latent,
                        rotary: query.rotary,
                    },
                    &norm,
                    angles,
                )
            });
            match result {
                Ok((values, _)) => Some(values),
                Err(_) => {
                    current = UNREPRESENTABLE_OUTPUT;
                    None
                }
            }
        };
        status[row] = prior[row] | current;
        if status[row] == 0 {
            latent[at.clone()]
                .copy_from_slice(&values.expect("a successful row has normalized values"));
        }
        status[row] = quantizer.row_status(&latent[at.clone()], &[status[row]])?[0];
        encoded.push(if status[row] == 0 {
            encode_latent_row(
                LatentKvFormat::Fp8E4M3Ue8M0Block32,
                &latent[at]
                    .iter()
                    .map(|&v| f32_to_bf16(v))
                    .collect::<Vec<_>>(),
            )?
        } else {
            EncodedLatent {
                values: vec![0; width],
                scales: vec![],
                scale_bytes: vec![127; width / 32],
            }
        });
    }
    Ok((latent, encoded, status))
}

#[cfg(test)]
mod status_tests {
    use super::*;
    #[test]
    fn nonfinite_glue_reports_flags_without_changing_query_or_storing_a_key() {
        let q = LatentQueryRope {
            heads: 3,
            latent: 512,
            rotary: 64,
        };
        let mut input = vec![1.0; 3 * 512];
        input[2 * 512 + 17] = f32::NAN;
        let (kept, native, status) =
            query_rope_status(q, &input, &[1.0, 0.0].repeat(32), &[0x80]).unwrap();
        assert_eq!(status, [0x81]);
        assert!(native.iter().all(|&v| v == 0));
        assert_eq!(
            kept.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let q = LatentWindowWrite {
            kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        input.truncate(512);
        input[0] = f32::NAN;
        let (latent, encoded, status) =
            window_rows_status(q, &input, &vec![0x3f80; 512], &[1.0, 0.0].repeat(32), &[0])
                .unwrap();
        assert_eq!(status, [1]);
        assert!(latent.iter().all(|v| v.to_bits() == 0));
        assert!(encoded[0].values.iter().all(|&v| v == 0));
        assert!(encoded[0].scale_bytes.iter().all(|&v| v == 127));
    }
}

#[cfg(test)]
mod status_boundary_tests {
    use super::*;
    fn query() -> LatentWindowWrite {
        LatentWindowWrite {
            kv: crate::latent_attention::LatentKvFormat::Fp8E4M3Ue8M0Block32,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        }
    }
    #[test]
    fn status_boundary_window_keeps_a_finite_fused_reduction() {
        let mut input = vec![0.0; 512];
        input[0] = f32::from_bits(0x5f7ffff3);
        input[1] = f32::from_bits(0x5a9ffffe);
        assert_eq!(
            input[1].mul_add(input[1], input[0] * input[0]).to_bits(),
            0x7f7fffff
        );
        let (output, encoded, status) = window_rows_status(
            query(),
            &input,
            &vec![0x3f80; 512],
            &[1.0, 0.0].repeat(32),
            &[0],
        )
        .unwrap();
        assert_eq!(status, [0]);
        assert!(output[0] > 20.0);
        assert_ne!(encoded[0].values[0], 0);
    }
    #[test]
    fn status_boundary_window_accumulates_current_overflow_with_prior_flags() {
        for (value, weight) in [(f32::MAX, 0x3f80), (1.0, 0x7f7f)] {
            let mut input = vec![0.0; 512];
            input[0] = value;
            let (output, _, status) = window_rows_status(
                query(),
                &input,
                &vec![weight; 512],
                &[1.0, 0.0].repeat(32),
                &[1],
            )
            .unwrap();
            assert_eq!(status, [5]);
            assert!(output.iter().all(|v| v.to_bits() == 0));
        }
    }
    #[test]
    fn status_boundary_query_keeps_current_flags_independent_of_prior_and_other_pairs() {
        let q = LatentQueryRope {
            heads: 3,
            latent: 512,
            rotary: 64,
        };
        let mut input = vec![0.0; 3 * 512];
        input[0] = f32::MAX;
        let (_, _, status) = query_rope_status(q, &input, &[1.0, 0.0].repeat(32), &[1]).unwrap();
        assert_eq!(status, [5]);
        input[2 * 512 + 511] = f32::NAN;
        let (_, _, status) = query_rope_status(q, &input, &[1.0, 0.0].repeat(32), &[0]).unwrap();
        assert_eq!(status, [5]);
    }
}

#[cfg(test)]
mod bf16_window_tests {
    use super::*;
    use crate::codec::{bf16_to_f32, f32_to_bf16};
    use crate::latent_attention::LatentKvFormat;
    use crate::latent_attention_oracle::{decode_native_page, encode_latent_row};
    #[test]
    fn bf16_store_is_exact_native_product_with_signed_zero_and_no_scale_plane() {
        let native = LatentKvFormat::Fp8E4M3Ue8M0Block32;
        let input = (0..512)
            .map(|i| {
                f32_to_bf16(if i == 0 {
                    -0.0
                } else {
                    ((i * 13) % 127) as f32 / 17.0 - 3.5
                })
            })
            .collect::<Vec<_>>();
        let row = encode_latent_row(native, &input).unwrap();
        let query = LatentWindowWrite {
            kv: LatentKvFormat::Bf16,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let stride = 136 * 1024 + 66;
        let mut out = vec![0xa5; stride * 2];
        store_window_rows(
            query,
            std::slice::from_ref(&row),
            &[137],
            &[1],
            &[0, 0],
            &[-1, 137],
            &mut out,
            &mut [],
            stride,
            usize::MAX,
        )
        .unwrap();
        let mut page = row.values.clone();
        page.extend_from_slice(&row.scale_bytes);
        let decoded = decode_native_page(native, 512, 1, &page, 0).unwrap();
        let begin = stride + 1024;
        for (i, value) in decoded.iter().enumerate() {
            let bits =
                u16::from_le_bytes(out[begin + i * 2..begin + i * 2 + 2].try_into().unwrap());
            assert_eq!(bf16_to_f32(bits).to_bits(), (*value as f32).to_bits());
        }
        assert_eq!(out[begin..begin + 2], [0, 128]);
        assert!(
            out[..begin]
                .iter()
                .chain(&out[begin + 1024..])
                .all(|&v| v == 0xa5)
        );
    }
    #[test]
    fn bf16_products_keep_native_refusals_and_do_not_partially_commit() {
        for (code, scale) in [(0x7f, 127), (0x7e, 254), (0x01, 0), (0x08, 1), (0x38, 255)] {
            let row = EncodedLatent {
                values: vec![code; 512],
                scales: vec![],
                scale_bytes: vec![scale; 16],
            };
            assert!(
                exact_window_bf16(&row, 512).is_err(),
                "code {code} scale {scale}"
            );
        }
        let row = EncodedLatent {
            values: vec![0; 512],
            scales: vec![],
            scale_bytes: vec![127; 16],
        };
        assert_eq!(exact_window_bf16(&row, 512).unwrap(), vec![0; 1024]);
        let query = LatentWindowWrite {
            kv: LatentKvFormat::Bf16,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Staged { slots: 2 },
        };
        let mut bad = row.clone();
        bad.values[511] = 0x7f;
        let mut destination = vec![0xa5; 2048];
        assert!(
            store_window_rows(
                query,
                &[row, bad],
                &[0, 1],
                &[0, 0],
                &[0],
                &[1],
                &mut destination,
                &mut [],
                2048,
                0
            )
            .is_err()
        );
        assert_eq!(destination, vec![0xa5; 2048]);
    }
}
