//! Deterministic special-value operands shared by all paired arms.
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
/// F32 rows; "finite" is activation-like, "spread" spans many binades.
pub(super) fn rows_of(width: usize, rows: u32, pattern: &str, mut state: u64) -> Vec<u8> {
    (0..width * rows as usize)
        .flat_map(|i| {
            let state = next(&mut state);
            let sign = (state as u32) & 0x8000_0000;
            let mantissa = state as u32 & 0x7f_ffff;
            let at = i % width;
            let first = i / width == 0;
            let bits = match pattern {
                "spread" => sign | (((state >> 32) as u32 % 100 + 77) << 23) | mantissa,
                "wide" => sign | (((state >> 32) as u32 % 254 + 1) << 23) | mantissa,
                "zero" => sign,
                "denormal" => sign | mantissa.max(1),
                "maximum" => sign | 0x7f7f_ffff,
                "nan_block" if first && at >= width - 32 => sign | 0x7fc1_0000,
                "inf_block" if first && at >= width - 32 => sign | 0x7f80_0000,
                "nan_first" if first && at == 0 => 0x7fc1_0000,
                "nan_last" if first && at == width - 1 => 0xff81_0000,
                "inf_first" if first && at == 0 => 0x7f80_0000,
                "inf_last" if first && at == width - 1 => 0xff80_0000,
                "boundaries" => {
                    let e = ((i / 32) % 40) as i32 - 20;
                    let v = 2.0f32.powi(e).to_bits();
                    sign | match i % 3 {
                        0 => v - 1,
                        1 => v,
                        _ => v + 1,
                    }
                }
                _ => sign | (((state >> 32) as u32 % 12 + 117) << 23) | mantissa,
            };
            bits.to_ne_bytes()
        })
        .collect()
}
/// BF16 weights of either sign in [0.25, 4), with one non-finite weight
/// for the weight patterns.
pub(super) fn weight_of(width: usize, pattern: &str, mut state: u64) -> Vec<u8> {
    (0..width)
        .flat_map(|i| {
            let state = next(&mut state);
            let bits = match pattern {
                "weight_nan" if i == width - 1 => 0x7fc1,
                "weight_inf" if i == 0 => 0xff80,
                _ => {
                    ((state >> 16) as u16 & 0x8000)
                        | ((((state >> 32) % 4) as u16 + 125) << 7)
                        | (state as u16 & 0x7f)
                }
            };
            bits.to_ne_bytes()
        })
        .collect()
}
