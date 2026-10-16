//! Integer 8x8 IDCT with libjpeg-turbo's `islow` scale and rounding.

const FIX_0_298631336: i64 = 2446;
const FIX_0_390180644: i64 = 3196;
const FIX_0_541196100: i64 = 4433;
const FIX_0_765366865: i64 = 6270;
const FIX_0_899976223: i64 = 7373;
const FIX_1_175875602: i64 = 9633;
const FIX_1_501321110: i64 = 12299;
const FIX_1_847759065: i64 = 15137;
const FIX_1_961570560: i64 = 16069;
const FIX_2_053119869: i64 = 16819;
const FIX_2_562915447: i64 = 20995;
const FIX_3_072711026: i64 = 25172;

fn descale(value: i64, shift: u32) -> i64 {
    (value + (1_i64 << (shift - 1))) >> shift
}

fn butterfly(input: [i64; 8], shift: u32) -> [i64; 8] {
    let z2 = input[2];
    let z3 = input[6];
    let z1 = (z2 + z3) * FIX_0_541196100;
    let t2 = z1 - z3 * FIX_1_847759065;
    let t3 = z1 + z2 * FIX_0_765366865;
    let t0 = (input[0] + input[4]) << 13;
    let t1 = (input[0] - input[4]) << 13;
    let e0 = t0 + t3;
    let e3 = t0 - t3;
    let e1 = t1 + t2;
    let e2 = t1 - t2;

    let mut o0 = input[7];
    let mut o1 = input[5];
    let mut o2 = input[3];
    let mut o3 = input[1];
    let z1 = o0 + o3;
    let z2 = o1 + o2;
    let z3 = o0 + o2;
    let z4 = o1 + o3;
    let z5 = (z3 + z4) * FIX_1_175875602;
    o0 *= FIX_0_298631336;
    o1 *= FIX_2_053119869;
    o2 *= FIX_3_072711026;
    o3 *= FIX_1_501321110;
    let z1 = -z1 * FIX_0_899976223;
    let z2 = -z2 * FIX_2_562915447;
    let z3 = -z3 * FIX_1_961570560 + z5;
    let z4 = -z4 * FIX_0_390180644 + z5;
    o0 += z1 + z3;
    o1 += z2 + z4;
    o2 += z2 + z3;
    o3 += z1 + z4;
    [
        descale(e0 + o3, shift),
        descale(e1 + o2, shift),
        descale(e2 + o1, shift),
        descale(e3 + o0, shift),
        descale(e3 - o0, shift),
        descale(e2 - o1, shift),
        descale(e1 - o2, shift),
        descale(e0 - o3, shift),
    ]
}

pub(super) fn block(coefficients: &[i32; 64], quant: &[u16; 64]) -> [u8; 64] {
    let mut work = [0_i64; 64];
    for col in 0..8 {
        if (1..8).all(|row| coefficients[row * 8 + col] == 0) {
            let dc = (i64::from(coefficients[col]) * i64::from(quant[col])) << 2;
            for row in 0..8 {
                work[row * 8 + col] = dc;
            }
        } else {
            let input = std::array::from_fn(|row| {
                i64::from(coefficients[row * 8 + col]) * i64::from(quant[row * 8 + col])
            });
            let output = butterfly(input, 11);
            for row in 0..8 {
                work[row * 8 + col] = output[row];
            }
        }
    }
    let mut pixels = [0_u8; 64];
    for row in 0..8 {
        let input: [i64; 8] = work[row * 8..row * 8 + 8].try_into().expect("IDCT row");
        let output = if input[1..].iter().all(|&value| value == 0) {
            [descale(input[0], 5); 8]
        } else {
            butterfly(input, 18)
        };
        for col in 0..8 {
            pixels[row * 8 + col] = (output[col] + 128).clamp(0, 255) as u8;
        }
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_block_uses_islow_rounding_and_level_shift() {
        let mut coefficients = [0_i32; 64];
        let quant = [1_u16; 64];
        assert_eq!(block(&coefficients, &quant), [128; 64]);
        coefficients[0] = 8;
        assert_eq!(block(&coefficients, &quant), [129; 64]);
        coefficients[0] = -8;
        assert_eq!(block(&coefficients, &quant), [127; 64]);
    }
}
