//! The audio encoder's own kernels (`cuda/mimo_encoder.cuh` in the bundle):
//! the residual quantizer and the 20-table embedding sum of MiMo's
//! `f32-pairs.v1` audio tokenizer, a discrete tokenizer and speech tables no
//! other model has. Their stages are registered variants
//! (`crate::encoders::spec`), so the shared encoder runtime launches them as
//! model-owned ops.
//!
//! The oracles compute what the kernels compute, bit for bit, except the
//! payload of a NaN the embedding sum writes (see `embedding_sum`). The F64
//! references `kernel_core::multimodal_oracle::{rvq_nearest,
//! multi_embedding_sum}` stay the accuracy references.

use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Refusal, Validate};

use super::bytes_at;
use crate::encoders::spec::{Rvq20Params, Sum20Params};

/// Residual vector quantization of F32 rows over stacked F32 codebooks
/// (`mimo26:quantize.rvq20-f32.v1`). Stage `s` searches the codebook rows
/// `offsets[s] .. offsets[s + 1]` (at most `max_codes` of them) for the row
/// nearest the remaining residual: the least F32 sum of squared differences,
/// fused in column order, ties to the lower code, a NaN or infinite distance
/// never chosen (code 0 then). The chosen row is subtracted in F32 and its
/// code written. All math: the stage sizes are device data in the offsets
/// table, which the host keeps within `codebook_rows`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rvq {
    /// The row width.
    pub dim: u32,
    /// Codes per row.
    pub stages: u32,
    /// The largest stage codebook.
    pub max_codes: u32,
    /// Rows of the stacked codebooks.
    pub codebook_rows: u32,
}

impl Rvq {
    /// The quantizer a declaration's stage runs over rows of `dim`.
    pub fn of(params: &Rvq20Params, dim: u32) -> Rvq {
        Rvq {
            dim,
            stages: params.codebooks.len() as u32,
            max_codes: params.codebooks.iter().copied().max().unwrap_or(0),
            codebook_rows: params.codebooks.iter().sum(),
        }
    }
}

impl Validate for Rvq {
    fn validate(&self) -> Result<(), Refusal> {
        if self.dim == 0 || self.stages == 0 || self.max_codes == 0 {
            return Err(Refusal("a quantizer needs a width, stages and codes"));
        }
        if self.max_codes > self.codebook_rows {
            return Err(Refusal(
                "a quantizer stage holds more codes than its codebooks",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A quantizer's buffers: the F32 `input` rows `[T, dim]`, the F32
    /// `codebooks` `[codebook_rows, dim]`, the i32 `codebook_offsets`
    /// (`stages + 1` entries, each stage's first row then the end), the i32
    /// `codes` `[T, stages]` it writes and the device count of live rows.
    /// Every view starts at 4-byte alignment.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RvqBuffers {
        pub input: value,
        pub codebooks: value,
        pub codebook_offsets: value,
        pub codes: value,
        pub live_rows: value,
    }
}

/// Sums each row's 20 codes' BF16 table rows in F32
/// (`mimo26:embed.sum20-f32.v1`): column `c` of row `r` is the F32 sum, in
/// table order, of `tables[t][codes[r][t]][c]` over the tables whose code is
/// within `[0, vocab)`; other codes add nothing. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingSum {
    pub tables: u32,
    /// Rows of each table.
    pub vocab: u32,
    pub width: u32,
}

impl EmbeddingSum {
    /// The tables of the 20-table sum at `params`.
    pub fn of(params: &Sum20Params) -> EmbeddingSum {
        EmbeddingSum {
            tables: 20,
            vocab: params.vocab,
            width: params.width,
        }
    }
}

impl Validate for EmbeddingSum {
    fn validate(&self) -> Result<(), Refusal> {
        if self.tables == 0 || self.vocab == 0 || self.width == 0 {
            return Err(Refusal("an embedding sum needs tables, rows and a width"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An embedding sum's buffers: the stacked BF16 `tables` `[tables,
    /// vocab, width]`, the i32 `codes` `[T, tables]`, the F32 `output` rows
    /// `[T, width]` and the device count of live rows. Every view starts at
    /// 4-byte alignment.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EmbeddingSumBuffers {
        pub tables: value,
        pub codes: value,
        pub output: value,
        pub live_rows: value,
    }
}

fn f32_at(buffer: &[u8], index: u64) -> Result<f32, Refusal> {
    let bytes = bytes_at(buffer, index * 4, 4)?;
    Ok(f32::from_le_bytes(bytes.try_into().expect("four bytes")))
}

fn i32_at(buffer: &[u8], index: u64) -> Result<i32, Refusal> {
    let bytes = bytes_at(buffer, index * 4, 4)?;
    Ok(i32::from_le_bytes(bytes.try_into().expect("four bytes")))
}

fn put(buffer: &mut [u8], index: u64, bytes: [u8; 4]) -> Result<(), Refusal> {
    let at = usize::try_from(index * 4).map_err(|_| Refusal("an oracle offset overflows"))?;
    buffer
        .get_mut(at..at + 4)
        .ok_or(Refusal("an oracle buffer is too small"))?
        .copy_from_slice(&bytes);
    Ok(())
}

/// What the quantizer kernel computes over its first `rows` rows.
pub fn rvq(
    op: &Rvq,
    input: &[u8],
    codebooks: &[u8],
    offsets: &[u8],
    codes: &mut [u8],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let dim = u64::from(op.dim);
    for row in 0..u64::from(rows.min(live_rows)) {
        let mut residual = (0..dim)
            .map(|col| f32_at(input, row * dim + col))
            .collect::<Result<Vec<_>, _>>()?;
        for stage in 0..u64::from(op.stages) {
            let start = i32_at(offsets, stage)?;
            let count = i32_at(offsets, stage + 1)? - start;
            let entry = |code: i64, col: u64| {
                let row = u64::try_from(i64::from(start) + code)
                    .map_err(|_| Refusal("a codebook offset is negative"))?;
                f32_at(codebooks, row * dim + col)
            };
            let (mut best, mut best_code) = (f32::INFINITY, 0_i64);
            for code in 0..i64::from(count).min(i64::from(op.max_codes)) {
                let mut distance = 0_f32;
                for col in 0..dim {
                    let delta = residual[col as usize] - entry(code, col)?;
                    distance = delta.mul_add(delta, distance);
                }
                if distance < best {
                    (best, best_code) = (distance, code);
                }
            }
            put(
                codes,
                row * u64::from(op.stages) + stage,
                (best_code as i32).to_le_bytes(),
            )?;
            for (col, value) in residual.iter_mut().enumerate() {
                *value -= entry(best_code, col as u64)?;
            }
        }
    }
    Ok(())
}

/// What the embedding sum kernel computes over its first `rows` rows. Bit
/// for bit where a sum is not NaN. A sum that adds a NaN table value, or
/// infinities of both signs, is NaN on both sides, but its payload bits are
/// the oracle's own: PTX leaves the payload of an F32 NaN result
/// unspecified, so a comparison takes any NaN for any NaN there.
pub fn embedding_sum(
    op: &EmbeddingSum,
    tables: &[u8],
    codes: &[u8],
    output: &mut [u8],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let (vocab, width) = (u64::from(op.vocab), u64::from(op.width));
    for row in 0..u64::from(rows.min(live_rows)) {
        for col in 0..width {
            let mut sum = 0_f32;
            for table in 0..u64::from(op.tables) {
                let code = i32_at(codes, row * u64::from(op.tables) + table)?;
                if let Ok(code) = u64::try_from(code)
                    && code < vocab
                {
                    let at = ((table * vocab + code) * width + col) * 2;
                    let bits = bytes_at(tables, at, 2)?;
                    sum += bf16_to_f32(u16::from_le_bytes([bits[0], bits[1]]));
                }
            }
            put(output, row * width + col, sum.to_le_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use kernel_core::multimodal_oracle::{multi_embedding_sum, rvq_nearest};

    use super::*;

    fn le<T: Copy, const N: usize>(values: &[T], to: impl Fn(T) -> [u8; N]) -> Vec<u8> {
        values.iter().flat_map(|&value| to(value)).collect()
    }

    fn codes(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    /// Two stages of 2 and 3 codes over rows of 2: well separated
    /// distances pick the F64 reference's codes; rows past the live count
    /// stay unwritten.
    #[test]
    fn the_quantizer_picks_the_references_codes() {
        let op = Rvq {
            dim: 2,
            stages: 2,
            max_codes: 3,
            codebook_rows: 5,
        };
        let books = [0.0, 0.0, 4.0, 4.0, 0.5, 0.0, -1.0, 0.0, 0.0, 1.0_f32];
        let input = [3.5, 4.0, 0.2, 0.9, 9.0, 9.0_f32];
        let mut out = vec![0xaa; 3 * 2 * 4];
        rvq(
            &op,
            &le(&input, f32::to_le_bytes),
            &le(&books, f32::to_le_bytes),
            &le(&[0, 2, 5_i32], i32::to_le_bytes),
            &mut out,
            2,
            3,
        )
        .unwrap();
        let wide = |values: &[f32]| values.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let books = wide(&books);
        let expected = rvq_nearest(
            &wide(&input[..4]),
            2,
            2,
            &[&books[..4], &books[4..]],
            &[2, 3],
        )
        .unwrap();
        let written = codes(&out[..16]);
        assert_eq!(
            written,
            expected
                .codes
                .iter()
                .map(|&code| code as i32)
                .collect::<Vec<_>>()
        );
        assert_eq!(written, [1, 1, 0, 2]);
        assert_eq!(&out[16..], [0xaa; 8]);
    }

    /// Equal distances pick the lower code; NaN distances never win, so a
    /// NaN row writes code 0 at every stage; codes past `max_codes` and
    /// past a stage's own size are not searched.
    #[test]
    fn the_quantizer_breaks_ties_low_and_skips_nan_and_capped_codes() {
        let op = Rvq {
            dim: 1,
            stages: 1,
            max_codes: 2,
            codebook_rows: 3,
        };
        let run = |input: f32, books: [f32; 3], offsets: [i32; 2]| {
            let mut out = vec![0; 4];
            rvq(
                &op,
                &input.to_le_bytes(),
                &le(&books, f32::to_le_bytes),
                &le(&offsets, i32::to_le_bytes),
                &mut out,
                1,
                1,
            )
            .unwrap();
            codes(&out)[0]
        };
        assert_eq!(run(1.0, [0.0, 2.0, 1.0], [0, 3]), 0);
        assert_eq!(run(f32::NAN, [0.0, 2.0, 1.0], [0, 3]), 0);
        // The third code is nearest but past `max_codes`.
        assert_eq!(run(1.9, [0.0, 2.5, 1.9], [0, 3]), 1);
        // A stage of one code searches only it.
        assert_eq!(run(2.0, [0.0, 2.0, 1.0], [0, 1]), 0);
    }

    /// The sum in table order in F32, codes outside the tables adding
    /// nothing, and the F64 reference's value where F32 rounds nothing.
    #[test]
    fn the_embedding_sum_adds_in_table_order_and_skips_outside_codes() {
        let op = EmbeddingSum {
            tables: 3,
            vocab: 2,
            width: 1,
        };
        let bf16 = |x: f32| kernel_core::codec::f32_to_bf16(x).to_le_bytes();
        let values = [1.0, 2.0, 256.0, 0.5, -256.0, 4.0_f32];
        let tables = le(&values, bf16);
        let mut out = vec![0; 2 * 4];
        let rows = [0, 1, 0, 1, 2, -1_i32];
        embedding_sum(&op, &tables, &le(&rows, i32::to_le_bytes), &mut out, 2, 2).unwrap();
        let sums = out
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect::<Vec<_>>();
        // Row 0: (1 + 0.5) - 256; row 1: 2 alone (codes 2 and -1 skipped).
        assert_eq!(sums, [-254.5, 2.0]);
        let decoded = values.map(f64::from);
        let reference = multi_embedding_sum(
            &[&decoded[..2], &decoded[2..4], &decoded[4..]],
            &[0, 1, 0],
            2,
            1,
        )
        .unwrap();
        assert_eq!(f64::from(sums[0]), reference.values[0]);
    }

    /// A selected NaN table value (one with a payload) or infinities of
    /// both signs make the column NaN; a NaN in a row no code selects, or
    /// behind a code outside the tables, changes nothing.
    #[test]
    fn the_embedding_sum_is_nan_where_it_adds_a_nan_or_opposite_infinities() {
        let op = EmbeddingSum {
            tables: 2,
            vocab: 2,
            width: 2,
        };
        let (nan, inf, minus_inf) = (0x7fc1_u16, 0x7f80_u16, 0xff80_u16);
        let one = kernel_core::codec::f32_to_bf16(1.0);
        // Table 0 rows [nan, inf], [1, 1]; table 1 rows [1, -inf], [nan, 1].
        let tables = le(
            &[nan, inf, one, one, one, minus_inf, nan, one],
            u16::to_le_bytes,
        );
        let sums = |codes: [i32; 2]| {
            let mut out = vec![0; 2 * 4];
            embedding_sum(&op, &tables, &le(&codes, i32::to_le_bytes), &mut out, 1, 1).unwrap();
            out.chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let [first, second] = sums([0, 0])[..] else {
            unreachable!()
        };
        assert!(first.is_nan());
        assert!(second.is_nan(), "inf + -inf");
        assert_eq!(sums([1, 0]), [2.0, f32::NEG_INFINITY]);
        // Table 1's NaN row sits behind a code past the tables' rows.
        assert_eq!(sums([1, 2]), [1.0, 1.0]);
        assert_eq!(sums([1, -1]), [1.0, 1.0]);
    }

    #[test]
    fn contracts_follow_the_declarations_variants() {
        let rvq = Rvq::of(
            &Rvq20Params {
                codebooks: vec![1024, 256, 128],
            },
            1024,
        );
        assert_eq!(
            rvq,
            Rvq {
                dim: 1024,
                stages: 3,
                max_codes: 1024,
                codebook_rows: 1408,
            }
        );
        assert!(rvq.validate().is_ok());
        assert!(Rvq { dim: 0, ..rvq }.validate().is_err());
        assert!(
            Rvq {
                codebook_rows: 1000,
                ..rvq
            }
            .validate()
            .is_err()
        );
        let sum = EmbeddingSum::of(&Sum20Params {
            vocab: 1280,
            width: 1024,
        });
        assert_eq!((sum.tables, sum.vocab, sum.width), (20, 1280, 1024));
        assert!(EmbeddingSum { vocab: 0, ..sum }.validate().is_err());
    }
}
