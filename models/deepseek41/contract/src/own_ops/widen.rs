//! The row widening before the embedding's all-reduce: each rank's BF16
//! embedding rows of its vocabulary shard (zero for the other shards' ids)
//! widened exactly to F32, the partials the rounded all-reduce sums and
//! rounds once. One rank holds each id's row, so the sum is that row and
//! its rounding gives back its BF16 bits, as the vendor's all-reduce of
//! the masked rows does (`ParallelEmbedding`, `inference/model.py:168-178`).
//! Two values do not keep their bits: the sum starts from +0.0, so a
//! negative zero comes back as +0.0 (as in the vendor's all-reduce), and a
//! NaN as the canonical BF16 NaN. The Flash checkpoint's embedding holds
//! neither (nor any subnormal).

use kernel_core::backend::Need;
use kernel_core::contract::Refusal;

/// `output[r] = f32(input[r])` over rows of `width` BF16 values, exact for
/// every number; a NaN widens to a NaN of unspecified sign and payload.
/// All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WidenRows {
    pub width: u32,
}

impl WidenRows {
    /// Refuses an empty row or a capacity of no rows.
    pub fn validate(&self, need: Need) -> Result<(), Refusal> {
        if self.width == 0 {
            return Err(Refusal("a row widening needs values"));
        }
        if need.capacity == 0 {
            return Err(Refusal("a row widening needs rows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A row widening's buffers: the BF16 input rows, the F32 output rows
    /// and the device `u32` count of live rows, whose rows alone are
    /// written.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WidenRowsBuffers {
        pub input: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// The widened rows of `input`'s first `live_rows` rows, as F32 values.
pub fn oracle(op: &WidenRows, input: &[u16], live_rows: u32) -> Result<Vec<f32>, Refusal> {
    let count = live_rows as usize * op.width as usize;
    let input = input
        .get(..count)
        .ok_or(Refusal("a widening's input is too small"))?;
    Ok(input
        .iter()
        .map(|&bits| f32::from_bits(u32::from(bits) << 16))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_keeps_every_number() {
        let op = WidenRows { width: 4 };
        // 1.0, -2.0, the smallest subnormal and -0.0.
        let input = [0x3f80, 0xc000, 0x0001, 0x8000, 0x7f80, 0, 0, 0];
        let wide = oracle(&op, &input, 1).unwrap();
        assert_eq!(wide[..2], [1.0, -2.0]);
        assert_eq!(wide[2].to_bits(), 0x0001_0000);
        assert_eq!(wide[3].to_bits(), 0x8000_0000);
        assert_eq!(oracle(&op, &input, 2).unwrap()[4], f32::INFINITY);
        assert!(oracle(&op, &input, 3).is_err());
        assert!(WidenRows { width: 0 }.validate(Need::rows(1)).is_err());
        assert!(op.validate(Need::rows(0)).is_err());
    }

    /// Every BF16 value but the negative zero and the NaNs, widened on the
    /// rank that holds its row with zeros on the others, comes back with
    /// its bits from the rounded all-reduce at every rank count.
    #[test]
    fn the_reduced_embedding_keeps_the_shard_row_bits() {
        let values: Vec<u16> = (0..=u16::MAX)
            .filter(|&bits| bits != 0x8000 && !f32::from_bits(u32::from(bits) << 16).is_nan())
            .collect();
        let op = WidenRows {
            width: values.len() as u32,
        };
        let wide = oracle(&op, &values, 1).unwrap();
        for ranks in [1, 2, 4] {
            for holder in 0..ranks {
                let partials: Vec<Vec<f32>> = (0..ranks)
                    .map(|rank| match rank == holder {
                        true => wide.clone(),
                        false => vec![0.0; wide.len()],
                    })
                    .collect();
                let reduced = kernel_core::oracle::all_reduce_rounded(&partials).unwrap();
                assert_eq!(reduced, values, "{ranks} ranks, row on rank {holder}");
            }
        }
        let negative_zero = oracle(&WidenRows { width: 1 }, &[0x8000], 1).unwrap();
        let reduced = kernel_core::oracle::all_reduce_rounded(&[negative_zero, vec![0.0]]).unwrap();
        assert_eq!(reduced, [0x0000]);
    }
}
