//! GLM's own row kernels around the all-reduce and the head
//! (`bundle/cuda/row/glm5next.cuh`): the single rounding of an all-reduced
//! F32 row, the widening of a BF16 row into an F32 partial and the
//! drafter's residual add; and the placement of the drafter's two
//! normalized rows as its input projection's input (two exact layout copies
//! on SM120). Each writes the rows below its device live count and nothing
//! else. The router's score compaction, the embedding's copy into every
//! residual stream and the head rows' gather by index are the shared row
//! copies (`crate::copies`).

use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::{Refusal, Validate};

use super::{live, round_bf16, take, take_mut};
use crate::contract::Shape;

/// `output[r] = bf16(input[r])` over rows of `width` F32 values: one
/// rounding to nearest, ties to even; a NaN gives a NaN of unspecified
/// sign and payload ([`super::round_bf16`], compared by
/// [`super::same_bf16`]). All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoundRows {
    pub width: u32,
}

impl RoundRows {
    /// The hidden rows of `shape`, which every all-reduce sums.
    pub fn of(shape: Shape) -> RoundRows {
        RoundRows {
            width: shape.profile.hidden,
        }
    }
}

impl Validate for RoundRows {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 {
            return Err(Refusal("a row rounding needs values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A row rounding's buffers: the F32 input rows, the BF16 output rows
    /// and the device `u32` count of live rows. The input is read in
    /// 16-byte vectors and starts at 16-byte alignment, the output at 8,
    /// the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct RoundRowsBuffers {
        pub input: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// What the rounding kernel writes over its first `rows` grid rows.
pub fn round_rows(
    op: &RoundRows,
    input: &[f32],
    output: &mut [u16],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let count = live(rows, live_rows) * op.width as usize;
    let input = take(input, count, "a rounding's input is too small")?;
    let output = take_mut(output, count, "a rounding's output is too small")?;
    for (out, &value) in output.iter_mut().zip(input) {
        *out = round_bf16(value);
    }
    Ok(())
}

/// `output[r] = f32(input[r])` over rows of `width` BF16 values, exact for
/// every number; a NaN gives a NaN of unspecified sign and payload (the
/// PTX ISA leaves `cvt.f32.bf16`'s NaN results open), compared by
/// [`super::same_f32`]. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WidenRows {
    pub width: u32,
}

impl WidenRows {
    pub fn of(shape: Shape) -> WidenRows {
        WidenRows {
            width: shape.profile.hidden,
        }
    }
}

impl Validate for WidenRows {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 {
            return Err(Refusal("a row widening needs values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A row widening's buffers: the BF16 input rows, the F32 output rows
    /// and the device `u32` count of live rows. The input starts at 8-byte
    /// alignment, the output at 16 (16-byte vector stores), the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct WidenRowsBuffers {
        pub input: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// What the widening kernel writes over its first `rows` grid rows, as F32
/// bits: a number's BF16 bits shifted up, a NaN as the canonical
/// `0x7fffffff`.
pub fn widen_rows(
    op: &WidenRows,
    input: &[u16],
    output: &mut [u32],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let count = live(rows, live_rows) * op.width as usize;
    let input = take(input, count, "a widening's input is too small")?;
    let output = take_mut(output, count, "a widening's output is too small")?;
    for (out, &bits) in output.iter_mut().zip(input) {
        let wide = u32::from(bits) << 16;
        *out = if f32::from_bits(wide).is_nan() {
            0x7fff_ffff
        } else {
            wide
        };
    }
    Ok(())
}

/// The MTP drafter's single residual stream plus a sublayer's output. All
/// math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidualAdd {
    pub width: u32,
}

impl ResidualAdd {
    pub fn of(shape: Shape) -> ResidualAdd {
        ResidualAdd {
            width: shape.profile.hidden,
        }
    }
}

impl Validate for ResidualAdd {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 {
            return Err(Refusal("a residual add needs values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A residual add's buffers: X, read and written, Y and the device
    /// `u32` count of live rows. X and Y are moved in 4-byte pairs and
    /// start at 4-byte alignment, as does the count.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct ResidualAddBuffers {
        pub x: value,
        pub y: value,
        pub live_rows: value,
    }
}

/// What the residual add leaves in `x` over its first `rows` grid rows.
pub fn residual_add(
    op: &ResidualAdd,
    x: &mut [u16],
    y: &[u16],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let count = live(rows, live_rows) * op.width as usize;
    let y = take(y, count, "a residual add's Y is too small")?;
    let x = take_mut(x, count, "a residual add's X is too small")?;
    for (x, &y) in x.iter_mut().zip(y) {
        *x = round_bf16(bf16_to_f32(*x) + bf16_to_f32(y));
    }
    Ok(())
}

/// The MTP drafter's input projection's input: each row of `[rows, 2 *
/// hidden]` BF16 values holds the normalized embedding row in its first
/// `hidden` columns and the normalized hidden row in the next, exact
/// (`cat[enorm(embedding), hnorm(hidden)]`). All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EhInput {
    pub hidden: u32,
}

impl EhInput {
    pub fn of(shape: Shape) -> EhInput {
        EhInput {
            hidden: shape.profile.hidden,
        }
    }
}

impl Validate for EhInput {
    fn validate(&self) -> Result<(), Refusal> {
        if self.hidden == 0 {
            return Err(Refusal("the drafter's input halves need values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// The drafter input placement's buffers: the normalized embedding
    /// rows, the normalized hidden rows, the `[rows, 2 * hidden]` output
    /// and the device `u32` count of live rows. Rows are moved in 16-byte
    /// vectors and start at 16-byte alignment, the count at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EhInputBuffers {
        pub embed: value,
        pub hidden: value,
        pub output: value,
        pub live_rows: value,
    }
}

/// What the two copies write over their first `rows` grid rows.
pub fn eh_input(
    op: &EhInput,
    embed: &[u16],
    hidden: &[u16],
    output: &mut [u16],
    live_rows: u32,
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let width = op.hidden as usize;
    let live = live(rows, live_rows);
    let embed = take(
        embed,
        live * width,
        "the drafter's embedding rows are too few",
    )?;
    let hidden = take(
        hidden,
        live * width,
        "the drafter's hidden rows are too few",
    )?;
    let output = take_mut(
        output,
        live * 2 * width,
        "the drafter's input rows are too few",
    )?;
    for (row, out) in output.chunks_exact_mut(2 * width).enumerate() {
        out[..width].copy_from_slice(&embed[row * width..(row + 1) * width]);
        out[width..].copy_from_slice(&hidden[row * width..(row + 1) * width]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::codec::f32_to_bf16;

    const POISON: u16 = 0xaaaa;

    #[test]
    fn roundings_and_widenings_cover_the_live_rows() {
        let op = RoundRows { width: 2 };
        let input = [1.0, f32::from_bits(0x3f80_8000), 3.0, 4.0, 5.0, 6.0];
        let mut output = [POISON; 6];
        round_rows(&op, &input, &mut output, 2, 3).unwrap();
        assert_eq!(output, [0x3f80, 0x3f80, 0x4040, 0x4080, POISON, POISON]);
        // The live count past the grid keeps to the grid.
        let mut output = [POISON; 6];
        round_rows(&op, &input, &mut output, 9, 1).unwrap();
        assert_eq!(output[2..], [POISON; 4]);
        assert!(round_rows(&op, &input[..3], &mut [0; 6], 2, 2).is_err());
        let op = WidenRows { width: 2 };
        let mut wide = [7; 4];
        widen_rows(&op, &[0x3f80, 0x7f81, 1, 2], &mut wide, 1, 2).unwrap();
        assert_eq!(wide, [0x3f80_0000, 0x7fff_ffff, 7, 7]);
        widen_rows(&op, &[0xff80, 0xffc3, 1, 2], &mut wide, 1, 2).unwrap();
        assert_eq!(wide[..2], [0xff80_0000, 0x7fff_ffff]);
    }

    #[test]
    fn residual_adds_round_once_in_place() {
        let op = ResidualAdd { width: 2 };
        // 1 + 2^-9 rounds back to 1 only once; NaN becomes 0x7fff.
        let small = f32_to_bf16(2_f32.powi(-9));
        let mut x = [0x3f80, 0x7f81, 0x4000, 0x4000];
        residual_add(&op, &mut x, &[small, 0x3f80, 0x3f80, 0x3f80], 1, 2).unwrap();
        assert_eq!(x, [0x3f80, 0x7fff, 0x4000, 0x4000]);
        let mut x = [0x3f80, 0x4000];
        residual_add(&op, &mut x, &[0x3f80, 0x3f80], 1, 1).unwrap();
        assert_eq!(x, [0x4000, 0x4040]);
    }

    #[test]
    fn drafter_inputs_place_both_halves() {
        let op = EhInput { hidden: 2 };
        let mut output = [POISON; 8];
        eh_input(&op, &[1, 2, 3, 4], &[5, 6, 7, 8], &mut output, 1, 2).unwrap();
        assert_eq!(output, [1, 2, 5, 6, POISON, POISON, POISON, POISON]);
    }
}
