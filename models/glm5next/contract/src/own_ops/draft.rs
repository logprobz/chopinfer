//! The MTP drafter's own row (`bundle/cuda/row/glm5next.cuh`): the gather
//! of its row inputs by source codes. Its argmax runs on the shared sharded
//! argmax (`crate::draft_argmax`).

use kernel_core::contract::{Refusal, Validate};

use super::{entry, live, take_mut};
use crate::contract::Shape;

/// The drafter's row inputs: hidden row `r` copies entry `-1 - code` of
/// the retained rows when its hidden code is negative, else row `code` of
/// the rows the previous program left (`last`); its token becomes
/// `drafts[code]` when its token code is at least zero and is kept
/// otherwise. `codes` holds the (hidden, token) pair per row, `width` BF16
/// values per hidden row. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftGather {
    pub width: u32,
}

impl DraftGather {
    pub fn of(shape: Shape) -> DraftGather {
        DraftGather {
            width: shape.profile.hidden,
        }
    }
}

impl Validate for DraftGather {
    fn validate(&self) -> Result<(), Refusal> {
        if self.width == 0 {
            return Err(Refusal("a draft gather needs row values"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A draft gather's buffers: the retained rows, the previous program's
    /// rows, the `i32` code pairs, the `u32` drafts, the device `u32` count
    /// of live rows, the output rows and the `u32` tokens. Rows are moved
    /// in 16-byte vectors and start at 16-byte alignment; the codes, drafts,
    /// count and tokens at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct DraftGatherBuffers {
        pub retained: value,
        pub last: value,
        pub codes: value,
        pub drafts: value,
        pub live_rows: value,
        pub output: value,
        pub tokens: value,
    }
}

/// What the gather kernel writes over its first `rows` grid rows. A code
/// naming a row or draft outside its buffer is refused.
#[allow(clippy::too_many_arguments)]
pub fn draft_gather(
    op: &DraftGather,
    retained: &[u16],
    last: &[u16],
    codes: &[i32],
    drafts: &[u32],
    live_rows: u32,
    output: &mut [u16],
    tokens: &mut [u32],
    rows: u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let width = op.width as usize;
    let live = live(rows, live_rows);
    let output = take_mut(output, live * width, "a draft gather's output is too small")?;
    for (row, out) in output.chunks_exact_mut(width).enumerate() {
        let hidden = entry(codes, 2 * row, "a draft gather's codes are too few")?;
        let token = entry(codes, 2 * row + 1, "a draft gather's codes are too few")?;
        let (from, at) = if hidden < 0 {
            (retained, (-1 - i64::from(hidden)) as usize)
        } else {
            (last, hidden as usize)
        };
        out.copy_from_slice(
            from.get(at * width..(at + 1) * width)
                .ok_or(Refusal("a hidden code names a row past its buffer"))?,
        );
        if token >= 0 {
            let draft = entry(drafts, token as usize, "a token code names no draft")?;
            *tokens
                .get_mut(row)
                .ok_or(Refusal("a draft gather's tokens are too few"))? = draft;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const POISON: u16 = 0xaaaa;

    #[test]
    fn gathers_copy_retained_or_last_rows_and_draft_tokens() {
        let op = DraftGather { width: 2 };
        let retained = [10, 11, 12, 13];
        let last = [20, 21, 22, 23, 24, 25];
        let mut output = [POISON; 8];
        let mut tokens = [1, 2, 3, 4];
        draft_gather(
            &op,
            &retained,
            &last,
            &[-2, -1, 2, 1, 0, -1, 9, 9],
            &[70, 71],
            3,
            &mut output,
            &mut tokens,
            4,
        )
        .unwrap();
        assert_eq!(output, [12, 13, 24, 25, 20, 21, POISON, POISON]);
        assert_eq!(tokens, [1, 71, 3, 4]);
        // A code past its buffer.
        let mut output = [0; 2];
        assert!(
            draft_gather(
                &op,
                &retained,
                &last,
                &[-3, -1],
                &[],
                1,
                &mut output,
                &mut [0],
                1
            )
            .is_err()
        );
        assert!(
            draft_gather(
                &op,
                &retained,
                &last,
                &[0, 2],
                &[7, 8],
                1,
                &mut output,
                &mut [0],
                1
            )
            .is_err()
        );
    }
}
