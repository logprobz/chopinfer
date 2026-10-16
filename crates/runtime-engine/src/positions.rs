//! Explicit 3-axis RoPE positions for prompts with media grids.
//!
//! The rule is `grid-3axis.v1`. Text rows take (p, p, p) for a running
//! position p. A span over a grid of `t` temporal groups and `h x w` merged
//! blocks starts at the running position `s`: its row `k` takes
//! (s + k / (h w), s + (k % (h w)) / w, s + k % w), and the running position
//! then advances by `max(t, h, w)`, one past the span's largest position, as
//! the reference model code continues at the maximum plus one. Every row
//! after the prompt, generated or drafted, takes (i + delta) on all axes for
//! its sequence index `i`, where delta is the prompt's largest position plus
//! one minus its length. For one temporal group (images) the advance is
//! `max(h, w)`.

/// One media span whose rows take grid positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PositionSpanV1 {
    pub prompt_row: u32,
    pub rows: u32,
    /// Temporal groups, then merged blocks along height and width.
    pub grid: [u32; 3],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionAxesError {
    /// The engine was not configured for explicit positions.
    Unsupported,
    EmptyGrid,
    /// A span's rows are not its grid's product.
    GridRows,
    SpansOutOfOrder,
    SpanOutsidePrompt,
    /// A span does not match one of the request's media spans.
    NotMediaSpan,
    /// A position does not fit the wire's i32.
    TooLarge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlacedSpan {
    span: PositionSpanV1,
    /// The running position at the span's first row.
    start: i64,
}

/// A request's position rule, evaluated for any sequence row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionAxesV1 {
    spans: Vec<PlacedSpan>,
    prompt_tokens: usize,
    delta: i64,
}

impl PositionAxesV1 {
    /// Places `spans` over a prompt whose sequence may reach `maximum_rows`.
    pub fn new(
        prompt_tokens: usize,
        maximum_rows: usize,
        spans: &[PositionSpanV1],
    ) -> Result<Self, PositionAxesError> {
        if i32::try_from(maximum_rows).is_err() || prompt_tokens > maximum_rows {
            return Err(PositionAxesError::TooLarge);
        }
        let mut placed = Vec::with_capacity(spans.len());
        // The running position minus the row index: each span moves it by
        // its advance minus its rows, which is never positive.
        let mut shift = 0_i64;
        let mut previous_end = 0_u64;
        let mut maximum = -1_i64;
        for span in spans {
            let [t, h, w] = span.grid.map(u64::from);
            if t == 0 || h == 0 || w == 0 {
                return Err(PositionAxesError::EmptyGrid);
            }
            if t * h * w != u64::from(span.rows) {
                return Err(PositionAxesError::GridRows);
            }
            let first = u64::from(span.prompt_row);
            let end = first + u64::from(span.rows);
            if first < previous_end {
                return Err(PositionAxesError::SpansOutOfOrder);
            }
            if end > prompt_tokens as u64 {
                return Err(PositionAxesError::SpanOutsidePrompt);
            }
            previous_end = end;
            let start = first as i64 + shift;
            let advance = t.max(h).max(w) as i64;
            maximum = maximum.max(start + advance - 1);
            placed.push(PlacedSpan { span: *span, start });
            shift += advance - i64::from(span.rows);
        }
        if previous_end < prompt_tokens as u64 {
            maximum = maximum.max(prompt_tokens as i64 - 1 + shift);
        }
        Ok(Self {
            spans: placed,
            prompt_tokens,
            delta: maximum + 1 - prompt_tokens as i64,
        })
    }

    /// Extends only grid spans' prefix identity; encoder caching stays independent.
    pub(crate) fn prefix_identity(&self, prompt_row: u32, media: [u8; 32]) -> [u8; 32] {
        use sha2::{Digest, Sha256};

        let Some(placed) = self
            .spans
            .iter()
            .find(|placed| placed.span.prompt_row == prompt_row)
        else {
            return media;
        };
        let mut sha = Sha256::new();
        sha.update(b"chopinfer.prefix.position.grid-3axis.v1");
        sha.update(media);
        for axis in placed.span.grid {
            sha.update(axis.to_le_bytes());
        }
        sha.finalize().into()
    }

    /// The offset every row after the prompt adds to its sequence index.
    #[must_use]
    pub const fn delta(&self) -> i64 {
        self.delta
    }

    /// The (T, H, W) position of sequence row `row`.
    #[must_use]
    pub fn position(&self, row: usize) -> [i32; 3] {
        let row = row as i64;
        if row >= self.prompt_tokens as i64 {
            return narrow([row + self.delta; 3]);
        }
        let after = self
            .spans
            .partition_point(|placed| i64::from(placed.span.prompt_row) <= row);
        let Some(placed) = after.checked_sub(1).map(|index| self.spans[index]) else {
            return narrow([row; 3]);
        };
        let [t, h, w] = placed.span.grid.map(i64::from);
        let k = row - i64::from(placed.span.prompt_row);
        if k < i64::from(placed.span.rows) {
            let s = placed.start;
            return narrow([s + k / (h * w), s + k % (h * w) / w, s + k % w]);
        }
        let end = i64::from(placed.span.prompt_row) + i64::from(placed.span.rows);
        narrow([placed.start + t.max(h).max(w) + (row - end); 3])
    }

    /// Positions of `rows` consecutive sequence rows from `first`.
    #[must_use]
    pub fn rows(&self, first: usize, rows: usize) -> Vec<[i32; 3]> {
        (first..first + rows)
            .map(|row| self.position(row))
            .collect()
    }
}

/// Every position lies in [0, row], so a row inside the bound `new` checked
/// always fits; a row past it saturates rather than wrapping.
fn narrow(position: [i64; 3]) -> [i32; 3] {
    position.map(|axis| i32::try_from(axis.max(0)).unwrap_or(i32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(prompt_row: u32, grid: [u32; 3]) -> PositionSpanV1 {
        PositionSpanV1 {
            prompt_row,
            rows: grid.iter().product(),
            grid,
        }
    }

    /// A direct transcription of the reference rule: walk the prompt, then
    /// continue at the maximum plus one.
    fn reference(prompt: usize, spans: &[PositionSpanV1], total: usize) -> Vec<[i32; 3]> {
        let mut out = Vec::new();
        let mut current = 0_i32;
        let mut row = 0;
        for span in spans {
            while row < span.prompt_row as usize {
                out.push([current; 3]);
                current += 1;
                row += 1;
            }
            let [t, h, w] = span.grid.map(|value| value as i32);
            for ti in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        out.push([current + ti, current + hi, current + wi]);
                    }
                }
            }
            current += t.max(h).max(w);
            row += span.rows as usize;
        }
        while row < prompt {
            out.push([current; 3]);
            current += 1;
            row += 1;
        }
        let next = out.iter().flatten().copied().max().unwrap_or(-1) + 1;
        let delta = next - prompt as i32;
        for row in prompt..total {
            out.push([row as i32 + delta; 3]);
        }
        out
    }

    fn check(prompt: usize, spans: &[PositionSpanV1]) -> PositionAxesV1 {
        let axes = PositionAxesV1::new(prompt, prompt + 6, spans).unwrap();
        assert_eq!(
            axes.rows(0, prompt + 6),
            reference(prompt, spans, prompt + 6)
        );
        axes
    }

    #[test]
    fn text_only_prompts_keep_one_dimensional_positions() {
        let axes = check(5, &[]);
        assert_eq!(axes.delta(), 0);
        assert_eq!(axes.position(7), [7, 7, 7]);
    }

    #[test]
    fn one_image_takes_grid_coordinates_and_text_resumes_after_the_larger_side() {
        // Two text rows, a 2 x 3 grid, then two text rows.
        let axes = check(10, &[span(2, [1, 2, 3])]);
        assert_eq!(axes.rows(0, 2), [[0, 0, 0], [1, 1, 1]]);
        assert_eq!(
            axes.rows(2, 6),
            [
                [2, 2, 2],
                [2, 2, 3],
                [2, 2, 4],
                [2, 3, 2],
                [2, 3, 3],
                [2, 3, 4]
            ]
        );
        assert_eq!(axes.rows(8, 2), [[5, 5, 5], [6, 6, 6]]);
        // Generated rows continue at the maximum plus one.
        assert_eq!(axes.delta(), 7 - 10);
        assert_eq!(axes.position(10), [7, 7, 7]);
        assert_eq!(axes.position(12), [9, 9, 9]);
    }

    #[test]
    fn an_image_at_the_prompt_start_or_end_and_two_images() {
        check(4, &[span(0, [1, 2, 2])]);
        let end = check(7, &[span(3, [1, 2, 2])]);
        assert_eq!(end.position(6), [3, 4, 4]);
        assert_eq!(end.position(7), [5, 5, 5]);
        let two = check(20, &[span(1, [1, 2, 4]), span(11, [1, 3, 2])]);
        assert_eq!(two.position(9), [5, 5, 5]);
        assert_eq!(two.position(11), [7, 7, 7]);
        check(
            20,
            &[span(1, [1, 2, 4]), span(9, [1, 3, 2]), span(15, [2, 1, 2])],
        );
    }

    /// More temporal groups than either side: text after the span resumes
    /// one past its largest position, where generated rows would continue.
    #[test]
    fn a_long_temporal_span_advances_past_its_largest_position() {
        let axes = check(6, &[span(1, [4, 1, 1])]);
        assert_eq!(
            axes.rows(1, 4),
            [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]]
        );
        assert_eq!(axes.rows(5, 1), [[5, 5, 5]]);
        assert_eq!(axes.position(6), [6, 6, 6]);
        let end = check(5, &[span(1, [4, 1, 1])]);
        assert_eq!(end.position(5), [5, 5, 5]);
    }

    /// One temporal group, the only grids an image modality sends: every
    /// position equals the former `max(h, w)` advance.
    #[test]
    fn single_group_spans_keep_their_positions() {
        let former = |prompt: usize, spans: &[PositionSpanV1], total: usize| {
            let mut out = Vec::new();
            let mut current = 0_i32;
            for span in spans {
                while out.len() < span.prompt_row as usize {
                    out.push([current; 3]);
                    current += 1;
                }
                let [_, h, w] = span.grid.map(|value| value as i32);
                for hi in 0..h {
                    for wi in 0..w {
                        out.push([current, current + hi, current + wi]);
                    }
                }
                current += h.max(w);
            }
            while out.len() < prompt {
                out.push([current; 3]);
                current += 1;
            }
            let next = out.iter().flatten().copied().max().unwrap_or(-1) + 1;
            out.extend((prompt..total).map(|row| [row as i32 + next - prompt as i32; 3]));
            out
        };
        for h in 1..=12 {
            for w in 1..=12 {
                let spans = [span(2, [1, h, w]), span(4 + h * w, [1, w, h])];
                let prompt = 7 + 2 * h as usize * w as usize;
                let axes = PositionAxesV1::new(prompt, prompt + 5, &spans).unwrap();
                assert_eq!(axes.rows(0, prompt + 5), former(prompt, &spans, prompt + 5));
            }
        }
    }

    #[test]
    fn chunks_and_verify_rows_see_the_same_positions_as_the_whole() {
        let axes = check(12, &[span(2, [1, 3, 2])]);
        let whole = axes.rows(0, 16);
        // A chunk boundary inside the span, then a verify item over four rows.
        let mut pieces = axes.rows(0, 5);
        pieces.extend(axes.rows(5, 7));
        pieces.extend(axes.rows(12, 4));
        assert_eq!(pieces, whole);
    }

    #[test]
    fn malformed_spans_are_refused() {
        let refused = |spans: &[PositionSpanV1]| PositionAxesV1::new(10, 12, spans).unwrap_err();
        assert_eq!(refused(&[span(0, [1, 0, 2])]), PositionAxesError::EmptyGrid);
        let mut short = span(0, [1, 2, 2]);
        short.rows = 3;
        assert_eq!(refused(&[short]), PositionAxesError::GridRows);
        assert_eq!(
            refused(&[span(4, [1, 2, 2]), span(6, [1, 1, 1])]),
            PositionAxesError::SpansOutOfOrder
        );
        assert_eq!(
            refused(&[span(8, [1, 2, 2])]),
            PositionAxesError::SpanOutsidePrompt
        );
        assert_eq!(
            PositionAxesV1::new(1, usize::MAX, &[]).unwrap_err(),
            PositionAxesError::TooLarge
        );
    }
}
