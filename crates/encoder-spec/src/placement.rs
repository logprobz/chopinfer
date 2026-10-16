//! Placement: where one item's encoder rows and the layout's learned
//! delimiter rows go in the language model's sequence, and the positions
//! they take. A pure function of the output layout and the item's rows
//! after the last stage.

use crate::components::layout::PositionRule;
use crate::components::{Item, Layout, mul};
use crate::geometry::Geometry;
use crate::{Declaration, Error};

/// A learned row of `layout.learned-delimiters.v1`, by its weight role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Delimiter {
    Start,
    Newline,
    End,
}

/// One consecutive piece of an item's span in the sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Piece {
    /// `rows` of the item's encoder rows from `media_row`.
    Rows { media_row: u64, rows: u64 },
    /// One learned row.
    Learned(Delimiter),
}

impl Piece {
    pub fn rows(self) -> u64 {
        match self {
            Piece::Rows { rows, .. } => rows,
            Piece::Learned(_) => 1,
        }
    }
}

/// An item's span: its pieces in sequence order, the position of each of
/// its rows, and the running position after it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Placement {
    /// For `layout.tokens.v1` one piece per temporal grid of a grid item
    /// (the frontend may write text between them, as video timestamps), or
    /// one piece; for `layout.learned-delimiters.v1` the delimited grid.
    pub pieces: Vec<Piece>,
    /// `(t, h, w)` of every row of the pieces, laid end to end from the
    /// running position `start`. Under `sequential` every axis is the row's
    /// running position.
    pub positions: Vec<[u64; 3]>,
    /// The running position of the row after the span.
    pub next: u64,
    /// Whether every row of the span carries the media mask.
    pub media_mask: bool,
}

impl Layout {
    /// The span of an item whose encoder rows are `encoded` (rows and the
    /// merged grid, if any), starting at running position `start`.
    ///
    /// `grid-3axis.v1` gives row `k` of a `[t, h, w]` grid
    /// `(s + k / (h w), s + (k % (h w)) / w, s + k % w)` and advances the
    /// running position by `max(t, h, w)`, one past the span's largest
    /// position, as the engine's `PositionAxesV1` does
    /// (runtime-engine/src/positions.rs).
    pub fn place(&self, encoded: Item, start: u64) -> Result<Placement, String> {
        let params = self.params();
        let mut pieces = Vec::new();
        match self {
            Layout::Tokens(_) => match encoded.grid {
                Some([t, h, w]) if t > 1 => {
                    let rows = mul(h, w)?;
                    for index in 0..t {
                        pieces.push(Piece::Rows {
                            media_row: index * rows,
                            rows,
                        });
                    }
                }
                _ => pieces.push(Piece::Rows {
                    media_row: 0,
                    rows: encoded.rows,
                }),
            },
            Layout::LearnedDelimiters(_) => {
                let [t, h, w] = encoded.grid.ok_or("learned delimiters need a grid")?;
                for index in 0..t {
                    pieces.push(Piece::Learned(Delimiter::Start));
                    for row in 0..h {
                        pieces.push(Piece::Rows {
                            media_row: (index * h + row) * w,
                            rows: w,
                        });
                        pieces.push(Piece::Learned(Delimiter::Newline));
                    }
                    pieces.push(Piece::Learned(Delimiter::End));
                }
            }
        }
        let rows: u64 = pieces.iter().map(|piece| piece.rows()).sum();
        let (positions, next) = match params.positions {
            PositionRule::Sequential => (
                (start..start + rows).map(|p| [p; 3]).collect(),
                start + rows,
            ),
            PositionRule::Grid3Axis => {
                let [t, h, w] = encoded.grid.ok_or("grid-3axis.v1 needs a grid")?;
                if !matches!(self, Layout::Tokens(_)) {
                    return Err("grid-3axis.v1 places grid rows only".into());
                }
                let plane = mul(h, w)?;
                (
                    (0..rows)
                        .map(|k| [start + k / plane, start + k % plane / w, start + k % w])
                        .collect(),
                    start + t.max(h).max(w),
                )
            }
        };
        Ok(Placement {
            pieces,
            positions,
            next,
            media_mask: params.media_mask,
        })
    }
}

impl Declaration {
    /// The span of the item `geometry` describes, from running position
    /// `start`.
    pub fn place(&self, geometry: &Geometry, start: u64) -> Result<Placement, Error> {
        let encoded = geometry.rows[geometry.rows.len() - 2];
        let placement = self
            .output
            .op
            .place(encoded, start)
            .map_err(|error| Error(format!("output: {error}")))?;
        let traced = geometry.rows[geometry.rows.len() - 1].rows;
        if placement.positions.len() as u64 != traced {
            return Err(Error(format!(
                "output: the span has {} rows, the trace {traced}",
                placement.positions.len()
            )));
        }
        Ok(placement)
    }
}
