//! Geometry: how one media item's rows run through a declaration's stages.
//! A pure function of the declaration and the item its recipe emits: the
//! item's rows after every stage (`Declaration::trace`), the encoder runs it
//! is cut into, and for every stage of a run its rows, attention sequences,
//! rotary positions and row reorder.

use std::ops::Range;

use crate::components::rotary::Axes;
use crate::components::{Component, Item, Reorder, StageOp, mul};
use crate::{Chunking, Declaration, Error, Patchify, Recipe, Sequences};

/// One media item as its recipe emits it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Media {
    /// A patch grid `[t, h, w]`: `t` temporal grids of `h x w` patches.
    Grid([u64; 3]),
    /// Rows without a grid, such as log-mel frames.
    Frames(u64),
}

impl Recipe {
    /// The side of the merge blocks the recipe's rows come in (1 for none).
    pub fn merge(&self) -> u32 {
        match self {
            Recipe::ImagePatchGrid(r) => r.merge,
            Recipe::ImageZeroPaddedTokenBudget(r) => r.merge,
            Recipe::VideoTimestampedGrids(r) => r.merge,
            Recipe::AudioLogMel(_) | Recipe::ImageCenteredPad(_) => 1,
        }
    }

    /// The item `media` enters the encoder as, or why the recipe's declared
    /// limits refuse it. This is the encoder's bound, not the processor's
    /// exact output set: a frontend's recipe also has a minimum size and an
    /// aspect limit, which the host checks for prepared media.
    pub fn admit(&self, media: Media) -> Result<Item, String> {
        let tag = self.tag();
        let grid = |merge: u32| {
            let Media::Grid([t, h, w]) = media else {
                return Err(format!("recipe {tag} emits a patch grid, not {media:?}"));
            };
            let merge = u64::from(merge);
            if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(format!(
                    "grid [{t}, {h}, {w}] is empty or not whole {merge} x {merge} merge blocks"
                ));
            }
            Ok([t, h, w])
        };
        let pixels = |h: u64, w: u64, patch: u32, maximum: u64| {
            let pixels = mul(mul(h, w)?, mul(u64::from(patch), u64::from(patch))?)?;
            if pixels > maximum {
                return Err(format!(
                    "{h} x {w} patches hold {pixels} pixels, above recipe {tag}'s {maximum}"
                ));
            }
            Ok(())
        };
        let item = |[t, h, w]: [u64; 3]| {
            Ok(Item {
                rows: mul(mul(t, h)?, w)?,
                grid: Some([t, h, w]),
            })
        };
        match self {
            Recipe::ImagePatchGrid(r) => {
                let [t, h, w] = grid(r.merge)?;
                if t != 1 {
                    return Err(format!("recipe {tag} emits one temporal grid, not {t}"));
                }
                pixels(h, w, r.patch, r.maximum_pixels)?;
                item([t, h, w])
            }
            Recipe::VideoTimestampedGrids(r) => {
                let [t, h, w] = grid(r.merge)?;
                let grids = u64::from(r.maximum_frames).div_ceil(u64::from(r.temporal_patch));
                if t > grids {
                    return Err(format!(
                        "recipe {tag} emits at most {grids} temporal grids, not {t}"
                    ));
                }
                pixels(h, w, r.patch, r.maximum_pixels_per_frame)?;
                item([t, h, w])
            }
            Recipe::AudioLogMel(r) => match media {
                Media::Frames(frames) if (1..=r.maximum_frames).contains(&frames) => Ok(Item {
                    rows: frames,
                    grid: None,
                }),
                _ => Err(format!(
                    "recipe {tag} emits 1 to {} frames, not {media:?}",
                    r.maximum_frames
                )),
            },
            Recipe::ImageZeroPaddedTokenBudget(r) => {
                let [t, h, w] = grid(r.merge)?;
                let merge = u64::from(r.merge);
                if t != 1 || mul(h / merge, w / merge)? > r.maximum_tokens {
                    return Err(format!(
                        "recipe {tag} emits one grid of at most {} merge blocks, not [{t}, {h}, {w}]",
                        r.maximum_tokens
                    ));
                }
                item([t, h, w])
            }
            Recipe::ImageCenteredPad(r) => {
                let [t, h, w] = grid(1)?;
                let cell = u64::from(r.downsample);
                let tokens = mul(h.div_ceil(cell), w.div_ceil(cell) + 1)? + 2;
                if t != 1 || tokens > u64::from(r.maximum_tokens) {
                    return Err(format!(
                        "recipe {tag} emits one grid within {} tokens, not [{t}, {h}, {w}]",
                        r.maximum_tokens
                    ));
                }
                item([t, h, w])
            }
        }
    }
}

impl Recipe {
    /// The item a media record of the batch wire takes, from its geometry:
    /// `[t, h, w, 0]` patches for grid recipes, `[frames, mel_bins, 0, 0]`
    /// for log-mel; or why this recipe's declared limits refuse it.
    pub fn admit_record(&self, geometry: [u32; 4]) -> Result<Item, String> {
        let [a, b, c, rest] = geometry.map(u64::from);
        match self {
            Recipe::AudioLogMel(r) => {
                if b != u64::from(r.mel_bins) || c != 0 || rest != 0 {
                    return Err(format!(
                        "recipe {} records [frames, {}, 0, 0], not {geometry:?}",
                        self.tag(),
                        r.mel_bins
                    ));
                }
                self.admit(Media::Frames(a))
            }
            _ => {
                if rest != 0 {
                    return Err(format!(
                        "recipe {} records [t, h, w, 0], not {geometry:?}",
                        self.tag()
                    ));
                }
                self.admit(Media::Grid([a, b, c]))
            }
        }
    }
}

impl Declaration {
    /// The item a media record of `geometry` takes, when it is within one of
    /// the declaration's recipe bounds (the first that admits it).
    pub fn admit_record(&self, geometry: [u32; 4]) -> Result<Item, Error> {
        let mut refusals = Vec::new();
        for recipe in &self.input.recipes {
            match recipe.admit_record(geometry) {
                Ok(item) => return Ok(item),
                Err(error) => refusals.push(error),
            }
        }
        Err(Error(format!("{}: {}", self.name, refusals.join("; "))))
    }
}

/// The rotary position of each row of a stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Positions {
    /// Each row's (row, column) in its patch grid.
    Grid(Vec<[u64; 2]>),
    /// Each row's index inside its attention sequence.
    Sequence(Vec<u64>),
}

/// One stage of one run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Step {
    pub stage: usize,
    /// The rows the stage reads, then the rows it writes.
    pub rows: [u64; 2],
    /// A transformer's attention sequence bounds (`cu_seqlens`).
    pub sequences: Option<Vec<u64>>,
    /// A transformer's rotary position of each row, in row order.
    pub positions: Option<Positions>,
    /// A transformer's row reorder, destination row to source row: the
    /// blocks its `reorder` names read their rows (and positions) gathered
    /// by it, and scatter them back after.
    pub reorder: Option<Vec<u64>>,
}

/// One encoder run: consecutive stages over a range of the item's rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Run {
    /// The item's rows the run reads at its first stage: payload rows, or
    /// the rows a joining stage writes.
    pub input: Range<u64>,
    /// For a run that starts at a joining stage, the joined row each of its
    /// rows reads.
    pub gather: Option<Vec<u64>>,
    /// The item's rows the run writes at its last stage.
    pub output: Range<u64>,
    pub steps: Vec<Step>,
}

/// One item's geometry under a declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Geometry {
    /// The item entering the first stage.
    pub input: Item,
    /// The item's rows after each stage, then after the output layout: the
    /// declaration's `trace`.
    pub rows: Vec<Item>,
    /// The runs, in order. Runs over the same stages cover the item's rows
    /// in order; a stage that joins segments starts new runs.
    pub runs: Vec<Run>,
}

/// The (row, column) of every patch of one `h x w` grid, in the order
/// `patchify` emits the patches.
pub fn grid_positions(patchify: Patchify, merge: u64, h: u64, w: u64) -> Vec<[u64; 2]> {
    let mut out = Vec::new();
    match patchify {
        Patchify::MergeBlocks => {
            for bh in 0..h / merge {
                for bw in 0..w / merge {
                    for mh in 0..merge {
                        for mw in 0..merge {
                            out.push([bh * merge + mh, bw * merge + mw]);
                        }
                    }
                }
            }
        }
        Patchify::RowMajor => {
            for y in 0..h {
                for x in 0..w {
                    out.push([y, x]);
                }
            }
        }
    }
    out
}

/// The rows `merge.space-to-depth.v1` gathers from a row-major `[t, h, w]`
/// patch grid: per `factor x factor` cell in row-major order (each
/// temporal grid in turn), its patches position-major (`ky * factor +
/// kx`), each the input row it reads or -1 where the cell runs past the
/// grid (a zero row).
pub fn space_to_depth_indices([t, h, w]: [u64; 3], factor: u64) -> Result<Vec<i32>, String> {
    if factor == 0 {
        return Err("a space-to-depth cell of side 0".into());
    }
    let (cells_h, cells_w) = (h.div_ceil(factor), w.div_ceil(factor));
    let mut out = Vec::new();
    for grid in 0..t {
        for cy in 0..cells_h {
            for cx in 0..cells_w {
                for ky in 0..factor {
                    for kx in 0..factor {
                        let (y, x) = (cy * factor + ky, cx * factor + kx);
                        out.push(if y < h && x < w {
                            i32::try_from((grid * h + y) * w + x)
                                .map_err(|_| format!("patch row of [{t}, {h}, {w}] past i32"))?
                        } else {
                            -1
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Bounds of consecutive sequences of `size` rows over `rows` rows.
fn bounds(rows: u64, size: u64) -> Vec<u64> {
    (0..=rows / size).map(|index| index * size).collect()
}

/// What `walk` needs about the run and the declaration around it.
struct Walk<'a> {
    declaration: &'a Declaration,
    components: &'a [&'a dyn Component],
    patchify: Option<Patchify>,
    merge: u64,
    /// Whether the run is one of the declaration's segments.
    segment: bool,
}

impl Walk<'_> {
    /// The steps of `stages` over the run's rows `part`, and its rows after.
    fn steps(&self, stages: Range<usize>, part: Item) -> Result<(Vec<Step>, Item), Error> {
        let capacity = u64::from(self.declaration.input.capacity_rows);
        let mut rows = part;
        let mut steps = Vec::new();
        for stage in stages {
            let component = self.components[stage];
            let fail =
                |error: String| Error(format!("stages[{stage}] ({}): {error}", component.id()));
            let output = component.item(rows).map_err(fail)?;
            if rows.rows > capacity || output.rows > capacity {
                return Err(fail(format!(
                    "a run reaches {} rows, above capacity_rows {capacity}",
                    rows.rows.max(output.rows)
                )));
            }
            let mut step = Step {
                stage,
                rows: [rows.rows, output.rows],
                sequences: None,
                positions: None,
                reorder: None,
            };
            if let StageOp::Transformer(transformer) = &self.declaration.stages[stage].op {
                self.transformer(transformer, rows, part, &mut step)
                    .map_err(fail)?;
            }
            steps.push(step);
            rows = output;
        }
        Ok((steps, rows))
    }

    fn transformer(
        &self,
        transformer: &crate::components::Transformer,
        rows: Item,
        part: Item,
        step: &mut Step,
    ) -> Result<(), String> {
        let params = transformer.params();
        let sequences = match (&params.sequences, &self.declaration.sequences) {
            (Some(Sequences::Groups { rows: size }), _) => {
                let size = u64::from(*size);
                if rows.rows % size != 0 {
                    return Err(format!("{} rows are not whole groups of {size}", rows.rows));
                }
                bounds(rows.rows, size)
            }
            (Some(Sequences::Item {}) | None, Sequences::Item {})
            | (Some(Sequences::Item {}), _) => {
                vec![0, rows.rows]
            }
            (None, Sequences::TemporalGrid {}) => {
                let [t, ..] = rows.grid.ok_or("temporal-grid sequences need grid rows")?;
                bounds(rows.rows, rows.rows / t)
            }
            (None, Sequences::Segments { .. }) if self.segment => vec![0, rows.rows],
            (None, _) => {
                return Err("rows a stage has joined need the stage's own sequences".into());
            }
            (Some(other), _) => {
                return Err(format!(
                    "stage-level sequences `{}` are not allowed here",
                    crate::components::op(other)
                ));
            }
        };
        let grid = || {
            let grid = rows.grid.filter(|grid| part.grid == Some(*grid));
            match (grid, self.patchify) {
                (Some([t, h, w]), Some(patchify)) => {
                    let one = grid_positions(patchify, self.merge, h, w);
                    Ok((t, h, w, one))
                }
                _ => Err("grid positions need the recipe's unmerged patch grid".to_owned()),
            }
        };
        if let Some(slot) = &params.rotary {
            step.positions = Some(match slot.component()?.axes() {
                Axes::Grid2d => {
                    let (t, _, _, one) = grid()?;
                    Positions::Grid((0..t).flat_map(|_| one.iter().copied()).collect())
                }
                Axes::Sequence => Positions::Sequence(
                    sequences
                        .windows(2)
                        .flat_map(|pair| 0..pair[1] - pair[0])
                        .collect(),
                ),
            });
        }
        if let Some(first) = params.reorder.first() {
            if params.reorder.iter().any(|r| r.reorder != first.reorder) {
                return Err("one transformer reorders its rows one way".into());
            }
            let Reorder::MergeGroupColumns(_) = first.reorder;
            let (t, h, w, _) = grid()?;
            let (m, group) = (self.merge, self.merge * self.merge);
            let mut map = Vec::with_capacity(rows.rows as usize);
            for index in 0..t {
                for bw in 0..w / m {
                    for bh in 0..h / m {
                        let block = index * h * w + (bh * (w / m) + bw) * group;
                        map.extend(block..block + group);
                    }
                }
            }
            step.reorder = Some(map);
        }
        step.sequences = Some(sequences);
        Ok(())
    }
}

impl Declaration {
    /// The geometry of one item that recipe `tag` emits as `media`: its rows
    /// after every stage, the runs it is cut into within `capacity_rows`,
    /// and each run's per-stage rows, sequences, positions and reorders.
    ///
    /// Before any stage joins segments, a run is one item (`item`), one
    /// segment (`segments`), or one temporal grid; with `chunking` a run
    /// packs as many whole temporal grids as `capacity_rows` holds. After a
    /// joining stage, runs take the joined rows in whole groups up to
    /// `capacity_rows`.
    pub fn geometry(&self, tag: &str, media: Media) -> Result<Geometry, Error> {
        let recipe = self
            .input
            .recipes
            .iter()
            .find(|recipe| recipe.tag() == tag)
            .ok_or_else(|| Error(format!("{}: no recipe {tag}", self.name)))?;
        let input = recipe.admit(media).map_err(Error)?;
        let rows = self.trace(input)?;
        let components = self
            .stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                stage
                    .op
                    .component()
                    .map_err(|error| Error(format!("stages[{index}]: {error}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let capacity = u64::from(self.input.capacity_rows);
        let mut walk = Walk {
            declaration: self,
            components: &components,
            patchify: self.input.patchify,
            merge: u64::from(recipe.merge()),
            segment: matches!(self.sequences, Sequences::Segments { .. }),
        };
        let joins: Vec<usize> = (0..self.stages.len())
            .filter(|&stage| components[stage].joins_segments())
            .collect();
        let first_end = joins.first().copied().unwrap_or(self.stages.len());

        let parts: Vec<Item> = match self.sequences {
            Sequences::Item {} => vec![input],
            Sequences::TemporalGrid {} => {
                let [t, h, w] = input.grid.ok_or_else(|| Error("no grid".into()))?;
                let per = match self.input.chunking {
                    Some(Chunking::WholeSequences) => capacity / (h * w),
                    None => 1,
                };
                if per == 0 {
                    return Err(Error(format!(
                        "a {h} x {w} grid is above capacity_rows {capacity}"
                    )));
                }
                (0..t)
                    .step_by(per as usize)
                    .map(|first| {
                        let grids = per.min(t - first);
                        Item {
                            rows: grids * h * w,
                            grid: Some([grids, h, w]),
                        }
                    })
                    .collect()
            }
            Sequences::Segments { input_rows } => {
                let size = u64::from(input_rows);
                (0..input.rows)
                    .step_by(size as usize)
                    .map(|first| Item {
                        rows: size.min(input.rows - first),
                        grid: None,
                    })
                    .collect()
            }
            Sequences::Groups { .. } => {
                return Err(Error(
                    "sequences: groups are a stage's own sequences".into(),
                ));
            }
        };
        let mut runs = Vec::new();
        let (mut first, mut written) = (0, 0);
        for part in parts {
            let (steps, after) = walk.steps(0..first_end, part)?;
            runs.push(Run {
                input: first..first + part.rows,
                gather: None,
                output: written..written + after.rows,
                steps,
            });
            first += part.rows;
            written += after.rows;
        }
        walk.segment = false;
        for (index, &join) in joins.iter().enumerate() {
            let end = joins.get(index + 1).copied().unwrap_or(self.stages.len());
            let StageOp::Group(group) = &self.stages[join].op else {
                return Err(Error(format!(
                    "stages[{join}]: joins segments without a gather rule"
                )));
            };
            let joined = written;
            let padded = rows[join].rows;
            let size = capacity / group.rows() * group.rows();
            let mut sources = group.sources(joined).into_iter();
            written = 0;
            for first in (0..padded).step_by(size.max(1) as usize) {
                let count = size.min(padded - first);
                let part = Item {
                    rows: count,
                    grid: None,
                };
                let (steps, after) = walk.steps(join..end, part)?;
                runs.push(Run {
                    input: first..first + count,
                    gather: Some(sources.by_ref().take(count as usize).collect()),
                    output: written..written + after.rows,
                    steps,
                });
                written += after.rows;
            }
        }
        let last = self.stages.len() - 1;
        if rows[last].rows != written {
            return Err(Error(format!(
                "runs write {written} rows, the trace {}",
                rows[last].rows
            )));
        }
        Ok(Geometry { input, rows, runs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_to_depth_gathers_cells_position_major_with_holes() {
        // A 2 x 3 grid in 2 x 2 cells: two cells, the right one half past
        // the grid.
        assert_eq!(
            space_to_depth_indices([1, 2, 3], 2).unwrap(),
            [0, 1, 3, 4, 2, -1, 5, -1]
        );
        // 7 x 5 in 3 x 3 cells: 3 x 2 cells of 9, the last row and column
        // of cells partly holes; every patch is read once.
        let indices = space_to_depth_indices([1, 7, 5], 3).unwrap();
        assert_eq!(indices.len(), 6 * 9);
        let mut read: Vec<i32> = indices.iter().copied().filter(|&i| i >= 0).collect();
        read.sort_unstable();
        assert_eq!(read, (0..35).collect::<Vec<_>>());
        assert_eq!(indices[9..18], [3, 4, -1, 8, 9, -1, 13, 14, -1]);
        assert!(space_to_depth_indices([1, 1, 1], 0).is_err());
    }

    #[test]
    fn grid_positions_follow_the_patch_order() {
        assert_eq!(
            grid_positions(Patchify::MergeBlocks, 2, 2, 4),
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3]
            ]
        );
        assert_eq!(
            grid_positions(Patchify::RowMajor, 1, 2, 2),
            [[0, 0], [0, 1], [1, 0], [1, 1]]
        );
        assert_eq!(bounds(12, 4), [0, 4, 8, 12]);
    }
}
