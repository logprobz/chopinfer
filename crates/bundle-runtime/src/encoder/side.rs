//! One run's side inputs, computed on the host from its geometry: the live
//! row counts, the attention sequence bounds, the rotary rows of the
//! transformer and the bilinear taps of a learned position grid, each in
//! the reference's F32 arithmetic. A model's registered rotary variants
//! compute their rows through the bundle's `Registered`.
//!
//! A declaration of one run per item uploads every side input before the
//! run's first launch (`Side::uploads`). A staged one
//! (`Declaration::staged`) has the launch list stage them in command order
//! (`Side::staged`): the rotary rows before each block, in the rows' order
//! or gathered by the block's reorder, and the reorder's index maps before
//! each gather and scatter.

use std::collections::BTreeMap;

use encoder_spec::components::rotary::{Rotary, Table};
use encoder_spec::components::{Position, RotarySlot, StageOp};
use encoder_spec::geometry::{Geometry, Positions, Run, grid_positions};
use kernel_core::rotary::{AngleRule, Axes, ComputedTable};

use super::lower::Lowered;
use super::registered::Registered;

/// Which order of a transformer's rows a staged table follows.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Order {
    /// The rows' own order.
    Rows,
    /// The order the transformer's reorder gathers its rows into.
    Gathered,
}

/// A side input a staged launch list stages, by transformer stage.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Input {
    Cos(usize, Order),
    Sin(usize, Order),
    /// The reorder's index map into `Order` (destination row to source
    /// row): `Gathered` gathers the rows, `Rows` scatters them back.
    Reorder(usize, Order),
}

fn le_bytes<T: Copy, const N: usize>(values: &[T], bytes: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&value| bytes(value)).collect()
}

/// A region by name and owning stage (`buffers::Region::owner`).
pub(super) type Key = (&'static str, Option<usize>);

/// The bytes of each side region a run uploads, in upload order.
#[derive(Debug, Default)]
pub(super) struct Side {
    /// Uploaded before the run's first launch (one run per item).
    pub uploads: Vec<(Key, Vec<u8>)>,
    /// Staged by the launch list (staged declarations).
    pub staged: BTreeMap<Input, Vec<u8>>,
    /// The attention sequences of each transformer stage of the run, and
    /// their bounds (staged declarations; others upload them).
    pub sequences: Vec<(usize, u32)>,
    pub bounds: Vec<(usize, Vec<u8>)>,
    /// Each count slot's live rows.
    pub counts: Vec<u32>,
}

/// `_interpolation_axis_taps_weights`: bilinear with `align_corners=True`
/// over a table side of `side`, in F32.
fn axis_taps(index: u64, size: u64, side: u32) -> ([u32; 2], [f32; 2]) {
    let source = index as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
    let floor = source.floor();
    let fraction = source - floor;
    let taps = [0, 1].map(|offset| (floor as u32 + offset).min(side - 1));
    let weights = [0.0_f32, 1.0].map(|offset| (1.0 - (fraction - offset).abs()).max(0.0));
    (taps, weights)
}

/// The four bilinear taps into a `table` grid of every position of an
/// `h x w` grid: indices and F32 weights in the order (r0, c0), (r0, c1),
/// (r1, c0), (r1, c1).
pub fn position_taps(
    positions: &[[u64; 2]],
    [h, w]: [u64; 2],
    table: [u32; 2],
) -> (Vec<i32>, Vec<f32>) {
    let mut indices = Vec::with_capacity(4 * positions.len());
    let mut weights = Vec::with_capacity(4 * positions.len());
    for &[row, column] in positions {
        let (row_taps, row_weights) = axis_taps(row, h, table[0]);
        let (column_taps, column_weights) = axis_taps(column, w, table[1]);
        for tap in 0..4 {
            indices.push((row_taps[tap / 2] * table[1] + column_taps[tap % 2]) as i32);
            weights.push(row_weights[tap / 2] * column_weights[tap % 2]);
        }
    }
    (indices, weights)
}

/// The rotary cos and sin rows `[rows, pairs]` F32 of `positions` under
/// `table` (`AngleRule::MulRecipF64PowF32`, axial): each row's grid row
/// times the first half of the inverse frequencies, then its column.
pub fn rotary(positions: &[[u64; 2]], theta: f64, pairs: usize) -> (Vec<f32>, Vec<f32>) {
    let table = ComputedTable {
        theta,
        rule: AngleRule::MulRecipF64PowF32,
        axes: Axes::Axial2d,
    };
    let mut cos = Vec::with_capacity(pairs * positions.len());
    let mut sin = Vec::with_capacity(pairs * positions.len());
    for &[row, column] in positions {
        let row = table
            .row([row as usize, column as usize, 0], pairs)
            .expect("an axial rotary table of validated pairs");
        for (c, s) in row {
            cos.push(c as f32);
            sin.push(s as f32);
        }
    }
    (cos, sin)
}

/// The rotary cos and sin rows `[rows, pairs]` F32 of each row's position
/// in its sequence under `table` (`AngleRule::MulRecipF64PowF32`, one
/// axis).
pub fn rotary_1d(positions: &[u64], theta: f64, pairs: usize) -> (Vec<f32>, Vec<f32>) {
    let table = ComputedTable {
        theta,
        rule: AngleRule::MulRecipF64PowF32,
        axes: Axes::Linear,
    };
    let mut cos = Vec::with_capacity(pairs * positions.len());
    let mut sin = Vec::with_capacity(pairs * positions.len());
    for &position in positions {
        let row = table
            .row([position as usize, 0, 0], pairs)
            .expect("a rotary table of validated pairs");
        for (c, s) in row {
            cos.push(c as f32);
            sin.push(s as f32);
        }
    }
    (cos, sin)
}

/// `rows` of `width` values each, gathered by `map` (destination row to
/// source row).
fn gather<T: Copy>(rows: &[T], width: usize, map: &[u64]) -> Vec<T> {
    map.iter()
        .flat_map(|&source| &rows[source as usize * width..(source as usize + 1) * width])
        .copied()
        .collect()
}

/// The side inputs of `run` of an item of `geometry`, whose recipe emits
/// patches in merge blocks of side `merge`.
pub(super) fn inputs(
    lowered: &Lowered,
    host: &dyn Registered,
    geometry: &Geometry,
    merge: u32,
    run: &Run,
) -> Result<Side, String> {
    let declaration = &lowered.declaration;
    let staged = lowered.staged;
    let mut counts = vec![0_u32; lowered.counts.len()];
    for step in &run.steps {
        let [read, write] = lowered.stage_counts[step.stage];
        counts[read] = step.rows[0] as u32;
        counts[write] = step.rows[1] as u32;
    }
    let mut side = Side {
        uploads: Vec::new(),
        counts,
        ..Side::default()
    };
    if !staged {
        side.uploads
            .push((("counts", None), le_bytes(&side.counts, u32::to_le_bytes)));
    }
    let mut taps = Vec::new();
    for step in &run.steps {
        match &declaration.stages[step.stage].op {
            StageOp::Transformer(transformer) => {
                let bounds = step
                    .sequences
                    .as_ref()
                    .ok_or("a transformer without sequences")?;
                let bounds: Vec<i32> = bounds.iter().map(|&bound| bound as i32).collect();
                let stage = step.stage;
                let own = Some(stage);
                side.sequences.push((stage, bounds.len() as u32 - 1));
                let bounds = le_bytes(&bounds, i32::to_le_bytes);
                let rotary_rows = match &transformer.params().rotary {
                    Some(RotarySlot::Shared(Rotary::Axial2d(axial))) => {
                        let Table::HfF32Angle = axial.table;
                        let Some(Positions::Grid(positions)) = &step.positions else {
                            return Err("an axial rotary without grid positions".into());
                        };
                        let pairs = 2 * axial.pairs_per_axis as usize;
                        let (cos, sin) = rotary(positions, axial.theta, pairs);
                        Some((cos, sin, pairs))
                    }
                    Some(RotarySlot::Shared(Rotary::OneD(params))) => {
                        let Table::HfF32Angle = params.table;
                        let Some(Positions::Sequence(positions)) = &step.positions else {
                            return Err("a 1D rotary without sequence positions".into());
                        };
                        let pairs = params.pairs as usize;
                        let (cos, sin) = rotary_1d(positions, params.theta, pairs);
                        Some((cos, sin, pairs))
                    }
                    Some(RotarySlot::Registered(registered)) => {
                        let component = registered.component.as_ref();
                        let positions = step
                            .positions
                            .as_ref()
                            .ok_or("a rotary without positions")?;
                        let (cos, sin) = host.rotary(component, positions)?;
                        let half = component.rotary_dims()? as usize / 2;
                        if cos.len() != step.rows[0] as usize * half || sin.len() != cos.len() {
                            return Err(format!("`{}`: a table of other rows", component.id()));
                        }
                        Some((cos, sin, half))
                    }
                    Some(RotarySlot::Unresolved(unresolved)) => {
                        return Err(format!("`{}` is not registered", unresolved.id));
                    }
                    None => None,
                };
                if !staged {
                    let (cos, sin, _) =
                        rotary_rows.ok_or("a transformer of one run per item without rotary")?;
                    side.uploads.push((("sequences", own), bounds));
                    side.uploads
                        .push((("cos", own), le_bytes(&cos, f32::to_le_bytes)));
                    side.uploads
                        .push((("sin", own), le_bytes(&sin, f32::to_le_bytes)));
                    continue;
                }
                side.bounds.push((stage, bounds));
                if let Some(map) = &step.reorder {
                    let mut back = vec![0_i32; map.len()];
                    for (destination, &source) in map.iter().enumerate() {
                        back[source as usize] = destination as i32;
                    }
                    let into: Vec<i32> = map.iter().map(|&source| source as i32).collect();
                    side.staged.insert(
                        Input::Reorder(stage, Order::Gathered),
                        le_bytes(&into, i32::to_le_bytes),
                    );
                    side.staged.insert(
                        Input::Reorder(stage, Order::Rows),
                        le_bytes(&back, i32::to_le_bytes),
                    );
                }
                let Some((cos, sin, half)) = rotary_rows else {
                    continue;
                };
                if let Some(map) = &step.reorder {
                    side.staged.insert(
                        Input::Cos(stage, Order::Gathered),
                        le_bytes(&gather(&cos, half, map), f32::to_le_bytes),
                    );
                    side.staged.insert(
                        Input::Sin(stage, Order::Gathered),
                        le_bytes(&gather(&sin, half, map), f32::to_le_bytes),
                    );
                }
                side.staged.insert(
                    Input::Cos(stage, Order::Rows),
                    le_bytes(&cos, f32::to_le_bytes),
                );
                side.staged.insert(
                    Input::Sin(stage, Order::Rows),
                    le_bytes(&sin, f32::to_le_bytes),
                );
            }
            StageOp::Position(Position::LearnedGridBilinearF32(params)) => {
                let [t, h, w] = geometry
                    .input
                    .grid
                    .ok_or("a position grid without a grid")?;
                let patchify = declaration
                    .input
                    .patchify
                    .ok_or("a grid without patchify")?;
                let one = grid_positions(patchify, u64::from(merge), h, w);
                let positions: Vec<[u64; 2]> = (0..t)
                    .flat_map(|_| one.iter().copied())
                    .skip(run.input.start as usize)
                    .take(step.rows[0] as usize)
                    .collect();
                taps.push((step.stage, position_taps(&positions, [h, w], params.table)));
            }
            _ => {}
        }
    }
    // The taps upload after the transformers' tables, as the tower they
    // replace did.
    for (stage, (indices, weights)) in taps {
        side.uploads.push((
            ("tap-indices", Some(stage)),
            le_bytes(&indices, i32::to_le_bytes),
        ));
        side.uploads.push((
            ("tap-weights", Some(stage)),
            le_bytes(&weights, f32::to_le_bytes),
        ));
    }
    Ok(side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_taps_match_hf_by_hand() {
        // An 8 x 12 grid in 2 x 2 merge blocks on a 48 x 48 table: row r
        // maps to 47 r / 7, column c to 47 c / 11.
        let positions = grid_positions(encoder_spec::Patchify::MergeBlocks, 2, 8, 12);
        let (indices, weights) = position_taps(&positions, [8, 12], [48, 48]);
        assert_eq!(indices.len(), 96 * 4);
        // Patch 0 is (0, 0): all weight on table row 0.
        assert_eq!(&indices[..4], &[0, 1, 48, 49]);
        assert_eq!(&weights[..4], &[1.0, 0.0, 0.0, 0.0]);
        // Patch 3 is (1, 1): row 47/7 = 6.714 between table rows 6 and 7,
        // column 47/11 = 4.273 between table columns 4 and 5.
        let (dr, dc) = (47.0_f32 / 7.0 - 6.0, 47.0_f32 / 11.0 - 4.0);
        assert_eq!(
            &indices[12..16],
            &[6 * 48 + 4, 6 * 48 + 5, 7 * 48 + 4, 7 * 48 + 5]
        );
        let expected = [
            (1.0 - dr) * (1.0 - dc),
            (1.0 - dr) * (1.0 - (dc - 1.0).abs()),
            (1.0 - (dr - 1.0).abs()) * (1.0 - dc),
            (1.0 - (dr - 1.0).abs()) * (1.0 - (dc - 1.0).abs()),
        ];
        assert_eq!(&weights[12..16], &expected);
        assert!((weights[12..16].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // The last patch is (7, 11): the far corner, its second taps clamped.
        assert_eq!(&indices[95 * 4..], &[2303; 4]);
        assert_eq!(&weights[95 * 4..], &[1.0, 0.0, 0.0, 0.0]);
        // A non-square table indexes rows by its column count.
        let (indices, _) = position_taps(&[[1, 0]], [2, 2], [3, 5]);
        assert_eq!(indices, [10, 11, 10, 11]);
    }

    #[test]
    fn rotary_rows_are_height_then_width_frequencies() {
        // HF's F32 inv_freq: an f64 power of the F32 exponent j / 36.
        let inverse = (0..36)
            .step_by(2)
            .map(|j| 1.0 / 10_000_f64.powf(f64::from(j as f32 / 36.0)) as f32)
            .collect::<Vec<_>>();
        let positions = grid_positions(encoder_spec::Patchify::MergeBlocks, 2, 8, 12);
        let (cos, sin) = rotary(&positions, 10_000.0, 36);
        assert_eq!(cos.len(), 96 * 36);
        // Patch 5 is (0, 3): zero height angles, width angles 3 f.
        assert!(cos[5 * 36..5 * 36 + 18].iter().all(|&x| x == 1.0));
        assert!(sin[5 * 36..5 * 36 + 18].iter().all(|&x| x == 0.0));
        assert_eq!(sin[5 * 36 + 18], 3.0_f64.sin() as f32);
        assert_eq!(cos[5 * 36 + 19], f64::from(3.0 * inverse[1]).cos() as f32);
        // Patch 2 is (1, 0).
        assert_eq!(cos[2 * 36], 1.0_f64.cos() as f32);
        assert_eq!(sin[2 * 36 + 18], 0.0);
    }
}
