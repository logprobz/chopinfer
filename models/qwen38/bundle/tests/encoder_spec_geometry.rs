//! The geometry `encoder-spec` derives from Qwen3.8's vision declaration,
//! which the shared encoder runtime runs, against the rule the bundle's own
//! tower ran before it: one run per image of `h * w` patches in 2 x 2
//! merge-block order and `h * w / 4` merged rows, one attention sequence,
//! over grid shapes up to the 65,536-patch bound and its extreme aspect
//! ratios, with the F32 payloads (4 bytes a value) the tower takes.

use encoder_spec::geometry::{Media, Positions};
use encoder_spec::{Environment, components::Element};

const VISION: &str = include_str!("../../contract/encoders/qwen38-vision.json");

/// (row, column) of every patch in payload order: block rows, block
/// columns, then the 2 x 2 patches of each block.
fn merge_block_positions(h: u64, w: u64) -> Vec<[u64; 2]> {
    let mut out = Vec::new();
    for block_row in 0..h / 2 {
        for block_col in 0..w / 2 {
            for in_row in 0..2 {
                for in_col in 0..2 {
                    out.push([block_row * 2 + in_row, block_col * 2 + in_col]);
                }
            }
        }
    }
    out
}

fn check(declaration: &encoder_spec::Declaration, [h, w]: [u32; 2]) {
    let at = format!("{h}x{w}");
    let derived = declaration
        .geometry(
            "image.patch-grid.v1",
            Media::Grid([1, u64::from(h), u64::from(w)]),
        )
        .unwrap();
    let rows = u64::from(h) * u64::from(w);
    let output = derived.rows[derived.rows.len() - 2].rows;
    assert_eq!(output, rows / 4, "{at}");
    let [run] = derived.runs.as_slice() else {
        panic!("{at}: one image is one run");
    };
    assert_eq!(run.input, 0..rows);
    assert_eq!(run.output, 0..output);
    let steps: Vec<[u64; 2]> = run.steps.iter().map(|step| step.rows).collect();
    let (p, m) = (rows, output);
    assert_eq!(
        steps,
        [[p, p], [p, p], [p, p], [p, p], [p, m], [m, m], [m, m]],
        "{at}"
    );
    let block = &run.steps[2];
    assert_eq!(block.sequences, Some(vec![0, rows]), "{at}");
    assert_eq!(block.reorder, None);
    let Some(Positions::Grid(positions)) = &block.positions else {
        panic!("{at}: rotary follows the grid");
    };
    assert_eq!(
        positions,
        &merge_block_positions(u64::from(h), u64::from(w)),
        "{at}"
    );
}

#[test]
fn geometry_matches_the_towers_rule() {
    let vision = encoder_spec::parse(VISION, &Environment::new("qwen38")).unwrap();
    assert_eq!(vision.input.payload, Element::F32);
    let mut grids = Vec::new();
    for h in (2..=64).step_by(2) {
        for w in (2..=64).step_by(2) {
            grids.push([h, w]);
        }
    }
    grids.extend([
        [2, 32_768],
        [32_768, 2],
        [4, 16_384],
        [256, 256],
        [254, 258],
        [128, 512],
        [512, 128],
        [180, 364],
    ]);
    for grid in grids {
        check(&vision, grid);
    }
    for [h, w] in [[258, 256], [2, 32_770], [3, 4]] {
        let grid = Media::Grid([1, h, w]);
        assert!(
            vision.geometry("image.patch-grid.v1", grid).is_err(),
            "{h}x{w}"
        );
    }
    assert!(
        vision
            .geometry("image.patch-grid.v1", Media::Grid([2, 2, 2]))
            .is_err()
    );
}
