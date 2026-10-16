//! The geometry `encoder-spec` derives from MiMo's encoder declarations:
//! audio's against independent segment arithmetic, its runs, rows,
//! sequence bounds, rotary positions and group gather across segment
//! boundaries and its positions; both
//! towers' rotary rows (the shared runtime's) against HF's F32 tables, and
//! the recipes' limits.

use bundle_mimo26::encoders::injection::validate_record_for;

fn validate_record(record: &BatchMediaEncodeV3) -> Result<(), &'static str> {
    validate_record_for(&model_mimo26_contract::geometry::V26_FLASH, record)
}
use encoder_spec::Declaration;
use encoder_spec::geometry::{Geometry, Media, Positions};
use model_mimo26_contract::encoders::spec;
use runtime_bundle_sdk::BatchMediaEncodeV3;

fn declaration(text: &str) -> Declaration {
    encoder_spec::parse(text, &spec::environment().unwrap()).unwrap()
}

fn record(encoder: u32, geometry: [u32; 4], output_rows: u64, payload: u64) -> BatchMediaEncodeV3 {
    BatchMediaEncodeV3 {
        encoder,
        geometry,
        output_rows: output_rows as u32,
        pages: vec![0],
        payload_offset: 0,
        payload_bytes: payload,
    }
}

fn last_rows(geometry: &Geometry) -> u64 {
    geometry.rows[geometry.rows.len() - 2].rows
}

/// HF's vision rotary table (`table.hf-f32-angle.v1`): the F32 exponent
/// `2i / 32`, an f64 power of theta rounded to F32, its F32 reciprocal, the
/// F32 product with the patch row (first 16 pairs) or column (next 16), and
/// the f64 cos and sin of that angle rounded to F32.
fn hf_table(grid: &[[u64; 2]]) -> (Vec<f32>, Vec<f32>) {
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &[row, column] in grid {
        for position in [row, column] {
            for pair in 0..16 {
                let exponent = (2 * pair) as f32 / 32.0;
                let frequency = 1.0 / 10_000_f64.powf(f64::from(exponent)) as f32;
                let angle = f64::from(position as f32 * frequency);
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
    }
    (cos, sin)
}

/// The shared runtime's vision rotary rows at a grid's positions as the
/// declaration derives them, against HF's table.
fn check_vision(declaration: &Declaration, tag: &str, [t, h, w]: [u32; 3]) {
    let derived = declaration
        .geometry(tag, Media::Grid([t, h, w].map(u64::from)))
        .unwrap();
    let rows = u64::from(t * h * w);
    let output = last_rows(&derived);
    assert_eq!(output, rows / 4);
    validate_record(&record(0, [t, h, w, 0], output, rows * 1536 * 4)).unwrap();
    for run in &derived.runs {
        let at = format!("{tag} {t}x{h}x{w} run {:?}", run.input);
        let positions = run.steps[1].positions.as_ref().unwrap();
        let Positions::Grid(grid) = positions else {
            panic!("{at}: vision rotary follows the grid");
        };
        let rows = bundle_runtime::encoder::side::rotary(grid, 10_000.0, 32);
        assert_eq!(rows, hf_table(grid), "{at}");
    }
}

#[test]
fn vision_rotary_rows_follow_the_derived_grid() {
    let vision = declaration(spec::VISION);
    for grid in [[1, 2, 2], [1, 4, 6], [1, 62, 66], [1, 2, 2048], [1, 64, 64]] {
        check_vision(&vision, "image.patch-grid.v1", grid);
    }
    for t in [1, 3, 9, 16] {
        for [h, w] in [[32, 32], [2, 512], [6, 10]] {
            check_vision(&vision, "video.timestamped-grids.v1", [t, h, w]);
        }
    }
    // Each recipe's limits bound what the declaration admits under it; the
    // bundle admits a record exactly when some recipe does ([2, 2, 2] is
    // video, [1, 34, 32] an image).
    for (tag, grid, record) in [
        ("image.patch-grid.v1", [1, 66, 64], false),
        ("image.patch-grid.v1", [2, 2, 2], true),
        ("image.patch-grid.v1", [1, 3, 4], false),
        ("video.timestamped-grids.v1", [17, 2, 2], false),
        ("video.timestamped-grids.v1", [1, 34, 32], true),
    ] {
        assert!(vision.geometry(tag, Media::Grid(grid)).is_err(), "{grid:?}");
        let [t, h, w] = grid.map(|side| side as u32);
        assert_eq!(
            vision.admit_record([t, h, w, 0]).is_ok(),
            record,
            "{grid:?}"
        );
    }
}

/// HF's 1D rotary table (`table.hf-f32-angle.v1`): the F32 exponent
/// `2i / 64`, an f64 power of theta rounded to F32, its F32 reciprocal, the
/// F32 product with the row's index in its sequence, and the f64 cos and
/// sin of that angle rounded to F32.
fn hf_table_1d(positions: &[u64], theta: f64) -> (Vec<f32>, Vec<f32>) {
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for &position in positions {
        for pair in 0..32 {
            let exponent = (2 * pair) as f32 / 64.0;
            let frequency = 1.0 / theta.powf(f64::from(exponent)) as f32;
            let angle = f64::from(position as f32 * frequency);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

/// The shared runtime's 1D rotary rows at `positions` against HF's table.
fn check_rotary_1d(positions: &[u64], theta: f64) {
    assert_eq!(
        bundle_runtime::encoder::side::rotary_1d(positions, theta, 32),
        hf_table_1d(positions, theta)
    );
}

fn check_audio(declaration: &Declaration, frames: u32, tables: bool) {
    let derived = declaration
        .geometry("audio.log-mel.v1", Media::Frames(u64::from(frames)))
        .unwrap();
    // Independent segment and group arithmetic, including odd-length tails.
    let segments = (0..frames)
        .step_by(6000)
        .map(|first| {
            let count = (frames - first).min(6000);
            (
                first,
                count,
                count.div_ceil(2),
                count.div_ceil(2).div_ceil(2),
            )
        })
        .collect::<Vec<_>>();
    let codes: u32 = segments.iter().map(|s| s.3).sum();
    let padded_codes = codes.div_ceil(4) * 4;
    let at = format!("{frames} frames");
    assert_eq!(last_rows(&derived), u64::from(padded_codes / 4), "{at}");
    assert_eq!(derived.rows[6].rows, u64::from(codes), "{at}");
    assert_eq!(derived.rows[7].rows, u64::from(padded_codes), "{at}");
    validate_record(&record(
        1,
        [frames, 128, 0, 0],
        last_rows(&derived),
        u64::from(frames) * 128 * 4,
    ))
    .unwrap();
    let (runs, chunks) = derived.runs.split_at(segments.len());
    assert_eq!(chunks.len(), padded_codes.div_ceil(8192) as usize, "{at}");
    let mut first_code = 0;
    for (run, &(first, f, a, c)) in runs.iter().zip(&segments) {
        let (first, f, a, c) = (u64::from(first), u64::from(f), u64::from(a), u64::from(c));
        assert_eq!(run.input, first..first + f, "{at}");
        assert_eq!(run.output, first_code..first_code + c, "{at}");
        first_code += c;
        let rows: Vec<[u64; 2]> = run.steps.iter().map(|step| step.rows).collect();
        assert_eq!(
            rows,
            [[f, f], [f, a], [a, a], [a, a], [a, c], [c, c], [c, c]],
            "{at}"
        );
        let block = &run.steps[2];
        assert_eq!(block.sequences, Some(vec![0, a]), "{at}");
        let Some(Positions::Sequence(positions)) = &block.positions else {
            panic!("{at}: tokenizer rotary follows the sequence");
        };
        if tables {
            check_rotary_1d(positions, 10_000.0);
        } else {
            assert!(positions.iter().copied().eq(0..a));
        }
    }
    for (run, first) in chunks.iter().zip((0..padded_codes).step_by(8192)) {
        let rows = u64::from((padded_codes - first).min(8192));
        let first = u64::from(first);
        assert_eq!(run.input, first..first + rows, "{at}");
        let sources: Vec<u64> = (first..first + rows)
            .map(|r| r.min(u64::from(codes) - 1))
            .collect();
        assert_eq!(run.gather.as_ref(), Some(&sources), "{at}");
        assert_eq!(run.output, first / 4..(first + rows) / 4, "{at}");
        let block = &run.steps[2];
        assert_eq!(block.stage, 9);
        let cu: Vec<u64> = (0..=rows / 4).map(|i| i * 4).collect();
        assert_eq!(block.sequences.as_ref(), Some(&cu), "{at}");
        let Some(Positions::Sequence(positions)) = &block.positions else {
            panic!("{at}: local rotary follows the group");
        };
        if tables {
            check_rotary_1d(positions, 640_000.0);
        } else {
            assert!(positions.iter().copied().eq((0..rows).map(|row| row % 4)));
        }
    }
}

#[test]
fn audio_geometry_matches_the_bundle() {
    let audio = declaration(spec::AUDIO);
    for frames in 1..=13_000 {
        check_audio(&audio, frames, false);
    }
    for frames in [
        1, 2, 3, 4, 5, 17, 1595, 5999, 6000, 6001, 12_000, 12_001, 18_003, 32_768, 32_772, 59_999,
        60_000, 60_001,
    ] {
        check_audio(&audio, frames, true);
    }
    for frames in [0, 60_002] {
        assert!(
            audio
                .geometry("audio.log-mel.v1", Media::Frames(frames))
                .is_err()
        );
        assert!(audio.admit_record([frames as u32, 128, 0, 0]).is_err());
    }
}
