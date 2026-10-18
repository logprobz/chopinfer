//! `image.centered-pad.v1` against the reference image preprocessing's
//! goldens in `fixtures/centered-pad.json`: the plan (canvas, content,
//! offset and token grid) of a sweep of source sizes, and per image the
//! SHA-256 of its F32 rows (bit-identical values), of their BF16 payload
//! (the reference's BF16 patches), its grid and token counts. Synthetic
//! sources come from `pattern`, which the golden generator mirrors; the
//! grayscale, alpha, palette and tRNS PNGs and the JPEGs are files decoded
//! as the reference's `Image.open` and `convert("RGB")` read them: alpha
//! discarded and an EXIF orientation kept as stored.
use std::path::Path;

use runtime_media::recipe::{
    CenteredPadPlan, CenteredPadRecipe, centered_pad_grid, centered_pad_plan,
    centered_pad_span_tokens, check_centered_pad_recipe,
};
use runtime_media::{Alpha, MediaLimits, Orientation, RgbImage, jpeg, png};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixtures().join("centered-pad.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 1 << 20,
        maximum_pixels: 1 << 28,
        maximum_audio_frames: 0,
        maximum_frames: 1,
        maximum_tokens: 8000,
        maximum_aspect_ratio: 30_000.0,
    }
}

fn recipe(golden: &Value) -> CenteredPadRecipe {
    let r = &golden["recipe"];
    let u = |key: &str| r[key].as_u64().unwrap();
    let vector = |key: &str| -> [f32; 3] {
        let values: Vec<f32> = r[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        values.try_into().unwrap()
    };
    let pad: Vec<u8> = r["pad"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();
    let recipe = CenteredPadRecipe {
        patch: u("patch") as u32,
        downsample: u("downsample") as u32,
        minimum_pixels: u("minimum_pixels"),
        maximum_tokens: u("maximum_tokens"),
        pad: pad.try_into().unwrap(),
        value_divisor: r["value_divisor"].as_f64().unwrap() as f32,
        mean: vector("mean"),
        std: vector("std"),
    };
    check_centered_pad_recipe(recipe).unwrap();
    recipe
}

fn numbers(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect()
}

/// `[canvas_h, canvas_w, content_h, content_w, offset_y, offset_x,
/// token_rows, token_columns]`.
fn flat(plan: CenteredPadPlan) -> Vec<u32> {
    vec![
        plan.canvas.0,
        plan.canvas.1,
        plan.content.0,
        plan.content.1,
        plan.offset.0,
        plan.offset.1,
        plan.tokens.0,
        plan.tokens.1,
    ]
}

/// The generator's integer patterns: a hashed noise, and diagonal 7 x 5
/// blocks.
fn pattern(kind: &str, width: u32, height: u32) -> RgbImage {
    let value = |x: u64, y: u64, c: u64| -> u8 {
        match kind {
            "noise" => {
                let mut h =
                    ((x * 0x9E37_79B1) ^ (y * 0x85EB_CA77) ^ (c * 0xC2B2_AE3D)) & 0xFFFF_FFFF;
                h ^= h >> 15;
                h = (h * 0x2C1B_3C6D) & 0xFFFF_FFFF;
                h ^= h >> 12;
                (h & 255) as u8
            }
            "blocks" => ((x / 7 * 37 + y / 5 * 59 + c * 85) % 256) as u8,
            "zero" => 0,
            _ => panic!("unknown pattern {kind}"),
        }
    };
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..u64::from(height) {
        for x in 0..u64::from(width) {
            data.extend((0..3).map(|c| value(x, y, c)));
        }
    }
    RgbImage::new(width, height, data, limits()).unwrap()
}

fn sha256(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes.into_iter().collect::<Vec<_>>());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[test]
fn plans_match_the_reference() {
    let golden = golden();
    let recipe = recipe(&golden);
    let sizes = golden["sizes"].as_array().unwrap();
    assert_eq!(sizes.len(), 960);
    let mut solved = 0;
    for row in sizes {
        let row = numbers(row);
        let (h, w) = (row[0], row[1]);
        let plan = centered_pad_plan(h, w, recipe, limits()).unwrap();
        assert_eq!(flat(plan), row[2..], "{h} x {w}");
        assert!(
            centered_pad_span_tokens(plan.tokens.0.into(), plan.tokens.1.into()) <= 1024,
            "{h} x {w}"
        );
        let (p, d) = (14, 3);
        let aligned = |side: u32| side.div_ceil(p) * p;
        let grid = |side: u32| (aligned(side) / p).div_ceil(d);
        solved += usize::from(
            centered_pad_span_tokens(grid(h).into(), grid(w).into()) > 1024 && h * w >= 295_936,
        );
    }
    assert!(solved > 50, "{solved} sizes solved the budget");
    // Span costs 1022 to 1026 at one token row: a 42-high source keeps its
    // canvas through 1024 and solves above it.
    for (width, cost) in [
        (42 * 1019_u32, 1022),
        (42 * 1020, 1023),
        (42 * 1021, 1024),
        (42 * 1021 + 1, 1025),
        (42 * 1023, 1026),
    ] {
        let columns = u64::from(width.div_ceil(14)).div_ceil(3);
        assert_eq!(centered_pad_span_tokens(1, columns), cost);
        let plan = centered_pad_plan(42, width, recipe, limits()).unwrap();
        assert_eq!(
            plan.canvas.1 == width.div_ceil(14) * 14,
            cost <= 1024,
            "{width}"
        );
    }
    // The largest square the pixel limit takes plans; one more row is
    // refused.
    assert_eq!(
        centered_pad_plan(16384, 16384, recipe, limits())
            .unwrap()
            .canvas,
        (1302, 1302)
    );
    let refusal = centered_pad_plan(16385, 16384, recipe, limits()).unwrap_err();
    assert!(refusal.to_string().contains("pixel limit"), "{refusal}");
}

#[test]
fn rows_match_the_reference_bit_for_bit() {
    let golden = golden();
    let recipe = recipe(&golden);
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 36);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let (width, height) = (
            case["width"].as_u64().unwrap() as u32,
            case["height"].as_u64().unwrap() as u32,
        );
        let image = match case["file"].as_str() {
            Some(file) => {
                let bytes = std::fs::read(fixtures().join(file)).unwrap();
                let image = if file.ends_with(".jpg") {
                    jpeg::decode_oriented(&bytes, limits(), Orientation::Stored).unwrap()
                } else {
                    png::decode_with_alpha(&bytes, limits(), Alpha::Discard).unwrap()
                };
                assert_eq!(
                    sha256(image.data.iter().copied()),
                    case["decoded_rgb_sha256"].as_str().unwrap(),
                    "{name}: decoded pixels"
                );
                image
            }
            None => pattern(case["pattern"].as_str().unwrap(), width, height),
        };
        assert_eq!((image.width, image.height), (width, height), "{name}");
        let plan = centered_pad_plan(height, width, recipe, limits()).unwrap();
        assert_eq!(flat(plan), numbers(&case["plan"]), "{name}");
        let grid = centered_pad_grid(&image, recipe, limits()).unwrap();
        assert_eq!(grid.grid[..], numbers(&case["grid"])[..], "{name}");
        assert_eq!(grid.tokens, case["tokens"].as_u64().unwrap(), "{name}");
        assert_eq!(
            centered_pad_span_tokens(plan.tokens.0.into(), plan.tokens.1.into()),
            case["span_tokens"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(grid.columns, 3 * 14 * 14, "{name}");
        let rows = sha256(grid.rows.iter().flat_map(|v| v.to_le_bytes()));
        assert_eq!(rows, case["rows_sha256"].as_str().unwrap(), "{name}");
        assert_eq!(
            sha256(grid.bf16_bytes()),
            case["bf16_sha256"].as_str().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn exif_orientation_is_kept_as_stored() {
    // The tagged orientation 6 would rotate 8 x 16 to 16 x 8; the recipe's
    // reference keeps the stored size, and the default decoder rotates.
    let bytes = std::fs::read(fixtures().join("centered-pad-jpeg-exif6-8x16.jpg")).unwrap();
    assert_eq!(
        jpeg::probe_oriented(&bytes, limits(), Orientation::Stored).unwrap(),
        (8, 16)
    );
    assert_eq!(jpeg::probe(&bytes, limits()).unwrap(), (16, 8));
    let stored = jpeg::decode_oriented(&bytes, limits(), Orientation::Stored).unwrap();
    let rotated = jpeg::decode(&bytes, limits()).unwrap();
    assert_eq!((stored.width, stored.height), (8, 16));
    assert_eq!((rotated.width, rotated.height), (16, 8));
    // Orientation 6 rotates clockwise: the rotated image's top row is the
    // stored left column read bottom to top.
    let pixel = |image: &RgbImage, x: u32, y: u32| {
        let at = ((y * image.width + x) * 3) as usize;
        image.data[at..at + 3].to_vec()
    };
    for x in 0..16 {
        assert_eq!(pixel(&rotated, x, 0), pixel(&stored, 0, 15 - x));
    }
}
