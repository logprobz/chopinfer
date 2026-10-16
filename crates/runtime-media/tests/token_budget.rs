//! `image.zero-padded-token-budget.v1` against the reference image
//! processor's goldens in `fixtures/zero-padded-token-budget.json`: every
//! canvas and content size of a sweep of sources (each canvas its own), and
//! the SHA-256 of each image's F32 rows (bit-identical values), grid and
//! token count. Synthetic sources come from `pattern`, which the golden
//! generator mirrors; the grayscale, alpha and tRNS sources are PNG files
//! decoded here with alpha discarded, as the reference's PIL conversion
//! does.
use std::path::Path;

use runtime_media::recipe::{
    TokenBudgetRecipe, padded_emits_size, padded_size, padded_token_budget_grid,
};
use runtime_media::{Alpha, MediaLimits, RgbImage, png};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixtures().join("zero-padded-token-budget.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 1 << 20,
        maximum_pixels: 1 << 26,
        maximum_audio_frames: 0,
        maximum_frames: 2,
        maximum_tokens: 8000,
        maximum_aspect_ratio: 10_000.0,
    }
}

fn recipe(golden: &Value) -> TokenBudgetRecipe {
    let r = &golden["recipe"];
    let u = |key: &str| r[key].as_u64().unwrap();
    let vector = |key: &str| {
        let values: Vec<f32> = r[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        <[f32; 3]>::try_from(values).unwrap()
    };
    TokenBudgetRecipe {
        patch: u("patch") as u32,
        temporal_patch: u("temporal_patch") as u32,
        merge: u("merge") as u32,
        minimum_tokens: u("minimum_tokens"),
        maximum_tokens: u("maximum_tokens"),
        value_scale: r["value_scale"].as_f64().unwrap(),
        mean: vector("mean"),
        std: vector("std"),
    }
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

#[test]
fn canvas_and_content_sizes_match_the_reference() {
    let golden = golden();
    let recipe = recipe(&golden);
    let sizes = golden["sizes"].as_array().unwrap();
    assert!(sizes.len() >= 400);
    for row in sizes {
        let [h, w, canvas_h, canvas_w, content_h, content_w] = <[u32; 6]>::try_from(
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let size = padded_size(h, w, recipe, limits()).unwrap();
        assert_eq!(
            (size.canvas, size.content),
            ((canvas_h, canvas_w), (content_h, content_w)),
            "{h} x {w}"
        );
        assert!(
            padded_emits_size(canvas_h, canvas_w, recipe, limits()),
            "{h} x {w}"
        );
    }
}

#[test]
fn rows_match_the_reference_bit_for_bit() {
    let golden = golden();
    let recipe = recipe(&golden);
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 25);
    let digest = |image: &RgbImage| {
        let grid = padded_token_budget_grid(image, recipe, limits()).unwrap();
        let mut digest = Sha256::new();
        for value in &grid.rows {
            digest.update(value.to_le_bytes());
        }
        let digest: String = digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        (grid, digest)
    };
    let mut transparent = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let (width, height) = (
            case["width"].as_u64().unwrap() as u32,
            case["height"].as_u64().unwrap() as u32,
        );
        let png = case["png"]
            .as_str()
            .map(|file| std::fs::read(fixtures().join(file)).unwrap());
        let image = match &png {
            Some(bytes) => png::decode_with_alpha(bytes, limits(), Alpha::Discard).unwrap(),
            None => pattern(case["pattern"].as_str().unwrap(), width, height),
        };
        assert_eq!((image.width, image.height), (width, height), "{name}");
        let (grid, rows) = digest(&image);
        let expected: Vec<u32> = case["grid"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(grid.grid[..], expected[..], "{name}");
        assert_eq!(grid.tokens, case["tokens"].as_u64().unwrap(), "{name}");
        assert_eq!(grid.columns, 3 * 2 * 14 * 14, "{name}");
        assert_eq!(rows, case["rows_sha256"].as_str().unwrap(), "{name}");
        // Compositing onto white would change every transparent case.
        if case["transparent"].as_bool().unwrap() {
            let composited = png::decode(png.as_ref().unwrap(), limits()).unwrap();
            assert_ne!(digest(&composited).1, rows, "{name}");
            transparent += 1;
        }
    }
    assert_eq!(transparent, 6);
}
