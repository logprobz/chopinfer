//! Qwen3.8's vision declaration against the host's media path: the patch
//! grids runtime-media's resize emits for the frontend's image recipe are
//! what the declaration admits, its derived rows are the host's tokens, and
//! its placement takes the engine's M-RoPE positions.

use encoder_spec::geometry::Media;
use encoder_spec::{Declaration, Environment, parse};
use runtime_media::MediaLimits;
use runtime_media::recipe::{ImageRecipe, image_grid, smart_resize};
use serde_json::Value;

const VISION: &str = include_str!("../encoders/qwen38-vision.json");
const TAG: &str = "image.patch-grid.v1";

fn vision() -> Declaration {
    parse(VISION, &Environment::new("qwen38")).unwrap()
}

fn modality() -> Value {
    let frontend: Value =
        serde_json::from_str(include_str!("../../compiler/src/frontend-multimodal.json")).unwrap();
    frontend["multimodal"]["modalities"][0].clone()
}

/// The host's recipe and limits for one image part
/// (runtime-host/src/generation.rs `preprocess_media_part_v1`).
fn recipe() -> (ImageRecipe, MediaLimits) {
    let modality = modality();
    let recipe = runtime_host::image_recipe_v1(&modality["recipe"]["parameters"]).unwrap();
    let limits = MediaLimits {
        maximum_bytes: 64 << 20,
        maximum_pixels: 89_478_485,
        maximum_audio_frames: 0,
        maximum_frames: 32,
        maximum_tokens: modality["maximum_tokens"].as_u64().unwrap(),
        maximum_aspect_ratio: 200.0,
    };
    (recipe, limits)
}

/// Source sides around every factor-32 and pixel-bound edge, both ways.
fn sides() -> Vec<u32> {
    let mut sides: Vec<u32> = (1..=40).collect();
    for edge in [64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        sides.extend([edge - 1, edge, edge + 1]);
    }
    sides.extend([100, 333, 1000, 1500, 3000, 4000, 5000, 6000, 9000]);
    sides
}

#[test]
fn resized_images_take_the_declared_rows() {
    let vision = vision();
    let (recipe, limits) = recipe();
    let mut admitted = 0;
    let mut largest = 0;
    for &height in &sides() {
        for &width in &sides() {
            let Ok((h, w)) = smart_resize(height, width, recipe, limits) else {
                continue;
            };
            let grid = [1, u64::from(h / recipe.patch), u64::from(w / recipe.patch)];
            let geometry = vision
                .geometry(TAG, Media::Grid(grid))
                .unwrap_or_else(|error| panic!("{height}x{width} -> {h}x{w}: {error}"));
            let tokens = u64::from(h / 32) * u64::from(w / 32);
            assert_eq!(geometry.rows.last().unwrap().rows, tokens);
            assert!(tokens <= limits.maximum_tokens);
            largest = largest.max(geometry.input.rows);
            admitted += 1;
        }
    }
    assert!(admitted > 3000, "{admitted}");
    // The host admits prepared images that are their own resize, every
    // image the recipe emits under the bounds it checks at load.
    runtime_media::recipe::check_emits_own_resizes(recipe, limits).unwrap();
    assert_eq!(largest, u64::from(vision.input.capacity_rows));
}

#[test]
fn image_grids_take_the_declared_rows() {
    let vision = vision();
    let (recipe, limits) = recipe();
    for [width, height] in [
        [1, 1],
        [17, 300],
        [300, 17],
        [64, 64],
        [100, 60],
        [257, 255],
    ] {
        let pixels = (0..width * height * 3).map(|i| i as u8).collect();
        let image = runtime_media::RgbImage::new(width, height, pixels, limits).unwrap();
        let grid = image_grid(&image, recipe, limits).unwrap();
        let [t, h, w] = grid.grid.map(u64::from);
        assert_eq!(t, 1);
        let geometry = vision.geometry(TAG, Media::Grid([1, h, w])).unwrap();
        assert_eq!(
            geometry.input.rows * grid.columns as u64,
            grid.rows.len() as u64
        );
        assert_eq!(geometry.rows.last().unwrap().rows, grid.tokens);
    }
}

/// The engine's `grid-3axis.v1` positions (runtime-engine/src/positions.rs)
/// of a prompt of text runs and images, against the declaration's
/// placement walked over the same prompt.
#[test]
fn image_spans_take_the_engines_positions() {
    use encoder_spec::placement::Piece;
    use runtime_engine::{PositionAxesV1, PositionSpanV1};

    let vision = vision();
    let prompts: [&[(u64, [u64; 2])]; 4] = [
        &[(3, [8, 12])],
        &[(0, [2, 2]), (5, [64, 2]), (1, [2, 64])],
        &[(7, [256, 256]), (0, [6, 4])],
        &[(2, [2, 32_768]), (4, [32_768, 2])],
    ];
    for prompt in prompts {
        let (mut rows, mut running) = (0_u64, 0_u64);
        let mut expected = Vec::new();
        let mut spans = Vec::new();
        for &(text, [h, w]) in prompt {
            expected.extend((running..running + text).map(|p| [p; 3]));
            rows += text;
            running += text;
            let geometry = vision.geometry(TAG, Media::Grid([1, h, w])).unwrap();
            let placement = vision.place(&geometry, running).unwrap();
            let encoded = geometry.rows[geometry.rows.len() - 2];
            // The host's position grid: the record's [t, h / merge, w / merge]
            // (runtime-host/src/generation.rs `position_grid_v1`).
            assert_eq!(encoded.grid, Some([1, h / 2, w / 2]));
            assert_eq!(
                placement.pieces,
                [Piece::Rows {
                    media_row: 0,
                    rows: encoded.rows
                }]
            );
            spans.push(PositionSpanV1 {
                prompt_row: rows as u32,
                rows: encoded.rows as u32,
                grid: [1, (h / 2) as u32, (w / 2) as u32],
            });
            expected.extend(placement.positions.iter().copied());
            rows += encoded.rows;
            running = placement.next;
        }
        expected.extend((running..running + 5).map(|p| [p; 3]));
        rows += 5;
        let axes = PositionAxesV1::new(rows as usize, rows as usize, &spans).unwrap();
        let engine: Vec<[u64; 3]> = axes
            .rows(0, rows as usize)
            .into_iter()
            .map(|position| position.map(|axis| axis as u64))
            .collect();
        assert_eq!(engine, expected, "{prompt:?}");
    }
}

/// The host loads the frontend's image recipe only if every size it emits
/// is its own resize.
#[test]
fn the_host_accepts_the_frontends_recipes() {
    let mut root: Value =
        serde_json::from_str(include_str!("../../compiler/src/frontend.json")).unwrap();
    let section: Value =
        serde_json::from_str(include_str!("../../compiler/src/frontend-multimodal.json")).unwrap();
    root["multimodal"] = section["multimodal"].clone();
    root["format"]["media"] = section["media"].clone();
    let bytes = serde_json::to_vec(&root).unwrap();
    let descriptor =
        runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(&bytes).unwrap();
    runtime_host::validate_media_recipes_v1(&descriptor).unwrap();
}
