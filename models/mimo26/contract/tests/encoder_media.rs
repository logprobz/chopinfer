//! MiMo's encoder declarations against the host's media path: the image and
//! video grids runtime-media's resize emits for the frontend's recipes are
//! what the vision declaration admits, its derived rows are the host's
//! tokens and its placement is the host's, and the audio declaration's rows
//! are the host's audio token rule at every frame count.

use encoder_spec::components::Item;
use encoder_spec::geometry::Media;
use encoder_spec::placement::Piece;
use encoder_spec::{Declaration, parse};
use model_mimo26_contract::encoders::spec;
use runtime_media::MediaLimits;
use runtime_media::recipe::{ImageRecipe, image_grid, smart_resize, video_grids};
use serde_json::Value;

fn vision() -> Declaration {
    parse(spec::VISION, &spec::environment().unwrap()).unwrap()
}

fn audio() -> Declaration {
    parse(spec::AUDIO, &spec::environment().unwrap()).unwrap()
}

fn modality(kind: &str) -> Value {
    let frontend: Value =
        serde_json::from_str(include_str!("../../compiler/src/frontend.json")).unwrap();
    frontend["multimodal"]["modalities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|modality| modality["kind"] == kind)
        .unwrap()
        .clone()
}

/// The host's limits for one part (runtime-host/src/generation.rs
/// `preprocess_media_part_v1` and `preprocess_video_part_v1`).
fn limits(modality: &Value, maximum_pixels: u64) -> MediaLimits {
    MediaLimits {
        maximum_bytes: 64 << 20,
        maximum_pixels,
        maximum_audio_frames: 24_000 * 600,
        maximum_frames: 32,
        maximum_tokens: modality["maximum_tokens"].as_u64().unwrap(),
        maximum_aspect_ratio: 200.0,
    }
}

fn image_recipe() -> (ImageRecipe, MediaLimits) {
    let image = modality("image");
    let recipe = runtime_host::image_recipe_v1(&image["recipe"]["parameters"]).unwrap();
    (recipe, limits(&image, 89_478_485))
}

fn sides() -> Vec<u32> {
    let mut sides: Vec<u32> = (1..=40).collect();
    for edge in [64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        sides.extend([edge - 1, edge, edge + 1]);
    }
    sides.extend([100, 333, 1000, 1500, 3000, 6000, 9000]);
    sides
}

fn final_rows(declaration: &Declaration, tag: &str, grid: [u64; 3]) -> u64 {
    let geometry = declaration
        .geometry(tag, Media::Grid(grid))
        .unwrap_or_else(|error| panic!("{tag} {grid:?}: {error}"));
    geometry.rows.last().unwrap().rows
}

#[test]
fn resized_images_take_the_declared_rows() {
    let vision = vision();
    let (recipe, limits) = image_recipe();
    let mut admitted = 0;
    for &height in &sides() {
        for &width in &sides() {
            let Ok((h, w)) = smart_resize(height, width, recipe, limits) else {
                continue;
            };
            let grid = [1, u64::from(h / 16), u64::from(w / 16)];
            let tokens = u64::from(h / 32) * u64::from(w / 32);
            assert_eq!(final_rows(&vision, "image.patch-grid.v1", grid), tokens);
            admitted += 1;
        }
    }
    assert!(admitted > 3000, "{admitted}");
    // The host admits prepared images that are their own resize, every
    // image the recipe emits under the bounds it checks at load.
    runtime_media::recipe::check_emits_own_resizes(recipe, limits).unwrap();
    for [width, height] in [[1, 1], [17, 300], [300, 17], [64, 64], [1100, 900]] {
        let pixels = vec![7; (width * height * 3) as usize];
        let image = runtime_media::RgbImage::new(width, height, pixels, limits).unwrap();
        let grid = image_grid(&image, recipe, limits).unwrap();
        let rows = final_rows(&vision, "image.patch-grid.v1", grid.grid.map(u64::from));
        assert_eq!(rows, grid.tokens);
    }
}

#[test]
fn video_grids_take_the_declared_rows() {
    let vision = vision();
    let video = modality("video");
    let parameters = &video["recipe"]["parameters"];
    let recipe = runtime_host::image_recipe_v1(&parameters["image"]).unwrap();
    let limits = limits(&video, 1 << 24);
    let total = parameters["total_max_pixels"].as_u64().unwrap();
    for frames in 1..=32 {
        let frame = runtime_media::recipe::video_frame_recipe(recipe, total, frames);
        runtime_media::recipe::check_emits_own_resizes(frame, limits)
            .unwrap_or_else(|error| panic!("{frames} frames: {error}"));
    }
    for (frames, [width, height]) in [
        (1, [40, 30]),
        (2, [40, 30]),
        (3, [33, 1000]),
        (17, [1000, 33]),
        (31, [64, 48]),
        (32, [64, 48]),
        (2, [640, 480]),
    ] {
        let image = runtime_media::RgbImage::new(
            width,
            height,
            vec![9; (width * height * 3) as usize],
            limits,
        )
        .unwrap();
        let frames: Vec<_> = (0..frames).map(|i| (f64::from(i), image.clone())).collect();
        let grids = video_grids(&frames, recipe, total, limits).unwrap();
        let [_, h, w] = grids[0].grid.grid;
        let t: u32 = grids.iter().map(|grid| grid.grid.grid[0]).sum();
        let tokens: u64 = grids.iter().map(|grid| grid.grid.tokens).sum();
        let grid = [t, h, w].map(u64::from);
        assert_eq!(
            final_rows(&vision, "video.timestamped-grids.v1", grid),
            tokens
        );
        // The host's grid render writes each temporal grid's rows after its
        // timestamp, at consecutive media rows
        // (runtime-host/src/generation.rs `expand_media_chat_v1`).
        let geometry = vision
            .geometry("video.timestamped-grids.v1", Media::Grid(grid))
            .unwrap();
        let placement = vision.place(&geometry, 11).unwrap();
        let mut media_row = 0;
        let expected: Vec<Piece> = grids
            .iter()
            .map(|grid| {
                let piece = Piece::Rows {
                    media_row,
                    rows: grid.grid.tokens,
                };
                media_row += grid.grid.tokens;
                piece
            })
            .collect();
        assert_eq!(placement.pieces, expected);
        assert!(
            placement
                .positions
                .iter()
                .copied()
                .eq((11..11 + tokens).map(|p| [p; 3]))
        );
        assert_eq!(placement.next, 11 + tokens);
    }
}

#[test]
fn images_and_audio_take_one_sequential_span() {
    let vision = vision();
    let audio = audio();
    for (declaration, tag, media) in [
        (&vision, "image.patch-grid.v1", Media::Grid([1, 6, 10])),
        (&audio, "audio.log-mel.v1", Media::Frames(6001)),
    ] {
        let geometry = declaration.geometry(tag, media).unwrap();
        let rows = geometry.rows.last().unwrap().rows;
        let placement = declaration.place(&geometry, 4).unwrap();
        assert_eq!(placement.pieces, [Piece::Rows { media_row: 0, rows }]);
        assert!(
            placement
                .positions
                .iter()
                .copied()
                .eq((4..4 + rows).map(|p| [p; 3]))
        );
        assert_eq!((placement.next, placement.media_mask), (4 + rows, false));
    }
}

#[test]
fn audio_rows_follow_the_host_token_rule() {
    let audio = audio();
    let sound = modality("audio");
    let recipe = runtime_host::audio_recipe_v1(&sound["recipe"]["parameters"]).unwrap();
    for frames in 1..=60_001 {
        let rows = audio
            .trace(Item {
                rows: frames,
                grid: None,
            })
            .unwrap()
            .last()
            .unwrap()
            .rows;
        let tokens = runtime_media::audio::strided_group_tokens(frames as usize, recipe).unwrap();
        assert_eq!(rows, tokens as u64, "{frames} frames");
    }
    for frames in [1, 6001, 60_001] {
        let geometry = audio
            .geometry("audio.log-mel.v1", Media::Frames(frames))
            .unwrap();
        assert_eq!(geometry.rows, audio.trace(geometry.input).unwrap());
    }
}

/// The host loads the frontend's media recipes only if every size each
/// emits, at every video frame budget, is its own resize.
#[test]
fn the_host_accepts_the_frontends_recipes() {
    let descriptor = runtime_frontend_contract::FrontendDescriptorV1::from_component_bytes(
        include_bytes!("../../compiler/src/frontend.json"),
    )
    .unwrap();
    runtime_host::validate_media_recipes_v1(&descriptor).unwrap();
}
