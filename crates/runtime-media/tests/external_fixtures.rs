use std::path::PathBuf;

use runtime_media::{MediaLimits, png, recipe};

fn fixture(name: &str) -> Vec<u8> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_MEDIA_FIXTURE_ROOT")
            .expect("set CHOPINFER_MEDIA_FIXTURE_ROOT for external fixtures"),
    );
    std::fs::read(root.join(name)).expect("functional media fixture")
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 2 << 20,
        maximum_pixels: 1 << 20,
        maximum_audio_frames: 400_000,
        maximum_frames: 32,
        maximum_tokens: 1024,
        maximum_aspect_ratio: 200.0,
    }
}

#[test]
#[ignore = "requires external functional media fixtures"]
fn blue_square_decodes_as_rgb() {
    let image = png::decode(&fixture("blue-square.png"), limits()).unwrap();
    assert_eq!((image.width, image.height), (256, 256));
    assert!(image.data.chunks_exact(3).any(|p| p[2] > p[0]));
}

#[test]
#[ignore = "requires external functional media fixtures"]
fn blue_square_has_expected_patch_grid() {
    let image = png::decode(&fixture("blue-square.png"), limits()).unwrap();
    let recipe = recipe::ImageRecipe {
        factor: 32,
        minimum_pixels: 8192,
        maximum_pixels: 1_048_576,
        upscale_small_side: true,
        interpolation: recipe::Interpolation::Bilinear,
        patch: 16,
        temporal_patch: 2,
        merge: 2,
        value_scale: 1.0,
        mean: [123.675, 116.28, 103.53],
        std: [58.395, 57.12, 57.375],
    };
    let grid = recipe::image_grid(&image, recipe, limits()).unwrap();
    assert_eq!(grid.grid, [1, 16, 16]);
    assert_eq!(grid.tokens, 64);
    assert_eq!(grid.rows.len(), 256 * 1536);
    assert_eq!(grid.bf16_bytes().len(), grid.rows.len() * 2);
}
