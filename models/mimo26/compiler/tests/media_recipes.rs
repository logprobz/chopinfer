use runtime_host::{audio_recipe_v1, image_recipe_v1};
use runtime_media::{MediaLimits, RgbImage, recipe};

fn parameters(kind: &str) -> serde_json::Value {
    let descriptor: serde_json::Value =
        serde_json::from_slice(model_mimo26::frontend::COMPONENT).unwrap();
    descriptor["multimodal"]["modalities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|modality| modality["kind"] == kind)
        .unwrap()["recipe"]["parameters"]
        .clone()
}

#[test]
fn declared_image_recipe_runs_with_raw_rgb_scale() {
    let recipe = image_recipe_v1(&parameters("image")).expect("host accepts declared image recipe");
    let limits = MediaLimits {
        maximum_bytes: 1 << 20,
        maximum_pixels: 1 << 20,
        maximum_audio_frames: 0,
        maximum_frames: 2,
        maximum_tokens: 1024,
        maximum_aspect_ratio: 100.0,
    };
    let rgb = [0, 0, 255].repeat(256 * 256);
    let image = RgbImage::new(256, 256, rgb, limits).unwrap();
    let grid = recipe::image_grid(&image, recipe, limits).unwrap();
    assert_eq!(grid.grid, [1, 16, 16]);
    assert_eq!(grid.tokens, 64);
    let expected = [
        -123.675 / 58.395,
        -116.28 / 57.12,
        (255.0 - 103.53) / 57.375,
    ];
    for row in grid.rows.chunks_exact(1536) {
        for (channel, expected) in row.chunks_exact(512).zip(expected) {
            assert!(channel.iter().all(|&value| (value - expected).abs() < 1e-5));
        }
    }
}

/// Limits of the deployed request path: decoded images up to 2^24 pixels,
/// resized into the recipe's budget.
fn served_limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 64 << 20,
        maximum_pixels: 1 << 24,
        maximum_audio_frames: 0,
        maximum_frames: 2,
        maximum_tokens: 1024,
        maximum_aspect_ratio: 200.0,
    }
}

/// Solid images of any size reach the tower as identical rows of the color's
/// normalized value, in R, G, B channel order. The grids follow the
/// checkpoint processor with the declared pixel bounds: identity,
/// a small image, a plain resize, a downscale past the
/// maximum budget and a narrow aspect ratio.
#[test]
fn solid_images_give_constant_rows_at_every_size() {
    let recipe = image_recipe_v1(&parameters("image")).unwrap();
    let colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
        [0, 0, 0],
        [255, 255, 0],
    ];
    let sizes = [
        ((256, 256), [1, 16, 16]),
        ((64, 64), [1, 6, 6]),
        ((1000, 750), [1, 46, 62]),
        ((2048, 1536), [1, 54, 72]),
        ((40, 600), [1, 38, 2]),
    ];
    for color in colors {
        let expected: [f32; 3] =
            std::array::from_fn(|c| (f32::from(color[c]) - recipe.mean[c]) / recipe.std[c]);
        for ((width, height), expected_grid) in sizes {
            let rgb = color.repeat(width as usize * height as usize);
            let image = RgbImage::new(width, height, rgb, served_limits()).unwrap();
            let grid = recipe::image_grid(&image, recipe, served_limits()).unwrap();
            assert_eq!(grid.grid, expected_grid, "{color:?} {width}x{height}");
            let first = grid.rows[..1536].to_vec();
            for (channel, expected) in first.chunks_exact(512).zip(expected) {
                assert!(
                    channel
                        .iter()
                        .all(|&value| (value - expected).abs() <= 1e-5),
                    "{color:?} {width}x{height}: {channel:?} vs {expected}"
                );
                assert!(channel.iter().all(|&value| value == channel[0]));
            }
            assert!(grid.rows.chunks_exact(1536).all(|row| row == first));
        }
    }
}

/// A two-color split fixes the (h-block, w-block, merge-h, merge-w) row order
/// and the (C, T, ph, pw) column order without a capture, for square and
/// non-square grids. The edge sits 16 pixels past the middle, inside a 2x2
/// merge block, so swapped merge axes show as well as swapped image axes.
#[test]
fn split_images_keep_row_and_channel_order() {
    let recipe = image_recipe_v1(&parameters("image")).unwrap();
    let red = [255_u8, 0, 0];
    let blue = [0_u8, 0, 255];
    for (width, height, vertical) in [(256, 256, true), (256, 256, false), (320, 192, true)] {
        let pixel = |x: u32, y: u32| {
            let first = if vertical {
                x < width / 2 + 16
            } else {
                y < height / 2 + 16
            };
            if first { red } else { blue }
        };
        let rgb = (0..height)
            .flat_map(|y| (0..width).flat_map(move |x| pixel(x, y)))
            .collect::<Vec<_>>();
        let image = RgbImage::new(width, height, rgb, served_limits()).unwrap();
        let grid = recipe::image_grid(&image, recipe, served_limits()).unwrap();
        let grid_w = grid.grid[2] as usize;
        assert_eq!(grid.grid, [1, height / 16, width / 16]);
        for (index, row) in grid.rows.chunks_exact(1536).enumerate() {
            let (block, merged) = (index / 4, index % 4);
            let y = ((block / (grid_w / 2)) * 2 + merged / 2) * 16;
            let x = ((block % (grid_w / 2)) * 2 + merged % 2) * 16;
            for (c, plane) in row.chunks_exact(512).enumerate() {
                for (t, patch) in plane.chunks_exact(256).enumerate() {
                    for (offset, &value) in patch.iter().enumerate() {
                        let color = pixel((x + offset % 16) as u32, (y + offset / 16) as u32);
                        let expected = (f32::from(color[c]) - recipe.mean[c]) / recipe.std[c];
                        assert!(
                            (value - expected).abs() <= 1e-5,
                            "{width}x{height} row {index} channel {c} frame {t} offset {offset}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn declared_audio_recipe_has_the_checkpoint_frame_rule() {
    let recipe = audio_recipe_v1(&parameters("audio")).expect("host accepts declared audio recipe");
    assert_eq!(
        (
            recipe.sample_rate,
            recipe.n_fft,
            recipe.window,
            recipe.hop,
            recipe.mel_bins
        ),
        (24000, 960, 960, 240, 128)
    );
    assert_eq!(
        runtime_media::audio::strided_group_tokens(1595, recipe).unwrap(),
        100
    );
    assert_eq!(
        runtime_media::audio::strided_group_tokens(6001, recipe).unwrap(),
        376
    );
}

#[test]
fn declared_video_recipe_contains_the_host_image_recipe() {
    let parameters = parameters("video");
    let image = image_recipe_v1(&parameters["image"]).expect("host accepts declared video recipe");
    assert_eq!(image.maximum_pixels, 262144);
    assert_eq!(parameters["total_max_pixels"].as_u64(), Some(268435456));
    assert_eq!(
        (image.factor, image.patch, image.temporal_patch, image.merge),
        (32, 16, 2, 2)
    );
}

fn frozen_limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 64 << 20,
        maximum_pixels: 1 << 20,
        maximum_audio_frames: 24_000 * 600,
        maximum_frames: 32,
        maximum_tokens: 3751,
        maximum_aspect_ratio: 100.0,
    }
}

/// Reads one array of the HF processor capture (regenerated 2026-10-12)
/// after checking its manifest digest. Returns the shape and the little-endian payload.
fn captured(name: &str, descr: &str) -> (Vec<usize>, Vec<u8>) {
    use sha2::Digest;
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../fixtures/mimo26/multimodal/processor");
    let bytes = std::fs::read(root.join(name)).expect("frozen processor capture");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("processor-manifest.json")).unwrap())
            .unwrap();
    let digest = sha2::Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    assert_eq!(
        manifest["files"][name].as_str(),
        Some(digest.as_str()),
        "{name} digest"
    );
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let start = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let header = std::str::from_utf8(&bytes[10..start]).unwrap();
    assert!(
        header.contains(&format!("'descr': '{descr}'")),
        "{name}: {header}"
    );
    assert!(header.contains("'fortran_order': False"));
    let shape = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let shape = shape
        .split(',')
        .map(str::trim)
        .filter(|dimension| !dimension.is_empty())
        .map(|dimension| dimension.parse().unwrap())
        .collect();
    (shape, bytes[start..].to_vec())
}

fn captured_f32(name: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, bytes) = captured(name, "<f4");
    let values = bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    (shape, values)
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Patch rows: max abs error 1e-5 before rounding, 99.9% identical BF16 rows.
fn assert_rows_match(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name} shape");
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let identical = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| bf16(**a) == bf16(**b))
        .count() as f64
        / actual.len() as f64;
    assert!(maximum <= 1e-5, "{name}: maximum absolute error {maximum}");
    assert!(
        identical >= 0.999,
        "{name}: BF16 identical ratio {identical}"
    );
}

#[test]
fn frozen_processor_image_rows_match_host_recipe() {
    let recipe = image_recipe_v1(&parameters("image")).unwrap();
    for name in ["blue-square", "frames-00", "frames-04"] {
        let (shape, rgb) = captured(&format!("{name}-rgb.npy"), "|u1");
        assert_eq!(shape, [256, 256, 3]);
        let image = RgbImage::new(256, 256, rgb, frozen_limits()).unwrap();
        let grid = recipe::image_grid(&image, recipe, frozen_limits()).unwrap();
        let (grid_shape, expected_grid) = captured(&format!("{name}-grid.npy"), "<i4");
        assert_eq!(grid_shape, [1, 3]);
        let expected_grid = expected_grid
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(
            grid.grid.as_slice(),
            expected_grid.as_slice(),
            "{name} grid"
        );
        let (_, expected) = captured_f32(&format!("{name}-pixels.npy"));
        assert_rows_match(name, &grid.rows, &expected);
    }
}

#[test]
fn frozen_processor_video_rows_match_host_recipe() {
    let parameters = parameters("video");
    let (shape, rgb) = captured("video-rgb.npy", "|u1");
    assert_eq!(shape, [8, 256, 256, 3]);
    let (_, labels) = captured_f32("video-frame-timestamps.npy");
    let frames = rgb
        .chunks_exact(256 * 256 * 3)
        .zip(labels)
        .map(|(frame, label)| {
            let image = RgbImage::new(256, 256, frame.to_vec(), frozen_limits()).unwrap();
            (f64::from(label), image)
        })
        .collect::<Vec<_>>();
    let grids = recipe::video_grids(
        &frames,
        image_recipe_v1(&parameters["image"]).unwrap(),
        parameters["total_max_pixels"].as_u64().unwrap(),
        frozen_limits(),
    )
    .unwrap();
    // The processor reports one [4, 16, 16] grid at 2 s per temporal patch.
    let (_, seconds) = captured_f32("video-seconds-per-grid.npy");
    assert_eq!(seconds, [2.0]);
    assert_eq!(
        grids
            .iter()
            .map(|grid| grid.timestamp.as_str())
            .collect::<Vec<_>>(),
        ["00:00", "00:02", "00:04", "00:06"]
    );
    assert!(grids.iter().all(|grid| grid.grid.grid == [1, 16, 16]));
    let rows = grids
        .into_iter()
        .flat_map(|grid| grid.grid.rows)
        .collect::<Vec<_>>();
    let (_, expected) = captured_f32("video-pixels.npy");
    assert_rows_match("video", &rows, &expected);
}

/// PNG decoding reproduces the captured decoder output bit for bit.
/// Needs the functional media in CHOPINFER_MIMO_FIXTURES; skipped otherwise.
#[test]
fn functional_media_decode_matches_capture() {
    let Some(root) = std::env::var_os("CHOPINFER_MIMO_FIXTURES") else {
        eprintln!("skipped: CHOPINFER_MIMO_FIXTURES is unset");
        return;
    };
    let root = std::path::PathBuf::from(root).join("functional");
    for (file, name) in [
        ("blue-square.png", "blue-square"),
        ("frames/00.png", "frames-00"),
        ("frames/04.png", "frames-04"),
    ] {
        let image =
            runtime_media::png::decode(&std::fs::read(root.join(file)).unwrap(), frozen_limits())
                .unwrap();
        assert_eq!(
            image.data,
            captured(&format!("{name}-rgb.npy"), "|u1").1,
            "{file}"
        );
    }
}

/// Upstream AutoImageProcessor integer resize and patch order. The queued
/// reference records only hashes here; RGB probes are regenerated exactly.
#[test]
fn checkpoint_processor_integer_rows_match_declared_recipe() {
    use sha2::{Digest, Sha256};
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../fixtures/mimo26/image-processor.json"
    ))
    .unwrap();
    let recipe = image_recipe_v1(&parameters("image")).unwrap();
    let raw_recipe = recipe::ImageRecipe {
        value_scale: 1.0,
        mean: [0.0; 3],
        std: [1.0; 3],
        ..recipe
    };
    let limits = MediaLimits {
        maximum_pixels: 32 << 20,
        ..served_limits()
    };
    for case in reference["images"].as_array().unwrap() {
        let width = case["width"].as_u64().unwrap() as u32;
        let height = case["height"].as_u64().unwrap() as u32;
        let mut rgb = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.extend([
                    (x * 255 / (width - 1).max(1)) as u8,
                    (y * 255 / (height - 1).max(1)) as u8,
                    if (x * 7 + y * 3) % 97 < 48 { 255 } else { 0 },
                ]);
            }
        }
        let image = RgbImage::new(width, height, rgb, limits).unwrap();
        let got = recipe::image_grid(&image, raw_recipe, limits).unwrap();
        let grid: Vec<u32> = serde_json::from_value(case["grid"].clone()).unwrap();
        assert_eq!(got.grid.as_slice(), grid.as_slice(), "{width}x{height}");
        let mut digest = Sha256::new();
        for value in got.rows {
            assert!(
                value.fract() == 0.0 && (0.0..=255.0).contains(&value),
                "noninteger resize value {value}"
            );
            digest.update([value as u8]);
        }
        let digest = format!("{:x}", digest.finalize());
        assert_eq!(
            digest,
            case["integer_patch_sha256"].as_str().unwrap(),
            "{width}x{height}"
        );
    }
}
