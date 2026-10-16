//! The image recipe against the pinned vLLM image processor.
//!
//! `fixtures/qwen38/image-processor.json` holds the recipe parameters, the
//! grids and the SHA256 of the f32 pixel rows that transformers 5.15.1's
//! `Qwen2VLImageProcessor` (torchvision backend) produced for the probe
//! images in the pinned vLLM image. The probes are regenerated here. With
//! `CHOPINFER_DATA` set, the rows are also compared element-wise with the
//! captured `qwen38-refs/processor/<name>-fast-pixels.npy`.
use runtime_host::image_recipe_v1;
use runtime_media::{MediaLimits, RgbImage, recipe};
use sha2::{Digest, Sha256};

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/qwen38/image-processor.json"
    ))
    .unwrap()
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 1 << 26,
        maximum_pixels: 1 << 25,
        maximum_audio_frames: 0,
        maximum_frames: 2,
        maximum_tokens: 1 << 16,
        maximum_aspect_ratio: 200.0,
    }
}

/// `fixtures/qwen38/capture/images.py`: a horizontal red ramp, a vertical
/// green ramp and a diagonal blue stripe edge.
fn gradient(width: u32, height: u32) -> RgbImage {
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            data.extend([
                (x * 255 / (width - 1).max(1)) as u8,
                (y * 255 / (height - 1).max(1)) as u8,
                if (x * 7 + y * 3) % 97 < 48 { 255 } else { 0 },
            ]);
        }
    }
    RgbImage::new(width, height, data, limits()).unwrap()
}

/// The functional `blue-square.png`: white 256x256 with a blue square over
/// rows and columns 48 to 208.
fn blue_square() -> RgbImage {
    let mut data = Vec::with_capacity(256 * 256 * 3);
    for y in 0..256 {
        for x in 0..256 {
            let inside = (48..=208).contains(&x) && (48..=208).contains(&y);
            data.extend(if inside { [0, 0, 255] } else { [255; 3] });
        }
    }
    RgbImage::new(256, 256, data, limits()).unwrap()
}

fn probe(name: &str, width: u32, height: u32) -> RgbImage {
    if name == "blue-square" {
        blue_square()
    } else {
        gradient(width, height)
    }
}

/// Little-endian f32 `.npy` data, checked against the expected element count.
fn read_npy(path: &std::path::Path, count: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let (header_len, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        ),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
    assert!(header.contains("'<f4'") && header.contains("'fortran_order': False"));
    let data = &bytes[start + header_len..];
    assert_eq!(data.len(), count * 4);
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn check_probe(name: &str) {
    let fixture = fixture();
    let recipe = image_recipe_v1(&fixture["recipe"]).expect("host accepts the recipe");
    let expected = &fixture["images"][name];
    let size = |key: &str| expected[key].as_u64().unwrap() as u32;
    let image = probe(name, size("width"), size("height"));
    let grid = recipe::image_grid(&image, recipe, limits()).unwrap();
    let want: Vec<u32> = serde_json::from_value(expected["grid"].clone()).unwrap();
    assert_eq!(grid.grid.as_slice(), want.as_slice(), "{name} grid");
    assert_eq!(grid.columns, 1536);
    let [t, h, w] = grid.grid;
    assert_eq!(grid.rows.len(), (t * h * w) as usize * 1536);

    if let Some(data) = std::env::var_os("CHOPINFER_DATA") {
        let path = std::path::Path::new(&data)
            .join("qwen38-refs/processor")
            .join(format!("{name}-fast-pixels.npy"));
        if path.exists() {
            let reference = read_npy(&path, grid.rows.len());
            let max_abs = grid
                .rows
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let same_bf16 = grid
                .rows
                .iter()
                .zip(&reference)
                .filter(|&(&a, &b)| bf16(a) == bf16(b))
                .count();
            eprintln!(
                "{name}: max abs {max_abs:e}, BF16-identical {same_bf16} of {}",
                reference.len()
            );
            assert_eq!(max_abs, 0.0, "{name} differs from the captured pixels");
        } else {
            eprintln!("{name}: {} absent, skipping", path.display());
        }
    }

    let mut hash = Sha256::new();
    for value in &grid.rows {
        hash.update(value.to_le_bytes());
    }
    let digest: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        digest,
        expected["pixels_sha256"].as_str().unwrap(),
        "{name} pixels"
    );
}

#[test]
fn smart_resize_matches_the_reference() {
    let fixture = fixture();
    let recipe = image_recipe_v1(&fixture["recipe"]).unwrap();
    for case in fixture["smart_resize"].as_array().unwrap() {
        let [h, w, rh, rw] = serde_json::from_value::<[u32; 4]>(case.clone()).unwrap();
        assert_eq!(
            recipe::smart_resize(h, w, recipe, limits()).unwrap(),
            (rh, rw),
            "{h}x{w}"
        );
    }
}

#[test]
fn blue_square_matches_the_reference() {
    check_probe("blue-square");
}

#[test]
fn gradient_333x517_matches_the_reference() {
    check_probe("gradient-333x517");
}

#[test]
fn small_100x60_matches_the_reference() {
    check_probe("small-100x60");
}

#[test]
fn wide_1000x40_matches_the_reference() {
    check_probe("wide-1000x40");
}

/// 18 MP in, 16.6 MP out: seconds in release, minutes in debug.
#[test]
#[cfg_attr(debug_assertions, ignore = "slow without optimizations")]
fn large_4500x4000_matches_the_reference() {
    check_probe("large-4500x4000");
}
