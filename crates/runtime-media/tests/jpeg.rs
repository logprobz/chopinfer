use std::path::Path;

use runtime_media::{MediaLimits, RgbImage, jpeg};

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 32 << 20,
        maximum_pixels: 16 * 1024 * 1024,
        maximum_audio_frames: 0,
        maximum_frames: 0,
        maximum_tokens: 0,
        maximum_aspect_ratio: 200.0,
    }
}

#[test]
#[ignore = "requires CHOPINFER_JPEG_CORPUS_FILE"]
fn real_baseline_or_progressive_jpeg_decodes() {
    let path = std::env::var("CHOPINFER_JPEG_CORPUS_FILE")
        .expect("set CHOPINFER_JPEG_CORPUS_FILE to a baseline or progressive JPEG");
    let bytes = std::fs::read(path).unwrap();
    let geometry = jpeg::probe(&bytes, limits()).unwrap();
    let image = jpeg::decode(&bytes, limits()).unwrap();
    assert_eq!((image.width, image.height), geometry);
}

#[test]
#[ignore = "requires CHOPINFER_JPEG_CAPTURE_ROOT"]
fn decoded_rgb_is_exact_against_mm11_captures() {
    let root = std::env::var("CHOPINFER_JPEG_CAPTURE_ROOT")
        .expect("set CHOPINFER_JPEG_CAPTURE_ROOT to MM-11 JPEG and raw RGB captures");
    let root = Path::new(&root);
    let mut count = 0;
    for entry in std::fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if !matches!(
            path.extension().and_then(|value| value.to_str()),
            Some("jpg" | "jpeg")
        ) {
            continue;
        }
        let encoded = std::fs::read(&path).unwrap();
        let (width, height) = jpeg::probe(&encoded, limits()).unwrap();
        let expected = RgbImage::new(
            width,
            height,
            std::fs::read(path.with_extension("rgb")).unwrap(),
            limits(),
        )
        .unwrap();
        let parity = jpeg::compare_capture(&encoded, &expected, limits()).unwrap();
        assert_eq!(
            parity.mismatched_channels,
            0,
            "{}: {parity:?}",
            path.display()
        );
        count += 1;
    }
    assert!(count > 0, "capture directory has no JPEG files");
}
