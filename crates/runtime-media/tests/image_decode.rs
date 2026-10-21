//! The decoders against the image decode corpus, `fixtures/image-decode.json`
//! (written by `tools/refrun/image_decode.py`): PNG, JPEG, GIF and WebP
//! files covering color types, transparency, first-frame selection and every
//! place Pillow 12.3 reads an orientation from. Each file decodes bit for bit
//! as two reference loaders do: `load_image` (transformers' loader:
//! `ImageOps.exif_transpose`, then `convert("RGB")`) under
//! [`Orientation::Exif`], and `open_convert` (`Image.open`, then
//! `convert("RGB")`) under [`Orientation::Stored`], both with alpha
//! discarded. A case that deliberately differs names its `divergence` and
//! the served outcome (`runtime`): a refusal, or the stored pixels.
use std::path::Path;

use runtime_media::{
    Alpha, ImagePolicy, MediaLimits, Orientation, decode_image, decode_working_bytes, probe_image,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixtures() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"))
}

fn golden() -> Value {
    let text = std::fs::read_to_string(fixtures().join("image-decode.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 16 << 20,
        maximum_pixels: 1 << 20,
        maximum_audio_frames: 0,
        maximum_frames: 1,
        maximum_tokens: 0,
        maximum_aspect_ratio: 200.0,
    }
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

const LOADERS: [(&str, Orientation); 2] = [
    ("load_image", Orientation::Exif),
    ("open_convert", Orientation::Stored),
];

/// What the served decode of `bytes` under `orientation` gave, in the
/// golden's terms.
fn served(bytes: &[u8], orientation: Orientation) -> Result<(u64, u64, String), String> {
    let policy = ImagePolicy {
        alpha: Alpha::Discard,
        orientation,
    };
    let image = decode_image(bytes, limits(), policy).map_err(|error| error.to_string())?;
    let (_, width, height) =
        probe_image(bytes, limits(), orientation).map_err(|error| format!("probe: {error}"))?;
    if (width, height) != (image.width, image.height) {
        return Err(format!(
            "probe gave {width} x {height}, decode {} x {}",
            image.width, image.height
        ));
    }
    decode_working_bytes(bytes, limits(), orientation)
        .map_err(|error| format!("working bytes: {error}"))?;
    Ok((
        u64::from(image.width),
        u64::from(image.height),
        sha256(&image.data),
    ))
}

#[test]
fn every_file_decodes_as_its_reference_loaders_do() {
    let golden = golden();
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() >= 100, "{} cases", cases.len());
    let mut mismatches = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let bytes = std::fs::read(fixtures().join(case["file"].as_str().unwrap())).unwrap();
        assert_eq!(
            sha256(&bytes),
            case["file_sha256"].as_str().unwrap(),
            "{name}"
        );
        let declared = LOADERS
            .iter()
            .any(|(loader, _)| case[*loader].get("runtime").is_some());
        assert_eq!(
            declared,
            case.get("divergence").is_some(),
            "{name}: a served outcome needs its divergence and the reverse"
        );
        for (loader, orientation) in LOADERS {
            let reference = &case[loader];
            let expected = reference.get("runtime").unwrap_or(reference);
            let outcome = served(&bytes, orientation);
            let matches = match (expected, &outcome) {
                (Value::String(refused), Err(_)) => refused == "refused",
                (expected, Err(_)) => expected.get("refused").is_some(),
                (expected, Ok((width, height, digest))) => {
                    expected["width"] == *width
                        && expected["height"] == *height
                        && expected["rgb_sha256"] == digest.as_str()
                }
            };
            if !matches {
                mismatches.push(format!(
                    "{name} [{loader}]: expected {expected}, served {outcome:?}"
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} decodes differ:\n{}",
        mismatches.len(),
        2 * cases.len(),
        mismatches.join("\n")
    );
}
