//! A model's image decode capture against the served decode, for the model's
//! own tests.
//!
//! runtime-media's image decode corpus (`tests/fixtures/image-decode/` and
//! `image-decode.json`, written by `tools/refrun/image_decode.py`) records
//! what transformers' `load_image` decodes from each file: orientation from
//! EXIF and XMP applied, then alpha discarded. A model's capture
//! (`model_capture` in that producer) repeats the `load_image` run where
//! the model's image processor is installed and writes
//! `fixtures/<m>/image-decode.json`. [`check_image_decode_v1`] requires the
//! model's recipe to serve that policy, the capture to agree with the
//! corpus, and every file to decode as the capture states, or as the
//! corpus's declared divergence serves it.

use std::path::Path;

use runtime_frontend::FrontendMediaRecipeV1;
use runtime_media::{Alpha, ImagePolicy, MediaLimits, Orientation, decode_image, probe_image};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// runtime-media's test fixtures, which hold the corpus.
const FIXTURES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../runtime-media/tests/fixtures"
);

/// Room for every corpus file; served limits are the declaration's, which
/// decide admission rather than pixels.
const LIMITS: MediaLimits = MediaLimits {
    maximum_bytes: 16 << 20,
    maximum_pixels: 1 << 20,
    maximum_audio_frames: 0,
    maximum_frames: 1,
    maximum_tokens: 0,
    maximum_aspect_ratio: 200.0,
};

fn read_json(path: &Path) -> Value {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// The served decode's size and RGB digest, or why it was refused.
fn served(bytes: &[u8], policy: ImagePolicy) -> Result<(u64, u64, String), String> {
    let image = decode_image(bytes, LIMITS, policy).map_err(|error| error.to_string())?;
    let (_, width, height) =
        probe_image(bytes, LIMITS, policy.orientation).map_err(|error| error.to_string())?;
    if (width, height) != (image.width, image.height) {
        return Err(format!(
            "probe gave {width} x {height}, decode {} x {}",
            image.width, image.height
        ));
    }
    Ok((
        u64::from(image.width),
        u64::from(image.height),
        sha256(&image.data),
    ))
}

/// Checks the capture at `capture` against what `recipe` serves, panicking
/// with every difference; returns the number of corpus files checked.
///
/// # Panics
///
/// When the recipe is not implemented or does not serve `load_image`'s
/// policy, when the capture differs from the corpus, or when a file
/// decodes otherwise than captured.
#[must_use]
pub fn check_image_decode_v1(recipe: &FrontendMediaRecipeV1, capture: &Path) -> usize {
    let policy =
        crate::image_policy_v1(recipe).unwrap_or_else(|error| panic!("{}: {error:?}", recipe.tag));
    assert_eq!(
        policy,
        ImagePolicy {
            alpha: Alpha::Discard,
            orientation: Orientation::Exif,
        },
        "{} does not serve load_image's policy",
        recipe.tag
    );
    let fixtures = Path::new(FIXTURES);
    let corpus = read_json(&fixtures.join("image-decode.json"));
    let captured = read_json(capture);
    let (cases, captured) = (
        corpus["cases"].as_array().expect("corpus cases"),
        captured["cases"].as_array().expect("captured cases"),
    );
    assert_eq!(cases.len(), captured.len(), "the capture covers the corpus");
    let mut mismatches = Vec::new();
    for (case, captured) in cases.iter().zip(captured) {
        let name = case["name"].as_str().expect("case name");
        let reference = &case["load_image"];
        let mut plain = reference.clone();
        if let Some(fields) = plain.as_object_mut() {
            fields.remove("runtime");
        }
        if captured["name"] != case["name"]
            || captured["file_sha256"] != case["file_sha256"]
            || captured["load_image"] != plain
        {
            mismatches.push(format!("{name}: captured {captured}, corpus {plain}"));
            continue;
        }
        let path = fixtures.join(case["file"].as_str().expect("case file"));
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
        assert_eq!(sha256(&bytes), case["file_sha256"], "{name}");
        let expected = reference.get("runtime").unwrap_or(reference);
        let outcome = served(&bytes, policy);
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
            mismatches.push(format!("{name}: expected {expected}, served {outcome:?}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} files differ:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.join("\n")
    );
    cases.len()
}
