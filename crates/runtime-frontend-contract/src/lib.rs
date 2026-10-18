#![forbid(unsafe_code)]

mod centered_pad;
mod descriptor;
mod tokenizer;

pub use centered_pad::MAXIMUM_DECODED_IMAGE_PIXELS_V1;
pub use descriptor::*;
pub use tokenizer::*;

/// Fixed parameters implemented by the host's v1 audio recipe.
pub fn audio_log_mel_v1_fixed_parameters() -> serde_json::Value {
    serde_json::json!({
        "sample_rate": AUDIO_SAMPLE_RATE_V1,
        "maximum_seconds": AUDIO_MAXIMUM_SECONDS_V1,
        "channels": 1, "resampler": "sinc-hann", "window_kind": "periodic-hann",
        "power": 1, "mel_scale": "htk", "mel_norm": null, "center": true,
        "pad_mode": "reflect", "segment_frames": 6000
    })
}

pub const AUDIO_SAMPLE_RATE_V1: u64 = 24_000;
pub const AUDIO_MAXIMUM_SECONDS_V1: u64 = 600;
