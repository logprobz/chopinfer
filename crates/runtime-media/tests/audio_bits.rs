use runtime_media::{DecodedAudio, MediaLimits, audio, wav};
use sha2::{Digest, Sha256};

fn recipe() -> audio::AudioRecipe {
    audio::AudioRecipe {
        sample_rate: 24_000,
        sinc_width: 6,
        rolloff: 0.99,
        n_fft: 960,
        window: 960,
        hop: 240,
        mel_bins: 128,
        f_min: 0.0,
        f_max: 12_000.0,
        log_floor: 1e-7,
        first_kernel: 3,
        second_kernel: 3,
        second_stride: 2,
        pool: 2,
        group: 4,
    }
}
fn limits() -> MediaLimits {
    MediaLimits {
        maximum_bytes: 64 << 20,
        maximum_pixels: 1,
        maximum_audio_frames: 24_000 * 600,
        maximum_frames: 1,
        maximum_tokens: 4000,
        maximum_aspect_ratio: 200.0,
    }
}
fn digest(audio: &DecodedAudio) -> String {
    let mel = audio::log_mel(audio, recipe(), limits()).unwrap();
    let mut hash = Sha256::new();
    for value in &mel.values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
#[test]
fn accepted_sample_rates_preserve_feature_bits() {
    // Captured from the original recursive FFT and dense mel implementation.
    for (rate, expected) in [
        (
            24_000,
            "1f5c12fddc671c6b064002c382d9895c193aa5f8a62755be879931de8464d0f0",
        ),
        (
            44_100,
            "245182dad57646180276352a83ac18a75c0b9116e1e9f7053365b297605e3def",
        ),
        (
            48_000,
            "b4e6982d3edaa8dc328b6fa42de97e3672139b551b00b2ef3ae9b8393b191d88",
        ),
    ] {
        let samples = (0..rate)
            .map(|i| ((i * 7919 % 65521) as f32 - 32760.0) / 65536.0)
            .collect();
        let audio = DecodedAudio {
            sample_rate: rate,
            samples,
        };
        assert_eq!(digest(&audio), expected, "{rate} Hz");
    }
}
/// A mono PCM16 WAV of `samples` at `rate`.
fn pcm16_wav(samples: &[i16], rate: u32) -> Vec<u8> {
    let data = u32::try_from(samples.len() * 2).unwrap();
    let mut out = b"RIFF".to_vec();
    out.extend((36 + data).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16_u32.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(1_u16.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * 2).to_le_bytes());
    out.extend(2_u16.to_le_bytes());
    out.extend(16_u16.to_le_bytes());
    out.extend(b"data");
    out.extend(data.to_le_bytes());
    for sample in samples {
        out.extend(sample.to_le_bytes());
    }
    out
}
/// A generated 24 kHz sweep of 382,752 frames: 1,595 mel frames and 100
/// audio tokens under the recipe.
fn generated_samples() -> Vec<i16> {
    (0..382_752)
        .map(|i| {
            let t = f64::from(i) / 24_000.0;
            (8000.0 * (std::f64::consts::TAU * (220.0 * t + 400.0 * t * t)).sin()) as i16
        })
        .collect()
}
#[test]
fn generated_wav_decodes_every_frame_exactly() {
    let samples = generated_samples();
    let audio = wav::decode(&pcm16_wav(&samples, 24_000), limits()).unwrap();
    assert_eq!(audio.sample_rate, 24_000);
    let expected: Vec<f32> = samples.iter().map(|&s| f32::from(s) / 32768.0).collect();
    assert_eq!(audio.samples, expected);
    assert_eq!(
        digest(&audio),
        digest(&DecodedAudio {
            sample_rate: 24_000,
            samples: expected,
        })
    );
}
#[test]
fn generated_wav_log_mel_has_expected_geometry() {
    let audio = wav::decode(&pcm16_wav(&generated_samples(), 24_000), limits()).unwrap();
    let mel = audio::log_mel(&audio, recipe(), limits()).unwrap();
    assert_eq!((mel.frames, mel.bins, mel.tokens), (1595, 128, 100));
    assert!(mel.values.iter().all(|v| v.is_finite()));
}
