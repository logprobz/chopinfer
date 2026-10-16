//! RIFF/WAVE PCM and IEEE-float decoding with channel-mean downmix.

use crate::{DecodedAudio, MediaError, MediaLimits, MediaResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WavInfo {
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: u64,
    format: u16,
    bits: u16,
    data_offset: usize,
    data_bytes: usize,
}

fn le16(bytes: &[u8]) -> u16 {
    u16::from_le_bytes(bytes.try_into().expect("two bytes"))
}
fn le32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().expect("four bytes"))
}

pub fn probe(bytes: &[u8]) -> MediaResult<WavInfo> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(MediaError::new("WAV RIFF header is invalid"));
    }
    let riff_end = (le32(&bytes[4..8]) as usize)
        .checked_add(8)
        .ok_or_else(|| MediaError::new("WAV RIFF size overflow"))?;
    if riff_end > bytes.len() || riff_end < 12 {
        return Err(MediaError::new("truncated WAV RIFF"));
    }
    let mut position = 12;
    let mut fmt = None;
    let mut data = None;
    while position + 8 <= riff_end {
        let size = le32(&bytes[position + 4..position + 8]) as usize;
        let start = position + 8;
        let end = start
            .checked_add(size)
            .ok_or_else(|| MediaError::new("WAV chunk size overflow"))?;
        if end > riff_end {
            return Err(MediaError::new("truncated WAV chunk"));
        }
        match &bytes[position..position + 4] {
            b"fmt " if fmt.is_none() => {
                if size < 16 {
                    return Err(MediaError::new("short WAV fmt chunk"));
                }
                fmt = Some(&bytes[start..end]);
            }
            b"data" if data.is_none() => data = Some((start, size)),
            _ => {}
        }
        position = end
            .checked_add(size & 1)
            .ok_or_else(|| MediaError::new("WAV chunk padding overflow"))?;
    }
    let fmt = fmt.ok_or_else(|| MediaError::new("WAV fmt chunk is missing"))?;
    let (data_offset, data_bytes) =
        data.ok_or_else(|| MediaError::new("WAV data chunk is missing"))?;
    let mut format = le16(&fmt[..2]);
    let channels = le16(&fmt[2..4]);
    let sample_rate = le32(&fmt[4..8]);
    let block_align = le16(&fmt[12..14]);
    let bits = le16(&fmt[14..16]);
    if format == 0xfffe {
        if fmt.len() < 40
            || le16(&fmt[16..18]) < 22
            || fmt[26..40] != [0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71]
        {
            return Err(MediaError::new("unsupported WAV extensible subtype"));
        }
        format = le16(&fmt[24..26]);
    }
    let valid = matches!((format, bits), (1, 8 | 16 | 24 | 32) | (3, 32 | 64));
    let expected_align = u32::from(channels) * u32::from(bits) / 8;
    if channels == 0
        || sample_rate == 0
        || !valid
        || u32::from(block_align) != expected_align
        || block_align == 0
        || data_bytes % usize::from(block_align) != 0
    {
        return Err(MediaError::new("unsupported or inconsistent WAV fmt"));
    }
    Ok(WavInfo {
        sample_rate,
        channels,
        frames: (data_bytes / usize::from(block_align)) as u64,
        format,
        bits,
        data_offset,
        data_bytes,
    })
}

pub fn decode(bytes: &[u8], limits: MediaLimits) -> MediaResult<DecodedAudio> {
    decode_cancellable(bytes, limits, &|| false)
}

pub fn decode_cancellable(
    bytes: &[u8],
    limits: MediaLimits,
    cancelled: &dyn Fn() -> bool,
) -> MediaResult<DecodedAudio> {
    if cancelled() {
        return Err(MediaError::new("audio preparation cancelled"));
    }
    limits.check_bytes(bytes.len())?;
    let info = probe(bytes)?;
    limits.check_audio(info.frames)?;
    let channels = usize::from(info.channels);
    let width = usize::from(info.bits / 8);
    let data = &bytes[info.data_offset..info.data_offset + info.data_bytes];
    let mut samples = Vec::with_capacity(info.frames as usize);
    for (index, frame) in data.chunks_exact(channels * width).enumerate() {
        if index % 1024 == 0 && cancelled() {
            return Err(MediaError::new("audio preparation cancelled"));
        }
        let mut sum = 0_f32;
        for item in frame.chunks_exact(width) {
            let sample = match (info.format, info.bits) {
                (1, 8) => (f32::from(item[0]) - 128.0) / 128.0,
                (1, 16) => f32::from(i16::from_le_bytes(item.try_into().expect("i16"))) / 32768.0,
                (1, 24) => {
                    let value = i32::from_le_bytes([
                        item[0],
                        item[1],
                        item[2],
                        if item[2] & 0x80 != 0 { 0xff } else { 0 },
                    ]);
                    value as f32 / 8_388_608.0
                }
                (1, 32) => {
                    i32::from_le_bytes(item.try_into().expect("i32")) as f32 / 2_147_483_648.0
                }
                (3, 32) => f32::from_le_bytes(item.try_into().expect("f32")),
                (3, 64) => f64::from_le_bytes(item.try_into().expect("f64")) as f32,
                _ => unreachable!(),
            };
            if !sample.is_finite() {
                return Err(MediaError::new("nonfinite WAV sample"));
            }
            sum += sample;
        }
        // Keep the reference's float32 mean and rounding for accepted audio.
        let mean = sum / channels as f32;
        if !mean.is_finite() {
            return Err(MediaError::new("nonfinite WAV channel mean"));
        }
        samples.push(mean);
    }
    Ok(DecodedAudio {
        sample_rate: info.sample_rate,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn decode_stops_when_cancelled_between_sample_blocks() {
        use std::cell::Cell;
        let bytes = float_wav(1, 32, &0.25_f32.to_le_bytes().repeat(8192));
        let polls = Cell::new(0);
        let result = decode_cancellable(&bytes, limits(), &|| {
            polls.set(polls.get() + 1);
            polls.get() == 3
        });
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        assert_eq!(polls.get(), 3);
    }

    #[test]
    fn stereo_pcm16_downmixes_by_channel_mean() {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&44_u32.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&24_000_u32.to_le_bytes());
        bytes.extend_from_slice(&96_000_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        for sample in [16_384_i16, -16_384, 32_767, 32_767] {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        assert_eq!(
            decode(&bytes, limits()).unwrap().samples,
            [0.0, 32_767.0 / 32_768.0]
        );
    }

    fn float_wav(channels: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        let align = channels * bits / 8;
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&3_u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&24_000_u32.to_le_bytes());
        bytes.extend_from_slice(&(24_000 * u32::from(align)).to_le_bytes());
        bytes.extend_from_slice(&align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn finite_float_wav_downmix_refuses_overflow() {
        // The review's 52-byte stereo IEEE-float WAV.
        for sample in [f32::MAX, -f32::MAX] {
            let wav = float_wav(2, 32, &sample.to_le_bytes().repeat(2));
            assert_eq!(wav.len(), 52);
            assert!(
                decode(&wav, limits()).is_err(),
                "finite channels overflowed the mean"
            );
        }
        let data: Vec<_> = [f32::MAX, f32::MAX, -f32::MAX, -f32::MAX]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert!(decode(&float_wav(4, 32, &data), limits()).is_err());
    }

    #[test]
    fn float_wav_preserves_finite_means_without_clipping_or_wider_sums() {
        let data: Vec<_> = [
            1.0_f32,
            f32::EPSILON / 4.0,
            -1.0,
            6.0,
            -3.0,
            3.0,
            -0.0,
            -0.0,
            -0.0,
            f32::MIN_POSITIVE,
            0.0,
            0.0,
        ]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
        let decoded = decode(&float_wav(3, 32, &data), limits()).unwrap();
        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(
            decoded
                .samples
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<_>>(),
            [0, 2.0_f32.to_bits(), 0, 0x002a_aaab]
        );
        let data: Vec<_> = [f32::MAX, -f32::MAX, -f32::MAX, f32::MAX]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        assert_eq!(
            decode(&float_wav(2, 32, &data), limits()).unwrap().samples,
            [0.0, 0.0]
        );
        let data: Vec<_> = [1.25_f64, -0.25, 3.5, 4.5]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        assert_eq!(
            decode(&float_wav(2, 64, &data), limits()).unwrap().samples,
            [0.5, 4.0]
        );
    }

    #[test]
    fn float_wav_refuses_nonfinite_channels_and_float64_conversion_overflow() {
        for sample in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert!(decode(&float_wav(1, 32, &sample.to_le_bytes()), limits()).is_err());
        }
        for sample in [f64::MAX, -f64::MAX, f64::INFINITY, f64::NAN] {
            assert!(decode(&float_wav(1, 64, &sample.to_le_bytes()), limits()).is_err());
        }
    }
}
