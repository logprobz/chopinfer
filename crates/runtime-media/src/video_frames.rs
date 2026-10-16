//! Frame extraction for video inputs: which decoded frames a request keeps,
//! the time label each kept frame carries, and the conversion of decoded 4:2:0
//! frames to the interleaved RGB bytes the image recipes consume.
//!
//! The rules follow the uniform-sampling OpenCV loader of the serving
//! reference. It counts frames from the container header, samples at most a
//! fixed number of them uniformly over that count, reads them in display
//! order, and converts each with FFmpeg's `yuv420p` to `bgr24` path before
//! swapping to RGB. The processor then labels the kept frames at a fixed
//! spacing, whatever their real timing.

use std::error::Error;
use std::fmt;

/// How many decoded frames a request keeps and how they are labeled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameSampling {
    /// The most frames kept from one video.
    pub maximum_frames: usize,
    /// The label spacing between consecutive kept frames, in seconds.
    pub seconds_per_frame: u32,
}

/// One frame a request keeps.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SampledFrame {
    /// The frame's position in display order among presentable frames.
    pub display_index: usize,
    /// The frame's time label in seconds.
    pub label_seconds: u32,
}

/// A refusal while extracting frames.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VideoFrameError {
    /// The frame geometry cannot carry 4:2:0 chroma.
    Geometry { width: u32, height: u32 },
    /// The NV12 buffer does not match the frame geometry.
    Length { expected: usize, actual: usize },
    /// No frame survived sampling.
    NoFrames,
}

impl fmt::Display for VideoFrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry { width, height } => write!(
                formatter,
                "video frame {width}x{height} is not an even, nonzero 4:2:0 geometry"
            ),
            Self::Length { expected, actual } => write!(
                formatter,
                "NV12 frame holds {actual} bytes, expected {expected}"
            ),
            Self::NoFrames => formatter.write_str("video has no presentable frames"),
        }
    }
}

impl Error for VideoFrameError {}

/// The frame count the loader samples over.
///
/// The header sample count is used unless the edit list hides more than one
/// of those samples, in which case the presentable count replaces it. A
/// one-frame difference is tolerated as a header miscount.
#[must_use]
pub fn sampling_total(header_samples: usize, presentable_frames: usize) -> usize {
    if presentable_frames > 0 && presentable_frames < header_samples.saturating_sub(1) {
        presentable_frames
    } else {
        header_samples
    }
}

/// Uniform frame indices over `total` frames, at most `maximum` of them.
///
/// Every frame is kept when `total` fits. Otherwise the indices are
/// `linspace(0, total - 1, maximum)` truncated toward zero, computed in f64 as
/// NumPy does, with the last index pinned to `total - 1`.
#[must_use]
pub fn uniform_frame_indices(total: usize, maximum: usize) -> Vec<usize> {
    let count = maximum.min(total);
    if count == total {
        return (0..total).collect();
    }
    if count <= 1 {
        return vec![0; count];
    }
    let step = (total - 1) as f64 / (count - 1) as f64;
    let mut indices: Vec<usize> = (0..count).map(|i| (i as f64 * step) as usize).collect();
    indices[count - 1] = total - 1;
    indices
}

/// The frames one video keeps, in display order, with their labels.
///
/// `header_samples` is the container's sample count and `presentable_frames`
/// the number of frames the edit list shows. A sampled index past the
/// presentable frames is dropped, as a failed read is, and labels count only
/// the frames that remain.
pub fn frame_plan(
    header_samples: usize,
    presentable_frames: usize,
    sampling: FrameSampling,
) -> Result<Vec<SampledFrame>, VideoFrameError> {
    let total = sampling_total(header_samples, presentable_frames);
    let plan: Vec<SampledFrame> = uniform_frame_indices(total, sampling.maximum_frames)
        .into_iter()
        .filter(|&index| index < presentable_frames)
        .enumerate()
        .map(|(position, display_index)| SampledFrame {
            display_index,
            label_seconds: (position as u32).saturating_mul(sampling.seconds_per_frame),
        })
        .collect();
    if plan.is_empty() {
        return Err(VideoFrameError::NoFrames);
    }
    Ok(plan)
}

/// BT.601 limited-range coefficients in FFmpeg's 16-bit SIMD form.
///
/// Each is `round(c * 2^13 / 2^16)` of the table coefficients FFmpeg derives
/// for `SWS_CS_DEFAULT` with default contrast, saturation and brightness.
const Y_COEFFICIENT: i32 = 9539;
const V_TO_RED: i32 = 13075;
const U_TO_BLUE: i32 = 16525;
const U_TO_GREEN: i32 = -3209;
const V_TO_GREEN: i32 = -6660;
const Y_OFFSET: i32 = 16 * 8;
const CHROMA_OFFSET: i32 = 128 * 8;

/// Signed 16-bit multiply keeping the high half, as `pmulhw` does.
fn multiply_high(value: i32, coefficient: i32) -> i32 {
    (value * coefficient) >> 16
}

fn saturate_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// Converts one NV12 frame to interleaved RGB.
///
/// `nv12` holds `height` rows of `width` luma bytes, then `height / 2` rows
/// of `width` interleaved U and V bytes. Each chroma sample covers a 2x2 luma
/// block with no interpolation. The arithmetic is FFmpeg's x86 SIMD
/// `yuv420p` to `bgr24` conversion, which the reference loader runs on
/// x86-64, with the channels then swapped to RGB. FFmpeg's portable C tables
/// differ by up to a few codes; qualification captures decide the final
/// match.
pub fn nv12_to_rgb(width: u32, height: u32, nv12: &[u8]) -> Result<Vec<u8>, VideoFrameError> {
    if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
        return Err(VideoFrameError::Geometry { width, height });
    }
    let (width, height) = (width as usize, height as usize);
    let luma_bytes = width * height;
    let expected = luma_bytes + luma_bytes / 2;
    if nv12.len() != expected {
        return Err(VideoFrameError::Length {
            expected,
            actual: nv12.len(),
        });
    }
    let (luma, chroma) = nv12.split_at(luma_bytes);
    let mut rgb = vec![0; luma_bytes * 3];
    for y in 0..height {
        let chroma_row = &chroma[(y / 2) * width..(y / 2 + 1) * width];
        for x in 0..width {
            let u = (i32::from(chroma_row[x & !1]) << 3) - CHROMA_OFFSET;
            let v = (i32::from(chroma_row[x | 1]) << 3) - CHROMA_OFFSET;
            let luma = multiply_high(
                (i32::from(luma[y * width + x]) << 3) - Y_OFFSET,
                Y_COEFFICIENT,
            );
            let green = multiply_high(u, U_TO_GREEN) + multiply_high(v, V_TO_GREEN);
            let out = &mut rgb[(y * width + x) * 3..][..3];
            out[0] = saturate_u8(luma + multiply_high(v, V_TO_RED));
            out[1] = saturate_u8(luma + green);
            out[2] = saturate_u8(luma + multiply_high(u, U_TO_BLUE));
        }
    }
    Ok(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_to_int16(value: i64) -> i32 {
        ((value + (1 << 15)) >> 16) as i32
    }

    #[test]
    fn coefficients_follow_the_default_colorspace_derivation() {
        // The unscaled luma gain for limited range, then each coefficient in
        // the 2^13 fixed point the SIMD path loads.
        let luma_gain = (65536i64 * 255) / 219;
        assert_eq!(round_to_int16(luma_gain << 13), Y_COEFFICIENT);
        assert_eq!(round_to_int16(104597 << 13), V_TO_RED);
        assert_eq!(round_to_int16(132201 << 13), U_TO_BLUE);
        assert_eq!(round_to_int16(-25675 << 13), U_TO_GREEN);
        assert_eq!(round_to_int16(-53279 << 13), V_TO_GREEN);
    }
}
