//! Parameterized visual preprocessing recipes.
use crate::core::{try_filled, try_with_capacity};
use crate::{MediaError, MediaLimits, MediaResult, RgbImage};

mod centered_pad;
mod resample;
mod token_budget;

pub use centered_pad::{
    CenteredPadPlan, CenteredPadRecipe, centered_pad_grid, centered_pad_plan,
    centered_pad_preparation_bytes, centered_pad_span_tokens, check_centered_pad_recipe,
};
pub use token_budget::{
    PaddedSize, TokenBudgetRecipe, check_token_budget_recipe, padded_emits_size,
    padded_preparation_bytes, padded_size, padded_token_budget_grid,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interpolation {
    /// f32 bilinear at half-pixel centers without antialiasing; normalizes
    /// as `(value * value_scale - mean) / std`.
    #[default]
    Bilinear,
    /// Antialiased bicubic on 8-bit samples with 16-bit fixed-point weights
    /// (the Pillow and PyTorch CPU uint8 path). Normalizes the 8-bit result
    /// with the scale folded into the statistics, as
    /// `(value - mean * s) / (std * s)` in f32 with `s = f32(1 / value_scale)`.
    BicubicAntialias,
}

#[derive(Clone, Copy, Debug)]
pub struct ImageRecipe {
    pub factor: u32,
    pub minimum_pixels: u64,
    pub maximum_pixels: u64,
    /// Raise a side shorter than `factor` to `factor` before rounding.
    pub upscale_small_side: bool,
    pub interpolation: Interpolation,
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub value_scale: f64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatchGrid {
    pub grid: [u32; 3],
    pub columns: usize,
    pub rows: Vec<f32>,
    pub tokens: u64,
}

impl PatchGrid {
    /// Row-major BF16 payload, with round-to-nearest-even conversion.
    pub fn bf16_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.rows.len() * 2);
        for &value in &self.rows {
            let bits = value.to_bits();
            let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
            bytes.extend_from_slice(&((rounded >> 16) as u16).to_le_bytes());
        }
        bytes
    }
}

fn round_even(x: f64) -> f64 {
    let floor = x.floor();
    if x - floor > 0.5 || (x - floor == 0.5 && floor as u64 % 2 == 1) {
        floor + 1.0
    } else {
        floor
    }
}

pub fn smart_resize(
    height: u32,
    width: u32,
    recipe: ImageRecipe,
    limits: MediaLimits,
) -> MediaResult<(u32, u32)> {
    limits.check_image(width, height)?;
    let f = recipe.factor as f64;
    if f == 0.0 || recipe.minimum_pixels == 0 || recipe.maximum_pixels < recipe.minimum_pixels {
        return Err(MediaError::new("invalid image resize recipe"));
    }
    let (mut h, mut w) = (height as f64, width as f64);
    if recipe.upscale_small_side && h.min(w) < f {
        if h < w {
            w = (w * f / h).trunc();
            h = f;
        } else {
            h = (h * f / w).trunc();
            w = f;
        }
    }
    let (mut rh, mut rw) = (round_even(h / f) * f, round_even(w / f) * f);
    if rh * rw > recipe.maximum_pixels as f64 {
        let beta = (h * w / recipe.maximum_pixels as f64).sqrt();
        rh = ((h / beta / f).floor() * f).max(f);
        rw = ((w / beta / f).floor() * f).max(f);
    } else if rh * rw < recipe.minimum_pixels as f64 {
        let beta = (recipe.minimum_pixels as f64 / (h * w)).sqrt();
        rh = (h * beta / f).ceil() * f;
        rw = (w * beta / f).ceil() * f;
    }
    if rh < f || rw < f || rh > u32::MAX as f64 || rw > u32::MAX as f64 {
        return Err(MediaError::new("image resize geometry invalid"));
    }
    let (rh, rw) = (rh as u32, rw as u32);
    limits.check_image(rw, rh)?;
    Ok((rh, rw))
}

/// Whether the recipe resizes some source to exactly `height x width`
/// pixels in whole merge blocks: whether it is its own resize. A source of
/// that size keeps it, so every such size is emitted. Every emitted size is
/// one when [`check_emits_own_resizes`] accepts the recipe and limits.
pub fn emits_size(height: u32, width: u32, recipe: ImageRecipe, limits: MediaLimits) -> bool {
    let block = u64::from(recipe.patch) * u64::from(recipe.merge);
    block != 0
        && u64::from(height) % block == 0
        && u64::from(width) % block == 0
        && smart_resize(height, width, recipe, limits).is_ok_and(|size| size == (height, width))
}

/// Refuses a recipe under which some size [`smart_resize`] emits is not
/// its own resize, so that [`emits_size`] admits exactly its outputs. A
/// size that rounds within the pixel bounds is its own resize. So is one
/// the over-maximum branch floors, if it lands at or above
/// `minimum_pixels`, and one the under-minimum branch ceils, if it lands
/// at or below `maximum_pixels`. For sides `x` and `y` of a source whose
/// aspect is at most `r`, scaled to `x * y = p` pixels, flooring or ceiling
/// each to a multiple of `f` lands strictly between
/// `p - f * sqrt(p) * s + f^2` and `p + f * sqrt(p) * s + f^2`, with
/// `s = sqrt(r) + 1 / sqrt(r)`; an emitted size also has both sides at
/// least `f`. The under-minimum branch is never taken when every rounded
/// size has at least `minimum_pixels`: sides of at least `f` (raised to it,
/// or whole pixels when `f` is 1) round to at least `f`. The emitted sizes
/// must also be sources the limits admit.
pub fn check_emits_own_resizes(recipe: ImageRecipe, limits: MediaLimits) -> MediaResult<()> {
    let ratio = limits.maximum_aspect_ratio;
    if recipe.factor == 0
        || recipe.minimum_pixels == 0
        || recipe.maximum_pixels < recipe.minimum_pixels
        || !ratio.is_finite()
        || ratio < 1.0
    {
        return Err(MediaError::new("invalid image resize recipe"));
    }
    let factor = f64::from(recipe.factor);
    let spread = ratio.sqrt() + 1.0 / ratio.sqrt();
    let (minimum, maximum) = (recipe.minimum_pixels as f64, recipe.maximum_pixels as f64);
    let floored =
        (maximum - factor * maximum.sqrt() * spread + factor * factor).max(factor * factor);
    if floored < minimum {
        return Err(MediaError::new(format!(
            "image resize recipe can floor below its {} minimum pixels (to {floored:.0})",
            recipe.minimum_pixels
        )));
    }
    // Clamping a short axis to one factor can raise the output above the
    // pixel cap. Such a recipe cannot promise idempotent emitted sizes.
    let clamped = (maximum * ratio).sqrt().div_euclid(factor) * factor * factor;
    if clamped > maximum {
        return Err(MediaError::new(
            "image resize factor clamp can exceed maximum pixels",
        ));
    }
    let squared = u64::from(recipe.factor) * u64::from(recipe.factor);
    let ceils =
        recipe.minimum_pixels > squared || !(recipe.upscale_small_side || recipe.factor == 1);
    let ceiled = minimum + factor * minimum.sqrt() * spread + factor * factor;
    if ceils && ceiled > maximum {
        return Err(MediaError::new(format!(
            "image resize recipe can ceil above its {} maximum pixels (to {ceiled:.0})",
            recipe.maximum_pixels
        )));
    }
    if recipe.maximum_pixels > limits.maximum_pixels {
        return Err(MediaError::new(
            "image resize recipe emits more pixels than a source may have",
        ));
    }
    Ok(())
}

/// The recipe each of `frames` video frames resizes with: the total pixel
/// budget shared over the frames, a temporal patch at a time, within the
/// recipe's own bounds.
pub fn video_frame_recipe(
    recipe: ImageRecipe,
    total_max_pixels: u64,
    frames: usize,
) -> ImageRecipe {
    let per_frame = (total_max_pixels.saturating_mul(recipe.temporal_patch as u64)
        / frames.max(1) as u64)
        .clamp(recipe.minimum_pixels, recipe.maximum_pixels);
    ImageRecipe {
        maximum_pixels: per_frame,
        ..recipe
    }
}

fn resized_planar(image: &RgbImage, h: u32, w: u32, recipe: ImageRecipe) -> MediaResult<Vec<f32>> {
    if recipe.std.iter().any(|&s| !s.is_finite() || s == 0.0)
        || !(recipe.value_scale as f32).is_finite()
    {
        return Err(MediaError::new("invalid image normalization"));
    }
    if recipe.interpolation == Interpolation::BicubicAntialias {
        return bicubic_planar(image, h, w, recipe);
    }
    let value_scale = recipe.value_scale as f32;
    let mut out = try_filled(3 * h as usize * w as usize, 0.0)?;
    for y in 0..h as usize {
        let fy = ((y as f32 + 0.5) * image.height as f32 / h as f32 - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(image.height as usize - 1);
        let y1 = (y0 + 1).min(image.height as usize - 1);
        let wy = fy - y0 as f32;
        for x in 0..w as usize {
            let fx = ((x as f32 + 0.5) * image.width as f32 / w as f32 - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(image.width as usize - 1);
            let x1 = (x0 + 1).min(image.width as usize - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let sample = |yy: usize, xx: usize| {
                    image.data[(yy * image.width as usize + xx) * 3 + c] as f32
                };
                let top = sample(y0, x0) * (1.0 - wx) + sample(y0, x1) * wx;
                let bottom = sample(y1, x0) * (1.0 - wx) + sample(y1, x1) * wx;
                out[(c * h as usize + y) * w as usize + x] =
                    ((top * (1.0 - wy) + bottom * wy) * value_scale - recipe.mean[c])
                        / recipe.std[c];
            }
        }
    }
    Ok(out)
}

/// The statistics `(mean * s, std * s)` that normalize 8-bit values with
/// the scale folded in: the f64 reciprocal `s = 1 / value_scale` is rounded
/// to f32 first, then multiplied in f32.
fn folded_statistics(
    value_scale: f64,
    mean: [f32; 3],
    std: [f32; 3],
) -> MediaResult<([f32; 3], [f32; 3])> {
    let inverse = (1.0 / value_scale) as f32;
    let mean = mean.map(|m| m * inverse);
    let std = std.map(|s| s * inverse);
    if std.iter().any(|&s| !s.is_finite() || s == 0.0) || mean.iter().any(|m| !m.is_finite()) {
        return Err(MediaError::new("invalid image normalization"));
    }
    Ok((mean, std))
}

fn bicubic_planar(image: &RgbImage, h: u32, w: u32, recipe: ImageRecipe) -> MediaResult<Vec<f32>> {
    let (mean, std) = folded_statistics(recipe.value_scale, recipe.mean, recipe.std)?;
    let resized = resample::bicubic_antialias(image, h, w)?;
    let plane = h as usize * w as usize;
    let mut out = try_filled(3 * plane, 0.0)?;
    for (i, pixel) in resized.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (f32::from(pixel[c]) - mean[c]) / std[c];
        }
    }
    Ok(out)
}

pub fn patch_grid(
    frames: &[RgbImage],
    recipe: ImageRecipe,
    limits: MediaLimits,
) -> MediaResult<PatchGrid> {
    if frames.is_empty() || frames.len() > limits.maximum_frames {
        return Err(MediaError::new("invalid frame or patch count"));
    }
    if frames
        .iter()
        .any(|frame| frame.width != frames[0].width || frame.height != frames[0].height)
    {
        return Err(MediaError::new("video frames must share geometry"));
    }
    let (h, w) = grid_size(&frames[0], frames.len(), recipe, limits)?;
    let planes = frames
        .iter()
        .map(|f| resized_planar(f, h, w, recipe))
        .collect::<MediaResult<Vec<_>>>()?;
    patch_rows(
        &planes.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        h,
        w,
        recipe.into(),
        limits,
    )
}

/// The resized height and width of `frames` frames shaped like `frame`,
/// checked against the patch grid.
fn grid_size(
    frame: &RgbImage,
    frames: usize,
    recipe: ImageRecipe,
    limits: MediaLimits,
) -> MediaResult<(u32, u32)> {
    if recipe.patch == 0 || recipe.temporal_patch == 0 || recipe.merge == 0 {
        return Err(MediaError::new("invalid frame or patch count"));
    }
    let (h, w) = smart_resize(frame.height, frame.width, recipe, limits)?;
    let block = recipe
        .patch
        .checked_mul(recipe.merge)
        .ok_or_else(|| MediaError::new("patch geometry overflow"))?;
    if h % block != 0 || w % block != 0 || frames % recipe.temporal_patch as usize != 0 {
        return Err(MediaError::new("image does not fit patch grid"));
    }
    Ok((h, w))
}

/// The patch geometry rows are cut with.
#[derive(Clone, Copy)]
struct Patching {
    patch: u32,
    temporal_patch: u32,
    merge: u32,
}

impl From<ImageRecipe> for Patching {
    fn from(recipe: ImageRecipe) -> Self {
        Patching {
            patch: recipe.patch,
            temporal_patch: recipe.temporal_patch,
            merge: recipe.merge,
        }
    }
}

/// Patch rows from normalized planes `[3, h, w]`, one per frame.
fn patch_rows(
    planes: &[&[f32]],
    h: u32,
    w: u32,
    recipe: Patching,
    limits: MediaLimits,
) -> MediaResult<PatchGrid> {
    let frames = planes.len();
    let gh = h / recipe.patch;
    let gw = w / recipe.patch;
    let gt = frames as u32 / recipe.temporal_patch;
    let tokens = u64::from(gt) * u64::from(gh / recipe.merge) * u64::from(gw / recipe.merge);
    limits.check_tokens(tokens)?;
    let columns =
        3usize * recipe.temporal_patch as usize * recipe.patch as usize * recipe.patch as usize;
    let count = (gt as usize)
        .checked_mul(gh as usize)
        .and_then(|n| n.checked_mul(gw as usize))
        .and_then(|n| n.checked_mul(columns))
        .ok_or_else(|| MediaError::new("patch allocation overflow"))?;
    let mut rows = try_with_capacity(count)?;
    for t in 0..gt as usize {
        for bh in 0..(gh / recipe.merge) as usize {
            for bw in 0..(gw / recipe.merge) as usize {
                for mh in 0..recipe.merge as usize {
                    for mw in 0..recipe.merge as usize {
                        for c in 0..3 {
                            for dt in 0..recipe.temporal_patch as usize {
                                for ph in 0..recipe.patch as usize {
                                    for pw in 0..recipe.patch as usize {
                                        let y = (bh * recipe.merge as usize + mh)
                                            * recipe.patch as usize
                                            + ph;
                                        let x = (bw * recipe.merge as usize + mw)
                                            * recipe.patch as usize
                                            + pw;
                                        rows.push(
                                            planes[t * recipe.temporal_patch as usize + dt]
                                                [(c * h as usize + y) * w as usize + x],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(PatchGrid {
        grid: [gt, gh, gw],
        columns,
        rows,
        tokens,
    })
}

/// One image repeated over the temporal patch, resized and normalized once.
pub fn image_grid(
    image: &RgbImage,
    recipe: ImageRecipe,
    limits: MediaLimits,
) -> MediaResult<PatchGrid> {
    let frames = recipe.temporal_patch as usize;
    if frames == 0 || frames > limits.maximum_frames {
        return Err(MediaError::new("invalid frame or patch count"));
    }
    let (h, w) = grid_size(image, frames, recipe, limits)?;
    let plane = resized_planar(image, h, w, recipe)?;
    patch_rows(&vec![plane.as_slice(); frames], h, w, recipe.into(), limits)
}

/// Decoded bytes per source pixel the bound allows: 16-bit RGBA PNG
/// scanlines (8), or a JPEG's component coefficients and planes, plus the
/// 3 of the RGB image.
const DECODE_BYTES_PER_PIXEL: u64 = 16 + 3;

/// A conservative bound on the host bytes an image of `width` x `height`
/// holds while it decodes and becomes patch rows with [`image_grid`],
/// counting every buffer as live at once: the decoder's scanlines and the
/// RGB image, the resampler's horizontal pass and output, the normalized F32
/// plane, the F32 patch rows and a payload of up to 4 bytes per value.
pub fn image_preparation_bytes(
    width: u32,
    height: u32,
    recipe: ImageRecipe,
    limits: MediaLimits,
) -> MediaResult<u64> {
    let (h, w) = smart_resize(height, width, recipe, limits)?;
    let source = u64::from(width) * u64::from(height);
    let resized = u64::from(h) * u64::from(w);
    let values = resized * 3 * u64::from(recipe.temporal_patch.max(1));
    [
        source.checked_mul(DECODE_BYTES_PER_PIXEL),
        u64::from(height).checked_mul(u64::from(w) * 3),
        resized.checked_mul(3),
        resized.checked_mul(3 * 4),
        values.checked_mul(4 + 4),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, part| {
        part.and_then(|part| sum.checked_add(part))
    })
    .ok_or_else(|| MediaError::new("image preparation size overflow"))
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimedGrid {
    pub timestamp: String,
    pub grid: PatchGrid,
}

pub fn video_grids(
    frames: &[(f64, RgbImage)],
    recipe: ImageRecipe,
    total_max_pixels: u64,
    limits: MediaLimits,
) -> MediaResult<Vec<TimedGrid>> {
    if frames.is_empty() || frames.len() > limits.maximum_frames || recipe.temporal_patch == 0 {
        return Err(MediaError::new("invalid video frame count"));
    }
    let frame_recipe = video_frame_recipe(recipe, total_max_pixels, frames.len());
    let mut out = Vec::new();
    for chunk in frames.chunks(recipe.temporal_patch as usize) {
        if chunk.iter().any(|(ts, _)| !ts.is_finite() || *ts < 0.0) {
            return Err(MediaError::new("invalid video timestamp"));
        }
        let mut images: Vec<_> = chunk.iter().map(|(_, f)| f.clone()).collect();
        while images.len() < recipe.temporal_patch as usize {
            images.push(images.last().expect("nonempty chunk").clone());
        }
        let seconds = chunk[0].0.floor() as u64;
        out.push(TimedGrid {
            timestamp: format!("{:02}:{:02}", seconds / 60, seconds % 60),
            grid: patch_grid(&images, frame_recipe, limits)?,
        });
    }
    let total = out.iter().map(|x| x.grid.tokens).sum();
    limits.check_tokens(total)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn over_maximum_retains_one_factor_on_the_short_axis() {
        let recipe = ImageRecipe {
            factor: 32,
            minimum_pixels: 1024,
            maximum_pixels: 8192,
            ..recipe()
        };
        assert_eq!(
            smart_resize(100, 10000, recipe, limits()).unwrap(),
            (32, 896)
        );
    }
    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1 << 20,
            maximum_audio_frames: 1,
            maximum_frames: 8,
            maximum_tokens: 1024,
            maximum_aspect_ratio: 200.0,
        }
    }
    fn recipe() -> ImageRecipe {
        ImageRecipe {
            factor: 2,
            minimum_pixels: 4,
            maximum_pixels: 100,
            upscale_small_side: true,
            interpolation: Interpolation::Bilinear,
            patch: 1,
            temporal_patch: 2,
            merge: 2,
            value_scale: 1.0,
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }
    #[test]
    fn resize_uses_even_ties_and_pixel_bounds() {
        assert_eq!(smart_resize(5, 5, recipe(), limits()).unwrap(), (4, 4));
        assert_eq!(smart_resize(10, 20, recipe(), limits()).unwrap(), (6, 14));
    }
    #[test]
    fn small_side_upscale_is_optional() {
        let recipe = ImageRecipe {
            factor: 32,
            minimum_pixels: 65536,
            maximum_pixels: 1 << 20,
            ..recipe()
        };
        assert_eq!(
            smart_resize(10, 1000, recipe, limits()).unwrap(),
            (32, 3200)
        );
        let recipe = ImageRecipe {
            upscale_small_side: false,
            ..recipe
        };
        // round(10 / 32) is 0, so the minimum-pixel branch scales both sides.
        assert_eq!(
            smart_resize(10, 1000, recipe, limits()).unwrap(),
            (32, 2560)
        );
    }
    #[test]
    fn bicubic_antialias_folds_scale_into_statistics() {
        let recipe = ImageRecipe {
            interpolation: Interpolation::BicubicAntialias,
            value_scale: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
            ..recipe()
        };
        let img = RgbImage::new(2, 2, [0, 128, 255].repeat(4), limits()).unwrap();
        let grid = image_grid(&img, recipe, limits()).unwrap();
        assert_eq!(grid.grid, [1, 2, 2]);
        let expected = [-1.0, -1.0, 0.5 / 127.5, 0.5 / 127.5, 1.0, 1.0];
        assert!(grid.rows.chunks_exact(6).all(|row| row == expected));
        let zero_scale = ImageRecipe {
            value_scale: 0.0,
            ..recipe
        };
        assert!(image_grid(&img, zero_scale, limits()).is_err());
    }
    #[test]
    fn bilinear_patch_order_and_image_repeat() {
        let img =
            RgbImage::new(2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], limits()).unwrap();
        let grid = image_grid(&img, recipe(), limits()).unwrap();
        assert_eq!(grid.grid, [1, 2, 2]);
        assert_eq!(grid.tokens, 1);
        assert_eq!(grid.columns, 6);
        assert_eq!(&grid.rows[..6], &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        assert_eq!(&grid.rows[6..12], &[4.0, 4.0, 5.0, 5.0, 6.0, 6.0]);
    }
    /// A size is emitted exactly when it is its own resize: every resize of
    /// a sweep of sources is one, and sizes below the minimum, over the
    /// maximum or past the aspect limit are not.
    #[test]
    fn emitted_sizes_are_their_own_resize() {
        let recipe = ImageRecipe {
            factor: 4,
            minimum_pixels: 64,
            maximum_pixels: 4096,
            patch: 2,
            merge: 2,
            ..recipe()
        };
        let limits = MediaLimits {
            maximum_pixels: 1 << 20,
            maximum_aspect_ratio: 20.0,
            ..limits()
        };
        check_emits_own_resizes(recipe, limits).unwrap();
        for height in 1..200 {
            for width in 1..200 {
                if let Ok((h, w)) = smart_resize(height, width, recipe, limits) {
                    assert!(
                        emits_size(h, w, recipe, limits),
                        "{height}x{width} -> {h}x{w}"
                    );
                }
            }
        }
        assert!(emits_size(8, 8, recipe, limits));
        assert!(emits_size(64, 64, recipe, limits));
        for [h, w] in [[4, 4], [68, 64], [4, 84], [8, 6], [6, 8]] {
            assert!(!emits_size(h, w, recipe, limits), "{h}x{w}");
        }
    }

    /// Recipes whose floored or ceiled resizes leave the pixel bounds are
    /// refused: their outputs would not be their own resize. A floor of
    /// 36 x 280 to 32 x 224 under bounds of 8,192 pixels lands at 7,168,
    /// which resizes again to 64 x 256.
    #[test]
    fn recipes_whose_resizes_leave_their_bounds_are_refused() {
        let recipe = ImageRecipe {
            factor: 32,
            minimum_pixels: 8192,
            maximum_pixels: 8192,
            patch: 16,
            merge: 2,
            ..recipe()
        };
        let limits = MediaLimits {
            maximum_pixels: 1 << 26,
            maximum_aspect_ratio: 200.0,
            ..limits()
        };
        assert_eq!(smart_resize(36, 280, recipe, limits).unwrap(), (32, 224));
        assert_eq!(smart_resize(32, 224, recipe, limits).unwrap(), (64, 256));
        assert!(!emits_size(32, 224, recipe, limits));
        let error = check_emits_own_resizes(recipe, limits).unwrap_err();
        assert!(error.to_string().contains("floor below"), "{error}");
        // Wide bounds hold; a narrow range or a maximum past the source
        // limit does not.
        let ceiled = ImageRecipe {
            minimum_pixels: 8192,
            maximum_pixels: 600_000,
            ..recipe
        };
        assert!(check_emits_own_resizes(ceiled, limits).is_ok());
        let tight = ImageRecipe {
            maximum_pixels: 9_000,
            ..ceiled
        };
        assert!(check_emits_own_resizes(tight, limits).is_err());
        let small = MediaLimits {
            maximum_pixels: 100_000,
            ..limits
        };
        assert!(check_emits_own_resizes(ceiled, small).is_err());
    }

    /// Tiny recipes are exact too: with a factor of 1 every side rounds to
    /// itself. The aspect limit must also keep the factor clamp within the cap.
    #[test]
    fn whole_pixel_recipes_emit_their_own_resizes() {
        let recipe = ImageRecipe {
            factor: 1,
            minimum_pixels: 1,
            maximum_pixels: 4,
            patch: 1,
            merge: 1,
            upscale_small_side: false,
            ..recipe()
        };
        assert!(check_emits_own_resizes(recipe, limits()).is_err());
        let limits = MediaLimits {
            maximum_aspect_ratio: 4.0,
            ..limits()
        };
        check_emits_own_resizes(recipe, limits).unwrap();
        for height in 1..60 {
            for width in 1..60 {
                if let Ok((h, w)) = smart_resize(height, width, recipe, limits) {
                    assert!(
                        emits_size(h, w, recipe, limits),
                        "{height}x{width} -> {h}x{w}"
                    );
                }
            }
        }
    }

    #[test]
    fn video_frames_share_the_total_budget() {
        let recipe = ImageRecipe {
            minimum_pixels: 16,
            maximum_pixels: 100,
            ..recipe()
        };
        assert_eq!(video_frame_recipe(recipe, 1_000, 2).maximum_pixels, 100);
        assert_eq!(video_frame_recipe(recipe, 100, 4).maximum_pixels, 50);
        assert_eq!(video_frame_recipe(recipe, 10, 8).maximum_pixels, 16);
    }

    #[test]
    fn video_repeats_odd_last_frame_and_formats_timestamp() {
        let img = RgbImage::new(2, 2, vec![7; 12], limits()).unwrap();
        let grids = video_grids(&[(61.8, img)], recipe(), 100, limits()).unwrap();
        assert_eq!(grids[0].timestamp, "01:01");
        assert!(grids[0].grid.rows.iter().all(|&x| x == 7.0));
    }
}
