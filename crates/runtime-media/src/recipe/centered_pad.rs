//! `image.centered-pad.v1`: an image fitted at its own aspect ratio into a
//! patch-aligned canvas whose delimited token span lies within a budget,
//! then centered on a constant color.
//!
//! The plan takes the source's `w x h` pixels:
//! - Below `minimum_pixels`, both sides scale by
//!   `s = (minimum_pixels / (w * h))^0.5` (f64 `pow`) and truncate.
//! - Each side aligns up to a multiple of `patch`; the canvas is
//!   `n_h x n_w` patches and `ceil(n_h / d) x ceil(n_w / d)` tokens after
//!   the `d x d` downsample (`d = downsample`, zero-padding odd grids).
//! - The span costs `rows * (columns + 1) + 2` tokens: one delimiter per
//!   token row and two per image. Over `maximum_tokens`, the canvas is
//!   solved again from the (possibly scaled) sides at ratio `r = h / w`:
//!   `x = sqrt((maximum_tokens - 2) / r + 0.25) - 0.5` token columns and
//!   `x * r` rows fit. Under one column the canvas is one cell wide and
//!   `(maximum_tokens - 2) / 2` cells high; under one row it is one cell
//!   high and `maximum_tokens - 3` cells wide; otherwise both sides scale
//!   by `b = min(floor(x) * c / w, floor(x * r) * c / h)` (`c = patch * d`)
//!   and floor to whole patches. A solved canvas over the budget is
//!   refused.
//!
//! The content is the source resized into the canvas at its own aspect
//! ratio (Pillow's `ImageOps.contain`): the side that does not fill is
//! `round(h / w * W)` or `round(w / h * H)` in f64, ties to even. A changed
//! size is resampled with Pillow's 22-bit bicubic ([`super::resample`]).
//! The content sits on a canvas filled with `pad` (`ImageOps.pad`) at
//! offset `round((W - content) * 0.5)` along the side that does not fill,
//! ties to even. Each 8-bit value `v` normalizes in f32 as
//! `(v / value_divisor - mean) / std`, each operation rounded, and the
//! rows are the canvas's patches in reading order, each channel-major
//! (`c * patch^2 + y * patch + x`). With [`ValueRounding::Bf16`] each
//! normalized value is then rounded once to the nearest BF16, ties to even,
//! and kept in f32: the input of a model whose processor casts its
//! normalized pixels to BF16 before the encoder.
use super::canvas::{CanvasLayout, CanvasRows};
use super::{CANVAS_HASH_CHUNK_BYTES, PatchGrid, resample, round_even};
use crate::core::try_filled;
use crate::{MediaError, MediaLimits, MediaResult, RgbImage};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenteredPadRecipe {
    pub patch: u32,
    pub downsample: u32,
    /// Sources with fewer pixels scale up to about this many.
    pub minimum_pixels: u64,
    /// The largest delimited span: `rows * (columns + 1) + 2` tokens.
    pub maximum_tokens: u64,
    /// The canvas color outside the content.
    pub pad: [u8; 3],
    pub value_divisor: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// How each normalized value is stored.
    pub value_rounding: ValueRounding,
}

/// How a normalized value is stored in the f32 rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ValueRounding {
    /// As normalized in f32.
    #[default]
    None,
    /// Rounded once to the nearest BF16, ties to even, and kept in f32
    /// (every value is finite, so the rounding never meets a NaN).
    Bf16,
}

impl ValueRounding {
    fn apply(self, value: f32) -> f32 {
        match self {
            ValueRounding::None => value,
            ValueRounding::Bf16 => {
                let bits = value.to_bits();
                f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
            }
        }
    }
}

/// Integers up to this convert to f64 exactly.
const EXACT: u64 = 1 << 53;

impl CenteredPadRecipe {
    /// The cell `patch * downsample`, checked with the token budget.
    fn cell(&self) -> MediaResult<u64> {
        let cell = u64::from(self.patch) * u64::from(self.downsample);
        if cell == 0
            || self.maximum_tokens < 4
            || self.minimum_pixels > EXACT
            || cell
                .checked_mul(self.maximum_tokens)
                .is_none_or(|side| side > u64::from(u32::MAX))
        {
            return Err(MediaError::new("invalid centered pad recipe"));
        }
        Ok(cell)
    }
}

/// The cost of a delimited span of `rows x columns` tokens.
pub fn centered_pad_span_tokens(rows: u64, columns: u64) -> u64 {
    rows * (columns + 1) + 2
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CenteredPadPlan {
    /// Height and width of the canvas, in whole patches.
    pub canvas: (u32, u32),
    /// Height and width of the resized source on the canvas.
    pub content: (u32, u32),
    /// Row and column of the content's top left on the canvas.
    pub offset: (u32, u32),
    /// Token rows and columns after the downsample.
    pub tokens: (u32, u32),
}

/// The plan of a `height x width` source.
pub fn centered_pad_plan(
    height: u32,
    width: u32,
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> MediaResult<CenteredPadPlan> {
    limits.check_image(width, height)?;
    let cell = recipe.cell()?;
    let (p, d) = (u64::from(recipe.patch), u64::from(recipe.downsample));
    let (mut h, mut w) = (u64::from(height), u64::from(width));
    if h * w < recipe.minimum_pixels {
        let ratio = (recipe.minimum_pixels as f64 / (w * h) as f64).powf(0.5);
        w = (w as f64 * ratio) as u64;
        h = (h as f64 * ratio) as u64;
    }
    let grid =
        |canvas_h: u64, canvas_w: u64| ((canvas_h / p).div_ceil(d), (canvas_w / p).div_ceil(d));
    let (mut canvas_h, mut canvas_w) = (h.div_ceil(p) * p, w.div_ceil(p) * p);
    let (mut rows, mut columns) = grid(canvas_h, canvas_w);
    if centered_pad_span_tokens(rows, columns) > recipe.maximum_tokens {
        let budget = recipe.maximum_tokens;
        let r = h as f64 / w as f64;
        let fit_w = (((budget - 2) as f64 / r + 0.25).sqrt()) - 0.5;
        let fit_h = fit_w * r;
        (canvas_h, canvas_w) = if fit_w < 1.0 {
            ((budget - 2) / 2 * cell, cell)
        } else if fit_h < 1.0 {
            (cell, (budget - 3) * cell)
        } else {
            let beta = ((fit_w.floor() as u64 * cell) as f64 / w as f64)
                .min((fit_h.floor() as u64 * cell) as f64 / h as f64);
            (
                (h as f64 * beta / p as f64).floor() as u64 * p,
                (w as f64 * beta / p as f64).floor() as u64 * p,
            )
        };
        (rows, columns) = grid(canvas_h, canvas_w);
        if canvas_h == 0
            || canvas_w == 0
            || centered_pad_span_tokens(rows, columns) > recipe.maximum_tokens
        {
            return Err(MediaError::new(
                "centered pad canvas exceeds its token budget",
            ));
        }
    }
    let size = |side: u64| {
        u32::try_from(side).map_err(|_| MediaError::new("image resize geometry invalid"))
    };
    let (source_h, source_w) = (f64::from(height), f64::from(width));
    let (fit_h, fit_w) = (canvas_h as f64, canvas_w as f64);
    let (mut content_h, mut content_w) = (canvas_h, canvas_w);
    if source_w / source_h != fit_w / fit_h {
        if source_w / source_h > fit_w / fit_h {
            content_h = round_even(source_h / source_w * fit_w) as u64;
        } else {
            content_w = round_even(source_w / source_h * fit_h) as u64;
        }
    }
    if content_h == 0 || content_w == 0 {
        return Err(MediaError::new("centered pad content is empty"));
    }
    let offset = (
        round_even((canvas_h - content_h) as f64 * 0.5) as u64,
        round_even((canvas_w - content_w) as f64 * 0.5) as u64,
    );
    let plan = CenteredPadPlan {
        canvas: (size(canvas_h)?, size(canvas_w)?),
        content: (size(content_h)?, size(content_w)?),
        offset: (size(offset.0)?, size(offset.1)?),
        tokens: (size(rows)?, size(columns)?),
    };
    // Only the canvas's pixels are limited: its aspect ratio can pass the
    // source's, since rounding can drop a side by a patch (a 79 x 15000
    // source plans a 28 x 7966 canvas).
    if canvas_h * canvas_w > limits.maximum_pixels {
        return Err(MediaError::new("decoded image pixel limit exceeded"));
    }
    Ok(plan)
}

/// Whether the recipe may emit a `height x width` canvas: one aligned to
/// the patch, within the token budget and the decoded pixel limit. Every
/// emitted canvas is such a size. Conversely, such a size with at least
/// `minimum_pixels` is its own plan, so a source of that size emits it;
/// a smaller one is admitted without showing a source reaches it. The
/// upscale's truncation and the solve's rounding (`floor(w * (a * c / w) /
/// patch)` can fall one patch short) both emit canvases under
/// `minimum_pixels` that are not their own plan, and those have no closed
/// form, so this admits a superset of the emitted canvases below
/// `minimum_pixels` and exactly the emitted ones at or above it.
pub fn centered_pad_emits_size(
    height: u32,
    width: u32,
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> bool {
    let Ok(_) = recipe.cell() else {
        return false;
    };
    let (p, d) = (u64::from(recipe.patch), u64::from(recipe.downsample));
    let (h, w) = (u64::from(height), u64::from(width));
    h > 0
        && w > 0
        && h % p == 0
        && w % p == 0
        && h * w <= limits.maximum_pixels
        && centered_pad_span_tokens((h / p).div_ceil(d), (w / p).div_ceil(d))
            <= recipe.maximum_tokens
}

/// Refuses a recipe [`centered_pad_plan`] or [`centered_pad_grid`] cannot
/// run: a zero patch or downsample, a budget under 4 tokens (the one-cell
/// canvases need at least one cell), cells past `u32`, or a normalization
/// with a zero, negative or non-finite divisor, a zero or non-finite std or
/// a non-finite mean.
pub fn check_centered_pad_recipe(recipe: CenteredPadRecipe) -> MediaResult<()> {
    recipe.cell()?;
    if !(recipe.value_divisor.is_finite() && recipe.value_divisor > 0.0)
        || recipe.std.iter().any(|&s| !s.is_finite() || s == 0.0)
        || recipe.mean.iter().any(|m| !m.is_finite())
    {
        return Err(MediaError::new("invalid image normalization"));
    }
    Ok(())
}

/// The normalized value of each 8-bit sample per channel.
fn normalized(recipe: CenteredPadRecipe) -> [[f32; 256]; 3] {
    let mut table = [[0.0; 256]; 3];
    for (c, values) in table.iter_mut().enumerate() {
        for (v, value) in values.iter_mut().enumerate() {
            *value = recipe
                .value_rounding
                .apply((v as f32 / recipe.value_divisor - recipe.mean[c]) / recipe.std[c]);
        }
    }
    table
}

/// Patch rows of one image: grid `[1, n_h, n_w]` and the downsampled token
/// count `tokens.0 * tokens.1`.
pub fn centered_pad_grid(
    image: &RgbImage,
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> MediaResult<PatchGrid> {
    centered_pad_canvas(image, recipe, limits)?.into_grid()
}

/// The 8-bit canvas [`centered_pad_grid`] cuts its rows from: the content
/// centered on the pad color, with each sample's normalized (and rounded)
/// value per channel.
pub fn centered_pad_canvas(
    image: &RgbImage,
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> MediaResult<CanvasRows> {
    check_centered_pad_recipe(recipe)?;
    let plan = centered_pad_plan(image.height, image.width, recipe, limits)?;
    let tokens = u64::from(plan.tokens.0) * u64::from(plan.tokens.1);
    limits.check_tokens(tokens)?;
    let (h, w) = (plan.canvas.0 as usize, plan.canvas.1 as usize);
    let (content_h, content_w) = (plan.content.0 as usize, plan.content.1 as usize);
    let (top, left) = (plan.offset.0 as usize, plan.offset.1 as usize);
    let resized = resample::bicubic_pillow(image, plan.content.0, plan.content.1)?;
    let mut canvas = try_filled(h * w * 3, 0u8)?;
    for pixel in canvas.chunks_exact_mut(3) {
        pixel.copy_from_slice(&recipe.pad);
    }
    for (y, row) in resized.chunks_exact(content_w * 3).enumerate() {
        let at = ((top + y) * w + left) * 3;
        canvas[at..at + content_w * 3].copy_from_slice(row);
    }
    debug_assert_eq!(resized.len(), content_h * content_w * 3);
    drop(resized);
    CanvasRows::new(
        plan.canvas.0,
        plan.canvas.1,
        canvas,
        normalized(recipe),
        CanvasLayout::Patches {
            patch: recipe.patch,
        },
        tokens,
    )
}

/// The recipe's share of the host bytes a [`centered_pad_canvas`]
/// preparation holds, beside the decoder's own: the resampler's weights for
/// both passes, its intermediate pass in either order and its output, the
/// canvas, the value tables and the chunk its payload is hashed through.
pub fn centered_pad_canvas_preparation_bytes(
    width: u32,
    height: u32,
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> MediaResult<u64> {
    let plan = centered_pad_plan(height, width, recipe, limits)?;
    let (content_h, content_w) = plan.content;
    let content = u64::from(content_h) * u64::from(content_w);
    let canvas = u64::from(plan.canvas.0) * u64::from(plan.canvas.1);
    let intermediate =
        (u64::from(height) * u64::from(content_w)).max(u64::from(content_h) * u64::from(width));
    [
        Some(resample::coefficient_bytes(width, content_w)),
        Some(resample::coefficient_bytes(height, content_h)),
        intermediate.checked_mul(3),
        content.checked_mul(3),
        canvas.checked_mul(3),
        Some(3 * 256 * 4 * 2),
        Some(CANVAS_HASH_CHUNK_BYTES as u64 + 3 * 4 * u64::from(recipe.patch).pow(2)),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, part| {
        part.and_then(|part| sum.checked_add(part))
    })
    .ok_or_else(|| MediaError::new("image preparation size overflow"))
}

/// A conservative bound on the host bytes an encoded image
/// holds while it decodes and becomes patch rows with
/// [`centered_pad_grid`], in the terms of
/// [`super::image_preparation_bytes`]: the decoder's buffers, the
/// resampler's weights for both passes ([`resample::coefficient_bytes`]),
/// its intermediate pass in either order and its output, the 8-bit canvas,
/// the normalization table, the F32 patch rows and a payload of up to 4
/// bytes per value.
pub fn centered_pad_preparation_bytes(
    bytes: &[u8],
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> MediaResult<u64> {
    let probe = crate::probe_oriented(bytes, limits, crate::Orientation::Stored)?;
    let (width, height) = match probe {
        crate::Probe::Jpeg { width, height } | crate::Probe::Png { width, height } => {
            (width, height)
        }
        _ => return Err(MediaError::new("centered pad requires a PNG or JPEG image")),
    };
    let decode = crate::image_decode_bytes(bytes, probe, limits)?;
    centered_pad_buffer_bytes(width, height, decode, recipe, limits)
}

fn centered_pad_buffer_bytes(
    width: u32,
    height: u32,
    decode_bytes: u64,
    recipe: CenteredPadRecipe,
    limits: MediaLimits,
) -> MediaResult<u64> {
    let plan = centered_pad_plan(height, width, recipe, limits)?;
    let (content_h, content_w) = plan.content;
    let content = u64::from(content_h) * u64::from(content_w);
    let canvas = u64::from(plan.canvas.0) * u64::from(plan.canvas.1);
    let intermediate =
        (u64::from(height) * u64::from(content_w)).max(u64::from(content_h) * u64::from(width));
    [
        Some(decode_bytes),
        Some(resample::coefficient_bytes(width, content_w)),
        Some(resample::coefficient_bytes(height, content_h)),
        intermediate.checked_mul(3),
        content.checked_mul(3),
        canvas.checked_mul(3),
        Some(3 * 256 * 4),
        canvas.checked_mul(3 * (4 + 4)),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, part| {
        part.and_then(|part| sum.checked_add(part))
    })
    .ok_or_else(|| MediaError::new("image preparation size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1 << 28,
            maximum_audio_frames: 1,
            maximum_frames: 1,
            maximum_tokens: 1 << 20,
            maximum_aspect_ratio: 100_000.0,
        }
    }

    fn recipe() -> CenteredPadRecipe {
        CenteredPadRecipe {
            patch: 14,
            downsample: 3,
            minimum_pixels: 295_936,
            maximum_tokens: 1024,
            pad: [127; 3],
            value_divisor: 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
            value_rounding: ValueRounding::None,
        }
    }

    fn plan(height: u32, width: u32) -> CenteredPadPlan {
        centered_pad_plan(height, width, recipe(), limits()).unwrap()
    }

    #[test]
    fn plans_scale_align_solve_and_center() {
        // Aligned, square and above the minimum: kept whole.
        assert_eq!(
            plan(560, 560),
            CenteredPadPlan {
                canvas: (560, 560),
                content: (560, 560),
                offset: (0, 0),
                tokens: (14, 14),
            }
        );
        // Under the minimum: 100 x 200 scales by sqrt(14.7968) to 384 x 769,
        // aligned to 392 x 770, and the content keeps the source's 1:2.
        let small = plan(100, 200);
        assert_eq!(small.canvas, (392, 770));
        assert_eq!(small.content, (385, 770));
        assert_eq!(small.offset, (4, 0));
        assert_eq!(small.tokens, (10, 19));
        // Over the budget: 1500 x 2000 (r = 0.75) fits x = 36.42 token
        // columns and 27.31 rows, so both sides scale by min(36 * 42 /
        // 2000, 27 * 42 / 1500) = 0.756 to 1134 x 1512, 27 x 36 tokens.
        let large = plan(1500, 2000);
        assert_eq!(large.canvas, (1134, 1512));
        assert_eq!(large.tokens, (27, 36));
        assert_eq!(centered_pad_span_tokens(27, 36), 1001);
        // Very tall and very wide sources collapse to one cell.
        assert_eq!(plan(100_000, 10).canvas, (511 * 42, 42));
        assert_eq!(plan(10, 100_000).canvas, (42, 1021 * 42));
    }

    #[test]
    fn rounding_can_drop_a_patch_and_pass_the_source_aspect() {
        // 79 x 15000 solves at a = 1 token column: 79 * (42 / 79) is just
        // under 42 in f64, so the canvas is 2 patches wide, not 3, as the
        // reference's plan_image_grid gives. Its 284.5:1 aspect passes the
        // source's 190:1 and a 200:1 limit, and its area is under the
        // minimum, so it is not its own plan.
        let narrow = MediaLimits {
            maximum_aspect_ratio: 200.0,
            ..limits()
        };
        let plan = centered_pad_plan(15000, 79, recipe(), narrow).unwrap();
        assert_eq!(plan.canvas, (7966, 28));
        assert_eq!(plan.tokens, (190, 1));
        assert!(narrow.check_image(28, 7966).is_err());
        assert_ne!(
            centered_pad_plan(7966, 28, recipe(), limits())
                .unwrap()
                .canvas,
            plan.canvas
        );
        assert!(centered_pad_emits_size(7966, 28, recipe(), narrow));
    }

    #[test]
    fn emitted_sizes_are_aligned_within_budget_and_pixels() {
        let limits = limits();
        assert!(centered_pad_emits_size(560, 560, recipe(), limits));
        // Its own plan at or above the minimum.
        for (h, w) in [(1134, 1512), (511 * 42, 42), (42, 1021 * 42)] {
            assert!(centered_pad_emits_size(h, w, recipe(), limits));
            assert_eq!(plan(h, w).canvas, (h, w));
        }
        assert!(!centered_pad_emits_size(561, 560, recipe(), limits));
        assert!(!centered_pad_emits_size(0, 560, recipe(), limits));
        // 34 x 30 tokens span 34 * 31 + 2 = 1056 > 1024.
        assert!(!centered_pad_emits_size(34 * 42, 30 * 42, recipe(), limits));
        let small = MediaLimits {
            maximum_pixels: 560 * 560 - 1,
            ..limits
        };
        assert!(!centered_pad_emits_size(560, 560, recipe(), small));
    }

    #[test]
    fn preparation_bytes_hold_the_resampler_weights() {
        // A 1 x 10000 source at patch and downsample 1, no minimum and a
        // 15002-token budget collapses to one 7500-high column: the
        // vertical pass's 7 weights per output hold 7500 * (7 * 12 + 16)
        // bytes beside the buffers.
        let recipe = CenteredPadRecipe {
            patch: 1,
            downsample: 1,
            minimum_pixels: 0,
            maximum_tokens: 15_002,
            ..recipe()
        };
        let plan = centered_pad_plan(10_000, 1, recipe, limits()).unwrap();
        assert_eq!((plan.canvas, plan.content), ((7500, 1), (7500, 1)));
        let decode = 10_000 * super::super::DECODE_BYTES_PER_PIXEL;
        let bytes = centered_pad_buffer_bytes(1, 10_000, decode, recipe, limits()).unwrap();
        let weights = 7500 * (7 * 12 + 16);
        let buffers = 10_000 * super::super::DECODE_BYTES_PER_PIXEL
            + 10_000 * 3
            + 7500 * 3
            + 7500 * 3
            + 3 * 256 * 4
            + 7500 * 3 * 8;
        assert_eq!(bytes, weights + buffers);
    }

    #[test]
    fn offsets_round_half_to_even() {
        // A 3-pixel margin centers at round(1.5) = 2, a 5-pixel one at
        // round(2.5) = 2.
        for (margin, offset) in [(1, 0), (3, 2), (5, 2), (7, 4)] {
            let canvas_w = 588_u64;
            let content_w = canvas_w - margin;
            assert_eq!(
                round_even((canvas_w - content_w) as f64 * 0.5) as u64,
                offset
            );
        }
    }

    #[test]
    fn rows_are_patches_in_reading_order_channel_major() {
        // A 588 x 588 source keeps its size, so the canvas is the source.
        let (h, w) = (588_u32, 588_u32);
        let data: Vec<u8> = (0..h * w)
            .flat_map(|i| {
                let (y, x) = (i / w, i % w);
                [(x % 251) as u8, (y % 241) as u8, ((x + y) % 7) as u8]
            })
            .collect();
        let image = RgbImage::new(w, h, data, limits()).unwrap();
        let grid = centered_pad_grid(&image, recipe(), limits()).unwrap();
        assert_eq!(grid.grid, [1, 42, 42]);
        assert_eq!(grid.columns, 588);
        assert_eq!(grid.tokens, 14 * 14);
        let value = |v: u8| (f32::from(v) / 255.0 - 0.5) / 0.5;
        // Patch (2, 5), channel 1 (the row), offset (3, 4).
        let row = 2 * 42 + 5;
        let at = row * 588 + 196 + 3 * 14 + 4;
        assert_eq!(grid.rows[at].to_bits(), value((2 * 14 + 3) as u8).to_bits());
        // Channel 0 holds the column.
        let at = row * 588 + 3 * 14 + 4;
        assert_eq!(grid.rows[at].to_bits(), value((5 * 14 + 4) as u8).to_bits());
    }

    #[test]
    fn padding_takes_the_pad_color() {
        // 300 x 600 is under the minimum and scales to 384 x 769 pixels,
        // aligned to 392 x 770: the content is 385 rows high and 4 pad
        // rows sit above it.
        let image = RgbImage::new(600, 300, vec![255; 600 * 300 * 3], limits()).unwrap();
        let plan = plan(300, 600);
        assert_eq!((plan.offset, plan.content), ((4, 0), (385, 770)));
        let grid = centered_pad_grid(&image, recipe(), limits()).unwrap();
        let pad = (127.0_f32 / 255.0 - 0.5) / 0.5;
        assert_eq!(grid.rows[0].to_bits(), pad.to_bits());
        // Row 4 of the first patch is content.
        assert_eq!(grid.rows[4 * 14].to_bits(), 1.0_f32.to_bits());
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        let step = 2f32.powi(-7);
        for (value, rounded) in [
            (1.0, 1.0),
            (1.0 + step / 2.0, 1.0),
            (1.0 + step * 1.5, 1.0 + step * 2.0),
            (1.0 + step * 0.75, 1.0 + step),
            (-1.0 - step / 4.0, -1.0),
            (0.0, 0.0),
        ] {
            assert_eq!(
                ValueRounding::Bf16.apply(value).to_bits(),
                f32::to_bits(rounded),
                "{value}"
            );
            assert_eq!(ValueRounding::None.apply(value).to_bits(), value.to_bits());
        }
        // Every normalized value of the vendor recipe is a BF16 value
        // once rounded, and the gray pad (127) is not exact before.
        let table = normalized(CenteredPadRecipe {
            value_rounding: ValueRounding::Bf16,
            ..recipe()
        });
        assert!(table.iter().flatten().all(|v| v.to_bits() & 0xffff == 0));
        assert_ne!(normalized(recipe())[0][127].to_bits() & 0xffff, 0);
    }

    #[test]
    fn recipes_refuse_what_cannot_run() {
        check_centered_pad_recipe(recipe()).unwrap();
        for bad in [
            CenteredPadRecipe {
                patch: 0,
                ..recipe()
            },
            CenteredPadRecipe {
                downsample: 0,
                ..recipe()
            },
            CenteredPadRecipe {
                maximum_tokens: 3,
                ..recipe()
            },
            CenteredPadRecipe {
                value_divisor: 0.0,
                ..recipe()
            },
            CenteredPadRecipe {
                std: [0.5, 0.0, 0.5],
                ..recipe()
            },
            CenteredPadRecipe {
                mean: [f32::NAN, 0.5, 0.5],
                ..recipe()
            },
        ] {
            assert!(check_centered_pad_recipe(bad).is_err(), "{bad:?}");
        }
    }
}
