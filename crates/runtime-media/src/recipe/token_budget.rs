//! `image.zero-padded-token-budget.v1`: an image resized at its own aspect
//! ratio into a canvas of whole merge blocks whose token count lies within a
//! budget, then zero padded on the right and bottom.
//!
//! A token is one block of `merge x merge` patches, so the canvas sides are
//! multiples of `f = patch * merge`:
//! - Each side is aligned up to a multiple of `f`.
//! - Below `minimum_tokens`, each side `x` becomes `align(ceil(x * s))` with
//!   `s = sqrt(minimum_tokens * f^2 / (h * w))` in f64.
//! - Above `maximum_tokens`, a binary search over content heights `c` in
//!   `1..=h` keeps the last `(align(c), align(max(1, floor(w * c / h))))`
//!   within the budget, starting from `(f, f)`. A source wider than
//!   `maximum_tokens * f` times its height has no candidate within the
//!   budget and keeps that one-block fallback, below `minimum_tokens`.
//!
//! The content is `max(1, min(side, floor(x * s)))` per side, with
//! `s = min(H / h, W / w)` in f64, capped at 1 once the source has at least
//! `minimum_tokens * f^2` pixels: smaller images may be enlarged to fill the
//! canvas, larger ones only shrink. A changed size is resampled with the
//! antialiased bicubic of [`super::resample`]; the 8-bit canvas, padding
//! included, is normalized as `(value - mean * s) / (std * s)` in f32 with
//! `s = f32(1 / value_scale)`, and every temporal patch repeats the image.
use super::canvas::{self, CanvasLayout, CanvasRows};
use super::{CANVAS_HASH_CHUNK_BYTES, PatchGrid, folded_statistics, resample};
use crate::core::try_filled;
use crate::{MediaError, MediaLimits, MediaResult, RgbImage};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenBudgetRecipe {
    pub patch: u32,
    pub temporal_patch: u32,
    pub merge: u32,
    pub minimum_tokens: u64,
    pub maximum_tokens: u64,
    pub value_scale: f64,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl TokenBudgetRecipe {
    /// The canvas alignment `patch * merge`, checked with the token bounds.
    fn block(&self) -> MediaResult<u64> {
        let block = u64::from(self.patch) * u64::from(self.merge);
        if block == 0
            || self.temporal_patch == 0
            || self.minimum_tokens == 0
            || self.maximum_tokens < self.minimum_tokens
            || block
                .checked_mul(block)
                .and_then(|area| area.checked_mul(self.maximum_tokens))
                .is_none_or(|pixels| pixels > EXACT)
        {
            return Err(MediaError::new("invalid token budget recipe"));
        }
        Ok(block)
    }
}

/// Integers up to this convert to f64 exactly.
const EXACT: u64 = 1 << 53;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PaddedSize {
    /// Height and width of the padded canvas, in whole merge blocks.
    pub canvas: (u32, u32),
    /// Height and width of the resized image at the canvas's top left.
    pub content: (u32, u32),
}

/// The canvas and content sizes of a `height x width` source.
pub fn padded_size(
    height: u32,
    width: u32,
    recipe: TokenBudgetRecipe,
    limits: MediaLimits,
) -> MediaResult<PaddedSize> {
    limits.check_image(width, height)?;
    let f = recipe.block()?;
    let (h, w) = (u64::from(height), u64::from(width));
    if h * w > EXACT {
        return Err(MediaError::new(
            "image too large for the token budget recipe",
        ));
    }
    let align = |side: u64| side.div_ceil(f) * f;
    let tokens = |h: u64, w: u64| (h / f) * (w / f);
    let (mut canvas_h, mut canvas_w) = (align(h), align(w));
    if tokens(canvas_h, canvas_w) < recipe.minimum_tokens {
        let scale = ((recipe.minimum_tokens * f * f) as f64 / (h * w) as f64).sqrt();
        canvas_h = align(((h as f64 * scale).ceil() as u64).max(1));
        canvas_w = align(((w as f64 * scale).ceil() as u64).max(1));
    }
    if tokens(canvas_h, canvas_w) > recipe.maximum_tokens {
        (canvas_h, canvas_w) = (f, f);
        let (mut low, mut high) = (1, h);
        while low <= high {
            let content = (low + high) / 2;
            let candidate = (
                align(content),
                align((((w * content) as f64 / h as f64).floor() as u64).max(1)),
            );
            if tokens(candidate.0, candidate.1) <= recipe.maximum_tokens {
                (canvas_h, canvas_w) = candidate;
                low = content + 1;
            } else {
                high = content - 1;
            }
        }
    }
    let mut scale = (canvas_h as f64 / h as f64).min(canvas_w as f64 / w as f64);
    if h * w >= recipe.minimum_tokens * f * f {
        scale = scale.min(1.0);
    }
    let content = |side: u64, canvas: u64| ((side as f64 * scale).floor() as u64).clamp(1, canvas);
    let size = |side: u64| {
        u32::try_from(side).map_err(|_| MediaError::new("image resize geometry invalid"))
    };
    let padded = PaddedSize {
        canvas: (size(canvas_h)?, size(canvas_w)?),
        content: (size(content(h, canvas_h))?, size(content(w, canvas_w))?),
    };
    limits.check_image(padded.canvas.1, padded.canvas.0)?;
    Ok(padded)
}

/// The token budgets [`check_token_budget_recipe`] accepts, with
/// [`TOKEN_BUDGET_GEOMETRY`]: those whose canvases the tests prove
/// [`padded_emits_size`] admits.
pub const TOKEN_BUDGET_MINIMUM_TOKENS: std::ops::RangeInclusive<u64> = 1..=16;
pub const TOKEN_BUDGET_MAXIMUM_TOKENS: std::ops::RangeInclusive<u64> = 256..=8000;
/// The only accepted `(patch, merge, temporal_patch)`: 28-pixel blocks.
pub const TOKEN_BUDGET_GEOMETRY: (u32, u32, u32) = (14, 2, 2);

/// Refuses a recipe [`padded_size`] or [`padded_token_budget_grid`] cannot
/// run, or outside the domain where [`padded_emits_size`] is shown to admit
/// every canvas: 28-pixel blocks, a minimum of at most 16 tokens and a
/// maximum of 256 to 8,000. There, a source below the minimum has both
/// sides under `16 * 28` pixels, and the tests show for every such source
/// that its upscaled canvas lies within 256 tokens, so it is never searched.
/// A searched source is above the maximum at its own height, so the search
/// stops at a content height `c` below it with `c + 1` over the budget; one
/// more row at most doubles the height blocks and quadruples the width
/// blocks, so the canvas holds over `maximum_tokens / 8 >= 32` tokens,
/// unless no height fits (the one-block fallback). Other budgets and block
/// sizes can search below the minimum to a canvas that is not its own
/// (patch 100 and merge 2 at 2 to 16 tokens pad 1 x 200 to 200 x 200).
pub fn check_token_budget_recipe(recipe: TokenBudgetRecipe) -> MediaResult<()> {
    recipe.block()?;
    if (recipe.patch, recipe.merge, recipe.temporal_patch) != TOKEN_BUDGET_GEOMETRY {
        return Err(MediaError::new(
            "the token budget recipe supports patch 14, merge 2 and temporal_patch 2 only",
        ));
    }
    if !TOKEN_BUDGET_MINIMUM_TOKENS.contains(&recipe.minimum_tokens)
        || !TOKEN_BUDGET_MAXIMUM_TOKENS.contains(&recipe.maximum_tokens)
    {
        return Err(MediaError::new(
            "the token budget recipe supports minimum_tokens 1 to 16 and maximum_tokens 256 to \
             8000 only",
        ));
    }
    if !(recipe.value_scale.is_finite() && recipe.value_scale > 0.0) {
        return Err(MediaError::new("invalid image normalization"));
    }
    folded_statistics(recipe.value_scale, recipe.mean, recipe.std).map(|_| ())
}

/// Whether the recipe pads some source to exactly `height x width`, for a
/// recipe [`check_token_budget_recipe`] accepts. A
/// canvas within the budget is aligned with its tokens, so a source of that
/// size keeps it. The one exception is the one-block fallback of sources
/// too wide for any candidate, below the minimum when that is above one
/// token: it is emitted when the limits admit such a source, and do so
/// exactly when they admit the narrowest, one pixel high and
/// `maximum_tokens * f + 1` wide (any other is larger in both pixels and
/// aspect ratio).
pub fn padded_emits_size(
    height: u32,
    width: u32,
    recipe: TokenBudgetRecipe,
    limits: MediaLimits,
) -> bool {
    let own = |height, width| {
        padded_size(height, width, recipe, limits).is_ok_and(|size| size.canvas == (height, width))
    };
    if own(height, width) {
        return true;
    }
    let Ok(block) = recipe.block() else {
        return false;
    };
    let fallback = u32::try_from(block).ok() == Some(height) && height == width;
    fallback
        && recipe
            .maximum_tokens
            .checked_mul(block)
            .and_then(|wide| u32::try_from(wide + 1).ok())
            .is_some_and(|wide| {
                padded_size(1, wide, recipe, limits)
                    .is_ok_and(|size| size.canvas == (height, width))
            })
}

/// Patch rows of one image repeated over the temporal patch.
pub fn padded_token_budget_grid(
    image: &RgbImage,
    recipe: TokenBudgetRecipe,
    limits: MediaLimits,
) -> MediaResult<PatchGrid> {
    padded_token_budget_canvas(image, recipe, limits)?.into_grid()
}

/// The 8-bit canvas [`padded_token_budget_grid`] cuts its rows from: the
/// resized content at the top left and zero elsewhere, whose value
/// `(0 - mean * s) / (std * s)` is the padding's.
pub fn padded_token_budget_canvas(
    image: &RgbImage,
    recipe: TokenBudgetRecipe,
    limits: MediaLimits,
) -> MediaResult<CanvasRows> {
    let frames = recipe.temporal_patch as usize;
    if frames == 0 || frames > limits.maximum_frames {
        return Err(MediaError::new("invalid frame or patch count"));
    }
    let PaddedSize {
        canvas: (h, w),
        content: (content_h, content_w),
    } = padded_size(image.height, image.width, recipe, limits)?;
    let (mean, std) = folded_statistics(recipe.value_scale, recipe.mean, recipe.std)?;
    let block = recipe.block()?;
    let tokens = u64::from(h) / block * (u64::from(w) / block);
    limits.check_tokens(tokens)?;
    let resized = resample::bicubic_antialias(image, content_h, content_w)?;
    let (w_us, content_w) = (w as usize, content_w as usize);
    let mut samples = try_filled(h as usize * w_us * 3, 0_u8)?;
    for (y, row) in resized.chunks_exact(content_w * 3).enumerate() {
        samples[y * w_us * 3..(y * w_us + content_w) * 3].copy_from_slice(row);
    }
    drop(resized);
    CanvasRows::new(
        h,
        w,
        samples,
        canvas::value_table(|c, v| (f32::from(v) - mean[c]) / std[c]),
        CanvasLayout::MergedPatches {
            patch: recipe.patch,
            temporal_patch: recipe.temporal_patch,
            merge: recipe.merge,
        },
        tokens,
    )
}

/// The recipe's share of the host bytes a [`padded_token_budget_canvas`]
/// preparation holds, beside the decoder's own: the resampler's weights for
/// both passes, its horizontal pass and output, the canvas, the value tables
/// and the chunk its payload is hashed through.
pub fn padded_canvas_preparation_bytes(
    width: u32,
    height: u32,
    recipe: TokenBudgetRecipe,
    limits: MediaLimits,
) -> MediaResult<u64> {
    let size = padded_size(height, width, recipe, limits)?;
    let (content_h, content_w) = size.content;
    let content = u64::from(content_h) * u64::from(content_w);
    let canvas = u64::from(size.canvas.0) * u64::from(size.canvas.1);
    [
        Some(resample::coefficient_bytes(width, content_w)),
        Some(resample::coefficient_bytes(height, content_h)),
        u64::from(height).checked_mul(u64::from(content_w) * 3),
        content.checked_mul(3),
        canvas.checked_mul(3),
        Some(3 * 256 * 4 * 2),
        Some(
            CANVAS_HASH_CHUNK_BYTES as u64
                + 3 * 4 * u64::from(recipe.temporal_patch) * u64::from(recipe.patch).pow(2),
        ),
    ]
    .into_iter()
    .try_fold(0_u64, |sum, part| {
        part.and_then(|part| sum.checked_add(part))
    })
    .ok_or_else(|| MediaError::new("image preparation size overflow"))
}

/// A conservative bound on the host bytes an image of `width` x `height`
/// holds while it decodes and becomes patch rows with
/// [`padded_token_budget_grid`], in the terms of
/// [`super::image_preparation_bytes`]: the decoder's buffers, the
/// resampler's horizontal pass and output, the normalized F32 canvas, the
/// F32 patch rows and a payload of up to 4 bytes per value.
pub fn padded_preparation_bytes(
    width: u32,
    height: u32,
    recipe: TokenBudgetRecipe,
    limits: MediaLimits,
) -> MediaResult<u64> {
    let size = padded_size(height, width, recipe, limits)?;
    let source = u64::from(width) * u64::from(height);
    let content = u64::from(size.content.0) * u64::from(size.content.1);
    let canvas = u64::from(size.canvas.0) * u64::from(size.canvas.1);
    let values = canvas
        .checked_mul(3)
        .and_then(|values| values.checked_mul(u64::from(recipe.temporal_patch)));
    [
        source.checked_mul(super::DECODE_BYTES_PER_PIXEL),
        u64::from(height).checked_mul(u64::from(size.content.1) * 3),
        content.checked_mul(3),
        canvas.checked_mul(3 * 4),
        values.and_then(|values| values.checked_mul(4 + 4)),
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

    /// The F32 implementation the canvas replaced, kept as its oracle.
    fn planar_grid(
        image: &RgbImage,
        recipe: TokenBudgetRecipe,
        limits: MediaLimits,
    ) -> MediaResult<PatchGrid> {
        let frames = recipe.temporal_patch as usize;
        let PaddedSize {
            canvas: (h, w),
            content: (content_h, content_w),
        } = padded_size(image.height, image.width, recipe, limits)?;
        let (mean, std) = folded_statistics(recipe.value_scale, recipe.mean, recipe.std)?;
        let resized = resample::bicubic_antialias(image, content_h, content_w)?;
        let (h_us, w_us, content_w) = (h as usize, w as usize, content_w as usize);
        let plane = h_us * w_us;
        let mut out = try_filled(3 * plane, 0.0)?;
        for c in 0..3 {
            out[c * plane..(c + 1) * plane].fill((0.0 - mean[c]) / std[c]);
        }
        for (y, row) in resized.chunks_exact(content_w * 3).enumerate() {
            for (x, pixel) in row.chunks_exact(3).enumerate() {
                for c in 0..3 {
                    out[c * plane + y * w_us + x] = (f32::from(pixel[c]) - mean[c]) / std[c];
                }
            }
        }
        let patching = super::super::Patching {
            patch: recipe.patch,
            temporal_patch: recipe.temporal_patch,
            merge: recipe.merge,
        };
        super::super::patch_rows(&vec![out.as_slice(); frames], h, w, patching, limits)
    }

    #[test]
    fn canvas_rows_match_the_planar_rows_bit_for_bit() {
        let recipe = TokenBudgetRecipe {
            patch: 14,
            temporal_patch: 2,
            merge: 2,
            minimum_tokens: 4,
            maximum_tokens: 256,
            value_scale: 0.00392156862745098,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.26130258, 0.27577711],
        };
        let limits = MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1 << 26,
            maximum_audio_frames: 1,
            maximum_frames: 2,
            maximum_tokens: 1 << 20,
            maximum_aspect_ratio: 200.0,
        };
        let mut seed = 0x9e37_79b9_u32;
        for (width, height) in [
            (1, 1),
            (28, 28),
            (61, 17),
            (300, 200),
            (17, 400),
            (900, 640),
        ] {
            let data = (0..width * height * 3)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as u8
                })
                .collect();
            let image = RgbImage::new(width, height, data, limits).unwrap();
            let canvas = padded_token_budget_canvas(&image, recipe, limits).unwrap();
            let planar = planar_grid(&image, recipe, limits).unwrap();
            let rows = canvas.rows().unwrap();
            assert_eq!(
                rows.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                planar.rows.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{width} x {height}"
            );
            assert_eq!(canvas.into_grid().unwrap(), planar, "{width} x {height}");
        }
    }

    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1 << 26,
            maximum_audio_frames: 1,
            maximum_frames: 2,
            maximum_tokens: 1 << 20,
            maximum_aspect_ratio: 200.0,
        }
    }

    fn recipe() -> TokenBudgetRecipe {
        TokenBudgetRecipe {
            patch: 2,
            temporal_patch: 2,
            merge: 2,
            minimum_tokens: 4,
            maximum_tokens: 64,
            value_scale: 1.0,
            mean: [0.0; 3],
            std: [1.0; 3],
        }
    }

    fn size(height: u32, width: u32) -> ((u32, u32), (u32, u32)) {
        let size = padded_size(height, width, recipe(), limits()).unwrap();
        (size.canvas, size.content)
    }

    #[test]
    fn sizes_align_upscale_and_search_the_budget() {
        // Aligned within the budget: unchanged.
        assert_eq!(size(8, 12), ((8, 12), (8, 12)));
        // Padded, never enlarged at or above 4 * 16 = 64 pixels.
        assert_eq!(size(9, 10), ((12, 12), (9, 10)));
        // Below the minimum: s = sqrt(64 / 6) scales 2 x 3 to 7 x 10, aligned
        // to 8 x 12, and the content fills the canvas at min(4, 4).
        assert_eq!(size(2, 3), ((8, 12), (8, 12)));
        // Small but within the budget: enlarged by min(8 / 7, 12 / 9).
        assert_eq!(size(7, 9), ((8, 12), (8, 10)));
        // Above 64 tokens: the search keeps content 19 x 47, aligned to
        // 20 x 48 at 60 tokens (20 x 50 aligns to 20 x 52 at 65), and the
        // image shrinks by min(20 / 30, 48 / 75).
        assert_eq!(size(30, 75), ((20, 48), (19, 48)));
        // Nothing fits: the canvas stays one block.
        let tiny = TokenBudgetRecipe {
            maximum_tokens: 1,
            minimum_tokens: 1,
            ..recipe()
        };
        let size = padded_size(4, 400, tiny, limits()).unwrap();
        assert_eq!(
            size,
            PaddedSize {
                canvas: (4, 4),
                content: (1, 4)
            }
        );
    }

    /// Every canvas of a sweep of sources is its own canvas, and sizes
    /// outside the budget or not aligned are not emitted.
    #[test]
    fn canvases_are_their_own_canvas() {
        for height in 1..120 {
            for width in 1..120 {
                let (canvas, _) = size(height, width);
                assert!(
                    padded_emits_size(canvas.0, canvas.1, recipe(), limits()),
                    "{height}x{width} -> {canvas:?}"
                );
            }
        }
        for [h, w] in [[4, 4], [8, 6], [4, 260], [36, 36]] {
            assert!(!padded_emits_size(h, w, recipe(), limits()), "{h}x{w}");
        }
    }

    fn shipped() -> TokenBudgetRecipe {
        TokenBudgetRecipe {
            patch: 14,
            merge: 2,
            minimum_tokens: 16,
            maximum_tokens: 8000,
            ..recipe()
        }
    }

    /// Every source below the minimum, for each accepted minimum: both
    /// sides under `minimum_tokens * 28` pixels. Its canvas holds the
    /// minimum to 256 tokens, the least accepted maximum, so it is never
    /// searched and is its own canvas at every accepted maximum.
    #[test]
    fn upscaled_canvases_stay_within_every_accepted_budget() {
        let limits = MediaLimits {
            maximum_aspect_ratio: 1000.0,
            ..limits()
        };
        for minimum_tokens in TOKEN_BUDGET_MINIMUM_TOKENS {
            let bound = |maximum_tokens| TokenBudgetRecipe {
                minimum_tokens,
                maximum_tokens,
                ..shipped()
            };
            let side = minimum_tokens as u32 * 28;
            let mut upscaled = 0;
            for height in 1..side {
                for width in 1..side {
                    if u64::from(height.div_ceil(28) * width.div_ceil(28)) >= minimum_tokens {
                        continue;
                    }
                    let (h, w) = padded_size(height, width, bound(256), limits)
                        .unwrap()
                        .canvas;
                    let tokens = u64::from(h / 28) * u64::from(w / 28);
                    assert!(
                        (minimum_tokens..=256).contains(&tokens),
                        "{minimum_tokens}: {height}x{width} -> {h}x{w}"
                    );
                    for maximum_tokens in [256, 8000] {
                        assert!(padded_emits_size(h, w, bound(maximum_tokens), limits));
                    }
                    upscaled += 1;
                }
            }
            assert!(upscaled > 0 || minimum_tokens == 1);
        }
    }

    /// Searched sources at the least and the shipped maximum, with the
    /// least and largest minimum: every canvas is admitted. The sweep
    /// covers every height and width to 700, every 37th to 6,000, and strips
    /// up to and past the width where the one-block fallback starts.
    #[test]
    fn searched_canvases_are_admitted_at_accepted_budgets() {
        let limits = MediaLimits {
            maximum_aspect_ratio: 300_000.0,
            ..limits()
        };
        for (minimum_tokens, maximum_tokens) in [(1, 256), (16, 256), (1, 8000), (16, 8000)] {
            let recipe = TokenBudgetRecipe {
                minimum_tokens,
                maximum_tokens,
                ..shipped()
            };
            check_token_budget_recipe(recipe).unwrap();
            let mut sources: Vec<(u32, u32)> = Vec::new();
            for height in 1..=700 {
                sources.extend((1..=700).map(|width| (height, width)));
            }
            sources.extend((1..=162).flat_map(|h| (1..=162).map(move |w| (h * 37, w * 37))));
            let widest = maximum_tokens as u32 * 28 + 2;
            for height in 1..=4 {
                sources.extend((1..=widest / 97).map(|step| (height, step * 97 * height)));
                sources.extend([widest - 2, widest - 1, widest].map(|w| (height, w * height)));
            }
            let mut searched = 0;
            for (height, width) in sources {
                for (height, width) in [(height, width), (width, height)] {
                    let Ok(size) = padded_size(height, width, recipe, limits) else {
                        continue;
                    };
                    let aligned = u64::from(height.div_ceil(28)) * u64::from(width.div_ceil(28));
                    searched += usize::from(aligned > maximum_tokens);
                    let (h, w) = size.canvas;
                    assert!(
                        padded_emits_size(h, w, recipe, limits),
                        "{minimum_tokens}/{maximum_tokens}: {height}x{width} -> {h}x{w}"
                    );
                }
            }
            assert!(searched > 10_000, "{maximum_tokens}: {searched}");
        }
    }

    /// Recipes outside the verified domain are refused at load, the
    /// reviewed counterexamples among them: patch 100 at 2 to 16 tokens pads
    /// 1 x 200 to 200 x 200, patch 50 at 16 to 128 tokens pads 1 x 1500 to
    /// 100 x 1500, and 16 to 16 tokens pads 21 x 28 to 28 x 28; none is its
    /// own canvas.
    #[test]
    fn recipes_outside_the_verified_domain_are_refused() {
        check_token_budget_recipe(shipped()).unwrap();
        let wide = |ratio| MediaLimits {
            maximum_aspect_ratio: ratio,
            ..limits()
        };
        for (recipe, source, canvas, ratio, message) in [
            (
                TokenBudgetRecipe {
                    patch: 100,
                    minimum_tokens: 2,
                    maximum_tokens: 16,
                    ..shipped()
                },
                (1, 200),
                (200, 200),
                200.0,
                "patch 14, merge 2",
            ),
            (
                TokenBudgetRecipe {
                    patch: 50,
                    minimum_tokens: 16,
                    maximum_tokens: 128,
                    ..shipped()
                },
                (1, 1500),
                (100, 1500),
                2000.0,
                "patch 14, merge 2",
            ),
            (
                TokenBudgetRecipe {
                    maximum_tokens: 16,
                    ..shipped()
                },
                (21, 28),
                (28, 28),
                200.0,
                "maximum_tokens 256 to 8000",
            ),
        ] {
            let size = padded_size(source.0, source.1, recipe, wide(ratio)).unwrap();
            assert_eq!(size.canvas, canvas, "{recipe:?}");
            assert!(!padded_emits_size(canvas.0, canvas.1, recipe, wide(ratio)));
            let error = check_token_budget_recipe(recipe).unwrap_err();
            assert!(error.to_string().contains(message), "{error}");
        }
        for recipe in [
            TokenBudgetRecipe {
                temporal_patch: 1,
                ..shipped()
            },
            TokenBudgetRecipe {
                minimum_tokens: 17,
                ..shipped()
            },
            TokenBudgetRecipe {
                maximum_tokens: 255,
                ..shipped()
            },
            TokenBudgetRecipe {
                maximum_tokens: 8001,
                ..shipped()
            },
        ] {
            assert!(check_token_budget_recipe(recipe).is_err(), "{recipe:?}");
        }
    }

    /// A source over `maximum_tokens * f = 256` times wider than high
    /// keeps the one-block fallback, a 1-token canvas below the 4-token
    /// minimum; it is emitted exactly when the limits admit such a source.
    #[test]
    fn the_one_block_fallback_is_emitted_when_a_source_reaches_it() {
        let wide = |ratio| MediaLimits {
            maximum_aspect_ratio: ratio,
            ..limits()
        };
        let reach = |width| padded_size(1, width, recipe(), wide(300.0)).unwrap().canvas;
        assert_eq!(reach(256), (4, 256));
        assert_eq!(reach(257), (4, 4));
        assert!(padded_emits_size(4, 4, recipe(), wide(257.0)));
        assert!(!padded_emits_size(4, 4, recipe(), wide(256.9)));
        assert!(!padded_emits_size(4, 4, recipe(), limits()));
    }

    /// Geometry whose products leave u64 is refused, not overflowed.
    #[test]
    fn oversized_geometry_is_refused() {
        let huge = TokenBudgetRecipe {
            patch: 65_536,
            merge: 65_536,
            ..recipe()
        };
        assert!(check_token_budget_recipe(huge).is_err());
        assert!(!padded_emits_size(8, 8, huge, limits()));
        let wide = TokenBudgetRecipe {
            patch: 1,
            merge: 1,
            temporal_patch: 1024,
            minimum_tokens: 1,
            maximum_tokens: 1 << 53,
            ..recipe()
        };
        let permissive = MediaLimits {
            maximum_pixels: u64::MAX,
            ..limits()
        };
        let error = padded_preparation_bytes(1 << 27, 1 << 26, wide, permissive).unwrap_err();
        assert!(error.to_string().contains("overflow"), "{error}");
    }

    #[test]
    fn invalid_recipes_are_refused() {
        for bad in [
            TokenBudgetRecipe {
                merge: 0,
                ..recipe()
            },
            TokenBudgetRecipe {
                temporal_patch: 0,
                ..recipe()
            },
            TokenBudgetRecipe {
                minimum_tokens: 0,
                ..recipe()
            },
            TokenBudgetRecipe {
                minimum_tokens: 65,
                ..recipe()
            },
            TokenBudgetRecipe {
                maximum_tokens: u64::MAX,
                ..recipe()
            },
        ] {
            assert!(padded_size(8, 8, bad, limits()).is_err(), "{bad:?}");
            assert!(check_token_budget_recipe(bad).is_err(), "{bad:?}");
        }
        for value_scale in [0.0, -1.0, f64::INFINITY] {
            let unscaled = TokenBudgetRecipe {
                value_scale,
                ..shipped()
            };
            assert!(
                check_token_budget_recipe(unscaled).is_err(),
                "{value_scale}"
            );
        }
    }

    #[test]
    fn padding_normalizes_zero_and_rows_repeat_the_image() {
        // A 2 x 3 image pads to 2 x 4: the second block's right column is
        // padding, normalized from 0.
        let recipe = TokenBudgetRecipe {
            patch: 1,
            minimum_tokens: 1,
            mean: [1.0, 2.0, 3.0],
            std: [2.0; 3],
            ..recipe()
        };
        let image = RgbImage::new(3, 2, [9, 10, 11].repeat(6), limits()).unwrap();
        let grid = padded_token_budget_grid(&image, recipe, limits()).unwrap();
        assert_eq!((grid.grid, grid.tokens, grid.columns), ([1, 2, 4], 2, 6));
        let content = [4.0; 6];
        let padding = [-0.5, -0.5, -1.0, -1.0, -1.5, -1.5];
        let rows: Vec<_> = grid.rows.chunks_exact(6).collect();
        assert!(rows[..4].iter().all(|row| *row == content));
        assert_eq!(rows[4..], [&content, &padding, &content, &padding]);
    }
}
