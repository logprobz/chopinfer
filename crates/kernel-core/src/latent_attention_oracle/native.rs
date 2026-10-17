//! Native latent storage and the attention stages over a window and pages.

use super::*;
use crate::codec::{SmallFloat, decode_ue8m0_f64, f32_to_bf16};
use crate::latent_attention::{LATENT_WINDOW_SLOTS, LatentWindowStorage};

fn native_geometry(kv: LatentKvFormat, width: usize) -> Result<(usize, usize), Refusal> {
    if width == 0 || !width.is_multiple_of(128) {
        return Err(Refusal("native latent width must be a multiple of 128"));
    }
    match kv {
        LatentKvFormat::Fp8E4M3Ue8M0Block32 => Ok((width, width / 32)),
        LatentKvFormat::Fp4E2M1E4M3Block16 => Ok((width / 2, width / 16)),
        _ => Err(Refusal("expected a native latent format")),
    }
}

pub(super) fn encode_native_row(kv: LatentKvFormat, row: &[u16]) -> Result<EncodedLatent, Refusal> {
    let (bytes, blocks) = native_geometry(kv, row.len())?;
    let fp8 = kv == LatentKvFormat::Fp8E4M3Ue8M0Block32;
    let block = row.len() / blocks;
    let format = if fp8 {
        SmallFloat::E4M3
    } else {
        SmallFloat::E2M1
    };
    let limit = format.max_finite() as f32;
    let mut values = vec![0; bytes];
    let mut scale_bytes = Vec::with_capacity(blocks);
    for (group, input) in row.chunks_exact(block).enumerate() {
        let input = input.iter().map(|&x| bf16_to_f32(x)).collect::<Vec<_>>();
        if input.iter().any(|x| !x.is_finite()) {
            return Err(Refusal("native latent input must be finite"));
        }
        let amax = input.iter().map(|x| x.abs()).fold(0_f32, f32::max);
        let (scale_code, scale) = if fp8 {
            // The reference multiplies by the rounded reciprocal before
            // taking ceil(log2), including at power-of-two boundaries.
            let ratio = amax.max(1e-4) * (1_f32 / 448.0);
            let bits = ratio.to_bits();
            let exponent = ((bits >> 23) & 255) as i32 - 127 + i32::from(bits & 0x7fffff != 0);
            let code = u8::try_from(exponent + 127)
                .ok()
                .filter(|&x| x != 255)
                .ok_or(Refusal("native latent scale is outside UE8M0"))?;
            (code, decode_ue8m0_f64(code) as f32)
        } else {
            let code = SmallFloat::E4M3.encode_f32(amax.max(6.0 * 2_f32.powi(-9)) / 6.0);
            (code, SmallFloat::E4M3.decode_f32(code))
        };
        scale_bytes.push(scale_code);
        for (column, &x) in input.iter().enumerate() {
            let code = format.encode_f32((x / scale).clamp(-limit, limit));
            let index = group * block + column;
            if fp8 {
                values[index] = code;
            } else {
                values[index / 2] |= code << (4 * (index % 2));
            }
        }
    }
    let encoded = EncodedLatent {
        values,
        scales: Vec::new(),
        scale_bytes,
    };
    let mut page = encoded.values.clone();
    page.extend_from_slice(&encoded.scale_bytes);
    decode_native_page(kv, row.len(), 1, &page, 0)?;
    Ok(encoded)
}

/// Decode a native code plane followed by its scale plane, with no rounding.
/// Refuse invalid scales, NaN codes and products not exactly finite BF16.
pub fn decode_native_page(
    kv: LatentKvFormat,
    width: usize,
    slots: usize,
    page: &[u8],
    slot: usize,
) -> Result<Vec<f64>, Refusal> {
    let (bytes, blocks) = native_geometry(kv, width)?;
    let size = bytes
        .checked_add(blocks)
        .and_then(|row| slots.checked_mul(row))
        .ok_or(Refusal("native latent page size overflows"))?;
    if slot >= slots || page.len() < size {
        return Err(Refusal("native latent row outside storage"));
    }
    let codes = &page[slot * bytes..(slot + 1) * bytes];
    let scales = &page[slots * bytes + slot * blocks..slots * bytes + (slot + 1) * blocks];
    let fp8 = kv == LatentKvFormat::Fp8E4M3Ue8M0Block32;
    let mut decoded = Vec::with_capacity(width);
    for column in 0..width {
        let (code, scale) = if fp8 {
            (
                SmallFloat::E4M3.decode_f64(codes[column]),
                decode_ue8m0_f64(scales[column / 32]),
            )
        } else {
            (
                SmallFloat::E2M1.decode_f64((codes[column / 2] >> (4 * (column % 2))) & 15),
                SmallFloat::E4M3.decode_f64(scales[column / 16]),
            )
        };
        let value = code * scale;
        if !scale.is_finite()
            || scale <= 0.0
            || !value.is_finite()
            || f64::from(bf16_to_f32(f32_to_bf16(value as f32))) != value
        {
            return Err(Refusal(
                "native latent code and scale must decode exactly to finite BF16",
            ));
        }
        decoded.push(value);
    }
    Ok(decoded)
}

/// A source's row-major index lists. Only `budget` entries per row are read.
/// -1 is a hole; any other negative entry or repeated position is refused.
#[derive(Clone, Copy, Debug)]
pub struct LatentKeyLists<'a> {
    pub indices: &'a [i32],
    pub stride: usize,
    pub budget: usize,
}

impl LatentKeyLists<'_> {
    fn validate(self, rows: usize) -> Result<(), Refusal> {
        if self.budget > self.stride || rows.checked_mul(self.stride) != Some(self.indices.len()) {
            return Err(Refusal("latent source key list shape mismatch"));
        }
        Ok(())
    }

    fn row(self, row: usize) -> Result<Vec<usize>, Refusal> {
        let mut positions = Vec::new();
        for &p in &self.indices[row * self.stride..row * self.stride + self.budget] {
            if p == -1 {
                continue;
            }
            let p = usize::try_from(p).map_err(|_| Refusal("latent key list entry below -1"))?;
            if positions.contains(&p) {
                return Err(Refusal("latent key list repeats a position"));
            }
            positions.push(p);
        }
        Ok(positions)
    }
}

/// Each sequence holds all its native FP8 code rows followed by scale rows.
/// Attention only reads this source; ring updates and verify rollback are
/// explicit caller-owned operations. The lists alone specify visibility.
#[derive(Clone, Copy, Debug)]
pub struct LatentWindowInput<'a> {
    pub pool: &'a [u8],
    pub sequence_stride: usize,
    pub storage: LatentWindowStorage,
    pub lists: LatentKeyLists<'a>,
}

#[derive(Clone, Copy, Debug)]
pub struct LatentCompressedInput<'a> {
    pub pool: &'a [u8],
    pub paging: LatentPaging<'a>,
    pub lists: LatentKeyLists<'a>,
}

/// Inputs to the native attention stage. Q is post-RoPE BF16 decoded to
/// f64; stored keys were rotated before quantization. Sink logits are F32.
#[derive(Clone, Copy, Debug)]
pub struct LatentSourcesInput<'a> {
    pub q: &'a [f64],
    pub row_batch: &'a [i32],
    pub window: LatentWindowInput<'a>,
    pub compressed: Option<LatentCompressedInput<'a>>,
    pub sinks: Option<&'a [f32]>,
}

/// F64 attention over the window list followed by the paged compressed
/// list, with one denominator and one sink. Partials omit the sink even
/// when the query declares it. Normalized results are before BF16 storage
/// and the inverse rotation: gate that stage separately with
/// [`latent_tail_rope`] on the stored BF16 attention result. This preserves
/// the mandatory rounding boundary rather than rotating an unrounded sum.
pub fn latent_attention_with_sources(
    query: LatentAttentionQuery,
    input: LatentSourcesInput<'_>,
) -> Result<LatentAttentionResult, Refusal> {
    query.validate()?;
    let LatentAttentionQuery::Attention {
        heads,
        latent,
        max_keys,
        softmax_scale_bits,
        page_size,
        kv,
        output,
        sink,
        second_fp4_keys,
        capacity,
        ..
    } = query
    else {
        return Err(Refusal("expected a latent attention query"));
    };
    if kv != LatentKvFormat::Fp8E4M3Ue8M0Block32 {
        return Err(Refusal(
            "source-aware latent oracle requires the native FP8 window",
        ));
    }
    let (heads, d, rows) = (heads as usize, latent as usize, input.row_batch.len());
    let key_count = input
        .window
        .lists
        .budget
        .checked_add(input.compressed.map_or(0, |s| s.lists.budget));
    if rows == 0
        || rows > capacity as usize
        || rows.checked_mul(heads).and_then(|n| n.checked_mul(d)) != Some(input.q.len())
        || second_fp4_keys != input.compressed.is_some()
        || sink != input.sinks.is_some()
        || input
            .sinks
            .is_some_and(|s| s.len() != heads || s.iter().any(|v| !v.is_finite()))
        || key_count.is_none_or(|n| n > max_keys as usize)
        || input
            .q
            .iter()
            .any(|&x| !x.is_finite() || f64::from(bf16_to_f32(f32_to_bf16(x as f32))) != x)
    {
        return Err(Refusal("latent source operands do not match the query"));
    }
    input.window.lists.validate(rows)?;
    if let Some(source) = input.compressed {
        source.lists.validate(rows)?;
    }
    let slots = match input.window.storage {
        LatentWindowStorage::Ring128 => LATENT_WINDOW_SLOTS as usize,
        LatentWindowStorage::Staged { slots } => slots as usize,
    };
    let window_bytes = slots
        .checked_mul(kv.token_bytes(latent) as usize)
        .ok_or(Refusal("latent window size overflows"))?;
    if slots == 0 || input.window.sequence_stride < window_bytes {
        return Err(Refusal("latent window stride is too short"));
    }
    let layout = Layout {
        latent: d,
        page_size: page_size as usize,
        kv: LatentKvFormat::Fp4E2M1E4M3Block16,
    };
    if input
        .compressed
        .is_some_and(|s| s.paging.page_stride < layout.page_bytes())
    {
        return Err(Refusal("latent compressed page stride is too short"));
    }
    let empty = OracleOutput {
        values: Vec::new(),
        sum_abs_products: Vec::new(),
        reduction_lengths: Vec::new(),
    };
    let mut normalized = empty.clone();
    let mut partial = LatentPartials {
        maximum: empty.clone(),
        denominator: empty.clone(),
        numerator: empty,
    };
    let push = |out: &mut OracleOutput, value, bound, k: usize| {
        out.values.push(value);
        out.sum_abs_products.push(bound);
        out.reduction_lengths.push(k.max(1));
    };
    for row in 0..rows {
        let mut keys = Vec::new();
        for position in input.window.lists.row(row)? {
            let sequence = usize::try_from(input.row_batch[row])
                .map_err(|_| Refusal("latent sequence is negative"))?;
            let base = sequence
                .checked_mul(input.window.sequence_stride)
                .ok_or(Refusal("latent window offset overflows"))?;
            let end = base
                .checked_add(window_bytes)
                .ok_or(Refusal("latent window offset overflows"))?;
            let page = input
                .window
                .pool
                .get(base..end)
                .ok_or(Refusal("latent window outside storage"))?;
            keys.push(decode_native_page(kv, d, slots, page, position)?);
        }
        if let Some(source) = input.compressed {
            for position in source.lists.row(row)? {
                let (base, slot) = layout.locate(
                    source.paging,
                    source.pool.len(),
                    input.row_batch[row],
                    position as i32,
                )?;
                keys.push(layout.decode(&source.pool[base..base + layout.page_bytes()], slot)?);
            }
        }
        let n = keys.len();
        let spread = if n == 0 { 0.0 } else { d as f64 / n as f64 };
        for head in 0..heads {
            let q = &input.q[(row * heads + head) * d..(row * heads + head + 1) * d];
            let softmax = head_softmax(q, &keys, f64::from(f32::from_bits(softmax_scale_bits)));
            if output == LatentAttentionOutput::Normalized {
                let sink = input.sinks.map(|s| f64::from(s[head]));
                let top = sink.map_or(softmax.maximum, |s| s.max(softmax.maximum));
                let rescale = if n == 0 {
                    0.0
                } else {
                    (softmax.maximum - top).exp()
                };
                let weights = softmax
                    .weights
                    .iter()
                    .map(|w| w * rescale)
                    .collect::<Vec<_>>();
                let mut den = NeumaierSum::default();
                for &w in &weights {
                    den.add(w);
                }
                if let Some(s) = sink {
                    den.add((s - top).exp());
                }
                let den = den.value();
                for column in 0..d {
                    if n == 0 {
                        push(&mut normalized, 0.0, 0.0, 1);
                        continue;
                    }
                    let (mut num, mut abs) = (NeumaierSum::default(), NeumaierSum::default());
                    for (&w, key) in weights.iter().zip(&keys) {
                        num.add(w * key[column]);
                        abs.add((w * key[column]).abs());
                    }
                    let value = num.value() / den;
                    let mut centered = NeumaierSum::default();
                    for ((&w, key), &g) in weights.iter().zip(&keys).zip(&softmax.magnitudes) {
                        centered.add(w * (key[column] - value).abs() * g);
                    }
                    push(
                        &mut normalized,
                        value,
                        (abs.value() + spread * centered.value()) / den,
                        n + usize::from(sink.is_some()),
                    );
                }
                continue;
            }
            let (mut l, mut lb) = (NeumaierSum::default(), NeumaierSum::default());
            for (&w, &g) in softmax.weights.iter().zip(&softmax.magnitudes) {
                l.add(w);
                lb.add(w * (1.0 + spread * g));
            }
            push(
                &mut partial.maximum,
                softmax.maximum,
                softmax.maximum_bound,
                if n == 0 { 1 } else { d },
            );
            push(&mut partial.denominator, l.value(), lb.value(), n);
            for column in 0..d {
                let (mut num, mut bound) = (NeumaierSum::default(), NeumaierSum::default());
                for ((&w, key), &g) in softmax.weights.iter().zip(&keys).zip(&softmax.magnitudes) {
                    num.add(w * key[column]);
                    bound.add((w * key[column]).abs() * (1.0 + spread * g));
                }
                push(&mut partial.numerator, num.value(), bound.value(), n);
            }
        }
    }
    match output {
        LatentAttentionOutput::Partial => Ok(LatentAttentionResult::Partial(partial)),
        LatentAttentionOutput::Normalized => Ok(LatentAttentionResult::Normalized(normalized)),
    }
}

/// Reference of the native softmax's probability rounding, separate from
/// the ideal f64 attention bound. Scores are already scaled; -inf is a
/// masked key. Values are `[scores.len(), width]`. For each 64-key tile,
/// update the running maximum, rescale the old numerator and denominator,
/// cast unnormalized exp(score - max) to BF16 for PV only, and add the
/// unrounded probabilities to the denominator. Add the F32 sink once at
/// the end. Continuous arithmetic here is f64, not a bitwise F32 emulator.
pub fn latent_online_softmax(
    scores: &[f64],
    values: &[f64],
    width: usize,
    sink: f32,
) -> Result<Vec<f64>, Refusal> {
    if width == 0
        || scores.len().checked_mul(width) != Some(values.len())
        || !sink.is_finite()
        || scores
            .iter()
            .any(|s| !s.is_finite() && *s != f64::NEG_INFINITY)
    {
        return Err(Refusal("latent online softmax shape or score mismatch"));
    }
    let mut maximum = f64::NEG_INFINITY;
    let mut denominator = 0.0;
    let mut numerator = vec![0.0; width];
    for (tile, logits) in scores.chunks(64).enumerate() {
        let next = logits.iter().copied().fold(maximum, f64::max);
        if next == f64::NEG_INFINITY {
            continue;
        }
        let rescale = (maximum - next).exp();
        denominator *= rescale;
        for v in &mut numerator {
            *v *= rescale;
        }
        maximum = next;
        for (key, &score) in logits.iter().enumerate() {
            if score == f64::NEG_INFINITY {
                continue;
            }
            let probability = (score - maximum).exp();
            denominator += probability;
            let rounded = f64::from(bf16_to_f32(f32_to_bf16(probability as f32)));
            for (col, sum) in numerator.iter_mut().enumerate() {
                let value = values[(tile * 64 + key) * width + col];
                if !value.is_finite() {
                    return Err(Refusal("latent online softmax value is not finite"));
                }
                *sum += rounded * value;
            }
        }
    }
    if maximum == f64::NEG_INFINITY {
        return Ok(numerator);
    }
    // Rescale after PV, so a large sink cannot affect probability rounding.
    let top = maximum.max(f64::from(sink));
    let scale = (maximum - top).exp();
    let den = denominator * scale + (f64::from(sink) - top).exp();
    Ok(numerator.iter().map(|n| n * scale / den).collect())
}

/// F64 reference for in-place adjacent-pair tail rotation. Values are
/// `[rows, heads, width]`, cos/sin are F32 `[rows, tail/2, 2]`. The caller
/// gates the BF16 store with its half-ulp allowance. Use `inverse = true`
/// on the stored BF16 attention result, never on an unnormalized partial.
/// No frequencies are read when tail is zero; the prefix is copied exactly.
pub fn latent_tail_rope(
    values: &[f64],
    width: usize,
    heads: usize,
    tail: usize,
    cos_sin: &[f32],
    inverse: bool,
) -> Result<OracleOutput, Refusal> {
    let row_width = width
        .checked_mul(heads)
        .filter(|&n| n > 0)
        .ok_or(Refusal("latent tail rotation shape mismatch"))?;
    if values.is_empty()
        || !values.len().is_multiple_of(row_width)
        || !matches!(tail, 0 | 64)
        || tail > width
        || (tail != 0 && cos_sin.len() != values.len() / row_width * tail)
        || (tail != 0 && cos_sin.iter().any(|v| !v.is_finite()))
    {
        return Err(Refusal("latent tail rotation shape mismatch"));
    }
    let mut result = OracleOutput::new(values.to_vec(), vec![0.0; values.len()], 2)?;
    for (i, vector) in values.chunks_exact(width).enumerate() {
        for pair in 0..tail / 2 {
            let at = i * width + width - tail + 2 * pair;
            let freq = (i / heads) * tail + 2 * pair;
            let c = f64::from(cos_sin[freq]);
            let s = f64::from(cos_sin[freq + 1]) * if inverse { -1.0 } else { 1.0 };
            let (x, y) = (
                vector[width - tail + 2 * pair],
                vector[width - tail + 2 * pair + 1],
            );
            result.values[at] = x * c - y * s;
            result.values[at + 1] = x * s + y * c;
            result.sum_abs_products[at] = (x * c).abs() + (y * s).abs();
            result.sum_abs_products[at + 1] = (x * s).abs() + (y * c).abs();
        }
    }
    Ok(result)
}
