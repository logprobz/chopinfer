//! Rotary position embedding as one rotation over explicit choices: the
//! rotated slice of each row, the pairing, the frequency table, the
//! direction and the rounding. Every computed table is a named, versioned
//! rule with explicit arithmetic, including any pinned geometry shortcuts.

use crate::contract::{DType, Refusal, RotaryStyle};
use crate::oracle::bf16_round;

/// The `len` values of each row from `offset` on are rotated as `len / 2`
/// pairs; the rest of the row passes through.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotarySlice {
    pub offset: usize,
    pub len: usize,
}

/// How a computed table turns a position into one angle per pair. Pair `i`
/// of `n` uses frequency exponent `2i / d`; `d` is the slice length, or half
/// of it per axis for `Axes::Axial2d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AngleRule {
    /// `rope.angle-div-pow-f64.v1`: `position / theta.powf(2i / d)`, then f64
    /// sine and cosine.
    DivPowF64,
    /// `rope.angle-mul-neg-pow-f64.v1`: `position * theta.powf(-2i / d)`,
    /// then f64 sine and cosine.
    MulNegPowF64,
    /// `rope.angle-mul-neg-pow-yarn-f64.v1`: the `MulNegPowF64` inverse
    /// frequency blended by YaRN before the position product.
    YarnF64(Yarn),
    /// `crate::hf_rope::REFERENCE`: F32 powf/reciprocal inverse frequencies,
    /// equal to the pinned Torch bits at theta 1e4 or 1e7 for 16 or 32
    /// pairs (read from `hf_rope::INVERSE_BITS` there), an F32 position product, f64
    /// cosine and sine of it, each rounded to F32 and then BF16.
    HfF32AngleBf16CosSin,
    /// `rope.angle-div-pow-f32.v1`: all F32, theta rounded to F32:
    /// `position / theta.powf(2i / d)`, then F32 sine and cosine.
    DivPowF32,
    /// `rope.angle-mul-recip-pow-f32.v1`: all F32, theta rounded to F32: the
    /// inverse frequency `1 / theta.powf(2i / d)`, the position times it,
    /// then F32 sine and cosine.
    MulRecipPowF32,
    /// `rope.angle-mul-recip-f64-pow-f32.v1`: the F32 exponent `2i / d`, an
    /// f64 power of theta rounded to F32, its F32 reciprocal, the F32
    /// position product, then f64 cosine and sine of that angle.
    MulRecipF64PowF32,
}

/// YaRN frequency interpolation in f64 with floored and ceiled correction
/// bounds (theta above 1 and `beta_fast > beta_slow > 0` only, with a
/// nonempty band): `low = max(floor(c(beta_fast)), 0)` and
/// `high = min(ceil(c(beta_slow)), d - 1)` with
/// `c(b) = d ln(original / (2 pi b)) / (2 ln theta)`; equal bounds widen
/// `high` by 0.001. Pair i keeps `smooth = 1 - clamp((i - low) / (high -
/// low), 0, 1)` of its frequency and takes the rest from `frequency /
/// factor`. Cosine and sine are not scaled: an attention factor belongs to
/// the softmax scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Yarn {
    pub factor: f64,
    pub original_max_position: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

/// Which position axis drives each pair. Rows give positions as (T, H, W);
/// `Linear` reads T for every pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axes {
    Linear,
    /// Interleaved M-RoPE sections: pair i reads H when `i % 3 == 1` and
    /// `i < 3 * sections[1]`, W when `i % 3 == 2` and `i < 3 * sections[2]`,
    /// and T otherwise. Zero sections equal `Linear`.
    Mrope([u32; 3]),
    /// Half split: the first half of the pairs reads axis 0 and the second
    /// half axis 1, each with its own `n / 2` frequencies (`d` = slice / 2).
    Axial2d,
}

impl Axes {
    /// The axis, frequency index and exponent denominator of `pair` of `pairs`.
    fn of(self, pair: usize, pairs: usize) -> (usize, usize, usize) {
        match self {
            Self::Linear => (0, pair, 2 * pairs),
            Self::Mrope(sections) => (mrope_axis(pair, sections), pair, 2 * pairs),
            Self::Axial2d => {
                let half = pairs / 2;
                (pair / half, pair % half, pairs)
            }
        }
    }
}

/// Which position axis (0 = T, 1 = H, 2 = W) drives rotary pair `pair`
/// under interleaved M-RoPE sections.
pub fn mrope_axis(pair: usize, sections: [u32; 3]) -> usize {
    if pair % 3 == 1 && pair < 3 * sections[1] as usize {
        1
    } else if pair % 3 == 2 && pair < 3 * sections[2] as usize {
        2
    } else {
        0
    }
}

/// A computed table: `theta`, the angle rule and the axis map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComputedTable {
    pub theta: f64,
    pub rule: AngleRule,
    pub axes: Axes,
}

/// Cosines and sines per position and pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FreqTable<'a> {
    Computed(ComputedTable),
    /// Captured `(cos, sin)` entries, `entries[position * pairs + pair]` for
    /// linear positions, for example loaded from a reference fixture.
    Captured(&'a [(f64, f64)]),
}

impl ComputedTable {
    /// `(cos, sin)` of every pair for one (T, H, W) position.
    pub fn row(&self, position: [usize; 3], pairs: usize) -> Result<Vec<(f64, f64)>, Refusal> {
        self.validate(pairs)?;
        Ok((0..pairs)
            .map(|pair| self.entry(position, pair, pairs))
            .collect())
    }

    /// `(cos, sin)` of pair `pair` of `pairs` for one (T, H, W) position.
    pub fn cos_sin(
        &self,
        position: [usize; 3],
        pair: usize,
        pairs: usize,
    ) -> Result<(f64, f64), Refusal> {
        self.validate(pairs)?;
        if pair >= pairs {
            return Err(Refusal("RoPE pair outside the table"));
        }
        Ok(self.entry(position, pair, pairs))
    }

    fn validate(&self, pairs: usize) -> Result<(), Refusal> {
        if pairs == 0 || !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(Refusal("RoPE table pairs or theta invalid"));
        }
        if self.axes == Axes::Axial2d && !pairs.is_multiple_of(2) {
            return Err(Refusal("axial 2D RoPE needs an even pair count"));
        }
        match self.rule {
            AngleRule::DivPowF32 | AngleRule::MulRecipPowF32
                if !(self.theta as f32).is_finite() =>
            {
                Err(Refusal("F32 RoPE theta overflows F32"))
            }
            AngleRule::HfF32AngleBf16CosSin
                if self.axes == Axes::Axial2d
                    || !(self.theta as f32).is_finite()
                    || self.theta as f32 <= 0.0 =>
            {
                Err(Refusal("HF RoPE geometry not implemented"))
            }
            AngleRule::YarnF64(yarn)
                if !yarn.factor.is_finite()
                    || yarn.factor <= 0.0
                    || yarn.original_max_position == 0
                    || yarn.band(self.axes.of(0, pairs).2, self.theta).is_none() =>
            {
                Err(Refusal("YaRN parameters invalid"))
            }
            _ => Ok(()),
        }
    }

    /// Unchecked: callers validate, or accept what the rule computes.
    pub(crate) fn entry(&self, position: [usize; 3], pair: usize, pairs: usize) -> (f64, f64) {
        let (axis, index, d) = self.axes.of(pair, pairs);
        let position = position[axis];
        let exponent = 2.0 * index as f64;
        let angle = match self.rule {
            AngleRule::DivPowF64 => position as f64 / self.theta.powf(exponent / d as f64),
            AngleRule::MulNegPowF64 => position as f64 * self.theta.powf(-exponent / d as f64),
            AngleRule::YarnF64(yarn) => {
                position as f64
                    * yarn.blend(self.theta.powf(-exponent / d as f64), index, d, self.theta)
            }
            AngleRule::HfF32AngleBf16CosSin => {
                let frequency =
                    if [16, 32].contains(&pairs) && (self.theta == 1e4 || self.theta == 1e7) {
                        // Width 32 has the same exponents as the even entries
                        // of the pinned width-64 table, with identical F32 bits.
                        f32::from_bits(
                            crate::hf_rope::INVERSE_BITS[usize::from(self.theta == 1e7)]
                                [index * (32 / pairs)],
                        )
                    } else {
                        hf_inverse_frequency(self.theta as f32, index, d)
                    };
                let angle = (position as f32 * frequency) as f64;
                return (
                    bf16_round(angle.cos() as f32 as f64),
                    bf16_round(angle.sin() as f32 as f64),
                );
            }
            AngleRule::DivPowF32 | AngleRule::MulRecipPowF32 => {
                let power = (self.theta as f32).powf(2.0 * index as f32 / d as f32);
                let angle = if self.rule == AngleRule::DivPowF32 {
                    position as f32 / power
                } else {
                    position as f32 * (1.0 / power)
                };
                return (f64::from(angle.cos()), f64::from(angle.sin()));
            }
            AngleRule::MulRecipF64PowF32 => {
                let exponent = (2 * index) as f32 / d as f32;
                let frequency = 1.0 / self.theta.powf(f64::from(exponent)) as f32;
                f64::from(position as f32 * frequency)
            }
        };
        let (sin, cos) = angle.sin_cos();
        (cos, sin)
    }
}

impl Yarn {
    /// The correction band `(low, high)` over `d / 2` frequencies. Only the
    /// usual band is implemented: theta above 1 and `beta_fast > beta_slow >
    /// 0`, so that `low <= high`; anything else is None.
    fn band(&self, d: usize, theta: f64) -> Option<(f64, f64)> {
        if !(theta > 1.0
            && self.beta_slow > 0.0
            && self.beta_fast > self.beta_slow
            && self.beta_fast.is_finite())
        {
            return None;
        }
        let correction = |rotations: f64| {
            d as f64
                * (self.original_max_position as f64 / (rotations * 2.0 * std::f64::consts::PI))
                    .ln()
                / (2.0 * theta.ln())
        };
        let low = correction(self.beta_fast).floor().max(0.0);
        let mut high = correction(self.beta_slow).ceil().min(d as f64 - 1.0);
        if !low.is_finite() || !high.is_finite() || low > high {
            return None;
        }
        if low == high {
            high += 0.001;
        }
        Some((low, high))
    }

    /// Callers validate the band first.
    fn blend(&self, frequency: f64, index: usize, d: usize, theta: f64) -> f64 {
        let (low, high) = self.band(d, theta).expect("validated YaRN band");
        let ramp = ((index as f64 - low) / (high - low)).clamp(0.0, 1.0);
        let smooth = 1.0 - ramp;
        frequency / self.factor * (1.0 - smooth) + frequency * smooth
    }
}

/// The arithmetic of one rotated element `x cos -/+ y sin`, where `x` is the
/// element and `y` its mate: the first of a pair subtracts, the second adds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RotaryArithmetic {
    /// Two f64 products and one f64 sum, the second element's sine term
    /// first: `x cos - y sin` and `y sin + x cos` (the order of `oracle::rope`).
    F64,
    /// `F64` with the second element's cosine term first, `x cos + y sin`
    /// (the order of `multimodal_oracle::scratch_rope`). Finite results equal
    /// `F64`'s; the order decides which NaN propagates when both terms are
    /// NaN in an unoptimized build (Rust leaves NaN payloads unspecified).
    F64CosFirst,
    /// F32 operands: the mate product `y * sin` in F32, then one F32 FMA
    /// `x * cos -/+ product`.
    F32Fma,
    /// Both products in f64, each rounded through F32 to BF16, then the sum
    /// rounded the same way.
    Bf16Products,
}

impl RotaryArithmetic {
    pub fn element(self, x: f64, y: f64, cos: f64, sin: f64, second: bool) -> f64 {
        match self {
            Self::F64 if second => y * sin + x * cos,
            Self::F64CosFirst if second => x * cos + y * sin,
            Self::F64 | Self::F64CosFirst => x * cos - y * sin,
            Self::F32Fma => {
                let product = y as f32 * sin as f32;
                f64::from((x as f32).mul_add(cos as f32, if second { product } else { -product }))
            }
            Self::Bf16Products => {
                let (a, b) = (bf16_round(x * cos), bf16_round(y * sin));
                bf16_round(if second { a + b } else { a - b })
            }
        }
    }
}

/// Rounding points of a rotation: `table` rounds each cosine and sine before
/// use and `output` rounds each rotated value. Both accept BF16 only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotaryRounding {
    pub table: Option<DType>,
    pub arithmetic: RotaryArithmetic,
    pub output: Option<DType>,
}

impl RotaryRounding {
    pub const F64: Self = Self {
        table: None,
        arithmetic: RotaryArithmetic::F64,
        output: None,
    };
}

/// Positions of the rows: one linear position, or (T, H, W), per row.
#[derive(Clone, Copy, Debug)]
pub enum RotaryPositions<'a> {
    Linear(&'a [usize]),
    Axes(&'a [[usize; 3]]),
}

impl RotaryPositions<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Linear(p) => p.len(),
            Self::Axes(p) => p.len(),
        }
    }

    fn get(&self, row: usize) -> [usize; 3] {
        match self {
            Self::Linear(p) => [p[row]; 3],
            Self::Axes(p) => p[row],
        }
    }
}

/// Every numerically relevant choice of a rotation. Pair i of the slice
/// rotates elements `(offset + i, offset + i + len / 2)` for NeoX and
/// `(offset + 2i, offset + 2i + 1)` for GPT-J. `inverse` rotates by the
/// negative angle (sine negated, exactly).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Rotation {
    pub slice: RotarySlice,
    pub pairing: RotaryStyle,
    pub inverse: bool,
    pub rounding: RotaryRounding,
}

impl Rotation {
    /// Rotates `input` [rows, head_dim], row r at position r of `positions`.
    pub fn apply(
        &self,
        input: &[f64],
        head_dim: usize,
        positions: RotaryPositions<'_>,
        table: &FreqTable<'_>,
    ) -> Result<Vec<f64>, Refusal> {
        let RotarySlice { offset, len } = self.slice;
        if head_dim == 0
            || len == 0
            || !len.is_multiple_of(2)
            || offset.checked_add(len).is_none_or(|end| end > head_dim)
            || input.len() != positions.len() * head_dim
        {
            return Err(Refusal("RoPE shape or slice mismatch"));
        }
        let bf16 = |point: Option<DType>| match point {
            None => Ok(false),
            Some(DType::Bf16) => Ok(true),
            Some(_) => Err(Refusal("RoPE rounding points are BF16 only")),
        };
        let (round_table, round_output) = (bf16(self.rounding.table)?, bf16(self.rounding.output)?);
        let pairs = len / 2;
        if let FreqTable::Computed(computed) = table {
            computed.validate(pairs)?;
        }
        if matches!(table, FreqTable::Captured(_)) && matches!(positions, RotaryPositions::Axes(_))
        {
            return Err(Refusal("captured RoPE tables take linear positions"));
        }
        let arithmetic = self.rounding.arithmetic;
        let mut output = input.to_vec();
        for row in 0..positions.len() {
            let position = positions.get(row);
            let base = row * head_dim + offset;
            for pair in 0..pairs {
                let (left, right) = match self.pairing {
                    RotaryStyle::Neox => (pair, pair + pairs),
                    RotaryStyle::GptJ => (pair * 2, pair * 2 + 1),
                };
                let (mut cos, mut sin) = match table {
                    FreqTable::Computed(computed) => computed.entry(position, pair, pairs),
                    FreqTable::Captured(entries) => *entries
                        .get(position[0] * pairs + pair)
                        .ok_or(Refusal("RoPE table too short"))?,
                };
                if round_table {
                    (cos, sin) = (bf16_round(cos), bf16_round(sin));
                }
                if self.inverse {
                    sin = -sin;
                }
                let x = input[base + left];
                let y = input[base + right];
                let mut a = arithmetic.element(x, y, cos, sin, false);
                let mut b = arithmetic.element(y, x, cos, sin, true);
                if round_output {
                    (a, b) = (bf16_round(a), bf16_round(b));
                }
                output[base + left] = a;
                output[base + right] = b;
            }
        }
        Ok(output)
    }
}

/// The allowance [`RotarySource::Computed`] grants one cosine or sine whose
/// exact value is `exact`, for pair `pair` of a `rotary_dim`-wide rotation
/// at `position`: the angle error `(8 + 3 |log2 f_i|) * 2^-24 * p * f_i`,
/// an F32 evaluation's `2^-21`, and a BF16 rounding's half step, at most
/// `2^-8` of the value's magnitude.
///
/// [`RotarySource::Computed`]: crate::contract::RotarySource::Computed
pub fn computed_allowance(
    theta: f64,
    rotary_dim: usize,
    pair: usize,
    position: usize,
    exact: f64,
) -> f64 {
    let frequency = theta.powf(-2.0 * pair as f64 / rotary_dim as f64);
    let angle = (8.0 + 3.0 * frequency.log2().abs()) * 2f64.powi(-24) * position as f64 * frequency;
    let unrounded = angle + 2f64.powi(-21);
    unrounded + 2f64.powi(-8) * (exact.abs() + unrounded)
}

/// Judges a backend's cosine and sine of pair `pair` of a computed
/// `rotary_dim`-wide rotation at `position` against those of the exact
/// angle (`AngleRule::DivPowF64`, as `oracle::rope` takes them without a
/// table), each within [`computed_allowance`].
pub fn check_computed(
    theta: f64,
    rotary_dim: usize,
    pair: usize,
    position: usize,
    cos_sin: (f64, f64),
) -> Result<(), Refusal> {
    if !rotary_dim.is_multiple_of(2) {
        return Err(Refusal("a rotation turns pairs of values"));
    }
    let exact = ComputedTable {
        theta,
        rule: AngleRule::DivPowF64,
        axes: Axes::Linear,
    }
    .cos_sin([position; 3], pair, rotary_dim / 2)?;
    for (actual, exact) in [(cos_sin.0, exact.0), (cos_sin.1, exact.1)] {
        let allowance = computed_allowance(theta, rotary_dim, pair, position, exact);
        // A theta outside the usual range can overflow the frequency, leaving
        // the reference or the allowance without a finite value.
        if !exact.is_finite() || !allowance.is_finite() {
            return Err(Refusal(
                "a computed RoPE reference is not finite at this theta",
            ));
        }
        if !actual.is_finite() || (actual - exact).abs() > allowance {
            return Err(Refusal(
                "a computed RoPE cosine or sine is outside its allowance",
            ));
        }
    }
    Ok(())
}

fn hf_inverse_frequency(theta: f32, index: usize, width: usize) -> f32 {
    1.0 / theta.powf(2.0 * index as f32 / width as f32)
}

#[cfg(test)]
mod tests {
    use super::{AngleRule, Axes, ComputedTable, check_computed};

    const LONG: usize = 1_048_575;
    const POSITIONS: [usize; 10] = [0, 1, 2, 36, 4095, 65535, 262143, 524287, 1_000_003, LONG];

    /// SM120's generic computed angle (`cuda/common/rotary.cuh`
    /// `generic_angle`), each approximate step pushed by its documented
    /// worst error in the direction `sign`: `log2f` one ulp, `exp2f` two,
    /// `__cosf` and `__sinf` 2^-21.41 on the reduced angle.
    fn sm120_generic(
        theta: f32,
        rotary: usize,
        pair: usize,
        position: usize,
        sign: i32,
    ) -> (f64, f64) {
        let nudge =
            |value: f32, ulps: i32| f32::from_bits(value.to_bits().wrapping_add_signed(ulps));
        let exponent = -2.0_f32 * pair as f32 / rotary as f32 * nudge(theta.log2(), sign);
        let frequency = nudge(exponent.exp2(), 2 * sign);
        let angle = position as f32 * frequency;
        let turns = (f64::from(angle) * 0.159_154_943_091_895_35).round_ties_even();
        let reduced = (-turns).mul_add(std::f64::consts::TAU, f64::from(angle)) as f32;
        let error = f64::from(sign) * 2f64.powf(-21.41);
        let (sin, cos) = f64::from(reduced).sin_cos();
        (cos + error, sin + error)
    }

    #[test]
    fn computed_rope_allowance_refuses_skipped_and_wrong_rotations() {
        for (theta, rotary) in [(1e4, 64), (1e7, 64), (1e4, 128)] {
            let pairs = rotary / 2;
            let exact = |position: usize, pair: usize| {
                ComputedTable {
                    theta,
                    rule: AngleRule::DivPowF64,
                    axes: Axes::Linear,
                }
                .cos_sin([position; 3], pair, pairs)
                .unwrap()
            };
            for pair in 0..pairs {
                // Where the pair turns by one, two and three radians, skipping
                // it or taking the next pair's angle fails.
                let frequency = theta.powf(-2.0 * pair as f64 / rotary as f64);
                for radians in [1.0, 2.0, 3.0] {
                    let position = (radians / frequency).round() as usize;
                    assert!(check_computed(theta, rotary, pair, position, (1.0, 0.0)).is_err());
                    let next = exact(position, (pair + 1) % pairs);
                    assert!(check_computed(theta, rotary, pair, position, next).is_err());
                    let entry = exact(position, pair);
                    assert_eq!(check_computed(theta, rotary, pair, position, entry), Ok(()));
                }
            }
        }
        // Rotary 64, theta 1e7, pairs 29 to 31 at a long position (pair 29's
        // exact angle is about 0.475; the BF16 reference is (0.890625,
        // 0.45703125)).
        for pair in 29..32 {
            assert!(check_computed(1e7, 64, pair, LONG, (1.0, 0.0)).is_err());
        }
        assert!(check_computed(1e7, 64, 29, LONG, (0.890625, 0.45703125)).is_ok());
        assert!(check_computed(1e7, 64, 29, LONG, (f64::NAN, 0.45703125)).is_err());
        assert!(check_computed(1e7, 64, 29, LONG, (0.890625, f64::INFINITY)).is_err());
        // A tiny theta overflows the frequency: no finite reference.
        assert!(check_computed(1e-320, 64, 31, 0, (123.0, f64::INFINITY)).is_err());
        assert!(check_computed(1e-320, 64, 31, 0, (1.0, 0.0)).is_err());
    }

    #[test]
    fn computed_rope_allowance_admits_the_references_and_both_sm120_paths() {
        for (theta, rotary) in [(1e4, 64), (1e7, 64), (1e4, 128), (5e5, 128), (1e6, 96)] {
            for pair in 0..rotary / 2 {
                for position in POSITIONS {
                    // The models' references: F32 tables, rounded to BF16 in
                    // BF16 runs. SM120's rotary-64 path equals the BF16 rule
                    // at theta 1e4 and 1e7.
                    for rule in [AngleRule::HfF32AngleBf16CosSin, AngleRule::MulRecipPowF32] {
                        let table = ComputedTable {
                            theta,
                            rule,
                            axes: Axes::Linear,
                        };
                        if rule == AngleRule::HfF32AngleBf16CosSin && rotary != 64 {
                            continue;
                        }
                        let entry = table.cos_sin([position; 3], pair, rotary / 2).unwrap();
                        assert_eq!(
                            check_computed(theta, rotary, pair, position, entry),
                            Ok(()),
                            "{rule:?} theta {theta} pair {pair} position {position}"
                        );
                    }
                    for sign in [-1, 1] {
                        let entry = sm120_generic(theta as f32, rotary, pair, position, sign);
                        assert_eq!(
                            check_computed(theta, rotary, pair, position, entry),
                            Ok(()),
                            "generic theta {theta} pair {pair} position {position} sign {sign}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn hf_powf_reciprocal_matches_pinned_inverse_bits() {
        for (theta, bits) in [1e4, 1e7].into_iter().zip(crate::hf_rope::INVERSE_BITS) {
            for (pair, expected) in bits.into_iter().enumerate() {
                assert_eq!(
                    super::hf_inverse_frequency(theta, pair, 64).to_bits(),
                    expected
                );
            }
        }
    }
}
