//! The clamped SwiGLU with BF16 rounding and ceil E4M3/32 quantization
//! (`SiluMulQuantizeQuery`), its admissible outcomes and its kit case.
//!
//! A kernel's `expf` is within [`EXP_ULPS`] of the exact exponential, so its
//! product can differ from another correct evaluation's. Every later F32
//! step (`1 + t`, `g / d`, `s * u`, the BF16 rounding) is a correctly
//! rounded monotone function of its operand, so evaluating the contract's
//! F32 formula at the two ends of the exponential's interval encloses
//! every kernel's BF16 product exactly ([`outcomes`]). The enclosure comes
//! from the inputs alone, never from a kernel's output. An element whose
//! ends round to one BF16 value is exact; at an element whose ends differ
//! (the value is within the propagated error of a BF16 rounding boundary)
//! the kernel's code must be one of the two adjacent outcomes. A group's
//! scale is exact unless its amax enclosure crosses a scale-selection
//! boundary, when it must be one of the two scales. [`Outcomes::check`]
//! counts both kinds.

use super::super::{OpCase, Required, Spec};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::gated_mlp::SwigluClamp;
use kernel_core::gated_mlp::quantized::{SiluMulQuantizeBuffers, SiluMulQuantizeQuery};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::silu_mul_quantize as row;

/// The documented error of CUDA's `expf`, 2 ulp, plus half an ulp for the
/// distance of the oracle's correctly rounded exponential from the exact
/// one, rounded up to whole ulps.
pub const EXP_ULPS: u32 = 3;
/// Elements of one scale group.
pub const GROUP: usize = 32;
/// The token rows the kit case serves.
pub const CAPACITY: u32 = 16;

/// `bits` moved `ulps` F32 steps up (`up`) or down, through zero and
/// saturating at the infinities.
fn step(value: f32, ulps: u32, up: bool) -> f32 {
    let mut v = value;
    for _ in 0..ulps {
        if v.is_infinite() && (v > 0.0) == up {
            break;
        }
        v = if up { v.next_up() } else { v.next_down() };
    }
    v
}

/// The lowest and highest F32 `expf(-g)` within [`EXP_ULPS`] of the exact
/// exponential. Past the overflow threshold by more than that many ulps of
/// the largest finite F32 (2^104 each), every such result is infinite.
fn exp_ends(g: f32) -> [f32; 2] {
    let exact = (-f64::from(g)).exp();
    let ulps = f64::from(EXP_ULPS);
    if exact >= f64::from(f32::MAX) + ulps * 2_f64.powi(104) {
        return [f32::INFINITY; 2];
    }
    let t = exact as f32;
    [step(t, EXP_ULPS, false).max(0.0), step(t, EXP_ULPS, true)]
}

/// The contract's F32 product before the BF16 rounding, with the
/// exponential `t` given: `fl(fl(g / fl(1 + t)) * u)`.
fn product_with(g: f32, u: f32, t: f32) -> f32 {
    (g / (1.0 + t)) * u
}

/// What a kernel may write for one quantized row.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcomes {
    pub width: usize,
    pub rows: usize,
    /// Per element, the lowest and highest BF16 product (as F32) any
    /// kernel within the exponential's error can form.
    pub values: Vec<(f32, f32)>,
    /// Per group, the lowest and highest UE8M0 scale code.
    pub scales: Vec<(u8, u8)>,
}

/// The UE8M0 ceil scale of a group amax (`block_quantize.cuh`'s
/// `ceil_scale`): the amax floored, times the F32 reciprocal of 448, its
/// exponent bits rounded up.
fn ceil_scale(amax: f32, floor: f32) -> u8 {
    let bits = (amax.max(floor) * (1.0 / 448.0)).to_bits();
    ((bits >> 23) + u32::from(bits & 0x7f_ffff != 0)) as u8
}

/// The E4M3 code of `value` at UE8M0 scale `scale`: `RNE(value / 2^(scale -
/// 127))`, the division in F32.
fn code(value: f32, scale: u8) -> u8 {
    SmallFloat::E4M3.encode_f32(value / f32::from_bits(u32::from(scale) << 23))
}

/// The admissible outcomes of `gate` and `up` `[rows, width]` (finite F32).
pub fn outcomes(
    query: SiluMulQuantizeQuery,
    gate: &[f32],
    up: &[f32],
    rows: usize,
) -> LabResult<Outcomes> {
    let width = query.width as usize;
    if gate.len() != rows * width || up.len() != gate.len() || !width.is_multiple_of(GROUP) {
        return Err("quantized SwiGLU operand shape mismatch".into());
    }
    let SwigluClamp { .. } = query.clamp;
    let (gate_max, limit) = (query.clamp.gate_max(), query.clamp.up_limit());
    let floor = f32::from_bits(query.amax_floor_bits);
    let bf16 = |v: f32| bf16_to_f32(f32_to_bf16(v));
    let values = gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| {
            if !gate.is_finite() || !up.is_finite() {
                return Err("quantized SwiGLU inputs must be finite".to_string());
            }
            let g = gate.min(gate_max);
            let u = up.clamp(-limit, limit);
            let ends = exp_ends(g).map(|t| bf16(product_with(g, u, t)));
            Ok((ends[0].min(ends[1]), ends[0].max(ends[1])))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scales = values
        .chunks_exact(GROUP)
        .map(|group| {
            let low = group
                .iter()
                .map(|&(lo, hi)| {
                    if lo <= 0.0 && hi >= 0.0 {
                        0.0
                    } else {
                        lo.abs().min(hi.abs())
                    }
                })
                .fold(0.0_f32, f32::max);
            let high = group
                .iter()
                .map(|&(lo, hi)| lo.abs().max(hi.abs()))
                .fold(0.0_f32, f32::max);
            (ceil_scale(low, floor), ceil_scale(high, floor))
        })
        .collect();
    Ok(Outcomes {
        width,
        rows,
        values,
        scales,
    })
}

/// How many elements and groups a check admitted at a boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Boundaries {
    /// Elements whose BF16 product has two admissible values.
    pub elements: usize,
    /// Groups whose scale has two admissible codes.
    pub groups: usize,
    /// Of `elements`, those where the kernel wrote the other outcome than
    /// the lower end's.
    pub elements_taken_high: usize,
}

impl Outcomes {
    fn groups(&self) -> usize {
        self.values.len() / GROUP
    }

    /// The elements and groups at a boundary, from the oracle alone.
    pub fn boundaries(&self) -> Boundaries {
        Boundaries {
            elements: self.values.iter().filter(|(lo, hi)| lo != hi).count(),
            groups: self.scales.iter().filter(|(lo, hi)| lo != hi).count(),
            elements_taken_high: 0,
        }
    }

    /// Checks each scale code (`[rows, width / 32]`) is admissible.
    pub fn check_scales(&self, scales: &[u8]) -> LabResult<()> {
        if scales.len() != self.groups() {
            return Err("quantized SwiGLU scale count mismatch".into());
        }
        for (group, (&scale, &(lo, hi))) in scales.iter().zip(&self.scales).enumerate() {
            if scale < lo || scale > hi {
                return Err(format!(
                    "group {group}: scale {scale:#04x} outside {lo:#04x}..={hi:#04x}"
                ));
            }
        }
        Ok(())
    }

    /// Checks each code is an admissible value's code at `scale_of(group)`.
    fn check_codes_with(
        &self,
        codes: &[u8],
        scales_of: impl Fn(usize) -> Vec<u8>,
    ) -> LabResult<usize> {
        if codes.len() != self.values.len() {
            return Err("quantized SwiGLU code count mismatch".into());
        }
        let mut high = 0;
        for (i, (&written, &(lo, hi))) in codes.iter().zip(&self.values).enumerate() {
            let scales = scales_of(i / GROUP);
            let low_codes = scales.iter().map(|&s| code(lo, s)).collect::<Vec<_>>();
            let high_codes = scales.iter().map(|&s| code(hi, s)).collect::<Vec<_>>();
            if low_codes.contains(&written) {
                continue;
            }
            if lo != hi && high_codes.contains(&written) {
                high += 1;
                continue;
            }
            return Err(format!(
                "element {i} (row {}, column {}): code {written:#04x}, admissible {low_codes:02x?} or {high_codes:02x?}",
                i / self.width,
                i % self.width
            ));
        }
        Ok(high)
    }

    /// Checks codes alone: each an admissible value's code at some
    /// admissible scale of its group.
    pub fn check_codes(&self, codes: &[u8]) -> LabResult<()> {
        self.check_codes_with(codes, |group| {
            let (lo, hi) = self.scales[group];
            (lo..=hi).collect()
        })
        .map(drop)
    }

    /// Checks a kernel's codes `[rows, width]` and scales together: every
    /// scale admissible, every code an admissible value's code at the
    /// scale the kernel wrote. Returns the boundary counts.
    pub fn check(&self, codes: &[u8], scales: &[u8]) -> LabResult<Boundaries> {
        self.check_scales(scales)?;
        let high = self.check_codes_with(codes, |group| vec![scales[group]])?;
        Ok(Boundaries {
            elements_taken_high: high,
            ..self.boundaries()
        })
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// F32 values in `[-span, span)` from `seed`, full mantissas.
pub fn spread(seed: u64, len: usize, span: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0 - 1.0) * span
        })
        .collect()
}

/// The case of a quantized SwiGLU over `capacity` rows: F32 gate and up in
/// `[-12, 12)`, past both clamps, and the admissible outcomes as judges of
/// the codes and the scales.
pub fn case<'a>(
    op: SiluMulQuantizeQuery,
    capacity: u32,
    prepared: Prepared<SiluMulQuantizeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let (width, rows) = (op.width as usize, capacity as usize);
    let roles = SiluMulQuantizeBuffers {
        gate: Spec::input(
            DType::F32,
            (Rows::Live, width),
            f32_bytes(&spread(51, rows * width, 12.0)),
        ),
        up: Spec::input(
            DType::F32,
            (Rows::Live, width),
            f32_bytes(&spread(52, rows * width, 12.0)),
        ),
        codes: Spec::output(DType::Fp8E4M3, (Rows::Live, width)),
        scales: Spec::output(DType::Ue8M0, (Rows::Live, width / GROUP)),
        live_rows: Spec::live(),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let live = live as usize;
        let gate = images.f32s("gate")?;
        let up = images.f32s("up")?;
        let admissible = outcomes(op, &gate[..live * width], &up[..live * width], live)?;
        let codes = admissible.clone();
        Ok(vec![
            (
                "codes".into(),
                Expected::Judge(Box::new(move |bytes: &[u8]| {
                    codes.check_codes(&bytes[..live * width])
                })),
            ),
            (
                "scales".into(),
                Expected::Judge(Box::new(move |bytes: &[u8]| {
                    admissible.check_scales(&bytes[..live * width / GROUP])
                })),
            ),
        ])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// SM120 with the lab's quantized SwiGLU entries.
pub fn sm120() -> Sm120 {
    row::lab_entries()
        .into_iter()
        .fold(Sm120::new(), Sm120::with_silu_mul_quantize)
}

/// Each lab entry's op at [`CAPACITY`] rows, required.
pub fn required<'a>() -> Vec<Required<'a>> {
    row::lab_entries()
        .into_iter()
        .map(|entry| {
            let op = entry.query;
            let need = Need::rows(CAPACITY);
            Required::new(
                format!("silu mul quantize width {}", op.width),
                need,
                move |backend, need| backend.silu_mul_quantize(&op, need),
                move |prepared| case(op, need.capacity, prepared),
            )
            .required()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(width: u32) -> SiluMulQuantizeQuery {
        SiluMulQuantizeQuery {
            width,
            clamp: SwigluClamp::symmetric(10.0),
            amax_floor_bits: 1e-4_f32.to_bits(),
        }
    }

    fn quantized(
        q: SiluMulQuantizeQuery,
        gate: &[f32],
        up: &[f32],
        rows: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let out = q.apply(gate, up, rows).unwrap();
        let kernel_core::quantize::Scales::Ue8M0(scales) = out.scales else {
            panic!("UE8M0 scales");
        };
        (out.codes, scales)
    }

    #[test]
    fn the_contracts_own_evaluation_is_admissible() {
        let q = query(576);
        let (gate, up) = (spread(1, 4 * 576, 12.0), spread(2, 4 * 576, 12.0));
        let admissible = outcomes(q, &gate, &up, 4).unwrap();
        let (codes, scales) = quantized(q, &gate, &up, 4);
        let counts = admissible.check(&codes, &scales).unwrap();
        assert_eq!(counts.groups, admissible.boundaries().groups);
        admissible.check_codes(&codes).unwrap();
    }

    #[test]
    fn a_one_step_error_away_from_any_boundary_fails() {
        let q = query(64);
        let (gate, up) = (spread(3, 64, 12.0), spread(4, 64, 12.0));
        let admissible = outcomes(q, &gate, &up, 1).unwrap();
        let (codes, scales) = quantized(q, &gate, &up, 1);
        admissible.check(&codes, &scales).unwrap();
        let exact_group = |g: usize| admissible.scales[g].0 == admissible.scales[g].1;
        // An exact element of an exact group with a nonzero finite code.
        let at = (0..64)
            .find(|&i| {
                let (lo, hi) = admissible.values[i];
                lo == hi && exact_group(i / GROUP) && codes[i] & 0x7f != 0 && codes[i] & 0x7f < 0x7e
            })
            .unwrap();
        // One code step off fails, with either check.
        let mut wrong = codes.clone();
        wrong[at] += 1;
        assert!(admissible.check(&wrong, &scales).is_err());
        assert!(admissible.check_codes(&wrong).is_err());
        // A product one BF16 ulp off whose code differs fails.
        let (value, _) = admissible.values[at];
        let off = bf16_to_f32(f32_to_bf16(value).wrapping_add(1));
        let mut wrong = codes.clone();
        wrong[at] = code(off, scales[at / GROUP]);
        if wrong[at] != codes[at] {
            assert!(admissible.check(&wrong, &scales).is_err());
        }
        // An exact group's scale one code off fails.
        let group = (0..2).find(|&g| exact_group(g)).unwrap();
        let mut wrong = scales.clone();
        wrong[group] += 1;
        assert!(admissible.check(&codes, &wrong).is_err());
        assert!(admissible.check_scales(&wrong).is_err());
    }

    #[test]
    fn boundaries_come_from_the_inputs() {
        // Products steered onto BF16 midpoints: u = m / silu(g) for a
        // midpoint m makes the F32 product land within a few ulps of m, so
        // the exponential's error decides the rounding.
        let q = query(32);
        let g = 1.5_f32;
        let silu = g / (1.0 + (-g).exp());
        let up = (0..32)
            .map(|i| {
                let lower = bf16_to_f32(f32_to_bf16(0.5 + i as f32 / 64.0));
                let midpoint = (lower + bf16_to_f32(f32_to_bf16(lower) + 1)) / 2.0;
                midpoint / silu
            })
            .collect::<Vec<_>>();
        let admissible = outcomes(q, &[g; 32], &up, 1).unwrap();
        let counts = admissible.boundaries();
        assert!(counts.elements > 0, "{counts:?}");
        let (codes, scales) = quantized(q, &[g; 32], &up, 1);
        admissible.check(&codes, &scales).unwrap();
        // The amax at a scale boundary: a product steered onto the BF16
        // midpoint between 448 * 2^-6 and the next value up, where the ceil
        // scale's exponent changes, gives two admissible scales.
        let midpoint = (448.0 + 450.0) / 2.0 * 2_f32.powi(-6);
        let up = vec![midpoint / silu; 32];
        let admissible = outcomes(q, &[g; 32], &up, 1).unwrap();
        let counts = admissible.boundaries();
        assert_eq!(counts.groups, 1, "{counts:?}");
        let (codes, scales) = quantized(q, &[g; 32], &up, 1);
        admissible.check(&codes, &scales).unwrap();
        // Saturated and clamped inputs are exact.
        let admissible = outcomes(q, &[-200.0; 32], &[5.0; 32], 1).unwrap();
        assert_eq!(admissible.boundaries(), Boundaries::default());
        assert!(
            admissible
                .values
                .iter()
                .all(|&(lo, hi)| lo == 0.0 && hi == 0.0)
        );
        assert!(outcomes(q, &[f32::NAN; 32], &[1.0; 32], 1).is_err());
    }
}
