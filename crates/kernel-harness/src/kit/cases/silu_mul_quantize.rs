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
//! judges codes and scales together and counts both kinds. Failed rows
//! have exact zero codes, fixed finite scale 1, and sticky status words.

pub mod sequence;

use super::super::{OpCase, Required, Spec};
use crate::LabResult;
use crate::conformance::{Expected, Images, Judge, Rows};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::gated_mlp::SwigluClamp;
use kernel_core::gated_mlp::quantized::{SiluMulQuantizeBuffers, SiluMulQuantizeQuery};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::silu_mul_quantize as row;
use std::cell::RefCell;
use std::rc::Rc;

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

/// The lowest and highest F32 `expf(-g)` within [`EXP_ULPS`] of the
/// correctly rounded exponential. Past the overflow threshold by more than
/// that many ulps of the largest finite F32 (2^104 each), every such result
/// is infinite. kernel-lab's expf error probe checks a kernel's `expf`
/// against these ends.
pub fn exp_ends(g: f32) -> [f32; 2] {
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
    /// Sticky row flags, including all caller-supplied bits.
    pub status: Vec<u32>,
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

/// The admissible outcomes of `gate` and `up` `[rows, width]` (finite F32),
/// refused where a product's enclosure reaches a BF16 infinity.
pub fn outcomes(
    query: SiluMulQuantizeQuery,
    gate: &[f32],
    up: &[f32],
    rows: usize,
) -> LabResult<Outcomes> {
    outcomes_with_status(query, gate, up, &vec![0; rows])
}

/// Preserve the existing boundary-aware gate on passing rows; failed rows
/// have one exact outcome, positive zero codes and UE8M0 scale 127.
pub fn outcomes_with_status(
    query: SiluMulQuantizeQuery,
    gate: &[f32],
    up: &[f32],
    prior: &[u32],
) -> LabResult<Outcomes> {
    let rows = prior.len();
    let width = query.width as usize;
    if gate.len() != rows * width || up.len() != gate.len() || !width.is_multiple_of(GROUP) {
        return Err("quantized SwiGLU operand shape mismatch".into());
    }
    let status = query
        .row_status(gate, up, prior)
        .map_err(|e| e.to_string())?;
    let SwigluClamp { .. } = query.clamp;
    let (gate_max, limit) = (query.clamp.gate_max(), query.clamp.up_limit());
    let floor = f32::from_bits(query.amax_floor_bits);
    let bf16 = |v: f32| bf16_to_f32(f32_to_bf16(v));
    let values = gate
        .iter()
        .zip(up)
        .enumerate()
        .map(|(at, (&gate, &up))| {
            if status[at / width] != 0 { return Ok((0.0, 0.0)); }
            if !gate.is_finite() || !up.is_finite() {
                return Err("quantized SwiGLU inputs must be finite".to_string());
            }
            let g = gate.min(gate_max);
            let u = up.clamp(-limit, limit);
            let ends = exp_ends(g).map(|t| bf16(product_with(g, u, t)));
            if !ends.iter().all(|v| v.is_finite()) {
                return Err(format!(
                    "quantized SwiGLU product of gate {gate:e} and up {up:e} rounds to a BF16 infinity"
                ));
            }
            Ok((ends[0].min(ends[1]), ends[0].max(ends[1])))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let scales = values
        .chunks_exact(GROUP)
        .enumerate()
        .map(|(at, group)| {
            if status[at * GROUP / width] != 0 {
                return (127, 127);
            }
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
        status,
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
    fn check_scales(&self, scales: &[u8]) -> LabResult<()> {
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

    /// Checks each code is an admissible value's code at the scale written
    /// for its group. Returns how many took the upper outcome.
    fn check_codes(&self, codes: &[u8], scales: &[u8]) -> LabResult<usize> {
        if codes.len() != self.values.len() {
            return Err("quantized SwiGLU code count mismatch".into());
        }
        let mut high = 0;
        for (i, (&written, &(lo, hi))) in codes.iter().zip(&self.values).enumerate() {
            let scale = scales[i / GROUP];
            let (low_code, high_code) = (code(lo, scale), code(hi, scale));
            if written == low_code {
                continue;
            }
            if lo != hi && written == high_code {
                high += 1;
                continue;
            }
            return Err(format!(
                "element {i} (row {}, column {}): code {written:#04x} at scale {scale:#04x}, admissible {low_code:#04x} or {high_code:#04x}",
                i / self.width,
                i % self.width
            ));
        }
        Ok(high)
    }

    /// Checks a kernel's codes `[rows, width]` and scales together: every
    /// scale admissible, every code an admissible value's code at the
    /// scale the kernel wrote. Returns the boundary counts.
    pub fn check(&self, codes: &[u8], scales: &[u8]) -> LabResult<Boundaries> {
        self.check_scales(scales)?;
        let high = self.check_codes(codes, scales)?;
        Ok(Boundaries {
            elements_taken_high: high,
            ..self.boundaries()
        })
    }
}

/// The judges of one launch's `live` rows: the scales', which keeps the
/// bytes it saw, then the codes', which checks the codes and those scales
/// together ([`Outcomes::check`]), since a code is only right at the scale
/// written beside it. A conformance run calls an oracle's judges in the
/// order the oracle lists them; a codes judge that runs without the
/// scales' first refuses.
fn judges(admissible: Outcomes, live: usize) -> Vec<(String, Expected)> {
    let width = admissible.width;
    let seen = Rc::new(RefCell::new(None::<Vec<u8>>));
    let (kept, scales_of) = (Rc::clone(&seen), admissible.clone());
    let scales: Judge = Box::new(move |bytes: &[u8]| {
        let scales = bytes[..live * width / GROUP].to_vec();
        scales_of.check_scales(&scales)?;
        *kept.borrow_mut() = Some(scales);
        Ok(())
    });
    let codes: Judge = Box::new(move |bytes: &[u8]| {
        let scales = seen
            .borrow_mut()
            .take()
            .ok_or("quantized SwiGLU codes judged before their scales")?;
        admissible.check(&bytes[..live * width], &scales).map(drop)
    });
    vec![
        ("scales".into(), Expected::Judge(scales)),
        ("codes".into(), Expected::Judge(codes)),
    ]
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
/// `[-12, 12)`, past both clamps, and the admissible outcomes as one joint
/// judge of the codes and the scales ([`judges`]).
pub fn case<'a>(
    op: SiluMulQuantizeQuery,
    capacity: u32,
    prepared: Prepared<SiluMulQuantizeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    case_with_status(op, capacity, prepared, false)
}

pub fn case_with_status<'a>(
    op: SiluMulQuantizeQuery,
    capacity: u32,
    prepared: Prepared<SiluMulQuantizeBuffers<View>>,
    invalid: bool,
) -> LabResult<OpCase<'a>> {
    let (width, rows) = (op.width as usize, capacity as usize);
    let mut gate = spread(51, rows * width, 12.0);
    let mut up = spread(52, rows * width, 12.0);
    let mut prior = vec![0; rows];
    if invalid {
        for row in 0..rows {
            let at = row * width + if row % 2 == 0 { 0 } else { width - 1 };
            match row % 8 {
                0 => gate[at] = f32::NAN,
                1 => gate[at] = f32::INFINITY,
                2 => gate[at] = f32::NEG_INFINITY,
                3 => up[at] = f32::NAN,
                4 => up[at] = f32::INFINITY,
                5 => up[at] = f32::NEG_INFINITY,
                6 => prior[row] = 0x8000_0002,
                _ => {}
            }
        }
    }
    let roles = SiluMulQuantizeBuffers {
        gate: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&gate)),
        up: Spec::input(DType::F32, (Rows::Live, width), f32_bytes(&up)),
        codes: Spec::output(DType::Fp8E4M3, (Rows::Live, width)),
        scales: Spec::output(DType::Ue8M0, (Rows::Live, width / GROUP)),
        live_rows: Spec::live(),
        status: super::block_quantize::status_role(&prior),
        status_row_offset: 0,
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let live = live as usize;
        let gate = images.f32s("gate")?;
        let up = images.f32s("up")?;
        let mut status = images
            .bytes("status")?
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        let admissible = outcomes_with_status(
            op,
            &gate[..live * width],
            &up[..live * width],
            &status[..live],
        )?;
        status[..live].copy_from_slice(&admissible.status);
        let mut expected = judges(admissible, live);
        expected.push((
            "status".into(),
            Expected::Bits(status.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ));
        Ok(expected)
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
        .flat_map(|entry| {
            [false, true].map(move |invalid| {
                let op = entry.query;
                let need = Need::rows(CAPACITY);
                Required::new(
                    format!("silu mul quantize width {} invalid {invalid}", op.width),
                    need,
                    move |backend, need| backend.silu_mul_quantize(&op, need),
                    move |prepared| case_with_status(op, need.capacity, prepared, invalid),
                )
                .required()
            })
        })
        .chain([sequence::required()])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonfinite_and_prior_status_rows_have_only_positive_zero_outcomes() {
        let q = query(64);
        let mut gate = vec![2.0; 8 * 64];
        let mut up = vec![3.0; 8 * 64];
        for (row, bad) in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY]
            .into_iter()
            .enumerate()
        {
            gate[row * 64 + 63] = bad;
            up[(row + 3) * 64] = bad;
        }
        let expected = outcomes_with_status(q, &gate, &up, &[0, 2, 0, 0, 0, 0, 0x80, 0]).unwrap();
        assert_eq!(expected.status, [1, 3, 1, 1, 1, 1, 0x80, 0]);
        let (packed, flags) = q
            .apply_with_status(&gate, &up, &[0, 2, 0, 0, 0, 0, 0x80, 0])
            .unwrap();
        assert_eq!(flags, expected.status);
        let kernel_core::quantize::Scales::Ue8M0(scales) = packed.scales else {
            panic!("UE8M0 scales")
        };
        expected.check(&packed.codes, &scales).unwrap();
        let mut partial = packed.codes.clone();
        partial[32] = 1;
        assert!(expected.check(&partial, &scales).is_err());
        let mut negative_zero = packed.codes;
        negative_zero[0] = 0x80;
        assert!(expected.check(&negative_zero, &scales).is_err());
    }

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

    /// Runs the case's judges of one launch in the order the oracle lists
    /// them, as a conformance run does.
    fn judged(admissible: &Outcomes, codes: &[u8], scales: &[u8]) -> LabResult<()> {
        for (name, expected) in judges(admissible.clone(), admissible.rows) {
            let Expected::Judge(judge) = expected else {
                panic!("{name} is judged");
            };
            judge(if name == "codes" { codes } else { scales })?;
        }
        Ok(())
    }

    #[test]
    fn the_contracts_own_evaluation_is_admissible() {
        let q = query(576);
        let (gate, up) = (spread(1, 4 * 576, 12.0), spread(2, 4 * 576, 12.0));
        let admissible = outcomes(q, &gate, &up, 4).unwrap();
        let (codes, scales) = quantized(q, &gate, &up, 4);
        let counts = admissible.check(&codes, &scales).unwrap();
        assert_eq!(counts.groups, admissible.boundaries().groups);
        judged(&admissible, &codes, &scales).unwrap();
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
        assert!(judged(&admissible, &wrong, &scales).is_err());
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
        assert!(judged(&admissible, &codes, &wrong).is_err());
    }

    #[test]
    fn codes_are_judged_at_the_scale_written_beside_them() {
        // Products of 7 and 7.03125 at the two ends of the exponential's
        // interval, whose ceil scales are 2^-6 (0x79) and 2^-5 (0x7a).
        let q = query(32);
        let up = vec![f32::from_bits(0x40b7_0fd4); 32];
        let admissible = outcomes(q, &[1.5; 32], &up, 1).unwrap();
        assert!(admissible.values.iter().all(|&v| v == (7.0, 7.03125)));
        assert_eq!(admissible.scales, vec![(0x79, 0x7a)]);
        let (codes, scales) = quantized(q, &[1.5; 32], &up, 1);
        judged(&admissible, &codes, &scales).unwrap();
        // Either outcome at either scale passes, each code at its own scale.
        for (value, scale) in [(7.0, 0x79), (7.03125, 0x79), (7.0, 0x7a), (7.03125, 0x7a)] {
            let codes = vec![code(value, scale); 32];
            judged(&admissible, &codes, &[scale]).unwrap();
        }
        // A code admissible only at 0x7a, written beside 0x79, decodes to
        // 3.5 and fails, jointly and through the case's judges.
        assert_eq!(code(7.0, 0x7a), 0x76);
        assert_eq!(code(7.0, 0x79), 0x7e);
        let wrong = vec![0x76; 32];
        assert!(admissible.check(&wrong, &[0x79]).is_err());
        assert!(judged(&admissible, &wrong, &[0x79]).is_err());
        // A codes judge that runs before the scales' refuses.
        let (_, Expected::Judge(codes_judge)) = judges(admissible, 1).remove(1) else {
            panic!("codes are judged");
        };
        assert!(codes_judge(&codes).is_err());
    }

    #[test]
    fn a_product_that_rounds_to_a_bf16_infinity_sets_status() {
        // The F32 product of 2e19 and 1.7e19 is finite; its BF16 rounding
        // is not, which the oracle reports as an unrepresentable output.
        let q = SiluMulQuantizeQuery {
            clamp: SwigluClamp::symmetric(2e19),
            ..query(32)
        };
        let (gate, up) = ([2e19_f32; 32], [1.7e19_f32; 32]);
        assert!((gate[0] * up[0]).is_finite());
        let (output, status) = q.apply_with_status(&gate, &up, &[0]).unwrap();
        assert_eq!(status, [4]);
        assert_eq!(output.codes, vec![0; 32]);
        let admissible = outcomes(q, &gate, &up, 1).unwrap();
        assert_eq!(admissible.status, [4]);
        admissible.check(&[0; 32], &[127]).unwrap();
        outcomes(q, &[2e19; 32], &[1e19; 32], 1).unwrap();
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
        assert_eq!(
            outcomes(q, &[f32::NAN; 32], &[1.0; 32], 1).unwrap().status,
            [1]
        );
    }
}
