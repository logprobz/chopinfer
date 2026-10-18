//! Payload admission at create: one integer check per placed scale view of
//! a rank image, run on the device, which reports the lowest index of an
//! invalid element. Create refuses the image on any report, naming the
//! view. The backend's entries run the checks (`crate::own_ops::state`).
//!
//! The checks are the engines' admission rules:
//!
//! - F32 scales (block-FP8 `weight_scale_inv`, NVFP4 `weight_scale_2`):
//!   finite and not negative;
//! - NVFP4 UE4M3 local scales (`weight_scale`, stored as F8E4M3 bytes): at
//!   most 0x7e;
//! - MXFP8 codes with their UE8M0 scales (`weight` with its U8
//!   `weight_scale`): no NaN code, no scale 255, and every scaled code a
//!   finite BF16 equal to it.

use crate::catalog::DType;
use crate::direct::Stored;

/// The value of a report no element lowered.
pub const NONE: u32 = u32::MAX;

/// One check.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Check {
    F32Scales,
    Ue4m3Scales,
    Mxfp8,
}

impl Check {
    pub const ALL: [Check; 3] = [Check::F32Scales, Check::Ue4m3Scales, Check::Mxfp8];

    /// The host's mirror of the device check: the lowest index of an
    /// invalid element of `codes` (with MXFP8's `scales`).
    pub fn first_invalid(self, codes: &[u8], scales: &[u8]) -> Option<u64> {
        let invalid = |i: usize| -> bool {
            match self {
                Self::F32Scales => {
                    let bits = u32::from_le_bytes(codes[4 * i..4 * i + 4].try_into().unwrap());
                    bits & 0x7f80_0000 == 0x7f80_0000 || bits > 0x8000_0000
                }
                Self::Ue4m3Scales => codes[i] > 0x7e,
                Self::Mxfp8 => bad_mxfp8(codes[i], scales[i / 32]),
            }
        };
        let count = match self {
            Self::F32Scales => codes.len() / 4,
            _ => codes.len(),
        };
        (0..count).find(|&i| invalid(i)).map(|i| i as u64)
    }
}

/// The integer MXFP8 rule.
fn bad_mxfp8(code: u8, scale: u8) -> bool {
    if code & 0x7f == 0x7f || scale == 0xff {
        return true;
    }
    if (3..=246).contains(&scale) {
        return false;
    }
    let magnitude = u32::from(code & 0x7f);
    if magnitude == 0 {
        return false;
    }
    let (e, f) = (magnitude >> 3, magnitude & 7);
    let mut m = if e == 0 { f } else { 8 | f };
    let mut k = if e == 0 { -9 } else { e as i32 - 10 } + i32::from(scale) - 127;
    while m & 1 == 0 {
        m >>= 1;
        k += 1;
    }
    let top = k + (31 - m.leading_zeros() as i32);
    k < -133 || top > 127
}

/// One admission launch: `check` over the view `view` of `count` elements
/// (MXFP8: codes, with their scales in `scales`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Admission {
    pub check: Check,
    pub view: String,
    pub scales: Option<String>,
    pub count: u64,
}

/// Every admission of a rank's stored views (`crate::direct`), in view
/// order: each F32 scale view, each NVFP4 UE4M3 scale view and each MXFP8
/// code view with its scales. Refuses an MXFP8 scale view whose codes are
/// not 32 per scale, and a view of more elements than a report can index.
pub fn admissions(stored: &[Stored]) -> Result<Vec<Admission>, String> {
    let mut admissions = Vec::new();
    for entry in stored {
        let name = &entry.view.name;
        let bytes = entry.bytes();
        let admission = if (name.ends_with(".weight_scale_inv")
            || name.ends_with(".weight_scale_2"))
            && entry.dtype == DType::F32
        {
            Admission {
                check: Check::F32Scales,
                view: name.clone(),
                scales: None,
                count: bytes / 4,
            }
        } else if name.ends_with(".weight_scale") && entry.dtype == DType::F8E4M3 {
            Admission {
                check: Check::Ue4m3Scales,
                view: name.clone(),
                scales: None,
                count: bytes,
            }
        } else if let Some(stem) = name.strip_suffix(".weight_scale")
            && entry.dtype == DType::U8
        {
            let codes = format!("{stem}.weight");
            let code_bytes = stored
                .iter()
                .find(|other| other.view.name == codes && other.dtype == DType::F8E4M3)
                .ok_or_else(|| format!("{name} has no E4M3 codes {codes}"))?
                .bytes();
            if code_bytes != 32 * bytes {
                return Err(format!(
                    "{codes} holds {code_bytes} codes for {bytes} scales"
                ));
            }
            Admission {
                check: Check::Mxfp8,
                view: codes,
                scales: Some(name.clone()),
                count: code_bytes,
            }
        } else {
            continue;
        };
        if admission.count >= u64::from(NONE) {
            return Err(format!("{name} holds too many elements to admit"));
        }
        admissions.push(admission);
    }
    Ok(admissions)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Shape;
    use crate::direct::checkpoint_layout;
    use crate::quantization::{FP8_BLOCK128, NVFP4_DENSE_ROUTED, NVFP4_ROUTED};

    /// Each layout admits its scale views: block FP8's F32 block scales,
    /// NVFP4's UE4M3 scales and F32 globals, and MXFP8 MTP experts with
    /// their scales.
    #[test]
    fn each_layout_admits_its_scale_views() {
        let count = |layout, check| {
            admissions(&checkpoint_layout(Shape::TP4, layout, 0).unwrap())
                .unwrap()
                .iter()
                .filter(|a| a.check == check)
                .count()
        };
        use Check::*;
        assert!(count(&FP8_BLOCK128, F32Scales) > 0);
        assert_eq!(count(&FP8_BLOCK128, Ue4m3Scales), 0);
        assert_eq!(count(&FP8_BLOCK128, Mxfp8), 0);
        for layout in [&NVFP4_ROUTED, &NVFP4_DENSE_ROUTED] {
            assert!(count(layout, F32Scales) > 0, "{}", layout.id);
            assert!(count(layout, Ue4m3Scales) > 0, "{}", layout.id);
        }
        // The MTP layer's gate/up and down experts.
        assert_eq!(count(&NVFP4_ROUTED, Mxfp8), 2);
        assert_eq!(count(&NVFP4_DENSE_ROUTED, Mxfp8), 0);
        let mxfp8 = admissions(&checkpoint_layout(Shape::TP4, &NVFP4_ROUTED, 0).unwrap())
            .unwrap()
            .into_iter()
            .find(|a| a.check == Mxfp8)
            .unwrap();
        assert_eq!(mxfp8.count, 288 * 1024 * 4096);
        assert!(mxfp8.scales.unwrap().ends_with(".weight_scale"));
    }
}
