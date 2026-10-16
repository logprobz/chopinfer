//! GLM's own kernels outside the steps: commit's copy of one entry in every
//! layer of a layered state buffer, and create's payload admission of the
//! scale and code bytes its engines decode (`crate::payload`).

use kernel_core::contract::{Refusal, Validate};

use super::take;
use crate::payload::Check;

/// `bytes` bytes in each of `layers` layers copied from the source to the
/// destination, the layers `source_stride` and `destination_stride` bytes
/// apart, byte for byte as one host copy per layer. No layer's source and
/// destination overlap. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateCopy {
    pub bytes: u64,
    pub layers: u32,
    pub source_stride: u64,
    pub destination_stride: u64,
}

impl StateCopy {
    /// The bytes from the source's first layer to its last layer's end.
    pub fn source_span(&self) -> u64 {
        u64::from(self.layers.saturating_sub(1)) * self.source_stride + self.bytes
    }

    /// The bytes from the destination's first layer to its last layer's
    /// end.
    pub fn destination_span(&self) -> u64 {
        u64::from(self.layers.saturating_sub(1)) * self.destination_stride + self.bytes
    }
}

impl Validate for StateCopy {
    fn validate(&self) -> Result<(), Refusal> {
        if self.bytes == 0 || self.layers == 0 {
            return Err(Refusal("a state copy needs bytes in layers"));
        }
        if self.layers > 1
            && (self.source_stride < self.bytes || self.destination_stride < self.bytes)
        {
            return Err(Refusal("a state copy's layers overlap"));
        }
        let span = |stride: u64| {
            u64::from(self.layers - 1)
                .checked_mul(stride)
                .and_then(|span| span.checked_add(self.bytes))
        };
        if span(self.source_stride).is_none() || span(self.destination_stride).is_none() {
            return Err(Refusal("a state copy's span overflows"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// A state copy's buffers: the source's first layer's entry and the
    /// destination's, each holding its span. Views start at 16-byte
    /// alignment for the copy's 16-byte words (with the strides and bytes
    /// multiples of 16); any alignment copies byte by byte.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct StateCopyBuffers {
        pub source: value,
        pub destination: value,
    }
}

/// What the copy writes into `destination`, whose bytes outside the copied
/// layers it keeps.
pub fn state_copy(op: &StateCopy, source: &[u8], destination: &mut [u8]) -> Result<(), Refusal> {
    op.validate()?;
    take(
        source,
        op.source_span() as usize,
        "a state copy's source is too small",
    )?;
    if (destination.len() as u64) < op.destination_span() {
        return Err(Refusal("a state copy's destination is too small"));
    }
    let bytes = op.bytes as usize;
    for layer in 0..u64::from(op.layers) {
        let from = (layer * op.source_stride) as usize;
        let to = (layer * op.destination_stride) as usize;
        destination[to..to + bytes].copy_from_slice(&source[from..from + bytes]);
    }
    Ok(())
}

/// One payload admission: the lowest index of the first `count` elements
/// that `rule` refuses (`Check::first_invalid`, the integer rules of the
/// engines' admission), lowered into a report by an atomic minimum, so the
/// report does not depend on the schedule: a report holding a lower value
/// keeps it, and an index past `0xfffffffe` reports `0xfffffffe`. An MXFP8
/// check reads one scale per 32 codes. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Admission {
    pub rule: Check,
    pub count: u64,
}

impl Admission {
    /// The bytes of the checked elements (`codes`) and of their scales.
    pub fn bytes(&self) -> (u64, u64) {
        match self.rule {
            Check::F32Scales => (4 * self.count, 0),
            Check::Ue4m3Scales => (self.count, 0),
            Check::Mxfp8 => (self.count, self.count.div_ceil(32)),
        }
    }
}

impl Validate for Admission {
    fn validate(&self) -> Result<(), Refusal> {
        if self.count == 0 {
            return Err(Refusal("an admission checks elements"));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An admission's buffers: the checked elements, the MXFP8 check's
    /// UE8M0 scales, and the `u32` report the host sets before the first
    /// launch. F32 scales start at 4-byte alignment, codes and scales at 1,
    /// the report at 4.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct AdmissionBuffers {
        pub codes: value,
        pub scales: option,
        pub first: value,
    }
}

/// The report the admission leaves in `first`.
pub fn admission(
    op: &Admission,
    codes: &[u8],
    scales: &[u8],
    first: &mut u32,
) -> Result<(), Refusal> {
    op.validate()?;
    let (code_bytes, scale_bytes) = op.bytes();
    let codes = take(
        codes,
        code_bytes as usize,
        "an admission's elements are too few",
    )?;
    let scales = take(
        scales,
        scale_bytes as usize,
        "an admission's scales are too few",
    )?;
    if let Some(at) = op.rule.first_invalid(codes, scales) {
        *first = (*first).min(at.min(0xffff_fffe) as u32);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_move_each_layers_entry_alone() {
        let op = StateCopy {
            bytes: 2,
            layers: 3,
            source_stride: 3,
            destination_stride: 4,
        };
        assert_eq!((op.source_span(), op.destination_span()), (8, 10));
        let source = [1, 2, 9, 3, 4, 9, 5, 6];
        let mut destination = [0xaa; 10];
        state_copy(&op, &source, &mut destination).unwrap();
        assert_eq!(destination, [1, 2, 0xaa, 0xaa, 3, 4, 0xaa, 0xaa, 5, 6]);
        assert!(state_copy(&op, &source[..7], &mut destination).is_err());
        assert!(state_copy(&op, &source, &mut destination[..9]).is_err());
        let overlapping = StateCopy {
            source_stride: 1,
            ..op
        };
        assert!(overlapping.validate().is_err());
        assert!(
            StateCopy {
                layers: 1,
                source_stride: 0,
                destination_stride: 0,
                ..op
            }
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn admissions_lower_the_report_to_the_first_refused_element() {
        let words = |values: &[u32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let op = Admission {
            rule: Check::F32Scales,
            count: 4,
        };
        let codes = words(&[0x3f80_0000, 0x8000_0000, 0x7f80_0000, 0xbf80_0000]);
        let mut first = u32::MAX;
        admission(&op, &codes, &[], &mut first).unwrap();
        assert_eq!(first, 2);
        // A lower report stays; only the first `count` elements count.
        let mut first = 1;
        admission(&op, &codes, &[], &mut first).unwrap();
        assert_eq!(first, 1);
        let mut first = u32::MAX;
        admission(&Admission { count: 2, ..op }, &codes, &[], &mut first).unwrap();
        assert_eq!(first, u32::MAX);
        assert!(admission(&Admission { count: 5, ..op }, &codes, &[], &mut first).is_err());
        // MXFP8: one scale per 32 codes, a short scale table refused.
        let op = Admission {
            rule: Check::Mxfp8,
            count: 33,
        };
        let mut codes = vec![0; 33];
        codes[32] = 0x7f;
        let mut first = u32::MAX;
        admission(&op, &codes, &[127, 127], &mut first).unwrap();
        assert_eq!(first, 32);
        assert!(admission(&op, &codes, &[127], &mut first).is_err());
    }
}
