//! GLM 5.3 Flash's own kernels as contracts: its row kernels (the single
//! rounding of an all-reduced F32 row, the widening of a BF16 row, the
//! drafter's residual add and the gather of the drafter's row inputs),
//! the placement of the drafter's input halves, the stacked KDA input
//! projections, commit's layered state copy and create's payload
//! admission. No other model runs them, so they live here rather than in
//! kernel-core (an op moves there when a second model needs it, as the
//! stream expansion, the score compaction and the head rows' gather by
//! index moved to the shared row copies, `crate::copies`).
//!
//! Each op is a contract type whose fields are the math the model needs
//! (widths, strides, placements, tables), a role record of its buffers and
//! a CPU oracle. Copies, single roundings and integer checks have bitwise
//! oracles that compute exactly what the kernel computes; the stacked
//! projection's MMA accumulation has an F64 reference with the numerical
//! contract's per-element bound. A backend prepares each op through
//! [`OwnOps`] or refuses it; SM120 choices (threads, vector widths, grids,
//! row buckets, compiled constants) are its refusals, not contract rules.

use kernel_core::backend::{Need, Prepared, Refused, View};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::Refusal;
use kernel_core::precision::OperationUse;

pub mod draft;
pub mod rows;
pub mod stacked;
pub mod state;

use draft::{DraftGather, DraftGatherBuffers};
use rows::{
    EhInput, EhInputBuffers, ResidualAdd, ResidualAddBuffers,
    RoundRows, RoundRowsBuffers, WidenRows, WidenRowsBuffers,
};
use stacked::{StackedProjection, StackedProjectionBuffers};
use state::{Admission, AdmissionBuffers, StateCopy, StateCopyBuffers};

/// A backend of GLM's own kernels. Each method prepares one op or refuses
/// it; every method refuses by default, so a backend implements the ops it
/// has kernels for. Each method's doc says which
/// [`kernel_core::backend::Counts`] its launches read and the rows each
/// launch records. Each role record states the alignment its views need.
///
/// The row ops read a device live count (`live_rows` or `count`) of at
/// most `Counts::rows` and write rows below it only, so their views hold
/// `Counts::rows` rows. A launch may grid more rows than
/// `Counts::rows` and records the rows it grids (SM120: the smallest row
/// bucket holding them, as the lowering records today).
pub trait OwnOps: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

    // Rows around the all-reduce.

    /// Rounds each live F32 row once to BF16.
    ///
    /// Launches read `Counts::rows`.
    fn round_rows(
        &self,
        op: &RoundRows,
        need: Need,
    ) -> Result<Prepared<RoundRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Widens each live BF16 row to F32, exactly.
    ///
    /// Launches read `Counts::rows`.
    fn widen_rows(
        &self,
        op: &WidenRows,
        need: Need,
    ) -> Result<Prepared<WidenRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Adds Y into X in place over the live rows, one rounding.
    ///
    /// Launches read `Counts::rows`.
    fn residual_add(
        &self,
        op: &ResidualAdd,
        need: Need,
    ) -> Result<Prepared<ResidualAddBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Places the drafter's normalized embedding and hidden rows side by
    /// side as its input projection's input, at the projection's use
    /// `usage` (whose compiled capacity the copies run at).
    ///
    /// Launches read `Counts::rows` and record them.
    fn eh_input(
        &self,
        op: &EhInput,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<EhInputBuffers<View>>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }

    // The MTP drafter's rows.

    /// Gathers the drafter's hidden input rows and tokens by source codes.
    ///
    /// Launches read `Counts::rows`. Non-negative hidden codes are rows of
    /// `last`, negative ones entries of `retained`; non-negative token
    /// codes are entries of `drafts`.
    fn draft_gather(
        &self,
        op: &DraftGather,
        need: Need,
    ) -> Result<Prepared<DraftGatherBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    // The KDA layers' input projections.

    /// A KDA layer's input projections of its normed rows in one launch,
    /// at the decode or verify use `usage`.
    ///
    /// Launches read `Counts::rows`, the live rows (a host scalar of the
    /// launch, at most the use's capacity), and record them.
    fn stacked_projection(
        &self,
        op: &StackedProjection,
        usage: OperationUse,
        need: Need,
    ) -> Result<Prepared<StackedProjectionBuffers<View>>, Refused> {
        let _ = (op, usage, need);
        Err(Refused::Unsupported)
    }

    // Request state and admission, outside the steps.

    /// Copies one entry in each layer of a layered state buffer.
    ///
    /// Launches read no counts and record one row.
    fn state_copy(
        &self,
        op: &StateCopy,
        need: Need,
    ) -> Result<Prepared<StateCopyBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Lowers a report to the lowest index of an element its rule refuses.
    ///
    /// Launches read no counts and record one row.
    fn admission(
        &self,
        op: &Admission,
        need: Need,
    ) -> Result<Prepared<AdmissionBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }
}

/// `value` rounded to BF16 as the own kernels round (`cvt.rn.bf16.f32`):
/// to nearest, ties to even. A NaN rounds to a NaN whose sign and payload
/// the entries do not specify; the oracles write the canonical `0x7fff` for
/// it, and a result is compared with them by [`same_bf16`].
pub fn round_bf16(value: f32) -> u16 {
    if value.is_nan() {
        0x7fff
    } else {
        f32_to_bf16(value)
    }
}

/// Whether BF16 bits `actual` are an oracle's `expected` result: the same
/// bits, or both NaN (an own kernel's NaN results are NaN of any sign and
/// payload).
pub fn same_bf16(expected: u16, actual: u16) -> bool {
    let nan = |bits: u16| bits & 0x7f80 == 0x7f80 && bits & 0x7f != 0;
    expected == actual || (nan(expected) && nan(actual))
}

/// Whether F32 bits `actual` are an oracle's `expected` result: the same
/// bits, or both NaN.
pub fn same_f32(expected: u32, actual: u32) -> bool {
    expected == actual || (f32::from_bits(expected).is_nan() && f32::from_bits(actual).is_nan())
}

/// The rows of a launch of `rows` grid rows that the device live count
/// `live` lets write.
fn live(rows: u32, live: u32) -> usize {
    rows.min(live) as usize
}

/// The first `count` elements of `values`, or a refusal naming them.
fn take<'a, T>(values: &'a [T], count: usize, name: &'static str) -> Result<&'a [T], Refusal> {
    values.get(..count).ok_or(Refusal(name))
}

/// The first `count` elements of `values`, writable, or a refusal.
fn take_mut<'a, T>(
    values: &'a mut [T],
    count: usize,
    name: &'static str,
) -> Result<&'a mut [T], Refusal> {
    values.get_mut(..count).ok_or(Refusal(name))
}

/// Entry `index` of a device table, or a refusal naming the table.
fn entry<T: Copy>(table: &[T], index: usize, name: &'static str) -> Result<T, Refusal> {
    table.get(index).copied().ok_or(Refusal(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Shape;

    struct Nothing;

    impl OwnOps for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    fn unsupported<T>(result: Result<T, Refused>) {
        assert!(matches!(result, Err(Refused::Unsupported)));
    }

    #[test]
    fn every_method_refuses_by_default() {
        let shape = Shape::TP4;
        let need = Need::rows(32);
        let usage = OperationUse {
            phase: kernel_core::precision::ExecutionPhase::Decode,
            capacity: 8,
            graph: true,
        };
        let backend = Nothing;
        assert_eq!(backend.name(), "nothing");
        unsupported(backend.round_rows(&RoundRows::of(shape), need));
        unsupported(backend.widen_rows(&WidenRows::of(shape), need));
        unsupported(backend.residual_add(&ResidualAdd::of(shape), need));
        unsupported(backend.eh_input(&EhInput::of(shape), usage, need));
        unsupported(backend.draft_gather(&DraftGather::of(shape), need));
        unsupported(backend.stacked_projection(
            &StackedProjection {
                k: 16,
                parts: vec![stacked::StackedPart::contiguous(16)],
            },
            usage,
            need,
        ));
        unsupported(backend.state_copy(
            &StateCopy {
                bytes: 16,
                layers: 1,
                source_stride: 16,
                destination_stride: 16,
            },
            need,
        ));
        unsupported(backend.admission(
            &Admission {
                rule: crate::payload::Check::F32Scales,
                count: 1,
            },
            need,
        ));
    }

    /// Ties go to the even neighbour; a NaN is a NaN, compared by class.
    #[test]
    fn roundings_are_to_nearest_even_and_nans_compare_by_class() {
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(round_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
        assert_eq!(round_bf16(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(round_bf16(f32::from_bits(0x7f7f_ffff)), 0x7f80);
        assert_eq!(round_bf16(f32::NEG_INFINITY), 0xff80);
        for nan in [0x7fc0_0000, 0xffc0_0001, 0x7f80_0001] {
            assert_eq!(round_bf16(f32::from_bits(nan)), 0x7fff);
            assert!(same_bf16(0x7fff, (nan >> 16) as u16 | 0x0040));
            assert!(same_f32(0x7fff_ffff, nan));
        }
        assert!(same_bf16(0xffc1, 0x7f81) && !same_bf16(0x7fff, 0x7f80));
        assert!(!same_bf16(0x3f80, 0x3f81) && !same_f32(0x7f80_0000, 0x7fc0_0000));
        for value in [0x0000, 0x8000, 0x7f80, 0xff80] {
            assert!(same_bf16(value, value));
        }
    }
}
