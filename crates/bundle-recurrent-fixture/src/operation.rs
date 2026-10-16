//! The one custom operation this fixture supplies for itself.
//!
//! The manifest declares this operation as a bundle custom operation, which is
//! the escape hatch a host never implements: no host-optimized or generic
//! candidate may satisfy it, so an installed bundle that does not carry it
//! leaves the operation unsupported. Here it is carried as a native entry in
//! this fixture's own module, reached through the host's `run_custom_operation`
//! service, which waits on the stream, establishes the context, lends the
//! operands, and contains a panic before it can cross back.
//!
//! The operation reads through no lent address. Doing that would be unsafe code
//! this fixture does not write, and the address space it is lent depends on the
//! host: a driver lends device addresses, and a host-memory backend lends its
//! own. What ties the computation to the operands is the lend itself, which is
//! checked against the geometry the instance staged before the call: the operand
//! count, each operand's exact byte length, and that something was staged at
//! all. A lend that does not match is refused with a code of this fixture's own,
//! and the instance then emits no row.

use runtime_bundle_abi::{CustomAddressSpaceV1, LentLaunchV1};

use crate::recurrent::{RecurrentLanesV1, fold_tokens_v1, logits_row_v1};

/// The identifier the host resolves, audits, and logs this operation by.
pub const RECURRENT_STEP_OPERATION_ID_V1: &[u8] = b"fixture.recurrent.step.v1";

/// How many operands the operation is lent: the tokens, then the row it writes.
pub const RECURRENT_STEP_OPERANDS_V1: usize = 2;

/// The operation ran and staged its result.
pub const RECURRENT_STEP_OK_V1: i32 = 0;

/// The operation was lent a different number of operands than it declared.
pub const RECURRENT_STEP_WRONG_OPERAND_COUNT_V1: i32 = 1;

/// One lent operand was not the byte length the instance staged for it.
pub const RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1: i32 = 2;

/// The operation was invoked with nothing staged, so there was nothing to run.
pub const RECURRENT_STEP_NOTHING_STAGED_V1: i32 = 3;

/// What one execution stages for the operation before it is invoked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedStepV1 {
    pub lanes: RecurrentLanesV1,
    pub tokens: Vec<u32>,
    pub vocabulary: u32,
    pub input_bytes: u64,
    pub output_bytes: u64,
}

/// What one execution takes back from the operation after it returns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepResultV1 {
    pub lanes: RecurrentLanesV1,
    pub logits: Vec<u8>,
}

/// What the operation was lent on its most recent call.
///
/// The instance checks this after the call, so a host that reported success
/// without invoking anything is caught rather than believed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedLendV1 {
    pub address_space: CustomAddressSpaceV1,
    pub operand_lengths: Vec<u64>,
}

/// The state one bundle custom operation owns across its calls.
#[derive(Debug, Default)]
pub struct RecurrentStepOperationV1 {
    staged: Option<StagedStepV1>,
    result: Option<StepResultV1>,
    observed: Option<ObservedLendV1>,
    invocations: usize,
}

impl RecurrentStepOperationV1 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            staged: None,
            result: None,
            observed: None,
            invocations: 0,
        }
    }

    /// Stages one step and clears whatever the previous call left behind.
    pub fn stage(&mut self, staged: StagedStepV1) {
        self.staged = Some(staged);
        self.result = None;
        self.observed = None;
    }

    /// Takes the result of the call that just ran, if it produced one.
    pub fn take_result(&mut self) -> Option<StepResultV1> {
        self.staged = None;
        self.result.take()
    }

    /// What the host lent on the most recent call.
    #[must_use]
    pub const fn observed(&self) -> Option<&ObservedLendV1> {
        self.observed.as_ref()
    }

    /// How many times the host has invoked this operation.
    #[must_use]
    pub const fn invocations(&self) -> usize {
        self.invocations
    }
}

/// Runs one recurrent step against the operands the host lent.
fn run_recurrent_step_v1(
    operation: &mut RecurrentStepOperationV1,
    launch: LentLaunchV1<'_>,
) -> i32 {
    operation.invocations += 1;
    operation.observed = Some(ObservedLendV1 {
        address_space: launch.address_space,
        operand_lengths: launch.ranges.iter().map(|range| range.length).collect(),
    });
    let Some(staged) = operation.staged.as_ref() else {
        return RECURRENT_STEP_NOTHING_STAGED_V1;
    };
    if launch.ranges.len() != RECURRENT_STEP_OPERANDS_V1 {
        return RECURRENT_STEP_WRONG_OPERAND_COUNT_V1;
    }
    if launch.ranges[0].length != staged.input_bytes
        || launch.ranges[1].length != staged.output_bytes
    {
        return RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1;
    }
    let lanes = fold_tokens_v1(&staged.lanes, &staged.tokens);
    let logits = logits_row_v1(&lanes, staged.vocabulary);
    operation.result = Some(StepResultV1 { lanes, logits });
    RECURRENT_STEP_OK_V1
}

// The C entry the host invokes, and the safe constructor that binds it to one
// live state value. The macro owns every pointer decode and contains a panic
// before it can reach native code, so nothing in this crate writes unsafe code.
runtime_bundle_abi::export_custom_operation_v1!(
    recurrent_step_operation_v1,
    RecurrentStepOperationV1,
    run_recurrent_step_v1,
);

/// Binds this fixture's operation entry to the state one execution staged.
pub(crate) fn bind_recurrent_step_v1(
    operation: &mut RecurrentStepOperationV1,
) -> runtime_bundle_abi::CustomOperationV1<'_> {
    recurrent_step_operation_v1(operation)
}

/// Names one operation code for a diagnostic, without inventing a meaning.
#[must_use]
pub fn recurrent_step_code_name_v1(code: i32) -> &'static str {
    match code {
        RECURRENT_STEP_OK_V1 => "ran",
        RECURRENT_STEP_WRONG_OPERAND_COUNT_V1 => "was lent the wrong number of operands",
        RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1 => "was lent an operand of the wrong length",
        RECURRENT_STEP_NOTHING_STAGED_V1 => "was invoked with nothing staged",
        _ => "reported a code this fixture does not define",
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{CustomAddressSpaceV1, LentLaunchV1, LentRangeV1, LentStreamV1};

    use crate::recurrent::{RECURRENT_LANES_V1, fold_tokens_v1};

    use super::{
        RECURRENT_STEP_NOTHING_STAGED_V1, RECURRENT_STEP_OK_V1,
        RECURRENT_STEP_WRONG_OPERAND_COUNT_V1, RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1,
        RecurrentStepOperationV1, StagedStepV1, run_recurrent_step_v1,
    };

    const VOCABULARY: u32 = 8;

    fn staged() -> StagedStepV1 {
        StagedStepV1 {
            lanes: [3; RECURRENT_LANES_V1],
            tokens: vec![4, 5],
            vocabulary: VOCABULARY,
            input_bytes: 8,
            output_bytes: u64::from(VOCABULARY) * 4,
        }
    }

    fn launch(lengths: &[u64]) -> Vec<LentRangeV1> {
        lengths
            .iter()
            .enumerate()
            .map(|(index, length)| LentRangeV1::new(index as u64 + 1, *length))
            .collect()
    }

    fn host_launch(ranges: &[LentRangeV1]) -> LentLaunchV1<'_> {
        LentLaunchV1 {
            address_space: CustomAddressSpaceV1::Host,
            stream: LentStreamV1::null(),
            ranges,
        }
    }

    #[test]
    fn a_staged_step_runs_and_reports_what_it_was_lent() {
        let mut operation = RecurrentStepOperationV1::new();
        operation.stage(staged());
        let ranges = launch(&[8, u64::from(VOCABULARY) * 4]);

        let code = run_recurrent_step_v1(&mut operation, host_launch(&ranges));

        assert_eq!(code, RECURRENT_STEP_OK_V1);
        assert_eq!(operation.invocations(), 1);
        let observed = operation.observed().expect("the lend was recorded").clone();
        assert_eq!(observed.address_space, CustomAddressSpaceV1::Host);
        assert_eq!(observed.operand_lengths, [8, u64::from(VOCABULARY) * 4]);
        let result = operation.take_result().expect("the step produced a result");
        assert_eq!(
            result.lanes,
            fold_tokens_v1(&[3; RECURRENT_LANES_V1], &[4, 5])
        );
        assert_eq!(result.logits.len(), VOCABULARY as usize * 4);
        // Taking the result also clears the staging, so a second call finds
        // nothing rather than replaying the step that already ran.
        assert_eq!(
            run_recurrent_step_v1(&mut operation, host_launch(&ranges)),
            RECURRENT_STEP_NOTHING_STAGED_V1
        );
    }

    #[test]
    fn a_lend_that_does_not_match_the_staged_geometry_produces_no_result() {
        for (lengths, expected) in [
            (vec![8], RECURRENT_STEP_WRONG_OPERAND_COUNT_V1),
            (
                vec![8, u64::from(VOCABULARY) * 4, 8],
                RECURRENT_STEP_WRONG_OPERAND_COUNT_V1,
            ),
            (
                vec![4, u64::from(VOCABULARY) * 4],
                RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1,
            ),
            (vec![8, 4], RECURRENT_STEP_WRONG_OPERAND_LENGTH_V1),
        ] {
            let mut operation = RecurrentStepOperationV1::new();
            operation.stage(staged());
            let ranges = launch(&lengths);

            let code = run_recurrent_step_v1(&mut operation, host_launch(&ranges));

            assert_eq!(code, expected, "lengths {lengths:?}");
            assert!(
                operation.take_result().is_none(),
                "a refused lend must leave no result: {lengths:?}"
            );
        }
    }
}
