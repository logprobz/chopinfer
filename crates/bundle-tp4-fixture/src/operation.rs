//! The one custom operation this fixture supplies for itself.
//!
//! The manifest declares this operation as a bundle custom operation, which is
//! the escape hatch a host never implements: no host-optimized or generic
//! candidate may satisfy it, so an installed bundle that does not carry it
//! leaves the operation unsupported. It is explicitly private because there is
//! nothing model-neutral about it - it produces a synthetic row of TP4's
//! declared width and nothing else could be asked to stand in for that.
//!
//! The operation reads through no lent address. Doing that would be unsafe code
//! this fixture does not write, and the address space it is lent depends on the
//! host: a driver lends device addresses, and a host-memory backend lends its
//! own. What ties the computation to the operands is the lend itself, checked
//! against the geometry the instance staged before the call: the operand count,
//! each operand's exact byte length, and that something was staged at all. A
//! lend that does not match is refused with a code of this fixture's own, and
//! the instance then emits no row.
//!
//! The shape is fixed and it is the declared one. The tokens operand is one
//! PrefillChunk's capacity of identifiers whatever the batch carries, zero past the
//! live ones, and a second operand carries how many of them are live. A DecodeStep
//! and a one-token prefill tail are therefore lent exactly what a full chunk is,
//! and exactly what the manifest declares, so one operation stays exact instead
//! of declaring one shape and being handed another.

use runtime_bundle_abi::{CustomAddressSpaceV1, LentLaunchV1};

use crate::synthetic::{TokenHistoryV1, fold_tokens_v1, synthetic_row_v1};

/// The identifier the host resolves, audits, and logs this operation by.
pub const TP4_FIXTURE_ROW_OPERATION_ID_V1: &[u8] = b"fixture.tp4.synthetic-row.v1";

/// How many operands the operation is lent: the fixed-capacity tokens, how many
/// of them are live, and the row it writes.
pub const TP4_FIXTURE_ROW_OPERANDS_V1: usize = 3;

/// The operation ran and staged its result.
pub const TP4_FIXTURE_ROW_OK_V1: i32 = 0;

/// The operation was lent a different number of operands than it declared.
pub const TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1: i32 = 1;

/// One lent operand was not the byte length the instance staged for it.
pub const TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1: i32 = 2;

/// The operation was invoked with nothing staged, so there was nothing to run.
pub const TP4_FIXTURE_ROW_NOTHING_STAGED_V1: i32 = 3;

/// More tokens were staged than the fixed-capacity operand can carry.
pub const TP4_FIXTURE_ROW_TOKENS_OVER_CAPACITY_V1: i32 = 4;

/// What one execution stages for the operation before it is invoked.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedRowV1 {
    /// Every rank image's digest, in rank order, as this instance
    /// authenticated them at create.
    pub image_digests: Vec<String>,
    /// The folded history of every token a committed execution appended.
    pub history: TokenHistoryV1,
    /// The live tokens of this batch, which are the ones the row is computed
    /// from and the ones the count operand counts.
    pub tokens: Vec<u32>,
    pub vocabulary: u32,
    /// The fixed capacity of the tokens operand, in bytes.
    pub input_bytes: u64,
    /// The byte length of the operand carrying the live token count.
    pub token_count_bytes: u64,
    pub output_bytes: u64,
}

impl StagedRowV1 {
    /// The fixed-capacity token operand this staging is lent, zero-padded.
    ///
    /// The instance copies exactly these bytes into the buffer the operation
    /// reads, so the padding is real zero in device memory rather than whatever
    /// the previous batch left there.
    #[must_use]
    pub fn input_operand_bytes(&self) -> Vec<u8> {
        let mut operand = vec![0_u8; self.input_bytes as usize];
        for (slot, token) in operand.chunks_exact_mut(4).zip(&self.tokens) {
            slot.copy_from_slice(&token.to_le_bytes());
        }
        operand
    }

    /// The live token count operand, as the bytes the operation is lent.
    #[must_use]
    pub fn token_count_operand_bytes(&self) -> Vec<u8> {
        let count = u32::try_from(self.tokens.len()).unwrap_or(u32::MAX);
        let mut operand = vec![0_u8; self.token_count_bytes as usize];
        operand[..4].copy_from_slice(&count.to_le_bytes());
        operand
    }
}

/// What one execution takes back from the operation after it returns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RowResultV1 {
    /// The history this batch would publish, if it is committed.
    pub history: TokenHistoryV1,
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
pub struct Tp4RowOperationV1 {
    staged: Option<StagedRowV1>,
    result: Option<RowResultV1>,
    observed: Option<ObservedLendV1>,
    invocations: usize,
}

impl Tp4RowOperationV1 {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            staged: None,
            result: None,
            observed: None,
            invocations: 0,
        }
    }

    /// Stages one row and clears whatever the previous call left behind.
    pub fn stage(&mut self, staged: StagedRowV1) {
        self.staged = Some(staged);
        self.result = None;
        self.observed = None;
    }

    /// Takes the result of the call that just ran, if it produced one.
    pub fn take_result(&mut self) -> Option<RowResultV1> {
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

/// Runs one synthetic row against the operands the host lent.
fn run_synthetic_row_v1(operation: &mut Tp4RowOperationV1, launch: LentLaunchV1<'_>) -> i32 {
    operation.invocations += 1;
    operation.observed = Some(ObservedLendV1 {
        address_space: launch.address_space,
        operand_lengths: launch.ranges.iter().map(|range| range.length).collect(),
    });
    let Some(staged) = operation.staged.as_ref() else {
        return TP4_FIXTURE_ROW_NOTHING_STAGED_V1;
    };
    if launch.ranges.len() != TP4_FIXTURE_ROW_OPERANDS_V1 {
        return TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1;
    }
    if launch.ranges[0].length != staged.input_bytes
        || launch.ranges[1].length != staged.token_count_bytes
        || launch.ranges[2].length != staged.output_bytes
    {
        return TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1;
    }
    // The count operand says how many of the fixed-capacity slots are live, so
    // a batch that would not fit them is one this operation cannot describe.
    if (staged.tokens.len() as u64) * 4 > staged.input_bytes {
        return TP4_FIXTURE_ROW_TOKENS_OVER_CAPACITY_V1;
    }
    let logits = synthetic_row_v1(
        &staged.image_digests,
        staged.history,
        &staged.tokens,
        staged.vocabulary,
    );
    operation.result = Some(RowResultV1 {
        history: fold_tokens_v1(staged.history, &staged.tokens),
        logits,
    });
    TP4_FIXTURE_ROW_OK_V1
}

// The C entry the host invokes, and the safe constructor that binds it to one
// live state value. The macro owns every pointer decode and contains a panic
// before it can reach native code, so nothing in this crate writes unsafe code.
runtime_bundle_abi::export_custom_operation_v1!(
    tp4_row_operation_v1,
    Tp4RowOperationV1,
    run_synthetic_row_v1,
);

/// Binds this fixture's operation entry to the state one execution staged.
pub(crate) fn bind_tp4_row_v1(
    operation: &mut Tp4RowOperationV1,
) -> runtime_bundle_abi::CustomOperationV1<'_> {
    tp4_row_operation_v1(operation)
}

/// Names one operation code for a diagnostic, without inventing a meaning.
#[must_use]
pub fn tp4_row_code_name_v1(code: i32) -> &'static str {
    match code {
        TP4_FIXTURE_ROW_OK_V1 => "ran",
        TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1 => "was lent the wrong number of operands",
        TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1 => "was lent an operand of the wrong length",
        TP4_FIXTURE_ROW_NOTHING_STAGED_V1 => "was invoked with nothing staged",
        TP4_FIXTURE_ROW_TOKENS_OVER_CAPACITY_V1 => {
            "was staged more tokens than its operand carries"
        }
        _ => "reported a code this fixture does not define",
    }
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::{CustomAddressSpaceV1, LentLaunchV1, LentRangeV1, LentStreamV1};

    use crate::contract::{TP4_FIXTURE_ROW_INPUT_BYTES_V1, TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1};
    use crate::synthetic::{fold_tokens_v1, program_image_digests_v1, state_seed_history_v1};

    use super::{
        StagedRowV1, TP4_FIXTURE_ROW_NOTHING_STAGED_V1, TP4_FIXTURE_ROW_OK_V1,
        TP4_FIXTURE_ROW_TOKENS_OVER_CAPACITY_V1, TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1,
        TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1, Tp4RowOperationV1, run_synthetic_row_v1,
    };

    const VOCABULARY: u32 = 8;
    const INPUT_BYTES: u64 = TP4_FIXTURE_ROW_INPUT_BYTES_V1;
    const COUNT_BYTES: u64 = TP4_FIXTURE_ROW_TOKEN_COUNT_BYTES_V1;

    fn staged() -> StagedRowV1 {
        staged_tokens(vec![4, 5])
    }

    fn staged_tokens(tokens: Vec<u32>) -> StagedRowV1 {
        StagedRowV1 {
            image_digests: program_image_digests_v1(),
            history: state_seed_history_v1(),
            tokens,
            vocabulary: VOCABULARY,
            input_bytes: INPUT_BYTES,
            token_count_bytes: COUNT_BYTES,
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
    fn a_staged_row_runs_and_reports_what_it_was_lent() {
        let mut operation = Tp4RowOperationV1::new();
        operation.stage(staged());
        let ranges = launch(&[INPUT_BYTES, COUNT_BYTES, u64::from(VOCABULARY) * 4]);

        let code = run_synthetic_row_v1(&mut operation, host_launch(&ranges));

        assert_eq!(code, TP4_FIXTURE_ROW_OK_V1);
        assert_eq!(operation.invocations(), 1);
        let observed = operation.observed().expect("the lend was recorded").clone();
        assert_eq!(observed.address_space, CustomAddressSpaceV1::Host);
        assert_eq!(
            observed.operand_lengths,
            [INPUT_BYTES, COUNT_BYTES, u64::from(VOCABULARY) * 4]
        );
        let result = operation.take_result().expect("the row was produced");
        assert_eq!(result.logits.len(), VOCABULARY as usize * 4);
        assert_eq!(
            result.history,
            fold_tokens_v1(state_seed_history_v1(), &[4, 5])
        );
        // Taking the result also clears the staging, so a second call finds
        // nothing rather than replaying the row that already ran.
        assert_eq!(
            run_synthetic_row_v1(&mut operation, host_launch(&ranges)),
            TP4_FIXTURE_ROW_NOTHING_STAGED_V1
        );
    }

    #[test]
    fn a_lend_that_does_not_match_the_staged_geometry_produces_no_result() {
        let row_bytes = u64::from(VOCABULARY) * 4;
        for (lengths, expected) in [
            (
                vec![INPUT_BYTES, COUNT_BYTES],
                TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1,
            ),
            (
                vec![INPUT_BYTES, COUNT_BYTES, row_bytes, INPUT_BYTES],
                TP4_FIXTURE_ROW_WRONG_OPERAND_COUNT_V1,
            ),
            // The whole point of the fixed capacity: a lend narrowed to the two
            // tokens this batch carries is not the shape the contract declares.
            (
                vec![8, COUNT_BYTES, row_bytes],
                TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1,
            ),
            (
                vec![INPUT_BYTES, 8, row_bytes],
                TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1,
            ),
            (
                vec![INPUT_BYTES, COUNT_BYTES, 4],
                TP4_FIXTURE_ROW_WRONG_OPERAND_LENGTH_V1,
            ),
        ] {
            let mut operation = Tp4RowOperationV1::new();
            operation.stage(staged());
            let ranges = launch(&lengths);

            let code = run_synthetic_row_v1(&mut operation, host_launch(&ranges));

            assert_eq!(code, expected, "lengths {lengths:?}");
            assert!(
                operation.take_result().is_none(),
                "a refused lend must leave no result: {lengths:?}"
            );
        }
    }

    /// The fixed-capacity operand is what the tokens are written into, padded
    /// with zero, and the count operand says how many of them are live.
    #[test]
    fn the_operands_carry_the_live_tokens_their_count_and_zero_padding() {
        let staged = staged();

        let tokens = staged.input_operand_bytes();
        let count = staged.token_count_operand_bytes();

        assert_eq!(tokens.len() as u64, INPUT_BYTES);
        assert_eq!(
            &tokens[..8],
            [4_u32.to_le_bytes(), 5_u32.to_le_bytes()].concat()
        );
        assert!(
            tokens[8..].iter().all(|byte| *byte == 0),
            "every slot past the live tokens is zero"
        );
        assert_eq!(count.len() as u64, COUNT_BYTES);
        assert_eq!(count, 2_u32.to_le_bytes());
    }

    /// A batch wider than the declared capacity is refused rather than truncated
    /// into an operand that could not carry it.
    #[test]
    fn more_tokens_than_the_operand_carries_are_refused() {
        let over_capacity = (0..=(INPUT_BYTES as u32 / 4)).collect::<Vec<_>>();
        let mut operation = Tp4RowOperationV1::new();
        operation.stage(staged_tokens(over_capacity));
        let ranges = launch(&[INPUT_BYTES, COUNT_BYTES, u64::from(VOCABULARY) * 4]);

        let code = run_synthetic_row_v1(&mut operation, host_launch(&ranges));

        assert_eq!(code, TP4_FIXTURE_ROW_TOKENS_OVER_CAPACITY_V1);
        assert!(operation.take_result().is_none());
    }
}
