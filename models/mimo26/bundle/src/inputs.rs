//! MiMo's mixed-step inputs: the shared packing, with sequence slots from
//! the sliding-window ring's ledger.

pub use bundle_runtime::inputs::{InputError, PackedStep, PhaseInputs, VerifyLimits};
use bundle_runtime::ledger::{Ledger, LedgerError};
use runtime_bundle_sdk::BatchRequestV1;

use crate::contract::Mimo;

/// Previews new sequence slots before validating and packing the entire
/// step. The caller opens the ring transaction only after this succeeds. A
/// step holds at most the `served` items the ring has slots for.
pub fn pack_for_ring_within(
    limits: bundle_runtime::ModelLimits,
    request: &BatchRequestV1,
    ledger: &Ledger,
    page_capacity: u64,
    verify: VerifyLimits,
    served: bundle_runtime::geometry::Items,
) -> Result<PackedStep, InputError> {
    if request.items().len() != request.appends().len() {
        return Err(InputError::InvalidBatch);
    }
    let items = crate::ring::items(request);
    let slots = ledger.preview(&items).map_err(|error| match error {
        LedgerError::NoSlot | LedgerError::UnknownRequest => InputError::MissingSlot,
        _ => InputError::InvalidAppend,
    })?;
    PackedStep::pack_within::<Mimo>(limits, request, page_capacity, verify, served, |id| {
        items
            .iter()
            .zip(&slots)
            .find(|(item, _)| item.request == id)
            .map(|(_, slot)| *slot)
    })
}

#[cfg(test)]
mod tests {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchWireVersionV1};

    use super::*;

    #[test]
    fn new_sequence_slots_pack_without_mutating_ring() {
        let request = BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![
                BatchItemV1::new(10, false, 1, true, vec![1]),
                BatchItemV1::new(11, true, 1, true, vec![2]),
            ],
            vec![
                BatchAppendV1::new(0, 1, 1, vec![5], None),
                BatchAppendV1::new(0, 1, 1, vec![6], None),
            ],
        );
        let served = bundle_runtime::geometry::Items::new(32).unwrap();
        let state = model_mimo26_contract::state::request_state(
            crate::topology::Tp::TP4,
            &crate::contract::KV_FORMATS[0],
            crate::prefix::Drafter::None,
        );
        let mut ledger = Ledger::new(
            served.get(),
            bundle_runtime::ledger::Placement::Ledger,
            bundle_runtime::ledger::Rules::of(&state),
            crate::contract::CONTEXT_TOKENS,
        );
        let packed = pack_for_ring_within(
            crate::contract::MODEL_LIMITS,
            &request,
            &ledger,
            32_768,
            VerifyLimits::NONE,
            served,
        )
        .expect("new sequence slots");
        assert_eq!(packed.prefill.slot_ids, [0]);
        assert_eq!(packed.decode.slot_ids, [1]);
        assert_eq!(ledger.slot_of(10), None);
        ledger
            .begin(&crate::ring::items(&request), Vec::new())
            .expect("open transaction after validation");
        assert_eq!(ledger.slot_of(10), Some(0));
        assert_eq!(ledger.slot_of(11), Some(1));
        ledger.cancel().expect("rollback new slots");
        assert_eq!(ledger.slot_of(10), None);
    }
}
