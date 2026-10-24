//! Prepared ops and storage of allocated verify steps
//! (`model_deepseek41_contract::verify_allocation`): an instance whose host
//! allocates prepares the verify allocation, the gather of packed feature
//! rows into the reserved layout and the allocated acceptance at create,
//! and allocates their storage on every rank before the KV pool. An
//! instance whose host does not allocate holds none of it.
//!
//! An allocated step keeps the reserved layout everywhere but the target
//! body: the host uploads its reserved rows, the allocation compacts the
//! target's per-row inputs to the computed rows on every rank, the target
//! computes the width, and its final-norm rows and packed features are
//! gathered back into the reserved rows by `expand` before the head and
//! the commit read them.
use super::results::{ITEMS, ROWS, STOP_CAPACITY};
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::resources::{Held, allocate_span};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::contract::DType;
use kernel_core::row_copy::{GatherByIndex, GatherByIndexBuffers};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::acceptance::Acceptance;
use model_deepseek41_contract::own_ops::acceptance::allocated::AllocatedAcceptanceBuffers;
use model_deepseek41_contract::own_ops::allocation::{
    BLOCK, ITEM_WORDS, Item, NONE, RULE_WORDS, VerifyAllocate, VerifyAllocateBuffers, encode_rule,
};
use runtime_bundle_abi::{
    BundleAbiError, BundleMemoryPoolHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1 as Span,
    HostServicesV1,
};
use runtime_bundle_sdk::{BatchRequestV1, VerifyAllocationV1};

pub(crate) struct Allocation {
    pub op: VerifyAllocate,
    pub allocate: Prepared<VerifyAllocateBuffers<View>>,
    /// Packed BF16 feature rows gathered by index.
    pub features: Prepared<GatherByIndexBuffers<View>>,
    pub acceptance: Prepared<AllocatedAcceptanceBuffers<View>>,
    /// The bytes of one F32 final-norm row.
    head_row_bytes: u64,
    pub ranks: Vec<Memory>,
}

/// One rank's allocation inputs and outputs (`VerifyAllocateBuffers`), the
/// device count of the step's reserved rows and the head's reserved input
/// rows; the confidence scores and the compacted rows are the drafter's
/// and the target's own storage.
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    pub items: Span,
    pub rule: Span,
    pub live_items: Span,
    pub status: Span,
    pub verified: Span,
    pub starts: Span,
    pub rows: Span,
    pub expand: Span,
    pub reserved: Span,
    pub head: Span,
}

impl Allocation {
    /// The ops for steps of up to `items` items (the instance's input
    /// capacity) and `verify_rows` reserved rows (its verify rows, at most
    /// a full block each), and as wide, over `features` bytes of packed
    /// features and `hidden` F32 final-norm values a row.
    pub fn new(
        backend: &dyn kernel_core::backend::Backend,
        own: &dyn OwnOps,
        vocab: u32,
        (items, verify_rows): (u32, u32),
        features: u64,
        hidden: u32,
    ) -> Result<Self, String> {
        let verify_rows = verify_rows.min(items * (1 + BLOCK as u32));
        let op = VerifyAllocate {
            items,
            reserved_rows: verify_rows,
            width: verify_rows,
        };
        let allocate = own
            .dspark_allocate(&op, Need::rows(items))
            .map_err(|e| format!("DSpark verify allocation: {e:?}"))?;
        let width = u32::try_from(features / 2).map_err(|_| "DSpark feature rows are too wide")?;
        let features = backend
            .gather_by_index(
                &GatherByIndex {
                    dtype: DType::Bf16,
                    width,
                    alignment_bytes: model_deepseek41_contract::rows::ALIGNMENT_BYTES,
                },
                Need::rows(verify_rows),
            )
            .map_err(|e| format!("DSpark reserved feature rows: {e:?}"))?;
        let acceptance = own
            .dspark_allocated_acceptance(
                &Acceptance {
                    vocab,
                    sample_rows: ROWS,
                    draft_items: ITEMS,
                    stop_ids: STOP_CAPACITY,
                },
                Need::rows(ITEMS),
            )
            .map_err(|e| format!("DSpark allocated acceptance: {e:?}"))?;
        Ok(Self {
            op,
            allocate,
            features,
            acceptance,
            head_row_bytes: u64::from(hidden) * 4,
            ranks: Vec::new(),
        })
    }

    /// One more rank's storage.
    pub fn allocate(
        &mut self,
        services: &HostServicesV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        held: &mut Vec<Held>,
    ) -> Result<(), BundleAbiError> {
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        let items = u64::from(self.op.items);
        self.ranks.push(Memory {
            items: alloc(items * ITEM_WORDS as u64 * 4)?,
            rule: alloc(RULE_WORDS as u64 * 4)?,
            live_items: alloc(4)?,
            status: alloc(4)?,
            verified: alloc(items * 4)?,
            starts: alloc(items * 4)?,
            rows: alloc(u64::from(self.op.width) * 4)?,
            expand: alloc(u64::from(self.op.reserved_rows) * 4)?,
            reserved: alloc(4)?,
            head: alloc(u64::from(self.op.reserved_rows) * self.head_row_bytes)?,
        });
        Ok(())
    }
}

/// One allocated verify step as the allocation reads it: each decode-phase
/// item's reservation (`allocation::Item`), the rule record, the width the
/// target computes and the reserved rows the host laid out.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Step {
    pub width: u32,
    pub reserved_rows: u32,
    pub items: Vec<Item>,
    pub rule: [u32; RULE_WORDS],
}

impl Step {
    /// `batch`'s allocation over its decode phase `decode`, packed in the
    /// reserved layout: every item's rows are its one row and its reserved
    /// drafts, and the items with drafts take the drafter's confidence rows
    /// in order. A step with no allocation is `None`.
    pub fn of(batch: &BatchRequestV1, decode: &PhaseInputs) -> Result<Option<Self>, String> {
        let Some(allocation) = batch.verify_allocation() else {
            return Ok(None);
        };
        allocation
            .validate(batch)
            .map_err(|_| "DSpark verify allocation does not fit its batch")?;
        let reservations = VerifyAllocationV1::reservations(batch);
        if !decode.decode || reservations.len() != decode.request_ids.len() {
            return Err("DSpark verify allocation needs one decode phase".into());
        }
        let mut items = Vec::with_capacity(reservations.len());
        let mut drafting = 0;
        let mut first_row = 0;
        for (item, &(index, reserved)) in reservations.iter().enumerate() {
            if decode.q_len.get(item) != Some(&(1 + reserved as i32))
                || decode.q_start.get(item) != Some(&(first_row as i32))
            {
                return Err("DSpark verify allocation rows differ from the packed items".into());
            }
            let score_row = if reserved == 0 {
                NONE
            } else {
                drafting += 1;
                drafting - 1
            };
            items.push(Item {
                reserved,
                kind: allocation.kinds()[index],
                score_row,
                first_row,
            });
            first_row += 1 + reserved;
        }
        if decode.positions.len() != first_row as usize {
            return Err("DSpark verify allocation rows differ from the packed rows".into());
        }
        Ok(Some(Self {
            width: allocation.width(),
            reserved_rows: first_row,
            items,
            rule: encode_rule(allocation),
        }))
    }

    /// The items as the allocation's U32 item rows.
    pub fn item_words(&self) -> Vec<u32> {
        self.items.iter().flat_map(|item| item.encode()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::own_ops::allocation::SamplingKindV1;
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchWireVersionV1, VerifyCalibrationV1};

    /// Three decode items reserving 5, 0 and 3 drafts, in the layout the
    /// host packs them.
    fn step(width: u32) -> (BatchRequestV1, PhaseInputs) {
        let reserved = [5_u32, 0, 3];
        let items = reserved
            .iter()
            .enumerate()
            .map(|(item, &drafts)| {
                if drafts == 0 {
                    BatchItemV1::new(item as u64 + 1, true, 1, true, vec![7])
                } else {
                    BatchItemV1::verify(item as u64 + 1, 1 + drafts, true, vec![7])
                }
            })
            .collect();
        let appends = reserved
            .iter()
            .enumerate()
            .map(|(item, &drafts)| {
                BatchAppendV1::new(
                    40,
                    u64::from(1 + drafts),
                    u64::from(41 + drafts),
                    vec![item as u64],
                    None,
                )
            })
            .collect();
        let calibration = VerifyCalibrationV1::new(&[(1.0, 0.0); 5], &[(2.0, -1.0); 5]).unwrap();
        let batch = BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
            .with_state_slots(vec![0, 1, 2])
            .with_verify_allocation(VerifyAllocationV1::new(
                width,
                1,
                calibration,
                vec![
                    SamplingKindV1::Sampled,
                    SamplingKindV1::Greedy,
                    SamplingKindV1::Greedy,
                ],
            ));
        let mut decode = PhaseInputs {
            decode: true,
            verify: true,
            request_ids: vec![1, 2, 3],
            ..PhaseInputs::default()
        };
        let mut first = 0;
        for drafts in reserved {
            decode.q_start.push(first);
            decode.q_len.push(1 + drafts as i32);
            decode
                .positions
                .extend((0..=drafts as i32).map(|row| 40 + row));
            first += 1 + drafts as i32;
        }
        (batch, decode)
    }

    /// Each item reserves its drafts at its packed first row, with its own
    /// kind, and the drafting items take confidence rows in order.
    #[test]
    fn a_step_reads_each_items_reservation_from_the_packed_layout() {
        let (batch, decode) = step(7);
        let step = Step::of(&batch, &decode).unwrap().unwrap();
        assert_eq!((step.width, step.reserved_rows), (7, 11));
        assert_eq!(
            step.items,
            [
                Item {
                    reserved: 5,
                    kind: SamplingKindV1::Sampled,
                    score_row: 0,
                    first_row: 0,
                },
                Item {
                    reserved: 0,
                    kind: SamplingKindV1::Greedy,
                    score_row: NONE,
                    first_row: 6,
                },
                Item {
                    reserved: 3,
                    kind: SamplingKindV1::Greedy,
                    score_row: 1,
                    first_row: 7,
                },
            ]
        );
        assert_eq!(step.item_words()[4..8], [0, 0, NONE, 6]);
        assert_eq!(step.rule[..2], [7, 1]);
        let (unallocated, decode) = step_without_allocation();
        assert_eq!(Step::of(&unallocated, &decode), Ok(None));
    }

    /// A width outside the items' rows or packed rows that differ from the
    /// reservations are refused.
    #[test]
    fn inconsistent_steps_are_refused() {
        for width in [2, 12] {
            let (batch, decode) = step(width);
            assert!(Step::of(&batch, &decode).is_err(), "{width}");
        }
        let (batch, mut decode) = step(7);
        decode.q_start[2] = 6;
        assert!(Step::of(&batch, &decode).is_err());
        let (batch, mut decode) = step(7);
        decode.positions.pop();
        assert!(Step::of(&batch, &decode).is_err());
    }

    fn step_without_allocation() -> (BatchRequestV1, PhaseInputs) {
        let (batch, decode) = step(7);
        let plain = BatchRequestV1::new(
            BatchWireVersionV1::V3,
            batch.items().to_vec(),
            batch.appends().to_vec(),
        )
        .with_state_slots(vec![0, 1, 2]);
        (plain, decode)
    }
}
