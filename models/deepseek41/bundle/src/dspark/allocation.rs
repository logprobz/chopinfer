//! Prepared ops and storage of allocated verify steps. Each DSpark
//! instance prepares the verify allocation, the gather of packed feature
//! rows into the reserved layout and the allocated acceptance at create,
//! and allocates their storage on every rank before the KV pool.
//!
//! The target and vocabulary head compute only allocated rows. The sampler
//! reads those logits through the computed-to-reserved map; packed feature
//! rows are gathered into the reserved layout that commit consumes.
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

#[derive(Default)]
pub(crate) struct RuleCache {
    calibration: Option<runtime_bundle_sdk::VerifyCalibrationV1>,
    words: [u32; RULE_WORDS],
}
impl RuleCache {
    fn encode(&mut self, allocation: &VerifyAllocationV1) -> [u32; RULE_WORDS] {
        if self.calibration.as_ref() != Some(allocation.calibration()) {
            self.words = encode_rule(allocation);
            self.calibration = Some(allocation.calibration().clone());
        }
        self.words[0] = allocation.width();
        self.words[1] = allocation.floor();
        self.words
    }
}

#[derive(Default)]
pub(crate) struct MetadataStaging {
    pub bytes: Vec<u8>,
    pub calibration: Vec<Option<[u32; RULE_WORDS - 2]>>,
}

pub(crate) struct Allocation {
    pub op: VerifyAllocate,
    pub allocate: Prepared<VerifyAllocateBuffers<View>>,
    /// Packed BF16 feature rows gathered by index.
    pub features: Prepared<GatherByIndexBuffers<View>>,
    pub acceptance: Prepared<AllocatedAcceptanceBuffers<View>>,
    pub ranks: Vec<Memory>,
    pub rule_cache: std::cell::RefCell<RuleCache>,
    pub staging: std::cell::RefCell<MetadataStaging>,
}

/// One rank's allocation inputs and outputs (`VerifyAllocateBuffers`), the
/// device count of the step's reserved rows; the confidence scores and
/// the compacted rows are the drafter's
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
    pub metadata: Span,
}

impl Allocation {
    /// The ops for steps of up to `items` items (the instance's input
    /// capacity) and `verify_rows` reserved rows (its verify rows, at most
    /// a full block each), and as wide, over `features` bytes of packed
    /// features a row.
    pub fn new(
        backend: &dyn kernel_core::backend::Backend,
        own: &dyn OwnOps,
        vocab: u32,
        (items, verify_rows): (u32, u32),
        features: u64,
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
            ranks: Vec::new(),
            rule_cache: Default::default(),
            staging: Default::default(),
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
        let item_bytes = items * ITEM_WORDS as u64 * 4;
        let metadata = alloc(item_bytes + 8 + RULE_WORDS as u64 * 4)?;
        let sub = |offset, length| Span::new(metadata.buffer, metadata.offset + offset, length);
        self.staging
            .borrow_mut()
            .bytes
            .resize(item_bytes as usize + 16, 0);
        self.staging.borrow_mut().calibration.push(None);
        self.ranks.push(Memory {
            items: sub(0, item_bytes),
            rule: sub(item_bytes + 8, RULE_WORDS as u64 * 4),
            live_items: sub(item_bytes, 4),
            status: alloc(4)?,
            verified: alloc(items * 4)?,
            starts: alloc(items * 4)?,
            rows: alloc(u64::from(self.op.width) * 4)?,
            expand: alloc(u64::from(self.op.reserved_rows) * 4)?,
            reserved: sub(item_bytes + 4, 4),
            metadata,
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
    pub fn with_rule_cache(
        batch: &BatchRequestV1,
        decode: &PhaseInputs,
        cache: &std::cell::RefCell<RuleCache>,
    ) -> Result<Option<Self>, String> {
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
            rule: cache.borrow_mut().encode(allocation),
        }))
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

    #[test]
    fn cached_calibration_keeps_width_and_floor_step_local() {
        let cache = std::cell::RefCell::new(RuleCache::default());
        for width in [7, 8, 7] {
            let (batch, decode) = step(width);
            let actual = Step::with_rule_cache(&batch, &decode, &cache)
                .unwrap()
                .unwrap();
            assert_eq!(actual.rule, encode_rule(batch.verify_allocation().unwrap()));
            assert_eq!(actual.rule[0], width);
        }
    }

    /// Each item reserves its drafts at its packed first row, with its own
    /// kind, and the drafting items take confidence rows in order.
    #[test]
    fn a_step_reads_each_items_reservation_from_the_packed_layout() {
        let (batch, decode) = step(7);
        let step = Step::with_rule_cache(&batch, &decode, &Default::default())
            .unwrap()
            .unwrap();
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
        assert_eq!(step.items[1].encode(), [0, 0, NONE, 6]);
        assert_eq!(step.rule[..2], [7, 1]);
        let (unallocated, decode) = step_without_allocation();
        assert_eq!(
            Step::with_rule_cache(&unallocated, &decode, &Default::default()),
            Ok(None)
        );
    }

    /// A width outside the items' rows or packed rows that differ from the
    /// reservations are refused.
    #[test]
    fn inconsistent_steps_are_refused() {
        for width in [2, 12] {
            let (batch, decode) = step(width);
            assert!(
                Step::with_rule_cache(&batch, &decode, &Default::default()).is_err(),
                "{width}"
            );
        }
        let (batch, mut decode) = step(7);
        decode.q_start[2] = 6;
        assert!(Step::with_rule_cache(&batch, &decode, &Default::default()).is_err());
        let (batch, mut decode) = step(7);
        decode.positions.pop();
        assert!(Step::with_rule_cache(&batch, &decode, &Default::default()).is_err());
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
