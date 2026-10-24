use super::super::{Item, oracle as reducer};
use super::*;
use runtime_bundle_abi::device_results::{
    BUNDLE_ACCEPTANCE_ALLOCATED_V1, BundleAcceptanceInputV1, BundleAcceptanceResultV1,
    DeviceAcceptanceV1,
};

/// A small deterministic generator, so failures reproduce.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

const VOCAB: u32 = 100;

fn sample(token: u32, status: u32) -> [u8; OUTPUT_BYTES] {
    let mut bytes = [0xa5; OUTPUT_BYTES];
    bytes[..4].copy_from_slice(&token.to_le_bytes());
    bytes[4..8].copy_from_slice(&status.to_le_bytes());
    bytes[8..16].copy_from_slice(&(-0.25_f64).to_le_bytes());
    bytes
}

/// One allocated step in the reserved layout: its operation, descriptors,
/// each item's verified drafts, proposals, sample rows and stop ids.
struct Step {
    op: Acceptance,
    items: Vec<Item>,
    verified: Vec<u32>,
    proposals: Vec<[u32; BLOCK]>,
    samples: Vec<u8>,
    stops: Vec<u32>,
}

impl Step {
    fn random(rng: &mut Rng) -> Self {
        let count = 1 + rng.below(32) as u32;
        let (mut items, mut verified, mut proposals) = (Vec::new(), Vec::new(), Vec::new());
        let (mut samples, mut stops) = (Vec::new(), Vec::new());
        for index in 0..count {
            let k = rng.below(BLOCK as u64 + 1) as u32;
            let first_sample = (samples.len() / OUTPUT_BYTES) as u32;
            let draft_item = if k == 0 {
                NONE
            } else {
                proposals.push(std::array::from_fn(|_| 10 + rng.below(4) as u32));
                proposals.len() as u32 - 1
            };
            let stop = rng.below(3) == 0;
            stops.push(if stop { 10 + rng.below(4) as u32 } else { 90 });
            items.push(Item {
                epoch: 7 + u64::from(index),
                remaining: 1 + rng.below(8),
                first_sample,
                draft_item,
                k,
                stop_offset: index,
                stop_count: 1,
                reserved: 0,
            });
            verified.push(rng.below(u64::from(k) + 1) as u32);
            for _ in 0..=k {
                let status = if rng.below(12) == 0 { 2 } else { 0 };
                samples.extend(sample(10 + rng.below(4) as u32, status));
            }
        }
        let sample_rows = (samples.len() / OUTPUT_BYTES) as u32;
        Self {
            op: Acceptance {
                vocab: VOCAB,
                sample_rows,
                draft_items: proposals.len() as u32,
                stop_ids: count,
            },
            items,
            verified,
            proposals,
            samples,
            stops,
        }
    }

    fn input(&self) -> Input<'_> {
        Input {
            items: &self.items,
            proposals: &self.proposals,
            samples: &self.samples,
            stops: &self.stops,
        }
    }

    fn allocated(&self, status: u32) -> Vec<Decision> {
        oracle(self.op, self.input(), &self.verified, status).unwrap()
    }

    /// The same step with every item's unverified rows and drafts removed:
    /// the reducer's walk over K'_i drafts, in the host encoding.
    fn compact(&self) -> Vec<Decision> {
        let (mut items, mut proposals, mut samples) = (Vec::new(), Vec::new(), Vec::new());
        for (item, &k) in self.items.iter().zip(&self.verified) {
            let first_sample = (samples.len() / OUTPUT_BYTES) as u32;
            let draft_item = if k == 0 {
                NONE
            } else {
                proposals.push(self.proposals[item.draft_item as usize]);
                proposals.len() as u32 - 1
            };
            let rows = item.first_sample as usize * OUTPUT_BYTES..;
            samples.extend_from_slice(&self.samples[rows][..(k as usize + 1) * OUTPUT_BYTES]);
            items.push(Item {
                first_sample,
                draft_item,
                k,
                ..*item
            });
        }
        let op = Acceptance {
            sample_rows: (samples.len() / OUTPUT_BYTES) as u32,
            draft_items: proposals.len() as u32,
            ..self.op
        };
        let input = Input {
            items: &items,
            proposals: &proposals,
            samples: &samples,
            stops: &self.stops,
        };
        reducer(op, input)
            .unwrap()
            .into_iter()
            .zip(&items)
            .map(|(mut decision, item)| {
                // The host encoding swaps the reducer's sample failures and
                // makes an EMPTY row relative to its item.
                if decision.failure == super::super::EMPTY {
                    decision.failure = EMPTY;
                    decision.failure_row -= item.first_sample;
                }
                decision
            })
            .collect()
    }
}

/// With valid reserved rows, an allocated walk is the reducer's walk over
/// each item's verified drafts alone, marked allocated with K'_i; the host
/// accepts every result as allocated with the verified drafts it names.
#[test]
fn allocated_walks_equal_the_reducer_over_the_verified_drafts() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let (mut short, mut empty) = (0, 0);
    for _ in 0..3000 {
        let step = Step::random(&mut rng);
        let allocated = step.allocated(super::super::super::allocation::OK);
        for (((decision, compact), item), &verified) in allocated
            .iter()
            .zip(step.compact())
            .zip(&step.items)
            .zip(&step.verified)
        {
            assert_eq!(
                *decision,
                Decision {
                    flags: compact.flags | ALLOCATED,
                    reserved: verified,
                    ..compact
                }
            );
            let input = BundleAcceptanceInputV1 {
                cycle_epoch: item.epoch,
                remaining_tokens: item.remaining,
                first_sample_row: item.first_sample,
                draft_item_index: item.draft_item,
                draft_tokens: item.k,
                stop_offset: item.stop_offset,
                stop_count: item.stop_count,
                reserved: 0,
            };
            let record = BundleAcceptanceResultV1::from_le_bytes(&decision.encode()).unwrap();
            match record.validate_allocation(input, true).unwrap() {
                DeviceAcceptanceV1::Prefix {
                    verified_drafts, ..
                } => assert_eq!(verified_drafts, verified),
                DeviceAcceptanceV1::EmptyDistribution {
                    verified_drafts, ..
                } => {
                    assert_eq!(verified_drafts, verified);
                    empty += 1;
                }
            }
            short += usize::from(verified < item.k);
        }
    }
    assert_eq!(ALLOCATED, BUNDLE_ACCEPTANCE_ALLOCATED_V1);
    assert!(short > 1000 && empty > 100, "{short} {empty}");
}

fn one(k: u32, verified: u32, tokens: &[u32], statuses: &[u32], status: u32) -> Decision {
    let samples = (0..=k as usize)
        .flat_map(|r| sample(tokens[r], statuses.get(r).copied().unwrap_or(0)))
        .collect::<Vec<_>>();
    oracle(
        Acceptance {
            vocab: VOCAB,
            sample_rows: k + 1,
            draft_items: u32::from(k > 0),
            stop_ids: 1,
        },
        Input {
            items: &[Item {
                epoch: 19,
                remaining: 9,
                first_sample: 0,
                draft_item: if k > 0 { 0 } else { NONE },
                k,
                stop_offset: 0,
                stop_count: 0,
                reserved: 0,
            }],
            proposals: &[[10, 11, 12, 13, 14]],
            samples: &samples,
            stops: &[90],
        },
        &[verified],
        status,
    )
    .unwrap()[0]
}

/// Stopping at K'_i is not a rejection; unverified rows are validated but
/// never walked; a failed allocation or K'_i above c_i fails the
/// descriptor, unmarked.
#[test]
fn unverified_rows_are_validated_but_never_walked() {
    let all = [10, 11, 12, 13, 14, 15];
    let allocated = |accepted, kept, flags, verified| Decision {
        epoch: 19,
        accepted,
        kept,
        flags: flags | ALLOCATED,
        failure: OK,
        failure_row: NONE,
        reserved: verified,
    };
    assert_eq!(one(5, 0, &all, &[], 0), allocated(0, 1, 0, 0));
    assert_eq!(one(5, 2, &all, &[], 0), allocated(2, 3, 0, 2));
    assert_eq!(one(5, 5, &all, &[], 0), allocated(5, 6, 0, 5));
    assert_eq!(
        one(5, 2, &[10, 99, 12, 13, 14, 15], &[], 0),
        allocated(1, 2, REJECTED, 2)
    );
    // EMPTY past the verified rows is never reached; at them it is
    // relative to the item and marked.
    assert_eq!(
        one(5, 2, &all, &[0, 0, 0, 2, 2, 2], 0),
        allocated(2, 3, 0, 2)
    );
    assert_eq!(
        one(5, 2, &all, &[0, 0, 2, 2, 2, 2], 0),
        Decision {
            flags: ALLOCATED,
            reserved: 2,
            ..Decision::failed(19, EMPTY, 2)
        }
    );
    // A corrupt unverified row refuses the view at its absolute row.
    assert_eq!(
        one(5, 2, &[10, 11, 12, 13, VOCAB, 15], &[], 0),
        Decision::failed(19, BAD_SAMPLE, 4)
    );
    assert_eq!(
        one(3, 4, &all, &[], 0),
        Decision::failed(19, BAD_DESCRIPTOR, NONE)
    );
    assert_eq!(
        one(3, 1, &all, &[], 1),
        Decision::failed(19, BAD_DESCRIPTOR, NONE)
    );
}
