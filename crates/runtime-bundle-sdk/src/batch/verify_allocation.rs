//! Batch wire V7: a verify step's allocation
//! (`runtime_bundle_abi::VERIFY_ALLOCATION_CAPABILITY_V1`).
//!
//! A V7 record is the magic, the u32 byte length of an enclosed V3 to V6
//! record and that record, then the allocation: the verify width (u32), the
//! floor (u32), the calibrated positions P (u32), P pairs of f32 bits (scale,
//! offset) for greedy items and then P for sampled ones, and a u32 count
//! equal to the enclosed item count with one u32 kind per item (0 greedy,
//! 1 sampled), little endian.
//!
//! The step's decode-phase items are its decode and verify items, in item
//! order; a verify item of `rows` rows reserves `rows - 1` drafts, a decode
//! item none. [`allocate_verify_drafts_v1`] is the rule a bundle verifies
//! by: every item verifies a prefix of its reserved drafts, and the verified
//! rows (one per item plus its verified drafts) sum to the width.

use super::{
    BATCH_REQUEST_MAGIC_V3, BATCH_REQUEST_MAGIC_V4, BATCH_REQUEST_MAGIC_V5, BATCH_REQUEST_MAGIC_V6,
    BatchCodecV1, BatchReadErrorV1, BatchReaderV1, BatchRequestV1, BatchWriteErrorV1, write_count,
};

/// The fixed prefix of one V7 batch request.
pub const BATCH_REQUEST_MAGIC_V7: &[u8; 12] = b"chopbatch-v7";

/// The most draft positions one calibration covers.
pub const VERIFY_ALLOCATION_POSITIONS_V1: usize = 16;

/// How the target samples an item, which selects its calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingKindV1 {
    Greedy,
    Sampled,
}

impl SamplingKindV1 {
    const fn index(self) -> usize {
        match self {
            Self::Greedy => 0,
            Self::Sampled => 1,
        }
    }
}

/// An allocation outside its rule's bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidVerifyAllocationV1;

/// Per kind and draft position, the scale and offset that turn a drafter's
/// raw confidence `s` into the probability `sigmoid(scale * s + offset)`
/// that the draft is accepted when every earlier one was. Floats are kept as
/// their bits, so calibrations compare exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyCalibrationV1 {
    /// Greedy, then sampled: `(scale, offset)` bits per position.
    kinds: [Vec<(u32, u32)>; 2],
}

impl VerifyCalibrationV1 {
    /// Both kinds name the same 1 to [`VERIFY_ALLOCATION_POSITIONS_V1`]
    /// positions with finite values.
    pub fn new(
        greedy: &[(f32, f32)],
        sampled: &[(f32, f32)],
    ) -> Result<Self, InvalidVerifyAllocationV1> {
        let bits = |pairs: &[(f32, f32)]| {
            pairs
                .iter()
                .map(|(scale, offset)| (scale.to_bits(), offset.to_bits()))
                .collect()
        };
        let calibration = Self {
            kinds: [bits(greedy), bits(sampled)],
        };
        calibration.validate()?;
        Ok(calibration)
    }

    fn validate(&self) -> Result<(), InvalidVerifyAllocationV1> {
        let positions = self.kinds[0].len();
        let finite = self.kinds.iter().flatten().all(|&(scale, offset)| {
            f32::from_bits(scale).is_finite() && f32::from_bits(offset).is_finite()
        });
        if !(1..=VERIFY_ALLOCATION_POSITIONS_V1).contains(&positions)
            || self.kinds[1].len() != positions
            || !finite
        {
            return Err(InvalidVerifyAllocationV1);
        }
        Ok(())
    }

    #[must_use]
    pub fn positions(&self) -> usize {
        self.kinds[0].len()
    }

    /// The `(scale, offset)` of a kind's draft position, counted from zero.
    #[must_use]
    pub fn pair(&self, kind: SamplingKindV1, position: usize) -> (f32, f32) {
        let (scale, offset) = self.kinds[kind.index()][position];
        (f32::from_bits(scale), f32::from_bits(offset))
    }

    /// The calibrated acceptance probability of a raw score at a draft
    /// position counted from zero, in f32 and within [0, 1]; a score that
    /// gives no number gives zero.
    #[must_use]
    pub fn probability(&self, kind: SamplingKindV1, position: usize, score: f32) -> f32 {
        let (scale, offset) = self.pair(kind, position);
        let p = 1.0 / (1.0 + (-(scale * score + offset)).exp());
        if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) }
    }
}

/// One verify step's allocation: the rows the target verifies, the drafts
/// every item verifies before any is ranked, the calibration, and each
/// enclosed item's kind.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifyAllocationV1 {
    width: u32,
    floor: u32,
    calibration: VerifyCalibrationV1,
    kinds: Vec<SamplingKindV1>,
}

impl VerifyAllocationV1 {
    #[must_use]
    pub const fn new(
        width: u32,
        floor: u32,
        calibration: VerifyCalibrationV1,
        kinds: Vec<SamplingKindV1>,
    ) -> Self {
        Self {
            width,
            floor,
            calibration,
            kinds,
        }
    }

    /// Rows the target verifies over the step's decode-phase items.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn floor(&self) -> u32 {
        self.floor
    }

    #[must_use]
    pub const fn calibration(&self) -> &VerifyCalibrationV1 {
        &self.calibration
    }

    /// One kind per enclosed item, in item order.
    #[must_use]
    pub fn kinds(&self) -> &[SamplingKindV1] {
        &self.kinds
    }

    /// Each decode-phase item's index among the enclosed items and its
    /// reserved drafts.
    #[must_use]
    pub fn reservations(request: &BatchRequestV1) -> Vec<(usize, u32)> {
        request
            .items()
            .iter()
            .enumerate()
            .filter(|(_, item)| item.is_decode() || item.is_verify())
            .map(|(index, item)| {
                let drafts = if item.is_verify() {
                    item.phase_tokens().saturating_sub(1)
                } else {
                    0
                };
                (index, drafts)
            })
            .collect()
    }

    /// One kind per item, a calibration covering every reservation, and a
    /// width from the decode-phase items through their reserved rows.
    pub fn validate(&self, request: &BatchRequestV1) -> Result<(), InvalidVerifyAllocationV1> {
        self.calibration.validate()?;
        let reservations = Self::reservations(request);
        let items = reservations.len() as u64;
        let rows: u64 = reservations
            .iter()
            .map(|&(_, drafts)| 1 + u64::from(drafts))
            .sum();
        let deepest = reservations
            .iter()
            .map(|&(_, drafts)| drafts as usize)
            .max()
            .unwrap_or(0);
        if self.kinds.len() != request.items().len()
            || items == 0
            || !(items..=rows).contains(&u64::from(self.width))
            || deepest > self.calibration.positions()
        {
            return Err(InvalidVerifyAllocationV1);
        }
        Ok(())
    }

    /// The floor every item verifies before any position is ranked: the
    /// largest up to the allocation's floor at which every item's floor
    /// (bounded by its reservation) and its own row fit the width.
    #[must_use]
    pub fn effective_floor(&self, reserved: &[u32]) -> u32 {
        (0..=self.floor)
            .rev()
            .find(|&floor| {
                reserved
                    .iter()
                    .map(|&drafts| 1 + u64::from(floor.min(drafts)))
                    .sum::<u64>()
                    <= u64::from(self.width)
            })
            .unwrap_or(0)
    }
}

/// The drafts each decode-phase item verifies: `reserved[i]` drafts with
/// raw scores `scores[i]` (at least `reserved[i]` of them) and kind
/// `kinds[i]`. Every item first verifies the effective floor (within its
/// reservation); the remaining rows of the width go to the positions with
/// the largest prefix survival, the product of the calibrated
/// probabilities through the position, ties to the earlier item and then
/// the earlier position. Survival never grows along an item's positions, so
/// each item verifies a prefix. When the width exceeds the reserved rows,
/// every reservation is verified. Consumes no randomness.
#[must_use]
pub fn allocate_verify_drafts_v1(
    allocation: &VerifyAllocationV1,
    reserved: &[u32],
    scores: &[&[f32]],
    kinds: &[SamplingKindV1],
) -> Vec<u32> {
    let floor = allocation.effective_floor(reserved);
    let mut verified: Vec<u32> = reserved.iter().map(|&drafts| floor.min(drafts)).collect();
    let used: u64 = verified.iter().map(|&drafts| 1 + u64::from(drafts)).sum();
    let mut left = u64::from(allocation.width()).saturating_sub(used);
    let mut candidates = Vec::new();
    for (item, (&drafts, (scores, &kind))) in
        reserved.iter().zip(scores.iter().zip(kinds)).enumerate()
    {
        let mut survival = 1.0_f32;
        for position in 0..drafts as usize {
            survival *= allocation
                .calibration()
                .probability(kind, position, scores[position]);
            if position as u32 >= floor {
                candidates.push((survival, item, position));
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    for (_, item, _) in candidates {
        if left == 0 {
            break;
        }
        verified[item] += 1;
        left -= 1;
    }
    verified
}

pub(super) fn encode_v7(request: &BatchRequestV1) -> Result<Vec<u8>, BatchWriteErrorV1> {
    let allocation = request
        .verify_allocation
        .as_ref()
        .ok_or(BatchWriteErrorV1::InvalidVerifyAllocation)?;
    allocation
        .validate(request)
        .map_err(|_| BatchWriteErrorV1::InvalidVerifyAllocation)?;
    let mut inner = request.clone();
    inner.verify_allocation = None;
    let payload = BatchCodecV1::encode(&inner)?;
    let mut bytes = BATCH_REQUEST_MAGIC_V7.to_vec();
    write_count(&mut bytes, "enclosed batch bytes", payload.len())?;
    bytes.extend_from_slice(&payload);
    bytes.extend_from_slice(&allocation.width.to_le_bytes());
    bytes.extend_from_slice(&allocation.floor.to_le_bytes());
    write_count(
        &mut bytes,
        "calibrated positions",
        allocation.calibration.positions(),
    )?;
    for kind in &allocation.calibration.kinds {
        for &(scale, offset) in kind {
            bytes.extend_from_slice(&scale.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
    }
    write_count(&mut bytes, "allocation kinds", allocation.kinds.len())?;
    for kind in &allocation.kinds {
        bytes.extend_from_slice(&(kind.index() as u32).to_le_bytes());
    }
    Ok(bytes)
}

pub(super) fn decode_v7(bytes: &[u8]) -> Result<BatchRequestV1, BatchReadErrorV1> {
    let mut reader = BatchReaderV1::new("verify allocation batch", bytes);
    reader.expect_magic(BATCH_REQUEST_MAGIC_V7)?;
    let length = reader.enclosed_length()?;
    let enclosed = reader.take(length)?;
    if ![
        BATCH_REQUEST_MAGIC_V3,
        BATCH_REQUEST_MAGIC_V4,
        BATCH_REQUEST_MAGIC_V5,
        BATCH_REQUEST_MAGIC_V6,
    ]
    .iter()
    .any(|magic| enclosed.starts_with(*magic))
    {
        return Err(BatchReadErrorV1::InvalidVerifyAllocation);
    }
    let mut request = BatchCodecV1::decode(enclosed)?;
    let width = reader.u32()?;
    let floor = reader.u32()?;
    let positions = reader.count("calibrated positions")?;
    if !(1..=VERIFY_ALLOCATION_POSITIONS_V1).contains(&positions) {
        return Err(BatchReadErrorV1::InvalidVerifyAllocation);
    }
    let mut kinds = [Vec::new(), Vec::new()];
    for kind in &mut kinds {
        for _ in 0..positions {
            kind.push((reader.u32()?, reader.u32()?));
        }
    }
    let calibration = VerifyCalibrationV1 { kinds };
    let count = reader.exact_count("allocation kinds", request.items().len() as u64)?;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        items.push(match reader.u32()? {
            0 => SamplingKindV1::Greedy,
            1 => SamplingKindV1::Sampled,
            _ => return Err(BatchReadErrorV1::InvalidVerifyAllocation),
        });
    }
    reader.finish()?;
    let allocation = VerifyAllocationV1::new(width, floor, calibration, items);
    allocation
        .validate(&request)
        .map_err(|_| BatchReadErrorV1::InvalidVerifyAllocation)?;
    request.verify_allocation = Some(allocation);
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::super::{BatchAppendV1, BatchItemV1, BatchWireVersionV1};
    use super::*;

    fn calibration() -> VerifyCalibrationV1 {
        VerifyCalibrationV1::new(&[(1.0, 0.0); 5], &[(0.8, -0.2); 5]).unwrap()
    }

    fn request(rows: &[u32]) -> BatchRequestV1 {
        let items = rows
            .iter()
            .enumerate()
            .map(|(index, &rows)| {
                if rows == 1 {
                    BatchItemV1::new(index as u64 + 1, true, 1, true, vec![7])
                } else {
                    BatchItemV1::verify(index as u64 + 1, rows, true, vec![7])
                }
            })
            .collect::<Vec<_>>();
        let appends = rows
            .iter()
            .map(|&rows| {
                BatchAppendV1::new(9, u64::from(rows), 9 + u64::from(rows), vec![1, 2], None)
            })
            .collect();
        BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
    }

    fn allocation(width: u32, floor: u32, items: usize) -> VerifyAllocationV1 {
        VerifyAllocationV1::new(
            width,
            floor,
            calibration(),
            vec![SamplingKindV1::Sampled; items],
        )
    }

    #[test]
    fn records_round_trip_over_every_enclosed_revision() {
        let plain = request(&[6, 6, 1, 4]);
        let allocated = plain.clone().with_verify_allocation(allocation(12, 1, 4));
        let bytes = allocated.encode().unwrap();
        assert!(bytes.starts_with(BATCH_REQUEST_MAGIC_V7));
        assert_eq!(BatchRequestV1::decode(&bytes).unwrap(), allocated);
        let slots = plain
            .clone()
            .with_state_slots(vec![1, 2, 3, 4])
            .with_verify_allocation(allocation(12, 1, 4));
        assert_eq!(
            BatchRequestV1::decode(&slots.encode().unwrap()).unwrap(),
            slots
        );
        let sampled = plain
            .with_draft_sampling(vec![super::super::DraftSamplingV1::GREEDY; 4])
            .with_verify_allocation(allocation(17, 0, 4));
        assert_eq!(
            BatchRequestV1::decode(&sampled.encode().unwrap()).unwrap(),
            sampled
        );
    }

    #[test]
    fn records_refuse_widths_kinds_and_calibrations_outside_the_rule() {
        let plain = request(&[6, 6, 1, 4]);
        // Four decode-phase items reserve 17 rows: widths 4 to 17 fit.
        for (width, kinds) in [(3, 4), (18, 4), (12, 3)] {
            assert_eq!(
                plain
                    .clone()
                    .with_verify_allocation(allocation(width, 1, kinds))
                    .encode(),
                Err(BatchWriteErrorV1::InvalidVerifyAllocation)
            );
        }
        let short = VerifyCalibrationV1::new(&[(1.0, 0.0); 4], &[(1.0, 0.0); 4]).unwrap();
        let shallow = VerifyAllocationV1::new(12, 0, short, vec![SamplingKindV1::Greedy; 4]);
        assert!(shallow.validate(&plain).is_err());
        for (greedy, sampled) in [
            (vec![(1.0, 0.0); 5], vec![(1.0, 0.0); 4]),
            (vec![], vec![]),
            (vec![(f32::NAN, 0.0); 5], vec![(1.0, 0.0); 5]),
            (vec![(1.0, 0.0); 17], vec![(1.0, 0.0); 17]),
        ] {
            assert!(VerifyCalibrationV1::new(&greedy, &sampled).is_err());
        }
        let mut bytes = plain
            .with_verify_allocation(allocation(12, 1, 4))
            .encode()
            .unwrap();
        let last = bytes.len() - 4;
        bytes[last..].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            BatchRequestV1::decode(&bytes),
            Err(BatchReadErrorV1::InvalidVerifyAllocation)
        );
    }

    /// Prefix survival ranks positions across items, the floor comes
    /// first, ties go to the earlier item and position, and the verified
    /// rows fill the width exactly.
    #[test]
    fn allocation_ranks_prefix_survival_after_the_floor() {
        let identity = VerifyCalibrationV1::new(&[(1.0, 0.0); 5], &[(1.0, 0.0); 5]).unwrap();
        let rule = |width, floor| {
            VerifyAllocationV1::new(
                width,
                floor,
                identity.clone(),
                vec![SamplingKindV1::Greedy; 3],
            )
        };
        let logit = |p: f32| (p / (1.0 - p)).ln();
        let confident: Vec<f32> = [0.95, 0.9, 0.9, 0.9, 0.9].map(logit).to_vec();
        let doubtful: Vec<f32> = [0.5, 0.9, 0.9, 0.9, 0.9].map(logit).to_vec();
        let middling: Vec<f32> = [0.8, 0.8, 0.5, 0.5, 0.5].map(logit).to_vec();
        let scores: [&[f32]; 3] = [&confident, &doubtful, &middling];
        let kinds = [SamplingKindV1::Greedy; 3];
        let reserved = [5, 5, 5];
        // 9 rows: three items and six drafts. Survival: confident .95 .855
        // .77 .69 .62, middling .8 .64 .32, doubtful .5 .45.
        assert_eq!(
            allocate_verify_drafts_v1(&rule(9, 0), &reserved, &scores, &kinds),
            [4, 0, 2]
        );
        // A floor of one gives every item its first draft first.
        assert_eq!(
            allocate_verify_drafts_v1(&rule(9, 1), &reserved, &scores, &kinds),
            [4, 1, 1]
        );
        // A floor that does not fit shrinks to the largest that does.
        assert_eq!(rule(7, 3).effective_floor(&reserved), 1);
        assert_eq!(rule(4, 3).effective_floor(&reserved), 0);
        // Reservations bound every item; a width past them verifies all.
        assert_eq!(
            allocate_verify_drafts_v1(&rule(20, 2), &[1, 5, 0], &scores, &kinds),
            [1, 5, 0]
        );
        // Equal survival goes to the earlier item, then the earlier position.
        let even: Vec<f32> = vec![0.0; 5];
        let tied: [&[f32]; 3] = [&even, &even, &even];
        assert_eq!(
            allocate_verify_drafts_v1(&rule(5, 0), &reserved, &tied, &kinds),
            [1, 1, 0]
        );
        let certain: Vec<f32> = vec![f32::INFINITY; 5];
        let sure: [&[f32]; 3] = [&certain, &certain, &certain];
        assert_eq!(
            allocate_verify_drafts_v1(&rule(6, 0), &reserved, &sure, &kinds),
            [3, 0, 0]
        );
        // A score without a number counts as rejection-certain.
        let lost: Vec<f32> = vec![f32::NAN; 5];
        let unknown: [&[f32]; 3] = [&lost, &confident, &confident];
        assert_eq!(
            allocate_verify_drafts_v1(&rule(7, 0), &reserved, &unknown, &kinds),
            [0, 2, 2]
        );
        for width in 3..=18 {
            let verified = allocate_verify_drafts_v1(&rule(width, 1), &reserved, &scores, &kinds);
            let rows: u32 = verified.iter().map(|k| k + 1).sum();
            assert_eq!(rows, width.min(18), "{width}");
            assert!(verified.iter().zip(reserved).all(|(k, c)| *k <= c));
        }
    }

    /// The kind selects the calibration.
    #[test]
    fn each_kind_reads_its_own_calibration() {
        let calibration = VerifyCalibrationV1::new(&[(1.0, 2.0)], &[(1.0, -2.0)]).unwrap();
        assert!(calibration.probability(SamplingKindV1::Greedy, 0, 0.0) > 0.88);
        assert!(calibration.probability(SamplingKindV1::Sampled, 0, 0.0) < 0.12);
        assert_eq!(calibration.pair(SamplingKindV1::Sampled, 0), (1.0, -2.0));
    }
}
