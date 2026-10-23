//! Batch wire V6: each item's sampling inputs for drafting
//! (`runtime_bundle_abi::DRAFT_SAMPLING_CAPABILITY_V1`).
//!
//! A V6 record is the magic, the u32 byte length of an enclosed V3, V4 or
//! V5 record and that record, then a u32 count equal to the enclosed item
//! count and one [`DRAFT_SAMPLING_BYTES_V1`]-byte record per item, in item
//! order: seed (u64), generated (u64), temperature, top-p and min-p (f64
//! bits), top-k (u32) and flags (u32), little endian. A drafter that
//! proposes, at draft position `j`, the argmax of its logits divided by the
//! temperature plus the host sampler's Gumbel noise at generated index
//! `generated + j` (`kernel_core::draft_noise`) proposes the target's own
//! sample whenever its logits rank the leading candidates as the target
//! does. Acceptance stays the host's: a draft is kept only when it equals
//! the target's sample at its row.

use super::{
    BATCH_REQUEST_MAGIC_V3, BATCH_REQUEST_MAGIC_V4, BATCH_REQUEST_MAGIC_V5, BatchCodecV1,
    BatchReadErrorV1, BatchReaderV1, BatchRequestV1, BatchWriteErrorV1, write_count,
};

/// The fixed prefix of one V6 batch request.
pub const BATCH_REQUEST_MAGIC_V6: &[u8; 12] = b"chopbatch-v6";

/// Bytes of one item's draft sampling record.
pub const DRAFT_SAMPLING_BYTES_V1: usize = 48;

/// The target applies processing this record does not carry: a logit
/// bias, a repetition, frequency or presence penalty, an EOS mask before
/// the minimum length, or a constraint mask. A drafter that applies only
/// the temperature may then propose tokens the target never samples.
pub const DRAFT_SAMPLING_PROCESSED_V1: u32 = 1;

/// One item's sampling inputs for drafting. Floats are kept as their bits,
/// so records compare exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DraftSamplingV1 {
    seed: u64,
    generated: u64,
    temperature: u64,
    top_p: u64,
    min_p: u64,
    top_k: u32,
    flags: u32,
}

/// A draft sampling record outside the host sampler's parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidDraftSamplingV1;

impl DraftSamplingV1 {
    /// A greedy item, or an item without sampled rows.
    pub const GREEDY: DraftSamplingV1 = DraftSamplingV1 {
        seed: 0,
        generated: 0,
        temperature: 0,
        top_p: 0x3ff0_0000_0000_0000,
        min_p: 0,
        top_k: 0,
        flags: 0,
    };

    /// The record of a request with `seed` that sampled `generated` tokens
    /// before the item's first sampled row. A zero temperature samples
    /// greedily; zero top-k, unit top-p and zero min-p disable their
    /// filters.
    pub fn new(
        seed: u64,
        generated: u64,
        temperature: f64,
        top_p: f64,
        min_p: f64,
        top_k: u32,
        flags: u32,
    ) -> Result<Self, InvalidDraftSamplingV1> {
        let record = Self {
            seed,
            generated,
            temperature: temperature.to_bits(),
            top_p: top_p.to_bits(),
            min_p: min_p.to_bits(),
            top_k,
            flags,
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> Result<(), InvalidDraftSamplingV1> {
        let (temperature, top_p, min_p) = (self.temperature(), self.top_p(), self.min_p());
        // Every comparison with NaN is false, so NaN fails each check.
        let valid = temperature.is_finite()
            && temperature >= 0.0
            && top_p > 0.0
            && top_p <= 1.0
            && (0.0..=1.0).contains(&min_p)
            && self.flags & !DRAFT_SAMPLING_PROCESSED_V1 == 0;
        if !valid {
            return Err(InvalidDraftSamplingV1);
        }
        Ok(())
    }

    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The Philox counter of the item's first sampled row: the target
    /// samples draft position `j` at `generated + j`.
    #[must_use]
    pub const fn generated(&self) -> u64 {
        self.generated
    }

    #[must_use]
    pub fn temperature(&self) -> f64 {
        f64::from_bits(self.temperature)
    }

    #[must_use]
    pub fn top_p(&self) -> f64 {
        f64::from_bits(self.top_p)
    }

    #[must_use]
    pub fn min_p(&self) -> f64 {
        f64::from_bits(self.min_p)
    }

    #[must_use]
    pub const fn top_k(&self) -> u32 {
        self.top_k
    }

    #[must_use]
    pub const fn flags(&self) -> u32 {
        self.flags
    }

    /// Whether the target samples the item at a positive temperature.
    #[must_use]
    pub fn sampled(&self) -> bool {
        self.temperature() > 0.0
    }

    #[must_use]
    pub fn encode(&self) -> [u8; DRAFT_SAMPLING_BYTES_V1] {
        let mut bytes = [0; DRAFT_SAMPLING_BYTES_V1];
        bytes[..8].copy_from_slice(&self.seed.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.generated.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.temperature.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.top_p.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.min_p.to_le_bytes());
        bytes[40..44].copy_from_slice(&self.top_k.to_le_bytes());
        bytes[44..].copy_from_slice(&self.flags.to_le_bytes());
        bytes
    }

    fn read(reader: &mut BatchReaderV1<'_>) -> Result<Self, BatchReadErrorV1> {
        let record = Self {
            seed: reader.u64()?,
            generated: reader.u64()?,
            temperature: reader.u64()?,
            top_p: reader.u64()?,
            min_p: reader.u64()?,
            top_k: reader.u32()?,
            flags: reader.u32()?,
        };
        record
            .validate()
            .map_err(|_| BatchReadErrorV1::InvalidDraftSampling)?;
        Ok(record)
    }
}

pub(super) fn encode_v6(request: &BatchRequestV1) -> Result<Vec<u8>, BatchWriteErrorV1> {
    let records = request
        .draft_sampling
        .as_ref()
        .ok_or(BatchWriteErrorV1::InvalidDraftSampling)?;
    if records.len() != request.items.len() || request.version != super::BatchWireVersionV1::V3 {
        return Err(BatchWriteErrorV1::InvalidDraftSampling);
    }
    let mut inner = request.clone();
    inner.draft_sampling = None;
    let payload = BatchCodecV1::encode(&inner)?;
    let mut bytes = BATCH_REQUEST_MAGIC_V6.to_vec();
    write_count(&mut bytes, "enclosed batch bytes", payload.len())?;
    bytes.extend_from_slice(&payload);
    write_count(&mut bytes, "draft sampling items", records.len())?;
    for record in records {
        bytes.extend_from_slice(&record.encode());
    }
    Ok(bytes)
}

pub(super) fn decode_v6(bytes: &[u8]) -> Result<BatchRequestV1, BatchReadErrorV1> {
    let mut reader = BatchReaderV1::new("draft sampling batch", bytes);
    reader.expect_magic(BATCH_REQUEST_MAGIC_V6)?;
    let length = reader.enclosed_length()?;
    let enclosed = reader.take(length)?;
    if ![
        BATCH_REQUEST_MAGIC_V3,
        BATCH_REQUEST_MAGIC_V4,
        BATCH_REQUEST_MAGIC_V5,
    ]
    .iter()
    .any(|magic| enclosed.starts_with(*magic))
    {
        return Err(BatchReadErrorV1::InvalidDraftSampling);
    }
    let mut request = BatchCodecV1::decode(enclosed)?;
    let count = reader.exact_count("draft sampling items", request.items.len() as u64)?;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        records.push(DraftSamplingV1::read(&mut reader)?);
    }
    reader.finish()?;
    request.draft_sampling = Some(records);
    Ok(request)
}
