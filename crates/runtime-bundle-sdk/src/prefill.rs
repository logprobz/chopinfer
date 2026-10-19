//! A negotiated prefill envelope around an unchanged batch record.

use crate::{BatchCodecV1, BatchReadErrorV1, BatchReaderV1, BatchRequestV1};
use std::collections::BTreeSet;

pub const MAGIC: &[u8; 16] = b"chop-prefill-v1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Boundary {
    pub request_id: u64,
    pub prompt_tokens: u64,
    pub chunk_end: u64,
    pub chunk_tokens: u32,
}

#[derive(Debug)]
pub enum Error {
    Batch(BatchReadErrorV1),
    Contract(&'static str),
}

impl From<BatchReadErrorV1> for Error {
    fn from(error: BatchReadErrorV1) -> Self {
        Self::Batch(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Batch(e) => e.fmt(f),
            Self::Contract(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for Error {}

pub struct Envelope<'a> {
    pub original: &'a [u8],
    pub batch: BatchRequestV1,
    pub boundaries: Vec<Boundary>,
}

fn validate(batch: &BatchRequestV1, boundaries: &[Boundary]) -> Result<(), Error> {
    if boundaries.len() > batch.items().len() {
        return Err(Error::Contract("too many prefill boundaries"));
    }
    let mut seen = BTreeSet::new();
    for boundary in boundaries {
        let Some(index) = batch
            .items()
            .iter()
            .position(|item| item.request_id() == boundary.request_id)
        else {
            return Err(Error::Contract("prefill boundary has no batch item"));
        };
        let item = &batch.items()[index];
        let append = batch
            .appends()
            .get(index)
            .ok_or(Error::Contract("prefill boundary has no append"))?;
        let chunk = u64::from(boundary.chunk_tokens);
        if chunk == 0
            || !seen.insert(boundary.request_id)
            || item.is_decode()
            || item.is_verify()
            || item.prompt_rows()
            || item.anchor()
            || boundary.prompt_tokens == 0
            || batch
                .media()
                .injects
                .iter()
                .any(|inject| inject.item_index as usize == index)
        {
            return Err(Error::Contract("invalid prefill boundary item"));
        }
        let next = append
            .previous_tokens()
            .checked_div(chunk)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(chunk))
            .ok_or(Error::Contract("prefill boundary overflows"))?;
        let end = append
            .previous_tokens()
            .checked_add(u64::from(item.phase_tokens()))
            .ok_or(Error::Contract("prefill append overflows"))?;
        if boundary.chunk_end != end
            || end != next.min(boundary.prompt_tokens)
            || end <= append.previous_tokens()
            || item.expects_logits() != (end == boundary.prompt_tokens)
        {
            return Err(Error::Contract(
                "prefill append does not end at its guaranteed boundary",
            ));
        }
    }
    Ok(())
}

pub fn encode(original: &[u8], boundaries: &[Boundary]) -> Result<Vec<u8>, Error> {
    let batch = BatchCodecV1::decode(original)?;
    validate(&batch, boundaries)?;
    let size =
        u32::try_from(original.len()).map_err(|_| Error::Contract("prefill batch is too large"))?;
    let count = u32::try_from(boundaries.len())
        .map_err(|_| Error::Contract("too many prefill boundaries"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve(
            original
                .len()
                .checked_add(24 + boundaries.len() * 32)
                .ok_or(Error::Contract("prefill envelope overflows"))?,
        )
        .map_err(|_| Error::Contract("prefill envelope allocation failed"))?;
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(original);
    for boundary in boundaries {
        for word in [
            boundary.request_id,
            boundary.prompt_tokens,
            boundary.chunk_end,
        ] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes.extend_from_slice(&boundary.chunk_tokens.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
    }
    Ok(bytes)
}

pub fn decode(bytes: &[u8]) -> Result<Envelope<'_>, Error> {
    let mut reader = BatchReaderV1::new("prefill envelope", bytes);
    reader.expect_magic(MAGIC)?;
    let size = reader.u32()? as usize;
    let count = reader.u32()? as usize;
    let original = reader.take(size)?;
    let batch = BatchCodecV1::decode(original)?;
    if count > batch.items().len() || count > reader.remaining() / 32 {
        return Err(Error::Contract("invalid prefill boundary count"));
    }
    let mut boundaries = Vec::with_capacity(count);
    for _ in 0..count {
        let boundary = Boundary {
            request_id: reader.u64()?,
            prompt_tokens: reader.u64()?,
            chunk_end: reader.u64()?,
            chunk_tokens: reader.u32()?,
        };
        if reader.u32()? != 0 {
            return Err(Error::Contract("nonzero prefill boundary reserved field"));
        }
        boundaries.push(boundary);
    }
    reader.finish()?;
    validate(&batch, &boundaries)?;
    Ok(Envelope {
        original,
        batch,
        boundaries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BatchAppendV1, BatchItemV1, BatchWireVersionV1};

    fn batch(version: BatchWireVersionV1) -> Vec<u8> {
        BatchCodecV1::encode(&BatchRequestV1::new(
            version,
            vec![BatchItemV1::new(7, false, 4, true, vec![1, 2, 3, 4])],
            vec![BatchAppendV1::new(0, 4, 4, vec![1], None)],
        ))
        .unwrap()
    }

    fn boundary() -> Boundary {
        Boundary {
            request_id: 7,
            prompt_tokens: 4,
            chunk_end: 4,
            chunk_tokens: 4,
        }
    }

    #[test]
    fn envelope_preserves_legacy_records_byte_for_byte() {
        for version in [
            BatchWireVersionV1::V1,
            BatchWireVersionV1::V2,
            BatchWireVersionV1::V3,
        ] {
            let original = batch(version);
            let bytes = encode(&original, &[boundary()]).unwrap();
            let read = decode(&bytes).unwrap();
            assert_eq!(read.original, original);
            assert_eq!(read.boundaries, [boundary()]);
            assert_eq!(read.batch, BatchCodecV1::decode(&original).unwrap());
            assert!(BatchCodecV1::decode(&bytes).is_err());
            assert!(decode(&original).is_err());
            assert!(
                decode(&encode(&original, &[]).unwrap())
                    .unwrap()
                    .boundaries
                    .is_empty()
            );
        }
    }

    #[test]
    fn truncated_overlong_duplicate_and_reserved_records_are_rejected() {
        let original = batch(BatchWireVersionV1::V3);
        let bytes = encode(&original, &[boundary()]).unwrap();
        for end in 0..bytes.len() {
            assert!(decode(&bytes[..end]).is_err(), "{end}");
        }
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(decode(&bad).is_err());
        let mut bad = bytes.clone();
        *bad.last_mut().unwrap() = 1;
        assert!(decode(&bad).is_err());
        let mut bad = bytes;
        bad[20..24].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&bad).is_err());
        assert!(encode(&original, &[boundary(), boundary()]).is_err());
    }

    #[test]
    fn contract_checks_identity_grid_and_logits_boundary() {
        let original = batch(BatchWireVersionV1::V3);
        for wrong in [
            Boundary {
                request_id: 8,
                ..boundary()
            },
            Boundary {
                chunk_tokens: 0,
                ..boundary()
            },
            Boundary {
                chunk_tokens: 3,
                ..boundary()
            },
            Boundary {
                chunk_end: 3,
                ..boundary()
            },
            Boundary {
                prompt_tokens: 8,
                ..boundary()
            },
        ] {
            assert!(encode(&original, &[wrong]).is_err());
        }
    }
}
