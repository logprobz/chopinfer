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

fn validate(batch: &BatchRequestV1, boundaries: &[Boundary], cuts: bool) -> Result<(), Error> {
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
            || end > next.min(boundary.prompt_tokens)
            || (!cuts && end != next.min(boundary.prompt_tokens))
            || end <= append.previous_tokens()
            || item.expects_logits() != (end == boundary.prompt_tokens)
        {
            return Err(Error::Contract(
                "prefill append does not end at its guaranteed boundary",
            ));
        }
        // A cuts envelope may carry a prompt-ending item whose anchor state
        // the bundle keeps inside the item. Grid-only envelopes refuse it.
        if item.anchor() && !(cuts && end == boundary.prompt_tokens) {
            return Err(Error::Contract("invalid prefill boundary item"));
        }
    }
    Ok(())
}

pub fn encode(original: &[u8], boundaries: &[Boundary]) -> Result<Vec<u8>, Error> {
    encode_policy(original, boundaries, false)
}

/// Encode endpoints for a separately negotiated complete-state cuts API.
/// The scheduler and bundle enforce the bundle's declared prefill granule.
/// Prompt-ending items may keep an anchor inside their rows, including at
/// nonzero starts. The scheduler enforces the instance's anchor allowance.
pub fn encode_with_cuts(original: &[u8], boundaries: &[Boundary]) -> Result<Vec<u8>, Error> {
    encode_policy(original, boundaries, true)
}

fn encode_policy(original: &[u8], boundaries: &[Boundary], cuts: bool) -> Result<Vec<u8>, Error> {
    let batch = BatchCodecV1::decode(original)?;
    validate(&batch, boundaries, cuts)?;
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
    decode_policy(bytes, false)
}

/// Decode only after the bundle opted into complete-state scheduler cuts.
pub fn decode_with_cuts(bytes: &[u8]) -> Result<Envelope<'_>, Error> {
    decode_policy(bytes, true)
}

fn decode_policy(bytes: &[u8], cuts: bool) -> Result<Envelope<'_>, Error> {
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
    validate(&batch, &boundaries, cuts)?;
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

    #[test]
    fn scheduler_cuts_require_separate_admission_and_preserve_grid_bytes() {
        for version in [
            BatchWireVersionV1::V1,
            BatchWireVersionV1::V2,
            BatchWireVersionV1::V3,
        ] {
            let original = batch(version);
            assert_eq!(
                encode(&original, &[boundary()]).unwrap(),
                encode_with_cuts(&original, &[boundary()]).unwrap()
            );
            let original = BatchRequestV1::new(
                version,
                vec![BatchItemV1::new(7, false, 64, false, vec![1; 64])],
                vec![BatchAppendV1::new(128, 64, 192, vec![1, 2, 3], None)],
            )
            .encode()
            .unwrap();
            let cut = Boundary {
                request_id: 7,
                prompt_tokens: 529,
                chunk_end: 192,
                chunk_tokens: 256,
            };
            assert!(encode(&original, &[cut]).is_err());
            let bytes = encode_with_cuts(&original, &[cut]).unwrap();
            assert!(decode(&bytes).is_err());
            let read = decode_with_cuts(&bytes).unwrap();
            assert_eq!(read.original, original);
            assert_eq!(read.boundaries, [cut]);
            for invalid in [
                Boundary {
                    chunk_end: 191,
                    ..cut
                },
                Boundary {
                    chunk_tokens: 80,
                    ..cut
                },
                Boundary {
                    prompt_tokens: 191,
                    ..cut
                },
                Boundary {
                    prompt_tokens: 192,
                    ..cut
                },
            ] {
                assert!(encode_with_cuts(&original, &[invalid]).is_err());
            }
        }
    }

    /// One prefill item from `previous` taking `rows` rows.
    fn item(id: u64, previous: u64, rows: u32, logits: bool) -> (BatchItemV1, BatchAppendV1) {
        (
            BatchItemV1::new(id, false, rows, logits, vec![1; rows as usize]),
            BatchAppendV1::new(
                previous,
                u64::from(rows),
                previous + u64::from(rows),
                vec![1],
                None,
            ),
        )
    }

    fn cut_batch(items: &[(BatchItemV1, BatchAppendV1)]) -> Vec<u8> {
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            items.iter().map(|(item, _)| item.clone()).collect(),
            items.iter().map(|(_, append)| append.clone()).collect(),
        )
        .encode()
        .unwrap()
    }

    fn cut(id: u64, prompt_tokens: u64, chunk_end: u64) -> Boundary {
        Boundary {
            request_id: id,
            prompt_tokens,
            chunk_end,
            chunk_tokens: 8192,
        }
    }

    /// Timing cuts end a grid chunk early: small, repeated, off-grid and
    /// packed earlier ends are admitted only through the cuts entry, and
    /// an end past the next grid boundary or the prompt, a non-final item
    /// with logits, a final item without them, an empty item and a
    /// repeated request are refused.
    #[test]
    fn timing_cut_envelopes_admit_only_complete_earlier_ends() {
        // A 64-row non-final cut, repeated 512-row cuts, an off-grid
        // restored start and the step that reaches the grid boundary.
        for (previous, rows, prompt) in [
            (0, 64, 9000),
            (512, 512, 9000),
            (7168, 512, 9000),
            (529, 495, 2029),
            (7680, 512, 9000),
            (8192, 512, 9000),
        ] {
            let original = cut_batch(&[item(3, previous, rows, false)]);
            let boundary = cut(3, prompt, previous + u64::from(rows));
            let bytes = encode_with_cuts(&original, &[boundary]).unwrap();
            assert_eq!(decode_with_cuts(&bytes).unwrap().boundaries, [boundary]);
            let grid = previous / 8192 * 8192 + 8192;
            assert_eq!(
                encode(&original, &[boundary]).is_ok(),
                boundary.chunk_end == grid.min(prompt)
            );
            assert_eq!(
                decode(&bytes).is_ok(),
                boundary.chunk_end == grid.min(prompt)
            );
        }
        // Two requests cut at different ends in one packed step, beside a
        // prompt end with logits.
        let packed = cut_batch(&[
            item(3, 1024, 512, false),
            item(4, 529, 31, false),
            item(5, 7936, 64, true),
        ]);
        let boundaries = [cut(3, 9000, 1536), cut(4, 2029, 560), cut(5, 8000, 8000)];
        let bytes = encode_with_cuts(&packed, &boundaries).unwrap();
        assert_eq!(decode_with_cuts(&bytes).unwrap().boundaries, boundaries);
        assert!(decode(&bytes).is_err());
        for (items, boundary) in [
            // Past the next grid boundary.
            (vec![item(3, 7936, 512, false)], cut(3, 9000, 8448)),
            // Past the prompt end.
            (vec![item(3, 1024, 512, true)], cut(3, 1300, 1536)),
            // A non-final cut with logits, and a final chunk without them.
            (vec![item(3, 0, 512, true)], cut(3, 9000, 512)),
            (vec![item(3, 1024, 476, false)], cut(3, 1500, 1500)),
            // An end the append does not reach.
            (vec![item(3, 0, 512, false)], cut(3, 9000, 576)),
        ] {
            let original = cut_batch(&items);
            assert!(
                encode_with_cuts(&original, &[boundary]).is_err(),
                "{boundary:?}"
            );
        }
        // Each request has one boundary per step.
        let original = cut_batch(&[item(3, 0, 512, false)]);
        assert!(encode_with_cuts(&original, &[cut(3, 9000, 512), cut(3, 9000, 512)]).is_err());
    }

    /// Only a cuts envelope carries an anchor item, and only one ending its
    /// prompt: the bundle keeps its state at the granule inside it.
    #[test]
    fn anchor_items_need_cuts_and_their_prompt_end() {
        let anchored = |previous: u64, rows: u32| {
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![
                    BatchItemV1::new(7, false, rows, true, vec![1; rows as usize])
                        .with_anchor(true),
                ],
                vec![BatchAppendV1::new(
                    previous,
                    u64::from(rows),
                    previous + u64::from(rows),
                    vec![1; (previous + u64::from(rows)).div_ceil(64) as usize],
                    None,
                )],
            )
            .encode()
            .unwrap()
        };
        for (previous, rows) in [(0, 789), (529, 260), (100001, 49), (100032, 65)] {
            let prompt_tokens = previous + u64::from(rows);
            let anchor = prompt_tokens / 64 * 64;
            assert!(previous < anchor && anchor < prompt_tokens);
            let end = Boundary {
                request_id: 7,
                prompt_tokens,
                chunk_end: prompt_tokens,
                chunk_tokens: 8192,
            };
            let original = anchored(previous, rows);
            assert!(encode(&original, &[end]).is_err());
            let bytes = encode_with_cuts(&original, &[end]).unwrap();
            assert!(decode(&bytes).is_err());
            let read = decode_with_cuts(&bytes).unwrap();
            assert_eq!(read.original, original);
            assert!(read.batch.items()[0].anchor());
            assert_eq!(read.boundaries, [end]);
            let short = Boundary {
                prompt_tokens: prompt_tokens + 64,
                ..end
            };
            assert!(encode_with_cuts(&original, &[short]).is_err());
            let mut bad = bytes;
            let prompt_offset = 24 + original.len() + 8;
            bad[prompt_offset..prompt_offset + 8]
                .copy_from_slice(&short.prompt_tokens.to_le_bytes());
            assert!(decode_with_cuts(&bad).is_err());
        }
        // The batch codec also refuses an anchor without prompt-end logits.
        assert!(
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![BatchItemV1::new(7, false, 768, false, vec![1; 768]).with_anchor(true)],
                vec![BatchAppendV1::new(0, 768, 768, vec![1; 12], None)],
            )
            .encode()
            .is_err()
        );
    }
}
