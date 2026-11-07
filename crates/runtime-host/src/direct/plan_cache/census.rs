//! Measured census counts reused across processes.
//!
//! A census reads tensor bytes, which the checkpoint identity does not cover,
//! so counts are reused only for the files they were measured from: the same
//! headers, every file's stamp ([`DirectCheckpointV1::stamps`]: device,
//! inode, length, modification and change times) and the same host
//! implementation, which measures them. Each answer is stored twice: under
//! the census documents it answers, which other bundle builds asking the same
//! census reuse, and under the plan requests that asked for them, which
//! determine the documents and let a start of the same bundle build skip
//! asking. Counts are stored only when no file's stamp changed during the
//! measurement and every file last changed [`SETTLED`] before it started, so
//! a write within one tick of the file system's clock cannot hide behind an
//! unchanged stamp. A reused answer still passes the bundle's digest check,
//! and a load still refuses counts that differ from the bytes it encodes.

use std::os::unix::fs::MetadataExt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::Cache;
use crate::direct::{DirectCheckpointV1, digest_v1};
use runtime_bundle_abi::weight_plan::{CensusRankV1, WeightCensusV1, WeightPlanRequestV1};

/// How long before a measurement every file must have last changed for its
/// counts to be stored: far longer than a file system clock's tick.
pub(in crate::direct) const SETTLED: Duration = Duration::from_secs(2);

const DOCUMENTS_TAG: &[u8] = b"chopinfer-weight-census-documents-v1\0";
const REQUESTS_TAG: &[u8] = b"chopinfer-weight-census-requests-v1\0";

/// A census answer, with the distinct items its measurement read and the
/// source bytes they took.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::direct) struct MeasuredV1 {
    pub answer: Vec<CensusRankV1>,
    pub unique: u64,
    pub bytes_read: u64,
}

/// Each rank's census document, decoded and as the bundle emitted it.
pub(in crate::direct) type DocumentsV1 = [(u32, WeightCensusV1, Vec<u8>)];

/// The census answers a cache holds over one checkpoint's files, as they
/// are stamped now, for one host implementation.
pub(in crate::direct) struct ReuseV1<'a> {
    cache: &'a Cache,
    checkpoint: &'a DirectCheckpointV1,
    stamps: Vec<String>,
    shared_transforms: [u8; 32],
}

impl<'a> ReuseV1<'a> {
    /// `None` without a cache or when a file's stamp cannot be read.
    pub fn new(
        cache: Option<&'a Cache>,
        checkpoint: &'a DirectCheckpointV1,
        shared_transforms: [u8; 32],
    ) -> Option<Self> {
        Some(Self {
            cache: cache?,
            checkpoint,
            stamps: checkpoint.stamps().ok()?,
            shared_transforms,
        })
    }

    /// The answer stored for the census of a load whose ranks' requests,
    /// without a census, have `requests` as their [`request_digest_v1`]s.
    pub fn for_requests(&self, requests: &[[u8; 32]]) -> Option<MeasuredV1> {
        decode_v1(&self.cache.read_bytes(&self.requests_key(requests))?)
            .filter(|stored| ranked_v1(stored, requests.len()))
    }

    /// The answer to `documents`, which the load with `requests` asked for:
    /// the one stored for the same documents, or else what `measure` returns,
    /// stored when every file was settled before `settled_before` and kept
    /// its stamp. A census that measures nothing is not stored. The flag
    /// says whether the answer was reused.
    pub fn for_documents(
        &self,
        requests: &[[u8; 32]],
        documents: &DocumentsV1,
        settled_before: SystemTime,
        measure: impl FnOnce() -> Result<MeasuredV1, String>,
    ) -> Result<(MeasuredV1, bool), String> {
        if documents
            .iter()
            .all(|(_, census, _)| census.items.is_empty())
        {
            return Ok((measure()?, false));
        }
        let digests: Vec<(u32, [u8; 32])> = documents
            .iter()
            .map(|(rank, _, bytes)| (*rank, digest_v1(bytes)))
            .collect();
        let key = self.documents_key(&digests);
        if let Some(reused) = self
            .cache
            .read_bytes(&key)
            .and_then(|bytes| decode_v1(&bytes))
            .filter(|stored| answers_v1(stored, documents, &digests))
        {
            self.cache
                .write(&self.requests_key(requests), &encode_v1(&reused));
            return Ok((reused, true));
        }
        let measured = measure()?;
        if self
            .checkpoint
            .stamps()
            .is_ok_and(|after| after == self.stamps)
            && settled_v1(self.checkpoint, settled_before)
        {
            let bytes = encode_v1(&measured);
            self.cache.write(&key, &bytes);
            self.cache.write(&self.requests_key(requests), &bytes);
        }
        Ok((measured, false))
    }

    fn documents_key(&self, documents: &[(u32, [u8; 32])]) -> String {
        let mut bytes = self.files(DOCUMENTS_TAG);
        bytes.extend_from_slice(&(documents.len() as u64).to_le_bytes());
        for (rank, digest) in documents {
            bytes.extend_from_slice(&rank.to_le_bytes());
            bytes.extend_from_slice(digest);
        }
        runtime_weights::sha256_bytes(&bytes)
    }

    fn requests_key(&self, requests: &[[u8; 32]]) -> String {
        let mut bytes = self.files(REQUESTS_TAG);
        bytes.extend_from_slice(&(requests.len() as u64).to_le_bytes());
        for digest in requests {
            bytes.extend_from_slice(digest);
        }
        runtime_weights::sha256_bytes(&bytes)
    }

    fn files(&self, tag: &[u8]) -> Vec<u8> {
        files_v1(
            tag,
            &self.checkpoint.identity(),
            &self.shared_transforms,
            &self.stamps,
        )
    }
}

/// `tag`, then what names the files and their measurement.
fn files_v1(
    tag: &[u8],
    identity: &[u8; 32],
    shared_transforms: &[u8; 32],
    stamps: &[String],
) -> Vec<u8> {
    let mut bytes = tag.to_vec();
    bytes.extend_from_slice(identity);
    bytes.extend_from_slice(shared_transforms);
    bytes.extend_from_slice(&(stamps.len() as u64).to_le_bytes());
    for stamp in stamps {
        bytes.extend_from_slice(&(stamp.len() as u64).to_le_bytes());
        bytes.extend_from_slice(stamp.as_bytes());
    }
    bytes
}

/// The SHA-256 of `request` (without a census) as the bundle receives it:
/// it covers the checkpoint, the package's components and the bundle module,
/// which together determine the census the bundle answers.
pub(in crate::direct) fn request_digest_v1(request: &WeightPlanRequestV1) -> [u8; 32] {
    digest_v1(&request.encode())
}

/// Whether every file of `checkpoint` was last modified and changed before
/// `before`.
fn settled_v1(checkpoint: &DirectCheckpointV1, before: SystemTime) -> bool {
    let Ok(before) = before.duration_since(UNIX_EPOCH) else {
        return false;
    };
    let earlier = |seconds: i64, nanos: i64| {
        u64::try_from(seconds)
            .ok()
            .zip(u32::try_from(nanos).ok())
            .is_some_and(|(seconds, nanos)| Duration::new(seconds, nanos) < before)
    };
    checkpoint.shards.iter().all(|shard| {
        shard.file.metadata().is_ok_and(|metadata| {
            earlier(metadata.mtime(), metadata.mtime_nsec())
                && earlier(metadata.ctime(), metadata.ctime_nsec())
        })
    })
}

/// Whether `stored` answers ranks of a `ranks`-rank load in ascending order
/// without duplicates, as a plan request lists them.
fn ranked_v1(stored: &MeasuredV1, ranks: usize) -> bool {
    stored.answer.iter().enumerate().all(|(index, answer)| {
        (answer.rank as usize) < ranks
            && index
                .checked_sub(1)
                .is_none_or(|previous| stored.answer[previous].rank < answer.rank)
    })
}

/// Whether `stored` answers `documents`: one rank each, in order, with the
/// document's digest and one count per item, none above the item's scales.
fn answers_v1(stored: &MeasuredV1, documents: &DocumentsV1, digests: &[(u32, [u8; 32])]) -> bool {
    stored.answer.len() == documents.len()
        && stored.answer.iter().zip(documents).zip(digests).all(
            |((answer, (rank, census, _)), (_, digest))| {
                answer.rank == *rank
                    && answer.document_sha256 == *digest
                    && answer.counts.len() == census.items.len()
                    && answer
                        .counts
                        .iter()
                        .zip(&census.items)
                        .all(|(counts, item)| {
                            let scales = item.rows.saturating_mul(item.groups);
                            counts.iter().all(|count| *count <= scales)
                        })
            },
        )
}

/// `u64 unique, u64 bytes_read, u32 ranks`, then per rank `u32 rank`, the
/// 32-byte document digest, `u64 count` and that many `[u64; 3]`, all
/// little-endian.
fn encode_v1(measured: &MeasuredV1) -> Vec<u8> {
    let counts: usize = measured.answer.iter().map(|rank| rank.counts.len()).sum();
    let mut bytes = Vec::with_capacity(20 + measured.answer.len() * 44 + counts * 24);
    bytes.extend_from_slice(&measured.unique.to_le_bytes());
    bytes.extend_from_slice(&measured.bytes_read.to_le_bytes());
    bytes.extend_from_slice(&(measured.answer.len() as u32).to_le_bytes());
    for rank in &measured.answer {
        bytes.extend_from_slice(&rank.rank.to_le_bytes());
        bytes.extend_from_slice(&rank.document_sha256);
        bytes.extend_from_slice(&(rank.counts.len() as u64).to_le_bytes());
        for count in rank.counts.iter().flatten() {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
    }
    bytes
}

fn decode_v1(bytes: &[u8]) -> Option<MeasuredV1> {
    let mut reader = Reader(bytes);
    let unique = reader.u64()?;
    let bytes_read = reader.u64()?;
    let ranks = reader.u32()?;
    let mut answer = Vec::new();
    for _ in 0..ranks {
        let rank = reader.u32()?;
        let document_sha256 = reader.take(32)?.try_into().ok()?;
        let count = usize::try_from(reader.u64()?).ok()?;
        // Refused before anything is allocated when the rest cannot hold it.
        let entries = reader.take(count.checked_mul(24)?)?;
        let counts = entries
            .chunks_exact(24)
            .map(|entry| {
                let mut entry = Reader(entry);
                [(); 3].map(|()| entry.u64().expect("24 bytes"))
            })
            .collect();
        answer.push(CensusRankV1 {
            rank,
            document_sha256,
            counts,
        });
    }
    reader.0.is_empty().then_some(MeasuredV1 {
        answer,
        unique,
        bytes_read,
    })
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, length: usize) -> Option<&'a [u8]> {
        let (head, tail) = self.0.split_at_checked(length)?;
        self.0 = tail;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests;
