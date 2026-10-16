//! Device-side poisoning and comparison (`families::device_check`), for
//! checks that run every phase at every live count of a serving domain:
//! copying each output to the host would take hours there.
//!
//! A batch enqueues poison fills, the launches under test and comparisons on
//! one stream; [`DeviceCheck::finish`] then reads the counts once. Each
//! comparison requires an output's live bytes to equal an expected buffer's,
//! every live element (at the output's element width) to differ from the
//! poison element, so the launch wrote it, and every byte past them to keep
//! [`poison::POISON`], as [`poison::check`] with a byte-prefix write map
//! does. Comparing a buffer with itself checks its write map alone. A
//! failing slot names its label; the host checks reproduce the details.

use crate::families::device_check::{COMPARE, CUBIN, FILL, THREADS};
use crate::{LabResult, Session, launch, poison};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Stream};

/// Most CTAs of one fill or comparison: four per SM120 SM.
const MAX_CTAS: u64 = 4 * 188;
/// The poison word: four [`poison::POISON`] bytes.
const POISON_WORD: u32 = u32::from_ne_bytes([poison::POISON; 4]);

pub struct DeviceCheck {
    fill: Function,
    compare: Function,
    /// Three counts per slot: differing live words, live elements that
    /// still hold the poison, changed poison words.
    counts: DeviceBuffer,
    slots: usize,
    labels: Vec<String>,
}

impl DeviceCheck {
    /// A checker of at most `slots` comparisons per batch.
    pub fn new(session: &Session, slots: usize) -> LabResult<Self> {
        let module = session.module(CUBIN)?;
        let function = |name: &str| module.function(name).map_err(|e| e.to_string());
        let check = Self {
            fill: function(FILL)?,
            compare: function(COMPARE)?,
            counts: session.allocate((3 * slots * 4).next_multiple_of(16))?,
            slots,
            labels: Vec::new(),
        };
        check.fill_words(&session.stream, &check.counts, 0)?;
        Ok(check)
    }

    fn ctas(vectors: u64) -> u32 {
        vectors.div_ceil(u64::from(THREADS)).clamp(1, MAX_CTAS) as u32
    }

    fn config(vectors: u64) -> LabResult<runtime_cuda::LaunchConfig> {
        launch::simple_config(
            (Self::ctas(vectors), 1, 1),
            (THREADS, 1, 1),
            0,
            runtime_cuda::LaunchConfig::NO_CLUSTER,
            false,
        )
    }

    fn vectors(buffer: &DeviceBuffer) -> LabResult<u64> {
        let bytes = buffer.byte_len();
        if bytes % 16 != 0 {
            return Err(format!(
                "a {bytes}-byte buffer is not whole 16-byte vectors"
            ));
        }
        Ok((bytes / 16) as u64)
    }

    fn fill_words(&self, stream: &Stream, buffer: &DeviceBuffer, word: u32) -> LabResult<()> {
        let bytes = buffer.byte_len() as u64;
        self.fill
            .launch(
                stream,
                Self::config(bytes.div_ceil(16))?,
                &[
                    KernelArgument::Buffer(buffer),
                    KernelArgument::U64(bytes),
                    KernelArgument::U32(word),
                ],
            )
            .map_err(|e| e.to_string())
    }

    /// Enqueues filling every byte of `buffers` with [`poison::POISON`] (as
    /// [`poison::poison`] does on the host).
    pub fn poison(&self, stream: &Stream, buffers: &[&DeviceBuffer]) -> LabResult<()> {
        for buffer in buffers {
            self.fill_words(stream, buffer, POISON_WORD)?;
        }
        Ok(())
    }

    /// Enqueues comparing `got`'s first `live_bytes` with `want`'s, checking
    /// that none of its live elements of `element_bytes` (2 or 4) holds the
    /// poison and that every later byte of `got` does.
    pub fn compare(
        &mut self,
        stream: &Stream,
        label: impl Into<String>,
        want: &DeviceBuffer,
        got: &DeviceBuffer,
        live_bytes: usize,
        element_bytes: usize,
    ) -> LabResult<()> {
        let label = label.into();
        if self.labels.len() == self.slots {
            return Err(format!(
                "{label}: all {} comparison slots are used",
                self.slots
            ));
        }
        if element_bytes != 2 && element_bytes != 4 {
            return Err(format!(
                "{label}: {element_bytes}-byte elements are not 2 or 4 bytes"
            ));
        }
        if live_bytes % 4 != 0 || live_bytes > got.byte_len() || live_bytes > want.byte_len() {
            return Err(format!(
                "{label}: {live_bytes} live bytes of a {}-byte output and a {}-byte expectation",
                got.byte_len(),
                want.byte_len()
            ));
        }
        let vectors = Self::vectors(got)?;
        self.compare
            .launch(
                stream,
                Self::config(vectors)?,
                &[
                    KernelArgument::Buffer(want),
                    KernelArgument::Buffer(got),
                    KernelArgument::U64((live_bytes / 4) as u64),
                    KernelArgument::U64(vectors),
                    KernelArgument::U32(POISON_WORD),
                    KernelArgument::U32(element_bytes as u32),
                    KernelArgument::Buffer(&self.counts),
                    KernelArgument::U32(self.labels.len() as u32),
                ],
            )
            .map_err(|e| e.to_string())?;
        self.labels.push(label);
        Ok(())
    }

    /// Waits for the session's stream, reads the batch's counts and clears
    /// them: the number of comparisons and each failing one's label and
    /// counts. A count of no comparison is an error.
    pub fn finish(&mut self, session: &Session) -> LabResult<(usize, Vec<String>)> {
        let bytes = poison::read(session, &self.counts)?;
        let count = |i: usize| u32::from_ne_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        let failures: Vec<String> = self
            .labels
            .iter()
            .enumerate()
            .filter_map(|(slot, label)| {
                let (differ, unwritten, changed) =
                    (count(3 * slot), count(3 * slot + 1), count(3 * slot + 2));
                (differ != 0 || unwritten != 0 || changed != 0).then(|| {
                    format!(
                        "{label}: {differ} live words differ, {unwritten} live elements \
                         unwritten, {changed} poison words changed"
                    )
                })
            })
            .collect();
        let unused = (3 * self.labels.len()..3 * self.slots).find(|&i| count(i) != 0);
        let compared = std::mem::take(&mut self.labels).len();
        self.fill_words(&session.stream, &self.counts, 0)?;
        if let Some(i) = unused {
            return Err(format!("count {i} of no comparison is {}", count(i)));
        }
        Ok((compared, failures))
    }
}

/// Splits a long exhaustive run over jobs: `CHOPINFER_LIVE_COUNTS`, an
/// inclusive range `first-last`.
pub const LIVE_COUNTS: &str = "CHOPINFER_LIVE_COUNTS";

/// The counts of `all` that `range` (a [`LIVE_COUNTS`] value) keeps: all of
/// them when it is unset, else its inclusive range, which must lie within
/// `all`.
pub fn live_counts_in(
    all: std::ops::RangeInclusive<u32>,
    range: Option<&str>,
) -> LabResult<std::ops::RangeInclusive<u32>> {
    let Some(range) = range else {
        return Ok(all);
    };
    let parsed = range
        .split_once('-')
        .and_then(|(first, last)| Some((first.parse::<u32>().ok()?, last.parse::<u32>().ok()?)));
    match parsed {
        Some((first, last)) if first <= last && all.contains(&first) && all.contains(&last) => {
            Ok(first..=last)
        }
        _ => Err(format!(
            "{LIVE_COUNTS}={range} is not a range first-last within {all:?}"
        )),
    }
}

/// [`live_counts_in`] the environment's [`LIVE_COUNTS`].
pub fn live_counts(all: std::ops::RangeInclusive<u32>) -> LabResult<std::ops::RangeInclusive<u32>> {
    live_counts_in(all, std::env::var(LIVE_COUNTS).ok().as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_count_ranges_stay_within_the_domain() {
        assert_eq!(live_counts_in(1..=8192, None).unwrap(), 1..=8192);
        assert_eq!(live_counts_in(1..=8192, Some("1-4096")).unwrap(), 1..=4096);
        assert_eq!(
            live_counts_in(1..=8192, Some("8192-8192")).unwrap(),
            8192..=8192
        );
        for bad in ["0-10", "10-9", "1-8193", "1", "a-b", ""] {
            assert!(live_counts_in(1..=8192, Some(bad)).is_err(), "{bad}");
        }
    }
}
