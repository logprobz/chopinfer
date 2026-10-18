//! The Engram rows of a step's target rows, gathered by the host from the
//! host-mapped tables (`chop.host-mapped-gather.v1`), and the bounded
//! readback of the draft ids a verify step hashes.
//!
//! A target row's hash reads its own token and up to `max_ngram - 1`
//! earlier ones (`own_ops::engram::HashTables::position`): first the step's
//! earlier rows, then the request's committed history. For a verify step the
//! step's rows are the bonus token and the drafts the drafter left on the
//! device, which [`read_draft_ids`] brings back in one copy of exactly
//! `width` int32 ids per drafting item. The step's rows are provisional:
//! only [`EngramRows::commit`] moves the kept ones into the history, so a
//! rejected draft never reaches a later step's hash.
//!
//! Each Engram layer's ids select rows of both planes of that layer's
//! table, codes and scales; [`EngramRows::gather`] has the host copy them
//! into the bundle's staging buffers on one stream, in row order then
//! column order, so the work after it on that stream reads them.

use std::collections::BTreeMap;
use std::ops::Range;

use model_deepseek41_contract::own_ops::engram::HashTables;
use runtime_bundle_abi::{
    BundleAbiError, BundleCompiledImageHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1,
    HostServicesV1, MappedRowGatherV1,
};

/// Why staging failed: a refused input, or the host's answer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    Refused(&'static str),
    Host(BundleAbiError),
}

impl From<BundleAbiError> for Error {
    fn from(error: BundleAbiError) -> Self {
        Self::Host(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// One plane of a mapped table: its view in the compiled image and the
/// bytes of one row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plane {
    pub view: Vec<u8>,
    pub row_bytes: u64,
}

/// One Engram layer's table: its two planes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Table {
    pub codes: Plane,
    pub scales: Plane,
}

/// The mapped tables of one rank: every Engram layer's, in layer order,
/// as views of `image`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Mapped<'a> {
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
    pub tables: &'a [Table],
}

/// Where one Engram layer's gathered rows go: one span per plane, each
/// `rows * columns * row_bytes` long.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Destination {
    pub codes: DeviceBufferSpanV1,
    pub scales: DeviceBufferSpanV1,
}

/// One request's target rows in a step: its tokens in order (all of a
/// prefill chunk, or the bonus token then the drafts of a verify), and
/// whether each is text (an image position takes no part in an n-gram).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Step<'a> {
    pub request: u64,
    pub tokens: &'a [u32],
    /// `None`: every token is text.
    pub text: Option<&'a [bool]>,
}

/// The hash ids of a step's target rows, `[row][layer][column]`, rows in
/// step order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Hashed {
    rows: usize,
    ids: Vec<u32>,
}

impl Hashed {
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
}

/// The hash and each request's committed history: its last
/// `max_ngram - 1` compressed ids, oldest first (`own_ops::engram::DEAD` for
/// an image position), in storage of that capacity whatever the prefix
/// length.
pub struct EngramRows {
    hash: HashTables,
    history: BTreeMap<u64, Vec<i64>>,
}

impl EngramRows {
    #[must_use]
    pub fn new(hash: HashTables) -> Self {
        Self {
            hash,
            history: BTreeMap::new(),
        }
    }

    fn lookback(&self) -> usize {
        self.hash.engram().max_ngram as usize - 1
    }

    fn width(&self) -> usize {
        self.hash.engram().layers.len() * self.hash.columns()
    }

    fn compressed(&self, step: &Step<'_>) -> Result<Vec<i64>> {
        self.compressed_range(step, 0..step.tokens.len())
    }

    /// The compressed ids of `step.tokens[range]`, after checking the
    /// step's text flags against all of its tokens.
    fn compressed_range(&self, step: &Step<'_>, range: Range<usize>) -> Result<Vec<i64>> {
        if step
            .text
            .is_some_and(|text| text.len() != step.tokens.len())
        {
            return Err(Error::Refused(
                "Engram text flags differ from the step's tokens",
            ));
        }
        range
            .map(|index| {
                let text = step.text.is_none_or(|text| text[index]);
                self.hash
                    .compress(step.tokens[index], text)
                    .map_err(|refusal| Error::Refused(refusal.0))
            })
            .collect()
    }

    /// The history after `old` with the first `kept` tokens of `step`
    /// appended: the last `lookback` ids, in storage of exactly that
    /// capacity. Only the tokens that window keeps are compressed.
    fn window(&self, old: &[i64], step: &Step<'_>, kept: usize) -> Result<Vec<i64>> {
        let lookback = self.lookback();
        let new = self.compressed_range(step, kept - kept.min(lookback)..kept)?;
        let mut window = Vec::with_capacity(lookback);
        window.extend_from_slice(&old[old.len() - old.len().min(lookback - new.len())..]);
        window.extend_from_slice(&new);
        Ok(window)
    }

    /// The hash ids of every target row of `steps`. Nothing is committed.
    pub fn hash(&self, steps: &[Step<'_>]) -> Result<Hashed> {
        let width = self.width();
        let lookback = self.lookback();
        let rows: usize = steps.iter().map(|step| step.tokens.len()).sum();
        let mut ids = vec![0; rows * width];
        let mut out = ids.chunks_mut(width);
        let mut recent = Vec::with_capacity(lookback + 1);
        for step in steps {
            let history = self
                .history
                .get(&step.request)
                .map_or(&[][..], Vec::as_slice);
            let compressed = self.compressed(step)?;
            for position in 0..compressed.len() {
                // Own id first, then older ones back to the history's start.
                recent.clear();
                recent.extend(compressed[..=position].iter().rev().take(lookback + 1));
                let missing = (lookback + 1).saturating_sub(recent.len());
                recent.extend(history.iter().rev().take(missing));
                let row = out.next().expect("one output row per target row");
                self.hash
                    .position(&recent, row)
                    .map_err(|refusal| Error::Refused(refusal.0))?;
            }
        }
        Ok(Hashed { rows, ids })
    }

    /// Has the host gather `hashed`'s rows of every Engram layer's table,
    /// both planes, from `mapped` into `destinations` (one per layer, in
    /// layer order) on `stream`.
    pub fn gather(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        mapped: Mapped<'_>,
        destinations: &[Destination],
        hashed: &Hashed,
    ) -> Result<()> {
        let layers = self.hash.engram().layers.len();
        let columns = self.hash.columns();
        if mapped.tables.len() != layers || destinations.len() != layers {
            return Err(Error::Refused("Engram staging names a table per layer"));
        }
        if hashed.ids.len() != hashed.rows * layers * columns {
            return Err(Error::Refused(
                "Engram hash ids are not rows * layers * columns",
            ));
        }
        // Every span is checked before the host copies anything, so a
        // refusal leaves the staging buffers untouched.
        let count = (hashed.rows * columns) as u64;
        for (table, destination) in mapped.tables.iter().zip(destinations) {
            for (plane, span) in [
                (&table.codes, destination.codes),
                (&table.scales, destination.scales),
            ] {
                if Some(span.length) != count.checked_mul(plane.row_bytes) {
                    return Err(Error::Refused(
                        "an Engram staging span is not rows * columns * row bytes",
                    ));
                }
            }
        }
        if hashed.rows == 0 {
            return Ok(());
        }
        let mut rows = Vec::with_capacity(hashed.rows * columns);
        for (layer, (table, destination)) in mapped.tables.iter().zip(destinations).enumerate() {
            rows.clear();
            rows.extend(
                hashed
                    .ids
                    .chunks(layers * columns)
                    .flat_map(|row| &row[layer * columns..(layer + 1) * columns]),
            );
            for (plane, span) in [
                (&table.codes, destination.codes),
                (&table.scales, destination.scales),
            ] {
                services.gather_mapped_rows(
                    stream,
                    span,
                    MappedRowGatherV1 {
                        image: mapped.image,
                        rank: mapped.rank,
                        view_id: &plane.view,
                        row_bytes: plane.row_bytes,
                        rows: &rows,
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Moves the first `kept` tokens of `step` into its request's history:
    /// all of a prefill chunk, or the tokens a verify accepted. A refusal
    /// leaves the history as it was.
    pub fn commit(&mut self, step: &Step<'_>, kept: usize) -> Result<()> {
        if kept > step.tokens.len() {
            return Err(Error::Refused(
                "Engram commit keeps more tokens than the step ran",
            ));
        }
        let window = self.window(self.history(step.request), step, kept)?;
        self.history.insert(step.request, window);
        Ok(())
    }

    /// Sets a request's history to the last tokens of its committed prefix
    /// (a restored prefix, or a preempted request resumed): `tokens` and
    /// their text flags, oldest first. Only the last `max_ngram - 1` are
    /// read; earlier ones are ignored. A refusal leaves the history as it
    /// was.
    pub fn restore(&mut self, request: u64, tokens: &[u32], text: Option<&[bool]>) -> Result<()> {
        let step = Step {
            request,
            tokens,
            text,
        };
        let window = self.window(&[], &step, tokens.len())?;
        self.history.insert(request, window);
        Ok(())
    }

    /// Forgets a retired or cancelled-before-commit request's history.
    pub fn forget(&mut self, request: u64) {
        self.history.remove(&request);
    }

    /// The compressed history of `request`, oldest first.
    #[must_use]
    pub fn history(&self, request: u64) -> &[i64] {
        self.history.get(&request).map_or(&[], Vec::as_slice)
    }
}

/// Copies the draft ids the drafter wrote on the device back to the host:
/// `source` holds exactly `ids.len() * W` little-endian int32 ids, one row
/// of `W` per drafting item, and nothing else is read.
pub fn read_draft_ids<const W: usize>(
    services: &HostServicesV1,
    stream: BundleStreamHandleV1,
    source: DeviceBufferSpanV1,
    ids: &mut [[u32; W]],
) -> Result<()> {
    let bytes = ids.len() * W * 4;
    if source.length != bytes as u64 {
        return Err(Error::Refused(
            "the draft id readback is not items * width ids",
        ));
    }
    if bytes == 0 {
        return Ok(());
    }
    let mut read = vec![0_u8; bytes];
    services.copy_device_to_host(stream, &mut read, source)?;
    for (row, chunk) in ids.iter_mut().zip(read.chunks_exact(W * 4)) {
        for (id, word) in row.iter_mut().zip(chunk.chunks_exact(4)) {
            *id = u32::from_le_bytes(word.try_into().expect("four bytes"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
