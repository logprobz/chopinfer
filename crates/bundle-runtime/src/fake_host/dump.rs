//! Device memory as the origins of its bytes, for snapshot byte dumps.
//!
//! A bundle's snapshot dump replays the [`super::SnapshotOp`]s its fake host
//! recorded (uploads, device copies, its emulated copy kernel and readbacks)
//! over a [`Memory`], then writes where each byte a capture or restore moved
//! came from. The copy kernel's addressing stays with the bundle; the dump
//! text and the replay mechanics are shared.

use std::collections::BTreeMap;
use std::rc::Rc;

use runtime_bundle_abi::DeviceBufferSpanV1;

use super::fnv1a;

/// Where a run of bytes came from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Origin {
    /// A buffer's bytes from `offset`, as they were before the copies.
    Held(u64, u64),
    /// An upload's host bytes from `at`.
    Uploaded(Rc<[u8]>, usize),
}

impl Origin {
    pub fn advance(&self, by: u64) -> Self {
        match self {
            Self::Held(buffer, offset) => Self::Held(*buffer, offset + by),
            Self::Uploaded(bytes, at) => Self::Uploaded(Rc::clone(bytes), at + by as usize),
        }
    }
}

/// Device memory as the runs of bytes that moved, by (buffer, start):
/// (length, origin). Every other byte is its own origin.
#[derive(Default)]
pub struct Memory(pub BTreeMap<(u64, u64), (u64, Origin)>);

impl Memory {
    fn overlapping(&self, buffer: u64, start: u64, end: u64) -> Vec<(u64, u64, Origin)> {
        let before = self
            .0
            .range(..(buffer, start))
            .next_back()
            .filter(|((held, run), (length, _))| *held == buffer && run + length > start);
        before
            .into_iter()
            .chain(self.0.range((buffer, start)..(buffer, end)))
            .map(|(&(_, run), (length, origin))| (run, *length, origin.clone()))
            .collect()
    }

    /// The origins of `span`'s bytes, as (length, origin) parts in order.
    pub fn read(&self, span: DeviceBufferSpanV1) -> Vec<(u64, Origin)> {
        let (buffer, end) = (span.buffer.raw, span.offset + span.length);
        let mut parts = Vec::new();
        let mut at = span.offset;
        for (run, length, origin) in self.overlapping(buffer, span.offset, end) {
            if run > at {
                parts.push((run - at, Origin::Held(buffer, at)));
                at = run;
            }
            let take = (run + length).min(end) - at;
            parts.push((take, origin.advance(at - run)));
            at += take;
        }
        if at < end {
            parts.push((end - at, Origin::Held(buffer, at)));
        }
        parts
    }

    /// Gives `span`'s bytes the origins of `parts`, which cover it.
    pub fn write(&mut self, span: DeviceBufferSpanV1, parts: Vec<(u64, Origin)>) {
        let (buffer, start, end) = (span.buffer.raw, span.offset, span.offset + span.length);
        for (run, length, origin) in self.overlapping(buffer, start, end) {
            self.0.remove(&(buffer, run));
            if run < start {
                self.0.insert((buffer, run), (start - run, origin.clone()));
            }
            if run + length > end {
                self.0.insert(
                    (buffer, end),
                    (run + length - end, origin.advance(end - run)),
                );
            }
        }
        let mut at = start;
        for (length, origin) in parts {
            if origin != Origin::Held(buffer, at) {
                self.0.insert((buffer, at), (length, origin));
            }
            at += length;
        }
        assert_eq!(at, end);
    }

    pub fn copy(&mut self, from: DeviceBufferSpanV1, to: DeviceBufferSpanV1) {
        assert_eq!(from.length, to.length);
        let parts = self.read(from);
        self.write(to, parts);
    }

    /// The uploaded bytes `span` holds; every byte must have been uploaded.
    pub fn bytes(&self, span: DeviceBufferSpanV1) -> Vec<u8> {
        self.read(span)
            .into_iter()
            .flat_map(|(length, origin)| match origin {
                Origin::Uploaded(bytes, at) => bytes[at..at + length as usize].to_vec(),
                Origin::Held(..) => panic!("{span:?} was not uploaded"),
            })
            .collect()
    }

    /// The moved runs whose (buffer, start) `keep` takes, in buffer order,
    /// as (buffer, start, length, origin), each joined with the next when
    /// the bytes continue.
    pub fn runs(&self, keep: impl Fn(u64, u64) -> bool) -> Vec<(u64, u64, u64, Origin)> {
        let mut joined: Vec<(u64, u64, u64, Origin)> = Vec::new();
        for (&(buffer, start), (length, origin)) in &self.0 {
            if !keep(buffer, start) {
                continue;
            }
            if let Some(last) = joined.last_mut()
                && last.0 == buffer
                && last.1 + last.2 == start
                && last.3.advance(last.2) == *origin
            {
                last.2 += length;
                continue;
            }
            joined.push((buffer, start, *length, origin.clone()));
        }
        joined
    }
}

/// A dump line's account of `length` bytes from `origin`: uploads in hex,
/// or by digest past `digest_above` bytes when set.
pub fn describe(origin: &Origin, length: u64, digest_above: Option<usize>) -> String {
    match origin {
        Origin::Held(buffer, offset) => format!("held {buffer:#x}+{offset}"),
        Origin::Uploaded(bytes, at) => {
            let bytes = &bytes[*at..*at + length as usize];
            if digest_above.is_some_and(|limit| bytes.len() > limit) {
                return format!("bytes fnv {:016x}", fnv1a(bytes));
            }
            let hex = bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            format!("bytes {hex}")
        }
    }
}

/// The copies a snapshot copy kernel makes of `pieces`, laid end to end
/// across the snapshot's `pages` of `page_bytes` in `pool`: (from, to) in
/// snapshot order, into the pages when `to_pages`, else out of them.
pub fn paged_copies(
    pieces: impl IntoIterator<Item = DeviceBufferSpanV1>,
    pool: DeviceBufferSpanV1,
    pages: &[u64],
    page_bytes: u64,
    to_pages: bool,
) -> Vec<(DeviceBufferSpanV1, DeviceBufferSpanV1)> {
    let mut copies = Vec::new();
    let mut snapshot = 0;
    for piece in pieces {
        let mut done = 0;
        while done < piece.length {
            let within = snapshot % page_bytes;
            let length = (page_bytes - within).min(piece.length - done);
            let paged = DeviceBufferSpanV1::new(
                pool.buffer,
                pool.offset + pages[(snapshot / page_bytes) as usize] * page_bytes + within,
                length,
            );
            let part = DeviceBufferSpanV1::new(piece.buffer, piece.offset + done, length);
            copies.push(if to_pages {
                (part, paged)
            } else {
                (paged, part)
            });
            snapshot += length;
            done += length;
        }
    }
    copies
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;

    fn span(buffer: u32, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, buffer),
            offset,
            length,
        )
    }

    #[test]
    fn copies_carry_origins_and_runs_join_continued_bytes() {
        let mut memory = Memory::default();
        let upload: Rc<[u8]> = vec![1, 2, 3, 4].into();
        memory.write(span(1, 0, 4), vec![(4, Origin::Uploaded(upload, 0))]);
        memory.copy(span(1, 0, 2), span(2, 8, 2));
        memory.copy(span(1, 2, 2), span(2, 10, 2));
        assert_eq!(memory.bytes(span(2, 8, 4)), [1, 2, 3, 4]);
        let runs = memory.runs(|buffer, _| buffer == span(2, 0, 0).buffer.raw);
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].1, runs[0].2), (8, 4));
        assert_eq!(describe(&runs[0].3, 4, None), "bytes 01020304");
        assert_eq!(
            describe(&runs[0].3, 4, Some(2)),
            format!("bytes fnv {:016x}", fnv1a(&[1, 2, 3, 4]))
        );
        let held = memory.read(span(2, 6, 4));
        assert_eq!(held[0], (2, Origin::Held(span(2, 0, 0).buffer.raw, 6)));
    }

    #[test]
    fn pieces_lie_end_to_end_across_the_pages() {
        let pool = span(9, 64, 1 << 20);
        let copies = paged_copies([span(1, 0, 6), span(2, 4, 3)], pool, &[5, 2], 8, true);
        let to = |offset: u64, length| span(9, 64 + offset, length);
        assert_eq!(
            copies,
            [
                (span(1, 0, 6), to(40, 6)),
                (span(2, 4, 2), to(46, 2)),
                (span(2, 6, 1), to(16, 1)),
            ]
        );
    }
}
