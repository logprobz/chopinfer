//! Prefix-cache snapshot headers and the prefix-state entry bodies.
//!
//! A snapshot's header follows its pieces ([`crate::snapshot::Table`]). The
//! host writes it at capture and reads it back at restore, which refuses a
//! snapshot whose header does not match. A [`HeaderLayout`] is an 8-byte
//! magic followed by little-endian 64-bit [`Word`]s; the magic names the
//! layout's version, so a bundle that keeps an older layout keeps its
//! snapshot bytes. [`HeaderLayout::V1`] is the shared layout: the tokens, the
//! drafter's frontier and cursor, the capturing request, the rank count and
//! the digest of the declared snapshot layout ([`layout_digest`]).
//!
//! The ABI entry bodies ([`describe`], [`capture`], [`restore`],
//! [`restore_in_slot`]) run a [`PrefixInstance`] only while it is usable:
//! not poisoned and not re-entered.

use std::cell::RefCell;
use std::collections::BTreeSet;

use kernel_core::contract::Refusal;
use runtime_bundle_abi::{
    BundleHandleV1, BundlePrefixStateInfoV1, BundleStatusV1, PrefixStateRequestV1,
};

use crate::state::{Capture, Owner, RequestState, Scope};
use crate::table::{InstanceTable, Live, usable, with_instance};

/// One word of a header after its magic. An optional value is `u64::MAX`
/// when absent.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Word {
    /// The committed tokens the snapshot covers; never zero.
    Tokens,
    /// Optional: the drafter rows that ran, at most the tokens.
    Frontier,
    /// The request that captured the snapshot.
    Owner,
    /// Optional: the rows the drafter's derived state covers, at most the
    /// tokens.
    Cursor,
    /// The capturing instance's rank count.
    Ranks,
    /// The [`layout_digest`] of the capturing declaration.
    Digest,
    /// Zero.
    Zero,
}

/// The byte layout of a header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeaderLayout {
    pub magic: [u8; 8],
    pub words: &'static [Word],
}

impl HeaderLayout {
    /// The shared layout, 64 bytes.
    pub const V1: Self = Self {
        magic: *b"CHOPSN01",
        words: &[
            Word::Tokens,
            Word::Frontier,
            Word::Owner,
            Word::Cursor,
            Word::Ranks,
            Word::Digest,
            Word::Zero,
        ],
    };

    pub const fn bytes(&self) -> u64 {
        8 + 8 * self.words.len() as u64
    }

    /// Checks that the layout holds the tokens and each other value at most
    /// once, and that `state` declares its size.
    pub fn check(&self, state: &RequestState) -> Result<(), Refusal> {
        let mut seen = BTreeSet::new();
        if !self.words.contains(&Word::Tokens)
            || self
                .words
                .iter()
                .any(|&word| word != Word::Zero && !seen.insert(word))
        {
            return Err(Refusal("a header holds the tokens and each value once"));
        }
        if state.header_bytes != self.bytes() {
            return Err(Refusal("the declared header is not the layout's size"));
        }
        Ok(())
    }
}

/// What a snapshot's header records of its request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Header {
    pub tokens: u64,
    pub frontier: Option<u64>,
    pub owner: Option<u64>,
    pub cursor: Option<u64>,
}

/// What a header records of the instance and its declaration: both sides of
/// a snapshot must agree on them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stamp {
    pub ranks: u64,
    pub digest: u64,
}

const ABSENT: u64 = u64::MAX;

impl Header {
    /// The header's bytes in `layout`. Every value it holds needs a word,
    /// and every word but an optional one a value.
    pub fn encode(&self, layout: &HeaderLayout, stamp: Stamp) -> Result<Vec<u8>, Refusal> {
        let held = |word| layout.words.contains(&word);
        if self.tokens == 0
            || [self.frontier, self.cursor]
                .iter()
                .flatten()
                .any(|&value| value > self.tokens)
        {
            return Err(Refusal("a header's rows lie past its tokens"));
        }
        if self.frontier.is_some() && !held(Word::Frontier)
            || self.cursor.is_some() && !held(Word::Cursor)
            || self.owner.is_some() != held(Word::Owner)
        {
            return Err(Refusal("the header layout does not hold these values"));
        }
        let mut bytes = layout.magic.to_vec();
        for &word in layout.words {
            let value = match word {
                Word::Tokens => self.tokens,
                Word::Frontier => self.frontier.unwrap_or(ABSENT),
                Word::Owner => self.owner.unwrap_or(ABSENT),
                Word::Cursor => self.cursor.unwrap_or(ABSENT),
                Word::Ranks => stamp.ranks,
                Word::Digest => stamp.digest,
                Word::Zero => 0,
            };
            bytes.extend(value.to_le_bytes());
        }
        Ok(bytes)
    }

    /// The header `bytes` hold in `layout`, if it is one captured with
    /// `stamp` at `tokens` committed tokens.
    pub fn decode(
        bytes: &[u8],
        layout: &HeaderLayout,
        stamp: Stamp,
        tokens: u64,
    ) -> Result<Self, Refusal> {
        let refused = Err(Refusal("the snapshot header does not match"));
        if bytes.len() as u64 != layout.bytes() || bytes[..8] != layout.magic || tokens == 0 {
            return refused;
        }
        let mut header = Self {
            tokens,
            frontier: None,
            owner: None,
            cursor: None,
        };
        let optional = |value| (value != ABSENT).then_some(value);
        for (&word, value) in layout.words.iter().zip(bytes[8..].chunks_exact(8)) {
            let value = u64::from_le_bytes(value.try_into().expect("8 bytes"));
            let valid = match word {
                Word::Tokens => value == tokens,
                Word::Frontier => {
                    header.frontier = optional(value);
                    header.frontier.is_none_or(|frontier| frontier <= tokens)
                }
                Word::Owner => {
                    header.owner = Some(value);
                    true
                }
                Word::Cursor => {
                    header.cursor = optional(value);
                    header.cursor.is_none_or(|cursor| cursor <= tokens)
                }
                Word::Ranks => value == stamp.ranks,
                Word::Digest => value == stamp.digest,
                Word::Zero => value == 0,
            };
            if !valid {
                return refused;
            }
        }
        Ok(header)
    }
}

/// A 64-bit FNV-1a digest of what a declaration's snapshot holds: each
/// captured buffer's name, layers, planes, scope (the positions that address
/// a ring or history cell) and capture, in order, and the header's size. It
/// does not depend on the instance's extent or the slot, so a snapshot
/// restores into any slot of any instance of the declaration.
pub fn layout_digest(state: &RequestState) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    let mut eat = |bytes: &[u8]| {
        for &byte in bytes {
            digest = (digest ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for buffer in &state.buffers {
        let (kind, rows) = match buffer.capture {
            Capture::None => continue,
            Capture::Full => (1, 0),
            Capture::RingWindow { tokens } => (2, tokens),
            Capture::LastRows { rows } => (3, rows),
        };
        eat(buffer.name.as_bytes());
        eat(&[0]);
        eat(&buffer.layers.to_le_bytes());
        eat(&(buffer.planes.len() as u64).to_le_bytes());
        for plane in &buffer.planes {
            eat(&plane.to_le_bytes());
        }
        let (scope, positions) = match buffer.scope {
            Scope::Slot => (1, 0),
            Scope::WindowRing { tokens } => (2, tokens),
            Scope::Rows { rows } => (3, rows),
        };
        eat(&[scope]);
        eat(&positions.to_le_bytes());
        eat(&[kind]);
        eat(&rows.to_le_bytes());
    }
    eat(&state.header_bytes.to_le_bytes());
    digest
}

/// Whether `request` may restore a snapshot of `state` whose header is
/// `header`. A declaration with frontier-owned state records the drafter's
/// frontier, and only the capturing request may restore a snapshot whose
/// frontier lags into a page before the last one: the continuation would
/// rewrite drafter rows in pages other requests share. Without such state
/// the header holds no frontier.
pub fn may_restore(state: &RequestState, header: &Header, request: u64) -> bool {
    let owned = state
        .buffers
        .iter()
        .any(|buffer| buffer.capture != Capture::None && buffer.owner == Owner::FrontierOwner);
    match header.frontier {
        None => !owned,
        Some(frontier) => {
            owned
                && header.tokens > 0
                && (frontier >= (header.tokens - 1) / state.page_tokens * state.page_tokens
                    || header.owner == Some(request))
        }
    }
}

/// An instance that serves prefix snapshots.
pub trait PrefixInstance: Live {
    /// Serving pages one snapshot takes.
    fn state_pages(&self) -> Result<u64, BundleStatusV1>;
    /// Writes the request's committed state into `request.pages`.
    fn capture(&mut self, request: &PrefixStateRequestV1) -> Result<(), BundleStatusV1>;
    /// Gives an unseen request the state a snapshot holds, in the host's
    /// state slot `slot` when the call names one.
    fn restore(
        &mut self,
        request: &PrefixStateRequestV1,
        slot: Option<u64>,
    ) -> Result<(), BundleStatusV1>;
}

fn status(result: Result<(), BundleStatusV1>) -> BundleStatusV1 {
    result.err().unwrap_or(BundleStatusV1::Success)
}

pub fn describe<T: PrefixInstance>(
    instances: &RefCell<InstanceTable<T>>,
    handle: BundleHandleV1,
) -> BundlePrefixStateInfoV1 {
    let Ok(mut table) = instances.try_borrow_mut() else {
        return BundlePrefixStateInfoV1::failure(BundleStatusV1::InvalidState);
    };
    usable(&mut table, handle)
        .and_then(|instance| instance.state_pages())
        .map_or_else(
            BundlePrefixStateInfoV1::failure,
            BundlePrefixStateInfoV1::success,
        )
}

pub fn capture<T: PrefixInstance>(
    instances: &RefCell<InstanceTable<T>>,
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
) -> BundleStatusV1 {
    with_instance(instances, handle, |instance| {
        status(instance.capture(&request))
    })
}

pub fn restore<T: PrefixInstance>(
    instances: &RefCell<InstanceTable<T>>,
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
) -> BundleStatusV1 {
    with_instance(instances, handle, |instance| {
        status(instance.restore(&request, None))
    })
}

pub fn restore_in_slot<T: PrefixInstance>(
    instances: &RefCell<InstanceTable<T>>,
    handle: BundleHandleV1,
    request: PrefixStateRequestV1,
    slot: u64,
) -> BundleStatusV1 {
    with_instance(instances, handle, |instance| {
        status(instance.restore(&request, Some(slot)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Collectives;
    use crate::state::synthetic;

    const STAMP: Stamp = Stamp {
        ranks: 4,
        digest: 0x1234,
    };

    #[test]
    fn headers_round_trip_in_every_layout() {
        // A layout with a cursor and reserved words, and one with a
        // frontier and owner, both older than the shared one.
        let cursor_layout = HeaderLayout {
            magic: *b"TESTCUR1",
            words: &[
                Word::Tokens,
                Word::Cursor,
                Word::Ranks,
                Word::Zero,
                Word::Zero,
                Word::Zero,
                Word::Zero,
            ],
        };
        let owner_layout = HeaderLayout {
            magic: *b"TESTOWN2",
            words: &[Word::Tokens, Word::Frontier, Word::Owner],
        };
        assert_eq!(
            (
                HeaderLayout::V1.bytes(),
                cursor_layout.bytes(),
                owner_layout.bytes()
            ),
            (64, 64, 32)
        );
        let cases = [
            (HeaderLayout::V1, Some(67), Some(9), Some(40)),
            (HeaderLayout::V1, None, Some(9), None),
            (cursor_layout, None, None, Some(68)),
            (cursor_layout, None, None, None),
            (owner_layout, Some(0), Some(u64::MAX), None),
            (owner_layout, None, Some(3), None),
        ];
        for (layout, frontier, owner, cursor) in cases {
            let header = Header {
                tokens: 68,
                frontier,
                owner,
                cursor,
            };
            let bytes = header.encode(&layout, STAMP).unwrap();
            assert_eq!(bytes.len() as u64, layout.bytes());
            assert_eq!(&bytes[..8], &layout.magic);
            assert_eq!(&bytes[8..16], &68_u64.to_le_bytes());
            assert_eq!(Header::decode(&bytes, &layout, STAMP, 68), Ok(header));
            // Another boundary, magic, version or rank count, and any
            // flipped bit of a checked word.
            assert!(Header::decode(&bytes, &layout, STAMP, 69).is_err());
            for other in [HeaderLayout::V1, cursor_layout, owner_layout] {
                if other != layout {
                    assert!(Header::decode(&bytes, &other, STAMP, 68).is_err());
                }
            }
            let stamp = Stamp { ranks: 2, ..STAMP };
            let ranked = layout.words.contains(&Word::Ranks);
            assert_eq!(Header::decode(&bytes, &layout, stamp, 68).is_err(), ranked);
            for (index, &word) in layout.words.iter().enumerate() {
                let mut corrupt = bytes.clone();
                corrupt[8 + 8 * index + 7] ^= 0x40;
                let decoded = Header::decode(&corrupt, &layout, STAMP, 68);
                // Only the request's id may be any value.
                assert_eq!(decoded.is_ok(), word == Word::Owner, "{word:?}");
            }
        }
        // Values the layout cannot hold, and rows past the tokens.
        let header = Header {
            tokens: 68,
            frontier: Some(67),
            owner: None,
            cursor: None,
        };
        assert!(header.encode(&cursor_layout, STAMP).is_err());
        assert!(header.encode(&HeaderLayout::V1, STAMP).is_err());
        let late = Header {
            cursor: Some(69),
            ..header
        };
        assert!(late.encode(&cursor_layout, STAMP).is_err());
        assert!(
            Header {
                tokens: 0,
                ..header
            }
            .encode(&owner_layout, STAMP)
            .is_err()
        );
    }

    #[test]
    fn layouts_hold_the_tokens_once_at_the_declared_size() {
        let state = synthetic();
        assert_eq!(HeaderLayout::V1.check(&state), Ok(()));
        let short = RequestState {
            header_bytes: 32,
            ..synthetic()
        };
        assert!(HeaderLayout::V1.check(&short).is_err());
        let twice = HeaderLayout {
            magic: *b"TESTTWO1",
            words: &[
                Word::Tokens,
                Word::Owner,
                Word::Owner,
                Word::Zero,
                Word::Zero,
                Word::Zero,
                Word::Zero,
            ],
        };
        assert!(twice.check(&state).is_err());
        let none = HeaderLayout {
            magic: *b"TESTNON1",
            words: &[Word::Zero; 7],
        };
        assert!(none.check(&state).is_err());
    }

    #[test]
    fn the_digest_follows_the_captured_layout_only() {
        let state = synthetic();
        let digest = layout_digest(&state);
        let mut uncaptured = synthetic();
        uncaptured.buffers[3].planes = vec![32];
        uncaptured.buffers[0].entries = crate::state::Entries::Fixed(3);
        uncaptured.page_tokens = 8;
        assert_eq!(layout_digest(&uncaptured), digest);
        let changes: [fn(&mut RequestState); 5] = [
            |state| state.buffers[0].planes.swap(0, 1),
            |state| state.buffers[1].layers = 3,
            |state| state.buffers[2].capture = Capture::LastRows { rows: 5 },
            |state| state.buffers[2].name = "rows",
            |state| state.header_bytes = 32,
        ];
        for change in changes {
            let mut changed = synthetic();
            change(&mut changed);
            assert_ne!(layout_digest(&changed), digest);
        }
    }

    /// Two rings of the same bytes and capture, one over eight positions
    /// and one over sixteen, address a position at different cells.
    #[test]
    fn the_digest_follows_the_physical_window() {
        let narrow = synthetic();
        let mut wide = synthetic();
        wide.buffers[0].scope = Scope::WindowRing { tokens: 16 };
        assert_eq!((narrow.validate(), wide.validate()), (Ok(()), Ok(())));
        assert_eq!(narrow.snapshot_bytes(), wide.snapshot_bytes());
        assert_eq!(narrow.buffers[0].capture, wide.buffers[0].capture);
        assert_ne!(layout_digest(&narrow), layout_digest(&wide));
        // So do a history's rows.
        let mut rows = synthetic();
        rows.buffers[2].scope = Scope::Rows { rows: 12 };
        assert_eq!(rows.validate(), Ok(()));
        assert_ne!(layout_digest(&narrow), layout_digest(&rows));
    }

    #[test]
    fn a_lagging_frontier_restores_only_into_its_owner() {
        let state = synthetic();
        let header = |tokens, frontier| Header {
            tokens,
            frontier,
            owner: Some(7),
            cursor: None,
        };
        // Four-token pages: a frontier in the last page restores anywhere.
        assert!(may_restore(&state, &header(10, Some(8)), 3));
        assert!(may_restore(&state, &header(9, Some(8)), 3));
        assert!(!may_restore(&state, &header(10, Some(7)), 3));
        assert!(may_restore(&state, &header(10, Some(7)), 7));
        // Frontier-owned state needs a frontier, and other state none.
        assert!(!may_restore(&state, &header(10, None), 7));
        let mut shared = synthetic();
        shared.buffers[2].owner = Owner::Shared;
        assert!(may_restore(&shared, &header(10, None), 3));
        assert!(!may_restore(&shared, &header(10, Some(9)), 7));
    }

    #[derive(Debug)]
    struct Fake {
        collectives: Collectives,
        calls: Vec<(u64, Option<u64>)>,
    }

    impl Live for Fake {
        fn collectives(&self) -> &Collectives {
            &self.collectives
        }
    }

    impl PrefixInstance for Fake {
        fn state_pages(&self) -> Result<u64, BundleStatusV1> {
            Ok(6)
        }

        fn capture(&mut self, request: &PrefixStateRequestV1) -> Result<(), BundleStatusV1> {
            self.calls.push((request.request_id, None));
            Ok(())
        }

        fn restore(
            &mut self,
            request: &PrefixStateRequestV1,
            slot: Option<u64>,
        ) -> Result<(), BundleStatusV1> {
            self.calls.push((request.request_id, slot));
            slot.map(|_| ()).ok_or(BundleStatusV1::Unsupported)
        }
    }

    #[test]
    fn entry_bodies_refuse_a_poisoned_instance() {
        let instances = RefCell::new(InstanceTable::new());
        let handle = instances
            .borrow_mut()
            .insert(Box::new(Fake {
                collectives: Collectives::default(),
                calls: Vec::new(),
            }))
            .unwrap();
        let request = |request_id| PrefixStateRequestV1 {
            request_id,
            tokens: 4,
            token_ids: Vec::new(),
            pages: vec![1],
        };
        let run = || {
            [
                describe(&instances, handle).status,
                capture(&instances, handle, request(1)),
                restore(&instances, handle, request(2)),
                restore_in_slot(&instances, handle, request(3), 5),
            ]
        };
        assert_eq!(
            run(),
            [
                BundleStatusV1::Success,
                BundleStatusV1::Success,
                BundleStatusV1::Unsupported,
                BundleStatusV1::Success,
            ]
        );
        assert_eq!(describe(&instances, handle).state_pages, 6);
        instances
            .borrow_mut()
            .get_mut(handle)
            .unwrap()
            .collectives
            .poison();
        assert_eq!(run(), [BundleStatusV1::InvalidState; 4]);
        let instance = &mut instances.borrow_mut();
        let calls = &instance.get_mut(handle).unwrap().calls;
        assert_eq!(*calls, [(1, None), (2, None), (3, Some(5))]);
    }
}
