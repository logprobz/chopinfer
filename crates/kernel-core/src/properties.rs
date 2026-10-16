//! Exactness properties: which bitwise equalities an op's rows keep across
//! chunk splits, launch sizes, batch neighbours and ties.
//!
//! A model requests the properties it relies on (verify rows equal decode
//! rows, prefill chunk granules, tie rules), and an implementation declares
//! the ones it guarantees. The vocabulary is model math: it names no kernel,
//! target or catalog, so every backend states its guarantees in it.

/// How a prefill entry's per-row bits depend on how a prompt is split into
/// chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Chunk {
    /// A row has the same bits in every chunk that holds it.
    Any,
    /// A row has the same bits in every split whose chunks start at
    /// absolute positions (counted from the sequence's first token) that
    /// are multiples of this many rows, for a prompt whose first call starts
    /// on such a multiple (position 0 or an anchor there). A call whose
    /// first row starts off that grid gets no chunk claim from `Granule`:
    /// a kernel may chunk from the call's first row or at absolute
    /// multiples, and the two agree only on the grid.
    Granule(u32),
    /// No claim.
    None,
}

/// How the rows of one launch relate to the same rows launched alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Rows {
    /// A launch of up to `max` rows gives every row the bits of that row
    /// launched alone through the operation's one-row entry.
    OneRow { max: u32 },
    /// No claim.
    None,
}

/// Whether a row's bits depend on the other rows or items of its launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Batch {
    /// A row's bits do not depend on the other rows' values. Where a kernel
    /// reads along a sequence (attention, recurrent state, convolutions,
    /// n-grams), the other rows are those of the other sequences.
    Independent,
    /// No claim.
    None,
}

/// The tie rule of a selection entry (top-k and argmax).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Ties {
    /// Equal values select the lower index first.
    LowerIndex,
    /// The entry selects nothing.
    NotApplicable,
}

/// Whether a prefill entry that writes per-slot state (a recurrent state, a
/// window ring, a raw-key ring) can also leave that state at in-step anchor
/// rows, which exact prefix capture needs inside a step.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Anchors {
    /// The entry writes no per-slot state (paged KV is captured by page).
    NotApplicable,
    /// The entry writes per-slot state only at the end of its rows.
    None,
    /// The entry also writes the state at each anchor row into a spare
    /// entry.
    Spare,
    /// The entry writes the state at an anchor row into a spare entry
    /// completely only when the anchor lies at least `rows` rows after the
    /// first row its launch processes of that sequence: the spare entry is
    /// taken from the launch's own rows (a window of `rows` positions before
    /// the anchor), so a nearer anchor leaves part of it stale. A need of
    /// this kind promises that every anchor it asks for lies at least
    /// `rows` rows after its launch's first row, as anchors on multiples of
    /// `rows` positions in launches that start on one (or at an anchor) do.
    SpareAfter { rows: u32 },
    /// A model's own launch after the entry derives the anchor rows' state
    /// from the entry's inputs and staged state (an anchor launch after a
    /// convolution).
    Companion,
    /// The entry's per-slot state at an anchor on a multiple of `granule`
    /// positions of its sequence is empty: the state holds only the rows
    /// past the last such multiple (an incomplete pool of `granule` rows),
    /// so capture at such an anchor needs no spare and no state, and a
    /// later chunk starting there reads none. Anchors elsewhere are not
    /// supported. A need of this kind promises that every anchor it asks
    /// for lies on a multiple of `granule` positions; it is granted only by
    /// an entry whose granule divides that alignment (or one that keeps a
    /// spare entry at every anchor). A zero granule states no alignment.
    Aligned { granule: u32 },
}

impl Anchors {
    /// Whether exact prefix capture can take this entry's state at every
    /// in-step anchor row.
    pub fn supported(self) -> bool {
        matches!(
            self,
            Anchors::NotApplicable | Anchors::Spare | Anchors::Companion
        )
    }

    /// Whether exact prefix capture can take this entry's state at in-step
    /// anchor rows that lie at least `rows` rows after their launch's first
    /// row.
    pub fn supported_after(self, rows: u32) -> bool {
        match self {
            Anchors::SpareAfter { rows: span } => span <= rows,
            anchors => anchors.supported(),
        }
    }

    /// Whether exact prefix capture can take this entry's state at in-step
    /// anchors on multiples of `granule` positions that lie at least
    /// `granule` rows after their launch's first row, as a served program's
    /// anchors do ([`Anchors::supported_after`], and an aligned entry whose
    /// granule divides `granule`).
    pub fn supported_at(self, granule: u32) -> bool {
        match self {
            Anchors::Aligned { granule: own } => {
                own != 0 && granule != 0 && granule.is_multiple_of(own)
            }
            anchors => anchors.supported_after(granule),
        }
    }

    /// Whether the entry keeps a spare entry at anchors, at any span or
    /// past one: a property its tests must prove.
    pub fn spare(self) -> bool {
        matches!(self, Anchors::Spare | Anchors::SpareAfter { .. })
    }

    /// Whether the entry claims an anchor property its tests must prove: a
    /// spare entry ([`Anchors::spare`]), or empty state at aligned anchors.
    pub fn claimed(self) -> bool {
        self.spare() || matches!(self, Anchors::Aligned { .. })
    }
}

/// What an entry's chunk, rows and batch claims are about.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum Scope {
    /// The entry's own row-valued outputs.
    Entry,
    /// The token rows of the routed operation the entry belongs to (router,
    /// route metadata, experts and combine). The entry itself writes route
    /// metadata (expert histograms, offsets, sorted rows), which depends on
    /// every routed row of the launch and on the chunk split: no consumer
    /// may reuse it beside other rows.
    RoutedOperation,
}

/// An entry's declared exactness properties.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct Invariance {
    pub chunk: Chunk,
    pub rows: Rows,
    pub batch: Batch,
    pub ties: Ties,
    pub anchors: Anchors,
    pub scope: Scope,
}

impl Invariance {
    /// No claims.
    pub const NONE: Invariance = Invariance {
        chunk: Chunk::None,
        rows: Rows::None,
        batch: Batch::None,
        ties: Ties::NotApplicable,
        anchors: Anchors::NotApplicable,
        scope: Scope::Entry,
    };

    /// A prefill entry whose rows ignore the chunk split and each other.
    pub const PREFILL: Invariance = Invariance {
        chunk: Chunk::Any,
        batch: Batch::Independent,
        ..Invariance::NONE
    };

    /// An entry that moves rows or state without arithmetic: every claim
    /// holds trivially.
    pub const COPY: Invariance = Invariance::any_rows(u32::MAX);

    /// An entry of at most `max` rows whose rows are one-row exact and, with
    /// more than one row, independent of each other.
    pub const fn one_row(max: u32) -> Invariance {
        Invariance {
            rows: Rows::OneRow { max },
            batch: if max > 1 {
                Batch::Independent
            } else {
                Batch::None
            },
            ..Invariance::NONE
        }
    }

    /// An entry of at most `max` rows that also serves prefill rows: one-row
    /// exact, so any chunk split gives a row the same bits.
    pub const fn any_rows(max: u32) -> Invariance {
        Invariance {
            chunk: Chunk::Any,
            ..Invariance::one_row(max)
        }
    }

    pub const fn with_ties(self, ties: Ties) -> Invariance {
        Invariance { ties, ..self }
    }

    pub const fn with_anchors(self, anchors: Anchors) -> Invariance {
        Invariance { anchors, ..self }
    }

    pub const fn with_chunk(self, chunk: Chunk) -> Invariance {
        Invariance { chunk, ..self }
    }

    /// These claims for the routed operation an entry of route metadata
    /// belongs to (`Scope::RoutedOperation`).
    pub const fn of_routed_operation(self) -> Invariance {
        Invariance {
            scope: Scope::RoutedOperation,
            ..self
        }
    }

    /// The properties this declaration claims, which tests must prove.
    pub fn claims(self) -> Vec<Property> {
        let mut claims = Vec::new();
        if self.chunk != Chunk::None {
            claims.push(Property::Chunk);
        }
        if self.rows != Rows::None {
            claims.push(Property::Rows);
        }
        if self.batch != Batch::None {
            claims.push(Property::Batch);
        }
        if self.ties != Ties::NotApplicable {
            claims.push(Property::Ties);
        }
        if self.anchors.claimed() {
            claims.push(Property::Anchors);
        }
        claims
    }

    /// Whether prefill chunks that start at multiples of `granule` rows keep
    /// every row's bits. A zero granule, declared or asked for, keeps no
    /// split, as in [`Invariance::meet`].
    pub fn chunks_at(self, granule: u32) -> bool {
        granule != 0
            && match self.chunk {
                Chunk::Any => true,
                Chunk::Granule(own) => own != 0 && granule.is_multiple_of(own),
                Chunk::None => false,
            }
    }

    /// Whether a launch of `rows` rows gives each row its one-row bits.
    pub fn one_row_at(self, rows: u32) -> bool {
        matches!(self.rows, Rows::OneRow { max } if max >= rows)
    }

    /// Whether these guarantees cover every claim of `need`, the exactness
    /// a model relies on: a chunk granule the guarantee keeps (none keeps a
    /// zero granule, [`Invariance::chunks_at`]), one-row bits up to the
    /// needed rows, batch independence, the needed tie rule, and anchors of
    /// the needed kind (an op without per-slot state meets any; a spare
    /// entry at any span meets a need past a span, and a spare entry past
    /// a span meets a need past at least that span; an aligned need is met
    /// by a granule dividing its alignment or a spare entry at any span,
    /// and a need of no alignment by none). A need
    /// with claims must be about the same scope. `NONE` needs nothing.
    pub fn provides(self, need: Invariance) -> bool {
        let chunk = match need.chunk {
            Chunk::None => true,
            Chunk::Any => self.chunk == Chunk::Any,
            Chunk::Granule(granule) => self.chunks_at(granule),
        };
        let rows = match need.rows {
            Rows::None => true,
            Rows::OneRow { max } => self.one_row_at(max),
        };
        let batch = need.batch == Batch::None || self.batch == Batch::Independent;
        let ties = need.ties == Ties::NotApplicable || self.ties == need.ties;
        let anchors = match (need.anchors, self.anchors) {
            (Anchors::NotApplicable | Anchors::None, _) => true,
            (_, Anchors::NotApplicable) => true,
            (Anchors::SpareAfter { .. }, Anchors::Spare) => true,
            (Anchors::SpareAfter { rows: wanted }, Anchors::SpareAfter { rows: own }) => {
                own <= wanted
            }
            // An aligned need is granted by a granule dividing its
            // alignment, or a spare entry at every anchor; one that states
            // no alignment is refused.
            (Anchors::Aligned { granule: wanted }, Anchors::Aligned { granule: own }) => {
                wanted != 0 && own != 0 && wanted.is_multiple_of(own)
            }
            (Anchors::Aligned { granule: wanted }, Anchors::Spare) => wanted != 0,
            (wanted, own) => own == wanted,
        };
        let scope = need.claims().is_empty() || self.scope == need.scope;
        chunk && rows && batch && ties && anchors && scope
    }

    /// The strongest declaration that both `self` and `other` provide: it
    /// provides every need both provide. An op that launches either entry
    /// for all its rows claims this. An op that launches one at some row
    /// counts and the other at others claims less, since its chunk and
    /// rows claims relate launches of different row counts (`routed`).
    /// Chunk granules meet at their least common multiple (a
    /// row split at multiples of both keeps its bits); when that exceeds
    /// `u32`, no granule is left to claim. Declarations about different
    /// scopes share no claim.
    pub fn meet(self, other: Invariance) -> Invariance {
        let anchors = match (self.anchors, other.anchors) {
            (Anchors::NotApplicable, anchors) | (anchors, Anchors::NotApplicable) => anchors,
            // Aligned anchors meet at the least common multiple of their
            // granules; a zero or overflowing granule grants no anchor.
            (Anchors::Aligned { granule: x }, Anchors::Aligned { granule: y }) => match (x, y) {
                (0, _) | (_, 0) => Anchors::None,
                _ => lcm(x, y).map_or(Anchors::None, |granule| Anchors::Aligned { granule }),
            },
            (Anchors::Spare, Anchors::Aligned { granule })
            | (Anchors::Aligned { granule }, Anchors::Spare) => match granule {
                0 => Anchors::None,
                granule => Anchors::Aligned { granule },
            },
            (x, y) if x == y => x,
            (Anchors::Spare, Anchors::SpareAfter { rows })
            | (Anchors::SpareAfter { rows }, Anchors::Spare) => Anchors::SpareAfter { rows },
            (Anchors::SpareAfter { rows: x }, Anchors::SpareAfter { rows: y }) => {
                Anchors::SpareAfter { rows: x.max(y) }
            }
            _ => Anchors::None,
        };
        // A zero granule states no alignment, so it grants no anchor need,
        // not even its own.
        let anchors = match anchors {
            Anchors::Aligned { granule: 0 } => Anchors::None,
            anchors => anchors,
        };
        if self.scope != other.scope {
            let anchors = if anchors.claimed() {
                Anchors::None
            } else {
                anchors
            };
            return Invariance {
                anchors,
                ..Invariance::NONE
            };
        }
        // A zero granule keeps no split, not even its own.
        let chunk = match (self.chunk, other.chunk) {
            (Chunk::Granule(0), _) | (_, Chunk::Granule(0)) => Chunk::None,
            (Chunk::Any, chunk) | (chunk, Chunk::Any) => chunk,
            (Chunk::Granule(x), Chunk::Granule(y)) => lcm(x, y).map_or(Chunk::None, Chunk::Granule),
            _ => Chunk::None,
        };
        let rows = match (self.rows, other.rows) {
            (Rows::OneRow { max: x }, Rows::OneRow { max: y }) => Rows::OneRow { max: x.min(y) },
            _ => Rows::None,
        };
        let batch = if self.batch == Batch::Independent && other.batch == Batch::Independent {
            Batch::Independent
        } else {
            Batch::None
        };
        let ties = if self.ties == other.ties {
            self.ties
        } else {
            Ties::NotApplicable
        };
        Invariance {
            chunk,
            rows,
            batch,
            ties,
            anchors,
            scope: self.scope,
        }
    }
}

/// The entry an op launches at one row count, for an op that sends each
/// row count to an entry of its own (`Invariance::routed`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowRoute<R> {
    /// The rows of the launch.
    pub rows: u32,
    /// The launched entry's declaration.
    pub invariance: Invariance,
    /// The one-row entry the declaration's rows claim refers to: the
    /// entry itself, unless its declaration names another entry whose
    /// one-row launch gives the same bits.
    pub reference: R,
}

impl Invariance {
    /// What an op claims that launches `routes[i]` at `routes[i].rows`
    /// rows, the routes covering every row count it serves from one up:
    /// the meet of their declarations, except for the claims that relate
    /// launches of different row counts, rows and chunk. Its rows claim
    /// holds only up to the first route that refers to another one-row
    /// entry than the one-row route or that follows a gap in the row
    /// counts. If every route claims the rows it launches and refers to
    /// that one-row entry, the op's rows also keep their one-row bits at
    /// every row count it serves: a row of any launch has the bits of that
    /// row alone through the shared reference, which is the op's own
    /// one-row launch. A single route keeps its chunk claim; more routes
    /// claim no chunk. One-row bits cover a launch's row outputs, not the
    /// state it hands on to the next chunk, so entries whose rows agree
    /// may still leave a later chunk different state.
    pub fn routed<R: PartialEq>(routes: &[RowRoute<R>]) -> Invariance {
        let Some(first) = routes.first() else {
            return Invariance::NONE;
        };
        let met = routes[1..]
            .iter()
            .fold(first.invariance, |held, route| held.meet(route.invariance));
        // The leading routes that launch row counts 1, 2, ... with the
        // one-row route's reference.
        let agreeing = routes
            .iter()
            .zip(1..)
            .take_while(|(route, rows)| route.rows == *rows && route.reference == first.reference)
            .count();
        // Declarations about different scopes share no claim (`meet`).
        let scoped = routes
            .iter()
            .all(|route| route.invariance.scope == first.invariance.scope);
        let composed = scoped
            && routes[..agreeing]
                .iter()
                .all(|route| route.invariance.one_row_at(route.rows));
        // Past the agreeing routes, the meet's claim holds up to them;
        // with every route agreeing, wherever it reaches.
        let bound = if agreeing == routes.len() {
            u32::MAX
        } else {
            agreeing as u32
        };
        let met_rows = match met.rows {
            Rows::OneRow { max } => max.min(bound),
            Rows::None => 0,
        };
        let shared = scoped && agreeing == routes.len();
        let served = if composed && shared {
            routes.len() as u32
        } else {
            0
        };
        Invariance {
            rows: match met_rows.max(served) {
                0 => Rows::None,
                max => Rows::OneRow { max },
            },
            chunk: if routes.len() == 1 {
                met.chunk
            } else {
                Chunk::None
            },
            ..met
        }
    }
}

/// The least common multiple of two non-zero granules, if it fits.
fn lcm(x: u32, y: u32) -> Option<u32> {
    let (mut a, mut b) = (x, y);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    (x / a).checked_mul(y)
}

/// A property a declaration can claim; each has one kernel-lab harness.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum Property {
    Chunk,
    Rows,
    Batch,
    Ties,
    Anchors,
}

/// The verify steps a served program promises exact rows for: each row of
/// such a step has the bits of the same row decoded alone, one step at a
/// time (AR).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactVerify {
    /// Most rows of a qualifying verify step.
    pub rows: u32,
    /// Most items of a qualifying verify step; AR itself depends on its
    /// batch, so the guarantee is for requests served one at a time.
    pub items: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guarantees_cover_only_the_claims_they_imply() {
        let none = Invariance::NONE;
        assert!(none.provides(Invariance::NONE));
        assert!(!none.provides(Invariance::PREFILL));
        assert!(!none.provides(Invariance::one_row(1)));
        // A copy keeps every claim.
        for need in [
            Invariance::PREFILL,
            Invariance::one_row(8),
            Invariance::any_rows(64),
            Invariance::PREFILL.with_chunk(Chunk::Granule(64)),
        ] {
            assert!(Invariance::COPY.provides(need), "{need:?}");
        }
        // Chunk granules: a guarantee at 64 keeps any multiple of 64.
        let granule = |n| Invariance::PREFILL.with_chunk(Chunk::Granule(n));
        assert!(granule(64).provides(granule(128)));
        assert!(!granule(64).provides(granule(32)));
        assert!(!granule(64).provides(Invariance::PREFILL));
        assert!(Invariance::PREFILL.provides(granule(64)));
        // One-row bits up to the needed rows.
        assert!(Invariance::one_row(8).provides(Invariance::one_row(4)));
        assert!(!Invariance::one_row(4).provides(Invariance::one_row(8)));
        // Batch independence and ties.
        let independent = Invariance {
            batch: Batch::Independent,
            ..Invariance::NONE
        };
        assert!(Invariance::one_row(2).provides(independent));
        assert!(!Invariance::one_row(1).provides(independent));
        let lower = Invariance::one_row(4).with_ties(Ties::LowerIndex);
        assert!(lower.provides(lower));
        assert!(!Invariance::one_row(4).provides(lower));
        // Anchors: a spare need takes a spare guarantee or no per-slot state.
        let spare = Invariance::PREFILL.with_anchors(Anchors::Spare);
        assert!(spare.provides(spare));
        assert!(Invariance::PREFILL.provides(spare));
        assert!(
            !Invariance::PREFILL
                .with_anchors(Anchors::None)
                .provides(spare)
        );
        // A spare entry past a span meets needs past at least that span,
        // never a need at any span; a spare entry at any span meets both.
        let after = |rows| Invariance::NONE.with_anchors(Anchors::SpareAfter { rows });
        assert!(after(16).provides(after(16)));
        assert!(after(16).provides(after(64)));
        assert!(!after(16).provides(after(4)));
        assert!(!after(16).provides(Invariance::NONE.with_anchors(Anchors::Spare)));
        assert!(!after(16).provides(spare));
        assert!(spare.provides(after(4)));
        assert!(Invariance::NONE.provides(after(4)));
        assert!(
            !Invariance::NONE
                .with_anchors(Anchors::Companion)
                .provides(after(4))
        );
        assert!(Anchors::SpareAfter { rows: 16 }.supported_after(64));
        assert!(!Anchors::SpareAfter { rows: 16 }.supported_after(4));
        assert!(!Anchors::SpareAfter { rows: 16 }.supported());
        assert!(Anchors::Spare.supported_after(1));
        assert_eq!(after(16).claims(), [Property::Anchors]);
        // Claims about a routed operation are not claims about one entry.
        assert!(!Invariance::PREFILL.provides(Invariance::PREFILL.of_routed_operation()));
        assert!(
            Invariance::PREFILL
                .of_routed_operation()
                .provides(Invariance::PREFILL.of_routed_operation())
        );
    }

    /// Declarations over every field's distinct cases, granules with common
    /// and coprime factors among them.
    fn declarations() -> Vec<Invariance> {
        let chunks = [
            Chunk::Any,
            Chunk::None,
            Chunk::Granule(0),
            Chunk::Granule(2),
            Chunk::Granule(3),
            Chunk::Granule(4),
            Chunk::Granule(6),
            Chunk::Granule(12),
            // Coprime, so their least common multiple overflows.
            Chunk::Granule(u32::MAX),
            Chunk::Granule(u32::MAX - 1),
        ];
        let rows = [Rows::None, Rows::OneRow { max: 1 }, Rows::OneRow { max: 8 }];
        let mut all = Vec::new();
        for chunk in chunks {
            for rows in rows {
                for batch in [Batch::Independent, Batch::None] {
                    for ties in [Ties::LowerIndex, Ties::NotApplicable] {
                        for anchors in [
                            Anchors::NotApplicable,
                            Anchors::None,
                            Anchors::Spare,
                            Anchors::SpareAfter { rows: 4 },
                            Anchors::SpareAfter { rows: 16 },
                            Anchors::Companion,
                            Anchors::Aligned { granule: 4 },
                            Anchors::Aligned { granule: 6 },
                        ] {
                            for scope in [Scope::Entry, Scope::RoutedOperation] {
                                all.push(Invariance {
                                    chunk,
                                    rows,
                                    batch,
                                    ties,
                                    anchors,
                                    scope,
                                });
                            }
                        }
                    }
                }
            }
        }
        all
    }

    #[test]
    fn the_meet_is_the_strongest_declaration_both_provide() {
        let all = declarations();
        // Every pair would take minutes in a debug build; a fixed stride
        // walks pairs from every part of the list.
        let pairs = (0..all.len() * all.len())
            .step_by(97)
            .map(|i| (all[i / all.len()], all[i % all.len()]));
        for (a, b) in pairs {
            let meet = a.meet(b);
            assert_eq!(meet, b.meet(a), "{a:?} {b:?}");
            assert!(a.provides(meet) && b.provides(meet), "{a:?} {b:?}");
            for need in &all {
                if a.provides(*need) && b.provides(*need) {
                    assert!(meet.provides(*need), "{a:?} {b:?} {need:?}");
                }
            }
        }
    }

    /// Every pair of anchor claims, spans among them: the meet is provided
    /// by both and provides every need both provide, in either scope.
    #[test]
    fn anchor_meets_are_the_strongest_both_provide() {
        let anchors = [
            Anchors::NotApplicable,
            Anchors::None,
            Anchors::Spare,
            Anchors::SpareAfter { rows: 0 },
            Anchors::SpareAfter { rows: 4 },
            Anchors::SpareAfter { rows: 16 },
            Anchors::Companion,
            Anchors::Aligned { granule: 0 },
            Anchors::Aligned { granule: 4 },
            Anchors::Aligned { granule: 6 },
            Anchors::Aligned { granule: 12 },
            Anchors::Aligned { granule: 64 },
            Anchors::Aligned { granule: u32::MAX },
            Anchors::Aligned {
                granule: u32::MAX - 1,
            },
        ];
        let all = anchors
            .iter()
            .flat_map(|&a| {
                [Scope::Entry, Scope::RoutedOperation].map(|scope| Invariance {
                    scope,
                    ..Invariance::NONE.with_anchors(a)
                })
            })
            .collect::<Vec<_>>();
        for &a in &all {
            for &b in &all {
                let meet = a.meet(b);
                assert_eq!(meet, b.meet(a), "{a:?} {b:?}");
                assert!(a.provides(meet) && b.provides(meet), "{a:?} {b:?}");
                for &need in &all {
                    if a.provides(need) && b.provides(need) {
                        assert!(meet.provides(need), "{a:?} {b:?} {need:?}");
                    }
                }
            }
        }
    }

    /// An aligned need states the alignment of every anchor it asks for:
    /// an entry whose state is empty at multiples of its granule grants it
    /// when that granule divides the alignment, and refuses a misaligned
    /// need, a need of no alignment and every unaligned anchor need. The
    /// served program's check takes the same rule at its granule.
    #[test]
    fn aligned_anchors_grant_only_the_alignment_they_divide() {
        let aligned = |granule| Invariance::PREFILL.with_anchors(Anchors::Aligned { granule });
        let pooled = aligned(4);
        assert_eq!(
            pooled.claims(),
            [Property::Chunk, Property::Batch, Property::Anchors]
        );
        // Granted: an alignment of a multiple of the granule.
        assert!(pooled.provides(aligned(4)));
        assert!(pooled.provides(aligned(64)));
        // Refused: a misaligned need, and a need of no alignment.
        assert!(!pooled.provides(aligned(6)));
        assert!(!pooled.provides(aligned(2)));
        assert!(!pooled.provides(aligned(0)));
        assert!(!aligned(0).provides(aligned(0)));
        // Unstated: needs at any anchor row, past a span, or by a
        // companion state no alignment.
        for anchors in [
            Anchors::Spare,
            Anchors::SpareAfter { rows: 64 },
            Anchors::Companion,
        ] {
            assert!(
                !pooled.provides(Invariance::PREFILL.with_anchors(anchors)),
                "{anchors:?}"
            );
        }
        // Needs without anchors stay granted, and aligned needs are granted
        // by a spare at every anchor and by an entry without per-slot state,
        // never by state kept only at the end or past a span.
        assert!(pooled.provides(Invariance::PREFILL));
        assert!(pooled.provides(Invariance::PREFILL.with_anchors(Anchors::None)));
        let spare = Invariance::PREFILL.with_anchors(Anchors::Spare);
        assert!(spare.provides(aligned(64)));
        assert!(!spare.provides(aligned(0)));
        assert!(Invariance::PREFILL.provides(aligned(64)));
        assert!(
            !Invariance::PREFILL
                .with_anchors(Anchors::None)
                .provides(aligned(64))
        );
        let after = Invariance::PREFILL.with_anchors(Anchors::SpareAfter { rows: 4 });
        assert!(!after.provides(aligned(64)));
        // Meets keep the common alignment.
        assert_eq!(pooled.meet(aligned(6)), aligned(12));
        assert_eq!(pooled.meet(spare), pooled);
        assert_eq!(pooled.meet(aligned(0)).anchors, Anchors::None);
        assert_eq!(pooled.meet(after).anchors, Anchors::None);
        // The served program's anchors sit on multiples of its granule.
        let at = |granule| Anchors::Aligned { granule: 4 }.supported_at(granule);
        assert!(at(64) && at(4));
        assert!(!at(6) && !at(2) && !at(0));
        assert!(!Anchors::Aligned { granule: 0 }.supported_at(64));
        assert!(!Anchors::Aligned { granule: 4 }.supported());
        assert!(!Anchors::Aligned { granule: 4 }.supported_after(64));
        // The other kinds keep their rule.
        for anchors in [
            Anchors::NotApplicable,
            Anchors::None,
            Anchors::Spare,
            Anchors::SpareAfter { rows: 16 },
            Anchors::SpareAfter { rows: 128 },
            Anchors::Companion,
        ] {
            assert_eq!(
                anchors.supported_at(64),
                anchors.supported_after(64),
                "{anchors:?}"
            );
        }
    }

    #[test]
    fn zero_and_overflowing_granules_keep_the_meet_the_strongest() {
        // Every pair of chunk claims, the boundary granules among them.
        let chunks = [
            Chunk::Any,
            Chunk::None,
            Chunk::Granule(0),
            Chunk::Granule(1),
            Chunk::Granule(2),
            Chunk::Granule(3),
            Chunk::Granule(65_536),
            Chunk::Granule(65_537),
            Chunk::Granule(u32::MAX - 1),
            Chunk::Granule(u32::MAX),
        ]
        .map(|chunk| Invariance::NONE.with_chunk(chunk));
        for a in chunks {
            for b in chunks {
                let meet = a.meet(b);
                assert!(a.provides(meet) && b.provides(meet), "{a:?} {b:?}");
                for need in chunks {
                    if a.provides(need) && b.provides(need) {
                        assert!(meet.provides(need), "{a:?} {b:?} {need:?}");
                    }
                }
            }
        }
        // A zero granule keeps no split: no declaration meets a need of
        // it, and a declaration of it meets no chunk need.
        let zero = Invariance::NONE.with_chunk(Chunk::Granule(0));
        for declared in chunks {
            assert!(!declared.provides(zero), "{declared:?}");
            assert!(!declared.chunks_at(0), "{declared:?}");
        }
        assert!(!zero.provides(Invariance::NONE.with_chunk(Chunk::Granule(4))));
        assert!(zero.provides(Invariance::NONE));
        assert!(!Invariance::COPY.provides(zero));
    }

    /// Routes for row counts `rows`, each launching `invariance` with
    /// one-row reference `reference`.
    fn routes(
        rows: impl IntoIterator<Item = u32>,
        invariance: Invariance,
        reference: &'static str,
    ) -> Vec<RowRoute<&'static str>> {
        rows.into_iter()
            .map(|rows| RowRoute {
                rows,
                invariance,
                reference,
            })
            .collect()
    }

    #[test]
    fn routed_rows_compose_over_one_reference() {
        // A one-row entry declared up to two rows, then an entry whose
        // one-row bits are the first entry's.
        let low = Invariance::one_row(2);
        let high = Invariance::any_rows(32);
        let sound = [routes(1..=2, low, "a"), routes(3..=8, high, "a")].concat();
        let composed = Invariance::routed(&sound);
        assert_eq!(composed, Invariance::one_row(8));
        assert!(composed.provides(Invariance::one_row(8)));
        assert!(!composed.provides(Invariance::one_row(9)));
        // A use of the low rows alone keeps the declaration's claim.
        assert_eq!(Invariance::routed(&routes(1..=1, low, "a")), low);
        assert_eq!(Invariance::routed(&sound[..2]), low);

        // Another reference: the meet's claim stops before it.
        let other = [routes(1..=2, low, "a"), routes(3..=8, high, "b")].concat();
        assert_eq!(Invariance::routed(&other), low);
        assert!(!Invariance::routed(&other).provides(Invariance::one_row(4)));
        let wide = Invariance::one_row(8);
        let split = [routes(1..=1, wide, "a"), routes(2..=2, wide, "b")].concat();
        assert_eq!(Invariance::routed(&split).rows, Rows::OneRow { max: 1 });

        // A gap: row count 3 has no declared route.
        let gap = [routes(1..=2, low, "a"), routes(4..=8, high, "a")].concat();
        assert_eq!(Invariance::routed(&gap), low);
        let gap = [routes(1..=2, high, "a"), routes(4..=8, high, "a")].concat();
        assert_eq!(Invariance::routed(&gap).rows, Rows::OneRow { max: 2 });
        // Row counts out of order are a gap at the first one.
        let shuffled = [routes([2, 1], high, "a"), routes(3..=8, high, "a")].concat();
        assert_eq!(Invariance::routed(&shuffled).rows, Rows::None);

        // An entry launched at more rows than it claims.
        let short = Invariance::one_row(4);
        let over = [routes(1..=2, low, "a"), routes(3..=8, short, "a")].concat();
        assert_eq!(Invariance::routed(&over), low);
        assert_eq!(
            Invariance::routed(&routes(1..=8, short, "a")),
            Invariance::one_row(4)
        );

        // The claims about one launch are the meet's (batch independence
        // from both); several routes claim no chunk.
        assert_eq!(Invariance::routed(&sound).chunk, Chunk::None);
        assert_eq!(
            Invariance::routed(&routes(1..=8, high, "a")),
            Invariance::one_row(32)
        );
        // Declarations about different scopes compose no rows claim, as
        // they meet to none.
        let scopes = [
            routes(1..=1, low, "a"),
            routes(2..=2, low.of_routed_operation(), "a"),
        ]
        .concat();
        assert_eq!(low.meet(low.of_routed_operation()).rows, Rows::None);
        assert_eq!(Invariance::routed(&scopes), Invariance::NONE);
        let unclaimed = [
            routes(1..=2, low, "a"),
            routes(3..=8, Invariance::NONE, "a"),
        ]
        .concat();
        assert_eq!(Invariance::routed(&unclaimed), Invariance::NONE);
        assert_eq!(Invariance::routed::<&str>(&[]), Invariance::NONE);

        // Chunk claims relate launches of different row counts too: two
        // routes of different references keep no chunk claim, though each
        // declares one and the meet keeps it.
        let both = Invariance::any_rows(2);
        let split = [routes(1..=1, both, "a"), routes(2..=2, both, "b")].concat();
        assert_eq!(both.meet(both).chunk, Chunk::Any);
        let chunkless = Invariance::routed(&split);
        assert_eq!(chunkless.chunk, Chunk::None);
        // Batch independence is about one launch and stays.
        let one_launch = Invariance {
            rows: Rows::OneRow { max: 1 },
            batch: Batch::Independent,
            ..Invariance::NONE
        };
        assert_eq!(chunkless, one_launch);
        assert!(!chunkless.provides(Invariance::PREFILL));
        // Nor after a gap.
        let gap = [routes(1..=1, both, "a"), routes(3..=3, both, "a")].concat();
        assert_eq!(Invariance::routed(&gap).chunk, Chunk::None);
        // Nor over one reference with one-row bits at every row count.
        assert_eq!(
            Invariance::routed(&routes(1..=2, both, "a")),
            Invariance::one_row(2)
        );
        // Chunk claims over one reference without one-row bits do not
        // compose. Entry a emits the state it carries along the rows and
        // adds 1 per row, entry b adds 2: both declare any chunk and their
        // one-row launches agree (state 0), yet one two-row chunk through
        // b emits [0, 2] and two one-row chunks through a emit [0, 1].
        let prefill = [
            routes(1..=1, Invariance::PREFILL, "a"),
            routes(2..=2, Invariance::PREFILL, "a"),
        ]
        .concat();
        assert_eq!(
            Invariance::PREFILL.meet(Invariance::PREFILL).chunk,
            Chunk::Any
        );
        assert_eq!(Invariance::routed(&prefill).chunk, Chunk::None);
        assert!(!Invariance::routed(&prefill).provides(Invariance::PREFILL));
        let granule = Invariance::PREFILL.with_chunk(Chunk::Granule(2));
        let granules = [routes(1..=2, granule, "a"), routes(3..=4, granule, "a")].concat();
        assert_eq!(Invariance::routed(&granules).chunk, Chunk::None);
        // Nor do one-row bits, which cover row outputs and not the state
        // a launch hands on. Entries A, B and C carry a state s into each
        // launch. A and B advance it by one per row and C hands it on
        // unchanged; at row i of a launch A emits s + i, B emits
        // s + 2 * floor(i / 2) and C emits s. A row alone emits s through
        // each, so routes of 1, 2 and 3 rows to A, B and C declare one-row
        // bits up to those rows over one reference, and a granule of 2.
        // Yet three rows from state 0 emit [0, 0, 0] whole (C) and, split
        // at row 2, [0, 0] (B) and then [2] (A from state 2).
        let granule_of = |rows| Invariance::one_row(rows).with_chunk(Chunk::Granule(2));
        let three = [
            routes(1..=1, granule_of(1), "a"),
            routes(2..=2, granule_of(2), "a"),
            routes(3..=3, granule_of(3), "a"),
        ]
        .concat();
        let met = granule_of(1).meet(granule_of(2)).meet(granule_of(3));
        assert_eq!(met.chunk, Chunk::Granule(2));
        let rows_only = Invariance::routed(&three);
        assert_eq!(rows_only.rows, Rows::OneRow { max: 3 });
        assert_eq!(rows_only.chunk, Chunk::None);
        assert!(!rows_only.provides(Invariance::PREFILL.with_chunk(Chunk::Granule(2))));
        // A single route is one entry: its claim stands.
        assert_eq!(Invariance::routed(&routes(1..=1, granule, "a")), granule);
    }

    #[test]
    fn granules_meet_at_their_least_common_multiple() {
        let granule = |n| Invariance::NONE.with_chunk(Chunk::Granule(n));
        assert_eq!(granule(64).meet(granule(128)), granule(128));
        assert_eq!(granule(4).meet(granule(6)), granule(12));
        assert_eq!(
            granule(64).meet(Invariance::NONE.with_chunk(Chunk::Any)),
            granule(64)
        );
        // Both provide a need at 128, so their meet does.
        assert!(granule(64).meet(granule(128)).provides(granule(128)));
        // No granule past u32 can be claimed.
        assert_eq!(
            granule(u32::MAX).meet(granule(u32::MAX - 1)).chunk,
            Chunk::None
        );
        assert_eq!(granule(0).meet(granule(4)).chunk, Chunk::None);
    }
}
