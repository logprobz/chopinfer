//! The entry that serves an op, whatever order a model registered its
//! entries in.
//!
//! A family decides which registered entries match an op from the op's own
//! fields: its contract record and, where the entry choice depends on how
//! the step runs, its use. Of the matching entries, those that hold the
//! op's rows and whose declarations claim the exactness the op needs
//! remain; the least capacity among them wins, then the least rank the
//! family gives. Entries left that would launch the same are one entry.
//! Two left that would launch differently are a tie the op does not
//! resolve, so the op is refused, naming them in sorted order ([`sole`],
//! the choice for entries that hold no capacity of their own).
//!
//! Each family states what an entry launches as a key of the launch's own
//! parts: its symbols and the instantiation, plan and argument shape they
//! launch with. A declaration's claims are not part of the key; the
//! exactness filter reads them. Of entries that launch the same, the one
//! of least whole declaration stands for them, so the prepared op's claims
//! and implementation name do not depend on the order either.

use std::fmt::Debug;

use kernel_core::backend::{Need, Refused};
use kernel_core::properties::Invariance;

/// A registered entry that matches an op.
pub(super) struct Candidate<E> {
    entry: E,
    /// What refusals name the entry by, such as its symbol.
    name: String,
    /// What the entry launches: entries of equal launches are one.
    launches: String,
    /// The entry's whole declaration, which picks the one that stands for
    /// entries of equal launches.
    declaration: String,
    /// What the entry's declaration claims every launch keeps.
    guarantees: Invariance,
    /// The most rows the entry holds.
    capacity: u32,
    /// Among entries of the least capacity, the lower rank wins.
    rank: u64,
}

impl<E: Debug> Candidate<E> {
    /// `entry`, named `name`, which launches `launches` and claims
    /// `guarantees`. It holds any rows and has rank zero.
    pub(super) fn new(
        entry: E,
        name: impl Into<String>,
        launches: String,
        guarantees: Invariance,
    ) -> Self {
        Self {
            declaration: format!("{entry:?}"),
            entry,
            name: name.into(),
            launches,
            guarantees,
            capacity: u32::MAX,
            rank: 0,
        }
    }
}

impl<E> Candidate<E> {
    pub(super) fn capacity(self, capacity: u32) -> Self {
        Self { capacity, ..self }
    }

    pub(super) fn rank(self, rank: u64) -> Self {
        Self { rank, ..self }
    }
}

/// The entry [`choose`] picks and what the op may claim through it.
pub(super) struct Chosen<E> {
    pub(super) entry: E,
    /// The meet of what every left registration of the entry's launch
    /// declares: the op may launch any of them, so it claims no more than
    /// all of them do, whatever order they were registered in.
    pub(super) guarantees: Invariance,
}

/// The entry of `candidates`, the `family` entries that match an op, that
/// serves the op at `need`. Refuses when none matches, none holds the rows,
/// none declares the needed exactness, or entries that would launch
/// differently tie.
pub(super) fn choose<E>(
    family: &str,
    candidates: impl IntoIterator<Item = Candidate<E>>,
    need: Need,
) -> Result<Chosen<E>, Refused> {
    let mut found = candidates.into_iter().collect::<Vec<_>>();
    if found.is_empty() {
        return Err(Refused::Rule(format!(
            "no SM120 {family} entry serves the op"
        )));
    }
    // Every later step takes the first of a sorted list, so the choice
    // and the refusal do not depend on the registration order.
    found.sort_by(|a, b| {
        let key = |c: &Candidate<E>| (c.capacity, c.rank, c.name.clone(), c.launches.clone());
        (key(a), &a.declaration).cmp(&(key(b), &b.declaration))
    });
    found.retain(|candidate| need.capacity <= candidate.capacity);
    let Some(first) = found.first() else {
        return Err(Refused::Rule(format!(
            "{} rows exceed every SM120 {family} entry's capacity",
            need.capacity
        )));
    };
    let guaranteed = first.guarantees;
    found.retain(|candidate| candidate.guarantees.provides(need.exact));
    let Some(best) = found.first() else {
        return Err(Refused::Inexact {
            needed: need.exact,
            guaranteed,
        });
    };
    let (capacity, rank) = (best.capacity, best.rank);
    found.retain(|candidate| (candidate.capacity, candidate.rank) == (capacity, rank));
    // Of equal launches, the least declaration stands for them.
    found.sort_by(|a, b| (&a.launches, &a.declaration).cmp(&(&b.launches, &b.declaration)));
    let launches = found[0].launches.clone();
    let guarantees = found
        .iter()
        .filter(|candidate| candidate.launches == launches)
        .map(|candidate| candidate.guarantees)
        .reduce(Invariance::meet)
        .expect("a candidate is left");
    let left = found.into_iter().map(|candidate| Launches(candidate));
    let one = sole(left, |left| left.0.name.clone(), family)?;
    Ok(Chosen {
        entry: one.expect("a candidate is left").0.entry,
        guarantees,
    })
}

/// A candidate compared by what it launches.
struct Launches<E>(Candidate<E>);

impl<E> PartialEq for Launches<E> {
    fn eq(&self, other: &Self) -> bool {
        self.0.launches == other.0.launches
    }
}

/// The one entry among `found` (the selection's entries that serve an op),
/// whatever the order the selection registered them in: `choose` without a
/// capacity, rank or exactness to choose by. Identical entries count as
/// one; two that differ would launch differently or claim differently, and
/// the op does not say which to take, so they are refused, named by `name`
/// in sorted order. `None` when no entry serves the op.
pub(super) fn sole<T: PartialEq>(
    found: impl IntoIterator<Item = T>,
    name: impl Fn(&T) -> String,
    family: &str,
) -> Result<Option<T>, Refused> {
    let mut distinct: Vec<T> = Vec::new();
    for entry in found {
        if !distinct.contains(&entry) {
            distinct.push(entry);
        }
    }
    if distinct.len() > 1 {
        let mut names = distinct.iter().map(&name).collect::<Vec<_>>();
        names.sort();
        return Err(Refused::Rule(format!(
            "two SM120 {family} entries serve the op, {} and {}",
            names[0], names[1]
        )));
    }
    Ok(distinct.pop())
}

/// Every order of `entries`, for tests that register a case's entries in
/// each order.
/// The field names of a struct, listed exhaustively: the list fails to
/// compile until a field added to the struct is listed, so a launch key
/// guard ([`keys_follow_launches`]) classifies every field.
#[cfg(test)]
macro_rules! field_names {
    ($ty:path { $($field:ident),* $(,)? }) => {{
        #[allow(clippy::unneeded_field_pattern)]
        fn _exhaustive(entry: &$ty) {
            let $ty { $($field: _),* } = entry;
        }
        vec![$(stringify!($field)),*]
    }};
}
#[cfg(test)]
pub(super) use field_names;

/// How one field of a family's entry relates to its launches, for
/// [`keys_follow_launches`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Class {
    /// Read by a launch (its code, plan or arguments): the key changes, and
    /// so do the launches or the code the symbol runs.
    Launch,
    /// A limit a launch checks: the launches inside both limits are equal,
    /// and the key holds it, since past the lesser limit only one launches.
    Bound,
    /// Read by no launch (a profile, claims, a capacity the selection
    /// settles): the key, the launches and the code stay equal.
    Inert,
    /// Written into the generated source but read by no kernel: the key
    /// and the launches stay equal, the source text may not.
    Unread,
    /// Selected by the op or fixed by the entry's contract: the changed
    /// entry does not serve the op, so it never stands beside the base.
    Selects,
}

/// A field's name, class and a change of that field alone.
#[cfg(test)]
pub(super) type Field<E> = (&'static str, Class, Box<dyn Fn(&mut E)>);

/// Guards a family's hand-written launch key. `names` lists every field
/// of the entry type ([`field_names`]), and `fields` classifies each one
/// exactly once with a change of that field alone. `launched` prepares an
/// entry alone and returns the code its symbols run and its launches at
/// counts that reach every launch the op makes, every one successful (it
/// unwraps them); it returns the refusal when the entry does not serve the
/// op. Each change must behave as its class says.
#[cfg(test)]
pub(super) fn keys_follow_launches<E: Clone + Debug, L: PartialEq + Debug>(
    base: &E,
    names: Vec<&'static str>,
    fields: Vec<Field<E>>,
    key: impl Fn(&E) -> String,
    launched: impl Fn(&E) -> Result<(String, L), Refused>,
) {
    let (code, launches) = launched(base).expect("the base entry serves the op");
    let mut classified = Vec::new();
    let mut wrong = Vec::new();
    for (name, class, change) in fields {
        classified.push(name);
        let mut changed = base.clone();
        change(&mut changed);
        if format!("{changed:?}") == format!("{base:?}") {
            wrong.push(format!("{name}: the change leaves the entry as it was"));
            continue;
        }
        let outcome = launched(&changed);
        if class == Class::Selects {
            if outcome.is_ok() {
                wrong.push(format!("{name}: the changed entry still serves the op"));
            }
            continue;
        }
        let (changed_code, changed_launches) = match outcome {
            Ok(outcome) => outcome,
            Err(refused) => {
                wrong.push(format!(
                    "{name} ({class:?}): the changed entry is refused: {refused}"
                ));
                continue;
            }
        };
        let same_key = key(&changed) == key(base);
        let same_launches = changed_launches == launches;
        let same_code = changed_code == code;
        let same_launched = same_launches && same_code;
        let holds = match class {
            Class::Launch => !same_key && !same_launched,
            Class::Bound => !same_key && same_launched,
            Class::Inert => same_key && same_launched,
            Class::Unread => same_key && same_launches,
            Class::Selects => unreachable!(),
        };
        if !holds {
            wrong.push(format!(
                "{name} ({class:?}): same key {same_key}, same launches {same_launches}, \
                 same code {same_code}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    let mut listed = names;
    listed.sort_unstable();
    classified.sort_unstable();
    assert_eq!(classified, listed, "every field is classified exactly once");
}

#[cfg(test)]
pub(super) fn orders<T: Clone>(entries: &[T]) -> Vec<Vec<T>> {
    if entries.len() <= 1 {
        return vec![entries.to_vec()];
    }
    (0..entries.len())
        .flat_map(|first| {
            let mut rest = entries.to_vec();
            let head = rest.remove(first);
            orders(&rest).into_iter().map(move |mut order| {
                order.insert(0, head.clone());
                order
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, capacity: u32) -> Candidate<(String, u32)> {
        Candidate::new(
            (name.to_owned(), capacity),
            name,
            name.to_owned(),
            Invariance::NONE,
        )
        .capacity(capacity)
    }

    fn chosen(entries: &[(&str, u32)], need: Need) -> Vec<Result<(String, u32), Refused>> {
        orders(entries)
            .into_iter()
            .map(|order| {
                choose(
                    "test",
                    order.iter().map(|&(name, capacity)| entry(name, capacity)),
                    need,
                )
                .map(|chosen| chosen.entry)
            })
            .collect()
    }

    #[test]
    fn the_least_fitting_capacity_wins_in_every_order() {
        let entries = [("a", 32), ("b", 8), ("c", 16)];
        for result in chosen(&entries, Need::rows(9)) {
            assert_eq!(result.unwrap(), ("c".to_owned(), 16));
        }
        for result in chosen(&entries, Need::rows(33)) {
            assert!(matches!(result, Err(Refused::Rule(rule)) if rule.contains("exceed")));
        }
    }

    #[test]
    fn different_entries_of_one_capacity_are_refused_and_equal_ones_are_one() {
        for result in chosen(&[("b", 16), ("a", 16), ("c", 32)], Need::rows(4)) {
            assert!(matches!(
                result,
                Err(Refused::Rule(rule)) if rule.contains("entries serve the op, a and b")
            ));
        }
        for result in chosen(&[("a", 16), ("a", 16), ("c", 32)], Need::rows(4)) {
            assert_eq!(result.unwrap(), ("a".to_owned(), 16));
        }
        // A tie past the least capacity is not the op's.
        for result in chosen(&[("a", 8), ("b", 16), ("c", 16)], Need::rows(4)) {
            assert_eq!(result.unwrap(), ("a".to_owned(), 8));
        }
    }

    #[test]
    fn rank_and_exactness_decide_before_a_tie() {
        let exact = Invariance::one_row(16);
        let make = |name: &'static str, rank, guarantees| {
            Candidate::new(name, name, name.to_owned(), guarantees)
                .capacity(16)
                .rank(rank)
        };
        let entries = [("ordinary", 0, Invariance::NONE), ("exact", 1, exact)];
        for order in orders(&entries) {
            let candidates = || order.iter().map(|&(name, rank, g)| make(name, rank, g));
            assert_eq!(
                choose("test", candidates(), Need::rows(16)).map(|c| c.entry),
                Ok("ordinary")
            );
            let needed = Need::rows(16).with_exact(exact);
            assert_eq!(
                choose("test", candidates(), needed).map(|c| c.entry),
                Ok("exact")
            );
            let past = Need::rows(16).with_exact(Invariance::one_row(32));
            assert!(matches!(
                choose("test", candidates(), past),
                Err(Refused::Inexact { guaranteed, .. }) if guaranteed == Invariance::NONE
            ));
        }
        assert!(matches!(
            choose("test", Vec::<Candidate<&str>>::new(), Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120 test entry")
        ));
    }

    /// The exactness filter runs before ties are judged: an entry that
    /// cannot keep what the op needs is not the op's, so it ties with
    /// nothing.
    #[test]
    fn entries_that_cannot_keep_the_need_do_not_tie() {
        let exact = Invariance::one_row(16);
        let entries = [("claims", exact), ("plain", Invariance::NONE)];
        for order in orders(&entries) {
            let candidates = || {
                order
                    .iter()
                    .map(|&(name, g)| Candidate::new(name, name, name.to_owned(), g))
            };
            let needed = Need::rows(16).with_exact(exact);
            assert_eq!(
                choose("test", candidates(), needed).map(|c| c.entry),
                Ok("claims")
            );
            assert!(matches!(
                choose("test", candidates(), Need::rows(16)),
                Err(Refused::Rule(rule)) if rule.contains("claims and plain")
            ));
        }
    }

    /// Identity is the launch key alone: entries whose keys are equal are
    /// one entry, whatever else their declarations say.
    #[test]
    fn equal_launch_keys_are_one_entry() {
        let entries = [
            ("a", "launch", Invariance::NONE),
            ("b", "launch", Invariance::one_row(16)),
        ];
        for order in orders(&entries) {
            let candidates = order
                .iter()
                .map(|&(name, key, g)| Candidate::new(key, name, key.to_owned(), g));
            let needed = Need::rows(16).with_exact(Invariance::one_row(16));
            // Only the claiming registration keeps the need, so the op
            // claims what it declares.
            let chosen = choose("test", candidates, needed).unwrap();
            assert_eq!(chosen.guarantees, Invariance::one_row(16));
        }
        for order in orders(&entries) {
            let candidates = order
                .iter()
                .map(|&(name, key, g)| Candidate::new(key, name, key.to_owned(), g));
            let chosen = choose("test", candidates, Need::rows(16)).unwrap();
            assert_eq!(chosen.entry, "launch");
            // The op may launch either registration, so it claims what both
            // declare, in every order.
            assert_eq!(chosen.guarantees, Invariance::NONE);
        }
    }
}
