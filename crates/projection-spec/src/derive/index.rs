//! Bound operands indexed by instance and tensor name for one derivation.
//!
//! A full served graph derives tens of thousands of operations over tens of
//! thousands of per-expert operands, so scanning every operand for each
//! operation is quadratic. [`derive`](super::derive) indexes the binding it
//! borrows for its whole run; helpers called outside a derivation, or with
//! another binding, scan as before. Results keep the binding's operand order.
use crate::bind::{BoundDeclaration, BoundOperand};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

#[derive(Default)]
struct Index {
    instances: HashMap<(String, Option<u64>), Vec<usize>>,
    tensors: HashMap<String, Vec<usize>>,
}

type Active = Option<(usize, usize, Rc<Index>)>;

thread_local! {
    static ACTIVE: RefCell<Active> = const { RefCell::new(None) };
}

/// Restores the enclosing derivation's index when dropped.
pub(super) struct Scope(Active);

impl Drop for Scope {
    fn drop(&mut self) {
        let previous = self.0.take();
        ACTIVE.with(|active| *active.borrow_mut() = previous);
    }
}

fn key(bound: &BoundDeclaration) -> (usize, usize) {
    (std::ptr::from_ref(bound) as usize, bound.operands.len())
}

/// Index `bound` until the returned scope drops. The caller must hold the
/// shared borrow of `bound` for the whole scope.
pub(super) fn scope(bound: &BoundDeclaration) -> Scope {
    let mut index = Index::default();
    for (position, operand) in bound.operands.iter().enumerate() {
        index
            .instances
            .entry((operand.projection.clone(), operand.layer))
            .or_default()
            .push(position);
        index
            .tensors
            .entry(operand.imported.weight.codes.tensor.clone())
            .or_default()
            .push(position);
    }
    let (address, length) = key(bound);
    let previous = ACTIVE.with(|active| {
        active
            .borrow_mut()
            .replace((address, length, Rc::new(index)))
    });
    Scope(previous)
}

fn active(bound: &BoundDeclaration) -> Option<Rc<Index>> {
    ACTIVE.with(|active| {
        active
            .borrow()
            .as_ref()
            .filter(|(address, length, _)| (*address, *length) == key(bound))
            .map(|(_, _, index)| Rc::clone(index))
    })
}

/// Operands of one projection instance, or whose code tensor is in `names`.
pub(super) fn operands<'a>(
    bound: &'a BoundDeclaration,
    instance: Option<(&str, Option<u64>)>,
    names: &BTreeSet<&str>,
) -> Vec<&'a BoundOperand> {
    let Some(index) = active(bound) else {
        return bound
            .operands
            .iter()
            .filter(|b| {
                instance.is_some_and(|(p, l)| b.projection == p && b.layer == l)
                    || names.contains(b.imported.weight.codes.tensor.as_str())
            })
            .collect();
    };
    let mut positions = BTreeSet::new();
    if let Some((projection, layer)) = instance {
        positions.extend(
            index
                .instances
                .get(&(projection.to_owned(), layer))
                .into_iter()
                .flatten(),
        );
    }
    for name in names {
        positions.extend(index.tensors.get(*name).into_iter().flatten());
    }
    positions
        .into_iter()
        .map(|&position| &bound.operands[position])
        .collect()
}
