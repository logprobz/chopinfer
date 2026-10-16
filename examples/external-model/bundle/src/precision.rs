//! Independent device exercise. The stateful ABI example stays available; the
//! SDK laboratory driver prepares these operations and checks their outputs.
//! The target's module prepares them over the entries embedded here
//! (`PreparedExercise::new` in `sm120`).

include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

pub fn cubins() -> impl Iterator<Item = (&'static str, &'static [u8])> {
    CUBINS.iter().map(|c| (c.name, c.bytes))
}

/// The entries of the embedded cubins.
pub(crate) fn symbols() -> impl Iterator<Item = &'static str> {
    CUBINS.iter().flat_map(|c| c.symbols.iter().copied())
}

pub use super::precision_program::PreparedExercise;
