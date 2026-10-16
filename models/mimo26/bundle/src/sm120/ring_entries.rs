//! The entries of the bundle-owned ring kernels per ring slot count: the
//! 32-slot entries of the topology's `ring` cubin, and the `_s{slots}`
//! entries of its `ring_slots` cubin for every other slot count.

use crate::topology::Tp;

/// The bundle-owned ring kernel `name` for a ring of `slots` slots:
/// `chop_mimo_ring_backup_v1` at TP4 and 32 slots, the `_s{slots}` entry
/// of the `ring_slots` cubin otherwise.
pub fn ring_symbol(tp: Tp, name: &str, slots: u32) -> String {
    if slots == crate::contract::DECODE_ENTRY_ROWS {
        crate::topology::bundle_symbol(tp, name)
    } else {
        crate::topology::bundle_symbol(tp, &format!("{name}_s{slots}"))
    }
}

/// The bundle-owned kernel that saves touched SWA ring entries before a
/// phase (`chop_mimo_ring_backup_v1` at TP4 and 32 ring slots).
pub fn ring_backup_symbol(tp: Tp, slots: u32) -> String {
    ring_symbol(tp, "ring_backup", slots)
}

/// The bundle-owned kernel that copies a snapshot between a slot and pages
/// (`chop_mimo_ring_prefix_copy_v1` at TP4 and 32 ring slots).
pub fn prefix_copy_symbol(tp: Tp, slots: u32) -> String {
    ring_symbol(tp, "ring_prefix_copy", slots)
}
