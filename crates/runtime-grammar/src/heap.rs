//! Upper estimates of the heap bytes owned collections hold, from their
//! capacities, so a cache that adds them up never reports less than it keeps.

/// A `Vec` or boxed slice with room for `capacity` items.
pub(crate) const fn vec_bytes<T>(capacity: usize) -> usize {
    capacity * size_of::<T>()
}

/// A `HashMap` or `HashSet` reporting `capacity`: its buckets, one control
/// byte per bucket, a trailing control group and padding to align the
/// control bytes.
pub(crate) fn table_bytes<T>(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let buckets = if capacity < 8 {
        (capacity + 1).next_power_of_two().max(4)
    } else {
        (capacity * 8 / 7).next_power_of_two()
    };
    buckets * (size_of::<T>() + 1) + 32
}

/// The allocation behind one `Arc<T>`: two counts and the value.
pub(crate) const fn arc_bytes<T>() -> usize {
    2 * size_of::<usize>() + size_of::<T>()
}
