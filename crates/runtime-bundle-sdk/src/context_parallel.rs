//! The create record's context-parallel field.
//!
//! A host serving a context-parallel group size above 1 sends this tag and a
//! little-endian `u32` size; at size 1 it sends nothing. Groups are `size`
//! consecutive ranks, and a sequence's physical page `j` lives on group rank
//! `j % size`, at the local index of its logical page (the host's
//! `runtime_kv::ContextParallelPagesV1`).

use runtime_bundle_abi::BundleStatusV1;

use crate::BatchReaderV1;

/// The field's tag (the host's `CONTEXT_PARALLEL_TAG_V1`).
pub const TAG: &[u8] = b"chop.context-parallel.v1";

/// How an instance's ranks form context-parallel groups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextParallelV1 {
    size: u32,
}

impl ContextParallelV1 {
    /// Every rank holds every page.
    pub const NONE: Self = Self { size: 1 };

    /// Ranks per group.
    #[must_use]
    pub const fn size(self) -> u32 {
        self.size
    }

    /// The group rank of topology rank `rank`.
    #[must_use]
    pub const fn group_rank(self, rank: u32) -> u32 {
        rank % self.size
    }

    /// The group topology rank `rank` belongs to.
    #[must_use]
    pub const fn group(self, rank: u32) -> u32 {
        rank / self.size
    }

    /// The group rank that holds sequence position `position` over physical
    /// pages of `page_tokens` tokens.
    #[must_use]
    pub const fn owner(self, position: u64, page_tokens: u64) -> u32 {
        (position / page_tokens % self.size as u64) as u32
    }
}

/// Whether the record continues with this field.
#[must_use]
pub fn is_next(remaining: &[u8]) -> bool {
    remaining.starts_with(TAG)
}

/// Reads the field at the current create-record cursor for a topology of
/// `ranks` ranks whose bundle declares the group sizes `declared` (1 may be
/// listed or not). A size of 1, which a host never sends, or one that does
/// not divide the ranks is `InvalidArgument`; an undeclared size is
/// `Unsupported`.
pub fn read(
    reader: &mut BatchReaderV1<'_>,
    ranks: u32,
    declared: &[u32],
) -> Result<ContextParallelV1, BundleStatusV1> {
    reader
        .expect_magic(TAG)
        .map_err(|_| BundleStatusV1::InvalidArgument)?;
    let size = reader.u32().map_err(|_| BundleStatusV1::InvalidArgument)?;
    if size < 2 || ranks == 0 || !ranks.is_multiple_of(size) {
        return Err(BundleStatusV1::InvalidArgument);
    }
    if !declared.contains(&size) {
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(ContextParallelV1 { size })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(size: u32) -> Vec<u8> {
        let mut bytes = TAG.to_vec();
        bytes.extend_from_slice(&size.to_le_bytes());
        bytes
    }

    fn decoded(size: u32, declared: &[u32]) -> Result<ContextParallelV1, BundleStatusV1> {
        let bytes = field(size);
        assert!(is_next(&bytes));
        let mut reader = BatchReaderV1::new("context parallel", &bytes);
        let value = read(&mut reader, 4, declared)?;
        reader.finish().unwrap();
        Ok(value)
    }

    #[test]
    fn a_declared_size_dividing_the_ranks_forms_consecutive_groups() {
        let pairs = decoded(2, &[1, 2, 4]).unwrap();
        assert_eq!(pairs.size(), 2);
        assert_eq!(
            (0..4)
                .map(|rank| pairs.group_rank(rank))
                .collect::<Vec<_>>(),
            [0, 1, 0, 1]
        );
        assert_eq!(
            (0..4).map(|rank| pairs.group(rank)).collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
        assert_eq!(
            (0..12)
                .map(|position| pairs.owner(position, 4))
                .collect::<Vec<_>>(),
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        );
        assert_eq!(ContextParallelV1::NONE.group_rank(3), 0);
        assert_eq!(decoded(4, &[4]).unwrap().size(), 4);
    }

    #[test]
    fn undeclared_unsent_and_uneven_sizes_are_refused() {
        assert_eq!(decoded(2, &[1]), Err(BundleStatusV1::Unsupported));
        assert_eq!(decoded(2, &[]), Err(BundleStatusV1::Unsupported));
        assert_eq!(decoded(1, &[1]), Err(BundleStatusV1::InvalidArgument));
        assert_eq!(decoded(3, &[3]), Err(BundleStatusV1::InvalidArgument));
        assert_eq!(decoded(0, &[]), Err(BundleStatusV1::InvalidArgument));
        let mut reader = BatchReaderV1::new("context parallel", &TAG[..8]);
        assert_eq!(
            read(&mut reader, 4, &[2]),
            Err(BundleStatusV1::InvalidArgument)
        );
    }
}
