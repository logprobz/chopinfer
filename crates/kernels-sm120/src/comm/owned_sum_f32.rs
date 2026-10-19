//! Local owner sum for a transport that supplies rank-indexed F32 inboxes.
//!
//! Each element starts at positive F32 zero, adds sources in ascending rank
//! order, and rounds once to BF16. The owner's partial comes from `input`;
//! every other source comes from `inbox[source * slot]`. All pointers are
//! local. This is only the arithmetic step of a collective: the caller must
//! complete peer copies before launching and gather the rounded output after.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, MAX_COMM_RANKS, Refusal};
use std::path::PathBuf;

pub const SYMBOL: &str = "chop_comm_owned_sum_round_f32_v1";
pub const THREADS: u32 = 256;

/// Bounds and launch geometry for one owner's contiguous element interval.
/// Every pointer is aligned to 16 bytes; `input` has `elements * 4` bytes,
/// `output` has `elements * 2`, and the inbox has `ranks * slot * 4`.
pub fn plan(elements: u64, slot: u64, rank: u32, ranks: u32) -> Result<LaunchPlan, Refusal> {
    if elements == 0
        || elements % 4 != 0
        || slot % 4 != 0
        || slot < elements
        || ranks == 0
        || ranks as usize > MAX_COMM_RANKS
        || rank >= ranks
        || slot
            .checked_mul(u64::from(ranks))
            .and_then(|v| v.checked_mul(4))
            .is_none()
    {
        return Err(Refusal(
            "owner sum requires bounded vector-aligned rank slots",
        ));
    }
    Ok(LaunchPlan {
        grid: [elements.div_ceil(1024).min(376) as u32, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    })
}

pub fn cubin_spec(name: String) -> CubinSpec {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"comm/owned_sum_f32.cuh\"".into()],
        defines: vec![("CHOP_COMM_MAX_RANKS".into(), MAX_COMM_RANKS.to_string())],
        entries: vec![EntrySpec::new(
            SYMBOL.into(),
            vec![
                pointer("input", "const float"),
                pointer("inbox", "const float"),
                pointer("output", "uint16_t"),
                EntryParam::U64 {
                    name: "elements".into(),
                },
                EntryParam::U64 {
                    name: "slot".into(),
                },
                EntryParam::U32 {
                    name: "rank".into(),
                },
                EntryParam::U32 {
                    name: "ranks".into(),
                },
            ],
            "chop::comm::owned_sum_f32::run(input, inbox, output, elements, slot, rank, ranks);"
                .into(),
            THREADS,
            1,
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owner_sum_rejects_truncated_slots_and_has_no_cooperative_requirement() {
        for rows in [1, 1023, 1024, 2048] {
            let p = plan(rows * 5120, 2048 * 5120, 3, 4).unwrap();
            assert_eq!(p.cluster, [1, 1, 1]);
            assert_eq!(p.workspace_bytes, 0);
            assert!(!p.pdl);
        }
        for (elements, slot, rank, ranks) in [
            (0, 1024, 0, 4),
            (3, 1024, 0, 4),
            (1024, 1025, 0, 4),
            (1028, 1024, 0, 4),
            (1024, 1024, 4, 4),
            (1024, 1024, 0, 0),
            (1024, 1024, 0, MAX_COMM_RANKS as u32 + 1),
            (1024, u64::MAX - 3, 0, 4),
        ] {
            assert!(plan(elements, slot, rank, ranks).is_err());
        }
    }
}
