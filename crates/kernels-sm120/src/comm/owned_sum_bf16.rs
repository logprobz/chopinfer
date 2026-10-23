//! Local steps of a transport that copies BF16 partials between ranks.
//!
//! The round entry rounds this rank's F32 partial once to BF16 for the
//! transport to send. The sum entry starts each element at positive F32
//! zero, adds the sources' BF16 values in ascending rank order, its own
//! rounded from `input` and every other from `inbox[source * slot]`, and
//! rounds once to BF16. Both call `comm::phased_bf16`'s device functions,
//! so the result has that all-reduce's bits. All pointers are local; the
//! caller completes the peer copies of rounded partials before the sum and
//! gathers the rounded output after.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, MAX_COMM_RANKS, Refusal};
use std::path::PathBuf;

pub const ROUND_SYMBOL: &str = "chop_comm_owned_round_bf16_v1";
pub const SUM_SYMBOL: &str = "chop_comm_owned_sum_bf16_v1";
pub const THREADS: u32 = 256;
/// Elements per thread step of the shared BF16 functions.
const VECTOR: u64 = 8;

fn grid(elements: u64) -> LaunchPlan {
    LaunchPlan {
        grid: [elements.div_ceil(1024).min(376) as u32, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    }
}

/// Rounding `elements` values: `input` has `elements * 4` bytes and
/// `output` `elements * 2`, both aligned to 16 bytes.
pub fn round_plan(elements: u64) -> Result<LaunchPlan, Refusal> {
    if elements == 0 || elements % VECTOR != 0 {
        return Err(Refusal(
            "BF16 rounding requires whole eight-element vectors",
        ));
    }
    Ok(grid(elements))
}

/// One owner's contiguous element interval. Every pointer is aligned to 16
/// bytes; `input` has `elements * 4` bytes, `output` `elements * 2`, and the
/// inbox `ranks * slot * 2`.
pub fn sum_plan(elements: u64, slot: u64, rank: u32, ranks: u32) -> Result<LaunchPlan, Refusal> {
    if elements == 0
        || elements % VECTOR != 0
        || slot % VECTOR != 0
        || slot < elements
        || ranks == 0
        || ranks as usize > MAX_COMM_RANKS
        || rank >= ranks
        || slot
            .checked_mul(u64::from(ranks))
            .and_then(|v| v.checked_mul(2))
            .is_none()
    {
        return Err(Refusal(
            "BF16 owner sum requires bounded vector-aligned rank slots",
        ));
    }
    Ok(grid(elements))
}

pub fn cubin_spec(name: String) -> CubinSpec {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let elements = || EntryParam::U64 {
        name: "elements".into(),
    };
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"comm/owned_sum_bf16.cuh\"".into()],
        defines: vec![("CHOP_COMM_MAX_RANKS".into(), MAX_COMM_RANKS.to_string())],
        entries: vec![
            EntrySpec::new(
                ROUND_SYMBOL.into(),
                vec![
                    pointer("input", "const float"),
                    pointer("output", "uint16_t"),
                    elements(),
                ],
                "chop::comm::owned_sum_bf16::round(input, output, elements);".into(),
                THREADS,
                1,
            ),
            EntrySpec::new(
                SUM_SYMBOL.into(),
                vec![
                    pointer("input", "const float"),
                    pointer("inbox", "const uint16_t"),
                    pointer("output", "uint16_t"),
                    elements(),
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
                "chop::comm::owned_sum_bf16::run(input, inbox, output, elements, slot, rank, ranks);"
                    .into(),
                THREADS,
                1,
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = include_str!("../../cuda/comm/owned_sum_bf16.cuh");

    #[test]
    fn plans_require_whole_vectors_and_bounded_slots() {
        for rows in [1, 1023, 1024, 2048] {
            for p in [
                round_plan(rows * 4096).unwrap(),
                sum_plan(rows * 4096, 2048 * 4096, 3, 4).unwrap(),
            ] {
                assert_eq!(p.block, [THREADS, 1, 1]);
                assert_eq!(p.cluster, [1, 1, 1]);
                assert_eq!(p.workspace_bytes, 0);
                assert!(!p.pdl);
            }
        }
        assert!(round_plan(0).is_err() && round_plan(4).is_err());
        for (elements, slot, rank, ranks) in [
            (0, 1024, 0, 4),
            (4, 1024, 0, 4),
            (1024, 1028, 0, 4),
            (1032, 1024, 0, 4),
            (1024, 1024, 4, 4),
            (1024, 1024, 0, 0),
            (1024, 1024, 0, MAX_COMM_RANKS as u32 + 1),
            (1024, u64::MAX - 7, 0, 4),
        ] {
            assert!(sum_plan(elements, slot, rank, ranks).is_err());
        }
    }

    #[test]
    fn entries_reuse_the_phased_bf16_arithmetic() {
        // The THREADS here are the shared functions' block size.
        assert!(
            include_str!("../../cuda/comm/allreduce_phased_f32.cuh")
                .contains("constexpr uint32_t kThreads = 256;")
        );
        assert_eq!(HEADER.matches("phased_bf16::send_rounded(").count(), 1);
        assert_eq!(HEADER.matches("phased_bf16::sum_block(").count(), 1);
        // No arithmetic of its own: no conversion and no element stores.
        assert!(!HEADER.contains("cvt.") && !HEADER.contains("output[") && !HEADER.contains("]="));
        let spec = cubin_spec("test".into());
        assert_eq!(
            spec.entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            [ROUND_SYMBOL, SUM_SYMBOL]
        );
    }
}
