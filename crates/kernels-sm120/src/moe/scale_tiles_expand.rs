//! Paired expansion from tile-order encoded scales to raw rows or gate
//! row tiles. These entries retain the v1 pair ABI and are unqualified.
//! Input banks are validated on the host; the launch waits for routing
//! and for earlier readers of the reused raw scratch. Banks are 16-byte
//! aligned; tiled output is 4-byte aligned. A null active list requires
//! a grid covering every expert, as in the v1 pair.
use super::{scale_expand, scale_tiles::ScaleTilesQuery};
use kernel_build::CubinSpec;
use kernel_core::backend::View;
use kernel_core::contract::{ArgumentList, LaunchPlan, Refusal};
pub use scale_expand::PairBank;
use std::path::PathBuf;

pub const PAIR_SYMBOL: &str = "chop_moe_scale_tiles_expand_pair_v1";
pub const PAIR_TILES_SYMBOL: &str = "chop_moe_scale_tiles_expand_pair_tiles_v1";
pub const COALESCED_SYMBOL: &str = "chop_moe_scale_tiles_expand_pair_coalesced_v1";

/// A separate candidate for tiled first outputs, with the same validated
/// spans and launch plan as `PAIR_TILES_SYMBOL`.
pub fn coalesced_cubin_spec(name: &str) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"moe/scale_tiles_expand_coalesced.cuh\"".into()],
        defines: vec![],
        entries: vec![scale_expand::pair_entry(
            COALESCED_SYMBOL,
            "run_tiles_pair_coalesced",
        )],
    }
}

pub fn minimum_bank_bytes(bank: PairBank, experts: u32) -> Result<u64, Refusal> {
    ScaleTilesQuery {
        width: bank.width,
        rows: bank.rows,
        groups: bank.groups,
        matrices: experts
            .checked_mul(bank.projections)
            .ok_or(Refusal("tile scale matrix count overflow"))?,
    }
    .minimum_bank_bytes()
}

/// A fixed grid over the same row segments and slots as the v1 pair.
/// Tiled first output requires a present first bank of whole 64x4 tiles.
pub fn plan(
    banks: [Option<PairBank>; 2],
    experts: u32,
    slots: u32,
    tiled_first: bool,
) -> Result<LaunchPlan, Refusal> {
    for bank in banks.iter().flatten() {
        minimum_bank_bytes(*bank, experts)?;
    }
    if tiled_first {
        scale_expand::pair_tiles_plan(banks, experts, slots)
    } else {
        scale_expand::pair_plan(banks, experts, slots)
    }
}

/// Check every input against the tile codec's minimum. In particular the
/// 18-group down bank at widths 1 and 2 is smaller than the v1 minimum.
pub fn arguments(
    banks: [Option<(PairBank, View, View)>; 2],
    active: Option<View>,
    experts: u32,
    tiled_first: bool,
) -> Result<ArgumentList, Refusal> {
    plan(banks.map(|b| b.map(|(q, _, _)| q)), experts, 1, tiled_first)?;
    scale_expand::pair_arguments_with(banks, active, experts, minimum_bank_bytes)
}

pub fn cubin_spec(name: &str) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"moe/scale_tiles_expand.cuh\"".into()],
        defines: vec![],
        entries: vec![
            scale_expand::pair_entry(PAIR_SYMBOL, "run_tiles_pair<false>"),
            scale_expand::pair_entry(PAIR_TILES_SYMBOL, "run_tiles_pair<true>"),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::Argument;
    use kernel_core::precision::{scale_offset, scale_tiles};
    fn bank(rows: u32, groups: u32, width: u32) -> PairBank {
        PairBank {
            width,
            rows,
            groups,
            projections: 1,
        }
    }
    #[test]
    fn tile_extent_tracks_partial_groups_and_rejects_short_or_aliased_banks() {
        for width in [1, 2, 4] {
            for (rows, groups) in [(65, 18), (576, 160), (5120, 18), (129, 3)] {
                let q = bank(rows, groups, width);
                let fixed = scale_tiles::fixed_bytes(rows.into(), groups.into(), width).unwrap();
                let bytes = scale_offset::bank_layout(&[fixed; 7]).unwrap().bytes;
                assert_eq!(minimum_bank_bytes(q, 7).unwrap(), bytes);
                let input = View { handle: 1, bytes };
                let output = View {
                    handle: 2,
                    bytes: u64::from(rows) * u64::from(groups) * 7,
                };
                let active = View {
                    handle: 3,
                    bytes: 32,
                };
                let banks = [None, Some((q, input, output))];
                let args = arguments(banks, Some(active), 7, false).unwrap();
                assert_eq!(args.0[0], Argument::NullDevicePointer);
                assert_eq!(args.0[1], Argument::DeviceBuffer(1));
                assert!(
                    arguments(
                        [
                            None,
                            Some((
                                q,
                                View {
                                    bytes: bytes - 1,
                                    ..input
                                },
                                output
                            ))
                        ],
                        None,
                        7,
                        false
                    )
                    .is_err()
                );
                assert!(
                    arguments(
                        banks,
                        Some(View {
                            handle: 1,
                            ..active
                        }),
                        7,
                        false
                    )
                    .is_err()
                );
                assert!(
                    arguments(
                        banks,
                        Some(View {
                            bytes: 31,
                            ..active
                        }),
                        7,
                        false
                    )
                    .is_err()
                );
                assert!(arguments(banks, None, 7, true).is_err());
                assert!(
                    arguments(
                        [
                            Some((
                                q,
                                input,
                                View {
                                    handle: 1,
                                    ..output
                                }
                            )),
                            None
                        ],
                        None,
                        7,
                        false
                    )
                    .is_err()
                );
            }
        }
    }
    #[test]
    fn tile_input_keeps_the_pair_grid_and_checks_raw_tile_shape() {
        let gate = bank(576, 160, 1);
        let down = bank(5120, 18, 4);
        let banks = [Some(gate), Some(down)];
        assert_eq!(
            plan(banks, 384, 48, true).unwrap(),
            scale_expand::pair_tiles_plan(banks, 384, 48).unwrap()
        );
        assert!(plan([None, None], 384, 1, false).is_err());
        assert!(plan(banks, 384, 0, false).is_err());
        assert!(plan(banks, 384, 385, false).is_err());
        assert!(plan([Some(bank(65, 18, 1)), None], 7, 7, true).is_err());
        assert!(plan([Some(bank(64, 1025, 1)), None], 7, 7, false).is_err());
        assert!(minimum_bank_bytes(bank(64, 4, 1), u32::MAX).is_err());
        for width in [1, 2, 4] {
            let down = bank(5120, 18, width);
            let v1 = scale_expand::ScaleExpandQuery {
                width,
                rows: 5120,
                groups: 18,
                projections: 1,
                experts: 384,
            }
            .minimum_bank_bytes()
            .unwrap();
            let tile = minimum_bank_bytes(down, 384).unwrap();
            assert_eq!(tile < v1, width < 4);
            assert_eq!(tile > v1, width == 4);
        }
    }
}
