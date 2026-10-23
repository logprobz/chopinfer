//! Decode complete `scale-row-tiles-offset-v1` banks through the shared
//! device helpers. The caller validates encoded contents before upload,
//! aligns the bank to 16 bytes, and keeps bank and output disjoint. Matrix
//! blocks are 4-byte aligned. Output has no alignment requirement.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::View;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
use kernel_core::precision::{scale_offset, scale_tiles};
use std::path::PathBuf;

pub const THREADS: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Form {
    Patch,
    Warp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScaleTilesQuery {
    pub width: u32,
    pub rows: u32,
    pub groups: u32,
    pub matrices: u32,
}

impl ScaleTilesQuery {
    /// Minimum bank size, with no exceptions. Contents need validation too.
    pub fn minimum_bank_bytes(&self) -> Result<u64, Refusal> {
        let fixed = scale_tiles::fixed_bytes(self.rows.into(), self.groups.into(), self.width)
            .map_err(|_| Refusal("invalid tile scale matrix shape or offset width"))?;
        if self.matrices == 0 {
            return Err(Refusal("tile scale decode needs matrices"));
        }
        let matrices = u64::from(self.matrices);
        let table = ((matrices + 1) * 4).next_multiple_of(scale_offset::TABLE_ALIGNMENT);
        let bytes = table + matrices * fixed;
        if bytes > u64::from(u32::MAX) {
            return Err(Refusal("tile scale bank exceeds u32 offsets"));
        }
        Ok(bytes)
    }

    pub fn output_bytes(&self) -> Result<u64, Refusal> {
        self.validate()?;
        Ok(u64::from(self.matrices) * u64::from(self.rows) * u64::from(self.groups))
    }
}

impl Validate for ScaleTilesQuery {
    fn validate(&self) -> Result<(), Refusal> {
        self.minimum_bank_bytes().map(|_| ())
    }
}

pub fn symbol(width: u32, form: Form) -> Result<&'static str, Refusal> {
    match (width, form) {
        (1, Form::Patch) => Ok("chop_moe_scale_tiles_decode_w1_v1"),
        (2, Form::Patch) => Ok("chop_moe_scale_tiles_decode_w2_v1"),
        (4, Form::Patch) => Ok("chop_moe_scale_tiles_decode_w4_v1"),
        (1, Form::Warp) => Ok("chop_moe_scale_tiles_decode_warp_w1_v1"),
        (2, Form::Warp) => Ok("chop_moe_scale_tiles_decode_warp_w2_v1"),
        (4, Form::Warp) => Ok("chop_moe_scale_tiles_decode_warp_w4_v1"),
        _ => Err(Refusal("tile scale offset width must be 1, 2 or 4")),
    }
}

pub fn plan(query: ScaleTilesQuery) -> Result<LaunchPlan, Refusal> {
    query.validate()?;
    let plan = LaunchPlan {
        grid: [query.rows.div_ceil(64), query.matrices, 1],
        block: [THREADS, 1, 1],
        cluster: [1; 3],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// Check extents and obvious aliases; alignment and contents are caller obligations.
pub fn arguments(
    query: ScaleTilesQuery,
    bank: View,
    output: View,
) -> Result<ArgumentList, Refusal> {
    plan(query)?;
    if bank.bytes < query.minimum_bank_bytes()? || bank.bytes > u64::from(u32::MAX) {
        return Err(Refusal("encoded tile scale bank extent is invalid"));
    }
    if output.bytes < query.output_bytes()? {
        return Err(Refusal("raw tile scale bank is too small"));
    }
    if bank.handle == output.handle {
        return Err(Refusal("tile scale decode buffers must be disjoint"));
    }
    Ok(ArgumentList(vec![
        Argument::DeviceBuffer(bank.handle),
        Argument::DeviceBuffer(output.handle),
        Argument::U32(query.rows),
        Argument::U32(query.groups),
    ]))
}

/// All widths and patch forms in one geometry-independent lab cubin.
pub fn cubin_spec(name: &str) -> CubinSpec {
    let entries = [Form::Patch, Form::Warp]
        .into_iter()
        .flat_map(|form| {
            [1, 2, 4].map(|width| {
                EntrySpec::new(
                    symbol(width, form).unwrap().into(),
                    vec![
                        super::ptr("bank", "const uint8_t"),
                        super::ptr("output", "uint8_t"),
                        EntryParam::U32 { name: "rows".into() },
                        EntryParam::U32 { name: "groups".into() },
                    ],
                    format!(
                        "chop::moe::scale_tiles::decode_bank<{width}, {}>(bank, output, rows, groups);",
                        form == Form::Warp
                    ),
                    THREADS,
                    1,
                )
            })
        })
        .collect();
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"moe/scale_tiles.cuh\"".into()],
        defines: vec![],
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> ScaleTilesQuery {
        ScaleTilesQuery {
            width: 1,
            rows: 65,
            groups: 18,
            matrices: 7,
        }
    }

    #[test]
    fn sizes_follow_codec_and_grid_covers_strips() {
        for width in [1, 2, 4] {
            for (rows, groups) in [
                (1, 1),
                (63, 5),
                (65, 18),
                (130, 7),
                (576, 160),
                (5120, 18),
                (64, 1024),
                (129, 3),
            ] {
                let q = ScaleTilesQuery {
                    width,
                    rows,
                    groups,
                    ..query()
                };
                let block = scale_tiles::fixed_bytes(rows.into(), groups.into(), width).unwrap();
                assert_eq!(
                    q.minimum_bank_bytes().unwrap(),
                    scale_offset::bank_layout(&[block; 7]).unwrap().bytes
                );
                assert_eq!(
                    q.output_bytes().unwrap(),
                    7 * u64::from(rows) * u64::from(groups)
                );
                let p = plan(q).unwrap();
                assert_eq!(p.grid, [rows.div_ceil(64), 7, 1]);
                assert_eq!(p.block, [THREADS, 1, 1]);
                assert_eq!((p.workspace_bytes, p.dynamic_shared_bytes), (0, 0));
                assert!(p.pdl);
            }
        }
    }

    #[test]
    fn invalid_shapes_counts_and_grids_are_refused() {
        for q in [
            ScaleTilesQuery {
                width: 3,
                ..query()
            },
            ScaleTilesQuery { rows: 0, ..query() },
            ScaleTilesQuery {
                rows: u32::MAX,
                ..query()
            },
            ScaleTilesQuery {
                groups: 0,
                ..query()
            },
            ScaleTilesQuery {
                groups: 1025,
                ..query()
            },
            ScaleTilesQuery {
                matrices: 0,
                ..query()
            },
            ScaleTilesQuery {
                matrices: u32::MAX,
                ..query()
            },
        ] {
            assert!(q.validate().is_err(), "{q:?}");
            assert!(q.output_bytes().is_err());
            assert!(plan(q).is_err());
        }
        let q = ScaleTilesQuery {
            matrices: 65_536,
            ..query()
        };
        assert!(q.validate().is_ok());
        assert!(plan(q).is_err());
        let high = ScaleTilesQuery {
            width: 1,
            rows: 16_384,
            groups: 1024,
            matrices: 257,
        };
        assert_eq!(high.output_bytes().unwrap(), 257 * (1_u64 << 24));
        assert!(plan(high).is_ok());
    }

    #[test]
    fn arguments_check_extents_aliases_and_abi() {
        let q = query();
        let bank = View {
            handle: 1,
            bytes: q.minimum_bank_bytes().unwrap(),
        };
        let output = View {
            handle: 2,
            bytes: q.output_bytes().unwrap(),
        };
        assert_eq!(
            arguments(q, bank, output).unwrap().0,
            vec![
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::U32(65),
                Argument::U32(18),
            ]
        );
        for bytes in [bank.bytes - 1, 1 << 32] {
            assert!(arguments(q, View { bytes, ..bank }, output).is_err());
        }
        assert!(
            arguments(
                q,
                bank,
                View {
                    bytes: output.bytes - 1,
                    ..output
                }
            )
            .is_err()
        );
        assert!(
            arguments(
                q,
                bank,
                View {
                    handle: bank.handle,
                    ..output
                }
            )
            .is_err()
        );
        assert!(
            arguments(
                q,
                bank,
                View {
                    bytes: output.bytes + 100,
                    ..output
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn cubin_declares_each_width_and_form_once() {
        let spec = cubin_spec("tile_decode_test");
        assert_eq!(spec.entries.len(), 6);
        for form in [Form::Patch, Form::Warp] {
            for width in [1, 2, 4] {
                let entries: Vec<_> = spec
                    .entries
                    .iter()
                    .filter(|e| e.symbol == symbol(width, form).unwrap())
                    .collect();
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].params.len(), 4);
                assert_eq!(entries[0].max_threads, THREADS);
            }
            assert!(symbol(3, form).is_err());
        }
    }
}
