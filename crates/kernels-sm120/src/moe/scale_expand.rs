//! Expand selected `scale-row-offset-v1` matrices into a raw scale bank.
//!
//! Build [`cubin_spec`] once, resolve [`symbol`] for the bank's width, then
//! use [`plan`] and [`arguments`]. The caller validates the encoded bank at
//! load, aligns it to 16 bytes and keeps all three buffers disjoint. A
//! non-null active list is aligned to 4 bytes and holds `experts + 1` i32s:
//! a count in `0..=experts`, then unique expert ids in `0..experts`. The
//! remaining words are unread. Null selects all experts. Only selected
//! experts' bytes are written, at their original matrix positions.
//!
//! Order routing, expansion and scale consumers on the same stream. The
//! entry launches with PDL: every CTA waits for its predecessor before it
//! reads the active list or writes the raw bank, so it follows the routes
//! and, transitively, every earlier reader of the raw bank whose kernel
//! waits for its own predecessor. A consumer reads the raw bank only after
//! its own wait. It requires no host readback, device allocation, scratch
//! initialization or live-count grid.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::View;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal, Validate};
pub use kernel_core::precision::scale_expand::ScaleExpandQuery;
use std::path::PathBuf;

pub const TILE_ROWS: u32 = 64;
pub const THREADS: u32 = 256;

pub fn symbol(width: u32) -> Result<&'static str, Refusal> {
    match width {
        1 => Ok("chop_moe_scale_expand_w1_v1"),
        2 => Ok("chop_moe_scale_expand_w2_v1"),
        4 => Ok("chop_moe_scale_expand_w4_v1"),
        _ => Err(Refusal("scale expansion offset width must be 1, 2 or 4")),
    }
}

/// A fixed grid covering the maximum active count, including empty steps.
pub fn plan(query: ScaleExpandQuery) -> Result<LaunchPlan, Refusal> {
    plan_slots(query, query.experts)
}

/// A grid of `slots` active-list positions, for a caller that bounds the
/// active count below the expert count (a decode bucket's rows times its
/// top-k). Slots past the count exit at once; experts past `slots` would be
/// skipped, so a null active list needs every expert's slot.
pub fn plan_slots(query: ScaleExpandQuery, slots: u32) -> Result<LaunchPlan, Refusal> {
    query.validate()?;
    if slots == 0 || slots > query.experts {
        return Err(Refusal("scale expansion slots must be 1..=experts"));
    }
    let plan = LaunchPlan {
        grid: [query.rows.div_ceil(TILE_ROWS), query.projections, slots],
        block: [THREADS, 1, 1],
        cluster: [1; 3],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// Check buffer extents and pack the entry ABI. Handles are opaque, so
/// alignment, distinct allocations and encoded contents remain caller
/// obligations. Equal handles are rejected as an obvious alias.
pub fn arguments(
    query: ScaleExpandQuery,
    bank: View,
    active: Option<View>,
    output: View,
) -> Result<ArgumentList, Refusal> {
    plan(query)?;
    if bank.bytes < query.minimum_bank_bytes()? || bank.bytes > u64::from(u32::MAX) {
        return Err(Refusal("encoded scale bank extent is invalid"));
    }
    if output.bytes < query.output_bytes()? {
        return Err(Refusal("raw scale bank is too small"));
    }
    if let Some(active) = active {
        if active.bytes < query.active_bytes()? {
            return Err(Refusal("active expert list is too small"));
        }
        if active.handle == output.handle || active.handle == bank.handle {
            return Err(Refusal("scale expansion buffers must be disjoint"));
        }
    }
    if bank.handle == output.handle {
        return Err(Refusal("scale expansion buffers must be disjoint"));
    }
    Ok(ArgumentList(vec![
        Argument::DeviceBuffer(bank.handle),
        active.map_or(Argument::NullDevicePointer, |v| {
            Argument::DeviceBuffer(v.handle)
        }),
        Argument::DeviceBuffer(output.handle),
        Argument::U32(query.rows),
        Argument::U32(query.groups),
        Argument::U32(query.projections),
        Argument::U32(query.experts),
    ]))
}

/// One geometry-independent cubin with all three offset widths. Register
/// this spec in its own kernel-build lock family; existing MoE cubins do
/// not include its header or acquire new entries.
pub fn cubin_spec(name: &str) -> CubinSpec {
    let entries = [1, 2, 4]
        .into_iter()
        .map(|width| {
            EntrySpec::new(
                symbol(width).unwrap().into(),
                vec![
                    super::ptr("bank", "const uint8_t"),
                    super::ptr("active", "const int32_t"),
                    super::ptr("output", "uint8_t"),
                    EntryParam::U32 { name: "rows".into() },
                    EntryParam::U32 { name: "groups".into() },
                    EntryParam::U32 { name: "projections".into() },
                    EntryParam::U32 { name: "experts".into() },
                ],
                format!("chop::moe::scale_expand::run<{width}>(bank, active, output, rows, groups, projections, experts);"),
                THREADS,
                1,
            )
        })
        .collect();
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"moe/scale_expand.cuh\"".into()],
        defines: vec![],
        entries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> ScaleExpandQuery {
        ScaleExpandQuery {
            width: 1,
            rows: 65,
            groups: 18,
            projections: 2,
            experts: 7,
        }
    }

    #[test]
    fn scale_expand_grid_covers_tail_rows_and_full_capacity() {
        for rows in [1, 63, 64, 65, 576, 5120] {
            let q = ScaleExpandQuery { rows, ..query() };
            let p = plan(q).unwrap();
            assert_eq!(p.grid, [rows.div_ceil(64), 2, 7]);
            assert_eq!(p.block, [256, 1, 1]);
            assert!(p.pdl);
            assert_eq!((p.workspace_bytes, p.dynamic_shared_bytes), (0, 0));
        }
        for q in [
            ScaleExpandQuery {
                experts: 65_536,
                ..query()
            },
            ScaleExpandQuery {
                projections: 65_536,
                experts: 1,
                ..query()
            },
        ] {
            assert!(q.validate().is_ok());
            assert!(plan(q).is_err());
        }
        for (rows, groups, projections, tiles) in [(576, 160, 2, 9), (5120, 18, 1, 80)] {
            let q = ScaleExpandQuery {
                rows,
                groups,
                projections,
                experts: 384,
                ..query()
            };
            assert_eq!(plan(q).unwrap().grid, [tiles, projections, 384]);
            // A one-row top-6 decode bucket lists at most six experts.
            assert_eq!(plan_slots(q, 6).unwrap().grid, [tiles, projections, 6]);
            assert!(plan_slots(q, 0).is_err() && plan_slots(q, 385).is_err());
        }
    }

    #[test]
    fn scale_expand_checks_extents_and_packs_nullable_list() {
        let q = query();
        let bank = View {
            handle: 1,
            bytes: q.minimum_bank_bytes().unwrap(),
        };
        let active = View {
            handle: 2,
            bytes: q.active_bytes().unwrap(),
        };
        let output = View {
            handle: 3,
            bytes: q.output_bytes().unwrap(),
        };
        let args = arguments(q, bank, Some(active), output).unwrap();
        assert_eq!(
            args.0,
            vec![
                Argument::DeviceBuffer(1),
                Argument::DeviceBuffer(2),
                Argument::DeviceBuffer(3),
                Argument::U32(65),
                Argument::U32(18),
                Argument::U32(2),
                Argument::U32(7),
            ]
        );
        assert_eq!(
            arguments(q, bank, None, output).unwrap().0[1],
            Argument::NullDevicePointer
        );
        assert!(
            arguments(
                q,
                View {
                    bytes: bank.bytes - 1,
                    ..bank
                },
                None,
                output
            )
            .is_err()
        );
        assert!(
            arguments(
                q,
                View {
                    bytes: 1 << 32,
                    ..bank
                },
                None,
                output
            )
            .is_err()
        );
        assert!(
            arguments(
                q,
                bank,
                Some(View {
                    bytes: active.bytes - 1,
                    ..active
                }),
                output
            )
            .is_err()
        );
        assert!(
            arguments(
                q,
                bank,
                None,
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
                None,
                View {
                    handle: 1,
                    ..output
                }
            )
            .is_err()
        );
        assert!(
            arguments(
                q,
                bank,
                Some(View {
                    handle: 3,
                    ..active
                }),
                output
            )
            .is_err()
        );
        assert!(arguments(ScaleExpandQuery { width: 3, ..q }, bank, None, output).is_err());
    }

    #[test]
    fn scale_expand_registers_three_geometry_independent_entries() {
        let spec = cubin_spec("scale_expand");
        let generated = kernel_build::generate(&spec).unwrap();
        assert_eq!(spec.entries.len(), 3);
        for (entry, width) in spec.entries.iter().zip([1, 2, 4]) {
            assert_eq!(entry.symbol, symbol(width).unwrap());
            assert_eq!(entry.params.len(), 7);
            assert!(generated.translation_unit.contains(&entry.symbol));
        }
        assert!(symbol(8).is_err());
    }
}
