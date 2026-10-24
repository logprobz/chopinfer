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

/// The paired entry: two banks of one layer, each at its own runtime width.
pub const PAIR_SYMBOL: &str = "chop_moe_scale_expand_pair_v1";

/// One bank of a paired expansion: its offset width and raw matrix shape.
/// Both banks share the expert count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairBank {
    pub width: u32,
    pub rows: u32,
    pub groups: u32,
    pub projections: u32,
}

impl PairBank {
    fn query(self, experts: u32) -> ScaleExpandQuery {
        ScaleExpandQuery {
            width: self.width,
            rows: self.rows,
            groups: self.groups,
            projections: self.projections,
            experts,
        }
    }

    /// Rows one CTA decodes: whole 64-row strips holding at least 8 KiB
    /// of scales where the matrix has them.
    pub fn tile_rows(self) -> u32 {
        (8192 / self.groups.max(1))
            .next_multiple_of(TILE_ROWS)
            .max(TILE_ROWS)
            .min(self.rows.next_multiple_of(TILE_ROWS))
    }

    fn tiles(self) -> u32 {
        self.projections * self.rows.div_ceil(self.tile_rows())
    }
}

/// The paired launch over `slots` active-list positions (every expert's
/// slot without a list): grid x covers the first bank's tiles, then the
/// second's. An absent bank takes no tiles; at least one is present.
pub fn pair_plan(
    banks: [Option<PairBank>; 2],
    experts: u32,
    slots: u32,
) -> Result<LaunchPlan, Refusal> {
    if banks.iter().all(Option::is_none) {
        return Err(Refusal("a paired scale expansion needs a bank"));
    }
    let mut tiles = 0_u32;
    for bank in banks.iter().flatten() {
        bank.query(experts).validate()?;
        tiles = tiles
            .checked_add(bank.tiles())
            .ok_or(Refusal("scale expansion tiles overflow"))?;
    }
    if slots == 0 || slots > experts || slots > 65_535 {
        return Err(Refusal("scale expansion slots must be 1..=experts"));
    }
    let plan = LaunchPlan {
        grid: [tiles, 1, slots],
        block: [THREADS, 1, 1],
        cluster: [1; 3],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// Check both banks' extents and pack the paired entry's ABI: per bank its
/// encoded view and raw output, or nulls and zero shape for an absent bank.
pub fn pair_arguments(
    banks: [Option<(PairBank, View, View)>; 2],
    active: Option<View>,
    experts: u32,
) -> Result<ArgumentList, Refusal> {
    pair_arguments_with(banks, active, experts, |bank, experts| {
        bank.query(experts).minimum_bank_bytes()
    })
}

/// The common paired ABI with an encoding-specific minimum input extent.
pub(super) fn pair_arguments_with(
    banks: [Option<(PairBank, View, View)>; 2],
    active: Option<View>,
    experts: u32,
    minimum: fn(PairBank, u32) -> Result<u64, Refusal>,
) -> Result<ArgumentList, Refusal> {
    pair_plan(banks.map(|bank| bank.map(|(bank, _, _)| bank)), experts, 1)?;
    let mut handles = Vec::new();
    for (bank, encoded, output) in banks.iter().flatten() {
        let query = bank.query(experts);
        if encoded.bytes < minimum(*bank, experts)? || encoded.bytes > u64::from(u32::MAX) {
            return Err(Refusal("encoded scale bank extent is invalid"));
        }
        if output.bytes < query.output_bytes()? {
            return Err(Refusal("raw scale bank is too small"));
        }
        handles.extend([encoded.handle, output.handle]);
    }
    if let Some(active) = active {
        if active.bytes < (u64::from(experts) + 1) * 4 {
            return Err(Refusal("active expert list is too small"));
        }
        handles.push(active.handle);
    }
    let mut sorted = handles.clone();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.len() != handles.len() {
        return Err(Refusal("scale expansion buffers must be disjoint"));
    }
    let pointer = |view: Option<View>| {
        view.map_or(Argument::NullDevicePointer, |v| {
            Argument::DeviceBuffer(v.handle)
        })
    };
    let [first, second] = banks;
    let mut arguments = vec![
        pointer(first.map(|(_, encoded, _)| encoded)),
        pointer(second.map(|(_, encoded, _)| encoded)),
        pointer(active),
        pointer(first.map(|(_, _, output)| output)),
        pointer(second.map(|(_, _, output)| output)),
    ];
    for bank in [first, second] {
        let shape = bank.map(|(bank, _, _)| {
            [
                bank.rows,
                bank.groups,
                bank.projections,
                bank.width,
                bank.tile_rows(),
            ]
        });
        arguments.extend(shape.unwrap_or([0; 5]).map(Argument::U32));
    }
    arguments.push(Argument::U32(experts));
    Ok(ArgumentList(arguments))
}

/// The paired entry writing its first bank's raw matrices as the
/// `row-tiles-64x128-groups` scale section a row-tiled gate/up consumer
/// reads: the paired entry's bytes, each matrix's 64-row strips stored as
/// their tiles of 4 groups (`kernel_core::transform::RowTileLayout` with
/// 4-byte tiles). The second bank is the paired entry's. It takes the
/// paired entry's ABI and grid, a first bank of whole tiles, and 4-byte
/// aligned outputs.
pub const PAIR_TILES_SYMBOL: &str = "chop_moe_scale_expand_pair_tiles_v1";

/// Refuses a first bank the tiled entry cannot write: absent, or rows that
/// are not whole 64-row strips or groups that are not whole 4-group tiles.
fn whole_tiles(first: Option<PairBank>) -> Result<(), Refusal> {
    match first {
        Some(bank) if bank.rows.is_multiple_of(TILE_ROWS) && bank.groups.is_multiple_of(4) => {
            Ok(())
        }
        _ => Err(Refusal(
            "a tiled scale expansion needs a first bank of whole 64-row by 4-group tiles",
        )),
    }
}

/// [`pair_plan`] for [`PAIR_TILES_SYMBOL`].
pub fn pair_tiles_plan(
    banks: [Option<PairBank>; 2],
    experts: u32,
    slots: u32,
) -> Result<LaunchPlan, Refusal> {
    whole_tiles(banks[0])?;
    pair_plan(banks, experts, slots)
}

/// [`pair_arguments`] for [`PAIR_TILES_SYMBOL`].
pub fn pair_tiles_arguments(
    banks: [Option<(PairBank, View, View)>; 2],
    active: Option<View>,
    experts: u32,
) -> Result<ArgumentList, Refusal> {
    whole_tiles(banks[0].map(|(bank, _, _)| bank))?;
    pair_arguments(banks, active, experts)
}

/// The paired entry's ABI, calling `function` (`run_pair` or
/// `run_pair_tiles` of `chop::moe::scale_expand`).
pub(super) fn pair_entry(symbol: &str, function: &str) -> EntrySpec {
    let params = ["first", "second"]
        .into_iter()
        .map(|bank| super::ptr(bank, "const uint8_t"))
        .chain([super::ptr("active", "const int32_t")])
        .chain(
            ["first_output", "second_output"]
                .into_iter()
                .map(|output| super::ptr(output, "uint8_t")),
        )
        .chain(["first", "second"].into_iter().flat_map(|bank| {
            ["rows", "groups", "projections", "width", "tile_rows"].map(|field| EntryParam::U32 {
                name: format!("{bank}_{field}"),
            })
        }))
        .chain([EntryParam::U32 {
            name: "experts".into(),
        }])
        .collect();
    EntrySpec::new(
        symbol.into(),
        params,
        format!(
            "chop::moe::scale_expand::{function}(\
            chop::moe::scale_expand::Bank{{first, first_output, first_rows, first_groups, first_projections, first_width, first_tile_rows}}, \
            chop::moe::scale_expand::Bank{{second, second_output, second_rows, second_groups, second_projections, second_width, second_tile_rows}}, \
            active, experts);"
        ),
        THREADS,
        1,
    )
}

/// One geometry-independent cubin with all three offset widths and the
/// paired entry. Register this spec in its own kernel-build lock family;
/// existing MoE cubins do not include its header or acquire new entries.
pub fn cubin_spec(name: &str) -> CubinSpec {
    let pair = pair_entry(PAIR_SYMBOL, "run_pair");
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
        .chain([pair])
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

/// One geometry-independent cubin with the tiled paired entry alone, for
/// its own lock family: [`cubin_spec`]'s cubins keep their entries and
/// source closure.
pub fn tiles_cubin_spec(name: &str) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"moe/scale_expand_tiles.cuh\"".into()],
        defines: vec![],
        entries: vec![pair_entry(PAIR_TILES_SYMBOL, "run_pair_tiles")],
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

    /// Gate/up with 160 groups and down with 18 groups exercise 64-row
    /// and 512-row tiles, respectively.
    #[test]
    fn paired_expansion_tiles_both_banks_and_packs_absent_ones_as_null() {
        let gate = PairBank {
            width: 1,
            rows: 576,
            groups: 160,
            projections: 2,
        };
        let down = PairBank {
            width: 4,
            rows: 5120,
            groups: 18,
            projections: 1,
        };
        assert_eq!((gate.tile_rows(), down.tile_rows()), (64, 512));
        let plan = pair_plan([Some(gate), Some(down)], 384, 6).unwrap();
        assert_eq!(plan.grid, [18 + 10, 1, 6]);
        assert!(plan.pdl);
        assert_eq!(
            pair_plan([None, Some(down)], 384, 384).unwrap().grid,
            [10, 1, 384]
        );
        assert!(pair_plan([None, None], 384, 6).is_err());
        assert!(pair_plan([Some(gate), None], 384, 0).is_err());
        assert!(pair_plan([Some(gate), None], 384, 385).is_err());
        let view = |handle, query: ScaleExpandQuery, output: bool| View {
            handle,
            bytes: if output {
                query.output_bytes().unwrap()
            } else {
                query.minimum_bank_bytes().unwrap()
            },
        };
        let (gq, dq) = (gate.query(384), down.query(384));
        let active = View {
            handle: 5,
            bytes: 385 * 4,
        };
        let args = pair_arguments(
            [
                Some((gate, view(1, gq, false), view(2, gq, true))),
                Some((down, view(3, dq, false), view(4, dq, true))),
            ],
            Some(active),
            384,
        )
        .unwrap();
        assert_eq!(args.0.len(), 16);
        assert_eq!(args.0[..5], [1, 3, 5, 2, 4].map(Argument::DeviceBuffer));
        assert_eq!(
            args.0[5..],
            [576, 160, 2, 1, 64, 5120, 18, 1, 4, 512, 384].map(Argument::U32)
        );
        let absent = pair_arguments(
            [None, Some((down, view(3, dq, false), view(4, dq, true)))],
            None,
            384,
        )
        .unwrap();
        assert_eq!(absent.0[0], Argument::NullDevicePointer);
        assert_eq!(absent.0[2], Argument::NullDevicePointer);
        assert_eq!(absent.0[5..10], [0; 5].map(Argument::U32));
        // Short or aliased views are refused.
        let short = View {
            bytes: dq.output_bytes().unwrap() - 1,
            ..view(4, dq, true)
        };
        assert!(
            pair_arguments([None, Some((down, view(3, dq, false), short))], None, 384).is_err()
        );
        assert!(
            pair_arguments(
                [None, Some((down, view(3, dq, false), view(3, dq, true)))],
                None,
                384
            )
            .is_err()
        );
    }

    #[test]
    fn scale_expand_registers_three_geometry_independent_entries() {
        let spec = cubin_spec("scale_expand");
        let generated = kernel_build::generate(&spec).unwrap();
        assert_eq!(spec.entries.len(), 4);
        assert_eq!(spec.entries[3].symbol, PAIR_SYMBOL);
        assert_eq!(spec.entries[3].params.len(), 16);
        assert!(generated.translation_unit.contains(PAIR_SYMBOL));
        for (entry, width) in spec.entries.iter().zip([1, 2, 4]) {
            assert_eq!(entry.symbol, symbol(width).unwrap());
            assert_eq!(entry.params.len(), 7);
            assert!(generated.translation_unit.contains(&entry.symbol));
        }
        assert!(symbol(8).is_err());
    }

    /// The tiled entry is the paired entry's ABI and grid over a first bank
    /// of whole 64-row by 4-group tiles, in its own cubin.
    #[test]
    fn tiled_pair_takes_the_paired_abi_over_whole_tiles() {
        let gate = PairBank {
            width: 1,
            rows: 576,
            groups: 160,
            projections: 2,
        };
        let down = PairBank {
            width: 4,
            rows: 5120,
            groups: 18,
            projections: 1,
        };
        let banks = [Some(gate), Some(down)];
        assert_eq!(
            pair_tiles_plan(banks, 384, 6).unwrap(),
            pair_plan(banks, 384, 6).unwrap()
        );
        for first in [
            None,
            Some(PairBank { rows: 600, ..gate }),
            Some(PairBank { groups: 18, ..gate }),
        ] {
            assert!(pair_tiles_plan([first, Some(down)], 384, 6).is_err());
        }
        let view = |handle, query: ScaleExpandQuery, output: bool| View {
            handle,
            bytes: if output {
                query.output_bytes().unwrap()
            } else {
                query.minimum_bank_bytes().unwrap()
            },
        };
        let (gq, dq) = (gate.query(384), down.query(384));
        let views = [
            Some((gate, view(1, gq, false), view(2, gq, true))),
            Some((down, view(3, dq, false), view(4, dq, true))),
        ];
        assert_eq!(
            pair_tiles_arguments(views, None, 384).unwrap(),
            pair_arguments(views, None, 384).unwrap()
        );
        assert!(pair_tiles_arguments([None, views[1]], None, 384).is_err());
        let spec = tiles_cubin_spec("tiles");
        let rows = cubin_spec("rows");
        let pair = &rows.entries[3];
        assert_eq!(spec.entries.len(), 1);
        let tiles = &spec.entries[0];
        assert_eq!(tiles.symbol, PAIR_TILES_SYMBOL);
        assert_eq!(tiles.params, pair.params);
        assert_eq!(
            tiles.body,
            pair.body.replace("::run_pair(", "::run_pair_tiles(")
        );
        assert!(
            kernel_build::generate(&spec)
                .unwrap()
                .translation_unit
                .contains(PAIR_TILES_SYMBOL)
        );
        assert!(
            !kernel_build::generate(&rows)
                .unwrap()
                .translation_unit
                .contains("scale_expand_tiles")
        );
    }
}
