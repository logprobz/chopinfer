//! The direct queues of steps above 64 rows (`chained_wide.cuh`), where an
//! expert may take more than 64 routed rows, at most one per token.
//!
//! The generic local route writes the metadata (expert ranges padded to
//! whole 64-row tiles, sorted routes and the active list). A pack writes
//! the streamed queue table from its counts and active list and clears the
//! gate finisher's ticket of every token of the capacity. The absorbing
//! gate queue and the two-strip down queue run every expert eight rows at a
//! time, the finishers fold each route's partials in K order and the down
//! finisher adds the routes and the shared expert's partial. Each is the
//! 64-row direct path's entry ([`absorb`], [`lanes`], [`strip2`],
//! [`down_scale_tiles`], [`bandwidth`], [`floor`]) compiled at the larger
//! capacity, which changes only its bounds checks, so every chain, partial
//! and output is computed as at 64 rows.
//!
//! [`floor`]: super::floor
use super::{
    absorb, bandwidth, down_scale_tiles,
    floor::{Entry, Stage},
    lanes, strip2,
};
use crate::moe::ptr;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

/// The header of every wide entry.
pub const HEADER: &str = "chained_wide.cuh";

/// Offset widths of the tile-order encoded scale entries.
pub const SCALE_WIDTHS: [u32; 3] = [1, 2, 4];

fn wide(e: &Entry) -> Result<(), Refusal> {
    if e.capacity() <= 64 || !e.prefix.contains("_floor_") {
        return Err(Refusal(
            "wide direct queues need a floor entry above 64 tokens",
        ));
    }
    Ok(())
}

/// The pack: the floor entry's prefix with `_wide_` for `_floor_`.
pub fn pack_symbol(e: &Entry) -> String {
    format!("{}_pack_v1", e.prefix.replace("_floor_", "_wide_"))
}

/// The pack's launch: one warp.
pub fn pack_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    wide(e)?;
    let plan = LaunchPlan {
        grid: [1, 1, 1],
        block: [32, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: e.workspace_bytes(rows)?,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// The pack over the generic route's counts and active list, writing the
/// queue table in `scratch` and clearing `tickets`.
pub fn pack_entry(e: &Entry) -> Result<EntrySpec, Refusal> {
    wide(e)?;
    Ok(EntrySpec::new(
        pack_symbol(e),
        vec![
            ptr("counts", "const int32_t"),
            ptr("active", "const int32_t"),
            ptr("live", "const int32_t"),
            ptr("scratch", "void"),
            ptr("tickets", "uint32_t"),
        ],
        format!(
            "chop::moe::wide::pack<{}>(counts,active,live,scratch,tickets);",
            e.config(false)
        ),
        32,
        1,
    ))
}

/// The gate finisher's tickets: one word per token of the capacity, in
/// whole 256-byte blocks.
pub fn ticket_bytes(e: &Entry) -> u64 {
    (4 * u64::from(e.capacity())).next_multiple_of(256)
}

/// The absorbing gate queue over a gate/up bank in quad-lane row tiles,
/// with raw scale tiles (`None`) or a tile-order encoded scale bank of
/// `Some(width)`.
pub fn gate_entry(e: &Entry, width: Option<u32>) -> Result<EntrySpec, Refusal> {
    wide(e)?;
    match width {
        None => lanes::absorb_entry(e),
        Some(width) => lanes::absorb_scale_tiles_entry(e, width),
    }
}

/// The gate queue's launch: one item per routed expert bound, 64-column
/// strip and 128-K block, at most eight CTAs per SM.
pub fn gate_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    wide(e)?;
    absorb::plan(e, Stage::GateSmall, rows)
}

/// The gate finisher: the served direct finisher (`loads` false) or the one
/// issuing every partial load before its first add.
pub fn gate_finish_entry(e: &Entry, loads: bool) -> Result<EntrySpec, Refusal> {
    wide(e)?;
    Ok(if loads {
        bandwidth::finish_loads_entry(e)
    } else {
        bandwidth::entry(e, Stage::GateFinish, bandwidth::Loads::Direct)
    })
}

pub fn gate_finish_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    wide(e)?;
    bandwidth::plan(e, Stage::GateFinish, rows)
}

/// The two-strip down queue with raw scales (`None`) or a tile-order
/// encoded scale bank of `Some(width)`.
pub fn down_entry(e: &Entry, width: Option<u32>) -> Result<EntrySpec, Refusal> {
    wide(e)?;
    match width {
        None => Ok(strip2::entry(e)),
        Some(width) => down_scale_tiles::strip2_entry(e, width),
    }
}

pub fn down_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    wide(e)?;
    strip2::plan(e, rows)
}

/// The down finisher: each route's partials in K order, the routes in
/// selection order, then the shared expert's partial.
pub fn down_finish_entry(e: &Entry) -> Result<EntrySpec, Refusal> {
    wide(e)?;
    Ok(e.entry(Stage::DownFinish))
}

pub fn down_finish_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    wide(e)?;
    e.plan(Stage::DownFinish, rows)
}

/// Every wide entry of `e`: the pack, the quad-lane gate queue with raw
/// and encoded scales, both gate finishers, the down queue with raw and
/// encoded scales and the down finisher.
pub fn entries(e: &Entry) -> Result<Vec<EntrySpec>, Refusal> {
    let mut out = vec![pack_entry(e)?, gate_entry(e, None)?];
    for width in SCALE_WIDTHS {
        out.push(gate_entry(e, Some(width))?);
    }
    out.extend([gate_finish_entry(e, false)?, gate_finish_entry(e, true)?]);
    out.push(down_entry(e, None)?);
    for width in SCALE_WIDTHS {
        out.push(down_entry(e, Some(width))?);
    }
    out.push(down_finish_entry(e)?);
    Ok(out)
}

/// [`entries`] of each bank in `entries`, in one module.
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for e in entries {
        specs.extend(self::entries(e)?);
    }
    Ok(super::spec(name.into(), HEADER, specs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::chained::quantized::Projection;
    use kernel_core::contract::{
        ActivationFormat, DType, Epilogues, GemmQuery, GroupedGemm, OutputQuantization,
        RoutedCapacity, SwigluColumns, WeightFormat,
    };
    use kernel_core::gated_mlp::{GatedMlpProjection, GatedMlpProjectionQuery, RouteWeightPoint};
    use kernel_core::moe_chained::{ChainedK, ChainedMlpProjectionQuery};

    fn fixture(capacity: u32) -> Entry {
        let projection = |gate: bool| {
            let k = if gate { 5120 } else { 576 };
            let p = Projection::new(
                ChainedMlpProjectionQuery {
                    projection: GatedMlpProjectionQuery {
                        gemm: GemmQuery {
                            m_capacity: capacity * 6,
                            n: if gate { 1152 } else { 5120 },
                            k,
                            weight: WeightFormat::Mxfp4,
                            activation: ActivationFormat::Mxfp8,
                            output: if gate { DType::Bf16 } else { DType::F32 },
                            epilogues: Epilogues {
                                swiglu: gate.then_some(SwigluColumns::ConcatenatedHalves),
                                output_quantization: gate.then_some(OutputQuantization::Mxfp8),
                                ..Default::default()
                            },
                            grouped: Some(GroupedGemm {
                                experts: 384,
                                top_k: 6,
                                capacity: RoutedCapacity::TotalRows(capacity * 6),
                            }),
                        },
                        projection: if gate {
                            GatedMlpProjection::GateUp
                        } else {
                            GatedMlpProjection::Down
                        },
                        swiglu_clamp: None,
                        route_weight: Some(RouteWeightPoint::BeforeDown),
                    },
                    k: ChainedK::new(k).unwrap(),
                },
                64,
                capacity,
            )
            .unwrap();
            if capacity <= 64 {
                p.with_streamed_chains().unwrap()
            } else {
                assert!(p.clone().with_streamed_chains().is_err());
                p.with_direct_chains().unwrap()
            }
        };
        let prefix = if capacity == 64 {
            "chop_test_moe_floor_e384_k6".to_string()
        } else {
            format!("chop_test_moe_floor_t{capacity}_e384_k6")
        };
        Entry::new(
            projection(true),
            projection(false),
            1e-4_f32.to_bits(),
            prefix,
        )
        .unwrap()
    }

    /// Each wide entry is the 64-row direct entry it is built from, with
    /// only the capacity of its configuration and its symbol changed: the
    /// same parameters, launch bounds, residency, template and policies.
    #[test]
    fn wide_entries_change_only_the_capacity_and_symbol() {
        let narrow = fixture(64);
        for capacity in [65, 192, 256, 384] {
            let e = fixture(capacity);
            let config = |e: &Entry| [e.config(true), e.config(false)];
            let [gate64, down64] = config(&narrow);
            let [gate, down] = config(&e);
            assert_eq!(gate, gate64.replace(",64,64,", &format!(",64,{capacity},")));
            assert_eq!(down, down64.replace(",64,64,", &format!(",64,{capacity},")));
            let mut pairs = vec![
                (
                    gate_entry(&e, None).unwrap(),
                    lanes::absorb_entry(&narrow).unwrap(),
                ),
                (
                    gate_finish_entry(&e, false).unwrap(),
                    bandwidth::entry(&narrow, Stage::GateFinish, bandwidth::Loads::Direct),
                ),
                (
                    gate_finish_entry(&e, true).unwrap(),
                    bandwidth::finish_loads_entry(&narrow),
                ),
                (down_entry(&e, None).unwrap(), strip2::entry(&narrow)),
                (
                    down_finish_entry(&e).unwrap(),
                    narrow.entry(Stage::DownFinish),
                ),
            ];
            for width in SCALE_WIDTHS {
                pairs.push((
                    gate_entry(&e, Some(width)).unwrap(),
                    lanes::absorb_scale_tiles_entry(&narrow, width).unwrap(),
                ));
                pairs.push((
                    down_entry(&e, Some(width)).unwrap(),
                    down_scale_tiles::strip2_entry(&narrow, width).unwrap(),
                ));
            }
            for (wide, served) in &pairs {
                let renamed = served
                    .symbol
                    .replace("_e384_k6_", &format!("_t{capacity}_e384_k6_"));
                assert_eq!(wide.symbol, renamed);
                assert_eq!(
                    wide.body,
                    served
                        .body
                        .replace(&served.symbol, &wide.symbol)
                        .replace(&gate64, &gate)
                        .replace(&down64, &down)
                );
                assert_eq!(
                    (&wide.params, wide.max_threads, wide.min_blocks),
                    (&served.params, served.max_threads, served.min_blocks)
                );
            }
            let symbols = entries(&e)
                .unwrap()
                .into_iter()
                .map(|s| s.symbol)
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(symbols.len(), 12);
            assert!(symbols.contains(&pack_symbol(&e)));
            assert_eq!(
                pack_symbol(&e),
                format!("chop_test_moe_wide_t{capacity}_e384_k6_pack_v1")
            );
            assert_eq!(
                ticket_bytes(&e),
                (4 * u64::from(capacity)).next_multiple_of(256)
            );
        }
        assert!(entries(&narrow).is_err());
        assert!(pack_plan(&narrow, 64).is_err());
    }

    /// The wide launches are the 64-row direct launches at each row count,
    /// and refuse rows above the capacity.
    #[test]
    fn wide_plans_follow_the_direct_plans_through_the_capacity() {
        let e = fixture(192);
        let narrow = fixture(64);
        for rows in [1, 33, 64] {
            assert_eq!(
                gate_plan(&e, rows).unwrap().grid,
                absorb::plan(&narrow, Stage::GateSmall, rows).unwrap().grid
            );
            assert_eq!(
                down_plan(&e, rows).unwrap().grid,
                strip2::plan(&narrow, rows).unwrap().grid
            );
            assert_eq!(
                gate_finish_plan(&e, rows).unwrap().grid,
                bandwidth::plan(&narrow, Stage::GateFinish, rows)
                    .unwrap()
                    .grid
            );
            assert_eq!(
                down_finish_plan(&e, rows).unwrap().grid,
                narrow.plan(Stage::DownFinish, rows).unwrap().grid
            );
            assert_eq!(e.workspace_bytes(rows), narrow.workspace_bytes(rows));
        }
        for rows in [65, 128, 192] {
            assert_eq!(gate_plan(&e, rows).unwrap().grid, [1504, 1, 1]);
            assert_eq!(down_plan(&e, rows).unwrap().grid, [1504, 1, 1]);
            assert_eq!(gate_finish_plan(&e, rows).unwrap().grid, [5, rows * 6, 1]);
            assert_eq!(down_finish_plan(&e, rows).unwrap().grid, [40, rows, 1]);
            let pack = pack_plan(&e, rows).unwrap();
            assert_eq!((pack.grid, pack.block), ([1, 1, 1], [32, 1, 1]));
            assert_eq!(pack.workspace_bytes, e.workspace_bytes(rows).unwrap());
            assert_eq!(
                e.workspace_bytes(rows).unwrap(),
                6400 + u64::from(rows) * 6 * 2 * 40 * 576 * 4
            );
        }
        for plan in [
            gate_plan(&e, 193),
            down_plan(&e, 193),
            gate_finish_plan(&e, 193),
            down_finish_plan(&e, 193),
            pack_plan(&e, 193),
            pack_plan(&e, 0),
        ] {
            assert!(plan.is_err());
        }
    }
}
