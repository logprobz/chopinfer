//! Exact low-shared-memory rework, with a direct-load control and L2 hints.
//!
//! The direct gate queues also have builds reading their bank in 64-row by
//! 128-K tiles ([`ROW_TILES_64X128`]); each lane loads the same bytes, so
//! their chains and partials are the row-major queues'.
//!
//! [`ROW_TILES_64X128`]: kernel_core::precision::layout::ROW_TILES_64X128
use super::{
    floor::{Entry, Stage},
    row_bias, streamed,
};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Loads {
    Direct,
    Prefetch,
}
impl Loads {
    pub fn tag(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Prefetch => "prefetch",
        }
    }
}
pub fn symbol(e: &Entry, s: Stage, loads: Loads) -> String {
    e.symbol(s)
        .replace("_floor_", &format!("_bandwidth_{}_", loads.tag()))
}
pub fn plan(e: &Entry, s: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
    if matches!(
        s,
        Stage::GateSmall | Stage::GateWide | Stage::DownSmall | Stage::DownWide
    ) {
        let p = if matches!(s, Stage::GateSmall | Stage::GateWide) {
            &e.gate
        } else {
            &e.down
        };
        let mut plan = streamed::plan(
            p,
            if matches!(s, Stage::GateSmall | Stage::DownSmall) {
                streamed::Stage::Small
            } else {
                streamed::Stage::Wide
            },
            rows,
        )?;
        plan.workspace_bytes = e.workspace_bytes(rows)?;
        return Ok(plan);
    }
    e.plan(s, rows)
}
pub fn entry(e: &Entry, s: Stage, loads: Loads) -> EntrySpec {
    let mut spec = e.entry(s);
    spec.symbol = symbol(e, s, loads);
    if matches!(
        s,
        Stage::GateSmall | Stage::GateWide | Stage::DownSmall | Stage::DownWide
    ) {
        spec.body = spec.body.replace(
            "chop::moe::floor::compute",
            if loads == Loads::Direct {
                "chop::moe::streamed::compute"
            } else {
                "chop::moe::bandwidth::compute"
            },
        );
        if loads == Loads::Prefetch {
            spec.body = spec
                .body
                .replace(",8>(", ",8,2>(")
                .replace(",64>(", ",64,2>(");
        }
    } else if s == Stage::GateFinish {
        spec.body = spec.body.replace(
            "chop::moe::floor::gate_finish",
            "chop::moe::bandwidth::gate_finish",
        );
    }
    spec
}
/// The gate stages with a row-tiled build, small then wide.
pub const TILED: [Stage; 2] = [Stage::GateSmall, Stage::GateWide];

/// The direct gate queue reading its bank in row tiles
/// (`row-tiles-64x128-groups`).
pub fn tiled_symbol(e: &Entry, s: Stage) -> String {
    symbol(e, s, Loads::Direct).replace("_bandwidth_direct_", "_bandwidth_tiles_")
}

/// The direct gate small or wide queue's ABI and launch with its bank in
/// row tiles: the same operands, read at the tiled addresses.
pub fn tiled_entry(e: &Entry, s: Stage) -> Result<EntrySpec, Refusal> {
    let rows = match s {
        Stage::GateSmall => 8,
        Stage::GateWide => 64,
        _ => return Err(Refusal("only the gate queues read a row-tiled bank")),
    };
    let gate = &e.gate.query.projection.gemm;
    if gate.k % 128 != 0 || e.gate.columns() % 64 != 0 {
        return Err(Refusal("row tiles need whole 64-row by 128-K tiles"));
    }
    let mut spec = entry(e, s, Loads::Direct);
    let call = format!(",{rows}>(input,");
    assert_eq!(
        spec.body.matches("chop::moe::streamed::compute<").count(),
        1
    );
    assert_eq!(spec.body.matches(&call).count(), 1);
    spec.symbol = tiled_symbol(e, s);
    spec.body = spec.body.replace(&call, &format!(",{rows},true>(input,"));
    Ok(spec)
}

pub fn router(e: &Entry) -> row_bias::Router {
    row_bias::Router {
        experts: e.experts(),
        top_k: e.top_k(),
        capacity: 64,
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    let stages = [
        Stage::GateSmall,
        Stage::GateWide,
        Stage::GateFinish,
        Stage::DownSmall,
        Stage::DownWide,
    ];
    super::spec(
        name.into(),
        "chained_bandwidth.cuh",
        entries
            .iter()
            .flat_map(|e| {
                [Loads::Direct, Loads::Prefetch]
                    .into_iter()
                    .flat_map(move |l| stages.into_iter().map(move |s| entry(e, s, l)))
            })
            .collect(),
    )
}

/// [`cubin_spec`] and, after its entries, the tiled direct gate queues of
/// each bank in `tiled`.
pub fn tiled_cubin_spec(
    name: &str,
    entries: &[Entry],
    tiled: &[Entry],
) -> Result<CubinSpec, Refusal> {
    let mut spec = cubin_spec(name, entries);
    for e in tiled {
        for s in TILED {
            spec.entries.push(tiled_entry(e, s)?);
        }
    }
    Ok(spec)
}

/// The gate finisher with the same bits and fewer serial L2 round trips
/// (`gate_finish_loads.cuh`): every K-block partial and the route weight
/// load before the first add. Same parameters and launch as the bandwidth
/// finisher ([`plan`] with [`Stage::GateFinish`]).
pub fn finish_loads_symbol(e: &Entry) -> String {
    symbol(e, Stage::GateFinish, Loads::Direct).replace("_gate_finish_v1", "_gate_finish_loads_v1")
}
pub fn finish_loads_entry(e: &Entry) -> EntrySpec {
    let mut spec = entry(e, Stage::GateFinish, Loads::Direct);
    spec.symbol = finish_loads_symbol(e);
    let from = "chop::moe::bandwidth::gate_finish<";
    assert_eq!(spec.body.matches(from).count(), 1);
    spec.body = spec
        .body
        .replace(from, "chop::moe::bandwidth_loads::gate_finish<");
    spec
}
pub fn finish_loads_cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    super::spec(
        name.into(),
        "gate_finish_loads.cuh",
        entries.iter().map(finish_loads_entry).collect(),
    )
}
