//! Exact low-shared-memory rework, with a direct-load control and L2 hints.
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

/// Lab variant of the gate finisher with the same bits
/// (`gate_finish_loads.cuh`): every K-block partial and the route weight
/// load before the first add. Same parameters and launch as the served
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
