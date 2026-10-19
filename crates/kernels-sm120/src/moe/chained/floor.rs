//! Experimental exact MoE pipeline. Metadata completes before any weight
//! prefetch; gate and down reuse chain scratch only behind a PDL wait.
//! The shared expert must use disjoint scratch and join before DownFinish.
use super::{quantized::Projection, streamed};
use crate::moe::{SM_COUNT, ptr};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    RouterPrepare,
    Router,
    Metadata,
    GateSmall,
    GateWide,
    GateFinish,
    DownSmall,
    DownWide,
    DownFinish,
    DownNegative,
}
pub const STAGES: [Stage; 10] = [
    Stage::RouterPrepare,
    Stage::Router,
    Stage::Metadata,
    Stage::GateSmall,
    Stage::GateWide,
    Stage::GateFinish,
    Stage::DownSmall,
    Stage::DownWide,
    Stage::DownFinish,
    Stage::DownNegative,
];

#[derive(Clone, Debug)]
pub struct Entry {
    pub gate: Projection,
    pub down: Projection,
    pub floor_bits: u32,
    pub prefix: String,
}
impl Entry {
    pub fn new(
        gate: Projection,
        down: Projection,
        floor_bits: u32,
        prefix: String,
    ) -> Result<Self, Refusal> {
        for p in [&gate, &down] {
            Projection::new(p.query, p.row_tile, p.tokens_capacity)?.with_streamed_chains()?;
        }
        let a = gate.query.projection.gemm;
        let b = down.query.projection.gemm;
        let g = a.grouped.ok_or(Refusal("floor requires grouped experts"))?;
        if !gate.gate_up()
            || down.gate_up()
            || !gate.streamed
            || !down.streamed
            || a.grouped != b.grouped
            || gate.tokens_capacity != down.tokens_capacity
            || gate.tokens_capacity != 64
            || gate.max_rows != 64
            || down.max_rows != 64
            // The split router reserves 40 KiB, plus four expert rows and
            // CUDA static overhead, within the 48 KiB static allocation.
            || g.experts > 448
            || g.experts % 32 != 0
            || g.top_k > 8
            || g.top_k == 0
            || gate.columns() != b.k
            || down.columns() != a.k
            || !f32::from_bits(floor_bits).is_finite()
            || f32::from_bits(floor_bits) <= 0.0
        {
            return Err(Refusal(
                "floor requires compatible 64-row streamed projections",
            ));
        }
        Ok(Self {
            gate,
            down,
            floor_bits,
            prefix,
        })
    }
    pub fn experts(&self) -> u32 {
        self.gate.query.projection.gemm.grouped.unwrap().experts
    }
    pub fn top_k(&self) -> u32 {
        self.gate.query.projection.gemm.grouped.unwrap().top_k
    }
    pub fn hidden(&self) -> u32 {
        self.down.columns()
    }
    pub fn intermediate(&self) -> u32 {
        self.gate.columns()
    }
    pub fn symbol(&self, stage: Stage) -> String {
        let suffix = match stage {
            Stage::RouterPrepare => "router_prepare",
            Stage::Router => "router",
            Stage::Metadata => "metadata",
            Stage::GateSmall => "gate_small",
            Stage::GateWide => "gate_wide",
            Stage::GateFinish => "gate_finish",
            Stage::DownSmall => "down_small",
            Stage::DownWide => "down_wide",
            Stage::DownFinish => "down_finish",
            Stage::DownNegative => "down_omit_first",
        };
        format!("{}_{suffix}_v1", self.prefix)
    }
    pub fn workspace_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.rows(rows)?;
        Ok(streamed::workspace_bytes(&self.gate, rows)?
            .max(streamed::workspace_bytes(&self.down, rows)?))
    }
    pub fn router_workspace_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.rows(rows)?;
        let rows = u64::from(rows);
        let tiles = u64::from(self.experts().div_ceil(16)) * rows.div_ceil(8);
        Ok((4 * (rows + tiles)).next_multiple_of(16)
            + (rows * u64::from(self.hidden()) * 6).next_multiple_of(16)
            + rows * u64::from(self.experts()) * u64::from(self.hidden().div_ceil(128)) * 4)
    }
    fn rows(&self, rows: u32) -> Result<(), Refusal> {
        if rows == 0 || rows > 64 {
            return Err(Refusal("floor supports 1..64 prepared rows"));
        }
        Ok(())
    }
    pub fn plan(&self, stage: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.rows(rows)?;
        let (grid, threads, workspace) = match stage {
            Stage::RouterPrepare => ([rows, 1, 1], 128, self.router_workspace_bytes(rows)?),
            Stage::Router => (
                [
                    self.experts().div_ceil(16),
                    rows.div_ceil(8),
                    self.hidden().div_ceil(128).div_ceil(4),
                ],
                128,
                self.router_workspace_bytes(rows)?,
            ),
            Stage::Metadata => (
                [1, 1, 1],
                self.experts().max((64 * self.top_k()).next_multiple_of(32)),
                self.workspace_bytes(rows)?,
            ),
            Stage::GateSmall | Stage::GateWide | Stage::DownSmall | Stage::DownWide => {
                let p = if matches!(stage, Stage::GateSmall | Stage::GateWide) {
                    &self.gate
                } else {
                    &self.down
                };
                let mut plan = streamed::plan(
                    p,
                    if matches!(stage, Stage::GateSmall | Stage::DownSmall) {
                        streamed::Stage::Small
                    } else {
                        streamed::Stage::Wide
                    },
                    rows,
                )?;
                plan.grid[0] = plan.grid[0].min(SM_COUNT * 4);
                plan.workspace_bytes = self.workspace_bytes(rows)?;
                return Ok(plan);
            }
            Stage::GateFinish => (
                [self.intermediate().div_ceil(128), rows * self.top_k(), 1],
                128,
                self.workspace_bytes(rows)?,
            ),
            Stage::DownFinish | Stage::DownNegative => (
                [self.hidden().div_ceil(128), rows, 1],
                128,
                self.workspace_bytes(rows)?,
            ),
        };
        let plan = LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: workspace,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }
    fn config(&self, gate: bool) -> String {
        let p = if gate { &self.gate } else { &self.down };
        let clamp = p.query.projection.swiglu_clamp.map_or_else(
            || "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
            |c| {
                format!(
                    "chop::moe::clamp::SwigluClamp<true,{}u,{}u>",
                    c.gate_max_bits, c.up_limit_bits
                )
            },
        );
        format!(
            "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{},{},{},{},{},64,64,{}>",
            if gate { "gate_up" } else { "down" },
            self.experts(),
            self.top_k(),
            p.columns(),
            p.query.projection.gemm.k,
            clamp
        )
    }
    pub fn entry(&self, stage: Stage) -> EntrySpec {
        let e = self.experts();
        let t = self.top_k();
        let h = self.hidden();
        let gate = self.config(true);
        let down = self.config(false);
        let live = ptr("live", "const int32_t");
        let scratch = ptr("scratch", "void");
        let tickets = ptr("tickets", "uint32_t");
        let rows = EntryParam::U32 {
            name: "rows".into(),
        };
        let (params, call) = match stage {
            Stage::RouterPrepare => (
                vec![ptr("input", "const float"), live, rows, scratch, tickets],
                format!(
                    "router_prepare<{e},{h}>(input,reinterpret_cast<const uint32_t*>(live),rows,scratch,tickets)"
                ),
            ),
            Stage::Router => (
                vec![
                    ptr("input", "const float"),
                    ptr("weight", "const uint16_t"),
                    scratch,
                    ptr("logits", "float"),
                    live,
                    rows,
                    ptr("domain", "const uint32_t"),
                    tickets,
                    ptr("bias", "const float"),
                    ptr("alternate_bias", "const float"),
                    ptr("bias_select", "const uint8_t"),
                    ptr("scores", "float"),
                    EntryParam::U32 {
                        name: "scaling".into(),
                    },
                    ptr("ids", "int32_t"),
                    ptr("weights", "float"),
                ],
                format!(
                    "router<{e},{t},{h}>(input,weight,scratch,logits,reinterpret_cast<const uint32_t*>(live),rows,domain,tickets,bias,alternate_bias,bias_select,scores,__uint_as_float(scaling),ids,weights)"
                ),
            ),
            Stage::Metadata => (
                vec![
                    ptr("ids", "const int32_t"),
                    ptr("counts", "int32_t"),
                    ptr("offsets", "int32_t"),
                    ptr("sorted", "int32_t"),
                    ptr("padded", "int32_t"),
                    live,
                    ptr("active", "int32_t"),
                    scratch,
                    tickets,
                ],
                format!(
                    "metadata<{e},{t},64>(ids,counts,offsets,sorted,padded,live,active,static_cast<int32_t*>(scratch),tickets)"
                ),
            ),
            Stage::GateSmall | Stage::GateWide | Stage::DownSmall | Stage::DownWide => {
                let c = if matches!(stage, Stage::GateSmall | Stage::GateWide) {
                    gate
                } else {
                    down
                };
                let m = if matches!(stage, Stage::GateSmall | Stage::DownSmall) {
                    8
                } else {
                    64
                };
                (
                    vec![
                        ptr("input", "const uint8_t"),
                        ptr("weight", "const uint8_t"),
                        ptr("weight_scales", "const uint8_t"),
                        ptr("scales", "const uint8_t"),
                        ptr("offsets", "const int32_t"),
                        ptr("sorted", "const int32_t"),
                        ptr("padded", "const int32_t"),
                        live,
                        scratch,
                    ],
                    format!(
                        "compute<{c},{m}>(input,weight,weight_scales,scales,offsets,sorted,padded,live,scratch)"
                    ),
                )
            }
            Stage::GateFinish => (
                vec![
                    scratch,
                    live,
                    ptr("weights", "const float"),
                    ptr("intermediate", "uint16_t"),
                    ptr("codes", "uint8_t"),
                    ptr("scales", "uint8_t"),
                    ptr("status", "uint32_t"),
                    tickets,
                ],
                format!(
                    "gate_finish<{gate},{}u>(scratch,live,weights,intermediate,codes,scales,status,tickets)",
                    self.floor_bits
                ),
            ),
            Stage::DownFinish | Stage::DownNegative => (
                vec![
                    scratch,
                    live,
                    ptr("ids", "const int32_t"),
                    ptr("shared", "const float"),
                    ptr("routed", "float"),
                    ptr("output", "float"),
                ],
                format!(
                    "down_finish<{down},{}>(scratch,live,ids,shared,routed,output)",
                    stage == Stage::DownNegative
                ),
            ),
        };
        EntrySpec::new(
            self.symbol(stage),
            params,
            format!("chop::moe::floor::{call};"),
            self.plan(stage, 64).unwrap().block[0],
            1,
        )
    }
}

pub fn cubins(prefix: &str, entries: &[Entry]) -> Vec<CubinSpec> {
    [
        ("compute", "chained_floor.cuh", &STAGES[3..]),
        ("route", "route_floor.cuh", &STAGES[..3]),
    ]
    .map(|(suffix, header, stages)| {
        super::spec(
            format!("{prefix}_{suffix}_sm120"),
            header,
            entries
                .iter()
                .flat_map(|entry| stages.iter().map(|&stage| entry.entry(stage)))
                .collect(),
        )
    })
    .to_vec()
}
