//! The model's own kernels on SM120: [`OwnOps`] over the entries the build
//! compiles (`own_entries.rs`). Launches only plan and pack arguments.

use super::own_entries::{THREADS, gate_geometries, gate_symbol, gates, rope_symbol};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::dspark::{
    ConfidenceHead, ConfidenceHeadBuffers, MarkovStep, MarkovStepBuffers,
};
use model_deepseek41_contract::own_ops::engram::{EngramGate, EngramGateBuffers};
use model_deepseek41_contract::own_ops::rope::{
    MAX_POSITIONS, PAIRS, RopeCosSin, RopeCosSinBuffers, RopeRows, check_geometry,
};
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};

/// The model's own kernels as its build compiles them for one geometry.
#[derive(Clone, Copy, Debug)]
pub struct Sm120Own {
    geometry: &'static Geometry,
}

impl Sm120Own {
    /// The own kernels of `geometry`: its Engram gates (when it has Engram
    /// layers) and the RoPE rows, which its rotary parameters must match.
    pub fn new(geometry: &'static Geometry) -> Self {
        Self { geometry }
    }
}

impl OwnOps for Sm120Own {
    fn dspark_acceptance(
        &self,
        op: &model_deepseek41_contract::own_ops::acceptance::Acceptance,
        need: Need,
    ) -> Result<
        Prepared<model_deepseek41_contract::own_ops::acceptance::AcceptanceBuffers<View>>,
        Refused,
    > {
        if op.vocab != self.geometry.shape.vocab {
            return Err(Refused::Rule(
                "DSpark acceptance differs from the bound vocabulary".into(),
            ));
        }
        super::acceptance::prepare(*op, need)
    }

    fn dspark_host_acceptance(
        &self,
        op: &model_deepseek41_contract::own_ops::acceptance::Acceptance,
        need: Need,
    ) -> Result<
        Prepared<model_deepseek41_contract::own_ops::acceptance::AcceptanceBuffers<View>>,
        Refused,
    > {
        if op.vocab != self.geometry.shape.vocab {
            return Err(Refused::Rule(
                "DSpark acceptance differs from the bound vocabulary".into(),
            ));
        }
        super::acceptance::prepare_host(*op, need)
    }

    fn engram_hash(
        &self,
        op: &model_deepseek41_contract::own_ops::engram_hash::EngramHash,
        need: Need,
    ) -> Result<
        Prepared<model_deepseek41_contract::own_ops::engram_hash::EngramHashBuffers<View>>,
        Refused,
    > {
        let g = self.geometry;
        let e = g.shape.engram;
        if op.vocab != g.shape.vocab
            || op.ngram != e.max_ngram
            || op.heads != e.heads
            || op.layers != e.layers.len() as u32
        {
            return Err(Refused::Rule(
                "Engram hash differs from the bound geometry".into(),
            ));
        }
        super::engram_hash::prepare(*op, need)
    }

    fn dspark_ring_gather(
        &self,
        op: &model_deepseek41_contract::own_ops::ring_gather::RingGather,
        need: Need,
    ) -> Result<
        Prepared<model_deepseek41_contract::own_ops::ring_gather::RingGatherBuffers<View>>,
        Refused,
    > {
        super::ring_gather::prepare(*op, need)
    }

    fn dspark_allocated_acceptance(
        &self,
        op: &model_deepseek41_contract::own_ops::acceptance::Acceptance,
        need: Need,
    ) -> Result<
        Prepared<
            model_deepseek41_contract::own_ops::acceptance::allocated::AllocatedAcceptanceBuffers<
                View,
            >,
        >,
        Refused,
    > {
        super::allocated_acceptance::prepare(*op, need)
    }

    fn dspark_allocate(
        &self,
        op: &model_deepseek41_contract::own_ops::allocation::VerifyAllocate,
        need: Need,
    ) -> Result<
        Prepared<model_deepseek41_contract::own_ops::allocation::VerifyAllocateBuffers<View>>,
        Refused,
    > {
        super::allocate::prepare(*op, need)
    }

    fn dspark_markov_step(
        &self,
        op: &MarkovStep,
        need: Need,
    ) -> Result<Prepared<MarkovStepBuffers<View>>, Refused> {
        super::dspark::markov(self.geometry, op, need)
    }

    fn dspark_confidence(
        &self,
        op: &ConfidenceHead,
        need: Need,
    ) -> Result<Prepared<ConfidenceHeadBuffers<View>>, Refused> {
        super::dspark::confidence(self.geometry, op, need)
    }

    fn name(&self) -> &str {
        "sm120"
    }

    fn engram_gate(
        &self,
        op: &EngramGate,
        need: Need,
    ) -> Result<Prepared<EngramGateBuffers<View>>, Refused> {
        op.validate(need)?;
        let compiled = gate_geometries().any(|g| g.id == self.geometry.id);
        if !compiled || !gates(self.geometry).contains(op) {
            return Err(Refused::Rule(format!(
                "sm120 compiles no Engram gate of width {} over {} streams for geometry {}",
                op.width, op.streams, self.geometry.id
            )));
        }
        let symbol = gate_symbol(self.geometry.kernel_tag, op);
        // Every row is one block with a fixed reduction order, so a row's
        // bits depend on its inputs alone.
        let info = OpInfo::new(
            "sm120.deepseek41.engram-gate.tree256.v2",
            Invariance::any_rows(need.capacity),
            vec![(symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Box::new(Gate {
            op: *op,
            capacity: need.capacity,
            symbol,
            info,
        }))
    }

    fn rope_cos_sin(
        &self,
        op: &RopeCosSin,
        need: Need,
    ) -> Result<Prepared<RopeCosSinBuffers<View>>, Refused> {
        op.validate(need)?;
        check_geometry(self.geometry)?;
        let symbol = rope_symbol(op.table, op.rows);
        let info = OpInfo::new(
            "sm120.deepseek41.rope-cos-sin.f64-sincos.v1",
            Invariance::any_rows(need.capacity),
            vec![(symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Box::new(Rope {
            op: *op,
            capacity: need.capacity,
            symbol,
            info,
        }))
    }

    fn widen_rows(
        &self,
        op: &WidenRows,
        need: Need,
    ) -> Result<Prepared<WidenRowsBuffers<View>>, Refused> {
        super::widen::prepare(op, need, self.geometry)
    }
}

fn plan(grid: [u32; 2]) -> LaunchPlan {
    LaunchPlan {
        grid: [grid[0], grid[1], 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1; 3],
        // Each own kernel waits for the previous grid first.
        pdl: true,
    }
}

fn rows_of(counts: Counts, capacity: u32) -> Result<u32, Refusal> {
    if counts.rows == 0 || counts.rows > capacity {
        return Err(Refusal("own-op rows outside the prepared capacity"));
    }
    Ok(counts.rows)
}

/// One block per (row, stream) of the step's rows; rows from the device
/// live count on return at once.
struct Gate {
    op: EngramGate,
    capacity: u32,
    symbol: String,
    info: OpInfo,
}

impl PreparedOp<EngramGateBuffers<View>> for Gate {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &EngramGateBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows_of(counts, self.capacity)?;
        // A row's streams and the weights are both `[stream][width]` BF16.
        let row = self.op.row_elements() as u64 * 2;
        buffers.x.covers(u64::from(rows) * row)?;
        buffers.output.covers(u64::from(rows) * row)?;
        buffers
            .kv
            .covers(u64::from(rows) * self.op.kv_elements() as u64 * 4)?;
        buffers.q_weight.covers(row)?;
        buffers.k_weight.covers(row)?;
        buffers.live_rows.covers(4)?;
        let text = match (self.op.masked, buffers.text) {
            (true, Some(text)) => {
                text.covers(u64::from(rows))?;
                Argument::DeviceBuffer(text.handle)
            }
            (false, None) => Argument::NullDevicePointer,
            (true, None) => return Err(Refusal("a masked Engram gate reads text flags")),
            (false, Some(_)) => return Err(Refusal("an open Engram gate reads no text flags")),
        };
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows,
            plan: plan([rows, self.op.streams]),
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(buffers.x.handle),
                Argument::DeviceBuffer(buffers.kv.handle),
                Argument::DeviceBuffer(buffers.q_weight.handle),
                Argument::DeviceBuffer(buffers.k_weight.handle),
                text,
                Argument::DeviceBuffer(buffers.live_rows.handle),
                Argument::DeviceBuffer(buffers.output.handle),
            ]),
            collective: false,
            persistent: false,
        }])
    }
}

/// One thread per (row, pair), eight rows per block.
struct Rope {
    op: RopeCosSin,
    capacity: u32,
    symbol: String,
    info: OpInfo,
}

impl PreparedOp<RopeCosSinBuffers<View>> for Rope {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RopeCosSinBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows_of(counts, self.capacity)?;
        buffers.output.covers(RopeCosSin::output_bytes(rows))?;
        let blocks = (rows * PAIRS as u32).div_ceil(THREADS);
        let output = Argument::DeviceBuffer(buffers.output.handle);
        let arguments = match (self.op.rows, buffers.positions, buffers.live_rows) {
            (RopeRows::Range, None, None) => {
                if u64::from(buffers.first) + u64::from(rows) > u64::from(MAX_POSITIONS) {
                    return Err(Refusal("RoPE range rows outside the position domain"));
                }
                vec![Argument::U32(buffers.first), Argument::U32(rows), output]
            }
            (RopeRows::Positions, Some(positions), Some(live_rows)) => {
                positions.covers(u64::from(rows) * 4)?;
                live_rows.covers(4)?;
                vec![
                    Argument::DeviceBuffer(positions.handle),
                    Argument::DeviceBuffer(live_rows.handle),
                    output,
                ]
            }
            _ => {
                return Err(Refusal(
                    "RoPE rows bind positions and live rows only by position",
                ));
            }
        };
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows,
            plan: plan([blocks, 1]),
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }])
    }
}
