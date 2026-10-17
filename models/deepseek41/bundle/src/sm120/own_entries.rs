//! The model's own SM120 entries as data: the Engram gate shapes and RoPE
//! tables the build compiles, their symbols and launch shape. The build
//! script compiles them (`own_cubin.rs`); `own_ops.rs` launches them.

use model_deepseek41_contract::own_ops::engram::EngramGate;
use model_deepseek41_contract::own_ops::rope::{RopeRows, RopeTable};

/// The `kernels.lock.d` family of the own-op cubin.
pub const FAMILY: &str = "deepseek41_own";
/// The own-op cubin.
pub const CUBIN: &str = "deepseek41_own_sm120";
/// Threads of every own-op block.
pub const THREADS: u32 = 256;

/// The Engram gate shapes (width, streams) compiled, each masked and not.
pub const GATE_SHAPES: [(u32, u32); 1] = [(5120, 4)];

/// The symbol of a compiled gate.
pub fn gate_symbol(op: &EngramGate) -> String {
    let mask = if op.masked { "masked" } else { "open" };
    format!(
        "chop_deepseek41_engram_gate_w{}_s{}_{mask}_v1",
        op.width, op.streams
    )
}

/// Whether the build compiles `op`.
pub fn gate_compiled(op: &EngramGate) -> bool {
    GATE_SHAPES.contains(&(op.width, op.streams))
}

/// Every compiled gate.
pub fn gates() -> Vec<EngramGate> {
    GATE_SHAPES
        .iter()
        .flat_map(|&(width, streams)| {
            [true, false].map(|masked| EngramGate {
                width,
                streams,
                masked,
            })
        })
        .collect()
}

/// The symbol of a RoPE table's row kernel.
pub fn rope_symbol(table: RopeTable, rows: RopeRows) -> String {
    let rows = match rows {
        RopeRows::Range => "range",
        RopeRows::Positions => "positions",
    };
    format!("chop_deepseek41_rope_{}_{rows}_v1", table.tag())
}
