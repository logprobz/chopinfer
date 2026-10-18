//! The model's own SM120 entries as data: the RoPE rows, compiled once, and
//! the Engram gate of each compiled geometry with Engram layers, in a cubin
//! of its own named by the geometry's kernel tag, so that adding a geometry
//! adds lock records and changes none. The build script compiles them
//! (`own_cubin.rs`); `own_ops.rs` launches them.

use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry};
use model_deepseek41_contract::own_ops::engram::EngramGate;
use model_deepseek41_contract::own_ops::rope::{RopeRows, RopeTable};

/// The `kernels.lock.d` family of the own-op cubins.
pub const FAMILY: &str = "deepseek41_own";
/// The RoPE rows' cubin, shared by every geometry.
pub const ROPE_CUBIN: &str = "deepseek41_rope_sm120";
/// Threads of every own-op block.
pub const THREADS: u32 = 256;

/// The compiled geometries that have Engram layers.
pub fn gate_geometries() -> impl Iterator<Item = &'static Geometry> {
    GEOMETRIES.iter().filter(|g| has_gates(g))
}

/// Whether `geometry` has Engram layers, and so a gate cubin.
pub fn has_gates(geometry: &Geometry) -> bool {
    !geometry.shape.engram.layers.is_empty()
}

/// The cubin of `geometry`'s Engram gates.
pub fn gate_cubin(geometry: &Geometry) -> String {
    format!("{}_engram_gate_sm120", geometry.kernel_tag)
}

/// The gates `geometry`'s cubin holds: masked and open.
pub fn gates(geometry: &Geometry) -> [EngramGate; 2] {
    [true, false].map(|masked| EngramGate::of(geometry, masked))
}

/// The symbol of a gate in the cubin of the geometry tagged `tag`.
pub fn gate_symbol(tag: &str, op: &EngramGate) -> String {
    let mask = if op.masked { "masked" } else { "open" };
    format!(
        "chop_{tag}_engram_gate_w{}_s{}_{mask}_v2",
        op.width, op.streams
    )
}

/// The symbol of a RoPE table's row kernel.
pub fn rope_symbol(table: RopeTable, rows: RopeRows) -> String {
    let rows = match rows {
        RopeRows::Range => "range",
        RopeRows::Positions => "positions",
    };
    format!("chop_deepseek41_rope_{}_{rows}_v1", table.tag())
}
