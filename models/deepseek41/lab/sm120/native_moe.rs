//! The native MoE prefill lab cubin: every tile variant of both Flash TP4
//! expert banks, the raw and omit-first controls and the instruction probe.
//! Serving compiles only its two tiles of the target bank
//! (`bundle/src/sm120/moe_native_entries.rs`); the lab compiles the same
//! entries under the same symbols, so their machine code can be compared.
#[path = "../../bundle/src/sm120/moe_native_entries.rs"]
pub mod serving;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::native_grouped::{self as native, Shape, Tile};
use model_deepseek41_contract::{geometry::FLASH, moe};
pub const FAMILY: &str = "native_moe_prefill_lab";
pub const CUBIN: &str = "native_moe_prefill_lab_deepseek41_sm120";
pub const KINDS: [GatedMlpProjection; 2] = [GatedMlpProjection::GateUp, GatedMlpProjection::Down];

pub fn shape(set: moe::Experts, kind: GatedMlpProjection) -> Shape {
    let query = moe::projection(&FLASH, 4, set, kind, serving::CAPACITY).expect("Flash TP4 MoE");
    Shape::from_query(&query, serving::CAPACITY).expect("native tiles serve Flash's experts")
}

pub fn tiles(kind: GatedMlpProjection) -> &'static [Tile] {
    match kind {
        GatedMlpProjection::GateUp => &native::GATE_UP_TILES,
        GatedMlpProjection::Down => &native::DOWN_TILES,
    }
}

pub fn instances() -> Vec<(Shape, Tile)> {
    moe::expert_sets(&FLASH)
        .into_iter()
        .flat_map(|set| KINDS.map(|kind| (shape(set, kind), kind)))
        .flat_map(|(shape, kind)| tiles(kind).iter().map(move |tile| (shape, *tile)))
        .collect()
}

pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    vec![native::cubin_spec(CUBIN, &instances(), true).expect("native MoE lab cubin")]
}
