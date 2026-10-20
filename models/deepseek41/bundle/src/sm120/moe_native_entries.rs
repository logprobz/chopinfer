//! The opt-in native MoE prefill entries (`kernels_sm120::moe::native_grouped`)
//! for Flash TP4's 384-expert bank: one serving tile per projection, shared
//! by the build script, the bundle and the qualification lab.
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::native_grouped::{self as native, Shape, Tile};
use model_deepseek41_contract::{geometry::FLASH, moe};
pub const FAMILY: &str = "native_moe_prefill";
pub const CUBIN: &str = "native_moe_prefill_deepseek41_sm120";
/// Entries compile for the largest prefill bucket; smaller steps launch
/// the same arithmetic with fewer live rows.
pub const CAPACITY: u32 = 8192;
pub const GATE_UP_TILE: Tile = native::GATE_UP_TILES[0];
pub const DOWN_TILE: Tile = native::DOWN_TILES[0];

/// The target bank's expert set (384 experts, top 6).
pub fn target_set() -> moe::Experts {
    moe::expert_sets(&FLASH)
        .into_iter()
        .find(|set| set.count == FLASH.shape.moe.experts)
        .expect("Flash's target expert bank")
}

pub fn shape(kind: GatedMlpProjection) -> Shape {
    let query = moe::projection(&FLASH, 4, target_set(), kind, CAPACITY).expect("Flash TP4 MoE");
    Shape::from_query(&query, CAPACITY).expect("native tiles serve Flash's experts")
}

pub fn tile(kind: GatedMlpProjection) -> Tile {
    match kind {
        GatedMlpProjection::GateUp => GATE_UP_TILE,
        GatedMlpProjection::Down => DOWN_TILE,
    }
}

pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    let instances =
        [GatedMlpProjection::GateUp, GatedMlpProjection::Down].map(|k| (shape(k), tile(k)));
    vec![native::cubin_spec(CUBIN, &instances, false).expect("native MoE serving cubin")]
}
