//! Keep the small tile's arithmetic while targeting eight resident CTAs.
use super::{quantized::Projection, streamed};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};

pub fn symbol(p: &Projection) -> String {
    format!("{}_small_resident8_v1", p.symbol.trim_end_matches("_v1"))
}
pub fn plan(p: &Projection, rows: u32) -> Result<LaunchPlan, Refusal> {
    if !p.gate_up() || p.row_tile != 64 || p.tokens_capacity != 64 || rows > 32 {
        return Err(Refusal(
            "resident gate requires the 64-row ABI and at most 32 live rows",
        ));
    }
    streamed::plan(p, streamed::Stage::Small, rows)
}
pub fn entry(p: &Projection) -> EntrySpec {
    assert!(plan(p, 32).is_ok());
    let mut e = streamed::entries(p)
        .into_iter()
        .find(|e| e.symbol == streamed::symbol(p, streamed::Stage::Small))
        .unwrap();
    e.symbol = symbol(p);
    e.min_blocks = 8;
    // The builder rejects spills. Only register allocation and instruction
    // scheduling may change; operands, chains and partial addresses do not.
    e
}
pub fn cubin_spec(name: &str, projections: &[Projection]) -> CubinSpec {
    super::spec(
        name.into(),
        "chained_streamed.cuh",
        projections.iter().map(entry).collect(),
    )
}
