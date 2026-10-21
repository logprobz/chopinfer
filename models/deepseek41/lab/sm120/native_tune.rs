//! The native MoE tuning lab cubin: the served gate/up and down tiles'
//! inner-loop variants (`native_grouped::Tune`: the bitwise `lean` loop and
//! four timing ablations) and two-CTA-per-SM tiles of both projections
//! (pack, served loop and lean loop) for the target bank.
use super::family::serving;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::native_grouped::{self as native, Tile, Tune};

pub const FAMILY: &str = "native_tune_lab";
pub const CUBIN: &str = "native_tune_lab_deepseek41_sm120";

/// The served tile of `kind` at 8192 rows.
pub fn served(kind: GatedMlpProjection) -> Tile {
    serving::tile(kind)
}

/// A 64-row tile of `kind` whose shared memory and bounded registers fit
/// two CTAs per SM.
pub fn paired(kind: GatedMlpProjection) -> Tile {
    match kind {
        GatedMlpProjection::GateUp => Tile {
            row_warps: 1,
            col_warps: 6,
            stages: 2,
        },
        GatedMlpProjection::Down => Tile {
            row_warps: 1,
            col_warps: 4,
            stages: 2,
        },
    }
}

pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    let mut entries = Vec::new();
    for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
        let shape = serving::shape(kind);
        entries
            .extend(native::tune_entries(&shape, &served(kind), &Tune::ALL).expect("tune entries"));
        // The paired compute entries bound their registers for two CTAs per
        // SM; the build refuses them if that spills.
        let mut pair = native::entries(&shape, &paired(kind), false).expect("paired tile entries");
        pair.extend(
            native::tune_entries(&shape, &paired(kind), &[Tune::Lean]).expect("paired lean entry"),
        );
        for e in &mut pair[1..] {
            e.min_blocks = 2;
        }
        entries.extend(pair);
    }
    let mut spec = native::cubin_spec(CUBIN, &[], false).expect("tune cubin");
    spec.entries = entries;
    vec![spec]
}
