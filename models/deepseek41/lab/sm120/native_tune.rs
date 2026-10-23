//! The native MoE tuning lab cubin: the served gate/up and down tiles'
//! inner-loop variants (`native_grouped::Tune`: the bitwise `lean` and
//! `no_nan` loops and four timing ablations), the NaN scans, two-CTA-per-SM
//! tiles of both projections (pack, served, lean and no_nan loops), and the
//! qualification entries of the binding candidates, for the target bank.
use super::family::serving;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::native_grouped::{self as native, Tile, Tune, Variant};

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

/// The bitwise candidates the qualification runs as tile variants: the
/// lean loop of the served gate/up tile and the two-CTA lean down tile.
pub fn candidates() -> [(GatedMlpProjection, Tile, Tune); 2] {
    [
        (
            GatedMlpProjection::GateUp,
            served(GatedMlpProjection::GateUp),
            Tune::Lean,
        ),
        (
            GatedMlpProjection::Down,
            paired(GatedMlpProjection::Down),
            Tune::Lean,
        ),
    ]
}

/// The variant a candidate is qualified with: gate/up's raw F32 sums, the
/// down's routed rows.
pub fn candidate_variant(kind: GatedMlpProjection) -> Variant {
    match kind {
        GatedMlpProjection::GateUp => Variant::Raw,
        GatedMlpProjection::Down => Variant::Serving,
    }
}

/// The paired tile's tuned loops.
pub const PAIRED_TUNES: [Tune; 2] = [Tune::Lean, Tune::NoNan];

pub fn cubins() -> Vec<kernel_build::CubinSpec> {
    let mut entries = Vec::new();
    for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
        let shape = serving::shape(kind);
        entries.extend(
            native::tune_entries(&shape, &served(kind), &Tune::ALL, Variant::Serving)
                .expect("tune entries"),
        );
        entries.push(native::nan_scan_entry(&shape));
        // The paired compute entries bound their registers for two CTAs per
        // SM; the build refuses them if that spills.
        let mut pair = native::entries(&shape, &paired(kind), false).expect("paired tile entries");
        pair.extend(
            native::tune_entries(&shape, &paired(kind), &PAIRED_TUNES, Variant::Serving)
                .expect("paired tuned entries"),
        );
        for e in &mut pair[1..] {
            e.min_blocks = 2;
        }
        entries.extend(pair);
    }
    // The candidates' qualification entries beside the timed ones.
    for (kind, tile, tune) in candidates() {
        let variant = candidate_variant(kind);
        if variant != Variant::Serving {
            let shape = serving::shape(kind);
            entries.extend(
                native::tune_entries(&shape, &tile, &[tune], variant).expect("candidate entry"),
            );
        }
    }
    let mut spec = native::cubin_spec(CUBIN, &[], false).expect("tune cubin");
    spec.entries = entries;
    vec![spec]
}
