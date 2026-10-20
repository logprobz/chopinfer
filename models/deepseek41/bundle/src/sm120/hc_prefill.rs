//! Measured prefill mixers. Expansion and other row counts keep their entries.
use kernel_build::CubinSpec;
use kernels_sm120::{
    Profile, ProfileNames,
    backend::Sm120,
    sinkhorn_hc::{Geometry as Mixer, prefill_warp as prefill},
};
use model_deepseek41_contract::{
    geometry::{FLASH, Geometry},
    hc,
    serving::PREFILL_TOKENS,
};

pub const FAMILY: &str = "sinkhorn_hc_prefill_measured";
pub const CUBIN: &str = "hc_prefill_warp_deepseek41_sm120";

pub fn qualified(g: &Geometry) -> bool {
    g == &FLASH
}

fn profile() -> Profile {
    Profile::synthetic(ProfileNames {
        name: "deepseek41",
        tag: FLASH.kernel_tag,
    })
}

fn mixer() -> Mixer {
    Mixer {
        mix: hc::mix(&FLASH),
        capacity: PREFILL_TOKENS,
    }
}

// Retain the complete qualified module so its cubin identity stays fixed.
fn entries(probe: bool) -> Vec<prefill::Entry> {
    let mut entries: Vec<_> = [(4, true), (8, true), (16, true), (8, false)]
        .into_iter()
        .map(|(rows, staged)| prefill::mix(profile(), mixer(), rows, staged, probe))
        .chain(
            [4, 8]
                .into_iter()
                .map(|chains| prefill::pipeline(profile(), mixer(), 8, chains, probe)),
        )
        .collect();
    if !probe {
        entries.extend(
            [
                (128, false, false),
                (256, false, false),
                (128, false, true),
                (256, false, true),
                (256, true, true),
            ]
            .into_iter()
            .map(|(threads, row_warps, broadcast)| {
                prefill::expand(profile(), mixer(), threads, row_warps, broadcast)
            }),
        );
    }
    entries
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(qualified) {
        return Ok(vec![]);
    }
    let entries: Vec<_> = entries(false).into_iter().chain(entries(true)).collect();
    Ok(vec![(
        FAMILY,
        prefill::cubin_spec(CUBIN.into(), &entries).map_err(|e| e.to_string())?,
    )])
}

pub fn register(backend: Sm120, g: &Geometry) -> Sm120 {
    if !qualified(g) {
        return backend;
    }
    // Two devices at 250 W: select only measured buckets with a clear gain.
    // Each row retains the original butterfly, norm tree and Sinkhorn order.
    backend
        .with_sinkhorn_lagged_prefill(
            prefill::pipeline(profile(), mixer(), 8, 8, false),
            vec![64, 192],
        )
        .with_sinkhorn_lagged_prefill(
            prefill::pipeline(profile(), mixer(), 8, 4, false),
            vec![2240, 2496, 2624, 8192],
        )
}
