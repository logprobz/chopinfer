//! Prefill-only candidates; none is registered for serving before measurement.
use kernel_build::CubinSpec;
use kernel_core::contract::LaunchPlan;
use kernels_sm120::sinkhorn_hc::{Geometry, lagged, lagged_tiled, prefill_warp as prefill};
use model_deepseek41_contract::geometry::FLASH;

pub const FAMILY: &str = "hc_prefill_warp";
pub const CANDIDATE: &str = "hc_prefill_warp_deepseek41_sm120";
pub const CONTROL: &str = "hc_prefill_warp_control_sm120";
pub const PROBES: &str = "hc_prefill_warp_reference_probes_sm120";
pub const MIX_VARIANTS: [(u32, bool); 4] = [(4, true), (8, true), (16, true), (8, false)];
pub const EXPAND_VARIANTS: [(u32, bool, bool); 5] = [
    (128, false, false),
    (256, false, false),
    (128, false, true),
    (256, false, true),
    (256, true, true),
];

pub fn profile() -> kernels_sm120::Profile {
    kernels_sm120::Profile::synthetic(kernels_sm120::ProfileNames {
        name: "deepseek41",
        tag: FLASH.kernel_tag,
    })
}
pub fn geometry() -> Geometry {
    Geometry {
        mix: model_deepseek41_contract::hc::mix(&FLASH),
        capacity: model_deepseek41_contract::serving::PREFILL_TOKENS,
    }
}
pub fn mix(rows: u32, staged: bool, probe: bool) -> prefill::Entry {
    prefill::mix(profile(), geometry(), rows, staged, probe)
}
pub fn expand((threads, row_warps, broadcast): (u32, bool, bool)) -> prefill::Entry {
    prefill::expand(profile(), geometry(), threads, row_warps, broadcast)
}
pub fn mix_control() -> prefill::Entry {
    let mut e = mix(8, true, true);
    e.reference.symbol = e.reference.symbol.replace("_v1", "_reverse_v1");
    e
}
pub fn expand_control() -> prefill::Entry {
    let mut e = expand((128, false, true));
    e.reference.symbol = e.reference.symbol.replace("_v1", "_reverse_v1");
    e
}
pub fn old_mix(rows: u32, probe: bool) -> (lagged::Entry, LaunchPlan) {
    if rows >= 256 {
        let e = lagged_tiled::entry(profile(), geometry(), 8, 256, probe);
        let p = e.plan(rows, 0).unwrap();
        (e.reference, p)
    } else {
        let e = lagged::entries(profile(), geometry(), probe)
            .into_iter()
            .find(|e| e.op == lagged::Op::MixCollapse { probe })
            .unwrap();
        let p = e.plan(rows, 0).unwrap();
        (e, p)
    }
}
pub fn old_expand(rows: u32) -> (lagged::Entry, LaunchPlan) {
    let e = lagged::entries(profile(), geometry(), false)
        .into_iter()
        .find(|e| e.op == lagged::Op::Expand)
        .unwrap();
    let p = e.plan(rows, 0).unwrap();
    (e, p)
}
pub fn candidates(probe: bool) -> Vec<prefill::Entry> {
    MIX_VARIANTS
        .into_iter()
        .map(|(r, c)| mix(r, c, probe))
        .chain(
            [4, 8]
                .into_iter()
                .map(|chains| prefill::pipeline(profile(), geometry(), 8, chains, probe)),
        )
        .chain(if probe {
            vec![]
        } else {
            EXPAND_VARIANTS.into_iter().map(expand).collect()
        })
        .collect()
}
pub fn specs() -> Vec<CubinSpec> {
    let all: Vec<_> = candidates(false)
        .into_iter()
        .chain(candidates(true))
        .collect();
    let candidate = prefill::cubin_spec(CANDIDATE.into(), &all).unwrap();
    let mut control =
        prefill::cubin_spec(CONTROL.into(), &[mix_control(), expand_control()]).unwrap();
    control.defines.extend([
        ("CHOP_HC_WARP_REVERSE_TREE".into(), "1".into()),
        ("CHOP_HC_WARP_REVERSE_STREAMS".into(), "1".into()),
    ]);
    let mut probes = lagged::cubin_spec(PROBES.into(), &[old_mix(1, true).0]);
    let e = lagged_tiled::entry(profile(), geometry(), 8, 256, true);
    let tiled = lagged_tiled::cubin_spec("reference".into(), &[e]).unwrap();
    probes.headers.extend(tiled.headers);
    probes.entries.extend(tiled.entries);
    vec![candidate, control, probes]
}
/// Upper-layer prefill skipping retains 64 + 128*l rows, plus the next HC halo.
pub fn profiled_rows() -> Vec<u32> {
    (0..=20)
        .map(|layer| 64 + 128 * layer)
        .chain([8192])
        .collect()
}
pub fn rows() -> Vec<u32> {
    let mut rows: Vec<_> = profiled_rows()
        .into_iter()
        .chain([
            1, 7, 17, 33, 63, 65, 127, 129, 255, 256, 257, 1024, 2048, 4095, 4096, 8191,
        ])
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}
