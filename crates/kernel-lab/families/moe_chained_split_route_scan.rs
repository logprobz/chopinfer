//! Lab instances of the split's decode route (`chained::split::routes`, all
//! 288 experts the rank's) for the warp-parallel metadata of
//! `route_local_scan.cuh`: the one-CTA route
//! (`chop::moe::local::route_decode`, the reference whose outputs the scan
//! must equal bit for bit), then the scan unmodified and with every
//! deliberate defect of `chop::moe::local::ScanMutation` (routes of a group
//! ranked from the last lane, group starts summed from the last group, no
//! padding, active experts one slot late, 32-expert chunks without their
//! carry), and the scan's missing-wait control. Shared with
//! `tests/moe_chained.rs`.

use kernel_build::CubinSpec;
use kernels_sm120::moe::chained::{self, LocalRoute, RouteKernel, split};
use std::path::PathBuf;

pub const FAMILY: &str = "moe_chained_split_route_scan";

/// The lab cubin.
pub const CUBIN: &str = "chop_moe_chain_lab_split_route_scan";
/// The one-CTA reference route's symbol.
pub const REFERENCE: &str = "chop_lab_split_route_one_cta_decode_v1";
/// The missing-wait control's cubin (`cuda/probes/route_scan_no_wait.cuh`):
/// the served scan with its `griddepcontrol.wait` made a no-op, alone in
/// its cubin.
pub const NO_WAIT_CUBIN: &str = "chop_moe_chain_lab_split_route_scan_no_wait";
/// The missing-wait control's symbol.
pub const NO_WAIT: &str = "chop_lab_split_route_scan_no_wait_decode_v1";

/// `chop::moe::local::ScanMutation` values; the first is the unmodified
/// kernel.
pub const MUTATIONS: [&str; 6] = [
    "none",
    "descending_lanes",
    "reversed_groups",
    "no_padding",
    "active_shift",
    "stale_carry",
];

/// The split's served decode route.
pub fn route() -> LocalRoute {
    split::routes()[0]
}

/// The symbol of the route's instance with `mutation`.
pub fn symbol(mutation: &str) -> String {
    format!("chop_lab_split_route_scan_{mutation}_decode_v1")
}

/// The reference and the scan instances, each the served decode entry with
/// its symbol and body replaced: the reference calls `route_decode`, a scan
/// instance `route_decode_scan` with its mutation as the last template
/// argument.
pub fn cubin() -> CubinSpec {
    let route = route();
    assert_eq!(route.kernels(), [RouteKernel::Decode]);
    let mut spec = chained::route_cubin_spec(split::PREFIX, &[route], &[])
        .expect("the split decode route's cubin");
    spec.name = CUBIN.into();
    let served = std::mem::take(&mut spec.entries);
    assert_eq!(served.len(), 1);
    // The one-CTA body, whichever of the two the served entry calls.
    let one_cta = served[0]
        .body
        .replace("::route_decode_scan<", "::route_decode<");
    assert_eq!(one_cta.matches("::route_decode<").count(), 1, "{one_cta}");
    assert_eq!(one_cta.matches(">(logits,").count(), 1, "{one_cta}");
    let mut reference = served[0].clone();
    reference.symbol = REFERENCE.into();
    reference.body = one_cta.clone();
    spec.entries.push(reference);
    for mutation in MUTATIONS {
        let mut instance = served[0].clone();
        instance.symbol = symbol(mutation);
        instance.body = one_cta
            .replace("::route_decode<", "::route_decode_scan<")
            .replace(
                ">(logits,",
                &format!(
                    ",chop::moe::SigmoidBias,chop::moe::local::ScanMutation::{mutation}>(logits,"
                ),
            );
        spec.entries.push(instance);
    }
    if !spec
        .headers
        .iter()
        .any(|h| h.contains("route_local_scan.cuh"))
    {
        spec.headers
            .push("#include \"moe/route_local_scan.cuh\"".into());
    }
    spec
}

/// The missing-wait control: the unmodified scan instance's entry compiled
/// against `cuda/probes/route_scan_no_wait.cuh`, which includes the served
/// route headers with every dependency wait a no-op.
pub fn no_wait_cubin() -> CubinSpec {
    let scan = cubin()
        .entries
        .into_iter()
        .find(|e| e.symbol == symbol(MUTATIONS[0]))
        .expect("the unmodified scan instance");
    let mut control = scan.clone();
    control.symbol = NO_WAIT.into();
    CubinSpec {
        name: NO_WAIT_CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: Vec::new(),
        headers: vec!["#include \"kernel-lab/cuda/probes/route_scan_no_wait.cuh\"".into()],
        defines: Vec::new(),
        entries: vec![control],
    }
}

pub fn cubins() -> Vec<CubinSpec> {
    crate::declared(&[CUBIN, NO_WAIT_CUBIN], vec![cubin(), no_wait_cubin()])
}
