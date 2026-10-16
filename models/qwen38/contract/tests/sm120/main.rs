//! The SM120 backend serves Qwen3.8's decode routes from its real route
//! cubins, registered one cubin at a time as a model registers them: TP4's
//! cubin holds the router, TP2's none, and every decode route registers
//! with the router kind it computes.

use kernel_core::backend::{Backend, Need};
use kernel_core::contract::{MoeQuery, RouterKind, RowQuery};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::backend::Sm120;
use kernels_sm120::moe::expert_parallel::{ROUTE_BLOCK, RouteContract, RouteEntry};
use model_qwen38_contract::sm120::moe::{DECODE_BUCKETS, route_catalog};

/// `entries` registered as a model registers them: decode routes with the
/// router kind they compute.
fn register(backend: Sm120, entries: &[RouteEntry], computes: RouterKind) -> Sm120 {
    entries.iter().cloned().fold(backend, |backend, entry| {
        if entry.symbol.contains("_decode_route_") {
            backend.with_decode_route(entry, computes)
        } else {
            backend.with_route_entry(entry)
        }
    })
}

#[test]
fn every_profiles_decode_routes_serve_from_their_own_cubin() {
    let catalog = route_catalog();
    let router = catalog
        .iter()
        .find_map(|entry| match entry.contract {
            RouteContract::Router(router) => Some(router),
            _ => None,
        })
        .unwrap();
    let RowQuery::Router { kind, .. } = router else {
        unreachable!()
    };
    let mut profiles = catalog
        .iter()
        .map(|entry| entry.profile)
        .collect::<Vec<_>>();
    profiles.dedup();
    assert!(profiles.len() >= 2, "TP4's and TP2's cubins");
    // One registration per physical cubin; only TP4's holds the router.
    let cubins = profiles
        .iter()
        .map(|&profile| {
            catalog
                .iter()
                .filter(|entry| entry.profile == profile)
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(cubins.iter().any(|cubin| {
        cubin
            .iter()
            .all(|entry| !matches!(entry.contract, RouteContract::Router(_)))
    }));
    let physical = cubins.iter().fold(Sm120::new(), |backend, cubin| {
        register(backend, cubin, kind)
    });
    let pooled = register(Sm120::new(), &catalog, kind);
    let mut served = 0;
    for cubin in &cubins {
        for &tokens in &DECODE_BUCKETS {
            let Some(entry) = cubin.iter().find(|entry| {
                entry
                    .symbol
                    .ends_with(&format!("_decode_route_t{tokens}_v1"))
            }) else {
                continue;
            };
            let routes = MoeQuery::LocalRoutes {
                experts: entry.experts,
                local_experts: entry.local_experts,
                top_k: entry.top_k,
                capacity: 8192,
            };
            let usage = OperationUse {
                phase: ExecutionPhase::Decode,
                capacity: tokens,
                graph: true,
            };
            // A decode route writes route metadata, so its declaration is
            // claimed for the routed operation.
            for need in [
                Need::rows(tokens),
                Need::rows(tokens).with_exact(entry.invariance.of_routed_operation()),
            ] {
                for backend in [&physical, &pooled] {
                    let op = backend
                        .moe_route(&router, &routes, 0, ROUTE_BLOCK, usage, need)
                        .unwrap_or_else(|refused| panic!("{} {need:?}: {refused}", entry.symbol));
                    assert_eq!(op.info().symbols, [(entry.symbol.clone(), 0)]);
                }
            }
            served += 1;
        }
    }
    assert_eq!(served, profiles.len() * DECODE_BUCKETS.len());
}
