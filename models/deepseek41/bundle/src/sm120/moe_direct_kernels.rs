//! Locked exact MoE metadata, direct-load chains and fused epilogues.
use kernel_build::CubinSpec;
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::chained::{absorb, bandwidth, floor, row_bias, strip2};
use model_deepseek41_contract::{geometry::Geometry, moe};

/// The served lower-latency router finish variant (`router_finish_r2.cuh`;
/// all chain loads first, counts scanned per expert warp, experts scored in
/// parallel), bitwise equal to the served finish.
pub const R2_VARIANT: u32 = 7;
/// The served r2 module compiles the qualified lab module's variants
/// unchanged (`routed_moe_router_finish_r2_lab`): the translation unit
/// decides register allocation and scheduling, and only these bytes were
/// qualified. Only `R2_VARIANT` unstamped is launched.
pub const R2_MODULE_VARIANTS: [u32; 3] = [1, 3, R2_VARIANT];

pub fn entries(g: &Geometry) -> Result<Vec<kernels_sm120::moe::chained::floor::Entry>, String> {
    let projections = super::moe::streamed_projections(g)?;
    moe::expert_sets(g)
        .into_iter()
        .map(|set| {
            let find = |kind| {
                projections
                    .iter()
                    .find(|p| {
                        p.tokens_capacity == 64
                            && p.query.projection.gemm.grouped.unwrap().experts == set.count
                            && p.query.projection.projection == kind
                    })
                    .unwrap()
                    .clone()
            };
            kernels_sm120::moe::chained::floor::Entry::new(
                find(GatedMlpProjection::GateUp),
                find(GatedMlpProjection::Down),
                1e-4_f32.to_bits(),
                format!(
                    "chop_{}_moe_floor_e{}_k{}",
                    g.kernel_tag, set.count, set.top_k
                ),
            )
            .map_err(|e| e.to_string())
        })
        .collect()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries
        .iter()
        .filter(|g| super::moe::streamed_qualified(g, 4))
    {
        let entries = entries(g)?;
        specs.push((
            "routed_moe_resident",
            kernels_sm120::moe::chained::streamed_resident::cubin_spec(
                &format!("bandwidth_moe_{}_resident_sm120", g.kernel_tag),
                &entries.iter().map(|e| e.gate.clone()).collect::<Vec<_>>(),
            ),
        ));
        specs.push((
            "routed_moe_router_finish_r2",
            kernels_sm120::moe::chained::router_finish::r2_cubin_spec(
                &format!("router_finish_moe_{}_r2_sm120", g.kernel_tag),
                &entries,
                &R2_MODULE_VARIANTS,
            ),
        ));
        specs.push((
            "routed_moe_gate_finish_loads",
            bandwidth::finish_loads_cubin_spec(
                &format!("bandwidth_moe_{}_finish_loads_sm120", g.kernel_tag),
                &entries,
            ),
        ));
        specs.push((
            "routed_moe_absorb",
            absorb::cubin_spec(
                &format!("bandwidth_moe_{}_absorb_sm120", g.kernel_tag),
                &entries,
            ),
        ));
        specs.push((
            "routed_moe_strip2",
            strip2::cubin_spec(
                &format!("bandwidth_moe_{}_strip2_sm120", g.kernel_tag),
                &entries,
            ),
        ));
        specs.push((
            "routed_moe_router_finish",
            kernels_sm120::moe::chained::router_finish::cubin_spec(
                &format!("router_finish_moe_{}_sm120", g.kernel_tag),
                &entries,
            ),
        ));
        specs.extend(
            floor::cubins(&format!("floor_moe_{}", g.kernel_tag), &entries)
                .into_iter()
                .map(|s| ("routed_moe_floor", s)),
        );
        specs.push((
            "routed_moe_bandwidth",
            bandwidth::cubin_spec(
                &format!("bandwidth_moe_{}_compute_sm120", g.kernel_tag),
                &entries,
            ),
        ));
        specs.push((
            "routed_moe_bandwidth",
            row_bias::cubin_spec(
                &format!("bandwidth_moe_{}_route_sm120", g.kernel_tag),
                &entries.iter().map(bandwidth::router).collect::<Vec<_>>(),
                &[],
            )
            .map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;
    #[test]
    fn direct_decode_one_keeps_the_full_compiled_router_grid() {
        for entry in entries(&FLASH).unwrap() {
            let router = bandwidth::router(&entry);
            assert_eq!(router.capacity, 64);
            assert_eq!(router.plan(1).unwrap().grid, [8, 1, 1]);
            assert!(router.plan(65).is_err());
            for rows in [1, 8, 64] {
                for stage in [
                    floor::Stage::GateSmall,
                    floor::Stage::DownSmall,
                    floor::Stage::GateFinish,
                ] {
                    let plan = bandwidth::plan(&entry, stage, rows).unwrap();
                    assert!(plan.workspace_bytes <= entry.workspace_bytes(rows).unwrap());
                    assert_eq!(plan.block, [128, 1, 1]);
                }
            }
        }
    }

    /// The tiled absorbing gate queue is the row-major one with its weights
    /// read at the row-tile addresses: same ABI, bounds and launch, a new
    /// symbol, and the module that adds it keeps every row-major entry.
    #[test]
    fn the_tiled_absorbing_gate_differs_only_in_its_weight_addresses() {
        let entries = entries(&FLASH).unwrap();
        for entry in &entries {
            let rows = absorb::entry(entry, floor::Stage::GateSmall);
            let tiled = absorb::tiled_entry(entry).unwrap();
            assert_eq!(
                tiled.symbol,
                rows.symbol.replace("_moe_absorb_", "_moe_absorb_tiles_")
            );
            assert_eq!(
                tiled.body.replace(&tiled.symbol, &rows.symbol),
                rows.body.replace(">(input,", ",true>(input,")
            );
            assert_eq!(
                (&tiled.params, tiled.max_threads, tiled.min_blocks),
                (&rows.params, rows.max_threads, rows.min_blocks)
            );
        }
        let served = absorb::cubin_spec("absorb", &entries);
        let with = absorb::tiled_cubin_spec("absorb", &entries, &entries[..1]).unwrap();
        let ids = |spec: &kernel_build::CubinSpec| {
            spec.entries
                .iter()
                .map(|e| (e.symbol.clone(), e.body.clone()))
                .collect::<Vec<_>>()
        };
        let (with, served) = (ids(&with), ids(&served));
        assert_eq!(with[..served.len()], served[..]);
        assert_eq!(with.len(), served.len() + 1);
    }

    /// The tiled direct gate queues (small and wide), the tiled resident
    /// gate and the streamed module's tiled gate queues are the row-major
    /// ones with their weights read at the row-tile addresses: same ABI,
    /// bounds and launch, new symbols, and the modules that add them keep
    /// every row-major entry.
    #[test]
    fn the_tiled_direct_resident_and_streamed_gates_differ_only_in_their_weight_addresses() {
        use kernels_sm120::moe::chained::{streamed, streamed_resident};
        let entries = entries(&FLASH).unwrap();
        let projections = super::super::moe::streamed_projections(&FLASH).unwrap();
        let backbone = projections
            .iter()
            .filter(|p| p.gate_up() && p.query.projection.gemm.grouped.unwrap().experts == 384)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(backbone.len(), 2);
        let mut pairs = Vec::new();
        for p in &projections {
            for stage in streamed::tiled_stages(p) {
                let rows = streamed::entries(p)
                    .into_iter()
                    .find(|e| e.symbol == streamed::symbol(p, stage))
                    .unwrap();
                let tiled = streamed::tiled_entry(p, stage);
                if !p.gate_up() {
                    assert!(tiled.is_err());
                    continue;
                }
                let n = if stage == streamed::Stage::Small {
                    8
                } else {
                    64
                };
                pairs.push((rows, tiled.unwrap(), n));
            }
        }
        assert_eq!(pairs.len(), 6);
        for entry in &entries {
            for (stage, rows) in [(floor::Stage::GateSmall, 8), (floor::Stage::GateWide, 64)] {
                pairs.push((
                    bandwidth::entry(entry, stage, bandwidth::Loads::Direct),
                    bandwidth::tiled_entry(entry, stage).unwrap(),
                    rows,
                ));
            }
            pairs.push((
                streamed_resident::entry(&entry.gate),
                streamed_resident::tiled_entry(&entry.gate).unwrap(),
                8,
            ));
            for stage in [floor::Stage::DownSmall, floor::Stage::DownWide] {
                assert!(bandwidth::tiled_entry(entry, stage).is_err());
            }
        }
        let mut symbols = std::collections::BTreeSet::new();
        for (rows, tiled, n) in &pairs {
            assert!(tiled.symbol.contains("_tiles_") && symbols.insert(tiled.symbol.clone()));
            assert_eq!(
                tiled.body.replace(&tiled.symbol, &rows.symbol),
                rows.body
                    .replace(&format!(",{n}>(input,"), &format!(",{n},true>(input,"))
            );
            assert_eq!(
                (&tiled.params, tiled.max_threads, tiled.min_blocks),
                (&rows.params, rows.max_threads, rows.min_blocks)
            );
        }
        let gates = entries.iter().map(|e| e.gate.clone()).collect::<Vec<_>>();
        for (served, with, added) in [
            (
                bandwidth::cubin_spec("bandwidth", &entries),
                bandwidth::tiled_cubin_spec("bandwidth", &entries, &entries[..1]).unwrap(),
                2,
            ),
            (
                streamed_resident::cubin_spec("resident", &gates),
                streamed_resident::tiled_cubin_spec("resident", &gates, &gates[..1]).unwrap(),
                1,
            ),
            (
                streamed::cubin_spec("streamed", &projections),
                streamed::tiled_cubin_spec("streamed", &projections, &backbone).unwrap(),
                3,
            ),
        ] {
            let ids = |spec: &kernel_build::CubinSpec| {
                spec.entries
                    .iter()
                    .map(|e| (e.symbol.clone(), e.body.clone()))
                    .collect::<Vec<_>>()
            };
            let (with, served) = (ids(&with), ids(&served));
            assert_eq!(with[..served.len()], served[..]);
            assert_eq!(with.len(), served.len() + added);
        }
    }
}
