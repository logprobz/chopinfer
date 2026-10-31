//! Served query domains and the keys that retain their baseline schedule.
use kernel_build::tuning::Key;
use model_deepseek41_contract::{
    geometry::FLASH,
    tuning::{self, Family},
};

fn add(
    out: &mut std::collections::BTreeSet<Key>,
    op: &str,
    weights: &str,
    shape: &[(&str, u32)],
    bucket: u32,
) {
    out.insert(Key {
        op: op.into(),
        weights: weights.into(),
        shape: shape.iter().map(|&(k, v)| (k.into(), v)).collect(),
        bucket,
        rows: None,
    });
}
fn domains() -> std::collections::BTreeSet<Key> {
    use model_deepseek41_contract::{
        admission::SUPPORTED_RANKS,
        moe,
        serving::{DECODE_BATCH, PREFILL_TOKENS},
        verify_bounds,
    };
    let mut out = std::collections::BTreeSet::new();
    let prefill_capacities = SUPPORTED_RANKS
        .iter()
        .flat_map(|&ranks| {
            let shape = crate::shape::Shape::new(
                &FLASH,
                bundle_runtime::geometry::Ranks::new(ranks).unwrap(),
            )
            .unwrap();
            crate::Kernels::moe_buckets(&shape)
                .iter()
                .filter_map(|&(phase, capacity)| {
                    (phase == kernel_core::precision::ExecutionPhase::Prefill).then_some(capacity)
                })
                .collect::<Vec<_>>()
        })
        .collect::<std::collections::BTreeSet<_>>();
    let route_capacities = crate::sm120::moe::streamed_projections(&FLASH)
        .unwrap()
        .into_iter()
        .map(|entry| entry.tokens_capacity)
        .collect::<std::collections::BTreeSet<_>>();
    for &ranks in SUPPORTED_RANKS {
        for phase in ["decode", "verify", "prefill"] {
            add(
                &mut out,
                &format!("moe.streamed.{phase}"),
                "mxfp4",
                &[("ranks", ranks)],
                verify_bounds::MOE_STREAMED,
            );
        }
        for set in moe::expert_sets(&FLASH) {
            let shape = [
                ("experts", set.count),
                ("ranks", ranks),
                ("top_k", set.top_k),
            ];
            for kind in ["gate_up", "down"] {
                add(
                    &mut out,
                    &format!("moe.prefill_{kind}.prefill"),
                    "mxfp4",
                    &[("experts", set.count), ("ranks", ranks)],
                    PREFILL_TOKENS,
                );
                for phase in ["decode", "verify", "prefill"] {
                    for site in ["compact", "active"] {
                        add(
                            &mut out,
                            &format!("moe.{site}_{kind}.{phase}"),
                            "mxfp4",
                            &shape,
                            DECODE_BATCH,
                        );
                    }
                    // The prefill preparation domain remains in sm120::moe_buckets.
                    for &capacity in &prefill_capacities {
                        for site in ["pipeline", "grouped"] {
                            add(
                                &mut out,
                                &format!("moe.{site}_{kind}.{phase}"),
                                "mxfp4",
                                &shape,
                                capacity,
                            );
                        }
                    }
                }
            }
        }
    }
    for set in moe::expert_sets(&FLASH) {
        let shape = [("experts", set.count), ("top_k", set.top_k)];
        for &capacity in &route_capacities {
            add(&mut out, "moe.route.decode", "mxfp4", &shape, capacity);
        }
        for phase in ["decode", "verify", "prefill"] {
            for site in [
                "direct_admit",
                "router_finish",
                "router_gap",
                "router_wide",
                "router_r2",
            ] {
                add(
                    &mut out,
                    &format!("moe.{site}.{phase}"),
                    "mxfp4",
                    &shape,
                    verify_bounds::MOE_DIRECT,
                );
            }
            for site in [
                "absorb",
                "strip2",
                "wide",
                "small",
                "gate_grid",
                "gate_resident",
                "gate_finish",
            ] {
                add(
                    &mut out,
                    &format!("moe.{site}.{phase}"),
                    "mxfp4",
                    &[("experts", set.count)],
                    verify_bounds::MOE_DIRECT,
                );
            }
        }
    }
    out
}

const UNTUNED: &[&str] = &[
    "moe.absorb.decode mxfp4 experts=128 t64: 1..=64",
    "moe.absorb.decode mxfp4 experts=384 t64: 1..=31,33..=64",
    "moe.absorb.prefill mxfp4 experts=128 t64: 1..=64",
    "moe.absorb.prefill mxfp4 experts=384 t64: 1..=64",
    "moe.absorb.verify mxfp4 experts=128 t64: 1..=64",
    "moe.absorb.verify mxfp4 experts=384 t64: 1..=31,33..=64",
    "moe.active_down.decode mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.active_down.decode mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.active_down.decode mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.active_down.decode mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.active_down.decode mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.active_down.decode mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.active_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.active_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.active_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.active_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.active_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.active_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.active_down.verify mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.active_down.verify mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.active_down.verify mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.active_down.verify mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.active_down.verify mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.active_down.verify mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.active_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.active_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.active_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.active_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.active_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.active_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t32: 2..=32",
    "moe.active_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.active_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.active_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.active_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.active_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.active_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t32: 2..=32",
    "moe.active_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.active_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.active_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.active_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.active_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.active_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t32: 2..=32",
    "moe.compact_down.decode mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.compact_down.decode mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.compact_down.decode mxfp4 experts=128,ranks=4,top_k=3 t32: 2..=32",
    "moe.compact_down.decode mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.compact_down.decode mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.compact_down.decode mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.compact_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.compact_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.compact_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.compact_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.compact_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.compact_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.compact_down.verify mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.compact_down.verify mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.compact_down.verify mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.compact_down.verify mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.compact_down.verify mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.compact_down.verify mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t32: 2..=32",
    "moe.compact_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t32: 1..=32",
    "moe.compact_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t32: 1..=32",
    "moe.compact_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t32: 1..=32",
    "moe.direct_admit.decode mxfp4 experts=128,top_k=3 t64: 3,5..=7,9..=15,17..=31,33..=64",
    "moe.direct_admit.decode mxfp4 experts=384,top_k=6 t64: 3,5..=7,9..=15,17..=31,33..=64",
    "moe.direct_admit.prefill mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.direct_admit.prefill mxfp4 experts=384,top_k=6 t64: 1..=64",
    "moe.direct_admit.verify mxfp4 experts=128,top_k=3 t64: 3,7,9..=15,17..=31,33..=47,49..=63",
    "moe.gate_finish.decode mxfp4 experts=128 t64: 1..=64",
    "moe.gate_finish.decode mxfp4 experts=384 t64: 2..=7,9..=15,17..=64",
    "moe.gate_finish.prefill mxfp4 experts=128 t64: 1..=64",
    "moe.gate_finish.prefill mxfp4 experts=384 t64: 1..=64",
    "moe.gate_finish.verify mxfp4 experts=128 t64: 1..=64",
    "moe.gate_finish.verify mxfp4 experts=384 t64: 1..=3,5,7..=15,17..=64",
    "moe.gate_grid.decode mxfp4 experts=128 t64: 1..=64",
    "moe.gate_grid.decode mxfp4 experts=384 t64: 1..=7,9..=31,33..=64",
    "moe.gate_grid.prefill mxfp4 experts=128 t64: 1..=64",
    "moe.gate_grid.prefill mxfp4 experts=384 t64: 1..=64",
    "moe.gate_grid.verify mxfp4 experts=128 t64: 1..=64",
    "moe.gate_grid.verify mxfp4 experts=384 t64: 1..=31,33..=64",
    "moe.gate_resident.decode mxfp4 experts=128 t64: 1..=64",
    "moe.gate_resident.decode mxfp4 experts=384 t64: 1..=15,17..=31,33..=64",
    "moe.gate_resident.prefill mxfp4 experts=128 t64: 1..=64",
    "moe.gate_resident.prefill mxfp4 experts=384 t64: 1..=15,17..=31,33..=64",
    "moe.gate_resident.verify mxfp4 experts=128 t64: 1..=64",
    "moe.gate_resident.verify mxfp4 experts=384 t64: 1..=3,5,7..=15,17..=31,33..=64",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.decode mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.decode mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.grouped_down.verify mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.grouped_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.grouped_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.decode mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.decode mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.prefill mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.prefill mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.pipeline_down.verify mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.pipeline_down.verify mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.decode mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.decode mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.prefill mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.prefill mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=1,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=2,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t512: 1..=512",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t2048: 1..=2048",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=128,ranks=4,top_k=3 t8192: 1..=8192",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=1,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=2,top_k=6 t8192: 1..=8192",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t512: 1..=512",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t2048: 1..=2048",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t4096: 1..=4096",
    "moe.pipeline_gate_up.verify mxfp4 experts=384,ranks=4,top_k=6 t8192: 1..=8192",
    "moe.prefill_down.prefill mxfp4 experts=128,ranks=1 t8192: 1..=8192",
    "moe.prefill_down.prefill mxfp4 experts=128,ranks=2 t8192: 1..=8192",
    "moe.prefill_down.prefill mxfp4 experts=128,ranks=4 t8192: 1..=511,513..=4095,4097..=8192",
    "moe.prefill_down.prefill mxfp4 experts=384,ranks=1 t8192: 1..=8192",
    "moe.prefill_down.prefill mxfp4 experts=384,ranks=2 t8192: 1..=8192",
    "moe.prefill_down.prefill mxfp4 experts=384,ranks=4 t8192: 1..=511,513..=2047,2049..=4095,4097..=8192",
    "moe.prefill_gate_up.prefill mxfp4 experts=128,ranks=1 t8192: 1..=8192",
    "moe.prefill_gate_up.prefill mxfp4 experts=128,ranks=2 t8192: 1..=8192",
    "moe.prefill_gate_up.prefill mxfp4 experts=128,ranks=4 t8192: 1..=511,513..=2047,2049..=4095,4097..=8192",
    "moe.prefill_gate_up.prefill mxfp4 experts=384,ranks=1 t8192: 1..=8192",
    "moe.prefill_gate_up.prefill mxfp4 experts=384,ranks=2 t8192: 1..=8192",
    "moe.prefill_gate_up.prefill mxfp4 experts=384,ranks=4 t8192: 1..=511,513..=2047,2049..=4095,4097..=8192",
    "moe.router_finish.decode mxfp4 experts=128,top_k=3 t64: 2..=64",
    "moe.router_finish.decode mxfp4 experts=384,top_k=6 t64: 2..=7,9..=15,17..=31,33..=64",
    "moe.router_finish.prefill mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_finish.prefill mxfp4 experts=384,top_k=6 t64: 1..=64",
    "moe.router_finish.verify mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_finish.verify mxfp4 experts=384,top_k=6 t64: 1..=3,5,7,9..=15,17..=31,33..=64",
    "moe.router_gap.decode mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_gap.decode mxfp4 experts=384,top_k=6 t64: 1,3,5..=64",
    "moe.router_gap.prefill mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_gap.prefill mxfp4 experts=384,top_k=6 t64: 1..=64",
    "moe.router_gap.verify mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_gap.verify mxfp4 experts=384,top_k=6 t64: 1,4,6,8,16,32..=64",
    "moe.router_r2.decode mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_r2.decode mxfp4 experts=384,top_k=6 t64: 2..=7,9..=15,17..=31,33..=64",
    "moe.router_r2.prefill mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_r2.prefill mxfp4 experts=384,top_k=6 t64: 1..=64",
    "moe.router_r2.verify mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_r2.verify mxfp4 experts=384,top_k=6 t64: 1..=3,5,7..=15,17..=31,33..=64",
    "moe.router_wide.decode mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_wide.decode mxfp4 experts=384,top_k=6 t64: 1..=64",
    "moe.router_wide.prefill mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_wide.prefill mxfp4 experts=384,top_k=6 t64: 1..=64",
    "moe.router_wide.verify mxfp4 experts=128,top_k=3 t64: 1..=64",
    "moe.router_wide.verify mxfp4 experts=384,top_k=6 t64: 1..=32",
    "moe.small.decode mxfp4 experts=128 t64: 9..=64",
    "moe.small.decode mxfp4 experts=384 t64: 9..=64",
    "moe.small.prefill mxfp4 experts=128 t64: 9..=64",
    "moe.small.prefill mxfp4 experts=384 t64: 9..=64",
    "moe.small.verify mxfp4 experts=128 t64: 9..=64",
    "moe.small.verify mxfp4 experts=384 t64: 9..=64",
    "moe.streamed.decode mxfp4 ranks=1 t64: 1..=64",
    "moe.streamed.decode mxfp4 ranks=2 t64: 1..=64",
    "moe.streamed.prefill mxfp4 ranks=1 t64: 1..=64",
    "moe.streamed.prefill mxfp4 ranks=2 t64: 1..=64",
    "moe.streamed.prefill mxfp4 ranks=4 t64: 1..=64",
    "moe.streamed.verify mxfp4 ranks=1 t64: 1..=64",
    "moe.streamed.verify mxfp4 ranks=2 t64: 1..=64",
    "moe.strip2.decode mxfp4 experts=128 t64: 1..=64",
    "moe.strip2.decode mxfp4 experts=384 t64: 1..=15,17..=31,33..=64",
    "moe.strip2.prefill mxfp4 experts=128 t64: 1..=64",
    "moe.strip2.prefill mxfp4 experts=384 t64: 1..=15,17..=31,33..=64",
    "moe.strip2.verify mxfp4 experts=128 t64: 1..=64",
    "moe.strip2.verify mxfp4 experts=384 t64: 1..=15,17..=31,33..=64",
    "moe.wide.decode mxfp4 experts=128 t64: 1..=64",
    "moe.wide.decode mxfp4 experts=384 t64: 1..=64",
    "moe.wide.prefill mxfp4 experts=128 t64: 1..=64",
    "moe.wide.prefill mxfp4 experts=384 t64: 1..=64",
    "moe.wide.verify mxfp4 experts=128 t64: 1..=64",
    "moe.wide.verify mxfp4 experts=384 t64: 1..=32",
];

#[test]
fn pin_untuned_served_keys() {
    let family = Family::Moe;
    let table = tuning::table(&FLASH, family).unwrap();
    assert!(!table.rows().is_empty());
    let domains = domains();
    let mut missing = Vec::new();
    let mut cases = 0;
    for key in &domains {
        let mut ranges: Vec<(u32, u32)> = Vec::new();
        for rows in 1..=key.bucket {
            if table
                .select(&key.op, &key.weights, &key.shape, key.bucket, rows)
                .is_none()
            {
                if let Some((_, hi)) = ranges.last_mut()
                    && *hi + 1 == rows
                {
                    *hi = rows;
                } else {
                    ranges.push((rows, rows));
                }
            }
            cases += 1;
        }
        assert!(
            table
                .select(&key.op, &key.weights, &key.shape, key.bucket, 0)
                .is_none()
        );
        assert!(
            table
                .select(
                    &key.op,
                    &key.weights,
                    &key.shape,
                    key.bucket,
                    key.bucket + 1
                )
                .is_none()
        );
        if !ranges.is_empty() {
            let rows = ranges
                .into_iter()
                .map(|(lo, hi)| {
                    if lo == hi {
                        lo.to_string()
                    } else {
                        format!("{lo}..={hi}")
                    }
                })
                .collect::<Vec<_>>()
                .join(",");
            missing.push(format!("{key}: {rows}"));
        }
    }
    assert_eq!(missing, UNTUNED, "untuned served keys changed");
    for row in table.rows() {
        assert!(
            domains.contains(&Key {
                rows: None,
                ..row.key.clone()
            }),
            "table row is outside the pinned served domains: {}",
            row.key
        );
    }
    assert!(cases > 0);
    eprintln!(
        "{}: {cases} served lookup cases, {} untuned keys",
        family.name(),
        missing.len()
    );
}
