use crate::sm120::{
    moe as serving, moe_grouped_prefill, moe_pipeline, moe_router_finish as router,
};
use kernel_build::tuning::{Key, Knob, Knobs, Row, Table, Target};
use kernel_core::{
    gated_mlp::GatedMlpProjection,
    precision::{ExecutionPhase, OperationUse},
};
use model_deepseek41_contract::{
    geometry::FLASH,
    moe,
    tuning::{Family, table},
};

fn phase_name(phase: ExecutionPhase) -> &'static str {
    match phase {
        ExecutionPhase::Decode => "decode",
        ExecutionPhase::Verify => "verify",
        ExecutionPhase::Prefill => "prefill",
    }
}
fn phases() -> [ExecutionPhase; 3] {
    [
        ExecutionPhase::Decode,
        ExecutionPhase::Verify,
        ExecutionPhase::Prefill,
    ]
}
fn kind_name(kind: GatedMlpProjection) -> &'static str {
    match kind {
        GatedMlpProjection::GateUp => "gate_up",
        GatedMlpProjection::Down => "down",
    }
}
fn add(
    t: &mut Table,
    site: &str,
    phase: ExecutionPhase,
    shape: &[(&str, u32)],
    bucket: u32,
    schedule: &str,
    cases: impl IntoIterator<Item = (u32, Knobs)>,
) {
    let mut cases = cases.into_iter().collect::<Vec<_>>();
    cases.sort_by_key(|(r, _)| *r);
    let mut ranges: Vec<(u32, u32, Knobs)> = Vec::new();
    for (r, knobs) in cases {
        if let Some((_, hi, old)) = ranges.last_mut()
            && *hi + 1 == r
            && old == &knobs
        {
            *hi = r;
        } else {
            ranges.push((r, r, knobs));
        }
    }
    for (lo, hi, knobs) in ranges {
        t.upsert(Row {
            key: Key {
                op: format!("moe.{site}.{}", phase_name(phase)),
                weights: "mxfp4".into(),
                shape: shape.iter().map(|&(k, v)| (k.into(), v)).collect(),
                bucket,
                rows: (lo != 1 || hi != bucket).then_some([lo, hi]),
            },
            schedule: schedule.into(),
            knobs,
        })
        .unwrap();
    }
}
fn generated() -> Table {
    let mut t = Table::new(Target {
        compute_capability: 120,
        sm_count: 188,
    });
    for ranks in [1, 2, 4] {
        if serving::streamed_qualified(&FLASH, ranks) {
            for phase in [ExecutionPhase::Decode, ExecutionPhase::Verify] {
                add(
                    &mut t,
                    "streamed",
                    phase,
                    &[("ranks", ranks)],
                    64,
                    "streamed",
                    (1..=64).map(|r| (r, Knobs::new())),
                );
            }
        }
        for set in moe::expert_sets(&FLASH) {
            let shape = [
                ("ranks", ranks),
                ("experts", set.count),
                ("top_k", set.top_k),
            ];
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                add(
                    &mut t,
                    &format!("prefill_{}", kind_name(kind)),
                    ExecutionPhase::Prefill,
                    &[("ranks", ranks), ("experts", set.count)],
                    8192,
                    "prefill",
                    (1..=8192).filter_map(|r| {
                        serving::prefill_selected(&FLASH, ranks, set.count, kind, r)
                            .then(|| (r, Knobs::new()))
                    }),
                );
                for capacity in [512, 2048, 4096, 8192] {
                    for (site, selected) in [
                        ("pipeline", moe_pipeline::selected as fn(_, _, _, _, _) -> _),
                        (
                            "grouped",
                            moe_grouped_prefill::selected as fn(_, _, _, _, _) -> _,
                        ),
                    ] {
                        for phase in phases() {
                            let usage = OperationUse {
                                phase,
                                capacity,
                                graph: false,
                            };
                            if selected(&FLASH, ranks, set, kind, usage) {
                                add(
                                    &mut t,
                                    &format!("{site}_{}", kind_name(kind)),
                                    phase,
                                    &shape,
                                    capacity,
                                    site,
                                    (1..=capacity).map(|r| (r, Knobs::new())),
                                );
                            }
                        }
                    }
                }
            }
        }
        for (site, entries) in [
            (
                "compact",
                serving::serving_compact_projections(&FLASH, ranks).unwrap(),
            ),
            (
                "active",
                serving::serving_active_projections(&FLASH, ranks).unwrap(),
            ),
        ] {
            for entry in entries {
                let grouped = entry.query.projection.gemm.grouped.unwrap();
                let shape = [
                    ("ranks", ranks),
                    ("experts", grouped.experts),
                    ("top_k", grouped.top_k),
                ];
                let kind = kind_name(entry.query.projection.projection);
                let phases = entry
                    .selection_phase
                    .map_or_else(|| phases().to_vec(), |p| vec![p]);
                for phase in phases {
                    add(
                        &mut t,
                        &format!("{site}_{kind}"),
                        phase,
                        &shape,
                        entry.tokens_capacity,
                        site,
                        (entry.min_rows..=entry.max_rows).map(|r| (r, Knobs::new())),
                    );
                }
            }
        }
    }
    for route in serving::streamed_routes(&FLASH) {
        let r = route.local;
        add(
            &mut t,
            "route",
            ExecutionPhase::Decode,
            &[("experts", r.experts), ("top_k", r.top_k)],
            r.capacity,
            "streamed",
            (1..=r.capacity).map(|r| (r, Knobs::new())),
        );
    }
    for set in moe::expert_sets(&FLASH) {
        let shape = [("experts", set.count), ("top_k", set.top_k)];
        for (order, (phase, rows)) in serving::migration_direct_buckets(set, 64)
            .into_iter()
            .enumerate()
        {
            let schedule = if rows > 32 && rows < 64 && set.count != 128 {
                "wide"
            } else if router::small_gap(set, phase, rows)
                && !router::with_base_admission(|| serving::migration_direct_buckets(set, 64))
                    .contains(&(phase, rows))
            {
                "gap"
            } else {
                "direct"
            };
            add(
                &mut t,
                "direct_admit",
                phase,
                &shape,
                64,
                schedule,
                [(rows, [("order".into(), Knob::Count(order as u32))].into())],
            );
        }
        for phase in phases() {
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
                    &mut t,
                    site,
                    phase,
                    &[("experts", set.count)],
                    64,
                    site,
                    (1..=64).filter_map(|r| {
                        serving::migration_direct_choice(site, set.count, phase, r)
                            .then(|| (r, Knobs::new()))
                    }),
                );
            }
            for site in ["router_finish", "router_gap", "router_wide", "router_r2"] {
                add(
                    &mut t,
                    site,
                    phase,
                    &shape,
                    64,
                    site,
                    (1..=64).filter_map(|rows| {
                        let chosen = match site {
                            "router_finish" => {
                                router::with_base_admission(|| router::qualified(set, phase, rows))
                            }
                            "router_gap" => router::small_gap(set, phase, rows),
                            "router_wide" => router::wide(set, phase, rows, 64),
                            _ => router::with_base_admission(|| router::r2(set, phase, rows)),
                        };
                        chosen.then(|| (rows, Knobs::new()))
                    }),
                );
            }
        }
    }
    t
}

#[test]
#[ignore = "one-time migration generator; writes to the external evidence directory"]
fn generate() {
    let out = std::path::PathBuf::from(
        std::env::var_os("CHOPINFER_TUNING_OUT").expect("external output directory"),
    );
    assert!(
        !out.canonicalize().unwrap().starts_with(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .canonicalize()
                .unwrap()
        )
    );
    std::fs::write(out.join("moe.json"), generated().render()).unwrap();
}

#[test]
fn differential() {
    let expected = generated();
    let actual = table(&FLASH, Family::Moe).unwrap();
    assert_eq!(actual.render(), expected.render());
    let mut cases = 0;
    for row in expected.rows() {
        let key = &row.key;
        for rows in 0..=key.bucket + 1 {
            assert_eq!(
                actual.select(&key.op, &key.weights, &key.shape, key.bucket, rows),
                expected.select(&key.op, &key.weights, &key.shape, key.bucket, rows)
            );
            cases += 1;
        }
    }
    assert!(cases > 100_000);
    eprintln!("MoE differential: {cases} cases");
}
