use crate::sm120::moe as serving;
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
        if legacy_streamed(&FLASH, ranks) {
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
                        legacy_prefill(&FLASH, ranks, set.count, kind, r).then(|| (r, Knobs::new()))
                    }),
                );
                for capacity in [512, 2048, 4096, 8192] {
                    for (site, selected) in [
                        ("pipeline", legacy_pipeline as fn(_, _, _, _, _) -> _),
                        ("grouped", legacy_grouped as fn(_, _, _, _, _) -> _),
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
            ("compact", legacy_compact(&FLASH, ranks).unwrap()),
            ("active", legacy_active(&FLASH, ranks).unwrap()),
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
    for route in legacy_routes(&FLASH) {
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
        for (order, (phase, rows)) in legacy_buckets(set, 64).into_iter().enumerate() {
            let schedule = if rows > 32 && rows < 64 && set.count != 128 {
                "wide"
            } else if legacy_small_gap(set, phase, rows)
                && !matches!(rows, 1 | 2 | 4 | 5 | 6 | 8 | 16 | 32 | 64)
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
                        legacy_direct_choice(site, set.count, phase, r).then(|| (r, Knobs::new()))
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
                            "router_finish" => legacy_qualified(set, phase, rows),
                            "router_gap" => legacy_small_gap(set, phase, rows),
                            "router_wide" => legacy_wide(set, phase, rows),
                            _ => legacy_r2(set, phase, rows),
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
    use crate::sm120::{moe_grouped_prefill, moe_pipeline, moe_router_finish as router};
    for ranks in [1, 2, 4] {
        assert_eq!(
            serving::streamed_qualified(&FLASH, ranks),
            legacy_streamed(&FLASH, ranks)
        );
        for set in moe::expert_sets(&FLASH) {
            for kind in [GatedMlpProjection::GateUp, GatedMlpProjection::Down] {
                for rows in 0..=8193 {
                    assert_eq!(
                        serving::prefill_selected(&FLASH, ranks, set.count, kind, rows),
                        legacy_prefill(&FLASH, ranks, set.count, kind, rows)
                    );
                }
                for capacity in [
                    0, 1, 8, 32, 64, 128, 512, 2048, 4095, 4096, 8191, 8192, 8193,
                ] {
                    for phase in phases() {
                        for graph in [false, true] {
                            let usage = OperationUse {
                                phase,
                                capacity,
                                graph,
                            };
                            assert_eq!(
                                moe_pipeline::selected(&FLASH, ranks, set, kind, usage),
                                legacy_pipeline(&FLASH, ranks, set, kind, usage)
                            );
                            assert_eq!(
                                moe_grouped_prefill::selected(&FLASH, ranks, set, kind, usage),
                                legacy_grouped(&FLASH, ranks, set, kind, usage)
                            );
                        }
                    }
                }
            }
        }
    }
    for set in moe::expert_sets(&FLASH) {
        for maximum in 32..=64 {
            assert_eq!(
                serving::migration_direct_buckets(set, maximum),
                legacy_buckets(set, maximum)
            );
        }
        for phase in phases() {
            for rows in 1..=64 {
                for site in [
                    "absorb",
                    "strip2",
                    "wide",
                    "small",
                    "gate_grid",
                    "gate_resident",
                    "gate_finish",
                ] {
                    assert_eq!(
                        serving::migration_direct_choice(site, set.count, phase, rows),
                        legacy_direct_choice(site, set.count, phase, rows)
                    );
                }
                let gap = legacy_small_gap(set, phase, rows);
                assert_eq!(
                    router::qualified(set, phase, rows),
                    gap || legacy_qualified(set, phase, rows)
                );
                assert_eq!(
                    router::r2(set, phase, rows),
                    gap || legacy_r2(set, phase, rows)
                );
                for maximum in 32..=64 {
                    assert_eq!(
                        router::wide(set, phase, rows, maximum),
                        rows <= maximum && legacy_wide(set, phase, rows)
                    );
                }
            }
        }
    }
}

fn legacy_streamed(g: &Geometry, ranks: u32) -> bool {
    g.id == model_deepseek41_contract::geometry::FLASH.id && ranks == 4
}

fn legacy_prefill(
    g: &Geometry,
    ranks: u32,
    experts: u32,
    kind: GatedMlpProjection,
    rows: u32,
) -> bool {
    legacy_streamed(g, ranks)
        && [512, 2048, 4096].contains(&rows)
        && !(experts == 128 && kind == GatedMlpProjection::Down && rows == 2048)
}

fn legacy_compact(g: &Geometry, ranks: u32) -> Result<Vec<quantized::Projection>, String> {
    if !legacy_streamed(g, ranks) {
        return Ok(Vec::new());
    }
    serving::projections(g, ranks)?
        .into_iter()
        .filter(|e| {
            e.row_tile == 16
                && e.query
                    .projection
                    .gemm
                    .grouped
                    .is_some_and(|g| g.experts == 128 && g.top_k == 3)
        })
        .map(|e| {
            e.with_compact_columns()
                .and_then(|e| e.with_max_rows(1))
                .map(|e| e.with_phase(kernel_core::precision::ExecutionPhase::Decode))
                .map_err(|e| e.to_string())
        })
        .collect()
}

fn legacy_active(g: &Geometry, ranks: u32) -> Result<Vec<quantized::Projection>, String> {
    if g.id != model_deepseek41_contract::geometry::FLASH.id || ranks != 4 {
        return Ok(Vec::new());
    }
    serving::active_projections(g, ranks)?
        .into_iter()
        .filter(|e| {
            e.gate_up()
                && e.tokens_capacity == DECODE_BATCH
                && e.query.projection.gemm.grouped.unwrap().experts == g.shape.moe.experts
        })
        .map(|e| e.with_max_rows(1).map_err(|e| e.to_string()))
        .collect()
}

fn legacy_routes(g: &Geometry) -> Vec<row_bias::Route> {
    moe::expert_sets(g)
        .into_iter()
        .flat_map(|set| {
            [8, 64].map(|capacity| row_bias::Route {
                local: LocalRoute {
                    experts: set.count,
                    local_experts: set.count,
                    top_k: set.top_k,
                    capacity,
                    pad: 64,
                },
            })
        })
        .collect()
}

fn legacy_pipeline(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
) -> bool {
    g.id == FLASH.id
        && ranks == 4
        && matches!(
            (set.count, set.top_k, usage.capacity),
            (384, 6, 8192) | (128, 3, 8192 | 4096)
        )
        && kind == GatedMlpProjection::GateUp
        && usage.phase == ExecutionPhase::Prefill
        && !usage.graph
}

fn legacy_grouped(
    g: &Geometry,
    ranks: u32,
    set: Experts,
    kind: GatedMlpProjection,
    usage: OperationUse,
) -> bool {
    g.id == FLASH.id
        && g.shape.hidden == FLASH.shape.hidden
        && g.shape.moe.intermediate == FLASH.shape.moe.intermediate
        && ranks == 4
        && set
            == Experts {
                count: 384,
                top_k: 6,
            }
        && kind == GatedMlpProjection::Down
        && usage.phase == ExecutionPhase::Prefill
        && usage.capacity == 8192
        && !usage.graph
}

fn legacy_direct_choice(site: &str, experts: u32, phase: ExecutionPhase, rows: u32) -> bool {
    use ExecutionPhase::{Decode, Verify};
    match site {
        "absorb" => experts == 384 && matches!((phase, rows), (Decode, 32) | (Verify, 32)),
        "strip2" => experts == 384 && matches!(rows, 16 | 32),
        "wide" => experts == 384 && phase == Verify && (33..=64).contains(&rows),
        "small" => rows <= 8,
        "gate_grid" => experts == 384 && matches!((phase, rows), (Decode, 8 | 32) | (Verify, 32)),
        "gate_resident" => {
            experts == 384
                && (matches!(rows, 16 | 32) || (phase == Verify && matches!(rows, 4 | 6)))
        }
        "gate_finish" => {
            experts == 384 && matches!((phase, rows), (Decode, 1 | 8 | 16) | (Verify, 4 | 6 | 16))
        }
        _ => unreachable!("direct selection site"),
    }
}

fn legacy_buckets(set: Experts, maximum_verify_rows: u32) -> Vec<(ExecutionPhase, u32)> {
    use ExecutionPhase::{Decode, Verify};
    let mut out = [1, 2, 4, 8, 16, 32].map(|n| (Decode, n)).to_vec();
    out.extend([1, 2, 4, 5, 6, 8, 16, 32, 64].map(|rows| (Verify, rows)));
    if set.count == 128 {
        out.push((Verify, 48));
    } else {
        out.extend((33..=maximum_verify_rows.min(63)).map(|rows| (Verify, rows)));
    }
    for rows in 2..=32 {
        if legacy_small_gap(set, Verify, rows) && !out.contains(&(Verify, rows)) {
            out.push((Verify, rows));
        }
    }
    out
}

fn legacy_small_gap(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    set.count == 384
        && set.top_k == 6
        && matches!((phase,rows),
        (ExecutionPhase::Decode,2|4)|(ExecutionPhase::Verify,2|3|5|7|9..=15|17..=31))
}
fn legacy_qualified(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    match (set.count, set.top_k, phase) {
        (384, 6, ExecutionPhase::Decode) => [1, 8, 16, 32].contains(&rows),
        (384, 6, ExecutionPhase::Verify) => [4, 6, 8, 16, 32].contains(&rows),
        (128, 3, ExecutionPhase::Decode) => rows == 1,
        _ => false,
    }
}
fn legacy_r2(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    legacy_qualified(set, phase, rows)
        && matches!(
            (set.count, set.top_k, phase, rows),
            (384, 6, ExecutionPhase::Decode, 1 | 8 | 16 | 32)
                | (384, 6, ExecutionPhase::Verify, 4 | 6 | 16 | 32)
        )
}
fn legacy_wide(set: Experts, phase: ExecutionPhase, rows: u32) -> bool {
    set.count == 384
        && set.top_k == 6
        && phase == ExecutionPhase::Verify
        && (33..=64).contains(&rows)
}

use kernels_sm120::moe::chained::{LocalRoute, quantized, row_bias};
use model_deepseek41_contract::{geometry::Geometry, moe::Experts, serving::DECODE_BATCH};
