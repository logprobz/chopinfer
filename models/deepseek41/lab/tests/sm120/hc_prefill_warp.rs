//! Current HC prefill kernels versus separately locked staging experiments.
#[path = "hc_prefill_warp/artifacts.rs"]
mod artifacts;
#[path = "hc_prefill_warp/catalog.rs"]
#[allow(dead_code)]
mod catalog;
#[path = "hc_prefill_warp/fixture.rs"]
mod fixture;
use fixture::{Fixture, compare};
use kernel_harness::{LabResult, Session, eviction::EvictionMode, poison};
use kernels_sm120::sinkhorn_hc::prefill_warp::{Entry, Kind};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
struct Case {
    rows: u32,
    origin: u32,
    seed: u64,
    pattern: &'static str,
}
impl Case {
    fn json(self) -> Value {
        json!({"rows":self.rows,"origin":self.origin,"seed":self.seed,"pattern":self.pattern})
    }
}
fn cases() -> Vec<Case> {
    let mut cases: Vec<_> = catalog::rows()
        .into_iter()
        .flat_map(|rows| {
            [17, 97].map(move |seed| Case {
                rows,
                origin: 0,
                seed,
                pattern: "finite",
            })
        })
        .collect();
    for rows in [17, 8192] {
        for pattern in [
            "zero",
            "tiny",
            "wide",
            "nan",
            "inf",
            "weight_nonfinite",
            "cancellation",
        ] {
            cases.push(Case {
                rows,
                origin: 0,
                seed: 173,
                pattern,
            });
        }
    }
    for (rows, origin) in [
        (64, 8128),
        (320, 7872),
        (2496, 5696),
        (1024, 4096),
        (192, 64),
    ] {
        cases.push(Case {
            rows,
            origin,
            seed: 313,
            pattern: "finite",
        });
    }
    cases
}
fn probe_cases() -> Vec<Case> {
    [17, 320, 8192]
        .into_iter()
        .flat_map(|rows| {
            [17, 97].map(move |seed| Case {
                rows,
                origin: 0,
                seed,
                pattern: "finite",
            })
        })
        .collect()
}
fn live_counts(rows: u32) -> [u32; 5] {
    [rows, 0, rows - 1, rows / 2, rows]
}
fn phases() -> [&'static str; 5] {
    [
        "eager",
        "graph-zero",
        "graph-tail",
        "graph-half",
        "graph-full",
    ]
}
fn coverage() -> Value {
    let plans:Vec<_>=catalog::candidates(false).into_iter().chain(catalog::candidates(true)).chain([catalog::mix_control(),catalog::expand_control()]).map(|e| {
        json!({"symbol":e.reference.symbol,"configured_shared_bytes":e.shared_bytes(),"plans":catalog::rows().into_iter().map(|rows| {
            let p=e.plan(rows,0).unwrap();json!({"rows":rows,"grid":p.grid,"block":p.block,"cluster":p.cluster,"dynamic_shared_bytes":p.dynamic_shared_bytes})
        }).collect::<Vec<_>>()})
    }).collect();
    json!({"cases":cases().into_iter().map(Case::json).collect::<Vec<_>>(),"probe_cases":probe_cases().into_iter().map(Case::json).collect::<Vec<_>>(),
        "variants":catalog::candidates(false).iter().map(|e|&e.reference.symbol).collect::<Vec<_>>(),
        "probe_variants":catalog::candidates(true).iter().map(|e|&e.reference.symbol).collect::<Vec<_>>(),"phases":phases(),
        "timing_rows":catalog::profiled_rows(),"plans":plans,"sharded_bitwise":cases().len()*catalog::candidates(false).len()*5,"replicated_bitwise":probe_cases().len()*catalog::candidates(true).len()*5,"pdl":4*catalog::candidates(false).len(),"negative":4,"timings":catalog::profiled_rows().len()*catalog::candidates(false).len(),"partitions":partitions(),"isolation_rows":[0,7,8,15,16,32]})
}
fn reference(
    e: &Entry,
    rows: u32,
    probe: bool,
) -> (
    kernels_sm120::sinkhorn_hc::lagged::Entry,
    kernel_core::contract::LaunchPlan,
) {
    if matches!(e.kind, Kind::Expand { .. }) {
        catalog::old_expand(rows)
    } else {
        catalog::old_mix(rows, probe)
    }
}
fn comparisons(s: &Session, m: &artifacts::Loaded, case: Case, probe: bool) -> LabResult<()> {
    let f = Fixture::new(s, case.rows, case.origin, case.seed, case.pattern)?;
    let entries = catalog::candidates(probe);
    let old_mix = catalog::old_mix(case.rows, probe);
    let old_expand = catalog::old_expand(case.rows);
    let old_mix_graph =
        s.capture(|stream| f.fire(s, m, stream, &old_mix.0.symbol, old_mix.1, 0, probe, false))?;
    let old_expand_graph = if probe {
        None
    } else {
        Some(s.capture(|stream| {
            f.fire(
                s,
                m,
                stream,
                &old_expand.0.symbol,
                old_expand.1,
                0,
                false,
                false,
            )
        })?)
    };
    let graphs = entries
        .iter()
        .map(|e| {
            s.capture(|stream| {
                f.fire(
                    s,
                    m,
                    stream,
                    &e.reference.symbol,
                    e.plan(case.rows, 0).unwrap(),
                    1,
                    probe,
                    false,
                )
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    for (phase, live) in live_counts(case.rows).into_iter().enumerate() {
        f.set_live(s, live)?;
        f.outputs[0].reset(s, false, probe)?;
        if phase == 0 {
            f.fire(
                s,
                m,
                &s.stream,
                &old_mix.0.symbol,
                old_mix.1,
                0,
                probe,
                false,
            )?;
        } else {
            old_mix_graph.launch(&s.stream).map_err(|e| e.to_string())?;
        }
        let expected_mix = f.outputs[0].read(s, false, probe, live)?;
        let expected_expand = if let Some(graph) = &old_expand_graph {
            f.outputs[0].reset(s, true, false)?;
            if phase == 0 {
                f.fire(
                    s,
                    m,
                    &s.stream,
                    &old_expand.0.symbol,
                    old_expand.1,
                    0,
                    false,
                    false,
                )?;
            } else {
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            f.outputs[0].read(s, true, false, live)?
        } else {
            vec![]
        };
        for (e, graph) in entries.iter().zip(&graphs) {
            let expand = matches!(e.kind, Kind::Expand { .. });
            f.outputs[1].reset(s, expand, probe)?;
            if phase == 0 {
                f.fire(
                    s,
                    m,
                    &s.stream,
                    &e.reference.symbol,
                    e.plan(case.rows, live).map_err(|e| e.to_string())?,
                    1,
                    probe,
                    false,
                )?;
            } else {
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            let actual = f.outputs[1].read(s, expand, probe, live)?;
            compare(
                if expand {
                    &expected_expand
                } else {
                    &expected_mix
                },
                &actual,
            )
            .map_err(|error| {
                format!(
                    "{} {:?} live={live} phase={}: {error}",
                    e.reference.symbol,
                    case.json(),
                    phases()[phase]
                )
            })?;
            let mut record = case.json();
            record["symbol"] = json!(e.reference.symbol);
            record["live"] = json!(live);
            record["phase"] = json!(phases()[phase]);
            record["probe"] = json!(probe);
            println!("HC_PREFILL_BITS {record}");
        }
    }
    Ok(())
}
fn controls(s: &Session, m: &artifacts::Loaded) -> LabResult<()> {
    for seed in [17, 97] {
        let f = Fixture::new(s, 64, 0, seed, "finite")?;
        for (e, probe) in [
            (catalog::mix_control(), true),
            (catalog::expand_control(), false),
        ] {
            let expand = matches!(e.kind, Kind::Expand { .. });
            let old = reference(&e, 64, probe);
            for arm in 0..2 {
                f.outputs[arm].reset(s, expand, probe)?;
            }
            f.fire(s, m, &s.stream, &old.0.symbol, old.1, 0, probe, false)?;
            f.fire(
                s,
                m,
                &s.stream,
                &e.reference.symbol,
                e.plan(64, 64).unwrap(),
                1,
                probe,
                false,
            )?;
            if compare(
                &f.outputs[0].read(s, expand, probe, 64)?,
                &f.outputs[1].read(s, expand, probe, 64)?,
            )
            .is_ok()
            {
                return Err(format!(
                    "arithmetic control escaped: {} seed={seed}",
                    e.reference.symbol
                ));
            }
            println!(
                "HC_PREFILL_NEGATIVE {}",
                json!({"symbol":e.reference.symbol,"seed":seed})
            );
        }
    }
    Ok(())
}
fn dependencies(s: &Session, m: &artifacts::Loaded) -> LabResult<()> {
    let module = s.module(kernels_sm120::probes::CUBIN)?;
    let producer = poison::LiveProducer::with_function(
        s,
        module
            .function(kernels_sm120::probes::LIVE)
            .map_err(|e| e.to_string())?,
    )?;
    for (rows, origin) in [(320, 7872), (8192, 0)] {
        let f = Fixture::new(s, rows, origin, 197, "finite")?;
        let staged = f.staged_inputs(s)?;
        let inputs = f.rewritten_inputs();
        let pairs: Vec<_> = staged.iter().zip(inputs).collect();
        let rewrites = poison::Rewrites::new(s, &pairs)?;
        for live in [rows - 1, rows] {
            for e in catalog::candidates(false) {
                let expand = matches!(e.kind, Kind::Expand { .. });
                let old = reference(&e, rows, false);
                f.set_live(s, live)?;
                f.outputs[0].reset(s, expand, false)?;
                f.fire(s, m, &s.stream, &old.0.symbol, old.1, 0, false, false)?;
                let expected = f.outputs[0].read(s, expand, false, live)?;
                f.outputs[1].reset(s, expand, false)?;
                producer.launch_rewriting(s, &f.live, &rewrites)?;
                f.fire(
                    s,
                    m,
                    &s.stream,
                    &e.reference.symbol,
                    e.plan(rows, live).unwrap(),
                    1,
                    false,
                    true,
                )?;
                compare(&expected, &f.outputs[1].read(s, expand, false, live)?).map_err(
                    |error| {
                        format!(
                            "PDL {} rows={rows} live={live}: {error}",
                            e.reference.symbol
                        )
                    },
                )?;
                println!(
                    "HC_PREFILL_PDL {}",
                    json!({"rows":rows,"origin":origin,"live":live,"symbol":e.reference.symbol})
                );
            }
        }
    }
    Ok(())
}
fn timings(s: &Session, m: &artifacts::Loaded) -> LabResult<()> {
    for rows in catalog::profiled_rows() {
        let f = Fixture::new(s, rows, 0, 1337, "finite")?;
        for e in catalog::candidates(false) {
            let expand = matches!(e.kind, Kind::Expand { .. });
            let old = reference(&e, rows, false);
            let plan = e.plan(rows, rows).unwrap();
            for arm in 0..2 {
                f.outputs[arm].reset(s, expand, false)?;
            }
            f.fire(s, m, &s.stream, &old.0.symbol, old.1, 0, false, false)?;
            f.fire(s, m, &s.stream, &e.reference.symbol, plan, 1, false, false)?;
            compare(
                &f.outputs[0].read(s, expand, false, rows)?,
                &f.outputs[1].read(s, expand, false, rows)?,
            )?;
            let graphs = [
                s.capture(|stream| f.fire(s, m, stream, &old.0.symbol, old.1, 0, false, false))?,
                s.capture(|stream| {
                    f.fire(s, m, stream, &e.reference.symbol, plan, 1, false, false)
                })?,
            ];
            let timing =
                kernel_harness::time_ab(s, 2, 9, [EvictionMode::Read; 2], |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })?;
            compare(
                &f.outputs[0].read(s, expand, false, rows)?,
                &f.outputs[1].read(s, expand, false, rows)?,
            )?;
            println!(
                "HC_PREFILL_TIMING {}",
                json!({"rows":rows,"symbol":e.reference.symbol,"reference":old.0.symbol,"old_us":timing.arms[0].median_ms*1000.,"new_us":timing.arms[1].median_ms*1000.,"ratio":timing.paired_ratio,"rounds":9,"expand":expand,
                "minimum_bytes":if expand {u64::from(rows)*(9*5120*2+20*4)} else {u64::from(rows)*(9*5120*2+24*4)+24*4*5120*4}})
            );
        }
    }
    Ok(())
}
#[test]
fn cpu_hc_warp_launches_cover_skipped_rows_and_bound_shared_memory() {
    let limits = artifacts::limits();
    let configured: std::collections::BTreeMap<_, _> = limits
        .into_iter()
        .filter(|(_, bytes)| *bytes > 48 * 1024)
        .collect();

    for e in catalog::candidates(false)
        .into_iter()
        .chain(catalog::candidates(true))
        .chain([catalog::mix_control(), catalog::expand_control()])
    {
        for rows in catalog::rows() {
            let p = e.plan(rows, 0).unwrap();
            artifacts::check(&e.reference.symbol, p, &configured).unwrap();
            if p.dynamic_shared_bytes > 48 * 1024 {
                assert!(artifacts::check(&e.reference.symbol, p, &Default::default()).is_err());
            }
        }
    }
    let mut bad = catalog::candidates(false)[0].plan(8192, 0).unwrap();
    bad.dynamic_shared_bytes = 49153;
    assert!(artifacts::check("missing-attribute", bad, &Default::default()).is_err());
    let rows = catalog::rows();
    assert!(rows.contains(&8192));
    for j in 0..=20 {
        assert!(rows.contains(&(64 + 128 * j)));
    }
    assert_eq!(catalog::old_mix(192, false).1.grid, [192, 1, 1]);
    assert_eq!(catalog::old_mix(320, false).1.grid, [40, 1, 1]);
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_WARP_ARTIFACTS"]
fn frozen_hc_warp_artifacts_match_cpu_inventory() {
    if std::env::var_os("CHOPINFER_HC_WARP_EXPORT").is_some() {
        artifacts::export().unwrap();
    }
    artifacts::verify().unwrap();
    println!("HC_PREFILL_MODULES {}", artifacts::expected().len());
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_WARP_ARTIFACTS"]
fn hc_warp_comparison_and_timing_on_gpu() {
    let ran = kernel_harness::gpu_test_with(artifacts::cubins().unwrap(), 8 << 30, |s| {
        let m = artifacts::Loaded::new(s)?;
        let shard = std::env::var("CHOPINFER_HC_WARP_SHARD")
            .ok()
            .map(|v| v.parse::<usize>().expect("shard is 0 or 1"));
        assert!(shard.is_none_or(|v| v < 2));
        for (index, case) in cases().into_iter().enumerate() {
            if shard.is_none_or(|v| index % 2 == v) {
                comparisons(s, &m, case, false)?;
            }
        }
        for case in probe_cases() {
            comparisons(s, &m, case, true)?;
        }
        controls(s, &m)?;
        dependencies(s, &m)?;
        invariances(s, &m)?;
        timings(s, &m)?;
        println!("HC_PREFILL_DONE {}", json!({"shard":shard}));
        Ok(())
    })
    .unwrap();
    if std::env::var_os("CHOPINFER_KERNEL_REQUIRE").is_some() {
        assert!(ran.is_some(), "required HC comparison skipped");
    }
}

fn partitions() -> Vec<Vec<u32>> {
    vec![
        vec![4096, 4096],
        vec![2048; 4],
        vec![17, 127, 1000, 2048, 5000],
    ]
}
fn invariances(s: &Session, m: &artifacts::Loaded) -> LabResult<()> {
    let f = Fixture::new(s, 8192, 17, 227, "finite")?;
    for e in catalog::candidates(false) {
        let expand = matches!(e.kind, Kind::Expand { .. });
        let old = reference(&e, 8192, false);
        f.set_live(s, 8192)?;
        f.outputs[0].reset(s, expand, false)?;
        f.fire(s, m, &s.stream, &old.0.symbol, old.1, 0, false, false)?;
        let expected = f.outputs[0].read(s, expand, false, 8192)?;
        for parts in partitions() {
            f.outputs[1].reset(s, expand, false)?;
            let mut first = 0;
            for &rows in &parts {
                f.set_live(s, rows)?;
                f.fire_range(
                    m,
                    &s.stream,
                    &e.reference.symbol,
                    e.plan(rows, rows).unwrap(),
                    1,
                    false,
                    true,
                    first,
                    rows,
                )?;
                first += rows;
            }
            compare(&expected, &f.outputs[1].read(s, expand, false, 8192)?)?;
            println!(
                "HC_PREFILL_PARTITION {}",
                json!({"symbol":e.reference.symbol,"rows":8192,"origin":17,"partitions":parts})
            );
        }
    }
    drop(f);
    let f = Fixture::new(s, 33, 0, 337, "finite")?;
    let inputs = [&f.x, &f.y, &f.pre, &f.post, &f.comb];
    let original = inputs
        .iter()
        .map(|span| {
            span.read(s)
                .map(|b| b[span.start..span.start + span.bytes].to_vec())
        })
        .collect::<LabResult<Vec<_>>>()?;
    for e in catalog::candidates(false) {
        let expand = matches!(e.kind, Kind::Expand { .. });
        let old = reference(&e, 33, false);
        for (span, bytes) in inputs.iter().zip(&original) {
            span.upload(s, bytes)?;
        }
        f.outputs[0].reset(s, expand, false)?;
        f.fire(s, m, &s.stream, &old.0.symbol, old.1, 0, false, false)?;
        let expected = f.outputs[0].read(s, expand, false, 33)?;
        for keep in [0, 7, 8, 15, 16, 32] {
            for (span, bytes) in inputs.iter().zip(&original) {
                span.upload(s, &fixture::neighbors(bytes, span.stride, keep))?;
            }
            f.outputs[1].reset(s, expand, false)?;
            f.fire(
                s,
                m,
                &s.stream,
                &e.reference.symbol,
                e.plan(33, 33).unwrap(),
                1,
                false,
                false,
            )?;
            let actual = f.outputs[1].read(s, expand, false, 33)?;
            let spans: Vec<_> = if expand {
                vec![&f.outputs[1].expanded]
            } else {
                f.outputs[1].mix.iter().take(4).collect()
            };
            for ((a, b), span) in expected.iter().zip(&actual).zip(spans) {
                let at = span.start + keep as usize * span.stride;
                if a[at..at + span.stride] != b[at..at + span.stride] {
                    return Err(format!(
                        "neighbor affected row={keep} {}",
                        e.reference.symbol
                    ));
                }
            }
            println!(
                "HC_PREFILL_ISOLATION {}",
                json!({"symbol":e.reference.symbol,"rows":33,"kept_row":keep})
            );
        }
    }
    Ok(())
}
