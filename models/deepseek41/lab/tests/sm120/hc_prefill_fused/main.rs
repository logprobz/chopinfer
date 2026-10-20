//! Exact HC prefill composition on checkpoint weights and seeded streams.
mod artifacts;
#[allow(dead_code)]
mod catalog;
mod device;
#[path = "../fusion_checks.rs"]
#[allow(dead_code)]
mod fusion_checks;
use device::{Device, Fixture, STRIDES};
use kernel_harness::{LabResult, Session, eviction::EvictionMode};
use serde_json::{Value, json};
const WIDTH: usize = 5120;
const FLAT: usize = WIDTH * 4;
const OFFSET: usize = 3;
const ROLES: [&str; 10] = [
    "expanded",
    "collapsed",
    "pre",
    "post",
    "comb",
    "normed",
    "widened",
    "codes",
    "scales",
    "status",
];
const PATTERNS: [&str; 8] = [
    "zero",
    "tiny",
    "maximum",
    "nan",
    "inf",
    "weight_nan",
    "cancellation",
    "sticky",
];
/// A start marker with seconds since the first marker, so a killed worker
/// still names the case it was running and how long earlier cases took.
fn progress(event: &str, label: &str) {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    eprintln!(
        "HC_PREFILL_FUSED_PROGRESS {}",
        json!({"event":event,"case":label,"elapsed_s":start.elapsed().as_secs_f64()})
    );
}
fn compare(a: &[Vec<u8>], b: &[Vec<u8>]) -> LabResult<()> {
    fusion_checks::compare_roles(&ROLES, a, b)
}
fn active(mut output: Vec<Vec<u8>>, rows: usize) -> Vec<Vec<u8>> {
    output[9] = output[9][OFFSET * 4..(OFFSET + rows) * 4].to_vec();
    output
}
fn check(
    d: &Device<'_>,
    label: &str,
    sticky: bool,
    memcheck: bool,
    failures: &mut fusion_checks::Failures,
) -> LabResult<Value> {
    progress("case", label);
    let before = failures.len();
    let graphs = (0..2)
        .map(|arm| d.s.capture(|stream| d.launch(arm, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    // Full 8K comparisons include both eager and graph execution. Small edge
    // fixtures cover zero, tail, half and replayed live counts with poison.
    let lives = if memcheck || d.rows > 320 {
        vec![d.rows]
    } else {
        fusion_checks::live_counts(d.rows).to_vec()
    };
    let mut digest = String::new();
    for live in lives {
        d.set_live(live)?;
        let mut previous: Option<Vec<Vec<u8>>> = None;
        for replay in 0..if memcheck { 1 } else { 2 } {
            for (arm, graph) in graphs.iter().enumerate() {
                d.reset(arm, sticky)?;
                if replay == 0 {
                    d.launch(arm, &d.s.stream)?;
                } else {
                    graph.launch(&d.s.stream).map_err(|e| e.to_string())?;
                }
            }
            d.s.stream.synchronize().map_err(|e| e.to_string())?;
            let old = d.outputs[0].read(d.s)?;
            let new = d.outputs[1].read(d.s)?;
            let context = format!("{label}:live={live}:replay={replay}");
            failures.record(&context, compare(&old, &new));
            failures.record(&context, d.guards(&old, live, sticky));
            failures.record(&context, d.guards(&new, live, sticky));
            if let Some(previous) = previous.as_ref() {
                failures.record(&context, compare(previous, &new));
            }
            if live == d.rows && (memcheck || replay == 1) {
                digest = kernel_build::digest(&new.iter().flatten().copied().collect::<Vec<_>>());
            }
            previous = Some(old);
        }
    }
    let record =
        json!({"case":label,"rows":d.rows,"passed":before==failures.len(),"output_sha256":digest});
    eprintln!("HC_PREFILL_FUSED_CASE {record}");
    Ok(record)
}
fn timing(d: &Device<'_>, label: &str) -> LabResult<Vec<Value>> {
    progress("timing", label);
    let mut records = vec![];
    d.set_live(d.rows)?;
    for arm in 0..2 {
        d.reset(arm, false)?;
    }
    let graphs = (0..2)
        .map(|arm| d.s.capture(|stream| d.launch(arm, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    for hot in [false, true] {
        let value = kernel_harness::with_timing_label(&format!("{label}:hot={hot}"), || {
            kernel_harness::time_ab_prepared(
                d.s,
                3,
                9,
                [EvictionMode::Read; 2],
                |arm, stream| {
                    if hot {
                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                    } else {
                        Ok(())
                    }
                },
                |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
            )
        })?;
        let record = json!({"case":label,"rows":d.rows,"hot":hot,"old_launches":3,"new_launches":d.entry.launches(),
            "old_us":value.arms[0].median_ms*1000.,"new_us":value.arms[1].median_ms*1000.,"ratio":value.paired_ratio});
        eprintln!("HC_PREFILL_FUSED_TIMING {record}");
        records.push(record);
    }
    Ok(records)
}
fn invariances(
    s: &Session,
    site: &artifacts::Site,
    failures: &mut fusion_checks::Failures,
) -> LabResult<usize> {
    let mut count = 0;
    let f = Fixture::new(site, 64, 197, "finite");
    for e in catalog::candidates() {
        let d = Device::new(s, &e, &f, true, false)?;
        let expected = active(d.once(0, false)?, 64);
        let mut joined = vec![vec![]; 10];
        for range in [0..7, 7..64] {
            let part = f.select(&range.collect::<Vec<_>>());
            let p = Device::new(s, &e, &part, true, false)?;
            for (out, bytes) in joined
                .iter_mut()
                .zip(active(p.once(1, false)?, p.rows as usize))
            {
                out.extend(bytes);
            }
        }
        failures.record(
            &format!("{}:chunk", e.symbol()),
            compare(&expected, &joined),
        );
        count += 1;
        let reversed = f.select(&(0..64).rev().collect::<Vec<_>>());
        let moved = Device::new(s, &e, &reversed, true, false)?;
        let actual = active(moved.once(1, false)?, 64);
        let restored = actual
            .iter()
            .zip(STRIDES.into_iter().chain([4]))
            .map(|(b, stride)| b.chunks_exact(stride).rev().flatten().copied().collect())
            .collect::<Vec<Vec<u8>>>();
        failures.record(
            &format!("{}:position", e.symbol()),
            compare(&expected, &restored),
        );
        count += 1;
        // Distinct buffers and zero global workspace are safe across streams.
        d.reset(1, false)?;
        moved.reset(1, false)?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let other = s.context.stream().map_err(|e| e.to_string())?;
        d.launch(1, &s.stream)?;
        moved.launch(1, &other)?;
        other.synchronize().map_err(|e| e.to_string())?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        failures.record(
            &format!("{}:concurrent-first", e.symbol()),
            compare(&expected, &active(d.outputs[1].read(s)?, 64)),
        );
        failures.record(
            &format!("{}:concurrent-second", e.symbol()),
            compare(&actual, &active(moved.outputs[1].read(s)?, 64)),
        );
        count += 1;
        let alias = Device::new(s, &e, &f, true, true)?;
        alias.set_live(63)?;
        let old = alias.once(0, false)?;
        let new = alias.once(1, false)?;
        failures.record(
            &format!("{}:coefficient-alias", e.symbol()),
            compare(&old, &new),
        );
        failures.record(
            &format!("{}:alias-guards", e.symbol()),
            alias.guards(&new, 63, false),
        );
        count += 1;
    }
    Ok(count)
}
fn run(s: &Session, memcheck: bool) -> LabResult<()> {
    let sites = artifacts::corpus()?;
    let mut failures = fusion_checks::Failures::new("HC_PREFILL_FUSED");
    let mut reports = vec![];
    let mut timings = vec![];
    for (site_index, site) in sites.iter().enumerate() {
        if memcheck && site_index != 0 {
            continue;
        }
        for rows in catalog::ROWS {
            if memcheck && rows != 8192 {
                continue;
            }
            progress("fixture", &format!("{}:rows={rows}", site.id));
            let f = Fixture::new(site, rows, 17 + site_index as u64, "finite");
            for e in catalog::candidates() {
                for widen in [false, true] {
                    let d = Device::new(s, &e, &f, widen, false)?;
                    let label =
                        format!("{}:rows={rows}:finite:{}:widen={widen}", site.id, e.label());
                    reports.push(check(&d, &label, false, memcheck, &mut failures)?);
                    if !memcheck
                        && (rows == 8192 || (site_index == 0 && [64, 320, 2496].contains(&rows)))
                    {
                        timings.extend(timing(&d, &label)?);
                    }
                }
            }
        }
    }
    let mut invariance_count = 0;
    let mut negatives = 0;
    if !memcheck {
        for pattern in PATTERNS {
            let f = Fixture::new(&sites[0], 7, 173, pattern);
            for e in catalog::candidates() {
                for widen in [false, true] {
                    let d = Device::new(s, &e, &f, widen, false)?;
                    let label = format!("edge:rows=7:{pattern}:{}:widen={widen}", e.label());
                    reports.push(check(
                        &d,
                        &label,
                        pattern == "sticky",
                        false,
                        &mut failures,
                    )?);
                }
            }
        }
        progress("invariances", &sites[0].id);
        invariance_count = invariances(s, &sites[0], &mut failures)?;
        progress("controls", &sites[1].id);
        let f = Fixture::new(&sites[1], 64, 97, "finite");
        for e in catalog::candidate_controls() {
            let d = Device::new(s, &e, &f, true, false)?;
            let sticky = e.needs_sticky();
            failures.require(
                e.symbol(),
                compare(&d.once(0, sticky)?, &d.once(1, sticky)?).is_err(),
                "mutation escaped equality gate",
            );
            negatives += 1;
        }
    }
    // Per candidate and widening: 24 site cases (3 sites x 8 row counts; one
    // 8192-row site under memcheck), 8 edge patterns and 12 timing records
    // (6 timed cases, cold and hot); 4 invariances per candidate and one
    // control per mutation.
    let arms = catalog::candidates().len();
    let expected = if memcheck { 2 * arms } else { 64 * arms };
    failures.require(
        "coverage",
        reports.len() == expected,
        "executed case count differs",
    );
    failures.require(
        "invariance coverage",
        invariance_count == if memcheck { 0 } else { 4 * arms },
        "invariance count differs",
    );
    failures.require(
        "control coverage",
        negatives
            == if memcheck {
                0
            } else {
                catalog::candidate_controls().len()
            },
        "negative count differs",
    );
    failures.require(
        "timing coverage",
        timings.len() == if memcheck { 0 } else { 24 * arms },
        "timing count differs",
    );
    let value = json!({"schema":"chopinfer.hc-prefill-fused-result.v1","memcheck":memcheck,"reports":reports,
        "timings":timings,"invariances":invariance_count,"negatives":negatives,"failures":failures.messages(),"passed":failures.len()==0});
    s.receipt(
        "hc-prefill-fused",
        value
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )?;
    eprintln!(
        "HC_PREFILL_FUSED_COMPLETE memcheck={memcheck} cases={expected} invariances={invariance_count} negatives={negatives} timings={} passed={}",
        if memcheck { 0 } else { 24 * arms },
        failures.len() == 0
    );
    failures.finish()
}
#[test]
fn reject_impossible_parameters_and_shared_scratch_on_cpu() {
    for e in catalog::candidates() {
        for rows in catalog::ROWS {
            for live in fusion_checks::live_counts(rows) {
                assert_eq!(e.plan(rows, live).unwrap().workspace_bytes, 0);
            }
            assert!(e.plan(rows, rows + 1).is_err());
        }
        assert!(e.plan(0, 0).is_err());
        assert!(e.plan(8193, 0).is_err());
    }
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_PREFILL_FUSED"]
fn stage_and_validate_artifacts_on_cpu() {
    artifacts::stage().unwrap();
    artifacts::verify().unwrap();
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_PREFILL_FUSED"]
fn validate_frozen_corpus_and_modules_on_cpu() {
    artifacts::verify().unwrap();
    println!("HC_PREFILL_FUSED_FROZEN sites=3 PASS");
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_PREFILL_FUSED"]
fn bitwise_checkpoint_seeded_8k_edges_and_paired_timing_on_gpu() {
    artifacts::verify().unwrap();
    assert!(kernel_harness::selected_ordinals().unwrap().is_some());
    assert!(
        kernel_harness::gpu_test_with(artifacts::probes(), 1 << 30, |s| run(s, false))
            .unwrap()
            .is_some()
    );
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_PREFILL_FUSED"]
fn memcheck_seeded_8k_on_gpu() {
    artifacts::verify().unwrap();
    assert!(kernel_harness::selected_ordinals().unwrap().is_some());
    assert!(
        kernel_harness::gpu_test_with(artifacts::probes(), 1 << 30, |s| run(s, true))
            .unwrap()
            .is_some()
    );
}
/// Host-side cost of one full-8K case outside the GPU: seeded fixture
/// generation, the poisoned upload images and readback copies of both arms
/// for both replays, the comparisons, guard scans and the output digest. The
/// dry run scales it to the whole matrix before a worker timeout is accepted.
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_HC_PREFILL_FUSED"]
fn host_budget_on_cpu() {
    artifacts::verify().unwrap();
    let sites = artifacts::corpus().unwrap();
    let rows = 8192_u32;
    let clock = std::time::Instant::now();
    let fixture = Fixture::new(&sites[0], rows, 17, "finite");
    let fixture_s = clock.elapsed().as_secs_f64();
    assert_eq!(fixture.rows, rows);
    let n = rows as usize;
    let sizes = STRIDES
        .iter()
        .map(|stride| stride * n)
        .chain([(n + OFFSET + 2) * 4])
        .collect::<Vec<_>>();
    let clock = std::time::Instant::now();
    let mut previous: Option<Vec<Vec<u8>>> = None;
    for replay in 0..2 {
        let mut arms = Vec::new();
        for _ in 0..2 {
            let poison = sizes.iter().map(|&n| vec![0xa5; n]).collect::<Vec<_>>();
            let mut copy = sizes.iter().map(|&n| vec![0; n]).collect::<Vec<_>>();
            for (out, image) in copy.iter_mut().zip(&poison) {
                out.copy_from_slice(image);
            }
            let status = copy.last_mut().unwrap();
            for (index, word) in status.chunks_exact_mut(4).enumerate() {
                let value = if index < OFFSET || index >= OFFSET + n {
                    0x1357_2468u32
                } else {
                    0
                };
                word.copy_from_slice(&value.to_ne_bytes());
            }
            arms.push(copy);
        }
        compare(&arms[0], &arms[1]).unwrap();
        for arm in &arms {
            fusion_checks::guards(&ROLES, arm, rows, rows, &[6], OFFSET, false).unwrap();
        }
        if let Some(previous) = &previous {
            compare(previous, &arms[1]).unwrap();
        }
        if replay == 1 {
            std::hint::black_box(kernel_build::digest(
                &arms[1].iter().flatten().copied().collect::<Vec<_>>(),
            ));
        }
        previous = Some(arms.swap_remove(0));
    }
    let case_s = clock.elapsed().as_secs_f64();
    println!(
        "HC_PREFILL_FUSED_HOST_BUDGET {}",
        json!({"rows":rows,"bytes_per_row":sizes.iter().sum::<usize>()/n,"fixture_s":fixture_s,"case_host_s":case_s})
    );
}
