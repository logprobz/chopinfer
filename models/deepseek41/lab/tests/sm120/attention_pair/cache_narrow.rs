//! Qualify cached selected keys only at Decode-2/4 and Verify-16.
use super::*;
use kernel_core::backend::Launch;
use kernels_sm120::latent::{sources::Schedule, sources_cached};

fn shapes(context: usize) -> Vec<Case> {
    [(2, false, 1), (4, false, 1), (4, true, 4)]
        .into_iter()
        .flat_map(|(sequences, verify, verify_width)| {
            [0, 1, 2].map(move |ratio| Case {
                context,
                sequences,
                verify,
                verify_width,
                ratio,
                holes: false,
                inactive: 0,
            })
        })
        .collect()
}

fn cases(memcheck: bool) -> Vec<Case> {
    let contexts: &[usize] = if memcheck { &[65] } else { &[64, 513, 16384] };
    contexts
        .iter()
        .flat_map(|&context| shapes(context))
        .flat_map(|case| {
            BTreeSet::from([0, 1, case.rows() / 2, case.rows() - 1, case.rows()])
                .into_iter()
                .map(move |inactive| Case {
                    inactive,
                    holes: inactive != 0 || case.context == 513,
                    ..case
                })
        })
        .collect()
}

fn entry(case: Case) -> sources_cached::Entry {
    let mut source = reference(case);
    source.schedule = Schedule::IndependentHeads;
    sources_cached::Entry {
        source,
        profile: false,
    }
}

fn fixture(case: Case, rank: u32) -> Fixture {
    assert!(rank < 4);
    let e = entry(case);
    let seed = 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(u64::from(rank) + 1);
    let mut host = Fixture::with_seed(case, &e.source, None, seed);
    let (decoded, valid) = e.cache_bytes(case.rows() as u32).unwrap();
    host.data.insert("decoded", vec![0xa5; decoded as usize]);
    host.data.insert("cache_valid", vec![0xa5; valid as usize]);
    host
}

fn calls(host: &Fixture, cached: bool) -> LabResult<Vec<Launch>> {
    let e = entry(host.case);
    let old = host.launches(&e.source)?;
    if !cached {
        return Ok(old);
    }
    let mut new = e
        .launches(
            &old[0],
            host.view("decoded"),
            host.view("cache_valid"),
            None,
        )
        .map_err(|e| e.to_string())?;
    new.push(old[1].clone());
    Ok(new)
}

fn reset(session: &Session, device: &DeviceFixture<'_>, poison: u8) -> LabResult<()> {
    for (&name, data) in &device.host.data {
        let bytes = if matches!(name, "workspace" | "out" | "decoded" | "cache_valid") {
            vec![poison; data.len()]
        } else {
            data.clone()
        };
        device.buffers[name]
            .copy_from_bytes(&bytes, &session.stream)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn output(session: &Session, device: &DeviceFixture<'_>) -> LabResult<[Vec<u8>; 2]> {
    Ok([
        device.download(session, "workspace")?,
        device.download(session, "out")?,
    ])
}

fn inactive(session: &Session, device: &DeviceFixture<'_>, poison: u8) -> LabResult<()> {
    let case = device.host.case;
    for name in ["workspace", "out", "decoded", "cache_valid"] {
        let bytes = device.download(session, name)?;
        let live = bytes.len() / case.rows() * case.live();
        if bytes[live..].iter().any(|&b| b != poison) {
            return Err(format!("narrow cache overwrote inactive {name}"));
        }
    }
    Ok(())
}

fn description(case: Case, rank: u32) -> Value {
    json!({"rank":rank,"context":case.context,"rows":case.rows(),
        "verify":case.verify,"ratio":case.ratio,"live":case.live(),"holes":case.holes})
}

fn compare(session: &Session, modules: &Modules, case: Case, rank: u32) -> LabResult<bool> {
    let host = fixture(case, rank);
    let device = DeviceFixture::new(session, &host)?;
    let old = calls(&host, false)?;
    let new = calls(&host, true)?;
    let graphs = [&old, &new]
        .into_iter()
        .map(|ops| session.capture(|stream| device.run(session, modules, ops, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    for poison in [0xa5, 0x5a] {
        reset(session, &device, poison)?;
        device.run(session, modules, &old, &session.stream)?;
        let expected = output(session, &device)?;
        reset(session, &device, poison)?;
        device.run(session, modules, &new, &session.stream)?;
        if output(session, &device)? != expected {
            return Err(format!(
                "eager narrow attention differs: {case:?} rank {rank}"
            ));
        }
        inactive(session, &device, poison)?;
        for graph in &graphs {
            reset(session, &device, poison)?;
            graph.launch(&session.stream).map_err(|e| e.to_string())?;
            if output(session, &device)? != expected {
                return Err(format!(
                    "captured narrow attention differs: {case:?} rank {rank}"
                ));
            }
        }
        inactive(session, &device, poison)?;
    }
    // Both controls must fail after a fresh reset, including the graph that
    // leaves all outputs unwritten. Restrict to a nonempty compressed case.
    let negative = case.ratio == 1 && case.inactive == 0 && case.context <= 65;
    if negative {
        reset(session, &device, 0x42)?;
        device.run(session, modules, &old, &session.stream)?;
        let expected = output(session, &device)?;
        reset(session, &device, 0x42)?;
        let mut wrong = new.clone();
        wrong[0].symbol = entry(case).decode_symbol(true);
        device.run(session, modules, &wrong, &session.stream)?;
        if output(session, &device)? == expected {
            return Err("cached-validity negative survived".into());
        }
        let missing = session.capture(|stream| device.run(session, modules, &new[..1], stream))?;
        reset(session, &device, 0x42)?;
        missing.launch(&session.stream).map_err(|e| e.to_string())?;
        if output(session, &device)? == expected {
            return Err("unwritten-output graph negative survived".into());
        }
        println!("NARROW_CACHE_NEGATIVE_OK {}", description(case, rank));
    }
    println!("NARROW_CACHE_CASE_OK {}", description(case, rank));
    Ok(negative)
}

#[test]
fn cpu_narrow_cache_matrix_and_rank_seeds() {
    let cases = cases(false);
    assert_eq!(cases.len(), 117);
    assert_eq!(self::cases(true).len(), 39);
    let buckets: BTreeSet<_> = cases.iter().map(|c| (c.rows(), c.verify)).collect();
    assert_eq!(
        buckets,
        BTreeSet::from([(2, false), (4, false), (16, true)])
    );
    for case in shapes(65) {
        let mut queries = BTreeSet::new();
        for rank in 0..4 {
            let host = fixture(case, rank);
            queries.insert(kernel_build::digest(&host.data["q"]));
            let old = calls(&host, false).unwrap();
            let new = calls(&host, true).unwrap();
            let served = candidate::serving_entries(geometry(), 4)
                .unwrap()
                .into_iter()
                .find(|e| e.query == entry(case).source.query && !e.name.ends_with("_draft_block"))
                .unwrap();
            assert_eq!(old[0].symbol, served.symbol());
            assert_eq!(old[0].plan, served.plan(case.rows() as u32).unwrap());
            assert_eq!(old.len(), 2);
            assert_eq!(new.len(), 3);
            assert_eq!(old[1], new[2]);
            assert_eq!(old[0].plan.grid, new[1].plan.grid);
            assert_eq!(new[1].plan.grid[2], 8);
            assert!(!old[0].symbol.contains("cached"));
        }
        assert_eq!(queries.len(), 4);
    }
}

fn run(memcheck: bool) {
    let shard = shard();
    let scope = std::env::var("CHOPINFER_ATTENTION_SCOPE").expect("narrow pair scope");
    assert!(matches!(scope.as_str(), "decode" | "verify"));
    let matrix: Vec<_> = cases(memcheck)
        .into_iter()
        .filter(|case| case.verify == (scope == "verify"))
        .collect();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 2 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut ranks = BTreeSet::new();
        let mut count = 0;
        let mut negatives = 0;
        for rank in 0..4 {
            if rank as usize % 2 != shard {
                continue;
            }
            ranks.insert(rank);
            for &case in &matrix {
                negatives += usize::from(compare(session, &modules, case, rank)?);
                count += 1;
            }
        }
        assert_eq!(ranks.len(), 2);
        let per_context = if scope == "decode" { 48 } else { 30 };
        assert_eq!(count, per_context * if memcheck { 1 } else { 3 });
        assert_eq!(negatives, if scope == "decode" { 4 } else { 2 });
        println!(
            "NARROW_CACHE_COMPLETE {}",
            json!({"shard":shard,"ranks":ranks,"scope":scope,
            "cases":count,"memcheck":memcheck,"poisons":2,"eager":true,"graph":true,
            "negative_cases":negatives,"negative_controls_per_case":2,
            "inputs":"rank-distinct synthetic operands, not recorded serving tensors"})
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}

#[test]
#[ignore = "requires an allocated pair and frozen attention cubins"]
fn gpu_narrow_cache_exact() {
    run(false);
}

#[test]
#[ignore = "requires an allocated pair under compute-sanitizer memcheck"]
fn gpu_narrow_cache_memcheck() {
    run(true);
}

#[test]
#[ignore = "requires passed narrow exactness, memcheck and an allocated timing pair"]
fn gpu_narrow_cache_timing() {
    let shard = shard();
    let scope = std::env::var("CHOPINFER_ATTENTION_SCOPE").expect("narrow pair scope");
    assert!(matches!(scope.as_str(), "decode" | "verify"));
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 2 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut count = 0;
        for rank in 0..4 {
            if rank as usize % 2 != shard {
                continue;
            }
            for case in [1024, 16384]
                .into_iter()
                .flat_map(shapes)
                .filter(|c| c.verify == (scope == "verify"))
            {
                let host = fixture(case, rank);
                let device = DeviceFixture::new(session, &host)?;
                let old = calls(&host, false)?;
                let new = calls(&host, true)?;
                reset(session, &device, 0x3c)?;
                device.run(session, &modules, &old, &session.stream)?;
                let expected = output(session, &device)?;
                for repeats in [1, 32] {
                    let graphs = [&old, &new]
                        .into_iter()
                        .map(|ops| {
                            session.capture(|stream| {
                                for _ in 0..repeats {
                                    device.run(session, &modules, ops, stream)?;
                                }
                                Ok(())
                            })
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    for aa in [true, false] {
                        let label = format!(
                            "narrow-cache-{scope}-rank{rank}-ctx{}-rows{}-ratio{}-repeats{repeats}-aa{aa}",
                            case.context, case.rows(), case.ratio
                        );
                        let pair = kernel_harness::with_timing_label(&label, || {
                            kernel_harness::time_ab(
                                session,
                                2,
                                9,
                                [kernel_harness::eviction::EvictionMode::Write; 2],
                                |arm, stream| {
                                    graphs[if aa { 0 } else { arm }]
                                        .launch(stream)
                                        .map_err(|e| e.to_string())
                                },
                            )
                        })?;
                        println!(
                            "NARROW_CACHE_TIMING {}",
                            json!({"case":description(case,rank),"scope":scope,"aa":aa,
                                "replays":repeats,"rounds":9,"operation":label,
                                "old_us":pair.arms[0].median_ms*1000.0/repeats as f32,
                                "new_us":pair.arms[1].median_ms*1000.0/repeats as f32,
                                "old_min_us":pair.arms[0].min_ms*1000.0/repeats as f32,
                                "new_min_us":pair.arms[1].min_ms*1000.0/repeats as f32,
                                "paired_ratio":pair.paired_ratio,
                                "identical_operands":true,"current_symbol":old[0].symbol,
                                "candidate_symbol":new[1].symbol})
                        );
                        count += 1;
                    }
                    reset(session, &device, 0x3c)?;
                    graphs[1]
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                    if output(session, &device)? != expected {
                        return Err("timed narrow graph differs from the current path".into());
                    }
                }
            }
        }
        assert_eq!(count, if scope == "decode" { 96 } else { 48 });
        println!(
            "NARROW_CACHE_TIMING_COMPLETE {}",
            json!({"shard":shard,"scope":scope,"cases":count,"aa_identical_inputs":true})
        );
        let receipt = session.receipt(
            "attention_cache_narrow",
            BTreeMap::from([
                ("scope".into(), json!(scope)),
                ("shard".into(), json!(shard)),
                ("measurements".into(), json!(count)),
                ("execution_mode".into(), json!("graph")),
            ]),
        )?;
        println!("NARROW_CACHE_TIMING_RECEIPT {}", receipt.display());
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}
