//! Eight-split decode comparisons and stage/byte-floor diagnostics.
use super::*;
use kernel_core::backend::Launch;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernels_sm120::latent::{sources::Schedule, sources_decode};

fn cases() -> Vec<Case> {
    [64, 128, 256, 512, 1024, 8192, 32768, 131072]
        .into_iter()
        .flat_map(|context| {
            [(1, false, 1), (8, false, 1), (8, true, 4)]
                .into_iter()
                .flat_map(move |(sequences, verify, verify_width)| {
                    [0, 1, 2].into_iter().map(move |ratio| Case {
                        context,
                        sequences,
                        verify,
                        verify_width,
                        ratio,
                        holes: false,
                        inactive: 0,
                    })
                })
        })
        .collect()
}
fn baseline(case: Case) -> Entry {
    let mut entry = reference(case);
    entry.schedule = Schedule::IndependentHeads;
    entry
}
fn variants(entry: &Entry) -> Vec<sources_decode::Entry> {
    decode_candidate::entries(geometry())
        .unwrap()
        .into_iter()
        .filter(|e| e.source.symbol() == entry.symbol())
        .collect()
}
fn fixture(case: Case, entry: &Entry) -> Fixture {
    let mut host = Fixture::new(case, entry);
    let bytes = case.rows() * fixture::HEADS * 8 * 8 * 8;
    host.data.insert("stamps", vec![0; bytes]);
    host
}
fn shape(case: Case) -> Value {
    json!({"context":case.context,"sequences":case.sequences,"rows":case.rows(),"verify":case.verify,"verify_width":case.verify_width,"ratio":case.ratio,"holes":case.holes,"inactive":case.inactive})
}

#[test]
fn cpu_decode_staging_preserves_eight_splits_and_exact_verify_width() {
    let limits = launch::limits().unwrap();
    let configured = launch::configure(&limits, |_, _| Ok(())).unwrap();
    let mut shapes = BTreeSet::new();
    let mut checked = 0;
    for case in cases() {
        shapes.insert((case.sequences, case.rows(), case.verify));
        let reference = baseline(case);
        let host = fixture(
            Case {
                context: 65,
                ..case
            },
            &reference,
        );
        let old = host.launches(&reference).unwrap();
        assert_eq!(old[0].plan.grid[2], 8);
        for entry in variants(&reference) {
            let call = entry.launch(&old[0], Some(host.view("stamps"))).unwrap();
            launch::check(&call.symbol, call.plan, &configured).unwrap();
            if call.plan.dynamic_shared_bytes > 48 * 1024 {
                assert!(launch::check(&call.symbol, call.plan, &BTreeMap::new()).is_err());
            }
            assert_eq!(call.plan.grid[0], case.rows() as u32);
            assert_eq!(call.plan.grid[2], 8);
            assert_eq!(
                call.plan.dynamic_shared_bytes,
                if entry.stage_keys == 32 { 42240 } else { 75008 }
            );
            assert_eq!(
                &call.arguments.0[..old[0].arguments.0.len()],
                old[0].arguments.0.as_slice()
            );
            if entry.profile {
                assert!(entry.launch(&old[0], None).is_err());
                assert!(entry.launch(&old[0], Some(host.view("workspace"))).is_err());
            }
            let mut wrong = old[0].clone();
            wrong.plan.grid[2] = 1;
            assert!(entry.launch(&wrong, Some(host.view("stamps"))).is_err());
            println!(
                "DECODE_PLAN {}",
                json!({"symbol":call.symbol,"rows":case.rows(),"grid":call.plan.grid,"block":call.plan.block,"dynamic_shared_bytes":call.plan.dynamic_shared_bytes})
            );
            checked += 1;
        }
        for call in &old {
            println!(
                "DECODE_PLAN {}",
                json!({"symbol":call.symbol,"rows":case.rows(),"grid":call.plan.grid,"block":call.plan.block,"dynamic_shared_bytes":call.plan.dynamic_shared_bytes})
            );
        }
    }
    assert_eq!(
        shapes,
        BTreeSet::from([(1, 1, false), (8, 8, false), (8, 32, true)])
    );
    assert_eq!(checked, 1296);
    let prefill = latent::entries(geometry(), 4)
        .unwrap()
        .into_iter()
        .find(|e| {
            matches!(
                e.query,
                LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill,
                    ..
                }
            )
        })
        .unwrap();
    assert_eq!(prefill.splits, 1);
    let wrong = sources_decode::Entry {
        source: prefill,
        stage_keys: 64,
        head_group: 0,
        packed: true,
        profile: false,
    };
    assert!(wrong.plan(1).is_err());
    println!("DECODE_CPU_OK shapes=72 bindings={checked} verify_rows=32");
}

fn compare(
    session: &Session,
    modules: &Modules,
    case: Case,
    timing: bool,
    negative: bool,
) -> LabResult<()> {
    let reference = baseline(case);
    let host = fixture(case, &reference);
    let device = DeviceFixture::new(session, &host)?;
    let old = host.launches(&reference)?;
    device.run(session, modules, &old, &session.stream)?;
    let expected = [
        device.download(session, "workspace")?,
        device.download(session, "out")?,
    ];
    for entry in variants(&reference).into_iter().filter(|e| !e.profile) {
        let source = entry.launch(&old[0], None).map_err(|e| e.to_string())?;
        let new = [source, old[1].clone()];
        for replay in 0..2 {
            device.reset(session)?;
            device.run(session, modules, &new, &session.stream)?;
            for (name, want) in ["workspace", "out"].into_iter().zip(&expected) {
                if device.download(session, name)? != *want {
                    return Err(format!(
                        "{} {case:?} {name} differs replay={replay}",
                        entry.symbol()
                    ));
                }
            }
        }
        let mut profiled = entry.clone();
        profiled.profile = true;
        device.reset(session)?;
        device.buffers["stamps"]
            .copy_from_bytes(&host.data["stamps"], &session.stream)
            .map_err(|e| e.to_string())?;
        let sample = [
            profiled
                .launch(&old[0], Some(host.view("stamps")))
                .map_err(|e| e.to_string())?,
            old[1].clone(),
        ];
        device.run(session, modules, &sample, &session.stream)?;
        for (name, want) in ["workspace", "out"].into_iter().zip(&expected) {
            if device.download(session, name)? != *want {
                return Err(format!("{} profile {name} differs", entry.symbol()));
            }
        }
        let stamps = device
            .download(session, "stamps")?
            .chunks_exact(8)
            .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut sums = [0_u64; 8];
        let mut count = 0;
        for row in 0..case.live() {
            for head in (0..fixture::HEADS).step_by(entry.group(case.rows() as u32) as usize) {
                for split in 0..8 {
                    let at = ((row * fixture::HEADS + head) * 8 + split) * 8;
                    let record = &stamps[at..at + 8];
                    let units = (128 + host.budget).div_ceil(64);
                    if record[6] == 0
                        || record[..6].iter().sum::<u64>() > record[6]
                        || record[7] != (units * (split + 1) / 8 - units * split / 8) as u64
                    {
                        return Err("invalid stage clock record".into());
                    }
                    for (sum, value) in sums.iter_mut().zip(record) {
                        *sum += value;
                    }
                    count += 1;
                }
            }
        }
        println!(
            "DECODE_CLOCK {}",
            json!({"case":shape(case),"symbol":entry.symbol(),"profile_symbol":profiled.symbol(),"ctas":count,"cycles":sums})
        );
        if timing {
            for mode in ["cold_source", "cold_attention", "warm_attention"] {
                let repeats = if mode == "warm_attention" { 32 } else { 1 };
                let graphs = [0, 1]
                    .into_iter()
                    .map(|arm| {
                        session.capture(|stream| {
                            let calls = if arm == 0 { old.as_slice() } else { &new };
                            let calls = if mode == "cold_source" {
                                &calls[..1]
                            } else {
                                calls
                            };
                            for _ in 0..repeats {
                                device.run(session, modules, calls, stream)?;
                            }
                            Ok(())
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                let pair = kernel_harness::time_ab(
                    session,
                    2,
                    9,
                    [kernel_harness::eviction::EvictionMode::Write; 2],
                    |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                )?;
                for (name, want) in ["workspace", "out"].into_iter().zip(&expected) {
                    if device.download(session, name)? != *want {
                        return Err(format!(
                            "{} {mode} graph replay {name} differs",
                            entry.symbol()
                        ));
                    }
                }
                let mut record = shape(case);
                record["symbol"] = json!(entry.symbol());
                record["mode"] = json!(mode);
                record["old_us"] = json!(pair.arms[0].median_ms * 1000.0 / repeats as f32);
                record["new_us"] = json!(pair.arms[1].median_ms * 1000.0 / repeats as f32);
                record["paired_ratio"] = json!(pair.paired_ratio);
                record["selected_kv_bytes"] = json!(host.unique_kv_bytes);
                record["row_kv_bytes"] = json!(host.row_kv_bytes);
                record["old_staged_payload_bytes"] =
                    json!(host.row_kv_bytes * (if case.rows() <= 4 { 16 } else { 2 }));
                record["new_staged_payload_bytes"] = json!(
                    host.row_kv_bytes
                        * fixture::HEADS.div_ceil(entry.group(case.rows() as u32) as usize)
                        * (if entry.stage_keys == 32 { 2 } else { 1 })
                );
                record["floor_us_at_1_tb_s"] = json!(host.unique_kv_bytes as f64 / 1_000_000.0);
                println!("DECODE_TIMING {record}");
            }
        }
    }
    if timing {
        let pair = kernel_harness::time_ab_prepared(
            session,
            2,
            9,
            [kernel_harness::eviction::EvictionMode::Write; 2],
            |_, stream| device.run(session, modules, &old[..1], stream),
            |_, stream| device.run(session, modules, &old[1..], stream),
        )?;
        println!(
            "DECODE_MERGE {}",
            json!({"case":shape(case),"median_us":pair.arms[0].median_ms*1000.0})
        );
    }
    if negative {
        device.reset(session)?;
        let mut wrong = old.clone();
        wrong[1].symbol = reference
            .sink_per_split_entry_spec()
            .map_err(|e| e.to_string())?
            .symbol;
        device.run(session, modules, &wrong, &session.stream)?;
        if device.download(session, "out")? == expected[1] {
            return Err("sink negative control was not rejected".into());
        }
        println!("DECODE_NEGATIVE_OK shard={}", shard());
    }
    println!("DECODE_CASE_OK {}", shape(case));
    Ok(())
}

#[test]
#[ignore = "requires the staged exclusive decode pair job"]
fn gpu_decode_staging() {
    let shard = shard();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 4 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut matrix = 0;
        let mut edges = 0;
        let mut negative = false;
        for (index, case) in cases().into_iter().enumerate() {
            if index % 2 != shard {
                continue;
            }
            let control = !negative && case.ratio != 0;
            compare(session, &modules, case, true, control)?;
            negative |= control;
            matrix += 1;
        }
        for (index, context) in [
            1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023,
        ]
        .into_iter()
        .enumerate()
        {
            if index % 2 != shard {
                continue;
            }
            for ratio in [0, 1, 2] {
                for inactive in [0, 1, 8] {
                    compare(
                        session,
                        &modules,
                        Case {
                            context,
                            sequences: 2,
                            verify: true,
                            verify_width: 4,
                            ratio,
                            holes: true,
                            inactive,
                        },
                        false,
                        false,
                    )?;
                    edges += 1;
                }
            }
        }
        assert!(negative && matrix == 36 && edges == 63);
        println!("DECODE_COMPLETE shard={shard} matrix={matrix} edges={edges}");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}

#[test]
#[ignore = "requires the staged exclusive decode pair job"]
fn gpu_decode_widening() {
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 1 << 20, |session| {
        let modules = Modules::load(session)?;
        let a = session.allocate(131072 * 4)?;
        let b = session.allocate(131072 * 4)?;
        let bindings = kernel_harness::launch::Bindings {
            buffers: BTreeMap::from([(1, &a), (2, &b)]),
        };
        for wrong in [false, true] {
            let call = Launch {
                symbol: sources_decode::widen_spec(wrong).symbol,
                rows: 1,
                plan: LaunchPlan {
                    grid: [256, 1, 1],
                    block: [256, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1, 1, 1],
                    pdl: false,
                },
                arguments: ArgumentList(vec![Argument::DeviceBuffer(1), Argument::DeviceBuffer(2)]),
                collective: false,
                persistent: false,
            };
            modules.launch(session, &session.stream, &call, &bindings)?;
            let mut old = vec![0; 131072 * 4];
            let mut new = old.clone();
            a.copy_to_bytes(&mut old, &session.stream)
                .map_err(|e| e.to_string())?;
            b.copy_to_bytes(&mut new, &session.stream)
                .map_err(|e| e.to_string())?;
            if (old == new) == wrong {
                return Err(format!(
                    "exhaustive widening comparison wrong={wrong} failed"
                ));
            }
        }
        println!("DECODE_WIDEN_OK shard={} pairs=131072 negative=1", shard());
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}
