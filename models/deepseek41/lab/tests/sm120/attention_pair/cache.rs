//! Selected-KV decode cache: compare every source partial and final output.
use super::*;
use kernel_core::backend::{Launch, View};
use kernels_sm120::latent::{sources::Schedule, sources_cached, sources_decode};

fn shapes(context: usize) -> Vec<Case> {
    [(1, false, 1), (8, false, 1), (1, true, 6), (8, true, 4)]
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
fn matrix() -> Vec<Case> {
    [1024, 8192, 32768, 131072]
        .into_iter()
        .flat_map(shapes)
        .collect()
}
fn boundaries() -> Vec<Case> {
    [
        1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023,
    ]
    .into_iter()
    .enumerate()
    .flat_map(|(index, context)| {
        shapes(context).into_iter().map(move |mut c| {
            c.holes = true;
            c.inactive = match index % 3 {
                0 => 0,
                1 => 1,
                _ => c.rows(),
            };
            c
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
fn fixture(case: Case, entry: &sources_cached::Entry) -> Fixture {
    let mut host = Fixture::new(case, &entry.source);
    let (bytes, valid) = entry.cache_bytes(case.rows() as u32).unwrap();
    host.data.insert("decoded", vec![0xa5; bytes as usize]);
    host.data.insert("cache_valid", vec![0xa5; valid as usize]);
    host.data.insert(
        "cache_stamps",
        vec![0; entry.stamp_bytes(case.rows() as u32).unwrap() as usize],
    );
    host
}
fn shape(case: Case) -> Value {
    json!({"context":case.context,"rows":case.rows(),"sequences":case.sequences,
        "verify":case.verify,"verify_width":case.verify_width,"ratio":case.ratio,
        "holes":case.holes,"inactive":case.inactive})
}
fn current(case: Case, host: &Fixture, entry: &sources_cached::Entry) -> LabResult<Vec<Launch>> {
    let mut old = host.launches(&entry.source)?;
    if (!case.verify && case.rows() == 8) || (case.verify && case.rows() == 32) {
        old[0] = sources_decode::Entry {
            source: entry.source.clone(),
            stage_keys: 64,
            head_group: 0,
            packed: false,
            profile: false,
        }
        .launch(&old[0], None)
        .map_err(|e| e.to_string())?;
    }
    Ok(old)
}
fn calls(host: &Fixture, entry: &sources_cached::Entry) -> LabResult<Vec<Launch>> {
    let baseline = host.launches(&entry.source)?;
    let mut calls = entry
        .launches(
            &baseline[0],
            host.view("decoded"),
            host.view("cache_valid"),
            Some(host.view("cache_stamps")),
        )
        .map_err(|e| e.to_string())?;
    calls.push(baseline[1].clone());
    Ok(calls)
}

#[test]
fn cpu_cache_preserves_splits_routes_and_disjoint_buffers() {
    let limits = launch::limits().unwrap();
    let configured = launch::configure(&limits, |_, _| Ok(())).unwrap();
    let mut buckets = BTreeSet::new();
    for case in shapes(65) {
        let entry = entry(case);
        let host = fixture(case, &entry);
        let old = host.launches(&entry.source).unwrap();
        let new = calls(&host, &entry).unwrap();
        buckets.insert((case.rows(), case.verify));
        assert_eq!(new.len(), 3);
        assert_eq!(new[2], old[1]);
        assert_eq!(new[1].plan.grid, old[0].plan.grid);
        assert_eq!(new[1].plan.grid[2], 8);
        assert_eq!(new[1].plan.dynamic_shared_bytes, 73984);
        assert_eq!(
            new[0].plan.grid,
            [
                if case.ratio == 0 { 32 } else { 160 },
                case.rows() as u32,
                1
            ]
        );
        for call in &new {
            launch::check(&call.symbol, call.plan, &configured).unwrap();
        }
        assert!(launch::check(&new[1].symbol, new[1].plan, &BTreeMap::new()).is_err());
        let mut short = host.view("decoded");
        short.bytes -= 1;
        assert!(
            entry
                .launches(&old[0], short, host.view("cache_valid"), None)
                .is_err()
        );
        let alias = View {
            handle: host.view("q").handle,
            ..host.view("decoded")
        };
        assert!(
            entry
                .launches(&old[0], alias, host.view("cache_valid"), None)
                .is_err()
        );
        assert!(
            entry
                .launches(&old[0], host.view("decoded"), host.view("decoded"), None)
                .is_err()
        );
        let mut profile = entry.clone();
        profile.profile = true;
        assert!(
            profile
                .launches(
                    &old[0],
                    host.view("decoded"),
                    host.view("cache_valid"),
                    None
                )
                .is_err()
        );
        let mut wrong = old[0].clone();
        wrong.plan.grid[2] = 16;
        assert!(
            entry
                .launches(&wrong, host.view("decoded"), host.view("cache_valid"), None)
                .is_err()
        );
        let serving = current(case, &host, &entry).unwrap();
        assert_eq!(
            serving[0].symbol.contains("stage64"),
            case.rows() == 8 || case.rows() == 32
        );
        assert_eq!(serving[1], old[1]);
        println!(
            "CACHE_PLAN {}",
            json!({"case":shape(case),"source_ctas":new[1].plan.grid.iter().product::<u32>(),
            "predecode_ctas":new[0].plan.grid.iter().product::<u32>(),"cache_bytes":entry.cache_bytes(case.rows() as u32).unwrap()})
        );
    }
    assert_eq!(
        buckets,
        BTreeSet::from([(1, false), (8, false), (6, true), (32, true)])
    );
    assert_eq!(matrix().len(), 48);
    assert_eq!(boundaries().len(), 168);
    println!("CACHE_CPU_OK matrix=48 boundaries=168 buckets=4");
}

fn reset(session: &Session, device: &DeviceFixture<'_>, poison: u8) -> LabResult<()> {
    device.reset(session)?;
    for name in ["decoded", "cache_valid"] {
        device.buffers[name]
            .copy_from_bytes(&vec![poison; device.host.data[name].len()], &session.stream)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}
fn compare(
    session: &Session,
    modules: &Modules,
    case: Case,
    timing: bool,
    negative: bool,
) -> LabResult<()> {
    let entry = entry(case);
    let host = fixture(case, &entry);
    let device = DeviceFixture::new(session, &host)?;
    let old = current(case, &host, &entry)?;
    device.reset(session)?;
    device.run(session, modules, &old, &session.stream)?;
    let expected = [
        device.download(session, "workspace")?,
        device.download(session, "out")?,
    ];
    let new = calls(&host, &entry)?;
    let check = || -> LabResult<()> {
        for (name, want) in ["workspace", "out"].into_iter().zip(&expected) {
            if device.download(session, name)? != *want {
                return Err(format!("cache {case:?} {name} differs"));
            }
        }
        Ok(())
    };
    for poison in [0xa5, 0x5a] {
        reset(session, &device, poison)?;
        device.run(session, modules, &new, &session.stream)?;
        check()?;
        for name in ["decoded", "cache_valid"] {
            let bytes = device.download(session, name)?;
            let live_bytes = bytes.len() / case.rows() * case.live();
            if bytes[live_bytes..].iter().any(|&b| b != poison) {
                return Err("cache wrote inactive rows".into());
            }
        }
    }
    let mut profiled = entry.clone();
    profiled.profile = true;
    reset(session, &device, 0x3c)?;
    device.run(session, modules, &calls(&host, &profiled)?, &session.stream)?;
    check()?;
    let stamps = device
        .download(session, "cache_stamps")?
        .chunks_exact(8)
        .map(|x| u64::from_ne_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut cycles = [0_u64; 8];
    let mut ctas = 0;
    for row in 0..case.live() {
        for head in (0..16).step_by(if case.rows() <= 4 { 1 } else { 8 }) {
            for split in 0..8 {
                let at = ((row * 16 + head) * 8 + split) * 8;
                let record = &stamps[at..at + 8];
                let units = if case.ratio == 0 { 2 } else { 10 };
                if record[6] == 0
                    || record[..6].iter().sum::<u64>() > record[6]
                    || record[7] != (units * (split + 1) / 8 - units * split / 8) as u64
                {
                    return Err("invalid cached source clock record".into());
                }
                for (sum, value) in cycles.iter_mut().zip(record) {
                    *sum += value;
                }
                ctas += 1;
            }
        }
    }
    println!(
        "CACHE_CLOCK {}",
        json!({"case":shape(case),"ctas":ctas,"cycles":cycles})
    );
    if timing {
        for mode in ["cold_sources", "cold_attention", "warm_attention"] {
            let repeats = if mode == "warm_attention" { 32 } else { 1 };
            let graphs = [0, 1]
                .into_iter()
                .map(|arm| {
                    session.capture(|stream| {
                        let operations = if arm == 0 { &old } else { &new };
                        let n = operations.len() - usize::from(mode == "cold_sources");
                        for _ in 0..repeats {
                            device.run(session, modules, &operations[..n], stream)?;
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
            // Check the captured candidate explicitly, independent of which
            // arm the alternating timer happened to execute last.
            reset(session, &device, 0x69)?;
            graphs[1]
                .launch(&session.stream)
                .map_err(|e| e.to_string())?;
            if mode == "cold_sources" {
                device.run(session, modules, &old[1..], &session.stream)?;
            }
            check()?;
            println!("CACHE_GRAPH_OK {}", json!({"case":shape(case),"mode":mode}));
            println!(
                "CACHE_TIMING {}",
                json!({"case":shape(case),"mode":mode,"old_symbol":old[0].symbol,
                "old_us":pair.arms[0].median_ms*1000.0/repeats as f32,
                "new_us":pair.arms[1].median_ms*1000.0/repeats as f32,"paired_ratio":pair.paired_ratio,
                "cache_bytes":entry.cache_bytes(case.rows() as u32).map_err(|e|e.to_string())?,
                "selected_kv_bytes":host.unique_kv_bytes,"row_kv_bytes":host.row_kv_bytes})
            );
        }
        let pair = kernel_harness::time_ab(
            session,
            2,
            9,
            [kernel_harness::eviction::EvictionMode::Write; 2],
            |_, stream| device.run(session, modules, &new[..1], stream),
        )?;
        println!(
            "CACHE_PREDECODE_TIMING {}",
            json!({"case":shape(case),"us":pair.arms[0].median_ms*1000.0})
        );
    }
    if negative {
        reset(session, &device, 0x42)?;
        let mut wrong = new.clone();
        wrong[0].symbol = entry.decode_symbol(true);
        device.run(session, modules, &wrong, &session.stream)?;
        if device.download(session, "out")? == expected[1] {
            return Err("cached validity negative control survived".into());
        }
        println!("CACHE_NEGATIVE_OK {}", shape(case));
    }
    println!("CACHE_CASE_OK {}", shape(case));
    Ok(())
}

#[test]
#[ignore = "requires the staged exclusive selected KV cache pair"]
fn gpu_selected_cache() {
    let shard = shard();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 4 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut controls = BTreeSet::new();
        let mut matrix_count = 0;
        let mut edges = 0;
        for (index, case) in matrix().into_iter().enumerate() {
            if index % 2 != shard {
                continue;
            }
            let key = (case.rows(), case.verify);
            let negative = case.ratio != 0 && !controls.contains(&key);
            compare(session, &modules, case, true, negative)?;
            if negative {
                controls.insert(key);
            }
            matrix_count += 1;
        }
        for (index, case) in boundaries().into_iter().enumerate() {
            if index % 2 != shard {
                continue;
            }
            compare(session, &modules, case, false, false)?;
            edges += 1;
        }
        assert_eq!((matrix_count, edges, controls.len()), (24, 84, 4));
        println!("CACHE_COMPLETE shard={shard} matrix=24 edges=84 negative=4");
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}
