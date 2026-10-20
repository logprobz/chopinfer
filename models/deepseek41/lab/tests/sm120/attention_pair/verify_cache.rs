//! Captured verify selections for the existing selected-key cache at exact widths.
use super::*;
use fixture::{HEADS, WIDTH, u32s};
use kernel_core::backend::Launch;
use kernels_sm120::latent::{sources::Schedule, sources_cached, sources_decode};

#[derive(Clone)]
struct Capture {
    id: String,
    case: Case,
    keys: Vec<u32>,
    digest: String,
}

fn captures() -> LabResult<Vec<Capture>> {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_VERIFY_SELECTIONS")
            .ok_or("set CHOPINFER_VERIFY_SELECTIONS to the frozen captured selections")?,
    );
    if !root.is_absolute() {
        return Err("capture root must be absolute".into());
    }
    let document: Value = serde_json::from_slice(
        &std::fs::read(root.join("manifest.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if document["schema"] != 1 {
        return Err("unknown verify capture schema".into());
    }
    let mut ids = BTreeSet::new();
    let mut result = Vec::new();
    for record in document["cases"].as_array().ok_or("capture cases absent")? {
        let number = |name: &str| -> LabResult<usize> {
            usize::try_from(
                record[name]
                    .as_u64()
                    .ok_or_else(|| format!("missing {name}"))?,
            )
            .map_err(|e| e.to_string())
        };
        let id = record["id"].as_str().ok_or("capture id absent")?;
        if id.is_empty()
            || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || !ids.insert(id.to_owned())
        {
            return Err("invalid or duplicate capture id".into());
        }
        let case = Case {
            context: number("context")?,
            sequences: number("sequences")?,
            verify: true,
            verify_width: number("width")?,
            ratio: number("ratio")?,
            holes: false,
            inactive: 0,
        };
        if case.rows() != number("rows")?
            || !matches!((case.sequences, case.verify_width), (1, 4 | 6 | 8) | (8, 4))
            || !matches!(case.ratio, 0..=2)
            || case.context == 0
            || case.context > 131072
        {
            return Err("unsupported captured verify shape".into());
        }
        let file = record["file"].as_str().ok_or("capture file absent")?;
        if !file
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.')
            || file.starts_with('.')
            || !file.ends_with(".bin")
        {
            return Err("invalid capture filename".into());
        }
        let path = root.join(file);
        if !std::fs::symlink_metadata(&path)
            .map_err(|e| e.to_string())?
            .is_file()
        {
            return Err("capture must be a regular file".into());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let digest = kernel_build::digest(&bytes);
        if record["sha256"] != digest
            || bytes.len() != case.rows() * if case.ratio == 0 { 0 } else { 512 * 4 }
        {
            return Err("capture size or digest mismatch".into());
        }
        let keys = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if keys.iter().any(|&k| {
            k != u32::MAX && k as usize >= (case.context + case.verify_width) / case.ratio.max(1)
        }) {
            return Err("captured selection exceeds its visible prefix".into());
        }
        result.push(Capture {
            id: id.into(),
            case,
            keys,
            digest,
        });
    }
    if result.is_empty() {
        return Err("empty verify capture set".into());
    }
    Ok(result)
}

fn entry(case: Case) -> sources_cached::Entry {
    let mut source = reference(case);
    source.schedule = Schedule::IndependentHeads;
    sources_cached::Entry {
        source,
        profile: false,
    }
}

fn fixture(capture: &Capture) -> Fixture {
    let e = entry(capture.case);
    let mut host = Fixture::with_compressed_keys(capture.case, &e.source, Some(&capture.keys));
    let (bytes, valid) = e.cache_bytes(capture.case.rows() as u32).unwrap();
    host.data.insert("decoded", vec![0xa5; bytes as usize]);
    host.data.insert("cache_valid", vec![0xa5; valid as usize]);
    host
}

fn calls(host: &Fixture, cached: bool) -> LabResult<Vec<Launch>> {
    let e = entry(host.case);
    let mut old = host.launches(&e.source)?;
    if cached {
        let mut new = e
            .launches(
                &old[0],
                host.view("decoded"),
                host.view("cache_valid"),
                None,
            )
            .map_err(|e| e.to_string())?;
        new.push(old.pop().unwrap());
        Ok(new)
    } else {
        if host.case.rows() == 32 {
            old[0] = sources_decode::Entry {
                source: e.source,
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
}

fn current_cached(case: Case) -> bool {
    matches!(case.rows(), 6 | 32)
}

fn reset(session: &Session, device: &DeviceFixture<'_>, poison: u8) -> LabResult<()> {
    for (&name, data) in &device.host.data {
        let scratch = matches!(name, "workspace" | "out" | "decoded" | "cache_valid");
        let bytes = if scratch {
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

fn output(session: &Session, device: &DeviceFixture<'_>, poison: u8) -> LabResult<Vec<Vec<u8>>> {
    let host = device.host;
    let mut output = Vec::new();
    for name in ["workspace", "out"] {
        let bytes = device.download(session, name)?;
        let end = bytes.len() / host.case.rows() * host.case.live();
        if bytes[end..].iter().any(|&b| b != poison) {
            return Err(format!("{name} wrote inactive rows"));
        }
        output.push(bytes[..end].to_vec());
    }
    Ok(output)
}

fn compare(session: &Session, modules: &Modules, host: &Fixture, graph: bool) -> LabResult<()> {
    let device = DeviceFixture::new(session, host)?;
    let old = calls(host, current_cached(host.case))?;
    let new = calls(host, true)?;
    for poison in [0xa5, 0x5a] {
        reset(session, &device, poison)?;
        device.run(session, modules, &old, &session.stream)?;
        let expected = output(session, &device, poison)?;
        reset(session, &device, poison)?;
        device.run(session, modules, &new, &session.stream)?;
        if output(session, &device, poison)? != expected {
            return Err("verify cache partial/output mismatch".into());
        }
        for name in ["decoded", "cache_valid"] {
            let bytes = device.download(session, name)?;
            let end = bytes.len() / host.case.rows() * host.case.live();
            if bytes[end..].iter().any(|&b| b != poison) {
                return Err("cache wrote inactive rows".into());
            }
        }
        if graph {
            let captured = session.capture(|stream| device.run(session, modules, &new, stream))?;
            reset(session, &device, poison)?;
            captured
                .launch(&session.stream)
                .map_err(|e| e.to_string())?;
            if output(session, &device, poison)? != expected {
                return Err("captured verify cache mismatch".into());
            }
        }
    }
    Ok(())
}

fn sliced(host: &Fixture, first: usize, rows: usize) -> Fixture {
    assert!(rows > 0 && first + rows <= host.case.rows() && rows % host.case.sequences == 0);
    let mut result = host.clone();
    result.case.verify_width = rows / host.case.sequences;
    result.case.inactive = rows - host.case.live().saturating_sub(first).min(rows);
    for name in [
        "q",
        "row_batch",
        "window_keys",
        "compressed_keys",
        "cos_sin",
        "workspace",
        "out",
        "decoded",
        "cache_valid",
    ] {
        let data = &host.data[name];
        let stride = data.len() / host.case.rows();
        result
            .data
            .insert(name, data[first * stride..(first + rows) * stride].to_vec());
    }
    result
        .data
        .insert("live_rows", u32s([result.case.live() as u32]));
    result
}

fn partition(session: &Session, modules: &Modules, host: &Fixture) -> LabResult<()> {
    let device = DeviceFixture::new(session, host)?;
    reset(session, &device, 0xa5)?;
    device.run(session, modules, &calls(host, true)?, &session.stream)?;
    let expected = output(session, &device, 0xa5)?;
    let mut joined = [Vec::new(), Vec::new()];
    let half = host.case.rows() / 2;
    for first in [0, half] {
        let part = sliced(host, first, half);
        let device = DeviceFixture::new(session, &part)?;
        reset(session, &device, 0xa5)?;
        device.run(session, modules, &calls(&part, true)?, &session.stream)?;
        for (target, bytes) in joined.iter_mut().zip(output(session, &device, 0xa5)?) {
            target.extend(bytes);
        }
    }
    if expected != joined {
        return Err("verify row partition changed partials or output".into());
    }
    Ok(())
}

fn controls(session: &Session, modules: &Modules, capture: &Capture, small: bool) -> LabResult<()> {
    let host = fixture(capture);
    compare(session, modules, &host, !small)?;
    partition(session, modules, &host)?;
    for mode in [
        "changed_query",
        "one_inactive",
        "all_inactive",
        "invalid_batch",
        "invalid_page",
        "holes",
    ] {
        let mut varied = host.clone();
        match mode {
            "changed_query" => {
                for b in varied.data.get_mut("q").unwrap().iter_mut().step_by(2) {
                    *b ^= 1;
                }
            }
            "one_inactive" | "all_inactive" => {
                varied.case.inactive = if mode == "all_inactive" {
                    host.case.rows()
                } else {
                    1
                };
                varied
                    .data
                    .insert("live_rows", u32s([varied.case.live() as u32]));
            }
            "invalid_batch" => {
                varied
                    .data
                    .insert("row_batch", u32s(vec![u32::MAX; host.case.rows()]));
            }
            "invalid_page" => {
                varied.data.get_mut("tables").unwrap().fill(0xff);
            }
            "holes" => {
                for name in ["window_keys", "compressed_keys"] {
                    for (i, key) in varied
                        .data
                        .get_mut(name)
                        .unwrap()
                        .chunks_exact_mut(4)
                        .enumerate()
                    {
                        if i % 128 < 64 {
                            key.fill(0xff);
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
        compare(session, modules, &varied, false)?;
    }
    if capture.case.ratio != 0 {
        let device = DeviceFixture::new(session, &host)?;
        reset(session, &device, 0xa5)?;
        device.run(session, modules, &calls(&host, true)?, &session.stream)?;
        let expected = output(session, &device, 0xa5)?;
        let mut wrong = calls(&host, true)?;
        wrong[0].symbol = entry(host.case).decode_symbol(true);
        reset(session, &device, 0xa5)?;
        device.run(session, modules, &wrong, &session.stream)?;
        if output(session, &device, 0xa5)? == expected {
            return Err("dropped-key negative control survived".into());
        }
    }
    println!(
        "VERIFY_CACHE_OK {}",
        json!({"id":capture.id,"sha256":capture.digest,"rows":capture.case.rows(),"ratio":capture.case.ratio,
        "poison_replays":14,"partition":true,"graph":!small,"negative":capture.case.ratio!=0})
    );
    Ok(())
}

fn timing(session: &Session, modules: &Modules, capture: &Capture) -> LabResult<()> {
    let host = fixture(capture);
    compare(session, modules, &host, true)?;
    let device = DeviceFixture::new(session, &host)?;
    let old = calls(&host, current_cached(host.case))?;
    let new = calls(&host, true)?;
    for mode in ["cold_sources", "cold_attention", "warm_attention"] {
        let repeats = if mode == "warm_attention" { 32 } else { 1 };
        let graphs = [&old, &new]
            .map(|calls| {
                session.capture(|stream| {
                    let end = calls.len() - usize::from(mode == "cold_sources");
                    for _ in 0..repeats {
                        device.run(session, modules, &calls[..end], stream)?;
                    }
                    Ok(())
                })
            })
            .into_iter()
            .collect::<LabResult<Vec<_>>>()?;
        let pair = kernel_harness::time_ab(
            session,
            2,
            9,
            [kernel_harness::eviction::EvictionMode::Write; 2],
            |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
        )?;
        println!(
            "VERIFY_CACHE_TIMING {}",
            json!({"id":capture.id,"sha256":capture.digest,"mode":mode,"rows":host.case.rows(),
            "ratio":host.case.ratio,"current_cached":current_cached(host.case),"old_us":pair.arms[0].median_ms*1000.0/repeats as f32,
            "new_us":pair.arms[1].median_ms*1000.0/repeats as f32,"paired_ratio":pair.paired_ratio})
        );
    }
    compare(session, modules, &host, true)?;
    Ok(())
}

fn run(timed: bool, small: bool) {
    let inputs = captures().unwrap();
    let shard = shard();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 1 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut count = 0;
        for input in &inputs {
            if timed {
                timing(session, &modules, input)?;
            } else {
                controls(session, &modules, input, small)?;
            }
            count += 1;
        }
        assert_eq!(count, inputs.len());
        println!(
            "VERIFY_CACHE_COMPLETE {}",
            json!({"shard":shard,"cases":count,"timed":timed,"memcheck":small})
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}

#[test]
#[ignore = "requires frozen captured verify selections and an allocated GPU pair"]
fn gpu_verify_cache_check() {
    run(false, false);
}
#[test]
#[ignore = "requires frozen captured verify selections and an allocated GPU pair"]
fn gpu_verify_cache_time() {
    run(true, false);
}
#[test]
#[ignore = "requires frozen captured verify selections and an allocated GPU pair"]
fn gpu_verify_cache_memcheck() {
    run(false, true);
}

#[test]
#[ignore = "requires CHOPINFER_VERIFY_SELECTIONS naming a frozen capture directory"]
fn frozen_verify_selections() {
    let inputs = captures().unwrap();
    for input in &inputs {
        let host = fixture(input);
        assert_eq!(
            host.data["compressed_keys"],
            input
                .keys
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>()
        );
        for cached in [false, true] {
            let calls = calls(&host, cached).unwrap();
            assert_eq!(
                calls.last().unwrap().plan.grid,
                [host.case.rows() as u32, HEADS as u32, 1]
            );
            let body = &calls[calls.len() - 2];
            assert_eq!(body.plan.grid[2], 8);
            assert_eq!(body.plan.block, [128, 1, 1]);
        }
    }
    println!("VERIFY_CAPTURE_CPU_OK cases={}", inputs.len());
}

#[test]
fn cpu_verify_cache_preserves_partitions_and_all_eight_splits() {
    for (sequences, width) in [(1, 4), (1, 6), (1, 8), (8, 4)] {
        for ratio in 0..=2 {
            let case = Case {
                context: 1024,
                sequences,
                verify: true,
                verify_width: width,
                ratio,
                holes: false,
                inactive: 0,
            };
            let keys = if ratio == 0 {
                vec![]
            } else {
                (0..case.rows() * 512)
                    .map(|i| {
                        if i % 19 == 0 {
                            u32::MAX
                        } else {
                            (i % 200) as u32
                        }
                    })
                    .collect()
            };
            let capture = Capture {
                id: "cpu".into(),
                case,
                keys,
                digest: String::new(),
            };
            let host = fixture(&capture);
            let new = calls(&host, true).unwrap();
            assert_eq!(new[1].plan.grid[2], 8);
            assert_eq!(new[2], calls(&host, false).unwrap()[1]);
            let half = host.case.rows() / 2;
            let parts = [sliced(&host, 0, half), sliced(&host, half, half)];
            for name in [
                "q",
                "row_batch",
                "window_keys",
                "compressed_keys",
                "cos_sin",
                "workspace",
                "out",
            ] {
                let joined = parts
                    .iter()
                    .flat_map(|p| p.data[name].clone())
                    .collect::<Vec<_>>();
                assert_eq!(joined, host.data[name]);
            }
            for part in parts {
                assert!(calls(&part, true).is_ok());
            }
            assert_eq!(host.data["out"].len(), case.rows() * HEADS * WIDTH * 2);
        }
    }
}
