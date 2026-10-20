//! Compare fixed cached-head groups on the exact served operands and split boundary.
use super::captured_replay::Captured;
use super::*;
use kernel_core::backend::Launch;
use kernels_sm120::latent::sources_cached::{Entry as Cache, GroupedEntry};

const GROUPS: [u32; 4] = [0, 2, 4, 8];
const MODES: [&str; 8] = [
    "original",
    "changed_query",
    "one_inactive",
    "all_inactive",
    "invalid_batch",
    "invalid_page",
    "holes",
    "all_holes",
];

fn cache(host: &Captured) -> Cache {
    Cache {
        source: host.source.clone(),
        profile: false,
    }
}
fn calls(host: &Captured, group: u32) -> LabResult<Vec<Launch>> {
    profile_calls(host, group, false)
}
fn profile_calls(host: &Captured, group: u32, profile: bool) -> LabResult<Vec<Launch>> {
    let mut native = host.clone();
    native.cached = false;
    let reference = native.calls()?;
    let mut entry = cache(host);
    entry.profile = profile;
    let stamps = if profile {
        Some(host.view("stamps")?)
    } else {
        None
    };
    let mut result = if group == 0 {
        entry.launches(
            &reference[0],
            host.view("decoded")?,
            host.view("cached_valid")?,
            stamps,
        )
    } else {
        GroupedEntry {
            cache: entry,
            heads: group,
        }
        .launches(
            &reference[0],
            host.view("decoded")?,
            host.view("cached_valid")?,
            stamps,
        )
    }
    .map_err(|e| e.to_string())?;
    result.push(reference[1].clone());
    Ok(result)
}
fn live(host: &Captured) -> usize {
    u32::from_le_bytes(host.data["live_rows"].as_slice().try_into().unwrap()) as usize
}
fn sliced(host: &Captured, first: usize, rows: usize) -> Captured {
    assert!(rows > 0 && first + rows <= host.rows as usize);
    let mut part = host.clone();
    for name in [
        "q",
        "row_batch",
        "window_keys",
        "compressed_keys",
        "cos_sin",
        "workspace",
        "out",
        "decoded",
        "cached_valid",
    ] {
        if let Some(bytes) = host.data.get(name) {
            assert_eq!(bytes.len() % host.rows as usize, 0);
            let stride = bytes.len() / host.rows as usize;
            part.data.insert(
                name.into(),
                bytes[first * stride..(first + rows) * stride].to_vec(),
            );
        }
    }
    for (dst, bytes) in part.expected.iter_mut().zip(&host.expected) {
        let stride = bytes.len() / host.rows as usize;
        *dst = bytes[first * stride..(first + rows) * stride].to_vec();
    }
    part.rows = rows as u32;
    part.data.insert(
        "live_rows".into(),
        (live(host).saturating_sub(first).min(rows) as u32)
            .to_le_bytes()
            .to_vec(),
    );
    part
}
fn varied(host: &Captured, mode: &str) -> Captured {
    let mut changed = host.clone();
    match mode {
        "original" => {}
        "changed_query" => {
            for b in changed.data.get_mut("q").unwrap().iter_mut().step_by(2) {
                *b ^= 1;
            }
        }
        "one_inactive" | "all_inactive" => {
            let rows = if mode == "all_inactive" {
                0
            } else {
                host.rows - 1
            };
            changed
                .data
                .insert("live_rows".into(), rows.to_le_bytes().to_vec());
        }
        "invalid_batch" => changed.data.get_mut("row_batch").unwrap().fill(0xff),
        "invalid_page" => {
            if let Some(bytes) = changed.data.get_mut("block_tables") {
                bytes.fill(0xff);
            } else {
                changed
                    .data
                    .get_mut("window_item_slots")
                    .unwrap()
                    .fill(0xff);
            }
        }
        "holes" | "all_holes" => {
            for name in ["window_keys", "compressed_keys"] {
                if let Some(bytes) = changed.data.get_mut(name) {
                    for (i, key) in bytes.chunks_exact_mut(4).enumerate() {
                        if mode == "all_holes" || i % 128 < 64 {
                            key.fill(0xff);
                        }
                    }
                }
            }
        }
        _ => panic!("unknown control"),
    }
    changed
}

struct Device<'a> {
    host: &'a Captured,
    buffers: BTreeMap<String, runtime_cuda::DeviceBuffer>,
}
impl<'a> Device<'a> {
    fn new(session: &Session, host: &'a Captured) -> LabResult<Self> {
        let buffers = host
            .data
            .iter()
            .map(|(name, bytes)| Ok((name.clone(), session.allocate(bytes.len().max(16))?)))
            .collect::<LabResult<_>>()?;
        Ok(Self { host, buffers })
    }
    fn reset(&self, session: &Session, poison: u8) -> LabResult<()> {
        for (name, data) in &self.host.data {
            let scratch = matches!(
                name.as_str(),
                "workspace" | "out" | "decoded" | "cached_valid"
            );
            let bytes = if scratch {
                vec![poison; data.len()]
            } else {
                data.clone()
            };
            self.buffers[name]
                .copy_from_bytes(&bytes, &session.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    fn run(
        &self,
        session: &Session,
        modules: &Modules,
        calls: &[Launch],
        stream: &runtime_cuda::Stream,
    ) -> LabResult<()> {
        let bindings = kernel_harness::launch::Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(name, data)| Ok((self.host.view(name)?.handle, data)))
                .collect::<LabResult<_>>()?,
        };
        for call in calls {
            // Validate once per dispatch without logging every timed graph node.
            launch::check(&call.symbol, call.plan, &modules.configured)?;
            kernel_harness::launch::launch(
                &session.context,
                modules.function(&call.symbol)?,
                stream,
                call.plan,
                &call.arguments,
                &bindings,
            )?;
        }
        Ok(())
    }
    fn read(&self, session: &Session, name: &str) -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; self.host.data[name].len()];
        self.buffers[name]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    fn output(&self, session: &Session) -> LabResult<[Vec<u8>; 2]> {
        Ok([self.read(session, "workspace")?, self.read(session, "out")?])
    }
    fn inactive(&self, session: &Session, poison: u8) -> LabResult<()> {
        for name in ["decoded", "cached_valid", "workspace", "out"] {
            let bytes = self.read(session, name)?;
            let end = bytes.len() / self.host.rows as usize * live(self.host);
            if bytes[end..].iter().any(|&b| b != poison) {
                return Err(format!("{name} wrote inactive rows"));
            }
        }
        Ok(())
    }
}
fn equal(actual: &[Vec<u8>; 2], expected: &[Vec<u8>; 2], label: &str) -> LabResult<()> {
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        if a != b {
            return Err(format!(
                "{label}: boundary {i} differs at {:?}",
                a.iter().zip(b).position(|(a, b)| a != b)
            ));
        }
    }
    Ok(())
}
fn controls(session: &Session, modules: &Modules, host: &Captured, graph: bool) -> LabResult<()> {
    for mode in MODES {
        let changed = varied(host, mode);
        let device = Device::new(session, &changed)?;
        for poison in [0xa5, 0x5a] {
            device.reset(session, poison)?;
            device.run(session, modules, &changed.calls()?, &session.stream)?;
            let expected = device.output(session)?;
            if mode == "original" {
                equal(&expected, &host.expected, &host.label)?;
            }
            for group in GROUPS {
                let candidate = calls(&changed, group)?;
                device.reset(session, poison)?;
                device.run(session, modules, &candidate, &session.stream)?;
                equal(
                    &device.output(session)?,
                    &expected,
                    &format!("{} {mode} h{group}", host.label),
                )?;
                device.inactive(session, poison)?;
                if graph && mode == "original" {
                    let executable = session
                        .capture(|stream| device.run(session, modules, &candidate, stream))?;
                    device.reset(session, poison)?;
                    executable
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                    equal(&device.output(session)?, &expected, "graph replay")?;
                }
            }
        }
    }
    for group in GROUPS {
        let mut joined = [Vec::new(), Vec::new()];
        let half = host.rows as usize / 2;
        for first in [0, half] {
            let part = sliced(host, first, half);
            let device = Device::new(session, &part)?;
            device.reset(session, 0xa5)?;
            device.run(session, modules, &calls(&part, group)?, &session.stream)?;
            for (dst, bytes) in joined.iter_mut().zip(device.output(session)?) {
                dst.extend(bytes);
            }
        }
        equal(&joined, &host.expected, "row partition")?;
    }
    let device = Device::new(session, host)?;
    let mut wrong = calls(host, 4)?;
    wrong[0].symbol = cache(host).decode_symbol(true);
    device.reset(session, 0xa5)?;
    device.run(session, modules, &wrong, &session.stream)?;
    if device.output(session)? == host.expected {
        return Err("dropped-key control survived".into());
    }
    println!(
        "VERIFY_GROUPED_OK {}",
        json!({"id":host.label,"rows":host.rows,"groups":GROUPS,
        "modes":MODES,"poison_replays":64,"partitions":4,"graph_replays":if graph {8} else {0},"negative":true,
        "partial_sha256":kernel_build::digest(&host.expected[0]),"output_sha256":kernel_build::digest(&host.expected[1])})
    );
    Ok(())
}
fn clocks(session: &Session, modules: &Modules, host: &Captured) -> LabResult<()> {
    let mut profiled = host.clone();
    let bytes = cache(host)
        .stamp_bytes(host.rows)
        .map_err(|e| e.to_string())? as usize;
    profiled.data.insert("stamps".into(), vec![0; bytes]);
    let device = Device::new(session, &profiled)?;
    for group in GROUPS {
        device.reset(session, 0xa5)?;
        device.run(
            session,
            modules,
            &profile_calls(&profiled, group, true)?,
            &session.stream,
        )?;
        equal(&device.output(session)?, &host.expected, "profile replay")?;
        let stamps = device
            .read(session, "stamps")?
            .chunks_exact(8)
            .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        let heads = if group == 0 {
            if host.rows <= 4 { 1 } else { 8 }
        } else {
            group
        } as usize;
        let units = (host.scalars["window_budget"].as_u64().unwrap()
            + host.scalars["compressed_budget"].as_u64().unwrap_or(0))
        .div_ceil(64) as usize;
        let mut cycles = [0_u64; 8];
        let mut ctas = 0;
        for row in 0..host.rows as usize {
            for head in 0..16 {
                for split in 0..8 {
                    let at = ((row * 16 + head) * 8 + split) * 8;
                    let record = &stamps[at..at + 8];
                    if head % heads != 0 {
                        if record.iter().any(|&x| x != 0) {
                            return Err("profile wrote an unowned head record".into());
                        }
                        continue;
                    }
                    if record[6] == 0
                        || record[..6].iter().sum::<u64>() > record[6]
                        || record[7] != (units * (split + 1) / 8 - units * split / 8) as u64
                    {
                        return Err("invalid cached-head clock record".into());
                    }
                    for (sum, value) in cycles.iter_mut().zip(record) {
                        *sum += value;
                    }
                    ctas += 1;
                }
            }
        }
        println!(
            "VERIFY_GROUPED_CLOCK {}",
            json!({"id":host.label,"rows":host.rows,"group":group,"ctas":ctas,"cycles":cycles,"bitwise":true})
        );
    }
    Ok(())
}
fn timing(session: &Session, modules: &Modules, host: &Captured) -> LabResult<()> {
    let device = Device::new(session, host)?;
    let old = host.calls()?;
    for group in GROUPS {
        let new = calls(host, group)?;
        for mode in ["cold_sources", "cold_attention", "warm_attention"] {
            let repeats = if mode == "warm_attention" { 32 } else { 1 };
            device.reset(session, 0xa5)?;
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
            // Check each complete arm after timing, including the merge omitted
            // by the source-only ablation. Never infer correctness from the last arm.
            for arm in [&old, &new] {
                device.reset(session, 0x5a)?;
                device.run(session, modules, arm, &session.stream)?;
                equal(
                    &device.output(session)?,
                    &host.expected,
                    "post-timing replay",
                )?;
            }
            println!(
                "VERIFY_GROUPED_TIMING {}",
                json!({"id":host.label,"rows":host.rows,"group":group,"mode":mode,
                "old_us":pair.arms[0].median_ms*1000.0/repeats as f32,
                "new_us":pair.arms[1].median_ms*1000.0/repeats as f32,"paired_ratio":pair.paired_ratio})
            );
        }
    }
    Ok(())
}
fn inputs() -> LabResult<Vec<Captured>> {
    let all = captured_replay::captures()?;
    let rows = std::env::var("CHOPINFER_VERIFY_ROWS")
        .ok()
        .map(|s| s.parse::<u32>().map_err(|e| e.to_string()))
        .transpose()?;
    if rows.is_some_and(|r| !matches!(r, 4 | 32)) {
        return Err("verify shard must be 4 or 32".into());
    }
    let all: Vec<_> = all
        .into_iter()
        .filter(|h| rows.is_none_or(|r| h.rows == r))
        .collect();
    if all.len() != if rows.is_some() { 12 } else { 24 } {
        return Err("verify capture shard lacks its 12 or 24 records".into());
    }
    Ok(all)
}
fn run(timed: bool, memcheck: bool) {
    let hosts = inputs().unwrap();
    let ran = kernel_harness::gpu_test_with(cubins().unwrap(), 1 << 30, |session| {
        let modules = Modules::load(session)?;
        for host in &hosts {
            if timed {
                timing(session, &modules, host)?;
            } else {
                controls(session, &modules, host, !memcheck)?;
                clocks(session, &modules, host)?;
            }
        }
        println!(
            "VERIFY_GROUPED_COMPLETE {}",
            json!({"cases":hosts.len(),"timed":timed,"memcheck":memcheck})
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some());
}
#[test]
#[ignore = "requires captured attention operands and an allocated GPU"]
fn gpu_verify_grouped_check() {
    run(false, false);
}
#[test]
#[ignore = "requires captured attention operands and an allocated GPU"]
fn gpu_verify_grouped_memcheck() {
    run(false, true);
}
#[test]
#[ignore = "requires captured attention operands and an allocated GPU"]
fn gpu_verify_grouped_time() {
    run(true, false);
}

#[test]
#[ignore = "requires CHOPINFER_ATTENTION_CAPTURE_ROOT naming a complete capture"]
fn frozen_verify_grouped() {
    let hosts = inputs().unwrap();
    for host in &hosts {
        for mode in MODES {
            let changed = varied(host, mode);
            let reference = changed.calls().unwrap();
            for group in GROUPS {
                let candidate = calls(&changed, group).unwrap();
                assert_eq!(candidate.last(), reference.last());
                assert_eq!(
                    candidate[1].plan.grid,
                    [
                        host.rows,
                        16 / if group == 0 {
                            if host.rows <= 4 { 1 } else { 8 }
                        } else {
                            group
                        },
                        8
                    ]
                );
            }
        }
        let half = host.rows as usize / 2;
        let parts = [sliced(host, 0, half), sliced(host, half, half)];
        for name in [
            "q",
            "row_batch",
            "window_keys",
            "compressed_keys",
            "cos_sin",
            "workspace",
            "out",
            "decoded",
            "cached_valid",
        ] {
            if host.data.contains_key(name) {
                assert_eq!(
                    [
                        parts[0].data[name].as_slice(),
                        parts[1].data[name].as_slice()
                    ]
                    .concat(),
                    host.data[name]
                );
            }
        }
        for part in parts {
            assert_eq!(part.scalars, host.scalars);
            for group in GROUPS {
                calls(&part, group).unwrap();
            }
        }
    }
    println!(
        "VERIFY_GROUPED_CPU_OK records={} modes=8 groups=4",
        hosts.len()
    );
}
