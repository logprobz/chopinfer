//! The cached F32 query norm entries against the served query norm/quantize
//! entry: bitwise normed rows, codes, scales and status over special
//! patterns, live counts, graph replays and output-flip negatives, then
//! paired graph timing.
#[allow(dead_code, reason = "the build example uses the family and specs")]
mod catalog;
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const ROWS: [u32; 4] = [1, 6, 8, 32];
const PATTERNS: [&str; 16] = [
    "finite",
    "spread",
    "wide",
    "zero",
    "denormal",
    "maximum",
    "nan_first",
    "nan_last",
    "inf_first",
    "inf_last",
    "boundaries",
    "sticky",
    "nan_block",
    "inf_block",
    "weight_nan",
    "weight_inf",
];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
const OFFSET: usize = 3;
/// Launches per chained timing graph: one cold launch is below the event
/// timer's resolution, so each arm is also timed as a PDL chain.
const CHAIN: usize = 16;

fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_NORM_QUERY_CACHED")
            .expect("frozen query norm artifacts required"),
    )
}
/// Module name to its bytes and the symbols the matrix uses.
type Modules = BTreeMap<String, (Vec<u8>, Vec<String>)>;
/// The served norm module, the lab cached module and the probe.
fn modules() -> LabResult<Modules> {
    let mut modules = Modules::new();
    let served = catalog::served();
    let (name, bytes) =
        fold_cubin_for_symbol(&served.symbol).ok_or("served entry must be embedded")?;
    modules.insert(name.into(), (bytes.to_vec(), vec![served.symbol.clone()]));
    let cached = std::fs::read(artifacts().join(format!("{}.cubin", catalog::MODULE)))
        .map_err(|e| e.to_string())?;
    let lock: serde_json::Value = serde_json::from_str(include_str!(
        "../../../kernels.lock.d/norm_query_cached.lock"
    ))
    .map_err(|e| e.to_string())?;
    if lock[0]["name"] != catalog::MODULE
        || lock[0]["cubin_sha256"] != kernel_build::digest(&cached)
    {
        return Err("cached module differs from the lab lock".into());
    }
    modules.insert(
        catalog::MODULE.into(),
        (cached, catalog::cached().symbols().to_vec()),
    );
    let probe = std::fs::read(artifacts().join("probe_sm120.cubin")).map_err(|e| e.to_string())?;
    let kernel_lab: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../../crates/kernel-lab/kernels.lock"
    ))
    .map_err(|e| e.to_string())?;
    if !kernel_lab
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["name"] == "probe_sm120" && c["cubin_sha256"] == kernel_build::digest(&probe))
    {
        return Err("timing probe differs from the kernel-lab lock".into());
    }
    modules.insert(
        "probe_sm120".into(),
        (probe, PROBE_SYMBOLS.iter().map(|s| (*s).into()).collect()),
    );
    Ok(modules)
}
fn coverage() -> serde_json::Value {
    let arms = catalog::WARPS.len();
    json!({"served": catalog::served().symbol, "cached": catalog::cached().symbols(),
        "warps": catalog::WARPS, "width": catalog::served().query.norm.width,
        "rows": ROWS, "patterns": PATTERNS, "bitwise_cases": arms * ROWS.len() * PATTERNS.len(),
        "timing_cases": arms * ROWS.len(), "negative_controls": arms * ROWS.len() * 4,
        "memcheck_cases": arms * 2})
}

#[test]
fn cached_query_entries_keep_the_served_abi_and_plans() {
    let served = catalog::served();
    let cached = catalog::cached();
    assert_eq!(served.query.norm.width, 1280);
    assert_eq!(served.query.norm.input, kernel_core::contract::DType::F32);
    assert!(fold_cubin_for_symbol(&served.symbol).is_some());
    let spec = catalog::specs().remove(0);
    assert_eq!(spec.entries.len(), catalog::WARPS.len());
    for ((entry, symbol), warps) in spec
        .entries
        .iter()
        .zip(cached.symbols())
        .zip(catalog::WARPS)
    {
        assert_eq!(entry.symbol, symbol);
        assert_eq!(entry.params, served.spec().params);
        for rows in ROWS {
            let old = served.plan(rows).unwrap();
            let new = cached.plan(rows, rows, warps).unwrap();
            assert_eq!(new.grid, [1, rows, 1]);
            assert_eq!(new.block, [warps * 32, 1, 1]);
            assert_eq!(
                (new.pdl, new.workspace_bytes),
                (old.pdl, old.workspace_bytes)
            );
        }
    }
    let c = coverage();
    assert_eq!(c["bitwise_cases"], 3 * 4 * 16);
    assert_eq!(c["negative_controls"], 3 * 4 * 4);
}

/// Explicit artifact staging and checking, never part of an ordinary run.
#[test]
#[ignore = "requires an external frozen artifact directory named by CHOPINFER_NORM_QUERY_CACHED"]
fn frozen_modules_are_complete() {
    let dir = artifacts();
    let modules = modules().unwrap();
    if std::env::var_os("CHOPINFER_NORM_QUERY_EXPORT").is_some() {
        let mut manifest = Vec::new();
        for (name, (bytes, symbols)) in &modules {
            std::fs::write(dir.join(format!("{name}.cubin")), bytes).unwrap();
            manifest
                .push(json!({"name":name,"sha256":kernel_build::digest(bytes),"symbols":symbols}));
        }
        std::fs::write(
            dir.join("modules.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("coverage.json"),
            serde_json::to_vec_pretty(&coverage()).unwrap(),
        )
        .unwrap();
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("modules.json")).unwrap()).unwrap();
    assert_eq!(manifest.as_array().unwrap().len(), modules.len());
    for (name, (bytes, symbols)) in modules {
        let record = manifest
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == name)
            .unwrap();
        assert_eq!(record["sha256"], kernel_build::digest(&bytes));
        assert_eq!(record["symbols"], json!(symbols));
        for symbol in symbols {
            assert!(
                bytes
                    .windows(symbol.len() + 1)
                    .any(|s| s[..symbol.len()] == *symbol.as_bytes() && s[symbol.len()] == 0)
            );
        }
    }
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(dir.join("coverage.json")).unwrap()
        )
        .unwrap(),
        coverage()
    );
    eprintln!(
        "NORM_QUERY_CACHED_MODULES {} PASS",
        manifest.as_array().unwrap().len()
    );
}

fn upload(buffer: &DeviceBuffer, bytes: &[u8], stream: &Stream) -> LabResult<()> {
    buffer
        .copy_from_bytes(bytes, stream)
        .map_err(|e| e.to_string())
}
fn download(buffer: &DeviceBuffer, len: usize, stream: &Stream) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; len];
    buffer
        .copy_to_bytes(&mut bytes, stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}
const ROLES: [&str; 4] = ["normed", "codes", "scales", "status"];
struct Output {
    normed: DeviceBuffer,
    codes: DeviceBuffer,
    scales: DeviceBuffer,
    status: DeviceBuffer,
}
impl Output {
    fn new(s: &Session, width: usize, rows: u32) -> LabResult<Self> {
        Ok(Self {
            normed: s.allocate(width * rows as usize * 4)?,
            codes: s.allocate(width * rows as usize)?,
            scales: s.allocate(width * rows as usize / 32)?,
            status: s.allocate((rows as usize + OFFSET + 2) * 4)?,
        })
    }
    fn buffers(&self) -> [&DeviceBuffer; 4] {
        [&self.normed, &self.codes, &self.scales, &self.status]
    }
    fn lens(width: usize, rows: u32) -> [usize; 4] {
        let values = width * rows as usize;
        [
            values * 4,
            values,
            values / 32,
            (rows as usize + OFFSET + 2) * 4,
        ]
    }
    fn poison(&self, s: &Session, width: usize, rows: u32, sticky: bool) -> LabResult<()> {
        let lens = Self::lens(width, rows);
        upload(&self.normed, &vec![0xa5; lens[0]], &s.stream)?;
        upload(&self.codes, &vec![0xa5; lens[1]], &s.stream)?;
        upload(&self.scales, &vec![0x5a; lens[2]], &s.stream)?;
        let mut flags = vec![0x1357_2468u32; rows as usize + OFFSET + 2];
        flags[OFFSET..OFFSET + rows as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.status,
            &flags
                .iter()
                .flat_map(|s| s.to_ne_bytes())
                .collect::<Vec<_>>(),
            &s.stream,
        )
    }
    fn read(&self, s: &Session, width: usize, rows: u32) -> LabResult<[Vec<u8>; 4]> {
        let lens = Self::lens(width, rows);
        Ok([
            download(&self.normed, lens[0], &s.stream)?,
            download(&self.codes, lens[1], &s.stream)?,
            download(&self.scales, lens[2], &s.stream)?,
            download(&self.status, lens[3], &s.stream)?,
        ])
    }
}
struct Inputs {
    input: DeviceBuffer,
    weight: DeviceBuffer,
    live: DeviceBuffer,
}
/// Arm 0 is the served entry, arm 1 + i the cached entry with WARPS[i].
fn launch_arm(
    arm: usize,
    function: &Function,
    rows: u32,
    i: &Inputs,
    out: &Output,
    stream: &Stream,
) -> LabResult<()> {
    let served = catalog::served();
    let plan = if arm == 0 {
        served.plan(rows).unwrap()
    } else {
        catalog::cached()
            .plan(rows, rows, catalog::WARPS[arm - 1])
            .unwrap()
    };
    let args = [
        A::Buffer(&i.input),
        A::Buffer(&i.weight),
        A::Buffer(&out.normed),
        A::NullBuffer,
        A::Buffer(&out.codes),
        A::Buffer(&out.scales),
        A::Buffer(&i.live),
        A::Buffer(&out.status),
        A::U64(OFFSET as u64),
        A::U32(served.query.norm.epsilon_bits),
    ];
    function
        .launch(stream, launch::config(plan)?, &args)
        .map_err(|e| e.to_string())
}
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
/// F32 rows; "finite" is activation-like, "spread" spans many binades.
fn rows_of(width: usize, rows: u32, pattern: &str, mut state: u64) -> Vec<u8> {
    (0..width * rows as usize)
        .flat_map(|i| {
            let state = next(&mut state);
            let sign = (state as u32) & 0x8000_0000;
            let mantissa = state as u32 & 0x7f_ffff;
            let at = i % width;
            let first = i / width == 0;
            let bits = match pattern {
                "spread" => sign | (((state >> 32) as u32 % 100 + 77) << 23) | mantissa,
                "wide" => sign | (((state >> 32) as u32 % 254 + 1) << 23) | mantissa,
                "zero" => sign,
                "denormal" => sign | mantissa.max(1),
                "maximum" => sign | 0x7f7f_ffff,
                "nan_block" if first && at >= width - 32 => sign | 0x7fc1_0000,
                "inf_block" if first && at >= width - 32 => sign | 0x7f80_0000,
                "nan_first" if first && at == 0 => 0x7fc1_0000,
                "nan_last" if first && at == width - 1 => 0xff81_0000,
                "inf_first" if first && at == 0 => 0x7f80_0000,
                "inf_last" if first && at == width - 1 => 0xff80_0000,
                "boundaries" => {
                    let e = ((i / 32) % 40) as i32 - 20;
                    let v = 2.0f32.powi(e).to_bits();
                    sign | match i % 3 {
                        0 => v - 1,
                        1 => v,
                        _ => v + 1,
                    }
                }
                _ => sign | (((state >> 32) as u32 % 12 + 117) << 23) | mantissa,
            };
            bits.to_ne_bytes()
        })
        .collect()
}
/// BF16 weights of either sign in [0.25, 4), with one non-finite weight
/// for the weight patterns.
fn weight_of(width: usize, pattern: &str, mut state: u64) -> Vec<u8> {
    (0..width)
        .flat_map(|i| {
            let state = next(&mut state);
            let bits = match pattern {
                "weight_nan" if i == width - 1 => 0x7fc1,
                "weight_inf" if i == 0 => 0xff80,
                _ => {
                    ((state >> 16) as u16 & 0x8000)
                        | ((((state >> 32) % 4) as u16 + 125) << 7)
                        | (state as u16 & 0x7f)
                }
            };
            bits.to_ne_bytes()
        })
        .collect()
}
fn compare(expected: &[Vec<u8>; 4], actual: &[Vec<u8>; 4]) -> LabResult<()> {
    for (role, (a, b)) in ROLES.into_iter().zip(expected.iter().zip(actual)) {
        if a != b {
            return Err(format!(
                "{role} mismatch at {:?}",
                a.iter().zip(b).position(|(a, b)| a != b)
            ));
        }
    }
    Ok(())
}

struct Loaded {
    functions: Vec<Function>,
    _modules: Vec<Module>,
}
fn load(s: &Session) -> LabResult<Loaded> {
    let modules = modules()?;
    let mut loaded = BTreeMap::new();
    for (name, (bytes, _)) in &modules {
        if name != "probe_sm120" {
            loaded.insert(name.clone(), s.load_cubin(bytes)?);
        }
    }
    let module_of = |symbol: &str| {
        modules
            .iter()
            .find(|(_, (_, symbols))| symbols.iter().any(|s| s == symbol))
            .map(|(name, _)| &loaded[name])
            .ok_or_else(|| format!("unstaged symbol {symbol}"))
    };
    let symbols = std::iter::once(catalog::served().symbol).chain(catalog::cached().symbols());
    let functions = symbols
        .map(|symbol| {
            module_of(&symbol)?
                .function(&symbol)
                .map_err(|e| e.to_string())
        })
        .collect::<LabResult<Vec<_>>>()?;
    Ok(Loaded {
        functions,
        _modules: loaded.into_values().collect(),
    })
}
fn probes() -> Cubins {
    let probe = Box::leak(
        std::fs::read(artifacts().join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    Cubins::new(
        Box::leak(Box::new([("probe_sm120", &probe[..])])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    )
}
/// Every cached arm against the served entry at every row count and
/// pattern, eager and two graph replays, at full, zero, partial and excess
/// live counts; `memcheck` runs one finite and one non-finite case per arm,
/// eagerly, without timing.
fn matrix(s: &Session, memcheck: bool) -> LabResult<serde_json::Value> {
    let k = load(s)?;
    let arms = k.functions.len();
    let cached = catalog::cached().symbols();
    let width = catalog::served().query.norm.width as usize;
    let (mut bitwise, mut negative) = (0, 0);
    let mut timings = Vec::new();
    for rows in if memcheck { vec![8] } else { ROWS.to_vec() } {
        let inputs = Inputs {
            input: s.allocate(width * rows as usize * 4)?,
            weight: s.allocate(width * 2)?,
            live: s.allocate(4)?,
        };
        let outputs = (0..arms)
            .map(|_| Output::new(s, width, rows))
            .collect::<LabResult<Vec<_>>>()?;
        let graphs = (0..arms)
            .map(|arm| {
                s.capture(|stream| {
                    launch_arm(arm, &k.functions[arm], rows, &inputs, &outputs[arm], stream)
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let chains = (0..arms)
            .map(|arm| {
                s.capture(|stream| {
                    (0..CHAIN).try_for_each(|_| {
                        launch_arm(arm, &k.functions[arm], rows, &inputs, &outputs[arm], stream)
                    })
                })
            })
            .collect::<LabResult<Vec<_>>>()?;
        let patterns: &[&str] = if memcheck {
            &["finite", "nan_last"]
        } else {
            &PATTERNS
        };
        for &pattern in patterns {
            upload(
                &inputs.input,
                &rows_of(width, rows, pattern, 0x91a2_03c4),
                &s.stream,
            )?;
            upload(
                &inputs.weight,
                &weight_of(width, pattern, 0x5eed_0f0e),
                &s.stream,
            )?;
            let lives: &[u32] = if memcheck {
                &[8]
            } else {
                &[rows, 0, rows.saturating_sub(1), rows + 7, rows]
            };
            for &live_rows in lives {
                let active = live_rows.min(rows) as usize;
                upload(&inputs.live, &live_rows.to_ne_bytes(), &s.stream)?;
                let mut eager: Option<Vec<[Vec<u8>; 4]>> = None;
                for graph in if memcheck {
                    vec![false]
                } else {
                    vec![false, true, true]
                } {
                    for output in &outputs {
                        output.poison(s, width, rows, pattern == "sticky")?;
                    }
                    for arm in 0..arms {
                        if graph {
                            graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                        } else {
                            launch_arm(
                                arm,
                                &k.functions[arm],
                                rows,
                                &inputs,
                                &outputs[arm],
                                &s.stream,
                            )?;
                        }
                    }
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let read = outputs
                        .iter()
                        .map(|o| o.read(s, width, rows))
                        .collect::<LabResult<Vec<_>>>()?;
                    for arm in 1..arms {
                        compare(&read[0], &read[arm]).map_err(|e| {
                            format!(
                                "{} rows={rows} live={live_rows} {pattern} graph={graph}: {e}",
                                cached[arm - 1]
                            )
                        })?;
                    }
                    if let Some(ref previous) = eager {
                        compare(&previous[0], &read[0])?;
                    } else {
                        eager = Some(read.clone());
                    }
                    let new = &read[arms - 1];
                    if !new[0][active * width * 4..].iter().all(|b| *b == 0xa5)
                        || !new[1][active * width..].iter().all(|b| *b == 0xa5)
                        || !new[2][active * width / 32..].iter().all(|b| *b == 0x5a)
                        || (0..OFFSET)
                            .chain(OFFSET + rows as usize..OFFSET + rows as usize + 2)
                            .any(|i| new[3][i * 4..i * 4 + 4] != 0x1357_2468u32.to_ne_bytes())
                    {
                        return Err(format!(
                            "rows={rows} live={live_rows} {pattern}: guard or inactive output changed"
                        ));
                    }
                    if pattern == "finite"
                        && live_rows > 0
                        && (new[1][..width].iter().all(|b| *b == 0)
                            || new[3][OFFSET * 4..OFFSET * 4 + 4] != [0; 4])
                    {
                        return Err("finite row produced no codes or a status".into());
                    }
                    if pattern == "nan_last"
                        && live_rows > 0
                        && new[3][OFFSET * 4..OFFSET * 4 + 4] == [0; 4]
                    {
                        return Err("non-finite row produced no status".into());
                    }
                }
            }
            for symbol in &cached {
                bitwise += 1;
                eprintln!(
                    "NORM_QUERY_CACHED_BITWISE {}",
                    json!({"symbol":symbol,"rows":rows,"pattern":pattern})
                );
            }
        }
        if memcheck {
            continue;
        }
        upload(
            &inputs.input,
            &rows_of(width, rows, "finite", 0x195a_dafe),
            &s.stream,
        )?;
        upload(
            &inputs.weight,
            &weight_of(width, "finite", 0x0dd_ba11),
            &s.stream,
        )?;
        upload(&inputs.live, &rows.to_ne_bytes(), &s.stream)?;
        for output in &outputs {
            output.poison(s, width, rows, false)?;
        }
        for graph in &graphs {
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
        }
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let expected = outputs[0].read(s, width, rows)?;
        for arm in 1..arms {
            compare(&expected, &outputs[arm].read(s, width, rows)?)?;
            for (role, buffer) in outputs[arm].buffers().into_iter().enumerate() {
                let at = if role == 3 { OFFSET * 4 } else { 0 };
                let mut bad = expected[role].clone();
                bad[at] ^= 1;
                upload(buffer, &bad, &s.stream)?;
                if compare(&expected, &outputs[arm].read(s, width, rows)?).is_ok() {
                    return Err("negative control escaped the comparison".into());
                }
                upload(buffer, &expected[role], &s.stream)?;
                negative += 1;
            }
            let label = format!("norm-query-cached:{}:rows={rows}", cached[arm - 1]);
            let timing = kernel_harness::with_timing_label(&label, || {
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |side, stream| {
                    graphs[if side == 0 { 0 } else { arm }]
                        .launch(stream)
                        .map_err(|e| e.to_string())
                })
            })?;
            let chained = kernel_harness::with_timing_label(&format!("{label}:chain"), || {
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |side, stream| {
                    chains[if side == 0 { 0 } else { arm }]
                        .launch(stream)
                        .map_err(|e| e.to_string())
                })
            })?;
            // The chain replays the timed inputs; its outputs must not move.
            s.stream.synchronize().map_err(|e| e.to_string())?;
            compare(&expected, &outputs[arm].read(s, width, rows)?)?;
            let per = |ms: f32| ms * 1000. / CHAIN as f32;
            let record = json!({"symbol":cached[arm - 1],"served":catalog::served().symbol,
                "warps":catalog::WARPS[arm - 1],"width":width,"rows":rows,
                "old_us":timing.arms[0].median_ms * 1000.,"new_us":timing.arms[1].median_ms * 1000.,
                "ratio":timing.paired_ratio,"chain":CHAIN,
                "old_chain_us":per(chained.arms[0].median_ms),"new_chain_us":per(chained.arms[1].median_ms),
                "chain_ratio":chained.paired_ratio});
            eprintln!("NORM_QUERY_CACHED_TIMING {record}");
            timings.push(record);
        }
    }
    let c = coverage();
    let expected = |key: &str| c[key].as_u64().unwrap();
    if bitwise as u64
        != if memcheck {
            expected("memcheck_cases")
        } else {
            expected("bitwise_cases")
        }
        || negative as u64
            != if memcheck {
                0
            } else {
                expected("negative_controls")
            }
        || timings.len() as u64
            != if memcheck {
                0
            } else {
                expected("timing_cases")
            }
    {
        return Err("executed coverage differs".into());
    }
    eprintln!(
        "NORM_QUERY_CACHED_COMPLETE memcheck={memcheck} bitwise={bitwise} negative={negative} timings={}",
        timings.len()
    );
    Ok(json!({"memcheck":memcheck,"bitwise":bitwise,"negative":negative,"timings":timings}))
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_NORM_QUERY_CACHED"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix(s, false)?;
        s.receipt(
            "norm-query-cached",
            value
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    });
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_NORM_QUERY_CACHED"]
fn memcheck_cached_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| matrix(s, true).map(|_| ()));
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
