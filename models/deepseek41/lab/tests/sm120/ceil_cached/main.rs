//! The one-pass cached ceil entries against the served parallel decode
//! quantizers: bitwise outputs over special patterns, live counts, graph
//! replays and output-flip negatives, then paired graph timing.
#[allow(dead_code, reason = "the build example uses the family and specs")]
mod catalog;
use bundle_deepseek41::sm120::fold_cubin_for_symbol;
use kernel_core::contract::DType;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::row::block_quantize::Entry;
use model_deepseek41_contract::serving::DECODE_BATCH;
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const ROWS: [u32; 3] = [1, 8, DECODE_BATCH];
const PATTERNS: [&str; 13] = [
    "finite",
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
];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
const OFFSET: usize = 3;

fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_CEIL_CACHED").expect("frozen ceil artifacts required"),
    )
}
/// Module name to its bytes and the symbols the matrix uses.
type Modules = BTreeMap<String, (Vec<u8>, Vec<String>)>;
/// The served parallel module(s), the lab cached module and the probe.
fn modules() -> LabResult<Modules> {
    let mut modules = Modules::new();
    for [old, _] in catalog::pairs() {
        let (name, bytes) =
            fold_cubin_for_symbol(&old.symbol).ok_or("served entry must be embedded")?;
        let module = modules
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), Vec::new()));
        module.1.push(old.symbol.clone());
    }
    let cached = std::fs::read(artifacts().join(format!("{}.cubin", catalog::MODULE)))
        .map_err(|e| e.to_string())?;
    let lock: serde_json::Value =
        serde_json::from_str(include_str!("../../../kernels.lock.d/ceil_cached.lock"))
            .map_err(|e| e.to_string())?;
    if lock[0]["name"] != catalog::MODULE
        || lock[0]["cubin_sha256"] != kernel_build::digest(&cached)
    {
        return Err("cached module differs from the lab lock".into());
    }
    modules.insert(
        catalog::MODULE.into(),
        (
            cached,
            catalog::pairs()
                .into_iter()
                .map(|[_, new]| new.symbol)
                .collect(),
        ),
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
    let pairs = catalog::pairs();
    json!({"entries": pairs.iter().map(|[o, n]| json!({"old":o.symbol,"new":n.symbol,"width":o.query.width,"f32":o.query.input == DType::F32})).collect::<Vec<_>>(),
        "rows": ROWS, "patterns": PATTERNS, "bitwise_cases": pairs.len() * ROWS.len() * PATTERNS.len(),
        "timing_cases": pairs.len() * ROWS.len(), "negative_controls": pairs.len() * ROWS.len() * 3,
        "memcheck_cases": pairs.len()})
}

#[test]
fn cached_pairs_cover_the_c1_widths_and_keep_their_plans() {
    let pairs = catalog::pairs();
    assert_eq!(pairs.len(), 11);
    let widths: Vec<_> = pairs
        .iter()
        .map(|[o, _]| (o.query.width, o.query.input))
        .collect();
    for hot in [(8192, DType::Bf16), (2048, DType::F32), (1280, DType::F32)] {
        assert!(widths.contains(&hot), "{hot:?}");
    }
    let mut symbols = std::collections::BTreeSet::new();
    for [old, new] in &pairs {
        assert!(old.parallel_groups && !old.cached_groups);
        assert!(new.parallel_groups && new.cached_groups);
        assert_eq!(old.query, new.query);
        assert_eq!(old.capacity, new.capacity);
        for rows in ROWS {
            assert_eq!(old.plan(rows).unwrap(), new.plan(rows).unwrap());
        }
        assert!(
            fold_cubin_for_symbol(&old.symbol).is_some(),
            "{}",
            old.symbol
        );
        assert!(symbols.insert(new.symbol.clone()));
    }
    let c = coverage();
    assert_eq!(c["bitwise_cases"], 11 * 3 * 13);
}

/// Explicit artifact staging and checking, never part of an ordinary run.
#[test]
#[ignore = "requires an external frozen artifact directory named by CHOPINFER_CEIL_CACHED"]
fn frozen_modules_are_complete() {
    let dir = artifacts();
    let modules = modules().unwrap();
    if std::env::var_os("CHOPINFER_CEIL_EXPORT").is_some() {
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
        "CEIL_CACHED_MODULES {} PASS",
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
struct Output {
    codes: DeviceBuffer,
    scales: DeviceBuffer,
    status: DeviceBuffer,
}
impl Output {
    fn new(s: &Session, width: usize, rows: u32) -> LabResult<Self> {
        Ok(Self {
            codes: s.allocate(width * rows as usize)?,
            scales: s.allocate(width * rows as usize / 32)?,
            status: s.allocate((rows as usize + OFFSET + 2) * 4)?,
        })
    }
    fn poison(&self, s: &Session, width: usize, rows: u32, sticky: bool) -> LabResult<()> {
        upload(&self.codes, &vec![0xa5; width * rows as usize], &s.stream)?;
        upload(
            &self.scales,
            &vec![0x5a; width * rows as usize / 32],
            &s.stream,
        )?;
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
    fn read(&self, s: &Session, width: usize, rows: u32) -> LabResult<[Vec<u8>; 3]> {
        Ok([
            download(&self.codes, width * rows as usize, &s.stream)?,
            download(&self.scales, width * rows as usize / 32, &s.stream)?,
            download(&self.status, (rows as usize + OFFSET + 2) * 4, &s.stream)?,
        ])
    }
}
#[allow(clippy::too_many_arguments)]
fn launch_entry(
    entry: &Entry,
    function: &Function,
    rows: u32,
    input: &DeviceBuffer,
    out: &Output,
    live: &DeviceBuffer,
    stream: &Stream,
) -> LabResult<()> {
    use runtime_cuda::KernelArgument as A;
    let args = [
        A::Buffer(input),
        A::Buffer(&out.codes),
        A::Buffer(&out.scales),
        A::Buffer(live),
        A::Buffer(&out.status),
        A::U64(OFFSET as u64),
    ];
    function
        .launch(stream, launch::config(entry.plan(rows).unwrap())?, &args)
        .map_err(|e| e.to_string())
}
fn inputs(entry: &Entry, rows: u32, pattern: &str, mut state: u64) -> Vec<u8> {
    let width = entry.query.width as usize;
    let values = (0..width * rows as usize).map(|i| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let sign = (state as u32) & 0x8000_0000;
        let random = sign | (((state >> 32) as u32 % 100 + 77) << 23) | (state as u32 & 0x7f_ffff);
        let at = i % width;
        let bits = match pattern {
            "wide" => sign | (((state >> 32) as u32 % 254 + 1) << 23) | (state as u32 & 0x7f_ffff),
            "zero" => sign,
            "denormal" => {
                sign | if entry.query.input == DType::F32 {
                    (state as u32 & 0x7f_ffff).max(1)
                } else {
                    (state as u32 & 0x7f_0000).max(0x1_0000)
                }
            }
            "maximum" => {
                sign | if entry.query.input == DType::F32 {
                    0x7f7f_ffff
                } else {
                    0x7f7f_0000
                }
            }
            "nan_block" if i / width == 0 && at >= width - 32 => sign | 0x7fc1_0000,
            "inf_block" if i / width == 0 && at >= width - 32 => sign | 0x7f80_0000,
            "nan_first" if i / width == 0 && at == 0 => 0x7fc1_0000,
            "nan_last" if i / width == 0 && at == width - 1 => 0xff81_0000,
            "inf_first" if i / width == 0 && at == 0 => 0x7f80_0000,
            "inf_last" if i / width == 0 && at == width - 1 => 0xff80_0000,
            "boundaries" => {
                let e = ((i / 32) % 100) as i32 - 50;
                let v = (448.0f32 * 2.0f32.powi(e)).to_bits();
                sign | match i % 3 {
                    0 => v - 1,
                    1 => v,
                    _ => v + 1,
                }
            }
            _ => random,
        };
        if entry.query.input == DType::F32 {
            bits.to_ne_bytes().to_vec()
        } else {
            ((bits >> 16) as u16).to_ne_bytes().to_vec()
        }
    });
    values.flatten().collect()
}
fn compare(expected: &[Vec<u8>; 3], actual: &[Vec<u8>; 3]) -> LabResult<()> {
    for (role, (a, b)) in ["codes", "scales", "status"]
        .into_iter()
        .zip(expected.iter().zip(actual))
    {
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
    functions: Vec<[Function; 2]>,
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
    let functions = catalog::pairs()
        .iter()
        .map(|[old, new]| {
            Ok([
                module_of(&old.symbol)?
                    .function(&old.symbol)
                    .map_err(|e| e.to_string())?,
                module_of(&new.symbol)?
                    .function(&new.symbol)
                    .map_err(|e| e.to_string())?,
            ])
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
/// Every pair at every row count and pattern, eager and two graph replays,
/// at full, zero, partial and excess live counts; `memcheck` runs one
/// finite and one non-finite case per pair, eagerly, without timing.
fn matrix(s: &Session, memcheck: bool) -> LabResult<serde_json::Value> {
    let k = load(s)?;
    let (mut bitwise, mut negative) = (0, 0);
    let mut timings = Vec::new();
    for (pair, functions) in catalog::pairs().iter().zip(&k.functions) {
        let width = pair[0].query.width as usize;
        let f32 = pair[0].query.input == DType::F32;
        for rows in if memcheck { vec![8] } else { ROWS.to_vec() } {
            let input = s.allocate(width * rows as usize * if f32 { 4 } else { 2 })?;
            let live = s.allocate(4)?;
            let outputs = [Output::new(s, width, rows)?, Output::new(s, width, rows)?];
            let graphs = (0..2)
                .map(|arm| {
                    s.capture(|stream| {
                        launch_entry(
                            &pair[arm],
                            &functions[arm],
                            rows,
                            &input,
                            &outputs[arm],
                            &live,
                            stream,
                        )
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
                    &input,
                    &inputs(&pair[0], rows, pattern, 0x91a2_03c4),
                    &s.stream,
                )?;
                let lives: &[u32] = if memcheck {
                    &[8]
                } else {
                    &[rows, 0, rows.saturating_sub(1), rows + 7, rows]
                };
                for &live_rows in lives {
                    let active = live_rows.min(rows);
                    upload(&live, &live_rows.to_ne_bytes(), &s.stream)?;
                    let mut eager = None;
                    for graph in if memcheck {
                        vec![false]
                    } else {
                        vec![false, true, true]
                    } {
                        for output in &outputs {
                            output.poison(s, width, rows, pattern == "sticky")?;
                        }
                        for arm in 0..2 {
                            if graph {
                                graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                            } else {
                                launch_entry(
                                    &pair[arm],
                                    &functions[arm],
                                    rows,
                                    &input,
                                    &outputs[arm],
                                    &live,
                                    &s.stream,
                                )?;
                            }
                        }
                        s.stream.synchronize().map_err(|e| e.to_string())?;
                        let old = outputs[0].read(s, width, rows)?;
                        let new = outputs[1].read(s, width, rows)?;
                        compare(&old, &new).map_err(|e| {
                            format!(
                                "{} rows={rows} live={live_rows} {pattern} graph={graph}: {e}",
                                pair[1].symbol
                            )
                        })?;
                        if let Some(ref previous) = eager {
                            compare(previous, &new)?;
                        } else {
                            eager = Some(old);
                        }
                        if !new[0][active as usize * width..].iter().all(|b| *b == 0xa5)
                            || !new[1][active as usize * width / 32..]
                                .iter()
                                .all(|b| *b == 0x5a)
                            || (0..OFFSET)
                                .chain(OFFSET + rows as usize..OFFSET + rows as usize + 2)
                                .any(|i| new[2][i * 4..i * 4 + 4] != 0x1357_2468u32.to_ne_bytes())
                        {
                            return Err(format!(
                                "{} rows={rows}: guard or inactive output changed",
                                pair[1].symbol
                            ));
                        }
                        if pattern == "finite"
                            && live_rows > 0
                            && new[0][..width].iter().all(|b| *b == 0)
                        {
                            return Err("finite row produced no codes".into());
                        }
                    }
                }
                bitwise += 1;
                eprintln!(
                    "CEIL_CACHED_BITWISE {}",
                    json!({"symbol":pair[1].symbol,"rows":rows,"pattern":pattern})
                );
            }
            if memcheck {
                continue;
            }
            upload(
                &input,
                &inputs(&pair[0], rows, "finite", 0x195a_dafe),
                &s.stream,
            )?;
            upload(&live, &rows.to_ne_bytes(), &s.stream)?;
            for output in &outputs {
                output.poison(s, width, rows, false)?;
            }
            for graph in &graphs {
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let expected = outputs[0].read(s, width, rows)?;
            compare(&expected, &outputs[1].read(s, width, rows)?)?;
            for (role, buffer, at) in [
                (0, &outputs[1].codes, 0),
                (1, &outputs[1].scales, 0),
                (2, &outputs[1].status, OFFSET * 4),
            ] {
                let mut bad = expected[role].clone();
                bad[at] ^= 1;
                upload(buffer, &bad, &s.stream)?;
                if compare(&expected, &outputs[1].read(s, width, rows)?).is_ok() {
                    return Err("negative control escaped the comparison".into());
                }
                upload(buffer, &expected[role], &s.stream)?;
                negative += 1;
            }
            let label = format!("ceil-cached:{}:rows={rows}", pair[1].symbol);
            let timing = kernel_harness::with_timing_label(&label, || {
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })
            })?;
            let record = json!({"symbol":pair[1].symbol,"served":pair[0].symbol,"width":width,"rows":rows,
                "old_us":timing.arms[0].median_ms * 1000.,"new_us":timing.arms[1].median_ms * 1000.,"ratio":timing.paired_ratio});
            eprintln!("CEIL_CACHED_TIMING {record}");
            timings.push(record);
        }
    }
    let c = coverage();
    let expected_bitwise = if memcheck {
        c["memcheck_cases"].as_u64().unwrap() * 2
    } else {
        c["bitwise_cases"].as_u64().unwrap()
    };
    if bitwise as u64 != expected_bitwise
        || negative as u64
            != if memcheck {
                0
            } else {
                c["negative_controls"].as_u64().unwrap()
            }
        || timings.len() as u64
            != if memcheck {
                0
            } else {
                c["timing_cases"].as_u64().unwrap()
            }
    {
        return Err("executed coverage differs".into());
    }
    eprintln!(
        "CEIL_CACHED_COMPLETE memcheck={memcheck} bitwise={bitwise} negative={negative} timings={}",
        timings.len()
    );
    Ok(json!({"memcheck":memcheck,"bitwise":bitwise,"negative":negative,"timings":timings}))
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_CEIL_CACHED"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix(s, false)?;
        s.receipt(
            "ceil-cached",
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
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_CEIL_CACHED"]
fn memcheck_cached_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| matrix(s, true).map(|_| ()));
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
