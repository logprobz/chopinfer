//! The cluster-split one-row ceil twins (`bundle_deepseek41::sm120::ceil_split`)
//! against the served decode quantizers they would replace at Flash TP4:
//! bitwise codes, scales and status over special patterns (non-finite
//! values in every part of the row), live counts, graph replays and
//! output-flip negatives, then paired timing of one launch and of a PDL
//! chain. Both modules are the bundle's embedded, locked cubins.
use bundle_deepseek41::sm120::{ceil_split as registration, fold_cubin_for_symbol};
use kernel_core::contract::DType;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::row::block_quantize::Entry;
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const RANKS: u32 = 4;
const ROWS: u32 = registration::ROWS;
const PATTERNS: [&str; 15] = [
    "finite",
    "wide",
    "zero",
    "denormal",
    "maximum",
    "nan_first",
    "nan_last",
    "nan_middle",
    "inf_first",
    "inf_last",
    "inf_quarter",
    "boundaries",
    "sticky",
    "nan_block",
    "inf_block",
];
const LIVES: [u32; 5] = [ROWS, 0, ROWS - 1, ROWS + 7, ROWS];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
const OFFSET: usize = 3;
/// Launches per chained timing graph: in a decode step each quantizer runs
/// behind a PDL predecessor, and one cold launch is below the event timer's
/// resolution.
const CHAIN: usize = 16;

fn pairs() -> Vec<[Entry; 2]> {
    registration::pairs(&FLASH, RANKS).unwrap()
}
fn artifacts() -> PathBuf {
    PathBuf::from(std::env::var_os("CHOPINFER_CEIL_SPLIT").expect("frozen ceil artifacts required"))
}
/// Module name to its bytes and the symbols the matrix uses.
type Modules = BTreeMap<String, (Vec<u8>, Vec<String>)>;
/// The embedded served and split modules, and the timing probe.
fn modules() -> LabResult<Modules> {
    let mut modules = Modules::new();
    for pair in pairs() {
        for entry in pair {
            let (name, bytes) =
                fold_cubin_for_symbol(&entry.symbol).ok_or("entry must be embedded")?;
            let module = modules
                .entry(name.into())
                .or_insert_with(|| (bytes.to_vec(), Vec::new()));
            if !module.1.contains(&entry.symbol) {
                module.1.push(entry.symbol.clone());
            }
        }
    }
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
    let pairs = pairs();
    json!({"entries": pairs.iter().map(|[o, n]| json!({"served":o.symbol,"split":n.symbol,"width":o.query.width,
            "f32":o.query.input == DType::F32,"parts":n.split_parts})).collect::<Vec<_>>(),
        "rows": ROWS, "patterns": PATTERNS, "lives": LIVES, "chain": CHAIN,
        "bitwise_cases": pairs.len() * PATTERNS.len(), "timing_cases": pairs.len(),
        "negative_controls": pairs.len() * 3, "memcheck_cases": pairs.len()})
}

#[test]
fn split_pairs_cover_the_c1_widths_and_keep_the_served_abi() {
    let pairs = pairs();
    assert_eq!(pairs.len(), 6);
    for [served, split] in &pairs {
        assert!(served.parallel_groups && served.early_release && served.split_parts == 0);
        assert!(split.cached_groups && split.early_release);
        assert_eq!(served.query, split.query);
        assert_eq!(served.spec().params, split.spec().params);
        assert!(served.capacity >= ROWS && split.capacity == ROWS);
        let [old, new] = [served.plan(ROWS).unwrap(), split.plan(ROWS).unwrap()];
        assert_eq!(old.grid, [1, 1, 1]);
        assert_eq!(new.grid, [split.split_parts, 1, 1]);
        assert_eq!(new.cluster, [split.split_parts, 1, 1]);
        assert!(old.pdl && new.pdl);
        for entry in [served, split] {
            assert!(
                fold_cubin_for_symbol(&entry.symbol).is_some(),
                "{}",
                entry.symbol
            );
        }
    }
    let widths: std::collections::BTreeSet<_> = pairs
        .iter()
        .map(|[o, _]| (o.query.width, o.query.input == DType::F32))
        .collect();
    assert_eq!(widths, [(2048, true), (8192, false)].into());
    for [_, split] in &pairs {
        // These patterns' single non-finite values lie outside part 0.
        let width = split.query.width as usize;
        for pattern in ["nan_middle", "inf_quarter"] {
            assert!(
                non_finite_column(pattern, width).unwrap() >= width / split.split_parts as usize
            );
        }
    }
    assert_eq!(coverage()["bitwise_cases"], 6 * 15);
}

/// Explicit artifact staging and checking, never part of an ordinary run.
#[test]
#[ignore = "requires an external frozen artifact directory named by CHOPINFER_CEIL_SPLIT"]
fn frozen_modules_are_complete() {
    let dir = artifacts();
    let modules = modules().unwrap();
    if std::env::var_os("CHOPINFER_CEIL_SPLIT_EXPORT").is_some() {
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
        assert_eq!(
            std::fs::read(dir.join(format!("{name}.cubin"))).unwrap(),
            bytes
        );
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
        "CEIL_SPLIT_MODULES {} PASS",
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
    fn new(s: &Session, width: usize) -> LabResult<Self> {
        Ok(Self {
            codes: s.allocate(width * ROWS as usize)?,
            scales: s.allocate(width * ROWS as usize / 32)?,
            status: s.allocate((ROWS as usize + OFFSET + 2) * 4)?,
        })
    }
    fn poison(&self, s: &Session, width: usize, sticky: bool) -> LabResult<()> {
        upload(&self.codes, &vec![0xa5; width * ROWS as usize], &s.stream)?;
        upload(
            &self.scales,
            &vec![0x5a; width * ROWS as usize / 32],
            &s.stream,
        )?;
        let mut flags = vec![0x1357_2468u32; ROWS as usize + OFFSET + 2];
        flags[OFFSET..OFFSET + ROWS as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.status,
            &flags
                .iter()
                .flat_map(|s| s.to_ne_bytes())
                .collect::<Vec<_>>(),
            &s.stream,
        )
    }
    fn read(&self, s: &Session, width: usize) -> LabResult<[Vec<u8>; 3]> {
        Ok([
            download(&self.codes, width * ROWS as usize, &s.stream)?,
            download(&self.scales, width * ROWS as usize / 32, &s.stream)?,
            download(&self.status, (ROWS as usize + OFFSET + 2) * 4, &s.stream)?,
        ])
    }
}
fn launch_entry(
    entry: &Entry,
    function: &Function,
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
        .launch(stream, launch::config(entry.plan(ROWS).unwrap())?, &args)
        .map_err(|e| e.to_string())
}
/// The column of a pattern's single non-finite value in row 0.
fn non_finite_column(pattern: &str, width: usize) -> Option<usize> {
    match pattern {
        "nan_first" | "inf_first" => Some(0),
        "nan_last" | "inf_last" => Some(width - 1),
        "nan_middle" => Some(width / 2 + 5),
        "inf_quarter" => Some(3 * width / 4 + 3),
        _ => None,
    }
}
fn inputs(entry: &Entry, pattern: &str, mut state: u64) -> Vec<u8> {
    let width = entry.query.width as usize;
    let single = non_finite_column(pattern, width);
    let values = (0..width * ROWS as usize).map(|i| {
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
            "nan_first" | "nan_middle" if i / width == 0 && Some(at) == single => 0x7fc1_0000,
            "nan_last" if i / width == 0 && Some(at) == single => 0xff81_0000,
            "inf_first" if i / width == 0 && Some(at) == single => 0x7f80_0000,
            "inf_last" | "inf_quarter" if i / width == 0 && Some(at) == single => 0xff80_0000,
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
    let functions = pairs()
        .iter()
        .map(|pair| {
            let function = |entry: &Entry| {
                module_of(&entry.symbol)?
                    .function(&entry.symbol)
                    .map_err(|e| e.to_string())
            };
            Ok([function(&pair[0])?, function(&pair[1])?])
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
/// Every pair at every pattern, eager and two graph replays, at full, zero
/// and excess live counts; `memcheck` runs one finite and two non-finite
/// cases per pair (the second outside part 0), eagerly, without timing.
fn matrix(s: &Session, memcheck: bool) -> LabResult<serde_json::Value> {
    let k = load(s)?;
    let (mut bitwise, mut negative) = (0, 0);
    let mut timings = Vec::new();
    for (pair, functions) in pairs().iter().zip(&k.functions) {
        let width = pair[0].query.width as usize;
        let f32 = pair[0].query.input == DType::F32;
        let input = s.allocate(width * ROWS as usize * if f32 { 4 } else { 2 })?;
        let live = s.allocate(4)?;
        let outputs = [Output::new(s, width)?, Output::new(s, width)?];
        let capture = |arm: usize, launches: usize| {
            s.capture(|stream| {
                (0..launches).try_for_each(|_| {
                    launch_entry(
                        &pair[arm],
                        &functions[arm],
                        &input,
                        &outputs[arm],
                        &live,
                        stream,
                    )
                })
            })
        };
        let graphs = [capture(0, 1)?, capture(1, 1)?];
        let chains = [capture(0, CHAIN)?, capture(1, CHAIN)?];
        let patterns: &[&str] = if memcheck {
            &["finite", "nan_last", "nan_middle"]
        } else {
            &PATTERNS
        };
        for &pattern in patterns {
            upload(&input, &inputs(&pair[0], pattern, 0x91a2_03c4), &s.stream)?;
            let lives: &[u32] = if memcheck { &[ROWS] } else { &LIVES };
            for &live_rows in lives {
                let active = live_rows.min(ROWS) as usize;
                upload(&live, &live_rows.to_ne_bytes(), &s.stream)?;
                let mut eager = None;
                for graph in if memcheck {
                    vec![false]
                } else {
                    vec![false, true, true]
                } {
                    for output in &outputs {
                        output.poison(s, width, pattern == "sticky")?;
                    }
                    for arm in 0..2 {
                        if graph {
                            graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                        } else {
                            launch_entry(
                                &pair[arm],
                                &functions[arm],
                                &input,
                                &outputs[arm],
                                &live,
                                &s.stream,
                            )?;
                        }
                    }
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let old = outputs[0].read(s, width)?;
                    let new = outputs[1].read(s, width)?;
                    compare(&old, &new).map_err(|e| {
                        format!(
                            "{} live={live_rows} {pattern} graph={graph}: {e}",
                            pair[1].symbol
                        )
                    })?;
                    if let Some(ref previous) = eager {
                        compare(previous, &new)?;
                    } else {
                        eager = Some(old);
                    }
                    if !new[0][active * width..].iter().all(|b| *b == 0xa5)
                        || !new[1][active * width / 32..].iter().all(|b| *b == 0x5a)
                        || (0..OFFSET)
                            .chain(OFFSET + ROWS as usize..OFFSET + ROWS as usize + 2)
                            .any(|i| new[2][i * 4..i * 4 + 4] != 0x1357_2468u32.to_ne_bytes())
                    {
                        return Err(format!(
                            "{}: guard or inactive output changed",
                            pair[1].symbol
                        ));
                    }
                    let status =
                        u32::from_ne_bytes(new[2][OFFSET * 4..OFFSET * 4 + 4].try_into().unwrap());
                    let flagged = pattern == "sticky"
                        || non_finite_column(pattern, width).is_some()
                        || pattern.ends_with("_block");
                    if live_rows > 0 && (pattern == "finite" || flagged) && (status != 0) != flagged
                    {
                        return Err(format!(
                            "{} {pattern}: row status {status:#x} against the pattern",
                            pair[1].symbol
                        ));
                    }
                    if live_rows > 0
                        && flagged
                        && (new[0][..width].iter().any(|b| *b != 0)
                            || new[1][..width / 32].iter().any(|b| *b != 127))
                    {
                        return Err(format!(
                            "{} {pattern}: a flagged row kept codes or scales",
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
                "CEIL_SPLIT_BITWISE {}",
                json!({"symbol":pair[1].symbol,"pattern":pattern})
            );
        }
        if memcheck {
            continue;
        }
        upload(&input, &inputs(&pair[0], "finite", 0x195a_dafe), &s.stream)?;
        upload(&live, &ROWS.to_ne_bytes(), &s.stream)?;
        for output in &outputs {
            output.poison(s, width, false)?;
        }
        for graph in &graphs {
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
        }
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let expected = outputs[0].read(s, width)?;
        compare(&expected, &outputs[1].read(s, width)?)?;
        for (role, buffer, at) in [
            (0, &outputs[1].codes, 0),
            (1, &outputs[1].scales, 0),
            (2, &outputs[1].status, OFFSET * 4),
        ] {
            let mut bad = expected[role].clone();
            bad[at] ^= 1;
            upload(buffer, &bad, &s.stream)?;
            if compare(&expected, &outputs[1].read(s, width)?).is_ok() {
                return Err("negative control escaped the comparison".into());
            }
            upload(buffer, &expected[role], &s.stream)?;
            negative += 1;
        }
        let label = format!("ceil-split:{}", pair[1].symbol);
        let single = kernel_harness::with_timing_label(&label, || {
            kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })
        })?;
        let chained = kernel_harness::with_timing_label(&format!("{label}:chain"), || {
            kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                chains[arm].launch(stream).map_err(|e| e.to_string())
            })
        })?;
        // The timed graphs replay the finite inputs; their outputs must not move.
        s.stream.synchronize().map_err(|e| e.to_string())?;
        for output in &outputs {
            compare(&expected, &output.read(s, width)?)?;
        }
        let per = |ms: f32| ms * 1000. / CHAIN as f32;
        let record = json!({"symbol":pair[1].symbol,"served":pair[0].symbol,"width":width,
            "parts":pair[1].split_parts,"old_us":single.arms[0].median_ms * 1000.,
            "new_us":single.arms[1].median_ms * 1000.,"ratio":single.paired_ratio,"chain":CHAIN,
            "old_chain_us":per(chained.arms[0].median_ms),"new_chain_us":per(chained.arms[1].median_ms),
            "chain_ratio":chained.paired_ratio});
        eprintln!("CEIL_SPLIT_TIMING {record}");
        timings.push(record);
    }
    let c = coverage();
    let expected = |key: &str| c[key].as_u64().unwrap();
    if bitwise as u64
        != if memcheck {
            expected("memcheck_cases") * 3
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
        "CEIL_SPLIT_COMPLETE memcheck={memcheck} bitwise={bitwise} negative={negative} timings={}",
        timings.len()
    );
    Ok(json!({"memcheck":memcheck,"bitwise":bitwise,"negative":negative,"timings":timings}))
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_CEIL_SPLIT"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix(s, false)?;
        s.receipt(
            "ceil-split",
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

/// Run under compute-sanitizer memcheck, racecheck and synccheck.
#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_CEIL_SPLIT"]
fn sanitized_split_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| matrix(s, true).map(|_| ()));
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
