//! Exact comparisons of the served serial entries and parallel decode entries.
use bundle_deepseek41::sm120::{ceil_parallel as registration, fold_cubin_for_symbol};
use kernel_core::contract::{DType, LaunchPlan};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::row::{block_quantize, silu_mul_quantize};
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS, geometry::GEOMETRIES, serving::DECODE_BATCH,
};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const ROWS: [u32; 3] = [1, 8, DECODE_BATCH];
const PATTERNS: [&str; 15] = [
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
    "up_nan",
    "up_inf",
    "nan_block",
    "inf_block",
];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
const OFFSET: usize = 3;

#[derive(Clone)]
enum Entry {
    Quant(block_quantize::Entry),
    Silu(silu_mul_quantize::Entry),
}
impl Entry {
    fn symbol(&self) -> &str {
        match self {
            Self::Quant(e) => &e.symbol,
            Self::Silu(e) => &e.symbol,
        }
    }
    fn width(&self) -> usize {
        (match self {
            Self::Quant(e) => e.query.width,
            Self::Silu(e) => e.query.width,
        }) as usize
    }
    fn f32(&self) -> bool {
        match self {
            Self::Quant(e) => e.query.input == DType::F32,
            Self::Silu(_) => true,
        }
    }
    fn plan(&self, rows: u32) -> LaunchPlan {
        match self {
            Self::Quant(e) => e.plan(rows),
            Self::Silu(e) => e.plan(rows),
        }
        .unwrap()
    }
}
fn pairs() -> Vec<[Entry; 2]> {
    let mut pairs = BTreeMap::new();
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            for (old, new) in registration::baseline_quantizers(g, ranks)
                .unwrap()
                .into_iter()
                .zip(registration::quantizers(g, ranks).unwrap())
            {
                pairs.insert(new.symbol.clone(), [Entry::Quant(old), Entry::Quant(new)]);
            }
            let new = registration::activation(g, ranks).unwrap();
            let mut old = new.clone();
            old.parallel_groups = false;
            old.symbol = old.symbol.replace("_parallel_v1", "_v1");
            pairs.insert(new.symbol.clone(), [Entry::Silu(old), Entry::Silu(new)]);
        }
    }
    pairs.into_values().collect()
}
fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_CEIL_ARTIFACTS").expect("frozen ceil artifacts required"),
    )
}
fn modules() -> BTreeMap<String, (Vec<u8>, Vec<String>)> {
    let mut modules = BTreeMap::<String, (Vec<u8>, Vec<String>)>::new();
    for pair in pairs() {
        for entry in pair {
            let (name, bytes) =
                fold_cubin_for_symbol(entry.symbol()).expect("served entry must be embedded");
            let module = modules
                .entry(name.into())
                .or_insert_with(|| (bytes.to_vec(), Vec::new()));
            if !module.1.iter().any(|s| s == entry.symbol()) {
                module.1.push(entry.symbol().into());
            }
        }
    }
    modules
}
fn verify_modules() -> LabResult<()> {
    let dir = artifacts();
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("modules.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut expected = modules();
    expected.insert(
        "probe_sm120".into(),
        (
            Vec::new(),
            PROBE_SYMBOLS.iter().map(|s| (*s).into()).collect(),
        ),
    );
    if manifest.as_array().ok_or("module array required")?.len() != expected.len() {
        return Err("module set differs".into());
    }
    for (name, (embedded, symbols)) in expected {
        let bytes = std::fs::read(dir.join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        let record = manifest
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == name)
            .ok_or("module missing from manifest")?;
        if record["sha256"] != kernel_build::digest(&bytes)
            || (!embedded.is_empty() && embedded != bytes)
        {
            return Err(format!("module digest mismatch: {name}"));
        }
        if !bytes.starts_with(b"\x7fELF") {
            return Err(format!("not a cubin: {name}"));
        }
        for symbol in symbols {
            if !bytes
                .windows(symbol.len() + 1)
                .any(|s| s[..symbol.len()] == *symbol.as_bytes() && s[symbol.len()] == 0)
            {
                return Err(format!("missing export {symbol} in {name}"));
            }
        }
    }
    Ok(())
}

#[test]
fn decode_selection_preserves_prefill_and_covers_all_widths() {
    use kernel_core::backend::{Backend, Need};
    use kernels_sm120::backend::Sm120;
    for pair in pairs() {
        match pair {
            [Entry::Quant(old), Entry::Quant(new)] => {
                for reverse in [false, true] {
                    let entries = if reverse {
                        [new.clone(), old.clone()]
                    } else {
                        [old.clone(), new.clone()]
                    };
                    let backend = entries
                        .into_iter()
                        .fold(Sm120::new(), Sm120::with_block_quantize);
                    for rows in ROWS {
                        assert_eq!(
                            backend
                                .block_quantize(&old.query, Need::rows(rows))
                                .unwrap()
                                .info()
                                .symbols[0]
                                .0,
                            new.symbol
                        );
                    }
                    assert_eq!(
                        backend
                            .block_quantize(&old.query, Need::rows(DECODE_BATCH + 1))
                            .unwrap()
                            .info()
                            .symbols[0]
                            .0,
                        old.symbol
                    );
                }
            }
            [Entry::Silu(mut old), Entry::Silu(new)] => {
                old.capacity = model_deepseek41_contract::serving::PREFILL_TOKENS;
                let backend = Sm120::new()
                    .with_silu_mul_quantize(old.clone())
                    .with_silu_mul_quantize(new.clone());
                assert_eq!(
                    backend
                        .silu_mul_quantize(&old.query, Need::rows(DECODE_BATCH))
                        .unwrap()
                        .info()
                        .symbols[0]
                        .0,
                    new.symbol
                );
                assert_eq!(
                    backend
                        .silu_mul_quantize(&old.query, Need::rows(DECODE_BATCH + 1))
                        .unwrap()
                        .info()
                        .symbols[0]
                        .0,
                    old.symbol
                );
            }
            _ => unreachable!(),
        }
    }
    let widths: Vec<_> = pairs().iter().map(|p| p[0].width()).collect();
    for width in [512, 576, 1280, 2048, 3456, 5120, 6144, 8192] {
        assert!(widths.contains(&width));
    }
}

/// Explicit artifact staging/checking, never part of an ordinary test run.
#[test]
#[ignore = "requires an external frozen artifact directory"]
fn frozen_modules_are_complete() {
    if std::env::var_os("CHOPINFER_CEIL_EXPORT").is_some() {
        let dir = artifacts();
        std::fs::create_dir_all(&dir).unwrap();
        assert!(
            !dir.canonicalize().unwrap().starts_with(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../..")
                    .canonicalize()
                    .unwrap()
            )
        );
        let mut all = modules();
        let probe = std::fs::read(
            std::env::var_os("CHOPINFER_CEIL_TIMING_PROBE")
                .expect("kernel-lab timing probe required"),
        )
        .unwrap();
        // Authenticate the timing probe against the existing lab lock before freezing it.
        let lock: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../crates/kernel-lab/kernels.lock"
        ))
        .unwrap();
        let pinned = lock
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "probe_sm120")
            .unwrap();
        assert_eq!(pinned["cubin_sha256"], kernel_build::digest(&probe));
        all.insert(
            "probe_sm120".into(),
            (probe, PROBE_SYMBOLS.iter().map(|s| (*s).into()).collect()),
        );
        let mut manifest = Vec::new();
        for (name, (bytes, symbols)) in all {
            std::fs::write(dir.join(format!("{name}.cubin")), &bytes).unwrap();
            manifest
                .push(json!({"name":name,"sha256":kernel_build::digest(&bytes),"symbols":symbols}));
        }
        std::fs::write(
            dir.join("modules.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let coverage = json!({"entries": pairs().iter().map(|p| json!({"old":p[0].symbol(),"new":p[1].symbol(),"width":p[0].width()})).collect::<Vec<_>>(), "rows": ROWS, "patterns": PATTERNS,
            "bitwise_cases":pairs().len()*ROWS.len()*PATTERNS.len(), "timing_cases":pairs().len()*ROWS.len(), "negative_controls":pairs().len()*ROWS.len()*3});
        std::fs::write(
            dir.join("coverage.json"),
            serde_json::to_vec_pretty(&coverage).unwrap(),
        )
        .unwrap();
    }
    verify_modules().unwrap();
    eprintln!(
        "CEIL_MODULES {} PASS (including probe_sm120)",
        modules().len() + 1
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
    up: &DeviceBuffer,
    out: &Output,
    live: &DeviceBuffer,
    stream: &Stream,
) -> LabResult<()> {
    use runtime_cuda::KernelArgument as A;
    let args = match entry {
        Entry::Quant(_) => vec![
            A::Buffer(input),
            A::Buffer(&out.codes),
            A::Buffer(&out.scales),
            A::Buffer(live),
            A::Buffer(&out.status),
            A::U64(OFFSET as u64),
        ],
        Entry::Silu(_) => vec![
            A::Buffer(input),
            A::Buffer(up),
            A::Buffer(&out.codes),
            A::Buffer(&out.scales),
            A::Buffer(live),
            A::Buffer(&out.status),
            A::U64(OFFSET as u64),
        ],
    };
    function
        .launch(stream, launch::config(entry.plan(rows))?, &args)
        .map_err(|e| e.to_string())
}
fn inputs(entry: &Entry, rows: u32, pattern: &str, mut state: u64) -> Vec<u8> {
    let width = entry.width();
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
                sign | if entry.f32() {
                    (state as u32 & 0x7f_ffff).max(1)
                } else {
                    (state as u32 & 0x7f_0000).max(0x1_0000)
                }
            }
            "maximum" => {
                sign | if entry.f32() {
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
        if entry.f32() {
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

#[test]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("ceil comparison needs an idle GPU").unwrap();
        return;
    }
    verify_modules().unwrap();
    let probe = Box::leak(
        std::fs::read(artifacts().join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    let cubins = Cubins::new(
        Box::leak(Box::new([("probe_sm120", &probe[..])])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    );
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, |s| {
        let mut timings = Vec::new();
        let mut bitwise = 0;
        let mut negative = 0;
        for pair in pairs() {
            let loaded = pair.iter().map(|entry| {
                let (name, _) = fold_cubin_for_symbol(entry.symbol()).unwrap();
                s.load_cubin(&std::fs::read(artifacts().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?)
            }).collect::<LabResult<Vec<_>>>()?;
            let functions = loaded.iter().zip(&pair).map(|(m,e)| m.function(e.symbol()).map_err(|e| e.to_string())).collect::<LabResult<Vec<_>>>()?;
            let width = pair[0].width();
            for rows in ROWS {
                let input = s.allocate(width*rows as usize*if pair[0].f32() {4} else {2})?;
                let up = s.allocate(width*rows as usize*4)?;
                let live = s.allocate(4)?;
                let outputs = [Output::new(s,width,rows)?, Output::new(s,width,rows)?];
                let graphs = (0..2).map(|arm| s.capture(|stream| launch_entry(&pair[arm],&functions[arm],rows,&input,&up,&outputs[arm],&live,stream))).collect::<LabResult<Vec<_>>>()?;
                for pattern in PATTERNS {
                    upload(&input,&inputs(&pair[0],rows,pattern,0x91a2_03c4),&s.stream)?;
                    // Exercise both gate and up non-finites on fused entries.
                    if matches!(&pair[0], Entry::Silu(_)) { upload(&up,&inputs(&pair[0],rows,match pattern { "up_nan" => "nan_last", "up_inf" => "inf_last", _ => pattern },0x3210_9876),&s.stream)?; }
                    for live_rows in [rows, 0, rows.saturating_sub(1), rows + 7, rows] {
                        let active = live_rows.min(rows);
                        upload(&live,&live_rows.to_ne_bytes(),&s.stream)?;
                        let mut eager = None;
                        for graph in [false,true,true] {
                            for output in &outputs { output.poison(s,width,rows,pattern=="sticky")?; }
                            for arm in 0..2 {
                                if graph { graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?; }
                                else { launch_entry(&pair[arm],&functions[arm],rows,&input,&up,&outputs[arm],&live,&s.stream)?; }
                            }
                            s.stream.synchronize().map_err(|e| e.to_string())?;
                            let old = outputs[0].read(s,width,rows)?;
                            let new = outputs[1].read(s,width,rows)?;
                            compare(&old,&new).map_err(|e| format!("{} rows={rows} live={live_rows} {pattern} graph={graph}: {e}",pair[0].symbol()))?;
                            if let Some(ref previous) = eager { compare(previous,&new)?; } else { eager=Some(old); }
                            assert!(new[0][active as usize*width..].iter().all(|b| *b==0xa5));
                            assert!(new[1][active as usize*width/32..].iter().all(|b| *b==0x5a));
                            for index in (0..OFFSET).chain(OFFSET+rows as usize..OFFSET+rows as usize+2) {
                                assert_eq!(&new[2][index*4..index*4+4],&0x1357_2468u32.to_ne_bytes());
                            }
                            if pattern=="finite" && live_rows>0 { assert!(new[0][..width].iter().any(|b| *b!=0)); }
                        }
                    }
                    bitwise+=1;
                    eprintln!("CEIL_BITWISE {} rows={rows} pattern={pattern} PASS",pair[0].symbol());
                }
                upload(&input,&inputs(&pair[0],rows,"finite",0x195a_dafe),&s.stream)?;
                if matches!(&pair[0],Entry::Silu(_)) { upload(&up,&inputs(&pair[0],rows,"finite",0x7777),&s.stream)?; }
                upload(&live,&rows.to_ne_bytes(),&s.stream)?;
                for output in &outputs { output.poison(s,width,rows,false)?; }
                for graph in &graphs { graph.launch(&s.stream).map_err(|e| e.to_string())?; }
                s.stream.synchronize().map_err(|e| e.to_string())?;
                let expected=outputs[0].read(s,width,rows)?;
                compare(&expected,&outputs[1].read(s,width,rows)?)?;
                for (role,buffer,at) in [(0,&outputs[1].codes,0),(1,&outputs[1].scales,0),(2,&outputs[1].status,OFFSET*4)] {
                    let mut bad=expected[role].clone(); bad[at]^=1;
                    upload(buffer,&bad,&s.stream)?;
                    assert!(compare(&expected,&outputs[1].read(s,width,rows)?).is_err());
                    upload(buffer,&expected[role],&s.stream)?;
                    negative+=1;
                }
                let label=format!("ceil:{}:rows={rows}",pair[0].symbol());
                let timing=kernel_harness::with_timing_label(&label,||kernel_harness::time_ab(s,3,15,[EvictionMode::Read;2],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string())))?;
                eprintln!("CEIL_TIMING {} rows={rows} old_us={} new_us={} ratio={}",pair[0].symbol(),timing.arms[0].median_ms*1000.,timing.arms[1].median_ms*1000.,timing.paired_ratio);
                timings.push(json!({"symbol":pair[0].symbol(),"rows":rows,"old_us":timing.arms[0].median_ms*1000.,"new_us":timing.arms[1].median_ms*1000.,"ratio":timing.paired_ratio}));
            }
        }
        assert_eq!(bitwise,pairs().len()*ROWS.len()*PATTERNS.len());
        assert_eq!(negative,pairs().len()*ROWS.len()*3);
        s.receipt("ceil-parallel",BTreeMap::from([("bitwise_cases".into(),json!(bitwise)),("negative_controls".into(),json!(negative)),("timings".into(),json!(timings))]))?;
        eprintln!("CEIL_COMPLETE bitwise={bitwise} negative={negative} timings={}",timings.len());
        Ok(())
    }).unwrap();
    assert!(result.is_some());
}

#[test]
fn special_block_fixtures_have_the_declared_values() {
    for pair in pairs() {
        let entry = &pair[0];
        let values = |pattern| {
            let bytes = inputs(entry, 2, pattern, 0x195a_dafe);
            if entry.f32() {
                bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            } else {
                bytes
                    .chunks_exact(2)
                    .map(|v| {
                        f32::from_bits(u32::from(u16::from_ne_bytes(v.try_into().unwrap())) << 16)
                    })
                    .collect::<Vec<_>>()
            }
        };
        let nan = values("nan_block");
        let inf = values("inf_block");
        assert_eq!(nan.iter().filter(|v| v.is_nan()).count(), 32);
        assert_eq!(inf.iter().filter(|v| v.is_infinite()).count(), 32);
        assert!(values("denormal").iter().all(|v| v.is_subnormal()));
        assert!(values("zero").iter().all(|v| *v == 0.0));
        assert!(values("finite").iter().all(|v| v.is_finite() && *v != 0.0));
    }
}
