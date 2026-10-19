//! Bitwise dense prefill ceil comparisons, row admission and graph replay.
#[path = "../../../bundle/src/sm120/dense.rs"]
#[allow(dead_code)]
mod dense;
#[path = "../../../bundle/src/sm120/latent_glue.rs"]
#[allow(dead_code)]
mod latent_glue;
#[path = "../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod moe;
#[path = "../../../bundle/src/sm120/ceil_prefill.rs"]
#[allow(dead_code)]
mod registration;
use kernel_core::contract::{DType, LaunchPlan};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernel_lab_families::ceil_prefill_status_mutant as mutant;
use kernels_sm120::row::block_quantize;
use model_deepseek41_contract::{
    admission::SUPPORTED_RANKS, geometry::GEOMETRIES, serving::DECODE_BATCH,
};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};
const ROWS: [u32; 3] = [1024, 2048, 8192];
const PATTERNS: [&str; 9] = [
    "finite",
    "wide",
    "zero",
    "denormal",
    "maximum",
    "nan_block",
    "inf_block",
    "boundaries",
    "sticky",
];
const OFFSET: usize = 3;
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
#[derive(Clone)]
struct Entry {
    inner: block_quantize::Entry,
    module: String,
}
impl Entry {
    fn symbol(&self) -> &str {
        &self.inner.symbol
    }
    fn width(&self) -> usize {
        self.inner.query.width as usize
    }
    fn f32(&self) -> bool {
        self.inner.query.input == DType::F32
    }
    fn plan(&self, rows: u32) -> LaunchPlan {
        self.inner.plan(rows).unwrap()
    }
}
fn pairs() -> Vec<[Entry; 2]> {
    let mut pairs = BTreeMap::new();
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            for warps in registration::WARPS {
                for (old, new) in registration::baseline_quantizers(g, ranks)
                    .unwrap()
                    .into_iter()
                    .zip(registration::quantizers(g, ranks, warps).unwrap())
                {
                    pairs.insert(
                        new.symbol.clone(),
                        [
                            Entry {
                                inner: old,
                                module: dense::cubin_name(g),
                            },
                            Entry {
                                inner: new,
                                module: registration::cubin_name(g),
                            },
                        ],
                    );
                }
            }
        }
    }
    pairs.into_values().collect()
}
fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_PREFILL_CEIL_CUBINS")
            .expect("frozen prefill ceil cubins required"),
    )
}
fn symbols() -> BTreeMap<String, Vec<String>> {
    let mut all = BTreeMap::<String, Vec<String>>::new();
    for pair in pairs() {
        for e in pair {
            all.entry(e.module).or_default().push(e.inner.symbol);
        }
    }
    all.insert(mutant::CUBIN.into(), vec![mutant::entry().symbol]);
    all.insert(
        "probe_sm120".into(),
        PROBE_SYMBOLS.iter().map(|s| s.to_string()).collect(),
    );
    for names in all.values_mut() {
        names.sort();
        names.dedup();
    }
    all
}
fn identities() -> BTreeMap<String, serde_json::Value> {
    let mut all = BTreeMap::new();
    for text in [
        include_str!("../../../bundle/kernels.lock.d/dense_quantize.lock"),
        include_str!("../../../bundle/kernels.lock.d/ceil_prefill.lock"),
        include_str!(
            "../../../../../crates/kernel-lab/kernels.lock.d/ceil_prefill_status_mutant.lock"
        ),
        include_str!("../../../../../crates/kernel-lab/kernels.lock"),
    ] {
        for record in serde_json::from_str::<Vec<serde_json::Value>>(text).unwrap() {
            all.insert(record["name"].as_str().unwrap().to_owned(), record);
        }
    }
    all
}
fn verify_modules() -> LabResult<()> {
    let locks = identities();
    for (name, symbols) in symbols() {
        let bytes =
            std::fs::read(artifacts().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        if locks[&name]["cubin_sha256"] != kernel_build::digest(&bytes) {
            return Err(format!("digest mismatch: {name}"));
        }
        for symbol in symbols {
            let name = format!("{symbol}\0");
            if !bytes.windows(name.len()).any(|v| v == name.as_bytes()) {
                return Err(format!("missing symbol {symbol}"));
            }
        }
    }
    Ok(())
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
    function.launch(stream,launch::config(entry.plan(rows))?,&[
        A::Buffer(input),A::Buffer(&out.codes),A::Buffer(&out.scales),A::Buffer(live),A::Buffer(&out.status),A::U64(OFFSET as u64),
    ]).map_err(|e|format!("{} rows={rows} warps={} block={:?} cluster=[1,1,1] dynamic_shared=0 attribute=0: {e}",entry.symbol(),entry.inner.threads()/32,entry.plan(rows).block))
}
fn inputs(entry: &Entry, rows: u32, pattern: &str, mut state: u64) -> Vec<u8> {
    let width = entry.width();
    let mut values = Vec::with_capacity(width * rows as usize * if entry.f32() { 4 } else { 2 });
    for i in 0..width * rows as usize {
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
            values.extend_from_slice(&bits.to_ne_bytes());
        } else {
            values.extend_from_slice(&((bits >> 16) as u16).to_ne_bytes());
        }
    }
    values
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
fn inventory_preserves_small_buckets_and_checks_all_launches() {
    use kernel_core::backend::{Backend, Need};
    use kernels_sm120::backend::Sm120;
    let pairs = pairs();
    assert_eq!(pairs.len(), 24);
    let locks = identities();
    for pair in &pairs {
        let mut decode = pair[0].inner.clone();
        decode.capacity = DECODE_BATCH;
        decode.symbol.push_str("_decode_control");
        let decode = decode.parallel().unwrap();
        for reverse in [false, true] {
            let entries = if reverse {
                [pair[1].inner.clone(), decode.clone()]
            } else {
                [decode.clone(), pair[1].inner.clone()]
            };
            let backend = entries
                .into_iter()
                .fold(Sm120::new(), Sm120::with_block_quantize);
            for rows in [1, DECODE_BATCH, DECODE_BATCH + 1, 1024, 2048, 8192] {
                let op = backend
                    .block_quantize(&decode.query, Need::rows(rows))
                    .unwrap();
                assert_eq!(
                    op.info().symbols[0].0,
                    if rows <= DECODE_BATCH {
                        &decode.symbol
                    } else {
                        pair[1].symbol()
                    }
                );
            }
        }
        for entry in pair {
            let resource = locks[&entry.module]["resources"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["symbol"] == entry.symbol())
                .unwrap();
            for rows in ROWS {
                let plan = entry.plan(rows);
                assert_eq!(plan.grid, [rows, 1, 1]);
                assert_eq!(plan.block, [entry.inner.threads(), 1, 1]);
                assert_eq!(plan.cluster, [1, 1, 1]);
                assert_eq!(plan.dynamic_shared_bytes, 0);
                assert!(plan.block[0] <= resource["max_threads"].as_u64().unwrap() as u32);
                assert!(resource["shared_static_bytes"].as_u64().unwrap() <= 49152);
                println!(
                    "P10_CONFIG module={} symbol={} rows={rows} block={:?} cluster={:?} dynamic_shared={} attribute=0 static_shared={} PASS",
                    entry.module,
                    entry.symbol(),
                    plan.block,
                    plan.cluster,
                    plan.dynamic_shared_bytes,
                    resource["shared_static_bytes"]
                );
            }
        }
    }
    let bad = mutant::entry();
    let plan = bad.plan(8192).unwrap();
    let resource = &locks[mutant::CUBIN]["resources"][0];
    assert_eq!(resource["symbol"], bad.symbol);
    assert_eq!(plan.block, [256, 1, 1]);
    assert_eq!(plan.cluster, [1, 1, 1]);
    assert_eq!(plan.dynamic_shared_bytes, 0);
    assert!(resource["max_threads"].as_u64().unwrap() >= 256);
    assert!(resource["shared_static_bytes"].as_u64().unwrap() <= 49152);
    println!(
        "P10_CONFIG module={} symbol={} rows=8192 block={:?} cluster={:?} dynamic_shared=0 attribute=0 static_shared={} PASS",
        mutant::CUBIN,
        bad.symbol,
        plan.block,
        plan.cluster,
        resource["shared_static_bytes"]
    );
    if std::env::var_os("CHOPINFER_PREFILL_CEIL_CUBINS").is_some() {
        verify_modules().unwrap();
    }
    if let Some(path) = std::env::var_os("CHOPINFER_PREFILL_CEIL_EXPORT") {
        let names = symbols();
        let selected: Vec<_> = names.keys().map(|name| locks[name].clone()).collect();
        let coverage = json!({"symbols":names,"records":selected,"pairs":pairs.len(),"rows":ROWS,"patterns":PATTERNS,
            "bitwise":pairs.len()*ROWS.len()*PATTERNS.len(),"timings":pairs.len()*ROWS.len(),"negative":1});
        std::fs::write(path, serde_json::to_vec_pretty(&coverage).unwrap()).unwrap();
    }
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_PREFILL_CEIL_CUBINS"]
fn prefill_ceil_bits_graphs_status_and_timing_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("prefill ceil comparison needs idle GPUs").unwrap();
        return;
    }
    verify_modules().unwrap();
    let probe = std::fs::read(artifacts().join("probe_sm120.cubin"))
        .unwrap()
        .leak();
    let cubins = Cubins::new(
        Box::leak(Box::new([("probe_sm120", &*probe)])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    );
    let result = kernel_harness::gpu_test_with(cubins, 1 << 30, compare_on_gpu).unwrap();
    assert!(result.is_some(), "required prefill ceil comparison skipped");
}

#[test]
fn nonfinite_fixtures_hit_the_final_group() {
    for pair in pairs() {
        let entry = &pair[0];
        for pattern in ["nan_block", "inf_block"] {
            let bytes = inputs(entry, 2, pattern, 0x91a2_03c4);
            let values: Vec<f32> = if entry.f32() {
                bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                    .collect()
            } else {
                bytes
                    .chunks_exact(2)
                    .map(|v| {
                        f32::from_bits(u32::from(u16::from_ne_bytes(v.try_into().unwrap())) << 16)
                    })
                    .collect()
            };
            for (i, value) in values.iter().enumerate() {
                if (entry.width() - 32..entry.width()).contains(&i) {
                    assert!(if pattern == "nan_block" {
                        value.is_nan()
                    } else {
                        value.is_infinite()
                    });
                } else {
                    assert!(value.is_finite(), "{pattern}: nonfinite value at {i}");
                }
            }
        }
    }
}

fn compare_on_gpu(s: &Session) -> LabResult<()> {
    let mut modules = BTreeMap::new();
    for name in symbols().keys().filter(|n| *n != "probe_sm120") {
        let image =
            std::fs::read(artifacts().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?;
        modules.insert(name.clone(), s.load_cubin(&image)?);
    }
    let mut bitwise = 0;
    let mut times = Vec::new();
    for pair in pairs() {
        let functions = pair
            .iter()
            .map(|e| {
                modules[&e.module]
                    .function(e.symbol())
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        let width = pair[0].width();
        for rows in ROWS {
            let input = s.allocate(width * rows as usize * if pair[0].f32() { 4 } else { 2 })?;
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
            for pattern in PATTERNS {
                upload(
                    &input,
                    &inputs(&pair[0], rows, pattern, 0x91a2_03c4),
                    &s.stream,
                )?;
                for live_rows in [rows, 0, rows - 1] {
                    upload(&live, &live_rows.to_ne_bytes(), &s.stream)?;
                    let mut eager = None;
                    for graph in [false, true] {
                        for out in &outputs {
                            out.poison(s, width, rows, pattern == "sticky")?;
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
                        if pattern == "finite" && live_rows > 0 {
                            assert!(new[0][..width].iter().any(|&v| v != 0 && v != 0xa5));
                            assert!(
                                new[2][OFFSET * 4..(OFFSET + live_rows as usize) * 4]
                                    .iter()
                                    .all(|&v| v == 0)
                            );
                        }
                        compare(&old,&new).map_err(|e|format!("{} rows={rows} live={live_rows} pattern={pattern} graph={graph}: {e}",pair[1].symbol()))?;
                        if let Some(ref reference) = eager {
                            compare(reference, &new)?;
                        } else {
                            eager = Some(old);
                        }
                        assert!(
                            new[0][live_rows as usize * width..]
                                .iter()
                                .all(|&v| v == 0xa5)
                        );
                        assert!(
                            new[1][live_rows as usize * width / 32..]
                                .iter()
                                .all(|&v| v == 0x5a)
                        );
                        for index in
                            (0..OFFSET).chain(OFFSET + rows as usize..OFFSET + rows as usize + 2)
                        {
                            assert_eq!(
                                &new[2][index * 4..index * 4 + 4],
                                &0x1357_2468u32.to_ne_bytes()
                            );
                        }
                    }
                }
                bitwise += 1;
                println!(
                    "P10_BITS {} rows={rows} pattern={pattern} PASS",
                    pair[1].symbol()
                );
            }
            upload(
                &input,
                &inputs(&pair[0], rows, "finite", 0x195a_dafe),
                &s.stream,
            )?;
            upload(&live, &rows.to_ne_bytes(), &s.stream)?;
            for out in &outputs {
                out.poison(s, width, rows, false)?;
            }
            for graph in &graphs {
                graph.launch(&s.stream).map_err(|e| e.to_string())?;
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            compare(
                &outputs[0].read(s, width, rows)?,
                &outputs[1].read(s, width, rows)?,
            )?;
            let label = format!("prefill-ceil:{}:rows={rows}", pair[1].symbol());
            let timing = kernel_harness::with_timing_label(&label, || {
                kernel_harness::time_ab(s, 2, 11, [EvictionMode::Read; 2], |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })
            })?;
            let record = json!({"symbol":pair[1].symbol(),"width":width,"f32":pair[0].f32(),"warps":pair[1].inner.prefill_warps,"rows":rows,
                    "old_ms":timing.arms[0].median_ms,"new_ms":timing.arms[1].median_ms,"ratio":timing.paired_ratio});
            println!("P10_TIMING {record}");
            times.push(record);
        }
    }
    // A real CUDA negative control removes whole-row admission after
    // the status OR. A NaN in the final group must still zero every group.
    let old = pairs()
        .into_iter()
        .find(|p| p[0].inner.query == mutant::entry().query)
        .unwrap()[0]
        .clone();
    let bad = Entry {
        inner: mutant::entry(),
        module: mutant::CUBIN.into(),
    };
    let rows = 8192;
    let width = old.width();
    let mut values = inputs(&old, rows, "nan_block", 0x195a_dafe);
    values[..4].copy_from_slice(&1.0_f32.to_ne_bytes());
    let input = s.allocate(values.len())?;
    upload(&input, &values, &s.stream)?;
    let live = s.allocate(4)?;
    upload(&live, &rows.to_ne_bytes(), &s.stream)?;
    let output = Output::new(s, width, rows)?;
    output.poison(s, width, rows, false)?;
    launch_entry(
        &old,
        &modules[&old.module]
            .function(old.symbol())
            .map_err(|e| e.to_string())?,
        rows,
        &input,
        &output,
        &live,
        &s.stream,
    )?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let expected = output.read(s, width, rows)?;
    assert!(expected[0][..width].iter().all(|&v| v == 0));
    output.poison(s, width, rows, false)?;
    launch_entry(
        &bad,
        &modules[&bad.module]
            .function(bad.symbol())
            .map_err(|e| e.to_string())?,
        rows,
        &input,
        &output,
        &live,
        &s.stream,
    )?;
    s.stream.synchronize().map_err(|e| e.to_string())?;
    let wrong = output.read(s, width, rows)?;
    assert!(wrong[0][..width].iter().any(|&v| v != 0));
    assert!(compare(&expected, &wrong).is_err());
    println!("P10_NEGATIVE rows=8192 PASS");
    assert_eq!(bitwise, pairs().len() * ROWS.len() * PATTERNS.len());
    assert_eq!(times.len(), pairs().len() * ROWS.len());
    s.receipt(
        "prefill-ceil",
        BTreeMap::from([
            ("bitwise".into(), json!(bitwise)),
            ("negative".into(), json!(1)),
            ("timings".into(), json!(times)),
        ]),
    )?;
    println!(
        "P10_COMPLETE bitwise={bitwise} timings={} negative=1",
        times.len()
    );
    Ok(())
}
