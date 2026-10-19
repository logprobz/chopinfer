//! Bitwise old-chain comparisons for the decode norm epilogues.
use bundle_deepseek41::sm120::{
    ceil_parallel, fold_cubin_for_symbol, norm_quantize as registration,
};
use kernel_core::contract::DType;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::row::{
    block_quantize,
    rmsnorm::{RmsNorm, RmsNormTarget},
    rmsnorm_quantize::Entry,
    rounding,
};
use model_deepseek41_contract::{geometry::GEOMETRIES, serving::DECODE_BATCH};
use runtime_cuda::{DeviceBuffer, Function, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const ROWS: [u32; 6] = [1, 2, 4, 8, 16, DECODE_BATCH];
const PATTERNS: [&str; 14] = [
    "finite",
    "zero",
    "denormal",
    "wide",
    "maximum",
    "nan_first",
    "nan_last",
    "inf_first",
    "inf_last",
    "weight_nan",
    "weight_inf",
    "sticky",
    "ties",
    "cancellation",
];
const OFFSET: usize = 3;
pub(super) const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];

struct Case {
    fused: Entry,
    norm: RmsNorm,
    norm_symbol: String,
    quant: block_quantize::Entry,
    widen: Option<String>,
}
impl Case {
    fn width(&self) -> usize {
        self.fused.query.norm.width as usize
    }
    fn f32(&self) -> bool {
        self.fused.query.norm.input == DType::F32
    }
    fn label(&self) -> String {
        format!("{}:widen={}", self.fused.symbol, self.widen.is_some())
    }
    fn symbols(&self) -> Vec<&str> {
        let mut out = vec![
            self.fused.symbol.as_str(),
            self.norm_symbol.as_str(),
            self.quant.symbol.as_str(),
        ];
        out.extend(self.widen.as_deref());
        out
    }
}
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for g in GEOMETRIES {
        for entry in registration::entries(g).unwrap() {
            let f32 = entry.query.norm.input == DType::F32;
            for widen in if f32 {
                &[false][..]
            } else {
                &[false, true][..]
            } {
                let norm_symbol = if f32 {
                    format!("chop_{}_q_norm_f32_v1", g.kernel_tag)
                } else {
                    format!("chop_row_rmsnorm_{}_w{}_v1", g.kernel_tag, g.shape.hidden)
                };
                let quant = ceil_parallel::quantizers(g, 1)
                    .unwrap()
                    .into_iter()
                    .find(|q| q.query == entry.query.quantizer())
                    .unwrap();
                cases.push(Case {
                    norm: RmsNorm::new(entry.query.norm, DECODE_BATCH, RmsNormTarget::SM120F)
                        .unwrap(),
                    norm_symbol,
                    quant,
                    fused: entry.clone(),
                    widen: widen.then(|| {
                        format!(
                            "chop_row_rounding_widen_{}_w{}_v1",
                            g.kernel_tag, g.shape.hidden
                        )
                    }),
                });
            }
        }
    }
    cases
}
pub(super) fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_NORM_ARTIFACTS").expect("frozen norm artifacts required"),
    )
}
fn modules() -> BTreeMap<String, (Vec<u8>, Vec<String>)> {
    let mut modules = BTreeMap::<String, (Vec<u8>, Vec<String>)>::new();
    for case in cases() {
        for symbol in case.symbols() {
            let (name, bytes) =
                fold_cubin_for_symbol(symbol).expect("served symbol must be embedded");
            let module = modules
                .entry(name.into())
                .or_insert_with(|| (bytes.to_vec(), Vec::new()));
            if !module.1.iter().any(|s| s == symbol) {
                module.1.push(symbol.into());
            }
        }
    }
    for symbol in super::window_fused::symbols() {
        let (name, bytes) = fold_cubin_for_symbol(&symbol).expect("window symbol must be embedded");
        let module = modules
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), Vec::new()));
        if !module.1.contains(&symbol) {
            module.1.push(symbol);
        }
    }
    modules
}

pub(super) fn upload(buffer: &DeviceBuffer, bytes: &[u8], stream: &Stream) -> LabResult<()> {
    buffer
        .copy_from_bytes(bytes, stream)
        .map_err(|e| e.to_string())
}
pub(super) fn download(buffer: &DeviceBuffer, len: usize, stream: &Stream) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; len];
    buffer
        .copy_to_bytes(&mut bytes, stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}
struct Output {
    buffers: [DeviceBuffer; 5],
    lengths: [usize; 5],
}
impl Output {
    fn new(s: &Session, c: &Case, rows: u32) -> LabResult<Self> {
        let n = c.width() * rows as usize;
        let lengths = [
            n * if c.f32() { 4 } else { 2 },
            n * 4,
            n,
            n / 32,
            (rows as usize + OFFSET + 2) * 4,
        ];
        Ok(Self {
            buffers: [
                s.allocate(lengths[0])?,
                s.allocate(lengths[1])?,
                s.allocate(lengths[2])?,
                s.allocate(lengths[3])?,
                s.allocate(lengths[4])?,
            ],
            lengths,
        })
    }
    fn poison(&self, s: &Session, rows: u32, sticky: bool) -> LabResult<()> {
        for i in 0..4 {
            upload(&self.buffers[i], &vec![0xa5; self.lengths[i]], &s.stream)?;
        }
        let mut flags = vec![0x1357_2468u32; rows as usize + OFFSET + 2];
        flags[OFFSET..OFFSET + rows as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.buffers[4],
            &flags
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            &s.stream,
        )
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .zip(self.lengths)
            .map(|(b, n)| download(b, n, &s.stream))
            .collect()
    }
}
fn compare(expected: &[Vec<u8>], actual: &[Vec<u8>]) -> LabResult<()> {
    for (role, (a, b)) in ["normed", "widened", "codes", "scales", "status"]
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

struct Inputs {
    x: DeviceBuffer,
    weight: DeviceBuffer,
    live: DeviceBuffer,
}
#[allow(clippy::too_many_arguments)]
fn run(
    c: &Case,
    functions: &[Function],
    fused: bool,
    rows: u32,
    input: &Inputs,
    out: &Output,
    stream: &Stream,
    eps: u32,
) -> LabResult<()> {
    use runtime_cuda::KernelArgument as A;
    let [normed, widened, codes, scales, status] = &out.buffers;
    if fused {
        let args = [
            A::Buffer(&input.x),
            A::Buffer(&input.weight),
            A::Buffer(normed),
            if c.widen.is_some() {
                A::Buffer(widened)
            } else {
                A::NullBuffer
            },
            A::Buffer(codes),
            A::Buffer(scales),
            A::Buffer(&input.live),
            A::Buffer(status),
            A::U64(OFFSET as u64),
            A::U32(eps),
        ];
        functions[0]
            .launch(stream, launch::config(c.fused.plan(rows).unwrap())?, &args)
            .map_err(|e| e.to_string())
    } else {
        let mut args = vec![
            A::Buffer(&input.x),
            A::Buffer(&input.weight),
            A::NullBuffer,
            A::NullBuffer,
            A::Buffer(normed),
            A::Buffer(&input.live),
            A::U32(eps),
        ];
        if c.f32() {
            args.extend([A::Buffer(status), A::U64(OFFSET as u64)]);
        }
        functions[1]
            .launch(stream, launch::config(c.norm.plan(rows).unwrap())?, &args)
            .map_err(|e| e.to_string())?;
        if c.widen.is_some() {
            functions[3]
                .launch(
                    stream,
                    launch::config(rounding::plan(rows).unwrap())?,
                    &[
                        A::Buffer(normed),
                        A::Buffer(widened),
                        A::Buffer(&input.live),
                    ],
                )
                .map_err(|e| e.to_string())?;
        }
        functions[2]
            .launch(
                stream,
                launch::config(c.quant.plan(rows).unwrap())?,
                &[
                    A::Buffer(normed),
                    A::Buffer(codes),
                    A::Buffer(scales),
                    A::Buffer(&input.live),
                    A::Buffer(status),
                    A::U64(OFFSET as u64),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

fn inputs(c: &Case, rows: u32, pattern: &str, mut state: u64) -> Vec<u8> {
    (0..c.width() * rows as usize)
        .flat_map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = (state as u32) & 0x8000_0000;
            let random =
                sign | (((state >> 32) as u32 % 20 + 117) << 23) | (state as u32 & 0x7f_ffff);
            let bits = match pattern {
                "zero" => sign,
                "denormal" => sign | ((state as u32 & 0x7f_0000).max(0x1_0000)),
                "wide" => {
                    sign | (((state >> 32) as u32 % 254 + 1) << 23) | (state as u32 & 0x7f_ffff)
                }
                "maximum" => sign | 0x7f7f_0000,
                "nan_first" if i == 0 => 0x7fc1_0000,
                "nan_last" if i == c.width() - 1 => 0xff81_0000,
                "inf_first" if i == 0 => 0x7f80_0000,
                "inf_last" if i == c.width() - 1 => 0xff80_0000,
                "ties" => sign | 0x3f80_8000 | (((i as u32) % 7) << 16),
                "cancellation" => sign | if i % 8 == 0 { 0x4700_0000 } else { 0x3780_0000 },
                _ => random,
            };
            if c.f32() {
                bits.to_ne_bytes().to_vec()
            } else {
                ((bits >> 16) as u16).to_ne_bytes().to_vec()
            }
        })
        .collect()
}
fn weights(c: &Case, pattern: &str) -> Vec<u8> {
    (0..c.width())
        .flat_map(|i| {
            let bits: u16 = match pattern {
                "weight_nan" if i == c.width() - 1 => 0x7fc1,
                "weight_inf" if i == 0 => 0xff80,
                _ => 0x3f00 + (i as u16 % 128),
            };
            bits.to_ne_bytes()
        })
        .collect()
}

#[test]
fn fixture_inventory_and_output_gate_detect_negative_controls() {
    let cases = cases();
    assert_eq!(cases.len(), GEOMETRIES.len() * 3);
    for c in &cases {
        for symbol in c.symbols() {
            assert!(fold_cubin_for_symbol(symbol).is_some(), "{symbol}");
        }
        for pattern in PATTERNS {
            assert_eq!(
                inputs(c, 2, pattern, 1).len(),
                c.width() * 2 * if c.f32() { 4 } else { 2 }
            );
        }
        let bytes = inputs(c, 2, "nan_last", 1);
        let at = (c.width() - 1) * if c.f32() { 4 } else { 2 };
        if c.f32() {
            assert!(f32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap()).is_nan());
        } else {
            assert_eq!(&bytes[at..at + 2], &0xff81u16.to_ne_bytes());
        }
    }
    let expected = vec![vec![0; 8]; 5];
    for role in 0..5 {
        let mut bad = expected.clone();
        bad[role][0] = 1;
        assert!(compare(&expected, &bad).is_err());
    }
}

#[test]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    if kernel_harness::selected_ordinals().unwrap().is_none() {
        kernel_harness::skip("norm fusion comparison needs an idle GPU").unwrap();
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
    let result=kernel_harness::gpu_test_with(cubins,1<<30,|s| {
        let mut bitwise=0; let mut negative=0; let mut timings=Vec::new();
        for c in cases() {
            let modules=c.symbols().iter().map(|symbol| {
                let (name,_)=fold_cubin_for_symbol(symbol).unwrap();
                s.load_cubin(&std::fs::read(artifacts().join(format!("{name}.cubin"))).map_err(|e|e.to_string())?)
            }).collect::<LabResult<Vec<_>>>()?;
            let functions=modules.iter().zip(c.symbols()).map(|(m,s)|m.function(s).map_err(|e|e.to_string())).collect::<LabResult<Vec<_>>>()?;
            let eps=c.fused.query.norm.epsilon_bits;
            for rows in ROWS {
                let input=Inputs {x:s.allocate(c.width()*rows as usize*if c.f32(){4}else{2})?,weight:s.allocate(c.width()*2)?,live:s.allocate(4)?};
                let outputs=[Output::new(s,&c,rows)?,Output::new(s,&c,rows)?];
                let graphs=(0..2).map(|arm|s.capture(|stream|run(&c,&functions,arm==1,rows,&input,&outputs[arm],stream,eps))).collect::<LabResult<Vec<_>>>()?;
                for pattern in PATTERNS {
                    upload(&input.x,&inputs(&c,rows,pattern,0x91a2_03c4),&s.stream)?;
                    upload(&input.weight,&weights(&c,pattern),&s.stream)?;
                    for live_rows in [rows,0,rows.saturating_sub(1),rows+7,rows] {
                        let active=live_rows.min(rows) as usize;
                        upload(&input.live,&live_rows.to_ne_bytes(),&s.stream)?;
                        let mut eager: Option<Vec<Vec<u8>>> = None;
                        for graph in [false,true,true] {
                            for out in &outputs {out.poison(s,rows,pattern=="sticky")?;}
                            for arm in 0..2 {
                                if graph {graphs[arm].launch(&s.stream).map_err(|e|e.to_string())?;}
                                else {run(&c,&functions,arm==1,rows,&input,&outputs[arm],&s.stream,eps)?;}
                            }
                            s.stream.synchronize().map_err(|e|e.to_string())?;
                            let old=outputs[0].read(s)?;let new=outputs[1].read(s)?;
                            compare(&old,&new).map_err(|e|format!("{} rows={rows} live={live_rows} {pattern} graph={graph}: {e}",c.label()))?;
                            if let Some(ref previous)=eager {compare(previous,&new)?;} else {eager=Some(old);}
                            for (role, bytes) in new.iter().enumerate().take(4) {
                                let start=if role==1 && c.widen.is_none() {0} else {outputs[1].lengths[role]/rows as usize*active};
                                assert!(bytes[start..].iter().all(|b|*b==0xa5),"inactive role {role}");
                            }
                            for index in (0..OFFSET).chain(OFFSET+rows as usize..OFFSET+rows as usize+2) {assert_eq!(&new[4][index*4..index*4+4],&0x1357_2468u32.to_ne_bytes());}
                            if pattern=="finite" && active>0 {assert!(new[2][..c.width()].iter().any(|b|*b!=0));}
                        }
                    }
                    bitwise+=1;eprintln!("NORM_BITWISE {} rows={rows} pattern={pattern} PASS",c.label());
                }
                upload(&input.x,&inputs(&c,rows,"finite",0x195a_dafe),&s.stream)?;
                upload(&input.weight,&weights(&c,"finite"),&s.stream)?;
                upload(&input.live,&rows.to_ne_bytes(),&s.stream)?;
                for output in &outputs {output.poison(s,rows,false)?;}
                for graph in &graphs {graph.launch(&s.stream).map_err(|e|e.to_string())?;}
                s.stream.synchronize().map_err(|e|e.to_string())?;
                let expected=outputs[0].read(s)?;compare(&expected,&outputs[1].read(s)?)?;
                for role in 0..5 {
                    let mut bad=expected[role].clone();bad[if role==4 {OFFSET*4} else {0}]^=1;
                    upload(&outputs[1].buffers[role],&bad,&s.stream)?;
                    assert!(compare(&expected,&outputs[1].read(s)?).is_err());
                    upload(&outputs[1].buffers[role],&expected[role],&s.stream)?;negative+=1;
                }
                // Changing epsilon is an arithmetic negative control, not just a byte mutation.
                run(&c,&functions,true,rows,&input,&outputs[1],&s.stream,1e6_f32.to_bits())?;
                s.stream.synchronize().map_err(|e|e.to_string())?;
                assert!(compare(&expected,&outputs[1].read(s)?).is_err());
                graphs[1].launch(&s.stream).map_err(|e|e.to_string())?;
                let label=format!("norm:{}:rows={rows}",c.label());
                let timing=kernel_harness::with_timing_label(&label,||kernel_harness::time_ab(s,3,15,[EvictionMode::Read;2],|arm,stream|graphs[arm].launch(stream).map_err(|e|e.to_string())))?;
                eprintln!("NORM_TIMING {} rows={rows} old_us={} new_us={} ratio={}",c.label(),timing.arms[0].median_ms*1000.,timing.arms[1].median_ms*1000.,timing.paired_ratio);
                timings.push(json!({"label":c.label(),"rows":rows,"old_us":timing.arms[0].median_ms*1000.,"new_us":timing.arms[1].median_ms*1000.,"ratio":timing.paired_ratio}));
            }
        }
        assert_eq!(bitwise,cases().len()*ROWS.len()*PATTERNS.len());
        assert_eq!(negative,cases().len()*ROWS.len()*5);
        s.receipt("norm-quantize",BTreeMap::from([("bitwise_cases".into(),json!(bitwise)),("negative_controls".into(),json!(negative)),("timings".into(),json!(timings))]))?;
        eprintln!("NORM_COMPLETE bitwise={bitwise} negative={negative} arithmetic_negative={} timings={}",timings.len(),timings.len());
        Ok(())
    }).unwrap();
    assert!(result.is_some());
}
pub(super) fn verify_modules() -> LabResult<()> {
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

/// Explicit artifact staging/checking, never part of an ordinary test run.
#[test]
#[ignore = "requires an external frozen artifact directory"]
fn frozen_modules_are_complete() {
    if std::env::var_os("CHOPINFER_NORM_EXPORT").is_some() {
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
            std::env::var_os("CHOPINFER_NORM_TIMING_PROBE")
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
        let coverage = json!({"entries": cases().iter().map(|c| json!({"label":c.label(),"new":c.fused.symbol,"width":c.width()})).collect::<Vec<_>>(), "rows": ROWS, "patterns": PATTERNS,
            "bitwise_cases":cases().len()*ROWS.len()*PATTERNS.len(), "timing_cases":cases().len()*ROWS.len(), "negative_controls":cases().len()*ROWS.len()*5,"window":super::window_fused::coverage()});
        std::fs::write(
            dir.join("coverage.json"),
            serde_json::to_vec_pretty(&coverage).unwrap(),
        )
        .unwrap();
    }
    verify_modules().unwrap();
    eprintln!(
        "NORM_MODULES {} PASS (including probe_sm120)",
        modules().len() + 1
    );
}
