//! Isolated expert reductions on real checkpoint operands and routed rows.
//! Ownership intervals and FP8 scale-promotion order are explicit inputs.
#[path = "common/accumulation.rs"]
mod accumulation;

use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::oracle::{OracleOutput, compensated_sum_with_magnitude};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use runtime_cuda::{KernelArgument, LaunchConfig};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Component, Path};

const PROBE: &str = "probe_grouped_accumulation_sm120";
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileSpec {
    path: String,
    sha256: String,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct BankSpec {
    n: usize,
    k: usize,
    weight: FileSpec,
    scales: Option<FileSpec>,
    global: f32,
    live_n: Option<usize>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Schedule {
    id: String,
    projection: String,
    offsets: Vec<u32>,
    baseline_k: u32,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseSpec {
    id: String,
    origin: String,
    source_sha256: String,
    ranks: usize,
    rank: usize,
    expert: usize,
    rows: usize,
    input: FileSpec,
    gate: BankSpec,
    up: BankSpec,
    down: BankSpec,
    schedules: Vec<Schedule>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    cases: Vec<CaseSpec>,
}
struct Bank {
    n: usize,
    k: usize,
    weight: Vec<u16>,
    scales: Vec<f32>,
    global: f32,
    live_n: usize,
}
fn digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|x| x.is_ascii_hexdigit())
}
fn read(root: &Path, file: &FileSpec, count: usize, width: usize) -> LabResult<Vec<u8>> {
    if !Path::new(&file.path)
        .components()
        .all(|p| matches!(p, Component::Normal(_)))
        || !digest(&file.sha256)
    {
        return Err("invalid file path/digest".into());
    }
    let size = count.checked_mul(width).ok_or("file size overflow")?;
    let path = root.join(&file.path);
    if size > 128 << 20 || std::fs::metadata(&path).map_err(|e| e.to_string())?.len() != size as u64
    {
        return Err("bounded operand size mismatch".into());
    }
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&data)) != file.sha256.to_lowercase() {
        return Err("operand digest mismatch".into());
    }
    Ok(data)
}
fn bf16(data: &[u8]) -> LabResult<Vec<u16>> {
    let values = data
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().expect("two bytes")))
        .collect::<Vec<_>>();
    if values.iter().any(|&v| !bf16_to_f32(v).is_finite()) {
        return Err("nonfinite BF16 operand".into());
    }
    Ok(values)
}
fn bank(root: &Path, spec: &BankSpec) -> LabResult<Bank> {
    if spec.n == 0
        || spec.n % 16 != 0
        || spec.n > 4096
        || spec.k == 0
        || spec.k % 16 != 0
        || spec.k > 16384
        || !spec.global.is_finite()
        || spec.global <= 0.
        || (spec.scales.is_some() && spec.k % 128 != 0)
        || spec.live_n.is_some_and(|n| n == 0 || n > spec.n)
    {
        return Err("invalid bank geometry/global".into());
    }
    let weight = bf16(&read(root, &spec.weight, spec.n * spec.k, 2)?)?;
    let scales = spec
        .scales
        .as_ref()
        .map(|f| read(root, f, spec.n * (spec.k / 128), 4))
        .transpose()?
        .map(|data| {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                .collect()
        })
        .unwrap_or_default();
    let bank = Bank {
        n: spec.n,
        k: spec.k,
        weight,
        scales,
        global: spec.global,
        live_n: spec.live_n.unwrap_or(spec.n),
    };
    if bank.scales.iter().any(|x| !x.is_finite() || *x < 0.) {
        return Err("invalid post scale".into());
    }
    Ok(bank)
}
fn schedule(schedule: &Schedule, k: usize, post_scale: bool) -> LabResult<()> {
    if schedule.id.is_empty()
        || !["gate_up", "down", "matrix"].contains(&schedule.projection.as_str())
        || schedule.offsets.len() < 2
        || schedule.offsets.len() > 65
        || schedule.offsets[0] != 0
        || schedule.offsets.last().copied() != Some(k as u32)
        || schedule.offsets.windows(2).any(|w| w[0] >= w[1])
        || schedule
            .offsets
            .iter()
            .any(|x| x % if post_scale { 128 } else { 16 } != 0)
        || ![0, 16, 32, 64, 128, 256, 512, 1024].contains(&schedule.baseline_k)
        || (post_scale && ![32, 128].contains(&schedule.baseline_k))
    {
        return Err("invalid explicit K ownership/promotion schedule".into());
    }
    Ok(())
}
fn oracle(
    bank: &Bank,
    x: &[u16],
    m: usize,
    columns: &[usize],
    reductions: Option<&[usize]>,
) -> LabResult<OracleOutput> {
    let mut values = Vec::new();
    let mut sums = Vec::new();
    for row in 0..m {
        for &col in columns {
            let (total, abs) = compensated_sum_with_magnitude((0..bank.k).map(|k| {
                let scale = if bank.scales.is_empty() {
                    1.
                } else {
                    f64::from(bank.scales[col * (bank.k / 128) + k / 128])
                };
                f64::from(bf16_to_f32(bank.weight[col * bank.k + k]))
                    * f64::from(bf16_to_f32(x[row * bank.k + k]))
                    * scale
            }));
            values.push(total * f64::from(bank.global));
            sums.push(abs * f64::from(bank.global));
        }
    }
    let mut result = OracleOutput::new(values, sums, bank.k).map_err(|e| e.to_string())?;
    if let Some(lengths) = reductions {
        if lengths.len() != m || lengths.iter().any(|&k| k == 0 || k > bank.k) {
            return Err("invalid mathematical reduction lengths".into());
        }
        result.reduction_lengths = lengths
            .iter()
            .flat_map(|&k| std::iter::repeat_n(k, columns.len()))
            .collect();
    }
    Ok(result)
}
fn fma(bank: &Bank, x: &[u16], schedule: &Schedule, row: usize, col: usize) -> f32 {
    let mut result = 0.0_f32;
    for range in schedule.offsets.windows(2) {
        let (mut total, mut raw, mut count) = (0.0_f32, 0.0_f32, 0);
        for k in range[0] as usize..range[1] as usize {
            raw = bf16_to_f32(bank.weight[col * bank.k + k])
                .mul_add(bf16_to_f32(x[row * bank.k + k]), raw);
            count += 1;
            if k + 1 == range[1] as usize
                || (!bank.scales.is_empty() && (k + 1) % 128 == 0)
                || (schedule.baseline_k != 0 && count == schedule.baseline_k)
            {
                total = if bank.scales.is_empty() {
                    total + raw
                } else {
                    raw.mul_add(bank.scales[col * (bank.k / 128) + k / 128], total)
                };
                raw = 0.;
                count = 0;
            }
        }
        result += total;
    }
    result * bank.global
}
fn run(
    session: &Session,
    bank: &Bank,
    x: &[u16],
    m: usize,
    schedule: &Schedule,
    run_k: u32,
    fma: bool,
) -> LabResult<Vec<f32>> {
    let padded = m.div_ceil(8) * 8;
    let mut rows = x.to_vec();
    rows.resize(padded * bank.k, 0);
    let weight = session.upload_bf16(&bank.weight)?;
    let activation = session.upload_bf16(&rows)?;
    let scale = session.upload_f32(if bank.scales.is_empty() {
        &[1.]
    } else {
        &bank.scales
    })?;
    let offsets = session.allocate(schedule.offsets.len() * 4)?;
    let bytes = schedule
        .offsets
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    offsets
        .copy_from_bytes(&bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    let output = session.allocate(padded * bank.n * 4)?;
    let module = session.module(PROBE)?;
    let args = [
        KernelArgument::Buffer(&weight),
        KernelArgument::Buffer(&activation),
        KernelArgument::Buffer(&scale),
        KernelArgument::Buffer(&offsets),
        KernelArgument::Buffer(&output),
        KernelArgument::U32(bank.n as u32),
        KernelArgument::U32(padded as u32),
        KernelArgument::U32(bank.k as u32),
        KernelArgument::U32((schedule.offsets.len() - 1) as u32),
        KernelArgument::U32(u32::from(!bank.scales.is_empty())),
        KernelArgument::U32(run_k),
        KernelArgument::U32(bank.global.to_bits()),
    ];
    let grid = if fma {
        ((bank.n * padded).div_ceil(256) as u32, 1, 1)
    } else {
        (bank.n as u32 / 16, padded as u32 / 8, 1)
    };
    module
        .function(if fma {
            "chop_probe_grouped_fma_v1"
        } else {
            "chop_probe_grouped_chain_v1"
        })
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            launch::simple_config(
                grid,
                (if fma { 256 } else { 32 }, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                false,
            )?,
            &args,
        )
        .map_err(|e| e.to_string())?;
    let mut values = session.download_f32(&output, padded * bank.n)?;
    values.truncate(m * bank.n);
    if values.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite reduction output".into());
    }
    Ok(values)
}
fn frozen(session: &Session, gate: &[f32], up: &[f32]) -> LabResult<Vec<u16>> {
    let g = session.upload_f32(gate)?;
    let u = session.upload_f32(up)?;
    let out = session.allocate(gate.len() * 2)?;
    let args = [
        KernelArgument::Buffer(&g),
        KernelArgument::Buffer(&u),
        KernelArgument::Buffer(&out),
        KernelArgument::U32(gate.len() as u32),
    ];
    session
        .module(PROBE)?
        .function("chop_probe_grouped_swiglu_v1")
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            launch::simple_config(
                (gate.len().div_ceil(256) as u32, 1, 1),
                (256, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                false,
            )?,
            &args,
        )
        .map_err(|e| e.to_string())?;
    session.download_bf16(&out, gate.len())
}
fn measure(
    session: &Session,
    bank: &Bank,
    x: &[u16],
    m: usize,
    schedule: &Schedule,
    reductions: Option<&[usize]>,
) -> LabResult<Value> {
    let columns = accumulation::spread(bank.live_n, 128);
    let reference = oracle(bank, x, m, &columns, reductions)?;
    let mut summaries = BTreeMap::new();
    let mut lengths = vec![schedule.baseline_k, 16, 32, 64, 128, 256, 512, 1024];
    lengths.sort();
    lengths.dedup();
    for k in lengths {
        // Post-scale grain cannot cross K128. Record only distinct candidates.
        if !bank.scales.is_empty() && k > 128 {
            continue;
        }
        let values = run(session, bank, x, m, schedule, k, false)?;
        if bank.scales.is_empty() && bank.global == 1. && m == 32 && schedule.offsets.len() == 2 {
            let module = session.module(accumulation::PROBE)?;
            let p = accumulation::Problem::upload(session, &bank.weight, x, bank.n, m, bank.k)?;
            let expected = p.run(
                session,
                &module,
                accumulation::Scheme::Chain(if k == 0 { None } else { Some(k) }),
            )?;
            if values
                .iter()
                .zip(expected)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("expert probe differs from original accumulation probe".into());
            }
        }
        let sampled = (0..m)
            .flat_map(|row| {
                let values = &values;
                columns.iter().map(move |&c| values[row * bank.n + c])
            })
            .collect::<Vec<_>>();
        if sampled
            .iter()
            .zip(&reference.sum_abs_products)
            .any(|(&v, &s)| s == 0. && v != 0.)
        {
            return Err("zero-S mismatch".into());
        }
        let summary = accumulation::summarize(&sampled, &reference, DType::F32);
        summaries.insert(format!("run_{k}"), summary.json());
    }
    let values = run(session, bank, x, m, schedule, schedule.baseline_k, true)?;
    for index in accumulation::spread(values.len(), 128) {
        if values[index].to_bits()
            != fma(bank, x, schedule, index / bank.n, index % bank.n).to_bits()
        {
            return Err("grouped FMA emulator mismatch".into());
        }
    }
    let sampled = (0..m)
        .flat_map(|row| {
            let values = &values;
            columns.iter().map(move |&c| values[row * bank.n + c])
        })
        .collect::<Vec<_>>();
    summaries.insert(
        "fma".into(),
        accumulation::summarize(&sampled, &reference, DType::F32).json(),
    );
    Ok(
        json!({"schedule":schedule.id,"projection":schedule.projection,"shape":{"n":bank.n,"live_n":bank.live_n,"m":m,"k":bank.k},"reduction_lengths":reductions,"columns":columns,
        "offsets":schedule.offsets,"baseline_k":schedule.baseline_k,"post_scale":!bank.scales.is_empty(),"global":bank.global,
        "small_tail_sample":m*columns.len()<100_000,
        "original_probe_bitwise_check":bank.scales.is_empty() && bank.global==1.0 && m==32 && schedule.offsets.len()==2,
        "fma_checked":values.len().min(128),"schemes":summaries}),
    )
}
fn has_violation(v: &Value) -> bool {
    match v {
        Value::Object(m) => m
            .iter()
            .any(|(k, v)| (k == "violations" && v.as_u64().unwrap_or(0) > 0) || has_violation(v)),
        Value::Array(a) => a.iter().any(has_violation),
        _ => false,
    }
}
#[test]
fn grouped_real() {
    gpu_test(4<<30,|session| {
        let file=std::env::var_os("CHOPINFER_GROUPED_MANIFEST").ok_or("set CHOPINFER_GROUPED_MANIFEST")?;let file=Path::new(&file);
        let manifest:Manifest=serde_json::from_slice(&std::fs::read(file).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
        if manifest.schema!="chopinfer.grouped-accumulation.v1" || manifest.cases.is_empty() || manifest.cases.len()>16 {return Err("invalid or empty grouped manifest".into());}
        let select=std::env::var("CHOPINFER_GROUPED_CASE").ok();let mut results=Vec::new();
        for c in manifest.cases.into_iter().filter(|c|select.as_ref().is_none_or(|x|x==&c.id)) {
            if c.rows==0 || c.rows>128 || ![2,4].contains(&c.ranks) || c.rank>=c.ranks || !digest(&c.source_sha256) || c.origin.is_empty() {return Err("invalid case provenance/rows".into());}
            let root=file.parent().ok_or("manifest parent")?;
            let gate=bank(root,&c.gate)?;let up=bank(root,&c.up)?;let down=bank(root,&c.down)?;
            if gate.n!=up.n || gate.k!=up.k || down.k!=gate.n || gate.n%128!=0 {return Err("expert bank geometry differs".into());}
            let x=bf16(&read(root,&c.input,c.rows*gate.k,2)?)?;
            let freeze=c.schedules.iter().find(|s|s.projection=="gate_up").ok_or("missing freeze schedule")?;
            for s in &c.schedules {if !["gate_up","down"].contains(&s.projection.as_str()) { return Err("invalid expert projection".into()); } schedule(s,if s.projection=="down"{down.k}else{gate.k},if s.projection=="down"{!down.scales.is_empty()}else{!gate.scales.is_empty()})?;}
            let g=run(session,&gate,&x,c.rows,freeze,freeze.baseline_k,false)?;let u=run(session,&up,&x,c.rows,freeze,freeze.baseline_k,false)?;
            let intermediate=frozen(session,&g,&u)?;
            let hash=format!("{:x}",Sha256::digest(intermediate.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>()));
            let mut rows=Vec::new();
            for s in &c.schedules {
                if s.projection=="gate_up" {
                    let mut gate_row=measure(session,&gate,&x,c.rows,s,None)?;
                    gate_row["bank"]=json!("gate");rows.push(gate_row);
                    let mut up_row=measure(session,&up,&x,c.rows,s,None)?;
                    up_row["bank"]=json!("up");rows.push(up_row);
                }
                else {
                    let mut down_row=measure(session,&down,&intermediate,c.rows,s,None)?;
                    down_row["bank"]=json!("down");rows.push(down_row);
                }
            }
            results.push(json!({"id":c.id,"origin":c.origin,"ranks":c.ranks,"rank":c.rank,"expert":c.expert,
                "source_sha256":c.source_sha256,"input_sha256":c.input.sha256,
                "bank_digests":{"gate":c.gate.weight.sha256,"up":c.up.weight.sha256,"down":c.down.weight.sha256,
                    "gate_scales":c.gate.scales.as_ref().map(|f|&f.sha256),"up_scales":c.up.scales.as_ref().map(|f|&f.sha256),
                    "down_scales":c.down.scales.as_ref().map(|f|&f.sha256)},"frozen_down_sha256":hash,"freeze_schedule":freeze.id,
                "scope":"isolated expert reductions; frozen unclamped BF16 intermediate; no routing/combine qualification","rows":rows}));
        }
        if results.is_empty(){return Err("empty grouped selection".into());}
        let result=json!(results);let path=session.receipt("grouped_accumulation",BTreeMap::from([
            ("cases".into(),result.clone()),("source_pin".into(),json!(std::env::var("CHOPINFER_ACCUMULATION_SOURCE").ok())),
            ("lab_lock_sha256".into(),json!(format!("{:x}",Sha256::digest(include_bytes!("../kernels.lock")))))]))?;
        println!("receipt {}",path.display());if has_violation(&result){return Err("grouped contract failures recorded".into());}Ok(())
    }).expect("real grouped accumulation");
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixCase {
    id: String,
    family: String,
    origin: String,
    source_sha256: String,
    rows: usize,
    bank: BankSpec,
    input: FileSpec,
    reduction_lengths: Vec<usize>,
    schedules: Vec<Schedule>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixManifest {
    schema: String,
    cases: Vec<MatrixCase>,
}

#[test]
fn matrix_real() {
    gpu_test(2 << 30, |session| {
        let path = std::env::var_os("CHOPINFER_MATRIX_MANIFEST").ok_or("set CHOPINFER_MATRIX_MANIFEST")?;
        let path = Path::new(&path);
        let manifest: MatrixManifest = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if manifest.schema != "chopinfer.matrix-accumulation.v1" || manifest.cases.is_empty() || manifest.cases.len() > 16 { return Err("invalid matrix manifest".into()); }
        let mut results = Vec::new();
        for case in manifest.cases {
            if case.rows == 0 || case.rows > 128 || !digest(&case.source_sha256) || case.family.is_empty() || case.origin.is_empty() { return Err("invalid matrix provenance".into()); }
            let root = path.parent().ok_or("manifest parent")?;
            let b = bank(root, &case.bank)?;
            let x = bf16(&read(root, &case.input, case.rows * b.k, 2)?)?;
            let mut rows = Vec::new();
            if case.schedules.is_empty() { return Err("empty matrix schedules".into()); }
            for s in &case.schedules {
                if s.projection != "matrix" { return Err("invalid matrix projection".into()); }
                schedule(s, b.k, !b.scales.is_empty())?;
                rows.push(measure(session, &b, &x, case.rows, s, Some(&case.reduction_lengths))?);
            }
            results.push(json!({"id":case.id,"family":case.family,"origin":case.origin,"source_sha256":case.source_sha256,
                "weight_sha256":case.bank.weight.sha256,"input_sha256":case.input.sha256,"rows":rows,
                "scope":"expanded decoded products with mathematical K; frozen stable-maximum PV, not full attention"}));
        }
        let value = json!(results);
        let path = session.receipt("matrix_accumulation", BTreeMap::from([
            ("cases".into(), value.clone()), ("source_pin".into(), json!(std::env::var("CHOPINFER_ACCUMULATION_SOURCE").ok())),
            ("lab_lock_sha256".into(), json!(format!("{:x}", Sha256::digest(include_bytes!("../kernels.lock")))))]))?;
        println!("receipt {}", path.display());
        if has_violation(&value) { return Err("matrix contract violations recorded".into()); }
        Ok(())
    }).expect("matrix accumulation");
}

fn pv_orientation_control(session: &Session, b: &Bank, x: &[u16], s: &Schedule) -> LabResult<()> {
    let expected = run(session, b, x, 32, s, 0, false)?;
    let weight = session.upload_bf16(&b.weight)?;
    let activation = session.upload_f32(&x.iter().map(|&v| bf16_to_f32(v)).collect::<Vec<_>>())?;
    let correction = session.upload_f32(&vec![1.0; 32 * (b.k / 32)])?;
    let output = session.allocate(32 * b.n * 4)?;
    let args = [
        KernelArgument::Buffer(&weight),
        KernelArgument::Buffer(&activation),
        KernelArgument::Buffer(&correction),
        KernelArgument::Buffer(&output),
        KernelArgument::U32(b.n as u32),
        KernelArgument::U32(32),
        KernelArgument::U32(b.k as u32),
        KernelArgument::U32(1),
        KernelArgument::U32(0),
    ];
    session
        .module("probe_non_dense_accumulation_sm120")?
        .function("chop_probe_pv_accumulation_v1")
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            launch::simple_config(
                (b.n as u32 / 8, 2, 1),
                (32, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                false,
            )?,
            &args,
        )
        .map_err(|e| e.to_string())?;
    let actual = session.download_f32(&output, 32 * b.n)?;
    if actual
        .iter()
        .zip(expected)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("PV operand orientation changes accumulator bits".into());
    }
    Ok(())
}

fn tma_chunk_control(session: &Session, b: &Bank, x: &[u16], s: &Schedule) -> LabResult<()> {
    let expected = run(session, b, &x[..8 * b.k], 8, s, 128, false)?;
    let weight = session.upload_bf16(&b.weight)?;
    let activation = session.upload_bf16(&x[..8 * b.k])?;
    let output = session.allocate(8 * b.n * 4)?;
    let args = [
        KernelArgument::Buffer(&weight),
        KernelArgument::Buffer(&activation),
        KernelArgument::Buffer(&output),
        KernelArgument::U32(b.k as u32),
    ];
    session
        .module(PROBE)?
        .function("chop_probe_tma_chunk_control_v1")
        .map_err(|e| e.to_string())?
        .launch(
            &session.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &args,
        )
        .map_err(|e| e.to_string())?;
    let values = session.download_f32(&output, 8 * b.n)?;
    if values
        .iter()
        .zip(expected)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err("TMA consumer differs from K128 probe".into());
    }
    Ok(())
}

fn synthetic(post: bool) -> (Bank, Vec<u16>, Schedule) {
    let (n, k, m) = (16, 256, 32);
    let weights = (0..n * k)
        .map(|i| {
            f32_to_bf16(
                (1. + ((i / k * 3 + i % 8) % 8) as f32 / 8.) * if i % 3 == 0 { -1. } else { 1. },
            )
        })
        .collect();
    let bank = Bank {
        n,
        k,
        weight: weights,
        scales: if post {
            (0..n * 2).map(|i| 2_f32.powi((i % 7) as i32 - 3)).collect()
        } else {
            vec![]
        },
        global: 1.,
        live_n: n,
    };
    let x = (0..m * k)
        .map(|i| f32_to_bf16(0.5 + (i % 31) as f32 / 64.))
        .collect();
    (
        bank,
        x,
        Schedule {
            id: "synthetic".into(),
            projection: "gate_up".into(),
            offsets: vec![0, k as u32],
            baseline_k: if post { 128 } else { 0 },
        },
    )
}
#[test]
fn grouped_adversarial() {
    gpu_test(2 << 30, |session| {
        let mut results = Vec::new();
        for post in [false, true] {
            let (b, x, s) = synthetic(post);
            if !post {
                pv_orientation_control(session, &b, &x, &s)?;
                tma_chunk_control(session, &b, &x, &s)?;
            }
            results.push(measure(session, &b, &x, 32, &s, None)?);
        }
        let v = json!(results);
        let p = session.receipt(
            "grouped_accumulation",
            BTreeMap::from([
                ("rows".into(), v.clone()),
                ("synthetic".into(), json!(true)),
                ("pv_operand_orientation_bitwise_check".into(), json!(true)),
                ("tma_chunk_bitwise_check".into(), json!(true)),
                (
                    "source_pin".into(),
                    json!(std::env::var("CHOPINFER_ACCUMULATION_SOURCE").ok()),
                ),
                (
                    "lab_lock_sha256".into(),
                    json!(format!(
                        "{:x}",
                        Sha256::digest(include_bytes!("../kernels.lock"))
                    )),
                ),
            ]),
        )?;
        println!("receipt {}", p.display());
        if has_violation(&v) {
            return Err("synthetic grouped contract failures".into());
        }
        Ok(())
    })
    .expect("grouped scale-order controls");
}
#[test]
fn post_scale_oracle_does_not_round_weights_to_bf16() {
    let (mut b, mut x, s) = synthetic(true);
    b.weight.fill(f32_to_bf16(1.));
    x.fill(f32_to_bf16(1.));
    b.scales.fill(1.0001);
    let o = oracle(&b, &x, 32, &[0], None).unwrap();
    assert_eq!(o.values[0], 256. * f64::from(1.0001_f32));
    assert_ne!(o.values[0], 256.);
    assert!(fma(&b, &x, &s, 0, 0) > 256.);
}
#[test]
fn reject_malformed_ownership_and_scale_schedule() {
    let mut s = Schedule {
        id: "bad".into(),
        projection: "gate_up".into(),
        offsets: vec![0, 64, 256],
        baseline_k: 128,
    };
    assert!(schedule(&s, 256, true).is_err());
    assert!(schedule(&s, 256, false).is_ok());
    s.offsets = vec![0, 256];
    s.baseline_k = 0;
    assert!(schedule(&s, 256, true).is_err());
    s.offsets = vec![0, 128, 128, 256];
    assert!(schedule(&s, 256, false).is_err());
}

#[test]
fn expanded_products_keep_original_visible_k() {
    let (b, x, _) = synthetic(false);
    let lengths = (0..32).map(|i| i + 1).collect::<Vec<_>>();
    let oracle = oracle(&b, &x, 32, &[0, 1], Some(&lengths)).unwrap();
    assert_eq!(&oracle.reduction_lengths[..4], &[1, 1, 2, 2]);
    assert_eq!(oracle.reduction_lengths.len(), 64);
    assert!(self::oracle(&b, &x, 32, &[0], Some(&[0; 32])).is_err());
}
